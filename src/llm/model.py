"""The full model: embeddings, a stack of blocks, a norm, and a projection to vocabulary size.

Short file. Everything interesting happened in notes 2 and 3; this is wiring.

See notes/04_the_model.md.
"""

from __future__ import annotations

import math
from typing import overload

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from llm.block import TransformerBlock
from llm.config import ModelConfig
from llm.norms import make_norm
from llm.rope import RopeCache


class GPTModel(nn.Module):
    """A decoder-only transformer language model.

    Two embedding tables, N identical blocks, a final norm, and an output head. The head is tied
    to the token embedding by default: "the vector that means cat" and "the direction that
    predicts cat" should be the same vector.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        # Exactly one of these is active. RoPE has no embedding table -- it is applied inside
        # attention, to q and k only -- so the cos/sin cache stands in for pos_emb and holds no
        # parameters at all.
        self.pos_emb = (
            nn.Embedding(cfg.context_length, cfg.emb_dim) if cfg.pos == "learned" else None
        )
        self.rope = (
            RopeCache(cfg.head_dim, cfg.context_length, cfg.rope_base)
            if cfg.pos == "rope"
            else None
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(cfg, i) for i in range(cfg.n_layers))

        # Pre-norm blocks normalize on the way in, so the last block's output is unnormalized.
        # Omitting this trains fine and underperforms silently -- the kind of bug the GPT-2
        # oracle catches and unit tests do not.
        self.final_norm = make_norm(cfg)

        self.head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            # Assignment, not copy: one tensor with two names. nn.Module.parameters() dedupes,
            # so the parameter count reflects the saving.
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2's initialization: normal(0, 0.02), biases zeroed."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        """Shrink the projections that write into the residual stream.

        Every block adds twice into the stream (attention and MLP), so its variance grows with
        depth. Scaling the *output* projections by 1/sqrt(2 * n_layers) keeps the scale stable
        however deep the model is.
        """
        scale = 1.0 / math.sqrt(2 * self.cfg.n_layers)
        with torch.no_grad():
            for block in self.blocks:
                block.attn.W_o.weight.mul_(scale)
                block.ffn.down.weight.mul_(scale)

    @overload
    def forward(self, idx: Tensor, targets: None = None, cache=None) -> Tensor: ...

    @overload
    def forward(self, idx: Tensor, targets: Tensor, cache=None) -> tuple[Tensor, Tensor]: ...

    def forward(
        self, idx: Tensor, targets: Tensor | None = None, cache=None
    ) -> Tensor | tuple[Tensor, Tensor]:
        """idx: (B, T) token ids. Returns logits (B, T, V), or (logits, loss) given targets.

        With a `cache`, `idx` holds only the *new* tokens and positions continue from where the
        cache left off.
        """
        _, T = idx.shape
        if T == 0:
            # A language model has nothing to condition on. Reshaping an empty tensor fails
            # several layers down with a message about dimension -1, which explains nothing.
            raise ValueError("idx is empty; a model needs at least one token to predict from")
        offset = cache.pos if cache is not None else 0
        if offset + T > self.cfg.context_length:
            raise ValueError(
                f"sequence length {offset + T} exceeds context_length {self.cfg.context_length}"
            )

        h = self.tok_emb(idx)
        if self.pos_emb is not None:
            # Offset matters: a token decoded at absolute position 40 must get position 40's
            # embedding, not position 0's.
            h = h + self.pos_emb(torch.arange(offset, offset + T, device=idx.device))
        h = self.drop(h)

        for block in self.blocks:
            h = block(h, rope=self.rope, cache=cache)

        # Once per forward, not once per layer - every layer wrote at the same position.
        if cache is not None:
            cache.advance(T)

        logits = self.head(self.final_norm(h))
        if targets is None:
            return logits

        # Flatten to (B*T, V) against (B*T,). cross_entropy takes raw logits, not probabilities:
        # it fuses log-softmax with the negative log-likelihood for numerical stability.
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Total parameters. Tied weights are counted once, which is the point of tying."""
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                total -= self.pos_emb.weight.numel()
        return total

    def parameter_breakdown(self) -> dict[str, int]:
        """Per-component counts, for comparison against docs/ARCHITECTURE.md section 5."""
        out = {"token embedding": self.tok_emb.weight.numel()}
        if self.pos_emb is not None:
            out["position embedding"] = self.pos_emb.weight.numel()
        out["blocks"] = sum(p.numel() for b in self.blocks for p in b.parameters())
        out["final norm"] = sum(p.numel() for p in self.final_norm.parameters())
        if not self.cfg.tie_weights:
            out["output head"] = self.head.weight.numel()
        return out

    @torch.no_grad()
    def generate_greedy(self, idx: Tensor, max_new_tokens: int) -> Tensor:
        """Deterministic generation: always take the most likely token.

        The minimal version, so note 4 can demonstrate the model end to end. Temperature, top-k,
        top-p and KV caching arrive in note 6.
        """
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                # Crop to what the model can see. Absolute position embeddings simply run out of
                # rows past context_length -- one of the reasons RoPE exists.
                window = idx[:, -self.cfg.context_length :]
                # Only the last position predicts a token that does not exist yet; the rest were
                # the training signal.
                next_id = self.forward(window)[:, -1, :].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_id], dim=1)
            return idx
        finally:
            self.train(was_training)


if __name__ == "__main__":
    from llm.config import GPT2_124M, SMALL

    for name, cfg in [("GPT2_124M", GPT2_124M), ("SMALL", SMALL)]:
        model = GPTModel(cfg)
        print(f"\n{name}")
        for component, n in model.parameter_breakdown().items():
            print(f"  {component:<22} {n:>12,}")
        print(f"  {'total (tied)':<22} {model.num_parameters():>12,}")
        print(f"  {'non-embedding':<22} {model.num_parameters(non_embedding=True):>12,}")
        print(f"  {'config predicted':<22} {cfg.n_params():>12,}")

    model = GPTModel(SMALL)
    idx = torch.randint(0, SMALL.vocab_size, (2, 16))
    targets = torch.randint(0, SMALL.vocab_size, (2, 16))
    logits, loss = model(idx, targets=targets)
    print(f"\nlogits {tuple(logits.shape)}   loss {loss.item():.4f}")
    print(f"ln(vocab_size) = {math.log(SMALL.vocab_size):.4f}   <- expected at random init")

    # Targets must be independent of the inputs for that expectation to hold. With tied weights
    # the residual stream carries the token embedding forward into a head that *is* the token
    # embedding, so an untrained model already leans toward predicting the current token.
    with torch.no_grad():
        p_self = model(idx).softmax(-1).gather(-1, idx.unsqueeze(-1)).mean()
    print(f"\nP(input token) = {p_self:.6f}  vs uniform {1 / SMALL.vocab_size:.6f}")
    print("Weight tying gives the current token a head start before any training.")
