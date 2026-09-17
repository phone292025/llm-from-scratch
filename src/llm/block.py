"""One transformer block.

    h = h + attn(norm(h))     mix information ACROSS positions
    h = h + mlp( norm(h))     compute WITHIN each position

Four lines of idea. Shape in equals shape out, which is precisely why blocks stack to any depth
with nothing needed in between.

See notes/03_transformer_block.md.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from llm.attention import CausalSelfAttention
from llm.config import ModelConfig
from llm.ffn import make_ffn
from llm.norms import make_norm


class TransformerBlock(nn.Module):
    """Pre-norm attention and feed-forward sublayers, each wrapped in a residual connection.

    Pre-norm (`h + f(norm(h))`) rather than post-norm (`norm(h + f(h))`). The reordering looks
    trivial and is not: post-norm puts the normalization *on* the residual path, interrupting the
    gradient highway at every layer, and deep post-norm transformers need careful warmup to
    converge at all. Pre-norm leaves the path from loss to embeddings completely clean.

    One consequence: nothing normalizes the output of the last block, so the model needs a final
    norm after the stack.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = make_norm(cfg)
        self.ffn = make_ffn(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, rope=None, cache=None) -> Tensor:
        # The `x +` is the whole reason deep stacks train. Differentiating x + f(x) gives
        # 1 + f'(x), and that 1 carries gradient backward through the addition unchanged no
        # matter what f does.
        x = x + self.drop(self.attn(self.norm1(x), rope=rope, cache=cache))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class PostNormBlock(nn.Module):
    """The 2017 ordering, kept only for the demo that shows why nobody uses it.

    Most transformer diagrams still draw this arrangement, because they are redrawings of the
    original paper's figure. Look for where the norm sits relative to the residual arrow.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.norm1 = make_norm(cfg)
        self.ffn = make_ffn(cfg)
        self.norm2 = make_norm(cfg)

    def forward(self, x: Tensor, rope=None, cache=None) -> Tensor:
        x = self.norm1(x + self.attn(x, rope=rope, cache=cache))
        x = self.norm2(x + self.ffn(x))
        return x


if __name__ == "__main__":
    from dataclasses import replace

    from llm.config import SMALL

    cfg = replace(SMALL, context_length=64)
    block = TransformerBlock(cfg)
    x = torch.randn(2, 16, cfg.emb_dim)

    print(f"in  {tuple(x.shape)}")
    print(f"out {tuple(block(x).shape)}   <- identical, which is why blocks stack\n")

    counts = {name: sum(p.numel() for p in mod.parameters())
              for name, mod in [("norm1", block.norm1), ("attn", block.attn),
                                ("norm2", block.norm2), ("ffn", block.ffn)]}
    total = sum(counts.values())
    for name, n in counts.items():
        print(f"  {name:<6} {n:>10,}  {n / total:>5.1%}")
    print(f"  {'total':<6} {total:>10,}")

    # A block whose sublayers output zero must be the identity -- that is the residual wiring.
    with torch.no_grad():
        for p in block.attn.W_o.parameters():
            p.zero_()
        for p in block.ffn.down.parameters():
            p.zero_()
    print(f"\nzeroed sublayers -> identity: {torch.allclose(block(x), x, atol=1e-6)}")
