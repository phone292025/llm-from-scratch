"""Causal self-attention.

The only operation in the whole model that moves information between positions.

The file contains five classes. The first three are a teaching ladder, each fixing a flaw in the
one before, and they are kept rather than deleted because the progression is the explanation.
`CausalSelfAttention` is the one the model actually uses.

See notes/02_attention.md.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from llm.config import SMALL, ModelConfig
from llm.rope import apply_rope


def causal_mask(seq_len: int, device=None) -> Tensor:
    """True where attention must be blocked: strictly above the diagonal, i.e. the future."""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


# --- the teaching ladder ----------------------------------------------------------------------


class NaiveAttention(nn.Module):
    """Stage 1: similarity with no parameters at all.

    Dot every input vector against every other. This works a little, because the dot product is
    large for vectors pointing the same way -- but there is nothing to learn, and it can only ever
    express "attend to things like me". A pronoun does not want another pronoun; it wants a noun.
    """

    def forward(self, x: Tensor) -> Tensor:
        weights = torch.softmax(x @ x.transpose(-2, -1), dim=-1)
        return weights @ x


class SingleHeadAttention(nn.Module):
    """Stages 2-4: learned projections, scaling, and the causal mask.

    Three matrices let the model decide what to look for separately from what it is. W_q and W_k
    being different is what makes attention directional: A can attend to B without B attending
    to A.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_out = d_out
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # Not persistent: derived from config, so it should follow .to(device) but stay out of
        # the checkpoint.
        self.register_buffer("mask", causal_mask(context_length), persistent=False)

    def forward(self, x: Tensor, return_weights: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        T = x.shape[-2]
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)

        # Scaling by sqrt(d) is not cosmetic. Dot products of random vectors have variance
        # proportional to d, and large values saturate softmax -- where its gradient vanishes and
        # attention stops learning entirely.
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_out)

        # -inf *before* softmax, so exp() gives exactly 0 and the surviving weights renormalize
        # to sum to 1. Zeroing after softmax leaves rows summing to less than 1, which silently
        # shrinks the output toward zero.
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))

        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = weights @ v
        return (context, weights) if return_weights else context


class MultiHeadWrapper(nn.Module):
    """Stage 5, the naive version: a loop over independent heads, concatenated.

    Correct, and slow -- it runs h separate small matmuls where one batched matmul would do.
    Kept as the reference that `CausalSelfAttention` is tested against.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        n_heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        out_bias: bool = True,
    ) -> None:
        super().__init__()
        if d_out % n_heads:
            raise ValueError(f"d_out={d_out} must be divisible by n_heads={n_heads}")
        self.heads = nn.ModuleList(
            SingleHeadAttention(d_in, d_out // n_heads, context_length, dropout, qkv_bias)
            for _ in range(n_heads)
        )
        self.W_o = nn.Linear(d_out, d_out, bias=out_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.W_o(torch.cat([head(x) for head in self.heads], dim=-1))


# --- the real one -----------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Batched multi-head (or grouped-query) causal attention.

    All heads in one matmul. The whole trick is two reshapes: split the feature dimension into
    (heads, head_dim), then move heads next to the batch dimension so `@` treats them as extra
    batch dims.

    With `n_kv_heads < n_heads` this is grouped-query attention: fewer key/value heads, each
    shared across a group of query heads, which shrinks the KV cache by the group size. See
    notes/02_attention.md section 5.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.kv_heads = cfg.kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_rep
        self.dropout = cfg.dropout

        # One fused projection for q, k and v. Under GQA the k and v halves are narrower, so the
        # output is (n_heads + 2 * kv_heads) * head_dim rather than 3 * emb_dim.
        self.q_dim = cfg.n_heads * cfg.head_dim
        self.kv_dim = cfg.kv_heads * cfg.head_dim
        self.W_qkv = nn.Linear(cfg.emb_dim, self.q_dim + 2 * self.kv_dim, bias=cfg.qkv_bias)

        # Without this the heads sit in disjoint slices, unable to influence each other. W_o
        # mixes them back into a single representation.
        self.W_o = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.linear_bias)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("mask", causal_mask(cfg.context_length), persistent=False)

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, T, _ = x.shape
        q, k, v = self.W_qkv(x).split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        # (B, T, d) -> (B, T, H, hd) -> (B, H, T, hd)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _expand_kv(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Share each key/value head across its group of query heads (GQA). A no-op for MHA."""
        if self.n_rep == 1:
            return k, v
        return (
            k.repeat_interleave(self.n_rep, dim=1),
            v.repeat_interleave(self.n_rep, dim=1),
        )

    def forward(self, x: Tensor, rope=None, cache=None) -> Tensor:
        """`rope`, when given, rotates q and k by position before the scores are formed.

        With a `cache`, only the new tokens are projected, and they attend over every key and
        value stored so far.
        """
        B, T, _ = x.shape
        # Checked here rather than left to the mask: the manual path would raise an opaque
        # broadcast error, and the SDPA path would quietly succeed with `is_causal` building its
        # own mask, so the same config would be rejected by one backend and accepted by the other.
        past = cache.pos if cache is not None else 0
        if past + T > self.cfg.context_length:
            raise ValueError(
                f"sequence length {past + T} exceeds context_length "
                f"{self.cfg.context_length}"
            )
        q, k, v = self._project(x)

        if rope is not None:
            # Rotate by ABSOLUTE position, and do it before the cache write, so a cached key
            # keeps the angle of the position it was written at. Rotating afterwards would
            # re-rotate the whole history on every decode step and destroy it.
            # v is deliberately left alone: position belongs in the comparison, not the content.
            cos, sin = rope.slice(past, T)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if cache is not None:
            # Cache the *narrow* k and v, before the GQA expansion. Storing the expanded copies
            # would throw away the memory saving that GQA exists for.
            k, v = cache.update(self.layer_idx, k, v)

        k, v = self._expand_kv(k, v)
        dropout = self.dropout if self.training else 0.0

        # One new token after a cached prefix is the whole of steady-state decoding, and it needs
        # no mask at all: the query sits at the newest position, so every cached key is in its
        # past. Building the mask anyway costs two aranges and a compare per layer per token,
        # and handing SDPA any explicit mask moves it off its fastest kernel -- together enough
        # to make cached decoding measurably *slower* than recomputing everything.
        unmasked = past > 0 and T == 1

        if self.cfg.attn == "sdpa" and past == 0:
            # Same mathematics as the manual path, but dispatches to FlashAttention: the
            # (B, H, T, T) score tensor is never materialized.
            ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout)
        elif self.cfg.attn == "sdpa" and unmasked:
            ctx = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)
        elif self.cfg.attn == "sdpa":
            # Several new tokens after a prefix (chunked prefill). is_causal assumes queries and
            # keys are aligned, which stops being true once the cache holds a prefix. attn_mask
            # is bool with True meaning "may attend".
            ctx = F.scaled_dot_product_attention(
                q, k, v, attn_mask=~self._offset_mask(T, past, x.device), dropout_p=dropout
            )
        else:
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            if not unmasked:
                scores = scores.masked_fill(self._offset_mask(T, past, x.device), float("-inf"))
            ctx = self.attn_dropout(torch.softmax(scores, dim=-1)) @ v

        # (B, H, T, hd) -> (B, T, H, hd) -> (B, T, d).
        # contiguous() is required because transpose only changes how the tensor is read, not
        # how it is laid out, and view needs a contiguous layout.
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(ctx)

    def _offset_mask(self, n_query: int, past: int, device) -> Tensor:
        """Causal mask for `n_query` new queries attending over `past + n_query` keys.

        Query i sits at absolute position `past + i` and may see keys 0..past+i, so the mask is
        the usual upper triangle shifted right by `past`. With `past == 0` this is exactly the
        square causal mask; when decoding one token at a time it is all-False, since a single
        new query may attend to the entire history.
        """
        if past == 0:
            return self.mask[:n_query, :n_query]
        total = past + n_query
        rows = torch.arange(past, total, device=device).unsqueeze(1)
        cols = torch.arange(total, device=device).unsqueeze(0)
        return cols > rows

    @torch.no_grad()
    def attention_weights(self, x: Tensor, rope=None) -> Tensor:
        """The (B, H, T, T) weight matrix, for inspection. Always takes the manual path."""
        T = x.shape[1]
        q, k, _ = self._project(x)
        if rope is not None:
            cos, sin = rope.slice(0, T)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k, _ = self._expand_kv(k, k)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        return torch.softmax(scores.masked_fill(self.mask[:T, :T], float("-inf")), dim=-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = SMALL
    x = torch.randn(2, 16, cfg.emb_dim)

    attn = CausalSelfAttention(cfg)
    print(f"{'input':<28} {tuple(x.shape)}")
    q, k, v = attn._project(x)
    print(f"{'q after head split':<28} {tuple(q.shape)}   (B, H, T, hd)")
    print(f"{'k after head split':<28} {tuple(k.shape)}")
    print(f"{'output':<28} {tuple(attn(x).shape)}")

    w = attn.attention_weights(x)
    print(f"\n{'attention weights':<28} {tuple(w.shape)}   <- quadratic in T")
    print(f"{'rows sum to':<28} {w.sum(-1).mean().item():.6f}")
    print(f"{'weight above diagonal':<28} {w.triu(1).abs().max().item():.1e}  (must be 0)")

    from dataclasses import replace

    gqa = CausalSelfAttention(replace(cfg, n_kv_heads=2))
    print(f"\nMHA qkv projection: {tuple(attn.W_qkv.weight.shape)}")
    print(f"GQA qkv projection: {tuple(gqa.W_qkv.weight.shape)}   <- narrower k and v")
