"""Rotary position embeddings.

Learned position embeddings add a vector for "you are at slot 5". RoPE instead **rotates** each
query and key by an angle proportional to its position. The trick is what that does to attention:
attention scores are dot products, and the dot product of two rotated vectors depends only on the
difference between their angles. Absolute positions go in, relative distance comes out, for free,
from geometry.

Three consequences follow:

- no parameters at all -- the position table is gone
- it extrapolates past the trained length, imperfectly but usably, because position 2000 is just
  a larger angle rather than a table row that does not exist
- it applies to q and k inside attention, never to v and never to the residual stream, because it
  is the *comparison* between positions that should be relative, not the content

See notes/07_modern_upgrades.md.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate each adjacent pair of dimensions of `x` by the given angles.

    `x` is (B, H, T, head_dim); `cos` and `sin` are (T, head_dim/2).

    Coordinates are taken two at a time -- (0,1), (2,3), ... -- and each pair treated as a point
    in a plane, turned by the angle for its position:

        x0' = x0 cos - x1 sin
        x1' = x0 sin + x1 cos

    Llama's implementation pairs (i, i + head_dim/2) instead. That is the same rotation under a
    permutation of the dimensions, and equivalent when training from scratch; this ordering is
    the one the original paper writes down, and the one note 7 explains.
    """
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    cos, sin = cos[None, None], sin[None, None]  # broadcast over batch and heads
    rotated = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return rotated.flatten(-2).type_as(x)


class RopeCache(nn.Module):
    """The cos and sin tables, computed once.

    Each pair of dimensions gets its own frequency, in a geometric series: the first pairs rotate
    fast and encode fine local position, the last rotate slowly and encode coarse long-range
    position. One table serves every layer, so it is built by the model and passed down.
    """

    def __init__(self, head_dim: int, max_len: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"head_dim={head_dim} must be even to rotate in pairs")
        self.head_dim, self.max_len, self.base = head_dim, max_len, base

        # theta_i = base^(-2i/d): 1.0 down to nearly 1/base across the pairs.
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        angles = torch.outer(torch.arange(max_len, dtype=torch.float32), inv_freq)
        # Not persistent: derived from config, so it should follow .to(device) but stay out of
        # checkpoints, exactly like the causal mask.
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def slice(self, offset: int, length: int) -> tuple[Tensor, Tensor]:
        """Angles for positions `offset .. offset+length`.

        The offset is what makes the KV cache work: a token decoded at absolute position 40 must
        be rotated by position 40's angle, not by 0 because it happens to be the first token of
        this forward pass.
        """
        end = offset + length
        if end > self.max_len:
            raise ValueError(
                f"position {end} is beyond the {self.max_len} this cache was built for"
            )
        return self.cos[offset:end], self.sin[offset:end]

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, max_len={self.max_len}, base={self.base}"


if __name__ == "__main__":
    from llm.console import use_utf8

    use_utf8()
    head_dim, T = 8, 6
    rope = RopeCache(head_dim, max_len=64)
    torch.manual_seed(0)
    q = torch.randn(1, 1, T, head_dim)
    k = torch.randn(1, 1, T, head_dim)
    cos, sin = rope.slice(0, T)
    qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

    print("1. rotation preserves length (it is a rotation, not a rescaling)")
    print(f"   before {q[0,0,3].norm():.6f}   after {qr[0,0,3].norm():.6f}")

    print("\n2. the score depends only on the distance between positions")
    print("   q at m, k at n, for several (m, n) with the same m-n:")
    for m, n in ((1, 0), (2, 1), (3, 2), (4, 3)):
        score = (qr[0, 0, m] * kr[0, 0, n]).sum()
        print(f"     m={m} n={n}  (m-n={m-n})  score {score:+.4f}")
    print("   Those differ, because q and k differ at each position. The property is about the")
    print("   *rotation*: rotating both by the same extra amount leaves the score unchanged.")

    shifted_cos, shifted_sin = rope.slice(10, T)
    q2, k2 = apply_rope(q, shifted_cos, shifted_sin), apply_rope(k, shifted_cos, shifted_sin)
    a = (apply_rope(q, cos, sin)[0, 0, 3] * apply_rope(k, cos, sin)[0, 0, 1]).sum()
    b = (q2[0, 0, 3] * k2[0, 0, 1]).sum()
    print(f"\n   same pair at positions 3,1: {a:+.6f}")
    print(f"   the same pair shifted to 13,11: {b:+.6f}")
    print("   Identical: the model cannot tell where the window sits, only how far apart things are.")

    print(f"\n3. no parameters: {sum(p.numel() for p in rope.parameters())}")
    print(f"   a learned table for 1024 positions x 512 dims would be "
          f"{1024 * 512:,} parameters")
