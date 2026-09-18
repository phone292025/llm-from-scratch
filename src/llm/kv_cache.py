"""Keys and values remembered across generation steps.

Generating token 500 without a cache reruns the model over all 499 previous tokens, recomputing
keys and values that have not changed since the step before. They *cannot* change: the causal
mask guarantees token 5's key depends only on tokens 1-5, so appending token 500 leaves it alone.

    without cache:  step n processes n tokens   -> O(T^2) for the whole sequence
    with cache:     step n processes 1 token    -> O(T)

The cost is memory, and it is not small. See notes/06_generation.md section 3.
"""

from __future__ import annotations

import torch
from torch import Tensor

from llm.config import ModelConfig


class KVCache:
    """Preallocated key/value storage, one pair of tensors per layer.

    Preallocated rather than grown by concatenation: `torch.cat` on every step of every layer
    reallocates and copies the whole history each time, which throws away most of the speedup
    the cache exists to provide.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        batch_size: int,
        max_len: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.cfg = cfg
        self.batch_size = batch_size
        self.max_len = max_len or cfg.context_length
        # Under GQA there are fewer key/value heads than query heads, and the cache stores the
        # narrow version -- which is the entire point of GQA at inference time.
        shape = (batch_size, cfg.kv_heads, self.max_len, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        self.pos = 0
        """How many positions are currently filled. Advanced once per forward, not per layer."""

    def update(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Write this step's keys and values, and return everything cached so far.

        Does **not** advance `pos`: every layer writes at the same position during one forward
        pass, so the model advances it once afterwards.
        """
        new = k.shape[2]
        end = self.pos + new
        if end > self.max_len:
            raise ValueError(
                f"cache holds {self.max_len} positions, step would need {end}. "
                f"Build the cache with a larger max_len."
            )
        self.k[layer][:, :, self.pos : end] = k
        self.v[layer][:, :, self.pos : end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def advance(self, n: int) -> None:
        self.pos += n

    def reset(self) -> None:
        """Reuse the allocation for a new sequence. Stale values are never read, because every
        read is bounded by `pos`."""
        self.pos = 0

    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

    @staticmethod
    def predicted_bytes(
        cfg: ModelConfig, batch_size: int, seq_len: int, bytes_per_element: int = 2
    ) -> int:
        """2 (k and v) x layers x kv_heads x head_dim x seq_len x batch x bytes."""
        return (
            2
            * cfg.n_layers
            * cfg.kv_heads
            * cfg.head_dim
            * seq_len
            * batch_size
            * bytes_per_element
        )


if __name__ == "__main__":
    from dataclasses import replace

    from llm.config import GPT2_124M, SMALL

    print(f"{'model':<22} {'kv heads':>9} {'1024 tokens, bf16':>20}")
    for name, cfg in [
        ("GPT-2 124M (MHA)", GPT2_124M),
        ("SMALL (MHA)", SMALL),
        ("MODERN (GQA, 4 kv)", replace(SMALL, n_kv_heads=4)),
        ("MQA (1 kv head)", replace(SMALL, n_kv_heads=1)),
    ]:
        mb = KVCache.predicted_bytes(cfg, batch_size=1, seq_len=1024) / 1024**2
        print(f"  {name:<20} {cfg.kv_heads:>9} {mb:>17.1f} MB")

    print("\nPer sequence. Batch 32 multiplies all of it by 32, and for a large model with long")
    print("context the cache exceeds the size of the model itself. That is what GQA is for.")

    cache = KVCache(SMALL, batch_size=2, max_len=64)
    print(f"\nallocated {cache.memory_bytes() / 1024**2:.1f} MB for 2 x 64 tokens (fp32)")
    k = torch.randn(2, SMALL.kv_heads, 5, SMALL.head_dim)
    k_all, _ = cache.update(0, k, k)
    cache.advance(5)
    print(f"after writing 5 positions: pos={cache.pos}, readable slice {tuple(k_all.shape)}")
