"""Normalization layers.

Deep stacks have a scale problem: each layer multiplies by weights, so activations that grow even
slightly per layer have exploded twelve layers later, and ones that shrink have vanished.
Normalizing each position's vector back to a fixed scale is what makes depth trainable.

Both are hand-written rather than using torch.nn.LayerNorm, because the mean/variance arithmetic
is the point. A test checks ours against PyTorch's.

See notes/03_transformer_block.md section 3.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from llm.config import ModelConfig


class LayerNorm(nn.Module):
    """Normalize each position to mean 0 and variance 1, then rescale and shift.

    Parameters are named `scale` and `shift` (gamma and beta). Initialized to 1 and 0, so the
    layer starts as pure normalization and learns how much of it to undo.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Over the last dimension only: each token normalizes independently of every other token
        # and of the rest of the batch. That independence is why this works at any batch size,
        # unlike BatchNorm.
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False is the biased (/n) variance. Using the sample (/n-1) variance instead
        # produces a small systematic mismatch against torch.nn.LayerNorm -- it still trains, so
        # the bug survives, but GPT-2 weight loading then gives subtly wrong output.
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.scale + self.shift


class RMSNorm(nn.Module):
    """LayerNorm without the mean subtraction or the bias.

    The finding, and it was a surprise, is that the re-centering does nothing useful -- all the
    benefit comes from re-scaling. Half the parameters and measurably faster, at no quality cost.
    Llama, Mistral and Gemma all use this.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Computed in fp32 even under autocast: squaring bf16 values loses precision where it
        # matters most, and this is cheap.
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.scale


def make_norm(cfg: ModelConfig, dim: int | None = None) -> nn.Module:
    """Build the norm the config asks for. This indirection is how v2 becomes a flag change."""
    dim = cfg.emb_dim if dim is None else dim
    return RMSNorm(dim) if cfg.norm == "rmsnorm" else LayerNorm(dim)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8) * 5 + 3  # deliberately off-centre and wide

    print(f"{'':<12} {'mean':>10} {'var':>10}")
    print(f"{'input':<12} {x.mean():>10.4f} {x.var(unbiased=False):>10.4f}")
    for name, layer in [("LayerNorm", LayerNorm(8)), ("RMSNorm", RMSNorm(8))]:
        out = layer(x)
        print(f"{name:<12} {out.mean():>10.4f} {out.var(unbiased=False):>10.4f}")

    ref = nn.LayerNorm(8)
    print(f"\nmatches torch.nn.LayerNorm: {torch.allclose(LayerNorm(8)(x), ref(x), atol=1e-6)}")
    print(f"LayerNorm params: {sum(p.numel() for p in LayerNorm(8).parameters())}")
    print(f"RMSNorm params:   {sum(p.numel() for p in RMSNorm(8).parameters())}  <- half")
