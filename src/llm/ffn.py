"""The feed-forward network: where each position does its own computing.

Attention only ever averages, and averaging is linear -- stack a hundred attention layers with
nothing between them and the whole thing collapses to one linear transformation. The MLP supplies
the nonlinearity, and two thirds of the non-embedding parameters.

Every position passes through independently. Row 5 of the input produces row 5 of the output and
never touches row 4.

See notes/03_transformer_block.md section 2.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from llm.config import ModelConfig


class GELU(nn.Module):
    """The tanh approximation, which is the variant GPT-2 uses.

    ReLU is a hard gate: negative in, exactly zero out, zero gradient, and a unit that goes
    negative can be stuck there permanently. GELU is a soft gate -- small negatives pass through
    attenuated, the gradient is never exactly zero, and the curve is smooth everywhere.
    """

    def forward(self, x: Tensor) -> Tensor:
        return (
            0.5
            * x
            * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        )


class GELUFeedForward(nn.Module):
    """Expand to 4x, apply the nonlinearity, contract back.

    The wide middle is where the computation happens: project into a space with room to test many
    features at once, gate each one, then project the results back down. The 4x ratio is empirical,
    from the original paper, and has survived largely unchanged.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.up = nn.Linear(cfg.emb_dim, cfg.ffn_hidden_dim, bias=cfg.linear_bias)
        self.act = GELU()
        self.down = nn.Linear(cfg.ffn_hidden_dim, cfg.emb_dim, bias=cfg.linear_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.act(self.up(x)))


class SwiGLUFeedForward(nn.Module):
    """A gated feed-forward network: one projection decides how much of another gets through.

    The elementwise multiply is a multiplicative interaction, which a plain MLP cannot express at
    all without extra depth.

    Three matrices instead of two, so the hidden width shrinks (to ~8d/3, see
    ModelConfig.ffn_hidden_dim) to keep the parameter count equal to the GELU version. Without
    that correction a SwiGLU model is simply bigger and the comparison is meaningless.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        h = cfg.ffn_hidden_dim
        # No biases: gated networks are conventionally built without them, and they contribute
        # nothing measurable.
        self.gate = nn.Linear(cfg.emb_dim, h, bias=False)
        self.up = nn.Linear(cfg.emb_dim, h, bias=False)
        self.down = nn.Linear(h, cfg.emb_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # silu(x) = x * sigmoid(x), essentially GELU's smoother cousin.
        return self.down(nn.functional.silu(self.gate(x)) * self.up(x))


def make_ffn(cfg: ModelConfig) -> nn.Module:
    return SwiGLUFeedForward(cfg) if cfg.act == "swiglu" else GELUFeedForward(cfg)


if __name__ == "__main__":
    from dataclasses import replace

    from llm.config import SMALL

    x = torch.randn(2, 8, SMALL.emb_dim)
    gelu_cfg = SMALL
    swiglu_cfg = replace(SMALL, act="swiglu")

    print(f"{'':<10} {'hidden':>8} {'params':>10} {'matrices':>10}")
    for name, cfg, n in [("GELU", gelu_cfg, 2), ("SwiGLU", swiglu_cfg, 3)]:
        ffn = make_ffn(cfg)
        params = sum(p.numel() for p in ffn.parameters())
        print(f"{name:<10} {cfg.ffn_hidden_dim:>8} {params:>10,} {n:>10}")
        assert ffn(x).shape == x.shape
    print("\nSwiGLU's narrower hidden width keeps the parameter count close to GELU's, so the")
    print("two can be compared fairly. Both map (B, T, d) -> (B, T, d).")

    ref = nn.functional.gelu(torch.randn(100), approximate="tanh")
    torch.manual_seed(0)
    t = torch.randn(100)
    print(f"\nGELU matches F.gelu(approximate='tanh'): "
          f"{torch.allclose(GELU()(t), nn.functional.gelu(t, approximate='tanh'), atol=1e-6)}")
