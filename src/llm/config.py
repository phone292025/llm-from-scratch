"""Every architecture knob in one place.

Pure data - nothing here imports torch. This is what makes v1 -> v2 a flag change instead of a
rewrite, and it is what lets two architectures be trained on identical data and compared.

See docs/ARCHITECTURE.md section 3.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

NormKind = Literal["layernorm", "rmsnorm"]
PosKind = Literal["learned", "rope", "none"]
ActKind = Literal["gelu", "swiglu"]
AttnKind = Literal["manual", "sdpa"]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int | None = None
    """Key/value heads. None means one per query head (plain MHA); fewer means GQA."""
    emb_dim: int = 768
    ffn_hidden: int | None = None
    """Feed-forward inner width. None means the standard width for the chosen activation."""

    norm: NormKind = "layernorm"
    pos: PosKind = "learned"
    act: ActKind = "gelu"
    attn: AttnKind = "sdpa"
    """'manual' is the readable reference implementation; 'sdpa' dispatches to FlashAttention."""

    qkv_bias: bool = True
    """GPT-2 has a bias on the QKV projection. Later work found it contributes nothing."""
    tie_weights: bool = True
    dropout: float = 0.0
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        if self.emb_dim % self.n_heads:
            raise ValueError(
                f"emb_dim={self.emb_dim} must divide evenly into n_heads={self.n_heads}"
            )
        if self.n_heads % self.kv_heads:
            raise ValueError(
                f"n_heads={self.n_heads} must be a multiple of n_kv_heads={self.kv_heads}, "
                f"so each key/value head serves an equal group of query heads"
            )

    @property
    def head_dim(self) -> int:
        return self.emb_dim // self.n_heads

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads or self.n_heads

    @property
    def n_rep(self) -> int:
        """Query heads per key/value head. 1 for MHA."""
        return self.n_heads // self.kv_heads

    @property
    def ffn_hidden_dim(self) -> int:
        """The inner width of the feed-forward network.

        GELU uses two matrices, so the classic 4x. SwiGLU uses three, so the width shrinks to
        8/3x to keep the parameter count equal -- otherwise a SwiGLU model is simply bigger and
        any comparison against the GELU baseline is meaningless. Rounded to a multiple of 256 for
        hardware alignment.
        """
        if self.ffn_hidden is not None:
            return self.ffn_hidden
        if self.act == "gelu":
            return 4 * self.emb_dim
        target = int(8 * self.emb_dim / 3)
        return ((target + 255) // 256) * 256

    def n_params(self, tied: bool | None = None) -> int:
        """Predicted parameter count, for comparison against the real one.

        A mismatch against `model.num_parameters()` means a missing bias or a dropped layer.
        """
        tied = self.tie_weights if tied is None else tied
        d, hd = self.emb_dim, self.head_dim
        norm_params = d if self.norm == "rmsnorm" else 2 * d

        qkv_out = (self.n_heads + 2 * self.kv_heads) * hd
        attn = d * qkv_out + (qkv_out if self.qkv_bias else 0)  # qkv projection
        attn += d * d + d  # output projection

        h = self.ffn_hidden_dim
        ffn = 3 * d * h if self.act == "swiglu" else (d * h + h) + (h * d + d)

        total = self.vocab_size * d  # token embedding
        if self.pos == "learned":
            total += self.context_length * d
        total += self.n_layers * (attn + ffn + 2 * norm_params)
        total += norm_params  # final norm
        if not tied:
            total += self.vocab_size * d
        return total


#: The 2019 architecture, exactly. Loads real OpenAI weights - the correctness oracle.
GPT2_124M = ModelConfig()

#: ~30M parameters. What actually trains well in 8GB - see docs/ARCHITECTURE.md section 6.
SMALL = ModelConfig(emb_dim=384, n_layers=6, n_heads=6, context_length=512)

#: v2: the same size as SMALL, with the modern components. See notes/07_modern_upgrades.md.
MODERN = replace(
    SMALL, norm="rmsnorm", pos="rope", act="swiglu", n_kv_heads=2, qkv_bias=False
)

PRESETS: dict[str, ModelConfig] = {"gpt2": GPT2_124M, "small": SMALL, "modern": MODERN}


if __name__ == "__main__":
    print(f"{'preset':<8} {'params':>12} {'d':>5} {'L':>3} {'H':>3} {'kv':>3} {'ffn':>6}  flags")
    for name, cfg in PRESETS.items():
        flags = f"{cfg.norm}/{cfg.pos}/{cfg.act}"
        print(
            f"{name:<8} {cfg.n_params():>12,} {cfg.emb_dim:>5} {cfg.n_layers:>3} "
            f"{cfg.n_heads:>3} {cfg.kv_heads:>3} {cfg.ffn_hidden_dim:>6}  {flags}"
        )
    print(f"\ngpt2 untied: {GPT2_124M.n_params(tied=False):,}  (should be 163,009,536)")
    print(f"gpt2 tied:   {GPT2_124M.n_params(tied=True):,}  (should be 124,412,160)")
