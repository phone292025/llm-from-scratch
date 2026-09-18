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
    linear_bias: bool = True
    """Bias on the *other* linear layers: the attention output projection and the GELU MLP.

    GPT-2 has them, so the default is True and the checkpoint loads. Modern models (Llama, PaLM,
    Gemma) drop biases everywhere, which is why SMALL and MODERN set this False -- `qkv_bias`
    alone is not enough to make a bias-free model, and believing otherwise leaves three bias
    tensors per block that the config claims are gone. SwiGLU never has biases either way.
    """
    tie_weights: bool = True
    dropout: float = 0.0
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        # Everything here is cheap and runs once. A config that is quietly wrong costs hours:
        # n_kv_heads=0 used to pass straight through `kv_heads` (0 is falsy) and silently build
        # plain MHA, so a run intended as MQA was not one and nothing said so.
        positive = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "emb_dim": self.emb_dim,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name}={value} must be >= 1")
        if self.n_kv_heads is not None and self.n_kv_heads < 1:
            raise ValueError(
                f"n_kv_heads={self.n_kv_heads} must be >= 1, or None for one per query head"
            )
        if self.ffn_hidden is not None and self.ffn_hidden < 1:
            raise ValueError(f"ffn_hidden={self.ffn_hidden} must be >= 1, or None for the default")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout={self.dropout} must be in [0, 1)")
        if self.rope_base <= 0:
            raise ValueError(f"rope_base={self.rope_base} must be positive")

        if self.emb_dim % self.n_heads:
            raise ValueError(
                f"emb_dim={self.emb_dim} must be divisible by n_heads={self.n_heads}"
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
        any comparison against the GELU baseline is meaningless.

        Rounded to the *nearest* multiple of 64, not up. 64 is the alignment that matters for
        tensor cores, and rounding up to a coarser boundary overshoots badly at small widths:
        at emb_dim=512 the target is 1365, which rounds up to 1536 (12% too many parameters)
        but to the nearest 64 gives 1344 -- the value CS336 uses for exactly this model size.
        """
        if self.ffn_hidden is not None:
            return self.ffn_hidden
        if self.act == "gelu":
            return 4 * self.emb_dim
        target = 8 * self.emb_dim / 3
        return max(64, round(target / 64) * 64)

    def n_params(self, tied: bool | None = None) -> int:
        """Predicted parameter count, for comparison against the real one.

        A mismatch against `model.num_parameters()` means a missing bias or a dropped layer.
        """
        tied = self.tie_weights if tied is None else tied
        d, hd = self.emb_dim, self.head_dim
        norm_params = d if self.norm == "rmsnorm" else 2 * d

        qkv_out = (self.n_heads + 2 * self.kv_heads) * hd
        attn = d * qkv_out + (qkv_out if self.qkv_bias else 0)  # qkv projection
        attn += d * d + (d if self.linear_bias else 0)  # output projection

        h = self.ffn_hidden_dim
        if self.act == "swiglu":
            ffn = 3 * d * h  # gate, up, down -- never biased
        else:
            bias = (h + d) if self.linear_bias else 0
            ffn = d * h + h * d + bias

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

#: What we actually train. Shape follows CS336 lecture 3 and their validated TinyStories run:
#: aspect ratio emb_dim/n_layers = 128, inside the 100-200 band that nearly every production
#: model sits in (BLOOM 205, PaLM 156, GPT-3 128, Llama 102). Bias terms off, as modern models
#: do everywhere -- which takes both flags, not just `qkv_bias`.
SMALL = ModelConfig(
    emb_dim=512,
    n_layers=4,
    n_heads=16,
    context_length=256,
    qkv_bias=False,
    linear_bias=False,
)

#: SMALL with a vocabulary sized for a simple corpus.
#: At vocab 50257 the embedding table is 60% of a model this small, and most of those rows are
#: never used by children's-story text. A 10k vocabulary trained on the corpus itself cuts the
#: table from 25.7M to 5.1M parameters and moves that capacity into the layers. Requires a
#: tokenizer trained on the corpus - see docs/CS336_FINDINGS.md.
TINYSTORIES = replace(SMALL, vocab_size=10_000)

#: v2: the same shape as SMALL, with the modern components. See notes/07_modern_upgrades.md.
#: **Not yet trainable**: `pos="rope"` has no implementation behind it until note 7, and
#: `GPTModel` refuses to build it rather than quietly producing a model with no sense of order.
MODERN = replace(SMALL, norm="rmsnorm", pos="rope", act="swiglu", n_kv_heads=4)

PRESETS: dict[str, ModelConfig] = {
    "gpt2": GPT2_124M,
    "small": SMALL,
    "tinystories": TINYSTORIES,
    "modern": MODERN,
}


if __name__ == "__main__":
    header = f"{'preset':<12} {'params':>12} {'non-emb':>11} {'emb %':>6} {'d':>5} {'L':>3} {'H':>3} {'kv':>3} {'ffn':>5} {'d/L':>5}"
    print(header)
    print("-" * len(header))
    for name, cfg in PRESETS.items():
        emb = cfg.vocab_size * cfg.emb_dim
        total = cfg.n_params()
        print(
            f"{name:<12} {total:>12,} {total - emb:>11,} {emb / total:>6.0%} "
            f"{cfg.emb_dim:>5} {cfg.n_layers:>3} {cfg.n_heads:>3} {cfg.kv_heads:>3} "
            f"{cfg.ffn_hidden_dim:>5} {cfg.emb_dim / cfg.n_layers:>5.0f}"
        )
    print("\nd/L is the aspect ratio. CS336 lecture 3: production models cluster in 100-200.")
    print(f"\ngpt2 untied: {GPT2_124M.n_params(tied=False):,}  (expect 163,037,184)")
    print(f"gpt2 tied:   {GPT2_124M.n_params(tied=True):,}  (expect 124,439,808)")
