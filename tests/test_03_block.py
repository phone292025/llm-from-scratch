"""Tests for note 3 - norms, feed-forward networks, and the block.

The parity tests against torch.nn.LayerNorm and F.gelu are what make these layers trustworthy for
GPT-2 weight loading later: a small systematic difference would still train fine, so nothing else
would catch it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from llm.block import PostNormBlock, TransformerBlock
from llm.config import SMALL, ModelConfig
from llm.ffn import GELU, GELUFeedForward, SwiGLUFeedForward, make_ffn
from llm.norms import LayerNorm, RMSNorm, make_norm

B, T = 2, 8
D = SMALL.emb_dim  # derived, so preset changes do not silently break these tests


@pytest.fixture
def cfg() -> ModelConfig:
    return replace(SMALL, context_length=32, attn="manual", dropout=0.0)


@pytest.fixture
def x() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, T, D)


# --- norms ----------------------------------------------------------------------------------


def test_layernorm_matches_pytorch(x: torch.Tensor) -> None:
    """Catches the unbiased-variance mistake and wrong eps placement.

    Either still trains, so only a parity check finds them -- and either would corrupt GPT-2
    weight loading.
    """
    mine, ref = LayerNorm(D), nn.LayerNorm(D)
    with torch.no_grad():
        mine.scale.copy_(ref.weight)
        mine.shift.copy_(ref.bias)
    assert torch.allclose(mine(x), ref(x), atol=1e-6)


def test_layernorm_matches_pytorch_with_learned_parameters(x: torch.Tensor) -> None:
    mine, ref = LayerNorm(D), nn.LayerNorm(D)
    w, b = torch.randn(D), torch.randn(D)
    with torch.no_grad():
        mine.scale.copy_(w)
        mine.shift.copy_(b)
        ref.weight.copy_(w)
        ref.bias.copy_(b)
    assert torch.allclose(mine(x), ref(x), atol=1e-5)


def test_layernorm_normalizes_each_position_independently(x: torch.Tensor) -> None:
    out = LayerNorm(D)(x)
    assert torch.allclose(out.mean(-1), torch.zeros(B, T), atol=1e-6)
    assert torch.allclose(out.var(-1, unbiased=False), torch.ones(B, T), atol=1e-3)


def test_layernorm_is_batch_size_independent(x: torch.Tensor) -> None:
    # Unlike BatchNorm: each token normalizes alone, so batching cannot change the result.
    norm = LayerNorm(D)
    assert torch.allclose(norm(x)[0], norm(x[:1])[0], atol=1e-6)


def test_rmsnorm_does_not_recentre(x: torch.Tensor) -> None:
    # The whole difference from LayerNorm: rescaling only, no mean subtraction.
    assert RMSNorm(D)(x + 5.0).mean().abs() > 0.1


def test_rmsnorm_matches_layernorm_on_zero_mean_input(x: torch.Tensor) -> None:
    centred = x - x.mean(-1, keepdim=True)
    assert torch.allclose(RMSNorm(D)(centred), LayerNorm(D)(centred), atol=1e-3)


def test_rmsnorm_has_half_the_parameters() -> None:
    assert sum(p.numel() for p in RMSNorm(D).parameters()) == D
    assert sum(p.numel() for p in LayerNorm(D).parameters()) == 2 * D


def test_norms_start_as_pure_normalization() -> None:
    # scale=1, shift=0 at init, so the layer begins as normalization and learns what to undo.
    assert torch.equal(LayerNorm(D).scale, torch.ones(D))
    assert torch.equal(LayerNorm(D).shift, torch.zeros(D))


def test_make_norm_respects_config(cfg: ModelConfig) -> None:
    assert isinstance(make_norm(cfg), LayerNorm)
    assert isinstance(make_norm(replace(cfg, norm="rmsnorm")), RMSNorm)


# --- feed-forward ---------------------------------------------------------------------------


def test_gelu_matches_pytorch_tanh_approximation() -> None:
    t = torch.linspace(-5, 5, 1000)
    assert torch.allclose(GELU()(t), F.gelu(t, approximate="tanh"), atol=1e-6)


def test_gelu_is_smooth_at_zero_unlike_relu() -> None:
    # ReLU's hard gate means a negative unit has exactly zero gradient and can stick there.
    x = torch.tensor([-2.0, -0.5, 0.0], requires_grad=True)
    GELU()(x).sum().backward()
    assert (x.grad != 0).all()


def test_ffn_preserves_shape(cfg: ModelConfig, x: torch.Tensor) -> None:
    for c in (cfg, replace(cfg, act="swiglu")):
        assert make_ffn(c)(x).shape == x.shape


def test_ffn_is_position_independent(cfg: ModelConfig, x: torch.Tensor) -> None:
    """The MLP never mixes positions - only attention does that."""
    ffn = make_ffn(cfg).eval()
    with torch.no_grad():
        out = ffn(x)
        perturbed = x.clone()
        perturbed[:, 0] = torch.randn(B, D)
        assert torch.allclose(ffn(perturbed)[:, 1:], out[:, 1:], atol=1e-6)


def test_gelu_expands_to_four_times(cfg: ModelConfig) -> None:
    assert cfg.ffn_hidden_dim == 4 * cfg.emb_dim
    assert GELUFeedForward(cfg).up.out_features == 4 * cfg.emb_dim


def test_swiglu_hidden_width_keeps_parameter_count_fair(cfg: ModelConfig) -> None:
    """Three matrices instead of two, so the width must shrink or the comparison is rigged.

    Exact parity is not reachable once the width is snapped to a multiple of 64, and real models
    do not chase it either -- CS336 lecture 3 reports d_ff/d_model between 2.67 and 4.0 across
    Llama, Mistral, Qwen and DeepSeek. A few percent is fine; 12% is not.
    """
    gelu = sum(p.numel() for p in GELUFeedForward(cfg).parameters())
    swiglu = sum(
        p.numel() for p in SwiGLUFeedForward(replace(cfg, act="swiglu")).parameters()
    )
    assert abs(gelu - swiglu) / gelu < 0.03


def test_swiglu_hidden_is_a_multiple_of_64(cfg: ModelConfig) -> None:
    # 64 is the alignment that matters for tensor cores.
    assert replace(cfg, act="swiglu").ffn_hidden_dim % 64 == 0


def test_swiglu_gate_actually_gates(cfg: ModelConfig, x: torch.Tensor) -> None:
    # Zero the gate projection: silu(0) = 0, so nothing passes through regardless of `up`.
    ffn = SwiGLUFeedForward(replace(cfg, act="swiglu"))
    with torch.no_grad():
        ffn.gate.weight.zero_()
    assert torch.allclose(ffn(x), torch.zeros_like(x), atol=1e-6)


# --- block ----------------------------------------------------------------------------------


def test_block_output_shape_equals_input(cfg: ModelConfig, x: torch.Tensor) -> None:
    # The property that makes an arbitrarily deep stack possible.
    assert TransformerBlock(cfg)(x).shape == x.shape


def test_block_with_zeroed_sublayers_is_the_identity(cfg: ModelConfig, x: torch.Tensor) -> None:
    """Proves the residual wiring: a block that contributes nothing must pass input through."""
    block = TransformerBlock(cfg).eval()
    with torch.no_grad():
        # Biases are optional (cfg.linear_bias), so zero whatever the projections actually have.
        for projection in (block.attn.W_o, block.ffn.down):
            for p in projection.parameters():
                p.zero_()
    assert torch.allclose(block(x), x, atol=1e-6)


def test_block_is_causal(cfg: ModelConfig) -> None:
    # Causality must survive assembly, not merely hold inside the attention module.
    block = TransformerBlock(cfg).eval()
    torch.manual_seed(0)
    x = torch.randn(1, T, D)
    x2 = x.clone()
    x2[:, T // 2 :] = torch.randn(1, T - T // 2, D)
    with torch.no_grad():
        assert torch.equal(block(x)[:, : T // 2], block(x2)[:, : T // 2])


def test_block_builds_for_every_config_combination(cfg: ModelConfig, x: torch.Tensor) -> None:
    for norm in ("layernorm", "rmsnorm"):
        for act in ("gelu", "swiglu"):
            for n_kv in (None, 2):
                c = replace(cfg, norm=norm, act=act, n_kv_heads=n_kv)
                assert TransformerBlock(c)(x).shape == x.shape


def test_block_parameter_count_matches_the_config_prediction(cfg: ModelConfig) -> None:
    """config.n_params() validates the full model later, so its per-block term must be right."""
    actual = sum(p.numel() for p in TransformerBlock(cfg).parameters())
    predicted = (
        cfg.n_params(tied=True)
        - cfg.vocab_size * cfg.emb_dim
        - cfg.context_length * cfg.emb_dim
        - 2 * cfg.emb_dim
    ) // cfg.n_layers
    assert actual == predicted


class _NoResidual(TransformerBlock):
    """Pre-norm, but the residual additions removed."""

    def forward(self, x, rope=None, cache=None):
        return self.ffn(self.norm2(self.attn(self.norm1(x))))


class _NoResidualNoNorm(TransformerBlock):
    """Neither residuals nor normalization - a plain deep stack."""

    def forward(self, x, rope=None, cache=None):
        return self.ffn(self.attn(x))


def _layer_grads(cfg: ModelConfig, block_cls, depth: int = 12) -> list[float]:
    torch.manual_seed(0)
    stack = nn.ModuleList(block_cls(cfg) for _ in range(depth))
    h = torch.randn(1, T, D)
    for blk in stack:
        h = blk(h)
    h.sum().backward()
    return [float(blk.ffn.down.weight.grad.abs().mean()) for blk in stack]


def test_gradient_reaches_the_first_layer(cfg: ModelConfig) -> None:
    """A real block passes gradient down twelve layers without it shrinking materially."""
    g = _layer_grads(cfg, TransformerBlock)
    assert 0.1 < g[0] / g[-1] < 10, f"layer0={g[0]:.2e} layer11={g[-1]:.2e}"


def test_gradient_vanishes_without_both_residuals_and_normalization(cfg: ModelConfig) -> None:
    """Remove residuals *and* normalization and the gradient at layer 0 collapses.

    Measured, because the usual claim is too strong. Residuals and normalization are partly
    redundant for gradient flow: with LayerNorm still in place, removing the residuals barely
    changes the gradient at layer 0 (see the test below). It takes losing both.

    Compared against the real block rather than across depth within the crippled stack. Without
    normalization the forward signal shrinks at every layer, so the whole stack -- not only its
    bottom -- arrives at ~1e-15, and a within-model ratio reads as healthy while the model is in
    fact dead. Whether the decay is progressive or uniform turns on details like whether the
    linear layers carry biases; the collapse against a working block does not.
    """
    healthy = _layer_grads(cfg, TransformerBlock)
    crippled = _layer_grads(cfg, _NoResidualNoNorm)
    ratio = crippled[0] / healthy[0]
    assert ratio < 1e-6, f"crippled={crippled[0]:.2e} healthy={healthy[0]:.2e}"


def test_normalization_alone_also_preserves_gradient(cfg: ModelConfig) -> None:
    """The honest counterpart: LayerNorm on its own keeps the scale stable across depth.

    Worth pinning down, because it contradicts the common shorthand that residual connections
    are what prevent vanishing gradients in a transformer.
    """
    g = _layer_grads(cfg, _NoResidual)
    assert 0.1 < g[0] / g[-1] < 10, f"layer0={g[0]:.2e} layer11={g[-1]:.2e}"


def test_postnorm_block_runs_and_differs_from_prenorm(cfg: ModelConfig, x: torch.Tensor) -> None:
    pre, post = TransformerBlock(cfg).eval(), PostNormBlock(cfg).eval()
    with torch.no_grad():
        assert post(x).shape == x.shape
        assert not torch.allclose(pre(x), post(x), atol=1e-3)


def test_dropout_disabled_in_eval_mode(cfg: ModelConfig, x: torch.Tensor) -> None:
    block = TransformerBlock(replace(cfg, dropout=0.5)).eval()
    with torch.no_grad():
        assert torch.equal(block(x), block(x))
