"""Tests for note 2 - attention.

Two tests here matter more than the rest.

test_causality_no_information_leak: a leak makes training loss look excellent and generation
produce garbage, because at generation time the future genuinely does not exist. Nothing else
catches it.

test_manual_matches_sdpa: the manual implementation checked against PyTorch's own. This is proof
of correctness rather than plausibility.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from llm.attention import (
    CausalSelfAttention,
    MultiHeadWrapper,
    NaiveAttention,
    SingleHeadAttention,
    causal_mask,
)
from llm.config import SMALL, ModelConfig

B, T = 2, 12


@pytest.fixture
def cfg() -> ModelConfig:
    # dropout=0 and manual attention so results are deterministic and comparable.
    return replace(SMALL, context_length=32, attn="manual", dropout=0.0)


@pytest.fixture
def x(cfg: ModelConfig) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, T, cfg.emb_dim)


# --- shapes ---------------------------------------------------------------------------------


def test_output_shape_matches_input(cfg: ModelConfig, x: torch.Tensor) -> None:
    # Shape in == shape out is what lets blocks stack to any depth.
    assert CausalSelfAttention(cfg)(x).shape == x.shape


def test_head_split_shapes(cfg: ModelConfig, x: torch.Tensor) -> None:
    q, k, v = CausalSelfAttention(cfg)._project(x)
    assert q.shape == (B, cfg.n_heads, T, cfg.head_dim)
    assert k.shape == v.shape == (B, cfg.kv_heads, T, cfg.head_dim)


def test_shorter_sequence_than_context_length_works(cfg: ModelConfig) -> None:
    attn = CausalSelfAttention(cfg)
    for t in (1, 5, cfg.context_length):
        assert attn(torch.randn(1, t, cfg.emb_dim)).shape == (1, t, cfg.emb_dim)


# --- properties -----------------------------------------------------------------------------


def test_attention_rows_sum_to_one(cfg: ModelConfig, x: torch.Tensor) -> None:
    # Masking with -inf before softmax renormalizes over the visible positions. Zeroing after
    # softmax instead would leave rows summing to less than 1.
    w = CausalSelfAttention(cfg).attention_weights(x)
    assert torch.allclose(w.sum(-1), torch.ones(B, cfg.n_heads, T), atol=1e-6)


def test_no_weight_on_future_positions(cfg: ModelConfig, x: torch.Tensor) -> None:
    w = CausalSelfAttention(cfg).attention_weights(x)
    assert w.triu(1).abs().max() == 0.0


def test_first_position_attends_only_to_itself(cfg: ModelConfig, x: torch.Tensor) -> None:
    w = CausalSelfAttention(cfg).attention_weights(x)
    assert torch.allclose(w[:, :, 0, 0], torch.ones(B, cfg.n_heads), atol=1e-6)


@pytest.mark.parametrize("attn_impl", ["manual", "sdpa"])
def test_causality_no_information_leak(cfg: ModelConfig, attn_impl: str) -> None:
    """Perturb a future token; every earlier position must be bit-for-bit unchanged.

    The most important test in the repo.
    """
    attn = CausalSelfAttention(replace(cfg, attn=attn_impl)).eval()
    torch.manual_seed(0)
    x = torch.randn(1, T, cfg.emb_dim)

    cut = T // 2
    x_changed = x.clone()
    x_changed[:, cut:] = torch.randn(1, T - cut, cfg.emb_dim)

    with torch.no_grad():
        before, after = attn(x), attn(x_changed)

    assert torch.equal(before[:, :cut], after[:, :cut]), "future tokens leaked into the past"
    assert not torch.equal(before[:, cut:], after[:, cut:]), "the perturbation had no effect"


def test_causal_mask_shape_and_content() -> None:
    m = causal_mask(4)
    assert m.dtype is torch.bool
    assert m.tolist() == [
        [False, True, True, True],
        [False, False, True, True],
        [False, False, False, True],
        [False, False, False, False],
    ]


def test_mask_is_not_in_the_state_dict(cfg: ModelConfig) -> None:
    # Derived from config, so it must follow .to(device) but stay out of checkpoints.
    assert "mask" not in CausalSelfAttention(cfg).state_dict()


# --- parity ---------------------------------------------------------------------------------


def test_manual_matches_sdpa(cfg: ModelConfig, x: torch.Tensor) -> None:
    """The manual implementation against PyTorch's own. Proof, not plausibility."""
    manual = CausalSelfAttention(replace(cfg, attn="manual")).eval()
    sdpa = CausalSelfAttention(replace(cfg, attn="sdpa")).eval()
    sdpa.load_state_dict(manual.state_dict())

    with torch.no_grad():
        assert torch.allclose(manual(x), sdpa(x), atol=1e-5)


def test_batched_heads_match_the_naive_loop(cfg: ModelConfig, x: torch.Tensor) -> None:
    """The two reshapes in CausalSelfAttention, against an explicit loop over heads."""
    cfg = replace(cfg, qkv_bias=False)
    fused = CausalSelfAttention(cfg).eval()
    loop = MultiHeadWrapper(
        cfg.emb_dim, cfg.emb_dim, cfg.context_length, cfg.n_heads, qkv_bias=False
    ).eval()

    hd, H = cfg.head_dim, cfg.n_heads
    W_q, W_k, W_v = fused.W_qkv.weight.split([H * hd, H * hd, H * hd], dim=0)
    for h, head in enumerate(loop.heads):
        sl = slice(h * hd, (h + 1) * hd)
        head.W_q.weight.data.copy_(W_q[sl])
        head.W_k.weight.data.copy_(W_k[sl])
        head.W_v.weight.data.copy_(W_v[sl])
    loop.W_o.load_state_dict(fused.W_o.state_dict())

    with torch.no_grad():
        assert torch.allclose(fused(x), loop(x), atol=1e-5)


def test_gqa_with_full_kv_heads_is_plain_mha(cfg: ModelConfig, x: torch.Tensor) -> None:
    explicit = CausalSelfAttention(replace(cfg, n_kv_heads=cfg.n_heads)).eval()
    implicit = CausalSelfAttention(replace(cfg, n_kv_heads=None)).eval()
    implicit.load_state_dict(explicit.state_dict())
    with torch.no_grad():
        assert torch.equal(explicit(x), implicit(x))


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4, 8, 16])
def test_gqa_runs_and_shrinks_the_kv_projection(cfg: ModelConfig, x: torch.Tensor, n_kv_heads: int) -> None:
    attn = CausalSelfAttention(replace(cfg, n_kv_heads=n_kv_heads))
    assert attn(x).shape == x.shape
    expected = (cfg.n_heads + 2 * n_kv_heads) * cfg.head_dim
    assert attn.W_qkv.weight.shape[0] == expected


def test_gqa_shares_kv_heads_across_groups(cfg: ModelConfig, x: torch.Tensor) -> None:
    attn = CausalSelfAttention(replace(cfg, n_kv_heads=cfg.n_heads // 4))
    _, k, _ = attn._project(x)
    k_expanded, _ = attn._expand_kv(k, k)
    assert k_expanded.shape[1] == cfg.n_heads
    # Each key/value head is copied across exactly n_rep query heads, and the next group differs.
    n_rep = 4
    for h in range(1, n_rep):
        assert torch.equal(k_expanded[:, 0], k_expanded[:, h])
    assert not torch.equal(k_expanded[:, 0], k_expanded[:, n_rep])


def test_rejects_heads_that_do_not_divide_the_embedding(cfg: ModelConfig) -> None:
    with pytest.raises(ValueError, match="divide evenly"):
        replace(cfg, n_heads=5)


def test_rejects_kv_heads_that_do_not_group_evenly(cfg: ModelConfig) -> None:
    with pytest.raises(ValueError, match="multiple of"):
        replace(cfg, n_heads=16, n_kv_heads=5)


# --- the scaling is not cosmetic ---------------------------------------------------------------


def test_scaling_keeps_softmax_out_of_saturation() -> None:
    """Without the sqrt(head_dim) divisor, softmax saturates and its gradient vanishes.

    Measured as entropy: a saturated distribution puts nearly all mass on one position, so its
    entropy collapses toward zero and attention can no longer learn.
    """
    torch.manual_seed(0)
    head_dim, t = 256, 64
    q, k = torch.randn(1, 1, t, head_dim), torch.randn(1, 1, t, head_dim)

    raw = q @ k.transpose(-2, -1)
    scaled = raw / math.sqrt(head_dim)

    def entropy(scores: torch.Tensor) -> float:
        p = torch.softmax(scores, dim=-1)
        return float(-(p * torch.log(p + 1e-12)).sum(-1).mean())

    assert entropy(scaled) > 3.0 * entropy(raw)
    assert entropy(scaled) > 0.8 * math.log(t)  # close to uniform, so gradients flow


# --- the teaching ladder ------------------------------------------------------------------------


def test_naive_attention_has_no_parameters(x: torch.Tensor) -> None:
    naive = NaiveAttention()
    assert list(naive.parameters()) == []
    assert naive(x).shape == x.shape


def test_naive_attention_is_not_causal() -> None:
    """Stage 1 has no mask: the last token influences the first. Exactly why it is unusable.

    The input is deliberately small in both dimension and magnitude. With wide, unit-variance
    vectors the unscaled dot products are large enough to saturate softmax completely -- every
    position then attends almost entirely to itself and the leak becomes invisible in float32,
    which is the same failure that test_scaling_keeps_softmax_out_of_saturation is about.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 6, 8) * 0.1
    naive = NaiveAttention()

    x2 = x.clone()
    x2[:, -1] = torch.randn(1, 8) * 0.1

    assert not torch.equal(naive(x)[:, 0], naive(x2)[:, 0])


def test_single_head_returns_weights_on_request(cfg: ModelConfig, x: torch.Tensor) -> None:
    head = SingleHeadAttention(cfg.emb_dim, cfg.head_dim, cfg.context_length)
    ctx, w = head(x, return_weights=True)
    assert ctx.shape == (B, T, cfg.head_dim)
    assert w.shape == (B, T, T)


def test_dropout_is_disabled_in_eval_mode(cfg: ModelConfig, x: torch.Tensor) -> None:
    attn = CausalSelfAttention(replace(cfg, dropout=0.5)).eval()
    with torch.no_grad():
        assert torch.equal(attn(x), attn(x))
