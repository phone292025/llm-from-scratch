"""Tests for note 7 - rotary position embeddings, and the modern config as a whole.

The load-bearing test is `test_the_score_depends_only_on_the_distance`. Everything else about
RoPE -- the norms, the frequencies, the parameter count -- can be right while that one is wrong,
and if it is wrong then RoPE is an expensive way of scrambling q and k. Nothing downstream would
say so: the model still trains, the loss still falls.

The second thing this file guards is the failure the old `NotImplementedError` in `GPTModel`
existed to prevent -- a model with *no* positional information, which is indistinguishable from a
working one except that it cannot tell "dog bites man" from "man bites dog".
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from llm.config import MODERN, SMALL, ModelConfig
from llm.generate import generate
from llm.kv_cache import KVCache
from llm.model import GPTModel
from llm.rope import RopeCache, apply_rope

B, T = 2, 12


@pytest.fixture
def cfg() -> ModelConfig:
    """MODERN's components at a size that runs fast. Dims come from the preset, never literals."""
    return replace(
        MODERN,
        vocab_size=256,
        context_length=64,
        emb_dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        dropout=0.0,
    )


@pytest.fixture
def rope(cfg: ModelConfig) -> RopeCache:
    return RopeCache(cfg.head_dim, cfg.context_length, cfg.rope_base)


@pytest.fixture
def qk(cfg: ModelConfig) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    shape = (B, cfg.n_heads, T, cfg.head_dim)
    return torch.randn(shape), torch.randn(shape)


def scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return q @ k.transpose(-2, -1)


# --- the rotation itself ------------------------------------------------------------------------


def test_rope_preserves_norms(rope: RopeCache, qk) -> None:
    """A rotation turns a vector; it must not lengthen or shorten it.

    If this fails, RoPE is rescaling q and k by position, which changes how peaked the softmax is
    at different places in the sequence -- a confound that has nothing to do with order.
    """
    q, _ = qk
    cos, sin = rope.slice(0, T)
    assert torch.allclose(apply_rope(q, cos, sin).norm(dim=-1), q.norm(dim=-1), atol=1e-5)


def test_position_zero_is_the_identity(rope: RopeCache, qk) -> None:
    # The angle at position 0 is 0, so the first token is left exactly as it was.
    q, _ = qk
    cos, sin = rope.slice(0, T)
    assert torch.allclose(apply_rope(q, cos, sin)[:, :, 0], q[:, :, 0], atol=1e-6)


def test_the_score_depends_only_on_the_distance(rope: RopeCache, qk) -> None:
    """The central property. Slide the whole window down the sequence: every score is unchanged.

    q at position m and k at position n give a score that is a function of (m - n) and the two
    vectors -- never of m and n themselves. That is the entire reason to rotate rather than add.
    """
    q, k = qk
    at_zero = scores(*(apply_rope(t, *rope.slice(0, T)) for t in (q, k)))
    for offset in (1, 7, rope.max_len - T):
        shifted = scores(*(apply_rope(t, *rope.slice(offset, T)) for t in (q, k)))
        assert torch.allclose(at_zero, shifted, atol=1e-5), f"offset {offset} changed the scores"


def test_rope_actually_encodes_distance(rope: RopeCache, qk) -> None:
    """The counterpart: it must not be *invariant to everything*.

    Multiplying q and k by the identity would pass the test above. Moving one of them relative to
    the other has to change the score, or no order information is being carried at all.
    """
    q, k = qk
    cos, sin = rope.slice(0, T)
    rotated = scores(apply_rope(q, cos, sin), apply_rope(k, cos, sin))
    assert not torch.allclose(rotated, scores(q, k), atol=1e-3)
    # Same pair of vectors, two different separations -> two different scores.
    near = (apply_rope(q, *rope.slice(1, T))[:, :, 0] * apply_rope(k, *rope.slice(0, T))[:, :, 0])
    far = (apply_rope(q, *rope.slice(9, T))[:, :, 0] * apply_rope(k, *rope.slice(0, T))[:, :, 0])
    assert not torch.allclose(near.sum(-1), far.sum(-1), atol=1e-3)


def test_frequencies_span_fast_to_slow(cfg: ModelConfig, rope: RopeCache) -> None:
    """First pair rotates once per token, last pair barely moves over the whole context.

    That spread is what lets one set of angles carry both "the previous word" and "far back".
    """
    # atan2, not arccos: near an angle of 0 the cosine is 1 - x^2/2, and float32 cannot
    # resolve the difference. The slowest pair turns by 3e-4 radians per token.
    radians_per_token = torch.atan2(rope.sin[1], rope.cos[1])
    assert math.isclose(radians_per_token[0].item(), 1.0, rel_tol=1e-6)
    slowest = cfg.rope_base ** (-(cfg.head_dim - 2) / cfg.head_dim)
    assert math.isclose(radians_per_token[-1].item(), slowest, rel_tol=1e-5)
    # Over the whole context the fast pair wraps many times; the slow one barely moves.
    assert radians_per_token[0] * cfg.context_length > 2 * math.pi
    assert radians_per_token[-1] * cfg.context_length < 0.1


def test_rope_has_no_parameters(rope: RopeCache) -> None:
    # The whole point: the position table is gone, not merely smaller.
    assert list(rope.parameters()) == []
    assert rope.state_dict() == {}  # buffers are non-persistent, so checkpoints stay clean


def test_odd_head_dim_is_refused() -> None:
    # Coordinates are rotated in pairs; an odd dimension leaves one with no partner.
    with pytest.raises(ValueError, match="even"):
        RopeCache(head_dim=7, max_len=16)


def test_reading_past_the_table_is_refused(rope: RopeCache) -> None:
    with pytest.raises(ValueError, match="beyond"):
        rope.slice(rope.max_len - 2, 4)


# --- inside the model ---------------------------------------------------------------------------


def test_the_modern_config_builds_and_runs(cfg: ModelConfig) -> None:
    """This raised NotImplementedError until rope.py existed."""
    model = GPTModel(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    assert model(idx).shape == (B, T, cfg.vocab_size)


def test_rope_replaces_the_position_table_rather_than_joining_it(cfg: ModelConfig) -> None:
    model = GPTModel(cfg)
    assert model.pos_emb is None and model.rope is not None
    assert "position embedding" not in model.parameter_breakdown()
    # And the config's arithmetic agrees with the model that was actually built.
    assert model.num_parameters() == cfg.n_params()


def test_the_rope_table_follows_the_model_onto_its_device(cfg: ModelConfig) -> None:
    """Registered as a buffer, so .to() moves it. A plain tensor attribute would not move, and
    the first CUDA forward pass would die on a device mismatch."""
    model = GPTModel(cfg)
    assert model.rope.cos.device == next(model.parameters()).device


def _shuffled_context(idx: torch.Tensor) -> torch.Tensor:
    """Permute everything except the final token, which stays put.

    The last position is the one that predicts, and it attends over the same *set* of keys either
    way -- so any change in its output came from order, not from content.
    """
    return torch.cat([idx[:, :-1][:, torch.randperm(idx.shape[1] - 1)], idx[:, -1:]], dim=1)


@pytest.mark.parametrize("pos", ["rope", "learned"])
def test_the_model_can_tell_word_order(cfg: ModelConfig, pos: str) -> None:
    """The bug the old `NotImplementedError` in GPTModel existed to prevent.

    A model that cannot see order still trains, still generates, and is wrong about everything
    order carries. Nothing downstream reports it.
    """
    torch.manual_seed(0)
    idx = torch.randint(0, cfg.vocab_size, (1, T))
    shuffled = _shuffled_context(idx)

    torch.manual_seed(0)
    model = GPTModel(replace(cfg, pos=pos)).eval()
    with torch.no_grad():
        assert not torch.allclose(model(idx)[0, -1], model(shuffled)[0, -1], atol=1e-4)


def test_one_layer_without_positions_is_genuinely_order_blind(cfg: ModelConfig) -> None:
    """The control, and a correction to what note 7 originally claimed.

    A *single* attention layer with no positional signal is a set operation: softmax over the
    same keys in a different order gives the same answer, to floating-point noise. This is what
    makes the test above meaningful -- it shows the difference RoPE creates is real and not an
    artifact of shuffling.
    """
    torch.manual_seed(0)
    idx = torch.randint(0, cfg.vocab_size, (1, T))
    torch.manual_seed(0)
    model = GPTModel(replace(cfg, pos="none", n_layers=1)).eval()
    with torch.no_grad():
        assert torch.allclose(model(idx)[0, -1], model(_shuffled_context(idx))[0, -1], atol=1e-5)


def test_causal_masking_leaks_position_once_there_are_two_layers(cfg: ModelConfig) -> None:
    """Stack a second layer and "no positional information" stops being true.

    Layer 1's output at position j summarizes the prefix 0..j, so its *length* is visible in the
    result; layer 2 reads those summaries and can tell how far along each one sits. The causal
    mask is itself a positional signal. (This is the "NoPE" result -- decoder-only models train
    without any position encoding, just less well.)

    So `pos="none"` is not the harmless null baseline it looks like, and a broken RoPE would not
    produce a model that simply ignores order -- it would produce one with a weak, implicit sense
    of it. That is much harder to notice, which is the whole reason this file exists.
    """
    torch.manual_seed(0)
    idx = torch.randint(0, cfg.vocab_size, (1, T))
    torch.manual_seed(0)
    model = GPTModel(replace(cfg, pos="none", n_layers=2)).eval()
    with torch.no_grad():
        assert not torch.allclose(model(idx)[0, -1], model(_shuffled_context(idx))[0, -1],
                                  atol=1e-4)


# --- rope and the KV cache together ---------------------------------------------------------


@pytest.mark.parametrize("attn", ["sdpa", "manual"])
def test_cached_decoding_matches_uncached_under_rope(cfg: ModelConfig, attn: str) -> None:
    """Rope plus a cache is where the offset bug lives.

    Decoding feeds one token at a time, so T == 1 every step. Rotating it by position 0 instead
    of by its absolute position gives a model that believes every new token is the start of the
    sequence -- and it still produces fluent text, because the *cached* keys remain correctly
    rotated relative to each other.
    """
    torch.manual_seed(0)
    model = GPTModel(replace(cfg, attn=attn)).eval()
    prompt = torch.randint(0, cfg.vocab_size, (B, 5))
    plain = generate(model, prompt, 15, temperature=0.0, use_cache=False)
    cached = generate(model, prompt, 15, temperature=0.0, use_cache=True)
    assert torch.equal(plain, cached)


def test_chunked_prefill_matches_one_pass_under_rope(cfg: ModelConfig) -> None:
    """The middle case: several new tokens on top of a cached prefix, so the offset is non-zero
    and T > 1 at the same time."""
    torch.manual_seed(0)
    model = GPTModel(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (B, 11))
    with torch.no_grad():
        whole = model(idx)
        cache = KVCache(cfg, batch_size=B, max_len=cfg.context_length)
        pieces = [model(idx[:, :4], cache=cache), model(idx[:, 4:9], cache=cache)]
        pieces += [model(idx[:, i : i + 1], cache=cache) for i in (9, 10)]
    assert torch.allclose(whole, torch.cat(pieces, dim=1), atol=1e-5)


# --- the whole modern stack ---------------------------------------------------------------------


@pytest.mark.parametrize("norm", ["layernorm", "rmsnorm"])
@pytest.mark.parametrize("pos", ["learned", "rope", "none"])
@pytest.mark.parametrize("act", ["gelu", "swiglu"])
@pytest.mark.parametrize("n_kv_heads", [None, 2])
def test_every_combination_of_components_builds_and_learns(
    cfg: ModelConfig, norm: str, pos: str, act: str, n_kv_heads: int | None
) -> None:
    """Twenty-four configurations. They are meant to be independent switches, and the only way to
    know they still are is to build all of them.

    'Learns' rather than 'runs': a forward pass proves the shapes line up, one optimizer step
    proves gradients reach every parameter that was created.
    """
    torch.manual_seed(0)
    model = GPTModel(replace(cfg, norm=norm, pos=pos, act=act, n_kv_heads=n_kv_heads))
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    # Targets independent of the inputs: with tied weights, asking the model to predict the
    # token it is reading starts below ln(V) before any training.
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    _, loss = model(idx, targets=targets)
    assert math.isclose(loss.item(), math.log(cfg.vocab_size), rel_tol=0.15)

    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.any()]
    assert not missing, f"no gradient reached {missing}"


def test_swiglu_pays_for_its_third_matrix() -> None:
    """Three matrices instead of two, so the hidden width shrinks to 8d/3 to pay for it.

    Not exact: 8d/3 at d=512 is 1365.33, and the width is rounded to a multiple of 64 for the
    hardware, which lands at 1344 and leaves SwiGLU 1.6% cheaper than the GELU MLP. Close enough
    that the comparison is fair; not close enough to claim 1%, which note 7 originally did.
    """
    d = SMALL.emb_dim
    gelu = 2 * d * SMALL.ffn_hidden_dim
    swiglu_cfg = replace(SMALL, act="swiglu")
    swiglu = 3 * d * swiglu_cfg.ffn_hidden_dim
    assert abs(swiglu - gelu) / gelu < 0.02, f"{swiglu:,} vs {gelu:,}"


def test_only_gqa_changes_the_parameter_budget() -> None:
    """MODERN flips four switches. Three are near-neutral by design; GQA is not, and the whole
    A/B is misread if that is not known up front.

    Each saving is checked against the closed form, and then against the sum -- which is what
    proves the four switches are independent rather than interacting.
    """
    base = SMALL
    hd, layers, d = base.head_dim, base.n_layers, base.emb_dim

    # RoPE: the position table, exactly, and nothing else.
    rope = base.n_params() - replace(base, pos="rope").n_params()
    assert rope == base.context_length * d

    # RMSNorm: one bias vector per norm -- two per block plus the final one.
    rms = base.n_params() - replace(base, norm="rmsnorm").n_params()
    assert rms == (2 * layers + 1) * d

    # SwiGLU: only the rounding residue, 0.3% of the model.
    swiglu = base.n_params() - replace(base, act="swiglu").n_params()
    assert 0 < swiglu / base.n_params() < 0.005

    # GQA: 12 of 16 KV heads removed, twice (k and v), in every layer. 4% of the model, and the
    # reason MODERN is not the same size as SMALL.
    gqa = base.n_params() - replace(base, n_kv_heads=MODERN.n_kv_heads).n_params()
    assert gqa == 2 * (base.n_heads - MODERN.n_kv_heads) * hd * d * layers
    assert 0.03 < gqa / base.n_params() < 0.05

    assert base.n_params() - MODERN.n_params() == rope + rms + swiglu + gqa
