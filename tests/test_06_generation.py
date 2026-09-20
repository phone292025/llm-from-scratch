"""Tests for note 6 - sampling, and the KV cache.

The cache is the most dangerous code in the repo. Every other bug announces itself: a shape is
wrong, a loss does not fall, the GPT-2 oracle writes noise. A cache bug writes *fluent English*
that is subtly not what the model would have said, because the only thing that changed is which
keys a query was allowed to see. Nothing about the output looks wrong.

So the load-bearing test here is parity: for the same seed, generation with the cache must produce
exactly the same token ids as generation without it, on every attention backend and both MHA and
GQA. Everything else in this file is supporting evidence.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from llm.config import SMALL, ModelConfig
from llm.generate import (
    apply_temperature,
    generate,
    generate_text,
    sample_next,
    top_k_filter,
    top_p_filter,
)
from llm.kv_cache import KVCache
from llm.model import GPTModel

B, PROMPT_LEN, NEW = 2, 5, 12


@pytest.fixture
def cfg() -> ModelConfig:
    # Small enough to be quick, still wide enough that GQA groups are meaningful.
    return replace(
        SMALL,
        vocab_size=256,
        context_length=32,
        emb_dim=64,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
    )


@pytest.fixture
def model(cfg: ModelConfig) -> GPTModel:
    torch.manual_seed(0)
    return GPTModel(cfg).eval()


@pytest.fixture
def prompt(cfg: ModelConfig) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, cfg.vocab_size, (B, PROMPT_LEN))


# --- the parity that matters ------------------------------------------------------------------


@pytest.mark.parametrize("attn", ["manual", "sdpa"])
@pytest.mark.parametrize("n_kv_heads", [None, 2, 1])
def test_cached_generation_is_identical_to_uncached(
    cfg: ModelConfig, prompt: torch.Tensor, attn: str, n_kv_heads: int | None
) -> None:
    """Greedy, so any divergence is a real difference in the logits and not sampling noise."""
    torch.manual_seed(0)
    m = GPTModel(replace(cfg, attn=attn, n_kv_heads=n_kv_heads)).eval()
    without = generate(m, prompt, NEW, temperature=0.0, use_cache=False)
    with_cache = generate(m, prompt, NEW, temperature=0.0, use_cache=True)
    assert torch.equal(without, with_cache)


@pytest.mark.parametrize("attn", ["manual", "sdpa"])
def test_cached_sampling_matches_uncached_for_the_same_seed(
    cfg: ModelConfig, prompt: torch.Tensor, attn: str
) -> None:
    torch.manual_seed(0)
    m = GPTModel(replace(cfg, attn=attn)).eval()
    a = generate(m, prompt, NEW, temperature=1.0, top_k=20, seed=7, use_cache=False)
    b = generate(m, prompt, NEW, temperature=1.0, top_k=20, seed=7, use_cache=True)
    assert torch.equal(a, b)


@pytest.mark.parametrize("attn", ["manual", "sdpa"])
@pytest.mark.parametrize("n_kv_heads", [None, 2])
def test_chunked_prefill_matches_one_full_forward(
    cfg: ModelConfig, attn: str, n_kv_heads: int | None
) -> None:
    """Three routes through cached attention, each checked against the plain forward pass.

    A first chunk (empty cache, square causal mask), a second multi-token chunk (a prefix is
    cached, so the mask is the triangle shifted right), then single tokens (no mask at all).
    generate() only ever uses the first and last, so without this the shifted mask would be
    exercised by nothing.
    """
    torch.manual_seed(0)
    m = GPTModel(replace(cfg, attn=attn, n_kv_heads=n_kv_heads)).eval()
    ids = torch.randint(0, cfg.vocab_size, (B, 11))
    with torch.no_grad():
        full = m(ids)
        cache = KVCache(m.cfg, batch_size=B, max_len=cfg.context_length)
        pieces = [m(ids[:, :4], cache=cache), m(ids[:, 4:9], cache=cache)]
        pieces += [m(ids[:, i : i + 1], cache=cache) for i in range(9, 11)]
    assert torch.allclose(torch.cat(pieces, dim=1), full, atol=1e-5)


def test_cache_survives_running_past_the_context_window(
    cfg: ModelConfig, model: GPTModel, prompt: torch.Tensor
) -> None:
    """Learned positions run out of rows, so the cache is abandoned and cropping takes over.

    The handover is the interesting part: it has to produce the same tokens the uncached path
    would have, not merely avoid crashing.
    """
    n = cfg.context_length * 2
    cached = generate(model, prompt, n, temperature=0.0, use_cache=True)
    plain = generate(model, prompt, n, temperature=0.0, use_cache=False)
    assert cached.shape == (B, PROMPT_LEN + n)
    assert torch.equal(cached, plain)


def test_prompt_longer_than_the_context_is_cropped(cfg: ModelConfig, model: GPTModel) -> None:
    long_prompt = torch.randint(0, cfg.vocab_size, (1, cfg.context_length + 7))
    out = generate(model, long_prompt, 3, temperature=0.0)
    assert out.shape == (1, cfg.context_length + 10)
    assert torch.equal(out[:, : cfg.context_length + 7], long_prompt)


# --- the cache itself -------------------------------------------------------------------------


def test_cache_stores_narrow_kv_under_gqa(cfg: ModelConfig) -> None:
    """Caching the *expanded* keys would throw away the only reason GQA exists at inference."""
    gqa = replace(cfg, n_kv_heads=1)
    cache = KVCache(gqa, batch_size=B, max_len=cfg.context_length)
    assert cache.k[0].shape == (B, gqa.kv_heads, cfg.context_length, gqa.head_dim)
    assert cache.k[0].shape[1] == 1 < cfg.n_heads


def test_position_advances_once_per_forward_not_once_per_layer(
    cfg: ModelConfig, model: GPTModel, prompt: torch.Tensor
) -> None:
    """Every layer writes at the same position.

    Advancing inside the layer loop would multiply the cursor by n_layers, and the model would
    then read keys that were never written.
    """
    cache = KVCache(cfg, batch_size=B, max_len=cfg.context_length)
    with torch.no_grad():
        model(prompt, cache=cache)
    assert cache.pos == PROMPT_LEN
    with torch.no_grad():
        model(prompt[:, :1], cache=cache)
    assert cache.pos == PROMPT_LEN + 1


def test_cache_returns_the_whole_history_not_just_the_new_step(cfg: ModelConfig) -> None:
    cache = KVCache(cfg, batch_size=1, max_len=cfg.context_length)
    step = torch.randn(1, cfg.kv_heads, 3, cfg.head_dim)
    k, _ = cache.update(0, step, step)
    assert k.shape[2] == 3
    cache.advance(3)
    k, _ = cache.update(0, step[:, :, :1], step[:, :, :1])
    assert k.shape[2] == 4
    assert torch.equal(k[:, :, :3], step)


def test_cache_refuses_to_overflow(cfg: ModelConfig) -> None:
    cache = KVCache(cfg, batch_size=1, max_len=4)
    step = torch.randn(1, cfg.kv_heads, 5, cfg.head_dim)
    with pytest.raises(ValueError, match="larger max_len"):
        cache.update(0, step, step)


def test_reset_makes_stale_values_unreachable(cfg: ModelConfig) -> None:
    cache = KVCache(cfg, batch_size=1, max_len=cfg.context_length)
    old = torch.randn(1, cfg.kv_heads, 4, cfg.head_dim)
    cache.update(0, old, old)
    cache.advance(4)
    cache.reset()
    assert cache.pos == 0
    new = torch.randn(1, cfg.kv_heads, 1, cfg.head_dim)
    k, _ = cache.update(0, new, new)
    assert k.shape[2] == 1
    assert torch.equal(k, new)


def test_allocated_memory_matches_the_prediction(cfg: ModelConfig) -> None:
    cache = KVCache(cfg, batch_size=B, max_len=cfg.context_length, dtype=torch.float32)
    predicted = KVCache.predicted_bytes(cfg, B, cfg.context_length, bytes_per_element=4)
    assert cache.memory_bytes() == predicted


def test_gqa_shrinks_the_cache_by_the_group_size(cfg: ModelConfig) -> None:
    mha = KVCache.predicted_bytes(cfg, 1, cfg.context_length)
    mqa = KVCache.predicted_bytes(replace(cfg, n_kv_heads=1), 1, cfg.context_length)
    assert mha == mqa * cfg.n_heads


# --- sampling ---------------------------------------------------------------------------------


def test_temperature_zero_is_greedy() -> None:
    logits = torch.tensor([[1.0, 5.0, 2.0], [9.0, 0.0, 3.0]])
    assert torch.equal(sample_next(logits, temperature=0.0), torch.tensor([[1], [0]]))


def test_temperature_leaves_the_ranking_alone() -> None:
    logits = torch.randn(4, 20)
    for t in (0.5, 1.0, 2.0):
        assert torch.equal(apply_temperature(logits, t).argsort(dim=-1), logits.argsort(dim=-1))


def test_low_temperature_concentrates_and_high_temperature_flattens() -> None:
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    cold = apply_temperature(logits, 0.2).softmax(-1).max()
    hot = apply_temperature(logits, 5.0).softmax(-1).max()
    assert cold > logits.softmax(-1).max() > hot


def test_top_k_keeps_exactly_k_candidates() -> None:
    logits = torch.randn(3, 50)
    kept = top_k_filter(logits, 5) != float("-inf")
    assert torch.equal(kept.sum(-1), torch.full((3,), 5))


def test_top_k_keeps_the_largest_ones() -> None:
    logits = torch.randn(1, 30)
    kept = (top_k_filter(logits, 4)[0] != float("-inf")).nonzero().flatten()
    assert set(kept.tolist()) == set(logits[0].topk(4).indices.tolist())


def test_top_k_is_a_no_op_at_or_above_the_vocabulary_size() -> None:
    logits = torch.randn(2, 10)
    assert torch.equal(top_k_filter(logits, 10), logits)
    assert torch.equal(top_k_filter(logits, 0), logits)


def test_top_p_keeps_the_smallest_set_that_passes_the_threshold() -> None:
    """Checked against an explicit walk down the sorted probabilities.

    The implementation filters in sorted order and scatters the mask back into vocabulary order,
    which is exactly the step where an off-by-one silently keeps the wrong tokens.
    """
    torch.manual_seed(1)
    logits = torch.randn(4, 24)
    p = 0.6
    kept = top_p_filter(logits.clone(), p) != float("-inf")
    probs = logits.softmax(-1)

    for row in range(logits.shape[0]):
        expected, total = set(), 0.0
        for i in probs[row].argsort(descending=True).tolist():
            expected.add(i)
            total += float(probs[row][i])
            if total > p:
                break
        assert set(kept[row].nonzero().flatten().tolist()) == expected


def test_top_p_nucleus_is_never_empty() -> None:
    """One token above p on its own must still survive, or multinomial has nothing to draw."""
    logits = torch.tensor([[20.0, 0.0, 0.0, 0.0]])
    kept = top_p_filter(logits, 0.5) != float("-inf")
    assert kept.sum() == 1
    assert bool(kept[0, 0])


def test_top_p_is_a_no_op_outside_the_open_unit_interval() -> None:
    logits = torch.randn(2, 8)
    for p in (0.0, 1.0):
        assert torch.equal(top_p_filter(logits, p), logits)


def test_sampling_is_reproducible_from_a_seed(model: GPTModel, prompt: torch.Tensor) -> None:
    a = generate(model, prompt, NEW, temperature=1.2, seed=3)
    b = generate(model, prompt, NEW, temperature=1.2, seed=3)
    c = generate(model, prompt, NEW, temperature=1.2, seed=4)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_filters_compose_without_emptying_the_distribution(
    model: GPTModel, prompt: torch.Tensor
) -> None:
    out = generate(model, prompt, NEW, temperature=0.8, top_k=10, top_p=0.9, seed=0)
    assert out.shape == (B, PROMPT_LEN + NEW)


# --- the loop ---------------------------------------------------------------------------------


def test_generation_extends_by_exactly_max_new_tokens(
    model: GPTModel, prompt: torch.Tensor
) -> None:
    assert generate(model, prompt, NEW, temperature=0.0).shape == (B, PROMPT_LEN + NEW)


def test_generation_preserves_the_prompt(model: GPTModel, prompt: torch.Tensor) -> None:
    out = generate(model, prompt, NEW, temperature=0.0)
    assert torch.equal(out[:, :PROMPT_LEN], prompt)


def test_generation_restores_training_mode(cfg: ModelConfig, prompt: torch.Tensor) -> None:
    m = GPTModel(cfg).train()
    generate(m, prompt, 2, temperature=0.0)
    assert m.training


def test_everything_after_eos_is_eos(model: GPTModel, prompt: torch.Tensor) -> None:
    """Rows finish at different steps, so a finished row keeps emitting eos to stay rectangular.

    The eos id is taken from the model's own unconditioned output, so it is guaranteed to appear.
    """
    natural = generate(model, prompt, NEW, temperature=0.0)
    eos = int(natural[0, PROMPT_LEN])

    out = generate(model, prompt, NEW, temperature=0.0, eos_id=eos)
    tail = out[0, PROMPT_LEN:]
    first = int((tail == eos).nonzero()[0])
    assert bool((tail[first:] == eos).all())


def test_generate_text_round_trips_through_the_tokenizer(cfg: ModelConfig) -> None:
    from llm.tokenizer import BPETokenizer

    tok = BPETokenizer()
    torch.manual_seed(0)
    m = GPTModel(replace(cfg, vocab_size=tok.vocab_size)).eval()
    text = generate_text(m, tok, "The cat", max_new_tokens=5, temperature=0.0)
    assert text.startswith("The cat")


def test_empty_prompt_samples_unconditionally_from_eot(cfg: ModelConfig) -> None:
    """A model needs something to condition on; <|endoftext|> is what precedes every document."""
    from llm.tokenizer import BPETokenizer

    tok = BPETokenizer()
    torch.manual_seed(0)
    m = GPTModel(replace(cfg, vocab_size=tok.vocab_size)).eval()
    assert generate_text(m, tok, "", max_new_tokens=3, temperature=0.0, eos_id=None)


def test_model_refuses_an_empty_batch(model: GPTModel) -> None:
    with pytest.raises(ValueError, match="at least one token"):
        model(torch.zeros(1, 0, dtype=torch.long))


# --- streaming -------------------------------------------------------------------------------


def test_iter_generate_yields_one_step_at_a_time(cfg: ModelConfig) -> None:
    from llm.generate import iter_generate

    torch.manual_seed(0)
    m = GPTModel(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    steps = list(iter_generate(m, idx, 7, temperature=0.0))
    assert len(steps) == 7
    for next_id, logits in steps:
        assert next_id.shape == (1, 1)
        assert logits.shape == (1, cfg.vocab_size)


def test_generate_equals_accumulating_the_stream(cfg: ModelConfig) -> None:
    """generate() is now a thin wrapper over iter_generate, so the two cannot drift apart."""
    from llm.generate import iter_generate

    torch.manual_seed(0)
    m = GPTModel(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 5))

    whole = generate(m, idx, 12, temperature=0.9, top_p=0.9, seed=3)
    streamed = idx
    for next_id, _ in iter_generate(m, idx, 12, temperature=0.9, top_p=0.9, seed=3):
        streamed = torch.cat([streamed, next_id], dim=1)
    assert torch.equal(whole, streamed)


def test_streamed_logits_are_before_filtering(cfg: ModelConfig) -> None:
    """The yielded scores show what the model believed, not what top-p left on the table.

    Filtered logits would be -inf almost everywhere, which would make the chat's "what it was
    weighing" view show a handful of survivors rather than the real distribution.
    """
    from llm.generate import iter_generate

    torch.manual_seed(0)
    m = GPTModel(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    _, logits = next(iter(iter_generate(m, idx, 1, temperature=0.8, top_p=0.5)))
    assert torch.isfinite(logits).all()
