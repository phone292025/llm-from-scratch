"""Tests for note 4 - the assembled model, and the GPT-2 oracle.

The parameter-count tests are unforgiving in a useful way: a missing bias, a dropped layer or a
wrong hidden width all change the number, so matching 124,439,808 exactly is strong evidence the
architecture is right.

The oracle tests load real OpenAI weights. They skip unless the checkpoint is already in the
HuggingFace cache, so CI does not pull 548 MB on every push. Run `python -m llm.gpt2_weights`
once locally to populate the cache and they become active.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from llm.config import GPT2_124M, PRESETS, SMALL, ModelConfig
from llm.gpt2_weights import gpt2_config, load_pretrained_gpt2
from llm.model import GPTModel

B, T = 2, 16


@pytest.fixture
def cfg() -> ModelConfig:
    return replace(SMALL, attn="manual", dropout=0.0)


@pytest.fixture
def model(cfg: ModelConfig) -> GPTModel:
    torch.manual_seed(0)
    return GPTModel(cfg).eval()


@pytest.fixture
def idx(cfg: ModelConfig) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, cfg.vocab_size, (B, T))


def _gpt2_is_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    result = try_to_load_from_cache("gpt2", "model.safetensors")
    return isinstance(result, str)


needs_gpt2 = pytest.mark.skipif(
    not _gpt2_is_cached(),
    reason="GPT-2 checkpoint not cached; run `python -m llm.gpt2_weights` to fetch it",
)


# --- shapes and wiring ------------------------------------------------------------------------


def test_logits_shape(model: GPTModel, idx: torch.Tensor, cfg: ModelConfig) -> None:
    assert model(idx).shape == (B, T, cfg.vocab_size)


def test_forward_with_targets_returns_loss(model: GPTModel, idx: torch.Tensor) -> None:
    logits, loss = model(idx, targets=idx)
    assert logits.shape[:2] == idx.shape
    assert loss.ndim == 0


def test_rejects_sequence_longer_than_context(model: GPTModel, cfg: ModelConfig) -> None:
    too_long = torch.zeros(1, cfg.context_length + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds context_length"):
        model(too_long)


def test_full_context_length_works(model: GPTModel, cfg: ModelConfig) -> None:
    ids = torch.zeros(1, cfg.context_length, dtype=torch.long)
    assert model(ids).shape == (1, cfg.context_length, cfg.vocab_size)


def test_eval_mode_is_deterministic(cfg: ModelConfig, idx: torch.Tensor) -> None:
    m = GPTModel(replace(cfg, dropout=0.5)).eval()
    with torch.no_grad():
        assert torch.equal(m(idx), m(idx))


@pytest.mark.parametrize("preset", list(PRESETS))
def test_every_preset_builds_and_runs(preset: str) -> None:
    cfg = replace(PRESETS[preset], n_layers=2, context_length=32)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    assert GPTModel(cfg)(ids).shape == (1, 8, cfg.vocab_size)


# --- parameter accounting ---------------------------------------------------------------------


def test_gpt2_parameter_count_is_exact() -> None:
    """124,439,808 tied, 163,037,184 untied. Any structural mistake moves this number.

    The book reports 163,009,536, which is 27,648 lower - exactly 12 layers x 2,304, the QKV
    bias. Its default config omits that bias; real GPT-2 has it, and the checkpoint will not
    load without it.
    """
    assert GPTModel(GPT2_124M).num_parameters() == 124_439_808
    assert GPTModel(replace(GPT2_124M, tie_weights=False)).num_parameters() == 163_037_184


def test_parameter_count_matches_config_prediction() -> None:
    # config.n_params() is used for planning before a model is built, so it must agree.
    for preset in PRESETS.values():
        cfg = replace(preset, n_layers=2)
        assert GPTModel(cfg).num_parameters() == cfg.n_params()


def test_tying_shares_storage_rather_than_copying(cfg: ModelConfig) -> None:
    tied = GPTModel(replace(cfg, tie_weights=True))
    assert tied.head.weight is tied.tok_emb.weight
    untied = GPTModel(replace(cfg, tie_weights=False))
    assert untied.head.weight is not untied.tok_emb.weight
    saved = untied.num_parameters() - tied.num_parameters()
    assert saved == cfg.vocab_size * cfg.emb_dim


def test_non_embedding_count_excludes_both_tables(model: GPTModel, cfg: ModelConfig) -> None:
    expected = model.num_parameters() - (cfg.vocab_size + cfg.context_length) * cfg.emb_dim
    assert model.num_parameters(non_embedding=True) == expected


def test_breakdown_sums_to_the_total(model: GPTModel) -> None:
    assert sum(model.parameter_breakdown().values()) == model.num_parameters()


# --- initialization ---------------------------------------------------------------------------


def test_loss_at_init_is_ln_vocab_size(cfg: ModelConfig) -> None:
    """An untrained model should be uniformly uncertain: ln(50257) = 10.82.

    Targets must be *independent* of the inputs. With tied weights the residual stream carries
    the token embedding into a head that is the token embedding, so an untrained model already
    favours predicting the current token - see the next test.
    """
    torch.manual_seed(0)
    m = GPTModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (4, 32))
    y = torch.randint(0, cfg.vocab_size, (4, 32))
    with torch.no_grad():
        _, loss = m(x, targets=y)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.5


def test_weight_tying_biases_an_untrained_model_toward_the_current_token(
    cfg: ModelConfig,
) -> None:
    """A real consequence of tying, and the reason the test above needs independent targets."""
    torch.manual_seed(0)
    x = torch.randint(0, cfg.vocab_size, (4, 32))
    uniform = 1.0 / cfg.vocab_size

    def self_probability(tie: bool) -> float:
        m = GPTModel(replace(cfg, tie_weights=tie)).eval()
        with torch.no_grad():
            probs = m(x).softmax(-1).gather(-1, x.unsqueeze(-1))
        return float(probs.mean())

    assert self_probability(tie=True) > 2 * uniform
    assert self_probability(tie=False) < 2 * uniform


def test_residual_projections_are_scaled_down(cfg: ModelConfig) -> None:
    """1/sqrt(2 * n_layers) on the projections that write into the residual stream."""
    m = GPTModel(cfg)
    expected = 0.02 / math.sqrt(2 * cfg.n_layers)
    got = m.blocks[0].attn.W_o.weight.std().item()
    assert abs(got - expected) / expected < 0.1


def test_biases_start_at_zero(cfg: ModelConfig) -> None:
    m = GPTModel(cfg)
    for name, p in m.named_parameters():
        if name.endswith(".bias"):
            assert torch.equal(p, torch.zeros_like(p)), name


# --- generation -------------------------------------------------------------------------------


def test_greedy_generation_extends_by_exactly_max_new_tokens(model: GPTModel) -> None:
    ids = torch.zeros(1, 4, dtype=torch.long)
    assert model.generate_greedy(ids, max_new_tokens=7).shape == (1, 11)


def test_greedy_generation_is_deterministic(model: GPTModel) -> None:
    ids = torch.zeros(1, 4, dtype=torch.long)
    assert torch.equal(
        model.generate_greedy(ids, max_new_tokens=5),
        model.generate_greedy(ids, max_new_tokens=5),
    )


def test_generation_preserves_the_prompt(model: GPTModel, cfg: ModelConfig) -> None:
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    assert torch.equal(model.generate_greedy(ids, max_new_tokens=3)[:, :5], ids)


def test_generation_crops_past_the_context_window(cfg: ModelConfig) -> None:
    """Absolute position embeddings run out of rows; cropping keeps generation running."""
    m = GPTModel(replace(cfg, context_length=16)).eval()
    ids = torch.zeros(1, 16, dtype=torch.long)
    assert m.generate_greedy(ids, max_new_tokens=4).shape == (1, 20)


def test_generation_restores_training_mode(cfg: ModelConfig) -> None:
    m = GPTModel(cfg).train()
    m.generate_greedy(torch.zeros(1, 2, dtype=torch.long), max_new_tokens=1)
    assert m.training


# --- the oracle -------------------------------------------------------------------------------


@needs_gpt2
def test_gpt2_weights_load_into_our_model() -> None:
    m = load_pretrained_gpt2("124M")
    assert m.num_parameters() == 124_439_808


@needs_gpt2
def test_gpt2_ranks_paris_for_the_capital_of_france() -> None:
    """Semantic proof that the weights landed in the right places.

    GPT-2 124M is small, so greedy decoding picks ' the' here; ' Paris' is 5th by probability.
    A single wrong transpose would put it nowhere near the top.
    """
    from llm.tokenizer import BPETokenizer

    tok = BPETokenizer()
    m = load_pretrained_gpt2("124M")
    ids = torch.tensor([tok.encode("The capital of France is")])
    with torch.no_grad():
        probs = m(ids)[0, -1].softmax(-1)

    paris = tok.encode(" Paris")[0]
    assert (probs > probs[paris]).sum().item() < 5, "' Paris' should be in the top 5"

    ids = torch.tensor([tok.encode("The capital of France is the city of")])
    with torch.no_grad():
        probs = m(ids)[0, -1].softmax(-1)
    assert (probs > probs[paris]).sum().item() < 3


@needs_gpt2
def test_gpt2_scores_real_english_far_better_than_random_weights() -> None:
    """The quantitative oracle: perplexity ~48 on English, against ~57,000 at random init."""
    from llm.tokenizer import BPETokenizer

    tok = BPETokenizer()
    text = (
        "The quick brown fox jumps over the lazy dog. Machine learning is a field of study "
        "in artificial intelligence concerned with the development of statistical algorithms."
    )
    ids = torch.tensor([tok.encode(text)])
    x, y = ids[:, :-1], ids[:, 1:]

    with torch.no_grad():
        _, trained = load_pretrained_gpt2("124M")(x, targets=y)
        _, random = GPTModel(gpt2_config("124M")).eval()(x, targets=y)

    assert trained.item() < 4.5, f"loaded GPT-2 should score well on English, got {trained:.2f}"
    assert random.item() > 10.0
