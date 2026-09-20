"""Tests for the chat fine-tune setup.

One thing here matters more than the rest: **the ChatML markers must be trainable.**

Qwen3-1.7B-Base reserves `<|im_start|>` and `<|im_end|>` in its vocabulary but never trained
them -- their embedding rows hold noise, at the same norm as the unused tail of the table. Qwen3
ties its embeddings, so that one row is both how the model reads a marker and how it writes one.
Combine that with a LoRA that targets only the attention and MLP projections, and no parameter in
the entire run can raise the probability of `<|im_end|>`.

The model then learns to answer correctly and cannot stop. It is a particularly nasty failure
because the loss falls normally, the answers look right, and more training does not help -- there
is nothing to train. These tests exist so it cannot come back.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from llm.chat_sft import PROBE_PROMPTS, init_chat_tokens, oasst_conversations

VOCAB, DIM = 64, 8
TRAINED_NORM = 1.6
UNTRAINED_NORM = 0.37


class FakeTokenizer:
    """Ids for the three special tokens; everything else is an ordinary row."""

    SPECIALS = {"<|endoftext|>": 60, "<|im_start|>": 61, "<|im_end|>": 62}

    def __len__(self) -> int:
        return 63

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.SPECIALS[token]


class FakeModel(nn.Module):
    """A model shaped like Qwen3's: tied embeddings, an untrained tail, reserved specials."""

    def __init__(self, im_end_trained: bool = False) -> None:
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DIM)
        with torch.no_grad():
            rows = torch.randn(VOCAB, DIM)
            rows = rows / rows.norm(dim=-1, keepdim=True)
            rows[:60] *= TRAINED_NORM          # ordinary, trained vocabulary
            rows[60] *= 1.88                   # <|endoftext|>, trained
            rows[61] *= UNTRAINED_NORM         # <|im_start|>, reserved and never trained
            rows[62] *= 1.88 if im_end_trained else UNTRAINED_NORM
            rows[63:] *= UNTRAINED_NORM        # the unused tail
            self.emb.weight.copy_(rows)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.emb


@pytest.fixture
def tok() -> FakeTokenizer:
    return FakeTokenizer()


# --- the bug that produced a model which could not stop -----------------------------------------


def test_untrained_markers_are_reseeded(tok) -> None:
    torch.manual_seed(0)
    model = FakeModel()
    before = model.emb.weight[62].norm().item()
    assert before == pytest.approx(UNTRAINED_NORM, abs=1e-5), "fixture should start untrained"

    init_chat_tokens(model, tok)
    after = model.emb.weight[62].norm().item()
    assert after > before * 3, f"{before:.3f} -> {after:.3f}: still noise"


def test_the_stop_marker_is_seeded_from_endoftext(tok) -> None:
    """Both mean "stop here", so the model starts with something it already understands rather
    than having to discover the meaning of a random vector."""
    torch.manual_seed(0)
    model = FakeModel()
    init_chat_tokens(model, tok)
    assert torch.allclose(model.emb.weight[62], model.emb.weight[60])


def test_the_start_marker_gets_a_typical_vector(tok) -> None:
    """No natural donor, so it gets the mean of the trained rows -- the standard choice for a new
    token. It must not be left at noise, and must not be a copy of endoftext either."""
    torch.manual_seed(0)
    model = FakeModel()
    init_chat_tokens(model, tok)
    start = model.emb.weight[61]
    assert start.norm() > UNTRAINED_NORM * 3
    assert not torch.allclose(start, model.emb.weight[60])


def test_a_trained_marker_is_left_alone(tok) -> None:
    """Run against an instruct model, or a second time against the same model, this must not
    overwrite weights that already carry meaning."""
    torch.manual_seed(0)
    model = FakeModel(im_end_trained=True)
    original = model.emb.weight[62].clone()
    init_chat_tokens(model, tok)
    assert torch.equal(model.emb.weight[62], original)


def test_running_twice_changes_nothing_the_second_time(tok) -> None:
    """Training seeds, then inference seeds again before loading the adapter. If the two differ,
    the adapter is applied to a model it was not fitted to."""
    torch.manual_seed(0)
    model = FakeModel()
    init_chat_tokens(model, tok)
    once = model.emb.weight.clone()
    init_chat_tokens(model, tok)
    assert torch.equal(model.emb.weight, once)


def test_seeding_is_reproducible_across_loads(tok) -> None:
    """`scripts/talk.py` reseeds a freshly loaded base model and expects the same result training
    got. Nothing random may enter."""
    torch.manual_seed(0)
    a = FakeModel()
    torch.manual_seed(0)
    b = FakeModel()
    init_chat_tokens(a, tok)
    init_chat_tokens(b, tok)
    assert torch.equal(a.emb.weight, b.emb.weight)


def test_ordinary_tokens_are_untouched(tok) -> None:
    torch.manual_seed(0)
    model = FakeModel()
    before = model.emb.weight[:60].clone()
    init_chat_tokens(model, tok)
    assert torch.equal(model.emb.weight[:60], before)


def test_lora_targets_the_output_head() -> None:
    """The other half of the fix. Seeding gives the markers a sensible starting vector; without
    `lm_head` in the LoRA targets, nothing can move them afterwards, because Qwen3 ties its
    embeddings and the base weight is frozen."""
    import inspect

    from llm import chat_sft

    source = inspect.getsource(chat_sft.main)
    assert '"lm_head"' in source, "lm_head must be a LoRA target or the model cannot learn to stop"


# --- the rest -----------------------------------------------------------------------------------


def test_probe_prompts_cover_the_observed_failures() -> None:
    """These are printed during training because loss says little about whether a chat model is
    any good. They must stay pinned to the exchanges that actually went wrong."""
    asked = " ".join(m[-1]["content"].lower() for m in PROBE_PROMPTS)
    for phrase in ("what sup", "who am i", "love me", "my name"):
        assert phrase in asked
    # At least one probe must be multi-turn, or nothing checks that history is used.
    assert any(len(m) > 1 for m in PROBE_PROMPTS)


def test_oasst_flattening_follows_the_best_ranked_reply() -> None:
    """OASST2 is a forest of message trees. Following rank 0 at each step gives the conversation
    a human preferred; taking any child would train on answers that were rated worse."""
    rows = [
        {"message_id": "a", "parent_id": None, "role": "prompter", "text": "hi",
         "lang": "en", "rank": None, "deleted": False},
        {"message_id": "b", "parent_id": "a", "role": "assistant", "text": "good reply",
         "lang": "en", "rank": 0, "deleted": False},
        {"message_id": "c", "parent_id": "a", "role": "assistant", "text": "worse reply",
         "lang": "en", "rank": 1, "deleted": False},
    ]
    out = oasst_conversations(rows, limit=10)
    assert len(out) == 1
    assert out[0]["messages"] == [{"role": "user", "content": "hi"},
                                  {"role": "assistant", "content": "good reply"}]


def test_oasst_skips_other_languages_and_deleted_rows() -> None:
    rows = [
        {"message_id": "a", "parent_id": None, "role": "prompter", "text": "hi",
         "lang": "de", "rank": None, "deleted": False},
        {"message_id": "b", "parent_id": None, "role": "prompter", "text": "yo",
         "lang": "en", "rank": None, "deleted": True},
    ]
    assert oasst_conversations(rows, limit=10) == []
