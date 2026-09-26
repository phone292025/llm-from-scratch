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

import json
import pathlib

import pytest
import torch
from torch import nn

from llm.chat_data import length_grouped_batches
from llm.chat_infer import find_checkpoint
from llm.chat_sft import (
    DATA_RECIPE,
    HELDOUT_PROMPTS,
    PROBE_PROMPTS,
    epoch_batches,
    init_chat_tokens,
    oasst_conversations,
    save_checkpoint,
    split_records,
)

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


# --- data order across epochs and resumes -------------------------------------------------------


class Lengths:
    """A dataset reduced to what batching reads: example i has length lengths[i]."""

    def __init__(self, lengths):
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, i):
        return [0] * self.lengths[i], [0] * self.lengths[i]


@pytest.fixture
def data() -> Lengths:
    return Lengths([(i * 37) % 50 + 1 for i in range(60)])


def test_epoch_zero_is_the_order_the_paused_run_used(data) -> None:
    """The chat run paused at step 200 was built with length_grouped_batches(seed). Its first epoch
    must come back identical, or resuming it lands on a different data stream."""
    assert epoch_batches(data, 4, seed=0, epoch=0) == length_grouped_batches(data, 4, seed=0)


def test_each_epoch_has_its_own_order_over_the_same_batches(data) -> None:
    first, second = (epoch_batches(data, 4, seed=0, epoch=e) for e in (0, 1))
    assert first != second
    assert sorted(map(sorted, first)) == sorted(map(sorted, second))


def test_an_epoch_order_depends_on_the_epoch_alone(data) -> None:
    """Resuming in epoch two must rebuild epoch two -- the old code seeded it with whichever step
    the wrap happened on, which a resume has no way to know."""
    assert epoch_batches(data, 4, seed=0, epoch=3) == epoch_batches(data, 4, seed=0, epoch=3)


# --- checkpoints that survive stop.bat ----------------------------------------------------------


class FakeAdapter:
    """Writes what save_pretrained writes; can be told to die partway, as a hard kill would."""

    def __init__(self, fail: bool = False):
        self.fail = fail

    def save_pretrained(self, path):
        path = pathlib.Path(path)
        path.mkdir(parents=True)
        (path / "adapter_config.json").write_text("{}")
        if self.fail:
            raise KeyboardInterrupt("killed mid-save")
        (path / "adapter_model.safetensors").write_bytes(b"weights")


@pytest.fixture
def opt():
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=0.1)


def test_a_save_is_complete_and_pointed_at(tmp_path, opt) -> None:
    final = save_checkpoint(tmp_path, FakeAdapter(), opt, 300, "some/base")
    assert final == tmp_path / "step-000300"
    assert find_checkpoint(tmp_path) == final
    assert json.loads((final / "state.json").read_text())["step"] == 300
    assert (final / "optimizer.pt").exists() and (final / "rng.pt").exists()


def test_a_kill_mid_save_leaves_the_last_good_checkpoint(tmp_path, opt) -> None:
    """The failure the old layout could not survive: interrupted while writing the adapter."""
    good = save_checkpoint(tmp_path, FakeAdapter(), opt, 200, "some/base")
    with pytest.raises(KeyboardInterrupt):
        save_checkpoint(tmp_path, FakeAdapter(fail=True), opt, 300, "some/base")
    assert find_checkpoint(tmp_path) == good
    assert not (tmp_path / "step-000300").exists()


def test_the_next_save_clears_an_abandoned_partial(tmp_path, opt) -> None:
    with pytest.raises(KeyboardInterrupt):
        save_checkpoint(tmp_path, FakeAdapter(fail=True), opt, 300, "some/base")
    save_checkpoint(tmp_path, FakeAdapter(), opt, 300, "some/base")
    assert not list(tmp_path.glob(".step-*.partial"))


def test_only_the_newest_checkpoints_are_kept(tmp_path, opt) -> None:
    """Each one is 1.1 GB. Two is enough to survive one bad save."""
    for step in (100, 200, 300, 400):
        save_checkpoint(tmp_path, FakeAdapter(), opt, step, "some/base", keep=2)
    kept = sorted(d.name for d in tmp_path.glob("step-*"))
    assert kept == ["step-000300", "step-000400"]


def test_the_original_layout_is_left_alone(tmp_path, opt) -> None:
    """The paused run's step-200 files sit directly in the run directory. New saves go beside them
    and must not delete them -- they are the only copy until the first new save lands."""
    (tmp_path / "adapter").mkdir()
    (tmp_path / "adapter" / "adapter_model.safetensors").write_bytes(b"old")
    (tmp_path / "state.json").write_text(json.dumps({"step": 200}))
    save_checkpoint(tmp_path, FakeAdapter(), opt, 300, "some/base")
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()
    assert find_checkpoint(tmp_path) == tmp_path / "step-000300"


# --- keeping validation honest ------------------------------------------------------------------


def _conv(tag: str) -> dict:
    return {"messages": [{"role": "user", "content": tag}, {"role": "assistant", "content": "ok"}]}


@pytest.fixture
def sources() -> tuple[list[dict], list[dict]]:
    return [_conv(f"persona {i}") for i in range(20)], [_conv(f"external {i}") for i in range(500)]


def _keys(records):
    return [json.dumps(r["messages"], sort_keys=True) for r in records]


def test_recipe_two_never_validates_on_training_data(sources) -> None:
    """The flaw it replaces: repeating the hand-written set before splitting put copies on both
    sides, and 22% of validation was conversations the model also trained on."""
    persona, external = sources
    train, val = split_records(persona, external, repeat=25, val_size=40, seed=0, recipe=2)
    assert not set(_keys(val)) & set(_keys(train))
    assert not set(_keys(val)) & set(_keys(persona)), "hand-written data belongs to training"
    assert len(val) == 40


def test_recipe_two_uses_every_conversation(sources) -> None:
    persona, external = sources
    train, val = split_records(persona, external, repeat=25, val_size=40, seed=0, recipe=2)
    assert len(train) == 25 * len(persona) + len(external) - 40
    assert sorted(set(_keys(train)) | set(_keys(val))) == sorted(set(_keys(persona + external)))


def test_recipe_one_is_kept_exactly_as_it_was(sources) -> None:
    """A run started under recipe 1 must resume onto the same data, so its code path is frozen.
    This is the original function, inline; the new one has to match it record for record."""
    import random

    persona, external = sources
    records = persona * 25 + external
    random.Random(0).shuffle(records)
    n = min(40, len(records) // 10)
    expected = records[n:], records[:n]
    got = split_records(persona, external, repeat=25, val_size=40, seed=0, recipe=1)
    assert _keys(got[0]) == _keys(expected[0]) and _keys(got[1]) == _keys(expected[1])


def test_recipe_one_really_did_leak(sources) -> None:
    """Documents why recipe 2 exists. Recipe 1 must stay exactly as it was, leak included, so if
    this ever fails, recipe 1 has been changed and old runs no longer resume onto their data."""
    persona, external = sources
    train, val = split_records(persona, external, repeat=25, val_size=40, seed=0, recipe=1)
    assert set(_keys(val)) & set(_keys(train))


def test_new_runs_use_the_leak_free_recipe() -> None:
    assert DATA_RECIPE == 2


def test_an_unknown_recipe_is_refused(sources) -> None:
    with pytest.raises(ValueError, match="recipe"):
        split_records(*sources, repeat=1, val_size=10, seed=0, recipe=9)


def test_checkpoints_record_their_recipe(tmp_path, opt) -> None:
    """Resume reads this to rebuild the right data. A checkpoint without it predates recipes and
    is recipe 1."""
    final = save_checkpoint(tmp_path, FakeAdapter(), opt, 100, "some/base", recipe=1)
    assert json.loads((final / "state.json").read_text())["data_recipe"] == 1


def test_heldout_probes_are_not_in_the_training_data() -> None:
    """They exist to show whether a behaviour generalized. The moment one is copied into the
    hand-written data it measures recall instead, like every original probe does."""
    normal = lambda t: " ".join(t.lower().split())
    trained = set()
    path = pathlib.Path("data/chat/persona.jsonl")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            trained |= {normal(m["content"]) for m in json.loads(line)["messages"]}
    for convo in HELDOUT_PROMPTS:
        for turn in convo:
            if turn["role"] == "user":
                assert normal(turn["content"]) not in trained, f"{turn['content']!r} is trained on"


def test_heldout_probes_cover_what_the_seen_ones_do() -> None:
    """Each behaviour probed from memory must also be probed fresh, or recall hides a gap."""
    def asks(prompts, *words):
        return any(all(w in p[-1]["content"].lower() for w in words) for p in prompts)

    assert any(len(p) > 1 for p in HELDOUT_PROMPTS), "one must test memory across turns"
    assert asks(HELDOUT_PROMPTS, "remember") or asks(HELDOUT_PROMPTS, "who")
    assert asks(HELDOUT_PROMPTS, "name")
