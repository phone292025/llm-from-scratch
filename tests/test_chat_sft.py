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
    check_fresh_start,
    check_same_data,
    data_fingerprint,
    epoch_batches,
    init_chat_tokens,
    oasst_conversations,
    save_checkpoint,
    split_records,
)

# peft warns whenever a tied model has lm_head in its adapter. The end-to-end tests below train
# exactly that, on purpose -- it is the configuration init_chat_tokens requires -- so the notices
# are expected. Filtered by message, so any other warning still shows.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Model has `tie_word_embeddings=True`:UserWarning"),
    pytest.mark.filterwarnings("ignore:Model with `tie_word_embeddings=True`:UserWarning"),
    pytest.mark.filterwarnings("ignore:Setting `save_embedding_layers` to `True`:UserWarning"),
]

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


def test_lora_targets_the_output_head(tiny_base) -> None:
    """The other half of the fix. Seeding gives the markers a sensible starting vector; without
    `lm_head` in the LoRA targets, nothing can move them afterwards, because Qwen3 ties its
    embeddings and the base weight is frozen.

    Checked on the model build_model actually returns. An earlier version searched main()'s
    source text for "lm_head", which broke the moment the configuration moved to another function
    -- while the model itself was unchanged."""
    import argparse

    from transformers import AutoTokenizer

    from llm.chat_sft import build_model

    args = argparse.Namespace(base=str(tiny_base), lora_r=4, lora_alpha=8, no_grad_checkpointing=True)
    model = build_model(args, AutoTokenizer.from_pretrained(tiny_base), "cpu")
    head = model.get_output_embeddings()
    assert hasattr(head, "lora_A"), "lm_head must be a LoRA target or the model cannot learn to stop"
    assert any(p.requires_grad for p in head.parameters())


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


# --- resuming onto the same data -----------------------------------------------------------------


def test_the_fingerprint_notices_a_one_character_edit(sources) -> None:
    persona, external = sources
    train, val = split_records(persona, external, repeat=25, val_size=40, seed=0)
    edited = [dict(r) for r in persona]
    edited[3] = _conv("persona 3.")
    train2, val2 = split_records(edited, external, repeat=25, val_size=40, seed=0)
    assert data_fingerprint(train, val) == data_fingerprint(*split_records(
        persona, external, repeat=25, val_size=40, seed=0))
    assert data_fingerprint(train, val) != data_fingerprint(train2, val2)


def test_resuming_onto_edited_data_is_refused() -> None:
    """Editing persona.jsonl mid-run must stop a resume, not quietly change the run's data."""
    with pytest.raises(SystemExit, match="persona.jsonl"):
        check_same_data({"data_fingerprint": "aaaa"}, "bbbb")
    check_same_data({"data_fingerprint": "aaaa"}, "aaaa")


def test_checkpoints_from_before_the_fingerprint_still_resume() -> None:
    check_same_data({"step": 200}, "anything")


def test_checkpoints_record_their_fingerprint(tmp_path, opt) -> None:
    final = save_checkpoint(tmp_path, FakeAdapter(), opt, 100, "some/base", fingerprint="abc123")
    assert json.loads((final / "state.json").read_text())["data_fingerprint"] == "abc123"


def test_a_fresh_run_will_not_start_on_top_of_a_finished_one(tmp_path, opt) -> None:
    """Pruning keeps the newest two checkpoints, so a new run in an old run's folder would delete
    the old model two saves in. run.bat instead of resume.bat must not be able to do that."""
    save_checkpoint(tmp_path, FakeAdapter(), opt, 759, "some/base", finished=True)
    with pytest.raises(SystemExit, match="--resume"):
        check_fresh_start(tmp_path, resume=False)
    check_fresh_start(tmp_path, resume=True)
    check_fresh_start(tmp_path / "somewhere-new", resume=False)


# --- the whole training command, end to end, on a model small enough for the CPU ---------------

PERSONA = [
    [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hey! What's up?"}],
    [{"role": "user", "content": "what sup"},
     {"role": "assistant", "content": "Not much. What's up with you?"}],
    [{"role": "user", "content": "my name is Phone"},
     {"role": "assistant", "content": "Nice to meet you, Phone."},
     {"role": "user", "content": "what's my name"},
     {"role": "assistant", "content": "Phone -- you told me a moment ago."}],
    [{"role": "user", "content": "who are you"}, {"role": "assistant", "content": "A language model."}],
    [{"role": "user", "content": "thanks"}, {"role": "assistant", "content": "Anytime."}],
    [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "See you."}],
]


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    """A Qwen3-shaped model and a byte-level tokenizer with the ChatML markers, built offline.

    Nothing is downloaded, so this runs wherever the tests do. One layer of width 32 trains in
    milliseconds a step on the CPU -- enough to drive the real training loop end to end.
    """
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    from llm.chat_data import DEFAULT_SYSTEM

    base = tmp_path_factory.mktemp("tiny-base")
    raw = Tokenizer(models.BPE())
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    texts = [DEFAULT_SYSTEM] + [m["content"] for convo in PERSONA for m in convo]
    raw.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=320, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"]))
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    assert len(tok.encode("<|im_end|>", add_special_tokens=False)) == 1, "markers must be one token"
    tok.save_pretrained(base)

    cfg = transformers.Qwen3Config(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=256,
        tie_word_embeddings=True)
    torch.manual_seed(0)
    transformers.Qwen3ForCausalLM(cfg).save_pretrained(base)
    return base


def _chat_sft(base, out, persona_file, monkeypatch, *extra) -> None:
    import sys

    from llm.chat_sft import main

    argv = ["chat_sft", "--base", str(base), "--out", str(out), "--persona", str(persona_file),
            "--persona-repeat", "3", "--ultrachat", "0", "--oasst", "0", "--max-len", "96",
            "--epochs", "1", "--batch-size", "1", "--accum", "2", "--lr", "1e-3", "--warmup", "1",
            "--lora-r", "4", "--lora-alpha", "8", "--eval-every", "2", "--sample-every", "9",
            "--save-every", "2", "--device", "cpu", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    main()


@pytest.fixture(scope="module")
def finished_run(tiny_base, tmp_path_factory):
    """One complete run: 18 conversations, 9 optimizer steps, saving every 2."""
    monkeypatch = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("chat-run")
    persona_file = root / "persona.jsonl"
    persona_file.write_text("\n".join(json.dumps({"messages": c}) for c in PERSONA), encoding="utf-8")
    try:
        _chat_sft(tiny_base, root / "a", persona_file, monkeypatch)
    finally:
        monkeypatch.undo()
    return root / "a", persona_file


def test_a_run_finishes_with_a_complete_marked_checkpoint(finished_run) -> None:
    run, _ = finished_run
    final = find_checkpoint(run)
    assert final == run / "step-000009"
    state = json.loads((final / "state.json").read_text())
    assert state["finished"] and state["data_recipe"] == DATA_RECIPE and state["data_fingerprint"]
    assert sorted(d.name for d in run.glob("step-*")) == ["step-000008", "step-000009"]


def test_a_killed_and_resumed_run_ends_identical_to_an_unbroken_one(
    finished_run, tiny_base, tmp_path, monkeypatch
) -> None:
    """The property every resume fix in this project was for, through the real command: weights,
    optimizer, data position and dropout's random state all restored. Simulated as a kill just
    after the step-8 save -- the pointer still on step 8, step 9 never written."""
    import shutil

    from safetensors.torch import load_file

    run, persona_file = finished_run
    killed = tmp_path / "killed"
    shutil.copytree(run, killed)
    shutil.rmtree(killed / "step-000009")
    (killed / "latest.json").write_text(json.dumps({"dir": "step-000008", "step": 8}))

    _chat_sft(tiny_base, killed, persona_file, monkeypatch, "--resume")

    a = load_file(run / "step-000009" / "adapter" / "adapter_model.safetensors")
    b = load_file(killed / "step-000009" / "adapter" / "adapter_model.safetensors")
    assert a.keys() == b.keys()
    for name in a:
        assert torch.equal(a[name], b[name]), f"{name} differs after resuming"


def test_the_loaded_chat_model_computes_what_was_trained(finished_run, tiny_base) -> None:
    """load_chat_model -- find the base from the run, seed the markers, untie, merge -- must give
    the function training produced: the unmerged adapter on the tied base."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from llm.chat_data import render
    from llm.chat_infer import base_of, load_chat_model

    run, _ = finished_run
    assert base_of(run, "fallback/base") == str(tiny_base)
    merged, tok, adapter = load_chat_model(base_of(run, "fallback/base"), run, "cpu", quiet=True)
    assert adapter == run / "step-000009" / "adapter"

    trained = AutoModelForCausalLM.from_pretrained(tiny_base, dtype=torch.float32)
    init_chat_tokens(trained, tok)
    trained = PeftModel.from_pretrained(trained, adapter).eval()
    x = tok(render([{"role": "user", "content": "my name is Phone"}]), return_tensors="pt",
            add_special_tokens=False).input_ids
    with torch.no_grad():
        assert torch.allclose(merged(x).logits, trained(x).logits, atol=1e-5)


def test_chat_validation_loss_is_the_mean_over_graded_tokens(tiny_base) -> None:
    """The end-to-end runs have no validation set (no external data offline), so evaluate()'s
    scoring loop never ran in a test. Checked here against the definition: every graded token
    counts once, prompts and padding not at all."""
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llm.chat_data import IGNORE, ChatDataset, collate
    from llm.chat_sft import evaluate

    tok = AutoTokenizer.from_pretrained(tiny_base)
    model = AutoModelForCausalLM.from_pretrained(tiny_base, dtype=torch.float32)
    data = ChatDataset([{"messages": c} for c in PERSONA], tok, 96)
    batches = length_grouped_batches(data, 2, seed=0)

    total, count = 0.0, 0
    with torch.no_grad():
        for batch in batches:
            x, y = collate([data[i] for i in batch], tok.pad_token_id)
            logits = model(input_ids=x).logits
            total += F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=IGNORE,
                                     reduction="sum").item()
            count += int((y != IGNORE).sum())

    got = evaluate(model, batches, data, tok.pad_token_id, "cpu", limit=len(batches))
    assert got == pytest.approx(total / count, rel=1e-5)
    assert model.training, "evaluate must hand the model back in training mode"


def test_the_knowledge_quiz_is_not_in_the_training_data() -> None:
    """scripts/knowledge_check.py measures what the model knows and whether it admits what it
    does not. If one of its questions is copied into the hand-written training data -- easy to do
    when writing honesty examples aimed at exactly its failures -- that question measures recall."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("knowledge_check", "scripts/knowledge_check.py")
    quiz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quiz)

    def normal(text):
        return " ".join(text.lower().replace("?", "").split())

    trained = set()
    for line in pathlib.Path("data/chat/persona.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            trained |= {normal(m["content"]) for m in json.loads(line)["messages"]}
    asked = [turn for _, turns, _ in quiz.TESTS for turn in turns]
    copied = [q for q in asked if normal(q) in trained]
    assert not copied, f"quiz questions found in the training data: {copied}"
