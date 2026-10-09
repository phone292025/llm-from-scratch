"""Tests for preference tuning (DPO).

The failures worth guarding against are the silent ones, where the loss still falls:

- **The sign.** Swap chosen and rejected anywhere and the model is trained to prefer the bad
  answer -- with a loss that falls exactly as nicely.
- **The reference.** It must be the chat model as it was before DPO, including after a resume,
  which rebuilds it. Built from the policy as it trains, every margin reads 0 and nothing is
  learned; built from a resumed, half-tuned policy, the run quietly changes objective.
- **Dropout.** Left on, the policy differs from the reference at step 0 by noise alone, and the
  margins start out random.
- **What is scored.** Only the final reply is a choice. Everything before it is shared context.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llm.chat_data import IGNORE, IM_END, render
from llm.chat_infer import find_checkpoint, resolve_adapter
from llm.chat_sft import read_state
from llm.dpo import (
    PreferenceDataset,
    build_pair,
    check_same_start,
    disable_dropout,
    dpo_loss,
    load_records,
    logprobs_for,
    pair_batches,
    problem_with,
    records_fingerprint,
    sequence_logprobs,
    split_pairs,
    summarize,
)

# peft warns whenever a tied model has lm_head in its adapter -- the configuration the chat model
# needs (see test_chat_sft). Filtered by message, so any other warning still shows.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Model has `tie_word_embeddings=True`:UserWarning"),
    pytest.mark.filterwarnings("ignore:Model with `tie_word_embeddings=True`:UserWarning"),
    pytest.mark.filterwarnings("ignore:Setting `save_embedding_layers` to `True`:UserWarning"),
]

DATA = Path("data/chat/preferences.jsonl")


class FakeTokenizer:
    """One id per character, so assertions can read the labels directly."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(i) for i in ids)


@pytest.fixture
def tok() -> FakeTokenizer:
    return FakeTokenizer()


def graded(tok, labels: list[int]) -> str:
    return tok.decode([t for t in labels if t != IGNORE])


GOOD = {"messages": [{"role": "user", "content": "who made you"}],
        "chosen": "Qwen's team, then a small project.", "rejected": "OpenAI made me."}


# --- the objective -------------------------------------------------------------------------------


def test_the_loss_starts_as_a_coin_flip() -> None:
    """At step 0 the policy is the reference, so no answer is preferred: loss log 2, margin 0."""
    lp = torch.tensor([-12.0, -30.0])
    loss, margin = dpo_loss(lp, lp - 5, lp, lp - 5, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2))
    assert torch.equal(margin, torch.zeros(2))


def test_the_gradient_raises_chosen_and_lowers_rejected() -> None:
    """The sign. Swapped, the model learns to prefer the bad answer while the loss falls just as
    nicely -- nothing else in a run would show it."""
    pi_c = torch.tensor([-10.0], requires_grad=True)
    pi_r = torch.tensor([-10.0], requires_grad=True)
    loss, _ = dpo_loss(pi_c, pi_r, torch.tensor([-10.0]), torch.tensor([-10.0]), beta=0.1)
    loss.backward()
    assert pi_c.grad.item() < 0 < pi_r.grad.item()   # descent raises chosen, lowers rejected


def test_only_movement_from_the_reference_counts() -> None:
    """An answer the chat model already preferred earns nothing for it. What DPO rewards is the
    change -- which is why the reference has to be the model before training, and nothing else."""
    ref_c, ref_r = torch.tensor([-5.0]), torch.tensor([-40.0])   # already strongly prefers chosen
    loss, margin = dpo_loss(ref_c, ref_r, ref_c, ref_r, beta=0.1)
    assert margin.item() == 0 and loss.item() == pytest.approx(math.log(2))


def test_the_margin_is_beta_times_the_relative_shift() -> None:
    # chosen rose by 2 nats and rejected fell by 5: a relative shift of 7
    pi_c, pi_r, ref_c, ref_r = (torch.tensor([v]) for v in (-8.0, -20.0, -10.0, -15.0))
    _, small = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1)
    _, large = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.5)
    assert small.item() == pytest.approx(0.7)
    assert large.item() == pytest.approx(3.5)


def test_accuracy_counts_only_a_real_preference() -> None:
    """At step 0 every margin is exactly 0 -- no preference yet, so 0%, not 50%."""
    stats = summarize(torch.tensor([0.0, 0.5, -0.1, 0.0]), torch.tensor([1.0, -1.0]),
                      torch.tensor([-2.0, 0.0]))
    assert stats["accuracy"] == 0.25 and stats["margin"] == pytest.approx(0.1)
    assert stats["chosen_shift"] == 0.0 and stats["rejected_shift"] == -1.0


# --- scoring a reply -----------------------------------------------------------------------------


class FixedLogits(nn.Module):
    """Stands in for a model: the same logits, whatever it is given to read."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.logits)


def test_a_reply_scores_the_sum_of_its_graded_token_logprobs() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    labels = torch.tensor([[IGNORE, IGNORE, 3, 1, IGNORE], [IGNORE, 4, 4, 0, 6]])
    got = sequence_logprobs(FixedLogits(logits), torch.zeros(2, 5, dtype=torch.long), labels)
    lsm = logits.log_softmax(-1)
    want = torch.stack([lsm[0, 2, 3] + lsm[0, 3, 1],
                        lsm[1, 1, 4] + lsm[1, 2, 4] + lsm[1, 3, 0] + lsm[1, 4, 6]])
    assert torch.allclose(got, want, atol=1e-6)


def test_only_graded_positions_receive_gradient() -> None:
    """The prompt is context, not a choice: nothing in it may be pushed up or down."""
    logits = torch.randn(1, 4, 6, requires_grad=True)
    labels = torch.tensor([[IGNORE, IGNORE, 2, 5]])
    ids = torch.zeros(1, 4, dtype=torch.long)
    sequence_logprobs(FixedLogits(logits), ids, labels).sum().backward()
    assert logits.grad[0, :2].abs().sum() == 0
    assert logits.grad[0, 2:].abs().sum() > 0


def test_bf16_logits_are_summed_in_fp32() -> None:
    """A reply's log-probability is a sum of dozens of terms, and bf16 keeps about three
    significant digits: a sum near -30 would be rounded to the nearest eighth of a nat."""
    labels = torch.tensor([[IGNORE, 1, 2]])
    got = sequence_logprobs(FixedLogits(torch.randn(1, 3, 8).bfloat16()),
                            torch.zeros(1, 3, dtype=torch.long), labels)
    assert got.dtype == torch.float32


# --- building pairs ------------------------------------------------------------------------------

PROMPT = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hey!"},
          {"role": "user", "content": "who made you"}]


def test_only_the_final_reply_is_graded(tok) -> None:
    """The earlier reply, "Hey!", is shared by both answers: context, not part of the choice."""
    (_, c_labels), (_, r_labels) = build_pair(tok, PROMPT, "Qwen's team.", "OpenAI.", system=None)
    assert graded(tok, c_labels) == "Qwen's team." + IM_END + "\n"
    assert graded(tok, r_labels) == "OpenAI." + IM_END + "\n"


def test_both_answers_continue_the_same_conversation(tok) -> None:
    """Token for token identical up to the reply -- otherwise the pair compares two questions."""
    (c_ids, c_labels), (r_ids, _) = build_pair(tok, PROMPT, "Yes.", "No way.")
    first = next(i for i, t in enumerate(c_labels) if t != IGNORE)
    assert c_ids[: first + 1] == r_ids[: first + 1]
    assert tok.decode(c_ids[: first + 1]) == render(PROMPT)


@pytest.mark.parametrize("bad, why", [
    ({**GOOD, "messages": []}, "ending with the user"),
    ({**GOOD, "messages": [*GOOD["messages"], {"role": "assistant", "content": "x"}]},
     "ending with the user"),
    ({**GOOD, "chosen": "  "}, "'chosen'"),
    ({k: v for k, v in GOOD.items() if k != "rejected"}, "'rejected'"),
    ({**GOOD, "rejected": f" {GOOD['chosen']} "}, "same answer"),
])
def test_a_malformed_record_is_refused_by_number(tok, bad, why) -> None:
    """Hand-written data: a typo must stop the run, not quietly leave a behaviour untrained."""
    assert why in problem_with(bad)
    with pytest.raises(ValueError, match="record 1"):
        PreferenceDataset([GOOD, bad], tok)


def test_a_pair_whose_prompt_fills_max_len_is_dropped(tok) -> None:
    """Nothing graded means log p = 0 for both answers: a pair that can only say "no preference"."""
    long = {**GOOD, "messages": [{"role": "user", "content": "y" * 400}]}
    data = PreferenceDataset([GOOD, long], tok, max_len=300)
    assert len(data) == 1 and data.dropped == 1 and data.records == [GOOD]


# --- order, split and resume checks --------------------------------------------------------------


def test_each_epoch_visits_every_pair_once() -> None:
    for epoch in range(3):
        flat = [i for batch in pair_batches(7, 2, seed=0, epoch=epoch) for i in batch]
        assert sorted(flat) == list(range(7))


def test_the_order_depends_on_seed_and_epoch_alone() -> None:
    """What makes a resume a seek rather than a replay."""
    assert pair_batches(9, 2, seed=0, epoch=1) == pair_batches(9, 2, seed=0, epoch=1)
    assert pair_batches(9, 2, seed=0, epoch=1) != pair_batches(9, 2, seed=0, epoch=2)


def test_at_most_a_fifth_is_held_out() -> None:
    records = [{"i": i} for i in range(23)]
    train, val = split_pairs(records, val_size=12, seed=0)
    assert len(val) == 4 and len(train) == 19
    assert sorted(r["i"] for r in train + val) == list(range(23))


def test_dropout_is_switched_off_everywhere() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.1), nn.Sequential(nn.Dropout(0.5)))
    disable_dropout(model)
    assert all(m.p == 0 for m in model.modules() if isinstance(m, nn.Dropout))


def test_the_fingerprint_notices_an_edited_answer() -> None:
    edited = {**GOOD, "chosen": GOOD["chosen"] + "!"}
    assert records_fingerprint([GOOD]) == records_fingerprint([dict(GOOD)])
    assert records_fingerprint([GOOD]) != records_fingerprint([edited])


def test_resuming_from_a_different_chat_model_is_refused(tmp_path) -> None:
    """--sft checkpoints/chat names whichever checkpoint is newest; train that run further and a
    resume would quietly rebuild the reference from a different model."""
    began = tmp_path / "chat" / "step-000759" / "adapter"
    check_same_start({"sft_adapter": str(began)}, began)
    check_same_start({}, began)   # older checkpoints did not record it, and cannot be checked
    with pytest.raises(SystemExit, match="--sft"):
        check_same_start({"sft_adapter": str(began)}, tmp_path / "chat" / "step-000800" / "adapter")


def test_missing_data_says_how_to_make_it(tmp_path) -> None:
    with pytest.raises(SystemExit, match="make_preference_data"):
        load_records(tmp_path / "nowhere.jsonl")


# --- the hand-written pairs ----------------------------------------------------------------------


def test_the_preference_data_is_well_formed() -> None:
    records = load_records(DATA)
    assert len(records) >= 60
    assert [problem_with(r) for r in records] == [None] * len(records)
    prompts = [json.dumps(r["messages"]) for r in records]
    assert len(set(prompts)) == len(prompts), "the same prompt appears twice"


def test_the_data_file_is_what_its_generator_writes() -> None:
    """The script is what gets reviewed; the file is what gets trained on. Edit one and not the
    other, and the reviewed pairs are not the trained ones."""
    spec = importlib.util.spec_from_file_location("make_preference_data",
                                                  "scripts/make_preference_data.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    assert load_records(DATA) == generator.rows()


def test_the_data_also_punishes_needless_refusals() -> None:
    """Pairs that only ever reward "I don't know" train a model that refuses everything. Enough
    rejected answers must be refusals of questions that have plain answers."""
    refusal = ("not sure", "can't", "don't know", "not able", "unable", "don't have",
               "not certain")
    records = load_records(DATA)
    calibrating = [r for r in records if any(w in r["rejected"].lower() for w in refusal)
                   and not any(w in r["chosen"].lower() for w in refusal)]
    assert len(calibrating) >= 10


REFUSAL_OPENING = re.compile(
    r"\W*(no\b|not\b|nobody|nothing|none\b|i (don't|can't|cannot|have no|haven't)|i'm (not|unable)"
    r"|i've never|there's no|there is no|that's not|you haven't|you've only)")


def test_style_does_not_give_the_answer_away() -> None:
    """DPO learns whatever separates chosen from rejected, wanted or not. The first version of the
    pairs had " -- " in 33 of 69 chosen answers against 4 rejected, and opened 37 chosen answers
    with a refusal against 7 rejected. The tuned model put dashes everywhere and answered a valid
    question with "There's no Cisco IOS". Surface features must be about as common on both sides."""
    records = load_records(DATA)

    def share(has) -> tuple[float, float]:
        return tuple(sum(bool(has(r[k])) for r in records) / len(records)
                     for k in ("chosen", "rejected"))

    assert share(lambda t: "--" in t) == (0, 0)
    chosen, rejected = share(lambda t: REFUSAL_OPENING.match(t.lower()))
    assert abs(chosen - rejected) < 0.15, (chosen, rejected)
    longer = sum(len(r["chosen"]) > len(r["rejected"]) for r in records) / len(records)
    assert 0.35 < longer < 0.65, f"the chosen answer is the longer one in {longer:.0%} of pairs"


def test_every_pair_fits_the_default_max_len_whole() -> None:
    """A truncated answer is graded on its first part only. Byte-level BPE never needs more tokens
    than an ASCII text has characters, so fitting in 512 characters means fitting in 512 tokens."""
    for r in load_records(DATA):
        for reply in (r["chosen"], r["rejected"]):
            text = render([*r["messages"], {"role": "assistant", "content": reply}])
            assert text.isascii() and len(text) < 512, text[:60]


# --- the whole command, end to end, on the tiny CPU model ----------------------------------------

PAIRS = [
    ("who made you", "Qwen's team, then a small project.", "OpenAI made me."),
    ("can you browse the web", "No, I can't.", "Yes, I browse all day."),
    ("who was Zed Quorn", "I don't know that name.", "A famous Roman general."),
    ("what's 2 plus 2", "4.", "I can't do maths."),
    ("are you a person", "No, I'm software.", "Yes, a real person."),
    ("what's my favourite colour", "You haven't told me.", "Blue."),
    ("capital of France", "Paris.", "I'm not sure."),
    ("who wins next year's cup", "Nobody knows yet.", "Brazil wins it."),
    ("can you send an email", "I can't send email.", "Sent!"),
    ("how many legs does a cat have", "Four.", "I don't know."),
]
RECORDS = [{"messages": [{"role": "user", "content": q}], "chosen": c, "rejected": r}
           for q, c, r in PAIRS]


def _run_dpo(sft, data, out, *extra) -> None:
    """`python -m llm.dpo` with settings that learn something in 16 steps of a tiny model."""
    from llm.dpo import main

    main(["--sft", str(sft), "--data", str(data), "--out", str(out), "--beta", "0.5",
          "--lr", "5e-3", "--epochs", "4", "--batch-size", "2", "--accum", "1", "--warmup", "1",
          "--max-len", "96", "--val-size", "2", "--eval-every", "4", "--save-every", "2",
          "--sample-every", "8", "--device", "cpu", *extra])


@pytest.fixture(scope="module")
def dpo_run(finished_run, tmp_path_factory):
    """One complete DPO run from the finished chat run: 8 training pairs, 16 steps, saving every 2."""
    sft_run, _ = finished_run
    root = tmp_path_factory.mktemp("dpo-run")
    data = root / "preferences.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in RECORDS), encoding="utf-8")
    _run_dpo(sft_run, data, root / "a")
    return root / "a", data, sft_run


def _metrics(run: Path) -> tuple[list[dict], list[dict]]:
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    return [r for r in rows if r["kind"] == "train"], [r for r in rows if r["kind"] == "eval"]


def test_the_policy_is_the_reference_until_it_is_trained(finished_run, tiny_base) -> None:
    """Loaded for DPO, the chat model in training mode must score every pair exactly as in eval
    mode, where the reference is taken. The chat LoRA has dropout 0.05: left on, two passes over
    the same pair disagree, and step 0 starts from noise."""
    from transformers import AutoTokenizer

    from llm.dpo import load_policy, reference_logprobs

    sft_run, _ = finished_run
    tok = AutoTokenizer.from_pretrained(tiny_base)
    data = PreferenceDataset(RECORDS, tok, max_len=96)
    model = load_policy(str(tiny_base), resolve_adapter(sft_run), tok, "cpu")
    ref_c, ref_r = reference_logprobs(model, data, tok.pad_token_id, "cpu", batch_size=4)
    model.train()
    every = list(range(len(data)))
    first = [lp.detach() for lp in logprobs_for(model, data, every, tok.pad_token_id, "cpu")]
    second = [lp.detach() for lp in logprobs_for(model, data, every, tok.pad_token_id, "cpu")]
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    assert torch.allclose(first[0], ref_c, atol=1e-5) and torch.allclose(first[1], ref_r, atol=1e-5)


def test_a_run_starts_as_a_coin_flip(dpo_run) -> None:
    """Step 0, before any update: every margin 0, the loss log 2. Dropout left on, or a reference
    taken from any other model, shows up here first."""
    train, _ = _metrics(dpo_run[0])
    assert train[0]["step"] == 0
    assert train[0]["loss"] == pytest.approx(math.log(2), abs=1e-4)
    assert train[0]["margin"] == pytest.approx(0.0, abs=1e-4)


def test_a_run_learns_to_prefer_the_chosen_answers(dpo_run) -> None:
    train, held = _metrics(dpo_run[0])
    late = train[-4:]
    assert all(r["accuracy"] == 1.0 for r in late)
    assert sum(r["loss"] for r in late) / len(late) < math.log(2) - 0.15
    assert held and held[-1]["step"] == train[-1]["step"], "the last step is always evaluated"


def test_a_dpo_checkpoint_says_what_it_is(dpo_run) -> None:
    out, data, sft_run = dpo_run
    final = find_checkpoint(out)
    assert final == out / "step-000016"
    state = read_state(final)
    assert state["kind"] == "dpo" and state["finished"] and state["beta"] == 0.5
    assert state["sft_adapter"] == str(resolve_adapter(sft_run))
    assert state["data_fingerprint"] == records_fingerprint(load_records(data))
    assert state["data_recipe"] is None, "chat_sft's data recipes say nothing about DPO data"
    assert sorted(d.name for d in out.glob("step-*")) == ["step-000014", "step-000016"]


def test_talking_to_the_tuned_model_uses_what_dpo_learned(dpo_run, tiny_base) -> None:
    """Through the loader talk.py uses -- base read from the run, markers seeded, head untied,
    adapter merged -- the tuned model must prefer the chosen answers by more than the chat model
    it started from did."""
    from llm.chat_infer import base_of, load_chat_model

    out, data, sft_run = dpo_run
    records = load_records(data)

    def preference(run: Path) -> float:
        model, tok, _ = load_chat_model(base_of(run, "fallback"), run, "cpu", quiet=True)
        pairs = PreferenceDataset(records, tok, max_len=96)
        with torch.no_grad():
            chosen, rejected = logprobs_for(model, pairs, list(range(len(pairs))),
                                            tok.pad_token_id, "cpu")
        return (chosen - rejected).mean().item()

    assert base_of(out, "fallback") == str(tiny_base)
    before, after = preference(sft_run), preference(out)
    assert after > before + 0.5, (before, after)


def test_a_killed_and_resumed_run_ends_identical_to_an_unbroken_one(dpo_run, tmp_path) -> None:
    """A resume rebuilds the reference -- from the chat model, before the DPO checkpoint is
    restored. Built after, it would be the half-tuned policy, and the run would quietly change
    objective. Simulated as a kill just after the step-14 save."""
    from safetensors.torch import load_file

    out, data, sft_run = dpo_run
    killed = tmp_path / "killed"
    shutil.copytree(out, killed)
    shutil.rmtree(killed / "step-000016")
    (killed / "latest.json").write_text(json.dumps({"dir": "step-000014", "step": 14}))

    _run_dpo(sft_run, data, killed, "--resume")

    a = load_file(out / "step-000016" / "adapter" / "adapter_model.safetensors")
    b = load_file(killed / "step-000016" / "adapter" / "adapter_model.safetensors")
    assert a.keys() == b.keys()
    for name in a:
        assert torch.equal(a[name], b[name]), f"{name} differs after resuming"


def test_resuming_onto_edited_preferences_is_refused(dpo_run, tmp_path) -> None:
    out, data, sft_run = dpo_run
    original = data.read_text(encoding="utf-8")
    edited = tmp_path / "edited.jsonl"
    edited.write_text(original.replace("Paris.", "Paris, France."), encoding="utf-8")
    assert edited.read_text(encoding="utf-8") != original
    run = tmp_path / "run"
    shutil.copytree(out, run)
    with pytest.raises(SystemExit, match="edited.jsonl"):
        _run_dpo(sft_run, edited, run, "--resume")


def test_a_new_run_will_not_start_on_top_of_a_finished_one(dpo_run) -> None:
    out, data, sft_run = dpo_run
    with pytest.raises(SystemExit, match="--resume"):
        _run_dpo(sft_run, data, out)


def test_dpo_needs_a_trained_chat_model_to_start_from(tmp_path) -> None:
    with pytest.raises(SystemExit, match="no trained chat adapter"):
        _run_dpo(tmp_path / "no-run", DATA, tmp_path / "out")
