"""Tests for instruction-tuning data.

The one that matters is the masking. If the prompt is not masked out, the model is graded on
reproducing the question, which teaches it to write questions rather than answers -- and the loss
curve looks perfectly healthy the whole time.
"""

from __future__ import annotations

import pytest
import torch

from llm.instruct import (
    IGNORE,
    InstructionDataset,
    build_example,
    build_prompt,
    collate,
    length_grouped_batches,
    split,
)


class FakeTokenizer:
    """One id per character, so token boundaries are obvious in assertions."""

    eot_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(i) for i in ids)


@pytest.fixture
def tok() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def records() -> list[dict]:
    return [
        {"instruction": f"Task {i}", "input": "ctx" if i % 3 else "", "output": f"Answer {i}" * (i % 5 + 1)}
        for i in range(40)
    ]


# --- the prompt layout --------------------------------------------------------------------------


def test_prompt_includes_the_input_only_when_there_is_one() -> None:
    assert "### Input:" not in build_prompt("Do a thing")
    assert "### Input:" in build_prompt("Do a thing", "some context")


def test_prompt_ends_at_the_response_marker() -> None:
    # The model must continue from exactly here, so nothing may follow it.
    assert build_prompt("Do a thing").endswith("### Response:\n")


def test_prompt_is_stable() -> None:
    """Chat time has to rebuild this byte for byte, or the model is being asked in a layout it
    was never trained on."""
    assert build_prompt(" Do a thing ") == build_prompt("Do a thing")


# --- masking ------------------------------------------------------------------------------------


def test_only_the_response_is_graded(tok: FakeTokenizer) -> None:
    ids, labels = build_example(tok, "Add", "", "42")
    graded = [(i, label) for i, label in enumerate(labels) if label != IGNORE]
    assert graded, "nothing graded at all"

    # Everything graded decodes to the response plus the stop token, in order.
    assert tok.decode([label for _, label in graded]) == "42" + chr(tok.eot_id)
    # And the prompt itself is entirely ignored.
    prompt_len = len(tok.encode(build_prompt("Add")))
    assert all(label == IGNORE for label in labels[: prompt_len - 1])


def test_labels_are_the_next_token(tok: FakeTokenizer) -> None:
    """labels[i] is what should follow ids[i]. Off by one here trains the model to predict the
    token it just read, which still produces a falling loss."""
    ids, labels = build_example(tok, "Add", "", "42")
    full = tok.encode(build_prompt("Add")) + tok.encode("42") + [tok.eot_id]
    assert ids == full[:-1]
    for i, label in enumerate(labels):
        if label != IGNORE:
            assert label == full[i + 1]


def test_the_last_graded_target_is_the_stop_token(tok: FakeTokenizer) -> None:
    # This is what teaches the model to stop rather than run on into the next thing.
    _, labels = build_example(tok, "Add", "", "42")
    assert [t for t in labels if t != IGNORE][-1] == tok.eot_id


def test_truncation_respects_max_len(tok: FakeTokenizer) -> None:
    ids, labels = build_example(tok, "Add", "", "x" * 999, max_len=64)
    assert len(ids) == len(labels) == 63


def test_examples_with_nothing_graded_are_dropped(tok: FakeTokenizer) -> None:
    """A response truncated away entirely leaves a row of all -100, whose mean loss is NaN."""
    long_instruction = "y" * 500
    ds = InstructionDataset(
        [{"instruction": long_instruction, "input": "", "output": "hello"}], tok, max_len=50
    )
    assert len(ds) == 0


# --- batching -------------------------------------------------------------------------------------


def test_collate_pads_to_the_longest_and_masks_the_padding(tok: FakeTokenizer) -> None:
    batch = [([1, 2, 3], [IGNORE, 2, 3]), ([4, 5], [IGNORE, 5])]
    x, y = collate(batch, pad_id=tok.eot_id)
    assert x.shape == y.shape == (2, 3)
    assert x[1].tolist() == [4, 5, tok.eot_id]
    # The pad position must not be graded, or the model learns to emit padding.
    assert y[1].tolist() == [IGNORE, 5, IGNORE]


def test_padding_is_ignored_by_the_loss() -> None:
    """cross_entropy skips -100 by default, which is why IGNORE is that value and not, say, 0."""
    logits = torch.randn(1, 3, 10)
    targets_all = torch.tensor([[4, 5, 6]])
    targets_masked = torch.tensor([[4, 5, IGNORE]])
    both = torch.nn.functional.cross_entropy(
        logits[:, :2].flatten(0, 1), targets_all[:, :2].flatten()
    )
    masked = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), targets_masked.flatten()
    )
    assert torch.allclose(both, masked)


def test_length_grouping_covers_every_example_exactly_once(
    records: list[dict], tok: FakeTokenizer
) -> None:
    ds = InstructionDataset(records, tok)
    batches = length_grouped_batches(ds, batch_size=7)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(ds)))


def test_length_grouping_cuts_padding(records: list[dict], tok: FakeTokenizer) -> None:
    ds = InstructionDataset(records, tok)
    lengths = [len(ds[i][0]) for i in range(len(ds))]

    def computed(batches):
        return sum(max(lengths[i] for i in b) * len(b) for b in batches)

    in_order = [list(range(i, min(i + 7, len(ds)))) for i in range(0, len(ds), 7)]
    grouped = length_grouped_batches(ds, 7)
    assert computed(grouped) <= computed(in_order)
    assert computed(grouped) < sum(lengths) * 1.3


# --- splitting --------------------------------------------------------------------------------


def test_split_holds_out_the_requested_size_and_shuffles(records: list[dict]) -> None:
    train, val = split(records, val_size=10)
    assert len(val) == 10 and len(train) == len(records) - 10
    # Shuffled, so the validation set is not just the tail of the file.
    assert val != records[:10]
    instructions = {r["instruction"] for r in train}
    assert not instructions & {r["instruction"] for r in val}
