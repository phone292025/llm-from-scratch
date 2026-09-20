"""Tests for multi-turn chat data.

Same load-bearing concern as `test_instruct.py`, one step harder. There, the risk was grading the
prompt. Here it is grading *any* user turn anywhere in a conversation -- which teaches the model
to write both halves of the dialogue, so it answers you and then invents your next message. The
loss curve looks healthy throughout.

The second thing worth guarding is that **every** assistant turn is graded. Grading only the last
one silently throws away most of the training signal in a six-turn conversation, and produces a
model that still cannot use the history.
"""

from __future__ import annotations

import pytest

from llm.chat_data import (
    DEFAULT_SYSTEM,
    IGNORE,
    IM_END,
    IM_START,
    ChatDataset,
    build_example,
    normalize,
    render,
)


class FakeTokenizer:
    """One id per character, so assertions can read the mask directly."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(i) for i in ids)


@pytest.fixture
def tok() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def conversation() -> list[dict]:
    return [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hey!"},
        {"role": "user", "content": "what sup"},
        {"role": "assistant", "content": "Not much."},
    ]


def graded_text(tok: FakeTokenizer, labels: list[int]) -> str:
    return tok.decode([t for t in labels if t != IGNORE])


# --- the layout ---------------------------------------------------------------------------------


def test_render_ends_ready_for_the_assistant(conversation) -> None:
    # The model continues from exactly here, so nothing may follow the header.
    text = render(conversation[:3])
    assert text.endswith(f"{IM_START}assistant\n")


def test_render_does_not_add_a_header_after_an_assistant_turn(conversation) -> None:
    # Used for training text, where the conversation is already complete.
    assert render(conversation).endswith(f"{IM_END}\n")


def test_render_inserts_the_system_prompt_once(conversation) -> None:
    text = render(conversation)
    assert text.count(DEFAULT_SYSTEM) == 1
    assert text.startswith(f"{IM_START}system\n")
    # A conversation that already carries one does not get a second.
    with_system = [{"role": "system", "content": "Be terse."}] + conversation
    assert render(with_system).count(f"{IM_START}system") == 1


def test_render_is_stable_under_whitespace(conversation) -> None:
    """Chat time rebuilds this string; if it drifts, the model is asked in a layout it never saw."""
    padded = [{**m, "content": f"  {m['content']}  "} for m in conversation]
    assert render(padded) == render(conversation)


# --- masking ------------------------------------------------------------------------------------


def test_every_assistant_turn_is_graded(tok, conversation) -> None:
    """The whole point of multi-turn data. Grading only the final turn throws away the rest."""
    _, labels = build_example(tok, conversation, system=None)
    graded = graded_text(tok, labels)
    assert "Hey!" in graded and "Not much." in graded


def test_no_user_text_is_ever_graded(tok, conversation) -> None:
    _, labels = build_example(tok, conversation, system=None)
    graded = graded_text(tok, labels)
    assert "hi" not in graded and "what sup" not in graded


def test_the_system_prompt_is_not_graded(tok, conversation) -> None:
    # Otherwise the model learns to emit its own system prompt at the top of a reply.
    _, labels = build_example(tok, conversation, system="Be helpful.")
    assert "Be helpful." not in graded_text(tok, labels)


def test_the_stop_marker_is_graded(tok, conversation) -> None:
    """This is what teaches the model to stop rather than write your next message for you."""
    _, labels = build_example(tok, conversation, system=None)
    assert graded_text(tok, labels).count(IM_END) == 2


def test_the_assistant_header_is_not_graded(tok, conversation) -> None:
    # The header is supplied at generation time, so predicting it is wasted signal.
    _, labels = build_example(tok, conversation, system=None)
    assert f"{IM_START}assistant" not in graded_text(tok, labels)


def test_labels_are_the_next_token(tok, conversation) -> None:
    """labels[i] is what should follow ids[i].

    Off by one trains the model to predict the token it just read, which still produces a falling
    loss and a model that repeats your input back -- exactly the failure this data set exists to
    fix.
    """
    ids, labels = build_example(tok, conversation, system=None)
    full = tok.encode(render(conversation, system=None))
    assert ids == full[:-1]
    for i, label in enumerate(labels):
        if label != IGNORE:
            assert label == full[i + 1]


def test_grading_starts_at_the_end_of_the_header(tok, conversation) -> None:
    """The last header position must predict the first token of the answer.

    If grading begins one position late, the model is never taught how to *start* a reply.
    """
    ids, labels = build_example(tok, conversation, system=None)
    first = next(i for i, t in enumerate(labels) if t != IGNORE)
    assert tok.decode(ids[: first + 1]).endswith(f"{IM_START}assistant\n")
    assert chr(labels[first]) == "H"  # the "H" of "Hey!"


# --- rows that must be dropped ------------------------------------------------------------------


def test_truncation_respects_max_len(tok, conversation) -> None:
    ids, labels = build_example(tok, conversation, max_len=40, system=None)
    assert len(ids) == len(labels) == 39


def test_conversations_with_nothing_graded_are_dropped(tok) -> None:
    """A first user turn longer than max_len leaves a row of all -100, whose mean loss is NaN."""
    long_turn = [{"role": "user", "content": "y" * 500},
                 {"role": "assistant", "content": "ok"}]
    assert len(ChatDataset([{"messages": long_turn}], tok, max_len=50, system=None)) == 0


@pytest.mark.parametrize("bad", [
    {"messages": []},
    {"messages": [{"role": "user", "content": "hi"}]},                        # no answer
    {"messages": [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "x"}]},
    {"messages": [{"role": "user", "content": "  "}, {"role": "assistant", "content": "x"}]},
    {"nothing": "useful"},
])
def test_unusable_rows_are_rejected(bad) -> None:
    assert normalize(bad) is None


def test_normalize_accepts_the_other_common_schema() -> None:
    """OASST-style rows use from/value instead of role/content."""
    row = {"conversations": [{"from": "human", "value": "hi"},
                             {"from": "gpt", "value": "hello"}]}
    assert normalize(row) == [{"role": "user", "content": "hi"},
                              {"role": "assistant", "content": "hello"}]


def test_a_leading_system_turn_is_replaced_not_kept() -> None:
    """Datasets carry their own system prompts; ours has to be the only one, or the model learns
    to expect whichever phrasing happened to come with the data."""
    row = {"messages": [{"role": "system", "content": "You are a pirate."},
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "ahoy"}]}
    assert normalize(row)[0]["role"] == "user"


# --- the dataset --------------------------------------------------------------------------------


def test_dataset_counts_what_it_dropped(tok, conversation) -> None:
    records = [{"messages": conversation}, {"messages": []}, {"nothing": "useful"}]
    data = ChatDataset(records, tok)
    assert len(data) == 1 and data.dropped == 2


def test_longer_conversations_give_more_signal(tok) -> None:
    """Six turns must yield three graded answers, not one. This is the difference between
    multi-turn training and single-turn training on the last exchange."""
    long = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(6)]
    _, labels = build_example(tok, long, system=None)
    assert graded_text(tok, labels).count(IM_END) == 3
