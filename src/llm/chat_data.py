"""Multi-turn conversation data, in ChatML, with the right things masked.

This is where the previous attempt went wrong, so it is worth being explicit about the three
differences from `instruct.py`:

1. **Conversations, not tasks.** Alpaca is 52k single-turn instructions -- "translate this",
   "summarize that". A model trained on it has never seen a conversation, so "what sup" gets
   answered as though it were a task to be described. Here every example is a real dialogue.

2. **Every assistant turn is graded, not just the last.** A three-exchange conversation is three
   training signals, each conditioned on everything before it. That is what teaches the model to
   use the history rather than answer each message from scratch.

3. **ChatML, which the base model already knows.** Qwen's vocabulary contains `<|im_start|>` and
   `<|im_end|>` as single tokens. Using its native format means the role markers are one token
   each and the model has seen them during pretraining.

The masking is the dangerous part, exactly as it was in `instruct.py`: if the user's turns are
graded, the model learns to write the user's side of the conversation, and the loss curve looks
perfectly healthy while it happens.
"""

from __future__ import annotations

from typing import Iterable, Protocol, Sequence

from torch.utils.data import Dataset

from llm.instruct import (  # noqa: F401  (re-exported)
    IGNORE,
    collate,
    length_grouped_batches,
)

#: One turn: {"role": "system" | "user" | "assistant", "content": str}.
Message = dict[str, str]

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

DEFAULT_SYSTEM = "You are a friendly assistant. You chat naturally and answer questions."


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


def render(messages: Sequence[Message], system: str | None = DEFAULT_SYSTEM) -> str:
    """The ChatML text for a conversation, ending ready for the assistant to continue.

    Chat time must rebuild this byte for byte, or the model is being asked in a layout it was
    never trained on -- which is its own silent failure, since it will still answer.
    """
    turns = list(messages)
    if system and (not turns or turns[0]["role"] != "system"):
        turns = [{"role": "system", "content": system}] + turns
    text = "".join(f"{IM_START}{m['role']}\n{m['content'].strip()}{IM_END}\n" for m in turns)
    if turns[-1]["role"] != "assistant":
        text += f"{IM_START}assistant\n"
    return text


def build_example(
    tok: Tokenizer, messages: Sequence[Message], max_len: int = 1024,
    system: str | None = DEFAULT_SYSTEM,
) -> tuple[list[int], list[int]]:
    """Token ids and labels for one conversation.

    Graded: the content of every assistant turn, plus its closing `<|im_end|>` -- which is what
    teaches the model to stop instead of carrying on and inventing the user's next message.
    Masked: the system prompt, every user turn, and the `<|im_start|>assistant\\n` header, since
    the header is given at generation time rather than predicted.
    """
    turns = list(messages)
    if system and (not turns or turns[0]["role"] != "system"):
        turns = [{"role": "system", "content": system}] + turns

    ids: list[int] = []
    labels: list[int] = []
    for turn in turns:
        header = tok.encode(f"{IM_START}{turn['role']}\n", add_special_tokens=False)
        body = tok.encode(f"{turn['content'].strip()}{IM_END}\n", add_special_tokens=False)
        ids += header + body
        labels += [IGNORE] * len(header)
        labels += body if turn["role"] == "assistant" else [IGNORE] * len(body)

    # Truncating mid-conversation is fine: whatever complete assistant turns survive are still
    # correctly conditioned on everything before them. Only the tail is lost.
    ids, labels = ids[:max_len], labels[:max_len]
    # Shift: labels[i] is the token that should follow ids[i].
    return ids[:-1], labels[1:]


def normalize(record: dict) -> list[Message] | None:
    """Reduce one dataset row to a plain list of turns, or None if it is unusable.

    UltraChat and OASST2 store conversations differently, and both contain rows that are empty,
    single-sided, or start with the assistant. Those train nothing useful, so they are dropped
    here rather than becoming rows of all -100 whose mean loss is NaN.
    """
    turns = record.get("messages") or record.get("conversation") or record.get("conversations")
    if not turns:
        return None

    out: list[Message] = []
    for turn in turns:
        role = turn.get("role") or turn.get("from") or ""
        content = turn.get("content") or turn.get("value") or ""
        role = {"human": "user", "gpt": "assistant", "prompter": "user"}.get(role, role)
        if role not in ("system", "user", "assistant") or not content.strip():
            return None
        out.append({"role": role, "content": content})

    # Drop the leading system turn; one of ours is added consistently instead.
    if out and out[0]["role"] == "system":
        out = out[1:]
    # A conversation has to start with the user and contain something to grade.
    if len(out) < 2 or out[0]["role"] != "user":
        return None
    if not any(m["role"] == "assistant" for m in out):
        return None
    return out


class ChatDataset(Dataset):
    """Tokenized conversations, filtered to those with something left to learn from."""

    def __init__(
        self, records: Iterable[dict], tok: Tokenizer, max_len: int = 1024,
        system: str | None = DEFAULT_SYSTEM,
    ) -> None:
        self.examples: list[tuple[list[int], list[int]]] = []
        self.dropped = 0
        for record in records:
            messages = normalize(record)
            if messages is None:
                self.dropped += 1
                continue
            ids, labels = build_example(tok, messages, max_len, system)
            # After truncation a long first user turn can leave nothing graded at all. Such a row
            # has a mean loss of NaN, which poisons the whole batch.
            if not ids or all(label == IGNORE for label in labels):
                self.dropped += 1
                continue
            self.examples.append((ids, labels))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> tuple[list[int], list[int]]:
        return self.examples[i]


if __name__ == "__main__":
    from llm.console import use_utf8

    use_utf8()

    class FakeTokenizer:
        """One id per character, so the mask is readable directly."""

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [ord(c) for c in text]

    conversation = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hey!"},
        {"role": "user", "content": "what sup"},
        {"role": "assistant", "content": "Not much."},
    ]
    print(render(conversation, system=None))
    print("-" * 76)

    ids, labels = build_example(FakeTokenizer(), conversation, system=None)
    graded = "".join(chr(t) if t != IGNORE else "." for t in labels)
    print("graded positions (dots are masked):\n")
    print(graded.replace("\n", "\\n"))
    print("\nBoth assistant turns are graded, and both closing <|im_end|> markers with them.")
    print("The user's turns are masked -- otherwise the model learns to write your half too.")
