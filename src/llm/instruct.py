"""Instruction data: the prompt format, and turning pairs into supervised examples.

A base model continues text. To make one answer instead, you show it thousands of
request-and-reply pairs in a fixed layout, and it learns that after the reply marker comes a
reply. That is all instruction tuning is.

The one detail that matters: **the loss is computed on the response only.** Grading the model on
its ability to reproduce the question teaches it to write questions, which is the opposite of the
goal, and it wastes most of the gradient on text the user will always supply. Masked positions are
set to -100, which is what cross_entropy ignores by default.

See docs/ROADMAP.md, v3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

#: Positions cross_entropy skips. Everything the model is not being graded on gets this.
IGNORE = -100

WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further "
    "context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
WITHOUT_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


class Tokenizer(Protocol):
    eot_id: int

    def encode(self, text: str) -> list[int]: ...


def build_prompt(instruction: str, input_text: str = "") -> str:
    """The text the model sees before it is expected to answer.

    Fixed wording, always. The model keys off this layout -- at chat time the prompt has to be
    built the same way, or it is being asked a question in a language it was never taught.
    """
    template = WITH_INPUT if input_text.strip() else WITHOUT_INPUT
    return template.format(instruction=instruction.strip(), input=input_text.strip())


def build_example(
    tok: Tokenizer, instruction: str, input_text: str, response: str, max_len: int = 512
) -> tuple[list[int], list[int]]:
    """One training example: token ids, and labels with the prompt masked out.

    Returns `(ids, labels)` already shifted the way the model expects -- `labels[i]` is what
    should follow `ids[i]`, so the caller does not shift again.
    """
    prompt_ids = tok.encode(build_prompt(instruction, input_text))
    answer_ids = tok.encode(response.strip()) + [tok.eot_id]

    ids = (prompt_ids + answer_ids)[:max_len]
    # Everything up to the end of the prompt is context, not a target. The first graded position
    # is the one that predicts the answer's first token, which is the last prompt token.
    labels = ([IGNORE] * len(prompt_ids) + answer_ids)[:max_len]

    # Shift: the model at position i predicts position i+1.
    return ids[:-1], labels[1:]


class InstructionDataset(Dataset):
    """Alpaca-style pairs, tokenized once, kept as lists of varying length."""

    def __init__(self, records: list[dict], tok: Tokenizer, max_len: int = 512) -> None:
        self.examples = [
            build_example(
                tok, r["instruction"], r.get("input", ""), r["output"], max_len
            )
            for r in records
        ]
        # A response truncated to nothing leaves an example with no graded position at all, which
        # contributes a NaN to the batch mean rather than simply nothing.
        self.examples = [e for e in self.examples if any(t != IGNORE for t in e[1])]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> tuple[list[int], list[int]]:
        return self.examples[i]

    @classmethod
    def from_json(cls, path: str | Path, tok: Tokenizer, max_len: int = 512):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), tok, max_len)


def collate(batch: list[tuple[list[int], list[int]]], pad_id: int):
    """Pad a batch to its longest example, not to a fixed width.

    Padding on the right is safe without an attention mask: attention is causal, so a real token
    can never see a pad that comes after it, and every pad position is masked out of the loss.
    """
    width = max(len(ids) for ids, _ in batch)
    x = torch.full((len(batch), width), pad_id, dtype=torch.long)
    y = torch.full((len(batch), width), IGNORE, dtype=torch.long)
    for row, (ids, labels) in enumerate(batch):
        x[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        y[row, : len(labels)] = torch.tensor(labels, dtype=torch.long)
    return x, y


def length_grouped_batches(
    dataset: InstructionDataset, batch_size: int, seed: int = 0
) -> list[list[int]]:
    """Batch examples of similar length together, then shuffle the batches.

    Every batch is padded to its longest member, so a 500-token example dragged in beside a
    50-token one makes the GPU compute ten times more padding than content. Grouping by length
    cuts that waste; shuffling the batch order keeps the training order random.

    The cost is that a batch is no longer an unbiased sample -- short answers train together.
    This is standard practice (HuggingFace calls it group_by_length) and the gradient is still
    an average over 32 real examples; only which 32 changes.
    """
    rng = np.random.default_rng(seed)
    order = sorted(range(len(dataset)), key=lambda i: len(dataset[i][0]))
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches


def split(records: list[dict], val_size: int = 500, seed: int = 0) -> tuple[list, list]:
    """Hold out a validation slice, shuffled so it is not the tail of the file."""
    order = np.random.default_rng(seed).permutation(len(records))
    shuffled = [records[i] for i in order]
    return shuffled[val_size:], shuffled[:val_size]


if __name__ == "__main__":
    from llm.console import use_utf8
    from llm.tokenizer import BPETokenizer

    use_utf8()
    tok = BPETokenizer()
    ids, labels = build_example(
        tok, "Give three tips for staying healthy.", "", "1. Eat well.\n2. Sleep.\n3. Move."
    )
    graded = sum(t != IGNORE for t in labels)
    print(f"{len(ids)} tokens, {graded} of them graded ({graded / len(ids):.0%})\n")
    print(build_prompt("Give three tips for staying healthy.") + "...\n")
    print("the last few positions, as the model sees them:")
    for i in range(len(ids) - 8, len(ids)):
        target = "-" if labels[i] == IGNORE else repr(tok.decode([labels[i]]))
        print(f"  sees {tok.decode([ids[i]])!r:<12} must predict {target}")
    print("\nEverything before the response marker predicts '-': not graded, because the user")
    print("supplies it. The gradient goes entirely into learning to answer.")
