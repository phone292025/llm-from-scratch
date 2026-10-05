"""Sliding-window batches, and turning a folder of text into a tokenized corpus on disk.

The whole supervised signal of pretraining is one line: targets are inputs shifted by one.

See notes/01_data_and_tokens.md.
"""

from __future__ import annotations

import collections
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from llm.tokenizer import BPETokenizer, TextTokenizer

# 50257 fits in uint16, which halves the size of a tokenized corpus on disk.
TOKEN_DTYPE = np.uint16
MAX_TOKEN_ID = np.iinfo(TOKEN_DTYPE).max  # 65535


class SlidingWindowDataset(Dataset):
    """Windows over a flat token stream.

    Item i is (x, y) where x = ids[start : start+T] and y = ids[start+1 : start+T+1]. One item of
    length T is T training examples, because the model predicts at every position at once.

    stride == context_length means no overlap: every token is used exactly once. A smaller stride
    produces more, but correlated, windows.
    """

    def __init__(self, ids: np.ndarray, context_length: int, stride: int) -> None:
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if len(ids) < context_length + 1:
            raise ValueError(
                f"corpus has {len(ids)} tokens, needs at least {context_length + 1} "
                f"for one window of context_length={context_length}"
            )
        self.ids = ids
        self.context_length = context_length
        self.stride = stride
        # Last valid start leaves room for the target's extra token: start + T + 1 <= len(ids).
        self._n = (len(ids) - context_length - 1) // stride + 1

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= i < self._n:
            raise IndexError(i)
        start = i * self.stride
        end = start + self.context_length
        # int64 because nn.Embedding and cross_entropy both require long indices.
        x = torch.from_numpy(self.ids[start:end].astype(np.int64))
        y = torch.from_numpy(self.ids[start + 1 : end + 1].astype(np.int64))
        return x, y


def make_dataloader(
    ids: np.ndarray,
    context_length: int,
    stride: int,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Batches of (x, y), each of shape (batch_size, context_length).

    drop_last avoids a final short batch, which would otherwise make loss curves jump.
    """
    dataset = SlidingWindowDataset(ids, context_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def read_documents(in_dir: Path, pattern: str = "*.txt") -> list[str]:
    """Read one document per file, newest-independent order (sorted, so runs are reproducible)."""
    paths = sorted(Path(in_dir).rglob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} under {in_dir}")
    return [p.read_text(encoding="utf-8", errors="replace") for p in paths]


def train_val_split(
    docs: list[str], val_fraction: float = 0.01, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Hold out whole documents, never lines.

    Splitting mid-document leaks context across the boundary: the model sees the first half in
    training and is then evaluated on the second half, which it can partly continue from memory.
    Validation loss then looks good for the wrong reason.
    """
    shuffled = list(docs)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
    return shuffled[n_val:], shuffled[:n_val]


def tokenize_corpus(docs: list[str], out_path: Path, tok: TextTokenizer | None = None) -> dict:
    """Tokenize once, store as a flat uint16 array, and report on what came out.

    Tokenizing every epoch is wasted work, and a memory-mapped array means the corpus never has to
    fit in RAM. The report exists so the tokenizer's behaviour on this text is visible before
    spending hours training -- a bad chars-per-token ratio is the signal that the vocabulary is
    wrong for this language.
    """
    tok = tok or BPETokenizer()
    # int64 first, then a checked narrowing. Writing straight into uint16 wraps silently --
    # token 70000 lands on disk as 4464 -- and the corpus is then wrong in a way no later stage
    # can see, because 4464 is a perfectly legal id.
    wide = np.fromiter(tok.encode_documents(docs), dtype=np.int64)
    if wide.size and wide.max() > MAX_TOKEN_ID:
        raise ValueError(
            f"token id {wide.max()} does not fit in {TOKEN_DTYPE.__name__} "
            f"(max {MAX_TOKEN_ID}); widen TOKEN_DTYPE for this tokenizer"
        )
    ids = wide.astype(TOKEN_DTYPE)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ids.tofile(out_path)

    n_chars = sum(len(d) for d in docs)
    counts = collections.Counter(ids.tolist())
    return {
        "path": str(out_path),
        "documents": len(docs),
        "characters": n_chars,
        "tokens": len(ids),
        "chars_per_token": n_chars / max(len(ids), 1),
        "unique_tokens": len(counts),
        "top_tokens": [(tok.decode([i]), n) for i, n in counts.most_common(10)],
    }


def load_tokens(path: str | Path) -> np.ndarray:
    """Memory-map a tokenized corpus. Nothing is read until it is indexed."""
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


if __name__ == "__main__":
    from llm.console import use_utf8

    use_utf8()
    text = "the cat sat on the mat and the dog sat on the log " * 20
    tok = BPETokenizer()
    ids = np.fromiter(tok.encode(text), dtype=TOKEN_DTYPE)

    ds = SlidingWindowDataset(ids, context_length=4, stride=4)
    x, y = ds[0]
    print(f"{len(ids)} tokens -> {len(ds)} windows of 4\n")
    print("x      :", x.tolist(), [tok.decode([i]) for i in x.tolist()])
    print("targets:", y.tolist(), [tok.decode([i]) for i in y.tolist()])
    print("\ntargets is x shifted left by one:")
    for i in range(len(x)):
        print(f"  given {tok.decode(x[: i + 1].tolist())!r:30} predict {tok.decode([int(y[i])])!r}")
