"""Turn a raw text corpus into what training reads: a tokenizer, and token ids on disk.

    python -m llm.prepare --name tinystories \
        --train data/raw/TinyStoriesV2-GPT4-train.txt \
        --valid data/raw/TinyStoriesV2-GPT4-valid.txt \
        --vocab-size 10000

Three stages, each skipped if its output already exists, so an interrupted run -- a crash, a
closed laptop -- resumes where it stopped instead of starting over:

    1. train a BPE vocabulary on the training split     -> data/<name>-<vocab>.json
    2. encode the training split                         -> data/processed/<name>-train.bin
    3. encode the validation split                       -> data/processed/<name>-valid.bin

Encoding runs in parallel across document-aligned chunks. That is exact, not approximate: chunks
start at <|endoftext|>, and special tokens are hard boundaries, so no token can span a cut.
"""

from __future__ import annotations

import argparse
import os
import time
from itertools import pairwise
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from llm.bpe import EOT, Tokenizer, find_chunk_boundaries, train_bpe
from llm.dataset import MAX_TOKEN_ID, TOKEN_DTYPE

_worker_tok: Tokenizer | None = None


def _init_worker(tokenizer_path: str) -> None:
    # Loaded once per process rather than pickled into every job: the tokenizer carries a merge
    # table and a cache that are expensive to ship and would be rebuilt anyway.
    global _worker_tok
    _worker_tok = Tokenizer.load(tokenizer_path)


def _encode_chunk(job: tuple[str, int, int]) -> np.ndarray:
    path, start, end = job
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="replace")
    # allow_special=True: in the raw corpus <|endoftext|> *is* the document separator, so here it
    # must become the boundary token rather than be encoded as literal text.
    ids = np.asarray(_worker_tok.encode(text, allow_special=True), dtype=np.int64)
    if ids.size and ids.max() > MAX_TOKEN_ID:
        raise ValueError(f"token id {ids.max()} does not fit in {TOKEN_DTYPE.__name__}")
    return ids.astype(TOKEN_DTYPE)


def _atomic_write(path: Path, write) -> None:
    """Write to a temporary file, then rename over the target.

    A crash midway leaves the temporary file behind and the target untouched -- never a truncated
    file with the right name, which the skip-if-exists logic would then trust.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def encode_file(
    src: str | Path, dst: str | Path, tokenizer_path: str | Path, workers: int = 4
) -> dict:
    """Encode a text file to a flat uint16 token array on disk."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    bounds = find_chunk_boundaries(src, max(1, workers) * 8, EOT.encode("utf-8"))
    jobs = [(str(src), a, b) for a, b in pairwise(bounds)]

    start = time.perf_counter()
    if workers <= 1:
        _init_worker(str(tokenizer_path))
        parts = [_encode_chunk(j) for j in jobs]
    else:
        with Pool(workers, initializer=_init_worker, initargs=(str(tokenizer_path),)) as pool:
            # imap, not imap_unordered: document order matters. Shuffling happens at training time
            # over windows, and a scrambled file would make it impossible to check against source.
            parts = list(pool.imap(_encode_chunk, jobs))

    ids = np.concatenate(parts) if parts else np.zeros(0, dtype=TOKEN_DTYPE)
    _atomic_write(dst, lambda p: ids.tofile(p))
    elapsed = time.perf_counter() - start
    return {
        "path": str(dst),
        "tokens": int(ids.size),
        "bytes": src.stat().st_size,
        "bytes_per_token": src.stat().st_size / max(ids.size, 1),
        "seconds": elapsed,
    }


def main() -> None:
    from llm.console import use_utf8

    use_utf8()
    p = argparse.ArgumentParser(description="Train a tokenizer and encode a corpus.")
    p.add_argument("--name", required=True, help="prefix for the output files")
    p.add_argument("--train", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    args = p.parse_args()

    tok_path = Path("data") / f"{args.name}-{args.vocab_size}.json"
    outputs = {
        "train": Path("data/processed") / f"{args.name}-train.bin",
        "valid": Path("data/processed") / f"{args.name}-valid.bin",
    }

    print(f"[1/3] tokenizer -> {tok_path}")
    if tok_path.exists():
        print("      exists, skipping")
    else:
        start = time.perf_counter()
        vocab, merges = train_bpe(
            args.train, args.vocab_size, [EOT], num_workers=args.workers, verbose=True
        )
        tok = Tokenizer(vocab, merges, [EOT])
        _atomic_write(tok_path, tok.save)
        longest = max((a + b for a, b in merges), key=len, default=b"")
        print(f"      {len(vocab):,} tokens in {time.perf_counter() - start:.0f}s; "
              f"longest learned: {longest.decode('utf-8', errors='replace')!r}")

    for i, split in enumerate(("train", "valid"), start=2):
        dst = outputs[split]
        print(f"[{i}/3] encode {split} -> {dst}")
        if dst.exists():
            n = dst.stat().st_size // np.dtype(TOKEN_DTYPE).itemsize
            print(f"      exists ({n:,} tokens), skipping")
            continue
        src = args.train if split == "train" else args.valid
        report = encode_file(src, dst, tok_path, args.workers)
        print(f"      {report['tokens']:,} tokens, {report['bytes_per_token']:.2f} bytes/token, "
              f"{report['seconds']:.0f}s")

    print("\nready. train with:")
    print(f"  python -m llm.train --preset {args.name} "
          f"--data {outputs['train']} --val-data {outputs['valid']} --tokenizer {tok_path}")


if __name__ == "__main__":
    main()
