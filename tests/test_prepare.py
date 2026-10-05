"""Tests for corpus preparation -- the step that turned 2 GB of TinyStories into token ids.

`llm.prepare` encodes in parallel across chunks and says the result is "exact, not approximate".
Every training run since has read its output, and until now nothing checked that claim. These do,
along with the resumability it promises: each stage is skipped when its output exists, so an
interrupted run picks up where it stopped.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from llm.bpe import EOT, Tokenizer, find_chunk_boundaries, train_bpe
from llm.dataset import TOKEN_DTYPE
from llm.prepare import encode_file
from llm.prepare import main as prepare_main

WINDOW = 1 << 16  # find_chunk_boundaries reads this much at a time


def _documents(n: int = 300) -> list[str]:
    words = "the cat sat on a mat and saw her friend play with café naïve ball 3.14 I'll go".split()
    return [" ".join(words[(i * 7 + j) % len(words)] for j in range(5 + i % 23)) + "."
            for i in range(n)]


@pytest.fixture(scope="module")
def corpus_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("prep") / "corpus.txt"
    path.write_text(EOT.join(_documents()), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def tokenizer_file(corpus_file):
    vocab, merges = train_bpe(corpus_file, 320, [EOT])
    return Tokenizer(vocab, merges, [EOT]).save(corpus_file.parent / "tok.json")


# --- encoding is exact however it is split ----------------------------------------------------


@pytest.mark.parametrize("workers", [1, 2])
def test_encoding_in_chunks_equals_encoding_the_whole_file(
    corpus_file, tokenizer_file, tmp_path, workers
) -> None:
    """Chunks start at <|endoftext|>, a hard boundary for every pre-token, so no token can span a
    cut. If that ever stopped being true, the corpus would silently differ from the text."""
    out = tmp_path / f"w{workers}.bin"
    report = encode_file(corpus_file, out, tokenizer_file, workers=workers)
    tok = Tokenizer.load(tokenizer_file)
    expected = tok.encode(corpus_file.read_text(encoding="utf-8"), allow_special=True)
    got = np.fromfile(out, dtype=TOKEN_DTYPE)
    assert got.tolist() == expected
    assert report["tokens"] == len(expected)


def test_the_encoded_corpus_decodes_back_to_the_text(corpus_file, tokenizer_file, tmp_path) -> None:
    out = tmp_path / "c.bin"
    encode_file(corpus_file, out, tokenizer_file, workers=1)
    tok = Tokenizer.load(tokenizer_file)
    assert tok.decode(np.fromfile(out, dtype=TOKEN_DTYPE).tolist()) == corpus_file.read_text(
        encoding="utf-8")


def test_document_boundaries_become_the_boundary_token(corpus_file, tokenizer_file, tmp_path) -> None:
    """In the raw corpus <|endoftext|> *is* the separator, so it must become one id, not the
    thirteen characters of text a document could otherwise contain."""
    out = tmp_path / "c.bin"
    encode_file(corpus_file, out, tokenizer_file, workers=1)
    ids = np.fromfile(out, dtype=TOKEN_DTYPE)
    assert int((ids == Tokenizer.load(tokenizer_file).eot_id).sum()) == len(_documents()) - 1


def test_no_temporary_file_is_left_behind(corpus_file, tokenizer_file, tmp_path) -> None:
    encode_file(corpus_file, tmp_path / "c.bin", tokenizer_file, workers=1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["c.bin"]


# --- chunk boundaries on long documents ---------------------------------------------------------


def test_a_boundary_straddling_a_read_window_is_still_found(tmp_path) -> None:
    """The search reads 64 KB at a time from each guessed cut. A separator lying across the end of
    one read must be found by the next, which is why the reader steps back by the separator's
    length. TinyStories documents are short enough that this path never ran in practice."""
    split = EOT.encode()
    tail = 20_000
    first = tail + 2 * WINDOW + 3   # puts the separator at (size // 2) + WINDOW - 5
    data = b"x" * first + split + b"y" * tail
    path = tmp_path / "long.bin"
    path.write_bytes(data)
    guess = len(data) // 2
    assert guess + WINDOW - 5 == first, "fixture arithmetic: separator must straddle the window"
    assert find_chunk_boundaries(path, 2, split) == [0, first, len(data)]


def test_no_separator_after_the_guess_means_one_chunk(tmp_path) -> None:
    path = tmp_path / "one.bin"
    path.write_bytes(b"z" * (3 * WINDOW))
    assert find_chunk_boundaries(path, 4, EOT.encode()) == [0, 3 * WINDOW]


def test_a_long_final_document_does_not_hang_the_search(tmp_path) -> None:
    """The bug this file found: with no separator between a guessed cut and the end of the file,
    the search re-read the last few bytes forever. Any corpus whose last document was longer than
    one chunk froze tokenizer training and corpus encoding, with no error."""
    path = tmp_path / "tail.bin"
    path.write_bytes(b"a" * 1000 + EOT.encode() + b"z" * (4 * WINDOW))
    assert find_chunk_boundaries(path, 4, EOT.encode()) == [0, path.stat().st_size]


# --- the command line, and resuming it ---------------------------------------------------------


def test_prepare_builds_all_three_outputs_then_skips_them(tmp_path, monkeypatch, capsys) -> None:
    """Training the tokenizer and encoding 2 GB both take long enough that an interruption is
    likely, so each stage is skipped when its output already exists."""
    monkeypatch.chdir(tmp_path)
    docs = _documents()
    (tmp_path / "train.txt").write_text(EOT.join(docs[:250]), encoding="utf-8")
    (tmp_path / "valid.txt").write_text(EOT.join(docs[250:]), encoding="utf-8")
    argv = ["prepare", "--name", "tiny", "--train", "train.txt", "--valid", "valid.txt",
            "--vocab-size", "300", "--workers", "1"]

    monkeypatch.setattr(sys, "argv", argv)
    prepare_main()
    outputs = [tmp_path / "data" / "tiny-300.json",
               tmp_path / "data" / "processed" / "tiny-train.bin",
               tmp_path / "data" / "processed" / "tiny-valid.bin"]
    assert all(p.exists() for p in outputs)
    first = [p.read_bytes() for p in outputs]
    capsys.readouterr()

    prepare_main()
    assert capsys.readouterr().out.count("skipping") == 3
    assert [p.read_bytes() for p in outputs] == first
