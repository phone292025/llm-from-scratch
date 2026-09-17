"""Tests for note 1 - tokenization and batching.

The one that matters most is test_targets_are_inputs_shifted_by_one. Get that wrong and the model
either learns to copy its input (loss near zero, output useless) or to predict two ahead (loss
stuck high), with no error either way.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llm.dataset import (
    SlidingWindowDataset,
    load_tokens,
    make_dataloader,
    tokenize_corpus,
    train_val_split,
)
from llm.tokenizer import EOT, UNK, BPETokenizer, SimpleTokenizer

TRICKY = [
    "Hello, world!",
    "   leading and trailing   ",
    "unicode: éàü 中文 \U0001f600",
    "newlines\nand\ttabs",
    "numbers 3.14159 and 1,000,000",
    'punctuation!? "quoted" (parens) -- dashes',
    "",
]


@pytest.fixture(scope="module")
def bpe() -> BPETokenizer:
    try:
        return BPETokenizer()
    except ImportError as exc:  # tiktoken's native extension can be blocked by Windows SAC
        pytest.skip(f"tiktoken unavailable: {exc}")


@pytest.fixture(scope="module")
def ids() -> np.ndarray:
    """A token stream for the dataset tests.

    Deliberately synthetic rather than tokenizer-produced: the sliding-window logic has nothing
    to do with the tokenizer, and keeping them independent means these tests still run when the
    tokenizer cannot be imported.
    """
    rng = np.random.default_rng(0)
    return rng.integers(0, 50257, size=500, dtype=np.uint16)


# --- tokenizer ------------------------------------------------------------------------------


@pytest.mark.parametrize("text", TRICKY)
def test_bpe_round_trips_exactly(bpe: BPETokenizer, text: str) -> None:
    assert bpe.decode(bpe.encode(text)) == text


def test_bpe_vocab_size(bpe: BPETokenizer) -> None:
    assert bpe.vocab_size == 50257
    assert bpe.eot_id == 50256


def test_leading_space_is_part_of_the_token(bpe: BPETokenizer) -> None:
    # The single most common source of hand-built-prompt bugs.
    assert bpe.encode("the") != bpe.encode(" the")
    assert len(bpe.encode(" the")) == 1


def test_bpe_splits_unknown_words_into_subwords(bpe: BPETokenizer) -> None:
    # No <|unk|> ever: the fallback is bytes, so rare words decompose rather than vanish.
    pieces = bpe.encode("unbelievable")
    assert len(pieces) > 1
    assert "".join(bpe.decode([i]) for i in pieces) == "unbelievable"


def test_eot_is_text_unless_allowed(bpe: BPETokenizer) -> None:
    # A document containing the literal string must not be able to inject a document boundary.
    assert bpe.encode(EOT) != [bpe.eot_id]
    assert bpe.encode(EOT, allow_special=True) == [bpe.eot_id]


def test_encode_documents_separates_with_eot(bpe: BPETokenizer) -> None:
    out = list(bpe.encode_documents(["one", "two"]))
    assert out.count(bpe.eot_id) == 2
    assert out[-1] == bpe.eot_id


def test_simple_tokenizer_maps_unseen_words_to_unk() -> None:
    tok = SimpleTokenizer.from_text("the cat sat on the mat")
    assert tok.encode("cat")[0] != tok.unk_id
    assert tok.encode("wizard") == [tok.unk_id]
    assert tok.decode(tok.encode("wizard")) == UNK


# --- dataset --------------------------------------------------------------------------------


def test_targets_are_inputs_shifted_by_one(ids: np.ndarray) -> None:
    """The entire supervised signal of pretraining. If this breaks, nothing else matters."""
    ds = SlidingWindowDataset(ids, context_length=8, stride=8)
    for i in range(len(ds)):
        x, y = ds[i]
        assert torch.equal(x[1:], y[:-1]), f"window {i} is misaligned"


def test_batch_shapes_and_dtype(ids: np.ndarray) -> None:
    loader = make_dataloader(ids, context_length=16, stride=16, batch_size=4, shuffle=False)
    x, y = next(iter(loader))
    assert x.shape == y.shape == (4, 16)
    # int64: nn.Embedding and cross_entropy both require long indices.
    assert x.dtype is torch.int64


def test_stride_equal_to_context_gives_no_overlap(ids: np.ndarray) -> None:
    ds = SlidingWindowDataset(ids, context_length=8, stride=8)
    x0, _ = ds[0]
    x1, _ = ds[1]
    assert torch.equal(x1, torch.from_numpy(ids[8:16].astype(np.int64)))
    assert not torch.equal(x0, x1)


def test_stride_smaller_than_context_overlaps(ids: np.ndarray) -> None:
    ds = SlidingWindowDataset(ids, context_length=8, stride=4)
    x0, _ = ds[0]
    x1, _ = ds[1]
    assert torch.equal(x0[4:], x1[:4])


def test_length_formula(ids: np.ndarray) -> None:
    for context_length, stride in [(4, 4), (8, 8), (8, 3), (16, 1)]:
        ds = SlidingWindowDataset(ids, context_length, stride)
        assert len(ds) == (len(ids) - context_length - 1) // stride + 1


def test_last_window_stays_in_bounds(ids: np.ndarray) -> None:
    # An off-by-one here raises only on the final item, so it survives casual testing.
    ds = SlidingWindowDataset(ids, context_length=7, stride=5)
    x, y = ds[len(ds) - 1]
    assert len(x) == len(y) == 7


def test_rejects_corpus_shorter_than_one_window() -> None:
    with pytest.raises(ValueError, match="at least"):
        SlidingWindowDataset(np.arange(4, dtype=np.uint16), context_length=8, stride=8)


def test_index_out_of_range_raises(ids: np.ndarray) -> None:
    ds = SlidingWindowDataset(ids, context_length=8, stride=8)
    with pytest.raises(IndexError):
        ds[len(ds)]


# --- corpus pipeline ------------------------------------------------------------------------


def test_val_split_shares_no_document_with_train() -> None:
    docs = [f"document number {i}" for i in range(100)]
    train, val = train_val_split(docs, val_fraction=0.1)
    assert len(train) + len(val) == len(docs)
    assert len(val) == 10
    assert not (set(train) & set(val))


def test_val_split_is_deterministic() -> None:
    docs = [f"doc {i}" for i in range(50)]
    assert train_val_split(docs, seed=0) == train_val_split(docs, seed=0)
    assert train_val_split(docs, seed=0) != train_val_split(docs, seed=1)


def test_tokenize_corpus_round_trips_through_disk(tmp_path, bpe: BPETokenizer) -> None:
    docs = ["the cat sat on the mat", "the dog sat on the log"]
    out = tmp_path / "corpus.bin"
    report = tokenize_corpus(docs, out, bpe)

    loaded = load_tokens(out)
    assert len(loaded) == report["tokens"]
    assert loaded.dtype == np.uint16
    assert list(loaded) == list(bpe.encode_documents(docs))


def test_tokenize_corpus_reports_compression(tmp_path, bpe: BPETokenizer) -> None:
    # ~4 chars per token is the expected ratio for English. A much lower number means the
    # vocabulary is a poor fit for this text - see docs/DATA_GUIDE.md.
    docs = ["The quick brown fox jumps over the lazy dog. " * 20]
    report = tokenize_corpus(docs, tmp_path / "c.bin", bpe)
    assert 3.0 < report["chars_per_token"] < 6.0
    assert report["documents"] == 1
