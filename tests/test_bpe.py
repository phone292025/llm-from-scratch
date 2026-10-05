"""Tests for the BPE trainer and tokenizer.

Three independent oracles, strongest last:

1. CS336's worked examples - exact merge sequences and encodings from the handout.
2. The optimized trainer against a naive reference that recounts every pair after every merge.
   The incremental bookkeeping is the part most likely to be subtly wrong, and this pins it.
3. The encoder, given GPT-2's merges recovered from tiktoken's ranks, against tiktoken itself.
   Same ids on arbitrary text means the encoding logic matches a production implementation.
"""

from __future__ import annotations

import random
from collections import Counter
from itertools import pairwise

import pytest

from llm.bpe import (
    EOT,
    GPT2_PATTERN,
    Tokenizer,
    count_file,
    count_pretokens,
    find_chunk_boundaries,
    train_bpe,
    train_from_counts,
)

SENNRICH = (
    "low low low low low lower lower widest widest widest "
    "newest newest newest newest newest newest"
)
SENNRICH_MERGES = [
    "s t", "e st", "o w", "l ow", "w est", "n e",
    "ne west", "w i", "wi d", "wid est", "low e", "lowe r",
]

TRICKY = [
    "Hello, world!",
    "   leading and trailing   ",
    "tabs\tand\nnewlines\n\n\n   indented",
    "unicode: café naïve Straße 中文 \U0001f600",
    "I'll we've they're it's",
    "numbers 3.14159 and 1,000,000",
    "a" * 50,
    "",
]


def _as_words(text: str) -> dict[tuple[bytes, ...], int]:
    """Whitespace pre-tokenization, as the CS336 worked example uses."""
    return {tuple(bytes([b]) for b in w.encode()): n for w, n in Counter(text.split()).items()}


def _naive_train(word_counts: dict, n_merges: int) -> list:
    """The textbook algorithm: recount every pair, every merge. Slow and obviously correct."""
    words = dict(word_counts)
    merges = []
    for _ in range(n_merges):
        pairs: Counter = Counter()
        for word, n in words.items():
            for pair in pairwise(word):
                pairs[pair] += n
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))
        merges.append(best)
        rewritten: dict = {}
        for word, n in words.items():
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    out.append(best[0] + best[1])
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            rewritten[tuple(out)] = rewritten.get(tuple(out), 0) + n
        words = rewritten
    return merges


def _generate_corpus(seed: int = 0, n_words: int = 4000) -> str:
    """Deterministic pseudo-English, built in the test so it exists wherever the tests run.

    The corpus under data/ is gitignored, so fixtures read from there would skip in CI -- which
    would quietly disable the naive-reference oracle, the one check on the trainer's bookkeeping.
    Word frequencies follow a rough Zipf curve so merges are learned in a realistic order, and the
    mix of punctuation, contractions, numbers, whitespace runs and non-ASCII exercises every branch
    of the pre-tokenizer.
    """
    rng = random.Random(seed)
    words = (
        "the a and to of was she he it in said little they her his with for on that "
        "day cat dog tree sun happy big friend played went saw very time one there "
        "could would because then wanted help found home mom dad garden ball bird"
    ).split()
    weights = [1 / (i + 1) for i in range(len(words))]
    extras = ["I'll", "don't", "it's", "café", "naïve", "3.14", "1,000", "2024", "Straße", "日本"]
    out = []
    for _ in range(n_words):
        w = rng.choice(extras) if rng.random() < 0.03 else rng.choices(words, weights)[0]
        if not out or out[-1].endswith((".", "!", "?")):
            w = w.capitalize()
        out.append(w + rng.choice([""] * 12 + [",", ".", "!", "?", ";"]))
        if rng.random() < 0.02:
            out.append(rng.choice(["\n\n", "\n", "   ", "\t"]))
    return " ".join(out)


@pytest.fixture(scope="module")
def corpus() -> str:
    return _generate_corpus()


@pytest.fixture(scope="module")
def verdict() -> str:
    """Real prose, when it has been downloaded. Only the extra tiktoken check uses it."""
    from pathlib import Path

    path = Path("data/raw/the-verdict.txt")
    if not path.exists():
        pytest.skip("data/raw/the-verdict.txt not downloaded")
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def trained(corpus: str) -> Tokenizer:
    counts = count_pretokens(corpus, [EOT])
    words = {tuple(bytes([b]) for b in t): n for t, n in counts.items()}
    vocab, merges = train_from_counts(words, 600, [EOT])
    return Tokenizer(vocab, merges, [EOT])


# --- oracle 1: CS336 worked examples ------------------------------------------------------------


def test_sennrich_merge_sequence_matches_cs336() -> None:
    vocab, merges = train_from_counts(_as_words(SENNRICH), 256 + 1 + 12, [EOT])
    assert [f"{a.decode()} {b.decode()}" for a, b in merges] == SENNRICH_MERGES


def test_sennrich_vocabulary_after_six_merges() -> None:
    vocab, _ = train_from_counts(_as_words(SENNRICH), 256 + 1 + 6, [EOT])
    learned = [t for i, t in sorted(vocab.items()) if 256 <= i < 262]
    assert learned == [b"st", b"est", b"ow", b"low", b"west", b"ne"]
    assert EOT.encode() in vocab.values()


def test_newest_encodes_as_ne_west() -> None:
    vocab, merges = train_from_counts(_as_words(SENNRICH), 256 + 1 + 6, [EOT])
    tok = Tokenizer(vocab, merges, [EOT], pattern=r"\S+")
    assert [tok.vocab[i] for i in tok.encode("newest")] == [b"ne", b"west"]


def test_cs336_encoding_example() -> None:
    vocab = {
        0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t",
        6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at",
    }
    merges = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]
    assert Tokenizer(vocab, merges).encode("the cat ate") == [9, 7, 1, 5, 10, 3]


def test_ties_break_toward_the_lexicographically_greater_pair() -> None:
    # ('e','s') and ('s','t') tie at 9 on the first round; ('s','t') is greater, so it goes first.
    # No special tokens, so vocab_size 257 leaves room for exactly one merge.
    _, merges = train_from_counts(_as_words(SENNRICH), 257, [])
    assert merges == [(b"s", b"t")]


# --- oracle 2: the optimized trainer against the naive one ---------------------------------------


def test_incremental_trainer_matches_naive_reference(corpus: str) -> None:
    """The index-and-update bookkeeping must reproduce the textbook algorithm exactly."""
    counts = count_pretokens(corpus, [EOT])
    words = {tuple(bytes([b]) for b in t): n for t, n in counts.items()}
    n = 250
    _, fast = train_from_counts(words, 256 + n, [])
    assert fast == _naive_train(words, n)


def test_incremental_trainer_matches_naive_on_adversarial_repeats() -> None:
    """Runs of one byte are where overlapping pairs and stale index entries bite."""
    words = {
        tuple(bytes([b]) for b in w.encode()): n
        for w, n in [("aaaa", 5), ("aaa", 3), ("aaaaaaa", 2), ("abab", 4), ("ababab", 1), ("ba", 6)]
    }
    _, fast = train_from_counts(words, 256 + 20, [])
    assert fast == _naive_train(words, 20)


# --- oracle 3: the encoder against tiktoken ------------------------------------------------------


@pytest.fixture(scope="module")
def gpt2_pair():
    tiktoken = pytest.importorskip("tiktoken")
    enc = tiktoken.get_encoding("gpt2")
    ranks = enc._mergeable_ranks

    merges = []
    for token, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
        if len(token) == 1:
            continue
        # Each multi-byte token is the merge of exactly two lower-ranked tokens; recover them by
        # running BPE on it with only the merges that existed before it.
        parts = [bytes([b]) for b in token]
        while True:
            best = None
            for i in range(len(parts) - 1):
                r = ranks.get(parts[i] + parts[i + 1])
                if r is not None and r < rank and (best is None or r < best[0]):
                    best = (r, i)
            if best is None:
                break
            i = best[1]
            parts[i : i + 2] = [parts[i] + parts[i + 1]]
        merges.append((parts[0], parts[1]))

    mine = Tokenizer({r: t for t, r in ranks.items()}, merges, pattern=enc._pat_str)
    return mine, enc


@pytest.mark.parametrize(
    "text",
    TRICKY
    + [
        "The capital of France is Paris.",
        "unbelievable antidisestablishmentarianism",
        "code: def f(x): return x**2  # comment",
        "Burmese: ကြောင်",
        "mixed\r\nline\rendings",
        " " * 20,
    ],
)
def test_encoder_matches_tiktoken_on_gpt2_merges(gpt2_pair, text: str) -> None:
    mine, enc = gpt2_pair
    assert mine.encode(text) == enc.encode(text, disallowed_special=())


def test_encoder_matches_tiktoken_on_real_prose(gpt2_pair, verdict: str) -> None:
    mine, enc = gpt2_pair
    assert mine.encode(verdict) == enc.encode(verdict, disallowed_special=())


# --- special tokens ------------------------------------------------------------------------------


def test_special_tokens_are_hard_boundaries_during_training() -> None:
    """No merge may span a document boundary.

    'ab' occurs only across the boundary here, so it must never be learned -- whereas without the
    boundary it would be the most frequent pair.
    """
    text = f"a{EOT}b" * 50 + "cd " * 10
    counts = count_pretokens(text, [EOT])
    words = {tuple(bytes([b]) for b in t): n for t, n in counts.items()}
    _, merges = train_from_counts(words, 256 + 1 + 5, [EOT])
    assert (b"a", b"b") not in merges


def test_special_tokens_do_not_enter_merge_statistics() -> None:
    counts = count_pretokens(f"hello{EOT}world{EOT}", [EOT])
    assert all(b"<|" not in t and b"|>" not in t for t in counts)
    assert counts[b"hello"] == 1 and counts[b"world"] == 1


def test_special_token_encodes_as_one_id_when_allowed(trained: Tokenizer) -> None:
    ids = trained.encode(f"one{EOT}two", allow_special=True)
    assert ids.count(trained.eot_id) == 1


def test_special_token_is_plain_text_unless_allowed(trained: Tokenizer) -> None:
    # Same safety rule as BPETokenizer: a document cannot inject a boundary.
    ids = trained.encode(EOT)
    assert trained.eot_id not in ids
    assert trained.decode(ids) == EOT


def test_overlapping_special_tokens_prefer_the_longest() -> None:
    double = EOT + EOT
    vocab, merges = train_from_counts(_as_words(SENNRICH), 256 + 2 + 3, [EOT, double])
    tok = Tokenizer(vocab, merges, [EOT, double])
    ids = tok.encode(f"x{double}y{EOT}z", allow_special=True)
    specials = [tok.vocab[i] for i in ids if tok.vocab[i] in (EOT.encode(), double.encode())]
    assert specials == [double.encode(), EOT.encode()]


def test_eot_takes_the_last_id() -> None:
    vocab, _ = train_from_counts(_as_words(SENNRICH), 256 + 1 + 4, [EOT])
    assert vocab[max(vocab)] == EOT.encode()


# --- encoding properties -------------------------------------------------------------------------


@pytest.mark.parametrize("text", TRICKY)
def test_round_trip_is_exact(trained: Tokenizer, text: str) -> None:
    assert trained.decode(trained.encode(text)) == text


def test_decode_replaces_malformed_utf8(trained: Tokenizer) -> None:
    # A lone continuation byte is not valid UTF-8 on its own.
    lone = trained._byte_to_id[bytes([0x80])]
    assert trained.decode([lone]) == "\ufffd"


def test_trained_tokenizer_compresses_its_own_corpus(trained: Tokenizer, corpus: str) -> None:
    """With no merges every token is one byte, so bytes/token is exactly 1.0.

    600 tokens trained on this 20 KB story measures 2.37. A real 10k vocabulary on TinyStories
    should land far higher; this only checks that training bought something substantial.
    """
    raw_bytes = len(corpus.encode("utf-8"))
    assert raw_bytes / len(trained.encode(corpus)) > 2.0


def test_encoding_is_deterministic(trained: Tokenizer, corpus: str) -> None:
    assert trained.encode(corpus) == trained.encode(corpus)


# --- streaming -----------------------------------------------------------------------------------


@pytest.mark.parametrize("allow_special", [False, True])
@pytest.mark.parametrize("seed", range(8))
def test_encode_iterable_matches_encode_under_any_chunking(
    trained: Tokenizer, seed: int, allow_special: bool
) -> None:
    """A chunk boundary can land mid-word, mid-whitespace run or mid-special-token.

    Encoding the pieces naively and concatenating gives a different answer whenever the cut splits
    a pre-token, so this cuts randomly and demands the streaming result match the whole-text one.
    """
    rng = random.Random(seed)
    pieces = ["the cat", " sat", "   ", "\n\n", "  on", EOT, "mat.", "\t", "café", "!!"]
    text = "".join(rng.choice(pieces) for _ in range(80))
    cuts = sorted(rng.sample(range(1, len(text)), k=min(25, len(text) - 1)))
    chunks = [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)], strict=True)]

    expected = trained.encode(text, allow_special=allow_special)
    assert list(trained.encode_iterable(chunks, allow_special=allow_special)) == expected


@pytest.mark.parametrize("allow_special", [False, True])
@pytest.mark.parametrize(
    "text",
    [
        # GPT-2's `\s+(?!\S)` splits this run as "\n\n   \n" + "\n" because it sees the `m`.
        # Re-encoding a prefix that ends inside the run, with nothing after it, merges it into one
        # token instead -- the first bug this test caught.
        "the cat\n\n   \n\nmat.  on",
        # `'ll` is only a contraction once both l's have arrived; cut after `'l` and a naive
        # streamer emits `'` as punctuation.
        "I'll we've they're don't",
        # A special token split across chunks.
        f"story one.{EOT}story two.",
        "tabs\t\tand   spaces\n",
    ],
)
def test_encode_iterable_survives_every_single_cut(
    trained: Tokenizer, text: str, allow_special: bool
) -> None:
    """Every possible two-chunk split, exhaustively -- random cuts found the first bug by luck."""
    expected = trained.encode(text, allow_special=allow_special)
    for cut in range(len(text) + 1):
        got = list(trained.encode_iterable([text[:cut], text[cut:]], allow_special=allow_special))
        assert got == expected, f"differs when cut at {cut}: {text[:cut]!r} | {text[cut:]!r}"


def test_encode_iterable_accepts_a_file_handle(trained: Tokenizer, tmp_path, corpus: str) -> None:
    path = tmp_path / "v.txt"
    path.write_text(corpus, encoding="utf-8")
    with open(path, encoding="utf-8") as f:
        assert list(trained.encode_iterable(f)) == trained.encode(corpus)


# --- vocabulary size -----------------------------------------------------------------------------


def test_vocab_size_is_exact(corpus: str) -> None:
    counts = count_pretokens(corpus, [EOT])
    words = {tuple(bytes([b]) for b in t): n for t, n in counts.items()}
    # The generated corpus is fully merged at about 480 tokens; sizes past that stop early by
    # design, which is its own test below.
    for size in (257, 300, 400):
        vocab, merges = train_from_counts(words, size, [EOT])
        assert len(vocab) == size
        assert len(merges) == size - 256 - 1


def test_stops_early_when_the_corpus_is_fully_merged() -> None:
    vocab, merges = train_from_counts(_as_words("ab ab ab"), 10_000, [])
    assert merges == [(b"a", b"b")]
    assert len(vocab) == 257


def test_rejects_a_vocab_too_small_for_the_bytes() -> None:
    with pytest.raises(ValueError, match="256 byte tokens"):
        train_from_counts({}, 256, [EOT])


# --- files and persistence -----------------------------------------------------------------------


def test_save_and_load_round_trip(trained: Tokenizer, tmp_path, corpus: str) -> None:
    path = trained.save(tmp_path / "tok.json")
    restored = Tokenizer.load(path)
    assert restored.vocab == trained.vocab
    assert restored.merges == trained.merges
    assert restored.encode(corpus) == trained.encode(corpus)
    assert restored.eot_id == trained.eot_id


def test_train_bpe_reads_a_file(tmp_path, corpus: str) -> None:
    path = tmp_path / "corpus.txt"
    path.write_text(corpus, encoding="utf-8")
    vocab, merges = train_bpe(path, 300, [EOT])
    assert len(vocab) == 300


def test_chunk_boundaries_land_on_document_starts(tmp_path) -> None:
    docs = [f"document number {i} " * (i % 7 + 1) for i in range(200)]
    path = tmp_path / "docs.txt"
    path.write_bytes(EOT.join(docs).encode("utf-8"))
    raw = path.read_bytes()

    bounds = find_chunk_boundaries(path, 8, EOT.encode())
    assert bounds[0] == 0 and bounds[-1] == len(raw)
    for b in bounds[1:-1]:
        assert raw[b : b + len(EOT)] == EOT.encode()


def test_parallel_pretokenization_matches_serial(tmp_path) -> None:
    docs = [f"story {i}: the cat sat on the mat {'again ' * (i % 5)}" for i in range(400)]
    path = tmp_path / "docs.txt"
    path.write_text(EOT.join(docs), encoding="utf-8")
    assert count_file(path, [EOT], num_workers=2) == count_file(path, [EOT], num_workers=1)


# --- drop-in compatibility -----------------------------------------------------------------------


def test_works_with_the_corpus_pipeline(trained: Tokenizer, tmp_path) -> None:
    from llm.dataset import load_tokens, tokenize_corpus

    report = tokenize_corpus(["one story", "another story"], tmp_path / "c.bin", trained)
    ids = load_tokens(tmp_path / "c.bin")
    assert list(ids).count(trained.eot_id) == 2
    assert report["tokens"] == len(ids)


def test_pattern_default_is_the_cs336_one() -> None:
    assert Tokenizer({i: bytes([i]) for i in range(256)}, []).pattern == GPT2_PATTERN
