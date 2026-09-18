"""Byte-level BPE: training a vocabulary, and encoding with it.

Note 1 argued against writing this: training a tokenizer teaches compression, not language
modeling. That held until the project chose TinyStories, whose published target (validation loss
1.80, CS336) is only meaningful with the same tokenizer CS336 used -- a 10,000-token BPE trained on
TinyStories itself. Cross-entropy is not comparable across vocabularies, so matching the target
means matching the tokenizer.

Follows the CS336 Assignment 1 specification: byte-level, GPT-2 pre-tokenization, special tokens
as hard boundaries that never enter merge statistics, and ties broken toward the
lexicographically greater pair.

See notes/01_data_and_tokens.md.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex

#: GPT-2's pre-tokenizer, exactly as the CS336 handout gives it. tiktoken's version is written with
#: possessive quantifiers for speed; the two agree on every case tested, but this is the one whose
#: output CS336's reference merges were computed from.
GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

EOT = "<|endoftext|>"

Pair = tuple[bytes, bytes]


# --- pre-tokenization --------------------------------------------------------------------------


def _special_pattern(special_tokens: Iterable[str]) -> regex.Pattern | None:
    """One alternation over the special tokens, longest first.

    Longest first matters when one special is a prefix of another -- `<|endoftext|><|endoftext|>`
    registered alongside `<|endoftext|>` must win where both match, or the longer token is never
    produced.
    """
    specials = sorted(set(special_tokens), key=len, reverse=True)
    if not specials:
        return None
    return regex.compile("|".join(regex.escape(s) for s in specials))


def count_pretokens(
    text: str, special_tokens: Iterable[str] = (), pattern: str = GPT2_PATTERN
) -> Counter[bytes]:
    """Split text into pre-tokens and count them.

    Special tokens are cut out first and treated as hard boundaries: `[doc 1]<|endoftext|>[doc 2]`
    pre-tokenizes the two documents separately, so no merge can ever span a document boundary.
    The special tokens themselves are not counted -- they must not contribute merge statistics.
    """
    compiled = regex.compile(pattern)
    split = _special_pattern(special_tokens)
    segments = split.split(text) if split else [text]

    counts: Counter[bytes] = Counter()
    for segment in segments:
        # finditer rather than findall: on a 2 GB corpus the list of matches would not fit.
        for match in compiled.finditer(segment):
            counts[match.group().encode("utf-8")] += 1
    return counts


def find_chunk_boundaries(path: str | Path, n_chunks: int, split_token: bytes) -> list[int]:
    """Byte offsets that divide a file into roughly equal chunks, each starting at `split_token`.

    Cutting at a document boundary is always safe, because no merge crosses one. A cut anywhere
    else could land mid-word -- or mid-UTF-8-character -- and change the counts.
    """
    size = os.path.getsize(path)
    if n_chunks <= 1 or size == 0:
        return [0, size]

    guesses = [size * i // n_chunks for i in range(n_chunks + 1)]
    window = 1 << 16
    with open(path, "rb") as f:
        for i in range(1, n_chunks):
            pos = guesses[i]
            f.seek(pos)
            while True:
                block = f.read(window)
                if not block:
                    guesses[i] = size
                    break
                found = block.find(split_token)
                if found != -1:
                    guesses[i] = pos + found
                    break
                # Step back by the token length so a token straddling two reads is still found.
                pos += len(block) - len(split_token) + 1
                f.seek(pos)
    return sorted(set(guesses))


def _count_chunk(args: tuple[str, int, int, tuple[str, ...], str]) -> Counter[bytes]:
    """Worker for parallel pre-tokenization. Top-level so Windows' spawn can pickle it."""
    path, start, end, special_tokens, pattern = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="replace")
    return count_pretokens(text, special_tokens, pattern)


def count_file(
    path: str | Path,
    special_tokens: Iterable[str] = (EOT,),
    num_workers: int = 1,
    pattern: str = GPT2_PATTERN,
) -> Counter[bytes]:
    """Pre-tokenize a file, in parallel across document-aligned chunks.

    Pre-tokenization is the bottleneck of BPE training and the only part that parallelizes: the
    merge loop that follows is inherently sequential.
    """
    path = str(path)
    specials = tuple(special_tokens)
    if num_workers <= 1 or not specials:
        return count_pretokens(Path(path).read_text(encoding="utf-8"), specials, pattern)

    split_token = specials[0].encode("utf-8")
    bounds = find_chunk_boundaries(path, num_workers * 4, split_token)
    jobs = [(path, a, b, specials, pattern) for a, b in zip(bounds, bounds[1:])]

    from multiprocessing import Pool

    total: Counter[bytes] = Counter()
    with Pool(num_workers) as pool:
        for counts in pool.imap_unordered(_count_chunk, jobs):
            total.update(counts)
    return total


# --- training ------------------------------------------------------------------------------------


def _pairs(word: tuple[bytes, ...]) -> Iterator[Pair]:
    return zip(word, word[1:])


def _merge_word(word: tuple[bytes, ...], pair: Pair, merged: bytes) -> tuple[bytes, ...]:
    """Replace every non-overlapping occurrence of `pair`, scanning left to right."""
    out: list[bytes] = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def train_from_counts(
    word_counts: dict[tuple[bytes, ...], int],
    vocab_size: int,
    special_tokens: Iterable[str] = (),
    verbose: bool = False,
) -> tuple[dict[int, bytes], list[Pair]]:
    """The merge loop, given pre-token frequencies.

    Separated from pre-tokenization so the CS336 worked example -- which pre-tokenizes by splitting
    on whitespace -- can be tested against the same code that trains the real vocabulary.

    Returns `vocab` (id -> bytes) and `merges` in order of creation. Ids 0-255 are the single
    bytes, merges follow, and special tokens take the last ids, matching GPT-2's convention of
    `<|endoftext|>` as the final id.

    The naive loop recounts every pair after every merge, which is O(merges x corpus) and hours on
    TinyStories. The only counts that change are those overlapping the merged pair, so this keeps
    an index from each pair to the words containing it and updates only those words.
    """
    specials = list(dict.fromkeys(special_tokens))
    n_merges = vocab_size - 256 - len(specials)
    if n_merges < 0:
        raise ValueError(
            f"vocab_size={vocab_size} is smaller than the 256 byte tokens plus "
            f"{len(specials)} special token(s)"
        )

    words = list(word_counts.keys())
    freqs = [word_counts[w] for w in words]

    pair_counts: Counter[Pair] = Counter()
    pair_words: defaultdict[Pair, set[int]] = defaultdict(set)
    for wi, word in enumerate(words):
        for pair in _pairs(word):
            pair_counts[pair] += freqs[wi]
            pair_words[pair].add(wi)

    merges: list[Pair] = []
    for step in range(n_merges):
        if not pair_counts:
            break  # the corpus is fully merged; the vocabulary simply ends up smaller
        # Highest count wins; ties go to the lexicographically greater pair. Comparing the
        # (count, pair) tuple does both in one pass, and is deterministic, so two runs on the same
        # data produce the same vocabulary.
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merged = best[0] + best[1]
        merges.append(best)

        for wi in pair_words.pop(best, ()):
            word = words[wi]
            f = freqs[wi]
            # Retract this word's old pairs, merge, then add its new ones. Only words that
            # contained the merged pair are touched, which is the whole optimization.
            for pair in _pairs(word):
                pair_counts[pair] -= f
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
            new = _merge_word(word, best, merged)
            words[wi] = new
            for pair in _pairs(new):
                pair_counts[pair] += f
                pair_words[pair].add(wi)
        pair_counts.pop(best, None)

        if verbose and (step % 1000 == 0 or step == n_merges - 1):
            shown = merged.decode("utf-8", errors="replace")
            print(f"  merge {step + 1:>6}/{n_merges}  {shown!r}")

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for pair in merges:
        vocab[len(vocab)] = pair[0] + pair[1]
    for s in specials:
        vocab[len(vocab)] = s.encode("utf-8")
    return vocab, merges


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: Iterable[str] = (EOT,),
    num_workers: int = 1,
    pattern: str = GPT2_PATTERN,
    verbose: bool = False,
) -> tuple[dict[int, bytes], list[Pair]]:
    """Train a byte-level BPE vocabulary on a text file. The CS336 `train_bpe` interface."""
    counts = count_file(input_path, special_tokens, num_workers, pattern)
    if verbose:
        print(f"  {len(counts):,} unique pre-tokens, {sum(counts.values()):,} total")
    word_counts = {tuple(bytes([b]) for b in token): n for token, n in counts.items()}
    return train_from_counts(word_counts, vocab_size, special_tokens, verbose)


# --- the tokenizer -------------------------------------------------------------------------------


class Tokenizer:
    """Encodes and decodes with a trained vocabulary and merge list.

    Interchangeable with `llm.tokenizer.BPETokenizer`: same `encode`, `decode`,
    `encode_documents`, `vocab_size` and `eot_id`, so the dataset and training code take either.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Pair],
        special_tokens: Iterable[str] = (),
        pattern: str = GPT2_PATTERN,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.pattern = pattern
        self._compiled = regex.compile(pattern)

        self._byte_to_id = {token: i for i, token in self.vocab.items()}
        for s in dict.fromkeys(special_tokens):
            encoded = s.encode("utf-8")
            if encoded not in self._byte_to_id:
                new_id = max(self.vocab, default=-1) + 1
                self.vocab[new_id] = encoded
                self._byte_to_id[encoded] = new_id
        self.special_tokens = list(dict.fromkeys(special_tokens))
        self._special_ids = {s: self._byte_to_id[s.encode("utf-8")] for s in self.special_tokens}
        self._special_split = _special_pattern(self.special_tokens)
        self._max_special = max((len(s) for s in self.special_tokens), default=0)

        # Rank = position in the merge list. Encoding applies the lowest-ranked pair present,
        # which reproduces training's merge order.
        self._ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        # Most text repeats the same few thousand words, so memoizing per pre-token turns the
        # expensive merge loop into a dict lookup for almost every word after the first page.
        self._cache: dict[bytes, list[int]] = {}

    # -- interface shared with BPETokenizer ---------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def eot_id(self) -> int:
        if EOT not in self._special_ids:
            raise AttributeError(f"this tokenizer has no {EOT} special token")
        return self._special_ids[EOT]

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """Encode text to ids.

        `allow_special` has the same meaning as in BPETokenizer: off, a literal `<|endoftext|>`
        in the input is encoded as ordinary text, so a document cannot inject a boundary.
        """
        ids: list[int] = []
        for kind, piece in self._segments(text, allow_special):
            if kind == "special":
                ids.append(self._special_ids[piece])
            else:
                for match in self._compiled.finditer(piece):
                    ids.extend(self._encode_pretoken(match.group().encode("utf-8")))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """Concatenate the bytes and decode.

        An arbitrary id sequence need not form valid UTF-8 -- a token can be half of a multi-byte
        character -- so malformed bytes become U+FFFD rather than raising.
        """
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def encode_documents(self, docs: Iterable[str]) -> Iterator[int]:
        for doc in docs:
            yield from self.encode(doc)
            yield self.eot_id

    # -- streaming ------------------------------------------------------------------------------

    #: How far past the end of a match the pre-tokenizer may look when deciding where the match
    #: ends. GPT-2's pattern needs 2 -- rejecting the contraction `'ll` at an apostrophe means
    #: reading the next two characters -- so 16 is a wide margin for any sane pattern.
    LOOKAHEAD = 16

    def encode_iterable(self, iterable: Iterable[str], allow_special: bool = False) -> Iterator[int]:
        """Encode a stream lazily -- a file handle, say -- in memory that does not grow with it.

        A chunk boundary can fall anywhere, and there are two ways that goes wrong.

        Re-encoding a prefix on its own changes its last pre-token. GPT-2 splits the whitespace
        run in `"\\n\\n  \\n\\nmat"` into `"\\n\\n  \\n"` + `"\\n"` because it sees the `m` next;
        encoded in isolation, with nothing after it, the same run becomes one token. So units are
        emitted exactly as segmented in the whole buffer, never by re-tokenizing a truncated string.

        A unit near the end is not final yet. `'l` at the end of a chunk is punctuation plus a
        letter, but `'ll` once the next chunk arrives. So nothing is emitted within LOOKAHEAD
        characters of the end, nor within reach of a special token that might still be incomplete.
        """
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            hold = self._max_special - 1 if allow_special and self._max_special else 0
            safe = len(buffer) - hold - self.LOOKAHEAD

            cut = 0
            for start, end, kind, piece in self._units(buffer, allow_special):
                if end > safe:
                    break
                if kind == "special":
                    yield self._special_ids[piece]
                else:
                    yield from self._encode_pretoken(piece.encode("utf-8"))
                cut = end
            buffer = buffer[cut:]
        if buffer:
            yield from self.encode(buffer, allow_special)

    def _units(self, text: str, allow_special: bool) -> Iterator[tuple[int, int, str, str]]:
        """Every pre-token and special token, with its span, in order."""
        pos = 0
        for kind, piece in self._segments(text, allow_special):
            if kind == "special":
                yield pos, pos + len(piece), kind, piece
            else:
                for match in self._compiled.finditer(piece):
                    yield pos + match.start(), pos + match.end(), "text", match.group()
            pos += len(piece)

    # -- internals ----------------------------------------------------------------------------

    def _segments(self, text: str, allow_special: bool) -> Iterator[tuple[str, str]]:
        """Alternating plain-text and special-token pieces, in order."""
        if not allow_special or self._special_split is None:
            if text:
                yield "text", text
            return
        pos = 0
        for match in self._special_split.finditer(text):
            if match.start() > pos:
                yield "text", text[pos : match.start()]
            yield "special", match.group()
            pos = match.end()
        if pos < len(text):
            yield "text", text[pos:]

    def _encode_pretoken(self, token: bytes) -> list[int]:
        """Apply merges to one pre-token: repeatedly merge the lowest-ranked pair present.

        Merging every occurrence of that pair at once is equivalent to merging them one at a time,
        because any pair involving a freshly merged token was itself created later and so ranks
        higher -- it can never jump the queue ahead of the remaining occurrences.
        """
        cached = self._cache.get(token)
        if cached is not None:
            return cached

        parts: list[bytes] = [bytes([b]) for b in token]
        while len(parts) > 1:
            best_rank = None
            best: Pair | None = None
            for pair in zip(parts, parts[1:]):
                rank = self._ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best = rank, pair
            if best is None:
                break
            parts = list(_merge_word(tuple(parts), best, best[0] + best[1]))

        ids = [self._byte_to_id[p] for p in parts]
        self._cache[token] = ids
        return ids

    # -- persistence --------------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """One JSON file. Bytes are stored as latin-1 strings: each byte maps to exactly one code
        point, so the round trip is exact, and ASCII tokens stay readable in the file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "llm-bpe-v1",
            "pattern": self.pattern,
            "special_tokens": self.special_tokens,
            "vocab": {str(i): t.decode("latin-1") for i, t in self.vocab.items()},
            "merges": [[a.decode("latin-1"), b.decode("latin-1")] for a, b in self.merges],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "llm-bpe-v1":
            raise ValueError(f"{path} is not a tokenizer saved by llm.bpe")
        vocab = {int(i): t.encode("latin-1") for i, t in payload["vocab"].items()}
        merges = [(a.encode("latin-1"), b.encode("latin-1")) for a, b in payload["merges"]]
        return cls(vocab, merges, payload["special_tokens"], payload["pattern"])


# --- command line ------------------------------------------------------------------------------


def main() -> None:
    import argparse
    import time

    from llm.console import use_utf8

    use_utf8()
    p = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    p.add_argument("input", help="UTF-8 text file; documents separated by <|endoftext|>")
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--out", default=None,
                   help="default: data/<input name>-<vocab size>.json, so runs never overwrite each other")
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = p.parse_args()

    start = time.perf_counter()
    vocab, merges = train_bpe(
        args.input, args.vocab_size, [EOT], num_workers=args.workers, verbose=True
    )
    elapsed = time.perf_counter() - start

    tok = Tokenizer(vocab, merges, [EOT])
    # A fixed default name would let a quick test run silently overwrite a real trained vocabulary.
    out = tok.save(args.out or Path("data") / f"{Path(args.input).stem}-{args.vocab_size}.json")
    # Special tokens are excluded: `<|endoftext|>` is always the longest, and says nothing about
    # what the merges learned.
    longest = max((a + b for a, b in merges), key=len, default=b"")
    print(f"\ntrained {len(vocab):,} tokens ({len(merges):,} merges) in {elapsed:.1f}s")
    print(f"longest learned token: {longest.decode('utf-8', errors='replace')!r} ({len(longest)} bytes)")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
