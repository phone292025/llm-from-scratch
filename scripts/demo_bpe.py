"""BPE demo - watch a vocabulary get learned from raw bytes.

    python scripts/demo_bpe.py

CPU only, a few seconds. Trains on the tiny story in data/raw/.
"""

from __future__ import annotations

from collections import Counter
from itertools import pairwise
from pathlib import Path

from llm.bpe import EOT, Tokenizer, count_pretokens, train_from_counts
from llm.console import use_utf8
from llm.tokenizer import BPETokenizer

CORPUS = Path("data/raw/the-verdict.txt")


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "-" * 76)


def show(token: bytes) -> str:
    return repr(token.decode("utf-8", errors="replace"))


def train(words: dict, size: int) -> Tokenizer:
    vocab, merges = train_from_counts(words, size, [EOT])
    return Tokenizer(vocab, merges, [EOT])


def main() -> None:
    use_utf8()
    if not CORPUS.exists():
        raise SystemExit(f"missing {CORPUS} - see notes/05_training.md")
    text = CORPUS.read_text(encoding="utf-8")
    raw = len(text.encode("utf-8"))

    # 1 -------------------------------------------------------------------------------------
    rule("1. Pre-tokenization: the corpus becomes a table of word counts")
    counts = count_pretokens(text, [EOT])
    words = {tuple(bytes([b]) for b in t): n for t, n in counts.items()}
    print(f"  {raw:,} bytes -> {sum(counts.values()):,} pre-tokens, {len(counts):,} distinct\n")
    for token, n in counts.most_common(8):
        print(f"    {n:>4} x {show(token)}")
    print("\n  BPE never looks at the text again. Every merge decision comes from this table,")
    print("  which is why training scales with distinct words rather than corpus size.")

    # 2 -------------------------------------------------------------------------------------
    rule("2. The first merges it discovers")
    tok = train(words, 256 + 1 + 20)
    for i, (a, b) in enumerate(tok.merges, 1):
        print(f"  {i:>3}. {show(a):>10} + {show(b):<10} -> {show(a + b)}")
    print("\n  Nobody told it about spaces or 'the'. The most frequent adjacent pair wins every")
    print("  round, and English's most frequent pieces surface on their own.")

    # 3 -------------------------------------------------------------------------------------
    rule("3. Compression grows with the vocabulary")
    print(f"  {'vocab':>7} {'merges':>7} {'tokens':>8} {'bytes/token':>12}")
    for size in (257, 300, 500, 1000, 2000):
        t = train(words, size)
        n = len(t.encode(text))
        print(f"  {len(t.vocab):>7,} {len(t.merges):>7,} {n:>8,} {raw / n:>12.2f}")
    print("\n  No merges means one byte per token -- exactly 1.00. Each merge makes some common")
    print("  pair cost one token instead of two, with fast diminishing returns.")

    # 4 -------------------------------------------------------------------------------------
    rule("4. How one word splits as the vocabulary grows")
    for word in (" portrait", " Gisburn", " exterminating"):
        print(f"\n  {word!r}")
        for size in (257, 500, 1000, 2000):
            t = train(words, size)
            pieces = [t.vocab[i].decode("utf-8", errors="replace") for i in t.encode(word)]
            print(f"    vocab {size:>5,}: {pieces}")
    print("\n  Common words collapse to one token early; rare ones stay in reusable pieces.")

    # 5 -------------------------------------------------------------------------------------
    rule("5. Ties break toward the lexicographically greater pair")
    sennrich = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    sw = {tuple(bytes([b]) for b in w.encode()): n for w, n in Counter(sennrich.split()).items()}
    print("  First-round pair counts on CS336's worked example:")
    pairs: Counter = Counter()
    for w, n in sw.items():
        for p in pairwise(w):
            pairs[p] += n
    for p, n in pairs.most_common(4):
        print(f"    {p[0].decode()}{p[1].decode()}: {n}")
    _, merges = train_from_counts(sw, 257, [])
    print(f"\n  'es' and 'st' tie at 9. First merge: {merges[0][0].decode()} {merges[0][1].decode()}")
    print("  Deterministic tie-breaking is what makes two runs on the same data produce the same")
    print("  vocabulary -- and ours reproduce CS336's merge sequence exactly.")

    # 6 -------------------------------------------------------------------------------------
    rule("6. Against GPT-2's tokenizer, on this story")
    ours = train(words, 2000)
    gpt2 = BPETokenizer()
    n_ours, n_gpt2 = len(ours.encode(text)), len(gpt2.encode(text))
    print(f"  {'':<22} {'vocab':>8} {'tokens':>8} {'bytes/token':>12}")
    print(f"  {'ours, trained here':<22} {len(ours.vocab):>8,} {n_ours:>8,} {raw / n_ours:>12.2f}")
    print(f"  {'GPT-2':<22} {gpt2.vocab_size:>8,} {n_gpt2:>8,} {raw / n_gpt2:>12.2f}")
    print("\n  Within 7% of GPT-2 with a vocabulary 25x smaller -- but read that as flattering. Ours")
    print("  is being scored on the very text it was trained on. ' Gisburn' became a single token")
    print("  at vocab 500 because the story repeats a character's name, not because it is common")
    print("  English. On unseen text the gap would be wider.")
    print("\n  The underlying trade is still real: a small vocabulary fitted to a corpus spends its")
    print("  tokens on that corpus's words. That is why the tinystories preset uses 10,000 tokens")
    print("  trained on TinyStories, and why its embedding table is 29% of the model, not 67%.")

    # 7 -------------------------------------------------------------------------------------
    rule("7. Round trip")
    sample = "I HAD always thought Jack Gisburn rather a cheap genius--café, naïve."
    ids = ours.encode(sample)
    print(f"  {sample!r}\n  -> {len(ids)} ids -> {ours.decode(ids)!r}")
    print(f"  exact: {ours.decode(ids) == sample}")
    print("\n  Byte-level, so nothing is ever unknown: 'café' and 'naïve' were never merged, but")
    print("  their raw bytes are in every vocabulary by construction.")


if __name__ == "__main__":
    main()
