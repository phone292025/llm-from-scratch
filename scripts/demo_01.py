"""Note 1 demo - what tokenization actually does to text.

    python scripts/demo_01.py
"""

from __future__ import annotations

import numpy as np

from llm.console import use_utf8
from llm.dataset import SlidingWindowDataset, tokenize_corpus
from llm.tokenizer import BPETokenizer, SimpleTokenizer

SAMPLE = (
    "I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it "
    "was no great surprise to me to hear that, in the height of his glory, he had dropped "
    "painting, married a rich widow, and established himself in a villa on the Riviera."
)


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "-" * 72)


def main() -> None:
    use_utf8()
    bpe = BPETokenizer()

    # 1 -------------------------------------------------------------------------------------
    rule("1. Why a fixed word vocabulary fails")
    simple = SimpleTokenizer.from_text(SAMPLE)
    print(f"vocabulary built from the sample: {simple.vocab_size} tokens\n")
    for phrase in ("a cheap genius", "a cheap wizard"):
        ids = simple.encode(phrase)
        print(f"  {phrase!r:18} -> {ids}  -> {simple.decode(ids)!r}")
    print("\n'wizard' was never seen, so it becomes <|unk|>. The information is gone -- and with a")
    print("real vocabulary this happens constantly, on names, typos and plurals alike.")

    # 2 -------------------------------------------------------------------------------------
    rule("2. BPE has no unknown tokens: rare words decompose")
    for word in ("genius", "wizard", "unbelievable", "Gisburn", "tokenization"):
        ids = bpe.encode(" " + word)
        pieces = [bpe.decode([i]) for i in ids]
        print(f"  {word:<14} {len(ids)} token(s)  {pieces}")
    print("\nCommon words are one token; rare ones split into reusable pieces, so the model can")
    print("generalize the 'un-' prefix and the '-able' suffix to words it has never seen.")

    # 3 -------------------------------------------------------------------------------------
    rule("3. The leading space is part of the token")
    for text in ("the", " the", "The", " The"):
        print(f"  {text!r:8} -> {bpe.encode(text)}")
    print("\nFour different tokens. Building a prompt with the wrong one quietly hurts quality.")

    # 4 -------------------------------------------------------------------------------------
    rule("4. The sliding window: targets are inputs shifted by one")
    ids = np.fromiter(bpe.encode("The cat sat on the mat"), dtype=np.uint16)
    ds = SlidingWindowDataset(ids, context_length=4, stride=4)
    x, y = ds[0]
    print(f"  x       {x.tolist()}  {[bpe.decode([i]) for i in x.tolist()]}")
    print(f"  targets {y.tolist()}  {[bpe.decode([i]) for i in y.tolist()]}\n")
    for i in range(len(x)):
        given = bpe.decode(x[: i + 1].tolist())
        print(f"    given {given!r:22} -> predict {bpe.decode([y[i]])!r}")
    print("\nOne window of 4 tokens is 4 training examples. That is why pretraining on plain")
    print("text is so efficient: the text is its own label.")

    # 5 -------------------------------------------------------------------------------------
    rule("5. The tokenizer is tuned for English - other scripts pay for it")
    samples = {
        "English": "The cat sat on the mat and looked at the rain.",
        "Code": "def add(a, b):\n    return a + b\n",
        "Burmese": "ကြောင်သည် ဖြစ်သည်",
    }
    print(f"  {'language':<10} {'chars':>6} {'tokens':>7} {'chars/token':>12}")
    for name, text in samples.items():
        n = len(bpe.encode(text))
        print(f"  {name:<10} {len(text):>6} {n:>7} {len(text) / n:>12.2f}")
    print("\nBurmese falls back to raw bytes, costing roughly 12x more tokens per character than")
    print("English. That shrinks the effective context window and wastes model capacity on")
    print("reassembling bytes. See docs/DATA_GUIDE.md.")

    # 6 -------------------------------------------------------------------------------------
    rule("6. Corpus report")
    report = tokenize_corpus([SAMPLE], "data/processed/demo.bin", bpe)
    for key in ("documents", "characters", "tokens", "chars_per_token", "unique_tokens"):
        value = report[key]
        print(f"  {key:<16} {value:.2f}" if isinstance(value, float) else f"  {key:<16} {value}")
    print(f"  {'top_tokens':<16} {[t for t, _ in report['top_tokens'][:5]]}")


if __name__ == "__main__":
    main()
