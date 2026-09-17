"""Text <-> token ids.

Two tokenizers. `SimpleTokenizer` is the naive word-level one, kept because watching it fail on an
unseen word is the motivation for everything else. `BPETokenizer` is the real one.

See notes/01_data_and_tokens.md.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

# GPT-2 splits on punctuation and whitespace, keeping the separators as tokens.
_SPLIT = re.compile(r'([,.:;?_!"()\']|--|\s)')

# Rejoining: a space before punctuation is an artifact of the split, so remove it.
_FIX_SPACING = re.compile(r'\s+([,.:;?!"()\'])')

UNK = "<|unk|>"
EOT = "<|endoftext|>"


def _split(text: str) -> list[str]:
    return [tok for piece in _SPLIT.split(text) if (tok := piece.strip())]


class SimpleTokenizer:
    """Word-level tokenizer with a fixed vocabulary.

    Teaching implementation only. Two flaws, both deliberate and both visible in the demo:
    every word outside the vocabulary collapses to <|unk|>, destroying information; and decode is
    lossy, because the original whitespace was thrown away by the split.
    """

    def __init__(self, vocab: list[str]) -> None:
        specials = [t for t in (UNK, EOT) if t not in vocab]
        self.itos: list[str] = list(vocab) + specials
        self.stoi: dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}
        self.unk_id = self.stoi[UNK]

    @classmethod
    def from_text(cls, text: str) -> SimpleTokenizer:
        return cls(sorted(set(_split(text))))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(tok, self.unk_id) for tok in _split(text)]

    def decode(self, ids: list[int]) -> str:
        return _FIX_SPACING.sub(r"\1", " ".join(self.itos[i] for i in ids))


class BPETokenizer:
    """GPT-2 byte-pair encoding, via tiktoken.

    Byte-level, so it round-trips any input exactly and never produces an unknown token. The
    vocabulary is 50257 = 50000 learned merges + 256 base bytes + <|endoftext|>.
    """

    eot_id = 50256
    vocab_size = 50257

    def __init__(self) -> None:
        # Imported here, not at module scope, so that everything else in the package stays
        # usable if tiktoken is unavailable -- its native extension is unsigned, and Windows
        # Smart App Control blocks it.
        import tiktoken

        self._enc = tiktoken.get_encoding("gpt2")

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """Encode text to ids.

        `allow_special` decides what happens to a literal "<|endoftext|>" in the input. Off (the
        default) it is encoded as ordinary text, which is what you want for untrusted input --
        otherwise a document could inject a document boundary into the training stream.
        """
        if allow_special:
            return self._enc.encode(text, allowed_special={EOT})
        # tiktoken's default is to *raise* on a special token rather than encode it as text.
        # disallowed_special=() turns the check off, so "<|endoftext|>" in a document becomes
        # ordinary tokens instead of an injected document boundary.
        return self._enc.encode(text, disallowed_special=())

    def decode(self, ids: Iterable[int]) -> str:
        return self._enc.decode(list(ids))

    def encode_documents(self, docs: Iterable[str]) -> Iterator[int]:
        """Encode documents, separating each with <|endoftext|>.

        Without the separator the model learns that one document flows into the next, and
        generation never stops.
        """
        for doc in docs:
            yield from self.encode(doc)
            yield self.eot_id


if __name__ == "__main__":
    sample = "I HAD always thought Jack Gisburn rather a cheap genius"

    simple = SimpleTokenizer.from_text(sample)
    print(f"SimpleTokenizer vocab: {simple.vocab_size} tokens")
    print("  known  :", simple.encode("Jack rather a genius"))
    print("  unseen :", simple.encode("Jack rather a wizard"), "  <- 'wizard' became <|unk|>")

    bpe = BPETokenizer()
    for word in ("wizard", "unbelievable", "Akwirw ier"):
        ids = bpe.encode(word)
        pieces = [bpe.decode([i]) for i in ids]
        print(f"  {word!r:18} -> {ids}  {pieces}")

    print()
    print(f"{'the'!r:>6} -> {bpe.encode('the')}")
    print(f"{' the'!r:>6} -> {bpe.encode(' the')}   <- leading space is part of the token")
