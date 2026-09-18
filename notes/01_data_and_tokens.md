# Note 1 — Data and tokens

Turning text into batches of integers the model can consume. Unglamorous, and the source of more
silent bugs than the model itself.

**Produces:** `src/llm/tokenizer.py`, `src/llm/bpe.py`, `src/llm/dataset.py`,
`tests/test_01_data.py`, `tests/test_bpe.py`

---

## 1. The problem

Neural networks multiply matrices. They cannot consume `"The cat sat"`. Something has to map text
to integers, and that mapping has to be reversible.

The obvious answers both fail:

**Split on words.** `"The cat sat" → [1, 2, 3]`. Then the model meets `"cats"` — not in the vocab.
Or `"Cat"`. Or a typo. English has a long tail of hundreds of thousands of word forms, and a fixed
word vocabulary meets unknown words constantly, at which point everything becomes `<|unk|>` and
information is destroyed. Worse, the model can never learn that *cat* and *cats* are related; they
are unrelated integers.

**Split on characters.** No unknowns ever, tiny vocabulary of ~100. But now "cat" costs 3 steps
instead of 1, sequences get 4–5× longer, attention cost grows quadratically with length, and the
model burns its early layers learning to reassemble letters into words.

## 2. BPE: the compromise

Byte-Pair Encoding starts from bytes and *learns* which pairs to merge, driven by frequency.

The algorithm, in full:

```
1. Start with a vocabulary of all 256 byte values.
2. Count every adjacent pair in the corpus.
3. Take the most frequent pair, merge it into a new single token, add it to the vocab.
4. Repeat until the vocabulary reaches the target size (50,257 for GPT-2).
```

Worked on a tiny corpus — `"low low low lower lowest"`:

```
start:     l o w _ l o w _ l o w _ l o w e r _ l o w e s t
"lo" is most frequent (5x)  ->  merge
           lo w _ lo w _ lo w _ lo w e r _ lo w e s t
"low" now most frequent (5x) -> merge
           low _ low _ low _ low e r _ low e s t
"low_" (4x) -> merge
           low_ low_ low_ low e r _ low e s t
```

The outcome is exactly what you want, and nobody designed it: **common words become single tokens,
rare words split into meaningful pieces.** GPT-2 encodes `" the"` as one token, while
`"unbelievable"` becomes `un|bel|iev|able` — so the model can generalize the `un-` prefix and the
`-able` suffix to words it has never seen. There are no unknown tokens ever, because the fallback is
raw bytes.

Notice `" the"` with the leading space. GPT-2's BPE attaches the space to the *following* word, so
`"the"` at the start of a sentence and `" the"` mid-sentence are **different tokens**. This trips
everyone up at least once, usually when hand-building a prompt and wondering why output quality
collapsed.

## 3. Two tokenizers, and why we ended up writing a trainer

**The original argument, kept because it was reasonable.** An earlier version of this section said
we would not write a BPE trainer: the merge loop teaches data compression rather than language
modeling, a subtle bug in it degrades everything downstream silently, and `tiktoken` — GPT-2's
actual vocabulary — was already available and buys the GPT-2 weight-loading oracle.

**What changed.** The project chose TinyStories as its first real corpus, because CS336 publishes a
target for it: validation loss **1.80**. That number was measured with a specific tokenizer — a
10,000-token BPE trained on TinyStories itself. And **cross-entropy is not comparable across
vocabularies**: the same text split into different tokens produces a different loss, because each
token carries a different amount of information. With GPT-2's 50,257-token vocabulary our loss would
be on a different scale, and the target would tell us nothing. Matching the target means matching
the tokenizer. The trainer stopped being optional.

It also turned out to be the largest efficiency win available. At `emb_dim=512` a 50,257-row
embedding table is 67% of the `small` model; a 10,000-row one is 29% of `tinystories`. No
architecture change in the roadmap comes close.

**So there are now two, used for different things:**

| | `tokenizer.BPETokenizer` | `bpe.Tokenizer` |
|---|---|---|
| Vocabulary | GPT-2's 50,257, via `tiktoken` | trained by us, any size |
| Used for | loading GPT-2 weights, the oracle | training our own models |
| Trainer | none needed | `llm.bpe.train_bpe` |

Both have the same interface, so the dataset and training code take either.

**How the trainer is verified.** A tokenizer bug is exactly the silent kind, so it is checked three
independent ways:

1. **CS336's worked examples** — the exact merge sequence `s t, e st, o w, l ow, w est, n e, ...`,
   `newest → [ne, west]`, and `"the cat ate" → [9, 7, 1, 5, 10, 3]`.
2. **Against a naive reference.** The fast trainer only updates pair counts for words touched by
   each merge; a from-scratch version recounts everything every round. They must agree exactly.
3. **Against tiktoken.** GPT-2's merge list can be recovered from tiktoken's ranks. Given those
   50,000 merges, our encoder produces **the same ids as tiktoken** on arbitrary text — CJK, emoji,
   Burmese, code, whitespace runs. That proves the encoding logic against a production tokenizer.

**One bug this caught worth knowing about.** Encoding a stream in chunks is not the same as
encoding each chunk and concatenating. GPT-2's whitespace rule splits `"\n\n   \n\nmat"` as
`"\n\n   \n"` + `"\n"` because it can see the `m` coming — encode the prefix alone, with nothing
after it, and the same run becomes one token. Contractions have the same problem: `'l` at the end of
a chunk is punctuation plus a letter, `'ll` once the next chunk arrives. Measured, naive
split-and-re-encode was wrong at 10 of 23 cut points on one test string and 13 of 25 on another.
`encode_iterable` now emits tokens as segmented in the whole buffer and holds back a margin, and a
test tries every possible cut position.

**The Burmese point still stands, more strongly.** GPT-2's BPE has essentially no Burmese merges,
so it falls back to raw bytes — **about 3 tokens per character**, against 0.25 for English, a 12×
cost on context, compute and memory. With the trainer in place, that is now a one-command fix:
train a vocabulary on Burmese text. See [DATA_GUIDE](../docs/DATA_GUIDE.md).

I will still write the naive word-level tokenizer first — about 40 lines — because feeling it break
on an unseen word is worth more than reading that it does.

## 4. Special tokens

Tokens that are not text but instructions to the model.

- `<|endoftext|>` (id **50256**) — the document boundary. Without it, concatenated documents run
  into each other and the model learns that recipes follow legal contracts. It is also GPT-2's
  padding token and its stop signal at generation time.
- `<|unk|>` — needed by the naive tokenizer, never by BPE.

`50257 = 50000 learned merges + 256 base bytes + 1 special token`. That is the whole vocabulary.

## 5. The sliding window

Here is where "predict the next token" becomes an actual tensor.

Given a token stream, an input is a window and the target is **the same window shifted right by
one**:

```
ids:      [464, 3797, 3332,  319,  262, 2603]
           the   cat   sat   on   the   mat

context length = 4

x       = [464, 3797, 3332,  319]      the cat sat on
targets = [3797, 3332,  319,  262]     cat sat on the
           ^     ^      ^      ^
           |     |      |      +-- given "the cat sat on", predict "the"
           |     |      +--------- given "the cat sat",    predict "on"
           |     +---------------- given "the cat",        predict "sat"
           +---------------------- given "the",            predict "cat"
```

One pair of tensors, four training examples. `targets = x` shifted by one, and **that one-line
relationship is the entire supervised signal in pretraining.** Get the off-by-one wrong in either
direction and the model either learns to copy its input (loss near zero, output useless) or to
predict two ahead (loss stuck high). Both are tested.

**Stride** controls the gap between consecutive windows:

- `stride == context_length` — no overlap. Every token used exactly once. Efficient. Use this.
- `stride < context_length` — windows overlap. More examples from the same text, but they are
  correlated, and tokens near the start of a window get seen many times with different context.
  Mild data augmentation on a small corpus; wasted compute on a large one.

## 6. Embeddings

Token id 3797 means nothing numerically — it is not "more" than 3796. So each id indexes into a
learned table:

```
nn.Embedding(50257, 768)    # a (50257, 768) matrix
```

Row 3797 is the 768-dimensional vector meaning *cat*. Initially random; training moves these
vectors until related words end up near each other. Nobody specifies that — it emerges, because
words used in similar contexts benefit from similar predictions.

`nn.Embedding` is **a lookup, not a matrix multiply.** Conceptually it is a one-hot vector times the
matrix; in practice it is indexing, which is why an embedding table can be 24% of the parameters
while costing 0% of the FLOPs.

## 7. Position: attention's blind spot

Attention computes a weighted sum over positions. A sum does not care about order. Without
position information, these are **literally identical** to the model:

```
"the dog bit the man"
"the man bit the dog"
```

Same tokens, same multiset, same output. That is not a small inaccuracy — it destroys language.

GPT-2's fix is a second learned table, `nn.Embedding(context_length, 768)`, giving one vector per
*slot*, added to the token embedding:

```
h = tok_emb(x) + pos_emb(arange(T))
```

Adding is odd the first time you see it — position and meaning, summed into the same 768 numbers?
It works because 768 dimensions is plenty of room, and training arranges for the two signals to
occupy directions that do not interfere much.

This scheme has two real flaws, both fixed in note 7 by RoPE:

1. **Hard context limit.** Position 1025 has no row in a table with 1024 rows. The model cannot run
   on longer sequences at all.
2. **Absolute, not relative.** It learns "position 5" rather than "three tokens back", so a pattern
   learned at position 5 does not transfer to position 500. Language cares about relative distance.

## 8. What gets built

```python
# tokenizer.py
class SimpleTokenizer:                 # naive, for the lesson
    def __init__(self, vocab: list[str]) -> None
    def encode(self, text: str) -> list[int]
    def decode(self, ids: list[int]) -> str

class BPETokenizer:                    # tiktoken wrapper, the real one
    eot_id: int = 50256
    vocab_size: int = 50257
    def encode(self, text: str, allow_special: bool = False) -> list[int]
    def decode(self, ids: list[int]) -> str
    def encode_documents(self, docs: Iterable[str]) -> list[int]   # joins with <|endoftext|>

# dataset.py
class SlidingWindowDataset(Dataset):
    def __init__(self, ids: np.ndarray, context_length: int, stride: int) -> None

def make_dataloader(ids, context_length, stride, batch_size, shuffle=True) -> DataLoader
def train_val_split(docs: list[str], val_fraction: float = 0.01) -> tuple[list[str], list[str]]
def tokenize_corpus(in_dir: Path, out_path: Path, tok: BPETokenizer) -> dict   # -> uint16 memmap
```

`tokenize_corpus` writes a flat `uint16` array to disk (50257 fits in 16 bits) and returns a report
— token count, compression ratio, most frequent tokens. Tokenize once, memory-map forever: the
corpus never needs to fit in RAM and epochs do not pay the cost again.

## Tests

| Test | Catches |
|---|---|
| `decode(encode(t)) == t` over tricky strings (unicode, emoji, whitespace runs) | tokenizer asymmetry |
| batch shapes are `(B, T)`, dtype `long` | wrong reshape |
| **`targets[:, i] == x[:, i+1]`** for all i | the off-by-one, the most damaging bug here |
| `stride == T` produces zero overlap across consecutive windows | window arithmetic |
| dataset length matches `(len(ids) - T - 1) // stride + 1` | silent truncation, or a last window that runs off the end |
| `" the"` and `"the"` encode differently | that the leading-space rule is understood |
| val split shares no *document* with train | leakage that makes validation loss meaningless |

## Demo

`scripts/demo_01.py` prints: the naive tokenizer failing on an unseen word; the same word under
BPE, split into subwords; a real batch with `x` and `targets` aligned side by side so the shift is
visible; and the token counts for English vs Burmese on the same sentence, so the tokenizer
decision is concrete rather than theoretical.

---

Next: **[note 2 — attention](02_attention.md)**.
