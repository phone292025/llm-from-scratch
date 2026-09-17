# Data guide

You said you will supply the training data. This is what I need and why, so you can prepare it
before we get to note 5.

---

## The short version

Drop plain-text files in `data/raw/`. UTF-8. That is all that is strictly required — I will write
the loader around whatever you give me.

```
data/
  raw/           <- your files land here, any structure
  processed/     <- I generate this: tokenized, memory-mapped
```

Then tell me three things:

1. **What language(s)** — this decides the tokenizer, and it is the biggest decision in the project.
2. **What kind of text** — prose, code, chat logs, technical docs, a mixture.
3. **Roughly how much** — the number of megabytes, so we can size the model to it.

---

## How much is enough

The honest numbers. Tokens, not megabytes: English averages **about 4 characters per token**, so
1 MB of English text ≈ 250,000 tokens.

| Corpus | Tokens | What comes out of it |
|---|---|---|
| 100 KB | 25 K | memorizes it. Useful as a correctness check, nothing more |
| 10 MB | 2.5 M | grammatical fragments, no coherence. A real signal that the pipeline works |
| 100 MB | 25 M | recognizable style, sentences that hold together for a clause or two |
| 1 GB | 250 M | genuinely readable short passages from a ~30 M model |
| 2.5 GB | 600 M | the Chinchilla-optimal amount for a 30 M model — the target |

The rule of thumb is **about 20 tokens per parameter**. Under that, the model is undertrained and
extra parameters are wasted; well over it, more data still helps but with diminishing returns.

If you have less than ~10 MB, that is fine — we just build a smaller model to match. The failure
mode to avoid is a large model on a small corpus: it memorizes instead of generalizing, validation
loss rises while training loss falls, and the samples are plagiarism rather than language.

---

## Language matters more than you think

GPT-2's tokenizer was trained almost entirely on English web text. On other scripts it falls back
to encoding raw UTF-8 bytes, which is correct but extremely wasteful.

| Text | Roughly |
|---|---|
| English | ~4 characters per token |
| Code | ~3 characters per token |
| Burmese, Thai, Khmer | **~1 token per byte, so 3 tokens per character** |

For Burmese, that means one word can cost 15–30 tokens where the English equivalent costs 2. The
consequences are concrete and all bad: the effective context window shrinks by roughly 10×,
training costs 10× more compute for the same content, and the model spends its capacity learning
to reassemble bytes into characters rather than learning the language.

**If the corpus is mostly Burmese, training a custom tokenizer is not optional** — it is the single
highest-impact change available, worth more than any architecture improvement in the roadmap. It
moves into v1 and we build a BPE trainer. Tell me early if this is the case; it reshapes the plan.

Mixed Burmese/English has the same problem in proportion to the Burmese share.

---

## Format

**Preferred: one `.txt` file per document.** Simple, inspectable, and easy to shuffle.

Also fine, just say which:

- One large `.txt` with documents separated by blank lines or a marker.
- `.jsonl`, one object per line, with the text under a key you name.
- A folder of `.md`, `.py` or similar — the extension tells me how to handle it.

Documents are joined with the `<|endoftext|>` token between them so the model learns where
documents begin and end, rather than running one into the next.

---

## What I will do to it

The pipeline, so nothing is a surprise:

1. **Read and normalize.** UTF-8, normalize line endings, drop control characters. No lowercasing
   and no punctuation stripping — the model should learn real text, including its capitalization.
2. **Deduplicate.** Near-duplicate documents get removed. This matters more than it sounds:
   duplicated text is memorized rather than learned, and it inflates validation scores if the same
   passage lands on both sides of the split.
3. **Hold out a validation split before tokenizing** — 1% or 10 MB, whichever is smaller, split by
   *document* and never by line. Splitting mid-document leaks the answer across the boundary and
   quietly makes validation loss meaningless.
4. **Tokenize once**, store as a flat `uint16` array (50257 fits in 16 bits) memory-mapped from
   disk. Tokenizing every epoch is a waste, and memory-mapping means the corpus never has to fit
   in RAM.
5. **Report** the token count, the compression ratio, and the top tokens, so we can see what the
   tokenizer made of your text before spending hours training on it.

---

## Things that will cost you a training run

Worth checking before you hand the data over:

- **Duplicated files** — the same book twice teaches the model that book, not the language.
- **Boilerplate** — Gutenberg licence headers, HTML nav bars, repeated footers. The model will
  learn to reproduce them, because they are the most predictable thing in the corpus.
- **Mojibake** — text that was decoded with the wrong encoding at some earlier point. It survives
  tokenization and the model learns the corruption as if it were language.
- **A test set hiding in the training set** — if you want to check the model on specific passages,
  pull them out *before* training.

I will run a check for all four and show you the report before any long run starts.
