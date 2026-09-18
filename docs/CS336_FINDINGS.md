# What CS336 changes about this project

Stanford's [CS336, Language Modeling from Scratch](https://cs336.stanford.edu/) is the same project
as this one, taught at graduate level. I read the lecture slides and the Assignment 1 handout and
pulled out what actually changes our design. Sources are listed at the bottom.

Three things were **wrong** in this repo and are now fixed. Several more were confirmed. A few are
open decisions for later.

---

## 1. Errors this found in our design

### Our `SMALL` model was badly proportioned

Lecture 3 surveys the aspect ratio — `emb_dim / n_layers` — across production models:

| Model | d/L |
|---|---|
| BLOOM | 205 |
| T5 v1.1 | 171 |
| PaLM 540B | 156 |
| GPT-3 / OPT / Mistral / Qwen / OLMo 3 | 128 |
| Llama / Llama 2 | 102 |
| Gemma 3 | 87 |

Essentially everything lives in **100–200**. Our `SMALL` was `384/6 = 64` — too deep and too
narrow, outside the band that every real model sits in.

**Fixed:** `SMALL` is now `emb_dim=512, n_layers=4` → aspect ratio **128**, which is also the exact
shape CS336 uses for its validated TinyStories run.

### The vocabulary was eating the model

At `emb_dim=384` with GPT-2's 50,257-token vocabulary, the embedding table was **64% of all
parameters** — and most of those rows are never used by a simple corpus. Capacity that should be
in the layers sits in a lookup table instead.

CS336 trains a **10,000-token** BPE vocabulary on TinyStories itself. At `d=512` that is the
difference between a 25.7M-parameter embedding table and a 5.1M one.

**Fixed:** added a `TINYSTORIES` preset at `vocab_size=10_000`. Embedding share drops from 67% to
**29%**. This partly reopens the "don't write a BPE trainer" decision in
[note 1](../notes/01_data_and_tokens.md) — see open questions below.

### The SwiGLU width rounding overshot

We rounded the `8/3 × d` target **up** to a multiple of 256. At `d=512` the target is 1365, which
rounds up to 1536 — **12% more parameters** than the GELU baseline, which would have quietly rigged
the v2 comparison in SwiGLU's favour.

**Fixed:** round to the *nearest* multiple of **64** (the alignment tensor cores actually care
about). At `d=512` that gives **1344** — the exact number CS336 specifies for this model size.

---

## 2. Confirmed

Design choices in [ARCHITECTURE.md](ARCHITECTURE.md) that the lecture data backs up:

| Our choice | CS336 |
|---|---|
| Pre-norm | "The one thing everyone agrees on." Only known exception is OPT-350M |
| RMSNorm for v2 | Consensus in Llama, PaLM, Chinchilla, T5. Fewer ops, fewer params, no quality cost |
| SwiGLU for v2 | "Fairly consistent" gains (Shazeer 2020, corroborated by Narang 2020). Llama 1/2/3, PaLM, Mistral, OLMo |
| `d_ff = 4d`, `8/3 d` gated | "Almost always true." Real range 2.67–4.0 |
| `head_dim × n_heads = d_model` | Standard, though the lecture notes it has "low to no validation" |
| **Dropout 0.0** | Older models used 0.1; newer ones rely on weight decay alone. Our call was right |
| Weight decay 0.1, not on norms/biases | Standard. Interestingly its effect is on *optimization dynamics*, not overfitting |
| GQA | "Low/no hit" on quality (Ainslie 2023), unlike MQA which does cost perplexity |
| Vocab 50,257 for English | Monolingual models sit at 30–50k. Multilingual need 100–250k |

One extra: **modern models drop bias terms everywhere**, not just on QKV — for memory and
optimization stability. "Everywhere" takes two flags: `SMALL` sets `qkv_bias=False` *and*
`linear_bias=False`, the second covering the attention output projection and the GELU MLP.
Setting only the first leaves three bias tensors per block in a model documented as having none.
`GPT2_124M` keeps both, since real
GPT-2 has it and weight loading requires it.

---

## 3. A correction to what I told you about scaling

I said "about 20 tokens per parameter" as the target. That is Chinchilla's **compute-optimal**
ratio — the best model for a *fixed training budget*. It is not what you want if you intend to
*use* the model, because then most of your compute goes on inference, and a smaller model trained
longer is cheaper to run.

What real models actually do:

| Model | tokens/param |
|---|---|
| GPT-3 | 2 |
| Chinchilla | 20 |
| Llama 65B | 22 |
| Llama 2 70B | 29 |
| Mistral 7B | 110 |
| **Llama 3 70B** | **215** |

The trend is strongly toward overtraining. For us this is good news: our models are small, so
overtraining is cheap, and there is no reason to stop at 20. [DATA_GUIDE.md](DATA_GUIDE.md) has
been corrected.

---

## 4. The most useful thing: a validated target

CS336's low-resource TinyStories configuration, with a **known** outcome:

```
vocab 10,000   context 256   d_model 512   d_ff 1344   4 layers   16 heads
~17M non-embedding parameters
batch 32 x 5000 steps x 256 context = 40,960,000 tokens
-> validation loss 1.80   (perplexity ~6.0)
```

Their reported runtime is 36 minutes on Apple M4 Max (MPS) and 1h22m on CPU. On an RTX 5070 it
should be well under that.

This matters more than it looks. Until now our note 5 success criterion was "loss goes down and the
text looks better", which is unfalsifiable. Now there is a number: **if our loss does not approach
1.80 on this configuration, our implementation is wrong**, not merely small. Their full run
(327,680,000 tokens) is the stretch target.

---

## 5. Worth adding to the roadmap

Stability tricks from lecture 3 that v1 does not have. None are needed at our scale, but they are
what the field now considers standard:

- **QK-norm** — RMSNorm applied to queries and keys before the attention softmax. Now in Gemma 2/3,
  Qwen 3, OLMo 2/3, DCLM. The strongest recent trend, and cheap to add to `CausalSelfAttention`.
- **Z-loss** — an auxiliary penalty keeping the output softmax normalizer near 1. Used by PaLM,
  Baichuan 2, DCLM, OLMo 2/3. Prevents a class of loss spikes.
- **Logit soft-capping** — `tanh`-bounding the logits. Gemma. May cost quality.
- **Interleaved sliding-window / full attention** — every 4th layer full, the rest local. Gemma 3/4,
  Llama 4, OLMo 3, Command A. The current answer to long context.

Added to [ROADMAP.md](ROADMAP.md) under v2.

---

## 6. Open decisions

**Should we write a BPE trainer after all?** *Resolved: yes, and it is built* — `src/llm/bpe.py`.
The deciding argument was not efficiency but comparability. Choosing TinyStories for its published
1.80 target means matching CS336's tokenizer, because cross-entropy is not comparable across
vocabularies. It reproduces CS336's worked-example merge sequence exactly, and given GPT-2's
merges its encoder matches tiktoken's output token for token. See
[note 1 §3](../notes/01_data_and_tokens.md).

**Do we still want the GPT-2 weight-loading oracle?** Yes — it is unaffected. `GPT2_124M` keeps the
50,257 vocabulary and the bias terms precisely so that it can load real weights. Only the models we
*train* change shape.

---

## Where our project deliberately differs

Not everything in CS336 is right for us:

- **They build the modern architecture directly** (RoPE, RMSNorm, SwiGLU from lecture 1). We build
  GPT-2 first to get the weight-loading oracle, then upgrade and *measure* each change. Their path
  is faster; ours is verifiable and produces an A/B.
- **They forbid `torch.nn` layers** to force from-scratch implementations. We hand-write the same
  layers but keep PyTorch's versions as parity tests — which is strictly better evidence than
  writing them alone.
- **Assignments 2–4 need infrastructure we do not have**: Triton kernels, multi-GPU distributed
  training, a hosted scaling-law API, Common Crawl-scale data. Skip.

---

## Sources

- Course site: https://cs336.stanford.edu/
- Lecture 3, "LM architecture and hyperparameters" (Tatsunori Hashimoto) —
  https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf
- Lecture 9 and 11, scaling laws — `lecture_09.pdf`, `lecture_11.pdf` in the same repo
- Assignment 1 handout —
  https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf
- Lecture videos:
  https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV
