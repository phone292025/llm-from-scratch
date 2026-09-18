# Architecture

The design of the whole system, decided up front so that no chapter has to be rewritten by a later
one. Read this once before starting, then use the shape table as a debugging reference forever.

---

## 1. What the model actually does

An LLM is one function applied over and over:

> given a sequence of tokens, produce a probability distribution over the next token

That is the entire task. Everything else — attention, normalization, residuals — is machinery for
making that one function accurate and trainable. Generation is just calling it in a loop and
feeding its own output back in.

```
  "The cat sat on the"
          |
          | tokenizer                                      <- note 1
          v
  [464, 3797, 3332, 319, 262]                 token ids, shape (T,)
          |
          | batching / sliding window                      <- note 1
          v
  x  (B, T)         targets (B, T) = x shifted left by one
          |
          | token embedding: a (vocab, d) lookup table     <- note 1
          v
  h  (B, T, d)
          |
    +-----+-------------------------------------------+
    |  x N transformer blocks                         |    <- notes 2 and 3
    |                                                 |
    |   h = h + attention( norm(h) )   <- mixes ACROSS positions
    |   h = h + mlp(       norm(h) )   <- thinks WITHIN each position
    |                                                 |
    +-----+-------------------------------------------+
          |
          | final norm
          v
  h  (B, T, d)
          |
          | output head: a (d, vocab) matrix
          v
  logits  (B, T, vocab)    one score per possible next token, at every position
          |
          +-- training:   cross_entropy(logits, targets)      <- note 5
          +-- generation: sample from softmax(logits[:, -1])  <- note 6
```

Two sentences worth memorizing:

- **Attention moves information between positions. The MLP does the thinking at each position.**
  A block does both, and residual connections let the model skip either one when it wants to.
- **The model predicts a next token at every position simultaneously**, not just at the end. A
  sequence of 1024 tokens yields 1024 training signals from one forward pass. That is why
  pretraining on plain text is so efficient, and it is why the causal mask must be exact: position
  5 must never see token 6, or it learns to cheat and then generates nothing.

---

## 2. Shape table

`B` = batch, `T` = sequence length, `d` = embedding dim, `H` = heads, `hd` = d/H = head dim,
`V` = vocab size. Reference model: d=768, H=12, hd=64, V=50257.

| Stage | Object | Shape | Notes |
|---|---|---|---|
| raw | text | `str` | |
| tokenize | ids | `(T,)` | ints in [0, V) |
| batch | x, targets | `(B, T)` | `targets[i] == x[i+1]` |
| token emb | h | `(B, T, d)` | lookup, not matmul |
| + position | h | `(B, T, d)` | learned: added here. RoPE: applied inside attention instead |
| block in | h | `(B, T, d)` | |
| qkv proj | q, k, v | `(B, T, d)` | one fused `Linear(d, 3d)`, then split |
| head split | q, k, v | `(B, H, T, hd)` | `.view(B,T,H,hd).transpose(1,2)` |
| scores | q @ kᵀ | `(B, H, T, T)` | **the memory hog** — quadratic in T |
| weights | softmax(scores/√hd) | `(B, H, T, T)` | every row sums to 1 |
| context | weights @ v | `(B, H, T, hd)` | |
| merge | ctx | `(B, T, d)` | `.transpose(1,2).contiguous().view(B,T,d)` |
| out proj | attn_out | `(B, T, d)` | |
| mlp hidden | — | `(B, T, 4d)` | 3072 when d=768 |
| block out | h | `(B, T, d)` | **same shape in as out — that is why blocks stack** |
| head | logits | `(B, T, V)` | |
| loss | scalar | `()` | flatten to `(B*T, V)` against `(B*T,)` |

The `(B, H, T, T)` score tensor is where the VRAM goes. At B=8, H=12, T=1024 in fp32 that is
**400 MB for one layer's scores** — times 12 layers, times activations kept for the backward pass.
That single line is why context length is expensive, and why FlashAttention (which never
materializes the tensor) matters on an 8 GB card.

---

## 3. The config object — the key design decision

The book hardcodes GPT-2. I am not going to, because I want to change the architecture later
without rewriting the model. Every knob lives in one dataclass:

```python
@dataclass
class ModelConfig:
    vocab_size:     int = 50257
    context_length: int = 1024
    n_layers:       int = 12
    n_heads:        int = 12
    n_kv_heads:     int | None = None    # None = MHA; fewer than n_heads = GQA
    emb_dim:        int = 768
    ffn_hidden:     int | None = None    # None = auto: 4d for GELU, ~8d/3 for SwiGLU
    norm: Literal["layernorm", "rmsnorm"] = "layernorm"
    pos:  Literal["learned", "rope", "none"] = "learned"
    act:  Literal["gelu", "swiglu"] = "gelu"
    attn: Literal["manual", "sdpa"] = "sdpa"
    qkv_bias:    bool  = True            # GPT-2 has it; modern models drop it
    linear_bias: bool  = True            # W_o and the GELU MLP; both flags make a bias-free model
    tie_weights: bool  = True
    dropout:     float = 0.0
```

`qkv_bias` on its own does **not** produce a bias-free model — it governs the QKV projection and
nothing else, leaving a bias on the attention output projection and on both GELU MLP matrices.
That is three tensors per block that a config claiming "no bias" would still be carrying, so the
two flags are separate and `SMALL` sets both.

Three named presets:

| Preset | Params | Settings | Purpose |
|---|---|---|---|
| `GPT2_124M` | 124 M | exactly the 2019 architecture | **loads real OpenAI weights** — the correctness oracle |
| `SMALL` | 38 M | d=512, 4 layers, 16 heads, T=256, no bias | what we train |
| `TINYSTORIES` | 18 M | SMALL with a 10k vocabulary | a corpus-sized vocabulary — see below |
| `MODERN` | 37 M | SMALL + rmsnorm + rope + swiglu + GQA(4) | the v2 architecture — **note 7, not buildable yet** |

`MODERN` is a declaration of intent, not a runnable config. `rope.py` does not exist until note 7,
and a model built with `pos="rope"` before then is bit-identical to one with `pos="none"`: it has
no idea what order its tokens arrived in, and it still trains, still logs a falling loss and still
emits text. `GPTModel.__init__` raises `NotImplementedError` rather than hand back such a model.

`SMALL`'s shape is not arbitrary. Its **aspect ratio** — `emb_dim / n_layers` — is 128, inside the
100–200 band that essentially every production model occupies (BLOOM 205, PaLM 156, GPT-3 128,
Llama 102). An earlier version of this file used d=384 over 6 layers, a ratio of 64, which is too
deep and narrow; see [CS336_FINDINGS.md](CS336_FINDINGS.md).

**Vocabulary size dominates a model this small.** At `d=512`, GPT-2's 50,257-token vocabulary is a
25.7 M-parameter embedding table — 67% of `SMALL`, and most of those rows go unused on a simple
corpus. A 10,000-token vocabulary trained on the corpus itself cuts that to 5.1 M and moves the
capacity into the layers. That is what `TINYSTORIES` is, and it needs a tokenizer trained on the
data rather than GPT-2's.

Because `SMALL` and `MODERN` differ only by flags, they can be trained on identical data and their
loss curves drawn on one chart. **That A/B is the most educational thing in this project**, and a
hardcoded model cannot do it. It is the main reason for the dataclass.

`attn="manual"` is the hand-written softmax version, kept forever as the readable reference and as
the thing tests compare against. `attn="sdpa"` calls `F.scaled_dot_product_attention`, which
dispatches to FlashAttention: identical math, never builds the `(B,H,T,T)` tensor, several times
faster and far lighter on memory. Both must return the same numbers — that is a test.

---

## 4. Module map

Arrows are imports. The graph is acyclic and shallow on purpose: any module can be opened and
understood without first reading the ones above it.

```
config.py     (no imports - pure data)
    |
    +--> tokenizer.py ----> dataset.py
    |
    +--> norms.py   ffn.py   rope.py (note 7)
    |         \       |       /
    |          +--> attention.py
    |                   |
    |                   v
    |                block.py
    |                   |
    |                   v
    +-------------->  model.py
                        |
         +--------------+---------------+
         v              v               v
     generate.py     train.py      gpt2_weights.py
     kv_cache.py     optim.py
                     checkpoint.py
```

| File | Owns | Deliberately does NOT own |
|---|---|---|
| `config.py` | every architecture knob | anything containing a tensor |
| `tokenizer.py` | text ↔ ids, special tokens | batching |
| `dataset.py` | sliding window, train/val split | tokenization |
| `norms.py` | LayerNorm, RMSNorm | where norms are placed (that is `block.py`) |
| `ffn.py` | GELU MLP, SwiGLU MLP | residuals |
| `rope.py` *(note 7)* | rotary angle tables, `apply_rope` | attention math |
| `attention.py` | Q/K/V, causal mask, heads, GQA | the positional scheme (it asks `rope.py`) |
| `block.py` | pre-norm ordering, the two residuals | what the sublayers are |
| `model.py` | embeddings, block stack, head, weight tying | how to sample or train |
| `generate.py` | greedy / temperature / top-k / top-p | the model |
| `train.py` | the loop, eval, logging | the optimizer schedule (that is `optim.py`) |

The rule I am holding myself to: **if a file needs a comment explaining why it also does X, then X
belongs in a different file.**

---

## 5. Where the parameters live

For `GPT2_124M` (d=768, L=12, V=50257, T=1024):

| Component | Formula | Count |
|---|---|---|
| token embedding | V × d | 38,597,376 |
| position embedding | T × d | 786,432 |
| per block — attn qkv | d×3d + 3d | 1,771,776 |
| per block — attn out | d×d + d | 590,592 |
| per block — mlp up | d×4d + 4d | 2,362,368 |
| per block — mlp down | 4d×d + d | 2,360,064 |
| per block — two layernorms | 4d | 3,072 |
| **one block** | | **7,087,872** |
| twelve blocks | | 85,054,464 |
| final layernorm | 2d | 1,536 |
| **total, untied** | | **163,037,184** |
| minus tied output head | V × d | −38,597,376 |
| **total, tied — the "124M"** | | **124,439,808** |

> Raschka's book reports 163,009,536 instead, which is 27,648 lower — exactly 12 layers ×
> 2,304, the QKV bias. His default config sets `qkv_bias=False`. Real GPT-2 *does* have that
> bias, and it is required for weight loading, so our default is `qkv_bias=True` and the numbers
> above are the ones to match. 124,439,808 is also what HuggingFace reports for `gpt2`.

Two things fall out of this table:

- **The embedding table is 24% of the model** and costs zero FLOPs — it is a lookup. Parameter
  count is a poor proxy for compute.
- The name "124M" only works *because* of weight tying: the output head reuses the token embedding
  matrix, transposed. It saves 38 M parameters and it is principled — "the vector meaning *cat*"
  and "the direction that predicts *cat*" ought to be the same vector. Reproducing this exact
  count in code is a test, because it catches a missing bias or a dropped layer instantly.

---

## 6. The 8 GB budget

The GPU has 8,151 MiB, minus a few hundred for Windows. Training memory for `P` parameters under
AdamW in fp32:

```
parameters      4P bytes
gradients       4P
Adam m and v    8P
---------------------
               16P   just to exist, before a single activation
```

| Model | 16P | Verdict on 8 GB |
|---|---|---|
| GPT-2 124M | 2.0 GB | fits, but leaves little for activations → tiny batches → slow and noisy |
| SMALL 30M | 0.5 GB | **comfortable** — room for real batch sizes and long context |

So: **`GPT2_124M` is for loading pretrained weights and doing inference; `SMALL` and `MODERN` are
for training.** Chasing 124 M from scratch on a laptop is a trap — a 30 M model trained on enough
tokens writes better text than a 124 M model starved of them. Rough guide (Chinchilla): about
**20 tokens per parameter**, so 30 M params wants roughly 600 M tokens to be properly fed.

Three mitigations, present from day one rather than bolted on later:

- **bf16 autocast** — halves activation memory. bf16 has fp32's exponent range, so no gradient
  scaler is needed, and Blackwell is fast at it.
- **Gradient accumulation** — a large *effective* batch built from small real ones, making batch
  size a knob independent of memory.
- **FlashAttention via SDPA** — the `(B,H,T,T)` tensor never exists.

---

## 7. Decisions, and why

| Decision | Choice | Reasoning |
|---|---|---|
| Norm placement | **pre-norm**, `h + attn(norm(h))` | post-norm is the 2017 original and needs careful warmup to converge at all; pre-norm leaves a clean gradient path straight through every residual. Every modern model is pre-norm |
| Attention impl | manual **and** SDPA | manual for reading and for tests; SDPA for running |
| Positions | learned now, RoPE in note 7 | learned is three lines and makes the concept obvious, and it is required for GPT-2 weight compatibility. RoPE then arrives as a measurable upgrade |
| QKV bias | on for GPT-2, off elsewhere | GPT-2 has it; later work found it contributes nothing. A config flag, not a rewrite |
| Weight tying | on | saves 24% of parameters, and is conceptually right |
| Dropout | **0.0** | the book uses 0.1. Dropout fights overfitting — it is for small data and many epochs. Pretraining makes roughly one pass over a large corpus, so there is nothing to overfit and dropout only slows learning. Keep the knob, leave it off |
| Tokenizer | `tiktoken` GPT-2 BPE | training a BPE tokenizer is a compression exercise, not a modeling one. Note 1 explains the merge algorithm in full; using a proven implementation removes a whole class of silent bugs |
| Precision | bf16 autocast, fp32 master weights | standard practice, and necessary to fit |

---

## 8. Test strategy

Every note ships tests. Three kinds, in increasing order of what they prove:

1. **Shape tests** — cheap, and they catch most mistakes: wrong transpose, wrong reshape, wrong dim.
2. **Property tests** — attention rows sum to 1; **causality**: perturb token `t+1` and assert the
   output at position `t` is bit-for-bit unchanged (the most important test in the repo — a leak
   here makes training look excellent and generation produce garbage); loss at random init ≈
   ln(V) = 10.82; save → load → forward gives identical logits.
3. **Parity tests** — my manual attention against `F.scaled_dot_product_attention`; my `LayerNorm`
   against `torch.nn.LayerNorm`; my block against a reference forward. These prove correctness
   rather than mere plausibility.

Then the end-to-end oracle: **load real OpenAI GPT-2 124M weights into my own `GPTModel`.** If every
matrix, transpose and norm is right, it writes fluent English; if anything is subtly wrong, it
writes noise. No unit test is that convincing. Loading goes through `safetensors` from HuggingFace
rather than the book's TensorFlow checkpoint path — identical weights, and no TensorFlow install
on Windows.
