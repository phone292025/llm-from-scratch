# Note 7 — Modernizing the architecture

Everything up to here is the 2019 design. This note replaces four of its components with what a
2024-era model uses, one at a time, measuring each change against the baseline.

**Produces:** `src/llm/rope.py`, plus `RMSNorm` and `SwiGLUFeedForward` becoming active

---

## 0. The method is the point

The temptation is to flip every flag at once, see a better loss, and conclude the modern
architecture is better. That teaches nothing, because it does not say *which* change helped or by
how much — and some will help less than expected.

So: **one change at a time, same data, same seed, same token budget, same parameter count.** Plot
each against `SMALL`. A change that does not help gets reverted, and that outcome is a real result,
not a failure.

Holding parameter count constant matters. SwiGLU adds a third matrix, so a naive swap makes the
model bigger and the comparison meaningless. The hidden dimension has to shrink to compensate.

```
runs/
  baseline_small.json
  rope.json
  rope_rmsnorm.json
  rope_rmsnorm_swiglu.json
  modern_gqa.json
```

---

## 1. RoPE — rotary position embeddings

**Replaces:** the learned position embedding table.

Learned absolute positions have two flaws (note 1 §7): a hard limit at `context_length`, and they
encode absolute slot rather than relative distance, so a pattern learned at position 5 does not
transfer to position 500.

RoPE's idea: instead of *adding* position information to the vector, **rotate** the query and key
vectors by an angle proportional to their position.

Take each pair of dimensions as a point in a 2D plane and rotate it by `m·θ`, where `m` is the
position:

```python
# for a pair (x1, x2) at position m with frequency theta
x1' = x1*cos(m·θ) - x2*sin(m·θ)
x2' = x1*sin(m·θ) + x2*cos(m·θ)
```

Different dimension pairs get different frequencies — `θ_i = 10000^(-2i/d)` — so low-index pairs
rotate fast and capture fine local position, while high-index pairs rotate slowly and capture
coarse long-range position. The same geometric-progression idea as sinusoidal encodings, applied as
a rotation rather than an addition.

**Why the rotation makes relative position emerge.** Attention computes `q_m · k_n`. Rotating `q` by
`m·θ` and `k` by `n·θ`, the dot product of two rotated vectors depends only on the *difference of
the angles* — that is, on `(m − n)`. Absolute positions go in; relative distance comes out, for
free, from geometry. That is the elegant part.

Consequences:

- No parameters at all. The table is gone.
- Applied inside attention, to q and k only — never to v, and never to the residual stream.
- Extrapolates beyond the trained length, imperfectly but usably, and can be extended further by
  scaling the frequency base.

Implementation: precompute `cos` and `sin` tables of shape `(max_len, head_dim/2)` once, cache
them, index by position, apply before the score matmul. About 30 lines.

**Expected result:** roughly neutral at short context, clearly better at long context, and one
fewer parameter table. Take this one for the extrapolation, not the loss curve.

## 2. RMSNorm

**Replaces:** LayerNorm. Covered in [note 3 §3](03_transformer_block.md) — drop the mean
subtraction and the bias, keep only re-scaling.

**Expected result:** essentially identical loss, measurably faster, half the norm parameters. The
interesting finding is the negative one — the mean-centering that LayerNorm does was never
contributing anything. Worth confirming yourself rather than believing.

## 3. SwiGLU

**Replaces:** the GELU MLP.

A gated activation. Instead of one projection through a nonlinearity, use two projections and let
one *gate* the other:

```python
# GELU MLP:  two matrices
h = W_down(gelu(W_up(x)))

# SwiGLU:    three matrices
h = W_down(silu(W_gate(x)) * W_up(x))
```

The elementwise multiply is the gate: the network computes a value and, separately, how much of
that value to let through. This is multiplicative rather than additive interaction, which a plain
MLP cannot express at all without depth.

`silu(x) = x · sigmoid(x)`, essentially GELU's smoother cousin.

**The parameter correction that makes the comparison fair:** three matrices instead of two means
the hidden dimension must shrink to keep the count equal. The standard choice is `8d/3` instead of
`4d`, rounded to a multiple of 256 for hardware alignment. For d=768: 3072 → 2048.

**Expected result:** a small but real improvement in loss at equal parameters. It is in Llama,
PaLM and Mistral for a reason.

## 4. Grouped-query attention

**Replaces:** full multi-head attention. Covered in [note 2 §5](02_attention.md).

This one is not about loss — it is about the KV cache, which at generation time is often larger
than the model (note 6 §3). Fewer KV heads divides that memory directly.

**Expected result:** loss slightly worse or unchanged at `n_kv = H/3`, KV cache a third of the size,
generation noticeably faster because less memory moves per step. A deliberate trade, and the right
one for an 8 GB card.

## 5. What to expect overall

Honest expectations, so the results are readable rather than disappointing:

| Change | Loss effect | Real benefit |
|---|---|---|
| RoPE | ~neutral short, better long | no parameter table, extrapolates |
| RMSNorm | ~neutral | faster, simpler |
| SwiGLU | small improvement | better quality per parameter |
| GQA | slightly worse | much smaller KV cache |

**Modern architectures are not dramatically better per parameter.** The gains are single-digit
percentages, and the bulk of the improvement from GPT-2 to today is data quality, data quantity,
and compute — not architecture. Measuring this yourself is the most useful thing in this note,
because it contradicts the impression the field's marketing gives.

The exceptions are structural rather than about loss: RoPE genuinely unlocks long context, and GQA
genuinely unlocks cheap inference. Those matter more than their loss curves suggest.

## 6. Where to go after this

Not in the plan, but the natural next steps once `MODERN` is trained:

- **Sliding-window attention** — attend to the last W tokens only, making cost linear in length.
- **Mixture of experts** — many MLPs, a router picking two per token. More parameters at constant
  compute. Wrong memory profile for 8 GB, but worth understanding.
- **Speculative decoding** — a small model drafts, the big one verifies in parallel. Pure inference
  speed, no quality cost.
- **Instruction tuning** — [ROADMAP](../docs/ROADMAP.md) v3, and the step that turns a text
  completer into something that answers.

## Tests

| Test | Catches |
|---|---|
| RoPE preserves vector norms (it is a rotation) | wrong sin/cos pairing |
| `q_m · k_n` after RoPE depends only on `m − n` | the central property — if this fails, RoPE is not doing its job |
| RoPE at position 0 is the identity | angle indexing off by one |
| SwiGLU at the corrected hidden dim matches the GELU MLP's parameter count within 1% | an unfair comparison |
| every combination of config flags builds and runs a forward pass | the config matrix |
| `MODERN` and `SMALL` have parameter counts within 1% | the comparison being meaningful at all |

## Demo

`scripts/demo_07.py` plots all five runs on one chart — baseline, +RoPE, +RMSNorm, +SwiGLU,
+GQA — with parameter counts and tokens/second in the legend. One image summarizing the entire
v1 → v2 transition, and the artifact worth keeping from this project.

---

Back to **[README](../README.md)**.
