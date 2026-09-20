# Note 7 — Modernizing the architecture

Everything up to here is the 2019 design. This note replaces four of its components with what a
2024-era model uses, one at a time, measuring each change against the baseline.

**Produces:** `src/llm/rope.py` — and with it `MODERN` becomes buildable, activating RMSNorm,
SwiGLU and GQA, which were written behind config flags in notes 2 and 3 and never exercised.

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

**Correction, found while writing the tests.** An earlier version of this note claimed `MODERN`
and `SMALL` land within 1% of each other. They do not: `MODERN` is **4.8% smaller**, and almost
all of that is GQA, which removes 12 of 16 KV heads in every layer. At the TinyStories
vocabulary the same gap reads **10.3%** (17.84 M -> 16.00 M): the savings are all in the
blocks, and a 10k embedding table no longer dilutes them. Same architecture, very different
number, which is worth knowing before reading the A/B. RoPE (−0.3%, the position
table), RMSNorm (−0.01%, one bias per norm) and SwiGLU (−0.3%, rounding the 8d/3 width to a
multiple of 64) are near-neutral as intended; GQA is not, and was never meant to be — §4 describes
it as a memory trade. The honest framing is that the first three runs are equal-parameter
comparisons and the fourth is not, so a loss *tie* on the GQA run is a win. `test_only_gqa_changes_the_parameter_budget`
pins each of the four savings to its closed form, so this cannot drift again.

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

**The one place it gets subtle: the KV cache.** Decoding feeds one token at a time, so inside
attention `T == 1` on every step. Rotate that token by position 0 — the obvious thing, since it is
the only token in the tensor — and the model believes every token it generates is the start of the
sequence. It still writes fluent text, because the *cached* keys stay correctly rotated relative to
one another. Two rules make it right: rotate by the cache's absolute offset, and rotate **before**
writing to the cache, so each key keeps the angle of the position it was written at. Rotating
afterwards would re-rotate the entire history on every step.

**A claim this note used to make, which is false.** "Without positional information attention is a
set operation, so the model cannot tell word order." True for *one* layer. With two, it is not:
layer 1's output at position `j` summarizes the prefix `0..j`, so the length of that prefix is
visible in the result, and layer 2 can read it. The causal mask is itself a positional signal —
this is the "NoPE" result, that decoder-only models train without any position encoding, just less
well. It matters here because it changes what a broken RoPE would look like: not a model that
ignores order, but one with a weak implicit sense of it. Much harder to notice.
`test_causal_masking_leaks_position_once_there_are_two_layers` measures it.

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

`tests/test_07_rope.py`, 44 tests.

| Test | Catches |
|---|---|
| `the_score_depends_only_on_the_distance` | the central property — slide the window down the sequence and every score is unchanged. If this fails, RoPE is an expensive way of scrambling q and k |
| `rope_actually_encodes_distance` | its counterpart: the identity matrix would pass the test above |
| `rope_preserves_norms` | wrong sin/cos pairing — a rotation must not rescale |
| `position_zero_is_the_identity` | angle indexing off by one |
| `frequencies_span_fast_to_slow` | the geometric progression, checked against the closed form |
| `cached_decoding_matches_uncached_under_rope` | the offset bug — fluent output, wrong model |
| `chunked_prefill_matches_one_pass_under_rope` | the middle case: non-zero offset *and* `T > 1` |
| `the_model_can_tell_word_order` | a positional scheme wired in but doing nothing |
| `one_layer_without_positions_is_genuinely_order_blind` | the control that makes the line above mean something |
| `causal_masking_leaks_position_once_there_are_two_layers` | the false claim corrected in §1 |
| `every_combination_of_components_builds_and_learns` | 24 configs: forward pass *and* one backward, so every parameter that was created gets a gradient |
| `only_gqa_changes_the_parameter_budget` | each of the four savings against its closed form, and the four against their sum |
| `swiglu_pays_for_its_third_matrix` | an unfair FFN comparison (within 2% at d=512, not 1% — 8d/3 does not land on a multiple of 64) |

## Demo

`scripts/demo_07.py` plots all five runs on one chart — baseline, +RoPE, +RMSNorm, +SwiGLU,
+GQA — with parameter counts and tokens/second in the legend. One image summarizing the entire
v1 → v2 transition, and the artifact worth keeping from this project.

---

Back to **[README](../README.md)**.
