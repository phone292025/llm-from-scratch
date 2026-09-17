# Note 2 — Attention

The hard one, and the only genuinely new idea in the architecture. Everything else is standard
neural network engineering. Go slowly here; the rest is easy afterwards.

**Produces:** `src/llm/config.py`, `src/llm/attention.py`, `tests/test_02_attention.py`

`config.py` was pulled forward from note 4: `CausalSelfAttention` needs it, and it is pure
data with no imports, so it costs nothing to define early.

---

## 1. The problem attention solves

> "The animal didn't cross the street because **it** was too tired."

To encode *it*, the model must look back at *animal*. Change one word:

> "The animal didn't cross the street because **it** was too wide."

Now *it* refers to *street*. Same sentence shape, same position, different referent — and the only
way to know is to look at the other words and weigh them.

The requirement, then: **every position needs to gather information from other positions, with the
weighting decided by content rather than by position.** Not "look 6 tokens back" — look at whatever
is relevant, wherever it is.

RNNs did this by compressing everything seen so far into a single fixed-size state passed along
step by step. Two fatal problems: the state is a bottleneck (a whole paragraph crushed into one
vector), and the steps are sequential, so nothing parallelizes. Attention fixes both — every
position can reach every other position directly, in one matrix multiply, all positions at once.

## 2. Queries, keys, values

The framing that makes the math obvious. A soft dictionary lookup:

- **Query** — what this position is looking for. *"I'm a pronoun; I need a noun to refer to."*
- **Key** — what this position advertises. *"I'm a singular animate noun."*
- **Value** — what this position actually contributes if attended to.

Compare one query against all keys → similarity scores. Softmax the scores → weights that sum to 1.
Take that weighted average of the values. That is attention, completely.

A Python dict does the hard version of this: one exact key match, all-or-nothing. Attention does
the soft version: every key matches a little, weighted by similarity, and the whole thing is
differentiable — which is the point, because it means the model can *learn* what to look for.

Crucially q, k, v are all computed **from the same input** — hence *self*-attention. Three learned
projections of the same vector, letting each position ask, advertise and offer different things.

## 3. Building it up

Five stages, each one fixing a flaw in the last. Written as five classes and kept in the file, not
deleted, because the progression is the lesson.

### Stage 1 — similarity with no parameters

Take the input vectors and dot each against each:

```python
scores  = x @ x.transpose(-2, -1)      # (T, T)
weights = torch.softmax(scores, dim=-1)
context = weights @ x                  # (T, d)
```

This already works a little: dot product is high for vectors pointing the same way, so similar
tokens attend to each other. But there is nothing to learn — no parameters — and it can only ever
express "attend to things like me". A pronoun does not want another pronoun; it wants a noun.

### Stage 2 — learned projections

Give the model three matrices so it can decide what to look for, separately from what it is:

```python
q = x @ W_q       # (T, d_head)
k = x @ W_k
v = x @ W_v
scores = q @ k.transpose(-2, -1)
```

Now "pronouns seek nouns" is learnable: `W_q` maps a pronoun to a query pointing toward noun-shaped
keys. The asymmetry matters — attention is directional, and `W_q ≠ W_k` is what lets A attend to B
without B attending to A.

### Stage 3 — the scaling, and why it is not cosmetic

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
```

Dot products of two random vectors of dimension `d` have variance proportional to `d`. With
`head_dim = 64`, raw scores land around ±8; with larger dims, larger still.

Feed large values into softmax and it saturates — one weight goes to ~1.0, the rest to ~0.0. The
problem is not the sharpness itself, it is the **gradient**: softmax's derivative vanishes in the
saturated region. Attention freezes at whatever pattern it started with and never learns.

Dividing by `√head_dim` normalizes the variance back to about 1, keeping softmax in its responsive
range. It is one line and training simply does not work without it.

### Stage 4 — the causal mask

The model must not see the future. Set scores for future positions to `-inf` **before** the
softmax:

```python
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = scores.masked_fill(mask, float("-inf"))
weights = torch.softmax(scores, dim=-1)
```

```
        the  cat  sat   on        <- attending to
the    [ .   -inf -inf -inf ]
cat    [ .    .   -inf -inf ]
sat    [ .    .    .   -inf ]
on     [ .    .    .    .   ]
```

**Why `-inf` before softmax rather than zeroing after.** Because `exp(-inf) = 0` and softmax then
renormalizes over the remaining positions, so the surviving weights still sum to exactly 1. Zeroing
after softmax leaves rows summing to less than 1 — the weighted average silently shrinks toward
zero, more so at early positions, and the model quietly breaks in a way that looks like slow
convergence rather than a bug.

The mask is built once and registered as a non-persistent buffer: it is derived from config, not
learned, so it should move with `.to(device)` but stay out of the checkpoint.

**This is the highest-risk code in the project.** A leak here shows up as suspiciously good
training loss and worthless generation. The test perturbs a future token and asserts earlier
outputs are bit-for-bit identical.

### Stage 5 — multiple heads

One attention pattern has to serve every kind of relation at once — grammatical agreement,
coreference, topic — and averaging them serves none well.

So run several in parallel on slices of the dimensions. With `d=768` and `H=12`, each head gets
`hd = 64` dimensions, so total compute is unchanged; the space is just partitioned.

The naive implementation is a loop over 12 separate modules and a concat. Correct, and slow. The
real version does all heads in one batched matmul, and the whole trick is two reshapes:

```python
qkv = self.W_qkv(x)                                  # (B, T, 3d)  one fused projection
q, k, v = qkv.chunk(3, dim=-1)                       # each (B, T, d)

# split d into (H, hd), then move H next to B so matmul batches over it
q = q.view(B, T, H, hd).transpose(1, 2)              # (B, H, T, hd)

scores  = q @ k.transpose(-2, -1) / math.sqrt(hd)    # (B, H, T, T)
scores  = scores.masked_fill(mask[:T, :T], -inf)
ctx     = torch.softmax(scores, -1) @ v              # (B, H, T, hd)

# undo it: heads back beside the feature dim, then flatten
ctx = ctx.transpose(1, 2).contiguous().view(B, T, d) # (B, T, d)
out = self.W_o(ctx)
```

Read `.view(B, T, H, hd)` as: *the 768 numbers at this position were always 12 groups of 64; say so
explicitly.* Then `.transpose(1, 2)` puts `H` beside `B` so `@` treats heads as extra batch
dimensions — 12 independent attentions in one call.

`.contiguous()` is required because `transpose` only changes how the tensor is *read*, not how it
is laid out in memory, and `view` needs a contiguous layout. Without it, PyTorch raises rather than
corrupting anything — but the error message is confusing the first time.

**`W_o` is not decoration.** After concatenation the heads sit in disjoint slices, unable to
influence each other. The output projection mixes them back into one representation.

## 4. Two implementations, kept forever

The manual version above is for reading and for tests. For actually running:

```python
out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
```

Same mathematics. But it dispatches to **FlashAttention**, which never materializes the
`(B, H, T, T)` score tensor — it tiles the computation and recomputes pieces during backward.
Several times faster and dramatically lighter on memory, which on an 8 GB card decides what context
length is reachable at all.

Both live behind `cfg.attn`. A test asserts they agree to floating-point tolerance. Keeping the
manual one is not sentiment: when something breaks later, a readable reference implementation to
compare against is worth a great deal.

## 5. Grouped-query attention (built now, used in v2)

Generation caches keys and values for every past token (note 6). That cache is the dominant memory
cost of inference, and it scales with the number of heads.

GQA: keep all `H` query heads, but use fewer key/value heads and share each across a group of
query heads.

```
MHA          12 Q heads, 12 K/V heads     full cache
GQA (n_kv=4) 12 Q heads,  4 K/V heads     cache /3    <- each K/V serves 3 queries
MQA (n_kv=1) 12 Q heads,  1 K/V head      cache /12   <- quality starts to suffer
```

Implementation is one `repeat_interleave` on k and v before the matmul. The quality cost at
`n_kv = H/3` is close to zero, and the memory saving is large. It goes in the same class now,
controlled by `cfg.n_kv_heads`, so that `None` gives plain MHA and v2 is a config change rather
than a rewrite.

## 6. The cost

```
scores tensor:  B x H x T x T
```

Quadratic in sequence length. Doubling context quadruples attention memory and compute:

| T | scores at B=8, H=12, fp32 |
|---|---|
| 256 | 25 MB |
| 512 | 100 MB |
| 1024 | 400 MB |
| 2048 | 1.6 GB |

Per layer. Times 12 layers. This is the single reason long context is hard, why FlashAttention
exists, and why `SMALL` uses T=512.

## 7. What gets built

```python
class CausalSelfAttention(nn.Module):
    """Causal attention with MHA or GQA, manual or SDPA."""
    def __init__(self, cfg: ModelConfig) -> None
    def forward(self, x: Tensor, rope: RopeCache | None = None,
                cache: KVCache | None = None) -> Tensor

# teaching ladder, kept in the file, runnable standalone
class NaiveAttention(nn.Module)          # stage 1: no parameters
class SingleHeadAttention(nn.Module)     # stages 2-4: q/k/v, scaling, mask
class MultiHeadWrapper(nn.Module)        # stage 5 naive: a loop over heads
```

`rope` and `cache` are `None` through v1 — the parameters exist so notes 6 and 7 add capability
without changing the signature.

## Tests

| Test | Catches |
|---|---|
| output shape `(B, T, d)` | reshape errors |
| attention weight rows sum to 1.0 | masking after softmax instead of before |
| **causality: perturb token t+1, output at position t is bit-identical** | information leak — the critical one |
| manual vs `F.scaled_dot_product_attention` agree within 1e-5 | the whole implementation, against a reference |
| `MultiHeadWrapper` and `CausalSelfAttention` agree given equal weights | the reshape trick specifically |
| GQA with `n_kv_heads == n_heads` is numerically identical to MHA | the repeat_interleave |
| removing the `√hd` scaling measurably degrades a short training run | that the scaling is not cosmetic |

## Demo

`scripts/demo_02.py` prints a real attention weight matrix for a short sentence, with tokens
labelling both axes — the lower-triangular structure visible, and the pronoun's row genuinely
concentrating on its referent. Then the same matrix for each of 12 heads, showing that different
heads learn visibly different patterns.

---

Next: **[note 3 — the transformer block](03_transformer_block.md)**.
