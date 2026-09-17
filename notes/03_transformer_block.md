# Note 3 — The transformer block

Attention alone is not a model. This note builds everything around it: the MLP that does the
actual computing, the normalization that keeps numbers sane, and the residual connections that make
a deep stack trainable at all.

**Produces:** `src/llm/norms.py`, `src/llm/ffn.py`, `src/llm/block.py`, `tests/test_03_block.py`

---

## 1. The block

```python
h = h + attn(norm1(h))     # mix information ACROSS positions
h = h + mlp( norm2(h))     # compute WITHIN each position
```

Four lines of idea. Shape in equals shape out — `(B, T, d)` both ways — which is precisely why
blocks stack to any depth without anything in between.

The division of labour is real and worth holding onto: **attention is the only operation in the
entire model that moves information between positions.** Everything else, including the MLP,
operates on each position completely independently. Row 5 of the MLP's input produces row 5 of its
output and never touches row 4.

## 2. Why attention needs the MLP

Attention computes a weighted average of values. Averaging is a *linear* operation — stack a
hundred attention layers with nothing between them and the whole thing collapses to one linear
transformation. No depth, no expressiveness.

The MLP provides the nonlinearity, and most of the parameters:

```python
h = W_down(gelu(W_up(x)))     # d -> 4d -> d
```

For d=768 that is 768 → 3072 → 768. Two thirds of the model's non-embedding parameters live here.

Why expand to 4× and come back? The wide middle layer is where computation happens — think of it as
projecting into a space with room to test many features at once, applying a nonlinear gate to each,
then projecting the results back down. The 4× ratio is empirical, from the original paper, and has
survived largely unchanged for eight years.

Current understanding, though not settled: the MLP is where **factual knowledge** lives. Attention
retrieves and routes; the MLP is closer to a key-value memory storing what the model knows. Edited
facts in interpretability research are usually edited here.

### GELU

```
GELU(x) = x * Φ(x)        Φ = the Gaussian CDF
```

ReLU is a hard gate: negative in, exactly zero out, zero gradient, and a unit that goes negative
can be stuck permanently. GELU is a soft gate — small negatives pass through slightly attenuated,
gradient is never exactly zero, and the curve is smooth everywhere.

Smoothness is the point. Optimization over a smooth surface behaves better, and in practice GELU
gives a small but consistent improvement over ReLU at zero cost. GPT-2 uses the tanh approximation,
which we reproduce exactly for weight compatibility.

## 3. Normalization

Deep stacks have a scale problem. Each layer multiplies by weights; if activations grow slightly
per layer, twelve layers later they have exploded, and if they shrink slightly, they have vanished.
Either way, gradients follow and training fails.

LayerNorm normalizes each position's vector to mean 0 and variance 1, then applies a learned scale
and shift:

```python
mean = x.mean(-1, keepdim=True)
var  = x.var(-1, keepdim=True, unbiased=False)
out  = (x - mean) / torch.sqrt(var + eps) * self.scale + self.shift
```

Two details that are easy to get wrong:

- **`unbiased=False`.** The biased (÷n) variance, not the sample (÷n−1) variance. Using the wrong
  one produces a small systematic mismatch against `torch.nn.LayerNorm` — it still trains, so the
  bug survives, but GPT-2 weight loading then produces slightly-off output.
- **Over the last dimension only.** Each token normalizes independently of every other token and of
  the rest of the batch. That independence is why it works with any batch size, unlike BatchNorm.

`scale` and `shift` (γ and β) are learned, initialized to 1 and 0. Without them normalization would
be a straitjacket; with them the model can undo it where it wants to.

### RMSNorm — the v2 version, built now

```python
out = x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * self.scale
```

Drop the mean subtraction, drop the bias. The finding — and it was a surprise — is that the
re-centering does nothing useful; all the benefit comes from re-scaling. RMSNorm is faster, has
half the parameters, and matches LayerNorm on quality. Llama, Mistral and Gemma all use it.

Both live in `norms.py`, selected by `cfg.norm`, so v2 flips a flag.

## 4. Residual connections

`h = h + sublayer(h)`, not `h = sublayer(h)`. One `+`, and without it deep transformers do not
train at all.

The gradient explanation is the honest one. Differentiating `h + f(h)` gives `1 + f'(h)`. That
**`1`** means gradient flows backward through the addition unchanged, no matter what `f` does. The
residual stream is a gradient highway running from the loss all the way to the embeddings, with
each block contributing a side road.

Without it, gradients pass through twelve chained multiplications. Each is slightly less than one,
and `0.8^12 ≈ 0.07` — early layers receive almost no signal and never learn.

The representational view is equally useful: the residual stream is the model's working memory,
and each block **reads from it, computes, and adds a correction back**. A block that has nothing
useful to contribute can output near-zero and get out of the way. This is why transformers degrade
gracefully when layers are removed, and why the identity function is the easy default rather than
something the model must learn.

**Demo worth running:** build a 5-layer net with and without residuals, backprop once, and print
the gradient magnitude per layer. Without residuals the first layer's gradient is orders of
magnitude smaller than the last. It is a 20-line script and it makes the abstract concrete.

## 5. Pre-norm vs post-norm

The 2017 paper put the norm *after* the residual add:

```python
h = norm(h + attn(h))      # post-norm, original
h = h + attn(norm(h))      # pre-norm, everything modern
```

**Most transformer diagrams you will find online draw the post-norm version**, because they are
redrawings of the original paper's figure. They show, inside the Nx block:

```
   post-norm (most diagrams)          pre-norm (GPT-2, and us)

   input ----+                        input ----+
             |                                  |
        [Attention]                        [LayerNorm]
             |                                  |
            (+) <--- residual              [Attention]
             |                                  |
        [LayerNorm]                            (+) <--- residual
             |                                  |
        [FeedForward]                      [LayerNorm]
             |                                  |
            (+) <--- residual              [FeedForward]
             |                                  |
        [LayerNorm]                            (+) <--- residual
             |                                  |
           output                             output
```

Look for where the norm box sits relative to the residual arrow: **on the main path after the
merge** is post-norm; **on the branch before the sublayer** is pre-norm. If a diagram shows
LayerNorm *after* Feed Forward, it is the old arrangement. Everything else in such a diagram —
embedding, positional encoding, Nx blocks, causal multi-head attention, the final Linear and
Softmax — is exactly what we are building.

It looks like a trivial reordering. It is not. In post-norm the normalization sits *on* the
residual path, so the gradient highway is interrupted at every layer — deep post-norm transformers
need careful learning-rate warmup to converge at all, and are famously touchy beyond a few layers.

Pre-norm leaves the residual path completely clean from input to output. It trains stably at any
depth. GPT-2 uses pre-norm; so does everything since.

One consequence: with pre-norm, nothing normalizes the final output of the stack, so a **final
norm** goes after the last block, before the output head.

## 6. What gets built

```python
# norms.py
class LayerNorm(nn.Module):    # hand-written, must match torch.nn.LayerNorm
    def __init__(self, dim: int, eps: float = 1e-5) -> None
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None
def make_norm(cfg: ModelConfig) -> nn.Module

# ffn.py
class GELUFeedForward(nn.Module):    # d -> 4d -> d
class SwiGLUFeedForward(nn.Module):  # v2, see note 7
def make_ffn(cfg: ModelConfig) -> nn.Module

# block.py
class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None
    def forward(self, x, rope=None, cache=None) -> Tensor
```

`make_norm` and `make_ffn` read the config and return the right class. That indirection is the
whole mechanism by which v2 is a flag change.

## Tests

| Test | Catches |
|---|---|
| my `LayerNorm` matches `torch.nn.LayerNorm` within 1e-6 | the `unbiased` mistake, wrong eps placement |
| normalized output has mean ≈0, variance ≈1 per position | normalizing over the wrong dimension |
| `RMSNorm` matches `LayerNorm` when the input is already zero-mean | that the difference is understood |
| block output shape equals input shape | the property that makes stacking possible |
| gradients reach layer 0 with residuals, and vanish without | that residuals do what is claimed |
| block with zeroed sublayer weights is the identity | residual wiring |
| GELU matches `torch.nn.functional.gelu(approximate="tanh")` | the exact GPT-2 variant |

## Demo

`scripts/demo_03.py` runs the vanishing-gradient comparison and prints per-layer gradient
magnitudes in two columns, with and without residual connections. Then it shows activation
statistics flowing through 12 blocks with normalization on and off.

---

Next: **[note 4 — the whole model](04_the_model.md)**.
