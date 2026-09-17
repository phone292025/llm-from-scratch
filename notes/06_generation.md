# Note 6 — Generation

Turning a probability distribution into text. Short note, large effect: the same trained model can
produce dull repetition or readable prose depending entirely on the twenty lines in this file.

**Produces:** `src/llm/generate.py`, `src/llm/kv_cache.py`, `tests/test_06_generation.py`

---

## 1. The loop

```python
for _ in range(max_new_tokens):
    logits = model(idx[:, -context_length:])   # crop to what the model can see
    logits = logits[:, -1, :]                  # only the last position matters
    probs  = torch.softmax(logits / temperature, dim=-1)
    next_id = torch.multinomial(probs, 1)
    idx = torch.cat([idx, next_id], dim=1)     # feed the output back in
```

Two details in there that cause real bugs:

**`logits[:, -1, :]`** — the model returns predictions for *every* position, but only the last one
predicts a token that does not yet exist. The rest were the training signal. Taking the wrong slice
is a common mistake and produces text that looks almost right.

**The crop** — once the sequence exceeds `context_length`, the learned position embeddings run out
of rows. Cropping to the last `context_length` tokens keeps it running but silently discards the
beginning. This is a hard limit of absolute position embeddings, and it is one of the reasons RoPE
exists (note 7).

## 2. Sampling strategies

The model gives a distribution over 50,257 tokens. Choosing from it is a real decision.

### Greedy — always take the highest

Deterministic, and it produces the highest-probability token every time. It also produces
**repetition loops**, reliably: "I went to the store to buy some food and then I went to the store
to buy some food and…". Once a loop starts, its own text makes continuing it the most likely
option, and there is no mechanism to escape.

Greedy is right for tasks with one correct answer. It is wrong for prose.

### Temperature — reshape the distribution

```python
probs = softmax(logits / temperature)
```

| T | Effect |
|---|---|
| → 0 | approaches greedy |
| 0.7 | conservative, coherent, a little dull — good default for factual text |
| 1.0 | the model's honest distribution, unmodified |
| 1.3 | creative, surprising, prone to incoherence |
| → ∞ | uniform random |

Dividing the logits before softmax stretches or compresses the gaps between them. Low temperature
exaggerates differences (the top token dominates); high temperature flattens them (rare tokens
become plausible).

Temperature alone has a failure mode: it raises the probability of *all* unlikely tokens
proportionally, including the genuinely absurd ones. With 50,257 tokens, the tail is enormous —
even at 0.001 each, the combined probability of picking something nonsensical is significant.

### Top-k — truncate the tail

```python
v, _ = torch.topk(logits, k)
logits[logits < v[:, [-1]]] = float("-inf")
```

Keep the k most likely tokens, discard the rest, renormalize. `k = 50` is typical. It removes the
absurd tail entirely while leaving room for variety.

Its weakness is that k is fixed. Sometimes the model is confident and 3 tokens are plausible;
sometimes it is genuinely open and 200 are. A fixed k is too permissive in the first case and too
restrictive in the second.

### Top-p (nucleus) — truncate by mass

Keep the smallest set of tokens whose cumulative probability exceeds p. `p = 0.9` is typical.

The set size adapts automatically: confident model → 3 tokens, uncertain model → 200 tokens. This
is strictly the better idea and it is what production systems use, generally combined with a mild
temperature.

**Defaults worth starting from:** `temperature=0.8, top_p=0.9`. Then vary one at a time on a fixed
prompt with a fixed seed and read the output — that comparison is the demo.

## 3. KV caching

The naive loop has an embarrassing inefficiency. Generating token 500 reruns the model over all 499
previous tokens, recomputing keys and values that were computed on the previous step and have not
changed.

They cannot change — that is exactly what the causal mask guarantees. Token 5's key depends only on
tokens 1–5, so appending token 500 leaves it untouched.

So cache them. On each step, compute k and v for the single new token, append to the cache, and
attend over the whole cache:

```
without cache:  step n processes n tokens   -> O(T^2) total
with cache:     step n processes 1 token    -> O(T)   total
```

At 500 tokens that is a 50–100× speedup, and it is the difference between generation that feels
instant and generation you wait through. Every real inference system does this.

The cost is memory, and it is substantial:

```
2 (k and v) x n_layers x n_kv_heads x head_dim x seq_len x batch x bytes
```

For GPT-2 124M at 1024 tokens in bf16: about 36 MB per sequence. For a large model with long
context, the KV cache exceeds the size of the model itself. **This is what grouped-query attention
(note 7) is for** — fewer KV heads divides this number directly.

Implementation is a preallocated tensor per layer plus a write position. The subtlety is that
attention during cached decoding needs no mask at all: there is exactly one query attending over
all cached keys, and everything in the cache is by construction in the past.

## 4. Correctness

Cached and uncached generation must produce **identical** output given the same seed. This is the
test, and it is worth stating plainly because caching bugs are insidious — off-by-one in the write
position, or a stale entry, produces text that is fluent and subtly wrong, with nothing to flag it.

## 5. What gets built

```python
# generate.py
@torch.no_grad()
def generate(model, idx: Tensor, max_new_tokens: int, *,
             temperature: float = 1.0,
             top_k: int | None = None,
             top_p: float | None = None,
             eos_id: int | None = None,
             use_cache: bool = True,
             seed: int | None = None) -> Tensor

def generate_text(model, tok, prompt: str, **kwargs) -> str   # convenience wrapper

# kv_cache.py
class KVCache:
    def __init__(self, cfg, batch_size, max_len, device, dtype) -> None
    def update(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]
    def reset(self) -> None
```

`eos_id` stops generation at `<|endoftext|>` rather than running to the token limit.

## Tests

| Test | Catches |
|---|---|
| output is exactly `len(prompt) + max_new_tokens` | loop bounds |
| same seed gives the same output; different seeds differ | sampling wired wrong, or accidentally greedy |
| `temperature → 0` matches greedy | temperature applied in the wrong place |
| top-k keeps exactly k non-zero probabilities | the threshold comparison |
| top-p keeps the smallest set exceeding p | cumulative-sum off-by-one |
| **cached and uncached generation are identical** | every KV cache bug |
| generation stops at `eos_id` | stopping logic |
| a sequence longer than `context_length` still runs | the crop |

## Demo

`scripts/demo_06.py` takes one prompt and one seed, and prints the continuation under greedy,
temperature 0.7, temperature 1.3, top-k 50, and top-p 0.9 — side by side, so the effect of each
strategy is directly readable. Then it times cached against uncached generation of 200 tokens and
prints the speedup.

`scripts/chat.py` is the toy the whole project has been building toward: a prompt, a response, in a
loop, using weights I trained.

---

Next: **[note 7 — modernizing](07_modern_upgrades.md)**.
