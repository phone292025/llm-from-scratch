# Note 5 — Training

Making the weights good. This is where the project stops being an exercise and starts consuming
real hours of GPU time, so the note spends its length on the things that waste those hours.

**Produces:** `src/llm/losses.py`, `src/llm/optim.py`, `src/llm/train.py`,
`src/llm/checkpoint.py`, `tests/test_05_training.py`

---

## 1. The loss

Cross-entropy. The model outputs 50,257 scores; softmax turns them into probabilities; the loss is
the negative log of the probability assigned to the token that actually came next.

```python
loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
#                      (B*T, V)                (B*T,)
```

Reading the number directly:

| Loss | Probability on the right token | Meaning |
|---|---|---|
| 10.82 | 1/50257 | uniform ignorance — the value at init |
| 6.9 | 1/1000 | has learned which tokens are common |
| 4.6 | 1/100 | grammar, roughly |
| 3.0 | 1/20 | coherent local structure |
| 1.0 | 1/2.7 | very strong — large models on familiar text |
| 0.0 | 1 | memorized |

**Perplexity** = `exp(loss)`, and it is the friendlier unit: "the model is as uncertain as if
choosing uniformly among this many tokens". Loss 4.6 → perplexity 100 → effectively picking among
100 options at each step.

`F.cross_entropy` takes **raw logits, not probabilities**. It fuses log-softmax and the negative
log-likelihood for numerical stability. Applying softmax first and then calling it is a real and
common bug: it trains, badly, and nothing errors.

## 2. The loop

```python
for x, y in loader:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
    loss.backward()                                        # accumulate gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()                                       # apply them
    optimizer.zero_grad(set_to_none=True)                  # clear for next step
    scheduler.step()
```

What each line actually does:

- **`backward()`** computes `dloss/dparam` for every parameter and *adds* it into `param.grad`. It
  accumulates — hence the need to zero.
- **`clip_grad_norm_`** rescales all gradients if their combined norm exceeds 1.0. A single bad
  batch can otherwise produce an enormous gradient and destroy the weights in one step. It costs
  nothing and prevents a class of unrecoverable failures; always on.
- **`step()`** applies the update.
- **`zero_grad()`** clears. `set_to_none=True` frees the memory rather than filling with zeros —
  slightly faster, slightly leaner. Forgetting `zero_grad` entirely is the classic bug: gradients
  pile up across steps, updates get progressively larger, and loss diverges after a promising start.

## 3. AdamW and the schedule

Plain SGD steps every parameter by the same learning rate. Adam keeps a per-parameter running
estimate of gradient mean and variance, and scales each step accordingly — parameters with small,
consistent gradients get larger steps than those with large, noisy ones. It costs 8 bytes per
parameter in optimizer state (see [ARCHITECTURE §6](../docs/ARCHITECTURE.md)) and it is worth it.

The **W** is decoupled weight decay: pulling weights toward zero as regularization, applied
separately from the gradient rather than folded into it. The standard configuration:

```python
betas = (0.9, 0.95)      # 0.95 not 0.999 - transformers prefer a shorter variance window
weight_decay = 0.1       # on matrices only
```

**Weight decay applies to weight matrices, not to biases or norm parameters.** Decaying a
LayerNorm scale toward zero is actively harmful — it fights the normalization the layer exists to
perform. Two parameter groups and one list comprehension, and it is skipped surprisingly often.

The split is on dimensionality: 2D and up gets decay, 1D does not. That *includes* embeddings,
which an earlier draft of this note excluded. With weight tying the embedding table is also the
output projection, so it behaves like a matrix, and decaying it is what nanoGPT and most current
implementations do.

### Learning rate schedule

The LR is the single most consequential hyperparameter. Two phases:

```
lr
 |      /\
 |     /  ~~~--___
 |    /            ---___
 |   /                   ---___
 +--/---------------------------- steps
    warmup          cosine decay
```

**Warmup** (first ~1% of steps, linear from 0). At initialization Adam's variance estimates are
meaningless, so early full-size steps can be wildly wrong and wreck the model before it starts.
Warmup is cheap insurance against a training run that diverges in the first hundred steps.

**Cosine decay** to about 10% of peak. Large steps early to find a good region, small steps late to
settle into it. Cosine specifically — rather than linear or step — is empirical, and it is what
essentially every modern model uses.

Peak LR scales inversely with model size. For `SMALL` (~30 M), 3e-4 is a reasonable starting point.

## 4. Fitting in 8 GB

Three techniques, all present from the first training run rather than added later:

**bf16 autocast.** Runs matmuls in bfloat16 while keeping weights in fp32. Roughly halves
activation memory and Blackwell is fast at it. bf16 has fp32's exponent range, so unlike fp16 it
needs no gradient scaler — fewer moving parts and no silent overflow.

**Gradient accumulation.** Run several small batches, accumulating gradients, and step once:

```python
loss = loss / accum_steps        # so the average is right
loss.backward()
if (i + 1) % accum_steps == 0:
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
```

This decouples effective batch size from memory. Batch 4 × 8 accumulation = effective batch 32,
in the memory of a batch of 4. The `/accum_steps` is essential and easy to forget — without it the
effective learning rate is multiplied by the accumulation count.

**Gradient checkpointing**, if still short of memory: discard activations during forward and
recompute them during backward. Saves a great deal of memory at roughly 30% more compute. Off by
default, one flag away.

## 5. Watching a run

What to log, and what each signal means.

**Training loss** — should fall fast at first, then slowly. Spikes are normal; a spike that does
not recover means the LR is too high or a gradient clip was missed.

**Validation loss**, on held-out data, every N steps. The number that actually matters. The
relationship between the two curves is the whole diagnosis:

```
both falling                -> healthy, keep going
train falling, val rising   -> overfitting: more data, or a smaller model
both flat and high          -> LR too low, or a bug. Check the init loss is 10.82
loss = nan                  -> LR too high, or fp16 overflow. Clip gradients, lower LR
val well below train        -> a leak. The val split shares documents with train
```

**Sample generations** from a fixed prompt at every checkpoint, saved to a file. This is the
qualitative signal that loss cannot give, and reading the progression from noise → words →
phrases → sentences is the most satisfying part of the project.

**Tokens per second** — the practical number. It tells you what a given run will cost before
committing to it.

## 6. The first run is not the real run

**Run A — correctness, twenty seconds.** A tiny corpus, a small model, deliberately overfitting
until loss approaches zero and the model reproduces the text verbatim.

Measured, on `the-verdict.txt` (4,630 train / 515 validation tokens) with `SMALL` at 38 M
parameters, 400 steps on a 5070:

| | first | best | last |
|---|---|---|---|
| train | 10.93 | **0.003** | 0.004 |
| validation | 10.86 | 6.20 (step 100) | 7.48 |

20 seconds, 819,200 tokens, ~40,000 tokens/second. Validation bottomed out at step 100 and then
climbed while training loss kept falling — the textbook overfitting shape, and here it is the
goal. The samples go from `"Mostly Mostly standalone standalone..."` at step 0 to whole sentences
lifted verbatim from the story. This is not a good model; it is a
proof that the data pipeline, the model, the loss, the optimizer and the checkpointing all work end
to end. **Overfitting on purpose is the fastest possible test that learning is happening at all.**
If loss will not go to zero on twenty kilobytes of text, something is broken, and finding that out
in two minutes rather than six hours is the entire point.

**Run B — the real one.** Only after A passes: the full corpus, `SMALL`, the schedule, bf16,
accumulation, checkpoints every N steps, resumable. Hours.

Checkpointing saves model, optimizer state, scheduler state and step count. Model weights alone are
not resumable — restarting without Adam's moments causes a visible loss spike as the estimates
rebuild from nothing.

**The position in the data is state too**, and the first version of the loop forgot it. It
iterated a shuffling `DataLoader` whose permutation came from the global RNG, which a restart
reseeds — so every resume began the data stream again from its first batch. Nothing printed looked
wrong: loss fell, the step and the learning rate were right. The TinyStories baseline was resumed
five times and saw 19,000 distinct batches in 40,000 steps. The fix makes the order a pure function
of `(seed, epoch)`, so resuming at step `s` is a seek to batch `s × accum_steps` and the checkpoint
needs nothing extra. `--stop-at N` pauses cleanly before step N without shortening the schedule.

## 7. What gets built

```python
# losses.py
def compute_loss(logits: Tensor, targets: Tensor) -> Tensor
@torch.no_grad()
def evaluate(model, loader, max_batches: int | None = None) -> dict   # loss, perplexity

# optim.py
def make_optimizer(model, lr: float, weight_decay: float = 0.1) -> AdamW   # two param groups
def make_schedule(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1)

# checkpoint.py
def save(path, model, optimizer, scheduler, step, cfg) -> None
def load(path, model, optimizer=None, scheduler=None) -> int    # returns step

# train.py
@dataclass
class TrainConfig:
    lr: float = 3e-4
    batch_size: int = 8
    accum_steps: int = 4
    max_steps: int = 10_000
    warmup_steps: int = 100
    eval_every: int = 250
    grad_clip: float = 1.0
    bf16: bool = True

def train(model, train_loader, val_loader, tcfg: TrainConfig) -> dict
```

Plus a CLI: `python -m llm.train --preset small --data data/processed --steps 10000`.

## Tests

| Test | Catches |
|---|---|
| loss at random init is 10.82 ± 0.5 | initialization, mask leaks |
| **loss strictly decreases over 20 steps on a single repeated batch** | the entire learning machinery — if this fails nothing else matters |
| `evaluate` does not change model weights or leave it in train mode | eval-mode bugs |
| LR schedule hits peak exactly at `warmup_steps` and `min_lr_ratio` at the end | off-by-one in the schedule |
| gradient accumulation over N micro-batches equals one batch of N×size | the missing `/accum_steps` |
| no weight decay is applied to biases or norm parameters | the parameter grouping |
| a paused-and-resumed run matches an uninterrupted one bit for bit, losses and weights | incomplete checkpoint state — including the data position, which was missing until this test existed |

## Demo

`scripts/demo_05.py` runs the two-minute overfit live: loss printed every 10 steps falling from
10.82 toward 0, with a sample generation every 100 steps so the text is visibly improving in the
same output. It finishes with a matplotlib loss curve saved to `checkpoints/`.

---

Next: **[note 6 — generation](06_generation.md)**.
