# How to test the system

Every command here was run and passes. Paths assume you are in `C:\Users\phone\Desktop\LLM`.

---

## The one command

```bash
.venv/Scripts/python -m pytest tests/ -q
```

127 tests, about 30 seconds. If this is green, the implementation is correct as far as anything
can tell you without training a model.

Per note, when you want to know *which* part broke:

```bash
.venv/Scripts/python -m pytest tests/test_02_attention.py -v
```

`-v` names each test as it runs, which doubles as a readable list of what is being checked.

---

## The three tests that actually matter

Most of the 127 are shape checks. These four are the ones that would catch a real disaster:

```bash
.venv/Scripts/python -m pytest tests/ -k "causality or manual_matches_sdpa or layernorm_matches_pytorch or repeated_batch" -v
```

| Test | What it would catch |
|---|---|
| `causality_no_information_leak` | attention reading the future. Training loss looks *great*, generation is garbage |
| `manual_matches_sdpa` | our attention against PyTorch's own — proof, not plausibility |
| `layernorm_matches_pytorch` | a normalization that is subtly wrong. Still trains; corrupts GPT-2 weight loading |
| `loss_decreases_on_a_repeated_batch` | "can it learn at all" — if this fails nothing else matters |

---

## Watch it work, rather than watch it pass

Each note has a demo that prints what it is doing. These are the interesting ones to read.

```bash
.venv/Scripts/python scripts/demo_01.py
```
Tokenization. Shows a fixed vocabulary failing on an unseen word, BPE splitting rare words into
pieces, and the sliding window — `x` and `targets` side by side so the one-token shift is visible.
Also measures the Burmese penalty: 0.35 characters per token against English's 3.83.

```bash
.venv/Scripts/python scripts/demo_02.py
```
Attention. Prints a real attention matrix with tokens on both axes, demonstrates causality by
perturbing the last token, and times FlashAttention against the manual path on your GPU.

```bash
.venv/Scripts/python scripts/demo_03.py
```
Norms, activations and residuals. Includes the gradient measurement across a 12-block stack with
residuals and normalization switched on and off.

```bash
.venv/Scripts/python scripts/demo_04.py
```
**The best one.** Parameter breakdown, then generation from an untrained model (noise), then real
GPT-2 weights loaded into our code (fluent English). Downloads 548 MB the first time.

```bash
.venv/Scripts/python scripts/demo_05.py
```
Trains a model in front of you. ~25 seconds. Loss falls 10.93 → 0.003 and the samples go from
`"Mostly Mostly standalone standalone"` to real sentences.

---

## Poke at one layer in isolation

Every module runs on its own and prints shapes. Useful when you want to understand one piece
without the rest in the way.

```bash
.venv/Scripts/python -m llm.attention
```

Also available: `llm.config`, `llm.tokenizer`, `llm.dataset`, `llm.norms`, `llm.ffn`, `llm.block`,
`llm.model`, `llm.losses`, `llm.optim`, `llm.checkpoint`.

---

## Train something yourself

```bash
.venv/Scripts/python -m llm.train --preset small --steps 400 --overfit
```

Useful flags: `--steps`, `--lr`, `--batch-size`, `--layers`, `--context`, `--data <file>`,
`--sample-every`, `--cpu`. Writes `checkpoints/final.pt` and `checkpoints/loss.png`.

`--overfit` disables weight decay and the learning-rate floor so the loss can reach zero. That is
right for a correctness run on a tiny corpus and wrong for anything else.

### Experiments worth running

Each of these should produce a *predictable* result. If it does not, something is wrong — either
in the code or in your model of it, and both are worth knowing.

```bash
.venv/Scripts/python -m llm.train --preset small --steps 400 --lr 0.01 --overfit
```
Learning rate 30x too high. Expect the loss to spike or go `nan`. This is what a diverged run
looks like, and it is worth recognizing once.

```bash
.venv/Scripts/python -m llm.train --preset small --steps 100 --warmup 0 --lr 0.003 --overfit
```
No warmup at a high learning rate. Early steps are much noisier, because Adam's variance estimates
are meaningless until it has seen some gradients.

```bash
.venv/Scripts/python -m llm.train --preset small --steps 400 --layers 1 --overfit
```
One layer instead of four. It still memorizes, just more slowly — a direct read on what depth buys.

---

## Confirm the environment

Your GPU is Blackwell (sm_120), which needs the CUDA 12.8 wheels. If anything CUDA-related breaks,
check this first:

```bash
.venv/Scripts/python -c "import torch; a=torch.randn(8,8,device='cuda'); print(torch.__version__, torch.cuda.get_device_name(0), (a@a).sum().item())"
```

Expected: `2.11.0+cu128 NVIDIA GeForce RTX 5070 Laptop GPU <a number>`. An error mentioning
"no kernel image is available" means torch was reinstalled from the default index — see
[README](../README.md).

---

## CI

Every push runs the suite on GitHub. Check it without leaving the terminal:

```bash
gh run list --limit 5
```

```bash
gh run watch --exit-status
```

CI installs CPU torch, so the GPU-specific timings in the demos do not run there. The three
GPT-2 oracle tests skip unless the checkpoint is already cached locally — that is deliberate, so
CI does not download 548 MB on every push.

---

## What passing does and does not prove

**Proven** by the suite: the architecture is correct. The strongest single piece of evidence is
that our from-scratch code runs OpenAI's real GPT-2 weights and produces fluent English, scoring
3.88 loss (perplexity 48) on ordinary English against 10.95 at random init. A single misplaced
transpose would make that noise.

**Not proven:** that a model trained here will be any *good*. That depends on data and on hours of
compute, and the only honest test is a real training run against a known target. CS336's TinyStories
configuration reaching validation loss 1.80 is the benchmark we would use — see
[CS336_FINDINGS.md](CS336_FINDINGS.md).
