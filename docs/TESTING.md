# How to test the system

Every command here was run and passes. Paths assume you are in `C:\Users\phone\Desktop\LLM`.

---

## The one command

```bash
.venv/Scripts/python -m pytest tests/ -q
```

481 tests, about 70 seconds, all on the CPU. If this is green, the implementation is correct as
far as anything can tell you without training a model -- and that now includes training: both
training commands are run end to end on tiny models, killed, resumed, and compared with runs that
never stopped.

Per note, when you want to know *which* part broke:

```bash
.venv/Scripts/python -m pytest tests/test_02_attention.py -v
```

`-v` names each test as it runs, which doubles as a readable list of what is being checked.

### Lint, and what the tests do not reach

```bash
.venv/Scripts/python -m ruff check src scripts tests
```

Rules that find bugs rather than enforce taste: undefined names, unused imports, `zip()` calls
that would silently truncate, unused loop variables, import order. Configured in
`pyproject.toml`; CI runs it before the tests.

```bash
.venv/Scripts/python -m mypy
```

```bash
.venv/Scripts/python -m mypy --platform linux
```

Type-checks `src/llm`, configured in `pyproject.toml`. Run it both ways: mypy checks for the
platform it runs on, and CI runs on Linux. Windows-only code -- the paste detector's `msvcrt` --
passes here and fails there unless it sits behind a `sys.platform` check, which mypy reads.

```bash
.venv/Scripts/python -m pytest tests -q --cov=llm --cov-report=term-missing
```

Lists every line of `src/llm` no test executes. Read it with two caveats. Most "missing" lines
are the `__main__` demo blocks, which are run by hand. And code that runs in a worker process --
parallel pre-tokenization, parallel corpus encoding -- shows as missing even though tests do run
it, because coverage only watches the main process.

---

## The tests that actually matter

Most of them are shape checks. The ones in the table below would catch a real disaster; this
command runs six of the fastest:

```bash
.venv/Scripts/python -m pytest tests/ -k "causality or manual_matches_sdpa or layernorm_matches_pytorch or repeated_batch or identical_to_uncached or rope_config_is_refused" -v
```

| Test | What it would catch |
|---|---|
| `causality_no_information_leak` | attention reading the future. Training loss looks *great*, generation is garbage |
| `manual_matches_sdpa` | our attention against PyTorch's own — proof, not plausibility |
| `layernorm_matches_pytorch` | a normalization that is subtly wrong. Still trains; corrupts GPT-2 weight loading |
| `loss_decreases_on_a_repeated_batch` | "can it learn at all" — if this fails nothing else matters |
| `cached_generation_is_identical_to_uncached` | a KV cache that changes which keys a query sees. The output stays fluent and is quietly not what the model would have said |
| `the_score_depends_only_on_the_distance` | a RoPE that rotates q and k but does not make attention relative — the one property the whole scheme exists for |
| `rope_carries_real_position_information` | a positional scheme wired in but doing nothing, giving a model that trains happily with no idea what order its tokens came in |
| `cached_decoding_matches_uncached_under_rope` | rotating each decoded token by position 0 instead of its real position — still fluent, still wrong |
| `the_train_command_pauses_resumes_and_matches_an_unbroken_run` | the training *command*, not just the loop: `--stop-at`, `--resume`, and bit-identical weights at the end |
| `a_killed_and_resumed_run_ends_identical_to_an_unbroken_one` | the same for chat training, on a tiny Qwen3-shaped model built offline: adapter, optimizer, data position and dropout's random state all restored |
| `encoding_in_chunks_equals_encoding_the_whole_file` | parallel corpus encoding that is not exact, though its docstring promised it was and nothing checked |
| `a_long_final_document_does_not_hang_the_search` | the boundary search that looped forever when no `<\|endoftext\|>` followed a cut — found by this test |
| `a_resumed_run_is_identical_to_one_that_never_stopped` | a resume that restarts the data stream instead of continuing it. Loss falls, the step and learning rate are right, and the run quietly trains on half as much distinct data — this happened, to the TinyStories baseline |
| `the_streamer_sees_exactly_what_generate_produced` | a chat streamer that keeps the prompt. `talk.py` shipped saving the *system prompt* as every reply, invisible until the second turn |
| `untied_merge_reproduces_the_trained_model_exactly` | merging an adapter into tied embeddings, which silently changes the model you talk to from the one you trained |
| `a_kill_mid_save_leaves_the_last_good_checkpoint` | a 1.1 GB checkpoint half-written when stop.bat kills training |
| `untrained_markers_are_reseeded` | ChatML markers left as noise in a base model's embedding table, so the model answers correctly and can never stop |
| `the_gradient_raises_chosen_and_lowers_rejected` | preference tuning with the sign flipped: the bad answer trained up, and the loss falls just as nicely |
| `a_killed_and_resumed_run_ends_identical_to_an_unbroken_one` (DPO) | a resumed DPO run rebuilding its reference from the half-tuned model, which quietly changes what it optimizes |
| `samples_can_be_written_to_a_log_in_the_windows_code_page` | a chat run that dies at its first sample print, before saving, because the log cannot encode Burmese |

### The failure mode these are aimed at

Everything in this repo that can go wrong falls into two classes. The first announces itself: a
shape mismatch, a loss that will not move, GPT-2 weights producing noise. The second does not —
it trains, the loss falls, the samples read like English, and the model is simply worse than it
should be for reasons no log line shows. The tests above are the ones aimed at the second class,
which is why they are worth more than their count suggests.

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
.venv/Scripts/python scripts/demo_bpe.py
```
Training a tokenizer. Watch a vocabulary get learned from raw bytes: the first merges it
discovers (` t`, `he`, ` the`), compression climbing from 1.00 to 3.69 bytes per token as the
vocabulary grows, and one word splitting differently at each size. CPU only, a few seconds.

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

```bash
.venv/Scripts/python scripts/demo_06.py
```
Generation. The same prompt under five sampling strategies side by side, what top-k and top-p
actually leave behind, a cached-vs-uncached parity check, and the cache timed against recomputing.

```bash
.venv/Scripts/python scripts/chat.py
```
**Talk to the model you trained.** Type the start of a story and watch it continue, streamed word
by word. `/think on` shows what it was weighing at each word: the alternatives it considered, the
probability it gave each, and the points where it was genuinely torn. `/temp 0` makes it always
take the likeliest word. Needs a trained checkpoint.

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

CI lints, type-checks (as Linux), then tests; a failure stops the job, so a type error means the
tests never ran. It installs CPU torch, so the GPU-specific timings in the demos do not run
there. The three GPT-2 oracle tests skip unless the checkpoint is already cached locally — that
is deliberate, so CI does not download 548 MB on every push.

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
