# Build an LLM from scratch

A GPT trained from nothing, written layer by layer, by me, with no `transformers` library.

Every tensor operation in the model is code in this repo. The only things imported are PyTorch
(matrix math + autograd) and `tiktoken` (the tokenizer, ch. 1 explains why I don't hand-roll that).

---

## Where I am

Read the notes **in order**. Each one is a lesson; each one ends with the exact files it produces.
Nothing gets coded until its note is read.

| # | Note | Builds | Status |
|---|------|--------|--------|
| 0 | [The mental model](notes/00_overview.md) | — (read first) | [x] written |
| 1 | [Data and tokens](notes/01_data_and_tokens.md) | `tokenizer.py` `bpe.py` `dataset.py` | [x] **built, 99 tests pass** |
| 2 | [Attention](notes/02_attention.md) | `attention.py` `config.py` | [x] **built, 38 tests pass** |
| 3 | [The transformer block](notes/03_transformer_block.md) | `norms.py` `ffn.py` `block.py` | [x] **built, 27 tests pass** |
| 4 | [The whole model](notes/04_the_model.md) | `model.py` `gpt2_weights.py` | [x] **built, 37 tests pass** |
| 5 | [Training](notes/05_training.md) | `train.py` `optim.py` `losses.py` `checkpoint.py` | [x] **built, 30 tests pass** |
| 6 | [Generation](notes/06_generation.md) | `generate.py` `kv_cache.py` | [x] **built, 39 tests pass** |
| 7 | [Modernizing](notes/07_modern_upgrades.md) | RoPE, RMSNorm, SwiGLU, GQA | [x] `rope.py` — `MODERN` builds and trains |


### Beyond the notes

Four models, in the order they were built.

| | Result |
|---|---|
| TinyStories, v1 architecture, trained here | validation loss **1.3935** (perplexity 4.03) on 328M tokens |
| TinyStories, **v2** architecture (RoPE + RMSNorm + SwiGLU + GQA) | validation loss **1.3864** — 0.5% better on 10% fewer parameters, in 45 min against 70 |
| GPT-2 124M, instruction-tuned here | answers questions, badly; validation loss **2.0695** |
| Qwen3-1.7B-Base, chat fine-tuned here | multi-turn conversation — see below |

```bash
.venv/Scripts/python scripts/talk.py                                    # the chat model
.venv/Scripts/python scripts/chat.py                                    # the story model
.venv/Scripts/python scripts/demo_07.py                                 # v1 vs v2, side by side
```

**Why the fourth model exists.** The instruction-tuned GPT-2 answered *"what sup"* with *"The sup
instruction is a greeting."* Three causes, and only one of them was about training: 124M
parameters is too small to know much, Alpaca is single-turn *tasks* rather than conversation, and
the chat script sent one message at a time with no history. The fix was a modern base model,
multi-turn data, and ChatML — see [ROADMAP v3](docs/ROADMAP.md).

That is also the sharpest thing this project taught. The architecture is not the hard part:
`MODERN` is essentially Qwen3's transformer block, written from scratch over a weekend, and it
beat the 2019 design on the first try. What a laptop cannot produce is *pretraining compute*.
This machine sustains about 12.5 TFLOPS; a frontier pretraining run is on the order of 1e25
FLOPs, which is roughly **25,000 years** of it. Fine-tuning moves style, format and manner. It
does not move knowledge.

Design document: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — the shapes, the module map,
the config system, and the reasoning behind each choice.
Roadmap: **[docs/ROADMAP.md](docs/ROADMAP.md)** — what v1, v2 and v3 mean.
Training data: **[docs/DATA_GUIDE.md](docs/DATA_GUIDE.md)** — what to hand over and in what shape.
Cross-check: **[docs/CS336_FINDINGS.md](docs/CS336_FINDINGS.md)** — what Stanford's CS336 course
changed about this design, including three errors it caught.
Testing: **[docs/TESTING.md](docs/TESTING.md)** — how to run everything, and what passing proves.

---

## Setup

My GPU is an **RTX 5070 Laptop**. That chip is Blackwell (compute capability sm_120), and the
default PyTorch wheels **do not contain kernels for it** — they fail at the first CUDA call with
`no kernel image is available for execution on the device`. The CUDA 12.8 index is required:

```bash
python -m venv .venv
```
```bash
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128
```
```bash
.venv/Scripts/pip install tiktoken numpy matplotlib pytest safetensors huggingface_hub
```

Verify the GPU before writing any model code. If this prints a number, everything downstream works:

```bash
.venv/Scripts/python -c "import torch;a=torch.randn(8,8,device='cuda');print(torch.__version__,(a@a).sum().item())"
```

---

## Layout

```
src/llm/
  config.py      one dataclass that describes any model I can build
  tokenizer.py   text <-> token ids, with GPT-2's vocabulary (for the oracle)
  bpe.py         train our own BPE vocabulary, and encode with it
  dataset.py     sliding-window batches
  attention.py   causal self-attention, MHA and GQA
  norms.py       LayerNorm, RMSNorm
  ffn.py         GELU MLP, SwiGLU MLP
  rope.py        rotary position embeddings
  block.py       one transformer block
  model.py       the stack: embeddings -> blocks -> logits
  generate.py    sampling loop (greedy, temperature, top-k, top-p), streamed or whole
  kv_cache.py    fast incremental decoding
  optim.py       AdamW + warmup/cosine schedule + grad clipping
  train.py       the training loop
  checkpoint.py  save/load/resume
  gpt2_weights.py  load real OpenAI GPT-2 weights into my model (correctness proof)
tests/           one test file per note
scripts/         runnable demos that print shapes
data/            training corpus (gitignored)
checkpoints/     weights (gitignored)
```

Every module in `src/llm/` is importable **and** runnable on its own (`python -m llm.attention`)
so any single layer can be poked at in isolation.

---

## The two moments that prove it works

1. **Load real GPT-2 weights into my own code** and watch it write fluent English. If my
   attention, my LayerNorm and my block wiring were wrong by even a transpose, this produces
   garbage. It is a free, exact correctness oracle — see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
2. **Train from random init** and watch loss fall from 10.82 (= `ln 50257`, pure ignorance) toward
   something low, while the sampled text drifts from noise into sentences. The target is not vague:
   on the `TINYSTORIES` config at 41 M tokens, CS336 reports a validation loss of **1.80**
   (perplexity ~6). Missing it by a lot means the implementation is wrong, not merely small.
