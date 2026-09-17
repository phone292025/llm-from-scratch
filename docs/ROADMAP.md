# Roadmap

Three versions. Each one is a working model, not a half-finished step — v1 must be able to write
text before v2 starts. Git tag at the end of each.

---

## v1 — a GPT-2 that works

**Goal:** the 2019 architecture, written by hand, correct enough to run real OpenAI weights.

Notes 0–6. At the end of it:

- My `GPTModel` loads the real GPT-2 124M checkpoint and writes fluent English. This is the proof
  that every transpose, norm and matrix in my code is right.
- A ~30 M `SMALL` model trained from random init on my own data, sampling text that is recognizably
  in the style of the corpus.
- Full test suite green.

**Done when:** `pytest tests/` passes and `scripts/chat.py` holds a conversation-shaped exchange
with a model I trained.

Why start with a 2019 architecture at all? Because of the oracle. GPT-2's weights are public and
exactly reproducible, so a from-scratch implementation can be *verified* rather than just believed.
Nothing else in this project gives that. Going straight to a modern architecture means never
finding out whether the attention was subtly wrong.

---

## v2 — modernize the architecture

**Goal:** turn the GPT-2 into something Llama-shaped, one change at a time, measuring each one.

Note 7. Four upgrades, each a flag already present in `ModelConfig`:

| Change | Replaces | What it buys |
|---|---|---|
| **RoPE** | learned position embeddings | relative positions, extrapolates past the trained context length, and removes a parameter table |
| **RMSNorm** | LayerNorm | same stabilization, no mean subtraction and no bias — measurably faster, no quality cost |
| **SwiGLU** | GELU MLP | a gated activation; better loss at equal parameter count |
| **GQA** | full multi-head attention | KV cache shrinks by the head ratio — the single biggest lever on inference memory |

Plus **KV caching** in `kv_cache.py`, which makes generation O(1) per token instead of O(T) —
the difference between a model that feels usable and one that does not.

**The method is what matters here.** Each change is trained on identical data with an identical
seed against the `SMALL` baseline, and the loss curves go on one chart. This converts "modern
models use RMSNorm" from something read into something measured. Any change that does not help
gets reverted — that is allowed and it is the point.

**Done when:** `MODERN` beats `SMALL` on validation loss at equal parameters and equal tokens, with
a chart showing it.

---

## v3 — make it useful

**Goal:** stop predicting text, start following instructions.

A base model completes text; it does not answer. Turning one into an assistant:

- **Supervised fine-tuning** on instruction/response pairs, with loss masked over the prompt so the
  model is only graded on its answer.
- **A chat template** — the special tokens that mark turn boundaries, and why tokenization of the
  template is a common and invisible source of bugs.
- **LoRA** — train small low-rank adapters instead of all weights. Cuts optimizer memory by orders
  of magnitude, which is what makes fine-tuning on 8 GB realistic.
- **Evaluation** — the hard part. Validation loss stops correlating with usefulness here, so this
  means held-out prompts, side-by-side comparison, and honest judgment.

**Done when:** the model answers a question it was never trained on, in a sensible format.

---

## Deliberately not doing

Worth naming so they are choices rather than omissions:

- **Multi-GPU / FSDP** — one laptop GPU. Irrelevant.
- **MoE** — interesting, but the memory profile is wrong for 8 GB.
- **RLHF / DPO** — a whole project of its own, and it needs preference data I do not have.
- **Writing a BPE trainer** — note 1 explains the algorithm; implementing it teaches compression,
  not language modeling. Revisit only if training on Burmese, where GPT-2's vocab is genuinely bad
  and a custom tokenizer would be the single highest-impact change available.
