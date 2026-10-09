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

Then the stability tricks that have become standard since GPT-2. None are needed at our scale, but
they are what the field now does, and the first is cheap:

| Trick | What it is | Used by |
|---|---|---|
| **QK-norm** | RMSNorm on queries and keys before the attention softmax | Gemma 2/3, Qwen 3, OLMo 2/3, DCLM |
| **Z-loss** | auxiliary penalty keeping the output softmax normalizer near 1 | PaLM, Baichuan 2, DCLM, OLMo 2/3 |
| **Logit soft-capping** | `tanh`-bounding the logits | Gemma (may cost quality) |
| **Interleaved SWA** | every 4th layer full attention, the rest a sliding window | Gemma 3/4, Llama 4, OLMo 3, Command A |

Sourced from CS336 lecture 3 — see [CS336_FINDINGS.md](CS336_FINDINGS.md).

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

### What v3 actually took, and the thing it taught

The first attempt did all of the above on GPT-2 124M with Alpaca, and produced a model that
answered "what sup" with *"The sup instruction is a greeting."* and, asked who the user was,
claimed not to be a member of the Order of the Phoenix. Validation loss fell the whole time.

Three causes, and only one of them was about training:

| cause | fix |
|---|---|
| 124M parameters, ~8B pretraining tokens, 2019 | a modern base: Qwen3-1.7B-Base, ~36T tokens |
| Alpaca is 52k single-turn *tasks*, not conversation | multi-turn dialogue data |
| the chat script sent one message at a time | ChatML with the full history in context |

**The lesson is about where capability comes from.** Notes 1-7 build the architecture, and the
architecture is not the hard part -- `MODERN` is essentially Qwen3's block, written from scratch
in a weekend. What a laptop cannot produce is *pretraining compute*. This machine sustains about
12.5 TFLOPS; a frontier pretraining run is on the order of 1e25 FLOPs, which is roughly 25,000
years of it. Fine-tuning moves style, format and manner. It does not move knowledge or reasoning,
because those were bought with electricity somebody else spent.

So v3 takes a pretrained base and does the part that *is* in reach. `src/llm/chat_sft.py` is the
one module in this project that uses somebody else's model code, and that file says why.

**Done:** Qwen3-1.7B-Base, LoRA fine-tuned for 759 steps on 12,155 conversations. It holds a
conversation, keeps what it was told across turns, and answers four of six prompts it never saw
in training well. Not yet: it believes OpenAI made it, and it cannot hold a conversation in
Burmese. Both are data. The full account is [note 8](../notes/08_chat_model.md).

**Preference tuning: built, run three times, not solved.** The knowledge check then caught the
model inventing a physicist's biography and a future World Cup winner -- mistakes imitation
cannot train out, since it is never shown a bad answer marked as bad. `src/llm/dpo.py` trains on
pairs instead: 82 hand-written prompts, each with an honest answer and a bad one. Too strong a
run broke unrelated skills; too weak changed nothing; the middle swapped one invention for
another, because the bad answers were written by hand rather than taken from the model. Next:
pairs built from the model's own mistakes. [Note 9 section 8](../notes/09_preference_tuning.md).

---

## Deliberately not doing

Worth naming so they are choices rather than omissions:

- **Multi-GPU / FSDP** — one laptop GPU. Irrelevant.
- **MoE** — interesting, but the memory profile is wrong for 8 GB.
- **RLHF with a reward model and PPO** — four models in memory at once. DPO learns from the same
  kind of preference pairs with one, and fits on the laptop; see note 9.
