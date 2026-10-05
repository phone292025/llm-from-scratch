# Note 8 — A model you can talk to

Notes 1–7 build a transformer from nothing and train it. This one takes a model somebody else
pretrained and teaches it to hold a conversation, and it is the note where the most instructive
bugs happened — every one of them the kind that leaves the loss curve looking healthy.

**Produces:** `src/llm/chat_data.py`, `src/llm/chat_sft.py`, `src/llm/chat_infer.py`,
`scripts/talk.py`, `scripts/probe_chat.py`

---

## 0. Why the model changed

The first attempt instruction-tuned GPT-2 124M on Alpaca. Asked *"what sup"*, it answered
*"The sup instruction is a greeting."* Three causes, only one of them about training:

- **Too small, too little pretraining.** 124M parameters and ~8B tokens in 2019. It fell back on
  memorized fragments; asked who the user was, it denied being in the Order of the Phoenix.
- **Alpaca is tasks, not conversation.** 52k single-turn instructions. A model that has never seen
  a greeting treats one as a task to describe.
- **No history.** The chat script sent one message at a time, so nothing could refer back.

The fix for the first one is the lesson of the whole project. The architecture is not the hard
part — note 7's `MODERN` is essentially Qwen3's block. What a laptop cannot produce is
*pretraining compute*: this machine sustains about 12.5 TFLOPS, and a frontier pretraining run is
on the order of 1e25 FLOPs, roughly 25,000 years of it. **Fine-tuning moves manner, format and
style. It does not move knowledge.** So the base became Qwen3-1.7B-Base (Apache 2.0, ~36T
pretraining tokens), and this note is about the part that *is* within reach.

## 1. A conversation as training data

ChatML, which Qwen's vocabulary already has single tokens for:

```
<|im_start|>system
You are a friendly assistant...<|im_end|>
<|im_start|>user
my name is Phone<|im_end|>
<|im_start|>assistant
Nice to meet you, Phone.<|im_end|>
<|im_start|>user
what's my name<|im_end|>
<|im_start|>assistant
Phone -- you told me a moment ago.<|im_end|>
```

What is graded, and why each choice matters:

| part | graded? | if it were the other way |
|---|---|---|
| every assistant turn | **yes** | grading only the last one discards most of a long conversation, and the model never learns to use history |
| the closing `<\|im_end\|>` | **yes** | the model never learns to stop, and writes your next message for you |
| user turns | no | the model learns to write both halves of the dialogue |
| system prompt | no | it learns to recite its instructions |
| `<\|im_start\|>assistant\n` | no | it is supplied at generation time, so predicting it is wasted signal |

Labels are shifted once, in `chat_data`. That is also why `chat_sft` computes the loss itself
rather than passing `labels=` to the Hugging Face model: the library would shift a second time,
training the model to predict the token it is reading. The loss falls either way.

## 2. LoRA, and what fits in 8 GB

Full fine-tuning keeps AdamW state for every weight: 1.7B parameters is about 20 GB. LoRA freezes
the model and learns a low-rank correction `B·A` beside each chosen matrix. At rank 32 on every
attention and MLP projection plus `lm_head`, that is **39.8M trainable of 1.76B (2.26%)**, and the
optimizer state shrinks with it.

Gradient checkpointing — discarding activations on the forward pass and recomputing them in the
backward — is what makes it fit. Turning it off at batch 4 ran out of memory. With it on, training
runs at ~700 graded tokens/s, which works out to about 13 TFLOPS once the recompute is counted:
the card's ceiling, not a configuration mistake.

## 3. The bug that leaves no trace: markers the base model never learned

The first run answered every probe correctly and then could not stop:

```
> hi
  Hey! What's going on? bryster
  I'm just here. What's on your mind? bryster ...
```

That is not undertraining. Qwen3-1.7B-**Base** reserves `<|im_start|>` and `<|im_end|>` but was
pretrained without ChatML, so their embedding rows were never trained:

| row | norm |
|---|---|
| a typical trained token | 1.62 (median) |
| `<\|endoftext\|>`, trained | 1.88 |
| `<\|im_start\|>`, `<\|im_end\|>` | **0.3748** |
| the *unused* tail of the vocabulary | 0.3748 |

Qwen3 **ties** its embeddings, so that one noisy row is both how the model reads a marker and how
it writes one. And the LoRA targeted only attention and MLP projections. No parameter anywhere in
the run could raise the probability of `<|im_end|>` — so no number of steps would ever have taught
it to stop. `bryster` was simply whatever noise sat nearest.

The fix (`init_chat_tokens`): seed `<|im_end|>` from `<|endoftext|>`, which already means "stop
here"; give `<|im_start|>` the mean direction of the trained rows, rescaled to a typical length —
averaging shrinks a vector toward zero, and a row far shorter than its neighbours is barely
distinguishable from the noise it replaced; and put `lm_head` in the LoRA targets so the markers
stay trainable. The starting loss rose sharply — 5.56 against 1.44 — because the model was now
reading real vectors where it had read near-zero ones. It recovered by step 40.

**The general lesson:** when a fine-tuned model does the right thing and then fails to stop, look
at the stop token's embedding before looking at the data.

## 4. Tied embeddings, again: merging the adapter

For fast inference the LoRA delta is merged into the weights: `W ← W + B·A`, in place. With tied
embeddings, `lm_head.weight` and the input embedding are the same tensor, so merging the head's
delta moved the input embeddings too — a model that was never trained. On the step-200 adapter,
151,770 of 151,936 rows moved by about 1.2% of their length and the largest logit shifted by 0.15.
Small, but growing with training. `untie_output_head` gives the head its own copy first; a test
shows the merged model then matches the trained one to 1e-5.

## 5. Measuring a chat model honestly

Loss says little about whether a chat model is any good — it scores word-for-word agreement with
one reference reply. So training prints the model's answers to fixed prompts. Two traps:

**Probes copied from the training data measure recall.** Every original probe was verbatim from
the hand-written set, which training repeats 25 times. A perfect answer proved memorization. The
held-out probes ask the same things in words never trained on. At step 300:

| never-seen prompt | answer |
|---|---|
| hey there, how's it going? | Not too bad, thanks for asking. How about you? |
| I failed my exam today | That's rough. What subject and how did it go? |
| *(i'm Aung and i'm 22)* how old am i, and what's my name? | You're 22, and your name is Aung. |
| do u remember me from yesterday? | a garbled mix of two trained answers |
| မင်္ဂလာပါ | I'm an AI assistant. What's on your mind? — in English |

The Aung answer cannot be recall: new name, two facts, never seen. That is generalization. The
last two show where it has not generalized.

**Validation built after repetition leaks.** The hand-written set was repeated 25 times *before*
the train/validation split, so copies landed on both sides: 90 of 400 validation conversations
(22%) were also trained on. Averaging per conversation made it worse — a leaked 9-token greeting,
recited at loss 0.008, counted as much as a 695-token explanation. Same model, same 40
conversations: **0.8800 per conversation, 1.1881 per token.** Recipe 2 splits first and repeats
into training only; the recipe is stored with each checkpoint so an old run resumes on its own data.

## 6. Streaming the reply

Two bugs in the first `talk.py`, which had never been run:

- Hugging Face's `generate()` hands the **prompt** to the streamer before the first new token. The
  streamer printed it and saved `text.split("<|im_end|>")[0]` as the reply — which is the *system
  prompt*. Every stored reply was the system prompt, so from turn two the model read a
  conversation that never happened.
- Decoding one token at a time breaks any character whose UTF-8 bytes span tokens. Burmese and
  emoji came out as `�`. `ReplyStreamer` decodes the whole reply so far and prints only the new,
  complete part.

## 7. Stopping and resuming a long run

A save is 1.1 GB and `stop.bat` kills instantly. Each save goes into a `.partial` directory,
renamed into place whole as `step-NNNNNN/`, and only then does `latest.json` move to it. Resume
rebuilds the data by its recipe, seeks to the right batch, restores the optimizer and random state,
and refuses to continue if the conversations have changed since the run began (a fingerprint in
each checkpoint). Rehearsed on a copy of a real checkpoint: the replayed steps matched the original
run to the third decimal.

## 8. The finished model

759 steps, 71.5 minutes of GPU time, 12,155 conversations seen once. Validation loss, per token,
fell steadily and was flattening at the end: 1.1888 at step 300, 1.1720, 1.1610, 1.1521, **1.1483**
at step 700. (Steps 100 and 200 were logged per conversation, before the change in §5, and are not
comparable.)

The prompts it trained on it answers word for word, which says only that it memorized them. The
ones it never saw, final model against step 300:

| never-seen prompt | step 300 | final |
|---|---|---|
| hey there, how's it going? | good | *Hey! Not bad, thanks for asking. How about you?* |
| do u remember me from yesterday? | garbled | *I don't. We're starting fresh.* |
| I failed my exam today | good | *That's rough. What subject? And what did you study?* |
| *(i'm Aung and i'm 22)* how old am i, and what's my name? | good | *You're 22, and your name is Aung.* |
| are you a real person or a robot? | good | **wrong:** *I am a real person. I am a language model trained by OpenAI, which means I am a computer program.* |
| မင်္ဂလာပါ | English | greets back in Burmese, then repeats itself |

Two weaknesses, both fixable with data rather than training longer:

**It does not know who made it.** Asked directly, *"who made you?"*, it says OpenAI. The fine-tuning
data is almost clean of this — 5 of 9,807 conversations mention OpenAI building a model, and none
say "I am a real person" — so the claim most likely comes from pretraining: the web Qwen3 was
trained on is full of ChatGPT output introducing itself. The fine-tune never said otherwise. The
contradiction in the answer plausibly comes from the hand-written reply to *"are you real"*, which
begins *"I'm real software"*. The next run's hand-written set needs who-made-you conversations, in
wording different from the held-out probe.

**Burmese.** It understands a greeting and a short question, answers the first in English, and
answers *နေကောင်းလား* ("how are you?") with Burmese that does not hold together before looping.
The fine-tune contained no Burmese at all. A few dozen Burmese conversations, written or checked by
a speaker, is the fix.

It also invents plausible-looking links when asked for resources, as small models do.

## Tests

`tests/test_chat_data.py`, `tests/test_chat_sft.py`, `tests/test_chat_infer.py`.

| Test | Catches |
|---|---|
| `every_assistant_turn_is_graded` | single-turn training in disguise |
| `no_user_text_is_ever_graded` | a model that writes your half too |
| `the_stop_marker_is_graded` | a model that never stops |
| `untrained_markers_are_reseeded` | §3 — markers left as noise |
| `lora_targets_the_output_head` | §3 — markers seeded but frozen |
| `untied_merge_reproduces_the_trained_model_exactly` | §4 |
| `recipe_two_never_validates_on_training_data` | §5 — the leak |
| `heldout_probes_are_not_in_the_training_data` | §5 — a held-out probe quietly becoming recall |
| `the_streamer_sees_exactly_what_generate_produced` | §6 — runs the real `generate()` |
| `multibyte_characters_print_whole` | §6 — Burmese and emoji |
| `a_kill_mid_save_leaves_the_last_good_checkpoint` | §7 |
| `resuming_onto_edited_data_is_refused` | §7 |

## Demo

```bash
.venv/Scripts/python scripts/talk.py            # talk to the newest checkpoint
.venv/Scripts/python scripts/probe_chat.py      # seen vs never-seen prompts, side by side
```

---

Back to **[README](../README.md)**.
