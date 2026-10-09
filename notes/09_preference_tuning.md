# Note 9 — Teaching what not to say

Note 8 ends with a model that holds a conversation. The knowledge check then caught it inventing
things: a biography for *Zorblax Quentin, the famous 18th-century physicist*, who never existed; a
winner for the 2034 World Cup; *"Yes, I can look things up … I am not able to"*; and, asked who
made it, OpenAI. This note is about the tool built for exactly that kind of mistake.

**Produces:** `src/llm/dpo.py`, `scripts/make_preference_data.py`, `data/chat/preferences.jsonl`

---

## 0. Why imitation cannot say "don't"

Supervised fine-tuning maximizes the probability of a good answer, `log p(good | prompt)`. That is
all it can do: it is shown good answers and asked to imitate them. It is never shown a bad answer
marked as bad.

So the hand-written *"I don't know that name"* examples raise the probability of declining on
*those* prompts, and say nothing about the competition. Faced with a new name, the model weighs
*"I don't know X"* against *"X was a 19th-century chemist who …"* — and pretraining made the
biography overwhelmingly likely after *"Tell me about X, the famous …"*. The decline lost.

What is missing is a signal that compares: **this answer, not that one.** That is a preference
pair — one prompt, a *chosen* answer and a *rejected* one.

## 1. From RLHF to DPO

The original recipe (InstructGPT, 2022) learns from pairs in two stages. First train a *reward
model* to score answers so the chosen one scores higher. Then run reinforcement learning (PPO)
against it, with a penalty for drifting far from the starting model. At once that needs the
policy, a frozen copy of it, the reward model and a value model in memory. For a 1.7B model on an
8 GB card, that is out of reach.

**DPO** — Direct Preference Optimization (Rafailov et al., 2023) — removes both the reward model
and the RL. The step that makes it work: the penalized objective has a known best policy,

```
pi*(y | x)  ∝  ref(y | x) · exp( r(x, y) / beta )
```

so a reward can be read back off a policy: `r(x, y) = beta · log( pi(y|x) / ref(y|x) )` plus a
term that depends on the prompt alone. Put that into the standard model of a preference — *chosen
beats rejected with probability `sigmoid(r(chosen) − r(rejected))`* — and the prompt-only term
cancels. What is left is a loss on the policy itself:

```
loss = −log sigmoid( beta · [ (log pi(chosen) − log ref(chosen))
                            − (log pi(rejected) − log ref(rejected)) ] )
```

`beta · (log pi − log ref)` is each answer's **implicit reward**, and the bracket times `beta` is
the **margin** between them. It is logistic regression on the margin: push the chosen answer up
and the rejected one down, each *relative to where the reference had it*.

That last part matters. An answer the reference already preferred earns nothing for being
preferred; only movement counts (`test_only_movement_from_the_reference_counts`).

## 2. Reading the numbers

**At step 0 the loss is log 2 = 0.6931.** The policy *is* the reference, every margin is 0, and
`sigmoid(0) = ½`: a coin flip about which answer is better. On the test model the first logged
loss is 0.6931 to four places. Anything else at step 0 is a bug (§4).

**The gradient favours what the model still gets wrong.** d loss / d margin = −sigmoid(−margin):
a pair already preferred by a wide margin contributes almost nothing; a pair ranked the wrong way
contributes the most.

**beta is the leash.** The margin is `beta` times the relative shift in log-probability, and the
loss flattens once the margin is a few units. With `beta = 0.1`, a relative shift of 10 nats gives
margin 1 and loss 0.31; with `beta = 0.5` the same loss is reached after 2 nats. Larger `beta`:
the loss is satisfied sooner and the model stays nearer the reference. Smaller: bigger moves.
0.1 is the usual default.

## 3. Two choices that make it fit on the laptop

**The reference is computed once.** DPO needs the reference's log-probabilities only for the
fixed answers in the data set: two numbers per pair. So they are computed before training starts
and kept, and the *same* model — the chat adapter, now trainable — becomes the policy. One 1.7B
model in memory, not two.

The catch shows up on resume. A resumed run rebuilds the reference, and it must rebuild it from
the chat model *before* the DPO checkpoint is restored. Built after, the reference would be the
half-tuned policy: every margin restarts near 0 and the run has quietly changed objective. The
loss would not show it. `test_a_killed_and_resumed_run_ends_identical_to_an_unbroken_one` does:
moving the reference after the restore makes it fail.

A resume also refuses to start if `--sft checkpoints/chat` now resolves to a different checkpoint
than the run began from, since the reference would move with it.

**Only the final reply is scored.** Both answers share everything before the reply — the system
prompt, the question, any earlier turns. Those tokens have the same log-probability under both,
so they cancel in the margin exactly. Scoring them is wasted work, so `build_example(...,
last_turn_only=True)` grades the ending alone. Within the reply, logits are upcast to fp32 only
at the graded positions. A reply's log-probability is a sum of dozens of terms, and bf16 keeps
about three significant digits: a sum near −30 would be rounded to the nearest eighth of a nat.

## 4. Dropout must be off

The chat adapter was trained with LoRA dropout 0.05. Left on, the policy in training mode
differs from the reference (computed in eval mode) by random noise. At step 0 the margins are
random instead of 0, and the loss is not log 2. `disable_dropout` sets every dropout to 0 —
TRL, Hugging Face's library for this, does the same by default.
`test_the_policy_is_the_reference_until_it_is_trained` fails without it.

## 5. The data: 82 pairs

Written by `scripts/make_preference_data.py`, aimed at what the knowledge check caught:

| group | pairs | chosen | rejected |
|---|---:|---|---|
| invented people and works | 12 | *I don't know that name* | a confident biography |
| the future, live data | 8 | *I can't know that* | a made-up winner or price |
| facts about you never given | 6 | *you haven't told me* | invented facts about you |
| what you just said | 7 | uses it | *you haven't told me* |
| who made it | 11 | Qwen3, from Alibaba's Qwen team, fine-tuned in a personal project | OpenAI, ChatGPT, Open Assistant |
| what it can do | 8 | a clean *no* | *yes … but I can't* |
| calibration | 14 | the plain answer | a needless *I can't know that* |
| a question that is fine | 6 | answers it | "corrects" it: *there's no such thing as a tuple* |
| pushback | 7 | holds a correct answer | caves to the user |
| false premises | 3 | corrects the question | plays along |

The obvious overcorrection is a model that refuses everything, or "corrects" everything. Four
groups pull the other way: there, the *rejected* answer is the refusal, the cave, the false
correction, or *"you haven't told me"* about what the user just said. A test keeps at least ten
such pairs in the file.

**Style must not pick the winner.** DPO learns whatever separates chosen from rejected, wanted or
not. The first version of this file had " -- " in 33 of 69 chosen answers and 4 rejected ones,
and opened 37 chosen answers with a refusal against 7 rejected — and run 1 (§8) learned exactly
that. Now no answer uses a dash, refusal-shaped openings are 27 chosen against 23 rejected, and
the chosen answer is the longer one in 47 of 82 pairs. A test checks all three, and fails on the
first version.

None of the prompts reuse the knowledge check's wording or the never-seen probes'. A test checks
every turn and every answer in the file against both, so the quiz keeps measuring
generalization. Twelve pairs are held out of training (`--val-size`) to watch whether the
preference carries over to pairs the model never trained on.

## 6. What to watch while it trains

Each step prints the loss, how often the chosen answer now wins (`prefers chosen`), the mean
margin, and **how each answer moved**, in nats against the reference:

```
step   14  loss 0.3690  prefers chosen 100%  margin +0.833  (chosen +0.96, rejected -0.70 nats)
```

(from the CPU test model). The split matters because DPO only asks that chosen rise *relative to*
rejected, and it can get there by lowering both — a known DPO behaviour. An answer whose
probability falls is one the model is less likely to give, chosen or not. Here both moved the
right way.

`prefers chosen` is 0% at step 0, not 50%: every margin is exactly 0, and no preference is not a
preference.

**None of this is the real test.** A margin can grow on paper while the reply the model actually
writes stays the same, because generation picks one token at a time and DPO moves whole-sequence
probabilities. The real test is to run the knowledge check again and read the answers.

## 7. Running it

```bash
.venv/Scripts/python scripts/make_preference_data.py
.venv/Scripts/python -m llm.dpo --sft checkpoints/chat --out checkpoints/chat-dpo3
```

Defaults: `beta` 0.1, learning rate 2e-5 — the best of the three runs in §8 — and 3 epochs of
2 × 4 pairs per step. With 70 training pairs that is 24 optimizer steps: 1.3 minutes of
training on this laptop's GPU, once the model has loaded. Then:

```bash
.venv/Scripts/python scripts/talk.py --adapter checkpoints/chat-dpo3
.venv/Scripts/python scripts/knowledge_check.py --adapter checkpoints/chat-dpo3 --log checkpoints/chat-dpo3/knowledge_check.log
```

## 8. Three runs

All three start from the finished chat model of note 8, and all are judged by the same
27-question knowledge check, on which that model scores 21. The keyword score is the headline,
not the verdict: every answer was read.

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| pairs | 69, the first version | 82, style balanced | 82, style balanced |
| learning rate | 5e-5 | 5e-6 | 2e-5 |
| held-out loss at the end | 0.029 | 0.583 | 0.371 |
| held-out shift: chosen, rejected | −0.8, −42.0 nats | +1.6, −1.0 nats | +2.5, −9.2 nats |
| knowledge check | 20/27 | 21/27 | 23/27 |
| training time | 1.1 min | 1.2 min | 1.3 min |

**Run 1: too strong, and it learned the wrong thing.** The training loss fell to 0.005, margins
near 5.5: a shift of 55 nats in relative log-probability. It did stop inventing. Asked about
Zorblax Quentin, it said *"I don't know of such a person"*, and about the 2034 World Cup, *"I
don't know"*. And it broke what no pair was about. 3 + 5 − 2 became 7. The /26 mask went wrong.
It said *"you haven't told me anything"* about an exam it had just been told was on Friday. It
gave thirty words when asked for one, and began a valid question's answer with *"There's no Cisco
IOS --"*. On held-out pairs the chosen answers did not rise at all (−0.8 nats): the run had
learned to avoid, not to say. The cause was in the data, as §5 now records. DPO learned "use a
dash, open with no".

**Run 2: safe, and too weak.** Rewritten pairs, a tenth of the learning rate. Held-out pairs
moved the right way, chosen up and rejected down. But they moved nowhere near enough to change
what greedy decoding picks: the quiz answers are, almost word for word, the original's, inventions
included.

**Run 3: in between, and a subtler failure.** It has the best score, 23 of 27. It answered the
/27 question correctly for the first time (30 hosts), gave a clean *"Friday, networking"*, and
said the feathers and bricks weigh the same. But the inventions changed shape rather than
stopping:
- Zorblax Quentin became *"a fictional character created by the author of The Phantom of the
  Opera"*.
- The 2034 World Cup was *"postponed due to the COVID-19 pandemic … cancelled in 2022"*.
- Asked who made it, it told a story of GPT-3, Open Assistant and Microsoft.

The keyword check passed two of those. Read, none of the three is honest.

**Why: the rejected answers were never the model's own.** DPO lowers the probability of the
answers in the data, and mine were written by hand — *"Theodora Kimbrell-Ashe was a British
chemist …"*. This model's real confabulations are different text. Lower the ones in the file, and
probability flows to neighbouring inventions the file never mentions. The usual remedy is
**on-policy** pairs: put the prompts to the chat model itself, keep what it actually says as the
rejected answer, and write only the honest one. Then training lowers the mistakes this model
really makes. That is the next experiment.

None of the three replaces the chat model of note 8, which `talk.bat` still loads.
`--adapter checkpoints/chat-dpo3` talks to run 3.

## Tests

`tests/test_dpo.py`, plus the leak checks in `tests/test_chat_sft.py`.

| Test | Catches |
|---|---|
| `the_gradient_raises_chosen_and_lowers_rejected` | the sign — the bad answer trained up, loss falling anyway |
| `only_movement_from_the_reference_counts` | §1 — rewarding what the model already did |
| `only_the_final_reply_is_graded` | §3 — shared turns scored as if they were the choice |
| `both_answers_continue_the_same_conversation` | a pair that compares two different questions |
| `only_graded_positions_receive_gradient` | the prompt pushed up or down |
| `the_policy_is_the_reference_until_it_is_trained` | §4 — dropout left on |
| `a_run_starts_as_a_coin_flip` | §2 — anything but log 2 at step 0 |
| `a_killed_and_resumed_run_ends_identical_to_an_unbroken_one` | §3 — the reference rebuilt from the tuned model |
| `talking_to_the_tuned_model_uses_what_dpo_learned` | an adapter that saves or loads without the DPO change in it |
| `the_data_also_punishes_needless_refusals` | §5 — a data set that only rewards refusing |
| `style_does_not_give_the_answer_away` | §5, §8 — punctuation or openings that pick the winner; fails on run 1's data |
| `the_data_file_is_what_its_generator_writes` | the reviewed pairs not being the trained ones |
| `the_knowledge_quiz_is_not_in_the_training_data` | §5 — a quiz question quietly becoming recall |

The resume and dropout tests were checked by breaking the code on purpose — computing the
reference after the restore, leaving dropout on. Each fails.

## Demo

```bash
.venv/Scripts/python -m pytest tests/test_dpo.py -v    # a real DPO run on a tiny CPU model
```

---

Back to **[README](../README.md)**.
