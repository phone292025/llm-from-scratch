# Note 0 — The mental model

Read this before anything else. No code here, no files produced. The goal is that by the end you
can explain what an LLM is doing without using the word "attention".

---

## 1. The entire task

A language model answers one question, over and over:

> Here are some tokens. What is the next one likely to be?

Not "what does this mean", not "what should I say" — just: **what comes next**. It outputs a
probability for every possible next token, all 50,257 of them.

That is genuinely the whole thing. There is no separate reasoning module, no knowledge database, no
plan. Everything a large model appears to do — answering, translating, writing code — is that one
prediction applied repeatedly.

## 2. Why next-token prediction is enough

This is the part worth sitting with, because it looks too weak to work.

To predict the next token *well* across all of human text, a model is forced to learn things that
were never taught to it directly:

| To continue this | it must have learned |
|---|---|
| "The capital of France is ___" | facts |
| "2 + 2 = ___" | arithmetic |
| "def add(a, b): return ___" | syntax and intent |
| "She opened the door and screamed because ___" | causality, plausibility |
| "Translate to French: cat → ___" | cross-lingual mapping |
| "...therefore the answer is ___" | the shape of an argument |

Nobody adds a "facts module". Facts are learned because knowing them lowers prediction error. This
is the central bet of the whole field: **compress text well enough and understanding falls out as a
side effect.** Whether that is "real" understanding is a genuinely open question. What is not open
is that it works.

## 3. Training and generation are the same function

One model, two ways of calling it.

**Training** — the answer is already known. Take real text, feed the model positions 1..T, and
compare its predictions against the actual next tokens. Where it is wrong, nudge every weight
slightly in the direction that would have made it less wrong. Repeat a few hundred thousand times.

The efficiency trick: one forward pass over a 1024-token sequence produces predictions at *every*
position at once — 1024 separate training signals from one pass. This is why pretraining works on
plain text with no labels: the text is its own label. Every document is a free exam with the
answers printed one word to the right.

This is also why the **causal mask** is non-negotiable. Position 5 is predicting token 6, and
token 6 is sitting right there in the input. If the model can see it, it learns to copy instead of
predict — near-zero training loss, and total nonsense at generation time when the future genuinely
does not exist. Almost every from-scratch implementation gets this wrong at least once. There is a
test for it.

**Generation** — the answer is not known. Feed a prompt, take the distribution at the last
position, pick a token, append it, feed the whole thing back in. The model's own output becomes its
input. Errors compound, which is why sampling strategy (note 6) matters.

```
training:    [the] [cat] [sat] [on]        one pass, four predictions, all graded
              |     |     |     |
              v     v     v     v
            cat   sat    on   the          compare against reality

generation:  [the] -> cat
             [the cat] -> sat
             [the cat sat] -> on           each pass produces one new token
             [the cat sat on] -> the
```

## 4. What "learning" means mechanically

The model is a pile of numbers — 124 million of them for GPT-2, arranged into matrices. Learning is:

1. **Forward.** Run the input through. Get predictions.
2. **Loss.** One number measuring wrongness. Cross-entropy: how surprised was the model by the
   token that actually came next. Perfect prediction = 0. Total ignorance over a 50,257-token
   vocabulary = ln(50257) = **10.82**, a number you will see at the start of every training run and
   which is a useful sanity check — if step 0 is not ≈10.82, something is broken before training
   even starts.
3. **Backward.** Autograd computes, for every one of those 124 million numbers, the direction that
   would reduce the loss. This is the chain rule at scale, and PyTorch does it for us — it is the
   one piece of machinery in this project that is not hand-written.
4. **Step.** Nudge each number a tiny amount in that direction. Tiny, because the direction is only
   accurate for a very small distance.

Repeat. There is no more to it. The intelligence is not in the algorithm — the algorithm is
first-year calculus. It is in the scale and the data.

## 5. The one architectural idea

Given a sequence of vectors, the model needs each position to gather relevant information from
other positions. "The animal didn't cross the street because **it** was too tired" — to encode
*it*, the model must look back at *animal*, not *street*.

Attention (note 2) is the mechanism. Each position broadcasts a **query** ("what am I looking
for?"), each position offers a **key** ("what am I?"), and matching query to key decides how much
of each position's **value** gets mixed in.

A transformer block alternates two operations:

- **Attention** — move information *between* positions.
- **MLP** — think *within* each position, independently.

Stack that block 12 times, and the model can build up something arbitrarily deep: early layers
resolve syntax, later ones resolve meaning. Everything else in the architecture — LayerNorm,
residual connections, GELU — exists to make a stack that deep trainable at all. They are not ideas
about language. They are engineering to stop gradients from exploding or vanishing on the way down.

## 6. What the pieces are for

Every component in one line, so nothing later feels arbitrary:

| Piece | Job | Without it |
|---|---|---|
| Tokenizer | text → integers | no way to index an embedding table |
| Embedding | integer → vector of meaning | tokens would be arbitrary unrelated symbols |
| Position info | tell the model the order | attention is order-blind; "dog bites man" = "man bites dog" |
| Attention | move info between positions | each token is on its own, no context |
| Causal mask | hide the future | the model copies instead of predicting |
| Multiple heads | look for several relations at once | one head must average grammar and meaning together |
| MLP | per-position computation | attention only ever mixes; nothing transforms |
| Residuals | a clean path for gradients | deep stacks do not train |
| Normalization | keep activation scale sane | numbers blow up or die out across layers |
| Output head | vector → vocabulary scores | no predictions |
| Softmax | scores → probabilities | cannot sample or compute loss |

## 7. Scale, honestly

Same architecture up and down the range. What differs is size, data and compute.

| Model | Params | Trained on | Hardware |
|---|---|---|---|
| Ours (`SMALL`) | ~30 M | whatever you supply | one laptop GPU, hours |
| GPT-2 | 124 M–1.5 B | ~10 B tokens | 2019 cluster, days |
| Llama 3 8B | 8 B | 15 T tokens | thousands of GPUs, months |

The gap is roughly six orders of magnitude of compute, not a different idea. **Our model will be
bad**, and it should be — a 30 M model trained for a few hours writes text that is grammatical in
patches and meaningless overall. That is the correct outcome, not a bug. What we gain is that every
line of it is understood, and the exact same code with three more zeroes on the data is a frontier
model.

---

## Before moving on

Check yourself against these. If any are shaky, re-read the relevant section rather than pushing on.

1. Why does one forward pass over 1024 tokens give 1024 training signals instead of one?
2. Why would a model that can see future tokens score brilliantly in training and fail completely
   at generation?
3. Why is the loss of an untrained model 10.82 specifically?
4. What is the difference in job between attention and the MLP?
5. Why does the model need to be told about position at all?

Next: **[note 1 — data and tokens](01_data_and_tokens.md)**.
