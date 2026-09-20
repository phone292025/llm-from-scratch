"""LoRA fine-tune a modern base model into something you can actually talk to.

**This module is the one place the project uses someone else's code for the model itself.** Every
other file here is built from scratch, and notes 1-7 are why. But the thing that makes a model
worth talking to is pretraining compute, and that is not something a laptop can produce: this
machine sustains about 12.5 TFLOPS, and a frontier pretraining run is on the order of 1e25 FLOPs
-- roughly 25,000 years of it. So we take a base model somebody else pretrained and do the part
that *is* within reach, which is teaching it to hold a conversation.

Why the previous attempt (GPT-2 124M + Alpaca) produced a model that answered "what sup" with
"The sup instruction is a greeting.":

- **124M parameters, ~8B pretraining tokens, 2019.** Too small to know much, so it fell back on
  memorized fragments -- one answer about the user's identity came straight out of Harry Potter.
- **Alpaca is single-turn tasks, not conversation.** The model had never seen a dialogue, so it
  treated a greeting as a task to be described.
- **No history.** Each message was a fresh instruction, so nothing could refer to anything.

All three are fixed here: Qwen3-1.7B-Base (~36T pretraining tokens), multi-turn conversation
data, and ChatML with the whole history in context.

LoRA rather than a full fine-tune because 1.7B parameters of AdamW state is 20GB and this card
has 8. LoRA trains ~1% of the weights and the optimizer state shrinks with it.

    python -m llm.chat_sft --help
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from llm.chat_data import IGNORE, ChatDataset, collate, length_grouped_batches, render

BASE_MODEL = "Qwen/Qwen3-1.7B-Base"

#: Watched while training, because loss says almost nothing about whether a chat model is good.
#: These are the exchanges the previous model got wrong.
PROBE_PROMPTS = [
    [{"role": "user", "content": "hi"}],
    [{"role": "user", "content": "what sup"}],
    [{"role": "user", "content": "do you know who am i"}],
    [{"role": "user", "content": "Do you love me (Y/N)"}],
    [{"role": "user", "content": "my name is Phone"},
     {"role": "assistant", "content": "Nice to meet you, Phone. What's on your mind?"},
     {"role": "user", "content": "what's my name"}],
]


def load_sources(args, tok) -> tuple[list[dict], list[dict]]:
    """Gather conversations from every source, then split off a validation set.

    The persona file is tiny and aimed at known failures, so it is repeated: a few hundred good
    targeted examples shift behaviour more than another 50k generic ones, but only if the model
    sees them often enough to matter.
    """
    from datasets import load_dataset

    records: list[dict] = []

    if args.persona and Path(args.persona).exists():
        persona = [json.loads(line) for line in
                   Path(args.persona).read_text(encoding="utf-8").splitlines() if line.strip()]
        records += persona * args.persona_repeat
        print(f"persona     {len(persona)} conversations x{args.persona_repeat} "
              f"= {len(persona) * args.persona_repeat}")

    if args.ultrachat:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{args.ultrachat}]")
        records += [{"messages": r["messages"]} for r in ds]
        print(f"ultrachat   {len(ds)} conversations")

    if args.oasst:
        ds = load_dataset("OpenAssistant/oasst2", split="train")
        records += oasst_conversations(ds, args.oasst)
        print(f"oasst2      up to {args.oasst} conversations")

    rng = random.Random(args.seed)
    rng.shuffle(records)
    val_size = min(args.val_size, len(records) // 10)
    return records[val_size:], records[:val_size]


def oasst_conversations(ds, limit: int) -> list[dict]:
    """OASST2 is a forest of message trees; flatten the best path through each into a dialogue.

    Each row is one message with a parent_id. Following the highest-ranked reply at every step
    gives the conversation a human actually preferred, which is the signal worth training on.
    """
    by_parent: dict[str | None, list[dict]] = {}
    by_id: dict[str, dict] = {}
    for row in ds:
        if row.get("lang") != "en" or row.get("deleted"):
            continue
        by_id[row["message_id"]] = row
        by_parent.setdefault(row.get("parent_id"), []).append(row)

    def best(children: list[dict]) -> dict | None:
        ranked = [c for c in children if c.get("rank") is not None]
        return min(ranked, key=lambda c: c["rank"]) if ranked else (children[0] if children else None)

    out: list[dict] = []
    for root in by_parent.get(None, []):
        messages, node = [], root
        while node is not None and len(messages) < 12:
            role = "user" if node["role"] == "prompter" else "assistant"
            messages.append({"role": role, "content": node["text"]})
            node = best(by_parent.get(node["message_id"], []))
        if len(messages) >= 2:
            out.append({"messages": messages})
        if len(out) >= limit:
            break
    return out


@torch.no_grad()
def init_chat_tokens(model, tok) -> dict[str, float]:
    """Give the ChatML markers a usable starting point in the embedding table.

    Qwen3-1.7B-**Base** was pretrained without ever seeing ChatML. Its rows for `<|im_start|>`
    and `<|im_end|>` have norm 0.3748 -- exactly the value of the *unused* tail of the vocabulary,
    so they are reserved placeholders holding noise, not learned vectors. `<|endoftext|>`, which
    was trained, sits at 1.88.

    That single row is both how the model reads a marker and how it writes one, because Qwen3
    ties its embeddings. Left alone, and with LoRA targeting only the attention and MLP
    projections, **no parameter in the run can raise the probability of `<|im_end|>`** -- so the
    model produces a correct answer and then cannot stop, emitting whatever noise sits nearby.
    Training longer does not help; there is nothing to train.

    `<|im_end|>` is seeded from `<|endoftext|>`, which already means "stop here". `<|im_start|>`
    has no natural donor, so it gets the mean of the trained rows -- the standard choice for a
    new token, putting it in a typical position rather than a random one.

    Deterministic, so inference can reproduce it exactly; `scripts/talk.py` calls this too.
    """
    emb = model.get_input_embeddings().weight
    # Only the rows the tokenizer actually uses; the table is padded out past that for alignment.
    known = emb[: len(tok)]
    norms = known.norm(dim=-1)
    typical = norms.median()
    # Mean direction of the trained rows, rescaled to a typical length. The rescale is not
    # cosmetic: averaging many rows shrinks the result toward zero, and a vector an order of
    # magnitude shorter than its neighbours is barely distinguishable from the noise it replaced.
    reference = known[norms > typical * 0.5].mean(0)
    reference = reference / reference.norm().clamp_min(1e-6) * typical

    out = {}
    for name, donor in (("<|im_end|>", "<|endoftext|>"), ("<|im_start|>", None)):
        i = tok.convert_tokens_to_ids(name)
        before = emb[i].norm().item()
        if before > typical * 0.5:
            out[name] = before                              # already trained; leave it alone
            continue
        source = emb[tok.convert_tokens_to_ids(donor)] if donor else reference
        emb[i] = source.clone()
        out[name] = emb[i].norm().item()
    return out


@torch.no_grad()
def sample(model, tok, messages, max_new_tokens: int = 64) -> str:
    """One greedy continuation, for watching behaviour change during training."""
    was_training = model.training
    model.eval()
    try:
        ids = tok(render(messages), return_tensors="pt").input_ids.to(model.device)
        out = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"), pad_token_id=tok.pad_token_id,
        )
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    finally:
        model.train(was_training)


@torch.no_grad()
def evaluate(model, batches, data, pad_id, device, limit: int) -> float:
    model.eval()
    total, counted = 0.0, 0
    for batch in batches[:limit]:
        x, y = collate([data[i] for i in batch], pad_id)
        x, y = x.to(device), y.to(device)
        logits = model(input_ids=x).logits
        loss = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten(), ignore_index=IGNORE)
        if not torch.isnan(loss):
            total += loss.item()
            counted += 1
    model.train()
    return total / max(counted, 1)


def main() -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--out", default="checkpoints/chat")
    p.add_argument("--persona", default="data/chat/persona.jsonl")
    p.add_argument("--persona-repeat", type=int, default=25)
    p.add_argument("--ultrachat", type=int, default=20000, help="conversations; 0 to skip")
    p.add_argument("--oasst", type=int, default=6000, help="conversations; 0 to skip")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--val-size", type=int, default=400)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--sample-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-grad-checkpointing", action="store_true",
                   help="faster, more memory: recomputing activations costs ~30% of throughput")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"base        {args.base}")
    train_records, val_records = load_sources(args, tok)

    train_data = ChatDataset(train_records, tok, args.max_len)
    val_data = ChatDataset(val_records, tok, args.max_len)
    tokens = sum(len(ids) for ids, _ in train_data.examples)
    graded = sum(sum(1 for t in labels if t != IGNORE) for _, labels in train_data.examples)
    print(f"train       {len(train_data):,} conversations, {tokens:,} tokens "
          f"({graded:,} graded, {graded / max(tokens, 1):.0%})")
    print(f"val         {len(val_data):,} conversations")
    print(f"dropped     {train_data.dropped + val_data.dropped:,} unusable rows")

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).to(device)
    seeded = init_chat_tokens(model, tok)
    print("chatml      " + "  ".join(f"{k} norm {v:.3f}" for k, v in seeded.items()))
    model.config.use_cache = False          # incompatible with gradient checkpointing
    if not args.no_grad_checkpointing:
        # Trades compute for memory: activations are thrown away on the forward pass and
        # recomputed during the backward. Roughly 30% slower, and the difference between
        # fitting in 8GB and not.
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # else checkpointing sees no grad-requiring input

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        # Every linear layer, attention and MLP alike. Attention-only LoRA is cheaper and
        # noticeably worse at absorbing a new response style.
        # lm_head is in the list on purpose. Without it nothing can move the output
        # distribution over the ChatML markers, and the model never learns to stop -- see
        # init_chat_tokens. It costs 2048x32 + 32x151936 = 4.9M extra parameters.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", "lm_head"],
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"lora        r={args.lora_r} alpha={args.lora_alpha}; "
          f"{trainable:,} trainable of {total:,} ({trainable / total:.2%})")

    batches = length_grouped_batches(train_data, args.batch_size, seed=args.seed)
    val_batches = length_grouped_batches(val_data, args.batch_size, seed=args.seed)
    steps_per_epoch = max(1, len(batches) // args.accum)
    max_steps = int(steps_per_epoch * args.epochs)
    print(f"schedule    {max_steps:,} optimizer steps "
          f"({args.batch_size}x{args.accum} = {args.batch_size * args.accum} conversations each)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, max_steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))

    start_step = 0
    if args.resume and (out / "adapter" / "adapter_model.safetensors").exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(out / "adapter" / "adapter_model.safetensors"))
        state = json.loads((out / "state.json").read_text())
        start_step = state["step"]
        opt.load_state_dict(torch.load(out / "optimizer.pt", map_location=device))
        print(f"resumed     from step {start_step}")

    metrics = (out / "metrics.jsonl").open("a", encoding="utf-8")
    pad_id = tok.pad_token_id
    model.train()
    t0 = time.perf_counter()
    cursor = (start_step * args.accum) % len(batches)
    seen = 0

    for step in range(start_step, max_steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step)

        running = 0.0
        for _ in range(args.accum):
            if cursor >= len(batches):           # next epoch: reshuffle the batch order
                batches = length_grouped_batches(train_data, args.batch_size,
                                                 seed=args.seed + step)
                cursor = 0
            x, y = collate([train_data[i] for i in batches[cursor]], pad_id)
            cursor += 1
            x, y = x.to(device), y.to(device)
            seen += int((y != IGNORE).sum())

            logits = model(input_ids=x).logits
            # The loss is computed here rather than by passing `labels=` to the model, because
            # chat_data already shifted them. Letting transformers shift again would train the
            # model to predict the token it is currently reading -- and the loss would still fall.
            loss = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten(), ignore_index=IGNORE)
            if torch.isnan(loss):
                continue                          # a batch with nothing graded; skip it
            (loss / args.accum).backward()
            running += loss.item() / args.accum

        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 10 == 0 or step == max_steps - 1:
            rate = seen / (time.perf_counter() - t0)
            print(f"  step {step:>5}  loss {running:7.4f}  lr {lr_at(step):.2e}  "
                  f"{rate:>8,.0f} graded tok/s", flush=True)
            metrics.write(json.dumps({"kind": "train", "step": step, "loss": running,
                                      "lr": lr_at(step)}) + "\n")
            metrics.flush()

        if args.eval_every and (step + 1) % args.eval_every == 0:
            val = evaluate(model, val_batches, val_data, pad_id, device, limit=40)
            print(f"    val loss {val:.4f}   perplexity {math.exp(val):.2f}", flush=True)
            metrics.write(json.dumps({"kind": "eval", "step": step, "loss": val}) + "\n")
            metrics.flush()

        if args.sample_every and (step + 1) % args.sample_every == 0:
            model.config.use_cache = True
            for messages in PROBE_PROMPTS:
                print(f"    > {messages[-1]['content']}\n      {sample(model, tok, messages)}",
                      flush=True)
            model.config.use_cache = False

        if args.save_every and (step + 1) % args.save_every == 0:
            model.save_pretrained(out / "adapter")
            torch.save(opt.state_dict(), out / "optimizer.pt")
            (out / "state.json").write_text(json.dumps({"step": step + 1, "base": args.base}))

    model.save_pretrained(out / "adapter")
    (out / "state.json").write_text(json.dumps({"step": max_steps, "base": args.base}))
    print(f"\ndone in {(time.perf_counter() - t0) / 60:.1f} min -> {out / 'adapter'}")
    metrics.close()


if __name__ == "__main__":
    main()
