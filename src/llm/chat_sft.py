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
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from llm.chat_data import IGNORE, ChatDataset, collate, length_grouped_batches, render
from llm.chat_infer import find_checkpoint
from llm.console import use_utf8
from llm.optim import lr_at_step

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

#: The same behaviours, in words the model has never been trained on. Every probe above is copied
#: verbatim from the hand-written data, which training repeats 25 times -- so a perfect answer to
#: one of them shows recall, not that the behaviour generalized. These are the ones that can show
#: the difference. A test keeps them out of the training data.
HELDOUT_PROMPTS = [
    [{"role": "user", "content": "hey there, how's it going?"}],
    [{"role": "user", "content": "do u remember me from yesterday?"}],
    [{"role": "user", "content": "are you a real person or a robot?"}],
    [{"role": "user", "content": "I failed my exam today"}],
    [{"role": "user", "content": "i'm Aung and i'm 22"},
     {"role": "assistant", "content": "Nice to meet you, Aung."},
     {"role": "user", "content": "how old am i, and what's my name?"}],
    [{"role": "user", "content": "\u1019\u1004\u103a\u1039\u1002\u101c\u102c\u1015\u102b"}],
]

#: How the conversations are split into training and validation. Stored with every checkpoint, so
#: a resumed run rebuilds exactly the data it began with even after this changes.
#:   1  repeat the hand-written set, then split. Copies land on both sides: 22% of validation was
#:      conversations the model also trained on, so validation loss partly measured recall.
#:   2  split first, validation from the external data only, then repeat the hand-written set
#:      into training. Nothing in validation is ever trained on.
DATA_RECIPE = 2


def split_records(
    persona: list[dict], external: list[dict], repeat: int, val_size: int, seed: int,
    recipe: int = DATA_RECIPE,
) -> tuple[list[dict], list[dict]]:
    """Training and validation conversations, by the given recipe. Deterministic in `seed`."""
    rng = random.Random(seed)
    if recipe == 1:
        # Kept exactly as it was, so runs started under it resume onto the same data.
        records = persona * repeat + external
        rng.shuffle(records)
        n = min(val_size, len(records) // 10)
        return records[n:], records[:n]
    if recipe != 2:
        raise ValueError(f"unknown data recipe {recipe}")
    external = list(external)
    rng.shuffle(external)
    n = min(val_size, len(external) // 10)
    train = persona * repeat + external[n:]
    rng.shuffle(train)
    return train, external[:n]


def data_fingerprint(train: list[dict], val: list[dict]) -> str:
    """A short hash of exactly which conversations, in which order, a run trains and validates on."""
    blob = json.dumps([[r["messages"] for r in train], [r["messages"] for r in val]],
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def check_same_data(state: dict, fingerprint: str,
                    source: str = "data/chat/persona.jsonl") -> None:
    """Refuse to resume onto different data than the run started with.

    Resume rebuilds the data from its sources, and one of them -- `source` -- is a file anyone can
    edit. Change it mid-run and a resume would carry on, from the right step with the right
    optimizer state, over different data, and nothing would say so. Checkpoints written before
    this check have no fingerprint and cannot be checked.
    """
    expected = state.get("data_fingerprint")
    if expected and expected != fingerprint:
        # Not "start a fresh run here": check_fresh_start refuses that, to protect this run.
        raise SystemExit(
            f"the training data has changed since this run started (fingerprint {expected}, now "
            f"{fingerprint}) -- most likely {source} was edited. Resuming would continue on "
            f"different data. Put the file back as it was, or start a new run with a different "
            f"--out."
        )


def check_fresh_start(out: Path, resume: bool) -> None:
    """Refuse to start a new run on top of an existing one.

    A fresh run saves step-000100, step-000200, ... into the same folder, and pruning keeps only
    the newest two -- so two saves in, the previous run's checkpoints, a finished model included,
    would be deleted. One mistaken double-click on run.bat instead of resume.bat would do it.
    """
    if not resume and find_checkpoint(out) is not None:
        raise SystemExit(
            f"{out} already holds a run (newest checkpoint: {find_checkpoint(out)}). Starting a "
            f"new one here would prune it away. Continue it with --resume, or pass a different "
            f"--out for a new run."
        )


def load_sources(args, tok, recipe: int = DATA_RECIPE) -> tuple[list[dict], list[dict]]:
    """Gather conversations from every source, then split them by `recipe`.

    The persona file is tiny and aimed at known failures, so it is repeated: a few hundred good
    targeted examples shift behaviour more than another 50k generic ones, but only if the model
    sees them often enough to matter.
    """
    from datasets import load_dataset

    persona: list[dict] = []
    external: list[dict] = []

    if args.persona and Path(args.persona).exists():
        persona = [json.loads(line) for line in
                   Path(args.persona).read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"persona     {len(persona)} conversations x{args.persona_repeat} "
              f"= {len(persona) * args.persona_repeat}")

    if args.ultrachat:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{args.ultrachat}]")
        external += [{"messages": r["messages"]} for r in ds]
        print(f"ultrachat   {len(ds)} conversations")

    if args.oasst:
        ds = load_dataset("OpenAssistant/oasst2", split="train")
        external += oasst_conversations(ds, args.oasst)
        print(f"oasst2      up to {args.oasst} conversations")

    return split_records(persona, external, args.persona_repeat, args.val_size, args.seed, recipe)


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
        messages: list[dict] = []
        node: dict | None = root
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
        ids = tok(render(messages), return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        out = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"), pad_token_id=tok.pad_token_id,
        )
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    finally:
        model.train(was_training)


@torch.no_grad()
def evaluate(model, batches, data, pad_id, device, limit: int) -> float:
    """Mean loss per graded token over the first `limit` validation batches.

    Per token, not per conversation: averaging each conversation's mean gave a two-line greeting
    the same weight as a forty-line explanation. Numbers from before this change read slightly
    differently for that reason alone.
    """
    model.eval()
    total, tokens = 0.0, 0
    for batch in batches[:limit]:
        x, y = collate([data[i] for i in batch], pad_id)
        x, y = x.to(device), y.to(device)
        logits = model(input_ids=x).logits
        total += F.cross_entropy(logits.flatten(0, 1).float(), y.flatten(),
                                 ignore_index=IGNORE, reduction="sum").item()
        tokens += int((y != IGNORE).sum())
    model.train()
    # Nothing scored has no mean. The old `max(tokens, 1)` turned it into a perfect 0.0000.
    return total / tokens if tokens else float("nan")


def epoch_batches(data, batch_size: int, seed: int, epoch: int) -> list[list[int]]:
    """The batch order for one epoch, drawn from (seed, epoch) alone.

    Epoch 0 is exactly the order this run always used, so a run paused before this change resumes
    onto the same stream. Before, later epochs were reshuffled with `seed + step` -- whatever step
    the wrap happened to land on -- and a resume always rebuilt epoch 0, so resuming anywhere in
    epoch two would have replayed epoch one. Neither mattered for the one-epoch run; both would
    have on the default of two.
    """
    return length_grouped_batches(data, batch_size, seed=seed + epoch)


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def save_checkpoint(out: Path, model, opt, step: int, base: str, keep: int = 2,
                    finished: bool = False, recipe: int | None = DATA_RECIPE,
                    fingerprint: str | None = None, extra: dict | None = None) -> Path:
    """Write a complete checkpoint into its own directory, then point `latest.json` at it.

    A save is 1.1 GB -- the adapter carries the 311M-entry embedding table because lm_head is a
    LoRA target -- and stop.bat ends training with a hard kill. The first version wrote the
    adapter, the optimizer and the step count over the previous ones in turn, so a kill mid-save
    left a truncated adapter, or a new adapter beside an old optimizer. Here everything goes into
    a `.partial` directory first, which is renamed into place whole; the pointer, a few bytes
    written atomically, moves last. A kill at any moment leaves the pointer on a checkpoint that
    finished writing.
    """
    final = out / f"step-{step:06d}"
    partial = out / f".{final.name}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    model.save_pretrained(partial / "adapter")
    torch.save(opt.state_dict(), partial / "optimizer.pt")
    rng: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng, partial / "rng.pt")
    (partial / "state.json").write_text(
        json.dumps({"step": step, "base": base, "finished": finished, "data_recipe": recipe,
                    "data_fingerprint": fingerprint, **(extra or {})}),
        encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    os.replace(partial, final)
    _write_json_atomic(out / "latest.json", {"dir": final.name, "step": step})

    # Keep the newest few, never the one the pointer names, and clear any abandoned partials.
    saved = sorted(out.glob("step-[0-9]*"), key=lambda d: d.name)
    for old in saved[:-keep]:
        if old != final:
            shutil.rmtree(old, ignore_errors=True)
    for stale in out.glob(".step-*.partial"):
        shutil.rmtree(stale, ignore_errors=True)
    return final


@dataclass
class RunData:
    """The conversations a run trains and validates on, fixed for the life of the run."""

    train: ChatDataset
    val: ChatDataset
    recipe: int
    fingerprint: str


def read_state(checkpoint: Path) -> dict:
    return json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--out", default="checkpoints/chat")
    p.add_argument("--persona", default="data/chat/persona.jsonl")
    p.add_argument("--persona-repeat", type=int, default=25)
    p.add_argument("--ultrachat", type=int, default=20000, help="conversations; 0 to skip")
    p.add_argument("--oasst", type=int, default=6000, help="conversations; 0 to skip")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--val-size", type=int, default=400)
    p.add_argument("--val-batches", type=int, default=100,
                   help="validation conversations scored at each eval; 40 was too noisy to read")
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
    p.add_argument("--device", default=None, help="cuda or cpu; default cuda when available")
    p.add_argument("--no-grad-checkpointing", action="store_true",
                   help="faster, more memory: recomputing activations costs ~30% of throughput")
    return p.parse_args(argv)


def prepare_data(args: argparse.Namespace, tok, found: Path | None) -> RunData:
    """The run's conversations -- for a resumed run, exactly the ones it started with.

    The split recipe is read from the checkpoint before anything is built, and the result is
    fingerprinted and compared with the checkpoint's, so a resume can neither change how the data
    is split nor carry on over conversations edited in the meantime.
    """
    recipe = read_state(found).get("data_recipe", 1) if found is not None else DATA_RECIPE
    kept = " (kept from the run being resumed)" if recipe != DATA_RECIPE else ""
    print(f"data        recipe {recipe}{kept}")
    train_records, val_records = load_sources(args, tok, recipe)
    fingerprint = data_fingerprint(train_records, val_records)
    if found is not None:
        check_same_data(read_state(found), fingerprint)
    print(f"            fingerprint {fingerprint}")

    train = ChatDataset(train_records, tok, args.max_len)
    val = ChatDataset(val_records, tok, args.max_len)
    tokens = sum(len(ids) for ids, _ in train.examples)
    graded = sum(sum(1 for t in labels if t != IGNORE) for _, labels in train.examples)
    print(f"train       {len(train):,} conversations, {tokens:,} tokens "
          f"({graded:,} graded, {graded / max(tokens, 1):.0%})")
    print(f"val         {len(val):,} conversations"
          + ("" if len(val) else " -- no validation loss this run; it comes from the external "
                                 "data only"))
    print(f"dropped     {train.dropped + val.dropped:,} unusable rows")
    return RunData(train, val, recipe, fingerprint)


def build_model(args: argparse.Namespace, tok, device: str):
    """The base model with its ChatML markers seeded, wrapped for LoRA."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    # bf16 on the GPU, where it halves memory at no cost in quality. On the CPU, fp32: most CPUs
    # have no bf16 matrix units, so bf16 there is slower and only less accurate.
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype)
    # transformers wraps .to() in a decorator that hides its signature from mypy.
    model = model.to(device)  # type: ignore[arg-type]
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
    wrapped = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    total = sum(p.numel() for p in wrapped.parameters())
    print(f"lora        r={args.lora_r} alpha={args.lora_alpha}; "
          f"{trainable:,} trainable of {total:,} ({trainable / total:.2%})")
    return wrapped


def schedule_length(args: argparse.Namespace, data: RunData) -> int:
    """Optimizer steps in the run: every conversation `epochs` times, batch_size x accum a step."""
    per_epoch = math.ceil(len(data.train) / args.batch_size)
    return int(max(1, per_epoch // args.accum) * args.epochs)


def restore(found: Path, out: Path, model, opt, device: str) -> int:
    """Load a checkpoint's adapter, optimizer state and random state. Returns the step it reached."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    # peft returns torch's _IncompatibleKeys, a namedtuple its annotations do not name.
    loaded: Any = set_peft_model_state_dict(
        model, load_file(found / "adapter" / "adapter_model.safetensors"))
    if loaded.unexpected_keys:
        # A partial load would carry on from half-fresh weights while claiming to resume.
        raise SystemExit(f"adapter at {found} does not fit this model: "
                         f"{loaded.unexpected_keys[:3]} ...")
    step = read_state(found)["step"]
    opt.load_state_dict(torch.load(found / "optimizer.pt", map_location=device))
    if (found / "rng.pt").exists():
        rng = torch.load(found / "rng.pt")
        torch.set_rng_state(rng["cpu"])
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    where = found.name if found != out else "original layout"
    print(f"resumed     from step {step} ({where})")
    return step


def print_samples(model, tok) -> None:
    """The probe prompts' answers -- memorized and never seen -- side by side in the log."""
    model.config.use_cache = True
    for label, prompts in (("seen in training", PROBE_PROMPTS), ("never seen", HELDOUT_PROMPTS)):
        print(f"    -- {label} --")
        for messages in prompts:
            print(f"    > {messages[-1]['content']}\n      {sample(model, tok, messages)}",
                  flush=True)
    model.config.use_cache = False


def run_training(args: argparse.Namespace, model, opt, tok, data: RunData, out: Path,
                 device: str, start_step: int, max_steps: int) -> Path:
    """Optimizer steps from `start_step` to `max_steps`, logging, evaluating, sampling and saving
    on their schedules. Returns the final checkpoint's directory."""

    def lr_at(step: int) -> float:
        # The same warmup-then-cosine schedule as pretraining (notes/05), floored at 10% of peak.
        return args.lr * lr_at_step(step, warmup_steps=args.warmup, total_steps=max_steps,
                                    min_lr_ratio=0.1)

    val_batches = length_grouped_batches(data.val, args.batch_size, seed=args.seed)
    # Where the run is in the data is arithmetic on the step, so resuming is a seek.
    epoch, cursor = divmod(start_step * args.accum, math.ceil(len(data.train) / args.batch_size))
    batches = epoch_batches(data.train, args.batch_size, args.seed, epoch)
    pad_id = tok.pad_token_id
    trainable = [p for p in model.parameters() if p.requires_grad]
    model.train()
    t0 = time.perf_counter()
    seen = 0

    with (out / "metrics.jsonl").open("a", encoding="utf-8") as metrics:

        def record(**fields) -> None:
            metrics.write(json.dumps(fields) + "\n")
            metrics.flush()

        for step in range(start_step, max_steps):
            for group in opt.param_groups:
                group["lr"] = lr_at(step)

            running = 0.0
            for _ in range(args.accum):
                if cursor >= len(batches):           # next epoch: its own order, from (seed, epoch)
                    epoch += 1
                    batches = epoch_batches(data.train, args.batch_size, args.seed, epoch)
                    cursor = 0
                x, y = collate([data.train[i] for i in batches[cursor]], pad_id)
                cursor += 1
                x, y = x.to(device), y.to(device)
                seen += int((y != IGNORE).sum())

                logits = model(input_ids=x).logits
                # The loss is computed here rather than by passing `labels=` to the model, because
                # chat_data already shifted them. Letting transformers shift again would train the
                # model to predict the token it is reading -- and the loss would still fall.
                loss = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten(), ignore_index=IGNORE)
                if torch.isnan(loss):
                    continue                          # a batch with nothing graded; skip it
                (loss / args.accum).backward()
                running += loss.item() / args.accum

            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            if step % 10 == 0 or step == max_steps - 1:
                rate = seen / (time.perf_counter() - t0)
                print(f"  step {step:>5}  loss {running:7.4f}  lr {lr_at(step):.2e}  "
                      f"{rate:>8,.0f} graded tok/s", flush=True)
                record(kind="train", step=step, loss=running, lr=lr_at(step))

            if args.eval_every and len(data.val) and (step + 1) % args.eval_every == 0:
                val = evaluate(model, val_batches, data.val, pad_id, device, limit=args.val_batches)
                print(f"    val loss {val:.4f}   perplexity {math.exp(val):.2f}", flush=True)
                record(kind="eval", step=step, loss=val)

            if args.sample_every and (step + 1) % args.sample_every == 0:
                print_samples(model, tok)

            if args.save_every and (step + 1) % args.save_every == 0 and step + 1 < max_steps:
                save_checkpoint(out, model, opt, step + 1, args.base, recipe=data.recipe,
                                fingerprint=data.fingerprint)

    final = save_checkpoint(out, model, opt, max_steps, args.base, finished=True,
                            recipe=data.recipe, fingerprint=data.fingerprint)
    print(f"\ndone in {(time.perf_counter() - t0) / 60:.1f} min -> {final / 'adapter'}")
    return final


def main(argv: list[str] | None = None) -> None:
    """Six steps: arguments, the run's data, the model, the schedule, any checkpoint, the loop."""
    from transformers import AutoTokenizer

    # run.bat sends this output to a log file, which Python writes in the Windows code page. The
    # Burmese never-seen probe cannot be written in it: the first sample print raised, before that
    # step's checkpoint was saved, so every resume would have died at the same step.
    use_utf8()
    args = parse_args(argv)
    out = Path(args.out)
    check_fresh_start(out, args.resume)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"base        {args.base}")

    # Found before the data is built: a resumed run must rebuild the data it started with.
    found = find_checkpoint(out) if args.resume else None
    data = prepare_data(args, tok, found)
    model = build_model(args, tok, device)
    max_steps = schedule_length(args, data)
    print(f"schedule    {max_steps:,} optimizer steps "
          f"({args.batch_size}x{args.accum} = {args.batch_size * args.accum} conversations each)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    start_step = 0
    if args.resume and found is None:
        print("resumed     nothing to resume in " + str(out) + " -- starting from step 0")
    if found is not None:
        start_step = restore(found, out, model, opt, device)

    run_training(args, model, opt, tok, data, out, device, start_step, max_steps)
    print("talk to it:  python scripts/talk.py")


if __name__ == "__main__":
    main()
