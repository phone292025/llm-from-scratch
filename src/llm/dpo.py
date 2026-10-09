"""Preference tuning with DPO: teach the model which of two answers is better.

Supervised fine-tuning shows the model good answers and asks it to imitate them. It never shows a
bad answer marked as bad. That is why the chat model, asked about a physicist who never existed,
wrote him a biography: nothing in its training ever said *don't*. Imitation can teach what to say;
it cannot teach what not to.

DPO -- Direct Preference Optimization (Rafailov et al., 2023) -- trains on pairs: one prompt, a
*chosen* answer and a *rejected* one. For each answer it compares how much more likely the model
now makes it than the model it started from (the *reference*):

    loss = -log sigmoid( beta * [ (log pi(chosen) - log ref(chosen))
                                - (log pi(rejected) - log ref(rejected)) ] )

Push the chosen answer up and the rejected one down, relative to where they began. `beta` sets how
far the model may drift from the reference: small beta, big moves allowed.

Two choices make this fit on the laptop:

- **The reference is computed once.** The reference is the chat model before DPO starts, and DPO
  only ever needs its log-probabilities for the fixed set of answers. So they are computed at the
  start and kept as numbers, and the same model -- its LoRA adapter now trainable -- becomes the
  policy. No second copy of a 1.7B model.
- **Only the final reply is scored.** Both answers share everything before it, which would add the
  same amount to both sides of the comparison and cancel.

See notes/09_preference_tuning.md.

    python -m llm.dpo --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from llm.chat_data import DEFAULT_SYSTEM, IGNORE, Message, build_example, collate
from llm.chat_infer import base_of, find_checkpoint, resolve_adapter
from llm.chat_sft import (
    BASE_MODEL,
    check_fresh_start,
    check_same_data,
    init_chat_tokens,
    read_state,
    restore,
    sample,
    save_checkpoint,
)
from llm.console import use_utf8
from llm.optim import lr_at_step

Example = tuple[list[int], list[int]]


# --- the objective -------------------------------------------------------------------------------


def dpo_loss(pi_chosen: Tensor, pi_rejected: Tensor, ref_chosen: Tensor, ref_rejected: Tensor,
             beta: float) -> tuple[Tensor, Tensor]:
    """The DPO loss, and the per-pair margin it is computed from.

    The margin is the difference between the two answers' implicit rewards, `beta * (log pi - log
    ref)`. At the start the policy *is* the reference, every margin is 0, and the loss is exactly
    log 2 -- a coin flip about which answer the model prefers. Training pushes margins positive.
    """
    margin = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
    return -F.logsigmoid(margin).mean(), margin


def sequence_logprobs(model, ids: Tensor, labels: Tensor) -> Tensor:
    """log p(reply | everything before it) for each row: the sum over its graded tokens.

    Only the graded positions are upcast to fp32 and normalized. A reply is a fraction of its row,
    and every position is a 151,936-wide row of logits.
    """
    logits = model(input_ids=ids).logits
    graded = labels != IGNORE
    picked = logits[graded].float()                                  # (graded tokens, vocab)
    token_logprobs = (picked.gather(-1, labels[graded].unsqueeze(-1)).squeeze(-1)
                      - picked.logsumexp(-1))
    rows = graded.nonzero()[:, 0]                                    # same order as logits[graded]
    return torch.zeros(len(ids), device=token_logprobs.device).index_add(0, rows, token_logprobs)


# --- the data ----------------------------------------------------------------------------------


def build_pair(tok, prompt: list[Message], chosen: str, rejected: str, max_len: int = 512,
               system: str | None = DEFAULT_SYSTEM) -> tuple[Example, Example]:
    """The same conversation ending two ways, with only the ending graded in each."""
    pair = []
    for reply in (chosen, rejected):
        convo = [*prompt, {"role": "assistant", "content": reply}]
        pair.append(build_example(tok, convo, max_len, system, last_turn_only=True))
    return pair[0], pair[1]


def problem_with(record: dict) -> str | None:
    """What makes a preference record unusable, or None. Hand-written data deserves a loud error,
    not a quiet drop: a typo here would otherwise just leave a behaviour untrained."""
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages or messages[-1].get("role") != "user":
        return "'messages' must be a conversation ending with the user's turn"
    for key in ("chosen", "rejected"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            return f"'{key}' must be a non-empty answer"
    if record["chosen"].strip() == record["rejected"].strip():
        # Identical answers have a margin of 0 forever: a constant loss and no gradient.
        return "'chosen' and 'rejected' are the same answer"
    return None


class PreferenceDataset:
    """Pairs from JSONL records: {"messages": [...ending with a user turn], "chosen", "rejected"}."""

    def __init__(self, records: list[dict], tok, max_len: int = 512) -> None:
        self.records: list[dict] = []
        self.pairs: list[tuple[Example, Example]] = []
        self.dropped = 0
        for n, r in enumerate(records):
            if (problem := problem_with(r)) is not None:
                raise ValueError(f"preference record {n}: {problem}")
            chosen, rejected = build_pair(tok, r["messages"], r["chosen"], r["rejected"], max_len)
            # A prompt that fills max_len leaves nothing graded, and log p of nothing is 0 for
            # both answers: a pair that can only ever say "no preference".
            if all(t == IGNORE for t in chosen[1]) or all(t == IGNORE for t in rejected[1]):
                self.dropped += 1
                continue
            self.records.append(r)
            self.pairs.append((chosen, rejected))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> tuple[Example, Example]:
        return self.pairs[i]


def load_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"no preference data at {path} -- write it with "
                         f"python scripts/make_preference_data.py")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def records_fingerprint(records: list[dict]) -> str:
    blob = json.dumps(records, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def split_pairs(records: list[dict], val_size: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Training and held-out records. At most a fifth is held out, so a small file still trains."""
    order = np.random.default_rng(seed).permutation(len(records)).tolist()
    n_val = min(val_size, len(records) // 5)
    return [records[i] for i in order[n_val:]], [records[i] for i in order[:n_val]]


def pair_batches(n: int, batch_size: int, seed: int, epoch: int) -> list[list[int]]:
    """Pair indices for one epoch, in an order drawn from (seed, epoch) alone -- so a resumed run
    can seek to its position instead of replaying."""
    order = np.random.default_rng([seed, epoch]).permutation(n).tolist()
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]


# --- scoring -----------------------------------------------------------------------------------


def logprobs_for(model, data: PreferenceDataset, indices: list[int], pad_id: int,
                 device: str) -> tuple[Tensor, Tensor]:
    """log pi(chosen) and log pi(rejected) for the given pairs, from one forward pass."""
    rows = [example for i in indices for example in data[i]]   # chosen, rejected, chosen, ...
    x, y = collate(rows, pad_id)
    lp = sequence_logprobs(model, x.to(device), y.to(device))
    return lp[0::2], lp[1::2]


@torch.no_grad()
def reference_logprobs(model, data: PreferenceDataset, pad_id: int, device: str,
                       batch_size: int) -> tuple[Tensor, Tensor]:
    """The reference's log-probabilities for every pair, computed once before training starts."""
    was_training = model.training
    model.eval()
    chosen, rejected = [], []
    for start in range(0, len(data), batch_size):
        c, r = logprobs_for(model, data, list(range(start, min(start + batch_size, len(data)))),
                            pad_id, device)
        chosen.append(c.cpu())
        rejected.append(r.cpu())
    model.train(was_training)
    if not chosen:
        return torch.zeros(0), torch.zeros(0)
    return torch.cat(chosen), torch.cat(rejected)


def disable_dropout(model: nn.Module) -> None:
    """With dropout on, the policy differs from the reference at step 0 by noise alone, and the
    margins DPO is built on start out random. TRL does the same by default."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0


def summarize(margins: Tensor, chosen_shift: Tensor, rejected_shift: Tensor) -> dict[str, float]:
    """How often the chosen answer is preferred, by how much, and how each answer moved.

    The two shifts are log pi - log ref in nats. DPO only asks that chosen rise *relative to*
    rejected, and it often gets there by lowering both -- worth watching, since an answer whose
    probability falls is one the model is less likely to give, chosen or not.
    """
    return {"accuracy": (margins > 0).float().mean().item(), "margin": margins.mean().item(),
            "chosen_shift": chosen_shift.mean().item(),
            "rejected_shift": rejected_shift.mean().item()}


@torch.no_grad()
def evaluate_pairs(model, data: PreferenceDataset, ref_c: Tensor, ref_r: Tensor, pad_id: int,
                   device: str, beta: float, batch_size: int) -> dict[str, float]:
    """The held-out pairs, scored as the training pairs are."""
    if len(data) == 0:
        return {}
    was_training = model.training
    model.eval()
    pi_c, pi_r = [], []
    for start in range(0, len(data), batch_size):
        c, r = logprobs_for(model, data, list(range(start, min(start + batch_size, len(data)))),
                            pad_id, device)
        pi_c.append(c.cpu())
        pi_r.append(r.cpu())
    model.train(was_training)
    chosen, rejected = torch.cat(pi_c), torch.cat(pi_r)
    loss, margins = dpo_loss(chosen, rejected, ref_c, ref_r, beta)
    return {"loss": loss.item(), **summarize(margins, chosen - ref_c, rejected - ref_r)}


def print_answers(model, tok, records: list[dict], limit: int = 4) -> None:
    """What the model now says to held-out prompts -- the test that matters, since a margin can
    grow on paper while the reply the model actually writes stays the same."""
    model.config.use_cache = True
    for r in records[:limit]:
        print(f"    > {r['messages'][-1]['content']}\n      {sample(model, tok, r['messages'])}",
              flush=True)
    model.config.use_cache = False


# --- the command ---------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sft", default="checkpoints/chat",
                   help="the chat run or adapter to start from; it is also the reference")
    p.add_argument("--base", default=None, help="default: the base model the --sft run used")
    p.add_argument("--data", default="data/chat/preferences.jsonl")
    p.add_argument("--out", default="checkpoints/chat-dpo")
    p.add_argument("--beta", type=float, default=0.1)
    # notes/09 section 8: 5e-5 broke skills no pair was about; 5e-6 changed no reply at all.
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=2, help="pairs per micro-batch")
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--val-size", type=int, default=12,
                   help="pairs held out to watch generalization (at most a fifth of the data)")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--sample-every", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--no-grad-checkpointing", action="store_true")
    return p.parse_args(argv)


def load_policy(base: str, sft_adapter: Path, tok, device: str, grad_checkpointing: bool = True):
    """The chat model with its adapter trainable again: the starting point and, frozen in numbers,
    the reference."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
    # Exactly as the adapter was trained: the markers seeded, the embeddings still tied.
    init_chat_tokens(model, tok)
    model.config.use_cache = False
    if grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    policy = PeftModel.from_pretrained(model, sft_adapter, is_trainable=True)
    disable_dropout(policy)
    return policy.to(device)  # type: ignore[arg-type]


def check_same_start(state: dict, sft_adapter: Path) -> None:
    """Refuse to resume from a different chat adapter than the run began with. The reference is
    rebuilt from it on every resume, and `--sft checkpoints/chat` names whichever checkpoint is
    newest -- so training that run further would quietly move the reference mid-run."""
    began = state.get("sft_adapter")
    if began and Path(began).resolve() != sft_adapter.resolve():
        raise SystemExit(f"this run started from {began}, but --sft now resolves to "
                         f"{sft_adapter}. Resume with --sft {began}, or start a new run with a "
                         f"different --out.")


def main(argv: list[str] | None = None) -> None:
    from transformers import AutoTokenizer

    use_utf8()   # the log is a redirected file, and the samples can be any language
    args = parse_args(argv)
    out = Path(args.out)
    check_fresh_start(out, args.resume)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    sft_adapter = resolve_adapter(args.sft)
    if sft_adapter is None:
        raise SystemExit(f"no trained chat adapter at {args.sft} -- DPO starts from one")
    base = args.base or base_of(args.sft, BASE_MODEL)
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"base        {base}")
    print(f"start from  {sft_adapter}  (also the reference)")

    records = load_records(args.data)
    fingerprint = records_fingerprint(records)
    found = find_checkpoint(out) if args.resume else None
    if found is not None:
        check_same_data(read_state(found), fingerprint, source=args.data)
        check_same_start(read_state(found), sft_adapter)
    train_records, val_records = split_pairs(records, args.val_size, args.seed)
    train = PreferenceDataset(train_records, tok, args.max_len)
    val = PreferenceDataset(val_records, tok, args.max_len)
    print(f"pairs       {len(train)} train, {len(val)} held out, {train.dropped + val.dropped} "
          f"dropped; fingerprint {fingerprint}")
    if len(train) == 0:
        raise SystemExit("no usable preference pairs -- check --data and --max-len")

    model = load_policy(base, sft_adapter, tok, device, not args.no_grad_checkpointing)
    pad_id = tok.pad_token_id
    # The reference: this same model before any DPO step, kept as numbers. Computed before any
    # checkpoint is restored, so a resumed run gets the same reference as an unbroken one.
    t_ref = time.perf_counter()
    ref_c, ref_r = reference_logprobs(model, train, pad_id, device, args.batch_size)
    val_ref_c, val_ref_r = reference_logprobs(model, val, pad_id, device, args.batch_size)
    print(f"reference   log-probabilities for {len(train) + len(val)} pairs in "
          f"{time.perf_counter() - t_ref:.1f}s")

    per_epoch = math.ceil(len(train) / args.batch_size)
    max_steps = int(max(1, per_epoch // args.accum) * args.epochs)
    print(f"schedule    {max_steps} optimizer steps of {args.batch_size}x{args.accum} pairs, "
          f"beta {args.beta}, lr {args.lr:g}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    start_step = 0
    if args.resume and found is None:
        print(f"resumed     nothing to resume in {out} -- starting from step 0")
    if found is not None:
        start_step = restore(found, out, model, opt, device)
    extra = {"kind": "dpo", "sft_adapter": str(sft_adapter), "beta": args.beta}

    # Where the run is in the data is arithmetic on the step, so resuming is a seek.
    epoch, cursor = divmod(start_step * args.accum, per_epoch)
    batches = pair_batches(len(train), args.batch_size, args.seed, epoch)
    model.train()
    t0 = time.perf_counter()
    with (out / "metrics.jsonl").open("a", encoding="utf-8") as metrics:

        def record(**fields) -> None:
            metrics.write(json.dumps(fields) + "\n")
            metrics.flush()

        for step in range(start_step, max_steps):
            lr = args.lr * lr_at_step(step, warmup_steps=args.warmup, total_steps=max_steps,
                                      min_lr_ratio=0.1)
            for group in opt.param_groups:
                group["lr"] = lr

            loss_sum = 0.0
            margins, chosen_shift, rejected_shift = [], [], []
            for _ in range(args.accum):
                if cursor >= len(batches):           # next epoch: its own order, from (seed, epoch)
                    epoch, cursor = epoch + 1, 0
                    batches = pair_batches(len(train), args.batch_size, args.seed, epoch)
                idx = batches[cursor]
                cursor += 1
                pi_c, pi_r = logprobs_for(model, train, idx, pad_id, device)
                rc, rr = ref_c[idx].to(pi_c.device), ref_r[idx].to(pi_r.device)
                loss, margin = dpo_loss(pi_c, pi_r, rc, rr, args.beta)
                (loss / args.accum).backward()
                loss_sum += loss.item() / args.accum
                margins.append(margin.detach().cpu())
                chosen_shift.append((pi_c - rc).detach().cpu())
                rejected_shift.append((pi_r - rr).detach().cpu())

            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            stats = summarize(torch.cat(margins), torch.cat(chosen_shift), torch.cat(rejected_shift))
            print(f"  step {step:>4}  loss {loss_sum:.4f}  prefers chosen {stats['accuracy']:4.0%}  "
                  f"margin {stats['margin']:+.3f}  (chosen {stats['chosen_shift']:+.2f}, "
                  f"rejected {stats['rejected_shift']:+.2f} nats)  lr {lr:.2e}", flush=True)
            record(kind="train", step=step, loss=loss_sum, lr=lr, **stats)

            last = step + 1 == max_steps
            if len(val) and (last or (args.eval_every and (step + 1) % args.eval_every == 0)):
                held = evaluate_pairs(model, val, val_ref_c, val_ref_r, pad_id, device,
                                      args.beta, args.batch_size)
                print(f"    held out: loss {held['loss']:.4f}, prefers chosen "
                      f"{held['accuracy']:.0%}, margin {held['margin']:+.3f} (chosen "
                      f"{held['chosen_shift']:+.2f}, rejected {held['rejected_shift']:+.2f})",
                      flush=True)
                record(kind="eval", step=step, **held)

            if args.sample_every and (step + 1) % args.sample_every == 0:
                print_answers(model, tok, val.records)

            if args.save_every and (step + 1) % args.save_every == 0 and not last:
                save_checkpoint(out, model, opt, step + 1, base, recipe=None,
                                fingerprint=fingerprint, extra=extra)

    final = save_checkpoint(out, model, opt, max_steps, base, finished=True, recipe=None,
                            fingerprint=fingerprint, extra=extra)
    print(f"\ndone in {(time.perf_counter() - t0) / 60:.1f} min -> {final / 'adapter'}")
    print(f"talk to it:  python scripts/talk.py --adapter {out}")


if __name__ == "__main__":
    main()
