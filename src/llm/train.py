"""The training loop.

    forward -> loss -> backward -> clip -> step -> zero_grad

Everything else in this file is bookkeeping around those six operations: mixed precision,
gradient accumulation, evaluation, sampling, checkpointing and logging.

Run it:

    python -m llm.train --preset small --data data/raw/the-verdict.txt --steps 600 --overfit

See notes/05_training.md.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from llm import checkpoint
from llm.config import PRESETS, ModelConfig
from llm.console import use_utf8
from llm.dataset import MAX_TOKEN_ID, TOKEN_DTYPE, make_dataloader
from llm.losses import evaluate
from llm.model import GPTModel
from llm.optim import make_optimizer, make_schedule
from llm.tokenizer import BPETokenizer


@dataclass
class TrainConfig:
    lr: float = 3e-4
    batch_size: int = 8
    accum_steps: int = 1
    """Micro-batches per optimizer step. Effective batch = batch_size * accum_steps."""
    max_steps: int = 1_000
    warmup_steps: int = 50
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    bf16: bool = True
    eval_every: int = 50
    eval_batches: int = 10
    sample_every: int = 0
    sample_prompt: str = "Every effort moves you"
    sample_tokens: int = 25
    log_every: int = 10
    seed: int = 0
    out_dir: Path = Path("checkpoints")
    save_every: int = 0
    stop_at: int | None = None
    """Stop cleanly before this step, saving a checkpoint, without shortening the schedule.

    For "run 35,000 of 40,000 tonight and the rest tomorrow": the learning rate still follows the
    40,000-step curve, and resuming finishes the run exactly as if it had never paused."""


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_text_corpus(
    path: str | Path, tok: BPETokenizer, val_fraction: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize one text file and split it by position.

    Splitting a single continuous document by position is not as clean as splitting by document
    (docs/DATA_GUIDE.md): the two halves share style and vocabulary, so validation loss is
    optimistic. For a correctness run on one short story that is acceptable and worth stating.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction={val_fraction} must be in (0, 1)")
    text = Path(path).read_text(encoding="utf-8")
    wide = np.fromiter(tok.encode(text), dtype=np.int64)
    if wide.size and wide.max() > MAX_TOKEN_ID:
        raise ValueError(f"token id {wide.max()} does not fit in {TOKEN_DTYPE.__name__}")
    ids = wide.astype(TOKEN_DTYPE)
    cut = int(len(ids) * (1 - val_fraction))
    return ids[:cut], ids[cut:]


def check_vocabulary(cfg: ModelConfig, ids: np.ndarray, source: str) -> None:
    """Refuse to start if the corpus contains ids the embedding table has no row for.

    On CPU this would surface as an IndexError somewhere inside nn.Embedding. On CUDA it is a
    device-side assert: asynchronous, so the traceback points at an unrelated line, and fatal to
    the whole context, so nothing after it is recoverable. Both happen after the run has started.

    The usual cause is a preset whose vocab_size was chosen for a tokenizer trained on the
    corpus -- TINYSTORIES expects 10,000 -- being fed GPT-2's 50,257-token BPE instead.
    """
    if ids.size == 0:
        raise ValueError(f"{source} produced no tokens")
    highest = int(ids.max())
    if highest >= cfg.vocab_size:
        raise ValueError(
            f"{source} contains token id {highest}, but the model's vocab_size is "
            f"{cfg.vocab_size}. The tokenizer and the preset disagree -- either train a "
            f"tokenizer sized for this preset (docs/CS336_FINDINGS.md) or use a preset whose "
            f"vocab_size matches the tokenizer."
        )


def _batch_stream(loader: DataLoader, seed: int, start: int = 0):
    """Batches forever, as a pure function of (seed, position in the stream).

    The order a run sees its data is part of its state, exactly like the weights and Adam's
    moments, and the first version of this loop stored it nowhere. It iterated a shuffling
    DataLoader, whose permutation is drawn from the global RNG, and `train()` reseeds that RNG on
    every start. So a resumed run did not continue the data stream -- it *restarted* it,
    retraining on the batches from the beginning of the run and never reaching the ones it was
    meant to see. The TinyStories baseline was resumed five times like that, and in 40,000 steps
    saw only 19,000 distinct batches, the earliest of them five times over.

    Here epoch `e` is the permutation drawn from `(seed, e)`, and batch `n` of the run is found by
    arithmetic, so resuming at step `s` is a seek to batch `s * accum_steps`. Nothing extra goes
    into the checkpoint: the step count already says where the stream is.

    The loader is read rather than iterated. A shuffling one gets a fresh permutation per epoch.
    One built from a precomputed list of batches -- length-grouped instruction data -- keeps that
    list for the first epoch and reshuffles the order of its batches for every epoch after, which
    the old loop never did. Anything else is taken in order.
    """
    dataset, collate, batch_sampler = loader.dataset, loader.collate_fn, loader.batch_sampler

    def epoch_batches(epoch: int) -> list[list[int]]:
        rng = np.random.default_rng([seed, epoch])
        if isinstance(batch_sampler, BatchSampler) and isinstance(
            batch_sampler.sampler, RandomSampler
        ):
            order = rng.permutation(len(dataset)).tolist()
            size = batch_sampler.batch_size
            count = len(order) // size if batch_sampler.drop_last else -(-len(order) // size)
            return [order[i * size : (i + 1) * size] for i in range(count)]
        batches = [list(b) for b in batch_sampler]
        if not isinstance(batch_sampler, BatchSampler) and epoch > 0:
            rng.shuffle(batches)
        return batches

    per_epoch = len(epoch_batches(0))
    if per_epoch == 0:
        raise ValueError("the training loader yields no batches -- the corpus is smaller than one")
    epoch, offset = divmod(start, per_epoch)
    while True:
        for indices in epoch_batches(epoch)[offset:]:
            yield collate([dataset[i] for i in indices])
        epoch, offset = epoch + 1, 0


@torch.no_grad()
def _sample(model: GPTModel, tok: BPETokenizer, prompt: str, n: int, device) -> str:
    """A greedy continuation, for watching the model improve during a run.

    Stops at the end-of-text token. Without that, an instruction-tuned model finishes its answer
    and then keeps going into whatever it would have generated next, which makes the sample look
    far worse than the model is.
    """
    from llm.generate import generate

    ids = torch.tensor([tok.encode(prompt)], device=device)
    out = generate(model, ids, n, temperature=0.0, eos_id=getattr(tok, "eot_id", None))
    text = tok.decode(out[0].tolist())
    return text.split("<|endoftext|>")[0] if "<|endoftext|>" in text else text


def train(
    model: GPTModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    tcfg: TrainConfig,
    tok: BPETokenizer | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
    resume_from: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> dict:
    """Run the loop. Returns the loss history.

    `resume_from`: a checkpoint to continue from -- weights, Adam's moment estimates, the
    scheduler's position and the step count. Without all of them a restart is visible in the loss
    curve: fresh moments cause a spike, and a fresh scheduler restarts warmup.

    `metrics_path`: a JSONL file every step and every evaluation is appended to as it happens. The
    in-memory history dies with the process; this is what survives a crash, and what the final
    loss curve is drawn from when a run was resumed partway through.
    """
    if tcfg.max_steps < 1:
        raise ValueError(f"max_steps={tcfg.max_steps} must be >= 1")
    if tcfg.accum_steps < 1:
        raise ValueError(f"accum_steps={tcfg.accum_steps} must be >= 1")
    if tcfg.warmup_steps >= tcfg.max_steps:
        # Warmup that outlasts the run means the cosine never decays: the model finishes at its
        # peak learning rate, and the last stretch of decay is where a surprising share of the
        # final loss improvement comes from.
        raise ValueError(
            f"warmup_steps={tcfg.warmup_steps} must be less than max_steps={tcfg.max_steps}, "
            f"or the learning rate never decays"
        )

    device = device or pick_device()
    torch.manual_seed(tcfg.seed)
    model.to(device).train()

    optimizer = make_optimizer(model, tcfg.lr, tcfg.weight_decay)
    scheduler = make_schedule(
        optimizer, tcfg.warmup_steps, tcfg.max_steps, tcfg.min_lr_ratio
    )
    # bf16 carries fp32's exponent range, so unlike fp16 it needs no gradient scaler.
    use_amp = tcfg.bf16 and device.type == "cuda"

    history: dict[str, list] = {"step": [], "train_loss": [], "val_step": [], "val_loss": [], "lr": []}
    tokens_seen = 0
    tokens_here = 0  # since this process started, for an honest throughput figure
    elapsed_before = 0.0
    start_step = 0

    if resume_from is not None and Path(resume_from).exists():
        extras: dict = {}
        done = checkpoint.load(resume_from, model, optimizer, scheduler, device, extras)
        start_step = done + 1
        # The optimizer's saved rate belongs to the schedule it was saved under. Resuming under a
        # longer one -- training further after a run finished -- would spend the first step at
        # that stale rate, so take it from this run's schedule at the resumed position instead.
        for group, base, fn in zip(optimizer.param_groups, scheduler.base_lrs,
                                   scheduler.lr_lambdas, strict=True):
            group["lr"] = base * fn(scheduler.last_epoch)
        tokens_seen = int(extras.get("tokens_seen", 0))
        elapsed_before = float(extras.get("elapsed", 0.0))
        if verbose:
            print(f"  resumed from {resume_from} at step {start_step} "
                  f"({tokens_seen:,} tokens, {elapsed_before / 60:.1f} min already done)\n")

    # Built after the resume, because where the stream starts depends on the step reached.
    batches = _batch_stream(train_loader, tcfg.seed, start_step * tcfg.accum_steps)

    metrics = None
    if metrics_path is not None:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        # Append, not overwrite: a resumed run continues the record of the run it resumes.
        metrics = open(metrics_path, "a", encoding="utf-8")

    def record(**fields) -> None:
        if metrics is not None:
            metrics.write(json.dumps(fields) + "\n")
            metrics.flush()

    def save_latest(step: int) -> None:
        checkpoint.save(
            Path(tcfg.out_dir) / "latest.pt", model, optimizer, scheduler, step, model.cfg,
            tokens_seen=tokens_seen, elapsed=elapsed_before + time.perf_counter() - start,
        )

    start = time.perf_counter()

    for step in range(start_step, tcfg.max_steps):
        optimizer.zero_grad(set_to_none=True)
        # Summed on the device. Calling .item() per micro-batch forces a sync each time, which
        # stalls the pipeline for a number nothing reads until the step is over.
        accumulated = torch.zeros((), device=device)

        for _ in range(tcfg.accum_steps):
            x, y = next(batches)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                _, loss = model(x, targets=y)
            # Divide before backward, or the effective learning rate is multiplied by
            # accum_steps. Easy to forget, and it looks like a badly tuned LR.
            (loss / tcfg.accum_steps).backward()
            accumulated += loss.detach() / tcfg.accum_steps
            tokens_seen += x.numel()
            tokens_here += x.numel()

        # A single bad batch can otherwise produce an enormous gradient and destroy the weights
        # in one step. Costs nothing; always on.
        if tcfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)

        # The rate this step actually used, read before the scheduler moves on. Reading it after
        # scheduler.step() logged the *next* step's rate, so every logged value was one step early.
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()

        # The one sync per step, rather than one per micro-batch.
        accumulated = float(accumulated)
        if not math.isfinite(accumulated):
            # NaN propagates silently: the loss line keeps printing, the weights are already
            # ruined, and the next hour of compute produces nothing recoverable.
            raise RuntimeError(
                f"loss became {accumulated} at step {step}. Lower the learning rate, check the "
                f"data for out-of-range ids, or raise grad_clip's aggressiveness."
            )

        history["step"].append(step)
        history["train_loss"].append(accumulated)
        history["lr"].append(lr)
        record(kind="train", step=step, loss=accumulated, lr=lr, tokens=tokens_seen)

        if verbose and tcfg.log_every and (step % tcfg.log_every == 0 or step == tcfg.max_steps - 1):
            # Counted, not inferred from batch_size x width: with variable-length batches the
            # width changes every step, so multiplying by the last one is meaningless. Measured
            # since this process started, so a resume does not dilute it.
            run = time.perf_counter() - start
            print(
                f"  step {step:>5}  loss {accumulated:7.4f}  ppl {math.exp(min(accumulated, 20)):>9,.1f}"
                f"  lr {lr:.2e}  {tokens_here / max(run, 1e-9):>8,.0f} tok/s"
            )

        is_last = step == tcfg.max_steps - 1
        if val_loader is not None and tcfg.eval_every and (step % tcfg.eval_every == 0 or is_last):
            stats = evaluate(model, val_loader, tcfg.eval_batches, device)
            history["val_step"].append(step)
            history["val_loss"].append(stats["loss"])
            record(kind="eval", step=step, loss=stats["loss"], tokens=tokens_seen)
            if verbose:
                print(f"    val loss {stats['loss']:.4f}  perplexity {stats['perplexity']:,.1f}")

        if verbose and tok and tcfg.sample_every and (step % tcfg.sample_every == 0 or is_last):
            text = _sample(model, tok, tcfg.sample_prompt, tcfg.sample_tokens, device)
            print(f'    sample: "{text}"')

        # Saved on the last step too, so a finished run is always resumable -- to train further,
        # or to evaluate without redoing any of it.
        stopping = tcfg.stop_at is not None and step + 1 >= tcfg.stop_at and not is_last
        if stopping or (
            tcfg.save_every and step > start_step and (step % tcfg.save_every == 0 or is_last)
        ):
            save_latest(step)
        if stopping:
            if verbose:
                print(f"\n  stopped before step {step + 1:,} of {tcfg.max_steps:,} as asked; "
                      f"resume with --resume to finish the schedule")
            break

    if metrics is not None:
        metrics.close()
    history["elapsed"] = elapsed_before + time.perf_counter() - start
    history["tokens_seen"] = tokens_seen
    history["finished"] = not history["step"] or history["step"][-1] == tcfg.max_steps - 1
    # Carried so the plot can draw the right "pure ignorance" line without being told twice.
    history["vocab_size"] = model.cfg.vocab_size
    return history


def plot_history(history: dict, path: str | Path, vocab_size: int | None = None) -> Path | None:
    """Loss curve. Returns None if matplotlib is unavailable.

    The dashed reference line is ln(vocab_size) -- the loss of a model that knows nothing. It has
    to come from the config: drawn at ln(50257) on a run with a 10,000-token vocabulary it sits
    1.6 nats too high, and every curve underneath it looks better than it is.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    vocab_size = vocab_size or history.get("vocab_size")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["step"], history["train_loss"], label="train", lw=1, alpha=0.8)
    if history["val_step"]:
        ax.plot(history["val_step"], history["val_loss"], label="validation", lw=2, marker="o", ms=3)
    if vocab_size:
        ignorance = math.log(vocab_size)
        ax.axhline(ignorance, ls="--", c="gray", lw=1,
                   label=f"ln({vocab_size:,}) = {ignorance:.2f}")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def read_metrics(path: str | Path) -> dict:
    """Rebuild the loss history from a metrics log, for plotting a run that was resumed.

    After a crash the steps since the last checkpoint are trained twice. The later record of each
    step wins, so the curve shows the run that actually produced the final weights.
    """
    train_by_step: dict[int, tuple[float, float]] = {}
    eval_by_step: dict[int, float] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["kind"] == "train":
                train_by_step[r["step"]] = (r["loss"], r["lr"])
            elif r["kind"] == "eval":
                eval_by_step[r["step"]] = r["loss"]
    steps = sorted(train_by_step)
    val_steps = sorted(eval_by_step)
    return {
        "step": steps,
        "train_loss": [train_by_step[s][0] for s in steps],
        "lr": [train_by_step[s][1] for s in steps],
        "val_step": val_steps,
        "val_loss": [eval_by_step[s] for s in val_steps],
    }


def _load_split(path: str | Path) -> np.ndarray:
    from llm.dataset import load_tokens

    return load_tokens(path)


def main() -> None:
    use_utf8()
    p = argparse.ArgumentParser(description="Train a GPT.")
    p.add_argument("--preset", default="small", choices=list(PRESETS))
    p.add_argument("--data", default="data/raw/the-verdict.txt",
                   help="a .txt file (split 90/10 here) or a pre-tokenized .bin from llm.prepare")
    p.add_argument("--val-data", default=None, help="the validation .bin, when --data is a .bin")
    p.add_argument("--tokenizer", default=None,
                   help="a tokenizer saved by llm.bpe; default is GPT-2's")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--context", type=int, default=None, help="override context_length")
    p.add_argument("--layers", type=int, default=None, help="override n_layers")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--sample-every", type=int, default=100)
    p.add_argument("--sample-prompt", default="Every effort moves you")
    p.add_argument("--save-every", type=int, default=0,
                   help="checkpoint every N steps; a crash then costs at most N steps")
    p.add_argument("--resume", action="store_true", help="continue from <out>/latest.pt")
    p.add_argument("--stop-at", type=int, default=None,
                   help="stop cleanly before this step without shortening the schedule; "
                        "finish later with --resume")
    p.add_argument("--final-eval", action="store_true",
                   help="score the whole validation set at the end, not a sample of it")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overfit", action="store_true",
                   help="correctness run: no weight decay or LR floor, so loss can reach ~0")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    cfg: ModelConfig = PRESETS[args.preset]
    if args.context:
        cfg = replace(cfg, context_length=args.context)
    if args.layers:
        cfg = replace(cfg, n_layers=args.layers)

    if args.tokenizer:
        from llm.bpe import Tokenizer

        tok = Tokenizer.load(args.tokenizer)
    else:
        tok = BPETokenizer()
    if tok.vocab_size != cfg.vocab_size:
        # Caught here, before a single step: an embedding table the wrong size for the tokenizer
        # either crashes on an out-of-range id or silently wastes rows that are never used.
        raise SystemExit(
            f"tokenizer has {tok.vocab_size:,} tokens but preset '{args.preset}' expects "
            f"{cfg.vocab_size:,}. Pass the matching --tokenizer, or a matching --preset."
        )

    if str(args.data).endswith(".bin"):
        if not args.val_data:
            raise SystemExit("--data is a .bin, so --val-data must name the validation .bin")
        train_ids, val_ids = _load_split(args.data), _load_split(args.val_data)
        # A real corpus has far more text than the run will see, so windows need not overlap.
        stride = cfg.context_length
    else:
        train_ids, val_ids = load_text_corpus(args.data, tok)
        stride = cfg.context_length // 4  # overlap, so a small corpus yields enough windows
    check_vocabulary(cfg, train_ids, f"{args.data} (train)")
    check_vocabulary(cfg, val_ids, f"{args.val_data or args.data} (validation)")

    train_loader = make_dataloader(train_ids, cfg.context_length, stride, args.batch_size)
    val_loader = make_dataloader(
        val_ids, cfg.context_length, cfg.context_length, args.batch_size,
        shuffle=False, drop_last=False,
    )

    tcfg = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        accum_steps=args.accum,
        max_steps=args.steps,
        warmup_steps=args.warmup,
        min_lr_ratio=args.min_lr_ratio,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        sample_every=args.sample_every,
        sample_prompt=args.sample_prompt,
        save_every=args.save_every,
        out_dir=Path(args.out),
        seed=args.seed,
        stop_at=args.stop_at,
        **({"weight_decay": 0.0, "min_lr_ratio": 0.0} if args.overfit else {}),
    )

    device = torch.device("cpu") if args.cpu else pick_device()
    # Seed *before* the model exists. train() reseeds for batch order and dropout, but by then
    # the weights have already been drawn -- so without this the run is not reproducible and the
    # seed field quietly means less than it looks like it does.
    torch.manual_seed(tcfg.seed)
    model = GPTModel(cfg)

    out = Path(args.out)
    metrics_path = out / "metrics.jsonl"
    resume_from = out / "latest.pt" if args.resume else None
    if not args.resume and metrics_path.exists():
        # A fresh run must not append to a previous run's record and draw both as one curve.
        metrics_path.unlink()

    tokens_per_step = args.batch_size * args.accum * cfg.context_length
    print(f"device       {device}  ({torch.cuda.get_device_name(0) if device.type == 'cuda' else ''})")
    print(f"preset       {args.preset}  {model.num_parameters():,} parameters "
          f"({model.num_parameters(non_embedding=True):,} non-embedding)")
    print(f"tokenizer    {args.tokenizer or 'GPT-2 (tiktoken)'}  {tok.vocab_size:,} tokens")
    print(f"corpus       {len(train_ids):,} train / {len(val_ids):,} val tokens")
    print(f"schedule     {args.steps:,} steps x {tokens_per_step:,} tokens = "
          f"{args.steps * tokens_per_step:,} tokens; warmup {args.warmup}, lr {args.lr}")
    if args.save_every:
        print(f"checkpoints  every {args.save_every} steps -> {out / 'latest.pt'}")
    if args.overfit:
        print("mode         overfit - weight decay and LR floor disabled so loss can reach ~0")
    print()

    history = train(
        model, train_loader, val_loader, tcfg, tok, device,
        resume_from=resume_from, metrics_path=metrics_path,
    )
    if not history["finished"]:
        # A paused run has no final weights yet; writing final.pt would label half a schedule
        # as the finished model.
        print(f"paused; {out / 'latest.pt'} holds the run. Resume with the same arguments "
              f"plus --resume.")
        return

    checkpoint.save(out / "final.pt", model, step=tcfg.max_steps, cfg=cfg,
                    tokens_seen=history["tokens_seen"])
    # Drawn from the log rather than memory, so a resumed run plots from step 0.
    curve = plot_history(read_metrics(metrics_path), out / "loss.png", cfg.vocab_size)

    print(f"\nfinished: {history['tokens_seen']:,} tokens, {history['elapsed'] / 60:.1f} min in total")
    if history["train_loss"]:
        print(f"train loss {history['train_loss'][-1]:.4f} (last step)")
    if args.final_eval:
        print("scoring the full validation set...")
        stats = evaluate(model, val_loader, max_batches=None, device=device)
        print(f"VALIDATION LOSS {stats['loss']:.4f}  perplexity {stats['perplexity']:.2f}  "
              f"({stats['batches']:,} batches)")
        record_path = out / "final_eval.json"
        record_path.write_text(json.dumps({**stats, "tokens_seen": history["tokens_seen"]}))
    print(f"saved {out / 'final.pt'}" + (f" and {curve}" if curve else ""))


if __name__ == "__main__":
    main()
