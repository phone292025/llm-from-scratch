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
import itertools
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from llm import checkpoint
from llm.config import PRESETS, ModelConfig
from llm.dataset import TOKEN_DTYPE, make_dataloader
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
    text = Path(path).read_text(encoding="utf-8")
    ids = np.fromiter(tok.encode(text), dtype=TOKEN_DTYPE)
    cut = int(len(ids) * (1 - val_fraction))
    return ids[:cut], ids[cut:]


def _infinite(loader: DataLoader):
    """Cycle a loader forever, reshuffling each pass, so training is step-based not epoch-based."""
    while True:
        yield from loader


@torch.no_grad()
def _sample(model: GPTModel, tok: BPETokenizer, prompt: str, n: int, device) -> str:
    ids = torch.tensor([tok.encode(prompt)], device=device)
    return tok.decode(model.generate_greedy(ids, max_new_tokens=n)[0].tolist())


def train(
    model: GPTModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    tcfg: TrainConfig,
    tok: BPETokenizer | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Run the loop. Returns the loss history."""
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
    batches = _infinite(train_loader)
    tokens_seen = 0
    start = time.perf_counter()

    for step in range(tcfg.max_steps):
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0

        for _ in range(tcfg.accum_steps):
            x, y = next(batches)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                _, loss = model(x, targets=y)
            # Divide before backward, or the effective learning rate is multiplied by
            # accum_steps. Easy to forget, and it looks like a badly tuned LR.
            (loss / tcfg.accum_steps).backward()
            accumulated += loss.item() / tcfg.accum_steps
            tokens_seen += x.numel()

        # A single bad batch can otherwise produce an enormous gradient and destroy the weights
        # in one step. Costs nothing; always on.
        if tcfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)

        optimizer.step()
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        history["step"].append(step)
        history["train_loss"].append(accumulated)
        history["lr"].append(lr)

        if verbose and tcfg.log_every and (step % tcfg.log_every == 0 or step == tcfg.max_steps - 1):
            elapsed = time.perf_counter() - start
            print(
                f"  step {step:>5}  loss {accumulated:7.4f}  ppl {math.exp(min(accumulated, 20)):>9,.1f}"
                f"  lr {lr:.2e}  {tokens_seen / max(elapsed, 1e-9):>8,.0f} tok/s"
            )

        is_last = step == tcfg.max_steps - 1
        if val_loader is not None and tcfg.eval_every and (step % tcfg.eval_every == 0 or is_last):
            stats = evaluate(model, val_loader, tcfg.eval_batches, device)
            history["val_step"].append(step)
            history["val_loss"].append(stats["loss"])
            if verbose:
                print(f"    val loss {stats['loss']:.4f}  perplexity {stats['perplexity']:,.1f}")

        if verbose and tok and tcfg.sample_every and (step % tcfg.sample_every == 0 or is_last):
            text = _sample(model, tok, tcfg.sample_prompt, tcfg.sample_tokens, device)
            print(f'    sample: "{text}"')

        if tcfg.save_every and step > 0 and step % tcfg.save_every == 0:
            checkpoint.save(
                Path(tcfg.out_dir) / "latest.pt", model, optimizer, scheduler, step, model.cfg
            )

    history["elapsed"] = time.perf_counter() - start
    history["tokens_seen"] = tokens_seen
    return history


def plot_history(history: dict, path: str | Path) -> Path | None:
    """Loss curve. Returns None if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["step"], history["train_loss"], label="train", lw=1, alpha=0.8)
    if history["val_step"]:
        ax.plot(history["val_step"], history["val_loss"], label="validation", lw=2, marker="o", ms=3)
    ax.axhline(math.log(50257), ls="--", c="gray", lw=1, label="ln(vocab) = 10.82")
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


def main() -> None:
    p = argparse.ArgumentParser(description="Train a GPT.")
    p.add_argument("--preset", default="small", choices=list(PRESETS))
    p.add_argument("--data", default="data/raw/the-verdict.txt")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--context", type=int, default=None, help="override context_length")
    p.add_argument("--layers", type=int, default=None, help="override n_layers")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--sample-every", type=int, default=100)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--overfit", action="store_true",
                   help="correctness run: no weight decay or LR floor, so loss can reach ~0")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    cfg: ModelConfig = PRESETS[args.preset]
    if args.context:
        cfg = replace(cfg, context_length=args.context)
    if args.layers:
        cfg = replace(cfg, n_layers=args.layers)

    tok = BPETokenizer()
    train_ids, val_ids = load_text_corpus(args.data, tok)
    stride = cfg.context_length // 4  # overlap, so a small corpus yields enough windows
    train_loader = make_dataloader(train_ids, cfg.context_length, stride, args.batch_size)
    val_loader = make_dataloader(
        val_ids, cfg.context_length, stride, args.batch_size, shuffle=False, drop_last=False
    )

    tcfg = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        accum_steps=args.accum,
        max_steps=args.steps,
        warmup_steps=args.warmup,
        eval_every=args.eval_every,
        sample_every=args.sample_every,
        out_dir=Path(args.out),
        **({"weight_decay": 0.0, "min_lr_ratio": 0.0} if args.overfit else {}),
    )

    device = torch.device("cpu") if args.cpu else pick_device()
    model = GPTModel(cfg)

    print(f"device       {device}  ({torch.cuda.get_device_name(0) if device.type == 'cuda' else ''})")
    print(f"preset       {args.preset}  {model.num_parameters():,} parameters")
    print(f"corpus       {args.data}  {len(train_ids):,} train / {len(val_ids):,} val tokens")
    print(f"windows      {len(train_loader.dataset):,} train (context {cfg.context_length}, stride {stride})")
    print(f"schedule     {args.steps} steps, warmup {args.warmup}, lr {args.lr}")
    if args.overfit:
        print("mode         overfit - weight decay and LR floor disabled so loss can reach ~0")
    print()

    history = train(model, train_loader, val_loader, tcfg, tok, device)

    out = Path(args.out)
    checkpoint.save(out / "final.pt", model, step=tcfg.max_steps, cfg=cfg)
    curve = plot_history(history, out / "loss.png")

    print(f"\nfinished in {history['elapsed']:.1f}s, {history['tokens_seen']:,} tokens seen")
    print(f"train loss {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")
    if history["val_loss"]:
        print(f"val   loss {history['val_loss'][0]:.4f} -> {history['val_loss'][-1]:.4f}")
    print(f"saved {out / 'final.pt'}" + (f" and {curve}" if curve else ""))


if __name__ == "__main__":
    main()
