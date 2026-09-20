"""Supervised fine-tuning: teach a base model to answer instead of continue.

    python -m llm.sft --data data/raw/alpaca_data.json

Starts from GPT-2's released weights -- a model that has read a broad slice of the web and so has
something to say -- and trains it on instruction/response pairs until it answers. Nothing about
the architecture changes; only the weights, and only nudged, at a learning rate fifty times lower
than pretraining used. Fine-tuning is a small correction to a model that already works, and too
high a rate erases what it knew.

See docs/ROADMAP.md, v3.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from llm import checkpoint
from llm.console import use_utf8
from llm.gpt2_weights import gpt2_config, load_pretrained_gpt2
from llm.instruct import (
    InstructionDataset,
    build_prompt,
    collate,
    length_grouped_batches,
    split,
)
from llm.model import GPTModel
from llm.train import TrainConfig, pick_device, plot_history, read_metrics, train


def build_loaders(
    records: list[dict], tok, batch_size: int, max_len: int, val_size: int
) -> tuple[DataLoader, DataLoader]:
    train_records, val_records = split(records, val_size)
    pad = partial(collate, pad_id=tok.eot_id)
    train_set = InstructionDataset(train_records, tok, max_len)
    val_set = InstructionDataset(val_records, tok, max_len)
    return (
        DataLoader(
            train_set,
            batch_sampler=length_grouped_batches(train_set, batch_size),
            collate_fn=pad,
        ),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=pad),
    )


def main() -> None:
    use_utf8()
    p = argparse.ArgumentParser(description="Fine-tune a base model to follow instructions.")
    p.add_argument("--data", default="data/raw/alpaca_data.json")
    p.add_argument("--from-checkpoint", default=None,
                   help="a checkpoint to start from; default is GPT-2 124M's released weights")
    p.add_argument("--tokenizer", default=None, help="only with --from-checkpoint")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--val-size", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--out", default="checkpoints/gpt2-instruct")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu") if args.cpu else pick_device()
    torch.manual_seed(args.seed)

    if args.from_checkpoint:
        from llm.bpe import Tokenizer

        cfg = checkpoint.load_config(args.from_checkpoint)
        model = GPTModel(cfg)
        checkpoint.load(args.from_checkpoint, model, map_location=device)
        tok = Tokenizer.load(args.tokenizer) if args.tokenizer else None
        if tok is None:
            raise SystemExit("--from-checkpoint needs the matching --tokenizer")
    else:
        from llm.tokenizer import BPETokenizer

        cfg = gpt2_config("124M")
        model = load_pretrained_gpt2("124M")
        tok = BPETokenizer()

    records = json.loads(Path(args.data).read_text(encoding="utf-8"))
    train_loader, val_loader = build_loaders(
        records, tok, args.batch_size, args.max_len, args.val_size
    )
    steps_per_epoch = len(train_loader) // args.accum
    max_steps = max(1, int(steps_per_epoch * args.epochs))

    out = Path(args.out)
    metrics_path = out / "metrics.jsonl"
    if not args.resume and metrics_path.exists():
        metrics_path.unlink()

    tcfg = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        accum_steps=args.accum,
        max_steps=max_steps,
        # Short warmup: the weights are already good, so the only risk is the first few steps
        # being large enough to damage them.
        warmup_steps=min(100, max_steps // 20),
        min_lr_ratio=0.1,
        weight_decay=0.0,
        eval_every=args.eval_every,
        eval_batches=20,
        sample_every=max(250, max_steps // 8),
        sample_prompt=build_prompt("Give three tips for staying healthy."),
        sample_tokens=60,
        save_every=args.save_every,
        out_dir=out,
        seed=args.seed,
    )

    print(f"device       {device}")
    print(f"model        {model.num_parameters():,} parameters "
          f"({'GPT-2 124M released weights' if not args.from_checkpoint else args.from_checkpoint})")
    print(f"data         {len(train_loader.dataset):,} train / {len(val_loader.dataset):,} val pairs")
    print(f"schedule     {args.epochs:g} epochs = {max_steps:,} steps of "
          f"{args.batch_size * args.accum} examples, lr {args.lr:g}")
    print(f"checkpoints  every {args.save_every} steps -> {out / 'latest.pt'}\n")

    history = train(
        model, train_loader, val_loader, tcfg, tok, device,
        resume_from=out / "latest.pt" if args.resume else None,
        metrics_path=metrics_path,
    )

    checkpoint.save(out / "final.pt", model, step=max_steps, cfg=cfg,
                    tokens_seen=history["tokens_seen"], instruct=True)
    curve = plot_history(read_metrics(metrics_path), out / "loss.png", cfg.vocab_size)
    print(f"\nfinished in {history['elapsed'] / 60:.1f} min")
    if history["val_loss"]:
        print(f"val loss {history['val_loss'][0]:.4f} -> {history['val_loss'][-1]:.4f}")
    print(f"saved {out / 'final.pt'}" + (f" and {curve}" if curve else ""))
    print(f"\ntalk to it:\n  python scripts/chat.py --checkpoint {out / 'final.pt'} "
          f"--tokenizer gpt2 --instruct")


if __name__ == "__main__":
    main()
