"""Saving and resuming a training run.

Model weights alone are not resumable. Restarting without Adam's moment estimates causes a
visible loss spike while they rebuild from nothing, and without the scheduler's step count the
learning rate restarts at warmup.

See notes/05_training.md section 6.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
from torch import nn, optim

from llm.config import ModelConfig


def save(
    path: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
    step: int = 0,
    cfg: ModelConfig | None = None,
    **extra: object,
) -> Path:
    """Write everything needed to continue exactly where the run stopped."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "model": model.state_dict(),
        "step": step,
        # The architecture travels with the weights, so a checkpoint can rebuild its own model.
        "cfg": dataclasses.asdict(cfg) if cfg is not None else None,
        **extra,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)
    return path


def load(
    path: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> int:
    """Restore into the objects given, and return the step the run had reached."""
    # weights_only=True refuses to unpickle arbitrary objects. Our payload is tensors and plain
    # data, so it costs nothing and closes the door on loading a hostile checkpoint.
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=True)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("step", 0))


def load_config(path: str | Path) -> ModelConfig:
    """Read just the architecture, so a checkpoint can build the model it belongs to."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not ckpt.get("cfg"):
        raise ValueError(f"{path} has no config stored")
    return ModelConfig(**ckpt["cfg"])


if __name__ == "__main__":
    import tempfile
    from dataclasses import replace

    from llm.config import SMALL
    from llm.model import GPTModel
    from llm.optim import make_optimizer, make_schedule

    cfg = replace(SMALL, n_layers=2, emb_dim=64, n_heads=4, vocab_size=512, context_length=32)
    torch.manual_seed(0)
    model = GPTModel(cfg)
    opt = make_optimizer(model, lr=1e-3)
    sched = make_schedule(opt, warmup_steps=2, total_steps=20)

    x = torch.randint(0, cfg.vocab_size, (2, 32))
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, targets=x)
        loss.backward()
        opt.step()
        sched.step()

    with tempfile.TemporaryDirectory() as tmp:
        path = save(Path(tmp) / "demo.pt", model, opt, sched, step=5, cfg=cfg)
        size = path.stat().st_size / 1024**2
        print(f"saved {size:.1f} MB: weights + optimizer state + scheduler + config")

        model2 = GPTModel(cfg)
        opt2 = make_optimizer(model2, lr=1e-3)
        sched2 = make_schedule(opt2, warmup_steps=2, total_steps=20)
        step = load(path, model2, opt2, sched2)

        model.eval(), model2.eval()
        with torch.no_grad():
            same = torch.equal(model(x), model2(x))
        print(f"resumed at step {step}")
        print(f"  logits identical:      {same}")
        print(f"  learning rate restored: {sched2.get_last_lr() == sched.get_last_lr()}")
        print(f"  config travels with it: {load_config(path) == cfg}")

    print("\nOptimizer state is most of that file. Resuming without Adam's moment estimates")
    print("causes a visible loss spike while they rebuild from nothing.")
