"""AdamW with two parameter groups, and a warmup-then-cosine learning rate schedule.

The learning rate is the single most consequential hyperparameter. Both pieces here exist to stop
a run destroying itself in the first hundred steps.

See notes/05_training.md section 3.
"""

from __future__ import annotations

import math

from torch import nn, optim


def make_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> optim.AdamW:
    """AdamW, decaying only the parameters where decay makes sense.

    Two groups, split on dimensionality:

    - **2D and up** (weight matrices, embeddings) get weight decay.
    - **1D** (biases, LayerNorm and RMSNorm scales) do not. Pulling a norm's scale toward zero is
      actively harmful - it fights the normalization the layer exists to perform.

    betas=(0.9, 0.95) rather than PyTorch's (0.9, 0.999): transformers prefer a shorter variance
    window, and 0.999 adapts too slowly to the loss landscape changing under it.

    Note on embeddings: this decays them, because with weight tying the embedding table *is* the
    output projection and behaves like a matrix. Some references exclude embeddings; the
    dimensionality rule is what nanoGPT and most current implementations use.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.ndim >= 2 else no_decay).append(param)

    return optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
        eps=eps,
    )


def lr_at_step(
    step: int, *, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1
) -> float:
    """The schedule as a plain multiplier on the peak learning rate.

    Linear warmup, then cosine decay to `min_lr_ratio` of peak, flat thereafter.

    **Warmup** matters because at initialization Adam's variance estimates are meaningless, so
    early full-size steps can be wildly wrong and wreck the model before it starts.

    **Cosine decay** gives large steps early to find a good region and small steps late to settle
    into it. Cosine specifically, rather than linear or stepped, is empirical - it is what
    essentially every modern model uses.
    """
    if warmup_steps > 0 and step < warmup_steps:
        # step + 1 so the first step is not a no-op at lr 0.
        return (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr_ratio
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def make_schedule(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> optim.lr_scheduler.LambdaLR:
    """Wrap `lr_at_step` as a PyTorch scheduler.

    Decay should terminate at exactly `total_steps`. Scheduling for more steps than you actually
    run leaves the model at a high learning rate when the run stops, and the last few hundred
    steps of decay are where a surprising amount of the final loss improvement happens.
    """
    return optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_at_step(
            step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        ),
    )


if __name__ == "__main__":
    from llm.config import SMALL
    from llm.model import GPTModel

    model = GPTModel(SMALL)
    opt = make_optimizer(model, lr=3e-4)
    decayed, plain = opt.param_groups
    print("parameter groups")
    print(f"  decayed  {sum(p.numel() for p in decayed['params']):>12,}  "
          f"({len(decayed['params'])} tensors, wd={decayed['weight_decay']})")
    print(f"  plain    {sum(p.numel() for p in plain['params']):>12,}  "
          f"({len(plain['params'])} tensors, wd={plain['weight_decay']})")
    print("  biases and norm scales are excluded: decaying a norm's scale toward zero fights")
    print("  the normalization the layer exists to do.")

    total, warmup = 1000, 100
    print(f"\nschedule (warmup {warmup}, total {total}, floor 10%)")
    for step in (0, 25, 50, 99, 100, 250, 500, 750, 999, 1000):
        mult = lr_at_step(step, warmup_steps=warmup, total_steps=total)
        bar = "#" * int(mult * 40)
        print(f"  step {step:>4}  lr x{mult:.3f}  {bar}")
