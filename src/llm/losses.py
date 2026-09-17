"""Cross-entropy, and evaluation over a loader.

The loss is the negative log of the probability the model assigned to the token that actually
came next. Reading the number directly:

    10.82  uniform ignorance over 50257 tokens - the value at random init
     6.9   knows which tokens are common
     4.6   grammar, roughly
     3.0   coherent local structure
     0.0   memorized

Perplexity = exp(loss), and is the friendlier unit: "as uncertain as choosing uniformly among
this many tokens".

See notes/05_training.md.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader


def compute_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean cross-entropy over every position in the batch.

    F.cross_entropy takes **raw logits, not probabilities**: it fuses log-softmax with the
    negative log-likelihood for numerical stability. Applying softmax first and then calling it
    is a real and common bug - it trains, badly, and nothing errors.
    """
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    max_batches: int | None = None,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Mean loss and perplexity over the loader, in eval mode.

    Restores whatever mode the model was in, so calling this mid-training cannot silently leave
    dropout disabled for the rest of the run.
    """
    was_training = model.training
    model.eval()
    device = device or next(model.parameters()).device

    total, batches = 0.0, 0
    try:
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            _, loss = model(x.to(device), targets=y.to(device))
            total += loss.item()
            batches += 1
    finally:
        model.train(was_training)

    if batches == 0:
        return {"loss": float("nan"), "perplexity": float("nan"), "batches": 0}
    mean = total / batches
    return {
        "loss": mean,
        # exp overflows to inf above ~709; an untrained model is nowhere near, but a diverged
        # run can be, and inf in a log line is less useful than a large number.
        "perplexity": math.exp(mean) if mean < 700 else float("inf"),
        "batches": batches,
    }


if __name__ == "__main__":
    import numpy as np

    from llm.config import SMALL
    from llm.dataset import make_dataloader
    from llm.model import GPTModel

    torch.manual_seed(0)
    cfg = SMALL
    model = GPTModel(cfg)
    ids = np.random.default_rng(0).integers(0, cfg.vocab_size, 20_000, dtype=np.uint16)
    loader = make_dataloader(ids, cfg.context_length, cfg.context_length, batch_size=4)

    stats = evaluate(model, loader, max_batches=5)
    print(f"untrained loss {stats['loss']:.4f}  perplexity {stats['perplexity']:,.0f}")
    print(f"ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.4f}")
    print(f"\nmodel still in train mode after evaluate(): {model.training}")
