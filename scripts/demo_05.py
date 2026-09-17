"""Note 5 demo - watch a model learn, live.

    python scripts/demo_05.py

Trains on the tiny story until it memorizes it. Roughly 25 seconds on a 5070.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import torch

from llm.config import SMALL
from llm.dataset import make_dataloader
from llm.losses import evaluate
from llm.model import GPTModel
from llm.optim import lr_at_step, make_optimizer
from llm.tokenizer import BPETokenizer
from llm.train import TrainConfig, load_text_corpus, pick_device, plot_history, train

CORPUS = Path("data/raw/the-verdict.txt")
PROMPT = "Every effort moves you"


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "-" * 76)


def main() -> None:
    torch.manual_seed(0)
    device = pick_device()
    tok = BPETokenizer()
    cfg = replace(SMALL, dropout=0.0)

    if not CORPUS.exists():
        raise SystemExit(f"missing {CORPUS} - see notes/05_training.md")

    # 1 -------------------------------------------------------------------------------------
    rule("1. Reading the loss")
    print(f"  {'loss':>6} {'perplexity':>12}   meaning")
    for loss, meaning in [
        (math.log(cfg.vocab_size), "uniform ignorance - the value at random init"),
        (6.9, "knows which tokens are common"),
        (4.6, "grammar, roughly"),
        (3.0, "coherent local structure"),
        (1.8, "CS336's TinyStories target"),
        (0.0, "memorized"),
    ]:
        print(f"  {loss:>6.2f} {math.exp(loss):>12,.1f}   {meaning}")
    print("\n  Perplexity = exp(loss): 'as uncertain as choosing uniformly among this many'.")

    # 2 -------------------------------------------------------------------------------------
    rule("2. The learning rate schedule")
    total, warmup = 400, 50
    for step in (0, 10, 25, 49, 100, 200, 300, 399):
        mult = lr_at_step(step, warmup_steps=warmup, total_steps=total, min_lr_ratio=0.0)
        print(f"  step {step:>4}  x{mult:.3f}  {'#' * int(mult * 44)}")
    print("\n  Warmup, because Adam's variance estimates are meaningless at step 0 and a full-size")
    print("  step can wreck the model before it starts. Then cosine decay: big steps to find a")
    print("  good region, small steps to settle into it.")

    # 3 -------------------------------------------------------------------------------------
    rule("3. Which parameters get weight decay")
    model = GPTModel(cfg)
    decay, plain = make_optimizer(model, lr=3e-4).param_groups
    print(f"  decayed  {sum(p.numel() for p in decay['params']):>12,}  matrices and embeddings")
    print(f"  plain    {sum(p.numel() for p in plain['params']):>12,}  biases and norm scales")
    print("\n  Decaying a norm's scale toward zero fights the normalization it exists to perform.")

    # 4 -------------------------------------------------------------------------------------
    rule("4. Before training")
    train_ids, val_ids = load_text_corpus(CORPUS, tok)
    stride = cfg.context_length // 4
    train_loader = make_dataloader(train_ids, cfg.context_length, stride, 8)
    val_loader = make_dataloader(val_ids, cfg.context_length, stride, 8, shuffle=False, drop_last=False)

    model = GPTModel(cfg).to(device)
    print(f"  corpus  {len(train_ids):,} train / {len(val_ids):,} val tokens")
    print(f"  model   {model.num_parameters():,} parameters on {device}")
    stats = evaluate(model, val_loader, device=device)
    print(f"  loss    {stats['loss']:.4f}  (ln vocab = {math.log(cfg.vocab_size):.4f})")
    ids = torch.tensor([tok.encode(PROMPT)], device=device)
    print(f'\n  sample: "{tok.decode(model.generate_greedy(ids, 20)[0].tolist())}"')
    print("\n  Noise. The architecture is complete; the knowledge is not.")

    # 5 -------------------------------------------------------------------------------------
    rule("5. Training - deliberately overfitting 4,630 tokens")
    print("  Weight decay and the LR floor are off, so loss can reach zero. Overfitting on")
    print("  purpose is the fastest possible proof that the whole pipeline works.\n")
    history = train(
        model,
        train_loader,
        val_loader,
        TrainConfig(
            lr=3e-4, max_steps=400, warmup_steps=50, weight_decay=0.0, min_lr_ratio=0.0,
            eval_every=100, sample_every=100, sample_prompt=PROMPT, log_every=50,
        ),
        tok=tok,
        device=device,
    )

    # 6 -------------------------------------------------------------------------------------
    rule("6. What the curves say")
    print(f"  {'':<8} {'first':>9} {'best':>9} {'last':>9}")
    tr, va = history["train_loss"], history["val_loss"]
    print(f"  {'train':<8} {tr[0]:>9.4f} {min(tr):>9.4f} {tr[-1]:>9.4f}")
    print(f"  {'val':<8} {va[0]:>9.4f} {min(va):>9.4f} {va[-1]:>9.4f}")
    best_at = history["val_step"][va.index(min(va))]
    print(f"\n  Validation bottomed out at step {best_at} and then climbed while training loss kept")
    print("  falling. That divergence is overfitting, and here it is the goal: with 4,630 tokens")
    print(f"  and {model.num_parameters():,} parameters the model can simply memorize the text.")
    print("\n  On a real corpus the same shape is a warning - it means stop, or get more data.")

    print(f"\n  {history['tokens_seen']:,} tokens in {history['elapsed']:.1f}s "
          f"({history['tokens_seen'] / history['elapsed']:,.0f} tok/s)")
    curve = plot_history(history, "checkpoints/loss.png")
    if curve:
        print(f"  loss curve saved to {curve}")

    # 7 -------------------------------------------------------------------------------------
    rule("7. After training")
    print(f'  sample: "{tok.decode(model.generate_greedy(ids, 30)[0].tolist())}"')
    print("\n  Real sentences - reproduced from the story, not composed. That is what a loss of")
    print("  ~0.00 on 4,630 tokens means. To get a model that writes rather than recites, it")
    print("  needs far more data than it has parameters. See docs/DATA_GUIDE.md.")


if __name__ == "__main__":
    main()
