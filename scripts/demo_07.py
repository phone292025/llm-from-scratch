"""Note 7 demo - the v1 vs v2 A/B, read off the two runs' metrics.

Two runs, identical in every respect but the architecture: same corpus, same tokenizer, same
40,000 steps of 8,192 tokens, same learning-rate schedule, same seed. One is `tinystories`
(LayerNorm, learned positions, GELU, full MHA), the other `modern-tinystories` (RMSNorm, RoPE,
SwiGLU, 4 KV heads).

    python scripts/demo_07.py
    python scripts/demo_07.py --runs checkpoints/tinystories-full checkpoints/modern-full

Reads `metrics.jsonl` from each run, so it works while training is still going and simply
compares over however many steps both runs have reached.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from llm.config import PRESETS
from llm.console import use_utf8
from llm.train import read_metrics as train_read_metrics

DEFAULT_RUNS = ["checkpoints/tinystories-full", "checkpoints/modern-full"]


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "-" * 76)


def read_metrics(run: Path) -> dict[str, dict[int, float]]:
    """{"train": {step: loss}, "eval": {step: loss}}, keyed for lookup by step.

    Parsing is `train.read_metrics`, so the rule that a later record for a repeated step wins --
    which is what makes a resumed run's curve show the steps that produced the final weights --
    lives in exactly one place.
    """
    path = run / "metrics.jsonl"
    if not path.exists():
        return {"train": {}, "eval": {}}
    history = train_read_metrics(path)
    return {
        "train": dict(zip(history["step"], history["train_loss"], strict=True)),
        "eval": dict(zip(history["val_step"], history["val_loss"], strict=True)),
    }


def smooth(series: dict[int, float], step: int, window: int = 50) -> float | None:
    """Mean of the `window` logged points ending at `step`.

    Single-step training loss is far too noisy to compare: the gap between the two runs is
    tenths, and neighbouring steps differ by as much. Averaging is not cosmetic here, it is what
    makes the comparison mean anything.
    """
    points = [loss for s, loss in series.items() if step - window < s <= step]
    return sum(points) / len(points) if points else None


def preset_of(run: Path) -> str | None:
    """Recover which preset a run used, from the header its log writes."""
    log = run / "train.log"
    if not log.exists():
        return None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[:10]:
        if line.startswith("preset"):
            return line.split()[1]
    return None


def plot(runs: list[Path], data: list[dict], out: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run, metrics in zip(runs, data, strict=True):
        evals = sorted(metrics["eval"])
        if evals:
            ax.plot(evals, [metrics["eval"][s] for s in evals], lw=2, marker="o", ms=3,
                    label=f"{run.name} (val)")
        trains = sorted(metrics["train"])
        if trains:
            averaged = [(s, smooth(metrics["train"], s)) for s in trains[::50]]
            ax.plot([s for s, _ in averaged], [v for _, v in averaged], lw=1, alpha=0.4,
                    label=f"{run.name} (train, smoothed)")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("v1 vs v2 on identical data")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    use_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=2, default=DEFAULT_RUNS, metavar=("BASELINE", "MODERN"))
    parser.add_argument("--plot", default="checkpoints/v1_vs_v2.png")
    args = parser.parse_args()

    runs = [Path(r) for r in args.runs]
    data = [read_metrics(r) for r in runs]
    missing = [r for r, d in zip(runs, data, strict=True) if not d["train"]]
    if missing:
        raise SystemExit(f"no metrics.jsonl in {', '.join(str(m) for m in missing)}")

    base_name, mod_name = (preset_of(r) for r in runs)

    rule("1. What is being compared")
    print(f"  {'':<22} {'baseline':>20} {'modern':>20}")
    rows = [("preset", base_name, mod_name)]
    if base_name in PRESETS and mod_name in PRESETS:
        b, m = PRESETS[base_name], PRESETS[mod_name]
        rows += [
            ("parameters", f"{b.n_params():,}", f"{m.n_params():,}"),
            ("norm", b.norm, m.norm),
            ("positions", b.pos, m.pos),
            ("activation", b.act, m.act),
            ("kv heads", str(b.kv_heads), str(m.kv_heads)),
        ]
    for label, left, right in rows:
        print(f"  {label:<22} {str(left):>20} {str(right):>20}")
    if base_name in PRESETS and mod_name in PRESETS:
        delta = PRESETS[mod_name].n_params() / PRESETS[base_name].n_params() - 1
        print(f"\n  modern is {abs(delta):.1%} {'smaller' if delta < 0 else 'larger'} -- almost "
              f"all of it GQA. See note 7 section 0.")

    rule("2. Smoothed training loss at matched steps")
    print("  50-step means; single steps are far noisier than the gap being measured.")
    print(f"\n  {'step':>7} {'baseline':>10} {'modern':>10} {'delta':>9}")
    shared = sorted(set(data[0]["train"]) & set(data[1]["train"]))
    if not shared:
        print("  (no overlapping steps yet)")
    else:
        # About ten evenly spaced rows, on a round stride, whatever the run has reached so far.
        stride = max(50, round(shared[-1] / 10 / 50) * 50)
        marks = [s for s in range(stride, shared[-1] + 1, stride)]
        for step in marks + ([shared[-1]] if shared[-1] not in marks else []):
            a, b = smooth(data[0]["train"], step), smooth(data[1]["train"], step)
            if a is not None and b is not None:
                print(f"  {step:>7,} {a:>10.4f} {b:>10.4f} {b - a:>+9.4f}")

    rule("3. Validation loss")
    print(f"  {'step':>7} {'baseline':>10} {'modern':>10} {'delta':>9}   {'perplexity':>20}")
    for step in sorted(set(data[0]["eval"]) & set(data[1]["eval"])):
        a, b = data[0]["eval"][step], data[1]["eval"][step]
        ppl = f"{math.exp(a):>8.2f} -> {math.exp(b):<8.2f}"
        print(f"  {step:>7,} {a:>10.4f} {b:>10.4f} {b - a:>+9.4f}   {ppl:>20}")

    finals = [json.loads((r / "final_eval.json").read_text()) if (r / "final_eval.json").exists()
              else None for r in runs]
    if all(finals):
        a, b = finals[0]["loss"], finals[1]["loss"]
        print(f"\n  full validation set: {a:.4f} -> {b:.4f}  ({b - a:+.4f}, "
              f"perplexity {math.exp(a):.2f} -> {math.exp(b):.2f})")
    else:
        print("\n  (--final-eval has not run yet on at least one side)")

    written = plot(runs, data, Path(args.plot))
    rule("4. Plot")
    print(f"  {written}" if written else "  matplotlib not installed; skipped")


if __name__ == "__main__":
    main()
