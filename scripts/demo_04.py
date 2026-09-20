"""Note 4 demo - the assembled model, and the moment it starts working.

    python scripts/demo_04.py

Downloads GPT-2 124M from HuggingFace (~548 MB) the first time, then caches it.
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch

from llm.config import GPT2_124M, PRESETS, SMALL
from llm.console import use_utf8
from llm.gpt2_weights import gpt2_config, load_pretrained_gpt2
from llm.model import GPTModel
from llm.tokenizer import BPETokenizer

PROMPTS = [
    "The capital of France is",
    "Every effort moves you",
    "In the beginning, the universe was",
]


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "-" * 76)


def main() -> None:
    use_utf8()
    torch.manual_seed(0)
    tok = BPETokenizer()

    # 1 -------------------------------------------------------------------------------------
    rule("1. Where the parameters live")
    model = GPTModel(GPT2_124M)
    breakdown = model.parameter_breakdown()
    total = model.num_parameters()
    for name, n in breakdown.items():
        print(f"  {name:<22} {n:>12,}  {n / total:>6.1%}  {'#' * int(38 * n / total)}")
    print(f"  {'total (tied)':<22} {total:>12,}")

    untied = GPTModel(replace(GPT2_124M, tie_weights=False)).num_parameters()
    print(f"\n  untied would be {untied:,}; tying saves {untied - total:,} ({1 - total / untied:.0%}).")
    print("  That is the whole reason a 163M-parameter model is called 'GPT-2 124M'.")
    print("  The embedding table is 31% of the model and costs zero FLOPs - it is a lookup.")

    # 2 -------------------------------------------------------------------------------------
    rule("2. Preset shapes")
    print(f"  {'preset':<19} {'params':>13} {'non-emb':>12} {'emb %':>6} {'d/L':>5}")
    # Counted from the config rather than by building each model -- a test pins n_params() to
    # num_parameters() for every preset, so the arithmetic is known to agree.
    for name, cfg in PRESETS.items():
        emb = cfg.vocab_size * cfg.emb_dim
        total = cfg.n_params()
        note = "  <- v2 architecture" if cfg.pos == "rope" else ""
        print(
            f"  {name:<19} {total:>13,} {total - emb:>12,} "
            f"{emb / total:>6.0%} {cfg.emb_dim / cfg.n_layers:>5.0f}{note}"
        )
    print("\n  d/L is the aspect ratio; production models cluster in 100-200 (CS336 lecture 3).")

    # 3 -------------------------------------------------------------------------------------
    rule("3. Loss at random init should be ln(vocab_size)")
    cfg = replace(SMALL, n_layers=2)
    m = GPTModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (4, 32))
    y = torch.randint(0, cfg.vocab_size, (4, 32))
    with torch.no_grad():
        _, loss_indep = m(x, targets=y)
        _, loss_self = m(x, targets=x)
        p_self = m(x).softmax(-1).gather(-1, x.unsqueeze(-1)).mean()

    print(f"  ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.4f}   <- pure ignorance")
    print(f"  loss, independent targets  {loss_indep:.4f}   <- matches")
    print(f"  loss, targets == inputs    {loss_self:.4f}   <- lower, and not a bug")
    print(f"\n  P(input token) = {p_self:.6f} vs uniform {1 / cfg.vocab_size:.6f}")
    print("  Weight tying carries the token embedding through the residual stream into a head")
    print("  that IS the token embedding, so an untrained model already leans toward repeating")
    print("  the current token. Check init loss against targets independent of the inputs.")

    # 4 -------------------------------------------------------------------------------------
    rule("4. Untrained model: the architecture is done, the knowledge is not")
    untrained = GPTModel(gpt2_config("124M")).eval()
    ids = torch.tensor([tok.encode(PROMPTS[0])])
    out = untrained.generate_greedy(ids, max_new_tokens=12)
    print(f"  {PROMPTS[0]!r}\n    -> {tok.decode(out[0].tolist())!r}")
    print("\n  Noise, and that is the correct result. Every layer works; nothing has been learned.")

    # 5 -------------------------------------------------------------------------------------
    rule("5. The oracle: real GPT-2 weights, our code")
    print("  Downloading GPT-2 124M from HuggingFace if not already cached (~548 MB)...\n")
    trained = load_pretrained_gpt2("124M")
    for prompt in PROMPTS:
        ids = torch.tensor([tok.encode(prompt)])
        out = trained.generate_greedy(ids, max_new_tokens=14)
        print(f"  {prompt!r}\n    -> {tok.decode(out[0].tolist())!r}\n")
    print("  Fluent English from code we wrote. A single wrong transpose in notes 2-4 would")
    print("  produce noise here. The repetition is greedy decoding, fixed in note 6.")

    # 6 -------------------------------------------------------------------------------------
    rule("6. The quantitative version of the same claim")
    text = (
        "The quick brown fox jumps over the lazy dog. Machine learning is a field of study "
        "in artificial intelligence concerned with the development of statistical algorithms."
    )
    ids = torch.tensor([tok.encode(text)])
    x, y = ids[:, :-1], ids[:, 1:]
    with torch.no_grad():
        _, l_trained = trained(x, targets=y)
        _, l_random = untrained(x, targets=y)

    print(f"  {'':<22} {'loss':>8} {'perplexity':>12}")
    print(f"  {'GPT-2 weights':<22} {l_trained:>8.3f} {math.exp(l_trained):>12.1f}")
    print(f"  {'random init':<22} {l_random:>8.3f} {math.exp(l_random):>12,.0f}")
    print(
        f"\n  Perplexity {math.exp(l_trained):.0f} means the model is about as uncertain as "
        f"picking uniformly\n  among {math.exp(l_trained):.0f} tokens. At random init it is "
        f"picking among {math.exp(l_random):,.0f} - the whole vocabulary."
    )

    # 7 -------------------------------------------------------------------------------------
    rule("7. ' Paris' - a sharper check than reading the output")
    paris = tok.encode(" Paris")[0]
    for prompt in ("The capital of France is", "The capital of France is the city of"):
        ids = torch.tensor([tok.encode(prompt)])
        with torch.no_grad():
            probs = trained(ids)[0, -1].softmax(-1)
        top = torch.topk(probs, 5)
        rank = int((probs > probs[paris]).sum()) + 1
        print(f"  {prompt!r}")
        print("    top-5: " + ", ".join(
            f"{tok.decode([i])!r} {v:.3f}" for v, i in zip(top.values, top.indices)))
        print(f"    ' Paris': p={probs[paris]:.4f}, rank {rank}\n")
    print("  Not the argmax - GPT-2 124M is small. But top-5, and 2nd once the sentence forces")
    print("  a city. A misplaced weight would put it nowhere near.")


if __name__ == "__main__":
    main()
