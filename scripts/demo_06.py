"""Note 6 demo - sampling strategies, and what the KV cache buys.

One prompt, one seed, five decoding strategies side by side, then the cache timed against the
uncached loop. Uses real GPT-2 weights, because the difference between temperature 0.7 and 1.3 is
only legible on a model that writes English. Downloads 548 MB the first time.

    python scripts/demo_06.py
"""

from __future__ import annotations

import time

import torch

from llm.console import use_utf8
from llm.generate import generate, generate_text, top_k_filter, top_p_filter
from llm.gpt2_weights import load_pretrained_gpt2
from llm.kv_cache import KVCache
from llm.tokenizer import BPETokenizer

PROMPT = "The future of artificial intelligence"
SEED = 0


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "-" * 76)


def main() -> None:
    use_utf8()
    tok = BPETokenizer()
    model = load_pretrained_gpt2("124M")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    ids = torch.tensor([tok.encode(PROMPT)], device=device)

    # 1 -------------------------------------------------------------------------------------
    rule("1. The same model, five ways of reading its distribution")
    strategies = [
        ("greedy", dict(temperature=0.0)),
        ("temperature 0.7", dict(temperature=0.7)),
        ("temperature 1.3", dict(temperature=1.3)),
        ("top-k 50", dict(temperature=1.0, top_k=50)),
        ("top-p 0.9", dict(temperature=0.8, top_p=0.9)),
    ]
    for name, kwargs in strategies:
        text = generate_text(model, tok, PROMPT, 35, seed=SEED, **kwargs)
        print(f"\n  {name}")
        print(f"    {text[len(PROMPT):].strip()!r}")

    print("\n  Same weights, same prompt, same seed. Only the choice of how to sample changed.")
    print("  Greedy is the one that loops; 1.3 is the one that stops making sense.")

    # 2 -------------------------------------------------------------------------------------
    rule("2. What the filters actually remove")
    with torch.no_grad():
        logits = model(ids)[0, -1]
    probs = logits.softmax(-1)
    top = probs.topk(8)
    print("  the model's own top 8 continuations:")
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        bar = "#" * int(60 * p / top.values[0].item())
        print(f"    {tok.decode([i])!r:>14} {p:>6.1%}  {bar}")

    survivors = {
        "unfiltered": int((probs > 0).sum()),
        "top-k 50": int((top_k_filter(logits, 50) != float("-inf")).sum()),
        "top-p 0.9": int((top_p_filter(logits, 0.9) != float("-inf")).sum()),
        "top-p 0.5": int((top_p_filter(logits, 0.5) != float("-inf")).sum()),
    }
    print(f"\n  {'filter':<14} {'candidates left':>16}")
    for name, n in survivors.items():
        print(f"  {name:<14} {n:>16,}")
    print("\n  top-k always keeps 50. top-p keeps however many this particular distribution")
    print("  needs, which is the argument for it: the model's confidence varies token to token.")

    # 3 -------------------------------------------------------------------------------------
    rule("3. Cached and uncached decoding must agree exactly")
    a = generate(model, ids, 40, temperature=0.0, use_cache=False)
    b = generate(model, ids, 40, temperature=0.0, use_cache=True)
    print(f"  identical token ids: {torch.equal(a, b)}")
    print("\n  This is the load-bearing check of the whole note. A cache bug does not crash and")
    print("  does not read as broken - it changes which keys a query was allowed to see, and the")
    print("  model goes on writing fluent English that is quietly not what it would have said.")

    # 4 -------------------------------------------------------------------------------------
    rule("4. What the cache costs, and what it buys")
    n = 200
    timings = {}
    for use_cache in (False, True):
        generate(model, ids, 8, temperature=0.0, use_cache=use_cache)  # warm the kernels
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        generate(model, ids, n, temperature=0.0, use_cache=use_cache)
        if device == "cuda":
            torch.cuda.synchronize()
        timings[use_cache] = time.perf_counter() - start

    print(f"  {n} tokens on {device}")
    print(f"    without cache {timings[False]:>7.2f}s   ({n / timings[False]:>6.1f} tok/s)")
    print(f"    with cache    {timings[True]:>7.2f}s   ({n / timings[True]:>6.1f} tok/s)")
    print(f"    speedup       {timings[False] / timings[True]:>7.1f}x")

    megabytes = KVCache.predicted_bytes(model.cfg, 1, model.cfg.context_length) / 1024**2
    print(f"\n  The cost is memory: {megabytes:.0f} MB per sequence at the full "
          f"{model.cfg.context_length}-token context, bf16.")
    print("  Batch 32 makes that 32 times larger. For a big model at long context the cache")
    print("  outgrows the model itself, which is the entire reason GQA exists.")


if __name__ == "__main__":
    main()
