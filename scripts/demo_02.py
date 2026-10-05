"""Note 2 demo - what attention actually computes.

    python scripts/demo_02.py
"""

from __future__ import annotations

import math
import time
from dataclasses import replace

import torch

from llm.attention import CausalSelfAttention, MultiHeadWrapper
from llm.config import SMALL
from llm.console import use_utf8
from llm.tokenizer import BPETokenizer

SENTENCE = "The animal did not cross the street because it was too tired"


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "-" * 76)


def show_matrix(weights: torch.Tensor, labels: list[str], places: int = 2) -> None:
    """Print a (T, T) attention matrix with tokens labelling both axes."""
    width = max(len(t) for t in labels) + 1
    print(" " * (width + 1) + "".join(f"{t[:5]:>6}" for t in labels))
    for i, row in enumerate(weights):
        cells = "".join(f"{v:>6.{places}f}" if v > 0 else "     ." for v in row.tolist())
        print(f"{labels[i]:>{width}} {cells}")


def main() -> None:
    use_utf8()
    torch.manual_seed(0)
    tok = BPETokenizer()
    ids = tok.encode(SENTENCE)
    labels = [tok.decode([i]).strip() for i in ids]
    T = len(ids)
    cfg = replace(SMALL, context_length=64, attn="manual", dropout=0.0)

    # 1 -------------------------------------------------------------------------------------
    rule("1. The causal mask: every row is a prefix of the sentence")
    x = torch.randn(1, T, cfg.emb_dim) * 0.3
    attn = CausalSelfAttention(cfg).eval()
    w = attn.attention_weights(x)[0, 0]
    show_matrix(w, labels)
    print("\nThe upper triangle is exactly zero: no position can see a later one. Weights are")
    print("random here because the model is untrained -- only the structure is meaningful.")

    # 2 -------------------------------------------------------------------------------------
    rule("2. Rows sum to 1, because -inf is applied before the softmax")
    print(f"  row sums          {w.sum(-1).tolist()[:6]} ...")
    print(f"  max weight above diagonal   {w.triu(1).abs().max().item():.1e}")
    print("\nMasking after the softmax instead would leave rows summing to less than 1, which")
    print("shrinks the output toward zero -- worst at early positions, and silent.")

    # 3 -------------------------------------------------------------------------------------
    rule("3. Causality, demonstrated: change the last token, the first output is untouched")
    x2 = x.clone()
    x2[:, -1] = torch.randn(1, cfg.emb_dim) * 0.3
    with torch.no_grad():
        before, after = attn(x), attn(x2)
    for pos in (0, T // 2, T - 1):
        delta = (before[0, pos] - after[0, pos]).abs().max().item()
        seen = "unchanged" if delta == 0 else f"changed by {delta:.4f}"
        print(f"  position {pos:>2} ({labels[pos]:<8}) {seen}")
    print("\nOnly the perturbed position and nothing before it. If early positions moved, the")
    print("model would be reading the future -- great training loss, useless generation.")

    # 4 -------------------------------------------------------------------------------------
    rule("4. Why the sqrt(head_dim) divisor is not cosmetic")
    q, k = torch.randn(1, 1, 32, 256), torch.randn(1, 1, 32, 256)
    raw = q @ k.transpose(-2, -1)

    def stats(scores: torch.Tensor) -> tuple[float, float, float]:
        p = torch.softmax(scores, dim=-1)
        ent = float(-(p * torch.log(p + 1e-12)).sum(-1).mean())
        return float(scores.abs().max()), float(p.max()), ent

    print(f"  {'':<10} {'|score| max':>12} {'max weight':>12} {'entropy':>10}")
    for name, s in [("unscaled", raw), ("scaled", raw / math.sqrt(256))]:
        score_max, weight_max, entropy = stats(s)
        print(f"  {name:<10} {score_max:>12.1f} {weight_max:>12.4f} {entropy:>10.3f}")
    print(f"  {'uniform':<10} {'-':>12} {1 / 32:>12.4f} {math.log(32):>10.3f}")
    print("\nUnscaled, one weight takes nearly all the mass. Softmax's gradient vanishes there,")
    print("so attention freezes at whatever pattern it started with and never learns.")

    # 5 -------------------------------------------------------------------------------------
    rule("5. Heads learn different patterns")
    full = attn.attention_weights(x)[0]
    print(f"  {full.shape[0]} heads, each a {full.shape[1]}x{full.shape[2]} matrix\n")
    for h in range(min(4, full.shape[0])):
        row = full[h, -1]
        top = torch.topk(row, 3)
        picks = ", ".join(f"{labels[i]}={v:.2f}" for v, i in zip(top.values, top.indices, strict=True))
        print(f"  head {h}: last token attends most to  {picks}")
    print("\nUntrained, so these are random -- but after training they specialize, one head on")
    print("grammatical agreement, another on coreference. One head would have to average them.")

    # 6 -------------------------------------------------------------------------------------
    rule("6. Batched heads vs a loop over heads: same result, different speed")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    big = replace(cfg, context_length=512)
    xb = torch.randn(8, 512, big.emb_dim, device=device)
    fused = CausalSelfAttention(big).to(device).eval()
    loop = (
        MultiHeadWrapper(
            big.emb_dim, big.emb_dim, big.context_length, big.n_heads, qkv_bias=big.qkv_bias
        )
        .to(device)
        .eval()
    )
    sdpa = CausalSelfAttention(replace(big, attn="sdpa")).to(device).eval()
    sdpa.load_state_dict(fused.state_dict())

    def timed(fn) -> float:
        """Wall time per call. CUDA launches are async, so synchronize or you time nothing."""
        with torch.no_grad():
            for _ in range(3):  # warm up: first call pays for kernel selection
                fn()
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                fn()
            if device == "cuda":
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) / 20

    def peak_mb(fn) -> float:
        if device != "cuda":
            return float("nan")
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            fn()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024**2

    print(f"  device: {device}, input {tuple(xb.shape)}\n")
    print(f"  {'':<22} {'time':>10} {'peak VRAM':>12}")
    for name, fn in [
        ("loop over heads", lambda: loop(xb)),
        ("batched (manual)", lambda: fused(xb)),
        ("batched (SDPA/Flash)", lambda: sdpa(xb)),
    ]:
        print(f"  {name:<22} {timed(fn) * 1e3:>7.2f} ms {peak_mb(fn):>9.0f} MB")
    with torch.no_grad():
        print(f"\n  manual and SDPA agree: {torch.allclose(fused(xb), sdpa(xb), atol=1e-4)}")
    print("\n  SDPA never materializes the (B,H,T,T) score tensor, so it is both faster and")
    print("  lighter than the batched manual path. The loop looks memory-cheap only because it")
    print("  builds one head's scores at a time and frees each before the next -- it pays for")
    print("  that in kernel launches, and it scales badly with head count.")

    # 7 -------------------------------------------------------------------------------------
    rule("7. The cost: scores are quadratic in sequence length")
    print(f"  {'T':>6} {'scores (B=8,H=12,fp32)':>26}")
    for t in (256, 512, 1024, 2048):
        mb = 8 * 12 * t * t * 4 / 1024**2
        print(f"  {t:>6} {mb:>22.0f} MB")
    print("\nPer layer, times 12 layers, plus activations kept for backward. This is why long")
    print("context is expensive, why FlashAttention exists, and why SMALL uses T=512.")


if __name__ == "__main__":
    main()
