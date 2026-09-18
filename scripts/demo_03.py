"""Note 3 demo - normalization, activations, and what residual connections actually buy.

    python scripts/demo_03.py
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from llm.block import PostNormBlock, TransformerBlock
from llm.config import SMALL
from llm.console import use_utf8
from llm.ffn import GELU, make_ffn
from llm.norms import LayerNorm, RMSNorm

CFG = replace(SMALL, context_length=64, attn="manual", dropout=0.0)
D = CFG.emb_dim


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "-" * 76)


# --- gradient experiments ----------------------------------------------------------------------


class NoResidual(TransformerBlock):
    def forward(self, x, rope=None, cache=None):
        return self.ffn(self.norm2(self.attn(self.norm1(x))))


class NoResidualNoNorm(TransformerBlock):
    def forward(self, x, rope=None, cache=None):
        return self.ffn(self.attn(x))


def layer_grads(block_cls, depth: int = 12) -> list[float]:
    torch.manual_seed(0)
    stack = nn.ModuleList(block_cls(CFG) for _ in range(depth))
    h = torch.randn(1, 8, D)
    for blk in stack:
        h = blk(h)
    h.sum().backward()
    return [float(blk.ffn.down.weight.grad.abs().mean()) for blk in stack]


def mlp_grads(residual: bool, depth: int = 20, d: int = 64) -> list[float]:
    """A plain MLP: no normalization anywhere, so residuals are the only thing holding it up."""
    torch.manual_seed(0)
    layers = nn.ModuleList(nn.Linear(d, d) for _ in range(depth))
    for layer in layers:
        nn.init.normal_(layer.weight, std=0.05)
        nn.init.zeros_(layer.bias)
    h = torch.randn(1, d)
    for layer in layers:
        out = F.gelu(layer(h))
        h = h + out if residual else out
    h.sum().backward()
    return [float(layer.weight.grad.abs().mean()) for layer in layers]


def bar(value: float, lo: float = 1e-15, hi: float = 1e1) -> str:
    """Log-scale bar, so fourteen orders of magnitude fit on one line."""
    import math

    if value <= 0:
        return ""
    frac = (math.log10(value) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    return "#" * max(0, min(40, int(frac * 40)))


def main() -> None:
    use_utf8()
    torch.manual_seed(0)
    x = torch.randn(2, 8, D) * 5 + 3  # deliberately off-centre and wide

    # 1 -------------------------------------------------------------------------------------
    rule("1. What normalization does to activation scale")
    print(f"  {'':<24} {'mean':>10} {'std':>10}")
    print(f"  {'input':<24} {x.mean():>10.4f} {x.std():>10.4f}")
    print(f"  {'LayerNorm':<24} {LayerNorm(D)(x).mean():>10.4f} {LayerNorm(D)(x).std():>10.4f}")
    print(f"  {'RMSNorm':<24} {RMSNorm(D)(x).mean():>10.4f} {RMSNorm(D)(x).std():>10.4f}")
    print("\nRMSNorm does not re-centre - it only rescales. The surprise of the last few years")
    print("is that the re-centring LayerNorm does was never contributing anything.")
    print(f"\n  LayerNorm parameters: {2 * D:>6}")
    print(f"  RMSNorm parameters:   {D:>6}   <- half, and faster")

    # 2 -------------------------------------------------------------------------------------
    rule("2. Activation scale drifting across an unnormalized stack")
    torch.manual_seed(0)
    h_norm = h_plain = torch.randn(1, 8, D)
    layers = [nn.Linear(D, D) for _ in range(8)]
    norm = LayerNorm(D)
    print(f"  {'layer':>6} {'with norm (std)':>18} {'without norm (std)':>20}")
    with torch.no_grad():
        for i, layer in enumerate(layers):
            h_norm = norm(F.gelu(layer(h_norm)))
            h_plain = F.gelu(layer(h_plain))
            print(f"  {i:>6} {h_norm.std():>18.4f} {h_plain.std():>20.4e}")
    print("\nNormalized, every layer sees std 1.0. Unnormalized, the scale falls ~20x over eight")
    print("layers before settling. The direction depends on the initialization -- weights a")
    print("little larger and it climbs instead. Either way the model no longer controls the")
    print("scale its layers operate at, and gradients inherit the same drift.")

    # 3 -------------------------------------------------------------------------------------
    rule("3. GELU vs ReLU: the gate is soft, so the gradient never hits exactly zero")
    pts = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0])
    print(f"  {'x':>6} {'ReLU':>10} {'GELU':>10} {'dGELU/dx':>10}")
    for p in pts:
        t = p.clone().requires_grad_(True)
        GELU()(t).backward()
        print(f"  {p:>6.1f} {F.relu(p):>10.4f} {GELU()(p):>10.4f} {t.grad:>10.4f}")
    print("\nReLU returns exactly 0 with exactly 0 gradient for every negative input, so a unit")
    print("that goes negative can be stuck there permanently. GELU always leaves a way back.")

    # 4 -------------------------------------------------------------------------------------
    rule("4. Where a block's parameters live")
    block = TransformerBlock(CFG)
    parts = [("norm1", block.norm1), ("attn", block.attn), ("norm2", block.norm2),
             ("ffn", block.ffn)]
    total = sum(p.numel() for p in block.parameters())
    for name, mod in parts:
        n = sum(p.numel() for p in mod.parameters())
        print(f"  {name:<8} {n:>10,}  {n / total:>6.1%}  {'#' * int(40 * n / total)}")
    print(f"  {'total':<8} {total:>10,}")
    print("\nTwo thirds is the feed-forward network. Attention gets the attention; the MLP holds")
    print("most of the parameters, and probably most of the model's factual knowledge.")

    # 5 -------------------------------------------------------------------------------------
    rule("5. Gradient reaching layer 0 of a 12-block stack")
    variants = [
        ("pre-norm + residual", TransformerBlock),
        ("norm, no residual", NoResidual),
        ("neither", NoResidualNoNorm),
    ]
    print(f"  {'':<22} {'layer 0':>11} {'layer 11':>11} {'ratio':>10}")
    for name, cls in variants:
        g = layer_grads(cls)
        print(f"  {name:<22} {g[0]:>11.2e} {g[-1]:>11.2e} {g[0] / g[-1]:>10.1e}")
    print("\nThe usual story is that residual connections are what prevent vanishing gradients.")
    print("Measured, that is too strong: with LayerNorm still in place, removing the residuals")
    print("barely moves layer 0. Normalization rescales at every layer, which by itself stops")
    print("the geometric decay. It takes losing BOTH to collapse the gradient.")

    # 6 -------------------------------------------------------------------------------------
    rule("6. The same experiment with no normalization anywhere (plain 20-layer MLP)")
    print("  Here residuals really are the only thing holding the gradient up.\n")
    print(f"  {'layer':>6} {'with residual':>16} {'without':>16}")
    with_res, without = mlp_grads(True), mlp_grads(False)
    for i in range(0, 20, 4):
        print(f"  {i:>6} {with_res[i]:>16.2e} {without[i]:>16.2e}")
    print(f"\n  {'layer 0':<14} residual {with_res[0]:.2e}   plain {without[0]:.2e}")
    print(f"  {'':<14} {with_res[0] / without[0]:.1e}x more signal reaching the first layer")
    print("\n  with residual   " + bar(with_res[0]))
    print("  without         " + bar(without[0]))

    # 7 -------------------------------------------------------------------------------------
    rule("7. A block that contributes nothing is the identity")
    blk = TransformerBlock(CFG).eval()
    xs = torch.randn(1, 8, D)
    with torch.no_grad():
        # Biases are optional (cfg.linear_bias), so zero whatever these projections have.
        for projection in (blk.attn.W_o, blk.ffn.down):
            for param in projection.parameters():
                param.zero_()
        print(f"  output == input: {torch.allclose(blk(xs), xs, atol=1e-6)}")
    print("\nThat is the representational argument for residuals, separate from gradients: each")
    print("block reads the residual stream, computes, and adds a correction back. Doing nothing")
    print("is the default, so a block only has to learn what it wants to change.")

    # 8 -------------------------------------------------------------------------------------
    rule("8. Pre-norm vs post-norm")
    pre, post = TransformerBlock(CFG).eval(), PostNormBlock(CFG).eval()
    with torch.no_grad():
        print(f"  pre-norm  output std: {pre(xs).std():.4f}")
        print(f"  post-norm output std: {post(xs).std():.4f}   <- normalized, so pinned to ~1")
    print("\n  pre-norm : h = h + attn(norm(h))   residual path stays clean end to end")
    print("  post-norm: h = norm(h + attn(h))   normalization sits ON the residual path")
    print("\nMost transformer diagrams draw post-norm, because they redraw the 2017 figure. GPT-2")
    print("and everything since use pre-norm: post-norm needs careful warmup to converge at all.")


if __name__ == "__main__":
    main()
