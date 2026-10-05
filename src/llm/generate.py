"""Turning a probability distribution into text.

Twenty lines that decide whether the same trained model produces dull repetition or readable
prose. See notes/06_generation.md.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor

from llm.kv_cache import KVCache
from llm.model import GPTModel
from llm.tokenizer import BPETokenizer


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Stretch or compress the gaps between logits before the softmax.

    Low temperature exaggerates differences so the top token dominates; high temperature flattens
    them so rare tokens become plausible. Dividing by something near zero overflows, so
    temperature 0 is handled by the caller as greedy decoding.
    """
    return logits if temperature == 1.0 else logits / temperature


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Keep the k largest logits, discard the rest.

    Removes the absurd tail entirely. Its weakness is that k is fixed: sometimes the model is
    confident and 3 tokens are plausible, sometimes 200 are.
    """
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Keep the smallest set of tokens whose cumulative probability exceeds p (nucleus sampling).

    Strictly the better idea than top-k, because the set size adapts: a confident model keeps 3
    tokens, an uncertain one keeps 200.
    """
    if not 0.0 < p < 1.0:
        return logits
    ordered, indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = ordered.softmax(dim=-1).cumsum(dim=-1)
    # Subtracting the token's own probability keeps the first token past the threshold, so the
    # nucleus is never empty even when one token already exceeds p on its own.
    remove = cumulative - ordered.softmax(dim=-1) > p
    return logits.masked_fill(remove.scatter(-1, indices, remove), float("-inf"))


def sample_next(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Pick one token per row from `(B, vocab)` logits."""
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = apply_temperature(logits, temperature)
    if top_k:
        logits = top_k_filter(logits, top_k)
    if top_p:
        logits = top_p_filter(logits, top_p)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate(
    model: GPTModel,
    idx: Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
    use_cache: bool = True,
    seed: int | None = None,
) -> Tensor:
    """Autoregressive generation. `temperature=0` means greedy.

    The whole sequence at once. `iter_generate` is the same loop yielding a token at a time, for
    printing as it goes or inspecting what the model was weighing.
    """
    for next_id, _ in iter_generate(
        model, idx, max_new_tokens,
        temperature=temperature, top_k=top_k, top_p=top_p,
        eos_id=eos_id, use_cache=use_cache, seed=seed,
    ):
        idx = torch.cat([idx, next_id], dim=1)
    return idx


@torch.no_grad()
def iter_generate(
    model: GPTModel,
    idx: Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
    use_cache: bool = True,
    seed: int | None = None,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield `(next_id, logits)` for each step, where `logits` are the raw scores that choice was
    made from -- before temperature or any filtering, so they show what the model actually
    believed rather than what the sampler was allowed to pick.

    With `use_cache`, the prompt is processed once and every later step feeds a single token.
    Without it, the whole (cropped) sequence is re-run each step. Both must produce identical
    output for the same seed -- that is a test, because caching bugs are fluent and subtly wrong.
    """
    was_training = model.training
    model.eval()
    device = idx.device
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    batch = idx.shape[0]
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    cache = None
    if use_cache:
        cache = KVCache(
            model.cfg,
            batch_size=batch,
            max_len=model.cfg.context_length,
            device=device,
            dtype=next(model.parameters()).dtype,
        )

    try:
        step_input = idx[:, -model.cfg.context_length :]
        for _ in range(max_new_tokens):
            if cache is not None and cache.pos + step_input.shape[1] > model.cfg.context_length:
                # The cache is full. Absolute position embeddings have no rows beyond this, so
                # fall back to the uncached path, which crops. RoPE has no rows to run out of --
                # its table is merely built to context_length -- so for a rope model this is a
                # choice, not a wall. Cropping stays the default because quality past the trained
                # length degrades either way; extrapolating well needs more than a longer table.
                cache, step_input = None, idx[:, -model.cfg.context_length :]

            logits = model(step_input, cache=cache)[:, -1, :]
            next_id = sample_next(logits, temperature, top_k, top_p, generator)

            if eos_id is not None:
                # Once a sequence is finished, keep emitting eos so the batch stays rectangular.
                next_id = torch.where(finished.unsqueeze(1), torch.full_like(next_id, eos_id), next_id)
                finished |= next_id.squeeze(1) == eos_id

            idx = torch.cat([idx, next_id], dim=1)
            yield next_id, logits
            # With a cache the model has already seen everything before this token.
            step_input = next_id if cache is not None else idx[:, -model.cfg.context_length :]

            if eos_id is not None and bool(finished.all()):
                break
    finally:
        model.train(was_training)


def generate_text(
    model: GPTModel, tok: BPETokenizer, prompt: str, max_new_tokens: int = 50, **kwargs
) -> str:
    """Convenience wrapper: string in, string out. The prompt is included in the returned text."""
    device = next(model.parameters()).device
    # An empty prompt means "sample unconditionally", which still needs one token to start from.
    # <|endoftext|> is the right one: in training it is what precedes every document.
    encoded = tok.encode(prompt) or [tok.eot_id]
    ids = torch.tensor([encoded], device=device)
    out = generate(model, ids, max_new_tokens, eos_id=kwargs.pop("eos_id", tok.eot_id), **kwargs)
    return tok.decode(out[0].tolist())


if __name__ == "__main__":
    import time

    from llm.console import use_utf8

    use_utf8()

    from llm.gpt2_weights import load_pretrained_gpt2

    tok = BPETokenizer()
    model = load_pretrained_gpt2("124M")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    prompt = "The future of artificial intelligence"

    settings = [
        ("greedy", dict(temperature=0.0)),
        ("temperature 0.7", dict(temperature=0.7)),
        ("temperature 1.3", dict(temperature=1.3)),
        ("top-k 50", dict(temperature=1.0, top_k=50)),
        ("top-p 0.9", dict(temperature=0.8, top_p=0.9)),
    ]
    for name, kwargs in settings:
        text = generate_text(model, tok, prompt, 30, seed=0, **kwargs)
        print(f"\n{name}\n  {text!r}")

    ids = torch.tensor([tok.encode(prompt)], device=device)
    for use_cache in (False, True):
        start = time.perf_counter()
        generate(model, ids, 100, temperature=0.0, use_cache=use_cache)
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"\n{'with' if use_cache else 'without':>7} cache: {time.perf_counter() - start:.2f}s")
