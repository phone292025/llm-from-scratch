"""Load OpenAI's trained GPT-2 weights into our own GPTModel.

This is the correctness oracle for the whole repo. If every matrix, transpose, normalization and
wiring decision in notes 2-4 is right, the model writes fluent English. If anything is subtly
wrong -- one missed transpose, a variance computed with the wrong denominator -- it writes noise.
No unit test is that convincing.

Weights come from HuggingFace as .safetensors rather than the original TensorFlow checkpoints:
identical numbers, and no TensorFlow install on Windows.

See notes/04_the_model.md section 6.
"""

from __future__ import annotations

import torch
from torch import nn

from llm.config import ModelConfig
from llm.model import GPTModel

#: HuggingFace repo ids and the config each one needs.
GPT2_SIZES: dict[str, dict] = {
    "124M": {"repo": "gpt2", "n_layers": 12, "n_heads": 12, "emb_dim": 768},
    "355M": {"repo": "gpt2-medium", "n_layers": 24, "n_heads": 16, "emb_dim": 1024},
    "774M": {"repo": "gpt2-large", "n_layers": 36, "n_heads": 20, "emb_dim": 1280},
    "1558M": {"repo": "gpt2-xl", "n_layers": 48, "n_heads": 25, "emb_dim": 1600},
}


def gpt2_config(size: str = "124M") -> ModelConfig:
    """The exact architecture of a released GPT-2, so its weights fit."""
    if size not in GPT2_SIZES:
        raise ValueError(f"unknown size {size!r}, expected one of {list(GPT2_SIZES)}")
    spec = GPT2_SIZES[size]
    return ModelConfig(
        vocab_size=50257,
        context_length=1024,
        n_layers=spec["n_layers"],
        n_heads=spec["n_heads"],
        emb_dim=spec["emb_dim"],
        norm="layernorm",
        pos="learned",
        act="gelu",
        qkv_bias=True,  # GPT-2 has it, and the checkpoint will not load without it
        tie_weights=True,
        dropout=0.0,
    )


def _assign(param: nn.Parameter, value: torch.Tensor, name: str) -> None:
    """Copy with a shape check, because a silent mismatch here is very expensive to debug."""
    if param.shape != value.shape:
        raise ValueError(f"{name}: model expects {tuple(param.shape)}, got {tuple(value.shape)}")
    with torch.no_grad():
        param.copy_(value)


def load_gpt2_weights(model: GPTModel, size: str = "124M") -> GPTModel:
    """Map a HuggingFace GPT-2 checkpoint onto our parameter names.

    The one real trap: OpenAI's code used Conv1D rather than Linear, so every linear weight is
    stored **transposed** relative to PyTorch's (out_features, in_features) convention. Miss a
    `.T` and the model loads without complaint and emits perfect nonsense.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    spec = GPT2_SIZES[size]
    path = hf_hub_download(repo_id=spec["repo"], filename="model.safetensors")
    sd = load_file(path)

    # Older exports prefix everything with "transformer."; newer ones do not.
    def get(key: str) -> torch.Tensor:
        for candidate in (key, f"transformer.{key}"):
            if candidate in sd:
                return sd[candidate]
        raise KeyError(f"{key!r} not found in checkpoint (have {len(sd)} tensors)")

    _assign(model.tok_emb.weight, get("wte.weight"), "tok_emb")
    _assign(model.pos_emb.weight, get("wpe.weight"), "pos_emb")

    for i, block in enumerate(model.blocks):
        p = f"h.{i}"
        _assign(block.norm1.scale, get(f"{p}.ln_1.weight"), f"{p}.ln_1.scale")
        _assign(block.norm1.shift, get(f"{p}.ln_1.bias"), f"{p}.ln_1.shift")

        # Conv1D stores (in, out); Linear wants (out, in).
        _assign(block.attn.W_qkv.weight, get(f"{p}.attn.c_attn.weight").T, f"{p}.W_qkv")
        _assign(block.attn.W_qkv.bias, get(f"{p}.attn.c_attn.bias"), f"{p}.W_qkv.bias")
        _assign(block.attn.W_o.weight, get(f"{p}.attn.c_proj.weight").T, f"{p}.W_o")
        _assign(block.attn.W_o.bias, get(f"{p}.attn.c_proj.bias"), f"{p}.W_o.bias")

        _assign(block.norm2.scale, get(f"{p}.ln_2.weight"), f"{p}.ln_2.scale")
        _assign(block.norm2.shift, get(f"{p}.ln_2.bias"), f"{p}.ln_2.shift")

        _assign(block.ffn.up.weight, get(f"{p}.mlp.c_fc.weight").T, f"{p}.ffn.up")
        _assign(block.ffn.up.bias, get(f"{p}.mlp.c_fc.bias"), f"{p}.ffn.up.bias")
        _assign(block.ffn.down.weight, get(f"{p}.mlp.c_proj.weight").T, f"{p}.ffn.down")
        _assign(block.ffn.down.bias, get(f"{p}.mlp.c_proj.bias"), f"{p}.ffn.down.bias")

    _assign(model.final_norm.scale, get("ln_f.weight"), "final_norm.scale")
    _assign(model.final_norm.shift, get("ln_f.bias"), "final_norm.shift")

    # GPT-2 ties its head to the token embedding, which our model already does.
    return model


def load_pretrained_gpt2(size: str = "124M") -> GPTModel:
    """Build the right architecture and fill it with real weights."""
    return load_gpt2_weights(GPTModel(gpt2_config(size)), size).eval()


if __name__ == "__main__":
    from llm.tokenizer import BPETokenizer

    tok = BPETokenizer()
    model = load_pretrained_gpt2("124M")
    print(f"loaded GPT-2 124M: {model.num_parameters():,} parameters\n")

    for prompt in (
        "The capital of France is",
        "Every effort moves you",
        "def fibonacci(n):",
    ):
        ids = torch.tensor([tok.encode(prompt)])
        out = model.generate_greedy(ids, max_new_tokens=12)
        print(f"  {prompt!r}\n    -> {tok.decode(out[0].tolist())!r}\n")
