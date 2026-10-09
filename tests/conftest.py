"""Fixtures shared by the chat and preference-tuning tests.

A Qwen3-shaped model and tokenizer small enough to train on the CPU in seconds, built offline so
the tests run in CI, and one finished chat run on it. Session-scoped: every test file that asks
for them shares the same ones.
"""

from __future__ import annotations

import json

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def _cpu_only():
    """Every test runs on the CPU, as in CI, and none may start CUDA at all.

    Session-scoped so it is in place before the session fixtures below run. Without it, saving a
    chat checkpoint asks CUDA for its random state, and that alone opens a GPU context of a few
    hundred MB -- enough to push a training run already holding 7.5 of 8 GB over the edge.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "is_available", lambda: False)
        yield
    assert not torch.cuda.is_initialized(), "a test started CUDA; tests must stay on the CPU"


PERSONA = [
    [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hey! What's up?"}],
    [{"role": "user", "content": "what sup"},
     {"role": "assistant", "content": "Not much. What's up with you?"}],
    [{"role": "user", "content": "my name is Phone"},
     {"role": "assistant", "content": "Nice to meet you, Phone."},
     {"role": "user", "content": "what's my name"},
     {"role": "assistant", "content": "Phone -- you told me a moment ago."}],
    [{"role": "user", "content": "who are you"}, {"role": "assistant", "content": "A language model."}],
    [{"role": "user", "content": "thanks"}, {"role": "assistant", "content": "Anytime."}],
    [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "See you."}],
]


@pytest.fixture(scope="session")
def tiny_base(tmp_path_factory):
    """A Qwen3-shaped model and a byte-level tokenizer with the ChatML markers, built offline.

    Nothing is downloaded, so this runs wherever the tests do. One layer of width 32 trains in
    milliseconds a step on the CPU -- enough to drive the real training loop end to end.
    """
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    from llm.chat_data import DEFAULT_SYSTEM

    base = tmp_path_factory.mktemp("tiny-base")
    raw = Tokenizer(models.BPE())
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    texts = [DEFAULT_SYSTEM] + [m["content"] for convo in PERSONA for m in convo]
    raw.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=320, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"]))
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    assert len(tok.encode("<|im_end|>", add_special_tokens=False)) == 1, "markers must be one token"
    tok.save_pretrained(base)

    cfg = transformers.Qwen3Config(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=256,
        tie_word_embeddings=True)
    torch.manual_seed(0)
    transformers.Qwen3ForCausalLM(cfg).save_pretrained(base)
    return base


def _run_chat_sft(base, out, persona_file, monkeypatch, *extra) -> None:
    import sys

    from llm.chat_sft import main

    argv = ["chat_sft", "--base", str(base), "--out", str(out), "--persona", str(persona_file),
            "--persona-repeat", "3", "--ultrachat", "0", "--oasst", "0", "--max-len", "96",
            "--epochs", "1", "--batch-size", "1", "--accum", "2", "--lr", "1e-3", "--warmup", "1",
            "--lora-r", "4", "--lora-alpha", "8", "--eval-every", "2", "--sample-every", "9",
            "--save-every", "2", "--device", "cpu", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    main()


@pytest.fixture(scope="session")
def finished_run(tiny_base, tmp_path_factory):
    """One complete run: 18 conversations, 9 optimizer steps, saving every 2."""
    monkeypatch = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("chat-run")
    persona_file = root / "persona.jsonl"
    persona_file.write_text("\n".join(json.dumps({"messages": c}) for c in PERSONA), encoding="utf-8")
    try:
        _run_chat_sft(tiny_base, root / "a", persona_file, monkeypatch)
    finally:
        monkeypatch.undo()
    return root / "a", persona_file




@pytest.fixture(scope="session")
def persona() -> list[list[dict]]:
    return PERSONA


@pytest.fixture(scope="session")
def run_chat_sft():
    """`python -m llm.chat_sft` as a function: (base, out, persona_file, monkeypatch, *extra)."""
    return _run_chat_sft
