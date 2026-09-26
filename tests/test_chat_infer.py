"""Tests for chat inference: streaming, loading, and finding checkpoints.

The streaming tests exist because `scripts/talk.py` shipped with a streamer that printed the
prompt and saved the *system prompt* as every reply -- invisible until someone had a second turn
of conversation, and nobody had. The load-bearing one runs Hugging Face's real `generate()`,
because the bug lived in how that function calls the streamer, not in anything a fake would
reproduce.
"""

from __future__ import annotations

import io
import json

import pytest
import torch

from llm.chat_infer import (
    REPLACEMENT,
    ReplyStreamer,
    find_checkpoint,
    fit_history,
    read_message,
    resolve_adapter,
    untie_output_head,
)

STOP = "<|im_end|>"

# peft warns whenever a tied model has lm_head in its adapter. These tests build that situation on
# purpose -- it is the one untie_output_head exists for -- so the warnings are the subject, not a
# symptom. Filtered by message so any *other* warning still shows.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Model has `tie_word_embeddings=True`:UserWarning"),
    pytest.mark.filterwarnings("ignore:Model with `tie_word_embeddings=True`:UserWarning"),
    pytest.mark.filterwarnings("ignore:Setting `save_embedding_layers` to `True`:UserWarning"),
]


class ByteTok:
    """Every id below 256 is that byte; 256 is the stop marker. The bottom of a real BPE vocabulary
    looks like this, which is exactly where multi-byte characters get split."""

    STOP_ID = 256

    def decode(self, ids, skip_special_tokens=False):
        out = b""
        for i in ids:
            out += STOP.encode() if i == self.STOP_ID else bytes([i]) if i < 256 else b""
        return out.decode("utf-8", errors="replace")


def stream(ids: list[int], prompt: list[int] = (1, 2, 3)) -> tuple[ReplyStreamer, str]:
    screen = io.StringIO()
    s = ReplyStreamer(ByteTok(), out=screen)
    s.put(torch.tensor([list(prompt)]))           # generate() sends the prompt first, as (1, n)
    for i in ids:
        s.put(torch.tensor([i]))                  # then one token per step, as (1,)
    s.end()
    return s, screen.getvalue()


# --- the streamer --------------------------------------------------------------------------------


def test_the_prompt_is_neither_printed_nor_kept() -> None:
    s, screen = stream(list(b"Hey!"), prompt=list(b"SYSTEM PROMPT"))
    assert "SYSTEM" not in screen
    assert s.reply == "Hey!"


def test_the_reply_stops_at_the_marker() -> None:
    """Anything after <|im_end|> is the model starting a new turn, not part of this reply."""
    s, screen = stream(list(b"Hey!") + [ByteTok.STOP_ID] + list(b"\nuser: more"))
    assert s.reply == "Hey!"
    assert screen == "Hey!"


def test_multibyte_characters_print_whole() -> None:
    """Burmese and emoji need several bytes each; decoded one token at a time, every fragment is
    U+FFFD. The streamer must hold a fragment back until the character is complete."""
    text = "မင်္ဂလာပါ 👋"
    s, screen = stream(list(text.encode()))
    assert REPLACEMENT not in screen
    assert screen == text and s.reply == text


def test_a_reply_cut_off_mid_character_drops_the_fragment() -> None:
    """The length limit can stop generation between the bytes of one character. What remains is
    half a character, U+FFFD when decoded -- not something the model said."""
    burmese = "\u1019\u1004\u103a"
    cut = list(burmese.encode())[:-1]          # the last byte never arrives
    s, screen = stream(cut)
    assert REPLACEMENT not in s.reply and REPLACEMENT not in screen
    assert s.reply == burmese[:-1]


def test_a_stop_marker_split_across_tokens_never_flashes() -> None:
    """If the marker arrives as ordinary text pieces, its prefix must not reach the screen."""
    s, screen = stream(list(b"Hey!") + list(STOP.encode()) + list(b"junk"))
    assert screen == "Hey!"
    assert "<|" not in screen


def test_the_streamer_sees_exactly_what_generate_produced() -> None:
    """The real thing: Hugging Face's generate() with a tiny random Qwen3 model.

    The streamer's tokens must be the generated tokens and nothing else. The shipped version
    also held the prompt, which is how the system prompt ended up saved as every reply.
    """
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen3Config(
        vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    prompt = torch.tensor([[5, 6, 7, 8]])
    s = ReplyStreamer(ByteTok(), out=io.StringIO())
    with torch.no_grad():
        out = model.generate(prompt, streamer=s, max_new_tokens=6, do_sample=False,
                             pad_token_id=0, eos_token_id=ByteTok.STOP_ID)
    assert s.ids == out[0, prompt.shape[1]:].tolist()


# --- merging the adapter into a tied model -------------------------------------------------------


def _tiny_tied_qwen():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return transformers.Qwen3ForCausalLM(cfg).eval()


def _trained_adapter(tmp_path):
    """An adapter fitted the way chat_sft fits one: tied base, lm_head among the targets."""
    peft = pytest.importorskip("peft")
    model = peft.get_peft_model(
        _tiny_tied_qwen(),
        peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "lm_head"]),
    )
    torch.manual_seed(1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.2)   # LoRA starts at zero; a trained adapter does not
    x = torch.tensor([[1, 5, 9, 13, 2]])
    with torch.no_grad():
        trained = model(x).logits
    model.save_pretrained(tmp_path)
    return x, trained


def test_merging_into_a_tied_model_moves_the_input_embeddings(tmp_path) -> None:
    """The hazard, documented: without untying, the lm_head delta lands in the shared tensor."""
    peft = pytest.importorskip("peft")
    _trained_adapter(tmp_path)
    base = _tiny_tied_qwen()
    before = base.get_input_embeddings().weight.detach().clone()
    merged = peft.PeftModel.from_pretrained(base, tmp_path).merge_and_unload()
    assert not torch.allclose(merged.get_input_embeddings().weight, before)


def test_untied_merge_reproduces_the_trained_model_exactly(tmp_path) -> None:
    """With the head untied first, the merged model computes what the trained one computed, and
    the input embeddings are untouched."""
    peft = pytest.importorskip("peft")
    x, trained = _trained_adapter(tmp_path)
    base = _tiny_tied_qwen()
    before = base.get_input_embeddings().weight.detach().clone()
    untie_output_head(base)
    merged = peft.PeftModel.from_pretrained(base, tmp_path).merge_and_unload()
    assert torch.equal(merged.get_input_embeddings().weight, before)
    with torch.no_grad():
        assert torch.allclose(merged(x).logits, trained, atol=1e-5)


def test_untying_an_already_untied_model_changes_nothing() -> None:
    model = _tiny_tied_qwen()
    untie_output_head(model)
    head = model.get_output_embeddings().weight
    untie_output_head(model)
    assert model.get_output_embeddings().weight is head


# --- finding a checkpoint ------------------------------------------------------------------------


def _fake_checkpoint(d, step: int) -> None:
    (d / "adapter").mkdir(parents=True)
    (d / "adapter" / "adapter_model.safetensors").write_bytes(b"x")
    (d / "adapter" / "adapter_config.json").write_text("{}")
    (d / "state.json").write_text(json.dumps({"step": step, "base": "some/base"}))


def test_the_pointer_wins(tmp_path) -> None:
    _fake_checkpoint(tmp_path / "step-000100", 100)
    _fake_checkpoint(tmp_path / "step-000200", 200)
    (tmp_path / "latest.json").write_text(json.dumps({"dir": "step-000100", "step": 100}))
    assert find_checkpoint(tmp_path) == tmp_path / "step-000100"


def test_the_original_layout_still_resumes(tmp_path) -> None:
    """The chat run in progress was saved before step directories existed."""
    _fake_checkpoint(tmp_path, 200)
    assert find_checkpoint(tmp_path) == tmp_path


def test_a_half_written_save_is_never_used(tmp_path) -> None:
    """stop.bat kills the process; a save cut short has no pointer to it and must be ignored."""
    _fake_checkpoint(tmp_path / "step-000100", 100)
    (tmp_path / "latest.json").write_text(json.dumps({"dir": "step-000100", "step": 100}))
    (tmp_path / ".step-000200.partial" / "adapter").mkdir(parents=True)
    assert find_checkpoint(tmp_path) == tmp_path / "step-000100"


def test_nothing_to_resume(tmp_path) -> None:
    assert find_checkpoint(tmp_path) is None


def test_resolve_accepts_an_adapter_or_a_run(tmp_path) -> None:
    _fake_checkpoint(tmp_path / "step-000300", 300)
    (tmp_path / "latest.json").write_text(json.dumps({"dir": "step-000300", "step": 300}))
    adapter = tmp_path / "step-000300" / "adapter"
    assert resolve_adapter(adapter) == adapter
    assert resolve_adapter(tmp_path) == adapter


# --- reading a message -------------------------------------------------------------------------


def _typed(lines, queued_after):
    """Fake console: `lines` arrive in order; `queued_after` lines were pasted along with the first."""
    feed = iter(lines)
    state = {"read": 0}

    def read_line(prompt):
        state["read"] += 1
        return next(feed)

    def more_waiting():
        return state["read"] <= queued_after

    return read_line, more_waiting


def test_a_pasted_block_is_one_message() -> None:
    """The bug: a pasted assignment arrived as seventy messages and got seventy replies."""
    paste = ["Task 1: Complete the Topology Design", "", "1. Establish wired connections",
             "* For Block A connection to Block B, both routers must be using Se2/0 interfaces."]
    read_line, more = _typed(paste + ["next message"], queued_after=len(paste) - 1)
    assert read_message(read_line, more) == "\n".join(paste)
    assert read_message(read_line, more) == "next message"


def test_a_typed_line_is_its_own_message() -> None:
    read_line, more = _typed(["hi", "how are you"], queued_after=0)
    assert read_message(read_line, more) == "hi"
    assert read_message(read_line, more) == "how are you"


def test_a_three_quote_block_spans_lines_anywhere() -> None:
    """Where a paste cannot be detected -- a pipe, another terminal -- a block still works."""
    read_line, more = _typed(['"""', "line one", "", "line two", '"""', "after"], queued_after=0)
    assert read_message(read_line, more) == "line one\n\nline two"
    assert read_message(read_line, more) == "after"


# --- keeping the conversation bounded ----------------------------------------------------------


def _turns(n):
    out = []
    for i in range(n):
        out += [{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": f"answer {i}"}]
    return out


def _count(messages):
    return 10 * len(messages)


def test_history_within_budget_is_untouched() -> None:
    messages = _turns(3) + [{"role": "user", "content": "now"}]
    assert fit_history(messages, _count, budget=1000) == (messages, 0)


def test_the_oldest_exchanges_go_first_and_in_pairs() -> None:
    messages = _turns(10) + [{"role": "user", "content": "now"}]
    kept, dropped = fit_history(messages, _count, budget=50)
    assert _count(kept) <= 50
    assert kept[-1]["content"] == "now"
    assert kept[0]["role"] == "user", "the history must never start with an orphaned reply"
    assert dropped % 2 == 0 and kept == messages[dropped:]


def test_the_newest_message_is_kept_even_alone_over_budget() -> None:
    """Trimming what the user just said would answer a different question."""
    messages = _turns(2) + [{"role": "user", "content": "a very long paste"}]
    kept, _ = fit_history(messages, lambda m: 9999, budget=100)
    assert kept == [messages[-1]]
