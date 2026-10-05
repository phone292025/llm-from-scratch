"""Loading the fine-tuned chat model and streaming its replies.

Split out of `scripts/talk.py` so it can be tested. The script had none, and two bugs lived in it
unseen, both of the kind that produce plausible output:

**The prompt was streamed as if it were the reply.** Hugging Face's `generate()` hands the whole
prompt to the streamer before the first new token -- which is why its own streamer has a
`skip_prompt` flag. Ours printed the prompt, and saved `text.split("<|im_end|>")[0]` as the
model's reply. The first `<|im_end|>` in that text closes the *system prompt*, so every "reply"
stored in the history was a copy of the system prompt, and from the second turn on the model was
reading a conversation it never had.

**One token at a time does not decode.** Byte-level BPE splits a character that needs several
UTF-8 bytes -- all of Burmese, every emoji -- across tokens, and decoding them one by one prints
`\\ufffd` for each fragment. The streamer here decodes the whole reply so far and prints only the
new, complete part.

There is a third, smaller fix in `load_chat_model`; see `untie_output_head`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import TextIO

import torch
from torch import nn

from llm.chat_data import IM_END

REPLACEMENT = "\ufffd"


class ReplyStreamer:
    """Receives tokens from `generate(streamer=...)`, prints the reply as it arrives, keeps it.

    The protocol is two methods: `put(ids)`, called first with the prompt and then once per new
    token, and `end()`. The first call is skipped -- that is the whole of the first bug.
    """

    def __init__(self, tok, out: TextIO | None = None, stop: str = IM_END) -> None:
        self.tok = tok
        self.out = out if out is not None else sys.stdout
        self.stop = stop
        self.ids: list[int] = []
        self._prompt_pending = True
        self._shown = 0
        self._stopped = False

    def put(self, value) -> None:
        if self._prompt_pending:
            self._prompt_pending = False
            return
        tokens = value.reshape(-1).tolist() if hasattr(value, "reshape") else list(value)
        self.ids.extend(int(t) for t in tokens)
        self._flush(final=False)

    def end(self) -> None:
        self._flush(final=True)

    @property
    def text(self) -> str:
        """Everything generated so far, decoded as a whole."""
        return self.tok.decode(self.ids, skip_special_tokens=False)

    @property
    def reply(self) -> str:
        """The reply as it belongs in the history: up to the stop marker, trimmed.

        A reply cut off by the length limit can end halfway through a multi-byte character,
        which decodes to U+FFFD. That half-character was never content, so it is dropped.
        """
        return self.text.split(self.stop)[0].rstrip(REPLACEMENT).strip()

    def _flush(self, final: bool) -> None:
        if self._stopped:
            return
        text = self.text
        if self.stop in text:
            text, self._stopped = text.split(self.stop)[0], True
        else:
            # An incomplete multi-byte character decodes to U+FFFD until its last byte arrives.
            # Hold it back rather than print a fragment; the next token usually completes it --
            # and at the end, a fragment left by the length limit is dropped rather than shown.
            text = text.rstrip(REPLACEMENT)
        if not final and not self._stopped:
            # A stop marker arriving as several pieces would otherwise flash its prefix.
            for k in range(len(self.stop) - 1, 0, -1):
                if text.endswith(self.stop[:k]):
                    text = text[:-k]
                    break
        if len(text) > self._shown:
            self.out.write(text[self._shown :])
            self.out.flush()
            self._shown = len(text)


BLOCK = '"""'


def read_message(read_line, more_waiting) -> str:
    """One message from the user, however many lines it spans.

    `read_line(prompt)` returns the next line; `more_waiting()` says whether more input is already
    queued. Reading line by line, the first version turned a pasted assignment into seventy
    separate messages: the model answered every fragment ("let me know if you need any help with
    Task 1"), the history ballooned, and when the program then crashed the rest of the paste landed
    in PowerShell as commands. Two ways to send several lines as one message:

    - **paste them.** Every line already waiting when the first is read is part of the same paste.
    - **type a block.** A line that is only three double quotes opens it; another closes it.
    """
    first = read_line("")
    if first.strip() == BLOCK:
        lines = []
        while (line := read_line("... ")).strip() != BLOCK:
            lines.append(line)
        return "\n".join(lines).strip()
    lines = [first]
    while more_waiting():
        lines.append(read_line(""))
    return "\n".join(lines).strip()


def console_input_waiting(wait: float = 0.03) -> bool:
    """Whether more keystrokes are already queued on the Windows console -- the rest of a paste.

    A paste reaches the console as a burst of key events; after the first line is read the rest
    are still queued. Typed input is not, because nobody types the next line within 30 ms.
    Anywhere else -- a pipe, another OS -- this says no, and a three-quote block still works.
    """
    # A platform check, not a guarded import: mypy reads it, so CI -- type-checking as Linux,
    # where msvcrt has no functions -- skips the rest instead of failing on it.
    if sys.platform != "win32":
        return False
    import msvcrt

    if not sys.stdin.isatty():
        return False
    deadline = time.monotonic() + wait
    while not msvcrt.kbhit():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.005)
    return True


def fit_history(messages: list[dict], count_tokens, budget: int) -> tuple[list[dict], int]:
    """Drop the oldest exchanges until the conversation fits in `budget` tokens.

    Returns the messages kept and how many were dropped. The newest message always stays, even if
    it alone is over budget -- trimming what the user just said would answer a different question.

    Why a budget at all: every turn resends the whole conversation, so without one the prompt grows
    without limit, and so does the GPU memory each reply needs. The model was also fine-tuned on
    conversations of at most 768 tokens, so the far past was never something it learned to use.
    """
    kept, dropped = list(messages), 0
    while len(kept) > 1 and count_tokens(kept) > budget:
        # A user turn and the reply to it go together, so the history never starts mid-exchange.
        cut = 2 if len(kept) > 2 and kept[1]["role"] == "assistant" else 1
        kept, dropped = kept[cut:], dropped + cut
    return kept, dropped


def untie_output_head(model: nn.Module) -> None:
    """Give the output head its own copy of the embedding matrix.

    Qwen3 ties its embeddings: one tensor is both the input lookup and the output projection. The
    LoRA was trained with `lm_head` as a target, so it adds a low-rank delta to the *output* path
    only, while the input lookup stays frozen. Merging the adapter adds that delta to the weight in
    place -- and because the tensor is shared, the input embeddings move too. Measured on the
    step-200 adapter, 151,770 of 151,936 rows shift by about 1.2% of their length. Small today,
    growing with training, and in principle the model you talk to is then not the model you
    trained.

    Two separate tensors keep the two paths exactly as they were during training. The cost is one
    extra copy of the table: 622 MB in bf16, which fits.
    """
    head, embed = model.get_output_embeddings(), model.get_input_embeddings()
    if head is None or head.weight.data_ptr() != embed.weight.data_ptr():
        return
    head.weight = nn.Parameter(embed.weight.detach().clone(), requires_grad=False)
    model.config.tie_word_embeddings = False


def find_checkpoint(run: str | Path) -> Path | None:
    """The directory holding the newest complete checkpoint of a chat run, or None.

    Two layouts. The current one keeps each save in `step-NNNNNN/` and points at the newest with
    `latest.json`, written last and atomically, so a save interrupted by stop.bat is simply never
    pointed at. The original layout put `adapter/`, `optimizer.pt` and `state.json` directly in
    the run directory; runs started before the change still resume from it.
    """
    run = Path(run)
    pointer = run / "latest.json"
    if pointer.exists():
        found = run / json.loads(pointer.read_text(encoding="utf-8"))["dir"]
        if (found / "adapter" / "adapter_model.safetensors").exists():
            return found
    if (run / "adapter" / "adapter_model.safetensors").exists() and (run / "state.json").exists():
        return run
    return None


def resolve_adapter(path: str | Path) -> Path | None:
    """Accept an adapter directory itself, or a run directory in either layout."""
    path = Path(path)
    if (path / "adapter_config.json").exists():
        return path
    found = find_checkpoint(path)
    return found / "adapter" if found else None


def _quiet_loading() -> None:
    """Hide download bars, load-progress bars and library notices. The chat window should say what
    is happening in its own words, not stream the internals of three libraries."""
    import warnings

    from huggingface_hub.utils import disable_progress_bars
    from transformers.utils import logging as hf_logging

    disable_progress_bars()
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    # peft reminds us on every load that lm_head is tied and in the adapter; untie_output_head is
    # the answer to exactly that, so the notice is noise here.
    warnings.filterwarnings("ignore", category=UserWarning, module="peft")


def load_chat_model(base: str, adapter: str | Path | None, device: str | None = None,
                    quiet: bool = False):
    """Base model, ChatML seeding, the adapter merged in. Returns `(model, tokenizer, adapter)`.

    The order matters and every step is required: seed the markers as training did (the adapter
    was fitted on top of that), untie so the merge touches only the output head, then merge.

    The base model is read from the local cache when it is there, so loading makes no network
    requests -- they are slow, and they are what prints the "unauthenticated requests" notice.
    """
    if quiet:
        _quiet_loading()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llm.chat_sft import init_chat_tokens

    def cached_first(loader, *args, **kwargs):
        try:
            return loader(*args, local_files_only=True, **kwargs)
        except OSError:
            return loader(*args, **kwargs)   # first run: not downloaded yet

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = cached_first(AutoTokenizer.from_pretrained, base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # bf16 on the GPU. On the CPU fp32 is both faster and more accurate -- most CPUs have no
    # bf16 matrix units -- and 7 GB of RAM is affordable where 7 GB of VRAM is not.
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = cached_first(AutoModelForCausalLM.from_pretrained, base, dtype=dtype)
    init_chat_tokens(model, tok)

    resolved = resolve_adapter(adapter) if adapter else None
    if resolved is not None:
        from peft import PeftModel

        untie_output_head(model)
        model = PeftModel.from_pretrained(model, resolved).merge_and_unload()
    return model.to(device).eval(), tok, resolved


def base_of(adapter: str | Path | None, default: str) -> str:
    """The base model a run was trained on, read from its state file when there is one."""
    if not adapter:
        return default
    found = find_checkpoint(adapter) or find_checkpoint(Path(adapter).parent)
    if found and (found / "state.json").exists():
        return json.loads((found / "state.json").read_text(encoding="utf-8")).get("base", default)
    return default


if __name__ == "__main__":
    from llm.console import use_utf8

    use_utf8()

    class ByteTok:
        """Byte-level toy: every id is one UTF-8 byte, like the bottom of a BPE vocabulary."""

        def decode(self, ids, skip_special_tokens=False):
            return bytes(ids).decode("utf-8", errors="replace")

    reply = "Hi! မင်္ဂလာပါ 👋"
    ids = list(reply.encode("utf-8"))
    print(f"{len(reply)} characters arrive as {len(ids)} byte tokens\n")

    print("one token at a time, decoded alone:")
    print("  " + "".join(ByteTok().decode([i]) for i in ids))

    print("\nReplyStreamer:")
    s = ReplyStreamer(ByteTok())
    sys.stdout.write("  ")
    s.put([0, 0, 0])              # the prompt: skipped
    for i in ids:
        s.put([i])
    s.end()
    print(f"\n\nreply kept for the history: {s.reply!r}")
