"""Talk to the chat model -- multi-turn, with history.

The difference from `scripts/chat.py` is the whole point of it: this sends the **entire
conversation** on every turn, in ChatML. The old script sent one message at a time wrapped in an
Alpaca instruction template, which is why "NO me do you know who am i" went nowhere -- the model
never saw the line it was responding to.

    python scripts/talk.py
    python scripts/talk.py --adapter checkpoints/chat/adapter
    python scripts/talk.py --base Qwen/Qwen3-1.7B-Base --no-adapter   # the raw base model

Commands: /new /system /temp /top_p /len /history /settings /help /quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from llm.chat_data import DEFAULT_SYSTEM, IM_END, render
from llm.console import use_utf8

GREY, BLUE, BOLD, OFF = "\033[90m", "\033[94m", "\033[1m", "\033[0m"


class Streamer:
    """Print tokens as they arrive, stopping cleanly at the end-of-turn marker.

    `skip_special_tokens` cannot do this job alone: the stop marker has to be *detected* to end
    generation, not merely hidden from the output.
    """

    def __init__(self, tok) -> None:
        self.tok, self.text = tok, ""

    def put(self, ids) -> None:
        piece = self.tok.decode(ids[0] if ids.ndim > 1 else ids, skip_special_tokens=False)
        self.text += piece
        visible = piece.replace(IM_END, "")
        if visible:
            sys.stdout.write(visible)
            sys.stdout.flush()

    def end(self) -> None:
        pass


def load(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = Path(args.adapter) if args.adapter else None
    base = args.base
    if adapter and (adapter.parent / "state.json").exists():
        import json
        base = json.loads((adapter.parent / "state.json").read_text()).get("base", base)

    print(f"{GREY}loading {base}...{OFF}", flush=True)
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)

    if adapter and adapter.exists() and not args.no_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()     # fold LoRA into the weights; faster at inference
        print(f"{GREY}adapter {adapter}{OFF}")
    elif not args.no_adapter:
        print(f"{GREY}no adapter at {adapter} -- this is the raw base model, which completes "
              f"text rather than answering{OFF}")

    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return model, tok


HELP = """
  /new              start a fresh conversation
  /system <text>    replace the system prompt
  /temp <float>     sampling temperature (0 = greedy)
  /top_p <float>    nucleus sampling threshold
  /len <int>        maximum tokens per reply
  /history          show what the model is actually being sent
  /settings         current settings
  /quit             exit
"""


def main() -> None:
    use_utf8()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--adapter", default="checkpoints/chat/adapter")
    p.add_argument("--no-adapter", action="store_true")
    p.add_argument("--system", default=DEFAULT_SYSTEM)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--len", type=int, default=400)
    args = p.parse_args()

    model, tok = load(args)
    im_end = tok.convert_tokens_to_ids(IM_END)
    system, messages = args.system, []
    temp, top_p, max_new = args.temp, args.top_p, args.len

    print(f"{GREY}Type /help for commands, /quit to leave.{OFF}\n")
    while True:
        try:
            line = input(f"{BOLD}you ›{OFF} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            cmd, _, rest = line[1:].partition(" ")
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd in ("new", "reset"):
                messages = []
                print(f"{GREY}new conversation{OFF}\n")
            elif cmd == "system" and rest:
                system = rest
                print(f"{GREY}system prompt set{OFF}\n")
            elif cmd == "temp" and rest:
                temp = float(rest)
            elif cmd == "top_p" and rest:
                top_p = float(rest)
            elif cmd == "len" and rest:
                max_new = int(rest)
            elif cmd == "history":
                print(f"{GREY}{render(messages, system)}{OFF}")
            elif cmd == "settings":
                print(f"{GREY}temp {temp}  top_p {top_p}  len {max_new}  "
                      f"turns {len(messages)}\n  system: {system}{OFF}\n")
            elif cmd == "help":
                print(f"{GREY}{HELP}{OFF}")
            else:
                print(f"{GREY}unknown command; /help for the list{OFF}")
            continue

        messages.append({"role": "user", "content": line})
        # The whole conversation, every turn. This is the fix.
        prompt = render(messages, system)
        ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)

        sys.stdout.write(BLUE)
        streamer = Streamer(tok)
        with torch.no_grad():
            model.generate(
                ids, streamer=streamer, max_new_tokens=max_new,
                do_sample=temp > 0, temperature=max(temp, 1e-5), top_p=top_p,
                eos_token_id=im_end, pad_token_id=tok.pad_token_id,
                repetition_penalty=1.05,
            )
        print(OFF)

        reply = streamer.text.split(IM_END)[0].strip()
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
