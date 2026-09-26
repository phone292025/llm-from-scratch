"""Talk to the chat model -- multi-turn, with history.

The difference from `scripts/chat.py` is the whole point of it: this sends the **entire
conversation** on every turn, in ChatML. The old script sent one message at a time wrapped in an
Alpaca instruction template, which is why "NO me do you know who am i" went nowhere -- the model
never saw the line it was responding to.

    python scripts/talk.py                                    # newest checkpoint of the chat run
    python scripts/talk.py --adapter checkpoints/chat/step-000400
    python scripts/talk.py --no-adapter                       # the raw base model, for contrast

Commands: /new /system /temp /top_p /len /history /settings /help /quit

Loading and streaming live in `llm.chat_infer`, where they are tested. This file is only the
conversation loop.
"""

from __future__ import annotations

import argparse
import sys

import torch

from llm.chat_data import DEFAULT_SYSTEM, IM_END, render
from llm.chat_infer import ReplyStreamer, base_of, load_chat_model
from llm.console import use_utf8

GREY, BLUE, BOLD, OFF = "\033[90m", "\033[94m", "\033[1m", "\033[0m"

HELP = """
  /new              start a fresh conversation
  /system <text>    replace the system prompt
  /temp <float>     sampling temperature (0 = greedy)
  /top_p <float>    nucleus sampling threshold
  /len <int>        maximum tokens per reply
  /history          show exactly what the model is being sent
  /settings         current settings
  /quit             exit
"""


def main() -> None:
    use_utf8()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--adapter", default="checkpoints/chat",
                   help="an adapter directory, or a run directory (its newest checkpoint is used)")
    p.add_argument("--no-adapter", action="store_true")
    p.add_argument("--system", default=DEFAULT_SYSTEM)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--len", type=int, default=400)
    # Off by default. The penalty applies to every token already in the context, history
    # included, so it pushes the model *away* from repeating your name back to you -- the exact
    # behaviour the fine-tune was for.
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    args = p.parse_args()

    adapter = None if args.no_adapter else args.adapter
    base = base_of(adapter, args.base)
    print(f"{GREY}loading {base}...{OFF}", flush=True)
    model, tok, resolved = load_chat_model(base, adapter)
    if resolved:
        print(f"{GREY}adapter {resolved}{OFF}")
    elif not args.no_adapter:
        print(f"{GREY}no checkpoint found at {args.adapter} -- this is the raw base model, which "
              f"completes text rather than answering{OFF}")

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
        # The whole conversation, every turn, tokenized exactly as training tokenized it.
        ids = tok(render(messages, system), return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)

        sys.stdout.write(BLUE)
        streamer = ReplyStreamer(tok)
        with torch.no_grad():
            model.generate(
                ids, streamer=streamer, max_new_tokens=max_new,
                do_sample=temp > 0, temperature=max(temp, 1e-5), top_p=top_p,
                eos_token_id=im_end, pad_token_id=tok.pad_token_id,
                repetition_penalty=args.repetition_penalty,
            )
        print(OFF)
        messages.append({"role": "assistant", "content": streamer.reply})


if __name__ == "__main__":
    main()
