"""Talk to the chat model -- multi-turn, with history.

The difference from `scripts/chat.py` is the whole point of it: this sends the **entire
conversation** on every turn, in ChatML. The old script sent one message at a time wrapped in an
Alpaca instruction template, which is why "NO me do you know who am i" went nowhere -- the model
never saw the line it was responding to.

    python scripts/talk.py                                    # newest checkpoint of the chat run
    python scripts/talk.py --adapter checkpoints/chat/step-000700
    python scripts/talk.py --no-adapter                       # the raw base model, for contrast

Paste as many lines as you like: a paste is one message. To type several lines by hand, put a
line containing only three double quotes before and after them.

Commands: /new /system /temp /top_p /len /history /settings /help /quit

Loading, streaming, reading a message and fitting the history all live in `llm.chat_infer`,
where they are tested. This file is only the conversation loop.
"""

from __future__ import annotations

import argparse

import torch

from llm.chat_data import DEFAULT_SYSTEM, IM_END, render
from llm.chat_infer import (
    ReplyStreamer,
    base_of,
    console_input_waiting,
    fit_history,
    load_chat_model,
    read_message,
)
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

  Paste several lines and they arrive as one message. To type several lines, put a line with
  only three double quotes before and after them.
"""


def main() -> None:
    use_utf8()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="LLM", help="what the model is called on screen")
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
    p.add_argument("--trained-len", type=int, default=768,
                   help="the longest conversation the fine-tune saw, in tokens (chat_sft --max-len)")
    # Chosen by arithmetic, not by finding the edge: at 2,048 tokens the conversation's keys and
    # values take about 0.23 GB beside the model's ~4.1 GB. Probing for the real limit on this
    # laptop is exactly what crashed it.
    p.add_argument("--max-context", type=int, default=2048,
                   help="oldest turns are dropped to keep the conversation under this many tokens")
    p.add_argument("--verbose", action="store_true", help="show which base model and checkpoint load")
    p.add_argument("--device", default=None,
                   help="cuda or cpu; default cuda when available. cpu is slower and needs ~7 GB of RAM")
    args = p.parse_args()

    name = args.name
    adapter = None if args.no_adapter else args.adapter
    base = base_of(adapter, args.base)
    print(f"{GREY}Loading {name}...{OFF}", flush=True)
    model, tok, resolved = load_chat_model(base, adapter, args.device, quiet=not args.verbose)
    if args.verbose:
        print(f"{GREY}base {base}\nadapter {resolved or 'none'}{OFF}")
    if not resolved and not args.no_adapter:
        print(f"{GREY}No trained checkpoint found at {args.adapter}, so this is the untrained base "
              f"model: it continues text rather than answering.{OFF}")

    im_end = tok.convert_tokens_to_ids(IM_END)
    system, messages = args.system, []
    temp, top_p, max_new = args.temp, args.top_p, args.len

    def count_tokens(msgs) -> int:
        return len(tok(render(msgs, system), add_special_tokens=False).input_ids)

    print(f"{GREY}{name} is ready. Type /help for commands, /quit to leave.{OFF}\n")
    while True:
        try:
            print(f"{BOLD}you ›{OFF} ", end="", flush=True)
            line = read_message(input, console_input_waiting)
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/") and "\n" not in line:
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
                print(f"{GREY}temp {temp}  top_p {top_p}  len {max_new}  turns {len(messages)}  "
                      f"context {count_tokens(messages)}/{args.max_context} tokens "
                      f"(fine-tuned on up to {args.trained_len})\n  system: {system}{OFF}\n")
            elif cmd == "help":
                print(f"{GREY}{HELP}{OFF}")
            else:
                print(f"{GREY}unknown command; /help for the list{OFF}")
            continue

        messages.append({"role": "user", "content": line})
        messages, dropped = fit_history(messages, count_tokens, args.max_context)
        if dropped:
            print(f"{GREY}(dropped the {dropped} oldest messages to keep the conversation under "
                  f"{args.max_context} tokens){OFF}")
        used = count_tokens(messages)
        if used > args.max_context:
            print(f"{GREY}(that message alone is {used} tokens -- longer than {name} was trained on "
                  f"by far, so expect a rough answer){OFF}")

        # The whole conversation, tokenized exactly as training tokenized it.
        ids = tok(render(messages, system), return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        print(f"{BLUE}{name} ›{OFF} {BLUE}", end="", flush=True)
        streamer = ReplyStreamer(tok)
        # Sampling knobs only when sampling; greedy decoding with them set earns a warning per turn.
        sampling = {"do_sample": True, "temperature": temp, "top_p": top_p} if temp > 0 else {"do_sample": False}
        try:
            with torch.no_grad():
                model.generate(
                    ids, streamer=streamer, max_new_tokens=max_new, **sampling,
                    eos_token_id=im_end, pad_token_id=tok.pad_token_id,
                    repetition_penalty=args.repetition_penalty,
                )
        except torch.OutOfMemoryError:
            print(OFF)
            torch.cuda.empty_cache()
            messages.pop()
            print(f"{GREY}(not enough GPU memory for that reply -- /new or a shorter message will "
                  f"fit){OFF}\n")
            continue
        except RuntimeError as err:
            if "CUDA" not in str(err) and "CUBLAS" not in str(err):
                raise
            # Unlike running out of memory, a CUDA error leaves the GPU unusable for this process.
            print(OFF)
            print(f"{GREY}(the GPU stopped with an error, so this session has to end -- start "
                  f"talk.py again. If it keeps happening, the BIOS update is the likely fix.)\n"
                  f"  {str(err).splitlines()[0][:160]}{OFF}")
            break
        print(OFF)
        messages.append({"role": "assistant", "content": streamer.reply})


if __name__ == "__main__":
    main()
