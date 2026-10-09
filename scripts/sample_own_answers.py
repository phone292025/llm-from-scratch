"""Ask the chat model every preference prompt, and record what it actually says.

    python scripts/sample_own_answers.py

Writes data/chat/own_answers.jsonl: each prompt of data/chat/preferences.jsonl with the chat
model's greedy reply -- the reply the knowledge check would see -- and an empty verdict, for a
person to judge: "bad" where the reply is worse than the pair's chosen answer, "good" where it is
fine. scripts/make_onpolicy_pairs.py then builds pairs whose rejected side is the model's own
mistake instead of one written by hand.

Why (notes/09 section 8): DPO lowers the probability of the rejected answers in the data. Three
runs on hand-written ones lowered *those* answers, and the model moved to inventions of its own
that no pair named. The mistakes worth lowering are the ones this model makes.

Each answer streams as it is generated, like the knowledge check; the file is rewritten after
every answer, so a stop loses nothing already asked.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from llm.chat_data import IM_END, render
from llm.chat_infer import ReplyStreamer, base_of, load_chat_model
from llm.console import Tee, use_utf8

GREY, BLUE, BOLD, OFF = "\033[90m", "\033[94m", "\033[1m", "\033[0m"


def main() -> None:
    use_utf8()
    os.system("")  # turn on colour codes in a plain Windows console window
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", default="checkpoints/chat")
    p.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--data", default="data/chat/preferences.jsonl")
    p.add_argument("--out", default="data/chat/own_answers.jsonl")
    p.add_argument("--len", type=int, default=100, help="longest reply, in tokens")
    p.add_argument("--log", default="checkpoints/chat/own_answers.log")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    records = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    log = open(args.log, "w", encoding="utf-8")
    out = Tee(log)
    out.write(f"{BOLD}The chat model's own answers{OFF}   {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    out.write(f"{GREY}Loading LLM...{OFF}\n")
    out.flush()
    model, tok, resolved = load_chat_model(base_of(args.adapter, args.base), args.adapter,
                                           args.device, quiet=True)
    out.write(f"{GREY}checkpoint {resolved}; {len(records)} prompts{OFF}\n\n")
    im_end = tok.convert_tokens_to_ids(IM_END)

    answers: list[dict] = []
    started = time.perf_counter()
    for n, record in enumerate(records, 1):
        messages = record["messages"]
        out.write(f"{BOLD}[{n:>2}/{len(records)}]{OFF} you › {messages[-1]['content']}\n"
                  f"  {BLUE}LLM ›{OFF} {BLUE}")
        out.flush()
        ids = tok(render(messages), return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        streamer = ReplyStreamer(tok, out=out)
        with torch.no_grad():
            model.generate(ids, streamer=streamer, max_new_tokens=args.len, do_sample=False,
                           eos_token_id=im_end, pad_token_id=tok.pad_token_id)
        out.write(f"{OFF}\n\n")
        out.flush()
        answers.append({"messages": messages, "answer": streamer.reply,
                        "finished": IM_END in streamer.text, "verdict": None, "why": ""})
        Path(args.out).write_text(
            "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in answers), encoding="utf-8")

    cut = sum(not a["finished"] for a in answers)
    out.write(f"{BOLD}{len(answers)} answers{OFF} in {time.perf_counter() - started:.0f}s, "
              f"{cut} cut off at {args.len} tokens -> {args.out}\n")
    out.flush()
    log.close()


if __name__ == "__main__":
    main()
