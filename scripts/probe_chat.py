"""Ask a chat checkpoint the probe prompts -- the memorized ones and the held-out ones -- side by side.

The probes printed during training are all copied from the hand-written training data, so a
perfect answer to one shows recall. The held-out set asks for the same behaviours in words the
model has never trained on; the difference between the two columns is what the fine-tune actually
generalized.

    python scripts/probe_chat.py                              # newest checkpoint, on the GPU
    python scripts/probe_chat.py --device cpu                 # alongside a training run
    python scripts/probe_chat.py --adapter checkpoints/chat/step-000300

Greedy decoding, so two runs on the same checkpoint print the same thing and checkpoints can be
compared line by line.
"""

from __future__ import annotations

import argparse
import time

import torch

from llm.chat_data import IM_END, render
from llm.chat_infer import base_of, load_chat_model
from llm.chat_sft import HELDOUT_PROMPTS, PROBE_PROMPTS
from llm.console import use_utf8


def answer(model, tok, messages, max_new_tokens: int) -> str:
    ids = tok(render(messages), return_tensors="pt", add_special_tokens=False).input_ids
    ids = ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=tok.convert_tokens_to_ids(IM_END),
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    use_utf8()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", default="checkpoints/chat")
    p.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--device", default=None, help="cuda or cpu; default cuda when available")
    p.add_argument("--len", type=int, default=80)
    args = p.parse_args()

    model, tok, resolved = load_chat_model(base_of(args.adapter, args.base), args.adapter,
                                           args.device)
    print(f"checkpoint  {resolved or 'none -- raw base model'}   device {model.device}\n")

    start = time.perf_counter()
    for label, prompts in (("SEEN IN TRAINING (recall)", PROBE_PROMPTS),
                           ("NEVER SEEN (generalization)", HELDOUT_PROMPTS)):
        print(label)
        for messages in prompts:
            context = " / ".join(m["content"] for m in messages[:-1])
            if context:
                print(f"  [{context}]")
            print(f"  > {messages[-1]['content']}")
            print(f"    {answer(model, tok, messages, args.len)}\n", flush=True)
    print(f"{time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
