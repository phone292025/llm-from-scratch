"""Talk to the model you trained.

    python scripts/chat.py

Type text, watch it continue. `/think on` shows what it was weighing at each word.

A note on what this is. The model is a *base* model: it continues text, it does not answer
questions, because nothing in children's stories looks like a conversation with an assistant. Ask
it a question and it will write a story in which somebody asks that question. Making it answer is
instruction fine-tuning -- v3 in docs/ROADMAP.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llm.bpe import Tokenizer
from llm.checkpoint import load, load_config
from llm.console import use_utf8
from llm.generate import iter_generate
from llm.model import GPTModel

DEFAULT_CKPT = "checkpoints/tinystories-full/final.pt"
DEFAULT_TOK = "data/tinystories-10000.json"

DIM, BOLD, CYAN, YELLOW, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[0m"

HELP = f"""
{BOLD}commands{RESET}
  /think on|off   show what the model weighed at each word
  /temp 0.8       0 = always the likeliest word, higher = more adventurous
  /top_p 0.9      consider only the likeliest words summing to this probability
  /top_k 0        consider only this many words (0 = off)
  /len 120        how many tokens to generate
  /seed 42        fix the randomness so a run repeats; /seed off for fresh
  /keep on|off    keep the story so far as context for the next turn
  /new            forget the story so far
  /settings       show the current settings
  /help           this
  /quit           leave
"""


class Chat:
    def __init__(self, model: GPTModel, tok: Tokenizer, device: str) -> None:
        self.model, self.tok, self.device = model, tok, device
        self.temperature, self.top_p, self.top_k = 0.8, 0.9, 0
        self.max_new, self.seed, self.think, self.keep = 120, None, False, True
        self.history: list[int] = []

    # -- generation ---------------------------------------------------------------------------

    def respond(self, prompt: str) -> None:
        ids = self.history + self.tok.encode(prompt) if self.keep else self.tok.encode(prompt)
        if not ids:
            ids = [self.tok.eot_id]
        # The model has a 256-token window; older text simply falls out of view.
        ids = ids[-self.model.cfg.context_length :]
        idx = torch.tensor([ids], device=self.device)

        produced: list[int] = []
        moments: list[tuple[int, float, list[tuple[str, float]]]] = []
        shown = ""

        print(f"{CYAN}", end="", flush=True)
        for next_id, logits in iter_generate(
            self.model, idx, self.max_new,
            temperature=self.temperature,
            top_k=self.top_k or None,
            top_p=self.top_p or None,
            eos_id=self.tok.eot_id,
            seed=self.seed,
        ):
            token = int(next_id[0, 0])
            if token == self.tok.eot_id:
                break
            produced.append(token)

            # Decode the whole tail each time and print only what is new. Decoding one token at a
            # time would break any character whose UTF-8 bytes span two tokens.
            full = self.tok.decode(produced)
            print(full[len(shown) :], end="", flush=True)
            shown = full

            if self.think:
                probs = logits[0].softmax(-1)
                top = torch.topk(probs, 5)
                entropy = float(-(probs * (probs + 1e-12).log()).sum())
                moments.append({
                    "at": len(produced) - 1,
                    "token": token,
                    "chosen_p": float(probs[token]),
                    "entropy": entropy,
                    "top": [
                        (self.tok.decode([int(i)]), float(p))
                        for p, i in zip(top.values, top.indices)
                    ],
                })
        print(RESET)

        if self.keep:
            self.history = (ids + produced)[-self.model.cfg.context_length :]
        if self.think and moments:
            self._show_thinking(moments, produced)

    def _show_thinking(self, moments: list[dict], produced: list[int]) -> None:
        """Where the model was least certain, and what else it nearly said.

        Entropy is the measure: near zero means one word was obvious, high means many were
        plausible. The high ones are the forks -- the points where the story could have gone
        somewhere else entirely.
        """
        confident = sum(m["chosen_p"] > 0.9 for m in moments)
        print(f"\n{DIM}--- what it was weighing ---{RESET}")
        print(f"{DIM}{len(moments)} words. {confident} were near-certain (>90% on one word); "
              f"the other {len(moments) - confident} were real choices.{RESET}")

        for m in sorted(moments, key=lambda m: -m["entropy"])[:5]:
            picked = self.tok.decode([m["token"]]).strip() or "␣"
            before = self.tok.decode(produced[max(0, m["at"] - 5) : m["at"]]).strip()
            in_top = any(word.strip() == picked for word, _ in m["top"])

            options = "  ".join(
                f"{YELLOW if word.strip() == picked else ''}{word.strip() or '␣'} {p:.0%}"
                f"{RESET if word.strip() == picked else ''}"
                for word, p in m["top"]
            )
            note = "" if in_top else f"  {DIM}(outside its own top five){RESET}"
            print(f"\n  {DIM}...{before}{RESET} {YELLOW}{picked}{RESET}"
                  f"  {DIM}<- picked at {m['chosen_p']:.0%}{RESET}{note}")
            print(f"      it also considered: {options}")

        print(f"\n{DIM}Yellow is what it picked. Sampling does not always take the likeliest word --")
        print(f"that is what makes the stories vary; /temp 0 makes it always take the top one.")
        print(f"These are the real probabilities it computed over all 10,000 words, not a")
        print(f"description of them.{RESET}")

    # -- commands -----------------------------------------------------------------------------

    def command(self, line: str) -> bool:
        """Returns False when it is time to quit."""
        parts = line.split()
        name, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")
        numbers = {"/temp": "temperature", "/top_p": "top_p", "/top_k": "top_k", "/len": "max_new"}

        if name in ("/quit", "/exit", "/q"):
            return False
        if name == "/help":
            print(HELP)
        elif name == "/new":
            self.history.clear()
            print(f"{DIM}forgotten{RESET}")
        elif name in ("/think", "/keep"):
            setattr(self, name[1:], arg != "off")
            print(f"{DIM}{name[1:]} {'on' if getattr(self, name[1:]) else 'off'}{RESET}")
        elif name == "/seed":
            self.seed = None if arg in ("", "off", "none") else int(arg)
            print(f"{DIM}seed {self.seed}{RESET}")
        elif name in numbers:
            try:
                value = float(arg)
            except ValueError:
                print(f"{DIM}{name} needs a number, e.g. {name} 0.8{RESET}")
                return True
            setattr(self, numbers[name], int(value) if name in ("/top_k", "/len") else value)
            print(f"{DIM}{numbers[name]} = {getattr(self, numbers[name])}{RESET}")
        elif name == "/settings":
            print(f"{DIM}temperature {self.temperature}  top_p {self.top_p}  top_k {self.top_k}  "
                  f"length {self.max_new}  seed {self.seed}  think {self.think}  keep {self.keep}\n"
                  f"context held: {len(self.history)}/{self.model.cfg.context_length} tokens{RESET}")
        else:
            print(f"{DIM}unknown command. /help for the list.{RESET}")
        return True


def main() -> None:
    use_utf8()
    p = argparse.ArgumentParser(description="Talk to a model you trained.")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--tokenizer", default=DEFAULT_TOK)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    if not Path(args.checkpoint).exists():
        raise SystemExit(f"no checkpoint at {args.checkpoint} -- train one first, see docs/TESTING.md")

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    cfg = load_config(args.checkpoint)
    model = GPTModel(cfg)
    load(args.checkpoint, model, map_location=device)
    model.to(device).eval()
    tok = Tokenizer.load(args.tokenizer)
    if tok.vocab_size != cfg.vocab_size:
        raise SystemExit(
            f"tokenizer has {tok.vocab_size:,} tokens, model expects {cfg.vocab_size:,}"
        )

    print(f"{BOLD}Your model{RESET}  {model.num_parameters():,} parameters, "
          f"{cfg.vocab_size:,}-token vocabulary, {cfg.context_length}-token memory, on {device}")
    print(f"{DIM}It continues text; it does not answer questions. Give it the start of a story.")
    print(f"Try: Once upon a time    or    Tom found a key in the garden")
    print(f"/help for commands, /think on to see what it is weighing, /quit to leave.{RESET}")

    chat = Chat(model, tok, device)
    while True:
        try:
            line = input(f"\n{BOLD}you ›{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            if not chat.command(line):
                break
            continue
        try:
            chat.respond(line)
        except KeyboardInterrupt:
            print(f"\n{DIM}stopped{RESET}")
    print(f"{DIM}bye{RESET}")


if __name__ == "__main__":
    main()
