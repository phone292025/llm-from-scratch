"""Build DPO pairs whose rejected answer is the chat model's own mistake.

    python scripts/make_onpolicy_pairs.py

Reads the hand-written pairs (data/chat/preferences.jsonl) and what the chat model actually said
to the same prompts, judged (data/chat/own_answers.jsonl, from scripts/sample_own_answers.py).
Where the model's answer was judged bad, it replaces the hand-written rejected answer: training
then lowers the mistake this model really makes. Where its answer was fine, the hand-written pair
stays as it was -- lowering an answer that is fine would teach the model to stop giving it.

Writes two files, same prompts in the same order, so the held-out split is the same twelve
prompts as the hand-written runs:

- data/chat/preferences_onpolicy.jsonl -- each mistake whole (run 4 in notes/09 section 8). The
  mistakes averaged 199 characters against 105 for the chosen answers, and the run learned "short
  beats long": terse answers, and repetition loops.
- data/chat/preferences_onpolicy_first.jsonl -- each mistake cut after the sentences that make
  it, which own_answers.jsonl records as `keep` (run 5): usually the first, the second where the
  first sentence is right ("No, I am not. I am a large language model trained by OpenAI...").
  The opening is what decides what the model writes, and it is about as long as a chosen answer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PAIRS = Path("data/chat/preferences.jsonl")
OWN = Path("data/chat/own_answers.jsonl")
OUT = Path("data/chat/preferences_onpolicy.jsonl")
OUT_FIRST = Path("data/chat/preferences_onpolicy_first.jsonl")

# Kept exactly as run 4 used it, so its file can be rebuilt byte for byte.
SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?=\s|$)")
# Splitting into sentences also has to step over initials: "directed by David O. Russell".
SENTENCE = re.compile(r"(?<!\b[A-Z])[.!?][\"')\]]*(?=\s|$)")


def trim(answer: str) -> str:
    """A reply cut off by the length limit, up to its last complete sentence -- so the rejected
    answer ends where the model would have paused, not halfway through a word."""
    ends = [m.end() for m in SENTENCE_END.finditer(answer)]
    return answer[: ends[-1]].strip() if ends else answer.strip()


def sentences(text: str) -> list[str]:
    out, start = [], 0
    for m in SENTENCE.finditer(text):
        out.append(text[start : m.end()].strip())
        start = m.end()
    if text[start:].strip():
        out.append(text[start:].strip())
    return out


def opening(answer: str, keep: int) -> str:
    """The first `keep` sentences: where the mistake is made."""
    return " ".join(sentences(answer)[:keep])


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def build(pairs: list[dict], own: list[dict], shorten: bool = False) -> list[dict]:
    if len(pairs) != len(own):
        raise SystemExit(f"{len(pairs)} pairs but {len(own)} own answers -- sample them again")
    out = []
    for n, (pair, mine) in enumerate(zip(pairs, own, strict=True)):
        if pair["messages"] != mine["messages"]:
            raise SystemExit(f"prompt {n} differs between the two files -- sample them again")
        if mine["verdict"] not in ("good", "bad"):
            raise SystemExit(f"prompt {n} has no verdict yet -- judge {OWN} first")
        if shorten and mine["verdict"] == "bad":
            rejected = opening(mine["answer"], mine["keep"])
        else:
            rejected = mine["answer"] if mine["finished"] else trim(mine["answer"])
        if mine["verdict"] == "bad" and rejected:
            out.append({"messages": pair["messages"], "chosen": pair["chosen"],
                        "rejected": rejected})
        else:
            out.append(pair)
    return out


def main() -> None:
    pairs, own = load(PAIRS), load(OWN)
    for path, shorten in ((OUT, False), (OUT_FIRST, True)):
        data = build(pairs, own, shorten)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data),
                        encoding="utf-8")
        replaced = [r for r, p in zip(data, pairs, strict=True) if r["rejected"] != p["rejected"]]
        print(f"{path}: {len(replaced)} of {len(data)} pairs reject the model's own mistake; "
              f"there, chosen {mean_length(replaced, 'chosen'):.0f} characters, rejected "
              f"{mean_length(replaced, 'rejected'):.0f}")


def mean_length(records: list[dict], key: str) -> float:
    return sum(len(r[key]) for r in records) / max(len(records), 1)


if __name__ == "__main__":
    main()
