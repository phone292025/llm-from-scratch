"""Build DPO pairs whose rejected answer is the chat model's own mistake.

    python scripts/make_onpolicy_pairs.py

Reads the hand-written pairs (data/chat/preferences.jsonl) and what the chat model actually said
to the same prompts, judged (data/chat/own_answers.jsonl, from scripts/sample_own_answers.py).
Where the model's answer was judged bad, it replaces the hand-written rejected answer: training
then lowers the mistake this model really makes. Where its answer was fine, the hand-written pair
stays as it was -- lowering an answer that is fine would teach the model to stop giving it.

Writes data/chat/preferences_onpolicy.jsonl, same prompts in the same order, so the held-out
split is the same twelve prompts as the hand-written runs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PAIRS = Path("data/chat/preferences.jsonl")
OWN = Path("data/chat/own_answers.jsonl")
OUT = Path("data/chat/preferences_onpolicy.jsonl")

SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?=\s|$)")


def trim(answer: str) -> str:
    """A reply cut off by the length limit, up to its last complete sentence -- so the rejected
    answer ends where the model would have paused, not halfway through a word."""
    ends = [m.end() for m in SENTENCE_END.finditer(answer)]
    return answer[: ends[-1]].strip() if ends else answer.strip()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def build(pairs: list[dict], own: list[dict]) -> list[dict]:
    if len(pairs) != len(own):
        raise SystemExit(f"{len(pairs)} pairs but {len(own)} own answers -- sample them again")
    out = []
    for n, (pair, mine) in enumerate(zip(pairs, own, strict=True)):
        if pair["messages"] != mine["messages"]:
            raise SystemExit(f"prompt {n} differs between the two files -- sample them again")
        if mine["verdict"] not in ("good", "bad"):
            raise SystemExit(f"prompt {n} has no verdict yet -- judge {OWN} first")
        rejected = mine["answer"] if mine["finished"] else trim(mine["answer"])
        if mine["verdict"] == "bad" and rejected:
            out.append({"messages": pair["messages"], "chosen": pair["chosen"],
                        "rejected": rejected})
        else:
            out.append(pair)
    return out


def main() -> None:
    pairs, own = load(PAIRS), load(OWN)
    data = build(pairs, own)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data),
                   encoding="utf-8")
    replaced = sum(r["rejected"] != p["rejected"] for r, p in zip(data, pairs, strict=True))
    print(f"{OUT}: {len(data)} pairs, {replaced} with the model's own mistake as the rejected "
          f"answer, {len(data) - replaced} as written by hand")


if __name__ == "__main__":
    main()
