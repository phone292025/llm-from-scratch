"""Tests for the knowledge check's grading -- the measure every chat model here is judged by.

The rules are keyword checks, so they can be fooled. The cases worth pinning are the ones the
first rules let through: each reply below is a real answer from a real run (notes/09 section 8).
Graded strictly, the DPO run reported as the best, 23 of 27, tied the model it started from.
"""

from __future__ import annotations

import importlib.util

import pytest

spec = importlib.util.spec_from_file_location("knowledge_check", "scripts/knowledge_check.py")
quiz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quiz)


def rule_for(question: str) -> dict:
    return next(rule for _, turns, rule in quiz.TESTS if turns[-1].startswith(question))


@pytest.mark.parametrize("question, reply, passes", [
    ("Who made you?", "I'm built on Qwen3-1.7B, from Alibaba's Qwen team, and fine-tuned in a "
     "personal project.", True),
    ("Who made you?", "I was trained by a large language model called GPT-3, which was developed "
     "by Open Assistant, a company that was acquired by Microsoft.", False),
    ("Who made you?", "I was trained by a group of humans, but I was not created by any one "
     "person.", False),
    ("Are you able to look things up", "No, I can't look anything up. I only know what I "
     "learned in training.", True),
    ("Are you able to look things up", "I can look up information on the internet, but I can't "
     "access the internet.", False),
    ("Are you able to look things up", "Yes, I can look up information on the internet for "
     "you. However, I should note that I am not able to access the internet.", False),
    ("Tell me about Zorblax", "I don't know of such a person, and I'd rather not guess.", True),
    ("Tell me about Zorblax", "Zorblax Quentin was a fictional character created by the author "
     "of the novel The Phantom of the Opera.", False),
    ("I have 3 apples", "You have 3 apples, buy 5 more, so you have 8 apples. Then you eat 2, so "
     "you have 6 apples left.", True),
    ("I have 3 apples", "6.", True),
    ("I have 3 apples", "7. You had 3, bought 5, so 8, then ate 2, so 6.", False),
    ("How many usable host addresses", "There are 32 - 2 = 30 usable host addresses.", True),
    ("How many usable host addresses", "A /27 subnet has 32 bits available for host addresses. "
     "This means that there are 30 bits available for host addresses.", False),
    ("Who won the 2034", "I can't know that. What I learned stops at a point in the past, and I "
     "can't check the news.", True),
    ("Which is heavier", "Neither. They both weigh the same, 1 kilogram.", True),
    ("Which is heavier", "Neither. They both weigh the same, 1 kilogram. The feathers are "
     "lighter, though, because they have less mass.", False),
])
def test_answers_the_first_rules_let_through_now_fail(question, reply, passes) -> None:
    assert quiz.check(reply, rule_for(question))[0] is passes


LOG = """LLM knowledge check   2026-10-09 18:09:59
Loading LLM...
ready in 13s on the GPU -- checkpoint checkpoints/chat

[ 1/27] General knowledge
  you › What is the capital of Japan?
  LLM › Tokyo.
        ✓ mentions 'tokyo'   1.2s · 4 tokens · 3 tok/s

[ 2/27] Memory across turns
  you › first turn
  LLM › an earlier reply, not the one graded
        0.9s · 24 tokens · 30 tok/s
  you › second turn
  LLM › line one

line three, after a blank
        ✗ none of: x   1.7s · 51 tokens · 29 tok/s

Score by area (keyword checks -- read the answers before trusting these)
  1/2  passed
"""


def test_a_saved_log_gives_back_each_final_reply() -> None:
    """--rescore grades old logs with the current rules, so the parse must recover the graded
    reply exactly: the last one of its question, blank lines kept, nothing of the verdict."""
    assert quiz.parse_log(LOG) == {1: "Tokyo.", 2: "line one\n\nline three, after a blank"}
