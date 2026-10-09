"""Tests for the live training dashboard's reading of a run's log.

It reports progress and time left while a run is going, so a misread here is a wrong promise
about when training ends. The log below is the chat fine-tune's own format.
"""

from __future__ import annotations

import importlib.util
import json
import time

import pytest

spec = importlib.util.spec_from_file_location("training_dashboard", "scripts/training_dashboard.py")
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)

LOG = """base        Qwen/Qwen3-1.7B-Base
schedule    806 optimizer steps (1x16 = 16 conversations each)
  step     0  loss  6.9781  lr 2.50e-06       678 graded tok/s
  step    90  loss  1.0009  lr 9.91e-05       677 graded tok/s
    val loss 1.3035   perplexity 3.68
    -- seen in training --
    > hi
      Hi! What can I do for you?
    -- never seen --
    > hey there, how's it going?
      Not too bad, thanks for asking.
    > မင်္ဂလာပါ
      Alright. Need anything?
  step   100  loss  1.2693  lr 9.86e-05       651 graded tok/s
"""


@pytest.fixture
def run(tmp_path, monkeypatch):
    folder = tmp_path / "chat-v2"
    folder.mkdir()
    (folder / "train.log").write_text(LOG, encoding="utf-8")
    (folder / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"kind": "train", "step": 0, "loss": 6.9781, "lr": 2.5e-06},
        {"kind": "eval", "step": 99, "loss": 1.3035},
        {"kind": "train", "step": 100, "loss": 1.2693, "lr": 9.86e-05}]), encoding="utf-8")
    monkeypatch.setattr(dashboard, "gpu", lambda: {})
    return folder


def test_progress_comes_from_the_schedule_and_the_last_step(run) -> None:
    status = dashboard.Run(run, started=time.time() - 101 * 7.0).status()
    assert (status["total"], status["done"], status["finished"]) == (806, 101, False)
    assert status["loss"] == 1.2693 and status["speed"] == "651"
    # 101 steps in 707 seconds is 7 s a step, so 705 steps to go is about 4,935 seconds.
    assert status["eta_seconds"] == pytest.approx(705 * 7.0, rel=0.01)


def test_the_newest_never_seen_answers_are_shown_whole(run) -> None:
    """Including the Burmese probe: the log is UTF-8, and the page must say so."""
    status = dashboard.Run(run, started=time.time()).status()
    assert status["sample_step"] == 100
    assert status["samples"] == [
        ("hey there, how's it going?", "Not too bad, thanks for asking."),
        ("မင်္ဂလာပါ", "Alright. Need anything?"),
    ]
    assert status["val"] == [[99, 1.3035]]


def test_a_half_written_metrics_line_is_skipped(run) -> None:
    """Training appends while the page reads; a read can land in the middle of a line."""
    with (run / "metrics.jsonl").open("a", encoding="utf-8") as metrics:
        metrics.write('\n{"kind": "train", "step": 110, "lo')
    status = dashboard.Run(run, started=time.time()).status()
    assert [step for step, _ in status["train"]] == [0, 100]


def test_a_stopped_run_shows_no_time_left(run) -> None:
    """Stopped at step 400 to resume later: the countdown must not keep running."""
    import os

    quiet = time.time() - 600
    os.utime(run / "train.log", (quiet, quiet))
    status = dashboard.Run(run, started=time.time() - 1000).status()
    assert status["stopped"] and status["eta_seconds"] is None and status["done"] == 101
    # Stopped right after the step-200 save: the log's last step line is older than that.
    (run / "latest.json").write_text(json.dumps({"dir": "step-000200", "step": 200}))
    assert dashboard.Run(run, started=time.time()).status()["done"] == 200


def test_a_finished_run_shows_no_time_left(run) -> None:
    with (run / "train.log").open("a", encoding="utf-8") as log:
        log.write("\ndone in 95.0 min -> checkpoints/chat-v2/step-000806/adapter\n")
    status = dashboard.Run(run, started=time.time()).status()
    assert status["finished"] and status["done"] == 806 and status["eta_seconds"] is None
