"""A live view of a chat training run, in the browser: progress, time left, loss, GPU, answers.

    python scripts/training_dashboard.py --run checkpoints/chat-v2

Serves http://localhost:8765 and updates every five seconds. It only reads -- the run's
train.log and metrics.jsonl, and nvidia-smi -- so closing it, or it crashing, cannot touch the
training. The time left is the average pace since the training process started, pauses for
validation and saving included, so it settles within a few minutes of opening.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCHEDULE = re.compile(r"schedule\s+([\d,]+) optimizer steps")
STEP = re.compile(r"^\s+step\s+(\d+)\s+loss\s+([\d.]+)\s+lr\s+(\S+)\s+([\d,]+) graded tok/s")


def training_started() -> float | None:
    """When the chat training process started, as a Unix time; None when none is running."""
    command = ("$p = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object "
               "{ $_.CommandLine -like '*llm.chat_sft*' } | Sort-Object CreationDate | "
               "Select-Object -First 1; if ($p) { [DateTimeOffset]::new($p.CreationDate)"
               ".ToUnixTimeSeconds() }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                         capture_output=True, text=True).stdout.strip()
    return float(out) if out else None


def gpu() -> dict:
    query = "temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        temp, util, used, total, power = (float(v) for v in out.split(","))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}
    return {"temp": temp, "util": util, "used_gib": used / 1024, "total_gib": total / 1024,
            "power": power}


def last_samples(log: list[str]) -> tuple[int | None, list[tuple[str, str]]]:
    """The newest block of never-seen probe answers, and the step it was printed after."""
    starts = [i for i, line in enumerate(log) if line.strip() == "-- never seen --"]
    if not starts:
        return None, []
    pairs, i = [], starts[-1] + 1
    while i + 1 < len(log) and log[i].startswith("    > "):
        pairs.append((log[i][6:].strip(), log[i + 1].strip()))
        i += 2
    # The log prints every tenth step, and samples come at the end of a ten-step block.
    step = next((int(m.group(1)) + 10 for line in reversed(log[: starts[-1]])
                 if (m := STEP.match(line))), None)
    return step, pairs


def read_metrics(path: Path) -> list[dict]:
    """The run's metric records. Training appends to this file while the page reads it, so the
    last line can be half-written: it is skipped, and arrives whole five seconds later."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


class Run:
    def __init__(self, folder: Path, started: float | None = None) -> None:
        self.folder = folder
        self.started = started if started is not None else training_started()

    def status(self) -> dict:
        text = (self.folder / "train.log").read_text(encoding="utf-8", errors="replace")
        log = text.splitlines()
        total = next((int(m.group(1).replace(",", "")) for line in log
                      if (m := SCHEDULE.search(line))), None)
        steps = [m for line in log if (m := STEP.match(line))]
        metrics = read_metrics(self.folder / "metrics.jsonl")
        done = int(steps[-1].group(1)) + 1 if steps else 0
        finished = "done in" in text
        # A running log changes every ten steps, about a minute and a half; validation adds a
        # minute more. Four quiet minutes means stopped -- and a resume is a new process, whose
        # start the time-left estimate must be measured from.
        stopped = not finished and time.time() - (self.folder / "train.log").stat().st_mtime > 240
        if stopped:
            self.started = None
        elif self.started is None and not finished:
            self.started = training_started()
        eta = None
        if self.started and done and total and not finished and not stopped:
            per_step = (time.time() - self.started) / done
            eta = (total - done) * per_step
        sample_step, samples = last_samples(log)
        return {
            "run": self.folder.name, "total": total, "done": total if finished else done,
            "finished": finished, "stopped": stopped, "eta_seconds": eta,
            "eta_clock": datetime.fromtimestamp(time.time() + eta).strftime("%H:%M") if eta else None,
            "elapsed_seconds": time.time() - self.started if self.started else None,
            "loss": float(steps[-1].group(2)) if steps else None,
            "lr": steps[-1].group(3) if steps else None,
            "speed": steps[-1].group(4) if steps else None,
            "train": [[m["step"], m["loss"]] for m in metrics if m.get("kind") == "train"],
            "val": [[m["step"], m["loss"]] for m in metrics if m.get("kind") == "eval"],
            "gpu": gpu(), "sample_step": sample_step, "samples": samples,
        }


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Training dashboard</title>
<style>
:root { --bg:#f7f7f5; --card:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#e4e4df;
        --accent:#2f6fde; --val:#d9822b; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161615; --card:#21211f; --ink:#ececea; --muted:#9a9a94; --line:#33332f; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 20px 16px 40px; }
h1 { font-size: 20px; margin: 0 0 2px; } .sub { color: var(--muted); margin-bottom: 16px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
        padding: 16px; margin-bottom: 14px; }
.bar { height: 14px; background: var(--line); border-radius: 7px; overflow: hidden; }
.fill { height: 100%; background: var(--accent); width: 0; transition: width .6s; }
.big { font-size: 30px; font-weight: 650; font-variant-numeric: tabular-nums; }
.row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
.k { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.v { font-size: 19px; font-weight: 600; font-variant-numeric: tabular-nums; }
svg { width: 100%; height: 240px; display: block; }
.legend span { margin-right: 14px; color: var(--muted); font-size: 13px; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; }
.q { color: var(--muted); margin-top: 10px; } .a { margin: 2px 0 0 0; }
footer { color: var(--muted); font-size: 12px; text-align: center; }
</style></head><body><main>
<h1 id="title">Training</h1><div class="sub" id="sub">connecting…</div>
<div class="card">
  <div class="row" style="align-items:end; margin-bottom:10px">
    <div><div class="k">Steps</div><div class="big" id="steps">–</div></div>
    <div><div class="k">Steps left</div><div class="big" id="left">–</div></div>
    <div><div class="k">Time left</div><div class="big" id="eta">–</div></div>
  </div>
  <div class="bar"><div class="fill" id="fill"></div></div>
</div>
<div class="card row">
  <div><div class="k">Training loss</div><div class="v" id="loss">–</div></div>
  <div><div class="k">Validation loss</div><div class="v" id="val">–</div></div>
  <div><div class="k">Learning rate</div><div class="v" id="lr">–</div></div>
  <div><div class="k">Speed</div><div class="v" id="speed">–</div></div>
  <div><div class="k">GPU</div><div class="v" id="gpu">–</div></div>
  <div><div class="k">GPU memory</div><div class="v" id="mem">–</div></div>
</div>
<div class="card">
  <div class="legend"><span><i class="dot" style="background:var(--accent)"></i>training loss
  (every 10 steps)</span><span><i class="dot" style="background:var(--val)"></i>validation loss
  (every 100)</span></div>
  <svg id="chart" viewBox="0 0 900 240" preserveAspectRatio="none"></svg>
</div>
<div class="card"><div class="k" id="samples-title">Latest answers to prompts it never saw</div>
  <div id="samples"></div></div>
<footer>Updates every 5 seconds. This page only reads the log; closing it does not stop training.
</footer>
</main><script>
const $ = id => document.getElementById(id);
const fmt = s => { if (s == null) return "–"; s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60);
  return h ? `${h} h ${m} min` : `${m} min`; };
function chart(train, val, total) {
  const W = 900, H = 240, P = 30, all = train.concat(val).map(p => p[1]);
  if (!all.length) { $("chart").innerHTML = ""; return; }
  const capped = all.filter(v => v < 3);  // the first steps start near 7; keep the useful range
  const lo = Math.min(...capped) - 0.05, hi = Math.max(...capped, lo + 0.1) + 0.05;
  const x = s => P + (W - 2 * P) * s / Math.max(total || 1, 1);
  const y = v => H - P - (H - 2 * P) * (Math.min(v, hi) - lo) / (hi - lo);
  let g = "";
  for (let i = 0; i <= 4; i++) { const v = lo + (hi - lo) * i / 4, yy = y(v);
    g += `<line x1="${P}" x2="${W - P}" y1="${yy}" y2="${yy}" stroke="var(--line)"/>` +
         `<text x="2" y="${yy + 4}" font-size="11" fill="var(--muted)">${v.toFixed(2)}</text>`; }
  const line = train.map(p => `${x(p[0])},${y(p[1])}`).join(" ");
  g += `<polyline points="${line}" fill="none" stroke="var(--accent)" stroke-width="2"/>`;
  g += val.map(p => `<circle cx="${x(p[0])}" cy="${y(p[1])}" r="5" fill="var(--val)"/>`).join("");
  $("chart").innerHTML = g;
}
async function tick() {
  try {
    const s = await (await fetch("/status")).json();
    $("title").textContent = `Training ${s.run}`;
    $("sub").textContent = s.finished ? "Finished." : s.stopped ?
      `Stopped at step ${s.done}. resume.bat continues from the last checkpoint.` :
      `Running for ${fmt(s.elapsed_seconds)}` + (s.eta_clock ? ` · expected to finish around ${s.eta_clock}` : "");
    $("steps").textContent = s.total ? `${s.done} / ${s.total}` : s.done;
    $("left").textContent = s.total ? Math.max(s.total - s.done, 0) : "–";
    $("eta").textContent = s.finished ? "done" : s.stopped ? "stopped" : fmt(s.eta_seconds);
    $("fill").style.width = s.total ? `${100 * s.done / s.total}%` : "0";
    $("loss").textContent = s.loss != null ? s.loss.toFixed(4) : "–";
    $("val").textContent = s.val.length ? s.val[s.val.length - 1][1].toFixed(4) : "–";
    $("lr").textContent = s.lr || "–";
    $("speed").textContent = s.speed ? `${s.speed} tok/s` : "–";
    $("gpu").textContent = s.gpu.temp != null ? `${s.gpu.temp}°C · ${s.gpu.util}%` : "–";
    $("mem").textContent = s.gpu.used_gib != null ?
      `${s.gpu.used_gib.toFixed(1)} / ${s.gpu.total_gib.toFixed(1)} GiB` : "–";
    chart(s.train, s.val, s.total);
    $("samples-title").textContent = s.sample_step != null ?
      `Latest answers to prompts it never saw (at step ${s.sample_step})` :
      "Answers appear after step 100";
    $("samples").innerHTML = s.samples.map(([q, a]) =>
      `<div class="q">› ${q.replace(/</g, "&lt;")}</div><div class="a">${a.replace(/</g, "&lt;") || "<i>(empty)</i>"}</div>`).join("");
  } catch (e) { $("sub").textContent = "waiting for the dashboard…"; }
}
tick(); setInterval(tick, 5000);
</script></body></html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="checkpoints/chat-v2")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    run = Run(Path(args.run))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/status":
                body = json.dumps(run.status()).encode("utf-8")
                kind = "application/json"
            else:
                body, kind = PAGE.encode("utf-8"), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:   # one line per request every 5 s is just noise
            pass

    print(f"Training dashboard for {args.run}: http://localhost:{args.port}  (close this window "
          f"to stop the dashboard; training is not affected)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
