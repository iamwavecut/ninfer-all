#!/usr/bin/env python3
"""Test workload for the fp8-dev trial serve (default port 10999).

Usage: trial-load.py [port] [run_dir] [phase[:workers] ...]

  port     10999 by default
  run_dir  the logs/trial-<stamp> dir; by default the NEWEST one under
           <script_dir>/logs, falling back to the trial deploy dir
           (F:/gsoft/ninfer-fp8dev/logs) so the .bat runs from any copy
  phases   one or more phase[:workers] items from smoke,prefill,decode,mixed
           (default: all at 4 workers). Comma or space separated both work;
           space-separated is what the .bat uses, because cmd splits batch
           arguments on commas. Long-context requests are KV-heavy: run
           prefill at 1 worker while a live session shares the cache
           (e.g. "prefill:1").

Phases, all logged to <run_dir>/load.csv (one row per request) and printed:
  smoke    one short chat, non-streamed          (liveness + API shape)
  prefill  4 concurrent ~16k-token prompts, streamed
           (time to first token + prefill throughput)
  decode   4 concurrent short prompts, 512 max tokens
           (decode throughput)
  mixed    1 prefill + 3 decode concurrent       (realistic 4-lane production load)

Stdlib only (urllib, json, threading, pathlib). The serve speaks the OpenAI
protocol. Streaming requests ask for include_usage, so token counts come from
the serve's final usage chunk; without it the count falls back to SSE chunks
(a lower bound under speculative decoding, marked in the note column).
TTFT is client-side (request sent -> first content chunk) and includes queue
time, so compare like-for-like only; serve.log is the authority.
"""

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "qwen3.8-27b-fp8-trial"

# ~16k tokens of ASCII (roughly 4 chars/token): well past every A8 prefill threshold.
# The nonce makes every fresh run a COLD prefill (the disk KV prefix cache survives
# across runs); within one run the repeated prompts still hit the cache, which is
# what the prefill phase is partly there to measure.
LONG_PROMPT = "".join(
    "This is prefill workload paragraph %d. The quick brown fox jumps over the lazy dog "
    "while the tensor cores accumulate products in floating point. " % i
    for i in range(480)
) + " Workload nonce %s." % time.strftime("%Y%m%d-%H%M%S")  # ~65k chars

SHORT_PROMPT = "Count from one to twenty, one number per line."


def request(port, payload, timeout=600.0):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % port,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def run_one(port, tag, prompt_chars, max_tokens, stream):
    """One chat completion. Returns a result dict."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": LONG_PROMPT if prompt_chars > 1000 else SHORT_PROMPT}],
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if stream:
        # Ask the serve for a final usage chunk; without it, token counts below
        # fall back to SSE-chunk counting, which under-speculative-decoding is a
        # lower bound (one chunk may carry several accepted tokens).
        payload["stream_options"] = {"include_usage": True}
    t0 = time.monotonic()
    ttft_ms = None
    out_tokens = 0
    total = time.monotonic()
    try:
        if not stream:
            with request(port, payload) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            total = time.monotonic()
            usage = data.get("usage") or {}
            out_tokens = int(usage.get("completion_tokens", 0))
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return {"tag": tag, "prompt_chars": prompt_chars, "max_tokens": max_tokens,
                    "ttft_ms": (total - t0) * 1000.0, "total_ms": (total - t0) * 1000.0,
                    "out_tokens": out_tokens, "prompt_tokens": prompt_tokens,
                    "status": "ok", "note": text[:80].replace("\n", " ")}
        usage = None
        with request(port, payload) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except ValueError:
                    continue
                u = obj.get("usage")
                if isinstance(u, dict):
                    usage = u
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        if ttft_ms is None:
                            ttft_ms = (time.monotonic() - t0) * 1000.0
                        out_tokens += 1
            total = time.monotonic()
        if ttft_ms is None:
            ttft_ms = (total - t0) * 1000.0
        # Prefer the serve's usage chunk (authoritative); fall back to SSE-chunk
        # counting, which is a lower bound under speculative decoding.
        if usage:
            out_final = int(usage.get("completion_tokens", out_tokens))
            prompt_final = int(usage.get("prompt_tokens", prompt_chars // 4))
            note = "streamed(usage)"
        else:
            out_final = out_tokens
            prompt_final = prompt_chars // 4
            note = "streamed(chunks)"
        return {"tag": tag, "prompt_chars": prompt_chars, "max_tokens": max_tokens,
                "ttft_ms": ttft_ms, "total_ms": (total - t0) * 1000.0,
                "out_tokens": out_final, "prompt_tokens": prompt_final,
                "status": "ok", "note": note}
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as err:
        total = time.monotonic()
        return {"tag": tag, "prompt_chars": prompt_chars, "max_tokens": max_tokens,
                "ttft_ms": ttft_ms if ttft_ms is not None else 0.0,
                "total_ms": (total - t0) * 1000.0, "out_tokens": out_tokens,
                "prompt_tokens": 0, "status": "error", "note": str(err)[:120]}


def run_phase(port, tag, jobs, workers=4):
    """jobs: list of (prompt_chars, max_tokens, stream).

    At most `workers` requests in flight at once. workers >= len(jobs) starts
    everything together (the old barrier behavior); workers == 1 runs the
    jobs strictly one after another (long-context KV safety)."""
    import queue as queue_mod
    pending = queue_mod.Queue()
    for i, job in enumerate(jobs):
        pending.put((i, job))
    results = [None] * len(jobs)

    def worker():
        while True:
            try:
                i, job = pending.get_nowait()
            except queue_mod.Empty:
                return
            results[i] = run_one(port, tag, *job)

    t0 = time.monotonic()
    lanes = max(1, min(workers, len(jobs)))
    threads = [threading.Thread(target=worker) for _ in range(lanes)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_ms = (time.monotonic() - t0) * 1000.0
    return results, wall_ms


# The trial deployment dir (trial-start.bat's %~dp0 when run from F:). Fallback so a
# .bat copy run from anywhere (e.g. the source tree) still finds the live run dir.
TRIAL_DEPLOY_LOGS = Path(r"F:\gsoft\ninfer-fp8dev\logs")


def newest_run_dir():
    """The newest logs/trial-* dir (stamp names sort chronologically)."""
    for logs in (Path(__file__).resolve().parent / "logs", TRIAL_DEPLOY_LOGS):
        candidates = sorted(p for p in logs.glob("trial-*") if p.is_dir())
        if candidates:
            return candidates[-1]
    return None


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 10999
    if len(sys.argv) > 2 and sys.argv[2]:
        run_dir = Path(sys.argv[2])
    else:
        run_dir = newest_run_dir()
        if run_dir is None:
            print("No logs\\trial-* dir found (script dir or the trial deploy dir); "
                  "run trial-start.bat first.")
            return 1
    # Phase items may arrive as one comma-joined argument (direct python call) or as
    # several space-separated arguments (the .bat, since cmd splits batch args on commas).
    items = []
    for a in sys.argv[3:]:
        items.extend(part for part in a.split(",") if part)
    if not items:
        items = ["smoke", "prefill", "decode", "mixed"]
    plan = []  # (phase, workers), in the requested order
    for item in items:
        name, _, w = item.partition(":")
        workers = int(w) if w else 4
        if name not in ("smoke", "prefill", "decode", "mixed"):
            print("Unknown phase: %s (choose from smoke,prefill,decode,mixed; "
                  "optionally name:N workers)" % name)
            return 1
        plan.append((name, workers))
    if not plan:
        print("No phases selected.")
        return 1
    phases = [p for p, _ in plan]
    wmap = dict(plan)
    csv_path = str(run_dir / "load.csv")
    print("Workload dir: %s" % run_dir)
    print("Phases: %s" % ",".join("%s:%d" % (p, w) for p, w in plan))

    lines = []
    with open(csv_path, "w", encoding="utf-8") as csv:
        csv.write("phase,seq,prompt_chars,max_tokens,ttft_ms,total_ms,out_tokens,"
                  "prompt_tokens,prefill_tok_s,decode_tok_s,status,note\n")

        def record(phase, seq, r):
            prefill = (r["prompt_tokens"] / (r["ttft_ms"] / 1000.0)) if r["ttft_ms"] else 0.0
            decode_span = (r["total_ms"] - (r["ttft_ms"] or 0.0)) / 1000.0
            decode = (r["out_tokens"] / decode_span) if decode_span > 0.05 and r["out_tokens"] else 0.0
            line = "%s,%d,%d,%d,%.0f,%.0f,%d,%d,%.0f,%.0f,%s,%s" % (
                phase, seq, r["prompt_chars"], r["max_tokens"], r["ttft_ms"], r["total_ms"],
                r["out_tokens"], r["prompt_tokens"], prefill, decode, r["status"],
                '"' + r["note"].replace('"', "'") + '"')
            csv.write(line + "\n")
            csv.flush()
            print(line)
            lines.append(line)

        # --- smoke ---------------------------------------------------------------
        if "smoke" in phases:
            r = run_one(port, "smoke", len(SHORT_PROMPT), 64, False)
            record("smoke", 1, r)
            if r["status"] != "ok":
                print("SMOKE FAILED - the serve is not answering; aborting the workload.")
                print("Summary written to %s" % csv_path)
                return 1

        # --- prefill: 4 ~16k-token prompts (workers per phase; 1 = sequential) ----
        if "prefill" in phases:
            results, wall = run_phase(port, "prefill",
                                      [(len(LONG_PROMPT), 16, True)] * 4,
                                      wmap["prefill"])
            for i, r in enumerate(results):
                record("prefill", i + 1, r)
            print("prefill phase wall: %.0f ms (%d worker(s))" % (wall, wmap["prefill"]))

        # --- decode: 4 short prompts, 512 max tokens -------------------------------
        if "decode" in phases:
            results, wall = run_phase(port, "decode",
                                      [(len(SHORT_PROMPT), 512, True)] * 4,
                                      wmap["decode"])
            for i, r in enumerate(results):
                record("decode", i + 1, r)
            print("decode phase wall: %.0f ms (%d worker(s))" % (wall, wmap["decode"]))

        # --- mixed: 1 prefill + 3 decode concurrent --------------------------------
        if "mixed" in phases:
            results, wall = run_phase(port, "mixed",
                                      [(len(LONG_PROMPT), 16, True),
                                       (len(SHORT_PROMPT), 256, True),
                                       (len(SHORT_PROMPT), 256, True),
                                       (len(SHORT_PROMPT), 256, True)],
                                      wmap["mixed"])
            for i, r in enumerate(results):
                record("mixed", i + 1, r)
            print("mixed phase wall: %.0f ms (%d worker(s))" % (wall, wmap["mixed"]))

        ok = all(l.split(",")[10] == "ok" for l in lines[1:])
        print()
        print("WORKLOAD %s - results: %s" % ("DONE" if ok else "DONE WITH ERRORS", csv_path))
        return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
