#!/usr/bin/env python3
"""serve_load.py: a burst of N streaming requests against one running ninfer-serve.

usage: serve_load.py --base-url URL --out DIR --clients 1,2,4,8,16,32,64 [--model NAME]
                     [--profile chat|history|document|prefix] [--max-tokens 512] [--natural-eos]

Every level sends N requests at once (a barrier releases them together), greedy, thinking off,
streamed with usage. Each request starts with a nonce, so no prefix is shared unless the profile
asks for it. The load profiles:

  chat      a short prompt of about 60 tokens, the answer's length --max-tokens (profile A).
  history   a document of --prompt-tokens per client sent once to warm its session, then the
            same document with a question: the measured request hits that prefix (profile B).
  document  a new document of --prompt-tokens per client and a question (profile C).
  prefix    one shared system prompt of --prompt-tokens and a different short question per
            client, all cold: how much of the shared prompt the server computes again.

Answers ignore EOS (exact lengths) unless --natural-eos. One JSON row per level goes to
DIR/<profile>.jsonl:

  aggregate_tps   generated tokens over the burst's wall time.
  steady_tps      tokens generated while all N streams decode (from the latest first token to the
                  earliest last token) over that window: the decode rate of N rows, and round_ms =
                  1000 N / steady_tps, the round time t(N).
  ttft_ms, itl_ms p50/p99/max; per-request decode tok/s; finish reasons; prompt and cached
  tokens; the device memory peak (nvidia-smi); /metrics lines with "ninfer:" after the level.

The client uses the standard library only.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid

SHORT_PROMPTS = [
    "Write a detailed essay on the history of the printing press and its effect on European science.",
    "Write a complete, commented Python module implementing a thread-safe LRU cache with TTL expiry.",
    "A train leaves city A at 09:00 at 80 km/h; another leaves city B, 440 km away, at 10:00 at 100 "
    "km/h towards A. Work out step by step when and where they meet.",
    "Подробно объясни, как работает спекулятивное декодирование в больших языковых моделях.",
    "List forty distinct everyday kitchen items, one per line, each with a one-sentence description.",
    "Explain how a CPU cache hierarchy works and why memory access patterns matter for performance.",
    "Describe the water cycle to a ten-year-old, then again to a university student.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
]
QUESTIONS = [
    "Summarize the text above in five sentences.",
    "List the people and places the text above mentions.",
    "What is the main argument of the text above? Answer in one paragraph.",
    "Write three questions a reader might ask about the text above, with answers.",
    "Retell the last part of the text above in your own words.",
    "Which facts in the text above are the most surprising? Explain why.",
    "Translate the first paragraph of the text above into French.",
    "Write a title and a short abstract for the text above.",
]
CHARS_PER_TOKEN = 4.0


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return round(ordered[index], 2)


class Memory:
    """nvidia-smi sampler of one device's used memory, 0.25 s apart."""

    def __init__(self, index: int):
        self.index, self.peak, self.current, self.stop = index, 0, 0, False
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self) -> None:
        while not self.stop:
            try:
                out = subprocess.run(["nvidia-smi", "-i", str(self.index), "--query-gpu=memory.used",
                                      "--format=csv,noheader,nounits"], capture_output=True,
                                     text=True, timeout=5).stdout
                self.current = int(out.strip().splitlines()[0])
                self.peak = max(self.peak, self.current)
            except Exception:  # noqa: BLE001 - no GPU tools: no memory column
                pass
            time.sleep(0.25)

    def reset(self) -> None:
        self.peak = self.current


class Corpus:
    """Plain text from files, cut by characters (the server reports the real token counts)."""

    def __init__(self, paths: list[str]):
        parts = []
        for path in paths:
            with open(path, encoding="utf-8", errors="replace") as handle:
                parts.append(handle.read())
        self.text = "\n\n".join(parts) or ("The quick brown fox jumps over the lazy dog. " * 2000)

    def slice(self, start_tokens: int, tokens: int) -> str:
        begin = int(start_tokens * CHARS_PER_TOKEN) % max(1, len(self.text))
        length = int(tokens * CHARS_PER_TOKEN)
        out = self.text[begin:begin + length]
        while len(out) < length:
            out += "\n\n" + self.text[:length - len(out)]
        return out


def stream_chat(base: urllib.parse.ParseResult, body: dict, timeout: float) -> dict:
    """One streamed chat completion: chunk times, usage, finish reason, error."""
    row: dict = {"chunks": [], "error": None}
    connection_type = http.client.HTTPSConnection if base.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(base.hostname, base.port or (443 if base.scheme == "https" else 80),
                                 timeout=timeout)
    row["sent"] = time.perf_counter()
    try:
        connection.request("POST", (base.path.rstrip("/") or "") + "/v1/chat/completions",
                           body=json.dumps(body).encode("utf-8"),
                           headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        response = connection.getresponse()
        if response.status != 200:
            row["error"] = "HTTP %d: %s" % (response.status, response.read()[:400].decode("utf-8", "replace"))
            return row
        while True:
            line = response.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            now = time.perf_counter()
            event = json.loads(payload)
            if event.get("error"):
                row["error"] = json.dumps(event["error"])[:400]
                break
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    row["chunks"].append(now)
                if choice.get("finish_reason"):
                    row["finish"] = choice["finish_reason"]
            if event.get("usage"):
                row["usage"] = event["usage"]
            if event.get("timings"):
                row["timings"] = event["timings"]
    except Exception as exc:  # noqa: BLE001 - a failed request is a result
        row["error"] = "%s: %s" % (type(exc).__name__, exc)
    finally:
        row["ended"] = time.perf_counter()
        connection.close()
    return row


def metrics_lines(base_url: str) -> list[str]:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/metrics", timeout=10) as response:
            text = response.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return []
    return [line for line in text.splitlines() if line.startswith("ninfer:") and "_bucket" not in line]


def served_model(base_url: str) -> str:
    """The first model /v1/models lists: the one a single-model server serves."""
    with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=30) as response:
        models = json.loads(response.read().decode("utf-8"))["data"]
    if not models:
        raise SystemExit("serve_load: the server lists no model")
    return models[0]["id"]


def build_messages(profile: str, index: int, args, corpus: Corpus, shared: str) -> list[dict]:
    nonce = uuid.uuid4().hex[:12]
    if profile == "chat":
        return [{"role": "user", "content": "[%s] %s" % (nonce, SHORT_PROMPTS[index % len(SHORT_PROMPTS)])}]
    if profile == "prefix":
        return [{"role": "system", "content": shared},
                {"role": "user", "content": "[%s] %s" % (nonce, QUESTIONS[index % len(QUESTIONS)])}]
    document = "Document %d.\n\n%s" % (index, corpus.slice(index * 7919, args.prompt_tokens))
    return [{"role": "user", "content": document},
            {"role": "assistant", "content": "I have read the document."},
            {"role": "user", "content": "[%s] %s" % (nonce, QUESTIONS[index % len(QUESTIONS)])}]


def body_of(messages: list[dict], args, max_tokens: int) -> dict:
    body = {"model": args.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    if not args.natural_eos:
        body["ignore_eos"] = True
    return body


def run_burst(base, bodies: list[dict], timeout: float) -> tuple[list[dict], float, float]:
    rows: list[dict | None] = [None] * len(bodies)
    barrier = threading.Barrier(len(bodies) + 1)

    def worker(i: int) -> None:
        barrier.wait()
        rows[i] = stream_chat(base, bodies[i], timeout)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(len(bodies))]
    for thread in threads:
        thread.start()
    barrier.wait()
    started = time.perf_counter()
    for thread in threads:
        thread.join()
    return [r or {"error": "no result", "chunks": []} for r in rows], started, time.perf_counter()


def summarize(profile: str, level: int, rows: list[dict], started: float, ended: float, args,
              memory: Memory | None) -> dict:
    ok = [r for r in rows if not r.get("error")]
    generated = sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in ok)
    prompt = sum((r.get("usage") or {}).get("prompt_tokens") or 0 for r in ok)
    cached = 0
    for r in ok:
        usage = r.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        cached += details.get("cached_tokens") or (r.get("timings") or {}).get("cache_n") or 0
    wall = ended - started
    ttft = [1000 * (r["chunks"][0] - r["sent"]) for r in ok if r["chunks"]]
    itl: list[float] = []
    per_request: list[float] = []
    for r in ok:
        times = r["chunks"]
        itl.extend(1000 * (b - a) for a, b in zip(times, times[1:]))
        tokens = (r.get("usage") or {}).get("completion_tokens") or len(times)
        if len(times) > 1 and times[-1] > times[0]:
            per_request.append((tokens - 1) / (times[-1] - times[0]))
    steady_tps = round_ms = window_s = None
    streams = [r for r in ok if len(r["chunks"]) > 1]
    if len(streams) == level and level > 0:
        window_begin = max(r["chunks"][0] for r in streams)
        window_end = min(r["chunks"][-1] for r in streams)
        if window_end - window_begin > 0.5:
            tokens = 0.0
            for r in streams:
                per_chunk = ((r.get("usage") or {}).get("completion_tokens") or len(r["chunks"])) / len(r["chunks"])
                tokens += per_chunk * sum(1 for t in r["chunks"] if window_begin < t <= window_end)
            window_s = window_end - window_begin
            steady_tps = tokens / window_s
            round_ms = 1000.0 * level / steady_tps
    finishes: dict[str, int] = {}
    for r in ok:
        finishes[r.get("finish") or "none"] = finishes.get(r.get("finish") or "none", 0) + 1
    return {
        "profile": profile, "clients": level, "completed": len(ok), "errors": [r["error"] for r in rows if r.get("error")][:8],
        "error_count": sum(1 for r in rows if r.get("error")), "max_tokens": args.max_tokens,
        "natural_eos": args.natural_eos, "prompt_tokens": prompt, "cached_tokens": cached,
        "generated_tokens": generated, "wall_s": round(wall, 3),
        "aggregate_tps": round(generated / wall, 2) if wall > 0 else None,
        "steady_tps": round(steady_tps, 2) if steady_tps else None,
        "round_ms": round(round_ms, 3) if round_ms else None,
        "steady_window_s": round(window_s, 3) if window_s else None,
        "ttft_ms": {"p50": percentile(ttft, 0.5), "p99": percentile(ttft, 0.99), "max": percentile(ttft, 1.0)},
        "itl_ms": {"p50": percentile(itl, 0.5), "p99": percentile(itl, 0.99), "max": percentile(itl, 1.0)},
        "per_request_tps": {"min": percentile(per_request, 0.0), "p50": percentile(per_request, 0.5),
                            "max": percentile(per_request, 1.0),
                            "mean": round(statistics.fmean(per_request), 2) if per_request else None},
        "finish": finishes, "peak_mib": memory.peak if memory else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", help="model name (default: the first /v1/models lists)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile", choices=("chat", "history", "document", "prefix"), default="chat")
    parser.add_argument("--clients", default="1,2,4,8")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--natural-eos", action="store_true")
    parser.add_argument("--corpus", action="append", default=[], help="text file(s) for documents")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--label", default="")
    parser.add_argument("--keep-requests", action="store_true", help="also write each request's row")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if not args.model:
        args.model = served_model(args.base_url)
    base = urllib.parse.urlparse(args.base_url)
    corpus = Corpus(args.corpus)
    memory = None if args.no_memory else Memory(args.gpu_index)
    shared = "You are a careful assistant. Reference material follows.\n\n" + corpus.slice(0, args.prompt_tokens)
    out_path = os.path.join(args.out, args.profile + ".jsonl")
    for level in [int(v) for v in args.clients.split(",") if v.strip()]:
        messages = [build_messages(args.profile, i, args, corpus, shared) for i in range(level)]
        if args.profile == "history":
            # Warm each session's prefix: the same document, a different nonce in the last turn.
            warm = [m[:-1] + [{"role": "user", "content": "Reply with one word: ready."}] for m in messages]
            run_burst(base, [body_of(m, args, 4) for m in warm], args.timeout)
        if memory:
            memory.reset()
        rows, started, ended = run_burst(base, [body_of(m, args, args.max_tokens) for m in messages], args.timeout)
        row = summarize(args.profile, level, rows, started, ended, args, memory)
        row["label"] = args.label
        row["metrics"] = metrics_lines(args.base_url)
        with open(out_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if args.keep_requests:
            with open(os.path.join(args.out, "%s-c%d-requests.jsonl" % (args.profile, level)), "w",
                      encoding="utf-8") as handle:
                for r in rows:
                    times = r.get("chunks") or []
                    handle.write(json.dumps({"error": r.get("error"), "usage": r.get("usage"),
                                             "finish": r.get("finish"), "chunks": len(times),
                                             "ttft_s": (times[0] - r["sent"]) if times else None,
                                             "decode_s": (times[-1] - times[0]) if len(times) > 1 else None}) + "\n")
        print("%s [%s] %s clients=%d done=%d err=%d gen=%d wall=%.1fs aggregate=%s steady=%s round=%sms "
              "ttft_p50=%s itl_p50=%s itl_p99=%s cached=%d/%d peak=%sMiB" % (
                  time.strftime("%H:%M:%SZ", time.gmtime()), args.profile, args.label, level, row["completed"],
                  row["error_count"], row["generated_tokens"], row["wall_s"], row["aggregate_tps"],
                  row["steady_tps"], row["round_ms"], row["ttft_ms"]["p50"], row["itl_ms"]["p50"],
                  row["itl_ms"]["p99"], row["cached_tokens"], row["prompt_tokens"], row["peak_mib"]), flush=True)
        if row["error_count"]:
            print("  errors: %s" % row["errors"][:3], flush=True)
    if memory:
        memory.stop = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
