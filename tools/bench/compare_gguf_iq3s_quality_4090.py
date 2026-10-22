#!/usr/bin/env python3
"""Compare exact greedy Engine token IDs for the sm_89 GGUF experiment."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from compare_gguf_iq3s_4090 import ProcessVram, gpu_sample


PROMPTS = {
    "cuda": "Explain how GPU memory bandwidth affects token-by-token model decoding in three sentences.",
    "german": "Erkläre den Unterschied zwischen Prefill und Decode in drei kurzen Sätzen.",
    "long": ("The archive records a blue lantern beside the northern gate. "
             "Every entry distinguishes observation, inference, and uncertainty. " * 160)
            + "\nQuestion: What color was the lantern and where was it?",
}


def write_backend_comparison(summary: dict, out: Path) -> dict:
    comparisons = {}
    for name in PROMPTS:
        comparisons[name] = {}
        for arm in ("base", "candidate"):
            plain = summary[f"{name}-none"][arm]["generated_ids"]
            draft = summary[f"{name}-dflash2"][arm]["generated_ids"]
            first = next((i for i, (a, b) in enumerate(zip(plain, draft), 1) if a != b),
                         min(len(plain), len(draft)) + 1 if len(plain) != len(draft) else None)
            comparisons[name][arm] = {"equal": plain == draft,
                                       "first_different_token_1_based": first,
                                       "plain_tokens": len(plain), "dflash2_tokens": len(draft)}
    (out / "backend-comparison.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8")
    return comparisons


def run(exe: Path, model: Path, prompt_file: Path, out: Path, name: str,
        mode: str, arm: str) -> dict:
    args = [str(exe), str(model), "--messages", str(prompt_file),
            "--max-new", "64", "--max-context", "8192", "--kv-dtype", "rk4v4",
            "--greedy", "--raw-output", "--print-token-ids", "--spec", mode]
    if mode == "dflash2":
        args += ["--draft-tokens", "5", "--ngram-draft-tokens", "0"]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", errors="strict")
    vram = ProcessVram(process.pid)
    samples = []
    while process.poll() is None:
        sample = gpu_sample()
        if sample is not None:
            sample["unix_seconds"] = time.time()
            sample["process_dedicated_vram_mib"] = vram.sample_mib()
            samples.append(sample)
        time.sleep(0.25)
    stdout, stderr = process.communicate()
    vram.close()
    stem = f"{name}-{mode}-{arm}"
    (out / f"{stem}.stdout").write_text(stdout, encoding="utf-8")
    (out / f"{stem}.stderr").write_text(stderr, encoding="utf-8")
    (out / f"{stem}.telemetry.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"{stem} exit {process.returncode}")
    ids = re.search(r"^tokens\s+generated ids\s+([\d ]+)$", stderr, re.M)
    prompt_tokens = re.search(r"^summary\s+prompt tokens\s+(\d+)$", stderr, re.M)
    if not ids or not prompt_tokens:
        raise RuntimeError(f"{stem}: missing token diagnostics")
    return {"command": args, "prompt_tokens": int(prompt_tokens.group(1)),
            "generated_ids": [int(value) for value in ids.group(1).split()],
            "text": stdout}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.base, args.candidate, args.weights):
        if not path.is_absolute() or not path.is_file():
            parser.error(f"expected an existing absolute file: {path}")
    args.out.mkdir(parents=True, exist_ok=False)
    summary = {}
    for name, prompt in PROMPTS.items():
        prompt_file = args.out / f"{name}.messages.json"
        prompt_file.write_text(json.dumps([{"role": "user", "content": prompt}]), encoding="utf-8")
        for mode in ("none", "dflash2"):
            arms = ("base", "candidate") if mode == "none" else ("candidate", "base")
            results = {}
            for arm in arms:
                exe = args.base if arm == "base" else args.candidate
                results[arm] = run(exe, args.weights, prompt_file, args.out, name, mode, arm)
            equal = (results["base"]["generated_ids"] == results["candidate"]["generated_ids"]
                     and results["base"]["text"] == results["candidate"]["text"]
                     and results["base"]["prompt_tokens"] == results["candidate"]["prompt_tokens"])
            summary[f"{name}-{mode}"] = {"equal": equal, "base": results["base"],
                                         "candidate": results["candidate"]}
            print(name, mode, "equal" if equal else "DIFFERENT",
                  results["base"]["prompt_tokens"], flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_backend_comparison(summary, args.out)
    if not all(item["equal"] for item in summary.values()):
        raise RuntimeError("token or text mismatch")


if __name__ == "__main__":
    main()
