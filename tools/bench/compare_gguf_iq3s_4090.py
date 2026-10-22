#!/usr/bin/env python3
"""Paired public-Engine RTX 4090 A/B with per-process GPU telemetry.

Both arms run identical requests. Reports and telemetry are raw local artifacts;
the model is opened at the supplied absolute path and never copied.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


CASES = ("pp512", "pp2048", "tg128", "pp512+tg128", "pp2048+tg128", "pp8192+tg128")
FIELDS = (
    "prefill_tok_s_mean",
    "decode_output_tok_s_mean",
    "total_seconds_mean",
)


def case_metrics(data: dict) -> dict:
    if data.get("schema_version") not in (16, 17, 18, 19):
        raise ValueError("expected benchmark schema 16–19")
    tests = {}
    for test in data["tests"]:
        values = {key: test[key] for key in FIELDS}
        ttft = [rep["timings"].get("first_token_seconds") for rep in test["reps"]]
        # Earlier reports have no observed TTFT. Never manufacture it from phase timers.
        values["ttft_seconds_mean"] = (statistics.mean(ttft)
                                       if ttft and all(t is not None for t in ttft) else None)
        tests[test["label"]] = values
    return tests


class ProcessVram:
    """Sample WDDM dedicated bytes by PID (NVML reports N/A on this host)."""

    def __init__(self, pid: int):
        self.query = ctypes.c_void_p()
        self.counter = ctypes.c_void_p()
        self.api = ctypes.WinDLL("pdh") if os.name == "nt" else None
        if self.api:
            status = self.api.PdhOpenQueryW(None, 0, ctypes.byref(self.query))
            if not status:
                status = self.api.PdhAddEnglishCounterW(
                    self.query, ctypes.c_wchar_p(
                        f"\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage"),
                    0, ctypes.byref(self.counter))
            if status:
                self.close()

    def sample_mib(self) -> float | None:
        if not self.query or not self.counter:
            return None
        class Value(ctypes.Structure):
            _fields_ = [("status", ctypes.c_ulong), ("bytes", ctypes.c_longlong)]
        class Item(ctypes.Structure):
            _fields_ = [("name", ctypes.c_wchar_p), ("value", Value)]
        if self.api.PdhCollectQueryData(self.query):
            return None
        size, count = ctypes.c_ulong(), ctypes.c_ulong()
        self.api.PdhGetFormattedCounterArrayW(
            self.counter, 0x400, ctypes.byref(size), ctypes.byref(count), None)
        if not size.value:
            return None
        buffer = ctypes.create_string_buffer(size.value)
        if self.api.PdhGetFormattedCounterArrayW(
                self.counter, 0x400, ctypes.byref(size), ctypes.byref(count), buffer):
            return None
        items = ctypes.cast(buffer, ctypes.POINTER(Item))
        values = [items[i].value.bytes for i in range(count.value)
                  if items[i].value.status in (0, 1)]
        return sum(values) / 2**20 if values else None

    def close(self):
        if self.query:
            self.api.PdhCloseQuery(self.query)
        self.query = ctypes.c_void_p()
        self.counter = ctypes.c_void_p()


def gpu_sample() -> dict[str, float] | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.gr,temperature.gpu,power.draw,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        return None
    try:
        clock, temperature, power, memory = (
            float(part.strip()) for part in result.stdout.splitlines()[0].split(",")
        )
    except (IndexError, ValueError):
        return None
    return {"clock_mhz": clock, "temperature_c": temperature,
            "power_w": power, "total_vram_mib": memory}


def run_one(exe: Path, weights: Path, out: Path, mode: str, pair: int,
            arm: str, warmup: int = 1) -> dict:
    stem = f"{mode}-{pair}-{arm}"
    report = out / f"{stem}.json"
    command = [str(exe), "--weights", str(weights), "-p", "512,2048", "-n", "128",
               "-pg", "512,128;2048,128;8192,128", "-r", "2", "--warmup", str(warmup),
               "--kv-dtype", "rk4v4", "--max-ctx", "16384", "-o", "json",
               "--output-file", str(report)]
    if mode == "dflash2":
        command += ["--spec", "dflash2", "--draft-tokens", "5",
                    "--ngram-draft-tokens", "0"]
    start = time.time()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", errors="replace")
    vram = ProcessVram(process.pid)
    samples = []
    while process.poll() is None:
        sample = gpu_sample()
        if sample is not None:
            sample["process_dedicated_vram_mib"] = vram.sample_mib()
            sample["unix_seconds"] = time.time()
            samples.append(sample)
        time.sleep(0.25)
    stdout, stderr = process.communicate()
    vram.close()
    (out / f"{stem}.stdout").write_text(stdout, encoding="utf-8")
    (out / f"{stem}.stderr").write_text(stderr, encoding="utf-8")
    (out / f"{stem}.telemetry.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"{stem} failed with exit {process.returncode}; see stderr")
    data = json.loads(report.read_text(encoding="utf-8"))
    if data["environment"]["gpu_name"] != "NVIDIA GeForce RTX 4090":
        raise RuntimeError(f"{stem}: unexpected GPU")
    if [test["label"] for test in data["tests"]] != list(CASES):
        raise RuntimeError(f"{stem}: unexpected workload labels")
    if any(len(test["reps"]) != 2 for test in data["tests"]):
        raise RuntimeError(f"{stem}: incomplete repetitions")
    for test in data["tests"]:
        expected_tokens = 129 if "tg128" in test["label"] else 1
        if any(rep["generated_output_tokens"] != expected_tokens for rep in test["reps"]):
            raise RuntimeError(f"{stem}: incomplete output in {test['label']}")
    expected_backend = "none" if mode == "plain" else "dflash2"
    if data["config"]["speculative_backend"] != expected_backend:
        raise RuntimeError(f"{stem}: unexpected speculative backend")
    tests = case_metrics(data)
    measured = {
        "mode": mode, "pair": pair, "arm": arm, "command": command,
        "start_unix_seconds": start, "elapsed_seconds": time.time() - start,
        "gpu_samples": len(samples),
        "gpu_clock_mhz_median": statistics.median(s["clock_mhz"] for s in samples) if samples else None,
        "gpu_clock_mhz_range": [min(s["clock_mhz"] for s in samples),
                                max(s["clock_mhz"] for s in samples)] if samples else None,
        "temperature_c_median": statistics.median(s["temperature_c"] for s in samples) if samples else None,
        "temperature_c_range": [min(s["temperature_c"] for s in samples),
                                max(s["temperature_c"] for s in samples)] if samples else None,
        "power_w_median": statistics.median(s["power_w"] for s in samples) if samples else None,
        "power_w_peak": max(s["power_w"] for s in samples) if samples else None,
        "total_vram_mib_peak": max(s["total_vram_mib"] for s in samples) if samples else None,
        "process_dedicated_vram_mib_peak": max(
            (s["process_dedicated_vram_mib"] for s in samples
             if s["process_dedicated_vram_mib"] is not None), default=None),
        "tests": tests,
    }
    (out / f"{stem}.process.json").write_text(json.dumps(measured, indent=2), encoding="utf-8")
    return measured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=6)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("pairs must be positive; use at least six for initial qualification")
    for path in (args.base, args.candidate, args.weights):
        if not path.is_absolute() or not path.is_file():
            parser.error(f"expected an existing absolute file: {path}")
    if args.base.resolve() == args.candidate.resolve():
        parser.error("the two executables must be distinct")
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {"base": str(args.base.resolve()), "candidate": str(args.candidate.resolve()),
                "weights": str(args.weights.resolve()), "pairs": args.pairs,
                "order": "AB/BA per pair, reversed again for DFlash2",
                "cases": list(CASES), "modes": ["plain", "dflash2"]}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for mode in ("plain", "dflash2"):
        for pair in range(1, args.pairs + 1):
            order = ("base", "candidate") if (pair + (mode == "dflash2")) % 2 else ("candidate", "base")
            for arm in order:
                path = args.base if arm == "base" else args.candidate
                result = run_one(path, args.weights, args.out, mode, pair, arm)
                print(mode, pair, arm, "done", result["gpu_clock_mhz_median"],
                      result["temperature_c_median"], result["power_w_median"],
                      result["total_vram_mib_peak"], flush=True)


if __name__ == "__main__":
    main()
