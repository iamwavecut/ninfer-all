"""Measure NInfer 4090 decode modes without hard-coding a supposed fastest configuration.

Runs the public Engine benchmark for matched prompt/output lengths, in rotating
candidate order to reduce monotonic GPU clock/temperature bias. Nothing is changed
in the model, device-profile cache or runtime configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Sequence


@dataclass(frozen=True)
class Candidate:
    kv: str
    backend: str = "none"
    drafts: int = 0

    @property
    def slug(self) -> str:
        return f"{self.kv}-{self.backend}{self.drafts if self.drafts else ''}"

    def flags(self) -> list[str]:
        args = ["--kv-dtype", self.kv]
        if self.backend != "none":
            args += ["--spec", self.backend, "--draft-tokens", str(self.drafts),
                     "--ngram-draft-tokens", "0"]
        return args


def candidates(preset: str, backends: Sequence[str] = ()) -> list[Candidate]:
    kv_types = ("rk4v4",) if preset == "quick" else ("rk4v4", "rk8v4")
    modes = [("none", 0)]
    if "mtp" in backends:
        modes += [("mtp", 3), ("mtp", 5)]
    if "dflash2" in backends:
        modes += [("dflash2", 3), ("dflash2", 5), ("dflash2", 7)]
    return [Candidate(kv, backend, drafts)
            for kv in kv_types for backend, drafts in modes]


def workloads(preset: str) -> list[tuple[int, int]]:
    return [(2048, 128)] if preset == "quick" else [(2048, 128), (32768, 128)]


def valid_speed(report: dict, expected_prompt: int, expected_decode: int) -> float:
    if (report.get("tool") != "ninfer_bench" or
            report.get("artifact_type") != "ninfer_bench_report" or
            report.get("schema_version") not in (16, 17, 18)):
        raise ValueError("unexpected benchmark schema")
    gpu = str(report.get("environment", {}).get("gpu_name", ""))
    if "RTX 4090" not in gpu:
        raise ValueError(f"benchmark ran on wrong GPU: {gpu!r}")
    tests = report.get("tests", [])
    if len(tests) != 1:
        raise ValueError(f"expected one benchmark case, got {len(tests)}")
    test = tests[0]
    if (int(test.get("n_prompt", -1)) != expected_prompt or
            int(test.get("n_gen", -1)) != expected_decode):
        raise ValueError("benchmark returned a different prompt/decode length")
    speed = float(test.get("decode_output_tok_s_mean", 0))
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("missing or invalid decode throughput")
    return speed


def make_command(bench: Path, model: Path, corpus: Path, device: int,
                 candidate: Candidate, prompt: int, decode: int,
                 result: Path) -> list[str]:
    return [
        str(bench), "--weights", str(model), "--corpus", str(corpus),
        "--device", str(device), "-pg", f"{prompt},{decode}",
        "--max-ctx", str(max(32768, prompt + decode + 256)),
        "--prefill-chunk", "1024",
        "--warmup", "1", "-r", "1",
        *candidate.flags(),
        "--output", "json", "--output-file", str(result),
    ]


def select_winner(readings: dict[str, list[float]], modes: Sequence[Candidate],
                  rounds: int, minimum_gain: float = 0.03) -> dict:
    """Require all rounds and >3% over plain decode; else preserve the baseline."""
    baseline = modes[0]
    base_runs = readings.get(baseline.slug, [])
    if len(base_runs) != rounds:
        return {"selected": None, "reason": "incomplete baseline"}
    base_rate = statistics.median(base_runs)
    qualified = [
        (statistics.median(v), m) for m in modes
        if len(v := readings.get(m.slug, [])) == rounds
    ]
    if not qualified:
        return {"selected": None, "reason": "no complete candidates"}
    best_rate, best = max(qualified, key=lambda entry: entry[0])
    if best_rate < base_rate * (1.0 + minimum_gain):
        best_rate, best = base_rate, baseline
    return {
        "selected": best.slug,
        "flags": best.flags(),
        "decode_tok_s_median": best_rate,
        "baseline_tok_s_median": base_rate,
        "relative_gain_percent": round(100 * (best_rate / base_rate - 1), 2),
        "note": "throughput only; validate long-context retrieval and numerical quality before deployment",
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--bench", type=Path, required=True)
    p.add_argument("--corpus", type=Path, default=Path("bench/fixtures/bench_corpus.ids"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--preset", choices=("quick", "full"), default="quick")
    p.add_argument("--backends", nargs="*", choices=("mtp", "dflash2"), default=[],
                   help="only draft components confirmed in the artifact; omitted means plain decode")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    opts = parser().parse_args(argv)
    if opts.rounds < 3 or opts.rounds > 20:
        raise SystemExit("--rounds must be 3..20 for a repeatable comparison")
    if opts.device < 0 or opts.timeout <= 0:
        raise SystemExit("invalid device or timeout")
    for path in (opts.weights, opts.bench, opts.corpus):
        if not path.is_file():
            raise SystemExit(f"required file not found: {path}")
    if opts.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {opts.out}")
    opts.out.mkdir(parents=True)
    matrix = candidates(opts.preset, opts.backends)
    plan = {
        "artifact_type": "ninfer_rtx4090_tuning",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "weights": str(opts.weights.resolve()),
        "bench": str(opts.bench.resolve()),
        "corpus": str(opts.corpus.resolve()),
        "device": opts.device,
        "preset": opts.preset,
        "rounds": opts.rounds,
        "workloads": workloads(opts.preset),
        "candidates": [asdict(m) for m in matrix],
        "method": "rotating order, one warmup and one measured request per process and round",
        "quality_checked": False,
    }
    (opts.out / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    selections: dict[str, dict] = {}
    samples: list[dict] = []
    for prompt, decode in workloads(opts.preset):
        workload = f"p{prompt}-g{decode}"
        readings: dict[str, list[float]] = {}
        for round_index in range(opts.rounds):
            # Rotate and reverse, so candidates do not always run at the same heat/clock stage.
            order = matrix[round_index % len(matrix):] + matrix[:round_index % len(matrix)]
            if round_index % 2:
                order.reverse()
            for candidate in order:
                stem = f"{workload}-{candidate.slug}-r{round_index+1}"
                result = opts.out / (stem + ".json")
                cmd = make_command(opts.bench, opts.weights, opts.corpus, opts.device,
                                   candidate, prompt, decode, result)
                if opts.dry_run:
                    samples.append({"case": stem, "command": cmd, "dry_run": True})
                    continue
                try:
                    run = subprocess.run(cmd, capture_output=True, text=True,
                                         timeout=opts.timeout, check=False)
                    (opts.out / (stem + ".stdout.txt")).write_text(run.stdout, encoding="utf-8")
                    (opts.out / (stem + ".stderr.txt")).write_text(run.stderr, encoding="utf-8")
                    if run.returncode != 0:
                        raise RuntimeError(f"ninfer_bench exit {run.returncode}")
                    report = json.loads(result.read_text(encoding="utf-8"))
                    speed = valid_speed(report, prompt, decode)
                    readings.setdefault(candidate.slug, []).append(speed)
                    samples.append({"case": stem, "mode": candidate.slug,
                                    "decode_tok_s": speed, "status": "ok"})
                    print(f"{stem}: {speed:.1f} tok/s", flush=True)
                except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired,
                        json.JSONDecodeError) as exc:
                    samples.append({"case": stem, "mode": candidate.slug,
                                    "status": "failed", "error": str(exc)})
                    print(f"{stem}: {exc}", file=sys.stderr, flush=True)
                finally:
                    (opts.out / "samples.json").write_text(
                        json.dumps(samples, indent=2) + "\n", encoding="utf-8")
        if not opts.dry_run:
            selections[workload] = select_winner(readings, matrix, opts.rounds)
            (opts.out / "recommendations.json").write_text(
                json.dumps(selections, indent=2) + "\n", encoding="utf-8")
    if opts.dry_run:
        print(f"dry-run plan: {len(samples)} runs -> {opts.out}")
        return 0
    print(json.dumps(selections, indent=2))
    return 0 if all(value["selected"] is not None for value in selections.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
