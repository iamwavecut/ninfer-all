"""Compare two CUDA Graph memory sweeps without claiming an inference speedup.

Each input is a JSON report from tools.bench.run_graph_allowance_sweep.
Observed VRAM deltas are device-wide, potentially affected by unrelated GPU
activity. This analysis cannot diagnose WDDM, latency or decode throughput.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


REQUIRED_CONFIG = ("model", "spec", "draft_tokens", "kv_dtype", "prefill_chunk")


def compare_reports(baseline: dict, candidate: dict) -> dict:
    a = baseline.get("config", {})
    b = candidate.get("config", {})
    mismatch = [name for name in REQUIRED_CONFIG if a.get(name) != b.get(name)]
    if mismatch:
        raise ValueError("incomparable sweep configurations: " + ", ".join(mismatch))
    if not baseline.get("runs") or not candidate.get("runs"):
        raise ValueError("both sweeps must contain samples")

    def grouped(report: dict) -> dict[int, list[dict]]:
        out: dict[int, list[dict]] = {}
        for sample in report["runs"]:
            out.setdefault(int(sample["capacity"]), []).append(sample)
        return out

    left, right = grouped(baseline), grouped(candidate)
    common = sorted(left.keys() & right.keys())
    if not common:
        raise ValueError("no matching KV capacities")
    rows = []
    for capacity in common:
        x, y = left[capacity], right[capacity]
        def measured(samples: list[dict]) -> list[float]:
            return [float(item["observed_bytes"]) for item in samples
                    if item.get("returncode") == 0
                    and item.get("observed_bytes") is not None
                    and item["observed_bytes"] > 0]
        a_good, b_good = measured(x), measured(y)
        enough = len(a_good) >= 2 and len(b_good) >= 2
        median_a = statistics.median(a_good) if a_good else None
        median_b = statistics.median(b_good) if b_good else None
        delta = None if median_a is None or median_b is None else median_b - median_a
        verdict = "insufficient samples"
        if enough:
            if max(b_good) < min(a_good):
                verdict = "lower observed VRAM delta (nonoverlapping sample ranges)"
            elif min(b_good) > max(a_good):
                verdict = "higher observed VRAM delta (nonoverlapping sample ranges)"
            else:
                verdict = "inconclusive: sample ranges overlap"
        rows.append({
            "capacity": capacity,
            "baseline_successes": len(a_good),
            "candidate_successes": len(b_good),
            "baseline_runs": len(x),
            "candidate_runs": len(y),
            "baseline_median_bytes": median_a,
            "candidate_median_bytes": median_b,
            "candidate_minus_baseline_bytes": delta,
            "verdict": verdict,
        })
    return {
        "scope": "CUDA Graph startup-memory observations, not throughput",
        "caveat": "Device-wide VRAM deltas; external GPU load and different graph sets can confound results.",
        "compared_configuration": {key: a.get(key) for key in REQUIRED_CONFIG},
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, help="optional JSON comparison file")
    args = parser.parse_args(argv)
    try:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        report = compare_reports(baseline, candidate)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    for row in report["rows"]:
        print(f"{row['capacity']:>9} | "
              f"{row['baseline_successes']}/{row['baseline_runs']} vs "
              f"{row['candidate_successes']}/{row['candidate_runs']} valid | "
              f"{row['verdict']}")
    print(report["caveat"])
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
