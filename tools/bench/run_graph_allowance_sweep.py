"""Measure CUDA Graph startup-memory deltas with repeated, matched model runs.

Reports are device-wide CUDA allocation observations, not VRAM exclusively
owned by this process; changes in WDDM/desktop GPU activity can dominate.
The historic ordinary graph count estimate is a legacy hint and is omitted
for speculative MTP and DFlash2; it is NOT a current topology guarantee.

Examples (from the repository root):

    python -m tools.bench.run_graph_allowance_sweep --cli PATH_TO_NINFER \\
        --model MODEL.ninfer --spec dflash2 --kv-dtype rk4v4 \\
        --capacities 8192 16384 32768 --repeats 5 --output profiles/baseline

See docs/performance/upstream-backport-review-2026-10.md for the A/B contract.
Missing or unparseable memory instrumentation is reported as incomplete.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLI = ROOT / "build/apps/ninfer"
DEFAULT_MODEL = Path.home() / "ai/models/qwen3_8_27b.ninfer"

MIB = 1024 * 1024

# Mirrors Variant::ordinary_graph_profiles in src/targets/qwen3_6_27b/impl/variant.cpp. Kept here so
# the report can state the predicted graph count next to the measured cost; if the source table
# changes this must change with it, which is why the values are quoted rather than derived.
ORDINARY_PROFILE_ENDS = (127, 511, 2047, 4095, 8197, 16389, 32767)

# Capacities either side of every profile boundary, plus the four points already measured by hand.
# Boundaries are where the graph count steps, so they are where a count-driven model and a
# frontier-driven model disagree most.
DEFAULT_CAPACITIES = (
    2048, 4096, 8198, 16390,
    32768, 32832, 40960, 49152, 65536, 81920, 103424, 131072,
)

# Editable sweep configuration. Keep these values explicit so saved benchmark artifacts identify
# the exact workload without depending on shell history.
CLI = DEFAULT_CLI
MODEL = DEFAULT_MODEL
CAPACITIES = DEFAULT_CAPACITIES
REPEATS = 3
SPEC: str | None = None
DRAFT_TOKENS = 3
KV_DTYPE = "int8"
PREFILL_CHUNK = 640
OUT = ROOT / "out/graph_allowance"

SUMMARY = re.compile(r"^\s*([a-zA-Z][a-zA-Z0-9 _/-]*?)\s{2,}(.+?)\s*$")
# A bare "B" unit is not a formatting curiosity: the engine clamps a negative device-wide delta to
# zero, so "0 B" is the signature of another process freeing VRAM during capture. It must parse.
GRAPH_LINE = re.compile(r"CUDA Graph memory\s+([\d.]+)\s*([KMG]?i?B)\s*/\s*([\d.]+)\s*([KMG]?i?B)")
GRAPH_ERROR = re.compile(
    r"CUDA Graph preparation consumed (\d+) bytes, exceeding the planned allowance of (\d+) bytes"
)
UNITS = {"B": 1, "KiB": 1024, "MiB": MIB, "GiB": 1024 * MIB}


def ordinary_graph_count(capacity: int, concurrency: int = 1) -> int:
    """Number of ordinary decode graphs captured for a capacity.

    Mirrors graph_profiles_through(capacity - 1, ORDINARY_PROFILE_ENDS) exactly, including its
    early return when a range lands on the frontier.
    """
    max_frontier = capacity - 1
    profiles = 0
    begin = 0
    for preferred_end in ORDINARY_PROFILE_ENDS:
        if begin > max_frontier:
            break
        end = min(preferred_end, max_frontier)
        profiles += 1
        if end == max_frontier:
            return profiles * concurrency
        begin = end + 1
    if begin <= max_frontier:
        profiles += 1
    return profiles * concurrency


def parse_bytes(value: str, unit: str) -> int:
    return int(round(float(value) * UNITS[unit]))


def measure(cli: Path, model: Path, capacity: int, spec: str | None, draft_tokens: int,
            kv_dtype: str, prefill_chunk: int) -> dict:
    command = [
        str(cli), str(model),
        "--prompt", "hi",
        "--max-context", str(capacity),
        "--kv-capacity", str(capacity),
        "--kv-dtype", kv_dtype,
        "--prefill-chunk", str(prefill_chunk),
        "--max-new", "1",
        "--greedy", "--no-thinking",
    ]
    if spec:
        command += ["--spec", spec, "--draft-tokens", str(draft_tokens), "--lm-head-draft"]

    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    wall = time.perf_counter() - started

    observed = allowance = None
    # A run that aborts still reports both figures, in exact bytes rather than the summary's
    # two-decimal MiB, so failures are the more precise data points.
    # Windows CLI and different logging backends may write the diagnostic to stdout.
    combined = result.stderr + "\n" + result.stdout
    error = GRAPH_ERROR.search(combined)
    if error:
        observed, allowance = int(error.group(1)), int(error.group(2))
    else:
        line = GRAPH_LINE.search(combined)
        if line:
            observed = parse_bytes(line.group(1), line.group(2))
            allowance = parse_bytes(line.group(3), line.group(4))

    return {
        "capacity": capacity,
        "returncode": result.returncode,
        "started": result.returncode == 0,
        "graph_blocked": error is not None,
        "observed_bytes": observed,
        "allowance_bytes": allowance,
        # This table only models ordinary graphs, not MTP/DFlash2 variants.
        "predicted_graphs": ordinary_graph_count(capacity) if spec is None else None,
        "wall_s": round(wall, 2),
        "stderr_tail": result.stderr.strip().splitlines()[-1] if result.returncode else "",
        # A run that neither reports a figure nor aborts is the one case this sweep cannot
        # interpret, so keep its output rather than recording a silent None.
        "unparsed_output": None if observed is not None else combined[-4000:],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, default=CLI, help="ninfer CLI executable")
    parser.add_argument("--model", type=Path, default=MODEL, help="read-only .ninfer artifact")
    parser.add_argument("--capacities", nargs="+", type=int, default=list(CAPACITIES))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--spec", choices=("none", "mtp", "dflash2"), default=SPEC or "none")
    parser.add_argument("--draft-tokens", type=int, default=DRAFT_TOKENS)
    parser.add_argument("--kv-dtype", default=KV_DTYPE)
    parser.add_argument("--prefill-chunk", type=int, default=PREFILL_CHUNK)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    spec = None if args.spec == "none" else args.spec
    if args.repeats < 2:
        parser.error("--repeats must be >= 2 to expose device-wide VRAM noise")
    if not args.capacities or any(v < 1 for v in args.capacities):
        parser.error("--capacities must contain positive integers")
    if args.draft_tokens < 1 or args.prefill_chunk < 1:
        parser.error("--draft-tokens and --prefill-chunk must be positive")
    if not args.cli.is_file():
        parser.error(f"engine binary not found: {args.cli}")
    if not args.model.is_file():
        parser.error(f"model artifact not found: {args.model}")

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    print(f"{'capacity':>9}  {'graphs':>6}  {'observed':>12}  {'allowance':>10}  "
          f"{'spread':>8}  result")
    for capacity in args.capacities:
        runs = [
            measure(args.cli, args.model, capacity, spec, args.draft_tokens,
                    args.kv_dtype, args.prefill_chunk)
            for _ in range(args.repeats)
        ]
        rows.extend(runs)
        seen = [r["observed_bytes"] for r in runs if r["observed_bytes"] is not None]
        graphs = runs[0]["predicted_graphs"]
        if not seen:
            print(f"{capacity:>9}  {'n/a' if graphs is None else graphs!s:>6}  "
                  "no graph figure reported")
            continue
        median = statistics.median(seen)
        spread = max(seen) - min(seen)
        allowance = next(r["allowance_bytes"] for r in runs if r["allowance_bytes"] is not None)
        blocked = sum(1 for r in runs if r["graph_blocked"])
        verdict = "ok" if blocked == 0 and len(seen) == len(runs) else (
            f"BLOCKED {blocked}/{len(runs)}" if blocked else "INCOMPLETE"
        )
        graph_label = "n/a" if graphs is None else str(graphs)
        print(f"{capacity:>9}  {graph_label:>6}  {median / MIB:>10.2f} MiB  "
              f"{allowance / MIB:>8.2f} MiB  {spread / MIB:>6.2f} MiB  {verdict}")

    report = {
        "schema_version": 2,
        "interpretation": ("Device-wide observed allocation deltas, not per-process GPU memory. "
                           "No WDDM or scheduling attribution is implied."),
        "config": {
            "cli": str(args.cli), "model": str(args.model), "repeats": args.repeats,
            "spec": spec, "draft_tokens": args.draft_tokens if spec else None,
            "kv_dtype": args.kv_dtype, "prefill_chunk": args.prefill_chunk,
            "capacities": args.capacities,
        },
        "runs": rows,
    }
    destination = args.output / f"graph-allowance-{args.spec}-{args.kv_dtype}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {destination}")
    return 0 if all(r["observed_bytes"] is not None for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
