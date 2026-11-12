"""Write the measured compile times of the heavy release objects for build-object-shard.py.

usage: object-weights.py OUTPUT LOG_DIR... [--minimum 120]
Each LOG_DIR holds downloaded build diagnostics, <platform>-logs-<toolkit>-objects-<n>/build-release/
.ninja_log, of a build that compiled the objects without cache hits. An object's weight is its longest
time over the toolkits, in whole seconds; objects under the minimum are left out and keep their
path-hash owner.
"""
import argparse
import json
from pathlib import Path
import re


def object_key(output):
    """The object's build-tree path without the platform's object suffix."""
    output = output.replace("\\", "/")
    return output[:-len(".obj")] if output.endswith(".obj") else output[:-len(".o")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--minimum", type=int, default=120)
    args = parser.parse_args()
    weights = {}
    for root in args.logs:
        for log in sorted(root.glob("*/build-release/.ninja_log")):
            match = re.match(r"(linux|windows)-logs-[\d.]+-objects-\d+$", log.parents[1].name)
            if not match:
                continue
            platform = weights.setdefault(match.group(1), {})
            for line in log.read_text().splitlines():
                parts = line.split("\t")
                if line.startswith("#") or len(parts) < 4 or not parts[3].endswith((".o", ".obj")):
                    continue
                key = object_key(parts[3])
                seconds = round((int(parts[1]) - int(parts[0])) / 1000)
                platform[key] = max(platform.get(key, 0), seconds)
    kept = {platform: dict(sorted((key, seconds) for key, seconds in times.items()
                                  if seconds >= args.minimum))
            for platform, times in sorted(weights.items())}
    args.output.write_text(json.dumps(kept, indent=1) + "\n")
    for platform, times in kept.items():
        print(f"{platform}: {len(times)} objects of at least {args.minimum} s, "
              f"{sum(times.values()) / 3600:.1f} compiler-hours")


if __name__ == "__main__":
    main()
