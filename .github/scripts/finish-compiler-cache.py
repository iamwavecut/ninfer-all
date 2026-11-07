"""Stop compiler work and prepare its bounded object cache for upload."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from build_telemetry import cache_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--evict-unused-since", type=float, metavar="UNIX_TIME",
                        help="Evict the entries no compilation has used since this time")
    args = parser.parse_args()
    if args.pid_file and args.pid_file.exists():
        try:
            os.kill(int(args.pid_file.read_text()), signal.SIGINT)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 6
        while args.pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.2)
    # Each successful compiler invocation has already committed its cache entry atomically.
    if args.evict_unused_since is not None:
        # A hit renews an entry's modification time; a restored entry keeps its archived one.
        age = max(1, math.ceil(time.time() - args.evict_unused_since))
        subprocess.run(["ccache", f"--evict-older-than={age}s"], check=True, timeout=60)
    subprocess.run(["ccache", "--cleanup"], check=True, timeout=60)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(cache_stats(), indent=2) + "\n")
    subprocess.run(["ccache", "--show-stats"], check=True, timeout=20)


if __name__ == "__main__":
    main()
