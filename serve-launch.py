#!/usr/bin/env python3
"""Launch the fp8-dev trial serve and mirror its stdout/stderr to a log file.

Usage: serve-launch.py <logfile> <program> [args...]

Why a Python launcher instead of `start ... cmd /c "serve ... > log 2>&1"`:
  * `start` re-tokenises the command line after the program name, so a nested
    `cmd /c "prog > log"` with many quoted args is fragile (quotes get mangled,
    the redirect can detach from the program). Passing simple no-space args to a
    Python program and doing the redirect inside Python is deterministic.
  * The serve runs as a child of this launcher and inherits its console, so the
    stop path (trial-stop.bat -> AttachConsole + GenerateConsoleCtrlEvent on the
    serve PID found via the port) still works exactly like the production stop.

The child writes straight to the log's file descriptor (no Python buffering on
that path). C stdio is block-buffered when its stdout is a file, so a line may
linger in the child's buffer until it flushes; the launcher also flushes on exit.
"""
import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 3:
        sys.stderr.write("usage: serve-launch.py <logfile> <program> [args...]\n")
        return 2
    log_path = sys.argv[1]
    cmd = sys.argv[2:]
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
