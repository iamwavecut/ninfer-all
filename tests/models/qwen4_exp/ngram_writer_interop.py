"""Check CUDA availability before loading the n-gram conversion test dependencies."""

import importlib.util
from pathlib import Path
import subprocess
import sys

# The writer test imports these; without them CTest reports a skip, as it does without CUDA.
WRITER_MODULES = ("numpy", "pytest", "safetensors", "torch")


def main() -> int:
    binary = sys.argv[1]
    probe = subprocess.run([binary, "ninfer_qwen4_exp_ngram_component_test", "--check-cuda"])
    if probe.returncode:
        return probe.returncode
    missing = [name for name in WRITER_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        print(f"SKIP: the n-gram writer test needs {', '.join(missing)} in {sys.executable}")
        return 77
    writer = Path(__file__).resolve().parents[2] / "convert/test_qwen4_exp_ngram.py"
    return subprocess.run([sys.executable, "-B", str(writer), binary]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
