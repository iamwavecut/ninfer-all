"""The CTest launcher must skip GPU work before importing conversion dependencies."""

from pathlib import Path
import subprocess
import sys

import pytest

WRITER_MODULES = ("numpy", "pytest", "safetensors", "torch")


@pytest.mark.parametrize("probe_result,modules,expected", [
    (0, True, 23),   # CUDA and the writer's modules: the converter runs.
    (0, False, 77),  # CUDA without the modules: a skip, before the converter starts.
    (77, True, 77),  # No CUDA: the probe's skip, whatever is installed.
    (77, False, 77),
    (1, False, 1),   # A failing probe fails the test.
])
def test_launcher_probes_before_running_conversion(tmp_path, probe_result, modules, expected):
    root = tmp_path / "tests"
    launcher = root / "models/qwen4_exp/ngram_writer_interop.py"
    launcher.parent.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "models/qwen4_exp/ngram_writer_interop.py"
    launcher.write_bytes(source.read_bytes())
    if modules:
        # The launcher's directory is first on sys.path; it only looks the modules up.
        for name in WRITER_MODULES:
            (launcher.parent / f"{name}.py").write_text("raise ImportError('not for import')\n")
    writer = root / "convert/test_qwen4_exp_ngram.py"
    writer.parent.mkdir()
    # The converter fails if invoked: unavailable CUDA must bypass its imports entirely.
    writer.write_text("raise SystemExit(23)\n")
    binary = tmp_path / "probe"
    binary.write_text(f"#!{sys.executable}\nimport sys\n"
                      "assert sys.argv[1:] == ['ninfer_qwen4_exp_ngram_component_test', '--check-cuda']\n"
                      f"raise SystemExit({probe_result})\n")
    binary.chmod(0o755)
    # -S removes site packages, matching the toolchain without pytest or CPU PyTorch.
    result = subprocess.run([sys.executable, "-S", str(launcher), str(binary)],
                            capture_output=True, text=True)
    assert result.returncode == expected
    assert ("SKIP: the n-gram writer test needs" in result.stdout) == (probe_result == 0 and not modules)
