"""GPU-free contract tests for the RTX graph allowance measurement runner."""
import json
from types import SimpleNamespace

from tools.bench import run_graph_allowance_sweep as sweep


def test_graph_diagnostic_reads_stdout_and_keeps_dflash_graph_count_unknown(monkeypatch, tmp_path):
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(returncode=0, stdout="CUDA Graph memory 11.50 MiB / 24 MiB\n", stderr="")

    monkeypatch.setattr(sweep.subprocess, "run", run)
    measured = sweep.measure(tmp_path / "ninfer.exe", tmp_path / "model.ninfer",
                             8192, "dflash2", 7, "rk4v4", 1024)
    assert measured["started"] and not measured["graph_blocked"]
    assert measured["observed_bytes"] == int(11.5 * sweep.MIB)
    assert measured["allowance_bytes"] == 24 * sweep.MIB
    assert measured["predicted_graphs"] is None
    assert commands[0][-5:] == ["--spec", "dflash2", "--draft-tokens", "7", "--lm-head-draft"]


def test_exact_failure_bytes_take_precedence_over_rounded_graph_line(monkeypatch, tmp_path):
    def run(*args, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stderr=("CUDA Graph memory 12.00 MiB / 2.00 MiB\n"
                    "CUDA Graph preparation consumed 12582913 bytes, "
                    "exceeding the planned allowance of 2097152 bytes"),
            stdout="",
        )

    monkeypatch.setattr(sweep.subprocess, "run", run)
    measured = sweep.measure(tmp_path / "ninfer.exe", tmp_path / "model.ninfer",
                             2048, None, 3, "int8", 640)
    assert measured["graph_blocked"]
    assert measured["observed_bytes"] == 12582913
    assert measured["allowance_bytes"] == 2097152
    assert measured["predicted_graphs"] == sweep.ordinary_graph_count(2048)


def test_cli_emits_reproducible_sweep_and_handles_speculative_counts(monkeypatch, tmp_path):
    exe = tmp_path / "ninfer.exe"
    model = tmp_path / "qwen.ninfer"
    exe.write_bytes(b"test")
    model.write_bytes(b"test")
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="",
                               stderr="CUDA Graph memory 0 B / 64 MiB")

    monkeypatch.setattr(sweep.subprocess, "run", run)
    code = sweep.main([
        "--cli", str(exe), "--model", str(model),
        "--spec", "dflash2", "--draft-tokens", "5",
        "--kv-dtype", "rk4v4", "--prefill-chunk", "1024",
        "--capacities", "2048", "8192", "--repeats", "2",
        "--output", str(tmp_path / "results"),
    ])
    assert code == 0
    assert len(calls) == 4
    report = json.loads((tmp_path / "results" /
                         "graph-allowance-dflash2-rk4v4.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == 2
    assert report["config"]["capacities"] == [2048, 8192]
    assert all(r["predicted_graphs"] is None for r in report["runs"])
    assert all(r["observed_bytes"] == 0 for r in report["runs"])


def test_missing_graph_diagnostics_is_incomplete_not_zero(monkeypatch, tmp_path):
    exe = tmp_path / "ninfer.exe"
    model = tmp_path / "qwen.ninfer"
    exe.write_bytes(b"x")
    model.write_bytes(b"x")
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=0, stdout="running", stderr=""))
    result = sweep.main([
        "--cli", str(exe), "--model", str(model),
        "--capacities", "4096", "--repeats", "2", "--output", str(tmp_path / "out"),
    ])
    assert result == 2
