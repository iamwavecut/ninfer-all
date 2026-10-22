"""CPU-only tests of the RTX 4090 public Engine tuning harness."""
import math
import pytest

from tools.bench.tune_rtx4090 import (
    Candidate, candidates, make_command, select_winner, valid_speed, workloads,
)


def test_candidates_have_single_baseline_and_unique_names():
    for preset in ("quick", "full"):
        modes = candidates(preset, ("mtp", "dflash2"))
        assert len(modes) == len({m.slug for m in modes})
        assert modes[0] == Candidate("rk4v4")
        assert all(m.backend == "none" or m.drafts > 0 for m in modes)
    assert workloads("full")[-1] == (32768, 128)
    assert candidates("quick") == [Candidate("rk4v4")]
    assert Candidate("rk4v4", "dflash2", 3) in candidates("quick", ("dflash2",))


def test_command_uses_real_api_and_isolates_ngram(tmp_path):
    command = make_command(tmp_path / "ninfer_bench", tmp_path / "model.ninfer",
                           tmp_path / "ids", 0, Candidate("rk4v4", "dflash2", 7),
                           2048, 128, tmp_path / "out.json")
    assert command[command.index("-pg") + 1] == "2048,128"
    assert command[command.index("--max-ctx") + 1] == "32768"
    assert command[command.index("--draft-tokens") + 1] == "7"
    assert command[command.index("--ngram-draft-tokens") + 1] == "0"


def test_report_rejects_wrong_device_and_nonfinite():
    report = {"schema_version": 18, "artifact_type": "ninfer_bench_report",
              "tool": "ninfer_bench", "environment": {"gpu_name": "NVIDIA GeForce RTX 4090"},
              "tests": [{"n_prompt": 2048, "n_gen": 128,
                         "decode_output_tok_s_mean": 178.5}]}
    assert valid_speed(report, 2048, 128) == 178.5
    report["environment"]["gpu_name"] = "NVIDIA RTX 4070 Ti"
    with pytest.raises(ValueError, match="wrong GPU"):
        valid_speed(report, 2048, 128)
    report["environment"]["gpu_name"] = "NVIDIA GeForce RTX 4090"
    report["tests"][0]["decode_output_tok_s_mean"] = math.nan
    with pytest.raises(ValueError, match="invalid decode"):
        valid_speed(report, 2048, 128)


def test_requires_complete_measurements_and_meaningful_gain():
    modes = [Candidate("rk4v4"), Candidate("rk4v4", "mtp", 3)]
    rates = {modes[0].slug: [100, 100, 100], modes[1].slug: [110, 110]}
    result = select_winner(rates, modes, 3)
    assert result["selected"] == modes[0].slug
    rates[modes[1].slug] = [101, 102, 102]
    assert select_winner(rates, modes, 3)["selected"] == modes[0].slug
    rates[modes[1].slug] = [108, 111, 105]
    winner = select_winner(rates, modes, 3)
    assert winner["selected"] == modes[1].slug
    assert winner["relative_gain_percent"] == 8
    del rates[modes[0].slug]
    assert select_winner(rates, modes, 3)["selected"] is None
