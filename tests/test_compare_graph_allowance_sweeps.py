"""GPU-free comparison tests for graph-memory reports."""
import pytest

from tools.bench.compare_graph_allowance_sweeps import compare_reports


def sample(capacity, observed, code=0):
    return {"capacity": capacity, "observed_bytes": observed,
            "returncode": code}


def sweep(values, **overrides):
    config = {"model": "C:/models/qwen.ninfer", "spec": "dflash2",
              "draft_tokens": 5, "kv_dtype": "rk4v4", "prefill_chunk": 1024}
    config.update(overrides)
    return {"config": config, "runs": values}


def test_nonoverlapping_observations_are_only_described_as_memory_deltas():
    first = sweep([sample(8192, 90), sample(8192, 100), sample(8192, 105)])
    second = sweep([sample(8192, 60), sample(8192, 65), sample(8192, 70)])
    result = compare_reports(first, second)
    row = result["rows"][0]
    assert row["baseline_median_bytes"] == 100
    assert row["candidate_median_bytes"] == 65
    assert row["candidate_minus_baseline_bytes"] == -35
    assert row["verdict"].startswith("lower observed VRAM delta")
    assert result["scope"] == "CUDA Graph startup-memory observations, not throughput"


def test_device_wide_noise_and_zero_deltas_do_not_become_a_speedup():
    first = sweep([sample(8192, 0), sample(8192, 80), sample(8192, 120)])
    second = sweep([sample(8192, 100), sample(8192, 110), sample(8192, 130)])
    row = compare_reports(first, second)["rows"][0]
    assert row["baseline_successes"] == 2
    assert row["verdict"].startswith("inconclusive")


def test_startup_failures_are_not_counted_as_valid_calibration_samples():
    first = sweep([sample(4096, 90), sample(4096, 100)])
    second = sweep([sample(4096, 10, code=1), sample(4096, 40)])
    row = compare_reports(first, second)["rows"][0]
    assert row["candidate_successes"] == 1
    assert row["verdict"] == "insufficient samples"


def test_mixed_backend_is_not_compared():
    with pytest.raises(ValueError, match="spec"):
        compare_reports(sweep([sample(4096, 60)]),
                        sweep([sample(4096, 50)], spec="mtp"))


def test_nonmatching_capacities_are_rejected():
    with pytest.raises(ValueError, match="no matching"):
        compare_reports(sweep([sample(4096, 60)]), sweep([sample(8192, 50)]))
