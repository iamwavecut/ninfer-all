from __future__ import annotations

import json

import pytest

from tools.convert import basalt_config as config


def translated(engine=None, **kwargs):
    return config.translate({"engine": {"kv": "int8", "resident-experts": True,
                                         **(engine or {})}}, "model.ninfer", has_mtp=True,
                            expert_bytes=1 << 20, **kwargs)


def test_real_defaults_use_gpu_zero_automatic_workers_and_profile_cache():
    out, report = translated({"pool-workers": 0, "expert-cache": 0})
    assert out["device"] == 0
    assert out["expert-cache-mib"] == "auto"
    assert "expert-cpu-threads" not in out
    assert out["ngram-io-depth"] == 256
    assert not report["unresolved"]
    unprofiled, _ = translated({"expert-cache": 0}, has_profile=False)
    assert unprofiled["expert-cache-mib"] == 0


def test_fp16_maps_to_native_fp16_without_a_precision_substitution():
    out, report = translated({"kv": "fp16"})
    assert out["kv-dtype"] == "fp16"
    assert "kv" not in report["unresolved"]


@pytest.mark.parametrize("mode", ["dynamic", "cpu", "1"])
def test_vision_placement_requires_explicit_resident_cost_choice(mode):
    out, report = translated({"vision": mode})
    assert "vision-residency" not in out and "vision" not in out
    assert report["unresolved"]["vision"]["value"] == mode


def test_flash_next_lookup_uses_the_working_backend_and_match_threshold():
    out, report = translated({"spec": 7, "suffix-draft": 3})
    assert out["draft-tokens"] == 6 and out["draft-min-p"] == 0
    assert out["lookup-ngram"] == 3 and out["ngram-draft-tokens"] == 0
    assert any("longer lookup window" in item for item in report["notes"])
    capped, _ = translated({"spec": 6, "mtp-max-t": 4, "suffix-draft": 5})
    assert capped["draft-tokens"] == 3 and capped["lookup-ngram"] == 5
    off, _ = translated({"spec": 0})
    assert off["spec"] == "none" and off["lookup-ngram"] == 0 and "draft-tokens" not in off


def test_config_versions_and_model_override_precedence():
    old, _ = config.translate({"v": 1, "server": {"gpu": 2}, "engine": {"kv": "int8"}},
                              "model.ninfer", has_mtp=False)
    assert old["device"] == 2
    new, report = config.translate({"engine": {"parallel": 3}, "models": [
        {"model": "a.basalt", "parallel": 2}]}, "a.ninfer", has_mtp=False)
    assert new["max-concurrency"] == 2 and report["source_version"] == 2
    with pytest.raises(ValueError, match="legacy"):
        config.translate({"v": 1, "version": 2, "models": []}, "a.ninfer", has_mtp=False)


def test_sampling_defaults_use_pack_metadata_then_explicit_launch():
    defaults = config.pack_sampling({"sampling": {"temperature": 0.4, "top_k": 12},
        "gguf_kv": {"general.sampling.temp": 0.9, "general.sampling.top_p": 0.95,
                    "general.sampling.min_p": 0.03, "general.sampling.penalty_repeat": 1.1}})
    assert defaults == {"temperature": 0.9, "top-k": 12, "top-p": 0.95,
                        "min-p": 0.03, "repetition-penalty": 1.1}
    out, report = translated({"temperature": 0.2}, sampling_defaults=defaults)
    assert out["temperature"] == 0.2 and out["top-p"] == 0.95 and out["top-k"] == 12
    assert out["min-p"] == 0.03
    assert report["unresolved"]["repetition-penalty"]["value"] == 1.1
    assert report["pack_sampling"] == defaults


@pytest.mark.parametrize("name,value", [("top-k", 64), ("temperature", 3.0),
                                         ("presence-penalty", 3), ("frequency-penalty", -3)])
def test_sampling_limits_are_reported_not_clamped_or_passed_through(name, value):
    out, report = translated({name: value})
    assert name not in out and report["unresolved"][name]["value"] == value


def test_explicit_disabled_jev_is_an_access_policy_decision():
    _, report = translated({"jev": 0})
    assert "jev" in report["unresolved"]
    _, enabled = translated({"jev": 4})
    assert not enabled["unresolved"]
    assert any("shared NInfer scheduler" in note for note in enabled["notes"])


def test_worker_limit_and_zero_draft_cap_cannot_emit_invalid_options():
    out, report = translated({"pool-workers": 300, "mtp-max-t": 1})
    assert "expert-cpu-threads" not in out and "draft-tokens" not in out
    assert set(report["unresolved"]) == {"pool-workers", "mtp-max-t"}


def test_cli_requires_both_acknowledgement_and_target_choice(tmp_path, monkeypatch):
    class Pack:
        sections = {"mtp/dense": None}
        meta = {"gguf_kv": {"qwen4exp.expert_count": 512}}

        def __init__(self, _):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def text(self, _):
            return "# experts v5:\n0 21 21 0 1000\n"

    monkeypatch.setattr(config, "BasaltFile", Pack)
    source, output = tmp_path / "basalt.json", tmp_path / "ninfer.json"
    source.write_text(json.dumps({"engine": {"kv": "q4", "resident-experts": True}}))
    args = [str(source), "--pack", "pack.basalt", "--artifact", "model.ninfer", "--output", str(output)]
    with pytest.raises(SystemExit):
        config.main(args + ["--acknowledge", "kv"])
    assert not output.exists()
    report = json.loads(output.with_name(output.name + ".report.json").read_text())
    assert report["source_options"]["kv"] == "q4"
    with pytest.raises(SystemExit):
        config.main(args + ["--set", "kv-dtype=bf16"])
    assert not output.exists()
    config.main(args + ["--acknowledge", "kv", "--set", "kv-dtype=bf16"])
    assert json.loads(output.read_text())["kv-dtype"] == "bf16"
