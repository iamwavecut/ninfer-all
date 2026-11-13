from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from safetensors.torch import save_file
import torch

from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts
from tools.convert.dspine import build_model, dspine_config, is_dspine
from tools.convert.sources.safetensors import SafetensorsSource

H, INNER, HEADS, KV, D, LAYERS, VOCAB, CODE, RANK = 16, 8, 4, 2, 8, 2, 40, 16, 32
TAPS = [0, 2]


def _config():
    return {
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": False,
        "hidden_size": H,
        "intermediate_size": INNER,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV,
        "head_dim": D,
        "num_hidden_layers": LAYERS,
        "max_position_embeddings": 4096,
        "vocab_size": VOCAB,
        "draft_vocab_size": VOCAB,
        "num_target_layers": 4,
        "block_size": 16,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "rope_scaling": None,
        "layer_types": ["sliding_attention"] * LAYERS,
        "sliding_window": 64,
        "code_bridge_second_pass": {"mode": "serial_table", "first_exact": LAYERS - 1, "head_topk": 16},
        "dflash_config": {
            "draft_block_causal": True,
            "mask_token_id": 7,
            "selected_prefix_code_bridge": {"code_size": CODE, "eps": 1e-6, "kappa": 10.0, "rank": RANK},
            "selected_prefix_decode_mode": "serial",
            "selected_prefix_feature_layers": list(range(1, LAYERS + 1)),
            "selected_prefix_feature_write_mode": "independent",
            "selected_prefix_feature_writes": True,
            "selected_prefix_last_write_refine": {
                "head_topk": 16, "kind": "exact_sender", "version": 1, "weight": 1.0},
            "selected_prefix_terminal": False,
            "target_layer_ids": TAPS,
        },
    }


def _tensors(seed=0):
    generator = torch.Generator().manual_seed(seed)

    def bf16(*shape):
        return torch.randn(*shape, generator=generator).to(torch.bfloat16)

    tensors = {
        "fc.weight": bf16(H, len(TAPS) * H),
        "hidden_norm.weight": bf16(H),
        "norm.weight": bf16(H),
        "draft_block_causal_version": torch.tensor(1, dtype=torch.int64),
        "last_write_refine_version": torch.tensor(1, dtype=torch.int64),
    }
    for i in range(LAYERS):
        p = f"layers.{i}."
        tensors |= {
            p + "input_layernorm.weight": bf16(H),
            p + "post_attention_layernorm.weight": bf16(H),
            p + "self_attn.q_proj.weight": bf16(HEADS * D, H),
            p + "self_attn.k_proj.weight": bf16(KV * D, H),
            p + "self_attn.v_proj.weight": bf16(KV * D, H),
            p + "self_attn.o_proj.weight": bf16(H, HEADS * D),
            p + "self_attn.q_norm.weight": bf16(D),
            p + "self_attn.k_norm.weight": bf16(D),
            p + "mlp.gate_proj.weight": bf16(INNER, H),
            p + "mlp.up_proj.weight": bf16(INNER, H),
            p + "mlp.down_proj.weight": bf16(H, INNER),
        }
    b = "selected_prefix_bridge."
    tensors |= {
        b + "codebook": bf16(VOCAB, CODE),
        b + "early_read_weights": bf16(LAYERS - 1, CODE, H),
        b + "read_weight": bf16(CODE, H),
        b + "feature_write_weights": bf16(LAYERS, H, RANK),
        b + "gate.weight": bf16(RANK, CODE),
        b + "gate.bias": bf16(RANK),
        b + "predecessor_gate.weight": bf16(RANK, CODE),
        b + "prev.weight": bf16(RANK, CODE),
        b + "out.weight": torch.zeros(CODE, RANK, dtype=torch.bfloat16),
        b + "independent_write_version": torch.tensor(1, dtype=torch.int64),
        b + "terminal_disabled_version": torch.tensor(1, dtype=torch.int64),
    }
    return tensors


def _checkpoint(path: Path, config=None, tensors=None) -> Path:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(_config() if config is None else config))
    save_file(_tensors() if tensors is None else tensors, path / "model.safetensors")
    return path


def _values(model, name):
    parameter = model.parameters[name]
    return parameter.source.values().reshape(parameter.shape)


def test_config_keeps_the_geometry_and_refuses_unimplemented_modes():
    config = dspine_config(_config())
    assert is_dspine(_config())
    assert config["architectures"] == ["DSpineDraftModel"]
    assert config["sliding_window"] == 64 and config["block_size"] == 16
    assert config["dspine_config"] == {
        "target_layer_ids": TAPS, "mask_token_id": 7, "code_size": CODE, "rank": RANK,
        "bridge_eps": pytest.approx(1e-6), "head_top_k": 16}
    plain = _config()
    del plain["dflash_config"]["selected_prefix_code_bridge"]
    assert not is_dspine(plain)
    for change in (
        lambda c: c["dflash_config"].update(selected_prefix_terminal=True),
        lambda c: c["dflash_config"].update(draft_block_causal=False),
        lambda c: c["dflash_config"].update(selected_prefix_feature_write_mode="shared"),
        lambda c: c["dflash_config"].update(selected_prefix_feature_layers=[1]),
        lambda c: c["dflash_config"].update(conv_kernel_size=4),
        lambda c: c["dflash_config"].update(selected_prefix_terminal_residual_rank=64),
        lambda c: c["code_bridge_second_pass"].update(first_exact=0),
        lambda c: c["code_bridge_second_pass"].update(head_topk=8),
        lambda c: c.update(layer_types=["full_attention"] * LAYERS),
        lambda c: c["dflash_config"].update(target_layer_ids=[0, 4]),
        lambda c: c["dflash_config"]["selected_prefix_code_bridge"].update(rank=48),
    ):
        config = deepcopy(_config())
        change(config)
        with pytest.raises(ValueError):
            dspine_config(config)


def test_bridge_parameters_map_the_checkpoint_and_the_sender_table_matches_its_formula(tmp_path):
    tensors = _tensors()
    with SafetensorsSource(_checkpoint(tmp_path / "draft", tensors=tensors)) as source:
        model = build_model(source)
        assert set(model.components) == {"dspine"}
        assert not any("codebook" in name for name in model.parameters)
        b = "selected_prefix_bridge."
        assert torch.equal(
            _values(model, "dspine/bridge/projection"),
            torch.cat([tensors[b + "gate.weight"], tensors[b + "predecessor_gate.weight"],
                       tensors[b + "prev.weight"]]))
        assert torch.equal(_values(model, "dspine/bridge/gate_bias"), tensors[b + "gate.bias"])
        for i in range(LAYERS):
            read = (tensors[b + "read_weight"] if i == LAYERS - 1
                    else tensors[b + "early_read_weights"][i])
            assert torch.equal(_values(model, f"dspine/bridge/layers/{i}/read"), read)
            assert torch.equal(_values(model, f"dspine/bridge/layers/{i}/write"),
                               tensors[b + "feature_write_weights"][i])
        assert torch.equal(_values(model, "dspine/bridge/head_write"),
                           tensors[b + "feature_write_weights"][-1].T.contiguous())
        assert model.parameters["dspine/bridge/head_write"].inputs == ()
        assert torch.equal(_values(model, "dspine/feature_projection"), tensors["fc.weight"])
        assert model.parameters["dspine/layers/1/attention/context_key"].inputs == (
            "dspine/context_input",)
        # The independent oracle: FP64 products of the represented codes, one BF16 rounding.
        codes = (CODE ** 0.5 * tensors[b + "codebook"].double()).to(torch.bfloat16).double()
        sender = torch.cat([tensors[b + "predecessor_gate.weight"],
                            tensors[b + "prev.weight"]]).double()
        expected = (codes @ sender.T).to(torch.bfloat16)
        actual = _values(model, "dspine/bridge/sender_table")
        assert actual.dtype == torch.bfloat16 and actual.shape == (VOCAB, 2 * RANK)
        # FP32 accumulation may round a value next to a tie the other way: one BF16 step at most.
        step = expected.double().abs().clamp_min(2.0 ** -126) * 2.0 ** -7
        assert bool(((actual.double() - expected.double()).abs() <= step).all())
        # Row reads at any offset give the same values as a whole read.
        source_table = model.parameters["dspine/bridge/sender_table"].source
        assert torch.equal(source_table.values(70, 2 * RANK * 3 + 5),
                           actual.reshape(-1)[70: 2 * RANK * 3 + 5])


@pytest.mark.parametrize(
    "name,value",
    [("selected_prefix_bridge.out.weight", torch.ones(CODE, RANK, dtype=torch.bfloat16)),
     ("last_write_refine_version", torch.tensor(2, dtype=torch.int64)),
     ("selected_prefix_bridge.terminal_disabled_version", torch.tensor(0, dtype=torch.int64))])
def test_markers_of_other_checkpoint_versions_are_refused(tmp_path, name, value):
    tensors = _tensors()
    tensors[name] = value
    with SafetensorsSource(_checkpoint(tmp_path / "draft", tensors=tensors)) as source:
        with pytest.raises(ValueError):
            build_model(source)


def test_cli_writes_a_standalone_q8_drafter(tmp_path):
    source = _checkpoint(tmp_path / "draft")
    output = tmp_path / "drafter.ninfer"
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "-m", "tools.convert", "--model", str(source), "--recipe",
               "dspine_q8", "--out", str(output), "--device", "cpu", "--name", "dspine-test"]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stderr
    with Artifact(output) as artifact:
        directory = artifact.directory
        assert set(directory.components) == {"dspine"}
        assert directory.components["dspine"]["config"]["dspine_config"]["rank"] == RANK

        def parts(name):
            return binding_parts(directory.bindings[name], artifact.by_id)

        def stored(name):
            return artifact.object(parts(name)[0][0])

        assert stored("dspine/layers/0/attention/query").format == "q8_g32_fp16"
        assert stored("dspine/feature_projection").format == "q8_g32_fp16"
        for name in ("dspine/bridge/projection", "dspine/bridge/layers/1/write",
                     "dspine/bridge/head_write", "dspine/bridge/sender_table"):
            assert stored(name).format == "bf16", name
        assert (parts("dspine/layers/1/attention/context_key")
                == parts("dspine/layers/1/attention/key"))
    rejected = subprocess.run(command[:-2] + ["--components", "text,dspine"], cwd=root,
                              capture_output=True, text=True)
    assert rejected.returncode != 0 and "converts alone" in rejected.stderr
