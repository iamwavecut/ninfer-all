"""Dense Qwen4 (Qwen4ForCausalLM) through the Qwen4 adapter: its text and MTP configs, the
parameters of its SwiGLU MLP in the text layers and the MTP block, and the PLE it may leave out.
The fixture's geometry is the 27B's as inferred from public code: no checkpoint is published."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from tools.convert.qwen4_exp import has_ple, is_dense, mtp_config, text_config

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "qwen4"
_PLE_FIELDS = ("ple_embed_dim", "ple_conv_kernel_size", "ngram_size", "heads_per_ngram",
               "ngram_vocab_size_base", "make_ngram_vocab_size_divisible_by", "split_ngram_parts")


def _checkpoint():
    return json.loads((_FIXTURES / "config.json").read_text())


def _without_ple(source):
    text = source["text_config"]
    text["ple_layer_ids"] = []
    for key in _PLE_FIELDS:
        text.pop(key, None)
    return source


def test_text_config_matches_the_artifact_fixture():
    # The C++ parser's test reads the same fixture, so the two stay in step.
    expected = json.loads((_FIXTURES / "text_config.json").read_text())
    assert text_config(_checkpoint()) == expected
    assert is_dense(_checkpoint()) and has_ple(_checkpoint())


def test_text_config_without_ple_has_no_ple_fields():
    config = text_config(_without_ple(_checkpoint()))
    assert config["ple_layers"] == []
    assert not {"ngram_seed", "ple_embed_dim", "ngram_size"} & set(config)
    source = _checkpoint()
    del source["text_config"]["ple_layer_ids"]
    for key in _PLE_FIELDS:
        source["text_config"].pop(key)
    assert text_config(source)["ple_layers"] == []
    assert not has_ple(source)


def test_a_text_only_config_rotates_one_axis():
    source = _checkpoint()
    del source["text_config"]["rope_parameters"]["mrope_section"]
    # 256 * 0.25 / 2 pairs, all on the first section.
    assert text_config(source)["rope_parameters"]["mrope_section"] == [32, 0, 0]


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda t: t.update(num_experts=512), "no MoE"),
        (lambda t: t.update(moe_intermediate_size=640), "no MoE"),
        (lambda t: t.pop("output_gate_type"), "must declare 'sigmoid'"),
        (lambda t: t.update(output_gate_type="silu"), "output_gate_type"),
        (lambda t: t.update(mlp_layer_types=["sparse"] * 64), "dense MLPs only"),
        (lambda t: t.update(model_type="qwen4_exp_text"), "model_type"),
        (lambda t: t.pop("intermediate_size"), "intermediate_size"),
        (lambda t: t.update(ple_layer_ids=[]), "no PLE fields"),
        (lambda t: t.pop("ngram_size"), "ngram_size"),
        (lambda t: t.update(attn_output_gate=False), "attn_output_gate"),
    ],
)
def test_text_config_refuses(edit, message):
    source = deepcopy(_checkpoint())
    edit(source["text_config"])
    with pytest.raises(ValueError, match=message):
        text_config(source)


def test_mtp_config():
    assert mtp_config(_checkpoint()) == {
        "architectures": ["Qwen4MTP"],
        "rope_theta": 10000000.0,
        "intermediate_size": 14336,
    }
    source = _checkpoint()
    source["text_config"]["mtp"]["intermediate_size"] = 7168
    assert mtp_config(source)["intermediate_size"] == 7168
    source["text_config"]["mtp"]["num_experts"] = 8
    with pytest.raises(ValueError, match="has no MoE"):
        mtp_config(source)


def _tiny_config(ple=True):
    text = {
        "model_type": "qwen4_text",
        "hidden_size": 8,
        "vocab_size": 12,
        "num_hidden_layers": 2,
        "max_position_embeddings": 64,
        "layer_types": ["linear_attention", "full_attention"],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.5,
                            "mrope_section": [1, 0, 0], "mrope_interleaved": True},
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 2,
        "linear_num_value_heads": 2,
        "linear_value_head_dim": 2,
        "linear_conv_kernel_dim": 4,
        "intermediate_size": 6,
        "hc_count": 2,
        "hc_lowrank": 3,
        "indexer_n_heads": 2,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 4,
        "indexer_compress_ratio": 4,
        "indexer_budget": 8,
        "eos_token_id": 11,
        "output_gate_type": "sigmoid",
        "mamba_ssm_dtype": "float32",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "mtp": {"num_hidden_layers": 1, "hybrid": True, "layer_types": ["full_attention"],
                "rope_theta": 10000000, "intermediate_size": 10},
        "ple_layer_ids": [],
    }
    if ple:
        text.update(ple_layer_ids=[1], ple_embed_dim=8, ple_conv_kernel_size=4, ngram_size=3,
                    heads_per_ngram=2, ngram_vocab_size_base=100,
                    make_ngram_vocab_size_divisible_by=8)
    return {"architectures": ["Qwen4ForConditionalGeneration"], "text_config": text}


def _write_source(root: Path, config: dict) -> None:
    from safetensors.torch import save_file

    root.mkdir()
    (root / "config.json").write_text(json.dumps(config))
    save_file({"model.language_model.layers.0.mlp.up_proj.weight":
               torch.arange(48).float().reshape(6, 8)}, root / "model.safetensors")
    for name, value in {
        "tokenizer.json": {"model": {"vocab": {str(i): i for i in range(6)}}},
        "tokenizer_config.json": {},
        "generation_config.json": {},
    }.items():
        (root / name).write_text(json.dumps(value))
    (root / "chat_template.jinja").write_text("{{ messages }}")


def test_build_model_maps_the_dense_mlp(tmp_path):
    from tools.convert.qwen4_exp import build_model
    from tools.convert.sources.safetensors import SafetensorsSource

    root = tmp_path / "source"
    _write_source(root, _tiny_config())
    with SafetensorsSource(root) as source:
        model = build_model(source, components=("text", "mtp"))
        names = set(model.parameters)
        for p in ("text/layers/0/", "text/layers/1/"):
            assert model.parameters[p + "mlp/gate"].shape == (6, 8)
            assert model.parameters[p + "mlp/gate"].inputs == (p + "ffn_input",)
            assert model.parameters[p + "mlp/down"].shape == (8, 6)
            assert model.parameters[p + "mlp/down"].inputs == (p + "mlp/product",)
            assert (p + "mlp/gate", p + "mlp/up") in model.packing_groups
        assert not any("/moe/" in name for name in names)
        up = model.parameters["text/layers/0/mlp/up"]
        assert torch.equal(up.source.values().reshape(6, 8),
                           torch.arange(48).float().reshape(6, 8))
        # The MTP block's MLP, ten rows wide here.
        assert model.parameters["mtp/layer/mlp/up"].shape == (10, 8)
        assert model.parameters["mtp/layer/mlp/down"].inputs == ("mtp/layer/mlp/product",)
        assert model.components["mtp"]["config"] == {
            "architectures": ["Qwen4MTP"], "rope_theta": 10000000.0, "intermediate_size": 10}
        # The model describes its table; its rows come with --components ...,ngram.
        assert "text/layers/0/ple/convolution" in names and "ngram/table" not in names
        assert set(model.components) == {"text", "mtp", "ngram"}
        assert model.config["architectures"] == ["Qwen4ForCausalLM"]


def test_build_model_without_ple_has_no_table(tmp_path):
    from tools.convert.qwen4_exp import build_model
    from tools.convert.sources.safetensors import SafetensorsSource

    root = tmp_path / "source"
    _write_source(root, _tiny_config(ple=False))
    with SafetensorsSource(root) as source:
        model = build_model(source, components=("text",))
        assert set(model.components) == {"text"}
        assert not any("/ple/" in name for name in model.parameters)
        with pytest.raises(ValueError, match="no n-gram table"):
            build_model(source, components=("text", "ngram"))


def test_dense_recipe_formats(tmp_path):
    from tools.convert.official_recipes import RECIPES
    from tools.convert.qwen4_exp import build_model
    from tools.convert.recipe import Recipe
    from tools.convert.sources.safetensors import SafetensorsSource

    root = tmp_path / "source"
    config = _tiny_config(ple=False)
    # HC8 blocks need whole 32-value rows of the hyper-connection matrices.
    config["text_config"].update(hidden_size=32, hc_lowrank=32)
    _write_source(root, config)
    with SafetensorsSource(root) as source:
        model = build_model(source, components=("text", "mtp"))
        recipe = Recipe(model)
        RECIPES["qwen4_dense"](model, recipe, {})
    formats = {name: {s.format for s in selections}
               for name, selections in recipe.selections.items()}
    expected = {
        "text/token_embedding": "q8_g32_fp16", "text/output_head": "q6_g64_fp16",
        "text/layers/0/mlp/gate": "q4_g64_fp16", "text/layers/0/mlp/up": "q4_g64_fp16",
        "text/layers/0/mlp/down": "q5_g64_fp16", "text/layers/0/gdn/query": "q4_g64_fp16",
        "text/layers/0/gdn/output": "q5_g64_fp16", "text/layers/1/attention/key": "q4_g64_fp16",
        "text/layers/1/attention/output": "q5_g64_fp16",
        "text/layers/0/attn_hc/down": "gguf_q8_0", "text/layers/1/mlp_hc/inject": "gguf_q8_0",
        "text/final_mixer/up": "gguf_q8_0", "mtp/layer/attn_hc/up": "gguf_q8_0",
        "mtp/layer/mlp/down": "q8_g32_fp16", "mtp/layer/attention/query": "q8_g32_fp16",
    }
    for name, format in expected.items():
        assert formats[name] == {format}, name
    # Kept as stored.
    for name in ("text/layers/0/attn_hc/norm", "text/layers/1/indexer/query",
                 "text/layers/0/gdn/a_projection", "mtp/fc_hidden"):
        assert formats[name] == {"bf16"}, name
    flash = Recipe(model)
    with pytest.raises(ValueError, match="Flash-Next mathematics"):
        RECIPES["qwen3_8_flash_next"](model, flash, {})


def test_hc8_blocks_decode_within_half_a_step():
    from tools.convert.gguf_blocks import dequantize_q8_0
    from tools.convert.model import Parameter
    from tools.convert.official_recipes import _hc8_source
    from tools.convert.sources.logical import array_source

    values = (torch.arange(4 * 64).float() * 0.37).sin().reshape(4, 64).bfloat16()
    parameter = Parameter("text/layers/0/attn_hc/down", (4, 64), array_source(values, "hc"),
                          inputs=("text/layers/0/attn_hc/normalized",))
    source = _hc8_source(parameter)
    rows = source.read_encoded(1, 3)
    assert rows.format == "gguf_q8_0" and tuple(rows.codes.shape) == (2, 2 * 34)
    decoded = torch.from_numpy(dequantize_q8_0(rows.codes.numpy()).reshape(2, 64))
    blocks = values[1:3].float().reshape(2, 2, 32)
    step = blocks.abs().amax(-1, keepdim=True) / 127
    assert ((decoded.reshape(2, 2, 32) - blocks).abs() <= step * 0.5 + 1e-3 * step).all()
    assert torch.equal(source.values(64, 192).reshape(2, 64), decoded)
