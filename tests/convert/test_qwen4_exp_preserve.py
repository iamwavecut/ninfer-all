from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
import pytest
from safetensors.torch import save_file
import torch

from tools.artifact.reader import Artifact
from tools.convert.__main__ import main
from tools.convert import qwen4_exp as model_adapter, qwen4_exp_gguf as q
from tools.convert import qwen4_exp_preserve as kept
from tools.convert.qwen4_exp_ngram import nvidia_q8_ngram_source, q8_table_digest
from tools.convert.recipe import Recipe
from tools.convert.sources.gguf import GGUFFile, TensorInfo, TYPE_BF16, TYPE_F32, write_gguf
from tools.convert.sources.safetensors import SafetensorsSource


CONFIG = Path(__file__).parents[1] / "fixtures/qwen4_exp/config.json"
PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def q8_oracle(values, *, hc=False):
    result = bytearray()
    flat = list(map(float, values))
    for first in range(0, len(flat), 32):
        block = flat[first:first + 32]
        scale = f32(max(map(abs, block)) / 127)
        packed = struct.pack("<e", scale)
        if hc:
            stored = struct.unpack("<e", packed)[0]
            codes = [max(-127, min(127, round(f32(x / stored)))) if stored else 0 for x in block]
        else:
            inverse = f32(1 / scale) if scale else 0
            codes = [int(math.copysign(math.floor(abs(f32(x * inverse)) + 0.5), x)) for x in block]
        result.extend(packed + struct.pack("<32b", *codes))
    return bytes(result)


def fp8_fixture(root, scale_value=0.001953125):
    config = json.loads(CONFIG.read_text())
    config["text_config"].update(ngram_vocab_size_base=100,
                                 make_ngram_vocab_size_divisible_by=8,
                                 heads_per_ngram=2, ple_embed_dim=640)
    (root / "config.json").write_text(json.dumps(config))
    descriptor = model_adapter.ngram_config(config)
    rows, width = descriptor["rows"], descriptor["row_width"]
    # Every finite E4M3FN code, zeros, extrema, negative and subnormal values.
    finite = np.array([i for i in range(256) if i not in (127, 255)], dtype=np.uint8)
    codes = np.resize(finite, (rows, width))
    codes[0] = 0
    scale = torch.tensor([scale_value], dtype=torch.bfloat16)
    tensors = {PREFIX + "weight_scale": scale}
    edges = [0, 3, 131, rows]
    for i, (first, last) in enumerate(zip(edges, edges[1:])):
        tensors[PREFIX + f"shard_{i}.weight"] = torch.from_numpy(codes[first:last].copy()).view(torch.float8_e4m3fn)
    path = root / "model-fp8-mtp-ple.safetensors"
    save_file(tensors, path)
    return path, descriptor, codes, float(scale[0])


def decode_e4(word):
    exponent, mantissa = (int(word) >> 3) & 15, int(word) & 7
    value = math.ldexp(mantissa, -9) if exponent == 0 else math.ldexp(1 + mantissa / 8, exponent - 7)
    return -value if word & 128 else value


@pytest.mark.parametrize("scale_value", [0.001953125, 0.0033, 0.123046875])
def test_nvidia_q8_reblock_matches_independent_fp32_oracle(tmp_path, scale_value):
    path, descriptor, codes, scale = fp8_fixture(tmp_path, scale_value)
    values = [f32(decode_e4(b) * scale) for b in codes.reshape(-1)]
    expected = q8_oracle(values)
    if scale_value != 0.001953125:
        assert q8_oracle(torch.tensor(values).bfloat16().float()) != expected
    with SafetensorsSource(path) as store:
        source = nvidia_q8_ngram_source(store, descriptor)
        assert source.read_encoded(0, 0).codes.shape == (0, 170)
        assert source.read_encoded(0, len(codes)).codes.numpy().tobytes() == expected
        for chunk in (1, 7, 131, 65536):
            assert q8_table_digest(source, rows_per_chunk=chunk) == hashlib.sha256(expected).hexdigest()
        assert source.values(159, 161).shape == (2,)
        # Reading a handful of rows never reads the complete 54 GB production table.
        before = store.bytes_read
        source.read_encoded(2, 5)
        assert store.bytes_read - before == 3 * 160


@pytest.mark.parametrize("fault", ["scale", "nan", "gap", "shape"])
def test_nvidia_q8_refuses_ambiguous_or_invalid_source(tmp_path, fault):
    path, descriptor, codes, _ = fp8_fixture(tmp_path)
    with SafetensorsSource(path) as store:
        tensors = {name: store.read_flat(name).reshape(store.describe(name).shape) for name in store.weight_map}
    if fault == "scale":
        tensors[PREFIX + "weight_scale"] = torch.tensor([0.0], dtype=torch.bfloat16)
    elif fault == "nan":
        tensors[PREFIX + "shard_0.weight"].view(torch.uint8)[0, 0] = 127
    elif fault == "gap":
        tensors[PREFIX + "shard_3.weight"] = tensors.pop(PREFIX + "shard_1.weight")
    else:
        tensors[PREFIX + "shard_0.weight"] = tensors[PREFIX + "shard_0.weight"][:, :32].contiguous()
    save_file(tensors, path)
    with SafetensorsSource(path) as store, pytest.raises(ValueError):
        nvidia_q8_ngram_source(store, descriptor).read_encoded(0, 1)


def test_table_cli_preserves_q8_bytes_and_checks_digest_before_writing(tmp_path):
    path, descriptor, codes, scale = fp8_fixture(tmp_path)
    expected = q8_oracle([f32(decode_e4(b) * scale) for b in codes.reshape(-1)])
    digest = hashlib.sha256(expected).hexdigest()
    output = tmp_path / "table.ninfer"
    args = ["--model", str(tmp_path), "--recipe", "qwen3_8_flash_next_gguf_preserve",
            "--components", "ngram", "--source", f"ngram={path}", "--device", "cpu",
            "--rows-per-chunk", "17", "--out", str(output)]
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        main(args + ["--ngram-sha256", "0" * 64])
    assert not output.exists()
    main(args + ["--ngram-sha256", digest])
    with Artifact(output) as artifact:
        table = artifact.directory.components["ngram"]["config"]
        assert table["table_sha256"] == digest and table["format"] == "gguf_q8_0"
        binding = artifact.directory.bindings["ngram/table"]
        assert artifact.read_object(binding["object"]) == expected


def test_preservation_config_and_norm_keep_stored_gamma(tmp_path):
    config = json.loads(CONFIG.read_text())
    original = deepcopy(config)
    changed = kept.model_config(config)
    assert config == original
    parsed = model_adapter.text_config(changed)
    assert parsed["fp32_direct"] and parsed["hc_prefill_inject"]
    assert not parsed["norm_unit_offset"] and not parsed["gdn_a_is_log"]
    values = np.array([1.0003, 0.9997, -0.12345679], dtype="<f4")
    path = tmp_path / "norm.gguf"
    write_gguf(path, {}, [("norm", values.shape, TYPE_F32, values.tobytes())])
    with GGUFFile(path) as gguf:
        assert kept.norm_source(gguf, "norm", True).values().numpy().tobytes() == values.tobytes()
    with pytest.raises(ValueError, match="Flash-Next"):
        kept.model_config({"architectures": ["AnotherFamily"]})


def test_hc8_uses_rounded_scale_and_even_ties(tmp_path):
    values = torch.tensor([0, 127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5] * 8).reshape(2, 32).bfloat16()
    values[1] *= 0.01953125
    path = tmp_path / "hc.gguf"
    write_gguf(path, {}, [("hc", (2, 32), TYPE_BF16, values.view(torch.uint16).numpy().tobytes())])
    with GGUFFile(path) as gguf:
        source = kept.hc8_source(gguf, "hc", (2, 32))
        assert source.read_encoded(0, 0).codes.shape == (0, 34)
        assert source.read_encoded(0, 2).codes.numpy().tobytes() == q8_oracle(values.float().reshape(-1), hc=True)
        assert source.read_encoded(0, 1).codes.numpy().tobytes() != q8_oracle(values[0].float())


def test_direct_projection_refuses_precision_loss(tmp_path):
    path = tmp_path / "wide.gguf"
    values = np.array([[1.0003, -0.125]], dtype="<f4")
    write_gguf(path, {}, [("projection", values.shape, TYPE_F32, values.tobytes())])
    with GGUFFile(path) as gguf, pytest.raises(ValueError, match="exactly representable"):
        kept.validate_direct(gguf)
    with GGUFFile(path) as gguf, pytest.raises(ValueError, match="exactly representable"):
        kept.validate_direct(gguf, compat_bf16=True)


def test_compat_hc_rounds_once_to_bf16_before_hc8(tmp_path):
    raw_values = np.linspace(-0.3, 0.7, 64, dtype=np.float32)
    raw = q8_oracle(raw_values)
    path = tmp_path / "quantized-hc.gguf"
    write_gguf(path, {}, [("hc", (2, 32), 8, raw)])
    from tools.convert.gguf_blocks import dequantize_q8_0
    exact = dequantize_q8_0(np.frombuffer(raw, dtype=np.uint8).reshape(2, 34))
    expected = q8_oracle(torch.from_numpy(exact).bfloat16().float().reshape(-1), hc=True)
    with GGUFFile(path) as gguf:
        with pytest.raises(ValueError, match="explicit --compat-bf16"):
            kept.hc8_source(gguf, "hc", (2, 32))
        assert kept.hc8_source(gguf, "hc", (2, 32), compat_bf16=True).read_encoded(
            0, 2).codes.numpy().tobytes() == expected


def test_bf16_controls_keep_grouped_gdn_head_order(tmp_path):
    weights = torch.arange(48).reshape(48, 1).expand(48, 32).contiguous().bfloat16()
    path = tmp_path / "heads.gguf"
    write_gguf(path, {}, [("control", (48, 32), TYPE_BF16, weights.view(torch.uint16).numpy().tobytes())])
    with GGUFFile(path) as gguf:
        values = kept.bf16_matrix(gguf, "control", (48, 32), select=q.untiled_heads()).rows(0, 48)
        assert values[:, 0].float().tolist() == [(r % 3) * 16 + r // 3 for r in range(48)]


def test_cli_rejects_unsupported_or_malformed_preservation_options():
    with pytest.raises(ValueError, match="requires qwen3_8_flash_next_gguf_preserve"):
        main(["--model", "unused", "--out", "unused", "--recipe", "qwen3_8_27b", "--compat-bf16"])
    with pytest.raises(ValueError, match="64 hexadecimal"):
        main(["--model", "unused", "--out", "unused", "--recipe", "qwen3_8_flash_next_gguf_preserve",
              "--ngram-sha256", "not-a-digest"])


@pytest.mark.parametrize("with_mtp", [False, True])
@pytest.mark.parametrize("with_vision", [False, True])
def test_full_shape_preservation_recipe_binds_native_operands(tmp_path, monkeypatch, with_mtp, with_vision):
    config = json.loads(CONFIG.read_text())
    config["text_config"]["num_experts"] = 10
    config = kept.model_config(config)
    (tmp_path / "config.json").write_text(json.dumps(config))
    from tools.convert import resources
    from tools.convert.model import Model
    from tools.convert.sources.logical import LogicalSource
    monkeypatch.setattr(resources, "load_resources", lambda *a, **kw: ({}, {}, 0, ()))
    original_source = Model.source

    def vision_source(self, name, store, format=None):
        if name.startswith("vision/"):
            return LogicalSource(self.parameters[name].shape, name, lambda first, last: torch.zeros(last - first))
        return original_source(self, name, store, format)

    monkeypatch.setattr(Model, "source", vision_source)
    monkeypatch.setattr(q, "_vision_source", lambda *a: object())

    class Source(GGUFFile):
        def __init__(self, mtp=False):
            self.path = Path("memory.gguf")
            self.kv = dict(q.MTP_HEADER if mtp else q.EXPECTED_HEADER, **{"qwen4exp.expert_count": 10,
                                               "qwen4exp.ple.eos_token_id": config["text_config"]["eos_token_id"]})
            self.tensors = {}
            tensors = q.expected_mtp_tensors(10) if mtp else q.expected_tensors(experts=10)
            for name, (shape, kind) in tensors.items():
                hc = "hc_" in name or name.startswith("output_hc_")
                dtype = TYPE_BF16 if hc and "norm" not in name else 8 if kind == "matrix" else TYPE_F32
                self.tensors[name] = TensorInfo(name, shape, dtype, 0)

        def truncated(self):
            return False

        def read_direct(self, name):
            value = -2.0 if name.endswith("ssm_a") else 1.0003 if "norm" in name or "conv1d" in name else 0.0
            return np.broadcast_to(np.float32(value), self.info(name).shape)

        def tensor_bytes(self, name, first=0, last=None):
            return np.zeros((self.info(name).nbytes if last is None else last) - first, dtype=np.uint8)

        def read_blocks(self, name, first, last):
            return np.zeros((last - first, self.info(name).shape[-1] // 32 * 34), dtype=np.uint8)

    with SafetensorsSource(tmp_path) as base:
        mtp_source = Source(mtp=True)
        components = ("text",) + (("mtp",) if with_mtp else ()) + (("vision",) if with_vision else ())
        model = model_adapter.build_model(base, components=components,
                                           companions={"mtp": mtp_source} if with_mtp else None)
        recipe = Recipe(model)
        monkeypatch.setattr(q, "configure_ngram_gguf", lambda *a: None)
        q.qwen3_8_flash_next_gguf_preserve(model, recipe,
                                         {"gguf": Source(), "ngram": Source(), "mtp": mtp_source},
                                         compat_bf16=True)
        assert set(recipe.selections) == set(model.parameters)
        for role in ("attn_hc", "mlp_hc"):
            prefix = "text/layers/0/" + role + "/"
            assert recipe.selections[prefix + "inject"][0].format == "gguf_q8_0"
            assert recipe.selections[prefix + "prefill_inject"][0].format == "bf16"
        norm = recipe.selections["text/layers/0/attn_hc/norm"][0]
        assert norm.format == "fp32" and float(norm.source.values()[0]) == float(np.float32(1.0003))
        decay = recipe.selections["text/layers/0/gdn/decay"][0]
        assert float(decay.source.values()[0]) == -2.0
        ple = recipe.selections["text/layers/1/ple/convolution"][0]
        gdn = recipe.selections["text/layers/0/gdn/convolution"][0]
        assert ple.format == gdn.format == "fp32"
        assert float(ple.source.values()[0]) == float(np.float16(1.0003))
        assert float(gdn.source.values()[0]) == float(np.float32(1.0003))
        if with_mtp:
            assert recipe.selections["mtp/embedding_norm"][0].format == "fp32"
            assert recipe.selections["mtp/layer/attn_hc/prefill_inject"][0].format == "bf16"
        prepared = recipe.prepare(device="cpu", rows_per_chunk=17)
        assert set(prepared.bindings) == set(model.parameters)
        formats = {job.spec.id: job.spec.format for job in prepared.weights}
        if with_vision:
            for name in ("vision/layers/0/attention/query_bias", "vision/position_embedding"):
                binding = prepared.bindings[name]
                assert all(formats[part["object"]] == "fp32" for part in binding.get("parts", [binding]))
            assert formats[prepared.bindings["vision/patch_embedding"]["object"]] == "bf16"
