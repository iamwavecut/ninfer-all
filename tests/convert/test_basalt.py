from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.convert import basalt, edge0, qwen4_exp
from tools.convert.basalt_config import translate
from tools.convert.resources import load_resources
from tools.convert.sources.basalt import BasaltFile, BasaltGGUF, HC8, HEADER, ENTRY, HC_HEADER, HC_ENTRY, mtp_pack
from tools.artifact.reader import Artifact
from tools.convert.sources.logical import array_source


class MemoryPack:
    def __init__(self, data, meta=None, version=1):
        self.data = data
        self.sections = {name: SimpleNamespace(size=len(value)) for name, value in data.items()}
        self.meta = meta or {"gguf_kv": {"general.architecture": "qwen4exp", "qwen4exp.expert_count": 2}, "tensors": {}}
        self.path = "fixture.basalt"
        self.version = version

    def read(self, name, offset=0, size=None):
        value = self.data[name]
        size = len(value) - offset if size is None else size
        assert 0 <= offset <= offset + size <= len(value)
        return np.frombuffer(value[offset:offset + size], dtype=np.uint8).copy()

    def text(self, name):
        return self.data[name].decode()


def test_container_checks_ranges_and_hashes(tmp_path):
    path = tmp_path / "model.basalt"
    sections = {"meta": b'{"tensors":{}}', "gguf": b"unchanged Q8 bytes"}
    with path.open("wb") as f:
        f.write(HEADER.pack(b"BASALTMF", 2, 2, (4 << 20) + len(sections["gguf"]), b""))
        for i, (name, blob) in enumerate(sections.items(), 1):
            f.write(ENTRY.pack(name.encode(), i * (2 << 20), len(blob), hashlib.sha256(blob).digest(), b""))
        for i, blob in enumerate(sections.values(), 1):
            f.seek(i * (2 << 20))
            f.write(blob)
    with BasaltFile(path) as pack:
        assert pack.read("gguf", 1, 4).tobytes() == b"ncha"
        pack.verify("gguf")
        with pytest.raises(ValueError, match="exceeds"):
            pack.read("gguf", 0, 100)
    with path.open("r+b") as f:
        f.seek(4 << 20)
        f.write(b"X")
    with BasaltFile(path) as pack:
        with pytest.raises(ValueError, match="SHA-256"):
            pack.verify("gguf")


def test_effective_dense_overrides_original_and_kind3_narrows():
    raw = np.array([1.0, 1.0003, -0.125, 0.0], dtype="<f4")
    words = np.array([0x3F00, 0xBF00, 0x4000, 0], dtype="<u2")
    meta = {"gguf_kv": {"general.architecture": "qwen4exp", "qwen4exp.expert_count": 2},
            "tensors": {"weight": {"shape": [2, 2], "type_id": 0, "bytes": 16, "offset": 0}}}
    index = "weight 0 4 16 8 0 8 2 2 " + "0 " * 10 + "\nconv 0 3 0 16 0 8 2 2 " + "0 " * 10
    pack = MemoryPack({"gguf": raw.tobytes(), "dense": raw.tobytes() + words.tobytes(),
                       "dense.index": index.encode(), "experts.layout": b"# experts v5:\n", "experts": b""}, meta)
    source = BasaltGGUF(pack)
    assert source.read_bf16_words("weight").reshape(-1).tolist() == words.tolist()
    assert source.info("conv").type_id == 1
    assert source.tensor_bytes("conv", 1, 7).tobytes() == raw.astype("<f2").tobytes()[1:7]


def test_expert_blob_deinterleaving_preserves_quantized_bytes():
    # Q2_0 gate/up/down have equal byte counts for this square product, but different logical rows.
    per = 640 * (2560 // 64) * 18
    payload = b"".join(bytes([i]) * per for i in range(6))
    pack = MemoryPack({"dense.index": b"", "experts.layout": f"# experts v5:\n0 42 42 0 {3 * per}\n".encode(),
                       "experts": payload})
    source = BasaltGGUF(pack)
    assert source.tensor_bytes("blk.0.ffn_up_exps.weight", per - 2, per + 2).tolist() == [1, 1, 4, 4]
    assert source.read_blocks("blk.0.ffn_down_exps.weight", 2560, 2561)[0, :8].tolist() == [5] * 8


def nvfp4_pack(scales=None):
    per = 640 * (2560 // 64) * 36
    scales = np.array([[0.12345679, 1.2345679, 2.3456788, 0],
                       [3.456789, 4.567891, 5.678912, 0]], dtype="<f4") if scales is None else scales
    payload = b"".join(b"".join(bytes([3 * e + r + 1]) * per for r in range(3)) +
                       scales[e].tobytes() for e in range(2))
    return MemoryPack({"dense.index": b"", "experts.layout":
                       f"# strata experts v6:\n0 40 40 0 {3 * per + 16}\n".encode(),
                       "experts": payload}), scales, per


def test_nvfp4_experts_preserve_blocks_and_fp32_scale_bits(tmp_path):
    from tools.convert.model import Model, Parameter
    pack, scales, per = nvfp4_pack()
    source = BasaltGGUF(pack)
    assert source.expert_scales[0].dtype == np.dtype("<f4")
    assert source.expert_scales[0].tobytes() == scales[:, :3].copy().tobytes()
    name = "text/layers/0/moe/expert_scales"
    operand = array_source(torch.from_numpy(source.expert_scales[0]), name)
    model = Model({"text": {"config": {"expert_output_scales": True}}})
    model.add(Parameter(name, operand.shape, operand, direct_format="fp32"))
    output = tmp_path / "scales.ninfer"
    basalt.convert(model, basalt.Recipe(model), output, device="cpu", rows_per_chunk=1)
    with Artifact(output) as artifact:
        obj = artifact.object(artifact.directory.bindings[name]["object"])
        assert obj.format == "fp32"
        assert artifact.read_object(obj.id) == scales[:, :3].copy().tobytes()
    for r, role in enumerate(("gate", "up", "down")):
        name = f"blk.0.ffn_{role}_exps.weight"
        assert source.tensor_bytes(name, per - 2, per + 2).tolist() == [r + 1] * 2 + [r + 4] * 2
        assert source.tensor_bytes(name, 1, 37).tolist() == [r + 1] * 36
        assert source.tensor_bytes(name, 7, 7).size == 0
    raw = bytearray(pack.data["experts"])
    raw[0] = 128
    pack.data["experts"] = bytes(raw)
    with pytest.raises(ValueError, match="bit 7"):
        source.tensor_bytes("blk.0.ffn_gate_exps.weight", 4, 5)


@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan])
def test_nvfp4_expert_scales_must_be_positive_finite(value):
    scales = np.ones((2, 4), dtype="<f4")
    scales[:, 3], scales[1, 1] = 0, value
    pack, _, _ = nvfp4_pack(scales)
    with pytest.raises(ValueError, match="positive finite"):
        BasaltGGUF(pack)


@pytest.mark.parametrize("edit,match", [("padding", "zero"), ("type", "requires NVFP4"),
                                        ("size", "byte count")])
def test_nvfp4_layout_rejects_invalid_tail_and_geometry(edit, match):
    pack, scales, per = nvfp4_pack()
    if edit == "padding":
        scales[0, 3] = 1
        pack, _, _ = nvfp4_pack(scales)
    else:
        pack.data["experts.layout"] = (f"# experts v6:\n0 {8 if edit == 'type' else 40} 40 "
                                        f"0 {3 * per + (0 if edit == 'size' else 16)}\n").encode()
    with pytest.raises(ValueError, match=match):
        BasaltGGUF(pack)


def test_v7_mixed_experts_keep_q6_k_gate_and_distinct_up_type():
    from tools.convert.sources.gguf import TensorInfo
    types = (14, 2, 8)
    sizes = [TensorInfo("fixture", shape, t, 0).nbytes for t, shape in zip(types,
             ((640, 2560), (640, 2560), (2560, 640)))]
    payload = b"".join(bytes([3 * e + r + 1]) * size for e in range(2) for r, size in enumerate(sizes))
    source = BasaltGGUF(MemoryPack({"dense.index": b"", "experts.layout":
        f"# experts v7:\n0 14 8 0 {sum(sizes)} 2\n".encode(), "experts": payload}))
    assert not source.expert_scales
    for r, role in enumerate(("gate", "up", "down")):
        name = f"blk.0.ffn_{role}_exps.weight"
        assert source.info(name).type_id == types[r]
        assert source.tensor_bytes(name, sizes[r] - 2, sizes[r] + 2).tolist() == [r + 1] * 2 + [r + 4] * 2


def hc_pack(name="matrix"):
    codes = np.arange(128, dtype=np.uint8).reshape(2, 64)
    scales = np.zeros((2, 8), dtype="<u2")
    scales[:, :2] = np.array([[0.01, 0.125], [0.0, 2.0]], dtype="<f2").view("<u2")
    blob = codes.tobytes() + scales.tobytes()
    header = HC_HEADER.pack(b"BASALTH8", 1, 1) + HC_ENTRY.pack(name.encode(), 2, 64, 8, 0, 256, len(blob))
    pack = MemoryPack({"hc8": header.ljust(256, b"\0") + blob}, version=2)
    return pack, codes, scales


def test_hc8_reblocks_raw_scale_bits_without_requantization():
    pack, codes, scales = hc_pack()
    hc = HC8(pack)
    out = hc.rows("matrix", 0, 2).reshape(2, 2, 34)
    assert np.array_equal(out[:, :, :2], scales[:, :2].copy().view(np.uint8).reshape(2, 2, 2))
    assert np.array_equal(out[:, :, 2:], codes.reshape(2, 2, 32))
    with pytest.raises(ValueError):
        hc.rows("matrix", 0, 3)


def test_v1_hc8_uses_rounded_scales_and_zero_underflow():
    source = SimpleNamespace(hc=SimpleNamespace(entries={}), pack=SimpleNamespace(version=1))
    values = np.array([1e-12] * 32 + list(np.linspace(-1, 1, 32)), dtype=np.float32).reshape(2, 32)
    logical = basalt.hc_source(source, "matrix", values.shape, lambda first, last: values[first:last])
    rows = logical.read_encoded(0, 2).codes.numpy()
    assert not rows[0].any()
    scale = np.float16(1 / 127)
    assert rows[1, :2].tobytes() == scale.tobytes()
    assert np.array_equal(rows[1, 2:].view(np.int8), np.rint(values[1] / np.float32(scale)).astype(np.int8))


def test_mtp_uses_pack_planes_and_own_norms(monkeypatch):
    monkeypatch.setattr(edge0, "HIDDEN", 64)
    monkeypatch.setattr(edge0, "EXPERT_WIDTH", 64)
    monkeypatch.setattr(edge0, "MTP_EXPERTS", 2)
    gu, dn = 128, 64
    codes_gu = np.arange(gu * 16, dtype=np.uint8).reshape(gu, 16)
    codes_dn = np.full((dn, 16), 67, dtype=np.uint8)
    scales_gu = np.arange(gu, dtype="<u2")
    scales_dn = np.arange(dn, dtype="<u2") + 256
    record = codes_gu.tobytes() + codes_dn.tobytes() + scales_gu.tobytes() + scales_dn.tobytes()
    stored = (torch.arange(64).to(torch.bfloat16).float().numpy() / 64 + 1).astype("<f4")
    pack = MemoryPack({"mtp/dense.index": b"norm f32 1 64 0 256\n", "mtp/dense": stored.tobytes(),
                       "mtp/experts": record * 2})
    mtp = mtp_pack(pack)
    for role, phase in (("gate", 0), ("up", 1)):
        rows = mtp.expert_rows(1, role)
        assert np.array_equal(rows[:, 2:], codes_gu[phase::2])
        assert rows[:, :2].copy().view("<u2").reshape(-1).tolist() == scales_gu[phase::2].tolist()
    assert mtp.expert_rows(0, "down")[:, 2:].tolist() == codes_dn.tolist()
    norm = basalt.stored_norm(mtp.f32("norm", 64), "norm")
    assert torch.equal(norm.values().float().reshape(-1), torch.from_numpy(stored))
    tuned = np.array([1.123456, 1e-9], dtype=np.float32)
    assert np.array_equal(basalt.stored_norm(tuned, "tuned").values().numpy(), tuned)


def test_frontend_and_architecture_are_loadable_resources(tmp_path):
    kv = dict(basalt.q.EXPECTED_HEADER, **{
        "qwen4exp.expert_count": 512, "qwen4exp.context_length": 262144,
        "qwen4exp.attention.layer_norm_rms_epsilon": 1e-6, "qwen4exp.rope.freq_base": 1e7,
        "qwen4exp.ple.eos_token_id": 1, "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen35",
        "tokenizer.ggml.tokens": ["<|endoftext|>", "<|im_end|>", "a", "b"],
        "tokenizer.ggml.token_type": [3, 3, 1, 1], "tokenizer.ggml.merges": [],
        "tokenizer.chat_template": "{{ messages }}"})
    pack = MemoryPack({}, {"gguf_kv": kv})
    config = basalt.model_config(pack)
    basalt.frontend_files(pack, tmp_path, config)
    text = qwen4_exp.text_config(config)
    assert text["num_experts"] == 512 and text["ple_layers"] == [1]
    references, blobs, count, special = load_resources(tmp_path, vocab_size=248320,
        vision_config={"patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2})
    assert count == 4 and special == (0, 1)
    assert (tmp_path / "chat_template.jinja").read_text() == "{{ messages }}"
    image = json.loads((tmp_path / "preprocessor_config.json").read_text())
    assert image["image_mean"] == [0.5] * 3 and image["size"]["longest_edge"] == 4194304
    with basalt.SafetensorsSource(tmp_path) as base:
        model = qwen4_exp.build_model(base, components=("text", "ngram", "mtp", "vision"))
        assert "ngram/table" in model.parameters and "mtp/embedding_norm" in model.parameters
        assert any(name.startswith("vision/") for name in model.parameters)
        assert model.config["norm_unit_offset"] is False
        assert model.parameters["text/layers/1/ple/convolution"].direct_format == "fp32"
        assert model.parameters["mtp/embedding_norm"].direct_format == "fp32"
        assert "text/layers/0/gdn/decay" in model.parameters
        assert "text/layers/0/gdn/a_log" not in model.parameters
        assert "text/layers/0/attn_hc/prefill_inject" in model.parameters
        assert "mtp/layer/mlp_hc/prefill_inject" in model.parameters


def test_q8ple_streams_from_pack_through_artifact_unchanged(tmp_path):
    config = json.loads((Path(__file__).parents[1] / "fixtures/qwen4_exp/config.json").read_text())
    config["text_config"].update(ngram_vocab_size_base=100,
        make_ngram_vocab_size_divisible_by=8, heads_per_ngram=2, ple_embed_dim=640)
    (tmp_path / "config.json").write_text(json.dumps(config))
    descriptor = qwen4_exp.ngram_config(config)
    rows, width = descriptor["rows"], descriptor["row_width"]
    raw = np.random.default_rng(7).integers(0, 256, (rows, width // 32, 34), dtype=np.uint8)
    raw[..., :2] = np.array([0.125], dtype="<f2").view(np.uint8)
    payload = raw.tobytes()
    pack = MemoryPack({"gguf": b"prefix" + payload, "dense.index": b"", "experts": b"",
                       "experts.layout": b"# experts v5:\n"})
    pack.meta["tensors"] = {basalt.q.NGRAM_TENSOR: {"shape": [width, rows], "type_id": 8,
                                                  "offset": 6, "bytes": len(payload)}}
    table = BasaltGGUF(pack)
    with basalt.SafetensorsSource(tmp_path) as base:
        model = qwen4_exp.build_model(base, components=("ngram",))
        recipe = basalt.Recipe(model)
        basalt.q.configure_ngram_gguf(model, recipe, table)
        output = tmp_path / "table.ninfer"
        basalt.convert(model, recipe, output, device="cpu", rows_per_chunk=17, max_file_bytes=12288)
    with Artifact(output) as result:
        obj = result.object(result.directory.bindings["ngram/table"]["object"])
        assert obj.format == "gguf_q8_0" and result.read_object(obj.id) == payload
        assert result.directory.components["ngram"]["config"]["table_sha256"] == hashlib.sha256(payload).hexdigest()


def test_components_never_silently_drop_a_drafter():
    pack = MemoryPack({name: b"" for name in ("meta", "gguf", "dense", "dense.index",
                                             "experts", "experts.layout")})
    assert basalt.components_for(pack) == ("text", "ngram")
    pack.sections["mtp/dense"] = SimpleNamespace(size=0)
    with pytest.raises(ValueError, match="incomplete MTP"):
        basalt.components_for(pack)
    pack.sections.update({"mtp/dense.index": None, "mtp/experts": None, "vision/mmproj": None})
    assert basalt.components_for(pack) == ("text", "ngram", "mtp", "vision")
    pack.sections["dspark/gguf"] = None
    with pytest.raises(ValueError, match="refusing to drop"):
        basalt.components_for(pack)


def test_direct_cast_checks_values_at_the_conversion_boundary():
    values = torch.tensor([1, 0.5, 1.123456, -2], dtype=torch.float32)
    source = basalt.lossless_bf16(array_source(values, "patch"))
    assert torch.equal(source.values(0, 2).float(), values[:2])
    with pytest.raises(ValueError, match="change stored values"):
        source.values(2, 4)


def test_text_scalar_mapping_retains_native_precision(monkeypatch):
    from tools.convert.sources.logical import LogicalSource
    q = basalt.q
    shapes = q.expected_tensors()

    class Source:
        def read_direct(self, name):
            value = -1.123456 if name.endswith("ssm_a") else 1.0009765625 if "ple_conv1d" in name else 1.123456
            return np.full(shapes[name][0], value, dtype=np.float32)

    def matrix(src, name, shape, *args):
        return LogicalSource(shape, name, lambda a, b: torch.zeros(b - a)), False

    monkeypatch.setattr(q, "matrix_source", matrix)
    monkeypatch.setattr(q, "hc_matrix_source", lambda src, name, shape: matrix(src, name, shape)[0])
    config = {"num_hidden_layers": 2, "num_experts": 10,
              "layer_types": ["linear_attention", "full_attention"], "ple_layers": [0],
              "gdn_a_is_log": False}
    mapped = q.text_sources(Source(), config, (1, 3),
        norm_source=lambda src, name, _: basalt.stored_norm(src.read_direct(name), name),
        direct_dtype=torch.float32)
    for name, expected in (("text/layers/0/ple/convolution", 1.0009765625),
                           ("text/layers/0/gdn/convolution", 1.123456),
                           ("text/layers/0/gdn/decay", -1.123456),
                           ("text/layers/1/attention/query_norm", 1.123456),
                           ("text/final_mixer/norm", 1.123456)):
        values = mapped[name][0].values()
        assert values.dtype == torch.float32
        assert np.all(values.numpy() == np.float32(expected))


@pytest.mark.parametrize("kind", [None, 1, 4])
def test_prefill_injection_keeps_original_bf16_beside_hc8(kind):
    pack, _, _ = hc_pack("blk.0.hc_attn_inject.weight")
    name = "blk.0.hc_attn_inject.weight"
    original = torch.linspace(-1, 1, 128).to(torch.bfloat16).reshape(2, 64)
    blob = original.view(torch.int16).numpy().tobytes()
    pack.data.update({"gguf": blob, "dense.index": b"", "experts": b"",
                      "experts.layout": b"# experts v5:\n"})
    if kind is not None:
        words = original.view(torch.int16).numpy().astype("<u2")
        # Low bits deliberately exceed the rounding midpoint: Basalt keeps only the high word.
        dense = ((words.astype("<u4") << 16) | 0xC001).tobytes() if kind == 1 else blob
        pack.data["dense"] = dense
        pack.data["dense.index"] = (
            f"{name} 0 {kind} 0 {len(dense)} 0 8 64 2 " + "0 " * 10).encode()
    pack.sections.update({n: SimpleNamespace(size=len(b)) for n, b in pack.data.items()})
    pack.meta["tensors"] = {name: {"shape": [64, 2], "type_id": 30, "offset": 0, "bytes": len(blob)}}
    source = BasaltGGUF(pack)
    assert source.info(name).type_id == 8
    assert np.array_equal(source.prefill_injection(name), original.float().numpy())
    assert source.tensor_bytes(name).tobytes() == source.hc.rows(name, 0, 2).tobytes()
    assert source._dense_bytes(name, 1, 7).tobytes() == blob[1:7]


def test_kind1_dense_reads_effective_high_words():
    words = np.array([0x3F80C001, 0xBF80FFFF, 0x3F7FFFFF, 0], dtype="<u4")
    index = "weight 0 1 0 16 0 8 2 2 " + "0 " * 10
    pack = MemoryPack({"dense": words.tobytes(), "dense.index": index.encode(),
                      "experts.layout": b"# experts v5:\n", "experts": b""})
    source = BasaltGGUF(pack)
    expected = (words >> 16).astype("<u2")
    assert source.info("weight").type_id == 30
    assert np.array_equal(source.read_bf16_words("weight").reshape(-1), expected)
    assert source.tensor_bytes("weight", 1, 7).tobytes() == expected.tobytes()[1:7]


@pytest.mark.parametrize("type_id,dtype", [(30, torch.bfloat16), (1, torch.float16), (0, torch.float32)])
def test_direct_matrix_rows_are_lazy_and_preserve_gathers(type_id, dtype):
    from tools.convert.sources.gguf import TensorInfo
    values = (torch.arange(12 * 32).reshape(12, 32).float() / 64).to(dtype)
    raw = values.view(torch.uint8).numpy().reshape(-1)
    reads = []

    def tensor_bytes(name, first=0, last=None):
        last = len(raw) if last is None else last
        reads.append((first, last))
        return raw[first:last].copy()

    source = SimpleNamespace(info=lambda _: TensorInfo("matrix", (3, 4, 32), type_id, 0),
                             tensor_bytes=tensor_bytes)
    indices = np.array([7, 2, 2, 8, 1])
    columns = np.roll(np.arange(32), 5)
    logical = basalt.q._bf16_rows(source, "matrix", (5, 32), lambda a, b: indices[a:b], columns)
    assert not reads
    expected = values[indices].to(torch.bfloat16)[:, np.argsort(columns)].reshape(-1)
    assert torch.equal(logical.values(35, 95), expected[35:95])
    assert len(reads) == 1 and reads[0][1] - reads[0][0] == 32 * values.element_size()
    reads.clear()
    assert torch.equal(logical.values(), expected)
    assert sum(last - first for first, last in reads) == 4 * 32 * values.element_size()
    assert logical.values(0, 0).numel() == 0


@pytest.mark.parametrize("scaled", [False, True])
def test_full_shape_recipe_prepares_all_components_lazily(tmp_path, monkeypatch, scaled):
    from tools.convert.sources.gguf import TensorInfo
    from tools.convert.sources.logical import LogicalSource
    from tools.convert.model import Model

    kv = dict(basalt.q.EXPECTED_HEADER, **{
        "qwen4exp.expert_count": 512, "qwen4exp.context_length": 262144,
        "qwen4exp.attention.layer_norm_rms_epsilon": 1e-6, "qwen4exp.rope.freq_base": 1e7,
        "qwen4exp.ple.eos_token_id": 1, "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen35",
        "tokenizer.ggml.tokens": ["<|endoftext|>", "<|im_end|>"],
        "tokenizer.ggml.token_type": [3, 3], "tokenizer.ggml.merges": [],
        "tokenizer.chat_template": "{{ messages }}"})
    pack = MemoryPack({}, {"gguf_kv": kv}, version=2)
    if scaled:
        pack.data["experts.layout"] = b"# experts v6:\n"
        pack.sections["experts.layout"] = SimpleNamespace(size=len(pack.data["experts.layout"]))
    basalt.frontend_files(pack, tmp_path, basalt.model_config(pack))
    reads = []

    class ZeroSource(BasaltGGUF):
        def __init__(self):
            self.pack, self.kv, self.path = pack, kv, pack.path
            self.tensors = {}
            self.expert_scales = {i: np.full((512, 3), 1.2345679, dtype=np.float32)
                                  for i in range(48)} if scaled else {}
            hc_entries = {}
            for name, (shape, kind) in basalt.q.expected_tensors().items():
                hc = "hc_" in name and "norm" not in name
                dtype = 8 if hc else (40 if scaled else 42) if "_exps." in name else (
                    30 if kind == "matrix" or "gate_inp_shexp" in name else 0)
                self.tensors[name] = TensorInfo(name, shape, dtype, 0)
                if hc:
                    hc_entries[name] = (*shape, 0, 0)
            self.hc = SimpleNamespace(entries=hc_entries, rows=lambda n, a, b:
                np.zeros((b - a, self.tensors[n].shape[-1] // 32 * 34), dtype=np.uint8))

        def tensor_bytes(self, name, byte_begin=0, byte_end=None):
            size = self.info(name).nbytes
            end = size if byte_end is None else byte_end
            reads.append(end - byte_begin)
            assert 0 <= byte_begin <= end <= size
            assert end - byte_begin <= 1 << 20
            if name.endswith("ssm_a"):
                return np.full((end - byte_begin) // 4, -1, dtype="<f4").view(np.uint8)
            return np.zeros(end - byte_begin, dtype=np.uint8)

        def prefill_injection(self, name):
            return np.zeros(self.info(name).shape, dtype=np.float32)

    def broadcast(shape, dtype):
        return np.lib.stride_tricks.as_strided(np.zeros(1, dtype=dtype), shape, (0,) * len(shape))

    mtp = SimpleNamespace(
        f32=lambda name, count: np.zeros(count, dtype=np.float32),
        bf16_words=lambda name, shape: broadcast(shape, np.uint16),
        q8_rows=lambda name, shape: broadcast((shape[0], shape[1] // 32 * 34), np.uint8),
        expert_rows=lambda expert, role: broadcast(
            (2560, 640 // 64 * 18) if role == "down" else (640, 2560 // 64 * 18), np.uint8))
    monkeypatch.setattr(basalt, "mtp_pack", lambda _: mtp)
    monkeypatch.setattr(basalt.q, "table_digest", lambda _: "fixture")
    original_source = Model.source

    def vision_source(self, name, store, format=None):
        if name.startswith("vision/"):
            return LogicalSource(self.parameters[name].shape, name, lambda a, b: torch.zeros(b - a))
        return original_source(self, name, store, format)

    monkeypatch.setattr(Model, "source", vision_source)
    with basalt.SafetensorsSource(tmp_path) as base:
        model = qwen4_exp.build_model(base, components=("text", "ngram", "mtp", "vision"))
        source = ZeroSource()
        table = model.components["ngram"]["config"]
        source.tensors[basalt.q.NGRAM_TENSOR] = TensorInfo(
            basalt.q.NGRAM_TENSOR, (table["rows"], table["row_width"]), 8, 0)
        mtp_names = edge0.mtp_sources(mtp, hc_source=lambda name, shape:
            basalt.block_rows_source(shape, name, 8, lambda a, b:
                np.zeros((b - a, shape[1] // 32 * 34), dtype=np.uint8)))
        for name, (operand, _) in mtp_names.items():
            if "/attn_hc/" in name or "/mlp_hc/" in name or "/final_mixer/" in name:
                if name.endswith(("/down", "/up", "/inject")):
                    source.hc.entries["mtp." + operand.label.split("[", 1)[0]] = (*operand.shape, 0, 0)
        source.hc.rows = lambda name, a, b: np.zeros(
            (b - a, source.hc.entries[name][1] // 32 * 34), dtype=np.uint8)
        recipe = basalt.Recipe(model)
        basalt.configure(pack, source, model, recipe, vision=object())
        if scaled:
            name = "text/layers/0/moe/expert_scales"
            assert model.parameters[name].direct_format == "fp32"
            assert recipe.selections[name][0].source.values().numpy().tobytes() == source.expert_scales[0].tobytes()
        assert "mtp/layer/moe/expert_scales" not in model.parameters
        prepared = recipe.prepare(device="cpu", rows_per_chunk=17)
        assert set(prepared.bindings) == set(model.parameters)
        formats = {job.spec.id: job.spec.format for job in prepared.weights}
        for name in ("vision/layers/0/attention/query_bias", "vision/position_embedding",
                     "text/layers/0/gdn/convolution", "mtp/embedding_norm"):
            binding = prepared.bindings[name]
            objects = binding.get("parts", [binding])
            assert all(formats[part["object"]] == "fp32" for part in objects)
        for name in ("text/layers/0/attn_hc/prefill_inject", "mtp/layer/mlp_hc/prefill_inject",
                     "vision/patch_embedding"):
            assert formats[prepared.bindings[name]["object"]] == "bf16"
        assert max(reads) <= 1 << 20


@pytest.mark.parametrize("type_id,dtype", [(0, np.dtype("<f4")), (1, np.dtype("<f2"))])
def test_direct_validation_streams_and_checks_final_chunk(type_id, dtype):
    from tools.convert.sources.gguf import TensorInfo
    info = TensorInfo("matrix", (4097, 512), type_id, 0)
    reads = []

    def tensor_bytes(name, first, last):
        reads.append((first, last))
        values = np.zeros((last - first) // dtype.itemsize, dtype=dtype)
        if last == info.nbytes:
            values[-1] = 1.0009765625
        return values.view(np.uint8)

    source = SimpleNamespace(tensors={"matrix": info}, tensor_bytes=tensor_bytes)
    with pytest.raises(ValueError, match="wider direct operand"):
        basalt.validate_direct_values(source)
    assert len(reads) == 3 and reads[-1][1] == info.nbytes
    assert max(last - first for first, last in reads) <= (1 << 20) * dtype.itemsize


@pytest.mark.parametrize("full,per,total", [(False, 131072, 262144), (True, 262144, 524288)])
def test_config_keeps_fp16_and_slot_semantics(full, per, total):
    source = {"server": {"parallel": 2, "slot-full-context": full, "gpu": 0},
              "engine": {"max-context": 262144, "kv": "fp16", "spec": 7,
                         "resident-experts": True, "expert-cache": "auto", "prefill": "auto"}}
    out, report = translate(source, "model.ninfer", has_mtp=True)
    assert out["kv-dtype"] == "fp16" and out["max-context"] == per and out["kv-capacity"] == total
    assert out["spec"] == "mtp" and out["draft-tokens"] == 6 and out["draft-min-p"] == 0
    assert not report["unresolved"]


def test_config_reports_unknowns_and_preserves_model_precedence():
    config = {"version": 2, "engine": {"kv": "fp16", "parallel": 4},
              "models": [{"model": "x.basalt", "parallel": 2, "kv": "q4", "made-up": 7}]}
    out, report = translate(config, "x.ninfer", has_mtp=False)
    assert out["max-concurrency"] == 2 and out["spec"] == "none"
    assert "kv-dtype" not in out and report["unresolved"]["kv"]["value"] == "q4"
    assert report["unresolved"]["made-up"]["value"] == 7
    with pytest.raises(ValueError, match="integer"):
        translate({"engine": {"parallel": True}}, "x", has_mtp=True)
