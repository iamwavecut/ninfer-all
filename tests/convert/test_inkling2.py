from __future__ import annotations

import json
import struct

import numpy as np
import pytest
import torch

from tools.convert import inkling2
from tools.convert.model import Model
from tools.convert.sources.gguf import TYPE_BF16, TYPE_F32, GGUFFile, write_gguf

TYPE_Q8_0, TYPE_Q4_K = 8, 12
DRAFTER = {"num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 32,
           "intermediate_size": 32, "markov_rank": 32, "num_hidden_layers": 1,
           "target_layer_ids": [0, 1], "block_size": 3}
CONFIG = {"hidden_size": 32, "vocab_size": 8, "unpadded_vocab_size": 4}


def _q8_0_blocks(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    blocks = rng.integers(0, 256, size=(rows, columns // 32, 34), dtype=np.uint8)
    scales = rng.uniform(0.01, 0.02, size=(rows, columns // 32)).astype(np.float16)
    blocks[..., :2] = scales[..., None].view(np.uint8)
    return blocks


def _q8_0_values(blocks: np.ndarray) -> np.ndarray:
    """ggml Q8_0: binary16 d times each signed code byte, exact in float64."""

    d = blocks[..., :2].copy().view(np.float16).astype(np.float64)
    codes = blocks[..., 2:].copy().view(np.int8).astype(np.float64)
    return (codes * d).reshape(-1)


def _q4_k_scale_min(scales: list[int], j: int) -> tuple[int, int]:
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    return ((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4))


def _q4_k_values(blocks: np.ndarray) -> np.ndarray:
    """ggml dequantize_row_q4_K in float64: each value d * sc * q - dmin * m, exact there."""

    values = []
    for block in blocks.reshape(-1, 144):
        d = float(block[0:2].copy().view(np.float16)[0])
        dmin = float(block[2:4].copy().view(np.float16)[0])
        scales, codes = [int(x) for x in block[4:16]], block[16:144]
        for j in range(4):
            low_scale, low_min = _q4_k_scale_min(scales, 2 * j)
            high_scale, high_min = _q4_k_scale_min(scales, 2 * j + 1)
            quarter = codes[32 * j:32 * (j + 1)]
            values += [d * low_scale * (int(q) & 0xF) - dmin * low_min for q in quarter]
            values += [d * high_scale * (int(q) >> 4) - dmin * high_min for q in quarter]
    return np.array(values)


def test_a_q4_k_sink_projection_is_its_blocks_values_rounded_once(tmp_path):
    rng = np.random.default_rng(3)
    heads, h = 4, 512
    blocks = rng.integers(0, 256, size=(heads, h // 256, 144), dtype=np.uint8)
    for offset in (0, 2):
        scale = rng.uniform(0.001, 0.01, size=(heads, h // 256)).astype(np.float16)
        blocks[..., offset:offset + 2] = scale[..., None].view(np.uint8)
    path = tmp_path / "sink.gguf"
    name = "blk.0.attn_sink_proj.weight"
    write_gguf(path, {"general.architecture": "inkling2"},
               [(name, (heads, h), TYPE_Q4_K, blocks.tobytes())])
    with GGUFFile(path) as gguf:
        got = inkling2._direct(gguf, name, (heads, h)).values(0, heads * h).numpy()
        with pytest.raises(ValueError, match="stored"):
            inkling2._direct(gguf, name, (heads, h // 2))
    # d * sc * q and dmin * m are exact in FP32; only their difference rounds.
    assert np.array_equal(got, _q4_k_values(blocks).astype(np.float32))


def test_a_q8_0_tap_projection_decodes_exactly(tmp_path):
    rng = np.random.default_rng(4)
    blocks = _q8_0_blocks(rng, 4, 64)
    path = tmp_path / "tap.gguf"
    name = "blk.0.loop_msg_query.weight"
    write_gguf(path, {"general.architecture": "inkling2"},
               [(name, (4, 64), TYPE_Q8_0, blocks.tobytes())])
    with GGUFFile(path) as gguf:
        got = inkling2._direct(gguf, name, (4, 64)).values(0, 256).numpy()
    assert np.array_equal(got.astype(np.float64), _q8_0_values(blocks))


def _f32(values) -> bytes:
    return np.ascontiguousarray(values, dtype=np.float32).tobytes()


def _bf16_words(rng: np.random.Generator, shape) -> np.ndarray:
    words = rng.integers(0, 1 << 16, size=shape, dtype=np.uint16)
    words[(words & 0x7F80) == 0x7F80] = 0x3F80  # no NaN or infinity
    return words


def _drafter_tensors(rng: np.random.Generator, **replace) -> dict:
    """llama.cpp's drafter tensors in `_dspark`'s order, up to the first layer's tensors."""

    h, vocab, rank = CONFIG["hidden_size"], CONFIG["unpadded_vocab_size"], DRAFTER["markov_rank"]
    taps, masks = len(DRAFTER["target_layer_ids"]), DRAFTER["block_size"] - 1
    mixes = rng.uniform(0.1, 1.0, size=(2, taps))
    mixes /= mixes.sum(1, keepdims=True)
    mask = rng.normal(size=h)
    tensors = {
        "blk.0.ctx_mix.weight": ((taps,), TYPE_F32, _f32(mixes[0])),
        "markov_ctx_mix.weight": ((taps,), TYPE_F32, _f32(mixes[1])),
        "mask_embd.weight": ((masks, h), TYPE_F32, _f32(np.tile(mask, (masks, 1)))),
        "token_embd_norm.weight": ((h,), TYPE_F32, _f32(np.linspace(0.5, 1.5, h))),
        "markov_ctx_norm.weight": ((h,), TYPE_F32, _f32(rng.normal(size=h))),
        "output_norm.weight": ((h,), TYPE_F32, _f32(rng.normal(size=h))),
        "markov_w1.weight": ((vocab, rank), TYPE_BF16, _bf16_words(rng, (vocab, rank)).tobytes()),
        "joint_proj.weight": ((3 * rank, 2 * rank + 2 * h), TYPE_BF16,
                              _bf16_words(rng, (3 * rank, 2 * rank + 2 * h)).tobytes()),
        "joint_proj.bias": ((3 * rank,), TYPE_F32, _f32(rng.normal(size=3 * rank))),
        "markov_w2.weight": ((vocab, rank), TYPE_BF16, _bf16_words(rng, (vocab, rank)).tobytes()),
        "markov_w2.bias": ((vocab,), TYPE_F32, _f32(rng.normal(size=vocab))),
    }
    tensors.update(replace)
    return tensors


def _write(path, tensors: dict) -> None:
    write_gguf(path, {"general.architecture": "dflash"},
               [(name, shape, type_id, data) for name, (shape, type_id, data) in tensors.items()])


def _dspark(path) -> Model:
    """`_dspark` with the file as drafter and model (it carries the model's embedding norm); the
    file ends before the first layer's tensors."""

    model = Model(components={})
    with GGUFFile(path) as gguf, pytest.raises(KeyError, match="blk.0.ctx_norm"):
        inkling2._dspark(model, gguf, DRAFTER, CONFIG, gguf)
    return model


def _words(parameter, shape) -> np.ndarray:
    values = parameter.source.values(0, int(np.prod(shape))).reshape(shape).to(torch.bfloat16)
    return values.view(torch.int16).numpy().view(np.uint16)


def test_the_drafters_bf16_markov_tables_and_mixes_are_read_as_stored(tmp_path):
    tensors = _drafter_tensors(np.random.default_rng(5))
    path = tmp_path / "drafter.gguf"
    _write(path, tensors)
    params = _dspark(path).parameters
    vocab, rank, h = CONFIG["unpadded_vocab_size"], DRAFTER["markov_rank"], CONFIG["hidden_size"]
    for name, tensor in (("dspark/markov_embed", "markov_w1.weight"),
                         ("dspark/markov_out", "markov_w2.weight")):
        assert params[name].direct_format == "bf16"
        want = np.frombuffer(tensors[tensor][2], dtype=np.uint16).reshape(vocab, rank)
        assert np.array_equal(_words(params[name], (vocab, rank)), want)
    bias = params["dspark/markov_out_bias"].source.values(0, vocab).numpy()
    assert np.array_equal(bias, np.frombuffer(tensors["markov_w2.bias"][2], dtype=np.float32))
    mixes = np.stack([np.frombuffer(tensors[t][2], dtype=np.float32)
                      for t in ("blk.0.ctx_mix.weight", "markov_ctx_mix.weight")])
    assert np.array_equal(params["dspark/mix"].source.values(0, mixes.size).numpy(),
                          mixes.reshape(-1))
    mask = np.frombuffer(tensors["mask_embd.weight"][2], dtype=np.float32)[:h]
    assert np.array_equal(params["dspark/mask_embedding"].source.values(0, h).numpy(), mask)


def test_a_drafter_table_that_bf16_does_not_hold_is_refused(tmp_path):
    rows, rank = CONFIG["unpadded_vocab_size"], DRAFTER["markov_rank"]
    blocks = _q8_0_blocks(np.random.default_rng(6), rows, rank)
    blocks[0, 0, 2] = 0x7B  # 123 * d needs more than BF16's 8 significant bits
    blocks[0, 0, :2] = np.array([0.0123], dtype=np.float16).view(np.uint8)
    path = tmp_path / "q8.gguf"
    _write(path, _drafter_tensors(np.random.default_rng(6), **{
        "markov_w1.weight": ((rows, rank), TYPE_Q8_0, blocks.tobytes())}))
    with GGUFFile(path) as gguf, pytest.raises(ValueError, match="not BF16 values"):
        inkling2._dspark(Model(components={}), gguf, DRAFTER, CONFIG, gguf)


def _not_softmax(rng):
    return {"blk.0.ctx_mix.weight": ((2,), TYPE_F32, _f32([0.6875, -0.5]))}


def _two_masks(rng):
    masks = rng.normal(size=(2, CONFIG["hidden_size"]))
    return {"mask_embd.weight": ((2, CONFIG["hidden_size"]), TYPE_F32, _f32(masks))}


@pytest.mark.parametrize("change, message", [
    (_not_softmax, "not softmax rows"),
    (_two_masks, "mask rows differ"),
])
def test_a_drafter_the_release_does_not_describe_is_refused(tmp_path, change, message):
    rng = np.random.default_rng(7)
    path = tmp_path / "drafter.gguf"
    _write(path, _drafter_tensors(rng, **change(rng)))
    with GGUFFile(path) as gguf, pytest.raises(ValueError, match=message):
        inkling2._dspark(Model(components={}), gguf, DRAFTER, CONFIG, gguf)


def test_a_drafter_of_another_models_embedding_norm_is_refused(tmp_path):
    rng = np.random.default_rng(8)
    drafter, target = tmp_path / "drafter.gguf", tmp_path / "target.gguf"
    _write(drafter, _drafter_tensors(rng))
    h = CONFIG["hidden_size"]
    write_gguf(target, {"general.architecture": "inkling2_loop"},
               [("token_embd_norm.weight", (h,), TYPE_F32, _f32(np.linspace(0.5, 1.6, h)))])
    with GGUFFile(drafter) as d, GGUFFile(target) as t, \
            pytest.raises(ValueError, match="embedding norm is not the model's"):
        inkling2._dspark(Model(components={}), d, DRAFTER, CONFIG, t)


def test_the_earlier_drafter_layout_is_refused(tmp_path):
    path = tmp_path / "old.gguf"
    write_gguf(path, {"general.architecture": "dflash"},
               [("dspark_fuse.weight", (2, 2), TYPE_F32, _f32(np.zeros((2, 2))))])
    with GGUFFile(path) as gguf, pytest.raises(ValueError, match="Q8_0_Old.gguf"):
        inkling2.dspark_config({}, CONFIG, gguf)


# incoai's DFlash2 drafter: the release's config.json and a small safetensors directory.
RELEASE_DFLASH2 = {
    "architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "hidden_act": "silu",
    "attention_bias": False, "is_causal": False, "hidden_size": 2304, "intermediate_size": 6912,
    "num_hidden_layers": 5, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
    "vocab_size": 201024, "num_target_layers": 48, "sliding_window": 2048, "rms_norm_eps": 1e-6,
    "layer_types": ["sliding_attention"] * 5,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
                      "final_logit_softcapping": 30.0, "mask_token_id": 200056,
                      "selector_rank": 256, "selector_softcap_mode": "after_correction",
                      "selector_top_k": 16, "target_layer_ids": [5, 23, 41, 48]},
}
LOOPED = {"hidden_size": 2304, "vocab_size": 201024, "unpadded_vocab_size": 200058,
          "num_hidden_layers": 24,
          "loop": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64}}


def _release(**changes) -> dict:
    config = json.loads(json.dumps(RELEASE_DFLASH2))
    for key, value in changes.items():
        if key in config["dflash_config"]:
            config["dflash_config"][key] = value
        else:
            config[key] = value
    return config


def test_the_dflash2_config_is_the_releases_with_the_final_norm_tap_counted():
    out = inkling2.dflash2_config(_release(), LOOPED)
    assert out["architectures"] == ["Inkling2DFlash2"]
    assert out["model_type"] == "inkling2_dflash2"
    assert out["target_layer_ids"] == [5, 23, 41, 48]
    # 48 visits' inputs and the final norm's rows.
    assert out["num_target_layers"] == 49
    assert out["selector_softcap"] == 30.0
    assert (out["block_size"], out["sliding_window"], out["rope_theta"]) == (8, 2048, 10000.0)
    assert (out["conv_kernel_size"], out["conv_group_size"]) == (2, 16)
    assert (out["selector_rank"], out["selector_top_k"], out["mask_token_id"]) == (256, 16, 200056)


@pytest.mark.parametrize("changes, message", [
    ({"selector_softcap_mode": "before_correction"}, "after_correction"),
    ({"target_layer_ids": [5, 23, 41]}, "final hidden"),
    ({"target_layer_ids": [0, 23, 48]}, "after the first"),
    ({"num_target_layers": 24}, "48 layer visits"),
    ({"vocab_size": 200064}, "vocab_size"),
    ({"mask_token_id": 200100}, "scored vocabulary"),
    ({"layer_types": ["full_attention"] + ["sliding_attention"] * 4}, "every layer must slide"),
])
def test_a_dflash2_drafter_the_release_does_not_describe_is_refused(changes, message):
    with pytest.raises(ValueError, match=message):
        inkling2.dflash2_config(_release(**changes), LOOPED)


def _write_safetensors(root, tensors: dict) -> None:
    """BF16 words [shape] by name, in one model.safetensors."""

    header, data, offset = {}, b"", 0
    for name, words in tensors.items():
        raw = words.astype("<u2").tobytes()
        header[name] = {"dtype": "BF16", "shape": list(words.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        data += raw
        offset += len(raw)
    encoded = json.dumps(header).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


SMALL_DFLASH2 = {"hidden_size": 32, "head_dim": 8, "num_attention_heads": 4,
                 "num_key_value_heads": 2, "intermediate_size": 48, "selector_rank": 16,
                 "vocab_size": 8, "conv_kernel_size": 2, "conv_group_size": 16,
                 "num_hidden_layers": 1, "target_layer_ids": [1, 2, 3]}


def _small_dflash2_tensors(rng) -> dict:
    h, d, q = 32, 8, 32
    kv, inner, rank, vocab, coefficients = 16, 48, 16, 8, 2 * 2 * (32 // 16)
    shapes = {
        "fc.weight": (h, 3 * h), "hidden_norm.weight": (h,), "norm.weight": (h,),
        "candidate_selector.hidden_projection.weight": (rank, h),
        "candidate_selector.predecessor_codebook": (vocab, rank),
        "candidate_selector.successor_codebook": (vocab, rank),
        "layers.0.input_layernorm.weight": (h,),
        "layers.0.post_attention_layernorm.weight": (h,),
        "layers.0.self_attn.q_norm.weight": (d,), "layers.0.self_attn.k_norm.weight": (d,),
        "layers.0.self_attn.q_proj.weight": (q, h), "layers.0.self_attn.k_proj.weight": (kv, h),
        "layers.0.self_attn.v_proj.weight": (kv, h), "layers.0.self_attn.o_proj.weight": (h, q),
        "layers.0.mlp.gate_proj.weight": (inner, h), "layers.0.mlp.up_proj.weight": (inner, h),
        "layers.0.mlp.down_proj.weight": (h, inner),
        "layers.0.attention_conv.base_kernel": (2, 2, h),
        "layers.0.attention_conv.kernel_projection.weight": (coefficients, h),
        "layers.0.mlp_conv.base_kernel": (2, 2, h),
        "layers.0.mlp_conv.kernel_projection.weight": (coefficients, h),
    }
    return {name: _bf16_words(rng, shape) for name, shape in shapes.items()}


class _Recipe:
    def __init__(self):
        self.assigned = {}

    def assign(self, name, **spec):
        self.assigned[name] = spec

    def group(self, names):
        raise AssertionError(f"unexpected group {names}")


def test_the_dflash2_drafter_is_read_as_stored_and_its_projections_become_q8_rows(tmp_path):
    from tools.convert.sources.safetensors import SafetensorsSource

    tensors = _small_dflash2_tensors(np.random.default_rng(9))
    _write_safetensors(tmp_path, tensors)
    model = Model(components={"text": {"config": {
        "architectures": [inkling2.TEXT_ARCHITECTURE], "num_hidden_layers": 0}}})
    with SafetensorsSource(tmp_path) as store:
        inkling2._dflash2(model, store, SMALL_DFLASH2)
        params = model.parameters
        for name, tensor in (("dflash2/feature_projection", "fc.weight"),
                             ("dflash2/candidate_selector/successor_codebook",
                              "candidate_selector.successor_codebook"),
                             ("dflash2/layers/0/mlp_conv/base_kernel",
                              "layers.0.mlp_conv.base_kernel"),
                             ("dflash2/layers/0/attention/key", "layers.0.self_attn.k_proj.weight")):
            assert np.array_equal(_words(params[name], tensors[tensor].shape), tensors[tensor])
        assert params["dflash2/feature_projection"].inputs == ("dflash2/target_features",)
        assert params["dflash2/layers/0/attention/output"].inputs == (
            "dflash2/layers/0/attention/output_input",)
        assert params["dflash2/layers/0/mlp_conv/kernel_projection"].inputs == (
            "dflash2/layers/0/mlp_norm",)
        assert params["dflash2/candidate_selector/hidden_projection"].inputs == (
            "dflash2/final_hidden",)
        recipe = _Recipe()
        inkling2.inkling2_gguf(model, recipe, {})
    formats = {name: spec["format"] for name, spec in recipe.assigned.items()}
    assert set(formats) == set(params)
    q8 = {name for name, f in formats.items() if f == "q8_g32_fp16"}
    assert q8 == {"dflash2/feature_projection"} | {
        "dflash2/layers/0/" + role for role in (
            "attention/query", "attention/key", "attention/value", "attention/output",
            "mlp/gate", "mlp/up", "mlp/down")}
    assert all(formats[name] == "bf16" for name in set(params) - q8)
    assert recipe.assigned["dflash2/candidate_selector/hidden_projection"][
        "activation_policy"] == "A16Only"


def test_a_dflash2_tensor_stored_wider_than_bf16_is_refused(tmp_path):
    from tools.convert.sources.safetensors import SafetensorsSource

    tensors = _small_dflash2_tensors(np.random.default_rng(10))
    _write_safetensors(tmp_path, tensors)
    path = tmp_path / "model.safetensors"
    raw = path.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + length])
    header["norm.weight"]["dtype"] = "F16"
    encoded = json.dumps(header).encode().ljust(length)
    path.write_bytes(raw[:8] + encoded + raw[8 + length:])
    with SafetensorsSource(tmp_path) as store, pytest.raises(ValueError, match="expected BF16"):
        inkling2._dflash2(Model(components={}), store, SMALL_DFLASH2)


def _mxfp8_values(rng: np.random.Generator, rows: int, k: int) -> torch.Tensor:
    """MXFP8 values as BF16 (which holds each exactly): finite E4M3 codes, subnormals included,
    times a power of two per 32 columns."""

    codes = rng.integers(0, 256, size=(rows, k), dtype=np.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x38  # no NaN code
    exponents = rng.integers(-12, 4, size=(rows, k // 32))
    values = torch.ldexp(torch.from_numpy(codes).view(torch.float8_e4m3fn).float(),
                         torch.from_numpy(np.repeat(exponents, 32, axis=1)))
    return values.to(torch.bfloat16)


def _banks(path, gate_up: torch.Tensor, down: torch.Tensor) -> None:
    tensors = {}
    for name, values in (("blk.0.ffn_gate_up_exps.weight", gate_up),
                         ("blk.0.ffn_down_exps.weight", down)):
        words = values.view(torch.int16).numpy().view(np.uint16)
        tensors[name] = (tuple(words.shape), TYPE_BF16, words.tobytes())
    _write(path, tensors)


def test_exact_bf16_expert_banks_are_stored_as_mxfp8_rows(tmp_path):
    rng = np.random.default_rng(7)
    gate_up, down = _mxfp8_values(rng, 512, 128), _mxfp8_values(rng, 256, 128)
    _banks(tmp_path / "banks.gguf", gate_up, down)
    with GGUFFile(tmp_path / "banks.gguf") as gguf:
        sources = inkling2._expert_banks(gguf, "blk.0.ffn_gate_up_exps.weight", (512, 128),
                                         "blk.0.ffn_down_exps.weight", (256, 128))
        for source, values in zip(sources, (gate_up, down)):
            rows = source.read_encoded(0, values.shape[0])
            assert rows.format == "mxfp8" and rows.packed
            assert rows.codes.dtype == torch.uint8 and rows.scales.dtype == torch.uint8
            decoded = torch.ldexp(rows.codes.view(torch.float8_e4m3fn).float(),
                                  rows.scales.repeat_interleave(32, dim=1).int() - 127)
            assert torch.equal(decoded, values.float())
            assert torch.equal(source.values().reshape(values.shape), values)


def test_a_layer_keeps_both_banks_bf16_when_one_is_not_exact_mxfp8(tmp_path):
    rng = np.random.default_rng(8)
    gate_up, down = _mxfp8_values(rng, 512, 128), _mxfp8_values(rng, 256, 128)
    down[3, 5] = 1.0078125  # eight significant bits: no E4M3 code holds it
    _banks(tmp_path / "banks.gguf", gate_up, down)
    with GGUFFile(tmp_path / "banks.gguf") as gguf:
        sources = inkling2._expert_banks(gguf, "blk.0.ffn_gate_up_exps.weight", (512, 128),
                                         "blk.0.ffn_down_exps.weight", (256, 128))
        assert all(source.read_encoded is None for source in sources)


def test_mxfp8_row_split_payloads_decode_to_their_values():
    from tools.artifact.codecs.row_split import dequantize_row_split
    from tools.artifact.formats import MXFP8
    from tools.artifact.layouts import row_split_geometry

    rng = np.random.default_rng(9)
    values = _mxfp8_values(rng, 6, 256)
    codes, scales = inkling2.encode_mxfp8_exact(values)
    g = row_split_geometry(MXFP8, (6, 256))
    payload = bytearray(g.payload_bytes)
    payload[g.base_offset:g.base_offset + codes.numel()] = codes.numpy().tobytes()
    payload[g.scale_offset:g.scale_offset + scales.numel()] = scales.numpy().tobytes()
    decoded = dequantize_row_split(bytes(payload), MXFP8, (6, 256), dtype=torch.float32)
    assert torch.equal(decoded, values.float())


def test_grouped_quantization_refuses_mxfp8():
    from tools.convert.quantization.groupwise import quantize_matrix

    with pytest.raises(ValueError, match="mxfp8 stores exact BF16 expert banks"):
        quantize_matrix(torch.ones(4, 128), "mxfp8")


def test_mxfp8_banks_convert_into_an_artifact_that_reads_back_their_values(tmp_path):
    from tools.artifact.codecs.row_split import dequantize_row_split
    from tools.artifact.reader import Artifact
    from tools.convert.methods import import_encoded
    from tools.convert.model import Parameter
    from tools.convert.pipeline import convert
    from tools.convert.recipe import Recipe

    rng = np.random.default_rng(10)
    gate_up, down = _mxfp8_values(rng, 512, 128), _mxfp8_values(rng, 256, 128)
    _banks(tmp_path / "banks.gguf", gate_up, down)
    output = tmp_path / "banks.ninfer"
    with GGUFFile(tmp_path / "banks.gguf") as gguf:
        sources = inkling2._expert_banks(gguf, "blk.0.ffn_gate_up_exps.weight", (512, 128),
                                         "blk.0.ffn_down_exps.weight", (256, 128))
        model = Model({"text": {"config": {}}})
        model.add(Parameter("gate_up", (512, 128), sources[0], inputs=("input",)))
        model.add(Parameter("down", (256, 128), sources[1], inputs=("product",)))
        recipe = Recipe(model)
        for name in ("gate_up", "down"):
            recipe.assign(name, format="mxfp8", method=import_encoded, activation_policy="A16Only")
        convert(model, recipe, output, device="cpu", rows_per_chunk=100)
    with Artifact(output) as artifact:
        for name, values in (("gate_up", gate_up), ("down", down)):
            obj = artifact.directory.bindings[name]["object"]
            decoded = dequantize_row_split(artifact.read_object(obj), "mxfp8", tuple(values.shape),
                                           dtype=torch.float32)
            assert torch.equal(decoded, values.float())
