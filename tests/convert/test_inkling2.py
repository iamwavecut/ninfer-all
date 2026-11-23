from __future__ import annotations

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
