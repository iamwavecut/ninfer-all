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
           "target_layer_ids": [0, 1]}
CONFIG = {"hidden_size": 32, "vocab_size": 4}


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


def _drafter_file(path, embedding_type: int, embedding: bytes) -> None:
    """The drafter's tensors up to its Markov embedding [vocab, rank]."""

    h, vocab, rank = CONFIG["hidden_size"], CONFIG["vocab_size"], DRAFTER["markov_rank"]
    ones = lambda *shape: np.ones(shape, dtype=np.float32).tobytes()  # noqa: E731
    write_gguf(path, {"general.architecture": "dflash"}, [
        ("dspark_fuse.weight", (2, 2), TYPE_F32, ones(2, 2)),
        ("dspark_norm.weight", (h,), TYPE_F32, ones(h)),
        ("output_norm.weight", (h,), TYPE_F32, ones(h)),
        ("mask_embd.weight", (h,), TYPE_F32, ones(h)),
        ("markov_w1.weight", (vocab, rank), embedding_type, embedding),
    ])


def test_the_drafters_markov_embedding_keeps_its_bf16_words(tmp_path):
    words = np.random.default_rng(5).integers(0, 1 << 16, size=(4, 32), dtype=np.uint16)
    words[(words & 0x7F80) == 0x7F80] = 0x3F80  # no NaN or infinity
    path = tmp_path / "bf16.gguf"
    _drafter_file(path, TYPE_BF16, words.tobytes())
    model = Model(components={})
    # The file ends after the embedding: the drafter's next tensor is missing.
    with GGUFFile(path) as gguf, pytest.raises(KeyError, match="joint_proj"):
        inkling2._dspark(model, gguf, DRAFTER, CONFIG)
    parameter = model.parameters["dspark/markov_embed"]
    assert parameter.direct_format == "bf16"
    got = parameter.source.values(0, 128).reshape(4, 32).to(torch.bfloat16)
    assert np.array_equal(got.view(torch.int16).numpy().view(np.uint16), words)


def test_a_drafter_embedding_that_bf16_does_not_hold_is_refused(tmp_path):
    blocks = _q8_0_blocks(np.random.default_rng(6), 4, 32)
    blocks[0, 0, 2] = 0x7B  # 123 * d needs more than BF16's 8 significant bits
    blocks[0, 0, :2] = np.array([0.0123], dtype=np.float16).view(np.uint8)
    path = tmp_path / "q8.gguf"
    _drafter_file(path, TYPE_Q8_0, blocks.tobytes())
    with GGUFFile(path) as gguf, pytest.raises(ValueError, match="not BF16 values"):
        inkling2._dspark(Model(components={}), gguf, DRAFTER, CONFIG)
