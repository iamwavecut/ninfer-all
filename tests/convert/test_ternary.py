from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tools.artifact.codecs.row_split import decode_row_split_codes, encode_row_split
from tools.convert import ternary
from tools.convert.__main__ import SourceInputs
from tools.convert.sources.gguf import (
    TYPE_BF16,
    TYPE_F32,
    TYPE_PQ2_0,
    TYPE_PTQ1_0,
    GGUFFile,
    encode_pq2_0,
    encode_ptq1_0,
    write_gguf,
)


def _pq2_0_file(tmp_path, rows: int, columns: int, seed: int):
    generator = np.random.default_rng(seed)
    values = generator.integers(-1, 2, size=(rows, columns // 128, 128)).astype(np.int8)
    scales = generator.uniform(0.004, 0.03, size=(rows, columns // 128)).astype(
        np.float16
    )
    blocks = encode_pq2_0(values, scales)
    path = tmp_path / "sample.gguf"
    write_gguf(
        path,
        {"general.architecture": "qwen35"},
        [("w", (rows, columns), TYPE_PQ2_0, blocks.tobytes())],
    )
    return path, values, scales


def test_ternary_source_passes_codes_through_and_decodes_exact_values(tmp_path):
    path, values, scales = _pq2_0_file(tmp_path, 12, 256, 3)
    with GGUFFile(path) as gguf:
        source = ternary.ternary_source(gguf, "w", (12, 256), ternary.rows())
        words = source.read_encoded(2, 7)
        assert words.format == "t2_g128_fp16"
        assert torch.equal(words.codes, torch.from_numpy(values[2:7]))
        assert torch.equal(words.scales, torch.from_numpy(scales[2:7]))
        expected = (
            values[2:7].astype(np.float32) * scales[2:7].astype(np.float32)[:, :, None]
        )
        assert torch.equal(
            source.rows(2, 7), torch.from_numpy(expected.reshape(5, 256))
        )
        payload = encode_row_split(words.codes, words.scales, "t2_g128_fp16", (5, 256))
        decoded_scales, decoded_codes = decode_row_split_codes(
            payload, "t2_g128_fp16", (5, 256)
        )
        assert torch.equal(decoded_codes, words.codes)
        assert torch.equal(decoded_scales, words.scales)


def test_selected_rows_follow_the_row_map(tmp_path):
    path, values, _ = _pq2_0_file(tmp_path, 16, 128, 5)
    order = np.array([9, 3, 12, 3], dtype=np.int64)
    with GGUFFile(path) as gguf:
        source = ternary.ternary_source(gguf, "w", (4, 128), lambda b, e: order[b:e])
        assert torch.equal(
            source.read_encoded(0, 4).codes, torch.from_numpy(values[order])
        )
        assert torch.equal(
            source.read_encoded(1, 3).codes, torch.from_numpy(values[order[1:3]])
        )


def test_grouped_value_heads_pair_with_their_key_head():
    permutation = ternary.tiled_to_grouped_permutation()
    assert sorted(permutation.tolist()) == list(range(48))
    for head in range(48):
        tiled = permutation[head]
        assert tiled % 16 == head // 3 and tiled // 16 == head % 3
    tiled = np.arange(48 * 2).reshape(96, 1)
    grouped = ternary.untile(tiled, 2)
    assert grouped[2:4, 0].tolist() == [32, 33]
    rows = ternary.untiled_rows(4096)(128, 130)
    assert rows.tolist() == [4096 + 16 * 128, 4096 + 16 * 128 + 1]


def test_query_and_gate_rows_split_each_interleaved_head():
    query = ternary.attention_rows(False)(250, 260)
    gate = ternary.attention_rows(True)(0, 3)
    assert query.tolist() == [250, 251, 252, 253, 254, 255, 512, 513, 514, 515]
    assert gate.tolist() == [256, 257, 258]


def test_sign_auxiliary_is_a_bf16_vector():
    signs = torch.tensor([1.0, -1.0] * 512)
    value = ternary.sign_auxiliary(signs)
    assert value.format == "bf16" and value.shape == (1024,)
    words = np.frombuffer(value.data, dtype="<u2")
    assert words[0] == 0x3F80 and words[1] == 0xBF80


def test_ptq1_0_rows_keep_their_blocks_and_decode_exact_values(tmp_path):
    generator = np.random.default_rng(7)
    values = generator.integers(-1, 2, size=(10, 3, 128)).astype(np.int8)
    scales = generator.uniform(0.004, 0.03, size=(10, 3)).astype(np.float16)
    blocks = encode_ptq1_0(values, scales)
    path = tmp_path / "ptq1.gguf"
    write_gguf(
        path,
        {"general.architecture": "qwen35"},
        [("w", (10, 384), TYPE_PTQ1_0, blocks.tobytes())],
    )
    order = np.array([8, 1, 4, 4, 0], dtype=np.int64)
    with GGUFFile(path) as gguf:
        source = ternary.ptq1_source(gguf, "w", (5, 384), lambda b, e: order[b:e])
        rows = source.read_encoded(1, 4)
        assert rows.format == "gguf_ptq1_0"
        assert np.array_equal(
            rows.codes.numpy(), blocks[order[1:4]].reshape(3, -1)
        ), "the encoded rows are the stored blocks, byte for byte"
        expected = values[order].astype(np.float32) * scales[order].astype(np.float32)[..., None]
        assert torch.equal(source.rows(0, 5), torch.from_numpy(expected.reshape(5, 384)))


def _control_file(tmp_path, type_id: int, words: np.ndarray):
    path = tmp_path / "control.gguf"
    write_gguf(
        path,
        {"general.architecture": "qwen35"},
        [("c", (48, 2), type_id, words.tobytes())],
    )
    return path


def test_gdn_controls_load_bf16_words_and_bf16_exact_f32_in_grouped_order(tmp_path):
    generator = np.random.default_rng(11)
    bf16 = generator.integers(0, 1 << 16, size=(48, 2), dtype=np.uint16)
    bf16[(bf16 & 0x7F80) == 0x7F80] = 0x3F80  # no NaN or infinity
    grouped = ternary.untile(bf16, 1)
    for type_id, stored in (
        (TYPE_BF16, bf16),
        (TYPE_F32, bf16.astype(np.uint32) << 16),
    ):
        with GGUFFile(_control_file(tmp_path, type_id, stored)) as gguf:
            values = ternary.control_source(gguf, "c").rows(0, 48)
            assert values.dtype == torch.bfloat16
            words = values.contiguous().view(torch.int16).numpy().view(np.uint16)
            assert np.array_equal(words.reshape(48, 2), grouped)


def test_gdn_controls_refuse_f32_values_bf16_cannot_hold(tmp_path):
    words = np.full((48, 2), 0x3F800001, dtype=np.uint32)
    with GGUFFile(_control_file(tmp_path, TYPE_F32, words)) as gguf:
        with pytest.raises(ValueError, match="BF16 cannot hold exactly"):
            ternary.control_source(gguf, "c")


def test_expected_tensors_name_the_kind_and_the_mtp_block():
    plain = ternary.expected_tensors("PTQ1_0")
    assert plain["blk.0.ffn_gate.weight"][1] == "PTQ1_0"
    assert plain["token_embd.weight"][1] == "PTQ1_0"
    assert plain["blk.1.ssm_alpha.weight"][1] == ("BF16", "F32")
    assert not any(name.startswith("blk.64.") for name in plain)
    mtp = ternary.expected_tensors("PTQ1_0", mtp=True)
    added = {name: kind for name, (_, kind) in mtp.items() if name not in plain}
    assert len(added) == 15 and all(name.startswith("blk.64.") for name in added)
    assert added["blk.64.nextn.eh_proj.weight"] == "blocks"
    assert added["blk.64.attn_q.weight"] == "blocks"
    assert added["blk.64.nextn.enorm.weight"] == "F32"
    assert mtp["blk.64.attn_q.weight"][0] == (2 * ternary.ATTENTION_QUERY_ROWS, ternary.HIDDEN)


def test_source_inputs_answer_membership_without_opening():
    sources = SourceInputs(
        object(), {"vision": Path("/nonexistent/mmproj.gguf")}, None, frozenset()
    )
    assert "vision" in sources and "base" in sources
    assert "mtp" not in sources
