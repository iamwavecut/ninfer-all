from __future__ import annotations

import numpy as np
import pytest

from tools.convert.sources import gguf as gguf_module
from tools.convert.sources.gguf import (
    TYPE_BF16,
    TYPE_F32,
    GGUFFile,
    GGUFSplit,
    open_gguf,
    write_gguf,
)

TYPE_Q8_0 = 8
ROWS, ROW_BYTES = 40, 2 * 34


def _fixture(tmp_path):
    generator = np.random.default_rng(5)
    blocks = generator.integers(0, 256, size=(ROWS, ROW_BYTES), dtype=np.uint8)
    words = generator.integers(0, 2**16, size=(6, 8), dtype=np.uint16)
    vector = generator.standard_normal(7).astype("<f4")
    path = tmp_path / "file.gguf"
    write_gguf(
        path,
        {"general.architecture": "test"},
        [
            ("q8", (ROWS, 64), TYPE_Q8_0, blocks.tobytes()),
            ("bf16", (6, 8), TYPE_BF16, words.astype("<u2").tobytes()),
            ("f32", (7,), TYPE_F32, vector.tobytes()),
        ],
    )
    return path, blocks, words, vector


def _stored(path, gguf: GGUFFile, name: str, begin: int, end: int) -> np.ndarray:
    start = gguf.data_offset + gguf.info(name).offset
    return np.frombuffer(path.read_bytes()[start + begin : start + end], dtype=np.uint8)


def test_reads_are_the_stored_bytes_and_never_map_the_file(tmp_path, monkeypatch):
    path, blocks, words, vector = _fixture(tmp_path)

    def refuse(*_args, **_kwargs):
        raise AssertionError("the GGUF reader mapped the file")

    monkeypatch.setattr(np, "memmap", refuse)
    with GGUFFile(path) as gguf:
        for name in ("q8", "bf16", "f32"):
            size = gguf.info(name).nbytes
            for begin, end in ((0, size), (3, size - 5), (17, 17)):
                assert np.array_equal(
                    gguf.tensor_bytes(name, begin, end), _stored(path, gguf, name, begin, end)
                )
        assert np.array_equal(gguf.read_blocks("q8", 5, 9), blocks[5:9])
        assert np.array_equal(gguf.read_bf16_words("bf16"), words)
        assert np.array_equal(gguf.read_bf16_words("bf16", 2, 5), words[2:5])
        assert np.array_equal(gguf.read_direct("f32"), vector)
        # Each read owns its bytes.
        first = gguf.read_blocks("q8", 0, 4)
        first[:] = 0
        assert np.array_equal(gguf.read_blocks("q8", 0, 4), blocks[:4])
        with pytest.raises(ValueError, match="outside"):
            gguf.read_bf16_words("bf16", 4, 7)
    with pytest.raises(ValueError, match="closed"):
        gguf.tensor_bytes("q8")


@pytest.mark.parametrize(
    ("gap_bytes", "runs"),
    [(0, [(8, 10), (30, 33)]), (ROW_BYTES * 19, [(8, 10), (30, 33)]), (ROW_BYTES * 20, [(8, 33)])],
)
def test_selected_rows_are_read_in_runs(tmp_path, monkeypatch, gap_bytes, runs):
    path, blocks, _, _ = _fixture(tmp_path)
    monkeypatch.setattr(gguf_module, "ROW_GAP_BYTES", gap_bytes)
    with GGUFFile(path) as gguf:
        reads = []
        read_blocks = gguf.read_blocks

        def recorded(name, begin=0, end=None):
            reads.append((begin, end))
            return read_blocks(name, begin, end)

        monkeypatch.setattr(gguf, "read_blocks", recorded)
        selected = [8, 9, 30, 31, 32]
        assert np.array_equal(gguf.read_rows("q8", np.array(selected)), blocks[selected])
        assert reads == runs
        for index in ([39, 0, 7, 7, 20], [3], [31, 30, 9, 8, 32, 8]):
            assert np.array_equal(gguf.read_rows("q8", np.array(index)), blocks[index])
        assert gguf.read_rows("q8", np.array([], dtype=np.int64)).shape == (0, ROW_BYTES)
        for index in ([ROWS], [-1]):
            with pytest.raises(ValueError, match="outside"):
                gguf.read_rows("q8", np.array(index))


def test_a_truncated_data_section_is_refused_before_reading(tmp_path):
    path, _, _, _ = _fixture(tmp_path)
    with GGUFFile(path) as gguf:
        end = gguf.data_offset + gguf.info("q8").offset + gguf.info("q8").nbytes
    path.write_bytes(path.read_bytes()[: end - 1])
    with GGUFFile(path) as gguf:
        with pytest.raises(ValueError, match="truncated"):
            gguf.read_blocks("q8")
        stored = _stored(path, gguf, "q8", 0, ROW_BYTES)
        assert np.array_equal(gguf.read_blocks("q8", 0, 1), stored.reshape(1, -1))


def _split(tmp_path):
    generator = np.random.default_rng(6)
    blocks = generator.integers(0, 256, size=(ROWS, ROW_BYTES), dtype=np.uint8)
    words = generator.integers(0, 2**16, size=(6, 8), dtype=np.uint16)
    vector = generator.standard_normal(7).astype("<f4")
    shards = [tmp_path / f"model-{i:05d}-of-00003.gguf" for i in (1, 2, 3)]
    # llama.cpp's split of a large model: the first shard holds the metadata and no tensor.
    write_gguf(shards[0], {"general.architecture": "test", "split.count": 3}, [])
    write_gguf(shards[1], {"split.count": 3},
               [("q8", (ROWS, 64), TYPE_Q8_0, blocks.tobytes()),
                ("bf16", (6, 8), TYPE_BF16, words.astype("<u2").tobytes())])
    write_gguf(shards[2], {"split.count": 3}, [("f32", (7,), TYPE_F32, vector.tobytes())])
    return shards, blocks, words, vector


def test_a_split_reads_each_tensor_from_the_shard_that_holds_it(tmp_path):
    shards, blocks, words, vector = _split(tmp_path)
    with open_gguf(shards[0]) as gguf:
        assert isinstance(gguf, GGUFSplit)
        assert gguf.kv["general.architecture"] == "test"
        assert list(gguf.tensors) == ["q8", "bf16", "f32"]
        assert not gguf.truncated()
        assert np.array_equal(gguf.read_blocks("q8", 5, 9), blocks[5:9])
        assert np.array_equal(gguf.read_rows("q8", np.array([31, 2, 2])), blocks[[31, 2, 2]])
        assert np.array_equal(gguf.read_bf16_words("bf16", 2, 5), words[2:5])
        assert np.array_equal(gguf.read_direct("f32"), vector)
        with pytest.raises(KeyError, match="absent"):
            gguf.tensor_bytes("other")
    with pytest.raises(ValueError, match="closed"):
        gguf.tensor_bytes("q8")
    # Another shard is a file of its own, as a table shard named by its own source is.
    with open_gguf(shards[2]) as last:
        assert not isinstance(last, GGUFSplit) and list(last.tensors) == ["f32"]


def test_a_split_opens_without_a_missing_shard_and_reports_a_cut_one(tmp_path):
    shards, blocks, _, _ = _split(tmp_path)
    shards[2].unlink()
    with open_gguf(shards[0]) as gguf:
        assert list(gguf.tensors) == ["q8", "bf16"]
        assert np.array_equal(gguf.read_blocks("q8"), blocks)
    data = shards[1].read_bytes()
    shards[1].write_bytes(data[: len(data) - 16])
    with open_gguf(shards[0]) as gguf:
        assert gguf.truncated()


def test_a_split_refuses_a_repeated_tensor_and_a_wrong_count(tmp_path):
    shards, _, _, vector = _split(tmp_path)
    write_gguf(shards[2], {"split.count": 3}, [("q8", (7,), TYPE_F32, vector.tobytes())])
    with pytest.raises(ValueError, match="repeats"):
        open_gguf(shards[0])
    write_gguf(shards[0], {"general.architecture": "test", "split.count": 4}, [])
    shards[2].unlink()
    with pytest.raises(ValueError, match="split.count"):
        open_gguf(shards[0])
    with pytest.raises(ValueError, match="first shard"):
        GGUFSplit(shards[1])
