from __future__ import annotations

import json
from pathlib import Path
import struct

import pytest

from tools.artifact.reader import Artifact
from tools.artifact.framing import HEADER, MAGIC, PART_MAGIC
from tools.artifact.schema import (
    ArtifactError,
    ResourceSpec,
    TensorSpec,
    encode_directory,
    parse_directory,
)
from tools.artifact.writer import ArtifactWriter, layout_directory


def _small(path: Path) -> dict:
    specs = [
        TensorSpec("w", (2, 2), "bf16", "contiguous_le_v1"),
        TensorSpec("scale", (), "fp32", "contiguous_le_v1"),
        ResourceSpec("template", 5),
    ]
    components = {
        "text": {
            "config": {"architectures": ["Qwen3_5ForCausalLM"], "hidden_size": 2},
            "resources": {"chat_template.jinja": "template"},
        }
    }
    bindings = {
        "text/weight": {"object": "w"},
        "text/reordered": {
            "parts": [
                {"object": "w", "range": [2, 4]},
                {"object": "w", "range": [0, 2]},
            ]
        },
    }
    uses = [
        {
            "parameter": "text/weight",
            "input": "text/input",
            "activation_policy": "AllowA4",
            "auxiliaries": {"activation_input_divisor": {"object": "scale"}},
        }
    ]
    with ArtifactWriter(
        path, specs, components=components, bindings=bindings, uses=uses
    ) as writer:
        writer.write_object("w", struct.pack("<4H", 0, 0x8000, 0x3F80, 0x7FC1))
        writer.write_object("scale", struct.pack("<f", 2.0))
        writer.write_object("template", b"hello")
        return writer.directory.to_json()


def _rewrite(path: Path, directory: dict) -> None:
    with path.open("r+b") as stream:
        magic, reserve, identity = HEADER.unpack(stream.read(HEADER.size))
        data = encode_directory(directory)
        assert len(data) <= reserve
        stream.seek(HEADER.size)
        stream.write(data + b" " * (reserve - len(data)))


def test_single_file_known_header_payload_and_refs(tmp_path):
    path = tmp_path / "test.ninfer"
    directory = _small(path)
    raw = path.read_bytes()
    magic, json_bytes, identity = HEADER.unpack(raw[:32])
    assert magic == b"NINFER\0\3" and len(identity) == 16
    payload_offset = (32 + json_bytes + 4095) // 4096 * 4096
    assert payload_offset == 4096
    assert raw[payload_offset : payload_offset + 8] == struct.pack(
        "<4H", 0, 0x8000, 0x3F80, 0x7FC1
    )
    assert raw[payload_offset + 8 : payload_offset + 256] == bytes(248)
    assert raw[payload_offset + 256 : payload_offset + 260] == struct.pack("<f", 2)
    assert raw[-5:] == b"hello"
    with Artifact(path) as reader:
        assert reader.directory.to_json() == directory
        assert reader.read_object("template") == b"hello"
        assert reader.read_range(reader.payload_bytes, 0) == b""
        with pytest.raises(ArtifactError, match="exceeds payload"):
            reader.read_range(reader.payload_bytes, 1)


def test_shards_use_recorded_names_and_open_only_when_needed(tmp_path):
    path = tmp_path / "large.ninfer"
    payload = bytes(range(251)) * 110
    specs = [ResourceSpec("data", len(payload))]
    with ArtifactWriter(
        path,
        specs,
        components={"text": {"config": {}, "resources": {"data": "data"}}},
        bindings={},
        max_file_bytes=12288,
    ) as writer:
        writer.write_object(
            "data", (payload[:5000], payload[5000:9000], payload[9000:])
        )
        directory = writer.directory.to_json()
        identity = writer.artifact_id
    assert [f["path"] for f in directory["files"]] == [
        None,
        "large.ninfer.part-0001",
        "large.ninfer.part-0002",
        "large.ninfer.part-0003",
    ]
    for index in range(1, 4):
        part = tmp_path / directory["files"][index]["path"]
        assert HEADER.unpack(part.read_bytes()[:32]) == (PART_MAGIC, index, identity)
        assert part.stat().st_size <= 12288
    renamed = tmp_path / "user-chosen-part"
    (tmp_path / directory["files"][1]["path"]).rename(renamed)
    directory["files"][1]["path"] = renamed.name
    _rewrite(path, directory)
    with Artifact(path) as reader:
        assert reader.read_object("data") == payload
        assert (
            b"".join(reader.iter_range(8120, 17300, chunk_bytes=503))
            == payload[8120:25420]
        )

    renamed.unlink()
    with Artifact(path) as reader:
        assert reader.read_range(0, 100) == payload[:100]
        with pytest.raises(FileNotFoundError):
            reader.read_range(8200, 10)


TEMPLATE = bytes(range(250)) * 16
WEIGHTS = bytes(range(256)) * 16


def _two_files(path: Path) -> ArtifactWriter:
    """An artifact of two 8192-byte files. The tensor starts at the next 256-byte boundary after
    the 4000-byte resource; the entry file's 4096 payload bytes end in that padding, which no write
    reaches."""

    return ArtifactWriter(
        path,
        [ResourceSpec("template", 4000), TensorSpec("w", (2048,), "bf16", "contiguous_le_v1")],
        components={
            "text": {
                "config": {"architectures": ["Qwen3_5ForCausalLM"], "hidden_size": 2},
                "resources": {"chat_template.jinja": "template"},
            }
        },
        bindings={"text/weight": {"object": "w"}},
        max_file_bytes=8192,
    )


def _write_two_files(writer: ArtifactWriter) -> None:
    writer.write_object("template", TEMPLATE)
    writer.write_object("w", WEIGHTS)


def test_a_part_ending_in_alignment_padding_gets_its_directory_length(tmp_path):
    path = tmp_path / "gap.ninfer"
    with _two_files(path) as writer:
        # Construction writes the framing only; writes extend the files.
        temporary = list(tmp_path.glob(".gap.ninfer*.tmp"))
        assert len(temporary) == 2 and all(p.stat().st_size <= 4096 for p in temporary)
        _write_two_files(writer)
        directory = writer.directory.to_json()
    assert [f["payload_bytes"] for f in directory["files"]] == [4096, 4096]
    assert path.stat().st_size == 8192
    assert (tmp_path / directory["files"][1]["path"]).stat().st_size == 8192
    with Artifact(path) as reader:
        assert reader.read_object("template") == TEMPLATE
        assert reader.read_object("w") == WEIGHTS
        assert reader.read_range(4000, 96) == bytes(96)


def test_the_writer_starts_only_with_room_for_every_file(tmp_path, monkeypatch):
    from tools.artifact import writer as writer_module

    path = tmp_path / "room.ninfer"
    monkeypatch.setattr(writer_module, "free_bytes", lambda directory: 16383)
    with pytest.raises(ArtifactError, match="needs 16,384 bytes"):
        _two_files(path)
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(writer_module, "free_bytes", lambda directory: 16384)
    with _two_files(path) as writer:
        _write_two_files(writer)
    assert sum(file.stat().st_size for file in tmp_path.iterdir()) == 16384
    # A file system that reports no capacity is not checked.
    monkeypatch.setattr(writer_module, "free_bytes", lambda directory: None)
    with _two_files(tmp_path / "unknown.ninfer") as writer:
        _write_two_files(writer)


def test_free_bytes_counts_what_this_process_may_fill(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tools.artifact import writer as writer_module

    assert writer_module.free_bytes(tmp_path) > 0
    if not hasattr(writer_module.os, "statvfs"):
        return
    stat = SimpleNamespace(f_blocks=100, f_bfree=50, f_bavail=40, f_frsize=4096)
    monkeypatch.setattr(writer_module.os, "statvfs", lambda directory: stat)
    monkeypatch.setattr(writer_module.os, "geteuid", lambda: 1000)
    assert writer_module.free_bytes(tmp_path) == 40 * 4096
    # Root may also fill the blocks reserved for it.
    monkeypatch.setattr(writer_module.os, "geteuid", lambda: 0)
    assert writer_module.free_bytes(tmp_path) == 50 * 4096
    stat.f_blocks = 0
    assert writer_module.free_bytes(tmp_path) is None


def test_the_writer_removes_temporaries_that_an_interrupted_writer_left(tmp_path, capsys):
    path = tmp_path / "room.ninfer"
    stale = [
        tmp_path / ".room.ninfer.ab12cd_3.tmp",
        tmp_path / ".room.ninfer.part-0001.zz9yx8wv.tmp",
        tmp_path / ".room.ninfer.part-0007.q1w2e3r4.tmp",
    ]
    unrelated = [
        tmp_path / ".room.ninfer.notes.tmp",
        tmp_path / ".other.ninfer.ab12cd34.tmp",
        tmp_path / "room.ninfer.ab12cd34.tmp",
    ]
    for file in stale + unrelated:
        file.write_bytes(bytes(100))
    with _two_files(path) as writer:
        _write_two_files(writer)
    assert [file.exists() for file in stale] == [False] * 3
    assert [file.exists() for file in unrelated] == [True] * 3
    assert capsys.readouterr().err.count("left by an interrupted conversion") == 3
    with Artifact(path) as reader:
        assert reader.read_object("w") == WEIGHTS


def test_a_second_writer_of_the_same_artifact_refuses_to_start(tmp_path):
    path = tmp_path / "busy.ninfer"
    with _two_files(path) as first:
        temporary = sorted(tmp_path.glob(".busy.ninfer*.tmp"))
        assert len(temporary) == 2
        with pytest.raises(ArtifactError, match="another conversion is writing busy.ninfer"):
            _two_files(path)
        assert sorted(tmp_path.glob(".busy.ninfer*.tmp")) == temporary
        _write_two_files(first)
    with Artifact(path) as reader:
        assert reader.read_object("template") == TEMPLATE
    assert sorted(tmp_path.iterdir()) == [path, tmp_path / "busy.ninfer.part-0001"]


@pytest.mark.parametrize("failure", ["missing", "producer"])
def test_incomplete_or_failed_output_is_not_published(tmp_path, failure):
    path = tmp_path / "failed.ninfer"
    with pytest.raises((ArtifactError, RuntimeError)):
        with ArtifactWriter(
            path,
            [ResourceSpec("r", 20000)],
            components={"text": {"config": {}}},
            bindings={},
            max_file_bytes=12288,
        ) as writer:
            writer.write_region("r", 0, b"abc")
            if failure == "producer":
                raise RuntimeError("source failed")
    assert list(tmp_path.iterdir()) == []


def test_existing_output_is_preserved(tmp_path):
    path = tmp_path / "existing.ninfer"
    path.write_bytes(b"old")
    with pytest.raises(FileExistsError):
        _small(path)
    assert path.read_bytes() == b"old"


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(
            lambda d: d["objects"][1].update(offset=1), id="overlapping-objects"
        ),
        pytest.param(
            lambda d: d["bindings"]["text/reordered"]["parts"][0].update(range=[0, 5]),
            id="binding-outside-parent",
        ),
        pytest.param(lambda d: d.update(components={}), id="no-components"),
    ],
)
def test_directory_rejects_invalid_structure(tmp_path, change):
    directory = _small(tmp_path / "base.ninfer")
    change(directory)
    with pytest.raises(ArtifactError):
        parse_directory(directory)


def test_unknown_codec_is_deferred_until_object_is_consumed(tmp_path):
    path = tmp_path / "future.ninfer"
    directory = _small(path)
    directory["objects"][0]["format"] = "future_codec"
    _rewrite(path, directory)
    with Artifact(path) as reader:
        assert reader.read_object("template") == b"hello"
        with pytest.raises(ArtifactError, match="future_codec"):
            reader.read_object("w")


def test_an_artifact_is_one_file_unless_a_limit_splits_it():
    size = 32_000_000_000
    description = {
        "components": {"text": {"config": {}}},
        "bindings": {},
        "uses": [],
        "objects": [
            {
                "id": "r",
                "kind": "resource",
                "encoding": "raw_bytes_v1",
                "offset": 0,
                "bytes": size,
            }
        ],
    }
    directory, _, _ = layout_directory("large.ninfer", description, size)
    assert [file.payload_bytes for file in directory.files] == [size]
    # A limit counts the framing: the entry's header and directory share its bytes.
    directory, _, start = layout_directory(
        "large.ninfer", description, size, max_file_bytes=size
    )
    assert len(directory.files) == 2
    assert start + directory.files[0].payload_bytes <= size
    assert 4096 + directory.files[1].payload_bytes <= size
    description["objects"][0]["bytes"] = size - start
    directory, _, same_start = layout_directory(
        "small.ninfer", description, size - start, max_file_bytes=size
    )
    assert same_start == start and len(directory.files) == 1
    with pytest.raises(ArtifactError):
        layout_directory("small.ninfer", description, 1 << 64)
