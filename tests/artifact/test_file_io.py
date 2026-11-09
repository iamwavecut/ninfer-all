from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from tools.artifact import file_io


@pytest.mark.parametrize("vectored", [True, False])
def test_pread_into_fills_the_buffer_or_refuses_a_short_file(tmp_path, monkeypatch, vectored):
    data = bytes(range(256)) * 100
    path = tmp_path / "data"
    path.write_bytes(data)
    monkeypatch.setattr(file_io, "IO_CHUNK_BYTES", 1000)
    if not vectored:
        monkeypatch.delattr(file_io.os, "preadv", raising=False)
    fd = file_io.open_read(path)
    try:
        out = bytearray(5000)
        file_io.pread_into(fd, out, 123)
        assert out == data[123:5123]
        with pytest.raises(OSError, match="short read"):
            file_io.pread_into(fd, bytearray(10), len(data) - 5)
    finally:
        os.close(fd)


@pytest.mark.parametrize(
    "system, interval",
    [
        (SimpleNamespace(name="posix", posix_fadvise=None), file_io.WRITEBACK_BYTES),
        (SimpleNamespace(name="nt"), file_io.WINDOWS_WRITEBACK_BYTES),
        (SimpleNamespace(name="posix"), None),
    ],
    ids=["linux", "windows", "macos"],
)
def test_writeback_syncs_the_output_at_each_platform_interval(monkeypatch, system, interval):
    synced = []
    monkeypatch.setattr(file_io, "os", system)
    monkeypatch.setattr(file_io, "datasync", synced.append)
    monkeypatch.setattr(file_io, "discard_cached_pages", lambda fd, *args: None)
    writeback = file_io.Writeback()
    if interval is None:
        writeback.written(7, file_io.WINDOWS_WRITEBACK_BYTES)
        writeback.written(8, file_io.WINDOWS_WRITEBACK_BYTES)
        assert synced == []
    else:
        writeback.written(7, interval - 1)
        assert synced == []
        writeback.written(8, 1)
        assert sorted(synced) == [7, 8]
        synced.clear()
        writeback.written(9, 1)
    writeback.flush()
    assert sorted(synced) == ([7, 8] if interval is None else [9])
