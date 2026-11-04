from __future__ import annotations

import os

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


def test_writeback_syncs_periodically_only_where_it_can_drop_the_pages(monkeypatch):
    synced = []
    monkeypatch.setattr(file_io, "datasync", synced.append)
    monkeypatch.setattr(file_io, "discard_cached_pages", lambda fd, *args: None)

    monkeypatch.delattr(file_io.os, "posix_fadvise", raising=False)
    writeback = file_io.Writeback()
    writeback.written(7, file_io.WRITEBACK_BYTES)
    writeback.written(8, file_io.WRITEBACK_BYTES)
    assert synced == []
    writeback.flush()
    assert sorted(synced) == [7, 8]

    synced.clear()
    monkeypatch.setattr(file_io.os, "posix_fadvise", lambda *args: None, raising=False)
    writeback = file_io.Writeback()
    writeback.written(7, file_io.WRITEBACK_BYTES - 1)
    assert synced == []
    writeback.written(8, 1)
    assert sorted(synced) == [7, 8]
