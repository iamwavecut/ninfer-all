"""Keep large offline transfers from retaining whole artifacts in Linux's page cache.

The positional-I/O helpers also give the artifact tools one portable spelling: Windows has no
os.pread/os.pwrite/os.fdatasync/posix_fadvise, and its os.open defaults to text mode. Large
inputs are read, never mapped: Windows keeps every touched page of a mapped view in the
process working set, so a pass over a file larger than RAM exhausts memory there.
"""

from __future__ import annotations

import os
import threading

IO_CHUNK_BYTES = 8 * 1024 * 1024
WRITEBACK_BYTES = 64 * 1024 * 1024
_PAGE_BYTES = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
# Emulated positional I/O moves the shared file offset; serialize it within this process.
_SEEK_LOCK = threading.Lock()


def open_read(path: str | os.PathLike[str]) -> int:
    return os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))


def pread(fd: int, count: int, offset: int) -> bytes:
    if hasattr(os, "pread"):
        return os.pread(fd, count, offset)
    with _SEEK_LOCK:
        os.lseek(fd, offset, os.SEEK_SET)
        chunks = []
        remaining = count
        while remaining:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


def pread_into(fd: int, buffer, offset: int) -> None:
    """Fill the writable `buffer` from file `offset`; a file that ends first is an error."""

    view = memoryview(buffer).cast("B")
    while view:
        count = min(len(view), IO_CHUNK_BYTES)
        if hasattr(os, "preadv"):
            count = os.preadv(fd, [view[:count]], offset)
        else:
            chunk = pread(fd, count, offset)
            count = len(chunk)
            view[:count] = chunk
        if count <= 0:
            raise OSError(f"short read at file offset {offset}")
        view = view[count:]
        offset += count


def pwrite(fd: int, data: bytes | memoryview, offset: int) -> int:
    if hasattr(os, "pwrite"):
        return os.pwrite(fd, data, offset)
    with _SEEK_LOCK:
        os.lseek(fd, offset, os.SEEK_SET)
        return os.write(fd, data)


def datasync(fd: int) -> None:
    (os.fdatasync if hasattr(os, "fdatasync") else os.fsync)(fd)


def discard_cached_pages(fd: int, offset: int = 0, count: int | None = None) -> None:
    if not hasattr(os, "posix_fadvise"):
        return
    if count is None:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    elif count > 0:
        begin = offset // _PAGE_BYTES * _PAGE_BYTES
        end = (offset + count + _PAGE_BYTES - 1) // _PAGE_BYTES * _PAGE_BYTES
        os.posix_fadvise(fd, begin, end - begin, os.POSIX_FADV_DONTNEED)


class Writeback:
    """Bound dirty output across all open shards; release clean pages after writeback.

    The periodic sync exists to drop the written pages from the page cache. Without
    posix_fadvise (Windows, macOS) it cannot, and each sync only stalls the writer, so there the
    output is synced once, by the final flush()."""

    def __init__(self) -> None:
        self._bytes = 0
        self._fds: set[int] = set()

    def written(self, fd: int, count: int) -> None:
        self._fds.add(fd)
        if not hasattr(os, "posix_fadvise"):
            return
        self._bytes += count
        if self._bytes >= WRITEBACK_BYTES:
            self.flush()

    def flush(self) -> None:
        for fd in self._fds:
            datasync(fd)
            discard_cached_pages(fd)
        self._fds.clear()
        self._bytes = 0
