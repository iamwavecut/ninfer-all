"""Read BASALTMF v1/v2 ranges; expose the stored Flash-Next tensors without unpacking the pack.

Basalt keeps GGUF originals beside the dense operands it actually serves. The dense index wins:
in particular --compat-bf16 has already rounded those operands. HC8 is a separate code/scale
layout, not ggml's quantizer. Both it and the interleaved expert blobs are rearranged, not refit.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import struct

import numpy as np

from tools.artifact.file_io import discard_cached_pages, pread_into
from .gguf import GGUFFile, TensorInfo, TYPE_BF16, TYPE_F16, TYPE_F32

HEADER = struct.Struct("<8sIIQ40s")
ENTRY = struct.Struct("<64sQQ32s16s")
HC_HEADER = struct.Struct("<8sII")
HC_ENTRY = struct.Struct("<128sIIIIQQ")


@dataclass(frozen=True)
class Section:
    offset: int
    size: int
    sha256: str


class BasaltFile:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = self.path.open("rb", buffering=0)
        try:
            raw = self._file.read(HEADER.size)
            if len(raw) != HEADER.size:
                raise ValueError("truncated Basalt header")
            magic, self.version, count, size, _ = HEADER.unpack(raw)
            if magic != b"BASALTMF" or self.version not in (1, 2) or not 0 < count <= 511:
                raise ValueError("expected BASALTMF version 1 or 2")
            if self.path.stat().st_size < size or size < 65536:
                raise ValueError("truncated Basalt file")
            self.sections = {}
            ranges = []
            for _ in range(count):
                raw = self._file.read(ENTRY.size)
                if len(raw) != ENTRY.size:
                    raise ValueError("truncated Basalt section directory")
                name, offset, length, digest, _ = ENTRY.unpack(raw)
                name = name.split(b"\0", 1)[0].decode("utf-8")
                if not name or name in self.sections or offset < 65536 or offset % (2 << 20):
                    raise ValueError(f"invalid Basalt section {name!r}")
                if offset + length > size:
                    raise ValueError(f"{name}: section exceeds declared file size")
                self.sections[name] = Section(offset, length, digest.hex())
                ranges.append((offset, offset + length))
            ordered = sorted(ranges)
            if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
                raise ValueError("overlapping Basalt sections")
            self.meta = json.loads(self.read("meta").tobytes())
        except BaseException:
            self.close()
            raise

    def read(self, section: str, offset: int = 0, size: int | None = None) -> np.ndarray:
        entry = self.sections[section]
        size = entry.size - offset if size is None else size
        if not 0 <= offset <= offset + size <= entry.size:
            raise ValueError(f"{section}: range [{offset}, {offset + size}) exceeds {entry.size}")
        out = np.empty(size, dtype=np.uint8)
        fd = self._file.fileno()
        pread_into(fd, out, entry.offset + offset)
        discard_cached_pages(fd, entry.offset + offset, size)
        return out

    def text(self, section: str) -> str:
        return self.read(section).tobytes().decode("utf-8")

    def verify(self, section: str) -> None:
        digest = hashlib.sha256()
        size = self.sections[section].size
        for offset in range(0, size, 16 << 20):
            digest.update(self.read(section, offset, min(16 << 20, size - offset)))
        if digest.hexdigest() != self.sections[section].sha256:
            raise ValueError(f"{section}: SHA-256 does not match the Basalt directory")

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class SectionBlob:
    def __init__(self, pack: BasaltFile, name: str):
        self.pack, self.name = pack, name

    def read(self, offset, size):
        return self.pack.read(self.name, offset, size)

    def close(self):
        pass


class HC8:
    def __init__(self, pack: BasaltFile):
        self.pack = pack
        self.entries = {}
        if "hc8" not in pack.sections:
            if pack.version == 2:
                raise ValueError("Basalt v2 requires its stored hc8 section")
            return
        magic, version, count = HC_HEADER.unpack(pack.read("hc8", 0, HC_HEADER.size))
        if magic != b"BASALTH8" or version != 1:
            raise ValueError("unsupported Basalt HC8 layout")
        data_start = HC_HEADER.size + count * HC_ENTRY.size
        for i in range(count):
            raw = pack.read("hc8", HC_HEADER.size + i * HC_ENTRY.size, HC_ENTRY.size)
            name, rows, cols, stride, _, offset, size = HC_ENTRY.unpack(raw)
            name = name.split(b"\0", 1)[0].decode()
            if (name in self.entries or rows < 1 or cols < 32 or cols % 32
                    or stride != ((cols // 32 + 7) & ~7) or offset < data_start
                    or size != rows * (cols + 2 * stride)
                    or offset + size > pack.sections["hc8"].size):
                raise ValueError(f"{name}: invalid HC8 entry")
            self.entries[name] = (rows, cols, stride, offset)

    def rows(self, name, first, last):
        rows, cols, stride, offset = self.entries[name]
        if not 0 <= first <= last <= rows:
            raise ValueError(f"{name}: HC8 rows out of bounds")
        codes = self.pack.read("hc8", offset + first * cols, (last - first) * cols)
        scales = self.pack.read("hc8", offset + rows * cols + first * stride * 2,
                                (last - first) * stride * 2)
        blocks = cols // 32
        out = np.empty((last - first, blocks, 34), dtype=np.uint8)
        out[..., :2] = scales.reshape(last - first, stride, 2)[:, :blocks]
        out[..., 2:] = codes.reshape(last - first, blocks, 32)
        return out.reshape(last - first, blocks * 34)


class BasaltGGUF(GGUFFile):
    """A GGUF-shaped view of the pack's effective operands, with bounded blob reads."""

    def __init__(self, pack: BasaltFile):
        self.pack, self.path = pack, pack.path
        self.kv = pack.meta["gguf_kv"]
        if self.kv.get("general.architecture") != "qwen4exp":
            raise ValueError("the Basalt bridge implements Flash-Next (qwen4exp)")
        self.tensors, self.locations, self.experts = {}, {}, {}
        self.expert_scales = {}
        self.narrow_f16, self.narrow_bf16 = set(), set()
        self.hc = HC8(pack)
        for name, item in pack.meta["tensors"].items():
            info = TensorInfo(name, tuple(reversed(item["shape"])), item["type_id"], 0)
            if info.nbytes != item["bytes"]:
                raise ValueError(f"{name}: metadata byte count disagrees with type/shape")
            self._add(info, item.get("section", "gguf"), item["offset"], item["bytes"])
        for line in pack.text("dense.index").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 19:
                raise ValueError("expected Basalt dense index v3 (19 columns)")
            name, shard, kind = fields[:3]
            if shard != "0":
                raise ValueError(f"{name}: unexpected dense shard {shard}")
            if kind == "0":
                continue
            offset, size, cols, rows = map(int, (fields[3], fields[4], fields[7], fields[8]))
            types = {"1": TYPE_F32, "2": TYPE_F32, "3": TYPE_F32,
                     "4": TYPE_BF16, "5": TYPE_F16}
            if kind not in types:
                raise ValueError(f"{name}: unsupported dense kind {kind}")
            shape = (rows, cols) if rows else (cols,)
            info = TensorInfo(name, shape, types[kind], 0)
            if info.nbytes != size:
                raise ValueError(f"{name}: dense index byte count disagrees with shape")
            self._add(info, "dense", offset, size)
            if kind == "1":
                # Bf16InF32 uses the high word, not a fresh round-to-nearest conversion.
                self.narrow_bf16.add(name)
                self.tensors[name] = TensorInfo(name, shape, TYPE_BF16, 0)
            elif kind == "3":
                # F16InF32 stores float words but Basalt narrows them before the kernel reads.
                self.narrow_f16.add(name)
                self.tensors[name] = TensorInfo(name, shape, TYPE_F16, 0)
        self.dense_tensors = dict(self.tensors)
        for name, (rows, cols, _, _) in self.hc.entries.items():
            if not name.startswith("mtp."):
                self.tensors[name] = TensorInfo(name, (rows, cols), 8, 0)
        lines = pack.text("experts.layout").splitlines()
        version = re.match(r"# (?:strata )?experts v([3567]):", lines[0]) if lines else None
        if version is None:
            raise ValueError("expected Basalt block expert layout v3, v5, v6 or v7")
        scaled = version[1] == "6"
        count = self.kv["qwen4exp.expert_count"]
        for line in lines:
            if not line.strip() or line.startswith("#"):
                continue
            values = list(map(int, line.split()))
            if len(values) not in (5, 6):
                raise ValueError("invalid Basalt expert layout row")
            layer, gt, dt, offset, blob = values[:5]
            ut = values[5] if len(values) == 6 else gt
            if scaled and (len(values) != 5 or gt != 40 or dt != 40):
                raise ValueError("Basalt v6 requires NVFP4 gate, up and down blocks")
            roles = [("gate", gt, 640, 2560), ("up", ut, 640, 2560), ("down", dt, 2560, 640)]
            position = 0
            for role, type_id, rows, cols in roles:
                name = f"blk.{layer}.ffn_{role}_exps.weight"
                info = TensorInfo(name, (count, rows, cols), type_id, 0)
                per = info.nbytes // count
                if name in self.experts:
                    raise ValueError(f"duplicate expert bank {name}")
                self.experts[name] = (offset, blob, position, per)
                self.tensors[name] = info
                position += per
            if offset < 0 or position + (16 if scaled else 0) != blob or offset + count * blob > pack.sections["experts"].size:
                raise ValueError(f"layer {layer}: expert layout byte count differs")
            if scaled:
                tails = np.stack([pack.read("experts", offset + e * blob + position, 16)
                                  for e in range(count)]).view("<f4").reshape(count, 4)
                if not np.isfinite(tails).all() or (tails[:, :3] <= 0).any() or (tails[:, 3] != 0).any():
                    raise ValueError(f"layer {layer}: expected three positive finite expert scales and a zero")
                self.expert_scales[layer] = np.ascontiguousarray(tails[:, :3])

    def _add(self, info, section, offset, size):
        if offset < 0 or offset + size > self.pack.sections[section].size:
            raise ValueError(f"{info.name}: tensor exceeds section {section}")
        self.tensors[info.name] = info
        self.locations[info.name] = (section, offset)

    def tensor_bytes(self, name, byte_begin=0, byte_end=None):
        info = self.info(name)
        end = info.nbytes if byte_end is None else byte_end
        if not 0 <= byte_begin <= end <= info.nbytes:
            raise ValueError(f"{name}: tensor byte range out of bounds")
        if name in self.hc.entries:
            stride = info.shape[-1] // 32 * 34
            first, last = byte_begin // stride, -(-end // stride)
            data = self.hc.rows(name, first, last).reshape(-1)
            return data[byte_begin - first * stride:end - first * stride]
        if name in self.experts:
            if info.type_id == 40 and byte_begin != end:
                first, last = byte_begin // 36 * 36, -(-end // 36) * 36
                if first != byte_begin or last != end:
                    return self.tensor_bytes(name, first, last)[byte_begin - first:end - first]
            base, blob, role, per = self.experts[name]
            data = np.empty(end - byte_begin, dtype=np.uint8)
            position = byte_begin
            while position < end:
                expert, within = divmod(position, per)
                count = min(end - position, per - within)
                data[position - byte_begin:position - byte_begin + count] = self.pack.read(
                    "experts", base + expert * blob + role + within, count)
                position += count
            if info.type_id == 40:
                # CPU and CUDA agree only on unsigned UE4M3 scale bytes (as in Edge0's importer).
                if (data.reshape(-1, 36)[:, :4] & 0x80).any():
                    raise ValueError(f"{name}: NVFP4 block scale has bit 7 set")
            return data
        return self._dense_bytes(name, byte_begin, end)

    def _dense_bytes(self, name, byte_begin, end):
        section, offset = self.locations[name]
        if name in self.narrow_f16 or name in self.narrow_bf16:
            first, last = byte_begin // 2, -(-end // 2)
            raw = self.pack.read(section, offset + first * 4, (last - first) * 4)
            data = ((raw.view("<u4") >> 16).astype("<u2") if name in self.narrow_bf16
                    else raw.view("<f4").astype("<f2")).view(np.uint8)
            return data[byte_begin - first * 2:end - first * 2]
        return self.pack.read(section, offset + byte_begin, end - byte_begin)

    def truncated(self):
        return False

    def prefill_injection(self, name):
        """The original BF16 injection remains alongside HC8 for Basalt's prompt path."""
        info = self.dense_tensors[name]
        if info.type_id != TYPE_BF16:
            raise ValueError(f"{name}: expected the stored BF16 prefill injection")
        words = self._dense_bytes(name, 0, info.nbytes).view("<u2")
        return (words.astype(np.uint32) << 16).view(np.float32).reshape(info.shape)

    def close(self):
        pass


def mtp_pack(pack: BasaltFile):
    """Reuse the existing Strata MTP layout and mapping, reading sections instead of loose files."""
    from ..edge0 import MtpPack, HIDDEN, EXPERT_WIDTH, MTP_EXPERTS

    class PackedMtp(MtpPack):
        def _tensor(self, name, kind, shape, size):
            if name not in self.tensors:
                raise ValueError(f"MTP tensor missing: {name}")
            actual, rows, cols, offset, nbytes = self.tensors[name]
            if (actual, (rows, cols), nbytes) != (kind, shape, size):
                raise ValueError(f"MTP tensor layout mismatch: {name}")
            if name not in self._cache:
                self._cache.clear()
                self._cache[name] = self._dense.read(offset, size)
            return self._cache[name]

    mtp = PackedMtp.__new__(PackedMtp)
    mtp.root = pack.path
    mtp.hidden, mtp.middle = HIDDEN, EXPERT_WIDTH
    mtp.tensors = {}
    for line in pack.text("mtp/dense.index").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, kind, rows, cols, offset, size = line.split()
        item = (kind, int(rows), int(cols), int(offset), int(size))
        if name in mtp.tensors or item[3] < 0 or item[3] + item[4] > pack.sections["mtp/dense"].size:
            raise ValueError(f"invalid MTP index entry {name}")
        mtp.tensors[name] = item
    gu = 2 * EXPERT_WIDTH * (HIDDEN // 64)
    dn = HIDDEN * (EXPERT_WIDTH // 64)
    mtp._planes = (gu * 16, dn * 16, gu * 2, dn * 2)
    mtp.record = sum(mtp._planes)
    if pack.sections["mtp/experts"].size != MTP_EXPERTS * mtp.record:
        raise ValueError("Basalt MTP expert section has the wrong size")
    mtp._dense = SectionBlob(pack, "mtp/dense")
    mtp._experts = SectionBlob(pack, "mtp/experts")
    mtp._cache = OrderedDict()
    return mtp
