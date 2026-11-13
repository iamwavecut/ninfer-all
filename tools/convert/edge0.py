"""An Edge0 release folder: a latent MoE over the Qwen3.8-Flash-Next skeleton, kept as stored.

Edge0-145B and Edge0-120B are published as Strata packs, one folder per model and expert format.
The folder holds the base model without its routed experts (``model-skeleton.gguf``, a llama.cpp
``qwen4exp`` GGUF that the Flash-Next GGUF recipe reads) and the latent MoE beside it:

``native_experts.txt``, ``experts.bin``
    One record per (layer, expert): the gate rows ``[640, ELL]``, the up rows ``[640, ELL]`` and
    the down rows ``[ELL, 640]`` in ggml blocks, back to back. An NVFP4 record ends with a 16-byte
    tail ``{s_gate, s_up, s_down, 0}`` (FP32), the multipliers of the three products' outputs.
``latent_dense_q8.bin``, ``latent_dense_q4.bin``
    For each layer the encoder ``D [G * ELL, 2560]`` and then the decoder ``U [G * 2560, ELL]``,
    in Q8_0 or Q4_0 blocks. Group ``g`` owns rows ``g * ELL ..`` of D and ``g * 2560 ..`` of U.
``labels.bin``
    INT32 ``[layers, experts]``: the group of each expert.
``scales/l{layer}.f32``
    Q3_K releases only: FP32 ``[ELL]`` per layer. The gate and up rows are stored times these
    column scales, so the encoder's output columns are multiplied by their reciprocals.
``mtp/``
    The base model's MTP block as Strata's runtime files: Q8_0 and BF16 matrices, F32 norms with
    their ``1 +`` applied, and Q2_0 experts as code and scale planes.
``ple-fp8.gguf``
    The n-gram table as the checkpoint ships it: one E4M3FN byte per value and one scale for the
    whole table (qwen4_exp_ngram.strata_fp8_source).

Every block row of the artifact is byte for byte a row of the release. Three things are computed:
the reciprocal column scales (one FP32 division each, as the reference engine computes them), the
MTP block's zero-centred norms (their stored ``1 + w`` less one), and the MTP block's BF16
hyper-connection matrices, which become ggml Q8_0 rows as in every Flash-Next GGUF conversion.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re

import numpy as np
import torch

from tools.artifact.file_io import discard_cached_pages, pread_into
from tools.artifact.formats import GGUF_FORMATS_BY_TYPE

from .gguf_blocks import dequantize_blocks, dequantize_q8_0, quantize_q8_0
from .methods import cast_direct
from .qwen4_exp_gguf import (
    EXPERT_WIDTH,
    HIDDEN,
    INDEXER_DIM,
    INDEXER_HEADS,
    KV_ROWS,
    LAYERS,
    LOWRANK,
    QUERY_ROWS,
    SHARED_WIDTH,
    STREAMS,
    WIDTH,
    _assign,
    _group_attention,
    _group_moe,
    _group_same_format,
    text_sources,
    validate,
)
from .sources.gguf import GGUFFile
from .sources.logical import EncodedRows, LogicalSource, array_source
from .ternary import _flat, attention_rows

GROUPS = 8
GGML_Q4_0, GGML_Q8_0, GGML_NVFP4, GGML_Q2_0 = 2, 8, 40, 42
NVFP4_TAIL = 16
MTP_EXPERTS = 512
# The pack's description of its formats (", Q3_K gu + shared colscale" in the Q3_K builds) is free
# text; the layer lines carry the formats.
_HEADER = re.compile(
    r"# strata native experts v6: latent A-only(?:, [^()]*)? "
    r"\(expert_embd ([0-9]+), expert_ff ([0-9]+)\)"
)
_DENSE_FILES = {"q8": ("latent_dense_q8.bin", GGML_Q8_0), "q4": ("latent_dense_q4.bin", GGML_Q4_0)}


def _row_bytes(type_id: int, columns: int) -> int:
    try:
        stored = GGUF_FORMATS_BY_TYPE[type_id]
    except KeyError:
        raise ValueError(f"ggml type {type_id} is not a supported block format") from None
    if columns % stored.block_elements:
        raise ValueError(f"{columns} columns are not whole {stored.name} blocks")
    return columns // stored.block_elements * stored.block_bytes


class _Blob:
    """Bounded reads of one large file that leave no pages in the page cache, as GGUFFile's."""

    def __init__(self, path: Path, size: int) -> None:
        if not path.is_file() or path.stat().st_size != size:
            actual = path.stat().st_size if path.is_file() else "no file"
            raise ValueError(f"{path}: {actual}, expected {size} bytes")
        self.path = path
        self._file = open(path, "rb", buffering=0)

    def read(self, offset: int, size: int) -> np.ndarray:
        if self._file is None:
            raise ValueError(f"{self.path}: the file is closed")
        out = np.empty(size, dtype=np.uint8)
        fd = self._file.fileno()
        pread_into(fd, out, offset)
        discard_cached_pages(fd, offset, size)
        return out

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class Edge0Pack:
    """An Edge0 release folder, after strict checks of every size its layout fixes."""

    def __init__(self, path: str | Path, hidden: int = HIDDEN) -> None:
        self.path = Path(path)
        self.hidden = hidden
        if not (self.path / "model.json").is_file() or not (
                self.path / "native_experts.txt").is_file():
            raise ValueError(f"{self.path}: not an Edge0 release folder (model.json and "
                             "native_experts.txt are required)")
        self.manifest = json.loads((self.path / "model.json").read_text())
        lines = (self.path / "native_experts.txt").read_text().splitlines()
        match = _HEADER.fullmatch(lines[0].strip()) if lines else None
        if match is None:
            raise ValueError(f"{self.path}/native_experts.txt: expected a v6 latent expert pack")
        self.width, self.middle = int(match[1]), int(match[2])
        records = [line.split() for line in lines[1:] if line.strip()]
        if not records or any(len(r) != 5 or not all(v.isdigit() for v in r) for r in records):
            raise ValueError(f"{self.path}/native_experts.txt: every layer line holds five numbers")
        records = [tuple(int(v) for v in r) for r in records]
        self.layers = len(records)
        self.experts = self.manifest.get("experts")
        if type(self.experts) is not int or self.experts <= 0:
            raise ValueError(f"{self.path}/model.json: experts must be a positive integer")
        if self.manifest.get("latent_dim") != self.width or (
                self.manifest.get("layers") != self.layers):
            raise ValueError(f"{self.path}: model.json and native_experts.txt disagree")
        types = {(r[1], r[2]) for r in records}
        if len(types) != 1:
            raise ValueError(f"{self.path}: the layers store their experts in different formats")
        self.gate_up_type, self.down_type = next(iter(types))
        self.gate_up_row = _row_bytes(self.gate_up_type, self.width)
        self.down_row = _row_bytes(self.down_type, self.middle)
        self.tail = NVFP4_TAIL if GGML_NVFP4 in (self.gate_up_type, self.down_type) else 0
        self.tail_offset = 2 * self.middle * self.gate_up_row + self.width * self.down_row
        self.record = self.tail_offset + self.tail
        for layer, (index, _, _, offset, record) in enumerate(records):
            if index != layer or record != self.record or (
                    offset != layer * self.experts * self.record):
                raise ValueError(f"{self.path}/native_experts.txt: layer {layer} is not a "
                                 f"contiguous run of {self.record}-byte records")
        labels = np.fromfile(self.path / "labels.bin", dtype="<i4")
        if labels.size != self.layers * self.experts or labels.min() < 0 or (
                labels.max() >= GROUPS):
            raise ValueError(f"{self.path}/labels.bin: expected {self.layers} x {self.experts} "
                             f"group labels below {GROUPS}")
        self._labels = labels.reshape(self.layers, self.experts)
        scales = [self.path / "scales" / f"l{layer}.f32" for layer in range(self.layers)]
        present = [path.is_file() for path in scales]
        if any(present) != all(present):
            raise ValueError(f"{self.path}/scales: column scales for some layers only")
        self._column_scales = scales if all(present) else None
        self._experts = _Blob(self.path / "experts.bin",
                              self.layers * self.experts * self.record)
        self._dense: dict[str, _Blob] = {}
        self._mtp: MtpPack | None = None

    def __enter__(self) -> Edge0Pack:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        self._experts.close()
        for blob in self._dense.values():
            blob.close()
        self._dense.clear()
        if self._mtp is not None:
            self._mtp.close()

    # --- the release's choices ----------------------------------------------------------------

    @property
    def released_dense(self) -> str:
        """The projection build the release's launcher selects: "q8" or "q4"."""

        fmt = self.manifest.get("env", {}).get("STRATA_LATENT_DENSE_FMT")
        if fmt not in ("8", "4"):
            raise ValueError(f"{self.path}/model.json: env.STRATA_LATENT_DENSE_FMT must be 8 or 4")
        return "q" + fmt

    @property
    def skeleton(self) -> Path:
        return self.path / "model-skeleton.gguf"

    @property
    def table(self) -> Path:
        return self.path / "ple-fp8.gguf"

    def check_tokenizer(self, tokenizer_json: Path) -> None:
        """Refuse a release whose vocabulary or merges differ from the base model's tokenizer.

        The release's `tokenizer/` holds the GGUF's vocabulary (`vocab.json`, every id up to the
        padded vocabulary), its merges and a llama.cpp form of the chat template without the
        `generation` markers; its `tokenizer.json` is a summary, not a Hugging Face tokenizer. The
        base model's frontend files therefore stay, once its ids and merges are the release's."""

        folder = self.path / "tokenizer"
        if not (folder / "vocab.json").is_file() or not (folder / "merges.txt").is_file():
            return
        base = json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))
        vocab = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
        expected = dict(base["model"]["vocab"])
        expected.update({token["content"]: token["id"] for token in base.get("added_tokens", ())})
        differ = [token for token, index in expected.items() if vocab.get(token) != index]
        merges = [line for line in (folder / "merges.txt").read_text(encoding="utf-8").splitlines()
                  if line]
        # A "#version" header; other lines that start with "#" are merges ("# #").
        if merges and merges[0].startswith("#version"):
            merges = merges[1:]
        base_merges = [m if isinstance(m, str) else " ".join(m) for m in base["model"]["merges"]]
        if differ or merges != base_merges:
            raise ValueError(f"{folder}: the release's tokenizer is not the base model's "
                             f"({len(differ)} tokens differ, merges equal: {merges == base_merges})")

    def latent_config(self) -> dict:
        return {"width": self.width, "groups": GROUPS, "expert_scales": self.tail != 0,
                "column_scale": self._column_scales is not None}

    # --- experts ------------------------------------------------------------------------------

    def labels(self, layer: int) -> np.ndarray:
        return self._labels[layer]

    def expert_rows(self, layer: int, expert: int, role: str, first: int, last: int) -> np.ndarray:
        """Encoded rows ``uint8 [last - first, row bytes]`` of one projection of one expert."""

        if role == "down":
            offset, rows, row_bytes = 2 * self.middle * self.gate_up_row, self.width, self.down_row
        else:
            offset = {"gate": 0, "up": self.middle * self.gate_up_row}[role]
            rows, row_bytes = self.middle, self.gate_up_row
        if not 0 <= first <= last <= rows:
            raise ValueError(f"expert rows [{first},{last}) are outside {rows}")
        base = (layer * self.experts + expert) * self.record + offset
        raw = self._experts.read(base + first * row_bytes, (last - first) * row_bytes)
        type_id = self.down_type if role == "down" else self.gate_up_type
        # ggml's CPU decoder reads bits 0-6 of an NVFP4 scale byte and its CUDA decoder all eight
        # (a signed E4M3 conversion): a byte with bit 7 set has no single value.
        if type_id == GGML_NVFP4 and (raw.reshape(-1, 36)[:, :4] & 0x80).any():
            raise ValueError(f"{self.path}: layer {layer} expert {expert} {role} has an NVFP4 "
                             "scale byte with bit 7 set")
        return raw.reshape(last - first, row_bytes)

    def expert_scales(self, layer: int) -> np.ndarray:
        """FP32 ``[experts, 3]``: the gate, up and down output multipliers of every expert."""

        if not self.tail:
            raise ValueError(f"{self.path}: these experts carry no output multipliers")
        raw = np.stack([self._experts.read((layer * self.experts + expert) * self.record
                                           + self.tail_offset, NVFP4_TAIL)
                        for expert in range(self.experts)])
        tails = raw.view("<f4").reshape(self.experts, 4)
        if not np.isfinite(tails).all() or (tails[:, :3] <= 0).any() or (tails[:, 3] != 0).any():
            raise ValueError(f"{self.path}: layer {layer} has an expert tail that is not three "
                             "positive finite multipliers and a zero")
        return np.ascontiguousarray(tails[:, :3])

    def column_scale(self, layer: int) -> np.ndarray:
        """FP32 ``[ELL]``: the multipliers of the encoder's output columns, the reciprocals of the
        stored scales (a scale that is not positive multiplies by one, as in the reference)."""

        if self._column_scales is None:
            raise ValueError(f"{self.path}: this release has no column scales")
        stored = np.fromfile(self._column_scales[layer], dtype="<f4")
        if stored.size != self.width or not np.isfinite(stored).all():
            raise ValueError(f"{self._column_scales[layer]}: expected {self.width} finite values")
        one = np.float32(1.0)
        positive = stored > 0
        return np.where(positive, one / np.where(positive, stored, one), one).astype("<f4")

    # --- projections --------------------------------------------------------------------------

    def dense_type(self, build: str) -> int:
        return _DENSE_FILES[build][1]

    def _projections(self, build: str) -> tuple[_Blob, int, int, int]:
        name, type_id = _DENSE_FILES[build]
        d_row, u_row = _row_bytes(type_id, self.hidden), _row_bytes(type_id, self.width)
        stride = GROUPS * self.width * d_row + GROUPS * self.hidden * u_row
        if build not in self._dense:
            self._dense[build] = _Blob(self.path / name, self.layers * stride)
        return self._dense[build], d_row, u_row, stride

    def encode_rows(self, build: str, layer: int, first: int, last: int) -> np.ndarray:
        """Rows of the encoder D ``[GROUPS * ELL, hidden]`` of one layer."""

        blob, d_row, _, stride = self._projections(build)
        if not 0 <= first <= last <= GROUPS * self.width:
            raise ValueError(f"encoder rows [{first},{last}) are outside {GROUPS * self.width}")
        raw = blob.read(layer * stride + first * d_row, (last - first) * d_row)
        return raw.reshape(last - first, d_row)

    def decode_rows(self, build: str, layer: int, first: int, last: int) -> np.ndarray:
        """Rows of the decoder U ``[GROUPS * hidden, ELL]`` of one layer."""

        blob, d_row, u_row, stride = self._projections(build)
        if not 0 <= first <= last <= GROUPS * self.hidden:
            raise ValueError(f"decoder rows [{first},{last}) are outside {GROUPS * self.hidden}")
        base = layer * stride + GROUPS * self.width * d_row
        raw = blob.read(base + first * u_row, (last - first) * u_row)
        return raw.reshape(last - first, u_row)

    # --- the MTP block ------------------------------------------------------------------------

    def mtp(self) -> MtpPack:
        if self._mtp is None:
            self._mtp = MtpPack(self.path / "mtp")
        return self._mtp


class MtpPack:
    """Strata's runtime files of the MTP block (its `tools/mtp_rt.py`): `dense.txt` names every
    tensor of `dense.bin` (``name kind rows cols offset bytes``), and `experts.bin` holds 512 Q2_0
    experts, each as the code plane of its interleaved gate and up rows (row ``2 r`` gate ``r``,
    row ``2 r + 1`` up ``r``), the code plane of its down rows, and then the two planes of their
    binary16 block scales."""

    def __init__(self, root: Path, hidden: int = HIDDEN, middle: int = EXPERT_WIDTH,
                 experts: int = MTP_EXPERTS) -> None:
        self.root = root
        self.hidden, self.middle = hidden, middle
        self.tensors: dict[str, tuple[str, int, int, int, int]] = {}
        for line in (root / "dense.txt").read_text().splitlines():
            if not line.strip():
                continue
            name, kind, rows, columns, offset, size = line.split()
            self.tensors[name] = (kind, int(rows), int(columns), int(offset), int(size))
        end = max(offset + size for _, _, _, offset, size in self.tensors.values())
        dense = root / "dense.bin"
        if not dense.is_file() or dense.stat().st_size < end:
            raise ValueError(f"{dense}: absent or shorter than its index")
        self._dense = _Blob(dense, dense.stat().st_size)
        self._cache: dict[str, np.ndarray] = {}
        if hidden % 64 or middle % 64:
            raise ValueError("Q2_0 rows hold whole blocks of 64 values")
        gate_up = 2 * middle * (hidden // 64)
        down = hidden * (middle // 64)
        self._planes = (gate_up * 16, down * 16, gate_up * 2, down * 2)
        self.record = sum(self._planes)
        self._experts = _Blob(root / "experts.bin", experts * self.record)

    def close(self) -> None:
        self._dense.close()
        self._experts.close()
        self._cache.clear()

    def _tensor(self, name: str, kind: str, shape: tuple[int, int], size: int) -> np.ndarray:
        if name not in self.tensors:
            raise ValueError(f"{self.root}/dense.txt: tensor {name} is absent")
        stored, rows, columns, offset, stored_size = self.tensors[name]
        if stored != kind or (rows, columns) != shape or stored_size != size:
            raise ValueError(f"{self.root}: {name} is {stored} [{rows}, {columns}] in "
                             f"{stored_size} bytes, expected {kind} {list(shape)} in {size}")
        if name not in self._cache:
            self._cache[name] = self._dense.read(offset, size)
        return self._cache[name]

    def q8_rows(self, name: str, shape: tuple[int, int]) -> np.ndarray:
        row_bytes = _row_bytes(GGML_Q8_0, shape[1])
        return self._tensor(name, "q8_0", shape, shape[0] * row_bytes).reshape(shape[0], row_bytes)

    def bf16_words(self, name: str, shape: tuple[int, int]) -> np.ndarray:
        return self._tensor(name, "bf16", shape, shape[0] * shape[1] * 2).view("<u2").reshape(shape)

    def f32(self, name: str, count: int) -> np.ndarray:
        return self._tensor(name, "f32", (1, count), count * 4).view("<f4")

    def expert_rows(self, expert: int, role: str) -> np.ndarray:
        """One projection of one expert as ggml Q2_0 rows: per block the binary16 scale and then
        the 16 code bytes, the block the planes were cut from."""

        base = expert * self.record
        codes_gu, codes_down, scales_gu, scales_down = self._planes
        if role == "down":
            rows, blocks = self.hidden, self.middle // 64
            codes = self._experts.read(base + codes_gu, codes_down).reshape(rows, blocks, 16)
            scales = self._experts.read(base + codes_gu + codes_down + scales_gu,
                                        scales_down).reshape(rows, blocks, 2)
        else:
            rows, blocks = self.middle, self.hidden // 64
            phase = {"gate": 0, "up": 1}[role]
            codes = self._experts.read(base, codes_gu).reshape(2 * rows, blocks, 16)[phase::2]
            scales = self._experts.read(base + codes_gu + codes_down, scales_gu).reshape(
                2 * rows, blocks, 2)[phase::2]
        out = np.empty((rows, blocks, 18), dtype=np.uint8)
        out[..., :2] = scales
        out[..., 2:] = codes
        return out.reshape(rows, blocks * 18)


# --- logical sources ------------------------------------------------------------------------------

def block_rows_source(shape: tuple[int, int], label: str, type_id: int, read) -> LogicalSource:
    """Encoded ggml rows from `read(first, last)` (uint8 ``[rows, row bytes]``), and their values
    on request."""

    stored = GGUF_FORMATS_BY_TYPE[type_id]
    empty = torch.empty(0, dtype=torch.float16)

    def encoded(first: int, last: int) -> EncodedRows:
        return EncodedRows(stored.name, torch.from_numpy(read(first, last)), empty)

    def values(first: int, last: int) -> torch.Tensor:
        return dequantize_blocks(type_id, read(first, last))

    return LogicalSource(shape, f"{label}[{stored.name}]{list(shape)}", _flat(values, shape[1]),
                         encoded)


def _direct(values: np.ndarray, dtype: torch.dtype, label: str) -> LogicalSource:
    return array_source(torch.from_numpy(np.array(values, copy=True)).to(dtype), label)


def _norm(stored: np.ndarray, label: str) -> LogicalSource:
    """A zero-centred norm the runtime files keep as ``1 + w`` in FP32. ``stored - 1`` is exact;
    it is the checkpoint's BF16 w unless ``1 + w`` was rounded, which this reports."""

    w = stored.astype(np.float32) - np.float32(1.0)
    kept = torch.from_numpy(w).to(torch.bfloat16)
    inexact = int((kept.float().numpy() + np.float32(1.0) != stored).sum())
    if inexact:
        print(f"{label}: {inexact} of {w.size} stored values are not 1 + a BF16 weight; "
              "the nearest BF16 weights are kept", flush=True)
    return array_source(kept, label)


def _q8_from_bf16(words: np.ndarray, label: str) -> LogicalSource:
    """A BF16 hyper-connection matrix as ggml Q8_0 rows (ggml's reference rounding), the form its
    kernels read (qwen4_exp_gguf.hc_matrix_source)."""

    shape = words.shape
    values = (words.astype(np.uint32) << 16).view(np.float32)
    blocks = quantize_q8_0(values)
    empty = torch.empty(0, dtype=torch.float16)

    def encoded(first: int, last: int) -> EncodedRows:
        return EncodedRows("gguf_q8_0", torch.from_numpy(blocks[first:last]), empty)

    def read(first: int, last: int) -> torch.Tensor:
        return torch.from_numpy(dequantize_q8_0(blocks[first:last]))

    return LogicalSource(shape, f"{label}[BF16 as Q8_0]{list(shape)}", _flat(read, shape[1]),
                         encoded)


def latent_sources(pack: Edge0Pack, build: str, layer: int, m: str
                   ) -> dict[str, tuple[LogicalSource, bool]]:
    """Every latent parameter of source layer `layer`'s MoE (prefix `m`); True marks encoded block
    rows."""

    out: dict[str, tuple[LogicalSource, bool]] = {}
    dense = pack.dense_type(build)
    out[m + "latent/encode"] = block_rows_source(
        (GROUPS * pack.width, pack.hidden), f"D[{layer}]", dense,
        lambda first, last: pack.encode_rows(build, layer, first, last)), True
    out[m + "latent/decode"] = block_rows_source(
        (GROUPS * pack.hidden, pack.width), f"U[{layer}]", dense,
        lambda first, last: pack.decode_rows(build, layer, first, last)), True
    out[m + "latent/labels"] = (
        _direct(pack.labels(layer), torch.int32, f"labels[{layer}]"), False)
    latent = pack.latent_config()
    if latent["expert_scales"]:
        out[m + "latent/expert_scales"] = (
            _direct(pack.expert_scales(layer), torch.float32, f"expert scales[{layer}]"), False)
    if latent["column_scale"]:
        out[m + "latent/column_scale"] = (
            _direct(pack.column_scale(layer), torch.float32, f"column scale[{layer}]"), False)
    for expert in range(pack.experts):
        e = m + f"experts/{expert}/"
        for role, shape, type_id in (
                ("gate", (pack.middle, pack.width), pack.gate_up_type),
                ("up", (pack.middle, pack.width), pack.gate_up_type),
                ("down", (pack.width, pack.middle), pack.down_type)):
            out[e + role] = block_rows_source(
                shape, f"{role}[{layer}][{expert}]", type_id,
                lambda first, last, expert=expert, role=role: pack.expert_rows(
                    layer, expert, role, first, last)), True
    return out


def mtp_sources(mtp: MtpPack) -> dict[str, tuple[LogicalSource, bool]]:
    """Every parameter of the `mtp` component from the release's runtime files; True marks encoded
    block rows."""

    out: dict[str, tuple[LogicalSource, bool]] = {}

    def q8(name: str, tensor: str, shape: tuple[int, int], stored=None, select=None):
        stored = shape if stored is None else stored
        if select is None:
            def read(first, last):
                return mtp.q8_rows(tensor, stored)[first:last]
        else:
            def read(first, last):
                return np.ascontiguousarray(mtp.q8_rows(tensor, stored)[select(first, last)])
        out[name] = block_rows_source(shape, tensor, GGML_Q8_0, read), True

    def norm(name: str, tensor: str, count: int):
        out[name] = _norm(mtp.f32(tensor, count), tensor), False

    def hc(prefix: str, stored: str, inject: bool):
        norm(prefix + "norm", stored + "hc_norm.weight", WIDTH)
        roles = [("down", "input_mix_weight_down", (LOWRANK, WIDTH)),
                 ("up", "input_mix_weight_up", (WIDTH, LOWRANK))]
        if inject:
            roles.append(("inject", "block_inject_weight", (STREAMS, WIDTH)))
        for role, tensor, shape in roles:
            tensor = stored + tensor + ".weight"
            out[prefix + role] = _q8_from_bf16(mtp.bf16_words(tensor, shape), tensor), True

    def bf16(name: str, tensor: str, shape: tuple[int, int]):
        words = np.ascontiguousarray(mtp.bf16_words(tensor, shape))
        out[name] = array_source(torch.from_numpy(words.view(np.int16)).view(torch.bfloat16),
                                 tensor), False

    norm("mtp/embedding_norm", "pre_fc_norm_embedding.weight", HIDDEN)
    norm("mtp/hidden_norm", "pre_fc_norm_hidden.weight", WIDTH)
    q8("mtp/fc_embedding", "fc_embedding.weight", (HIDDEN, HIDDEN))
    q8("mtp/fc_hidden", "fc_hidden.weight", (HIDDEN, HIDDEN))
    hc("mtp/final_mixer/", "hyper_connection_mixer.", inject=False)
    p = "mtp/layer/"
    hc(p + "attn_hc/", "attn_hyper_connection.", inject=True)
    hc(p + "mlp_hc/", "mlp_hyper_connection.", inject=True)
    a = p + "attention/"
    joint = (2 * QUERY_ROWS, HIDDEN)
    q8(a + "query", "self_attn.q_proj.weight", (QUERY_ROWS, HIDDEN), joint, attention_rows(False))
    q8(a + "gate", "self_attn.q_proj.weight", (QUERY_ROWS, HIDDEN), joint, attention_rows(True))
    q8(a + "key", "self_attn.k_proj.weight", (KV_ROWS, HIDDEN))
    q8(a + "value", "self_attn.v_proj.weight", (KV_ROWS, HIDDEN))
    q8(a + "output", "self_attn.o_proj.weight", (HIDDEN, QUERY_ROWS))
    norm(a + "query_norm", "self_attn.q_norm.weight", 256)
    norm(a + "key_norm", "self_attn.k_norm.weight", 256)
    i = p + "indexer/"
    query, key = INDEXER_HEADS * INDEXER_DIM, INDEXER_DIM
    index = "self_attn.indexer.index_qk_proj.weight"
    q8(i + "query", index, (query, HIDDEN), (query + key, HIDDEN),
       lambda first, last: np.arange(first, last))
    q8(i + "key", index, (key, HIDDEN), (query + key, HIDDEN),
       lambda first, last: np.arange(first, last) + query)
    norm(i + "query_norm", "self_attn.indexer.q_layernorm.weight", INDEXER_DIM)
    norm(i + "key_norm", "self_attn.indexer.k_layernorm.weight", INDEXER_DIM)
    m = p + "moe/"
    bf16(m + "router", "mlp.gate.weight", (MTP_EXPERTS, HIDDEN))
    bf16(m + "shared_score", "mlp.shared_expert_gate.weight", (1, HIDDEN))
    q8(m + "shared/gate", "mlp.shared_expert.gate_proj.weight", (SHARED_WIDTH, HIDDEN))
    q8(m + "shared/up", "mlp.shared_expert.up_proj.weight", (SHARED_WIDTH, HIDDEN))
    q8(m + "shared/down", "mlp.shared_expert.down_proj.weight", (HIDDEN, SHARED_WIDTH))
    for expert in range(MTP_EXPERTS):
        e = m + f"experts/{expert}/"
        for role, shape in (("gate", (EXPERT_WIDTH, HIDDEN)), ("up", (EXPERT_WIDTH, HIDDEN)),
                            ("down", (HIDDEN, EXPERT_WIDTH))):
            out[e + role] = block_rows_source(
                shape, f"mtp {role}[{expert}]", GGML_Q2_0,
                lambda first, last, expert=expert, role=role: mtp.expert_rows(
                    expert, role)[first:last]), True
    return out


# --- the recipe -----------------------------------------------------------------------------------

def with_latent_moe(config: dict, pack: Edge0Pack) -> dict:
    """The base model's HF config with the pack's latent MoE."""

    out = deepcopy(config)
    text = out.get("text_config", out)
    if text.get("num_experts") != pack.experts or text.get("moe_intermediate_size") != pack.middle:
        raise ValueError(f"{pack.path}: {pack.experts} experts of width {pack.middle} are not "
                         "this model's")
    if text.get("num_hidden_layers") != pack.layers or text.get("hidden_size") != pack.hidden:
        raise ValueError(f"{pack.path}: the pack's layers or hidden width are not this model's")
    text["latent_moe"] = pack.latent_config()
    return out


def _configure(model, recipe, sources, build: str | None) -> None:
    from .qwen4_exp_ngram import configure_ngram

    if "text" not in model.components:
        configure_ngram(model, recipe, sources["ngram"])
        return
    pack = sources["pack"]
    if not isinstance(pack, Edge0Pack):
        raise ValueError("--source pack must name an Edge0 release folder")
    config = model.config
    if config.get("latent_moe") != pack.latent_config():
        raise ValueError("the model's latent MoE differs from the pack's (convert with "
                         "--source pack, which sets it)")
    build = pack.released_dense if build is None else build
    gguf = sources["gguf"]
    if not isinstance(gguf, GGUFFile):
        raise ValueError("--source gguf must name the release's model-skeleton.gguf")
    validate(gguf, config, model.source_layers, routed=False)
    found = text_sources(gguf, config, model.source_layers, routed=False)
    source_layers = model.source_layers or tuple(range(LAYERS))
    for layer, source_layer in enumerate(source_layers):
        found |= latent_sources(pack, build, source_layer, f"text/layers/{layer}/moe/")
    if "mtp" in model.components:
        # Iterate: a source mapping opens a name it is asked whether it holds.
        mtp = sources["mtp"] if "mtp" in tuple(sources) else None
        if isinstance(mtp, GGUFFile):
            from .qwen4_exp_gguf import mtp_sources as gguf_mtp_sources, validate_mtp

            experts = model.components["mtp"]["config"]["num_experts"]
            validate_mtp(mtp, experts)
            found |= gguf_mtp_sources(mtp, experts)
        elif mtp is None:
            if model.components["mtp"]["config"]["num_experts"] != MTP_EXPERTS:
                raise ValueError(f"the release's MTP block keeps {MTP_EXPERTS} experts")
            found |= mtp_sources(pack.mtp())
        else:
            raise ValueError("--source mtp must name an MTP GGUF (default: the release's mtp/)")
    formats: dict[str, str] = {}
    for name, (source, encoded) in found.items():
        formats[name] = _assign(recipe, name, source, encoded, model)
    if "vision" in model.components:
        from .gguf_blocks import _vision_source

        vision = _vision_source(model, sources)
        for name, parameter in model.parameters.items():
            if name.startswith("vision/"):
                formats[name] = parameter.direct_format
                recipe.assign(name, format=parameter.direct_format, method=cast_direct,
                              source=model.source(name, vision))
    if set(formats) != set(model.parameters) - {"ngram/table"}:
        missing = sorted(set(model.parameters) - set(formats) - {"ngram/table"})[:5]
        extra = sorted(set(formats) - set(model.parameters))[:5]
        raise ValueError(f"the release leaves logical parameters without a source: {missing} "
                         f"(unknown: {extra})")
    # A malformed model, missing component or invalid source must fail before the large scan.
    if "ngram" in model.components:
        configure_ngram(model, recipe, sources["ngram"])
    for layer in range(len(source_layers)):
        p = f"text/layers/{layer}/"
        if config["layer_types"][layer] == "full_attention":
            _group_attention(recipe, p, formats)
        else:
            n = p + "gdn/"
            qkv = [n + "query", n + "key", n + "value"]
            recipe.group(qkv + [n + "z"] if formats[n + "z"] == formats[n + "query"] else qkv)
            recipe.group([n + "a_projection", n + "b_projection"])
        # Each expert's [gate; up] rows and down rows in expert order: one bank each per layer.
        _group_moe(recipe, p + "moe/", config["num_experts"], formats)
        if layer in config["ple_layers"]:
            _group_same_format(recipe, [p + "ple/key", p + "ple/value"], formats)
    if "mtp" in model.components:
        _group_attention(recipe, "mtp/layer/", formats)
        _group_moe(recipe, "mtp/layer/moe/", model.components["mtp"]["config"]["num_experts"],
                   formats)


def edge0_pack(model, recipe, sources):
    """An Edge0 release folder (`--source pack=FOLDER`) with the projections its launcher selects
    (Q8_0 for Edge0-145B, Q4_0 for Edge0-120B-NVFP4). The skeleton, the MTP block and the n-gram
    table come from the folder unless `--source gguf`, `--source mtp` or `--source ngram` names
    another file; `--model` supplies the base Qwen3.8-Flash-Next config and frontend files, which
    must carry the folder's vocabulary and merges."""

    _configure(model, recipe, sources, None)


def edge0_pack_q8(model, recipe, sources):
    """An Edge0 release folder with its Q8_0 projections (`latent_dense_q8.bin`)."""

    _configure(model, recipe, sources, "q8")


def edge0_pack_q4(model, recipe, sources):
    """An Edge0 release folder with its Q4_0 projections (`latent_dense_q4.bin`)."""

    _configure(model, recipe, sources, "q4")


RECIPES = {
    "edge0_pack": edge0_pack,
    "edge0_pack_q8": edge0_pack_q8,
    "edge0_pack_q4": edge0_pack_q4,
}

__all__ = [
    "Edge0Pack",
    "GROUPS",
    "MtpPack",
    "RECIPES",
    "block_rows_source",
    "edge0_pack",
    "edge0_pack_q4",
    "edge0_pack_q8",
    "latent_sources",
    "mtp_sources",
    "with_latent_moe",
]
