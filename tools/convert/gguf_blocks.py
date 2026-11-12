"""A Qwen3.5/3.8 dense GGUF kept in its own ggml block formats (GSQ-RCO mixed-precision releases).

Every text projection, the token table, the output head and the MTP head of an ``-mtp`` build are
stored as the exporter's own blocks, one ggml type per tensor, so no weight is re-quantised: a row
of the artifact is byte for byte a row of the GGUF. An untied BF16 output head, which some
community builds keep, stays BF16, its words exact. llama.cpp's Qwen3.5 exporter conventions are
undone where they touch rows or small tensors: value heads return to the grouped order,
zero-centred norms drop their stored ``1 + w`` and ``ssm_a`` becomes ``A_log`` again.

``ssm_out`` is the exception. Its tiled value heads are its input columns, which a row copy cannot
move without re-quantising. Its encoded rows keep the stored order, and their import gives the Use an
``input_columns`` auxiliary: column ``c`` of the stored matrix multiplies element ``input_columns[c]``
of the grouped activation. Its values, like every source's, are in the grouped order, so a recipe
that re-quantises ``ssm_out`` writes grouped columns and no gather.

The GDN controls ``ssm_alpha`` and ``ssm_beta`` are BF16 operands of the Engine. Exporters store
them as BF16 words or as Q8_0 blocks (Underdog-Saluki and similar releases), whose values are
rounded to BF16: the one place a stored value changes.

Vision comes from the release's ``mmproj`` GGUF (``--source vision=mmproj.gguf``) with the official
Vision formats, DFlash2 from ``--source dflash2`` and the frontend resources from ``--model``, which
needs only the base checkpoint's config and tokenizer files.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from tools.artifact.formats import GGUF_FORMATS_BY_TYPE

from .methods import cast_direct, grouped_absmax, import_encoded
from .official_recipes import Q4, Q5, Q6, Q8
from .sources.gguf import TYPE_BF16, GGUFFile
from .sources.logical import EncodedRows, LogicalSource, array_source
from .ternary import (
    ATTENTION_HEAD_DIM,
    ATTENTION_KV_ROWS,
    ATTENTION_QUERY_ROWS,
    GDN_CHANNELS,
    GDN_HEAD_DIM,
    GDN_KEY_DIM,
    GDN_KEY_HEADS,
    GDN_TAPS,
    GDN_VALUE_DIM,
    GDN_VALUE_HEADS,
    HIDDEN,
    INTERMEDIATE,
    LAYERS,
    VOCABULARY,
    RowMap,
    _direct,
    _flat,
    _norm,
    attention_rows,
    full_attention,
    rows,
    untile,
    untiled_rows,
)

MTP_BLOCK = LAYERS
# The storage the GDN controls may have (gdn_control_source).
GDN_CONTROL_TYPES = ("BF16", "Q8_0")
# The storage the output head may have: stored blocks, or BF16 words (bf16_source).
HEAD_TYPES = ("blocks", "BF16")

EXPECTED_HEADER = {
    "general.architecture": "qwen35",
    "qwen35.embedding_length": HIDDEN,
    "qwen35.feed_forward_length": INTERMEDIATE,
    "qwen35.attention.head_count": 24,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": ATTENTION_HEAD_DIM,
    "qwen35.attention.value_length": ATTENTION_HEAD_DIM,
    "qwen35.ssm.conv_kernel": GDN_TAPS,
    "qwen35.ssm.state_size": GDN_HEAD_DIM,
    "qwen35.ssm.group_count": GDN_KEY_HEADS,
    "qwen35.ssm.time_step_rank": GDN_VALUE_HEADS,
    "qwen35.ssm.inner_size": GDN_VALUE_DIM,
    "qwen35.full_attention_interval": 4,
}


def _attention_tensors(prefix: str) -> dict[str, tuple[tuple[int, ...], str]]:
    return {
        prefix + "attn_q.weight": ((2 * ATTENTION_QUERY_ROWS, HIDDEN), "blocks"),
        prefix + "attn_k.weight": ((ATTENTION_KV_ROWS, HIDDEN), "blocks"),
        prefix + "attn_v.weight": ((ATTENTION_KV_ROWS, HIDDEN), "blocks"),
        prefix + "attn_output.weight": ((HIDDEN, ATTENTION_QUERY_ROWS), "blocks"),
        prefix + "attn_q_norm.weight": ((ATTENTION_HEAD_DIM,), "F32"),
        prefix + "attn_k_norm.weight": ((ATTENTION_HEAD_DIM,), "F32"),
    }


def _layer_tensors(prefix: str) -> dict[str, tuple[tuple[int, ...], str]]:
    return {
        prefix + "attn_norm.weight": ((HIDDEN,), "F32"),
        prefix + "post_attention_norm.weight": ((HIDDEN,), "F32"),
        prefix + "ffn_gate.weight": ((INTERMEDIATE, HIDDEN), "blocks"),
        prefix + "ffn_up.weight": ((INTERMEDIATE, HIDDEN), "blocks"),
        prefix + "ffn_down.weight": ((HIDDEN, INTERMEDIATE), "blocks"),
    }


def expected_tensors(mtp: bool) -> dict[str, tuple[tuple[int, ...], str | tuple[str, ...]]]:
    """Row-major shape and kinds of every tensor: "blocks" (any stored ggml block type) and the
    ggml type names it may have."""

    out = {
        "token_embd.weight": ((VOCABULARY, HIDDEN), "blocks"),
        "output.weight": ((VOCABULARY, HIDDEN), HEAD_TYPES),
        "output_norm.weight": ((HIDDEN,), "F32"),
    }
    for layer in range(LAYERS):
        p = f"blk.{layer}."
        out |= _layer_tensors(p)
        if full_attention(layer):
            out |= _attention_tensors(p)
            continue
        out |= {
            p + "attn_qkv.weight": ((GDN_CHANNELS, HIDDEN), "blocks"),
            p + "attn_gate.weight": ((GDN_VALUE_DIM, HIDDEN), "blocks"),
            p + "ssm_out.weight": ((HIDDEN, GDN_VALUE_DIM), "blocks"),
            p + "ssm_alpha.weight": ((GDN_VALUE_HEADS, HIDDEN), GDN_CONTROL_TYPES),
            p + "ssm_beta.weight": ((GDN_VALUE_HEADS, HIDDEN), GDN_CONTROL_TYPES),
            p + "ssm_a": ((GDN_VALUE_HEADS,), "F32"),
            p + "ssm_dt.bias": ((GDN_VALUE_HEADS,), "F32"),
            p + "ssm_conv1d.weight": ((GDN_CHANNELS, GDN_TAPS), "F32"),
            p + "ssm_norm.weight": ((GDN_HEAD_DIM,), "F32"),
        }
    if mtp:
        p = f"blk.{MTP_BLOCK}."
        out |= _layer_tensors(p) | _attention_tensors(p)
        out |= {
            p + "nextn.eh_proj.weight": ((HIDDEN, 2 * HIDDEN), "blocks"),
            p + "nextn.enorm.weight": ((HIDDEN,), "F32"),
            p + "nextn.hnorm.weight": ((HIDDEN,), "F32"),
            p + "nextn.shared_head_norm.weight": ((HIDDEN,), "F32"),
        }
    return out


def has_mtp(gguf: GGUFFile) -> bool:
    return f"blk.{MTP_BLOCK}.nextn.eh_proj.weight" in gguf.tensors


def validate(gguf: GGUFFile) -> None:
    """Refuse any GGUF that is not a dense Qwen3.5/3.8-27B text tower of stored block types."""

    for key, expected in EXPECTED_HEADER.items():
        if gguf.kv.get(key) != expected:
            raise ValueError(
                f"{gguf.path}: {key} = {gguf.kv.get(key)!r}, expected {expected!r}"
            )
    mtp = has_mtp(gguf)
    blocks = gguf.kv.get("qwen35.block_count")
    if blocks != LAYERS + (1 if mtp else 0):
        raise ValueError(f"{gguf.path}: {blocks} blocks, expected {LAYERS} (+1 for MTP)")
    expected = expected_tensors(mtp)
    if set(gguf.tensors) != set(expected):
        missing = sorted(set(expected) - set(gguf.tensors))[:5]
        extra = sorted(set(gguf.tensors) - set(expected))[:5]
        raise ValueError(
            f"{gguf.path}: tensor set mismatch (missing {missing}, extra {extra})"
        )
    for name, (shape, kind) in expected.items():
        info = gguf.tensors[name]
        kinds = (kind,) if isinstance(kind, str) else kind
        accepted = info.type_name in kinds or (
            "blocks" in kinds and info.type_id in GGUF_FORMATS_BY_TYPE
        )
        if info.shape != shape or not accepted:
            raise ValueError(f"{gguf.path}: {name} is {info.type_name} {info.shape}")
    end = max(info.offset + info.nbytes for info in gguf.tensors.values())
    if gguf.data_bytes_available < end:
        raise ValueError(f"{gguf.path}: the data section is truncated")


def block_format(gguf: GGUFFile, tensor: str) -> str:
    return GGUF_FORMATS_BY_TYPE[gguf.info(tensor).type_id].name


GGML_Q8_0 = 8
GGML_Q2_0 = 42


def dequantize_q2_0(blocks: np.ndarray) -> np.ndarray:
    """ggml Q2_0 rows: 64 values per 18-byte block, a binary16 d then 16 code bytes; value j is
    d * (c - 1) with c the bit pair 2 (j % 4) of byte j / 4. gguf-py predates the type."""

    rows = blocks.shape[0]
    b = np.ascontiguousarray(blocks).reshape(rows, -1, 18)
    d = b[..., :2].copy().view("<f2").astype(np.float32)
    q = b[..., 2:]
    codes = np.stack([(q >> (2 * i)) & 3 for i in range(4)], axis=-1).reshape(rows, b.shape[1], 64)
    return ((codes.astype(np.float32) - 1.0) * d).reshape(rows, -1)


def dequantize_q8_0(blocks: np.ndarray) -> np.ndarray:
    """ggml Q8_0 rows: 32 values per 34-byte block, a binary16 d then 32 signed code bytes q;
    value j is d * q[j], exact in FP32."""

    rows = blocks.shape[0]
    b = np.ascontiguousarray(blocks).reshape(rows, -1, 34)
    d = b[..., :2].copy().view("<f2").astype(np.float32)
    q = b[..., 2:].copy().view(np.int8).astype(np.float32)
    return (q * d).reshape(rows, -1)


def quantize_q8_0(values: np.ndarray) -> np.ndarray:
    """ggml quantize_row_q8_0_ref over rows of whole blocks, in FP32: per 32 values a binary16
    d = amax / 127 and codes round(x / d), halves away from zero. Returns [rows, row bytes]."""

    rows = values.shape[0]
    blocks = np.ascontiguousarray(values, dtype=np.float32).reshape(rows, -1, 32)
    d = np.abs(blocks).max(axis=-1) / np.float32(127.0)
    inverse = np.divide(np.float32(1.0), d, out=np.zeros_like(d), where=d != 0)
    scaled = blocks * inverse[..., None]
    codes = (np.sign(scaled) * np.floor(np.abs(scaled) + np.float32(0.5))).astype(np.int8)
    out = np.empty((rows, blocks.shape[1], 34), dtype=np.uint8)
    out[..., :2] = d.astype("<f2")[..., None].view(np.uint8)
    out[..., 2:] = codes.view(np.uint8)
    return out.reshape(rows, -1)


def dequantize_q4_0(blocks: np.ndarray) -> np.ndarray:
    """ggml Q4_0 rows: 32 values per 18-byte block, a binary16 d then 16 code bytes; value j is
    d * (c - 8), c the low nibble of byte j for j < 16 and the high nibble of byte j - 16 after."""

    rows = blocks.shape[0]
    b = np.ascontiguousarray(blocks).reshape(rows, -1, 18)
    d = b[..., :2].copy().view("<f2").astype(np.float32)
    q = b[..., 2:]
    codes = np.concatenate([q & 15, q >> 4], axis=-1).astype(np.float32) - np.float32(8.0)
    return (codes * d).reshape(rows, -1)


# ggml's E2M1 code values, doubled (kvalues_mxfp4), and UE4M3 scale bytes halved to match.
_NVFP4_VALUES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
                         dtype=np.float32)


def ue4m3_half(words: np.ndarray) -> np.ndarray:
    """ggml_ue4m3_to_fp32: half the unsigned E4M3 value (bias 7) of each byte, which reads bits
    0-6 only; 0x00 and 0x7F are zero."""

    x = np.asarray(words, dtype=np.uint8)
    exponent = ((x >> 3) & 15).astype(np.int32)
    mantissa = (x & 7).astype(np.float32)
    raw = np.where(exponent == 0, np.ldexp(mantissa, -9),
                   np.ldexp(np.float32(1.0) + mantissa / np.float32(8.0), exponent - 7))
    return np.where((x == 0) | (x == 0x7F), np.float32(0.0),
                    raw.astype(np.float32) * np.float32(0.5))


def dequantize_nvfp4(blocks: np.ndarray) -> np.ndarray:
    """ggml NVFP4 rows: 64 values per 36-byte block, four UE4M3 scale bytes (one per 16 values)
    then 32 code bytes; value j of sub-block s is kvalues[c] * ue4m3_half(d[s]), c the low nibble of
    byte 8 s + j for j < 8 and the high nibble of byte 8 s + j - 8 after. Exact in FP32."""

    rows = blocks.shape[0]
    b = np.ascontiguousarray(blocks).reshape(rows, -1, 36)
    scale = ue4m3_half(b[..., :4])[..., None]
    q = b[..., 4:].reshape(rows, b.shape[1], 4, 8)
    codes = np.concatenate([q & 15, q >> 4], axis=-1)
    return (_NVFP4_VALUES[codes] * scale).reshape(rows, -1)


GGML_Q4_0 = 2
GGML_NVFP4 = 40
_LOCAL_DECODERS = {GGML_Q2_0: dequantize_q2_0, GGML_Q4_0: dequantize_q4_0,
                   GGML_Q8_0: dequantize_q8_0, GGML_NVFP4: dequantize_nvfp4}


def dequantize_blocks(type_id: int, blocks: np.ndarray) -> torch.Tensor:
    """FP32 values ``[rows, columns]`` of encoded ggml rows ``blocks`` (``uint8 [rows, row
    bytes]``) of block type `type_id`: local decoders for Q2_0, Q4_0, Q8_0 and NVFP4, gguf-py's
    for the others."""

    rows = blocks.shape[0]
    local = _LOCAL_DECODERS.get(type_id)
    if local is not None:
        return torch.from_numpy(np.ascontiguousarray(local(blocks))).reshape(rows, -1)
    try:
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
    except ImportError as error:
        raise ValueError(
            "reading the values of a GGUF block matrix needs the `gguf` package; "
            "storing its blocks does not"
        ) from error
    values = dequantize(np.ascontiguousarray(blocks), GGMLQuantizationType(type_id))
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)).reshape(rows, -1)


def _dequantize(gguf: GGUFFile, tensor: str, first: int, last: int) -> torch.Tensor:
    return _dequantize_rows(gguf, tensor, gguf.read_blocks(tensor, first, last))


def _dequantize_rows(gguf: GGUFFile, tensor: str, blocks: np.ndarray) -> torch.Tensor:
    """Values of encoded rows ``blocks`` (``uint8 [rows, row bytes]``) of a stored block matrix."""

    return dequantize_blocks(gguf.info(tensor).type_id, blocks)


def block_source(
    gguf: GGUFFile,
    tensor: str,
    shape: tuple[int, int],
    select: RowMap,
    columns: np.ndarray | None = None,
) -> LogicalSource:
    """Rows of a stored block matrix as encoded ggml rows, and their values on request.

    ``columns`` describes a matrix stored over permuted input columns: stored column ``c``
    multiplies logical input element ``columns[c]``. The encoded rows keep the stored order and
    carry it (EncodedRows.input_columns) to the Use that imports them; the values return to the
    logical order, which every value-reading method quantises.
    """

    format = block_format(gguf, tensor)
    empty = torch.empty(0, dtype=torch.float16)
    gather, logical = None, None
    if columns is not None:
        columns = np.asarray(columns, dtype=np.int64)
        if not np.array_equal(np.sort(columns), np.arange(shape[1])):
            raise ValueError(f"{tensor}: stored input columns must permute [0, {shape[1]})")
        gather = columns.astype("<i4").tobytes()
        # Logical column g is the stored column that multiplies element g.
        logical = torch.from_numpy(np.argsort(columns))

    def encoded(begin: int, end: int) -> EncodedRows:
        blocks = gguf.read_rows(tensor, select(begin, end))
        return EncodedRows(format, torch.from_numpy(blocks), empty, input_columns=gather)

    def values(first: int, last: int) -> torch.Tensor:
        stored = _dequantize_rows(gguf, tensor, gguf.read_rows(tensor, select(first, last)))
        return stored if logical is None else stored[:, logical]

    return LogicalSource(
        shape, f"{tensor}[{format}]{list(shape)}", _flat(values, shape[1]), encoded
    )


def bf16_source(gguf: GGUFFile, tensor: str, shape: tuple[int, int]) -> LogicalSource:
    """A stored BF16 matrix as its exact words, read a run of rows at a time."""

    def values(first: int, last: int) -> torch.Tensor:
        words = gguf.read_bf16_words(tensor, first, last)
        return torch.from_numpy(words.view(np.int16)).view(torch.bfloat16)

    return LogicalSource(shape, f"{tensor}[BF16]{list(shape)}", _flat(values, shape[1]))


def head_source(gguf: GGUFFile) -> tuple[LogicalSource, bool]:
    """The output head: stored blocks as encoded rows (True), or BF16 words as values (False)."""

    shape = gguf.info("output.weight").shape
    if gguf.info("output.weight").type_id == TYPE_BF16:
        return bf16_source(gguf, "output.weight", shape), False
    return block_source(gguf, "output.weight", shape, rows()), True


def gdn_control_source(gguf: GGUFFile, tensor: str) -> LogicalSource:
    """A GDN A/B control as its BF16 operand, its value heads in the grouped order: BF16 words
    as stored, Q8_0 values rounded to BF16."""

    if gguf.info(tensor).type_id == GGML_Q8_0:
        values = untile(_dequantize(gguf, tensor, 0, GDN_VALUE_HEADS).numpy(), 1)
        return array_source(
            torch.from_numpy(np.ascontiguousarray(values)).to(torch.bfloat16), tensor
        )
    words = untile(gguf.read_bf16_words(tensor), 1)
    return array_source(
        torch.from_numpy(np.ascontiguousarray(words.view(np.int16))).view(torch.bfloat16),
        tensor,
    )


def q2_native_source(
    gguf: GGUFFile, tensor: str, shape: tuple[int, int], select: RowMap
) -> LogicalSource:
    """Import GSQ Q2_0 as native offset codes and the exact signed binary16 scale words.

    The row map selects rows within a tensor or flattened expert bank. The physical K padding
    holds logical integer zero (stored code 1); wholly padding groups have positive-zero scales.
    No floating-point dequantization or quantizer participates in the encoded import.
    """
    if gguf.info(tensor).type_id != GGML_Q2_0:
        raise ValueError(f"{tensor}: native Q2 import requires stored Q2_0 blocks")
    if len(shape) != 2 or shape[1] != gguf.info(tensor).shape[-1] or shape[1] % 64:
        raise ValueError(f"{tensor}: native Q2 import requires complete 64-value blocks")
    raw_source = block_source(gguf, tensor, shape, select)
    k = shape[1]
    groups, padded_groups = k // 64, ((k + 127) // 128) * 2

    def encoded(begin: int, end: int) -> EncodedRows:
        blocks = gguf.read_rows(tensor, select(begin, end)).reshape(end - begin, groups, 18)
        scale_words = np.zeros((end - begin, padded_groups), dtype="<u2")
        scale_words[:, :groups] = np.ascontiguousarray(blocks[..., :2]).view("<u2").squeeze(-1)
        scales = scale_words.view("<f2")
        if not np.isfinite(scales).all():
            raise ValueError(f"{tensor}: Q2 scales must be finite binary16 words")
        codes = np.full((end - begin, padded_groups, 16), 0x55, dtype=np.uint8)
        codes[:, :groups] = blocks[..., 2:]
        return EncodedRows("q2_g64_fp16", torch.from_numpy(codes.reshape(end - begin, -1)),
                           torch.from_numpy(scales), packed=True)

    return LogicalSource(shape, f"{tensor}[q2_g64_fp16]{list(shape)}",
                         raw_source.read_values, encoded)


def q8_native_source(
    gguf: GGUFFile, tensor: str, shape: tuple[int, int], select: RowMap
) -> LogicalSource:
    """Preserve compatible Q8_0 integer codes and binary16 scales as native Q8 planes.

    Native Q8 excludes -128 and negative-zero scales, and requires zero codes for zero-scale
    groups. Refuse an incompatible source instead of silently changing its representation.
    """
    if gguf.info(tensor).type_id != 8:
        raise ValueError(f"{tensor}: native Q8 import requires stored Q8_0 blocks")
    if len(shape) != 2 or shape[1] != gguf.info(tensor).shape[-1] or shape[1] % 32:
        raise ValueError(f"{tensor}: native Q8 import requires complete 32-value blocks")
    raw_source = block_source(gguf, tensor, shape, select)
    groups, padded_groups = shape[1] // 32, ((shape[1] + 127) // 128) * 4

    def encoded(begin: int, end: int) -> EncodedRows:
        blocks = gguf.read_rows(tensor, select(begin, end)).reshape(end - begin, groups, 34)
        raw_words = np.ascontiguousarray(blocks[..., :2]).view("<u2").squeeze(-1)
        raw_scales = raw_words.view("<f2")
        raw_codes = blocks[..., 2:]
        if (not np.isfinite(raw_scales).all() or (raw_words == 0x8000).any() or
                (raw_codes == 0x80).any() or
                ((raw_scales == 0)[..., None] & (raw_codes != 0)).any()):
            raise ValueError(f"{tensor}: Q8_0 source is outside the native Q8 code/scale contract")
        scales = np.zeros((end - begin, padded_groups), dtype="<f2")
        scales[:, :groups] = raw_scales
        codes = np.zeros((end - begin, padded_groups, 32), dtype=np.uint8)
        codes[:, :groups] = raw_codes
        return EncodedRows("q8_g32_fp16", torch.from_numpy(codes.reshape(end - begin, -1)),
                           torch.from_numpy(scales), packed=True)

    return LogicalSource(shape, f"{tensor}[q8_g32_fp16]{list(shape)}",
                         raw_source.read_values, encoded)


def tiled_input_columns() -> np.ndarray:
    """For each stored ``ssm_out`` column (tiled value heads), its grouped activation element."""

    columns = np.arange(GDN_VALUE_DIM, dtype=np.int64)
    tiled_head, lane = columns // GDN_HEAD_DIM, columns % GDN_HEAD_DIM
    per_key = GDN_VALUE_HEADS // GDN_KEY_HEADS
    grouped_head = (tiled_head % GDN_KEY_HEADS) * per_key + tiled_head // GDN_KEY_HEADS
    return (grouped_head * GDN_HEAD_DIM + lane).astype(np.int32)


def _attention(
    gguf: GGUFFile, g: str, a: str
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    q_shape, kv_shape = (ATTENTION_QUERY_ROWS, HIDDEN), (ATTENTION_KV_ROWS, HIDDEN)
    encoded = {
        a + "query": block_source(gguf, g + "attn_q.weight", q_shape, attention_rows(False)),
        a + "gate": block_source(gguf, g + "attn_q.weight", q_shape, attention_rows(True)),
        a + "key": block_source(gguf, g + "attn_k.weight", kv_shape, rows()),
        a + "value": block_source(gguf, g + "attn_v.weight", kv_shape, rows()),
        a + "output": block_source(
            gguf, g + "attn_output.weight", (HIDDEN, ATTENTION_QUERY_ROWS), rows()
        ),
    }
    direct = {
        a + "query_norm": _norm(gguf, g + "attn_q_norm.weight", True),
        a + "key_norm": _norm(gguf, g + "attn_k_norm.weight", True),
    }
    return encoded, direct


def _layer(
    gguf: GGUFFile, g: str, p: str
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    encoded = {
        p + "mlp/gate": block_source(gguf, g + "ffn_gate.weight", (INTERMEDIATE, HIDDEN), rows()),
        p + "mlp/up": block_source(gguf, g + "ffn_up.weight", (INTERMEDIATE, HIDDEN), rows()),
        p + "mlp/down": block_source(
            gguf, g + "ffn_down.weight", (HIDDEN, INTERMEDIATE), rows()
        ),
    }
    direct = {
        p + "input_norm": _norm(gguf, g + "attn_norm.weight", True),
        p + "post_attention_norm": _norm(gguf, g + "post_attention_norm.weight", True),
    }
    return encoded, direct


def text_sources(
    gguf: GGUFFile, *, mtp: bool
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    """Encoded block sources and direct sources of every GGUF-backed text (and MTP) parameter; a
    BF16 output head is a direct source."""

    encoded = {
        "text/token_embedding": block_source(
            gguf, "token_embd.weight", (VOCABULARY, HIDDEN), rows()
        ),
    }
    direct = {"text/final_norm": _norm(gguf, "output_norm.weight", True)}
    head, stored_blocks = head_source(gguf)
    (encoded if stored_blocks else direct)["text/output_head"] = head
    for layer in range(LAYERS):
        g, p = f"blk.{layer}.", f"text/layers/{layer}/"
        e, d = _layer(gguf, g, p)
        encoded |= e
        direct |= d
        if full_attention(layer):
            e, d = _attention(gguf, g, p + "attention/")
            encoded |= e
            direct |= d
            continue
        n = p + "gdn/"
        qkv = g + "attn_qkv.weight"
        encoded[n + "query"] = block_source(gguf, qkv, (GDN_KEY_DIM, HIDDEN), rows())
        encoded[n + "key"] = block_source(gguf, qkv, (GDN_KEY_DIM, HIDDEN), rows(GDN_KEY_DIM))
        encoded[n + "value"] = block_source(
            gguf, qkv, (GDN_VALUE_DIM, HIDDEN), untiled_rows(2 * GDN_KEY_DIM)
        )
        encoded[n + "z"] = block_source(
            gguf, g + "attn_gate.weight", (GDN_VALUE_DIM, HIDDEN), untiled_rows(0)
        )
        # Tiled input columns: an import keeps them and its Use gathers the grouped activation.
        encoded[n + "output"] = block_source(
            gguf,
            g + "ssm_out.weight",
            (HIDDEN, GDN_VALUE_DIM),
            rows(),
            tiled_input_columns(),
        )
        for role, tensor in (("a_projection", "ssm_alpha.weight"), ("b_projection", "ssm_beta.weight")):
            direct[n + role] = gdn_control_source(gguf, g + tensor)
        ssm_a = untile(gguf.read_direct(g + "ssm_a"), 1).astype(np.float64)
        if not np.all(ssm_a < 0):
            raise ValueError(f"{g}ssm_a must be strictly negative (-exp(A_log))")
        direct[n + "a_log"] = _direct(
            np.log(-ssm_a).astype(np.float32), torch.float32, g + "ssm_a"
        )
        direct[n + "dt_bias"] = _direct(
            untile(gguf.read_direct(g + "ssm_dt.bias"), 1), torch.float32, g + "ssm_dt.bias"
        )
        taps = gguf.read_direct(g + "ssm_conv1d.weight")
        channels = np.concatenate(
            [taps[: 2 * GDN_KEY_DIM], untile(taps[2 * GDN_KEY_DIM :], GDN_HEAD_DIM)]
        )
        direct[n + "convolution"] = _direct(
            channels.T, torch.bfloat16, g + "ssm_conv1d.weight"
        )
        direct[n + "norm"] = _norm(gguf, g + "ssm_norm.weight", False)
    if mtp:
        g, p = f"blk.{MTP_BLOCK}.", "mtp/layers/0/"
        e, d = _layer(gguf, g, p)
        encoded |= e
        direct |= d
        e, d = _attention(gguf, g, p + "attention/")
        encoded |= e
        direct |= d
        encoded["mtp/input_projection"] = block_source(
            gguf, g + "nextn.eh_proj.weight", (HIDDEN, 2 * HIDDEN), rows()
        )
        for role, tensor in (
            ("embedding_norm", "nextn.enorm.weight"),
            ("hidden_norm", "nextn.hnorm.weight"),
            ("final_norm", "nextn.shared_head_norm.weight"),
        ):
            direct["mtp/" + role] = _norm(gguf, g + tensor, True)
    return encoded, direct


def _vision_format(name: str) -> str:
    if name == "vision/patch_embedding":
        return Q6
    if name.startswith("vision/merger/"):
        return Q8
    if name.endswith(("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")):
        return Q4
    return Q5


def _vision_source(model, sources):
    from .sources.gguf_source import GGUFSource, HFAliasSource, qwen35_vision_name_map

    mmproj = sources["vision"]
    if not isinstance(mmproj, GGUFSource):
        return mmproj
    config = model.components["vision"]["config"]
    names, transforms, shapes, composites = qwen35_vision_name_map(
        config["depth"],
        hidden_size=config["hidden_size"],
        in_channels=config.get("in_channels", 3),
        temporal_patch_size=config["temporal_patch_size"],
        patch_size=config["patch_size"],
    )
    return HFAliasSource(
        mmproj, names, transforms=transforms, shapes=shapes, composites=composites
    )


def _companions(model, recipe, sources) -> None:
    """Vision from the mmproj GGUF in the official formats; DFlash2 as the official recipe."""

    vision = _vision_source(model, sources) if "vision" in model.components else None
    for name, parameter in model.parameters.items():
        if name.startswith("vision/"):
            source = model.source(name, vision)
            if parameter.projection:
                recipe.assign(
                    name, format=_vision_format(name), method=grouped_absmax, source=source
                )
            else:
                recipe.assign(
                    name,
                    format=parameter.direct_format,
                    method=cast_direct,
                    source=source,
                )
        elif name.startswith(("dflash/", "dflash2/")) and parameter.projection:
            if name.endswith(
                (
                    "/attention_conv/kernel_projection",
                    "/mlp_conv/kernel_projection",
                    "/candidate_selector/hidden_projection",
                )
            ):
                continue
            recipe.assign(name, format=Q8, method=grouped_absmax)
    for backend in ("dflash", "dflash2"):
        if backend not in model.components:
            continue
        layers = model.components[backend]["config"]["num_hidden_layers"]
        for layer in range(layers):
            prefix = f"{backend}/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


def _group_same_format(recipe, names: list[str], formats: dict[str, str]) -> None:
    if len({formats[name] for name in names}) == 1:
        recipe.group(names)


def qwen3_8_27b_gguf(model, recipe, sources):
    """A Qwen3.8-27B GGUF in its own block formats from `--source gguf=MODEL.gguf`."""

    config = model.config
    if (
        "num_experts" in config
        or config["hidden_size"] != HIDDEN
        or config["num_hidden_layers"] != LAYERS
    ):
        raise ValueError("the GGUF block recipe requires the Qwen3.8-27B Dense geometry")
    gguf = sources["gguf"]
    if not isinstance(gguf, GGUFFile):
        raise ValueError("--source gguf must name the model's .gguf file")
    validate(gguf)
    mtp = "mtp" in model.components
    if mtp and not has_mtp(gguf):
        raise ValueError(f"{gguf.path}: the MTP component needs an -mtp GGUF (blk.64.nextn.*)")
    _companions(model, recipe, sources)
    encoded, direct = text_sources(gguf, mtp=mtp)
    formats = {}
    for name, source in encoded.items():
        format = source.read_encoded(0, 1).format
        formats[name] = format
        recipe.assign(
            name, format=format, method=import_encoded, source=source, activation_policy="AllowA8"
        )
    for name, source in direct.items():
        recipe.assign(
            name,
            format=model.parameters[name].direct_format,
            method=cast_direct,
            source=source,
        )
    prefixes = [f"text/layers/{layer}/" for layer in range(LAYERS)]
    if mtp:
        prefixes.append("mtp/layers/0/")
    for p in prefixes:
        if p.startswith("text/") and not full_attention(int(p.split("/")[2])):
            n = p + "gdn/"
            qkv = [n + "query", n + "key", n + "value"]
            recipe.group(qkv + [n + "z"] if formats[n + "z"] == formats[n + "query"] else qkv)
            recipe.separate([n + "a_projection", n + "b_projection"])
        else:
            a = p + "attention/"
            _group_same_format(
                recipe, [a + "query", a + "key", a + "gate", a + "value"], formats
            )
        _group_same_format(recipe, [p + "mlp/gate", p + "mlp/up"], formats)


RECIPES = {"qwen3_8_27b_gguf": qwen3_8_27b_gguf}

__all__ = [
    "EXPECTED_HEADER",
    "RECIPES",
    "block_format",
    "bf16_source",
    "block_source",
    "expected_tensors",
    "gdn_control_source",
    "has_mtp",
    "head_source",
    "qwen3_8_27b_gguf",
    "q2_native_source",
    "q8_native_source",
    "text_sources",
    "tiled_input_columns",
    "validate",
]
