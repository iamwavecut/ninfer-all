from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.artifact.formats import GGUF_FORMATS, GGUF_FORMATS_BY_TYPE
from tools.artifact.layouts import encoded_size, gguf_blocks_geometry
from tools.convert import gguf_blocks
from tools.convert.sources.gguf import GGUFFile, write_gguf

TYPE_Q8_0 = 8

# ggml-common.h: block values and bytes of every type the importer keeps.
GGML_BLOCKS = {
    2: (32, 18),
    6: (32, 22),
    8: (32, 34),
    10: (256, 84),
    11: (256, 110),
    12: (256, 144),
    13: (256, 176),
    14: (256, 210),
    16: (256, 66),
    17: (256, 74),
    18: (256, 98),
    19: (256, 50),
    20: (32, 18),
    21: (256, 110),
    22: (256, 82),
    23: (256, 136),
    29: (256, 56),
    39: (32, 17),
    40: (64, 36),
    42: (64, 18),
}


def _q8_0_file(tmp_path, rows: int, columns: int, seed: int, name: str = "w"):
    generator = np.random.default_rng(seed)
    blocks = generator.integers(0, 256, size=(rows, columns // 32, 34), dtype=np.uint8)
    scales = generator.uniform(0.01, 0.02, size=(rows, columns // 32)).astype(np.float16)
    blocks[:, :, :2] = scales.view(np.uint8).reshape(rows, columns // 32, 2)
    path = tmp_path / "blocks.gguf"
    write_gguf(
        path,
        {"general.architecture": "qwen35"},
        [(name, (rows, columns), TYPE_Q8_0, blocks.tobytes())],
    )
    return path, blocks


def _q8_0_values(blocks: np.ndarray) -> np.ndarray:
    """Independent ggml Q8_0 dequantization: binary16 d times each signed code byte."""

    scales = blocks[..., :2].copy().view(np.float16).astype(np.float32)
    codes = blocks[..., 2:].copy().view(np.int8).astype(np.float32)
    return (codes * scales).reshape(blocks.shape[0], -1)


def test_formats_are_ggml_blocks():
    assert {spec.ggml_type for spec in GGUF_FORMATS.values()} == set(GGML_BLOCKS)
    for type_id, (elements, size) in GGML_BLOCKS.items():
        spec = GGUF_FORMATS_BY_TYPE[type_id]
        assert (spec.block_elements, spec.block_bytes) == (elements, size)
        geometry = gguf_blocks_geometry(spec, (3, 4 * elements))
        assert geometry.row_bytes == 4 * size
        assert encoded_size("gguf_blocks_v1", spec, (3, 4 * elements)) == 12 * size
    with pytest.raises(ValueError):
        gguf_blocks_geometry("gguf_q4_k", (2, 128))


def test_rows_are_copied_byte_for_byte_through_the_row_map(tmp_path):
    path, blocks = _q8_0_file(tmp_path, 12, 64, 1)
    order = np.array([7, 2, 11, 2], dtype=np.int64)
    with GGUFFile(path) as gguf:
        source = gguf_blocks.block_source(gguf, "w", (4, 64), lambda b, e: order[b:e])
        words = source.read_encoded(0, 4)
        assert words.format == "gguf_q8_0"
        assert torch.equal(words.codes, torch.from_numpy(blocks.reshape(12, -1)[order]))
        assert torch.equal(
            source.read_encoded(1, 3).codes, torch.from_numpy(blocks.reshape(12, -1)[order[1:3]])
        )


def test_row_values_are_ggml_dequantization(tmp_path):
    path, blocks = _q8_0_file(tmp_path, 6, 64, 2)
    with GGUFFile(path) as gguf:
        source = gguf_blocks.block_source(gguf, "w", (6, 64), lambda b, e: np.arange(b, e))
        assert torch.equal(source.rows(3, 5), torch.from_numpy(_q8_0_values(blocks[3:5])))


def test_output_projection_columns_read_the_grouped_value_heads():
    columns = gguf_blocks.tiled_input_columns()
    assert sorted(columns.tolist()) == list(range(6144))
    for tiled_head in (0, 1, 15, 16, 31, 47):
        key_head, repeat = tiled_head % 16, tiled_head // 16
        assert columns[tiled_head * 128 + 5] == (key_head * 3 + repeat) * 128 + 5


def test_q8_0_gdn_controls_round_to_bf16_in_grouped_head_order(tmp_path):
    # Underdog-Saluki-style exports store ssm_alpha/ssm_beta as Q8_0 blocks: their values reach the
    # Engine's BF16 operand, the exporter's tiled value heads back in the grouped order.
    heads = gguf_blocks.GDN_VALUE_HEADS
    path, blocks = _q8_0_file(tmp_path, heads, 64, 5, "blk.0.ssm_alpha.weight")
    with GGUFFile(path) as gguf:
        values = gguf_blocks.gdn_control_source(gguf, "blk.0.ssm_alpha.weight").rows(0, heads)
    tiled = _q8_0_values(blocks)
    # Grouped head k * 3 + r is the exporter's tiled head r * 16 + k.
    grouped = [r * 16 + k for k in range(16) for r in range(3)]
    assert values.dtype == torch.bfloat16
    assert torch.equal(values, torch.from_numpy(tiled[grouped]).to(torch.bfloat16))


SSM_OUT = "blk.0.ssm_out.weight"
OUTPUT = "text/layers/0/gdn/output"
GROUPED = "text/layers/0/gdn/core_output"


def _ssm_out(gguf, rows: int):
    """The stored rows of a tiled ssm_out, values in the grouped order, as block_source reads it."""

    return gguf_blocks.block_source(
        gguf,
        SSM_OUT,
        (rows, gguf_blocks.GDN_VALUE_DIM),
        lambda b, e: np.arange(b, e),
        gguf_blocks.tiled_input_columns(),
    )


def _convert(tmp_path, source, configure):
    from tools.artifact.reader import Artifact
    from tools.convert.model import Model, Parameter
    from tools.convert.pipeline import convert
    from tools.convert.recipe import Recipe

    model = Model({"text": {"config": {}}})
    model.add(Parameter(OUTPUT, source.shape, source, inputs=(GROUPED,)))
    recipe = Recipe(model)
    configure(recipe)
    path = tmp_path / "gathered.ninfer"
    convert(model, recipe, path, device="cpu", rows_per_chunk=3)
    return Artifact(path)


def _matrix(artifact, name: str) -> tuple[str, np.ndarray | bytes]:
    from tools.artifact.codecs.direct import decode_direct
    from tools.artifact.codecs.row_split import dequantize_row_split

    obj = artifact.object(artifact.directory.bindings[name]["object"])
    raw = artifact.read_object(obj.id)
    if obj.format.startswith("gguf_"):
        return obj.format, raw
    if obj.format == "fp32":
        return obj.format, decode_direct(raw, obj.format, obj.shape).numpy()
    return obj.format, dequantize_row_split(raw, obj.format, obj.shape, dtype=torch.float32).numpy()


def _gather(artifact, name: str) -> np.ndarray | None:
    (use,) = [u for u in artifact.directory.uses if u["parameter"] == name]
    reference = use.get("auxiliaries", {}).get("input_columns")
    if reference is None:
        return None
    return np.frombuffer(artifact.read_object(reference["object"]), dtype="<i4")


def test_requantised_ssm_out_reads_the_grouped_activation_without_a_gather(tmp_path):
    # A value method writes ssm_out over grouped columns, so the native matrix times the grouped
    # activation x is the stored (tiled) matrix times x gathered through the stored order.
    from tools.convert.methods import cast_direct, grouped_absmax

    rows = 8
    path, blocks = _q8_0_file(tmp_path, rows, gguf_blocks.GDN_VALUE_DIM, 6, SSM_OUT)
    tiled = _q8_0_values(blocks).astype(np.float64)
    x = np.random.default_rng(7).standard_normal(gguf_blocks.GDN_VALUE_DIM)
    reference = tiled @ x[gguf_blocks.tiled_input_columns()]
    for format, method in (("fp32", cast_direct), ("q8_g32_fp16", grouped_absmax)):
        with GGUFFile(path) as gguf:
            source = _ssm_out(gguf, rows)
            with _convert(tmp_path / format, source, lambda recipe: recipe.assign(
                    OUTPUT, format=format, method=method, source=source)) as artifact:
                stored, native = _matrix(artifact, OUTPUT)
                assert stored == format
                assert _gather(artifact, OUTPUT) is None
        product = native.astype(np.float64) @ x
        if format == "fp32":
            # The exact values, each stored column moved to the grouped element it multiplies.
            assert np.array_equal(native[:, gguf_blocks.tiled_input_columns()], tiled)
            assert np.allclose(product, reference, rtol=1e-12, atol=1e-12)
        else:
            # Q8 rounding only; the stored column order would miss by the size of the product.
            assert np.linalg.norm(product - reference) < 1e-2 * np.linalg.norm(reference)


def test_imported_ssm_out_keeps_its_blocks_and_carries_the_gather(tmp_path):
    from tools.convert.methods import import_encoded

    rows = 4
    path, blocks = _q8_0_file(tmp_path, rows, gguf_blocks.GDN_VALUE_DIM, 8, SSM_OUT)
    with GGUFFile(path) as gguf:
        source = _ssm_out(gguf, rows)
        with _convert(tmp_path, source, lambda recipe: recipe.assign(
                OUTPUT, format="gguf_q8_0", method=import_encoded, source=source,
                activation_policy="AllowA8")) as artifact:
            stored, raw = _matrix(artifact, OUTPUT)
            gather = _gather(artifact, OUTPUT)
    assert stored == "gguf_q8_0" and raw == blocks.tobytes()
    assert np.array_equal(gather, gguf_blocks.tiled_input_columns())


def test_a_parameter_sharing_gathered_rows_reads_through_their_gather(tmp_path):
    from tools.artifact.reader import Artifact
    from tools.convert.methods import import_encoded
    from tools.convert.model import Model, Parameter
    from tools.convert.pipeline import convert
    from tools.convert.recipe import Recipe

    rows = 4
    path, _ = _q8_0_file(tmp_path, rows, gguf_blocks.GDN_VALUE_DIM, 9, SSM_OUT)
    shared = "text/layers/1/gdn/output"
    with GGUFFile(path) as gguf:
        source = _ssm_out(gguf, rows)
        model = Model({"text": {"config": {}}})
        model.add(Parameter(OUTPUT, source.shape, source, inputs=(GROUPED,)))
        model.add(Parameter(shared, source.shape, source, inputs=("text/layers/1/gdn/core",)))
        recipe = Recipe(model)
        recipe.assign(OUTPUT, format="gguf_q8_0", method=import_encoded, source=source)
        recipe.share(shared, OUTPUT)
        convert(model, recipe, tmp_path / "shared.ninfer", device="cpu")
    with Artifact(tmp_path / "shared.ninfer") as artifact:
        assert np.array_equal(_gather(artifact, shared), gguf_blocks.tiled_input_columns())


def test_a_gather_on_any_other_format_is_refused(tmp_path):
    from tools.convert.methods import AuxiliaryValue, grouped_absmax, import_encoded
    from tools.convert.model import Model, Parameter
    from tools.convert.recipe import Recipe

    rows = 4
    path, _ = _q8_0_file(tmp_path, rows, gguf_blocks.GDN_VALUE_DIM, 10, SSM_OUT)
    columns = AuxiliaryValue(
        "int32",
        (gguf_blocks.GDN_VALUE_DIM,),
        gguf_blocks.tiled_input_columns().astype("<i4").tobytes(),
    )
    with GGUFFile(path) as gguf:
        source = _ssm_out(gguf, rows)

        def recipe() -> Recipe:
            model = Model({"text": {"config": {}}})
            model.add(Parameter(OUTPUT, source.shape, source, inputs=(GROUPED,)))
            return Recipe(model)

        explicit = recipe()
        explicit.assign(OUTPUT, format="q8_g32_fp16", method=grouped_absmax, source=source)
        explicit.use(OUTPUT, GROUPED, auxiliaries={"input_columns": columns})
        with pytest.raises(ValueError, match=f"{OUTPUT}@{GROUPED}: an input gather"):
            explicit.prepare(device="cpu")
        # Rows split between the imported blocks and a native format share the one Use.
        mixed = recipe()
        mixed.assign(OUTPUT, format="gguf_q8_0", method=import_encoded, source=source)
        mixed.assign(OUTPUT, rows=(0, 2), format="q8_g32_fp16", method=grouped_absmax)
        with pytest.raises(ValueError, match="needs GGUF block rows, not gguf_q8_0, q8_g32_fp16"):
            mixed.prepare(device="cpu")


def test_ssm_out_values_and_imatrix_importance_name_the_same_grouped_column(tmp_path):
    # grouped_search weights logical column g of the values by importance element g: both must
    # come from the same stored column. Stored column c holds codes (c % 128 - 64, c // 128).
    from tools.convert import imatrix

    pytest.importorskip("gguf")
    from tests.convert.test_grouped_search import _write_imatrix

    k = gguf_blocks.GDN_VALUE_DIM
    stored = np.arange(k)
    blocks = np.zeros((2, k // 32, 34), dtype=np.uint8)
    blocks[:, :, :2] = np.float16(1.0).reshape(1).view(np.uint8)
    codes = np.stack([stored % 128 - 64, stored // 128]).astype(np.int8)
    blocks[:, :, 2:] = codes.reshape(2, k // 32, 32).view(np.uint8)
    path = tmp_path / "ssm_out.gguf"
    write_gguf(path, {"general.architecture": "qwen35"},
               [(SSM_OUT, (2, k), TYPE_Q8_0, blocks.tobytes())])
    with GGUFFile(path) as gguf:
        values = _ssm_out(gguf, 2).rows(0, 2).numpy()
    column = (values[1] * 128 + values[0] + 64).astype(np.int64)
    _write_imatrix(tmp_path / "imatrix.gguf", {SSM_OUT: stored.astype(np.float64)}, count=1.0)
    config = {"layer_types": ["linear_attention"], "linear_num_key_heads": 16,
              "linear_num_value_heads": 48, "linear_value_head_dim": 128}
    importance = imatrix.from_gguf(tmp_path / "imatrix.gguf", config)[OUTPUT].numpy()
    assert not np.array_equal(column, stored)
    assert np.array_equal(importance.astype(np.int64), column)


def _header(head_type: int):
    """A dense Qwen3.8-27B GGUF's header and tensor table, stored blocks as Q8_0; no data."""

    from types import SimpleNamespace

    from tools.convert.sources.gguf import TYPE_IDS, TensorInfo

    tensors, offset = {}, 0
    for name, (shape, kind) in gguf_blocks.expected_tensors(False).items():
        kinds = (kind,) if isinstance(kind, str) else kind
        type_id = (head_type if name == "output.weight"
                   else TYPE_Q8_0 if "blocks" in kinds else TYPE_IDS[kinds[0]])
        tensors[name] = TensorInfo(name, shape, type_id, offset)
        offset += tensors[name].nbytes
    kv = gguf_blocks.EXPECTED_HEADER | {"qwen35.block_count": gguf_blocks.LAYERS}
    return SimpleNamespace(path="header.gguf", kv=kv, tensors=tensors, data_bytes_available=offset)


def test_the_output_head_may_be_stored_blocks_or_bf16_words():
    from tools.convert.sources.gguf import TYPE_BF16, TYPE_F16, TYPE_F32

    gguf_blocks.validate(_header(TYPE_Q8_0))
    gguf_blocks.validate(_header(TYPE_BF16))
    for refused in (TYPE_F16, TYPE_F32):
        with pytest.raises(ValueError, match="output.weight is"):
            gguf_blocks.validate(_header(refused))
    table = _header(TYPE_BF16)
    info = table.tensors["token_embd.weight"]
    table.tensors["token_embd.weight"] = type(info)(info.name, info.shape, TYPE_BF16, info.offset)
    with pytest.raises(ValueError, match="token_embd.weight is BF16"):
        gguf_blocks.validate(table)


def test_a_bf16_output_head_keeps_its_exact_words(tmp_path):
    from tools.artifact.reader import Artifact
    from tools.convert.methods import cast_direct
    from tools.convert.model import Model, Parameter
    from tools.convert.pipeline import convert
    from tools.convert.recipe import Recipe
    from tools.convert.sources.gguf import TYPE_BF16

    rows, columns = 6, 64
    words = np.random.default_rng(12).integers(0, 1 << 16, size=(rows, columns), dtype=np.uint16)
    words[0, :3] = (0x7F80, 0xFFC1, 0x8000)  # inf, a NaN payload and -0 pass unchanged
    path = tmp_path / "head.gguf"
    write_gguf(path, {"general.architecture": "qwen35"},
               [("output.weight", (rows, columns), TYPE_BF16, words.tobytes())])
    inputs = ("text/final_hidden", "mtp/final_hidden")
    with GGUFFile(path) as gguf:
        source, stored_blocks = gguf_blocks.head_source(gguf)
        assert not stored_blocks and source.read_encoded is None
        assert np.array_equal(source.rows(2, 4).view(torch.int16).numpy().view(np.uint16),
                              words[2:4])
        model = Model({"text": {"config": {}}})
        model.add(Parameter("text/output_head", (rows, columns), source, inputs=inputs))
        recipe = Recipe(model)
        # As qwen3_8_27b_gguf assigns a direct source: the parameter's direct format.
        recipe.assign("text/output_head", format=model.parameters["text/output_head"].direct_format,
                      method=cast_direct, source=source)
        convert(model, recipe, tmp_path / "head.ninfer", device="cpu", rows_per_chunk=4)
    with Artifact(tmp_path / "head.ninfer") as artifact:
        obj = artifact.object(artifact.directory.bindings["text/output_head"]["object"])
        assert obj.format == "bf16"
        assert artifact.read_object(obj.id) == words.tobytes()
        uses = [u for u in artifact.directory.uses if u["parameter"] == "text/output_head"]
        assert sorted(u["input"] for u in uses) == sorted(inputs)
        assert all(u["activation_policy"] == "A16Only" and "auxiliaries" not in u for u in uses)


def test_a_block_output_head_stays_stored_blocks(tmp_path):
    path, blocks = _q8_0_file(tmp_path, 4, 64, 13, "output.weight")
    with GGUFFile(path) as gguf:
        source, stored_blocks = gguf_blocks.head_source(gguf)
        rows = source.read_encoded(0, 4)
    assert stored_blocks and rows.format == "gguf_q8_0" and rows.input_columns is None
    assert torch.equal(rows.codes, torch.from_numpy(blocks.reshape(4, -1)))
