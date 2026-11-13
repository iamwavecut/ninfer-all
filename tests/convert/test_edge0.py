from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.convert import edge0, gguf_blocks, qwen4_exp, qwen4_exp_gguf, qwen4_exp_ngram
from tools.convert.sources.gguf import GGUFFile, write_gguf

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "qwen4_exp"
TYPE_I8, TYPE_IQ4_NL, TYPE_NVFP4 = 24, 20, 40
LAYERS, EXPERTS, WIDTH, MIDDLE, HIDDEN = 2, 4, 64, 64, 64


def _checkpoint():
    return json.loads((_FIXTURES / "config.json").read_text())


def _ue4m3_half(byte: int) -> float:
    # ggml_ue4m3_to_fp32 as ggml-impl.h defines it.
    if byte in (0, 0x7F):
        return 0.0
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    raw = mantissa * 2.0 ** -9 if exponent == 0 else (1 + mantissa / 8) * 2.0 ** (exponent - 7)
    return raw * 0.5


def _nvfp4_blocks(generator, rows: int, columns: int) -> np.ndarray:
    blocks = generator.integers(0, 256, size=(rows, columns // 64, 36), dtype=np.uint8)
    # Scale bytes over the whole range of the encoder's output, zero and the NaN byte included.
    blocks[..., :4] = generator.integers(0, 0x7F, size=(rows, columns // 64, 4), dtype=np.uint8)
    blocks[0, 0, :4] = (0, 0x7F, 0x01, 0x7E)
    return blocks.reshape(rows, -1)


def test_nvfp4_rows_decode_as_ggml_defines_them():
    generator = np.random.default_rng(3)
    blocks = _nvfp4_blocks(generator, 3, 128)
    values = gguf_blocks.dequantize_nvfp4(blocks)
    kvalues = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)
    b = blocks.reshape(3, 2, 36)
    for row in range(3):
        for block in range(2):
            for j in range(64):
                s, k = divmod(j, 16)
                byte = int(b[row, block, 4 + 8 * s + k % 8])
                code = byte & 15 if k < 8 else byte >> 4
                expected = np.float32(kvalues[code] * _ue4m3_half(int(b[row, block, s])))
                assert values[row, block * 64 + j] == expected
    gguf = pytest.importorskip("gguf")
    from gguf.quants import dequantize

    if not hasattr(gguf.GGMLQuantizationType, "NVFP4"):
        pytest.skip("this gguf-py predates NVFP4")
    reference = dequantize(blocks, gguf.GGMLQuantizationType.NVFP4).reshape(3, -1)
    assert np.array_equal(values, reference)


def test_q4_0_rows_decode_as_ggml_defines_them():
    generator = np.random.default_rng(4)
    blocks = generator.integers(0, 256, size=(2, 3, 18), dtype=np.uint8)
    blocks[..., :2] = generator.uniform(0.01, 0.02, size=(2, 3)).astype(np.float16).view(
        np.uint8).reshape(2, 3, 2)
    values = gguf_blocks.dequantize_q4_0(blocks.reshape(2, -1))
    for row in range(2):
        for block in range(3):
            d = np.float32(blocks[row, block, :2].copy().view(np.float16)[0])
            for j in range(32):
                byte = int(blocks[row, block, 2 + j % 16])
                code = byte & 15 if j < 16 else byte >> 4
                assert values[row, block * 32 + j] == d * np.float32(code - 8)


def _write_pack(root: Path, generator, *, nvfp4: bool) -> dict:
    """A two-layer release of four experts, latent width 64 and hidden width 64."""

    root.mkdir()
    gate_up_type = down_type = TYPE_NVFP4 if nvfp4 else TYPE_IQ4_NL
    row = 36 if nvfp4 else 2 * 18
    tail = 16 if nvfp4 else 0
    record = 2 * MIDDLE * row + WIDTH * row + tail
    lines = [f"# strata native experts v6: latent A-only (expert_embd {WIDTH}, expert_ff {MIDDLE})"]
    lines += [f"{layer} {gate_up_type} {down_type} {layer * EXPERTS * record} {record}"
              for layer in range(LAYERS)]
    (root / "native_experts.txt").write_text("\n".join(lines) + "\n")
    experts = generator.integers(0, 256, size=(LAYERS, EXPERTS, record), dtype=np.uint8)
    scales = generator.uniform(0.5, 2.0, size=(LAYERS, EXPERTS, 3)).astype("<f4")
    if nvfp4:
        tails = np.zeros((LAYERS, EXPERTS, 4), dtype="<f4")
        tails[..., :3] = scales
        experts[..., -16:] = tails.view(np.uint8).reshape(LAYERS, EXPERTS, 16)
        # Scale bytes are unsigned E4M3: bit 7 clear.
        blocks = experts[..., :-16].reshape(LAYERS, EXPERTS, -1, 36)
        blocks[..., :4] &= 0x7F
        experts[..., :-16] = blocks.reshape(LAYERS, EXPERTS, -1)
    (root / "experts.bin").write_bytes(experts.tobytes())
    labels = generator.integers(0, 8, size=(LAYERS, EXPERTS), dtype="<i4")
    (root / "labels.bin").write_bytes(labels.tobytes())
    dense = {}
    for build, block in (("q8", 34), ("q4", 18)):
        d = generator.integers(0, 256, size=(LAYERS, 8 * WIDTH, HIDDEN // 32 * block),
                               dtype=np.uint8)
        u = generator.integers(0, 256, size=(LAYERS, 8 * HIDDEN, WIDTH // 32 * block),
                               dtype=np.uint8)
        name = f"latent_dense_{build}.bin"
        (root / name).write_bytes(b"".join(d[layer].tobytes() + u[layer].tobytes()
                                          for layer in range(LAYERS)))
        dense[build] = (d, u)
    column = None
    if not nvfp4:
        (root / "scales").mkdir()
        column = generator.uniform(0.5, 2.0, size=(LAYERS, WIDTH)).astype("<f4")
        column[0, :2] = (0.0, -1.0)
        for layer in range(LAYERS):
            (root / "scales" / f"l{layer}.f32").write_bytes(column[layer].tobytes())
    (root / "model.json").write_text(json.dumps({
        "experts": EXPERTS, "latent_dim": WIDTH, "layers": LAYERS,
        "env": {"STRATA_LATENT_DENSE_FMT": "8" if nvfp4 else "4"}}))
    return {"experts": experts, "scales": scales, "labels": labels, "dense": dense,
            "column": column, "row": row}


def test_a_release_folder_reads_back_every_stored_row(tmp_path):
    generator = np.random.default_rng(7)
    written = _write_pack(tmp_path / "pack", generator, nvfp4=True)
    with edge0.Edge0Pack(tmp_path / "pack", hidden=HIDDEN) as pack:
        assert pack.latent_config() == {"width": WIDTH, "groups": 8, "expert_scales": True,
                                        "column_scale": False}
        assert pack.released_dense == "q8"
        assert (pack.gate_up_type, pack.down_type) == (TYPE_NVFP4, TYPE_NVFP4)
        row = written["row"]
        record = written["experts"][1, 2]
        assert np.array_equal(pack.expert_rows(1, 2, "gate", 0, MIDDLE),
                              record[:MIDDLE * row].reshape(MIDDLE, row))
        assert np.array_equal(pack.expert_rows(1, 2, "up", 3, 5),
                              record[(MIDDLE + 3) * row:(MIDDLE + 5) * row].reshape(2, row))
        assert np.array_equal(pack.expert_rows(1, 2, "down", 0, WIDTH),
                              record[2 * MIDDLE * row:(2 * MIDDLE + WIDTH) * row].reshape(WIDTH, row))
        assert np.array_equal(pack.expert_scales(1), written["scales"][1])
        assert np.array_equal(pack.labels(0), written["labels"][0])
        for build in ("q8", "q4"):
            d, u = written["dense"][build]
            assert np.array_equal(pack.encode_rows(build, 1, 5, 9), d[1, 5:9])
            assert np.array_equal(pack.decode_rows(build, 0, 0, 8 * HIDDEN), u[0])


def test_column_scales_are_the_reference_reciprocals(tmp_path):
    generator = np.random.default_rng(8)
    written = _write_pack(tmp_path / "pack", generator, nvfp4=False)
    with edge0.Edge0Pack(tmp_path / "pack", hidden=HIDDEN) as pack:
        assert pack.latent_config()["column_scale"] and not pack.latent_config()["expert_scales"]
        assert pack.released_dense == "q4"
        stored = written["column"][0]
        multipliers = pack.column_scale(0)
        # The reference multiplies by 1 / v for v > 0 and by one otherwise, in FP32.
        assert multipliers[0] == 1 and multipliers[1] == 1
        assert np.array_equal(multipliers[2:], np.float32(1) / stored[2:])
        with pytest.raises(ValueError, match="no output multipliers"):
            pack.expert_scales(0)


@pytest.mark.parametrize("damage, message", [
    (lambda root: (root / "experts.bin").write_bytes(b"\0" * 10), "expected"),
    (lambda root: (root / "labels.bin").write_bytes(np.full(LAYERS * EXPERTS, 8, "<i4").tobytes()),
     "group labels"),
    (lambda root: (root / "scales" / "l1.f32").unlink(), "some layers only"),
    (lambda root: (root / "native_experts.txt").write_text("# strata native experts v5\n"),
     "v6 latent"),
])
def test_a_damaged_release_folder_is_refused(tmp_path, damage, message):
    _write_pack(tmp_path / "pack", np.random.default_rng(9), nvfp4=False)
    damage(tmp_path / "pack")
    with pytest.raises(ValueError, match=message):
        edge0.Edge0Pack(tmp_path / "pack", hidden=HIDDEN)


def test_a_tail_must_hold_three_positive_multipliers(tmp_path):
    written = _write_pack(tmp_path / "pack", np.random.default_rng(10), nvfp4=True)
    experts = written["experts"].copy()
    experts[0, 1, -4:] = np.array([1.0], dtype="<f4").view(np.uint8)
    (tmp_path / "pack" / "experts.bin").write_bytes(experts.tobytes())
    with edge0.Edge0Pack(tmp_path / "pack", hidden=HIDDEN) as pack:
        with pytest.raises(ValueError, match="positive finite multipliers and a zero"):
            pack.expert_scales(0)
        assert np.array_equal(pack.expert_scales(1), written["scales"][1])


def test_a_header_may_describe_the_pack_formats(tmp_path):
    _write_pack(tmp_path / "pack", np.random.default_rng(14), nvfp4=False)
    index = tmp_path / "pack" / "native_experts.txt"
    lines = index.read_text().splitlines()
    lines[0] = (f"# strata native experts v6: latent A-only, Q3_K gu + shared colscale "
                f"(expert_embd {WIDTH}, expert_ff {MIDDLE})")
    index.write_text("\n".join(lines) + "\n")
    with edge0.Edge0Pack(tmp_path / "pack", hidden=HIDDEN) as pack:
        assert (pack.width, pack.middle) == (WIDTH, MIDDLE)


def _write_tokenizers(root: Path, base: Path) -> None:
    # The release's GGUF export (vocabulary padded past the added tokens) and the HF tokenizer.
    (root / "tokenizer").mkdir()
    vocab = {"a": 0, "b": 1, "ab": 2, "#": 5, "##": 6, "<|im_end|>": 3, "[PAD4]": 4}
    (root / "tokenizer" / "vocab.json").write_text(json.dumps(vocab))
    (root / "tokenizer" / "merges.txt").write_text("#version: 0.2\na b\n# #\n")
    (root / "tokenizer" / "tokenizer.json").write_text(json.dumps({"model": "gpt2"}))
    base.write_text(json.dumps({
        "model": {"vocab": {"a": 0, "b": 1, "ab": 2, "#": 5, "##": 6},
                  "merges": [["a", "b"], ["#", "#"]]},
        "added_tokens": [{"id": 3, "content": "<|im_end|>"}]}))


def test_the_release_tokenizer_must_be_the_base_models(tmp_path):
    _write_pack(tmp_path / "pack", np.random.default_rng(15), nvfp4=False)
    base = tmp_path / "tokenizer.json"
    _write_tokenizers(tmp_path / "pack", base)
    with edge0.Edge0Pack(tmp_path / "pack", hidden=HIDDEN) as pack:
        pack.check_tokenizer(base)
        (tmp_path / "pack" / "tokenizer" / "merges.txt").write_text("#version: 0.2\na b\n")
        with pytest.raises(ValueError, match="merges equal: False"):
            pack.check_tokenizer(base)
        (tmp_path / "pack" / "tokenizer" / "merges.txt").write_text("a b\n# #\n")
        pack.check_tokenizer(base)
        (tmp_path / "pack" / "tokenizer" / "vocab.json").write_text(json.dumps({"a": 0, "b": 2}))
        with pytest.raises(ValueError, match="5 tokens differ"):
            pack.check_tokenizer(base)


def test_an_nvfp4_scale_byte_with_bit_7_is_refused(tmp_path):
    written = _write_pack(tmp_path / "pack", np.random.default_rng(12), nvfp4=True)
    experts = written["experts"].copy()
    row = written["row"]
    # Expert 2 of layer 1: the second scale byte of the down projection's last row.
    experts[1, 2, (2 * MIDDLE + WIDTH - 1) * row + 1] |= 0x80
    (tmp_path / "pack" / "experts.bin").write_bytes(experts.tobytes())
    with edge0.Edge0Pack(tmp_path / "pack", hidden=HIDDEN) as pack:
        with pytest.raises(ValueError, match="bit 7"):
            pack.expert_rows(1, 2, "down", WIDTH - 2, WIDTH)
        assert np.array_equal(pack.expert_rows(1, 2, "down", 0, WIDTH - 1),
                              experts[1, 2, 2 * MIDDLE * row:(2 * MIDDLE + WIDTH - 1) * row]
                              .reshape(WIDTH - 1, row))
        pack.expert_rows(1, 2, "gate", 0, MIDDLE)


def test_latent_sources_name_every_parameter_of_a_layer(tmp_path):
    written = _write_pack(tmp_path / "pack", np.random.default_rng(11), nvfp4=True)
    with edge0.Edge0Pack(tmp_path / "pack", hidden=HIDDEN) as pack:
        found = edge0.latent_sources(pack, "q4", 1, "m/")
        expected = {"m/latent/encode", "m/latent/decode", "m/latent/labels",
                    "m/latent/expert_scales"}
        expected |= {f"m/experts/{e}/{role}" for e in range(EXPERTS)
                     for role in ("gate", "up", "down")}
        assert set(found) == expected
        encode, encoded = found["m/latent/encode"]
        assert encoded and encode.shape == (8 * WIDTH, HIDDEN)
        rows = encode.read_encoded(3, 6)
        assert rows.format == "gguf_q4_0"
        assert torch.equal(rows.codes, torch.from_numpy(written["dense"]["q4"][0][1, 3:6]))
        assert torch.equal(encode.rows(3, 6), torch.from_numpy(
            gguf_blocks.dequantize_q4_0(written["dense"]["q4"][0][1, 3:6])))
        down, encoded = found["m/experts/3/down"]
        assert encoded and down.shape == (WIDTH, MIDDLE)
        assert down.read_encoded(0, 1).format == "gguf_nvfp4"
        labels, encoded = found["m/latent/labels"]
        assert not encoded and labels.values().dtype == torch.int32
        assert torch.equal(labels.values(), torch.from_numpy(written["labels"][1]))
        scales, _ = found["m/latent/expert_scales"]
        assert scales.shape == (EXPERTS, 3) and scales.values().dtype == torch.float32


def _q2_0_rows(generator, rows: int, columns: int) -> np.ndarray:
    blocks = generator.integers(0, 256, size=(rows, columns // 64, 18), dtype=np.uint8)
    blocks[..., :2] = generator.uniform(0.01, 0.02, size=(rows, columns // 64)).astype(
        np.float16).view(np.uint8).reshape(rows, columns // 64, 2)
    return blocks


def test_mtp_experts_return_from_their_planes_to_q2_0_rows(tmp_path, capsys):
    # Strata's mtp_rt.py blob: interleaved gate/up code rows, down code rows, then their scales.
    generator = np.random.default_rng(12)
    hidden, middle, experts = 128, 64, 2
    root = tmp_path / "mtp"
    root.mkdir()
    rows = {}
    blobs = []
    for expert in range(experts):
        gate = _q2_0_rows(generator, middle, hidden)
        up = _q2_0_rows(generator, middle, hidden)
        down = _q2_0_rows(generator, hidden, middle)
        rows[expert] = {"gate": gate, "up": up, "down": down}
        inter = np.empty((2 * middle, hidden // 64, 18), dtype=np.uint8)
        inter[0::2], inter[1::2] = gate, up
        blobs.append(inter[..., 2:].tobytes() + down[..., 2:].tobytes()
                     + inter[..., :2].tobytes() + down[..., :2].tobytes())
    (root / "experts.bin").write_bytes(b"".join(blobs))
    q8 = generator.integers(0, 256, size=(4, 2 * 34), dtype=np.uint8)
    weights = np.array([0.5, -0.25, 0, 2 ** -20, 2 ** -10 + 2 ** -20], dtype="<f4")
    norm = (np.float32(1) + weights).astype("<f4")
    words = generator.integers(0, 1 << 16, size=(2, 3), dtype="<u2")
    dense = q8.tobytes() + norm.tobytes() + words.tobytes()
    (root / "dense.bin").write_bytes(dense)
    (root / "dense.txt").write_text(
        f"fc.weight q8_0 4 64 0 {q8.size}\n"
        f"norm.weight f32 1 5 {q8.size} 20\n"
        f"mix.weight bf16 2 3 {q8.size + 20} 12\n")
    mtp = edge0.MtpPack(root, hidden=hidden, middle=middle, experts=experts)
    try:
        for expert in range(experts):
            for role in ("gate", "up", "down"):
                assert np.array_equal(mtp.expert_rows(expert, role),
                                      rows[expert][role].reshape(len(rows[expert][role]), -1))
        assert np.array_equal(mtp.q8_rows("fc.weight", (4, 64)), q8)
        assert np.array_equal(mtp.bf16_words("mix.weight", (2, 3)), words)
        with pytest.raises(ValueError, match="expected q8_0"):
            mtp.q8_rows("norm.weight", (1, 32))
        source = edge0._norm(mtp.f32("norm.weight", 5), "norm.weight")
        values = source.values()
        assert values.dtype == torch.bfloat16
        # Every BF16 weight comes back exactly; 2^-10 + 2^-20 is no BF16 value, which is reported.
        assert values.float().tolist() == [0.5, -0.25, 0.0, 2 ** -20, 2 ** -10]
        assert "1 of 5 stored values are not 1 + a BF16 weight" in capsys.readouterr().out
    finally:
        mtp.close()


def test_with_latent_moe_sets_the_pack_on_the_model_config(tmp_path):
    _write_pack(tmp_path / "pack", np.random.default_rng(13), nvfp4=False)
    checkpoint = _checkpoint()
    with edge0.Edge0Pack(tmp_path / "pack", hidden=HIDDEN) as pack:
        with pytest.raises(ValueError, match="experts of width"):
            edge0.with_latent_moe(checkpoint, pack)
        tiny = deepcopy(checkpoint)
        tiny["text_config"].update(num_experts=EXPERTS, moe_intermediate_size=MIDDLE,
                                   num_hidden_layers=LAYERS, hidden_size=HIDDEN)
        config = edge0.with_latent_moe(tiny, pack)
    assert config["text_config"]["latent_moe"] == {
        "width": WIDTH, "groups": 8, "expert_scales": False, "column_scale": True}
    assert "latent_moe" not in tiny["text_config"]


def _tiny_table_config():
    # Four hash heads of 101, 103, 107 and 109 rows (padded to 424), 160 values per row.
    config = deepcopy(_checkpoint())
    config["text_config"].update(ngram_vocab_size_base=100, make_ngram_vocab_size_divisible_by=8,
                                 heads_per_ngram=2, ple_embed_dim=640)
    return config


def test_strata_fp8_table_becomes_the_table_artifact(tmp_path):
    from tools.artifact.codecs.fp8_row import encode_fp8_row_interleaved
    from tools.artifact.reader import Artifact
    from tools.convert.pipeline import convert
    from tools.convert.recipe import Recipe

    class Base:
        config = _tiny_table_config()
        root = tmp_path

    generator = np.random.default_rng(14)
    rows = 424
    codes = generator.integers(0, 256, size=(rows, 160), dtype=np.uint8)
    codes[(codes & 0x7F) == 0x7F] = 1
    codes[5, 7], codes[9, 0] = 0x7F, 0xFF
    scale = 209 * 2.0 ** -20
    shard = tmp_path / "ple-fp8.gguf"
    write_gguf(shard, {"general.architecture": "strata-ple", "strata.ple.format": "f8_e4m3",
                       "strata.ple.scale": scale},
               [(qwen4_exp_gguf.NGRAM_TENSOR, (rows, 160), TYPE_I8, codes.tobytes())])
    stored = codes.copy()
    stored[5, 7] = stored[9, 0] = 0
    expected = encode_fp8_row_interleaved(
        torch.from_numpy(stored), torch.full((rows,), scale, dtype=torch.float16), (rows, 160))
    with GGUFFile(shard) as table:
        assert qwen4_exp_ngram.is_strata_fp8_table(table)
        model = qwen4_exp.build_model(Base(), components=("ngram",))
        source, counter = qwen4_exp_ngram.strata_fp8_source(
            table, model.components["ngram"]["config"])
        values = source.rows(4, 6)
        assert values[1, 7] == 0 and values[0, 3] == torch.tensor(
            stored[4, 3]).view(torch.float8_e4m3fn).float() * scale
        assert counter["nan_codes"] == 1
        recipe = Recipe(model)
        edge0.edge0_pack(model, recipe, {"ngram": table})
        output = tmp_path / "table.ninfer"
        convert(model, recipe, output, device="cpu")
    with Artifact(output) as artifact:
        descriptor = artifact.directory.components["ngram"]["config"]
        assert descriptor["format"] == "fp8_e4m3fn_row_fp16"
        assert descriptor["table_sha256"] == hashlib.sha256(expected).hexdigest()
        stored_object = artifact.directory.bindings["ngram/table"]["object"]
        assert artifact.read_object(stored_object) == expected


def test_a_table_scale_must_be_a_binary16_value(tmp_path):
    shard = tmp_path / "ple-fp8.gguf"
    write_gguf(shard, {"strata.ple.format": "f8_e4m3", "strata.ple.scale": 0.1},
               [(qwen4_exp_gguf.NGRAM_TENSOR, (424, 160), TYPE_I8, bytes(424 * 160))])
    with GGUFFile(shard) as table, pytest.raises(ValueError, match="binary16"):
        qwen4_exp_ngram.strata_fp8_source(table, {"rows": 424, "row_width": 160})
