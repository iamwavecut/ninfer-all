import hashlib

import numpy as np
import pytest

from scripts.pods.attach_dflash2 import attach, plan
from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.sources.gguf import TYPE_F32, TYPE_IDS, GGUFFile, write_gguf

TYPE_Q4_0 = TYPE_IDS["Q4_0"]
H, Q, K, I, RANK, VOCAB, PROJ = 64, 32, 16, 32, 32, 64, 16

RAW = {
    "architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "hidden_act": "silu",
    "attention_bias": False, "is_causal": False, "hidden_size": H, "vocab_size": VOCAB,
    "num_target_layers": 4, "intermediate_size": I, "num_attention_heads": 2,
    "num_key_value_heads": 1, "head_dim": 16, "num_hidden_layers": 1,
    "max_position_embeddings": 128, "rms_norm_eps": 1e-6, "layer_types": ["sliding_attention"],
    "sliding_window": 16, "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
    "dflash_config": {"block_size": 8, "target_layer_ids": [1, 3], "mask_token_id": 63,
                      "conv_kernel_size": 2, "conv_group_size": 16, "selector_rank": RANK,
                      "selector_top_k": 4},
}


def q4_0(rows, columns, seed):
    return np.random.default_rng(seed).integers(0, 256, rows * columns // 32 * 18,
                                                dtype=np.uint8).tobytes()


def f32(shape, value=1.5):
    return np.full(shape, value, dtype="<f4").tobytes()


def tensors(norm=1.5, extra=False):
    out = [("fc.weight", (H, 2 * H), TYPE_Q4_0, q4_0(H, 2 * H, 1)),
           ("enc.output_norm.weight", (H,), TYPE_F32, f32((H,), norm)),
           ("output_norm.weight", (H,), TYPE_F32, f32((H,))),
           ("selector_hidden.weight", (RANK, H), TYPE_Q4_0, q4_0(RANK, H, 2)),
           ("selector_predecessor.weight", (VOCAB, RANK), TYPE_Q4_0, q4_0(VOCAB, RANK, 3)),
           ("selector_successor.weight", (VOCAB, RANK), TYPE_Q4_0, q4_0(VOCAB, RANK, 4))]
    b = "blk.0."
    out += [(b + "attn_norm.weight", (H,), TYPE_F32, f32((H,))),
            (b + "ffn_norm.weight", (H,), TYPE_F32, f32((H,))),
            (b + "attn_q.weight", (Q, H), TYPE_Q4_0, q4_0(Q, H, 5)),
            (b + "attn_k.weight", (K, H), TYPE_Q4_0, q4_0(K, H, 6)),
            (b + "attn_v.weight", (K, H), TYPE_Q4_0, q4_0(K, H, 7)),
            (b + "attn_output.weight", (H, Q), TYPE_Q4_0, q4_0(H, Q, 8)),
            (b + "attn_q_norm.weight", (16,), TYPE_F32, f32((16,))),
            (b + "attn_k_norm.weight", (16,), TYPE_F32, f32((16,))),
            (b + "ffn_gate.weight", (I, H), TYPE_Q4_0, q4_0(I, H, 9)),
            (b + "ffn_up.weight", (I, H), TYPE_Q4_0, q4_0(I, H, 10)),
            (b + "ffn_down.weight", (H, I), TYPE_Q4_0, q4_0(H, I, 11)),
            (b + "attn_conv_base", (2, 2, H), TYPE_F32, f32((2, 2, H), -0.25)),
            (b + "attn_conv_proj.weight", (PROJ, H), TYPE_Q4_0, q4_0(PROJ, H, 12)),
            (b + "ffn_conv_base", (2, 2, H), TYPE_F32, f32((2, 2, H), 0.75)),
            (b + "ffn_conv_proj.weight", (PROJ, H), TYPE_Q4_0, q4_0(PROJ, H, 13))]
    if extra:
        out.append(("blk.1.attn_norm.weight", (H,), TYPE_F32, f32((H,))))
    return out


def inputs(root, **kwargs):
    gguf_path, base_path = root / "drafter.gguf", root / "base.ninfer"
    write_gguf(gguf_path, {"general.architecture": "dflash2"}, tensors(**kwargs))
    config = {"architectures": ["Qwen4ExpForCausalLM"], "hidden_size": H, "vocab_size": VOCAB,
              "num_hidden_layers": 4}
    with ArtifactWriter(base_path, [TensorSpec("head", (VOCAB, H), "bf16", "contiguous_le_v1")],
                        components={"text": {"config": config}},
                        bindings={"text/output_head": {"object": "head"}},
                        uses=[{"parameter": "text/output_head", "input": "text/final_hidden",
                               "activation_policy": "AllowA8"}]) as w:
        w.write_object("head", bytes(range(256)) * (VOCAB * H * 2 // 256))
    return gguf_path, base_path


def test_attachment_keeps_blocks_and_target_bytes(tmp_path):
    gguf_path, base_path = inputs(tmp_path)
    out = tmp_path / "result.ninfer"
    with Artifact(base_path) as base, GGUFFile(gguf_path) as gguf:
        report = attach(base, gguf, RAW, out)
        assert report["file_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
        with Artifact(out) as result:
            d = result.directory
            assert result.read_object("head") == base.read_object("head")
            assert d.components["text"] == base.directory.components["text"]
            assert d.components["dflash2"]["target"] == "text"
            assert d.components["dflash2"]["config"]["dflash_config"]["target_layer_ids"] == [1, 3]
            q = d.bindings["dflash2/layers/0/attention/query"]["object"]
            assert result.read_object(q) == bytes(gguf.tensor_bytes("blk.0.attn_q.weight"))
            assert result.by_id[q].format == "gguf_q4_0"
            assert result.by_id[q].layout == "gguf_blocks_v1"
            for role in ("key", "value"):
                assert (d.bindings[f"dflash2/layers/0/attention/context_{role}"] ==
                        d.bindings[f"dflash2/layers/0/attention/{role}"])
            norm = d.bindings["dflash2/context_norm"]["object"]
            assert result.by_id[norm].format == "bf16"
            assert result.read_object(norm) == np.full(H, 0x3FC0, "<u2").tobytes()
            base_kernel = d.bindings["dflash2/layers/0/attention_conv/base_kernel"]["object"]
            assert result.by_id[base_kernel].shape == (2, 2, H)
            assert result.read_object(base_kernel) == np.full(4 * H, 0xBE80, "<u2").tobytes()
            uses = {(u["parameter"], u["input"]): u.get("activation_policy") for u in d.uses}
            assert uses[("text/output_head", "dflash2/final_hidden")] == "AllowA8"
            assert uses[("dflash2/layers/0/attention/context_key", "dflash2/context_input")] == "AllowA8"
            assert uses[("dflash2/layers/0/mlp/down", "dflash2/layers/0/mlp_product")] == "AllowA8"
            assert not any(p.endswith("codebook") for p, _ in uses)
            assert report["dflash2_bindings"] == 23
            assert report["dflash2_objects_verified"] == 21
            with pytest.raises(ValueError, match="already contains"):
                plan(d, gguf, RAW)


@pytest.mark.parametrize("kwargs, message", [({"norm": 1.0000001}, "not BF16"),
                                             ({"extra": True}, "unexpected GGUF tensors")])
def test_refused_before_output(tmp_path, kwargs, message):
    gguf_path, base_path = inputs(tmp_path, **kwargs)
    out = tmp_path / "refused.ninfer"
    with Artifact(base_path) as base, GGUFFile(gguf_path) as gguf:
        with pytest.raises(ValueError, match=message):
            attach(base, gguf, RAW, out)
    assert not out.exists()


def test_geometry_mismatch_refused(tmp_path):
    gguf_path, base_path = inputs(tmp_path)
    raw = {**RAW, "intermediate_size": 64}
    with Artifact(base_path) as base, GGUFFile(gguf_path) as gguf:
        with pytest.raises(ValueError, match="shape"):
            attach(base, gguf, raw, tmp_path / "refused.ninfer")
