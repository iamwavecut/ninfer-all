from __future__ import annotations

import hashlib
import json
import struct

import numpy as np
import pytest

from tools.convert import deepseek_v41
from tools.convert.sources.gguf import TYPE_BF16, TYPE_F32, GGUFFile, write_gguf

TYPE_Q8_0, TYPE_Q5_K, TYPE_IQ4_NL, TYPE_I32, TYPE_I64, TYPE_MXFP4 = 8, 13, 20, 26, 27, 39

HIDDEN, VOCAB, HEADS, HEAD_DIM, ROPE = 256, 512, 2, 64, 16
Q_RANK, O_RANK, GROUPS = 64, 32, 2
INDEX_HEADS, INDEX_DIM = 2, 32
EXPERTS, WIDTH, HC = 4, 64, 4
RATIOS = [0, 2, 1]
ENGRAM_LAYERS, NGRAM, ENGRAM_HEADS, ENGRAM_DIM = [0], 3, 2, 32
PRIMES = [5, 7, 11, 13]
ENGRAM_ROWS = sum(PRIMES)
COMPRESSED_VOCAB = 100
# The drafter: one block over the streams entering blocks 1 and 2, two experts, one a token.
DRAFT_TARGETS, DRAFT_EXPERTS, DRAFT_RANK, NOISE = [1, 2], 2, 16, 5
# The vision tower: two blocks of width 32 (two heads of 16), patches of 2 pixels merged 3 x 3.
V_LAYERS, V_HIDDEN, V_HEADS, V_WIDTH, V_PATCH, V_MERGE, IMAGE_TOKEN = 2, 32, 2, 48, 2, 3, 500


def _hf_config(**overrides):
    text = dict(
        model_type="deepseek_v41_text", vocab_size=VOCAB, hidden_size=HIDDEN,
        moe_intermediate_size=WIDTH, num_hidden_layers=3, num_attention_heads=HEADS,
        num_key_value_heads=1, head_dim=HEAD_DIM, qk_rope_head_dim=ROPE, q_lora_rank=Q_RANK,
        o_lora_rank=O_RANK, o_groups=GROUPS, hidden_act="silu", swiglu_limit=10.0,
        rms_norm_eps=1e-20, attention_bias=False, tie_word_embeddings=False,
        max_position_embeddings=4096, rope_theta=10000,
        rope_scaling={"rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1,
                      "original_max_position_embeddings": 1024},
        n_routed_experts=EXPERTS, n_shared_experts=1, num_experts_per_tok=2,
        scoring_func="sqrtsoftplus", topk_method="noaux_tc", norm_topk_prob=True,
        routed_scaling_factor=1.5, sliding_window=8, compress_ratios=RATIOS + [0],
        kv_source_layer_ids=[1, 2], index_source_layer_ids=[1, 2], candidate_source_layer_id=2,
        candidate_topk_blocks=4, candidate_block_size=8, index_n_heads=INDEX_HEADS,
        index_head_dim=INDEX_DIM, index_topk=16, compress_rope_theta=160000, hc_mult=HC,
        hc_sinkhorn_iters=20, hc_eps=1e-6, engram_layer_ids=ENGRAM_LAYERS,
        engram_num_embeddings=[ENGRAM_ROWS], engram_max_ngram_size=NGRAM,
        engram_n_heads=ENGRAM_HEADS, engram_head_dim=ENGRAM_DIM, engram_pad_token_id=2,
        engram_compressed_vocab_size=COMPRESSED_VOCAB, num_nextn_predict_layers=1,
        dspark_block_size=3, dspark_noise_token_id=NOISE, dspark_target_layer_ids=DRAFT_TARGETS,
        dspark_markov_rank=DRAFT_RANK, dspark_n_routed_experts=DRAFT_EXPERTS,
        dspark_num_experts_per_tok=1)
    text.update(overrides)
    vision = dict(model_type="deepseek_v41_vision", num_hidden_layers=V_LAYERS,
                  hidden_size=V_HIDDEN, num_attention_heads=V_HEADS, intermediate_size=V_WIDTH,
                  patch_size=V_PATCH, rope_theta=10000, downsample_ratio=V_MERGE,
                  max_image_tokens=1024, min_pixels=64, max_wh_ratio=None)
    return {"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41",
            "bos_token_id": 0, "eos_token_id": 1, "image_token_id": IMAGE_TOKEN,
            "text_config": text, "vision_config": vision}


def _gguf_kv(**overrides):
    f = "deepseek41."
    kv = {
        "general.architecture": "deepseek41", "split.count": 2, f + "block_count": 3,
        f + "embedding_length": HIDDEN, f + "attention.head_count": HEADS,
        f + "attention.head_count_kv": 1, f + "attention.key_length": HEAD_DIM,
        f + "attention.value_length": HEAD_DIM, f + "rope.dimension_count": ROPE,
        f + "attention.q_lora_rank": Q_RANK, f + "attention.output_group_count": GROUPS,
        f + "attention.output_lora_rank": O_RANK, f + "attention.sliding_window": 8,
        f + "expert_count": EXPERTS, f + "expert_used_count": 2,
        f + "expert_feed_forward_length": WIDTH, f + "expert_shared_count": 1,
        f + "expert_gating_func": 4, f + "expert_weights_norm": True,
        f + "expert_weights_scale": 1.5, f + "expert_scoring_func": "sqrtsoftplus",
        f + "attention.indexer.head_count": INDEX_HEADS,
        f + "attention.indexer.key_length": INDEX_DIM, f + "attention.indexer.top_k": 16,
        f + "attention.candidate_block_size": 8, f + "attention.candidate_topk_blocks": 4,
        f + "attention.candidate_source_layer_id": 2, f + "hyper_connection.count": HC,
        f + "hyper_connection.sinkhorn_iterations": 20, f + "hyper_connection.epsilon": 1e-6,
        f + "engram_head_dim": ENGRAM_DIM, f + "engram_n_heads": ENGRAM_HEADS,
        f + "engram_max_ngram_size": NGRAM, f + "engram_pad_token_id": 2,
        f + "rope.scaling.original_context_length": 1024, f + "rope.freq_base": 10000.0,
        f + "attention.compress_rope_freq_base": 160000.0, f + "rope.scaling.factor": 16.0,
        f + "rope.scaling.yarn_beta_fast": 32.0, f + "rope.scaling.yarn_beta_slow": 1.0,
        f + "attention.layer_norm_rms_epsilon": 1e-20,
        f + "attention.compress_ratios": RATIOS + [0], f + "kv_source_layer_ids": [1, 2],
        f + "index_source_layer_ids": [1, 2], f + "engram.layer_ids": ENGRAM_LAYERS,
        f + "engram.num_embeddings": [ENGRAM_ROWS], f + "swiglu_clamp_exp": [10.0] * 3,
        f + "swiglu_clamp_shexp": [10.0] * 3,
    }
    kv.update(overrides)
    return kv


def _blocks(generator, rows, columns, block_values, block_bytes, scale_bytes=2):
    count = columns // block_values
    blocks = generator.integers(0, 256, size=(rows, count, block_bytes), dtype=np.uint8)
    scales = generator.uniform(0.01, 0.02, size=(rows, count, scale_bytes // 2))
    blocks[:, :, :scale_bytes] = scales.astype(np.float16).view(np.uint8).reshape(
        rows, count, scale_bytes)
    return blocks.reshape(rows, -1)


def _release(tmp_path, kv=None, offsets=None):
    """A three-block DeepSeek-V4.1 release (window-only, ratio 2, ratio 1; Engram in block 0): the
    HF text files, a GGUF split in two shards and the Engram GGUF."""

    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text(json.dumps(_hf_config()))
    vocab = {f"t{i}": i for i in range(3, VOCAB)}
    added = [{"id": i, "content": c, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": True}
             for i, c in enumerate(["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>",
                                    "<｜▁pad▁｜>"])]
    vocab.update({token["content"]: token["id"] for token in added})
    (hf / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": vocab, "merges": []}, "added_tokens": added}))
    # As the release's: no added_tokens_decoder.
    (hf / "tokenizer_config.json").write_text(json.dumps({"add_bos_token": False}))

    generator = np.random.default_rng(11)
    shards, stored = [[], []], {}

    def put(name, shape, kind, data, shard):
        shards[shard].append((name, shape, kind, data.tobytes()))
        stored[name] = data

    def q8(name, rows, columns, shard=0):
        put(name, (rows, columns), TYPE_Q8_0, _blocks(generator, rows, columns, 32, 34), shard)

    def vector(name, shape, kind=TYPE_F32, shard=0):
        values = generator.normal(size=shape).astype(np.float32)
        if kind == TYPE_BF16:
            words = (values.view(np.uint32) >> 16).astype(np.uint16)
            shards[shard].append((name, shape, kind, words.tobytes()))
            stored[name] = (words.astype(np.uint32) << 16).view(np.float32)
        else:
            put(name, shape, kind, values, shard)

    def integers(name, values, kind):
        put(name, values.shape, kind, values, 0)

    put("token_embd.weight", (VOCAB, HIDDEN), TYPE_Q5_K,
        _blocks(generator, VOCAB, HIDDEN, 256, 176, 4), 0)
    q8("output.weight", VOCAB, HIDDEN, 1)
    vector("output_norm.weight", (HIDDEN,), shard=1)
    columns = (NGRAM - 1) * ENGRAM_HEADS
    integers("engram_token_map.weight",
             generator.integers(0, COMPRESSED_VOCAB, size=(VOCAB,)).astype("<i4"), TYPE_I32)
    integers("engram_multipliers.weight", np.array([[1, 3, 5]], dtype="<i8"), TYPE_I64)
    integers("engram_primes.weight", np.array([PRIMES], dtype="<i8"), TYPE_I64)
    integers("engram_offsets.weight",
             np.array([offsets or [0, 5, 12, 23]], dtype="<i8"), TYPE_I64)
    mixes = (2 + HC) * HC
    for i, ratio in enumerate(RATIOS):
        g, shard = f"blk.{i}.", i % 2
        for site in ("attn", "ffn"):
            vector(g + f"hc_{site}_fn.weight", (mixes, HC * HIDDEN), shard=shard)
            vector(g + f"hc_{site}_base.weight", (mixes,), shard=shard)
            vector(g + f"hc_{site}_scale.weight", (3,), shard=shard)
        vector(g + "attn_norm.weight", (HIDDEN,), shard=shard)
        vector(g + "ffn_norm.weight", (HIDDEN,), shard=shard)
        q8(g + "attn_q_a.weight", Q_RANK, HIDDEN, shard)
        q8(g + "attn_kv.weight", HEAD_DIM, HIDDEN, shard)
        vector(g + "attn_q_a_norm.weight", (Q_RANK,), shard=shard)
        vector(g + "attn_kv_a_norm.weight", (HEAD_DIM,), shard=shard)
        q8(g + "attn_q_b.weight", HEADS * HEAD_DIM, Q_RANK, shard)
        q8(g + "attn_output_a.weight", GROUPS * O_RANK, HEADS * HEAD_DIM // GROUPS, shard)
        q8(g + "attn_output_b.weight", HIDDEN, GROUPS * O_RANK, shard)
        vector(g + "attn_sinks.weight", (HEADS,), shard=shard)
        if ratio:
            vector(g + "attn_compressor_kv.weight", (HEAD_DIM, HIDDEN), TYPE_BF16, shard)
            if ratio > 1:
                vector(g + "attn_compressor_gate.weight", (HEAD_DIM, HIDDEN), TYPE_BF16, shard)
            vector(g + "attn_compressor_norm.weight", (HEAD_DIM,), shard=shard)
            vector(g + "indexer.attn_k.weight", (INDEX_DIM, HEAD_DIM), TYPE_BF16, shard)
            vector(g + "indexer.k_norm.weight", (INDEX_DIM,), shard=shard)
            vector(g + "indexer.attn_q_b.weight", (INDEX_HEADS * INDEX_DIM, Q_RANK), TYPE_BF16,
                   shard)
            vector(g + "indexer.proj.weight", (INDEX_HEADS, HIDDEN), TYPE_BF16, shard)
        vector(g + "ffn_gate_inp.weight", (EXPERTS, HIDDEN), TYPE_BF16, shard)
        vector(g + "exp_probs_b.bias", (EXPERTS,), shard=shard)
        for bank, rows, k in (("gate", WIDTH, HIDDEN), ("up", WIDTH, HIDDEN),
                              ("down", HIDDEN, WIDTH)):
            put(g + f"ffn_{bank}_exps.weight", (EXPERTS, rows, k), TYPE_Q8_0,
                _blocks(generator, EXPERTS * rows, k, 32, 34), shard)
        q8(g + "ffn_gate_shexp.weight", WIDTH, HIDDEN, shard)
        q8(g + "ffn_up_shexp.weight", WIDTH, HIDDEN, shard)
        q8(g + "ffn_down_shexp.weight", HIDDEN, WIDTH, shard)
        if i in ENGRAM_LAYERS:
            q8(g + "engram_wkv.weight", (HC + 1) * HIDDEN, columns * ENGRAM_DIM, shard)
            vector(g + "engram_q.weight", (HC, HIDDEN), TYPE_BF16, shard)
            vector(g + "engram_k.weight", (HC, HIDDEN), TYPE_BF16, shard)
    first = tmp_path / "ds-00001-of-00002.gguf"
    write_gguf(first, kv or _gguf_kv(), shards[0])
    write_gguf(tmp_path / "ds-00002-of-00002.gguf", {"split.count": 2}, shards[1])
    table = tmp_path / "engram.gguf"
    rows = _blocks(generator, ENGRAM_ROWS, ENGRAM_DIM, 32, 18)
    write_gguf(table, {"general.architecture": "deepseek41-engram"},
               [("blk.0.engram_embd.weight", (ENGRAM_ROWS, ENGRAM_DIM), TYPE_IQ4_NL,
                 rows.tobytes())])
    stored["engram"] = rows
    return hf, first, table, stored


def _drafter_kv(**overrides):
    f = "dflash."
    kv = {
        "general.architecture": "dflash", f + "block_count": 1, f + "embedding_length": HIDDEN,
        f + "attention.head_count": HEADS, f + "attention.head_count_kv": 1,
        f + "attention.key_length": HEAD_DIM, f + "attention.value_length": HEAD_DIM,
        f + "rope.dimension_count": ROPE, f + "rope.freq_base": 10000.0,
        f + "attention.layer_norm_rms_epsilon": 1e-20, f + "attention.q_lora_rank": Q_RANK,
        f + "attention.sliding_window": 8, f + "expert_count": DRAFT_EXPERTS,
        f + "expert_used_count": 1, f + "expert_feed_forward_length": WIDTH,
        f + "expert_shared_count": 1, f + "expert_weights_scale": 1.5,
        f + "expert_weights_norm": True, f + "expert_gating_func": 4,
        f + "swiglu_clamp_exp": [10.0], f + "swiglu_clamp_shexp": [10.0],
        f + "attention.output_group_count": GROUPS, f + "attention.output_lora_rank": O_RANK,
        f + "attention.compress_ratios": [0], f + "hyper_connection.count": HC,
        f + "hyper_connection.sinkhorn_iterations": 20, f + "hyper_connection.epsilon": 1e-6,
        f + "hyper_connection.single_pass": True, f + "block_size": 3,
        f + "target_layers": DRAFT_TARGETS, f + "sample_from_anchor": True,
        f + "has_confidence_head": True,
    }
    kv.update(overrides)
    return kv


def _drafter(tmp_path, kv=None):
    """A one-block DSpark drafter GGUF (dflash) with MXFP4 experts."""

    generator = np.random.default_rng(23)
    tensors, stored = [], {}

    def put(name, shape, kind, data):
        tensors.append((name, shape, kind, data.tobytes()))
        stored[name] = data

    def q8(name, rows, columns):
        put(name, (rows, columns), TYPE_Q8_0, _blocks(generator, rows, columns, 32, 34))

    def vector(name, shape, kind=TYPE_F32):
        values = generator.normal(size=shape).astype(np.float32)
        if kind == TYPE_BF16:
            words = (values.view(np.uint32) >> 16).astype(np.uint16)
            tensors.append((name, shape, kind, words.tobytes()))
            stored[name] = (words.astype(np.uint32) << 16).view(np.float32)
        else:
            put(name, shape, kind, values)

    q8("fc.weight", HIDDEN, len(DRAFT_TARGETS) * HIDDEN)
    vector("enc.output_norm.weight", (HIDDEN,))
    vector("output_norm.weight", (HIDDEN,))
    vector("markov_w1.weight", (VOCAB, DRAFT_RANK), TYPE_BF16)
    vector("markov_w2.weight", (VOCAB, DRAFT_RANK), TYPE_BF16)
    vector("conf_proj.weight", (1, HIDDEN + DRAFT_RANK))
    mixes = (2 + HC) * HC
    g = "blk.0."
    for site in ("attn", "ffn"):
        vector(g + f"hc_{site}_fn.weight", (mixes, HC * HIDDEN))
        vector(g + f"hc_{site}_base.weight", (mixes,))
        vector(g + f"hc_{site}_scale.weight", (3,))
    vector(g + "attn_norm.weight", (HIDDEN,))
    vector(g + "ffn_norm.weight", (HIDDEN,))
    q8(g + "attn_q_a.weight", Q_RANK, HIDDEN)
    q8(g + "attn_kv.weight", HEAD_DIM, HIDDEN)
    vector(g + "attn_q_a_norm.weight", (Q_RANK,))
    vector(g + "attn_kv_a_norm.weight", (HEAD_DIM,))
    q8(g + "attn_q_b.weight", HEADS * HEAD_DIM, Q_RANK)
    q8(g + "attn_output_a.weight", GROUPS * O_RANK, HEADS * HEAD_DIM // GROUPS)
    q8(g + "attn_output_b.weight", HIDDEN, GROUPS * O_RANK)
    vector(g + "attn_sinks.weight", (HEADS,))
    vector(g + "ffn_gate_inp.weight", (DRAFT_EXPERTS, HIDDEN))
    vector(g + "exp_probs_b.bias", (DRAFT_EXPERTS,))
    for bank, rows, k in (("gate", WIDTH, HIDDEN), ("up", WIDTH, HIDDEN), ("down", HIDDEN, WIDTH)):
        blocks = generator.integers(0, 256, size=(DRAFT_EXPERTS * rows, k // 32 * 17),
                                    dtype=np.uint8)
        put(g + f"ffn_{bank}_exps.weight", (DRAFT_EXPERTS, rows, k), TYPE_MXFP4, blocks)
    q8(g + "ffn_gate_shexp.weight", WIDTH, HIDDEN)
    q8(g + "ffn_up_shexp.weight", WIDTH, HIDDEN)
    q8(g + "ffn_down_shexp.weight", HIDDEN, WIDTH)
    path = tmp_path / "dspark.gguf"
    write_gguf(path, kv or _drafter_kv(), tensors)
    return path, stored


def _mmproj(tmp_path, projector="deepseek4v"):
    """A two-block DeepSeek-V4.1 vision tower as llama.cpp's clip GGUF: BF16 matrices, the patch
    convolution, norms, biases and span rows as F32 holding BF16 values."""

    generator = np.random.default_rng(31)
    tensors, stored = [], {}

    def bf16_values(shape):
        values = generator.normal(size=shape).astype(np.float32)
        return ((values.view(np.uint32) >> 16) << 16).view(np.float32)

    def put(name, shape, kind=TYPE_F32):
        values = bf16_values(shape)
        stored[name] = values
        if kind == TYPE_BF16:
            words = (values.view(np.uint32) >> 16).astype(np.uint16)
            tensors.append((name, shape, kind, words.tobytes()))
        else:
            tensors.append((name, shape, kind, values.tobytes()))

    put("v.patch_embd.weight", (V_HIDDEN, 3, V_PATCH, V_PATCH))
    put("v.patch_embd.bias", (V_HIDDEN,))
    for i in range(V_LAYERS):
        g = f"v.blk.{i}."
        put(g + "ln1.weight", (V_HIDDEN,))
        put(g + "ln2.weight", (V_HIDDEN,))
        put(g + "attn_qkv.weight", (3 * V_HIDDEN, V_HIDDEN), TYPE_BF16)
        put(g + "attn_qkv.bias", (3 * V_HIDDEN,))
        put(g + "attn_out.weight", (V_HIDDEN, V_HIDDEN), TYPE_BF16)
        put(g + "attn_out.bias", (V_HIDDEN,))
        put(g + "ffn_gate.weight", (V_WIDTH, V_HIDDEN), TYPE_BF16)
        put(g + "ffn_up.weight", (V_WIDTH, V_HIDDEN), TYPE_BF16)
        put(g + "ffn_down.weight", (V_HIDDEN, V_WIDTH), TYPE_BF16)
    put("v.post_ln.weight", (V_HIDDEN,))
    put("mm.1.weight", (HIDDEN, V_HIDDEN * V_MERGE * V_MERGE), TYPE_BF16)
    put("mm.1.bias", (HIDDEN,))
    put("mm.2.weight", (HIDDEN, HIDDEN), TYPE_BF16)
    put("mm.2.bias", (HIDDEN,))
    for name in ("v.token_embd.img_start", "v.image_newline", "v.token_embd.img_end"):
        put(name, (HIDDEN,))
    v = "clip.vision."
    kv = {"general.architecture": "clip", "clip.has_vision_encoder": True,
          "clip.projector_type": projector, "clip.use_silu": True,
          v + "block_count": V_LAYERS, v + "embedding_length": V_HIDDEN,
          v + "attention.head_count": V_HEADS, v + "feed_forward_length": V_WIDTH,
          v + "patch_size": V_PATCH, v + "projection_dim": HIDDEN,
          v + "projector.scale_factor": V_MERGE, v + "image_min_pixels": 64,
          v + "image_mean": [0.5] * 3, v + "image_std": [0.5] * 3,
          v + "attention.layer_norm_epsilon": 1e-6}
    path = tmp_path / "mmproj.gguf"
    write_gguf(path, kv, tensors)
    return path, stored


def _bias_vl(tmp_path, layers=3):
    """The release's bias_vl tensors as tools/convert/fetch_tensors.py writes them."""

    from tools.convert.fetch_tensors import write_safetensors

    generator = np.random.default_rng(37)
    values = {f"layers.{i}.ffn.gate.bias_vl": generator.normal(size=EXPERTS).astype(np.float32)
              for i in range(layers)}
    path = tmp_path / "bias_vl.safetensors"
    write_safetensors(path, {name: ({"dtype": "F32", "shape": [EXPERTS]}, v.tobytes())
                             for name, v in values.items()}, {})
    return path, values


def _convert(tmp_path, hf, first, table, components=("text", "engram"), drafter=None,
             mmproj=None, bias_vl=None, resource_overrides=None):
    from tools.convert.pipeline import convert
    from tools.convert.recipe import Recipe
    from tools.convert.sources.safetensors import SafetensorsSource

    output = tmp_path / ("-".join(components) + ".ninfer")
    with SafetensorsSource(hf) as base, GGUFFile(table) as engram:
        # A table alone reads the GGUF for the Engram hash its profile is checked against.
        gguf = (deepseek_v41.GGUFShards(first) if "text" in components or resource_overrides
                else None)
        dspark = None
        if drafter:
            dspark = (SafetensorsSource(drafter) if drafter.is_dir()
                      else deepseek_v41.GGUFShards(drafter))
        vision = GGUFFile(mmproj) if mmproj else None
        bias = SafetensorsSource(bias_vl) if bias_vl else None
        model = deepseek_v41.build_model(base, gguf, components=components, dspark=dspark,
                                         vision=vision, bias_vl=bias,
                                         resource_overrides=resource_overrides)
        recipe = Recipe(model)
        sources = ({"base": base, "engram": engram} | ({"gguf": gguf} if gguf else {})
                   | ({"dspark": dspark} if dspark else {}))
        deepseek_v41.deepseek_v41_gguf(model, recipe, sources)
        convert(model, recipe, output, device="cpu", rows_per_chunk=96)
        for source in (gguf, dspark, vision):
            if source is not None:
                source.__exit__(None, None, None)
        if bias is not None:
            bias.close()
    return output


def test_text_config_refuses_other_variants():
    config = deepseek_v41.text_config(_hf_config())
    assert config["compress_ratios"] == RATIOS  # the drafter's entry is dropped
    for bad in (dict(num_key_value_heads=2), dict(scoring_func="sigmoid"),
                dict(compress_ratios=[0, 4, 1]), dict(hidden_act="gelu"),
                dict(norm_topk_prob=False), dict(tie_word_embeddings=True),
                dict(rope_scaling={"rope_type": "linear"})):
        with pytest.raises(ValueError):
            deepseek_v41.text_config(_hf_config(**bad))


@pytest.mark.parametrize("bad", [
    {"deepseek41.expert_gating_func": 2},
    {"deepseek41.attention.compress_ratios": [0, 1, 1, 0]},
    {"deepseek41.attention.candidate_source_layer_id": 1},
    {"deepseek41.expert_weights_scale": 2.5},
    {"deepseek41.swiglu_clamp_exp": [7.0] * 3},
    {"deepseek41.expert_scoring_func": "softmax"},
])
def test_the_gguf_metadata_must_match_the_config(tmp_path, bad):
    from tools.convert.sources.safetensors import SafetensorsSource

    hf, first, _, _ = _release(tmp_path, _gguf_kv(**bad))
    with SafetensorsSource(hf) as base, deepseek_v41.GGUFShards(first) as gguf:
        with pytest.raises(ValueError):
            deepseek_v41.build_model(base, gguf)


def test_engram_columns_must_stay_inside_their_table(tmp_path):
    from tools.convert.sources.safetensors import SafetensorsSource

    hf, first, _, _ = _release(tmp_path, offsets=[0, 5, 12, 24])
    with SafetensorsSource(hf) as base, deepseek_v41.GGUFShards(first) as gguf:
        with pytest.raises(ValueError, match="past its table"):
            deepseek_v41.build_model(base, gguf)


def test_the_artifact_keeps_the_gguf_rows_and_the_engram_table(tmp_path):
    from tools.artifact.reader import Artifact

    hf, first, table, stored = _release(tmp_path)
    output = _convert(tmp_path, hf, first, table)
    with Artifact(output) as artifact:
        directory = artifact.directory
        text = directory.components["text"]
        config = text["config"]
        assert config["architectures"] == ["DeepseekV41ForCausalLM"]
        assert config["engram"]["primes"] == [PRIMES]
        assert config["engram"]["multipliers"] == [[1, 3, 5]]
        bindings = directory.bindings

        def stored_bytes(name):
            return artifact.read_object(bindings[name]["object"])

        def floats(name):
            return np.frombuffer(stored_bytes(name), dtype=np.float32)

        # The shards' blocks row for row, the first shard's and the second's.
        assert stored_bytes("text/token_embedding") == stored["token_embd.weight"].tobytes()
        assert stored_bytes("text/output_head") == stored["output.weight"].tobytes()
        for layer in range(3):
            g, p = f"blk.{layer}.", f"text/layers/{layer}/"
            gate, up = stored[g + "ffn_gate_exps.weight"], stored[g + "ffn_up_exps.weight"]
            expected = np.concatenate([np.concatenate([gate[e * WIDTH:(e + 1) * WIDTH],
                                                       up[e * WIDTH:(e + 1) * WIDTH]])
                                       for e in range(EXPERTS)])
            gate_up = np.frombuffer(stored_bytes(p + "moe/gate_up"),
                                    dtype=np.uint8).reshape(-1, gate.shape[1])
            assert np.array_equal(gate_up, expected)
            assert stored_bytes(p + "moe/down") == stored[g + "ffn_down_exps.weight"].tobytes()
            # BF16 and F32 vectors become FP32 exactly.
            assert np.array_equal(floats(p + "moe/router"),
                                  stored[g + "ffn_gate_inp.weight"].reshape(-1))
            assert np.array_equal(floats(p + "hc_attention/fn"),
                                  stored[g + "hc_attn_fn.weight"].reshape(-1))
            assert (p + "attention/compressor/kv" in bindings) == (layer in (1, 2))
            assert (p + "attention/compressor/gate" in bindings) == (layer == 1)
        assert np.array_equal(floats("text/layers/2/attention/indexer/query"),
                              stored["blk.2.indexer.attn_q_b.weight"].reshape(-1))
        # The Engram gate is the exact FP32 product of its BF16 factors.
        assert np.array_equal(floats("text/layers/0/engram/gate"),
                              (stored["blk.0.engram_q.weight"] *
                               stored["blk.0.engram_k.weight"]).reshape(-1))
        assert "text/layers/1/engram/gate" not in bindings
        # The table keeps its IQ4_NL rows, named by their digest.
        table_bytes = stored["engram"].tobytes()
        assert stored_bytes("engram/table") == table_bytes
        engram = directory.components["engram"]["config"]
        assert engram["table_sha256"] == hashlib.sha256(table_bytes).hexdigest()
        assert engram["format"] == "gguf_iq4_nl" and engram["rows"] == ENGRAM_ROWS
        # The release names no added tokens in tokenizer_config.json; the artifact's does.
        resources = text["resources"]
        tokenizer_config = json.loads(artifact.read_object(resources["tokenizer_config.json"]))
        assert sorted(tokenizer_config["added_tokens_decoder"]) == ["0", "1", "2"]
        assert artifact.read_object(resources["engram_token_map"]) == \
            stored["engram_token_map.weight"].tobytes()
        assert artifact.read_object(resources["chat_template.jinja"]) == \
            (deepseek_v41.DEFAULT_RESOURCES / "chat_template.jinja").read_bytes()


def test_the_model_and_its_table_convert_apart(tmp_path):
    from tools.artifact.reader import Artifact

    hf, first, table, stored = _release(tmp_path)
    digest = hashlib.sha256(stored["engram"].tobytes()).hexdigest()
    with Artifact(_convert(tmp_path, hf, first, table, ("text",))) as artifact:
        directory = artifact.directory
        assert "engram/table" not in directory.bindings
        assert "text/token_embedding" in directory.bindings
        assert directory.components["engram"]["config"]["table_sha256"] == digest
    with Artifact(_convert(tmp_path, hf, first, table, ("engram",))) as artifact:
        directory = artifact.directory
        assert set(directory.bindings) == {"engram/table"}
        assert "text" not in directory.components
        assert directory.components["engram"]["config"]["table_sha256"] == digest


def test_the_engram_fingerprint_matches_the_engine():
    # tests/models/deepseek_v41/test_engram_table.cpp computes the same value for this hash.
    config = {"engram_layer_ids": [1, 3], "engram_num_embeddings": [40, 50],
              "engram_max_ngram_size": 3, "engram_n_heads": 2}
    constants = {"pad_id": 0, "multipliers": [[3, 5, 7], [11, 13, 17]],
                 "primes": [[7, 11, 13, 5], [13, 11, 17, 7]],
                 "offsets": [[0, 7, 18, 31], [0, 13, 24, 41]],
                 "token_map": np.arange(16, dtype="<i4") % 9}
    assert deepseek_v41.engram_hash_fingerprint(config, constants) == 0xAE571DFFA7A86675


def _engram_profile(rows, fingerprint, ranked):
    return (b"NFNGHOT1" + struct.pack("<4Q", rows, fingerprint, 777, len(ranked))
            + np.asarray(ranked, dtype="<u4").tobytes())


def test_the_table_stores_a_checked_hot_row_profile(tmp_path):
    from tools.artifact.reader import Artifact
    from tools.convert.sources.safetensors import SafetensorsSource

    hf, first, table, _ = _release(tmp_path)
    with SafetensorsSource(hf) as base, deepseek_v41.GGUFShards(first) as gguf:
        config = deepseek_v41.text_config(base.config)
        fingerprint = deepseek_v41.engram_hash_fingerprint(
            config, deepseek_v41.engram_constants(gguf, config))
    good = tmp_path / "good.hot"
    good.write_bytes(_engram_profile(ENGRAM_ROWS, fingerprint, [7, 0, ENGRAM_ROWS - 1]))
    for components in (("engram",), ("text", "engram")):
        output = _convert(tmp_path, hf, first, table, components,
                          resource_overrides={"engram.hot": str(good)})
        with Artifact(output) as artifact:
            role = artifact.directory.components["engram"]["resources"]["hot_profile"]
            assert artifact.read_object(role) == good.read_bytes()
            assert "engram/table" in artifact.directory.bindings
    bad = {
        "foreign": _engram_profile(ENGRAM_ROWS, fingerprint ^ 1, [7]),
        "resized": _engram_profile(ENGRAM_ROWS + 1, fingerprint, [7]),
        "repeated": _engram_profile(ENGRAM_ROWS, fingerprint, [7, 7]),
        "past": _engram_profile(ENGRAM_ROWS, fingerprint, [ENGRAM_ROWS]),
    }
    for name, data in bad.items():
        path = tmp_path / f"{name}.hot"
        path.write_bytes(data)
        with pytest.raises(ValueError, match="Engram hash|duplicate or out-of-range"):
            _convert(tmp_path, hf, first, table, ("engram",),
                     resource_overrides={"engram.hot": str(path)})
    # The profile needs a stored table, and the GGUF's Engram hash to be checked against.
    with SafetensorsSource(hf) as base, deepseek_v41.GGUFShards(first) as gguf:
        with pytest.raises(ValueError, match="engram.hot"):
            deepseek_v41.build_model(base, gguf, components=("text",),
                                     resource_overrides={"engram.hot": str(good)})
        with pytest.raises(ValueError, match="engram.hot"):
            deepseek_v41.build_model(base, None, components=("engram",),
                                     resource_overrides={"engram.hot": str(good)})


def test_the_drafter_keeps_its_blocks(tmp_path):
    from tools.artifact.reader import Artifact

    hf, first, table, _ = _release(tmp_path)
    drafter, stored = _drafter(tmp_path)
    output = _convert(tmp_path, hf, first, table, ("text", "engram", "dspark"), drafter)
    with Artifact(output) as artifact:
        directory = artifact.directory
        config = directory.components["dspark"]["config"]
        assert config == {"architectures": ["DeepseekV41DSpark"],
                          "model_type": "deepseek_v41_dspark", "num_hidden_layers": 1,
                          "block_size": 3, "markov_rank": DRAFT_RANK,
                          "n_routed_experts": DRAFT_EXPERTS, "num_experts_per_tok": 1,
                          "noise_token_id": NOISE, "target_layer_ids": DRAFT_TARGETS}
        bindings = directory.bindings

        def stored_bytes(name):
            return artifact.read_object(bindings[name]["object"])

        assert stored_bytes("dspark/fc") == stored["fc.weight"].tobytes()
        # The MXFP4 experts as stored, each expert's gate rows then its up rows.
        gate, up = stored["blk.0.ffn_gate_exps.weight"], stored["blk.0.ffn_up_exps.weight"]
        expected = np.concatenate([np.concatenate([gate[e * WIDTH:(e + 1) * WIDTH],
                                                   up[e * WIDTH:(e + 1) * WIDTH]])
                                   for e in range(DRAFT_EXPERTS)])
        assert stored_bytes("dspark/layers/0/moe/gate_up") == expected.tobytes()
        assert stored_bytes("dspark/layers/0/moe/down") == \
            stored["blk.0.ffn_down_exps.weight"].tobytes()
        assert artifact.by_id[bindings["dspark/layers/0/moe/down"]["object"]].format == "gguf_mxfp4"
        # The Markov tables stay BF16, exactly; the confidence head is FP32.
        embed = np.frombuffer(stored_bytes("dspark/markov_embed"), dtype=np.uint16)
        assert np.array_equal((embed.astype(np.uint32) << 16).view(np.float32),
                              stored["markov_w1.weight"].reshape(-1))
        assert np.array_equal(np.frombuffer(stored_bytes("dspark/confidence"), dtype=np.float32),
                              stored["conf_proj.weight"].reshape(-1))


def _stq_pack(tmp_path, config=None, expert_codebook="pair1"):
    """The drafter of a Sushi STQ pack (mtp.0): random trellis words and sign vectors of every
    matrix, the trunk's beside SPEC1 metadata and the routed experts' beside PAIR1, BF16 vectors,
    the release's config."""

    from tools.convert.fetch_tensors import write_safetensors

    generator = np.random.default_rng(41)
    pack = tmp_path / "stq"
    pack.mkdir()
    (pack / "config.json").write_text(json.dumps(config or _hf_config()))
    files = {"trunk": {}, "experts": {}}
    stored = {}

    def bf16(name, shape, file="trunk"):
        words = (generator.normal(size=shape).astype(np.float32).view(np.uint32) >> 16)
        stored[name] = words.astype(np.uint16)
        files[file][name] = ({"dtype": "BF16", "shape": list(shape)}, stored[name].tobytes())

    def stq(name, k, n, words, count=None, file="trunk"):
        lead = [] if count is None else [count]
        trellis = generator.integers(0, 1 << 16, size=lead + [k // 16, n // 16, words],
                                     dtype=np.uint16)
        files[file][name + ".trellis"] = ({"dtype": "U16", "shape": list(trellis.shape)},
                                          trellis.tobytes())
        stored[name + ".trellis"] = trellis
        for part, size in (("suh", k), ("svh", n)):
            signs = generator.choice([-1.0, 1.0], size=lead + [size]).astype(np.float16)
            files[file][f"{name}.{part}"] = ({"dtype": "F16", "shape": list(signs.shape)},
                                             signs.tobytes())
            stored[f"{name}.{part}"] = signs

    g, mixes = "mtp.0.", (2 + HC) * HC
    stq(g + "main_proj", len(DRAFT_TARGETS) * HIDDEN, HIDDEN, 64)
    for name, shape in (("main_norm.weight", (HIDDEN,)), ("norm.weight", (HIDDEN,)),
                        ("markov_head.embed.weight", (VOCAB, DRAFT_RANK)),
                        ("markov_head.head.weight", (VOCAB, DRAFT_RANK)),
                        ("confidence_head.proj.weight", (HIDDEN + DRAFT_RANK,)),
                        ("attn_norm.weight", (HIDDEN,)), ("ffn_norm.weight", (HIDDEN,)),
                        ("attn.q_norm.weight", (Q_RANK,)), ("attn.kv_norm.weight", (HEAD_DIM,)),
                        ("attn.attn_sink", (HEADS,)), ("ffn.gate.weight", (DRAFT_EXPERTS, HIDDEN)),
                        ("ffn.gate.bias", (DRAFT_EXPERTS,))):
        bf16(g + name, shape)
    for site in ("attn", "ffn"):
        bf16(g + f"hc_{site}_fn", (mixes, HC * HIDDEN))
        bf16(g + f"hc_{site}_base", (mixes,))
        bf16(g + f"hc_{site}_scale", (3,))
    stq(g + "attn.wq_a", HIDDEN, Q_RANK, 64)
    stq(g + "attn.wkv", HIDDEN, HEAD_DIM, 64)
    stq(g + "attn.wq_b", Q_RANK, HEADS * HEAD_DIM, 64)
    stq(g + "attn.wo_a", HEADS * HEAD_DIM // GROUPS, O_RANK, 64, GROUPS)
    stq(g + "attn.wo_b", GROUPS * O_RANK, HIDDEN, 64)
    stq(g + "ffn.shared_experts.w1", HIDDEN, WIDTH, 64)
    stq(g + "ffn.shared_experts.w3", HIDDEN, WIDTH, 64)
    stq(g + "ffn.shared_experts.w2", WIDTH, HIDDEN, 64)
    for bank, k, n in (("gate_proj", HIDDEN, WIDTH), ("up_proj", HIDDEN, WIDTH),
                       ("down_proj", WIDTH, HIDDEN)):
        stq(g + f"mlp.switch_mlp.{bank}.g0", k, n, 32, DRAFT_EXPERTS, "experts")
    names = {"trunk": "model-mtp-trunk-0.safetensors", "experts": "model-mtp-0.safetensors"}
    metadata = {"trunk": {"format": "stq", "codebook": "spec1", "k": "4"},
                "experts": {"format": "stq", "codebook": expert_codebook,
                            "k": "2" if expert_codebook == "pair1" else "4"}}
    for file, tensors in files.items():
        write_safetensors(pack / names[file], tensors, metadata[file])
    (pack / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {name: names[file] for file, tensors in files.items() for name in tensors}}))
    return pack, stored


def test_the_stq_drafter_keeps_its_words(tmp_path):
    from tools.artifact.reader import Artifact

    hf, first, table, _ = _release(tmp_path)
    pack, stored = _stq_pack(tmp_path)
    output = _convert(tmp_path, hf, first, table, ("text", "dspark"), pack)
    with Artifact(output) as artifact:
        directory = artifact.directory
        assert directory.components["dspark"]["config"]["stq"] == {
            "trunk": "spec1_w15_n64", "experts": "pair1_w16_n32"}
        bindings = directory.bindings

        def stored_bytes(name):
            return artifact.read_object(bindings[name]["object"])

        # Trellis words and signs exactly as the pack stores them, a leading count on the grouped
        # output projection and the expert banks.
        for name, tensor in (("dspark/fc", "mtp.0.main_proj"),
                             ("dspark/layers/0/attention/output_a", "mtp.0.attn.wo_a"),
                             ("dspark/layers/0/moe/shared_down", "mtp.0.ffn.shared_experts.w2"),
                             ("dspark/layers/0/moe/up", "mtp.0.mlp.switch_mlp.up_proj.g0")):
            for part in ("trellis", "suh", "svh"):
                assert stored_bytes(f"{name}/{part}") == stored[f"{tensor}.{part}"].tobytes()
        assert artifact.by_id[bindings["dspark/fc/trellis"]["object"]].format == "uint16"
        assert artifact.by_id[bindings["dspark/fc/suh"]["object"]].format == "fp16"
        # Vectors widen to FP32 exactly; the Markov tables stay BF16.
        tap_norm = np.frombuffer(stored_bytes("dspark/tap_norm"), dtype=np.float32)
        assert np.array_equal(tap_norm.view(np.uint32) >> 16, stored["mtp.0.main_norm.weight"])
        assert stored_bytes("dspark/markov_head") == stored["mtp.0.markov_head.head.weight"].tobytes()


@pytest.mark.parametrize("bad", ["config", "codebook"])
def test_the_stq_pack_must_match_the_release(tmp_path, bad):
    from tools.convert.sources.safetensors import SafetensorsSource

    hf, first, _, _ = _release(tmp_path)
    config = _hf_config()
    if bad == "config":
        config["text_config"]["dspark_block_size"] = 4
    pack, _ = _stq_pack(tmp_path, config, "pair1" if bad == "config" else "spec1")
    with SafetensorsSource(hf) as base, deepseek_v41.GGUFShards(first) as gguf, \
            SafetensorsSource(pack) as dspark:
        with pytest.raises(ValueError):
            deepseek_v41.build_model(base, gguf, components=("text", "dspark"), dspark=dspark)


@pytest.mark.parametrize("bad", [
    {"dflash.target_layers": [0, 2]},
    {"dflash.expert_used_count": 2},
    {"dflash.hyper_connection.single_pass": False},
    {"dflash.attention.compress_ratios": [2]},
])
def test_the_drafter_metadata_must_match_the_config(tmp_path, bad):
    from tools.convert.sources.safetensors import SafetensorsSource

    hf, first, _, _ = _release(tmp_path)
    drafter, _ = _drafter(tmp_path, _drafter_kv(**bad))
    with SafetensorsSource(hf) as base, deepseek_v41.GGUFShards(first) as gguf, \
            deepseek_v41.GGUFShards(drafter) as dspark:
        with pytest.raises(ValueError):
            deepseek_v41.build_model(base, gguf, components=("text", "engram", "dspark"),
                                     dspark=dspark)


def test_the_vision_tower_keeps_the_release_values(tmp_path):
    from tools.artifact.reader import Artifact

    hf, first, table, _ = _release(tmp_path)
    mmproj, stored = _mmproj(tmp_path)
    bias_vl, bias = _bias_vl(tmp_path)
    output = _convert(tmp_path, hf, first, table, ("text", "engram", "vision"), mmproj=mmproj,
                      bias_vl=bias_vl)
    with Artifact(output) as artifact:
        directory = artifact.directory
        config = directory.components["vision"]["config"]
        assert config == {"architectures": ["DeepseekV41VisionModel"],
                          "model_type": "deepseek_v41_vision", "num_hidden_layers": V_LAYERS,
                          "hidden_size": V_HIDDEN, "num_attention_heads": V_HEADS,
                          "intermediate_size": V_WIDTH, "patch_size": V_PATCH,
                          "downsample_ratio": V_MERGE, "max_image_tokens": 1024,
                          "min_pixels": 64, "rope_theta": 10000.0, "max_wh_ratio": None,
                          "image_token_id": IMAGE_TOKEN, "projection_dim": HIDDEN,
                          "rms_norm_eps": 1e-6}
        # The release's preprocessing, which the frontend reads.
        resource = directory.components["vision"]["resources"]["processor_config.json"]
        processor = json.loads(artifact.read_object(resource))["image_processor"]
        assert processor == {"patch_size": V_PATCH, "downsample_ratio": V_MERGE, "min_pixels": 64,
                             "max_image_tokens": 1024, "max_wh_ratio": None, "resample": 3,
                             "pad_color": [127] * 3, "image_mean": [0.5] * 3,
                             "image_std": [0.5] * 3, "image_token": "<｜deepseek_image｜>"}
        bindings = directory.bindings

        def values(name):
            binding = bindings[name]
            data = artifact.read_object(binding["object"])
            if artifact.by_id[binding["object"]].format == "bf16":
                words = np.frombuffer(data, dtype=np.uint16)
                return (words.astype(np.uint32) << 16).view(np.float32)
            return np.frombuffer(data, dtype=np.float32)

        # The convolution as the Linear over (channel, y, x); matrices BF16, vectors FP32, exactly.
        assert artifact.by_id[bindings["vision/patch_embedding"]["object"]].format == "bf16"
        assert np.array_equal(values("vision/patch_embedding"),
                              stored["v.patch_embd.weight"].reshape(-1))
        for i in range(V_LAYERS):
            g, p = f"v.blk.{i}.", f"vision/layers/{i}/"
            for name, tensor in (("attention/qkv", "attn_qkv.weight"),
                                 ("attention/qkv_bias", "attn_qkv.bias"),
                                 ("attention/output", "attn_out.weight"),
                                 ("attention_norm", "ln1.weight"), ("mlp_norm", "ln2.weight"),
                                 ("mlp/down", "ffn_down.weight")):
                assert np.array_equal(values(p + name), stored[g + tensor].reshape(-1)), name
            # Gate and up share one parent, the gate's rows first.
            parent = lambda name: [part["object"] for part in bindings[name]["parts"]]
            assert parent(p + "mlp/gate") == parent(p + "mlp/up")
            words = np.frombuffer(artifact.read_object(parent(p + "mlp/gate")[0]), np.uint16)
            both = (words.astype(np.uint32) << 16).view(np.float32)
            assert np.array_equal(both, np.concatenate([stored[g + "ffn_gate.weight"].reshape(-1),
                                                        stored[g + "ffn_up.weight"].reshape(-1)]))
        for name, tensor in (("vision/aligner/up", "mm.1.weight"),
                             ("vision/aligner/out_bias", "mm.2.bias"),
                             ("vision/final_norm", "v.post_ln.weight"),
                             ("vision/image_start", "v.token_embd.img_start"),
                             ("vision/image_newline", "v.image_newline"),
                             ("vision/image_end", "v.token_embd.img_end")):
            assert np.array_equal(values(name), stored[tensor].reshape(-1)), name
        assert np.array_equal(values("vision/image_router_bias"),
                              np.concatenate([bias[f"layers.{i}.ffn.gate.bias_vl"]
                                              for i in range(3)]))


def test_the_vision_tower_needs_its_sources(tmp_path):
    from tools.convert.sources.safetensors import SafetensorsSource

    hf, first, _, _ = _release(tmp_path)
    mmproj, _ = _mmproj(tmp_path)
    short, _ = _bias_vl(tmp_path, layers=2)
    with SafetensorsSource(hf) as base, deepseek_v41.GGUFShards(first) as gguf, \
            GGUFFile(mmproj) as vision, SafetensorsSource(short) as bias:
        with pytest.raises(ValueError, match="bias_vl"):
            deepseek_v41.build_model(base, gguf, components=("text", "vision"), vision=vision)
        with pytest.raises(ValueError, match="lacks layers.2"):
            deepseek_v41.build_model(base, gguf, components=("text", "vision"), vision=vision,
                                     bias_vl=bias)
    other, _ = _mmproj(tmp_path, projector="gemma4v")
    with SafetensorsSource(hf) as base, deepseek_v41.GGUFShards(first) as gguf, \
            GGUFFile(other) as vision, SafetensorsSource(short) as bias:
        with pytest.raises(ValueError, match="projector"):
            deepseek_v41.build_model(base, gguf, components=("text", "vision"), vision=vision,
                                     bias_vl=bias)


def test_fetch_tensors_reads_only_the_matching_tensors(tmp_path):
    from tools.convert import fetch_tensors
    from tools.convert.sources.safetensors import SafetensorsSource

    generator = np.random.default_rng(41)
    shards = {"a.safetensors": {"layers.0.ffn.gate.bias_vl": np.float32,
                                "layers.0.ffn.gate.weight": np.float32},
              "b.safetensors": {"layers.1.ffn.gate.bias_vl": np.float32,
                                "embed.weight": np.float32}}
    expected, weight_map = {}, {}
    for file, names in shards.items():
        tensors = {}
        for name in names:
            values = generator.normal(size=(5,)).astype(np.float32)
            expected[name] = values
            tensors[name] = ({"dtype": "F32", "shape": [5]}, values.tobytes())
            weight_map[name] = file
        fetch_tensors.write_safetensors(tmp_path / file, tensors, {})
    reads = []

    def read(file, begin, end):
        reads.append((file, begin, end))
        with open(tmp_path / file, "rb") as stream:
            stream.seek(begin)
            return stream.read(end - begin)

    got = fetch_tensors.fetch(read, {"weight_map": weight_map}, ["layers.*.ffn.gate.bias_vl"])
    assert list(got) == ["layers.0.ffn.gate.bias_vl", "layers.1.ffn.gate.bias_vl"]
    # Two header reads per shard and one data read per tensor, nothing else.
    assert len(reads) == 2 * 2 + 2
    out = tmp_path / "bias.safetensors"
    fetch_tensors.write_safetensors(out, got, {"repo": "r"})
    with SafetensorsSource(out) as source:
        for name, values in got.items():
            assert np.array_equal(source.read_flat(name).numpy(), expected[name])
    with pytest.raises(ValueError, match="matches"):
        fetch_tensors.fetch(read, {"weight_map": weight_map}, ["nothing.*"])


# The earlier layout's metadata that llama.cpp's layout replaces (ISTA-DASLab's
# convert_ds41_upstream.py, OLD_KEYS).
EARLIER_KEYS = ("attention.candidate_block_size", "attention.candidate_topk_blocks",
                "attention.candidate_source_layer_id", "kv_source_layer_ids",
                "index_source_layer_ids", "expert_scoring_func", "engram.layer_ids",
                "engram.num_embeddings", "engram_head_dim", "engram_n_heads",
                "engram_max_ngram_size", "engram_pad_token_id")
HASH_TENSORS = ("engram_token_map.weight", "engram_primes.weight", "engram_offsets.weight",
                "engram_multipliers.weight")


def _upstream(tmp_path, first, table, bias, keep=None, link=False, **kv_overrides):
    """_release's GGUFs in llama.cpp's layout, rewritten as ISTA-DASLab's convert_ds41_upstream.py
    does: llama.cpp's metadata names and the Engram hash in metadata (offsets into the whole
    table), `engram_kv`, the gate as the F32 product of its BF16 factors, the image routing bias
    `exp_probs_b_vl`, a metadata-only first shard, every tensor in the second and the Engram GGUF
    as the third (linked under the shard's name with `link`). `keep` prunes the routed experts."""

    f = "deepseek41."
    with deepseek_v41.GGUFShards(first) as old, GGUFFile(table) as engram:
        tensors = {name: (info, file.tensor_bytes(name).tobytes())
                   for file in old.files for name, info in file.tensors.items()}
        integers = {name: np.frombuffer(tensors[name][1],
                                        dtype="<i4" if name == HASH_TENSORS[0] else "<i8")
                    for name in HASH_TENSORS}
        primes = integers["engram_primes.weight"].tolist()
        kv = {key: value for key, value in old.kv.items()
              if not key.startswith("split.") and key.removeprefix(f) not in EARLIER_KEYS}
        kv.update({
            f + "attention.kv_source_layers": old.kv[f + "kv_source_layer_ids"],
            f + "attention.indexer.source_layers": old.kv[f + "index_source_layer_ids"],
            f + "attention.indexer.candidate_source_layer":
                old.kv[f + "attention.candidate_source_layer_id"],
            f + "attention.indexer.candidate_top_k_blocks":
                old.kv[f + "attention.candidate_topk_blocks"],
            f + "attention.indexer.candidate_block_size":
                old.kv[f + "attention.candidate_block_size"],
            f + "ple.layers": old.kv[f + "engram.layer_ids"],
            f + "ple.ngram_size": old.kv[f + "engram_max_ngram_size"],
            f + "ple.heads_per_ngram": old.kv[f + "engram_n_heads"],
            f + "embedding_length_per_layer_input": old.kv[f + "engram_head_dim"],
            f + "ple.layer_multipliers": integers["engram_multipliers.weight"].tolist(),
            f + "ple.head_offsets": [sum(primes[:i]) for i in range(len(primes))],
            f + "ple.head_vocab_sizes": primes,
            f + "engram.pad_token_id": old.kv[f + "engram_pad_token_id"],
            f + "engram.token_map": integers["engram_token_map.weight"].tolist(),
            "split.count": 3,
        })
        experts = list(range(EXPERTS)) if keep is None else list(keep)
        kv[f + "expert_count"] = len(experts)
        kv.update(kv_overrides)
        written = []
        for name, (info, raw) in tensors.items():
            shape = info.shape
            if name in HASH_TENSORS or name.endswith((".engram_q.weight", ".engram_k.weight")):
                continue
            if name.endswith(("_exps.weight", ".ffn_gate_inp.weight", ".exp_probs_b.bias")):
                rows = np.frombuffer(raw, dtype=np.uint8).reshape(shape[0], -1)[experts]
                shape, raw = (len(experts), *shape[1:]), rows.tobytes()
            written.append((name.replace(".engram_wkv.", ".engram_kv."), shape, info.type_id,
                            raw))
        for layer in ENGRAM_LAYERS:
            g = f"blk.{layer}."
            words = [np.frombuffer(tensors[g + f"engram_{w}.weight"][1], dtype="<u2")
                     for w in "qk"]
            q, k = ((w.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
                    for w in words)
            written.append((g + "engram_gate.weight", (HC, HIDDEN), TYPE_F32,
                            (q * k).astype("<f4").tobytes()))
        for layer in range(3):
            row = bias[f"layers.{layer}.ffn.gate.bias_vl"][experts]
            written.append((f"blk.{layer}.exp_probs_b_vl.bias", (len(experts),), TYPE_F32,
                            row.astype("<f4").tobytes()))
        info = engram.info("blk.0.engram_embd.weight")
        table_rows = engram.tensor_bytes(info.name).tobytes()
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    shard = upstream / "ds-00001-of-00003.gguf"
    write_gguf(shard, kv, [])
    write_gguf(upstream / "ds-00002-of-00003.gguf", {"split.no": 1, "split.count": 3}, written)
    engram_file = upstream / "engram.gguf"
    write_gguf(engram_file, {"deepseek41.engram.source": "release", "split.no": 2,
                             "split.count": 3},
               [("per_layer_token_embd.weight", info.shape, TYPE_IQ4_NL, table_rows)])
    if link:
        (upstream / "ds-00003-of-00003.gguf").symlink_to(engram_file.name)
    return shard, engram_file


def _artifact_content(path):
    from tools.artifact.reader import Artifact

    with Artifact(path) as artifact:
        return (artifact.directory.to_json(),
                {obj.id: artifact.read_object(obj.id) for obj in artifact.directory.objects})


@pytest.mark.parametrize("link", [False, True])
def test_both_gguf_layouts_convert_to_the_same_artifact(tmp_path, link):
    hf, first, table, _ = _release(tmp_path)
    mmproj, _ = _mmproj(tmp_path)
    bias_path, bias = _bias_vl(tmp_path)
    components = ("text", "engram", "vision")
    earlier = _convert(tmp_path, hf, first, table, components, mmproj=mmproj, bias_vl=bias_path)
    shard, engram = _upstream(tmp_path, first, table, bias, link=link)
    with deepseek_v41.GGUFShards(shard) as gguf:
        assert gguf.upstream and len(gguf.files) == (3 if link else 2)
    upstream_mmproj, _ = _mmproj(tmp_path / "upstream", projector="deepseek41v")
    later = _convert(tmp_path / "upstream", hf, shard, engram, components, mmproj=upstream_mmproj)
    assert _artifact_content(later) == _artifact_content(earlier)
    # The table alone, from either Engram GGUF.
    table_only = [_convert(tmp_path / name, hf, None, source, ("engram",))
                  for name, source in (("upstream", engram), ("", table))]
    assert _artifact_content(table_only[0]) == _artifact_content(table_only[1])


def test_a_pruned_build_converts_with_its_own_expert_count(tmp_path):
    from tools.artifact.reader import Artifact

    hf, first, table, stored = _release(tmp_path)
    mmproj, _ = _mmproj(tmp_path)
    _, bias = _bias_vl(tmp_path)
    keep = [0, 3]
    shard, engram = _upstream(tmp_path, first, table, bias, keep=keep)
    upstream_mmproj, _ = _mmproj(tmp_path / "upstream", projector="deepseek41v")
    output = _convert(tmp_path / "upstream", hf, shard, engram, ("text", "engram", "vision"),
                      mmproj=upstream_mmproj)
    with Artifact(output) as artifact:
        directory = artifact.directory
        assert directory.components["text"]["config"]["n_routed_experts"] == len(keep)
        bindings = directory.bindings

        def stored_bytes(name):
            return artifact.read_object(bindings[name]["object"])

        for layer in range(3):
            g, p = f"blk.{layer}.", f"text/layers/{layer}/"
            gate, up = stored[g + "ffn_gate_exps.weight"], stored[g + "ffn_up_exps.weight"]
            expected = np.concatenate([np.concatenate([gate[e * WIDTH:(e + 1) * WIDTH],
                                                       up[e * WIDTH:(e + 1) * WIDTH]])
                                       for e in keep])
            assert stored_bytes(p + "moe/gate_up") == expected.tobytes()
            down = stored[g + "ffn_down_exps.weight"].reshape(EXPERTS, -1)[keep]
            assert stored_bytes(p + "moe/down") == down.tobytes()
            router = stored[g + "ffn_gate_inp.weight"].reshape(EXPERTS, -1)[keep]
            assert np.array_equal(np.frombuffer(stored_bytes(p + "moe/router"), np.float32),
                                  router.reshape(-1))
        image_bias = np.frombuffer(stored_bytes("vision/image_router_bias"), np.float32)
        assert np.array_equal(image_bias, np.concatenate(
            [bias[f"layers.{i}.ffn.gate.bias_vl"][keep] for i in range(3)]))


@pytest.mark.parametrize(("keep", "match"), [([1], "pruning"), ([0, 1, 2, 3, 0], "pruning")])
def test_an_expert_count_outside_the_release_is_refused(tmp_path, keep, match):
    from tools.convert.sources.safetensors import SafetensorsSource

    hf, first, table, _ = _release(tmp_path)
    _, bias = _bias_vl(tmp_path)
    shard, _ = _upstream(tmp_path, first, table, bias, keep=keep)
    with SafetensorsSource(hf) as base, deepseek_v41.GGUFShards(shard) as gguf:
        with pytest.raises(ValueError, match=match):
            deepseek_v41.build_model(base, gguf, components=("text",))


@pytest.mark.parametrize("bad", [
    {"deepseek41.ple.ngram_size": 4},
    {"deepseek41.attention.indexer.candidate_source_layer": 1},
    {"deepseek41.attention.kv_source_layers": [1]},
    {"deepseek41.engram.pad_token_id": 3},
    {"deepseek41.ple.head_vocab_sizes": [5, 7, 11, 14]},
    {"deepseek41.ple.head_offsets": [0, 5, 12, 24]},
    {"deepseek41.engram.token_map": [0] * (VOCAB - 1)},
])
def test_llama_cpp_metadata_must_match_the_config(tmp_path, bad):
    from tools.convert.sources.safetensors import SafetensorsSource

    hf, first, table, _ = _release(tmp_path)
    _, bias = _bias_vl(tmp_path)
    shard, _ = _upstream(tmp_path, first, table, bias, **bad)
    with SafetensorsSource(hf) as base, deepseek_v41.GGUFShards(shard) as gguf:
        with pytest.raises(ValueError):
            deepseek_v41.build_model(base, gguf, components=("text",))


def test_llama_cpp_engram_offsets_address_the_whole_table():
    class Shards:
        path, upstream = "upstream.gguf", True
        kv = {"deepseek41.engram.token_map": [0, 1, 2, 3],
              "deepseek41.ple.layer_multipliers": [1, 3, 5, 7, 9, 11],
              "deepseek41.ple.head_vocab_sizes": [5, 7, 11, 13, 3, 5, 7, 11],
              "deepseek41.ple.head_offsets": [0, 5, 12, 23, 36, 39, 44, 51]}

    config = {"engram_layer_ids": [1, 14], "engram_max_ngram_size": 3, "engram_n_heads": 2,
              "vocab_size": 4, "engram_compressed_vocab_size": 4, "engram_pad_token_id": 2,
              "engram_num_embeddings": [36, 26]}
    constants = deepseek_v41.engram_constants(Shards(), config)
    assert constants["primes"] == [[5, 7, 11, 13], [3, 5, 7, 11]]
    assert constants["offsets"] == [[0, 5, 12, 23], [0, 3, 8, 15]]
    with pytest.raises(ValueError, match="bucket counts"):
        deepseek_v41.engram_constants(Shards(), {**config, "engram_num_embeddings": [36, 27]})


def test_the_image_routing_bias_source_follows_the_layout(tmp_path):
    from tools.convert.sources.safetensors import SafetensorsSource

    hf, first, table, _ = _release(tmp_path)
    mmproj, _ = _mmproj(tmp_path)
    bias_path, bias = _bias_vl(tmp_path)
    shard, _ = _upstream(tmp_path, first, table, bias)
    with SafetensorsSource(hf) as base, deepseek_v41.GGUFShards(shard) as gguf, \
            GGUFFile(mmproj) as vision, SafetensorsSource(bias_path) as bias_vl:
        with pytest.raises(ValueError, match="carries it"):
            deepseek_v41.build_model(base, gguf, components=("text", "vision"), vision=vision,
                                     bias_vl=bias_vl)
