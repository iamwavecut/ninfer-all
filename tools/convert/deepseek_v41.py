"""DeepSeek-V4.1-Flash (DeepseekV41ForCausalLM) from a split llama.cpp GGUF, kept in its blocks.

DeepSeek-V4.1-Flash is a 40-block MoE transformer whose residual stream is four parallel copies
mixed by hyper-connections (Sinkhorn-normalised), with one latent attention per block: 64 query
heads of 512 values over one shared key/value latent, a sliding window of 128 positions, and in
the blocks with a compression ratio the compressed latents that a two-level indexer selects; a
routed MoE (sqrt-softplus scores, top 6 of 384) beside one shared expert, both with clamped
SwiGLU; and two Engram blocks that add an n-gram table's rows, gated by the stream. The recipe
imports the GGUF's block matrices row for row, as stored; the routed experts' gate and up banks
are interleaved per expert ([gate rows; up rows] of expert 0, then expert 1, ...), which moves
whole rows only. BF16 and F32 tensors (router, compressor, indexer, hyper-connection matrices,
norms, sinks) become FP32 vectors, exactly; the Engram gate is the FP32 product of its q and k
weights (exact for BF16 factors).

The GGUFs come in two layouts, told apart by their metadata, and both convert to the same
artifact. llama.cpp's own (b11870 and newer; ISTA-DASLab's files since 2026-10-27) names the
Engram hash in metadata (`deepseek41.ple.*`, `deepseek41.engram.token_map`, row offsets into the
whole table), stores the gate as its product (`engram_gate`) and the image routing bias
(`exp_probs_b_vl`), and makes the Engram GGUF the split's last shard. The earlier layout of
ISTA-DASLab's llama.cpp branch keeps the hash in integer tensors, the gate as `engram_q` and
`engram_k`, and no image routing bias. A build with fewer routed experts than the release (the
Coder keeps 192 of 384 per block) converts as the release's model with its GGUF's expert count.

The Engram tables (`--source engram=DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`) form the `engram`
component: one table of IQ4_NL rows, the first Engram block's rows then the second's, which the
model names by the SHA-256 of its bytes. Both layouts hold the same bytes: llama.cpp's as one
`per_layer_token_embd` tensor, the earlier one as a tensor per Engram block. `--components
text,engram` stores the model with its table, `text` the model alone (its table read from a table
artifact at run time) and `engram` that table artifact. `--resource engram.hot=PROFILE` stores a
hot-row profile (ninfer-ngram-profile's NFNGHOT1 file) beside the table, as
`engram.resources.hot_profile`, after checking it against the Engram hash of `--source gguf`, which
a table alone then also needs.

The release's DSpark drafter (`--source dspark=dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf`, llama.cpp
architecture `dflash`) forms the `dspark` component beside `text` (`--components text,engram,dspark`):
three blocks of the backbone's shape whose window rings the trunk's streams entering its last three
blocks fill (`fc`, its norm), a routed MoE of 128 experts (3 a token) in MXFP4 blocks, the final
norm, the Markov bias tables (BF16) and the confidence head, imported as stored like the backbone.
The drafter of beamster's Sushi STQ pack (`--source dspark=PACK_DIR`, a directory with the pack's
config.json, its index and its six `model-mtp*` files) forms the same component from the release's
`mtp.0..2` tensors in STQ: each matrix its trellis words and sign vectors as stored
(STQ_CODEBOOKS), its vectors as the release's.

The release's vision tower forms the `vision` component beside `text` (`--components
text,engram,vision`): the ViT and the aligner from llama.cpp's `clip` file
(`--source vision=mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf`, projector `deepseek4v`, the release's
BF16 weights bit for bit), its matrices BF16, its norms, biases and the three image-span rows
(start, row end, end) FP32, and the routing bias of image positions, one FP32 row of the router's
width per block: the GGUF's `blk.N.exp_probs_b_vl.bias` in llama.cpp's layout; for an earlier-layout
GGUF the release's `layers.N.ffn.gate.bias_vl` (`--source bias_vl=FILE`, a safetensors file that
tools/convert/fetch_tensors.py reads out of the release by byte range).

    python -m tools.convert --model HF_DIR --recipe deepseek_v41_gguf \\
        --source gguf=DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00006.gguf \\
        --source engram=DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf \\
        --source dspark=dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf \\
        --source vision=mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf \\
        --components text,engram,dspark,vision --out deepseek-v4.1-flash.ninfer

`HF_DIR` holds the base release's `config.json`, `tokenizer.json` and `tokenizer_config.json`
(no weights are read from it; its vision_config and image_token_id describe the tower). The chat template and the generation defaults come from
tools/convert/templates/deepseek_v41 unless HF_DIR or `--resource` provides them.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np
import torch

from tools.artifact.formats import GGUF_FORMATS_BY_TYPE

from .gguf_blocks import block_source
from .methods import cast_direct, import_encoded
from .resources import _dump_json, _normalize_added_decoder, token_domain
from .sources.gguf import GGUFFile
from .sources.logical import EncodedRows, LogicalSource, array_source, select_rows
from .sources.safetensors import SafetensorsSource, tensor_source
from .ternary import rows

ARCHITECTURES = ("DeepseekV41ForCausalLM",)
TABLE_ARCHITECTURE = "DeepseekV41EngramTable"
DSPARK_ARCHITECTURE = "DeepseekV41DSpark"
VISION_ARCHITECTURE = "DeepseekV41VisionModel"
# llama.cpp's projector name in ISTA-DASLab's mmproj, and in Unsloth's (the same tensors).
VISION_PROJECTORS = ("deepseek4v", "deepseek41v")
BIAS_VL = "layers.{}.ffn.gate.bias_vl"
# The drafter's Markov tables, kept in BF16 as stored.
MARKOV_TABLES = ("dspark/markov_embed", "dspark/markov_head")
RESOURCES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
             "generation_config.json")
DEFAULT_RESOURCES = Path(__file__).resolve().parent / "templates" / "deepseek_v41"
# llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS.
SQRT_SOFTPLUS_GATING = 4
# The codebooks of an STQ pack's drafter (beamster's Sushi STQ packs, `--source dspark=PACK_DIR`):
# each matrix's trellis words (u16 [in/16][out/16][words], 16 x 16 weights a tile) as stored, its
# input and output sign vectors (F16) beside them; the trunk's tiles SPEC1 with 15-bit codewords
# at 64 words (4 bits a weight), the routed experts' PAIR1 with 16-bit codewords at 32 words.
STQ_CODEBOOKS = {"trunk": "spec1_w15_n64", "experts": "pair1_w16_n32"}
STQ_WORDS = {"spec1_w15_n64": 64, "pair1_w16_n32": 32}
# Model keys an STQ pack's config must share with the release beside its dspark_* keys.
STQ_PACK_KEYS = ("vocab_size", "hidden_size", "moe_intermediate_size", "num_hidden_layers",
                 "num_attention_heads", "head_dim", "q_lora_rank", "o_lora_rank", "o_groups",
                 "num_nextn_predict_layers", "swiglu_limit", "rms_norm_eps", "sliding_window",
                 "hc_mult", "hc_eps", "hc_sinkhorn_iters", "routed_scaling_factor")
TABLE_FORMAT = "gguf_iq4_nl"
# The Engram rows: llama.cpp's one tensor of every Engram block's rows, block after block, or the
# earlier layout's tensor per Engram block.
TABLE_TENSOR = "per_layer_token_embd.weight"
ENGRAM_TENSOR = "blk.{}.engram_embd.weight"
SPLIT_NAME = re.compile(r"(.*)-(\d{5})-of-(\d{5})\.gguf")
# Metadata of the earlier layout under llama.cpp's names (ISTA-DASLab's convert_ds41_upstream.py).
UPSTREAM_KEYS = {
    "attention.candidate_block_size": "attention.indexer.candidate_block_size",
    "attention.candidate_topk_blocks": "attention.indexer.candidate_top_k_blocks",
    "attention.candidate_source_layer_id": "attention.indexer.candidate_source_layer",
    "kv_source_layer_ids": "attention.kv_source_layers",
    "index_source_layer_ids": "attention.indexer.source_layers",
    "engram.layer_ids": "ple.layers",
    "engram_head_dim": "embedding_length_per_layer_input",
    "engram_n_heads": "ple.heads_per_ngram",
    "engram_max_ngram_size": "ple.ngram_size",
    "engram_pad_token_id": "engram.pad_token_id",
}


def upstream_layout(kv: dict) -> bool:
    """Whether DeepSeek-V4.1 GGUF metadata is in llama.cpp's layout (b11870 and newer), not the
    earlier layout of ISTA-DASLab's llama.cpp branch."""

    return "deepseek41.attention.kv_source_layers" in kv


class GGUFShards:
    """The files of one split GGUF (`NAME-0000k-of-0000n.gguf`) as one set of tensors; the first
    file's metadata describes the model.

    In llama.cpp's DeepSeek-V4.1 layout (`upstream`) the last shard is the shared Engram GGUF,
    which a download links under the shard's name; without the link the set opens without it, and
    `--source engram` names the file."""

    def __init__(self, first: str | Path) -> None:
        first = Path(first)
        match = SPLIT_NAME.fullmatch(first.name)
        if match is None:
            paths = [first]
        else:
            stem, count = match[1], int(match[3])
            if int(match[2]) != 1:
                raise ValueError(f"{first}: name the first shard of the split GGUF")
            paths = [first.with_name(f"{stem}-{i:05d}-of-{count:05d}.gguf")
                     for i in range(1, count + 1)]
        self.files = [GGUFFile(paths[0])]
        self.kv = self.files[0].kv
        self.upstream = upstream_layout(self.kv)
        for path in paths[1:]:
            if path == paths[-1] and self.upstream and not path.exists():
                continue
            self.files.append(GGUFFile(path))
        if int(self.kv.get("split.count", 1)) != len(paths):
            raise ValueError(f"{first}: split.count is {self.kv.get('split.count')}, "
                             f"{len(paths)} files named")
        self.path = first
        self.tensors = {}
        self._owner = {}
        for file in self.files:
            for name, info in file.tensors.items():
                if name in self._owner:
                    raise ValueError(f"{file.path}: tensor {name} repeats another shard's")
                self._owner[name] = file
                self.tensors[name] = info

    def __enter__(self) -> GGUFShards:
        return self

    def __exit__(self, *_exc) -> None:
        for file in self.files:
            file.__exit__()

    def file(self, name: str) -> GGUFFile:
        try:
            return self._owner[name]
        except KeyError as error:
            raise KeyError(f"{self.path}: tensor {name} is absent") from error

    def info(self, name: str):
        return self.file(name).info(name)

    def integers(self, name: str, dtype) -> np.ndarray:
        """An integer tensor (I32/I64) in its row-major shape."""
        info = self.info(name)
        if np.dtype(dtype).itemsize != {"I32": 4, "I64": 8}.get(info.type_name):
            raise ValueError(f"{name}: expected {np.dtype(dtype)} values, found {info.type_name}")
        raw = np.asarray(self.file(name).tensor_bytes(name))
        return np.frombuffer(raw.tobytes(), dtype=dtype).reshape(info.shape)


def _int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name}: expected a positive integer")
    return value


def text_config(source: dict) -> dict:
    """DeepSeek-V4.1-Flash's text config, normalised for src/models/deepseek_v41."""

    if source.get("architectures") != list(ARCHITECTURES):
        raise ValueError(f"unsupported DeepSeek architecture {source.get('architectures')!r}")
    t = source.get("text_config")
    if not isinstance(t, dict) or t.get("model_type") != "deepseek_v41_text":
        raise ValueError("expected DeepSeek-V4.1's text_config")
    if t.get("hidden_act") != "silu" or t.get("attention_bias"):
        raise ValueError("DeepSeek-V4.1 uses SwiGLU and no attention biases")
    if (t.get("scoring_func") != "sqrtsoftplus" or t.get("topk_method") != "noaux_tc"
            or t.get("norm_topk_prob") is not True or t.get("n_shared_experts") != 1):
        raise ValueError("DeepSeek-V4.1 routes by sqrt-softplus scores with a selection bias, "
                         "renormalised, beside one shared expert")
    if t.get("tie_word_embeddings") is not False:
        raise ValueError("DeepSeek-V4.1's output head is untied from its token table")
    rope = t.get("rope_scaling") or {}
    if rope.get("rope_type") != "yarn":
        raise ValueError("DeepSeek-V4.1's compressed blocks rotate with YaRN")
    out = {"architectures": list(ARCHITECTURES), "model_type": "deepseek_v41"}
    for key in ("hidden_size", "vocab_size", "num_hidden_layers", "max_position_embeddings",
                "num_attention_heads", "head_dim", "qk_rope_head_dim", "q_lora_rank",
                "o_lora_rank", "o_groups", "sliding_window", "moe_intermediate_size",
                "n_routed_experts", "num_experts_per_tok", "index_n_heads", "index_head_dim",
                "index_topk", "candidate_topk_blocks", "candidate_block_size", "hc_mult",
                "hc_sinkhorn_iters", "engram_max_ngram_size", "engram_n_heads",
                "engram_head_dim"):
        out[key] = _int(t.get(key), key)
    if t.get("num_key_value_heads") != 1:
        raise ValueError("DeepSeek-V4.1's attention shares one key/value latent")
    layers = out["num_hidden_layers"]
    ratios = t.get("compress_ratios")
    if not isinstance(ratios, list) or len(ratios) < layers:
        raise ValueError("compress_ratios must name every block")
    # Entries past the backbone's blocks belong to the DSpark drafter (dspark_config).
    out["compress_ratios"] = [int(r) for r in ratios[:layers]]
    if any(r not in (0, 1, 2) for r in out["compress_ratios"]):
        raise ValueError("compression ratios other than 0, 1 and 2 are not implemented")
    for key in ("kv_source_layer_ids", "index_source_layer_ids", "engram_layer_ids",
                "engram_num_embeddings"):
        value = t.get(key)
        if not isinstance(value, list) or any(type(v) is not int or v < 0 for v in value):
            raise ValueError(f"{key} must be a list of non-negative integers")
        out[key] = list(value)
    out["candidate_source_layer_id"] = int(t.get("candidate_source_layer_id", -1))
    out["engram_pad_token_id"] = int(t["engram_pad_token_id"])
    out["engram_compressed_vocab_size"] = _int(t.get("engram_compressed_vocab_size"),
                                               "engram_compressed_vocab_size")
    for key in ("rms_norm_eps", "hc_eps", "swiglu_limit", "routed_scaling_factor",
                "rope_theta", "compress_rope_theta"):
        out[key] = float(t[key])
    out["rope_scaling"] = {"factor": float(rope["factor"]),
                           "beta_fast": float(rope["beta_fast"]),
                           "beta_slow": float(rope["beta_slow"]),
                           "original_max_position_embeddings":
                               _int(rope["original_max_position_embeddings"],
                                    "original_max_position_embeddings")}
    out["tie_word_embeddings"] = False
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if key in source:
            out[key] = source[key]
    return out


def with_gguf_expert_count(config: dict, gguf: GGUFShards) -> dict:
    """The text config with the GGUF's routed experts per block: an expert-pruned build (the Coder
    keeps 192 of the release's 384) is the release's model with fewer experts."""

    count = gguf.kv.get("deepseek41.expert_count")
    if type(count) is not int or count == config["n_routed_experts"]:
        return config
    if not config["num_experts_per_tok"] <= count < config["n_routed_experts"]:
        raise ValueError(f"{gguf.path}: {count} routed experts per block cannot be a pruning of "
                         f"the release's {config['n_routed_experts']}")
    return {**config, "n_routed_experts": count}


def validate(gguf: GGUFShards, config: dict) -> None:
    """The GGUF's architecture metadata against the release's config.json, under the names of the
    GGUF's layout."""

    kv = gguf.kv
    if kv.get("general.architecture") != "deepseek41":
        raise ValueError(f"{gguf.path}: expected a DeepSeek-V4.1 (deepseek41) GGUF")
    f = "deepseek41."
    if gguf.upstream:
        def key_name(key):
            return f + UPSTREAM_KEYS.get(key, key)
    else:
        def key_name(key):
            return f + key
    layers = config["num_hidden_layers"]
    expected = {
        "block_count": layers,
        "embedding_length": config["hidden_size"],
        "attention.head_count": config["num_attention_heads"],
        "attention.head_count_kv": 1,
        "attention.key_length": config["head_dim"],
        "attention.value_length": config["head_dim"],
        "rope.dimension_count": config["qk_rope_head_dim"],
        "attention.q_lora_rank": config["q_lora_rank"],
        "attention.output_group_count": config["o_groups"],
        "attention.output_lora_rank": config["o_lora_rank"],
        "attention.sliding_window": config["sliding_window"],
        "expert_count": config["n_routed_experts"],
        "expert_used_count": config["num_experts_per_tok"],
        "expert_feed_forward_length": config["moe_intermediate_size"],
        "expert_shared_count": 1,
        "expert_gating_func": SQRT_SOFTPLUS_GATING,
        "expert_weights_norm": True,
        "attention.indexer.head_count": config["index_n_heads"],
        "attention.indexer.key_length": config["index_head_dim"],
        "attention.indexer.top_k": config["index_topk"],
        "attention.candidate_block_size": config["candidate_block_size"],
        "attention.candidate_topk_blocks": config["candidate_topk_blocks"],
        "attention.candidate_source_layer_id": config["candidate_source_layer_id"],
        "hyper_connection.count": config["hc_mult"],
        "hyper_connection.sinkhorn_iterations": config["hc_sinkhorn_iters"],
        "engram_head_dim": config["engram_head_dim"],
        "engram_n_heads": config["engram_n_heads"],
        "engram_max_ngram_size": config["engram_max_ngram_size"],
        "engram_pad_token_id": config["engram_pad_token_id"],
        "rope.scaling.original_context_length":
            config["rope_scaling"]["original_max_position_embeddings"],
    }
    for key, value in expected.items():
        name = key_name(key)
        if kv.get(name) != value:
            raise ValueError(f"{gguf.path}: {name} is {kv.get(name)!r}, the config's {value!r}")
    lists = {
        "attention.compress_ratios": config["compress_ratios"],
        "kv_source_layer_ids": config["kv_source_layer_ids"],
        "index_source_layer_ids": config["index_source_layer_ids"],
        "engram.layer_ids": config["engram_layer_ids"],
    }
    # llama.cpp's layout states each Engram block's rows as its columns' bucket counts
    # (engram_constants).
    if not gguf.upstream:
        lists["engram.num_embeddings"] = config["engram_num_embeddings"]
    for key, value in lists.items():
        name = key_name(key)
        stored = list(kv.get(name, []))
        if stored[:len(value)] != value:
            raise ValueError(f"{gguf.path}: {name} differs from the config")
    reals = {
        "expert_weights_scale": config["routed_scaling_factor"],
        "rope.freq_base": config["rope_theta"],
        "attention.compress_rope_freq_base": config["compress_rope_theta"],
        "rope.scaling.factor": config["rope_scaling"]["factor"],
        "rope.scaling.yarn_beta_fast": config["rope_scaling"]["beta_fast"],
        "rope.scaling.yarn_beta_slow": config["rope_scaling"]["beta_slow"],
        "attention.layer_norm_rms_epsilon": config["rms_norm_eps"],
        "hyper_connection.epsilon": config["hc_eps"],
    }
    for key, value in reals.items():
        if not np.isclose(float(kv.get(f + key, np.nan)), value, rtol=1e-6, atol=0.0):
            raise ValueError(f"{gguf.path}: {f + key} differs from the config's {value!r}")
    # llama.cpp's layout states the scores by expert_gating_func alone.
    if not gguf.upstream and kv.get(f + "expert_scoring_func") != "sqrtsoftplus":
        raise ValueError(f"{gguf.path}: expected sqrt-softplus expert scores")
    for key in ("swiglu_clamp_exp", "swiglu_clamp_shexp"):
        if any(float(v) != config["swiglu_limit"] for v in kv.get(f + key, [])[:layers]):
            raise ValueError(f"{gguf.path}: {key} differs from swiglu_limit")


def _dspark_release_config(source: dict, config: dict) -> dict:
    """The `dspark` component's config from the release's dspark_* keys: blocks of the backbone's
    geometry (config), window attention only, and a MoE of their own width."""

    t = source["text_config"]
    layers, vocab = config["num_hidden_layers"], config["vocab_size"]
    count = _int(t.get("num_nextn_predict_layers"), "num_nextn_predict_layers")
    out = {"architectures": [DSPARK_ARCHITECTURE], "model_type": "deepseek_v41_dspark",
           "num_hidden_layers": count}
    for key, name in (("dspark_block_size", "block_size"), ("dspark_markov_rank", "markov_rank"),
                      ("dspark_n_routed_experts", "n_routed_experts"),
                      ("dspark_num_experts_per_tok", "num_experts_per_tok")):
        out[name] = _int(t.get(key), key)
    noise = t.get("dspark_noise_token_id")
    if type(noise) is not int or not 0 <= noise < vocab:
        raise ValueError("dspark_noise_token_id must be a token id")
    out["noise_token_id"] = noise
    targets = t.get("dspark_target_layer_ids")
    if (not isinstance(targets, list) or not targets
            or any(type(v) is not int or not 0 <= v < layers for v in targets)
            or targets != sorted(set(targets))):
        raise ValueError("dspark_target_layer_ids must name backbone blocks in order")
    out["target_layer_ids"] = list(targets)
    if out["num_experts_per_tok"] > out["n_routed_experts"]:
        raise ValueError("the drafter routes to more experts than it has")
    if list(t.get("compress_ratios", []))[layers:] != [0] * count:
        raise ValueError("the DSpark blocks attend through their window only (compress ratio 0)")
    return out


def dspark_config(source: dict, config: dict, dspark: GGUFShards) -> dict:
    """The release's drafter config (_dspark_release_config), checked against the drafter GGUF's
    metadata."""

    out = _dspark_release_config(source, config)
    count = out["num_hidden_layers"]
    kv, f = dspark.kv, "dflash."
    if kv.get("general.architecture") != "dflash":
        raise ValueError(f"{dspark.path}: expected the release's DSpark drafter (dflash) GGUF")
    expected = {
        "block_count": count, "embedding_length": config["hidden_size"],
        "attention.head_count": config["num_attention_heads"], "attention.head_count_kv": 1,
        "attention.key_length": config["head_dim"], "attention.value_length": config["head_dim"],
        "rope.dimension_count": config["qk_rope_head_dim"],
        "attention.q_lora_rank": config["q_lora_rank"],
        "attention.output_group_count": config["o_groups"],
        "attention.output_lora_rank": config["o_lora_rank"],
        "attention.sliding_window": config["sliding_window"],
        "expert_count": out["n_routed_experts"], "expert_used_count": out["num_experts_per_tok"],
        "expert_feed_forward_length": config["moe_intermediate_size"], "expert_shared_count": 1,
        "expert_gating_func": SQRT_SOFTPLUS_GATING, "expert_weights_norm": True,
        "hyper_connection.count": config["hc_mult"],
        "hyper_connection.sinkhorn_iterations": config["hc_sinkhorn_iters"],
        "hyper_connection.single_pass": True, "block_size": out["block_size"],
        "sample_from_anchor": True, "has_confidence_head": True,
    }
    for key, value in expected.items():
        if kv.get(f + key) != value:
            raise ValueError(f"{dspark.path}: {f + key} is {kv.get(f + key)!r}, expected "
                             f"{value!r}")
    if (list(kv.get(f + "target_layers", [])) != out["target_layer_ids"]
            or list(kv.get(f + "attention.compress_ratios", [])) != [0] * count):
        raise ValueError(f"{dspark.path}: the target layers or compress ratios differ from the "
                         "config's")
    reals = {"expert_weights_scale": config["routed_scaling_factor"],
             "rope.freq_base": config["rope_theta"],
             "attention.layer_norm_rms_epsilon": config["rms_norm_eps"],
             "hyper_connection.epsilon": config["hc_eps"]}
    for key, value in reals.items():
        if not np.isclose(float(kv.get(f + key, np.nan)), value, rtol=1e-6, atol=0.0):
            raise ValueError(f"{dspark.path}: {f + key} differs from the config's {value!r}")
    for key in ("swiglu_clamp_exp", "swiglu_clamp_shexp"):
        values = list(kv.get(f + key, []))
        if len(values) < count or any(float(v) != config["swiglu_limit"] for v in values[:count]):
            raise ValueError(f"{dspark.path}: {key} differs from swiglu_limit")
    return out


def _safetensors_metadata(file: Path) -> dict:
    with file.open("rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        return json.loads(stream.read(length)).get("__metadata__", {})


def dspark_stq_config(source: dict, config: dict, pack: SafetensorsSource) -> dict:
    """The release's drafter config (_dspark_release_config) for the drafter of an STQ pack
    (STQ_PACK), whose own config must state the release's model and drafter keys and whose drafter
    files must hold the codebooks this recipe stores: SPEC1 (`k` 4) beside the trunk's matrices,
    PAIR1 (`k` 2) beside the routed experts'."""

    out = _dspark_release_config(source, config)
    release, stated = source["text_config"], pack.config.get("text_config", {})
    for key, value in release.items():
        if key.startswith("dspark_") or key in STQ_PACK_KEYS:
            if stated.get(key) != value:
                raise ValueError(f"{pack.path}: text_config.{key} differs from the release's")
    files = sorted({pack.weight_map[name] for name in pack.weight_map if name.startswith("mtp.")})
    if not files:
        raise ValueError(f"{pack.path}: the pack holds no drafter tensors (mtp.*)")
    for file in files:
        meta = _safetensors_metadata(file)
        experts = any(owner == file and ".switch_mlp." in name
                      for name, owner in pack.weight_map.items())
        want = ("pair1", "2") if experts else ("spec1", "4")
        if (meta.get("format"), meta.get("codebook"), meta.get("k")) != ("stq", *want):
            raise ValueError(f"{file}: expected STQ {want[0]} at k={want[1]}, found {meta}")
    out["stq"] = dict(STQ_CODEBOOKS)
    return out


def vision_config(mmproj: GGUFFile, source: dict, text: dict) -> dict:
    """The `vision` component's config from the release's vision_config and image_token_id,
    checked against the mmproj's metadata and the text model's width."""

    kv = mmproj.kv
    if kv.get("general.architecture") != "clip" or not kv.get("clip.has_vision_encoder"):
        raise ValueError(f"{mmproj.path}: expected a llama.cpp clip GGUF with a vision encoder")
    if kv.get("clip.projector_type") not in VISION_PROJECTORS:
        raise ValueError(f"{mmproj.path}: expected DeepSeek-V4.1's projector (deepseek4v), found "
                         f"{kv.get('clip.projector_type')!r}")
    raw = source.get("vision_config")
    if not isinstance(raw, dict) or raw.get("model_type") != "deepseek_v41_vision":
        raise ValueError("config.json has no deepseek_v41_vision vision_config")
    out = {"architectures": [VISION_ARCHITECTURE], "model_type": "deepseek_v41_vision"}
    for key in ("num_hidden_layers", "hidden_size", "num_attention_heads", "intermediate_size",
                "patch_size", "downsample_ratio", "max_image_tokens", "min_pixels"):
        out[key] = _int(raw.get(key), f"vision_config.{key}")
    out["rope_theta"] = float(raw["rope_theta"])
    ratio = raw.get("max_wh_ratio")
    if ratio is not None and (type(ratio) is not int or ratio <= 0):
        raise ValueError("vision_config.max_wh_ratio must be a positive integer or null")
    out["max_wh_ratio"] = ratio
    token = source.get("image_token_id")
    if type(token) is not int or not 0 <= token < text["vocab_size"]:
        raise ValueError("config.json's image_token_id must be a token id")
    out["image_token_id"] = token
    out["projection_dim"] = text["hidden_size"]
    # vision.py's RMSNorm keeps its default epsilon.
    out["rms_norm_eps"] = 1e-6
    v = "clip.vision."
    expected = {"block_count": out["num_hidden_layers"], "embedding_length": out["hidden_size"],
                "attention.head_count": out["num_attention_heads"],
                "feed_forward_length": out["intermediate_size"], "patch_size": out["patch_size"],
                "projection_dim": out["projection_dim"],
                "projector.scale_factor": out["downsample_ratio"],
                "image_min_pixels": out["min_pixels"]}
    for key, value in expected.items():
        if kv.get(v + key) != value:
            raise ValueError(f"{mmproj.path}: {v + key} is {kv.get(v + key)!r}, the config's "
                             f"{value!r}")
    if (list(kv.get(v + "image_mean", [])) != [0.5] * 3
            or list(kv.get(v + "image_std", [])) != [0.5] * 3 or not kv.get("clip.use_silu")):
        raise ValueError(f"{mmproj.path}: expected pixels scaled to [-1, 1] and a SiLU MLP")
    if not np.isclose(float(kv.get(v + "attention.layer_norm_epsilon", np.nan)),
                      out["rms_norm_eps"], rtol=1e-6, atol=0.0):
        raise ValueError(f"{mmproj.path}: the tower's norm epsilon differs from 1e-6")
    heads = out["num_attention_heads"]
    if out["hidden_size"] % heads or (out["hidden_size"] // heads) % 4:
        raise ValueError("vision heads must split the width into heads of 2D-rotated pairs")
    return out


def processor_config(vision: dict) -> dict:
    """The image preprocessing of the release's inference/image_processor.py as the frontend reads
    it: the patch grid, the token budget and its minimum pixel count, the bicubic resize onto gray
    127 (centered, aspect kept) or, past max_wh_ratio, a plain bicubic resize, and the values
    (x / 255 - 0.5) / 0.5."""

    return {"image_processor": {
        "patch_size": vision["patch_size"], "downsample_ratio": vision["downsample_ratio"],
        "min_pixels": vision["min_pixels"], "max_image_tokens": vision["max_image_tokens"],
        "max_wh_ratio": vision["max_wh_ratio"], "resample": 3, "pad_color": [127, 127, 127],
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "image_token": "<｜deepseek_image｜>"}}


def engram_constants(gguf: GGUFShards, config: dict) -> dict:
    """The Engram hash of the release: the compressed id of every token, a multiplier per (Engram
    block, look-back) and a prime and a row offset into the block's table per (Engram block,
    column), column (n - 2) * heads + head for the n-gram of n tokens. llama.cpp's layout states
    them in metadata, its offsets into the whole table; the earlier layout in integer tensors."""

    layers = len(config["engram_layer_ids"])
    n, heads = config["engram_max_ngram_size"], config["engram_n_heads"]
    columns = (n - 1) * heads
    if gguf.upstream:
        def values(key, dtype, shape):
            stored = gguf.kv.get("deepseek41." + key)
            if not isinstance(stored, list) or len(stored) != int(np.prod(shape)):
                raise ValueError(f"{gguf.path}: deepseek41.{key} must hold "
                                 f"{int(np.prod(shape))} integers")
            return np.array(stored, dtype=dtype).reshape(shape)

        token_map = values("engram.token_map", "<i4", (config["vocab_size"],))
        multipliers = values("ple.layer_multipliers", "<i8", (layers, n))
        primes = values("ple.head_vocab_sizes", "<i8", (layers, columns))
        # A block's table is its columns' buckets; its rows follow the blocks before it.
        if primes.sum(axis=1).tolist() != list(config["engram_num_embeddings"]):
            raise ValueError(f"{gguf.path}: the Engram bucket counts differ from the config's "
                             "table rows")
        starts = np.cumsum(primes.sum(axis=1)) - primes.sum(axis=1)
        offsets = values("ple.head_offsets", "<i8", (layers, columns)) - starts[:, None]
    else:
        token_map = gguf.integers("engram_token_map.weight", "<i4")
        multipliers = gguf.integers("engram_multipliers.weight", "<i8")
        primes = gguf.integers("engram_primes.weight", "<i8")
        offsets = gguf.integers("engram_offsets.weight", "<i8")
    if token_map.shape != (config["vocab_size"],):
        raise ValueError("engram_token_map must map every token")
    if (multipliers.shape != (layers, n) or primes.shape != (layers, columns)
            or offsets.shape != (layers, columns)):
        raise ValueError("the Engram hash tensors do not match the Engram blocks")
    if (token_map.min() < 0 or token_map.max() >= config["engram_compressed_vocab_size"]
            or (multipliers <= 0).any() or (primes <= 0).any() or (offsets < 0).any()):
        raise ValueError("the Engram hash tensors hold out-of-range values")
    # Every product of a compressed id and a multiplier fits an int64, as training hashed them.
    if int(multipliers.max()) > np.iinfo(np.int64).max // config["engram_compressed_vocab_size"]:
        raise ValueError("an Engram multiplier would overflow the hash")
    for layer, rows_in_layer in enumerate(config["engram_num_embeddings"]):
        if int((offsets[layer] + primes[layer]).max()) > rows_in_layer:
            raise ValueError("an Engram hash column addresses rows past its table")
    return {"token_map": token_map.astype("<i4"), "multipliers": multipliers.tolist(),
            "primes": primes.tolist(), "offsets": offsets.tolist(),
            "pad_id": int(token_map[config["engram_pad_token_id"]])}


def engram_hash_fingerprint(config: dict, constants: dict) -> int:
    """The fingerprint a hot-row profile of the Engram table records (engram_hash_fingerprint in
    models/deepseek_v41/engram.h): FNV-1a 64 over the little-endian u64 values ngram_size, heads,
    pad_id, the block count and the blocks, then each block's rows, multipliers, primes and
    offsets, then the token map's size and compressed ids."""

    layers = list(config["engram_layer_ids"])
    values = [config["engram_max_ngram_size"], config["engram_n_heads"], constants["pad_id"],
              len(layers), *layers]
    for index, rows_in_layer in enumerate(config["engram_num_embeddings"]):
        values += [rows_in_layer, *constants["multipliers"][index], *constants["primes"][index],
                   *constants["offsets"][index]]
    token_map = [int(v) for v in constants["token_map"]]
    values += [len(token_map), *token_map]
    value = 0xCBF29CE484222325
    for byte in struct.pack(f"<{len(values)}Q", *values):
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def attach_engram_profile(model, path: Path, config: dict, constants: dict) -> None:
    """Store a hot-row profile beside the Engram table after checking its hash and rows."""

    data = path.read_bytes()
    if len(data) < 40 or data[:8] != b"NFNGHOT1":
        raise ValueError(f"{path}: not an n-gram hot-row profile")
    table_rows, fingerprint, _tokens, count = struct.unpack_from("<4Q", data, 8)
    if len(data) - 40 != count * 4:
        raise ValueError(f"{path}: invalid hot-row profile row count")
    rows = model.components["engram"]["config"]["rows"]
    if table_rows != rows or fingerprint != engram_hash_fingerprint(config, constants):
        raise ValueError(f"{path}: profile was counted for another Engram hash")
    ranked = np.frombuffer(data, dtype="<u4", offset=40)
    if (len(ranked) and int(ranked.max()) >= rows) or len(np.unique(ranked)) != count:
        raise ValueError(f"{path}: duplicate or out-of-range profile rows")
    object_id = "resource/engram/hot_profile"
    model.resources[object_id] = data
    model.components["engram"].setdefault("resources", {})["hot_profile"] = object_id


def engram_table_config(config: dict) -> dict:
    """The `engram` component's config: one table of every Engram block's rows in order."""

    return {"architectures": [TABLE_ARCHITECTURE], "model_type": "deepseek_v41_engram",
            "layers": list(config["engram_layer_ids"]),
            "layer_rows": list(config["engram_num_embeddings"]),
            "rows": int(sum(config["engram_num_embeddings"])),
            "row_width": config["engram_head_dim"]}


def _load_resources(root: Path, vocab: int, overrides: dict):
    references, payloads = {}, {}
    for role in RESOURCES:
        if role in overrides:
            path = Path(overrides[role])
        elif (root / role).is_file():
            path = root / role
        else:
            path = DEFAULT_RESOURCES / role
        data = path.read_bytes()
        if not data:
            raise ValueError(f"{path}: resource is empty")
        object_id = f"resource/text/{role}"
        references[role] = object_id
        payloads[object_id] = data
    tokenizer = json.loads(payloads["resource/text/tokenizer.json"])
    tokenizer_config = json.loads(payloads["resource/text/tokenizer_config.json"])
    # The release's tokenizer_config.json names no added tokens; the runtime reads them there.
    if _normalize_added_decoder(tokenizer, tokenizer_config):
        payloads["resource/text/tokenizer_config.json"] = _dump_json(tokenizer_config)
    count, special = token_domain(tokenizer, tokenizer_config, vocab)
    return references, payloads, count, special


def _direct(gguf: GGUFShards, tensor: str, shape: tuple[int, ...]) -> LogicalSource:
    """F32 as stored, BF16 and F16 widened exactly."""

    info = gguf.info(tensor)
    if info.type_name not in ("F32", "BF16", "F16"):
        raise ValueError(f"{tensor}: expected a float vector or matrix, found {info.type_name}")
    values = gguf.file(tensor).read_direct(tensor).astype(np.float32).reshape(shape)
    return array_source(torch.from_numpy(np.ascontiguousarray(values)), tensor)


def _matrix(gguf: GGUFShards, tensor: str, shape: tuple[int, int]) -> LogicalSource:
    info = gguf.info(tensor)
    if info.type_id not in GGUF_FORMATS_BY_TYPE:
        raise ValueError(f"{tensor}: expected ggml block rows, found {info.type_name}")
    stored = (int(np.prod(info.shape[:-1])), info.shape[-1])
    if stored != shape:
        raise ValueError(f"{tensor}: stored {info.shape}, expected rows x columns {shape}")
    return block_source(gguf.file(tensor), tensor, shape, rows())


def _stacked(parts: list[LogicalSource], label: str) -> LogicalSource:
    """The rows of `parts` one after the other."""

    k = parts[0].shape[1]
    if any(len(p.shape) != 2 or p.shape[1] != k for p in parts):
        raise ValueError(f"{label}: stacked parts must be matrices of one width")
    starts = np.cumsum([0] + [p.shape[0] for p in parts]).tolist()

    def pieces(begin: int, end: int):
        for part, low, high in zip(parts, starts, starts[1:]):
            a, b = max(begin, low), min(end, high)
            if a < b:
                yield part, a - low, b - low

    def values(begin: int, end: int) -> torch.Tensor:
        first, last = begin // k, -(-end // k)
        flat = torch.cat([p.values(a * k, b * k) for p, a, b in pieces(first, last)])
        return flat[begin - first * k:end - first * k]

    def encoded(begin: int, end: int) -> EncodedRows:
        blocks = [p.read_encoded(a, b) for p, a, b in pieces(begin, end)]
        if len({b.format for b in blocks}) != 1:
            raise ValueError(f"{label}: stacked parts are stored in different formats")
        if len(blocks) == 1:
            return blocks[0]
        return EncodedRows(blocks[0].format, torch.cat([b.codes for b in blocks]),
                           torch.cat([b.scales for b in blocks]))

    return LogicalSource((starts[-1], k), label, values, encoded)


def _exact_bf16(gguf: GGUFFile, tensor: str, shape: tuple[int, ...]) -> LogicalSource:
    """`tensor` as FP32 values that BF16 holds exactly; a wider value is an error (llama.cpp writes
    the release's BF16 patch embedding, norms, biases and span rows as F32)."""

    info = gguf.info(tensor)
    if info.type_name not in ("BF16", "F32", "F16"):
        raise ValueError(f"{tensor}: expected BF16 (or F32/F16) values, found {info.type_name}")
    values = torch.from_numpy(
        np.ascontiguousarray(gguf.read_direct(tensor).astype(np.float32).reshape(shape)))
    if not torch.equal(values.to(torch.bfloat16).to(torch.float32), values):
        raise ValueError(f"{tensor}: values are not BF16 values")
    return array_source(values, tensor)


def image_router_bias(source, layers: int, experts: int) -> LogicalSource:
    """The routing bias of image positions, [block][expert] FP32, from the release's bias_vl
    tensors (a SafetensorsSource)."""

    rows = []
    for i in range(layers):
        name = BIAS_VL.format(i)
        if not source.has(name):
            raise ValueError(f"--source bias_vl lacks {name}")
        info = source.describe(name)
        if info.dtype != "F32" or info.shape != (experts,):
            raise ValueError(f"{name}: {info.dtype} {info.shape}, expected F32 ({experts},)")
        rows.append(source.read_flat(name))
    return array_source(torch.stack(rows).contiguous(), "bias_vl")

def expert_gate_up(gguf: GGUFShards, layer: int, experts: int, width: int,
                   hidden: int) -> LogicalSource:
    """`ffn_gate_exps` and `ffn_up_exps` of a block as one bank, each expert's gate rows followed by
    its up rows."""

    g = f"blk.{layer}."
    gate = _matrix(gguf, g + "ffn_gate_exps.weight", (experts * width, hidden))
    up = _matrix(gguf, g + "ffn_up_exps.weight", (experts * width, hidden))
    if gguf.info(g + "ffn_gate_exps.weight").type_id != gguf.info(g + "ffn_up_exps.weight").type_id:
        raise ValueError(f"{g}: expert gate and up banks are stored in different formats")
    both = _stacked([gate, up], f"{g}ffn_gate_exps+ffn_up_exps")
    up_base = experts * width
    spans = []
    for e in range(experts):
        spans += [(e * width, (e + 1) * width), (up_base + e * width, up_base + (e + 1) * width)]
    return select_rows(both, tuple(spans))


def table_tensors(table: GGUFFile, config: dict) -> list[tuple[str, int]]:
    """The Engram GGUF's tensors of rows and their row counts, in table order: llama.cpp's one
    tensor of every Engram block's rows, or the earlier layout's tensor per Engram block."""

    if TABLE_TENSOR in table.tensors:
        parts = [(TABLE_TENSOR, int(sum(config["engram_num_embeddings"])))]
    elif table.kv.get("general.architecture") == "deepseek41-engram":
        parts = [(ENGRAM_TENSOR.format(layer), rows_in_layer) for layer, rows_in_layer in
                 zip(config["engram_layer_ids"], config["engram_num_embeddings"])]
    else:
        raise ValueError(f"{table.path}: expected the release's Engram GGUF ({TABLE_TENSOR}, or "
                         "deepseek41-engram tables)")
    width = config["engram_head_dim"]
    for name, count in parts:
        info = table.info(name)
        if info.shape != (count, width) or info.type_name != "IQ4_NL":
            raise ValueError(f"{table.path}: {name} is {info.type_name} {info.shape}; "
                             f"expected IQ4_NL ({count}, {width})")
    return parts


def engram_table_source(table: GGUFFile, config: dict) -> LogicalSource:
    """The Engram GGUF's rows as one table, every Engram block's rows in order."""

    width = config["engram_head_dim"]
    parts = [block_source(table, name, (count, width), rows())
             for name, count in table_tensors(table, config)]
    return _stacked(parts, f"{table.path}:engram") if len(parts) > 1 else parts[0]


def table_digest(table: GGUFFile, config: dict) -> str:
    """SHA-256 of the Engram rows' bytes, which the artifact stores unchanged, row for row."""

    digest = hashlib.sha256()
    step = 1 << 28
    for name, _ in table_tensors(table, config):
        size = table.info(name).nbytes
        for begin in range(0, size, step):
            digest.update(table.tensor_bytes(name, begin, min(begin + step, size)))
    return digest.hexdigest()


def gguf_image_router_bias(gguf: GGUFShards, layers: int, experts: int) -> LogicalSource:
    """The routing bias of image positions, [block][expert] FP32, from the GGUF's
    `blk.N.exp_probs_b_vl.bias` (llama.cpp's layout)."""

    rows = []
    for i in range(layers):
        name = f"blk.{i}.exp_probs_b_vl.bias"
        info = gguf.info(name)
        if info.type_name != "F32" or info.shape != (experts,):
            raise ValueError(f"{name}: {info.type_name} {info.shape}, expected F32 ({experts},)")
        rows.append(gguf.file(name).read_direct(name))
    return array_source(torch.from_numpy(np.stack(rows)), "exp_probs_b_vl")


def build_model(base, gguf: GGUFShards | None, *, components=("text", "engram"),
                dspark: GGUFShards | None = None, vision: GGUFFile | None = None,
                bias_vl=None, resource_overrides=None):
    """DeepSeek-V4.1-Flash's text component, every parameter's source the GGUF, the `engram`
    component that describes (and with `engram` selected, stores) its Engram table, with `dspark`
    selected the DSpark drafter from its own GGUF, and with `vision` the vision tower from the
    mmproj and the image positions' routing bias from the GGUF (llama.cpp's layout) or the
    release's bias_vl tensors (the earlier layout)."""

    from .model import Model, Parameter

    selected = tuple(components)
    model_parts = set(selected) - {"dspark", "vision"}
    if len(set(selected)) != len(selected) or model_parts not in (
            {"text", "engram"}, {"text"}, {"engram"}) or (
            ("dspark" in selected or "vision" in selected) and "text" not in selected):
        raise ValueError("DeepSeek-V4.1 converts --components text,engram (the model with its "
                         "Engram table), text (the model alone) or engram (the table alone), "
                         "and dspark (its drafter) and vision (its vision tower) beside text")
    if ("dspark" in selected) != (dspark is not None):
        raise ValueError("the dspark component converts from --source dspark=DRAFTER.gguf")
    if ("vision" in selected) != (vision is not None):
        raise ValueError("the vision component converts from --source vision=MMPROJ.gguf")
    # llama.cpp's layout carries the image routing bias; the earlier one takes the release's.
    if (vision is not None and gguf is not None and not gguf.upstream) != (bias_vl is not None):
        raise ValueError("--source bias_vl=BIAS_VL.safetensors gives the vision component of an "
                         "earlier-layout GGUF its image routing bias; a GGUF in llama.cpp's "
                         "layout carries it (blk.N.exp_probs_b_vl.bias)")
    config = text_config(base.config)
    records = {"engram": {"config": engram_table_config(config)}}
    resource_overrides = dict(resource_overrides or {})
    profile = resource_overrides.pop("engram.hot", None)
    if profile is not None and ("engram" not in selected or gguf is None):
        raise ValueError("engram.hot stores a hot-row profile beside the Engram table "
                         "(--components ...,engram), checked against the Engram hash of "
                         "--source gguf=SHARD1")
    if "text" not in selected:
        if resource_overrides:
            raise ValueError("resource overrides have no selected frontend consumer")
        model = Model(records)
        _engram_table(model)
        if profile is not None:
            attach_engram_profile(model, Path(profile), config, engram_constants(gguf, config))
        return model
    if gguf is None:
        raise ValueError("DeepSeek-V4.1 converts its text component from --source gguf=SHARD1")
    config = with_gguf_expert_count(config, gguf)
    validate(gguf, config)
    constants = engram_constants(gguf, config)
    config["engram"] = {"multipliers": constants["multipliers"], "primes": constants["primes"],
                        "offsets": constants["offsets"], "pad_id": constants["pad_id"]}
    h, vocab = config["hidden_size"], config["vocab_size"]
    references, payloads, count, special = _load_resources(base.root, vocab, resource_overrides)
    references["engram_token_map"] = "resource/text/engram_token_map"
    payloads["resource/text/engram_token_map"] = constants["token_map"].tobytes()
    records["text"] = {"config": config, "resources": references}
    if dspark is not None:
        drafter_config = (dspark_stq_config(base.config, config, dspark)
                          if isinstance(dspark, SafetensorsSource)
                          else dspark_config(base.config, config, dspark))
        records["dspark"] = {"config": drafter_config, "target": "text"}
    if vision is not None:
        vc = vision_config(vision, base.config, config)
        records["vision"] = {"config": vc, "target": "text",
                             "resources": {"processor_config.json":
                                           "resource/vision/processor_config.json"}}
        payloads["resource/vision/processor_config.json"] = _dump_json(processor_config(vc))
    model = Model(records, resources=payloads, token_count=count, special_token_ids=special)

    def matrix(name, tensor, shape, inputs=(), source=None, file=gguf):
        model.add(Parameter(name, shape, source or _matrix(file, tensor, shape),
                            inputs=tuple(inputs)))

    def vector(name, tensor, shape, source=None, file=gguf, stored="fp32"):
        model.add(Parameter(name, shape, source or _direct(file, tensor, shape),
                            direct_format=stored))

    matrix("text/token_embedding", "token_embd.weight", (vocab, h))
    matrix("text/output_head", "output.weight", (vocab, h), ("text/final_hidden",))
    vector("text/final_norm", "output_norm.weight", (h,))
    heads, d = config["num_attention_heads"], config["head_dim"]
    hc = config["hc_mult"]
    mixes = (2 + hc) * hc
    q_rank, o_rank, groups = config["q_lora_rank"], config["o_lora_rank"], config["o_groups"]
    ih, idim = config["index_n_heads"], config["index_head_dim"]
    w = config["moe_intermediate_size"]

    # The parameters every block has, the backbone's and the drafter's: hyper-connections, norms
    # and the window attention (attention_block), and the MoE of `e` routed experts beside the shared
    # one (moe_block). A backbone block's compressor and indexer come between the two.
    def attention_block(file, index, p):
        g = f"blk.{index}."
        for site, prefix in (("attn", "hc_attention/"), ("ffn", "hc_ffn/")):
            vector(p + prefix + "fn", g + f"hc_{site}_fn.weight", (mixes, hc * h), file=file)
            vector(p + prefix + "base", g + f"hc_{site}_base.weight", (mixes,), file=file)
            vector(p + prefix + "scale", g + f"hc_{site}_scale.weight", (3,), file=file)
        vector(p + "attention_norm", g + "attn_norm.weight", (h,), file=file)
        vector(p + "ffn_norm", g + "ffn_norm.weight", (h,), file=file)
        a = p + "attention/"
        matrix(a + "query_a", g + "attn_q_a.weight", (q_rank, h), (p + "attention_input",),
               file=file)
        matrix(a + "kv", g + "attn_kv.weight", (d, h), (p + "attention_input",), file=file)
        vector(a + "query_a_norm", g + "attn_q_a_norm.weight", (q_rank,), file=file)
        vector(a + "kv_norm", g + "attn_kv_a_norm.weight", (d,), file=file)
        matrix(a + "query_b", g + "attn_q_b.weight", (heads * d, q_rank), (a + "query_latent",),
               file=file)
        matrix(a + "output_a", g + "attn_output_a.weight", (groups * o_rank, heads * d // groups),
               (a + "grouped_output",), file=file)
        matrix(a + "output_b", g + "attn_output_b.weight", (h, groups * o_rank),
               (a + "output_latent",), file=file)
        vector(a + "sinks", g + "attn_sinks.weight", (heads,), file=file)

    def moe_block(file, index, p, e):
        g, m = f"blk.{index}.", p + "moe/"
        vector(m + "router", g + "ffn_gate_inp.weight", (e, h), file=file)
        vector(m + "router_bias", g + "exp_probs_b.bias", (e,), file=file)
        matrix(m + "gate_up", None, (e * 2 * w, h), (m + "input",),
               expert_gate_up(file, index, e, w, h))
        matrix(m + "down", g + "ffn_down_exps.weight", (e * h, w), (m + "product",), file=file)
        matrix(m + "shared_gate", g + "ffn_gate_shexp.weight", (w, h), (m + "input",), file=file)
        matrix(m + "shared_up", g + "ffn_up_shexp.weight", (w, h), (m + "input",), file=file)
        matrix(m + "shared_down", g + "ffn_down_shexp.weight", (h, w), (m + "shared_product",),
               file=file)

    for i in range(config["num_hidden_layers"]):
        g, p = f"blk.{i}.", f"text/layers/{i}/"
        a = p + "attention/"
        attention_block(gguf, i, p)
        if i in config["kv_source_layer_ids"]:
            c = a + "compressor/"
            vector(c + "kv", g + "attn_compressor_kv.weight", (d, h))
            if config["compress_ratios"][i] > 1:
                vector(c + "gate", g + "attn_compressor_gate.weight", (d, h))
            vector(c + "norm", g + "attn_compressor_norm.weight", (d,))
            vector(a + "indexer/key", g + "indexer.attn_k.weight", (idim, d))
            vector(a + "indexer/key_norm", g + "indexer.k_norm.weight", (idim,))
        if i in config["index_source_layer_ids"]:
            vector(a + "indexer/query", g + "indexer.attn_q_b.weight", (ih * idim, q_rank))
            vector(a + "indexer/weights", g + "indexer.proj.weight", (ih, h))
        moe_block(gguf, i, p, config["n_routed_experts"])
        if i in config["engram_layer_ids"]:
            n = p + "engram/"
            columns = (config["engram_max_ngram_size"] - 1) * config["engram_n_heads"]
            matrix(n + "projection", g + ("engram_kv.weight" if gguf.upstream else
                                          "engram_wkv.weight"),
                   ((hc + 1) * h, columns * config["engram_head_dim"]), (n + "rows",))
            if gguf.upstream:
                # llama.cpp's converter stores the product of the BF16 factors, exact in F32.
                vector(n + "gate", g + "engram_gate.weight", (hc, h))
            else:
                q = gguf.file(g + "engram_q.weight").read_direct(g + "engram_q.weight")
                k = gguf.file(g + "engram_k.weight").read_direct(g + "engram_k.weight")
                gate = (q.astype(np.float32) * k.astype(np.float32)).reshape(hc, h)
                vector(n + "gate", None, (hc, h),
                       array_source(torch.from_numpy(np.ascontiguousarray(gate)),
                                    g + "engram_q*engram_k"))
    if dspark is not None and "stq" in records["dspark"]["config"]:
        _stq_drafter(model, dspark, records["dspark"]["config"], config)
    elif dspark is not None:
        drafter = records["dspark"]["config"]
        rank, taps = drafter["markov_rank"], len(drafter["target_layer_ids"])
        matrix("dspark/fc", "fc.weight", (h, taps * h), ("dspark/taps",), file=dspark)
        vector("dspark/tap_norm", "enc.output_norm.weight", (h,), file=dspark)
        vector("dspark/final_norm", "output_norm.weight", (h,), file=dspark)
        vector("dspark/markov_embed", "markov_w1.weight", (vocab, rank), file=dspark,
               stored="bf16")
        vector("dspark/markov_head", "markov_w2.weight", (vocab, rank), file=dspark,
               stored="bf16")
        vector("dspark/confidence", "conf_proj.weight", (h + rank,), file=dspark)
        for b in range(drafter["num_hidden_layers"]):
            attention_block(dspark, b, f"dspark/layers/{b}/")
            moe_block(dspark, b, f"dspark/layers/{b}/", drafter["n_routed_experts"])
    if vision is not None:
        vc = records["vision"]["config"]
        vh, vw, patch = vc["hidden_size"], vc["intermediate_size"], vc["patch_size"]
        merged = vh * vc["downsample_ratio"] ** 2

        def dense(name, tensor, shape, inputs):
            model.add(Parameter(name, shape, _exact_bf16(vision, tensor, shape),
                                inputs=tuple(inputs), residency="vision"))

        def exact(name, tensor, shape):
            model.add(Parameter(name, shape, _exact_bf16(vision, tensor, shape),
                                direct_format="fp32", residency="vision"))

        # llama.cpp's patch convolution [1024, 3, 14, 14] is the release's Linear over the patch's
        # (channel, y, x) values.
        dense("vision/patch_embedding", "v.patch_embd.weight", (vh, 3 * patch * patch),
              ("vision/patches",))
        exact("vision/patch_bias", "v.patch_embd.bias", (vh,))
        for i in range(vc["num_hidden_layers"]):
            g, p = f"v.blk.{i}.", f"vision/layers/{i}/"
            exact(p + "attention_norm", g + "ln1.weight", (vh,))
            dense(p + "attention/qkv", g + "attn_qkv.weight", (3 * vh, vh),
                  (p + "attention_input",))
            exact(p + "attention/qkv_bias", g + "attn_qkv.bias", (3 * vh,))
            dense(p + "attention/output", g + "attn_out.weight", (vh, vh),
                  (p + "attention_output",))
            exact(p + "attention/output_bias", g + "attn_out.bias", (vh,))
            exact(p + "mlp_norm", g + "ln2.weight", (vh,))
            dense(p + "mlp/gate", g + "ffn_gate.weight", (vw, vh), (p + "mlp_input",))
            dense(p + "mlp/up", g + "ffn_up.weight", (vw, vh), (p + "mlp_input",))
            dense(p + "mlp/down", g + "ffn_down.weight", (vh, vw), (p + "mlp_product",))
        exact("vision/final_norm", "v.post_ln.weight", (vh,))
        dense("vision/aligner/up", "mm.1.weight", (h, merged), ("vision/merged",))
        exact("vision/aligner/up_bias", "mm.1.bias", (h,))
        dense("vision/aligner/out", "mm.2.weight", (h, h), ("vision/aligner_hidden",))
        exact("vision/aligner/out_bias", "mm.2.bias", (h,))
        exact("vision/image_start", "v.token_embd.img_start", (h,))
        exact("vision/image_newline", "v.image_newline", (h,))
        exact("vision/image_end", "v.token_embd.img_end", (h,))
        layers, experts = config["num_hidden_layers"], config["n_routed_experts"]
        router_bias = (gguf_image_router_bias(gguf, layers, experts) if gguf.upstream
                       else image_router_bias(bias_vl, layers, experts))
        model.add(Parameter("vision/image_router_bias", (layers, experts), router_bias,
                            direct_format="fp32", residency="vision"))
    if "engram" in selected:
        _engram_table(model)
        if profile is not None:
            attach_engram_profile(model, Path(profile), config, constants)
    return model


def _stq_drafter(model, pack: SafetensorsSource, drafter: dict, config: dict) -> None:
    """The `dspark` component from an STQ pack's drafter (`mtp.0..2`, the release's names, the
    routed experts in the pack's one expert group): each STQ matrix as its trellis words
    (`NAME/trellis`, UINT16 [in/16][out/16][words], a leading [count] for the grouped output
    projection's groups and the routed experts), input signs (`NAME/suh`, FP16 [in]) and output
    signs (`NAME/svh`, FP16 [out]), all as stored; the vectors FP32 (BF16 widened exactly), the
    Markov tables BF16."""

    from .model import Parameter

    h, vocab = config["hidden_size"], config["vocab_size"]
    heads, d, hc = config["num_attention_heads"], config["head_dim"], config["hc_mult"]
    mixes = (2 + hc) * hc
    q_rank, o_rank, groups = config["q_lora_rank"], config["o_lora_rank"], config["o_groups"]
    w, e = config["moe_intermediate_size"], drafter["n_routed_experts"]
    rank, taps = drafter["markov_rank"], len(drafter["target_layer_ids"])
    trunk, routed = drafter["stq"]["trunk"], drafter["stq"]["experts"]

    def direct(name, tensor, shape, stored, dtypes, exact=False):
        info = pack.describe(tensor)
        if (info.dtype not in dtypes or int(np.prod(info.shape)) != int(np.prod(shape))
                or (exact and tuple(info.shape) != tuple(shape))):
            raise ValueError(f"{tensor}: expected {'/'.join(dtypes)} {shape}, found "
                             f"{info.dtype} {list(info.shape)}")
        model.add(Parameter(name, shape, tensor_source(pack, tensor, shape,
                                                       source_shape=tuple(info.shape)),
                            direct_format=stored))

    def vector(name, tensor, shape, stored="fp32"):
        direct(name, tensor, shape, stored, ("F32", "BF16") if stored == "fp32" else ("BF16",))

    def stq(name, tensor, k, n, codebook, count=None):
        lead = () if count is None else (count,)
        direct(name + "/trellis", tensor + ".trellis", lead + (k // 16, n // 16, STQ_WORDS[codebook]),
               "uint16", ("U16",), exact=True)
        direct(name + "/suh", tensor + ".suh", lead + (k,), "fp16", ("F16",), exact=True)
        direct(name + "/svh", tensor + ".svh", lead + (n,), "fp16", ("F16",), exact=True)

    last = f"mtp.{drafter['num_hidden_layers'] - 1}."
    stq("dspark/fc", "mtp.0.main_proj", taps * h, h, trunk)
    vector("dspark/tap_norm", "mtp.0.main_norm.weight", (h,))
    vector("dspark/final_norm", last + "norm.weight", (h,))
    vector("dspark/markov_embed", last + "markov_head.embed.weight", (vocab, rank), "bf16")
    vector("dspark/markov_head", last + "markov_head.head.weight", (vocab, rank), "bf16")
    vector("dspark/confidence", last + "confidence_head.proj.weight", (h + rank,))
    for b in range(drafter["num_hidden_layers"]):
        g, p = f"mtp.{b}.", f"dspark/layers/{b}/"
        if pack.has(g + "mlp.switch_mlp.gate_proj.g1.trellis"):
            raise ValueError(f"{pack.path}: drafter block {b} splits its experts into groups")
        for site, prefix in (("attn", "hc_attention/"), ("ffn", "hc_ffn/")):
            vector(p + prefix + "fn", g + f"hc_{site}_fn", (mixes, hc * h))
            vector(p + prefix + "base", g + f"hc_{site}_base", (mixes,))
            vector(p + prefix + "scale", g + f"hc_{site}_scale", (3,))
        vector(p + "attention_norm", g + "attn_norm.weight", (h,))
        vector(p + "ffn_norm", g + "ffn_norm.weight", (h,))
        a, m = p + "attention/", p + "moe/"
        stq(a + "query_a", g + "attn.wq_a", h, q_rank, trunk)
        stq(a + "kv", g + "attn.wkv", h, d, trunk)
        vector(a + "query_a_norm", g + "attn.q_norm.weight", (q_rank,))
        vector(a + "kv_norm", g + "attn.kv_norm.weight", (d,))
        stq(a + "query_b", g + "attn.wq_b", q_rank, heads * d, trunk)
        stq(a + "output_a", g + "attn.wo_a", heads * d // groups, o_rank, trunk, groups)
        stq(a + "output_b", g + "attn.wo_b", groups * o_rank, h, trunk)
        vector(a + "sinks", g + "attn.attn_sink", (heads,))
        vector(m + "router", g + "ffn.gate.weight", (e, h))
        vector(m + "router_bias", g + "ffn.gate.bias", (e,))
        stq(m + "shared_gate", g + "ffn.shared_experts.w1", h, w, trunk)
        stq(m + "shared_up", g + "ffn.shared_experts.w3", h, w, trunk)
        stq(m + "shared_down", g + "ffn.shared_experts.w2", w, h, trunk)
        stq(m + "gate", g + "mlp.switch_mlp.gate_proj.g0", h, w, routed, e)
        stq(m + "up", g + "mlp.switch_mlp.up_proj.g0", h, w, routed, e)
        stq(m + "down", g + "mlp.switch_mlp.down_proj.g0", w, h, routed, e)


def _engram_table(model):
    from .model import Parameter

    config = model.components["engram"]["config"]
    shape = (config["rows"], config["row_width"])

    def unavailable(begin, end):
        raise ValueError("the Engram table has no default source: provide --source engram")

    model.add(Parameter("engram/table", shape, LogicalSource(shape, "engram", unavailable),
                        residency="engram"))


def configure_engram(model, recipe, table) -> None:
    """Bind the IQ4_NL table rows and record the digest of their stored bytes."""

    if not isinstance(table, GGUFFile):
        raise ValueError("--source engram must name the release's Engram GGUF")
    descriptor = model.components["engram"]["config"]
    config = {"engram_layer_ids": descriptor["layers"],
              "engram_num_embeddings": descriptor["layer_rows"],
              "engram_head_dim": descriptor["row_width"]}
    source = engram_table_source(table, config)
    if source.read_encoded(0, 1).format != TABLE_FORMAT:
        raise ValueError("the Engram GGUF must hold IQ4_NL rows")
    descriptor["format"] = TABLE_FORMAT
    size = sum(table.info(name).nbytes for name, _ in table_tensors(table, config))
    print(f"hashing the Engram table ({size / 1e9:.1f} GB)", flush=True)
    descriptor["table_sha256"] = table_digest(table, config)
    if "engram/table" in model.parameters:
        recipe.assign("engram/table", format=TABLE_FORMAT, method=import_encoded, source=source)


def deepseek_v41_gguf(model, recipe, sources):
    """Every matrix keeps its GGUF blocks, the vision tower's its BF16 values; vectors are FP32 (the
    drafter's Markov tables BF16, as stored). A block's query-latent and key/value projections share one parent when the GGUF stores
    them in one format, as do the shared expert's gate and up rows, the drafter's blocks as the
    backbone's. The Engram table keeps its IQ4_NL rows (`--source engram`, always read: the model
    records the digest of its table even when the rows go into an artifact of their own)."""

    configure_engram(model, recipe, sources["engram"])
    if "text" not in model.components:
        return
    if model.config.get("architectures") != list(ARCHITECTURES):
        raise ValueError("the deepseek_v41_gguf recipe requires a DeepSeek-V4.1 model")
    formats = {}
    for name, parameter in model.parameters.items():
        if name == "engram/table":
            continue
        if name.startswith("vision/") and parameter.projection:
            # The tower's BF16 matrices as they are.
            formats[name] = "bf16"
            recipe.assign(name, format="bf16", method=cast_direct, source=parameter.source,
                          activation_policy="A16Only")
            continue
        if (parameter.direct_format in ("fp32", "fp16", "uint16") or name in MARKOV_TABLES) and \
                not parameter.projection:
            recipe.assign(name, format=parameter.direct_format, method=cast_direct,
                          source=parameter.source)
            continue
        fmt = formats[name] = parameter.source.read_encoded(0, 1).format
        recipe.assign(name, format=fmt, method=import_encoded, source=parameter.source,
                      **({"activation_policy": "AllowA8"} if parameter.projection else {}))
    config = model.config
    blocks = [f"text/layers/{i}/" for i in range(config["num_hidden_layers"])]
    if "dspark" in model.components and "stq" not in model.components["dspark"]["config"]:
        blocks += [f"dspark/layers/{b}/" for b in
                   range(model.components["dspark"]["config"]["num_hidden_layers"])]
    for p in blocks:
        pair = [p + "attention/query_a", p + "attention/kv"]
        if formats[pair[0]] == formats[pair[1]]:
            recipe.group(pair)
        pair = [p + "moe/shared_gate", p + "moe/shared_up"]
        if formats[pair[0]] != formats[pair[1]]:
            raise ValueError(f"{p}: shared gate and up are stored in different formats")
        recipe.group(pair)
    if "vision" in model.components:
        for i in range(model.components["vision"]["config"]["num_hidden_layers"]):
            recipe.group([f"vision/layers/{i}/mlp/gate", f"vision/layers/{i}/mlp/up"])


RECIPES = {"deepseek_v41_gguf": deepseek_v41_gguf}
