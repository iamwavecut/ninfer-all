"""Inkling 2 (Inkling2LoopForConditionalGeneration) from llama.cpp GGUFs, kept in their blocks.

Inkling 2 Mini is a looped MoE transformer of 24 stored layers: the prelude [0, 6), a core [6, 18)
that runs three times and the coda [18, 24), 48 layer visits, each with its own keys, values and
short-convolution history. Attention has no rotation: a relative-position feature per query head
scores a learned table of distances, a per-query sink logit joins each softmax denominator, Q and
K are clipped to an L2 norm, V is RMS-normed, and causal 4-tap convolutions run over K, V and both
residual updates. Every layer reads its inputs through a depth attention over the residual blocks
of 6 layers, and a core layer's passes after the first take messages from the previous pass through
rank-64 controllers. The MoE routes 8 of 128 experts by sigmoid score plus a selection bias and runs
SiTU (gated by a capped sigmoid product) over [gate; up] rows with a per-expert scale of each
product. The recipe imports the GGUF's block matrices row for row, as stored; a row of the artifact
is byte for byte a row of the GGUF, except that an MXFP4 expert bank (UD-Q2_K_XL's layer 20 down
bank), which the expert products do not take, becomes the same values in Q8_0 blocks. A BF16
matrix (UD-Q8_K_XL's attention projections, token table, head and last four layers' experts) keeps
its words and multiplies BF16 inputs. Other BF16 and F32 tensors (sink and dynamic-tap projections,
router, convolutions, clips, the distance tables, mixing queries, norms, controllers, expert
scales) become FP32, exactly, as do Q8_0 ones, whose values d * q FP32 holds (IsValorum's APEX
GGUFs store the sink and dynamic-tap projections in Q8_0).

The GGUFs come in two layouts that convert to the same artifact: llama.cpp's (`inkling2_loop`, 24
blocks, a block's four message queries in `blk.N.loop_msg_query` and its controllers
`loop_{q,k,v,ffn}_ctrl_*`) and the earlier one of Unsloth's root files (`inkling2`, 48 unrolled
blocks in its metadata, the queries in the global `loop_res_query` and `loop_kv_res_query` and the
controllers `loop_{attn,k,v,mlp}_*`).

The release's vision tower forms the `vision` component beside `text` (`--components text,vision`,
`--source vision=mmproj-BF16.gguf`, projector `inkling2v`): a ViT with a class row, learned
position rows resized to each image and 2D-rotated heads, then 2x2 patch merging and a two-layer
projector, its vectors FP32 and its matrices BF16 or, from a quantized mmproj such as ggml-org's
`mmproj-Inkling-2-Mini-Q8_0.gguf`, the GGUF's block rows as stored. The release's DSpark drafter
(`--components text,dspark`, `--source dspark=dspark-Inkling-2-Mini-Q8_0.gguf`, llama.cpp
architecture `dflash`) forms the `dspark` component: three blocks over the target's tapped hidden
states, each with its own context projections, and a recurrent Markov head that corrects the
target's head.

    python -m tools.convert --model HF_DIR --recipe inkling2_gguf \\
        --source gguf=Inkling-2-Mini-UD-IQ4_XS.gguf \\
        --source vision=mmproj-BF16.gguf --source dspark=dspark-Inkling-2-Mini-Q8_0.gguf \\
        --components text,vision,dspark --out inkling-2-mini-UD-IQ4_XS.ninfer

`HF_DIR` holds the release's `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`chat_template.jinja` and `generation_config.json` (no weights are read from it).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from tools.artifact.formats import GGUF_FORMATS_BY_TYPE

from .gguf_blocks import (GGML_MXFP4, GGML_Q8_0, bf16_source, block_source, dequantize_blocks,
                          dequantize_q8_0,
                          mxfp4_as_q8_0)
from .methods import cast_direct, import_encoded
from .resources import _dump_json, token_domain
from .sources.gguf import TYPE_BF16, GGUFFile
from .sources.logical import EncodedRows, LogicalSource, array_source
from .ternary import rows

ARCHITECTURES = ("Inkling2LoopForConditionalGeneration",)
TEXT_ARCHITECTURE = "Inkling2ForCausalLM"
VISION_ARCHITECTURE = "Inkling2VisionModel"
DSPARK_ARCHITECTURE = "Inkling2DSpark"
RESOURCES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)
IMAGE_TOKEN = "<|unused_200054|>"
# The four inputs a core layer's controllers mix messages into, in the order of the artifact's
# message queries and controllers.
CONTROLLERS = ("query", "key", "value", "ffn")
UPSTREAM_CONTROLLERS = ("loop_q_ctrl", "loop_k_ctrl", "loop_v_ctrl", "loop_ffn_ctrl")
EARLIER_CONTROLLERS = ("loop_attn", "loop_k", "loop_v", "loop_mlp")


def _int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name}: expected a positive integer")
    return value


def _real(value, name):
    if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name}: expected a positive number")
    return float(value)


def text_config(source: dict) -> dict:
    """Inkling 2's text config, normalised for src/models/inkling2."""

    if source.get("architectures") != list(ARCHITECTURES) or source.get("model_type") != "inkling2_loop":
        raise ValueError(f"unsupported Inkling 2 architecture {source.get('architectures')!r}")
    t = source.get("text_config")
    if not isinstance(t, dict) or t.get("model_type") != "inkling2_loop_text":
        raise ValueError("config.json has no inkling2_loop_text text_config")
    if t.get("q_bias") or t.get("o_bias"):
        raise ValueError("attention biases are not implemented")
    if t.get("use_value_norm_no_gamma") is not True or t.get("attn_res_normalize_values") is not True:
        raise ValueError("Inkling 2 RMS-normalizes its values (no gamma) and its mixed blocks")
    if t.get("n_shared_experts", 0) != 0 or t.get("use_gate_bias") is not True:
        raise ValueError("Inkling 2 routes with a selection bias and has no shared expert")
    out = {"architectures": [TEXT_ARCHITECTURE], "model_type": "inkling2"}
    for key in ("hidden_size", "vocab_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "n_routed_experts", "num_experts_per_tok",
                "intermediate_size", "sconv_kernel_size", "n_layers_per_attn_res_block",
                "sliding_window_size", "d_rel", "rel_extent", "log_scaling_n_floor",
                "unpadded_vocab_size"):
        out[key] = _int(t.get(key), f"text_config.{key}")
    for key in ("swa_head_dim", "swa_num_attention_heads", "swa_num_key_value_heads"):
        plain = key.replace("swa_", "")
        if t.get(key, out[plain]) != out[plain]:
            raise ValueError(f"text_config.{key}: sliding-window layers of another geometry are "
                             "not implemented")
    out["max_position_embeddings"] = _int(t.get("model_max_length"), "text_config.model_max_length")
    for key in ("rms_norm_eps", "qk_l2_clip_threshold", "rel_l2_clip_threshold",
                "final_logit_softcapping", "mlp_gate_softcap", "mlp_up_softcap", "route_scale"):
        out[key] = _real(t.get(key), f"text_config.{key}")
    alpha = t.get("log_scaling_alpha")
    if type(alpha) not in (int, float) or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("text_config.log_scaling_alpha must be a finite number at least 0")
    out["log_scaling_alpha"] = float(alpha)
    layers = out["num_hidden_layers"]
    local = t.get("local_layer_ids")
    if (not isinstance(local, list) or len(set(local)) != len(local)
            or any(type(i) is not int or not 0 <= i < layers for i in local)):
        raise ValueError("text_config.local_layer_ids must be distinct layer indices")
    out["layer_types"] = ["sliding_attention" if i in local else "full_attention"
                          for i in range(layers)]
    if out["num_attention_heads"] % out["num_key_value_heads"]:
        raise ValueError("query heads must be a multiple of the key/value heads")
    if out["num_experts_per_tok"] > out["n_routed_experts"]:
        raise ValueError("more experts per token than experts")
    if out["unpadded_vocab_size"] > out["vocab_size"]:
        raise ValueError("the unpadded vocabulary is larger than the embedding table")
    loop = t.get("loop_config") or {}
    passes = loop.get("num_passes", 1)
    if loop.get("is_varloop"):
        raise ValueError("a varying loop is not implemented")
    if type(passes) is not int or passes < 1:
        raise ValueError("text_config.loop_config.num_passes must be a positive integer")
    block = out["n_layers_per_attn_res_block"]
    if passes > 1:
        start, end = loop.get("core_start"), loop.get("core_end")
        if (type(start) is not int or type(end) is not int or not 0 < start < end <= layers
                or start % block or (end - start) % block):
            raise ValueError("the looped core must start after the first block and span whole "
                             "residual blocks")
        out["loop"] = {"core_start": start, "core_end": end, "num_passes": passes,
                       "gate_rank": _int(loop.get("gate_rank"), "loop_config.gate_rank")}
    else:
        out["loop"] = {"core_start": 0, "core_end": 0, "num_passes": 1, "gate_rank": 0}
    if source.get("eos_token_id") is not None:
        out["eos_token_id"] = source["eos_token_id"]
    return out


def visits(config: dict) -> list[tuple[int, int]]:
    """(layer, pass) of every layer visit, in order."""

    loop = config["loop"]
    start, end, passes = loop["core_start"], loop["core_end"], loop["num_passes"]
    if passes == 1:
        return [(i, 0) for i in range(config["num_hidden_layers"])]
    out = [(i, 0) for i in range(start)]
    for k in range(passes):
        out += [(i, k) for i in range(start, end)]
    return out + [(i, 0) for i in range(end, config["num_hidden_layers"])]


def upstream_layout(gguf: GGUFFile) -> bool:
    """llama.cpp's layout (`inkling2_loop`) or the earlier unrolled one (`inkling2`)."""

    arch = gguf.kv.get("general.architecture")
    if arch not in ("inkling2_loop", "inkling2"):
        raise ValueError(f"{gguf.path}: expected an Inkling 2 GGUF, found {arch!r}")
    return arch == "inkling2_loop"


def validate(gguf: GGUFFile, config: dict) -> None:
    """The GGUF's architecture metadata against the release's config.json."""

    kv, upstream = gguf.kv, upstream_layout(gguf)
    f = gguf.kv["general.architecture"] + "."
    loop = config["loop"]
    schedule = visits(config)
    blocks = config["num_hidden_layers"] if upstream else len(schedule)
    per_block = [kind == "sliding_attention" for kind in config["layer_types"]]
    if not upstream:
        per_block = [per_block[layer] for layer, _ in schedule]
    expected = {
        "block_count": blocks,
        "embedding_length": config["hidden_size"],
        "attention.key_length": config["head_dim"],
        "attention.value_length": config["head_dim"],
        "attention.sliding_window": config["sliding_window_size"],
        "attention.relative_length": config["d_rel"],
        "attention.relative_extent": config["rel_extent"],
        "attention.log_scale_floor": config["log_scaling_n_floor"],
        "expert_count": config["n_routed_experts"],
        "expert_used_count": config["num_experts_per_tok"],
        "expert_feed_forward_length": config["intermediate_size"],
        "shortconv.l_cache": config["sconv_kernel_size"],
        "attn_res.block_size": config["n_layers_per_attn_res_block"],
        "vocab_size": config["vocab_size"],
        "vocab_size_unpadded": config["unpadded_vocab_size"],
        "rope.dimension_count": 0,
    }
    if upstream:
        expected.update({"loop.count": loop["num_passes"], "loop.start": loop["core_start"],
                         "loop.end": loop["core_end"]})
    else:
        expected.update({"num_loops": loop["num_passes"], "loop.core_start": loop["core_start"],
                         "loop.core_end": loop["core_end"]})
    if loop["num_passes"] > 1:
        expected["loop.gate_rank"] = loop["gate_rank"]
    for key, value in expected.items():
        if kv.get(f + key) != value:
            raise ValueError(f"{gguf.path}: {f + key} is {kv.get(f + key)!r}, the config's "
                             f"{value!r}")
    if upstream and kv.get(f + "loop.varying", False):
        raise ValueError(f"{gguf.path}: a varying loop is not implemented")
    for key, value in (("attention.head_count", config["num_attention_heads"]),
                       ("attention.head_count_kv", config["num_key_value_heads"])):
        stored = kv.get(f + key)
        if (stored if isinstance(stored, list) else [stored] * blocks) != [value] * blocks:
            raise ValueError(f"{gguf.path}: {f + key} differs from the config's {value}")
    if [bool(v) for v in kv.get(f + "attention.sliding_window_pattern", [])] != per_block:
        raise ValueError(f"{gguf.path}: sliding_window_pattern differs from local_layer_ids")
    reals = {"attention.layer_norm_rms_epsilon": config["rms_norm_eps"],
             "attention.qk_clip": config["qk_l2_clip_threshold"],
             "attention.relative_clip": config["rel_l2_clip_threshold"],
             "attention.log_scale_alpha": config["log_scaling_alpha"],
             "final_logit_softcapping": config["final_logit_softcapping"],
             "activation.situ_beta": config["mlp_gate_softcap"],
             "activation.situ_linear_beta": config["mlp_up_softcap"],
             "expert_weights_scale": config["route_scale"]}
    for key, value in reals.items():
        if not np.isclose(float(kv.get(f + key, np.nan)), value, rtol=1e-6, atol=0.0):
            raise ValueError(f"{gguf.path}: {f + key} differs from the config's {value!r}")


def gguf_reals(gguf: GGUFFile, config: dict) -> dict:
    """The config with the GGUF's FP32 constants, which llama.cpp computes with: the clips, the
    epsilon and the logarithmic scale in FP32 as stored."""

    f = gguf.kv["general.architecture"] + "."
    out = dict(config)
    for key, name in (("attention.layer_norm_rms_epsilon", "rms_norm_eps"),
                      ("attention.qk_clip", "qk_l2_clip_threshold"),
                      ("attention.relative_clip", "rel_l2_clip_threshold"),
                      ("attention.log_scale_alpha", "log_scaling_alpha")):
        out[name] = float(np.float32(gguf.kv[f + key]))
    return out


def _load_resources(root: Path, vocab: int, overrides: dict):
    references, payloads = {}, {}
    for role in RESOURCES:
        path = Path(overrides[role]) if role in overrides else root / role
        data = path.read_bytes()
        if not data:
            raise ValueError(f"{path}: resource is empty")
        object_id = f"resource/text/{role}"
        references[role] = object_id
        payloads[object_id] = data
    tokenizer = json.loads(payloads["resource/text/tokenizer.json"])
    tokenizer_config = json.loads(payloads["resource/text/tokenizer_config.json"])
    count, special = token_domain(tokenizer, tokenizer_config, vocab)
    return references, payloads, count, special


def _direct(gguf: GGUFFile, tensor: str, shape: tuple[int, ...]) -> LogicalSource:
    """F32 as stored, BF16 and F16 widened exactly, Q8_0 blocks decoded exactly."""

    info = gguf.info(tensor)
    if info.type_id == GGML_Q8_0:
        values = dequantize_q8_0(gguf.read_blocks(tensor))
    elif info.type_name in ("F32", "BF16", "F16"):
        values = gguf.read_direct(tensor).astype(np.float32)
    else:
        # Vector-role params are usually F32/BF16/F16/Q8_0, but third-party quantizers
        # may store them in other block formats (e.g. Q4_K for attn_sink_proj).
        # Decode via the shared dequantizer so any GGUF block type is accepted.
        values = np.asarray(dequantize_blocks(info.type_id, gguf.read_blocks(tensor)),
                            dtype=np.float32)
    if values.size != int(np.prod(shape)):
        raise ValueError(f"{tensor}: stored {info.shape}, expected {shape}")
    return array_source(torch.from_numpy(np.ascontiguousarray(values.reshape(shape))), tensor)


def _exact_bf16(gguf: GGUFFile, tensor: str, shape: tuple[int, ...]) -> LogicalSource:
    """`tensor` as FP32 values that BF16 holds exactly; a wider value is an error (llama.cpp writes
    some of the release's BF16 tensors as F32)."""

    source = _direct(gguf, tensor, shape)
    values = source.values(0, int(np.prod(shape))).reshape(shape)
    if not torch.equal(values.to(torch.bfloat16).to(torch.float32), values):
        raise ValueError(f"{tensor}: values are not BF16 values")
    return source


def _matrix(gguf: GGUFFile, tensor: str, shape: tuple[int, int]) -> LogicalSource:
    """A matrix as stored: ggml block rows (encoded), or BF16 words (values, no read_encoded)."""

    info = gguf.info(tensor)
    if info.type_id not in GGUF_FORMATS_BY_TYPE and info.type_id != TYPE_BF16:
        raise ValueError(f"{tensor}: expected ggml block rows or BF16, found {info.type_name}")
    stored = (int(np.prod(info.shape[:-1])), info.shape[-1])
    if stored != shape:
        raise ValueError(f"{tensor}: stored {info.shape}, expected rows x columns {shape}")
    if info.type_id == TYPE_BF16:
        return bf16_source(gguf, tensor, shape)
    source = block_source(gguf, tensor, shape, rows())
    if info.type_id != GGML_MXFP4:
        return source

    # The expert products have no MXFP4 kernel (UD-Q2_K_XL stores one down bank so): the same
    # values as Q8_0 blocks, each code's doubled value times 2^(e - 128) as d.
    def encoded(begin: int, end: int) -> EncodedRows:
        stored_rows = source.read_encoded(begin, end)
        blocks = mxfp4_as_q8_0(stored_rows.codes.numpy())
        return replace(stored_rows, format="gguf_q8_0", codes=torch.from_numpy(blocks))

    return replace(source, label=f"{tensor}[gguf_mxfp4 as gguf_q8_0]{list(shape)}",
                   read_encoded=encoded)


def message_queries(gguf: GGUFFile, config: dict, layer: int) -> LogicalSource:
    """The four message queries of core layer `layer`, [query; key; value; ffn] x hidden."""

    h = config["hidden_size"]
    if upstream_layout(gguf):
        return _direct(gguf, f"blk.{layer}.loop_msg_query.weight", (4, h))
    core = config["loop"]["core_end"] - config["loop"]["core_start"]
    i = layer - config["loop"]["core_start"]
    q = gguf.read_direct("loop_res_query.weight").astype(np.float32).reshape(2 * core, h)
    kv = gguf.read_direct("loop_kv_res_query.weight").astype(np.float32).reshape(2 * core, h)
    rows_ = np.stack([q[2 * i], kv[2 * i], kv[2 * i + 1], q[2 * i + 1]])
    return array_source(torch.from_numpy(np.ascontiguousarray(rows_)),
                        f"loop queries of layer {layer}")


def vision_config(mmproj: GGUFFile, source: dict, text: dict) -> dict:
    """The `vision` component's config from the release's vision_config, checked against the
    mmproj's metadata and the text model's width."""

    kv = mmproj.kv
    if kv.get("general.architecture") != "clip" or not kv.get("clip.has_vision_encoder"):
        raise ValueError(f"{mmproj.path}: expected a llama.cpp clip GGUF with a vision encoder")
    if kv.get("clip.vision.projector_type", kv.get("clip.projector_type")) != "inkling2v":
        raise ValueError(f"{mmproj.path}: expected Inkling 2's projector (inkling2v)")
    raw = source.get("vision_config")
    if not isinstance(raw, dict) or raw.get("model_type") != "inkling2_vision":
        raise ValueError("config.json has no inkling2_vision vision_config")
    out = {"architectures": [VISION_ARCHITECTURE], "model_type": "inkling2_vision"}
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "native_patch_size",
                "adapter_merge_size", "projection_hidden_dim", "posemb_grid_size",
                "text_hidden_size"):
        out[key] = _int(raw.get(key), f"vision_config.{key}")
    if raw.get("projector_bias"):
        raise ValueError("a projector with biases is not implemented")
    if out["text_hidden_size"] != text["hidden_size"]:
        raise ValueError("vision_config.text_hidden_size differs from the text model's width")
    out["intermediate_size"] = int(round(out["hidden_size"] * float(raw.get("mlp_ratio", 0))))
    out["layer_norm_eps"] = float(np.float32(kv["clip.vision.attention.layer_norm_epsilon"]))
    out["rope_theta"] = 10000.0
    out["max_image_tokens"] = 4096
    v = "clip.vision."
    expected = {"block_count": out["num_hidden_layers"], "embedding_length": out["hidden_size"],
                "attention.head_count": out["num_attention_heads"],
                "feed_forward_length": out["intermediate_size"],
                "patch_size": out["native_patch_size"], "projection_dim": text["hidden_size"],
                "spatial_merge_size": out["adapter_merge_size"]}
    for key, value in expected.items():
        if kv.get(v + key) != value:
            raise ValueError(f"{mmproj.path}: {v + key} is {kv.get(v + key)!r}, the config's "
                             f"{value!r}")
    if not np.isclose(float(raw.get("layer_norm_eps", np.nan)), out["layer_norm_eps"],
                      rtol=1e-6, atol=0.0):
        raise ValueError(f"{mmproj.path}: the tower's norm epsilon differs from the config's")
    if (list(kv.get(v + "image_mean", [])) != [0.5] * 3
            or list(kv.get(v + "image_std", [])) != [0.5] * 3):
        raise ValueError(f"{mmproj.path}: expected pixels scaled to [-1, 1]")
    heads = out["num_attention_heads"]
    if out["hidden_size"] % heads or (out["hidden_size"] // heads) % 4:
        raise ValueError("vision heads must split the width into heads of 2D-rotated pairs")
    return out


def processor_config(vision: dict, image_token_id: int) -> dict:
    """The image preprocessing as the frontend reads it: the image padded with -1 to whole
    merged cells at its own size (resized, aspect kept, only past max_image_tokens cells), values
    (x - 127.5) / 127.5, one prompt token per merged cell after <|content_image|>."""

    return {"image_processor": {
        "patch_size": vision["native_patch_size"], "merge_size": vision["adapter_merge_size"],
        "max_image_tokens": vision["max_image_tokens"], "pad_value": -1.0,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "image_token": IMAGE_TOKEN, "image_token_id": image_token_id}}


def dspark_config(source: dict, config: dict, dspark: GGUFFile) -> dict:
    """The `dspark` component's config from the release's mtp_config, checked against the drafter
    GGUF's metadata."""

    raw = source.get("mtp_config")
    if not isinstance(raw, dict) or raw.get("markov_head_type") != "rnn_hidden":
        raise ValueError("config.json has no mtp_config with a recurrent Markov head")
    out = {"architectures": [DSPARK_ARCHITECTURE], "model_type": "inkling2_dspark"}
    for key in ("num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
                "intermediate_size", "block_size", "markov_rank", "sliding_window_size"):
        out[key] = _int(raw.get(key), f"mtp_config.{key}")
    if raw.get("hidden_size") != config["hidden_size"] or raw.get("q_bias") or raw.get("o_bias"):
        raise ValueError("the drafter's width differs from the model's, or it has biases")
    out["rope_theta"] = _real(raw.get("rope_theta"), "mtp_config.rope_theta")
    out["rms_norm_eps"] = _real(raw.get("rms_norm_eps"), "mtp_config.rms_norm_eps")
    local = raw.get("local_layer_ids")
    if (not isinstance(local, list)
            or any(type(i) is not int or not 0 <= i < out["num_hidden_layers"] for i in local)):
        raise ValueError("mtp_config.local_layer_ids must be drafter layer indices")
    out["layer_types"] = ["sliding_attention" if i in local else "full_attention"
                          for i in range(out["num_hidden_layers"])]
    targets = raw.get("target_layer_ids")
    total = len(visits(config))
    if (not isinstance(targets, list) or len(targets) < 2 or targets != sorted(set(targets))
            or any(type(v) is not int or not 0 <= v <= total for v in targets)
            or targets[-1] != total):
        raise ValueError("mtp_config.target_layer_ids must name layer visits in order, the last "
                         "the final hidden state")
    out["target_layer_ids"] = list(targets)
    if out["num_attention_heads"] % out["num_key_value_heads"]:
        raise ValueError("drafter query heads must be a multiple of its key/value heads")
    kv, f = dspark.kv, "dflash."
    if kv.get("general.architecture") != "dflash":
        raise ValueError(f"{dspark.path}: expected the release's DSpark drafter (dflash) GGUF")
    expected = {
        "block_count": out["num_hidden_layers"], "embedding_length": config["hidden_size"],
        "feed_forward_length": out["intermediate_size"],
        "attention.head_count": out["num_attention_heads"],
        "attention.head_count_kv": out["num_key_value_heads"],
        "attention.key_length": out["head_dim"], "attention.value_length": out["head_dim"],
        "rope.dimension_count": out["head_dim"], "block_size": out["block_size"],
        "dspark.markov_rank": out["markov_rank"], "sample_from_anchor": False,
        "has_confidence_head": False, "attention.causal": False,
        "attention.sliding_window": out["sliding_window_size"],
        "vocab_size": config["vocab_size"], "vocab_size_unpadded": config["unpadded_vocab_size"],
    }
    for key, value in expected.items():
        if kv.get(f + key) != value:
            raise ValueError(f"{dspark.path}: {f + key} is {kv.get(f + key)!r}, expected "
                             f"{value!r}")
    if list(kv.get(f + "target_layers", [])) != out["target_layer_ids"]:
        raise ValueError(f"{dspark.path}: the target layers differ from the config's")
    pattern = [kind == "sliding_attention" for kind in out["layer_types"]]
    if [bool(v) for v in kv.get(f + "attention.sliding_window_pattern", [])] != pattern:
        raise ValueError(f"{dspark.path}: the sliding-window pattern differs from the config's")
    reals = {"rope.freq_base": out["rope_theta"],
             "attention.layer_norm_rms_epsilon": out["rms_norm_eps"],
             "final_logit_softcapping": config["final_logit_softcapping"]}
    for key, value in reals.items():
        if not np.isclose(float(kv.get(f + key, np.nan)), value, rtol=1e-6, atol=0.0):
            raise ValueError(f"{dspark.path}: {f + key} differs from the config's {value!r}")
    out["rms_norm_eps"] = float(np.float32(kv[f + "attention.layer_norm_rms_epsilon"]))
    # The anchor's embedding goes through the target's embedding norm; its mask rows are the
    # drafter's own embedding.
    out["mask_token_id"] = int(kv.get("tokenizer.ggml.mask_token_id", 0))
    return out


def build_model(base, gguf: GGUFFile, *, vision: GGUFFile | None = None,
                dspark: GGUFFile | None = None, resource_overrides=None):
    """Inkling 2's text component, every parameter's source the GGUF, with `vision` its vision
    tower from the mmproj and with `dspark` its drafter from the drafter GGUF."""

    from .model import Model, Parameter

    config = text_config(base.config)
    validate(gguf, config)
    config = gguf_reals(gguf, config)
    h, vocab = config["hidden_size"], config["vocab_size"]
    references, payloads, count, special = _load_resources(
        base.root, vocab, dict(resource_overrides or {}))
    if count != config["unpadded_vocab_size"]:
        raise ValueError(f"the tokenizer has {count} tokens, the config's unpadded vocabulary "
                         f"{config['unpadded_vocab_size']}")
    records = {"text": {"config": config, "resources": references}}
    if vision is not None:
        vc = vision_config(vision, base.config, config)
        tokenizer = json.loads(payloads["resource/text/tokenizer.json"])
        ids = {t["content"]: t["id"] for t in tokenizer.get("added_tokens", [])}
        if IMAGE_TOKEN not in ids:
            raise ValueError(f"the tokenizer lacks the image token {IMAGE_TOKEN}")
        vc["image_token_id"] = ids[IMAGE_TOKEN]
        records["vision"] = {"config": vc, "target": "text",
                             "resources": {"processor_config.json":
                                           "resource/vision/processor_config.json"}}
        payloads["resource/vision/processor_config.json"] = _dump_json(
            processor_config(vc, ids[IMAGE_TOKEN]))
    if dspark is not None:
        records["dspark"] = {"config": dspark_config(base.config, config, dspark),
                             "target": "text"}
    model = Model(records, resources=payloads, token_count=count, special_token_ids=special)

    def matrix(name, tensor, shape, inputs=(), file=gguf):
        model.add(Parameter(name, shape, _matrix(file, tensor, shape), inputs=tuple(inputs)))

    def vector(name, tensor, shape, file=gguf, source=None):
        model.add(Parameter(name, shape, source or _direct(file, tensor, shape),
                            direct_format="fp32"))

    matrix("text/token_embedding", "token_embd.weight", (vocab, h))
    vector("text/embedding_norm", "token_embd_norm.weight", (h,))
    matrix("text/output_head", "output.weight", (vocab, h), ("text/final_hidden",))
    vector("text/final_norm", "output_norm.weight", (h,))
    vector("text/output_mix", "output_res_query.weight", (h,))
    heads, kv_heads, d = (config["num_attention_heads"], config["num_key_value_heads"],
                          config["head_dim"])
    rel, extent, taps = config["d_rel"], config["rel_extent"], config["sconv_kernel_size"]
    window = config["sliding_window_size"]
    e, w = config["n_routed_experts"], config["intermediate_size"]
    loop = config["loop"]
    upstream = upstream_layout(gguf)
    controllers = UPSTREAM_CONTROLLERS if upstream else EARLIER_CONTROLLERS
    for i in range(config["num_hidden_layers"]):
        g, p = f"blk.{i}.", f"text/layers/{i}/"
        a, m = p + "attention/", p + "moe/"
        full = config["layer_types"][i] == "full_attention"
        matrix(a + "query", g + "attn_q.weight", (heads * d, h), (a + "query_input",))
        matrix(a + "gate", g + "attn_gate.weight", (heads * d, h), (a + "query_input",))
        matrix(a + "relative", g + "attn_rel.weight", (heads * rel, h), (a + "query_input",))
        matrix(a + "key", g + "attn_k.weight", (kv_heads * d, h), (a + "key_input",))
        matrix(a + "value", g + "attn_v.weight", (kv_heads * d, h), (a + "value_input",))
        matrix(a + "output", g + "attn_output.weight", (h, heads * d), (a + "gated_output",))
        vector(a + "sinks", g + "attn_sink_proj.weight", (heads, h))
        vector(a + "query_clip", g + "attn_q_clip.weight", (heads, d))
        vector(a + "relative_clip", g + "attn_rel_clip.weight", (heads, rel))
        vector(a + "distances", g + "attn_rel_bias.weight", (window if not full else extent, rel))
        vector(a + "key_conv", g + "attn_k_conv.weight", (kv_heads * d, taps))
        vector(a + "value_conv", g + "attn_v_conv.weight", (kv_heads * d, taps))
        vector(a + "output_conv", g + "attn_post_conv.weight", (h, taps))
        if full:
            vector(a + "query_norm", g + "attn_q_inp_norm.weight", (h,))
            vector(a + "key_norm", g + "attn_k_inp_norm.weight", (h,))
            vector(a + "value_norm", g + "attn_v_inp_norm.weight", (h,))
            vector(a + "key_taps", g + "attn_k_conv_proj.weight", (kv_heads * taps, h))
            vector(a + "value_taps", g + "attn_v_conv_proj.weight", (kv_heads * taps, h))
        else:
            vector(a + "input_norm", g + "attn_norm.weight", (h,))
        if i > 0:
            vector(p + "attention_mix", g + "attn_res_query.weight", (1 if not full else 3, h))
        vector(p + "ffn_mix", g + "ffn_res_query.weight", (h,))
        vector(p + "ffn_norm", g + "ffn_norm.weight", (h,))
        vector(m + "router", g + "ffn_gate_inp.weight", (e, h))
        vector(m + "router_bias", g + "exp_probs_b.bias", (e,))
        matrix(m + "gate_up", g + "ffn_gate_up_exps.weight", (e * 2 * w, h), (m + "input",))
        matrix(m + "down", g + "ffn_down_exps.weight", (e * h, w), (m + "product",))
        for tensor, name in (("ffn_gate_up_exps.scale", "gate_up_scale"),
                             ("ffn_down_exps.scale", "down_scale")):
            if g + tensor in gguf.tensors:
                vector(m + name, g + tensor, (e,))
        vector(m + "output_conv", g + "ffn_post_conv.weight", (h, taps))
        if loop["num_passes"] > 1 and loop["core_start"] <= i < loop["core_end"]:
            r = loop["gate_rank"]
            vector(p + "loop/queries", None, (4, h), source=message_queries(gguf, config, i))
            for name, stored in zip(CONTROLLERS, controllers):
                c = f"{p}loop/{name}/"
                stem = g + stored + "_"
                vector(c + "down", stem + "down.weight", (r, 2 * h))
                vector(c + "gain", stem + "gain.weight", (h, r))
                vector(c + "forget", stem + "forget.weight", (h, r))
    if vision is not None:
        _vision(model, vision, records["vision"]["config"], config)
    if dspark is not None:
        _dspark(model, dspark, records["dspark"]["config"], config)
    return model


def _vision(model, vision: GGUFFile, vc: dict, config: dict) -> None:
    """The tower's matrices, BF16 or the mmproj's block rows as stored, and its FP32 vectors
    (llama.cpp writes the release's BF16 patch embedding, position rows and norms as F32: each
    must be a BF16 value)."""

    from .model import Parameter

    vh, vw, patch = vc["hidden_size"], vc["intermediate_size"], vc["native_patch_size"]
    merged = vh * vc["adapter_merge_size"] ** 2
    grid = vc["posemb_grid_size"]

    def dense(name, tensor, shape, inputs):
        blocks = vision.info(tensor).type_id in GGUF_FORMATS_BY_TYPE
        source = (_matrix(vision, tensor, shape) if blocks
                  else _exact_bf16(vision, tensor, shape))
        model.add(Parameter(name, shape, source, inputs=tuple(inputs), residency="vision"))

    def exact(name, tensor, shape):
        model.add(Parameter(name, shape, _direct(vision, tensor, shape), direct_format="fp32",
                            residency="vision"))

    # llama.cpp's patch convolution [vh, 3, p, p] is the release's Linear over the patch's
    # (channel, y, x) values.
    dense("vision/patch_embedding", "v.patch_embd.weight", (vh, 3 * patch * patch),
          ("vision/patches",))
    exact("vision/class_embedding", "v.class_embd", (vh,))
    exact("vision/class_position", "v.class_position_embd.weight", (vh,))
    exact("vision/positions", "v.position_embd.weight", (grid * grid, vh))
    exact("vision/pre_norm", "v.pre_ln.weight", (vh,))
    exact("vision/pre_norm_bias", "v.pre_ln.bias", (vh,))
    for i in range(vc["num_hidden_layers"]):
        g, p = f"v.blk.{i}.", f"vision/layers/{i}/"
        a = p + "attention/"
        exact(p + "attention_norm", g + "ln1.weight", (vh,))
        exact(p + "attention_norm_bias", g + "ln1.bias", (vh,))
        for site in ("q", "k", "v"):
            name = {"q": "query", "k": "key", "v": "value"}[site]
            dense(a + name, g + f"attn_{site}.weight", (vh, vh), (p + "attention_input",))
            exact(a + name + "_bias", g + f"attn_{site}.bias", (vh,))
        dense(a + "output", g + "attn_out.weight", (vh, vh), (a + "output_input",))
        exact(a + "output_bias", g + "attn_out.bias", (vh,))
        exact(p + "attention_scale", g + "ls1.weight", (vh,))
        exact(p + "mlp_norm", g + "ln2.weight", (vh,))
        exact(p + "mlp_norm_bias", g + "ln2.bias", (vh,))
        dense(p + "mlp/up", g + "ffn_up.weight", (vw, vh), (p + "mlp_input",))
        exact(p + "mlp/up_bias", g + "ffn_up.bias", (vw,))
        dense(p + "mlp/down", g + "ffn_down.weight", (vh, vw), (p + "mlp_product",))
        exact(p + "mlp/down_bias", g + "ffn_down.bias", (vh,))
        exact(p + "mlp_scale", g + "ls2.weight", (vh,))
    hidden = vc["projection_hidden_dim"]
    dense("vision/projector/up", "mm.1.weight", (hidden, merged), ("vision/merged",))
    dense("vision/projector/out", "mm.2.weight", (config["hidden_size"], hidden),
          ("vision/projector_hidden",))


def _dspark(model, dspark: GGUFFile, drafter: dict, config: dict) -> None:
    """The drafter's matrices as stored (Q8_0 in the release's GGUF), its vectors FP32, the Markov
    embedding BF16 as stored and the fusion weights of its feature mixtures FP32."""

    from .model import Parameter

    h, vocab = config["hidden_size"], config["vocab_size"]
    heads, kv_heads, d = (drafter["num_attention_heads"], drafter["num_key_value_heads"],
                          drafter["head_dim"])
    width, rank = drafter["intermediate_size"], drafter["markov_rank"]
    layers, taps = drafter["num_hidden_layers"], len(drafter["target_layer_ids"])

    def matrix(name, tensor, shape, inputs):
        model.add(Parameter(name, shape, _matrix(dspark, tensor, shape), inputs=tuple(inputs)))

    def vector(name, tensor, shape, stored="fp32"):
        model.add(Parameter(name, shape, _direct(dspark, tensor, shape), direct_format=stored))

    vector("dspark/fuse", "dspark_fuse.weight", (layers + 1, taps))
    vector("dspark/context_norm", "dspark_norm.weight", (h,))
    vector("dspark/final_norm", "output_norm.weight", (h,))
    vector("dspark/mask_embedding", "mask_embd.weight", (h,))
    vector("dspark/markov_embed", "markov_w1.weight", (vocab, rank), stored="bf16")
    vector("dspark/joint", "joint_proj.weight", (3 * rank, 2 * rank + 2 * h))
    vector("dspark/joint_bias", "joint_proj.bias", (3 * rank,))
    vector("dspark/hidden_out", "dspark_hidden_out.weight", (h, rank))
    vector("dspark/hidden_out_bias", "dspark_hidden_out.bias", (h,))
    for b in range(layers):
        g, p = f"blk.{b}.", f"dspark/layers/{b}/"
        a = p + "attention/"
        vector(p + "context_norm", g + "dflare_norm.weight", (h,))
        vector(p + "attention_norm", g + "attn_norm.weight", (h,))
        vector(p + "ffn_norm", g + "ffn_norm.weight", (h,))
        matrix(a + "query", g + "attn_q.weight", (heads * d, h), (p + "attention_input",))
        matrix(a + "key", g + "attn_k.weight", (kv_heads * d, h), (p + "attention_input",))
        matrix(a + "value", g + "attn_v.weight", (kv_heads * d, h), (p + "attention_input",))
        matrix(a + "output", g + "attn_output.weight", (h, heads * d), (a + "output_input",))
        matrix(a + "context_key", g + "attn_k_ctx.weight", (kv_heads * d, h),
               (p + "context_input",))
        matrix(a + "context_value", g + "attn_v_ctx.weight", (kv_heads * d, h),
               (p + "context_input",))
        vector(a + "query_norm", g + "attn_q_norm.weight", (d,))
        vector(a + "key_norm", g + "attn_k_norm.weight", (d,))
        vector(a + "context_key_norm", g + "attn_k_ctx_norm.weight", (d,))
        matrix(p + "mlp/gate", g + "ffn_gate.weight", (width, h), (p + "mlp_input",))
        matrix(p + "mlp/up", g + "ffn_up.weight", (width, h), (p + "mlp_input",))
        matrix(p + "mlp/down", g + "ffn_down.weight", (h, width), (p + "mlp_product",))


def inkling2_gguf(model, recipe, sources):
    """Every matrix keeps its GGUF blocks, the vision tower's its BF16 values or, from a
    quantized mmproj, its blocks; vectors are FP32 (the drafter's Markov embedding BF16, as
    stored). A layer's query, gate and relative projections, which read one input, share one
    parent when the GGUF stores them in one format, as do the drafter's Q, K and V, its context
    K and V, and its MLP's gate and up rows."""

    if model.config.get("architectures") != [TEXT_ARCHITECTURE]:
        raise ValueError("the inkling2_gguf recipe requires an Inkling 2 model")
    formats = {}
    for name, parameter in model.parameters.items():
        # A BF16 matrix (the tower's, or a text matrix the GGUF stores in BF16) keeps its words and
        # multiplies BF16 inputs.
        if parameter.projection and parameter.source.read_encoded is None:
            formats[name] = "bf16"
            recipe.assign(name, format="bf16", method=cast_direct, source=parameter.source,
                          activation_policy="A16Only")
            continue
        if parameter.source.read_encoded is None:
            recipe.assign(name, format=parameter.direct_format, method=cast_direct,
                          source=parameter.source)
            continue
        fmt = formats[name] = parameter.source.read_encoded(0, 1).format
        # The tower decodes its blocks to BF16 rows and keeps its inputs BF16.
        policy = "A16Only" if name.startswith("vision/") else "AllowA8"
        recipe.assign(name, format=fmt, method=import_encoded, source=parameter.source,
                      **({"activation_policy": policy} if parameter.projection else {}))

    def group(names):
        if len({formats[name] for name in names}) == 1:
            recipe.group(names)

    for i in range(model.config["num_hidden_layers"]):
        a = f"text/layers/{i}/attention/"
        group([a + "query", a + "gate", a + "relative"])
    if "dspark" in model.components:
        for b in range(model.components["dspark"]["config"]["num_hidden_layers"]):
            p = f"dspark/layers/{b}/"
            group([p + "attention/query", p + "attention/key", p + "attention/value"])
            group([p + "attention/context_key", p + "attention/context_value"])
            group([p + "mlp/gate", p + "mlp/up"])


RECIPES = {"inkling2_gguf": inkling2_gguf}
