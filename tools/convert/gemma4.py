"""Gemma 4 (Gemma4ForConditionalGeneration) text model from a llama.cpp GGUF, kept in its blocks.

The QAT releases (`google/gemma-4-26B-A4B-it-qat-q4_0-gguf`) store every matrix as ggml Q4_0, the
tied token table as Q6_K and every vector as F32. The recipe imports the blocks row for row: a row of
the artifact is byte for byte a row of the GGUF. Norm gammas are stored as Gemma 4 applies them
(`w`, not `1 + w`); the router's scale vector, the per-expert output scales and the layer scalars
stay F32 vectors that the runtime applies.

    python -m tools.convert --model HF_DIR --recipe gemma4_gguf \\
        --source gguf=gemma-4-26B_q4_0-it.gguf --out gemma-4-26B-A4B-it-qat.ninfer

`HF_DIR` holds the release's `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`chat_template.jinja` and `generation_config.json` (no weights are read from it).

The stored `tokenizer.json` carries Google's glitch-token remediation (see
`remediate_glitch_tokens`): the BPE merges that build the tokens Google flagged in Gemma 4 E2B,
E4B and 12B are removed, so their strings encode as trained subwords.

`--components text,mtp --source mtp=ASSISTANT.gguf` adds the release's multi-token-prediction
assistant (Gemma4AssistantForCausalLM, llama.cpp architecture `gemma4-assistant`) as the `mtp`
component, its blocks imported the same way: a four-block drafter that reads the target's last
sliding-window and full-attention KV and its final hidden state.

`--components text,dflash --source dflash=DIR` adds a DFlash drafter (z-lab's `DFlashDraftModel`
layout in a safetensors directory: the masked block drafter over the target's tapped hidden states)
as the `dflash` component, and `--components text,dspark --source dspark=DIR` a DSpark drafter (the
same backbone with a rank-256 `vanilla` Markov head, `markov_rank` in its config.json) as the
`dspark` component. Both read the target's token rows and head; their projections become Q8_G32 rows
(grouped absmax), their norms and the Markov tables stay BF16 as stored.

`--components text,vision --source vision=MMPROJ.gguf` adds the vision tower (Gemma4VisionModel and
the projection into the text model, llama.cpp's `clip` file with projector `gemma4v`) as the
`vision` component: its BF16 matrices as `bf16`, its norms FP32, the patch embedding (llama.cpp's
convolution [1152, 3, 16, 16], a matrix over (channel, y, x)) and the position table BF16 when
every value is a BF16 value, and the release's processor_config.json as a resource. HF_DIR's
config.json supplies the tower's RoPE, pooling and standardization and the media token ids.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tools.artifact.formats import GGUF_FORMATS_BY_TYPE

from .gguf_blocks import block_source
from .methods import cast_direct, grouped_absmax, import_encoded
from .resources import token_domain
from .sources.gguf import GGUFFile
from .sources.logical import LogicalSource, array_source
from .sources.safetensors import SafetensorsSource, tensor_source
from .ternary import rows

ARCHITECTURES = ("Gemma4ForConditionalGeneration", "Gemma4ForCausalLM")
# The masked block drafters a Gemma 4 artifact can carry: DFlash, and DSpark (DFlash's backbone with
# a Markov head over the drafted chain).
DRAFTERS = ("dflash", "dspark")
DRAFTER_ARCHITECTURE = "Gemma4DFlashDraftModel"
DRAFTER_Q8 = "q8_g32_fp16"
MARKOV_RANKS = (256,)
TEXT_RESOURCES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)


def _int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name}: expected a positive integer")
    return value


def text_config(source: dict) -> dict:
    """The text component's config: Gemma 4's geometry, normalised for src/models/gemma4."""

    architectures = source.get("architectures")
    if not (isinstance(architectures, list) and architectures and architectures[0] in ARCHITECTURES):
        raise ValueError(f"unsupported Gemma 4 architecture {architectures!r}")
    raw = source.get("text_config", source)
    if raw.get("model_type") != "gemma4_text":
        raise ValueError("text.model_type must be gemma4_text")
    if raw.get("hidden_activation") != "gelu_pytorch_tanh":
        raise ValueError("Gemma 4 uses the tanh GELU")
    if raw.get("hidden_size_per_layer_input", 0) or raw.get("num_kv_shared_layers", 0):
        raise ValueError("per-layer embeddings and shared KV layers (E2B/E4B) are not implemented")
    if raw.get("attention_k_eq_v") is not True:
        raise ValueError("the implemented Gemma 4 shares K and V in its full-attention layers")
    out = {"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4_text"}
    for key in ("hidden_size", "vocab_size", "num_hidden_layers", "max_position_embeddings",
                "num_attention_heads", "num_key_value_heads", "head_dim",
                "num_global_key_value_heads", "global_head_dim", "sliding_window",
                "intermediate_size"):
        out[key] = _int(raw.get(key), "text." + key)
    moe = bool(raw.get("enable_moe_block", False))
    out["enable_moe_block"] = moe
    if moe:
        out["num_experts"] = _int(raw.get("num_experts"), "text.num_experts")
        out["top_k_experts"] = _int(raw.get("top_k_experts"), "text.top_k_experts")
        out["moe_intermediate_size"] = _int(raw.get("moe_intermediate_size"),
                                            "text.moe_intermediate_size")
    layers = raw.get("layer_types")
    if (not isinstance(layers, list) or len(layers) != out["num_hidden_layers"]
            or any(k not in ("sliding_attention", "full_attention") for k in layers)):
        raise ValueError("text.layer_types must name every block")
    out["layer_types"] = list(layers)
    rope = raw.get("rope_parameters") or {}
    full, sliding = rope.get("full_attention", {}), rope.get("sliding_attention", {})
    if full.get("rope_type") != "proportional" or sliding.get("rope_type", "default") != "default":
        raise ValueError("Gemma 4 rotates full attention proportionally and sliding attention fully")
    out["rope_theta_full"] = float(full["rope_theta"])
    out["rope_theta_sliding"] = float(sliding["rope_theta"])
    out["partial_rotary_factor_full"] = float(full.get("partial_rotary_factor", 1.0))
    out["rms_norm_eps"] = float(raw.get("rms_norm_eps", 1e-6))
    out["final_logit_softcapping"] = float(raw.get("final_logit_softcapping") or 0.0)
    tied = raw.get("tie_word_embeddings", source.get("tie_word_embeddings", True))
    if tied is not True:
        raise ValueError("Gemma 4 ties its token table and output head")
    out["tie_word_embeddings"] = True
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if key in raw:
            out[key] = raw[key]
    return out


VISION_RESOURCES = ("processor_config.json",)

GLITCH_TOKENS = Path(__file__).with_name("gemma4_glitch_tokens.json")


def remediate_glitch_tokens(data: bytes) -> tuple[bytes, dict]:
    """Google's glitch-token remediation of the Gemma 4 tokenizer, for every model size.

    The Gemma 4 releases share one tokenizer. Google patched it in E2B, E4B and 12B
    (gemma4_glitch_tokens.json names the pull requests): tokens whose embeddings collapsed to
    nearly one vector lose the BPE merges that build them, and the one single-character token
    among them is renamed to a placeholder, so its character falls back to its UTF-8 bytes. Token
    ids and the vocabulary size stay. Each patch lists the tokens flagged in its own model; the
    union applies here, to the 26B-A4B and 31B releases that Google has not patched, whose token
    tables hold all 261 of these tokens in the same collapsed cluster. The edit removes the listed
    merges still present and renames the token where the original character still names it, then
    writes the file as the release formats it (Python's JSON, two-space indent), so applying one
    model's list alone reproduces Google's patched file byte for byte. A tokenizer without any of
    them (already remediated) is returned unchanged."""

    patch = json.loads(GLITCH_TOKENS.read_text(encoding="utf-8"))
    tokenizer = json.loads(data)
    model = tokenizer.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise ValueError("tokenizer.json: Gemma 4 uses a BPE model")
    merges, vocab = model["merges"], model["vocab"]
    removed = {tuple(pair) for pair in patch["removed_merges"]}

    def pair(merge):
        return tuple(merge) if isinstance(merge, list) else tuple(merge.split(" ", 1))

    kept = [merge for merge in merges if pair(merge) not in removed]
    names = {index: token for token, index in vocab.items()}
    renames = {}
    for entry in patch["renamed"]:
        name = names.get(entry["id"])
        if name == entry["token"]:
            renames[name] = entry["placeholder"]
        elif name not in (None, entry["placeholder"]):
            raise ValueError(f"tokenizer.json: token {entry['id']} is {name!r}, not Gemma 4's "
                             f"{entry['token']!r}")
    edit = {"removed_merges": len(merges) - len(kept), "renamed_tokens": len(renames)}
    if not edit["removed_merges"] and not renames:
        return data, edit
    model["merges"] = kept
    # The renamed entry keeps its place in the vocabulary.
    model["vocab"] = {renames.get(token, token): index for token, index in vocab.items()}
    return json.dumps(tokenizer, ensure_ascii=False, indent=2).encode("utf-8"), edit


def _load_resources(root: Path, vocab: int, overrides: dict, vision: bool):
    references, payloads = {"text": {}}, {}
    roles = [("text", role) for role in TEXT_RESOURCES]
    roles += [("vision", role) for role in VISION_RESOURCES] if vision else []
    for component, role in roles:
        path = Path(overrides[role]) if role in overrides else root / role
        data = path.read_bytes()
        if not data:
            raise ValueError(f"{path}: resource is empty")
        object_id = f"resource/{component}/{role}"
        references.setdefault(component, {})[role] = object_id
        payloads[object_id] = data
    payloads["resource/text/tokenizer.json"], edit = remediate_glitch_tokens(
        payloads["resource/text/tokenizer.json"])
    tokenizer = json.loads(payloads["resource/text/tokenizer.json"])
    tokenizer_config = json.loads(payloads["resource/text/tokenizer_config.json"])
    count, special = token_domain(tokenizer, tokenizer_config, vocab)
    return references, payloads, count, special, edit


def _direct(gguf: GGUFFile, tensor: str, shape: tuple[int, ...]) -> LogicalSource:
    values = gguf.read_direct(tensor).astype(np.float32).reshape(shape)
    return array_source(torch.from_numpy(np.ascontiguousarray(values)), tensor)


def _matrix(gguf: GGUFFile, tensor: str, shape: tuple[int, int]) -> LogicalSource:
    info = gguf.info(tensor)
    if info.type_id not in GGUF_FORMATS_BY_TYPE:
        raise ValueError(f"{tensor}: expected ggml block rows, found {info.type_name}")
    stored = (int(np.prod(info.shape[:-1])), info.shape[-1])
    if stored != shape:
        raise ValueError(f"{tensor}: stored {info.shape}, expected rows x columns {shape}")
    return block_source(gguf, tensor, shape, rows())


def _bf16_matrix(gguf: GGUFFile, tensor: str, shape: tuple[int, int]) -> LogicalSource:
    info = gguf.info(tensor)
    if info.type_name not in ("BF16", "F32", "F16"):
        raise ValueError(f"{tensor}: expected a BF16 (or F32/F16) matrix, found {info.type_name}")
    return _exact_bf16(gguf, tensor, shape)


def _exact_bf16(gguf: GGUFFile, tensor: str, shape: tuple[int, ...]) -> LogicalSource:
    """`tensor` as FP32 values that BF16 holds exactly; a wider value is an error, since the tower
    runs these weights in BF16 (llama.cpp writes the release's BF16 patch embedding and position
    table as F32)."""

    values = torch.from_numpy(
        np.ascontiguousarray(gguf.read_direct(tensor).astype(np.float32).reshape(shape)))
    if not torch.equal(values.to(torch.bfloat16).to(torch.float32), values):
        raise ValueError(f"{tensor}: values are not BF16 values")
    return array_source(values, tensor)


def vision_config(mmproj: GGUFFile, source: dict, target: dict) -> dict:
    """The `vision` component's config from a llama.cpp `clip` GGUF with projector `gemma4v` and
    the release's config.json, checked against each other and against the text model."""

    kv = mmproj.kv
    if kv.get("general.architecture") != "clip" or not kv.get("clip.has_vision_encoder"):
        raise ValueError("the vision source must be a llama.cpp clip GGUF with a vision encoder")
    if kv.get("clip.vision.projector_type") != "gemma4v":
        raise ValueError("the vision source's projector must be gemma4v (Gemma 4)")
    raw = source.get("vision_config")
    if not isinstance(raw, dict) or raw.get("model_type") != "gemma4_vision":
        raise ValueError("config.json has no gemma4_vision vision_config")
    if raw.get("hidden_activation") != "gelu_pytorch_tanh":
        raise ValueError("the Gemma 4 vision MLP uses the tanh GELU")
    if raw.get("use_clipped_linears"):
        raise ValueError("clipped vision linears are not implemented")
    rope = raw.get("rope_parameters") or {}
    if rope.get("rope_type", "default") not in ("default", "axial"):
        raise ValueError("the Gemma 4 vision tower rotates axially")
    v = "clip.vision."
    out = {
        "architectures": ["Gemma4VisionModel"],
        "hidden_size": _int(kv.get(v + "embedding_length"), "vision.embedding_length"),
        "intermediate_size": _int(kv.get(v + "feed_forward_length"), "vision.feed_forward_length"),
        "num_hidden_layers": _int(kv.get(v + "block_count"), "vision.block_count"),
        "num_attention_heads": _int(kv.get(v + "attention.head_count"), "vision.head_count"),
        "patch_size": _int(kv.get(v + "patch_size"), "vision.patch_size"),
        "projection_dim": _int(kv.get(v + "projection_dim"), "vision.projection_dim"),
        "rms_norm_eps": float(kv.get(v + "attention.layer_norm_epsilon", 1e-6)),
        "head_dim": _int(raw.get("head_dim"), "vision_config.head_dim"),
        "pooling_kernel_size": _int(raw.get("pooling_kernel_size"),
                                    "vision_config.pooling_kernel_size"),
        "position_embedding_size": _int(raw.get("position_embedding_size"),
                                        "vision_config.position_embedding_size"),
        "rope_theta": float(rope.get("rope_theta", 100.0)),
        "standardize": bool(raw.get("standardize", False)),
        "default_output_length": _int(raw.get("default_output_length", 280),
                                      "vision_config.default_output_length"),
    }
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "patch_size"):
        if raw.get(key) != out[key]:
            raise ValueError(f"vision {key}: GGUF {out[key]} differs from config.json "
                             f"{raw.get(key)}")
    if out["head_dim"] * out["num_attention_heads"] != out["hidden_size"]:
        raise ValueError("vision heads x head_dim must be the hidden size")
    if out["head_dim"] % 4:
        raise ValueError("vision head_dim must split into two rotated halves of pairs")
    if out["projection_dim"] != target["hidden_size"]:
        raise ValueError("the vision projection must reach the text hidden size")
    if raw.get("num_key_value_heads", out["num_attention_heads"]) != out["num_attention_heads"]:
        raise ValueError("the Gemma 4 vision tower has as many key/value heads as query heads")
    for key in ("image_token_id", "video_token_id", "boi_token_id", "eoi_token_id"):
        out[key] = _int(source.get(key), key)
    text = source.get("text_config", {})
    out["bidirectional_attention"] = text.get("use_bidirectional_attention") == "vision"
    if text.get("use_bidirectional_attention") not in (None, "vision"):
        raise ValueError("only the vision-block bidirectional attention is implemented")
    return out


def assistant_config(mtp: GGUFFile, target: dict) -> dict:
    """The `mtp` component's config from a gemma4-assistant GGUF, checked against the target whose
    KV and hidden state it reads."""

    kv = mtp.kv
    if kv.get("general.architecture") != "gemma4-assistant":
        raise ValueError("the MTP source must be a gemma4-assistant GGUF")
    a = "gemma4-assistant."
    layers = kv.get(a + "block_count")
    sliding = kv.get(a + "attention.sliding_window_pattern")
    if not isinstance(sliding, list) or len(sliding) != layers:
        raise ValueError("gemma4-assistant: sliding_window_pattern must name every block")
    heads_kv = kv.get(a + "attention.head_count_kv")
    if not isinstance(heads_kv, list) or len(heads_kv) != layers:
        raise ValueError("gemma4-assistant: head_count_kv must give every block")
    if kv.get(a + "attention.shared_kv_layers") != layers:
        raise ValueError("gemma4-assistant: every block must read the target's KV")
    vocab, hidden = mtp.info("token_embd.weight").shape
    out = {
        "architectures": ["Gemma4AssistantForCausalLM"],
        "hidden_size": int(kv[a + "embedding_length"]),
        "backbone_hidden_size": int(kv[a + "embedding_length_out"]),
        "intermediate_size": int(kv[a + "feed_forward_length"]),
        "num_hidden_layers": int(layers),
        "layer_types": ["sliding_attention" if s else "full_attention" for s in sliding],
        "num_attention_heads": int(kv[a + "attention.head_count"]),
        "num_key_value_heads": int(next(h for h, s in zip(heads_kv, sliding) if s)),
        "num_global_key_value_heads": int(next(h for h, s in zip(heads_kv, sliding) if not s)),
        "head_dim": int(kv[a + "attention.key_length_swa"]),
        "global_head_dim": int(kv[a + "attention.key_length"]),
        "sliding_window": int(kv[a + "attention.sliding_window"]),
        "rms_norm_eps": float(kv[a + "attention.layer_norm_rms_epsilon"]),
        "vocab_size": int(vocab),
    }
    if hidden != out["hidden_size"]:
        raise ValueError("gemma4-assistant: token table width differs from embedding_length")
    for key in ("vocab_size", "num_attention_heads", "num_key_value_heads",
                "num_global_key_value_heads", "head_dim", "global_head_dim", "sliding_window"):
        if out[key] != target[key]:
            raise ValueError(f"gemma4-assistant: {key} {out[key]} differs from the target's "
                             f"{target[key]}")
    if out["backbone_hidden_size"] != target["hidden_size"]:
        raise ValueError("gemma4-assistant: embedding_length_out must be the target hidden size")
    if not any(sliding) or all(sliding):
        raise ValueError("gemma4-assistant: needs sliding-window and full-attention blocks")
    return out


def drafter_config(store: SafetensorsSource, target: dict, component: str) -> dict:
    """The `dflash` or `dspark` component's config from a DFlashDraftModel directory, checked
    against the target whose token rows, head and tapped hidden states it reads. `dspark` requires
    the vanilla Markov head (`markov_rank` 256, `markov_head_type` vanilla), `dflash` its absence.
    The fields are flat; layer_types names each layer's attention."""

    source, where = store.config, component
    if source.get("architectures") != ["DFlashDraftModel"] or source.get("model_type") != "qwen3":
        raise ValueError(f"--source {component} must be a DFlashDraftModel (qwen3) directory")
    for key, expected in (("hidden_act", "silu"), ("attention_bias", False),
                          ("hidden_size", target["hidden_size"]),
                          ("vocab_size", target["vocab_size"]),
                          ("num_target_layers", target["num_hidden_layers"])):
        if source.get(key, False if key == "attention_bias" else None) != expected:
            raise ValueError(f"{where}.{key} is {source.get(key)!r}, expected {expected!r}")
    if source.get("tie_word_embeddings", True) is not True:
        raise ValueError(f"{where}: the drafter reads the target's tied token rows and head")
    if source.get("rope_scaling") not in (None, {}) or (
            (source.get("rope_parameters") or {}).get("rope_type", "default") != "default"):
        raise ValueError(f"{where}: only the default rotation is implemented")
    out = {"architectures": [DRAFTER_ARCHITECTURE], "model_type": "gemma4_dflash"}
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "vocab_size", "num_target_layers",
                "block_size", "sliding_window"):
        out[key] = _int(source.get(key), f"{where}.{key}")
    layers = source.get("layer_types")
    if (not isinstance(layers, list) or len(layers) != out["num_hidden_layers"]
            or any(kind not in ("sliding_attention", "full_attention") for kind in layers)):
        raise ValueError(f"{where}.layer_types must name every layer's attention")
    out["layer_types"] = list(layers)
    out["rms_norm_eps"] = float(source.get("rms_norm_eps", 1e-6))
    theta = (source.get("rope_parameters") or {}).get("rope_theta", source.get("rope_theta"))
    if not isinstance(theta, (int, float)) or theta <= 0:
        raise ValueError(f"{where}.rope_theta must be positive")
    out["rope_theta"] = float(theta)
    draft = source.get("dflash_config")
    if not isinstance(draft, dict):
        raise ValueError(f"{where}: config.json has no dflash_config")
    taps = draft.get("target_layer_ids")
    if (not isinstance(taps, list) or not taps or taps != sorted(set(taps))
            or any(type(v) is not int or not 0 <= v < target["num_hidden_layers"] for v in taps)):
        raise ValueError(f"{where}.dflash_config.target_layer_ids must name distinct target "
                         "blocks in order")
    out["target_layer_ids"] = list(taps)
    mask = draft.get("mask_token_id")
    if type(mask) is not int or not 0 <= mask < target["vocab_size"]:
        raise ValueError(f"{where}.dflash_config.mask_token_id lies outside the vocabulary")
    out["mask_token_id"] = mask
    # The drafter's logits are the target head's under the target's final soft cap.
    cap = float(source.get("final_logit_softcapping") or 0.0)
    if cap != target["final_logit_softcapping"]:
        raise ValueError(f"{where}.final_logit_softcapping is {cap}, the target's "
                         f"{target['final_logit_softcapping']}")
    out["logit_softcap"] = cap
    rank = source.get("markov_rank", 0) or 0
    if component == "dspark":
        if rank not in MARKOV_RANKS or source.get("markov_head_type", "vanilla") != "vanilla":
            raise ValueError("--source dspark must carry the vanilla Markov head of rank 256 "
                             "(markov_rank, markov_head_type)")
        if source.get("confidence_head_with_markov") or source.get("confidence_head"):
            raise ValueError("dspark: a confidence head is not implemented")
    elif rank:
        raise ValueError("--source dflash carries a Markov head: convert it as dspark")
    out["markov_rank"] = int(rank)
    return out


def _drafter_tensors(config: dict) -> dict:
    """The checkpoint tensor of every drafter parameter (suffix after the component's prefix),
    with its logical shape and inputs (projections only)."""

    h, d = config["hidden_size"], config["head_dim"]
    q, kv = config["num_attention_heads"] * d, config["num_key_value_heads"] * d
    inner, vocab = config["intermediate_size"], config["vocab_size"]
    taps = len(config["target_layer_ids"])
    out = {"feature_projection": ("fc.weight", (h, taps * h), ("target_features",)),
           "context_norm": ("hidden_norm.weight", (h,), ()),
           "final_norm": ("norm.weight", (h,), ())}
    if config["markov_rank"]:
        r = config["markov_rank"]
        out["markov/embedding"] = ("markov_head.markov_w1.weight", (vocab, r), ())
        out["markov/output"] = ("markov_head.markov_w2.weight", (vocab, r), ())
    for i in range(config["num_hidden_layers"]):
        g, p = f"layers.{i}.", f"layers/{i}/"
        a = p + "attention/"
        out |= {
            p + "input_norm": (g + "input_layernorm.weight", (h,), ()),
            p + "post_attention_norm": (g + "post_attention_layernorm.weight", (h,), ()),
            a + "query_norm": (g + "self_attn.q_norm.weight", (d,), ()),
            a + "key_norm": (g + "self_attn.k_norm.weight", (d,), ()),
            a + "query": (g + "self_attn.q_proj.weight", (q, h), (p + "attention_input",)),
            a + "key": (g + "self_attn.k_proj.weight", (kv, h), (p + "attention_input",)),
            a + "value": (g + "self_attn.v_proj.weight", (kv, h), (p + "attention_input",)),
            a + "output": (g + "self_attn.o_proj.weight", (h, q), (a + "output_input",)),
            p + "mlp/gate": (g + "mlp.gate_proj.weight", (inner, h), (p + "mlp_input",)),
            p + "mlp/up": (g + "mlp.up_proj.weight", (inner, h), (p + "mlp_input",)),
            p + "mlp/down": (g + "mlp.down_proj.weight", (h, inner), (p + "mlp_product",)),
        }
    return out


def _drafter(model, store: SafetensorsSource, component: str, config: dict) -> None:
    """A drafter's parameters as the checkpoint stores them (BF16); a checkpoint with any other
    tensor (its own token rows or head, a gate or confidence head) is refused."""

    from .model import Parameter

    wanted = _drafter_tensors(config)
    stored = {tensor for tensor, _, _ in wanted.values()}
    extra = sorted(set(store.weight_map) - stored)
    if extra:
        raise ValueError(f"{component}: tensors the drafter does not implement: {extra[:6]}")
    for suffix, (tensor, shape, inputs) in wanted.items():
        if not store.has(tensor):
            raise ValueError(f"{component}: the checkpoint has no {tensor}")
        info = store.describe(tensor)
        if info.dtype != "BF16" or tuple(info.shape) != shape:
            raise ValueError(f"{component}/{tensor}: stored {info.dtype} {info.shape}, expected "
                             f"BF16 {shape}")
        name = f"{component}/{suffix}"
        model.add(Parameter(name, shape, tensor_source(store, tensor, shape),
                            inputs=tuple(f"{component}/{i}" for i in inputs),
                            residency=component))


def build_model(base, gguf: GGUFFile, *, mtp: GGUFFile | None = None,
                vision: GGUFFile | None = None, drafters: dict | None = None,
                resource_overrides=None):
    """The text component of a Gemma 4 GGUF, every parameter's source the GGUF, with `mtp` the
    assistant's component from its own GGUF, with `vision` the vision tower's from an mmproj
    GGUF and with `drafters` ({"dflash" or "dspark": SafetensorsSource}) masked block drafters
    from their safetensors directories."""

    from .model import Model, Parameter

    config = text_config(base.config)
    h, vocab = config["hidden_size"], config["vocab_size"]
    references, payloads, count, special, edit = _load_resources(
        base.root, vocab, dict(resource_overrides or {}), vision is not None)
    records = {"text": {"config": config, "resources": references["text"]}}
    if mtp is not None:
        records["mtp"] = {"config": assistant_config(mtp, config), "target": "text"}
    if vision is not None:
        records["vision"] = {"config": vision_config(vision, base.config, config),
                             "resources": references["vision"], "target": "text"}
    drafters = dict(drafters or {})
    if set(drafters) - set(DRAFTERS):
        raise ValueError(f"Gemma 4 drafters are {DRAFTERS}")
    for component, store in drafters.items():
        records[component] = {"config": drafter_config(store, config, component),
                              "target": "text"}
    model = Model(records, resources=payloads, token_count=count, special_token_ids=special,
                  resource_edits={"tokenizer.json": {"glitch_tokens": edit}})
    for component, store in drafters.items():
        _drafter(model, store, component, records[component]["config"])

    def matrix(name, tensor, shape, inputs, source=gguf):
        model.add(Parameter(name, shape, _matrix(source, tensor, shape), inputs=tuple(inputs),
                            residency=name.split("/", 1)[0]))

    def vector(name, tensor, shape, source=gguf):
        model.add(Parameter(name, shape, _direct(source, tensor, shape), direct_format="fp32",
                            residency=name.split("/", 1)[0]))

    # The tied table: the head reads it as a projection, the embedding gathers its rows.
    matrix("text/output_head", "token_embd.weight", (vocab, h), ("text/final_hidden",))
    vector("text/final_norm", "output_norm.weight", (h,))
    for i, kind in enumerate(config["layer_types"]):
        g, p = f"blk.{i}.", f"text/layers/{i}/"
        full = kind == "full_attention"
        d = config["global_head_dim"] if full else config["head_dim"]
        kv = config["num_global_key_value_heads"] if full else config["num_key_value_heads"]
        q = config["num_attention_heads"] * d
        a = p + "attention/"
        matrix(a + "query", g + "attn_q.weight", (q, h), (p + "mixer_input",))
        matrix(a + "key", g + "attn_k.weight", (kv * d, h), (p + "mixer_input",))
        if not full:
            matrix(a + "value", g + "attn_v.weight", (kv * d, h), (p + "mixer_input",))
        matrix(a + "output", g + "attn_output.weight", (h, q), (p + "attention_output",))
        vector(a + "query_norm", g + "attn_q_norm.weight", (d,))
        vector(a + "key_norm", g + "attn_k_norm.weight", (d,))
        vector(p + "input_norm", g + "attn_norm.weight", (h,))
        vector(p + "post_attention_norm", g + "post_attention_norm.weight", (h,))
        vector(p + "ffn_norm", g + "ffn_norm.weight", (h,))
        vector(p + "post_ffn_norm", g + "post_ffw_norm.weight", (h,))
        vector(p + "layer_scalar", g + "layer_output_scale.weight", (1,))
        width = config["intermediate_size"]
        matrix(p + "mlp/gate", g + "ffn_gate.weight", (width, h), (p + "mlp_input",))
        matrix(p + "mlp/up", g + "ffn_up.weight", (width, h), (p + "mlp_input",))
        matrix(p + "mlp/down", g + "ffn_down.weight", (h, width), (p + "mlp_product",))
        if config["enable_moe_block"]:
            e, w = config["num_experts"], config["moe_intermediate_size"]
            m = p + "moe/"
            vector(p + "mlp/post_norm", g + "post_ffw_norm_1.weight", (h,))
            vector(m + "pre_norm", g + "pre_ffw_norm_2.weight", (h,))
            vector(m + "post_norm", g + "post_ffw_norm_2.weight", (h,))
            vector(m + "router", g + "ffn_gate_inp.weight", (e, h))
            vector(m + "router_scale", g + "ffn_gate_inp.scale", (h,))
            vector(m + "expert_scale", g + "ffn_down_exps.scale", (e,))
            matrix(m + "gate_up", g + "ffn_gate_up_exps.weight", (e * 2 * w, h), (m + "input",))
            matrix(m + "down", g + "ffn_down_exps.weight", (e * h, w), (m + "product",))
    if mtp is not None:
        assistant = records["mtp"]["config"]
        ah, b = assistant["hidden_size"], assistant["backbone_hidden_size"]
        # The tied table: the drafter's head reads it; its inputs are the target's embeddings.
        matrix("mtp/output_head", "token_embd.weight", (vocab, ah), ("mtp/final_hidden",), mtp)
        vector("mtp/final_norm", "output_norm.weight", (ah,), mtp)
        matrix("mtp/pre_projection", "nextn.pre_projection.weight", (ah, 2 * b),
               ("mtp/stem_input",), mtp)
        matrix("mtp/post_projection", "nextn.post_projection.weight", (b, ah),
               ("mtp/final_hidden",), mtp)
        for i, kind in enumerate(assistant["layer_types"]):
            g, p = f"blk.{i}.", f"mtp/layers/{i}/"
            d = assistant["global_head_dim" if kind == "full_attention" else "head_dim"]
            q = assistant["num_attention_heads"] * d
            a = p + "attention/"
            matrix(a + "query", g + "attn_q.weight", (q, ah), (p + "mixer_input",), mtp)
            matrix(a + "output", g + "attn_output.weight", (ah, q), (p + "attention_output",),
                   mtp)
            vector(a + "query_norm", g + "attn_q_norm.weight", (d,), mtp)
            vector(p + "input_norm", g + "attn_norm.weight", (ah,), mtp)
            vector(p + "post_attention_norm", g + "post_attention_norm.weight", (ah,), mtp)
            vector(p + "ffn_norm", g + "ffn_norm.weight", (ah,), mtp)
            vector(p + "post_ffn_norm", g + "post_ffw_norm.weight", (ah,), mtp)
            vector(p + "layer_scalar", g + "layer_output_scale.weight", (1,), mtp)
            width = assistant["intermediate_size"]
            matrix(p + "mlp/gate", g + "ffn_gate.weight", (width, ah), (p + "mlp_input",), mtp)
            matrix(p + "mlp/up", g + "ffn_up.weight", (width, ah), (p + "mlp_input",), mtp)
            matrix(p + "mlp/down", g + "ffn_down.weight", (ah, width), (p + "mlp_product",), mtp)
    if vision is not None:
        vc = records["vision"]["config"]
        vh, vw, patch = vc["hidden_size"], vc["intermediate_size"], vc["patch_size"]

        def dense(name, tensor, shape, inputs):
            model.add(Parameter(name, shape, _bf16_matrix(vision, tensor, shape),
                                inputs=tuple(inputs), residency="vision"))

        def table(name, tensor, shape):
            model.add(Parameter(name, shape, _exact_bf16(vision, tensor, shape),
                                direct_format="bf16", residency="vision"))

        def norm(name, tensor, shape):
            vector(name, tensor, shape, vision)

        dense("vision/patch_embedding", "v.patch_embd.weight", (vh, 3 * patch * patch),
              ("vision/patches",))
        table("vision/position_table", "v.position_embd.weight",
              (2 * vc["position_embedding_size"], vh))
        if vc["standardize"]:
            norm("vision/std_bias", "v.std_bias", (vh,))
            norm("vision/std_scale", "v.std_scale", (vh,))
        dense("vision/projection", "mm.input_projection.weight", (vc["projection_dim"], vh),
              ("vision/soft_tokens",))
        d = vc["head_dim"]
        for i in range(vc["num_hidden_layers"]):
            g, p = f"v.blk.{i}.", f"vision/layers/{i}/"
            a = p + "attention/"
            for part, tensor in (("query", "attn_q"), ("key", "attn_k"), ("value", "attn_v")):
                dense(a + part, g + tensor + ".weight", (vh, vh), (p + "mixer_input",))
            dense(a + "output", g + "attn_out.weight", (vh, vh), (p + "attention_output",))
            norm(a + "query_norm", g + "attn_q_norm.weight", (d,))
            norm(a + "key_norm", g + "attn_k_norm.weight", (d,))
            norm(p + "input_norm", g + "ln1.weight", (vh,))
            norm(p + "post_attention_norm", g + "attn_post_norm.weight", (vh,))
            norm(p + "ffn_norm", g + "ln2.weight", (vh,))
            norm(p + "post_ffn_norm", g + "ffn_post_norm.weight", (vh,))
            dense(p + "mlp/gate", g + "ffn_gate.weight", (vw, vh), (p + "mlp_input",))
            dense(p + "mlp/up", g + "ffn_up.weight", (vw, vh), (p + "mlp_input",))
            dense(p + "mlp/down", g + "ffn_down.weight", (vh, vw), (p + "mlp_product",))
    return model


def gemma4_gguf(model, recipe, sources):
    """Every matrix keeps its GGUF blocks; vectors are FP32. Q, K and V of a layer share one parent
    when the GGUF stores them in one format (a mixed-format quantization such as Q4_K_M keeps V
    apart), and the dense FFN's gate and up rows share one. A DFlash or DSpark drafter's
    projections become Q8_G32 rows (grouped absmax, FP16 scales); its norms and Markov tables stay
    BF16 as stored."""

    if model.config.get("architectures") != ["Gemma4ForCausalLM"]:
        raise ValueError("the gemma4_gguf recipe requires a Gemma 4 model")
    formats = {}
    for name, parameter in model.parameters.items():
        if name.split("/", 1)[0] in DRAFTERS:
            if parameter.projection:
                recipe.assign(name, format=DRAFTER_Q8, method=grouped_absmax,
                              source=parameter.source)
            else:
                recipe.assign(name, format="bf16", method=cast_direct, source=parameter.source)
            continue
        if name.startswith("vision/"):
            # The tower's BF16 weights as they are; its norms and standardization FP32.
            fmt = formats[name] = "bf16" if parameter.projection else parameter.direct_format
            recipe.assign(name, format=fmt, method=cast_direct, source=parameter.source,
                          **({"activation_policy": "A16Only"} if parameter.projection else {}))
        elif parameter.projection:
            fmt = formats[name] = parameter.source.read_encoded(0, 1).format
            recipe.assign(name, format=fmt, method=import_encoded, source=parameter.source,
                          activation_policy="AllowA8")
        else:
            recipe.assign(name, format="fp32", method=cast_direct, source=parameter.source)
    for i, kind in enumerate(model.config["layer_types"]):
        a = f"text/layers/{i}/attention/"
        group = [a + "query", a + "key"] + ([] if kind == "full_attention" else [a + "value"])
        if len({formats[name] for name in group}) == 1:
            recipe.group(group)
    mlps = [f"text/layers/{i}/mlp/" for i in range(len(model.config["layer_types"]))]
    if "mtp" in model.components:
        mlps += [f"mtp/layers/{i}/mlp/"
                 for i in range(model.components["mtp"]["config"]["num_hidden_layers"])]
    if "vision" in model.components:
        layers = model.components["vision"]["config"]["num_hidden_layers"]
        for i in range(layers):
            a = f"vision/layers/{i}/attention/"
            recipe.group([a + "query", a + "key", a + "value"])
        mlps += [f"vision/layers/{i}/mlp/" for i in range(layers)]
    for p in mlps:
        if formats[p + "gate"] != formats[p + "up"]:
            raise ValueError(f"{p}: gate and up are stored in different formats")
        recipe.group([p + "gate", p + "up"])


RECIPES = {"gemma4_gguf": gemma4_gguf}
