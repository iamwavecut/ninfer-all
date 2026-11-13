"""Tencent DSpine drafters as standalone NInfer artifacts (docs/maintainer/dspine.md).

A DSpine drafter (`DFlashDraftModel` whose `dflash_config` names `selected_prefix_code_bridge`) is
a masked-block drafter of the DFlash family: Qwen3-style sliding-window layers over a ring of the
target's tapped hidden states, causal inside the block, with a code bridge that writes into the
residual after every layer, and a serial refinement of the last write that chains the block's
candidates. The artifact holds the drafter alone, in its `dspine` component. The target's token
embedding and output head stay in the target's artifact; the engine derives the head-dependent
table (each token's head row through the last write) when it loads the two together.

The codebook is not stored. Its only runtime use is the anchor's code, and the anchor's sender
rows are those of the stored sender table: sender_table[v] = BF16(BF16(sqrt(C) * codebook[v]) .
[predecessor_gate; prev]^T), accumulated in FP32 as the reference's build_serial_tables does.
"""

from __future__ import annotations

from math import sqrt

import torch

from .config_fields import _f32, _fixed, _positive, _rope_source
from .methods import grouped_absmax
from .model import Model
from .qwen3_5 import _Builder
from .sources.logical import LogicalSource, transpose_source
from .sources.safetensors import SafetensorsSource, tensor_source

ARCHITECTURES = ("DFlashDraftModel",)
COMPONENT = "dspine"
_BRIDGE = "selected_prefix_bridge."
# The scalar version markers the reference refuses unless they are 1.
_VERSIONS = (
    "draft_block_causal_version",
    "last_write_refine_version",
    _BRIDGE + "independent_write_version",
    _BRIDGE + "terminal_disabled_version",
)


def is_dspine(config: dict) -> bool:
    """Whether a checkpoint config describes a DSpine drafter."""
    draft = config.get("dflash_config")
    return (
        (config.get("architectures") or [None])[0] in ARCHITECTURES
        and isinstance(draft, dict)
        and isinstance(draft.get("selected_prefix_code_bridge"), dict)
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("DSpine: " + message)


def dspine_config(raw: dict) -> dict:
    """The `dspine` component's config from the checkpoint's, refusing what the engine lacks."""
    _require(is_dspine(raw), "expected DFlashDraftModel with selected_prefix_code_bridge")
    label = "dspine"
    for key, value in (
        ("model_type", "qwen3"),
        ("hidden_act", "silu"),
        ("attention_bias", False),
        ("tie_word_embeddings", False),
    ):
        _fixed(raw, key, value, label)
    draft = raw["dflash_config"]
    for key in (
        "selector_rank",
        "selector_top_k",
        "projector_type",
        "attention_sink_bias",
        "conv_kernel_size",
        "conv_group_size",
        "attention_value_scale",
        "selected_prefix_feature_reliability",
        "selected_prefix_feature_next_token_rows",
        # The terminal token residual adds a per-token term to the sender table's values
        # (the reference's fold_terminal_sender_table); the converter does not fold it.
        "selected_prefix_terminal_residual_rank",
    ):
        _require(not draft.get(key), f"unsupported dflash_config.{key}")
    for key, value in (
        ("draft_block_causal", True),
        ("selected_prefix_terminal", False),
        ("selected_prefix_decode_mode", "serial"),
        ("selected_prefix_feature_writes", True),
        ("selected_prefix_feature_write_mode", "independent"),
    ):
        _require(
            draft.get(key) == value and type(draft.get(key)) is type(value),
            f"dflash_config.{key} must be {value!r}",
        )
    result = {"architectures": ["DSpineDraftModel"], "model_type": "qwen3"}
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "num_hidden_layers",
        "max_position_embeddings",
        "vocab_size",
        "num_target_layers",
        "block_size",
    ):
        result[key] = _positive(raw.get(key), label + "." + key)
    if raw.get("draft_vocab_size") is not None:
        _require(raw["draft_vocab_size"] == result["vocab_size"], "draft_vocab_size != vocab_size")
    _require(
        result["num_attention_heads"] % result["num_key_value_heads"] == 0
        and result["head_dim"] % 2 == 0,
        "invalid attention geometry",
    )
    _require(2 <= result["block_size"] <= 16, "block_size must lie in [2, 16]")
    result["rms_norm_eps"] = _f32(raw.get("rms_norm_eps", 1e-6), label + ".rms_norm_eps")
    rope = _rope_source(raw, label)
    result["rope_parameters"] = {
        "rope_theta": _f32(
            rope.get("rope_theta", raw.get("rope_theta", 10_000_000)), label + ".rope_theta"
        )
    }
    layers = raw.get("layer_types")
    _require(
        isinstance(layers, list)
        and len(layers) == result["num_hidden_layers"]
        and all(kind == "sliding_attention" for kind in layers),
        "every drafter layer must be sliding_attention",
    )
    result["layer_types"] = list(layers)
    result["sliding_window"] = _positive(raw.get("sliding_window"), label + ".sliding_window")
    taps = draft.get("target_layer_ids")
    _require(
        isinstance(taps, list)
        and taps
        and all(type(i) is int and 0 <= i < result["num_target_layers"] for i in taps)
        and len(set(taps)) == len(taps),
        "invalid target_layer_ids",
    )
    mask = draft.get("mask_token_id")
    _require(type(mask) is int and 0 <= mask < result["vocab_size"], "mask token outside vocab")
    features = draft.get("selected_prefix_feature_layers")
    _require(
        features == list(range(1, result["num_hidden_layers"] + 1)),
        "every drafter layer must carry a bridge write",
    )
    bridge = draft["selected_prefix_code_bridge"]
    code_size = _positive(bridge.get("code_size"), label + ".code_size")
    rank = _positive(bridge.get("rank"), label + ".rank")
    _require(code_size % 8 == 0 and rank % 32 == 0 and rank <= 512, "bridge code/rank geometry")
    bridge_eps = _f32(bridge.get("eps"), label + ".bridge.eps")
    widths = []
    refine = draft.get("selected_prefix_last_write_refine")
    if refine is not None:
        _require(
            isinstance(refine, dict)
            and set(refine) == {"kind", "version", "head_topk", "weight"}
            and refine["kind"] == "exact_sender"
            and refine["version"] == 1,
            "selected_prefix_last_write_refine must be exact_sender version 1",
        )
        widths.append(refine["head_topk"])
    second = raw.get("code_bridge_second_pass")
    if second is not None:
        _require(
            isinstance(second, dict)
            and set(second) <= {"mode", "first_exact", "head_topk"}
            and second.get("mode") == "serial_table"
            and second.get("first_exact") == result["num_hidden_layers"] - 1,
            "code_bridge_second_pass must refine the last write by serial_table",
        )
        widths.append(second.get("head_topk"))
    _require(widths and len(set(widths)) == 1, "the refinement widths are missing or disagree")
    _require(widths[0] == 16, "the refinement takes the head's top 16")
    result["dspine_config"] = {
        "target_layer_ids": list(taps),
        "mask_token_id": mask,
        "code_size": code_size,
        "rank": rank,
        "bridge_eps": bridge_eps,
        "head_top_k": widths[0],
    }
    return result


def _check_markers(store: SafetensorsSource) -> None:
    for name in _VERSIONS:
        info = store.describe(name)
        _require(info.dtype in ("I64", "I32") and info.shape == (), f"{name} must be a scalar")
        _require(int(store.read_flat(name)[0]) == 1, f"unsupported {name}")
    # The terminal projection is disabled: the reference refuses a nonzero one.
    _require(
        not bool(torch.any(store.read_flat(_BRIDGE + "out.weight") != 0)),
        "the terminal projection must be zero",
    )


def _concat_rows(sources: list[LogicalSource], label: str) -> LogicalSource:
    """Matrices of equal width, one below the other."""
    width = sources[0].shape[1]
    _require(all(len(s.shape) == 2 and s.shape[1] == width for s in sources), label)
    edges = [0]
    for source in sources:
        edges.append(edges[-1] + source.shape[0] * width)

    def read(begin: int, end: int) -> torch.Tensor:
        parts = []
        for source, low, high in zip(sources, edges, edges[1:]):
            a, b = max(begin, low), min(end, high)
            if a < b:
                parts.append(source.values(a - low, b - low))
        return torch.cat(parts)

    return LogicalSource((edges[-1] // width, width), label, read)


def sender_table_source(
    store: SafetensorsSource, vocab: int, code: int, rank: int, *, rows_per_chunk: int = 8192
) -> LogicalSource:
    """[V, 2R]: each token's [predecessor gate; value] rows, as the reference's table builder."""
    codebook = tensor_source(store, _BRIDGE + "codebook", (vocab, code))
    sender = torch.cat(
        [
            tensor_source(store, _BRIDGE + name, (rank, code)).values().reshape(rank, code)
            for name in ("predecessor_gate.weight", "prev.weight")
        ]
    ).to(torch.float32)
    scale = sqrt(code)
    width = 2 * rank

    def read(begin: int, end: int) -> torch.Tensor:
        first, last = begin // width, (end + width - 1) // width
        parts = []
        for row in range(first, last, rows_per_chunk):
            stop = min(last, row + rows_per_chunk)
            codes = (scale * codebook.rows(row, stop).to(torch.float32)).to(torch.bfloat16)
            parts.append((codes.to(torch.float32) @ sender.T).to(torch.bfloat16).reshape(-1))
        values = torch.cat(parts)
        return values[begin - first * width : end - first * width]

    return LogicalSource((vocab, width), f"{store.path}:sender table", read)


def build_model(store: SafetensorsSource) -> Model:
    """The standalone `dspine` artifact's logical parameters."""
    config = dspine_config(store.config)
    _check_markers(store)
    h, v = config["hidden_size"], config["vocab_size"]
    bridge = config["dspine_config"]
    c, r = bridge["code_size"], bridge["rank"]
    layers = config["num_hidden_layers"]
    model = Model({COMPONENT: {"config": config}})
    builder = _Builder(model)
    # The layers bind as a DFlash drafter's do; only the backend prefix differs.
    builder.draft(
        store,
        {**config, "dflash_config": {"target_layer_ids": bridge["target_layer_ids"]}},
        {"hidden_size": h, "vocab_size": v},
        COMPONENT,
    )
    p = COMPONENT + "/bridge/"
    builder.add(p + "gate_bias", store, _BRIDGE + "gate.bias", (r,))
    projection = _concat_rows(
        [
            tensor_source(store, _BRIDGE + name, (r, c))
            for name in ("gate.weight", "predecessor_gate.weight", "prev.weight")
        ],
        "DSpine bridge projection",
    )
    model.add(
        _parameter(p + "projection", projection, (p + "code_input",)),
    )
    for i in range(layers):
        if i == layers - 1:
            read = tensor_source(store, _BRIDGE + "read_weight", (c, h))
        else:
            read = tensor_source(
                store,
                _BRIDGE + "early_read_weights",
                (c, h),
                offset=i * c * h,
                source_shape=(layers - 1, c, h),
            )
        model.add(_parameter(f"{p}layers/{i}/read", read, (f"{p}layers/{i}/read_input",)))
        write = tensor_source(
            store,
            _BRIDGE + "feature_write_weights",
            (h, r),
            offset=i * h * r,
            source_shape=(layers, h, r),
        )
        model.add(_parameter(f"{p}layers/{i}/write", write, (f"{p}layers/{i}/message",)))
    last = tensor_source(
        store,
        _BRIDGE + "feature_write_weights",
        (h, r),
        offset=(layers - 1) * h * r,
        source_shape=(layers, h, r),
    )
    # The last write's columns, one row each: the engine scales them by the final norm's weight
    # and runs the target's head over them to build the head-write table.
    model.add(_parameter(p + "head_write", transpose_source(last, (1, 0), (r, h)), ()))
    model.add(_parameter(p + "sender_table", sender_table_source(store, v, c, r), ()))
    return model


def _parameter(name: str, source: LogicalSource, inputs: tuple[str, ...]):
    from .model import Parameter

    return Parameter(name, tuple(source.shape), source, None, inputs, "bf16", COMPONENT)


def dspine_bf16(model, recipe, sources) -> None:
    """Every DSpine weight in BF16, as the checkpoint stores it."""
    _share_context(model, recipe)


def dspine_q8(model, recipe, sources) -> None:
    """The drafter's layer and feature projections in Q8 (groups of 32, FP16 scales); the bridge
    and its tables stay BF16, which the refinement's roundings depend on."""
    for name, parameter in model.parameters.items():
        if parameter.projection and not name.startswith(COMPONENT + "/bridge/"):
            recipe.assign(name, format="q8_g32_fp16", method=grouped_absmax)
    _share_context(model, recipe)


def _share_context(model, recipe) -> None:
    for layer in range(model.components[COMPONENT]["config"]["num_hidden_layers"]):
        prefix = f"{COMPONENT}/layers/{layer}/attention/"
        for role in ("key", "value"):
            recipe.share(prefix + "context_" + role, prefix + role)


RECIPES = {"dspine_bf16": dspine_bf16, "dspine_q8": dspine_q8}
