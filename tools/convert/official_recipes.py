"""Official representation recipes built from the same public conversion functions."""

from __future__ import annotations

from .methods import (
    cast_direct,
    fp8_row_maxabs,
    grouped_absmax,
    grouped_search,
    import_encoded,
)
from .sources.compressed_tensors import compressed_matrix_source

Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"
FP8 = "fp8_e4m3fn_row_bf16"


def _assign(recipe, name, format, *, source=None, method=grouped_absmax):
    method = method if format in (Q4, Q5, Q6, Q8) else cast_direct
    recipe.assign(name, format=format, method=method, source=source)


def _optional(model, recipe, *, method=grouped_absmax):
    for name, parameter in model.parameters.items():
        if not parameter.projection:
            continue
        if name.startswith("vision/"):
            if name == "vision/patch_embedding":
                format = Q6
            elif name.startswith("vision/merger/"):
                format = Q8
            elif name.endswith(
                ("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")
            ):
                format = Q4
            else:
                format = Q5
            _assign(recipe, name, format, method=method)
        elif name.startswith(("mtp/", "dflash/", "dflash2/")):
            if name.endswith(
                (
                    "/moe/router",
                    "/moe/shared_score",
                    "/attention_conv/kernel_projection",
                    "/mlp_conv/kernel_projection",
                    "/candidate_selector/hidden_projection",
                )
            ):
                continue
            _assign(recipe, name, Q8, method=method)
    for backend in ("dflash", "dflash2"):
        if backend not in model.components:
            continue
        layers = model.components[backend]["config"]["num_hidden_layers"]
        for layer in range(layers):
            prefix = f"{backend}/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


def _dense_groupwise(model, recipe, vocabulary, gate_up=Q4, *, method=grouped_absmax):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe, method=method)
    _assign(recipe, "text/token_embedding", vocabulary, method=method)
    _assign(recipe, "text/output_head", vocabulary, method=method)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        if name.endswith(("/mlp/gate", "/mlp/up")):
            format = gate_up
        elif name.endswith(
            (
                "/attention/query",
                "/attention/key",
                "/gdn/query",
                "/gdn/key",
            )
        ):
            format = Q4
        else:
            format = Q5
        _assign(recipe, name, format, method=method)


def qwen3_6_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q6)


def qwen3_8_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q8)


def qwen3_8_27b_q6(model, recipe, sources):
    _dense_groupwise(model, recipe, Q8, gate_up=Q6)


def qwen3_8_27b_imatrix(model, recipe, sources):
    """Qwen3.8-27B sized for one 24 GB card: ``grouped_search`` with signed scales over every
    text matrix, weighted by ``--source imatrix=PATH`` (a file from ``tools.convert.imatrix``);
    a Q4 embedding, a Q6 head, and Q4 mixer outputs and MLP down in layers 36-63. Vision, MTP
    and draft components keep the ``grouped_absmax`` formats of ``_optional``.

    The Q4 MLP down (5120x17408) needs the Q4 ``linear_add`` route of that shape."""

    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    search = {
        "method": grouped_search,
        "parameters": {"imatrix": str(sources["imatrix"].path), "negative_scales": True},
    }
    recipe.assign("text/token_embedding", format=Q4, **search)
    recipe.assign("text/output_head", format=Q6, **search)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        layer = int(name.split("/")[2])
        if name.endswith(
            (
                "/attention/query",
                "/attention/key",
                "/gdn/query",
                "/gdn/key",
                "/mlp/gate",
                "/mlp/up",
            )
        ):
            format = Q4
        elif name.endswith(("/attention/output", "/gdn/output", "/mlp/down")) and layer >= 36:
            format = Q4
        else:
            format = Q5
        recipe.assign(name, format=format, **search)


def qwen3_6_35b_a3b(model, recipe, sources):
    if "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen3.5 MoE mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(
            (
                "/gdn/a_projection",
                "/gdn/b_projection",
                "/moe/router",
                "/moe/shared_score",
            )
        ):
            continue
        if "/moe/experts/" in name:
            layer = int(name.split("/")[2])
            format = (
                (Q6 if layer in (34, 38, 39) else Q5) if name.endswith("/down") else Q4
            )
        else:
            format = Q8
        _assign(recipe, name, format)


def _flash_next_expert_values(parameter):
    """Keep fused HF expert banks from treating compressed codes as unscaled values."""
    import torch

    from .sources.logical import LogicalSource

    source = parameter.source

    def read(begin, end):
        values = source.values(begin, end)
        if values.dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            raise ValueError(f"{source.label}: HF expert banks require decoded floating-point "
                             "values; fused FP8 source scales are not supported yet")
        return values

    return LogicalSource(source.shape, source.label, read)


def flash_next_mtp_formats(model, recipe):
    """The HF MTP block: Q4 routed experts, BF16 dense and shared projections."""
    formats = {}
    for name, parameter in model.parameters.items():
        if not name.startswith("mtp/"):
            continue
        format = Q4 if "/moe/experts/" in name else parameter.direct_format
        source = _flash_next_expert_values(parameter) if format == Q4 else None
        _assign(recipe, name, format, source=source)
        if format == Q4:
            recipe.assign(name, activation_policy="AllowA8")
        formats[name] = format
    return formats


def qwen3_8_flash_next(model, recipe, sources):
    """HF BF16/block-FP8 quality candidate: Q4 gate/up, Q5 down; no sensitive-layer mask.

    Dense GDN/QSA and shared experts use Q8, embedding Q8 and head Q6. HC, router,
    indexer, PLE, vision and small GDN gates retain their direct representation.
    The table comes from sharded HF BF16 rows (written as FP8) or an IQ4_NL GGUF.
    """
    from .qwen4_exp_ngram import configure_ngram
    from .sources.safetensors import SafetensorsSource

    if "text" in model.components and model.config.get("architectures") != ["Qwen4ExpForCausalLM"]:
        raise ValueError("qwen3_8_flash_next requires Qwen3.8-Flash-Next mathematics")
    if "mtp" in model.components and "mtp" in sources and not isinstance(sources["mtp"], SafetensorsSource):
        raise ValueError("qwen3_8_flash_next requires an HF safetensors MTP source; "
                         "use qwen3_8_flash_next_gguf to import an MTP GGUF")
    for name, parameter in model.parameters.items():
        if not name.startswith("text/"):
            continue
        if name == "text/token_embedding":
            _assign(recipe, name, Q8)
        elif name == "text/output_head":
            _assign(recipe, name, Q6)
        elif parameter.projection:
            if "/moe/experts/" in name:
                format = Q5 if name.endswith("/down") else Q4
                source = _flash_next_expert_values(parameter)
            elif "/moe/shared/" in name or "/attention/" in name or (
                "/gdn/" in name and not name.endswith(("/a_projection", "/b_projection"))
            ):
                format = Q8
                source = None
            else:
                continue
            _assign(recipe, name, format, source=source)
            recipe.assign(name, activation_policy="AllowA8")
    flash_next_mtp_formats(model, recipe)
    if "ngram" in model.components:
        configure_ngram(model, recipe, sources["ngram"])


def _hc8_source(parameter):
    """A hyper-connection matrix as Q8_0 blocks, HC8's rounding (qwen4_exp_preserve.hc8_source):
    an FP16 scale absmax / 127, then nearest-even codes against that rounded scale."""
    import numpy as np
    import torch

    from .gguf_blocks import dequantize_q8_0
    from .sources.logical import EncodedRows, LogicalSource

    source, shape = parameter.source, parameter.shape
    width = shape[1]
    if width % 32:
        raise ValueError(f"{parameter.name}: HC8 needs whole 32-value blocks")

    def encoded(first, last):
        values = source.rows(first, last).float().numpy().reshape(last - first, width // 32, 32)
        scale = (np.abs(values).max(axis=-1) / np.float32(127)).astype("<f2")
        if not np.isfinite(scale).all():
            raise ValueError(f"{parameter.name}: HC8 scale exceeds FP16")
        divisor = scale.astype(np.float32)[..., None]
        normalized = np.divide(values, divisor, out=np.zeros_like(values), where=divisor > 0)
        blocks = np.empty((*scale.shape, 34), dtype=np.uint8)
        blocks[..., :2] = scale[..., None].view(np.uint8)
        blocks[..., 2:] = np.clip(np.rint(normalized), -127, 127).astype(np.int8).view(np.uint8)
        return EncodedRows("gguf_q8_0", torch.from_numpy(blocks.reshape(last - first, -1)),
                           torch.empty(0, dtype=torch.float16))

    def values(first, last):
        low, high = first // width, -(-last // width)
        decoded = dequantize_q8_0(encoded(low, high).codes.numpy()).reshape(-1)
        return torch.from_numpy(decoded[first - low * width:last - low * width])

    return LogicalSource(shape, parameter.source.label + "[HC8]", values, encoded)


def qwen4_dense(model, recipe, sources):
    """Dense Qwen4 (Qwen4ForCausalLM) from its HF BF16 checkpoint, sized for one 24 GB card at the
    27B's geometry: Q4 MLP gate/up and Q5 down; GDN/QSA query and key Q4, their other projections
    Q5 (Qwen3.8-27B's policy); hyper-connection matrices HC8 Q8_0 (the Flash-Next GGUF releases'
    kernels, half the BF16 bytes); embedding Q8 and head Q6; the MTP block's MLP and attention Q8.
    The indexer, PLE, vision, norms and the small GDN gates keep their direct representation. The
    table, when the model has PLE, comes from sharded HF BF16 rows (written as FP8) or an IQ4_NL
    GGUF."""
    from .methods import import_encoded
    from .qwen4_exp_ngram import configure_ngram

    if "text" in model.components and model.config.get("architectures") != ["Qwen4ForCausalLM"]:
        raise ValueError("qwen4_dense requires dense Qwen4 mathematics")
    for name, parameter in model.parameters.items():
        if not name.startswith(("text/", "mtp/")):
            continue
        if name == "text/token_embedding":
            _assign(recipe, name, Q8)
            continue
        if name == "text/output_head":
            _assign(recipe, name, Q6)
            continue
        if not parameter.projection:
            continue
        mtp = name.startswith("mtp/")
        role = name.rsplit("/", 2)
        if role[-2] in ("attn_hc", "mlp_hc", "final_mixer") and role[-1] in ("down", "up",
                                                                             "inject"):
            recipe.assign(name, format="gguf_q8_0", method=import_encoded,
                          source=_hc8_source(parameter))
            continue
        if name.endswith(("/mlp/gate", "/mlp/up")):
            format = Q8 if mtp else Q4
        elif name.endswith("/mlp/down"):
            format = Q8 if mtp else Q5
        elif "/gdn/" in name and name.endswith(("/a_projection", "/b_projection")):
            continue
        elif "/attention/" in name or "/gdn/" in name:
            format = Q8 if mtp else (
                Q4 if name.endswith(("/attention/query", "/attention/key", "/gdn/query",
                                     "/gdn/key")) else Q5)
        else:
            continue
        _assign(recipe, name, format)
        recipe.assign(name, activation_policy="AllowA8")
    if "ngram" in model.components:
        configure_ngram(model, recipe, sources["ngram"])


def qwen3_6_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q8)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        layer = int(name.split("/")[2])
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        direct = (
            ("/attention/" in name and not name.endswith("/output") and layer < 24)
            or (name.endswith("/attention/output") and layer in (3, 7))
            or (name.endswith("/gdn/output") and layer == 4)
        )
        if direct:
            continue
        recipe.assign(
            name,
            format="nvfp4",
            method=import_encoded,
            source=model.source(name, quantized, "nvfp4"),
            activation_policy="AllowA4",
        )


def qwen3_8_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or name == "text/token_embedding":
            continue
        source = model.source(name, quantized)
        if not parameter.projection or name.endswith(
            ("/gdn/a_projection", "/gdn/b_projection")
        ):
            recipe.assign(name, source=source)
            continue
        layer = int(name.split("/")[2]) if name.startswith("text/layers/") else -1
        format = "nvfp4" if "/mlp/" in name and layer < 56 else FP8
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, quantized, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


def qwen3_8_27b_nvfp4_nvidia(model, recipe, sources):
    """nvidia/Qwen3.8-27B-NVFP4 (ModelOpt AutoQuant) layout: every text MLP
    projection is NVFP4; attention and GDN projections are per-row FP8. The
    source output head is NVFP4, but the runtime registers the vocabulary
    projection only for FP8, so it is dequantised and re-quantised there."""
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or name == "text/token_embedding":
            continue
        source = model.source(name, quantized)
        if not parameter.projection or name.endswith(
            ("/gdn/a_projection", "/gdn/b_projection")
        ):
            recipe.assign(name, source=source)
            continue
        if name == "text/output_head":
            recipe.assign(
                name,
                format=FP8,
                method=fp8_row_maxabs,
                source=source,
                activation_policy="AllowA8",
            )
            continue
        format = (
            "nvfp4"
            if name.startswith("text/layers/") and "/mlp/" in name
            else FP8
        )
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, quantized, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


def qwen3_8_27b_nvfp4_orcarouter(model, recipe, sources):
    """orcarouter/Qwen3.8-27B-Uncensored-NVFP4 (GPTQ compressed-tensors)
    layout: MLP projections of layers 0..55 are NVFP4 packed; the last eight
    layers' MLP projections and every attention/GDN projection are per-row
    FP8; the embedding and output head remain BF16 and are re-quantised."""
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    recipe.assign("text/output_head", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if (
            not name.startswith("text/layers/")
            or not parameter.projection
        ):
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.assign(name, source=model.source(name, quantized))
            continue
        layer = int(name.split("/")[2])
        format = "nvfp4" if ("/mlp/" in name and layer < 56) else FP8
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, quantized, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


_MOE_EXPERT_SOURCE = {
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
}


def _moe_encoded_source(store, parameter, name):
    """Map one routed or shared expert matrix onto its per-expert NVFP4 prefix.

    The MoE description reads routed experts out of a single fused `experts.gate_up_proj`
    tensor with an offset, so the generic factory cannot reach an encoded source for them
    and says so. The compressed-tensors checkpoint stores every expert as its own prefix,
    which is a plain matrix, so the mapping is stated here once instead.
    """
    head, _, tail = name.partition("/moe/")
    layer = head.rsplit("/", 1)[-1]
    if tail.startswith("experts/"):
        expert, _, role = tail[len("experts/") :].partition("/")
        prefix = f"mlp.experts.{expert}."
    else:
        prefix, role = "mlp.shared_expert.", tail.rsplit("/", 1)[-1]
    if role not in _MOE_EXPERT_SOURCE:
        raise ValueError(f"{name}: unknown expert matrix {role!r}")
    leaf = prefix + _MOE_EXPERT_SOURCE[role]
    for root in ("model.language_model.layers.", "model.layers."):
        prefix = f"{root}{layer}.{leaf}"
        if store.has(prefix + ".weight_packed"):
            return compressed_matrix_source(store, prefix, parameter.shape, "nvfp4")
    raise ValueError(f"{name}: no NVFP4 source for {leaf} in {store.path}")


def qwen3_6_35b_a3b_nvfp4(model, recipe, sources):
    """`qwen3_6_35b_a3b` with routed and shared experts imported as NVFP4 instead of Q4/Q5.

    Everything outside the experts keeps the representation of the groupwise recipe, so the
    two artifacts differ in exactly one mechanism and are comparable to each other.
    """
    if "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen3.5 MoE mathematics")
    quantized = sources["quantized"]
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(
            (
                "/gdn/a_projection",
                "/gdn/b_projection",
                "/moe/router",
                "/moe/shared_score",
            )
        ):
            continue
        if "/moe/experts/" in name or "/moe/shared/" in name:
            recipe.assign(
                name,
                format="nvfp4",
                method=import_encoded,
                source=_moe_encoded_source(quantized, parameter, name),
                activation_policy="AllowA4",
            )
        else:
            _assign(recipe, name, Q8)
    # The op takes the routed experts as one plane per bank, so the banks the model declares have
    # to become one parent each. Default packing keeps parents apart when their sources were
    # quantised against different divisors, which is right where the consumer reads one divisor and
    # wrong here: this plane is a stack by construction, and its kernels take the divisor from the
    # row they are reading.
    for names in model.packing_groups:
        # Text layers only: the MTP component declares the same role names and keeps its groupwise
        # Q8 experts, which this recipe does not assign and must not group.
        if all(
            name.startswith("text/layers/")
            and ("/moe/experts/" in name or "/moe/shared/" in name)
            for name in names
        ):
            recipe.group(names)


RECIPES = {
    "qwen3_8_flash_next": qwen3_8_flash_next,
    "qwen4_dense": qwen4_dense,
    "qwen3_6_27b": qwen3_6_27b,
    "qwen3_6_27b_nvfp4": qwen3_6_27b_nvfp4,
    "qwen3_8_27b": qwen3_8_27b,
    "qwen3_8_27b_q6": qwen3_8_27b_q6,
    "qwen3_8_27b_imatrix": qwen3_8_27b_imatrix,
    "qwen3_8_27b_nvfp4": qwen3_8_27b_nvfp4,
    "qwen3_8_27b_nvfp4_nvidia": qwen3_8_27b_nvfp4_nvidia,
    "qwen3_8_27b_nvfp4_orcarouter": qwen3_8_27b_nvfp4_orcarouter,
    "qwen3_6_35b_a3b": qwen3_6_35b_a3b,
    "qwen3_6_35b_a3b_nvfp4": qwen3_6_35b_a3b_nvfp4,
}
