"""Native GGUF operands with Basalt's direct-weight and HC8 conventions."""

from copy import deepcopy

import numpy as np
import torch

from . import qwen4_exp_gguf as q
from .gguf_blocks import dequantize_q8_0
from .sources.gguf import TYPE_BF16, TYPE_F16, TYPE_F32
from .sources.logical import EncodedRows, LogicalSource, array_source


def model_config(config):
    out = deepcopy(config)
    text = out.get("text_config", out)
    if text.get("architectures", out.get("architectures")) not in (
            ["Qwen4ExpForCausalLM"], ["Qwen4ExpForConditionalGeneration"]):
        raise ValueError("the preservation recipe requires Qwen3.8-Flash-Next")
    text.update(fp32_direct=True, norm_unit_offset=False, hc_prefill_inject=True, gdn_a_is_log=False)
    return out


def norm_source(gguf, name, zero_centred):
    # Gamma is already 1+w in the GGUF. Keep those FP32 bits and disable the runtime offset.
    return array_source(torch.from_numpy(np.array(gguf.read_direct(name), copy=True)), name)


def validate_direct(gguf, *, compat_bf16=False):
    """Fail rather than silently narrow a projection whose native operand remains BF16."""
    for name, info in gguf.tensors.items():
        if info.type_id not in (TYPE_F16, TYPE_F32):
            continue
        if "norm" in name or name.endswith(("ssm_a", "ssm_dt.bias", "conv1d.weight")):
            continue
        dtype = np.dtype("<f2" if info.type_id == TYPE_F16 else "<f4")
        step = (1 << 20) * dtype.itemsize
        for first in range(0, info.nbytes, step):
            values = gguf.tensor_bytes(name, first, min(first + step, info.nbytes)).view(dtype).astype(np.float32)
            cast = torch.from_numpy(values).bfloat16().float().numpy()
            if not np.isfinite(values).all() or not np.isfinite(cast).all():
                raise ValueError(f"{name}: non-finite BF16 projection")
            bf16_operand = name.endswith(("ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
                "indexer.k_proj.weight", "indexer.q_proj.weight", "ssm_alpha.weight", "ssm_beta.weight",
                "ple_key.weight", "ple_value.weight")) or ("hc_" in name and "norm" not in name)
            if not (compat_bf16 and bf16_operand) and not np.array_equal(values, cast):
                raise ValueError(f"{name}: preserving this projection requires values exactly representable in BF16")


def bf16_matrix(gguf, name, shape, *, compat_bf16=False, select=None):
    info = gguf.info(name)
    if info.type_id not in (TYPE_BF16, TYPE_F16, TYPE_F32) and not compat_bf16:
        raise ValueError(f"{name}: requires original BF16 operands or explicit --compat-bf16")
    source, _ = q.matrix_source(gguf, name, shape, select or q.rows())

    def read(first, last):
        values = source.values(first, last).bfloat16()
        if not torch.isfinite(values.float()).all():
            raise ValueError(f"{name}: non-finite BF16 operand")
        return values
    return LogicalSource(shape, name + "[BF16 operand]", read)


def hc8_source(gguf, name, shape, *, compat_bf16=False):
    """HC8: FP16(absmax/127), then nearest-even codes against that rounded scale."""
    source = bf16_matrix(gguf, name, shape, compat_bf16=compat_bf16)
    width = shape[1]

    def encoded(first, last):
        if first == last:
            return EncodedRows("gguf_q8_0", torch.empty((0, width // 32 * 34), dtype=torch.uint8),
                               torch.empty(0, dtype=torch.float16))
        values = source.rows(first, last).float().numpy().reshape(last - first, width // 32, 32)
        scale = (np.abs(values).max(axis=-1) / np.float32(127)).astype("<f2")
        if not np.isfinite(scale).all():
            raise ValueError(f"{name}: HC8 scale exceeds FP16")
        divisor = scale.astype(np.float32)[..., None]
        normalized = np.divide(values, divisor, out=np.zeros_like(values), where=divisor > 0)
        codes = np.clip(np.rint(normalized), -127, 127).astype(np.int8)
        blocks = np.empty((*scale.shape, 34), dtype=np.uint8)
        blocks[..., :2] = scale[..., None].view(np.uint8)
        blocks[..., 2:] = codes.view(np.uint8)
        return EncodedRows("gguf_q8_0", torch.from_numpy(blocks.reshape(last - first, -1)),
                           torch.empty(0, dtype=torch.float16))

    def values(first, last):
        low, high = first // width, -(-last // width)
        decoded = dequantize_q8_0(encoded(low, high).codes.numpy()).reshape(-1)
        return torch.from_numpy(decoded[first - low * width:last - low * width])

    return LogicalSource(shape, name + "[HC8]", values, encoded)


def _hc_operands(found, gguf, model, *, mtp=False, compat_bf16=False):
    for logical in tuple(found):
        parts = logical.split("/")
        if parts[-1] not in ("down", "up", "inject") or parts[-2] not in ("attn_hc", "mlp_hc", "final_mixer"):
            continue
        if parts[-2] == "final_mixer":
            prefix = f"blk.{q.MTP_BLOCK}.nextn.hc_head_" if mtp else "output_hc_"
        else:
            layer = q.MTP_BLOCK if mtp else (model.source_layers or tuple(range(q.LAYERS)))[int(parts[2])]
            prefix = f"blk.{layer}.hc_{'attn' if parts[-2] == 'attn_hc' else 'ffn'}_"
        stored = prefix + parts[-1] + ".weight"
        shape = model.parameters[logical].shape
        found[logical] = hc8_source(gguf, stored, shape, compat_bf16=compat_bf16), True
        if parts[-1] == "inject":
            found[logical.rsplit("/", 1)[0] + "/prefill_inject"] = (
                bf16_matrix(gguf, stored, shape, compat_bf16=compat_bf16), False)
    return found


def text_sources(gguf, model, *, compat_bf16=False):
    validate_direct(gguf, compat_bf16=compat_bf16)
    found = q.text_sources(gguf, model.config, model.source_layers,
                           norm_source=norm_source, direct_dtype=torch.float32)
    for layer, source_layer in enumerate(model.source_layers or range(model.config["num_hidden_layers"])):
        prefix, stored = f"text/layers/{layer}/", f"blk.{source_layer}."
        operands = {prefix + "moe/router": stored + "ffn_gate_inp.weight"}
        if model.config["layer_types"][layer] == "full_attention":
            operands.update({prefix + "indexer/" + role: stored + f"indexer.{short}_proj.weight"
                             for role, short in (("query", "q"), ("key", "k"))})
        else:
            operands.update({prefix + "gdn/" + role: stored + name
                             for role, name in (("a_projection", "ssm_alpha.weight"),
                                                ("b_projection", "ssm_beta.weight"))})
        if layer in model.config["ple_layers"]:
            operands[prefix + "ple/value"] = stored + "ple_value.weight"
            key = stored + "ple_key.weight"
            if gguf.info(key).type_id not in (8, 42):
                operands[prefix + "ple/key"] = key
        for logical, name in operands.items():
            found[logical] = bf16_matrix(gguf, name, model.parameters[logical].shape,
                                         compat_bf16=compat_bf16,
                                         select=q.untiled_heads() if logical.endswith(
                                             ("/a_projection", "/b_projection")) else None), False
        if layer not in model.config["ple_layers"]:
            continue
        name = f"blk.{source_layer}.ple_conv1d.weight"
        # Basalt narrows this operand once at load (dense index kind 3), unlike GDN convolution.
        values = torch.from_numpy(np.array(gguf.read_direct(name), copy=True)).half().float()
        if not torch.isfinite(values).all():
            raise ValueError(f"{name}: PLE convolution exceeds FP16")
        found[f"text/layers/{layer}/ple/convolution"] = array_source(values, name + "[FP16]"), False
    return _hc_operands(found, gguf, model, compat_bf16=compat_bf16)


def mtp_sources(gguf, experts, model, *, compat_bf16=False):
    validate_direct(gguf, compat_bf16=compat_bf16)
    found = q.mtp_sources(gguf, experts, norm_source=norm_source)
    for logical, stored in (("mtp/layer/moe/router", f"blk.{q.MTP_BLOCK}.ffn_gate_inp.weight"),
            ("mtp/layer/indexer/query", f"blk.{q.MTP_BLOCK}.indexer.q_proj.weight"),
            ("mtp/layer/indexer/key", f"blk.{q.MTP_BLOCK}.indexer.k_proj.weight")):
        found[logical] = bf16_matrix(gguf, stored, model.parameters[logical].shape,
                                     compat_bf16=compat_bf16), False
    return _hc_operands(found, gguf, model, mtp=True, compat_bf16=compat_bf16)


def vision_source(source):
    """Matrices remain BF16 only when the source values are already exact in that format."""
    def read(first, last):
        values = source.values(first, last)
        cast = values.bfloat16()
        if not torch.equal(values.float(), cast.float()):
            raise ValueError(f"{source.label}: BF16 conversion would change stored values")
        return cast
    return LogicalSource(source.shape, source.label, read)
