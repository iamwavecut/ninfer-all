"""Dense Llama GGUF import, preserving stored matrix blocks and unscaled NeoX RoPE.

The supported contract has bias-free attention, no Q/K normalization or sink, and a dense
SiLU-gated MLP in every block. Nanbeige4.3-33B is one instance of this architecture.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import torch
from tools.artifact.formats import GGUF_FORMATS_BY_TYPE
from .gguf_blocks import block_source
from .methods import cast_direct, import_encoded
from .resources import token_domain
from .sources.gguf import GGUFFile
from .sources.logical import LogicalSource, array_source, select_rows
from .ternary import rows

ARCHITECTURES = ("LlamaForCausalLM",)
RESOURCES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")

def text_config(source):
    if source.get("architectures") != list(ARCHITECTURES) or source.get("model_type") != "llama":
        raise ValueError("expected LlamaForCausalLM/model_type llama")
    if source.get("hidden_act") != "silu":
        raise ValueError("Llama requires SiLU SwiGLU")
    for name in ("attention_bias", "mlp_bias", "tie_word_embeddings", "rope_scaling"):
        if source.get(name):
            raise ValueError(f"Llama: unsupported {name}")
    if source.get("rope_parameters", {}).get("rope_type", "default") != "default":
        raise ValueError("Llama: only unscaled RoPE is implemented")
    out = {"architectures": list(ARCHITECTURES), "model_type": "llama", "tie_word_embeddings": False, "qk_layout": "neox"}
    for name in ("hidden_size", "vocab_size", "num_hidden_layers", "max_position_embeddings",
                 "num_attention_heads", "num_key_value_heads", "intermediate_size"):
        value = source.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Llama: {name} must be a positive integer")
        out[name] = value
    out["head_dim"] = source.get("head_dim", out["hidden_size"] // out["num_attention_heads"])
    if (out["head_dim"] != 128 or out["hidden_size"] != out["head_dim"] * out["num_attention_heads"]
            or out["num_attention_heads"] % out["num_key_value_heads"] or out["hidden_size"] % 256
            or out["intermediate_size"] % 256):
        raise ValueError("Llama: unsupported head or projection geometry")
    for name, default in (("rms_norm_eps", 1e-6), ("rope_theta", 10000.0)):
        out[name] = float(source.get(name, default))
        if not math.isfinite(out[name]) or out[name] <= 0:
            raise ValueError(f"Llama: invalid {name}")
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if name in source: out[name] = source[name]
    return out

def validate(gguf, config):
    if gguf.kv.get("general.architecture") != "llama":
        raise ValueError("expected a llama GGUF")
    fields = {"block_count": "num_hidden_layers", "embedding_length": "hidden_size",
              "feed_forward_length": "intermediate_size", "attention.head_count": "num_attention_heads",
              "attention.head_count_kv": "num_key_value_heads", "rope.dimension_count": "head_dim"}
    for key, name in fields.items():
        if gguf.kv.get("llama." + key) != config[name]:
            raise ValueError(f"llama.{key} differs from {name}")
    for key, name in (("rope.freq_base", "rope_theta"), ("attention.layer_norm_rms_epsilon", "rms_norm_eps")):
        if not np.isclose(gguf.kv.get("llama." + key, 0), config[name], rtol=1e-6, atol=0):
            raise ValueError(f"llama.{key} differs from {name}")
    if gguf.kv.get("llama.rope.scaling.type", "none") not in ("none", ""):
        raise ValueError("scaled GGUF RoPE is not implemented")
    for name in gguf.tensors:
        if name.endswith(".bias") or "_norm" in name and ("attn_q_norm" in name or "attn_k_norm" in name):
            raise ValueError(f"Llama: unsupported operand {name}")

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
    """F32 as stored, BF16 and F16 widened exactly."""

    info = gguf.info(tensor)
    if info.type_name not in ("F32", "BF16", "F16"):
        raise ValueError(f"{tensor}: expected a float vector or matrix, found {info.type_name}")
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


def unpermute_qk(source, heads, dim):
    """GGUF Llama interleaves rotary pairs; HF uses halves. Move whole stored rows only."""
    spans = tuple((h * dim + 2 * i + half, h * dim + 2 * i + half + 1)
                  for h in range(heads) for half in range(2) for i in range(dim // 2))
    return select_rows(source, spans)

def build_model(base, gguf: GGUFFile, *, resource_overrides=None):
    """Bind the dense Llama mathematics to the stored GGUF operands."""

    from .model import Model, Parameter

    config = text_config(base.config)
    validate(gguf, config)
    h, vocab = config["hidden_size"], config["vocab_size"]
    references, payloads, count, special = _load_resources(
        base.root, vocab, dict(resource_overrides or {}))
    model = Model({"text": {"config": config, "resources": references}}, resources=payloads,
                  token_count=count, special_token_ids=special)

    def matrix(name, tensor, shape, inputs=(), source=None):
        model.add(Parameter(name, shape, source or _matrix(gguf, tensor, shape),
                            inputs=tuple(inputs)))

    def vector(name, tensor, shape):
        model.add(Parameter(name, shape, _direct(gguf, tensor, shape), direct_format="fp32"))

    matrix("text/token_embedding", "token_embd.weight", (vocab, h))
    matrix("text/output_head", "output.weight", (vocab, h), ("text/final_hidden",))
    vector("text/final_norm", "output_norm.weight", (h,))
    heads, kv_heads, d = (config["num_attention_heads"], config["num_key_value_heads"],
                          config["head_dim"])
    for i in range(config["num_hidden_layers"]):
        g, p = f"blk.{i}.", f"text/layers/{i}/"
        a = p + "attention/"
        matrix(a + "query", g + "attn_q.weight", (heads * d, h), (p + "mixer_input",),
               unpermute_qk(_matrix(gguf, g + "attn_q.weight", (heads * d, h)), heads, d))
        matrix(a + "key", g + "attn_k.weight", (kv_heads * d, h), (p + "mixer_input",),
               unpermute_qk(_matrix(gguf, g + "attn_k.weight", (kv_heads * d, h)), kv_heads, d))
        matrix(a + "value", g + "attn_v.weight", (kv_heads * d, h), (p + "mixer_input",))
        matrix(a + "output", g + "attn_output.weight", (h, heads * d), (p + "attention_output",))
        vector(p + "input_norm", g + "attn_norm.weight", (h,))
        vector(p + "ffn_norm", g + "ffn_norm.weight", (h,))
        width = config["intermediate_size"]
        matrix(p + "mlp/gate", g + "ffn_gate.weight", (width, h), (p + "mlp_input",))
        matrix(p + "mlp/up", g + "ffn_up.weight", (width, h), (p + "mlp_input",))
        matrix(p + "mlp/down", g + "ffn_down.weight", (h, width), (p + "mlp_product",))
    return model

def llama_gguf(model, recipe, sources):
    if model.config.get("architectures") != list(ARCHITECTURES):
        raise ValueError("llama_gguf requires a Llama model")
    for name, parameter in model.parameters.items():
        if parameter.direct_format == "fp32":
            recipe.assign(name, format="fp32", method=cast_direct, source=parameter.source)
        else:
            fmt = parameter.source.read_encoded(0, 1).format
            recipe.assign(name, format=fmt, method=import_encoded, source=parameter.source,
                          **({"activation_policy": "AllowA8"} if parameter.projection else {}))

RECIPES = {"llama_gguf": llama_gguf}
