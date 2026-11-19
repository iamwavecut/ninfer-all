"""Convert a Flash-Next .basalt pack to NInfer v3, keeping its effective weights.

Run with python -m tools.convert.basalt INPUT.basalt --output MODEL.ninfer.
Only small frontend files and the vision GGUF are staged; table and expert reads go directly
to the source pack. No download, substitute donor, or whole-checkpoint materialization occurs.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import re
from pathlib import Path
import tempfile

import numpy as np
import torch

from . import qwen4_exp, qwen4_exp_gguf as q
from .basalt_config import pack_sampling
from .edge0 import block_rows_source, mtp_sources
from .gguf_blocks import _vision_source
from .methods import cast_direct
from .pipeline import convert
from .recipe import Recipe
from .resources import STD_QWEN_SPLIT_PATTERN
from .sources.basalt import BasaltFile, BasaltGGUF, mtp_pack
from .sources.gguf import TYPE_F16, TYPE_F32
from .sources.gguf_source import GGUFSource
from .sources.logical import LogicalSource, array_source
from .sources.safetensors import SafetensorsSource


def model_config(pack: BasaltFile) -> dict:
    """Flash-Next's fixed mathematical architecture with the pack's own variable metadata."""
    kv = pack.meta["gguf_kv"]
    for key, expected in q.EXPECTED_HEADER.items():
        if kv.get(key) != expected:
            raise ValueError(f"{key}: expected {expected!r}, got {kv.get(key)!r}")
    if kv.get("qwen4exp.rope.scaling.type", "none") != "none":
        raise ValueError("the pack has a RoPE scaling policy; supply its equivalent at serving time")
    t = dict(model_type="qwen4_exp_text", hidden_act="silu", attention_bias=False,
             mamba_ssm_dtype="float32", output_gate_type="sigmoid", indexer_kv_heads=1,
             hidden_size=2560, vocab_size=248320, num_hidden_layers=48,
             num_attention_heads=24, num_key_value_heads=2, head_dim=256,
             linear_num_key_heads=16, linear_key_head_dim=128, linear_num_value_heads=48,
             linear_value_head_dim=128, linear_conv_kernel_dim=4,
             num_experts=kv["qwen4exp.expert_count"], num_experts_per_tok=10,
             moe_intermediate_size=640, shared_expert_intermediate_size=640,
             hc_count=4, hc_lowrank=320, indexer_n_heads=4, indexer_head_dim=128,
             indexer_compress_ratio=4, indexer_budget=2048,
             ple_embed_dim=2560, ngram_size=3, heads_per_ngram=8,
             ngram_vocab_size_base=20000000, make_ngram_vocab_size_divisible_by=128,
             ple_conv_kernel_size=4, ple_layer_ids=[2], tie_word_embeddings=False,
             eos_token_id=kv["qwen4exp.ple.eos_token_id"],
             max_position_embeddings=kv["qwen4exp.context_length"],
             rms_norm_eps=kv["qwen4exp.attention.layer_norm_rms_epsilon"],
             layer_types=["full_attention" if i % 4 == 3 else "linear_attention" for i in range(48)],
             rope_parameters={"rope_theta": kv["qwen4exp.rope.freq_base"],
                              "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10],
                              "mrope_interleaved": True},
             fp32_direct=True, norm_unit_offset=False, hc_prefill_inject=True, gdn_a_is_log=False,
             expert_output_scales=("experts.layout" in pack.sections and
                                   bool(re.match(r"# (?:strata )?experts v6:", pack.text("experts.layout")))),
             mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=False,
             mtp={"num_hidden_layers": 1, "hybrid": True, "num_experts": 512,
                  "layer_types": ["full_attention"], "rope_theta": kv["qwen4exp.rope.freq_base"]})
    return {"architectures": ["Qwen4ExpForConditionalGeneration"], "text_config": t,
            "vision_config": dict(direct_format="fp32", depth=27, hidden_size=1152, intermediate_size=4304,
                                  num_heads=16, patch_size=16, temporal_patch_size=2,
                                  spatial_merge_size=2, num_position_embeddings=2304,
                                  out_hidden_size=2560, in_channels=3,
                                  hidden_act="gelu_pytorch_tanh", deepstack_visual_indexes=[])}


def frontend_files(pack: BasaltFile, root: Path, config: dict) -> None:
    """Translate Basalt's GPT-2 vocabulary representation, preserving IDs, merges and template."""
    kv = pack.meta["gguf_kv"]
    tokens = kv["tokenizer.ggml.tokens"]
    types = kv["tokenizer.ggml.token_type"]
    if (kv.get("tokenizer.ggml.model") != "gpt2" or kv.get("tokenizer.ggml.pre") != "qwen35"
            or len(tokens) != len(types) or len(set(tokens)) != len(tokens)):
        raise ValueError("expected the pack's Qwen35 byte-BPE vocabulary and token types")
    vocab = {word: i for i, word in enumerate(tokens)}
    if "tokenizer/vocab.json" in pack.sections and json.loads(pack.text("tokenizer/vocab.json")) != vocab:
        raise ValueError("Basalt tokenizer vocabulary differs from its GGUF metadata")
    added = [{"id": i, "content": word, "single_word": False, "lstrip": False,
              "rstrip": False, "normalized": False, "special": types[i] == 3}
             for i, word in enumerate(tokens) if types[i] in (3, 4)]
    tokenizer = {"version": "1.0", "truncation": None, "padding": None,
                 "added_tokens": added, "normalizer": None,
                 "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                     {"type": "Split", "pattern": {"Regex": STD_QWEN_SPLIT_PATTERN},
                      "behavior": "Isolated", "invert": False},
                     {"type": "ByteLevel", "add_prefix_space": False,
                      "trim_offsets": False, "use_regex": False}]},
                 "decoder": {"type": "ByteLevel", "add_prefix_space": False,
                             "trim_offsets": False, "use_regex": False},
                 "post_processor": None,
                 "model": {"type": "BPE", "dropout": None, "unk_token": None,
                           "continuing_subword_prefix": "", "end_of_word_suffix": "",
                           "fuse_unk": False, "byte_fallback": False, "vocab": vocab,
                           "merges": kv["tokenizer.ggml.merges"]}}
    eos = kv["tokenizer.ggml.eos_token_id"]
    tokenizer_config = {"add_bos_token": False, "add_prefix_space": False,
                        "pad_token": "<|endoftext|>", "eos_token": tokens[eos],
                        "added_tokens_decoder": {str(x["id"]): {k: v for k, v in x.items() if k != "id"}
                                                 for x in added}}
    generation = {"eos_token_id": eos}
    generation.update({name.replace("-", "_"): value for name, value in pack_sampling(pack.meta).items()})
    files = {"config.json": config, "tokenizer.json": tokenizer,
             "tokenizer_config.json": tokenizer_config, "generation_config.json": generation}
    # Basalt vision/preprocess.hpp: 8..4096 merged tokens, each a 32x32 pixel square.
    # NInfer's frontend requires explicit normalization and pixel budgets in both resources.
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        files[name] = {"patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                       "image_mean": [0.5] * 3, "image_std": [0.5] * 3,
                       "size": {"shortest_edge": 8 * 32 * 32, "longest_edge": 4096 * 32 * 32}}
    for name, value in files.items():
        (root / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    template = (pack.text("tokenizer/chat_template.jinja") if "tokenizer/chat_template.jinja" in pack.sections
                else kv["tokenizer.chat_template"])
    (root / "chat_template.jinja").write_text(template, encoding="utf-8", newline="")


def stored_norm(stored: np.ndarray, label: str):
    # Keep gamma itself: subtracting one before narrowing loses tuned FP32 norm values.
    return array_source(torch.from_numpy(stored.astype(np.float32, copy=True)), label)


def validate_direct_values(source: BasaltGGUF) -> None:
    """Matrices still served in BF16 must fit; scalar operands retain their native FP32 values."""
    for name, info in source.tensors.items():
        if info.type_id not in (TYPE_F16, TYPE_F32):
            continue
        if "norm" in name or name.endswith(("ssm_a", "ssm_dt.bias", "conv1d.weight")):
            continue
        dtype = np.dtype("<f2" if info.type_id == TYPE_F16 else "<f4")
        chunk_bytes = (1 << 20) * dtype.itemsize
        for offset in range(0, info.nbytes, chunk_bytes):
            raw = source.tensor_bytes(name, offset, min(offset + chunk_bytes, info.nbytes))
            values = np.frombuffer(raw, dtype=dtype).astype(np.float32)
            cast = torch.from_numpy(values).to(torch.bfloat16).float().numpy()
            if not np.array_equal(values, cast):
                raise ValueError(f"{name}: {info.type_name} values need a wider direct operand than BF16")


def lossless_bf16(source: LogicalSource) -> LogicalSource:
    """Check each streamed projection slice before a BF16 cast, including vision patch weights."""
    def read(begin, end):
        values = source.values(begin, end)
        kept = values.to(torch.bfloat16)
        if not torch.equal(values.float(), kept.float()):
            raise ValueError(f"{source.label}: direct BF16 conversion would change stored values")
        return kept
    return LogicalSource(source.shape, source.label, read)


def components_for(pack: BasaltFile) -> tuple[str, ...]:
    required = {"meta", "gguf", "dense", "dense.index", "experts", "experts.layout"}
    if required - pack.sections.keys():
        raise ValueError(f"missing pack sections: {sorted(required - pack.sections.keys())}")
    known = required | {"hc8", "profile", "mtp/dense", "mtp/dense.index", "mtp/experts",
                        "mtp/draft_vocab", "vision/mmproj", "vision/text_vocab"}
    unknown = [name for name in pack.sections if name not in known and not name.startswith("tokenizer/")]
    if unknown:
        raise ValueError(f"unmapped pack sections: {unknown}; refusing to drop a component")
    result = ["text", "ngram"]
    if any(name.startswith("mtp/") for name in pack.sections):
        missing = {"mtp/dense", "mtp/dense.index", "mtp/experts"} - pack.sections.keys()
        if missing:
            raise ValueError(f"incomplete MTP component: {sorted(missing)}")
        result.append("mtp")
    if "vision/mmproj" in pack.sections:
        result.append("vision")
    elif "vision/text_vocab" in pack.sections:
        raise ValueError("vision vocabulary exists without its projector")
    return tuple(result)


def hc_source(source: BasaltGGUF, name: str, shape: tuple[int, int], fallback=None):
    if name in source.hc.entries:
        if source.hc.entries[name][:2] != shape:
            raise ValueError(f"{name}: HC8 shape differs from model")
        return block_rows_source(shape, name, 8, lambda first, last: source.hc.rows(name, first, last))
    if source.pack.version == 2:
        raise ValueError(f"{name}: missing stored HC8 operand")
    # v1 computes this same layout at load. Quantize against the rounded FP16 scale (HC8),
    # not ggml's float scale/away-from-zero quantizer.
    def read(first, last):
        values = fallback(first, last).reshape(last - first, -1, 32).astype(np.float32)
        scale = (np.max(np.abs(values), axis=2) / np.float32(127)).astype(np.float16)
        with np.errstate(divide="ignore", invalid="ignore"):
            codes = np.where(scale[..., None] > 0,
                np.rint(values / scale.astype(np.float32)[..., None]), 0)
        out = np.empty((*scale.shape, 34), dtype=np.uint8)
        out[..., :2] = scale[..., None].view(np.uint8)
        out[..., 2:] = np.clip(codes, -127, 127).astype(np.int8).view(np.uint8)
        return out.reshape(last - first, -1)
    return block_rows_source(shape, name + "[v1 HC8]", 8, read)


def configure(pack, source, model, recipe, vision=None):
    q.validate(source, model.config, model.source_layers)
    validate_direct_values(source)
    found = q.text_sources(source, model.config, model.source_layers,
        norm_source=lambda src, name, offset: stored_norm(src.read_direct(name), name),
        direct_dtype=torch.float32)
    for layer, scales in (source.expert_scales.items() if model.config.get("expert_output_scales") else ()):
        found[f"text/layers/{layer}/moe/expert_scales"] = (
            array_source(torch.from_numpy(scales), f"expert scales[{layer}]"), False)
    for logical in tuple(found):
        parts = logical.split("/")
        if parts[-1] not in ("down", "up", "inject") or parts[-2] not in ("attn_hc", "mlp_hc", "final_mixer"):
            continue
        stored = ("output_hc_" if parts[-2] == "final_mixer" else
                  f"blk.{parts[2]}.hc_{'attn' if parts[-2] == 'attn_hc' else 'ffn'}_") + parts[-1] + ".weight"
        shape = model.parameters[logical].shape
        found[logical] = hc_source(source, stored, shape,
            lambda first, last, stored=stored: source.read_direct(stored)[first:last]), True
        if parts[-1] == "inject":
            found[logical.rsplit("/", 1)[0] + "/prefill_inject"] = (
                array_source(torch.from_numpy(source.prefill_injection(stored)).to(torch.bfloat16),
                             stored + "[prefill BF16]"), False)
    if "mtp" in model.components:
        mtp = mtp_pack(pack)
        def mtp_hc(name, shape):
            return hc_source(source, "mtp." + name, shape,
                lambda first, last: (mtp.bf16_words(name, shape)[first:last].astype(np.uint32) << 16).view(np.float32))
        found.update(mtp_sources(mtp, hc_source=mtp_hc, norm_source=stored_norm, prefill_inject=True))
    formats = {name: q._assign(recipe, name, item, encoded, model)
               for name, (item, encoded) in found.items()}
    if vision is not None:
        view = _vision_source(model, {"vision": vision})
        for name, parameter in model.parameters.items():
            if name.startswith("vision/"):
                direct = not parameter.inputs
                fmt = "fp32" if direct else parameter.direct_format
                operand = model.source(name, view)
                recipe.assign(name, format=fmt, method=cast_direct,
                              source=operand if direct else lossless_bf16(operand))
                formats[name] = fmt
    if set(formats) != set(model.parameters) - {"ngram/table"}:
        raise ValueError("Basalt mapping did not bind every selected parameter")
    q.configure_ngram_gguf(model, recipe, source)
    for layer in range(model.config["num_hidden_layers"]):
        p = f"text/layers/{layer}/"
        if model.config["layer_types"][layer] == "full_attention":
            q._group_attention(recipe, p, formats)
        else:
            n = p + "gdn/"
            q._group_same_format(recipe, [n + r for r in ("query", "key", "value", "z")], formats)
            q._group_same_format(recipe, [n + "a_projection", n + "b_projection"], formats)
        q._group_moe(recipe, p + "moe/", model.config["num_experts"], formats)
    if "mtp" in model.components:
        q._group_attention(recipe, "mtp/layer/", formats)
        q._group_moe(recipe, "mtp/layer/moe/", 512, formats)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows-per-chunk", type=int, default=512)
    parser.add_argument("--max-file-gib", type=float)
    parser.add_argument("--verify-sections", action="store_true", help="check every source section SHA-256 before writing")
    args = parser.parse_args(argv)
    if args.rows_per_chunk < 1 or (args.max_file_gib is not None and args.max_file_gib <= 0):
        parser.error("chunk and shard sizes must be positive")
    with ExitStack() as stack:
        pack = stack.enter_context(BasaltFile(args.input))
        components = components_for(pack)
        if args.verify_sections:
            for section in pack.sections:
                pack.verify(section)
        source = BasaltGGUF(pack)
        root = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="ninfer-basalt-")))
        config = model_config(pack)
        frontend_files(pack, root, config)
        vision = None
        if "vision/mmproj" in pack.sections:
            path = root / "mmproj.gguf"
            size = pack.sections["vision/mmproj"].size
            with path.open("xb") as out:
                for offset in range(0, size, 16 << 20):
                    out.write(pack.read("vision/mmproj", offset, min(16 << 20, size - offset)))
            vision = stack.enter_context(GGUFSource(path))
        base = stack.enter_context(SafetensorsSource(root))
        model = qwen4_exp.build_model(base, components=tuple(components))
        recipe = Recipe(model)
        configure(pack, source, model, recipe, vision)
        # Keep original small resources and provenance too: reconstructing a frontend representation
        # must never discard the author's tokenizer files, template or license.
        for section in pack.sections:
            if section.startswith("tokenizer/") or section in ("meta", "profile", "vision/text_vocab"):
                model.resources["resource/basalt/" + section] = pack.read(section).tobytes()
        if "mtp/draft_vocab" in pack.sections:
            model.resources["resource/basalt/mtp/draft_vocab"] = pack.read("mtp/draft_vocab").tobytes()
        report = convert(model, recipe, args.output, device="cpu", rows_per_chunk=args.rows_per_chunk,
                         name=pack.meta.get("name"),
                         max_file_bytes=None if args.max_file_gib is None else int(args.max_file_gib * (1 << 30)),
                         provenance={"source": str(args.input), "format": "basalt", "version": pack.version,
                                     "sections": {name: entry.sha256 for name, entry in pack.sections.items()},
                                     "converter": "tools.convert.basalt"},
                         progress=lambda i, n, job: print(f"[{i + 1}/{n}] {job.parameters[0]}", flush=True))
        print(f"Wrote {report['payload_bytes']} payload bytes to {args.output}")


if __name__ == "__main__":
    main()
