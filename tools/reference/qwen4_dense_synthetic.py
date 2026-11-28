"""A synthetic dense Qwen4 safetensors checkpoint at the geometry of tests/fixtures/qwen4.

No dense Qwen4 checkpoint is public; this one carries the converter's and the FP64 reference's
tensor names and shapes with seeded random weights: a SwiGLU MLP per layer, one PLE block on layer
index 1, an MTP block, and an n-gram table whose rows are FP8 codes times 2^-8 with a 448 in every
row, so the converter's FP8 row encoding is lossless. The tokenizer files come from a Qwen3.5-frontend
model directory (Qwen3.8-Flash-Next's). See docs/qwen4-dense.md#checks-on-a-synthetic-checkpoint.

    python3.11 -m tools.reference.qwen4_dense_synthetic --out DIR --tokenizer DIR [--layers 64]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from tools.convert.qwen4_exp import ngram_hash_constants, text_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--fixture", type=Path,
                        default=Path(__file__).resolve().parents[2] / "tests/fixtures/qwen4/config.json")
    parser.add_argument("--layers", type=int, default=64, help="a multiple of 4")
    parser.add_argument("--device", default="cpu", help="the torch device that draws the weights")
    parser.add_argument("--no-ple", action="store_true",
                        help="a model without PLE: no PLE fields, tensors or n-gram table")
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.fixture.read_text())
    source.pop("vision_config", None)
    t = source["text_config"]
    t["num_hidden_layers"] = args.layers
    t["layer_types"] = (["linear_attention"] * 3 + ["full_attention"]) * (args.layers // 4)
    t["ngram_vocab_size_base"] = 4096
    t["split_ngram_parts"] = 4
    if args.no_ple:
        t["ple_layer_ids"] = []
        for key in ("ple_embed_dim", "ple_conv_kernel_size", "ngram_size", "heads_per_ngram",
                    "ngram_vocab_size_base", "make_ngram_vocab_size_divisible_by",
                    "split_ngram_parts"):
            t.pop(key, None)
    assert len(t["layer_types"]) == args.layers
    (out / "config.json").write_text(json.dumps(source, indent=2))
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "generation_config.json"):
        shutil.copy(args.tokenizer / name, out / name)
    text = text_config(source)

    h, hc, low = t["hidden_size"], t["hc_count"], t["hc_lowrank"]
    width = hc * h
    vocab, inter = t["vocab_size"], t["intermediate_size"]
    nk, dk, nv, dv = (t["linear_num_key_heads"], t["linear_key_head_dim"],
                      t["linear_num_value_heads"], t["linear_value_head_dim"])
    heads, kv, d = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    ih, idim = t["indexer_n_heads"], t["indexer_head_dim"]
    ple_dim = t.get("ple_embed_dim", 0)
    gen = torch.Generator(device=args.device)
    gen.manual_seed(20261122)

    def normal(shape, std):
        return (torch.randn(shape, generator=gen, device=args.device) * std).to(
            torch.bfloat16).cpu()

    def uniform(shape, lo, hi):
        return (torch.rand(shape, generator=gen, device=args.device) * (hi - lo) + lo).to(
            torch.bfloat16).cpu()

    def linear(rows, cols, gain=1.0):
        return normal((rows, cols), gain / math.sqrt(cols))

    def hc_block(prefix, inject):
        out = {prefix + "hc_norm.weight": normal((width,), 0.1),
               prefix + "input_mix_weight_down.weight": linear(low, width),
               prefix + "input_mix_weight_up.weight": linear(width, low)}
        if inject:
            out[prefix + "block_inject_weight.weight"] = linear(hc, width)
        return out

    def attention(prefix):
        a = prefix + "self_attn."
        return {a + "q_proj.weight": linear(2 * heads * d, h), a + "k_proj.weight": linear(kv * d, h),
                a + "v_proj.weight": linear(kv * d, h), a + "o_proj.weight": linear(h, heads * d, 0.5),
                a + "q_norm.weight": normal((d,), 0.1), a + "k_norm.weight": normal((d,), 0.1),
                a + "indexer.index_qk_proj.weight": linear((ih + 1) * idim, h),
                a + "indexer.q_layernorm.weight": normal((idim,), 0.1),
                a + "indexer.k_layernorm.weight": normal((idim,), 0.1)}

    def gdn(prefix):
        g = prefix + "linear_attn."
        channels = 2 * nk * dk + nv * dv
        return {g + "in_proj_qkv.weight": linear(channels, h), g + "in_proj_z.weight": linear(nv * dv, h),
                g + "in_proj_a.weight": linear(nv, h), g + "in_proj_b.weight": linear(nv, h),
                g + "A_log": torch.log(uniform((nv,), 1.0, 16.0).float()).to(torch.bfloat16),
                g + "dt_bias": normal((nv,), 0.5),
                g + "conv1d.weight": normal((channels, 1, t["linear_conv_kernel_dim"]), 0.5),
                g + "norm.weight": (1.0 + normal((dv,), 0.1).float()).to(torch.bfloat16),
                g + "out_proj.weight": linear(h, nv * dv, 0.5)}

    def mlp(prefix, middle):
        m = prefix + "mlp."
        return {m + "gate_proj.weight": linear(middle, h), m + "up_proj.weight": linear(middle, h),
                m + "down_proj.weight": linear(h, middle, 0.5)}

    def ple(prefix):
        p = prefix + "ple."
        out = {p + "key_proj.weight": linear(width, ple_dim), p + "value_proj.weight": linear(h, ple_dim),
               p + "norm_key.weight": normal((width,), 0.1), p + "norm_query.weight": normal((width,), 0.1),
               p + "norm_conv.weight": normal((width,), 0.1),
               p + "conv1d.weight": normal((width, 1, t["ple_conv_kernel_size"]), 0.5)}
        multipliers, head_vocab = ngram_hash_constants(text)
        offsets = [sum(head_vocab[:i]) for i in range(len(head_vocab))]
        e = p + "ple_embedding."
        out[e + "layer_multipliers"] = torch.tensor(multipliers, dtype=torch.int64)
        out[e + "ngram_heads_vocab_sizes"] = torch.tensor(head_vocab, dtype=torch.int64)
        out[e + "ngram_heads_offsets"] = torch.tensor(offsets, dtype=torch.int64)
        divisible = t["make_ngram_vocab_size_divisible_by"]
        rows = -(-sum(head_vocab) // divisible) * divisible
        row = ple_dim // len(head_vocab)
        codes = torch.randint(-127, 127, (rows, row), generator=gen, device=args.device).to(torch.int8)
        table = codes.view(torch.float8_e4m3fn).float()
        table = torch.nan_to_num(table, nan=0.0).clamp(-448, 448)
        table[:, 0] = 448.0
        table = (table * 2.0 ** -8).to(torch.bfloat16).cpu()
        parts = t["split_ngram_parts"]
        per = -(-rows // parts)
        for i in range(parts):
            out[e + f"ngram_embedding.shard_{i}.weight"] = table[i * per:(i + 1) * per].clone()
        return out

    index = {}
    def write(name, tensors):
        save_file(tensors, out / name)
        for key in tensors:
            index[key] = name
        print(f"wrote {name}: {sum(v.numel() for v in tensors.values()) / 1e9:.3f} B values",
              flush=True)

    P = "model.language_model."
    write("embed.safetensors", {P + "embed_tokens.weight": normal((vocab, h), 1.0)})
    write("head.safetensors", {"lm_head.weight": linear(vocab, h, 6.0),
                               **hc_block(P + "hyper_connection_mixer.", False)})
    ple_blocks = [i - 1 for i in t["ple_layer_ids"]]
    for i, kind in enumerate(t["layer_types"]):
        prefix = P + f"layers.{i}."
        tensors = {**hc_block(prefix + "attn_hyper_connection.", True),
                   **hc_block(prefix + "mlp_hyper_connection.", True),
                   **(attention(prefix) if kind == "full_attention" else gdn(prefix)),
                   **mlp(prefix, inter)}
        if i in ple_blocks:
            tensors.update(ple(prefix))
        write(f"layer-{i:03d}.safetensors", tensors)
    m = "mtp."
    write("mtp.safetensors", {m + "pre_fc_norm_embedding.weight": normal((h,), 0.1),
                              m + "pre_fc_norm_hidden.weight": normal((width,), 0.1),
                              m + "fc_embedding.weight": linear(h, h),
                              m + "fc_hidden.weight": linear(h, h),
                              **hc_block(m + "layers.0.attn_hyper_connection.", True),
                              **hc_block(m + "layers.0.mlp_hyper_connection.", True),
                              **attention(m + "layers.0."), **mlp(m + "layers.0.", inter),
                              **hc_block(m + "hyper_connection_mixer.", False)})
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": index}, indent=1))
    print("SYNTH DONE", flush=True)


if __name__ == "__main__":
    main()
