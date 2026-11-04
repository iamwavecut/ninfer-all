#!/usr/bin/env python3
"""Append Qwen3.8-Flash-Next's DFlash2 drafter from its llama.cpp GGUF without changing the target.

Every drafter matrix keeps its ggml blocks as stored (``gguf_blocks_v1``); the norms and the
convolution bases, F32 words in the GGUF whose values are BF16, become BF16 words (a value that is
not BF16 is refused). Each layer's context K/V projections share the block's own K/V objects, as
in the official recipe, and the drafter invokes the target's output head with the activation
contract the target declares for its own final hidden state. The component's config is the
release's ``config.json``, checked against the target's text geometry.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from scripts.pods.attach_mtp import spec, verify_copy
from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.qwen3_5 import draft_config
from tools.convert.sources.gguf import BLOCK_FORMATS, TYPE_F32, GGUFFile

PREFIX = "dflash2-import/"

# Logical parameter (after "dflash2/") -> GGUF tensor, the drafter-wide tensors.
GLOBAL = {
    "feature_projection": "fc.weight",
    "context_norm": "enc.output_norm.weight",
    "final_norm": "output_norm.weight",
    "candidate_selector/hidden_projection": "selector_hidden.weight",
    "candidate_selector/predecessor_codebook": "selector_predecessor.weight",
    "candidate_selector/successor_codebook": "selector_successor.weight",
}
# Logical parameter (after "dflash2/layers/{l}/") -> GGUF tensor (after "blk.{l}.").
LAYER = {
    "input_norm": "attn_norm.weight",
    "post_attention_norm": "ffn_norm.weight",
    "attention/query": "attn_q.weight",
    "attention/key": "attn_k.weight",
    "attention/value": "attn_v.weight",
    "attention/output": "attn_output.weight",
    "attention/query_norm": "attn_q_norm.weight",
    "attention/key_norm": "attn_k_norm.weight",
    "mlp/gate": "ffn_gate.weight",
    "mlp/up": "ffn_up.weight",
    "mlp/down": "ffn_down.weight",
    "attention_conv/base_kernel": "attn_conv_base",
    "attention_conv/kernel_projection": "attn_conv_proj.weight",
    "mlp_conv/base_kernel": "ffn_conv_base",
    "mlp_conv/kernel_projection": "ffn_conv_proj.weight",
}


def geometry(config, target):
    """Each parameter's row-major shape and the activations its Uses name."""
    h, d = target["hidden_size"], config["head_dim"]
    q, k = config["num_attention_heads"] * d, config["num_key_value_heads"] * d
    i, draft = config["intermediate_size"], config["dflash_config"]
    taps, groups = draft["conv_kernel_size"], h // draft["conv_group_size"]
    n = "dflash2/"
    out = {
        n + "feature_projection": ((h, len(draft["target_layer_ids"]) * h), (n + "target_features",)),
        n + "context_norm": ((h,), ()),
        n + "final_norm": ((h,), ()),
        n + "candidate_selector/hidden_projection": ((draft["selector_rank"], h), (n + "final_hidden",)),
        n + "candidate_selector/predecessor_codebook": ((target["vocab_size"], draft["selector_rank"]), ()),
        n + "candidate_selector/successor_codebook": ((target["vocab_size"], draft["selector_rank"]), ()),
    }
    for layer in range(config["num_hidden_layers"]):
        p = f"{n}layers/{layer}/"
        out.update({
            p + "input_norm": ((h,), ()),
            p + "post_attention_norm": ((h,), ()),
            p + "attention/query": ((q, h), (p + "query_projection_input",)),
            p + "attention/key": ((k, h), (p + "query_projection_input",)),
            p + "attention/value": ((k, h), (p + "query_projection_input",)),
            p + "attention/output": ((h, q), (p + "attention_output",)),
            p + "attention/query_norm": ((d,), ()),
            p + "attention/key_norm": ((d,), ()),
            p + "mlp/gate": ((i, h), (p + "mlp_input",)),
            p + "mlp/up": ((i, h), (p + "mlp_input",)),
            p + "mlp/down": ((h, i), (p + "mlp_product",)),
            p + "attention_conv/base_kernel": ((2, taps, h), ()),
            p + "attention_conv/kernel_projection": ((2 * taps * groups, h), (p + "attention_conv_input",)),
            p + "mlp_conv/base_kernel": ((2, taps, h), ()),
            p + "mlp_conv/kernel_projection": ((2 * taps * groups, h), (p + "mlp_conv_input",)),
        })
    return out


def source_name(parameter):
    name = parameter.removeprefix("dflash2/")
    if name.startswith("layers/"):
        _, layer, rest = name.split("/", 2)
        return f"blk.{layer}.{LAYER[rest]}"
    return GLOBAL[name]


class Encoded:
    """The drafter's objects, read from the GGUF in the form the artifact stores them."""

    def __init__(self, gguf):
        self.gguf, self.objects = gguf, {}

    def add(self, object_id, tensor, shape):
        info = self.gguf.info(tensor)
        if tuple(info.shape) != tuple(shape):
            raise ValueError(f"{tensor}: shape {info.shape}, expected {shape}")
        if info.type_id in BLOCK_FORMATS:
            fmt = BLOCK_FORMATS[info.type_id]
            self.objects[object_id] = ("blocks", tensor)
            return TensorSpec(object_id, tuple(shape), fmt.name, "gguf_blocks_v1")
        if info.type_id != TYPE_F32:
            raise ValueError(f"{tensor}: unsupported GGUF type {info.type_name}")
        words = self.gguf.read_direct(tensor).astype("<f4").view("<u4")
        if np.any(words & 0xFFFF):
            raise ValueError(f"{tensor}: F32 words are not BF16 values")
        self.objects[object_id] = ("bf16", (words >> 16).astype("<u2").tobytes())
        return TensorSpec(object_id, tuple(shape), "bf16", "contiguous_le_v1")

    def iter_object(self, object_id):
        kind, value = self.objects[object_id]
        if kind == "bf16":
            yield value
            return
        data = self.gguf.tensor_bytes(value)
        for begin in range(0, len(data), 8 << 20):
            yield data[begin:begin + (8 << 20)].tobytes()


def plan(base, gguf, raw_config):
    """The drafter's component, objects, bindings and Uses beside the target's."""
    if "dflash2" in base.components or any(k.startswith("dflash2/") for k in base.bindings):
        raise ValueError("target already contains DFlash2; refusing to replace it")
    text = base.components.get("text", {}).get("config", {})
    if text.get("architectures") != ["Qwen4ExpForCausalLM"]:
        raise ValueError("expected the Flash-Next architecture")
    config = draft_config(raw_config, text, "dflash2")
    shapes = geometry(config, text)
    encoded = Encoded(gguf)
    existing = {obj.id for obj in base.objects}
    specs, bindings, uses = [], {}, []
    shared = {}
    for parameter, (shape, inputs) in shapes.items():
        role = parameter.rsplit("/", 1)[-1]
        if role in ("context_key", "context_value"):
            continue
        tensor = source_name(parameter)
        object_id = PREFIX + tensor
        if object_id in existing:
            raise ValueError(f"object {object_id} already exists in the target")
        object_spec = encoded.add(object_id, tensor, shape)
        specs.append(object_spec)
        bindings[parameter] = {"object": object_id}
        policy = "AllowA8" if object_spec.layout == "gguf_blocks_v1" else None
        for name in inputs:
            if policy is None:
                raise ValueError(f"{parameter}: a projection must keep its GGUF blocks")
            uses.append({"parameter": parameter, "input": name, "activation_policy": policy})
        if role in ("key", "value"):
            shared[parameter] = object_id
    # Each layer's context projections are its own K/V matrices over the committed features.
    for parameter, object_id in shared.items():
        context = parameter.rsplit("/", 1)[0] + "/context_" + parameter.rsplit("/", 1)[1]
        bindings[context] = {"object": object_id}
        uses.append({"parameter": context, "input": "dflash2/context_input",
                     "activation_policy": "AllowA8"})
    used = {source_name(p) for p in shapes if not p.endswith(("/context_key", "/context_value"))}
    if set(gguf.tensors) != used:
        raise ValueError(f"unexpected GGUF tensors: {sorted(set(gguf.tensors) ^ used)[:8]}")
    head = [u for u in base.uses if u["parameter"] == "text/output_head"
            and u["input"] == "text/final_hidden"]
    if len(head) != 1:
        raise ValueError("target output head has no unambiguous activation contract")
    head_use = json.loads(json.dumps(head[0]))
    head_use["input"] = "dflash2/final_hidden"
    components = {**json.loads(json.dumps(base.components)),
                  "dflash2": {"config": config, "target": "text"}}
    all_bindings = {**json.loads(json.dumps(base.bindings)), **bindings}
    all_uses = [*json.loads(json.dumps(list(base.uses))), *uses, head_use]
    return specs, encoded, components, all_bindings, all_uses


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def attach(base, gguf, raw_config, output: Path, *, provenance=None):
    report_path = Path(str(output) + ".conversion.json")
    if output.exists() or report_path.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    specs, encoded, components, bindings, uses = plan(base.directory, gguf, raw_config)
    transformation = {"type": "append_dflash2_gguf", "parent_artifact_id": base.artifact_id.hex(),
                      "source": provenance or {},
                      "dflash2_bytes": sum(len(b) for o in specs for b in encoded.iter_object(o.id))}
    all_specs = [spec(obj, obj.id) for obj in base.objects] + specs
    sources = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with ArtifactWriter(output, all_specs, components=components, bindings=bindings, uses=uses,
                        metadata=base.directory.metadata,
                        provenance={**base.directory.provenance,
                                    "dflash2_attachment": transformation}) as writer:
        total = sum(obj.bytes for obj in base.objects)
        done, last = 0, time.monotonic()
        for obj in base.objects:
            writer.write_object(obj.id, base.iter_object(obj.id))
            sources[obj.id] = (base, obj.id)
            done += obj.bytes
            if time.monotonic() - last >= 15:
                print(json.dumps({"stage": "base", "bytes": done, "total": total}), flush=True)
                last = time.monotonic()
        for object_spec in specs:
            writer.write_object(object_spec.id, encoded.iter_object(object_spec.id))
            sources[object_spec.id] = (encoded, object_spec.id)
        identity = writer.artifact_id.hex()
    with Artifact(output) as result:
        digests, file_digest = verify_copy(result, output, sources)
        if (result.directory.components != components or result.directory.bindings != bindings or
                list(result.directory.uses) != uses):
            raise ValueError("output metadata mismatch")
    report = {"artifact_id": identity, "file": output.name, "bytes": output.stat().st_size,
              "transformation": transformation, "components": components,
              "base_objects_verified": len(base.objects), "dflash2_objects_verified": len(specs),
              "dflash2_bindings": sum(k.startswith("dflash2/") for k in bindings),
              "object_sha256": digests, "file_sha256": file_digest,
              "elapsed_seconds": time.monotonic() - started}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="the target .ninfer artifact")
    parser.add_argument("--gguf", type=Path, required=True, help="the drafter's GGUF")
    parser.add_argument("--config", type=Path, required=True, help="the drafter's config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[], metavar="KEY=VALUE",
                        help="provenance recorded with the attachment (repeatable)")
    args = parser.parse_args()
    provenance = dict(item.split("=", 1) for item in args.source)
    provenance["gguf_sha256"] = file_sha(args.gguf)
    provenance["config_sha256"] = file_sha(args.config)
    raw = json.loads(args.config.read_text())
    with Artifact(args.base) as base, GGUFFile(args.gguf) as gguf:
        report = attach(base, gguf, raw, args.output, provenance=provenance)
    print(json.dumps({k: report[k] for k in ("artifact_id", "file", "bytes", "file_sha256",
                                              "dflash2_objects_verified", "dflash2_bindings")}))


if __name__ == "__main__":
    main()
