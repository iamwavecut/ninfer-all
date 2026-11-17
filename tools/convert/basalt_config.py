"""Translate Basalt serving JSON into a NInfer --config JSON and an explicit migration report.

Each model entry is one launch. Unknown or non-equivalent options stop emission until the caller
acknowledges them by name; their original values remain in the report. Hardware auto policies are
reported as policies, never advertised as equal byte reservations or throughput.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .sources.basalt import BasaltFile


def normalized(values):
    if not isinstance(values, dict):
        raise ValueError("configuration sections must be objects")
    result = {}
    for key, value in values.items():
        key = key.strip().lower().removeprefix("--").replace("_", "-")
        if key in result:
            raise ValueError(f"duplicate normalized option {key}")
        result[key] = value
    return result


def integer(value, name, low=0, high=None):
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(f"{name}: expected an integer in {low}..{high or 'unbounded'}")
    return value


def boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name}: expected true or false")
    return value


def pack_sampling(meta):
    names = {"temp": "temperature", "top_p": "top-p", "top_k": "top-k", "min_p": "min-p",
             "penalty_repeat": "repetition-penalty", "penalty_last_n": "penalty-last-n"}
    result = normalized(meta.get("sampling") or {})
    for key, target in names.items():
        value = meta.get("gguf_kv", {}).get("general.sampling." + key)
        if value is not None:
            result[target] = value
    return {key: value for key, value in result.items() if value is not None}


def translate(config, artifact, *, has_mtp, model_index=0, expert_bytes=None,
              has_profile=True, sampling_defaults=None):
    if not isinstance(config, dict):
        raise ValueError("configuration must be an object")
    version = config.get("version", config.get("v", 2 if "models" in config else 1))
    if type(version) is not int or version not in (1, 2):
        raise ValueError("expected Basalt configuration version 1 or 2")
    if "v" in config and (version != 1 or config["v"] != version):
        raise ValueError("v is the legacy version-1 marker")
    unknown_top = set(config) - {"version", "v", "server", "engine", "models"}
    if unknown_top:
        raise ValueError(f"unknown top-level fields: {sorted(unknown_top)}")
    values = normalized(sampling_defaults or {})
    values.update(normalized(config.get("server", {})))
    values.update(normalized(config.get("engine", {})))
    if version == 2:
        models = config.get("models")
        if not isinstance(models, list) or not 0 <= model_index < len(models):
            raise ValueError("select a model entry with --model-index")
        values.update(normalized(models[model_index]))
    elif "models" in config or model_index:
        raise ValueError("models/model-index require configuration version 2")
    original = dict(values)
    out = {"model": str(artifact)}
    notes, unresolved = [], {}

    def take(name, default=None):
        return values.pop(name, default)

    def issue(name, value, reason, targets=()):
        unresolved[name] = {"value": value, "reason": reason}
        if targets:
            unresolved[name]["choose_target"] = list(targets)

    take("model")
    parallel = integer(take("parallel", 1), "parallel", 1, 8)
    full = boolean(take("slot-full-context", False), "slot-full-context")
    total = integer(take("max-context", 8192), "max-context", 1)
    per = total if full else max(256, total // parallel) if parallel > 1 else total
    out.update({"max-concurrency": parallel, "max-context": per, "kv-capacity": per * parallel})
    notes.append(f"{parallel} shared-engine sequences; {per} tokens each, {per * parallel} total KV tokens.")
    notes.append("Explicit KV capacity preserves the context ceiling; NInfer budgets that pool at startup, "
                 "rather than Basalt's elastic backing borrowed from its expert-cache tail.")
    kv = take("kv", "fp16")
    if kv in ("int8", "fp16"):
        out["kv-dtype"] = kv
        notes.append("Native FP16 KV selected: FP32 K/V projections and key norm/RoPE, rounded once on append."
                     if kv == "fp16" else "INT8 KV precision retained; the engines' block layouts and scales differ.")
    else:
        issue("kv", kv, "No matching KV codec. Choose --set kv-dtype=... "
              "explicitly rather than silently changing precision.", ("kv-dtype",))
    window = integer(take("spec", 8 if has_mtp else 0), "spec", 0, 8)
    if window > 1 and not has_mtp:
        raise ValueError("speculation was requested but the source pack has no MTP")
    suffix = integer(take("suffix-draft", 3), "suffix-draft")
    mtp_cap = integer(take("mtp-max-t", 0), "mtp-max-t")
    out["spec"] = "mtp" if window > 1 else "none"
    if window > 1:
        out["draft-tokens"] = min(window, mtp_cap or window) - 1
        out["draft-min-p"] = 0
        notes.append(f"Basalt spec={window} counts the seed; the MTP cap gives {out['draft-tokens']} drafts.")
        if out["draft-tokens"] == 0:
            issue("mtp-max-t", mtp_cap, "NInfer MTP needs at least one draft; choose a positive draft budget explicitly.",
                  ("draft-tokens",))
            del out["draft-tokens"]
    # Flash-Next uses context lookup, not the Qwen3.5-only ngram-copy proposer.
    out["ngram-draft-tokens"] = 0
    out["lookup-ngram"] = suffix if window > 1 else 0
    if out["lookup-ngram"]:
        notes.append("Prompt lookup keeps the suffix-match threshold. NInfer shares the MTP draft width; "
                     "Basalt can verify a longer lookup window and gates it with its draft policy.")
    relaxed = take("relaxed-drafting", 0)
    if relaxed not in (0, False):
        issue("relaxed-drafting", relaxed, "Lossy relaxed acceptance has no equivalent; exact acceptance is retained.")
    for name, target in {"host": "host", "port": "port", "alias": "model-id", "api-key": "api-key",
                         "seed": "seed"}.items():
        if name in values:
            out[target] = take(name)
    ranges = {"temperature": (0, 2), "top-p": (0, 1), "min-p": (0, 1),
              "presence-penalty": (-2, 2), "frequency-penalty": (-2, 2), "top-k": (0, 20)}
    for name, (low, high) in ranges.items():
        if name not in values:
            continue
        value = take(name)
        if name == "top-k" and value == -1:
            value = 0
        if (type(value) in (int, float) and math.isfinite(value) and low <= value <= high
                and (name != "top-k" or int(value) == value)):
            out[name] = int(value) if name == "top-k" else value
        else:
            issue(name, value, f"NInfer accepts {low}..{high}; no sampling value was clamped.", (name,))
    repeat = take("repetition-penalty", 1)
    last_n = take("penalty-last-n", None)
    if repeat != 1:
        issue("repetition-penalty", repeat, "NInfer has additive presence/frequency penalties, not this multiplicative penalty.")
    if last_n is not None:
        issue("penalty-last-n", last_n, "The sampling penalty window has no equivalent launch option.")
    if "pool-workers" in values:
        workers = integer(take("pool-workers"), "pool-workers")
        if workers > 256:
            issue("pool-workers", workers, "NInfer accepts at most 256 expert CPU workers.")
        elif workers:
            out["expert-cpu-threads"] = workers
    gpu = take("gpu", 0)
    if type(gpu) is int and gpu >= 0 or isinstance(gpu, str) and gpu.isdigit():
        out["device"] = int(gpu)
    elif gpu == "auto":
        issue("gpu", gpu, "Basalt selects cards dynamically; choose a NInfer --set device=N or devices=... .",
              ("device", "devices"))
    else:
        issue("gpu", gpu, "Peer expert tiers differ from pipeline/tensor parallelism; choose a device topology.",
              ("device", "devices"))
    vision = take("vision", "off")
    if vision == "off":
        out["vision"] = False
    elif vision in ("dynamic", "cpu"):
        issue("vision", vision, "Flash-Next currently requires resident Vision; choose --set vision=true "
              "to accept that VRAM cost, or --set vision=false explicitly.", ("vision",))
    else:
        issue("vision", vision, "A separate encoder GPU has no identical NInfer placement.", ("vision",))
    ple = take("ple-io", "direct")
    if ple == "ram":
        out.update({"ngram-residency": "ram", "ngram-lock": True})
    elif ple == "direct":
        out.update({"ngram-residency": "disk", "ngram-io": "direct"})
    else:
        issue("ple-io", ple, "Unknown table I/O policy.")
    out["ngram-io-depth"] = integer(take("ple-inflight", 256), "ple-inflight", 1, 1024)
    resident = boolean(take("resident-experts", False), "resident-experts")
    mmap = boolean(take("mmap-experts", False), "mmap-experts")
    out["expert-residency"] = "host" if resident else "disk"
    if resident:
        notes.append("Host residency keeps NInfer's expert bank pinned; Basalt's GPU-complement RAM saving/fallback is engine-specific.")
    elif not mmap:
        issue("resident-experts", None, "Basalt's automatic host/complement fallback differs; choose host or disk explicitly.",
              ("expert-residency",))
    cache = take("expert-cache", 0)
    profiled = has_profile or bool(values.get("expert-profile"))
    if cache in ("auto", -1) or type(cache) is int and cache == 0 and profiled:
        out["expert-cache-mib"] = "auto"
        notes.append("Automatic expert cache uses NInfer's memory planner, not Basalt's slot reservation.")
    elif type(cache) is int and cache == 0:
        out["expert-cache-mib"] = 0
    elif type(cache) is int and cache > 0 and expert_bytes is not None:
        out["expert-cache-mib"] = (cache * expert_bytes + (1 << 20) - 1) // (1 << 20)
        notes.append("Expert slot budget converted to MiB and rounded up; allocation granularity differs.")
    else:
        issue("expert-cache", cache, "A numeric slot budget needs the source expert blob size.")
    prefill = take("prefill", "auto")
    if prefill == "auto":
        notes.append("Automatic prefill uses NInfer's device profile and chunk defaults.")
    elif type(prefill) is int and prefill > 0 and prefill % 128 == 0:
        out["prefill-chunk"] = prefill
    else:
        issue("prefill", prefill, "NInfer requires positive chunks divisible by 128; no silent rounding.")
    jev = integer(take("jev", 0), "jev", 0, 8)
    if jev:
        notes.append(f"Jev endpoints are built in; Basalt's {jev} dedicated lanes are replaced by the shared NInfer scheduler.")
    elif "jev" in original:
        issue("jev", jev, "NInfer exposes decision endpoints unconditionally; acknowledge this access-policy change.")
    else:
        notes.append("NInfer exposes decision endpoints by default; this launch does not disable them.")
    for name, value in values.items():
        issue(name, value, "No verified one-to-one launch equivalent; retain intent with --set and acknowledge this field.")
    return out, {"source_version": version, "model_index": model_index, "source_options": original,
                 "pack_sampling": sampling_defaults or {},
                 "notes": notes, "unresolved": unresolved}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--acknowledge", action="append", default=[], metavar="SOURCE_OPTION")
    parser.add_argument("--set", action="append", default=[], metavar="TARGET_OPTION=JSON_VALUE")
    args = parser.parse_args(argv)
    with BasaltFile(args.pack) as pack:
        sizes = {int(line.split()[4]) for line in pack.text("experts.layout").splitlines()
                 if line.strip() and not line.startswith("#")}
        out, report = translate(json.loads(args.input.read_text(encoding="utf-8")),
                                args.artifact.resolve(), has_mtp="mtp/dense" in pack.sections,
                                model_index=args.model_index,
                                expert_bytes=next(iter(sizes)) if len(sizes) == 1 else None,
                                has_profile=("profile" in pack.sections or
                                    pack.meta["gguf_kv"].get("qwen4exp.expert_count") == 512),
                                sampling_defaults=pack_sampling(pack.meta))
    overrides = {}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            parser.error("--set needs OPTION=JSON_VALUE")
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        if value is None or isinstance(value, (dict, list)):
            parser.error("--set values must be scalars")
        overrides.update(normalized({key: value}))
    acknowledgements = set(args.acknowledge)
    if acknowledgements - report["unresolved"].keys():
        parser.error("--acknowledge must name an unresolved source option")
    report.update(overrides=overrides, acknowledged=sorted(acknowledgements))
    pending = report["unresolved"].keys() - acknowledgements
    choices = {name: issue["choose_target"] for name, issue in report["unresolved"].items()
               if issue.get("choose_target") and not any(key in overrides for key in issue["choose_target"])}
    report_path = args.output.with_name(args.output.name + ".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if pending or choices:
        missing = "; ".join(f"{name}: set {' or '.join(keys)}" for name, keys in choices.items())
        parser.error(f"resolve {', '.join(sorted(pending)) or 'target choices'}; {missing}; "
                     f"see {report_path}; launch file was not written")
    out.update(overrides)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"ninfer-serve --config {json.dumps(str(args.output))}")


if __name__ == "__main__":
    main()
