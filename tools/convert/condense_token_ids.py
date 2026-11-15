"""Rewrite every *_token_id in the condensed model's configs to the condensed vocabulary.

usage: python3 condense_token_ids.py ORIGINAL_DIR CONDENSED_DIR

Reads ids from the ORIGINAL configs, finds the token string each id names in the original
tokenizer.json, and writes the id of that same string in CONDENSED_DIR's tokenizer.json into the
CONDENSED_DIR copy of generation_config.json and config.json (also under text_config).
An id whose token does not exist in the condensed vocabulary is reported and the run fails.
"""

import json
import sys
from pathlib import Path

original, condensed = Path(sys.argv[1]), Path(sys.argv[2])


def table(directory: Path) -> dict[str, int]:
    tok = json.loads((directory / "tokenizer.json").read_text())
    ids = dict(tok["model"]["vocab"])
    for entry in tok.get("added_tokens", []):
        ids[entry["content"]] = entry["id"]
    return ids


old, new = table(original), table(condensed)
old_by_id = {i: t for t, i in old.items()}
failures = []


def convert(owner: str, key: str, value):
    if value is None:
        return None
    if isinstance(value, list):
        return [convert(owner, key, v) for v in value]
    token = old_by_id.get(value)
    if token is None or token not in new:
        failures.append(f"{owner}: {key}={value} ({token!r}) is not in the condensed vocabulary")
        return value
    print(f"{owner}: {key} {value} ({token}) -> {new[token]}")
    return new[token]


def rewrite(owner: str, data: dict) -> None:
    for key, value in list(data.items()):
        if key.endswith("_token_id") or key.endswith("_token_ids"):
            data[key] = convert(owner, key, value)
        elif isinstance(value, dict):
            rewrite(f"{owner}.{key}", value)


for name in ("generation_config.json", "config.json"):
    source = original / name
    if not source.exists():
        continue
    data = json.loads(source.read_text())
    rewrite(name, data)
    if name == "config.json":
        size = -(-len(new) // 128) * 128  # padded to the kernel's 128-row tiles
        data["vocab_size"] = size
        if "text_config" in data:
            data["text_config"]["vocab_size"] = size
    (condensed / name).write_text(json.dumps(data, indent=2))

cfg = condensed / "tokenizer_config.json"
if cfg.exists():
    data = json.loads(cfg.read_text())
    for key in ("eos_token", "bos_token", "pad_token", "unk_token"):
        token = data.get(key)
        token = token.get("content") if isinstance(token, dict) else token
        if token and token not in new:
            failures.append(f"tokenizer_config.json: {key}={token!r} is not in the condensed vocabulary")

if failures:
    sys.exit("\n".join(failures))
print("ok")
