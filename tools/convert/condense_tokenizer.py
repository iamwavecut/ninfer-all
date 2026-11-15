"""Rebuild tokenizer.json / tokenizer_config.json of a vocabulary-condensed Qwen GGUF.

usage: python3 condense_tokenizer.py MODEL.gguf ORIGINAL_DIR CONDENSED_DIR

ORIGINAL_DIR holds the original tokenizer.json / tokenizer_config.json (it is only read).
CONDENSED_DIR receives the rewritten files and config.json's vocab_size.
The GGUF's tokenizer.ggml.tokens decides every id; nothing is guessed from the original.
"""

import json
import shutil
import sys
from pathlib import Path

from gguf import GGUFReader

gguf_path, original, target = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
reader = GGUFReader(gguf_path)


def strings(key):
    field = reader.fields.get(key)
    if field is None:
        return None
    return [bytes(field.parts[i]).decode("utf-8") for i in field.data]


def ints(key):
    field = reader.fields.get(key)
    if field is None:
        return None
    return [int(field.parts[i][0]) for i in field.data]


tokens = strings("tokenizer.ggml.tokens")
merges = strings("tokenizer.ggml.merges")
types = ints("tokenizer.ggml.token_type")
if tokens is None:
    sys.exit("the GGUF has no tokenizer.ggml.tokens")
print(f"GGUF: {len(tokens)} tokens, {len(merges or [])} merges")

tok = json.loads((original / "tokenizer.json").read_text())
cfg_path = original / "tokenizer_config.json"
cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

old_vocab = tok["model"]["vocab"]
new_vocab = {token: i for i, token in enumerate(tokens)}
if len(new_vocab) != len(tokens):
    sys.exit("duplicate token strings in the GGUF list")

kept_same = sum(1 for t, i in new_vocab.items() if old_vocab.get(t) == i)
moved = sum(1 for t, i in new_vocab.items() if t in old_vocab and old_vocab[t] != i)
foreign = [t for t in new_vocab if t not in old_vocab]
print(f"same id as original: {kept_same}; renumbered: {moved}; not in original: {len(foreign)}")
if foreign:
    print("  e.g.", foreign[:5])

# Merges keep the original's JSON shape (strings "a b" or [a, b] pairs).
tok["model"]["vocab"] = new_vocab
if merges is not None:
    pairs = tok["model"].get("merges") or []
    as_lists = bool(pairs) and isinstance(pairs[0], list)
    kept = [m for m in merges if m.split(" ", 1)[0] in new_vocab]
    tok["model"]["merges"] = [m.split(" ", 1) for m in kept] if as_lists else kept
else:
    print("warning: the GGUF has no merges; the original merges are kept where both sides survive")
    keep = lambda m: all(p in new_vocab for p in (m if isinstance(m, list) else m.split(" ", 1)))
    tok["model"]["merges"] = [m for m in tok["model"]["merges"] if keep(m)]

# Added (special) tokens follow their new ids; dropped ones are removed.
added, dropped = [], []
for entry in tok.get("added_tokens", []):
    new_id = new_vocab.get(entry["content"])
    if new_id is None:
        dropped.append(entry["content"])
        continue
    added.append({**entry, "id": new_id})
tok["added_tokens"] = sorted(added, key=lambda e: e["id"])
if dropped:
    print(f"added tokens dropped by the condensation: {dropped}")

if "added_tokens_decoder" in cfg:
    decoder = {}
    for entry in tok["added_tokens"]:
        old = next((v for v in cfg["added_tokens_decoder"].values()
                    if v.get("content") == entry["content"]), {})
        decoder[str(entry["id"])] = old or {"content": entry["content"], "special": entry["special"]}
    cfg["added_tokens_decoder"] = decoder

target.mkdir(parents=True, exist_ok=True)
(target / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False, indent=2))
if cfg:
    (target / "tokenizer_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

# vocab_size, and any special-token id stored as a number in the model configs.
for name in ("config.json", "generation_config.json"):
    source = original / name
    if not source.exists():
        continue
    data = json.loads(source.read_text())
    if name == "config.json":
        data["vocab_size"] = -(-len(tokens) // 128) * 128
        if "text_config" in data:
            data["text_config"]["vocab_size"] = -(-len(tokens) // 128) * 128
    (target / name).write_text(json.dumps(data, indent=2))
    ids = [(k, v) for k, v in json.dumps(data).count("_token_id") and data.items() if k.endswith("_token_id")]
    if ids:
        print(f"check these ids in {name} (condensing may have moved them): {ids}")

for name in ("chat_template.jinja", "special_tokens_map.json", "vocab.json", "merges.txt"):
    if (original / name).exists() and name in ("chat_template.jinja", "special_tokens_map.json"):
        shutil.copy(original / name, target / name)
print("wrote", target)
