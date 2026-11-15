"""Carry a token-frequency ranking (int64 counts, one per token id) over to a condensed vocabulary.

usage: python3 condense_ranking.py ORIGINAL_DIR CONDENSED_DIR RANKING_IN OUT

ORIGINAL_DIR   original model dir (tokenizer.json, config.json) the ranking was built for
CONDENSED_DIR  dir written by condense_tokenizer.py (condensed tokenizer.json, config.json)
RANKING_IN     e.g. freq_corpus/fixtures/ranking/ranking.train.counts.i64
OUT            new file, int64 little-endian, vocab_size(condensed) entries

Each condensed id takes the count of the same token string in the original vocabulary; a token the
original does not have gets 0 (the proposal's special-token ids are forced into the shortlist
anyway).
"""

import json
import sys
from pathlib import Path

import numpy as np

original, condensed = Path(sys.argv[1]), Path(sys.argv[2])
ranking_in, out = Path(sys.argv[3]), Path(sys.argv[4])


def token_ids(directory: Path) -> dict[str, int]:
    tok = json.loads((directory / "tokenizer.json").read_text())
    ids = dict(tok["model"]["vocab"])
    for entry in tok.get("added_tokens", []):
        ids[entry["content"]] = entry["id"]
    return ids


def vocab_size(directory: Path) -> int:
    config = json.loads((directory / "config.json").read_text())
    return int(config.get("vocab_size") or config["text_config"]["vocab_size"])


old_vocab, new_vocab = vocab_size(original), vocab_size(condensed)
size = ranking_in.stat().st_size
if size == 0 or size % (old_vocab * 8):
    sys.exit(f"{ranking_in}: {size} bytes is not a whole number of {old_vocab}-entry int64 rows")
print(f"ranking has {size // (old_vocab * 8)} row(s) of {old_vocab}; the first is used")
old_counts = np.fromfile(ranking_in, dtype="<i8", count=old_vocab)

old_ids, new_ids = token_ids(original), token_ids(condensed)
counts = np.zeros(new_vocab, dtype="<i8")
missing = 0
for token, new_id in new_ids.items():
    if not 0 <= new_id < new_vocab:
        sys.exit(f"condensed id {new_id} of {token!r} is outside vocab_size {new_vocab}")
    old_id = old_ids.get(token)
    if old_id is None or not 0 <= old_id < old_vocab:
        missing += 1
        continue
    counts[new_id] = old_counts[old_id]

kept = int(old_counts[[old_ids[t] for t in new_ids if t in old_ids and old_ids[t] < old_vocab]].sum())
print(f"{new_vocab} ids; {missing} without a counterpart in the original (count 0)")
print(f"mass kept: {kept / max(int(old_counts.sum()), 1):.4%} of the original counts")
counts.tofile(out)
print("wrote", out, out.stat().st_size, "bytes")
