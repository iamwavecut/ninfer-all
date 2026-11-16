"""The ``--model`` directory of a Qwen3.8-27B GGUF whose vocabulary was condensed.

A condensed GGUF (one pruned with bsaleh03/ASCII-Condensed-prune-tools, for example) keeps a
subset of the token rows and numbers them anew: Unsloth's UD-Q3_K_XL condensed to 141,141 rows
moves ``<|endoftext|>`` from 248044 to 140865. ``qwen3_8_27b_gguf`` imports such a GGUF as it
does a standard one and pads its token table and output head with zero rows to whole 128-row tiles
(141,184). The frontend resources must follow the new ids; this tool writes them from the GGUF and
the original checkpoint's files:

- ``tokenizer.json``: the GGUF's tokens at the GGUF's ids and its merges, in the original file's
  structure (pre-tokenizer, decoder, added-token flags). The GGUF's trailing unused entries (the
  ``[PAD248077]`` .. ``[PAD248319]`` rows a llama.cpp export appends) stay outside the tokenizer,
  as they stay outside the original one, so they are never sampled.
- ``tokenizer_config.json`` with its added-token table at the new ids.
- ``config.json`` and ``generation_config.json``: every ``*_token_id`` and ``*_token_ids`` moved
  to the new id of the same token text, and ``vocab_size`` set to the padded row count.
- The chat template, the special-token map and the Vision processor configurations, copied.
- With ``--ranking-out``, the proposal head's token-frequency ranking (``--proposal --ranking``)
  at the new ids.

An id moves to the id of the same token text. Every added token and every configured id must
exist in the GGUF, or the tool stops.

    python -m tools.convert.condense_vocabulary --gguf MODEL.gguf --original QWEN_DIR --out DIR
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .gguf_blocks import padded_rows, tokenizer_ids
from .sources.gguf import GGUFFile

COPIED = (
    "chat_template.jinja",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)
# llama.cpp's token types (gguf TokenType): the padding rows past a tokenizer's vocabulary.
TOKEN_UNUSED = 5


def public_tokens(tokens: Sequence[str], types: Sequence[int] | None) -> int:
    """The GGUF's tokens before its trailing run of unused entries: the tokenizer's domain."""

    count = len(tokens)
    if types is not None:
        if len(types) != count:
            raise ValueError(f"{len(tokens)} tokens but {len(types)} token types")
        while count and types[count - 1] == TOKEN_UNUSED:
            count -= 1
    return count


@dataclass
class Tokenizer:
    """A condensed tokenizer.json and what the condensation changed."""

    data: dict
    dropped_added: list[str] = field(default_factory=list)
    dropped_merges: int = 0


def condense_tokenizer(
    original: dict, tokens: Sequence[str], merges: Sequence[str] | None
) -> Tokenizer:
    """The original tokenizer.json over ``tokens``, token i at id i.

    The added tokens keep their flags and take the ids of their texts; the model vocabulary holds
    the other tokens, as in the original. ``merges`` (the GGUF's, else the original's) keep their
    order; a merge stays when both of its parts and their join are tokens.
    """

    if len(set(tokens)) != len(tokens):
        raise ValueError("the GGUF lists a token text twice")
    ids = {token: index for index, token in enumerate(tokens)}
    out = copy.deepcopy(original)
    added, dropped = [], []
    for entry in original.get("added_tokens", []):
        index = ids.get(entry["content"])
        if index is None:
            dropped.append(entry["content"])
        else:
            added.append({**entry, "id": index})
    added.sort(key=lambda entry: entry["id"])
    added_texts = {entry["content"] for entry in added}
    out["added_tokens"] = added
    out["model"]["vocab"] = {
        token: index for token, index in ids.items() if token not in added_texts
    }

    original_merges = original["model"].get("merges") or []
    as_pairs = bool(original_merges) and isinstance(original_merges[0], list)
    source = list(merges) if merges is not None else list(original_merges)
    kept = []
    for merge in source:
        pair = merge if isinstance(merge, list) else merge.split(" ", 1)
        if len(pair) == 2 and pair[0] in ids and pair[1] in ids and pair[0] + pair[1] in ids:
            kept.append(list(pair) if as_pairs else " ".join(pair))
    out["model"]["merges"] = kept
    return Tokenizer(out, dropped, len(source) - len(kept))


def condense_tokenizer_config(config: dict, tokenizer: dict) -> dict:
    """tokenizer_config.json with its added-token table at the condensed ids."""

    present = tokenizer_ids(tokenizer)
    for key in ("eos_token", "bos_token", "pad_token", "unk_token"):
        token = config.get(key)
        token = token.get("content") if isinstance(token, dict) else token
        if token and token not in present:
            raise ValueError(f"tokenizer_config.json: {key} {token!r} is not a condensed token")
    out = copy.deepcopy(config)
    if "added_tokens_decoder" in config:
        by_text = {entry.get("content"): entry for entry in config["added_tokens_decoder"].values()}
        out["added_tokens_decoder"] = {
            str(entry["id"]): by_text.get(entry["content"])
            or {"content": entry["content"], "special": entry.get("special", False)}
            for entry in tokenizer["added_tokens"]
        }
    return out


def move_token_ids(
    data: Any, original: dict[str, int], condensed: dict[str, int], where: str
) -> list[str]:
    """Move every ``*_token_id`` and ``*_token_ids`` value of a config, at any depth, to the
    condensed id of the same token text, in place. Returns one line per id moved."""

    text_of = {index: text for text, index in original.items()}
    moved: list[str] = []

    def move(value: Any, name: str) -> Any:
        if value is None:
            return None
        if isinstance(value, list):
            return [move(item, name) for item in value]
        if type(value) is not int:
            raise ValueError(f"{name} is {value!r}, not a token id")
        text = text_of.get(value)
        if text is None or text not in condensed:
            raise ValueError(f"{name} {value} ({text!r}) is not a condensed token")
        moved.append(f"{name}: {value} {text} -> {condensed[text]}")
        return condensed[text]

    def walk(node: Any, prefix: str) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            name = f"{prefix}.{key}"
            if key.endswith("_token_id") or key.endswith("_token_ids"):
                node[key] = move(value, name)
            else:
                walk(value, name)

    walk(data, where)
    return moved


def set_vocab_size(config: dict, rows: int) -> None:
    """``vocab_size`` where the config keeps it: the text config of a multimodal checkpoint, the
    top level of a text one."""

    if isinstance(config.get("text_config"), dict):
        config["text_config"]["vocab_size"] = rows
    if "vocab_size" in config or not isinstance(config.get("text_config"), dict):
        config["vocab_size"] = rows


def vocab_size(config: dict) -> int:
    text = config.get("text_config")
    return int(text["vocab_size"] if isinstance(text, dict) else config["vocab_size"])


@dataclass
class Ranking:
    counts: np.ndarray
    missing: int
    kept: float


def condense_ranking(
    counts: np.ndarray, original: dict[str, int], condensed: dict[str, int], rows: int
) -> Ranking:
    """A ranking's counts at the condensed ids, ``rows`` entries long. A condensed token the
    original lacks counts 0 (the proposal forces the special tokens in anyway)."""

    out = np.zeros(rows, dtype="<i8")
    missing = 0
    for text, index in condensed.items():
        if not 0 <= index < rows:
            raise ValueError(f"condensed id {index} of {text!r} is outside {rows} rows")
        source = original.get(text)
        if source is None or not 0 <= source < len(counts):
            missing += 1
            continue
        out[index] = counts[source]
    total = int(counts.sum())
    return Ranking(out, missing, int(out.sum()) / total if total else 0.0)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    from .proposal import DEFAULT_RANKING

    parser = argparse.ArgumentParser(
        prog="python -m tools.convert.condense_vocabulary",
        description="Write the --model directory of a Qwen3.8-27B GGUF with a condensed "
        "vocabulary, for the qwen3_8_27b_gguf recipe.",
    )
    parser.add_argument("--gguf", required=True, type=Path, help="the condensed GGUF")
    parser.add_argument(
        "--original",
        required=True,
        type=Path,
        help="the original checkpoint's directory: its tokenizer and config files (read only)",
    )
    parser.add_argument("--out", required=True, type=Path, help="the directory to write")
    parser.add_argument(
        "--ranking",
        type=Path,
        default=DEFAULT_RANKING,
        help="an int64 token-count ranking of the original vocabulary (default: the converter's)",
    )
    parser.add_argument(
        "--ranking-out", type=Path, help="write the ranking at the condensed ids here"
    )
    args = parser.parse_args(argv)
    if args.out.resolve() == args.original.resolve():
        parser.error("--out must differ from --original")

    with GGUFFile(args.gguf) as gguf:
        tokens = gguf.kv.get("tokenizer.ggml.tokens")
        types = gguf.kv.get("tokenizer.ggml.token_type")
        merges = gguf.kv.get("tokenizer.ggml.merges")
        table = gguf.tensors.get("token_embd.weight")
    if not isinstance(tokens, list) or not tokens:
        parser.error(f"{args.gguf}: the GGUF has no tokenizer.ggml.tokens")
    if table is None or table.shape[0] != len(tokens):
        parser.error(
            f"{args.gguf}: {len(tokens)} tokens but token_embd.weight has "
            f"{None if table is None else table.shape[0]} rows"
        )
    rows = padded_rows(len(tokens))
    count = public_tokens(tokens, types)

    original_tokenizer = read_json(args.original / "tokenizer.json")
    original_ids = tokenizer_ids(original_tokenizer)
    condensed = condense_tokenizer(original_tokenizer, tokens[:count], merges)
    if condensed.dropped_added:
        parser.error(f"the GGUF lacks the added tokens {condensed.dropped_added}")
    condensed_ids = tokenizer_ids(condensed.data)
    same = sum(1 for text, index in condensed_ids.items() if original_ids.get(text) == index)
    foreign = [text for text in condensed_ids if text not in original_ids]
    print(
        f"{len(tokens)} GGUF rows, {rows} with the padding; {count} tokens "
        f"({len(original_ids)} in the original), {len(tokens) - count} trailing unused rows left "
        f"out; {same} tokens keep their id, {len(foreign)} are not in the original"
    )
    if condensed.dropped_merges:
        print(f"{condensed.dropped_merges} merges dropped: a part or their join is not a token")

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "tokenizer.json", condensed.data)
    if (args.original / "tokenizer_config.json").exists():
        config = read_json(args.original / "tokenizer_config.json")
        write_json(
            args.out / "tokenizer_config.json", condense_tokenizer_config(config, condensed.data)
        )
    for name in ("config.json", "generation_config.json"):
        if not (args.original / name).exists():
            continue
        data = read_json(args.original / name)
        for line in move_token_ids(data, original_ids, condensed_ids, name):
            print(line)
        if name == "config.json":
            set_vocab_size(data, rows)
        write_json(args.out / name, data)
    for name in COPIED:
        if (args.original / name).exists():
            shutil.copyfile(args.original / name, args.out / name)

    if args.ranking_out is not None:
        entries = vocab_size(read_json(args.original / "config.json"))
        size = args.ranking.stat().st_size
        if size == 0 or size % (entries * 8):
            parser.error(f"{args.ranking}: {size} bytes are not whole rows of {entries} counts")
        counts = np.fromfile(args.ranking, dtype="<i8", count=entries)
        ranking = condense_ranking(counts, original_ids, condensed_ids, rows)
        ranking.counts.tofile(args.ranking_out)
        print(
            f"ranking: {ranking.missing} tokens without an original count, "
            f"{ranking.kept:.2%} of the counts kept, wrote {args.ranking_out}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
