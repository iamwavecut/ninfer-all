from __future__ import annotations

import json

import numpy as np
import pytest

from tools.convert import condense_vocabulary
from tools.convert.gguf_blocks import padded_rows
from tools.convert.resources import token_domain
from tools.convert.sources.gguf import write_gguf

TYPE_Q8_0 = 8
NORMAL, CONTROL, UNUSED = 1, 3, 5
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|image_pad|>"]
# A byte-level BPE in Qwen's layout: base characters, merged tokens, then the added tokens, and
# rows past the tokenizer up to the checkpoint's vocab_size.
ORIGINAL = ["a", "b", "c", "d", "e", "Ġ", "ab", "abc", "de", "Ġa", "Ġab"] + SPECIALS
MERGES = ["a b", "ab c", "d e", "Ġ a", "Ġa b"]
VOCAB_SIZE = 20


def _original(directory):
    vocab = {token: index for index, token in enumerate(ORIGINAL) if token not in SPECIALS}
    added = [
        {"id": ORIGINAL.index(token), "content": token, "single_word": False, "lstrip": False,
         "rstrip": False, "normalized": False, "special": True}
        for token in SPECIALS
    ]
    byte_level = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False,
                  "use_regex": True}
    tokenizer = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": None, "pre_tokenizer": byte_level, "post_processor": None,
        "decoder": byte_level,
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": None, "end_of_word_suffix": None,
                  "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": MERGES},
    }
    config = {
        "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "bos_token": None,
        "added_tokens_decoder": {str(entry["id"]): {k: v for k, v in entry.items() if k != "id"}
                                 for entry in added},
    }
    ids = {token: index for index, token in enumerate(ORIGINAL)}
    model = {
        "image_token_id": ids["<|image_pad|>"],
        "text_config": {"vocab_size": VOCAB_SIZE, "bos_token_id": ids["<|endoftext|>"],
                        "eos_token_id": ids["<|endoftext|>"], "pad_token_id": None},
    }
    generation = {"eos_token_id": [ids["<|im_end|>"], ids["<|endoftext|>"]],
                  "pad_token_id": ids["<|endoftext|>"], "temperature": 1.0}
    directory.mkdir()
    for name, data in (("tokenizer.json", tokenizer), ("tokenizer_config.json", config),
                       ("config.json", model), ("generation_config.json", generation)):
        (directory / name).write_text(json.dumps(data), encoding="utf-8")
    (directory / "chat_template.jinja").write_text("{{ messages }}\n", encoding="utf-8")
    (directory / "preprocessor_config.json").write_text('{"patch_size": 16}', encoding="utf-8")


def _condensed_gguf(path, tokens, types, merges):
    rows = len(tokens)
    write_gguf(
        path,
        {"general.architecture": "qwen35", "tokenizer.ggml.model": "gpt2",
         "tokenizer.ggml.tokens": tokens, "tokenizer.ggml.token_type": types,
         "tokenizer.ggml.merges": merges},
        [("token_embd.weight", (rows, 32), TYPE_Q8_0, bytes(rows * 34))],
    )


def _condense(tmp_path, ranking=None):
    """Prune "de" as the prune tools do: kept tokens keep their order, the specials and the
    exporter's trailing [PAD] rows follow. The GGUF still lists the merge "d e"."""

    _original(tmp_path / "original")
    kept = [token for token in ORIGINAL if token != "de"]
    pads = [f"[PAD{index}]" for index in range(len(ORIGINAL), VOCAB_SIZE)]
    types = [CONTROL if t in SPECIALS else NORMAL for t in kept] + [UNUSED] * len(pads)
    _condensed_gguf(tmp_path / "condensed.gguf", kept + pads, types, MERGES)
    argv = ["--gguf", str(tmp_path / "condensed.gguf"), "--original",
            str(tmp_path / "original"), "--out", str(tmp_path / "out")]
    if ranking is not None:
        argv += ["--ranking", str(ranking), "--ranking-out", str(tmp_path / "ranking.i64")]
    assert condense_vocabulary.main(argv) == 0
    read = lambda name: json.loads((tmp_path / "out" / name).read_text(encoding="utf-8"))
    return kept, read


def test_the_resources_follow_the_gguf_ids(tmp_path, capsys):
    counts = np.arange(100, 100 + VOCAB_SIZE, dtype="<i8")
    counts.tofile(tmp_path / "counts.i64")
    kept, read = _condense(tmp_path, tmp_path / "counts.i64")
    new = {token: index for index, token in enumerate(kept)}
    rows = padded_rows(len(kept) + VOCAB_SIZE - len(ORIGINAL))

    tokenizer = read("tokenizer.json")
    assert tokenizer["model"]["vocab"] == {t: new[t] for t in kept if t not in SPECIALS}
    assert [(e["id"], e["content"], e["special"]) for e in tokenizer["added_tokens"]] == [
        (new[t], t, True) for t in SPECIALS
    ]
    # "d e" joins to a pruned token; the others keep their order.
    assert tokenizer["model"]["merges"] == ["a b", "ab c", "Ġ a", "Ġa b"]
    assert tokenizer["pre_tokenizer"]["type"] == "ByteLevel"
    config = read("tokenizer_config.json")
    assert {int(k): v["content"] for k, v in config["added_tokens_decoder"].items()} == {
        new[t]: t for t in SPECIALS
    }
    assert config["eos_token"] == "<|im_end|>"
    # The trailing unused rows stay outside the tokenizer's domain, as in the original.
    assert token_domain(tokenizer, config, rows) == (
        len(kept), tuple(new[t] for t in SPECIALS)
    )

    model = read("config.json")
    assert model["image_token_id"] == new["<|image_pad|>"]
    assert model["text_config"] == {"vocab_size": rows, "bos_token_id": new["<|endoftext|>"],
                                    "eos_token_id": new["<|endoftext|>"], "pad_token_id": None}
    assert "vocab_size" not in model
    generation = read("generation_config.json")
    assert generation == {"eos_token_id": [new["<|im_end|>"], new["<|endoftext|>"]],
                          "pad_token_id": new["<|endoftext|>"], "temperature": 1.0}
    for name in ("chat_template.jinja", "preprocessor_config.json"):
        copied = (tmp_path / "out" / name).read_bytes()
        assert copied == (tmp_path / "original" / name).read_bytes()

    ranking = np.fromfile(tmp_path / "ranking.i64", dtype="<i8")
    assert ranking.size == rows
    assert [int(ranking[new[t]]) for t in kept] == [100 + ORIGINAL.index(t) for t in kept]
    assert not ranking[len(kept):].any()
    report = capsys.readouterr().out
    assert f"{VOCAB_SIZE - len(ORIGINAL)} trailing unused rows left out" in report
    assert "1 merges dropped" in report


def test_text_tokenizes_to_the_same_tokens(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    _condense(tmp_path)
    original = tokenizers.Tokenizer.from_file(str(tmp_path / "original" / "tokenizer.json"))
    condensed = tokenizers.Tokenizer.from_file(str(tmp_path / "out" / "tokenizer.json"))
    for text in ("abc ab ab", "<|im_start|>abc<|im_end|>", "cab a"):
        assert condensed.encode(text).tokens == original.encode(text).tokens
        assert condensed.decode(condensed.encode(text).ids, skip_special_tokens=False) == text
    # A pruned token's text falls back to the tokens it was merged from.
    assert original.encode("de").tokens == ["de"]
    assert condensed.encode("de").tokens == ["d", "e"]


def test_a_missing_special_token_or_configured_id_stops_the_tool(tmp_path):
    _original(tmp_path / "original")
    kept = [token for token in ORIGINAL if token != "<|image_pad|>"]
    types = [CONTROL if t in SPECIALS else NORMAL for t in kept]
    _condensed_gguf(tmp_path / "condensed.gguf", kept, types, MERGES)
    argv = ["--gguf", str(tmp_path / "condensed.gguf"), "--original",
            str(tmp_path / "original"), "--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit):
        condense_vocabulary.main(argv)
    assert not (tmp_path / "out").exists()

    original = {"<|im_end|>": 13, "x": 3}
    with pytest.raises(ValueError, match=r"eos_token_id 3 \('x'\) is not a condensed token"):
        condense_vocabulary.move_token_ids({"eos_token_id": 3}, original, {"<|im_end|>": 7}, "c")
    with pytest.raises(ValueError, match="pad_token '<|endoftext|>' is not a condensed token"):
        condense_vocabulary.condense_tokenizer_config(
            {"pad_token": "<|endoftext|>"}, {"model": {"vocab": {"a": 0}}, "added_tokens": []}
        )


def test_only_the_trailing_unused_rows_leave_the_domain():
    tokens = ["a", "[PAD1]", "b", "[PAD3]", "[PAD4]"]
    assert condense_vocabulary.public_tokens(tokens, [1, 5, 1, 5, 5]) == 3
    assert condense_vocabulary.public_tokens(tokens, None) == 5
    with pytest.raises(ValueError, match="5 tokens but 4 token types"):
        condense_vocabulary.public_tokens(tokens, [1, 1, 1, 1])
