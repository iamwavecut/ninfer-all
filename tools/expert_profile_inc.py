#!/usr/bin/env python3
"""Writes src/models/expert_profiles/builtin.inc from routing profiles recorded with
--expert-profile-out.

Each --profile FAMILY=PATH names a model family ("qwen3.8-flash-next", "gemma4", "qwen3.5-moe",
"flint", "deepseek-v4.1") and a recorded profile of one of its releases. The built-in profile keeps
the counts and the layers' bank widths, not the artifact identity: it serves every artifact of that
family and geometry (models/expert_profile.h), so a family has one profile per geometry
(DeepSeek-V4.1: the release's 384 experts a block and the Coder build's 192). The output holds only
the profiles named, unless --keep reads the profiles it already holds: then a --profile of the same
family and bank widths replaces its entry and any other is added. Counts above 2^32 - 1 are scaled
down together, which keeps their order.
"""

import argparse
import json
import pathlib
import re
import sys

HEADER = """// The built-in routing profiles (models/expert_profile.cpp includes this file). Each was recorded
// with --expert-profile-out and converted by tools/expert_profile_inc.py.
"""


def load(path: pathlib.Path) -> list[list[int]]:
    data = json.loads(path.read_text())
    if data.get("version") != 1 or not isinstance(data.get("counts"), list):
        raise ValueError(f"{path}: not a version 1 expert profile")
    counts = data["counts"]
    for layer in counts:
        if not layer or any(not isinstance(v, int) or v < 0 for v in layer):
            raise ValueError(f"{path}: counts must be nonnegative integers")
        if len(layer) > 0xFFFF:
            raise ValueError(f"{path}: a bank wider than 65535 experts")
    peak = max(max(layer) for layer in counts)
    if peak > 0xFFFFFFFF:
        scale = 0xFFFFFFFF / peak
        counts = [[int(v * scale) for v in layer] for layer in counts]
    return counts


def lines(values: list[int], indent: str = "    ", width: int = 100) -> str:
    out, line = [], indent
    for v in values:
        item = f"{v}, "
        if len(line) + len(item) > width:
            out.append(line.rstrip())
            line = indent
        line += item
    if line.strip():
        out.append(line.rstrip())
    return "\n".join(out)


def written(path: pathlib.Path) -> list[tuple[str, list[list[int]]]]:
    """The profiles a builtin.inc this tool wrote holds, in its order."""
    text = path.read_text()
    declared = re.findall(r"constexpr std::uint(?:16|32)_t k(Widths|Counts)(\d+)\[\] = \{(.*?)\};",
                          text, re.S)
    arrays = {(kind, int(index)): [int(v) for v in re.findall(r"\d+", body)]
              for kind, index, body in declared}
    profiles = []
    for family, widths, counts in re.findall(r'\{"([^"]+)", kWidths(\d+), kCounts(\d+)\}', text):
        flat, layers, at = arrays[("Counts", int(counts))], [], 0
        for width in arrays[("Widths", int(widths))]:
            layers.append(flat[at:at + width])
            at += width
        profiles.append((family, layers))
    return profiles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", default=[], metavar="FAMILY=PATH")
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    parser.add_argument("--keep", action="store_true",
                        help="keep the profiles the output holds; a --profile of the same family "
                             "and bank widths replaces one")
    args = parser.parse_args()
    profiles = written(args.output) if args.keep and args.output.exists() else []
    for spec in args.profile:
        family, sep, path = spec.partition("=")
        if not sep or not family or not path:
            parser.error(f"--profile {spec!r}: expected FAMILY=PATH")
        counts = load(pathlib.Path(path))
        widths = [len(layer) for layer in counts]
        same = [i for i, (f, c) in enumerate(profiles)
                if f == family and [len(layer) for layer in c] == widths]
        if same:
            profiles[same[0]] = (family, counts)
        else:
            profiles.append((family, counts))
    text = [HEADER]
    entries = []
    for index, (family, counts) in enumerate(profiles):
        widths = [len(layer) for layer in counts]
        flat = [v for layer in counts for v in layer]
        text.append(f"constexpr std::uint16_t kWidths{index}[] = {{\n{lines(widths)}\n}};")
        text.append(f"constexpr std::uint32_t kCounts{index}[] = {{\n{lines(flat)}\n}};\n")
        entries.append(f'    {{"{family}", kWidths{index}, kCounts{index}}},')
    if not entries:
        entries.append('    {"", {}, {}},')
    text.append("constexpr Builtin kBuiltinProfiles[] = {\n" + "\n".join(entries) + "\n};\n")
    args.output.write_text("\n".join(text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
