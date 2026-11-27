#!/usr/bin/env python3
"""Fit the routed-expert route table of src/ops/linear/gguf/moe_route_table.h.

Each input is one ninfer_gguf_moe_matrix_bench run, named with the shape it ran:

    python3 tools/bench/gguf_moe_route_table.py \
        flint.txt:2048:512:6:256 gemma4.txt:2816:704:8:128 ...

(file:hidden:width:top_k:experts). For every format and product (gate and up over `hidden`, down
over `width`) the vector kernel's time over the matrix kernel's at the decode calls of 9 to 64
tokens is fitted as a line in the call's routed rows per expert of the layer (pairs / experts) by
least squares. The output is the table's rows, one per format and reduction length k, sorted.
"""

import re
import sys
from collections import defaultdict

LINE = re.compile(
    r"(\S+)\s+(gate_up|down)\s+T=(\d+)\s+experts\s+\d+: matrix\s+([\d.]+) us.*vector\s+([\d.]+) us"
)


def fit(points):
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    slope = sum((x - mx) * (y - my) for x, y in points) / sxx if sxx else 0.0
    return my - slope * mx, slope


def main(arguments):
    curves = defaultdict(list)
    for argument in arguments:
        path, hidden, width, top_k, experts = argument.rsplit(":", 4)
        hidden, width, top_k, experts = int(hidden), int(width), int(top_k), int(experts)
        with open(path) as f:
            for line in f:
                m = LINE.match(line)
                if not m:
                    continue
                name, product, tokens, matrix, vector = m.groups()
                tokens = int(tokens)
                if tokens <= 8:
                    continue
                k = hidden if product == "gate_up" else width
                curves[(name, k)].append((tokens * top_k / experts, float(vector) / float(matrix)))
    for (name, k), points in sorted(curves.items()):
        intercept, slope = fit(points)
        print(f"    {{GgmlType::{name}, {k}, {intercept:.3f}F, {slope:.3f}F}},")


if __name__ == "__main__":
    main(sys.argv[1:])
