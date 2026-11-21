"""FP64 reference of Inkling 2's vision tower (the ViT, the 2 x 2 merge and the projector into the
text model) over a llama.cpp `clip` GGUF with projector inkling2v, and of its image preprocessing.

An independent transcription of llama.cpp's inkling2v graph (tools/mtmd/models/inkling2v.cpp and
the mtmd_graph helpers it calls) for golden values. The patch projection has no bias; the learned
position rows of a G x G grid are added to it, resized to the image's W x H patches when the sizes
differ (ggml_interpolate's bilinear mode: half-pixel centers, the lower neighbour clamped before
its weight is taken, the weight clamped to [0, 1]); a class row (the class embedding plus its
position row) goes first. LayerNorm (biased variance, with bias) gives the residual. Per block:
x += ls1 * (out(attention(LN1(x))) + b_out) and x += ls2 * (down(gelu(up(LN2(x)) + b_up)) +
b_down), the attention's Q, K and V with biases, each head of D values turned by a 2D RoPE (the
adjacent pairs of its first half by the row's column position, those of its second half by its row
position, pair i at theta^(-2i / (D / 2)); the class row at (0, 0), patch (x, y) at (x + 1, y + 1)),
unmasked softmax attention with scale 1 / sqrt(D) over every row. The class row is dropped, each 2 x
2 cell of patches becomes one row of 4 hidden values (value dy * 2 hidden + dx * hidden + c), cells
row by row, and the projector (mm.1, the erf GELU, mm.2, no biases) takes it to the text width.
Every value is FP64 over the GGUF's BF16/F32 values and its exactly decoded blocks (a Q8_0
mmproj).

    python -m tools.reference.inkling2_vision --mmproj MMPROJ.gguf \\
        --patches patches.bin --grid W H --out rows.bin

`--patches` holds the engine's input, BF16 words [H * W, 3 * patch * patch] (a .npy array, or the
raw words of a .bin file), each patch's values (channel, y, x) with x fastest and the patches row by
row; `--out` receives the projector rows [(H / 2) * (W / 2), text hidden] in FP64 (.npy, or raw
for .bin).

`--image FILE` instead derives the patches from an image as llama.cpp's inkling2v preprocessor
does (needs Pillow): the image padded on its right and bottom to whole 2 x 2 cells of patches with
the raw value -1, then (x - 127.5) / 127.5 in FP32, rounded to BF16 (an image past
`--max-tokens` cells is first resized, Pillow's bicubic, to the aligned size llama.cpp's
calc_size_preserved_ratio gives). `--patches-out` saves those patches; their SHA-256 is printed.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np

from tools.convert.gguf_blocks import dequantize_blocks
from tools.convert.sources.gguf import TYPE_BF16, TYPE_F16, TYPE_F32, GGUFFile


def bf16_words(words: np.ndarray) -> np.ndarray:
    return (words.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def to_bf16_words(values: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even BF16 words of FP32 values."""
    bits = values.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def gelu_erf(x: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def shape_of(mmproj: GGUFFile) -> dict:
    kv, v = mmproj.kv, "clip.vision."
    return {"layers": int(kv[v + "block_count"]), "hidden": int(kv[v + "embedding_length"]),
            "heads": int(kv[v + "attention.head_count"]), "patch": int(kv[v + "patch_size"]),
            "merge": int(kv[v + "spatial_merge_size"]),
            "eps": float(kv[v + "attention.layer_norm_epsilon"]), "theta": 10000.0}


def resize_positions(positions: np.ndarray, grid: int, w: int, h: int) -> np.ndarray:
    """Rows [h * w, hidden] of the position rows [grid * grid, hidden] (row y * grid + x)."""

    table = positions.reshape(grid, grid, -1)
    if w == grid and h == grid:
        return table.reshape(grid * grid, -1)
    # ggml_interpolate's arithmetic is FP32; its coordinates are reproduced in FP32 here.
    def axis(n_out: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sf = np.float32(n_out) / np.float32(grid)
        f = (np.arange(n_out, dtype=np.float32) + np.float32(0.5)) / sf - np.float32(0.5)
        lo = np.floor(f).astype(np.int64)
        hi = np.clip(lo + 1, 0, grid - 1)
        lo = np.clip(lo, 0, grid - 1)
        d = np.clip(f - lo.astype(np.float32), np.float32(0), np.float32(1)).astype(np.float64)
        return lo, hi, d

    x0, x1, dx = axis(w)
    y0, y1, dy = axis(h)
    dx, dy = dx[None, :, None], dy[:, None, None]
    a = table[y0][:, x0]
    b = table[y0][:, x1]
    c = table[y1][:, x0]
    d = table[y1][:, x1]
    out = a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + c * (1 - dx) * dy + d * dx * dy
    return out.reshape(h * w, -1)


def tower(mmproj: GGUFFile, patches: np.ndarray, grid_w: int, grid_h: int) -> np.ndarray:
    """Projector rows [(grid_h / 2) * (grid_w / 2), text hidden] of the patches
    [grid_h * grid_w, 3 * patch * patch]."""

    s = shape_of(mmproj)
    h, heads, eps, m = s["hidden"], s["heads"], s["eps"], s["merge"]
    d = h // heads
    n = grid_w * grid_h
    if patches.shape[0] != n or grid_w % m or grid_h % m:
        raise ValueError("the patches do not fill a grid of whole merge cells")
    def read(name: str) -> np.ndarray:
        info = mmproj.info(name)
        if info.type_id in (TYPE_F32, TYPE_F16, TYPE_BF16):
            return mmproj.read_direct(name).astype(np.float64)
        blocks = mmproj.read_blocks(name, 0, None)
        return dequantize_blocks(info.type_id, blocks).numpy().astype(np.float64)

    positions = read("v.position_embd.weight")
    grid = int(round(math.sqrt(positions.shape[0])))
    x = patches @ read("v.patch_embd.weight").reshape(h, -1).T
    x = x + resize_positions(positions, grid, grid_w, grid_h)
    cls = read("v.class_embd").reshape(1, h) + read("v.class_position_embd.weight").reshape(1, h)
    x = np.concatenate([cls, x], axis=0)
    x = layer_norm(x, read("v.pre_ln.weight"), read("v.pre_ln.bias"), eps)

    # RoPE angles [n + 1, D/2]: D/4 pair frequencies by the column position, then by the row.
    quarter = d // 4
    freq = 1.0 / s["theta"] ** (2.0 * np.arange(quarter, dtype=np.float64) / (d // 2))
    px = np.concatenate([[0], np.arange(n) % grid_w + 1]).astype(np.float64)
    py = np.concatenate([[0], np.arange(n) // grid_w + 1]).astype(np.float64)
    angle = np.concatenate([px[:, None] * freq, py[:, None] * freq], axis=1)
    cos, sin = np.cos(angle)[:, None, :], np.sin(angle)[:, None, :]

    def rope(t: np.ndarray) -> np.ndarray:  # [rows, heads, d], pairs (2i, 2i + 1)
        a, b = t[..., 0::2], t[..., 1::2]
        out = np.empty_like(t)
        out[..., 0::2] = a * cos - b * sin
        out[..., 1::2] = a * sin + b * cos
        return out

    rows = n + 1
    for i in range(s["layers"]):
        g = f"v.blk.{i}."
        normed = layer_norm(x, read(g + "ln1.weight"), read(g + "ln1.bias"), eps)
        q, k, v = (normed @ read(g + f"attn_{t}.weight").T + read(g + f"attn_{t}.bias")
                   for t in ("q", "k", "v"))
        q, k, v = (t.reshape(rows, heads, d) for t in (q, k, v))
        q, k = rope(q), rope(k)
        qh, kh, vh = (t.transpose(1, 0, 2) for t in (q, k, v))  # [heads, rows, d]
        scores = qh @ kh.transpose(0, 2, 1) / math.sqrt(d)
        scores -= scores.max(axis=-1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(axis=-1, keepdims=True)
        o = (p @ vh).transpose(1, 0, 2).reshape(rows, h)
        x = x + (o @ read(g + "attn_out.weight").T + read(g + "attn_out.bias")) * read(g + "ls1.weight")
        normed = layer_norm(x, read(g + "ln2.weight"), read(g + "ln2.bias"), eps)
        up = gelu_erf(normed @ read(g + "ffn_up.weight").T + read(g + "ffn_up.bias"))
        x = x + (up @ read(g + "ffn_down.weight").T + read(g + "ffn_down.bias")) * read(g + "ls2.weight")

    x = x[1:].reshape(grid_h // m, m, grid_w // m, m, h)  # [Y, dy, X, dx, c]
    cells = x.transpose(0, 2, 1, 3, 4).reshape(-1, m * m * h)
    return gelu_erf(cells @ read("mm.1.weight").T) @ read("mm.2.weight").T


def plan(width: int, height: int, patch: int, merge: int, budget: int) -> tuple[int, int]:
    """The pixel size an image is resized to: itself within the budget, else llama.cpp's
    calc_size_preserved_ratio (FP32) at whole cells."""

    align = patch * merge
    cells = math.ceil(width / align) * math.ceil(height / align)
    if cells <= budget:
        return width, height
    f32 = np.float32
    max_pixels = budget * align * align
    # std::round: halves away from zero.
    w = max(align, int(math.floor(float(f32(width) / f32(align)) + 0.5)) * align)
    h = max(align, int(math.floor(float(f32(height) / f32(align)) + 0.5)) * align)
    if w * h > max_pixels:
        beta = np.sqrt(f32(height) * f32(width) / f32(max_pixels))
        h = max(align, int(np.floor(f32(height) / beta / f32(align))) * align)
        w = max(align, int(np.floor(f32(width) / beta / f32(align))) * align)
    return w, h


def image_patches(path: Path, patch: int, merge: int, budget: int) -> tuple[np.ndarray, int, int]:
    """BF16 words [grid_h * grid_w, 3 * p * p] of an image and its grid (W, H)."""

    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    w, h = plan(image.width, image.height, patch, merge, budget)
    if (w, h) != (image.width, image.height):
        image = image.resize((w, h), Image.Resampling.BICUBIC)
    align = patch * merge
    pw, ph = -(-w // align) * align, -(-h // align) * align
    raw = np.full((ph, pw, 3), -1.0, dtype=np.float32)
    raw[:h, :w] = np.asarray(image, dtype=np.float32)
    x = ((raw - np.float32(127.5)) / np.float32(127.5)).transpose(2, 0, 1)
    gh, gw = ph // patch, pw // patch
    x = x.reshape(3, gh, patch, gw, patch).transpose(1, 3, 0, 2, 4).reshape(gh * gw, -1)
    return to_bf16_words(x), gw, gh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mmproj", type=Path, default=None)
    parser.add_argument("--patches", type=Path, default=None)
    parser.add_argument("--grid", type=int, nargs=2, metavar=("W", "H"))
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--patches-out", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.image is not None:
        words, w, h = image_patches(args.image, 20, 2, args.max_tokens)
        digest = hashlib.sha256(words.astype("<u2").tobytes()).hexdigest()
        print(f"{args.image}: grid {w} x {h} patches, patches sha256 {digest}")
        if args.patches_out is not None:
            words.astype("<u2").tofile(args.patches_out) if args.patches_out.suffix == ".bin" \
                else np.save(args.patches_out, words)
    if args.mmproj is None:
        return
    with GGUFFile(args.mmproj) as mmproj:
        features = 3 * shape_of(mmproj)["patch"] ** 2
        if args.image is None:
            if args.patches is None or args.grid is None:
                parser.error("give --patches and --grid, or --image")
            words = (np.fromfile(args.patches, dtype="<u2").reshape(-1, features)
                     if args.patches.suffix == ".bin" else np.load(args.patches))
            w, h = args.grid
        rows = tower(mmproj, bf16_words(words), w, h)
        print(f"projector rows {rows.shape}, mean |v| {np.abs(rows).mean():.6g}")
        if args.out is not None:
            # A .bin file receives the raw little-endian FP64 values (the C++ test reads them).
            rows.astype("<f8").tofile(args.out) if args.out.suffix == ".bin" else np.save(args.out, rows)


if __name__ == "__main__":
    main()
