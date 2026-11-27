"""The DFlash and DSpark drafters (src/models/dflash/drafter.h) in the masked-drafter runtime
(src/models/dspine/runtime.h) against an FP64 oracle of the drafter.

Writes random BF16 weights of a small drafter of the Gemma 4 layout (64 query heads over 8 KV heads
of 128, a sliding-window layer and a full-attention layer), a target (BF16 token rows and head),
random features of two sequences and a schedule of runtime calls, runs
ninfer_dflash_runtime_driver from the ninfer_tests bundle over them, once without and once with
the Markov head, and checks:

* every proposal against the drafter's FP64 evaluation over the context its lane holds (the
  sliding-window layer the last 2048 positions, the full-attention layer all of them): each
  draft's capped logit (DFlash) or edge (DSpark, teacher-forced on the device's drafts) within
  0.25 of the oracle's best candidate's, at least 85% of the drafts the oracle's own choice, and
  each agreeing draft's confidence within 0.06 of the oracle's;
* rings that held either layer's positions wrongly would not pass: the oracle with the
  full-attention layer limited to the window, and the oracle with the sliding-window layer seeing
  the full-attention layer's ring, each move the long sequence's best logits by at least 0.25 on
  average, and the device's confidences there lie at least three times farther from each than
  from the right oracle (the query and key norms make the attention sharp enough to depend on
  which positions it sees);
* a lane copied from another, and a lane loaded from another's context-cache image, propose bit
  for bit what that lane proposes.

The long sequence commits 3000 positions in one call, wider than the sliding window's ring, and
both sequences commit a decode step in one row commit. Exit 77 without CUDA or without numpy and
torch.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

MODULES = ("numpy", "torch")
DRIVER = "ninfer_dflash_runtime_driver"
MAGIC, VERSION = 0x534C4644, 1

H, INNER, HEADS, KV, D = 256, 512, 64, 8, 128
LAYER_TYPES = ["sliding_attention", "full_attention"]
VOCAB, TARGET_LAYERS, RANK = 2048, 6, 256
TAPS = [1, 3, 5]
WINDOW, BLOCK, MASK, DOMAIN, SOFTCAP = 2048, 8, 2047, 2040, 30.0
EPS, THETA = 1e-6, 1_000_000.0
FULL = 4096  # dflash::full_window at max_context 3100 and a block of 8

SEQUENCES, KEPT_STATES, MAX_CONTEXT = 2, 1, 3100
DRAFT_TOKENS, DRAFT_ROWS, COMMIT_ROWS = 7, 2, 2
FEATURE_COLUMNS, PIECE_COLUMNS = 3000, 4096
POSITIONS = (41, 3001)  # each sequence's features: a commit, then one row commit

GAP, AGREEMENT, CONFIDENCE, SENSITIVITY, DISCRIMINATION = 0.25, 0.85, 0.06, 0.25, 3.0

torch = None  # imported once main() has found it


def config(markov: bool) -> dict:
    return {
        "architectures": ["Gemma4DFlashDraftModel"],
        "hidden_size": H, "intermediate_size": INNER, "num_hidden_layers": len(LAYER_TYPES),
        "num_attention_heads": HEADS, "num_key_value_heads": KV, "head_dim": D,
        "vocab_size": VOCAB, "num_target_layers": TARGET_LAYERS, "block_size": BLOCK,
        "sliding_window": WINDOW, "layer_types": LAYER_TYPES, "rms_norm_eps": EPS,
        "rope_theta": THETA, "target_layer_ids": TAPS, "mask_token_id": MASK,
        "logit_softcap": SOFTCAP, "markov_rank": RANK if markov else 0,
    }


def bf16(x):
    """FP32 values rounded to BF16, as FP64."""
    return x.float().bfloat16().double()


def weights(markov: bool, generator):
    """Random weights, BF16-represented, in the driver's order."""
    normal = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)
    w = {"table": bf16(normal(VOCAB, H)), "head": bf16(normal(VOCAB, H) * 8 / H**0.5),
         "context_norm": bf16(1 + 0.1 * normal(H)), "final_norm": bf16(1 + 0.1 * normal(H)),
         "fc": bf16(normal(H, len(TAPS) * H) / (len(TAPS) * H) ** 0.5)}
    if markov:
        w["markov_w1"] = bf16(normal(VOCAB, RANK) * 0.35)
        w["markov_w2"] = bf16(normal(VOCAB, RANK) * 0.35)
    w["layers"] = []
    for _ in LAYER_TYPES:
        w["layers"].append({
            "input_norm": bf16(1 + 0.1 * normal(H)), "post_norm": bf16(1 + 0.1 * normal(H)),
            "q_norm": bf16(1.7 + 0.1 * normal(D)), "k_norm": bf16(1.7 + 0.1 * normal(D)),
            "q": bf16(normal(HEADS * D, H) / H**0.5), "k": bf16(normal(KV * D, H) / H**0.5),
            "v": bf16(normal(KV * D, H) / H**0.5), "o": bf16(normal(H, HEADS * D) * 2 / (HEADS * D) ** 0.5),
            "gate": bf16(normal(INNER, H) / H**0.5), "up": bf16(normal(INNER, H) / H**0.5),
            "down": bf16(normal(H, INNER) / INNER**0.5)})
    return w


def words(values) -> bytes:
    return struct.pack(f"<{len(values)}I", *values)


def bf16_bytes(x) -> bytes:
    return x.float().bfloat16().view(torch.int16).numpy().astype("<i2").tobytes()


def schedule(markov: bool, w, features, anchors) -> tuple[bytes, list]:
    """The schedule's bytes and the proposals it makes: (sequence, frontier, anchor, k) rows of
    each draft call."""
    text = json.dumps(config(markov)).encode()
    out = bytearray(words([MAGIC, VERSION, len(text)]))
    out += text + b"\0" * (-len(text) % 4)
    out += words([DOMAIN, SEQUENCES, KEPT_STATES, MAX_CONTEXT, DRAFT_TOKENS, DRAFT_ROWS,
                  COMMIT_ROWS, FEATURE_COLUMNS, PIECE_COLUMNS])
    out += bf16_bytes(w["table"]) + bf16_bytes(w["head"])
    out += bf16_bytes(w["context_norm"]) + bf16_bytes(w["final_norm"]) + bf16_bytes(w["fc"])
    if markov:
        out += bf16_bytes(w["markov_w1"]) + bf16_bytes(w["markov_w2"])
    for layer in w["layers"]:
        for name in ("input_norm", "post_norm", "q_norm", "k_norm", "q", "k", "v", "o", "gate",
                     "up", "down"):
            out += bf16_bytes(layer[name])
    for sequence in features:
        out += words([sequence.shape[0]]) + sequence.float().numpy().astype("<f4").tobytes()
    calls, proposals = [], []

    def store(seq, first, n, column):
        for tap in range(len(TAPS)):
            calls.append([1, seq, first, n, tap, column])

    def draft(rows):
        calls.append([7, len(rows), rows[0][4]] + [v for r in rows for v in (r[0], r[2], r[1])])
        proposals.append([(r[3], r[1], r[2], r[4]) for r in rows])

    lane = lambda seq, state=0: state * SEQUENCES + seq
    store(0, 0, 40, 0)
    calls.append([2, lane(0), 0, 0, 40])
    store(1, 0, 3000, 0)
    calls.append([2, lane(1), 0, 0, 3000])
    # rows: (lane, frontier, anchor, sequence, k)
    draft([(lane(0), 40, anchors[0], 0, 7), (lane(1), 3000, anchors[1], 1, 7)])
    store(0, 40, 1, 0)
    store(1, 3000, 1, 1)
    calls.append([3, 2, 0, lane(0), 40, lane(1), 3000])
    draft([(lane(0), 41, anchors[2], 0, 3)])
    draft([(lane(1), 3001, anchors[3], 1, 7)])
    calls.append([4, lane(1), lane(1, 1)])
    draft([(lane(1, 1), 3001, anchors[3], 1, 7)])
    calls.append([5, lane(1), 3001])
    calls.append([6, lane(0, 1), 3001])
    draft([(lane(0, 1), 3001, anchors[3], 1, 7)])
    out += words([len(calls)])
    for call in calls:
        out += words(call)
    return bytes(out), proposals


def rms(x, weight):
    return x / torch.sqrt((x * x).mean(-1, keepdim=True) + EPS) * weight


def rope(x, positions):
    """Split-half NeoX rotation of x [T, heads, D] at positions [T]."""
    half = D // 2
    inverse = THETA ** (-torch.arange(half, dtype=torch.float64) * 2 / D)
    angle = positions.double()[:, None] * inverse[None, :]
    cos, sin = angle.cos()[:, None, :], angle.sin()[:, None, :]
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)


def logits_of(w, sequence, frontier, anchor, k, full_window=FULL, sliding_window=WINDOW):
    """The drafter's FP64 logits [k, vocab] of the mask columns after `anchor` at `frontier`, over
    the context of the sequence's first `frontier` positions (BF16 features)."""
    feats = bf16(sequence[:frontier]).reshape(frontier, len(TAPS) * H)
    context = rms(feats @ w["fc"].T, w["context_norm"])
    ctx_pos = torch.arange(frontier)
    ids = torch.tensor([anchor] + [MASK] * k)
    pos = torch.arange(frontier, frontier + k + 1)
    x = w["table"][ids].clone()
    for kind, layer in zip(LAYER_TYPES, w["layers"]):
        kc = rope(rms((context @ layer["k"].T).view(frontier, KV, D), layer["k_norm"]), ctx_pos)
        vc = (context @ layer["v"].T).view(frontier, KV, D)
        h = rms(x, layer["input_norm"])
        q = rope(rms((h @ layer["q"].T).view(k + 1, HEADS, D), layer["q_norm"]), pos)
        kb = rope(rms((h @ layer["k"].T).view(k + 1, KV, D), layer["k_norm"]), pos)
        vb = (h @ layer["v"].T).view(k + 1, KV, D)
        keys, values = torch.cat([kc, kb]), torch.cat([vc, vb])
        key_pos = torch.cat([ctx_pos, pos])
        window = sliding_window if kind == "sliding_attention" else full_window
        # The ring holds [frontier - window, frontier); a key is seen within the window's distance.
        held = torch.cat([ctx_pos >= frontier - window, torch.ones(k + 1, dtype=torch.bool)])
        seen = held[None, :] & ((pos[:, None] - key_pos[None, :]).abs() < window)
        group = HEADS // KV
        kh = keys.repeat_interleave(group, dim=1)   # [keys, heads, D]
        vh = values.repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, kh) / D**0.5
        scores = scores.masked_fill(~seen[None], float("-inf"))
        attended = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vh).reshape(k + 1, HEADS * D)
        x = x + attended @ layer["o"].T
        h = rms(x, layer["post_norm"])
        gate, up = h @ layer["gate"].T, h @ layer["up"].T
        x = x + (torch.nn.functional.silu(gate) * up) @ layer["down"].T
    normed = rms(x, w["final_norm"])
    return normed[1:] @ w["head"].T


def capped_logits(w, sequence, frontier, anchor, k, **rings):
    logits = logits_of(w, sequence, frontier, anchor, k, **rings)
    return SOFTCAP * torch.tanh(logits / SOFTCAP)


def laws(markov, w, capped, anchor, got):
    """Per draft, the oracle's 16 candidates and their edges, teacher-forced on the device's
    drafts (a DSpark edge adds the Markov bias of the previous draft)."""
    out, previous = [], anchor
    for i, d in enumerate(got):
        row = capped[i, :DOMAIN]
        top = torch.topk(row, 16).indices
        edges = row[top]
        if markov:
            edges = edges + w["markov_w2"][top] @ w["markov_w1"][previous]
        out.append((top, edges))
        previous = int(d)
    return out


def weight_of(top, edges, d):
    """The device's draft's weight in the softmax of the 16 edges (0 outside them)."""
    where = (top == d).nonzero()
    return torch.softmax(edges, 0)[where[0, 0]].item() if where.numel() else 0.0


def check(markov, w, features, proposals, drafts, confidences, report):
    """Every proposal against the oracle; returns the failures."""
    failures, agree, total = [], 0, 0
    sensitivity, right, wrong = [], [], [[], []]
    rings = ({"full_window": WINDOW}, {"sliding_window": FULL})
    for call, rows in enumerate(proposals):
        for j, (seq, frontier, anchor, k) in enumerate(rows):
            got, conf = drafts[call][j], confidences[call][j]
            capped = capped_logits(w, features[seq], frontier, anchor, k)
            for i, (top, edges) in enumerate(laws(markov, w, capped, anchor, got)):
                d, best = int(got[i]), int(top[edges.argmax()])
                where = (top == d).nonzero()
                if where.numel() == 0:
                    failures.append(f"call {call} row {j} draft {i}: {d} outside the oracle's 16")
                else:
                    gap = (edges.max() - edges[where[0, 0]]).item()
                    if gap > GAP:
                        failures.append(f"call {call} row {j} draft {i}: {d} trails the oracle's "
                                        f"best {best} by {gap:.3f}")
                    if d == best:
                        agree += 1
                        error = abs(float(conf[i]) - weight_of(top, edges, d))
                        if error > CONFIDENCE:
                            failures.append(f"call {call} row {j} draft {i}: confidence "
                                            f"{float(conf[i]):.4f}, oracle "
                                            f"{weight_of(top, edges, d):.4f}")
                total += 1
            if seq != 1 or call not in (0, 2):
                continue
            # The long sequence: how far each ring's wrong extent moves the oracle, and how much
            # worse the device's confidences fit it than the right oracle.
            right += [abs(float(conf[i]) - weight_of(t, e, int(got[i])))
                      for i, (t, e) in enumerate(laws(markov, w, capped, anchor, got))]
            for r, extent in enumerate(rings):
                other = capped_logits(w, features[seq], frontier, anchor, k, **extent)
                if call == 0:
                    sensitivity.append((capped[:, :DOMAIN].max(-1).values -
                                        other[:, :DOMAIN].max(-1).values).abs().mean().item())
                wrong[r] += [abs(float(conf[i]) - weight_of(t, e, int(got[i])))
                             for i, (t, e) in enumerate(laws(markov, w, other, anchor, got))]
    mean = lambda values: sum(values) / len(values)
    report(f"{'dspark' if markov else 'dflash'}: {agree}/{total} drafts the oracle's choice; "
           f"the long sequence's confidences {mean(right):.4f} from the oracle, "
           f"{mean(wrong[0]):.4f} and {mean(wrong[1]):.4f} from the oracles whose full-attention "
           f"layer sees the window and whose sliding-window layer sees the full ring "
           f"(logits moved {sensitivity[0]:.3f} and {sensitivity[1]:.3f})")
    if agree < AGREEMENT * total:
        failures.append(f"only {agree} of {total} drafts are the oracle's choice")
    for layer, moved, errors in zip(("full-attention", "sliding-window"), sensitivity, wrong):
        if moved < SENSITIVITY:
            failures.append(f"the {layer} ring's extent moves the logits by only {moved:.3f}")
        if mean(errors) < DISCRIMINATION * mean(right):
            failures.append(f"the device's confidences fit the oracle with a wrong {layer} ring "
                            f"almost as well as the right one")
    # Copied and loaded lanes propose bit for bit what their source lane proposes.
    for call in (3, 4):
        if drafts[call] != drafts[2] or confidences[call] != confidences[2]:
            failures.append(f"call {call} differs from the lane it copies")
    return failures


def main() -> int:
    global torch
    if len(sys.argv) != 2:
        print("usage: runtime_check.py NINFER_TESTS", file=sys.stderr)
        return 2
    missing = [m for m in MODULES if importlib.util.find_spec(m) is None]
    if missing:
        print(f"SKIP: missing {', '.join(missing)}")
        return 77
    import torch as _torch
    torch = _torch
    bundle = sys.argv[1]
    probe = subprocess.run([bundle, DRIVER, "--check-cuda"])
    if probe.returncode == 77:
        print("SKIP: no CUDA device")
        return 77
    if probe.returncode != 0:
        return 1
    generator = torch.Generator().manual_seed(20261127)
    features = [torch.randn(n, len(TAPS), H, generator=generator, dtype=torch.float64)
                for n in POSITIONS]
    anchors = [int(a) for a in torch.randint(0, DOMAIN, (4,), generator=generator)]
    failures = []
    with tempfile.TemporaryDirectory() as root:
        for markov in (False, True):
            w = weights(markov, generator)
            data, proposals = schedule(markov, w, features, anchors)
            path, output = Path(root) / "schedule.bin", Path(root) / "output.bin"
            path.write_bytes(data)
            run = subprocess.run([bundle, DRIVER, str(path), str(output)])
            if run.returncode != 0:
                print(f"FAIL: the driver exited {run.returncode}")
                return 1
            raw, at = output.read_bytes(), 0
            drafts, confidences = [], []
            for rows in proposals:
                cells = sum(r[3] for r in rows)
                values = struct.unpack_from(f"<{cells}i", raw, at)
                at += 4 * cells
                scores = struct.unpack_from(f"<{cells}f", raw, at)
                at += 4 * cells
                drafts.append([list(values[sum(r[3] for r in rows[:j]):][:rows[j][3]])
                               for j in range(len(rows))])
                confidences.append([list(scores[sum(r[3] for r in rows[:j]):][:rows[j][3]])
                                    for j in range(len(rows))])
            failures += [("dspark: " if markov else "dflash: ") + f for f in
                         check(markov, w, features, proposals, drafts, confidences, print)]
    for failure in failures:
        print("FAIL:", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
