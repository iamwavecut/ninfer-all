"""The DSpine runtime (src/models/dspine/runtime.h) against an FP64 oracle of the drafter.

Converts a small random DSpine checkpoint (tools/convert, recipe dspine_bf16), writes a target
(BF16 token rows and head), random features of three sequences and a schedule of runtime calls,
runs ninfer_dspine_runtime_driver from the ninfer_tests bundle over them and checks:

* every proposal against the drafter's FP64 evaluation over the context its lane holds,
  teacher-forced on the device's drafts: each draft's edge within 0.25 logits of the oracle's best
  candidate's, at least 85% of the drafts the oracle's own choice, each confidence within 0.06 of
  the oracle's and within 0.01 on average. The device's drafter keeps its states in BF16, so a
  candidate within those 0.25 logits of the oracle's 16th may take another's place in its 16: a
  confidence is compared with the range the oracle's takes over every such candidate set;
* a lane loaded from a context-cache image proposes bit for bit what the lane it was saved from
  proposes;
* a lane that committed positions past a frontier proposes there bit for bit what a copy taken at
  that frontier proposes (a family's rewind leaves the later slots in place);
* the oracle's choices depend on the context, so a ring the runtime fills wrongly cannot pass.

The schedule takes FP32 and BF16 taps, a commit from a column offset, a prompt pass wider than the
ring (its last window commits, then decode steps wrap the ring), row commits of two sequences, lane
copies, and an image saved from one lane and loaded over another lane's ring. Exit 77 without CUDA
or without numpy, safetensors and torch.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

MODULES = ("numpy", "safetensors", "torch")
DRIVER = "ninfer_dspine_runtime_driver"
MAGIC, VERSION = 0x4E505344, 1

H, INNER, HEADS, KV, D, LAYERS = 256, 512, 32, 8, 128, 2
VOCAB, CODE, RANK, TARGET_LAYERS = 2048, 128, 128, 6
TAPS = [1, 3, 5]
WINDOW, BLOCK, MASK, DOMAIN = 2048, 16, 2047, 2040
EPS, BRIDGE_EPS, THETA = 1e-6, 1e-6, 10_000_000
FLT_MIN = 1.1754943508222875e-38

SEQUENCES, KEPT_STATES, MAX_CONTEXT = 3, 1, 2200
DRAFT_TOKENS, DRAFT_ROWS, COMMIT_ROWS = 15, 4, 2
FEATURE_COLUMNS, PIECE_COLUMNS = 2100, 256
POSITIONS = (43, 8, 2100)  # each sequence's features

GAP, AGREEMENT, CONFIDENCE, MEAN_CONFIDENCE, SENSITIVITY = 0.25, 0.85, 0.06, 0.01, 0.25

torch = None  # imported once main() has found it


def config() -> dict:
    return {
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": False,
        "hidden_size": H,
        "intermediate_size": INNER,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV,
        "head_dim": D,
        "num_hidden_layers": LAYERS,
        "max_position_embeddings": 4096,
        "vocab_size": VOCAB,
        "draft_vocab_size": VOCAB,
        "num_target_layers": TARGET_LAYERS,
        "block_size": BLOCK,
        "rms_norm_eps": EPS,
        "rope_parameters": {"rope_theta": THETA, "rope_type": "default"},
        "rope_scaling": None,
        "layer_types": ["sliding_attention"] * LAYERS,
        "sliding_window": WINDOW,
        "code_bridge_second_pass": {"mode": "serial_table", "first_exact": LAYERS - 1,
                                    "head_topk": 16},
        "dflash_config": {
            "draft_block_causal": True,
            "mask_token_id": MASK,
            "selected_prefix_code_bridge": {"code_size": CODE, "eps": BRIDGE_EPS, "kappa": 10.0,
                                            "rank": RANK},
            "selected_prefix_decode_mode": "serial",
            "selected_prefix_feature_layers": list(range(1, LAYERS + 1)),
            "selected_prefix_feature_write_mode": "independent",
            "selected_prefix_feature_writes": True,
            "selected_prefix_last_write_refine": {"head_topk": 16, "kind": "exact_sender",
                                                  "version": 1, "weight": 1.0},
            "selected_prefix_terminal": False,
            "target_layer_ids": TAPS,
        },
    }


def checkpoint_tensors(generator) -> dict:
    """Weights scaled so that attention over the context, the MLP and the bridge's write each move
    the residual by about its own size, and the head's logits spread over a few units."""

    def normal(*shape, scale=1.0, mean=0.0):
        return (mean + scale * torch.randn(*shape, generator=generator)).to(torch.bfloat16)

    def gain(n, mean=1.0):
        return normal(n, scale=0.1, mean=mean)

    tensors = {
        "fc.weight": normal(H, len(TAPS) * H, scale=1 / math.sqrt(len(TAPS) * H)),
        "hidden_norm.weight": gain(H),
        "norm.weight": gain(H),
        "draft_block_causal_version": torch.tensor(1, dtype=torch.int64),
        "last_write_refine_version": torch.tensor(1, dtype=torch.int64),
    }
    for i in range(LAYERS):
        p = f"layers.{i}."
        tensors |= {
            p + "input_layernorm.weight": gain(H),
            p + "post_attention_layernorm.weight": gain(H),
            p + "self_attn.q_proj.weight": normal(HEADS * D, H, scale=1 / math.sqrt(H)),
            p + "self_attn.k_proj.weight": normal(KV * D, H, scale=1 / math.sqrt(H)),
            p + "self_attn.v_proj.weight": normal(KV * D, H, scale=1 / math.sqrt(H)),
            p + "self_attn.o_proj.weight": normal(H, HEADS * D, scale=3 / math.sqrt(HEADS * D)),
            p + "self_attn.q_norm.weight": gain(D, 1.5),
            p + "self_attn.k_norm.weight": gain(D, 1.5),
            p + "mlp.gate_proj.weight": normal(INNER, H, scale=1 / math.sqrt(H)),
            p + "mlp.up_proj.weight": normal(INNER, H, scale=1 / math.sqrt(H)),
            p + "mlp.down_proj.weight": normal(H, INNER, scale=1 / math.sqrt(INNER)),
        }
    codebook = torch.randn(VOCAB, CODE, generator=generator)
    b = "selected_prefix_bridge."
    tensors |= {
        b + "codebook": (codebook / codebook.norm(dim=-1, keepdim=True)).to(torch.bfloat16),
        b + "early_read_weights": normal(LAYERS - 1, CODE, H, scale=1 / math.sqrt(H)),
        b + "read_weight": normal(CODE, H, scale=1 / math.sqrt(H)),
        b + "feature_write_weights": normal(LAYERS, H, RANK, scale=1 / math.sqrt(RANK)),
        b + "gate.weight": normal(RANK, CODE, scale=1 / math.sqrt(CODE)),
        b + "gate.bias": normal(RANK, scale=0.1),
        b + "predecessor_gate.weight": normal(RANK, CODE, scale=1 / math.sqrt(CODE)),
        b + "prev.weight": normal(RANK, CODE, scale=1 / math.sqrt(CODE)),
        b + "out.weight": torch.zeros(CODE, RANK, dtype=torch.bfloat16),
        b + "independent_write_version": torch.tensor(1, dtype=torch.int64),
        b + "terminal_disabled_version": torch.tensor(1, dtype=torch.int64),
    }
    return tensors


def lane(sequence: int, state: int = 0) -> int:
    return state * SEQUENCES + sequence


class Schedule:
    """The driver's calls, and what each lane's ring slots hold after them: (position, sequence,
    feature position) of the features its K/V came from."""

    def __init__(self):
        self.calls: list[tuple[int, list[int]]] = []
        self.columns: dict[int, dict[int, tuple[int, int]]] = {}
        self.lanes: list[dict] = [{} for _ in range(SEQUENCES * (1 + KEPT_STATES))]
        self.image: dict | None = None
        self.drafts: list[tuple[list, int]] = []

    def store(self, sequence, first, n, tap, bf16, column):
        self.calls.append((1, [sequence, first, n, tap, int(bf16), column]))
        for i in range(n):
            self.columns.setdefault(column + i, {})[tap] = (sequence, first + i)

    def _place(self, target, position, column):
        taps = self.columns[column]
        sources = set(taps.values())
        assert len(taps) == len(TAPS) and len(sources) == 1, f"column {column} mixes features"
        self.lanes[target][position % WINDOW] = (position, *sources.pop())

    def commit(self, target, position, column, count):
        self.calls.append((2, [target, position, column, count]))
        for i in range(count):
            self._place(target, position + i, column + i)

    def rows(self, column, rows):
        self.calls.append((3, [len(rows), column] + [w for row in rows for w in row]))
        for i, (target, position) in enumerate(rows):
            self._place(target, position, column + i)

    def copy(self, source, target):
        self.calls.append((4, [source, target]))
        self.lanes[target] = dict(self.lanes[source])

    def save(self, source, position):
        self.calls.append((5, [source, position]))
        self.image = {slot: self.lanes[source].get(slot) for slot in range(min(position, WINDOW))}

    def load(self, target, position):
        self.calls.append((6, [target, position]))
        assert self.image is not None and len(self.image) == min(position, WINDOW)
        for slot, entry in self.image.items():
            if entry is None:
                self.lanes[target].pop(slot, None)
            else:
                self.lanes[target][slot] = entry

    def draft(self, rows, k):
        self.calls.append((7, [len(rows), k] + [w for row in rows for w in row]))
        self.drafts.append(([(*row, self._context(row[0], row[2])) for row in rows], k))

    def _context(self, source, frontier):
        entries = []
        for position in range(max(0, frontier - WINDOW), frontier):
            entry = self.lanes[source].get(position % WINDOW)
            assert entry is not None and entry[0] == position, f"lane {source} lacks {position}"
            entries.append(entry)
        return entries

    def encode(self, path, embed, head, features):
        words = [MAGIC, VERSION, H, VOCAB, TARGET_LAYERS, DOMAIN, len(TAPS), SEQUENCES,
                 KEPT_STATES, MAX_CONTEXT, DRAFT_TOKENS, DRAFT_ROWS, COMMIT_ROWS, FEATURE_COLUMNS,
                 PIECE_COLUMNS]
        with open(path, "wb") as out:
            out.write(struct.pack(f"<{len(words)}I", *words))
            out.write(embed.view(torch.int16).numpy().tobytes())
            out.write(head.view(torch.int16).numpy().tobytes())
            for sequence in features:
                out.write(struct.pack("<I", sequence.shape[0]))
                out.write(sequence.contiguous().numpy().tobytes())
            out.write(struct.pack("<I", len(self.calls)))
            for code, operands in self.calls:
                out.write(struct.pack(f"<{1 + len(operands)}I", code, *operands))


def build_schedule(anchors: list[int]) -> Schedule:
    s = Schedule()
    taps = range(len(TAPS))
    # Sequence 0: a pass over positions 0..23 (FP32 taps, from column 3), one over 24..39 (BF16).
    for t in taps:
        s.store(0, 0, 24, t, False, 3)
    s.commit(lane(0), 0, 3, 24)
    for t in taps:
        s.store(0, 24, 16, t, True, 0)
    s.commit(lane(0), 24, 0, 16)
    s.copy(lane(0), lane(0, 1))  # the state kept at 40
    # Decode steps of sequence 0 (40..42) beside sequence 1 (0..2), a column each.
    for step in range(3):
        for t in taps:
            s.store(0, 40 + step, 1, t, False, 0)
            s.store(1, step, 1, t, True, 1)
        s.rows(0, [(lane(0), 40 + step), (lane(1), step)])
    # Sequence 2: one pass wider than the ring (its last window commits), then decode steps past
    # the ring's end beside sequence 1 (3, 4), then alone.
    for t in taps:
        s.store(2, 0, 2095, t, False, 0)
    s.commit(lane(2), 0, 0, 2095)
    for step in range(2):
        for t in taps:
            s.store(2, 2095 + step, 1, t, True, 0)
            s.store(1, 3 + step, 1, t, False, 1)
        s.rows(0, [(lane(2), 2095 + step), (lane(1), 3 + step)])
    for step in range(3):
        for t in taps:
            s.store(2, 2097 + step, 1, t, False, 0)
        s.rows(0, [(lane(2), 2097 + step)])
    # Sequence 1: a pass over 3..7 whose first two positions are committed already.
    for t in taps:
        s.store(1, 3, 5, t, True, 0)
    s.commit(lane(1), 5, 2, 3)
    # Lane (1, 1): lane 0's ring, then sequence 2's image at 2100 over it.
    s.copy(lane(0), lane(1, 1))
    s.save(lane(2), 2100)
    s.load(lane(1, 1), 2100)
    a = anchors
    s.draft([(lane(0), a[0], 43), (lane(0, 1), a[1], 40), (lane(1), a[2], 8),
             (lane(2), a[3], 2100)], 7)
    s.draft([(lane(1), a[4], 8)], 15)
    s.draft([(lane(1, 1), a[3], 2100)], 7)  # the loaded image ...
    s.draft([(lane(2), a[3], 2100)], 7)  # ... and its source
    s.draft([(lane(0), a[5], 43)], 1)
    s.draft([(lane(0), a[1], 40)], 7)  # a lane past its frontier ...
    s.draft([(lane(0, 1), a[1], 40)], 7)  # ... and the copy taken there
    return s


def read_outputs(path: Path, drafts) -> list:
    data = path.read_bytes()
    offset, results = 0, []
    for rows, k in drafts:
        n = len(rows) * k
        ids = struct.unpack_from(f"<{n}i", data, offset)
        confidence = struct.unpack_from(f"<{n}f", data, offset + 4 * n)
        offset += 8 * n
        results.append(([ids[j * k:(j + 1) * k] for j in range(len(rows))],
                        [confidence[j * k:(j + 1) * k] for j in range(len(rows))]))
    assert offset == len(data), "the driver wrote another number of proposals"
    return results


class Oracle:
    """The drafter in FP64 from the checkpoint's values (docs/maintainer/dspine.md)."""

    def __init__(self, tensors, embed, head):
        def g(name):
            return tensors[name].to(torch.float64)

        b = "selected_prefix_bridge."
        self.embed, self.head = embed.to(torch.float64), head.to(torch.float64)
        self.fc, self.hidden_norm, self.final_norm = (g("fc.weight"), g("hidden_norm.weight"),
                                                      g("norm.weight"))
        self.layers = []
        for i in range(LAYERS):
            p = f"layers.{i}."
            self.layers.append({
                "input_norm": g(p + "input_layernorm.weight"),
                "post_norm": g(p + "post_attention_layernorm.weight"),
                "q": g(p + "self_attn.q_proj.weight"),
                "k": g(p + "self_attn.k_proj.weight"),
                "v": g(p + "self_attn.v_proj.weight"),
                "o": g(p + "self_attn.o_proj.weight"),
                "q_norm": g(p + "self_attn.q_norm.weight"),
                "k_norm": g(p + "self_attn.k_norm.weight"),
                "gate": g(p + "mlp.gate_proj.weight"),
                "up": g(p + "mlp.up_proj.weight"),
                "down": g(p + "mlp.down_proj.weight"),
                "read": g(b + "read_weight") if i == LAYERS - 1
                else g(b + "early_read_weights")[i],
                "write": g(b + "feature_write_weights")[i],
            })
        self.projection = torch.cat(
            [g(b + n) for n in ("gate.weight", "predecessor_gate.weight", "prev.weight")])
        self.gate_bias = g(b + "gate.bias")
        # The reference's serial table: each token's [predecessor gate; value] rows from its BF16
        # code, one BF16 rounding.
        codes = (math.sqrt(CODE) * g(b + "codebook")).to(torch.bfloat16).to(torch.float64)
        senders = torch.cat([g(b + "predecessor_gate.weight"), g(b + "prev.weight")])
        self.sender = (codes @ senders.T).to(torch.bfloat16).to(torch.float64)
        self.head_write = self.head @ (self.final_norm[:, None] * self.layers[-1]["write"])
        self.inv_freq = float(THETA) ** (-torch.arange(0, D, 2, dtype=torch.float64) / D)

    @staticmethod
    def norm(x, weight):
        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + EPS) * weight

    def rope(self, x, positions):
        angle = positions.to(torch.float64)[:, None] * self.inv_freq[None, :]
        cos, sin = angle.cos()[:, None, :], angle.sin()[:, None, :]
        low, high = x[..., : D // 2], x[..., D // 2:]
        return torch.cat([low * cos - high * sin, high * cos + low * sin], -1)

    def context(self, rows, positions):
        """Each layer's context K and V of feature rows [n, taps * H] at `positions`."""
        n = rows.shape[0]
        c = self.norm(rows @ self.fc.T, self.hidden_norm)
        return [(self.rope(self.norm((c @ layer["k"].T).view(n, KV, D), layer["k_norm"]),
                           positions),
                 (c @ layer["v"].T).view(n, KV, D)) for layer in self.layers]

    def propose(self, context, context_positions, anchor, frontier, senders):
        """Each draft row's edges over the vocabulary (logit units: divided by the final norm's
        denominator) given its sender, senders[i] for row i, and its head logits over the candidate
        domain."""
        k, w = len(senders), len(senders) + 1
        positions = frontier + torch.arange(w, dtype=torch.int64)
        key_positions = torch.cat([context_positions, positions])
        visible = (positions[:, None] - key_positions[None, :]).abs() < WINDOW
        visible[:, context_positions.shape[0]:] &= torch.ones(w, w, dtype=torch.bool).tril()
        x = self.embed[[anchor] + [MASK] * k]
        normed = self.norm(x, self.layers[0]["input_norm"])
        for index, layer in enumerate(self.layers):
            q = self.rope(self.norm((normed @ layer["q"].T).view(w, HEADS, D), layer["q_norm"]),
                          positions)
            keys = self.rope(self.norm((normed @ layer["k"].T).view(w, KV, D), layer["k_norm"]),
                             positions)
            values = (normed @ layer["v"].T).view(w, KV, D)
            keys = torch.cat([context[index][0], keys]).repeat_interleave(HEADS // KV, dim=1)
            values = torch.cat([context[index][1], values]).repeat_interleave(HEADS // KV, dim=1)
            scores = torch.einsum("ihd,jhd->hij", q, keys) / math.sqrt(D)
            scores = scores.masked_fill(~visible[None], float("-inf"))
            attended = torch.einsum("hij,jhd->ihd", torch.softmax(scores, -1), values)
            x = x + attended.reshape(w, HEADS * D) @ layer["o"].T
            y = self.norm(x, layer["post_norm"])
            x = x + (torch.nn.functional.silu(y @ layer["gate"].T) * (y @ layer["up"].T)) \
                @ layer["down"].T
            # The bridge's write: its message replaces each column's predecessor by the anchor's
            # sender rows in column 1 and by the packed rows of column i - 1 after it.
            rms = torch.sqrt(torch.clamp((x * x).mean(-1), min=FLT_MIN))
            projected = (x / (rms[:, None] + BRIDGE_EPS)) @ layer["read"].T
            codes = math.sqrt(CODE) * projected / (projected.norm(dim=-1, keepdim=True)
                                                   + BRIDGE_EPS)
            packed = codes @ self.projection.T
            q_r, s_r, v_r = packed[:, :RANK], packed[:, RANK:2 * RANK], packed[:, 2 * RANK:]
            s_prev = torch.cat([self.sender[anchor, :RANK][None], s_r[1:-1]])
            v_prev = torch.cat([self.sender[anchor, RANK:][None], v_r[1:-1]])
            message = torch.zeros(w, RANK, dtype=torch.float64)
            message[1:] = torch.sigmoid(q_r[1:] + s_prev + self.gate_bias) * v_prev
            out = x.clone()
            out[1:] = x[1:] + rms[1:, None] * (message[1:] @ layer["write"].T)
            x = out
            d = torch.sqrt((x * x).mean(-1) + EPS)
            weight = (self.layers[index + 1]["input_norm"] if index + 1 < LAYERS
                      else self.final_norm)
            normed = x / d[:, None] * weight
        logits = normed[1:] @ self.head.T
        steps = []
        for i, sender in enumerate(senders):
            n = i + 1
            sent = (torch.sigmoid(q_r[n] + self.sender[sender, :RANK] + self.gate_bias)
                    * self.sender[sender, RANK:])
            edges = d[n] * logits[i] + rms[n] * (self.head_write @ (sent - message[n]))
            steps.append((edges / d[n], logits[i, :DOMAIN]))
        return steps


def confidence_range(edges, logits, chosen):
    """The least and greatest probability of `chosen` under softmax(edges) over the candidate sets
    that keep every logit above the oracle's 16th by more than GAP and fill the other places from
    the logits within GAP of it, or None when no such set holds `chosen`."""
    order = torch.argsort(logits, descending=True)
    sixteenth = logits[order[15]].item()
    sure = [c for c in order[:16].tolist() if logits[c].item() > sixteenth + GAP]
    near = [c for c in order.tolist() if abs(logits[c].item() - sixteenth) <= GAP]
    if chosen in sure:
        fixed, pool = sure, near
    elif chosen in near:
        fixed, pool = sure + [chosen], [c for c in near if c != chosen]
    else:
        return None
    free = 16 - len(fixed)
    weight = torch.exp(edges - edges[fixed + pool].max())
    base = weight[fixed].sum().item()
    others = sorted(weight[pool].tolist())
    heaviest, lightest = base + sum(others[len(others) - free:]), base + sum(others[:free])
    return weight[chosen].item() / heaviest, weight[chosen].item() / lightest


def verify(oracle, features, schedule, results) -> int:
    failures = []
    rows_of = [f.to(torch.bfloat16).to(torch.float64).reshape(f.shape[0], -1) for f in features]

    def context_of(entries):
        positions = torch.tensor([e[0] for e in entries], dtype=torch.int64)
        rows = (torch.stack([rows_of[sequence][p] for _, sequence, p in entries]) if entries
                else torch.zeros(0, len(TAPS) * H, dtype=torch.float64))
        return oracle.context(rows, positions), positions

    steps = agreed = changed = probed = 0
    worst_gap = worst_error = error_sum = 0.0
    errors = 0
    for call, ((rows, k), (drafts, confidence)) in enumerate(zip(schedule.drafts, results)):
        for j, (source, anchor, frontier, entries) in enumerate(rows):
            ids, probabilities = list(drafts[j]), list(confidence[j])
            label = f"call {call} row {j} (lane {source} at {frontier})"
            if any(not 0 <= x < DOMAIN for x in ids):
                failures.append(f"{label}: drafts {ids} leave the candidate domain")
                continue
            senders = [anchor] + ids[:-1]
            reference = oracle.propose(*context_of(entries), anchor, frontier, senders)
            # The first two calls also run without a context: the choices must depend on it.
            blind = oracle.propose(*context_of([]), anchor, frontier, senders) if call < 2 else None
            for i, (edges, logits) in enumerate(reference):
                candidates = torch.topk(logits, 16).indices
                best = candidates[torch.argmax(edges[candidates])].item()
                gap = (edges[best] - edges[ids[i]]).item()
                steps += 1
                agreed += best == ids[i]
                worst_gap = max(worst_gap, gap)
                if gap > GAP:
                    failures.append(f"{label} draft {i}: {ids[i]} trails the oracle's {best} by "
                                    f"{gap:.3f} logits")
                bounds = confidence_range(edges, logits, ids[i])
                if bounds is not None:
                    low, high = bounds
                    error = max(low - probabilities[i], probabilities[i] - high, 0.0)
                    worst_error, error_sum, errors = max(worst_error, error), error_sum + error, \
                        errors + 1
                    if error > CONFIDENCE:
                        failures.append(f"{label} draft {i}: confidence {probabilities[i]:.4f}, "
                                        f"the oracle's {low:.4f} to {high:.4f}")
                if blind is not None:
                    blind_edges, blind_logits = blind[i]
                    blind_candidates = torch.topk(blind_logits, 16).indices
                    probed += 1
                    changed += blind_candidates[torch.argmax(
                        blind_edges[blind_candidates])].item() != best
    for a, b, what in ((2, 3, "the lane loaded from an image"),
                       (5, 6, "the lane past its frontier")):
        if results[a] != results[b]:
            failures.append(f"{what} proposes {results[a]}, its source {results[b]}")
    agreement = agreed / max(steps, 1)
    mean_error = error_sum / max(errors, 1)
    sensitivity = changed / max(probed, 1)
    if agreement < AGREEMENT:
        failures.append(f"{agreed} of {steps} drafts are the oracle's choice (< {AGREEMENT:.0%})")
    if mean_error > MEAN_CONFIDENCE:
        failures.append(f"mean confidence error {mean_error:.4f} > {MEAN_CONFIDENCE}")
    if sensitivity < SENSITIVITY:
        failures.append(f"without its context the oracle changes {changed} of {probed} choices: "
                        "the check cannot see the ring")
    summary = (f"{steps} drafts, {agreed} the oracle's choice, worst gap {worst_gap:.3f} logits, "
               f"confidence error max {worst_error:.4f} mean {mean_error:.5f}; without the "
               f"context {changed} of {probed} choices change; image and rewind exact")
    for failure in failures:
        print("FAIL", failure)
    print(("FAIL" if failures else "PASS") + " DSpine runtime: " + summary)
    return 1 if failures else 0


def main() -> int:
    binary = sys.argv[1]
    probe = subprocess.run([binary, DRIVER, "--check-cuda"])
    if probe.returncode:
        return probe.returncode
    missing = [name for name in MODULES if importlib.util.find_spec(name) is None]
    if missing:
        print(f"SKIP: the DSpine runtime check needs {', '.join(missing)} in {sys.executable}")
        return 77
    global torch
    import torch as imported
    from safetensors.torch import save_file

    torch = imported
    generator = torch.Generator().manual_seed(20261112)
    tensors = checkpoint_tensors(generator)
    embed = torch.randn(VOCAB, H, generator=generator).to(torch.bfloat16)
    head = (3 / math.sqrt(H) * torch.randn(VOCAB, H, generator=generator)).to(torch.bfloat16)
    features = [torch.randn(n, len(TAPS), H, generator=generator) for n in POSITIONS]
    anchors = torch.randint(0, DOMAIN, (6,), generator=generator).tolist()
    anchors[4] = VOCAB - 3  # an id outside the candidate domain sends as well
    schedule = build_schedule(anchors)
    repository = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory(prefix="ninfer-dspine-") as scratch:
        root = Path(scratch)
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(json.dumps(config()))
        save_file(tensors, str(checkpoint / "model.safetensors"))
        artifact = root / "drafter.ninfer"
        convert = subprocess.run(
            [sys.executable, "-B", "-m", "tools.convert", "--model", str(checkpoint), "--recipe",
             "dspine_bf16", "--out", str(artifact), "--device", "cpu", "--name",
             "dspine-runtime-check"],
            cwd=repository, capture_output=True, text=True)
        if convert.returncode:
            print(convert.stdout + convert.stderr)
            print("FAIL: tools.convert refused the test checkpoint")
            return 1
        schedule_path, output_path = root / "schedule.bin", root / "drafts.bin"
        schedule.encode(schedule_path, embed, head, features)
        run = subprocess.run([binary, DRIVER, str(artifact), str(schedule_path),
                              str(output_path)])
        if run.returncode:
            return run.returncode
        results = read_outputs(output_path, schedule.drafts)
    return verify(Oracle(tensors, embed, head), features, schedule, results)


if __name__ == "__main__":
    raise SystemExit(main())
