"""FP64 reference forward of Inkling 2 Mini (Inkling2LoopForConditionalGeneration, text) over a
llama.cpp GGUF.

An independent transcription of llama.cpp's `inkling2` / `inkling2_loop` graphs (ggml-org
llama.cpp#32694 and Unsloth's looped layout) for golden values. Every value is FP64 over the GGUF's
exactly decoded blocks, so the only difference from a real engine is the engine's own arithmetic.

The model: the token row, RMS-normed. Then 48 layer visits of 24 stored layers: the prelude [0, 6),
the core [6, 18) three times and the coda [18, 24). Every visit keeps its own keys, values and
short-convolution history. A visit's attention reads its inputs through a depth attention over the
residual blocks so far (`mix`): the RMS-normed block states and the current state, weighted by a
softmax of their dot products with a learned query. Sliding-window visits mix one input and norm it
once; full-attention visits mix and norm one each for Q, K and V. K and V pass a causal 4-tap short
convolution (with per-head input-dependent taps on the full-attention visits), Q and K are clipped
to an L2 norm per head (Q after a per-channel scale), V is RMS-normed without a weight, and a
16-wide relative feature per head, clipped the same way, scores a learned table of distances
[0, extent) that adds to each logit. Full-attention visits scale Q and the relative feature by
1 + alpha log(max(1, (pos + 1) / floor)). A per-query, per-head sink logit joins each softmax
denominator. The output is gated by a sigmoid, projected, and convolved again. The first visit of
each 6-layer block starts a new residual stream (the block's input joins the mixed states); the
others add to theirs. The MoE input is mixed the same way and RMS-normed; 8 of 128 experts by
sigmoid score plus a selection bias, renormalized and scaled by 8, run SiTU (4 tanh(g / 4)
sigmoid(g) times 25 tanh(u / 25)) with per-expert output scales, and the sum is convolved and added.
Core visits after the first pass start again from the prelude's output; their four inputs (Q, K, V
and the MoE input) take a message mixed from the previous pass's states through a gain and a
forget gate of rank 64. The output mixes every block, RMS-norms, projects, soft-caps at 30 and drops
the padded vocabulary.

    python -m tools.reference.inkling2 --gguf Inkling-2-Mini-UD-IQ4_XS.gguf --tokens ids.json \\
        [--visits 48] [--out reference.npz]

`ids.json` holds the token ids (a JSON list). It prints the NLL of every target (token i + 1 given
tokens 0..i) and their total; `--out` keeps every visit's output, the routed experts, the DSpark
taps and the logits.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tools.convert.gguf_blocks import dequantize_blocks
from tools.convert.sources.gguf import TYPE_BF16, TYPE_F16, TYPE_F32, GGUFFile

DIRECT_TYPES = (TYPE_F32, TYPE_F16, TYPE_BF16)


class Weights:
    def __init__(self, gguf: GGUFFile):
        self.gguf = gguf
        self.cache: dict[str, np.ndarray] = {}

    def has(self, name: str) -> bool:
        return name in self.gguf.tensors

    def rows(self, name: str, begin: int = 0, end: int | None = None) -> np.ndarray:
        """Rows [begin, end) of a matrix (a 3-D bank counts its rows over every expert)."""
        info = self.gguf.info(name)
        if info.type_id in DIRECT_TYPES:
            # Only the rows asked for: a BF16 token table (GSQ-RCO's) is read a row at a time.
            width = info.shape[-1]
            end = int(np.prod(info.shape[:-1])) if end is None else end
            size = 4 if info.type_id == TYPE_F32 else 2
            raw = self.gguf.tensor_bytes(name, begin * width * size, end * width * size)
            if info.type_id == TYPE_F32:
                values = np.frombuffer(raw, dtype="<f4")
            elif info.type_id == TYPE_F16:
                values = np.frombuffer(raw, dtype="<f2")
            else:
                values = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
            return values.astype(np.float64).reshape(end - begin, width)
        blocks = self.gguf.read_blocks(name, begin, end)
        return dequantize_blocks(info.type_id, blocks).numpy().astype(np.float64)

    def matrix(self, name: str) -> np.ndarray:
        if name not in self.cache:
            self.cache[name] = self.rows(name)
        return self.cache[name]

    def vector(self, name: str) -> np.ndarray:
        return self.gguf.read_direct(name).astype(np.float64).reshape(-1)

    def table(self, name: str) -> np.ndarray:
        """A float matrix, rows as the GGUF lists them (slowest dimension first)."""
        info = self.gguf.info(name)
        return self.gguf.read_direct(name).astype(np.float64).reshape(info.shape)


def rms(x: np.ndarray, eps: float, w: np.ndarray | None = None) -> np.ndarray:
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return y * w if w is not None else y


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def clip(x: np.ndarray, threshold: float, scale: np.ndarray | None = None) -> np.ndarray:
    """Each row (last axis) scaled down to an L2 norm of at most `threshold`."""
    if scale is not None:
        x = x * scale
    norm = np.sqrt((x * x).sum(-1, keepdims=True)) + 1e-6
    return x * np.clip(threshold / norm, 0.0, 1.0)


def mix(states: list[np.ndarray], query: np.ndarray) -> np.ndarray:
    """Depth attention: `states` [S] x [n, H] (the current state last), `query` [Q, H] -> [Q, n, H]
    of softmax-weighted RMS-normed states."""
    s = np.stack([rms(state, 1e-6) for state in states])  # [S, n, H]
    scores = np.einsum("snh,qh->qns", s, query) / math.sqrt(s.shape[-1])
    scores = np.exp(scores - scores.max(-1, keepdims=True))
    scores /= scores.sum(-1, keepdims=True)
    return np.einsum("qns,snh->qnh", scores, s)


def convolve(x: np.ndarray, weight: np.ndarray, dynamic: np.ndarray | None) -> np.ndarray:
    """Causal short convolution of x [n, C] with taps weight [C, L] over the last L - 1 inputs and
    the current one (zero history), plus per-head input-dependent taps dynamic [n, heads * L], then
    the residual and the scale 1/sqrt((dynamic ? 2 : 1) L + 1)."""
    n, channels = x.shape
    taps = weight.shape[1]
    padded = np.concatenate([np.zeros((taps - 1, channels)), x])
    out = np.zeros_like(x)
    for i in range(taps):
        out += weight[:, i][None, :] * padded[i:i + n]
    if dynamic is not None:
        heads = dynamic.shape[1] // taps
        width = channels // heads
        d = dynamic.reshape(n, heads, taps)
        for lag in range(taps):
            window = padded[taps - 1 - lag:taps - 1 - lag + n].reshape(n, heads, width)
            out += (window * d[:, :, lag:lag + 1]).reshape(n, channels)
    return (x + out) / math.sqrt((2 if dynamic is not None else 1) * taps + 1)


class Config:
    def __init__(self, gguf: GGUFFile):
        kv = gguf.kv
        arch = kv["general.architecture"]
        if arch not in ("inkling2", "inkling2_loop"):
            raise ValueError(f"{gguf.path}: expected an inkling2 GGUF, found {arch}")
        a = arch + "."
        self.layers = 24 if arch == "inkling2" else int(kv[a + "block_count"])
        if arch == "inkling2":
            self.passes = int(kv.get(a + "num_loops", 1))
            self.core = (int(kv[a + "loop.core_start"]), int(kv[a + "loop.core_end"]))
            visits = int(kv[a + "block_count"])
            self.layers = visits - (self.passes - 1) * (self.core[1] - self.core[0])
            pattern = list(kv[a + "attention.sliding_window_pattern"])
            sliding = [pattern[self.visit_of(p)] for p in range(self.layers)]
        else:
            self.passes = int(kv[a + "loop.count"])
            self.core = (int(kv[a + "loop.start"]), int(kv[a + "loop.end"]))
            sliding = list(kv[a + "attention.sliding_window_pattern"])
        self.sliding = [bool(s) for s in sliding]
        self.arch = arch
        self.hidden = int(kv[a + "embedding_length"])
        heads = kv[a + "attention.head_count"]
        kv_heads = kv[a + "attention.head_count_kv"]
        self.heads = int(heads[0] if isinstance(heads, list) else heads)
        self.kv_heads = int(kv_heads[0] if isinstance(kv_heads, list) else kv_heads)
        self.head_dim = int(kv[a + "attention.key_length"])
        self.eps = float(kv[a + "attention.layer_norm_rms_epsilon"])
        self.window = int(kv[a + "attention.sliding_window"])
        self.rel = int(kv[a + "attention.relative_length"])
        self.extent = int(kv[a + "attention.relative_extent"])
        self.qk_clip = float(kv[a + "attention.qk_clip"])
        self.rel_clip = float(kv[a + "attention.relative_clip"])
        self.log_floor = float(kv[a + "attention.log_scale_floor"])
        self.log_alpha = float(kv[a + "attention.log_scale_alpha"])
        self.block = int(kv[a + "attn_res.block_size"])
        self.taps = int(kv[a + "shortconv.l_cache"])
        self.experts = int(kv[a + "expert_count"])
        self.top_k = int(kv[a + "expert_used_count"])
        self.route_scale = float(kv[a + "expert_weights_scale"])
        self.gate_cap = float(kv[a + "activation.situ_beta"])
        self.up_cap = float(kv[a + "activation.situ_linear_beta"])
        self.softcap = float(kv[a + "final_logit_softcapping"])
        self.vocab = int(kv[a + "vocab_size_unpadded"])

    def visit_of(self, physical: int) -> int:
        extra = (self.passes - 1) * (self.core[1] - self.core[0])
        return physical if physical < self.core[1] else physical + extra

    def schedule(self) -> list[tuple[int, int]]:
        """(physical layer, pass) of every visit in execution order."""
        out = [(p, 0) for p in range(self.core[0])]
        for k in range(self.passes):
            out += [(p, k) for p in range(*self.core)]
        out += [(p, 0) for p in range(self.core[1], self.layers)]
        return out


def message_queries(w: Weights, c: Config, physical: int) -> np.ndarray:
    """The four message queries (Q, K, V, MoE input) of a core layer, [4, H]."""
    if c.arch == "inkling2_loop":
        return w.table(f"blk.{physical}.loop_msg_query.weight")
    i = physical - c.core[0]
    q = w.table("loop_res_query.weight")
    kv = w.table("loop_kv_res_query.weight")
    return np.stack([q[2 * i], kv[2 * i], kv[2 * i + 1], q[2 * i + 1]])


def controller(w: Weights, c: Config, physical: int, which: int):
    names = (("loop_q_ctrl", "loop_k_ctrl", "loop_v_ctrl", "loop_ffn_ctrl")
             if c.arch == "inkling2_loop" else ("loop_attn", "loop_k", "loop_v", "loop_mlp"))
    stem = f"blk.{physical}.{names[which]}_"
    return (w.table(stem + "down.weight"), w.table(stem + "gain.weight"),
            w.table(stem + "forget.weight"))


def control(x: np.ndarray, message: np.ndarray, parts) -> np.ndarray:
    down, gain, forget = parts
    h = np.concatenate([rms(x, 1e-6), rms(message, 1e-6)], -1) @ down.T
    return x - np.tanh(h @ forget.T) * x + np.tanh(h @ gain.T) * message


def attention(q, k, v, r, table, sinks, pos, sliding, c: Config) -> np.ndarray:
    n = q.shape[0]
    group = c.heads // c.kv_heads
    dist = (pos[:, None] - pos[None, :]).astype(np.int64)
    visible = dist >= 0
    if sliding:
        visible &= dist < c.window
    extent = c.window if sliding else c.extent
    out = np.zeros((n, c.heads, c.head_dim))
    for h in range(c.heads):
        s = (q[:, h, :] @ k[:, h // group, :].T) / math.sqrt(c.head_dim)
        bias_rows = r[:, h, :] @ table.T  # [n, extent]
        inside = (dist >= 0) & (dist < extent)
        s = s + np.where(inside, np.take_along_axis(
            bias_rows, np.clip(dist, 0, extent - 1), axis=1), 0.0)
        s = np.where(visible, s, -np.inf)
        top = np.maximum(s.max(-1, keepdims=True), sinks[:, h:h + 1])
        p = np.exp(s - top)
        denominator = p.sum(-1, keepdims=True) + np.exp(sinks[:, h:h + 1] - top)
        out[:, h, :] = (p / denominator) @ v[:, h // group, :]
    return out


def moe(w: Weights, c: Config, physical: int, x: np.ndarray, record: dict) -> np.ndarray:
    g = f"blk.{physical}."
    logits = x @ w.matrix(g + "ffn_gate_inp.weight").T
    probs = sigmoid(logits)
    selection = probs + (w.vector(g + "exp_probs_b.bias") if w.has(g + "exp_probs_b.bias") else 0)
    width = w.gguf.info(g + "ffn_down_exps.weight").shape[-1]
    gu_scale = (w.vector(g + "ffn_gate_up_exps.scale") if w.has(g + "ffn_gate_up_exps.scale")
                else np.ones(c.experts))
    down_scale = (w.vector(g + "ffn_down_exps.scale") if w.has(g + "ffn_down_exps.scale")
                  else np.ones(c.experts))
    chosen = np.argsort(-selection, axis=-1, kind="stable")[:, :c.top_k]
    out = np.zeros_like(x)
    for e in np.unique(chosen):
        tokens, _ = np.nonzero(chosen == e)
        weights = (probs[tokens, e] / (np.take_along_axis(probs, chosen, -1)[tokens].sum(-1) + 1e-20)
                   * c.route_scale)
        e = int(e)
        gu = x[tokens] @ w.rows(g + "ffn_gate_up_exps.weight", e * 2 * width,
                                (e + 1) * 2 * width).T * gu_scale[e]
        gate, up = gu[:, :width], gu[:, width:]
        act = (c.gate_cap * np.tanh(gate / c.gate_cap) * sigmoid(gate)
               * c.up_cap * np.tanh(up / c.up_cap))
        down = act @ w.rows(g + "ffn_down_exps.weight", e * c.hidden, (e + 1) * c.hidden).T
        out[tokens] += weights[:, None] * down * down_scale[e]
    record.setdefault("experts", []).append(np.stack(chosen))
    return out


def forward(gguf: GGUFFile, ids: list[int], visits: int | None = None) -> dict[str, np.ndarray]:
    c = Config(gguf)
    w = Weights(gguf)
    n = len(ids)
    pos = np.arange(n, dtype=np.float64)
    tau = 1.0 + c.log_alpha * np.log(np.maximum(1.0, (pos + 1.0) / c.log_floor))
    x = np.stack([w.rows("token_embd.weight", t, t + 1)[0] for t in ids])
    cur = rms(x, c.eps, w.vector("token_embd_norm.weight"))
    out: dict[str, np.ndarray] = {}
    record: dict = {}
    blocks: list[np.ndarray] = []
    prelude_blocks: list[np.ndarray] = []
    prelude_cur = None
    bank: list[np.ndarray] = []
    bank_last = None
    taps = {}
    schedule = c.schedule()
    if visits is not None:
        schedule = schedule[:visits]
    hd = c.head_dim
    for il, (ip, k) in enumerate(schedule):
        g = f"blk.{ip}."
        core = c.core[0] <= ip < c.core[1]
        messages = None
        if core and ip == c.core[0]:
            if k == 0:
                prelude_blocks = list(blocks)
                prelude_cur = cur
            else:
                bank = blocks[len(prelude_blocks) + 1:]
                bank_last = cur
                blocks = list(prelude_blocks)
                cur = prelude_cur
        if core and k > 0:
            messages = mix(bank + [bank_last], message_queries(w, c, ip))
        taps[il] = cur
        sliding = c.sliding[ip]
        inp = cur
        inputs = [cur, cur, cur]
        if ip > 0:
            query = w.table(g + "attn_res_query.weight").reshape(-1, c.hidden)
            mixed = mix(blocks + [cur], query)
            inputs = [mixed[0 if sliding else j] for j in range(3)]
        if messages is not None:
            inputs = [control(inputs[j], messages[j], controller(w, c, ip, j)) for j in range(3)]
        if sliding:
            normed = [rms(v, c.eps, w.vector(g + "attn_norm.weight")) for v in inputs]
        else:
            normed = [rms(inputs[0], c.eps, w.vector(g + "attn_q_inp_norm.weight")),
                      rms(inputs[1], c.eps, w.vector(g + "attn_k_inp_norm.weight")),
                      rms(inputs[2], c.eps, w.vector(g + "attn_v_inp_norm.weight"))]
        q = normed[0] @ w.matrix(g + "attn_q.weight").T
        kd = None if sliding else normed[1] @ w.matrix(g + "attn_k_conv_proj.weight").T
        vd = None if sliding else normed[2] @ w.matrix(g + "attn_v_conv_proj.weight").T
        kk = convolve(normed[1] @ w.matrix(g + "attn_k.weight").T,
                      w.table(g + "attn_k_conv.weight"), kd)
        vv = convolve(normed[2] @ w.matrix(g + "attn_v.weight").T,
                      w.table(g + "attn_v_conv.weight"), vd)
        r = normed[0] @ w.matrix(g + "attn_rel.weight").T
        q = clip(q.reshape(n, c.heads, hd), c.qk_clip, w.table(g + "attn_q_clip.weight"))
        kk = clip(kk.reshape(n, c.kv_heads, hd), c.qk_clip)
        vv = rms(vv.reshape(n, c.kv_heads, hd), c.eps)
        r = clip(r.reshape(n, c.heads, c.rel), c.rel_clip, w.table(g + "attn_rel_clip.weight"))
        if not sliding:
            q = q * tau[:, None, None]
            r = r * tau[:, None, None]
        sinks = normed[0] @ w.matrix(g + "attn_sink_proj.weight").T
        table = w.table(g + "attn_rel_bias.weight")
        a = attention(q, kk, vv, r, table, sinks, pos, sliding, c).reshape(n, c.heads * hd)
        a = a * sigmoid(normed[0] @ w.matrix(g + "attn_gate.weight").T)
        a = convolve(a @ w.matrix(g + "attn_output.weight").T,
                     w.table(g + "attn_post_conv.weight"), None)
        if ip % c.block == 0:
            blocks.append(inp)
            cur = a
        else:
            cur = inp + a
        ffn_in = cur
        m = mix(blocks + [cur], w.vector(g + "ffn_res_query.weight")[None, :])[0]
        if messages is not None:
            m = control(m, messages[3], controller(w, c, ip, 3))
        m = rms(m, c.eps, w.vector(g + "ffn_norm.weight"))
        y = convolve(moe(w, c, ip, m, record), w.table(g + "ffn_post_conv.weight"), None)
        cur = ffn_in + y
        out[f"visit-{il}"] = cur.copy()
    final = mix(blocks + [cur], w.vector("output_res_query.weight")[None, :])[0]
    final = rms(final, c.eps, w.vector("output_norm.weight"))
    taps[len(schedule)] = final
    logits = final @ w.matrix("output.weight")[:c.vocab].T
    logits = c.softcap * np.tanh(logits / c.softcap)
    out["final_hidden"] = final
    out["logits"] = logits
    for i, value in taps.items():
        out[f"tap-{i}"] = value
    out["experts"] = np.stack(record["experts"]) if record.get("experts") else np.zeros(0)
    return out


def target_nll(logits: np.ndarray, ids: list[int]) -> np.ndarray:
    nll = []
    for i in range(len(ids) - 1):
        row = logits[i]
        top = row.max()
        nll.append(top + math.log(np.exp(row - top).sum()) - row[ids[i + 1]])
    return np.array(nll)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--tokens", required=True, help="JSON list of token ids")
    parser.add_argument("--visits", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    ids = json.loads(Path(args.tokens).read_text())
    with GGUFFile(args.gguf) as gguf:
        result = forward(gguf, ids, args.visits)
    if args.visits is None:
        nll = target_nll(result["logits"], ids)
        for i, value in enumerate(nll):
            print(f"target {i + 1} ({ids[i + 1]}): nll {value:.6f}")
        print(f"total nll {nll.sum():.6f} over {len(nll)} targets, mean {nll.mean():.6f}")
        print("argmax", [int(v) for v in result["logits"].argmax(-1)])
    if args.out:
        np.savez(args.out, **result)


if __name__ == "__main__":
    main()
