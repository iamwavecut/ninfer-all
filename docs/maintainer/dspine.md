# DSpine

DSpine ([Tencent](https://github.com/Tencent/DSpine), "Draft in Parallel, Condition Through
Depth: Adjacent Causal Injection for Speculative Decoding",
[arXiv 2609.36173](https://arxiv.org/abs/2609.36173), Apache-2.0) is a masked-block drafter of the
[DFlash](dflash.md) family. Its layers attend a ring of the target's tapped hidden states as
DFlash's do, but causally inside the drafted block. After every layer a code bridge writes into each
block column a message from the column before it, so the drafter carries the predicted
predecessor's features through its depth. A serial refinement of the last write then chains the
head's candidates into drafts.

NInfer runs a DSpine drafter as its own artifact beside the model: `--spec dspine --draft-model
PATH` (`draft-model` in a launch file). Every family takes one: the Qwen3.5 core (dense and MoE
releases), Qwen3.8-Flash-Next with its Edge0 form, Gemma 4, Flint and DeepSeek-V4.1. A drafter is
trained for one target, and startup refuses a drafter whose hidden width, vocabulary or target layer
count differs from the model's. Tencent publishes one drafter, for Qwen3.8-27B
([converted](https://huggingface.co/WaveCut/Qwen3.8-27B-DSpine-b16-NInfer-v3)). The other families'
paths have run only in synthetic tests, against drafters of their geometry with random weights.

This reference owns the drafter's mathematics, artifact and state. [DFlash and DFlash2](dflash.md)
defines the conditioning, verification and publication that DSpine shares, and
[Engine architecture](engine-architecture.md) the scheduling.

| Code | Owns |
|---|---|
| [`tools/convert/dspine.py`](../../tools/convert/dspine.py) | the checkpoint's validation and the `dspine` component |
| [`models/dspine/config.h`](../../src/models/dspine/config.h), [`model.h`](../../src/models/dspine/model.h) | the config and the materialized drafter |
| [`models/dspine/drafter.h`](../../src/models/dspine/drafter.h) | the context projection, the drafted block and the head-write table |
| [`models/dspine/runtime.h`](../../src/models/dspine/runtime.h) | lanes, features and staging for Gemma 4, Flint and DeepSeek-V4.1 |
| [`ops/dspine.h`](../../include/ninfer/ops/dspine.h) | the bridge and refinement Ops |

## The artifact

```bash
python -m tools.convert --model CHECKPOINT --recipe dspine_q8 --out drafter.ninfer
```

converts a DSpine checkpoint (`DFlashDraftModel` whose `dflash_config` has
`selected_prefix_code_bridge`) into an artifact whose only component is `dspine`, architecture
`DSpineDraftModel`. The converter refuses what the engine does not implement: a block that is not
causal, a terminal projection, a decode mode other than `serial`, writes other than independent
ones after every layer, a refinement other than `exact_sender` version 1 over the head's top 16,
DFlash2's convolution and selector fields, and version markers other than 1.

| Recipe | Formats |
|---|---|
| `dspine_q8` | the layer and feature projections Q8 (groups of 32, FP16 scales); the bridge and its tables BF16 |
| `dspine_bf16` | every weight BF16, as the checkpoint stores it |

The bridge stays BF16 in both recipes, because the refinement's roundings depend on it. The layers
bind as a DFlash drafter's do, and each layer's context K/V shares the data of its own K/V. The
bridge parameters are:

| Parameter | Shape | Source |
|---|---|---|
| `bridge/projection` | `[3R, C]` | the gate, predecessor-gate and value weights, one below the other |
| `bridge/gate_bias` | `[R]` | the gate's bias |
| `bridge/layers/i/read` | `[C, H]` | `early_read_weights[i]`, and `read_weight` for the last layer |
| `bridge/layers/i/write` | `[H, R]` | `feature_write_weights[i]` |
| `bridge/head_write` | `[R, H]` | the last layer's write, transposed |
| `bridge/sender_table` | `[V, 2R]` | each token's predecessor-gate and value rows |

The codebook is not stored. At inference its only use is a token's sender rows, which the converter
evaluates as the reference's table builder does, with FP32 accumulation:

```text
sender_table[v] = BF16( BF16(sqrt(C) * codebook[v]) . [W_pred_gate; W_prev]^T )
```

The table that the refinement reads through the target's head depends on that head and its format,
so the engine derives it when it loads the two together ([below](#candidates-and-the-refinement)).
The checkpoint's `kappa` is a training parameter that inference does not read.

Tencent's Qwen3.8-27B drafter, revision `5f54174d`:

| Quantity | Value |
|---|---:|
| Hidden width H / vocabulary V | 5120 / 248320 (the target's) |
| Layers | 5, sliding window 8192 |
| Intermediate width | 17408 |
| Q heads / KV heads / head dimension | 32 / 8 / 128 |
| Target layers tapped | `[1,16,31,46,61]` of 64 |
| Block | 16: the anchor and 15 masks |
| Mask token | 248070 |
| Code width C / message rank R | 1024 / 512 |
| Norm and bridge epsilon / RoPE theta | `1e-6` / `1e7` |
| Parameters | 2,025,912,064, BF16 in the checkpoint |

The config accepts 32 query and 8 KV heads of 128 (the attention Op's profile), a rank that is a
multiple of 32 up to 512, a block of 2 to 16, a trained window of 2048, 4096 or 8192, norm epsilon
`1e-6` and RoPE theta `1e7`, and refuses other values.

## Conditioning and context

The drafter conditions on the target as [DFlash](dflash.md#target-conditioning-and-context) does:
the tapped layers' features side by side, projected and normalized into `c_t`, and each layer's
context K and V projected from `c_t`. The family defines a tap:

| Family | Feature of target layer l at position t |
|---|---|
| Qwen3.5 core | the residual after block l (DFlash's taps) |
| Qwen3.8-Flash-Next | the mean of the four hyper-connection streams after block l (DFlash2's taps) |
| Gemma 4, Flint | the residual after block l |
| DeepSeek-V4.1 | the mean of the residual streams after block l |

A feature is rounded to BF16 when it is stored. The context of a sequence lives in a ring of K/V
slots per drafter layer. Its capacity is the smallest of 2048, 4096 and 8192 that holds
`max_context + block` positions, at most the trained window, and the attention reads it with a
window equal to its capacity. A narrower window than the trained one is exact: no visible key is
farther away than it. The Qwen3.8-27B drafter takes a ring of 2048 positions up to a 2,032-token
context, 4096 up to 4,080 tokens and 8192 beyond.

```text
ring bytes = layers * 2 (K, V) * 2 bytes * KV heads * head_dim * capacity
           = 5 * 2 * 2 * 8 * 128 * 8192 = 160 MiB a lane (40 MiB at 2048)
```

## The drafted block

The block holds the anchor `a`, the token at the frontier F, and K masks at positions `F+1..F+K`,
with K at most the trained block less one. Column i attends the ring's context and the block
columns `j <= i`, each key at a distance below the window
([`sliding_window_attention.h`](../../include/ninfer/ops/sliding_window_attention.h),
`SlidingWindowMask::Causal`). Each layer runs DFlash's backbone up to its MLP output:

```text
n   = plain_rmsnorm(x, input_norm)                    (x: the token rows, then each bridge's out)
y   = x + W_output attention(q, k, v)                 (q, k, v as DFlash)
raw = y + W_down(SiLU(W_gate m) * W_up m),  m = plain_rmsnorm(y, post_attention_norm)
```

The bridge after layer l writes into every column but the anchor's:

```text
rms_i     = sqrt(max(mean_h raw_i^2, FLT_MIN))
p_i       = W_read^l BF16(raw_i / (rms_i + e_b))
code_i    = BF16(sqrt(C) * p_i / (|p_i| + e_b))
[g_i; s_i; w_i] = W_bridge code_i                     (receiver gate, sender gate, sender value)
msg_0     = 0
msg_i     = BF16(sigmoid(g_i + s_(i-1) + b) * w_(i-1))         i >= 1
out_0     = raw_0
out_i     = BF16(raw_i + rms_i * W_write^l msg_i)              i >= 1
```

The converter stores a layer's query, key and value as rows of one parent, and its gate and up as
rows of another, as it does a DFlash drafter's. Where those parents and the output and down
projections have a registered fused form (the `dspine_q8` drafter for Qwen3.8-27B: Q8 parents
`[6144,5120]` and `[34816,5120]`, down `[5120,17408]`), a layer projects q, k and v in one pass
(`attn_input_proj`), normalizes and rotates q and k in place in one Op (`rmsnorm_rope`, up to 16
columns in up to 8 rows), gates the MLP inside its projection (`linear_swiglu`) and adds the down
projection into the residual as it computes it (`linear_add`), as it does an output projection of a
registered form (the 27B drafter's `[5120,4096]` has none). The context's K and
V project in one pass where `linear_pair` takes them. Each fused Op is qualified against its own
FP64 oracle; a fused layer rounds at fewer points than the separate Ops, so its drafts may differ
in the last bits, never the verified output.

The anchor is a known token, so column 1 reads its sender rows `[s_0; w_0]` from the sender table
at `a`, not from the anchor column's own code. `out` is the next layer's x and, normalized by the
next layer's input norm or after the last layer by the final norm, its n. The last layer also keeps
the final norm's denominator `d_i = sqrt(mean_h out_i^2 + e)` for the refinement.
[`dspine.h`](../../include/ninfer/ops/dspine.h) states every rounding point.

## Candidates and the refinement

The target's own head maps each mask column's final hidden `h_i` to logits `u_i`; Gemma 4's head
runs without its final soft cap, since the refinement needs logits linear in the hidden. The 16
best candidates of each row come from the public vocabulary. On the Qwen3.5 core a Q8, FP8 or T2
head of 248,320 rows by 5120 ranks them in the same pass that projects the hidden, with FP32 scores
and no dense logits (`linear_topk`); every other head writes its logits and `candidate_topk` ranks
them. With `--lm-head-draft` the candidates come from the artifact's proposal head instead: its
131,072 rows, the shortlist DFlash2 drafts with, ranked the same way and mapped to their token ids.
A token outside the shortlist is then never drafted; the head-write table still comes from the full
head, so every candidate's table row is the full head's. Draft row i is
block column `n = i+1`. Its sender t is the anchor for row 0 and the previous row's draft otherwise,
and an edge to candidate c replaces the column's own message by the one that t sends:

```text
m_t[r]    = sigmoid(g_n[r] + sender_table[t, r] + b[r]) * sender_table[t, R + r]
edge(t,c) = d_n * u_i[c] + rms_n * sum_r (m_t[r] - msg_n[r]) * TG[C_i[c], r]
TG[v, r]  = BF16( Head(w_final * W_write^last[:, r])[v] )
```

`d_n * u_i[c]` is the candidate's logit before the final norm's division, and the sum adds what t's
message would add to it through the head. An edge is thus the logit, times `d_n`, that the
candidate would take had the last write carried t's message, with the column's own norm
denominator. The engine derives the head-write table TG once at startup, running the target's head
over the R columns of the last write scaled by the final norm's weight. It takes `R * V * 2` bytes,
243 MiB for the Qwen3.8-27B drafter, on the head's device.

A row at temperature 0 takes the candidate of greatest edge (the lowest id on a tie). A row at
temperature T draws from `softmax(edge / (d_n * T))` over the 16 candidates, truncated by the
request's top-k, top-p and min-p and renormalized, with the key of the request's seed, position
`F+i` and the DSpine proposal purpose; the Op records that law as the proposal's q. A draft's
confidence is its probability under `softmax(edge / d_n)`.

The reference computes the greedy score as `Head(w_final * raw_n) + rms_n * m_t . TG`, in FP32 from
the stored raw row. NInfer evaluates the same quantity from the head's output, so the two round
differently. The reference's sampled walk also normalizes each sender's refined row by that row's
own RMS and reads the sender rows in FP32 from the codebook. NInfer keeps the column's `d_n` and
the BF16 sender table in both walks. The target verifies against the recorded q, so these
differences can change acceptance only, not the output distribution.

## Verification by family

| Family | Drafts a round | Proposal | Acceptance | `--draft-min-p P` |
|---|---|---|---|---|
| Qwen3.5 core | 1 to min(15, block − 1); with `--adaptive-mtp` the window K or K/2 a round | greedy, or sampled at the request's temperature | DFlash2's: greedy prefix, or rejection sampling against the recorded q | refused |
| Qwen3.8-Flash-Next, Edge0 | 1 to min(15, block − 1) | greedy | coupled, as MTP's | verifies through the first draft of confidence at or below P |
| Gemma 4, Flint, DeepSeek-V4.1 | 1 to 7 | greedy | coupled: the target's keyed draws in plain decode rows | as Flash-Next |

Every family commits only tokens that the target selects, so the output is the target's at any
temperature. `--gumbel-sampling` changes only the target's draws: the greedy proposals stay greedy
and the Qwen3.5 core keeps rejection sampling, as DFlash2 does. A DSpine round runs no n-gram copy
proposal unless `--ngram-draft-tokens` sets its width, at least the draft window (the Qwen3.5 core).
A copy round of several requests runs the drafter at that width as well, for the rows without a
copy, so above one request the width is at most the block less one. Under `--adaptive-mtp` the
Qwen3.5 core drafts and verifies each round without a copy at the window K or at K/2, whichever
the measured survival and round time favor ([Adaptive MTP](../serving.md#adaptive-mtp)). The
block is causal, so a narrow round drafts the first K/2 drafts of the wide round, up to the
rounding of kernels that take another route at the other width; a DFlash or DFlash2 block attends
both ways and drafts other tokens at another width.

## State

**Qwen3.5 core.** The ring is the masked draft's local state, one per state image, so checkpoints,
forks, Host replicas and the hybrid prefix cache carry it as they carry DFlash2's
([backend storage](dflash.md#backend-storage-and-lifecycle)). Pending features and their catch-up
are DFlash's. The head-write table is a persistent tensor beside the draft context, and the drafter's
weights load on the head's device (rank 0 of pipeline stages or tensor parallelism). Session
persistence (`--slot-save-path`) refuses DSpine as it refuses DFlash and DFlash2. The Legacy disk
tier keys its directory by the drafter's artifact as well as the model's, so one drafter's ring
never seeds another's.

**Qwen3.8-Flash-Next and Edge0.** The drafter takes DFlash2's place in the executor: its features, ring lanes,
commits and CUDA Graphs, with DSpine's ring capacity, on one device or on the head's device under
tensor parallelism; pipeline stages are refused. The context cache's store name carries the
drafter's artifact digest and ring capacity.

**Gemma 4, Flint and DeepSeek-V4.1** keep their own sequences, so
[`dspine::Runtime`](../../src/models/dspine/runtime.h) holds the drafter's state beside them: a ring
lane per sequence and one more per sequence with checkpoints, the head-write table, the features of
a pass, and pinned staging. A pass stores the taps of the rows it computes. A prompt chunk commits
its last window of rows into the sequence's lane; a decode step commits its rows with
`commit_rows`, which is device work only, so a decode graph may capture it; a verification commits
the rows it keeps. Branch passes commit nothing. A snapshot copies the lane with the target's state,
and a context-cache image carries the lane's slots of the positions it holds. A rewind leaves the
lane's later positions in place, and the sequence writes them again before a proposal reads them.
Only a ring that has wrapped (a context longer than its capacity) then holds keys of positions given
back in its oldest slots, which can lower acceptance and leaves the output exact. DeepSeek-V4.1's
hold state takes no lane, with the same consequence. Proposals for up to eight sequences run as one
block, eagerly.

When the drafter taps one of DeepSeek-V4.1's decoder blocks (those after its last KV source block),
the [prefill tail](../deepseek-v41.md#the-prefill-tail) is off: the drafter reads that block's output
at every row the sequence keeps.

## Tests

| Test | Checks |
|---|---|
| `tests/convert/test_dspine.py` | the converter's refusals, parameter layout and sender table against the formula |
| `tests/ops/test_dspine.cpp` | each Op against an FP64 oracle of its formula, on represented inputs |
| `tests/ops/test_sliding_window_attention.cpp`, `test_kv_cache_append.cpp` | the causal mask and the 8192-slot ring |
| `tests/models/dspine/runtime_check.py` | the runtime and drafter over a converted random drafter against an FP64 model: drafts, confidences, lanes, copies and images |
| `tests/test_cli_options.cpp`, `tests/test_serve_options.cpp` | the options and their refusals |
