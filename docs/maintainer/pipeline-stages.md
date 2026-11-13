# Multi-GPU: layer pipeline stages

`--devices A,B,...` splits the model's layers into one pipeline stage per GPU. Each stage owns its
layers whole: weights, the KV cache of its attention layers, the recurrent state of its GDN layers,
and the scratch it runs in. The point is memory: a model that does not fit one card, or a context
that does not, spreads across several with every card's memory usable for KV.

## What it is and is not

**It is a memory feature.** The stages run in sequence: while one computes, the others wait.
Decode is weight-bandwidth-bound and each stage reads only its own weights, so a single stream is
about as fast as one GPU, minus the boundary hops. Splitting the batch into micro-batches would make
each stage read its weights once per micro-batch and win nothing.

**It is not tensor parallelism.** No layer is divided across devices, so no collective runs inside a
layer. `--tensor-parallel` divides them instead; see [Tensor parallelism](#tensor-parallelism).

**Linux and Windows.** Repeating one device id (`--devices 0,0`) exercises the whole stage path on
a single card, which is how the path is tested without a second GPU. Under Windows a GeForce card
runs the WDDM driver model, which gives no peer access, so the stages exchange their boundary rows
through pinned host memory (`StageLink`), as Linux does between cards without peer access. WDDM
charges pinned host memory against a card's memory (measured on one card); assuming it charges
every card of the pipeline, the pinned host KV cache is sized for the rank with the least device
memory free. This repository has no Windows host with several GPUs: the Windows path is compiled in
CI and was run by a contributor, not measured here.

**The cards may differ.** Every listed device must run the build's code, which `DeviceContext`
(`core/device.cu`) checks before any weight is uploaded by loading a kernel on each, and the model
family checks each device's compute capability. An `86` build therefore pipelines RTX 30 with RTX
40 cards, and an `80` build, whose sm_80 code also runs on 8.6 and 8.9, an A100 with either. A build
of several architectures (`70;75;80;86;89`, `75;80;86;89;120a`) pipelines cards of different code
tiers, such as a V100 with a TITAN RTX and RTX 30 cards. Each rank runs the code it would run alone:
its tier's kernels, shapes and routes (`ops/common/device_tier.h`), the weight layout and
activation permissions of its tier (`apply_device_layouts`, `apply_rank_policies`), and the route
profile of its own card. The context-cost model of the prefix cache is the first device's.

**Peer access is checked before it is used.** A pair that reports peer capability
(`cudaDeviceCanAccessPeer`) is refused, without touching its mappings, when either card sits in an
IOMMU group that translates its DMA (`/sys/bus/pci/devices/<address>/iommu_group/type` other than
`identity`; boot with `iommu=pt`) or is a GeForce card whose BAR1 is smaller than its memory (the
GeForce P2P driver patches need a resizable-BAR VBIOS, and without one they hand out mailbox
mappings that drop stores, return garbage loads and can reset the host when probed;
`core/peer_preflight.h`). A pair that passes is probed once: a kernel stores into and loads from the
peer's memory and the copy engine copies into it, and any wrong value disables peer access for that
direction. Refused directions stage through pinned host memory and are reported at startup as
`peer access:` warnings with the reason.

## How it works

- **`StagePlan`** (`core/stage_plan.h`) says which contiguous layers each stage owns.
  `--stage-layers 30,34` sets the counts. Without it `default_stage_layers`
  (`models/qwen3_5/load.cpp`) sizes each layer from the artifact and calls `solve_stage_plan`, an
  exact memory-balancing solver (maximum page groups every stage can hold, then least-full stage),
  verified against a brute-force partition oracle. It works from the bytes free on each device, the
  stored KV page-group and per-slot GDN state bytes of each layer (from the engine's KV storage
  format and slot count), and rough constants for what a stage needs beyond weights (context,
  workspace, graphs; more on rank 0 for the head). The head stage also holds the embedding, the
  head, and as selected MTP's layer with its KV per page group, the proposal head, a DFlash drafter
  and a resident Vision tower; automatic KV capacity keeps its headroom free on every device. It is
  a good default rather than exact; if it finds nothing fits it deals the layers evenly and lets the
  Program's planning report the device that runs out.
- **Placement.** `bind_text` places every layer's weights on its stage's device
  (`Bindings::place`, recorded for rank 0 as well so an object shared between a rank-0 layer and a
  later stage's layer is refused as a conflicting placement). The embedding, final norm and head stay
  on rank 0.
- **The head stays on rank 0.** The last stage sends the residual back; rank 0 then runs the final
  norm, head and sampling exactly as on one GPU. That costs one extra ~20 us hop per forward pass and
  keeps the round buffers, prefill buffers and every other head-side structure where they are.
- **Per-rank state.** `DeviceKVPagePool` is one page-group allocator over planes bound to several
  ranks' backings, so admission, prefix reuse and the context-cache policy never see how many devices
  hold the cache. Block tables are one host shadow with a device copy per attention-bearing rank
  (`KVExecutionTablePool`); GDN state is one `LinearAttentionStatePool` shard per stage
  (`StateImageDevicePool`); the host image of a state keeps the whole model's layers at their global
  offsets. Data movement takes `RankStreams`, so each plane's copy goes on its own rank's stream, and a
  single stream given to a multi-rank object fails instead of using the wrong device.
- **Capacity.** Each further device has its own affine reservation curve
  (`RankCapacityCurve`); the resolved page-group count is the smallest any device allows, and a
  failure names the device that could not hold the plan. Ranks sharing one physical device split its
  free memory.
- **`StageLink`** (`core/stage_link.h`) moves the residual between stages, and the small control
  tensors (positions, KV rows, slot indices, valid columns) the layers read from rank 0, through a
  ring of pinned-host slots. The copies are ordinary graph nodes. Decode captures the whole pass into
  one multi-device graph, so `StageLink` tracks, per event, which capture made its last record: a wait
  inside a capture may only target an event recorded in that capture, and an eager wait may not target
  one whose last record was in a capture (`cudaErrorInvalidValue`, even after the graph has
  launched).
- **`TextContext::run_staged`** packs the control tensors on rank 0, sends each stage its block up
  front, runs stage 0, and for each later stage rebinds the context's device, stream and workspace
  (`ScopedDeviceRank`, `ScopedArenaRank`), receives the residual and control block, runs the stage's
  layers and passes the residual on.

## What is not covered yet

- **Prefill does not overlap stages.** A prefill chunk runs through the stages in turn and the
  engine synchronizes after each chunk, so at any moment one stage is busy. Overlapping stages needs
  micro-chunks inside a chunk (later stages start on micro-chunk 0 while stage 0 runs micro-chunk 1);
  that changes the chunk shapes the kernels see, and its benefit can only be measured on real cards.
- **Vision works** across stages with resident or CPU residency. The tower, the encode workspace
  and the scatter of the image's embeddings into the prompt stay on rank 0 with the embedding, before
  the layers run; the only thing later stages need is the multimodal RoPE positions, three rows per
  column, which travel in the control block (sized for them: eight int32 per boundary column). The
  default split counts a resident tower on the head stage. Overlay residency stays refused at load,
  before the weights: it borrows its encode window from the evict-ranked weights, and a later stage
  holds whole layers that nothing could lend.
- **MTP works** across stages: the replay records and their fold are per state shard, on the shard's
  own device, and the draft layer and head stay on rank 0. **DFlash and DFlash2 work** too; see
  below. **The context cache works**: its transactions copy each rank's planes and state shards on
  that rank's streams.

### DFlash across stages

A masked drafter reads target features, the outputs of a few layers spread through the model:
Qwen3.8-27B's DFlash2 adapter taps layers 5, 19, 33, 47 and 61 of 64 at width 5,120, the 35B-A3B
DFlash adapter layers 1, 6, 11, 16, 22, 27, 32 and 37 of 40 at width 2,048. The drafter, its context
KV and the feature buffers live on rank 0 with the head, so a tapped layer on a later stage sends its
output there, the way the residual travels.

1. **A link per tapped layer on a later stage** (`StageRuntime::features`), from that stage to rank
   0, built beside `back` in `make_stage_runtime`: a `StageLink` of two slots, each holding the
   widest pass (`stage_boundary_columns`) in BF16, plus a stream of the stage's own and the events
   that hand work to it and back.
2. **Capture on the stage.** As a tapped layer finishes, a stage-local tap copies its output into
   that tap's slab of a staging tensor in the stage's workspace and starts the transfer on the
   feature stream, so it crosses while the stage's later layers run. The stage's stream takes the
   feature stream back after the residual has left, which keeps the staging alive until every
   copy is done.
3. **Re-capture on rank 0.** Rank 0 has nothing to do until the residual returns, so `run_staged`
   queues the receives there first: each tapped output lands in one residual's room on rank 0's
   stream and goes to the sink's own `capture_layer`, so the prefill copy, the batch scatter and the
   check that every tapped layer was published are the ones one device runs. Stage 0's taps go to the
   sink directly.
4. **Graphs and planning.** The links' send and receive halves are capture-safe and the feature
   stream joins the capture through its events, so the DFlash decode graphs capture the extra copies
   with the residual's. The workspace plan reserves the staging slabs on each later stage and one
   receive slab on rank 0, sized for the stage with the most taps, and the default split counts the
   drafter's weights on the head stage, as it counts MTP's layer.

Cost, for a 30/34 split of the 27B, which puts the taps 33, 47 and 61 on the second stage: 30 KiB per
column, so 240 KiB for an eight-column DFlash2 verify and 30 MiB for the default 1,024-token prefill
chunk, sent while the stage's later layers run; only what is still in flight when the stage's last
layer finishes adds to the pass. Each tap's two slots pin twice the widest pass in host memory, 60 MiB
for those three taps at that chunk, beside the residual links' own slots.

## Verification

- `ninfer_multi_gpu_test`, three suites in one executable (ranks share device 0, the staged protocol
  forced): the stage link (byte integrity across sizes and ring reuse, a two-hop chain, payload
  halves through separate graphs, the slot-reuse dependency read off the graph, eager use of a slot
  before and after a capture), the multi-rank KV pool and block-table copies against the single-rank
  pool as the oracle in both plane orders, and the stage solver against a brute-force partition
  enumeration.
- `ninfer_kv_capacity_test`: the per-device curve, including a device without KV and errors that
  name the device.
- `ninfer_qwen3_5_stages_real_test` (set `NINFER_TEST_ARTIFACT`): greedy output with `--devices 0,0`
  and `0,0,0` equals `--device 0` byte for byte, since the layers run the same kernels on the same
  data. Rows: graphs and eager, forced pinned-host transport, an uneven split
  (`NINFER_TEST_STAGE_LAYERS`, 20,44 by default), MTP, DFlash or DFlash2 (whichever drafter the
  artifact carries, each compared with the same backend on one device), Vision plain and with MTP
  (an image prompt spanning two prefill chunks, when the artifact has a tower), and a continuation
  that reuses the context cache. `NINFER_TEST_DEVICE_IDS=0,1` puts the stages on real cards.
- `ninfer_qwen3_5_loading_real_test`: the default split covers the model, gives the head stage
  fewer layers, and gives a device with twice the memory more.

Ranks sharing a device cannot show a wrong-device pointer: a stale pointer into "another rank's"
memory still works there. That class of bug needs a real second card.

## Measured on two real GPUs (2026-09-21)

Rented Linux boxes, CUDA 12.8 build, Qwen3.6-27B groupwise-int (the pinned Hugging Face artifact),
int8 KV. Neither box has peer access, so every boundary transfer goes through pinned host memory.

| | 2x RTX 3090 (PCIe 3.0 x16 each) | 2x RTX A4000 (x8 + x16, cross-NUMA) |
|---|---|---|
| `ninfer_qwen3_5_stages_real_test` | every row byte-identical to one device: 2 stages, 3 stages (devices 0,1,0), graphs and eager, forced staged transport, uneven split, MTP, Vision (plain and MTP, an image over two prefill chunks), prefix reuse | the model does not fit one 16 GB card, so every row is compared with the first: identical across cut points, graphs and eager, staged transport, MTP, prefix reuse |
| CLI, prefill chunk 1024, splits 20/44, 44/20, 32/32 and the default | output identical to one device | output identical across splits, and identical to the 3090 output |
| Decode, plain | 46.9 tok/s on one card, 48.5 split | 24.1 tok/s |
| Decode, MTP3 | 100.2 tok/s on one card, 105.3 split | 52.6 tok/s |
| Prefill | 1.56k tok/s on one card and split | 825 tok/s |
| Weights per card (default split) | 15.3 GiB on one card, 8.35 GiB on the first split card | 7.71 GiB |
| `--kv-capacity auto` at `--max-context 262144` | one card: refused (minimum reservation 9.2 GB, 8.76 GB free after weights); split: 262,144 tokens, 11.2 GiB free afterwards | 262,144 tokens, 3.9 GiB free afterwards |

A `ninfer-serve` check on the 3090 pair (four lanes, concurrent and repeated chat requests, prefix
cache on) gave the same output split as on one card for every request run alone, run in parallel
once, and repeated; a second concurrent round differed on one request, and one card differs from
itself the same way, because lane batching changes the arithmetic.

What this says: the split costs nothing measurable at decode on PCIe 3.0 x16 (one 20 KB residual
crossing per stage boundary per token is about 25 us against a ~21 ms step), and it is what makes the
27B's full 262,144-token context fit at all. It does not speed anything up: prefill is the same and
decode is within noise. The A4000 pair runs the 27B with its full context and its decode rate matches
the plan's estimate for a pipeline on 448 GB/s cards.

On a card with fewer SMs than the tuned route tables assume (the A4000 has 48), a GDN gating
schedule whose cooperative grid would not be resident falls through to the first later route that
fits the device; its speed there has not been tuned.

Later checks on real pairs, none with peer access: Ternary Bonsai 2 27B on two RTX 4090s generates
byte for byte what one GPU does in every row of the stage test, and suspends and resumes with its
prefix reuse intact; Qwen3.8-Flash-Next runs its experts over two RTX 3090 Ti, two RTX 4090 and two
RTX 5090 ([measurements](../qwen3-8-flash-next.md#measurements)), and its host and disk experts over
two RTX 4090s, each stage copying and caching its own layers' experts on its own GPU
([measurements](../qwen3-8-flash-next.md#host-and-disk-experts-over-two-rtx-4090s)).

### DFlash across stages, measured (2026-10-06)

Rented Linux pairs, none with peer access, so every hop goes through pinned host memory; CUDA 12.8
builds for `80`, `86` and `89`, CUDA 13.1 for `120a`; the pinned Qwen3.8-27B (DFlash2 bundle,
`1cbd84e7`) and Qwen3.6-35B-A3B (DFlash bundle, `ee449580`) artifacts; int8 KV.

`ninfer_qwen3_5_stages_real_test` with the stages on two cards (`NINFER_TEST_DEVICE_IDS=0,1`) passed
all 22 rows, 11 per artifact, on 2x RTX 3090 (sm_86), 2x RTX 4090 (sm_89), 2x RTX 5090 (sm_120a) and
2x A100 PCIe 40 GB (sm_80): two and three stages, graphs and eager, forced staged transport, an
uneven split, MTP, and DFlash2 or DFlash in two stages, staged transport, three stages eager and an
uneven split, each over a 1,300-token prompt in three chunks and a continuation served from the
context cache, byte-identical to the same backend on one card.

Speed with the CLI (`--greedy --no-thinking`, seven drafts): a 37-token chat prompt with 256 new
tokens ("short") and a 3,322-token prompt with 64 ("long"), tok/s on one card and then split over the
two by the default split, mean of two or three runs. Every split run generated the one-card text.

| | 2x RTX 3090 | 2x RTX 4090 | 2x RTX 5090 | 2x A100 PCIe |
|---|---|---|---|---|
| 27B DFlash2, short, decode | 206.1 / 223.2 | 264.8 / 264.8 | 416.4 / 418.4 | 242.5 / 233.8 |
| 27B DFlash2, long, prefill | 1,247 / 1,270 | 2,267 / 2,227 | 2,965 / 2,990 | 1,357 / 1,297 |
| 27B DFlash2, long, decode | 106.5 / 114.6 | 135.6 / 135.6 | 213.6 / 214.3 | 125.7 / 130.9 |
| 27B plain, short, decode | 47.9 / 48.9 | 55.8 / 55.5 | 91.7 / 91.0 | 52.3 / 54.7 |
| 35B-A3B DFlash, short, decode | 482.8 / 490.7 | 837.0 / 818.2 | 1,260 / 1,225 | 475.0 / 365.9 |
| 35B-A3B DFlash, long, prefill | 6,320 / 6,233 | 12,033 / 11,667 | 11,300 / 11,750 | 7,327 / 5,173 |
| 35B-A3B DFlash, long, decode | 195.8 / 205.5 | 343.0 / 337.3 | 471.7 / 462.2 | 199.2 / 197.2 |
| 35B-A3B plain, short, decode | 206.1 / 207.9 | 274.0 / 268.7 | not run | 207.6 / 181.9 |

The 3090 column was measured with taps sent together after a stage's last layer; sending each tap as
its layer finishes left decode equal within 1% on the 4090 and A100 pairs and raised the 35B-A3B's
long prefill from 11,433 to 11,667 tok/s (4090) and from 4,777 to 5,173 (A100). The weights per card in the default split were
10.3 + 7.74 GiB for the 27B with DFlash2 (18.0 GiB on one card) and 10.6 + 9.41 GiB for the 35B-A3B
(20.0 GiB); the 27B's workspace grew from 183.9 to 233.9 MiB per card for the boundary and the tap
staging.

What this says: on the GeForce pairs the split costs DFlash and DFlash2 no more than it costs plain
decoding, within about 3% either way (the 3090 pair is faster split, as it was on 2026-09-21), and
the features ride along at no measurable price at decode. The A100 pair is the exception, and not
because of DFlash: the 35B-A3B lost 12% split without speculation and 23% with DFlash. The same split
with both stages on one A100 (`--devices 0,0`, local or forced staged links) cost nothing (466-473
tok/s on one card, 476-481 split), each A100 alone ran at the same speed, and an nsys profile of the
split showed the kernels themselves running longer (the MoE path kernel +21%, the verify GDN layers
+46%) rather than gaps opening between them. That points at the cards' clocks while they wait for each
other (the two sit on different sockets of that host); GPU clock metrics are not permitted in the
container, so the cause is not confirmed.

## Tensor parallelism

`--devices A,B,... --tensor-parallel` divides every layer of a Qwen3.5-core model over the devices
instead of dealing layers out. It targets decode: each card reads only its share of the weights for
every token. Qwen3.8-Flash-Next divides its layers too ([tensor parallelism of
Qwen3.8-Flash-Next](#tensor-parallelism-of-qwen38-flash-next)); Gemma 4, Flint and
DeepSeek-V4.1-Flash divide only their routed experts ([expert parallelism of the other MoE
families](#expert-parallelism-of-the-other-moe-families)).

- **The shares.** Rank r holds a contiguous range of GDN key heads with the value heads that read
  them (the value-to-key head map is contiguous), their q/k/v/z projection rows, convolution
  channels, recurrent and convolution state, and the matching input columns of the GDN output
  projection; and a contiguous range of the dense FFN's gate and up rows with those input columns
  of down. The ranges differ in size when the cards do. The gating projection, `A_log`, `dt_bias`,
  the gated norm and the layer norms are small, so every rank holds all of them: it gates every
  head and keeps its own. Attention divides by KV heads: a rank holds a contiguous range of them with
  the query heads that read them (their query, gate, key and value projection rows and the matching
  input columns of the output projection, the per-dimension query and key norms whole) and those
  heads' planes of every layer's KV cache. The ranks holding KV heads are a prefix of the ranks;
  with more ranks than the heads allow, the others hold no attention, unless `--tensor-attention
  positions` divides it by cache pages ([attention by positions](#attention-by-positions)). The
  embedding, the final norm and MTP's stem stay on rank 0, and so do the whole output and proposal
  heads; every rank also holds a row range of each head, which decode projects divided, and its
  share of MTP's attention with its KV and its rows of MTP's dense FFN in place of rank 0's whole
  ones (below).
  `TensorShare` and `tensor_parallel_share`
  (`models/qwen3_5/config.h`) state a rank's ranges and geometry.
- **Expert parallelism.** In a MoE model a rank holds a contiguous range of the routed experts in
  place of FFN rows: one device selection of each expert bank, its experts side by side as the bank
  stores them (`Bindings::joint_piece`). Every rank holds the router (rank 0 its own copy, the
  others a selection of the router object) and routes every token itself; rank 0 holds the shared
  expert. `ops::sparse_moe_share` runs the decode, Small-T and prefill routes of SparseMoe over the
  share: a selected expert of another rank contributes nothing, and the rank writes the FP32 sum of
  its paths, the shared one on rank 0, as its partial for the FFN exchange. The adaptive prefill
  band (47 to 52 tokens) takes the plain prefill route under a share. The experts divide at any
  expert; the automatic split counts the experts as divided weights, the router as held by every
  rank and the shared expert as rank 0's. Rank 0 runs the shared expert beside its routed ones,
  so it takes fewer of them (`shares_beside_first`): a token's routed work is
  `num_experts_per_tok * moe_intermediate_size` rows, the shared expert adds its own rows to rank
  0's, and the expert ranges make every rank's rows over its weight the same (Qwen3.6-35B-A3B on
  four even ranks: 40, 72, 72 and 72 of the 256 experts; on four RTX 5060 Ti it decodes at 179.2
  and 174.8-175.3 tok/s after a 989- and a 6,768-token prompt against 177.7-178.0 and 173.9-174.3
  with 64 each, prefills 0.2% to 0.8% faster and leaves rank 0 1.67 GiB more free memory, int8 KV,
  two runs each, 2026-11-04). Experts in host memory or on disk
  (`--expert-residency host|disk`) and NVFP4 expert banks have no share and are refused.
- **Divided expert columns.** Two ranks of about equal weight (each 40% to 60% of the split)
  divide every expert's 512 intermediate columns in halves instead, routed and shared alike, where
  the banks cut there (row-split groups do): each rank holds every expert's 256 gate rows and 256
  up rows of its half, one device selection of the gate/up bank taking those row ranges of each
  expert (`ObjectSelection::rows`, `Bindings::joint_piece` with a row cut), and those 256 columns of
  every down projection (a column cut). Each rank runs every selected path at width 256
  (`SparseMoeShare::intermediate`) and the FFN exchange adds the two halves, so a token's routed
  work divides evenly whichever experts it selects (with whole experts halved between two ranks,
  uniform routes would give one rank five or more of a token's eight 73% of the time, and its
  shared expert to rank 0 besides). A SwiGLU column is the whole expert's; only the
  association of the down product's FP32 sum changes. Uneven weights and more ranks keep whole
  experts.
- **Where the shares end** (`plan_tensor_shares`, `models/qwen3_5/load/text.cpp`). A share is a
  piece of a stored object: a row range and column ranges (`artifact/selection.h`), gathered from
  the file as byte runs at upload, so its bounds must fall on whole stored blocks (columns: one for
  a contiguous or row-scaled matrix, the group for ggml blocks, 128 for row-split groups, 64 for
  NVFP4; rows: NVFP4's 128-row scale tiles and its divisor rows), on whole tiles of the share's
  Linear route (`ops::linear_share_units`: T2 32 rows and 512 columns, FP8 64 and 512, NVFP4 128 and
  256) and, for a Hadamard-rotated input, on whole 1024-column blocks of the rotation, and a
  projection whose stored columns are gathered (GSQ-RCO) must not cut a block the gather reads.
  `Bindings::column_ends` and
  `Bindings::row_ends` list those positions per tensor; the head ends every layer's GDN projections
  allow and the row ends every FFN allows are the candidates, and `split_shares` (`core/tensor_split.h`)
  places each rank's end at the candidate nearest its weight's cumulative target. The KV heads end
  where every attention layer's projections allow (a rotated output projection: an even KV head of
  four, whose 1,536-column groups end on 1,024-column blocks only in pairs), split over the first
  ranks, as many as those ends give a share, by the same weights. The stored objects themselves get
  no device demand.
- **The weights.** `--tensor-split A,B,...` gives them, one per device. Without it the engine
  measures each card's device-to-device copy bandwidth (`rank_copy_bandwidth`, 128 MiB; ranks on one
  card divide its figure; cards of one model, by name, memory, multiprocessors and capability, take
  the mean of theirs, since a share that followed the noise between them moved from start to start
  and changed the partial sums the ranks add) as its speed, works out how much of the divided weights, GDN state and KV
  cache for the configured context and concurrency each card can hold beside what every rank holds
  and, on rank 0, the embedding, the head and, as selected, MTP's layer, the proposal head, a
  DFlash drafter and a resident Vision tower, the heads' row ranges and MTP's attention with its KV
  and dense FFN counting with the divided weights (`default_tensor_split`, `models/qwen3_5/load.cpp`), and fills the cards in proportion to speed, capping a card at its room
  and passing the excess to the others (`fill_shares`). Automatic KV capacity also keeps its
  headroom free on every card, and when the longest context fits nowhere it divides the cards by
  room instead of speed, so the KV it shrinks to is the most that fits. What the Engine reserves
  beside the weights (workspace, round buffers, logits, CUDA Graphs, the exchange) is known only to
  the loaded model's planner, so the first load does not count it. After that load each card's
  spare is its free memory less the planner's reservation for the KV asked for (the explicit
  capacity, or the longest context times the concurrency under automatic capacity, with its
  headroom; `runtime::kv_capacity_spare`). When a card falls short, or the first split capped a
  card that has room, the model is loaded again with `refit_tensor_split`'s shares: each card may
  take its share plus its spare, in bytes of the divided work per share, the shares follow speed
  within that room, and a card's room shrinks until the nearest split the weights allow fits it
  (plan_load sizes each candidate's divided weights, KV and GDN state). When the cards together
  lack room for that KV, automatic capacity shrinks it until the card short of the largest part of
  its own KV fits, so the refit shares the shortfall out in proportion to each card's KV, searching
  from the best split measured toward the even shortfall (a share moves whole units, and past a
  head's end a whole head of every layer); it loads again only for at least a page more KV and only
  when every card then holds the longest context. When the cards lack room even for the smallest
  reservation, no split starts and the model is not loaded again. The startup log names the spares
  and the splits, also when a card falls short and the split stays. A second refit corrects the
  first; there is no third. Decode reads every weight once per token, so bandwidth sets each rank's
  time.
- **The pass** (`TextContext::run_tensor_parallel`). Rank 0 sends the residual and the control
  tensors to every rank over the stage links; each rank then keeps a replica of the residual. Each
  layer has two exchanges (`core/tensor_parallel.h`): after the mixer (attention: the ranks holding
  KV heads, each over its own planes and block-table copy; GDN: every rank) and after the FFN. A
  contributor writes its output projection as FP32 partials; every rank then adds all partials to
  its replica in rank order and rounds once to BF16, so the replicas stay identical. All of it
  captures into the decode graphs.
- **Divided heads** (`TextContext::project_head`). Every rank holds a contiguous row range of the
  output head and of the proposal head, in proportion to the split and ending where the stored
  blocks and the head's Linear route allow (`bind_head_shares`, `models/qwen3_5/load/text.cpp`).
  Decode, verification and MTP drafts project the head divided: rank 0 sends the final hidden state
  over the forward links, every rank projects its rows at once, each other rank returns its logits
  over a link of its own (`StageRuntime::TensorParallel::heads`), and rank 0 places every range in
  its rows of the logits. A logit is still one dot product over the whole hidden state, so the
  logits are one device's wherever the share takes the whole head's route; prefill, offline scoring
  and the heads that `--lm-head-q4` and `--lm-head-q6` transcode (a piece cannot be) project whole on
  rank 0, which keeps the whole heads. Without the split rank 0 projected the heads alone while the
  other cards waited: on four RTX A4000s the Q8 output head over a round's four columns took 4.0 ms
  and the three Q4 proposal-head drafts 3.1 ms of a 39.4 ms MTP round. Divided, Qwen3.8-27B
  (int8 KV, three MTP drafts through the proposal head, one request, even split) decodes a round in
  31.1 ms against 34.5 and a token without drafts in 16.2 ms against 18.4 on four RTX 5060 Ti
  (PCIe 4.0 x8, host transport), and a round in 34.4 ms against 39.4 on the four A4000s (capped at
  110 W), with the same text and accepted drafts; prefill is unchanged. The ranges cost every card
  their bytes (0.40 GiB on each of four cards for Qwen3.8-27B with its proposal head).
- **Divided MTP layer** (`TextContext::mtp_divided_layer`). Every rank holding KV heads holds its
  share of MTP's attention, cut as a layer's (`bind_attention_shares`), with those heads' planes of
  the MTP cache; every rank holds its rows of MTP's dense FFN, cut where a layer's FFN is
  (`bind_dense_shares`). Rank 0 holds neither whole, only the whole K and V rows besides. In decode
  each MTP forward sends the stem's output and the positions, rope positions, table rows and valid
  columns over the control and forward links; each rank attends with its heads and projects its
  query heads' columns of the output into an FP32 partial, and one exchange stores their sum,
  rounded to BF16 once, in every rank's buffer, which each rank adds to its copy of the residual as
  one device adds its output projection. Each rank then projects its FFN rows from its own copy and
  a second exchange sums them for rank 0. A prompt chunk's keys and values come from rank 0, which
  projects them whole and sends each rank its heads' (1 KiB a token for Qwen3.8-27B on four
  cards, against 10 KiB of hidden state), and
  only a final chunk's last column runs the rest of the layer divided. Drafts only propose, so the
  round's text stays the target's. The exchange slots alternate across passes and these calls
  (`StageRuntime::TensorParallel::next_exchange`), since the drafts follow the verification in one
  capture. Position-split attention, or stored runs that do not cut at the shares, keep the
  attention whole on rank 0 (`TextContext::mtp_ffn` then divides the FFN alone, sending the
  post-attention hidden state); a MoE FFN stays on rank 0. On four RTX 5060 Ti (Qwen3.8-27B, Q8 MTP
  layer, int8 KV, three drafts, even split) dividing the FFN took a round from 24.9-25.0 ms to
  23.6-23.9, and dividing the attention as well from 22.5-22.6 ms (after the scatter-flag
  exchange) to 22.1-22.2, with the same accepted drafts and text and the same prefill; on two RTX
  3090 a round takes the same time (21.0-21.7 ms against 21.1-21.8, the same text). Against the
  whole layer on rank 0, rank 0 holds 0.31 GiB less and each other card 0.11 GiB more.
- **Transports and forms.** An exchange moves the partials one of three ways: through mapped pinned
  host memory (`host`: the copy engines write each partial there, every rank's kernels read the
  others' copies; works between any cards), by reading the other cards' memory (`peer-read`), or by
  writing into it (`peer-write`: a rank stores what another rank sums into that rank's memory, so
  every kernel reads only its own card). The two peer transports need peer access between every pair
  of cards. Each moves an exchange in one of four forms. Words, for partials of up to 32 KiB (a
  decode step or a short verification) with every rank on its own card: each rank's reduce kernel
  stores its FP32 values with the exchange's epoch as 8-byte words, into its own mapped host row or
  into its row in every other card's memory, and polls the others' words, with no copy and no
  cross-device fence; each rank counts its exchanges in device memory, so a graph replays them
  without new arguments, and a wait longer than five minutes traps. A whole reduce: every rank sums
  every value. A reduce-scatter and all-gather: each rank sums the partials only over its own range
  of the values, whole 64-value blocks, and the other ranks' finished BF16 ranges then reach its
  replica; every value is still the same rank-order sum, and a rank moves 1/N of each other rank's
  partial and (N-1)/N of a replica instead of N-1 whole partials. Through host memory a publish
  copies only the ranges the other ranks read, and an owner copies its finished range there; by peer
  writes the owner stores its finished range straight into the other replicas. Scatter flags,
  through host memory with every rank on its own card and for partials of up to 1 MiB: the
  reduce-scatter's publish copies as above, but one kernel per rank (two blocks) sums its range,
  stores the finished values into its own mapped host row and reads the others', waiting on flags in
  host memory that each rank's blocks raise once their copies and stores have reached it, instead of
  on fences between the cards; a flag carries the exchange's epoch from the words' count. On four
  RTX 5060 Ti an exchange of a four-column verification (80 KiB) took 43.0 us against 51.2 for the
  reduce-scatter with fences, and one of 640 KiB 203 us against 220; a decode round of Qwen3.8-27B
  with three drafts took 22.4-22.6 ms against 23.6-23.8, with the same text.
- **Choosing a route.** Which route is fastest depends on the partial's size, the number of cards
  and the platform. On a 4x RTX 5060 Ti box with the community peer-access driver host memory beat
  peer reads (#170); on a 4x RTX A4000 box peer reads beat host memory, on four of its cards peer
  writes beat both at every size, and a whole reduce by peer reads took more than six times the
  reduce-scatter's time at 40 MiB (measurements below). At startup the exchange times every route
  the cards allow, in captured graphs on every rank at once, at partial sizes from 4 KiB doubling up
  to its slot and at the last size of each route that stops short of the slot (words, scatter flags,
  a peer-write whole reduce into its inbox rows), and frees the buffers of the routes it did not
  choose. Between two measured sizes a
  route's time is read off the line through its two times, at sixteenths of the step, since one
  octave apart a crossover could be off by half the size; an exchange takes the fastest route at the
  smallest of all these sizes that holds it. `--tensor-parallel-transport host|peer-read|peer-write`
  keeps every exchange to one transport (its forms still measured), except that under peer-write a
  two-rank FP32 sum too large for an inbox row reduces whole by peer reads; a peer transport
  without peer access refuses to start. The startup log names the pairs without peer access
  (`peer access: none between CUDA devices 0-1; ...`), or says every pair reads peer memory, and
  then the routes chosen
  with their times (`tensor-parallel exchange, 4 ranks, measured at startup: peer-write words to 32
  KiB (14.0 us), peer-write whole to 100 KiB (29.1 us), peer-write scatter to 40 MiB (3837.0 us)`).
  `tools/tp_exchange_bench.cu` prints what the exchange measured at startup and every route at every
  size beside the chosen ones.
- **Prompt pieces.** A text prompt chunk of at least 768 tokens of a model with dense FFNs
  exchanges each step in two to four pieces of whole 64-column blocks, each at least 384 columns
  wide (`run_tensor_parallel`; Pieces in `core/tensor_parallel.h`). Every rank computes a piece on
  its own stream and hands it to a side stream of its own, which publishes, reduces and completes
  that piece's exchange while the rank's stream computes the next piece; the rank's next step on a
  piece waits for that piece's exchange. A piece attends over the keys up to its own last position
  and carries the Gated DeltaNet state to the next piece, so it computes what a prompt chunk of its
  width computes: in pieces of 384, 384 and 221 columns a 989-token prompt gave the greedy text of
  `--prefill-chunk 384` without pieces, character for character. Verification, DFlash taps,
  position-split attention, sequence batches, batched prompt segments, image and video prompts
  and steering exchange each step whole, and so do MoE models, whose pieces would each
  read most expert weights (Qwen3.6-35B-A3B prefilled a 7,000-token prompt at 5,300 tok/s in four
  pieces against 6,700 in one on two RTX 3090s). Narrower pieces lose more to their projections'
  narrower GEMMs than the overlap gains: in pieces of 256 columns the four A4000s below prefilled a
  6,768-token prompt at 1,402 tok/s against 1,478 whole, and two Tesla V100s prefilled the dense
  Qwen3.8-27B GSQ-RCO IQ3_S at 657 tok/s in four pieces of its chunk against 807 whole (host
  transport). With pieces of at least 384 columns Qwen3.8-27B (int8 KV, even split, one request)
  prefills a 6,773-token prompt at 1,506 tok/s against 1,194 and a 994-token one at 1,361 against
  1,087 on four RTX 5060 Ti (host transport), and at 1,615 against 1,492 and 1,510 against 1,330 on
  four RTX A4000s (peer-write, capped at 110 W). A pieces build's text can differ from an unpieced
  one's as one prompt-chunk width's differs from another's.
- **Rounded prompt partials** (`--tensor-parallel-prompt-partials bf16|int8`; Rounded partials
  in `core/tensor_parallel.h`). Where transfers bound prefill, a prompt chunk's exchanges can move
  each rank's partial rounded: to BF16, half the bytes, or to 8-bit blocks of 32 values with an
  FP32 scale, 9/32 of them. A publish rounds the partial into a device copy and moves the copy as
  its transport moves partials (host slots, peer-write inbox rows, or the copy the other ranks read
  by peer), and a reduce adds the rounded partials in FP32 in rank order, so the replicas stay
  identical. Whole reduces and reduce-scatters round on every transport, for Qwen3.5-core models
  and Qwen3.8-Flash-Next; decode, verification, the MTP layer, words and scatter flags keep FP32.
  Rounded partials move fewer bytes, so the startup measurement also times the whole and scatter
  routes with partials rounded as asked, and a rounded exchange takes its route from those times
  (the startup log lists them after `bf16 partials:` or `int8 partials:`). The rounding adds one
  rounding of each share's output to every projection a prompt runs, so prompt outputs and
  perplexity differ slightly from FP32 partials'. On four RTX 5060 Ti on PCIe 3.0 x8 (host
  transport, int8 KV, plain decode, two runs each) Qwen3.8-27B prefilled a 989-token prompt at
  1,431-1,434 tok/s with FP32 partials, 1,685-1,687 with BF16 and 1,483-1,485 with 8-bit, and a
  6,768-token one at 1,587-1,597, 1,936-1,957 and 1,664-1,666; Qwen3.8-Flash-Next Q2_0 at
  1,659-1,664, 1,829-1,865 and 1,875-1,899, and 1,635-1,644, 1,803-1,843 and 1,862-1,880, with
  the same decode speed. With MTP and an even split BF16 partials prefilled a 6,773-token prompt
  at 1,472 tok/s against 1,167-1,169 and a 994-token one at 1,370-1,372 against 1,106-1,109, with
  the same accepted drafts and greedy texts. Perplexity over the 40 KB slice (Qwen3.8-27B, 2,048-token
  disjoint windows, two RTX 3090s, host transport) is 7.0952 with FP32 partials, 7.0943 with BF16
  and 7.0981 with 8-bit, scored at 1,107, 1,223 and 1,321 tok/s; over its first 16 KB on the four
  5060 Ti 5.0702, 5.0686 and 5.0684, at 1,060, 1,206 and 1,100 tok/s; Qwen3.8-Flash-Next over the
  slice (int8 KV, 4,096-token windows) on the two 3090s 6.4854, 6.4937 and 6.4638, at 1,001-1,005,
  1,150 and 1,246 tok/s. A prompt piece's rounded partials take their rounded bytes in a
  peer-write inbox row, so the later pieces of a chunk fit where their FP32 values would not and
  reduce on the route the measurement chose (on two RTX A4000s 8-bit partials of a piece's size
  reduce whole by peer writes). Where the peers write each other's memory the transfers
  do not bound prefill and rounding does not pay: on two RTX A4000s (peer-write, 140 W, plain
  decode, two runs each) Qwen3.8-27B prefilled a 1,006-token prompt at 1,014-1,021 tok/s with FP32
  partials, 1,009-1,010 with BF16 and 1,005-1,006 with 8-bit, and a 6,786-token one at
  1,076-1,081, 1,068-1,069 and 1,066; a 706-token prompt, one chunk without pieces, at 870-897,
  896-899 and 900-904.
- **Numerics.** Against one device the order of the FP32 additions of a projection's output
  differs: the shares' partial sums are added, then rounded once, where one device rounds the full
  sum. On `120a` NVFP4 and FP8 shares quantize their activations for W4A4 and W8A8 routes (under
  Coverage). An NVFP4 share takes one device's 16-value groups and static input divisor and divides
  on every tile, so its codes do not depend on the tile: they are one device's where one device runs
  its TMA route (from 256 to 1,024 tokens, by shape), and below that, where one device's MMA route
  multiplies by a reciprocal, a value at an exact tie can differ by one code. An FP8 share of an
  output or down projection scales each token over its own columns, a finer scale than one device's
  over the whole row. Logits are close, not identical, so greedy text can diverge where two tokens
  tie closely.
- **Coverage.** Every weight format of a dense Qwen3.5-core model: GGUF, BF16, row-split
  Q4/Q5/Q6/Q8 and T2, FP8 and NVFP4, Hadamard-rotated or not; any split the stored blocks allow on
  two to eight ranks, cards of different models where the build runs on each; MTP, the masked
  drafters (DFlash, DFlash2: rank 0's replica after each layer is the residual one device taps),
  grafted prefixes (each rank appends its KV heads) and Vision (on rank 0). A share's projections have no tuned shapes, so they take Linear's
  runtime-shape routes (the shape tables' kernel templates with the rows and K read from the
  operands; up to 8 tokens, Q4 and Q5 shares whose K is whole 256- or 512-column slabs run the
  small-T MMA of the full shapes, `bench/ops/tp_share_linear_bench.cu` comparing it with the other
  templates at a share's shape; on `120a` NVFP4 shares that allow A4 run W4A4 from 9 tokens and FP8 shares that allow A8
  run W8A8 from 17 tokens, A16 below, on the tile whose CTAs fill the fewest weighted waves of the
  card's multiprocessors): the GDN input projection as parts,
  one Linear per piece into the q/k/v and z planes; gate and up as two FP32 products that SwiGLU
  rounds once; the output and down projections as FP32 partials. A rotated share turns its whole
  input, or its own 1024-column blocks of it, with its slice of the sign vector
  (`WeightUse::hadamard_begin`) in its own device's copy (`AuxiliaryReplicas`), so a rank reads no
  other card's memory. The native MoE models take expert parallelism (above). Refused:
  `--stage-layers` and `--vision-residency overlay` (as for pipeline stages). Ranks may share a
  device (`--devices 0,0`), which is how the path is tested on one card.
- **Kernels at the share shapes.** The convolution split, the projected convolution and the replay
  record and fold keep the whole-layer geometries as template constants and take any other share's
  rows and heads at run time (`CausalConvSplitGeometry`, `ProjectedConvRows`, `FoldGeometryDynamic`);
  the op tests qualify halves, quarters and uneven shares (three key heads with nine value heads,
  five with ten). Causal attention keeps the query heads per KV head of its two model geometries
  (24 query heads over 4 KV heads, 16 over 2) as template constants and reads how many KV heads a
  call holds from its launch grid, so a share (18/3, 12/2 and 6/1; 8/1) runs the model's kernels
  without instantiations of its own and takes the routes of its model's device-profile entry
  (`ModelQHeads`, `geometry.cuh`); KV append takes one to four heads. The op tests
  (`--tensor-shares-only`) qualify the four shares against the oracle at decode, verification and
  prompt widths, single and batched, over every storage.

Measured on Qwen3.8-27B GSQ-RCO IQ3_S (2026-10-25, `scripts/pods/serve_ab.py`, greedy, four short
prompts of 256 tokens, two passes and two server starts per configuration, plus one 7,014-token
prompt; attention then ran on rank 0 alone). The answers were compared between a configuration's
repeats, not with one device's.

- **2x RTX 3090** (Vast, PCIe 4.0 x8 links, no peer access, so the mapped host transport). Decode
  72.9-75.2 tok/s against 57.0-57.3 on one card (+27% to +31%) and 58.4-58.5 with two pipeline
  stages; 70.2 against 54.2 after the long prompt (+29%; the second server start, as the first
  overlapped a compile job). With `--spec mtp --draft-tokens 3` 105.5-160.3 against 89.0-134.6
  (+13% to +28%; between server starts the rate moved by up to 9%, and MTP acceptance shifts by a
  few drafts per answer because the hidden states round differently). Prefill pays for it: the
  7,014-token prompt runs at 1,187-1,292 tok/s against 1,495-1,643 (-21%), and a 40-token prompt's
  TTFT is 83-116 ms against 53-57 ms, since every layer of a chunk crosses the bus twice. Four
  ranks on the two cards (`--devices 0,1,0,1`) decode at 71-72 tok/s, and at 80.5-116.3 with MTP,
  below one card's 89-134: a check of the four-rank shares, not a four-card measurement.
- **Uneven shares on the same kind of pair** (Vast 2x RTX 3090, mapped host transport, 2026-10-26;
  the same four prompts and a 7,054-token one, two passes per configuration). One card 56.5-57.3
  tok/s. The automatic split put 6.35 GiB on rank 0 (which then also held attention and the head) and
  4.64 GiB on rank 1 and decoded at 72.7-75.6, as the equal `--tensor-split 1,1` (70.8-75.4).
  `--tensor-split 3,1` decodes at 59.5-61.2 (+5% to +7%): the larger share sets the pace.
  `--devices 0,0,1` with the automatic split (two ranks on one card) decodes at 70.8-73.2, and
  prefills the long prompt at 799-904 tok/s against 1,550-1,574. One answer of five matched one
  card's under every split; the others part from it after 17 to 470 characters.
- **2x Tesla V100-SXM2-32GB** (Vast, NVLink, so peer reads; the sm_70 routes). Decode 60.2-60.4
  tok/s against 43.9 on one card (+37%); through the mapped host transport 54.6-55.0 (+25%). With MTP 83.4-122.3 against 58.1-87.1 (+29% to +44%). Over
  NVLink prefill gains as well: the 7,014-token prompt runs at 1,300 tok/s against 865-867 (+50%),
  decode after it at 58.3 against 42.9 (+36%), and a 40-token prompt's TTFT is 113-135 ms against
  129-130 ms.
- **Other weight formats on 2x RTX 3090** (Vast, PCIe 4.0 x16 on one NUMA node, no peer access, so
  the mapped host transport; 2026-10-26; the four short prompts and a 7,054-token one, one pass per
  configuration). The official Qwen3.8-27B (row-split Q4/Q5) with `--tensor-split 1,1` decodes at
  59.7-60.0 tok/s against 46.9-47.4 on one card (+26% to +27%), and at 56.6 against 45.3 after the
  long prompt (+25%). Prefill loses: the long prompt runs at 1,219 tok/s against 1,572 (-22%), and a
  short prompt's TTFT is 105-140 ms against 82-95 ms. With `--spec dflash2 --draft-tokens 7` the
  pair is slower than one card, 56.9-181.0 tok/s against 63.2-197.9 (-8% to -10%): a verify round
  is a narrow prefill. Ternary Bonsai 2 27B (T2) does not gain: 87.9-88.8 tok/s against 90.1-91.0
  (-2% to -3%), 119.5-330.2 against 131.1-356.6 with DFlash2 (-4% to -11%), and the long prompt
  prefills at 1,373 tok/s against 2,737 (-50%). One card runs Bonsai's projections on the A8
  integer route of their tuned shapes, where a share takes the A16 runtime-shape route. The 27B
  pair gave one card's answer on one prompt of five and Bonsai's pair on three of five; the others
  part after 260-510 characters into different, coherent text.
- **Native MoE expert parallelism on 2x RTX 3090** (the same pair, 2026-10-26; the four short
  prompts and a 7,054-token one, one pass). Qwen3.6-35B-A3B fits one card, which decodes faster:
  184.3-185.8 tok/s against 201.7-202.2 (-8%), 173.4 against 186.8 after the long prompt (-7%). With
  `--spec mtp --draft-tokens 3` the pair gains 4% to 13% on the short prompts (278.5-375.1 against
  261.1-360.0) and loses 3% after the long one (223.8 against 230.8). The long prompt prefills at
  5,280 tok/s against 7,424 (-29%), and a short prompt's TTFT is 44.8-66.8 ms against 38.6-43.7 ms.
  Perplexity over 7,041 tokens of `docs/maintainer/engine-architecture.md` (2,048-token windows,
  stride 1,024, int8 KV) is 27.138 with `--tensor-split 1,1` and 27.194 with `3,1` against 27.076
  (+0.23%, +0.44%). No short answer matched one card's (one of five with MTP); they part after 25 to
  231 characters. One card's own decode is unchanged against master (201.7-202.2 against
  202.0-202.7 tok/s).
- **Attention divided by KV heads on 2x A40** (Vast, PCIe 4.0 x16; 2026-10-26; the four short
  prompts and a 7,041-token one, one pass per configuration, against one A40). Qwen3.8-27B GSQ-RCO
  IQ3_S decodes at 61.3-61.8 tok/s against 39.0-39.2 (+57% to +58%) and at 59.8 against 37.5 after
  the long prompt (+60%); `--tensor-split 3,1` at 46.3-46.6 (+19%) and `--devices 0,0,1` at
  58.7-59.1 (+50% to +51%); with `--spec mtp --draft-tokens 3` at 93.0-143.4 against 69.8-105.6
  (+30% to +43%). The long prompt prefills at 2,093 tok/s against 1,576 (+33%); a short prompt's TTFT
  is 79-84 ms against 58 ms. Qwen3.6-35B-A3B decodes at 170.4-170.9 against 159.2-159.3 (+7%) and
  164.7 against 147.8 after the long prompt (+11%), and prefills it at 8,356 tok/s against 7,064
  (+18%). Ternary Bonsai 2 27B decodes at 73.5-74.4 against 64.2-64.6 (+15%) and 71.5 against 60.1
  (+19%), but prefills at 1,789 tok/s against 2,592 (-31%) with a short TTFT of 93-107 ms against
  39-45 ms: its shares take the A16 runtime-shape route. Perplexity over the same 7,041 tokens
  (2,048-token windows, stride 1,024) is 13.958 against 13.950 for the 27B (+0.06%; +0.08% with
  `3,1`), 27.182 against 27.076 for the 35B-A3B (+0.39%) and 19.636 against 19.618 for Bonsai
  (+0.09%). The pair gave one card's answer on none of five prompts for the 27B, one for the 35B-A3B
  and three for Bonsai; the others part after 17 to 504 characters, as one card's own MTP answers
  part from its plain ones.
- **Divided expert columns and low-latency words** (2026-10-27; greedy, 256-token answers to two
  short prompts and one after a 9,705-token prompt, one pass per configuration, against master's
  tensor parallelism, which divided whole experts and exchanged by fenced copies). Qwen3.6-35B-A3B
  on 2x RTX 3090 (Vast, no peer access, so the mapped host transport) decodes at 218.1 tok/s against
  183.8 (+18.7%) and above one card's 198.5 (+9.9%); the long prompt prefills at 4,577 tok/s
  against 4,489 (+2.0%), one card 7,399. On another 3090 pair the divided columns with fenced copies
  decoded at 186.2 tok/s against 183.0, and the words took it to 202.7. On 2x Tesla V100-SXM2
  (Vast; the IOMMU translates the cards' DMA, so the preflight refuses peer access and the host
  transport runs) the pair decodes at 106.4 tok/s against 98.8 (+7.7%) and one card's 99.7
  (+6.7%), and prefills at 1,495 tok/s against 1,440 (+3.8%), one card 1,199. Perplexity over 40 KB
  of PG-19 (2,048-token windows, stride 1,024, int8 KV) is 6.3987 against one card's 6.4024 on the
  3090s and 6.3978 against 6.4006 (master's pair 6.3991) on the V100s. One card's own decode and
  prefill stay master's: 198.5 against 198.4 tok/s on a 3090, and on a V100 over the nine KV
  formats -0.6% to +0.4% decode and -0.5% to 0% prefill.
  Dense Qwen3.8-27B GSQ-RCO IQ3_S on another V100 pair (host transport for both builds) decodes
  at 56.1 tok/s against master's 54.1 (+3.7%, the words) and prefills the long prompt at 667 tok/s
  against 666. The column cuts load slower: the 35B-A3B pair's weights (millions of lines of a few
  hundred bytes) take 57 s on the 3090s and 61 s on the V100s, where master's whole experts load in
  7.3 s and 4.7 s. The official Qwen3.8-27B `groupwise-int`, which master's pair could not start on
  V100s (its shares refused the panel layout), decodes on a third V100 pair at 26.0 tok/s against
  one card's 16.2 (+60%) and prefills the long prompt at 1,395 tok/s against 891 (+57%); its
  perplexity over the slice is 6.4481 against one card's 6.4471.
- **Reduce-scatter** (2026-10-29, against master's whole reduces; RunPod 4x A40, PCIe 4.0 x16, the
  fourth card on the second socket; the IOMMU translates the cards' DMA, so the preflight refuses
  peer access and the host transport runs). `tools/tp_exchange_bench.cu`, in graphs at a 5,120-wide
  residual: two ranks take 23.9 us per exchange at two columns against 29.7 for the reduce-scatter,
  32.9 against 32.5 at four, 57.3 against 42.3 at eight and 6.65 against 4.90 ms at 2,048 (a 40 MiB
  prompt chunk); four ranks 52.1 against 51.4 us at four columns, 89.1 against 66.0 at eight and
  37.7 against 11.5 ms at 2,048. Stored FP32 sums give four ranks the same crossover (88.3 against
  74.5 us at eight columns, 39.1 against 15.0 ms at 2,048), but two ranks storing FP32 sums were
  never faster scattered (1.005 to 1.75 times a whole reduce), so they reduce whole. With both ranks
  on one card, through peer reads, a whole reduce stays ahead up to 260 KiB (3.4 against 4.2 us in
  two runs) and the two tie at 320 KiB (4.4-4.8 against 4.3-4.4 us); through host memory two runs
  disagree at 260 and 320 KiB (58.1-88.6 against 61.3-81.9 us, 70.9-90.9 against 66.8-97.6 us), so
  the crossover lies within their spread. The copy engines bring in the reduce-scatter's bytes about twice as fast as its
  kernels read them from mapped memory (two ranks, 2,048 columns: 2.44 against 4.90 ms, the
  publish's copies left out). Qwen3.8-27B (`download-model.sh qwen38-27b`, `--max-context 16384`;
  greedy 256-token answers to two short prompts and one after a 9,715-token prompt, two rounds): the
  long prompt prefills at 1,334 tok/s against 1,183 on two cards (+12.8%; TTFT 7.28 s against 8.21
  s), at 1,031 against 434 on four (2.37x; 9.43 s against 22.38 s), and at 1,074 against 1,052 with
  both ranks on one card (`--devices 0,0`, peer reads; +2.0%). Four cards answer the two short
  prompts (37 and 39 tokens) after 96-104 ms against 121-131 ms. Decode stays master's within 0.5%
  (two cards 49.4-51.1 tok/s against 49.4-51.4, four 63.7-66.5 against 63.9-66.6), except one
  four-card round of a short prompt at 60.9 against 65.0-65.8. All 18 answers equal master's.
  Perplexity over 40 KB of WikiText (9,234 tokens, 4,096-token windows, stride 2,048, int8 KV)
  equals master's to the last digit: 6.22103 on two cards and on one card's two ranks, 6.22150 on
  four; it scores 721 tok/s against 642 on two cards, 592 against 578 on one card and 533 against
  300 on four. These runs reduce-scattered every partial from 256 KiB.
- **W4A4 and W8A8 shares on 2x RTX 5090** (2026-10-29, against master's A16 shares; Vast, CUDA
  13.4, no peer access, so the host transport). Qwen3.8-27B NVFP4 (`--max-context 16384`; greedy
  256-token answers to two short prompts and one after a 9,715-token prompt, two rounds): the long
  prompt prefills at 4,215-4,647 tok/s against 2,758-2,837 (+49% to +69%; TTFT 2.09-2.30 s against
  3.42-3.52 s), and at 4,260-4,559 against 3,037-3,072 with `--spec mtp --draft-tokens 3
  --lm-head-draft` (+39% to +50%); one card prefills it at 11,307-11,353 tok/s. The short prompts'
  TTFT is 44-78 ms against 63-95 ms. Decode runs no changed route (one to four tokens stay below
  both thresholds): master's pair decoded at 54.0-60.6 tok/s in both rounds, this build's at
  56.0-58.5 in one round and at 92.8-97.1 in the other, a spread whose cause is not known. With the
  fenced stage release of the TMA pipelines ([Op development](op-development.md#tma-pipeline-stages))
  three more rounds prefilled at 4,128-4,749 tok/s (4,246-4,708 with MTP), decoding at 53.7-57.0
  or at 93.4-97.6 tok/s again. Perplexity
  over 320 KB of WikiText (65,304 tokens, 4,096-token windows, stride 2,048, int8 KV; before NVFP4
  shares divided on every tile and before the fence) is 6.0801 against one card's 6.0880 and 6.0666
  with the shares on A16; over 40 KB, two ranks on one card (`--devices 0,0`) give the two cards'
  6.18009 in three runs, where without the fence three runs gave 6.1586, 6.1827 and 6.1901.

### Attention by positions

`--tensor-attention positions` divides each attention layer's KV cache by page instead of by KV
head, so every rank attends, also where the ranks outnumber the KV heads (eight ranks over four KV
heads). Up to eight ranks (`kCausalAttentionMaximumMergeRanks`). It is a choice, not the default:
on the hosts measured below it is slower than dividing by heads.

- **The shares.** Every rank holds the query, gate, key and value projections whole with their
  norms, and a run of query heads of the output projection's input columns
  (`TensorShare::query_head_begin`, `query_head_end`), placed by the same weights at the ends every
  layer's output projection allows. Its KV planes hold every KV head of the physical pages `p` with
  `p % R == r`, at local page `p / R` (`KVPlaneGeometry::page_stride`, `page_residue`); every rank
  has the same block tables. Each rank's planes are 1/R of the pool, as by heads.
- **The step.** Each rank projects, normalizes and rotates every head of the call's tokens, and
  appends the tokens whose pages it holds (the append metadata's `local_page` is -1 for the
  others). Attention runs over the rank's own pages and writes, per column and query head, the
  normalized FP32 output and its log-sum-exp, `-inf` where the rank holds none of the column's keys:
  257 floats per head and column (`causal_attention_partial_elements`). Decode, verification and a
  short prefill take the small-T routes, where a split over another rank's pages is empty. A prompt
  chunk of one sequence takes the m8n8k4 prompt kernel written for sm_70 (`sm70/prompt.cu`; the
  route emits m8n8k4 on every architecture), which skips the 32-key tiles the rank does not hold.
- **The merge.** The partial goes into the mixer exchange's slot
  (`TensorParallelExchange::publish_whole`). The exchange's next step gathers every rank's partial
  (`gather`) and merges the rank's run of query heads (`causal_attention_merge_partials`: weights
  `exp(lse - max)`, sums in rank order, the gate applied after a BF16 rounding as on one device);
  the rank then runs its share of the output projection as an FP32 partial, which reduces as by
  heads. A layer's attention thus crosses the bus twice per step, by heads once, and every rank
  projects every head.
- **Page copies.** A copy between two pages that different ranks hold (a forked prefix's
  copy-on-write) is a peer copy on the source rank's stream (`DeviceKVPagePool`). The host KV tier
  keeps one record per page, whichever rank holds it.
- **Numerics.** The merge rescales each rank's FP32 output by its log-sum-exp, where one device
  runs one online softmax over every key: the same mathematics, rounded differently. The op tests
  (`--position-split-only`) merge two and three ranks over every storage at decode, verification,
  short-prefill and prompt widths, a rank without keys included, against the FP64 oracle, and check
  that a merge of a run of query heads is those heads of the whole merge, byte for byte.
- **Measured** (2026-10-27, the workload of [the measurements above](#tensor-parallelism),
  Qwen3.6-35B-A3B). On two cards every rank holds a KV head either way, and by positions every rank
  also projects every head and attention crosses the bus twice: 207.5 tok/s against 218.1 by heads
  on 2x RTX 3090, with the long prompt at 2,944 tok/s against 4,577; 100.6 against 106.4 and 1,419
  against 1,495 on 2x Tesla V100-SXM2. Four ranks on the two cards (`--devices 0,1,0,1`) decode at
  155.5 tok/s against 171.3 by heads (where two ranks hold no attention) on the 3090s and at 79.2
  against master's 92.8 on the V100s, and prefill at 1,329 against 1,732 and 800 against 948.
  Perplexity over the 40 KB slice is 6.3960 against 6.3987 by heads (3090) and 6.3970 against
  6.3978 (V100). A host with more cards than KV heads, where positions let every card attend, was
  not measured.

### Tensor parallelism of Qwen3.8-Flash-Next

`--tensor-parallel` divides every layer of Qwen3.8-Flash-Next over the devices. Each rank keeps a
replica of the FP32 hyper-connection stack (four streams of the hidden width per token) and runs
every layer over it (`Executor::run_tensor_segment`, `models/qwen4_exp/executor.cpp`).

- **The shares** (`TensorShare`, `plan_tensor_shares`, `models/qwen4_exp/model.cpp`). A Gated
  DeltaNet layer divides by key heads: rank r holds a contiguous range of them with the three value
  heads per key head that read them, their query, key, value and z projection rows, convolution
  channels, `a`/`b` projection rows and `A_log`/`dt_bias` entries, recurrent and convolution state,
  and the matching input columns of the output projection; the gated norm is whole. A
  sparse-attention layer divides by its two KV heads: each of the first one or two ranks holds one
  with its twelve query heads (query, gate, key and value rows, the output projection's input
  columns, that head's KV pages); the indexer, with its projection, pooled keys and tail, is whole
  on each of those ranks, and they select the same blocks. The routed experts divide as whole
  experts, a contiguous range per rank of the GGUF banks on the devices. The shared expert divides
  by its 640-wide middle: each rank holds a run of it, those rows of its gate and up and those
  values of every down row, in whole 128-row tiles of the expert matrix kernel where every rank
  gets one. Its runs even out each rank's expert rows per token (`shared_runs`): a rank's routed
  experts give it 10 x 640 rows per token in proportion to their count, and the shared expert's
  640 fill up from the rank with the fewest, so a rank 0 whose room caps its experts (241 of 512
  on two RTX 3090s with MTP) takes four of the five tiles. The runs follow the expert ranges, not
  the split's weights: each rank rounds the dot of its run on its own, so the sum depends on the
  runs, and weights measured at startup move between starts (two ranks at an even split hold two
  and a half of the five tiles, which then rounded to three or two from one start to the next and
  changed greedy texts); a run moves only when the experts move by about five. The startup log
  lists every rank's runs and head rows. Every rank holds the token embedding, the hyper-connection mixers, the final mixer, the
  router, the shared expert's score and the PLE, and a contiguous range of the head's rows; the
  whole head, the MTP block and Vision stay on rank 0 too. A share ends on whole stored blocks (`Bindings::row_ends`, `Bindings::column_ends`) and whole tiles of its
  Linear route; without `--tensor-split` the shares follow each card's copy bandwidth within its
  free memory, counting the routed experts as the divided weights. Cards of one model take one
  bandwidth, so the expert ranges and with them the sums stay the same from start to start: with a
  bandwidth of each card's own, one of 512 experts moved between two of four RTX 5060 Ti from one
  start to the next, which changed the greedy texts and perplexity over 16 KB of the 40 KB slice
  from 6.3754 to 6.3251.
- **A layer.** Every rank reads the stack (the attention mixer), runs its mixer share into an FP32
  partial of the output projection and exchanges it (`TensorParallelExchange`,
  `TensorParallelSum::StoreBf16`): every rank stores the rank-order sum rounded to BF16, as one
  device's projection stores its output. A rank without a share of a sparse-attention layer
  publishes nothing in that exchange. Every rank then writes and reads the stack (the MoE mixer),
  routes the call itself from the same input, runs the pairs routed to its experts and its run of
  the shared expert into a 2^-32 fixed-point part (`moe_experts_gguf_part`; a run narrower than the
  whole expert takes the vector products at decode widths), stores the part as an FP32 partial and
  exchanges again (`StoreFp32`), so every rank adds the same FP32 MoE output to its stack. That is
  two exchanges per layer, 96 per pass, on the routes the exchange measured at startup as
  Qwen3.5-core's are (`--tensor-parallel-transport` applies), except that the expert exchange of
  two ranks never scatters: its FP32 ranges would cross the bus with as many bytes as the whole
  partials, so it reduces whole on the fastest whole route. Each rank embeds the tokens from its
  replica of the table; a media prompt's merged embeddings are copied to every rank once the tower
  has run.
- **The head** (`run_divided_head`). After the last layer every rank reads its own stack through
  its replica of the final mixer and projects its rows of the head; each rank after the first sends
  its rows' logits to rank 0 over a stage link (pinned host slots), at most 16 MiB of the widest
  share's columns at a time, and rank 0 places every rank's rows into the logits. Each row is the
  same projection one device runs, of the same stack. A prompt pass with logits of more columns
  than one gather holds (a scoring call's, up to 512 positions) projects them with the whole head
  on rank 0, which holds it too (`run_whole_head`), in groups of as many columns as rank 0's
  workspace holds for its other work, halving from the most a pass scores: gathering the other
  ranks' rows of so many positions through host memory cost more than their share of the
  projection saved. Perplexity over the 40 KB slice on two RTX 3090s scored at 946-950 tok/s with
  the divided head, 985 with the whole head a gather's columns at a time, 1,001-1,005 in groups of
  the workspace and 1,014 with master's head on rank 0 alone, whose workspace held all of a
  pass's columns. An MTP
  draft projects the head the same way
  (`run_divided_mtp_head`): rank 0 sends its read of the block's stack to every other rank over a
  stage link, and the ranks project their rows of it.
- **Prompt pieces** (`Executor::run_tensor_pieces`). Without `--prefill-chunk` a prompt runs in
  chunks of 2,048 tokens when every device keeps room for them (the free memory after an executor
  of chunks of 1,024 covers that executor's workspace again and a margin of 384 MiB; on two RTX
  3090s the wider chunk takes 0.4 GiB more on each), else of 1,024. A text prompt chunk wider than
  1,024 tokens runs in pieces of 1,024 from its first column, the last one the rest (at most four,
  the pieces of `TensorParallelExchange`). Each exchange crosses one piece at a time on the ranks'
  side streams while their compute streams run the next piece, and a rank's next step on a piece
  first takes that piece's exchange back. A piece runs as a chunk of its own columns does: its
  mixers carry the Gated DeltaNet, convolution and indexer states and the PLE history on to the
  next piece, its sparse attention reads the keys up to its own last position, and its experts
  route and run its own tokens. A chunk therefore gives what chunks of 1,024 give, and the default
  chunk of 2,048 the tokens of the former default of 1,024. The mixer's exchange stores into a plane
  of its own, which the experts' exchange of the piece before cannot overwrite. A media prompt, a
  verification, a branch pass and a chunk exchanged in BF16 or 8-bit prompt partials
  (`--tensor-parallel-prompt-partials`) run whole: those partials cross the bus fast enough that
  with BF16 the pieces prefilled 1.6-2.0% slower than whole chunks of 2,048. Measured
  (2026-11-04, two RTX 3090s without peer access, GSQ-RCO Q2_0, BF16 KV, two runs each): a
  9,741-token prompt prefilled at 2,618-2,633 tok/s against 2,232-2,254 with whole chunks of 1,024
  and a 31,869-token prompt at 2,564-2,582 against 2,188-2,213 (+17%), with the same tokens,
  including a 256-token answer; decode runs no pieces and stayed within its spread between starts.
  Pieces of 512 in chunks of 1,024 or 2,048 prefilled 1.3-2.1% slower than whole chunks: their
  narrower projections and expert calls lost more than the overlap gained. With the other
  eight KV formats (one run of the 9,741-token prompt each) prefill rose 13-17%, with the same
  tokens.
- **Graphs and branches.** A segment forks the other ranks' streams from rank 0's and joins them
  back, so the decode and verification graphs capture every rank's work; each segment makes an
  even number of exchanges and starts in slot 0. A replay is launched on rank 0's stream and
  follows that stream's work alone, while each rank uploads its inputs of the pass (token ids,
  positions, PLE rows) on its own stream, so rank 0's stream first waits for every other rank's
  uploads (`fence_uploads`); without the wait a replay could read the previous step's inputs. With
  the wait, shared-expert runs that do not follow measured weights and one bandwidth per card
  model, four RTX 5060 Ti gave one greedy text of two prompts in three starts and one perplexity in
  five, and two RTX 3090s the same log probabilities of a prompt's first 16 tokens in five starts,
  three of them eager; with the runs by expert work, two starts of each box gave one text. A snapshot, a sequence image and a branch pass
  keep every rank's state of each layer, layer by layer and rank by rank.
- **Numerics.** The replicas run the same kernels on the same values, so on cards of one model they
  stay equal. The divided output projections, each rank's dot over its run of the shared expert's
  down rows and the FP32 sum of the expert parts round differently than one device, so logits are
  close to one device's, not identical.
- **Refused.** Host or disk experts and native expert banks (`--expert-residency device` with GGUF
  banks only), `--kv-resident` and `--stage-layers`.
- **Measured** (2026-10-27; GSQ-RCO Q2_0 with `--no-ngram-table`, the workload of [the
  measurements above](#tensor-parallelism), against two pipeline stages and master's expert
  parallelism). It is a decode trade: every layer crosses the bus twice per step, which a long
  prompt pays for. 2x RTX 3090 (mapped host transport): decode 114.7 tok/s against 107.4 with stages
  (+6.8%) and 94.0 with expert parallelism; the long prompt prefills at 1,971 tok/s against 3,162
  (-38%) and 1,997. With `--spec mtp --draft-tokens 3` 151.6 tok/s against 147.9 and 149.6, at an
  acceptance of 0.560 against 0.593 (the hidden states round differently). Over the nine KV formats
  the pair decodes 6.8% to 8.8% faster than stages. 2x Tesla V100-SXM2 (host transport): 55.2 tok/s
  against 54.5 (+1.3%) and 52.4; prefill 1,085 tok/s against 1,454 (-25%) and 956; with MTP 79.3
  against 82.5 (-3.9%) and 83.0, at an acceptance of 0.524 against 0.586; over the nine KV formats
  +1.3% to +1.5% against stages. Perplexity over the 40 KB slice is 6.8209 against stages' 6.8188
  on the 3090s and 6.8141 against 6.7975 (+0.24%) on the V100s. On the 3090s stages and expert
  parallelism give the same greedy text; the divided layers part from it early, at close ties.
  Every rank's run of the shared expert and rows of the head, against both on rank 0 (2026-11-04,
  int8 KV, a 989-token and a 6,768-token prompt, two runs each): four RTX 5060 Ti (host
  transport) decode at 107.4-107.5 and 104.9-105.1 tok/s against 98.7-98.9 and 96.7-97.0, and
  with MTP at 160.9-166.8 and 121.2-125.7 against 141.6-147.9 and 107.6-112.2; rank 0 holds
  667 MiB of workspace against 984. Two RTX 3090s decode at 117.5-117.6 and 113.4-113.7 against
  117.3-117.4 and 113.1-113.2, and with MTP at 188.7-188.8 against 176.5-176.6 on the short
  prompt but 140.3-140.6 against 142.5-142.8 on the long one (233 accepted drafts against 237).
  Prefill changes by -1.6% (the 3090s with MTP) to +1.3% (the 5060 Ti).
  Shared-expert runs by expert work against even runs (2026-11-04, same workload, two runs each):
  with MTP rank 0's room caps its experts (241 of 512 on the 3090s, 112 on the 5060 Ti) and it
  takes four or two of the five tiles. A draft round takes 17.83-17.88 and 18.34-18.36 ms against
  17.98 and 18.44-18.48 on the 3090s, 22.17-22.21 and 22.62-22.68 against 22.31-22.42 and
  22.69-22.79 on the 5060 Ti (-0.3% to -0.8%). Tokens per second follow the drafts the changed
  sums accept: 185.2-185.7 and 144.3-144.4 against 192.4 and 137.6-137.9 on the 3090s (116 and
  145 rounds against 111 and 151), 151.6-152.0 and 117.6-117.9 against 153.0-153.7 and
  115.4-115.9 on the 5060 Ti. Without MTP the experts divide evenly, the runs stay even and
  decode is unchanged; prefill moves by -0.3% to +0.3%.

### Expert parallelism of the other MoE families

Gemma 4, Flint and DeepSeek-V4.1-Flash keep every layer on rank 0 and divide only the routed
experts of each MoE block: rank r holds experts `[e_r, e_{r+1})`, a contiguous range
of `--tensor-split`'s weights (`split_shares` over single experts). Rank 0 runs attention, the
router, the shared expert, the head, the drafters and Vision, as one device does.

- **A MoE call** (`ExpertParallelExchange`, `core/expert_parallel.h`). Rank 0 routes the call, then
  sends what the experts read (the q8_1 activation of a call of up to eight tokens, or the BF16 rows
  of a wider one; DeepSeek also the guard's factors)
  with the ids and weights to every other rank over a stage link. Each rank keeps the pairs routed to
  its own experts (`moe_keep_experts`; ids of other ranks' experts become -1, which
  `moe_sort_routes` leaves out), runs them on its own stream and workspace, and sends back their
  weighted sum as a 2^-32 fixed-point part (I64 [T][H]). Rank 0 runs its own part meanwhile, then
  the block's work that reads only the block's input (Gemma 4's dense MLP, Flint's and DeepSeek's
  shared expert), adds the parts as integers and hands the sum to the block's combine: the
  exchange's round trip to the other ranks and their experts take place behind that work. The
  exchange is a pair of StageLinks
  per peer (peer copies or mapped pinned host memory), legal eagerly and inside the decode graphs,
  whose capture the peers' streams join.
- **Numerics.** Each pair's contribution is computed and rounded exactly as on one device and integer
  addition commutes, so the sum does not depend on the split. Where one device also sums in this
  fixed point the result is one device's bit for bit: the decode and verification calls of up to
  eight tokens of Gemma 4, Flint and DeepSeek. Their prompt chunks on one device weigh the
  experts' FP32 rows in FP32 in the block's combine; under expert parallelism the same rows are
  weighted into the fixed-point sum (`moe_weighted_rows`), which rounds differently.
- **Residency.** Gemma 4 and Flint divide device-resident experts: each
  rank binds its rows of every bank as a piece of its own (with the matrix kernel's zero tail) and
  builds expert tables whose other entries are null. DeepSeek-V4.1-Flash, whose experts live in
  pinned host memory or the artifact's files, gives every rank an `ExpertDelivery` of its own over
  its experts (tables of those alone, local ids from `moe_local_experts`): its own layer banks and
  device cache, filled over its own bus. Without `--tensor-split` the device families size the
  shares by each card's copy bandwidth within its free memory (`models::default_expert_split`).
  DeepSeek sizes them once the deliveries are about to start, so that every rank's misses take the
  same time over its bus (`delivery_shares`): rank r's cache holds `room_r` bytes of its share
  (its free memory beside the CUDA Graph reserve and the delivery's tables; rank 0's beside the
  model, its caches and workspace) and its bus copies the rest at `bus_r` (a pinned host copy
  measured at startup, `rank_host_copy_bandwidth`), so with `B` bytes of experts in all each rank
  takes `(room_r + t * bus_r) / B` with `t = (B - sum room) / sum bus`. Rank 0, which also holds
  the model, takes fewer experts; when the caches together hold every expert the shares follow the
  cards' copy bandwidth within that room. DeepSeek runs a prompt in spans as on one device
  ([prompt spans](../deepseek-v41.md#prompt-spans)): the span's buffers stay on rank 0, every MoE
  call keeps a chunk's width, and each rank's delivery fills a block's bank once for all of the
  span's chunks. Refused: Gemma 4's and Flint's host or disk experts, DeepSeek's device experts
  and its CPU share (`--expert-dma-share`, `--expert-cpu-threads`).

Measured on 2x RTX 3090 (Vast, PCIe 4.0 x16, no peer access, so the mapped host transport;
2026-10-26; `scripts/pods/serve_ab.py`, greedy, four short prompts of 256 tokens and a prompt of
about 7,000 tokens, one pass per configuration). Every MoE call crosses the bus twice, so on this
pair the exchange costs more decode time than the divided experts save:

- **Gemma 4** 26B-A4B QAT (fits one card), against one card: decode 164.6-166.2 tok/s against
  216.3-218.4 (-24%), 155.1 against 195.5 after the long prompt (-21%); with `--spec mtp
  --draft-tokens 3` 240.4-377.0 against 249.7-398.1 (-6% to +1%) and 199.1 against 197.1. A short
  prompt's TTFT is 44.3-49.2 ms against 45.1-51.7 ms; the 6,966-token prompt prefills at 4,858
  tok/s against 5,995 (-19%). Perplexity over 6,952 tokens of the same document (2,048-token
  windows, stride 1,024, int8 KV) is 1,653.84 against 1,647.85 (+0.36%; the instruction-tuned
  model scores raw Markdown poorly), the same for `--tensor-split 1,1` and `3,1`. One answer of five
  matched one card's; the others part after 12 to 452 characters, from prompts the pair prefilled
  with the fixed-point weighting.
- **Flint** UD-Q4_K_XL, which does not fit one 24 GB card with its experts, against one card with
  host experts (98% of decode routes from its device cache): decode 141.5-141.9 tok/s against
  147.3-150.4 (-4% to -6%), 135.1 against 142.1 after the long prompt (-5%). A short prompt's TTFT
  is 43.0-45.2 ms against 78.0-99.4 ms and the 6,932-token prompt prefills at 4,593 tok/s against
  5,042 (-9%). Perplexity over 6,927 tokens (F16 KV) is 45.397 against 45.362 (+0.08%), the same
  for `--tensor-split 1,1` and `3,1`. No answer matched the host run's; they part after 23 to 923
  characters.
- **DeepSeek-V4.1-Flash** GSQ-RCO IQ1_M (`--no-ngram-table`), on 2x A40 (Vast, PCIe 4.0 x16) against
  one A40, experts in pinned host memory: each card serves 192 experts from its own cache (33 GiB)
  and bus. Decode 41.9-42.7 tok/s against 34.0-37.9 (+12% to +24%), 39.6 against 32.0 after the
  6,940-token prompt (+24%); with `--spec dspark` 50.4-77.3 against 46.1-70.2 (+9% to +16%) and
  71.6 against 57.2 (+25%). A short prompt's TTFT is 197-277 ms against 463-657 ms and the long
  prompt prefills at 701 tok/s against 561 (+25%). Disk experts on the pair decode at 32.7-35.3
  (28.7 after the long prompt), within -10% to +1% of one card's host experts. Perplexity over
  6,940 tokens (BF16 caches) is 16.359 against 16.407 (-0.29%), the same for `--tensor-split 3,1`. The long answer matched one
  card's; the short ones part after 2 to 88 characters, from prompts the pair prefilled with the
  fixed-point weighting. Host and disk experts on the pair gave the same five answers.
- **Rank 0's dense MLP behind the exchange** (Gemma 4, 2x RTX 3090, 2026-11-04, int8 KV, a
  1,032- and a 6,906-token prompt, two runs each, against the MLP before the routing): decode
  180.6-180.9 and 175.1 tok/s against 166.0-166.5 and 161.7-161.9 (+8.2% to +9.0%), prefill
  3,041-3,048 and 3,463-3,464 against 2,920-2,927 and 3,329-3,331 (+4%), with `--spec mtp
  --draft-tokens 2` 293.2 and 246.8 against 277.8 and 235.0, with the same texts and accepted
  drafts. Fewer experts on rank 0, which also runs the attention and the dense MLP, did not pay:
  with 35% to 45% of them Gemma 4 decoded 0.1% to 1.1% slower than with half and Flint 0.3% to
  1.6% slower, so the shares stay those of the cards' bandwidth and room.

What the transport costs was measured before this was built:

- **Transport** (`tools/tp_probe.cu`, rented boxes, 2026-09-21): neither a 2x RTX 3090 box (one slot
  PCIe 3.0 x4), a second 2x RTX 3090 box with both slots x16, nor a 2x RTX A4000 box had peer access
  or NVLink. The x16 pair measured 15 us (host-mapped kernel) to 24 us (staged, in a graph) for one
  column, 33-38 us for four, 75-78 us for sixteen, 130 us for thirty-two and 3.4 ms for a 20 MB
  prefill chunk, half the x4 box's time at every size. An all-reduce of one decode column
  (20 KB, FP32) costs 17-33 us whether done by a host-mapped kernel, a staged pinned-host copy or
  NCCL; at 640 KB it is 190-440 us and at 20 MB 5-13 ms. The transports are within about 20% of each
  other at every size, which is why the exchange is a fenced copy (or a peer read) and a local
  reduce that captures into graphs.
- **Low-latency words** (`tools/tp_probe.cu`, `host-ll`, rented boxes, 2026-10-27): each value
  travels with its epoch in one 8-byte store and the receiver polls the words themselves, as
  sFNX's ring does. A 2x RTX 3090 box without peer access exchanged one 20 KB column in 12.5 us in a
  graph, against 16.8 us for the host-mapped kernel with flags and 19-25 us for the staged copy
  (eager, joint graph). A 2x Tesla V100-SXM2 box took 10.1 us against 23.3 and 31.7 us. Words
  double the bytes, so they lose from four columns on the 3090 box (53 us against 28 for the staged
  copy) and from 32 columns on the V100 box (214 us against 197); the exchange uses them up to
  32 KiB per partial. The V100 box reports peer access, but its IOMMU translates the cards' DMA
  (`iommu.strict=1`), so the preflight refuses it: there the probe's peer kernel delivered correct
  data at about 0.4 GB/s (a 20 MB exchange in 51 ms, against 3.7 ms through host memory).
- **Measured routes** (`tools/tp_exchange_bench.cu`, a rented 4x RTX A4000 box with peer access,
  PCIe 4.0 x16 under one EPYC 7272, driver 595.91, 2026-11-02; hidden 5120 added into a BF16
  residual). On four cards one decode column (20 KiB) took 24.1 us through host words, 15.7 us by a
  whole reduce reading the peers and 9.9 us by peer-written words; four columns (a verification of
  three drafts) 31.4 us by peer reads and 23.4 us by peer-written whole partials; a 40 MiB prompt
  chunk 5.6 ms by a reduce-scatter through host memory, 4.1 ms by one reading the peers, 3.8 ms by
  one writing them and 27.4 ms by a whole reduce reading them. On two cards one column took 7.3 us
  by peer reads and 4.9 us by peer-written words. The routes chosen at startup came within 8% of the
  fastest at every size; the worst were 31.1 us against 29.1 at eight columns on four cards, where
  the startup times put the two reduce-scatters the other way round, and 5.3 us against 4.9 for one
  column on two. Measuring took 0.4 s on two cards and 1.0 s on four for a 20 MiB slot. Qwen3.8-27B
  (groupwise, INT8 KV, MTP with three drafts, `--tensor-split 1,1,1,1`) on the four cards took
  38.4-40.1 ms per decode round against 39.4-41.1 with `--tensor-parallel-transport peer-read` and
  40.0-41.5 with `host`, and prefilled a 6,768-token prompt at 1,445 tok/s against 1,370 and 1,238;
  `peer-write` alone took 38.2-39.8 ms and 1,447 tok/s. On two of the cards the measured routes and
  `peer-write` decoded within 1% of each other. Every answer was the same.
