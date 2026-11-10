# Hybrid prefix cache

The hybrid prefix cache (HPC) is the default prefix-cache mode of `ninfer-serve`
(`ContextCacheOptions::mode = ContextCacheMode::Hybrid`; `--use-alt-prefix-caching` names it). It
reuses prompt prefixes through content-addressed 64-token KV blocks and sparse recurrent-state
snapshots, kept on the Device and in one pinned Host tier. The checkpoint catalog
([Resource scheduling and context cache](resource-scheduling-and-context-cache.md),
`ContextCacheMode::Legacy`) is the Engine's default and serves what the hybrid cache does not;
`--use-original-prefix-caching` selects it in the server. The two modes coexist. Ian Ranson designed and built the
mode in [Wallawalla47/ninfer-custom](https://github.com/Wallawalla47/ninfer-custom).

This document is the authority for the Hybrid mode. §0–§11 describe the design as implemented,
marking the parts of the original design that were not built; §12–§15 cover optional features,
tests, configuration and risks.

Scope: an alternative to NInfer's prefix-reuse, checkpoint-retention and cache-pressure
system for `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM` on one RTX 5090 (`sm_120a`), with
`max_concurrency` 1..8, every KV profile (BF16, INT8-G64, FP8-E4M3FN-row256, NVFP4-G16, K8V4),
and every speculative backend (none, MTP, DFlash, DFlash2).

The mode also runs over pipeline stages and tensor parallelism (`--devices` with more than one
entry), where `ninfer-serve` selects it only when `--use-alt-prefix-caching` is named. Its state
images are sharded over the ranks like the live state, and its copies run on every rank's stream.
The checkpoint catalog's own policies are rejected with it (`--auto-long-anchors`,
`--auto-prefix-grid`, `--context-cache-policy`, `--release-diverged-checkpoints`,
`--thorough-admission-search`, `--recency-eviction`, `--value-aware-demote`, the disk tier). Host
restores and snapshot copies take the rank-stream form of the KV and StateImage copy routines, and a
Vision prefill uses the concurrent overlay window when one is configured. The measurements below are the original
author's on an RTX 5090.

Coexistence rules:

- The mode is fixed at Engine construction. `ninfer-serve` selects Hybrid unless
  `--use-original-prefix-caching` is given or an option only the catalog serves selects the
  catalog: its capacities (`--device-state-slots`, `--host-state-slots`, `--host-kv-mib`,
  `--max-private-continuations`, `--max-shared-prefixes`, `--max-long-anchors-per-continuation`),
  anchors and policies, the disk tier, slot files, several `--devices`, `--no-prefix-reuse` or a
  `--graft`. With `--use-alt-prefix-caching` named, several `--devices` keep Hybrid and the other
  options are rejected instead; Hybrid-only
  options are rejected with the catalog. `--host-cache-mib` applies to both modes (in Hybrid mode
  it sizes the Host slab pool). Gemma 4 and Qwen3.8-Flash-Next run context caches of their own,
  and their Engine reports the Legacy mode.
- Hybrid mode reuses the active-execution machinery unchanged: `LogicalKVPageStore`,
  `KVAddressSpaceStore`, execution rows, the active `StateStore` images, prefill/decode/speculative
  paths. It replaces only admission-source selection, retention, capture and pressure. Tree nodes
  hold non-writer references to logical pages; active address spaces map those pages exactly as a
  Legacy shared-prefix fork does.
- Nothing in §14.1 is deleted while Legacy remains a supported mode.

---

## Status

Hybrid mode configures itself: the only capacity a deployment chooses is `--host-cache-mib`
(default 8192, 0 = Device only); every other value is derived from the rest of the configuration
(§14.2).

| area | state |
|---|---|
| §5 index, §9 eviction (device LRU, host superseded-first then GDSF, dead KV), §7.1 tap planner | implemented; host-only unit tests (`ninfer_prefix_cache_index_test`) |
| §5.4 automatic Device sizing | implemented: in Hybrid mode the Main pool is not clamped to `C·L`, and `ninfer-serve` defaults `--kv-capacity` to `auto`, so free VRAM becomes Device block cache |
| §5.4 unified Host slab pool | implemented: KV blocks, snapshot images (split over slabs) and snapshot tails share one pinned pool; GDSF and the dead-KV sweep decide the split at run time |
| §6 admission as the Engine's materialization transaction | implemented: staging reserves every Device page and the state slot, Host restores run on a dedicated restore stream, and activation forks the lane at once; its Device work queues behind the copies it reads, layer by layer (§6.4, §6.5, §12.1) |
| §6.6 prefetching the blocked head | implemented: Host-only path blocks are copied into spare Device cache while the FIFO head waits |
| §7.1 taps | implemented: exact taps (explicit, generation opener, structural) are taken inside the prefill pass (§7.2) or, where it cannot take them, split the chunk; flexible taps (prompt tail, ladder) are realized at chunk boundaries at no extra forward pass |
| §7.4 tap publication | implemented: deferred until the anchoring blocks commit (a frontier inside a block, or an MTP backend one token behind); exact-frontier tails are copied into cache-owned pages |
| §7.6 block publication, write-through | implemented: blocks join the tree at every commit; a lane's blocks are written to Host in one batch when it releases them |
| §7.7 endpoint snapshots at finish and consistent abort | implemented; a turn ended by a stop string publishes the state from where the match began; the catalog returns the sequence to that state before Finish |
| §6.2 cost model | implemented: the Engine's context-cost coefficients (prefill and Host-to-Device transfer) rank sources and value snapshots |
| §5.5 persistence across restarts | implemented (opt-in `--prefix-cache-file`) |
| §12.2 in-flight prefix coalescing | implemented: requests sharing a new prefix with a lane still prefilling wait for its snapshot at the divergence instead of prefilling the prefix again; in this line a request is admitted while another prefills with `--concurrent-prefill`, or without it only while every staged prefill is a last chunk that can share a pass, so coalescing needs the option |
| reuse-loss attribution | implemented: the request log's `materialization` record carries `cached_prefix_tokens` (longest cached block prefix, reusable or not), and `restored_host_bytes` |
| §7.2 in-pass exact taps | implemented for text chunks on one device without a masked drafter, up to four a chunk; vision chunks, tensor parallelism and DFlash/DFlash2 split the chunk at the tap. Ternary Bonsai 2 27B on an RTX 4090: a 213-token first prompt with a system-block tap prefilled in 47.1 ms instead of 52.8 (47.5 instead of 53.3 with MTP) |
| §11.2 KV transfer Op | copy-engine path only (`cudaMemcpy2DAsync` runs over consecutive pages and slabs) |
| §6.3 persistent backfill proof | not issued: a blocked FIFO head is never overtaken (a proof with a growth reserve moved latency from short to long requests without a net gain, §6.3) |
| §12 optional features other than 12.1 and 12.2, §13.2 Op qualification | not implemented |
| §13.3 real-artifact scenarios | `ninfer_qwen3_5_hybrid_prefix_real_test`: Host vs Device restore exactness (with and without MTP), generation-opener and system-block reuse, Device-only mode, protocol cache hints, Vision, persistence across a restart, a repeated prompt on one lane whose pool is its context (`full-pool`, §6.3); `NINFER_HYBRID_KV_DTYPE` runs them for every KV storage (bf16, int8, fp8, nvfp4, k8v4 pass). `ninfer_ngram_concurrent_real` runs on Hybrid with `NINFER_NGRAM_TEST_CONTEXT_CACHE=hybrid` |

## 0. Summary

| | Legacy mode | Hybrid mode |
|---|---|---|
| Unit of reuse | owner/checkpoint (private continuation, shared prefix, 5 checkpoint kinds) | content-addressed 64-token KV block + sparse state snapshot |
| Identity | per-owner token ledgers, session index, shortlist keys, markers | exact token path in one radix tree (hash only for lookup) |
| KV sharing across sessions | only through an optional shared-prefix publication that must beat a private baseline | unconditional: identical blocks are one physical page |
| State checkpoint capture | prefill chunk split at the frontier + copy | a copy of the committed state at a chunk boundary; only exact taps (client breakpoints, generation opener, structural boundaries) split a chunk (§7.1) |
| Admission | bounded heuristic search over complete targets; up to 250 ms planning allowance per admission boundary | O(prompt blocks) lookup + arithmetic capacity check, no search |
| Pressure | monotone degradation graph, joint post-state projection, ordered stage peaks, one global transaction | independent eviction per tier: device LRU and host GDSF (GreedyDual-Size-Frequency) |
| Host tier | variable-extent arena with allocator geometry and a separate StateImage slot count | one pinned slab pool; KV blocks and snapshots share it with no fixed split between them |
| Device ↔ host | demote on pressure (copy before release) | asynchronous write-through when a lane releases its blocks; device eviction never copies |
| Code | ≈25 k lines in `runtime/engine/context_cache`, `program/{planning,transactions,storage}`, host arena, resource contracts | ≈6.1 k lines in `runtime/prefix_cache`, `program/prefix` and the Engine's `HybridResourceManager`, plus ≈1.8 k lines of integration |

What the design changes for users:

- **More reuse.** Every committed full block of every request becomes reusable, including
  cancelled and partially prefilled requests. Snapshots are deduplicated by content, so one
  snapshot serves every session that shares its prefix. Tap density is limited only by host memory,
  not by per-continuation anchor counts or catalog slots.
- **Lower TTFT.** No planning search. Host restores never hold an admission: they overlap the
  restored request's first prefill pass layer by layer while other lanes keep decoding (§6.5).
  Only exact taps split a prefill chunk, and the template's rewrite frontiers no longer do.
- **Throughput unchanged.** Kernels, page layout, block tables and CUDA Graph keys are unchanged.
  A flexible tap costs one Device copy of the state image (about 0.25 ms) at a chunk boundary.
  Cold prefill and decode match Legacy at the same pool size (on an RTX 5090: 10.2k against 10.1k
  tok/s prefill, 84.6 against 84.9 tok/s decode).
- **Better memory efficiency.** Shared prefixes are stored once. KV that no snapshot can reach
  ("dead KV") is reclaimed first. The host pool has no fragmentation and no split between KV and
  state.

---

## 1. Why a second design

Evidence is taken from the Legacy sources and documents.

1. **Ownership instead of content.** Reuse depends on which *owner* holds a checkpoint: private
   continuation, shared prefix, SessionIndex binding, or one of five checkpoint kinds
   (`SessionEndpoint`, `TurnClosure`, `ResponseReplay`, `LongAnchor`, `SharedStablePrefix`). It is
   also bounded by catalog slots (`max_private_continuations`, `max_shared_prefixes`,
   `max_long_anchors_per_continuation`). Two sessions with the same 30 k-token system+tools prefix
   share KV only if a shared prefix was published. Publication must be *strictly better than the
   private-only baseline* (resource-scheduling §7.2, §8.3).
2. **Planning cost on the critical path.** Materialization runs a bounded heuristic search over
   complete pressure targets. The Program projects joint post-states, stage peaks and Host
   allocator geometry for each target (§7.3–§8.7). Every admission boundary has a 250 ms
   planning allowance. That time sits directly in TTFT for the head request and
   stalls the worker for everyone else.
3. **Split chunks.** A checkpoint at an arbitrary frontier splits a prefill chunk (`text.cpp`
   `prefill_split_frontier_`), producing a short, inefficient chunk. `rewrite_execution_frontiers`
   adds more splits so replay matches the original decomposition.
4. **Host fragmentation and partitioning.** Host KV is a variable-extent arena whose feasibility
   depends on extent geometry, not free bytes (paged-kv §5.3). Host StateImages are a separate
   slot count. `--host-cache-mib` resolves a *static* split between them at startup.
5. **Fragility.** Correctness depends on invariants spread across the ResourceManager and Program
   boundary: capability generations, resource revisions, owner edges, claims, seal windows,
   borrowed-read ownership, and absolute ResourceResult adoption. Recent fix history
   (planning races, shared catalog saturation, lease reclaim) shows the maintenance cost.
6. **Size.** About 25 k lines: `context_cache/` 7.0 k, `planning/` pressure/recovery/request 6.6 k,
   `transactions/` 4.5 k, `storage/` 5.2 k, Host arena + prefix identity + resource contracts
   1.8 k.

The fundamental constraint does not change. Qwen3.5 is hybrid: 48/64 (27B) or 30/40 (35B-A3B)
layers are Gated DeltaNet with a recurrent state that cannot be rolled back. A prefix is therefore
reusable only at a position where the complete recurrent state exists. HPC keeps that rule and
builds everything else around making such states **cheap to create, cheap to store, and exactly
addressable**.

---

## 2. Prior art used

| Source | Idea adopted |
|---|---|
| vLLM automatic prefix caching | Full blocks keyed by (parent, tokens, extra keys). Reference-counted. Freed blocks go to an LRU queue in reverse order, so tail blocks are evicted first. Evictable blocks count as free capacity. |
| vLLM hybrid/Mamba prefix caching | Recurrent state is cached only at block-aligned positions. A hit is the deepest aligned position that has state. |
| SGLang RadixAttention / MambaRadixCache | Radix tree over token blocks. Recurrent-state entries attach to tree nodes with their own eviction. Leaf-first eviction. |
| SGLang HiCache | Host tier with write-through, batched or kernel-based non-contiguous transfers, and layer-ordered loading. |
| llama.cpp context checkpoints | A few full-state checkpoints per sequence for recurrent/SWA models. |
| LMCache | Layer-wise pipelined KV loading overlapped with compute. |
| GreedyDual-Size-Frequency (Cherkasova, 1998) | Size- and cost-aware eviction with O(1) aging via an inflation value `L`. |

NInfer-specific additions in the design:

1. The **in-pass tap**: the prefill pass runs each GDN layer's convolution and recurrence up to an
   exact tap into the snapshot slot and continues from it, so the tap costs no extra pass (§7.2).
2. One pinned **slab pool** for both KV and state.
3. **Dead-KV** accounting: KV that no snapshot can reach has zero reuse value.

---

## 3. Model facts that size the design

### 3.1 KV bytes

Per token, all full-attention layers, K+V including scales (from paged-kv §4.3):

| profile | per token/head | Qwen3.6/3.8-27B (16 FA × 4 KV heads) | page (64 tok) | 35B-A3B (10 × 2) | page |
|---|---:|---:|---:|---:|---:|
| BF16 | 1024 B | 64.0 KiB | 4.00 MiB | 20.0 KiB | 1.25 MiB |
| INT8-G64 | 528 B | 33.0 KiB | 2.06 MiB | 10.3 KiB | 0.64 MiB |
| FP8-row256 | 516 B | 32.3 KiB | 2.02 MiB | 10.1 KiB | 0.63 MiB |
| K8V4 | 402 B | 25.1 KiB | 1.57 MiB | 7.9 KiB | 0.49 MiB |
| NVFP4-G16 | 288 B | 18.0 KiB | 1.13 MiB | 5.6 KiB | 0.35 MiB |

MTP adds one layer's worth of pages to the bundle (+1/16 for 27B). A DFlash draft with
full-attention layers adds its own BF16 pool.

### 3.2 State image

Fixed per-sequence state (qwen3_5-model "State and numerical boundaries", dflash "Backend storage"):

| component | 27B | 35B-A3B |
|---|---:|---:|
| GDN recurrent, FP32 `[128,128,Hv]` × GDN layers | 144 MiB | 60 MiB |
| conv history, `(2·16·128 + Hv·128)·3` BF16 × GDN layers | 2.81 MiB | 1.41 MiB |
| continuation hidden | 10 KiB | 4 KiB |
| DFlash2 local rings (official config) | 40 MiB | config-derived |
| **total (DFlash2)** | **186.8 MiB** (matches the 195,897,344 B Host StateImage in README) | — |

One 27B snapshot equals about 91 INT8 pages (≈5.8 k tokens of KV), 166 NVFP4 pages, or 47 BF16
pages. **Snapshots dominate storage cost. Their placement is the main policy decision.**

### 3.3 RTX 5090 transfer and compute budget

These are the design's planning values. The implementation ranks and values with the Engine's
context-cost coefficients (§6.2, §11.4). On the measuring machine, whose GPU link runs at PCIe 5.0
x8, restores move 24–27 GB/s.

| operation | value used for design |
|---|---|
| HBM D2D copy of one 186.8 MiB image (read+write) | ≈ 0.25 ms |
| PCIe 5.0 x16 pinned H2D / D2H (practical) | ≈ 50 GB/s each direction |
| H2D snapshot | ≈ 3.9 ms |
| H2D 100 k tokens INT8 KV (3.38 GB) | ≈ 68 ms |
| Prefill, 27B, ~121 k context (README median) | ≈ 4.2 k tok/s, so 100 k tokens ≈ 24 s |
| Write-through rate at 4.2 k tok/s INT8 prefill | ≈ 140 MB/s (0.3 % of PCIe) |

Consequences:

- A host-resident prefix is about 300× cheaper to restore than to recompute. **The host tier
  carries most of the reuse value.** The device tier is a latency optimisation.
- A state snapshot is cheap to create (a Device copy of the image takes about 0.25 ms) and cheap
  to move (4–8 ms over PCIe). **Snapshots can be dense. The limit is host bytes, not time.**

---

## 4. Architecture

```text
                 Frontend (gateway threads)
   tokens, media digests, boundary hints, per-block chained hashes
                          │ PreparedPrompt
                          ▼
┌──────────────────────── Engine worker (single mutation owner) ─────────────────────────┐
│ Scheduler (unchanged FIFO) ─ HybridResourceManager (the ResourceManager surface)       │
│      │ quote / reserve / progress / reclaim / prefetch / terminal settlement           │
│      ▼                                                                                 │
│ Program                                                                                │
│  ├─ PrefixIndex (model-agnostic, src/runtime/prefix_cache/)                            │
│  │    block tree ─ nodes: 64 tokens + extra key, device page ids, host slab, pins      │
│  │    snapshots attached to nodes (+ optional tail)                                    │
│  │    device LRU (tail-first)           host eviction (superseded, GDSF, dead KV)      │
│  │    TapPlanner                         CacheCostModel (prefill and H2D costs)        │
│  ├─ Qwen3.5 binding (src/models/qwen3_5/program/prefix/)                               │
│  │    HybridPrefixCache: Host slab pool, residency, restore and write-through copies   │
│  │    admission, taps, block commit, finish; StateImage parts; persistence             │
│  ├─ Device typed KV pools (unchanged layout) + active block tables (unchanged)         │
│  ├─ Device state pool: C active slots + D snapshot slots                               │
│  └─ streams: restore (dedicated), write-through (the Program's transfer stream)        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

Ownership against AGENTS.md boundaries:

- **Core** owns physical primitives: the paged KV cache's record-addressed Host copies, which
  copy runs of consecutive pages and slabs as one strided transfer per plane.
- **Ops** gain nothing: an in-pass tap (§7.2) calls the convolution and recurrence Ops once per
  piece, and a KV page transfer kernel (§11.2) is not implemented. Flexible taps are Device copies
  of the committed StateImage.
- **Program** owns the whole cache: exact identity, physical residency, refs, eviction, taps.
  There is no cross-layer logical/physical split, because the cache's policy decisions are
  local physical facts. In Hybrid mode `ResourceManager`, `MaterializationPlanner`,
  `SharedCapturePlanner`, portfolio value and the target/pressure machinery are not constructed
  or called; they remain the Legacy-mode implementation.
- **Runtime** keeps the scheduler and the common request contracts. `HybridResourceManager`
  presents the ResourceManager surface to the Engine core (§8.2), and `runtime/prefix_cache/`
  holds the model-agnostic index, tap planner and cost model.
- **Frontend** supplies tokens, media digests, boundary hints and block hashes. It no longer
  produces cache "candidates", evidence flags or session keys for the cache.

---

## 5. Data model

### 5.1 Block key and exact identity

Block `b` of a prompt covers absolute positions `[64b, 64b+64)`.

```text
BlockKey(b) = ( parent NodeId,
                tokens[64b .. 64b+63],               // TokenId, exact
                extra = media_key(b) )               // 0 without Vision items and adapter scales
media_key(b) = a running hash, seeded with the prompt's LoRA adapter key (0 when every adapter
               scale is 0), over, for each Vision item whose token span overlaps block b and every
               earlier item (Vision changes later MRoPE positions via rope_delta):
               (content_digest[32], modality, grid, span begin/length)
lookup_hash(b) = xxh3_64(lookup_hash(b-1) || tokens || extra), lookup_hash(-1) = seed
```

- `lookup_hash` only selects a child in the parent's child map. A hit always compares
  `tokens` and `extra` exactly. A forced hash collision must not produce a false hit (tested,
  §13.1).
- Everything else that affects KV or state is fixed for the Engine lifetime: model artifact,
  KV profile, speculative backend, RoPE/YaRN options, control vectors. None of it is part of the
  key. Positions are a function of tokens and media, so they are covered by the key. A request's
  LoRA adapter scales are not fixed, so their key seeds `extra`
  ([adapters](../serving.md#lora-adapters-and-control-vectors)).
- A snapshot inside the first block hangs from the root, whose key has no `extra`, so a prompt with
  a nonzero adapter key neither publishes nor resumes from one.
- Thinking mode, templates, tools and roles are all rendered into tokens, so they need no key
  field.
- The Frontend computes `lookup_hash` for every full block during preparation, off the worker, and
  stores it in `PreparedPrompt::block_hashes`.

### 5.2 Nodes

The records below list the state the design keeps; `PrefixIndex`
(`src/runtime/prefix_cache/prefix_index.h`) holds it in its own layout.

```cpp
struct BlockNode {                      // host-side; stored in a slab-allocated arena
    NodeId   id; uint32_t generation;   // generation-checked handles
    NodeId   parent;
    uint32_t depth;                     // block index b; the node covers [64b, 64b+64)
    uint64_t lookup_hash, extra;
    TokenId  tokens[64];                // exact tokens (256 B)
    ChildMap children;                  // small open-addressed map keyed by lookup_hash
    // Residency
    int32_t  device_page[kMaxPools];    // per enabled pool; -1 when not device resident
    int32_t  host_slab;                 // -1 when not host resident
    Residency device_state;             // Absent | Resident | Filling(event)
    Residency host_state;               // Absent | Resident | Filling(event)
    // Liveness
    uint32_t active_refs;               // active sequences whose table maps this node
    uint32_t restore_pins;              // in-flight transfers reading/writing this node
    uint32_t live_snapshots_below;      // snapshots whose path includes this node
    uint64_t last_access_tick;
    IntrusiveLink device_lru;           // present iff active_refs==0 && device Resident
    IntrusiveLink host_dead;            // present iff live_snapshots_below==0 && no refs && host Resident
    SnapshotId first_snapshot;          // snapshots anchored here (intrusive list)
};
```

A node is only inserted when **all enabled pools** have committed all 64 positions of the block
(§7.6). A node may be device-only, host-only or both. A node with neither copy is deleted, along
with its subtree (§9.4).

### 5.3 Snapshots

```cpp
struct Snapshot {
    SnapshotId id; uint32_t generation;
    NodeId   anchor;                    // node of the last full block below frontier (root if F<64)
    uint32_t frontier;                  // F: the snapshot state is after tokens [0,F)
    uint8_t  tail_len;                  // F - 64*(anchor.depth+1), 0..63
    TokenId  tail_tokens[63];           // exact; only tail_len are meaningful
    // Tail KV (only when tail_len>0): one private, immutable page bundle
    int32_t  tail_device_page[kMaxPools];  int32_t tail_host_slab;
    // State image
    int16_t  device_slot;               // index into the D device snapshot slots, or -1
    SlabList host_slabs;                // ceil(image_bytes / slab_bytes) slabs, or empty
    Residency device_state, host_state;
    uint32_t restore_pins;
    SnapshotKind kind;                  // Tap | Endpoint | OutputBoundary | Boundary (§9.3)
    bool superseded;                    // §9.3
    bool branch_point;                  // §9.3
    // Host eviction (GDSF)
    double   gdsf_h; uint32_t freq;
    uint64_t last_hit_tick;
};
```

- **Tap snapshots** have `tail_len == 0`: frontier on a page boundary, KV fully in the tree.
- **Endpoint snapshots** are taken at request finish or cancel at an exact frontier. They own the
  partial last page, which moves from the finishing sequence without a copy.
- A snapshot is **valid** iff every node on `path(anchor)`, the tail bundle (if any) and the image
  each have at least one resident copy (device or host). Validity is maintained eagerly:
  removing the last copy of a node deletes its subtree's snapshots (§9.4).
- There is at most one snapshot per `(anchor, tail tokens)`. Publishing a duplicate is a no-op
  that refreshes recency.

### 5.4 Memory pools

- **Device KV pools**: unchanged geometry, plane order, block-table matrix and consumer views
  (paged-kv §3–§4, §6.4, §10–§11). A page group is in exactly one state: `Free`, `Active`
  (owned by one sequence, private), `Cached` (owned by a node or snapshot tail), or `Filling`
  (transfer destination). `available = free + evictable_cached`.
- **Main pool size**: automatic sizing keeps the capacity curve. In Hybrid mode the upper clamp
  `M_max = C·L` is removed (only the int32 token representation bounds it): pages beyond the active
  leases are cache-only device capacity. `ninfer-serve` defaults `--kv-capacity` to `auto` in
  Hybrid mode. Active admission still requires `M ≥ max(L, C)`.
- **Device state pool**: `C` active slots plus `D = device_snapshot_slots` snapshot slots in the
  existing slot-indexed layout (`ssm_states [128,128,Hv,C+D]` per layer, and so on). Snapshot slots
  are also the staging destination for taps (§7.4). Default `D = C + 2`, with or without a Host
  tier: one resident snapshot per live conversation, one staging slot, and one more for a shared
  prefix or another conversation. A Host tier smaller than the conversations it serves cannot
  stand in for the last one, since their images take the slabs their blocks need: in the replay of
  issue #222's workloads (§9.3; two lanes, four conversations of 8,900-token turns, KV k8v4),
  `C + 1` reused 0 % of the prompt tokens with a 1 GiB Host tier and 1.7 % with 2 GiB, `C + 2`
  27 % and 71 %. Over the whole replay `C + 2` cut the modelled prefill time by 5.5 %: by 27 % on
  the cases a 2 GiB tier changed, 8.7 % on those a 4 GiB tier changed (13 faster, 11 slower, by up
  to 43 %), and within 2 % with 8 and 12 GiB. The slot costs one StateImage of VRAM (about
  147 MiB for Qwen3.8-27B with MTP), which automatic sizing takes from Device KV. A default `D`
  yields to the KV capacity: when an explicit capacity, or automatic sizing's floor
  `M = max(L, C)` beside its headroom, does not fit beside `D` slots, the capacity resolver
  (`runtime/engine/kv_capacity.cpp`, from the per-slot bytes on the target's capacity curve) keeps
  the largest `D ≥ 1` that fits, the plan is built again with it, and startup logs the change. A
  given `D` is kept; a capacity that does not fit beside it fails naming `--device-snapshot-slots`.
- **HostSlabPool**: one pinned allocation of `host_cache_bytes`, divided into slabs. A slab holds
  one block bundle: the Main page packed with its Host page layout, then the backend page at the
  next 4 KiB boundary; `slab_bytes` is that record rounded up to 4 KiB. A snapshot image uses
  `image_slabs = ceil(image_bytes / slab_bytes)` slabs, not necessarily contiguous (the image's
  packed byte `o` lives in slab `o / slab_bytes`), and a snapshot with a tail one slab more. The
  free list is an index stack, O(1). There is no fragmentation and no geometry check: capacity is
  `free_slabs`. A nonzero budget smaller than one snapshot plus one block is rejected at startup.
  The pool is pinned in chunks of at most 4 GiB of whole slabs (one very large pinned allocation can
  fail or stall under WDDM); a slab never crosses chunks. A strided copy run joins consecutive
  pages only while their slab records share a chunk and advance by one pitch no larger than the
  device's maximum copy pitch. Any two records form a candidate run, so records in different
  chunks, or gigabytes apart in one, must be split explicitly. A 52 GB Host tier otherwise produced
  an invalid `cudaMemcpy2DAsync` that aborted the server.

### 5.5 Persistence (opt-in)

With `persistent_file` (`--prefix-cache-file`), the Host tier outlives the process. `ninfer-serve`
resolves the path to an absolute one at launch. It rejects a directory, a missing parent
directory, or `--host-cache-mib 0`, so an unusable location fails before any caching:

- **Save**, in the worker's orderly stop, which `Engine::stop()` or the Engine's destruction
  starts. `ninfer-serve` calls `stop()` on a confirmed Ctrl+C (a second press within 5 s,
  `serve/stop_control.h`), Ctrl+Break, `SIGTERM` or console close. Running and queued requests
  then fail as Unavailable at the next unit boundary instead of holding the stop until they
  finish; the worker answers them before the cleanup that saves, so no answer waits for the
  save. One more Ctrl+C exits at once without saving: through the shared
  `PrefixCacheSaveControl` (`HybridPrefixCacheOptions::persistent_save`) the save stops before
  its next slab or before the rename, deletes its temporary file and ends, and `ninfer-serve`
  waits up to 2 s for that before exiting. A save not yet begun never begins, and one still in
  the Device drain is not waited for. The console-close handler blocks while `main` unwinds, but
  Windows ends the process about 5 s after a close, which leaves the temporary file until the
  next save. Either way an unfinished save leaves the previous file in place.
  `Program::shutdown_cleanup` releases every lane first, so requests still in flight write their
  committed blocks through, and saves before the cleanup drops the cache. Every Host write
  lands, then every retained (not superseded, §9.3) Host-resident snapshot whose anchor path is
  Host-resident, and exactly those paths, are written parents first with their slab bytes (a
  temporary file renamed into place). Dead KV, superseded snapshots and Device-only content are
  not saved. A worker that ended on a fatal error saves
  nothing, so the previous file stays in place. A worker recovery (OOM or recoverable logic
  error) empties the cache like every cleanup, so a later save holds only what was cached after it.
- **Load**, at Engine construction before any request: the file is used only when its fingerprint
  (absolute artifact path, size and modification time, prefill signature, KV storage, speculative
  backend, RoPE scaling, the control vectors' digest, and the product binary's identity — build id plus executable size and
  time, so any rebuild invalidates it) and its Host geometry equal the running Engine's, and its
  size matches its tables. The file (format 5) holds the header, the block table, the snapshot
  table (each snapshot's anchor, tail, kind, the hits it still counted, and its branch-point and
  periodic flags, §9.3) and only then the
  slab bytes, so the loader chooses before it reads any slab. A Host tier
  that holds the whole file restores all of it. A smaller one restores whole snapshots, greedily
  by the GDSF density of §9.3 without its time-dependent base — (1 + hits) × the prefill a
  restore saves from the root, per byte of image, tail and anchor path — each with the path
  blocks it needs and no others, and reads only those slabs (`plan_host_restore`); the startup
  log then warns `prefix cache partly restored` with the kept and saved counts and the Host tier
  the file needs. A tier that holds none of the snapshots restores nothing, since a restore keeps
  only the blocks its snapshots resume through; the log then warns `prefix cache not restored`
  with the tier the file needs, and that the save at shutdown replaces the file. Entries are rebuilt Host-only, parents before children, without evicting
  anything, and their first requests restore blocks and images through the ordinary Host restore
  path. A damaged file loads nothing.
- The startup log shows the read of an accepted file as a progress phase (`loading prefix cache`,
  then `prefix cache read | bytes | time | rate`), then reports what was restored or why nothing
  was; the shutdown log reports the save.

---

## 6. Admission (lookup, choice, capacity)

### 6.1 Lookup

`PrefixCache::match(prompt)` runs on the worker:

1. Walk from the root. For block `b < floor((n-1)/64)`, find the child by `block_hashes[b]`,
   then compare tokens and extra exactly. Stop at the first miss. This gives the matched path
   `N_0..N_{k-1}`, `k` full blocks.
2. Collect candidate snapshots: every snapshot anchored on the matched path (walk each node's
   list) whose tail tokens equal `prompt[64·(anchor.depth+1) .. F)`, with **`F ≤ n − 1`**. At
   least one prompt token is always prefilled, so logits, continuation hidden and MTP/DFlash
   bridges come from the normal path. No full-hit path exists.
3. Frontiers inside a media item's token span are not valid resume points, and neither are
   frontiers past the item inside the block it begins in (`media_exclusion`). A block's key
   carries every item that begins before the block's end, while a snapshot's tail is compared as
   tokens alone and media tokens are placeholders: an item that begins in the frontier's own
   partial block would be in no key. Taps and endpoints are never placed there (a tap that would
   be moves to the item's first token), and match rejects such frontiers defensively, which also
   retires snapshots a file saved before this rule holds.

Complexity is O(k + snapshots on path). A 240 k prompt is 3,750 hash probes and exact 256 B
compares: tens of microseconds.

### 6.2 Choice

Candidates: root, the deepest valid snapshot `s*`, and the deepest valid snapshot whose image and
path are fully device resident, `s_dev`. At most three. For each:

```text
cost(s) = restore_seconds(s) + prefill_seconds(base=F_s, tokens=n-F_s)
restore_seconds = (host_path_bytes + host_image_bytes + host_tail_bytes) / measured_h2d_bw
                  + per_batch_latency
```

`prefill_seconds` is the existing calibrated machine model (`context_cost.cpp`: linear tokens
plus `B·S + S(S+1)/2` attention pairs, keyed by `prefill_signature`). Pick the minimum. Ties
favour fewer transferred bytes. This is O(1) and deterministic, and the cost model only ranks
feasible choices.

### 6.3 Capacity

```text
need_pages(s) = host_only_path_blocks(s)          // H2D destinations (shared once restored)
              + (s.tail_len>0 ? 1 : 0)            // private COW copy of the tail
              + growth_pages(prompt n, base F_s)  // existing entitlement: remaining prompt +
                                                  // bounded output window + per-lane cushion
need_state_slot = 1 active slot (a lane implies a free active slot)
```

The request is admissible with `s` iff `need_pages(s) ≤ available_pages`, where evictable pages
exclude pages pinned by this admission's own path. If not, try the next cheaper-to-materialise
candidate: `s_dev`, then root.

- Root infeasible because active entitlements hold the pages: `TemporarilyBlocked`.
- Root infeasible even on an empty cache: `PermanentlyInfeasible`.

This is the same classification as today, computed by arithmetic.

**Backfill proof** (engine-architecture §5.2) uses the same arithmetic:
`need(borrower) + need(head root) ≤ free + all_evictable + Σ donor entitlements released`.
Hybrid mode does not issue it: a blocked FIFO head is never overtaken.

A proof with this arithmetic needs a growth reserve: without one, a borrower can take the pages a
long answer would grow into, and that answer then ends early (bounded completion, paged-kv §6.2).
With a 16K-token reserve per active sequence it was safe, but it moved latency instead of removing
it: queued short requests finished in 1.6 s instead of 35 s, while the long request already
decoding slowed by about their prefill time and the blocked long head started that much later. A
version worth adding has to bound the borrower's cost to the donor, keep the donor's decode rounds
running while a borrower prefills, or suspend a borrower to the Host tier when a donor needs its
pages (active-request preemption, a product change).

Every unpinned Device-resident cache entry (node or snapshot tail) owns its page bundle
exclusively, so evicting one returns exactly one page to each pool and the arithmetic is exact.
Exact-frontier tap tails are copied into their own pages for this reason (§7.4). Candidates with a
block another admission is still filling are skipped.

Neither a tap's tail copy nor the tail an admission pins while it copies it is part of any
entitlement. The derived output budget (`Engine::concurrent_output_budget`, the serving default
output limit) therefore leaves `max_new_taps + 1` pages of each pool free beside a lane's
entitlement. Without them, one lane over a pool no larger than its context reserves every page, and
a repeated prompt (an OCR page sent again, a regenerated answer) prefills from the root. A request
whose explicit limit fills the pool still runs, publishing no tailed tap and resuming from no tailed
snapshot.

### 6.4 Admit

Admission is the Engine's materialization transaction (`hybrid_reserve_materialization`, then
`progress_context_transaction` until `Published`). The first progress step stages it
(`hybrid_stage`), in the same Engine call as the quote:

1. Re-validate the quoted snapshot and path. Pin every path node up to `F_s` (removed from the
   device LRU) and pin `s`.
2. Evict from the device LRU until the pools hold every page the admission needs, then **reserve**
   them: `entitlement − F_s/64` for the fork, plus one per host-only path block and host-only tail.
   Reserve the lane's StateImage destination slot.
3. If anything is host-only, open a restore batch on the dedicated restore stream (ordered after
   the compute stream's queued work, since destinations may have been read by earlier kernels):
   - host-only blocks: materialize destination pages from the reservation, adopt them as cache
     references, mark the nodes `Filling`, and copy the slabs in runs of consecutive pages and
     slabs;
   - host-only tail: the same into a cache-owned page (the snapshot's tail is Device-resident
     again afterwards);
   - host-only image: copy the image slabs straight into the reserved destination slot.
   The batch is enqueued in the order a forward pass reads it (§6.5).

Activation (`hybrid_activate`) then builds the lane at once, without waiting for the copies: the
reservations return to the pools and the page-prefix fork takes them back; path pages are shared; a
partial tail is D2D-copied into a private page; the state is reset, D2D-copied from the snapshot's
device slot, or restored in place. The lane takes over the path pins and the snapshot pin is
dropped; a restore batch holds its own pins on the nodes and snapshot it copies until it lands.

Admission never evicts anything that another sequence has pinned. Evictions come from the device
LRU (§9.1) and complete synchronously: dropping a device copy is bookkeeping, since the data is
already backed up or deliberately discarded.

### 6.5 Restoring

A restore batch never holds its admission. `submit_restore` enqueues it on the restore stream in
the order a forward pass reads it, recording an event after each part:

1. the **prelude**: the snapshot tail bundle, every restored block's backend page, and the image's
   continuation hidden and DFlash local state;
2. then, for each model layer in forward order, that layer's state: a full-attention layer's KV
   planes of every restored block, or a linear-attention layer's conv and recurrent state.

`land_restore` makes the compute stream wait for the prelude (activation copies the tail and the
MTP bridge reads the backend pages and hidden) and hands the lane a ticket for the per-layer
events. The lane's first prefill pass waits for each layer's event just before that layer
(`TextContext::run_layers`), so it computes the early layers while later ones are still arriving;
later passes are stream-ordered behind it. The chunk function itself looks the events up by
ticket (`PrefillContext::take_layer_ready`) just before its pass, because `poll()` retires a
landed batch and recycles its events, and program code between chunks polls: a batch already
retired has landed, so the pass has nothing to wait for. A lane released before its first pass makes the compute
stream wait for the whole batch, because the state slot it returns may still be a destination.

The index keeps the batch's nodes and tail `Filling`, and the batch keeps them and its snapshot
pinned, until `poll()` (every admission quote, and the block publication after each KV commit)
or `drain()` sees the last event. So no eviction,
dead-KV reclaim or Host write can free a slab or page a copy still reads or writes. Other admissions
skip `Filling` candidates for that short window.

Consequences:

- A Host revisit no longer waits for the host to notice the copy finished: under concurrency that
  was the rest of the other lane's decode round (~12 ms measured), and with an idle Engine a timer
  tick.
- The copy overlaps the restored request's own first prefill chunk: for a long context, the prefill
  of new tokens hides most of a multi-GB restore.
- Other lanes' decode rounds queue behind that prefill chunk as for any prefill. When the chunk is
  much shorter than the copy (a few new tokens after a very long restored context), they can wait up
  to the copy's remaining time once.

Failure: a CUDA error in a restore job is Engine-wide, as today. A request cancelled while staged
(before activation) waits for the restore stream, returns every destination and reservation, and
releases its pins.

### 6.6 Prefetching the blocked head

When the Device pool cannot hold every conversation, each admission restores most of its path
from the Host tier (the production agent log of 2026-09-26: 2.59 TB, a median 2.7 GB per turn).
For a short turn the first prefill pass waits for those copies, and other lanes' decode queues
behind it. The FIFO head usually waited seconds to minutes for a lane or for pages before that,
so its restore can happen during the wait instead.

- When the head stays blocked (hybrid mode does not backfill, §6.3), the Engine asks the Program to
  prefetch for it. The Program matches the prompt as a quote would, takes the source admission
  would choose, and copies up to 256 of its Host-only path blocks into Device pages. They are
  ordinary cached blocks: `Filling` until the batch lands, then unpinned at the MRU end of the
  device LRU. Admission later finds them Device-resident and restores only the rest.
- Pages come from the free pool and from host-backed cached blocks, least recently used first.
  The head's path is pinned meanwhile. A prefetch never drops a block's last copy, never waits for
  a transfer, and never runs beside an open admission or another prefetch. Active leases may evict
  the prefetched blocks like any cache.
- One batch is about 550 MB for 27B INT8 KV (about 20 ms of PCIe). The Engine retries for the same
  head only while the last attempt copied blocks or the spare room grew, because matching walks
  the whole prompt path.
- A quote whose candidates cross a landing prefetch waits for it (`settle_prefetch`) instead of
  skipping those candidates for a shallower source. A lane's own restore batches are never waited
  for synchronously (§6.5).

In the production log about 57 % of the restored bytes could have been copied this way. The rest
did not fit beside the active sequences while the head waited.

---

## 7. Prefill, taps, decode, finish

### 7.1 Tap planning

`plan_taps(n, base F_s, hints, existing frontiers, media exclusions)` returns taps sorted by
position, each `{p, placement}` with `F_s < p ≤ n − 1`, `p` not strictly inside a media item's
exclusion (its token span, extended to the end of the block it begins in, §6.1; a candidate
inside one moves to the item's first token), and no existing or already-planned snapshot at or
near `p` on this path.

Candidates, in priority order:

| priority | source | placement | rationale |
|---|---|---|---|
| 1 | explicit markers (OpenAI breakpoints, Anthropic `cache_control`, C++ `PromptInput` markers) | exact | the client names reuse points |
| 2 | start of the final assistant generation opener | exact | the next turn re-renders the history after it (for example, stripped thinking) |
| 3 | end of tools, end of the leading System/Developer block | exact | compaction and new sessions sharing the preamble |
| 4 | `n − 1` ("prompt tail") | flexible | regenerate / retry of the same prompt |
| 5 | geometric ladder: for k = 0,1,…, `n − G·2^k` (G = `tap_ladder_tokens`), snapped back to a message boundary within the minimum gap | flexible | edits deep in history. Spacing grows with distance, so there are about log2(n/G) taps |

- **Exact** taps snapshot the state at `p` itself. The prefill pass takes up to four of them
  inside a chunk (§7.2); elsewhere a tap splits the chunk at `p` (one more forward pass). Semantic
  boundaries are where later prompts diverge, so the exact position is worth it.
- **Flexible** taps never split: the snapshot is taken at the end of the chunk that reaches `p`, or
  at the start of the prompt's final chunk when `p` lies in it (the prompt tail always does). A
  chunk boundary costs nothing beyond the snapshot copy.
- Semantic boundaries (generation opener, structural) closer than 64 tokens form one cluster and
  only its earliest position is kept: a short user turn puts the system-block end next to the
  opener, and a snapshot at the system-block end serves new conversations sharing it while the
  next turn recomputes fewer than 64 tokens.
- A snapshot serves every prompt that diverges after it, so proximity only makes a *later*
  candidate redundant: a non-explicit candidate is dropped when an accepted tap, an existing
  snapshot on the path or the base lies at most 64 tokens before it. Ladder candidates keep
  `tap_min_gap` (default `max(1024, chunk)`) on both sides instead. The opener therefore absorbs
  the prompt tail in chat traffic, leaving one exact tap per request.
- Only markers with explicit evidence (a client-named breakpoint) are priority 1. Protocol-automatic
  markers (OpenAI default caching, Anthropic automatic `cache_control`) are planned as structural
  boundaries, but they mark the conversation's latest turn rather than a prefix conversations
  share.
- Explicit and structural taps (not protocol-automatic ones) are published as `Boundary`
  snapshots: conversations may share them, so a lineage moving past one supersedes it only while
  no other conversation has continued from it (§9.3). A cluster keeps this from any member it
  drops.
- A request that resumed from an endpoint snapshot proves its client echoes generated turns token
  for token (agent loops, preserved reasoning), so its own endpoint serves the next turn and the
  generation opener is not tapped. Measured on a tool-calling agent trace (27B NVFP4, INT8 KV,
  `--preserve-thinking`), the opener split cost about 15 ms of a ~70–130 ms turn TTFT. Requests
  that resumed from a tap keep the exact opener: their client re-renders history.
- A `Disposable` request (a decision read) keeps no tap past its last explicit marker: no request
  continues it, so a snapshot of its own text (the question) serves nobody and an exact tap there
  only costs a pass.
- On an Engine with a drafter (`--spec`), a request without drafting
  (`ExecutionOptions::speculative_drafting = false`, one output token; the decision reads) keeps
  only exact taps, and the drafter's prompt work (the MTP bridge and KV, the DFlash context) runs
  only below the deepest one or its resume point: the snapshots it publishes there carry the
  drafter's state for drafting requests, and the rest of its prompt runs without the drafter. It
  publishes no endpoint, since its drafter state stops short of it.
- Hints are the Frontend's existing `message_boundaries` plus the structural boundaries it already
  computes.
- At most `max_new_taps` (default 8 with a Host tier, 2 without) taps are kept per request.

Tap placement is a function of the prompt, so it is deterministic. Endpoint snapshots are exact
(§7.7).

Exact taps that the pass cannot take are the only prefill splits in Hybrid mode. The chat template's rewrite execution
frontiers (after each assistant header, after `<think>`, after the reasoning close) split Legacy
prefill for its rewrite checkpoints and execution provenance. Hybrid captures nothing there, so a
Hybrid lane ignores them. Each split is a whole extra pass over the model. A short chat turn used
to prefill as four passes (reuse point, opener, `<think>`, reasoning close). Now it takes one, with
its exact taps inside the pass, or two with a tap the pass cannot take.

### 7.2 In-pass exact taps

A text prefill chunk on one device takes up to four exact taps strictly inside it without ending
the pass at them (`TextContext::set_gdn_taps`):

- Before the chunk, each tap reserves a staging snapshot slot (as a chunk-boundary tap does, §7.4);
  a tap without a free slot, inside an exclusion or at most at the lane's last capture ends the
  chunk at it instead.
- Every GDN layer runs its causal convolution and its gated delta rule once per piece: from the
  lane's source slot to the first tap's slot, from each tap's slot to the next, and from the last
  tap's slot to the destination. Each call starts its chunk grid at the piece's first column, as
  the pass split at the tap would, so the tapped convolution history and recurrent state equal
  that pass's. The attention layers, the MLPs and the KV writes run over the whole chunk.
- The final norm's column `p − 1` is copied into the slot's continuation hidden.
- After the chunk the taps are published as pending taps (§7.4); a chunk that did not take them
  (its length or a vision span did not hold them all) releases the slots, and the tap splits the
  next chunk.

Vision chunks, tensor parallelism (whose GDN heads are divided over the ranks) and a masked
drafter (DFlash, DFlash2: its context ring in the image would have to end at the tap too) split
the chunk at the tap. MTP needs nothing: tap snapshots carry no MTP state, and a resume runs the
MTP bridge from the resumed frontier.

### 7.4 Tap publication

At the chunk boundary the tap takes a staging device snapshot slot, copies the lane's committed
StateImage into it (including the continuation hidden of column `p − 1`, which prefill keeps while
taps remain) and waits as a **pending tap** of the lane. It is published as soon as the blocks it
anchors on are committed tree nodes:

- the anchor `p/64 − 1`, which for an MTP backend (one token behind the text frontier) commits
  only after the next chunk even when `p` is page-aligned;
- for `p % 64 ≠ 0`, block `p/64` as well, whose first `p % 64` columns are copied into a
  cache-owned tail page, so every cache entry keeps an exclusive page (§6.3).

Publication creates a `Snapshot{Tap}` with `device_slot` = the staging slot and enqueues D2H
write-through of the image (and tail) into host slabs; `live_snapshots_below++` along the path.
Once write-through completes, the slot becomes an **evictable device snapshot** (§9.2). A finishing
lane publishes its remaining pending taps from its own last page, or drops them.

### 7.5 Recomputed region

When `F_s < 64k`, blocks `[F_s/64 .. k)` already exist in the tree but are recomputed, because the
recurrent state must be rebuilt. The sequence writes them into its own private pages. At block
commit (§7.6), each such block finds the existing node, so the private page stays private
(`Duplicate`) and is freed at finish. The tree is not modified. KV-write elision, which maps the
existing pages and discards those writes, is an optional optimisation (§12.4).

### 7.6 Block commit (prefill and decode)

After any commit that advances every enabled pool's committed frontier across a block end
`64(b+1)`:

- If `child(N_{b−1}, key(b))` exists with a Device copy: the sequence's page stays private
  (`Duplicate`). If it exists host-only, it adopts the sequence's page as its Device copy. Otherwise
  **insert** a node owning the sequence's page. The node stays pinned for this sequence.
- D2H write-through of the lane's Device-only nodes is enqueued in one batch when the lane releases
  its path (finish, abort or cancel), before the nodes become evictable. Batching coalesces runs of
  consecutive pages and slabs into one strided copy per plane, and pinned nodes need no backup yet.

The block-table row is unchanged, so no GPU work is needed.

Only committed tokens count. Speculative provisional lead, rejected columns and MTP/DFlash lag are
excluded by requiring every enabled pool's *committed* frontier (paged-kv §9.1).

Decode blocks become cacheable the moment they fill. A later request that shares a long generated
output, such as an agent re-sending the assistant turn, can resume from the endpoint snapshot
(§7.7).

### 7.7 Finish and cancel

At terminal settlement (Finish, or Cancel after at least one committed unit):

1. If the request is cache-enabled, every pool has committed the frontier (the backend up to its
   own frontier, `F_end − 1` for MTP) and `F_end − (deepest snapshot on path) ≥ 64`:
   - The active StateImage itself becomes the snapshot image (frozen, no copy) in a device snapshot
     slot; if none is free, one is evicted (§9.2), otherwise the endpoint is skipped.
   - The partial last page (`F_end % 64 ≠ 0`) moves to the snapshot as its tail. There is no COW,
     because the writer is terminating. The backend page keeps its own committed extent.
   - Publish `Snapshot{Endpoint}` and enqueue D2H write-through of the image and tail.
2. For every pinned node: `active_refs--`. A node that reaches 0 goes to the device LRU tail
   (§9.1), inserted in **reverse block order**, so deeper blocks are evicted before shallower
   ones.
3. Free `Duplicate` and growth pages. Release the active state slot.

**A turn that a stop string ends.** The visible text ends where the stop string's match began,
and a state that has read tokens of the match continues no later prompt. The output session
reports the round in which a match prefix is first held back and the number of the round's tokens
before it (`OutputDecision::stop_hold_after`), and whether a finish cuts the output there
(`stop_cut`; a stop string that stays in the output ends inside the round's last token and cuts
nothing the state has read).

- The match begins in the round that ends the turn: the Engine commits only the round's tokens
  before it (`CommitDecision::accepted_tokens`), so `F_end` is that position and step 1 publishes
  the endpoint as for any turn, with every speculative backend and without the hybrid cache too.
- The match begins in an earlier round: the lane keeps the state after exactly the tokens before it
  in a device snapshot slot of its own (`HybridLaneState::hold`), replaced when a later match
  begins. A round without drafts (or the prompt's last step) has not read the token that begins the
  match, so the lane's StateImage is copied. An MTP round folds the row's replay records once more,
  with that many columns and the kept image as destination, before the fold that commits the
  round, and copies the target's hidden at that column: the same state transition a round ending
  there commits. At Finish the kept state is published as the turn's `Snapshot{Endpoint}` at its
  frontier (its tail copied from the lane's page, as a tap's), under the conditions of step 1.
  The state at `F_end` is not published. Any other finish drops the kept state.
- A DFlash or DFlash2 round keeps the same folded image and hidden. Its image also carries the
  drafter's local context, which ends at the round's base until the next round appends the
  round's target features: the kept image takes a copy of the lane's context and the features of
  the tokens before the match (`enqueue_dflash_context_append` with the kept image as
  destination), through the pinned ingress that a terminal row's append uses, so the host waits
  for it. A row whose image is a pending fork keeps nothing. A round without drafts keeps its
  image only when the context already ends at the frontier (after a prompt).
- The checkpoint catalog keeps the same state in a Device StateImage of the lane's own, outside
  the catalog's slots (`models::stop_hold_slots`, one for each concurrent request;
  `ContextCacheOptions::stop_hold_images`, dropped at startup when the KV capacity does not fit
  beside them). The terminal commit copies it back over the lane's image and cuts the ledger, the
  prefix identity and the KV extents to its frontier, so Finish catalogs an ordinary endpoint
  there. A checkpoint of the sequence past that frontier, or an image the sequence shares, keeps
  the turn as it ended.

`Disabled` requests (Serve warmup, `--no-prefix-reuse`) never insert nodes or snapshots.
Their pages are freed at terminal.

---

## 8. Engine and API

### 8.1 Program API

```cpp
struct HybridAdmissionQuote {               // hybrid_quote(): the selected source, exact at admission
    runtime::Readiness readiness;           // Ready | TemporarilyBlocked | PermanentlyInfeasible
    runtime::LaneId destination;
    runtime::RequestPlanSummary summary;    // carries the reuse frontier, reported as cached_tokens
    std::shared_ptr<detail::HybridQuoteImpl> impl;
};
[[nodiscard]] HybridAdmissionQuote hybrid_quote(const PreparedPrompt&, const RequestBasePlan&,
                                                runtime::LaneId destination);
[[nodiscard]] runtime::ContextTransactionReserveStatus
    hybrid_reserve_materialization(HybridAdmissionQuote&&, PreparedPrompt&&,
                                   runtime::CancellationFlagView);   // §6.4; progressed by
                                                                     // progress_context_transaction
[[nodiscard]] std::uint32_t hybrid_reclaim_device_kv(std::uint32_t main_pages,
                                                     std::uint32_t backend_pages);   // §9.5
[[nodiscard]] std::optional<std::uint32_t> hybrid_prefetch(const PreparedPrompt&,
                                                           const RequestBasePlan&);  // §6.6
[[nodiscard]] std::uint32_t hybrid_prefetch_room() const noexcept;
void set_hybrid_cost(const runtime::prefix_cache::CacheCostModel&);                   // §6.2
void set_hybrid_coalesce_wait_limit(double seconds);                                 // §12.2
[[nodiscard]] HybridCachePersistence attach_hybrid_cache_file(...);                  // §5.5
[[nodiscard]] std::optional<HybridCachePersistence> hybrid_shutdown_save() const;
[[nodiscard]] HybridPrefixCacheStats hybrid_stats() const noexcept;
// Unchanged: advance_prefill, decode, append_forced_tokens, commit, abort_pending, finish, abort.
```

Not called in Hybrid mode (retained for Legacy): `inspect_admission`, `seal_identity`,
`begin_pressure_planning`, `shared_capture_split_prefill_work`, `start_resource_transaction`,
`finalize_context_transaction`, `inspect_capture`, `checkpoint_recovery_work`,
`begin_capture_pressure_planning`, `shared_capture_matches`, `skip_capture`,
`reserve_active_capture*`, `retained_device_kv_pages`, `release_continuation`,
`release_shared_prefix`, `resource_revision`. The Hybrid methods throw `std::logic_error` in Legacy
mode and vice versa.

### 8.2 Engine

The Engine selects one of two resource managers at construction from `ContextCacheMode`. Legacy
keeps `ResourceManager` and the planners. Hybrid constructs `HybridResourceManager`, which presents
the same surface to the Engine core, so scheduling, prefill, decode and commit orchestration are
shared by both modes; it tracks lane ownership and forwards admission and terminal settlement,
while retention policy lives in the Program's index.

- Admission is the ordinary context transaction: quote, reserve, then
  `progress_context_transaction` until the started sequence is published (§6.4). Restores never
  hold it (§6.5).
- `ensure_base_plan` builds no candidates, and no `CaptureOffer` is produced: taps are internal to
  prefill.
- Device KV lease settlement evicts from the device LRU (`hybrid_reclaim_device_kv`) instead of
  releasing owners.
- While the FIFO head stays blocked, the Engine asks for a prefetch of its Host-only path (§6.6).
- `HybridPrefixCacheStats` feed `record_prefix_selection` and the request-log fields.

### 8.3 Frontend and protocol mapping

- `PreparedPrompt` gains `block_hashes` and `tap_hints` (sorted `{position, priority}`), filled in
  both modes (cheap). Hybrid mode ignores `PreparedContextCache` session key, retention and
  opportunities, and `PromptIdentity` rewrite fields; Legacy ignores the new fields.
- `ContextCacheHints` is unchanged. In Hybrid mode:
  - explicit-evidence markers map to priority-1 tap hints, automatic protocol markers to
    `Automatic` hints (planned as structural, published as ordinary taps);
  - `session_key`, `retention`, `allow_engine_automatic_shared_prefixes` and
    `update_session_index` are ignored. Content addressing finds session chains without them.
- External protocol behaviour is identical in both modes:
  - OpenAI `prompt_cache_*` and breakpoints, and Anthropic `cache_control` validation, limits
    (four distinct breakpoints) and error cases are unchanged.
  - In Hybrid mode, breakpoints mean "tap here" (priority 1). Protocol write-policy modes
    (`mode:"explicit"`, Anthropic `cache_control`-only requests) do not suppress automatic taps:
    taps are cheap and content-deduplicated, so markers can only add reuse.
  - `cached_tokens` and `cache_read_input_tokens` report `reuse_tokens`. The streaming
    `message_start` timing is unchanged, because the quote fixes the value at admission.

---

## 9. Eviction

### 9.1 Device KV (LRU, tail-first)

- One intrusive list of `Cached` device pages whose owner has no refs and no pins: nodes with
  `active_refs == 0 && restore_pins == 0`, and snapshot tails with `restore_pins == 0`.
- On a hit, touched nodes move to the MRU end.
- On release, nodes are appended in reverse block order.
- Allocation pops from the LRU head:
  - **Host-backed** entries (host `Resident`): drop the device copy. Free.
  - **Unbacked** entries, which occur only when the host is disabled, full, or write-through is
    still filling: the node loses its last copy, so apply §9.4.

Two lists are kept, *backed* and *unbacked*, and backed entries are always consumed first.

A snapshot tail joins the list only when its snapshot holds no Device slot, and it joins when the
snapshot gives the slot up (§9.2). The slot image is restorable only together with its tail, and
it costs at most one page per slot. A tail evicted alone left its snapshot in the slot but no
longer complete on the Device, so Host eviction (§9.3) deleted the snapshot instead of keeping its
Device copy. A larger Host tier backs more tails, which this list consumes first, so a rotation
of conversations resuming from slot snapshots reused fewer prompt tokens with 4 GiB than with
2 GiB (the numbers are in §9.3).

### 9.2 Device snapshot slots

Slots hold at most `D` images. A sequence about to snapshot past its previous snapshot (the one
it resumed from, or its previous tap in the same prompt) supersedes it first (§9.3), so the slot
comes from its own lineage when it can. A tap or endpoint takes a slot in this order:

1. a free slot;
2. a slot owned by a superseded snapshot, oldest supersession first; the snapshot keeps its Host
   copy or, without one, is deleted;
3. a slot owned by a Device-only snapshot (no Host copy: the Host tier is full, disabled or
   evicted it): the one with the lowest GDSF priority (§9.3), and only when the new snapshot's
   estimated priority (the hits its lineage carries, §9.3, and the blocks since the sequence's
   previous snapshot its own) is at least as high. The snapshot is lost, and `L` rises to its
   priority;
4. the least recently hit Host-backed snapshot's slot, which loses only its Device copy.

Otherwise the tap or endpoint is skipped, and `L` rises to its estimated priority, as if it had
been inserted and evicted at once. A Device-only owner's slot is its last copy, so value decides
there rather than recency: least-recently-hit eviction let a short request's snapshots push out the
snapshot a long conversation resumes from, and taking Host-backed slots first left owners that
Host eviction had made Device-only in their slots for good. Without the rise, nothing would age
owners that are never hit again: decision reads about a stream of new states hit each state's
snapshot two or three times, and once those snapshots filled the Host tier and the slots, every
later state's snapshot was refused (about 50 states into a run on an RTX 3090 with an 8 GiB Host
tier) and every read prefilled its state again. A snapshot that keeps its Host copy when its
slot goes puts its tail on the Device LRU (§9.1).

### 9.3 Host (superseded snapshots first, then GDSF; dead-KV first)

**Supersession.** In a conversation the next request resumes from the newest snapshot of its
lineage, not from the one the current request resumed from. When a sequence is about to publish a
deeper snapshot on its path, the snapshot it resumed from becomes *superseded*, and so does the
previous tap it captured in the same prompt when the new snapshot is a tap, before the new snapshot
takes a Device slot or Host slabs: only a request diverging before the newer snapshot (a retry, an
edit) can still use them. The endpoint leaves the prompt's last tap retained, since a next turn
whose template re-renders the reply (stripped thinking) resumes from that tap. For the same reason
an endpoint supersedes the snapshot its request resumed from only when that is an endpoint too (a
client that echoes the generated text resumes its next turn from the new endpoint): a tap or
boundary stays retained. A short follow-up turn captures no tap of its own (it is shorter than the
minimum tap separation), so its endpoint used to supersede the tap a re-rendering client's next
turn resumes from, and that turn missed under pressure. A hit on a superseded
snapshot retains it again, until that lineage moves on: the hit is as likely the same conversation
coming back, such as the next turn after a retry that resumed from the same tap. A tap that only the
next tap of its own prompt superseded is different. No request has resumed from it, so a hit on it
comes from a request that diverged from that prompt after the tap (another question about the same
document, an edit). Requests branch there, and each branch would supersede it again as it moves on,
so the hit makes the tap a *branch point*: nothing supersedes it, and GDSF alone values it (§5.5
saves the flag). A `Boundary` snapshot (a tap at a client breakpoint or structural boundary, or at
the divergence of coalesced requests, §12.2) may be shared by conversations, so it is superseded
only while no other conversation has continued from it: while the tree below it is a single chain.
Until then it is just its lineage's snapshot. A request with `CacheRetentionHint::Disposable` (the
decision endpoints' reads) has no lineage, since no request continues it, and supersedes nothing it
resumed from. The first of several reads that share a boundary would otherwise supersede it with its
endpoint, so far the only branch below the boundary, and the boundary would then give up its Device
slot and Host slabs first while the other reads still wait to resume from it.

Allocation of `k` slabs:

1. **Dead KV sweep**: pop host-resident nodes with `live_snapshots_below == 0`,
   `active_refs == 0` and no pins, leaves first. These can never produce a hit.
2. Otherwise the oldest superseded snapshot gives up its host copy (`L` is unchanged), then
   repeat step 1. One still complete on the Device keeps that copy, whose slot goes first when a
   new snapshot needs one (§9.2); any other is deleted. Deleting the Device copy here too would let
   a lane's block write-through into a small Host tier at release delete the superseded taps of
   its prompt, and with them every snapshot the next question about the same document could resume
   from. For the same reason a lane's Host writes skip the newest tap that a later tap of its
   prompt superseded: its block write-through, whose blocks are that tap's own, and the image
   writes of its later taps and endpoint. When other conversations hold the Device slots, the taps
   of a new document's first question pass one slot along, each keeping only its Host copy, and the
   write-through used to evict them for their own blocks; the second question then recomputed the
   whole document (115,210 tokens after a four-conversation rotation with an 8 GiB Host tier, in
   the replay of §5.4). Once the Host tier is full, the next tap's image takes the same slabs:
   with the slot tails of §9.1 and without this rule, a 115,210-token document asked about after a
   three-conversation rotation with a 2 GiB Host tier was recomputed by all ten questions.
3. Otherwise evict the retained snapshot with minimum `H`, then repeat step 1 (its exclusive path
   has just become dead). Update `L := H(victim)`. A victim still complete on the Device (image in
   a device slot and tail resident) only gives up its host copy and stays restorable.

A snapshot image write stops at step 3 when the victim is worth more than the new snapshot
(`H(victim) > H(new)`): GDSF never inserts what it would evict next, and the new snapshot stays
Device-only (§9.2). `L` still rises to `H(new)`, as if the new snapshot had been inserted and
evicted, so retained snapshots that are not hit again age out. A lane's block write-through at
release stops at step 3 the same way at a victim worth more than the lane's most valuable retained
snapshot (`retained_priority` of the snapshots it published), and at every retained victim when
the lane published none: the blocks are worth what the snapshots anchored on them are worth.
Unbounded, a conversation's release evicted retained snapshots of the other conversations, which
are worth more, for blocks no snapshot of its own needed, and the next turn of each of them
prefilled its context again.

```text
H(s)  = L_s + F(s) · C(s) / Z(s)                // L_s: L when s was published or last hit
C(s)  = prefill_seconds(base=F_a, tokens=F_s−F_a) − restore_seconds(image)   // clamp ≥ 0
        where a = nearest retained (not superseded) snapshot above s on its path, whatever its
        tail (or root, F_a = 0)
Z(s)  = image_bytes + tail_bytes + exclusive_path_bytes(s)
        exclusive_path_bytes: host bytes of the nodes on path(s) below which s is the only
        retained snapshot
F(s)  = 1 + hits(s)                             // hits counted now, see Hits below
```

- Each snapshot records its nearest snapshot above it on its path (`ancestor`) and the snapshots
  that record it (`dependents`); a new snapshot between them takes over the dependents below it,
  and a removed one hands its dependents to its ancestor. `C` and `Z` are recomputed for the
  snapshot and for those whose nearest retained ancestor or exclusive path it changes, when it is
  published, hit, superseded, retained again or removed.
- Valuing against the nearest *retained* ancestor makes a conversation's newest endpoint carry the
  whole conversation-specific prefill (its superseded predecessors go first), and a shared
  boundary its own prefix. Valuing only against aligned (`tail_len == 0`) ancestors, as before,
  priced every tailed endpoint as a full re-prefill from the root: stale endpoints of long
  conversations crowded out shared boundaries and fresh endpoints, which were evicted before the
  next turn (the production agent log of 2026-09-26: 68 turns re-prefilled 758K tokens behind a
  stuck resume point, 24 subagent turns prefilled from the root beside a cached 19.8K-token
  shared prefix).
- `C` uses the same calibrated model as §6.2, so eviction and admission agree about value.
- **Hits.** A snapshot's hits count only while they are recent. A clock advances by one for every
  block inserted into the tree and by `image_slabs` for every snapshot published: by the slabs of
  new content, so the cache turns over once in about `max_nodes` (Device pages plus Host slabs)
  units. Once a third of that has passed since a snapshot's last hit (or its publication or
  restore, §5.5), its hits count as zero, except that a *periodic* snapshot counts half of them for
  another third; a hit adds one to the hits still counted. A snapshot becomes periodic when a hit
  comes at least half a window after the hit or publication before it (§5.5 saves the flag):
  requests return to it after a pause, such as a document asked about once per round of other
  conversations, or a conversation resumed once per round. The half step keeps such a snapshot
  when the working set overflows the cache: conversations that missed are prefilled again and
  republish their ladder taps, so one round can pass a whole window, and dropping every hit there
  evicted the document for a conversation that missed anyway (issue #248: a 115,210-token
  document asked once per round of a four-conversation rotation on an RTX 4090 with an 8 GiB Host
  tier was recomputed three times out of eight). A snapshot hit only in one burst gets no half
  step: a document asked about ten times in a row would hold the cache a window longer than the
  hits that earned it, against the conversations that follow (issue #222's order). When a
  request supersedes the snapshot it resumed from, that snapshot's counted hits pass to the
  snapshots the request publishes next and to their claims (§9.2): a conversation is resumed once
  per turn, but each turn publishes new snapshots, which otherwise start again at `F = 1`.
- Aging comes from `L` and from the hit window, so snapshots that are no longer hit fall behind new
  ones without timers. With `F` growing for good, a document hit nine times kept its rank long after
  its questions stopped (issue #222): a four-conversation rotation that followed it on an
  RTX 4090 (KV k8v4, 131,072-token cache, 8 GiB Host tier) reused 15 % of its 1,070,925 prompt
  tokens instead of 83 % on a fresh engine. Replaying that workload and its neighbours through the
  index (rotations of three to six conversations, documents of 57,741 and 115,210 tokens before or
  after them, a document asked about in bursts between conversation turns, conversations sharing a
  system prompt; Host tiers of 0 to 12 GiB), the window of a third, the lineage hits, the endpoint
  rule above, the write-through rule of step 2 and `D = C + 2` cut the modelled prefill time from
  83,443 s to 69,306 s in all 248 cases: 113 faster by more than 2 %, 3 slower. A document asked
  about again after 24 or 32 conversation turns with a 12 GiB Host tier has lost its rank by then
  and is recomputed once (+18 % and +14 %), and with a 1 GiB Host tier the finished conversations'
  lineage hits keep them ahead of new conversations sharing a system prompt (+13 %). The other
  windows tried changed the total by −0.1 % to +0.7 %: the whole capacity and a half slowed 9 and
  7 cases (by up to 38 % and 21 %), a quarter and a sixth 3 each (by up to 12 % and 15 %).
  Capping the hits a lineage passes on at 1, 2 or 4 did not help (+1.1 %, +0.1 %, 0 %).
- A later replay of 266 cases (the workloads above with `D = C + 2`, and a
  document asked about between the turns of a rotation of four to six conversations with 4 to
  12 GiB Host tiers) measured the slot tails of §9.1, step 2's rule for image writes, the
  write-through bound and the periodic half step together: the modelled prefill fell from
  79,045 s to 74,855 s, 42 cases faster by more than 2 %, 8 slower. Leaving one out gives
  78,484 s (slot tails), 77,358 s (image writes), 76,408 s (write-through bound) and 75,548 s (no
  half step). The half step for every branch point, burst or not, gives 75,204 s and 15 slower
  cases (up to +47 %); for periodic branch points only, 75,039 s. The slower cases: a
  57,741-token document before five conversations of 4,400-token turns or six of 8,900 with a
  2 GiB Host tier (+3.5 %, +12.1 %: more conversations resume than there are Device slots, so the
  owners of the slots replace one another); a document asked about after every 24 or 32
  4,400-token turns (+3.2 %, +4.0 %); a document asked about between the turns of four or five
  conversations of 8,900-token turns with 4 or 8 GiB (+3.1 % to +8.7 %); and 24 conversations
  sharing a system prompt with 1 GiB (+6.2 %). In issue #222's order (a 115,210-token document
  asked about ten times, then four conversations of 8,900-token turns) the rotation reused
  80.8 % of its prompt tokens with a 2 GiB Host tier, 5.8 % with 3 GiB and 17.5 % with 4 GiB
  before these rules, and 82.4 %, 82.4 % and 79.9 % with them. Of the cases in which a larger
  Host tier (0 to 12 GiB) was slower by more than 5 %, 6 of 17 remain, by up to 23 %.
- On an RTX 4090 (Qwen3.8-27B IQ3_S, KV k8v4, 131,072-token cache, one request at a time), the
  rotation in issue #222's order reused 889,765 of 1,068,544 prompt tokens with a 2 GiB Host tier
  and 791,976 with 4 GiB without a half step, 649,379 and 444,894 with a half step for every
  branch point, and 880,875 and 863,111 with the rules above. Against the branch-point half step,
  with a document asked about once per round of four conversations and 8 GiB the conversations
  reused 636,405 of 1,091,843 tokens instead of 408,919, and three cases lost: four conversations
  alone with 1 GiB (248,981 of 1,068,538 instead of 364,617), the document asked once per round
  of five conversations with 8 GiB (recomputed five times out of nine instead of twice, 705.3 s
  of prefill instead of 572.8 s), and six conversations of 8,900-token turns after a 57,741-token
  document with 2 GiB (133,511 of 1,602,812 instead of 142,322). Five conversations of
  4,400-token turns after that document reused 65,824 of 659,092 tokens, against 114,178 without
  a half step and 35,149 with the branch-point one.
- Victim selection scans the snapshot table, O(snapshots); valuation walks at most one path per
  affected snapshot. Snapshot count is bounded by `host_cache_bytes / image_bytes` (about 1,300
  for a 230 GiB Host tier with 27B DFlash2 images), so the policy is independent of the tier size:
  a small tier reaches step 2 and 3 often, a large one rarely.
- Pinned objects (`restore_pins`, `active_refs`, in-flight fills) are never evicted. Pinning a
  snapshot also pins its anchor path, so the path nodes keep their copies while it is pinned.

### 9.4 Losing the last copy

A node whose device and host copies are both gone is deleted together with its subtree. Every
snapshot anchored in the subtree is deleted, and `live_snapshots_below` is decremented along the
paths. A node with `active_refs > 0` always has a device copy, so this never affects an active
sequence. The subtree walk runs only in the unbacked cases of §9.1 and §9.2, and is bounded by
the subtree size.

### 9.5 Active guarantee

- Active entitlement is unchanged: the bounded output window, extension at round boundaries, and
  bounded completion when the pool cannot extend (paged-kv §6.2).
- Cached pages never count against it: `available = free + evictable`, so active growth always
  wins.
- Nothing active is ever evicted.

---

## 10. Invariants

1. Each device page group and each host slab has exactly one owner state:
   Free/Active/Cached/Filling, or Free/Node/SnapshotImage/SnapshotTail/Filling.
2. A tree node is immutable: its tokens, extra key and payload never change after insertion.
3. Every node on an active sequence's mapped path has `active_refs > 0` and a device-resident
   copy until the sequence releases it.
4. A snapshot exists only while it is valid (§5.3). Match returns only valid snapshots, and only
   with `F ≤ n − 1`.
5. Exact identity: a match implies exact equality of tokens and extra keys for every block and
   tail token. Hashes never decide equality.
6. A Filling destination is never mapped into a block table, and its readers wait on its event.
   A Filling source is never freed.
7. Taps are bit-identical to an unsplit run ending at the tap, for the same phase.
   Identity-padding for phase alignment is bit-exact.
8. `Disabled` requests leave no nodes or snapshots behind.
9. The worker is the only mutator. Transfers change residency only in `poll_transfers()` or at
   commit.
10. Cost models only rank choices. Capacity and validity checks never depend on predictions.

---

## 11. RTX 5090 / sm_120a optimisation requirements

### 11.1 Streams

- Restores run on a dedicated non-blocking restore stream, ordered after the compute stream's
  queued work (destinations may have been read by earlier kernels), with an event after the
  prelude and after each layer (§6.5).
- Write-through runs on the Program's transfer stream. Every write shares that stream and every
  restore the restore stream, so each list completes in order.
- Stream priorities are not set. The design gave restores the highest and write-through the
  lowest priority; measurement has not shown the need.

### 11.2 KV page transfer

Device plane order is unchanged: one page is `planes` contiguous slices, 64 planes for 27B INT8
plus MTP. The Host slab layout packs a block's Main page with its Host page layout and the backend
page at the next 4 KiB boundary, so a transfer is a pure copy with no requantisation, and restore is
bit-exact for every KV profile.

Implemented: the copy engines. Runs of consecutive device pages whose slab records advance by one
constant pitch (within one pinned chunk and the device's maximum copy pitch) move as one
`cudaMemcpy2DAsync` per plane; a restore orders the planes by model layer.

Not implemented, kept as options if restore bandwidth or SM interference ever matters: an SM copy
kernel reading pinned slabs through mapped pointers (16 B vector accesses, a capped grid so other
lanes' decode keeps its SMs, layer-major work items), and `cudaMemcpyBatchAsync` over coalesced
runs. The design's acceptance bar was ≥ 85 % of the contiguous `cudaMemcpyAsync` peak for 2 MiB
copies in each direction; the copy-engine path reaches 24–27 GB/s on the measuring machine's
PCIe 5.0 x8 link.

### 11.3 Other rules

- A snapshot image copy is a set of Device copies over the image's parts (48 SSM + 48 conv +
  hidden + ring), about 0.25 ms for a 27B DFlash2 image.
- CUDA Graph keys, decode graphs and block-table publication are unchanged. Prefill is not
  graphed.
- All cache bookkeeping is host-side O(1) or O(path). Index arenas are reserved at startup from
  `M`, `D` and `host_cache_bytes`.

### 11.4 Cost coefficients

Admission choice (§6.2) and Host valuation (§9.3) use the Engine's context-cost model: prefill
chunk, token and attention-pair coefficients measured for the artifact's prefill signature, and the
Host-to-Device transfer coefficients of the RTX 5090 class (a local preset can replace them).
Uncalibrated terms keep the index's generic defaults. No separate calibration runs at startup.

---

## 12. Optional features (each gated by its own measurement)

1. **Layer-pipelined restore** (implemented, §6.5). The first prefill pass's layer `l` waits on
   the restore's event for layer `l` (`cudaStreamWaitEvent`), so the restore overlaps compute; the
   admission itself never waits for the copies.
   - Gate: TTFT reduction ≥ 30 ms p50 on host-restore requests of ≥ 50 k tokens.
2. **In-flight prefix coalescing** (implemented, `hybrid_await_sibling`). An admissible FIFO head
   whose prompt shares more tokens with a lane still prefilling than the cache can serve now
   reports `TemporarilyBlocked`, holding no pins, until that lane's snapshot at the divergence
   publishes; it is then admitted from it.
   - Target: the block boundary `⌊d/64⌋·64` below the divergence point `d` (at most `n-1`),
     added to the sibling's remaining plan as an exact tap. A frontier inside a block would
     publish only once the sibling completes that block. The sibling's deepest planned exact tap
     at or below `d` serves instead when it is at least that deep, or when prefilling the tokens
     between them costs no more than one chunk's fixed cost.
   - A snapshot the sibling has captured in the shared prefix but not yet published (its block or
     MTP backend page is incomplete) is also waited for. It publishes within the sibling's next
     chunk.
   - It waits only when the prefill it saves exceeds the split it may add, and when twice the
     sibling's predicted prefill up to the target (other lanes' decode rounds interleave) is within
     half the Engine's queue timeout (`pending_timeout_ms`). The saving is its prefill from its
     own resume point less its prefill from the target: the pass it runs either way is not saved.
     Counting it as saved once made questions that share their first token after a cached state
     wait one token deeper and split the sibling's pass for it. A waiting head stays in the FIFO.
   - Vision prompts do not coalesce: placeholder tokens are equal for different media.
   - Every sibling prefill boundary re-arms admission, and the head stops waiting once the sibling
     passes the target, finishes or is cancelled. So a missing snapshot never blocks it for long.
   - Gate: total prefill tokens reduced on the concurrent-shared-prefix trace, with no TTFT
     regression for other requests.
3. **Output-boundary snapshots** (not implemented). The Frontend's output parser signals structural boundaries
   (after `</think>`, before `<tool_call>`), and the Program snapshots the state at that exact
   committed frontier.
   - Without speculation: D2D of the active slot at the round boundary.
   - With speculation: a two-segment ReplaySSM Fold, `S0 → slot` over the first `j` records, then
     `slot → active` over the rest. This is bit-identical by the Fold contract. Conv and ring taps
     are split as in §7.2.
   - Gate: reuse gain on thinking-stripped agent traces.
4. **KV-write elision** (not implemented) for the recomputed region (§7.5): map existing pages and send those KV
   writes to a scratch page.
   - Gate: measurable page savings under pressure.

---

## 13. Test specification

The test design follows. §13.1 and §13.3 are implemented (`ninfer_prefix_cache_index_test`,
`ninfer_qwen3_5_hybrid_prefix_real_test`, see Status); §13.2 applies to the unimplemented Ops.

### 13.1 Host-only unit tests (`tests/test_prefix_cache_index.cpp`)

- **BlockTree**
  - insert/match with exact tokens and extra keys;
  - a forced constant `lookup_hash` must not produce a false hit;
  - different media digest with the same tokens must miss;
  - tail snapshot matches on exact tail tokens only;
  - `F ≤ n − 1` is enforced;
  - duplicate snapshot publication is a no-op.
- **Liveness**
  - `live_snapshots_below` stays correct under interleaved publish, evict and subtree deletion;
  - losing the last copy deletes exactly the subtree's snapshots.
- **Device LRU**
  - tail-first order after release;
  - a snapshot's tail stays on the Device while the snapshot holds a slot, and becomes evictable
    when it gives the slot up;
  - hits move the whole path to MRU;
  - backed entries are consumed before unbacked;
  - pinned entries are never returned.
- **Host GDSF**
  - hand-computed cases: two siblings, nested ancestor/descendant, frequency;
  - dead KV is reclaimed before any snapshot;
  - the victim's exclusive path is freed;
  - `L` is monotonic.
- **Supersession and admission**
  - a lineage's previous snapshot is superseded, and a hit retains it again;
  - a hit on a tap that only its prompt's next tap superseded makes it a branch point, and
    questions about one document keep resuming from the first request's ladder tap through a Host
    tier its blocks overflow;
  - the first question about a document whose taps pass one Device slot along keeps its newest
    superseded tap through its block write-through and its later image writes, and the next
    question resumes from it;
  - a lane's block write-through displaces no retained snapshot worth more than the lane's most
    valuable retained one, and none when the lane published none;
  - hits count until a third of the block capacity has been published since the last hit, a
    periodic snapshot's half of them for another third (a branch point or not, and none for one hit
    in a burst), and a hit adds to the hits still counted;
  - a superseded snapshot passes the hits it counted to its lineage's next snapshots and their
    claims, and an endpoint supersedes only an endpoint;
  - a `Boundary` snapshot is superseded only while no other conversation continued below it;
  - Device slots go to superseded owners, then to the least valuable Device-only owner for a new
    snapshot worth as much, then to the least recently hit Host-backed owner;
  - a Host image write displaces only snapshots worth no more than the new one.
- **HostSlabPool**
  - exhaustion, reuse and chunked allocation;
  - a slab never crosses chunks.
- **TapPlanner** (property tests over random prompts and boundaries)
  - all taps 64-aligned, in `(F_s, n−1]`, outside media exclusions;
  - deduplicated against the path;
  - priority order and budget respected;
  - the ladder stays within about log2(n/G) + 1 taps;
  - deterministic.
- **Admission arithmetic**
  - candidate fallback order;
  - Blocked vs Infeasible classification;
  - backfill proof on constructed states;
  - quote staleness via `cache_epoch`.

### 13.2 Op qualification (op-development.md contract, real 27B and 35B-A3B shapes)

- **In-pass taps** call the qualified convolution and GDN Ops once per piece with their state
  slots, so they need no Op of their own; the real-artifact restores (§13.3) resume from them.
- **`paged_kv_transfer`**
  - exact bytes for BF16, INT8, FP8, NVFP4 and K8V4, with Main+MTP and Main+DraftFull bundles,
    random page IDs, both directions;
  - bandwidth ≥ 85 % of the contiguous peak;
  - a concurrent-decode interference measurement is recorded.

### 13.3 Real-artifact integration (the design's scenario set)

Each scenario runs for `--kv-dtype` in {bf16, int8, fp8, nvfp4, k8v4} × backend in {none, mtp,
dflash2}. Scenarios marked "full matrix" run for every combination; the rest use int8 × {none,
dflash2}.

| scenario | assertion |
|---|---|
| resume-equivalence (full matrix) | resumed vs root: first-token logits within tolerance; greedy top-1 agreement over 256 tokens ≥ the chunk-decomposition baseline (root with a different `--prefill-chunk`); CausalScoring NLL difference within the same baseline |
| multi-turn exact endpoint | turn k+1 reuses `F_end(k)` exactly, and the tail was COW'd |
| re-rendered history (thinking stripped) | reuse ≥ `floor64(n_k − 1)` |
| edit deep message | reuse = the deepest ladder tap below the edit |
| shared preamble across 8 sessions | device pages for the preamble counted once; all 8 hit |
| concurrency 8, identical prompts arriving together | correct outputs; no double restore of a Filling node |
| host-only restore (D snapshot slots evicted) | same outputs as device resume; `Restoring` observed |
| cancellation mid-prefill | taps and completed blocks published; next identical request reuses them |
| Vision: same image vs different image, identical tokens | hit vs miss; no resume inside a span |
| pressure with active growth | active request extends by evicting cache; bounded completion unchanged |
| one lane, pool = context, derived budget | the repeat of a prompt resumes at its generation opener (§6.3) |
| host cache 0 | device-only cache works; unbacked eviction deletes subtrees correctly |
| `Disabled` request | tree and snapshot counts unchanged |
| protocol usage | `cached_tokens` / `cache_read_input_tokens` equal `reuse_tokens`; streaming `message_start` exact |

Fault injection covers:

- a failed host allocation at startup, which is a clear startup error;
- a forced CUDA error in a transfer job, which gives Engine-wide failure and cleanup with no leaked
  pins (checked by `prefix_cache_stats` after restart);
- `compute-sanitizer --tool memcheck` on one concurrency-8 scenario, for the lifetime claims.


---

## 14. Legacy code and configuration

### 14.1 Code that Hybrid mode bypasses (retained for Legacy mode)

This list records what Hybrid mode does not use. It is deleted only if Legacy mode is later
retired by an explicit decision.

- `src/runtime/engine/context_cache/`: `resource_manager.h`, `materialization_planner.h`,
  `shared_capture_planner.h`, `resource_search.h`, `context_portfolio_value.h`,
  `materialization_budget.h`, `seal_window_claim.h`. `context_cost.*` is reduced to the machine
  model.
- `src/models/qwen3_5/program/planning/`: `pressure.cpp`, `pressure_planner.{h,cpp}`,
  `checkpoint_recovery.cpp`, `resource_projection.h`, `rebuild_work.h`. `request_plan.cpp` is
  reduced to base planning.
- `src/models/qwen3_5/program/transactions/`: `materialization.cpp`, `capture.cpp`, `commit.cpp`.
- `src/models/qwen3_5/program/storage/`: `kv_store.h` (logical pages, address spaces) and
  `host_kv_store.h` are replaced. `state_store.h` is reduced to active and snapshot slots.
  `context.cpp` is rewritten.
- `src/core/host_kv_arena.*` and `src/models/qwen3_5/program/prefix_identity.*`.
- `src/runtime/contract/resources.h`: the pressure, target, candidate, checkpoint-ref and
  revision types.
- Options: Legacy-only `device_state_slots`, `host_state_slots`, `host_kv_capacity_bytes`,
  `max_private_continuations`, `max_shared_prefixes`, `max_long_anchors_per_continuation`,
  `long_anchor_min_spacing_tokens` and their CLI flags are rejected in Hybrid mode.
- Tests asserting owner, catalog or pressure-target behaviour keep running in Legacy mode.

### 14.2 Configuration surface

```cpp
enum class ContextCacheMode : std::uint8_t { Legacy, Hybrid };

struct HybridPrefixCacheOptions {                           // used only when mode == Hybrid
    std::optional<std::uint32_t> device_snapshot_slots;     // --device-snapshot-slots
    bool fit_device_snapshot_slots = false;                 // set when the count is a default
    std::optional<std::uint32_t> max_new_taps;              // --cache-taps-per-request
    std::optional<std::uint32_t> tap_ladder_tokens;         // --cache-tap-ladder (G)
    std::optional<std::uint32_t> tap_min_gap_tokens;        // --cache-tap-min-gap
    std::filesystem::path persistent_file;                  // --prefix-cache-file (§5.5)
    std::string persistent_identity;                        // set by the product binary
    PrefixCacheSaveControl persistent_save;                 // abandons the save (§5.5)
};

struct ContextCacheOptions {                  // existing struct, extended
    ContextCacheMode mode = ContextCacheMode::Legacy; // the Engine's default; ninfer-serve selects Hybrid
    HybridPrefixCacheOptions hybrid;
    // Hybrid mode: host_cache_budget_bytes (--host-cache-mib) sizes the one slab pool.
    // ... existing Legacy fields unchanged ...
};
```

Engine construction (`normalize_engine_options`) resolves every unset Hybrid value, and
`Engine::options()` reports the effective ones. With `C = max_concurrency`, `chunk =
prefill_chunk` and a Host tier present when the budget is nonzero:

| value | default | reason |
|---|---|---|
| `host_cache_budget_bytes` | 8 GiB (`kDefaultHybridHostCacheBytes`); 0 disables the Host tier | the Legacy Host KV default |
| `device_snapshot_slots` | `C + 2` | one resident snapshot per live conversation, one staging slot, and one for a shared prefix or another conversation, which a small Host tier cannot hold (§5.4) |
| `max_new_taps` | 8 with a Host tier, 2 without | without Host, taps would evict other conversations' snapshots |
| `tap_ladder_tokens` | `max(4096, 2·chunk)` | ladder taps land on chunk boundaries; a finer ladder only duplicates them |
| `tap_min_gap_tokens` | `max(1024, chunk)` | the same |
| KV capacity (`ninfer-serve`) | `--kv-capacity auto` unless given | free VRAM becomes Device block cache |

Legacy capacity options (`device_state_slots`, catalogs, anchors) are rejected, and the Legacy
Host pools (`host_state_slots`, `host_kv_capacity_bytes`) are zero in Hybrid mode.

---

## 15. Risks and mitigations

| risk | mitigation |
|---|---|
| Exact taps cost one prefill split each | Only semantic boundaries are exact (explicit, generation opener, structural); the prompt tail and ladder are flexible and cost no split. The proximity rule leaves one exact tap per chat request, and none after an endpoint resume. The pass takes exact taps inside a text chunk (§7.2); vision chunks, tensor parallelism and DFlash/DFlash2 still split at them. |
| Snapshots crowd KV out of host memory | GDSF weighs bytes against recompute seconds on the same scale for both. Dead-KV sweep runs first. |
| Write-through competes with decode | Copy engines only, batched when a lane releases its blocks; decode rounds/s were unchanged with prefetch and write-through active. Write-through can be throttled when decode is active without changing correctness. |
| Unbacked device eviction deletes subtrees | Only happens with no or full host cache. Backed entries are always preferred. |
| Large pinned allocations on Windows | Chunked allocation; the startup ledger reports the resolved size. |
| Hybrid mode ignores session keys | Content addressing reproduces chain reuse; the protocol surface is unchanged in both modes. |
