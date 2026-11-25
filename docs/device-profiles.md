# Device route profiles

Many NInfer operations have several interchangeable schedules: CTA shapes and warp counts of the
small-T attention kernel, the column tiles of the ternary and groupwise projections, whether the
probability-times-value product accumulates in FP16, which prompt-attention kernel prefills. The
compiled tables that choose between them were measured on one card. The best choice depends on the
part: its SM count sets how many CTAs fill a wave, its memory system sets where a kernel stops being
latency-bound, and a schedule that is best on an RTX 5090 can be several times slower on an RTX 3090.

A device route profile records, for one GPU model, the schedule each operation takes per width band
as a measurement found it. The engine installs it before any operation runs; operations without an
entry keep their compiled route.

## Where profiles come from

Up to three layers answer, per route key:

1. **Your profile file.** `$NINFER_DEVICE_PROFILES`, else `$XDG_CACHE_HOME/ninfer/device-profiles.json`,
   else `~/.cache/ninfer/device-profiles.json` (`%LOCALAPPDATA%\ninfer\device-profiles.json` on
   Windows). `--device-profile-path` names another file.
2. **The built-in table.** Profiles measured on 67 GPU models ship inside the binary
   (`src/runtime/engine/device_profiles.json`); the Profile column of the
   [card tables](gpus.md#cards) shows which. Each desktop, workstation and data-center entry was
   measured at the card's default power limit. 48 entries were measured on November 23, 2026 by the
   release image of master 5342f9f, whose one executable serves every card (`cuda13.4` with
   `75;80;86;89;120a`; `cuda12.9` with `70` for the Tesla V100-SXM2-32GB), with that build's route
   catalog. 11 were measured on November 24, 2026 the same way by the release image of master
   6e1937e (`cuda12.9` for the Tesla V100s): the RTX 4070, 4070 Ti, 4070 Ti Super and 4080, RTX
   3090 Ti, RTX A2000 12GB, RTX 2070, 2070 Super and TITAN RTX, Tesla V100-SXM2-16GB and
   V100-FHHL-16GB. The other 8 keep the measurements of earlier builds, because Vast.ai and RunPod
   had none of these cards at its default power limit on either day, or the rented host did not
   start: the RTX 5090 D v2, RTX PRO 4500 Blackwell Server Edition, RTX 3050, RTX A4500, RTX 2060,
   Tesla T4, Tesla V100-PCIE-32GB, and the RTX 4080 Laptop GPU, whose entry comes from a CUDA 12.8
   build on one laptop whose power limit is not recorded. A measured entry keeps the
   built-in bands of the keys calibration does not measure (below), and the RTX 3090 entry also
   keeps its `gguf_vector/*` bands, which whole-model decode chose. Variants that report different
   names have their own entries, such as the PRO 6000's Workstation (600 W), Max-Q (300 W) and
   Server editions. The Turing entries come from the `75` code, the only code that runs on them;
   its route catalog lacks the keys a Turing build never selects.
3. **The compiled route** of each operation, for a key neither names.

For a key both the file and the built-in table name, the file's bands answer up to their last width
and the built-in bands past it. A key the file does not name keeps its built-in bands, so calibrating
an RTX 3090, 4090 or 5090 replaces the keys calibration measures and keeps every other built-in key
(the `unified/*` switches calibration does not measure, below). A key calibration measured and found
the compiled route best is stored with an empty schedule, which overrides a built-in choice for it.

A GPU with neither a file entry nor a built-in one is calibrated once when the engine starts, before
the weights are loaded. For the built-in table this took 88 to 288 seconds on Ampere, Ada and
Blackwell cards with 46 or more SMs, 223 to 375 seconds on smaller ones, 171 to 336 seconds on
Turing cards and 12 seconds on the Tesla V100, and the result is written
to the profile file for later starts. A `70` build runs its own native projection and attention
routes, which no profile key steers, so it calibrates the two-stage GDN switch and the GGUF vector
and MoE load schedules alone.

An entry applies only to the same hardware class (the GPU name and compute capability) and the same
SM count, so a laptop part that shares a desktop part's name but not its SM count is calibrated on its
own. With several GPUs (`--devices`), every distinct device gets its own profile.

`--device-profile auto` (the default) follows the order above, `--device-profile calibrate` measures
again and stores the result over the file's entry, and `--device-profile off` uses the compiled
tables only.

## Stale and outdated profiles

Each calibrated entry records its provenance in a `calibration` block:

```json
{"hardware_class": "nvidia-geforce-rtx-3090-sm86", "multiprocessors": 82, "origin": "ninfer-calibrate on NVIDIA GeForce RTX 3090",
 "calibration": {"route_catalog": "33f546be871d5bec", "ninfer_build": "v0.12.0-41-gabcdef0",
                 "cuda_driver": 12080, "cuda_runtime": 12080, "date": "2026-10-02T09:30:00Z"},
 "routes": {"...": [[32, "r16c8"]]}}
```

`route_catalog` is a digest of every route key calibration measures and of the candidate schedules
it times for each (`src/calibration/route_catalog.cpp`). When a new build adds, renames or drops a
key or a schedule, its digest changes, and an entry of the profile file measured against another
digest is not applied: the engine logs a warning naming both digests and the build that measured
the entry, and the built-in profile (or, for a GPU without one, a fresh calibration at start)
takes its place. `--device-profile calibrate` or `ninfer-calibrate` measures it again. An entry in
the file without a `calibration` block is ignored the same way. The built-in table is compiled with
the binary and is not checked.

The file's format is `ninfer.device-route-profiles` with `schema_version` 2. A file of another
version is not read: the engine logs that it was written by another build and asks for a
recalibration, which replaces a file of an older version as a whole (its entries are unusable by
this build). A file of a newer version is left untouched.

## Calibrating by hand

```bash
ninfer-calibrate --print > my-gpu.json
```

`ninfer-calibrate` measures every route family on the current GPU (`--device N`) and stores the profile
where the engine looks for it (`--out PATH` elsewhere); `--print` writes the stored entry, with its
`calibration` block, to stdout. `--detail` logs every candidate's time, `--only PREFIX` limits the
run to route keys with that prefix, and `--no-ternary`, `--no-groupwise`, `--no-attention` and
`--no-linear-attention` skip whole families. A limited run replaces only the keys it measured: the
stored entry keeps the others when it was measured against the same route catalog, and is replaced
whole otherwise. Close other GPU work first: the measurement assumes an idle device.

Each candidate is timed through the same dispatch an inference call uses, on synthetic weights and
caches of the registered model shapes, with the L2 cache flushed before every sample, as the median
of eleven runs. A candidate replaces the compiled route only when it is at least 3 % faster, the win
survives a second interleaved measurement, and its output matches the compiled route's to within
5 %.

## What is calibrated

| route key | chooses | measured on |
|---|---|---|
| `t2_i8_route` | small-T kernel or prefill GEMM per width, ternary (Bonsai) projections | the GDN layer's projections at widths 16 to 192 |
| `t2_i8_small/<N>x<K>` | row and column tile of the ternary small-T kernel | every ternary projection shape, widths 1 to 32 |
| `q4_q5_attn_input/...`, `q4_q5_gdn_input/...`, `q5_linear_add/...`, `q4_linear_swiglu/...` | fused groupwise (Qwen3.6/3.8) projection schedules | the model shapes, widths 1 to 64 |
| `attn_i8_small/h24/<kv>/w<W>` | warps, CTAs per SM, key block, split QK (`q`) and early fetch (`e`) of the INT8-family small-T attention; for `int8` also `8x2x32p`, the pipelined kernel that keeps V codes INT8 in shared memory and streams the next 32-key tile through two stages | query widths 1 to 8, KV windows 8K, 64K and 262K, for `int8`, `rk8v4`, `rk4v4`, `rk4v4-e8`, `rk2v4-e8` |
| `attn_pv_f16` | FP16 accumulation of the probability-times-value product per key tile | a 1024-token prompt chunk at 32K and a decode step at 131K |
| `attn_pack_gqa` | the standard INT8 prompt kernel with each KV head's query heads packed into its tiles (PackGQA) | a 1024-token prompt chunk at 32K and 131K |
| `attn_prompt_fast` | the fast prompt-attention kernel (rows kept in registers, FP16 PV per tile) | a wave-aligned prompt chunk at 32K and 131K |
| `attn_prompt_split` | whether the fast prompt kernels (the INT8 family and the row-scaled FP8 cache) may split a launch's keys over several CTAs per row block and merge their partial rows (`off`), for calls of more than 64 columns whose row blocks alone would leave SMs idle; the split count comes from a cost model over the SM count | a 256-token continuation at 32K and 131K and a wave-aligned chunk at 131K, where the fast kernel serves prompts |
| `attn_parallel_tiles` | single-row chunked small-T attention over an INT8-family cache as parallel query tiles (one batched append, one split-KV launch over every tile, one reduce) instead of serial fused chunks; widths without an exact 2-8 column tile divisor stay serial; `NINFER_ATTN_PARALLEL_TILES=0\|1` overrides | 16 and 32 verify columns at 32K and 131K |
| `q4_linear_add/5120x<k>` | the small-T Q4 residual projections (`small_t_c8`, `small_t_c16`, `small_t_c32`) for the attention or GDN output (k = 6144) and the MLP down (k = 17408) of a Q4-output artifact | widths 1 to 32 |
| `gguf_vector/q4_k`, `gguf_vector/q6_k` | scalar or cooperative vector loads in the public GGUF Linear Op and SwiGLU, measured on both 17,408x5,120 and 5,120x17,408 plain projections and the 34,816x5,120 gate/up projection | widths 1 to 8; calibration requires both plain and fused sweeps to win. The RTX 3090 profile takes cooperative Q6_K at widths 1 to 8, and Q4_K at width 1, where the fused form alone does not win but whole-model decode does |
| `gguf_matrix/<format>` for `q4_0`, `q5_0`, `q8_0`, `iq4_nl`, `q4_k`, `q5_k`, `q2_0`, `iq4_xs`, `nvfp4`, `ptq1_0` | `async`: the format's integer matrix kernel staging both tiles through `cp.async` at each column tile (16, 32, 64 and 128 columns) (three barriers an iteration instead of four) against the compiled loop, over the 17,408x5,120 and 5,120x17,408 projections in BF16. The formats whose 256-value step is a whole number of 16-byte words copy 16 bytes at a time, Q2_0, IQ4_XS and PTQ1_0 8 bytes; rows whose bytes or start are multiples of 4 bytes only (the 704- and 2,112-wide rows of the 32-value formats) copy 4 bytes. Listed on sm_80, 86, 89 and 120a builds; taken only where the row count is a whole number of 128-row tiles, the row bytes and the weight pointer are multiples of 4 bytes and the shared memory fits (Q4_K takes 92.5 KiB at 128 columns; Q8_0 fits an RTX 3090's 99 KiB only at 64 columns and narrower). The values are those of the compiled loop | widths 9 to 4,096 (9, 16, 17, 32, 33, 64, 65, 128, 129, 256, 512, 768, 1,024, 1,280, 1,536, 2,048, 3,072, 4,096); calibration keeps `async` on a 3% win that holds at 1.5% on a recheck. Past the last band the compiled route runs. Of the 53 built-in sm_80 and newer classes, the 51 whose entries measured these keys all take `async` at some width for Q4_0, Q5_0, Q4_K, Q5_K, IQ4_NL and Q2_0, 20 for Q8_0 and 20 for IQ4_XS; the RTX 5090 D v2 and RTX PRO 4500 Blackwell Server Edition entries predate them. NVFP4 is measured on 48 classes, which all take it at some width and 20 at every width. PTQ1_0 (dense products only, run by the matrix kernel past the vector kernels' 20 columns) is measured on four classes at their default power limits on November 24, 2026: the RTX 5090 and RTX PRO 6000 Blackwell Workstation Edition take `async` from 33 columns, the RTX 3090 from 33 to 1,536 columns and the RTX 4090 at 33 to 64, 66 to 128 and 513 to 4,096 columns; the other classes run the compiled loop |
| `gguf_moe_matrix/<format>` for the same formats and `iq1_m` | the same staging in the routed MoE matrix kernel, over 64 experts of 640x2,560 rows at 8 to 1,024 pairs per expert (8, 16, 17, 32, 33, 48, 64, 65, 128, 256, 512, 1,024; the pairs divided by the expert count set the tile, the narrowest of 16, 32, 64 and 128 columns that holds them); IQ1_M, which has no dense matrix kernel, copies 8 bytes at a time. Taken only for banks whose expert pointers are 16-byte aligned (`MoeTable::aligned16`, which Gemma 4, Flint, Qwen3.8-Flash-Next and Edge0 set for their device banks and for the banks, slots and slot pools that bring host or disk experts to the device, and DeepSeek-V4.1 for the banks and slots that bring its host or disk experts); the row stride sets the copy as in the dense kernel | pairs per expert, 8 to 1,024. The 51 sm_80 and newer classes whose entries measured these keys take `async` at some pair count for every listed format, at every pair count for Q4_0, Q5_0, Q4_K, Q5_K, IQ4_NL, Q2_0 and IQ4_XS (48 for IQ1_M, 5 for Q8_0); NVFP4 is measured on 48 of them, which all take it and 42 at every pair count |
| `gguf_moe_vector/q4_k`, `gguf_moe_vector/q6_k`, `gguf_moe_up/q4_k`, `gguf_moe_up/q6_k` | scalar or cooperative GGUF MoE vector loads, measured with three experts and reordered slots; generic products test separate plain and fused SwiGLU forms on 640x2,560 rows at chunks 1, 2, 4, 8, and decode-up uses 640x2,560 rows at chunks 1, 2, 4 | per-type, per-chunk calibration; the RTX 3090 profile takes cooperative Q6_K products at chunks 1 to 4 and Q6_K decode-up at chunk 4 |
| `gguf_moe_columns/<format>/spread`, `gguf_moe_columns/<format>/whole` for `q8_0`, `q4_0`, `q5_0`, `q2_0`, `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`, `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`, `iq3_s`, `iq1_s`, `iq1_m`, `iq4_nl`, `iq4_xs`, `mxfp4`, `nvfp4` | `one`: a MoE vector product of a decode or verification call of 2, 4 or 8 tokens runs its tokens one column a pass instead of all of them at once, which computes the same bits. `spread` keys serve rows whose K leaves 32-value slices past whole warps (Inkling 2's 2,304 and 1,280; Q2_K and IQ2_XS have no such form), `whole` keys the others. Both a [gate; up] product with its down product (Inkling 2, Edge0) and a fused SwiGLU or GeGLU pair with its down product (Gemma 4, Flint, DeepSeek-V4.1, Qwen3.8-Flash-Next) must run faster | 48 experts, every token routing 8 and a call of 8 tokens routing 32 of them once and 16 twice, at 2,304x1,280 (`spread`) and 2,048x1,024 (`whole`), chunks 2, 4 and 8. 58 entries carry these keys, measured at each card's default power limit by the release image of master 6e1937e on November 24, 2026 (the RTX 5090 by #453's build earlier that day). Of the 38 keys an entry takes `one` at some chunk for 38 on the sm_80 cards (33 to 36 at every chunk), 37 on the Tesla V100s (32 to 34), 35 to 37 on Blackwell (15 to 28), 30 to 36 on sm_86 (13 to 30), 26 to 36 on Turing (13 to 21) and 13 to 32 on Ada (0 to 13: the L4 13, the L40 and L40S 15, the RTX 4090 23); a second RTX 5090 chose the same bands for 33 keys, the other five differing at 2 or 4 tokens. All 58 take `one` at some chunk for 12 keys, every IQ1_S and IQ1_M form among them; 6 do for Q8_0's `spread` key and 19 for its `whole` key. The RTX 3090's NVFP4, IQ4_XS, Q4_K, Q5_K and IQ4_NL `spread` keys run the requested chunk against its calibration: Inkling 2's UD-IQ4_XS decoded with DSpark after a 32,070-token prompt at 114.7 tok/s with the calibrated bands and 119.8 without any columns key on each of two RTX 3090s at 350 W (four alternating runs each, every run slower with the bands), and at 119.0 with only the NVFP4 key at the requested chunk. The other 9 classes run the requested chunk: the RTX 5090 D v2, RTX PRO 5000 Blackwell, RTX 5880 Ada, RTX 4080 Laptop GPU, RTX 3050, RTX A4500, RTX 2060, Tesla T4 and Tesla V100-PCIE-32GB |
| `q6_head/248320x5120` | the Q6 vocabulary head's per-row GEMV (`gemv`, widths 1 and 2) or small-T MMA (`small_t`, widths up to 32) | widths 1 to 32 |
| `q8_ksplit/<rows>x<k>` | the four-K-warp form (`kw4`: at most 128 registers a thread, four CTAs an SM) of the 8-column tile of the Q8 G32 K-split contraction, which MTP heads, DFlash2 drafters and Q8 heads run, instead of the route's own schedule | the eleven contraction shapes of those layers, widths 1 to 8 |
| `unified/q4_q5_attn_input`, `unified/q4_q5_gdn_input`, `unified/q4_linear_swiglu`, `unified/q4_linear_add`, `unified/q5_linear_add/5120x<k>` | upstream's routes for the fused groupwise projections over the unified Linear templates (`unified`) instead of this line's own | the model shapes, widths 1 to 64 (the small-T launches only where those are all the switch covers) |
| `unified/nvfp4_attn_input`, `unified/nvfp4_gdn_input`, `unified/nvfp4_linear_add`, `unified/nvfp4_linear_swiglu` | the switch of the NVFP4 fused projections onto the unified Linear templates (`unified`) instead of this line's own chain. Under the A16 policies both chains are registered at every width a fused launch serves, so a key names a chain rather than a width | the model shapes, widths 1 to 1024. The two chains share the T = 1 decode kernel and, above the own small-T ceiling (32 columns; 16 for SwiGLU), the same wide fused A16 MMA, so what the key really decides is the small-T band: this line's own small-T kernels against the unified SIMT and sliced-K ladder |
| `nvfp4_int8/attn_input`, `nvfp4_int8/gdn_input`, `nvfp4_int8/linear_add`, `nvfp4_int8/linear_swiglu` | the same switch for the NVFP4 fused projections under `--nvfp4-w4a8-int`, where the activation is quantised to s8 and the stored E2M1 codes feed the s8 tensor cores. The band chooses the wide mainloop -- this line's own MMQ against the unified grouped ladder -- while T <= 8 runs the matching row-form kernel (MMVQ, or the unified narrow one at the call's exact width) under either answer. Kept apart from the `unified/nvfp4_*` keys above so calibrating one family never overwrites the other | the model shapes, widths 1 to 1024 at all four keys: this family pads its token tile instead of stopping at a small-T ceiling, so the SwiGLU key covers the whole sweep too |
| `unified/q4_linear_topk`, `unified/q8_*`, `unified/fp8_*`, `unified/bf16_*`, `unified/context_kv_materialize` | the same switch for the Q4 top-k head, the Q8 fused projections (attention and GDN inputs, LinearAdd, pair, SwiGLU, top-k, grouped convolution, context-KV materialization), the FP8 fused projections (attention and GDN inputs with their conv forms, LinearAdd, SwiGLU, the FP8 top-k), and the BF16 attention input and LinearAdd | not calibrated: the built-in RTX 3090, 4090 and 5090 profiles carry the widths where each Op's benchmark ran faster on the unified routes over two rounds per table (at least 3 % over the geometric mean, no case more than 3 % slower), and keep them under a calibrated file entry; elsewhere a hand-edited profile entry or `NINFER_LINEAR_ROUTES` sets them. Those benchmarks ran to width 1024: a wider FP8 call with A16 activations (every FP8 call of a build other than `120a`, which alone carries the FP8 A8 routes) takes the unified LinearAdd and SwiGLU, and the unified pure Linear, where no profile band reaches its width, because the legacy A16 routes run it in 4-token passes that each read the whole weight |
| `w4_cublas/<op>`, `w4_cublas/<op>/a8` | with `--prefill-cublas`, whether a prompt call of each projection (`q4_linear_swiglu`, `linear_add/5120x17408`, `linear_add/5120x6144`, `q4_q5_gdn_input`, `q4_q5_attn_input`) takes the cuBLAS route (`on`, `off`): the plain key at widths the integer A8 mainloop cannot take, `/a8` at the multiples of its 128-column tile. Where no band reaches a width, calls of 512 columns and more take the route; no band sends a call of 128 columns or fewer (a decode or verify round) to it | widths 136 to 1,500, and 256 to 2,048 under `/a8` |
| `gdn_two_stage/h<value heads>` | the two-stage GDN prefill (fused Q/K normalization and control preparation, then one FP32-state recurrence that also writes the output) instead of the WY/state-passing/output pipeline | prompts of 16 to 8192 tokens, 48 and 32 value heads |

`NINFER_SMALLT_PV_F16`, `NINFER_PROMPT_PV_F16`, `NINFER_PROMPT_PACK_GQA`, `NINFER_PROMPT_FAST` and
`NINFER_GDN_TWO_STAGE` (`0` or `1`) override the profile for their route; `NINFER_PROMPT_SPLITS=0`
turns the prompt key splits off and `N` forces N splits wherever a call may split; `NINFER_LINEAR_ROUTES`
(`legacy` or `unified`) takes one table for every Linear shape and every `unified/*` switch. `--fast-prefill-kernel` turns the fast prompt kernel on regardless.

## Batch composition

A profile picks each width's fastest schedule, and schedules differ in the order in which they sum.
A step that serves four requests runs the projections at width four and splits the attention
differently from a step that serves one, so a request's numbers depend slightly on what shares its
steps, and a greedy answer can change where its two most likely tokens are nearly tied. In a check
on an RTX 5090 (Ternary Bonsai 2, four lanes), one request sent four times at once matched its lone
answer in one copy of four with the built-in profile, and in all four with `--device-profile off`
and `NINFER_PREFILL_ALIGN=0` (the compiled routes and requested chunks) or without the profile's ternary small-T
routes; four different documents at once matched their lone answers in one or two of four with the
profile and in three or four without it. The answers that differ part at such a word and stay
equivalent: four 60,000-token needle documents served at once returned all twelve codes in order,
with and without speculation, on `rk8v4` and `rk4v4`. Where answers should depend on batch
composition as little as possible, run with `--device-profile off` and `NINFER_PREFILL_ALIGN=0`.

## Sizes derived from the device

Some launch sizes follow the device directly and need no profile:

- **Prefill chunk.** `--prefill-chunk` is adjusted to the nearby multiple of 128 whose
  prompt-attention grid leaves the least of its last wave idle on this GPU (see
  `causal_softmax_attention_prompt_aligned_chunk`); `NINFER_PREFILL_ALIGN=0` keeps the requested
  chunk.
- **Wave counts.** The chunked GDN output kernel and the sparse-MoE prefill size their grids from the
  SM count and the kernel's occupancy, and the small-T attention holds its split count to whole waves
  of the device.

## SparseMoe policy keys

SparseMoe, with resident or streamed experts, reads two optional keys outside automatic
calibration:

- `sparse_moe/prefill/<codec>` (`q4-q5`, `q4-q6`, `q8-q8`, `nvfp4`): a band ending at 1 stores the
  token count from which calls take the prefill route instead of Small-T. Accepted values are 2 to
  47 (13 to 47 for NVFP4); anything else keeps the compiled frontier. The workspace query and the
  launch read the same value.
- `sparse_moe/prefill-tiles/q4-q5` and `/q4-q6`, with T as the axis: `narrow` or `wide` forces the
  prefill tile width; an empty schedule keeps the automatic choice.

The built-in profiles use two of them. The V100-SXM2-32GB takes Small-T up to T=32 for Q8 experts
(frontier 33): eight Qwen3.6-35B-A3B requests with three MTP drafts and 512-token prompts produced
2.2% more tokens per second with identical outputs. The RTX 5090 takes the narrow Q4+Q5 tile at
T=1024 (the default prompt chunk): whole generation ran 0.5% faster with identical outputs.
`ninfer_sparse_moe_bench --candidate` and `ninfer_sparse_moe_test --candidates-only` compare and
qualify the candidates a profile may name.
