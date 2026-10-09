# Qwen3.8-Flash-Next: mathematics and qualification record

- **Target:** HF `Qwen/Qwen3.8-Flash-Next`, `architectures = ["Qwen4ExpForConditionalGeneration"]`, `model_type = "qwen4_exp"`
  (text `qwen4_exp_text`), checkpoint revision used by Strata `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- **Engine:** the NInfer consolidated line; the original design baseline was `f118551f`.
- **Status:** M0–M10 are complete within the agreed scope, including the October 9 M7 acceptance decision below.
  This reference preserves the mathematics, measurements and limitations of the completed implementation.
  Current commands and product behavior are in [the product guide](../qwen3-8-flash-next.md).
  Every number marked "est." is arithmetic, not a measurement.
- **Authority order for the mathematics** (highest first):
  1. transformers `src/transformers/models/qwen4_exp/modular_qwen4_exp.py` (+ generated `modeling_qwen4_exp.py`), main
     at the SHA in `ref/transformers/MAIN_SHA` - the only executable reference written by the model's owners' partners;
  2. vLLM `qwen4_exp` (NVIDIA path, `ref/vllm/nvidia/*`) for MTP (transformers has no MTP) and for fused-kernel
     semantics;
  3. llama.cpp PR #27742 (`ref/llamacpp/qwen4exp.cpp`, arch `qwen4exp`) for the GGUF names and the GGUF-side hash;
  4. Strata (`src/strata`, MIT) for an independently validated CUDA transcription and its pitfalls list.
  Where they disagree this document says so and follows (1).
- **Abbreviations:** `H` = 2560 hidden, `HC` = 4 streams, `R` = the residual stack `(HC, H)`, `LR` = 320, `E` = 512
  experts, `K` = 10 routed per token, `I` = 640 expert width, `T` = tokens in a step (1 decode, ≤ 9 verify, chunk
  for prefill).

References cited below by short path: `ref/transformers/` is transformers' `models/qwen4_exp` (PR #48337),
`ref/vllm/` vLLM's `models/qwen4_exp` (PR #53896), `ref/llamacpp/` llama.cpp PR #27742, `ref/hf_*` the files of the
HF repository, and Strata is github.com/Niko1221/Strata (MIT).

## Qualification record

**Current delivery decision (October 8, 2026).** All model and expert arithmetic runs on the GPU.
The completed native/CPU experiments are retained without further development. The working
artifact family is the published GGUF representation with the existing external IQ4_NL table.
The delivery added missing MTP to the existing artifacts while preserving their text weights,
Vision, tokenizer and table descriptor. Artifact directories were inspected before assembly to
avoid duplicating existing MTP. The private IQ4 table has the same row digest and configuration as the public
table and provides no new representation.

The October 8 header audit confirmed that the public Q2_0, IQ3_S and Coder IQ1_M NInfer models
have no MTP bindings. The private GGUF MTP artifact supplies 28 existing physical objects
(2,791,415,296 bytes) and 1,567 logical bindings; the three target configurations are compatible,
including the Coder target's independently pruned expert bank. Qualified candidates replaced the
files in the same public repositories, as authorized by the user. Large downloads, artifact
assembly and GPU qualification ran on the RTX 3090 pod; MLX repositories were excluded. The CPU pod archived
the n-gram corpus and was deleted with its volume. Temporary transfers use the
existing private `WaveCut/ninfer-cache` bucket. After verified public updates, temporary copies
and the disposable pod with its disk were deleted. Source and measurement receipts were preserved.

Q2, IQ3 and Coder passed host/disk MTP qualification on RTX 3090, including exact fixed-mode
repetition, and replaced the files in their existing public repositories. Their new sizes are
41,327,319,040, 58,521,103,360 and 33,303,628,800 bytes. Readback verified every previous object
and each added MTP object; publication verified remote sizes, SHA-256 and metadata.
Their Vision checks preserve the existing plain-decoding fallback, and subsequent text requests resume MTP.
The first media check incorrectly demanded MTP for media; source inspection identified the
existing restriction and the qualification now reports it explicitly. Q2 MTP was slower than
plain decoding in the short measured workload; the product guide retains every range and the
overlapping-transfer limitation. No MTP speedup is claimed.
The CPU pod and its 160 GB volume were deleted after the completed corpus and Q2 candidate
were archived and their receipts collected. The corpus archive passed a full SHA-256 readback.
After verifying the public Q2 replacement, five temporary Bucket files, including a failed
assembly copy, were deleted (82,654,771,724 bytes). Their metadata and deletion receipt remain local.

The initial broad n-gram corpus contains 3,711,509 training tokens and 1,103,918 held-out
tokens, sampled deterministically from four pinned Common Corpus shards and one pinned
github-code-clean shard. It spans the observed language/domain labels with equal per-group
token ceilings, retaining provenance, sparse groups and an explicit document/repository split.
This is a coverage sample, not a claim to cover every language or represent user traffic.
Source labels remain automatically assigned, and exact snippet deduplication does not remove
every near-duplicate. Python and the native C++ profiler produced byte-identical 104,893,176-byte
profiles containing 26,223,284 ranked rows. Held-out hit rates are 33.79% at 256 MiB and 49.64%
at 4 GiB. Across groups with at least 2,048 held-out tokens, minima are 6.04% and 20.54%; the
aggregate does not establish uniform language coverage. Embedded and explicit profiles passed
the full-model Engine comparison against uncached reads, with actual RAM hits and exact tokens.
The existing public table repository now includes the embedded resource, an explicit profile
sidecar and all qualification reports. Remote sizes, SHA-256 and metadata were verified.
The private bucket retains the corpus and profile with full SHA-256 readbacks. Cache residency stays opt-in.

The two offload paths are host RAM to GPU and disk to GPU. Host mode keeps the expert banks in
RAM and caches frequently used experts on the GPU. Disk mode retains the OS page cache and its
bounded pinned transfer buffer. The user excluded an additional application-managed RAM cache
of experts on October 8. Do not implement or benchmark that extra tier. Retain the existing
device expert cache and both offload paths. This decision concerns expert weights; the separate
M6 n-gram row profile shipped as described above.
No CPU arithmetic fallback or automatic CPU/DMA split is part of this delivery. These decisions
supersede the earlier hybrid-platform proposal below; its completed evidence remains recorded.

| milestone | done | where |
|---|---|---|
| M0 | the text config normalised by the converter and parsed strictly in C++, one shared fixture | `tools/convert/qwen4_exp.py`, `src/models/qwen4_exp/config.*`, `tests/fixtures/qwen4_exp/` |
| M1 | an FP64 reference forward of the text path (PLE, hyper-connections, GDN, QSA, MoE, mixer, head) loading tensors lazily from the checkpoint | `tools/reference/qwen4_exp.py` |
| M2 | the converter's logical parameters for the text component; the PLE n-gram hash constants (held to the checkpoint's stored buffers) and row ids; n-gram table rows read from disk (default) or RAM | `tools/convert/qwen4_exp.py`, `src/models/qwen4_exp/ngram_hash.*`, `ngram_table.*` |
| M3 | `hyper_connection` read/write, `ple_inject`, `ngram_embed_rows` (FP8 row-scale, BF16), the sigmoid gate of `gated_rmsnorm`, `qsa_indexer` (pooled keys, top-512 block selection), `sparse_softmax_attention` (every `--kv-dtype` storage since October 2026), `moe_route` (512-expert top-10 and shared gate), `moe_experts_bf16`, each against its FP64 oracle | `src/ops/{hyper_connection,ple_inject,qsa_indexer,sparse_attention,moe_route,moe_experts}/` |
| M3 | the Ops composed over the checkpoint's first four layers (three GDN, one QSA) and the head on BF16 weights, against the FP64 reference on 16 tokens: stack relative L2 ≈ 1.1e-2 per layer, top-1 agreement 16/16 (RTX 3090) | `tests/models/qwen4_exp/test_slice_real.cpp`, `tools/reference/fetch_slice.py` |

| M4 | the family's own load and execution behind the public Engine (CLI, serving, perplexity): GGUF block banks bound as stored, the stage split over `--devices` (balanced by stored bytes, shared auxiliaries copied to every rank that reads them), one FIFO sequence, decode replaying one CUDA graph per stage | `src/models/qwen4_exp/{model,executor}.*`, `src/runtime/engine/qwen4_exp_core.*` |
| M4 | the GSQ-RCO GGUF releases (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S and the Coder build's 256-expert IQ1_M) imported without requantization, with their n-gram table in the same artifact; perplexity within 0.2% of llama.cpp's on the same windows | `tools/convert/qwen4_exp_gguf.py`, `src/models/qwen4_exp/ngram_component.*`, `docs/qwen3-8-flash-next.md` |
| M4 | expert residency: device banks, pinned host banks with a device expert cache (wide calls copy the uncached routed experts into a device slot pool), or the artifact's files streamed into device slots (parallel reads); `moe_experts_gguf` over expert tables, vector products up to eight tokens and ggml's matrix kernel above that from device memory | `src/models/qwen4_exp/{expert_cache,expert_stream,read_pool}.*`, `src/ops/moe_experts/moe_experts_gguf.cu` |

| M5 | the n-gram table described by every model and stored inside it or in a table artifact of its own, refused when missing or different; up to eight concurrent requests with batched decode (the experts read once per batch), FIFO admission between prefill chunks; the context cache's live and turn-closure reuse of a sequence's recurrent state; structured output through the grammar's token masks; the Qwen3.5 Vision tower from the release's mmproj with three-axis RoPE for media prompts | `src/models/qwen4_exp/ngram_component.*`, `executor.*`, `src/runtime/engine/qwen4_exp_core.*`, `tools/convert/qwen4_exp_gguf.py` |

| M5 | MTP speculative decoding (October 2026): the `mtp` component from Unsloth's MTP GGUFs (`eh_proj` split between blocks into `fc_embedding` and `fc_hidden`, hyper-connections decoded to BF16); the MTP block's own sparse-attention KV, pooled keys and indexer tail, its catch-up over every token the model commits; verification of every decoding request's anchor and drafts in one pass, Gated DeltaNet replay records and their fold, the indexer tails and the PLE history advanced over the kept positions (`ple_inject_record`, `ple_history_advance`); the Qwen3.5 family's acceptance (`speculative_accept_greedy_drafts`) with penalties, grammar masks and logprobs; the context cache's sequence images (KV, pooled keys, recurrent and MTP state) in pinned host memory and in disk files | `tools/convert/qwen4_exp*.py`, `src/models/qwen4_exp/{model,executor}.*`, `src/runtime/engine/qwen4_exp_core.*`, `src/ops/{hyper_connection,ple_inject}/` |
| M5 | 32K/128K public Engine needle checks on two RTX 3090s: 33,024/131,008 input tokens, facts at positions 32,768/98,304, different four-digit facts, plain and MTP K=4, int8 KV, prefix reuse off | `tests/models/qwen4_exp/test_long_context_real.cpp`; October 8, 2026 |

Native Q2 storage/import, vector expert Ops and grouped integer tensor-core prefill are qualified
against independent oracles on October 8. Full-model quality and scoring time were measured;
native does not attain GGUF quality/performance parity. Keep the optimized GGUF artifact route
for workloads where these costs matter; native support remains an explicit artifact choice.
The RadixArk NVFP4 checkpoint was not qualified.
The sparse-attention layers' KV takes every `--kv-dtype` format since October 2026.

### Milestone outcomes (October 2026)

| milestone | disposition | evidence and limits |
|---|---|---|
| M2 | complete within the agreed scope | IQ4_NL is the selected table; native-Q2 slices, exact Q2/Q8 import, materialization, packing and companion binding qualified; full HF-to-Q4/Q5 artifacts excluded; the completed FP8-table experiment was rejected by the user because of its 80% larger table; redundant private artifacts were deleted and measured results retained |
| M3 | Op qualification complete; native quality/performance limitations remain | measured full native-Q2 quality: 391,000 targets, overall PPL +0.2594%, worst window +4.2972%, one scoring pair +14.83% time; BF16/Q2/Q4/Q5/Q6/Q8, grouped A8/A16 MMA, mixed projections and both layouts pass independent FP64 oracles, repeats, graph replay and route boundaries; QSA covers nine KV formats; native does not replace the optimized GGUF artifact route |
| M6 | complete within the agreed scope | the qualified broad profile is published in the existing IQ4 table repository; Linux buffered/direct I/O measured with verified cold first request and two warm repeats: 0–4.096 microseconds of PLE stall per request; NumPy/C++ profile bytes match; embedded/explicit profiles pass full-model Engine token and RAM-hit checks; draft/verify top-2 hints pass B=1/8, K=15 eager/graph checks; cache stays opt-in after first-request regressions; Windows execution excluded by the user |
| M7 | complete under the user's October 9 acceptance decision | all nine requests repeat within each configuration and match across hosts through the first EOS; all 64 tokens match at 128K; post-EOS differences at 4K/32K are accepted and remain recorded; the A6000-route diagnostic leaves the split output unchanged |
| M8 | completed experiment retained; further native/CPU development excluded | DMA=1 evidence remains: native-Q2 logits equal the two-3090 resident run at 32 teacher-forced positions; 31.38 tok/s at K=4 versus 23.98 plain and 35.19 GiB peak RSS on a 256 GB host; CPU mixing 3.11–5.26 tok/s and maximum relative logit L2 0.60 remain disclosed; ongoing delivery uses GPU arithmetic with GGUF RAM/disk offload |
| M9 | complete controls and measured selection; no confidence-floor speed gain established | native grouped A8/A16 MMA improves the prior native routes but retains reported GGUF regressions; fused HC write/read, multi-column HC/GDN/QSA verify and QSA indexer MMA at T>=128 are qualified; BF16 pooled keys rejected by the independent selection oracle; DMA=1 remains selected and draft minimum probability remains zero; HC and QSA gains are measured at Op scope only |

M0, M1, M4 and M10 are complete.

**Scope change (October 8, 2026).** Full HF-to-Q4/Q5 model conversion and its full-model
quality/performance campaign are removed from delivery, including the M8 full-Q4 benchmark.
Existing Q4/Q5 format support and affected Op oracles remain. After reviewing the completed
comparison, the user rejected the FP8 n-gram table because it increases table storage from
28.80 GB to 51.84 GB. IQ4_NL remains the release choice. Further FP8-table work and publication
of its two experimental repositories are excluded. At the user's cleanup request, both rejected
FP8 repositories and the redundant private IQ4 table were deleted from HF; their reports and
prior measurements remain locally. Together their model files totalled 121,291,481,344 bytes.
Retained companion links now point to the identical public IQ4 table. Existing format support
does not add a native FP8 expert mode.
Windows execution is excluded because the user has no Windows test host. DMA=1 is the delivered
native-host mode and default; CPU mixing below one remains explicitly experimental, with further
full-model numerical qualification outside this delivery. Existing CPU Op oracles and reported
full-model deviations remain part of the evidence.

**State after the final October 8 campaign.** Conversion, independent oracles, cache/MTP,
long-context, native-host logits/throughput, Linux I/O and FP8/IQ4 quality evidence are retained.
DMA=1 is now the default; CLI and serving option tests pass locally, and both affected Engine
validation files pass strict Linux/GCC syntax checks. GPU results used explicit DMA=1 before
that default change; arithmetic is unchanged. The `/stats` counter repair also passed its
contract test and six public requests. All five NInfer rentals are confirmed exited with
stopped desired state at that point. On October 8 the five superseded rentals and their
970 GB of allocated disks were deleted after verifying local results and the four retained
private model archives. The native and FP8 private HF model cards were
updated and read back, with unchanged payload sizes/digests and private visibility.

Public MTP replacements and the qualified hot profile are complete. Source delivery is tracked in
[PR #14](https://github.com/iamwavecut/ninfer-all/pull/14). The final RTX 3090 rental and its 220 GB
disk were deleted after verified publication and local collection of every result and source snapshot.
Automatic CPU/DMA split calibration, Windows qualification, further CPU-mixing refinement and
optional router lookahead remain outside this completed scope. Native scoring/kernel regressions
remain visible. The retained pod harness reproduces the recorded experiments.
An October 8 inventory found a retained two-3090 rental running again after a reported stop.
The harness had accepted `actual_status=exited` without canceling `intended_status=running`;
explicit stop and the guard now require both terminal compute and a stopped desired state.
The queued-start regression suite passes (22 tests). The instance is confirmed exited/stopped,
with files retained. The provider reports 112.4 minutes of uptime, about $1.52 at its $0.81/h
rate; this is an estimate, not a billing receipt or an exact restart timestamp. All its harness
jobs were terminal. A redundant collection was interrupted by shutdown; earlier completed
receipts and the remote files remain available.
The numerical, cancellation and fixed-mode repetition contracts remain in force. Earlier
scoring regressions and post-EOS pipeline differences remain disclosed beside later results.

**Subsequent M9 work.** `--draft-min-p` passed the affected GPU qualification on October 8.
It computes draft confidence through the existing `target_logprobs` Op, retains the full captured
draft chain and shortens verification through the first draft at or below the absolute floor.
The batch uses its largest selected prefix and individual acceptance extents. Replay records
separate allocation capacity from active width, CUDA Graphs are keyed by width, and the floor is
part of disk-cache identity. Zero preserves the fixed-window behavior. Local CLI/server option
tests and strict Linux/GCC syntax checks pass; no throughput improvement is claimed. Real-model
checks passed K=1 and K=4, floors 0/0.3/1, exact fixed-mode repeats, cancellation/recovery and
three concurrent sequences on two GPUs. Floor 1 agrees with K=1. The retained
single-3090 build pod and older two-3090 pod could not resume because provider resources were
unavailable. Pod 54851854 resumed with a 30-minute guard, installed missing build tools, and is
built the affected checks in its 62 GiB tmpfs. The first test launch failed because that mount is
noexec; copying the completed executables to disk fixed the launch, and the checks passed at
20:35:47 UTC. The routine is `scripts/pods/draft_confidence_checks.sh`; draft-confidence calibration
uses the conservative zero default. On the published Q2 representation on one RTX 3090,
floors 0 and 0.3 produced the same tokens, draft counts and acceptance for two short prompts,
with three repetitions each. This limited sweep establishes no benefit for 0.3. Keep zero rather
than claiming calibrated acceleration. No further native-format performance campaign is scheduled.

**M6 corpus decision (October 8, 2026).** The user retained the representative n-gram profile
and requested broad coverage of languages, prose and code, without preferred languages.
Use pinned public sources and deterministic samples across language, collection and programming
language groups. Bound each site's, author's or repository's contribution. Separate profile
construction and held-out evaluation by source document or repository. Record the resulting
mixture, missing groups and per-group hit rates at the same RAM budgets. This measures row reuse,
not model quality. A favorable aggregate must not conceal poor coverage in individual groups.
The corpus was built on the CPU Pod; the profile was built on the RTX 3090 host after the CPU
Pod was deleted. Reuse the existing profiler and keep caching opt-in
until the measured Engine benefit supports another choice.

The five-stratum raw frequency corpus could not be recovered from this checkout, its two Claude
worktrees, the neighboring NInfer checkouts or `/home/wavecut/services/ninfer-3090/tools/freq_corpus`
on aifarm. Commit `eea471ce5600402914091ee4f0aa8d859a459966` explicitly leaves the raw corpora ignored;
the retained ranking counts cannot reconstruct token order for n-gram profiling. The broad
replacement corpus above supersedes the proposed scope exclusion.

The following records the completed, subsequently rejected FP8-table experiment. The full FP8
table was built from the pinned BF16 source's 128 n-gram tensors on October 8.
It has 320,001,536 rows of 162 bytes and occupies 51,840,252,928 bytes as a v3 artifact,
80% more row storage than the 90-byte IQ4_NL representation. Its native-Q2/MTP companion
copies the existing weights and resources unchanged and replaces the table descriptor.
Both private HF uploads passed privacy, size, SHA-256 and metadata readback checks.
The full FP8 table then passed the public Engine hybrid test with direct I/O: three exact
fixed-mode repeats each for plain and MTP K=4, active cancellation/recovery, routing-profile
recording and nonzero table reads. The RTX 3090 / EPYC 7302P workload used 128 input tokens,
16 generated tokens, int8 KV, a 4 GiB expert cache, DMA share 0.5 and eight CPU workers.
Plain request times were 9.34945/11.7956/11.0541 s; K=4 times were
8.3734/8.46306/8.37219 s. This is functional evidence, not a comparable FP8 quality or
throughput advantage over IQ4.

The final FP8/IQ4 comparison used the same executable and native weights on two RTX 3090s,
BF16 KV, context 32,768, stride 16,384 and the held-out corpus's 391,000 targets, six streams
and 18 windows. IQ4 PPL is 5.04508077; FP8 is 4.98215281 (-1.2473%). Fourteen windows improve
and four worsen, with worst +0.2624% (first GitHub-code window) and best -13.2749% (last arXiv
window). Both code-domain aggregates worsen about 0.105%. Scoring is 443.49/446.33 seconds
(FP8 +0.6419%); one sequential pair cannot establish a stable speed difference. The current
IQ4 baseline differs from the earlier A6000 native result (5.00450069); changed hardware and
source snapshot prevent attributing that difference to any one optimization. The FP8 quality
gain does not remove the earlier native-versus-GGUF regressions. Full domain results are in
the product guide.

The M8 integration registers native host weight objects separately on Linux and copies their
planes directly; other platforms use pageable weights and bounded pinned copy staging.
Routing profiles select the initial cached experts, bind counts to the artifact and layer shapes,
and can record updated counts at request boundaries. A mapped 32-bit release/acquire handshake
connects the CPU supervisor to eager and captured decode/verify/MTP execution. Preparation,
CPU completion and retirement keep the shared staging buffers live through their last GPU
consumer. The public options select DMA share, CPU workers and adaptive caching; defaults
keep the cache and decode split fixed. The context-image directory includes the CPU backend
and resolved cache placement. A cancellation drains CPU work, transfers and n-gram reads,
then discards the incomplete sequence rather than caching it. The whole-expert FP64 test covers
Q2/Q4/Q5, CPU-only and DMA-only misses, mixed hits, adaptive swaps and T=1/3/16/17; it now
also captures repeated exchanges and cancellation/recovery with a deliberately delayed worker.
The CUDA integration passed on an RTX 3090 with CUDA 13.1 and an EPYC 7302P host (256 GB RAM).
The first build exposed an undefined error-check macro in host registration; that was fixed.
The first mixed CPU/DMA test timed out: first-use kernel loading held the CUDA driver lock
while the GPU waited for preparation and the supervisor tried to submit a copy. Eager exchanges
now wait on the host and retire before returning; captured exchanges retain GPU waits.
The affected hybrid oracle/graph/cancellation test then passed in 2.86 seconds with lazy loading
enabled. Registered materialization, masked native GPU routes, scalar/AVX2 CPU experts and the
FP32 residual merge also passed. EPYC 9354 / GCC 13.3 later passed 33 AVX-512 VNNI oracle cases.
The final BF16 full-model comparison at prefixes 128/2,112, 16 teacher-forced positions each,
found all logit words identical between host DMA=1 on one RTX 3090 and resident execution on
two RTX 3090s. DMA=0/0.5 repeated exactly and retained top-1 at all positions, but maximum
relative L2 was 0.324/0.604, with maximum KL 0.00761/0.00194. Those CPU modes remain experimental.
The final public serving grid uses 219 input and 64 output tokens, int8 KV, a 4 GiB cache,
eight CPU workers and three requests per mode. DMA=1/K=4 achieves 31.38 tok/s median versus
23.98 plain; CPU DMA=0/0.5 achieves 3.11/5.26. All generated tokens match across this grid.
Peak RSS is 35.19 GiB for K=4 and 32.68 GiB plain; the tested 256 GB host exceeds the 64 GB
Plan B target. The GGUF baseline failed at startup, so these rates do not establish a gain
over optimized GGUF. Full distributions and measurement conditions are in the product guide.
The full-model test then exposed an unnecessary requirement that the GGUF MTP companion also
use native weights. Native banks now map to the hybrid scheduler independently, and GGUF banks
retain their mapped-host GPU route. After that fix, public Engine plain/K=4 runs passed three
fixed-mode repeats each, cancellation after generated tokens, recovery and routing-profile
recording. The workload was 128 input and 16 output tokens, int8 KV, a 4 GiB cache, DMA share
0.5 and eight CPU workers. FP8 source downloads ran concurrently, so these request times are
not an isolated throughput comparison.

Per-step n-gram lookahead copies the draft winner and runner-up to distinct pinned slots on an
auxiliary stream and hashes private contexts while later draft steps run. Verification also
hints the two candidates after each actual verified prefix; predicted tokens never advance the
next column's context. The GPU test covers eager execution, graph replay, B=1/8, every candidate
at each of 15 steps and 16 verify columns, EOS, error propagation and recovery. An initial host
synchronization of a captured CUDA event failed with `cudaErrorInvalidValue`; completion now
uses the joined compute stream, and the affected test passes in 0.73 s. Exact BF16 top-2
selection retains its separate independent oracle at the real vocabulary width.

Direct I/O uses one bounded speculative batch with separate staging and a separate read queue.
Only complete successful batches become visible; hints never admit rows into the persistent
cache. Linux native-queue and positioned-read fallback tests cover exact rows, failure/retry,
bounded depth, eviction and destruction during pending reads. The updated reader test passes
on the RTX 3090 host too (0.28 s), and the full FP8-table Engine check passes. Buffered/mapped
OS hints remain advisory. The final public Engine I/O run verifies table-page eviction before
the first request and two warm repeats, with DMA=1/K=4 and no persistent row cache. Buffered
and direct reads incur 0–4.096 microseconds of PLE stall per request on the rental's Samsung SSD;
the product guide retains all three decode rates and counters for each mode. `/stats` initially
omitted n-gram counters; the omission and report contract were fixed before remeasuring those
two modes. This qualifies the measured row-read stall, not an isolated speedup from lookahead.
The representative hot profile subsequently shipped as described above; Windows execution is excluded.

Embedded hot profiles now use the existing raw-resource framing. Conversion accepts a default
`ngram.hot` beside the checkpoint or `--resource ngram.hot=PATH` and stores it only with table
rows. `ram-hot` reads the selected table's profile unless an explicit profile overrides it;
other modes leave the resource unread. Hash mismatch, duplicate/out-of-range rows and malformed
framing are refused. Python conversion tests pass (49 affected cases; 36 rechecked after the
no-PLE default-profile fix), as do the Linux CPU reader/profile checks and strict host syntax
checks of the loader, Engine construction and component tests. Two test-run failures, a local
variable collision and a read-only working directory, were fixed before the CPU pass.
The C++ component/resource tests and Python-writer/C++-reader/GPU interop pass on RTX 3090;
the interop check took 3.02 seconds. Its initial failure was a missing pytest dependency,
now included in bootstrap. Public Engine qualification subsequently passed for the broad profile,
as recorded above. The five-stratum corpus is archived in the private Bucket; the earlier
technical-text profile's 25.5% held-out hit rate was not evidence for a representative default.
Older private table archives predate the embedded resource; the existing public table now contains it.

Fused HC write/read preserves the public FP32 stack rounding, supports BF16/FP32 branch output,
and allows the previous and next injection planes to share storage. The text and MTP executors
use it between attention and MLP. Its independent FP64 oracle covers T=1..9 and 37, both output
dtypes, aliasing, guards and changed-input graph replay; the GPU test passes in 3.04 s.
Against separate public write/read calls on RTX 3090 / CUDA 13.1, 61 paired graph samples per
cell cover T=1/4/8/9/16/128/512, both dtypes, and reused/64 MiB-evicted L2 inputs. Medians fall
2.0–6.6% in 26 of 28 cells; two T=1 cells tie and none regress. Each graph loses one node.
The full FP8-table Engine check also passes; no whole-model speedup has been measured.

The QSA indexer keeps its original SIMT scoring below 128 columns. At T>=128 it prepares
the query once and scores four queries against 64 pooled keys per CTA using three TF32
high/residual products. Public pooled keys remain FP32. The independent FP64 selection
oracle passes in 10.67 seconds, including T=127/128/129, capacities 512/700/8192/32768,
dense-to-sparse transitions, exact ties, guards and changed-position graph replay.
On RTX 3090 / CUDA 13.1, 33 rotated samples for each of the original, staged-SIMT and MMA
candidates cover reused and 64 MiB-evicted L2. MMA lowers all 12 T=128/512 medians by
30.8–65.0%; short calls regress by up to 40%, so their original route is retained.
The final public Op check confirms 30.8–66.2% lower wide medians against the retained original
measurements. Workspace grows by 2048 bytes per token and the wide graph gains one node.
The staged-SIMT alternative was removed. No whole-model or other-GPU gain is claimed.

The BF16 pooled-key experiment is closed without changing the stored state. Across three
seeds and 32 queries each at 4K/32K/128K, 167/288 selections change and 166 changes exceed
the existing near-threshold tolerance. The largest changed score distance is 0.004997
relative to the threshold. These synthetic represented-input results reject that candidate;
they are not a model-quality measurement. Existing HC/GDN/QSA verify paths already process
multiple columns, with independent Op checks and the completed K=1/4/8/15 Engine suite.

**Qualification contract (October 8, 2026).** The agreed contract is numerical correctness against
independent mathematical oracles and exact repetition with the execution configuration fixed.
Changing verification width, prefill partition or batch composition can change reductions and a near-tie argmax;
plain-versus-MTP token differences must be reported, but are not a byte-identity requirement.
Sampling RNG purposes also differ between plain and speculative decoding. Exact context-image
restore remains required for the same prefix computation history and execution profile, including
n-gram enablement. The native Q2 cache check initially compared cold 512+218 prefill against a
512+188 prefix followed by 30 tokens: their eight-token answers differ. Live-prefix and disk-prefix
continuations agree exactly; the test now checks that required identity and records the cold
difference separately, rather than treating a change in prefill partition as a restore failure.
The complete native Q2 Engine suite then passed with `target_differences=0`,
`batch_differences=5`, `failures=0`: K=1/4/8/15, output/thinking budgets, seeded repeats,
cancellation, logprobs, disk restore and cache-profile isolation. This does not establish
full-model quality or throughput. The converter/MTP/FP8 source/table suite has 131 passing tests;
the live range-fetch check retrieved one 5,120-byte HF MTP norm and verified its file hash,
not the entire MTP block.

The October 8 disk-cache qualification passed exact row checks (including split rows, eviction
and failed-batch admission) and 12 public Engine comparisons: plain/MTP K=4, buffered reads with
no cache or a 4 GiB cache, direct reads with a 4 GiB cache, and two requests per mode. Repeated
cached requests hit every row. The first buffered requests were 9.55%/11.40% slower with the cache;
repeat generation times differed by less than 0.4%. A separate old/new CLI pair took
0.795/0.810 seconds with identical answers. These are single samples, not a timing distribution.
The disk cache therefore defaults to zero; `ram-hot` retains its 4096 MiB default. Capacity includes
the cache index and is not RSS. Windows remains untested. Full results and measurement boundaries
are in the product guide; the later Linux queue/lookahead qualification is described above.
The rebuilt CLI passes its option tests and a public Engine request confirms zero disk-cache
capacity and zero rows served from the row cache when no budget is specified.

The first build exhausted the 180 GiB volume after 14m24s; logs contain ENOSPC and NVCC crashes,
with no cgroup OOM event. Model files were retained. Removing download caches, compressing the
previous benchmark executable and moving compiler scratch to tmpfs allowed the retry to finish
in 13m01s. Fatbinary compression now uses `balance`; the failed attempt is not a valid speed
baseline. The subsequent test wrapper lacked GNU `time` after its Engine checks had passed;
only the comparison was resumed, without a peak-RSS measurement.

The next A16 pass replaces wide scalar expert dots with grouped BF16 tensor-core dots,
keeping integer weight codes exact and carrying the down input in high/residual components.
The unchanged independent oracle, layout/mixed-bank and graph/repeat checks passed (12.22 s).
Against a retained vector executable on RTX 3090, 13/14 wide real-Q2 cells improved by
41.2–89.5%; T=16/cyclic regressed 2.2%. T=512 changed from 30.9/31.9 ms to 3.27/3.35 ms,
but A16 is still 50.9–120.3% slower than GGUF across the wide cells. Its relative L2 increased
from about 1.6e-7 to 2.4e-6, within the same bounds. Small-width timing differences are
unattributed. The current public Engine run passed with 3 target-mode differences, 7 batch
differences and no failures, compared with 0/5/0 above; no full-model A16 speed or quality
claim follows. Full workload tables and remaining costs are in the product guide.

The FP8 n-gram writer is qualified on October 8 with real row width 160, uneven HF shard
boundaries, multiple writer chunk sizes, and single-file/multipart `.ninfer` artifacts.
The quantizer matches an independent scalar codebook oracle at rounding and FP16 scale
boundaries; the Python-writer/C++-reader/GPU chain matches independent FP64 row products
rounded to BF16, including reordered and repeated row reads. Six affected C++/GPU checks
pass on RTX 3090 with CUDA 13.1. Initial reader-call and test-dispatch mistakes were corrected
before that passing run. This does not measure conversion of the full 102 GB HF table or
its effect on model quality. Hashing and writing make two bounded passes through the source.

The official FP8 source was checked at revision `236dfdf285828023ca3bcd3f37366c58a3469b13`:
its index names individual `experts.<e>.gate_proj/up_proj/down_proj.weight` tensors and
`weight_scale_inv` companions; its config declares 128x128 weight blocks. The six fetched
tensors show BF16 scales, not the FP32 scale storage initially assumed. The earlier wording
"fused FP8 expert sources" assumed the BF16 checkpoint's fused layout and is not the actual
official FP8 layout. The converter now decodes these matrices using their stored multipliers
and preserves FP32 products until the selected encoding's next cast. It also resolves split
versus fused experts against recipe-selected sources. The first check exposed an existing
Qwen3.5-only config validator in the shared builder; Flash-Next source validation now uses
its own parser, including MTP's expert count.
The corrected suite passed 131 tests in 3.07 s, including source overrides and unequal
text/MTP expert counts. Three real matrices (4,915,200 values, 640×2560 gate/up and 2560×640
down) match an independent FP64 E4M3FN codebook/block-product oracle exactly after FP32
rounding, at two read chunk sizes. BF16/F32 scale boundaries, partial blocks, non-finite
values and invalid metadata are covered synthetically. This qualifies source interpretation;
full-model FP8 conversion, throughput and quality remain unmeasured. The FP8 checkpoint's
own n-gram encoding still requires a separate IQ4_NL or BF16 table source.

Intermediate generated models are retained in private Hugging Face model repositories with
conversion reports, source revisions, table dependencies and qualification limits. The pod
harness archives explicit paths, verifies visibility and uploaded bytes, and retains sources;
production promotion and later deletion are separate decisions (`scripts/pods/README.md`).
The initial archives completed on October 8: [GGUF Q2_0 + MTP](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GGUF-Q2_0-MTP-NInfer-exp-20261008)
(69,229,477,120 bytes) and [native Q2 + MTP](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-native-Q2-MTP-NInfer-exp-20261008)
(40,651,086,080 bytes). Private visibility, SHA-256, sizes and all three metadata files were
verified against the uploaded revisions. The native artifact's card names the first artifact
as its external n-gram companion. Full-model production qualification is still incomplete.

The native-Q2 `3..5` and `1..4` slices passed public Engine qualification on RTX A6000 with
CUDA 13.1 and int8 KV: 32/700 input tokens, 16 outputs and exact fixed-mode repeats. The
converter preserved 3,125/4,691 selected bindings and every selected physical object, activation
use and frontend resource relative to the full native artifact. The no-PLE slice reads no table
rows and rejects unused table options; the PLE slice reads the standalone table and refuses a
missing companion. The standalone IQ4 table preserves all source table bytes, verified by digest.
Both slices and the table are private HF archives with verified file sizes, SHA-256 and metadata;
the slices are debug artifacts, not complete models. The first PLE test used a stale binary:
deploying source with its local timestamp let Ninja skip a changed test. The harness now updates
timestamps only when file contents change. A missing JSON include dependency in the M7 test was
also fixed; the rebuilt slice tests pass. These fixes do not qualify full-model HF/FP8 quality.

The October 8 M7 run uses the same native-Q2 artifact and executable on RTX A6000 and two RTX
3090s, with int8 KV, 512-token prefill chunks, no prefix reuse and plain decoding. Three requests
per context generate 64 tokens with model stops disabled. Both configurations repeat exactly;
the 128K outputs also match across hosts. At 4K and 32K, each of the three split samples differs
in 53/64 tokens, starting at zero-based output index 9. All prefixes through the first configured
EOS (token 248046 at index 4) match. The raw 64-token identity check still fails. On October 9,
the user accepted post-EOS differences and closed M7. The accepted criterion is exact fixed-mode
repetition and cross-host equality through the first EOS; all nine comparisons pass it.
`scripts/pods/compare_pipeline.py --eos-token 248046 SINGLE SPLIT` reports acceptance separately
from raw byte identity and retains every full-output difference. The hosts
use different default route profiles and drivers; the completed diagnostic reusing A6000 route
choices on the 3090s leaves every split output unchanged. Those choices are not a 3090 calibration.
Timing and comparison limits are in
the product guide.

October 8 qualification found and fixed premature reuse of the first stage's MTP embedding
buffer during peer copies. The traced 24 repeats per graph mode now have identical valid KV,
MTP intermediates and logits, including alternating delayed copies. Public Engine qualification
without the trace passes too, including 48 delayed-copy repeats and the logprobs/cache checks.
The greedy suite still reports six plain/MTP differences and seven batch-composition differences;
those are recorded under the agreed numerical contract, not hidden by the ordering fix.
Output/thinking budgets and context-image profile
isolation have regression coverage. Plain and MTP K=4 passed the 32K/128K needle checks with
different facts and normal model stops. These are behavioral checks, not long-context throughput
comparisons: the MTP run overlapped CPU conversion and plane verification. The n-gram
cold/warm mode and held-out profile measurements, and the independent CPU/GPU expert comparison,
are recorded in `docs/qwen3-8-flash-next.md`; they do not complete M6 or integrate M8.

### Speculative decoding: what exists (October 2026)

**MTP.** The checkpoint's MTP module (section 1.9) is one QSA layer with a full 512-expert MoE, about 2.6 B parameters
without the shared embedding. Sources beside the BF16 checkpoint's `mtp.*` tensors: Unsloth's MTP GGUFs
(`unsloth/Qwen3.8-Flash-Next-GGUF`, folder `MTP/`: the shared modules in BF16, 5.2 GB, Q8_0, 2.8 GB, and Q4_K_M,
1.9 GB), which the GGUF block path could import as stored; NVIDIA's NVFP4 release, which ships the MTP layer in FP8;
and `quimmedes/Qwen3.8-Flash-Next-MTP-GGUF` (Q4_0 and Q2_K; its earlier files came from a broken conversion). What
others measured: vLLM on two DGX Spark (TP2 + EP, eager, one request, greedy, non-thinking, 256-token answers)
24.9 tok/s without speculation and 50.4 with MTP at four drafts, 3.37 tokens per round (3.00 at three drafts), from
PixelML's drafter card below; Unsloth's llama.cpp branch 1.3-1.7x, 100 to 170 tok/s on one RTX PRO 6000 (its own
docs); one Reddit user reports 95 to 170 tok/s in vLLM (anecdotal). Strata measured per-step acceptance 0.89, 0.86 and
0.85 for the first three drafts.

Expected here: a token reads about 4 GB of weights (Measurements, docs/qwen3-8-flash-next.md), the routed experts
0.66 GB of it. A round of three drafts reads the dense weights once, the routed experts of four columns (up to four
times 0.66 GB) and three MTP steps (one MoE layer and the head each), about 1.8-1.9 times a plain step, for 3.0-3.3
tokens: about 1.6-1.8x decode with every expert on the GPUs. With host or disk experts each extra column brings its own
uncached experts across the bus, so the gain shrinks with the cache's miss rate; on the RTX PRO 6000, whose cache ends
up holding nearly every expert, it should approach the device figure. These are estimates from the byte counts, not
measurements.

**Built (October 2026).** The `mtp` component comes from Unsloth's `shared-` MTP GGUFs (block 48, `nextn.*`); its
norms are stored as `1 + w` (their `w` is BF16-exact, checked on the `shared-Q8_0` file), `eh_proj` is
`[fc_embedding | fc_hidden]` with the embedding's columns first and splits between blocks, and its hyper-connection
matrices keep their Q8_0 blocks, which the HC kernels read. The hidden norm follows vLLM: one RMSNorm over all 10240
values of a cell (llama.cpp normalises per stream). The block runs on the head rank (the last stage); a split looks
the token embedding up on rank 0 and copies the rows over. Its cells are vLLM's: cell i pairs the target's pre-mixer
stack of position i with token i + 1 and rotates at i; the executor advances it over every token the model commits
(prompt chunks, plain decode steps, verified prefixes) and keeps the stack of the last committed position for the
next cell. Drafting is greedy over the public tokens, with the block's QSA run on scratch copies of its indexer tails
(a catch-up redoes those cells), each step selecting its own blocks rather than reusing step 0's. A round verifies
every speculating request's anchor and drafts in one target pass whose Gated DeltaNet layers record their
transitions (`gated_delta_net_replay_record`, the convolution's raw input in the record) and whose QSA indexers and
PLE layer work on scratch state; the commit folds the kept transitions (`GdnReplayFoldPlan`), appends the kept
positions' recorded indexer projections to the live tails, advances the PLE history over the recorded convolution
input, hashes the n-gram context over the kept tokens and runs the MTP catch-up. Pooled keys and KV written for
rejected positions stay where they are: no query reads a block or a position past its own before the sequence
writes it again. A verification of one request replays a graph per stage, and so does the draft chain of one
request on one GPU with its experts there; several requests' rounds, the catch-up and disk experts run eagerly.
Measured on two RTX 3090s with every expert on the GPUs (docs/qwen3-8-flash-next.md, Measurements): a 117-token
coding answer decodes at 92.9-93.0 tok/s plain and 167.8-171.0 tok/s with three drafts (3.87 tokens a
round), 171.9-173.6 with four; with host experts on one card three drafts gain about 12% (three runs each, pinned to the GPU's NUMA node).

**DFlash and DFlash2.** No DFlash2 drafter exists for this model: z-lab and incoai publish theirs for Qwen3.8-27B,
Qwen3.6-35B-A3B and other models, not for Flash-Next, and a Reddit thread asking for one (r/LocalLLaMA, September
2026) points at the training cost. The one public DFlash-family drafter is
`PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash` (DeepSpec DFlash, `Qwen3DSparkModel`, 498 M parameters in five layers,
block 7, taps `[3, 15, 23, 35, 43]` on each tapped layer's HC-contracted 2,560-wide residual, trained on the NVFP4
release's own outputs, embedding and head bound from the target at load). Its card measures +3.9% aggregate
throughput over MTP at four drafts (95% CI +2.1% to +5.8%) in the same vLLM setup, ahead on maths, even on code and
slower on chat, and calls it a research artifact. It would need the same verify path as MTP plus a drafter
architecture of its own here; MTP comes first.

---

## 1. Forward mathematics

### 1.0 Config fields that matter (text_config) and their derived values

| field | value | used by |
|---|---|---|
| `hidden_size` | 2560 | everything |
| `num_hidden_layers` | 48 | `layer_types` = `[lin, lin, lin, full] x 12` (`full_attention_interval` 4): layers 3,7,...,47 are QSA, the other 36 GDN |
| `vocab_size` | 248320 | embed, head, n-gram hash (multiplier bound) |
| `tie_word_embeddings` | **false** | separate `lm_head.weight` |
| `rms_norm_eps` | 1e-6 | all norms |
| `hc_count`, `hc_lowrank` | 4, 320 | hyper-connections |
| `linear_num_key_heads`, `linear_num_value_heads` | 16, 48 | GDN (GQA ratio 3, `repeat_interleave`) |
| `linear_key_head_dim`, `linear_value_head_dim` | 128, 128 | GDN |
| `linear_conv_kernel_dim` | 4 | GDN causal conv |
| `output_gate_type` | `"sigmoid"` | **GDN gated norm activation** (Qwen3.5 uses SiLU here) |
| `mamba_ssm_dtype` | float32 | GDN recurrent state |
| `num_attention_heads`, `num_key_value_heads`, `head_dim` | 24, 2, 256 | QSA main attention |
| `partial_rotary_factor` | 0.25 | rotary dim 64 of 256 |
| `rope_parameters` | theta 1e7, `mrope_section [11,11,10]`, `mrope_interleaved true` | QSA + indexer |
| `indexer_n_heads`, `indexer_kv_heads`, `indexer_head_dim` | 4, 1, 128 | QSA indexer |
| `indexer_compress_ratio`, `indexer_budget` | 4, 2048 | block = 4 tokens; `block_topk` = 512 blocks |
| `num_experts`, `num_experts_per_tok`, `norm_topk_prob` | 512, 10, true (class default) | router |
| `moe_intermediate_size`, `shared_expert_intermediate_size` | 640, 640 | experts |
| `ple_layer_ids` | [2] (**one-indexed**) | PLE on `layer_idx = 1` (GGUF `blk.1`) |
| `ple_embed_dim` | 2560 | 16 heads x 160 |
| `ngram_size`, `heads_per_ngram` | 3, 8 | 16 hash heads (8 bigram + 8 trigram) |
| `ngram_vocab_size_base`, `make_ngram_vocab_size_divisible_by` | 20,000,000, 128 | table rows 320,001,536 |
| `split_ngram_parts` | 128 | checkpoint stores the table as 128 row shards of 2,500,012 rows |
| `ple_conv_kernel_size` | 4 (dilation = `ngram_size` = 3) | PLE conv, history 9 tokens |
| `seed` | 1234 (class default; not in config.json) | n-gram multipliers |
| `eos_token_id` | 248044 (endoftext) | **n-gram cut token**. Note: chat turns end with im_end 248046, which does NOT cut |
| `mtp.num_hidden_layers`, `mtp.layer_types`, `mtp.rope_theta` | 1, `["full_attention"]` (= QSA), 1e7 | MTP |
| `mtp_use_dedicated_embeddings` | false | MTP shares `embed_tokens` and `lm_head` |
| `max_position_embeddings` | 262144 | |

All RMSNorm weights of class `Qwen4ExpTextRMSNorm` / `Qwen3_5RMSNorm` / vLLM `GemmaRMSNorm` are **zero-centred**:
`y = x * rsqrt(mean(x^2) + eps) * (1 + w)`, computed in FP32. The GDN gated norm (`linear_attn.norm.weight`) is the
exception: plain `w` (initialised to ones), as in Qwen3.5. A loader that folds `1 + w` once must do it for exactly the
zero-centred class and not for `linear_attn.norm`.

**There is no final `model.norm`.** The index has no `model.language_model.norm.weight`; the final
`hyper_connection_mixer` (with its own `hc_norm`) is the last op before `lm_head`.

### 1.1 Tensor census (from the 131 safetensors headers; all BF16 unless noted)

Prefix `L.` = `model.language_model.`; `{l}` = layer; `{s}` = 0..127.

| tensor | shape | count | role |
|---|---|---|---|
| `L.embed_tokens.weight` | [248320, 2560] | 1 | token embedding |
| `lm_head.weight` | [248320, 2560] | 1 | head (untied) |
| `L.layers.{l}.attn_hyper_connection.hc_norm.weight` | [10240] | 48 | per-stream zero-centred norm gamma |
| `...attn_hyper_connection.input_mix_weight_down.weight` | [320, 10240] | 48 | HC down |
| `...attn_hyper_connection.input_mix_weight_up.weight` | [10240, 320] | 48 | HC up |
| `...attn_hyper_connection.block_inject_weight.weight` | [4, 10240] | 48 | HC inject |
| `...mlp_hyper_connection.{same four}` | same | 48 each | |
| `L.hyper_connection_mixer.{hc_norm, input_mix_weight_down, input_mix_weight_up}` | [10240], [320,10240], [10240,320] | 1 | final mixer, no inject |
| `...linear_attn.in_proj_qkv.weight` | [10240, 2560] | 36 | q 2048 / k 2048 / v 6144 rows |
| `...linear_attn.in_proj_z.weight` | [6144, 2560] | 36 | output gate |
| `...linear_attn.in_proj_a.weight`, `in_proj_b.weight` | [48, 2560] | 36 each | decay / beta |
| `...linear_attn.A_log`, `dt_bias` | [48] (BF16) | 36 each | |
| `...linear_attn.conv1d.weight` | [10240, 1, 4] | 36 | depthwise causal conv |
| `...linear_attn.norm.weight` | [128] | 36 | gated RMSNorm (plain w) |
| `...linear_attn.out_proj.weight` | [2560, 6144] | 36 | |
| `...self_attn.q_proj.weight` | [12288, 2560] | 12 | per head: q (256) then gate (256) |
| `...self_attn.k_proj.weight`, `v_proj.weight` | [512, 2560] | 12 each | 2 KV heads |
| `...self_attn.q_norm.weight`, `k_norm.weight` | [256] | 12 each | zero-centred |
| `...self_attn.o_proj.weight` | [2560, 6144] | 12 | |
| `...self_attn.indexer.index_qk_proj.weight` | [640, 2560] | 12 | rows 0..511 = 4 q heads, 512..639 = 1 key head |
| `...self_attn.indexer.q_layernorm.weight`, `k_layernorm.weight` | [128] | 12 each | zero-centred |
| `...mlp.gate.weight` | [512, 2560] | 48 | router |
| `...mlp.experts.gate_up_proj` | [512, 1280, 2560] | 48 | per expert rows 0..639 gate, 640..1279 up (`chunk(2)`) |
| `...mlp.experts.down_proj` | [512, 2560, 640] | 48 | |
| `...mlp.shared_expert.{gate,up}_proj.weight` | [640, 2560] | 48 each | |
| `...mlp.shared_expert.down_proj.weight` | [2560, 640] | 48 | |
| `...mlp.shared_expert_gate.weight` | [1, 2560] | 48 | scalar sigmoid gate |
| `L.layers.1.ple.key_proj.weight` | [10240, 2560] | 1 | |
| `L.layers.1.ple.value_proj.weight` | [2560, 2560] | 1 | |
| `L.layers.1.ple.norm_{key,query,conv}.weight` | [10240] | 1 each | grouped (per 2560) zero-centred |
| `L.layers.1.ple.conv1d.weight` | [10240, 1, 4] | 1 | depthwise, dilation 3 |
| `L.layers.1.ple.ple_embedding.ngram_embedding.shard_{s}.weight` | [2500012, 160] | 128 | the table, row-sharded on dim 0 |
| `L.layers.1.ple.ple_embedding.layer_multipliers` | I64 [3] | 1 | hash multipliers |
| `L.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes`, `ngram_heads_offsets` | I64 [16] | 1 each | |
| `mtp.pre_fc_norm_embedding.weight` | [2560] | 1 | zero-centred |
| `mtp.pre_fc_norm_hidden.weight` | [10240] | 1 | zero-centred, **one** norm over 10240 (not grouped) |
| `mtp.fc_embedding.weight`, `mtp.fc_hidden.weight` | [2560, 2560] | 1 each | |
| `mtp.layers.0.*` | one QSA decoder layer (attn+indexer, MoE 512x10 + shared, two HC) | | no PLE, no GDN |
| `mtp.hyper_connection_mixer.*` | as the main mixer | 1 | |
| `model.visual.*` | 27-block ViT, hidden 1152, 16 heads, mlp 4304, patch 16, temporal 2, merge 2, pos_embed [2304,1152], merger 4608->2560 | | identical structure to Qwen3.5-MoE vision; `deepstack_visual_indexes = []` |

Checkpoint total 359,999,963,128 B in 131 shards, which my census reproduces exactly (360.0 GB):

| component | params | BF16 bytes |
|---|---:|---:|
| routed experts (48 x 512 x 3 x 640 x 2560) | 120.80 B | 241.6 GB |
| n-gram table (320,001,536 x 160) | 51.20 B | 102.4 GB |
| GDN mixers (36 x 57.96 M) | 2.087 B | 4.17 GB |
| QSA mixers incl. indexer (12 x 51.45 M) | 0.617 B | 1.23 GB |
| hyper-connections (96 halves x 6.60 M + mixer) | 0.641 B | 1.28 GB |
| embed / lm_head | 0.636 B each | 1.27 GB each |
| routers + shared experts | 0.299 B | 0.60 GB |
| PLE projections + norms + conv | 32.8 M | 66 MB |
| MTP (dense 0.091 B + experts 2.517 B) | 2.61 B | 5.2 GB |
| vision | 0.449 B | 0.90 GB |
| **text model without table** | **125.74 B** ("125B") | 251.5 GB |

Active per token: experts 48 x 10 x 4.915 M = 2.36 B; dense path incl. head 4.31 B; total ~6.7 B ("A6B").

### 1.2 Embedding and the residual stack

```
x0[t]   = embed_tokens[token_t]                               (H)      ; image/video rows replaced by vision features
R[t]    = repeat(x0[t], HC)  -> (HC, H), stream c = x0[t]               (modular L1014: hidden.repeat(1,1,hc_count))
```

`R` is carried in FP32 between layers in every serious implementation (vLLM keeps it in the activation dtype, BF16;
Strata and llama.cpp keep FP32). **Decision for NInfer: FP32 R** (40 KB/token, negligible), because the residual is
summed 97 times and the oracle comparison is FP32.

### 1.3 Hyper-connection (`Qwen4ExpTextGatedResidual`, "gated residual")

One module per half (attention half, MLP half) per layer, plus the final mixer (no inject). Weights per module:
`g` = `hc_norm` [HC*H], `Wd` = `input_mix_weight_down` [LR, HC*H], `Wu` = `input_mix_weight_up` [HC*H, LR],
`Wi` = `block_inject_weight` [HC, HC*H].

**Read (mix)**, for one token, `R` flattened to 10240 with stream-major layout `R[c*H + d]`:
```
xn[c,:]  = R[c,:] * rsqrt(mean_d(R[c,d]^2) + eps) * (1 + g[c,:])        grouped RMSNorm, one group per stream
lo       = silu( (Wd @ xn) / HC )                                         (LR)   <- the /HC is INSIDE silu
gate     = sigmoid( Wu @ lo )                                             (HC*H)
mixed[d] = (1/HC) * sum_c gate[c*H+d] * xn[c,d]                           (H)    block input
w_inj[c] = 2 * sigmoid( (Wi @ xn)[c] / HC )                               (HC)   <- centred on 1
```
**Write (combine)** after the block produced `y` (H):
```
R'[c,d] = R[c,d] + y[d] * w_inj[c]          (the UN-normalised R; y broadcast to every stream, per-stream weight)
```
The final mixer is the read without `w_inj`; its `mixed` is the model's last hidden state (input to `lm_head`).

Decoder layer (modular L801-838):
```
if layer has PLE:  R = R + PLE(R, token ids)                      (section 1.4; applied to the stack, before the read)
(a, R0, w1) = attn_HC.read(R);  y1 = mixer(a)  [GDN or QSA];  R = R0 + y1 (x) w1
(m, R1, w2) = mlp_HC.read(R);   y2 = MoE(m);                   R = R1 + y2 (x) w2
```
Fusion opportunity (vLLM `combine_and_mix`, Strata `fused_gr`): the write of one half is folded into the norm of the
next read, so `R` is written once per half. The read is 3 small GEMVs over 6.6 M BF16 weights (13 MB per half,
1.27 GB per token for 96 halves): at decode this is ~22% of the dense bytes and must be one or two kernels per half,
not six (Strata measured 262 ms/token with a naive kernel, 134x off the bandwidth floor).

### 1.4 PLE: hashed n-gram embedding injected at layer index 1

Applies to `layer_idx = 1` only (`ple_layer_ids` is **one-indexed**: `[2]` -> second layer; GGUF `blk.1.ple_*`). It is
added to the stack **before** that layer's attention hyper-connection read.

**1.4.1 Constants** (all derivable; the checkpoint also stores them as I64 buffers - load and cross-check both):
```
n_heads        = (ngram_size-1) * heads_per_ngram = 16       head h in [0,8): bigram; [8,16): trigram
head_dim_ng    = ple_embed_dim / n_heads = 160
head_vocab[h]  = the (global_h+1)-th prime > base-1  (base = 20,000,000; global_h = ple_layer_index*16 + h, here = h)
               = 20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                 20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171
head_offset[h] = sum_{j<h} head_vocab[j]  = 0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228,
                 140000297, 160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275
rows           = ceil(320,001,446 / 128) * 128 = 320,001,536      (90 trailing pad rows never addressed)
multipliers    = _build_layer_multipliers(vocab=248320, ngram_size=3, ple_layer_index=0, seed=1234):
                   max_long = 2^63-1; half = (max_long // 248320) // 2; base_seed = 1234 + 10007*0
                   m[i] = 2 * (splitmix64((base_seed + 0x9E3779B97F4A7C15*(i+1)) mod 2^64) mod half) + 1
               = [23703573157769, 20109073645365, 8052911324071]  (Strata's transcription of the GGUF metadata)
splitmix64(v): v += 0x9E3779B97F4A7C15; v = (v ^ v>>30) * 0xBF58476D1CE4E5B9; v = (v ^ v>>27) * 0x94D049BB133111EB;
               return v ^ v>>31            (all mod 2^64)
```
The multiplier bound guarantees `token * m < 2^63`, so signed and unsigned 64-bit arithmetic agree; XOR of
non-negatives stays non-negative.

**1.4.2 Context with the EOS cut** (`_shift_right_ignore_eos`, modular L643-657). For token at position p with
predecessors `p-1, p-2`:
```
ctx[0] = tok[p]
for s in 1..2:  ctx[s] = tok[p-s] if p-s >= segment_start(p) else EOS
segment_start(p) = (last position q < p with tok[q] == EOS) + 1, or the sequence start
```
Equivalently (llama.cpp/Strata form): walk predecessors newest first; once a predecessor is EOS or missing, it and every
older one read as EOS. The token's own EOS does not cut its own context. A sequence start is preceded by EOS
(HF initialises the cached context to `[EOS, EOS]`). Padding positions (conv mask) read as EOS. **Image/video
placeholder positions hash with their placeholder ids** (`<|image_pad|>` 248056, `<|video_pad|>` 248057) in HF;
llama.cpp substitutes `ple.image_token_id` for any embedding-batch position, which agrees for images.

**1.4.3 Row ids** (16 per token):
```
mixed2 = (ctx[0]*m[0]) XOR (ctx[1]*m[1])
mixed3 = mixed2 XOR (ctx[2]*m[2])
row[h]      = (mixed2 mod head_vocab[h]) + head_offset[h]          h = 0..7
row[8+h']   = (mixed3 mod head_vocab[8+h']) + head_offset[8+h']    h' = 0..7
```
`mod` is a true modulo by a non-power-of-two prime (not a mask).

**1.4.4 Gather.** `emb[t] = concat_h table[row[h]]` -> 2560, **head-slowest** (head h occupies `[160h, 160h+160)`).

**1.4.5 The PLE block** (`Qwen4ExpTextPLELayer.forward`, modular L763-783), one token, `R` = stack (HC, H):
```
k       = groupnorm_HC( key_proj @ emb , norm_key )          (HC, H)   key_proj [10240, 2560]
v       = value_proj @ emb                                   (H)       value_proj [2560, 2560]
q       = groupnorm_HC( R , norm_query )                     (HC, H)
s[c]    = sum_d k[c,d] * q[c,d] / sqrt(H)
s'[c]   = sign(s[c]) * sqrt(max(|s[c]|, 1e-6))               signed square root
G[c,:]  = sigmoid(s'[c]) * v                                 (HC, H)   "gated value"
N       = groupnorm_HC( G , norm_conv )                      (HC, H)   conv input
conv[c,d] = silu( sum_{k=0..3} W[cH+d, k] * N_{t - 3*(3-k)}[c,d] )      depthwise, dilation 3, taps t-9, t-6, t-3, t
R      <- R + G + conv
```
`groupnorm_HC` = the zero-centred RMSNorm applied per 2560-wide stream. The conv state is the last 9 tokens of `N`
(HF `conv_states[1]`, length `(kernel-1)*dilation = 9`); the n-gram context is the last 2 token ids
(`conv_states[2]`). Both are per-sequence recurrent state and must be checkpointed with the GDN states.
`sign(0) = 0`, so `s = 0` gives gate 0.5.

### 1.5 Gated DeltaNet layers (36 layers)

Identical to Qwen3.5's `Qwen3_5GatedDeltaNet` except the output-gate activation. With `a` = HC-mixed block input (H):
```
qkv   = in_proj_qkv @ a                     (10240)  = q (16x128) | k (16x128) | v (48x128), contiguous runs
qkv   = silu( causal_conv1d_k4(qkv) )       depthwise over 10240 channels, state = last 3 inputs, SiLU on all 10240
z     = in_proj_z @ a                       (48x128)
b     = in_proj_b @ a ;  beta = sigmoid(b)                     (48)
g     = -exp(A_log) * softplus(in_proj_a @ a + dt_bias)        (48)   decay in log space
q,k   = l2norm(q), l2norm(k) per 128-dim head (eps 1e-6);  q *= 1/sqrt(128)
q,k   = repeat_interleave(.., 3) over heads  -> v-head j uses k-head floor(j/3)
S_j   = exp(g_j) * S_j ;  S_j += beta_j * k_j (v_j - S_j^T k_j)^T ;  o_j = S_j^T q_j     (FP32 state 128x128 per v-head)
y_j   = rmsnorm(o_j) * w_norm * SIGMOID(z_j)                   <- Qwen3.5: SiLU(z_j)
out   = out_proj @ y                        (2560)
```
Shapes equal Qwen3.6/3.8-27B's GDN (16 key / 48 value heads, 128 dims), so the existing NInfer GDN kernels apply as
is once the gate activation is a parameter. State per layer: 48 x 128 x 128 FP32 = 3 MiB; conv state 10240 x 3.

### 1.6 Qwen Sparse Attention (QSA) layers (12 layers, and the MTP layer)

**1.6.1 Projections** (block input `a`, position p):
```
qg       = q_proj @ a  -> view (24, 512):  q_h = qg[h, 0:256],  gate_h = qg[h, 256:512]   (per-head halves)
q_h      = rope64( rmsnorm0(q_h, q_norm) )       k = rope64( rmsnorm0(k_proj @ a, k_norm) )   (2, 256)
v        = v_proj @ a                                                                       (2, 256)
iq, ik   = index_qk_proj @ a  -> iq (4, 128) rows 0..511,  ik (128) rows 512..639
iq       = rope64( rmsnorm0(iq, q_layernorm) )        ik is cached RAW (no norm, no rope)
```
`rope64`: rotate-half (NeoX) RoPE on the first 64 dims only (`partial_rotary_factor 0.25`; dims 64..255 / 64..127
pass through), theta 1e7, 32 frequencies. M-RoPE: `mrope_section [11,11,10]` interleaved over the 32 frequency slots
(Qwen3-VL/3.5 interleaved layout: slot i takes T/H/W position by `i mod 3` within the section lengths). For text all
three position ids are equal, so this reduces to ordinary 1-D RoPE; vision positions follow Qwen3.5's 3-D rule
unchanged. The **same 64-dim rotation applies to the 128-dim indexer heads** (config validation requires rotary_dim ≤
indexer_head_dim).

**1.6.2 Indexer and selection** (modular `Qwen4ExpTextQSAIndexer.forward`, L388-474). For query position p, the
visible keys are positions `0..p` (causal). Let `n = p+1`, `nb = floor(n / 4)` complete blocks; block `b` = positions
`4b..4b+3` (blocks are aligned to absolute position 0 of the sequence, not to the chunk).
```
K_b      = rope64_at(pos = 4b)( rmsnorm0( mean_{i<4} ik[4b+i] , k_layernorm ) )     pooled key, rotated at the
                                                                                     block's FIRST position
score_b  = (1/sqrt(128)) * sum_{h<4} relu( iq_h . K_b )                              relu per head, then sum
Sel      = top-min(512, nb) blocks by score   ->  all 4 positions of each selected block
         + the tail positions 4nb..p (0..3 of them), always selected
```
Main attention for query p attends **only** to `Sel` (≤ 2048 + 3 = 2051 positions), causal mask still applied:
```
attn_h = softmax_{j in Sel}( q_h . k_{kv(h)}(j) / 16 ) . v_{kv(h)}(j)      kv(h) = floor(h / 12)
o      = o_proj @ concat_h( attn_h * sigmoid(gate_h) )
```
Consequences:
- Context ≤ 2051 tokens (nb ≤ 512): selection is the identity -> plain dense causal GQA.
- No sliding/local window and no sink beyond the tail rule.
- Pooled keys are deterministic functions of the cached raw keys: an engine caches `K_b` once per completed block
  (FP32 recommended, 512 B per block per layer) plus the raw keys of the incomplete tail (≤ 3 x 128).
- Score scale `1/sqrt(128)` does not change top-k; ties: `torch.topk` order is unspecified. Use "higher score, then
  lower block index" and document it (Strata, llama.cpp do the same).
- **Divergence to avoid:** a cell-level top-2051 over block-broadcast scores (Strata's original `topk_512`) differs from
  HF when the tail is empty (it admits 3 positions of the 513th block). Select **blocks**, then append the tail.
- Prefill: every query row has its own selection (the per-row block-causal mask `4b+3 ≤ p` and its own tail).

**1.6.3 KV and indexer cache per QSA layer:** K, V: 2 heads x 256 x 2 (K,V) = 1024 values/token (2 KiB in BF16,
24 KiB/token over 12 layers); pooled indexer keys 128 FP32 per 4 tokens (128 B/token/layer, 1.5 KiB/token over 12).

### 1.7 MoE (every layer, and the MTP layer)

Qwen3-Next / Qwen3.5-MoE semantics, unchanged (modeling L961-994):
```
p        = softmax(gate.weight @ m)   over 512 (FP32)
(w, ids) = top10(p);  w /= sum(w)                         norm_topk_prob = true
y        = sum_j w_j * down_j( silu(gate_j @ m) * (up_j @ m) )      expert width 640; gate rows first in gate_up_proj
y       += sigmoid(shared_expert_gate @ m) * down_s( silu(gate_s @ m) * (up_s @ m) )
```
The router GEMV is the selection-sensitive step: keep its weight BF16 (or better) and its activation unrounded; Strata
measured that a BF16-rounded activation flips top-10 selections (logit error 8.1e-3).

### 1.8 Final mixer and head

```
h_final = hyper_connection_mixer.read(R)        (no inject, no final RMSNorm)
logits  = lm_head @ h_final                      (248320)
```

### 1.9 MTP (`mtp.*`, one QSA layer, "hybrid": keeps the main model's HC stream count, no PLE)

From vLLM `qwen4_exp_mtp` (`ref/vllm/nvidia/mtp.py`; Strata reports draft acceptance 0.89/0.86/0.85 at steps 1-3
with this transcription). Cell i of the MTP sequence pairs the main model's **final multi-stream residual `R_i`
(before the final mixer)** with token `x_{i+1}`, at rope position i (vLLM convention), and predicts `x_{i+2}`:
```
e   = fc_embedding @ rmsnorm0(embed_tokens[x_{i+1}], pre_fc_norm_embedding)                 (H)
hn  = rmsnorm0(R_i flattened 10240, pre_fc_norm_hidden)    ONE norm over all 10240 values
h   = per stream c: fc_hidden @ hn[c]                                                         (HC, H)
R   = h + broadcast(e)                                     combine with unit weight (no inject)
R   = MTP decoder layer (QSA attention with its own indexer/KV, MoE 512x10 + shared, two HC modules)
out = mtp.hyper_connection_mixer.read(R)  -> lm_head (shared)  -> draft logits
next draft step: R (pre-mixer stack of this step) replaces R_i; token = this step's draft
```
SGLang agrees with vLLM; llama.cpp (`qwen4exp.cpp:581`) applies `pre_fc_norm_hidden` per stream (gamma reshaped
[2560, 4]) and folds `eh_proj = [fc_embedding | fc_hidden]` - treat it as the outlier.
vLLM selects QSA blocks at draft step 0 and reuses that selection for later steps (`set_skip_topk`; the paper says
the same). Strata attends
densely over a 32,768-cell MTP window instead (exact below 2051 cells). Either only changes draft quality, never output.
`mtp.layers.0.mlp.*` is a full 512-expert MoE (2.52 B params): drafting costs as much expert traffic as one main layer
per draft token.

### 1.10 Vision

`model.visual` is the Qwen3.5-MoE vision tower verbatim (27 blocks, 1152 hidden, 16 heads, MLP 4304 GELU-tanh,
patch 16, temporal patch 2, spatial merge 2, learned 2304-entry position embedding with interpolation, merger
LayerNorm -> 4608 -> GELU -> 2560, no deepstack). Image features replace `<|image_pad|>` rows of `x0`; 3-D M-RoPE
positions as Qwen3.5. Preprocessor: `ref/hf_preprocessor_config.json` / `hf_video_preprocessor_config.json`. PLE hashes
the placeholder ids (1.4.2). Phase-later; the existing Qwen3.5 vision overlay should carry over with `out_hidden_size`
2560.

### 1.11 Tokenizer, chat template, special tokens

- Qwen2 BPE tokenizer, vocab 248,320 (`ref/hf_tokenizer_config.json`), same special-token block as Qwen3.5/3.6
  (`<|endoftext|>` 248044, `<|im_start|>` 248045, `<|im_end|>` 248046, vision 248053-248057, `<tool_call>` 248058,
  `<think>`/`</think>` 248068/248069, audio/tts 248070-248076). Check the pretokenize regex against the Qwen3.6
  tokenizer before reusing ours (`ref/hf_tokenizer_config.json: pretokenize_regex`).
- `generation_config`: eos `[248046, 248044]`, pad 248044, `do_sample`, temperature 1.0, top_k 20, top_p 0.95.
- Chat template (`ref/hf_chat_template.jinja`) is the Qwen3.5 XML tool-call template plus a **`reasoning_effort`**
  variable (`xhigh` default, `medium`, `low`) that injects a system-prompt sentence, `enable_thinking`, and
  `preserve_thinking` (default true: past assistant turns keep their `<think>` blocks). The serving layer must pass
  `reasoning_effort` through.

---

## 2. Byte census, residency and memory plans

### 2.1 Stored bytes by representation

| component | params | BF16 | 8-bit (int8 g64 / FP8) | 4.5 bpw (Q4 g64, NVFP4, IQ4_NL) | ~3 bpw (IQ3_XXS class) | ~2.3 bpw (Q2_0 / IQ2_XS class) |
|---|---:|---:|---:|---:|---:|---:|
| routed experts | 120.80 B | 241.6 GB | 128.3 GB | 67.9 GB | 46.2 GB | 34.0-34.9 GB |
| one expert (gate+up+down) | 4.915 M | 9.83 MB | 5.22 MB | 2.76 MB | 1.88 MB | 1.38-1.42 MB |
| experts read per token (480) | 2.36 B | 4.72 GB | 2.51 GB | 1.33 GB | 0.90 GB | 0.66-0.68 GB |
| n-gram table | 51.20 B | 102.4 GB | 51.2 GB (+row scales) | 28.8 GB (90 B/row) | - | - |
| GDN + QSA mixers | 2.70 B | 5.41 GB | 2.87 GB | 1.52 GB | | |
| hyper-connections | 0.641 B | 1.28 GB | 0.68 GB | (not advised) | | |
| router (48 x 512 x 2560) | 63 M | 126 MB | keep BF16 | | | |
| shared experts | 236 M | 472 MB | 251 MB | 133 MB | | |
| lm_head | 0.636 B | 1.27 GB | 0.68 GB | 0.36 GB | | |
| embed_tokens (gather only) | 0.636 B | 1.27 GB | host RAM is enough | | | |
| PLE projections | 32.8 M | 66 MB | | | | |
| MTP dense / MTP experts | 0.09 B / 2.52 B | 0.18 / 5.03 GB | 0.10 / 2.67 GB | - / 1.42 GB | | 0.70 GB |
| vision | 0.45 B | 0.90 GB | | | | |

The published GGUFs (ISTA-DASLab GSQ-RCO, `ref/gguf_gsq_readme.md`) put everything except the table into shard 1:
Q2_0 37.6 GB (2.40 bpw average), IQ2_XS 39.2 GB, IQ3_XXS 47.0 GB, IQ3_S 54.8 GB; shard 2 is the table at IQ4_NL,
28.8 GB, identical across variants. `ffn_down_exps` (640 rows) can only be Q2_0 or IQ4_NL because K/I-quants need
256-wide blocks. They ship no MTP; Strata fetches the 31 `mtp.*` tensors from the BF16 checkpoint by HTTP range.
Official `Qwen/Qwen3.8-Flash-Next-FP8` and community NVFP4 (`nvidia/...-NVFP4`) checkpoints also exist
(`ref/gguf_other_repos.txt`).

### 2.2 What must be touched per token

| item | per decode token | per verify window of T tokens | where it should live |
|---|---|---|---|
| dense mixers + HC + routers + shared experts + head | all (~3.4 GB at W4-dense/BF16-HC, ~5.2 GB at int8-dense) | once (weights shared by T columns) | VRAM |
| routed experts | 480 distinct (~2% of 24,576) | ~U(T) x 480 distinct, U(4) ≈ 2.9 (Strata) | VRAM cache + host RAM |
| n-gram rows | 16 rows (2.5 KB at FP8, 1.4 KB at 4-bit) | 16T rows | disk / page cache / RAM |
| embed row | 1 row | T rows | host RAM (gather + 5 KB H2D) or VRAM |
| QSA KV | ≤ 2051 positions x 12 layers x 2 KiB (BF16) ≈ 48 MiB, bounded | same set + new | VRAM |
| GDN state | 36 x 3 MiB FP32 | read + write | VRAM |

QSA makes attention traffic **flat in context length** past 2051 tokens. The decode cost is dominated by the dense
sweep (once per window) and the expert reads (grow with T).

### 2.3 Per-sequence state and KV sizes

| item | size |
|---|---|
| QSA K+V | 24 KiB/token BF16; ~12.4 KiB int8 g64; 9.6 KiB rk8v4-class (K int8 + V int4) - 12 layers |
| pooled indexer keys | 1.5 KiB/token (FP32, 12 layers) + 3 raw tail keys per layer |
| MTP layer KV + indexer | +1/12 of the above if the MTP keeps full history (or a 32K window as Strata) |
| GDN recurrent | 36 x 48 x 128 x 128 x 4 B = 108 MiB (FP32) |
| GDN conv | 36 x 10240 x 3 x 4 B = 4.2 MiB |
| PLE conv history + n-gram context | 9 x 10240 x 4 B = 360 KiB + 2 token ids |
| context 128K | KV 3.0 GiB BF16 / 1.2 GiB rk8v4; indexer 0.19 GiB |
| context 262K | KV 6.0 GiB BF16 / 2.4 GiB rk8v4; indexer 0.38 GiB |

A checkpoint (StateImage) is 113 MiB + the PLE state; ReplaySSM records are as for 27B.

### 2.4 Memory plans

Assumptions: RTX 3090 24 GB, ~22.5 GB usable after the CUDA context and display; PCIe 4.0 x16 ~25 GB/s; DDR4-3200
dual channel ~40 GB/s effective for streaming CPU kernels (DDR5-6000 ~65 GB/s); NVMe random 4 KiB read ~80-120 us
QD1, >300K IOPS at depth.

**Plan A - 1x 3090 + 128 GB RAM, Q4-class experts (capacity reference, outside this delivery).**
- VRAM: dense ~3.4 GB (W4 g64 GDN/QSA/shared/head, BF16 HC + router) or ~5.2 GB (int8 dense); MTP 1.4 GB (Q4 experts
  in VRAM, dense BF16); KV+state at 128K rk8v4 1.5 GB; prefill workspace 1.5 GB; **expert cache ~13 GB ≈ 4,700 Q4
  experts (19%)**.
- Host: 68 GB experts pinned (per-layer `cudaHostRegister` ranges; see risk R6), table 28.8 GB (4-bit) in RAM or on
  disk, embed 1.3 GB, OS ~10 GB. Fits 128 GB; does not fit 64 GB.
- Est. decode (greedy, MTP window 4, ~2.8 tokens/round, hit rate 0.72 adaptive per Strata's curve at ~4,500 slots):
  CPU misses ≈ 0.28 x 1390 x 2.76 MB ≈ 1.07 GB/round → 27 ms (DDR4) / 16 ms (DDR5); GPU ≈ 6.2 GB/round → 8 ms;
  **≈ 45-65 tok/s DDR4, 60-85 tok/s DDR5** (±40%). For comparison (anecdotal, r/LocalLLaMA): llama.cpp on the same
  class of machine (3090 + 128 GB DDR4, UD-Q4_K_XL, experts on CPU without a GPU cache or MTP) reports ~16 tok/s
  decode and ~190 tok/s prefill.

**Plan B - 1x 3090 + 64 GB RAM, ≤ 2.5 bpw experts (Strata's operating point).**
- Experts 34-35 GB host (imported GGUF Q2_0/IQ2_XS or an own 2-bit groupwise format); table on disk (default) with an
  optional 4-8 GB hot set in RAM; cache ~13 GB ≈ 9,400 experts (38%), hit ≈ 0.8-0.85.
- Est. decode **90-130 tok/s** greedy with MTP; plain decode ~55-75. Strata measured 89-93 tok/s on a 3090 with
  IQ3_XXS on an AVX2-only EPYC VM (`bench/results/2026-09-29-rtx3090-epyc-milan`), its paper estimates ~130-140 for
  Q2_0.
- Q4 experts do **not** fit 64 GB; they would have to be read from SSD through the page cache (Strata's low-RAM mode
  ran a 111 GB file on 64 GB at 7-8.5 tok/s). Not a supported plan.

**Plan C - 2x 3090 via pipeline stages (`--devices 0,1`, Linux), experts VRAM-resident.**
- Each stage owns 24 layers: experts 17 GB at 2.25-2.5 bpw + its dense ~1.7 GB + KV/state; stage 0 also holds the
  head (0.36-0.68 GB), the MTP layer (0.9 GB at 2-bit experts) and round state. ≈ 21-22 GB per card at ≤ 64K context;
  128K needs rk8v4 KV. Host RAM only for the table (disk or RAM) and the embed.
- Stage hand-off carries the **10,240-float stack R** per token (40 KB), not a 2,560-wide hidden: 4x the existing
  payload, still negligible over PCIe.
- Est. decode (all experts resident, per round ≈ 3.4 GB dense + 1.9 GB experts at 800 GB/s ≈ 7 ms + ~3 ms
  launches/attention/drafting): **110-160 tok/s** greedy with MTP; ~70-90 plain. Strata's 5090 (1.8 TB/s, nearly all
  experts cached) measured 165-179 tok/s.
- IQ3-class (46 GB) or Q4 (68 GB) experts on two 3090s need Plan A's hybrid path per stage (each stage caches its own
  layers' experts; the host arena is shared).

**Prefill (all plans).** Compute: ~13.4 GFLOP/token. With experts resident and our A8 int8 routes, est. 2-3K tok/s on
a 3090. In hybrid mode every prefill chunk touches nearly all 24,576 experts; streaming the uncached share over PCIe
costs (1 - cache share) x expert bytes / 25 GB/s per chunk (0.85 s at 2.25 bpw, 2.2 s at Q4) against ~3.3 s of GPU
compute for an 8K chunk, so chunks ≥ 8K hide it: est. **1.5-2.5K tok/s**. Strata measured 2,160 tok/s at 32K on a
3090 with IQ3_XXS.

### 2.5 The n-gram table: storage, residency and option surface (product requirement)

**Requirement.** By default the table is read from disk (positioned reads / memory map through the OS page cache, with
lookahead prefetch); an option loads it, or a profile-selected hot part of it, into RAM.

**Representation in the artifact.** A dedicated **n-gram table section** (Section 3.6 describes the container):
- rows in the HF row-index space (head offsets as above), row-contiguous, no padding between rows; the 90 trailing pad
  rows dropped (`n_rows = 320,001,446` addressed; header records 320,001,536 for cross-checking);
- section payload aligned to 2 MiB in the file (huge-page friendly for `ram`), row index -> byte offset is
  `row * row_bytes` (a 4 KiB page holds 25.6 FP8 rows; ~4% of reads straddle two pages, accepted rather than padding
  rows to 256 B, which would cost +60%);
- formats, chosen by the converter recipe:
  - `fp8_e4m3_rowscale` - 160 B + 2 B FP16 scale per row = 162 B/row, 51.8 GB. The completed experiment
    was rejected because of its larger table; retained format support is not a release recommendation;
  - `int4_g32_iq4nl` - 5 x (FP16 scale + 16 B codes, IQ4_NL codebook, split-half nibbles) = 90 B/row, 28.8 GB. Bit-exact
    import of GGUF `per_layer_token_embd` (IQ4_NL) and the GSQ evaluations were done with it. Selected release format;
  - `bf16` - 320 B/row, 102.4 GB. Reference/oracle only.
  Sources: the HF BF16 shards (the checkpoint stores the table in BF16, 128 row shards), the GSQ GGUF shard 2
  (IQ4_NL), or an existing FP8 table (`nvidia/Qwen3.8-Flash-Next-NVFP4` ships `model-fp8-mtp-ple.safetensors`,
  53.7 GB; `Qwen/Qwen3.8-Flash-Next-FP8`) imported encoded when its scale granularity matches.
- metadata: format, `row_bytes`, `n_rows`, `head_dim_ng` 160, `n_heads` 16, multipliers[3], head_vocab[16],
  head_offset[16], eos id, image/video placeholder ids, a content digest.
- **Placement (hybrid):** the table is described by the model artifact's **`ngram` component** (constants, row
  format, SHA-256 of the rows), and its rows are stored either in the same file, as a self-contained model, or in a
  **table artifact** of their own, a `.ninfer` with the `ngram` component alone, which every quantization shares
  (GSQ ships the same 28.8 GB IQ4_NL table as the second shard of every release, the Coder build's included). The
  runtime reads only what a placement needs: nothing of the table at load, the rows each token addresses afterwards
  (or all of it with `--ngram-ram`). A model stored without its rows takes them from `--ngram-table PATH`; the
  loader refuses a table whose constants, format or digest disagree with the model's, and refuses to start a model
  with no table at all unless `--no-ngram-table` asks for the experimental table-less mode (the PLE injection is
  skipped, equal to an all-zero table; Q2_0 WikiText-2 perplexity 2.66 -> 5.01). The published conversions are
  table-less models plus one table repository. A side companion file without a container came first; a single file
  per model replaced it, and the hybrid replaced that so that several published checkpoints share one table.

**Runtime option surface** (CLI and serve config; names follow our `--kebab` convention). **Status (October
2026):** built as planned below, with the budget as `--ngram-ram-mib N` (default 0 for `disk`, 4096 for
`ram-hot`). `ram-hot` takes the selected table's embedded profile or the explicit file
`ninfer-ngram-profile` writes. Conversion embeds `ngram.hot` when supplied; the published table
now includes the broad profile, qualified through the public Engine as recorded above.
Buffered/direct misses use a core-owned bounded
io_uring/IOCP queue driven by one worker; Linux hosts that disable io_uring retain positioned-read workers.
The Linux queue, reader and direct-I/O public Engine checks pass; the final buffered/direct
serving run measures near-zero PLE stalls with cold and warm table pages. IOCP remains untested;
Windows qualification is excluded. A pass starts its reads when it hashes its tokens and uploads the rows just before the PLE layer, so
the embedding and block 0 run meanwhile. The October 8 implementation adds a budgeted four-way CLOCK row cache
for `disk` (opt-in, including the index; zero disables it) and advisory lookahead for the next prompt
chunk and the top two tokens of each MTP draft step and actual verify prefix. Linux buffered/mapped reads use `WILLNEED`;
Windows mapped reads use `PrefetchVirtualMemory`. Linux row/Engine qualification passes; Windows is untested.
First-request regressions keep the disk cache disabled by default. Direct I/O has bounded
speculative staging; Windows buffered I/O uses the same bounded staging with buffered reads,
untested on Windows. Draft/verify hints pass GPU ordering
and graph checks; the serving run does not isolate their speedup. `ram-hot` keeps its rows in
row order behind a bit per table row (40 MiB) instead of an open-addressing table, and admits nothing online.
The table as planned:

| option | values | default | meaning |
|---|---|---|---|
| `--ngram-residency` | `disk`, `ram`, `ram-hot` | `disk` | where rows come from |
| `--ngram-io` | `buffered`, `direct`, `mmap` | `buffered` | disk mode: `pread` into a bounded row cache through the OS page cache (`posix_fadvise(RANDOM)`; Windows `FILE_FLAG_RANDOM_ACCESS`); `direct` = O_DIRECT / `FILE_FLAG_NO_BUFFERING` 4 KiB-aligned reads that bypass the page cache (Strata's default on Windows: keeps RAM for the expert arena); `mmap` = map + `madvise(MADV_RANDOM)`, prefetch via `madvise(WILLNEED)` / `PrefetchVirtualMemory` |
| `--ngram-ram-mib N` | number | `disk`: 0; `ram-hot`: 4096 | rows plus index: CLOCK cache in `disk` (zero disables), profile-selected resident rows in `ram-hot` |
| `--ngram-hot-profile PATH` | file | built-in profile shipped in the artifact | row-frequency profile for `ram-hot` |
| `--ngram-lock` | flag | off | `mlock`/`VirtualLock` the resident rows (`ram`, `ram-hot`) |
| `--ngram-io-depth N` | int | 64 | outstanding reads (io_uring on Linux, overlapped I/O / IOCP on Windows) |

- `disk`: rows for the next step are requested as soon as their token ids are known (below); a user-space row cache
  (CLOCK, bounded by the budget) absorbs repeats within a conversation; the page cache does the rest.
- `ram`: the whole section is read into anonymous memory at startup (2 MiB pages where available; optional lock).
  28.8 GB (4-bit) or 51.8 GB (FP8).
- `ram-hot`: a hot set of rows chosen **offline from text alone** - the hash depends only on token ids, so the profile
  is built by tokenizing a corpus, computing 16 row ids per token, and counting (no model run). The profile stores row
  ids by descending frequency; at startup the top rows fitting the budget are loaded into an open-addressing table
  (row id -> slot). Misses fall back to `disk`. Optional online admission (CLOCK over the same budget) adapts to the
  conversation. The converter ships a default profile computed from our frequency corpus
  (`tools/freq_corpus`, 5 strata).
- The rows are needed at **layer 1** of the step. Lookahead available:
  - decode with MTP: the window's tokens are the bonus token (known when verify ends) and the drafts (known during
    drafting, ~1-3 ms); issue each token's 16 reads the moment it exists. The GPU runs layer 0 (and the drafting)
    before it needs them; a mapped-memory ready flag waited on by the graph (as Strata's doorbell) is the
    synchronisation, so a late read stalls only layer 1;
  - speculative prefetch: also issue `WILLNEED` hints for the rows of the top-2 candidates of each draft step and of
    the verify row (cheap: 16 page hints per candidate);
  - prefill: the whole chunk's ids are known; issue all `16 x chunk` reads (deduplicated, page-sorted) one chunk
    ahead, overlapped with the previous chunk's GPU work (32K random reads per 2K tokens ≈ 0.1 s at 300K IOPS against
    ~1 s of GPU compute).
- Rows are dequantized on the GPU (2.5 KB H2D per token), never on the CPU, so `fp8`/`int4` stay compact in RAM and on
  the bus.
- Statistics (`/stats`): row requests, row-cache hits, hot-set hits, disk reads, p50/p99 read latency, layer-1 stall
  time.

---

## 3. Original architecture mapping (historical)

This section preserves the original design at `f118551f`, including proposals superseded during
implementation. It is not a list of remaining work or the current architecture contract. Current
ownership and execution are defined in [Engine architecture](engine-architecture.md); the
qualification record above identifies delivered behavior and excluded experiments.

Engine facts this section relies on were read from the tree at `f118551f`. The ones that shape the design:
no plugin registry (`src/models/registry.cpp` knows `Qwen3_5`/`Qwen3_5Moe`; `model_instance.h` L16 fixes
`ModelContract = models::qwen3_5::RuntimeTypes`); every Op is shape-registered and **H = 2560 is registered nowhere**
(only `linear/gguf` with K%512==0 and `linear/bf16/bf16_general` are shape-generic); `sparse_moe` is hard-wired to
256 experts / top-8 / H 2048 / I 512; attention registers [256,24,4] and [256,16,2] only; the residual is one BF16
[H,T] tensor; all text weights are device-resident (`prepare.cpp` refuses other residencies); pinned host memory is
charged against VRAM under WDDM (`program_impl.cpp` ~L363-395); `--devices` is Linux-only; KV pages are 64 tokens;
prefill is eager, decode rounds are captured per (family, B, frontier envelope, topology class, verify width).

### 3.0 Product-scope decisions this model forces (AGENTS.md: these need an explicit product change)

1. **New mathematical architecture** `Qwen4ExpForConditionalGeneration` (README/AGENTS "Product and architecture"
   must name it).
2. **GPU expert execution with host-RAM or disk weights.** The existing GGUF paths support both offload
   placements on one GPU. The earlier CPU/GPU hybrid proposal was implemented as an experiment;
   further CPU expert work is excluded by the current delivery decision.
3. **Host/disk-resident model data on the token path** (the n-gram table), required by the product statement in
   2.5.
4. Concurrency: hybrid mode starts at C = 1 (each extra row multiplies distinct experts and CPU work); GPU-resident
   mode keeps 1..8.

### 3.1 Family layout

Create `src/models/qwen4_exp/` mirroring `qwen3_5`'s split, as an explicit second family (no base class, no
registry):

| path | contents |
|---|---|
| `config.{h,cpp}` | strict parser for `qwen4_exp`/`qwen4_exp_text` keys (1.0); normalises `full_attention` -> QSA; validates the invariants transformers validates (indexer_kv_heads 1, budget % ratio, rotary ≤ indexer dim, PLE on a linear layer, eos set) and the shapes this engine implements (H 2560, 512x10, I 640, HC 4, LR 320, 16x160 n-gram) |
| `load/` | bindings: `text/token_embedding` (Host residency allowed), `text/output_head`, `text/layers/i/{attn_hc,mlp_hc}/{norm,down,up,inject}`, `gdn/*` (as qwen3_5), `qsa/{query_gate,key,value,output,query_norm,key_norm,index_query,index_key,index_query_norm,index_key_norm}`, `moe/{router,shared_score,shared/*,experts bank}`, `ple/{key,value,norm_key,norm_query,norm_conv,conv}`, `text/final_mixer/*`, `mtp/*`; the `ngram/table` binding; expert residency (Device / HostArena) |
| `execution/` | `TextContext` equivalent with an FP32 stack `R [HC,H,T]`; `hc_read`/`hc_write` calls; `gdn_mix` reused from qwen3_5 semantics with the sigmoid gate; `qsa_mix`; `moe` (resident or hybrid); `ple_inject`; `mtp_*` |
| `program/` | Program contract implementation; starts as a trimmed copy of qwen3_5's program (stores, transactions, graphs, prefix cache), with the deltas in 3.5-3.8; shared code moves down to `src/runtime`/`src/core` only where both families need it byte-for-byte |
| `frontend/` | reuse qwen3_5's tokenizer/template/processor code by linking, plus `reasoning_effort`; sampling presets for the new architecture |

`ModelInstance` changes from a single alias to a closed two-member variant (`qwen3_5` | `qwen4_exp`) dispatched
once at construction; the ResourceManager/RequestRecord templates are instantiated for both contracts. This is the
largest structural change outside the model and is a prerequisite for everything else (risk R1).

### 3.2 Op reuse and new Ops

| need | existing Op | status |
|---|---|---|
| GDN recurrence 16/48 d128, chunked prefill, batch update, ReplaySSM record/fold | `gated_delta_net*`, `gdn_replay` | **reuse as is** (27B geometry) |
| GDN conv 10240 = 2048/2048/6144 | `causal_conv1d_silu(_split)` | **reuse** (FN profile is registered) |
| GDN input projection [16384, 2560] (qkv 10240 + z 6144) | `gdn_input_proj` | new shape registration at K 2560 (Q4/Q8/int-A8 routes); measure on sm_86 |
| GDN a/b gating [96, 2560] | `gdn_gating_proj`, `gdn_norm_gating_proj` | new shape |
| GDN output norm with **sigmoid** gate | `gated_rmsnorm` (SiLU) | add a gate-activation parameter (sigmoid) to the contract; oracle update |
| QSA q/gate/k/v/index projections [12288+512+512+640, 2560] | `attn_input_proj` | new geometry; the converter de-interleaves q/gate as for Qwen3.5 |
| q/k norm + 64-dim interleaved M-RoPE at D256; indexer q norm + RoPE at D128 | `rmsnorm`, `rope` (MRoPE D256/R64 registered) | D256 reuse; **D128/R64 MRoPE** for the indexer is a new rope geometry |
| KV append, 2 KV heads, D256, all codecs | `kv_cache_append` | **reuse** |
| sigmoid output gate | fused in attention epilogue / `sigmoid_mul` | reuse |
| RMSNorm (zero-centred / plain) | `rmsnorm(unit_offset)` | reuse; grouped (per-stream) norm is a loop over HC or a new grouped form |
| sampling, argmax, penalties, grammar masks | `sample`, `argmax` | reuse (V 248320 unchanged) |
| speculative accept (greedy, sparse), verify input prep, MTP round helpers | `speculative_round`, `mtp_round` | reuse; MTP bridge payload changes to the 10240-wide stack |
| head [248320, 2560], `linear_topk` | `linear` / `linear_topk` | new shape registrations (Q8/Q4/Q6) |
| embedding [248320, 2560] | `embedding` | new table shape, or a host gather (below) |
| vision tower | qwen3_5 vision (same tower) | reuse; merger output 2560 |

**New Op families** (each with a contract header in `include/ninfer/ops/`, wrapper/launcher/kernel split, FP32/FP64
oracle tests in `tests/ops/`, and a `bench/ops` microbenchmark):

1. **`hyper_connection`** - `hc_read(R, norm, down, up, inject?) -> (mixed, w_inj)` and
   `hc_write(R, y, w_inj) -> R`, plus the fused `hc_write_read` (previous half's write folded into the next read's
   norm, vLLM `combine_and_mix`). Decode: weight-streaming GEMVs over 13 MB per half; multi-column form for verify
   windows (weights read once for T ≤ 16); prefill: two small GEMMs ([T,10240]x[10240,320], [T,320]x[320,10240]).
   Weights BF16 or ggml Q8_0 (`A16Only`: activations stay FP32; the GGUF recipe stores Q8_0 since October 9,
   2026). Oracle: FP64 formula of 1.3 over the values the weights represent.
2. **`ple_inject`** - `(emb [2560,T], R, hist[9,10240]) -> (R', hist')`: key/value projections (BF16 GEMV/GEMM),
   three grouped norms, signed-sqrt sigmoid gate, dilated causal conv (taps t-9/-6/-3/0), SiLU, residual add, history
   advance; plus a `ngram_dequant_rows` gather-decode (FP8-rowscale / IQ4_NL / BF16 rows -> FP32). Prefill form uses
   the chunk's own rows for in-chunk taps. Oracle: FP64 formula of 1.4.5 against independently decoded rows.
3. **`qsa_indexer`** - per QSA layer: `index_append` (raw key -> tail; on block completion pool, norm, rope at
   block start, write the pooled plane), `index_score_select` (decode/verify: score all complete blocks of the row,
   radix top-512 with lower-index tie break, emit selected positions + tail; prefill: per-row block-causal scoring as a
   [chunk x blocks] BF16 MMA over pooled keys, per-row top-512). Output: a position list per (row, layer)
   (≤ 2051 int32). Oracle: exact selected set vs an FP64 reference over the same pooled keys (selection is a set
   equality test with an explicit tie/near-tie tolerance rule: disagreement only allowed where the FP64 scores of
   the swapped blocks differ by < 1e-6 relative).
4. **`sparse_softmax_attention`** - attention of query rows over an explicit selected-position list through the
   paged KV view: GQA [256, 24, 2] (12 q heads share one kv head and one selection, so the 12 heads form the MMA M
   dimension: FlashMLA-sparse style), gate epilogue, all KV codecs we support on sm_86 (BF16, int8 g64, rk8v4 first).
   Decode/verify: split-K over the ≤ 2051 positions; prefill: per-row selections, tile = (row, kv head). When
   `n ≤ 2051` the selection is the identity and the existing dense `causal_softmax_attention` route (new [256,24,2]
   registration) is used instead - same math, oracle-identical.
5. **`sparse_moe_512`** (or a generalisation of `sparse_moe` with a new contract) - router [512,2560] BF16 + shared
   score row (513 rows), softmax-top-10 with renormalisation (our existing "top-k logits then softmax" form is
   mathematically identical), routed SwiGLU over expert banks with per-expert pointers (**indirection table**, so an
   expert may live in a VRAM cache slot or a resident bank), shared expert with sigmoid scalar gate, weighted merge,
   residual-free output (the HC write does the add). Routes: decode T=1, small-T (verify ≤ 16), prefill grouped tiles
   (A8 int8 MMA on sm_86 as for our dense routes). Formats: Q4 g64 gate/up + Q5/Q6 down (as 35B), Q8, and the 2-bit
   `q2_g64` below. Oracle as for `sparse_moe`.
6. **Hybrid MoE runtime (not an Op; Program-owned)** - see 3.4.
7. **`q2_g64_fp16` format** (2.25 bpw, Q2_0-compatible grid `(code-1) * scale`, 64-wide groups, FP16 scale) with
   decode/small-T/prefill GPU kernels and CPU kernels, so GSQ-RCO Q2_0 experts import byte-exactly
   (`import_encoded`). Rationale: Q2_0 is the GSQ build with the best speed (README: 3.4x prompt throughput of
   IQ2_XS) and its grid suits int8 dot products on both GPU (`dp4a`/s8 MMA) and CPU (VNNI). IQ-quant experts remain
   importable through the existing `gguf_*` formats for GPU-resident Plan C, but are not recommended for the CPU
   path (CPU-arithmetic-bound at ~5 GB/s/core, Strata finding 7).

### 3.3 Weight formats and the converter recipe

- New adapter `tools/convert/qwen4_exp.py` (like `qwen3_5.py`): `text_config()` normalises the HF config; `_Builder`
  maps HF names to logical roles: q_proj per-head [q|gate] de-interleave; `index_qk_proj` split 512/128; experts
  `gate_up_proj[e]` rows 0..639 gate / 640..1279 up, `down_proj[e]`; zero-centred norms kept as `w` (the Ops take
  `unit_offset`); `linear_attn.norm` plain; `A_log`/`dt_bias` -> fp32; conv squeezed to [4, C].
- Recipes in `official_recipes.py`:
  - `qwen3_8_flash_next` (format support; full-model conversion excluded): experts Q4 g64 gate/up +
    Q5 g64 down; no sensitive-layer Q6 mask or KL sweep is part of this delivery. GDN/QSA projections
    use Q8 (A8-int prefill permission) or Q4, HC
    BF16, router + shared score BF16, shared experts Q8, head Q6, embed Q8 (host gather), PLE projections BF16, MTP
    dense BF16 + MTP experts Q4, vision BF16. Source: HF BF16 (`--model`), or the official FP8 checkpoint via
    `sources/compressed_tensors.py`.
  - `qwen3_8_flash_next_gsq_q2` (Plans B/C): experts imported encoded from the GSQ-RCO Q2_0 GGUF
    (`--source gguf=...-00001-of-00002.gguf`) into `q2_g64_fp16`; dense tensors imported as their `gguf_*` blocks
    (GPU) where the runtime has a route, otherwise dequantized (`sources/gguf_source.py`) and requantized to our
    formats; MTP from HF BF16 (`--source mtp=` a local safetensors subset produced by a range-fetch helper modelled
    on Strata's `tools/mtp_fetch.py`: 31 tensors from 28 shards, ~5 GB, SHA-256 pinned). GGUF conventions undone as
    the ternary path does: tiled GDN value heads -> grouped order, `1+w` norms -> `w`, `ssm_a` -> `A_log`, fused
    `nextn_eh_proj [fc_embedding | fc_hidden]` split back.
  - the n-gram table, the same artifact's `ngram` component: from the GGUF shard 2 (`--source ngram=`,
    `per_layer_token_embd`, IQ4_NL, imported byte-exactly - note GGUF's split-half nibble order); the 128 HF BF16 row
    shards now stream to `fp8_e4m3fn_row_fp16` / `row_interleaved_v1`: 160 E4M3FN codes and one
    FP16 row multiplier, 162 bytes per row. `fp8_row_maxabs` rounds max|row|/448 to the stored
    scale before normalizing and rounding codes. The descriptor hashes the quantized row bytes;
    the writer rereads the BF16 source in bounded chunks, without a table-sized temporary file.
- Expert banks: one parent object per (layer, projection) holding 512 experts contiguously, expert-major, each
  expert's rows contiguous, so an expert is one contiguous byte range (cache slot copy = one DMA, CPU kernel = one
  stream). 73,728 logical expert parameters are grouped by `recipe.group`.
- Size: Plan A artifact ≈ 68 GB experts + ~5 GB rest; GSQ Q2 artifact ≈ 34 GB + ~4 GB, each with the 28.8 GB
  n-gram table, in one file (the writer splits into parts only when `--max-file-bytes` asks). Tensor axes must stay < 2^31 elements: the expert bank
  [512*1280, 2560] = 1.68e9 elements is fine; the n-gram table [320,001,446, 160] is fine per axis.

### 3.4 Retained hybrid experiment (frozen; current delivery uses GPU arithmetic)

The following records the implemented experiment, not additional delivery requirements.
Program-owned machinery (Ops stay closed: they receive an expert pointer table and a "parts" input):

1. **Host expert arena.** All experts of all layers in host RAM in their artifact encoding. Linux: `cudaHostRegister`
   in per-layer ranges (Strata: a single 34-43 GB registration fails on Windows; a DMA must not span ranges).
   **Windows/WDDM: pinned memory is charged against VRAM** (our own measurement), so the arena stays pageable there;
   GPU-bound copies go through a bounded pinned staging ring (Strata pins ≤ 8 GiB under WDDM and stages the rest).
2. **VRAM expert cache.** Byte-sized slots per layer (Strata finding 8: +13%), a device-resident residency table
   `(layer, expert) -> slot | -1`, initial fill from a routing profile shipped with the artifact (or recorded
   locally), adaptive swaps between rounds with evict-now/admit-on-copy-completion (finding 10). The `sparse_moe_512`
   Op reads expert pointers from the table, so a hit costs nothing extra.
3. **Doorbell.** After each layer's router, a captured kernel writes `(ids, weights, x)` into mapped pinned memory
   and bumps a sequence counter; a host thread spins on it, partitions misses into (a) DMA-to-GPU share (copy
   engine, `--expert-dma-share`, 20% for 2-bit / 55% for IQ per Strata), (b) CPU share computed in place by a thread
   pool, results written to mapped memory; the GPU computes hits and DMA'd experts, then a captured
   `wait_flag_ge` kernel spins on the CPU completion counter before the merge. One graph per window size stays valid
   (all per-round values live in device/mapped memory). The shared expert runs on the GPU after the ring, inside the
   CPU window (Strata P3).
4. **CPU expert kernels.** Q4 g64 and q2_g64 SwiGLU experts with int8-quantized activations: AVX-512 VNNI
   (`vpdpbusd`) and AVX2 (`vpmaddubsw`/AVX-VNNI) paths, multi-token reuse for verify windows (2.0-2.4x at 3 tokens,
   Strata). Oracle: FP64 against independently decoded weights, same criteria as the GPU route.
5. **Prefill streaming.** Per layer, uncached experts used by the chunk are DMA'd from the arena into borrowed
   cache slots one layer ahead and run through the grouped prefill route; chunk ≥ 8K to hide PCIe (2.4).
6. **Lookahead router prefetch** (optional, measured): apply layer l+1's router to layer l's input to start copies
   early (Strata uses it in the low-RAM mode).

Determinism: CPU and GPU expert arithmetic round differently. Qualification uses a fixed split
and no adaptive swaps. The delivered default is `--expert-dma-share 1`, which matched every
resident logit word in the final dense/sparse-prefix comparison. CPU mixes below one repeat
within a fixed mode but have not passed full-model numerical qualification; their further
refinement is excluded by the user's October 8 decision. CPU Op qualification retains its
independent FP64 oracle.

### 3.5 Residual stack, HC and the pipeline stages

- `R` is FP32 `[HC, H, T]` in the Program workspace; `run_layers` threads `R` instead of the BF16 hidden.
- Pipeline (`--devices`): `StageLink` payload becomes `R` (40 KB per column instead of 5 KB at H 2560) plus the
  pending `(y, w_inj)` of a fused write, or the write is materialised before the hop (simpler; Strata does the
  latter: "materialise R for the head"). The last stage returns `R` to rank 0, which runs the final mixer, head,
  sampling and the MTP layer (MTP needs `R`, not the mixed vector).
- Stage solver: per-layer weight bytes now include the expert bank (17 GB/24 layers at 2.25 bpw); the rank-0
  surcharge includes the head, final mixer and MTP.

### 3.6 KV, indexer, state and context-cache implications

- **KV pool:** Main Text pool for the 12 QSA layers (2 KV heads, D256) with the existing codecs; add a **pooled
  indexer-key plane** to the same page group (64-token page = 16 blocks x 128 FP32 = 8 KiB per page per layer), so
  pooled keys follow page sharing, COW, Host and disk tiers and prefix digests with no new store. Blocks never cross
  a page (64 % 4 = 0).
- **StateImage** gains: the indexer raw tail (≤ 3 x 128 FP32 per QSA layer), the PLE conv history (9 x 10240 FP32)
  and n-gram context (2 token ids), and the MTP continuation is the 10240-wide `R` of the last position (today H-wide).
  ReplaySSM covers GDN; the indexer tail and PLE history are small enough to snapshot per verify column and select
  on commit.
- **MTP KV:** private pool on rank 0 as today; either QSA with reuse of step-0 selection (vLLM, official) or a
  dense window (Strata). Start with the official behaviour.
- **Verify rollback:** pooled keys of blocks completed by rejected columns must be invalidated; since a pooled key is
  a pure function of 4 raw keys, recompute it on commit for the accepted frontier (at most 2 blocks per round).
- **Checkpoint identity** must include the PLE state: a prefix is reusable only if the same token prefix produced
  it (it always is: PLE is a function of the tokens), so no extra key is needed, but the restore must bring the PLE
  history and n-gram context back with the GDN state.

### 3.7 CUDA graphs

- Decode/verify graphs per (B, verify width) as today. QSA cost is bounded once `n > 2051`, so the frontier envelope
  matters only for the indexer scan (`ceil(n/4)` blocks) and for the dense-identity route below 2052 positions:
  topology classes = {dense ≤ 2051, sparse}; the indexer kernels read `n` from device memory and are launched with
  a capacity grid (Strata's step buffer pattern).
- The PLE row upload, the doorbell and the CPU-completion wait are graph nodes reading mapped/device memory, so
  graphs need no host values per round.
- Prefill stays eager.

### 3.8 Serving and frontend

- `reasoning_effort` request field (OpenAI `reasoning_effort`, Anthropic `thinking` budget mapping) passed to the
  template; schema tests and `docs/serving.md` updated together (external contract).
- `/stats`: expert cache hit rate, CPU expert ms, DMA share, n-gram I/O counters.

### 3.9 Strata reuse (MIT, copyright 2026 Niko1221 and contributors; attribution in CONTRIBUTORS.md and file headers)

| reuse (adapt under MIT) | write fresh |
|---|---|
| n-gram hash and row constants, EOS-cut semantics, and **its parity vectors** (`src/kernels/ple_parity.cpp`, `ple_reader_test.cpp`) as oracle fixtures | every Op wrapper/contract/workspace/qualification (our Op contract) |
| `src/ngram/ple_reader.cpp` design: I/O thread, bounded row cache, Windows unbuffered reads, keepalive | the n-gram component and its loader |
| doorbell protocol + `wait_flag_ge` + mapped staging (`elementwise.cu`, `verify_kernels.cu:426`) | sparse attention over our paged KV codecs and 64-token pages |
| expert cache policy (byte-sized slots, profile fill, adaptive swaps, evict-now/admit-later), `expert_source.cpp` low-RAM lookahead | MoE 512 GPU routes at our formats (A8 MMA prefill, decode/small-T) |
| CPU Q2_0 AVX-512 VNNI / AVX2 expert kernels (`src/kernels/cpu/expert.cpp`, `kq_avx2.cpp`) as the starting point for `q2_g64` CPU kernels | HC kernels (Strata's `fused_gr.cu` is a reference for the fusion, not for our contracts) |
| `qsa_select.cu` radix block top-k as an algorithm reference | GDN (ours is ahead: chunked WY, ReplaySSM) |
| `tools/mtp_fetch.py` range-fetch + SHA-256 pinning logic | converter adapter and recipes |
| routing profiles format idea (`--expert-profile-save`) | Program integration (transactions, graphs, context cache) |

Do **not** adopt the "experimental speed projection" (an abliteration vector, `reports/strata.md` 2.10).
