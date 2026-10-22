# RTX 4090 GGUF variance and greedy diagnosis, October 2026

This follows [PR #3](https://github.com/spk-c/ninfer-all/pull/3) and the
[complete loop experiment report](rtx4090-gguf-2026-10.md). The IQ3_S T6 loop
has no established stable Engine benefit and remains OFF, as does BF16x8 SiLU.
No new CUDA candidate or route change was made in this investigation.

Full raw measurements, including the earlier adverse results and discarded
captures, are published in the [GitHub evidence release](https://github.com/spk-c/ninfer-all/releases/tag/rtx4090-diagnosis-20261022).
The archive contains Engine JSON, stdout/stderr, per-process and global telemetry,
Nsight Systems reports and SQLite exports, Compute reports and counter/source
exports, numerical/build logs, prompts and raw diagnostic tensors. It contains
no model weights. The [per-process case CSV](rtx4090-diagnosis-2026-10-processes.csv)
contains all twelve primary controlled processes; the repeat has its own CSV.

## Conditions and scope

One Windows/WDDM RTX 4090, driver 617.42, CUDA 13.4, MSVC 19.44, Release sm_89.
All work used `C:\coding\ninfer-all`, local branch `codex/rtx4090-optimization`.
The existing artifact at
`C:\ninfer-all\models\Huihui-Swift15-Abliterated-IQ3S-DFlash2.ninfer`
was opened read-only. No conversion, repacking, download or model write occurred.
No other architecture or model was executed. Master was not changed or merged.

The controlled executable is the same baseline `build-rtx4090/bench/ninfer_bench.exe`
in both arms, derived from PR #3's `0be32db4`, with the T6 loop OFF. Both labels
are baselines, not a candidate comparison. DFlash2 uses five drafts and zero
n-gram drafts; plain and DFlash2 are evaluated separately. Public Engine workloads
are pp512, pp2048, tg128, pp512+tg128, pp2048+tg128 and pp8192+tg128,
rk4v4, CUDA Graphs, max context 16384, two measured reps per case.

## 1. Identical-process variance

`tools/bench/control_gguf_4090.py` locks graphics to 2100 MHz, memory to
10501 MHz and power limit to 450 W. A complete conditioning process warms the
OS file cache, then every case gets two full Engine warmups; prefix reuse is
disabled. This defines a warm-cache workload, not a forced identical L2 bit
state or a cold-file experiment. Each measured process starts at 50–52 C
(observed 52 C); order alternates AB/BA over three pairs per mode. GPU work
is serialized. Original unlocked clocks and the 450 W limit are restored.

All 1770 primary global telemetry samples show 2100/10501 MHz. The only clock
event masks are zero and GPU idle; no power/thermal throttle was sampled.
Global temperatures span 48–68 C including conditioning. Measured-process
temperatures span 48–67 C; this controls entry temperature, not every instant
of temperature during different workload phases. Board power peaks at
344.31 W plain and 345.14 W DFlash2. Process medians span 284.14–287.365 W
plain and 127.43–184.04 W DFlash2; these include loading and idle gaps and are
not energy-per-token measurements. Sampled PID VRAM peaks are 12306.24 MiB
plain and 14940.26 MiB DFlash2. Device-wide memory includes the desktop.
Sampling cannot establish an instantaneous allocation or power maximum.

All decode B/A changes, percent (positive means B faster):

| Mode/pair | tg128 | pp512+tg128 | pp2048+tg128 | pp8192+tg128 |
|---|---:|---:|---:|---:|
| Plain 1 | -1.640 | -2.154 | -0.210 | +0.208 |
| Plain 2 | +0.007 | +0.038 | +0.185 | -0.241 |
| Plain 3 | +0.369 | -0.093 | +0.130 | +0.112 |
| DFlash2 1 | -0.160 | +0.178 | +0.024 | -0.155 |
| DFlash2 2 | +0.186 | +0.064 | +0.362 | -0.074 |
| DFlash2 3 | +0.212 | +0.051 | -1.656 | +0.284 |

The full CSV also retains prefill and observed Engine TTFT. Plain prefill
changes span -1.797% to +0.305%; DFlash2 -1.344% to +3.574% (pp512, pair3).
Plain tg128 TTFT worsens by +1.682%, +6.346%, +3.027%, despite the much
smaller throughput spread; DFlash2 tg128 TTFT spans -1.593% to +1.231%.
Absolute tg128 TTFT spans17.443–18.550 ms plain and17.777–18.267 ms DFlash2.
Other TTFT changes span -0.304% to +1.827% plain and -3.454% to +1.361%
DFlash2. Positive TTFT change means slower. These adverse results are not
discarded. The later plain pairs repeat the first plain throughput regressions;
an additional complete DFlash2 pair repeats the final DFlash2 regression and
pp512 prefill anomaly (results below).

The first additional attempt failed its50–52 C thermal gate: after the saved
conditioning process the GPU was already below50 C and the gate timed out at
40 C after240 seconds. No measured pair began. Its telemetry/failure record
is retained. The follow-up uses a separately declared40–42 C gate; both actual
entries are42 C. It is not pooled with the52 C series. Both measured processes
retain2100/10501 MHz in every matched global sample, with no sampled power or
thermal event. Their sampled temperatures span42–62 C, peak powers343.65 and
342.95 W, and PID VRAM peaks14940.26 MiB each. Initial conditioning includes a
clock-lock transition; it is not counted as a measured comparison.

| Additional DFlash2 pair, B/A percent | tg128 | pp512+tg128 | pp2048+tg128 | pp8192+tg128 |
|---|---:|---:|---:|---:|
| Decode | +0.082 | +2.215 | +3.989 | -0.238 |

Complete values are in the [repeat CSV](rtx4090-diagnosis-2026-10-repeat-processes.csv).
Prefill changes span-1.044% to+5.695%, and TTFT-5.693% to+1.053%.
The original negative pp2048 decode change and positive pp512 prefill anomaly
do not repeat with the same sign/magnitude, but the new+3.989% decode and
+5.695% prefill differences are substantial B/B variation, not optimization
gains. The controlled GPU conditions therefore do **not** eliminate the problem.
In pp2048+tg128, A's two prefill times are0.736958/0.657614 s and decode
0.561191/0.600892 s; B's are0.657577/0.657584 and0.558013/0.558193 s.
Excess times concentrate in individual repetitions. All four report exactly
24 speculative rounds,120 drafts,103 accepted drafts, so a changed amount of
speculative work does not explain those excess times. Their source is not
established by low-frequency GPU telemetry.

The earlier unlocked-clock experiment and repeats had worst changes of -9.20%
plain and -9.01% DFlash2; repeat plain pp8192 regressions were -8.17%/-7.05%
and DFlash2 pp2048 -8.02%. They remain valid observations, not a speedup claim.
Those 7–9% swings were not reproduced with these combined controls. The old
captures lack memory-clock, throttle and synchronized per-case clock evidence:
this experiment does **not** prove DVFS, temperature, cache or power was the
individual historical cause. Residual variation persists with fixed clocks.

### Host/queue attribution

Four additional Systems node captures run pp8192+tg128 with two warmups,
one measured rep, the same baseline and a <=52 C thermal entry. Two allow
free clocks; two lock 2100/10501 MHz, all at 450 W. Only the measured region
is captured (`cudaProfilerApi`, graph node tracing). Per `decode.ordinary_round`,
the analysis takes the union of GPU kernels within the NVTX interval, separating
time before its first kernel, inter-kernel gaps and time after its last kernel.

| Capture | Mean round ms | GPU active ms | Before first kernel ms | Inter-kernel gaps ms | After ms |
|---|---:|---:|---:|---:|---:|
| free1 | 15.9380 | 14.9353 | 0.9179 | 0.0714 | 0.0133 |
| free2 | 15.9635 | 14.9373 | 0.9417 | 0.0699 | 0.0146 |
| fixed1 | 16.6283 | 15.5852 | 0.9400 | 0.0887 | 0.0144 |
| fixed2 | 16.4290 | 15.4199 | 0.9061 | 0.0901 | 0.0130 |

Each has 128 rounds. Before-first-kernel time includes CPU work, submission
and device queuing; it is not a pure CPU-function measurement. Roughly 0.9 ms
per round is outside kernel execution there. The fixed captures still differ
about 1.1% in GPU active time. Profiler timings are separate observations,
not substituted for the unprofiled Engine rates, and do not reproduce a 9%
stall. `summarize_gguf_diagnosis.py` reproduces the CSV and interval analysis.

Two further fixed-clock **whole-process** captures target the anomalous
pp2048+DFlash2 case, with two warmups and two measured reps, CPU scheduling
and base WDDM tracing. Their last two `generate` ranges correspond to the
measured repetitions; warmups and graph preparation are excluded from attribution.

| Capture/rep | Engine prefill ms | Engine decode ms | Prefill target kernels ms | Decode target kernels ms | Decode round time outside target kernels ms |
|---|---:|---:|---:|---:|---:|
| fixed1/1 | 658.259 | 577.431 | 650.586 | 554.418 | 21.983 |
| fixed1/2 | 661.036 | 577.205 | 652.150 | 552.351 | 23.662 |
| fixed2/1 | 659.324 | 586.462 | 651.177 | 558.034 | 27.024 |
| fixed2/2 | 658.071 | 585.728 | 650.151 | 558.946 | 25.612 |

Engine timers and NVTX round sums have different boundaries; the latter omit
fallback/post-round work, so these columns are not an additive decomposition
of Engine decode time. GPU-active time is the union of target kernels. CPU
scheduling records show small ready-to-run intervals, but both target processes
carry `System profiling might have not been started correctly.` Although CUDA
and NVTX ranges were collected, CPU-state completeness is therefore unverified;
the saved running/ready/blocked attribution is provisional. Base WDDM events
do not supply a complete GPU context-switch/preemption trace. Do not infer
the absence of memory migration or desktop interference from them. Full
tracing also adds overhead; these are not substitute baseline throughput samples.

The80 ms prefill/43 ms decode excesses did not recur in these four measured
reps. The captures establish a non-kernel time component and residual variation,
but do not identify the historical or new unprofiled spikes' cause. CPU scheduling,
driver queuing and other WDDM activity remain unproven explanations, not findings.
`analyze_gguf_wddm.py` retains every interval and the target warning.

Three profiler starts were invalid and are retained: two requested `-r2`
with `--profile-measured` (the benchmark requires exactly one repetition),
then the first whole-process command combined `capture-range=none` with
`capture-range-end=stop` (rejected by Nsight). The runner now rejects the first
combination and omits the second option in whole-process mode. A WDDM-only
CLI-help probe also remains in the archive and is not a model measurement.

## 2. First divergent target and verification audit

The diagnostic-only patches `qwen_target_trace.patch` and
`qwen_attention_trace.patch` add synchronous binary copies to separate CLI
builds. They are **not applied to product sources**. Raw BF16 target logits,
positions, input IDs, argmax, drafts, licensed IDs/counts and accepted prefix
counts are retained. Diagnostic latency is not benchmark evidence. The normal
CLI was rebuilt after reverting instrumentation. Existing short CUDA and
3585-token long prompt fixtures are reused, greedy, rk4v4, five drafts.

For the long prompt, the first differing raw logits occur at feed cache
position **3585**, verification column zero, predicting output token **2**.
This is before any draft is accepted: 244202/248320 logits differ, maximum
absolute difference 1.9375, RMS 0.327274. Both choose token1144 (32.75 plain,
32.25 verify). There is already a target evaluation difference, independent
of acceptance decisions.
The first differing vocabulary coordinate is ID0:8.6875 plain versus8.125
verify at that same step; full vectors are retained.

The first greedy divergence is feed position **3591**, verification column
zero, predicting output token **8**. Plain scores tokens2570 and14235 both
25.125 and picks2570 by the existing tie order. Verify scores14235 at25.375
and2570 at25.125 and chooses14235. Maximum full-logit difference is0.921875,
RMS0.167151. The short CUDA prompt first differs in logits at feed68 and in
argmax at feed100 (output34), reproducing the earlier baseline observation.

All audited licensed columns equal the recorded target argmax and the
independently computed raw-logit argmax; every accepted draft equals that
target. Counts are 16 long graph, 15 long eager, 40 short and 16 BF16 columns,
including graph priming where present. There is no observed wrong acceptance
or stale licensed token. Actual long-prompt graph/eager logits and positions
are **bit-identical within each mode**, excluding dummy priming position zero.
With **BF16 KV**, every compared full raw-logit vector is identical across
plain and DFlash2 for 16 outputs. This is finite coverage, not a general
BF16 identity guarantee.

Layer traces on the common first input show identical hidden states through
layers0–6, then the first difference after **layer7**, the second full-attention
layer: 2365/5120 BF16 values, max0.00390625, relative L2 difference0.000648.
By layer63, 5068/5120 values differ, max3.375, relative L2 difference0.159679.
Small early differences can materially change later logits.

### Independent qualification of the first Attention difference

Eager snapshots capture public Q, gate, K/V inputs, stored paged K/V bytes
and FP16 scales, block tables and Attention output at layer7, feed3585.
Plain width1 and verify width6 have identical first-column Q/gate/K/V and
identical logical stored K/V through the causal position. The oracle independently
decodes Lloyd4 K and signed-int4 V with the stored scales, uses FP64 H256/16
on Q (equivalent to inverse-rotating decoded K), FP64 dot/softmax/PV and sigmoid
gate, without transient Q quantization or another CUDA kernel as oracle.

| Public gated Attention, 6144 values | Relative L2 to FP64 | Maximum absolute error | Gross limit | Pass |
|---|---:|---:|---:|---|
| Plain | 0.00371848 | 0.00299844 | 0.00660203 | yes |
| Verify column0 | 0.00371501 | 0.00299844 | 0.00660203 | yes |

These are the **existing** rk4v4 gated criteria from the Op tests: relative L2
<=0.00325+2^-9, gross absolute <=0.0011+(2/256+2^-9)*max(abs(reference)).
No threshold was relaxed. 86/6144 output values differ between widths.
Thus the first divergence is width-dependent numerical Attention evaluation
over identical represented inputs, with both outputs within the named Op
criterion. It is not evidence of corrupted cache contents or a draft-acceptance
bug. Later layer/logit amplification remains observable; passing this one
Attention case is not a full-model FP64 qualification.

Forcing the **existing** `NINFER_SMALLT_PV_F16=0` in two more long-prompt
processes does not remove the difference: first raw difference still3585,
max2.16015625/RMS0.324997, first argmax divergence still3591; now plain14235
(25.375 vs25.25), verify2570 (25.375 vs25.125). This override changes each
backend's results and is not adopted as a fix. The evidence does not isolate
one instruction, query quantization or reduction schedule as the sole cause.

The CLI contract already documents verification-width floating-point differences
for greedy decoding (`docs/cli.md`). Within this coverage no numerical-tolerance
or acceptance correctness violation was found; cross-backend greedy token
identity is demonstrably absent. There is also a pre-existing **documentation
contract discrepancy**: `softmax_attention.h` says all PV accumulation is FP32,
but `small_t_i8_launch.cuh` selects FP16 tile accumulation via `attn_pv_f16`.
This report records that inconsistency; it does not silently redefine arithmetic
or change routes on untested architectures. It remains a separate contract
resolution item, not a claimed runtime fix.

## 3. Benchmark schema compatibility

The matrix reader, RTX4090 tuner, speed-of-light accounting and paired GGUF
reader now accept `ninfer_bench_report` versions **16,17,18,19**. Existing
report fields retain their meaning; missing observed `first_token_seconds`
is unavailable, never reconstructed from phase times. Unknown versions remain
rejected. Parameterized reader tests cover every supported version and missing
TTFT; 39 affected Python tests pass. A saved corpus check also loaded114 real
reports through the matrix and GGUF readers (63 schema18,51 schema19), with
zero failures; versions16/17 are covered by fixtures rather than a real local
report corpus. No benchmark writer or Engine timing
semantics changed. Python3.11 compilation and diff checks pass.

## 4. GGUF memory evidence and decision

Existing full Compute captures are reused, rather than starting another kernel
search. The original cumulative Systems attribution remains in the linked
report: GGUF vector products dominate (92.0% plain,78.3% DFlash2 GPU time),
IQ3_S alone30.53%/26.85%. The representative unfused IQ3_S product is
`[10240,5120] x [5120,T]`, T1 ordinary/T6 verification, through
`gguf_linear.cpp` -> q8_1 activation preparation -> bridge type/width/fusion/K
dispatch -> `ggml_vec_iq3_s.cu` -> `ggml_bridge_vec.cuh` vector launcher.
The exact stored matrix is22528000 bytes (110 bytes per256-value block).
RTX4090 has GDDR6X DRAM; these are DRAM measurements, not HBM.

| Baseline captured kernel | T1 | T6 |
|---|---:|---:|
| Duration us | 28.512 | 30.304 |
| DRAM read bytes | 22958592 | 22719104 |
| DRAM bytes / stored matrix | 1.01911 | 1.00848 |
| DRAM read GB/s | 805.23 | 749.71 |
| L1 global-load sectors, bytes (32B/sector) | 176029696 | 453246976 |
| L2 total sector bytes | 29713152 | 33536448 |
| Registers/thread | 56 | 128 |
| Achieved occupancy % | 65.99 | 30.96 |
| Long-scoreboard cycles per issue-active warp | 29.17 | 7.01 |

The full raw CSV includes units and German-localized numeric strings; dots in
integer counters are thousands separators. These are individual profiled
observations, not a distribution or controlled Engine comparison. These counters
come from kernel-replay profiles (the T6 log records45 replay passes and a
system-memory backup warning). Replay/cache preparation can differ from an
unprofiled Engine, so these counters do not establish every layer's natural
cache residency. Export with `ncu --import REPORT --page raw --csv --print-units base`.
Measured DRAM writes
are zero in this measurement interval; output stores can remain dirty in L2.
L2 totals include all sector traffic, not just weight reads. Long-scoreboard
waiting is a memory-dependency symptom, not by itself proof of DRAM saturation.

The source/SASS counters distinguish narrow weight metadata loads from vector
activation loads. Dynamic active-thread `LDG.E.U8` counts are3276800 in both;
`LDG.E.U16`12288000/16384000; `LDG.E.128`1638400/9830400 (T1/T6).
Subtracting the q8_1 activation-scale loads (819200/4915200 halfwords) leaves
11468800 weight halfword loads in both. Logical weight requests total
26.2144MB:128 requested bytes per110 stored bytes. Repeated per-lane half
scales cost14 extra bytes/block; repeated scale nibbles cost4. This is16.36%
redundant logical metadata demand, largely served by L1, **not**16.36% extra
DRAM traffic. T6 activation int4-vector loads rise sixfold (26.2144 to157.2864MB),
but measured physical DRAM weight traffic does not. Packed110-byte blocks and
8/16-bit scalar loads explain inefficient request width; activations already
use128-bit loads. L1 sector demand is not interchangeable with physical DRAM.

The concrete bottleneck is a near-one-pass compressed-weight DRAM stream with
memory dependency latency and, at T6, lower occupancy/register pressure plus
substantial cache-served activation demand. There is **no measured sixfold
weight reload** to eliminate. The retained T6 baseline has0.48 eligible
warps/scheduler/cycle and35.79%
compute utilization, below74.51% DRAM utilization, with no spilling. Its
78643200 active-thread DP4As imply2.60 trillion thread DP4A instructions/s;
these counters do not establish a DP4A execution-throughput limit.
Metadata sharing is a possible future hypothesis, but neither its Engine
payoff nor the register/shuffle cost is established.
The prior row1/activation-streaming candidates regressed and remain rejected;
the rolled-loop result remains unstable. Therefore no new opt-in CUDA candidate
was implemented or enabled. None of these observations establishes an Engine
acceleration or an improvement on other architectures.

## Reproduction and remaining limits

The scripts store exact commands and telemetry with every process. Use Python3.11:

```powershell
py -3.11 tools/bench/control_gguf_4090.py --exe C:/coding/ninfer-all/build-rtx4090/bench/ninfer_bench.exe --weights C:/ninfer-all/models/Huihui-Swift15-Abliterated-IQ3S-DFlash2.ninfer --out profiles/bench/new-controlled-bb
py -3.11 -m pytest tests/test_bench_matrix.py tests/test_tune_rtx4090.py tests/test_speed_of_light.py tests/test_gguf_bench_reader.py -q
```

Apply only the chosen diagnostic patch to build a separately named trace CLI,
revert it and rebuild the normal CLI before normal use. `trace_qwen_greedy.py`
supports target, attention and existing-pv32 studies; the analysis/oracle scripts
consume their saved tensors without loading the model. Never use trace builds
for performance claims.

Historical variance has been bounded under the stated combined controls but
not causally reproduced; even the controlled follow-up retains up to3.99%
decode and5.69% prefill B/B differences. Numerical evidence is finite: two prompts for raw
logits, graph/eager long coverage, one BF16-KV long comparison, one captured
first-divergence Attention oracle case. It does not certify arbitrary prompts,
contexts, all layers, concurrency or GPUs. The FP32/FP16 documentation mismatch
remains disclosed. No automatic activation, merge or master update occurred.
