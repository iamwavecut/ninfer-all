# RTX 4090 attention precision and benchmark stability

This is the stabilization follow-up to [Draft PR #3](https://github.com/spk-c/ninfer-all/pull/3).
It changes documentation, independent tests and observability, not CUDA arithmetic,
dispatch schedules, device profiles or experimental defaults. The existing artifact
`C:\ninfer-all\models\Huihui-Swift15-Abliterated-IQ3S-DFlash2.ninfer` is read-only.
Work stays in `C:\coding\ninfer-all`; master and PR #3's head are not changed.
Complete raw tensors, telemetry, Nsight reports/SQLite, build logs and the initial
failed qualification are published in the
[stabilization evidence release](https://github.com/spk-c/ninfer-all/releases/tag/rtx4090-stability-20261022).

## Reviewable separation

The branch starts directly at `feature/rtx4090-sm89-tooling`, not the experimental
optimization branch. No BF16x8-SiLU or IQ3_S-T6-loop kernel, CMake option or launcher
change is included. The base branch's already-existing SFU switches remain OFF.
There is no new CUDA candidate, calibration or automatic path selection.

The independently reviewable commits are:

| Commit | Scope |
|---|---|
| `3d6db9fa` | Windows compiler PATH, vcpkg/SDK resource compiler, target-only test builds, optional bounded build parallelism, MSVC-compatible spelling of the same half-to-float conversion |
| `e7f22084` | Observed TTFT/schema19, readers16â€“19, per-repetition benchmark NVTX markers and reader tests |
| `7a6905cf` | Correct precision contract, independent attention qualification and exact near-tie argmax fixtures |
| Follow-up diagnostic commit | Reproducible trace matrix, explicit verification audit, phase analysis and this report |

The CPU half conversion is a build portability fix using the same F16C conversion,
not a new CPU expert algorithm. The test oracle uses C++20's double-precision pi
constant instead of the nonportable `M_PI`. Product Engine/model APIs and ownership
do not change. Python tools orchestrate the public CLI; there is no Python inference
implementation. Trace patches are diagnostic-only and are reverted after building
a separately named executable.

## Implemented and tested precision contract

The old sentence promising FP32 accumulation for every PV stage was incorrect.
`small_t_i8.cuh` and its pipelined sibling have a `PvF16` specialization: FP16
P/V operands feed FP16 tensor-core tile accumulators, then the result is widened
and added to the FP32 running accumulator. The alternative uses FP32 PV
accumulators. `prompt_i8.cuh` has the same tile option for the prompt path.
The RTX4090 built-in profile has `attn_pv_f16=on`; the respective diagnostic
overrides are `NINFER_SMALLT_PV_F16` and `NINFER_PROMPT_PV_F16`.

| Stage | Existing arithmetic / semantic boundary |
|---|---|
| Public query and persistent KV | BF16 query; BF16 or independently decoded stored rk4v4 K/V and stored scales |
| INT8-family QK | Private query preparation/quantization, INT32 group products combined in FP32 |
| PV key tile | FP16 operands, FP16 or FP32 accumulation according to the existing profile |
| Cross-tile sum, split state, merge, normalization | FP32 |
| Public Attention output | BF16; optional gate has the existing additional BF16 rounding boundary |
| Mathematical oracle | Naive represented-input FP64 dot, stable softmax and PV; packed codes decoded independently |

The actual accuracy promise is the named **output-error criterion over the tested
domain**, not FP32 arithmetic everywhere, a universal bound for adversarial BF16
magnitudes, or bit identity between widths. FP16 has finite range; the existing
tests do not certify arbitrary large finite BF16 inputs. Changing the documentation
does not add a new guarantee or relax any numerical threshold.

| Criterion | BF16 KV | rk4v4 |
|---|---:|---:|
| Ungated relative L2 maximum | 0.0028 | 0.00325 |
| Gross absolute floor | 0.001 | 0.0011 |
| Gross relative term times max absolute reference | 2/256 | 2/256 |
| Additional gated relative-L2 and gross-relative allowance | 2^-9 | 2^-9 |

The new precision subset invokes append-and-attend, cached-only and batched
production routes directly for D256/H24/KV4, fragmented pages, BF16/rk4v4,
widths1/2/4/6/16 at contexts0/68/512/3585 and widths1/6 at8192. Each entry
is compared to its independent FP64 oracle, including Graph execution, cache
effects, guards and the gated boundary. Separate CTests unset the overrides
for the device default, force both overrides to0, or force both to1. Thus the
FP32 and FP16 variants use the same original criterion, not mutual agreement.

All three card-profile precision CTests pass (37.54 seconds total). The test
explicitly installs the compiled RTX4090 profile, with no profile file,
calibration or persistence; standalone Ops otherwise use the fallback profile.
Default rounded statistics equal forced FP16, consistently with the source route.
Each variant produces176 oracle observations (132 ungated and44 gated).

| Profile / KV | Maximum ungated relative L2 | Maximum gated relative L2 | Maximum fraction of gross bound |
|---|---:|---:|---:|
| Default or forced FP16 / BF16 | 0.0020900 | 0.0026671 | 0.411 |
| Forced FP32 / BF16 | 0.0020900 | 0.0026671 | 0.411 |
| Default or forced FP16 / rk4v4 | 0.0017557 | 0.0024214 | 0.350 |
| Forced FP32 / rk4v4 | 0.0017442 | 0.0024072 | 0.350 |

The initial three new tests crashed because this new fixture selected the enum
builder overload instead of the existing packed CachePlan overload, leaving
packed buffers null. A debugger established the host null write. The fixture
was corrected; no production math or allocation fix was made. The first passing
rerun also exposed a coverage gap: standalone Ops had not installed the Engine's
card profile. The explicit compiled-profile setup above closes that gap; both
initial failures and earlier fallback qualification remain in the raw evidence.

## Plain versus DFlash2: separate three questions

1. **Numerical tolerance:** the independently qualified Attention result versus
   its represented-input FP64 oracle. Full-model logit differences have no
   registered tolerance in this repository; the tools do not invent one.
2. **Token identity:** exact generated IDs and first differing output position.
   A changed argmax near a tie can fail identity without violating verification.
3. **Verification correctness:** every valid target argmax agrees with its raw
   logits; accepted drafts form the correct longest available prefix; the bonus
   count and every licensed token are correct. These are exact checks.

`test_target_trace.py` deliberately injects wrong argmax, wrong licensed token,
wrong accepted-prefix/bonus counts, masked tails and nonfinite evidence. The
GPU argmax fixtures include exact ties and adjacent represented BF16 values
25.125/25.25, with an exact scalar argmax/Top-2 oracle. Raw-logit comparison
alone is never used as evidence of a verification error.

### Model matrix

Thirty serialized CLI processes cover prompt lengths68/593/3585, BF16/rk4v4,
plain decode and DFlash2 widths2/4/6/16 (draft counts1/3/5/15). The short prompt
generates40 outputs; the medium/long prompts16. Fixtures are committed under
`bench/fixtures/greedy_target/`. CUDA Graphs, greedy decoding, max context8192,
five-draft mode only where width6 is selected, and n-gram drafts disabled.

The matrix deliberately reuses the archived PR #3 baseline trace binary with
BF16x8 SiLU, SFU and T6-loop OFF. It ran concurrently with compilation and is
**numerical evidence only**, not timing or resource-efficiency evidence.
Provenance is recorded with the raw data. Fresh stable-branch Op qualification
and diagnostic-vector cross-checks are separate below. Older trace buffers infer
the valid width from clamped positions; the new patch also records the explicit
target-valid-column tensor. Dummy graph priming is excluded from actual audits.

| KV / prompt tokens | W2 | W4 | W6 | W16 |
|---|---|---|---|---|
| BF16 /68 | identical | identical | identical | identical |
| BF16 /593 | identical | identical | identical | identical |
| BF16 /3585 | identical | identical | identical | identical |
| rk4v4 /68 | identical | identical | first difference34 | first difference34 |
| rk4v4 /593 | identical | identical | first difference9 | first difference9 |
| rk4v4 /3585 | identical | identical | first difference8 | identical |

Across24 paired comparisons,19 have identical tokens and5 do not. Complete
raw-logit parity is narrower:6/12 BF16 pairs and3/12 rk4v4 pairs. In particular,
all three width16 BF16 runs already differ in logits on the first target input,
despite retaining token identity. Increasing width does not monotonically make
token identity better or worse; the long rk4v4 W16 run matches within16 outputs
where W6 diverges. No width or KV codec is adopted as a universal identity fix.

All780 audited valid target columns pass the exact verification checks. All24
pairs contain a shared-prefix Top-2 margin <=0.25, totaling47 observed columns.
That0.25 is an explicitly chosen diagnostic margin, **not** a numerical tolerance.
Comparisons after generated histories diverge are marked as such and do not
attribute differences solely to width. The [complete matrix CSV](rtx4090-stability-2026-10-matrix.csv)
retains all pairs, first logit differences, margins, identities and audit coverage;
full raw tensors and per-column analyses are in the evidence archive.

A fresh stable-branch diagnostic binary cross-checks long-context rk4v4 plain
and width6 DFlash2 against the archived matrix: all15 actual plain rounds and
all4 actual DFlash2 rounds have exactly matching positions and complete raw
BF16 logit vectors. The new explicit valid-column metadata yields zero audit
errors. This supports reuse of that numerical evidence without attributing its
timings to this branch.

Widths in the table are configured physical verification widths. With only16
requested outputs, medium/long width16 runs initially have15 valid target
columns because one output is already committed; the remaining budget further
caps later rounds. Full live width16 is qualified by the Op tests, not claimed
for every model-matrix round.

No wrong acceptance or Op-tolerance violation is established by these model
comparisons. Conversely, token drift is real and a finite passing local oracle
case does not certify ideal full-model FP64 behavior or arbitrary prompts.

## Phase-aligned Baseline/Baseline analysis

The earlier controlled investigation retained up to3.99% decode and5.69% prefill
B/B variation even at fixed clocks; the historical -7% to-9% swings were not
causally explained. Earlier CPU/WDDM captures warned about system profiling and
did not reproduce the large individual-repetition excesses. Those findings remain
in [PR #3's complete report](https://github.com/spk-c/ninfer-all/blob/codex/rtx4090-gguf-vector/docs/performance/rtx4090-diagnosis-2026-10.md).

The stable benchmark now wraps every warmup and measured repetition in an exact
NVTX marker `bench.warmup/LABEL/INDEX` or `bench.measured/LABEL/INDEX`. The phase
analyzer requires explicit measured markers instead of guessing the last two
generations. It aligns each repetition with the native Engine JSON and records:

- host NVTX phase wall time, separately for every prefill chunk and decode step;
- the union of overlapping target CUDA kernel elapsed intervals, clipped at phase boundaries;
- phase wall time outside those intervals and its longest contiguous gap;
- host submit/wait wall intervals as overlapping observations, not additive CPU busy time;
- sampled clock/memory-clock/temperature/power/event masks aligned using the Nsight UTC session epoch.

Host wall time includes scheduling/driver/GPU waits; CUDA kernel elapsed intervals
are not SM-busy counters or proof of no preemption. CPU scheduling completeness
is false when absent or warned, and remains unverified merely when events exist
without warnings. Base WDDM coverage is explicitly incomplete for GPU context
preemption/residency attribution. Empty telemetry samples stay unavailable, and
low-rate telemetry cannot exclude sub-sample excursions. No CPU-busy or WDDM
causality claim is fabricated from those measurements.

Six identical-binary processes (two plain, two DFlash2, then two targeted
DFlash2 repeats) supply12 measured repetitions. Workload: pp2048+tg128,
rk4v4, max context16384, CUDA Graphs, two Engine warmups per process;
DFlash2 draft5, n-gram drafts0. There was no concurrent compilation or
numerical GPU job. Requested limits were2100/10501 MHz and450 W, with entry
at most52 C. Entry temperatures were36–41 C. Idle entry clock samples can
be lower; every sampled measured phase reports2100/10501 MHz. Engine warmups
standardize execution/cache preparation, but do not prove identical L2 contents;
the OS model-file cache is warm, not flushed between processes.

These are Nsight-instrumented B/B observations, not uninstrumented A/B speedup
measurements. Native times and NVTX scope times are kept distinct.

| Mode/capture/rep | Native prefill ms | Native decode ms | Prefill kernel union ms | Decode kernel union ms | Decode outside all traced CUDA ms | TTFT ms |
|---|---:|---:|---:|---:|---:|---:|
| plain/fixed-1/0 | 652.96 | 2161.24 | 645.00 | 1967.15 | 197.16 | 653.09 |
| plain/fixed-1/1 | 651.65 | 2133.06 | 643.02 | 1965.80 | 169.81 | 651.81 |
| plain/fixed-2/0 | 654.36 | 2101.56 | 644.59 | 1968.14 | 134.19 | 654.55 |
| plain/fixed-2/1 | 653.53 | 2089.52 | 646.16 | 1966.59 | 123.56 | 653.62 |
| dflash2/fixed-1/0 | 665.51 | 587.55 | 656.74 | 556.80 | 31.38 | 665.63 |
| dflash2/fixed-1/1 | 747.18 | 671.90 | 734.21 | 631.46 | 41.14 | 747.35 |
| dflash2/fixed-2/0 | 663.74 | 586.18 | 654.75 | 557.53 | 29.18 | 663.86 |
| dflash2/fixed-2/1 | 659.05 | 584.02 | 649.95 | 559.10 | 25.41 | 659.18 |
| dflash2-repeat/fixed-1/0 | 672.93 | 585.25 | 665.32 | 556.15 | 29.62 | 673.06 |
| dflash2-repeat/fixed-1/1 | 660.31 | 579.04 | 651.75 | 555.65 | 23.78 | 660.44 |
| dflash2-repeat/fixed-2/0 | 661.02 | 630.60 | 653.37 | 599.43 | 31.78 | 661.15 |
| dflash2-repeat/fixed-2/1 | 666.98 | 590.43 | 658.89 | 559.76 | 31.27 | 667.10 |

Plain native decode varies3.43% (max/min minus one), while its kernel union
varies0.12%. The outside-CUDA part falls from197.16 to123.56 ms and host
submit wall time from172.73 to102.33 ms; wait wall time stays1985.86–1988.92 ms.
This localizes most of that Plain variation to intervals outside recorded CUDA
activity. Submit/wait include driver and scheduling waits; this does not identify
a CPU instruction hotspot or prove WDDM preemption.

DFlash2 has an adverse outlier:747.18 ms prefill and671.90 ms decode. All four
initial repetitions have the same24 rounds and103 accepted drafts. In the two
targeted repeat processes the largest decode is630.60 ms, again with increased
kernel elapsed time, so the symptom is not dismissed as noise. Across all eight
DFlash2 repetitions, native decode varies16.04% and prefill13.37%; decode kernel
union varies13.64% (555.65–631.46 ms), while outside-CUDA decode varies23.78–41.14 ms.
This differs from Plain's predominantly uncovered intervals. Fixed sampled clocks
and matching acceptance counts do not explain the kernel elapsed increase;
preemption, competing activity, residency and profiler effects remain unseparated.
No deterministic scheduling root cause or stabilization fix is established.

Prefill kernel union covers most prefill wall time. Native-minus-NVTX-scope
residuals are retained rather than forced to zero: the Engine and NVTX scopes
use different boundaries. Transfers/memsets are included separately so their
time is not labeled CPU overhead. CPU scheduling was disabled in these light
captures, and WDDM was not traced: **both are incomplete**. Earlier warned CPU/WDDM
captures from PR #3 do not close this gap.

The [complete phase CSV](rtx4090-stability-2026-10-phases.csv) retains every
repetition's throughput, TTFT, scope residuals, submit/wait intervals and phase
telemetry. Sampled board peak VRAM per process is retained too; it includes the
desktop/other allocations and is not an exact per-process allocation peak.
Raw JSON contains Engine-owned resource peaks. Half-second board sampling has
only1–4 observations in many short phases and cannot rule out excursions.


## Reproduction and checks

```powershell
.\scripts\build-rtx4090.ps1 -Test -Target ninfer_tests -Jobs 8
ctest --test-dir build-rtx4090 -R '^ninfer_attention_precision_' -j1 -V
py -3.11 -m pytest tests/test_target_trace.py tests/test_engine_phases.py tests/test_gguf_bench_reader.py tests/test_bench_matrix.py tests/test_tune_rtx4090.py tests/test_speed_of_light.py -q
```

Apply `tools/bench/qwen_target_trace.patch` only for a separately named diagnostic
CLI, then revert the patch and rebuild the normal CLI. The patch synchronizes
and copies raw tensors; its timings are unsuitable for performance claims.

```powershell
git apply tools/bench/qwen_target_trace.patch
.\scripts\build-rtx4090.ps1 -Target ninfer -Jobs 8
Copy-Item build-rtx4090/apps/ninfer.exe build-rtx4090/apps/ninfer_stability_trace.exe
git apply -R tools/bench/qwen_target_trace.patch
.\scripts\build-rtx4090.ps1 -Jobs 8
```

With the diagnostic binary:

```powershell
py -3.11 tools/bench/trace_qwen_greedy.py --study stability --exe C:/coding/ninfer-all/build-rtx4090/apps/ninfer_stability_trace.exe --weights C:/ninfer-all/models/Huihui-Swift15-Abliterated-IQ3S-DFlash2.ninfer --prompts bench/fixtures/greedy_target --out profiles/bench/new-target-matrix
py -3.11 tools/bench/analyze_stability_matrix.py profiles/bench/new-target-matrix
py -3.11 tools/bench/profile_gguf_variance_4090.py --nsys "C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.3.2/target-windows-x64/nsys.exe" --exe C:/coding/ninfer-all/build-rtx4090/bench/ninfer_bench.exe --weights C:/ninfer-all/models/Huihui-Swift15-Abliterated-IQ3S-DFlash2.ninfer --out C:/coding/ninfer-all/profiles/bench/new-phases --prompt 2048 --reps 2 --captures 2 --modes fixed --whole-process
py -3.11 tools/bench/analyze_engine_phases.py profiles/bench/new-phases
```

Repeat the profiler command in another output directory with `--dflash2` for its
separate baseline pair; add `--cpu-context --wddm` only when those incomplete
traces are needed. `--profile-measured` intentionally still requires one test and
one repetition; use whole-process capture for multiple measured reps.

Affected native checks pass: report schema contract, exact argmax/Top-2,
speculative-round semantics, CPU GGUF conversion, and the three precision
variants. The six affected Python suites pass56 tests. Syntax compilation,
diagnostic patch applicability and whitespace checks are recorded with the
final evidence. No CUDA implementation, model source or dispatch default is
changed relative to the PR base.

The first unbounded parallel build exhausted host memory; it was stopped and
restarted at bounded parallelism (first4, then8 with measured headroom). The
Windows wrapper initially lacked target selection and was corrected. Final
native builds succeed. An existing unused `tile_acc` warning in the FP32 prompt
specialization remains; it is unrelated to these documentation/tests.

Limitations: one WDDM RTX4090, driver617.42, CUDA13.4/MSVC19.44; no other GPU or
artifact qualified. The numerical matrix does not measure performance. No stable
speedup, universal token identity, arbitrary-input accuracy bound or resolved
historical scheduling cause is claimed. No merge or activation of experiments.
