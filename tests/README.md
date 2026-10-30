# Tests

The retained tests protect current `.ninfer`, numerical operator, model, runtime-transaction,
benchmark-report, and external protocol behavior. Repository verification principles are defined in
[`../AGENTS.md`](../AGENTS.md); Op contract and CUDA implementation guidance is in
[`../docs/maintainer/op-development.md`](../docs/maintainer/op-development.md).

## Organization

- `artifact/` — v3 framing, directory/binding records, codecs, sharding, selected-object
  materialization and Python-writer/C++-reader interoperability;
- `convert/` — source interpretation, Qwen logical mapping, recipe overrides/sharing, optional
  components, resources, proposals and numerical conversion methods;
- `reference/` — bounded checkpoint range fetch, exact subset bytes and hash manifests;
- `models/qwen3_5/` — config/binding, frontend, state/context stores, workspace, MTP alignment and
  opt-in real Engine integration;
- `ops/` — semantic Op qualification with independent mathematical or state-transition oracles;
  Linear and fused Linear suites are separated by their supported weight/activation paths;
- root C++ tests — core storage, runtime admission/resource policy, public API, serving protocols,
  logging, benchmark reports and causal-scoring evaluation;
- `test_serve_corpus.py` — agreement between the serving request-log schema and its measurement
  consumer.

Tests are grouped by observable risk, not by mirroring every source file or class.
`CMakeLists.txt` includes explicit registrations from `cmake/`, `artifact/`, `models/qwen3_5/`
and `ops/`. Registration helpers live in `cmake/NinferTests.cmake`; included manifests keep
executables and CTest working directories under `build/tests/`.

Tests link into one executable, `ninfer_tests`, rather than one each: every test that links the
Op library would otherwise carry its own copy of the whole kernel image (~450 MB on sm_86), which
added up to tens of GB. Each test keeps its name as a program in the bundle and still runs in its
own process under CTest. Run one by hand with `build/tests/ninfer_tests <name> [args...]`;
`ninfer_tests --list` prints the names. `ninfer_jinja_test` and
`ninfer_artifact_materialization_test` stay standalone because Python tests invoke them by path.
A test's helpers and entry function must be internal (anonymous namespace) so that programs do not
collide at link time. The mechanism is `cmake/NinferBundles.cmake`.
`ops/op_tester.h` and `ops/op_check.h` own only reusable device/guard and comparison mechanics.
Concrete numerical criteria remain named by the semantic Op suite; there are no cross-Op tolerance
presets.

`ops/quantized_weight.h` is the common packed-weight fixture for Q4/Q5/Q6/Q8, FP8 and NVFP4 Op tests. It
owns deterministic payload generation, device `Weight` views, row views, and independent logical
weight decoding.

## Build and run

Select a Python environment with the dependencies for the tests first. The maintained environment
uses Python 3.11; CMake finds Python 3 without restricting its minor version.
`Python3_EXECUTABLE` selects the interpreter used by interop and frontend tests explicitly.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

On a machine without a usable CUDA device -- no GPU, an old driver, or only the CUDA toolkit's stub
`libcuda` (`LD_LIBRARY_PATH` pointing at a directory with `libcuda.so.1` linked to
`$CUDA_HOME/lib64/stubs/libcuda.so`, as CI does) -- every test that needs a device reports as
skipped and the host tests still run. A test asks `tests/cuda_availability.h`; one that fails on its
first CUDA call with the runtime's no-device message is skipped by CTest the same way.

`scripts/check-host-compile.py BUILD_DIR...` compiles the host code of every translation unit of
configured builds (`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`) for syntax only, without device code: each
`.cpp`, and each `.cu` through nvcc's host preprocessing, cudafe++ and the host compiler. A source
whose command differs between the builds only in the architecture numbers is compiled once, and
`--shard I/N` checks the I-th of N parts. CI runs it on Windows with MSVC for every architecture,
since nothing else there compiles Windows code; one Linux configuration takes about 80 s on 28
cores.

Alternatively, `cmake --preset dev` enables products, tests and benchmarks together.
After building, `ctest --preset dev` runs the same CTest suite. See
[Build system](../docs/maintainer/build-system.md) for local interpreter presets.

The chat-template reference test uses Python Jinja2.

Run a focused target for a localized change:

```bash
cmake --build build --parallel --target ninfer_tests
ctest --test-dir build -R ninfer_sampling_test --output-on-failure
```

Enable uniform floating-point error records when establishing or reviewing an Op criterion:

```bash
NINFER_OP_REPORT_STATS=1 \
  ctest --test-dir build -V -R '^ninfer_(rmsnorm|softmax_attention)_test$'
```

Every participating comparison emits one `OP_ERROR_STATS` record containing the stable case label,
actual error, active limit, and error-to-limit ratio. The switch changes reporting only; the same
statistics still drive the normal verdict. Passing tests remain quiet without it.

## Running a test under compute-sanitizer

`memcheck` and `initcheck` both work on these binaries, including the large statically linked ones.
Invoke it unqualified so `PATH` resolves it:

```bash
compute-sanitizer --tool initcheck --error-exitcode 9 \
  build/tests/ninfer_tests ninfer_softmax_attention_test
```

**Use the compute-sanitizer of the toolkit that built the binary.** With several toolkits installed,
an older copy (2024.1.0, from CUDA 12.4) was seen to print its banner and `ERROR SUMMARY: 0 errors`,
exit 0 and **never execute the binary** — a clean report having checked nothing.

Two messages that look like failures and are not:

- *"Target application terminated before first instrumented API call"* — the test made no CUDA
  calls, usually because it skipped for a missing `NINFER_*_WEIGHTS` environment variable. Set the
  variable, or pick a test that does not need one.
- A single `-k 'regex:a|b'`-style argument disappearing in PowerShell: `|` is parsed as a pipeline
  before the tool sees it. One pattern per invocation.

`ncu` needs one thing more: access to the GPU performance counters, or it fails with
`ERR_NVGPUCTRPERM`. On Windows they are administrator-only by default; an elevated shell satisfies
that per run, with nothing persistent and no reboot (`scripts/sweeps/admin-profile.ps1` does it).
Rented GPU containers usually deny the counters altogether; Nsight Systems still works there.

The variable-width DFlash2 target-attention subset can be run with
`./build/tests/ninfer_tests ninfer_softmax_attention_test --dflash2-only`. It covers D256/Q24/KV4 across all five
cache codecs, W=2..16, B=1..8, request-local prefixes, cache effects, and Graph metadata/input
updates. The default executable also runs the existing attention geometries and prefill tests.

`--quantized-depth-only` checks FP8, NVFP4 and K8V4 after 8,192 and 61,440 context tokens,
at one query column and four verification columns, for both D256 geometries. It compares
append-and-attend and cached-only Graph execution over fragmented pages with independently decoded
KV and the FP64 attention oracle. The existing format-specific error criteria and cache, input,
guard and workspace checks apply.
NVFP4 verification also straddles the 8K/16K split tiers and the cached-only grid crossover with
a 65,536-key execution envelope, so the captured capacity does not stand in for actual positions.

Linear tests are independently runnable by weight and activation-compute profile:

```bash
cmake --build build --parallel --target ninfer_tests
ctest --test-dir build -R '^ninfer_linear_(q4|q5|q6|q8)_a16_test$' --output-on-failure
```

All Linear files use `ops/linear/linear_test_common.{h,cpp}` and the same
`ops/quantized_weight.h` fixture as the fused projection tests. The fixture produces the complete
packed GPU payload and exact-decodes the logical float rows used by the one
`cpu_linear_gemm_fp64()` reference. The reference performs naive double accumulation and never
reproduces a production route's activation quantization, staging, reduction tree, or BF16 output
rounding. Each activation compute path selects one centrally defined comparison tolerance for its
whole suite; private kernel, schedule, launcher, and T selection do not change it. Individual test
files call public `linear()` and contain no private selector, launcher, schedule, or kernel
assertions.

The Linear, LinearAdd and LinearSwiGLU common `.cpp` implementations each compile once into a
test support library. Both those libraries and the Op test executables receive the oracle's
`-fno-fast-math` and `-ffp-contract=off` options on GNU/Clang C++ compilers.

Run the native Python suites with the project Python environment:

```bash
python3 -m pytest \
  tests/artifact tests/convert \
  tests/test_serve_corpus.py tests/test_speed_of_light.py
```

The Python suites exercise conversion and encoded output, without running model inference.
The maintained environment uses Python 3.11 with the dependencies for those suites. C++ binding and Engine tests
cover consumption of their resulting representation.

The real loading test accepts an explicit artifact path and optional component selection:

```bash
./build/tests/ninfer_tests ninfer_qwen3_5_loading_real_test \
  --artifact out/qwen3_6_27b.ninfer --vision --speculative mtp --proposal optimized
```

Add `--host-only` to check semantic binding without uploading weights. This does not construct a
Program or establish native Op support.

The C++ prefix/MTP integration test is separately opt-in because it loads the full artifact and
runs the real engine. It accepts Qwen3.6 and Qwen3.8 artifacts; its prompt-token goldens are kept
per model generation because each registers its own chat template:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_6_27b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_5_prefix_real_test --output-on-failure
```

The causal-scoring integration test uses the same artifact variable and checks a full 1,024-column
score tile, overlapping target suffixes, and repeated-window State/KV isolation:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build -R ninfer_qwen3_5_score_real_test --output-on-failure
```

Run the 35B-A3B MoE route independently:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_6_35b_a3b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_5_moe_real_test --output-on-failure
```

Qwen3.8-Flash-Next runs end to end from a converted GSQ-RCO release. The generate test answers
fact prompts and a 4,463-token needle, decodes three sequences as one batch and from a restored
snapshot, and, where decode replays CUDA graphs, checks every token against eager decode.
`NINFER_QWEN4_EXP_EXPERTS` picks `device`, `host` or `disk`, `NINFER_QWEN4_EXP_DEVICES` the pipeline
devices, and `NINFER_QWEN4_EXP_EXPERT_CACHE_MIB` the device expert cache of host or disk experts
(a 24 GB card's on a larger one); a model stored without its n-gram table takes the table artifact
from `NINFER_QWEN4_EXP_NGRAM_TABLE`:

```bash
NINFER_QWEN4_EXP_ARTIFACT=$PWD/models/flash-next-q2_0.ninfer \
NINFER_QWEN4_EXP_NGRAM_TABLE=$PWD/models/flash-next-ngram-table.ninfer \
NINFER_QWEN4_EXP_EXPERTS=host \
  ctest --test-dir build -R ninfer_qwen4_exp_generate_real --output-on-failure
```

Its Ops are also composed over a slice of the BF16 checkpoint against an FP64 golden.
The public Engine regression requires a text+ngram+MTP artifact and, by default, two 24 GiB GPUs.
It qualifies K=1/4/8/15 for output and thinking budgets, post-thinking sampling, penalties, seeded
repetition, JSON grammar and stop tokens, including three concurrent requests and cancellation.
It reports token differences from plain decode and serial batches; exact repetition holds the
execution configuration fixed. Independent Op suites qualify the arithmetic, including the
vector/matrix MoE boundary at T=8/9 and the wider verification batches.
It forces host images to spill, restarts the Engine, and verifies both restored continuation and
isolation between verify widths and table-enabled versus `--no-ngram-table` state:

```bash
NINFER_FLASH_NEXT_ARTIFACT=$PWD/models/flash-next-q2_0-mtp.ninfer \
NINFER_FLASH_NEXT_DEVICES=0,1 \
  ctest --test-dir build -R '^ninfer_qwen4_exp_engine_real$' --output-on-failure
```

`NINFER_FLASH_NEXT_CACHE_ONLY=1` runs the image spill/restore/isolation regression alone.
`NINFER_FLASH_NEXT_SAMPLING_ONLY=1` repeats the seeded K=1 request over three sequence slots,
with and without CUDA graphs, and reports the first differing token's top-logprob records.
On Linux, `ninfer_qwen4_exp_peer_copy_order_test` repeats that workload while delaying alternate
copies of the first prompt chunk's MTP embeddings from GPU 0 to GPU 1. It checks that the next
chunk cannot overwrite the source before the peer copy consumes it. The delay fixture assumes
this prompt's seven MTP cells and device order `0,1`; use that order for this test.

`tools/reference/fetch_slice.py` range-fetches the first blocks, embedding, final mixer, head and
the needed n-gram rows (`--exclude ngram_embedding` keeps the 102 GB table out), and
`tools/reference/qwen4_exp.py --layers 4 --head --out DIR/golden` writes the golden:

```bash
NINFER_QWEN4_EXP_SLICE=$PWD/fn_slice \
  ctest --test-dir build -R ninfer_qwen4_exp_slice_real --output-on-failure
```

`ninfer_qwen4_exp_ngram_writer_interop_test` converts a small sharded BF16 table with the
production Python writer, reads single-file and multipart artifacts through `NgramTableReader`,
and checks reordered and repeated GPU-decoded rows exactly against independently decoded
FP64 products rounded to BF16. `tests/convert/test_qwen4_exp_ngram.py` separately checks the
quantizer against a scalar codebook oracle, source-shard boundaries, digest stability and
invalid source metadata. It does not require the full 102 GB HF table.
The CTest launcher probes CUDA through the native test binary before importing the conversion
dependencies. A host without a usable GPU returns skip code 77 without requiring pytest or
PyTorch; a host with a GPU runs the complete conversion and decoding check.

`tests/convert/test_block_fp8.py` checks HF 128×128 E4M3FN expert matrices against an
independent FP64 codebook/block-product oracle with BF16 or FP32 scales, partial reads and
invalid representations. It also covers fused/split source selection for Qwen3.5 and
Flash-Next text/MTP. Running the file with a range-fetched checkpoint directory qualifies
its actual matrix values at two chunk sizes; `scripts/pods/fp8_source_checks.sh` combines
that real-source check with the affected converter suite.

Token log probabilities run through the public Engine on either family. The test checks that each
record spells its token's text and that the drawn token tops its distribution under greedy
decoding, that gathering them changes no generated token, that streaming, a stop string, three
requests decoded as one batch and a JSON grammar's mask all reach the records, that schemas with an
array, string or integer root constrain the answer from its first token, and that a request
without them gathers none. `NINFER_LOGPROBS_ARTIFACT` names the artifact, and the test is skipped
without it; `NINFER_LOGPROBS_NGRAM_TABLE` names a Qwen3.8-Flash-Next table artifact,
`NINFER_LOGPROBS_SPECULATIVE=mtp` or `dflash2` verifies the tokens in speculative rounds, and
`NINFER_LOGPROBS_CONTEXT_CACHE=hybrid` admits every request through the hybrid prefix cache.
`NINFER_LOGPROBS_DEVICES=0,1` selects a two-GPU layer pipeline:

```bash
NINFER_LOGPROBS_ARTIFACT=$PWD/out/qwen3_8_27b.ninfer NINFER_LOGPROBS_SPECULATIVE=dflash2 \
  ctest --test-dir build -R ninfer_engine_logprobs_real_test --output-on-failure
```

Without `NINFER_TEST_ARTIFACT`, CTest marks these real Engine tests as skipped. Run GPU integration
tests serially. `NINFER_PREFIX_REAL_SCENARIO` selects a focused prefix scenario such as `vision`,
`pressure-resume` or `concurrent`; the default is `all`. These integration checks
use behavior and state accounting rather than another numerical path's generated tokens as a golden.

The capability-evaluation coordinator has its own environment and unittest entry point:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m unittest discover \
  -s eval/tests -p 'test_*.py'
```

Run the serving contract manually after starting a resident server in another terminal:

```bash
./build/apps/ninfer-serve out/qwen3_6_27b.ninfer \
  --host 127.0.0.1 --port 18080
```

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 --model qwen3.6-27b
```

This smoke check is intentionally not a CTest: it needs the real artifact, a supported GPU, and a
server process that remains alive while the client exercises OpenAI Responses/Chat, Anthropic,
state, streaming, and multimodal requests.

The thinking-preservation fixture starts and stops its own server, submits a fixed two-step tool
history, compares stripped and preserved closed-turn prompt lengths, and verifies compatible
prefix reuse, speculative execution, frontier bounds and Responses inheritance:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.ninfer --backend mtp

python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_35b_a3b.ninfer --backend dflash
```

The shared messages are in
[`fixtures/serve/qwen3_6_thinking_preservation.json`](fixtures/serve/qwen3_6_thinking_preservation.json).

## What belongs here

A permanent test should protect one current risk, such as:

- exact artifact bytes, geometry, object binding, or conversion transform;
- a numerical operator contract with an independent oracle;
- model Frontend or Program frontier, prefix, MTP, or multimodal behavior;
- generated-token commit/stop/cancel consistency;
- public benchmark or OpenAI/Anthropic observable behavior;
- a reproduced supported bug.

Performance-only assertions belong in benchmarks and profiler review. Source scans,
implementation-shape assertions, trivial getters/configuration, retired command surfaces, and
broad additions without a concrete regression risk do not belong in the permanent suite.

## DFlash2 Engine integration

The DFlash prefill regression checks actual draft KV contents after a StateImage fork and a
conflicting decode binding, including shortened chunks and oversized local/full KV appends. It
uses native Program storage and the production prefill route; select the draft component stored in
the artifact:

```bash
cmake --build build -j --target ninfer_qwen3_5_dflash_prefill_real_test
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
  build/tests/ninfer_qwen3_5_dflash_prefill_real_test dflash2
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_6_35b_a3b.ninfer \
  build/tests/ninfer_qwen3_5_dflash_prefill_real_test dflash
```

The Engine test uses an artifact containing DFlash2 and checks output budgets, speculative activity,
a forced thinking-control append, penalty-enabled sampling, compact batches with unequal budgets,
same-route same-seed replay,
retained/fresh prefix behavior and absence of a full backend KV pool. A shared DFlash/DFlash2 fixture starts decode at token 63, verifies across the page
boundary, stops after one target column at token 64, and checks the exact retained frontier and
subsequent generation with and without reuse.
The KV Store test checks exact mapping and reservation accounting for the same transition.
K>=7 also exercises a stop inside a licensed block; K=15 additionally checks oversized prefill,
local ring wrap, and the logical context-capacity tail. Optional Vision runs image/video capture
and prefix restore. Zero extra Device StateImage slots exercise Host snapshot/restore.

```bash
cmake --build build -j --target ninfer_tests
NINFER_TEST_ARTIFACT=out/qwen3_8_27b.ninfer \
  build/tests/ninfer_tests ninfer_qwen3_5_dflash2_real_test 15 1 1 8
NINFER_TEST_ARTIFACT=out/qwen3_8_27b.ninfer \
  build/tests/ninfer_tests ninfer_qwen3_5_dflash2_real_test 7 1 0 2 bf16 1 0
NINFER_TEST_ARTIFACT=out/qwen3_8_27b_nvfp4.ninfer \
  build/tests/ninfer_tests ninfer_qwen3_5_dflash2_real_test 2 0 0 2 int8
```

Arguments are K, Graph enabled, optimized head enabled, maximum B, target KV (`bf16` or `int8`),
Vision enabled, and extra Device StateImage slots. Defaults are `15 1 1 8 bf16 0 3`. Run GPU
integration tests serially. The individual Op suites remain the numerical/state-transition oracle;
the fixed Engine fixture does not define bit parity across arbitrary floating-point routes.

## Structured output

The CPU grammar and protocol tests and the GPU sampling/speculative tests qualify this path:

```bash
ctest --test-dir build --output-on-failure -R 'ninfer_(structured_output|sampling|speculative_round|openai_schema|openai_responses|anthropic_schema|cli_options|prompt_input|tool_call_parser|qwen3_5_frontend|qwen3_5_structured_round)_test$'
```

`tests/test_structured_output_live.py` runs a temporary loopback server and validates completed
outputs with the independent Python `jsonschema` validator. Use Python 3.11 with `requests` and
`jsonschema` installed in a test environment. Supply an explicit v3 artifact containing the
selected speculative backends; the test never downloads or converts weights.

```bash
python tests/test_structured_output_live.py \
  --server build/apps/ninfer-serve --artifact /absolute/path/model.ninfer \
  --output-dir work/structured-live --modes none mtp dflash2 dflash2-eager
```

It checks conflicting prompts, greedy and stochastic generation, schema versus JSON object mode,
concurrent and mixed traffic, prefix reuse, SSE, token limits, disconnect cleanup, compile errors,
Responses, and Anthropic Messages. It also runs a tool-result-to-bounded-JSON exchange while
reasoning and multiple tools remain enabled, including a Markdown-formatted example in the prompt.
CPU tests cover numeric endpoints, schema enforcement with tool alternatives, split and mixed-token
reasoning transitions, speculative mask previews, injected thinking-budget closure, and prompt cache
boundary preservation. `--concurrency 8 --draft-tokens 15 --modes dflash2` exercises
the largest draft and batch dimensions. A separate DFlash-capable artifact can use `--modes dflash`.
The server is terminated on success or failure. An occupied test port causes the test to stop.

## Container build cache

Check the Dockerfile's incremental-build and configuration-invalidation behavior without compiling
or running NInfer:

```bash
bash tests/test_docker_build_cache.sh docker
```

This uses the real build stage with a tiny CMake fixture and isolated cache mounts. It checks added
and removed compiler flags, changed and restored cached defaults, environment flags, old header
timestamps, non-code edits, and one-file incremental compilation. `NINFER_TOOLCHAIN_IMAGE=<image>`
stands in for the Dockerfile's toolchain stage, as CI does with the one it publishes. Docker BuildKit is required;
`podman` can be passed instead to check that builder. Logs and the fixture remain in an ignored
`build-cache-test.*` directory; the test image and its small build caches remain in the builder.
The check needs the Dockerfile's build dependencies but no GPU or model weights.

## CPU structured-output qualification

`ninfer_structured_output_test` and `ninfer_unicode_scalar_output_test` use a byte
vocabulary and EOS without loading an artifact or executing CUDA. The stdin JSONL
`ninfer_native_schema_probe` preserves schema declaration order and tests native
compilation, token masks, accepted transitions and EOS. It is an executable fixture,
not an unattended CTest target.

`text/native_schema_oracle.py` generates and independently checks the corpus using
Python `jsonschema`, calendar/time arithmetic, strict UTF-8 and Node.js ECMA262 regexes.
Node is found on PATH, or selected through `NINFER_ORACLE_NODE`. Pass `--schemas` for
an optional JSON list of `{name, schema}` application schemas; these are compile-only.
No model or inference HTTP server is used by these tools.

```bash
cmake --build build --target ninfer_structured_output_test ninfer_unicode_scalar_output_test ninfer_native_schema_probe --parallel 2
ctest --test-dir build -R '^ninfer_(structured_output|unicode_scalar_output)_test$' --output-on-failure
python3 tests/text/native_schema_oracle.py --emit-corpus corpus.jsonl --report oracle.json
build/tests/ninfer_native_schema_probe < corpus.jsonl > native.jsonl
python3 tests/text/native_schema_oracle.py --native-results native.jsonl --report comparison.json
```

These host tests establish supported grammar semantics. Actual tokenizer,
speculative decoding and concurrent serving behavior are exercised separately
by the real-model structured-output tests described above.

The `ninfer_unique_strings_test`, `ninfer_structured_unique_test` and
`ninfer_unique_strings_masks_test` are CPU-only. They exercise decoded-string
equality, remaining enum prefixes, nested scopes, rollback/forks, actual mask versus
commit consistency, multi-item vocabulary tokens, EOS and reasoning/MTP boundaries.
The JSONL `ninfer_schema_normalization_probe` reports normalized schemas for an
independent JSON Schema validator to compare accepted instances. Neither probe
loads model artifacts.

`schema_semantics_oracle.py` generates independent Draft 2020-12 cases for
string uniqueness, escaped equality, finite-domain exhaustion, nested scopes,
union intersections, local references and explicit unsupported contracts.
`compare_schema_semantics.py` compares the native byte-vocabulary probe with
the unchanged input schema. Accepting an invalid value always fails; expected
compile errors and declared serialization restrictions are reported separately.

```bash
python3 tests/text/schema_semantics_oracle.py --output semantic-corpus.jsonl
build/tests/ninfer_native_schema_probe < semantic-corpus.jsonl > semantic-native.jsonl
python3 tests/text/compare_schema_semantics.py --cases semantic-corpus.jsonl --results semantic-native.jsonl --report semantic-comparison.json
ctest --test-dir build -R '^ninfer_(structured_output|unicode_scalar_output|unique_strings|structured_unique|unique_strings_masks)_test$' --output-on-failure
```

Native Qwen MoE plane relocation and streaming arithmetic:

```bash
./build/tests/ninfer_tests ninfer_qwen_native_expert_layout_test
./build/tests/ninfer_tests ninfer_sparse_moe_test --streaming
```

The first test runs on the CPU and checks exact RowSplit planes and NVFP4 scale/divisor relocation.
The second needs a free GPU. It qualifies host delivery against the independent FP64 SparseMoe
oracle at 1, 2, 8, 9, 13, 20, 32, 46, 47, 64, 257, 768 and 4097 columns (NVFP4 without a `120a`
build: 1, 2, 8, 9 and 12), with persistent caches of 0, 16 and 128 MiB, and requires the Q4/Q5,
Q4/Q6 and Q8 outputs to equal the resident route's bits at each width. It replays Graphs and
retains output guards and input-preservation checks. Narrow NVFP4 reads represented BF16 input;
grouped NVFP4 uses the existing admitted W4A4 criterion and arithmetic. The ordinary SparseMoe test checks the resident route. These synthetic
Op tests do not establish full-model quality or Engine lifecycle correctness.

The real-artifact gate compares resident/host/disk causal scores, single-request MTP and each
corresponding lane of a compact eight-request MTP batch. It also checks long prefill, active
cancellation, worker recovery and suspend/Graph restore. Each selected artifact needs MTP:

```bash
NINFER_NATIVE_MOE_ARTIFACT=/absolute/path/to/model.ninfer \
  ./build/tests/ninfer_tests ninfer_qwen_native_streaming_real_test
```

`NINFER_NATIVE_PREFILL_PROFILE=1` selects a bounded matched 2053-token request profile, with one
output token, resident/host/disk placement and cache budgets 0/128 MiB. It prints two request
samples per configuration, copied bytes and memory, excluding model startup from request timing.

The native pipeline gate is a separate, short B=1 test without MTP. It compares resident, host
and disk placement with CUDA Graphs enabled, two fresh 6-token requests per placement, four output
tokens each, explicit KV capacity 512, prefill chunks of 128 and a 128 MiB total expert cache.
It checks corresponding token IDs, both published rank device IDs and allocations, startup-free
monotonic delivery counters, and repeated decode. No download or full model matrix is performed.

```bash
# Two logical ranks on one CUDA device (the default selection).
NINFER_NATIVE_MOE_ARTIFACT=/absolute/path/to/groupwise-model.ninfer \
NINFER_NATIVE_PIPELINE_CHECK=1 NINFER_NATIVE_PIPELINE_DEVICES=0,0 \
  ./build/tests/ninfer_tests ninfer_qwen_native_streaming_real_test

# Two physical CUDA devices. Requires both devices to be free and the groupwise artifact.
NINFER_NATIVE_MOE_ARTIFACT=/absolute/path/to/groupwise-model.ninfer \
NINFER_NATIVE_PIPELINE_CHECK=1 NINFER_NATIVE_PIPELINE_DEVICES=0,1 \
  ./build/tests/ninfer_tests ninfer_qwen_native_streaming_real_test
```

The physical selection disables device-profile calibration (`device_profile=off`) for this
correctness gate. It does not qualify device-profile performance. The runner must record distinct
GPU UUIDs for CUDA devices 0 and 1. A `0,0` result does not qualify two physical GPUs. On `sm_75`,
use the pinned groupwise Qwen3.6 artifact; NVFP4 expert banks are not supported.

## Dense Llama / Nanbeige qualification

`ninfer_dense_decoder_test` checks represented-input FP64 numerical oracles at Nanbeige's width,
GQA 56/8, position 262143 and 262144-key cache, with both 1024- and 8192-key segmentation.
`ninfer_llama_frontend_test` checks the independent HF Metaspace fixtures; its optional argument
is an equivalent fixture carrying a real tokenizer and cases.
`ninfer_llama_real_test` is manual because the full artifact needs a V100-sized device and the
capacity case needs 52 GiB of RAM. See [the model guide](../docs/llama.md#qualification) for its
`logits`, `state`, `state-boundary`, `capacity-fast` and `capacity` modes and the limits of the resulting evidence. Conversion tests
run with Python 3.11: `python3.11 -m pytest tests/convert/test_llama.py`.
