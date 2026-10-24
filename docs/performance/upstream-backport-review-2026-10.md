# Targeted upstream review: RTX 4090 / Qwen3.8-27B

Source reviewed: `Neroued/ninfer` at `64d10fbf9f8e6bfc79ab0570b94674c5b9997929`.
Fork baseline: `spk-c/ninfer-all` at `cae6a232ca3c278d7d5573a948371e231d9448be`.

This is an **investigation branch**, not a release or speedup qualification.
The production kernel arithmetic, dispatch defaults, model artifact and `master`
are intentionally unchanged. No end-to-end throughput improvement is claimed.

## Scope of upstream changes

| Upstream change | Fork disposition | Reason |
| --- | --- | --- |
| `4201b5d2` DFlash per-chunk prefill binding | **Already covered in the fork** | `tests/models/qwen3_5/test_dflash_prefill_real.cpp` exercises forked state slots, conflicting decode bindings, split chunks and oversized appends for DFlash and DFlash2. `src/models/qwen3_5/program/prefill.cpp` binds current prefill destinations. Keep regression, not duplicate upstream's refactored implementation. |
| `0a8530d4` XGrammar counted JSON scalar repetitions | **Already present, divergent implementation** | Fork `third_party/xgrammar/cpp/json_schema_converter.cc` constructs the counted scalar repeat; `compiled_grammar.cc` and `grammar_functor.cc` have `PopulateRepeatMasks` and `NormalizeRepetitionUnit`. Fork's grammar cache serialization ID is customized. Do not overwrite it with upstream's `v18`. |
| `19367582` CUDA Graph startup memory calibration | **Not ported** | Broad runtime/loader ABI changes (59 files in upstream), potentially incompatible with fork's DFlash2, residency, CUDA contexts and memory planning. First obtain matched, repeated startup VRAM measurements. |
| `a8e212ac` context, tokenizer and transfer overhead | **Not ported** | Large runtime scheduling change; isolate time in prepare/prefill/decode and establish paired end-to-end tests first. |
| `569861a3` 10240 offset RMSNorm specialization | **Not ported** | Targets a particular geometry, not an established hotspot in our 27B GGUF trace. |
| `bace20dc` configurable CUDA scheduling | **Already present** | `NINFER_CUDA_SYNC` with spin/blocking/yield/auto exists in `src/core/device.cu`. |
| `abb7f14f` Prometheus metrics | **Already present** | `GET /metrics` is documented and exposed by the fork. |
| Qwen4Exp Text/MTP/Graph and expert cache | **Separate model initiative** | Qwen3.8-Flash-Next is not the current dense Qwen3.8-27B DFlash2 artifact. |

## Safe CUDA Graph memory experiment

The existing graph allowance sweep was made configurable from PowerShell and can
now explicitly select DFlash2. Its numbers are **device-wide allocation
deltas**, not VRAM attributable solely to NInfer. The ordinary-graph count
calculation is a legacy approximation and does **not** describe MTP/DFlash2
capture topology. In those modes the report uses `null` instead.

Run from the repo root, after building a Windows sm_89 CLI. Close other GPU
workloads and repeat under comparable driver, clocks, model and cache settings.

```powershell
python -m tools.bench.run_graph_allowance_sweep `
  --cli ".\build-rtx4090\apps\ninfer.exe" `
  --model "C:\ninfer-all\models\Huihui-Swift15-Abliterated-IQ3S-DFlash2.ninfer" `
  --spec dflash2 --draft-tokens 5 --kv-dtype rk4v4 `
  --prefill-chunk 1024 --capacities 8192 16384 32768 `
  --repeats 5 --output ".\profiles\upstream\baseline"
```

To compare two checked-in builds with the **same model path** and settings,
run the sweep a second time using the candidate executable and a different
`--output` directory. Then inspect the observed deltas:

```powershell
python -m tools.bench.compare_graph_allowance_sweeps `
  ".\profiles\upstream\baseline\graph-allowance-dflash2-rk4v4.json" `
  ".\profiles\upstream\candidate\graph-allowance-dflash2-rk4v4.json" `
  --output ".\profiles\upstream\comparison.json"
```

The comparator rejects mismatched model, backend, draft width, KV codec and
prefill chunk; only matching capacities are compared. Failed startups, missing
measurements and 0-byte deltas are excluded from valid samples. Non-overlapping
sample ranges indicate a **VRAM observation**, not evidence of a throughput
gain or its cause. Overlapping ranges remain inconclusive.

The runner exits with status 2 if the CLI supplies no recognizable graph
figures, rather than silently treating unavailable instrumentation as zero.

## Separate, required inference-speed qualification

Use the already existing `tools/bench/run_interleaved_ab.py`,
`tools/bench/analyze_engine_phases.py` and `docs/rtx4090.md` for matched
decode, prefill and TTFT comparisons. In particular, our previously measured
DFlash2 baseline showed material run-to-run variance, so a purported
improvement smaller than the paired baseline spread is unqualified.

No numerical default, CUDA kernel, cache behavior, release or `master` change
is authorized by these diagnostics. Validate the Python regression tests with:

```powershell
python -m pytest -q tests/test_graph_allowance_sweep.py tests/test_compare_graph_allowance_sweeps.py
```
