# Basalt Flash-Next packs

Convert a local Qwen3.8-Flash-Next `BASALTMF` v1/v2 pack into NInfer v3. The converter reads the
pack's effective dense weights, routed experts, PLE table, and any included MTP and Vision
components. It uses the pack itself, including its tokenizer and chat template, as the source.

From the repository root, with the [conversion dependencies](weight-conversion.md) installed:

```bash
python -m tools.convert.basalt model.basalt --output model.ninfer
```

`--verify-sections` checks the source section SHA-256 values before conversion. `--rows-per-chunk`
(default 512) bounds conversion batches; `--max-file-gib 16` splits the output into artifact shards.
Keep all resulting shards together. Expert and PLE data are streamed directly from the source;
temporary disk space holds only frontend resources and the embedded Vision GGUF. The destination
needs space for the converted weights as well as the original pack.

## What is preserved

| Source | NInfer representation |
|---|---|
| Q8PLE `per_layer_token_embd.weight` | Original Q8_0 codes and FP16 scales, with the table digest recorded in the artifact |
| Routed experts | Original quantized blocks, reorganized into NInfer's parameter layout |
| NVFP4 expert scales | Original FP32 gate/up/down multipliers, applied by the GGUF expert path |
| `dense.index` overrides | Effective stored values, taking precedence over the pack's original GGUF |
| HC8 | Original int8 codes and FP16 scales, reblocked from separate planes into Q8_0 blocks |
| HC injection | Both the original BF16 prompt operand and the HC8 decode operand |
| Norms, convolutions and GDN decay | FP32 values; FP16/BF16 values widen exactly, norm gamma stays unfolded, and negative decay coefficients avoid a log/exp round trip |
| Included MTP | The pack's own dense weights, quantized experts, norms and HC8; the existing NInfer MTP executor consumes them |
| Included Vision | BF16 projections plus FP32 direct norms, biases and positional embeddings |
| Frontend | Token IDs, BPE merges, special tokens, chat template and sampling defaults |

Q8PLE is independent of expert precision. Each 160-value row occupies 170 bytes: five Q8_0 blocks
containing signed codes and FP16 scales. Conversion records the source table's SHA-256 and
preserves each expert matrix's stored quantization.

The artifact records its weight conventions with `fp32_direct`, `norm_unit_offset`,
`hc_prefill_inject`, `gdn_a_is_log` and `expert_output_scales` in the text config, and
`direct_format` in the Vision config.
Existing artifacts retain their defaults. A v1 pack has no stored HC8 section; conversion constructs
the HC8 copies using Basalt's FP16-scale, nearest-even quantizer.

Original metadata, tokenizer resources, expert profile and MTP draft-vocabulary data are retained
as artifact resources. Expert placement uses NInfer's planner; its MTP executor uses the full draft
head rather than Basalt's vocabulary-subset optimization. DFlash2 remains a separate, explicit
serving choice.

Scope is Flash-Next geometry with Basalt expert layouts v3/v5/v6/v7, including mixed GGUF
quantizations, native NVFP4 expert packs and Q8PLE tables. NVFP4 retains its per-expert scale tail
as a separate FP32 operand. Scaled experts use the vector/matrix GGUF route rather than the
unscaled fused-decode route; existing unscaled artifacts keep their current execution paths.
Unknown sections, incomplete components and layouts outside this scope stop conversion with a
named error. Matrix values that cannot be represented by the destination's BF16 projection path
also stop conversion instead of silently rounding. Container framing and execution kernels differ:
preserving stored weights does not imply byte-identical files or bit-identical generated answers.

## Serving JSON

Convert the selected Basalt launch entry using the same source pack:

```bash
python -m tools.convert.basalt_config serve.json --pack model.basalt \
  --artifact model.ninfer --output ninfer.json
ninfer-serve --config ninfer.json
```

`--model-index` selects an entry in a multi-model Basalt config. The converter writes
`ninfer.json.report.json` with the source settings, mapped policies, and fields requiring a decision.
It writes the launch file only after those fields are resolved. Review the report before serving.

Parallel slots map to shared-engine `max-concurrency`; the source context and `slot-full-context`
choice determine per-request context and aggregate capacity. Seed-inclusive speculative counts
map to NInfer's draft count. Pack sampling defaults apply before explicit launch values. Jev uses
NInfer's shared scheduler rather than Basalt's separate lane reservation.

Basalt `kv: "fp16"` maps directly to NInfer `kv-dtype: "fp16"`: FP32 key/value projections and
key normalization/rotation, rounded once into native FP16 cache planes. Main and MTP attention
consume those planes; remaining activation and state representations retain NInfer's conventions.
See [native FP16 KV](qwen3-8-flash-next.md#native-fp16-kv).

Some settings have different semantics. Elastic allocation is not a fixed capacity reservation.
Flash-Next Vision is resident in NInfer;
expert-cache sizing and host residency have their own allocation policies. These differences are
reported rather than used to silently lower context, change precision, or select a new drafter.

For a deliberate choice, name the source field with `--acknowledge SOURCE_OPTION` and provide
the destination value with `--set TARGET_OPTION=JSON_VALUE`. For example, to choose resident
Vision explicitly when the source used dynamic Vision placement:

```bash
python -m tools.convert.basalt_config serve.json --pack model.basalt \
  --artifact model.ninfer --output ninfer.json \
  --acknowledge vision --set vision=true --set vision-residency=resident
```

Use the source-option names printed in your report; further decisions may be required. An
acknowledgement alone cannot satisfy a field that requires an explicit destination choice.
The generated [flat JSON launch file](cli.md#launch-configuration-files) follows the same layering
and validation rules as INI files and command-line options.
