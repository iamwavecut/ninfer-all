# Flash-Next from native sources with DFlash2

`qwen3_8_flash_next_gguf_preserve` builds a Flash-Next artifact directly from GGUF text weights,
an n-gram source and an optional BF16 Vision projector. It requires no `.basalt` input. The
ordinary `qwen3_8_flash_next_gguf` recipe retains its existing conversion choices.

## Operand conventions

The preservation recipe keeps each routed expert's encoded blocks and scales, including a
mixed-format expert bank. Row permutations change addressing, rather than requantizing those
blocks. Stored norm gamma remains FP32 with the runtime's unit offset disabled; GDN decay
remains the stored negative `-exp(A_log)`. GDN convolution remains FP32. PLE convolution uses
Basalt's explicit FP16 load-time rounding and is represented exactly in the FP32 operand.

HC8 uses an FP16 block scale followed by nearest-even rounding against that stored scale.
Prefill retains a separate BF16 injection operand. These are distinct from the ordinary GGUF
recipe's ggml Q8_0 HC quantizer, which rounds codes against the unrounded scale.

`--compat-bf16` explicitly reproduces Basalt's compatibility casts for the operands its
converter requires in BF16: HC matrices, routers, indexer projections, GDN controls and PLE
projections. The native Q2_0/Q8_0 PLE key remains encoded. This flag leaves routed experts and
other encoded projections untouched, and is recorded in artifact provenance. Without it,
incompatible operands are refused rather than silently rounded. Vision matrices must already
be exactly representable in BF16; Vision vectors remain FP32.

## Q8PLE directly from NVIDIA FP8

The `ngram` source may name NVIDIA's `model-fp8-mtp-ple.safetensors`, beside its `config.json`.
Its consecutively numbered E4M3FN shards and one BF16 `ngram_embedding.weight_scale` are read in
bounded chunks. Codes multiplied by that scale in FP32 are reblocked with ggml's reference Q8_0
rounding, with no intervening BF16 cast. The result is five 34-byte blocks per 160-value row.
Existing BF16 HF and GGUF n-gram sources retain their existing formats.

`--ngram-sha256 HEX` requires the SHA-256 of the encoded table before any artifact is written.
It works both for a standalone table and for text that embeds or references the table. The
ordinary conversion report retains the table format and digest.

```bash
python3.11 -m tools.convert --model /models/Flash-Next-config \
  --recipe qwen3_8_flash_next_gguf_preserve --compat-bf16 \
  --components text,ngram,vision \
  --source gguf=/models/text-00001-of-00002.gguf \
  --source ngram=/models/nvidia/model-fp8-mtp-ple.safetensors \
  --source vision=/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf \
  --ngram-sha256 e02bf3d78153f7c95390ffca1b4279776c90ef302a345aa56bf6b3b694d65e42 \
  --device cpu --rows-per-chunk 65536 --out /models/flash-next-q8ple.ninfer
```

The digest above identifies the 54,400,261,120-byte Q8PLE payload published in the
[Q8PLE source recipe](https://huggingface.co/Shockem/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-Q8PLE-Basalt).
Its original FP8 source is
[`nvidia/Qwen3.8-Flash-Next-NVFP4` at `fc694b54`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/tree/fc694b54).
Use the text GGUF of the required quantization and fine-tune, rather than regenerating learned
expert grids from floating-point weights. `--model` supplies the matching tokenizer, template,
generation configuration and, with Vision, image/video preprocessor configurations.

## Attach and select DFlash2

The target conversion and drafter attachment are separate operations. The attachment preserves
all existing artifact objects and appends the drafter's objects; its report verifies their hashes.

```bash
python3.11 scripts/pods/attach_dflash2.py \
  --base /models/flash-next-q8ple.ninfer \
  --gguf /models/Qwen3.8-Flash-Next-DFlash2-Q4_0.gguf \
  --config /models/Qwen3.8-Flash-Next-DFlash2/config.json \
  --output /models/flash-next-q8ple-dflash2.ninfer

ninfer-serve /models/flash-next-q8ple-dflash2.ninfer \
  --model-id flash-next --expert-residency host --ngram-residency disk \
  --max-context 262144 --max-concurrency 2 --kv-resident all --kv-dtype bf16 \
  --spec dflash2 --draft-tokens 7 --vision
```

Use [nazeshinjite's Q4_0 GGUF at `cbe1b23f`](https://huggingface.co/nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF/tree/cbe1b23f)
and [incoai's draft configuration](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2).
`--spec dflash2` selects that drafter even when an artifact also contains MTP; MTP is not loaded.
The example deliberately omits MTP from the target. A stock MTP GGUF is a different drafter from
a separately self-distilled MTP section. DFlash2 requires its trained target vocabulary and hidden
geometry; its attachment checks those contracts. Pipeline stages are unsupported for this path.

The launch keeps BF16 KV and full device-resident context explicit. Main K/V alone uses 6 GiB
per full 262,144-token sequence, plus recurrent state, draft rings, weights, workspace and the
expert cache. Host experts consume substantial pinned RAM; `--expert-residency disk` changes
placement when that budget is unsuitable, without changing the encoded weights. The Q8PLE table
remains file-backed rather than loading another 50.7 GiB into RAM.

These steps reproduce source payloads and the stated conversion conventions. `.ninfer` and
`.basalt` have different containers, layouts, optional drafters and execution arithmetic; neither
whole-file byte identity nor identical generated answers follows from matching weight payloads.
