# Dense Qwen4

NInfer runs dense Qwen4 (`Qwen4ForCausalLM`, the text component of
`Qwen4ForConditionalGeneration`, text `model_type` `qwen4_text`): the blocks of
[Qwen3.8-Flash-Next](qwen3-8-flash-next.md) (`Qwen4ExpForCausalLM`) with a SwiGLU MLP in every
layer and in the MTP block in place of the MoE. It is a form of the Flash-Next family: the same
load, executor, Engine core, frontend, n-gram table, context cache, MTP drafting and tensor
parallelism.

No dense Qwen4 checkpoint, `config.json` or reference implementation is public (checked
2026-11-22). Qwen's Qwen4-Max is an API model with the same organs as Flash-Next, and its
announcement says smaller Qwen4 models will be open-sourced in the same lineage. The geometry the
engine accepts follows the public preparations for a dense Qwen4-27B: TensorFold's
`docs/recipes/qwen4-dense.md`, tqllm's pull request 15, and the shapes of the "Qwen4-27B dense"
kernels in Intel's vllm-xpu-kernels pull request 22 (hidden 5120, hyper-connection rank 640, 24
query and 4 KV heads of 256, 48 Gated DeltaNet value heads of 128, an MLP of 14,336). Until a
release exists, the code is checked on a synthetic checkpoint of that geometry
([checks](#checks-on-a-synthetic-checkpoint)); a release whose config or tensor names differ needs
the converter and the config parser adjusted, and its weights need the checks below.

## What the engine accepts

`tools/convert/qwen4_exp.py` reads `config.json` of an `architectures`
`["Qwen4ForConditionalGeneration"]` or `["Qwen4ForCausalLM"]` checkpoint and the C++ parser
(`src/models/qwen4_exp/config.cpp`) checks the text config it writes. Both refuse every other value
by name.

| Field | Accepted |
|---|---|
| `hidden_size`, `hc_lowrank` | 2560 and 320, or 5120 and 640 (`hc_count` 4) |
| `num_attention_heads`, `num_key_value_heads`, `head_dim` | 24 query heads of 256 over 2 or 4 KV heads, `attn_output_gate` with `output_gate_type` `"sigmoid"` (declared: TensorFold reads an absent value as SiLU) |
| `linear_*` | Gated DeltaNet with 16 key heads and 48 value heads of 128, a convolution of 4 |
| `indexer_*` | 4 indexer heads of 128, one key head, compression 4, budget 2048 |
| `intermediate_size` | a multiple of 256 (the MTP block's `intermediate_size` defaults to it) |
| `layer_types`, `num_hidden_layers` | any number of `linear_attention` and `full_attention` layers |
| `ple_layer_ids` | empty, or one Gated DeltaNet layer with Flash-Next's PLE fields (an embedding of 2560 from 16 n-gram heads of 160, n-grams of 3, a convolution of 4) |
| MoE fields, `mlp_layer_types` other than dense | refused |

A checkpoint without PLE leaves the PLE and n-gram fields out; its model has no n-gram table and
reads none. TensorFold's `split_ngram_parts` only shards the table for training and is ignored.

## Converting

The checkpoint's safetensors are the source; no GGUF layout of dense Qwen4 exists, and the
converter refuses `--source gguf` and `--source pack` for it.

```bash
# Text, MTP block and the n-gram table in one file (a model with PLE).
python3.11 -m tools.convert --model models/Qwen4-27B --recipe qwen4_dense \
  --components text,mtp,ngram --source ngram=models/Qwen4-27B \
  --out models/qwen4-27b-ninfer-v3.ninfer

# With the Vision tower (`model.visual.*`) of a Qwen4ForConditionalGeneration checkpoint; the
# folder also needs preprocessor_config.json and video_preprocessor_config.json.
python3.11 -m tools.convert --model models/Qwen4-27B --recipe qwen4_dense \
  --components text,mtp,ngram,vision --source ngram=models/Qwen4-27B \
  --out models/qwen4-27b-vision-ninfer-v3.ninfer

# A model without PLE has no table.
python3.11 -m tools.convert --model models/Qwen4-27B --recipe qwen4_dense \
  --components text,mtp --out models/qwen4-27b-ninfer-v3.ninfer
```

The `qwen4_dense` recipe sizes the 27B geometry for one 24 GB card:

| Parameters | Format |
|---|---|
| MLP gate and up | Q4 (`q4_g64_fp16`) |
| MLP down | Q5 (`q5_g64_fp16`) |
| Gated DeltaNet and attention query and key | Q4 |
| Their other projections (value, z, gate, output) | Q5 |
| Hyper-connection down, up and inject | HC8 (`gguf_q8_0`, the Flash-Next GGUF releases' kernels) |
| Token embedding, head | Q8, Q6 |
| MTP block's MLP and attention | Q8 |
| Indexer, PLE, norms, convolutions, Gated DeltaNet `a`/`b` projections, Vision tower | kept as stored (BF16) |
| n-gram table | FP8 rows from the checkpoint's BF16 shards, or an IQ4_NL GGUF table |

At the 27B geometry with 64 layers this artifact is 19.4 GB with the table (17.6 GiB of device
weights) and 20.3 GB with the Vision tower. A recipe file that assigns nothing keeps every weight in
BF16, the representation the FP64 reference compares. `--layers A..B` writes a qualification slice
of source layers A to B-1 with the embedding, the final mixer and the head, as for Flash-Next.

## Running

```bash
# One GPU.
./build/apps/ninfer-serve models/qwen4-27b-ninfer-v3.ninfer --max-context 32768

# MTP drafting from the model's MTP block.
./build/apps/ninfer-serve models/qwen4-27b-ninfer-v3.ninfer --max-context 32768 \
  --spec mtp --draft-tokens 3

# Two GPUs as pipeline stages, or dividing every layer (Linux).
./build/apps/ninfer-serve models/qwen4-27b-ninfer-v3.ninfer --devices 0,1 --max-context 65536
./build/apps/ninfer-serve models/qwen4-27b-ninfer-v3.ninfer --devices 0,1 --tensor-parallel \
  --max-context 65536
```

`ninfer` and `ninfer-perplexity` take the same options. Everything the Flash-Next guide describes
for its text path applies: one to eight concurrent requests in prompt chunks, the context cache
with host and disk images, a checkpoint inside a long last message, the prefix cache file and
session slots ([Serving](serving.md#context-cache-of-the-other-families)), every Flash-Next
`--kv-dtype`, the n-gram table's residencies (`--ngram-residency`, `--ngram-table`,
`--no-ngram-table`), structured output, [banned words](serving.md#banned-words), log
probabilities, thinking budgets and MTP drafts for several requests. A prompt call of more than
eight tokens attends on tensor cores where they measured faster than the SIMT kernel: with
two KV heads as in [Flash-Next](qwen3-8-flash-next.md#prompt-attention-on-tensor-cores), with four
for a BF16 or FP16 cache, `bf16p` too on Ampere and Ada, and on Turing every format but `bf16`
and `bf16p` ([measurements](#prompt-attention-on-tensor-cores)).
Tensor parallelism divides the Gated DeltaNet key heads, the sparse-attention KV heads with their
cache and the MLP rows in shares sized to each card (`--tensor-split`); the head, the MTP block and
the n-gram rows stay on the first device. The devices' MLP partial sums are exchanged as FP32 and
added in a fixed order, so the result depends on the split (Flash-Next's expert sums are exact
fixed-point sums).

Dense Qwen4 has no routed experts. The engine refuses `--expert-residency host|disk`,
`--expert-cache-mib`, `--expert-host-mib`, `--expert-dma-share`, `--expert-route-bias`, the other
hybrid-expert options and routing profiles for it. No DFlash2 drafter exists for it.

`--vision` runs the artifact's `vision` component: the checkpoint's Qwen3.5 Vision tower, whose
merger writes the text model's hidden width, resident on the device through the Flash-Next Vision
path, with images and video in the context cache.

## Checks on a synthetic checkpoint

[`qwen4_dense_synthetic.py`](../tools/reference/qwen4_dense_synthetic.py) writes a safetensors
checkpoint of the geometry in `tests/fixtures/qwen4` with seeded random weights: 64 layers of 5120
with 4 KV heads and an MLP of 14,336, PLE on layer 1 with a table whose rows FP8 represents exactly,
and the MTP block (55.7 GB of BF16). `--no-ple` writes a model without PLE.

```bash
python3.11 -m tools.reference.qwen4_dense_synthetic --out models/qwen4-synth \
  --tokenizer models/Qwen3.8-Flash-Next
python3.11 -m tools.convert --model models/qwen4-synth --recipe qwen4_dense \
  --components text,mtp,ngram --source ngram=models/qwen4-synth --out models/qwen4-synth.ninfer
```

Results on two RTX 3090s at 350 W (CUDA 13.4, sm_86, 2026-11-22):

| Check | Result |
|---|---|
| Op tests at the 5120 geometry (hyper-connection, PLE injection, sparse attention with groups of 6 and 12) against their FP64 oracles; `compute-sanitizer` racecheck and synccheck | pass; 0 hazards, 0 errors |
| A 4-block slice in BF16 against the FP64 reference (`tools.reference.qwen4_exp --compare`, prefixes of 128 and 2,112 tokens, 16 steps each) | logits relative L2 2.8-3.0e-3, KL 0.7-1.3e-4; the top-1 token differs at 1 of 32 steps, where the FP64 logits of the two tokens differ by 0.025 |
| The same slice in the `qwen4_dense` recipe | relative L2 4.9-5.4e-2, KL 0.9-1.1e-2; top-1 differs at 5 of 32 steps (FP64 margins up to 0.22) |
| `ninfer_qwen4_exp_generate_real`: the 64-layer artifact on one GPU and as two stages; an 8-layer model without PLE | pass: graph and eager decode, steps in turn, batched repeats, restored sequence images, K/V-only steps, speculative repeats, packed prompts |
| `ninfer-perplexity`, 1,900 tokens, every Flash-Next `--kv-dtype` | mean NLL 16.5735-16.5755 (`bf16` 16.574581; `bf16p` and two stages equal to it; tensor parallelism 16.575851) |
| `ninfer_qwen4_exp_engine_real`, two stages, MTP with 1, 4, 8 and 15 drafts | pass except the checks that need a trained model (below) |
| Its media test on the 64-layer artifact with Qwen3.8-27B's Vision tower | pass: follow-up turns past an image, a new image after a reused prefix, a changed image encoded, images spilled to disk and restored by a restarted Engine |
| `ninfer-serve` with the context cache, a 1,612-token prompt three times | 1,595 and 1,607 prompt tokens reused (shared prefix, turn closure) |

`ninfer` with `--kv-dtype int8`, a 2,807-token prompt and 256 tokens, one request:

| Devices | Prompt | Decode |
|---|---:|---:|
| one RTX 3090 | 853 tok/s | 35.9 tok/s |
| two, `--tensor-parallel` | 1,150 tok/s | 45.4 tok/s |

Random weights do not check what a trained model does. The MTP block's drafts are not accepted (0
of 762), so `--spec mtp` decodes slower here (21.4 tok/s). The model emits no end-of-sequence token,
which the Engine test's EOS check needs. Near-tied tokens are common: plain and MTP decoding first
differ at a token whose top two log probabilities are 0.125 apart, and the Engine test's stop-token
case, which stops at the plain decode's sixth token, then runs to its output limit. The model's
quality, MTP acceptance and speed with real weights are unmeasured until a checkpoint is released.

## Prompt attention on tensor cores

With four KV heads, each KV head has six query heads, and the tensor-core kernel puts them in its
16-row tile, so each staged K and V tile serves six rows instead of twelve. The sparse attention Op
at 1,024 queries over 2,051 positions (the first query at position 16,384), four KV heads, the SIMT
kernel's time over the tensor-core kernel's, 2026-11-23 (CUDA 13.4, the release build's
architectures):

| KV format | RTX 3090 | RTX 5080 | TITAN RTX |
|---|---:|---:|---:|
| `bf16` | 2.54 | 2.83 | 0.29 |
| `fp16` | 1.51 | 1.46 | 1.53 |
| `bf16p` | 1.09 | 0.98 | 0.22 |
| `int8` | 0.79 | 0.66 | 1.16 |
| `rk8v4` | 0.80 | 0.66 | 1.16 |
| `fp8` | 0.84 | 0.72 | 1.21 |
| `k8v4` | 0.84 | 0.72 | 1.24 |
| `nvfp4` | 0.86 | 0.71 | 1.21 |
| `rk4v4` | 0.84 | 0.67 | 1.21 |
| `rk4v4-e8` | 0.83 | 0.66 | 1.20 |
| `rk2v4-e8` | 0.80 | 0.68 | 1.13 |

The engine runs the tensor-core kernel where the column is above 1: `bf16`, `fp16` and `bf16p` on
Ampere and Ada (Ada follows the RTX 3090; no Ada card was measured), `bf16` and `fp16` on
Blackwell, every format but `bf16` and `bf16p` on Turing, whose BF16 tiles it emulates. Volta keeps
the SIMT kernel. `ninfer_sparse_attention_test` passes on the three cards with each kernel forced
for every call and with this choice, and `compute-sanitizer racecheck` on the RTX 3090 and the
TITAN RTX reports no hazards in the tensor-core kernel's calls of six query heads. The end-to-end
prefill speed of a model with four KV heads was not measured.
