# Inkling 2 Mini

NInfer-all runs Thinking Machines' [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
(`Inkling2LoopForConditionalGeneration`; its text component `Inkling2ForCausalLM`), a looped
mixture-of-experts model of 26.4B parameters with 3.4B active per token (5.4B with its three
passes), from a llama.cpp GGUF: Unsloth's
[UD-IQ4_XS](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) (or its UD-IQ1_M, UD-Q2_K_XL,
UD-Q3_K_XL and UD-Q8_K_XL), IsValorum's
[APEX-I MiniPlus](https://huggingface.co/IsValorum/Inkling-2-Mini-APEX-I-MiniPlus-V2.1-GGUF) or
another quantization of the release, with the release's vision tower (BF16 or Q8_0) and DSpark
drafter. It is the engine's first looped
transformer and its own model family: its own executor (`src/models/inkling2`) and Engine core
(`src/runtime/engine/inkling2_core`), sharing the Qwen3.5 frontend in an Inkling dialect for
tokenization, the chat template, output parsing and the serving protocols.

## What runs

- **The model.** 24 stored layers of width 2304 run as 48 visits: the prelude (layers 0-5), the
  core (layers 6-17) three times with the same weights, then the coda (layers 18-23). Every visit
  has its own keys, values and convolution history. Attention has no rotation: 24 query heads over
  8 key/value heads of width 128, a 16-wide relative feature per query head that scores a learned
  table of 1,024 distances, a sink logit per query and head, Q and K clipped to an L2 norm, V
  RMS-normed, causal 4-tap convolutions over K, V and both residual updates (dynamic per-head taps
  over K and V in the full-attention layers), and a sigmoid gate over the heads. Layers 5, 11, 17
  and 23 attend over every position and scale Q and the relative feature with the log of the
  position; the others attend over a window of 1,024 positions. Each layer reads its inputs through
  a depth attention over the residual blocks of 6 layers (up to 8 states, up to 4 queries), and a
  core layer's second and third passes take messages from the pass before through rank-64
  controllers. Each layer routes a token to 8 of 128 experts (width 1152) by sigmoid score plus a
  selection bias and runs SiTU, a capped gate, over each expert's [gate; up] rows; there is no
  shared expert. The head scores 200,058 of its 201,024 rows, soft-capped at 30.
- **The weights stay the GGUF's blocks.** Every matrix keeps its ggml format, imported row for row
  without requantization: in UD-IQ4_XS the routed experts are NVFP4 in 20 of the 24 layers (Q5_K,
  Q4_K or IQ4_XS [gate; up] banks and Q5_0 or IQ4_NL down banks in the other 4), attention Q8_0
  (Q6_K or Q5_K in 7 layers' projections), the token table Q8_0 and the head Q6_K. In MiniPlus the
  [gate; up] banks are IQ3_XXS or IQ3_S and the down banks Q5_0 or IQ4_NL, attention Q4_K (Q6_K
  output projections in 4 layers, Q8_0 gates), the token table Q4_K and the head Q6_K. The router,
  convolutions, distance tables, mixing queries, controllers, norms and expert scales are FP32;
  MiniPlus stores its sink and dynamic-tap projections in Q8_0, which the converter decodes to FP32
  exactly.
  Vector kernels multiply up to eight tokens, integer matrix kernels the prompt chunks, the expert
  products the routed experts only; a layer's query, gate, relative, key and value products of one
  format share vector launches ([GGUF products](gguf.md#products)). The NVFP4 experts' matrix
  kernel stages its tiles through `cp.async` where the card's
  [device profile](device-profiles.md#what-is-calibrated) takes it: a 20,430-token prompt
  prefilled at 3.07-3.08k instead of 2.96-2.97k tokens/s on an RTX 3090, 4.61-4.62k instead of
  4.31-4.33k on an RTX PRO 4500 Blackwell and 6.81-6.82k instead of 6.43k on an RTX 4090. A GGUF's
  BF16 matrices keep their words: UD-Q8_K_XL stores
  attention, the token table, the head and layers 20-23's expert banks in BF16. A model with any
  BF16 matrix runs every call on BF16 rows: Linear's A16 route multiplies the BF16 matrices into
  FP32, a SIMT kernel the BF16 experts (FP32 sums of each expert's pairs, 32 at a time), and the
  integer matrix kernels the GGUF matrices at every width.
- **The looped state.** A decode or verification call of at most 16 rows spreads each depth mix
  and residual update over one-warp blocks of 128 channels instead of one block per row. Each warp
  keeps its sums, and every reader adds them in the order the one-block kernel adds its warps', so
  the values are the same bits. A controller's nine products run in two launches, and the route
  kernel also groups the pairs by expert for the expert kernels and zeroes the sums the vector
  expert products add into. A layer's query and key/value epilogues share one launch, and SiTU
  writes the down products' q8_1 input itself, the bytes the quantizer would write. These kernels
  took decode on an RTX 3090 from 88.9 to 111.4 tok/s with the same tokens
  ([measurements](#measurements)); the shared launches then added 1.9% without drafts on an RTX
  PRO 4500 Blackwell (126.6 to 129.0 tok/s) and 2.2% on an RTX 4090 (139.4-139.9 to 142.8-142.9
  tok/s), and 1.1-1.4% with DSpark, again with the same tokens.
- **Long contexts.** A prompt chunk of a full-attention visit first computes, on tensor cores, its
  rows' attention over the keys at least 1,024 positions (the relative table's extent) before every
  row of a group of 32: F16 queries and probabilities, keys and values as stored (a plane format's
  chunks decoded to F16), FP32 scores, softmax and sums, two blocks a multiprocessor (one on
  Turing). A plane's next tile waits in registers as its stored bits and is decoded as it enters
  shared memory, so its loads overlap the current tile's products. Every visit's prompt chunk then
  reads the nearer keys on tensor cores too (a full visit's rows starting from that state, a
  sliding-window visit's over its window), 32 rows to a block: each query enters the products as an
  F16 value and its remainder times 2^12, and the relative bias comes from products of the rows'
  relative features and the table rows of the distances a 32-key tile spans, split the same way (a
  sum of FP32 products for each score where a chunk's positions are not consecutive, as with
  branches), with FP32 masks, softmax and sums and F16 probabilities. A full visit's decode and
  verification rows also run on tensor cores: four warps read a split's 16-key tiles in turn, keys
  and values decoded from their stored format straight into the products' operands, and each query
  and probability enters them as an F16 value and its remainder times 2^12 (about 22 significant
  bits), with FP32 scores, softmax and sums. A row's keys split into about two blocks a
  multiprocessor (at most 512, each at least 64 keys), the same split for every row of every call;
  a verification call's rows of one sequence go up to four to a block, two to each set of four
  warps, which read each key tile once for them, and each row reads the tiles of its own split.
  Every product and sum the kernel rounds is an explicit operation, so the compiler cannot fuse a
  block's second row differently from a decode row: a verification row computes the bits a decode
  step computes. Volta, whose tensor-core products are emulated, keeps the SIMT kernels for decode
  rows (about 32 blocks a multiprocessor) and for the near field of prompt chunks. On an RTX 3090 a
  114,754-token prompt prefills at 2.13k tok/s instead of 1.44k with the near field on the SIMT
  kernel and the far field one block a multiprocessor, and after 114,754 tokens decode runs at 86.1
  tok/s instead of 68.6 with the SIMT decode kernel ([measurements](#long-contexts)).
- **Requests.** One to eight concurrent sequences (`--max-concurrency`), prompt chunks of 2048
  tokens, prefix reuse across turns and requests with host-memory images
  ([below](#prefix-reuse)), branched reads for [typed decisions](serving.md#typed-decisions) and
  reranking (the prompt opens the answer's text block, so the token read is the answer's: the
  documented ticket's three questions answer at confidences of 0.996 to 1.0), streaming, stop
  strings, causal scoring (`ninfer-perplexity`), and the model's tool calls: a block whose header
  names the function and whose JSON body (`<|content_invoke_tool_json|>{"name":...,"args":{...}}`)
  becomes an OpenAI tool call. With
  `--structured-output`, a response format and strict tool arguments hold under their grammars: a
  strict call's JSON names a declared function with its arguments under the tool's schema, and
  beside a response format the turn's call blocks come before the text block that holds the
  response, or the turn holds calls and no text block ([serving](serving.md)).
- **Reasoning.** The model writes every block header itself and decides whether it reasons; the
  reasoning block arrives as `reasoning_content`. The chat template states a thinking effort level
  in a system block: `reasoning_effort` `none`, `minimal`, `low`, `medium`, `high` or `xhigh` (0,
  0.1, 0.2, 0.7, 0.9, 0.99; `high` without one). `enable_thinking: false` sets `none`, the
  model's own lowest level, which the release notes do not fully disable reasoning. A thinking
  budget (`--thinking-budget`, `--default-thinking-budget`, a request's) counts the tokens of the
  thinking blocks the model opens, at `none` too (which a forced tool call also sets); once
  reached, the engine closes the open block with Qwen's early-close guidance and `<|end_message|>`,
  masks `<|content_thinking|>` for the rest of the response and decodes it without drafts.
  `--reasoning-loop conclude` closes it the same way ([Serving](serving.md#openai-chat-completions)).
- **Sampling defaults.** The release's: temperature 1.0.
- **Speculative decoding.** The release's DSpark drafter (the artifact's `dspark` component,
  `--spec dspark`) proposes a block of up to 7 drafts from the target's tapped hidden states: three
  non-causal blocks over the anchor and mask rows, then a recurrent Markov head of rank 256 that
  corrects the target head's logits draft by draft. Without `--spec`, an artifact that carries the
  drafter drafts with it while the Engine serves one request at a time: 3 drafts a round, the
  round's width adapting from 1 to 3 by how far the drafts survive and how long the rounds take
  (`--adaptive-mtp`), each proposal ending after its first draft at or below probability 0.5
  (`--draft-min-p`); the request decodes plainly while its timed rounds commit tokens slower than
  plain steps do, trying the other mode every 64 steps ([Adaptive MTP](serving.md#adaptive-mtp)).
  The target keeps a draft only while it equals the token it draws itself, so
  the output is plain decoding's. Context lookup (`--lookup-ngram`, 1 to 7 drafts, 3 by default)
  also drafts, and so does a [DSpine](maintainer/dspine.md) drafter trained for this model
  (`--spec dspine --draft-model PATH`; none is published), which taps the visits' states; with a
  random drafter of this geometry the text equals plain decoding's, greedy and sampled.
- **Images.** The release's vision tower (the artifact's `vision` component, `--vision`): a ViT
  with a class row, learned position rows resized to each image and 2D-rotated heads, then 2 x 2
  patch merging and a two-layer projector, up to 4,096 positions an image (fewer with
  `--vision-max-merged`). Its matrices are BF16 or, from a Q8_0 mmproj, the GGUF's blocks, which
  the tower decodes to BF16 rows before each product; cuBLAS multiplies BF16 inputs into FP32
  results either way. They stay on the GPU or, with `--vision-residency overlay`, in pinned host
  memory through two device slots.
- **Expert residency.** The routed experts stay on the GPU, in pinned host memory or in the
  artifact's files, with a device cache and a CPU share of decode misses
  ([below](#expert-residency)). Several GPUs divide the routed experts on the devices (`--devices
  0,1 --tensor-parallel`, [expert parallelism](maintainer/pipeline-stages.md#expert-parallelism-of-the-other-moe-families)),
  and DSpark drafts there by default too ([measurements](#measurements)).

Not built for this family: MTP, DFlash, n-gram copy drafts and their archive, `--draft-confidence`,
rotation scaling (the attention has no rotation), video, pipeline stages, and of
the context cache the shared paged KV pool; the server refuses their options at startup ([Serving](serving.md#context-cache-of-the-other-families)). Of the
`--kv-dtype` formats, MXFP8 (the default), `int8`, `rk8v4`, `rk4v4` and F16 (`bf16` or `fp16`) are
built ([KV cache](#kv-cache)); `fp8`, `nvfp4`, `k8v4`, `rk4v4-e8`, `rk2v4-e8` and `bf16p` are not,
for the reasons [Flint's KV cache](flint.md#kv-cache) lists. DSpark drafts for one request at a
time; with several requests decoding, each decodes a token a step.

## Converting

The converter reads the weights from the GGUFs and the tokenizer, chat template and generation
config from the Hugging Face release:

```bash
mkdir -p models/inkling2/hf
for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json; do
  curl -fsSL -o models/inkling2/hf/$f https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/$f
done
hf download unsloth/Inkling-2-Mini-GGUF Inkling-2-Mini-UD-IQ4_XS.gguf \
  dspark-Inkling-2-Mini-Q8_0.gguf --local-dir models/inkling2
hf download ggml-org/Inkling-2-Mini-GGUF mmproj-Inkling-2-Mini-Q8_0.gguf --local-dir models/inkling2

python -m tools.convert --model models/inkling2/hf --recipe inkling2_gguf \
  --source gguf=models/inkling2/Inkling-2-Mini-UD-IQ4_XS.gguf \
  --source vision=models/inkling2/mmproj-Inkling-2-Mini-Q8_0.gguf \
  --source dspark=models/inkling2/dspark-Inkling-2-Mini-Q8_0.gguf \
  --components text,vision,dspark --name Inkling-2-Mini-UD-IQ4_XS \
  --out models/Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer --device cpu
```

The conversion writes one 16,494,910,976-byte (15.4 GiB) `.ninfer` file of 1,185 objects: the text
model, the vision tower (Q8_0 matrices) and the DSpark drafter (Q8_0). With Unsloth's
`mmproj-BF16.gguf` as the `vision` source the tower keeps BF16 matrices and the file has
16,790,806,016 bytes (15.6 GiB). Unsloth's repository holds each quantization in two layouts, the
root file's `inkling2` (48 unrolled blocks in its metadata) and llama.cpp's `inkling2_loop` (24
blocks, `Mainline_Support/*.gguf_file`). Both convert to the same artifact: from UD-IQ4_XS the two
payloads are equal byte for byte, and the headers differ only in the recorded source path and the
random artifact id.

IsValorum's APEX-I MiniPlus V2.1 converts the same way with its own Q8_0 mmproj and Unsloth's
drafter, which its repository does not carry:

```bash
hf download IsValorum/Inkling-2-Mini-APEX-I-MiniPlus-V2.1-GGUF \
  Inkling-2-Mini-APEX-I-MiniPlus-V2.1.gguf mmproj-Inkling-2-Mini-Q8_0.gguf \
  --local-dir models/inkling2/apex

python -m tools.convert --model models/inkling2/hf --recipe inkling2_gguf \
  --source gguf=models/inkling2/apex/Inkling-2-Mini-APEX-I-MiniPlus-V2.1.gguf \
  --source vision=models/inkling2/apex/mmproj-Inkling-2-Mini-Q8_0.gguf \
  --source dspark=models/inkling2/dspark-Inkling-2-Mini-Q8_0.gguf \
  --components text,vision,dspark --name Inkling-2-Mini-APEX-I-MiniPlus-V2.1 \
  --out models/Inkling-2-Mini-APEX-I-MiniPlus-V2.1-ninfer-v3.ninfer --device cpu
```

It writes 13,740,465,920 bytes (12.8 GiB) of 1,173 objects. Each conversion took about two minutes
on a rented A40's container.

Unsloth's UD-IQ1_M, UD-Q2_K_XL and UD-Q3_K_XL convert like UD-IQ4_XS (`--source gguf=` names the
file) to 8,785,600,768, 10,080,361,216 and 14,392,937,728 bytes, in about 30 seconds each on a
rented RTX 3090 host. UD-Q2_K_XL stores layer 20's down bank in MXFP4, which the expert products
do not take: the converter writes the same values as Q8_0 blocks, each code's doubled value as
the int8 code and 2^(e - 128) as the F16 scale, checking every block against the MXFP4 decode.
UD-Q8_K_XL converts the same way to 23,685,230,592 bytes in 55 seconds on a rented RTX A6000 host;
its BF16 matrices, expert banks included, are copied word for word.

## Running

```bash
# A chat server over 64K-token contexts; one request at a time drafts with DSpark.
ninfer-serve models/Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer --model-id inkling-2-mini \
  --max-context 65536

# Four concurrent requests with images.
ninfer-serve models/Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer --model-id inkling-2-mini \
  --max-context 32768 --max-concurrency 4 --vision

# One prompt from the CLI.
ninfer models/Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer --prompt "Write a binary search in Rust."

# Experts in pinned host memory, for a card with less memory than the model needs.
ninfer-serve models/Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer --model-id inkling-2-mini \
  --max-context 32768 --expert-residency host
```

With its experts on the GPU the model keeps 14.6 GiB of weights (15.0 GiB with the DSpark drafter)
and 1.2 GiB of workspace, and a sequence's F16 keys and values take 32 KiB per token of context
(its 8 full-attention visits) beside 224 MiB of sliding-window rings and convolution histories. One
sequence of 12,288 tokens plans 16.5 GiB of the GPU, 17.0 GiB with the drafter. With
`--expert-residency host` the GPU keeps 1.6 GiB of weights (1.95 GiB with the drafter), two
whole-layer banks of 1,337 MiB and a cache of as many experts as fit, and the host pins 13.1 GiB
for the routed experts; `disk` reads them from the artifact through the OS page cache.

## Prefix reuse

With the context cache on (the default), a prompt resumes from a state a sequence or the host store
keeps instead of prefilling those tokens again, as for [Flint](flint.md#prefix-reuse): a sequence
keeps its state when its request ends and a checkpoint of its sliding-window rings where the
prompt's last user turn closes; what a sequence holds before another request takes it, and its
state at the end of the prompt's shared prefixes (the leading system messages and tool
declarations, a declared cache boundary), go to images in pinned host memory within
`--host-cache-mib`. Every visit's state belongs to the image: its rings, its full-attention rows and
its convolution histories. A prompt resumes only where a state was kept, not at an arbitrary common
prefix, except on its sequence: a next prompt that departs from what the sequence read after its
last prompt (a reply the template renders otherwise) resumes where it departs, the sequence taken
back there, while its rings and convolution histories still hold that position. A long last message also keeps a state inside it, where a later prompt with the same
document and another question resumes, and the images go to a disk tier, a prefix cache file and
session slot files ([Serving](serving.md#context-cache-of-the-other-families)).

## KV cache

Inkling 2 stores the keys and values of its eight full-attention visits, the part of its cache that
grows with the context, in MXFP8 unless `--kv-dtype` names another format: E4M3 values over one
UE8M0 power-of-two scale per 32 values, the scale the group's largest magnitude over 448 rounded up
to a power of two. This is the KV format the release trained with
([#394](https://github.com/iamwavecut/ninfer-all/issues/394)) and the rule vLLM's Inkling 2 kernels
quantize MXFP8 with; the stored codes and scales equal that rule bit for bit
(`ninfer_inkling2_kv_planes_test`). A token's full-attention K and V take 16.5 KiB instead of F16's
32 KiB. `int8` (int8 codes over one FP16 scale per 64 values, keys and values), `rk8v4` (the same
keys, values as packed int4 over one scale per 32) and `rk4v4` (4-bit Lloyd-Max keys over the same
values) store the same visits unrotated: the K and V come out of their convolutions clipped and
RMS-normed, and rotating the keys as [Flint's](flint.md#kv-cache) `rk4v4` does raised the mean NLL
over 67,977 tokens (`rk8v4` 1.784842 against 1.784735, `rk4v4` 1.785690 against 1.785611). Whatever
the format, the windowed visits' rings of 1,024 positions, the convolution histories and a DSpark
drafter's caches stay F16: rounding the rings to MXFP8 as well raised the mean NLL (+0.029% over
67,977 tokens and +0.091% over 9,718, against +0.024% and +0.015% for the full visits alone) and
saves no memory, since a ring does not grow with the context.

| `--kv-dtype` | A token | KV of one sequence at 4,096 / 40,960 tokens | Mean NLL, 9,718 tokens | Mean NLL, 67,977 tokens | 30K prompt: prefill, then decode |
|---|---:|---:|---:|---:|---:|
| `bf16` or `fp16` (F16) | 32 KiB | 352 / 1,708 MiB | 1.858657 | 1.784503 | 1.28k, 83.1 tok/s |
| `mxfp8` (default) | 16.5 KiB | 286 / 1,084 MiB | 1.858934 (+0.015%) | 1.784935 (+0.024%) | 1.25k, 80.4 tok/s |
| `int8` | 16.5 KiB | 286 / 1,084 MiB | 1.858933 (+0.015%) | 1.784764 (+0.015%) | 1.26k, 80.4 tok/s |
| `rk8v4` | 12.75 KiB | 270 / 933 MiB | 1.858810 (+0.008%) | 1.784735 (+0.013%) | 1.25k, 80.2 tok/s |
| `rk4v4` | 8.75 KiB | 253 / 772 MiB | 1.859427 (+0.041%) | 1.785611 (+0.062%) | 1.25k, 79.9 tok/s |

UD-IQ4_XS on an RTX 3090 (350 W, PCIe 4.0 x16), a CUDA 13.4.3 build for `75;80;86;89;120a`,
November 22, 2026. The NLL columns score the first 40,000 bytes and all 277,332 bytes of the
WikiText file `00.txt` in 4,096-token windows at a 2,048-token stride; this build's F16 value on the
40,000 bytes differs from the 1.859871 a CUDA 12.8 build measured ([below](#smaller-quantizations)),
so each format is compared with the F16 run of the same build. Speed: one server per format with
`--spec none`, one run each; a 5K prompt prefilled at 2.10-2.11k tok/s and then decoded 100.2-101.4
tok/s in all five. On this text `rk8v4` raised the NLL less than MXFP8 in a smaller cache, and
`int8` no more than MXFP8 in the same 16.5 KiB; MXFP8 stays the default because the release
trained with it.

## Measurements

`ninfer_bench` against `llama-bench` from Unsloth's llama.cpp prebuilt
[`b12588-mix-c0c3de8`](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8)
(upstream b12588 with Unsloth's Inkling 2 changes; mainline llama.cpp b12602 does not load the
GGUF, whose looped layers use Unsloth's layout) on the same UD-IQ4_XS GGUF (flash attention on, F16
KV), one sequence, three repetitions each, on an RTX 3090 (350 W, CUDA 12.8, build `86`, experts on
the GPU), November 21, 2026. NInfer-all's prompts are Inkling-tokenized text,
llama-bench's its own tokens. Decode after 10,000 tokens follows a 10,000-token prompt
(`-pg 10000,128` for NInfer-all, `tg128 @ d10000` for llama.cpp).

| Engine | Prefill 2,048 | Prefill 10,000 | Decode 128 | Decode after 10,000 |
|---|---:|---:|---:|---:|
| NInfer-all | 2,659 | 2,031 | 111.4 | 95.9 |
| llama.cpp | 1,301 | 1,228 | 58.6 | 55.9 |

NInfer-all against llama.cpp: 2.04, 1.65, 1.90 and 1.72 times. NInfer-all's spreads stay under 0.2%,
llama.cpp's under 3.9% (its prefill of 2,048 tokens ±50.6 tok/s). Before the split depth mixes,
the batched controllers and the sorted route ([what runs](#what-runs)) NInfer-all decoded 88.9 and
79.0 tok/s and prefilled 2,663 and 2,030; the prefill rates moved by -0.15% and +0.06%.

- **DSpark.** The CLI on the same card, 384 tokens a request, reasoning first, without `--spec`
  (DSpark, 3 drafts adapting, floor 0.5) against `--spec none`, one run each. Greedy: an essay
  prompt 125.3 against 109.7 tok/s (1.14x), a Python task 154.5 against 109.9 (1.41x), a question
  about a 5,748-token document 126.0 against 99.7 (1.26x). At the release's temperature 1.0: the
  essay 108.8 and 115.4 with two seeds against 109.7 (0.99x and 1.05x), the Python task 136.5
  against 109.9 (1.24x). The text equals plain decoding's in every run that has a plain-decoding
  twin (greedy, and seed 1). A round accepts 1.46-1.90 tokens on the essay, 2.34-2.68 on the
  Python task and 2.36 on the document. llama.cpp's server with the same drafter
  (`--spec-type draft-dspark`, 7 drafts) decoded the greedy essay at 37.0 tok/s against its 57.3
  without it, 133 of 850 drafts accepted.
- **Fixed widths.** With the slower decode kernels before the split depth mixes, 3 drafts without
  a floor decoded sampled essays 0.87-0.95 times as fast as plain decoding and 1 draft 1.06-1.12
  times; 7 drafts with floor 0.6 decoded the greedy essay at 94.9 tok/s against 106.3 with 3 at
  the same floor. The floor 0.5 beside the adaptive width added 0.9-5.3% on five of six requests
  and cost 0.6% on the sixth.
- **Context lookup.** `--lookup-ngram 3` decoded the greedy essay at 111.6 tok/s and the Python
  task at 117.9, against 109.7 and 109.9 without drafts, with the same text.
- **Concurrency.** `ninfer-serve --max-concurrency 4`, four greedy chat requests of up to 256 tokens
  at once: 970 tokens in 4.9 s, 198.1 tok/s together (53.4-56.5 each); the first request's text
  equals its text alone, which decoded at 111.0 tok/s.
- **Expert parallelism.** Two RTX 3090s (350 W, CUDA 12.8, build `86`) without peer access, so
  the exchange stages through pinned host memory; `--devices 0,1 --tensor-parallel` gives each
  card 64 experts by measured bandwidth (0-63 and 64-127). One request at a time through
  `ninfer-serve`, 384 tokens, two passes alternating with one card. Without drafts the split
  decodes 89.4-91.6 tok/s through the server; in the CLI (`--spec none`) it decodes 86.7-91.6
  against one card's 110.2-111.8 on the same prompts, 0.78-0.82 times as fast, since every MoE
  block crosses the bus twice (86.7 with 32 experts on the first card and 96 on the second). DSpark with the automatic settings
  decodes the greedy essay at 120.0 and 120.7 tok/s against 90.4 and 91.1 (1.33x), the Python
  task at 154.6 and 155.2 against 90.7 and 91.6 (1.70x) and the essay at seed 7 at 105.3 and
  105.9 against 89.4 and 90.2 (1.18x), with plain decoding's text; one card with DSpark decodes
  them at 125.2-125.5, 144.1-144.9 and 114.4-115.0. Four greedy requests of 256 tokens at once
  (`--max-concurrency 4`, no drafts) decode 238.5 tok/s together on the split against 209.4 on
  one card (1.14x), each request's text equal to its text alone. With one request and an
  8,192-token context the server's devices use about 11,100 and 7,600 MiB against one card's
  18,300 MiB. Prompt rates over the split were not measured beyond a 33-token prompt.

### Long contexts

`ninfer-serve --max-context 131072` on an RTX 3090 (350 W, PCIe 4.0 x16, CUDA 13.4.3, build
`75;80;86;89;120a`, MXFP8 KV, experts on the GPU), November 22, 2026: one greedy request of 256
tokens after each of three fixed prompts, one run each, with the [long-context
kernels](#what-runs) and the build before them. The server's own prefill and decode rates:

| Prompt tokens | Prefill, before | Prefill | `--spec none` decode, before | `--spec none` decode | DSpark decode, before | DSpark decode |
|---:|---:|---:|---:|---:|---:|---:|
| 7,968 | 1.91k | 2.19k | 96.7 | 100.4 | 101.3 | 111.0 |
| 32,008 | 1.20k | 2.01k | 78.6 | 90.3 | 111.9 | 117.4 |
| 120,103 | 501 | 1.52k | 46.8 | 67.1 | 62.8 | 57.4 |

DSpark's text equals `--spec none`'s in both builds at every length; the text itself changed with
the far field's F16 probabilities and the new splits. DSpark's rate follows the drafts the text
accepts and the widths its rounds adapt to: after 120,103 tokens a round took 27.3 ms and kept 1.6
tokens, so plain decoding at 14.9 ms a token stays faster there; after 32,008 tokens a round took
21.1 ms and kept 2.5 tokens (plain decoding 11.1 ms a token), after 7,968 tokens 16.9 ms and 1.9
(10.0 ms). With 128 splits, grouping the verification rows took a round after 120,103 tokens from
30.1 to 28.8 ms (1.4 and 1.5 drafts a round). The prompt kernels' mean NLL over the first 200,000
bytes of WikiText `00.txt` (49,478 tokens, 16,384-token windows at an 8,192-token stride) is
1.739363 against 1.739591 before (-0.013%), and `ninfer-perplexity` scored them in 102 s against
125 s.

The DSpark columns above decode without the speculation gate. With it ([Adaptive
MTP](serving.md#adaptive-mtp)), on another RTX 3090 (390 W, PCIe 4.0 x16, CUDA 13.4.3, the same
build targets) the same three requests decode, in tok/s:

| Prompt tokens | `--spec none` | DSpark without the gate (mean of 3 runs) | DSpark with the gate |
|---:|---:|---:|---:|
| 7,968 | 102.6 | 115.0 | 114.5 |
| 32,008 | 92.5 | 121.8 | 120.4 |
| 120,103 | 69.6 | 59.8 | 70.4 |

After 120,103 tokens the request decodes plainly most of the time: 70.4 against 59.8 tok/s
(1.18x), 1.2% above `--spec none`. After 7,968 and 32,008 tokens it keeps drafting, and its probes
of plain decoding cost 0.5% and 1.1%; the ungated runs spread by 0.5, 1.0 and 0.7 tok/s at the
three lengths. The text equals `--spec none`'s in every run.

The decode and verification attention on tensor cores, November 23, 2026: `ninfer-serve
--max-context 131072`, MXFP8 KV, one greedy request of 256 tokens after each of three fixed
prompts, one run each, with DSpark gated as above; the build before it (release 2026-11-22.6 on
the RTX 3090, master 3f82067 built alike on the RTX 5090) against the build with it. The server's
own decode rates, in tok/s:

| Card | Prompt tokens | `--spec none`, before | `--spec none` | DSpark, before | DSpark |
|---|---:|---:|---:|---:|---:|
| RTX 3090 (370 W, PCIe 4.0 x16) | 35 | 105.9 | 110.8 | 105.5 | 109.5 |
| | 32,070 | 90.8 | 99.9 | 100.6 | 121.7 |
| | 114,754 | 68.6 | 86.1 | 71.3 | 97.7 |
| RTX 5090 (600 W, PCIe 3.0 x16) | 35 | 168.1 | 179.7 | 186.4 | 192.9 |
| | 32,070 | 152.9 | 164.1 | 205.1 | 243.1 |
| | 114,754 | 129.7 | 144.3 | 165.6 | 190.4 |

Prompt chunks keep their kernels, and prefill stayed within 1% (after 114,754 tokens 1.31k against
1.32k tok/s on the RTX 3090, whose host also ran other tenants' jobs at a load average of about
50). DSpark's text equals `--spec none`'s in both builds after every prompt on both cards, and so
does the CLI's after 32,070 tokens with each KV format (`mxfp8`, `bf16`, `int8`, `rk8v4`, `rk4v4`)
on the RTX 5090; the text itself changed with the kernel's rounding. With `--spec none` after
32,070 tokens the CLI on the RTX 3090 decoded at 99.7 tok/s against 90.6 with `int8`, 101.6
against 90.7 with `rk8v4`, 101.5 against 90.5 with `rk4v4` and 95.6 against 90.1 with `bf16`.

The kernel alone, in µs, against the SIMT kernel at the split each uses: random caches at the
model's 8 KV heads and 24 query heads, 114,754 keys (the F16 window ring: 1,024 keys, 16 splits):

| GPU | Splits | MXFP8 decode | MXFP8, four verification rows | F16 decode | F16 window ring |
|---|---|---:|---:|---:|---:|
| Quadro RTX 6000 | 288 → 18 | 1,083.5 → 572.6 | 3,054 → 1,326 | 1,003 → 916 | 25.1 → 20.3 |
| RTX 3090 | 328 → 20 | 625 → 294 | 1,861 → 689 | 605 → 545 | 21.9 → 12.0 |
| RTX 4090 | 512 → 32 | 324.8 → 271.0 | 873 → 332 | 556.5 → 507.7 | 15.5 → 9.4 |
| RTX 5090 | 512 → 42 | 207.0 → 158.1 | 458 → 236 | 329.7 → 290.5 | 14.4 → 10.3 |

These are the kernel's first form, whose products and sums the compiler could fuse; with every
rounding explicit it took 293.4, 689.4, 544.9 and 12.6 µs on the RTX 3090 and 158.5, 240.0, 289.8
and 10.2 µs on the RTX 5090. Against an FP64 oracle both kernels' outputs have the same error (relative RMS 1.705e-3 with
MXFP8 after 114,754 keys, mostly the BF16 output's rounding). On a Tesla V100, whose tensor-core
products are emulated, the tensor-core kernel took 1,914 µs against the SIMT kernel's 1,235
(MXFP8) and 1,624 against 841 (F16), so Volta keeps the SIMT kernel.

The near field of prompt chunks on tensor cores, November 23, 2026: the CLI (`--greedy
--max-context 131072 --max-new 4 --spec none`, MXFP8 KV, experts on the GPU, CUDA 13.4, build
`75;80;86;89;120a`) after two fixed prompts, master db39615 against the same tree with the kernel,
the two builds alternating. The CLI's prefill rate, in tok/s:

| Card | Prompt tokens | Prefill, before | Prefill |
|---|---:|---:|---:|
| RTX 3090 (350 W, PCIe 4.0 x16) | 32,070 | 1.98k, 1.96k | 2.76k, 2.75k |
| | 114,754 | 1.44k | 1.83k |
| RTX PRO 4500 Blackwell (200 W, PCIe 4.0 x16) | 32,070 | 3.17k, 3.16k | 3.97k, 3.95k |
| | 114,754 | 2.43k | 2.88k |

Both builds give the same first tokens after both prompts on both cards. On the RTX 3090 the
114,754-token prefill spends 62.8 s in GPU kernels: the far field 29.2 s (46.5%), the near field
6.0 s (9.6%). The mean NLL over the first 200,000 bytes of WikiText `00.txt` (49,478 tokens,
16,384-token windows at an 8,192-token stride) is 1.739533 against 1.739363 on the RTX 3090
(+0.010%) and 1.739307 against 1.739413 on the RTX PRO 4500 (-0.006%); `ninfer-perplexity` scored
them in 68.1 s against 84.5 s and in 46.8 s against 54.3 s.

The far field at two blocks a multiprocessor with a plane's next tile held as its stored bits,
November 23, 2026: the same CLI runs, master 4e1561d against the same tree with both changes, the
two builds alternating. The CLI's prefill rate, in tok/s:

| Card | Prompt tokens | Prefill, before | Prefill |
|---|---:|---:|---:|
| RTX 3090 (350 W, PCIe 4.0 x16) | 32,070 | 2.73k, 2.73k | 2.87k, 2.87k |
| | 114,754 | 1.83k | 2.13k |
| RTX PRO 4500 Blackwell (200 W, PCIe 4.0 x16) | 32,070 | 3.98k, 3.95k | 4.15k, 4.14k |
| | 114,754 | 2.88k | 3.31k |

Both builds give the same first tokens, and the mean NLL above is 1.739533 with both on the RTX
3090: the changes move no bits. On the RTX 3090 the far field of the 114,754-token prefill took
29.2 s of GPU time before, 24.9 s with the staged bits alone and 19.9 s with both (36.9% of 53.8
s); with F16 keys and values (`--kv-dtype bf16`) it takes 16.7 s at one block a multiprocessor and
15.1 s at two. With the staged bits alone the CLI prefilled at 2.78k tok/s (32,070 tokens) and
1.95k (114,754) on the RTX 3090, 4.06k and 3.06k on the RTX PRO 4500. F16 products for MXFP8's
codes times their power-of-two scale, exact against the FP32 path for every pair of codes and every
exponent, moved the far field from 19.9 to 19.2 s on the RTX 3090 and the RTX PRO 4500's prefill
from 4.18k, 4.16k to 4.16k, 4.14k and from 3.34k to 3.29k, so the decode keeps its FP32 products.

### APEX-I MiniPlus

IsValorum's APEX-I MiniPlus V2.1 (12.8 GiB with its Q8_0 tower and the DSpark drafter) on a rented
A40 (300 W, PCIe 4.0 x16, CUDA 12.8, build `86`), November 21, 2026:

- **Speed.** `ninfer_bench`, three repetitions: prefill of 2,048 tokens 2,776 tok/s, of 10,000
  tokens 2,128; decode of 128 tokens 104.3 tok/s, 90.2 after 10,000. The CLI's 384-token greedy
  essay, one run each: 103.1 tok/s with `--spec none`, 115.6 with DSpark (1.69 tokens a round),
  with the same text. UD-IQ4_XS on another A40 host ([below](#across-cards)) gave 2,432, 1,897,
  85.3 and 75.5, and 84.4 and 97.3 on the essay; this was not an A/B on one host.
- **Quality.** Perplexity on the first 40,000 bytes of WikiText (`--kv-dtype bf16`, 9,718 scored
  tokens): a mean NLL of 1.870942 (perplexity 6.494411) against UD-IQ4_XS's 1.859871 (6.422907),
  +0.60%.
- **Memory.** With every expert on the GPU: 12.1 GiB of weights, 12.8 GiB with the tower and the
  drafter. One sequence plans 13.8 GiB of the GPU at 8,192 tokens and 14.5 GiB at 32,768 without
  the drafter, 14.7 and 15.5 GiB with the tower and the drafter; no 16 GB card has run it.
  `--expert-residency host` pins 10.9 GiB of host memory for the routed experts and keeps 1.16 GiB
  of weights and two 1,002 MiB layer banks on the GPU; `host` and `disk` (the A40's cache held
  every expert) gave the device run's 384 greedy tokens byte for byte.
- **Images.** The image prompt of examples/cli/messages/image_natural.json (greedy) names the
  mailbox's 24 and the sun on the right, as UD-IQ4_XS does.

### Smaller quantizations

Unsloth's UD-IQ1_M (8.2 GiB), UD-Q2_K_XL (9.4 GiB) and UD-Q3_K_XL (13.4 GiB), each with the Q8_0
tower and the DSpark drafter, on a rented RTX 3090 (350 W, PCIe 4.0 x16, CUDA 12.8), November 21,
2026, one run each:

| Quantization | Unsloth's KLD to UD-Q8_K_XL | Mean NLL, 40 KB WikiText | Essay, `--spec none` | Essay, DSpark | GPU plan: 8K, `--spec none` / 32K, tower and DSpark |
|---|---:|---:|---:|---:|---:|
| UD-IQ1_M | 0.0867 | 1.929600 (+3.75%) | 124.3 tok/s | 128.9 (1.67 a round) | 9.2 / 10.9 GiB |
| UD-Q2_K_XL | 0.0595 | 1.901978 (+2.26%) | 121.3 | 130.3 (1.92) | 10.4 / 12.1 GiB |
| UD-Q3_K_XL | 0.0174 | 1.861615 (+0.09%) | 113.5 | 126.1 (2.03) | 14.4 / 16.1 GiB |

The NLL differences are against UD-IQ4_XS's 1.859871 (`--kv-dtype bf16`, 9,718 scored tokens).
With `--expert-residency host` (6.3, 7.5 and 11.4 GiB pinned) or `disk` each gave the device
run's 384 greedy tokens, reasoning included, byte for byte, as did DSpark; the 3090's cache held
every expert. Each names the mailbox's 24 and the sun on the right in the image prompt of
examples/cli/messages/image_natural.json. No card under 24 GB has run them.

### UD-Q8_K_XL

Unsloth's UD-Q8_K_XL (22.1 GiB), the reference of Unsloth's KL measurements, keeps BF16 attention,
token table, head and layers 20-23's experts, and NVFP4 experts in layers 0-19. On a rented RTX
A6000 (300 W, PCIe 4.0 x16, CUDA 12.8), November 21, 2026, one run each:

- **Memory.** 21.3 GiB of weights, 22.0 GiB with the tower and the drafter; one sequence plans
  22.9 GiB of the GPU at 8,192 tokens with `--spec none` and 24.7 GiB at 32,768 with the tower and
  the drafter. `--expert-residency host` pins 18.3 GiB of host memory and keeps 3.07 GiB of weights
  and two 1,944 MiB layer banks on the GPU.
- **Speed.** The 384-token greedy essay decodes at 67.3 tok/s with `--spec none`, 72.9 with DSpark
  (1.76 tokens a round), 63.8 with host experts and 63.5 with disk experts (the A6000's cache held
  every expert). A 9,742-token prompt prefills at 1,710 tok/s, then decodes at 61.6 tok/s.
- **Quality.** A mean NLL of 1.857855 on the 40 KB WikiText slice (`--kv-dtype bf16`, 9,718
  scored tokens), against UD-IQ4_XS's 1.859871 (-0.11%). Host, disk and DSpark runs give the
  device run's 384 greedy tokens byte for byte; the image prompt gives the mailbox's 24 and the sun
  on the right.

### Across cards

The release packages of master on cards rented from RunPod and Vast.ai, November 21, 2026:
`e64a7089`, and `b0b49ba6` (the same Linux code with an MSVC fix) on the RTX 3090, the A100 80GB
PCIe, the RTX 4000 Ada, the RTX A4500, both Server Editions and the Vast.ai cards; `cuda12.9` before
Blackwell and `cuda13.4` on it. They ran the benchmarks above and the CLI's 384-token greedy essay
with DSpark (the default) and with `--spec none`, one run each. Every card ran at its default power
limit. The cards of 20 GB and more and the Tesla V100-SXM2-32GB kept the experts on the GPU, on x16
links: PCIe 5.0 with pinned host-to-device copies of 56.3 and 56.9 GB/s for the two Server Editions,
PCIe 3.0 with 12.4 GB/s for the V100, PCIe 4.0 with 23.4 to 28.6 GB/s for the rest (the other
Blackwell cards, PCIe 5.0 devices, linked at 4.0 on these hosts). The RTX A4500 had no built-in
device profile; it ran with the one `ninfer-calibrate` measured on it first, which the build now
carries. llama-bench ran with one thread per CPU of the container's quota where the host set one:
with its default of one thread per host CPU, 120 threads on the RTX 5090 host's 17 CPUs prefilled
2,048 tokens at 1,677 tok/s and 128 threads on the A100 host's 13.6 CPUs at 966, against 2,446 and
1,249 with the quota's count.

NInfer-all / llama.cpp, tok/s:

| Card | Power limit | Prefill 2,048 | Prefill 10,000 | Decode 128 | Decode after 10,000 |
|---|---:|---:|---:|---:|---:|
| RTX PRO 6000 Blackwell Workstation Edition | 600 W | 8,965 / 2,687 | 7,235 / 2,443 | 178.4 / 97.2 | 160.2 / 93.8 |
| RTX 5090 | 575 W | 8,367 / 2,446 | 6,664 / 2,318 | 182.1 / 96.9 | 163.6 / 90.1 |
| RTX PRO 6000 Blackwell Server Edition | 600 W | 7,848 / 2,312 | 6,434 / 2,176 | 158.0 / 84.4 | 143.5 / 81.0 |
| RTX 4090 | 450 W | 5,847 / 2,665 | 4,537 / 2,307 | 140.1 / 75.0 | 125.3 / 71.0 |
| L40S | 350 W | 5,053 / 2,509 | 3,985 / 2,201 | 113.8 / 65.2 | 102.8 / 61.3 |
| RTX 6000 Ada | 300 W | 4,722 / 2,428 | 3,640 / 2,278 | 134.6 / 72.0 | 120.2 / 68.0 |
| RTX PRO 4500 Blackwell | 200 W | 3,872 / 1,939 | 3,067 / 1,681 | 125.5 / 78.4 | 112.1 / 74.1 |
| RTX PRO 4500 Blackwell Server Edition | 165 W | 3,228 / 1,742 | 2,568 / 1,512 | 112.4 / 71.1 | 100.6 / 67.3 |
| RTX 3090 | 350 W | 2,656 / 1,402 | 2,046 / 1,236 | 112.0 / 59.5 | 96.7 / 56.1 |
| RTX A6000 | 300 W | 2,577 / 1,307 | 1,980 / 1,151 | 99.1 / 55.9 | 86.5 / 52.4 |
| A40 | 300 W | 2,432 / 1,224 | 1,897 / 1,051 | 85.3 / 50.8 | 75.5 / 47.4 |
| A100-SXM4-80GB | 400 W | 2,388 / 1,249 | 1,925 / 1,136 | 107.1 / 47.6 | 91.5 / 46.1 |
| A100 80GB PCIe | 300 W | 2,376 / 1,230 | 1,916 / 1,119 | 106.1 / 47.0 | 90.7 / 45.3 |
| Tesla V100-SXM2-32GB | 300 W | 2,124 / 566 | 1,589 / 471 | 72.8 / 35.3 | 64.4 / 34.2 |
| RTX A5000 | 230 W | 2,001 / 1,061 | 1,509 / 981 | 94.3 / 54.6 | 80.7 / 51.0 |
| RTX 4000 Ada | 130 W | 1,907 / 1,139 | 1,450 / 924 | 64.7 / 45.7 | 57.4 / 42.2 |
| RTX A4500 | 200 W | 1,683 / 963 | 1,280 / 843 | 81.5 / 49.5 | 70.1 / 46.4 |

NInfer-all prefills 1.68 to 3.75 times as fast as llama.cpp at 2,048 tokens and 1.52 to 3.38 times
at 10,000, and decodes 1.42 to 2.26 times as fast, 1.36 to 2.00 times after 10,000 tokens. The
prefill gap is widest on the Tesla V100 (3.75 times at 2,048 tokens), then on the RTX 5090 and both
PRO 6000s (3.34 to 3.42 times), where llama.cpp prefills at about the RTX 4090's rate; on the other
cards it is 1.68 to 2.19 times. The decode gap is narrowest on the RTX 4000 Ada (1.42 times). Both
A100s prefill no faster than the A40 in either engine (NInfer-all 2,376 and 2,388 against 2,432
tok/s), while their decode follows their memory bandwidth (106.1 and 107.1 against 85.3).

The essay, tok/s; on every card DSpark's text equals plain decoding's:

| Card | Plain | DSpark | DSpark against plain | Tokens a round |
|---|---:|---:|---:|---:|
| RTX PRO 6000 Blackwell Workstation Edition | 176.0 | 231.3 | 1.31x | 2.08 |
| RTX 5090 | 179.4 | 213.3 | 1.19x | 1.71 |
| RTX PRO 6000 Blackwell Server Edition | 155.8 | 207.0 | 1.33x | 2.08 |
| RTX 4090 | 138.4 | 159.2 | 1.15x | 1.72 |
| L40S | 112.8 | 129.5 | 1.15x | 1.89 |
| RTX 6000 Ada | 132.9 | 151.6 | 1.14x | 1.89 |
| RTX PRO 4500 Blackwell | 124.2 | 132.9 | 1.07x | 1.65 |
| RTX PRO 4500 Blackwell Server Edition | 111.2 | 118.2 | 1.06x | 1.68 |
| RTX 3090 | 110.6 | 125.7 | 1.14x | 1.91 |
| RTX A6000 | 97.6 | 110.1 | 1.13x | 1.70 |
| A40 | 84.4 | 97.3 | 1.15x | 1.70 |
| A100-SXM4-80GB | 105.7 | 123.1 | 1.16x | 1.74 |
| A100 80GB PCIe | 105.0 | 122.1 | 1.16x | 1.73 |
| Tesla V100-SXM2-32GB | 72.1 | 82.6 | 1.15x | 1.70 |
| RTX A5000 | 92.8 | 101.0 | 1.09x | 1.73 |
| RTX 4000 Ada | 64.0 | 69.4 | 1.08x | 1.67 |
| RTX A4500 | 81.3 | 88.7 | 1.09x | 2.10 |

Five cards (A40, A100-SXM4-80GB, L40S, RTX PRO 4500 Blackwell and RTX 3090) ran twice, the first
four on another host of the same type with the quota's thread count, the RTX 3090 against the source
build above. NInfer-all's rates moved by at most 2.9% (the PRO 4500's prefill of 2,048 tokens),
under 0.9% on the 3090, and the tables hold the second runs. llama.cpp on the 3090, with its quota's
31 threads, prefilled 2,048 tokens at 1,402 tok/s against 1,301 in the table above.

#### Experts in host memory

The model's 15.0 GiB of weights do not fit on the cards of 8, 12 and 16 GB. On these NInfer-all kept
the routed experts in pinned host memory (`--expert-residency host`) beside the device cache it
sizes itself, and llama-bench ran with `--fit-target 1024 --fit-ctx 10240`: llama.cpp fits the model
to the card itself, with llama-server's default margin of 1024 MiB, and keeps on the CPU what does
not fit. The cards ran on Vast.ai hosts at their default power limits on PCIe 4.0: x16 links with
pinned host-to-device copies of 24.5 to 28.9 GB/s, the x8 links of the RTX 4060 Ti, RTX 5060 Ti and
RTX 5060 at 13.4, 14.4 and 14.5 GB/s. CPU threads compute part of both engines' expert work;
llama-bench ran one thread per CPU of the container's quota.

NInfer-all / llama.cpp, tok/s:

| Card | Power limit | Host CPU, llama.cpp threads | Prefill 2,048 | Prefill 10,000 | Decode 128 | Decode after 10,000 |
|---|---:|---|---:|---:|---:|---:|
| RTX 5080 | 360 W | Ryzen 9 7900, 23 | 4,044 / 1,607 | 3,238 / 1,377 | 116.5 / 41.3 | 104.1 / 48.2 |
| RTX 4080 Super | 320 W | Ryzen 9 5950X, 30 | 3,788 / 1,708 | 2,953 / 1,379 | 104.4 / 59.2 | 92.1 / 57.5 |
| RTX 5070 | 250 W | Core i5-11600KF, 11 | 2,542 / 778 | 2,048 / 724 | 83.5 / 35.6 | 78.9 / 34.9 |
| RTX 5060 Ti 16 GB | 180 W | Ryzen 5 7500F, 11 | 1,871 / 892 | 1,502 / 793 | 72.1 / 45.6 | 64.4 / 43.8 |
| RTX 4060 Ti 16 GB | 165 W | Ryzen 9 7900X, 11 | 1,689 / 918 | 1,308 / 719 | 53.8 / 39.4 | 47.2 / 31.0 |
| RTX A4000 | 140 W | EPYC 7B13, 30 | 1,363 / 629 | 1,002 / 486 | 57.2 / 37.7 | 49.1 / 34.7 |
| RTX 3060 12 GB | 170 W | Threadripper PRO 3945WX, 5 | 1,002 / 430 | 759 / 387 | 48.8 / 24.8 | 42.3 / 24.0 |
| RTX 5060 8 GB | 145 W | Ryzen 9 5950X, 30 | 1,267 / 305 | 1,074 / 293 | 33.1 / 14.7 | 28.8 / 14.4 |

NInfer-all prefills 1.84 to 4.15 times as fast as llama.cpp at 2,048 tokens and 1.82 to 3.66 times
at 10,000, and decodes 1.37 to 2.82 times as fast, 1.41 to 2.26 times after 10,000 tokens; the
widest gaps are on the RTX 5060, RTX 5070 and RTX 5080, the narrowest on the RTX 4060 Ti, on an x8
link, and on the RTX A4000, whose host ran at a load average of 52 to 69 from other tenants.
NInfer-all's device cache held 9.8 to 10.3 GiB of the 13.1 GiB of experts on the 16 GB cards, 6.2
and 6.4 GiB on the 12 GB ones and 2.2 GiB on the 8 GB one; on the essay it served 98.5 to 99.1%,
94.5 and 94.8%, and 76.7% of the expert routes. NInfer-all's spreads stayed at or under 2.8%.
llama.cpp's decode spread most on the RTX 5080, 34.7 to 46.0 tok/s between repetitions, and there it
decoded faster after 10,000 tokens than without them (48.2 against 41.3).

The essay with the experts in host memory, tok/s; DSpark's text equals plain decoding's on every
card:

| Card | Plain | DSpark | DSpark against plain | Tokens a round |
|---|---:|---:|---:|---:|
| RTX 5080 | 114.1 | 126.1 | 1.11x | 1.64 |
| RTX 4080 Super | 102.1 | 112.9 | 1.11x | 1.72 |
| RTX 5070 | 80.9 | 79.2 | 0.98x | 1.75 |
| RTX 5060 Ti 16 GB | 70.3 | 72.1 | 1.03x | 2.07 |
| RTX 4060 Ti 16 GB | 52.3 | 55.7 | 1.07x | 1.66 |
| RTX A4000 | 59.5 | 63.2 | 1.06x | 1.68 |
| RTX 3060 12 GB | 47.4 | 48.8 | 1.03x | 1.69 |
| RTX 5060 8 GB | 33.6 | 32.5 | 0.97x | 1.71 |

With the experts in host memory DSpark gains 0.97 to 1.11 times, less than on the cards that keep
them on the GPU; the RTX 5070 and RTX 5060 decode 2.1% and 3.3% slower with it. The RTX 4080 Super
ran NInfer-all on two hosts, a Threadripper 3990X and a Ryzen 9 5950X: its benchmark rates moved by
at most 2.9% (decode after 10,000 tokens, 89.5 against 92.1 tok/s), the CLI's plain essay by 4.0%
(98.2 against 102.1), and the tables hold the second host. The first plan put every expert of
llama.cpp on the CPU (`-ot exps=CPU`); so it prefilled 10,000 tokens at 346 tok/s and decoded at
18.1 on the RTX 5070, against 724 and 35.6 with its fit, and at 338 and 7.7 on the first RTX 4080
Super host, against 1,379 and 59.2 with the fit on the second.

The RTX 5060 also ran with the experts on disk (`--expert-residency disk`, the host's NVMe, the
artifact evicted from the page cache before each start, which the runs then refill: 5.1 of its 15.6
GiB were cached during the second essay): it prefilled 2,048 and 10,000 tokens at 617 and 591 tok/s
and decoded at 9.6 tok/s, 8.3 after 10,000 tokens; the essay decoded at 8.8 tok/s, 8.7 with DSpark,
with the same text as with host experts. CPU threads computed 22.1% of the essay's expert routes
from the file, and copies brought 47.8 GiB to the card over its 384 tokens.

## Numerical checks

- **Kernels.** `ninfer_inkling2_kernels_test` qualifies the executor's own kernels against FP64
  oracles at Inkling 2 Mini's shapes: the depth mixes, the controllers, the attention's clips,
  convolutions, relative bias and sinks, the residual updates, routing and SiTU; the split rows,
  the batched controller products and the sorted route against their one-block forms bit for bit;
  verification rows of two to four positions against decode rows bit for bit (FP32 partials of
  every split and the outputs) at every position of two spans, 200 positions with 16 splits and 48
  with one; the CPU's expert products against the GPU's. `ninfer_inkling2_dspark_kernels_test`
  qualifies the drafter's kernels (its uniform-block attention over window rings and full caches,
  its mixes, rotations, the recurrent head's steps, ties to the lower id).
  `ninfer_inkling2_vision_kernels_test` qualifies the tower's kernels.
- **Against an FP64 forward.** `tools/reference/inkling2.py` runs the model in FP64 over the GGUF's
  exactly decoded blocks. On 1,150 WikiText tokens (past the 1,024-position window) NInfer-all
  scores a mean NLL of 1.560199 against its 1.559721 (+0.03%).
- **Against llama.cpp.** Perplexity on 120 KB of WikiText (`--context 4096 --stride 2048`) over the
  six 4,096-token windows both score alike: a mean NLL of 1.733757 against llama.cpp's 1.734757
  (-0.06%); per window -0.30% to +0.08% (llama.cpp's windows read from its four-digit running
  perplexities). On 40 KB, device, host and disk experts give the same perplexity, 6.415115.
- **Expert parallelism.** With a three-token prompt every call stays within eight tokens, where one
  card also sums the experts in fixed point: the split's 384 greedy tokens equal one card's byte
  for byte, with and without DSpark. A prompt chunk weighs the experts' rows in fixed point on the
  split and in FP32 on one card, so after the 33-token essay prompt the split's greedy text departs
  from one card's after about 150 tokens; 64/64 and 32/96 experts give the same text, with and
  without DSpark, and the same perplexity. On the first 40 KiB of WikiText (`--kv-dtype bf16`)
  the split scores a mean NLL of 1.866484 against one card's 1.866193 (+0.016%).
- **The vision tower.** `ninfer_inkling2_vision_real_test` (`NINFER_INKLING_VISION_ARTIFACT`,
  `NINFER_INKLING_VISION_IMAGE` and `NINFER_INKLING_VISION_DIR`, with the rows of
  `tools/reference/inkling2_vision.py` there) runs examples/cli/media/natural_scene.png, a
  40 x 26-patch grid, through the device tower against the FP64 reference of the artifact's
  mmproj: worst row cosine 0.999172 and relative RMS error 0.90% with the BF16 tower, 0.999283 and
  0.93% with the Q8_0 tower; the overlay tower gives the same rows bit for bit. Against the BF16
  tower's reference the Q8_0 tower scores 0.998210 and 1.66%, and the two FP64 references differ by
  0.999272 and 1.39%: the Q8_0 matrices' own rounding. ggml-org's and IsValorum's Q8_0 mmprojs hold
  the same tensors byte for byte and give the same rows. The GGUF product, which quantizes each
  input to q8_1 as llama.cpp does, scored 0.974362 and 2.67% against the Q8_0 reference, so the
  tower decodes the blocks to BF16 rows instead. On examples/cli/messages/image_natural.json
  (greedy) both towers name the mailbox's 24 and the sun on the right; the Q8_0 tower's text
  departs from the BF16 tower's 324 characters into the reasoning.
- **DSpark.** With 7 drafts a round over 256 greedy tokens of the essay prompt, NInfer-all's drafter
  had 133 drafts accepted of 832, llama.cpp's implementation 133 of 850.

## Expert residency

`--expert-residency device` keeps the routed GGUF banks on the GPU. `host` keeps them in pinned host
memory and `disk` in the artifact's files, read through the OS page cache. Both run the experts
with the resident kernels on the GPU, except the share of decode and verification misses that CPU
threads compute, through the transport Gemma 4 and Flint use
([Gemma 4's expert residency](gemma4.md#expert-residency) describes the layer banks, the cache, its
rounds and the growing KV): a whole-layer bank for a prompt chunk's missing experts, a cache of the
experts decode routes most (`--expert-cache-mib`), and a CPU share of a decode call's misses
(`--expert-dma-share`, `--expert-cpu-threads`) computed from the stored blocks into the same 2^-32
fixed-point sums as the GPU's, so outputs do not depend on the split. A model with BF16 matrices
(UD-Q8_K_XL) has no CPU share: its calls read BF16 rows, and the GPU copies every miss. At startup the cache takes
the most routed experts of the build's routing profile for this geometry, recorded over four chat
requests unlike the measured ones (`--expert-profile PATH` reads another); from then on it follows
the routes.

On the RTX 3090 above, the greedy essay, 384 tokens: with host experts the cache (13.1 GiB) holds
every expert the profile routes, and decode reaches 100.3 and 100.8 tok/s in two runs (100% hits)
and 117.8 with DSpark, against 109.7 and 125.3 on the device; disk experts, the artifact in a warm
page cache, 99.9 and 114.3 with DSpark. Without the profile the cache fills as the first requests
route: host 86.2 and 87.7 tok/s (95.1% hits), disk 45.6 (93.5%). The profile adds 0.6 to 0.9 s to
a host start and 1.6 s to a disk start. With a cache of part of the experts the profile also raises
the hits: with `--expert-cache-mib 8192` (1,879 slots) a build of November 21 decoded 1,536 tokens
1.05 times as fast (85.2 and 87.8 tok/s against 81.1 and 83.1, 98.2% hits against 97.3%), and its
first 384 tokens 0.89 to 0.99 times as fast in three pairs (73.6-77.9 against 79.0-83.0; with
4,096 MiB 0.98 times on the essay and 0.89 times on the Python task), each run's text equal to the
device's. The build of November 22 no longer decodes the first tokens slower: three essays after
a start decode at 87.1, 90.0 and 89.9 tok/s with the built-in profile and at 87.3, 90.1 and 90.0
with an all-zero one (8,192 MiB, the same text). With 4,096 MiB the CPU share, 18 pinned threads
by measured speeds, decodes the three essays at 61.6-61.9 tok/s against 58.8-59.5 with
`--expert-dma-share 1`, which copies every miss, with the same text.
