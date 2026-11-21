# Inkling 2 Mini

NInfer runs Thinking Machines' [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
(`Inkling2LoopForConditionalGeneration`; its text component `Inkling2ForCausalLM`), a looped
mixture-of-experts model of 26.4B parameters with 3.4B active per token (5.4B with its three
passes), from a llama.cpp GGUF: Unsloth's
[UD-IQ4_XS](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) or another quantization of the
release, with the release's vision tower and DSpark drafter. It is the engine's first looped
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
  (Q6_K or Q5_K in 7 layers' projections), the token table Q8_0 and the head Q6_K. The router,
  convolutions, distance tables, mixing queries, controllers, norms and expert scales are FP32.
  Vector kernels multiply up to eight tokens, integer matrix kernels the prompt chunks, the expert
  products the routed experts only.
- **The looped state.** A decode or verification call of at most 16 rows spreads each depth mix
  and residual update over one-warp blocks of 128 channels instead of one block per row. Each warp
  keeps its sums, and every reader adds them in the order the one-block kernel adds its warps', so
  the values are the same bits. A controller's nine products run in two launches, and the route
  kernel also groups the pairs by expert for the expert kernels. These kernels took decode on an
  RTX 3090 from 88.9 to 111.4 tok/s with the same tokens ([measurements](#measurements)).
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
  model's own lowest level, which the release notes do not fully disable reasoning. Thinking
  budgets do not apply: the model opens its thinking blocks itself and has no early-close control,
  as with Gemma 4 and Flint.
- **Sampling defaults.** The release's: temperature 1.0.
- **Speculative decoding.** The release's DSpark drafter (the artifact's `dspark` component,
  `--spec dspark`) proposes a block of up to 7 drafts from the target's tapped hidden states: three
  non-causal blocks over the anchor and mask rows, then a recurrent Markov head of rank 256 that
  corrects the target head's logits draft by draft. Without `--spec`, an artifact that carries the
  drafter drafts with it while the Engine serves one request at a time: 3 drafts a round, the
  round's width adapting from 1 to 3 by how far the drafts survive and how long the rounds take
  (`--adaptive-mtp`), each proposal ending after its first draft at or below probability 0.5
  (`--draft-min-p`). The target keeps a draft only while it equals the token it draws itself, so
  the output is plain decoding's. Context lookup (`--lookup-ngram`, 1 to 7 drafts, 3 by default)
  also drafts, and so does a [DSpine](maintainer/dspine.md) drafter trained for this model
  (`--spec dspine --draft-model PATH`; none is published), which taps the visits' states; with a
  random drafter of this geometry the text equals plain decoding's, greedy and sampled.
- **Images.** The release's vision tower (the artifact's `vision` component, `--vision`): a ViT
  with a class row, learned position rows resized to each image and 2D-rotated heads, then 2 x 2
  patch merging and a two-layer projector, up to 4,096 positions an image (fewer with
  `--vision-max-merged`). Its matrices stay on the GPU or, with `--vision-residency overlay`, in
  pinned host memory through two device slots.
- **Expert residency.** The routed experts stay on the GPU, in pinned host memory or in the
  artifact's files, with a device cache and a CPU share of decode misses
  ([below](#expert-residency)). Several GPUs divide the routed experts (`--devices 0,1
  --tensor-parallel`, [expert parallelism](maintainer/pipeline-stages.md#expert-parallelism-of-the-other-moe-families));
  this path has not run on more than one GPU yet.

Not built for this family: the quantized `--kv-dtype` formats (the KV cache is F16, `bf16` or
`fp16`), MTP, DFlash, n-gram copy drafts and their archive, `--draft-confidence`, rotation scaling
(the attention has no rotation), thinking budgets, video, pipeline stages, and of the context
cache the shared paged KV pool, the disk tier and session slots. DSpark drafts for one request at a
time; with several requests decoding, each decodes a token a step.

## Converting

The converter reads the weights from the GGUFs and the tokenizer, chat template and generation
config from the Hugging Face release:

```bash
mkdir -p models/inkling2/hf
for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json; do
  curl -fsSL -o models/inkling2/hf/$f https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/$f
done
hf download unsloth/Inkling-2-Mini-GGUF Inkling-2-Mini-UD-IQ4_XS.gguf mmproj-BF16.gguf \
  dspark-Inkling-2-Mini-Q8_0.gguf --local-dir models/inkling2

python -m tools.convert --model models/inkling2/hf --recipe inkling2_gguf \
  --source gguf=models/inkling2/Inkling-2-Mini-UD-IQ4_XS.gguf \
  --source vision=models/inkling2/mmproj-BF16.gguf \
  --source dspark=models/inkling2/dspark-Inkling-2-Mini-Q8_0.gguf \
  --components text,vision,dspark --name Inkling-2-Mini-UD-IQ4_XS \
  --out models/Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer --device cpu
```

The conversion takes about 25 seconds on a 64-CPU container and writes one 16,790,806,016-byte
(15.6 GiB) `.ninfer` file of 1,185 objects: the text model, the vision tower (BF16 matrices) and
the DSpark drafter (Q8_0). Unsloth's repository holds each quantization in two layouts, the root
file's `inkling2` (48 unrolled blocks in its metadata) and llama.cpp's `inkling2_loop` (24 blocks,
`Mainline_Support/*.gguf_file`). Both convert to the same artifact: from UD-IQ4_XS the two payloads
are equal byte for byte, and the headers differ only in the recorded source path and the random
artifact id.

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
prefix.

## Measurements

`ninfer_bench` against llama.cpp b12588's `llama-bench` on the same UD-IQ4_XS GGUF (flash
attention on, F16 KV), one sequence, three repetitions each, on an RTX 3090 (350 W, CUDA 12.8,
build `86`, experts on the GPU), November 21, 2026. NInfer's prompts are Inkling-tokenized text,
llama-bench's its own tokens. Decode after 10,000 tokens follows a 10,000-token prompt
(`-pg 10000,128` for NInfer, `tg128 @ d10000` for llama.cpp).

| Engine | Prefill 2,048 | Prefill 10,000 | Decode 128 | Decode after 10,000 |
|---|---:|---:|---:|---:|
| NInfer | 2,659 | 2,031 | 111.4 | 95.9 |
| llama.cpp | 1,301 | 1,228 | 58.6 | 55.9 |

NInfer against llama.cpp: 2.04, 1.65, 1.90 and 1.72 times. NInfer's spreads stay under 0.2%,
llama.cpp's under 3.9% (its prefill of 2,048 tokens ±50.6 tok/s). Before the split depth mixes,
the batched controllers and the sorted route ([what runs](#what-runs)) NInfer decoded 88.9 and
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

## Numerical checks

- **Kernels.** `ninfer_inkling2_kernels_test` qualifies the executor's own kernels against FP64
  oracles at Inkling 2 Mini's shapes: the depth mixes, the controllers, the attention's clips,
  convolutions, relative bias and sinks, the residual updates, routing and SiTU; the split rows,
  the batched controller products and the sorted route against their one-block forms bit for bit;
  the CPU's expert products against the GPU's. `ninfer_inkling2_dspark_kernels_test` qualifies the
  drafter's kernels (its uniform-block attention over window rings and full caches, its mixes,
  rotations, the recurrent head's steps, ties to the lower id).
  `ninfer_inkling2_vision_kernels_test` qualifies the tower's kernels.
- **Against an FP64 forward.** `tools/reference/inkling2.py` runs the model in FP64 over the GGUF's
  exactly decoded blocks. On 1,150 WikiText tokens (past the 1,024-position window) NInfer scores a
  mean NLL of 1.560199 against its 1.559721 (+0.03%).
- **Against llama.cpp.** Perplexity on 120 KB of WikiText (`--context 4096 --stride 2048`) over the
  six 4,096-token windows both score alike: a mean NLL of 1.733757 against llama.cpp's 1.734757
  (-0.06%); per window -0.30% to +0.08% (llama.cpp's windows read from its four-digit running
  perplexities). On 40 KB, device, host and disk experts give the same perplexity, 6.415115.
- **The vision tower.** `ninfer_inkling2_vision_real_test` (`NINFER_INKLING_VISION_ARTIFACT`,
  `NINFER_INKLING_VISION_IMAGE` and `NINFER_INKLING_VISION_DIR`, with the rows of
  `tools/reference/inkling2_vision.py` there) runs examples/cli/media/natural_scene.png, a
  40 x 26-patch grid, through the device tower against the FP64 reference: worst row cosine
  0.999172, relative RMS error 0.90%; the overlay tower gives the same rows bit for bit.
- **DSpark.** With 7 drafts a round over 256 greedy tokens of the essay prompt, NInfer's drafter had
  133 drafts accepted of 832, llama.cpp's implementation 133 of 850.

## Expert residency

`--expert-residency device` keeps the routed GGUF banks on the GPU. `host` keeps them in pinned host
memory and `disk` in the artifact's files, read through the OS page cache. Both run the experts
with the resident kernels on the GPU, except the share of decode and verification misses that CPU
threads compute, through the transport Gemma 4 and Flint use
([Gemma 4's expert residency](gemma4.md#expert-residency) describes the layer banks, the cache, its
rounds and the growing KV): a whole-layer bank for a prompt chunk's missing experts, a cache of the
experts decode routes most (`--expert-cache-mib`), and a CPU share of a decode call's misses
(`--expert-dma-share`, `--expert-cpu-threads`) computed from the stored blocks into the same 2^-32
fixed-point sums as the GPU's, so outputs do not depend on the split. At startup the cache takes
the most routed experts of the build's routing profile for this geometry, recorded over four chat
requests unlike the measured ones (`--expert-profile PATH` reads another); from then on it follows
the routes.

On the RTX 3090 above, the greedy essay, 384 tokens: with host experts the cache (13.1 GiB) holds
every expert the profile routes, and decode reaches 100.3 and 100.8 tok/s in two runs (100% hits)
and 117.8 with DSpark, against 109.7 and 125.3 on the device; disk experts, the artifact in a warm
page cache, 99.9 and 114.3 with DSpark. Without the profile the cache fills as the first requests
route: host 86.2 and 87.7 tok/s (95.1% hits), disk 45.6 (93.5%). The profile adds 0.6 to 0.9 s to
a host start and 1.6 s to a disk start. With a cache of part of the experts the profile also raises
the hits, but the first tokens decode slower: with `--expert-cache-mib 8192` (1,879 slots), 1,536
tokens decode 1.05 times as fast (85.2 and 87.8 tok/s against 81.1 and 83.1, 98.2% hits against
97.3%) and 384 tokens 0.89 to 0.99 times as fast in three pairs (73.6-77.9 against 79.0-83.0); with
4,096 MiB, 384 tokens decode 0.98 times as fast on the essay (three pairs) and 0.89 times on the
Python task (one pair). Every run's text equals the device's.
