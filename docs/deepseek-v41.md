# DeepSeek-V4.1-Flash

NInfer runs DeepSeek's [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(`DeepseekV41ForCausalLM`) from IST Austria's
[GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) (IQ1_M)
or its expert-pruned
[Coder build](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF) (192
of the 384 experts a block), with the Engram table GGUF (IQ4_NL) and the vision tower (the
repository's BF16 mmproj). It is its own
model family next to Qwen3.5, Flash-Next, Gemma 4 and Flint: its own executor
(`src/models/deepseek_v41`) and Engine core
(`src/runtime/engine/deepseek_v41_core`), sharing the Qwen3.5 frontend for tokenization, chat
templates, output parsing and the serving protocols in a DeepSeek dialect.

## What runs

- **The model.** 40 blocks of hidden width 5120 whose residual stream is four copies mixed by
  hyper-connections (a Sinkhorn normalization of 20 iterations at every sublayer). Each block has
  one latent attention: a low-rank query (rank 1280) of 64 heads of 512 values over a single key
  and value latent (keys are values), an interleaved RoPE on its last 64 values, a learned sink
  logit per head and a grouped low-rank output (8 groups of rank 1024). Every block attends over a
  window of 128 positions; blocks 2 to 19 also attend over latents compressed two tokens to one,
  blocks 20 to 39 over one compressed latent per token, of which an indexer (32 heads of 128) keeps
  the best 512 for each query; block 20 first keeps each query's best 2048 blocks of eight latents,
  and the later blocks rank only inside those: once a query sees more than 2,048 blocks they read
  and score only the kept blocks' keys and select among their scores (four consecutive prompt
  positions share one list of the blocks any of them keeps), which gives the selection of scoring
  every position. The MoE routes each token to 6 of 384 experts
  (sqrt-softplus scores, SwiGLU of width 2304 with its gate clamped at 10 and its up projection
  within ±10) beside a shared expert. Blocks 1 and 14 add Engram rows, the 2- to 4-grams ending
  at each position hashed into a table and gated by the stream. A 129,280-token head.
- **The weights stay the GGUF's blocks.** Every matrix keeps its ggml format, imported row for row
  without requantization and multiplied by the GGUF kernels. The routed experts' 80 banks (each
  block's gate with up, and its down) are IQ1_S (55 banks), IQ1_M (12), IQ2_XXS (11) and IQ2_S
  (2); the attention's query and latent down-projections Q8_0; the rest of the attention, the
  shared experts, the Engram projections, the token table and the head Q5_K. The router, the
  compressors, the indexers, the hyper-connection matrices, the norms and the sinks are FP32.
- **The release's roundings.** The attention keeps what the release's `kernel.py` rounds: the window
  latents in FP8 (E4M3 per 32 values with a power-of-two scale), the indexer's queries and keys in
  FP4 (E2M1 per 32, power-of-two scale), the compressed latents in FP4 (per 16, E4M3 scale) and the
  attention output in FP8 before its projection, each scale computed in FP32 as the release does.
  The caches hold those values as stored codes and scales: a window row is 512 E4M3 codes and 16
  E8M0 scales (528 bytes), a compressed latent 256 bytes of E2M1 pairs and 32 E4M3 scales (288
  bytes), an indexer key 64 bytes of E2M1 pairs and 4 E8M0 scales. Attention and the indexer
  decode them to the same values.
- **Requests.** One to eight concurrent sequences (`--max-concurrency`), prompt chunks of 2048
  tokens (`--prefill-chunk`, at most 4096; narrower on a card without room for them) run block by
  block in spans of up to eight chunks ([below](#prompt-spans)), the last 19 blocks only over the
  prompt rows the output depends on ([below](#the-prefill-tail)), prefix reuse across turns and
  requests with
  host-memory images ([below](#prefix-reuse)), branched reads for
  [typed decisions](serving.md#typed-decisions), structured output, streaming, log probabilities,
  causal scoring (`ninfer-perplexity`; a window that does not start with `<｜begin▁of▁sentence｜>`
  gets it in front), and the model's DSML tool calls returned as OpenAI tool calls, under the
  tool's JSON Schema when it is `strict` and beside a response format:

  ```text
  <｜DSML｜ calls>
  <｜DSML｜ invoke name="get_weather">
  <｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>
  </｜DSML｜ invoke>
  </｜DSML｜ calls>
  ```

  A value with `string="true"` is the text as written, any other its JSON. A forced `tool_choice`
  ends the generation prompt with the opening of that call, so the model continues with its
  parameters.
- **Reasoning.** The chat template renders the conversation as the release's `encoding.py` encodes
  it. Thinking is on by default: the generation prompt opens the reasoning and the model closes it
  with `</think>`, and the reasoning arrives as `reasoning_content`. `reasoning_effort` `low`,
  `high` (the default) or `max` writes the release's effort line (50, 75 or 100) into the system
  block; `enable_thinking: false` or `reasoning_effort: "none"` closes the reasoning at once.
  A reached thinking budget, or the [reasoning-loop guard](serving.md#openai-chat-completions)
  under `conclude`, closes the reasoning with Qwen's early-close guidance and the `</think>` token,
  and the model answers from what it has thought.
- **Sampling defaults.** The release's: temperature 1.0, top-p 1.0 (top-k 20, the candidates the
  sampler keeps).
- **Speculative decoding with the DSpark drafter.** An artifact converted with the release's drafter
  (the `dspark` component, from beamster's STQ pack as in the published artifact, or from Unsloth's
  GGUF, [below](#the-dspark-drafter)) drafts with `--spec dspark`: up to 5 tokens a round
  (`--draft-tokens`, 1 to 5), kept while the drafter's confidence is at least `--draft-confidence`
  (0.5 by default), verified by the target in one pass of decode rows. Its tokens and log
  probabilities are plain decoding's.
- **Speculative decoding by context lookup.** Without `--spec`, `--lookup-ngram N` proposes what
  followed the last N tokens' earlier occurrence in the sequence (up to `--draft-tokens`, 3 by
  default, at most 7): the occurrence the sequence was already copying while it still matches, else
  the newest, and the target verifies the proposal in one pass of decode rows; the output
  is the one plain decoding gives.
- **Expert residency.** The routed experts stay on the GPU, in pinned host memory or in the
  artifact's files ([below](#expert-residency)); host memory is the residency to use. With host
  experts, CPU threads compute a share of a decode call's missing experts while the GPU copies the
  rest ([the CPU share](#the-cpu-share)).
- **The Engram table.** Stored in the model's artifact or in a table artifact of its own, read row
  by row from its file, held in RAM, or with its most read rows in RAM ([below](#the-engram-table)).
- **Images.** With `--vision`, the release's vision tower turns each picture into a span of prompt
  positions, prepared as the release's image processor prepares it; the text model routes those
  positions with the release's image routing bias and leaves them out of the Engram
  ([below](#images)).

Not built for this family: the quantized `--kv-dtype` formats (the caches are BF16), MTP, DFlash
and n-gram copy drafts, DSpark drafts while several requests decode, rotation scaling
(`--rope-yarn`, `--rope-scaling-factor`: the compressed blocks carry the release's own YaRN), video
and the CPU vision tower, pipeline stages, and of the context cache the shared paged KV pool, the
disk tier and session slots. Several GPUs divide the routed experts instead (`--devices 0,1
--tensor-parallel` with host or disk experts: each device its own banks, cache and copies, no CPU
share, prompts in [spans](#prompt-spans) as on one device; without `--tensor-split` the first
device, which also holds the model, serves fewer experts, so that every device's misses take the
same time over its bus; [expert
parallelism](maintainer/pipeline-stages.md#expert-parallelism-of-the-other-moe-families)).

The Q4_K async prompt tiles (`gguf_matrix/q4_k`, `gguf_moe_matrix/q4_k`) do not apply to this
family: its routed experts are IQ1_S, IQ1_M, IQ2_XXS and IQ2_S, none Q4_K.

## Converting

The converter reads the weights from the GGUF shards, the Engram GGUF, the drafter of beamster's
Sushi STQ pack and the vision tower's mmproj, and the tokenizer and configuration from the base
release. The release ships no Jinja chat template; the converter stores its own transcription of
the release's encoder (`tools/convert/templates/deepseek_v41`).

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 config.json tokenizer.json tokenizer_config.json \
  --local-dir models/deepseek-v4.1/hf
hf download ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF \
  --revision 65d53bc8be65123e255ff02f3ded958b6dbf54fd \
  --include 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-*.gguf' \
  --include 'DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf' \
  --include 'mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf' --local-dir models/deepseek-v4.1
hf download beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw \
  --revision 07fec972538639726b868f81e89d9d4f5005fe10 --include config.json \
  --include model.safetensors.index.json --include 'model-mtp-*.safetensors' \
  --local-dir models/deepseek-v4.1/stq

# The model with its DSpark drafter and vision tower; it names its Engram table by the SHA-256 of
# the table's rows.
python -m tools.convert --model models/deepseek-v4.1/hf --recipe deepseek_v41_gguf \
  --components text,dspark,vision \
  --source gguf=models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00006.gguf \
  --source engram=models/deepseek-v4.1/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf \
  --source dspark=models/deepseek-v4.1/stq \
  --source vision=models/deepseek-v4.1/mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf \
  --name DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M \
  --out models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-ninfer-v3.ninfer --device cpu
# The Engram table, one artifact for every DeepSeek-V4.1-Flash-1017 conversion, with an optional
# hot-row profile for --ngram-residency ram-hot (the published table holds the broad one); the
# profile is checked against the Engram hash of the model's GGUF (--source gguf).
python -m tools.convert --model models/deepseek-v4.1/hf --recipe deepseek_v41_gguf --components engram \
  --source engram=models/deepseek-v4.1/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf \
  --source gguf=models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00006.gguf \
  --resource engram.hot=models/deepseek-v4.1/engram-broad-v1.hot \
  --name DeepSeek-V4.1-Flash-Engram-IQ4_NL \
  --out models/DeepSeek-V4.1-Flash-Engram-IQ4_NL-ninfer-v3.ninfer --device cpu
```

These are the files in llama.cpp's layout (b11870 and newer), which ISTA-DASLab publishes since
2026-10-27 (the revision above): five shards, the first holding only metadata, and the Engram GGUF
as the sixth, which llama.cpp opens through a link named `...-00006-of-00006.gguf`; the converter
needs no link and takes the file from `--source engram`. GGUFs downloaded earlier (four shards,
revision `653a9874aa38c356233c5f3cc8c96b19a43b57eb`) convert too, to the same artifact: apart from
the source paths it records, every object, configuration and resource is the same bytes. They lack
the routing bias of image positions, so for the vision tower the converter reads the release's
`layers.N.ffn.gate.bias_vl` by byte range from its safetensors shards
(`tools/convert/fetch_tensors.py`, 65 KB of its 48 shards):

```bash
python -m tools.convert.fetch_tensors deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --match 'layers.*.ffn.gate.bias_vl' --out models/deepseek-v4.1/bias_vl.safetensors
# As above with the earlier shards, and the bias.
python -m tools.convert --model models/deepseek-v4.1/hf --recipe deepseek_v41_gguf \
  --components text,dspark,vision \
  --source gguf=models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf \
  --source engram=models/deepseek-v4.1/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf \
  --source dspark=models/deepseek-v4.1/stq \
  --source vision=models/deepseek-v4.1/mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf \
  --source bias_vl=models/deepseek-v4.1/bias_vl.safetensors \
  --name DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M \
  --out models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-ninfer-v3.ninfer --device cpu
```

ISTA-DASLab's Coder build (`ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF`, IQ2_XXS,
revision `54229d38d588a66d1c6cff08af7b281b9326c656`) keeps 192 of the 384 routed experts per block
and comes only in llama.cpp's layout. It converts with the same command, its first shard as
`--source gguf` and the base release's config; the converter takes the expert count from the GGUF.
Its repository's Engram GGUF and mmproj are the base repository's files, so it uses the same Engram
table. The model is one file of 71.9 GiB, published with the STQ drafter and the vision tower
([running](#running)). 11 of its 80 expert banks, in 10 of its 40 blocks, are IQ2_XS, which CPU
threads do not compute: in those blocks the GPU copies every miss whatever `--expert-dma-share`
says.

The model reads the Engram GGUF only for the table's digest, and of the STQ pack only its config,
its index and its six drafter files (3.85 GB), checking that the pack states the release's model and
drafter keys. `--components text,dspark` leaves the vision tower out and `--components text` the
drafter too; the objects they keep are the same bytes either way. The model with its drafter and
tower is one file of 117.2 GiB, of which the tower takes 0.90 GiB; the table is one of 103.0 GiB
(103.3 GiB with the published hot-row profile, [below](#the-hot-row-profile)).
On the 96 threads of a 48-core EPYC 7K62, with a download writing to the same disk, the model took
6 min 23 s; on 24 cores of a Threadripper 3960X the table took 3 min 29 s.

The drafter can come from Unsloth's `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` instead, the
release's drafter in Q8_0 with MXFP4 routed experts, which takes twice the STQ drafter's device
memory at the same acceptance ([the STQ drafter](#the-stq-drafter)):

```bash
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf \
  --local-dir models/deepseek-v4.1
# As above, with the GGUF as the drafter's source.
python -m tools.convert --model models/deepseek-v4.1/hf --recipe deepseek_v41_gguf \
  --components text,dspark,vision \
  --source gguf=models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00006.gguf \
  --source engram=models/deepseek-v4.1/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf \
  --source dspark=models/deepseek-v4.1/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf \
  --source vision=models/deepseek-v4.1/mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf \
  --name DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M \
  --out models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-Unsloth-DSpark-ninfer-v3.ninfer --device cpu
```

Its text and vision objects are the same bytes as with the STQ drafter; the file is
129,946,453,248 bytes (121.0 GiB).

## Running

Three artifacts are published: the model, with the STQ drafter and the vision tower, as
[WaveCut/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-NInfer-v3](https://huggingface.co/WaveCut/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-NInfer-v3),
ISTA-DASLab's Coder build with the same drafter and tower as
[WaveCut/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-NInfer-v3](https://huggingface.co/WaveCut/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-NInfer-v3),
and the table both read as
[WaveCut/DeepSeek-V4.1-Flash-Engram-IQ4_NL-NInfer-v3](https://huggingface.co/WaveCut/DeepSeek-V4.1-Flash-Engram-IQ4_NL-NInfer-v3).

```bash
M=models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-ninfer-v3.ninfer
# Or the Coder build: M=models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-ninfer-v3.ninfer
T=models/DeepSeek-V4.1-Flash-Engram-IQ4_NL-ninfer-v3.ninfer
# A chat server for two concurrent requests over 32K-token contexts, experts in pinned host memory.
ninfer-serve $M --ngram-table $T --model-id deepseek-v4.1-flash \
  --max-context 32768 --max-concurrency 2 --expert-residency host
# The same server reading images (OpenAI image_url parts, Anthropic image blocks).
ninfer-serve $M --ngram-table $T --model-id deepseek-v4.1-flash \
  --max-context 32768 --max-concurrency 2 --expert-residency host --vision

# One answer from the CLI.
ninfer $M --ngram-table $T --expert-residency host --prompt "Explain how a heat pump works."
```

On a 24 GB card with host experts the GPU holds the 6.0 GiB of dense weights, two layer banks
(7.1 GiB), a 1.3 GiB workspace, the latent caches (40 MiB at 32,768 tokens of context) and an
expert cache that takes the rest (7.9 GiB on an RTX 4090, of which a prompt span borrows up to
1.3 GiB while it runs); the host pins 106.7 GiB of experts. With disk experts the host pins 138 MiB
of transfer buffers and the experts' pages sit in the OS page cache.

## Expert residency

The routed experts are 106.7 GiB of the artifact, more than any one card holds, so the model runs
with `--expert-residency host` or `disk`; `device` needs a card with room for every expert. The GPU
runs the experts it caches or receives with the resident kernels, through the transport Gemma 4 and
Flint use ([Gemma 4's expert residency](gemma4.md#expert-residency) describes the layer banks, the
cache and its rounds):

- **Host.** The experts are read from the artifact once at startup into pinned host memory, and a
  layer's misses are copied from there. The host needs that memory beside the Engram table's page
  cache (or its RAM copy) and the rest of the system: 106.7 GiB pinned, so a host with 128 GB
  keeps about 12 GiB for everything else (not tested; the measurements below ran with 188 GiB or more).
  The startup reads the experts with direct I/O, past the page cache, on eight threads into
  ordinary pages and pins them once read.
- **Disk.** The experts stay in the artifact's files. A layer's misses are read with ordinary file
  reads through the OS page cache into bounded pinned transfer buffers, then copied to the device;
  NInfer maps no expert file into memory, so a miss never waits on a page fault inside a copy.
  Whatever the page cache holds is read at memory speed, the rest at the storage's.
  `--expert-mirror` spreads these reads over a second drive and `--expert-direct` lets the drive
  write them into device memory ([disk expert reads](expert-disk-reads.md)).
- **Layer banks and cache.** A prompt chunk's experts missing from the cache land in a whole-layer
  bank laid out like the resident bank; two banks let the next layer arrive while the current one
  computes. `--expert-cache-mib` (default: what is free after startup) holds the experts decode
  routes most, filled at startup from a routing profile (`--expert-profile`, else the built-in
  one for the model's experts a block: the release's, recorded on summaries of four corpora and on
  chat, code and reasoning requests, or the [Coder build's](#coder-build), recorded on 48 coding,
  agent and reasoning requests).
- **CPU share.** With host experts, CPU threads compute a share of a decode or verification
  call's misses from the pinned copy while the GPU copies and runs the rest
  ([below](#the-cpu-share)). Prompt chunks and spans copy every miss.

## The CPU share

A decode or verification call of up to eight tokens leaves part of its missing experts to CPU
threads, which compute them from the stored blocks while the GPU copies and runs the rest; the GPU
then adds the CPU's sums to its own:

- **The split.** `--expert-dma-share auto` (the default) gives each call the split whose copies and
  CPU products end soonest by the copy and CPU times of earlier calls; a fixed `F` copies that
  fraction of the misses and leaves the CPU the experts with the fewest pairs; `1` copies every
  miss. `--expert-cpu-threads N` sets the pool, by default one thread per physical core and at
  most half the usable CPUs, pinned and spinning while requests decode.
- **Arithmetic.** For each pair the CPU computes the expert's gate and up rows over the call's q8_1
  activation (the guarded input the GPU quantized, copied to host memory with the route weights and
  guard factors), the clamped SwiGLU with the guard factor taken back, rounded to BF16 and
  quantized as q8_1, and the down rows, and adds each row times the pair's route weight into its
  token's 2^-32 fixed-point sums, as the GPU's vector kernels compute them
  ([Qwen3.8-Flash-Next's CPU share](qwen3-8-flash-next.md#the-cpu-share) gives the IQ1_S, IQ1_M,
  IQ2_XXS and IQ2_S slice products). Integer sums do not depend on their order, so tokens, log
  probabilities and context states do not depend on the split, the share or the thread count.
- **Disk experts.** An expert the CPU computes from disk needs the same file read as a copied one,
  and on the host below every split of disk experts decoded slower than copying every miss
  ([measurements](#the-cpu-share-on-one-rtx-4090)), so `auto` copies every miss of disk experts; a
  share below `1` still gives the CPU its part.

Flash-Next's CPU share also covers the missing experts with at most eight pairs in calls of 9 to 64
tokens ([the CPU share](qwen3-8-flash-next.md#the-cpu-share)); DeepSeek-V4.1 does not have that
route yet: its expert kernels take one route per call, and the shared expert delivery serves only
narrow calls.

## IQ1_M experts in prompt chunks

Calls of more than eight tokens run the routed experts on ggml's integer tensor-core matrix kernel,
the IQ1_M banks (12 in the IQ1_M build, 44 in the Coder build) once a call routes enough rows to
each expert: 6 on average for their down products and more than 16 for their gate and up products,
from 384 and 1,025 tokens with 384 experts a block and from 192 and 513 with the Coder build's 192
(a 2048-token chunk routes 32 or 64). Below that they run on the vector kernel in groups of eight
columns: the matrix kernel would leave most of its tile columns empty and gather a row for every
pair, where the vector kernel reads each token's row once for all of its experts. Up to 16 rows an
expert its column tiles are 16 wide, and in prompt steps on an RTX 4090 it ran gate and up products
there at 0.89 to 1.02 times the vector kernel's speed
([IQ1_M routes per card](#iq1_m-routes-per-card)). Causal scoring (`ninfer-perplexity`) runs
256-token steps: 8 rows an expert with the Coder build's 192 experts, whose down products take the
matrix kernel and gate and up products the vector kernel, and 4 with 384 experts, all on the vector
kernel. llama.cpp multiplies IQ1_M on its vector kernels only; NInfer's tile loader unpacks each
row of eight grid values g, shifted by ±1/8, into the int8 values 8g ± 1 and each 16 values' scale
into d (2s + 1) / 8, the integers and scales of its own vector kernel. `--iq1m-experts vector` keeps
every IQ1_M call on the vector kernel. The two routes keep the activation's scale per 32 values in
FP32 (matrix) or binary16 (vector) and sum in different orders, so their outputs differ slightly;
`ninfer_linear_gguf_moe_iq1m_test` holds both against FP64 products at the experts' shapes. On the
Tesla V100 the IQ1_M banks stay on the vector kernel.

## Prompt spans

A prompt runs in chunks of 2048 tokens unless `--prefill-chunk` asks for another multiple of 128
(at most 4096); a card without room for a chunk's workspace takes the widest of 2048, 1024 and 512
below it that fits, and the startup log says so. A 2048-token chunk routes to nearly every expert
of every block, so a prompt run chunk by chunk copies nearly all the experts the cache lacks once
per chunk: about 100 GiB a chunk with host experts. With host or disk experts the executor runs a
prompt's chunks block by block instead:

- **One sweep per span.** Every chunk of a span passes a block before any passes the next, so the
  block's bank is filled once for all of them, and the delivery fills the next block's bank while
  the chunks compute. A chunk's arithmetic is the same as chunk by chunk, so the outputs are the
  same ([below](#numerical-checks)).
- **Span length.** Eight chunks (16,384 tokens), four or two, within `--max-context`: the longest
  whose buffers fit in the memory the expert cache would get and take no more than an eighth of
  that memory and the two banks together. The startup log names it (`prompt chunks of 2048 tokens
  block by block in spans of 16384 tokens`); a card without room for two chunks runs chunk by
  chunk.
- **Memory.** A span keeps its residual streams, coefficients and index selections between blocks:
  1,323 MiB for 16,384 tokens. Beside the default elastic expert cache, the span borrows them from
  the cache while it runs and gives them back after it, so decode keeps the whole cache; a span
  the cache cannot lend them to at that moment runs as two halves of whole chunks, each a span the
  cache may lend to, down to chunk by chunk. With a fixed
  `--expert-cache-mib` they stay in the workspace. The Engram rows of a span's tokens are staged
  on the card (about 95 MiB more than for one chunk).
- **Several GPUs.** With `--tensor-parallel` the span's buffers stay on the first device, which
  also sizes the span, and each MoE call still takes one chunk: every device's delivery fills its
  bank of a block once for all of the span's chunks. Measured on two RTX 3090s without peer
  access (2026-11-04, IQ1_M with host experts, one run each): a 9,672-token prompt prefilled at
  670.9 tok/s against 451.4 chunk by chunk (one RTX 3090 with spans: 576.5), with the same
  tokens; decode did not change (30.3 tok/s after that prompt, 41.4-41.7 on a short one).

## The prefill tail

Blocks 21 to 39 come after block 20, the last block that keeps compressed caches. Each attends over
block 20's caches and its own window of 128 positions and keeps nothing else, so the state those
blocks leave at a position, and the logits there, depend only on their rows within 19 × 127 + 1 =
2,414 positions before it. `--prefill-tail` chooses the prompt rows they run:

- **`exact`** (the default). A prompt step runs blocks 21 to 39 from the row 2,414 positions
  before the state its run of steps keeps (the prompt's end, the checkpoint where the last user
  turn closes, a shared prefix), starting at the prompt chunk that holds that row; a step that
  ends before that chunk leaves them out. Each chunk runs whole or not at all, the rows a step runs
  read none it left out, and the products keep the shapes of a full run, so tokens, log
  probabilities and kept states are `full`'s bit for bit.
- **`full`** runs every row through every block.
- **`N`** (at least 128, the window) runs the last `N` rows before each kept state, from that row
  rather than from its chunk: a lossy mode. Its products take other shapes, so they round
  differently from `full`'s, and below 2,414 the kept states and logits also lose rows they depend
  on.
- **A stopped prompt.** A request stopped inside its prompt leaves those blocks behind the rest of
  the sequence's state. NInfer keeps no image of that state; the next prompt on the sequence starts
  from its last checkpoint, from the host store or from the beginning.
- **The last block.** A prompt chunk that produces no logits runs block 39 only up to its window
  latents: its attention output and experts feed only the head. The latents are those of the whole
  block, so tokens, log probabilities and kept states keep their bits under every tail.
- **Scoring.** Causal scoring computes every position's logits, so it runs every row. A chunk before
  the first scored position runs block 39 only up to its window latents, unless an `N` tail shorter
  than the chunk would then leave rows out.

Prefill rates, the outputs and the cost of a stopped prompt are
[measured below](#prefill-tail-iq1_m-experts-and-engram-reads).

## The Engram table

The two Engram blocks read 24 rows per token each (the 2-, 3- and 4-grams ending at the position,
hashed per head), IQ4_NL rows of 256 values. Before each pass the executor reads the rows it
addresses and copies them to the device:

- **Where it is stored.** In its own artifact: the model names the table by digest, and
  `--ngram-table PATH` supplies it (NInfer refuses a table with another digest). A model run without
  its table is refused unless `--no-ngram-table` runs it without the Engram, which degrades the
  output: on a 1,112-token WikiText slice the perplexity rose from 3.07 to 9.68 with the build that
  added the model ([numerical checks](#numerical-checks)).
- **How it is read.** `--ngram-residency disk` (the default) reads each pass's rows from the file
  (`--ngram-io`, `--ngram-io-depth`, `--ngram-ram-mib` for a row cache), `ram` holds the whole
  table in RAM (`--ngram-lock` pins it), and `ram-hot` holds the rows a hot-row profile ranks first,
  as many as `--ngram-ram-mib` holds (4,096 MiB by default), and reads the rest from the file
  ([the hot-row profile](#the-hot-row-profile)). The table is 103.0 GiB, so `ram` and host experts
  together need about 210 GiB of RAM.
- **Repeated rows.** A pass that addresses one row several times, as a prompt that repeats its
  n-grams or an image's positions (which all hash the pad n-grams) do, reads it from the file once
  and copies it to the other places (`--ngram-dedupe off` reads every place).
- **Reading ahead.** While a prompt step computes, the rows of the sequence's next prompt step are
  read into a second pinned buffer, so that step starts without waiting for the file
  (`--ngram-read-ahead off` reads each step's rows when it starts). The Engine passes the next
  step's tokens with the current one; a step that does not match them, such as a request stopped in
  between, reads its own rows. The second buffer takes 114 MiB of pinned memory with spans of
  16,384 tokens and 14 MiB with chunks of 2048
  ([measurements](#prefill-tail-iq1_m-experts-and-engram-reads)).

### The hot-row profile

A row depends on its token and the three before it and nothing else, so `ninfer-ngram-profile`
counts the rows a corpus reads from its tokens alone, without running the model, and ranks them by
reads ([Flash-Next's hot-row profile](qwen3-8-flash-next.md#the-n-gram-rows) describes the tool and
its file). Given a DeepSeek-V4.1 artifact it hashes the Engram's 48 rows a token and sizes a row at
144 bytes. A profile names the table's row count and a fingerprint of the Engram hash (the n-gram
sizes, heads, pad id, each block's multipliers, primes and offsets, and the model's token map);
NInfer refuses one that differs, repeats a row or names a row past the table.

The table artifact may carry a profile in `engram.resources.hot_profile`; `ram-hot` reads it when
`--ngram-hot-profile` is absent, and an explicit file overrides it. Conversion stores the file that
`--resource engram.hot=PATH` names, after the same checks; the table's digest and rows stay the
same bytes. The published table carries a broad profile: 86,355,715 rows counted from 3,734,454
tokens (5,831 documents of the corpus Flash-Next's profile was counted from). On 1,116,980 held-out
tokens of the same sources (53,615,040 row reads, 47.6% of them of rows the profile lists) the
leading rows serve this share of reads per RAM budget:

| RAM budget, MiB | 256 | 512 | 1,024 | 2,048 | 4,096 | 8,192 | 16,384 and more |
|---|---:|---:|---:|---:|---:|---:|---:|
| Held-out row hits | 18.4% | 23.4% | 28.1% | 32.8% | 37.7% | 43.1% | 47.6% |

At the default 4,096 MiB the RAM holds 29,117,807 rows. A row of 144 bytes and 48 reads a token give
lower hit rates per MiB than Flash-Next's 90-byte rows and 16 reads a token. Hits measure row reuse,
not speed or quality.

At startup `ram-hot` reads whole every 4 MiB piece of the table that holds more than about 500 hot
rows and the hot rows of the other pieces one by one. At 4,096 MiB that is nearly the whole file: on
an RTX 3090 host (RunPod, container disk, November 13, 2026) the engine was ready after 2 min 26 s
against 8.3 s with `disk`. The rows are the same either way: there the IQ1_M release with disk
experts scored the first 16 KB of `wikitext/00.txt` (3,547 tokens) to the same total NLL, bit for
bit, with `disk` and with `ram-hot` (perplexity 2.09739).

## The DSpark drafter

DeepSeek-V4.1-Flash-1017 ships DSpark, a drafter of three blocks of the backbone's shape. Two
quantizations of it convert to the artifact's `dspark` component: Unsloth's
`dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf`, in Q8_0 with MXFP4 routed experts, whose blocks the
converter stores as they are, and the drafter of beamster's Sushi STQ pack
([the STQ drafter](#the-stq-drafter)). With `--spec dspark` the executor loads either beside the
model:

- **What it reads.** At every position the backbone computes, the mean of the four streams entering
  blocks 37, 38 and 39, projected (`fc`) and normalized into a key and value latent of each drafter
  block, written to that block's window ring: the prompt's last 128 rows, every decode row and the
  rows a verification keeps.
- **A round.** The drafter runs the last accepted token and four noise tokens through its three
  blocks in one pass (each row attends over the ring and over all five rows; the MoE routes each
  row to 3 of 128 experts beside a shared expert), adds a rank-256 Markov term of the previous token
  to the backbone's head logits and takes each row's greedy token, so five rows give five drafts.
  A confidence head scores each draft; the round keeps the drafts up to the first scored below
  `--draft-confidence` (0.5 by default, Strata's for this drafter; 0 keeps every draft), at most
  `--draft-tokens`, and with none decodes one token.
- **Exactness.** The target verifies the drafts in one pass of decode rows. Its attention splits a
  row's keys the same way however many rows a pass has, and the log probabilities split a row's
  vocabulary the same way however many rows they reduce, so a verified token and its log
  probability are bit for bit those of plain decoding: greedy runs with and without the drafter
  gave the same tokens and log probabilities with host and disk experts on an RTX 4090 and with
  host experts on an RTX PRO 6000. A sampled request keeps a draft only while it equals the
  target's own draw, as context-lookup drafts are kept.
- **One request.** The drafter drafts while one request decodes; with more, each round decodes one
  token per request.
- **Memory.** Unsloth's drafter's weights take 7.4 GiB of device memory, almost all of it its 384
  MXFP4 experts, and the STQ drafter's 3.58 GiB, or 2.0 GiB with the 64 experts a block it keeps on
  the card by default ([the drafter's experts](#the-drafters-experts)); the rings take 0.39 MiB per sequence (twice that
  with prefix checkpoints) and the tapped streams 7.5 MiB. The expert cache gives up that memory:
  on an RTX 4090 with host experts it keeps 73 experts with Unsloth's drafter and 618 with the STQ
  drafter instead of 1,137, and with Unsloth's drafter prompt spans shorten to 8,192 tokens.

On October 26, 2026, two rented hosts ran Unsloth's drafter on the prompts of the
[Strata comparison](#against-strata-on-one-host) (one request, greedy, thinking off, host experts):
that comparison's RTX 4090 host, with and without [the CPU share](#the-cpu-share), and an RTX PRO
6000 (600 W, PCIe 5.0 x16) with a 24-core Threadripper PRO 7965WX and 188 GiB of RAM, NVIDIA driver
595.99, with the build before the CPU share, which copied every miss as `--expert-dma-share 1` does
(the CPU share was not run there):

| card | drafts | decode, short answer | decode after 8,086 tokens | prefill of 8,086 tokens | drafts accepted, short and long |
|---|---|---:|---:|---:|---:|
| RTX PRO 6000 | none | 50.8 tok/s | 48.3 tok/s | 1,210 tok/s | |
| | `--spec dspark` | 83.1 tok/s | 75.0 tok/s | 1,290 tok/s | 38 of 58, 16 of 26 |
| | `--spec dspark --draft-tokens 3` | 83.3 tok/s | 77.0 tok/s | 1,290 tok/s | 37 of 53, 16 of 24 |
| RTX 4090, CPU share | none | 38.9 tok/s | 31.4 tok/s | 938 tok/s | |
| | `--spec dspark` | 39.9 tok/s | 33.2 tok/s | 693 tok/s | 43 of 68, 16 of 31 |
| RTX 4090, `--expert-dma-share 1` | none | 33.5 tok/s | 26.6 tok/s | 929 tok/s | |
| | `--spec dspark` | 25.1 tok/s | 20.2 tok/s | 688 tok/s | 43 of 68, 16 of 31 |

- **A card that caches most experts gains.** With the drafter the RTX PRO 6000 keeps 10,317 of the
  15,360 experts on the card (11,387 without) and decodes 1.6 times as fast after the short prompt
  and 1.55 times after the long one; three drafts a round decode as fast as five.
- **A 24 GB card gains with the CPU share.** With the drafter the RTX 4090 keeps 73 experts on the
  card (1,137 without), so nearly every expert a round's rows route to is missing. With the CPU
  share the drafter decodes 1.03 and 1.06 times as fast as plain decoding; copying every miss it
  decodes 25% and 24% slower. Either way prefill slows by 26%, in spans of 8,192 tokens
  ([why](#the-stq-drafter)). On the
  same host Strata's `--dspark on` decodes at 34.3 and 38.7 tok/s
  ([above](#against-strata-on-one-host)).
- **Strata on the RTX PRO 6000.** Strata 0.2.6 exited with `router picked expert 2147483647` in
  each of its ten runs on that host: the flags its app starts the engine with, `--dspark on` and
  `--dspark on --dspark-block 3`, each on both prompts; the app's flags with `--reserve-mib 4096`
  and `16384`, which left 4 and 16 GiB of the card free; `--dspark off`; and `--dspark off
  --vram-experts 0.5`. That host has no Strata numbers.
- **Memory.** On the RTX PRO 6000 the server peaked at 116.2 GiB of resident host memory with the
  drafter and 116.0 GiB without it.

One run per row; no spread is measured.

A [DSpine](maintainer/dspine.md) drafter trained for this model can draft instead:
`--spec dspine --draft-model PATH --draft-tokens N` (1 to 7) loads it beside a model without
`--spec dspark`, and `--draft-min-p P` verifies through the first draft whose confidence is at or
below P. Its rounds verify as DSpark's do, for up to eight requests at once. If it taps a block
after the last KV source block, the [prefill tail](#the-prefill-tail) is off. No DSpine drafter for
DeepSeek-V4.1 is published yet.

### The STQ drafter

beamster's `DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw` pack, made for sushi (an engine for Apple
Silicon), stores the release's drafter (`mtp.0..2`) in STQ, a trellis code. A matrix's weights
are 16 × 16 tiles, each a bitstream of 16-bit words read as overlapping codewords: 64 words a tile
in the trunk (SPEC1, 15-bit codewords, 4 bits a weight) and 32 in the routed experts (PAIR1, 16-bit
codewords, 2 bits a weight). A hash of a weight's codeword selects two FP16 halves whose sum, times
4,096, is the weight, and the matrix works between orthonormal 128-point Hadamard transforms with
FP16 sign vectors: y = s_out ⊙ H(Wᵀ H(s_in ⊙ x)). The converter stores the words and signs as the
pack does (`uint16` and `fp16` objects) and the drafter's vectors as the release's. The executor's
STQ product decodes each tile in registers and keeps the activations in FP32, for the 5 rows of a
draft and the ring's rows in tiles of 8; the grouped `wo_a` and the routed experts (3 of 128 a row,
all on the card) run as products over ids. The rest of a round, the rings, the Markov term and the
confidence head are those of Unsloth's drafter.

The two drafters quantize the same weights differently; their vectors, Markov tables and
confidence heads are equal. Decoded, the STQ trunk's matrices agree with Unsloth's Q8_0 ones at
cosines of 0.996 to 0.998, and the gate, up and down products of five sampled experts a block with
its MXFP4 ones at 0.91 to 0.97, except two of block 2's down products at 0.84 and 0.89.

On October 27, 2026, both drafters ran beside the same GSQ backbone (the artifacts' text objects
equal byte for byte) on two rented hosts: an RTX 4090 (450 W, PCIe 3.0 x16) with a 24-core EPYC
7352 and 251 GiB of RAM, NVIDIA driver 615.71.09, and an RTX PRO 6000 (500 W, PCIe 5.0 x16) with a
Core Ultra 9 285K and 249 GiB of RAM, driver 580.178.04. One request, greedy, host experts, the
CPU share on: the two prompts of the [Strata comparison](#against-strata-on-one-host) (the short
answer, up to 128 tokens; 32 tokens after the 8,086-token prompt), a code request (384 tokens), a
reasoning request with thinking on (384 tokens) and a summary of the 8,086-token prompt (up to 256
tokens). Decode rates are the second run of each request, its prompt from the prefix cache;
prefill is the 8,086-token prompt's first run.

Decode and prefill, tok/s:

| card | drafter | `--draft-confidence` | experts on the card | short answer | after 8,086 tokens | code | reasoning | summary | prefill of 8,086 tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| RTX 4090 | none | | 1,137 | 28.1 | 28.7 | 22.8 | 30.3 | 20.5 | 620 |
| | Unsloth's | 0.5 | 73 | 22.0 | 19.7 | 23.2 | 26.0 | 18.2 | 405 |
| | STQ | 0.5 | 618 | 25.9 | 24.3 | 27.1 | 36.6 | 20.9 | 634 |
| | Unsloth's | 0 | 73 | 14.6 | 11.8 | 20.1 | 23.6 | 11.7 | 405 |
| | STQ | 0 | 618 | 18.9 | 14.9 | 24.4 | 30.5 | 13.9 | 633 |
| RTX 4090, `--expert-cache-mib 500` | none | | 70 | 20.9 | 19.6 | 18.1 | 22.0 | 17.3 | 623 |
| | Unsloth's | 0.5 | 70 | 22.5 | 19.9 | 23.1 | 27.5 | 19.1 | 409 |
| | STQ | 0.5 | 70 | 23.4 | 21.4 | 24.3 | 29.5 | 18.7 | 624 |
| RTX PRO 6000 | none | | 11,387 | 84.2 | 79.6 | 81.4 | 82.4 | 79.4 | 1,250 |
| | Unsloth's | 0.5 | 10,316 | 114.3 | 107.0 | 170.0 | 143.0 | 123.2 | 1,250 |
| | STQ | 0.5 | 10,868 | 119.2 | 102.2 | 168.8 | 136.5 | 115.9 | 1,260 |
| | Unsloth's | 0 | 10,316 | 94.9 | 72.4 | 158.2 | 127.9 | 92.2 | 1,240 |
| | STQ | 0 | 10,868 | 97.6 | 70.9 | 157.5 | 124.3 | 96.0 | 1,260 |

Drafts over the five requests' first runs; tokens per target pass count verification passes and
the steps that decode one token:

| card | drafter | `--draft-confidence` | drafts accepted | accepted at positions 1 to 5 | tokens per target pass |
|---|---|---|---:|---:|---:|
| RTX 4090 | Unsloth's | 0.5 | 724 of 1,035 | 264, 183, 123, 94, 60 | 3.00 |
| | STQ | 0.5 | 728 of 987 | 265, 182, 127, 94, 60 | 3.03 |
| | Unsloth's | 0 | 761 of 1,635 | 258, 186, 136, 108, 73 | 3.34 |
| | STQ | 0 | 762 of 1,630 | 259, 189, 134, 106, 74 | 3.35 |
| RTX PRO 6000 | Unsloth's | 0.5 | 657 of 905 | 241, 155, 113, 82, 66 | 3.07 |
| | STQ | 0.5 | 655 of 889 | 236, 162, 110, 84, 63 | 3.05 |
| | Unsloth's | 0 | 688 of 1,440 | 231, 166, 124, 91, 76 | 3.40 |
| | STQ | 0 | 692 of 1,420 | 226, 170, 121, 96, 79 | 3.45 |

- **Acceptance is equal.** On each card the two drafters accept the same number of drafts within
  1%, with close counts at each position; at the 0.5 floor the STQ drafter proposes 2 to 5% fewer.
  Every arm gave plain decoding's tokens on its card (the two cards' answers differ from each
  other, so acceptance compares within a card).
- **On a 24 GB card the STQ drafter is the faster one.** With it the RTX 4090 keeps 618 experts on
  the card, with Unsloth's 73, and decodes 1.15 to 1.41 times as fast as with Unsloth's drafter.
  Against plain decoding it gains 19% on code and 21% on reasoning and 2% on the summary, and loses
  8% on the short answer and 15% after the long prompt; Unsloth's drafter gains only on code (2%).
  With both caches held at 500 MiB the STQ drafter decodes 1.04 to 1.08 times as fast as Unsloth's
  on four requests and 0.98 times on the summary, so most of its lead is the cache it leaves.
- **Prefill beside Unsloth's drafter.** It slows the RTX 4090's prefill from 620 to 405 tok/s; the
  STQ drafter's 634 tok/s matches plain prefill. A span borrows its buffers from the expert cache;
  beside Unsloth's drafter the 73-expert cache cannot lend a whole prompt's, the 8,086-token prompt
  runs in pieces and every layer's experts cross the bus twice. Nsight Systems counted 198 GiB of
  host-to-device copies during the request with Unsloth's drafter against 97 GiB with the STQ
  drafter, at the same kernel time (4.9 s). With the cache fixed at 500 MiB, Unsloth's spans shrink
  to 4,096 tokens and the prompt takes two (409 tok/s), while the other arms keep one span of
  16,384. The RTX PRO 6000 prefills at 1,240 to 1,260 tok/s in every arm.
- **A card that caches most experts.** On the RTX PRO 6000 both drafters decode 1.3 to 2.1 times
  as fast as plain decoding. The STQ drafter is 4% faster than Unsloth's on the short answer and 1
  to 6% slower on the other four requests; what costs it the difference (its product's decode
  arithmetic, for one) is not measured.
- **The floor.** At `--draft-confidence 0` both drafters gain tokens per target pass (3.34 to 3.45
  against 3.00 to 3.07) and decode slower on every request on both cards: each verification
  carries all five drafts, and the extra rows route to more experts.
- **Memory.** The server's resident host memory peaked at 115.6 to 116.0 GiB in every arm.

One run per row (the short answer's rate is the mean of its two warm runs); no spread is measured.

### The drafter's experts

`--draft-experts N` keeps on the card each drafter block's N experts that the drafter routes to
most, and routes the block among those N; the other experts stay in pinned host memory and the
drafter never reads them. The order is Basalt's (MIT), counted on live runs of the drafter. `0`
keeps every expert on the card. The default is 64 with the STQ drafter and host or disk experts,
whose expert cache takes the memory the drafter gives back, and every expert otherwise: with every
expert on the card nothing takes it, and Unsloth's drafter was not measured. Drafts change, verified tokens do not: the target verifies every
draft as before. Each block must keep at least the 3 experts a row routes to.

On November 10, 2026, the Coder build's STQ drafter ran on an RTX 4090 (450 W, a RunPod host with
a 32-core EPYC 7532) with host experts and on an RTX PRO 6000 Blackwell Workstation (600 W, a Vast
host) with every expert on the card. One CLI request, greedy, thinking off: a code request
answered with 300 tokens on the RTX 4090 and 256 on the RTX PRO 6000. The output bytes were the same
in every run on a card:

| card | kept a block | drafter experts on the card | expert cache | decode | drafts accepted |
|---|---:|---:|---:|---:|---:|
| RTX 4090 | no drafter | | 11.6 GiB, 1,824 slots | 35.1 / 35.1 tok/s | |
| | 128 | 3,262 MiB | 7.95 GiB, 1,374 slots | 42.2 / 42.5 tok/s | 221 of 263 |
| | 64 | 1,631 MiB | 9.53 GiB, 1,571 slots | 47.3 / 47.4 tok/s | 222 of 261 |
| | 32 | 815 MiB | 10.3 GiB, 1,670 slots | 45.1 / 45.2 tok/s | 207 of 267 |
| RTX PRO 6000 | 128 | 3,262 MiB | | 173.4 tok/s | 184, 77.0% of drafts |
| | 64 | 1,631 MiB | | 175.9 tok/s | 184, 79.0% of drafts |
| | 32 | 815 MiB | | 165.5 tok/s | 177, 75.6% of drafts |
| RTX PRO 6000 Server, no Engram table | 128 | 3,262 MiB | | 163.2 tok/s | 223, 76.9% of drafts |
| | 64 | 1,631 MiB | | 155.9 tok/s | 219, 72.8% of drafts |

With host experts the memory the drafter gives back goes to the expert cache: 64 kept decode 12%
faster than all 128, 32 kept accept fewer drafts. With every expert on the card nothing takes that
memory: 64 decoded 1.4% faster than 128 on the Workstation card and 4.5% slower on the Server card
(a 300-token answer, `--no-ngram-table`, the same bytes either way), one run each, so device
experts keep all 128.

## Images

An artifact converted with the release's vision tower (the `vision` component: the mmproj's BF16
tower and aligner, its preprocessing configuration and the text model's routing bias of image
positions) reads images with `--vision`:

- **Preprocessing.** As the release's `image_processor.py`: an image of fewer than 295,936 pixels
  is first scaled up to that area, then sized to whole 14-pixel patches; when its span would
  take more prompt positions than the budget, it shrinks to the largest grid of 3 × 3-patch cells
  within it. The image is resized to fit that size with its aspect kept (Pillow's bicubic filter,
  reproduced bit for bit), centered on gray (127) and scaled to (x / 255 - 0.5) / 0.5.
- **The prompt.** An image takes a span of `<｜deepseek_image｜>` positions where the user's turn
  places it: a learned start row, the aligner's cells grid row by grid row, each followed by a
  learned newline row, and a learned end row, rows × (columns + 1) + 2 positions. The release's
  budget is 1,024 positions (a 31 × 31-cell grid at most for a square image);
  `--vision-max-merged N` lowers it, and an image part with `detail: "low"` takes at most a
  quarter of it.
- **The tower.** 32 blocks of width 1,024 (16 heads of 64, a SwiGLU MLP of width 2,816, a 2D RoPE
  over the patch grid) attend over all of an image's patches; after the final norm the aligner
  projects each 3 × 3 cell of patches through a GELU into the text model's width of 5,120. The
  matrices multiply BF16 inputs into FP32 sums.
- **The text model.** The span's rows take the place of the tokens' embeddings. The router adds the
  release's image routing bias at those positions, and the Engram neither adds rows there nor
  looks back past them: an n-gram that would reach into an image starts after it.
- **When it runs.** Each image is encoded once per request, when a prompt chunk first reaches its
  span, and its rows stay in host memory until the prompt is done. Prefix reuse keys an image by
  its content and grid, so a cached prompt with another picture at the same place is not reused.
- **Residency.** `--vision-residency resident` (the default) keeps the tower's 0.90 GiB of BF16
  matrices on the GPU; `overlay` keeps the 32 blocks' 0.77 GiB in pinned host memory and copies
  each block's matrices into one of two device slots while the block before computes, the patch
  projection and the aligner (0.14 GiB) staying on the GPU. Both give the same rows bit for bit.
  `cpu` is refused: this family's tower runs on the GPU. The device memory either takes comes out
  of the expert cache (on an RTX 4090 with host experts, 135 of its 1,126 expert slots resident and
  32 with `overlay`); an encode's 365 MiB of scratch fits in the prompt chunks' workspace.

Not supported with images: video, branched reads and causal scoring.

## Prefix reuse

With the context cache on (the default), a prompt resumes from a state a sequence or the host store
keeps instead of prefilling those tokens again:

- **On its sequence.** A sequence keeps its state when its request ends, and a checkpoint of its
  window rings and compressor tails where the prompt's last user turn closes; the next turn of a
  conversation resumes from either.
- **From host memory.** Before another request takes a sequence, what it held goes to an image in
  pinned host memory: the window rings, the compressor tails and the compressed caches up to its
  length. A prompt on any sequence resumes from the longest image it starts with. `--host-kv-mib`
  (or `--host-cache-mib`) sets the store.
- **Shared prefixes.** A prompt chunk also ends where the frontend marks a prefix other requests
  share (the end of the leading system messages, of the tool declarations, a declared cache
  boundary), and the state there goes to an image too. That end also ends a prompt span, so the
  first prompt with such a prefix copies or reads the experts once more than a prompt without it.
  Flash-Next keeps the state inside its span instead; here a span keeps one state, the one its
  prefill tail and drafter taps are planned for (`state_at`), and a second one is not
  implemented.

A window state cannot be cut back, so a prompt resumes only where a state was kept, not at an
arbitrary common prefix.

A resumed prompt does not reproduce a cold prefill's log probabilities bit for bit, and a decode
moves them no more than a prefill split at the same place does. On an A40 (IQ1_M, host experts),
the first answer token of a 134-token second turn had a log probability 0.076 from the cold
prefill's when the turn resumed from the state the first turn's decode left, and 0.18 from it when
the turn resumed from a prefilled prefix of the same 119 tokens; greedy, both answers left the cold
prefill's text at a near tie. On Qwen3.8-27B the same three paths put that log probability within
0.004 of each other and gave the same text.

A turn that a stop string ends returns to the state where the stop string's match began, the one
its visible text continues: the live state has read the tokens the cut removed. Each sequence
holds a second copy of its rings and tails for it, written when a step's sampled token begins a
match, and by a round's commit when the match begins among the round's tokens (a verification
writes every row by its position, so the committed bytes also serve the position before the
match). The round itself is not changed, so the turn produces the tokens it produces without the
stop string. On an RTX 4090 with the Coder model and host experts, the turn after a reply of 186
tokens that a stop string cut at its 182nd reused 203 of its 217 prompt tokens, every token the
two prompts share, as after a reply that its output limit ended there; the same with the DSpark
drafter, context lookup and neither.

## Measurements

### Against Strata on one host

October 25 and 26, 2026, one rented host: an RTX 4090 (450 W, PCIe 4.0 x16), two 32-core AMD EPYC
engineering samples (128 threads) and 409 GiB of RAM, NVIDIA driver 615.71. One sequence, greedy
decoding, thinking off, the same prompt tokens in both engines: a 14-token question and an
8,086-token summary request (32 tokens of answer).

- **NInfer** served the model with `ninfer-serve --max-context 16384 --max-concurrency 1
  --expert-residency host`: the current build on October 26 (61 CPU threads by default) and the
  native release build of `b3402a6fd`, which ran prompts chunk by chunk and copied every miss.
- **Strata** 0.2.6, its latest release (its DeepSeek engine `strata-dsv41` 0.1.51, build
  `562a0c2ed`, from the release's Linux CUDA 13 package), ran the same GGUF shards and Engram GGUF
  through its `pack_gguf.py` index, with the DSpark drafter from Unsloth's
  `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` and Strata's routing profile, as
  `strata-dsv41 --max-context 16384 --eos 1 --temperature 0` over the prompt's token ids. The first
  Strata row has the flags Strata's app starts the engine with; the next four are the settings its
  tuner tries. `--experts 5` keeps 5 of each token's 6 experts, so it computes another model.

| engine and settings | decode, short answer | decode after 8,086 tokens | prefill of 8,086 tokens |
|---|---:|---:|---:|
| NInfer, host experts | 38.9 tok/s | 31.4 tok/s | 938 tok/s |
| NInfer, host experts, `--expert-dma-share 1` | 33.5 tok/s | 26.6 tok/s | 929 tok/s |
| NInfer `b3402a6fd`, host experts | 25.8 tok/s | 22.1 tok/s | 498 tok/s |
| NInfer, host experts, `--spec dspark` | 39.9 tok/s | 33.2 tok/s | 693 tok/s |
| Strata, `--dspark auto --dspark-conf 0.65` | 36.0 tok/s | 38.4 tok/s | 635 tok/s |
| Strata, `--dspark on --dspark-conf 0.65` | 34.3 tok/s | 38.7 tok/s | 690 tok/s |
| Strata, `--dspark on --dspark-conf 0.65 --dspark-block 3` | 39.9 tok/s | 37.1 tok/s | 660 tok/s |
| Strata, `--dspark auto --dspark-conf 0.65 --cpu-threads 64` | 38.0 tok/s | 38.6 tok/s | 659 tok/s |
| Strata, `--dspark auto --dspark-conf 0.65 --mmap-experts off --ssd-cache 0` | 39.4 tok/s | 38.3 tok/s | 658 tok/s |
| Strata, `--dspark auto --dspark-conf 0.65 --experts 5` (5 of 6 experts) | 44.8 tok/s | 42.1 tok/s | 644 tok/s |

- **Decode.** Strata computes the experts its VRAM cache misses on CPU threads beside the GPU:
  12,396 of the 20,400 expert products of its first short answer (61%) and 3,921 of 7,440 after
  the long prompt (53%). With the build `7731f3d30` NInfer's device cache served 62 to 72% of the
  decode routes, its CPU share computed 16 to 22% and the GPU copied the rest
  ([the CPU share](#the-cpu-share-on-one-rtx-4090)). With all six experts and no drafts, Strata's
  settings decode 0.93 to 1.01 times as fast as NInfer after the short prompt and 1.22 to 1.23
  times after the long one; with both engines' drafters, Strata's fastest setting for each prompt
  decodes 1.00 and 1.17 times as fast as `--spec dspark`
  ([the decode step](#the-decode-step-on-one-rtx-4090) shows where NInfer's time goes).
- **Prefill.** Strata prefills in windows of 16,384 tokens; NInfer runs the 8,086 tokens as one
  [prompt span](#prompt-spans), 1.9 times the release's rate and 1.36 to 1.48 times Strata's. With
  the drafter NInfer's spans shorten to 8,192 tokens and it prefills at 693 tok/s, against Strata's
  635 to 690.
- **Drafts.** With `--dspark on`, Strata accepted 22 of 30 drafts on the short answer and 10 of 12
  on the long one; `--dspark auto` offered none. NInfer's `--spec dspark` accepted 43 of 68 and
  16 of 31 ([the drafter](#the-dspark-drafter)) and decodes 1.03 and 1.06 times as fast as without
  drafts.
- **After a long prompt.** NInfer's slower decode after the 8,086 tokens follows the answer's
  routes, not the context's length. On another RTX 4090 host (EPYC 7K62, PCIe 4.0 x16) with host
  experts, one server answered in turn a short question (256 tokens), a question about one
  paragraph of the document (a 250-token prompt), the summary request (400 tokens) and the short
  question again: 31.0, 28.7, 28.1 and 31.7 tok/s. In every 5-second window of those answers the GPU
  copied about 78 GB of experts (15.6 GB/s), 516 MB a token for the short answer and 545 to 623 MB
  for the document's: a summary routes to more experts the device cache lacks. On an RTX 5090 host
  (PCIe 5.0 x16, a cache of 15% of the experts) the same four answers decoded at 48.5 to 49.6
  tok/s. Strata copies no missing expert: its CPU threads compute them.
- **Repeatability.** Strata's greedy answers to the short prompt ran 74 to 119 tokens over its six
  runs. NInfer's greedy tokens were the same in every run, in both builds, and so were the current
  build's log probabilities, with and without the CPU share. The current build's short answer is 94
  tokens and the release's 78: the current GPU rounds the IQ1_S, IQ1_M and IQ2_XXS products step by
  step, as the CPU share computes them ([the CPU share](#the-cpu-share)). The release's log
  probabilities after the long prompt, which followed the short one on each server, differed
  between servers by up to 0.024 at answer tokens 28 and 29: a race in the expert cache that the
  current build fixes ([outputs](#prompt-spans)).
- **Memory.** Strata peaked at 95.6 to 103.3 GiB of resident host memory, NInfer at 115.6 to 115.7 GiB.

### The CPU share on one RTX 4090

On October 26, the host of the Strata comparison (61 CPU threads in the default pool), the same
two requests and `ninfer-serve --max-context 16384 --max-concurrency 1`; disk experts with the
model file in the page cache:

| experts | `--expert-dma-share` | decode, short answer | decode after 8,086 tokens | prefill of 8,086 tokens |
|---|---|---:|---:|---:|
| host | `auto` | 38.9 tok/s | 31.4 tok/s | 938 tok/s |
| host | `1` (every miss copied) | 33.5 tok/s | 26.6 tok/s | 929 tok/s |
| host | `0` (every miss on the CPU) | 31.6 tok/s | 24.0 tok/s | 936 tok/s |
| disk | `auto` (every miss copied) | 18.3 tok/s | 13.5 tok/s | 598 tok/s |
| disk | `0` | 2.9 tok/s | 1.9 tok/s | 573 tok/s |
| disk | `0`, `--expert-cpu-threads 8` | 6.3 tok/s | | |

- **Host experts.** The split by measured speeds decodes 1.16 and 1.18 times as fast as copying
  every miss; with every miss on the CPU decode is 6% and 10% slower than copying. Prefill does
  not use the CPU.
- **Where the routes go.** With the build `7731f3d30`, two servers with a request log decoded 400
  tokens after each prompt
  (`ignore_eos`), at 32.0 and 25.5 tok/s copying every miss and 36.9 and 31.0 tok/s with `auto`.
  Over the decode intervals the device cache served 72 to 74% of the routes after the short prompt
  and 69 to 75% after the long one, and the GPU copied 0.53 to 0.54 and 0.53 to 0.68 GB per token;
  with `auto` the cache served 69 to 72% and 62 to 69%, the CPU computed 16 to 18% and 19 to 22%,
  and the GPU copied 0.25 to 0.31 and 0.29 to 0.39 GB per token.
- **Disk experts.** A build that split disk experts by measured speeds decoded at 3.4 and 2.3 tok/s
  with 61 threads and 13.0 and 8.5 with 8, against 16.0 and 12.2 when that build copied every miss;
  with every miss on the CPU, more threads were slower too. Prefill, which the CPU share does not
  compute, ran at 573 tok/s beside the 61-thread pool and 598 tok/s without a pool. The cause of
  the larger pool's cost is not established. `auto` therefore copies every miss of disk experts.
- **Outputs.** Every run with a CPU share, host and disk, with and without `--spec dspark`, gave
  the tokens and log probabilities of copying every miss: the 94 tokens of the short answer and the
  32 after the long prompt.
- **Memory.** The server peaked at 115.7 GiB of resident host memory with host experts and 9.1 GiB
  with disk experts, with and without the CPU share.

One run per row; no spread is measured.

### The decode step on one RTX 4090

On October 26, Nsight Systems (`nsys profile --trace=cuda --cuda-graph-trace=node`) traced one
request decoding 200 tokens after the short prompt on the host of the Strata comparison, host
experts, `auto`, with the build `7731f3d30` and with the current one. Medians over 190 decode
steps:

| | `7731f3d30` | current |
|---|---:|---:|
| decode step | 32.6 ms | 27.5 ms |
| waits for missing experts (40 a step) | 13.5 ms | 11.9 ms |
| other kernels | 14.1 ms | 11.8 ms |
| GPU idle | 5.0 ms | 3.8 ms |
| copied to the GPU | 214 MB | 171 MB |
| GGUF vector products | 523 a step, 7.1 ms | 203 a step, 5.1 ms |
| grouped products (routed experts and output projection) | 160 a step, 2.5 ms | 200 a step, 3.6 ms |
| activation guards, quantizations and unscalings a step | 483, 643, 443 | 203, 363, 163 |
| `gemv`, 147 a step | 10.4 µs each | 6.9 µs each |
| `hc_split`, 80 a step | 6.8 µs each | 5.2 µs each |
| `hc_pre_norm`, 81 a step | 5.7 µs each | 3.8 µs each |

- **The grouped output projection is one product.** The attention output's eight head groups each
  go through their own 1,024 rows of `output_a`. A decode or verification call of up to eight
  tokens ran them as eight GGUF vector products, each with its own activation guard, quantization
  and unscaling, which read the weights at about 470 GB/s. The groups now run as the eight experts
  of one grouped vector product, with one guard, quantization and unscaling for every group; each
  token's group computes the arithmetic of that group's own product, so every value keeps its bits.
  Longer calls keep the eight products.
- **Loads ahead.** `gemv` (the hyper-connections' mixes, rows of 20,480 values), `hc_split` and
  `hc_pre_norm` give each row one block. With one load in flight per thread a block read about
  8 GB/s: the 24 rows of a mix took 10.4 µs. Each thread now issues four loads ahead (eight in
  `hc_pre_norm`, which keeps its 5,120 values in registers between the norm and the mix) and adds
  them in the previous order, so every sum keeps its bits.
- **Columns over the card.** `guard_columns` scales each activation column by a power of two before
  the GGUF products with one block of up to 1,024 threads per column, and the hyper-connections'
  4 × 4 Sinkhorn normalization sits on 16 lanes of one warp, each lane adding its row or column in
  the serial loop's order.
- **Outputs.** Greedy tokens and log probabilities equal those of `7731f3d30` with host and disk
  experts, with and without the CPU share and `--spec dspark`; the DeepSeek kernels test compares
  the grouped output projection bit for bit with the groups' own products.
- **Where the rest goes.** The waits for missing experts take 43% of the step: each of the 40 MoE
  layers waits for the copies and the CPU share of its misses. The current build copied 171 MB a
  step against 214 MB; the cause of that difference is not measured.
- **The profiler** slows decode: the two builds decoded at 28.6 and 34.5 tok/s under it.

### Prompt spans

On the same host, two fresh servers of each build (`--max-context 32768`; the span build of October
25, before the CPU share, and the release) took the two requests and a third, a 24,252-token prompt
that resumes from the 8,084-token state of the second and prefills 16,168 new tokens:

| build, host experts | decode, short answer | after 8,086 tokens | prefill of 8,086 tokens | prefill of 16,168 more tokens | decode after them |
|---|---:|---:|---:|---:|---:|
| NInfer span build, spans of 16,384 tokens | 26.0, 25.9 tok/s | 22.2, 22.1 tok/s | 926, 925 tok/s | 1,200, 1,200 tok/s | 22.5, 22.5 tok/s |
| NInfer `b3402a6fd`, chunk by chunk | 26.0, 26.1 tok/s | 22.2, 22.2 tok/s | 499, 499 tok/s | 505, 505 tok/s | 23.0, 22.8 tok/s |

- **Prefill.** One span sweeps the experts once where four or eight chunks swept them four or eight
  times: 1.86 times the rate for 8,086 tokens, 2.4 times for 16,168.
- **Decode.** Equal after the short and the 8,086-token prompts. After the 16,384-token span the
  first 32 tokens decode 2% slower: the span borrowed 1,323 MiB of the cache, which comes back
  empty and refills with the experts decode misses. Spans keep 94 MiB of Engram staging, so the
  cache starts 1.1% smaller (8,094 against 8,188 MiB).
- **Outputs.** Both builds produced the same tokens in every run. In the release, and in the span
  build with a fixed `--expert-cache-mib 8150` (spans of 4,096 tokens), the long prompt's log
  probabilities after the short request differed between servers by up to 0.024 at answer tokens
  28 and 29. A decode call that read an expert from a lent bank slot could move that expert to its
  cache slot while the call was served; a miss of the same call then took the lent slot and
  overwrote it before the call's cached experts had read it. The current build moves such an
  expert only at a later call: four servers with the fixed cache then gave a fresh server's log
  probabilities exactly, and the default cache's spans gave the same log probabilities in every
  run before and after the change.

A second RTX 4090 host (450 W, PCIe 4.0 x16, a 48-core EPYC 7K62, 503 GiB of RAM) ran the span
build too: with host experts it prefilled 8,086 tokens at 906 tok/s, and with disk experts and the
model file in the page cache at 390 tok/s and 16,168 more at 610 tok/s, decoding at 9.1 and 7.2
tok/s. On a host of the same type the release build of `ed8338c52` prefills 8,086 tokens at 915
tok/s with host experts and 341 with disk experts ([per card](#per-card)).

### Per card

October 26, 2026, one rented host per card ran the native release build of `ed8338c52` (the CUDA
13.4 Linux archive) with the same prompts and `ninfer-serve --max-context 16384 --max-concurrency 1`
on one GPU: host experts first, then disk experts with the model file read into the page cache. The
RTX PRO 6000 row is the release build of `b3402a6fd` on October 25, which ran prompts chunk by chunk
and copied every miss. The server sizes the device expert cache to the card; the model has 15,360
routed experts:

| card | host | experts | device cache | decode, short answer | decode after 8,086 tokens | prefill of 8,086 tokens | ready to serve |
|---|---|---|---:|---:|---:|---:|---:|
| RTX 5090 (600 W) | Core Ultra 7 265K, PCIe 5.0 x16 | pinned RAM | 2,236 (15%) | 51.5 tok/s | 42.2 tok/s | 912 tok/s | 25 s |
| | | disk, page cache warm | 2,266 | 16.7 tok/s | 12.0 tok/s | 440 tok/s | 9 s |
| RTX 4090 (450 W) | EPYC 7K62, PCIe 4.0 x16 | pinned RAM | 1,126 (7%) | 33.8 tok/s | 27.5 tok/s | 915 tok/s | 34 s |
| | | disk, page cache warm | 1,156 | 9.5 tok/s | 7.8 tok/s | 341 tok/s | 16 s |
| RTX 3090 (300 W) | two Xeon E5-2690 v4, PCIe 3.0 x16 | pinned RAM | 1,150 (7%) | 19.8 tok/s | 15.2 tok/s | 416 tok/s | 50 s |
| | | disk, page cache warm | 1,181 | 6.0 tok/s | 5.0 tok/s | 157 tok/s | 20 s |
| RTX PRO 6000 (600 W), `b3402a6fd` | EPYC 9B14, PCIe 5.0 x16 | pinned RAM | 11,401 (74%) | 48.0 tok/s | 45.3 tok/s | 1,310 tok/s | 64 s |
| | | disk, page cache warm | 11,431 | 37.4 tok/s | 33.4 tok/s | 431 tok/s | 29 s |

- **The cache and the host link set decode.** The RTX PRO 6000 keeps three quarters of the experts
  on the card; the 24 and 32 GB cards keep 7 to 15%, and their CPU threads compute a share of the
  rest ([the CPU share](#the-cpu-share)) while the GPU copies the others over the host link: PCIe
  5.0, 4.0 and 3.0 on the RTX 5090, RTX 4090 and RTX 3090 hosts.
- **Placements agree on each card.** Host and disk experts produced the same tokens for both
  prompts on every card. The short answer is 86 tokens on the RTX 5090, 94 on the RTX 4090, 110 on
  the RTX 3090 and 72 on the RTX PRO 6000: each card's device profile selects its own kernel
  routes, which round differently.
- **The RTX 3090 host** has two sockets; the server ran pinned to the CPUs of the GPU's NUMA node
  after the model files' pages were evicted, but the 106.7 GiB of pinned experts exceed that node's
  free memory, so part of them sits on the other node. Unpinned, host experts decoded at 19.4 and
  15.5 tok/s and prefilled at 410 tok/s.
- **Memory.** The server peaked at 115.5 to 115.7 GiB of resident host memory with host experts
  and 9.0 to 9.2 GiB with disk experts, and it fills the card in both placements.
- **Startup.** Host experts are read past the page cache into pinned memory before the server is
  ready; the disk rows' times follow reading the model file into the page cache (4 s on the RTX
  5090 host, whose page cache held it already, 50 s on the RTX 4090 host and 65 s on the RTX 3090
  host).

### RTX 3090 host

October 25, 2026, one host with two RTX 3090s (350 and 390 W, PCIe 4.0 x16), a 24-core
Threadripper 3960X and 251 GB of RAM, CUDA 12.8, a build that ran prompts chunk by chunk: one
sequence, greedy decoding, `ninfer-serve --max-concurrency 1` on one GPU, the same two requests
(an answer of 82 tokens to the short one):

| placement | decode, short answer | decode after 8,086 tokens | prefill of 8,086 tokens | ready to serve |
|---|---:|---:|---:|---:|
| experts in pinned RAM | 21.8 tok/s | 18.9 tok/s | 410 tok/s | 33 s |
| experts in pinned RAM, `--lookup-ngram 4` | 22.1 tok/s | 18.0 tok/s | 416 tok/s | |
| experts on disk, page cache warm | 7.4 tok/s | 7.4 tok/s | 117 tok/s | 13 s |

- **Placements agree.** Host experts, disk experts and host experts with context lookup produced
  the same greedy tokens for both prompts.
- **Context lookup.** On the long prompt 1 of 6 drafts was accepted: a summary repeats little of
  its source.
- **Startup.** The ready times come from a build whose expert cache starts from the built-in
  routing profile. With host experts the weights took 26.0 s (4.3 GiB/s: eight threads
  reading past the page cache into ordinary pages, pinned once read), the server was ready 33 s
  after it started, and stopping it took 15 s; that run produced the host row's tokens for both
  prompts, at 22.2 and 19.3 tok/s of decode and 419 tok/s of prefill. With disk experts the server
  was ready in 13 s and produced the same tokens, at 8.7 and 7.4 tok/s of decode and 120 tok/s of
  prefill. Host and disk experts score the WikiText slice to the same perplexity, 3.067.

### 16 GB

The model runs with 16 GB of free GPU memory. A process held the rest of the RTX 4090 of the
[prompt spans](#prompt-spans) disk run (16,375 MiB left free) and of a Tesla V100-SXM2-32GB
(16,158 MiB left free; a 32-core Xeon Gold 6130, 376 GiB of RAM, PCIe 3.0 x16, the CUDA 12.9
release archive). Beside the two layer banks (7.1 GiB) the expert cache keeps 38 to 86 of the
15,360 experts, so nearly every routed expert is copied:

| card | build | experts | device cache | decode, short answer | decode after 8,086 tokens | prefill of 8,086 tokens |
|---|---|---|---:|---:|---:|---:|
| RTX 4090 | span build of October 25, spans of 4,096 tokens | pinned RAM | 54 | 21.4 tok/s | 20.0 tok/s | 679 tok/s |
| RTX 4090 | `b3402a6fd` | pinned RAM | 57 | 21.3 tok/s | 20.0 to 20.1 tok/s | 457 tok/s |
| RTX 4090 | `b3402a6fd` | disk, page cache warm | 86 | 5.9 to 6.3 tok/s | 5.9 to 6.0 tok/s | 113 to 119 tok/s |
| Tesla V100 | `b3402a6fd` | pinned RAM | 38 | 10.5 to 10.6 tok/s | 9.3 to 9.4 tok/s | 198 to 199 tok/s |
| Tesla V100 | `b3402a6fd` | disk, page cache warm | 68 | 3.9 to 4.0 tok/s | 3.5 tok/s | 73.5 to 73.6 tok/s |

- **A real 16 GB card** keeps a few hundred MiB less free after its own CUDA context than these
  caps, and that memory comes out of the expert cache and the spans; no 16 GB card was measured.
- **The V100** copies over PCIe 3.0 and decodes at half the RTX 4090's rate.
- **Placements agree.** Host and disk experts produced the same tokens on each card.

Rows give one run, or the range of two; no spread is measured.

### Long prompts

November 4, 2026, one RTX 3090 of a host with a 48-core EPYC 7K62 and the 610.43.03 driver, the
sm_86 build with experts in pinned RAM, `--no-ngram-table` (the Engram table was not on the host),
`--max-context 65536`, one sequence, greedy decoding. The index sources of blocks 21 to 39 scored
every visible position, and now score only the blocks that block 20 keeps; a cell lists each run:

| index sources after block 20 | prefill of 27,264 tokens | decode after them | prefill of 58,117 tokens | decode after them | decode, short answer |
|---|---:|---:|---:|---:|---:|
| every position | 635 tok/s | 22.0 tok/s | 645, 632, 631 tok/s | 21.8, 22.1, 20.7 tok/s | 27.0, 27.8, 26.0 tok/s |
| the kept blocks | 638 tok/s | 22.6 tok/s | 653, 639, 639 tok/s | 22.7, 22.1, 21.2 tok/s | 27.5, 27.5, 26.5 tok/s |

- **Prefill** of the 58,117-token prompt rose by 1.0 to 1.3% and of the 27,264-token prompt by
  0.4%: scoring the whole context was a small part of a prompt chunk's time.
- **Decode** after the long prompts rose by 0.3 to 4.2% per run. The short answer never sees more
  than 2,048 blocks, so its difference of -1.1 to +2.0% between the builds is the spread of runs.
- **Outputs.** Both builds produced the same tokens in every run.

### Coder build

November 4, 2026, ISTA-DASLab's Coder build ([converting](#converting)) on one rented host: an RTX
4090 (450 W, PCIe 4.0 x16), a 64-core EPYC 7B13 in a container of 61 CPUs, 503 GiB of RAM, the
sm_89 build of `379c0d999`. `ninfer-serve --max-context 16384 --max-concurrency 1`, one sequence,
greedy decoding, thinking off: a 25-token code request answered with 256 tokens and a summary of a
7,775-token C++ file in 128 tokens. The disk row ran after the model file was read into the page
cache. The model has 7,680 routed experts:

| experts | device cache | decode, code answer | decode after 7,775 tokens | prefill of 7,775 tokens | ready to serve |
|---|---:|---:|---:|---:|---:|
| pinned RAM | 1,429 | 37.2 tok/s | 39.2 tok/s | 1,040 tok/s | 22 s |
| pinned RAM, `--spec dspark` | 979 | 47.9 tok/s | 44.3 tok/s | 1,020 tok/s | 22 s |
| disk, page cache warm | 1,444 | 15.7 tok/s | 17.8 tok/s | 774 tok/s | 9 s |

- **Placements and drafts agree.** The three servers gave the same answers. The DSpark drafter,
  trained for the unpruned model, accepted 192 of 250 and 75 of 101 drafts.
- **The CPU share.** With the CLI and its default context the code request decoded at 31.7 tok/s
  with every miss copied, 37.5 with half of them on 30 CPU threads and 37.3 with the automatic
  share (11% of the routed experts on the CPU), with the same tokens; disk experts decoded it at
  16.1 tok/s. In the 10 blocks with an IQ2_XS bank the GPU copies every miss.
- **The cache.** These runs filled it from the first requests: the code request hit 70 to 72% of
  its routed experts. The build now starts it from the Coder build's own routing profile, recorded
  on 48 coding, agent and reasoning requests. On November 4, on another RTX 4090 host (450 W, a
  32-core EPYC 7542 in a container of 31 CPUs), the CLI with host experts answered three held-out
  requests (code, code with reasoning, prose; two runs each) with the same tokens from either
  start. With the profile they hit 75.4, 80.6 and 76.3% of their routed experts against 71.4, 75.5
  and 74.8%, copied 5.4, 7.1 and 1.8% fewer bytes, and decoded within -0.6 to +0.9% of the empty
  start. Disk experts in the page cache hit 76.3% against 72.3% on the code request and decoded at
  11.1 against 10.7 tok/s, one run each. Seeding adds 0.5 s to startup with host experts and 1.6 s
  with disk experts.
- **The profile in `ninfer-serve`.** The server's warmup request fills an empty cache as well
  (1,809 of 1,821 slots), so over ten server starts at 2,048 and 16,384 tokens of context each
  request hit within 0.3 points as often from either start (the short ones 74.0 to 77.3% of their
  routed experts). Decode differed by -3.7 to +3.7% per request, and the profile delayed the first
  answer by 40 to 120 ms.
- **Memory.** The CLI peaked at 62.2 GiB of resident host memory with host experts and 0.8 GiB with
  disk experts.
- **Images.** With `--vision` the model read the title, the red circles and the shapes' order in
  `examples/cli/media/visual_chart.png` and described `natural_scene.png` correctly, with the same
  text from the resident and the streamed tower.

One run per row. These runs share no host, build or prompt with the IQ1_M rows above.

### Prefill tail, IQ1_M experts and Engram reads

November 5, 2026, the Coder build with experts in pinned RAM on two rented hosts:

- an RTX 4090 (450 W of its 480 W default, PCIe 4.0 x16), a 32-core EPYC 7542 (64 threads),
  503 GiB of RAM, driver 595.71.05, the sm_89 builds (CUDA 12.8) of this version and of
  `6d8dc3c1e`, the version before it;
- an RTX 3090 (350 W, PCIe 4.0 x16), a 64-core EPYC 7B12 in a container of 61 CPUs, 503 GB of RAM,
  driver 615.71.09, the sm_86 build of this version.

`ninfer-serve --expert-residency host --max-context 40960 --max-concurrency 1 --vision`, a fresh
server for each configuration and round, started after the Engram table was dropped from the page
cache; two rounds, a nonce in every prompt, 16 tokens of answer. "Options off" is
`--prefill-tail full --iq1m-experts vector --ngram-dedupe off --ngram-read-ahead off`; "IQ1_M
matrix" is the same with `--iq1m-experts matrix`. These builds took IQ1_M gate and up products to
the matrix kernel from 8.5 rows an expert; this version waits for more than 16
([IQ1_M routes per card](#iq1_m-routes-per-card)), which moves those products of the 282- to
356-token prompts and of the 29,009-token prompts' last chunk to the vector kernel. Prefill in
tok/s, the mean of the two rounds:

| prompt | RTX 4090, `6d8dc3c1e` | RTX 4090, options off | RTX 4090, defaults | RTX 3090, options off | RTX 3090, IQ1_M matrix | RTX 3090, defaults |
|---|---:|---:|---:|---:|---:|---:|
| 77 to 80 tokens | 61.3 | 59.1 | 60.0 | 55.0 | 55.6 | 54.6 |
| 352 to 356 tokens | 180.3 | 176.0 | 178.6 | 159.7 | 166.0 | 166.8 |
| an image and its question, 282 to 286 tokens | 119.4 | 116.0 | 116.4 | 113.0 | 114.0 | 114.3 |
| 5,851 to 5,855 tokens of C++ | 1,133 | 1,126 | 1,447 | 598 | 839 | 981 |
| 8,085 to 8,090 tokens of prose | 1,144 | 1,155 | 1,637 | 590 | 828 | 1,054 |
| 29,009 to 29,012 tokens of prose | 1,182 | 1,183 | 2,362 | 573 | 803 | 1,332 |

- **Long prompts.** On the RTX 4090 the defaults prefill 1.28, 1.43 and 2.00 times as fast as
  `6d8dc3c1e`, and with the options off within -0.6 to +1.0% of it. On the RTX 3090 the IQ1_M
  matrix kernel alone prefills 1.40 times as fast as the options off and the defaults 1.64 to 2.32
  times.
- **Short prompts** have fewer than 2,414 rows, so `exact` runs them whole, and the shortest take
  the vector kernel. On the RTX 3090 the 352- to 356-token prompts, which these builds ran on the
  matrix kernel, prefilled 4.0 to 4.5% faster, the two rounds of each configuration within 0.5%. On the RTX 4090
  the configurations' means differ by -3.6 to -1.0% from `6d8dc3c1e`, while two runs of one
  configuration differed by up to 8% (112.1 and 120.7 tok/s): no difference is resolved there.
- **Decode** after each prompt ran at 22.5 to 29.1 tok/s in every configuration on both cards.
- **Outputs.** On the RTX 4090, greedy with log probabilities, `exact` gave `full`'s tokens and log
  probabilities bit for bit, and with `--iq1m-experts vector` `6d8dc3c1e`'s, for prompts of 66,
  342, 1,607 and 6,143 tokens, a second turn after the last of them and two requests that share a
  system prompt of about 4,400 tokens. With the IQ1_M matrix kernel the greedy tokens of four of
  the seven left the vector kernel's at answer tokens 2 to 34; the 66-token prompt, too short for
  the matrix kernel, gave `6d8dc3c1e`'s tokens and log probabilities.
- **Perplexity.** Over 8,903 tokens of WikiText (40 KB, BF16 KV) this version scores 3.7456 with
  host and with disk experts and 3.7503 with `--iq1m-experts vector`, as `6d8dc3c1e` does.

The next measurements ran on the same RTX 4090 with a build of this change whose IQ1_M route
differed only for calls of at most 16 rows an expert:

- **More outputs.** `exact` gave `full`'s tokens and log probabilities bit for bit, and with
  `--iq1m-experts vector` `6d8dc3c1e`'s, for prompts of 6,048 to 31,752 tokens and a request stopped
  inside its prompt and sent again; `exact` also gave `full`'s with DSpark drafts and with two
  concurrent sequences.
- **A stopped prompt.** A 33,725-token request stopped 4 s into its prompt and sent again after
  another request prefilled all 33,725 tokens under `exact`, in 14.0 s (18.6 s with
  `--iq1m-experts vector`). `full` resumed from the state the stopped request left and prefilled
  17,341 tokens in 12.6 s (15.4 s with the vector kernel, 15.6 s with `6d8dc3c1e`).
- **Two sequences.** With `--max-concurrency 2` a 22,474-token prompt prefilled at 2,236 tok/s under
  `exact` against 1,501 under `full`, and at 2,482 against 1,581 beside a 6,147-token prompt,
  whose own rate rose from 645 to 720 tok/s.
- **The lossy tail.** On prompts of 22,470 to 31,752 tokens `--prefill-tail 1024` and `512`
  prefilled 3.1 to 10.2% faster than `exact`, one run each. Their greedy answers to two summary
  requests left `full`'s at answer tokens 15 to 34; three questions about a passphrase hidden at
  10, 50 and 90% of a 25,939-token document gave `full`'s answer tokens in both modes.

With `--ngram-io direct`, which reads the Engram rows past the page cache, the two Engram options
gave these prefill rates and reads from the table per request (two rounds, with a build of this
change that ordered a pass's rows with `std::sort`, 63 ms for a 16,384-token span's 786,432 rows on
this host's CPU against 13 ms for the radix sort that replaced it):

| prompt | both off | `--ngram-read-ahead off` | `--ngram-dedupe off` | defaults |
|---|---:|---:|---:|---:|
| an image and its question | 161.2 tok/s, 55.5 MiB | 161.9 tok/s | 161.1 tok/s | 161.4 tok/s, 5.7 MiB |
| 5,852 tokens of C++ | 1,376 tok/s, 1,132 MiB | 1,452 tok/s | 1,347 tok/s | 1,476 tok/s, 443 MiB |
| 8,085 tokens of prose | 1,632 tok/s, 1,565 MiB | 1,658 tok/s | 1,612 tok/s | 1,638 tok/s, 1,306 MiB |
| 29,010 tokens of prose | 2,172 tok/s, 5,609 MiB | 2,231 tok/s | 2,324 tok/s | 2,342 tok/s, 4,288 MiB |

Reading ahead acts between a prompt's spans of 16,384 tokens, so of the shorter prompts it reads
ahead only the few tokens after the last user turn's checkpoint. Through the page cache (the
default `--ngram-io buffered`), with the table dropped from it before each server started (two
rounds), reading each repeated row once cut the read calls by 87% for the image, 61% for the C++,
17% for the 8,085 tokens of prose and 24% for the 29,009 tokens, and changed the prompt time by
-5.2 to +0.1% (faster for five of the six prompts, where one server of `--ngram-dedupe off` ran
its three shortest prompts 3 to 12% slower than the other).

### IQ1_M routes per card

November 5, 2026, `ninfer_gguf_moe_matrix_bench --formats IQ1_M` on one rented card of each class
at its default power limit (the RTX 4090 at 450 W of its 480 W default): the sm_75, sm_86 and
sm_89 builds with CUDA 12.8 and the 120a build with CUDA 13.1. The benchmark times each projection
of a call alone, its activation's quantization included, over random routes to distinct experts,
one format per process. The table gives the vector kernel's time over the matrix kernel's at
DeepSeek-V4.1's shapes, six experts a token, by the call's routed rows per expert; above 1 the
matrix kernel is faster:

| card | experts | gate and up: 4 rows | 6 | 7 | 7.5 | 8.5 | down: 1 row | 4 | 5.5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RTX 2080 Ti | 384 | 0.73 | 0.83 | 0.89 | 0.93 | 1.06 | 0.82 | 0.88 | 0.97 | 1.03 |
| | 192 | 0.72 | 0.71 | 0.90 | 0.93 | 1.04 | 0.81 | 0.88 | 0.97 | 1.01 |
| RTX 3090 | 384 | 1.02 | 1.19 | 1.29 | 1.36 | 1.54 | 1.32 | 1.43 | 1.54 | 1.62 |
| | 192 | 1.02 | 1.15 | 1.29 | 1.33 | 1.49 | 1.31 | 1.41 | 1.53 | 1.59 |
| RTX 4090 | 384 | 0.82 | 0.93 | 1.03 | 1.06 | 1.20 | 1.15 | 1.21 | 1.32 | 1.39 |
| | 192 | 0.81 | 0.93 | 1.03 | 1.06 | 1.18 | 1.13 | 1.21 | 1.31 | 1.36 |
| RTX 5090 | 384 | 0.70 | 0.85 | 0.93 | 0.99 | 1.09 | 0.75 | 0.86 | 0.98 | 1.04 |
| | 192 | 0.67 | 0.80 | 0.87 | 0.91 | 1.03 | 0.74 | 0.84 | 0.97 | 1.01 |

In prompt steps the matrix kernel keeps less of that lead. On the RTX 4090 (the Coder build with
host experts, `ninfer` answering two tokens under Nsight Systems, lazy module loading, in a build
that took gate and up products to the matrix kernel from 8.5 rows), the IQ1_M products of a prompt
step took these times per call, the mean of its 16 gate-and-up and 28 down calls, in µs:

| prompt | rows an expert | gate and up: matrix | vector | vector/matrix | down: matrix | vector | vector/matrix |
|---|---:|---:|---:|---:|---:|---:|---:|
| 235 tokens | 7.3 | not run | 2,567 | | 1,443 | 1,880 | 1.30 |
| 278 tokens | 8.6 | 3,230 | 2,880 | 0.89 | 1,579 | 2,119 | 1.34 |
| 326 tokens | 10.1 | 3,499 | 3,246 | 0.93 | 1,712 | 2,388 | 1.39 |
| 388 tokens | 12.0 | 3,849 | 3,701 | 0.96 | 1,877 | 2,729 | 1.45 |
| 451 tokens | 14.0 | 4,204 | 4,203 | 1.00 | 2,067 | 3,083 | 1.49 |
| 514 tokens | 16.0 | 4,558 | 4,655 | 1.02 | 2,239 | 3,433 | 1.53 |
| 635 tokens | 19.7 | 4,048 | 5,553 | 1.37 | 1,997 | 4,098 | 2.05 |

- **The thresholds.** Gate and up products take the matrix kernel past 16 rows an expert, where
  its column tiles are 32 wide instead of 16. Down products take it from 6 rows, the highest
  crossing of the benchmark (the RTX 2080 Ti and 5090, between 5.5 and 6 rows). With one rule for
  every card the RTX 3090 keeps the vector kernel for gate and up products of 4 to 16 rows, where
  the benchmark has the matrix kernel ahead (1.02 at 4 rows to 1.54 at 8.5); the RTX 3090, 5090 and
  2080 Ti were not profiled in prompt steps.
- **The benchmark against prompt steps.** At 256 and 320 tokens the benchmark's vector/matrix
  ratios were 1.13 and 1.35 for gate and up products and 1.65 and 1.96 for down products, where the
  prompt steps of 278 and 326 tokens gave 0.89 and 0.93, 1.34 and 1.39. With 2048-token chunks the
  steps' IQ1_M, IQ2_XXS and IQ2_XS matrix kernels ran within -2 to +4% of the benchmark's times.
- **Module loading.** With `CUDA_MODULE_LOADING=EAGER` the 326-token step's matrix kernel took
  2,683 µs for gate and up and 1,323 µs for down, 23% less than with lazy loading; the 278-token
  step's took 3,197 and 1,561 µs, 1% less. In the benchmark the matrix kernel of a format timed
  after another ran up to 1.24 times as long ([one format per
  process](../bench/README.md#linear-op-benchmark)). The cause is not established; the thresholds
  follow the default lazy loading.
- **Few rows an expert.** The vector kernel reads each token's activation once for all of its
  experts; the matrix kernel gathers and quantizes a row for every routed pair and fills a tile of
  columns for every expert, most of them empty in a short call.
- **Qwen3.8-Flash-Next's shapes** (2,560 to 640, 512 experts, ten a token): the matrix kernel
  served gate and up faster from 0.3 rows an expert on the RTX 3090 (1.20) and from 0.6 on the
  RTX 4090 (1.01); on the RTX 2080 Ti and 5090 it was slower up to 5 rows (0.73 to 0.90) and
  faster from 7.5 (1.03 to 1.14); 256 experts gave the same crossings. The rule keeps them on the
  vector kernel up to 16 rows as at DeepSeek-V4.1's shapes: no published Flash-Next artifact has
  IQ1_M experts, so none of its prompt steps was profiled. Their down products (640 inputs, not a
  multiple of IQ1_M's 256-value blocks) have no matrix route.

## Numerical checks

- **Kernels.** `ninfer_deepseek_v41_kernels_test` qualifies the executor's own kernels against FP64
  oracles at the release's shapes: the hyper-connections and their Sinkhorn normalization, the
  latent and indexer projections' norms, rotations and roundings, the compressors, the index scores
  and both selections, the sparse attention with its sinks and splits, the output's inverse
  rotation, routing, the clamped SwiGLU, the expert combination, the Engram rows and gate, and the
  activation guard of the integer matrix kernels.
- **Against an FP64 forward.** `tools/reference/deepseek_v41.py` runs the model in FP64 over the
  GGUF's exactly decoded blocks and the Engram GGUF's rows, with the release's attention roundings.
  Over the 1,112 scored tokens of the WikiText slice above, its negative log-likelihood sums to
  1238.7 against 1246.3 for the build that added the model on October 25 (perplexity 3.046 against
  3.067, 0.7% higher in NInfer, whose matrix kernels take BF16 activations rounded to ggml's q8_1
  blocks, as llama.cpp's do). NInfer's
  greedy next token matched the reference's argmax at 21 of 22 positions spread over the slice (the
  exception has a top-two logit gap of 0.32) and at 6 of the 10 positions where the reference's top
  two logits are closest (gaps of 0.01 to 0.04). The FP64 forward took 2 h 53 min on the
  Threadripper 3960X. Since then the GPU's IQ1_S, IQ1_M and IQ2_XXS products round step by step as
  the [CPU share](#the-cpu-share) does, which changes the outputs; this check was not rerun.
- **The STQ product.** `ninfer_linear_stq_test` qualifies the STQ drafter's matrix product against
  an FP64 oracle over its own decode of the trellis words and sign vectors, at the drafter's shapes
  and 1 to 8 rows: `fc` (15,360 to 5,120), `wkv`, `wq_b` with BF16 output, `wo_b`, the shared
  expert's down product, the grouped `wo_a` (8 groups of 4,096 to 1,024) and the routed experts'
  gate and down products through expert ids, within 2·10⁻⁵ of the output's RMS (plus BF16's half
  step where the output is BF16); its decoder reproduces a PAIR1 vector from sushi's tests. The
  converter's tests convert a small STQ pack and compare each stored trellis, sign and vector with
  the pack's bytes.
- **The vision tower's kernels.** `ninfer_deepseek_v41_vision_kernels_test` qualifies them against
  FP64 oracles at the tower's widths: the residual with the next projection's RMS norm, the 2D
  rotation of Q and K (angles past 70 radians), the softmax, SwiGLU, the erf GELU with its bias,
  the aligner's unfold of whole and padded grids and the span's rows.
- **Images.** `ninfer_deepseek_frontend_test` checks the grid plans, the bicubic resize and padding
  of a synthetic picture and the patches and span of `examples/cli/media/natural_scene.png` against
  values the release's `image_processor.py` and Pillow wrote. `ninfer_deepseek_v41_vision_real_test`
  encodes that picture (55 × 37 patches, 13 × 19 aligner cells) with the resident and the streamed
  tower, whose rows are equal bit for bit, and compares them with the FP64 forward of
  `tools/reference/deepseek_v41_vision.py` over the mmproj's weights: on an RTX 4090 the worst
  row's cosine was 0.981 and the relative RMS error 2.6%, within the test's bounds of 0.98 and 3%;
  the tower rounds its matrices' inputs, Q, K, V and the attention probabilities to BF16 and keeps
  the residual in FP32. Served with `--vision`, the model read the CLI examples `image_chart`,
  `image_natural` and `multi_image_compare` correctly with either residency, with the same tokens.
