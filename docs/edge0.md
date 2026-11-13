# Edge0-145B and Edge0-120B

Edge0's models keep Qwen3.8-Flash-Next's skeleton (Gated DeltaNet, sparse attention,
hyper-connections, the n-gram embedding, the router, the shared expert and the MTP block) and
replace every text layer's 512 routed experts by latent experts: each expert belongs to one of 8
groups and computes in its group's latent width (1792 for Edge0-145B, 1280 for Edge0-120B), which a
group encoder and decoder project from and back to the model's 2560 ([the
mathematics](maintainer/qwen4-exp-model.md#17a-edge0s-latent-moe-text_configlatent_moe)). NInfer
runs them as a form of the Qwen3.8-Flash-Next family: the [Flash-Next guide](qwen3-8-flash-next.md)'s
options, placements and modes apply unless this page says otherwise.

The publisher releases each model as a folder that its engine (`edge0-strata`, a fork of Strata)
reads, in two expert builds:

| Release | Experts (gate/up, down) | Expert bytes | Encoder and decoder |
|---|---|---:|---|
| [Edge0-145B-NVFP4](https://huggingface.co/Edge0/Edge0-145B-NVFP4) | NVFP4, NVFP4, with per-expert FP32 multipliers | 47.6 GB | Q8_0 |
| [Edge0-145B-q3k](https://huggingface.co/Edge0/Edge0-145B-q3k) | Q3_K, IQ4_NL, with per-layer column scales | 40.1 GB | Q8_0 |
| [Edge0-120B-NVFP4](https://huggingface.co/Edge0/Edge0-120B-NVFP4) | NVFP4, NVFP4, with per-expert FP32 multipliers | 34.0 GB | Q4_0 |
| [Edge0-120B-q3k](https://huggingface.co/Edge0/Edge0-120B-q3k) | Q3_K, IQ4_NL, with per-layer column scales | 28.6 GB | Q4_0 |

Every folder also holds the skeleton as a llama.cpp GGUF (6.0 GB), the n-gram table in FP8
(`ple-fp8.gguf`, 51.2 GB, the same in all four), the MTP block and the publisher's tokenizer. The
folders hold no Vision tower: `edge0-strata` reads pictures with the base model's, the
`mmproj-Qwen3.8-Flash-Next-BF16.gguf` of
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
and the NInfer artifacts carry that tower. The models are under the Edge0 Community Model License
1.0 and the Qwen license of the base model; a converted artifact is a conversion under those
licenses.

NInfer artifacts of the four releases are published with the Vision tower and without the table:
[Edge0-145B-NVFP4](https://huggingface.co/WaveCut/Edge0-145B-NVFP4-NInfer-v3),
[Edge0-145B-q3k](https://huggingface.co/WaveCut/Edge0-145B-q3k-NInfer-v3),
[Edge0-120B-NVFP4](https://huggingface.co/WaveCut/Edge0-120B-NVFP4-NInfer-v3) and
[Edge0-120B-q3k](https://huggingface.co/WaveCut/Edge0-120B-q3k-NInfer-v3). The table is published
once, as [Edge0-ngram-table-FP8](https://huggingface.co/WaveCut/Edge0-ngram-table-FP8-NInfer-v3)
([downloads](recipes.md#edge0-145b-and-edge0-120b)). The commands below use their file names.

## Convert

`--model` needs the base model's configuration and tokenizer files (`config.json`,
`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`, and for
the Vision tower `preprocessor_config.json` and `video_preprocessor_config.json`) from
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next); the converter refuses a
folder whose vocabulary or merges differ from them. `--source pack=FOLDER` names the release folder,
which supplies the skeleton, the experts, the n-gram table and the MTP block; `--source vision=FILE`
names the Vision tower's mmproj.

```bash
# The n-gram table once, as an artifact of its own that every Edge0 model names by digest, with a
# hot-row profile for --ngram-residency ram-hot (the published table holds the IQ4_NL table's).
python3.11 -m tools.convert --model /path/to/Qwen3.8-Flash-Next \
  --recipe edge0_pack --source pack=/path/to/Edge0-120B-q3k \
  --resource ngram.hot=/path/to/flash-next.hot \
  --components ngram --device cpu --out models/Edge0-ngram-table-FP8-ninfer-v3.ninfer

# Each model without the table, with its MTP block and the base model's Vision tower.
python3.11 -m tools.convert --model /path/to/Qwen3.8-Flash-Next \
  --recipe edge0_pack --source pack=/path/to/Edge0-120B-q3k \
  --source vision=/path/to/mmproj-Qwen3.8-Flash-Next-BF16.gguf \
  --components text,vision,mtp --device cpu --name Edge0-120B-q3k \
  --out models/Edge0-120B-q3k-ninfer-v3.ninfer
```

`edge0_pack` stores the encoder and decoder build the release's launcher selects;
`edge0_pack_q8` and `edge0_pack_q4` choose `latent_dense_q8.bin` or `latent_dense_q4.bin`
instead. `--components text,ngram` writes one self-contained file. Every conversion reads the
51.2 GB table once to hash its rows, since the model records the digest of the table it reads.

Every expert row, every encoder and decoder row and the table's codes are stored byte for byte as
the release holds them. NVFP4 rows keep ggml's NVFP4 blocks (`gguf_nvfp4`, [GGUF block
formats](gguf.md)), and their record tails become the FP32 `expert_scales`; the Q3_K builds'
column scales become their reciprocals, one FP32 division each as `edge0-strata` computes them. The
table becomes `fp8_e4m3fn_row_fp16` rows of 160 codes with the release's global scale in every row
(exact in FP16). The MTP block's Q2_0 experts become `gguf_q2_0` blocks, its norms drop their stored
`1 + w`, and its BF16 hyper-connection matrices become Q8_0 rows, as in every Flash-Next GGUF
conversion. The Vision tower keeps the mmproj's BF16 tensors (0.9 GB), as for
[Flash-Next](qwen3-8-flash-next.md#convert).

## Run

```bash
M=models/Edge0-120B-q3k-ninfer-v3.ninfer
T=models/Edge0-ngram-table-FP8-ninfer-v3.ninfer

# One GPU with room for every expert (Edge0-120B-q3k on a 48 GB card).
./build/apps/ninfer-serve $M --ngram-table $T --max-context 32768

# The same with the Vision tower, for images and video.
./build/apps/ninfer-serve $M --ngram-table $T --vision --max-context 32768

# One GPU: the experts in pinned host memory, the most used of them cached on the GPU.
./build/apps/ninfer-serve $M --ngram-table $T --expert-residency host --max-context 32768

# One GPU and little RAM: the experts stay in the artifact's files and stream into a GPU cache.
./build/apps/ninfer-serve $M --ngram-table $T --expert-residency disk --max-context 32768

# Several GPUs dividing every layer: each GPU holds whole groups with their encoder, decoder and
# experts (Linux).
./build/apps/ninfer-serve models/Edge0-145B-q3k-ninfer-v3.ninfer --ngram-table $T \
  --devices 0,1 --tensor-parallel --max-context 32768
```

The encoder, the decoder, the labels and the scales stay in device memory under every placement:
3.74 GB for Edge0-145B (Q8_0) and 2.67 GB for Edge0-120B with Q8_0 projections, 1.42 GB with Q4_0.
`--vision` adds the Vision tower's 0.9 GB of GPU memory. Stored bytes before KV and workspace,
without the tower:

| Model | Device residency | Pinned host experts |
|---|---|---:|
| Edge0-145B-NVFP4 | about 58 GB | 47.6 GB |
| Edge0-145B-q3k | about 50 GB | 40.1 GB |
| Edge0-120B-NVFP4 | about 41 GB | 34.0 GB |
| Edge0-120B-q3k | about 36 GB | 28.6 GB |

The table is read 16 rows a token from its file, as for Flash-Next; `--ngram-residency ram` loads
all 51.2 GB, and `ram-hot` keeps the rows its embedded profile ranks first in 4 GiB of RAM
(`--ngram-ram-mib`). The published table holds the broad profile of Flash-Next's IQ4_NL table byte
for byte ([the n-gram rows](qwen3-8-flash-next.md#the-n-gram-rows)): the same hash addresses the
same rows. With `ram-hot`, Flash-Next Q2_0 bound to the FP8 table scores the first 40 KB of
WikiText (9,234 tokens) at perplexity 2.531971, as with `disk`. Flash-Next's 26.92 GiB IQ4_NL table, the same table at lower precision, works in its
place (`--ngram-table`, [another table of the same shape](qwen3-8-flash-next.md#another-table-of-the-same-shape))
at 0.6% to 1.2% higher perplexity ([measurements](#with-flash-nexts-table)).

What differs from Flash-Next:

- **Tensor parallelism** divides the groups, not the experts' rows: each GPU takes a run of whole
  groups with their encoder and decoder rows and the experts the labels assign to them. The groups
  differ in size (8 to 200 of a layer's 512 experts), so each layer cuts its runs where the experts
  before the cut come nearest each card's share of the split. Each part adds into the same
  fixed-point sums, so the routed experts' output equals one GPU's whatever the cut; the mixers
  and the shared expert divide and round per card as for Flash-Next. Tensor parallelism needs
  device experts, as for Flash-Next.
- **Host and disk experts** keep the CPU share: CPU threads compute their share of a call's missing
  experts from the GPU's quantized latent inputs, bit for bit as the GPU's vector kernel would, and
  the GPU runs the encoder and decoder of every token.
- **Prompt calls** past 2048 tokens of one call run in equal slices of the call's width, which take
  the unsliced call's kernels, so the slices store the same bits in one slice's workspace.
- **Expert profiles**: host experts fill the GPU cache at startup from Edge0's built-in routing
  profile, recorded on Edge0-120B-q3k over 48 chat, code, long-prompt and tool requests (all four
  releases share the router), not from Flash-Next's: a layer's 64 most routed experts in the two
  profiles have a third in common on average. `--expert-profile` replaces it with counts recorded
  by `--expert-profile-out`. Disk experts fill their cache as the layers route, as for Flash-Next.
- **Drafting**: MTP also by default with device experts and one request at a time
  ([below](#drafting)). The releases have no DFlash2 drafter.

`edge0-strata` applies the decoder to each routed expert's quantized output before the router
weights add them; NInfer applies it once per routed group to the weighted sum of that group's
experts. The two agree in exact arithmetic and differ by the rounding of that one quantization
point.

## Drafting

`--spec mtp` drafts with the release's MTP block and `--lookup-ngram` by context lookup, as for
Flash-Next. `--spec dspine --draft-model PATH` drafts with a [DSpine](maintainer/dspine.md) drafter
trained for the release through Flash-Next's DSpine path; Edge0 publishes none, and no DSpine
drafter has run on a latent MoE yet. Without `--spec`, an Edge0 model drafts three tokens a round
with `--draft-min-p 0.5` with host experts, as Flash-Next does, and also with device experts when
the engine serves one request at a time (`--max-concurrency 1`, the default): each layer's chain of
group encoders, latent experts and decoders makes a token's step cost enough that the verification
pays on prose as well ([measurements](#mtp)). With disk experts, or device experts and several
requests, it decodes without drafts unless `--spec mtp` asks for them.

## Measurements

One A40 (48 GB, 300 W, driver 595.91.07) in a RunPod container with a 7.65-CPU quota of a Xeon Gold
6342 and 50 GB of RAM, the CUDA 12.8 `sm_86` build, November 12-13, 2026. Disk experts read the
container's disk through the page cache. Each configuration ran once.

### Perplexity

The first 40 KB of `wikitext/00.txt` in `perplexity-1m` (9,234 tokens), INT8 KV. Every placement
that fits gives the same value:

| Release | Device experts | Host experts | Disk experts |
|---|---:|---:|---:|
| Edge0-145B-NVFP4 | does not fit | 3.067910 | 3.067910 |
| Edge0-145B-q3k | does not fit | 3.127031 | 3.127031 |
| Edge0-120B-NVFP4 | 4.795757 | 4.795757 | 4.795757 |
| Edge0-120B-q3k | 4.964565 | 4.964565 | 4.964565 |

Edge0-120B-q3k with device experts by `--kv-dtype`: `bf16` and `bf16p` 4.925191, `rk4v4` 4.952749,
`fp8` 4.953239, `int8` 4.964565, `k8v4` 4.966284, `nvfp4` 4.970178, `rk8v4` 4.994503. Two
tensor-parallel ranks on the one card (`--devices 0,0 --tensor-parallel`, each with whole groups
holding half of a layer's 512 experts) score 4.998407 with INT8 KV and 4.977586 with BF16 KV, 0.7%
and 1.1% above one rank, about as far as the KV formats' roundings move it; their greedy answer
follows one rank's for its first two sentences, then takes the other side of a near tie.

### With Flash-Next's table

The same text and KV with Flash-Next's IQ4_NL table in place of the FP8 table:

| Release | Experts | FP8 table | IQ4_NL table | Change |
|---|---|---:|---:|---:|
| Edge0-145B-NVFP4 | disk | 3.067910 | 3.094717 | +0.87% |
| Edge0-145B-q3k | disk | 3.127031 | 3.145938 | +0.60% |
| Edge0-120B-NVFP4 | device | 4.795757 | 4.850954 | +1.15% |
| Edge0-120B-q3k | device | 4.964565 | 5.017691 | +1.07% |

The IQ4_NL column comes from conversions of each release that named the IQ4_NL table; the published
Edge0-120B-q3k with `--ngram-table` and the IQ4_NL table gives the same 5.017691. The greedy sky
answers change the wording of their last sentence and stay correct; `image_chart` answers the same.
Edge0-120B-q3k decoded the sky answer without drafts at 54.1, 55.1 and 55.1 tokens per second with
the FP8 table and 54.2, 55.0 and 55.1 with the IQ4_NL table, alternating on one pod. The IQ4_NL
table fits in RAM where the FP8 table does not: with `--ngram-residency ram` it took 27,465 MiB and
the process peaked at 29.0 GB of the container's 50 GB.

### Decode

Greedy, one request, 256 tokens at most after a 33-token prompt (an answer of 99 to 105 tokens), in
tokens per second without drafts and with MTP (three drafts, `--draft-min-p 0.5`). Each release
gives the same text with and without drafts under every placement. Host experts start with the
cache filled from the built-in routing profile: Edge0-120B-q3k's routes then hit the cache 100.0%
of the time without drafts, against 67.3% and 43.5 tokens per second on a build without the
profile. Rows marked * ran on that build, without `--draft-min-p`, while a conversion used 3 of the
container's CPUs.

| Release | Experts | Expert cache | Without drafts | MTP | Drafts kept |
|---|---|---:|---:|---:|---:|
| Edge0-145B-NVFP4 | host | 33.3 GiB | 39.5 | 59.6 | 52.6% |
| Edge0-145B-NVFP4 | disk | 33.6 GiB | 14.1 | 29.0 | 52.6% |
| Edge0-145B-q3k | host | 33.3 GiB | 42.6 | 71.1 | 59.0% |
| Edge0-145B-q3k | disk | 33.6 GiB | 16.5 | 32.3 | 59.0% |
| Edge0-120B-NVFP4 * | device | | 53.4 | 91.6 | 66.7% |
| Edge0-120B-NVFP4 * | host | 31.7 GiB | 40.5 | 65.0 | 66.7% |
| Edge0-120B-NVFP4 * | disk | 35.7 GiB | 20.1 | 40.4 | 66.7% |
| Edge0-120B-q3k | host | 26.7 GiB | 49.2 | 73.9 | 52.7% |
| Edge0-120B-q3k | disk | 35.7 GiB | 23.3 | 38.1 | 52.7% |

Disk rows depend on how much of the artifact the container's page cache holds: the same
Edge0-145B-NVFP4 run, with the same cache hits and bytes read, decoded 20.2 tokens per second
without drafts right after the conversion had written the file and 14.1 in the row above.

### MTP

Edge0-120B-q3k with device experts, 256 greedy tokens after each prompt, decode in tokens per
second:

| Prompt | Without drafts | MTP | Drafts kept |
|---|---:|---:|---:|
| Code (a parser and its tests) | 54.6 | 100.7 | 76.0% |
| Prose (the water cycle) | 54.1 | 69.1 | 43.5% |
| A 5,415-token document and a summary | 53.7 | 79.8 | 59.2% |
| Prose, two tensor-parallel ranks on the card | 50.4 | 62.2 | 45.1% |

Eight requests at once through `ninfer-serve` (160 tokens each, about 850 in all) decoded 131.9 and
148.1 tokens per second in all without drafts and 111.3 and 129.5 with MTP, so the engine drafts by
default with device experts only for one request at a time.

### Vision

The four artifacts with the Vision tower, `--vision`, greedy, without drafts, 128 tokens at most,
the repository's CLI examples (`examples/cli/messages/`). Each answer is the expected one:

| Release | Experts | `image_chart` | `image_natural` |
|---|---|---|---|
| Edge0-145B-NVFP4 | disk | NIFER VISION 731；3；左侧 | the scene; mailbox 24; the sun on the right |
| Edge0-145B-q3k | disk | NIFER VISION 731；3；左侧 | the scene; mailbox 24; the sun on the right |
| Edge0-120B-NVFP4 | device | NIFER VISION 731；3；左侧 | the scene; mailbox 24; the sun on the right |
| Edge0-120B-q3k | device | NIFER VISION 731；3；左侧 | the scene; mailbox 24; the sun on the right |

Edge0-120B-q3k with device experts also answered `multi_image_compare` (2；3；五角星) and
`mixed_image_video` (NIFER-9, the picture's first word and the video's last number). Its
`image_chart` answer was the same with host experts and with the default MTP drafting, which kept
every draft and decoded at 100.8 tokens per second. The tower encoded a picture in 30 ms beside
device experts and in 125 to 146 ms beside disk experts. The tower changes no text: each artifact's
perplexity and greedy sky answer equal those of the text-only conversion.

### Serving

`ninfer-serve` with Edge0-120B-q3k and host experts, eight sequences, structured output on: eight
requests at once returned 845 tokens in 7.7 s, a `json_schema` request returned JSON its schema
admits, `logprobs` returned the top tokens of each position, thinking stayed on by default, and the
same 15,748-token prompt sent twice reused 15,741 cached tokens, answering in 0.54 s instead of
9.05 s.

### Against edge0-strata

The same eight greedy chat requests (128 tokens at most, the release's chat template) to
`edge0-strata` with its launch template for Edge0-120B-q3k (`--spec 4 --spec-min-p 0.5`, every
expert on the GPU) and to `ninfer-serve` with device experts. Two settings were needed to run
`edge0-strata` on this card and compare it: its all-resident verification path hung the first
decode window, as the fork's notes report on an ARM64 host, so `STRATA_VERIFY_ALL_RESIDENT=0` turned
it off; and its server adds the model card's `presence_penalty` 1.5 to greedy requests that do not
set one, so every request set it to 0.

- **Answers.** With MTP (three drafts, `--draft-min-p 0.5`), NInfer gave six of the eight answers
  token for token as `edge0-strata`, and the token log probabilities of the common prefixes differ
  by 0.02 or less on average per answer (0.21 at most). The two other answers part at near ties
  (`" from"` at -0.695 against `" of"` at -0.822 in `edge0-strata`). Without drafts NInfer's
  answers part from `edge0-strata`'s and from its own drafted ones at such ties as well: the
  verification windows round differently from one-token steps.
- **Speed.** `edge0-strata` decoded 44.2 to 76.9 tokens per second per request, NInfer with MTP
  76.1 to 118.8, the same request 1.41 to 1.74 times as fast, and NInfer without drafts 53.9 to
  54.7.

### Other GPUs

The five Op tests of the latent experts and their formats (`qwen4_exp_config`, `moe_latent_gguf`,
`moe_experts_gguf`, `linear_gguf`, `gguf_expert_cpu`) pass on each card below. A qualification
slice of each expert build (text layers 3 to 5 without the n-gram table, `tools.convert --layers
3..5`) scores the same 40 KB with INT8 KV and device, host and disk experts alike on each card (on
the A40, device experts only; the whole models above cover its placements):

| GPU | Build | Edge0-120B-q3k slice | Edge0-145B-NVFP4 slice |
|---|---|---:|---:|
| Tesla V100-SXM2 32 GB, 250 of 300 W | CUDA 12.8, `sm_70` | 12.707054 | 12.726173 |
| RTX 2080 Ti, 240 of 250 W | CUDA 13.4, `75;120a` | 12.700707 | 12.720653 |
| A40 | CUDA 12.8, `sm_86` | 12.700075 | 12.722557 |
| RTX 5090, 500 of 575 W | CUDA 13.4, `75;120a` | 12.700458 | 12.721601 |

The cards differ from the fourth or fifth significant digit on, by their kernels' roundings. The
`sm_80` and `sm_89` builds were compiled, not run. `compute-sanitizer` found no memory errors in
the 160 cases of `moe_latent_gguf` on the A40, and its racecheck found no shared-memory
hazards in them on the RTX 5090.
