# Launch recipes

The `ninfer-serve` configurations behind the [reference tables](performance.md), one section per
model. Each recipe is a [launch configuration file](cli.md#launch-configuration-files) in the
repository's `configs/` directory; `ninfer-serve --config configs/<file>.ini` runs it from the
directory that holds `models/`, where the [quick start](../README.md#quick-start) downloads the
artifacts. Each file names its model, card and options in its first lines. `ninfer-serve --help` is
the exact source for option spelling and defaults; [server options](serving.md#server-options) and
the [CLI options](cli.md) describe every flag, and [Supported GPUs](gpus.md) says what fits a given
amount of GPU memory.

Options on the command line override the file's: another window, port or draft count, another
artifact path, or `--spec none` without drafts.

```bash
ninfer-serve --config configs/bonsai2-27b-dflash2.ini --max-context 131072 --port 9000
ninfer-serve /data/bonsai.ninfer --config configs/bonsai2-27b-dflash2.ini
```

## Container or build

The image keeps the files in `/opt/ninfer/configs`. Its working directory is `/models`; `-w /`
makes a file's `models/` paths reach the mounted `/models`:

```bash
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 -w / \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/iamwavecut/ninfer-all:cuda13.4 serve --config /opt/ninfer/configs/bonsai2-27b-dflash2.ini
```

A build listens on 127.0.0.1:8080 unless `--host` and `--port` (or `host` and `port` in a file) say
otherwise; the container listens on 0.0.0.0:8080 unless they do.

## Flags in these recipes

Each entry of a file is a flag without its dashes (`kv-dtype = rk8v4` is `--kv-dtype rk8v4`, a switch
is `true`):

| Flag | What it sets |
|---|---|
| `--model-id` | the public model name in the API |
| `--max-context`, `--kv-capacity` | each sequence's context ceiling, and the shared KV capacity (`auto` takes what is free after the weights; omitted means `auto` with the default hybrid prefix cache and `--max-context` with the checkpoint catalog) |
| `--kv-dtype` | KV-cache storage: ten formats from `bf16` to `rk2v4-e8` and the lossless `bf16p`; `rk8v4` is the all-round choice, `rk4v4` the longest windows ([context and memory](cli.md#context-and-memory)) |
| `--gdn-state-fp16` | keep the Gated DeltaNet recurrent state in FP16 |
| `--rope-yarn` | past the native window, YaRN instead of plain RoPE |
| `--spec`, `--draft-tokens`, `--draft-model` | the speculative backend (`mtp`, `dflash`, `dflash2`, `dspine` with its drafter artifact) and the drafts of a round (1 to 15; 1 to 7 for Gemma 4) |
| `--adaptive-mtp` | pick each round's width from the drafts' measured survival: MTP, and on Qwen3.5-core models DFlash, DFlash2 and DSpine (`--draft-tokens` or half) |
| `--vision`, `--vision-residency`, `--vision-max-merged` | image and video input, where the tower lives (`overlay` borrows device memory per image), and the merged-token budget of one item |
| `--max-concurrency` | requests admitted at once, 1 to 8 |
| `--devices` | one pipeline stage per listed GPU (Linux) |
| `--expert-residency`, `--expert-cache-mib` | native Qwen MoE, Gemma 4, Qwen3.8-Flash-Next, Flint, DeepSeek-V4.1-Flash and Inkling 2: where the routed experts live and their persistent device cache budget ([expert residency](cli.md#common-options)) |
| `--ngram-table`, `--ngram-residency` | Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash: the n-gram (Engram) table's file and where its rows live |
| `--disk-kv-path`, `--disk-kv-gib`, `--disk-kv-restore` | the disk tier of the context cache |

## Ternary Bonsai 2 27B

The fastest single stream, MTP, the largest context a 24 GB card holds, and a disk tier.

```bash
# Fastest single stream: DFlash2 with five drafts over the full 262,144-token window of rk8v4.
ninfer-serve --config configs/bonsai2-27b-dflash2.ini

# MTP drafting through the proposal head: --spec mtp --draft-tokens 3.
ninfer-serve --config configs/bonsai2-27b-mtp.ini

# The largest context a 24 GB card holds: 958,464 tokens of rk4v4 with --rope-yarn.
ninfer-serve --config configs/bonsai2-27b-max-context.ini

# Adaptive MTP over 198,400 tokens and a 64 GiB disk tier that keeps evicted conversations
# (--disk-kv-path /var/cache/ninfer; give another path on the command line).
ninfer-serve --config configs/bonsai2-27b-disk-tier.ini
```

- **Draft count.** Five drafts are the all-round choice. Seven are faster on short answers; after
  long documents three to seven win ([draft length](performance/reference-2026-09.md#draft-length)).
- **Images.** `--vision --vision-residency overlay --vision-max-merged 12288` adds them. The encode
  borrows the drafter's memory, so the whole window still fits a 24 GB card.
- **Past 958,464 tokens.** The RTX 5090 and the PRO 6000 hold the 1,048,576-token maximum with
  `rk4v4`, DFlash2 or MTP included; the RTX 5090 holds 974,848 with `rk8v4`. Filled to 1,048,576
  tokens, the model misses the code at 90% (about 943K); up to about 880K it found every code on
  every card.

## Ternary Bonsai 2.5 27B

PQ2_0 with MTP, and the largest context a 24 GB card holds with PTQ1_0.

```bash
# The fastest single stream: PQ2_0 with MTP and three drafts over 262,144 tokens of rk8v4.
ninfer-serve --config configs/bonsai2.5-27b-mtp.ini

# The largest context a 24 GB card holds: PTQ1_0, 1,019,904 tokens of rk4v4 with --rope-yarn.
ninfer-serve --config configs/bonsai2.5-27b-max-context.ini
```

- **Which artifact.** PTQ1_0 decodes one request faster without a drafter on every measured card
  and holds the longest windows; PQ2_0 is faster with MTP, with several requests and on prompts
  ([measurements](performance/bonsai-2-5-2026-11.md)).
- **Past 1,019,904 tokens.** 1,019,904 is the RTX 4090's window; the RTX 3090 holds 1,032,192, and
  the RTX 5090 and the PRO 6000 the 1,048,576-token maximum with `rk4v4` and either artifact, MTP
  included.

## Qwen3.8-27B

The official `groupwise-int` artifact and the GSQ-RCO IQ3_S release.

```bash
# The official artifact on a 24 GB card: DFlash2 with five drafts over 192,512 tokens of rk4v4.
ninfer-serve --config configs/qwen3.8-27b-dflash2.ini

# GSQ-RCO IQ3_S: MTP over 176,128 tokens of rk8v4.
ninfer-serve --config configs/qwen3.8-27b-iq3s-mtp.ini
```

- **The official artifact with `rk8v4`.** The same speculation fits 131,072 tokens on an RTX 4090.
  An RTX 5090 takes the full 262,144 with either KV format.
- **More context with a drafter.** `--full-draft-head --ngram-draft-tokens 0` drafts through the full
  output head without copy drafting and gives back about 0.8 GiB with DFlash2: 167,936 tokens of
  `rk8v4` on an RTX 4090 ([largest context](performance.md#largest-context)).
- **IQ3_S.** It holds 413,696 tokens of `rk8v4` with MTP 3 on an RTX 4090, drafts with
  `--spec dflash2 --draft-tokens 5` as well, and `--vision` adds images.

## Qwen3.8-27B fine-tunes

Huihui abliterated, HauhauCS Aggressive and MXFP8-CRACK. All three have the official Qwen3.8-27B identity and size, so the Qwen3.8-27B recipes above serve
them too; CRACK has no DFlash2 adapter. Their cards' configurations:

```bash
# Huihui abliterated on one RTX 3090: 198,400 tokens of rk8v4 with MTP and Vision
# (--vision --vision-residency overlay --vision-max-merged 12288).
ninfer-serve --config configs/qwen3.8-27b-abliterated.ini

# HauhauCS Aggressive: DFlash2 with seven drafts over 32,768 tokens.
ninfer-serve --config configs/qwen3.8-27b-hauhaucs.ini

# MXFP8-CRACK: MTP over 32,768 tokens.
ninfer-serve --config configs/qwen3.8-27b-crack.ini
```

## Qwen3.8-Flash-Next

Every expert on the GPUs, in pinned host memory, or on disk.

```bash
hf download WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3 \
  Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-ninfer-v3.ninfer --local-dir models
hf download WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3 \
  Qwen3.8-Flash-Next-ngram-table-IQ4_NL-ninfer-v3.ninfer --local-dir models

# Every expert on one RTX PRO 6000 (96 GB), 32,768 tokens; the table's rows read from its file.
ninfer-serve --config configs/qwen3.8-flash-next-device.ini

# Every expert on two GPUs, one pipeline stage each (Linux): 24 GB cards for Q2_0, 32 GB for IQ3_S.
ninfer-serve --config configs/qwen3.8-flash-next-two-gpus.ini

# One 24 GB GPU: the experts in pinned host memory, the most used of them cached on the GPU.
ninfer-serve --config configs/qwen3.8-flash-next-host.ini

# One 24 GB GPU and little RAM: the experts stay in the file and stream into a GPU cache.
ninfer-serve --config configs/qwen3.8-flash-next-disk.ini

# The container, experts in host memory.
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 -w / \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/iamwavecut/ninfer-all:cuda13.4 serve --config /opt/ninfer/configs/qwen3.8-flash-next-host.ini
```

- **Other releases.** IQ3_S and the Coder IQ1_M build take the same files with their own artifact on
  the command line (`ninfer-serve models/<file>.ninfer --config ...`) and the same table. The
  [performance page](performance.md#qwen38-flash-next) shows the placements measured for each.
- **Memory.** Host experts pin 34 GB of RAM for Q2_0, 50 GB for IQ3_S and 25 GB for the Coder
  build. Disk experts need under 1 GB of RAM; the page cache does the rest. The GPU expert cache
  takes what is free after startup, or `--expert-cache-mib`.
- **The n-gram table.** Its rows are read from the file, 16 per token, unless
  `--ngram-residency ram` loads all 28.8 GB or `ram-hot` keeps the rows a profile ranks first in a
  RAM budget ([the n-gram rows](qwen3-8-flash-next.md#the-n-gram-rows)). A model started
  without its table is refused; `--no-ngram-table` overrides that, an experimental mode with no
  practical use.
- **Drafting.** `--spec mtp --draft-tokens 3` drafts with the MTP block the published artifacts
  carry.
- **Images and video.** `--vision` adds the Vision tower (0.9 GB on the GPU).

## Edge0-145B and Edge0-120B

Converted from Edge0's release folders ([convert](edge0.md#convert)) and published: each model
with the base model's Vision tower and without its n-gram table, and the table once for all four.

```bash
hf download WaveCut/Edge0-ngram-table-FP8-NInfer-v3 Edge0-ngram-table-FP8-ninfer-v3.ninfer --local-dir models
hf download WaveCut/Edge0-120B-q3k-NInfer-v3 Edge0-120B-q3k-ninfer-v3.ninfer --local-dir models
hf download WaveCut/Edge0-145B-q3k-NInfer-v3 Edge0-145B-q3k-ninfer-v3.ninfer --local-dir models
hf download WaveCut/Edge0-145B-NVFP4-NInfer-v3 Edge0-145B-NVFP4-ninfer-v3.ninfer --local-dir models

# Every expert on one 48 GB GPU, 32,768 tokens; one request at a time drafts with the MTP block.
ninfer-serve --config configs/edge0-120b-q3k-device.ini

# One GPU: Edge0-145B-q3k's experts in pinned host memory, the most used of them cached on the GPU.
ninfer-serve --config configs/edge0-145b-q3k-host.ini

# One GPU and little RAM: Edge0-145B-NVFP4's experts stay in the file and stream into a GPU cache.
ninfer-serve --config configs/edge0-145b-nvfp4-disk.ini
```

- **Other releases.** Each release runs with any of the three files and its own artifact on the
  command line (`ninfer-serve models/<file>.ninfer --config ...`); all four share the table.
  [WaveCut/Edge0-120B-NVFP4-NInfer-v3](https://huggingface.co/WaveCut/Edge0-120B-NVFP4-NInfer-v3)
  is the fourth.
- **Memory.** Host experts pin 40.1 GB of RAM for Edge0-145B-q3k, 47.6 GB for -NVFP4, 28.6 GB for
  Edge0-120B-q3k and 34.0 GB for -NVFP4; device experts need about 36 GB of GPU memory for
  Edge0-120B-q3k and 41 GB for -NVFP4 before KV ([run](edge0.md#run)).
- **Vision.** `--vision` (or `vision = true` in a configuration file) adds the Vision tower for
  images and video, 0.9 GB of GPU memory ([measurements](edge0.md#vision)).
- **Drafting.** Host experts, and device experts serving one request at a time, draft three tokens
  a round with the MTP block; with disk experts `--spec mtp --draft-tokens 3` asks for it. The
  [measurements](edge0.md#measurements) on an A40 show what each placement decodes with and without
  drafts.

## Gemma 4 26B-A4B

Converted from Google's QAT GGUF release.

```bash
# The release's GGUF and its Hugging Face text files; one conversion writes the artifact.
mkdir -p models/gemma4/hf
for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json \
         processor_config.json; do
  curl -fsSL -o models/gemma4/hf/$f https://huggingface.co/google/gemma-4-26B-A4B-it/resolve/main/$f
done
hf download google/gemma-4-26B-A4B-it-qat-q4_0-gguf gemma-4-26B_q4_0-it.gguf gemma-4-26B-it-mmproj.gguf \
  --local-dir models/gemma4
# With the release's QAT assistant (multi-token prediction) as the artifact's `mtp` component and
# its vision tower (the mmproj) as the `vision` component.
hf download unsloth/gemma-4-26B-A4B-it-qat-GGUF mtp-gemma-4-26B-A4B-it.gguf --local-dir models/gemma4
python -m tools.convert --model models/gemma4/hf --recipe gemma4_gguf --components text,mtp,vision \
  --source gguf=models/gemma4/gemma-4-26B_q4_0-it.gguf \
  --source mtp=models/gemma4/mtp-gemma-4-26B-A4B-it.gguf \
  --source vision=models/gemma4/gemma-4-26B-it-mmproj.gguf --name Gemma-4-26B-A4B-it-QAT-Q4_0-MTP \
  --out models/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-ninfer-v3.ninfer --device cpu

# Four concurrent requests over 40K-token contexts, two drafts a round, images and video; thinking
# stays off unless a request asks.
ninfer-serve --config configs/gemma4-26b-a4b.ini
```

## Gemma 4 31B

The dense 31B from the same QAT release ([Gemma 4 31B](gemma4.md#gemma-4-31b)); its 16.4 GiB of
weights need a 24 GB card and run on one GPU.

```bash
hf download WaveCut/gemma-4-31B-it-QAT-Q4_0-MTP-NInfer-v3 gemma-4-31B-it-QAT-Q4_0-MTP-ninfer-v3.ninfer \
  --local-dir models

# One request over a 56K-token context, two drafts a round, images and video (59,392 tokens
# is the longest with both on an RTX 3090).
ninfer-serve --config configs/gemma4-31b.ini
```

## Flint

Unsloth's UD-Q4_K_XL GGUF as one artifact ([Flint](flint.md)); 23.4 GiB of weights fit a 32 GB card.

```bash
hf download WaveCut/Flint-UD-Q4_K_XL-NInfer-v3 flint-UD-Q4_K_XL-ninfer-v3.ninfer --local-dir models

# Every expert on a 32 GB GPU: about 25.5 GiB for one 32K-token sequence.
ninfer-serve --config configs/flint-device.ini

# A 16 or 24 GB GPU: the experts in pinned host memory (21.5 GiB), the most used cached on the GPU,
# 65,536 tokens.
ninfer-serve --config configs/flint-host.ini
```

## DeepSeek-V4.1-Flash

The model and its Engram table are two artifacts; the routed experts (106.7 GiB) never fit one card,
so they live in pinned host memory or stay in the file ([DeepSeek-V4.1-Flash](deepseek-v41.md)).

```bash
hf download WaveCut/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-NInfer-v3 \
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-ninfer-v3.ninfer --local-dir models
hf download WaveCut/DeepSeek-V4.1-Flash-Engram-IQ4_NL-NInfer-v3 \
  DeepSeek-V4.1-Flash-Engram-IQ4_NL-ninfer-v3.ninfer --local-dir models

# One GPU and room to pin 106.7 GiB of host memory: the experts in RAM, the most used cached on the
# GPU, two concurrent requests.
ninfer-serve --config configs/deepseek-v4.1-flash-host.ini

# Less RAM: the experts stay in the file and reach the GPU through the page cache.
ninfer-serve --config configs/deepseek-v4.1-flash-disk.ini
```

ISTA-DASLab's expert-pruned Coder build reads the same table; its routed experts take 61.5 GiB:

```bash
hf download WaveCut/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-NInfer-v3 \
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-ninfer-v3.ninfer --local-dir models
ninfer-serve --config configs/deepseek-v4.1-flash-coder-host.ini
```

## Inkling 2 Mini

Unsloth's UD-IQ4_XS GGUF with the release's vision tower, its DSpark drafter and incoai's DFlash2
drafter as one artifact ([Inkling 2 Mini](inkling2.md)); 15.0 GiB of weights with DSpark, which
these configs draft with.

```bash
hf download WaveCut/Inkling-2-Mini-UD-IQ4_XS-NInfer-v3 Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer \
  --local-dir models

# A 24 GB GPU: one request over a 64K-token context with images, drafted by DSpark; an RTX 3090
# keeps 3.1 GiB free and decodes 137 tok/s.
ninfer-serve --config configs/inkling2-mini.ini

# Less GPU memory: the experts in 13.1 GiB of pinned host memory, the most used cached on the GPU,
# a 32K-token context; on an RTX 3090 the cache holds all of them and decodes 123 tok/s.
ninfer-serve --config configs/inkling2-mini-host.ini

# 8 GB, with a 12K-token context: the CLI with these settings on an RTX 5060 cached 2.2 GiB of
# experts and decoded the essay at 33.6 tok/s, 8.8 with --expert-residency disk.
ninfer-serve --config configs/inkling2-mini-host.ini --max-context 12288
```

IsValorum's APEX-I MiniPlus V2.1 (12.8 GiB, IQ3 experts, its Q8_0 vision tower) takes the same
configs with its path on the command line, which wins over the file's `model`:

```bash
hf download WaveCut/Inkling-2-Mini-APEX-I-MiniPlus-V2.1-NInfer-v3 \
  Inkling-2-Mini-APEX-I-MiniPlus-V2.1-ninfer-v3.ninfer --local-dir models

# 16 GB: every expert on the GPU, images and DSpark over an 8K-token context, 14.7 GiB planned
# (no 16 GB card has run it).
ninfer-serve models/Inkling-2-Mini-APEX-I-MiniPlus-V2.1-ninfer-v3.ninfer \
  --config configs/inkling2-mini.ini --max-context 8192

# Less: the experts in 10.9 GiB of pinned host memory.
ninfer-serve models/Inkling-2-Mini-APEX-I-MiniPlus-V2.1-ninfer-v3.ninfer \
  --config configs/inkling2-mini-host.ini
```

## Qwen3.6-35B-A3B NVFP4

RTX 50 series and RTX PRO 6000 only.

```bash
# MTP and images over 262,144 tokens of rk8v4.
ninfer-serve --config configs/qwen3.6-35b-a3b-nvfp4.ini
```

The NVFP4 experts prefill through W4A4 on Blackwell's FP4 tensor cores, so this needs a `120a`
build (the image has one); `sm_86` and `sm_89` builds refuse the file. On an RTX 5090 it starts in
17 s and leaves 7.4 GiB of the card free.

## Cards of 8 to 12 GB

The windows the [small cards](performance/small-cards-2026-10.md) page measured, checked again in
November 2026: the server started with `--max-concurrency 1` and answered a request. These three
cards carry built-in device profiles; another card of this size calibrates once at its first start,
in one to three minutes.

```bash
# 8 GB (RTX 3060 Ti): Ternary Bonsai 2 with 24,576 tokens of rk4v4 and no drafter. MTP does not fit.
ninfer-serve --config configs/bonsai2-27b-8gb.ini

# 11 GB (RTX 2080 Ti): 139,264 tokens of rk4v4 with MTP. Without drafting the card holds 196,608:
# --spec none --max-context 196608 --kv-capacity 196608.
ninfer-serve --config configs/bonsai2-27b-11gb.ini

# 12 GB (RTX 3060): 266,240 tokens of rk4v4.
ninfer-serve --config configs/bonsai2-27b-12gb.ini

# 12 GB: the GSQ-RCO IQ3_S release with 8,192 tokens of rk4v4 and no drafter.
ninfer-serve --config configs/qwen3.8-27b-iq3s-12gb.ini
```

- **Qwen3.8-Flash-Next** runs on 8 and 12 GB with `configs/qwen3.8-flash-next-host.ini` or
  `qwen3.8-flash-next-disk.ini` (32,768 tokens of context, no other change). It keeps 2.83 GiB of dense weights on
  the GPU and fills the rest of the card with an expert cache.
- **Gemma 4** needs 16 GB: its 13.4 GiB of weights do not fit 12 GB.
- **Output heads.** `--lm-head-q4` and `--embedding-q4` do not apply to Ternary Bonsai 2, whose
  head is not stored as row-split Q8_G32.

## A card without a built-in profile

```bash
ninfer-calibrate --print > my-gpu.json
```

The engine measures it by itself at first start. Running it by hand refreshes the profile after a
driver or clock change. See [device profiles](device-profiles.md).
