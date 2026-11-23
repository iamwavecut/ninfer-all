# NInfer-all

A C++/CUDA inference engine for the Qwen3.5 model family (Qwen3.6 and Qwen3.8 27B, Qwen3.6-35B-A3B,
their fine-tunes and Ternary Bonsai 2), Qwen3.8-Flash-Next and Edge0's latent-MoE models built on it,
Gemma 4 26B-A4B and 31B, Recursive's Flint, DeepSeek-V4.1-Flash and Thinking Machines' looped
Inkling 2 Mini, on NVIDIA GPUs from Volta to consumer Blackwell, from 8 GB of memory. It serves the
OpenAI, Anthropic and llama.cpp APIs from one GPU, or from up to eight as pipeline stages on Linux. The same chat model
answers [typed decisions](docs/serving.md#typed-decisions): choice, yes/no and score questions about
a state, each option with its probability, through TypeSafe's System One and OpenAI's Decisions APIs.

**[What's New](#whats-new)** · [Quick start](#quick-start) · [Supported GPUs](#supported-gpus) · [Supported Models](#supported-models) ·
[Highlights](#highlights) · [Features](#features) · [Docker](#docker) · [Building](#building) ·
[Documentation](docs/README.md)

NInfer-all continues [NInfer](https://github.com/Neroued/ninfer) through the NInfer-3090 line and
carries the work of several forks; [CONTRIBUTORS.md](CONTRIBUTORS.md) credits each of them.

> [!NOTE]
> **Support GPU testing.** Choose a custom amount on [GitHub Sponsors](https://github.com/sponsors/iamwavecut)
> to help cover GPU rentals for correctness checks, kernel tuning, and benchmarks.
> This is a voluntary contribution without sponsor benefits.
> For compute credits, GPU access, or individual sponsorship terms,
> [contact the maintainer](https://selit.ski/#contact) directly.

## What's New

<!-- Novelties only: a new model, family, artifact, platform, option, endpoint, format or mode, or a measured speedup of a path that worked. No fixes of any kind (crashes, wrong output, failed starts, missed cases, regressions and the speedups that win them back, the options and defaults they bring), no guards, diagnostics, internal, CI, build, test or documentation changes, and no chronicle: a date without a novelty gets no line. One line per date, a few words per item, each linked to the document with its details; no descriptions, mechanisms, measurements or hardware. Full rule: CONTRIBUTING.md, "Documentation describes the current state". -->

- **2026-11-23:** Inkling 2 [decoding faster](docs/inkling2.md#what-runs), [decoding faster after long prompts](docs/inkling2.md#long-contexts) and [prefilling long prompts faster](docs/inkling2.md#long-contexts); Gemma 4 and Flint decoding faster on Blackwell cards ([Gemma 4](docs/gemma4.md#what-runs), [Flint](docs/flint.md#what-runs)); Flint [decoding faster with `int8`, `rk8v4` and `rk4v4` KV](docs/flint.md#kv-cache); Flash-Next [prefilling long prompts faster](docs/qwen3-8-flash-next.md#execution); faster prompt chunks with [`cp.async` tiles](docs/device-profiles.md#what-is-calibrated) for NVFP4 weights; vLLM's [`bad_words`](docs/serving.md#banned-words), strings a reply never writes, and the CLI's [`--bad-word`](docs/cli.md); [`--no-structured-output-instruction`](docs/serving.md#structured-output), which keeps a constrained request's messages as the caller sent them.
- **2026-11-22:** [POST /shutdown](docs/serving.md#stop-the-server), the HTTP form of the graceful stop, registered by [`--admin-token`](docs/serving.md#server-options); [POST /v1/answer_now](docs/serving.md#answer-now), which ends a running reply's thinking; Inkling 2's [MXFP8 KV cache](docs/inkling2.md#kv-cache), its default (`--kv-dtype mxfp8`), and its [`int8`, `rk8v4` and `rk4v4` KV formats](docs/inkling2.md#kv-cache); Flint's [`int8`, `rk8v4` and `rk4v4` KV formats](docs/flint.md#kv-cache); for Gemma 4, Flint, DeepSeek-V4.1, Inkling 2 and Flash-Next a [disk tier, prefix cache file and session slots](docs/serving.md#context-cache-of-the-other-families) (`--disk-kv-path`, `--prefix-cache-file`, `--slot-save-path`), and a checkpoint inside a long message, so a document followed by another question resumes after the document; [thinking budgets](docs/serving.md#openai-chat-completions) and `--reasoning-loop conclude` for Inkling 2, Flint and Gemma 4; Inkling 2 [prefilling and decoding faster at long contexts](docs/inkling2.md#long-contexts).
- **2026-11-21:** [Inkling 2 Mini](docs/inkling2.md), the first looped transformer, from Unsloth's UD-IQ4_XS GGUF with the release's vision tower and DSpark drafter, [published](#supported-models); its [Q8_0 vision tower](docs/inkling2.md#converting) from a llama.cpp mmproj, and IsValorum's [APEX-I MiniPlus](docs/inkling2.md#apex-i-miniplus), [published](#supported-models); Unsloth's [UD-IQ1_M, UD-Q2_K_XL and UD-Q3_K_XL](docs/inkling2.md#smaller-quantizations) (8.2 to 13.4 GiB), [published](#supported-models); Unsloth's [UD-Q8_K_XL](docs/inkling2.md#ud-q8_k_xl) with its BF16 attention, head and experts (22.1 GiB), [published](#supported-models).
- **2026-11-20:** Faster GGUF prompt chunks with [`cp.async` tiles](docs/device-profiles.md#what-is-calibrated) for Q4_0, Q5_0, Q8_0, IQ4_NL, Q5_K, Q2_0, IQ4_XS and IQ1_M weights; [device profiles](docs/gpus.md) for nine more GPUs.
- **2026-11-19:** [Gemma 4 31B](docs/gemma4.md#gemma-4-31b) with its MTP assistant and vision tower, [published](#supported-models); [Basalt Flash-Next pack and serving-JSON conversion](docs/basalt-conversion.md), [Q8PLE n-gram tables](docs/qwen3-8-flash-next.md#q8ple-tables) and their [conversion from NVIDIA's FP8 PLE](docs/flash-next-native-dflash2.md#q8ple-directly-from-nvidia-fp8) with a [source-preserving Flash-Next recipe](docs/flash-next-native-dflash2.md), and [native FP16 KV](docs/qwen3-8-flash-next.md#native-fp16-kv) for Flash-Next and Edge0 (`--kv-dtype fp16`).
- **2026-11-16:** [System One image boxes and image URLs](docs/serving.md#image-boxes), [`--media-urls`](docs/serving.md#media-urls) that sets the size of a fetched media URL (URLs are refused without it), and Qwen3.8-27B GGUFs with a [condensed vocabulary](docs/weight-conversion.md#a-condensed-vocabulary).
- **2026-11-13:** [Edge0-145B and Edge0-120B](docs/edge0.md), latent-MoE models on the Flash-Next skeleton, converted from Edge0's release folders and [published](#supported-models) for all four releases with the base model's Vision tower, ggml's [NVFP4 blocks](docs/gguf.md), Flash-Next and Edge0 models reading [either family's n-gram table](docs/qwen3-8-flash-next.md#another-table-of-the-same-shape), [DSpine drafting](docs/cli.md#speculative-decoding) in every family from a drafter artifact of its own (`--spec dspine --draft-model`) with [Tencent's Qwen3.8-27B drafter converted](https://huggingface.co/WaveCut/Qwen3.8-27B-DSpine-b16-NInfer-v3), [`--adaptive-mtp` for DFlash, DFlash2 and DSpine](docs/serving.md#adaptive-mtp) on Qwen3.5-core models, and DeepSeek-V4.1 [hot Engram rows in RAM](docs/deepseek-v41.md#the-hot-row-profile) (`--ngram-residency ram-hot`), with a hot-row profile in the published Engram table and in [Edge0's FP8 n-gram table](docs/edge0.md).
- **2026-11-12:** [Launch configuration files](docs/cli.md#launch-configuration-files) (`--config`), with [one for each launch recipe](docs/recipes.md) in `configs/`.
- **2026-11-11:** [Faster prefill](docs/maintainer/qwen3_5-model.md#prefill-decode-and-mtp) and [request preparation](docs/maintainer/qwen3_5-model.md#request-preparation) in every family, disk experts read [from a second drive or straight into GPU memory](docs/expert-disk-reads.md) (`--expert-mirror`, `--expert-direct`), and [Gumbel-max sampling with drafts drawn the same way](docs/cli.md#speculative-decoding) (`--gumbel-sampling`).
- **2026-11-10:** [DFlash2 drafting for Qwen3.6-35B-A3B](docs/cli.md#speculative-decoding), Flash-Next artifacts of [Unsloth's Dynamic v3.1 GGUFs](docs/qwen3-8-flash-next.md#convert), Flash-Next prompts past 8,192 tokens [prefilling 1.1-1.4 times as fast](docs/qwen3-8-flash-next.md#expert-placements) with host or disk experts, Qwen3.5-core prompts that the hybrid cache taps [prefilling in one pass](docs/maintainer/hybrid-prefix-cache-spec.md#72-in-pass-exact-taps) (a 213-token first prompt 11% sooner), the hybrid cache over pipeline stages and tensor parallelism with [`--use-alt-prefix-caching`](docs/serving.md#server-options), DeepSeek-V4.1 [attention caches in the release's FP8/FP4 codes](docs/deepseek-v41.md#what-runs), DSpark drafting with host or disk experts [keeping 64 drafter experts a block on the GPU](docs/deepseek-v41.md#the-drafters-experts), 1.12 times as fast on an RTX 4090 (`--draft-experts`), Flash-Next with device experts [decoding 1.10 times as fast](docs/qwen3-8-flash-next.md#execution) (1.12 with MTP), and context lookup that [keeps the occurrence it copies](docs/flint.md#measurements), 1.06-1.40 times as fast on answers that copy one of two versions of a file (`--lookup-ngram`), and [`--kv-dtype bf16p`](docs/cli.md#context-and-memory), BF16's KV bits in 23% fewer bytes.
- **2026-11-07:** Flash-Next [media prompts in the context cache](docs/qwen3-8-flash-next.md#media-conversations-on-one-rtx-4090) and [MTP drafts for media requests](docs/qwen3-8-flash-next.md#mtp-speculative-decoding).
- **2026-11-06:** Faster Q4_K [prompts](docs/device-profiles.md#what-is-calibrated) for dense layers and routed experts.
- **2026-11-05:** Faster DeepSeek-V4.1 [prefill tail](docs/deepseek-v41.md#the-prefill-tail), [IQ1_M expert prompts](docs/deepseek-v41.md#iq1_m-experts-in-prompt-chunks) and [Engram reads](docs/deepseek-v41.md#the-engram-table).
- **2026-11-04:** Flash-Next [DFlash2 drafting](docs/qwen3-8-flash-next.md#dflash2-speculative-decoding), faster Flash-Next [tensor parallelism](docs/maintainer/pipeline-stages.md#tensor-parallelism-of-qwen38-flash-next) and [concurrent requests with host experts](docs/qwen3-8-flash-next.md#sequences-kv-and-the-context-cache), [BF16 and 8-bit tensor-parallel prompt partials](docs/maintainer/pipeline-stages.md#tensor-parallelism), faster [expert parallelism](docs/maintainer/pipeline-stages.md#expert-parallelism-of-the-other-moe-families), and DeepSeek-V4.1-Flash [conversion of ISTA-DASLab's llama.cpp-layout GGUFs and Coder build](docs/deepseek-v41.md#converting) with a [published Coder artifact](docs/deepseek-v41.md#running).
- **2026-11-03:** Faster [tensor-parallel decode and prefill](docs/maintainer/pipeline-stages.md#tensor-parallelism) for Qwen3.5-core models and llama.cpp [control vectors and direction LoRA adapters](docs/serving.md#lora-adapters-and-control-vectors).
- **2026-11-02:** [Tensor parallelism](docs/maintainer/pipeline-stages.md#tensor-parallelism) selects exchange routes by size from startup timings, including peer writes.
- **2026-10-30:** [Bundled media libraries](docs/release-archive-linux.md#media-libraries), [pre-Blackwell NVFP4](docs/cli.md#common-options), Qwen3.5 FP32 logits, a faster first request after startup, and [prefill pacing](docs/serving.md#execution-behavior).
- **2026-10-29:** [Faster tensor-parallel prefill](docs/maintainer/pipeline-stages.md#tensor-parallelism), with native NVFP4/FP8 shares on RTX 50.
- **2026-10-27:** [Tensor parallelism](docs/maintainer/pipeline-stages.md#tensor-parallelism) speeds up decode without peer access and covers all Flash-Next layers.
  Qwen3.5 gains [invariant attention](docs/maintainer/qwen3_5-model.md#invariant-attention).
  The [DeepSeek-V4.1-Flash STQ drafter](docs/deepseek-v41.md#the-stq-drafter) uses half the VRAM of Unsloth's drafter.
- **2026-10-26:** Flash-Next [prefills and decodes faster with disk experts](docs/qwen3-8-flash-next.md#expert-placements).
  [Tensor parallelism](docs/maintainer/pipeline-stages.md#tensor-parallelism) supports all dense weight formats and DFlash.
  All MoE families support [expert parallelism](docs/maintainer/pipeline-stages.md#expert-parallelism-of-the-other-moe-families).
  [DeepSeek-V4.1-Flash](docs/deepseek-v41.md) gains CPU-assisted expert execution, DSpark drafting, and image input.
- **2026-10-25:** NInfer supports [DeepSeek-V4.1-Flash](docs/deepseek-v41.md) from IQ1_M GGUF with host or disk experts.
  Dense Qwen3.5-core models gain `--tensor-parallel`.
- **2026-10-24:** [Flint](docs/flint.md#measurements) decodes 348 tok/s on an RTX 5090 and 210 on an RTX 4090 with host experts.
- **2026-10-23:** NInfer supports [Flint](docs/flint.md) from Unsloth's UD-Q4_K_XL GGUF.
  [Native releases](https://github.com/iamwavecut/ninfer-all/releases/latest) cover Windows and Debian 13 with CUDA 12.9 and 13.4.
- **2026-10-22:** [Typed decisions](docs/serving.md#typed-decisions) are available at `/v1/decisions` and `/v1/systemone`.
  Every model family supports YaRN.
- **2026-10-20:** [CPU threads compute a share of host experts](docs/qwen3-8-flash-next.md#the-cpu-share) beside the GPU.
- **2026-10-18:** The hybrid prefix cache is the server default.
- **2026-10-17:** [Pipeline stages across GPU architectures](docs/build-linux.md#cards-of-several-architectures-in-one-pipeline) in one build.
- **2026-10-16:** [Device profiles](docs/gpus.md) for 36 more GPUs; Gemma 4 and Qwen MoE experts in host memory or on disk.
- **2026-10-14:** Turing (`sm_75`): the [TITAN RTX](docs/performance/titan-rtx-2026-10.md) and RTX 20 series.
- **2026-10-12:** Volta (`sm_70`): the [Tesla V100](docs/performance/tesla-v100-2026-10.md).
- **2026-10-11:** [Gemma 4 26B-A4B](docs/gemma4.md) with MTP, images and video.
- **2026-10-06:** [Qwen3.8-Flash-Next](docs/qwen3-8-flash-next.md), experts on the GPU, in host memory or on disk.
- **2026-09-26:** NInfer-all and [Qwen3.8-27B GSQ-RCO IQ3_S](https://huggingface.co/WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3).
- **2026-09-23:** [Ternary Bonsai 2 27B](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3) on one RTX 3090.

## Quick start

Download an artifact ([all models](#supported-models)) and serve it from the [Docker image](#docker). The server
speaks the OpenAI and Anthropic APIs on http://localhost:8080/v1, answers typed decisions there, and
each card picks up its device profile on its own.

```bash
hf download WaveCut/Ternary-Bonsai-2-27B-NInfer-v3 Ternary-Bonsai-2-27B-ninfer-v3.ninfer --local-dir models

docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/iamwavecut/ninfer-all:cuda13.4 serve /models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --model-id bonsai2-27b --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 \
  --gdn-state-fp16 --spec dflash2 --draft-tokens 5

curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"bonsai2-27b","messages":[{"role":"user","content":"Hello"}]}'

curl http://localhost:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"ticket": "My card was charged twice for one order."},
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "payments and refunds", "shipping": "deliveries"}},
    "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"}}}'
```

The second request is a [typed decision](docs/serving.md#typed-decisions): each answer comes with the
probability of every option, read from the model's first-token log probabilities of the labels.

Flags for every model and memory size, from an 8 GB card to the largest context: [launch recipes](docs/recipes.md),
each also a [configuration file](docs/cli.md#launch-configuration-files) in `configs/` that `--config`
reads.

<details>
<summary><b>Without Docker</b> — the same flags on a build</summary>

```bash
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 5

# The same options from the repository's configuration file.
ninfer-serve --config configs/bonsai2-27b-dflash2.ini
```

A build listens on 127.0.0.1:8080 unless `--host` and `--port` say otherwise. [Building](#building) shows
how to make one.

</details>

## Supported GPUs

NVIDIA GPUs from Volta (compute capability 7.0) to consumer Blackwell (12.0), from 8 GB of memory:
Ternary Bonsai 2 27B runs on an 8 GB RTX 3060 Ti, and Qwen3.8-Flash-Next streams its experts from
host memory or disk through 8 GB. The two [Docker images](#docker) each contain all NInfer targets
for their CUDA version in a shared runtime: CUDA 12.9 includes Volta, and CUDA 13.4 includes Blackwell.
The native release pipeline also assembles a portable Windows or Linux package with **both** CUDA
backends: `ninfer-<os>-x64-cuda12.9-13.4-<commit12>`. Its small command executables share one
runtime per CUDA family. The launcher selects CUDA 12 for Volta, CUDA 13 for consumer Blackwell,
and prefers CUDA 13 on overlapping targets when the driver supports it. All selected `--devices`
must fit one backend. Use `--cuda-backend auto|12|13` to select it explicitly. No CUDA Toolkit
installation is needed. Keep `backends/` and `help/` beside the executables. Linux also needs the
recorded Debian system libraries. `--help` and `--version` do not load CUDA.

Every card by name, with what has run on it:
[Supported GPUs](docs/gpus.md).

*Click a section to expand it.*

<details>
<summary><b>Architectures and builds</b> — six architectures, two CUDA variants</summary>

| Architecture | Compute capability | Build | CUDA | Docker tags | Status |
|---|---|---|---|---|---|
| Blackwell: GeForce RTX 50, RTX PRO Blackwell | 12.0 | `120a` | 13.1 or newer, not 13.2.0 or 13.2.1 | `cuda13.4` | measured; built-in device profiles |
| Ada Lovelace | 8.9 | `89`, or the `86` build | 12.8 or newer | both, native `89` | measured; built-in device profiles |
| Ampere | 8.6 | `86` (the default) | 12.8 or newer | both | measured (the reference card); built-in device profiles |
| Ampere, A100 class | 8.0 | `80` | 12.8 or newer | both | built-in device profiles; not benchmarked |
| Turing | 7.5 | `75` | 12.8 or newer | both | measured; built-in device profiles |
| Volta | 7.0 | `70` | 12.8 or 12.9 | `cuda12.9` | measured; built-in device profiles |
| Hopper; data-center Blackwell; GB10; Pascal and older | | none | | | no build |

</details>

<details>
<summary><b>What fits in your GPU memory</b> — measured, from 8 GB to 48 GB</summary>

| GPU memory | Runs | Measured on |
|---|---|---|
| 8 GB | Ternary Bonsai 2 27B (24,576 tokens with `rk4v4`, no MTP); Qwen3.8-Flash-Next and Inkling 2 Mini with their experts in host memory or on disk | RTX 3060 Ti; RTX 5060 for Inkling 2 Mini |
| 11 GB | Ternary Bonsai 2 27B (200,704 tokens with `rk4v4`; 143,360 with MTP) | RTX 2080 Ti |
| 12 GB | Ternary Bonsai 2 27B (266,240 tokens); Qwen3.8-27B GSQ-RCO IQ3_S (8,192 tokens, no MTP); Inkling 2 Mini with its experts in host memory | RTX 3060; also the RTX 5070 for Inkling 2 Mini |
| 16 GB | Gemma 4 26B-A4B Q4_0 (61,440 tokens); Qwen3.8-27B IQ3_S (241,664 tokens); the official Qwen3.8-27B with `--embedding-q4 --lm-head-q4`; Inkling 2 Mini with its experts in host memory | Tesla V100-SXM2-16GB; RTX 5080, 5060 Ti, 4080 Super, 4060 Ti and A4000 for Inkling 2 Mini |
| 24 GB | Gemma 4 31B Q4_0 (101,376 tokens; 87,040 with MTP); the official Qwen3.8-27B and its MXFP8 tune; Qwen3.6-35B-A3B `groupwise-int` on an RTX 3090; Flash-Next with every expert on the GPU over two cards; DeepSeek-V4.1-Flash with its experts in host memory or on disk | RTX 3090, RTX 4090, TITAN RTX |
| 32 GB | Qwen3.6-35B-A3B NVFP4 (Blackwell only) | RTX 5090 |
| 48 GB | Flash-Next Q2_0 with every expert on one card | L40S, RTX PRO 6000 |

</details>

<details>
<summary><b>Device profiles and limits</b> — how a card gets its kernel routes</summary>

- **Device profiles.** A profile picks the kernel route of every operation and width. The binary
  carries profiles measured on 60 GPU models, from the Tesla V100 and RTX 2060 Super to the RTX PRO
  6000 (the Profile column of [supported GPUs](docs/gpus.md#cards)); a built-in profile applies to
  the same GPU name, compute capability and SM count. Any other card is calibrated once at first
  start, which takes about half a minute to three and a half minutes, and the result is stored for
  later starts; `ninfer-calibrate` measures again after a driver or clock change
  ([device profiles](docs/device-profiles.md)). The `70` build runs its own
  routes, so the V100 profiles hold only the GDN prefill switch and, for the 32 GB card, the Q8 MoE
  prefill frontier.
- **Architecture limits.** NVFP4 expert weights need a `120a` build. The Turing and Volta builds
  replace instructions their GPUs lack and reshape kernels to their shared memory
  ([Turing](docs/build-linux.md#turing-sm_75), [Tesla V100](docs/build-linux.md#tesla-v100-sm_70)).
  Configure accepts no other target, so Pascal and older, Hopper and data-center Blackwell have no
  build.
- **Speed per card.** [Performance](docs/performance.md) lists the measured rates of every card.

</details>

## Supported Models

Each name links to its Hugging Face card. `hf download REPO FILE.ninfer --local-dir models` fetches a file.
The Bonsai, GSQ-RCO, Flash-Next, Edge0, Gemma 4, Flint, DeepSeek-V4.1-Flash and Inkling 2 artifacts use formats that only this engine reads.
Rows go from the fewest bits per weight to the most.
✓ means the artifact carries the component, — that it does not.
Drafting lists the drafters the artifact carries; a linked one is an artifact of its own.
Flash-Next, Edge0 and DeepSeek-V4.1-Flash models also need an n-gram table, an artifact of its own:
their Notes link it, and [N-gram tables](#n-gram-tables) lists them all.

### Ternary Bonsai 2 27B

| Model | Size, GiB | Weights | Vision | Drafting | Notes |
|---|---:|---|:-:|---|---|
| [Ternary Bonsai 2 27B](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3) | 8.87 | ternary, 2.125 bits | ✓ | MTP<br>DFlash2<br>Proposal head | Bonsai-trained MTP head and DFlash2 adapter |

### Qwen3.8-27B

| Model | Size, GiB | Weights | Vision | Drafting | Notes |
|---|---:|---|:-:|---|---|
| [GSQ-RCO IQ3_S](https://huggingface.co/WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3) | 13.99 | GGUF blocks, 3.5 bits | ✓ | MTP<br>DFlash2<br>Proposal head<br>[DSpine](https://huggingface.co/WaveCut/Qwen3.8-27B-DSpine-b16-NInfer-v3) | ISTA-DASLab's GGUF blocks byte for byte with a Q6_K MTP head |
| [Official](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | 19.03 | groupwise-int (Q4/Q5) | ✓ | MTP<br>DFlash2<br>Proposal head<br>[DSpine](https://huggingface.co/WaveCut/Qwen3.8-27B-DSpine-b16-NInfer-v3) | The official artifact, which the reference tables use |
| [Abliterated](https://huggingface.co/WaveCut/Huihui-Qwen3.8-27B-abliterated-NInfer-v3) | 19.03 | groupwise-int (Q4/Q5) | ✓ | MTP<br>DFlash2<br>Proposal head<br>[DSpine](https://huggingface.co/WaveCut/Qwen3.8-27B-DSpine-b16-NInfer-v3) | The official recipe |
| [Uncensored, HauhauCS Aggressive](https://huggingface.co/WaveCut/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-DFlash2-NInfer-v3) | 19.03 | groupwise-int (Q4/Q5) | ✓ | MTP<br>DFlash2<br>Proposal head<br>[DSpine](https://huggingface.co/WaveCut/Qwen3.8-27B-DSpine-b16-NInfer-v3) | The tune's MTP head |
| [MXFP8-CRACK](https://huggingface.co/WaveCut/Qwen3.8-27B-MXFP8-CRACK-NInfer-v3) | 16.96 | groupwise-int | ✓ | MTP<br>Proposal head<br>[DSpine](https://huggingface.co/WaveCut/Qwen3.8-27B-DSpine-b16-NInfer-v3) | Without bundled DFlash2 adapter |
| [DSpine Drafter (Tencent)](https://huggingface.co/WaveCut/Qwen3.8-27B-DSpine-b16-NInfer-v3) | 2.27 | Q8 layers and a BF16 code bridge | — | — | Drafter artifact loaded with `--spec dspine --draft-model` beside any Qwen3.8-27B artifact |

### Qwen3.6-35B-A3B

| Model | Size, GiB | Weights | Vision | Drafting | Notes |
|---|---:|---|:-:|---|---|
| [NVFP4](https://huggingface.co/WaveCut/Qwen3.6-35B-A3B-NVFP4-NInfer-v3) | 21.04 | NVFP4 experts, Q8 projections | ✓ | MTP<br>DFlash2<br>Proposal head | RedHatAI's NVFP4 experts code for code, incoai's DFlash2 drafter; `sm_120a` GPUs only |

### Qwen3.8-Flash-Next

| Model | Size, GiB | Weights | Vision | Drafting | Notes |
|---|---:|---|:-:|---|---|
| [Coder GSQ-RCO IQ1_M](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-NInfer-v3) | 30.76 | GGUF blocks | ✓ | MTP<br>DFlash2 | Expert-pruned coding build, 256 experts per layer; shared-Q8_0 MTP, incoai's Q4_0 DFlash2 drafter; requires an n-gram table ([IQ4_NL](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3) or [FP8](https://huggingface.co/WaveCut/Edge0-ngram-table-FP8-NInfer-v3)) |
| [GSQ-RCO Q2_0](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3) | 38.23 | GGUF blocks: ISTA-DASLab's GSQ-RCO 2.4 bits | ✓ | MTP<br>DFlash2 | Shared-Q8_0 MTP, incoai's Q4_0 DFlash2 drafter; requires an n-gram table ([IQ4_NL](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3) or [FP8](https://huggingface.co/WaveCut/Edge0-ngram-table-FP8-NInfer-v3)) |
| [UD-IQ2_XS](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-UD-IQ2_XS-NInfer-v3) | 47.43 | GGUF blocks: Unsloth Dynamic v3.1 | ✓ | MTP<br>DFlash2 | Shared-Q8_0 MTP, incoai's Q4_0 DFlash2 drafter; requires an n-gram table ([IQ4_NL](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3) or [FP8](https://huggingface.co/WaveCut/Edge0-ngram-table-FP8-NInfer-v3)) |
| [GSQ-RCO IQ3_S](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3) | 54.24 | GGUF blocks: ISTA-DASLab's GSQ-RCO 3.5 bits | ✓ | MTP<br>DFlash2 | Shared-Q8_0 MTP, incoai's Q4_0 DFlash2 drafter; requires an n-gram table ([IQ4_NL](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3) or [FP8](https://huggingface.co/WaveCut/Edge0-ngram-table-FP8-NInfer-v3)) |
| [UD-IQ3_XXS](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-UD-IQ3_XXS-NInfer-v3) | 54.66 | GGUF blocks: Unsloth Dynamic v3.1 | ✓ | MTP<br>DFlash2 | Shared-Q8_0 MTP, incoai's Q4_0 DFlash2 drafter; requires an n-gram table ([IQ4_NL](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3) or [FP8](https://huggingface.co/WaveCut/Edge0-ngram-table-FP8-NInfer-v3)) |

### Edge0

| Model | Size, GiB | Weights | Vision | Drafting | Notes |
|---|---:|---|:-:|---|---|
| [Edge0-120B q3k](https://huggingface.co/WaveCut/Edge0-120B-q3k-NInfer-v3) | 34.60 | Q3_K/IQ4_NL latent experts, Q4_0 encoders and decoders | ✓ | MTP | Edge0's release byte for byte, base model's Vision tower; every expert on one 48 GB card; requires an n-gram table ([FP8](https://huggingface.co/WaveCut/Edge0-ngram-table-FP8-NInfer-v3) or [IQ4_NL](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3)) |
| [Edge0-145B q3k](https://huggingface.co/WaveCut/Edge0-145B-q3k-NInfer-v3) | 47.43 | Q3_K/IQ4_NL latent experts, Q8_0 encoders and decoders | ✓ | MTP | Edge0's release byte for byte, base model's Vision tower; experts in host memory or on disk on one 48 GB card; requires an n-gram table ([FP8](https://huggingface.co/WaveCut/Edge0-ngram-table-FP8-NInfer-v3) or [IQ4_NL](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3)) |
| [Edge0-120B NVFP4](https://huggingface.co/WaveCut/Edge0-120B-NVFP4-NInfer-v3) | 39.58 | NVFP4 latent experts, Q4_0 encoders and decoders | ✓ | MTP | Edge0's release byte for byte, base model's Vision tower; every expert on one 48 GB card; requires an n-gram table ([FP8](https://huggingface.co/WaveCut/Edge0-ngram-table-FP8-NInfer-v3) or [IQ4_NL](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3)) |
| [Edge0-145B NVFP4](https://huggingface.co/WaveCut/Edge0-145B-NVFP4-NInfer-v3) | 54.41 | NVFP4 latent experts, Q8_0 encoders and decoders | ✓ | MTP | Edge0's release byte for byte, base model's Vision tower; experts in host memory or on disk on one 48 GB card; requires an n-gram table ([FP8](https://huggingface.co/WaveCut/Edge0-ngram-table-FP8-NInfer-v3) or [IQ4_NL](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3)) |

### Gemma 4

| Model | Size, GiB | Weights | Vision | Drafting | Notes |
|---|---:|---|:-:|---|---|
| [Gemma 4 26B-A4B it QAT](https://huggingface.co/WaveCut/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-NInfer-v3) | 14.75 | GGUF Q4_0 blocks | ✓ | MTP | Google's QAT GGUF byte for byte; MTP is Google's QAT assistant |
| [Gemma 4 31B it QAT](https://huggingface.co/WaveCut/gemma-4-31B-it-QAT-Q4_0-MTP-NInfer-v3) | 17.77 | GGUF Q4_0 blocks | ✓ | MTP | Google's QAT GGUF byte for byte, the dense model; MTP is Google's QAT assistant for it |

### Flint

| Model | Size, GiB | Weights | Vision | Drafting | Notes |
|---|---:|---|:-:|---|---|
| [Flint UD-Q4_K_XL](https://huggingface.co/WaveCut/Flint-UD-Q4_K_XL-NInfer-v3) | 23.38 | GGUF blocks, Q8_0 to IQ3_S | — | — | Unsloth's UD-Q4_K_XL GGUF byte for byte; drafts by context lookup |

### DeepSeek-V4.1-Flash

| Model | Size, GiB | Weights | Vision | Drafting | Notes |
|---|---:|---|:-:|---|---|
| [GSQ-RCO IQ1_M](https://huggingface.co/WaveCut/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-NInfer-v3) | 117.18 | GGUF blocks, IQ1_S to IQ2_S experts | ✓ | DSpark | ISTA-DASLab's GGUF byte for byte; experts in host memory or on disk; vision tower (images); DSpark drafter (`--spec dspark`) or context lookup; requires the [Engram table](https://huggingface.co/WaveCut/DeepSeek-V4.1-Flash-Engram-IQ4_NL-NInfer-v3) |
| [GSQ-RCO Coder IQ2_XXS](https://huggingface.co/WaveCut/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-NInfer-v3) | 71.85 | GGUF blocks, IQ1_M to IQ2_XS experts | ✓ | DSpark | ISTA-DASLab's expert-pruned coding build byte for byte, 192 of 384 experts per block; otherwise as the IQ1_M artifact; requires the [Engram table](https://huggingface.co/WaveCut/DeepSeek-V4.1-Flash-Engram-IQ4_NL-NInfer-v3) |

### Inkling 2 Mini

| Model | Size, GiB | Weights | Vision | Drafting | Notes |
|---|---:|---|:-:|---|---|
| [Inkling 2 Mini UD-IQ1_M](https://huggingface.co/WaveCut/Inkling-2-Mini-UD-IQ1_M-NInfer-v3) | 8.18 | GGUF blocks, IQ1_S to IQ2_S [gate; up], Q2_0 down banks | ✓ | DSpark | Unsloth's UD-IQ1_M GGUF byte for byte; otherwise as UD-IQ4_XS |
| [Inkling 2 Mini UD-Q2_K_XL](https://huggingface.co/WaveCut/Inkling-2-Mini-UD-Q2_K_XL-NInfer-v3) | 9.39 | GGUF blocks, IQ1_S to IQ2_S [gate; up], Q2_0 and NVFP4 down banks | ✓ | DSpark | Unsloth's UD-Q2_K_XL GGUF byte for byte, its one MXFP4 bank as the same values in Q8_0; otherwise as UD-IQ4_XS |
| [Inkling 2 Mini APEX-I MiniPlus V2.1](https://huggingface.co/WaveCut/Inkling-2-Mini-APEX-I-MiniPlus-V2.1-NInfer-v3) | 12.80 | GGUF blocks, IQ3_XXS and IQ3_S experts | ✓ | DSpark | IsValorum's APEX-I MiniPlus GGUF byte for byte with its Q8_0 vision tower; Unsloth's DSpark drafter; otherwise as UD-IQ4_XS |
| [Inkling 2 Mini UD-Q3_K_XL](https://huggingface.co/WaveCut/Inkling-2-Mini-UD-Q3_K_XL-NInfer-v3) | 13.40 | GGUF blocks, NVFP4 and IQ3 experts | ✓ | DSpark | Unsloth's UD-Q3_K_XL GGUF byte for byte; otherwise as UD-IQ4_XS |
| [Inkling 2 Mini UD-IQ4_XS](https://huggingface.co/WaveCut/Inkling-2-Mini-UD-IQ4_XS-NInfer-v3) | 15.36 | GGUF blocks, NVFP4 experts in 20 of 24 layers | ✓ | DSpark | Unsloth's UD-IQ4_XS GGUF byte for byte; Q8_0 vision tower (images); DSpark drafter, by default for one request at a time, or context lookup |
| [Inkling 2 Mini UD-Q8_K_XL](https://huggingface.co/WaveCut/Inkling-2-Mini-UD-Q8_K_XL-NInfer-v3) | 22.06 | BF16 attention, head, token table and layers 20-23's experts; NVFP4 experts in 20 layers | ✓ | DSpark | Unsloth's UD-Q8_K_XL GGUF byte for byte; a card of 32 GB or more, or host experts; otherwise as UD-IQ4_XS |

Tencent's [DSpine drafter for Qwen3.8-27B](https://huggingface.co/WaveCut/Qwen3.8-27B-DSpine-b16-NInfer-v3)
(2.27 GiB, Q8 layers and a BF16 code bridge) is an artifact of its own, which
`--spec dspine --draft-model` loads beside any Qwen3.8-27B artifact.

### N-gram tables

A model names the table it was converted with by digest; pass the table with `--ngram-table`.

Every Qwen3.8-Flash-Next and Edge0 model reads either of these two tables. The engine accepts any
table of the model's shape, in any row format
([another table of the same shape](docs/qwen3-8-flash-next.md#another-table-of-the-same-shape)).
Perplexity: the first 40 KB of WikiText, INT8 KV, one A40.

| Table | Size, GiB | Rows | Hot-row profile | Named by | Perplexity: Flash-Next Q2_0 · Edge0-120B-q3k |
|---|---:|---|:-:|---|---|
| [IQ4_NL](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3) | 26.92 | IQ4_NL blocks, 90 bytes a row: the GSQ-RCO and Unsloth releases' table byte for byte | ✓ | Flash-Next builds, Coder | 2.542153 · 5.017691 |
| [FP8](https://huggingface.co/WaveCut/Edge0-ngram-table-FP8-NInfer-v3) | 48.38 | FP8 E4M3 codes and an FP16 scale, 162 bytes a row: Edge0's `ple-fp8.gguf` codes byte for byte, its one scale in every row | ✓ | Edge0 builds | 2.531971 · 4.964565 |

DeepSeek-V4.1-Flash models read the Engram table, the release's IQ4_NL rows with a hot-row profile
(`--ngram-residency ram-hot`).

| Table | Size, GiB | Rows | Hot-row profile | Named by |
|---|---:|---|:-:|---|
| [Engram](https://huggingface.co/WaveCut/DeepSeek-V4.1-Flash-Engram-IQ4_NL-NInfer-v3) | 103.32 | IQ4_NL blocks: the release's rows | ✓ | GSQ-RCO IQ1_M, Coder IQ2_XXS |

The official NInfer artifacts (Qwen3.6-27B, Qwen3.6-35B-A3B and Qwen3.8-27B, `groupwise-int` and
`nvfp4`) load too; the [documentation map](docs/README.md#model-artifacts) links them. Weight
conversion shows how the [Bonsai](docs/weight-conversion.md#ternary-bonsai-2-27b) and
[GSQ-RCO](docs/weight-conversion.md#a-mixed-precision-qwen38-27b-gguf) artifacts are built.

## Highlights

Ternary Bonsai 2 27B decodes 397 tok/s on an RTX 5090 with DFlash2 and holds a 978,944-token context with
every planted code found. Qwen3.8-Flash-Next decodes 138 tok/s with every expert on one RTX PRO 6000, and
96 tok/s with its experts in host memory on one RTX 5090. DeepSeek-V4.1-Flash decodes 52 tok/s on one
RTX 5090, 48 on one RTX PRO 6000 and 39 on one RTX 4090 with its 106.7 GiB of experts in host memory. Inkling 2 Mini
decodes 182 tok/s on an RTX 5090, 1.4 to 2.8 times llama.cpp's rate on 25 cards, and runs on an 8 GB card with its
experts in host memory. Every table of every card is on the [performance page](docs/performance.md).

*Click a section to expand it.*

<details>
<summary><b>Ternary Bonsai 2 27B</b> — up to 397 tok/s, a million tokens of context</summary>

Measured in September 2026, one card each, greedy, one request unless the column says otherwise.

| Card | Short chat, DFlash2 7 drafts (tok/s) | After a 261K-token document (tok/s) | 261K-token prompt, to the first token | Largest context, all three codes found | Eight requests at once, MTP 3 (tok/s) |
|---|---:|---:|---:|---:|---:|
| RTX 3090 | 202 | 90 | 215 s | 970,752 | 551 |
| RTX 4090 | 256 | 123 | 102 s | 958,464 | 824 |
| RTX 5090 | 397 | 218 | 82 s | 978,944 | 1,063 |
| RTX PRO 6000 | 381 | 218 | 78 s | 1,048,576, two of three codes | 1,155 |

The RTX PRO 6000 (96 GB) starts at the engine's ceiling of 1,048,576 tokens with every KV format and drafter.
Filled to it, the model misses the code at 90% (about 943K); up to about 880K it found every code on every card.

</details>

<details>
<summary><b>Qwen3.8-27B</b> — up to 237 tok/s, 872,448 tokens of context filled</summary>

The official `groupwise-int` artifact, measured in September 2026, one card each, greedy, one request unless the column says otherwise.

| Card | Short chat, DFlash2 7 drafts (tok/s) | Largest context, all three codes found | Eight requests at once, MTP 3 (tok/s) |
|---|---:|---:|---:|
| RTX 3090 | 118 | 417,792 | 329 |
| RTX 4090 | 149 | 405,504 | 442 |
| RTX 5090 | 236 | 872,448 | 690 |
| RTX PRO 6000 | 237 | 1,048,576, two of three codes | 739 |

- **RTX PRO 6000 against RTX 5090.** Re-measured in October beside an RTX 5090, both at 600 W, the
  PRO 6000 came back within 2% of September wherever the drafts accepted the same share. One
  request's decode is bound by memory bandwidth, and both cards have 1.79 TB/s of GDDR7 (Bonsai 2
  without speculation: 167.7 against 170.4 tok/s). The PRO 6000's 188 SMs against 170 show where
  compute decides: the 261K prompt is 2% faster and eight requests at once 6 to 7% faster
  ([re-check](docs/performance.md#rtx-pro-6000-against-rtx-5090-october-2026)).
- **Draft length.** DFlash2 with seven drafts is fastest on short answers; after long documents the
  best count is three to seven. MTP runs up to fifteen drafts and is fastest at three to five.

</details>

<details>
<summary><b>Qwen3.8-Flash-Next</b> — 138 tok/s with the experts on a GPU, 98 tok/s from host memory</summary>

The Q2_0 release (35.9 GiB model and 26.8 GiB n-gram table), one request, greedy, October 2026. Decode of
a short answer, in tok/s:

| Card | Every expert on the GPUs | Experts in host memory | Experts on disk, in the page cache | Experts on disk, cold |
|---|---:|---:|---:|---:|
| RTX PRO 6000 | 138 (one card) | 55 | 76 | 35 |
| RTX 5090 | 136 (two cards) | 98 | 72 | 38 |
| RTX 4090 | 97 (two cards) | 74 | 47 | 21 |
| RTX 3090 | 90 (two RTX 3090 Ti) | 39 | 28 | 20 |

Prefill of a 4,463-token prompt, in tok/s:

| Card | Every expert on the GPUs | Experts in host memory | Experts on disk, in the page cache | Experts on disk, cold |
|---|---:|---:|---:|---:|
| RTX PRO 6000 | 2,939 | 1,297 | 2,024 | 1,000 |
| RTX 5090 | 3,809 | 3,290 | 1,827 | 1,108 |
| RTX 4090 | 3,608 | 1,801 | 1,144 | 558 |
| RTX 3090 | 1,504 | 663 | 503 | 474 |

- **The host matters.** Host and disk rows depend on the host as much as on the card. The RTX 5090
  host had PCIe 5.0 x16, the RTX PRO 6000 and RTX 4090 hosts PCIe 4.0 x16, and the RTX 3090 host
  PCIe 3.0 x16 and two sockets (the test ran pinned to the GPU's node). The host, disk and cold
  columns of the three GeForce cards were measured on October 26, the rest on October 5 and 6.
- **Prompt spans.** With host or disk experts a prompt prefills in spans of up to eight chunks
  (sixteen for one request at a time), so each layer's missing experts cross the bus once per
  span. From the CLI with Q2_0 disk experts in the page cache, a 7,680-token prompt prefilled at
  1,830 tok/s on the RTX 5090 and the answer decoded at 115 tok/s, against 450 and 32 chunk by
  chunk
  ([three cards](docs/qwen3-8-flash-next.md#expert-placements)).
- **Other releases and cards.** IQ3_S, the Coder IQ1_M build and the 8 and 12 GB cards are in the
  [expert placements](docs/performance.md#expert-placements). Six reasoning requests at once on two
  RTX 5090s produced 193 tok/s of output with Q2_0.

</details>

<details>
<summary><b>DeepSeek-V4.1-Flash</b> — 52 tok/s on one RTX 5090, 48 on one RTX PRO 6000 (DSpark: 1.3 to 2.1 times as fast), 34 to 39 on one RTX 4090</summary>

ISTA-DASLab's GSQ-RCO IQ1_M GGUF (112.7 GiB, and its 103.0 GiB Engram table), one request, greedy,
thinking off, one GPU, October 2026. The routed experts stay in pinned host memory or in the file,
the card caches as many as fit, and CPU threads compute a share of each decode call's missing host
experts:

| Card | Experts cached on the card | Host experts: short answer (tok/s) | after 8,086 tokens (tok/s) | prefill of 8,086 tokens (tok/s) | Disk experts, page cache warm: short answer (tok/s) |
|---|---:|---:|---:|---:|---:|
| RTX PRO 6000 | 74% | 48.0 | 45.3 | 1,310 | 37.4 |
| RTX 5090 | 15% | 51.5 | 42.2 | 912 | 16.7 |
| RTX 4090 | 7% | 33.8 | 27.5 | 915 | 9.5 |
| RTX 3090 | 7% | 19.8 | 15.2 | 416 | 6.0 |

The RTX PRO 6000 row is an earlier build that ran prompts chunk by chunk and copied every missing
expert; the others ran on October 26 on hosts with PCIe 5.0 (RTX 5090), 4.0 (RTX 4090) and 3.0
(RTX 3090) links ([per card](docs/deepseek-v41.md#per-card)). On an RTX 4090 host with 64 CPU cores
the build decodes at 38.9 and 31.4 tok/s, against 33.5 and 26.6 when it copies every miss
([the CPU share](docs/deepseek-v41.md#the-cpu-share)); there Strata 0.2.6 decodes the same GGUF at
34 to 40 tok/s, computing about half the experts on its CPU threads, and prefills at 635 to 690
tok/s ([measurements](docs/deepseek-v41.md#measurements)). With host experts the server peaks at 116
GiB of host memory. On October 27, with the release's DSpark drafter (`--spec dspark`, 3.58 GiB on
the card in the STQ encoding the artifact carries), an RTX PRO 6000 decoded five requests 1.28 to
2.07 times as fast as without it, and an RTX 4090 decoded code and reasoning 1.19 and 1.21 times as
fast, the other three requests 0.85 to 1.02 times, prefilling at plain speed
([the drafter](docs/deepseek-v41.md#the-stq-drafter)). With `--vision` it reads images through
the release's vision tower ([images](docs/deepseek-v41.md#images)).

</details>

<details>
<summary><b>Inkling 2 Mini</b> — 182 tok/s on an RTX 5090, 1.4 to 2.8 times llama.cpp's decode, from 8 GB</summary>

Unsloth's UD-IQ4_XS GGUF with the vision tower and the DSpark drafter (a 15.4 GiB artifact), one request, F16 KV,
November 21, 2026, against llama.cpp on the same GGUF (Unsloth's build `b12588-mix-c0c3de8`; mainline
llama.cpp does not load it). A prompt of 10,000 tokens and a decode of 128, in tok/s, NInfer-all / llama.cpp:

| Card | Experts | Prefill of 10,000 tokens | Decode |
|---|---|---:|---:|
| RTX 5090 | on the card | 6,664 / 2,318 | 182.1 / 96.9 |
| RTX 4090 | on the card | 4,537 / 2,307 | 140.1 / 75.0 |
| RTX 3090 | on the card | 2,046 / 1,236 | 112.0 / 59.5 |
| RTX 5080 | in host memory | 3,238 / 1,377 | 116.5 / 41.3 |
| RTX 3060 12 GB | in host memory | 759 / 387 | 48.8 / 24.8 |
| RTX 5060 8 GB | in host memory | 1,074 / 293 | 33.1 / 14.7 |

On cards of 8 to 16 GB the routed experts stay in pinned host memory and the card caches the most used,
while llama.cpp places the model with its own `--fit`. From disk (`--expert-residency disk`) the RTX 5060
prefills 591 tok/s and decodes 9.6. While one request runs, the release's DSpark drafter decodes the same
text 1.06 to 1.33 times as fast with the experts on the card. Every card, the drafter and the hosts are in
[Inkling 2 Mini's measurements](docs/inkling2.md#across-cards).

With the [Docker image](#docker), or the same `--config` on a
[native release](https://github.com/iamwavecut/ninfer-all/releases/latest):

```bash
hf download WaveCut/Inkling-2-Mini-UD-IQ4_XS-NInfer-v3 Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer --local-dir models

# 24 GB and more: every expert on the card, images, DSpark, a 64K-token context.
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 -w / \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/iamwavecut/ninfer-all:cuda13.4 serve --config /opt/ninfer/configs/inkling2-mini.ini

# 8 to 16 GB: the experts in 13.1 GiB of pinned host memory, a 32K-token context
# (the 8 GB RTX 5060 above ran with --max-context 12288).
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 -w / \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/iamwavecut/ninfer-all:cuda13.4 serve --config /opt/ninfer/configs/inkling2-mini-host.ini
```

</details>

## Features

*Click a section to expand it.*

<details>
<summary><b>Models and weight formats</b> — Qwen3.5 family, GGUF block formats, Ternary Bonsai 2, Flash-Next, Gemma 4, Flint, DeepSeek-V4.1-Flash, Inkling 2 Mini</summary>

- **Qwen3.5 family.** Dense and MoE (`Qwen3_5ForCausalLM`, `Qwen3_5MoeForCausalLM`) with MTP,
  DFlash (35B-A3B) and DFlash2 (Qwen3.8-27B, 35B-A3B) drafters, Vision for images and video, and
  `groupwise-int`, FP8, NVFP4 and ternary weights. Native MoE routed experts can stay in pinned
  host memory or the artifact's files, brought into whole-layer GPU banks with an optional device
  cache ([expert residency](docs/maintainer/qwen3_5-model.md#routed-expert-residency)). NVFP4
  expert banks prefill through W4A4 and need a `120a` build.
- **GGUF block formats.** Releases that pick a ggml type per tensor, such as ISTA-DASLab's GSQ-RCO,
  convert without requantization and multiply their blocks in place. The 3.5-bit Qwen3.8-27B IQ3_S
  release holds 10.95 GiB of weights instead of 15.9, scores WikiText-2 perplexity 7.071 (the
  official artifact: 7.286) and decodes at 59.9 tok/s against the official artifact's 40.3 on an
  RTX 3090 ([GGUF block formats](docs/gguf.md)).
- **Ternary Bonsai 2 27B.** PrismML's [ternary Qwen3.8-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
  runs from `t2_g128_fp16` weights: 2.125 bits per weight, imported without rounding, with the
  Hadamard rotations fused into the norms and gates and integer activations for its projections.
  The `bonsai2_27b_ternary` recipe adds ProCreations' MTP head and DFlash2 adapter and an exact
  proposal head.
- **Qwen3.8-Flash-Next.** The 125B-parameter MoE (512 experts, about 6B active) from the GSQ-RCO
  GGUF releases, with its experts on the GPUs, in pinned host memory with a GPU cache
  (`--expert-residency host`) or in the artifact's files read into a GPU cache
  (`--expert-residency disk`, under 1 GB of RAM). It serves up to eight requests with prefix reuse,
  structured output, images and video, and MTP drafting. On two RTX 5090s the Q2_0 release scores
  93.3% on AIME 2025 and 86.4% on GPQA-Diamond ([Qwen3.8-Flash-Next](docs/qwen3-8-flash-next.md)).
- **Gemma 4 26B-A4B.** Google's QAT Q4_0 GGUF release, imported block for block (13.4 GiB), or a
  mixed-format GGUF of it: the sliding-window and full-attention blocks (the latter caching values
  only), the dense MLP beside 128 routed experts, every `--kv-dtype` format, up to eight concurrent
  requests with prefix reuse through host memory and shared system prompts, tool calls and
  structured output, images and video through the release's vision tower (on the
  GPU, streamed from pinned memory or on CPU threads), and Google's multi-token-prediction assistant
  for speculative decoding. On an RTX 3090 it prefills a 10,000-token prompt at 6,045 tok/s and decodes at 215
  tok/s, 190 after 10,000 tokens, against llama.cpp's 4,697, 169 and 152 on the same GGUF; with two
  drafts a round chat answers decode at 291 tok/s instead of 209 ([Gemma 4](docs/gemma4.md)).
- **Gemma 4 31B.** The dense model from Google's QAT Q4_0 GGUF (16.4 GiB on the GPU) with its own
  QAT assistant and the vision tower, on one 24 GB card: on an RTX 3090 it holds 101,376 tokens of
  F16 context and decodes 42 tok/s plainly, 95 to 110 tok/s with two drafts a round
  ([Gemma 4 31B](docs/gemma4.md#gemma-4-31b)).
- **Flint.** Recursive's 39.8B coding MoE (2.5B active), from a llama.cpp GGUF such as Unsloth's
  UD-Q4_K_XL, imported block for block (23.4 GiB): sliding-window and position-free full-attention
  blocks with learned sinks, 256 routed experts beside a shared one, up to eight concurrent
  requests with prefix reuse through host memory, the model's own reasoning, GLM-4.7 tool calls,
  structured output, typed decisions, context-lookup drafting, and experts on the GPU, in pinned
  host memory or in the artifact's files. On an RTX 5090 it prefills a 10,000-token prompt at
  14,656 tok/s and decodes at 348 tok/s, 315 after 10,000 tokens, against llama.cpp's 8,679, 223
  and 209 on the same GGUF; on an RTX 3090, with its experts in host memory, at 5,755, 161 and 140
  against llama.cpp's 2,553, 80 and 77 with four blocks' experts on the CPU
  ([Flint](docs/flint.md)).
- **DeepSeek-V4.1-Flash.** DeepSeek's 40-block MoE (384 routed experts, 6 per token) from
  ISTA-DASLab's GSQ-RCO IQ1_M GGUF, imported block for block, with its Engram table: four-stream
  hyper-connections, latent attention over a 128-position window and indexer-selected compressed
  latents with the release's FP8 and FP4 roundings, Engram rows read from the table's file or
  RAM, up to eight concurrent requests with prefix reuse through host memory, reasoning, DSML
  tool calls, structured output, typed decisions, images through the release's vision tower, the
  release's DSpark drafter and context-lookup drafting. Its 106.7 GiB of
  routed experts stay in pinned host memory or in the artifact's files, read into bounded
  buffers rather than mapped, and CPU threads compute a share of each decode call's missing host
  experts. Prompts run block by block in spans of up to 16,384 tokens, so each block's experts
  cross the bus once per span. On one RTX 4090 with its experts in host memory it decodes at 31.4
  to 38.9 tok/s and prefills 8,086 tokens at 938 tok/s; on the same host
  Strata 0.2.6 decodes at 34 to 40 tok/s and prefills at 635 to 690
  ([DeepSeek-V4.1-Flash](docs/deepseek-v41.md)).
- **Inkling 2 Mini.** Thinking Machines' looped 26.4B MoE (3.4B active per token), the engine's
  first looped transformer, from Unsloth's UD-IQ4_XS GGUF (15.4 GiB), its UD-IQ1_M to UD-Q3_K_XL (8.2 to 13.4 GiB), its UD-Q8_K_XL (22.1 GiB) or
  IsValorum's APEX-I MiniPlus (12.8 GiB) imported block for block: 24 stored layers run as 48 visits, the 12 core layers three
  times with messages between passes, sliding-window and full attention with a learned distance
  table and sinks, depth attention over the earlier layers' states, 128 routed experts, up to eight
  concurrent requests with prefix reuse through host memory, the model's own reasoning and tool-call
  blocks, structured output, typed decisions, images through the release's vision tower, the
  release's DSpark drafter and context-lookup drafting, and experts on the GPU, in pinned host
  memory or in the artifact's files. On an RTX 3090 it prefills a 10,000-token prompt at 2,031 tok/s
  and decodes at 111.4 tok/s, 95.9 after 10,000 tokens, against llama.cpp's 1,228, 58.6 and 55.9 on
  the same GGUF; DSpark decodes greedy answers 1.14 to 1.41 times as fast
  ([Inkling 2 Mini](docs/inkling2.md)).
- **CPU share of host and disk experts.** With Flash-Next, Gemma 4, Flint, Inkling 2 or native Qwen3.5/3.6
  MoE experts off the GPU, or DeepSeek-V4.1-Flash's experts in host memory, CPU threads compute a
  share of each decode call's cache misses from host memory while the GPU copies the rest, bit for
  bit equal to the GPU's products (`--expert-dma-share auto`). The pool takes `min(physical cores, usable CPUs / 2)` threads by
  default, where usable CPUs are the affinity mask bounded by a container's CPU quota (15 threads
  on a 30-CPU quota); `--expert-cpu-threads N` sets another count
  ([common options](docs/cli.md#common-options)).

</details>

<details>
<summary><b>Speculative decoding</b> — MTP, DFlash, DFlash2, DSpark, DSpine, n-gram copying and context lookup</summary>

- MTP, DFlash and DFlash2 with 1 to 15 drafts; the artifact's proposal head drafts by default when it
  stores one. Under MTP and DFlash a sampled request keeps a draft only while it equals the model's
  own keyed draw, so every emitted token is the model's sample; DFlash2 verifies its proposal
  distribution by rejection sampling.
- Gemma 4 drafts with Google's multi-token-prediction assistant (`--spec mtp`, 1 to 7 drafts), four
  blocks that read the model's own KV cache; its verification rows are bit for bit plain decode rows,
  so the output is the one plain decoding gives.
- DeepSeek-V4.1-Flash drafts with the release's DSpark drafter (`--spec dspark`, 1 to 5 drafts kept
  while its confidence holds): three blocks that read the backbone's last three blocks' streams. Its
  verification rows are bit for bit plain decode rows. In beamster's STQ encoding, which the
  published artifact carries (3.58 GiB on the card), it decodes 1.28 to 2.07 times as fast on an
  RTX PRO 6000; on an RTX 4090, whose expert cache it shrinks, it decodes code and reasoning 1.19
  and 1.21 times as fast and other requests 0.85 to 1.02 times.
- Inkling 2 Mini drafts with the release's DSpark drafter (`--spec dspark`, 1 to 7 drafts), three
  non-causal blocks over the target's tapped states and a recurrent head that corrects the target
  head's logits; without `--spec` it drafts while the Engine serves one request at a time, 3 drafts
  a round adapting to their survival, and the output is plain decoding's. On an RTX 3090 it decodes
  greedy answers 1.14 to 1.41 times as fast and sampled ones 0.99 to 1.24 times; over two RTX
  3090s with `--tensor-parallel`, where it also drafts by default, 1.33 to 1.70 and 1.18 times.
- A DSpine drafter is an artifact of its own (`--spec dspine --draft-model PATH`): 1 to 15 drafts
  on Qwen3.5-core models and Qwen3.8-Flash-Next, 1 to 7 on Gemma 4, Flint, DeepSeek-V4.1 and
  Inkling 2.
  Tencent's drafter for Qwen3.8-27B, converted, decodes a code answer 3.08 times and a prose answer
  1.86 times as fast as plain decoding on an RTX 5090, where DFlash2 reaches 3.29 and 2.05
  ([measurements](docs/performance.md#dspine-on-qwen38-27b)).
- Flint, which has no drafter, DeepSeek-V4.1-Flash without `--spec` and Inkling 2 draft by context lookup
  (`--lookup-ngram N`, 1 to 7 drafts): what followed the last N tokens' earlier occurrence (the one
  the sequence was already copying while it still matches, else the newest), verified in one pass of
  decode rows.
- `--adaptive-mtp` picks each round's width from measured draft survival and round cost: an MTP
  round's, and on Qwen3.5-core models a DFlash, DFlash2 or DSpine round's (`--draft-tokens` or half).
- N-gram copy drafting beside the neural drafter (on with `--spec`) verifies up to 15 tokens copied
  from earlier prompt, tool-result or output text ([ngram copy proposals](docs/ngram.md)).

</details>

<details>
<summary><b>Context and memory</b> — ten KV formats, up to 1,048,576 tokens, a context cache with a disk tier</summary>

- Ten KV formats, from `bf16` to `rk2v4-e8` at 216 bytes per token and KV head; `rk8v4` is the
  all-round choice, `rk4v4` holds the native 262,144-token window of Qwen3.8-27B on a 24 GB card,
  and `bf16p` keeps BF16's bits in 784 bytes instead of 1,024
  ([context and memory](docs/cli.md#context-and-memory)).
- Up to 1,048,576 tokens of context, four times the native window, with plain RoPE or YaRN
  (`--rope-yarn`, for Qwen3.5, Qwen3.8-Flash-Next and Gemma 4; Flint's full-attention blocks and
  Inkling 2's attention carry no rotation and DeepSeek-V4.1's compressed blocks the release's own
  YaRN, so none of them takes scaling).
- A hybrid prefix cache by default, which shares content-addressed KV blocks and sparse state
  snapshots across requests on the device and in pinned host memory; or a checkpoint catalog
  (`--use-original-prefix-caching`) with an optional disk tier that survives restarts
  (`--disk-kv-path`, `--disk-kv-restore`) and endpoint and branch anchors that let an edited or
  regenerated turn resume where it left the conversation.
  Gemma 4, Flint, DeepSeek-V4.1, Inkling 2 and Flash-Next keep state images in pinned host memory
  with a disk tier, a prefix cache file and session slots of their own
  ([Serving](docs/serving.md#context-cache-of-the-other-families)).
- Precision trades that buy context: `--embedding-q4`, `--lm-head-q6`, `--gdn-state-fp16`
  ([quality trades](docs/maintainer/quality-trade-experiments.md)).
- A token embedding kept in mapped host memory (`--embedding-residency host`), which frees its
  VRAM for context with no change to the output.

</details>

<details>
<summary><b>Serving</b> — OpenAI, Anthropic and llama.cpp APIs, typed decisions, model suspend, several models behind one server</summary>

- OpenAI Responses and Chat Completions, Anthropic Messages, llama.cpp's `/completion`,
  `/tokenize`, `/detokenize`, `/apply-template`, `/v1/rerank`, `/metrics`, `/slots` and `/props`,
  streaming, tool calls, structured output through xgrammar (`--structured-output`), token log
  probabilities and prompt grafts ([HTTP serving](docs/serving.md)).
- **Typed decisions.** `/v1/systemone` (TypeSafe's System One API, as llama.cpp, vLLM and Ollama
  serve it) and `/v1/decisions` (OpenAI's Decisions API) answer choice, yes/no and score questions
  about a state with the served chat model: one prompt per question, the answer read from the
  first token's exact label log probabilities, the state prefilled once and every question run
  after it as a branch of one read; up to 200 questions, 255 options, Runoff's order-independent
  choices, a conversation as the state, images and video as data URLs (with `--vision`), and a
  margin by which a client escalates near ties ([typed decisions](docs/serving.md#typed-decisions)).
- **Model suspend.** `--model-suspend` lets an idle server give its device memory back without
  exiting and take it again on the next request, retained conversations included. Ternary Bonsai 2
  27B on an RTX 3090: 7.9 GiB down to 0.3 GiB in 0.33 s, back in 1.1 s
  ([model suspend](docs/serving.md#model-suspend)).
- **Several models behind one server.** With `--models-dir` or a llama.cpp-style
  `--models-preset`, `ninfer-serve` is a router with llama.cpp's model API; with `--model-suspend` a
  model sleeps instead of unloading: two 27B models on one 24 GB RTX 3090 swap in 1.8 s instead of
  an 11.6 s cold load ([several models](docs/serving.md#several-models-router)).
- **Control vectors and direction LoRA adapters.** Qwen3.5-core models load llama.cpp control
  vectors and the rank-one LoRA adapters that abliteration tools write, such as AtomicChat's for
  Ternary Bonsai 2 27B; a request sets adapter scales through llama.cpp's `lora` field
  ([LoRA adapters and control vectors](docs/serving.md#lora-adapters-and-control-vectors)).
- `GET /stats` and a [monitoring dashboard](tools/monitor/README.md), request logs with rotation,
  and llama.cpp's WebUI, built into the Linux packages and the Docker images.

</details>

<details>
<summary><b>Kernels and devices</b> — route profiles per card, pipeline stages, FP8 and NVFP4 tensor-core routes</summary>

- **Device route profiles.** Each card's measured profile picks the kernel route for every
  operation and width: built in for the cards in [supported GPUs](#supported-gpus), measured once
  at first start on any other ([device profiles](docs/device-profiles.md)).
- **Pipeline stages.** `--devices 0,1,...` splits the layers over up to eight GPUs on Linux, each
  stage owning its weights, KV and state; the cards may differ where the build runs on each, such as
  RTX 30 with RTX 40 cards on the `86` build ([pipeline stages](docs/maintainer/pipeline-stages.md)).
  `--tensor-parallel` divides every layer over the devices instead: each holds a share of the GDN
  heads, of the attention's KV heads with their cache and of the dense FFN rows or MoE experts in
  any weight format, sized to the card, and reads only those per token
  ([tensor parallelism](docs/maintainer/pipeline-stages.md#tensor-parallelism)), Qwen3.8-Flash-Next
  its GDN key heads, attention KV heads and routed experts. Gemma 4, Flint, DeepSeek-V4.1-Flash and
  Inkling 2 divide only their routed experts this way.
- FP8 A8 and NVFP4 W4A4 tensor-core routes on every `120a` build, an FP4 tensor-core prompt kernel
  for an NVFP4 KV cache, and programmatic dependent launches on Blackwell.

</details>

## Docker

The release pipeline packages two Debian 13 / Linux x64 images from the native Linux archives.
Each image contains one executable with all NInfer architectures supported by its CUDA variant.
The image build does not compile NInfer again.

| Image tag | Toolkit | Embedded GPU architectures |
|---|---|---|
| `ghcr.io/iamwavecut/ninfer-all:cuda12.9` | CUDA 12.9.2 | `sm_70`, `sm_75`, `sm_80`, `sm_86`, `sm_89` |
| `ghcr.io/iamwavecut/ninfer-all:cuda13.4` | CUDA 13.4.3 | `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_120a` |

Choose `cuda12.9` for Volta and `cuda13.4` for Blackwell. Both cover Turing, Ampere and Ada.
Both support a layer pipeline (`--devices`) across different architectures included in that image.
The host needs a compatible NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Windows archives, Linux archives and Docker images use the same two architecture sets:

![GPU architectures included in NInfer's CUDA 12.9.2 and CUDA 13.4.3 builds](docs/assets/cuda-architectures.svg)

Each archive and image contains all device images inside its outline. These are the project's build targets,
not the complete list of architectures supported by NVIDIA's Toolkits.
See the [release process](docs/maintainer/release-process.md) for build and verification status.

<details>
<summary><b>Commands, volumes and tags</b> — what the container runs</summary>

- **Tags.** `cuda12.9` and `cuda13.4` update after both Linux CUDA builds and both Docker jobs succeed for the
  current `master`. Windows builds do not block them.
  Pin a build with `cuda12.9-<commit12>` or `cuda13.4-<commit12>`, where `commit12` is the first
  12 hexadecimal characters of its commit SHA. Both images must build before either floating tag
  moves. The pipeline no longer updates `latest` or `multi` tags.
- **Volumes.** `/models` holds artifacts, `/cache` the device profile measured on first start, and
  `/grafts/<model>/` optional [prompt grafts](docs/serving.md#prompt-grafts).
- **Verification.** The new image variants still need a complete build and GPU inference checks.
  Earlier image tests on RTX 3090 and RTX 5090 do not establish these variants' runtime behavior.

The container's command chooses what runs:

| command | runs |
|---|---|
| `serve`, `ninfer`, `perplexity`, `calibrate` `[args]` | that binary with any artifact and flags; `serve` listens on `0.0.0.0:8080` unless its flags or `--config` files give `host` or `port`. The image keeps the [configuration files](docs/cli.md#launch-configuration-files) in `/opt/ninfer/configs`: `docker run -w / ... serve --config /opt/ninfer/configs/<file>.ini` |
| `run <model> [profile]` (the default: `run qwen38-27b`) | the [launcher profiles](docs/maintainer/launcher-profiles.md) of `scripts/run.sh` for the official `qwen38-27b` and `qwen36-35b-a3b`, with its `NINFER_*` overrides (`-e NINFER_SPEC=mtp`, `-e NINFER_CONTEXT=131072`, ...) |
| `download <model>` | `scripts/download-model.sh` into `/models`: `qwen38-27b`, `qwen36-27b` or `qwen36-35b-a3b` |

The official Qwen3.8-27B with the measured `tuned` profile, on http://localhost:8080/v1:

```bash
docker run --rm -v "$PWD/models:/models" ghcr.io/iamwavecut/ninfer-all:cuda13.4 download qwen38-27b
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/iamwavecut/ninfer-all:cuda13.4 run qwen38-27b
```

[compose.yaml](compose.yaml) wires the GPU, the port and the volumes for the same:
`docker compose run --rm ninfer download qwen38-27b`, then `docker compose up -d`.
`NINFER_CUDA_TAG=cuda12.9 docker compose up -d` selects the Volta-capable image; Compose defaults
to `cuda13.4`. The published images use their multiarch binary automatically.
`docker build -t ninfer .` builds a separate Ubuntu image from source (`--build-arg ARCHS=86` for one architecture,
`--build-arg 'ARCHS=75;86;120a'` for the multi-architecture build).

</details>

## Building

```bash
scripts/fetch-media-sources.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target ninfer-serve ninfer-calibrate
```

The first command downloads the FFmpeg and dav1d releases that a Linux build compiles in.

`CMAKE_CUDA_ARCHITECTURES` takes one build from [supported GPUs](#supported-gpus), or a list of them
for a pipeline across cards of different architectures
([several architectures](docs/build-linux.md#cards-of-several-architectures-in-one-pipeline)).

<details>
<summary><b>Toolchains and options</b> — CUDA versions per target, guides</summary>

`120a` runs the `mma.sync` compatibility path, which the ternary route needs. CUDA 12.8 or newer builds
`75`, `80`, `86` and `89`, and `70` needs CUDA 12.8 or 12.9. A `120a` build needs CUDA 13.1 or newer except
13.2.0 and 13.2.1: those releases, 12.8 and 12.9 miscompile sm_120a kernels, and configure refuses them
([CUDA toolkits](docs/build-linux.md#cuda-toolkits)). The [Linux](docs/build-linux.md)
and [Windows](docs/build-windows.md) guides cover dependencies, the opt-in build options and the
launcher scripts; [tests](tests/README.md) and [benchmarks](bench/README.md) have their own guides.

</details>

## License

Apache-2.0, as upstream. The Bonsai artifact's weights come from PrismML, ProCreations and Qwen,
all Apache-2.0; its card lists the notices. The Qwen3.8-Flash-Next artifacts carry the Qwen
Community License 1.0 of their model.
