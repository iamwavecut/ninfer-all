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
models are under the Edge0 Community Model License 1.0 and the Qwen license of the base model; a
converted artifact is a conversion under those licenses.

## Convert

`--model` needs the base model's configuration and tokenizer files (`config.json`,
`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`) from
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next); the converter refuses a
folder whose vocabulary or merges differ from them. `--source pack=FOLDER` names the release folder,
which supplies the skeleton, the experts, the n-gram table and the MTP block.

```bash
# The n-gram table once, as an artifact of its own that every Edge0 model names by digest.
python3.11 -m tools.convert --model /path/to/Qwen3.8-Flash-Next \
  --recipe edge0_pack --source pack=/path/to/Edge0-120B-q3k \
  --components ngram --device cpu --out models/edge0-ple-fp8-table.ninfer

# Each model without the table, with its MTP block.
python3.11 -m tools.convert --model /path/to/Qwen3.8-Flash-Next \
  --recipe edge0_pack --source pack=/path/to/Edge0-120B-q3k \
  --components text,mtp --device cpu --name Edge0-120B-q3k --out models/edge0-120b-q3k.ninfer
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
conversion.

## Run

```bash
# One GPU with room for every expert (Edge0-120B-q3k on a 48 GB card).
./build/apps/ninfer-serve models/edge0-120b-q3k.ninfer --ngram-table models/edge0-ple-fp8-table.ninfer \
  --max-context 32768

# One GPU: the experts in pinned host memory, the most used of them cached on the GPU.
./build/apps/ninfer-serve models/edge0-120b-q3k.ninfer --ngram-table models/edge0-ple-fp8-table.ninfer \
  --expert-residency host --max-context 32768

# One GPU and little RAM: the experts stay in the artifact's files and stream into a GPU cache.
./build/apps/ninfer-serve models/edge0-120b-q3k.ninfer --ngram-table models/edge0-ple-fp8-table.ninfer \
  --expert-residency disk --max-context 32768

# Several GPUs dividing every layer: each GPU holds whole groups with their encoder, decoder and experts (Linux).
./build/apps/ninfer-serve models/edge0-145b-q3k.ninfer --ngram-table models/edge0-ple-fp8-table.ninfer \
  --devices 0,1 --tensor-parallel --max-context 32768
```

The encoder, the decoder, the labels and the scales stay in device memory under every placement:
3.74 GB for Edge0-145B (Q8_0) and 2.67 GB for Edge0-120B with Q8_0 projections, 1.42 GB with Q4_0.
Stored bytes before KV and workspace:

| Model | Device residency | Pinned host experts |
|---|---|---:|
| Edge0-145B-NVFP4 | about 58 GB | 47.6 GB |
| Edge0-145B-q3k | about 50 GB | 40.1 GB |
| Edge0-120B-NVFP4 | about 41 GB | 34.0 GB |
| Edge0-120B-q3k | about 36 GB | 28.6 GB |

The table is read 16 rows a token from its file, as for Flash-Next; `--ngram-residency ram` loads
all 51.2 GB.

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
- **Expert profiles**: host and disk experts fill the GPU cache at startup from Edge0's built-in
  routing profile, recorded on Edge0-120B-q3k over 48 chat, code, long-prompt and tool requests
  (all four releases share the router), not from Flash-Next's: a layer's 64 most routed experts
  in the two profiles have a third in common on average. `--expert-profile` replaces it with counts
  recorded by `--expert-profile-out`.
- **Drafting**: MTP (`--spec mtp`) uses the release's block; context lookup (`--lookup-ngram`) works
  as for Flash-Next. The releases have no DFlash2 drafter and no Vision tower.

`edge0-strata` applies the decoder to each routed expert's quantized output before the router
weights add them; NInfer applies it once per routed group to the weighted sum of that group's
experts. The two agree in exact arithmetic and differ by the rounding of that one quantization
point.
