# Qwen3.8-Flash-Next

NInfer runs Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`, 125B text parameters, about 6B
active per token) from ISTA-DASLab's GSQ-RCO GGUF releases and Unsloth's Dynamic v3.1 ones, converted
without requantization:

- [Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF):
  Q2_0 (37.6 GB), IQ2_XS (39.2 GB), IQ3_XXS (47.0 GB), IQ3_S (54.8 GB);
- [Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF);
- [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF): UD-IQ2_XS
  (47.0 GB) and UD-IQ3_XXS (54.8 GB) were converted and run. Of its other sizes, UD-IQ1_M stores
  IQ2_XS expert blocks, which have no CPU product, and UD-Q4_K_XL and the larger ones store Q5_1
  tensors, which have no kernel; none of those was converted.

Each GSQ-RCO release is two GGUF shards: the model, and the 28.8 GB n-gram table of the model's
per-layer embedding (PLE), byte for byte the same in every release, the Coder build's included. The
table shard of Unsloth's UD-IQ2_XS release holds the same rows (the same SHA-256 of the rows). A NInfer
model artifact describes the table it reads in its `ngram` component (the hash constants, the row
format and the SHA-256 of the rows) and either stores the rows too, as one self-contained file, or
leaves them to a table artifact of their own that every Flash-Next model can share. Nothing reads
the table at load; each token reads the 16 rows it addresses from the file, unless the table, or the
part of it a profile ranks most used, is loaded into RAM ([the n-gram rows](#the-n-gram-rows)). A
model stored without its rows takes them from `--ngram-table PATH`: the table the model names, or
any other table of its shape ([another table of the same shape](#another-table-of-the-same-shape));
without a table the engine refuses to start
([running without it](#without-the-n-gram-table) is an experiment, not a mode).

The published conversions store the models without the table, which is published once:

| Artifact | Size | |
|---|---:|---|
| [n-gram table](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3) | 26.92 GiB | IQ4_NL rows and an optional broad hot-row profile |
| [Q2_0](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3) | 38.23 GiB | Vision and shared-Q8_0 MTP |
| [IQ3_S](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3) | 54.24 GiB | Vision and shared-Q8_0 MTP |
| [Coder IQ1_M](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-NInfer-v3) | 30.76 GiB | 256 text experts per layer, Vision and shared-Q8_0 MTP |
| [Unsloth UD-IQ2_XS](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-UD-IQ2_XS-NInfer-v3) | 47.43 GiB | Unsloth Dynamic v3.1, Vision, shared-Q8_0 MTP and the DFlash2 drafter |
| [Unsloth UD-IQ3_XXS](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-UD-IQ3_XXS-NInfer-v3) | 54.66 GiB | Unsloth Dynamic v3.1, Vision, shared-Q8_0 MTP and the DFlash2 drafter |

The model runs with up to eight concurrent requests, a context cache of prompt prefixes, structured
output, images and video through its Vision tower (`--vision`, from an artifact converted with the
tower), and MTP speculative decoding (`--spec mtp`). The three published NInfer models above now
include Unsloth's shared-Q8_0 MTP block. The upstream GSQ-RCO GGUF sources omit it, so new
conversions must supply it separately; see [MTP](#mtp-speculative-decoding).

The Q4_K async prompt tiles (`gguf_moe_matrix/q4_k`) apply to Q4_K expert banks whose tables are
16-byte aligned (`MoeTable::aligned16`): device banks, a rank's share under tensor parallelism, and
the slot pools that host and disk experts' prompt chunks fill. The published Q2_0, IQ3_S and IQ1_M
releases have no Q4_K experts.

## Convert

`--model` needs only the HF checkpoint's configuration and tokenizer files (`config.json`,
`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`) from
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

```bash
# The model with its n-gram table, one self-contained file (--components text,ngram, the default).
python3 -m tools.convert --model /path/to/Qwen3.8-Flash-Next \
  --recipe qwen3_8_flash_next_gguf \
  --source gguf=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --source ngram=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --device cpu --rows-per-chunk 65536 \
  --name qwen3.8-flash-next --out models/flash-next-q2_0.ninfer

# Or the table once, as an artifact of its own, and each release without it.
python3 -m tools.convert --model /path/to/Qwen3.8-Flash-Next \
  --recipe qwen3_8_flash_next_gguf --components ngram \
  --source ngram=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --device cpu --rows-per-chunk 65536 --out models/flash-next-ngram-table.ninfer
python3 -m tools.convert --model /path/to/Qwen3.8-Flash-Next \
  --recipe qwen3_8_flash_next_gguf --components text \
  --source gguf=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --source ngram=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --device cpu --rows-per-chunk 65536 \
  --name qwen3.8-flash-next --out models/flash-next-q2_0.ninfer
```

Every conversion containing PLE reads the table shard once more to hash it (a minute or so for 28.8 GB), since
the model records the digest of the table it reads whether or not it stores the rows.

A release split by llama.cpp (`llama-gguf-split`) converts the same way. Unsloth's Dynamic v3.1
files of [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
have this layout: the first shard holds the metadata and no tensor, the next one or two hold the
model, and the last holds the table. `--source gguf=` names the first shard; the converter opens
the shards beside it that the name counts (`-00001-of-0000N.gguf`) and reads each tensor from the
shard that stores it. An absent shard is left out, so the table shard may lie elsewhere; a model
tensor that no present shard stores, a cut shard, a tensor stored twice and a `split.count` that
differs from the name are refused. The table shard is still the `ngram` source, and any release's
table shard serves: the rows are the same.

```bash
# Unsloth UD-IQ2_XS without the table, with the Vision tower and the shared-Q8_0 MTP block.
python3 -m tools.convert --model /path/to/Qwen3.8-Flash-Next \
  --recipe qwen3_8_flash_next_gguf --components text,vision,mtp \
  --source gguf=/path/to/Qwen3.8-Flash-Next-UD-IQ2_XS-00001-of-00003.gguf \
  --source ngram=/path/to/Qwen3.8-Flash-Next-UD-IQ2_XS-00003-of-00003.gguf \
  --source vision=/path/to/mmproj-BF16.gguf \
  --source mtp=/path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --device cpu --rows-per-chunk 65536 \
  --name qwen3.8-flash-next --out models/flash-next-ud-iq2_xs.ninfer
```

`--layers A..B` makes a qualification artifact from source layers A through B-1, using zero-based,
half-open bounds. It supports the HF, GGUF and native-Q2 recipes. The embedding, final mixer and
head remain; selected blocks and their activation uses are renumbered from zero, while source
lookups keep the original indices. `provenance.source_layers` records those indices. PLE moves
with its block and keeps the same table constants. If the range contains no PLE, the artifact
omits the `ngram` component and no table source or runtime override is needed. The Engine refuses
unused table options for that slice.

```bash
# Source blocks 3 (QSA) and 4 (GDN), with no PLE or table scan.
python3.11 -m tools.convert --model /path/to/Qwen3.8-Flash-Next \
  --recipe qwen3_8_flash_next_gsq_q2 --layers 3..5 --components text \
  --source gguf=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --device cpu --rows-per-chunk 65536 --out models/flash-next-layers-3-5.ninfer
```

A sliced model runs its shortened stack through the public Engine. Its outputs are for binding,
state and numerical qualification; they do not represent full-model quality or logits at the
same layers of a complete forward, whose input residual also includes the preceding layers.

`qwen3_8_flash_next_gsq_q2` imports the trained Q2_0 routed experts into `q2_g64_fp16` planes
with exact code and signed-scale words. Q2_0/Q8_0 shared projections retain their codes/scales as
native Q2/Q8; other shared block formats decode and round to BF16. Other projections and the MTP
block keep their GGUF representations. Native banks support device residency and host residency at
concurrency one; native disk residency is not implemented. A8 uses grouped integer MMA from 16
columns, decoding the stored native codes in registers without repacking the weights.

The native recipe is an explicit artifact choice; the GGUF artifacts stay the published
representation. On the held-out quick corpus (391,000 targets, context 32,768, BF16 KV, one RTX
A6000) the native-Q2 artifact scored 5.004501 against 4.991555 for the GGUF one (+0.26%, worst
window +4.3%), and scoring took 15% longer. Its expert Op is within a few percent of the GGUF Op at
decode widths with A8 activations, between 28% faster and 34% slower at prompt widths, and slower
everywhere with A16 activations.

`qwen3_8_flash_next` supports HF-source conversion: text routed experts Q4 gate/up and
Q5 down, shared experts and GDN/QSA projections Q8, embedding Q8 and head Q6. HC, router,
indexer, PLE, vision and small GDN gates stay in their direct formats. MTP routed experts use
Q4 and its other projections BF16. No sensitive-layer Q6 mask is selected without quality
measurements. Both text recipes accept the IQ4_NL GGUF table or HF BF16 n-gram shards via
`--source ngram=PATH`. HF rows become `fp8_e4m3fn_row_fp16`: 160 E4M3FN bytes and an FP16
scale per row. Full-model quality and memory requirements of this recipe are unmeasured;
full HF-to-Q4/Q5 artifacts and their model campaign are outside the current delivery scope.
The official FP8 checkpoint's individual expert matrices are supported alongside BF16 fused
banks. E4M3FN codes are multiplied by their stored BF16 or FP32 `weight_scale_inv` values in
128×128 blocks, then rounded to FP32 before the selected artifact quantizer runs. Despite
its name, `weight_scale_inv` is the dequantization multiplier. Source overrides resolve the
selected checkpoint's layout, including MTP. Unscaled FP8 banks are refused.
An FP8 text checkpoint still needs a separate supported table source: IQ4_NL GGUF or HF BF16
shards. Import of the FP8 checkpoint's own n-gram encoding is not implemented.

```bash
python3.11 -m tools.convert --model /path/to/HF-BF16-checkpoint \
  --recipe qwen3_8_flash_next --components text,ngram,mtp \
  --source ngram=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --device cpu --out models/flash-next-q4-q5.ninfer

# Write just the FP8 table from an already downloaded HF checkpoint.
python3.11 -m tools.convert --model /path/to/HF-BF16-checkpoint \
  --recipe qwen3_8_flash_next --components ngram \
  --source ngram=/path/to/HF-BF16-checkpoint --device cpu \
  --out models/flash-next-ngram-fp8.ninfer

# Fetch only the HF MTP tensors into a new directory, with bounded 8 MiB weight reads.
python3.11 -m tools.reference.fetch_slice --repo Qwen/Qwen3.8-Flash-Next \
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 --mtp --out /path/to/mtp-subset
```

The subset includes a config, safetensors index and SHA-256 manifest for files and individual
tensors. Pass `--source mtp=/path/to/mtp-subset` to the native or GGUF recipe to use it; the HF
MTP block then uses Q4 routed experts and BF16 elsewhere. A server that ignores byte ranges is
refused before a full shard is read. Failed copies retain a `.partial` file, never a published
safetensors file; use a fresh output directory after resolving a failed transfer.

An FP8 table (`fp8_e4m3fn_row_fp16`, 51.8 GB) from the HF BF16 shards is supported but not
published: it improved held-out perplexity by 1.25% (worse on both code domains) for 80% more
table storage than IQ4_NL, which stays the release representation.

`--components text,vision` (with or without `ngram`) adds the Vision tower from the release's
`mmproj-Qwen3.8-Flash-Next-BF16.gguf` (`--source vision=PATH`): the same tower as Qwen3.5/3.6
(27 blocks of width 1152, merging 2×2 patches onto the text model's 2,560), kept in BF16, 0.9 GB.
`--model` then also needs `preprocessor_config.json` and `video_preprocessor_config.json`.

`--components text,mtp` (with or without `ngram` and `vision`) adds the MTP block from one of
Unsloth's MTP GGUFs (`--source mtp=PATH`; [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF),
folder `MTP/`, a `shared-` file, which borrows the model's token embedding and head):

```bash
python3 -m tools.convert --model /path/to/Qwen3.8-Flash-Next \
  --recipe qwen3_8_flash_next_gguf --components text,ngram,mtp \
  --source gguf=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --source ngram=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --source mtp=/path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --device cpu --rows-per-chunk 65536 \
  --name qwen3.8-flash-next --out models/flash-next-q2_0-mtp.ninfer
```

The block keeps its matrices in the GGUF's blocks and its 512 experts, also beside the Coder
build's 256; its norms drop their stored `1 + w`, and its hyper-connection matrices keep the
`shared-Q8_0` file's Q8_0 blocks, which their kernels read. The `shared-Q8_0` block adds 2.8 GB to
the model.

The DFlash2 drafter ([DFlash2 speculative decoding](#dflash2-speculative-decoding)) is appended to
a converted artifact as its `dflash2` component, from [nazeshinjite's Q4_0 GGUF](https://huggingface.co/nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF)
of [incoai's drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) and incoai's
`config.json`. The artifact's other objects keep their bytes; the drafter keeps its Q4_0 blocks
and adds 339 MB:

```bash
python3.11 scripts/pods/attach_dflash2.py --base models/flash-next-q2_0.ninfer \
  --gguf /path/to/Qwen3.8-Flash-Next-DFlash2-Q4_0.gguf \
  --config /path/to/Qwen3.8-Flash-Next-DFlash2/config.json \
  --output models/flash-next-q2_0-dflash2.ninfer
```

Every matrix keeps the block type the release chose (see [GGUF block formats](gguf.md)); the expert
banks keep the exporter's expert-major layout, so one expert is one contiguous range of bytes. The
recipe undoes llama.cpp's exporter conventions as the Qwen3.8-27B GGUF recipe does (grouped GDN
value heads, `1 + w` norms, `A_log`, the head-interleaved query and gate). The n-gram table keeps
the release's IQ4_NL rows, and its component carries the hash constants they were written for; the
runtime derives them again from the model's configuration and refuses a table that disagrees
([another table of the same shape](#another-table-of-the-same-shape)).

## Run

```bash
# One GPU with room for every expert (48 GB for Q2_0, the 96 GB RTX PRO 6000 for IQ3_S).
./build/apps/ninfer-serve models/flash-next-q2_0.ninfer --ngram-table models/flash-next-ngram-table.ninfer \
  --max-context 32768

# Two 24 GB GPUs (two 32 GB for IQ3_S): every expert in device memory, one pipeline stage per GPU (Linux).
./build/apps/ninfer-serve models/flash-next-q2_0.ninfer --ngram-table models/flash-next-ngram-table.ninfer \
  --devices 0,1 --max-context 32768

# The same GPUs dividing every layer instead: GDN key heads, attention KV heads, routed experts (Linux).
./build/apps/ninfer-serve models/flash-next-q2_0.ninfer --ngram-table models/flash-next-ngram-table.ninfer \
  --devices 0,1 --tensor-parallel --max-context 32768

# One GPU: the experts in pinned host memory, the most used of them cached on the GPU.
./build/apps/ninfer-serve models/flash-next-q2_0.ninfer --ngram-table models/flash-next-ngram-table.ninfer \
  --expert-residency host --max-context 32768

# One GPU and little RAM: the experts stay in the artifact's files and stream into a GPU cache.
./build/apps/ninfer-serve models/flash-next-q2_0.ninfer --ngram-table models/flash-next-ngram-table.ninfer \
  --expert-residency disk --max-context 32768
```

A model converted with its table needs no `--ngram-table`. `ninfer` and `ninfer-perplexity` take
the same options, and the Docker image's `serve` command takes them with the files under `/models`
([launch recipes](recipes.md#qwen38-flash-next)).

| Option | Meaning |
|---|---|
| `--expert-residency device\|host\|disk` | expert banks in the stage devices' memory (default); in page-locked host memory that the expert kernels read across the bus; or left in the artifact's files, each layer's routed experts read into a device cache before they run |
| `--expert-cache-mib N\|auto` | with host or disk experts, device memory for the most used experts: `auto` (default) starts at half of what each device has free after loading less a margin, and once the warm-up has captured the CUDA Graphs and loaded the kernels requests run, and every sequence's snapshot is in place, adds what is still free beyond 384 MiB (the startup line reports the cache and the graphs); with host experts it also takes the prefill buffers and a prompt span's stack between prompts ([the expert cache](#the-expert-cache)); `N` is the cache's size, which the warm-up leaves alone; `0` disables the host-mode cache (disk mode needs one) |
| `--expert-misses staged\|mapped` | GGUF host experts: a decode or verification call copies each routed expert its device cache lacks into device memory by the copy engine while the cached ones run, into a cache slot when one is free or when the expert's routes beat the coldest cached expert the call does not route ([the expert cache](#the-expert-cache)), else into staging (`staged`, the default); or its expert kernels read the missing experts across the bus (`mapped`) |
| `--expert-dma-share F\|auto` | host or disk experts: fraction of a call's missing experts the GPU runs (copied to it), `0..1`; the CPU computes the rest from RAM while the GPU runs the cached ones. `auto` (the default) gives each call of GGUF experts the split whose copies and CPU products end soonest by the times measured on earlier calls, and copies every miss of native experts. Calls of up to eight tokens (decode steps, one request's verification) give the CPU any missing expert, calls of 9 to 64 tokens (several requests' verifications, short prompt chunks) those with at most eight pairs; wider calls run every expert on the GPU. GGUF CPU products are the GPU's bit for bit ([the CPU share](#the-cpu-share)) |
| `--expert-tail-skip N` | GGUF host experts with staged misses: a decode or verification call leaves out each routed expert its device cache lacks when every token of the call ranks it past place `N` (`1..9`) of its top ten, instead of copying it; the remaining experts keep their router weights (no renormalization). Lossy, off by default; prefill always runs every routed expert. On an RTX 3090 with Q2_0 and four short chat prompts, `7` decoded 9.0-17.6% and `6` 11.9-24.4% faster, copying 22-43% and 29-52% fewer expert bytes; perplexity over 5,115 tokens scored one token per call went from 4.667 to 4.710 with `7` and 4.688 with `6`, an order opposite to how much each leaves out, so that sample does not resolve the cost |
| `--expert-prefill-buffers 1\|2` | GGUF host experts: device buffers a wide prefill call copies its layer's uncached experts into; `2` alternates two, so the next layer's copies run while this layer's experts do, when half the free memory holds both. When half the free memory holds not even one, prompt chunks and verifications copy their missing experts through the staging in rounds instead. Default: `2` with the automatic cache, where both buffers hold cache slots between prompts, else `1`; with an explicit `--expert-cache-mib` the buffers take memory beside the cache. On an RTX 4090 with IQ3_S host experts the second buffer prefilled a 31,865-token prompt at 3,430 instead of 2,683 tokens/s and decoded within 2.4% of the one-buffer speed (one run each); on an RTX 3090 with Q2_0, before the buffers became cache slots, 8.5-10.5% faster at 5,462 tokens and 1.9-2.3% at 31,860 |
| `--iq1m-experts matrix\|vector` | GGUF IQ1_M experts of calls past eight tokens: ggml's integer matrix kernel once a call routes more than 16 rows to each expert on average for gate and up products and 6 for down products, the vector kernel below (default), or always the vector kernel in groups of eight columns; the two round the sums differently ([IQ1_M experts](deepseek-v41.md#iq1_m-experts-in-prompt-chunks)) |
| `--expert-cpu-threads N` | host or disk experts with a CPU share: `1..256` CPU threads, the calling one included; default is one per physical core the process may use (its affinity mask), at most half its CPUs (the mask and a container CPU quota); the rest stay with the threads that drive the GPU and serve requests ([the CPU share on one RTX 4090](#the-cpu-share-on-one-rtx-4090)). Each thread is pinned to a physical core and spins while requests decode |
| `--expert-cache-adaptive` | replace cold cached native experts between calls; off by default because it changes the CPU/GPU arithmetic partition |
| `--expert-profile PATH` | fill the expert cache at startup from counts recorded for the same artifact (`--expert-profile-out`); default: the built-in profile of the model family and its bank widths, where the build has one. Without the MTP block (no drafts, DFlash2, perplexity) the text layers' counts of a profile recorded with it |
| `--expert-profile-out PATH` | record every layer's expert routes after requests, for a later `--expert-profile` |
| `--expert-mirror DIR` | disk experts: copies of the artifact's files on a second drive; reads alternate between the drives by 1 MiB stripe of the file offset ([disk expert reads](expert-disk-reads.md)) |
| `--expert-direct auto\|on\|off` | disk experts on Linux: the NVMe drive writes the experts straight into device memory through NInfer's kernel modules; `auto` (default) when they are loaded and a self-test read passes, `on` fails startup otherwise ([disk expert reads](expert-disk-reads.md)) |
| `--ngram-table PATH` | the table artifact to read the n-gram rows from; required for a model stored without its table. Any table of the model's hash constants, row count and row width, in any row format ([another table of the same shape](#another-table-of-the-same-shape)) |
| `--ngram-residency disk\|ram\|ram-hot` | where the n-gram rows come from: the table's file, 16 rows a token (default); the whole 28.8 GB table in RAM; or the rows a hot-row profile ranks first in RAM and the rest from the file ([the n-gram rows](#the-n-gram-rows)) |
| `--ngram-io buffered\|direct\|mmap` | how rows are read from the file: positioned reads through the OS page cache (default), reads past it (`O_DIRECT`, `FILE_FLAG_NO_BUFFERING`), or copies out of a mapping |
| `--ngram-io-depth N` | row reads in flight (1 to 1024, default 64) |
| `--ngram-dedupe on\|off` | read a row that a pass addresses several times (a repeated n-gram) once and copy it to its other places (default `on`); the rows are the same either way |
| `--ngram-hot-profile PATH` | `ram-hot`: override the selected table's embedded profile with a `ninfer-ngram-profile` file |
| `--ngram-ram-mib N` | `disk`: cached rows and their index (default `0`, off); `ram-hot`: resident profile rows and their index (default 4096 MiB) |
| `--ngram-lock` | `ram`, `ram-hot`: lock the resident rows in physical memory (`mlock`, `VirtualLock`; needs the memlock limit or the privilege) |
| `--no-ngram-table` | run without the n-gram table: see [below](#without-the-n-gram-table) |
| `--kv-resident N\|all\|auto` | positions of each sparse-attention layer's KV a sequence keeps in device memory; a longer `--max-context` streams those layers ([streamed KV](#streamed-sparse-attention-kv)). `auto` (default): 32,768 with host or disk experts, all with device experts |
| `--rope-yarn`, `--rope-yarn-factor F` | YaRN past the native 262,144 positions, as for the Qwen3.5 family: at `--max-context` / 262,144 once the window passes it, or at a fixed factor `F` (`1..4`) for every position. It reaches every rotation the model makes, the sparse-attention layers', their indexer's and the MTP block's, and changes every answer, short ones too. Without it positions past 262,144 rotate unscaled. `--rope-scaling-factor` is not available for this model. The startup log names the factor, and the request log's `server_start` record reports the resolved rotation (`engine.rope`) |
| `--devices A,B,...` | one pipeline stage per GPU; layers are split so that every stage holds about the same stored bytes (`--stage-layers` overrides); with host or disk experts each GPU copies and caches the experts of its own stage's layers ([measured](#host-and-disk-experts-over-two-rtx-4090s)) |
| `--tensor-parallel`, `--tensor-split A,B,...` | with `--devices`: every layer divided over the GPUs, its GDN key heads, its sparse-attention KV heads and its routed experts (GGUF banks on the devices), each GPU keeping the hyper-connection stack, running a run of the shared expert and projecting its rows of the head; MTP and Vision on the first ([tensor parallelism](maintainer/pipeline-stages.md#tensor-parallelism-of-qwen38-flash-next)) |

With device experts every expert runs on the GPU. With GGUF host and disk experts, by default the
CPU computes a share of the missing experts of each call of up to eight tokens, and with host
experts of those with at most eight pairs in calls of up to 64 ([the CPU share](#the-cpu-share));
host experts draft with the artifact's DFlash2 drafter, four tokens a round, or without one with its
MTP block, three ([DFlash2](#dflash2-speculative-decoding), [MTP](#mtp-speculative-decoding)); and
the sparse-attention KV past 32,768 positions streams from host memory ([streamed
KV](#streamed-sparse-attention-kv)), which leaves the device memory to the expert cache.
Native-format experiments are retained for reference. The existing GGUF artifacts support both host
and disk residency; disk residency uses a CLOCK device cache. Disk residency relies on the OS page
cache for repeated file reads; the engine keeps no RAM cache of experts of its own, only the device
cache and bounded transfer buffers.

### The expert cache

With host experts the device cache's slots follow the expert tier that every MoE family with host or
disk experts shares (`core/expert_tier.h`), one pool of slots per layer. Each decode or verification
call adds one to the count of every (token, expert) pair it routes; every four decode passes a round
swaps up to 96 of the most routed experts the cache lacks into the slots of the least routed ones it
holds, best gain first, each only when its count reaches the victim's plus 1.5 (and at least 2),
then multiplies every count by 0.7. Each pass until the next round starts up to 24 of those swaps (a
quarter of the limit), best gain first: their victims leave the tables at once, and their
candidates' copies run on a side stream beside the draft, ahead of the next verification, whose
missing experts would otherwise queue on the bus behind a whole round. The candidates' table entries
move to their slots once the copies have landed. A miss the GPU copies anyway takes a slot at once
when one is free, or under the same rule against the coldest cached expert its call does not route
(a prompt chunk of more than 64 tokens, which copies into a slot pool, only a free one); a miss the
CPU computes moves no expert bytes and enters only through a round. Prompt chunks feed the
statistics only. A routing profile (`--expert-profile`, else the built-in one of the model family)
fills the slots at startup and, ranking its counts over every layer, gives each text layer the slots
its most routed experts take; the MTP block keeps an even share. A run without the MTP block (no
drafts, DFlash2, perplexity) takes the text layers' counts of a profile recorded with it. Strata
(MIT) measured this tier and its constants on Qwen3.8-Flash-Next.

With the automatic cache and host experts, the device buffers that prompt chunks copy a layer's
uncached experts into (`--expert-prefill-buffers`) and the stack of a prompt span (40 KiB a span
token: 0.63 GiB for sixteen 1,024-token chunks) are cache slots between prompts: every layer
gets the same share of them, free until misses and rounds fill them. A prompt pass takes them
back: the most routed of the experts they hold move into the slots of colder ones in the cache's
own memory, the rest leave, and the copies into the buffers wait for the moves. The next decode
or verification pass returns them zeroed and refills them at once with the most routed experts the
cache lacks (those the tier counts at least twice), copied on the side stream as a round's are and
serving once a layer's copies have landed, so the decode after a prompt does not wait for many
rounds to fill them again; buffers' slots left over stay free. On an RTX 4090 with the IQ3_S
release's host experts the 256 tokens after a 31,865-token document decoded 1.052 times as fast as
without the refill (geomean over the nine `--kv-dtype` formats, 0.992-1.118); after short prompts
the ratio stayed within the run-to-run spread (code 0.993, prose 1.011).

With host experts the GPU holds only the dense weights (3.7 GB for the Q2_0 release), the
per-request state and the expert cache; the host needs the expert banks in page-locked memory
(34 GB for Q2_0). With disk experts the host needs no copy of the banks at all: the expert cache
reads the missing experts from the artifact's files through the OS page cache, eight reads in
flight, into a 256 MB page-locked staging ring, so the page cache keeps whatever the system can
spare and the rest comes from the disk. A read of the artifact's files that fails while the Engine
runs (an expert or an n-gram row) fails the requests of that round with `unavailable`; the expert
cache forgets what it held, the Engine counts an `engine_recoveries` and serves the queue on.

Native host banks copy every miss to the GPU by default (`--expert-dma-share 1`). Values below one
let the CPU compute a share of the missing experts; this mode is experimental: on one RTX 3090 with
an 8-core EPYC 7302P it decoded 3.1 tok/s at a share of 0 and 5.3 at 0.5 against 31.4 at 1, and its
logits stay within top-1 but are not numerically qualified (maximum relative L2 0.32-0.60 against
resident experts). A GGUF bank, including a GGUF MTP companion beside native text weights, keeps
its GPU route. Linux registers native weight objects separately; Windows uses pageable weights and
bounded pinned staging. Use `--max-concurrency 1` for native host execution. Under WDDM (Windows,
and WSL2, recognised by `/dev/dxg`) a device's copies may queue behind one another whatever their
streams, so the tables, the CPU/GPU partition and the CPU's sum that a decode graph takes after it
waited for the host reach the device through mapped host memory and a copy kernel; a queued copy
there would hold back the expert copies that end the wait. The bytes are the same either way.

### The CPU share

With GGUF host experts, CPU threads compute part of a call's missing experts from their stored
blocks while the GPU copies and runs the rest (`--expert-dma-share auto`, the default, or a fixed
share below `1`); the GPU then adds the CPU's products into the same fixed-point sum as its own.
The CPU reproduces the vector kernel, which rounds each 32 activations' q8_1 scale to binary16
where the matrix kernel keeps it in FP32 and sums in another order, so it takes only experts that
the GPU would run on the vector kernel in every placement: in a call of up to eight tokens (a
decode step, one request's verification) any missing expert, and in a call of 9 to 64 tokens (a
round of several requests' verifications, a short prompt chunk) those with at most eight pairs,
which `ops::moe_experts_gguf` runs on the vector kernel there while the others take the matrix
kernel. Wider calls (prompt chunks) run every expert on the GPU with the matrix kernel. A call's
result therefore does not depend on where its experts reside. A fixed share gives the CPU the
experts with the fewest pairs. `auto` gives each call the split whose copies and CPU products end
soonest by two straight lines fitted to earlier calls' copy and CPU times (a sample counts at most
four times or a quarter of its line's value); a side left unused for eight calls takes one expert,
so both lines follow the machine.

The vector kernel computes a few-pair expert's products more slowly than the matrix kernel, so the
division costs cached and device experts some speed. On an RTX A6000 (two Xeon E5-2699 v4 sockets
with a 10.56-CPU quota, PCIe 3.0 x16, the Q2_0 release, `--max-concurrency 8`, 2026-11-11), against
the matrix kernel for every expert of those calls (the CPU then shared calls of up to eight tokens
only): with host experts and a 12 GiB cache filled from the text layers' routing profile, 48-token
perplexity windows scored 77.3-78.9 instead of 44.6-44.9 tokens/s, eight requests under
`--spec mtp --draft-tokens 3` decoded 98.4-103.6 instead of 59.9-62.1, and a fresh 44- to 46-token
prompt's first token came after 176-242 instead of 277-438 ms; with the automatic cache (34.1 GiB)
the eight requests decoded 174.4-174.6 instead of 168.0-176.9 tokens/s and the first token came
after 116-118 instead of 112-114 ms; device experts scored 267.7-269.6 instead of 288.2-289.8
tokens/s on the 48-token windows and decoded 176.3-181.2 instead of 173.3-177.3 under MTP. On
48-token windows over 40 KB with the 12 GiB cache, `auto` scored 70.5 tokens/s and the fixed shares
`1`, `0.75` and `0.5` 43.4, 51.4 and 65.3. An expert the CPU computed takes no slot and crosses no
bus; the expert cache's rounds decide whether it enters the cache ([the expert
cache](#the-expert-cache)). The CPU threads are pinned one to a physical core and spin while
requests decode, so a call's products start without a wake-up; the thread that drives the GPU is
left unpinned.

The CPU computes the GPU's decode products bit for bit, so outputs, log probabilities and context
images do not depend on the share, on `auto`'s choices or on the thread count. The activations are
quantized as the GPU quantizes them (q8_1: per 32 values `d = amax / 127`, codes `round(v / d)`,
`d` rounded to binary16). A 32-value slice gives the exact integer dot of its codes and the
activation's, and its product is `(factor * d) * dot`, each step rounded, with the slice's exact
factor (the block's binary16 multiplier times its integer scale); IQ2_S's and IQ1_M's are
`d * fma(factor0, dot0, factor1 * dot1)` over their two halves (IQ1_M's codes are eight times its
values, `8c - 7` or `8c - 9`, and its factors `d (2s + 1) / 8` carry the eighth), Q6_K's
`d * (factor0 * dot0 + factor1 * dot1)`, Q4_K's and Q5_K's `(factor * d) * dot` less the
block's minimum times the slice's activation sum (q8_1's `s`), and IQ1_S's
`factor * fma(d, dot, (delta - 1) * s)` over its codes 0 to 2. Slice `s` joins lane `s mod 32` in
slice order, and the 32 lanes meet as in a warp's butterfly (16, 8, 4, 2, 1 apart). SiLU (and
Gemma 4's GELU) is `ops/common/silu.h` on both sides, built from correctly rounded operations only,
and the middle is rounded to BF16 before it is quantized for the down product. No compiler fuses
these steps on either side: the GPU kernels spell them with `_rn` intrinsics and the CPU sources
build without floating-point contraction. Q2_0, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ1_S,
IQ1_M, IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS have CPU products, in AVX2 (with FMA and F16C) where the CPU has it and
in portable C++ elsewhere, with the same bits.

### The n-gram rows

A pass reads the 16 rows each of its tokens addresses (1,440 bytes a token in IQ4_NL) as soon as it
has hashed them, `--ngram-io-depth` reads at once, and uploads them just before the PLE layer
(block 1): the GPU embeds the tokens and runs block 0 while the rows are read, and waits only if
the read takes longer. A prompt span reads the rows of all its chunks in one batch the same way,
while every chunk passes block 0. `--ngram-io` chooses how the file is read. `buffered` (the default) goes
through the OS page cache, which then keeps a 4 KiB page for each row it read; `direct` bypasses the
OS page cache, so every row-cache miss reads from the drive; `mmap` copies the rows out
of a read-only mapping of the file. A row the pass addresses several times (a repeated n-gram) is
read once and copied to its other places (`--ngram-dedupe off` reads every place); the pass then
reads its rows in file order. On an RTX 4090 with the Coder IQ1_M release's experts in host memory,
the table dropped from the page cache before each server started and four rounds, a 29,613-token
prompt read 28.8 MiB of rows instead of 40.7 MiB in 29% fewer read calls, an image's prompt 0.1 MiB
instead of 0.6 MiB in 87% fewer, and prompts of 92 to 29,615 tokens prefilled 0.2 to 2.1% faster
than with `--ngram-dedupe off`, within the spread of the rounds.

`disk` keeps recently read rows in a four-way CLOCK cache. `--ngram-ram-mib` bounds its
allocation including row tags and CLOCK state, independent of the table's size. The cache is off
by default; for example, `--ngram-ram-mib 4096` enables a 4 GiB budget. Its empty index and row storage use demand-paged
memory; the Linux cache avoids huge pages so one short row does not fault 2 MiB. The budget is
allocated capacity, not current RSS; buffered and
mapped reads can also occupy the OS page cache. Failed batches admit no rows.

Once a prompt step (a chunk, or a span) has read its own rows, it hints the rows of the
Engine's next step, which are then read while the GPU runs this one. MTP hints its anchor
and both highest-logit candidates at each draft step. An auxiliary CUDA stream copies each
step's two tokens and hashes private contexts while later draft steps run. Verification also
hints both candidates after each actual verified prefix; predictions do not change the next
column's context. Linux buffered/mapped readers
issue `POSIX_FADV_WILLNEED`; Windows mapped readers use `PrefetchVirtualMemory`. These are advisory
reads, with duplicate rows and shared pages coalesced. A worker thread issues a batch of more than
256 page ranges and drops a batch that arrives while it still issues the previous one. Direct I/O
and Windows buffered I/O start one bounded speculative batch in separate staging; busy workers
skip extra hints. Only complete successful
batches can supply demand reads, and hints alone admit no rows into the persistent cache.

Buffered/direct row misses use a bounded OS queue: io_uring on Linux and overlapped reads with
IOCP on Windows. One worker drives the queue while device execution continues; direct reads use
at most `--ngram-io-depth` aligned bounce buffers. A failed batch drains its outstanding reads
before returning and admits no rows to the cache. If Linux disables io_uring, the reader uses
positioned-read workers; a buffered batch of at most 1,024 rows then first reads, on the engine
thread, the rows the page cache holds (`RWF_NOWAIT`, or `mincore` over a mapping of the file where
the file system refuses that flag, as overlayfs does), and wakes only as many workers as rows
remain. On an RTX 4090 host in a container without io_uring, the 64 rows of a decode verification
took 3.6 ms per round across 64 woken workers; read on the engine thread, the GPU's idle time
before a verification fell from 2.4 ms to 0.8 ms a round.
Bulk RAM/hot-set loading and mapped reads retain their existing paths.
The Windows reader paths have not been run.

On a serving workload with one RTX 3090, the table's pages evicted before the first request, the
PLE layer waited 0 to 4 microseconds per request for its rows with buffered or direct reads. The
row cache stays off by default: with a 4 GiB cache the first requests took 10-11% longer, and
repeated requests that hit every row generated no faster.

`--ngram-residency ram` reads the whole table into RAM at startup (2 MiB pages where Linux offers
them), and `ram-hot` reads only the rows a hot-row profile ranks most used, as many as
`--ngram-ram-mib` holds beside their index (a bit per table row, 40 MiB), and reads the rest from
the file as `disk` does. A row depends on its token and the two before it and nothing else, so a
profile is counted from text alone, without running the model:

The selected table artifact may carry a profile in `ngram.resources.hot_profile`.
`ram-hot` uses it when `--ngram-hot-profile` is absent; an explicit file overrides it.
An external table supplies its own profile, and a table with no embedded profile still needs
the explicit file. Other residencies leave the resource unread. Conversion embeds a checkpoint's
`ngram.hot` file when present, or the file selected by `--resource ngram.hot=PATH`; the hash,
row bounds and uniqueness must match. This does not change the table digest or its row bytes.

Chat JSONL documents use the artifact's Frontend, including tool definitions, tool calls and the
assistant generation prompt. Set `"enable_thinking": false` or `true` per document when the corpus
uses a fixed thinking mode; without it the chat template chooses its default. Invalid messages
are rejected rather than omitted from the profile.

```bash
# Count the rows a corpus addresses: UTF-8 files, or .jsonl with a "text" or chat "messages" per line.
./build/apps/ninfer-ngram-profile models/flash-next-q2_0.ninfer --out models/flash-next.hot \
  corpus/*.jsonl
# The share of a held-out corpus's row reads the profile's leading rows serve, per RAM budget.
./build/apps/ninfer-ngram-profile models/flash-next-q2_0.ninfer --evaluate models/flash-next.hot \
  heldout/*.jsonl
./build/apps/ninfer-serve models/flash-next-q2_0.ninfer --ngram-table models/flash-next-ngram-table.ninfer \
  --ngram-residency ram-hot --ngram-hot-profile models/flash-next.hot --ngram-ram-mib 4096
```

`--ngram-lock` keeps the resident rows in physical memory. `/stats` reports the rows the passes
read, how many RAM served, the read latency and the time the PLE layer waited
([Stats](serving.md#structured-request-log)), and `ninfer` prints them after its answer.

The published table carries a broad hot-row profile: 26,223,284 ranked rows counted from
3,711,509 tokens sampled with equal per-group ceilings from four Common Corpus shards (prose, web,
press, science) and one github-code-clean shard. On 1,103,918 held-out tokens of the same sources
the leading rows serve this share of row reads per RAM budget:

| RAM budget, MiB | 256 | 512 | 1,024 | 2,048 | 4,096 and more |
|---|---:|---:|---:|---:|---:|
| Held-out row hits | 33.8% | 38.5% | 42.9% | 48.5% | 49.6% |

Across the 239 language and domain groups with at least 2,048 held-out tokens the hit rate at 4 GiB
runs from 20.5% to 95.5% (median 49.4%); Japanese books have the lowest coverage. Hits measure row
reuse, not speed or quality. Edge0's FP8 table carries the same profile, since one hash addresses
both tables' rows.

### Another table of the same shape

A model reads any n-gram table artifact of its hash constants, row count and row width, in any row
format the runtime decodes (IQ4_NL, FP8 with an FP16 row scale, BF16): the table its `ngram`
component names, or another one. The engine checks the table's shape against the model's
configuration, not its values; for another table it warns at startup with both tables' formats and
digests. A table of the same shape with other values starts too and degrades the output, as a wrong
table would.

Flash-Next's IQ4_NL table and [Edge0's](edge0.md) FP8 table are two quantizations of one BF16
table, so each model family reads either. Over three of the HF checkpoint's 128 BF16 table shards
(7.5 million rows at the start, the middle and the end of the table, the 90 zero pad rows left out),
every IQ4_NL row has a cosine of at least 0.9939 with its FP8 row and of at least 0.9945 with its
BF16 row; each BF16 row against the next, unrelated one has a median cosine of 0.02 to 0.04, none
of the 7.5 million pairs reaching 0.9.

On one A40 (the CUDA 12.8 `sm_86` build, November 13, 2026), the first 40 KB of `wikitext/00.txt`
with INT8 KV and device experts, the GSQ-RCO Q2_0 release scores 2.542153 with its IQ4_NL table and
2.531971 (0.40% lower) with Edge0's 48.38 GiB FP8 table; Edge0's figures are in
[its guide](edge0.md#with-flash-nexts-table). Q2_0's greedy answer to the sky prompt keeps its
content and changes its wording with the FP8 table; it decodes at 87.6 and 87.5 tokens per second.

### Without the n-gram table

The model was trained with its n-gram embedding, so the engine refuses a model whose table it
cannot find. `--no-ngram-table` starts it anyway, without the PLE injection (exactly what an
all-zero table gives), and warns at startup: this is a non-standard, experimental mode with no
practical use. On the Q2_0 release it nearly doubles WikiText-2 perplexity, 2.66 to 5.01 over the
fourteen windows of the comparison below; short factual answers survive, but nothing measured
improves.

### Streamed sparse-attention KV

Each query of a sparse-attention layer reads 512 blocks of four positions that the indexer selects
and its own incomplete block, so most of a long context's KV sits unread at each step. With
`--kv-resident N` below `--max-context` (by default 32,768 positions with host or disk experts) a
sequence keeps each of the 12 layers' whole KV in pinned host memory, addressed by the GPU, and the
device holds a pool of N / 4 blocks per layer (at least 513 for each token of a verification
window). Before each decode or verification call's attention, a kernel looks up every block the
queries read, claims the missing ones, gives each a pool slot by a CLOCK sweep (a slot the call
uses is never taken), copies them across the bus from the host copy and re-points the table; the
blocks the call wrote are copied again when it reads them, so no slot holds bytes older than the
host copy. A prompt chunk reads its layer's whole history from a device staging copy, refreshed
from the host copy once per span of chunks. The attention reads the same bytes either way: a
streamed and a resident sequence give the same logits. The device memory saved goes to the expert
cache. The residency scheme follows Strata's `kv_stream` (MIT).

### Sequences, KV and the context cache

`--kv-dtype` stores the KV cache of the 12 sparse-attention layers in any of the ten formats the
Qwen3.5 family uses. A position costs 24 KiB in `bf16` (the default), 18.4 KiB in the lossless
`bf16p`, 12.4 KiB in `int8`, 12.1 KiB
in `fp8`, 9.6 KiB in `rk8v4`, 9.4 KiB in `k8v4`, 6.8 KiB in `nvfp4`, 6.6 KiB in `rk4v4` and
`rk4v4-e8`, and 5.1 KiB in `rk2v4-e8`; the indexer's pooled keys stay FP32 beside it, another
1.5 KiB a position in every format. The sparse attention decodes each row it reads, keys in the
rotated basis the cache stores them in. What the quantized formats cost in quality has not been
measured on this model yet.

`--max-concurrency N` (one to eight) runs that many requests at once, each on its own sequence with
its own KV and recurrent state, so every sequence costs device memory (the KV of `--max-context`
positions, 24 KiB a position in BF16 and less in a quantized `--kv-dtype`, plus 74 MiB of recurrent
state and 74 MiB for the snapshot of it that the context cache and branched reads take, allocated
at startup). Requests are admitted
in arrival order. Prompts prefill a step at a time, each step from the prompt with the fewest tokens
left, so a short or cached request is not held behind a long prompt for its whole prefill (a prompt
passed over eight times goes next, and a prompt with media, once begun, finishes first). A step is
one chunk, or with host or disk experts on one GPU a span of up to eight chunks of the prompt
(sixteen under `--max-concurrency 1`, where no other prompt waits behind a step), media
or not (see [Execution](#execution)); while other requests decode, a span is two chunks at most,
since they wait for the whole step. Before a step, while requests are decoding, they run
`--decode-rounds-per-prefill` rounds (by default the chunk size over 64, 16 at the default chunk)
for each chunk's worth of prompt the step computes, at least one, so a short prompt gets in after
one round and the streams keep their share during a long one; each round is one batched pass whose
experts read their weights once for the whole batch, so the batch costs little more than one token
while the experts dominate the step. The mixers' projections and the steps without state (the
Gated DeltaNet and sparse-attention input projections, gates, norms and rotation, and their output
gates and projections) also run over every request's columns at once, in calls of up to eight
columns; only the convolution, recurrence, KV append, indexer and attention run request by
request, on each request's own state. A vector product gives a column the same bits at any width
up to eight, so a batched round computes what the requests' separate rounds would. On an RTX 3090
with host experts (IQ3_S, 2026-11-04), eight requests released at once (about 1,100 prompt tokens
and 128 answer tokens each) finished 2.0-4.2% faster with MTP over the nine `--kv-dtype` formats,
and 11.5% faster without drafts (BF16 KV). A speculative round counts as the tokens
its requests fed on average, so drafting streams get as many tokens per chunk as plain ones and
hold a prompt back no longer.

With host or disk experts on one GPU, a step of a prompt without media or branched reads also
takes the next chunks of the other such prompts that are prefilling, in the same order, while they
fit the span: their chunks reach each layer together, so the layer's uncached experts cross the
bus once for all of them. A step that does not end its prompt takes whole chunks, so each prompt
runs the chunks it would run alone and computes the same bits. A prompt that would resume from a
prefix another prompt of the step is about to keep waits for it, and a step that passes a shared
prefix (below) runs alone. On the RTX 3090 above (host
experts), eight requests got their last first token after 6.2-6.7 s instead of 19.3-20.9 s
(9.5-9.7 instead of 22.8-23.7 s with BF16 KV) and finished 1.77-1.94 times as fast with MTP over
the nine formats and 1.78 times as fast without drafts; two and four requests finished 1.37 and
1.50-1.56 times as fast (BF16 KV).

With the context cache on (the default), a sequence keeps its state when its request ends, and a
snapshot of its recurrent state where the prompt's last user turn closes (or at the prompt's end
when the template marks no turn), 74 MiB on the device: a later prompt that continues what the
sequence holds resumes from its live state, and one that repeats the prompt up to that point (the
next turn of a chat, which renders the previous answer without its reasoning) resumes from the
snapshot.
A turn that a stop string ends resumes where its visible text ends. The sequence keeps only the
tokens before the one that begins the stop string's match: a round that holds such a token commits
the tokens before it, and the next step reads the rest. A match that stays open keeps a second
snapshot of the sequence there (another 74 MiB a sequence, with the context cache), which the turn
goes back to when the stop string ends it in a later step, so the next prompt continues that state.
Either way only the new tokens are prefilled; the response's prompt summary reports how many were
reused. A request goes to the free sequence that holds the longest such prefix of its prompt.
A prefix with images or video is the same only with the same media at the same places (each
image's or video's content digest, grid and timestamps); a prompt that resumes past them
encodes only its media after the prefix.
`--no-prefix-reuse` (ninfer-serve) prefills every prompt from scratch, without the step ends below
or at the turn closure.

What a sequence gives up for a new request (its turn-closure snapshot and the state it ended in,
each of 128 tokens or more, the end of a disposable request such as a decision read excepted) is
kept as an image in host memory, and so is the state where the frontend marks a prefix other
requests share (the end of the leading system messages, of the tool declarations, a declared
cache boundary, 128 tokens or more): a prompt step ends there, or with host or disk experts on one
GPU, where the tokens on each side fill more than a narrow chunk (64 tokens, 8 with
`--expert-tail-skip`), a span runs on past it in the chunks two steps split there would run and keeps the state
there as it passes, so the experts cross the bus once for both sides
([measured](#media-conversations-on-one-rtx-4090)). Requests with the same prefix resume after it
on any sequence, also when they arrive together, since a prompt that has not started prefilling
looks for images again at its first step. An image is the paged KV and pooled keys
of its positions, its recurrent state, and the MTP block's, about 114 MB plus 26 KiB a position in
BF16 KV. The images live in 8 MiB pages of pinned memory that stay pinned for later images: the
pages of two images of a whole context (within a quarter of the budget) are pinned at startup.
Once the Engine has had nothing to run for 50 ms, it pins free pages back up to that many in 64 MiB
chunks, within the budget; a request that arrives meanwhile waits for the chunk in progress (about
32 ms, [measured](#media-conversations-on-one-rtx-4090)), and the store pins pages itself only when
the free ones fall short. Pinning while a request waits is slow where the
experts and the page cache fill the host's memory: on an RTX 4090 host with IQ3_S host experts,
pinning the two images of a 32K-token prompt held a short prompt after it for 2.3 s, and faulting
in pageable memory for them 1.8 s; into the pages pinned at startup they copy in 15 ms, and the
short prompt's first token came 0.45 s after it arrived (2.9 s before). Pinning the startup
pages (2 GiB with the default 8 GiB budget) adds about 2 s to startup. The other families pin
their whole store at startup and never pin while a request waits. A prompt that continues a kept prefix further than any free
sequence resumes from its image, copied back into a sequence. `--host-kv-mib` (or `--host-cache-mib`) is the budget, 8 GiB by
default; past it the least recently used images go to the disk tier with `--disk-kv-path DIR`
(`--disk-kv-gib`, 64 by default), one file each under a directory per artifact, KV format, MTP
draft width and n-gram table (its SHA-256, or none), or are dropped without one. A different verify
width or another n-gram table can change sequence state, so those runs cannot restore another
profile's images.
A request that does not draft (a decision read) runs its prompt without the MTP block, so its
sequence's snapshots and images carry no MTP state and serve only requests that do not draft.
With `--disk-kv-restore` a
prompt also resumes from an image on
disk, which a later run finds there too.

Structured output (`--structured-output` for the server, `--json`/`--json-schema` for the CLI) works
as for Gemma 4 ([Structured output](gemma4.md#structured-output)): the grammar's token mask applies
to every sampled token, it is computed on a thread of its own while the step runs, a row whose mask
is late samples once it is ready without holding the others, and a request drafts while its masks
take at most 4 ms. So do
[token log probabilities](serving.md#token-log-probabilities): a request that asks gathers each
sampled token's top 20 from the same logits before sampling.

`--vision` loads the Vision tower of an artifact converted with it (0.9 GB of BF16 weights on the
first device, beside the token embedding) and takes images and video as the Qwen3.5 family does:
the frontend renders the media tokens, the tower encodes the prompt's media before its first
chunk, and their merged embeddings replace those tokens' embeddings. A media prompt rotates its
positions on the three RoPE axes the frontend computes (text positions on all three, then each
later token at its index plus the prompt's offset); it prefills in a pass of its own. The context
cache keeps media prompts as it keeps text ones: the next turn of a conversation with images
resumes past them, and the tower encodes only the media after the resumed prefix, whose merged
tokens are what `--vision-max-merged` limits. The MTP block follows media prompts too, so their
requests draft: the cell before a media token takes that token's merged embedding, and the cells
rotate at the target's positions, on the three axes over media.

### MTP speculative decoding

`--spec mtp --draft-tokens N` (1 to 15) decodes with the MTP block of an artifact converted with it
([Convert](#convert)); without the block the engine refuses to start. Without `--spec`, an artifact
with the block and host experts drafts up to three tokens a round with `--draft-min-p 0.5` (a
verification window's routes share the layer's missing experts, which the CPU computes while the
cached ones run; a round of several requests verifies at most 64 tokens,
[below](#mtp-with-several-requests)), unless it also carries the DFlash2 drafter, which then drafts
on one device ([DFlash2](#dflash2-speculative-decoding)); with disk experts it decodes without
drafts ([below](#mtp-with-several-requests)), and `--spec none` decodes without drafts. An
[Edge0](edge0.md#drafting) artifact also drafts with device experts when the engine serves one
request at a time. Decoding
requests then run rounds: the MTP block (one more sparse-attention layer with its own 512-expert
MoE, its own final mixer and the model's head) drafts N tokens from the stack the model left at the
request's last position (the head's argmax at each step; with `--gumbel-sampling` a sampled
request's draw under the target's keyed noise, [coupled draws](cli.md#speculative-decoding)), the
model verifies the last sampled token
and the N drafts in one pass, and the acceptance keeps the drafts while they equal the model's own
draw at each position, then commits the model's draw at the first difference (or after the last
draft). The acceptance samples as the request does (greedy, temperature and top-k/top-p/min-p,
presence and frequency penalties counting the drafts kept before each position, the grammar's masks,
logprobs from each position's own distribution), drawing each position with the request's seed and
that position as the key, the key plain decoding uses there. Every committed token is thus the
model's own draw whatever was drafted. Verification width and batch composition can still change
floating-point reductions, so neither greedy nor sampled answers are promised to match plain decode
byte for byte. The kernels are qualified against independent mathematical oracles; exact repetition
holds the execution configuration fixed. Acceptance is bounded by the remaining output and
thinking-token budgets, including the correction or bonus token.

`--draft-min-p P` optionally ends a round's drafting at the first draft whose absolute
probability over the public vocabulary is at or below P. The round verifies the drafts up to and
including that one, so every round verifies at least one; P=0 disables the calculation and P=1
drafts and verifies one. A batch drafts until each of its requests has reached such a draft (or the
last step) and verifies each request's own selected prefix, the rows packed at their own widths
(all at the largest one when a request gathers logprobs). This is distinct from the sampling
`--min-p` filter. With a floor each draft step runs from its own CUDA Graph and the host reads the
step's probabilities before it starts the next, so `drafted_tokens` counts the steps that ran;
each verification width up to 8 has its own CUDA Graph. A round of several requests runs without
graphs; the graphs of a single request's passes are captured at startup and shared by the
sequences. The floor is part of the disk context-cache profile. `--spec mtp` keeps the full window
unless `--draft-min-p` is set; the automatic drafting uses 0.5 unless `--draft-min-p` sets another
floor (0 keeps the full window). On an RTX 4090 with the IQ3_S release's
host experts (EPYC 7V13, 32-CPU quota, 64K context; 256 greedy tokens after a code prompt, a prose
prompt and a 31,865-token document, once per `--kv-dtype` format) three drafts with the floor 0.5
decoded 1.040 times as fast as without it (geomean of 27 pairs, 0.953-1.131, slower in 5): a round
drafted 2.48 tokens instead of 3 and committed 2.27 instead of 2.31.

The verification leaves the sequence's state where it was: each Gated DeltaNet layer records its
transitions and the commit replays the kept ones into the state (the Qwen3.5 family's ReplaySSM
fold), the sparse-attention indexer's tail and the PLE convolution history advance over the kept
positions from what the verification recorded, and the n-gram context is hashed again over them. The
MTP block follows every token the model takes in (prompt chunks too), with a KV cache of its own the
size of one sparse-attention layer's (2 KiB a position in BF16), so a request can draft as soon as
its prompt is in, a prefix the context cache restores included. Following a token runs the block's
attention, which leaves its KV and indexer state, and its router, whose routes the expert cache
counts, but not its experts: only a draft reads the stack they leave. Up to eight requests run their
rounds together, one MTP pass for all of them per draft and one verification pass. A request decodes
without drafts when its prompt ends with a media token, near the end of its context, with one token
left in its output or thinking budget, and while a `--post-thinking` request still reasons. N-gram
copy proposals (`--ngram-draft-tokens`) are not available for this model; `--mtp-attention-window` and
`--lm-head-draft` are refused.

With `--lookup-ngram N`, a request whose last `N` tokens (the token it feeds next last) appeared
earlier in its sequence proposes what followed them then, up to the round's width (`--draft-tokens`,
or with `--adaptive-mtp` the width the controller chose), in place of the MTP block's drafts; the MTP block drafts the other requests of the round, and none when every
request has such a proposal. The proposal is verified like any draft, so a wrong one costs speed,
not tokens. Its rounds are reported with the n-gram proposals (`ngram_rounds`).

With `--adaptive-mtp`, `--draft-tokens K` is the most drafts a round makes. The controller the
Qwen3.5 models use (see [Adaptive MTP](serving.md#adaptive-mtp)) picks each round's width from
the requests' measured draft survival and the round times it has measured at each width, but here
it goes down to one draft: the MTP block runs one step per draft, so a narrower round also drafts
less, where the Qwen3.5 models' drafts are ready before the round. Each step count has its own
draft-chain graph and each width its own verification graph, captured at startup. A round of
several requests drafts three steps at most ([below](#mtp-with-several-requests)).

#### MTP with several requests

A round of several requests verifies their anchors and drafts in one pass. With host experts and a
CPU share, the CPU computes missing experts of a pass of up to eight tokens, and of a pass of up to
64 tokens those with at most eight pairs ([the CPU share](#the-cpu-share)); a wider pass copies
every missing expert to the GPU. The automatic drafting therefore keeps a round of several requests
within 64 tokens, which eight requests drafting four tokens each (40 tokens) stay within. On an RTX
A6000 (two Xeon E5-2699 v4 sockets with a 10.56-CPU quota, the Q2_0 release's host experts with a 12
GiB cache filled from the text layers' routing profile, `--max-concurrency 8`, the automatic
drafting with the release's DFlash2 drafter; 256 greedy tokens a request after a short prompt, two
runs a cell and six for eight requests without drafts, 2026-11-11), in aggregate tokens per second:

| Requests | Without drafts | Four drafts a request (the default) | Rounds of at most eight tokens |
|---:|---:|---:|---:|
| 1 | 50.5-66.3 | 83.1-105.1 | four drafts |
| 2 | 72.2-79.5 | 94.6-105.6 | 94.6-101.6 |
| 3 | 82.4-89.3 | 104.4-108.9 | 92.1-95.6 |
| 4 | 90.0-93.3 | 107.3-116.7 | 97.7-102.5 |
| 8 | 85.3-101.0 | 114.0-118.6 | 97.1-100.8 |

Rounds of at most eight tokens leave three drafts a request for two requests, one for three or four
and none for five to eight. With every expert of 9- to 64-token calls on the matrix kernel and that
bound, the default drafting had decoded 81.6-106.3, 91.0-103.2, 93.6-96.3, 92.3-104.0 and 93.8-100.4
tokens/s at one, two, three, four and eight requests on this host.

Greedy answers of concurrent requests depend on which requests share each round, which varies from
run to run, so they are not reproducible bit for bit: on the RTX A6000 above, answers of two to
eight requests, with or without drafts, differed between two runs of one build in some cells and
matched in others. A lone request's answers matched in every run.

With `--adaptive-mtp` the controller picks the width of a lone request's round. A batch drafts
until each of its requests has reached the floor or the round's width, so a round of more requests
with the controller's widths (up to eight drafts, the floor 0.5) ran nearly every step for all of
them and gave 0.89 and 0.91 times the speed without drafts at four and eight requests; the packed
plan then left drafts unverified, and once kept 18 of 555 at two requests. A round of several
requests therefore drafts three steps at most, each request verifying its drafts up to the floor:
on an RTX 4090 with Q2_0 host experts and the automatic cache (2026-11-03, when the CPU shared
any missing expert of verifications of up to 64 tokens), 1.16, 1.14, 1.09, 1.02 and 1.08 times the
speed without drafts at one, two, three, four and eight requests.

With disk experts a layer reads the experts its cache lacks before it runs, and a verification's
drafts route to experts of their own: three drafts every round gave 0.91, 0.82 and 0.64 times the
speed without drafts at one, two and eight requests, so the Engine drafts for disk experts only
under `--spec mtp`.

### DFlash2 speculative decoding

`--spec dflash2 --draft-tokens N` (1 to 7) decodes with the DFlash2 drafter of an artifact that
carries it ([Convert](#convert)); without it the engine refuses to start, and the MTP block, if the
artifact has one, is not loaded. The drafter is five sliding-window layers of the model's width
conditioned on the model's own states: after each of blocks 5, 15, 25, 35 and 45 the mean of the
four hyper-connection streams is kept for every committed position and projected into each
layer's ring of the last 2,048 positions (40 MiB a request). A round proposes all N drafts in one
pass over the last token and N masks, through the model's head and a selector that walks the 16
likeliest tokens of each position. The model verifies them as it verifies MTP's drafts, with the
same acceptance, so every committed token is the model's own draw. `--draft-min-p P` verifies up
to the first draft whose selector confidence (its weight among the 16) is at or below P. A single
request's proposal and its commits run from CUDA Graphs captured at startup. `--lookup-ngram N`
proposes from the context in place of the drafter as it does under MTP; `--adaptive-mtp` needs
`--spec mtp`, and pipeline stages are refused. The drafter's mathematics is in
[DFlash and DFlash2](maintainer/dflash.md#qwen38-flash-next).

`--spec dspine --draft-model PATH --draft-tokens N` drafts instead with a
[DSpine](maintainer/dspine.md) drafter trained for this model, up to its block less one: through
the same features, ring lanes, rounds, acceptance and `--draft-min-p`, with DSpine's block and code
bridge, on one device or under tensor parallelism. No DSpine drafter for Qwen3.8-Flash-Next is
published yet.

Without `--spec`, an artifact with the drafter and host experts on one device drafts four tokens a
round with `--draft-min-p 0.7`; over pipeline stages its MTP block drafts instead. On an RTX 3090
with host experts (EPYC 7K62, 96 CPUs; 256 greedy tokens after a code prompt and after a prose
prompt, and the answer to a 144-token prompt; the mean of two runs) that decoded 92.3, 77.9 and 58.2
tokens per second with the IQ3_S release, 1.10, 1.06 and 1.17 times the MTP block's automatic
drafting, and 122.4, 93.7 and 113.8 with the Coder IQ1_M release, 1.19, 1.09 and 1.15 times MTP and
1.72, 1.31 and 1.59 times without drafts. Every answer matched decoding without drafts. Seven drafts
with the floor 0.8 decoded within 5% of four (1.13 times MTP in geometric mean, as four) while
verifying 68-73% more drafts. A round of several requests verifies at most 64 tokens, within which
eight requests draft four tokens each ([MTP with several requests](#mtp-with-several-requests));
concurrent answers to one prompt can differ from run to run with or without drafts.
`--lookup-ngram 3` beside seven drafts decoded 85.9, 76.4 and 49.5 tokens per second on IQ3_S, below
the drafter alone. Under tensor parallelism over two RTX 3090s (Q2_0, device experts, which draft
only under `--spec`), four drafts with the floor 0.7 decoded 1.46, 1.21 and 1.24 times as fast as
without drafts after the three prompts and 1.43 times after a 9,741-token document (MTP's three
drafts 1.47, 1.18, 0.98 and 1.26), and every answer matched decoding without drafts, where MTP's did
not.

## Execution

- The residual is a four-stream hyper-connection stack kept in FP32 between layers.
- The 36 Gated DeltaNet layers run the Qwen3.5 GDN kernels with a sigmoid output gate; the 12 sparse
  attention layers run the block indexer and attend only to the blocks it selects (plain dense
  attention below 2,051 positions).
- The PLE layer reads its 16 n-gram rows per token from the table's file (IQ4_NL rows decoded on
  the GPU).
- A layer's projections of one input run together: the GGUF ones quantize the input to q8_1 once,
  and the BF16 ones that a call of up to eight tokens runs as vector products (a Gated DeltaNet
  layer's `a` and `b`) share one launch. The router's top-10 selection for up to eight tokens runs
  in its logits launch, in the CTA that finishes last. Each product and selection keeps the bits of
  its own launch.
- The 512-expert MoE groups each layer's (token, expert) pairs by expert on the GPU and runs one
  pass over each selected expert's rows for all of its tokens, through device tables of expert base
  pointers: an expert is read wherever the table points, in device memory, a cache slot or the
  pinned host block. Up to eight tokens run vector products; wider calls run ggml's integer
  tensor-core matrix kernel (IQ1_M experts once a call routes more than 16 rows to each expert,
  unless `--iq1m-experts vector`) over the routed pairs from device memory; up to 64 tokens an
  expert with at most eight pairs keeps the vector products, in every placement, so that the CPU can
  compute it as the GPU does. The choice depends only on the call's width and the expert's pairs,
  and a prompt chunk of up to 64 tokens runs alone in every placement (without the lossy
  `--expert-tail-skip`), so a call's result does not depend on where its experts reside. The matrix
  kernel runs one CTA per row tile and per column tile that holds pairs (the sort that groups the
  pairs lists those tiles for each tile width), rather than one per expert and possible column tile
  (on an RTX 3090 with IQ3_S host experts, a 31,865-token prompt prefilled at 1.71k instead of 1.61k
  tokens/s). That kernel reads a 640-value down row in 256-value steps, so it decodes the bytes
  after a down matrix as the down's blocks: in a bank they are the next expert's down or zeros after
  the last one, and every cache or stream slot keeps zeros after its down, which a smaller down from
  another layer does not uncover (another format's bytes there can hold a non-finite scale, and NaN
  follows). With host experts, a wider prompt call first copies the routed experts the cache does
  not hold into a device pool on each GPU (0.7 GB for Q2_0), a bank per projection laid out as the
  host's, so experts adjacent on the host cross the bus in one copy and each expert crosses it once
  per chunk; zeros follow each run of copied downs. While the cache has free slots, such an expert
  goes into one of them instead, and later calls find it cached. On one GPU a prompt longer than a
  chunk, or the steps of several prompts prefilling at once, run in spans of up to eight chunks
  layer by layer: each layer runs its attention over every chunk of the span, then its experts over
  every chunk, so the layer's uncached experts cross the bus once per span, during the attention
  phase that precedes them, and a span's short last chunk runs from the same pool (a 5,669-token
  prompt prefilled at 1,334 instead of 678 tokens/s on an RTX 3090 when spans came; the bank-shaped
  pool and the two phases then took a 9,297-token Q2_0 prompt from 1.12-1.18k to 1.35k tokens/s from
  a cold expert cache). The expert phase runs up to 4,096 tokens of consecutive chunks in one call,
  each chunk routed on its own, so the matrix kernel takes 128-column tiles rather than the 32 of a
  1,024-token chunk; a last chunk of up to 64 tokens (eight with `--expert-tail-skip`) runs alone,
  after the layer's wider calls, since it may copy experts into cache slots those calls read (with
  IQ3_S, a 31,860-token prompt's expert matrix products took 4.47 instead of 6.52 s on an RTX 3090).
  With two prefill buffers (the default with the automatic expert cache, `--expert-prefill-buffers`)
  the pool has two buffers that consecutive layers alternate between: each layer's copies start as
  soon as the layer two back is done with its buffer, while the layer before it runs. A thread of
  its own issues each layer's copies, hundreds of operations that would otherwise hold the launching
  thread while the copy queue is full, so the layers in between keep being launched. An engine of
  one sequence (`--max-concurrency 1`, the CLI) keeps no other prompt waiting behind a step, so its
  spans take sixteen chunks where the further eight chunks' planes (46 KiB a token) fit 1/24 of the
  memory the device has free at startup: in a prompt past eight chunks each layer's uncached experts
  then cross the bus half as often ([Expert placements](#expert-placements)). Weighted expert
  outputs are summed in fixed point, so the result does not depend on the order experts finish in.
- A prompt chunk whose rows produce no logits (each chunk before the prompt's last, and a scoring
  call's chunks before its first scored position) runs the last layer, a sparse attention layer,
  only up to its key/value and indexer-key appends. In a span, the last layer's expert calls run
  only where one of their chunks produces logits, so those experts cross the bus less often. With
  MTP every chunk runs whole, since the MTP block reads each position's final stack. The appended
  rows come from the same projections and kernels, so caches, states and logits keep their bits.
- Decode steps, verifications of up to 64 tokens and prompt chunks of up to 64 tokens (eight with
  `--expert-tail-skip`; every verification and prompt chunk where half the free memory cannot hold a
  slot pool) of GGUF host experts run in two stages. A kernel splits the call's routes into the
  cached experts, which run at once, and the missing ones. A call of up to eight tokens posts them
  to a host service through mapped memory, without a host synchronization; the service copies each
  missing expert into the slot of the layer's least recently used expert of the lowest frequency
  tier (or into staging, or the CPU computes its share), the layer waits for it (on the device in a
  replayed graph, on the host in an eager pass such as a prompt chunk's), and the second stage runs
  it. A wider call runs eagerly and as over device banks: the host waits for the split, copies the
  missing experts into cache slots or into half of the 128 MiB staging, a round at a time, and each
  round's products run while the next round's copies cross the bus; in a call of up to 64 tokens CPU
  threads meanwhile compute a share of the missing experts with at most eight pairs, the ones the
  GPU would run on the vector products. The service spins for half a second after each pass start,
  call and served request; asleep, it still looks for requests every 10 ms, since a replayed layer
  posts its request from the device and no host call wakes the service for it. The decode form of
  the expert kernels needs no sort or memset: one small kernel lists each slot's pairs, quantizes
  the tokens' inputs once and zeroes the sum; a block of the fused gate/up kernel takes 32 rows of
  one expert and quantizes its piece of the middle for the down kernel itself; one token's down runs
  row by row over all its experts, summing a row in registers. The decoding of Q2_0, the routed
  experts' format, is integer-throughput bound, so its activations come transposed and its codes
  need two operations a word. The shared expert runs on a second stream meanwhile. All of it is the
  separate kernels' arithmetic bit for bit. Device and disk experts take the same decode form in one
  stage, the shared expert on the second stream as well: on an RTX PRO 6000 Blackwell Server (600
  W), the UD-IQ2_XS release with device experts and no n-gram table decoded a 300-token greedy
  answer at 131.2 instead of 118.9 tok/s, with MTP and 4 drafts at 171.3 instead of 153.0, and with
  disk experts at 70.6 instead of 67.5, the same bytes (November 10, 2026).
- The expert cache counts the routes each forward pass took (decayed per token) and, between
  passes, copies the experts it needed most into its slots and points the tables at them: for
  every layer with mapped misses, and after a prompt chunk with staged ones. With disk experts the
  cache works per layer instead: once a layer has routed its tokens, the experts it lacks are read
  into the slots least recently used (never one the same call needs), and only then do the
  layer's experts run. A prompt longer than a chunk runs in the same spans as with host experts,
  each chunk's experts on their own: the chunks of a span reach a layer one after another, so the
  experts the first chunks read are still in their slots for the later ones, and an expert is read
  once per span rather than once per chunk. At a span's last chunk the layer's residents that chunk
  does not route to are released: once no slot is empty, later reads take released slots first,
  oldest release first, before the least recently used, unless a call routes to them again. The
  span's later layers thus read their experts into the earlier layers' stale ones, and every layer
  keeps the experts of the prompt's end. Empty slots are always taken first. The MTP block's experts
  (stored as Q8_0, wider than the text layers') have slots of their own, for the experts of one
  draft step of eight tokens (80 slots), when the rest of the budget still holds a text layer's
  widest call; the text layers' slots are then sized for their own experts (11,815 Q2_0 slots
  instead of 3,504 on an RTX 3090). Only drafts run the block's experts: with 80 slots instead of
  one for each of its 512 experts, three drafts every round gave 78.5 instead of 65.8 tokens/s at
  two requests and 103.6 instead of 79.4 at eight (aggregate, RTX 4090, Q2_0, the mean of two runs
  at eight), and one request's two runs 60.2 and 55.6 against 52.6 and 54.4.
- A decode step (one token) replays a CUDA graph per pipeline stage, captured at startup and
  shared by the sequences: a step of another sequence than the last captures it again and updates
  the graph in place; the token's position reaches the sparse-attention kernels in device memory.
  Disk experts need the host between a layer's routing and its experts, so their steps stay eager.
  `--no-cuda-graph` decodes eagerly everywhere.
- With MTP, a verification of one request (at most eight tokens, so its experts take the vector
  products) replays graphs of its own the same way, and so does the draft chain of one request on
  one GPU, host experts included; several requests' rounds, the MTP catch-up and disk experts run
  eagerly. With staged host misses the MTP block serves its draft steps' missing experts (at most
  eight tokens a step) as a decode layer does, copied or computed by the CPU, so the drafts follow
  the block's own routing. On an RTX 4090 with the IQ3_S release's host experts (EPYC 7742, 64K
  context, three drafts with `--draft-min-p 0.5`, 256 greedy tokens after a code prompt, a prose
  prompt and a 31,865-token document, once per `--kv-dtype` format) serving those misses instead of
  dropping them committed 5-21% more tokens a round and decoded 1.025 times as fast (geomean of 27
  pairs, 0.959-1.092; slower in 4, the two largest losses after the document).
- A pass whose layers may wait on the device for their missing experts ends synchronized: CUDA
  loads a kernel lazily at its first launch, a load may wait for the device to go idle, and a
  launch during such a wait would deadlock with the service it waits for.

## Measurements

### Expert placements

Greedy decoding, one request, the generate test (`ninfer_qwen4_exp_generate_real`) without drafts.
Decode is measured over a short answer (78 tokens for Q2_0, 81 or 82 for IQ3_S) and over the first
five tokens after a 4,463-token prompt; prefill is that prompt in 512-token chunks, which with host
or disk experts ran as a span of eight chunks and a last chunk of 367 tokens (the span of that
build; an engine of one sequence now takes the prompt as one span). Host memory is
the process's peak resident set (with host experts, the pinned bank), device memory the most
`nvidia-smi` showed in use. With host or disk experts the expert cache takes the memory the GPU has
free and fills while the requests run.

One GPU, October 26, 2026, a build of `master` with CUDA 13.4 (`120a` on the RTX 5090, `sm_86` on
the others), the n-gram table as the separate artifact, one run per row:

- RTX 5090: 575 W, PCIe 5.0 x16, a 16-core Ryzen 9 7950X and 124 GB of RAM.
- RTX 4090: 450 W, PCIe 4.0 x16, a 48-core EPYC 7K62 and 503 GB of RAM.
- RTX 3090: 370 W, PCIe 3.0 x16, two 26-core Xeon Platinum 8272CL sockets (four NUMA nodes) and
  1.5 TB of RAM. The test ran pinned to the CPUs of the GPU's NUMA node after the model files'
  pages were evicted, so the pinned bank and the page cache are on that node.

| Card | Release | Experts | Device memory | Host memory | Decode, short answer | Decode after 4,463 tokens | Prefill |
|---|---|---|---:|---:|---:|---:|---:|
| RTX 5090 | Q2_0 | pinned host memory | 30.1 GiB | 33.3 GiB | 97.8 tok/s | 83.2 tok/s | 3,290 tok/s |
| RTX 5090 | Q2_0 | disk, the files in the page cache | 30.1 GiB | 1.7 GiB + page cache | 72.1 tok/s | 102.9 tok/s | 1,827 tok/s |
| RTX 5090 | Q2_0 | disk, the files' pages evicted every second | 30.1 GiB | 1.7 GiB | 37.7 tok/s | 66.4 tok/s | 1,108 tok/s |
| RTX 5090 | IQ3_S | pinned host memory | 30.1 GiB | 48.7 GiB | 82.5 tok/s | 70.7 tok/s | 2,451 tok/s |
| RTX 5090 | IQ3_S | disk, the files in the page cache | 30.1 GiB | 2.0 GiB + page cache | 56.4 tok/s | 35.5 tok/s | 770 tok/s |
| RTX 5090 | IQ3_S | disk, the files' pages evicted every second | 30.1 GiB | 2.0 GiB | 28.5 tok/s | 16.7 tok/s | 399 tok/s |
| RTX 4090 | Q2_0 | pinned host memory | 22.1 GiB | 33.1 GiB | 73.9 tok/s | 66.9 tok/s | 1,801 tok/s |
| RTX 4090 | Q2_0 | disk, the files in the page cache | 22.1 GiB | 1.5 GiB + page cache | 46.5 tok/s | 42.5 tok/s | 1,144 tok/s |
| RTX 4090 | Q2_0 | disk, the files' pages evicted every second | 22.1 GiB | 1.5 GiB | 21.0 tok/s | 14.6 tok/s | 558 tok/s |
| RTX 3090 | Q2_0 | pinned host memory | 22.2 GiB | 33.1 GiB | 39.1 tok/s | 45.7 tok/s | 663 tok/s |
| RTX 3090 | Q2_0 | disk, the files in the page cache | 22.1 GiB | 1.5 GiB + page cache | 27.8 tok/s | 27.5 tok/s | 503 tok/s |
| RTX 3090 | Q2_0 | disk, the files' pages evicted every second | 22.1 GiB | 1.5 GiB | 19.7 tok/s | 14.4 tok/s | 474 tok/s |
| RTX 3090 | IQ3_S | pinned host memory | 22.2 GiB | 48.5 GiB | 27.1 tok/s | 33.1 tok/s | 438 tok/s |
| RTX 3090 | IQ3_S | disk, the files in the page cache | 22.1 GiB | 1.7 GiB + page cache | 17.4 tok/s | 13.7 tok/s | 256 tok/s |
| RTX 3090 | IQ3_S | disk, the files' pages evicted every second | 22.1 GiB | 1.7 GiB | 12.3 tok/s | 6.0 tok/s | 179 tok/s |

- **Prompt spans.** With host or disk experts a prompt longer than a chunk prefills in spans of up
  to eight chunks (sixteen now for one request at a time, below), so a layer's missing experts
  cross the host link once per span rather than once per chunk; the chunk still sets each call's
  width. The CLI with 1,024-token chunks and MTP drafts
  on the 7,680-token needle prompt with Q2_0 disk experts, 16 greedy tokens, this build
  against one that ran disk experts chunk by chunk with the MTP block's experts in the text layers'
  slots, interleaved on the same hosts, two runs each:

  | Card | Model files | Prefill, tok/s | Decode after it, tok/s |
  |---|---|---:|---:|
  | RTX 5090 | in the page cache | 1,830 against 449-450 | 114.7-114.8 against 31.4-31.9 |
  | RTX 5090 | pages evicted every second | 970-1,000 against 194-196 | 55.7-58.6 against 15.4 |
  | RTX 4090 | in the page cache | 1,640-1,670 against 494 | 44.8-44.9 against 23.7-24.0 |
  | RTX 4090 | pages evicted every second | 532-549 against 131-133 | 12.4-12.7 against 13.6-13.9 |
  | RTX 3090 | in the page cache | 927-988 against 305-307 | 30.2-31.0 against 17.4 |
  | RTX 3090 | pages evicted every second | 469-471 against 103-104 | 12.3-12.4 against 6.4-6.5 |

- **Decode after the prompt.** Against the same test chunk by chunk (and without the releases), on
  the same hosts, the Q2_0 rows prefill 3.9-4.3 times as fast with host experts, 1.0-1.4 times
  from the page cache and 1.0-2.0 times with the pages evicted. The five tokens after the prompt
  decode 10-12% slower from disk (2% faster on the RTX 3090 with the pages evicted): the span's last
  chunk keeps its experts, but this prompt ends in a separate 367-token chunk; without the releases
  the RTX 5090 decoded them at the rate chunk by chunk. With host experts they decode 1% slower on
  the RTX 5090, 8% slower on the RTX 4090 and 2% faster on the RTX 3090.
- **Spans of sixteen chunks.** On November 10, 2026 a rented RTX 5090 (575 W, driver 595.91.07,
  CUDA 13.4; EPYC 7543, 12.75 CPUs of a host other tenants loaded to 24-56 of its 120 threads) ran
  the UD-IQ2_XS release from the CLI with host experts, 1,024-token chunks and no drafts, this build
  against one held at spans of eight chunks, in interleaved pairs. Prefill, tok/s:

  | Prompt | Spans of eight chunks | Sixteen (one request at a time) |
  |---|---:|---:|
  | 7,680 tokens, one span either way | 4.20k, 3.89k | 4.26k, 3.75k |
  | 20,181 tokens | 4.01k, 3.90k | 4.34k, 4.29k |
  | 32,256 tokens | 4.33k, 4.23k, 4.13k | 5.28k, 5.22k, 5.30k |
  | 64,512 tokens | 4.77k, 3.83k | 5.31k, 5.23k |
  | 32,256 tokens, disk experts in the page cache | 1.84k, 1.69k | 2.38k, 2.29k |
  | 32,235 tokens, the DFlash2 drafter loaded | 4.76k, 3.81k | 5.24k, 5.31k |
  | 32,235 tokens, `--expert-cache-mib 7000` | 3.13k, 2.90k | 4.30k, 4.33k |

  The answers were the same text. With the span's stack in the automatic cache between prompts
  the engine started with 23.4 GiB of expert cache (14,544 slots) instead of 23.1 GiB (14,400);
  with disk experts, which lend nothing, with 23.2 instead of 23.6 GiB. The 256 tokens after a
  32,235-token prompt decoded at 73.0-80.7 tok/s (median 76.3 of five runs) against 75.1-80.8
  (77.5). After a 52-token prompt they decoded at 78.1-85.6 (median 82.4 of eight runs) against
  76.2-86.4 (81.8) on the build before the change; ten pairs against the eight-chunk build of an
  earlier commit gave 76.5-86.0 (81.4) against 81.2-86.6 (84.7), and that build ran 1.2% ahead of
  the build before the change as well, so the host's load, not the spans, sets these differences.
  Not measured: another card, a prompt step between the decode rounds of other requests, and the
  DeepSeek-V4.1 executor, whose spans are 16,384 tokens already.
- **NUMA.** Unpinned on another two-socket RTX 3090 host (Xeon E5-2690 v4), Q2_0 decoded the short
  answer 9% slower with host experts and 43% slower with disk experts in the page cache than pinned
  to the GPU's node.
- **The host.** Host and disk experts cross the host link, and disk experts are read by the CPU
  threads, so these rows measure the host as much as the card.

Every expert on the GPUs, and the RTX PRO 6000, October 5 and 6, 2026, earlier builds: CUDA 12.8
and the table inside the artifact on the two RTX 3090 Ti; CUDA 12.8 and the separate table on two
RTX 4090s at 450 W (PCIe 4.0 x16, no peer access) in a cloud VM with two EPYC 7543 sockets, both
GPUs on NUMA node 0, and 694 GB of RAM; on Blackwell a `120a` build with CUDA 13.1 and the separate
table, single-GPU rows pinned to the CPUs of the GPU's NUMA node:

- RTX PRO 6000 Blackwell Workstation Edition: 600 W, a PCIe 4.0 x16 host link, 64 vCPUs and
  1.1 TB of RAM; short answer 87 tokens for IQ3_S.
- RTX 5090s: boards with a 600 W default limit, PCIe 5.0 x16, 512 threads and 1 TB of RAM; short
  answer 81 tokens for IQ3_S.

The ranges are two runs.

| Hardware and placement | Release | Device memory | Host memory | Decode, short answer | Decode after 4,463 tokens | Prefill |
|---|---|---:|---:|---:|---:|---:|
| 2× RTX 3090 Ti (PCIe, no P2P), experts on the GPUs (`--devices 0,1`) | Q2_0 | 18.4 + 19.4 GB | 0.7 GB | 90.2 tok/s | 107 tok/s | 1,504 tok/s |
| 2× RTX 4090, experts on the GPUs (`--devices 0,1`) | Q2_0 | 18.5 + 19.4 GiB | 0.8 GiB | 92.6-100.7 tok/s | 116-120 tok/s | 3,482-3,734 tok/s |
| 2× RTX 5090, experts on the GPUs (`--devices 0,1`) | Q2_0 | 18.8 + 19.6 GiB | 0.9 GiB | 136.1-136.3 tok/s | 157.0-157.2 tok/s | 3,807-3,811 tok/s |
| 2× RTX 5090, experts on the GPUs (`--devices 0,1`) | IQ3_S | 26.5 + 27.8 GiB | 1.0 GiB | 121.6-121.9 tok/s | 141.8-142.0 tok/s | 3,310-3,341 tok/s |
| RTX PRO 6000, experts on the GPU | Q2_0 | 37.7 GiB | 0.8 GiB | 136.0-139.7 tok/s | 160.7-162.6 tok/s | 2,925-2,952 tok/s |
| RTX PRO 6000, experts on the GPU | IQ3_S | 53.7 GiB | 1.0 GiB | 123.8-125.9 tok/s | 146.8-146.9 tok/s | 2,540-2,541 tok/s |
| RTX PRO 6000, experts in pinned host memory | Q2_0 | 38.5 GiB | 34.0 GB pinned | 54.8-55.0 tok/s | 67.7-70.0 tok/s | 1,296-1,297 tok/s |
| RTX PRO 6000, experts in pinned host memory | IQ3_S | 55.0 GiB | 50.3 GB pinned | 29.3 tok/s | 31.2 tok/s | 803 tok/s |
| RTX PRO 6000, experts on disk, the files in the page cache | Q2_0 | 93.6 GiB | 1.0 GiB + page cache | 75.9 tok/s | 115.8 tok/s | 2,024 tok/s |
| RTX PRO 6000, experts on disk, the files in the page cache | IQ3_S | 93.5 GiB | 1.2 GiB + page cache | 65.6 tok/s | 106.0 tok/s | 1,661 tok/s |
| RTX PRO 6000, experts on disk, the files' pages evicted every second | Q2_0 | 93.5 GiB | 1.0 GiB | 35.1 tok/s | 87.1 tok/s | 1,000 tok/s |
| RTX PRO 6000, experts on disk, the files' pages evicted every second | IQ3_S | 93.5 GiB | 1.2 GiB | 28.7 tok/s | 78.6 tok/s | 773 tok/s |

- **Device expert cache.** On the PRO 6000 the cache takes 94 GB, and nearly every expert ends up
  resident.
- **CUDA graphs** add 27% to the short-answer decode on the two RTX 5090s (107.0 tok/s eager), 23
  to 26% on the PRO 6000, 11% on the two RTX 3090 Ti, and 1% and 10% on the two RTX 4090s in their
  two runs (91.5-91.7 tok/s eager); disk experts decode eagerly.

Every GSQ-RCO release (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S and the Coder IQ1_M) answers the generate
test's prompts with disk and host experts, with and without CUDA graphs; with disk experts the
process peaks at 1.05 to 1.19 GB of RAM. The published layout (models without the table, the shared
table artifact) passed device, host and disk placements, batched against single decoding, a
restored snapshot, an image question per release and the server's concurrency, prefix reuse and
JSON Schema checks on one NVIDIA L40S.

Where a decode step goes, from an Nsight Systems trace of the generate test's graph-replayed steps
(Q2_0, every expert on that L40S): about 1,780 kernels and copies in 10.6 ms, 0.5 ms of it idle
between them. A token read about 4 GB of weights when the hyper-connection projections were BF16,
the largest share (1.27 GB, more than the ten routed experts of every layer, 0.66 GB); they are
Q8_0 now, 0.67 GB.

### Host and disk experts over two RTX 4090s

Two RTX 4090s (450 W, PCIe without peer access, so the stages' transfers go through host memory)
in a Vast container with an EPYC 7713 and 503 GB of RAM ran the Q2_0 release with its IQ4_NL table
on two pipeline stages, 2026-11-07. Host and disk experts scored the first 40 KB of the WikiText
stream (`scripts/pods/residency_check.sh` with `ARGS='--devices 0,1'`) at perplexity 2.540839
each, the value one GPU gives with either. Through `ninfer-serve`, one request at a time, 64 greedy
tokens after a 15-token and after a 5,879-token prompt, each configuration run once:

| Experts | Options | Drafts | Expert cache | Prefill, 5,879 tokens | Decode, short / long |
|---|---|---|---:|---:|---:|
| host, one GPU | `--max-context 32768` | DFlash2, 4 | | 2,581 tok/s | 116.8 / 127.4 tok/s |
| host, two stages | `--max-context 32768` | MTP, 3 | 17.8 + 16.3 GiB | 3,806 tok/s | 204.0 / 208.5 tok/s |
| disk, two stages | `--ngram-residency disk --max-context 131072 --kv-capacity 131072 --kv-dtype rk8v4` | none | 20.4 + 19.9 GiB | 1,128 tok/s | 44.4 / 104.6 tok/s |

All three gave the same tokens. The two caches together hold nearly every expert, so after its
first reads the disk run decodes from the GPUs; its short prompt came first, with the caches cold.
Before this build either placement over stages stopped at the first expert copy of the second GPU.

### Host experts on one RTX 3090

One RTX 3090 (24 GB, PCIe 4.0 x16) with an i9-11900KF and 125 GB RAM, driver 580.82.09 and
CUDA 13.1 ran the Q2 model (GSQ-RCO Q2_0 with the IQ4_NL table) through `ninfer-serve` with host
experts, int8 KV, an 8,192-position context and one request at a time. Three prompts (a Python
LRU cache, the blue sky, the history of the Great Wall in Chinese) ran in that order twice, 256
greedy output tokens each, so a prompt's second run follows the other two. Each binary ran every
configuration once; the baseline is commit `a9155e0`.

| Mode | Baseline decode, tok/s (first / second run) | October 9 build |
|---|---|---|
| plain, code / prose / Chinese | 69.2/78.4, 77.9/80.6, 74.1/79.4 | 87.3/92.3, 93.7/94.3, 93.4/94.3 |
| MTP 2 drafts | 68.6/70.0, 71.9/73.4, 57.2/62.4 | 118.8/118.6, 117.3/114.8, 95.8/101.2 |
| MTP 3 drafts | 66.7/66.9, 67.2/68.3, 53.0/57.5 | 121.2/122.6, 117.7/111.1, 90.0/94.6 |
| MTP 4 drafts | 63.1/62.2, 60.5/60.9, 44.9/48.2 | 111.1/105.9, 101.9/96.2, 74.6/78.5 |
| disk experts, code / prose (first run) | 64.5, 66.8 | 68.1, 71.4 |

The baseline predates the tiered admission of staged misses and Q2_0's transposed decoding. Run
back to back, a repeated prompt decodes at
98.5 (code) and 100.4 (prose) tokens/s plain and 137.6 and 123.7 with three drafts. A cache that
admitted every miss into the least recently used slot let a passing decode evict the experts other
prompts keep routing to; the tiers (below 1.5, below 6, and more routes) restored second-run starts
of 212-305 ms instead of 284-397 ms on a second RTX 3090 for at most 4.4% of first-run decode.

### Host experts and long contexts on one RTX 4090

One RTX 4090 (450 W, PCIe 4.0 x16, driver 590.48.01, CUDA 12.8) in a Vast container with two
EPYC 7742 sockets, a 61.44-CPU quota and 503 GB of RAM ran the IQ3_S release with its IQ4_NL table
through `ninfer-serve`: host experts, one request at a time, 256 greedy output tokens, every server
pinned to the GPU's NUMA node (`taskset -c 0-63,128-191`), 2026-10-20. The baseline is commit
`5ed86dff9`, the build before the expert tier, the CPU share by default, MTP by default and the
streamed sparse-attention KV; its default copies every miss and drafts nothing, and its MTP row
adds `--spec mtp --draft-tokens 3 --expert-dma-share auto`.

At a 65,536-position context, a Python LRU cache prompt and a blue-sky prompt twice and a
31,865-token document once per server, two rounds in rotated server order (eight short answers
per row):

| Build | Expert cache | Decode, short prompts | Decode after 31,865 tokens | Prefill | Bus bytes |
|---|---:|---|---|---|---:|
| baseline | 14,288 MiB | 61.4-75.5 (68.3) tok/s | 68.7 tok/s | 2,833-2,846 tok/s | 660 GB |
| baseline, MTP and CPU share | 14,080 MiB | 66.5-77.1 (71.4) tok/s | 66.5-75.0 tok/s | 2,576-2,714 tok/s | 901 GB |
| defaults | 16,059 MiB | 72.6-82.5 (77.0) tok/s | 66.0-79.0 tok/s | 2,840 tok/s | 223 GB |
| defaults, 59 CPU threads | 16,059 MiB | 70.2-77.8 (73.1) tok/s | 64.4-75.6 tok/s | 2,818-2,831 tok/s | 218 GB |
| defaults, 59 CPU threads, `--kv-resident all` | 15,264 MiB | 68.7-76.9 (71.8) tok/s | 63.2-75.2 tok/s | 2,833-2,859 tok/s | 226 GB |

The bus bytes are every expert byte the server copied to the GPU over its ten requests. The rows
with 59 CPU threads (the pool's size before it took half the CPUs) ran in one sitting with the
baseline; the defaults ran about an hour later on the same host. The defaults copy a quarter of the baseline MTP row's bytes and decode the
short prompts 7.8% faster on its mean, with overlapping ranges; streamed and resident KV decode
alike at this context. Before the prefill buffers became cache slots the 59-thread defaults' cache
held 14,776 MiB and decoded the short prompts at 64.9-72.0 (68.3) tok/s.

An Nsight Systems trace of one 256-token code answer shows where the step goes instead. With the
defaults (then a 59-thread CPU pool) the GPU spent 39% of the answer computing, 38% in the kernels
that wait for the host's expert service (the missing experts' copies and the CPU's products) and
23% idle, and the copies took 23.8 GB; the baseline's MTP row spent 41%, 38% and 21%, with
42.5 GB of copies.

At a 458,752-position context, the same two short prompts and a 136,077-token document once,
one server each: the baseline keeps every layer's KV on the device and has 4,589 MiB left for
experts; the defaults keep the 12 sparse-attention layers' KV in pinned host memory (11,648 MiB)
with 832 MiB of their blocks on the device, and the cache takes 14,643 MiB.

| Build | Expert cache | Decode, code / prose | Decode after 136,077 tokens | Prefill |
|---|---:|---|---|---|
| baseline | 4,589 MiB | 39.2 / 43.0 tok/s | 37.7 tok/s | 2,612 tok/s |
| defaults, 59 CPU threads, `--kv-resident all` | 4,755 MiB | 39.0 / 31.3 tok/s | 34.7 tok/s | 2,548 tok/s |
| defaults, 59 CPU threads | 14,643 MiB | 70.0 / 67.2 tok/s | 62.3 tok/s | 2,700 tok/s |
| defaults | 14,643 MiB | 70.3 / 67.3 tok/s | 67.6 tok/s | 2,708 tok/s |

The baseline's and the defaults' answers part after a few tokens to a few hundred. Since the CPU's
products of K-quant, Q4_0 and Q5_0 blocks were made equal to the GPU's, the GPU's vector kernels
round each step of those slices' products, the one change to the GPU arithmetic of this release's
decode, which keeps 177 tensors in Q4_K, Q5_K and Q6_K; prompt chunks run the matrix kernels, which
did not change. The defaults' configurations (with and without lent buffers, 30 or 59 CPU threads,
streamed or resident KV) give each other's answers token for token with equal log probabilities, and
the 40 KB perplexity slice scores 1.980071 on both builds.

### The CPU share on one RTX 4090

The RTX 4090 host of the previous section, servers pinned to the GPU's NUMA node, the IQ3_S release
at a 65,536-position context with its other defaults, 2026-10-20. Three prompts (a Python LRU
cache, the blue sky, the history of the Great Wall in Chinese) ran twice per server with 256 greedy
output tokens, in two rounds whose server order rotates: 12 answers per row.

| `--expert-dma-share`, `--expert-cpu-threads` | Decode, tok/s (range, mean) | Misses on the CPU | Bus bytes per token |
|---|---|---:|---:|
| 1, the GPU only | 62.7-75.2, 68.4 | — | 172.5 MB |
| `auto`, 59 threads | 64.3-78.6, 71.1 | 37.3% | 85.8 MB |
| `auto`, 16 threads | 65.4-82.8, 74.2 | 29.9% | 102.4 MB |

Every answer with a CPU share has the GPU-only answer's tokens and log probabilities. The bus bytes
are every expert byte the server copied, prompt chunks included, over the tokens it generated. An
expert the CPU computes takes no slot and crosses no bus, so the share halves the bytes; decode
gains less, 4% with 59 threads and 8% with 16: the bus no longer bounds a step on this host. With
59 threads the CPU took a larger share of the misses and still decoded slower than with 16 in 11 of
the 12 pairs of answers. A second sweep (the same three prompts once per server, two rounds) decoded
at a mean of 72.5 tok/s with 8 threads, 74.4 with 16, 74.3 with 24, 73.4 with 32 and 71.3 with 59,
every answer equal; the default pool takes half the process's CPUs (30 here). Spinning on all 61
CPUs of the container's quota, the pool had the process throttled in 208 of 211 scheduling periods
while it decoded.

A second RTX 4090 rental with a 30-CPU quota (the same release and context, 2026-10-20, a later
build) answered the Python and blue-sky prompts twice and a 31,865-token prompt once per server:

| `--expert-cpu-threads` | Decode, tok/s: Python, blue sky, 32K prompt, Python, blue sky | Throttled periods |
|---|---|---:|
| 8 | 81.8, 76.4, 70.4, 82.0, 76.4 | 4 of 291 |
| 15 (the default here) | 84.8, 79.5, 75.2, 84.5, 84.0 | 3 of 278 |
| 22 | 91.3, 82.8, 77.4, 91.4, 87.0 | 4 of 270 |
| 29 | 92.3, 85.2, 76.5, 92.6, 86.7 | 13 of 271 |

Every answer was equal. Here more threads decoded 3-8% faster at 22 than at the default, where the
first host gained nothing past 16; the default stays at half the CPUs, and `--expert-cpu-threads`
tunes a host.

### The expert tier's paced rounds on one RTX 4090

One RTX 4090 with a Ryzen 9 9950X (32 CPUs, the default 15-thread pool), the IQ3_S release at a
65,536-position context with its other defaults, 2026-10-20: one server per `--kv-dtype` format and
build, the Python LRU cache and blue-sky prompts twice and the 31,865-token prompt once, 256 greedy
output tokens. Starting a round's swaps a quarter per pass instead of all after the round's pass
([the expert cache](#the-expert-cache)) cut the GPU's wait for the first layer's misses from 2.6 to
0.9 ms a step and decoded faster with every format:

| `--kv-dtype` | bf16 | int8 | fp8 | k8v4 | nvfp4 | rk8v4 | rk4v4 | rk4v4-e8 | rk2v4-e8 |
|---|---|---|---|---|---|---|---|---|---|
| Decode, whole round / paced, tok/s | 86.7 / 91.8 | 88.9 / 93.0 | 85.3 / 92.4 | 90.8 / 95.3 | 86.8 / 92.8 | 92.7 / 94.6 | 89.0 / 97.2 | 88.1 / 93.6 | 75.6 / 82.9 |
| Paced / whole round, geometric mean | 1.058 | 1.045 | 1.083 | 1.049 | 1.068 | 1.018 | 1.093 | 1.063 | 1.096 |

Over the 45 pairs of answers the paced build decoded 0.970-1.144 times as fast, 1.064 on the
geometric mean; three pairs (two rk8v4 and one int8 blue-sky answer) were slower. Every answer had
the same tokens.

On the same host `auto` had read the CPU's gate and up time as nearly zero in about two of five
calls: the CPU pool's calling thread could return before the last job's thread stamped its end, and
read the previous phase's stamp. With the stamp in place before the pool returns, the same 45 pairs
decoded 0.985-1.128 times as fast, 1.031 on the geometric mean; per format 0.990 (rk4v4, all five
pairs slower by 0.2-1.5%) to 1.068 (k8v4). Every answer had the same tokens.

### MTP

The Q2_0 release with Unsloth's `shared-Q8_0` MTP block, converted into one file with its
table, on two RTX 3090s (350 W, PCIe 4.0 x16, one GPU per socket of an EPYC 7663 host with 629 GB
of RAM, no peer access), 2026-10-07: greedy decoding of a 117-token answer to a 37-token coding
prompt, two runs each. Every speculative output is the plain decode's, token for token.

| Placement | Drafts | Decode | Drafts accepted | Tokens a round |
|---|---:|---:|---:|---:|
| 2× RTX 3090, experts on the GPUs | none | 92.9-93.0 tok/s | | |
| 2× RTX 3090, experts on the GPUs | 1 | 130.1-130.9 tok/s | 96.6% | 1.97 |
| 2× RTX 3090, experts on the GPUs | 2 | 150.3-150.5 tok/s | 91.5% | 2.83 |
| 2× RTX 3090, experts on the GPUs | 3 | 167.8-171.0 tok/s | 95.6% | 3.87 |
| 2× RTX 3090, experts on the GPUs | 4 | 171.9-173.6 tok/s | 91.0% | 4.64 |
| RTX 3090, experts in pinned host memory | none | 39.3-44.1 tok/s | | |
| RTX 3090, experts in pinned host memory | 3 | 47.4-49.4 tok/s | 95.6% | 3.87 |

Coding answers are what MTP drafts best; a prose answer (80 tokens on the water cycle) keeps 38.6%
of three drafts, 2.16 tokens a round, and decodes at 91.6 tok/s, no faster than without drafts.
With host experts a verification's four columns route to up to four times as many experts, each
missing one crossing the bus, so three drafts gained about 12% here; after the October 9
host-expert work three drafts decode 90-123 tok/s against 87-94 plain (table above).

### Media conversations on one RTX 4090

One RTX 4090 (480 W) in a Vast container with an EPYC 7713 ran the Q2_0 release with its table and
`--vision` through `ninfer-serve`, 2026-11-07: host experts, one request at a time, the release's
DFlash2 drafts (the default), a four-turn chat whose system text is the first 24,000 characters of
the WikiText stream, a 768x512 image in the first and third user turns, greedy answers of up to 128
tokens with thinking off, each turn sending the whole history again; then a conversation with the
same system text and another question, and a text conversation with characters 30,000-54,000.
Three starts of each build, interleaved: before media prompts were cached (`d6a658b8a`), the first
build that cached them (`0c08dfefd`), and `e38464e9b`, whose first step passes the system text's end.
The client sent each request as soon as the last answer came:

| Turn | Prompt tokens | `d6a658b8a` | `0c08dfefd`: reused, prefill | `e38464e9b` |
|---|---:|---:|---:|---:|
| 1, image | 5,831 | 1.55-1.67 s | 0, 2.82-3.17 s | 1.60-1.76 s |
| 2 | 5,985 | 1.35-1.38 s | 5,958, 0.11-0.15 s | 0.12-0.14 s |
| 3, image | 6,422-6,434 | 1.45 s | 6,006, 1.03-1.31 s | 1.03-1.15 s |
| 4 | 6,572-6,586 | 1.48-1.52 s | 6,548, 0.10-0.12 s | 0.11-0.12 s |
| Same system text | 5,441 | 1.32-1.34 s | 5,417, 0.66-0.88 s | 0.75-0.78 s |

The four turns prefilled in 5.86-6.03 s, 4.07-4.64 s and 2.87-3.14 s. In `0c08dfefd` the first
turn's step ended at the system text, whose state the store keeps for the next conversation, and
the image and question after it were a second pass, which with host experts copied the experts
again; here one span runs both sides and keeps the state at the system text's end as it passes. The
third turn's 416 new tokens still take one pass that copies the experts. `e38464e9b`'s answers are
`0c08dfefd`'s; `d6a658b8a`'s differ from the first turn on, since its media prompts did not end
steps at the cache's frontiers (the end of the system text and of the last user message).

The text conversation's first turn (6,383 tokens, nothing reused, the same answer in all three
builds) took 1.60-1.63 s in `d6a658b8a` and 2.57-2.77 s in the later two. The earlier conversations'
images had taken the host pages pinned at startup (about 1.9 GB here), so storing the next images
pinned more pages while the request waited; run before the media conversation, it took 1.72-1.89 s
in all three (one start each). The store pins pages only until it holds its budget (8 GiB by
default) and reuses them after; `d6a658b8a` stored no media images. No request left the Engine the
50 ms of idle time after which it pins free pages back.

Another RTX 4090 (450 W) host with an EPYC 9334 ran the same chat on 2026-11-08 against
`e38464e9b` and this build, which pins free pages back while it is idle, with the client pausing
before each request after the first; three starts of each build, interleaved, the server on the
GPU's NUMA node (`taskset`). Prompt times as the server reports them, `e38464e9b` / this build:

| Client pause | 0 | 0.1 s | 3 s |
|---|---:|---:|---:|
| Text conversation, turn 1 | 2.10-2.21 / 2.14-2.17 s | 2.07-2.10 / 2.06-2.07 s | 2.06-2.08 / 1.49-1.50 s |
| Text conversation, turn 2 | 0.25-0.30 / 0.29-0.30 s | 0.26 / 0.24-0.25 s | 0.26-0.27 / 0.09-0.10 s |
| Same system text | 0.43-0.45 / 0.43 s | 0.42-0.44 / 0.27-0.28 s | 0.42-0.45 / 0.13 s |
| Media turns 2 and 4, client's wall time | 0.23, 0.43-0.44 / 0.23, 0.43-0.44 s | 0.23-0.24, 0.43-0.45 / 0.25-0.26, 0.46 s | 0.24, 0.45 / 0.24-0.25, 0.44-0.45 s |

With 3 s pauses the free pages were pinned back before each request, so storing images no longer
pinned pages while a request waited: the text conversation's first turn took less than it takes as
the first request after startup (1.61-1.62 s in both builds, two starts with the text conversation
first). With 0.1 s pauses about one chunk was pinned before the next request came, and a short turn
that came during a chunk waited for it, up to 30 ms. Without pauses both builds took the same times.
A 64 MiB chunk pinned in 32 ms on this host (median of 24, at most 35 ms; 128 MiB took 61 ms). Both
builds gave the same answers in every turn. This build decoded at 0.999-1.008 of `e38464e9b`'s rate
a turn without pauses, 0.986-0.998 with 0.1 s pauses and 0.980-1.023 with 3 s pauses (geomeans
1.003, 0.992 and 0.999); it pins nothing while a request runs.

With `--spec mtp --draft-tokens 3` the media requests draft. They decoded the four answers at 135.7,
178.5, 118.6 and 163.7 tok/s, against 70.9, 88.4, 87.6 and 99.2 with `--spec none`, the same tokens
(the media cache's first build, one start each, on an earlier rental of the same kind). Without
drafts the three builds decode at the same rate: over the chat above and the same chat without its
system text, two starts each with `--spec none`, 78.7, 79.9 and 81.6 tok/s, and `d6a658b8a` with
`--spec mtp` (which did not draft media requests there) 77.3; `e38464e9b` against `0c08dfefd`, with
the same answers, 0.90-1.10 a turn. With DFlash2 drafts it decoded the table's answers at 0.78-1.21
of `0c08dfefd`'s rate a turn, 1.3% and 0.4% below it overall in two sets of three starts; the second
window of DFlash2 features that a span keeps for one sequence takes 50 MiB of workspace and left the
expert cache 89 MiB smaller.

### Hyper-connections in Q8_0

The hyper-connection projections (97 down/up pairs and 96 inject matrices) are stored as ggml Q8_0,
0.67 GB instead of 1.27 GB in BF16. On the rented RTX 3090 of the tier measurements, plain decode
went from 81.3-93.6 to 84.5-97.7 tok/s and decode with three MTP drafts from 84.5-114.4 to 83.0-120.4.
Up to eight tokens the kernels decode the weights to exact FP32 values; wider calls round them to
BF16 for the tensor-core products, which costs prompts and batches of nine or more tokens 5-57% more
for these reads. Perplexity with host experts (4,096-token windows advancing by 2,048, int8 KV):

| Corpus | BF16 | Q8_0 | Change |
|---|---:|---:|---:|
| `perplexity-1m` | 3.93783 | 3.93842 | +0.015% |
| `perplexity-heldout-2026-09` | 4.69573 | 4.69673 | +0.021% |

### Against llama.cpp

llama.cpp runs the same GGUFs with the experts on the CPU (`--n-cpu-moe 48`). On the IQ3_S host
above (23 threads) llama-bench gives 29.6 tok/s decode (tg128) and 312 tok/s prefill (pp512) for
IQ3_S, and 12.7 and 368 tok/s for Q2_0; the Q2_0 RTX 3090 host (32 threads) gave 11.2 and 267
tok/s for Q2_0. Its speed follows the host CPU, and its Q2_0 CPU path is the slower one.

Perplexity agrees with llama.cpp: over the first 72 KB of the WikiText sample in
`eval/corpora/perplexity-1m`, 2,560-token windows advancing by 1,024 targets (llama.cpp's
`--ppl-stride 1024 -c 2048`, which widens the window to 2,560), the fourteen windows both evaluate
identically (14,336 targets) give, with a BF16 KV cache in both:

| Release | NInfer | llama.cpp | Window by window |
|---|---:|---:|---|
| Q2_0 | 2.6579 | 2.6502 | -0.014 to +0.016 nats |
| IQ3_S | 2.1317 | 2.1278 | -0.022 to +0.016 nats |

Q2_0 read 2.6558 on the RTX 3090 before wide MoE calls moved to ggml's matrix kernel and wide
hyper-connection reads to BF16 GEMMs. On an L40S the four combinations of the two changes give
2.6489 (both, today's kernels), 2.6494 (the matrix kernel alone), 2.6495 (neither) and 2.6555 (the
GEMMs alone): rounding-order differences within 0.25% that move with the kernel mix and the GPU,
not a loss from either change.

The two agree as closely deep into a long context. On the four PG-19 streams of `perplexity-1m`
joined (261,412 tokens), 65,536-token windows advancing by 32,768 targets (llama.cpp's
`--ppl-stride 32768 -c 49152`) score only the last 32,768 positions of each window, all of them
far past the 2,051 below which the sparse layers attend densely. The five windows both evaluate
identically (163,840 targets), on two RTX 5090s with a BF16 KV cache:

| Window | NInfer | llama.cpp | Difference |
|---|---:|---:|---:|
| 1 | 7.6956 | 7.6874 | +0.0011 nats |
| 2 | 8.9497 | 8.9476 | +0.0002 nats |
| 3 | 9.8699 | 9.8700 | 0.0000 nats |
| 4 | 9.0243 | 9.0153 | +0.0010 nats |
| 5 | 5.6261 | 5.6275 | -0.0002 nats |
| All five | 8.0834 | 8.0802 | +0.0004 nats |

NInfer is the CUDA 12.9 `120a` build that served the evaluation below, llama.cpp `a7fb71f` with
CUDA 12.9 and every layer on the GPUs (2026-10-06). llama.cpp's window values come from its
running perplexity, printed to four decimals, so they are good to about 0.0003 nats. Perplexity
runs prompt passes only; the decode steps that wrote the evaluation's answers are not part of it.

### Reasoning benchmarks

Accuracy on the two reasoning benchmarks of the GSQ-RCO card that EvalScope scores without a code
sandbox, for the Q2_0 release with its table, every expert on two RTX 5090s (`--devices 0,1`), a
BF16 KV cache and six requests at a time: thinking on, temperature 1.0, top-p 0.95, top-k 20 and
one sampled run as in the Qwen3.8-27B campaigns, but at most 106,000 output tokens, what six
sequences hold on the two cards beside the weights
([`eval/configs/qwen3_8_flash_next_reasoning.yaml`](../eval/configs/qwen3_8_flash_next_reasoning.yaml),
2026-10-05). The answers cut at that limit were then continued from where they stopped up to the
27B campaigns' budgets, 122,880 output tokens for AIME and 245,760 for GPQA:

| Benchmark | NInfer, 106,000 tokens | Cut there | NInfer, 27B budgets | Q2_0, model card | BF16, model card |
|---|---:|---:|---:|---:|---:|
| AIME 2025 | 93.33 (28/30) | 1 | 93.33 (28/30) | 96.67 | 100.00 |
| GPQA-Diamond | 84.34 (167/198) | 10 | 86.36 (171/198) | 89.39 | 91.92 |

The card does not state its sampling, output limit or number of runs. A cut answer has given no
answer and counts as wrong; of the answers that finished within 106,000 tokens, 28 of 29 (AIME)
and 167 of 188 (GPQA) are right. One run of GPQA-Diamond's 198 questions has a standard error of
about 2.6 points, so the 106,000-token score sits two standard errors below the card. Nine of the
ten cut GPQA answers were still reasoning coherently at the limit and one was repeating a codon of
its question's DNA sequence; the cut AIME answer was still calculating. Continued by the same server
binary with the same sampling (the request's rendered chat prompt and the reasoning so far, sent as
a raw prompt; EvalScope's own answer extraction scores the result), four of the ten GPQA answers
came out right and four wrong, and two ran out without an answer, one still reasoning at 245,760
tokens and the looping one at the server's 247,000-token context; the AIME answer reached 122,880
still calculating. With those budgets GPQA-Diamond sits 3.0 points below the card, about 1.2
standard errors.

The answers are long: AIME 2025's averaged 26,100 output tokens (median 13,700), GPQA-Diamond's
23,200 (median 9,100), and 27 GPQA answers ran past 65,536. The run took 7 h 44 min (AIME
1 h 12 min, GPQA 6 h 33 min): 5.37 million output tokens at 193 tokens per second across the six
requests. The server was a `120a` build of `3e842a91` with CUDA 12.9, from before `120a` builds
moved to CUDA 13.1; the kernels 12.9 was found to miscompile serve an NVFP4 KV cache, which this
run did not use, and the same binary's prompt passes match llama.cpp over the long windows above.
