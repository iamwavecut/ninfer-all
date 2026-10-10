# Qwen3.8-Flash-Next

NInfer runs Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`, 125B text parameters, about 6B
active per token) from ISTA-DASLab's GSQ-RCO GGUF releases, converted without requantization:

- [Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF):
  Q2_0 (37.6 GB), IQ2_XS (39.2 GB), IQ3_XXS (47.0 GB), IQ3_S (54.8 GB);
- [Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF).

Each release is two GGUF shards: the model, and the 28.8 GB n-gram table of the model's per-layer
embedding (PLE), byte for byte the same in every release, the Coder build's included. A NInfer
model artifact describes the table it reads in its `ngram` component (the hash constants, the row
format and the SHA-256 of the rows) and either stores the rows too, as one self-contained file, or
leaves them to a table artifact of their own that every Flash-Next model can share. Nothing reads
the table at load; each token reads the 16 rows it addresses from the file, unless the table, or the
part of it a profile ranks most used, is loaded into RAM ([the n-gram rows](#the-n-gram-rows)). A
model stored without its rows takes them from `--ngram-table PATH`, which
must hold the table the model names; without a table the engine refuses to start
([running without it](#without-the-n-gram-table) is an experiment, not a mode).

The published conversions store the models without the table, which is published once:

| Artifact | Size | |
|---|---:|---|
| [n-gram table](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3) | 26.92 GiB | IQ4_NL rows and an optional broad hot-row profile |
| [Q2_0](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3) | 38.49 GiB | Vision and shared-Q8_0 MTP |
| [IQ3_S](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3) | 54.50 GiB | Vision and shared-Q8_0 MTP |
| [Coder IQ1_M](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-NInfer-v3) | 31.02 GiB | 256 text experts per layer, Vision and shared-Q8_0 MTP |

The model runs with up to eight concurrent requests, a context cache of prompt prefixes, structured
output, images and video through its Vision tower (`--vision`, from an artifact converted with the
tower), and MTP speculative decoding (`--spec mtp`). The three published NInfer models above now
include Unsloth's shared-Q8_0 MTP block. The upstream GSQ-RCO GGUF sources omit it, so new
conversions must supply it separately; see [MTP](#mtp-speculative-decoding).

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

The native-Q2 ranges `3..5` and `1..4` were qualified on RTX A6000 with CUDA 13.1 and int8 KV:
all 3,125/4,691 selected bindings and weight bytes matched the full native artifact, and the
public Engine produced exact repeats for 32/700-token inputs and 16 generated tokens. The first
range has no PLE and reads no n-gram rows; the second rebases PLE to layer 0 and requires its table
companion. Both missing-companion and unused-table refusal paths passed. These checks establish
slice conversion and execution behavior, not full-model quality.

`qwen3_8_flash_next_gsq_q2` imports the trained Q2_0 routed experts into `q2_g64_fp16` planes
with exact code and signed-scale words. Q2_0/Q8_0 shared projections retain their codes/scales as
native Q2/Q8; other shared block formats decode and round to BF16. Other projections and the MTP
block keep their GGUF representations. Native banks support device residency and CPU/GPU hybrid
host residency at concurrency one; native disk residency is not implemented. A8 uses grouped integer MMA from 16 columns,
decoding the stored native codes in registers without repacking the weights.

On October 8, all 73,872 routed/shared native projections were checked against their source
blocks (exact code/scale words for Q2/Q8; BF16 casts for other shared formats). The packed-plane
import took 171.4 s versus 531.2 s for the earlier unpack/repack implementation on the same
Xeon 8167M host, CPU PyTorch on Python 3.11, 12 OpenMP threads and 65,536-row chunks. These
are individual runs, not repeated timing distributions; the newer run also preserved shared
Q2 projections that the first recipe unnecessarily cast to BF16. The resulting text+MTP artifact
is 40,651,086,080 bytes and reuses the original table artifact. Its full-model qualification
initially exposed mixed shared gate/up formats; each projection now keeps its own format and
activation permission.
Its public Engine functional suite now passes K=1/4/8/15, budgets, cancellation, repeats,
logprobs and cache isolation: zero failures, zero plain/MTP differences and five batch-composition
differences in the tested requests. Live-prefix and disk-prefix MTP continuations match exactly.
Cold prefill (512+218 columns) differs from prefix reuse (512+188, then 30); the cache test
records that difference and compares live/disk continuation with the same computation history.

On October 8, the public Engine scored the native-Q2 and original GGUF-Q2 artifacts on the same
391,000 targets in the held-out quick corpus (18 windows, context 32,768, stride 16,384, BF16 KV,
RTX A6000, CUDA 13.1). Both use the same standalone IQ4 table. Overall PPL is 4.991555 for GGUF
and 5.004501 for native (+0.2594%, mean NLL +0.002590). The six domain changes are:

| Domain | GGUF PPL | Native PPL | Change |
|---|---:|---:|---:|
| Wikipedia English | 6.311970 | 6.328184 | +0.2569% |
| Wikipedia Chinese | 16.387842 | 16.439307 | +0.3140% |
| arXiv | 7.621604 | 7.732637 | +1.4568% |
| GitHub code | 2.240673 | 2.233451 | -0.3223% |
| Own code | 2.286422 | 2.285044 | -0.0603% |
| Synthetic chat | 3.804479 | 3.801502 | -0.0783% |

Nine windows have higher NLL and nine lower. The worst window is arXiv index 2 (+4.2972% PPL,
NLL +0.042074); the best is Chinese Wikipedia index 1 (-5.1888%, NLL -0.053283). Scoring took
483.46 s for GGUF and 555.13 s for native (+14.83%); this is one sequential pair, not a timing
distribution. These results quantify a quality difference and a measured scoring regression;
they do not establish native throughput parity or qualify the separate HF Q4/Q5/FP8 recipe.
Native is an explicit artifact choice for native formats and host DMA controls; it does not
replace optimized GGUF. Retain the GGUF artifact route when its measured quality and prefill
advantages matter. The native arithmetic passes its independent oracles, but these full-model
and Op costs remain limitations.

The native vector Op passed independent FP64 qualification for BF16/Q2/Q4/Q5/Q6/Q8, both
layouts and A16/A8, including graph replay and fixed-profile repeats. On the same real layer-12
Q2 weights, H=2560/I=640/top-10, a single RTX 3090, CUDA 13.1, nine timed samples after warmup:

| T | GGUF warm median, reused/disjoint experts, ms | Native A8, ms | Native A16, ms |
|---|---|---|---|
| 1 | 0.094208 / 0.094016 | 0.089920 / 0.089088 | 0.099328 / 0.098304 |
| 2 | 0.141312 / 0.141312 | 0.139264 / 0.141312 | 0.158816 / 0.163840 |
| 5 | 0.284672 / 0.282624 | 0.289792 / 0.290816 | 0.331776 / 0.339968 |
| 8 | 0.435200 / 0.431104 | 0.448512 / 0.449536 | 0.518144 / 0.533504 |

A8 gains 4.6–5.2% at T=1 and regresses up to 4.3% at T=8; A16 regresses 4.4–23.8% across
these cases. Relative L2 versus the original packed-weight FP64 oracle is 1.58–1.61e-7 for A16,
0.00909–0.00946 for A8 and 0.00911–0.00961 for GGUF. All three repeat exactly. The benchmark
also computes a zero-weight shared expert; its CPU oracle excludes that contribution. Samples
after a 32 MiB memset were separately recorded, but that operation's actual L2 eviction is
unproven. These are Op results; the full-model quality comparison above and host-throughput
measurements below have their own workloads and limits. The native Op keeps FP32 middle/products and allocates activation scratch even
for A16; no workspace reduction is claimed.

The subsequent grouped-MMA qualification passed at T=1/2/8/9/15/16/17/31/32/33/65/129,
including mixed gate/up formats, both layouts and graph replay. A8's wide route groups up to
32 columns per expert. The same real layer-12 probe, nine warm samples per cell, gave:

| T | GGUF, reused/cyclic experts, ms | Native A8, ms | Native A16, ms |
|---|---|---|---|
| 16 | 0.329 / 0.702 | 0.270 / 0.885 | 0.980 / 1.006 |
| 31 | 0.368 / 0.730 | 0.394 / 0.956 | 1.867 / 1.930 |
| 32 | 0.366 / 0.728 | 0.385 / 0.959 | 1.956 / 2.018 |
| 33 | 0.372 / 0.723 | 0.427 / 0.963 | 2.019 / 2.067 |
| 64 | 0.506 / 0.754 | 0.445 / 1.008 | 3.816 / 3.951 |
| 128 | 0.788 / 0.950 | 0.565 / 1.143 | 7.773 / 8.026 |
| 512 | 1.488 / 1.601 | 1.525 / 1.863 | 31.076 / 32.005 |

Wide inputs repeat the eight represented BF16 columns; cyclic routing uses 80 experts, not
disjoint sets for every wide column. All routes pass the original FP64 oracle and exact repeats.
A8 ranges from 28.3% faster (T=128, reused) to 33.7% slower (T=64, cyclic) than GGUF here;
A16 is slower in every wide case, up to 20.9 times at T=512, reused. At T=1/2/5/8 in this run,
A8 ranges from 4.8% faster to 12.3% slower. The earlier eight-column MMA experiment took
2.919/2.906 ms at T=512; 32-column grouping reduces those times but increases T=32/reused
from 0.310 to 0.385 ms. Earlier vector measurements above remain relevant; differences between
runs have not been attributed. These results qualify the implementation, not an overall speedup.

The October 8 A16 update groups routed columns from T=16 and uses BF16 tensor-core dots.
Integer weight codes remain exact; stored FP16 scales are applied after each 32-value dot.
The FP32 SwiGLU intermediate uses a BF16 high part plus a BF16 residual for the down projection.
It reuses the existing workspace capacity and route-group scratch; no persistent weight
repacking or new model artifact is needed. A8's arithmetic remains as above.

One RTX 3090, CUDA 13.1, the same layer-12 Q2 fixture and eight repeated BF16 input columns,
nine timed samples after warmup, reused/cyclic-80 expert selection:

| T | Retained vector A16, ms | Grouped A16, ms | GGUF, ms | Native A8, ms |
|---|---|---|---|---|
| 1 | 0.098304 / 0.098304 | 0.089088 / 0.089088 | 0.083968 / 0.083840 | 0.080672 / 0.080896 |
| 2 | 0.158720 / 0.162816 | 0.144352 / 0.148512 | 0.124768 / 0.124928 | 0.125952 / 0.126976 |
| 5 | 0.331776 / 0.339968 | 0.299008 / 0.308224 | 0.251808 / 0.249856 | 0.260096 / 0.262144 |
| 8 | 0.518048 / 0.532416 | 0.470016 / 0.519168 | 0.385024 / 0.382976 | 0.415776 / 0.417792 |
| 16 | 0.997376 / 1.021950 | 0.538624 / 1.044480 | 0.323584 / 0.692224 | 0.270336 / 0.883712 |
| 31 | 1.894400 / 1.941500 | 0.800640 / 1.141760 | 0.364544 / 0.726848 | 0.394016 / 0.959488 |
| 32 | 1.940480 / 1.975300 | 0.777216 / 1.129470 | 0.362496 / 0.726016 | 0.381952 / 0.964608 |
| 33 | 2.030590 / 2.045950 | 0.805888 / 1.125380 | 0.369664 / 0.720736 | 0.424960 / 0.961536 |
| 64 | 3.903490 / 3.961660 | 0.878592 / 1.164290 | 0.502784 / 0.750592 | 0.441344 / 1.021950 |
| 128 | 7.772160 / 7.913470 | 1.241980 / 1.480700 | 0.785440 / 0.874336 | 0.564192 / 1.144830 |
| 512 | 30.892000 / 31.870800 | 3.268610 / 3.350530 | 1.483650 / 1.598460 | 1.526690 / 1.855490 |

Grouped A16 improves 13 of 14 wide cells by 41.2–89.5% over the retained vector executable;
T=16/cyclic regresses 2.2%, retained as an explicit exception. A16 remains 50.9–120.3% slower
than the optimized GGUF route in these wide cells. The T<16 route is still the vector kernel;
its 2.5–9.9% timing differences between executables are unattributed and are not credited to
the grouped implementation. Post-memset measurements are retained separately; the 32 MiB
memset is not proof of a cold cache. No end-to-end speed or model-quality improvement is claimed.

The existing FP64 oracle bounds were unchanged. BF16/Q2/Q4/Q5/Q6/Q8, both layouts, mixed
projections, boundary widths, fixed-mode repeats and CUDA Graph replay passed in 12.22 s.
On the real Q2 fixture, A16 relative L2 increases from about 1.6e-7 to 2.4e-6 for wide calls;
all 22 oracle/repeat cases pass. The public native-Q2 Engine suite passed with
`target_differences=3`, `batch_differences=7`, `failures=0` (previous qualification: 0/5/0).
These mode differences remain visible under the agreed numerical/repeatability contract.
The artifact's current activation permissions select A8 for its quantized expert banks;
the Engine suite establishes integration and repeatability, not full-model A16 throughput.
`scripts/pods/native_a16_checks.sh` runs the qualification;
`scripts/pods/compare_native_moe.py JOB_DIRECTORY` compares every saved probe cell.

The fused HC write/read uses the public FP32 stack boundary and supports BF16/FP32 branch
outputs. On October 8, its independent FP64 oracle and changed-input graph checks passed on
RTX 3090 / CUDA 13.1. Against separate public write/read calls, 61 paired graph samples per
cell cover T=1/4/8/9/16/128/512, both dtypes, and reused/64 MiB-evicted L2 inputs. Medians are
2.0–6.6% lower in 26/28 cells; two T=1 cells tie, with no regressions in this matrix. One graph
node is removed per call. The text/MTP executor integration passes the full FP8-table Engine
check, but its whole-model speed effect has not been measured.

The QSA indexer uses three TF32 high/residual MMA products for T>=128 and keeps its original
SIMT route for narrower calls. Public pooled keys remain FP32. Independent FP64 selection,
route-boundary and graph checks pass on RTX 3090 / CUDA 13.1. With 33 rotated samples per
candidate and reused/64 MiB-evicted L2, all 12 T=128/512 cells at 700/8192/32768 pooled-key
capacities have 30.8–65.0% lower medians. MMA regresses short calls by up to 40%, so it is
not selected there. The wide path adds 2048 bytes of workspace per token and one graph node;
no whole-model or other-GPU improvement is established. BF16 pooled-key storage was rejected:
166/288 synthetic queries change their selected set beyond the existing near-tie allowance.

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

The FP8-table experiment below was rejected by the user on October 8 because its table is
80% larger. IQ4_NL is the selected release representation. The FP8 table and companion model
remain private archives; their measurements are retained for reproducibility.

HF table conversion reads the numbered BF16 shards in bounded chunks. It makes two passes:
one to hash the exact quantized bytes named by the model descriptor, then one to write them.
This needs no table-sized temporary file, but reads and quantizes the source twice. The full
320,001,536-row payload is 51,840,248,832 bytes, 80% more row storage than IQ4_NL.
The complete 51,840,252,928-byte artifact and its native-Q2/MTP companion were converted and
archived privately on October 8; uploaded sizes, SHA-256, visibility and metadata were verified.
The companion preserves all model weights and resources and changes its table descriptor.
The full table passes public Engine plain/K=4 fixed-mode repeats and cancellation/recovery with
direct I/O and native host experts. The final quality comparison below uses the public scoring
Engine. Models naming an IQ4_NL table cannot substitute an FP8 table at runtime:
the companion must record the FP8 table's format and digest.

On two RTX 3090s / CUDA 13.1, the same executable scored the native-Q2 weights with each table,
BF16 KV, context 32,768 and stride 16,384, over 391,000 held-out targets in six streams and
18 windows. Fourteen windows improve with FP8 and four worsen. The worst change is +0.2624%
on the first GitHub-code window; the best is -13.2749% on the last arXiv window.

| domain | IQ4_NL PPL | FP8 PPL | change |
|---|---:|---:|---:|
| Wikipedia English | 6.372558 | 6.360064 | -0.1961% |
| Wikipedia Chinese | 16.565810 | 16.091824 | -2.8612% |
| arXiv | 7.995071 | 7.642498 | -4.4099% |
| GitHub code | 2.235469 | 2.237799 | +0.1043% |
| own code | 2.282026 | 2.284420 | +0.1049% |
| synthetic chat | 3.804761 | 3.799902 | -0.1277% |
| overall | 5.045081 | 4.982153 | -1.2473% |

Scoring took 443.49 seconds with IQ4_NL and 446.33 with FP8 (+0.6419%) in one sequential pair;
this does not establish a stable speed difference or a generation-throughput advantage. The
earlier A6000 native-Q2 result (5.004501) used another source snapshot and hardware configuration,
so it cannot isolate the later optimizations' quality effect. These table results do not erase
the native-versus-GGUF regressions above. FP8 remains a rejected experiment with 80% more
table storage; both code-domain aggregates worsen slightly.

The converter/subset/FP8 source/table suite passes 131 tests. Three real FP8 expert matrices
(4,915,200 values) match an independent FP64 codebook/block-product oracle exactly after
FP32 rounding. This establishes source interpretation, not full-model quality.
Six affected C++/GPU checks also pass
on RTX 3090 with CUDA 13.1, including the Python-writer/reader/GPU chain across artifact part
boundaries. A live check on the pinned revision fetched the
5,120-byte `mtp.pre_fc_norm_embedding.weight` range and matched the stored file's SHA-256;
the complete MTP subset has not been downloaded in this qualification.

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

Every matrix keeps the block type the release chose (see [GGUF block formats](gguf.md)); the expert
banks keep the exporter's expert-major layout, so one expert is one contiguous range of bytes. The
recipe undoes llama.cpp's exporter conventions as the Qwen3.8-27B GGUF recipe does (grouped GDN
value heads, `1 + w` norms, `A_log`, the head-interleaved query and gate). The n-gram table keeps
the release's IQ4_NL rows, and its component carries the hash constants they were written for; the
runtime derives them again from the model's configuration and refuses a table that disagrees, or a
table artifact whose digest or row format differs from the one the model names.

## Run

```bash
# One GPU with room for every expert (48 GB for Q2_0, the 96 GB RTX PRO 6000 for IQ3_S).
./build/apps/ninfer-serve models/flash-next-q2_0.ninfer --ngram-table models/flash-next-ngram-table.ninfer \
  --max-context 32768

# Two 24 GB GPUs (two 32 GB for IQ3_S): every expert in device memory, one pipeline stage per GPU (Linux).
./build/apps/ninfer-serve models/flash-next-q2_0.ninfer --ngram-table models/flash-next-ngram-table.ninfer \
  --devices 0,1 --max-context 32768

# One GPU: the experts in pinned host memory, the most used of them cached on the GPU.
./build/apps/ninfer-serve models/flash-next-q2_0.ninfer --ngram-table models/flash-next-ngram-table.ninfer \
  --expert-residency host --max-context 32768

# One GPU and little RAM: the experts stay in the artifact's files and stream into a GPU cache.
./build/apps/ninfer-serve models/flash-next-q2_0.ninfer --ngram-table models/flash-next-ngram-table.ninfer \
  --expert-residency disk --max-context 32768
```

A model converted with its table needs no `--ngram-table`. `ninfer` and `ninfer-perplexity` take
the same options, and the Docker image's `serve` command takes them with the files under `/models`
([Running](../README.md#running)).

| Option | Meaning |
|---|---|
| `--expert-residency device\|host\|disk` | expert banks in the stage devices' memory (default); in page-locked host memory that the expert kernels read across the bus; or left in the artifact's files, each layer's routed experts read into a device cache before they run |
| `--expert-cache-mib N\|auto` | with host or disk experts, device memory for the most used experts: `auto` (default) takes what each device has free after startup less a margin, and after the warm-up what is still free beyond 640 MiB; `N` is the cache's size, which the warm-up leaves alone; `0` disables the host-mode cache (disk mode needs one) |
| `--expert-misses staged\|mapped` | GGUF host experts: a decode or verification call copies each routed expert its device cache lacks straight into the slot of the layer's least recently used expert of the lowest frequency tier, by the copy engine while the cached ones run, and keeps it there (`staged`, the default); or its expert kernels read the missing experts across the bus and the cache admits between passes (`mapped`) |
| `--expert-dma-share F` | host experts: fraction of a call's missing experts the GPU runs (copied to it), `0..1` (default `1`); below `1` the CPU computes the rest from RAM while the GPU runs the cached ones, in its own arithmetic, and their slots fill behind the call; prefill always runs on the GPU. Neutral on an RTX 3090 with an 8-core CPU |
| `--expert-cpu-threads N` | host experts with a CPU share: `1..256` CPU workers; default is the physical cores less two (counted as half the logical threads), at most 16 |
| `--expert-cache-adaptive` | replace cold cached native experts between calls; off by default because it changes the CPU/GPU arithmetic partition |
| `--expert-profile PATH` | fill the host expert cache at startup from counts recorded for the same artifact (`--expert-profile-out`) |
| `--expert-profile-out PATH` | record every layer's expert routes after requests, for a later `--expert-profile` |
| `--ngram-table PATH` | the table artifact to read the n-gram rows from; required for a model stored without its table, and it must hold the table the model names (same SHA-256 and row format) |
| `--ngram-residency disk\|ram\|ram-hot` | where the n-gram rows come from: the table's file, 16 rows a token (default); the whole 28.8 GB table in RAM; or the rows a hot-row profile ranks first in RAM and the rest from the file ([the n-gram rows](#the-n-gram-rows)) |
| `--ngram-io buffered\|direct\|mmap` | how rows are read from the file: positioned reads through the OS page cache (default), reads past it (`O_DIRECT`, `FILE_FLAG_NO_BUFFERING`), or copies out of a mapping |
| `--ngram-io-depth N` | row reads in flight (1 to 1024, default 64) |
| `--ngram-hot-profile PATH` | `ram-hot`: override the selected table's embedded profile with a `ninfer-ngram-profile` file |
| `--ngram-ram-mib N` | `disk`: cached rows and their index (default `0`, off); `ram-hot`: resident profile rows and their index (default 4096 MiB) |
| `--ngram-lock` | `ram`, `ram-hot`: lock the resident rows in physical memory (`mlock`, `VirtualLock`; needs the memlock limit or the privilege) |
| `--no-ngram-table` | run without the n-gram table: see [below](#without-the-n-gram-table) |
| `--devices A,B,...` | one pipeline stage per GPU; layers are split so that every stage holds about the same stored bytes (`--stage-layers` overrides) |

Every expert runs on the GPU by default. GGUF host experts can give the CPU a share of a call's
missing experts (`--expert-dma-share`); native-format experiments are retained for reference. The
existing GGUF artifacts support both host and disk residency. With host experts every expert
layer gets the same number of cache slots (the MTP block's wider experts take more of the bytes);
staged misses fill the text layers' slots by recency within tiers of decayed route counts (about
once lately, a few times, often), so a passing decode does not evict a working set other prompts
keep routing to, and a prompt chunk lets the cache admit its
most routed experts by decayed counts; disk residency uses a CLOCK device cache. Disk residency
relies on the OS page cache for repeated file reads. An additional application-managed RAM
cache of experts is excluded from the current delivery. The device cache and bounded transfer
buffers remain in use.

With host experts the GPU holds only the dense weights (3.7 GB for the Q2_0 release), the
per-request state and the expert cache; the host needs the expert banks in page-locked memory
(34 GB for Q2_0). With disk experts the host needs no copy of the banks at all: the expert cache
reads the missing experts from the artifact's files through the OS page cache, eight reads in
flight, into a 256 MB page-locked staging ring, so the page cache keeps whatever the system can
spare and the rest comes from the disk.

Native host banks copy every miss to the GPU by default (`--expert-dma-share 1`). Values below
one enable experimental CPU mixing; CPU results merge in route order. A GGUF bank, including a GGUF MTP companion beside
native text weights, retains its mapped-host GPU route; native routing profiles cover only
native banks. Linux registers native weight objects separately; Windows uses pageable weights
and bounded pinned staging. Use `--max-concurrency 1` for native host execution.

On October 8, RTX 3090 / CUDA 13.1 / EPYC 7302P with 256 GB RAM passed the native hybrid
FP64 oracle, scalar/AVX2 CPU tests, graph replay and stalled-worker cancellation/recovery.
EPYC 9354 / Linux / GCC 13.3 also passed all 33 AVX-512 VNNI oracle cases. The full native-Q2
plus GGUF-MTP artifact passed public Engine plain/K=4 repeats, active cancellation/recovery
and profile recording. DMA=1 is the qualified host mode; further CPU-mixing qualification and
Windows execution are excluded from this delivery by the user's October 8 decision.

The full-model logit comparison used BF16 KV, prefill chunks of 512, concurrency one, no drafting,
prefixes of 128 and 2,112 tokens, and 16 teacher-forced positions after each prefix. Every BF16
logit word matched between one RTX 3090 with host experts at DMA=1 and two RTX 3090s with resident
experts. All modes repeated exactly twice; all 248,077 public-vocabulary logits were finite.
CPU shares at DMA=0/0.5 kept top-1 at all 32 positions, but maximum relative L2 error versus the
resident logits was 0.324/0.604 (maximum KL 0.00761/0.00194). These CPU modes have not passed
full-model numerical qualification; top-1 agreement does not establish it.

Public `/completion` measurements on the single RTX 3090 / EPYC 7302P used the same 219 input
tokens and 64 output tokens, int8 KV at context 4096, a 4 GiB expert cache, eight CPU workers,
concurrency one and no prefix reuse. Each row contains all three requests: the first followed
verified eviction of the table's page cache, then two warm requests. Decode uses the 63
intervals after the first output token. All output tokens matched across these configurations.

| DMA share / MTP / table I/O | decode tok/s, three requests | median tok/s | peak process RSS, GiB |
|---|---|---:|---:|
| 0 / K=4 / buffered | 3.106, 3.107, 3.124 | 3.107 | 35.19 |
| 0.5 / K=4 / buffered | 5.221, 5.265, 5.264 | 5.264 | 35.19 |
| 1 / K=4 / buffered | 31.294, 31.376, 31.388 | 31.376 | 35.19 |
| 1 / K=4 / direct | 31.450, 31.451, 31.440 | 31.450 | 35.20 |
| 1 / plain / buffered | 23.990, 23.984, 23.977 | 23.984 | 32.68 |

K=4 improves the median decode rate by 30.8% over plain for this workload. CPU mixing is
5.96–10.1 times slower than DMA=1. The GGUF baseline process failed at startup, so this grid
does not compare native execution with optimized GGUF. The host had 256 GB RAM; process RSS
does not qualify a physical 64 GB host, and the OS page cache can consume additional memory.

### The n-gram rows

A pass reads the 16 rows each of its tokens addresses (1,440 bytes a token in IQ4_NL) as soon as it
has hashed them, `--ngram-io-depth` reads at once, and uploads them just before the PLE layer
(block 1): the GPU embeds the tokens and runs block 0 while the rows are read, and waits only if
the read takes longer. `--ngram-io` chooses how the file is read. `buffered` (the default) goes
through the OS page cache, which then keeps a 4 KiB page for each row it read; `direct` bypasses the
OS page cache, so every row-cache miss reads from the drive; `mmap` copies the rows out
of a read-only mapping of the file.

`disk` keeps recently read rows in a four-way CLOCK cache. `--ngram-ram-mib` bounds its
allocation including row tags and CLOCK state, independent of the table's size. The cache is off
by default; for example, `--ngram-ram-mib 4096` enables a 4 GiB budget. Its empty index and row storage use demand-paged
memory; the Linux cache avoids huge pages so one short row does not fault 2 MiB. The budget is
allocated capacity, not current RSS; buffered and
mapped reads can also occupy the OS page cache. Failed batches admit no rows.

The Engine hints the next prompt chunk's rows while the current chunk runs. MTP hints its anchor
and both highest-logit candidates at each draft step. An auxiliary CUDA stream copies each
step's two tokens and hashes private contexts while later draft steps run. Verification also
hints both candidates after each actual verified prefix; predictions do not change the next
column's context. Linux buffered/mapped readers
issue `POSIX_FADV_WILLNEED`; Windows mapped readers use `PrefetchVirtualMemory`. These are advisory
reads, with duplicate rows and shared pages coalesced. Direct I/O and Windows buffered I/O start one bounded
speculative batch in separate staging; busy workers skip extra hints. Only complete successful
batches can supply demand reads, and hints alone admit no rows into the persistent cache.
Windows buffered staging remains unqualified on Windows. Eager and graph checks cover B=1/8, 15 draft steps,
16 verify columns, EOS and callback-error recovery. An initial attempt to synchronize a captured
event failed; completion now waits for the joined compute stream, and the corrected test passes.
The full FP8-table Engine plain/K=4 functional check also passes. Whole-model speedup remains unmeasured.

Buffered/direct row misses use a bounded OS queue: io_uring on Linux and overlapped reads with
IOCP on Windows. One worker drives the queue while device execution continues; direct reads use
at most `--ngram-io-depth` aligned bounce buffers. A failed batch drains its outstanding reads
before returning and admits no rows to the cache. If Linux disables io_uring (as the October 8
A6000 rental did), the reader uses positioned-read workers. Bulk RAM/hot-set loading and mapped
reads retain their existing paths.

The Linux queue and integrated reader passed exact byte, split-row, duplicate, bounded-depth,
short-file/retry and overlapping-batch checks on the local Linux 7.0.14 container with GCC 12.2;
native queue depths 1/5/64 covered buffered and direct reads. The restricted A6000 host passed
the fallback reader checks. Bounded direct lookahead also passes exact native/fallback checks,
including failed batches, cache admission and destruction while reads are pending. RTX 3090 /
EPYC 7302P / CUDA 13.1 passes the updated reader, draft/verify ordering and full FP8-table Engine
tests. Windows IOCP execution remains untested and is excluded from the current qualification.

The final public Engine I/O run used the serving workload above, DMA=1, K=4 and no persistent
row cache, on the provider's Samsung MZQLB7T6HMLA-00007 SSD. Three buffered requests achieved
31.258/31.346/31.363 decode tok/s; direct I/O achieved 31.475/31.436/31.453. First-request table
page eviction was verified; subsequent requests were warm. Per-request `/stats` deltas report
2.048/2.048/4.096 microseconds of PLE stall for buffered reads and 0/2.048/2.048 microseconds
for direct reads, over 4,464/4,544/4,544 rows. This establishes near-zero row-read stalls for
this workload, not an end-to-end speedup attributable to the queue or lookahead. Modes ran
sequentially. The stats endpoint initially omitted the Engine's n-gram counters; that omission
was fixed, the report contract test passed, and both I/O modes were remeasured.

The row cache remains opt-in: the qualification's first requests took 10–11% longer with a
4 GiB cache, while repeated requests had all-row hits without an established end-to-end gain.

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

### Without the n-gram table

The model was trained with its n-gram embedding, so the engine refuses a model whose table it
cannot find. `--no-ngram-table` starts it anyway, without the PLE injection (exactly what an
all-zero table gives), and warns at startup: this is a non-standard, experimental mode with no
practical use. On the Q2_0 release it nearly doubles WikiText-2 perplexity, 2.66 to 5.01 over the
fourteen windows of the comparison below; short factual answers survive, but nothing measured
improves.

`--kv-dtype` stores the KV cache of the 12 sparse-attention layers in any of the nine formats the
Qwen3.5 family uses. A position costs 24 KiB in `bf16` (the default), 12.4 KiB in `int8`, 12.1 KiB
in `fp8`, 9.6 KiB in `rk8v4`, 9.4 KiB in `k8v4`, 6.8 KiB in `nvfp4`, 6.6 KiB in `rk4v4` and
`rk4v4-e8`, and 5.1 KiB in `rk2v4-e8`; the indexer's pooled keys stay FP32 beside it, another
1.5 KiB a position in every format. The sparse attention decodes each row it reads, keys in the
rotated basis the cache stores them in. What the quantized formats cost in quality has not been
measured on this model yet.

`--max-concurrency N` (one to eight) runs that many requests at once, each on its own sequence with
its own KV and recurrent state, so every sequence costs device memory (the KV of `--max-context`
positions, 24 KiB a position in BF16 and less in a quantized `--kv-dtype`, plus 74 MiB of recurrent
state). Requests are admitted
in arrival order. Prompts prefill a step at a time, each step from the prompt with the fewest
tokens left, so a short or cached request is not held behind a long prompt for its whole prefill
(a prompt passed over eight times goes next, and a prompt with media, once begun, finishes
first). A step is one chunk, or with host experts on one GPU a span of up to eight chunks of a text
prompt (see [Execution](#execution)); while other requests decode, a span is two chunks at most,
since they wait for the whole step. After each step the requests that are decoding run
`--decode-rounds-per-prefill` rounds (by default the chunk size over 64, 16 at the default chunk)
before the next step, each round one batched pass whose experts read their weights once for the
whole batch, so the batch costs little more than one token while the experts dominate the step.

With the context cache on (the default), a sequence keeps its state when its request ends, and a
snapshot of its recurrent state where the prompt's last user turn closes (or at the prompt's end
when the template marks no turn), 74 MiB on the device: a later prompt that continues what the
sequence holds resumes from its live state, and one that repeats the prompt up to that point (the
next turn of a chat, which renders the previous answer without its reasoning) resumes from the
snapshot.
Either way only the new tokens are prefilled; the response's prompt summary reports how many were
reused. A request goes to the free sequence that holds the longest such prefix of its prompt.
`--no-prefix-reuse` (ninfer-serve) prefills every prompt from scratch.

What a sequence gives up for a new request (its turn-closure snapshot and the state it ended in,
each of 128 tokens or more) is kept as an image in pinned host memory: the paged KV and pooled keys
of its positions, its recurrent state, and the MTP block's, about 114 MB plus 26 KiB a position in
BF16 KV. A prompt that continues a kept prefix further than any free sequence resumes from its
image, copied back into a sequence. `--host-kv-mib` (or `--host-cache-mib`) is the budget, 8 GiB by
default; past it the least recently used images go to the disk tier with `--disk-kv-path DIR`
(`--disk-kv-gib`, 64 by default), one file each under a directory per artifact, KV format, MTP
draft width and n-gram enablement, or are dropped without one. A different verify width or disabling
the n-gram table can change sequence state, so those runs cannot restore another profile's images.
With `--disk-kv-restore` a
prompt also resumes from an image on
disk, which a later run finds there too.

Structured output (`--structured-output` for the server, `--json`/`--json-schema` for the CLI) works
as for the Qwen3.5 family: the grammar's token mask applies to every sampled token. So do
[token log probabilities](serving.md#token-log-probabilities): a request that asks gathers each
sampled token's top 20 from the same logits before sampling.

`--vision` loads the Vision tower of an artifact converted with it (0.9 GB of BF16 weights on the
first device, beside the token embedding) and takes images and video as the Qwen3.5 family does:
the frontend renders the media tokens, the tower encodes the prompt's media before its first
chunk, and their merged embeddings replace those tokens' embeddings. A media prompt rotates its
positions on the three RoPE axes the frontend computes (text positions on all three, then each
later token at its index plus the prompt's offset); it prefills in a pass of its own and is not
kept for reuse by the context cache. Flash-Next media requests use plain decoding even when
MTP is enabled. A fresh text-only request can use MTP again after the media request finishes.

### MTP speculative decoding

`--spec mtp --draft-tokens N` (1 to 15) decodes with the MTP block of an artifact converted with it
([Convert](#convert)); without the block the engine refuses to start. Text-only decoding requests then
runs rounds: the MTP block (one more sparse-attention layer with its own 512-expert MoE, its own
final mixer and the model's head) drafts N tokens greedily from the stack the model left at the
request's last position, the model verifies the last sampled token and the N drafts in one pass,
and the acceptance keeps the drafts the model's own sampling agrees with, then one token sampled
from the model at the first disagreement (or after the last draft). The acceptance samples as the
request does (greedy, temperature and top-k/top-p/min-p, presence and frequency penalties counting
the drafts kept before each position, the grammar's masks, logprobs from each position's own
distribution). Verification width and batch composition change floating-point reductions, so
greedy answers need not match plain decode byte for byte. The kernels are qualified against
independent mathematical oracles; exact repetition holds the execution configuration fixed.
Plain and MTP sampling also use different RNG purposes: the same seed does not promise the same
sampled answer across backends or execution widths. Acceptance is bounded by the
remaining output and thinking-token budgets, including the correction or bonus token.

`--draft-min-p P` optionally shortens the verification after the first draft whose absolute
probability over the public vocabulary is at or below P. It includes that draft, so every round
verifies at least one; P=0 disables the calculation and P=1 verifies one. A batch verifies its
largest selected prefix, with acceptance bounded separately for each request. This is distinct
from the sampling `--min-p` filter. The full captured draft chain still runs, and `drafted_tokens`
counts that work. Only verification shrinks; a throughput gain is not established. The floor is
part of the disk context-cache profile, and each verification width has its own CUDA Graph.
GPU qualification passed K=1/4, floors 0/0.3/1, fixed-mode repetition, cancellation/recovery and
three concurrent sequences. Floor 1 agrees with K=1. The release checks below establish no gain
from floor 0.3, so zero remains the default.

The verification leaves the sequence's state where it was: each Gated DeltaNet layer records its
transitions and the commit replays the kept ones into the state (the Qwen3.5 family's ReplaySSM
fold), the sparse-attention indexer's tail and the PLE convolution history advance over the kept
positions from what the verification recorded, and the n-gram context is hashed again over them.
The MTP block follows every token the model takes in (prompt chunks too), with a KV cache of its own
the size of one sparse-attention layer's (2 KiB a position in BF16), so a request can draft as soon
as its prompt is in, a prefix the context cache restores included. Up to eight requests run their
rounds together, one MTP pass for all of them per draft and one verification pass. A request
decodes without drafts when its prompt has media, near the end of its context, with one token
left in its output or thinking budget, and while a
`--post-thinking` request still reasons. N-gram copy proposals (`--ngram-draft-tokens`) are not
available for this model; `--mtp-attention-window` and `--lm-head-draft` are refused.

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
draft-chain graph and each width its own verification graph, captured at their second use.

## Execution

- The residual is a four-stream hyper-connection stack kept in FP32 between layers.
- The 36 Gated DeltaNet layers run the Qwen3.5 GDN kernels with a sigmoid output gate; the 12 sparse
  attention layers run the block indexer and attend only to the blocks it selects (plain dense
  attention below 2,051 positions).
- The PLE layer reads its 16 n-gram rows per token from the table's file (IQ4_NL rows decoded on
  the GPU).
- The 512-expert MoE groups each layer's (token, expert) pairs by expert on the GPU and runs one
  pass over each selected expert's rows for all of its tokens, through device tables of expert
  base pointers: an expert is read wherever the table points, in device memory, a cache slot or the
  pinned host block. Up to eight tokens run vector products; wider calls run ggml's integer
  tensor-core matrix kernel over the routed pairs from device memory. That kernel reads a 640-value
  down row in 256-value steps, so it decodes the bytes after a down matrix as the down's blocks:
  in a bank they are the next expert's down or zeros after the last one, and every cache or stream
  slot keeps zeros after its down, which a smaller down from another layer does not uncover (another
  format's bytes there can hold a non-finite scale, and NaN follows). With host experts, a wide call
  first copies the routed experts the cache does not hold into a device pool (one slot per expert on
  each GPU, 0.7 GB for Q2_0), so each expert crosses the bus once per chunk. On one GPU a prompt
  longer than a chunk runs in spans of up to eight chunks layer by layer: every chunk of a span
  passes a layer before any passes the next, so the layer's uncached experts cross the bus once per
  span (a 5,669-token prompt prefills at 1,334 instead of 678 tokens/s on an RTX 3090). Weighted
  expert outputs are summed in fixed point, so the result does not depend on the order experts
  finish in.
- Up to eight tokens of GGUF host experts run in two stages. A kernel splits the call's routes into
  the cached experts, which run at once, and the missing ones, which it posts to a host service
  through mapped memory, without a host synchronization; the service copies each missing expert
  into the slot of the layer's least recently used expert of the lowest frequency tier (or the CPU
  computes its share), the
  layer waits for it on the device, and the second stage runs it. The decode form of the expert
  kernels needs no sort or memset: one small kernel lists each slot's pairs, quantizes the tokens'
  inputs once and zeroes the sum; a block of the fused gate/up kernel takes 32 rows of one
  expert and quantizes its piece of the middle for the down kernel itself; one token's down runs
  row by row over all its experts, summing a row in registers. The decoding of Q2_0, the routed
  experts' format, is integer-throughput bound, so its activations come transposed and its codes
  need two operations a word. The shared expert runs on a second stream meanwhile. All of it is
  the separate kernels' arithmetic bit for bit.
- The expert cache counts the routes each forward pass took (decayed per token) and, between
  passes, copies the experts it needed most into its slots and points the tables at them: for
  every layer with mapped misses, and after a prompt chunk with staged ones. With disk experts the
  cache works per layer instead: once a layer has routed its tokens, the experts it lacks are read
  into the slots least recently used (never one the same call needs), and only then do the
  layer's experts run.
- A decode step (one token) replays a CUDA graph per pipeline stage, captured at a sequence's
  second decode step; the token's position reaches the sparse-attention kernels in device memory.
  Disk experts need the host between a layer's routing and its experts, so their steps stay eager.
  `--no-cuda-graph` decodes eagerly everywhere.
- With MTP, a verification of one request (at most eight tokens, so its experts take the vector
  products) replays graphs of its own the same way, and so does the draft chain of one request on
  one GPU, host experts included; several requests' rounds, the MTP catch-up and disk experts run
  eagerly. With staged host misses the MTP block runs only its cached experts, their weights
  renormalized per token: it only steers drafts, which verification checks, so the output is
  unchanged while drafting never waits for the bus.
- A pass whose layers may wait on the device for their missing experts ends synchronized: CUDA
  loads a kernel lazily at its first launch, a load may wait for the device to go idle, and a
  launch during such a wait would deadlock with the service it waits for.

## Measurements

### October 9 decode speed on one RTX 3090

One RTX 3090 (24 GB, PCIe 4.0 x16) with an i9-11900KF and 125 GB RAM, driver 580.82.09 and
CUDA 13.1 ran the Q2 model (GSQ-RCO Q2_0 with the IQ4_NL table) through `ninfer-serve` with host
experts, int8 KV, an 8,192-position context and one request at a time. Three prompts (a Python
LRU cache, the blue sky, the history of the Great Wall in Chinese) ran in that order twice, 256
greedy output tokens each, so a prompt's second run follows the other two. Each binary ran every
configuration once; the baseline is commit `a9155e0`.

| Mode | Baseline decode, tok/s (first / second run) | This build |
|---|---|---|
| plain, code / prose / Chinese | 69.2/78.4, 77.9/80.6, 74.1/79.4 | 87.3/92.3, 93.7/94.3, 93.4/94.3 |
| MTP 2 drafts | 68.6/70.0, 71.9/73.4, 57.2/62.4 | 118.8/118.6, 117.3/114.8, 95.8/101.2 |
| MTP 3 drafts | 66.7/66.9, 67.2/68.3, 53.0/57.5 | 121.2/122.6, 117.7/111.1, 90.0/94.6 |
| MTP 4 drafts | 63.1/62.2, 60.5/60.9, 44.9/48.2 | 111.1/105.9, 101.9/96.2, 74.6/78.5 |
| disk experts, code / prose (first run) | 64.5, 66.8 | 68.1, 71.4 |

A 5,669-token prompt prefilled at 1,253 instead of 727 tokens/s, and its 64 output tokens decoded
at 76.8 instead of 52.0 tokens/s, but this build computed long prompts wrongly with host experts
(see the prefetch fix below), so these two figures do not stand. Run back to back, a repeated prompt decodes at 98.5 (code) and
100.4 (prose) tokens/s plain and 137.6 and 123.7 with three drafts. Within each binary every mode
(plain, MTP 2-4, disk, a CPU share of 0.5, mapped misses) produced the same tokens; the two
binaries' outputs part after 40-70 tokens, most likely because the router's sums now split each
row over four warps and its wide calls run as a BF16 GEMM (not isolated).

The first request's time to first token is unchanged (462 and 464 ms), but a prompt's second run
started later: 298-387 ms instead of 174-320 ms. Staged misses admitted every missing expert into
the least recently used slot, so a decode moved the cache to its own experts and evicted the other
prompts' (46-58 MiB copied a token in second runs, against 3-14 MiB before). Letting decayed counts
veto an admission restored those starts but cost first runs 13-22% of their decode speed. An
admission now takes the least recently used slot of the lowest tier of decayed route counts (below
1.5, below 6, the rest), so the experts other prompts keep routing to outlast a passing decode. On
a second RTX 3090 (a rented host, driver 580.82.09, CUDA 12.8, the same protocol), against recency
alone, second runs started after 212-305 instead of 284-397 ms and first runs after 367-416 instead
of 406-416 ms; first-run decode changed by -4.4% to -0.6% (Chinese 86.2 instead of 90.2 tokens/s),
second-run decode by -1.6% to +1.0%, and the six requests took 19.44 instead of 19.53 s. Two tiers
(often or not, at 6 or 12 routes) and the thresholds 1/4 and 3/12 started second runs after 208-371
ms at a similar decode cost and took longer in total. Recency alone repeated its admissions and
copies exactly from run to run.

Measured improvements by kernel, from node traces of plain decode (each a share of a 10.1 ms
token): Q2_0's transposed decoding took the routed up kernel from 26.8 to 18.9 µs and the down
kernel from 25.7 to 21.5 µs per layer; Q3_K's split decoding cut its dense products from 1.06 to
0.98 ms a token. Shared memory staging of the weights (cp.async) and unconditional clamped loads in
the dense kernels were slower or neutral and were dropped.

### October 9 fix: prefetched expert tables

From commit `5118e069d` on, a host-expert prompt call of 256 tokens or more prefetched each layer's
uncached experts into the slot pool and copied the layer's expert tables from one of two pinned
buffers. A copy reads its buffer when it runs, and the host enqueues layers ahead of the device,
so a later layer's tables could replace a buffer's contents before its copy ran: such calls
multiplied some layers' tokens by other layers' experts. Each buffer is now rewritten only after
the copy that last read it has run. Over the first 40 KB of the WikiText stream of
`eval/corpora/perplexity-1m` (4,096-token windows, a 2,048-token stride, int8 KV) the Q2 model
with host experts read 3.995-4.047, varying from run to run, before the fix and 2.5390 after it,
as disk experts and the build before the prefetch (`a9155e0`, 2.5407) do. Decode steps and prompt
calls below 256 tokens never took that path, so the decode measurements above stand. On the
second RTX 3090 of the tier measurements, a 4,958-token prompt prefills at 1,234 tokens/s with the
fix and 758 with `a9155e0`; its 64 output tokens decode at 79.1 and 61.2 tokens/s.

### October 9 hyper-connection matrices in Q8_0

The hyper-connection projections were the largest read of a decoded token: 97 down/up pairs and
96 inject matrices, 1.27 GB in BF16. They are now ggml Q8_0 (32 inputs of a row share a binary16
scale), 0.67 GB. The GGUF recipe quantizes the releases' BF16 matrices with ggml's
`quantize_row_q8_0_ref` and keeps the MTP GGUF's own Q8_0 blocks, which it used to decode to
BF16. The three public artifacts were rewritten the same way with `scripts/pods/requantize_hc.py`,
every other object copied byte for byte: Q2_0 40,709,159,936 bytes, IQ3_S 57,902,944,256 and the
Coder build 32,685,469,696, each 618,159,104 smaller. Over the text matrices the represented
values differ from the BF16 ones by at most 0.0334, 4.79e-4 RMS.

The read's kernels for Q8_0 (one RTX 3090, CUDA 12.8, the fused write/read as a CUDA Graph, L2
evicted between samples, medians of 61):

| Tokens | 1 | 4 | 8 | 9 | 16 | 128 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16, µs | 23.6 | 28.7 | 44.0 | 43.0 | 45.1 | 98.3 | 384.0 |
| Q8_0, µs | 19.5 | 27.6 | 38.9 | 67.6 | 68.6 | 122.9 | 403.5 |

Up to eight tokens the kernels decode four weights at a time into exact FP32 values and keep the
activations FP32; four down rows share each CTA's activation loads. A first version that gave each
lane a whole 34-byte block made its activation loads 128 bytes apart and was 54-93% slower than
BF16 from four tokens on. Wider calls round the Q8_0 values to BF16 for the tensor-core products,
which costs two dequantizations of 13 µs a call: prompts and batches of nine or more tokens a step
pay 5-57% more for these reads.

Through `ninfer-serve` on the second RTX 3090 (the rented host of the tier measurements: driver
580.82.09, CUDA 12.8) with the October 9 protocol, the Q2 model before and after the rewrite, each
configuration run twice: plain decode went from 81.3-93.6 to 84.5-97.7 tokens/s, 3.6-6.0% faster
for every prompt and run (a configuration's two runs within 0.6% of each other), and decode with
three MTP drafts from 84.5-114.4 to 83.0-120.4 tokens/s, 0.4-10.0% faster per prompt on average
(its runs within 5%; one run of the first Chinese answer was 1.8% slower). The weights on the GPU
went from 3.38 to 2.83 GiB and the device expert cache grew from 17,340 to 17,786 MiB, but its hit
rate moved by at most half a point, so the gain is the reads'. With the prefetch fix, a 4,958-token
prompt prefilled at 1,268 instead of 1,234 tokens/s and its 64 output tokens decoded at 81.9
instead of 79.1 tokens/s (one run each). Both answer coherently; their outputs part within the
first tokens, as two quantizations do.

Perplexity over the quick corpora (4,096-token windows advancing by 2,048, int8 KV, host experts,
with the prefetch fix), the BF16 matrices against Q8_0:

| Model, corpus | BF16 | Q8_0 | Change |
|---|---:|---:|---:|
| Q2_0, `perplexity-1m` | 3.93783 | 3.93842 | +0.015% |
| Q2_0, `perplexity-heldout-2026-09` | 4.69573 | 4.69673 | +0.021% |

Domain by domain the Q2_0 changes run from -0.11% (this repository's code) to +0.36% (new English
Wikipedia articles). The other two builds, whose text matrices are the same bytes, were checked on
two 40 KB slices only, the start of the WikiText stream above and of the held-out English Wikipedia
stream (9,234 and 9,477 tokens):

| Model | WikiText slice | Change | Held-out Wikipedia slice | Change |
|---|---:|---:|---:|---:|
| Q2_0 | 2.5390 → 2.5440 | +0.20% | 7.3517 → 7.4048 | +0.72% |
| IQ3_S | 1.9767 → 1.9724 | -0.22% | 6.5552 → 6.5604 | +0.08% |
| Coder | 4.2254 → 4.1985 | -0.64% | 8.9087 → 8.9281 | +0.22% |

Over so few tokens one quantization moves perplexity by up to 0.7% either way; the corpora above
are the measure.
AIME and GPQA were not rerun: one AIME problem is 3.3 points and one GPQA-Diamond run's standard
error about 2.5, far more than these differences could move, and the pair takes most of a day on
one RTX 3090.

### October 8 public MTP attachments

The three existing public model repositories now include the shared-Q8_0 MTP block.
Each update adds 28 physical objects (2,791,415,296 bytes) and 1,567 logical bindings.
Readback verified every previous text/Vision object, tokenizer resource and table descriptor.
The Coder model keeps its 256-expert text banks and the MTP block's independent 512-expert bank.
Each repository retains the original conversion report, attachment report, qualification and checksums.

All three models passed host-to-GPU and disk-to-GPU requests on one RTX 3090, CUDA 13.1,
int8 KV, 4,096 context capacity, 128-token prefill chunks and an 8 GiB device expert cache.
The two prompts ask for a Python function to merge sorted lists and an explanation of the blue sky.
Each mode ran three greedy repetitions of 64 output tokens, with prefix reuse disabled and EOS ignored.
All fixed modes repeated exactly and MTP accepted drafts. The four Q2 modes also produced identical tokens.
Vision remained enabled: a red-image request used the existing plain fallback, then a fresh text request resumed MTP.
The initial media check failed because it incorrectly required drafting for media; source inspection corrected that expectation.

| Model / expert residency / drafts / minimum probability | Prompt 1 median (range), tok/s | Prompt 2 median (range), tok/s |
|---|---:|---:|
| Q2 / host / 4 / 0 | 38.43 (25.90–41.98) | 32.00 (26.02–35.37) |
| Q2 / disk / 4 / 0 | 13.64 (10.61–13.82) | 11.72 (9.88–11.88) |
| Q2 / host / plain / 0 | 50.68 (31.28–55.58) | 45.30 (35.36–49.92) |
| Q2 / host / 4 / 0.3 | 38.20 (26.59–42.29) | 31.81 (26.35–35.37) |
| IQ3 / host / 4 / 0 | 22.84 (16.28–25.07) | 13.96 (11.19–16.13) |
| IQ3 / disk / 4 / 0 | 10.56 (6.75–10.59) | 8.47 (6.17–8.53) |
| Coder / host / 4 / 0 | 27.01 (19.27–28.95) | 18.66 (14.69–20.90) |
| Coder / disk / 4 / 0 | 12.06 (11.79–12.31) | 9.54 (9.46–9.70) |

The Q2 MTP medians are 24.2% and 29.4% below plain decoding in this workload.
Artifact preparation and transfers overlapped these checks, and the OS page cache was uncontrolled.
These measurements include first requests and do not isolate an MTP speed effect or establish cold-disk throughput.
The 0.3 confidence floor changed neither tokens nor draft/acceptance counts; zero remains the default.
The campaign establishes working MTP attachments, fixed-mode repetition and Vision preservation, without a speedup claim.
Physical low-RAM hosts, an RTX 3080 with 20 GB, and Windows were not qualified.
`scripts/pods/mtp_qualify.py` retains the requests, options and all samples.

### Broad n-gram profile

The optional broad profile contains 26,223,284 ranked rows from 3,711,509 training tokens.
The held-out corpus contains 1,103,918 tokens in 445 observed domain/language groups.
It samples four Common Corpus shards at `307910e4c5d040d6f318e6edf2a2b97849155771` and one
github-code-clean shard at `c48d40f9e70f0196f8236901ee35807f7d6c44c0`.
Equal per-group token ceilings, document/repository separation and capped source contributions avoid selecting preferred languages.
The sources cover prose, web, press, science and code. Labels are automatic, sparse groups remain visible,
and exact snippet deduplication does not eliminate all near-duplicates. This finite sample does not represent every language or user traffic.

| RAM budget, MiB | Held-out row hits |
|---:|---:|
| 256 | 33.79% |
| 512 | 38.49% |
| 1024 | 42.92% |
| 2048 | 48.48% |
| 4096 | 49.64% |
| 8192–32768 | 49.64% |

Across 239 groups with at least 2,048 held-out tokens, minimum/median/maximum hits are
6.04%/32.85%/90.64% at 256 MiB and 20.54%/49.38%/95.47% at 4 GiB.
Japanese books have the lowest coverage at both budgets. Row hits measure reuse, not model quality or inference speed.
NumPy and the native C++ profiler produced byte-identical 104,893,176-byte profiles.
Embedding the profile increases the table container by 0.364%, while preserving every IQ4 row byte and the row digest.
The existing public table repository includes that resource and the standalone `broad-v1.hot` file.
It also publishes the complete per-group evaluation, corpus provenance, attachment proof and Engine report.
The private transfer bucket retains the corpus archive and a profile copy, both verified by SHA-256 readback.

The Engine passed disk, embedded-profile and explicit-profile configurations on the Q2 MTP model
and RTX 3090 above. Two prompts, two repeats and 32 output tokens matched exactly across all three modes.
Both profile modes served resident rows from a 256 MiB RAM budget. Transfer overlap and uncontrolled page cache
prevent isolated performance attribution. Earlier first-request regressions below remain applicable evidence; caching stays opt-in.
`scripts/pods/ngram_corpus_build.py`, `ngram_profile_broad.py` and `ngram_profile_engine.py` retain the corpus and check procedures.

### October 8 qualification and CPU prototype

On two RTX 3090s with CUDA 13.1, driver 595.91.07, Q2_0 experts, the shared-Q8_0 MTP
component and int8-group64 KV, Engine startup previously took 248.6 seconds. Removing the
directory parser's quadratic JSON callback scan reduced observed subsequent startups to
18.5–21.1 seconds. This compares one old startup with the qualification runs; it is not a decode
throughput measurement. Duplicate JSON members remain rejected.

The fixed-seed MTP qualification also found an ordering bug across the two GPUs: after an
asynchronous MTP embedding copy, the next prompt chunk could overwrite GPU 0's source buffer
before GPU 1 read it. The first seven MTP KV rows changed between otherwise identical requests,
occasionally changing the sampled answer at generated token 30. CUDA Graphs were not the cause:
the failure occurred with them disabled too. A source-stream wait now protects that buffer until
the copy completes. With a temporary trace and alternating 200 ms peer-copy delays, all 24
repeats in each graph mode agree byte for byte in the valid KV rows, MTP checkpoints, full head
logits and answer. The corresponding run before the fix had varying KV/logits despite matching
answers; the delay by itself was not a deterministic answer-failure trigger. The trace has been
removed from the product; the Linux Engine regression retains the delay fixture and passes all
48 repeats without tracing. Public Engine, output/thinking budgets, logprobs and cache checks
pass; the greedy suite reports six plain/MTP and seven batch-composition differences, as allowed
by the documented fixed-mode contract. The wait's
throughput cost has not been separately measured.

The experimental `bench/ops/q2_cpu_probe.cpp` evaluates the first 80 stored experts from layer 12,
with BF16 sinusoidal inputs, uniform top-10 weights, H=2560 and I=640. Its independent FP64 oracle
decodes Q2_0 scales and codes and evaluates the original inputs through SwiGLU and the down
projection. Internal A8 quantization and BF16 intermediates are not oracle boundaries. Across
T=1/2/5/8, repeated or disjoint expert sets, and 1/8/26 threads, relative L2 is 0.00911–0.00961 and
maximum error over oracle peak is 0.00791–0.01211, within the production Op's 0.04/0.16 criteria.
An earlier oracle reused the quantized intermediates: its approximately 9e-8 error measured that
arithmetic alone and does not establish full mathematical accuracy.

The host was a Xeon Platinum 8167M; 26 physical cores on NUMA node 0, AVX2/F16C, OpenMP static
work sharing. Each cell is the median of nine calls, in milliseconds, shown as warm / after a
256 MiB cache flush. These calls exclude the shared expert, GPU work, transfers and dispatch.

| tokens | same 10 experts | disjoint 10 experts per token |
|---|---:|---:|
| 1 | 1.677 / 1.398 | 1.417 / 1.402 |
| 2 | 2.215 / 2.311 | 2.081 / 2.423 |
| 5 | 3.709 / 5.083 | 4.019 / 5.662 |
| 8 | 4.273 / 9.210 | 4.410 / 9.223 |

Timing varied: T=8 disjoint/warm spans 3.882–7.570 ms; T=1 is sometimes slower warm than after
the flush. The cause has not been isolated. Eight threads take 12.204 / 22.496 ms for T=8 disjoint;
one thread takes 95.039 / 95.113 ms. Copying these same packed expert banks from pinned memory to
GPU 0 costs 1.19 ms for ten experts (13.824 MB), 2.37 ms for twenty, 5.93 ms for fifty and 9.48 ms
for eighty; these are all-miss H2D measurements without GPU compute.

`bench/ops/q2_gpu_probe.cpp` runs the same eight represented input/routing cases against those
FP64 outputs. All pass the same error criteria and repeat bit for bit. GPU 0's resident medians
are 0.094/0.095 ms at T=1 (reused/disjoint), 0.142/0.141 at T=2, 0.285/0.283 at T=5, and
0.435/0.431 at T=8. Unlike the CPU probe, the public GPU Op also computes a shared expert with
zero contribution. Routing, transfers and Engine dispatch are excluded. A preceding 32 MiB
device memset gives medians approximately 6–8 microseconds lower; cache eviction was not
verified and the cause of that timing difference remains unresolved.

Adding the separately measured all-miss H2D cost to resident compute suggests that CPU compute
could compete at T=2/5/8 when every disjoint expert misses the device cache. This sum is not a
measured hybrid execution. Resident GPU compute is substantially cheaper; CPU work should
therefore target cache misses. CPU execution has not been integrated into the Engine and no
end-to-end hybrid gain has been established.

The earlier n-gram qualification, before the disk row cache and advisory lookahead, runs the
same 80-token greedy coding request with int8-group64 KV,
context 4096 and both GPUs. All ten answers agree byte for byte. Each cold run discarded only
the artifact table's page-cache pages and verified residency with `mincore`; one of 7,031,284
pages remained resident before the first cold run. Each cell below reports cold / subsequent
warm, one run of each rather than a timing distribution. Reader stall totals include startup
and warmup, not just generation.

| n-gram mode | startup, seconds | decode, tok/s | total PLE reader stall, ms |
|---|---:|---:|---:|
| buffered | 18.7 / 18.5 | 92.4 / 91.8 | 2.64 / 8.91 |
| direct | 18.5 / 18.6 | 91.8 / 92.3 | 5.15 / 3.87 |
| mmap | 18.5 / 18.5 | 92.3 / 92.1 | 4.51 / 3.90 |
| ram | 41.1 / 29.1 | 94.2 / 94.2 | 0.11 / 0.11 |
| ram-hot, 256 MiB | 19.3 / 20.1 | 92.2 / 92.4 | 4.22 / 3.06 |

Full RAM residency consumes another 28.8 GB and adds 22.4 seconds to cold startup relative to
buffered reads here. The profile trained on this guide and the Engine architecture contains
216,052 rows from 22,034 tokens. On the held-out README and serving guide (56,462 tokens,
903,392 row accesses), it covers 25.5% of accesses; increasing the budget from 256 MiB to
32 GiB cannot improve this profile because its entire row set already fits. The coding
generation's measured hot-row hit rate is only 0.7%. Ideal page-cache hit-rate upper bounds
from the held-out access trace are 45.9/58.3/72.8/100% at 256/512/1024/2048 MiB; these are
trace-derived bounds, not measured OS hit rates. These results do not establish a general
advantage for `ram-hot` or close the planned asynchronous I/O and draft-lookahead work.

The October 8 disk-cache qualification uses the native Q2 model, its IQ4_NL table companion,
two RTX 3090s, CUDA 13.1 and int8 KV. Exact reader checks cover buffered, direct and mapped I/O,
single-file and split rows, bounded eviction, duplicate rows and failed-batch admission. The
Engine test uses 700 raw input tokens in 512+188 chunks and 24 output tokens, with context 4096.
All 12 token/content/finish comparisons match within each fixed MTP setting. Every repeated
cached request serves all its rows from the cache.

| MTP K | read mode / cache | first request, s | repeated request, s |
|---|---|---:|---:|
| 0 | buffered / off | 0.722970 | 0.690307 |
| 0 | buffered / 4 GiB | 0.792032 | 0.688651 |
| 0 | direct / 4 GiB | 0.837846 | 0.689248 |
| 4 | buffered / off | 0.855488 | 0.833481 |
| 4 | buffered / 4 GiB | 0.953051 | 0.830672 |
| 4 | direct / 4 GiB | 0.986878 | 0.831778 |

These are one ordered sample per cell, without a controlled cold-cache distribution. With the
same buffered I/O, enabling the cache costs 9.55%/11.40% on the first request. The repeat's summed
reader latency falls from 13.09/12.94 ms to 2.49/2.35 ms, but generation improves by only
0.24%/0.34%; this does not establish a repeatable throughput gain. Direct-I/O first requests are
slower still, but change both I/O and cache relative to the disabled baseline. First-request
reader counters include startup/warmup; the repeated-request deltas and all request times in the
table exclude them. The cache budget includes its tags and CLOCK state; per-mode peak RSS was
not measured. Windows cache/hint execution has not been tested.

A separate CLI comparison retains the pre-cache executable and compares it with the new
executable at an explicit 4 GiB cache budget. Both process the same 632-token prompt and 48
greedy output tokens with MTP K=4. The table's page-cache residency was checked after eviction
before each process; normal startup and warmup then preceded generation. Answers match exactly.

| metric | pre-cache executable | new executable, 4 GiB cache |
|---|---:|---:|
| prefill, s | 0.448 | 0.456 |
| decode, s | 0.343 | 0.352 |
| generation, s | 0.795 | 0.810 |
| n-gram read per pass, microseconds | 20,468.2 | 20,890.8 |
| total PLE reader stall, ms | 1.24 | 15.10 |
| rows served from RAM | 0.0% | 60.3% |

This pair is also one sample per mode. Reader statistics include startup/warmup. Generation is
1.89% slower with the cache; no end-to-end gain is claimed. Fatbinary compression differs between
the executables, so startup comparisons would not isolate cache cost. The cache stays opt-in,
with a zero default for `disk`; `ram-hot` retains its 4096 MiB default. This measurement predates
the OS queue and per-step draft hints. The final I/O measurement above qualifies their row-read
stalls on Linux; Windows execution is excluded. The broad profile above subsequently passed
embedded and explicit resource qualification through the full-model Engine.

### Earlier generation measurements

Q2_0 release, greedy decoding, CUDA 12.8, 2026-10-05. Decode is measured over the 78 tokens of a
short answer (prompt of 20 tokens) and over the first tokens after a 4,463-token prompt; prefill is
that prompt in 512-token chunks.

| Hardware and placement | Device memory | Host memory | Decode, short answer | Decode after 4,463 tokens | Prefill |
|---|---:|---:|---:|---:|---:|
| 2× RTX 3090 Ti (PCIe, no P2P), experts on the GPUs (`--devices 0,1`) | 18.4 + 19.4 GB | 0.7 GB | 90.2 tok/s | 107 tok/s | 1,504 tok/s |
| RTX 3090, experts in pinned host memory, 17.7 GB expert cache | 22.6 GB | 34 GB pinned | 48.9 tok/s | 42.1 tok/s | 842 tok/s |
| RTX 3090, experts on disk, artifact in the page cache | 22.6 GB | 0.9 GB + page cache | 47.0 tok/s | 39.1 tok/s | 630 tok/s |
| RTX 3090, experts on disk, page cache dropped every second (NVMe) | 22.6 GB | 0.9 GB | 17.2 tok/s | 11.3 tok/s | 195 tok/s |

Host memory is the process's peak resident set (the pinned bank for host experts). Decode after
the long prompt covers its first five tokens only. CUDA graphs add 11% to the short-answer decode on
the two GPUs (81.2 tok/s eager) and 7% with host experts (44.3 tok/s); disk experts decode eagerly.
Prefill touches nearly every expert of every layer in each chunk, and with host or disk experts
each of them crosses the bus or comes off the disk once per chunk, so larger `--prefill-chunk`
values serve more tokens per copy: with host experts on an RTX 3090 Ti the long prompt takes 6.05 s
in 512-token chunks and 4.58 s in 2,048 (10.97 s before the experts went through device slots).

The IQ3_S release on one RTX 3090 (310 W power limit, PCIe 4.0 x16) in a host with 62 GB of RAM,
23 cores of an AMD EPYC 7663 and an NVMe drive, 2026-10-05, with the Q2_0 release in the same
sitting; the artifacts are single files with their n-gram tables, and decode counts the 86 tokens of
the short answer:

| Release and placement | Device memory | Host memory | Decode, short answer | Decode after 4,463 tokens | Prefill |
|---|---:|---:|---:|---:|---:|
| IQ3_S, experts in pinned host memory, 16.2 GB expert cache | 22.1 GiB | 50.3 GB pinned | 34.4 tok/s | 30.2 tok/s | 533 tok/s |
| IQ3_S, experts on disk, the page cache holding what of the 83.6 GB file fits | 22.1 GiB | page cache | 19.0 tok/s | 21.6 tok/s | 209 tok/s |
| IQ3_S, experts on disk, the file's pages evicted every second | 22.1 GiB | — | 10.7 tok/s | 5.4 tok/s | 47 tok/s |
| Q2_0, experts in pinned host memory | 22.1 GiB | 34.0 GB pinned | 50.2 tok/s | 43.7 tok/s | 847 tok/s |

The Q2_0 release (with the separate table artifact) on RTX 4090s at 450 W (PCIe 4.0 x16, no peer
access) in a cloud VM with two EPYC 7543 sockets of 60 vCPUs each, both GPUs on NUMA node 0, 694 GB
of RAM and its container disk, 2026-10-06. The single-GPU rows ran pinned to node 0's CPUs
(`taskset -c 0-59`; the container refuses a memory policy, so first touch places the pages there);
the ranges are two runs:

| Hardware and placement | Device memory | Host memory | Decode, short answer | Decode after 4,463 tokens | Prefill |
|---|---:|---:|---:|---:|---:|
| 2× RTX 4090, experts on the GPUs (`--devices 0,1`) | 18.5 + 19.4 GiB | 0.8 GiB | 92.6-100.7 tok/s | 116-120 tok/s | 3,482-3,734 tok/s |
| RTX 4090, experts in pinned host memory, 15.9 GB expert cache | 20.7 GiB | 34.0 GB pinned | 52.6-52.7 tok/s | 44.3-44.6 tok/s | 1,137-1,139 tok/s |
| RTX 4090, experts on disk, artifact in the page cache | 20.7 GiB | 1.0 GiB + page cache | 42.0 tok/s | 45.6 tok/s | 723 tok/s |
| RTX 4090, experts on disk, the artifact's and table's pages evicted every second | 20.5 GiB | 1.0 GiB | 17.7 tok/s | 12.2 tok/s | 160 tok/s |

Left unpinned on that VM, the host-expert row decodes at 39.5-40.1 tok/s and prefills at 775-827
tok/s, the page-cache row at 31.9-33.0 tok/s and 571-674 tok/s, and the evicted row at 15.7 tok/s
and 144 tok/s: the copies out of host memory then cross the socket link. CUDA graphs add 1% and 10%
to the short-answer decode on the two GPUs in the two runs (91.5-91.7 tok/s eager) and 5 to 6% with
pinned host experts (49.9-50.1 tok/s eager). Host memory is the process's peak resident set, device
memory the most `nvidia-smi` showed in use.

Both releases on Blackwell, 2026-10-06, a `120a` build with CUDA 13.1, the table as the separate
artifact, the single-GPU rows pinned to the CPUs of the GPU's NUMA node, two runs where a range is
given:

- RTX PRO 6000 Blackwell Workstation Edition: 600 W, a PCIe 4.0 x16 host link, 64 vCPUs and
  1.1 TB of RAM.
- RTX 5090s: boards with a 600 W default limit, PCIe 5.0 x16, 512 threads and 1 TB of RAM.

The short answer is 78 tokens for Q2_0, and 87 (PRO 6000) or 81 (RTX 5090) for IQ3_S.

| Hardware and placement | Release | Device memory | Host memory | Decode, short answer | Decode after 4,463 tokens | Prefill |
|---|---|---:|---:|---:|---:|---:|
| RTX PRO 6000, experts on the GPU | Q2_0 | 37.7 GiB | 0.8 GiB | 136.0-139.7 tok/s | 160.7-162.6 tok/s | 2,925-2,952 tok/s |
| RTX PRO 6000, experts on the GPU | IQ3_S | 53.7 GiB | 1.0 GiB | 123.8-125.9 tok/s | 146.8-146.9 tok/s | 2,540-2,541 tok/s |
| RTX PRO 6000, experts in pinned host memory | Q2_0 | 38.5 GiB | 34.0 GB pinned | 54.8-55.0 tok/s | 67.7-70.0 tok/s | 1,296-1,297 tok/s |
| RTX PRO 6000, experts in pinned host memory | IQ3_S | 55.0 GiB | 50.3 GB pinned | 29.3 tok/s | 31.2 tok/s | 803 tok/s |
| RTX PRO 6000, experts on disk, the files in the page cache | Q2_0 | 93.6 GiB | 1.0 GiB + page cache | 75.9 tok/s | 115.8 tok/s | 2,024 tok/s |
| RTX PRO 6000, experts on disk, the files in the page cache | IQ3_S | 93.5 GiB | 1.2 GiB + page cache | 65.6 tok/s | 106.0 tok/s | 1,661 tok/s |
| RTX PRO 6000, experts on disk, the files' pages evicted every second | Q2_0 | 93.5 GiB | 1.0 GiB | 35.1 tok/s | 87.1 tok/s | 1,000 tok/s |
| RTX PRO 6000, experts on disk, the files' pages evicted every second | IQ3_S | 93.5 GiB | 1.2 GiB | 28.7 tok/s | 78.6 tok/s | 773 tok/s |
| 2× RTX 5090, experts on the GPUs (`--devices 0,1`) | Q2_0 | 18.8 + 19.6 GiB | 0.9 GiB | 136.1-136.3 tok/s | 157.0-157.2 tok/s | 3,807-3,811 tok/s |
| 2× RTX 5090, experts on the GPUs (`--devices 0,1`) | IQ3_S | 26.5 + 27.8 GiB | 1.0 GiB | 121.6-121.9 tok/s | 141.8-142.0 tok/s | 3,310-3,341 tok/s |
| RTX 5090, experts in pinned host memory | Q2_0 | 30.0 GiB | 34.0 GB pinned | 74.1-74.2 tok/s | 93.9-94.2 tok/s | 1,676-1,678 tok/s |
| RTX 5090, experts in pinned host memory | IQ3_S | 30.0 GiB | 50.3 GB pinned | 40.7 tok/s | 41.0 tok/s | 1,100 tok/s |
| RTX 5090, experts on disk, the files in the page cache | Q2_0 | 29.9 GiB | 1.0 GiB + page cache | 67.5 tok/s | 76.7 tok/s | 1,739 tok/s |
| RTX 5090, experts on disk, the files in the page cache | IQ3_S | 29.9 GiB | 1.1 GiB + page cache | 55.3 tok/s | 44.3 tok/s | 439 tok/s |
| RTX 5090, experts on disk, the files' pages evicted every second | Q2_0 | 29.9 GiB | 1.0 GiB | 25.3 tok/s | 28.7 tok/s | 694 tok/s |
| RTX 5090, experts on disk, the files' pages evicted every second | IQ3_S | 29.9 GiB | 1.1 GiB | 18.8 tok/s | 14.9 tok/s | 171 tok/s |

**Native-Q2 pipeline comparison, October 8.** The same executable and artifact ran on RTX A6000
(driver 595.91.07) and two RTX 3090s (driver 590.48.01), with int8 KV, 512-token prefill chunks,
plain decoding and no prefix reuse. Each context has three requests and 64 generated tokens;
model stops were disabled to keep the decode workload fixed. No warmup sample was discarded.

| Input tokens | A6000 request seconds, all samples | 2×3090 request seconds, all samples | Median change | Prefill tok/s, A6000 / 2×3090 | Decode tok/s, A6000 / 2×3090 |
|---:|---|---|---:|---:|---:|
| 4,032 | 3.974 / 3.976 / 3.992 | 2.591 / 2.583 / 2.589 | -34.88% | 1,267.9 / 2,167.7 | 79.18 / 87.19 |
| 33,024 | 28.396 / 28.715 / 28.904 | 15.400 / 15.469 / 15.575 | -46.13% | 1,184.0 / 2,242.1 | 76.60 / 85.22 |
| 131,008 | 128.705 / 128.904 / 128.959 | 66.493 / 66.848 / 67.172 | -48.14% | 1,023.4 / 1,983.7 | 70.88 / 78.52 |

Phase rates use the median phase time; decode covers the 63 intervals after the first token.
These compare the two host configurations, including their hardware and selected kernel routes;
they do not isolate a gain from splitting the model. A test-package export also overlapped part
of the A6000 run. Both configurations repeat exactly. Cross-host identity passes at 128K and
fails at 4K/32K: all three samples differ in 53 of 64 tokens, beginning at output index 9, after
the first model EOS at index 4. The prefixes through that EOS are identical. The full fixed-length
output difference remains visible; its cause is not established by these timings or by prefix
agreement. A completed diagnostic applied the A6000 route choices to the two RTX 3090s, using
Op defaults for additional 3090 profile keys. Its outputs are identical to the original split
run at every context and repeat; this route change did not remove the cross-host difference.
It is not a calibration of those routes for the 3090. On October 9, the user accepted differences
after EOS and closed M7. All nine cross-host comparisons match through the first EOS, and all
fixed-mode repeats remain exact. The full-output differences and unresolved cause remain recorded.
The comparison tool's `--eos-token 248046` option applies this accepted boundary without hiding raw differences.

**MTP.** The Q2_0 release with Unsloth's `shared-Q8_0` MTP block, converted into one file with its
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

Before verification and drafting replayed CUDA graphs the same runs decoded at 89.5-90.7 tok/s plain
(128.3-128.9, 146.2-147.9, 165.9-166.5 and 167.3-169.0 with one to four drafts). Coding answers are what MTP drafts best; a prose answer (80 tokens on the water cycle) keeps 38.6% of three drafts, 2.16 tokens a round, and decodes at 91.6 tok/s, no faster than without drafts.
With host experts a verification's four columns route to up to four times as many experts, each
missing one crossing the bus, so three drafts gain about 12% (three runs each, pinned to the GPU's NUMA node). The first prompt after startup
prefills in 102-121 ms (379-406 ms before the startup warm-up, which loads the
kernels the first request would otherwise load).

**Device expert cache.** With host or disk experts the device expert cache takes the memory the GPU
has free, and the generate test answers its prompts right after the load. The rows therefore
measure a cache that is still filling: 94 GB of it on the PRO 6000, where nearly every expert ends
up resident, and 30 GB on the RTX 5090.

**Host link.** Host and disk experts cross the host link: PCIe 5.0 x16 on the RTX 5090 host,
PCIe 4.0 x16 on the PRO 6000 host. That is the likeliest reason the RTX 5090's host-expert row beats
the PRO 6000's.

**CUDA graphs.** Graphs add 27% to the short-answer decode on the two RTX 5090s (107.0 tok/s eager)
and 23 to 26% on the PRO 6000 (111.0-111.2 tok/s eager).

On the current code every GSQ-RCO release, converted into one file with its table, answers the
generate test's prompts (the facts and the 4,463-token needle) on that card with disk experts and
with host experts, with CUDA graphs and without: Q2_0, IQ2_XS (39.2 GB, 35.5 GB pinned), IQ3_XXS
(47.0 GB, 42.9 GB pinned), IQ3_S, and the Coder build's IQ1_M (29.6 GB, 256 experts, 25.1 GB pinned).
Q2_0 also ran with its experts on two GPUs before the table moved into the artifact. With disk
experts the process peaked at 1.05 to 1.19 GB of RAM for IQ2_XS, IQ3_XXS and IQ1_M, and at 1.10 GB
for IQ3_S (an L40S host).

The published layout, each model without its table and with its Vision tower plus the shared table
artifact, was checked on one NVIDIA L40S (an sm_86 build): the generate test with device, host and
disk experts for Q2_0, host and disk for IQ3_S and the Coder build, three sequences decoded as one
batch against each decoded alone, a restored snapshot, an image question per release, and the
server's concurrency, prefix reuse and JSON Schema checks.

Where a decode step goes, from an Nsight Systems trace of the generate test's graph-replayed steps
(Q2_0, every expert on that L40S): about 1,780 kernels and copies in 10.6 ms, 0.5 ms of it idle
between them. A token reads about 4 GB of weights, and the BF16 hyper-connection projections are
the largest share: 97 down/up pairs of 6.5 MB each, 1.27 GB, more than the ten routed experts of
every layer (0.66 GB). Their GEMVs take 2.2 ms, the routed experts 1.75 ms, the GGUF projections
of the Gated DeltaNet and attention layers with the head about 2.9 ms, and the router, shared
experts, activation quantization, recurrent and sparse-attention kernels the rest. With host
experts on a 24 GB card the expert kernels read what the cache lacks across the bus and take most
of the step instead.

llama.cpp runs the same GGUFs with the experts on the CPU (`--n-cpu-moe 48`). On the second
machine above (23 threads) llama-bench gives 29.6 tok/s decode (tg128) and 312 tok/s prefill
(pp512) for IQ3_S, and 12.7 and 368 tok/s for Q2_0; the first RTX 3090's machine (32 threads) gave
11.2 and 267 tok/s for Q2_0. Its speed follows the host CPU, and its Q2_0 CPU path is the slower one.

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
