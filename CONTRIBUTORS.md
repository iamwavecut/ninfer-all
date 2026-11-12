# Contributors

NInfer-all is maintained by [iamwavecut](https://github.com/iamwavecut). It grew from
[Neroued/ninfer](https://github.com/Neroued/ninfer) through the NInfer-3090 line that
[Don-Chad](https://github.com/Don-Chad/ninfer-3090) started and
[ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090) continued (v0.12.0 and the
multi-GPU pipeline stages, most of both by [Warlax](https://github.com/WarlaxZ)), and it carries
work from the forks and pull requests listed below. Ported commits keep their authors and name
their source commit, so `git log` is the complete record; this page keeps the credit visible.

## Pull requests merged into the NInfer-3090 line

| Contributor | Contribution | Pull request |
| --- | --- | --- |
| [airtonix](https://github.com/airtonix) | Linux and Docker build and release support | [#1](https://github.com/Don-Chad/ninfer-3090/pull/1) |
| [justinlime](https://github.com/justinlime) | NixOS build support | [#5](https://github.com/Don-Chad/ninfer-3090/pull/5) |
| [sry9681](https://github.com/sry9681) | Device-wide GPU-memory startup fix | [#6](https://github.com/Don-Chad/ninfer-3090/pull/6) |
| [ColeWheatley](https://github.com/ColeWheatley) | SM86 runtime/GDN fixes, ECC diagnostics, and GeForce-safe Docker support | [#7](https://github.com/Don-Chad/ninfer-3090/pull/7) |
| [iamwavecut](https://github.com/iamwavecut) | swscale destination-alignment JPEG safety fix | [#11](https://github.com/Don-Chad/ninfer-3090/pull/11) |
| [nasedkinpv](https://github.com/nasedkinpv) | Tool-call parser crash fix | [#12](https://github.com/Don-Chad/ninfer-3090/pull/12) |
| [wmehanna](https://github.com/wmehanna) | In-place system-turn rendering for Claude Code prefix reuse | [#13](https://github.com/Don-Chad/ninfer-3090/pull/13) |
| [mgscreativa](https://github.com/mgscreativa) | Thinking-budget boundary fix; premature KV-loan race diagnosis and gate fix behind a vision `std::bad_alloc`; the lost wakeup that hung Flash-Next's long prefills with host experts | [#156](https://github.com/ashalliants/ninfer-3090/pull/156), [#138](https://github.com/ashalliants/ninfer-3090/pull/138) (kept in [#145](https://github.com/ashalliants/ninfer-3090/pull/145)), [#243](https://github.com/iamwavecut/ninfer-all/pull/243) |
| [liqlos](https://github.com/liqlos) (Aleksei Bukhalov) | GGUF matrix products that store BF16 after the stream-K reduction, Q4_K and Q6_K vector loads shared across lanes, and Q4_K matrix tiles staged through `cp.async` for dense layers and routed experts | [#127](https://github.com/iamwavecut/ninfer-all/pull/127), [#134](https://github.com/iamwavecut/ninfer-all/pull/134), [#257](https://github.com/iamwavecut/ninfer-all/pull/257) |
| [eruca](https://github.com/eruca) | A second window of hits for a document asked about every round, and the Host-tier workloads that showed fewer conversations cached with 4 GiB than with 2 GiB | [#256](https://github.com/iamwavecut/ninfer-all/pull/256) |
| [dzaurov](https://github.com/dzaurov) | Default device snapshot slots that give way to the KV capacity, `output_config.effort` beside disabled Thinking in the Messages API, the n-gram writer interop test's skip without its Python modules, and the RTX 5080 field report | [#283](https://github.com/iamwavecut/ninfer-all/pull/283), [#284](https://github.com/iamwavecut/ninfer-all/pull/284), [#285](https://github.com/iamwavecut/ninfer-all/pull/285) |

## Work carried from other forks

| Source | Contribution |
| --- | --- |
| [TertiumOrganum1/ninfer-3090](https://github.com/TertiumOrganum1/ninfer-3090) | the `rk4v4-e8` KV cache, the 128x64 int8 tile for ternary prefill, tool-call recovery, shared-capture surplus handling, `sm_120a` on the `mma.sync` path |
| [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) (UDPSendToFailed, Daniel Parker, Matt Anderson, keylimesoda, Sergiusz Michalik) | the E8-lattice KV codecs and `rk2v4-e8`, the whole-program CUDA build, shared-memory scale reads in INT8 attention, keys past 262,144 and four times the native window with YaRN, SFU sigmoid/SiLU/softplus, `TCP_NODELAY`, the WDDM evictable budget and DirectStorage restores on Windows, MTP windows up to 15, `/metrics`, `/slots` and `/props`, the compiled-in WebUI, split compilation |
| [IMGillusion/ninfer-disk-kv](https://github.com/IMGillusion/ninfer-disk-kv) | the disk KV tier and `--context-cache-policy rolling` |
| [kido5217/frinfer](https://github.com/kido5217/frinfer) (Fedor Suchkov) | token log probabilities on Chat Completions and Responses |
| [MirkoCovizzi/ninfer-rtx5090-mobile](https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile) | adaptive MTP |
| [Wallawalla47/ninfer-custom](https://github.com/Wallawalla47/ninfer-custom) and its successor [Infernix](https://github.com/wallawalla47/infernix) (Ian Ranson, Duncan Betts, Fedor Suchkov) | the fast INT8 and NVFP4 prompt-attention kernels, the hybrid prefix cache, n-gram copy drafting (with remesis), recency eviction, Device KV lease growth, the Host cache budget, automatic long anchors, `--assistant-prefill`, grouped `--help`, levelled logs, programmatic dependent launches, split-KV attention for short prefill steps, ModelOpt and Quasar NVFP4/FP8 conversion, a native Windows build, the reasoning-loop guard, FP32 head logits, the lossless `bf16p` KV form |
| [gzenz/ninfer](https://github.com/gzenz/ninfer) (Gideon Zenz, David Oelfke) | thorough admission search, value-aware demotion, concurrent prefill, worker out-of-memory recovery, the MTP attention window, post-thinking sampling, `GET /stats` and the `tools/monitor` dashboard, request-log rotation, derived session keys, CPU Vision, RoPE position interpolation, branch anchors, PackGQA |
| [tmark00/ninfer](https://github.com/tmark00/ninfer) | table-decoded E8 roots, the SM-count RMSNorm cutoff, the WebUI MCP proxy |
| [quardex/ninfer-all](https://github.com/quardex/ninfer-all) (A. Ugryumov) | the Turing (`sm_75`) build: its MMA forms, BF16-cache attention on FP16 tensor cores, every KV coding in 64 KiB of shared memory, the Turing path of the Q8 K-split contraction and the TITAN RTX profile |
| [2-com-org/ninfer-qwen](https://github.com/2-com-org/ninfer-qwen) (Micky Zhang) | host-side time to first token: the n-gram index on zero pages and built beside the prefill, NFC normalization only between NFC boundaries, Jinja `strip` over the trimmed characters, word-wide prefix shortlist digests, the small-T attention modules loaded at Program startup |
| [Dylan997S/ninfer-dspine-rtx5090](https://github.com/Dylan997S/ninfer-dspine-rtx5090) | DSpine on the Qwen3.5 core: the split of its code bridge into Ops and the table form of its refinement, adapted to this tree |
| pelebel, natpate | openable server URLs and CORS preflight echoes |
| Matt Anderson | wide Q5 split tiles and an idle Engine that fails only the request it cannot place |
| hyz615, lededev, NikolaFC | schema assertions in constrained decoding; the disk tier persisted on an orderly stop and MSVC-safe test builtins; Q8_0 `ssm_alpha`/`ssm_beta` in GGUF conversion |
| [b1231227](https://github.com/b1231227) | a BF16 output head in the dense GGUF recipe, and the native re-quantisation of a GGUF that exposed `ssm_out`'s stored column order ([#185](https://github.com/iamwavecut/ninfer-all/pull/185)) |
| [13thblade/ninfer-all-nvfp4](https://github.com/13thblade/ninfer-all-nvfp4) (13thblade) | NVFP4 weights on GPUs without FP4 tensor cores: the wide W4A16 kernels and the W4A8 int8 routes (`--nvfp4-w4a8-int`, `--nvfp4-w4a16`) with their tensor-parallel shares, and the integer E2M1/E4M3 expansions of the NVFP4 and FP8 KV caches |
| [ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090) (AshAlliants) | the prefill stall budget (`--prefill-stall-budget-ms`) |
| [ahostbr](https://github.com/ahostbr) | the DeepSeek-V4.1 prefill proposals behind the decoder blocks' prompt tail, the IQ1_M matrix route of routed experts and the Engram rows read once and ahead ([#238](https://github.com/iamwavecut/ninfer-all/issues/238), [#239](https://github.com/iamwavecut/ninfer-all/issues/239), [#240](https://github.com/iamwavecut/ninfer-all/issues/240)) |

## Pull requests to Neroued/ninfer ported here

| PR | Contribution | Author |
| --- | --- | --- |
| #54 | the exception that terminates the process is named | Aleksandr Iakimov |
| #97, #297 | incremental native compilation in container builds; layout state survives an overflow | Duncan Betts |
| #162, #163 | llama.cpp model metadata on `/v1/models`; timings during prompt progress | Héctor Ramón Jiménez |
| #197 | `ignore_eos` documented and pinned in the schema test | Thireus |
| #199, #221, #264, #268, #273, #286-#290 | sparse-MoE Q4 quads, MTP topology classes, the NVFP4 SwiGLU tail tile, the gated small-T attention epilogue, NVFP4 expert banks and the `qwen3_6_35b_a3b_nvfp4` recipe | Mykhailo Dementii |
| #282 | GGUF as a conversion source | giveen |
| #284 | the `qwen3_8_27b_q6` recipe | bingchengcc |
| #294 | structured output through xgrammar | Andrey Shvartsman |
| #295 | `reasoning.summary` and encrypted reasoning content | Mac |
| #299 | a repeated tool-call parameter keeps its last value | adubkov |
| #300 | the agent-harness XML tool-call forms | pkochubey, Ian Ranson |
| #304, #305, #307 | the speed-of-light estimator, fused attention RMSNorm with NVFP4 quantization, single-pass target log probabilities | Duncan Betts |
| #309 | quoted reasoning closes and tool-call markers | Fedor Suchkov |
| #311 | pipelined Q4/Q5 groups for the attention input | MOVIBALE |
| #316 | Copilot tool shapes and longer tool names | Damian Sromek |

Upstream `master` changes ported here keep their authors: the unified Linear templates, the
two-stage GDN prefill, parallel query tiles, MX FP8 MMA and Minnnn's column-band routing.

## Models and references

- The Ternary Bonsai 2 27B artifact combines PrismML's ternary weights, ProCreations' MTP head and
  DFlash2 adapter, and Qwen's checkpoint, all Apache-2.0.
- The Qwen3.8-Flash-Next artifacts import ISTA-DASLab's GSQ-RCO GGUF releases and Unsloth's MTP
  block, under the Qwen Community License 1.0 of the model.
- Qwen3.8-Flash-Next support was designed against the transformers, vLLM and llama.cpp
  implementations of the model and [Strata](https://github.com/Niko1221/Strata) (MIT, copyright
  2026 Niko1221 and contributors), whose transcription served as a reference for the n-gram
  hash, the expert cache and its measured pitfalls, and whose CPU-computed expert misses led to
  the CPU share. The expert tier that every MoE family with host or disk experts shares
  (`core/expert_tier`: frequency counts, rounds and their measured constants) and the residency map
  of streamed sparse-attention KV (`ops/kv_stream`: block claims and the clock sweep) follow
  Strata's designs.
- The DeepSeek-V4.1-Flash artifacts import ISTA-DASLab's GSQ-RCO IQ1_M GGUF and its Engram table
  GGUF, under the model's MIT license. Its support follows DeepSeek's release code
  (`inference/model.py`, `kernel.py` and `encoding/encoding.py`) and was measured against Strata's
  `strata-dsv41`. The order in which `--draft-experts` keeps the DSpark drafter's experts is
  [Basalt](https://github.com/jesdga95/basalt)'s profile `deepseek41-40x384-dspark.prof` (MIT,
  copyright 2026 Jesús Gulfo).
- The DSpine drafter artifact converts Tencent's
  [Qwen3.8-27B-DSpine-b16](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16) (Apache-2.0).
  DSpine support follows Tencent's reference implementation
  ([Tencent/DSpine](https://github.com/Tencent/DSpine), Apache-2.0) and its paper,
  [arXiv 2609.36173](https://arxiv.org/abs/2609.36173).
- The SiLU shared by the GPU's GGUF expert kernels and the CPU's expert products evaluates e^x with
  the single-precision polynomial of Stephen L. Moshier's Cephes Math Library.
- llama.cpp's GGUF block kernels are vendored in `third_party/ggml-quants` under their MIT license;
  its `README.ninfer.md` lists NInfer's changes to them.
