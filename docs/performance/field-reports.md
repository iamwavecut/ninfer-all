# Field reports

Measurements that users took on their own machines and sent as GitHub issues. Each report keeps
its reporter's setup, method and caveats. The methods differ from the
[project's methods](../performance.md), so these numbers are not read against the project's tables
or against each other as a benchmark. [Send a report](#send-a-report) to add a card or a workload.

| Card | OS | Model | Workload | Report |
|---|---|---|---|---|
| RTX 5070 Ti, 16 GB | Windows 11 | Qwen3.8-27B GSQ-RCO IQ3_S, Swift 1.5 IQ3_S | single requests to 104K tokens, against llama.cpp | [RTX 5070 Ti](#rtx-5070-ti-windows-11) |
| RTX 5080, 16 GB | Windows 11, WSL2 | Qwen3.8-27B GSQ-RCO IQ3_S | single requests to 160K tokens, largest windows | [RTX 5080](#rtx-5080-wsl2) |
| RTX 4090, 24 GB | Windows | Qwen3.8-27B GSQ-RCO IQ3_S, Swift 1.5 IQ3_S | Qwen Code sessions with DFlash2 5 and 7 | [RTX 4090: DFlash2](#rtx-4090-qwen-code-with-dflash2) |
| RTX 4090, 24 GB | Windows | Swift Qwen3.8 GSQ-RCO IQ3_S | agent traffic with MTP 3, four lanes and the hybrid prefix cache | [RTX 4090: MTP 3](#rtx-4090-agent-traffic-with-mtp-3-and-the-hybrid-prefix-cache) |
| RTX 4080 and RTX 4090 | Windows 11 | Qwen3.8-27B GSQ-RCO IQ3_S | Hermes Desktop | [RTX 4080 and 4090](#rtx-4080-and-rtx-4090-windows-11) |
| RTX 4090, 24 GB | not given | Qwen3.8-27B UD-Q3_K_XL, GSQ-RCO IQ3_S | llmprobe's hard suite, 50 questions | [RTX 4090: llmprobe](#rtx-4090-llmprobe-hard-suite) |
| RTX 5060 Ti, 16 GB, two and four | Linux | Qwen3.8-27B NVFP4 | `--tensor-parallel` prompts to 28.8K tokens, KV capacity | [RTX 5060 Ti](#rtx-5060-ti-two-and-four-cards) |
| Tesla T10, 16 GB, two | Linux | Inkling 2 Mini UD-IQ4_XS | `--tensor-parallel` without peer access, retrieval prompts to 261,760 tokens, against llama.cpp | [Tesla T10](#tesla-t10-two-cards) |
| RTX 3070 Ti, 8 GB | Windows 11 | Inkling 2 Mini UD-IQ4_XS | experts in host memory, the largest context with `rk4v4`, a 168K-token retrieval prompt | [RTX 3070 Ti](#rtx-3070-ti-windows-11) |

## RTX 5070 Ti, Windows 11

Reported by @NickWalters in
[iamwavecut/ninfer-all#111](https://github.com/iamwavecut/ninfer-all/issues/111), 20 October 2026.

- RTX 5070 Ti 16 GB at its default 300 W limit, driver 610.47. The card also drives one 3440x1440
  desktop, which held about 1.3 GB of its memory at idle.
- Windows 11 Pro (build 26200), Ryzen 7 9800X3D, 32 GB of RAM.
- `master` at `c04ea31`, a `120a` build with CUDA 13.4 and Visual Studio 2026 Build Tools from
  `scripts\build-native.bat`.
- ISTA-DASLab `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`, converted with
  `--recipe qwen3_8_27b_gguf --components text,mtp`.

The MTP rows:

```powershell
ninfer-serve.exe qwen3_8_27b_gsq_rco_iq3_s.noprop.ninfer `
  --model-id qwen3.8-27b --max-concurrency 1 `
  --gdn-state-fp16 --kv-dtype rk4v4-e8 --host-cache-mib 0 `
  --max-context 98304 --kv-capacity auto --kv-headroom-mib 192 `
  --spec mtp --draft-tokens 3
```

The rows without a drafter replace the last two lines with
`--max-context 122880 --kv-capacity 122880 --spec none`. llama.cpp is the b10930 CUDA 13.3 release
build with the same GGUF, `-np 1`, a 120K window, a `q4_0` KV cache and no MTP.

One request at a time, 512 output tokens, temperature 0. The prompt is source code from this
repository followed by "Summarize what this code does in detail." Prompt rate is prompt tokens over
the time to the first token; decode excludes the first token. Each cell is one run, prompt / decode
in tok/s:

| Prompt tokens | NInfer, MTP 3, 96K window | NInfer, no drafter, 120K window | llama.cpp, no MTP, 120K window |
|---:|---:|---:|---:|
| 1.2K | 1,908 / 133.1 | 1,977 / 57.7 | 1,288 / 51.3 |
| 16.8K | 2,037 / 113.9 | 2,058 / 56.4 | 1,581 / 46.1 |
| 33.4K | 1,924 / 112.6 | 1,945 / 55.2 | 1,387 / 42.2 |
| 56.1K | 1,798 / 110.7 | 1,805 / 53.3 | 1,172 / 38.2 |
| 86.0K | 1,650 / 102.8 | | |
| 104.2K | | 1,584 / 50.7 | 899 / 31.0 |

**Memory.** Peak use in `nvidia-smi`, desktop included, was 15,600 MiB with MTP and 15,443 to
15,463 MiB without. With `rk4v4-e8` the largest windows that started were 98,304 tokens with MTP 3
and 122,880 without a drafter.

**Swift 1.5.** UkisAI's `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` is a Qwen3.8-27B fine-tune
whose config, tokenizer and chat template are byte-identical to Qwen's. It converts with the same
recipe and the Qwen base folder as `--model` (846 objects, 39 s on the CPU) and serves with the same
flags. In the reporter's eval of 10 API tasks and 2 agent tasks, 3 runs each, both models passed 34
of 36. Swift used 17% fewer tokens, and on the API tasks it averaged 125 output tok/s against 114
for the stock model (completion tokens over wall time).

**Conversion without CUDA.** A PyTorch without CUDA converts with `--device cpu`; the converter's
error message names the option.

## RTX 5080, WSL2

Reported by @dzaurov in [iamwavecut/ninfer-all#270](https://github.com/iamwavecut/ninfer-all/issues/270),
7 November 2026. The server runs daily as a Claude Code backend (`ANTHROPIC_BASE_URL`).

- RTX 5080 16 GB at its default 360 W limit, driver 617.42. The card also drives one 2560x1440
  desktop, which held 0.3 to 0.8 GB of its memory.
- Windows 11 Pro (build 26300), Core i7-14700, 32 GB of RAM; WSL2 (kernel 6.18) with Ubuntu 24.04,
  given 24 GB and 24 threads.
- `master` at `bdb141f4`, a `120a` build with CUDA 13.3 and GCC 13 from
  [the Linux instructions](../build-linux.md).
- ISTA-DASLab `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`, converted with
  `--recipe qwen3_8_27b_gguf --components text,vision,mtp,dflash2 --proposal`.

The MTP rows:

```bash
ninfer-serve qwen3_8_27b_gsq_rco_iq3_s.ninfer --model-id qwen3.8-27b \
  --max-context 106496 --kv-capacity 106496 --kv-dtype rk4v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3
```

The rows without a drafter use `--max-context 163840 --kv-capacity 163840` and no `--spec`. The
method is that of the [RTX 5070 Ti report](#rtx-5070-ti-windows-11), with thinking off and a unique
nonce at the start of each prompt. The host ran unrelated CPU-heavy test suites during the runs
(load average 3 to 20). Prompt / decode in tok/s, one cell per run:

| Prompt tokens | MTP 3, 104K window | no drafter, 160K window |
|---:|---:|---:|
| 1.3K | 2,708 / 159.3 | 2,687 / 71.5 · 2,677 / 71.4 |
| 16.9K | 2,789 / 77.1 | 2,799 / 49.7 · 2,791 / 49.6 |
| 33.5K | 2,655 / 130.0 | 2,666 / 67.8 · 2,662 / 67.9 |
| 56.2K | 2,202 / 134.1 | 2,195 / 65.9 · 2,207 / 65.7 |
| 86.1K | 2,120 / 132.1 | |
| 104.1K | 1,938 / 127.3 | 1,962 / 61.7 · 1,948 / 61.7 |
| 160.1K | | 1,789 / 42.9 · 1,781 / 42.8 · 1,801 / 56.9 · 1,748 / 57.3 |

**Decode in two groups.** Six more runs at 160K, alternating `master`, `master` with
`--host-cache-mib 0` and an October build, each decoded at 43 to 44 or at 57 to 59 tok/s for the
whole answer, every variant in both groups. At 16.9K the groups are about 50 and 70 tok/s. The
server's `throughput` lines show the slower rate steady after the first window. The reporter
attributes it to WDDM residency on a nearly full card. MTP acceptance varies with the summary: the
77.1 at 16.9K is one such run, against 148.9 for the October build at the same length.

**Largest windows**, bisected in 4,096-token steps with `rk4v4`, `--gdn-state-fp16` and the
desktop attached:

| | `master` at `bdb141f4` | with `--device-snapshot-slots 1` | October build (`d1a0e070`) |
|---|---:|---:|---:|
| MTP 3 | 122,880 | 131,072 | 131,072 |
| no drafter | 184,320 | 192,512 | 196,608 |

The gap is the hybrid prefix cache's default device snapshot slots
([#266](https://github.com/iamwavecut/ninfer-all/issues/266)); since
[PR #285](https://github.com/iamwavecut/ninfer-all/pull/285) the default slots give way to a window
that would not fit beside them. DFlash2 5 fits an 8,192-token window with this artifact.

**WSL.** After `kill -9` of a running server, about 235 MiB of device memory stayed allocated
inside WSL until `wsl --shutdown`, and later startups failed by that margin. A SIGTERM stop returned
it.

## RTX 4090: Qwen Code with DFlash2

Reported by @spk-c in [iamwavecut/ninfer-all#96](https://github.com/iamwavecut/ninfer-all/issues/96)
and its [follow-up](https://github.com/iamwavecut/ninfer-all/issues/96#issuecomment-6297428000),
17 to 20 October 2026: Qwen Code sessions with tool calls on an RTX 4090 (24 GB) under Windows.
Both setups below name `--derive-session-keys` and `--auto-long-anchors`, which select the
checkpoint catalog instead of the default hybrid prefix cache
([`--use-alt-prefix-caching`](../serving.md#server-options)).

### DFlash2 5, Qwen3.8-27B IQ3_S

The first session served Qwen3.8-27B GSQ-RCO IQ3_S with:

```powershell
.\build-ninja\apps\ninfer-serve.exe `
  ".\models\Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer" `
  --model-id qwen3.8-27b --host 127.0.0.1 --port 8080 `
  --max-context 131072 --kv-capacity 131072 --kv-dtype rk8v4 `
  --gdn-state-fp16 --max-concurrency 1 `
  --spec dflash2 --draft-tokens 5 `
  --kv-lease-growth --derive-session-keys --auto-long-anchors `
  --vision --vision-residency overlay
```

The server log of 221 completed requests (prompts up to 101,389 tokens) gives:

| Measure | Value |
|---|---:|
| decode, median of the requests | 168 tok/s |
| decode, mean of the requests | 180.9 tok/s |
| decode, highest request (very high draft acceptance) | 484.4 tok/s |
| prompt rate, median | about 2,070 tok/s |
| prompt-cache hits, median | 94.6% |
| time to the first token, median | about 1.9 s |

Requests with more than 95% of the prompt in the cache began to answer in about a second; long
prompts with almost nothing reused sometimes waited more than 30 s, about what a cold prefill of
that length takes at these rates. The log held no `WARN`, `ERROR` or `FATAL` entry.

Against llama.cpp with a Q4_K_M quant, in separate tests: a short decode ran at 70.57 tok/s on
llama.cpp and 193.77 on NInfer, and a prompt at 2,347 and 2,828 tok/s. In long sessions llama.cpp
decoded about 40 tok/s, at most about 100, by the reporter's recollection. The quants, windows and
speculation differ, so this is no controlled comparison.

### DFlash2 7, Swift 1.5 IQ3_S

The later sessions used build `5ed86dff` (`89`, CUDA 13.4) with a locally calibrated RTX 4090
profile, Huihui's Qwen3.8-27B Swift 1.5 GSQ-RCO IQ3_S artifact with its DFlash2 drafter, Qwen
Code 0.25.3, a 176,128-token window, `rk4v4` with `--kv-capacity auto`, `--host-cache-mib 16384`,
`--max-concurrency 2`, `--derive-session-keys`, `--auto-long-anchors`, `--kv-lease-growth`,
`--gdn-state-fp16` and Vision overlay. Each column is one long session with its own task mix:

| Measure | DFlash2 5 | DFlash2 7 | DFlash2 7, cuBLAS prefill 2,048 |
|---|---:|---:|---:|
| coding requests | 387 | 309 | 226 |
| decode, median | 154.3 tok/s | 162.1 tok/s | 156.2 tok/s |
| decode, weighted by output | 146.6 tok/s | 174.8 tok/s | 148.4 tok/s |
| time to the first token, median | 1.50 s | 2.17 s | 1.37 s |
| prompt-cache hits, weighted | 93.78% | 91.96% | 93.39% |
| Engine recoveries | 0 | 0 | 0 |

- **Draft count.** In the reporter's long-output decode tests DFlash2 7 led DFlash2 5 throughout,
  210.3 against 184.3 tok/s at a 160K context.
- **cuBLAS prefill.** `--prefill-cublas --no-prefill-cublas-projections --prefill-chunk 2048`
  raised a synthetic 160K cold prefill with DFlash2 5 from 2,391.7 to 2,470.2 tok/s (+3.3%). The
  session with it showed no overall gain: its compaction prompts ran at about 2,320 tok/s against
  2,398 without it, on different prompts. The reporter keeps cuBLAS prefill off.
- **Compaction.** The DFlash2 7 session without cuBLAS ran 14 context compactions, about 23.7
  minutes in total, with a median time to the first token of about 54.4 s. Qwen Code's compaction
  rewrites most of the conversation, so the earlier prefix checkpoints no longer match it. Cache
  reuse otherwise stayed high above 100K prompt tokens.
- **MTP 3 with the hybrid prefix cache.** On the same card, a synthetic 100K test gave about 2,759
  tok/s cold prefill, 140.4 tok/s decode and a 0.088 s time to the first token on an exact repeat
  (99.99% cache hits). One real session gave a median decode of about 135.5 tok/s and 49.5%
  weighted cache reuse, with drops on growing or rewritten conversations. The drafter, model
  variant, settings and workloads differed from the DFlash2 sessions, so this does not isolate the
  cache.
- **Reasoning effort.** Qwen Code 0.25.3 with `"model": {"reasoningEffort": "medium"}` sent no
  effort: `requested_reasoning_effort` was `null` on all 561 requests of the two DFlash2 7
  sessions, and the server ran without `--default-reasoning-effort`. Such a request renders with
  the template's default effort, `xhigh` for the Qwen3.8 template. In one session of 4 h 42 min,
  1,416,104 of 1,712,734 output tokens (82.7%) were thinking, with 23 compactions.
  `--default-reasoning-effort medium` sets the effort of every request that names none
  ([reasoning effort](../serving.md#openai-chat-completions)).

## RTX 4090: agent traffic with MTP 3 and the hybrid prefix cache

Reported by @foxhoundhcm in
[iamwavecut/ninfer-all#96](https://github.com/iamwavecut/ninfer-all/issues/96#issuecomment-6251383234),
18 October 2026: an RTX 4090 (24 GB) under Windows, an `89` build of `4027aa63` with CUDA 13.4.3
(the cold-prefix table below ran on `bda41158`), a Swift Qwen3.8-27B GSQ-RCO IQ3_S artifact:

```powershell
ninfer-serve.exe swift-qwen38-rco-iq3s.ninfer `
  --host 127.0.0.1 --port 18006 --model-id qwen3.8-27b `
  --kv-dtype rk4v4 --spec mtp --draft-tokens 3 --mtp-attention-window 4096 --lm-head-draft `
  --max-context 262144 --default-max-tokens 32768 --kv-capacity auto --max-concurrency 4 `
  --use-alt-prefix-caching --host-cache-mib 6144 --concurrent-prefill --structured-output `
  --gdn-state-fp16 --vision --vision-residency overlay `
  --model-suspend --sleep-idle-seconds 900 --recover-invariant-failures --stats-port 18007 `
  --request-log-jsonl req.jsonl --request-log-max-mib 200 --request-log-keep 3
```

Cold prefixes, 256 output tokens, 3 runs per depth:

| Depth | Prompt tokens | Decode, tok/s | Prompt, tok/s | MTP acceptance |
|---|---:|---:|---:|---:|
| short | 145 | 187.5 | 1,423 | 0.724 |
| middle | 44,006 | 151.9 | 3,194 | 0.646 |
| deep | 133,007 | 135.8 | 2,533 | 0.603 |

From the request log of 7,127 agent requests (median prompt 87K tokens, largest 231K):

| Measure | Median | 90th percentile | Highest |
|---|---:|---:|---:|
| decode, tok/s | 123.6 | 176.0 | 482.9 |
| prompt rate of requests that computed 1K tokens or more, tok/s | 2,314 | 3,532 | 3,738 |
| time to the first token, s | 0.56 | 11.1 | 226.7 |

Time to the first token of the prompts longer than 40K tokens, by the share served from the cache:

| Cache hits | Requests | Median | 90th percentile |
|---:|---:|---:|---:|
| 0 to 50% | 225 | 24.1 s | 72.6 s |
| 50 to 90% | 216 | 9.3 s | 33.3 s |
| 90 to 99% | 1,618 | 1.38 s | 4.3 s |
| above 99% | 2,932 | 0.38 s | 0.8 s |

A prefix the cache has never seen costs a full prefill, about 2.9K tok/s on this card: about 35 s
for 100K tokens. Against the checkpoint catalog on the same card:

- three concurrent requests that share a new 73K-token prefix computed 73,085 prompt tokens with the
  hybrid prefix cache and 218,877 with the catalog; the burst took 24.7 s against 75.6 s;
- the same prompt sent again started in 0.10 s instead of 25.6 s, reusing `shared_stable_prefix`
  instead of starting from `root`;
- in real traffic, `shared_stable_prefix` reuse rose from 612 of 6,575 requests on the catalog
  (before 11 October) to 1,756 of 7,127 on the hybrid prefix cache, and requests below 50% cache
  hits fell from 362 to 225.

A single cold prefill takes the same time with either cache.

- **KV format.** `rk4v4` instead of `rk8v4` took the runtime's memory from 9.66 to 7.54 GiB, which
  pays for the Host prefix cache.
- **Host cache.** `--host-cache-mib 6144` runs on this card and 12288 did not start.
- **Draft count.** MTP 5 committed 17 to 22% more tokens per round than MTP 3, while its
  acceptance at 120K fell from 0.607 to 0.482.
- **Concurrency.** Four lanes keep the same window and KV pool as one, and 2 to 4 of them were in
  use. Without `--default-max-tokens`, a request's output limit follows its lane's share of the
  pool ([default output limit](../serving.md#default-output-limit)): a 60K prompt received 5,983
  tokens at four lanes and 71,519 at two, and long agent turns ended with
  `finish_reason=length`.
- **Slow turns.** The request log attributes a slow turn: `prefix_reuse_path`,
  `prefix_cache_hit_tokens`, `computed_prefill_tokens` and the prefill, decode and first-token
  times ([request log](../serving.md#structured-request-log)). `root` on an otherwise unchanged
  long conversation marks a rewritten prefix: a compaction or summary, a changed system prompt or
  tool schema, or a client that serializes the history differently.

## RTX 4080 and RTX 4090, Windows 11

Reported by @jabbor in [iamwavecut/ninfer-all#70](https://github.com/iamwavecut/ninfer-all/issues/70),
13 October 2026: `Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` on an RTX 4080 and on an RTX 4090,
each machine with 64 GB of RAM, served to Hermes Desktop. The reporter found it much faster than
llama.cpp, with larger contexts. The report gives no rates.

## RTX 4090: llmprobe hard suite

Reported by @lin16303 in
[iamwavecut/ninfer-all#471](https://github.com/iamwavecut/ninfer-all/issues/471), 25 November
2026: [llmprobe](https://github.com/ddalcu/llmprobe) `--eval --eval-suite hard` on an RTX 4090, 50
questions from GPQA Diamond, SuperGPQA, AIME 2025 and COMPSEC subsets, against two Qwen3.8-27B
artifacts. llmprobe reports the decode rate and the time to the first token for information only.

| Artifact | Decode | First token | Correct | Out of tokens | Tokens in / out | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| Unsloth's UD-Q3_K_XL, converted by the reporter with `tools.convert` | 143.6 tok/s | 85 ms | 41 / 50 | 2 | 129,258 / 590,283 | 70 min 51 s |
| `Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3` | 153.8 tok/s | 81 ms | 38 / 50 | 7 | 129,296 / 785,371 | 86 min 7 s |

Three questions (6 points) separate the two scores; the standard error of the difference between
two independent 50-question scores at these rates is about 8 points, and the report has no
per-question results for a paired comparison. The OS, build, command, drafter, sampling settings and
power limit are not in the report.

## RTX 5060 Ti, two and four cards

Reported by @sean640112 and @mmknet2025 in
[iamwavecut/ninfer-all#170](https://github.com/iamwavecut/ninfer-all/issues/170), 28 and 29 October
2026. Both run `master` at `33163f05`, a `120a` build, with tensor parallelism over RTX 5060 Ti 16 GB
cards.

| | sean640112 | mmknet2025 |
|---|---|---|
| Cards | four, PCIe Gen4 x8 each | two, PCIe Gen5 x8 + x8 |
| Peer access | every pair, community P2P driver 610.57.04; NCCL all-reduce about 12.2 GB/s | both directions, community P2P patch, driver 615.71.09 |
| Toolchain | CUDA 13.3.73, GCC 13.4.0, Ubuntu 26.04.1 | CUDA 13.4.92, GCC 13.4 |
| Artifact | `neroued/Qwen3.8-27B-nvfp4-NInfer` (23,719,715,844 bytes) | an NVFP4 conversion of Qwen3.8-27B with 19.8 GiB of weights (named in the issue) |

sean640112's command; the two-card arm uses `--devices 0,1` and `--max-context 32768`:

```bash
ninfer-serve qwen3_8_27b_nvfp4.ninfer --devices 0,1,2,3 --tensor-parallel \
  --max-context 262144 --kv-capacity auto --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft --max-concurrency 1
```

mmknet2025 runs `--devices 0,1 --tensor-parallel --max-context 131072 --kv-capacity 131072` with
the same KV and drafter flags. Greedy requests, a cold prefix for the time to the first token;
prompt rate is prompt tokens over that time. sean640112 starts one server per arm, mmknet2025
reports the median of three runs. Prompt / decode in tok/s:

| Prompt tokens | four cards, Gen4 x8 | two cards, Gen4 x8 | two cards, Gen5 x8 |
|---:|---:|---:|---:|
| 742 | | | 1,103 / 72.1 |
| 940 | 418 / | 631 / | |
| 5,777 | | | 1,283 / 77.5 |
| 7,012 | 544 / 82.1 | 975 / 75.3 | |
| 28,852 | 542 / 78.3 | 958 / 67.9 | |

The 940-token rows give no decode rate: the second request of the pair hit the prefix cache. On the
same box four cards prefill the 7,012- and 28,852-token prompts 1.77 to 1.79 times slower than two
and decode 9% to 15% faster. A groupwise-int conversion of the same model on the four cards
(`master` at `adf969f`) prefilled 28.8K tokens at 531 tok/s and decoded at 74.3. On the two-card
box another engine, a fork of the upstream project with its own NVFP4 artifact,
`--prefill-chunk 3072` and an exchange that overlaps the prompt's compute, prefilled 5,775 tokens
at 6,441 tok/s; engine and artifact differ, so the reporter does not read it as an A/B.

**Capacity.** On two cards an int8 KV cache of 262,144 tokens does not fit: device 0 needs
5,656,908,608 bytes for the runtime reservation and has 4,232,773,632. An explicit
`--kv-capacity 131072` starts, with 11.4 and 8.33 GiB of weights on the two cards. With
`--kv-capacity auto` the same 131,072-token window stops on device 0, 34 MB short: the reservation of
3,165,475,648 bytes plus 1 GiB of automatic headroom against 4,204,920,832 bytes left after the
weights. Four cards hold 262,144 tokens with `--kv-capacity auto`, with 7.16, 4.04, 4.33 and
4.33 GiB of weights; the startup report lists 5.25 GiB free.

## Tesla T10, two cards

Reported by @uos1231234 in
[iamwavecut/ninfer-all#436](https://github.com/iamwavecut/ninfer-all/issues/436), 23 November 2026.

- Two Tesla T10 16 GB (`sm_75`) on PCIe Gen3 x16 under one host bridge, one NUMA node, without peer
  access (`cudaDeviceCanAccessPeer` returns 0 both ways). Driver 615.71.09.
- Ubuntu 22.04, Xeon E5-2686 v4 (36 threads), 31,924 MiB of RAM.
- Release 2026-11-23.5 (`64dec86e`), the Linux x64 CUDA 13.4 package.
- [WaveCut/Inkling-2-Mini-UD-IQ4_XS-NInfer-v3](https://huggingface.co/WaveCut/Inkling-2-Mini-UD-IQ4_XS-NInfer-v3/tree/50d6ad60cfcd170633ead778e3a09958fc1e7887),
  `Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer` (16,494,910,976 bytes).

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
  ninfer-serve Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer --cuda-backend 13 \
    --model-id inkling-2-mini --max-context 262144 --max-concurrency 1 --structured-output \
    --devices 1,0 --tensor-parallel --tensor-split 1,3 \
    --kv-dtype mxfp8 --prefill-chunk 2048 --default-max-tokens 32768 \
    --spec dspark --draft-tokens 3 --adaptive-mtp --draft-min-p 0.5
```

`--tensor-parallel` divides the routed experts: the startup log gives the first device of `--devices`
experts 0-31 and the second 32-127. The rows without drafts replace the last line with `--spec none`.

Each row is one request to a loaded server through the native `/completion` stream: a filler corpus
with one ledger ID at its start and one at its end, a grammar for the two IDs, greedy sampling, a
256-token output budget and `cache_prompt: false`. First content is the time from the request to
the first non-empty `content` event. Prefill and decode are the engine's `timings`; decode counts
the model's thinking tokens (136 of 151 at 32K, 86 of 101 at the longer prompts). Every run
evaluated the whole prompt with no cached tokens, stopped at EOS and returned both IDs. The runner
script is in the issue.

| Prompt tokens | Drafts | First content (s) | Prefill (tok/s) | Decode (tok/s) | Drafts accepted |
|---:|---|---:|---:|---:|---:|
| 32,768 | none | 35.60 | 1,017 | 44.1 | |
| 32,768 | DSpark, adaptive | 34.52 | 1,013 | 71.6 | 100 / 138 |
| 131,072 | none | 166.77 | 799 | 36.7 | |
| 261,760 | none | 426.57 | 618 | 29.5 | |

On the same host with the same 32,768 token IDs, llama.cpp with Unsloth's UD-Q4_K_M split by
layers and an F16 KV cache reached first content in 98.21 s (prefill 334 tok/s), and a banded
tensor-parallel llama.cpp configuration with IQ4_NL and an F16 KV cache in 111.87 s (293 tok/s).
Their weights and KV formats differ from NInfer's. Visible-text throughput, latency percentiles and
DSpark at 131K and 261K tokens were not measured.

## RTX 3070 Ti, Windows 11

Reported by @cedrik-fuoco in
[iamwavecut/ninfer-all#451](https://github.com/iamwavecut/ninfer-all/issues/451) and
[#467](https://github.com/iamwavecut/ninfer-all/pull/467), 24 November 2026.

- RTX 3070 Ti 8 GB (`sm_86`), Windows 11 with the WDDM driver.
- The change of #467 (`83e37de`), an `86` build with CUDA 12.9 and Visual Studio 2022 Build Tools
  17.14 (MSVC 14.44.35207): `cmake -G Ninja -DCMAKE_CUDA_ARCHITECTURES=86` in vcpkg manifest mode.
- `Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer`.

```bat
ninfer-serve Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer --model-id inkling-2-mini ^
  --host 127.0.0.1 --port 8094 --max-context 243200 --kv-dtype rk4v4 --expert-residency host
```

The build measured here ran Inkling 2's passes with host experts eagerly under WDDM, without CUDA
Graphs, and kept no memory free beside them. Inkling 2 now keeps its CUDA Graphs there
([Windows build](../build-windows.md#build)) and the 320 MiB it keeps free for them, about 37,400
tokens of `rk4v4` context (8.75 KiB a token) below the ceiling measured here.

- **Largest context.** On the 128-token grid the largest `--max-context` that started was 243,200
  tokens; 243,712 was refused. At 243,200 the server took 6.99 GiB and left 81 MiB of the card free.
- **Decode.** 52.4 tok/s on a 512-token answer.
- **Long prompt.** A 168,201-token prompt with a needle at half its depth: the answer named the
  needle; prefill ran at 409 tok/s and decode at 20 to 33 tok/s.
- **Same text.** A fixed prompt at temperature 0 (seed 1234, 512 tokens) gave the same text at
  212,992 and 243,200 tokens of context, and the same text as a build without #467.

The CPU, host memory, driver version and power limit are not in the report. Prefill and decode at
other prompt lengths and the other KV formats were not measured.

## Send a report

Open an [issue](https://github.com/iamwavecut/ninfer-all/issues) with:

- the card, its memory, power limit and driver, and whether it also drives a display;
- the OS, the CPU and its RAM;
- the build: commit, architecture (`120a`, `89`, `86`, ...), CUDA and compiler;
- the artifact and how it was converted;
- the full `ninfer-serve` or `ninfer` command;
- the method: prompt and output lengths, sampling, how many runs, and how the rates were computed;
- the results, with anything that failed or did not fit.
