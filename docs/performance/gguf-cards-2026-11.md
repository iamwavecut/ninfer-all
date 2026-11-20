# GGUF models across cards, November 2026

Prompt and decode rates of the GGUF families on eight cards from one release build (`4d58dc620c`,
the CUDA 12.9 package before Blackwell and the CUDA 13.4 package on Blackwell): Gemma 4 26B-A4B and
31B, Flint, Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash. Each prompt runs twice: with the built-in
device profile, whose bands stage the MMQ tiles through `cp.async`
([what is calibrated](../device-profiles.md#what-is-calibrated)), and with the same bands on the
compiled tile loop. Both give the same mean NLL on every card. The setup is at the
[end](#how-these-were-measured).

## Gemma 4 26B-A4B

[QAT Q4_0 release](https://huggingface.co/WaveCut/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-NInfer-v3), tok/s;
prompts with the compiled loop → with the built-in profile; decode of 256 tokens without drafts:

| Card | Power limit | Prompt 3,655 | Prompt 364 | Decode |
|---|---:|---:|---:|---:|
| RTX PRO 6000 Blackwell Server Edition | 600 W | 15.7k → 21.0k | 5.55k → 8.49k | 344.5 |
| RTX 5090 | 450 W | 16.6k → 20.5k | 6.09k → 9.12k | 398.8 |
| RTX 4090 | 450 W | 15.7k-15.8k | | 264.0-264.3 |
| L40S | 350 W | 12.3k → 14.2k | 6.00k → 6.48k | 218.2 |
| RTX 3090 | 390 W | 6.72k → 7.79k | 3.72k → 3.93k | 216.6 |
| A100-SXM4-80GB | 400 W | 5.79k → 7.26k | 2.23k → 3.33k | 232.2 |
| A100-SXM4-40GB | 400 W | 5.74k → 6.98k | 2.38k → 3.17k | 221.7 |
| RTX 5060 Ti, host experts | 180 W | 3.77k → 4.78k | 1.15k → 1.54k | 181.2 |

The RTX 4090 ran the built-in profile only, twice: its rental ended before the compiled-loop runs
and Flint.

## Flint

[UD-Q4_K_XL release](https://huggingface.co/WaveCut/Flint-UD-Q4_K_XL-NInfer-v3), tok/s; experts in
pinned host memory on cards under 40 GB:

| Card | Prompt 3,502 | Decode |
|---|---:|---:|
| RTX PRO 6000 Blackwell Server Edition | 13.5k → 19.3k | 300.3 |
| RTX 5090 | 13.5k → 17.5k | 282.9 |
| L40S | 10.8k → 13.1k | 207.6 |
| RTX 3090 | 4.24k → 4.82k | 151.1 |
| A100-SXM4-80GB | 5.40k → 7.55k | 206.4 |
| A100-SXM4-40GB | 5.28k → 6.97k | 196.5 |
| RTX 5060 Ti | 942.9 → 983.9 | 62.8 |

Host experts depend on the host: the RTX 3090 host's load average stayed between 10 and 25 during
these runs, and the RTX 5060 Ti's slot is PCIe 3.0.

## Gemma 4 31B

[QAT Q4_0 release](https://huggingface.co/WaveCut/gemma-4-31B-it-QAT-Q4_0-MTP-NInfer-v3), the dense
model, tok/s:

| Card | Prompt 3,655 | Prompt 364 | Decode |
|---|---:|---:|---:|
| L40S | 2.78k → 2.85k | 2.14k → 2.10k | 37.4 |
| RTX 3090 | 1.47k → 1.47k | 1.17k → 1.16k | 42.5 |
| A100-SXM4-80GB | 1.29k → 1.29k | 1.07k → 1.11k | 59.3 |
| A100-SXM4-40GB | 1.27k → 1.28k | 1.04k → 1.03k | 56.3 |
| RTX 4090 | 3.05k → 3.20k-3.21k | 2.54k → 2.61k | 49.1 |
| RTX 5090, 600 W board | 4.33k → 4.59k | 3.68k → 3.74k | 81.9 |

In `4d58dc620c` the 31B stops at startup on every Blackwell card (`gemma4 ffn_combine: too many
resources requested for launch`); the RTX 4090 and RTX 5090 rows come from the next build, which
fixes that. On the RTX 4090 both builds give the same mean NLL and text; `4d58dc620c` measured
3.21k-3.22k, 2.63k and 49.2 there.

## Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash

| Model | Card | Prompt | Decode |
|---|---|---:|---:|
| Flash-Next GSQ-RCO Q2_0 | RTX PRO 6000 Blackwell Server Edition | 4.84k → 6.17k | 157.4 |
| DeepSeek-V4.1 Coder IQ2_XXS, host experts | RTX PRO 6000 Blackwell Server Edition | 1.54k → 1.62k | 77.9 |
| Flash-Next GSQ-RCO Q2_0 | A100-SXM4-80GB | 2.07k → 2.44k | 103.0 |

## Mean NLL

`ninfer-perplexity`'s mean NLL in nats a token, 4,096-token disjoint windows of the first 40,000
bytes of `eval/corpora/perplexity-1m/data/wikitext/00.txt`, BF16 KV (Flash-Next: 2,048-token windows
of the first 12,000 bytes, INT8 KV). Each card gives one value with both profiles; cards differ from
each other because their profiles pick different kernels:

| Card | Gemma 4 26B-A4B | Gemma 4 31B | Flint | Flash-Next | DeepSeek-V4.1 |
|---|---:|---:|---:|---:|---:|
| RTX PRO 6000 Blackwell Server Edition | 7.931294 |  | 3.017757 | 0.844785 |  |
| RTX 5090 | 7.927601 |  | 3.011890 |  |  |
| L40S | 7.948346 | 7.846023 | 3.005831 |  |  |
| RTX 3090 | 7.928512 | 7.854544 | 3.018639 |  |  |
| A100-SXM4-80GB | 7.948594 | 7.843899 | 3.007847 | 0.842853 |  |
| A100-SXM4-40GB | 7.948594 | 7.843899 | 3.007847 |  |  |
| RTX 5060 Ti | 7.935034 |  | 3.012899 |  |  |
| RTX 4090 | 7.937462 | 7.842334 | | | |
| RTX 5090, 600 W board | 7.927601 | 7.832638 | | | 2.654687 |

## How these were measured

Rented single-GPU hosts on Vast.ai at their default power limit (hosts below it were refused), on
20 November 2026. The `ninfer` CLI, greedy, `--max-context 8192`, one run per configuration after a
warm-up run that loads the weights into the page cache. The 3,655-token prompt is the first 16,000
bytes of `eval/corpora/perplexity-1m/data/wikitext/00.txt` (3,502 tokens for Flint, 3,740 for
Flash-Next, 3,552 for DeepSeek); the 364-token prompt is the first 1,300 bytes of `01.txt`. The
compiled-loop runs set `NINFER_DEVICE_PROFILES` to a file with each card's built-in bands, every
`async` band of `gguf_matrix/*` and `gguf_moe_matrix/*` changed to the compiled loop. Flash-Next
reads its n-gram table rows from the file; DeepSeek runs with host experts, thinking off and no
Engram table.

| Card | Host CPU | PCIe |
|---|---|---|
| RTX PRO 6000 Blackwell Server Edition | Intel(R) Xeon(R) Platinum 8559C, 96 threads | 5.0 x16 |
| RTX 5090 | AMD Eng Sample: 100-000001247-12, 256 threads | 5.0 x16 |
| RTX 4090 | AMD EPYC 7402P 24-Core Processor | not recorded |
| RTX 5090, 600 W board | not recorded | not recorded |
| L40S | AMD Ryzen Threadripper 7960X 24-Cores, 48 threads | 4.0 x16 |
| RTX 3090 | Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz, 28 threads | 3.0 x16 |
| A100-SXM4-80GB | AMD EPYC 7742 64-Core Processor, 255 threads | 4.0 x16 |
| A100-SXM4-40GB | Intel(R) Xeon(R) CPU @ 2.20GHz, 12 threads | 4.0 x16 |
| RTX 5060 Ti | Intel(R) Xeon(R) CPU E5-2673 v4 @ 2.30GHz, 80 threads | 3.0 x16 |
