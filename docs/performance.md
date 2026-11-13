# Performance

Every measured result in one place, card by card: what fits, decode and prompt rates by model, KV
format, context depth and drafter, several requests at once, Qwen3.8-Flash-Next's expert
placements, and quality across cards. Which GPUs run NInfer at all is on [Supported GPUs](gpus.md).
The [measurement pages](#measurement-pages) keep each campaign's setup, analysis and extra tables.

Three methods appear below, and every table names its own:

- **Server**: `ninfer-serve` with one request at a time and
  [`tools/bench/refbench.py`](../tools/bench/README.md#reference-measurements-refbenchpy), greedy,
  thinking off. *Short chat* is the mean decode rate of five 512-token answers; a depth is a cold
  document of that many tokens followed by an answer of about 400 words; prompt time is the
  server's time to the first token.
- **Engine**: `ninfer_bench`, the Engine without HTTP. `pp512` is a 512-token prompt, `tg128` 128
  decoded tokens.
- **CLI**: one `ninfer` request; its summary gives the prompt and decode rates.

Numbers from different methods, hosts or builds are not read against each other as a benchmark.
Drafters are written as MTP or DFlash2 with their draft count; acceptance is accepted / drafted
drafts with the tokens committed per verify round in brackets.

## Cards

| Card | Memory | Build | Power limit | Measured on |
|---|---:|---|---|---|
| RTX PRO 6000 Blackwell Workstation Edition | 96 GB | `120a` | 600 W | [reference](performance/reference-2026-09.md), [Flash-Next](qwen3-8-flash-next.md#measurements), [DeepSeek](deepseek-v41.md#measurements) |
| RTX 5090 | 32 GB | `120a` | 575 W; 450 W for IQ3_S; 600 W boards for Flash-Next | [reference](performance/reference-2026-09.md), [Flash-Next](qwen3-8-flash-next.md#measurements), [GGUF](gguf.md#measurements), [Flint](flint.md#measurements), [DeepSeek](deepseek-v41.md#measurements) |
| RTX 5070 Ti | 16 GB | `120a` | 300 W | [field report](performance/field-reports.md#rtx-5070-ti-windows-11) |
| RTX 5060 Ti, two and four | 16 GB each | `120a` | | [field report](performance/field-reports.md#rtx-5060-ti-two-and-four-cards) |
| RTX 4090 | 24 GB | `89` | 450 W; 500 W boards for Flint | [reference](performance/reference-2026-09.md), [Flash-Next](qwen3-8-flash-next.md#measurements), [GGUF](gguf.md#measurements), [Flint](flint.md#measurements), [DeepSeek](deepseek-v41.md#measurements), [field reports](performance/field-reports.md) |
| L40S | 48 GB | | | [Flash-Next](qwen3-8-flash-next.md#measurements) |
| RTX 3090 Ti, two | 24 GB each | `86` | | [Flash-Next](qwen3-8-flash-next.md#measurements) |
| A40 | 48 GB | `86` | 300 W | [Flint](flint.md#measurements), [Edge0](edge0.md#measurements) |
| RTX 3090 | 24 GB | `86` | 300 to 420 W by campaign | every page |
| RTX 3060 | 12 GB | `86`, the image | 170 W | [small cards](performance/small-cards-2026-10.md) |
| RTX 3060 Ti | 8 GB | `86`, the image | 200 W | [small cards](performance/small-cards-2026-10.md) |
| TITAN RTX | 24 GB | `75` | 280 W | [TITAN RTX](performance/titan-rtx-2026-10.md) |
| RTX 2080 Ti | 11 GB | `75`, the image | 250 W | [small cards](performance/small-cards-2026-10.md) |
| Tesla V100-SXM2-16GB | 16 GB | `70` | 300 W | [Tesla V100](performance/tesla-v100-2026-10.md) |
| Tesla V100-SXM2-32GB | 32 GB | `70` | 300 W | [32 GB follow-up](performance/tesla-v100-2026-10.md#follow-up-on-the-32-gb-and-16-gb-cards-15-16-october) |

## Largest context

The largest `--max-context` (= `--kv-capacity`, one request) the server starts with, bisected in
4,096-token steps, with `--gdn-state-fp16` and `--rope-yarn` past 262,144; 1,048,576 is the
engine's ceiling, and the RTX PRO 6000 starts every configuration there with at least 50 GiB free.
On the 8 to 12 GB cards a step also had to answer one request. Each cell gives the window without
a drafter / with MTP 3 / with DFlash2 5:

| Model | KV | RTX PRO 6000 | RTX 5090 | RTX 4090 | RTX 3090 | Tesla V100 | RTX 3060 | RTX 2080 Ti | RTX 3060 Ti |
|---|---|---|---|---|---|---|---|---|---|
| Ternary Bonsai 2 27B | `rk8v4` | 1,048,576 / 1,048,576 / 1,048,576 | 978,944 / 901,120 / 901,120 | 659,456 / 602,112 / 581,632 | 663,552 / 606,208 / 585,728 | | | | |
| Ternary Bonsai 2 27B | `rk4v4` | 1,048,576 / 1,048,576 / 1,048,576 | 1,048,576 / 1,048,576 / 1,048,576 | 958,464 / 876,544 / 831,488 | 970,752 / 884,736 / 839,680 | 499,712 / 421,888 / 344,064 | 266,240 | 200,704 / 143,360 | 24,576 / does not fit |
| Ternary Bonsai 2 27B | `rk4v4-e8` | 1,048,576 | 1,048,576 | 958,464 | 970,752 | | | | |
| Ternary Bonsai 2 27B | `int8` | | | | | | | 106,496 | |
| Qwen3.8-27B | `rk8v4` | 1,048,576 / 1,048,576 / 1,048,576 | 598,016 / 544,768 / 491,520 | 278,528 / 241,664 / 167,936 | 286,720 / 249,856 / 176,128 | | | | |
| Qwen3.8-27B | `rk4v4` | 1,048,576 / 1,048,576 / 1,048,576 | 872,448 / 794,624 / 716,800 | 405,504 / 356,352 / 245,760 | 417,792 / 364,544 / 258,048 | | | | |
| Qwen3.8-27B | `rk4v4-e8` | 1,048,576 | 872,448 | 405,504 | 417,792 | | | | |
| Qwen3.8-27B IQ3_S | `rk4v4` | | | | | 241,664 / 176,128 / 32,768 | 8,192 / does not fit | | |
| Gemma 4 26B-A4B Q4_0 | `bf16` | | | | | 61,440 / 28,672 (MTP 2) | | | |

Gemma 4's windows at eight sequences on an RTX 3090 are in [its section](#gemma-4-26b-a4b). An RTX
5070 Ti that also drives a desktop started IQ3_S with `rk4v4-e8` at 122,880 tokens without a
drafter and 98,304 with MTP 3 ([field report](performance/field-reports.md#rtx-5070-ti-windows-11)).

Filled to that window without a drafter (one request: a document with three codes at 33, 66 and
90%, asked back in order):

| Card | Model | KV | Tokens | Codes found | Prompt time | Decode after it, tok/s |
|---|---|---|---:|---:|---:|---:|
| RTX PRO 6000 | Ternary Bonsai 2 27B | `rk4v4` | 1,048,576 | 2/3 | 910 s | 55.3 |
| RTX PRO 6000 | Qwen3.8-27B | `rk4v4` | 1,048,576 | 2/3 | 999 s | 42.9 |
| RTX 5090 | Ternary Bonsai 2 27B | `rk4v4` | 1,048,576 | 2/3 | 937 s | 56.2 |
| RTX 5090 | Ternary Bonsai 2 27B | `rk8v4` | 978,944 | 3/3 | 772 s | 46.9 |
| RTX 5090 | Qwen3.8-27B | `rk4v4` | 872,448 | 3/3 | 755 s | 47.6 |
| RTX 5090 | Qwen3.8-27B | `rk8v4` | 598,016 | 3/3 | 378 s | 49.1 |
| RTX 4090 | Ternary Bonsai 2 27B | `rk4v4` | 958,464 | 3/3 | 991 s | 30.9 |
| RTX 4090 | Ternary Bonsai 2 27B | `rk8v4` | 659,456 | 3/3 | 484 s | 32.9 |
| RTX 4090 | Qwen3.8-27B | `rk4v4` | 405,504 | 3/3 | 293 s | 35.4 |
| RTX 4090 | Qwen3.8-27B | `rk8v4` | 278,528 | 3/3 | 166 s | 36.8 |
| RTX 3090 | Ternary Bonsai 2 27B | `rk4v4` | 970,752 | 3/3 | 2,359 s | 17.7 |
| RTX 3090 | Ternary Bonsai 2 27B | `rk8v4` | 663,552 | 3/3 | 1,137 s | 24.4 |
| RTX 3090 | Qwen3.8-27B | `rk4v4` | 417,792 | 3/3 | 644 s | 23.8 |
| RTX 3090 | Qwen3.8-27B | `rk8v4` | 286,720 | 3/3 | 353 s | 31.0 |

At 1,048,576 tokens both cards miss the code at 90%, with `--rope-yarn` and with plain RoPE alike.
Up to the 262,144-token window every card and KV format found all three codes
([needles](performance/reference-2026-09.md)).

## Ternary Bonsai 2 27B

### Decode by depth

Server, one request, window 262,144 tokens; decode tok/s after a cold document of that many tokens.
VRAM is what the server holds when idle (the KV cache, workspaces and graphs are planned at
startup):

| Card | KV | Drafter | VRAM | short chat | 1,024 | 8,192 | 32,768 | 65,536 | 131,072 | 196,608 | 261,120 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RTX PRO 6000 | `rk4v4` | none | 12.6 GiB | 167.2 | 166.6 | 161.3 | 155.1 | 146.4 | 131.3 | 119.5 | 109.8 |
| RTX PRO 6000 | `rk4v4` | DFlash2 5 | 14.0 GiB | 360.9 | 401.3 | 317.4 | 284.2 | 242.7 | 197.9 | 179.4 | 193.0 |
| RTX PRO 6000 | `rk4v4` | MTP 3 | 13.3 GiB | 330.4 | 266.0 | 306.9 | 229.8 | 226.2 | 208.6 | 198.3 | 217.9 |
| RTX PRO 6000 | `rk4v4-e8` | none | 12.6 GiB | 167.3 | 166.9 | 161.8 | 155.2 | 146.6 | 131.3 | 119.7 | 110.0 |
| RTX PRO 6000 | `rk4v4-e8` | DFlash2 5 | 14.0 GiB | 367.4 | 285.1 | 307.0 | 319.9 | 222.8 | 203.7 | 180.8 | 189.7 |
| RTX PRO 6000 | `rk8v4` | none | 14.6 GiB | 167.5 | 166.8 | 162.2 | 151.5 | 139.8 | 121.2 | 107.3 | 96.4 |
| RTX PRO 6000 | `rk8v4` | DFlash2 5 | 16.0 GiB | 361.7 | 304.6 | 318.1 | 309.6 | 250.7 | 204.6 | 233.5 | 200.9 |
| RTX PRO 6000 | `rk8v4` | MTP 3 | 15.4 GiB | 314.7 | 313.0 | 293.3 | 258.1 | 229.7 | 191.4 | 192.0 | 210.5 |
| RTX 5090 | `rk4v4` | none | 12.4 GiB | 171.4 | 170.8 | 165.9 | 159.0 | 150.2 | 134.6 | 122.3 | 112.4 |
| RTX 5090 | `rk4v4` | DFlash2 5 | 13.8 GiB | 366.2 | 408.8 | 292.7 | 325.6 | 259.2 | 203.3 | 179.4 | 168.4 |
| RTX 5090 | `rk4v4` | MTP 3 | 13.1 GiB | 321.1 | 268.3 | 279.3 | 291.5 | 228.9 | 217.2 | 203.0 | 217.6 |
| RTX 5090 | `rk4v4-e8` | none | 12.4 GiB | 170.4 | 169.9 | 166.4 | 158.2 | 149.4 | 134.1 | 122.1 | 112.1 |
| RTX 5090 | `rk4v4-e8` | DFlash2 5 | 13.8 GiB | 387.9 | 315.1 | 332.1 | 301.1 | 241.6 | 219.2 | 219.9 | 197.8 |
| RTX 5090 | `rk8v4` | none | 14.4 GiB | 171.5 | 171.0 | 165.8 | 155.4 | 143.8 | 124.7 | 110.7 | 99.4 |
| RTX 5090 | `rk8v4` | DFlash2 5 | 15.8 GiB | 368.2 | 336.1 | 305.9 | 343.2 | 248.9 | 215.2 | 238.9 | 189.4 |
| RTX 5090 | `rk8v4` | MTP 3 | 15.2 GiB | 329.0 | 336.5 | 299.9 | 306.0 | 244.5 | 207.1 | 198.7 | 207.4 |
| RTX 4090 | `rk4v4` | none | 12.3 GiB | 107.9 | 107.6 | 105.1 | 99.3 | 93.3 | 83.0 | 75.1 | 63.8 |
| RTX 4090 | `rk4v4` | DFlash2 5 | 13.8 GiB | 242.8 | 179.5 | 181.8 | 236.3 | 156.9 | 134.9 | 140.4 | 115.3 |
| RTX 4090 | `rk4v4` | MTP 3 | 13.0 GiB | 206.9 | 158.6 | 179.9 | 180.7 | 142.6 | 134.7 | 131.6 | 116.3 |
| RTX 4090 | `rk4v4-e8` | none | 12.3 GiB | 107.9 | 107.6 | 105.3 | 99.3 | 93.3 | 83.0 | 75.1 | 63.8 |
| RTX 4090 | `rk4v4-e8` | DFlash2 5 | 13.8 GiB | 251.5 | 198.5 | 212.9 | 197.2 | 146.2 | 129.0 | 155.6 | 116.4 |
| RTX 4090 | `rk8v4` | none | 14.3 GiB | 108.4 | 107.8 | 105.2 | 97.8 | 89.9 | 77.3 | 67.9 | 56.4 |
| RTX 4090 | `rk8v4` | DFlash2 5 | 15.8 GiB | 244.8 | 243.6 | 206.7 | 180.0 | 160.9 | 125.8 | 118.7 | 122.9 |
| RTX 4090 | `rk8v4` | MTP 3 | 15.1 GiB | 203.4 | 191.4 | 184.0 | 164.8 | 153.7 | 127.7 | 121.8 | 111.0 |
| RTX 3090 | `rk4v4` | none | 12.2 GiB | 93.8 | 93.2 | 90.3 | 83.7 | 76.1 | 64.4 | 55.5 | 49.2 |
| RTX 3090 | `rk4v4` | DFlash2 5 | 13.7 GiB | 200.8 | 172.3 | 180.3 | 174.7 | 116.5 | 97.5 | 73.5 | 78.6 |
| RTX 3090 | `rk4v4` | MTP 3 | 12.9 GiB | 173.4 | 145.1 | 138.3 | 149.7 | 116.9 | 96.5 | 89.4 | 83.7 |
| RTX 3090 | `rk4v4-e8` | none | 12.2 GiB | 93.6 | 93.2 | 90.4 | 84.0 | 76.5 | 64.9 | 56.0 | 49.8 |
| RTX 3090 | `rk4v4-e8` | DFlash2 5 | 13.7 GiB | 202.8 | 157.2 | 173.3 | 140.0 | 119.4 | 109.2 | 113.9 | 76.3 |
| RTX 3090 | `rk8v4` | none | 14.2 GiB | 94.1 | 93.5 | 90.8 | 84.3 | 76.8 | 65.3 | 56.3 | 50.0 |
| RTX 3090 | `rk8v4` | DFlash2 5 | 15.7 GiB | 201.4 | 198.4 | 161.1 | 135.4 | 124.9 | 92.6 | 85.3 | 89.7 |
| RTX 3090 | `rk8v4` | MTP 3 | 15.0 GiB | 172.2 | 143.6 | 184.8 | 150.4 | 109.4 | 93.7 | 94.7 | 89.3 |
| Tesla V100 | `rk4v4` | none | 11.8 GiB | 73.2 | 73.0 | 71.8 | 67.2 | — | 54.6 | — | — |
| Tesla V100 | `rk4v4` | DFlash2 5 | 13.5 GiB | 116.1 | 100.9 | 88.0 | 91.1 | — | 45.0 | — | — |
| Tesla V100 | `rk4v4` | MTP 3 | 12.7 GiB | 123.1 | 100.1 | 118.7 | 87.4 | — | 60.0 | — | — |

### Prompt time by depth

Server, the same runs; seconds to the first token after a cold document of that many tokens:

| Card | KV | Drafter | 1,024 | 8,192 | 32,768 | 65,536 | 131,072 | 196,608 | 261,120 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| RTX PRO 6000 | `rk4v4` | none | 0.16 | 1.11 | 4.9 | 11.4 | 28.9 | 52.7 | 82.0 |
| RTX PRO 6000 | `rk4v4` | DFlash2 5 | 0.17 | 1.12 | 4.9 | 11.5 | 29.2 | 53.1 | 82.5 |
| RTX PRO 6000 | `rk4v4` | MTP 3 | 0.17 | 1.11 | 4.9 | 11.4 | 29.0 | 52.9 | 82.1 |
| RTX PRO 6000 | `rk4v4-e8` | none | 0.16 | 1.10 | 4.9 | 11.3 | 28.7 | 52.1 | 81.1 |
| RTX PRO 6000 | `rk4v4-e8` | DFlash2 5 | 0.17 | 1.12 | 4.9 | 11.4 | 28.9 | 52.5 | 81.6 |
| RTX PRO 6000 | `rk8v4` | none | 0.16 | 1.07 | 4.7 | 10.8 | 27.4 | 50.1 | 78.3 |
| RTX PRO 6000 | `rk8v4` | DFlash2 5 | 0.17 | 1.12 | 4.9 | 11.2 | 28.3 | 51.2 | 79.0 |
| RTX PRO 6000 | `rk8v4` | MTP 3 | 0.16 | 1.11 | 4.8 | 11.1 | 28.0 | 50.8 | 78.5 |
| RTX 5090 | `rk4v4` | none | 0.17 | 1.17 | 5.2 | 12.0 | 30.2 | 54.9 | 85.2 |
| RTX 5090 | `rk4v4` | DFlash2 5 | 0.18 | 1.20 | 5.3 | 12.1 | 30.6 | 55.4 | 85.9 |
| RTX 5090 | `rk4v4` | MTP 3 | 0.18 | 1.18 | 5.2 | 12.0 | 30.4 | 55.0 | 85.5 |
| RTX 5090 | `rk4v4-e8` | none | 0.17 | 1.17 | 5.1 | 11.9 | 30.1 | 54.3 | 84.2 |
| RTX 5090 | `rk4v4-e8` | DFlash2 5 | 0.18 | 1.19 | 5.2 | 12.1 | 30.4 | 54.9 | 85.0 |
| RTX 5090 | `rk8v4` | none | 0.17 | 1.14 | 5.0 | 11.5 | 29.0 | 52.7 | 81.9 |
| RTX 5090 | `rk8v4` | DFlash2 5 | 0.18 | 1.19 | 5.2 | 11.9 | 29.8 | 53.5 | 82.7 |
| RTX 5090 | `rk8v4` | MTP 3 | 0.18 | 1.18 | 5.1 | 11.8 | 29.6 | 53.2 | 82.2 |
| RTX 4090 | `rk4v4` | none | 0.21 | 1.44 | 6.4 | 14.7 | 37.2 | 67.6 | 105.2 |
| RTX 4090 | `rk4v4` | DFlash2 5 | 0.21 | 1.47 | 6.5 | 15.0 | 37.7 | 68.4 | 106.2 |
| RTX 4090 | `rk4v4` | MTP 3 | 0.21 | 1.45 | 6.4 | 14.8 | 37.4 | 67.9 | 105.5 |
| RTX 4090 | `rk4v4-e8` | none | 0.21 | 1.44 | 6.3 | 14.6 | 36.8 | 66.7 | 103.5 |
| RTX 4090 | `rk4v4-e8` | DFlash2 5 | 0.22 | 1.47 | 6.5 | 14.8 | 37.3 | 67.6 | 104.6 |
| RTX 4090 | `rk8v4` | none | 0.20 | 1.42 | 6.3 | 14.4 | 36.2 | 65.8 | 102.1 |
| RTX 4090 | `rk8v4` | DFlash2 5 | 0.22 | 1.47 | 6.4 | 14.7 | 36.9 | 66.8 | 103.3 |
| RTX 4090 | `rk8v4` | MTP 3 | 0.21 | 1.44 | 6.3 | 14.5 | 36.6 | 66.3 | 102.6 |
| RTX 3090 | `rk4v4` | none | 0.42 | 2.99 | 13.3 | 30.7 | 77.9 | 142.0 | 221.1 |
| RTX 3090 | `rk4v4` | DFlash2 5 | 0.43 | 3.06 | 13.5 | 31.2 | 79.2 | 143.8 | 223.7 |
| RTX 3090 | `rk4v4` | MTP 3 | 0.42 | 3.00 | 13.3 | 30.8 | 78.3 | 142.7 | 222.0 |
| RTX 3090 | `rk4v4-e8` | none | 0.42 | 2.98 | 13.2 | 30.4 | 77.1 | 140.2 | 218.2 |
| RTX 3090 | `rk4v4-e8` | DFlash2 5 | 0.43 | 3.06 | 13.5 | 31.0 | 78.3 | 141.8 | 220.5 |
| RTX 3090 | `rk8v4` | none | 0.42 | 2.97 | 13.1 | 30.2 | 76.2 | 138.3 | 214.5 |
| RTX 3090 | `rk8v4` | DFlash2 5 | 0.43 | 3.05 | 13.4 | 30.8 | 77.4 | 139.8 | 216.9 |
| RTX 3090 | `rk8v4` | MTP 3 | 0.42 | 2.99 | 13.2 | 30.3 | 76.5 | 138.8 | 215.4 |
| Tesla V100 | `rk4v4` | none | 1.09 | 8.2 | 38.6 | — | 261.2 | — | — |
| Tesla V100 | `rk4v4` | DFlash2 5 | 1.13 | 8.4 | 40.2 | — | 265.9 | — | — |
| Tesla V100 | `rk4v4` | MTP 3 | 1.10 | 8.3 | 39.3 | — | 262.8 | — | — |

### Draft acceptance

| Card | KV | Drafter | short chat | 1,024 | 8,192 | 32,768 | 65,536 | 131,072 | 196,608 | 261,120 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RTX PRO 6000 | `rk4v4` | DFlash2 5 | 0.39 (2.91) | 0.46 (3.26) | 0.33 (2.62) | 0.31 (2.54) | 0.26 (2.31) | 0.23 (2.14) | 0.24 (2.17) | 0.32 (2.57) |
| RTX PRO 6000 | `rk4v4` | MTP 3 | 0.54 (2.61) | 0.38 (2.13) | 0.52 (2.60) | 0.33 (1.99) | 0.36 (2.06) | 0.38 (2.14) | 0.41 (2.25) | 0.56 (2.67) |
| RTX PRO 6000 | `rk4v4-e8` | DFlash2 5 | 0.40 (2.96) | 0.26 (2.31) | 0.31 (2.53) | 0.37 (2.84) | 0.23 (2.12) | 0.24 (2.19) | 0.24 (2.18) | 0.30 (2.51) |
| RTX PRO 6000 | `rk8v4` | DFlash2 5 | 0.39 (2.91) | 0.30 (2.47) | 0.33 (2.65) | 0.35 (2.75) | 0.28 (2.38) | 0.24 (2.19) | 0.36 (2.81) | 0.33 (2.63) |
| RTX PRO 6000 | `rk8v4` | MTP 3 | 0.50 (2.48) | 0.50 (2.48) | 0.48 (2.46) | 0.42 (2.24) | 0.39 (2.15) | 0.35 (2.05) | 0.44 (2.30) | 0.60 (2.78) |
| RTX 5090 | `rk4v4` | DFlash2 5 | 0.39 (2.92) | 0.46 (3.28) | 0.28 (2.40) | 0.38 (2.89) | 0.29 (2.46) | 0.24 (2.20) | 0.24 (2.19) | 0.26 (2.28) |
| RTX 5090 | `rk4v4` | MTP 3 | 0.50 (2.49) | 0.37 (2.12) | 0.43 (2.29) | 0.50 (2.48) | 0.36 (2.06) | 0.40 (2.19) | 0.43 (2.27) | 0.56 (2.67) |
| RTX 5090 | `rk4v4-e8` | DFlash2 5 | 0.43 (3.10) | 0.31 (2.53) | 0.35 (2.73) | 0.34 (2.67) | 0.26 (2.29) | 0.28 (2.38) | 0.34 (2.67) | 0.34 (2.67) |
| RTX 5090 | `rk8v4` | DFlash2 5 | 0.39 (2.93) | 0.34 (2.69) | 0.31 (2.54) | 0.41 (3.02) | 0.27 (2.35) | 0.26 (2.27) | 0.36 (2.81) | 0.29 (2.46) |
| RTX 5090 | `rk8v4` | MTP 3 | 0.52 (2.55) | 0.55 (2.63) | 0.47 (2.40) | 0.55 (2.65) | 0.43 (2.29) | 0.40 (2.21) | 0.45 (2.36) | 0.57 (2.71) |
| RTX 4090 | `rk4v4` | DFlash2 5 | 0.39 (2.91) | 0.24 (2.17) | 0.25 (2.26) | 0.43 (3.11) | 0.25 (2.23) | 0.25 (2.22) | 0.32 (2.60) | 0.30 (2.51) |
| RTX 4090 | `rk4v4` | MTP 3 | 0.53 (2.57) | 0.34 (2.01) | 0.44 (2.31) | 0.49 (2.49) | 0.36 (2.09) | 0.41 (2.21) | 0.47 (2.40) | 0.53 (2.59) |
| RTX 4090 | `rk4v4-e8` | DFlash2 5 | 0.41 (3.01) | 0.28 (2.40) | 0.33 (2.63) | 0.32 (2.58) | 0.21 (2.07) | 0.23 (2.12) | 0.37 (2.85) | 0.30 (2.51) |
| RTX 4090 | `rk8v4` | DFlash2 5 | 0.39 (2.93) | 0.39 (2.94) | 0.31 (2.56) | 0.27 (2.36) | 0.26 (2.27) | 0.21 (2.06) | 0.24 (2.19) | 0.33 (2.63) |
| RTX 4090 | `rk8v4` | MTP 3 | 0.51 (2.51) | 0.46 (2.38) | 0.45 (2.34) | 0.42 (2.24) | 0.43 (2.28) | 0.39 (2.18) | 0.44 (2.32) | 0.55 (2.64) |
| RTX 3090 | `rk4v4` | DFlash2 5 | 0.39 (2.91) | 0.31 (2.52) | 0.36 (2.78) | 0.40 (3.03) | 0.25 (2.26) | 0.27 (2.34) | 0.23 (2.13) | 0.33 (2.63) |
| RTX 3090 | `rk4v4` | MTP 3 | 0.52 (2.54) | 0.39 (2.17) | 0.38 (2.15) | 0.51 (2.53) | 0.41 (2.22) | 0.41 (2.23) | 0.50 (2.47) | 0.56 (2.65) |
| RTX 3090 | `rk4v4-e8` | DFlash2 5 | 0.39 (2.94) | 0.26 (2.30) | 0.34 (2.67) | 0.27 (2.34) | 0.25 (2.22) | 0.29 (2.44) | 0.41 (3.07) | 0.27 (2.33) |
| RTX 3090 | `rk8v4` | DFlash2 5 | 0.39 (2.92) | 0.39 (2.91) | 0.30 (2.46) | 0.26 (2.27) | 0.27 (2.35) | 0.22 (2.12) | 0.27 (2.33) | 0.36 (2.80) |
| RTX 3090 | `rk8v4` | MTP 3 | 0.51 (2.51) | 0.37 (2.12) | 0.62 (2.90) | 0.50 (2.49) | 0.34 (2.04) | 0.37 (2.10) | 0.50 (2.51) | 0.55 (2.65) |
| Tesla V100 | `rk4v4` | DFlash2 5 | 0.39 (2.91) | — | 0.28 (2.36) | — | — | 0.24 (2.17) | — | — |
| Tesla V100 | `rk4v4` | MTP 3 | 0.49 (2.45) | — | 0.52 (2.53) | — | — | 0.39 (2.17) | — | — |

### Best draft count

Server, `rk8v4`, one request; `--draft-tokens` from 1 to 15 at short chat and after 32,768 and
131,072 tokens. The best count with its decode tok/s and acceptance, one run each (acceptance
moves with the answer, so neighbouring counts within a few percent are ties;
[full sweeps](performance/reference-2026-09.md#draft-length)):

| Card | No drafter: short chat / 32,768 / 131,072 | Drafter | Short chat | 32,768 | 131,072 |
|---|---|---|---|---|---|
| RTX 5090 | 171.6 / 155.2 / 124.5 | MTP | 5: 321.5 (0.38) | 3: 295.6 (0.52) | 3: 207.3 (0.40) |
| RTX 5090 | 171.6 / 155.2 / 124.5 | DFlash2 | 7: 396.7 (0.33) | 4: 338.9 (0.48) | 6: 216.8 (0.23) |
| RTX 4090 | 108.3 / 97.9 / 77.3 | MTP | 3: 201.2 (0.50) | 3: 175.2 (0.47) | 3: 130.9 (0.41) |
| RTX 4090 | 108.3 / 97.9 / 77.3 | DFlash2 | 7: 256.4 (0.31) | 4: 220.4 (0.47) | 6: 144.9 (0.24) |
| RTX 3090 | 92.5 / 82.5 / 63.4 | MTP | 3: 171.4 (0.52) | 3: 128.6 (0.40) | 1: 96.9 (0.73) |
| RTX 3090 | 92.5 / 82.5 / 63.4 | DFlash2 | 7: 202.0 (0.31) | 5: 176.2 (0.41) | 3: 97.2 (0.36) |

### Engine and CLI rates

Prompt and decode in tok/s:

| Card | Method, KV | Prompt 512 | Prompt 2,048 | Long prompt | Decode | Decode after the long prompt |
|---|---|---:|---:|---:|---:|---:|
| RTX 3090 | engine, `rk8v4` | 3,026 | 3,108 | 2,292 (65,536) | 93.9 | 93.7 |
| TITAN RTX | engine, `rk8v4` | 1,304 | 1,375 | 951 (65,536) | 63.4 | 61.8 |
| Tesla V100 | engine, `bf16` | 984 | 1,072 | 929 (16,384) | 73.6 | 66.7 |
| RTX 3060 Ti | CLI, `int8` | | | 1,160 (2,273) | 50.1 | |
| RTX 2080 Ti | CLI, `int8` | | | 1,080 (2,273) | 58.5 | |
| RTX 3060 | CLI, `int8` | | | 937 (2,273) | 40.5 | |

Engine rows decode 128 tokens from the start, CLI rows 256. With a drafter, decode tok/s
(acceptance):

| Card | Method | MTP 3 | DFlash2 5 |
|---|---|---:|---:|
| RTX 3090 | engine, 512-token prompt, 128 tokens | 174.1 (0.49, 2.46) | 149.1 (0.23, 2.15) |
| TITAN RTX | engine, 512-token prompt, 128 tokens | 106.8 (0.47, 2.40) | 92.1 (0.24, 2.19) |
| RTX 2080 Ti | CLI, 2,273-token prompt, 256 tokens | 95.9 (2.33) | |
| RTX 3060 | CLI, 2,273-token prompt, 256 tokens | 69.2 (2.31) | |

### KV formats

Engine, a 16,384-token prompt, then 64 decoded tokens, one run each; prompt / decode tok/s:

| KV | Bytes per token | Tesla V100 | TITAN RTX | RTX 3090 |
|---|---:|---:|---:|---:|
| `bf16` | 65,536 | 920 / 66.8 | 1,065 / 62.8 | 2,687 / 92.0 |
| `int8` | 33,792 | 947 / 69.3 | 1,199 / 62.6 | |
| `fp8` | 33,024 | 914 / 67.8 | 601 / 59.7 | |
| `rk8v4` | 26,112 | 934 / 69.0 | 1,251 / 62.1 | 2,875 / 93.9 |
| `k8v4` | 25,728 | 907 / 66.6 | 634 / 59.6 | |
| `nvfp4` | 18,432 | 903 / 66.1 | 628 / 62.0 | |
| `rk4v4` | 17,920 | 937 / 68.5 | 1,234 / 62.0 | |
| `rk4v4-e8` | 17,920 | 935 / 68.7 | 1,231 / 61.6 | |
| `rk2v4-e8` | 13,824 | 926 / 67.1 | 1,217 / 61.6 | |

After a 65,536-token prompt the TITAN RTX prompts at 951 (`rk8v4`), 826 (`int8`) and 607 (`bf16`)
tok/s and decodes at 61.8, 62.8 and 62.4. On Turing the FP8, NVFP4 and K8V4 prompt kernels run at
about half the rate of the INT8 family ([TITAN RTX](performance/titan-rtx-2026-10.md#kv-formats)).
On the V100 `--prefill-chunk 4096` raises `rk4v4` prompts from 965 to 1,033 tok/s at 4,096 tokens
and from 860 to 914 at 16,384, for 628 MiB of workspace instead of 229
([prompt chunks](performance/tesla-v100-2026-10.md#prompt-chunks)).

### Several requests at once

Server, total generated tok/s when N requests arrive together: the short-chat prompts in turn, 512
tokens each, `rk8v4`, `--max-concurrency 8 --max-context 16384 --kv-capacity 131072`. An empty
cell at 8: eight lanes of 16,384 tokens do not fit beside that drafter, and the row ran with four.

| Card | Drafter | 1 | 2 | 4 | 8 |
|---|---|---:|---:|---:|---:|
| RTX PRO 6000 | none | 165.5 | 321.4 | 599.1 | 1,043.5 |
| RTX PRO 6000 | MTP 3 | 274.4 | 516.0 | 805.0 | 1,155.3 |
| RTX PRO 6000 | DFlash2 5 | 270.6 | 523.9 | 735.5 | 890.4 |
| RTX 5090 | none | 169.8 | 327.4 | 608.6 | 1,053.3 |
| RTX 5090 | MTP 3 | 263.3 | 511.3 | 771.1 | 1,063.0 |
| RTX 5090 | DFlash2 5 | 278.1 | 511.7 | 631.9 | 799.1 |
| RTX 4090 | none | 107.4 | 206.9 | 383.5 | 670.6 |
| RTX 4090 | MTP 3 | 182.1 | 318.3 | 533.8 | 824.1 |
| RTX 4090 | DFlash2 5 | 197.1 | 354.2 | 504.9 | 673.5 |
| RTX 3090 | none | 93.1 | 176.9 | 324.0 | 546.4 |
| RTX 3090 | MTP 3 | 141.7 | 267.3 | 413.7 | 550.8 |
| RTX 3090 | DFlash2 5 | 154.7 | 256.0 | 334.1 | 386.2 |
| Tesla V100 | none | 71.6 | 129.1 | 214.4 | 283.5 |
| Tesla V100 | MTP 3 | 101.5 | 157.2 | 182.6 | 204.9 |
| Tesla V100 | DFlash2 5 | 89.4 | 104.7 | 116.4 | |

The same with 256-token answers, `--max-context 8192 --kv-capacity 65536`, one run each:

| Card | Drafter | 1 | 2 | 4 | 8 |
|---|---|---:|---:|---:|---:|
| TITAN RTX | none | 54.8 | 119.3 | 211.9 | 327.1 |
| TITAN RTX | MTP 3 | 94.9 | 161.1 | 247.5 | 308.0 |
| RTX 3090 | none | 73.2 | 172.2 | 308.2 | 498.0 |
| RTX 3090 | MTP 3 | 133.3 | 255.6 | 404.6 | 521.3 |

### Quality

`ninfer-perplexity --quick` on `ninfer-ppl-1m-v1`:

| Card | `rk8v4` | `bf16` | `rk2v4-e8` |
|---|---:|---:|---:|
| RTX 3090 | 5.631 | | |
| Tesla V100 | 5.6305 | | |
| TITAN RTX | 5.6314 | 5.6272 | 5.8189 |

Against `rk8v4`'s 5.631 the reference measurement gives 5.650 for `rk4v4-e8` and 5.820 for
`rk2v4-e8`.

## Qwen3.8-27B

The official [`groupwise-int` artifact](https://huggingface.co/neroued/Qwen3.8-27B-NInfer).

### Decode by depth

Server, one request. On the 24 GB cards each KV format runs at the largest window DFlash2 fits
beside it, for every drafter; `rk4v4-e8` without a drafter runs the full window. A dash: past the
window.

| Card | KV | Drafter | Window | VRAM | short chat | 1,024 | 8,192 | 32,768 | 65,536 | 131,072 | 196,608 | 261,120 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RTX PRO 6000 | `rk4v4` | none | 262,144 | 21.8 GiB | 89.6 | 89.5 | 88.0 | 85.9 | 83.2 | 78.0 | 73.7 | 69.9 |
| RTX PRO 6000 | `rk4v4` | DFlash2 5 | 262,144 | 24.0 GiB | 219.9 | 189.4 | 179.6 | 183.7 | 154.0 | 142.4 | 166.4 | 116.1 |
| RTX PRO 6000 | `rk4v4` | MTP 3 | 262,144 | 22.5 GiB | 186.4 | 164.5 | 169.6 | 149.0 | 161.0 | 143.4 | 142.9 | 128.7 |
| RTX PRO 6000 | `rk4v4-e8` | none | 262,144 | 21.8 GiB | 89.7 | 89.5 | 88.0 | 85.9 | 83.2 | 78.1 | 73.8 | 70.0 |
| RTX PRO 6000 | `rk4v4-e8` | DFlash2 5 | 262,144 | 24.0 GiB | 222.7 | 184.6 | 171.7 | 214.9 | 148.5 | 142.2 | 150.3 | 121.6 |
| RTX PRO 6000 | `rk8v4` | none | 262,144 | 23.8 GiB | 89.6 | 89.4 | 88.0 | 84.9 | 81.1 | 74.5 | 69.1 | 64.4 |
| RTX PRO 6000 | `rk8v4` | DFlash2 5 | 262,144 | 26.0 GiB | 219.1 | 195.9 | 186.0 | 217.2 | 156.9 | 160.2 | 164.5 | 120.9 |
| RTX PRO 6000 | `rk8v4` | MTP 3 | 262,144 | 24.6 GiB | 187.5 | 154.3 | 172.4 | 176.4 | 164.2 | 135.8 | 129.6 | 120.4 |
| RTX 5090 | `rk4v4` | none | 262,144 | 21.6 GiB | 91.9 | 91.7 | 90.2 | 88.1 | 85.3 | 80.0 | 75.4 | 71.6 |
| RTX 5090 | `rk4v4` | DFlash2 5 | 262,144 | 23.8 GiB | 214.9 | 193.9 | 197.2 | 204.7 | 151.8 | 129.7 | 136.0 | 116.1 |
| RTX 5090 | `rk4v4` | MTP 3 | 262,144 | 22.3 GiB | 182.7 | 153.4 | 158.9 | 171.8 | 165.7 | 140.8 | 132.6 | 129.0 |
| RTX 5090 | `rk4v4-e8` | none | 262,144 | 21.6 GiB | 91.5 | 91.3 | 90.2 | 87.8 | 84.9 | 79.7 | 75.2 | 71.3 |
| RTX 5090 | `rk4v4-e8` | DFlash2 5 | 262,144 | 23.8 GiB | 220.3 | 165.0 | 176.0 | 202.2 | 147.0 | 147.9 | 128.7 | 128.8 |
| RTX 5090 | `rk8v4` | none | 262,144 | 23.6 GiB | 92.0 | 91.7 | 90.2 | 87.1 | 83.3 | 76.5 | 71.0 | 66.2 |
| RTX 5090 | `rk8v4` | DFlash2 5 | 262,144 | 25.8 GiB | 221.0 | 186.7 | 204.6 | 205.9 | 157.8 | 152.7 | 134.0 | 129.8 |
| RTX 5090 | `rk8v4` | MTP 3 | 262,144 | 24.4 GiB | 184.7 | 153.7 | 169.8 | 180.1 | 140.5 | 130.3 | 130.5 | 114.0 |
| RTX 4090 | `rk4v4` | none | 245,760 | 21.3 GiB | 54.7 | 54.6 | 53.9 | 52.4 | 50.7 | 47.5 | 44.8 | — |
| RTX 4090 | `rk4v4` | DFlash2 5 | 245,760 | 23.5 GiB | 137.8 | 108.4 | 115.3 | 142.7 | 93.8 | 86.4 | 80.4 | — |
| RTX 4090 | `rk4v4` | MTP 3 | 245,760 | 21.9 GiB | 108.9 | 93.5 | 102.0 | 105.3 | 92.8 | 85.0 | 83.3 | — |
| RTX 4090 | `rk4v4-e8` | none | 262,144 | 21.5 GiB | 54.7 | 54.6 | 54.0 | 52.4 | 50.7 | 47.5 | 44.8 | 40.6 |
| RTX 4090 | `rk8v4` | none | 167,936 | 21.2 GiB | 54.8 | 54.7 | 54.0 | 52.0 | 49.7 | 45.6 | — | — |
| RTX 4090 | `rk8v4` | DFlash2 5 | 167,936 | 23.5 GiB | 138.1 | 127.1 | 129.8 | 140.3 | 102.2 | 90.9 | — | — |
| RTX 4090 | `rk8v4` | MTP 3 | 167,936 | 21.9 GiB | 109.0 | 102.6 | 101.7 | 108.5 | 94.9 | 77.3 | — | — |
| RTX 3090 | `rk4v4` | none | 258,048 | 21.3 GiB | 48.0 | 47.9 | 47.1 | 45.0 | 42.3 | 38.0 | 34.4 | — |
| RTX 3090 | `rk4v4` | DFlash2 5 | 258,048 | 23.6 GiB | 115.9 | 88.5 | 86.1 | 93.2 | 70.6 | 60.5 | 54.1 | — |
| RTX 3090 | `rk4v4` | MTP 3 | 258,048 | 22.0 GiB | 95.9 | 79.0 | 88.1 | 89.4 | 77.5 | 66.3 | 61.5 | — |
| RTX 3090 | `rk4v4-e8` | none | 262,144 | 21.4 GiB | 48.0 | 47.9 | 47.1 | 45.2 | 42.5 | 38.3 | 34.7 | 32.0 |
| RTX 3090 | `rk8v4` | none | 176,128 | 21.3 GiB | 48.2 | 48.0 | 47.2 | 45.1 | 42.6 | 38.4 | — | — |
| RTX 3090 | `rk8v4` | DFlash2 5 | 176,128 | 23.5 GiB | 117.1 | 101.4 | 101.3 | 110.8 | 78.7 | 72.6 | — | — |
| RTX 3090 | `rk8v4` | MTP 3 | 176,128 | 22.0 GiB | 97.2 | 90.6 | 90.5 | 83.6 | 83.9 | 60.9 | — | — |

### Prompt time by depth

| Card | KV | Drafter | 1,024 | 8,192 | 32,768 | 65,536 | 131,072 | 196,608 | 261,120 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| RTX PRO 6000 | `rk4v4` | none | 0.30 | 1.87 | 7.9 | 17.2 | 40.5 | 70.0 | 104.9 |
| RTX PRO 6000 | `rk4v4` | DFlash2 5 | 0.31 | 1.88 | 7.9 | 17.3 | 40.9 | 70.5 | 105.6 |
| RTX PRO 6000 | `rk4v4` | MTP 3 | 0.31 | 1.87 | 7.8 | 17.2 | 40.6 | 70.1 | 105.0 |
| RTX PRO 6000 | `rk4v4-e8` | none | 0.30 | 1.87 | 7.8 | 17.1 | 40.3 | 69.4 | 104.0 |
| RTX PRO 6000 | `rk4v4-e8` | DFlash2 5 | 0.31 | 1.88 | 7.9 | 17.2 | 40.6 | 69.9 | 104.5 |
| RTX PRO 6000 | `rk8v4` | none | 0.30 | 1.86 | 7.7 | 16.9 | 39.4 | 67.7 | 100.8 |
| RTX PRO 6000 | `rk8v4` | DFlash2 5 | 0.31 | 1.87 | 7.8 | 17.0 | 39.7 | 68.3 | 101.7 |
| RTX PRO 6000 | `rk8v4` | MTP 3 | 0.30 | 1.86 | 7.8 | 16.9 | 39.6 | 68.1 | 101.5 |
| RTX 5090 | `rk4v4` | none | 0.34 | 2.04 | 8.5 | 18.4 | 43.0 | 73.8 | 110.4 |
| RTX 5090 | `rk4v4` | DFlash2 5 | 0.34 | 2.06 | 8.6 | 18.6 | 43.4 | 74.4 | 111.6 |
| RTX 5090 | `rk4v4` | MTP 3 | 0.34 | 2.07 | 8.6 | 18.7 | 43.6 | 74.8 | 111.7 |
| RTX 5090 | `rk4v4-e8` | none | 0.34 | 2.05 | 8.5 | 18.5 | 43.0 | 73.7 | 109.8 |
| RTX 5090 | `rk4v4-e8` | DFlash2 5 | 0.34 | 2.07 | 8.6 | 18.7 | 43.4 | 74.1 | 110.4 |
| RTX 5090 | `rk8v4` | none | 0.33 | 2.03 | 8.4 | 18.2 | 42.1 | 71.9 | 107.0 |
| RTX 5090 | `rk8v4` | DFlash2 5 | 0.34 | 2.05 | 8.5 | 18.3 | 42.4 | 72.4 | 107.7 |
| RTX 5090 | `rk8v4` | MTP 3 | 0.33 | 2.03 | 8.4 | 18.2 | 42.2 | 72.0 | 107.1 |
| RTX 4090 | `rk4v4` | none | 0.56 | 3.24 | 13.1 | 27.8 | 63.0 | 106.0 | — |
| RTX 4090 | `rk4v4` | DFlash2 5 | 0.57 | 3.27 | 13.2 | 28.1 | 63.5 | 106.7 | — |
| RTX 4090 | `rk4v4` | MTP 3 | 0.57 | 3.25 | 13.1 | 27.9 | 63.2 | 106.2 | — |
| RTX 4090 | `rk4v4-e8` | none | 0.56 | 3.23 | 13.1 | 27.7 | 62.6 | 105.0 | 154.1 |
| RTX 4090 | `rk8v4` | none | 0.56 | 3.23 | 13.0 | 27.6 | 62.2 | — | — |
| RTX 4090 | `rk8v4` | DFlash2 5 | 0.57 | 3.26 | 13.2 | 27.8 | 62.7 | — | — |
| RTX 4090 | `rk8v4` | MTP 3 | 0.57 | 3.24 | 13.1 | 27.6 | 62.3 | — | — |
| RTX 3090 | `rk4v4` | none | 1.15 | 5.63 | 23.2 | 50.2 | 118.3 | 203.4 | — |
| RTX 3090 | `rk4v4` | DFlash2 5 | 1.17 | 5.73 | 23.6 | 51.0 | 119.4 | 205.2 | — |
| RTX 3090 | `rk4v4` | MTP 3 | 1.17 | 5.68 | 23.3 | 50.3 | 118.6 | 203.9 | — |
| RTX 3090 | `rk4v4-e8` | none | 1.15 | 5.62 | 23.1 | 50.0 | 117.3 | 201.4 | 300.9 |
| RTX 3090 | `rk8v4` | none | 1.14 | 5.61 | 23.1 | 49.9 | 116.4 | — | — |
| RTX 3090 | `rk8v4` | DFlash2 5 | 1.17 | 5.73 | 23.4 | 50.6 | 117.7 | — | — |
| RTX 3090 | `rk8v4` | MTP 3 | 1.16 | 5.65 | 23.1 | 49.9 | 117.0 | — | — |

### Draft acceptance

| Card | KV | Drafter | short chat | 1,024 | 8,192 | 32,768 | 65,536 | 131,072 | 196,608 | 261,120 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RTX PRO 6000 | `rk4v4` | DFlash2 5 | 0.44 (3.18) | 0.36 (2.75) | 0.33 (2.66) | 0.37 (2.85) | 0.30 (2.47) | 0.30 (2.48) | 0.42 (3.11) | 0.26 (2.31) |
| RTX PRO 6000 | `rk4v4` | MTP 3 | 0.60 (2.78) | 0.49 (2.46) | 0.54 (2.62) | 0.45 (2.34) | 0.54 (2.62) | 0.50 (2.49) | 0.54 (2.61) | 0.50 (2.48) |
| RTX PRO 6000 | `rk4v4-e8` | DFlash2 5 | 0.45 (3.21) | 0.34 (2.70) | 0.31 (2.54) | 0.47 (3.33) | 0.28 (2.38) | 0.29 (2.47) | 0.36 (2.78) | 0.28 (2.40) |
| RTX PRO 6000 | `rk8v4` | DFlash2 5 | 0.44 (3.16) | 0.37 (2.84) | 0.35 (2.77) | 0.47 (3.37) | 0.30 (2.51) | 0.35 (2.75) | 0.41 (3.04) | 0.28 (2.38) |
| RTX PRO 6000 | `rk8v4` | MTP 3 | 0.60 (2.80) | 0.44 (2.31) | 0.54 (2.62) | 0.60 (2.85) | 0.57 (2.74) | 0.48 (2.45) | 0.49 (2.46) | 0.48 (2.44) |
| RTX 5090 | `rk4v4` | DFlash2 5 | 0.42 (3.10) | 0.37 (2.81) | 0.38 (2.91) | 0.43 (3.17) | 0.29 (2.44) | 0.25 (2.26) | 0.31 (2.55) | 0.27 (2.34) |
| RTX 5090 | `rk4v4` | MTP 3 | 0.58 (2.71) | 0.44 (2.31) | 0.48 (2.43) | 0.57 (2.69) | 0.57 (2.68) | 0.48 (2.44) | 0.48 (2.44) | 0.51 (2.51) |
| RTX 5090 | `rk4v4-e8` | DFlash2 5 | 0.44 (3.19) | 0.28 (2.40) | 0.32 (2.61) | 0.43 (3.15) | 0.28 (2.37) | 0.32 (2.58) | 0.29 (2.42) | 0.32 (2.58) |
| RTX 5090 | `rk8v4` | DFlash2 5 | 0.44 (3.19) | 0.34 (2.71) | 0.41 (3.05) | 0.43 (3.15) | 0.31 (2.53) | 0.33 (2.63) | 0.30 (2.46) | 0.31 (2.56) |
| RTX 5090 | `rk8v4` | MTP 3 | 0.59 (2.75) | 0.44 (2.32) | 0.53 (2.57) | 0.61 (2.81) | 0.44 (2.30) | 0.44 (2.30) | 0.49 (2.46) | 0.44 (2.32) |
| RTX 4090 | `rk4v4` | DFlash2 5 | 0.44 (3.15) | 0.30 (2.51) | 0.34 (2.71) | 0.49 (3.47) | 0.27 (2.36) | 0.28 (2.37) | 0.28 (2.37) | — |
| RTX 4090 | `rk4v4` | MTP 3 | 0.57 (2.69) | 0.45 (2.36) | 0.53 (2.62) | 0.59 (2.79) | 0.50 (2.48) | 0.48 (2.45) | 0.52 (2.56) | — |
| RTX 4090 | `rk8v4` | DFlash2 5 | 0.44 (3.16) | 0.39 (2.95) | 0.41 (3.05) | 0.48 (3.40) | 0.32 (2.56) | 0.30 (2.48) | — | — |
| RTX 4090 | `rk8v4` | MTP 3 | 0.57 (2.68) | 0.52 (2.57) | 0.52 (2.55) | 0.61 (2.80) | 0.53 (2.56) | 0.42 (2.29) | — | — |
| RTX 3090 | `rk4v4` | DFlash2 5 | 0.45 (3.24) | 0.30 (2.51) | 0.30 (2.51) | 0.38 (2.86) | 0.27 (2.35) | 0.26 (2.31) | 0.27 (2.35) | — |
| RTX 3090 | `rk4v4` | MTP 3 | 0.58 (2.73) | 0.42 (2.26) | 0.53 (2.60) | 0.59 (2.75) | 0.53 (2.57) | 0.50 (2.50) | 0.54 (2.61) | — |
| RTX 3090 | `rk8v4` | DFlash2 5 | 0.46 (3.27) | 0.38 (2.86) | 0.39 (2.92) | 0.48 (3.39) | 0.32 (2.57) | 0.34 (2.68) | — | — |
| RTX 3090 | `rk8v4` | MTP 3 | 0.59 (2.76) | 0.53 (2.59) | 0.55 (2.65) | 0.52 (2.55) | 0.58 (2.74) | 0.42 (2.25) | — | — |

### Best draft count

The same sweep as for Bonsai 2; 262,144-token window on the RTX 5090, 139,264 on the 24 GB cards:

| Card | No drafter: short chat / 32,768 / 131,072 | Drafter | Short chat | 32,768 | 131,072 |
|---|---|---|---|---|---|
| RTX 5090 | 91.6 / 86.7 / 76.3 | MTP | 3: 184.2 (0.59) | 5: 183.3 (0.50) | 4: 134.2 (0.40) |
| RTX 5090 | 91.6 / 86.7 / 76.3 | DFlash2 | 7: 236.2 (0.36) | 6: 226.0 (0.43) | 4: 164.0 (0.44) |
| RTX 4090 | 54.8 / 52.0 / 45.6 | MTP | 4: 113.5 (0.52) | 3: 114.7 (0.66) | 3: 80.3 (0.45) |
| RTX 4090 | 54.8 / 52.0 / 45.6 | DFlash2 | 7: 148.8 (0.35) | 7: 151.9 (0.39) | 7: 102.3 (0.27) |
| RTX 3090 | 47.9 / 44.8 / 38.1 | MTP | 3: 95.9 (0.58) | 3: 93.2 (0.63) | 3: 68.5 (0.51) |
| RTX 3090 | 47.9 / 44.8 / 38.1 | DFlash2 | 7: 117.5 (0.37) | 6: 109.3 (0.41) | 4: 79.0 (0.45) |

### Engine and CLI rates

Engine, `rk8v4`, the mean of three runs after one warm-up; tok/s:

| Card | Prompt 512 | Prompt 2,048 | Decode 128 | MTP 3, 512-token prompt |
|---|---:|---:|---:|---:|
| RTX 3090 | 1,807 | 1,771 | 48.8 | 78.6 (0.32, 1.95) |
| TITAN RTX | 775 | 793 | 29.8 | 35.3 (0.34, 2.02) |

On the Tesla V100 the weights fit only with `--embedding-q4 --lm-head-q4` (14.7 GiB) or
`--lm-head-q6` (15.0 GiB); the CLI then decodes 30.4 tok/s.

### KV formats on Qwen3.8-27B

One RTX 3090. Perplexity is `ninfer-perplexity --quick` on `ninfer-ppl-1m-v1` (context 4,096, stride 2,048,
261,167 scored tokens). Decode is 128 steps after a prompt of that depth without a drafter, the
mean of two runs, on a card capped at 315 W of its 350 W default:

| KV format | Bytes per token | Perplexity | Change against `bf16` | Decode at 4,096 | at 16,384 | at 32,768 |
|---|---:|---:|---:|---:|---:|---:|
| `bf16` | 65,536 | 4.342517 | — | 36.58 | 34.34 | 31.93 |
| `int8` | 33,792 | 4.342425 | −0.002% | 35.31 | 34.34 | 33.63 |
| `fp8` | 33,024 | 4.344724 | +0.051% | 35.05 | 33.09 | 30.34 |
| `rk8v4` | 26,112 | 4.346413 | +0.090% | 35.65 | 34.73 | 33.17 |
| `k8v4` | 25,728 | 4.347258 | +0.109% | 34.22 | 31.83 | 28.90 |
| `nvfp4` | 18,432 | 4.352201 | +0.223% | 35.37 | 32.78 | 29.86 |
| `rk4v4` | 17,920 | +0.214% against `int8` | | within 1% of `rk8v4` | | |

The `rk4v4` row was measured in a later session (`int8` 4.343155 there) and is stated against that
session's `int8`. A simulation of `rk2v4-e8`'s keys on Qwen3.8-27B cost +3.08%.
[Context and memory](cli.md#context-and-memory) turns these into recommendations.

### Drafters by concurrency

One RTX 3090, INT8 KV, thinking off, greedy; aggregate decode tok/s through the server, decode
time from the request log:

| Requests at once | MTP 3 | DFlash2 7 | Tokens per round, MTP / DFlash2 |
|---:|---:|---:|---|
| 1 | 135.0 | 187.1 | 3.51 / 5.57 |
| 2 | 238.2 | 313.5 | 3.51 / 5.82 |
| 4 | 387.4 | 406.2 | 3.54 / 5.63 |
| 8 | 522.8 | does not fit | 3.53 / — |

Speculation pays because a multi-column round costs about one sweep of the weights; batching
amortizes that sweep already, so the lead shrinks as requests are added while acceptance holds.
DFlash2's draft weights and its refusal of `--lm-head-q6` stop it at eight lanes and cost it about
65K tokens of context against MTP on a 24 GB card. Over a reasoning, a code and a summarization
prompt seven drafts were best on average, but the reasoning prompt kept improving to twelve while
summarization fell off there.

### DSpine on Qwen3.8-27B

Tencent's DSpine drafter ([converted](https://huggingface.co/WaveCut/Qwen3.8-27B-DSpine-b16-NInfer-v3),
Q8) beside the official artifact on one RTX 5090 at its 575 W default: the CLI with a 16,384-token
window and `bf16` KV, thinking off, one greedy request per configuration; decode tok/s, with the
tokens a round in parentheses. The Python answer continues a 1,024-token Python prompt, the English
one an English prose prompt.

| Answer | No drafts | MTP 3 | DFlash2 7 | DFlash2 15 | DSpine 7 | DSpine 11 | DSpine 15 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Python, 1,024 tokens | 90.7 | 251.8 (3.37) | 322.0 (4.55) | 405.1 (6.72) | 325.7 (4.92) | 421.5 (7.31) | 473.4 (8.38) |
| English, 1,024 tokens | 90.4 | 182.7 (2.45) | 187.8 (2.67) | 163.4 (2.73) | 172.9 (2.62) | 149.3 (2.60) | 156.0 (2.76) |
| code, 400 tokens | 90.3 | 232.3 (3.12) | 297.8 (4.20) | 292.7 (4.81) | 278.1 (4.20) | 251.1 (4.37) | 284.7 (4.99) |
| prose, 269 tokens | 90.3 | 185.2 (2.49) | 184.9 (2.61) | 150.4 (2.47) | 167.7 (2.54) | 151.8 (2.61) | 146.7 (2.56) |

On the long Python answer DSpine with 15 drafts keeps 8.38 tokens a round against DFlash2's 6.72 and
decodes 1.17 times as fast as DFlash2's best count. On the short code answer and on prose DSpine's
best count stays 4% and 9% behind DFlash2 7, at nearly the same tokens a round: a round of seven
drafts takes 15.2 ms of kernel time against DFlash2's 14.2 ms (Nsight Systems, the difference
between 608- and 96-token runs of the Python answer). The Q8 projection kernels account for the gap,
2.92 ms a round against 1.96 ms: DSpine's five layers with 17,408-wide MLPs, and its candidates from
the full 248,320-row head where DFlash2 reads its 131,072-row proposal head. Past eight drafts the
target's verification costs about 2 ms more a round with either drafter, so 15 drafts pay only where
most of them are kept. The BF16 drafter (`dspine_bf16`) keeps the same tokens a round as the Q8 one
and decoded 254.1 and 155.3 tok/s on the code and prose answers at seven drafts. A sampled prose
answer (temperature 0.7, seed 1) decoded 90.5 tok/s without drafts, 210.2 with DFlash2 7 and 171.7
with DSpine 7. After startup the Q8 drafter leaves 2.7 GiB less free than no drafter (DFlash2 2.5
GiB, MTP 0.8 GiB), and the full 262,144-token window still fits with `rk4v4` and `rk8v4`. A
verification's width changes floating-point results, as [the CLI guide](cli.md#speculative-decoding)
describes; DSpine and DFlash2 rounds of the same width gave the same tokens on the Python, code and
prose answers, and on the English one at seven drafts.

### DSpine under the model card's conditions

The same drafter and artifact through `ninfer-serve` on one RTX 5090 at its 600 W default
(2026-11-13), as Tencent's model card measures it: reasoning effort `xhigh`, at most 2,048 new
tokens, prompts from GSM8K, MATH-500, HumanEval, MBPP and MT-Bench (two a dataset for one request
at a time, eight for eight at once), `bf16` KV, no copy rounds (`--ngram-draft-tokens 0`), the
full head unless a row names the proposal head (`--lm-head-draft`). Aggregate tok/s over the
client's wall time, round time from the request log, and tokens a round (τ) overall and per
dataset:

| Requests, temperature | Drafter | tok/s | Round | τ | GSM8K | MATH-500 | HumanEval | MBPP | MT-Bench |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1, 0 | DFlash2 7 | 298.3 | 14.06 ms | 4.36 | 5.27 | 5.29 | 3.84 | 4.15 | 4.35 |
| 1, 0 | DFlash2 15 | 299.2 | 16.40 ms | 5.10 | 6.30 | 6.11 | 4.36 | 4.97 | 5.22 |
| 1, 0 | DSpine 7, proposal head | 294.8 | 14.24 ms | 4.36 | 5.52 | 5.04 | 3.87 | 4.13 | 4.47 |
| 1, 0 | DSpine 15 | 303.3 | 17.21 ms | 5.43 | 7.20 | 5.96 | 4.64 | 5.80 | 5.67 |
| 1, 0 | DSpine 15, proposal head | 296.7 | 16.65 ms | 5.14 | 6.54 | 5.88 | 4.31 | 5.51 | 5.36 |
| 8, 0 | none | 457.8 | | | | | | | |
| 8, 0 | DFlash2 7 | 629.1 | 45.14 ms | 4.46 | 4.93 | 5.14 | 4.42 | 4.37 | 3.94 |
| 8, 0 | DFlash2 15 | 520.1 | 57.63 ms | 4.93 | 5.75 | 6.32 | 4.85 | 4.70 | 4.15 |
| 8, 0 | DSpine 7 | 604.9 | 46.97 ms | 4.47 | 5.22 | 5.01 | 4.42 | 4.48 | 3.81 |
| 8, 0 | DSpine 7, proposal head | 597.2 | 46.12 ms | 4.38 | 5.07 | 4.94 | 4.33 | 4.39 | 3.78 |
| 8, 0 | DSpine 15 | 480.1 | 59.42 ms | 4.86 | 6.45 | 6.25 | 5.05 | 4.63 | 3.86 |
| 8, 0 | DSpine 15, proposal head | 486.9 | 59.07 ms | 4.78 | 6.32 | 6.24 | 4.96 | 4.41 | 3.76 |
| 8, 1 | none | 459.8 | | | | | | | |
| 8, 1 | DFlash2 7 | 575.0 | 44.34 ms | 4.05 | 4.86 | 5.22 | 4.09 | 3.85 | 3.52 |
| 8, 1 | DSpine 7, proposal head | 556.8 | 44.43 ms | 4.02 | 4.82 | 4.67 | 4.13 | 3.82 | 3.25 |
| 8, 1 | DSpine 15, proposal head | 466.7 | 59.23 ms | 4.43 | 5.60 | 5.41 | 4.66 | 4.03 | 3.72 |

DSpine 15 keeps 7.20 tokens a round on GSM8K at temperature 0, the card's figure; the card's
MATH-500, HumanEval, MBPP and MT-Bench figures (6.49, 5.71, 6.14, 4.48) come from its full
datasets on the BF16 model under SGLang, these from two to eight prompts each on the groupwise-int
artifact. The card compares DSpine's block of 16 with DFlash2's of 8, and most of its gain is that
width: at 15 drafts each, DSpine keeps 5.43 tokens a round against DFlash2's 5.10 for one request
(DFlash2 through the proposal head; 5.14 for DSpine there) and 4.86 against 4.93 for eight. On the
RTX 5090 a round of 15 drafts costs 2.3 ms more than one of seven for one request and 12.5 ms more
for eight, whose verification of 128 columns is bound by compute (a Q4 [34816, 5120] projection
takes 65.5 µs at 8 columns and 81.9 µs at 16 in graph replay), so seven drafts decode 1.26 times as
fast as 15 at eight requests. At temperature 1 the drafters keep 0.89-0.93 times their greedy
tokens a round, the ratio the card reports. A run of eight requests ends on its slowest answers,
so its aggregate tok/s varies with the longest one; the round times compare the drafters more
closely. The fused layer routes and the one-pass candidate head (docs/maintainer/dspine.md) cut
the 15-draft round from 17.44 to 17.21 ms for one request and from 61.17 to 59.42 ms for eight.
The proposal head keeps fewer tokens a round for DSpine (5.43 against 5.14 for one request), which
its 3% shorter round does not repay, so the automatic head stays the full one for DSpine.
With [`--adaptive-mtp`](serving.md#adaptive-mtp) each round drafts 15 or seven: eight requests at
temperature 0 decode 561.0 tok/s for DSpine and 601.1 for DFlash2, between the two fixed widths.

### MTP draft count by content

One RTX 3090, `rk4v4` KV with `--lm-head-q6 --embedding-q4 --gdn-state-fp16`, greedy, 512
generated tokens, one stream; decode tok/s with tokens per round in brackets:

| `--draft-tokens` | edit code (copies the prompt) | write new code | explain a concept | short story |
|---:|---:|---:|---:|---:|
| none | 47.3 | 47.5 | 47.5 | 47.6 |
| 3 | 141.9 (3.9) | 118.2 (3.3) | 102.1 (2.8) | 77.1 (2.1) |
| 5 | 177.7 (5.7) | 124.8 (4.0) | 95.6 (3.1) | 75.1 (2.4) |
| 7 | 226.0 (7.4) | 130.6 (4.3) | 103.9 (3.4) | 67.7 (2.2) |
| 9 | 226.4 (8.8) | 129.9 (5.0) | 91.3 (3.5) | 56.3 (2.2) |
| 15 | 262.6 (12.5) | 114.7 (5.4) | 75.1 (3.5) | 47.6 (2.2) |

Each round verifies K+1 columns and runs K draft steps whether or not the drafts survive, so a
larger K pays only where the head keeps guessing right ([choosing a draft
count](cli.md#choosing-a-draft-count)).

### Several requests at once

The same server runs as for Bonsai 2, total generated tok/s:

| Card | Drafter | 1 | 2 | 4 | 8 |
|---|---|---:|---:|---:|---:|
| RTX PRO 6000 | none | 88.8 | 172.6 | 327.9 | 569.8 |
| RTX PRO 6000 | MTP 3 | 154.4 | 300.9 | 515.8 | 739.3 |
| RTX PRO 6000 | DFlash2 5 | 176.1 | 319.4 | 473.7 | 534.6 |
| RTX 5090 | none | 91.0 | 172.0 | 324.7 | 569.1 |
| RTX 5090 | MTP 3 | 160.6 | 299.0 | 490.0 | 690.0 |
| RTX 5090 | DFlash2 5 | 171.6 | 311.1 | 430.9 | 469.4 |
| RTX 4090 | none | 54.3 | 105.6 | 198.4 | 356.7 |
| RTX 4090 | MTP 3 | 92.8 | 174.9 | 300.3 | 442.2 |
| RTX 4090 | DFlash2 5 | 107.7 | 185.1 | 295.2 | |
| RTX 3090 | none | 48.1 | 91.9 | 172.2 | 290.2 |
| RTX 3090 | MTP 3 | 88.2 | 154.8 | 238.4 | 328.9 |
| RTX 3090 | DFlash2 5 | 89.1 | 155.4 | 206.0 | |

### RTX PRO 6000 against RTX 5090, October 2026

The PRO 6000's server rows measured again next to an RTX 5090 (600 W boards), build `796f98500`,
CUDA 13.1; decode tok/s, prompt time in seconds, acceptance in brackets:

| Model, KV, drafter, request | RTX PRO 6000, September | RTX PRO 6000, October | RTX 5090, October |
|---|---:|---:|---:|
| Bonsai 2, `rk8v4`, none, short chat | 167.5 | 167.7 | 170.4 |
| Bonsai 2, `rk8v4`, DFlash2 7, short chat | 381 | 378.8 | 372.1 |
| Bonsai 2, `rk8v4`, MTP 3, short chat | 314.7 | 313.9 | 321.2 |
| Bonsai 2, `rk8v4`, none, prompt of 261,120 tokens | 78.3 s | 75.0 s | 76.7 s |
| Bonsai 2, `rk8v4`, none, decode after it | 96.4 | 96.7 | 99.0 |
| Bonsai 2, `rk4v4`, MTP 3, decode after it | 217.9 (0.56) | 220.9 (0.56) | 194.8 (0.46) |
| Bonsai 2, `rk8v4`, MTP 3, decode after it | 210.5 (0.60) | 186.4 (0.49) | 207.6 (0.57) |
| Bonsai 2, `rk8v4`, MTP 3, eight requests at once | 1,155.3 | 1,159.6 | 1,081.5 |
| Qwen3.8, `rk8v4`, none, short chat | 89.6 | 90.0 | 92.1 |
| Qwen3.8, `rk8v4`, DFlash2 7, short chat | 237 | 238.5 | 233.5 |
| Qwen3.8, `rk8v4`, MTP 3, eight requests at once | 739.3 | 749.5 | 709.9 |

One request decodes no faster on the PRO 6000 than on the RTX 5090: that decode is bound by memory
bandwidth, 1.79 TB/s of GDDR7 on both. Its 188 SMs against 170 show where compute decides: it
prefills the 261K prompt 2% faster and runs eight requests at once 6 to 7% faster.

## Qwen3.8-27B GSQ-RCO IQ3_S

The [3.5-bit GGUF release](https://huggingface.co/WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3),
10.9 GiB of weights.

### Server

`rk8v4`, one request; IQ3_S / the official artifact on the same card in the same sitting (RTX 3090
at 420 W and RTX 4090 with a 176,128-token window, RTX 5090 at 450 W with 262,144); the Tesla V100
runs IQ3_S alone with a 65,536-token window (16,384 for DFlash2):

| Measure | RTX 5090 | RTX 4090 | RTX 3090 | Tesla V100 |
|---|---:|---:|---:|---:|
| short chat, no drafter, tok/s | 107.5 / 88.1 | 70.6 / 55.0 | 59.9 / 40.3 | 45.5 |
| short chat, MTP 3, tok/s | 221.4 / 178.1 | 146.4 / 109.3 | 108.7 / 82.7 | 77.0 (0.58, 2.72) |
| short chat, DFlash2 5, tok/s | 226.5 / 222.5 | 169.4 / 141.3 | 115.6 / 109.2 | 65.3 (0.46, 3.28) |
| decode after 32,768 tokens, no drafter, tok/s | 100.3 / 83.2 | 66.0 / 52.2 | 50.4 / 38.4 | 42.5 |
| prompt time, 32,768 tokens, s | 8.6 / 9.3 | 9.3 / 12.8 | 20.5 / 22.2 | 38.9 |

On the V100, decode after 1,024 and 8,192 tokens is 45.4 and 44.9 tok/s without a drafter, 65.0
and 64.3 with MTP 3 (64.2 after 32,768) and 53.4 and 53.1 with DFlash2 5; prompt time is 1.18 s at
1,024 tokens and 8.3 s at 8,192.

Users' measurements are in the [field reports](performance/field-reports.md): an RTX 5070 Ti
from 1.2K to 104K prompt tokens against llama.cpp, RTX 4090s serving long Qwen Code and agent
sessions with DFlash2 and MTP, and Qwen3.8-27B NVFP4 with tensor parallelism over two and four
RTX 5060 Ti.

### Engine and CLI rates

Prompt and decode in tok/s:

| Card | Method, KV | Prompt 512 | Prompt 2,048 | Long prompt | Decode | Decode after the long prompt |
|---|---|---:|---:|---:|---:|---:|
| RTX 3090 | engine, `rk8v4` | 1,826 | 1,877 | | 59.2 | |
| TITAN RTX | engine, `rk8v4` | 757 | 772 | 701 (16,384); 598 (65,536) | 34.8 | 33.1; 33.4 |
| Tesla V100 | engine, `bf16` | 944 | 1,044 | 914 (16,384) | 45.3 | 42.6 |
| Tesla V100 | llama.cpp, F16 KV | 768 | 792 | | 36.6 | 31.2 |
| RTX 3060 | CLI, `int8` | | | 567 (2,273) | 25.8 | |

With MTP 3 the engine decodes 79.1 tok/s on the RTX 3090 and 41.6 on the TITAN RTX (both 0.29,
1.87); with DFlash2 3 the TITAN RTX decodes 32.5 (0.18, 1.52).

### Quality

WikiText-2 test, disjoint 2,048-token windows, BF16 KV: 7.071 ([GGUF](gguf.md#measurements)),
7.0703 on the Tesla V100; the official artifact 7.286.

## Qwen3.8-27B MXFP8

The [MXFP8 tune](https://huggingface.co/WaveCut/Qwen3.8-27B-MXFP8-CRACK-NInfer-v3), 17 GiB: too
large for 16 GB. On the TITAN RTX (engine, `rk8v4`) it prompts at 810 tok/s (512 tokens) and 832
(2,048) and decodes at 30.9.

## Qwen3.6-35B-A3B

- **One RTX 3090** (capped at 315 W, the official `groupwise-int` artifact, 19.6 GiB of weights),
  decode tok/s without a drafter at 4,096 / 16,384 / 32,768 tokens of context, the mean of two
  runs: `rk8v4` 175.15 / 168.50 / 158.47, `int8` 171.76 / 165.02 / 154.88, `bf16` 166.16 / 156.00 /
  142.99, `fp8` 166.01 / 148.82 / 132.00, `nvfp4` 169.04 / 151.81 / 131.75, `k8v4` 165.66 / 143.58 /
  122.78 ([configuration calculator](config-calculator.html)).
- **Two TITAN RTX as pipeline stages**: 146.6 tok/s, 146.2 with MTP 3 (2.70 tokens a round), CLI,
  a 144-token prompt and 128 tokens. The artifact does not fit one TITAN RTX beside the runtime.
- **NVFP4 on an RTX 5090**: the
  [NVFP4 artifact](https://huggingface.co/WaveCut/Qwen3.6-35B-A3B-NVFP4-NInfer-v3) starts in 17 s
  and leaves 7.4 GiB of the card free.

### Host experts on one RTX 4090

`ninfer-serve` with the official `groupwise-int` artifact, `--expert-residency host`, a
131,072-token context, three MTP drafts and the automatic expert cache, on one RTX 4090 (PCIe 4.0
x16, a container with a 30-CPU quota, pinned to the GPU's NUMA node), 2026-10-20; 256 greedy
tokens after a 42-token code prompt and a 43-token prose prompt (twice each) and after a
31,865-token document (once). `c04ea310` lends neither the layer banks nor free KV pages to the
expert cache and copies every miss; the current build does both and, by default, has the CPU compute
a share of each narrow call's misses ([routed expert
residency](maintainer/qwen3_5-model.md#routed-expert-residency)). Decode in tok/s; after the
document and its prefill: `c04ea310` / current / current with `--expert-dma-share 1` (no CPU share):

| `--kv-dtype` | Decode, short prompts: `c04ea310` | current | no CPU share | Decode after the document | Prefill of the document |
|---|---|---|---|---|---|
| `bf16` | 183-189 | 211-245 | 200-230 | 162 / 186 / 178 | 7,700 / 8,405 / 8,386 |
| `int8` | 195-210 | 213-254 | 200-233 | 168 / 203 / 191 | 7,621 / 7,999 / 7,942 |
| `fp8` | 189-215 | 205-249 | 198-227 | 176 / 181 / 174 | 6,012 / 6,326 / 6,303 |
| `k8v4` | 198-210 | 219-241 | 206-220 | 165 / 188 / 180 | 5,551 / 5,731 / 5,713 |
| `nvfp4` | 191-210 | 221-242 | 212-227 | 174 / 191 / 184 | 5,306 / 5,425 / 5,405 |
| `rk8v4` | 202-216 | 220-238 | 215-226 | 182 / 197 / 190 | 7,747 / 8,098 / 8,091 |
| `rk4v4` | 201-218 | 219-246 | 209-232 | 172 / 192 / 181 | 7,755 / 8,100 / 7,835 |
| `rk4v4-e8` | 191-223 | 212-225 | 197-217 | 180 / 179 / 173 | 7,741 / 7,934 / 7,945 |
| `rk2v4-e8` | 180-227 | 203-241 | 195-228 | 180 / 184 / 177 | 7,714 / 7,858 / 7,945 |

Over all 45 decode measurements the current build decoded 0.997-1.329 times as fast as
`c04ea310` (geometric mean 1.109) and 0.960-1.260 times without the CPU share (1.063); the slowest
pairs were after the document with `rk4v4-e8` (0.997, and 0.960 without the CPU share). The lent
banks and pages raised the cache from 4,156-4,756 to 5,415-5,539 expert slots and cut the expert
bytes copied over the run from 169-197 GB to 122-125 GB (146-151 GB without the CPU share). The
CPU share changes no output: with and without it, all 45 answers were equal with equal log
probabilities. Every answer differs from `c04ea310`'s, whose GPU computed SiLU with CUDA's
`expf` (an approximate hardware exponential); the current kernels compute it from correctly rounded
operations, as the CPU does. Startup took 1.2-1.7 s longer and device memory was unchanged (22,860 MiB).

## Qwen3.8-Flash-Next

### Expert placements

Greedy, one request. Decode is measured over a short answer (78 tokens for Q2_0, 81 to 87 for
IQ3_S) and over the first tokens after a 4,463-token prompt; prefill is that prompt. Host memory
is the process's peak resident set, device memory the most `nvidia-smi` showed. The host-memory
and disk rows of one card use its free memory as the expert cache. The single RTX 5090, RTX 4090
and RTX 3090 rows are the generate test of October 26 in 512-token prefill chunks (with host or disk
experts a span of eight chunks and a last chunk of 367 tokens), one run each;
the others come from builds of October 5 and 6, where a range is two runs
([hosts and builds](qwen3-8-flash-next.md#expert-placements)).

| Card | Release | Experts | Device memory | Host memory | Decode, short answer | Decode after 4,463 tokens | Prefill |
|---|---|---|---:|---:|---:|---:|---:|
| RTX PRO 6000 | Q2_0 | on the GPU | 37.7 GiB | 0.8 GiB | 136.0-139.7 | 160.7-162.6 | 2,925-2,952 |
| RTX PRO 6000 | IQ3_S | on the GPU | 53.7 GiB | 1.0 GiB | 123.8-125.9 | 146.8-146.9 | 2,540-2,541 |
| RTX PRO 6000 | Q2_0 | pinned host memory | 38.5 GiB | 34.0 GB | 54.8-55.0 | 67.7-70.0 | 1,296-1,297 |
| RTX PRO 6000 | IQ3_S | pinned host memory | 55.0 GiB | 50.3 GB | 29.3 | 31.2 | 803 |
| RTX PRO 6000 | Q2_0 | disk, in the page cache | 93.6 GiB | 1.0 GiB | 75.9 | 115.8 | 2,024 |
| RTX PRO 6000 | IQ3_S | disk, in the page cache | 93.5 GiB | 1.2 GiB | 65.6 | 106.0 | 1,661 |
| RTX PRO 6000 | Q2_0 | disk, pages evicted every second | 93.5 GiB | 1.0 GiB | 35.1 | 87.1 | 1,000 |
| RTX PRO 6000 | IQ3_S | disk, pages evicted every second | 93.5 GiB | 1.2 GiB | 28.7 | 78.6 | 773 |
| 2× RTX 5090 | Q2_0 | on the GPUs | 18.8 + 19.6 GiB | 0.9 GiB | 136.1-136.3 | 157.0-157.2 | 3,807-3,811 |
| 2× RTX 5090 | IQ3_S | on the GPUs | 26.5 + 27.8 GiB | 1.0 GiB | 121.6-121.9 | 141.8-142.0 | 3,310-3,341 |
| RTX 5090 | Q2_0 | pinned host memory | 30.1 GiB | 33.3 GiB | 97.8 | 83.2 | 3,290 |
| RTX 5090 | IQ3_S | pinned host memory | 30.1 GiB | 48.7 GiB | 82.5 | 70.7 | 2,451 |
| RTX 5090 | Q2_0 | disk, in the page cache | 30.1 GiB | 1.7 GiB | 72.1 | 102.9 | 1,827 |
| RTX 5090 | IQ3_S | disk, in the page cache | 30.1 GiB | 2.0 GiB | 56.4 | 35.5 | 770 |
| RTX 5090 | Q2_0 | disk, pages evicted every second | 30.1 GiB | 1.7 GiB | 37.7 | 66.4 | 1,108 |
| RTX 5090 | IQ3_S | disk, pages evicted every second | 30.1 GiB | 2.0 GiB | 28.5 | 16.7 | 399 |
| 2× RTX 4090 | Q2_0 | on the GPUs | 18.5 + 19.4 GiB | 0.8 GiB | 92.6-100.7 | 116-120 | 3,482-3,734 |
| RTX 4090 | Q2_0 | pinned host memory | 22.1 GiB | 33.1 GiB | 73.9 | 66.9 | 1,801 |
| RTX 4090 | Q2_0 | disk, in the page cache | 22.1 GiB | 1.5 GiB | 46.5 | 42.5 | 1,144 |
| RTX 4090 | Q2_0 | disk, pages evicted every second | 22.1 GiB | 1.5 GiB | 21.0 | 14.6 | 558 |
| 2× RTX 3090 Ti | Q2_0 | on the GPUs | 18.4 + 19.4 GB | 0.7 GB | 90.2 | 107 | 1,504 |
| RTX 3090 | Q2_0 | pinned host memory | 22.2 GiB | 33.1 GiB | 39.1 | 45.7 | 663 |
| RTX 3090 | IQ3_S | pinned host memory | 22.2 GiB | 48.5 GiB | 27.1 | 33.1 | 438 |
| RTX 3090 | Q2_0 | disk, in the page cache | 22.1 GiB | 1.5 GiB | 27.8 | 27.5 | 503 |
| RTX 3090 | IQ3_S | disk, in the page cache | 22.1 GiB | 1.7 GiB | 17.4 | 13.7 | 256 |
| RTX 3090 | Q2_0 | disk, pages evicted every second | 22.1 GiB | 1.5 GiB | 19.7 | 14.4 | 474 |
| RTX 3090 | IQ3_S | disk, pages evicted every second | 22.1 GiB | 1.7 GiB | 12.3 | 6.0 | 179 |
| L40S (48 GB) | Coder IQ1_M | pinned host memory | | | 35 | | 1,648 |
| L40S (48 GB) | Coder IQ1_M | disk, in the page cache | | | 42 | | 1,938 |

The Coder IQ1_M rows are the 28.4 GiB model beside the shared n-gram table on one NVIDIA L40S (48 GB).
The host link carries host and disk experts: PCIe 5.0 x16 on the RTX 5090 hosts, 4.0 x16 on the
RTX PRO 6000 and RTX 4090 hosts, 3.0 x16 on the two-socket RTX 3090 host, whose rows ran pinned to
the GPU's NUMA node. With host or disk experts a prompt prefills in spans of up to eight chunks, so
a layer's missing experts cross the link once per span. From the CLI with its defaults, Q2_0 disk
experts in the page cache and a 7,680-token prompt, the RTX 5090 prefilled at 1,830 tok/s and
decoded the 16 tokens after it at 115 tok/s, against 450 and 32 when disk experts ran chunk by
chunk ([all three cards](qwen3-8-flash-next.md#expert-placements)).

Other cards ran the Q2_0 release from the CLI with other prompts, 128 generated tokens, a cold
expert cache, the files in the page cache (most or all of them on the 8 and 12 GB cards); prefill /
decode tok/s with the expert-cache hits:

| Card | Experts | Prompt | Prefill / decode | Hits |
|---|---|---:|---|---:|
| TITAN RTX, two | on the GPUs, two pipeline stages | 144 tokens | — / 76.0 | |
| TITAN RTX | pinned host memory | 144 tokens | — / 53.5 | 43.7% |
| TITAN RTX | disk | 144 tokens | — / 30.3 | 83.5% |
| Tesla V100 | pinned host memory | 66 tokens | 38.8 / 35.9 | 58.6% |
| Tesla V100 | disk | 66 tokens | 38.2 / 30.8 | 81.0% |
| RTX 3060 | pinned host memory | 69 / 2,273 tokens | 58.9 / 39.9; 569 / 38.2 | 51.3%; 26.4% |
| RTX 3060 | disk | 69 / 2,273 tokens | 37.9 / 23.5; 215 / 22.9 | 70.2%; 44.2% |
| RTX 3060 Ti | pinned host memory | 69 / 2,273 tokens | 73.3 / 33.0; 723 / 30.6 | 29.3%; 13.7% |
| RTX 3060 Ti | disk | 69 / 2,273 tokens | 52.8 / 17.6; 356 / 16.4 | 39.9%; 23.3% |

The RTX 3060 and 3060 Ti hold 4.9 to 5.3 GiB and 0.9 to 1.3 GiB of expert cache; their
host-memory rows peak at 32.7 GiB of RAM, their disk rows at 1.2 to 1.3 GiB.

### Long contexts with host experts on one RTX 4090

The IQ3_S release through `ninfer-serve` with host experts and its defaults (three MTP drafts, the
CPU share, the streamed sparse-attention KV), against commit `5ed86dff9` with three drafts and the
CPU share; 256 greedy tokens after short prompts and after a long document
([details](qwen3-8-flash-next.md#host-experts-and-long-contexts-on-one-rtx-4090)):

| Context | Build | Expert cache | Decode, short prompts | Decode after the document | Prefill |
|---:|---|---:|---|---|---:|
| 65,536 | `5ed86dff9` | 14,080 MiB | 66.5-77.1 | 66.5-75.0 (31,865 tokens) | 2,576-2,714 |
| 65,536 | defaults | 16,059 MiB | 72.6-82.5 | 66.0-79.0 (31,865 tokens) | 2,840 |
| 458,752 | `5ed86dff9`, no drafts | 4,589 MiB | 39.2-43.0 | 37.7 (136,077 tokens) | 2,612 |
| 458,752 | defaults | 14,643 MiB | 67.3-70.3 | 67.6 (136,077 tokens) | 2,708 |

The same release, context and prompts on a second RTX 4090 rental (a container with a 30-CPU
quota, pinned to the GPU's NUMA node), 2026-10-20, against `c04ea310` for every `--kv-dtype`;
decode in tok/s over the two short prompts answered twice and after the document, and its prefill,
as `c04ea310` / current:

| `--kv-dtype` | Decode, short prompts: `c04ea310` | current | Decode after the document | Prefill of the document |
|---|---|---|---|---|
| `bf16` | 78.0-82.6 | 78.9-86.1 | 72.8 / 74.4 | 2,861 / 3,736 |
| `int8` | 79.9-85.9 | 79.5-85.6 | 75.0 / 75.7 | 2,842 / 3,678 |
| `fp8` | 80.8-89.2 | 81.0-90.4 | 76.2 / 76.0 | 2,767 / 3,599 |
| `k8v4` | 83.0-90.4 | 81.5-91.8 | 75.9 / 77.6 | 2,778 / 3,562 |
| `nvfp4` | 76.3-84.0 | 75.9-87.8 | 72.7 / 77.0 | 2,774 / 3,546 |
| `rk8v4` | 82.6-95.9 | 80.7-93.9 | 74.9 / 77.9 | 2,858 / 3,704 |
| `rk4v4` | 80.4-89.1 | 83.5-91.6 | 74.3 / 78.9 | 2,835 / 3,667 |
| `rk4v4-e8` | 76.5-87.6 | 79.0-91.2 | 71.9 / 73.2 | 2,865 / 3,710 |
| `rk2v4-e8` | 69.1-74.4 | 69.2-73.3 | 60.2 / 60.9 | 2,829 / 3,667 |

Decode changed by 0.956-1.062 times over the 45 measurements (geometric mean 1.016), within the
spread of repeated runs; every answer equals `c04ea310`'s. The document prefilled 1.27-1.31 times
as fast (two prefill buffers by default), and the first token of a short prompt after it came in
0.45-0.71 s instead of 1.03-2.75 s (sequence images in pages pinned at startup). Startup took 0-2.9 s
longer, most of it pinning those pages; device memory was within 80 MiB.

### MTP

Two RTX 3090s (350 W), the Q2_0 release with Unsloth's `shared-Q8_0` MTP block, greedy decoding of
a 117-token answer to a 37-token coding prompt, two runs each; every speculative output is the
plain decode's, token for token:

| Placement | Drafts | Decode, tok/s | Drafts accepted | Tokens a round |
|---|---:|---:|---:|---:|
| 2× RTX 3090, experts on the GPUs | none | 92.9-93.0 | | |
| 2× RTX 3090, experts on the GPUs | 1 | 130.1-130.9 | 96.6% | 1.97 |
| 2× RTX 3090, experts on the GPUs | 2 | 150.3-150.5 | 91.5% | 2.83 |
| 2× RTX 3090, experts on the GPUs | 3 | 167.8-171.0 | 95.6% | 3.87 |
| 2× RTX 3090, experts on the GPUs | 4 | 171.9-173.6 | 91.0% | 4.64 |

A prose answer keeps 38.6% of three drafts and gains nothing.

### Serving with host experts on one RTX 3090

`ninfer-serve`, INT8 KV, an 8,192-position context, one request at a time; three prompts (Python
code, prose, Chinese) twice, 256 greedy tokens each; decode tok/s of the first / second run:

| Mode | Code | Prose | Chinese |
|---|---|---|---|
| no drafter | 87.3 / 92.3 | 93.7 / 94.3 | 93.4 / 94.3 |
| MTP 2 | 118.8 / 118.6 | 117.3 / 114.8 | 95.8 / 101.2 |
| MTP 3 | 121.2 / 122.6 | 117.7 / 111.1 | 90.0 / 94.6 |
| MTP 4 | 111.1 / 105.9 | 101.9 / 96.2 | 74.6 / 78.5 |
| disk experts, first run | 68.1 | 71.4 | |

### Quality

The first 72 KB of the WikiText sample in `perplexity-1m`, 2,560-token windows advancing by 1,024,
BF16 KV, the fourteen windows llama.cpp evaluates identically: Q2_0 2.6579 on an RTX 3090, 2.6489
on an L40S and 2.6509 on the Tesla V100 (llama.cpp 2.6502); IQ3_S 2.1317 (llama.cpp 2.1278). Over
five 65,536-token windows of PG-19 on two RTX 5090s: 8.0834 against llama.cpp's 8.0802. Host and
disk experts give the same perplexity on every card checked (2.6057 on the V100 over the first
40 KB, 2.53655 on the TITAN RTX over the first 40,000 bytes of `wikitext/00.txt`, INT8 KV).

## Edge0-145B and Edge0-120B

Every release on one A40, with each expert placement that fits it, with and without MTP, and against the publisher's engine: [Edge0 measurements](edge0.md#measurements).

## Gemma 4 26B-A4B

Google's [QAT Q4_0 release](https://huggingface.co/WaveCut/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-NInfer-v3)
with its MTP assistant, F16 KV (`bf16`) unless a row names another.

### Engine rates

Prompt and decode in tok/s; llama.cpp on the same GGUF with flash attention:

| Card | Engine | Prompt 2,048 | Prompt 10,000 | Prompt 40,000 | Decode | after 10,000 | after 40,000 |
|---|---|---:|---:|---:|---:|---:|---:|
| RTX 3090 (350 W) | NInfer | 6,566 | 6,045 | 4,698 | 215.4 | 190.4 | 162.2 |
| RTX 3090 (350 W) | llama.cpp | 4,925 | 4,697 | 3,767 | 168.6 | 151.7 | 135.7 |

| Card | Engine | Prompt 512 | Prompt 2,048 | Prompt 16,384 | Decode 128 | after 16,384 |
|---|---|---:|---:|---:|---:|---:|
| RTX 3090 | NInfer | 5,068 | 6,976 | | 222.9 | |
| TITAN RTX | NInfer | 2,532 | 3,571 | 2,660 | 158.8 | 158.5 |
| Tesla V100 | NInfer | 2,631 | 3,005 | 2,572 | 167.7 | 145.3 |
| Tesla V100 | llama.cpp | 1,608 | 1,514 | | 120.0 | 108.7 |

The MTP assistant with 2 drafts decodes 435.2 tok/s on the RTX 3090 and 290.7 on the TITAN RTX
(engine, 512-token prompt; 0.95, 2.91 on both).

### KV formats on Gemma 4

Decode tok/s of 128 tokens on an RTX 3090 held at 300 W, the HauhauCS Balanced artifact, one
request, no drafts; prompts stay within 1.5% of F16 in every format:

| `--kv-dtype` | empty context | after 10,000 | after 40,000 |
|---|---:|---:|---:|
| `bf16` | 199.8 | 176.3 | 150.4 |
| `int8` | 195.0 | 175.2 | 149.2 |
| `rk8v4` | 194.6 | 175.9 | 149.7 |
| `rk4v4` | 193.8 | 175.5 | 150.5 |
| `rk4v4-e8` | 192.6 | 174.8 | 149.7 |
| `rk2v4-e8` | 185.6 | 167.7 | 144.7 |
| `fp8` | 177.9 | 154.7 | 130.5 |
| `k8v4` | 177.0 | 152.6 | 129.1 |
| `nvfp4` | 176.8 | 149.7 | 125.5 |

The FP8 and NVFP4 formats lose 11 to 17% already on an empty context: the RTX 3090 has no E4M3 or
E2M1 conversion instruction (sm_89 adds them), and those decodes run in software.

The longest `--max-context` that `ninfer-serve --max-concurrency 8` starts with on the RTX 3090,
HauhauCS artifact, context cache on, no vision tower:

| `--kv-dtype` | 8 sequences, MTP 2 | 8 sequences, no drafts |
|---|---:|---:|
| `bf16` | 24,576 | 36,864 |
| `int8` | 82,944 | 96,256 |
| `fp8` | 84,992 | 98,304 |
| `rk8v4` | 92,160 | 103,424 |
| `k8v4` | 94,208 | 105,472 |
| `nvfp4` | 163,840 | 179,200 |
| `rk4v4`, `rk4v4-e8` | 164,864 | 180,224 |
| `rk2v4-e8` | 173,056 | 186,368 |

Perplexity by format is in [Gemma 4](gemma4.md#quality-and-speed).

### Server

Ten chat requests one at a time to `ninfer-serve` on the RTX 3090, 384 generated tokens each; every
configuration generates the same tokens:

| Decode tok/s | greedy | after a 13,600-token prompt | sampled (T 1, top-k 64, top-p 0.95) |
|---|---:|---:|---:|
| no drafter | 209.1 | 186.0 | 207.9 |
| MTP 1 (acceptance 0.81) | 280.1 | 240.2 | |
| MTP 2 (0.70; 0.63 sampled) | 290.7 | 241.9 | 270.6 |
| MTP 3 (0.62; 0.55 sampled) | 293.7 | 229.2 | 268.0 |

llama.cpp's server with the same GGUF and drafter decodes the greedy requests at 131.4 tok/s, 186.4
with one draft, 201.6 with two and 195.2 with three. Parallel streams, total tok/s:

| Streams | no drafter | with the assistant |
|---:|---:|---|
| 1 | 208.4 | 295.5 with three drafts, 291.8 with two |
| 2 | 310.1 | 370.0 with one draft each |
| 3 | 354.7 | 365.7 with one draft each |
| 4 | 397.2 | 387.6 with one draft each, so four or more decode plainly |

On the Tesla V100 (window 28,672 tokens), short chat decodes 164.3 tok/s without a drafter and
226.3 with MTP 2 (0.70, 2.40); after 1,024 and 8,192 tokens 155.6 and 149.5, with MTP 2 187.7 and
167.0. Prompt time is 0.36 s at 1,024 tokens and 2.9 s at 8,192.

## Flint

Recursive's Flint from Unsloth's
[UD-Q4_K_XL GGUF](https://huggingface.co/WaveCut/Flint-UD-Q4_K_XL-NInfer-v3), F16 KV, one sequence.

### Engine rates

Prompt and decode in tok/s; llama.cpp b11804 on the same GGUF with flash attention. The 23.4 GiB of
weights fit only the RTX 5090, where both engines keep every expert on the GPU. On the 24 GB cards
NInfer keeps the experts in pinned host memory with the device cache it sizes itself (19 GiB), and
llama.cpp keeps the experts of its first four blocks on the CPU (`-ncmoe 4`, the fewest that run a
10,000-token prompt):

| Card | Engine | Prompt 2,048 | Prompt 10,000 | Decode | after 10,000 |
|---|---|---:|---:|---:|---:|
| RTX 5090 (575 W) | NInfer | 15,670 | 14,656 | 347.7 | 314.6 |
| RTX 5090 (575 W) | llama.cpp | 8,702 | 8,679 | 222.7 | 209.2 |
| RTX 4090 (500 W) | NInfer, host experts | 12,423 | 11,466 | 210.2 | 188.0 |
| RTX 4090 (500 W) | llama.cpp, `-ncmoe 4` | 4,868 | 4,725 | 165.7 | 149.4 |
| RTX 3090 (370 W) | NInfer, host experts | 6,225 | 5,755 | 161.4 | 140.2 |
| RTX 3090 (370 W) | llama.cpp, `-ncmoe 4` | 2,633 | 2,553 | 79.9 | 77.1 |

[Flint's measurements](flint.md#measurements) name the hosts and add disk experts and context
lookup.

## DeepSeek-V4.1-Flash

ISTA-DASLab's GSQ-RCO IQ1_M GGUF with its Engram table
([artifact](https://huggingface.co/WaveCut/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-NInfer-v3)), BF16
caches, one request, greedy, thinking off, the same prompt tokens in both engines. One rented host:
an RTX 4090 (450 W, PCIe 4.0 x16), two 32-core AMD EPYC engineering samples and 409 GiB of RAM.
NInfer: `ninfer-serve` with host experts, by default with CPU threads computing a share of each
decode call's missing experts. Strata: 0.2.6, its latest release (`strata-dsv41` 0.1.51), all 6
experts, the flags its app starts the engine with and the four settings its tuner tries:

| Engine and settings | Decode, short answer | after 8,086 tokens | Prefill of 8,086 tokens |
|---|---:|---:|---:|
| NInfer | 38.9 | 31.4 | 938 |
| NInfer, `--expert-dma-share 1` (every miss copied) | 33.5 | 26.6 | 929 |
| NInfer `b3402a6fd` (prompts chunk by chunk, every miss copied) | 25.8 | 22.1 | 498 |
| Strata, `--dspark auto --dspark-conf 0.65` (its app) | 36.0 | 38.4 | 635 |
| Strata, `--dspark on` | 34.3 | 38.7 | 690 |
| Strata, `--dspark on --dspark-block 3` | 39.9 | 37.1 | 660 |
| Strata, `--cpu-threads 64` | 38.0 | 38.6 | 659 |
| Strata, `--mmap-experts off --ssd-cache 0` | 39.4 | 38.3 | 658 |

Strata computes 53 to 61% of the routed experts on its CPU threads beside the GPU. With the build
`7731f3d30` NInfer's device cache served 62 to 72% of the decode routes, its CPU share 16 to 22%,
and the GPU copied the rest
([the CPU share](deepseek-v41.md#the-cpu-share-on-one-rtx-4090)). With `--experts 5` (5 of the 6
experts, another model's output) Strata decodes 44.8 and 42.1 tok/s.

With Unsloth's DSpark drafter, October 26, the same prompts and host experts, on that host and
on an RTX PRO 6000 host (Threadripper PRO 7965WX), where the build before the CPU share copied every
miss:

| Card | Drafts | Decode, short answer | after 8,086 tokens | Prefill of 8,086 tokens |
|---|---|---:|---:|---:|
| RTX PRO 6000 (600 W) | none | 50.8 | 48.3 | 1,210 |
| RTX PRO 6000 (600 W) | `--spec dspark` | 83.1 | 75.0 | 1,290 |
| RTX 4090 (450 W) | none | 38.9 | 31.4 | 938 |
| RTX 4090 (450 W) | `--spec dspark` | 39.9 | 33.2 | 693 |
| RTX 4090 (450 W), `--expert-dma-share 1` | `--spec dspark` | 25.1 | 20.2 | 688 |

Strata 0.2.6 exited with an error in every configuration tried on the RTX PRO 6000 host
([the drafter](deepseek-v41.md#the-dspark-drafter)). The published artifact carries the STQ drafter
since October 27, which keeps Unsloth's acceptance with half its device memory
([the STQ drafter](deepseek-v41.md#the-stq-drafter)).

NInfer's native release build of `b3402a6fd` on one GPU of a rented host per card, the same prompts:

| Card | Experts | Decode, short answer | after 8,086 tokens | Prefill of 8,086 tokens |
|---|---|---:|---:|---:|
| RTX PRO 6000 (600 W) | host | 48.0 | 45.3 | 1,310 |
| RTX PRO 6000 (600 W) | disk, page cache warm | 37.4 | 33.4 | 431 |
| RTX 5090 (575 W) | host | 29.2 | 22.5 | 499 |
| RTX 5090 (575 W) | disk, page cache warm | 12.5 | 10.5 | 181 |
| RTX 4090 (450 W) | host | 25.9 | 22.2 | 490 |
| RTX 4090 (450 W) | disk, page cache warm | 8.5 | 7.3 | 123 |

That build ran prompts chunk by chunk. [Prompt spans](deepseek-v41.md#prompt-spans) prefill the
8,086 tokens at 906 tok/s with host experts and 390 with disk experts on an RTX 4090 host of the
same type (EPYC 7K62), with the same decode rates.

ISTA-DASLab's expert-pruned Coder build
([artifact](https://huggingface.co/WaveCut/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-NInfer-v3)),
November 4, the sm_89 build of `379c0d999` on an RTX 4090 host with a 64-core EPYC 7B13 (a container
of 61 CPUs): a 25-token code request answered with 256 tokens and a 7,775-token summary request
([the Coder build](deepseek-v41.md#coder-build)):

| Experts | Drafts | Decode, code answer | after 7,775 tokens | Prefill of 7,775 tokens |
|---|---|---:|---:|---:|
| host | none | 37.2 | 39.2 | 1,040 |
| host | `--spec dspark` | 47.9 | 44.3 | 1,020 |
| disk, page cache warm | none | 15.7 | 17.8 | 774 |

DeepSeek-V4.1-Flash has no pipeline stages; several GPUs divide only its routed experts
([expert parallelism](maintainer/pipeline-stages.md#expert-parallelism-of-the-other-moe-families)).
[DeepSeek-V4.1-Flash's measurements](deepseek-v41.md#measurements) add the hosts, the device
cache each card holds, context lookup, startup and what limits each engine.

## Measurement pages

| Page | What it measures | Hardware |
|---|---|---|
| [Reference measurements, September 2026](performance/reference-2026-09.md) | Ternary Bonsai 2 27B and Qwen3.8-27B at full context: decode and prompt time by depth, drafters, largest context per card, needles, several requests at once, what the device profiles change, and the October re-check | RTX 3090, 4090, 5090, PRO 6000 |
| [Tesla V100, October 2026](performance/tesla-v100-2026-10.md) | a `70` build on 16 GB: what fits, engine and server rates of Ternary Bonsai 2, the Qwen3.8-27B releases, Flash-Next and Gemma 4, perplexity against other cards, llama.cpp on the same card | Tesla V100-SXM2-16GB |
| [TITAN RTX, October 2026](performance/titan-rtx-2026-10.md) | a `75` build on one and two 24 GB cards: every model's prompt and decode rates against an RTX 3090, drafters, KV formats, depth, Flash-Next and the 35B-A3B over two stages, the device profile, where the time goes, perplexity | TITAN RTX, RTX 3090 |
| [8, 11 and 12 GB cards, October 2026](performance/small-cards-2026-10.md) | the published image on small cards: which models load, CLI rates, the largest context, Flash-Next with host and disk experts | RTX 3060 Ti, RTX 2080 Ti, RTX 3060 |
| [Qwen3.8-Flash-Next](qwen3-8-flash-next.md#measurements) | every expert placement of the GSQ-RCO releases, MTP, perplexity against llama.cpp, AIME and GPQA | RTX 3090, 3090 Ti, 4090, 5090, PRO 6000, L40S |
| [GGUF block formats](gguf.md#measurements) | the Qwen3.8-27B GSQ-RCO IQ3_S release against the official artifact: perplexity, reasoning benchmarks, speed | RTX 3090, 4090, 5090 |
| [Gemma 4 26B-A4B](gemma4.md#measurements) | against llama.cpp, KV formats, the assistant's drafts, parallel streams | RTX 3090 |
| [Flint](flint.md#measurements) | against llama.cpp, host and disk experts, context lookup, typed decisions | RTX 3090, 4090, 5090, A40 |
| [Field reports](performance/field-reports.md) | users' own measurements: Qwen3.8-27B IQ3_S and its Swift 1.5 fine-tune against llama.cpp, long Qwen Code and agent sessions with DFlash2 and MTP, prefix-cache hits and the time to the first token, Qwen3.8-27B NVFP4 with tensor parallelism over two and four cards | RTX 5070 Ti, 4090, 4080, 5060 Ti |
| [Quality trades](maintainer/quality-trade-experiments.md) | what each precision option costs in perplexity and buys in memory or speed | RTX 3090 |
| [Configuration calculator](config-calculator.html) | which context, KV format and speculation fit an RTX 3090, from measured sizes | RTX 3090 |

The tools behind them: [`tools/bench/refbench.py`](../tools/bench/README.md#reference-measurements-refbenchpy)
for the server tables, the [serving corpus runner](../tools/bench/README.md#serving-corpus-benchmark),
`ninfer_bench` for the Engine without HTTP and the Op benchmarks for single operations
([benchmarks](../bench/README.md)), and `ninfer-perplexity` for quality ([perplexity](perplexity.md)).
An Op benchmark establishes an Op result, not an end-to-end one; a whole request does.

## Determinism

Every configuration reproduces its own output exactly. Configurations do not reproduce each other:
a verification evaluates K+1 columns in one pass where plain decode evaluates one, a batch
evaluates several requests together, and the different reduction orders can flip a near-tied
argmax. MTP and DFlash2 outputs differ from each other and from plain greedy decoding, and DFlash2's
output varies with the draft count, without a measurable perplexity cost
([batch composition](device-profiles.md#batch-composition)).
