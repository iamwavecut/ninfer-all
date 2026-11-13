# NInfer CLI

`build/apps/ninfer` (the Docker image's `ninfer` command) runs one request against one v3
`.ninfer` artifact. Build NInfer or pull the image, and download an artifact, as the
[project README](../README.md) describes before following this guide.

The examples use the official Qwen3.8-27B `groupwise-int` artifact with INT8 KV storage.

## Text input

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Summarize the difference between prefill and decode." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Exactly one of `--prompt` and `--messages` is required. The CLI normally omits `--kv-capacity`, so
the shared Main Text KV pool follows the example's 32,768-token `--max-context`.

Answer content is streamed to stdout. Reasoning, model loading (including the registered target and
canonical `weights_id`), timings, throughput, GPU memory, and speculative-decoding statistics are
written to stderr, so stdout can be redirected independently. FFmpeg's media-decoding messages are
records prefixed `media |`: FFmpeg errors are warnings and everything milder is `debug`.

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Return one sentence." \
  --max-context 4096 \
  --max-new 64 \
  --kv-dtype int8 \
  > answer.txt 2> run.log
```

`--chat-template FILE` overrides the artifact's built-in template with a local Jinja file.
Changes to the file take effect after restarting NInfer:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --chat-template tools/chat_templates/qwen3_8.jinja --prompt "Hello"
```

Omitted thinking and effort options use the selected template's defaults. `--no-thinking` or
`--reasoning-effort none` requests disabled thinking; other effort values cannot be combined with
`--no-thinking`. The template interprets the selected effort; a value the template rejects renders
as the nearest one it accepts (see [serving](serving.md)). `--greedy` selects exact argmax
decoding independently.

`--thinking-budget N` places a positive upper bound on accepted model-origin tokens while the
new-turn Qwen thinking block remains open. If the model has not emitted `</think>` at that exact
boundary, Engine appends [Qwen's canonical early-close guidance](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md)
and `</think>` to the same resident sequence without sampling, publishes the guidance through the
reasoning stream, then resumes ordinary generation from the updated context. A natural thinking
close, stop condition, cancellation, or total output/context limit at the boundary takes priority
and suppresses this insertion. The option cannot be combined with `--no-thinking` or
`--reasoning-effort none`, which both disable thinking, but it can be combined with any other
`--reasoning-effort`.

`--reasoning-loop off|stop|conclude` (default `off`) ends the reply (`stop`) or closes the thinking
with the same control span and lets the model answer (`conclude`) when the open thinking keeps
repeating itself. Every 512 thinking tokens the guard checks three signs: 25 % of the last 2,000
words lie inside a 12-word passage seen three times, the last 2,048 characters repeat one period of
at most 64 characters, or they do so with their numbers masked while every number of the period
keeps its own step and one of them moves (a loop that counts, `'time1','time2',...`; a counter
beside values that change otherwise is an enumeration and does not fire). `conclude` stops
instead when the output left cannot hold the control span and one more token; Gemma 4 and Flint
have no early-close control, so there it always stops. The run summary prints `reasoning loop`
when the guard fires. The option cannot be combined with `--no-thinking`. The check and its
limits are described in [serving](serving.md#openai-chat-completions), under `--reasoning-loop`.

`--max-new` counts every committed generated token, including internally inserted control tokens.
The inserted suffix is never truncated, and a request is never rejected for lacking room for it.
When the output capacity left after the budget cannot hold the complete tokenizer-derived control
suffix plus one post-close model token, Engine lowers the effective budget so both fit. When the
whole capacity is no larger than that, the budget cannot be enforced, so thinking runs to the
`--max-new` limit and may end inside the reasoning. The run summary prints `effective thinking
budget` when it differs from the requested one, and nothing extra when the budget is unenforced; the
exact rule is in [Chat Completions](serving.md#openai-chat-completions).

Normal output sends the inserted guidance to stderr as reasoning. `--print-token-ids` includes the
inserted IDs, while `--raw-output` preserves the raw control representation.

For example, this allows at most 512 model-origin thinking tokens while retaining enough total
output capacity for the inserted suffix and the answer:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Explain speculative decoding, then give a concise conclusion." \
  --max-context 4096 \
  --max-new 1024 \
  --thinking-budget 512 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

## Startup memory profile

GPU residency is frozen when the Engine starts:

- no `--spec` omits MTP/DFlash/DFlash2 weights and state and the optimized proposal head, except
  where the Engine chooses a drafter (Qwen3.8-Flash-Next and Gemma 4 with host experts; see
  `--spec` below);
- `--spec mtp`, `--spec dflash` (35B-A3B), and `--spec dflash2` (Qwen3.8-27B, Qwen3.6-35B-A3B,
  Qwen3.8-Flash-Next) load only the selected speculative backend;
- `--spec dspine --draft-model PATH` loads that drafter artifact on the device of the model's head
  and derives its head-write table from the head;
- a speculative backend loads the artifact's optimized proposal head when it stores one, unless
  `--full-draft-head` keeps the drafts on the full output head and omits it;
- Vision is disabled by default, omitting its weights and Vision-specific unified-workspace extent;
- `--vision` loads the weights, expands the one Program workspace for Vision encode/handoff, and
  enables image/video input.
- the one-request CLI uses root-only context mode, so it does not reserve an extra Device
  checkpoint StateImage or capture a continuation that no later request could consume.

The complete `.ninfer` inventory is still validated. These choices are not lazy loading: an Engine
started without Vision rejects media and cannot enable Vision later. DFlash/DFlash2 and Vision may
be enabled together; these backends apply to generated-text decode after multimodal prefill and does not
accelerate Vision encode. The default speculative and Vision settings produce the smallest resident
profile.

## Structured messages

`--messages` accepts either a non-empty JSON message array or an object containing `messages`
and an optional `tools` array.

```json
[
  {
    "role": "system",
    "content": "Answer concisely."
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image",
        "image": "examples/cli/media/visual_chart.png"
      },
      {
        "type": "text",
        "text": "Describe the chart."
      }
    ]
  }
]
```

Run message files from the repository root when they contain repository-relative media paths:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --kv-dtype int8 \
  --vision \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Supported roles are `system`, `developer`, `user`, `assistant`, and `tool`.
The selected template formats these roles. The maintained Qwen templates keep system/developer
messages at their input positions.

Message content may be a string or an ordered array containing:

| Content type | Source field | Accepted source |
|---|---|---|
| text | `text` | string |
| image / image_url | `image` or `image_url` | local path, HTTP(S) URL, or base64 data URI |
| video / video_url | `video` or `video_url` | local path, HTTP(S) URL, or base64 data URI |

`image_url` and `video_url` may be strings or objects containing a string `url`. The decoded
formats are listed in [media formats](serving.md#media-formats). Assistant
history may include `reasoning_content` and `tool_calls`; a tool result uses role `tool` and
`tool_call_id`.

See [`examples/cli/`](../examples/cli/) for committed text, image, video, mixed-media, thinking,
long-decode, and long-context inputs.

## Speculative decoding

Speculative decoding is disabled by default. Select MTP, the 35B-A3B DFlash or the Qwen3.8-27B
DFlash2 backend with one to fifteen draft positions. Only one backend can be enabled per Engine.
Drafts go through the artifact's optimized proposal head when it stores one: a vocabulary shortlist
(131,072 rows of the 27B's 248,320) that is cheaper to project at every draft step, for the memory
of that matrix (0.3 GiB with the 27B GSQ-RCO IQ3_S artifact). `--lm-head-draft` requires the
proposal head and `--full-draft-head` keeps the full output head; both need a selected backend.
Both masked-draft backends may be combined with `--vision`:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 \
  --max-new 512 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For DFlash:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --kv-dtype int8 \
  --spec dflash --draft-tokens 7 --lm-head-draft
```

For Qwen3.8-27B artifacts containing the DFlash2 companion weights, select
`--spec dflash2 --draft-tokens 7`, optionally with `--lm-head-draft` and `--vision`; this is what
the launchers pass. DFlash2 accepts every draft count from 1 through 15. Both `groupwise-int` and
`nvfp4` artifacts use the same Engine route, including CUDA Graph, concurrent requests, sampling
penalties, and prefix reuse. An artifact without the companion weights reports a missing DFlash2
component when selected. Vision, MTP and DFlash follow the same rule: their weights are required
only when that component is enabled at startup.

A Qwen3.6-35B-A3B artifact converted with
[incoai's DFlash2 drafter](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2)
(`--components ...,dflash2 --source dflash2=PATH`) takes the same `--spec dflash2`. That drafter was
trained on blocks of eight positions, so `--draft-tokens` past 7 asks it for positions it never saw.
On an RTX 5090 with the NVFP4 artifact resident (400 greedy tokens, 2026-11-10) a code answer
decoded 403 tokens per second without drafts, 829 with three MTP drafts and 805, 919 and 892 with
three, five and seven DFlash2 drafts; a prose answer 403, 548 and 465, 420 and 385; the benchmark
corpus (`ninfer_bench`, 128 tokens) 403, 497 and 427, 350 and 455. Draft with it where the output is
code; MTP stays ahead elsewhere.

A [DSpine](maintainer/dspine.md) drafter is an artifact of its own, which `--spec dspine
--draft-model PATH --draft-tokens N` loads beside the model it was trained for; startup refuses a
model of another hidden width, vocabulary or layer count. Tencent's drafter for Qwen3.8-27B is
published converted as
[Qwen3.8-27B-DSpine-b16-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-27B-DSpine-b16-NInfer-v3),
and `python -m tools.convert --model CHECKPOINT --recipe dspine_q8 --out drafter.ninfer` converts
another DSpine checkpoint (`dspine_bf16` keeps every weight in BF16). Every family takes a DSpine
drafter: Qwen3.5-core models and Qwen3.8-Flash-Next with up to 15 drafts a round (its block of 16
less the anchor), Gemma 4, Flint and DeepSeek-V4.1 with up to 7. On Qwen3.5-core models
`--lm-head-draft` takes the drafter's candidates from the artifact's proposal head, as DFlash2's.
On Qwen3.5-core models a sampled request verifies DSpine's
sampled proposal by rejection sampling, as DFlash2's; the other families verify its greedy drafts
with coupled draws, and `--draft-min-p P` verifies a round through its first draft whose confidence
is at or below P. On an RTX 5090 with the official Qwen3.8-27B artifact, 15 DSpine drafts decoded a
1,024-token Python answer at 473.4 tokens per second against 405.1 with 15 DFlash2 drafts and 90.7
without drafts; on shorter code and on prose seven DFlash2 drafts stayed 4-9% faster than DSpine's
best count, whose round costs more ([measurements](performance.md#dspine-on-qwen38-27b)). The best
count depends on the answer and on how many requests share a round; `ninfer-serve --adaptive-mtp`
picks 15 or 7 drafts each round from measured survival and round time
([Adaptive MTP](serving.md#adaptive-mtp)).

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --spec dspine --draft-model models/Qwen3.8-27B-DSpine-b16-ninfer-v3.ninfer --draft-tokens 7
```

A request that samples (temperature above zero) verifies MTP and DFlash drafts with coupled
draws: the model draws each verified position with the request's seed and that position as the key,
exactly as plain decoding does, and keeps a draft only while it equals that draw. Every emitted
token is the model's own sample; the drafts decide only how many tokens one pass emits, not which.
What can still separate a speculative answer from a plain one with the same seed is the
floating-point effect of the verification width, as for greedy decoding. DFlash2 verifies its
retained proposal distribution by rejection sampling.

`--gumbel-sampling` draws each sampled token by Gumbel-max instead: the token whose logit plus
keyed Gumbel noise (keyed by the seed, the position and the token) is largest. The tokens follow the
same distribution, but a seed gives other tokens than the default draw, so the option is off by
default. Drafts from an MTP block (Qwen3.5-core models, Qwen3.8-Flash-Next) and from Gemma 4's
assistant then are the drafter's own draws under the same noise, so a draft and the target's draw
agree wherever the two distributions agree. Whether the target keeps more of these drafts than of
greedy ones depends on the drafter and the temperature: Qwen3.6-35B-A3B's MTP block kept fewer at
0.7 and more at 1.0. DSpark and lookup drafts, and DSpine's outside the Qwen3.5 core, stay
greedy; DFlash2 and DSpine on Qwen3.5-core models keep their rejection sampling for the tokens they
verify. A greedy request drafts and decodes as without the option (on Qwen3.6-35B-A3B, 3 of 3
answers with the same bytes and log probabilities). One request, three prompts, seeds 1 to 3, 256
tokens a response, on an A40 (2026-11-11) and, for
Qwen3.6-35B-A3B, an RTX 5090 (2026-11-12):

| Model and drafter | Temperature | Acceptance, default | Acceptance, `--gumbel-sampling` | tok/s, default | tok/s, `--gumbel-sampling` |
|---|---|---|---|---|---|
| Qwen3.8-27B IQ3_S, MTP | 0.7 | 0.396 | 0.424 | 66.8 | 69.0 |
| Qwen3.8-27B IQ3_S, MTP | 1.0 | 0.396 | 0.407 | 66.6 | 67.4 |
| Gemma 4 26B-A4B, assistant | 0.7 | 0.460 | 0.472 | 204.3 | 205.1 |
| Gemma 4 26B-A4B, assistant | 1.0 | 0.443 | 0.485 | 200.0 | 207.5 |
| Qwen3.8-Flash-Next Q2_0, MTP | 0.7 | 0.404 | 0.418 | 93.6 | 95.5 |
| Qwen3.8-Flash-Next Q2_0, MTP | 1.0 | 0.362 | 0.403 | 88.8 | 93.7 |
| Qwen3.6-35B-A3B NVFP4, MTP (RTX 5090) | 0.7 | 0.459 | 0.443 | 563.7 | 547.4 |
| Qwen3.6-35B-A3B NVFP4, MTP (RTX 5090) | 1.0 | 0.414 | 0.424 | 530.2 | 533.3 |

With the option, Gemma 4's and Flash-Next's speculative answers were the bytes of their plain answers
with the same seed (21 of 21 each); Qwen3.8-27B's and Qwen3.6-35B-A3B's differed as their greedy
ones do, by the verification width's floating-point effect.

### Choosing a draft count

The draft count is a trade on what the output looks like. Each round verifies K+1 columns and runs
K draft-head steps whether or not the drafts survive, so a larger K pays only where the head keeps
guessing right.

- **MTP:** three is the default. It is best or within 2% on prose, and larger counts lose up to 38%
  there. When the output mostly reproduces the input -- refactoring, renaming, applying an edit and
  returning the whole file -- 11 to 15 is up to 1.85x faster than three, and for a coding assistant
  that mostly writes new code seven is about 10% faster. `--lookup-ngram` adds nothing on top of
  MTP there, because the head already copies.
- **DFlash2:** seven is the checkpoint recommendation and the best mean on this card. The best K
  still depends on the workload, so a deployment serving one kind of work should sweep its own.
  The optimized proposal head is within noise of the full head for DFlash2 at every count, so
  `--full-draft-head` can save its memory there.
- **DFlash:** seven forms the measured block length eight; fifteen uses the maximum supported block
  length sixteen.

The tables are in [performance](performance.md#mtp-draft-count-by-content); the
[reference measurements](performance/reference-2026-09.md#draft-length) sweep one to fifteen drafts
of both MTP and DFlash2 on four cards.

## Common options

The table lists executable defaults. The examples above select INT8 KV and MTP3.

| Option | Meaning | Default |
|---|---|---:|
| `--max-context N` | per-sequence logical context ceiling | `2048` |
| `--rope-yarn` | past the model's native window, YaRN at factor `--max-context` / native instead of plain RoPE, for every family: the Qwen3.5 text and MTP layers, Qwen3.8-Flash-Next's sparse attention with its indexer and MTP block, Gemma 4's full-attention blocks (its sliding-window blocks never leave their window); Flint, whose full-attention blocks carry no rotation, and DeepSeek-V4.1, whose compressed blocks carry the release's own YaRN, refuse it. Qwen documents the extension for its models; for Gemma 4 it is the same mathematics without a published or measured quality ([Gemma 4](gemma4.md#running)) | off |
| `--rope-yarn-factor F` | YaRN at a fixed factor in `[1,4]` for every position whatever `--max-context` is; it grows neither the default context nor the KV pool | `1` (as `--rope-yarn` decides) |
| `--rope-scaling-factor F` | instead of YaRN, linear position interpolation for the Qwen3.5 text and MTP layers: a position past `--rope-scaling-original-context` rotates at original + (position - original) / F, so positions up to it keep their exact angles; `[1,32]`, excludes `--rope-yarn` and `--rope-yarn-factor`, and raises neither the four-times-native window cap nor the KV pool. The DFlash adapter keeps its plain RoPE; Qwen3.8-Flash-Next and Gemma 4 refuse it and take YaRN only, Flint and DeepSeek-V4.1 refuse both | `1` (off) |
| `--rope-scaling-original-context N` | the interpolation threshold | the native window |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `2048` |
| `--kv-headroom-mib N` | device memory in MiB that `--kv-capacity auto` leaves free after sizing the KV pool; requires `auto`. `--vram-headroom-mib` is accepted as an alias | `1024` |
| `--prefill-chunk N` | text-prefill chunk, in multiples of 128; with an `int8` or `rk*` KV cache a Qwen3.5-core model runs the nearby multiple of 128 that best fills the GPU's prompt-attention waves, up to 256 tokens more (see [device profiles](device-profiles.md)) | `1024`; `4096` for native MoE experts in host memory or files; `2048` for Qwen3.8-Flash-Next with `--tensor-parallel` when every device has room ([prompt pieces](maintainer/pipeline-stages.md#tensor-parallelism-of-qwen38-flash-next)); `2048` for DeepSeek-V4.1, which takes at most `4096` and a narrower chunk where the card lacks room ([its prompt chunks](deepseek-v41.md#prompt-spans)) |
| `--prefill-tail exact\|full\|N` | DeepSeek-V4.1: the prompt rows that blocks 21 to 39, which keep no compressed cache, run: `exact` runs those the kept state and the last token's logits depend on, from the prompt chunk at or below them, so the output is `full`'s; `full` runs every row; `N` (at least 128) runs the last `N` rows before each kept state, a lossy mode: the output changes ([the prefill tail](deepseek-v41.md#the-prefill-tail)) | `exact` |
| `--max-new N` | requested output-token limit | `128` |
| `--device N` | CUDA device index | `0` |
| `--devices A,B,...` | one pipeline stage per listed CUDA device (2 to 8; the cards may differ where the build runs on each; see [pipeline stages](maintainer/pipeline-stages.md)); on the Qwen3.5 family MTP, DFlash, DFlash2 and resident or CPU Vision run across the stages (the Vision tower stays on the first device; `--vision-residency overlay` is refused); overrides `--device` | none |
| `--stage-layers A,B,...` | layers per stage, in `--devices` order; omitted means a split chosen from each device's free memory | memory-balanced |
| `--tensor-parallel` | the `--devices` share every layer instead of dividing the layers: GDN heads, KV heads with their query heads and cache, and dense FFN rows or MoE experts per device, the head, the shared expert and MTP on the first; Qwen3.5-core models in every weight format, MoE experts on the devices ([tensor parallelism](maintainer/pipeline-stages.md#tensor-parallelism)); Qwen3.8-Flash-Next by GDN key heads, sparse-attention KV heads and GGUF expert banks on the devices ([its tensor parallelism](maintainer/pipeline-stages.md#tensor-parallelism-of-qwen38-flash-next)); Gemma 4 and Flint divide only their routed experts on the devices, DeepSeek-V4.1-Flash its host or disk experts ([expert parallelism](maintainer/pipeline-stages.md#expert-parallelism-of-the-other-moe-families)) | off |
| `--tensor-split A,B,...` | with `--tensor-parallel`, the relative size of each device's share, one positive number per `--devices` entry; shares end on whole stored blocks, so the split is the nearest one the weights allow; on a Qwen MoE divided by whole experts the first device's share includes the shared expert it runs, so it takes fewer routed experts | by each card's measured copy bandwidth (cards of one model take the mean of theirs), capped by its free memory beside the Engine's reservation; DeepSeek-V4.1-Flash: by each card's room for its expert cache and its measured bus speed, so every card's expert misses take the same time |
| `--tensor-attention heads\|positions` | with `--tensor-parallel` on a Qwen3.5-core model, how attention divides: `heads` gives each device a range of KV heads with their query heads and cache planes, the devices past the KV heads holding none; `positions` gives every device every head of the cache pages `p % devices == device`, and the devices merge their partial outputs ([attention by positions](maintainer/pipeline-stages.md#attention-by-positions)) | `heads` |
| `--tensor-parallel-transport auto\|host\|peer-read\|peer-write` | with `--tensor-parallel` on a Qwen3.5-core model or Qwen3.8-Flash-Next, how the devices exchange their partial sums: through pinned host memory (`host`, any cards), by reading or by writing each other's memory (`peer-read`, `peer-write`, which need peer access between every pair of devices), or `auto`: each exchange size takes whichever route was fastest when the engine timed them on these devices at startup, which the startup log lists ([choosing a route](maintainer/pipeline-stages.md#tensor-parallelism)) | `auto` |
| `--tensor-parallel-prompt-partials fp32\|bf16\|int8` | with `--tensor-parallel` on a Qwen3.5-core model or Qwen3.8-Flash-Next, how a prompt chunk's partial sums cross between the GPUs: as computed (`fp32`), rounded to BF16 (`bf16`, half their bytes) or to 8-bit blocks of 32 values with a scale (`int8`, 9/32 of their bytes). Rounding speeds up prefill where host transfers bound it, but rounds each device's partial, so prompt outputs and perplexity differ slightly ([rounded prompt partials](maintainer/pipeline-stages.md#tensor-parallelism)); decode and verification keep FP32 | `fp32` |
| `--invariant-attention` | Qwen3.5-core models: every text attention call of 1 to 16 columns per sequence (64 for one sequence verifying a lookup window longer than 15) reduces its keys in fixed segments of the context, so a column's attention output is the same bytes at any decode width, verification window and batch ([invariant attention](maintainer/qwen3_5-model.md#invariant-attention)); MTP attention keeps its tuned splits | off |
| `--expert-residency device\|host\|disk` | routed MoE experts on GPU, in pinned host memory, or in artifact files: [native Qwen](maintainer/qwen3_5-model.md#routed-expert-residency), [Gemma 4](gemma4.md#expert-residency), [Flint](flint.md#expert-residency), [DeepSeek-V4.1](deepseek-v41.md#expert-residency) and [Flash-Next](qwen3-8-flash-next.md#run). Native Qwen, Gemma 4, Flint and DeepSeek-V4.1 bring misses into whole-layer device banks | `device` |
| `--embedding-residency device\|host` | Qwen3.5 family: the input token embedding in the GPU's weight arena, or in pinned host memory mapped into the GPU, which the gather reads in place over the bus each step. The table's VRAM goes to the KV cache (Qwen3.8-27B: 388 MiB for an IQ2_S table, 1.26 GiB for Q8); output is unchanged. Excludes `--embedding-q4/q6`; refused for an artifact whose embedding is tied to the output head and for the other families. On Windows (WDDM) mapped host memory may still count against the card | `device` |
| `--expert-cache-mib N\|auto` | with host or disk experts, device memory for the most used experts; native Qwen, Gemma 4, Flint and DeepSeek-V4.1 allow `0` and then keep only their layer banks, and Flash-Next disk mode needs a cache. With automatic KV, native Qwen gives the cache priority over KV beyond one request's context | `auto` (what is free after startup) |
| `--expert-misses staged\|mapped` | Qwen3.8-Flash-Next only, GGUF host experts: copy each missing expert to the device while the cached experts run, into a cache slot when the [expert tier](qwen3-8-flash-next.md#the-expert-cache) admits it (`staged`), or let the expert kernels read the missing ones across the bus (`mapped`) | `staged` |
| `--expert-dma-share F\|auto` | Host or disk experts: the GPU's share of a call's cache misses, `0..1`; the CPU computes the rest (GGUF and Qwen3.5/3.6 MoE experts: the GPU's results bit for bit); `auto` splits each such call by the copy and CPU times measured on earlier calls and copies every miss of Flash-Next native experts. Flash-Next host experts' calls of 9 to 64 tokens give the CPU only missing experts with at most eight pairs; wider Flash-Next calls, Gemma 4, Flint and DeepSeek-V4.1 calls of more than eight tokens, and Qwen3.5/3.6 calls at or past the codec's prefill frontier or that route to half the experts or more, always use the GPU ([the CPU share](qwen3-8-flash-next.md#the-cpu-share)); DeepSeek-V4.1's `auto` copies every miss of disk experts ([its CPU share](deepseek-v41.md#the-cpu-share)) | `auto` |
| `--iq1m-experts matrix\|vector` | DeepSeek-V4.1 and Qwen3.8-Flash-Next: IQ1_M routed experts of calls past eight tokens on the integer matrix kernel once a call routes more than 16 rows to each expert on average for their gate and up products and 6 for their down products (the vector kernel below), or always on the vector kernel in groups of eight columns; the two round the sums differently, so outputs differ slightly ([IQ1_M experts](deepseek-v41.md#iq1_m-experts-in-prompt-chunks)) | `matrix` |
| `--expert-tail-skip N` | Qwen3.8-Flash-Next only, GGUF host experts with staged misses: a decode or verification call leaves out each routed expert its device cache lacks when every token of the call ranks it past place `N` (`1..9`) of its top ten; the other experts keep their router weights, so results change | off |
| `--expert-prefill-buffers 1\|2` | Qwen3.8-Flash-Next only, GGUF host experts: device buffers a long prompt's uncached experts are copied into; `2` copies the next layer's while the current layer's experts run, when half the free memory holds both. With the automatic cache the buffers hold expert cache slots between prompts ([the expert cache](qwen3-8-flash-next.md#the-expert-cache)) | `2` with the automatic expert cache, else `1` |
| `--expert-cpu-threads N` | Host or disk experts with a CPU share: `1..256` CPU threads, the calling one included, pinned one to a physical core | `min(physical cores, usable CPUs / 2)`, at least one. Usable CPUs: the process's affinity mask, bounded by a container's CPU quota; physical cores: those in the mask. A 30-CPU container quota gives 15 threads, a 16-core/32-thread desktop 16 |
| `--expert-cache-adaptive` | Qwen3.8-Flash-Next only: replace cold cached experts between calls; changes the CPU/GPU arithmetic partition | off |
| `--expert-profile PATH` | host or disk experts: fill the expert cache from counts recorded for the same artifact and bank geometry | the model family's built-in profile, where the build has one |
| `--expert-profile-out PATH` | host or disk experts: record expert counts for this artifact after requests | none |
| `--expert-mirror DIR` | disk experts: `DIR` on a second drive holds copies of the artifact's files under the same names; reads alternate between the two drives by 1 MiB stripe of the file offset, and results do not change ([disk expert reads](expert-disk-reads.md)) | none |
| `--expert-direct auto\|on\|off` | disk experts on Linux: the NVMe drive writes the experts straight into device memory through NInfer's kernel modules (`kernel/ninfer-direct`); `auto` uses that when the modules are loaded and a self-test read passes, `on` fails startup otherwise; results do not change. Not yet run on hardware ([disk expert reads](expert-disk-reads.md)) | `auto` |
| `--ngram-table PATH` | Qwen3.8-Flash-Next: the n-gram table artifact of a model published without its table, or any table of the model's shape in any row format in place of its own ([another table of the same shape](qwen3-8-flash-next.md#another-table-of-the-same-shape)); DeepSeek-V4.1: its Engram table artifact | the model's own table |
| `--ngram-residency disk\|ram\|ram-hot` | Qwen3.8-Flash-Next: where the n-gram rows come from: the table's file, 16 rows per token; all of the table in RAM; or the rows a hot-row profile ranks first in RAM and the rest from the file (see [the n-gram rows](qwen3-8-flash-next.md#the-n-gram-rows)); DeepSeek-V4.1 reads its Engram rows the same three ways (24 per token in each of its two Engram blocks; [the Engram table](deepseek-v41.md#the-engram-table)) | `disk` |
| `--ngram-io buffered\|direct\|mmap` | Qwen3.8-Flash-Next: how rows are read from the file: through the OS page cache, past it, or out of a mapping | `buffered` |
| `--ngram-io-depth N` | Qwen3.8-Flash-Next: row reads in flight, `1..1024` | `64` |
| `--ngram-dedupe on\|off` | Qwen3.8-Flash-Next and DeepSeek-V4.1, rows read from the file: a pass that addresses one row several times (a repeated n-gram) reads it once and copies it to the others; the rows are the same either way | `on` |
| `--ngram-read-ahead on\|off` | DeepSeek-V4.1, rows read from the file: while a prompt step computes, read the Engram rows of the sequence's next prompt step into a second pinned buffer ([the Engram table](deepseek-v41.md#the-engram-table)); the rows are the same either way | `on` |
| `--ngram-hot-profile PATH` | `ram-hot`: override the selected table's embedded profile with a `ninfer-ngram-profile` file | selected table's embedded profile |
| `--ngram-ram-mib N` | `disk`: budget for cached rows and their index (`0` disables); `ram-hot`: budget for the profile's resident rows and index | `0` for disk; `4096` for ram-hot |
| `--ngram-lock` | `ram`, `ram-hot`: lock the resident rows in physical memory (`mlock`, `VirtualLock`) | off |
| `--no-ngram-table` | Qwen3.8-Flash-Next: run without the n-gram table, a non-standard experimental mode (the Q2_0 release's WikiText-2 perplexity rises from 2.66 to 5.01); DeepSeek-V4.1: without its Engram table (the IQ1_M release's perplexity on a 1,112-token WikiText slice rose from 3.07 to 9.68 on October 25) | off |
| `--kv-dtype bf16\|int8\|fp8\|rk8v4\|rk4v4\|rk4v4-e8\|rk2v4-e8\|nvfp4\|k8v4\|bf16p` | KV-cache storage. `rk8v4` is opt-in RotorQuant, `rk4v4` opt-in Lloyd-Max 4-bit keys, `rk4v4-e8` opt-in E8-lattice INT4 keys and `rk2v4-e8` opt-in E8 root-code keys; those nine are accepted on every build target; `bf16p` is opt-in BF16 packed losslessly into 23% fewer bytes, for Qwen3.5 and Qwen3.8-Flash-Next from sm_75, without `--tensor-attention positions` (see [Context and memory](#context-and-memory)); for Qwen3.8-Flash-Next it is the storage of the sparse-attention layers' KV; for Gemma 4 the sliding-window blocks' K and V, with the full-attention blocks' one value plane in an unrotated coding (see [Gemma 4](gemma4.md#kv-cache-formats)); Flint keeps F16 only (`bf16`), DeepSeek-V4.1 BF16 only | `bf16` |
| `--kv-resident N\|all\|auto` | Qwen3.8-Flash-Next: positions of each sparse-attention layer's KV a sequence keeps in device memory; past them the layers' KV streams from pinned host memory (see [streamed KV](qwen3-8-flash-next.md#streamed-sparse-attention-kv)) | `auto`: 32,768 with host or disk experts, all with device experts |
| `--spec none\|mtp\|dflash\|dflash2\|dspark\|dspine` | speculative backend; Qwen3.8-Flash-Next takes `mtp` from an artifact converted with its MTP block (see [MTP speculative decoding](qwen3-8-flash-next.md#mtp-speculative-decoding)) and `dflash2` from one carrying its DFlash2 drafter (see [DFlash2 speculative decoding](qwen3-8-flash-next.md#dflash2-speculative-decoding)), Gemma 4 from one converted with its assistant (see [Gemma 4](gemma4.md#speculative-decoding-with-the-assistant)); DeepSeek-V4.1 takes `dspark` from an artifact converted with its DSpark drafter (see [the DSpark drafter](deepseek-v41.md#the-dspark-drafter)); every family takes `dspine` with `--draft-model` (see [DSpine](maintainer/dspine.md)); Flint has no drafter of its own and takes `--lookup-ngram` without `--spec`, as DeepSeek-V4.1 does without `dspark` | for Flash-Next artifacts with host experts `dflash2` with 4 drafts (with `--draft-min-p 0.7`) on one device when they carry the DFlash2 drafter, else (over pipeline stages too) `mtp` with 3 drafts (with `--draft-min-p 0.5`) when they carry the MTP block, a round of several requests with a CPU share verifying at most 64 tokens ([MTP with several requests](qwen3-8-flash-next.md#mtp-with-several-requests)); for [Edge0](edge0.md#drafting) artifacts also with device experts and `--max-concurrency 1`; `mtp` with 3 drafts for Gemma 4 artifacts with their assistant and host or disk experts; otherwise off |
| `--draft-model PATH` | DSpine only: the drafter artifact, trained for the model | unset |
| `--draft-tokens N` | `1..15` for MTP, DFlash and DFlash2; `1..7` for Qwen3.8-Flash-Next's DFlash2 drafter (its block of 8), Gemma 4's assistant and the context lookup of Flint and DeepSeek-V4.1; `1..5` for DSpark, its block; for DSpine up to its block less one and at most 15, or 7 on Gemma 4, Flint and DeepSeek-V4.1 | unset (their lookup: 3; DSpark: 5) |
| `--draft-min-p P` | Flash-Next MTP and DFlash2, and DSpine on every family but the Qwen3.5 core: verify a round's drafts through the first at or below this probability (MTP: its absolute probability, which also ends the drafting; DFlash2 and DSpine: the drafter's confidence); see [MTP](qwen3-8-flash-next.md#mtp-speculative-decoding), [DFlash2](qwen3-8-flash-next.md#dflash2-speculative-decoding) and [DSpine](maintainer/dspine.md#verification-by-family) | `0` (off) with `--spec`; `0.5` when the engine chooses MTP and `0.7` when it chooses DFlash2 (no `--spec`), unless set |
| `--draft-experts N` | DSpark only: each drafter block keeps its `N` most routed experts on the GPU and routes among them, the rest in pinned memory; `0` keeps all ([details](deepseek-v41.md#the-drafters-experts)) | `64` with the STQ drafter and host or disk experts, otherwise all |
| `--draft-confidence C` | DSpark only: verify a round's drafts up to the first its confidence head scores below `C` (`0..1`); `0` verifies every draft | `0.5` |
| `--lm-head-draft` | require the optimized proposal head | used when the artifact stores it |
| `--full-draft-head` | draft through the full output head even when the artifact stores a proposal head | off |
| `--mtp-attention-window N` | MTP only: the draft head attends to the first 64 keys and the newest `N` before its query, verification keeps full attention (see [serving](serving.md#mtp-attention-window)) | `0` (whole history) |
| `--lookup-ngram N` | context-lookup drafting alongside `--spec` (Flint and DeepSeek-V4.1 without it; DeepSeek-V4.1 refuses it beside `--spec dspark`): the last `N` tokens are matched against the sequence so far and what followed is proposed, from the occurrence the sequence was already copying while it still matches, else from the newest; exact, since verification rejects a wrong guess | `0` (off) |
| `--ngram-draft-tokens N` | copy drafting alongside `--spec`: up to `N` tokens (1..63; with DFlash, DFlash2 or DSpine at least `--draft-tokens`) copied from earlier prompt, tool-result or output text that the last `--ngram-min-match` tokens match, verified by the target; `0` disables it; see [Ngram copy proposals](ngram.md) | `15` with `--spec`, else `0` |
| `--ngram-min-match N` | shortest match a copy is drawn from, `4..64` | `12` |
| `--prefill-cublas` | hand wide prefill GEMMs to cuBLAS: a large prefill speedup for a small perplexity cost, and it wants a larger `--prefill-chunk` to pay (see [quality trades](maintainer/quality-trade-experiments.md)) | off |
| `--nvfp4-w4a8-int` | run NVFP4 weights on the int8 tensor cores (W4A8), prefill and decode alike: activations are quantised to int8 with one scale per token and 32 columns, the stored E2M1 codes widen to int8 exactly, and their E4M3 group scales apply in FP32. About 1.2-1.4% relative error from the activation quantisation; on an RTX 50-series card it replaces the W4A4 routes. The device profile's `nvfp4_int8/*` keys choose the wide kernel | off: BF16 activations (W4A16) before Blackwell, W4A4 on it |
| `--nvfp4-w4a16` | on an RTX 50-series card, run NVFP4 weights on the dequantising BF16-activation routes (W4A16) instead of W4A4, more precise and slower; other cards run them by default, so there it changes nothing. Wins over `--nvfp4-w4a8-int` | off |
| `--no-prefill-cublas-projections` | with `--prefill-cublas`, keep the attention and GDN input projections off that route | projections on |
| `--lm-head-q4`, `--lm-head-q6` | store the output head as Q4 or Q6 while loading | off |
| `--embedding-q4`, `--embedding-q6` | store the token embedding as Q4 or Q6 while loading | off |
| `--mtp-experts-q4` | Qwen3.6-35B-A3B: store the MTP layer's routed experts in the text layers' formats | off |
| `--gdn-state-fp16` | keep the GDN recurrent state in FP16 | FP32 |
| `--mlp-a8-decode` | integer-activation MLP gate_up at decode and verify widths | off |
| `--no-prefill-a8` | return full prefill tiles to their A16 routes, which is how the integer routes are measured on a whole request | integer routes |
| `--vision` | enable image/video input and load Vision GPU allocations | off |
| `--vision-residency resident\|overlay\|cpu` | `overlay` keeps the Vision tower host-pinned and borrows device memory per image from the evictable text weight tail (no resident Vision cost; needs CUDA VMM); for Gemma 4 the tower's blocks stay pinned and stream through a two-block device window. `--vision-offload on\|off` is accepted as an alias for `overlay\|resident`. `cpu` encodes on CPU threads from host FP32 weights, with no device Vision memory and `--vision-max-merged` capped at 256 unless given. Qwen3.8-Flash-Next keeps its tower resident; DeepSeek-V4.1 keeps it resident or, with `overlay`, streams its blocks from pinned memory through a two-block device window, and refuses `cpu` ([DeepSeek-V4.1](deepseek-v41.md#images)) | `resident` |
| `--vision-cpu` | `--vision` with `--vision-residency cpu` | off |
| `--vision-max-merged N` | merged-token budget of one media item; larger media downscales at preprocessing. Gemma 4: the most soft tokens an image or video frame may take, rounded down to its budgets 70, 140, 280, 560, 1120 (an image takes 280, 70 with `detail: low` and this bound with `detail: high`; a video frame 70). DeepSeek-V4.1: the most prompt positions an image's span takes, at most the release's 1024 (a quarter with `detail: low`) | 16384 |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--gpu-clock-floor MHZ` | hold the graphics clock of the model's GPUs at MHZ or more while requests run and for 2 s after the last one (NVML locked clocks; an idle card then returns to the driver's clocks and idle power), so the stages of a layer pipeline do not wait at idle clocks (four RTX 3060 under Windows idled at 210 MHz between stages; two RTX 4090 under Linux kept their boost clocks without it, and decode stayed within 0.6 %). Giving the clocks back resets the cards' locked clocks, including a lock set before the start; a killed process leaves them locked until `nvidia-smi -rgc`. Needs administrator rights on Windows and root on Linux (a container usually lacks them); without them the start reports it and keeps the driver's clocks | off |
| `--wddm-evictable-budget` | Windows builds: budget the KV cache against dedicated memory less the weights and a 512 MiB desktop floor, committing the device arenas at startup; another process taking memory can then slow generation | off |
| `--chat-template FILE` | use a local Jinja template | artifact template |
| `--no-thinking` | disable thinking | template default |
| `--thinking-budget N` | positive model-origin thinking-token cap; omitted means unlimited | unset |
| `--reasoning-loop off\|stop\|conclude` | end the reply or close the thinking when it keeps repeating itself | `off` |
| `--reasoning-effort none\|minimal\|low\|medium\|high\|xhigh\|max` | pass an effort value to the selected template | template default |
| `--greedy` | exact argmax decoding | off |
| `--post-thinking` | sample the answer with the post-thinking preset (temperature `0.2`) from the token after the reasoning block closes | off |
| `--post-thinking-temperature F`, `--post-thinking-top-p F`, `--post-thinking-top-k N` | post-thinking overrides; each implies `--post-thinking` | preset |
| `--post-thinking-sampler temp=F,top_p=F,top_k=N[,min_p=F,presence=F,frequency=F]` | the same overrides in one flag | preset |
| `--temperature F` | sampling temperature override | registered model/mode default |
| `--top-p F` | nucleus-threshold override | registered model/mode default |
| `--top-k N` | top-k-threshold override (`0..20`; zero selects the top-20 cap) | registered model/mode default |
| `--min-p F` | min-p-threshold override | registered model/mode default |
| `--presence-penalty F` | presence-penalty override | registered model/mode default |
| `--frequency-penalty F` | frequency-penalty override | registered model/mode default (`0`) |
| `--seed N` | sampling seed | `0` |
| `--gumbel-sampling` | draw sampled tokens by Gumbel-max over keyed noise, and MTP-style drafts as the drafter's own draws under that noise ([coupled draws](#speculative-decoding)); a seed gives other tokens than the default draw | off |
| `--lora FNAME[,...]`, `--lora-scaled FNAME:SCALE[,...]` | llama.cpp LoRA adapters that remove a residual direction, at scale 1 or the given scale ([adapters and control vectors](serving.md#lora-adapters-and-control-vectors)) | none |
| `--lora-init-without-apply` | load the adapters at scale 0 | off |
| `--control-vector FNAME[,...]`, `--control-vector-scaled FNAME:SCALE[,...]` | llama.cpp control vectors added to the residual, at scale 1 or the given scale | none |
| `--control-vector-layer-range START END` | add only the control vectors of layers START to END | every layer a file names |
| `--log-colours on\|off` | `on` gives every statistic of the stderr summary a stable colour, even when stderr is redirected | off |
| `--log-level trace\|debug\|info\|warning\|error\|critical\|off` | stderr verbosity | `info` |

[Quality trades](maintainer/quality-trade-experiments.md) measures what each precision option costs.

When a sampling flag is omitted, Engine selects the general-task preset for the loaded architecture
and rendered prompt mode:

| Model | Prompt mode | Temperature | Top-p | Top-k | Min-p | Presence penalty |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B, Qwen3.8-27B and the artifacts built from it | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.6-27B, Qwen3.8-27B and the artifacts built from it | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | thinking | `1.0` | `0.95` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.8-Flash-Next | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.8-Flash-Next | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |

Frequency penalty is `0` in every registered preset. Task-specific profiles such as Qwen's
precise-coding profile use explicit sampling overrides. `ninfer-serve` drops the presence penalty
for a request that offers tools ([HTTP serving](serving.md#server-options)).

Repeat `--stop-token-id`, `--stop`, or `--reasoning-stop` to add stop conditions. Use
`--raw-output` to expose the frontend's raw output stream and `--print-token-ids` to include
generated token IDs in diagnostics.

Run `./build/apps/ninfer --help` for the exact option contract.

## Launch configuration files

`--config FILE` reads a run's options from a file. `ninfer`, `ninfer-serve`, `ninfer-perplexity`,
`ninfer_bench`, `ninfer-calibrate` and `ninfer-ngram-profile` take it, and the repository's
`configs/` directory holds a file for each [launch recipe](recipes.md).

```ini
; Ternary Bonsai 2 27B on a 24 GB card
model          = models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer
model-id       = bonsai2-27b
max-context    = 262144
kv-capacity    = 262144
kv-dtype       = rk8v4
gdn-state-fp16 = true
spec           = dflash2
draft-tokens   = 5
```

```bash
ninfer-serve --config configs/bonsai2-27b-dflash2.ini
ninfer-serve --config configs/bonsai2-27b-dflash2.ini --max-context 131072 --port 9000
```

The format is llama.cpp's preset format, which [`--models-preset`](serving.md#several-models-router)
also reads:

- Each line is `key = value`. The key is the option without its leading dashes, in any case. The
  value is the text after the first `=`; matching quotes around it are removed.
- `true` is the switch itself (`gdn-state-fp16 = true` is `--gdn-state-fp16`), and `false` is its
  `--no-` form. Every other value, `off`, `0` and `auto` included, is the option's value.
- A repeatable option (`stop`, `lora`, `control-vector`) takes one line per value.
- Blank lines and lines that start with `;` or `#` are comments.
- `model` (or `artifact`) names the artifact; an artifact path on the command line wins.
  `cuda-backend` selects the portable package's CUDA backend. `version` is ignored.
- A file has no `[section]` and no `config` key. `ninfer_bench` names its artifact with `weights`.
- Relative paths start in the directory the program runs in, not in the file's directory.

`--config` can repeat. The files apply in the order given and the command line applies last, at any
position of `--config` on the command line. A later layer overrides an earlier one by these rules:

- An option replaces every earlier option of the same setting: a value, a repeatable option's whole
  list (one `--stop` on the command line drops every `stop` line of a file), a switch in either
  form, and the other spellings of the setting (`--device` replaces a file's `devices`,
  `--full-draft-head` its `lm-head-draft`).
- An option that contradicts an earlier layer's option drops that option: `--spec none` also drops
  a file's `draft-tokens` and `adaptive-mtp`, `--spec mtp` its `draft-model`, `--no-thinking` drops
  `thinking-budget`, and `--no-vision` drops `vision-residency`. The same contradiction inside one
  layer is an error.
- Two choices stay errors across layers: `--device` with `--devices` inside one layer, and in
  `ninfer-serve` a hybrid prefix cache option (`--device-snapshot-slots`, `--cache-taps-per-request`,
  `--cache-tap-ladder`, `--cache-tap-min-gap`, `--prefix-cache-file`) together with an option that
  selects the checkpoint catalog (`--use-original-prefix-caching`, `--no-prefix-reuse`, `--devices`
  with several devices, `--graft`).

Every switch of every program also takes a `--no-` form, so the command line can turn off a switch
that a file turns on. `ninfer-serve` records the merged options, with the API key redacted, as
`arguments` in the request log's `server_start` record. In router mode the router's files and
command line are layers above each model's preset sections.

The container image keeps the files in `/opt/ninfer/configs`. Its working directory is `/models`;
`-w /` makes a file's `models/` paths reach the mounted `/models`:

```bash
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 -w / \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/iamwavecut/ninfer-all:cuda13.4 serve --config /opt/ninfer/configs/bonsai2-27b-dflash2.ini
```

The container's address (`NINFER_HOST`, `NINFER_PORT`) is the first layer, so a file's or the command
line's `host` and `port` override it.

## CUDA synchronization

`NINFER_CUDA_SYNC` selects the CUDA device synchronization schedule at startup for both the CLI
and HTTP server, on every device the Engine uses. When unset, CUDA's own default applies (`auto`), a
heuristic that spins while the host has more cores than active CUDA contexts. `spin` always
spins, trading one busy core for the lowest synchronization latency. Use `blocking` to let the
waiting thread sleep; the decode performance cost depends on the host. `yield` yields the CPU
while waiting.

```bash
NINFER_CUDA_SYNC=blocking ./build/apps/ninfer models/qwen3_8_27b.ninfer --prompt "Hello"
```

The Engine-ready log reports the selected mode. Empty or unrecognized values, or failure to apply
the schedule, fail startup. This controls device scheduling (including stream synchronization);
it does not override individual CUDA event creation flags.

## Device stall watch

A device that stops finishing work raises no CUDA error, so the CLI, the HTTP server and the tools
end the process instead: when a wait of any thread for the device, or a probe the process submits
to every device it uses every 5 s, lasts 120 s while no host service answers the device ([Stop the
server](serving.md#stop-the-server) shows the messages). `NINFER_DEVICE_STALL_SECONDS` sets that
limit in whole seconds, and `0` turns the waits' watch and the probes off. Set it under tools that
slow the device many times over, such as Nsight Compute and compute-sanitizer, where one wait can
last minutes:

```bash
NINFER_DEVICE_STALL_SECONDS=0 ncu --set full \
  ./build/apps/ninfer models/qwen3_8_27b.ninfer --prompt "Hello"
```

A value that is not a whole number of seconds fails startup.

## Context and memory

The registered model IDs have a native context limit of 262,144 tokens. `--max-context` accepts up
to four times that, 1,048,576 tokens, with plain RoPE or YaRN (`--rope-yarn`) past the native
window. What fits depends on the card, the artifact, the media workload, the output budget and the
KV format. The artifact describes its model configuration and weight representations;
`--kv-dtype` independently selects runtime KV storage, BF16 by default. Nine formats — `bf16`,
`int8`, `fp8`, `rk8v4`, `rk4v4`, `rk4v4-e8`, `rk2v4-e8`, `k8v4`, `nvfp4` — are accepted on every
build target, and the tenth, `bf16p`, from sm_75 on: the Blackwell-only `mma.sync...kind::f8f6f4` restriction applies to FP8/NVFP4
*weights and activations*, not to KV storage. Measured size, decode speed and perplexity of seven of
them on an RTX 3090 are in [performance](performance.md#kv-formats-on-qwen38-27b) and the
[configuration calculator](config-calculator.html).

`rk8v4` is the best all-round choice: rotated INT8 keys with a packed signed int4 value plane,
about 23% smaller than INT8 for about 0.082% perplexity, and the flattest decode curve of any
format measured. `nvfp4` buys the most context — 45% smaller than INT8 — at about 13% of decode
speed at a 32K cache depth. `rk4v4` keeps `rk8v4`'s values and stores keys as 4-bit Lloyd-Max
indices: 31% smaller than `rk8v4`, within 3% of `nvfp4`'s size, better perplexity than `nvfp4`
(+0.21% against INT8) and `rk8v4`'s decode speed, so it is the choice when context is the limit.
`rk4v4-e8` has `rk4v4`'s size but snaps each octet of a scaled G64 key group to the nearest E8
lattice point before the codes are clamped to [-8, 7]; the coset bit is not stored, so per-value
error is no better than plain INT4. `rk2v4-e8` stores each 8-dimension key block in two bytes, the
nearest of E8's 240 roots and a byte holding a log-radius and a residual axis: 216 bytes per token
and KV head against 280 for `rk4v4-e8` and 408 for `rk8v4`, the one format that holds 1,048,576
tokens beside Ternary Bonsai 2 on a 24 GB card, at a quality cost (quick-corpus perplexity 5.631 to
5.820 on Bonsai 2, and fewer accepted DFlash2 drafts). `fp8` and `k8v4` are each beaten by `rk8v4`
on size, speed and quality together, so neither has a niche.

`bf16p` keeps every BF16 bit in 23% fewer bytes: a K or V row of 256 values takes 392 bytes instead
of 512, 784 bytes per token and KV head against 1,024, so the answers equal `bf16`'s token for
token. The form is Infernix's: each value keeps its sign and seven fraction bits, and a 4-bit offset
from the row's 14-binade exponent window; up to six exponents outside the window go into an 8-byte
side record. A row that needs more has no packed form: it is stored as NaN, and its request ends as
one with non-finite logits (none did in the runs below). Qwen3.5 models and Qwen3.8-Flash-Next take
it; Gemma 4 (an F16 cache), Volta and `--tensor-attention positions` refuse it, and Flint and
DeepSeek-V4.1 keep their own caches. On an RTX PRO 6000 Blackwell Server (2026-11-10, 256 greedy
tokens, two runs each, the same tokens in every pair), Qwen3.8-Flash-Next UD-IQ2_XS with its experts
on the GPU decoded as fast as under `bf16` at a 30,216-token prompt (111.4-113.6 against
110.9-113.4 tok/s) and prefilled 1-4% faster. Ternary Bonsai 2 decoded 12.5% slower at a
6,797-token prompt (121.8 against 139.2 tok/s), 24% slower at 30,216 tokens (91.9 against 120.8)
and 13% slower with MTP (223.5-223.8 against 255.9-257.4), and prefilled up to 18% slower (5.85k
against 7.09-7.16k tok/s at 30,216 tokens): its dense attention kernels decode each packed tile
into shared memory before the tensor-core products, and took 2.1 (prompt) and 2.5 (decode) times
as long per call as with BF16 tiles. On an RTX 4090 the same Bonsai runs decoded 6.7% slower at
6,797 tokens (93.8 against 100.4-100.6 tok/s), 12.6% slower at 30,216 (75.1 against 85.9-86.0) and
3.3% slower with MTP (176.4 against 182.4-182.5), and prefilled 5-14% slower. The prepared prompt
must fit
`--max-context`; generation stops at the remaining context capacity when necessary.
`--kv-capacity N` controls the shared physical Main Text KV pool independently and is rounded up to
the 64-token page size. `--kv-capacity auto` loads the selected weights, measures the remaining GPU
memory, and directly chooses the largest legal page capacity for the complete enabled runtime
layout. This includes the selected speculative backend, fixed sequence state, unified workspace,
and CUDA Graph allowance, while leaving the default 1 GiB automatic headroom
unallocated. It does not probe allocations or resize the pool at request time. The single-request
CLI normally leaves the option omitted so it follows
`--max-context`; the distinction matters primarily to a concurrent Engine or server.

At Engine startup NInfer reserves model weights, persistent sequence state, one phase-reused
Program workspace, and a separate CUDA Graph driver allowance. With Vision enabled, that one
workspace contains a general execution prefix and a fixed item-output handoff region. Vision encode
may reuse the full backing before producing the output; Text/MTP/decode work remains inside the
general prefix while the handoff is live. The capacity is therefore the maximum legal simultaneous
extent, not the sum of Text, Vision scratch, and Vision output allocations. Text prefill uses
`min(--prefill-chunk,--max-context)`; Vision keeps the existing 32,768-token aggregate prompt budget
but plans Device execution for the registered 16,384-token maximum single item. Requests perform no
project-owned device allocation or growth. Context-cache capacity controls are intentionally absent
from this one-request interface; the persistent Engine and server routes own cross-request reuse and
optional Host backing.

All weight, sequence, workspace, and graph allocations are released when the Engine is destroyed.

## JSON and JSON Schema output

`--json` constrains output to a JSON object. `--json-schema FILE` constrains it to the supported
JSON Schema subset described in [serving](serving.md#structured-output). Thinking defaults off;
an explicit `--reasoning-effort` or `--thinking-budget` retains reasoning before the constrained answer.
The two format flags are mutually exclusive and reject raw output and custom stops. The CLI adds
the requested format/schema to the model's instructions before tokenization. They work with
ordinary, MTP, DFlash and DFlash2 execution.

```bash
./build/apps/ninfer model.ninfer --prompt 'Return the city as JSON' --json --max-new 128
./build/apps/ninfer model.ninfer --prompt 'Return a weather record' --json-schema weather.schema.json --max-new 128
```

Read the reported finish reason: an output/context limit or cancellation can truncate the JSON.
