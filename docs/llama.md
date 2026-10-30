# Dense Llama and Nanbeige4.3-33B

The explicit `LlamaForCausalLM` family runs Nanbeige4.3-33B through the public `.ninfer` Engine.
It has its own config, operand bindings and executor in `src/models/llama`, and a runtime core in
`src/runtime/engine/llama_core`. The shared frontend implements Nanbeige's Metaspace BPE, official
ChatML template, reasoning channels and XML function-call syntax.

The admitted mathematics is bias-free dense Llama with an untied output head, SiLU SwiGLU,
128-dimensional heads, unscaled full-head RoPE, and no Q/K norm, attention gate or softmax sink.
Nanbeige has 52 layers, hidden width 7168, FFN width 22016, 56 query heads, 8 KV heads and
RoPE theta 120,000,000. Its native config permits 524288 positions; this port was qualified at
262144. Other architectures, RoPE scaling, vision and draft-model backends are rejected.
One GPU and one active sequence are supported. Prompt chunks are bounded at 512 tokens; optional
context lookup can verify up to seven drafts. Streamed execution is eager rather than captured
in CUDA Graphs.

## Convert

Obtain the official config/tokenizer files from
[Nanbeige/Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B), and a GGUF from
[AtomicChat/Nanbeige4.3-33B-GGUF](https://huggingface.co/AtomicChat/Nanbeige4.3-33B-GGUF).
The tested source is `Nanbeige4.3-33B-AD-IQ3_XXS.gguf` at revision
`560e151e11dfdd4b008d394f6e16bf196a952bd0`; config/tokenizer revision
`df919e1e2cf077a22204c78b6c826bf3db18bc08`.
Extract `tokenizer_config.json`'s `chat_template` into `chat_template.jinja` beside those files.
No HF safetensors are needed for this recipe.

```sh
python3.11 -m tools.convert --model /models/nanbeige-hf --recipe llama_gguf \
  --source gguf=/models/Nanbeige4.3-33B-AD-IQ3_XXS.gguf \
  --out /models/Nanbeige4.3-33B-IQ3_XXS.ninfer --device cpu
```

Matrices retain their GGUF blocks without requantization. GGUF Llama stores Q/K rotary pairs
interleaved; the converter moves complete encoded rows into NeoX half-head order. No coefficients
are decoded/re-encoded for this permutation. V and the other matrices keep their row order. Norms
widen exactly to FP32. The loader requires the artifact's explicit `qk_layout: neox` contract.

## Run on a V100 16 GB

```sh
CUDA_VISIBLE_DEVICES=0 build/apps/ninfer-serve /models/Nanbeige4.3-33B-IQ3_XXS.ninfer \
  --model-id nanbeige4.3-33b --host 0.0.0.0 --port 8000 \
  --max-context 262144 --kv-capacity 262144 --max-concurrency 1 \
  --prefill-chunk 512 --host-cache-mib 512 --no-cuda-graph
```

The dynamic IQ3_XXS source is 13.9 GiB; individual tensors retain IQ3_S, IQ3_XXS, Q5_K, Q6_K
and other stored formats rather than receiving one uniform three-bit codec.
A 262144-token F16 KV cache needs **52 GiB of system RAM**, plus weights, staging and runtime
resources. Reserve enough RAM for that cache independently of `--host-cache-mib`, which bounds
additional prefix images. Active KV uses pageable host pages allocated as the sequence grows;
an initial prefix sized to available VRAM remains cached on the GPU, retaining 256 MiB
of planning headroom. The other rows are gathered into bounded 8192-token
pinned transfer tiles and copied to the GPU, where all attention arithmetic runs. Partial tiles transfer only
their populated rows. Query subtiles
bound workspace independently of transfer size and projection chunk size. Full-prefix snapshots
retain a frontier; saved images contain the represented F16 bytes.

This makes the context fit in RAM without allocating a context-sized GPU buffer. It does not make
long-context decoding fast: PCIe must carry almost the entire KV cache for each generated token
at a full window. Short-prompt rates cannot be extrapolated to 256k. Full-prompt prefill and
long-context retrieval quality require separate measurements; the capacity test below supplies an
explicit synthetic cache and does not stand in for them.

## Qualification

Tested on Tesla V100-SXM2-16GB, CUDA 12.9, `sm_70`, Release build, Python 3.11.16.

- `pytest tests/convert/test_llama.py`: exact encoded-row permutation/storage and rejection of
  different mathematics. The full 52-layer artifact was also converted and loaded.
- `ninfer_llama_frontend_test`: independent Hugging Face tokenizers fixtures, plus the real release's
  vocabulary; initial spaces, literal Metaspace markers, Unicode byte fallback and added tokens.
- `ninfer_dense_decoder_test`: represented-input FP64 oracles for RMSNorm at width 7168, NeoX
  rotation through position 262143, F16 K/V rounding, GQA 56/8, causal masks and segment merging.
  The full 262144-key softmax is checked against a periodic-cache multiplicity oracle in FP64.
- `ninfer_llama_real_test ARTIFACT logits IDS.json OUT.f32`: full-model logits. Against llama.cpp
  on the same GGUF and identical 20-token input, raw RMS error was 0.0842, max error 0.3304,
  reference-to-native KL 0.0000566, same top token, 19/20 top-token overlap. At 1537 tokens,
  across the resident/streamed cache boundary, raw RMS error was 0.3714, max error 0.9098,
  KL 0.00000250, same top token and 19/20 overlap. These two inputs are supplementary model
  parity checks, not a general quality benchmark; the activation/accumulation paths differ.
- `ninfer_llama_real_test ARTIFACT state IDS.json`: snapshots, host-image reload and independent
  branches preserve exact BF16 continuation logits across position 1024.
- `ninfer_llama_real_test ARTIFACT capacity`: backs 55,834,574,848 bytes of host KV, executes the
  last legal position of the 262144-token window with finite logits, and rejects the next token.
  In the 8192-token transfer-tile implementation, that one synthetic full-cache token took
  19.35 seconds (about 0.052 tok/s); host-cache population plus that call took 63.70 seconds.

A first 53-token code request generated 128 tokens at 33.0 tok/s with 128-token prompt chunks.
This is one short-context observation. The llama.cpp reference's 4k configuration gave 34.2 tok/s
on a related request, and its 256k CPU-KV configuration gave 25.5 tok/s on a short request but
failed from CUDA allocation pressure on a larger prompt. Those requests have different template
and prompt-token counts and do not establish a speed comparison.

With 512-token prompt chunks in the deployed 256k configuration, the same 53-token code request
produced 128 tokens at 33.07 tok/s (prefill 162.5 tok/s). A separate repeated-reference retrieval
prompt of 11542 tokens returned the correct identifier: prefill 143.94 tok/s, six output tokens
at 1.11 tok/s, 84.71 seconds total. This is a simple retrieval check, not a broad quality test.
The API also returned a parsed `read_file` tool call with the requested path. The slowdown with
context is material: the bounded transfer path trades GPU capacity for host bandwidth and eager
attention/transfer overhead. It has no measured long-context speed advantage over another
optimized implementation.

The deployed server OpenCode provider uses `http://127.0.0.1:8000/v1`, selects
`ninfer/nanbeige4.3-33b`; the original profile declared context 262144/output 8192, while
the deployed fast profile declares context 8192/output 2048. An actual OpenCode run called
its `read` tool on a local fixture and returned `NINFER_TOOLCHECK_OK` (exit 0). The standalone
launcher is [scripts/start-nanbeige-v100-256k.sh](../scripts/start-nanbeige-v100-256k.sh); on the
server it is also installed as `/workspace/start-nanbeige-256k.sh`. The existing model router
uses its own launcher and loads Nanbeige on startup while retaining the two Qwen choices.

## Fast interactive profile

```sh
sh scripts/start-nanbeige-v100-fast.sh
```

This profile uses max-context/kv-capacity 8192, concurrency 1, prefill chunk 512, host prefix-image
budget 2048 MiB, context lookup with a four-token match and three automatic drafts, no CUDA Graphs,
thinking disabled by default and default output budget 2048. A request can explicitly enable
thinking. Disabling reasoning changes the default answering policy, not the underlying decode
rate; both sides of the measurements below explicitly disabled it. The weights remain the same
IQ3_XXS artifact without further quantization.

The original executor held exactly 1024 KV positions on GPU and copied the entire 8192-position
staging allocation even for a partial tile. The optimized executor fits whole 1024-position pages
to free VRAM after allocating its bounded workspace, reserving 256 MiB at planning. Attention
uses the actual resident stride, image reload restores every resident page, and a 2D transfer
uploads only the populated rows of each K/V head. The deployed V100 selected 4096 resident
positions. This raised device KV from 208 MiB to 832 MiB; startup reported 377.5 MiB free. The
prefix-image RAM budget rose from 512 MiB to 2 GiB, separately from active KV (at most 1.625 GiB
at 8k). These resource costs buy less PCIe traffic and longer reusable OpenCode prefixes.

Measurements on the same V100, artifact, greedy sampling/seed, cache reuse disabled, code/prose
outputs truncated at 160 tokens; two runs per short workload:

| Workload | Original fixed-1024 KV, no lookup | Final fast 8k profile |
| --- | --- | --- |
| LRU-cache code, 57-token prompt | 32.32 / 32.32 tok/s | 32.30 / 32.32 tok/s |
| Russian HTTP-cache explanation, 70-token prompt | 31.79 / 31.78 tok/s | 31.49 / 31.50 tok/s |
| Repeated square functions | 31.92 / 31.90 tok/s | 51.30 / 51.35 tok/s |
| 2048-token reference continuation, 32 output tokens | 4.44 tok/s | 36.86 tok/s |
| 6144-token reference continuation, 32 output tokens | 2.56 tok/s | 13.28 tok/s |

Every measured greedy output matched the baseline exactly. The context-continuation prompts are
repetitive and favor lookup; their gains cannot be generalized to arbitrary prose at the same
length. The code/prose and reference-continuation cases use different input shapes and output
lengths, so compare each row only with its own baseline. The ordinary prose slowdown of about
0.9% remains visible; the profile was selected for its substantial repeat/code and resident-prefix
gains, not a universal TPS improvement. No stable 60+ tok/s claim is supported.

A residency-only trial with 3072 GPU positions (a second idle server retained a CUDA context)
gave 24.27 tok/s at 2k and 4.76 tok/s at 6k; adding lookup gave 37.10 and 10.13 tok/s. That was a
selection trial, not an alternative final baseline. Explicit --draft-tokens without a separate
--spec backend is rejected by the existing CLI; this profile uses lookup's automatic three drafts.

Expanded-cache qualification passed exact snapshot/image/branch continuation checks across
positions 1024, 3072 and 4096. The 1537-token full-model logits retained the earlier reference KL
2.50e-6 and the same top token. The fast-capacity test backed 1744830464 host-KV bytes, executed
position 8191 with finite logits in 0.550 s (synthetic cache, plain decode), and rejected position
8192. The original full 256k capacity result above predates this residency change; a complete real
256k prompt remains untested. This fast profile intentionally limits the request window to 8k.
