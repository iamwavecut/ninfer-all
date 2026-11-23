# NInfer ggml block-quant sources

These files are copied from [llama.cpp](https://github.com/ggml-org/llama.cpp),
commit [`a25c9865fe03c954c93fd755b5d79ae86ba99750`](https://github.com/ggml-org/llama.cpp/commit/a25c9865fe03c954c93fd755b5d79ae86ba99750),
`ggml/include/` and `ggml/src/`, under the MIT license; see [LICENSE](LICENSE). Changes are
marked `NInfer:` in the source. Q4_0's DP4A matrix product (`mmq-vec-dot.cuh`, the route of GPUs
without integer tensor cores such as Tesla V100) takes its -8 offset term from the quantized
activation instead of the block's FP16 sum, which overflows once 32 activations sum past 65504.
IQ1_M, which llama.cpp multiplies on its vector kernels only, has a matrix tile loader
(`mmq-load-tiles.cuh`) and Ampere-config entries copied from IQ2_S's: each row of eight grid
values g with its ±1/8 shift becomes the int8 values 8g ± 1, each 16 values' scale d(2s + 1)/8,
as in NInfer's own vector kernel. NInfer runs it for routed experts only.

NInfer compiles ggml's integer tensor-core matrix multiplication (`ggml-cuda/mmq.cuh`) and its
dequantizers through the bridge in `src/ops/linear/gguf/`, as a separate archive, and keeps only the
headers that code includes. Products of up to eight tokens use NInfer's own kernel
(`ggml_bridge_vec.cuh`), which reads the grids and block structs from `ggml-common.h`.

The flash-attention kernel (`ggml-cuda/fattn-mma-f16.cuh` with `fattn-common.cuh` and
`cp-async.cuh`, same base commit) runs Gemma 4's prompt attention (head widths 256 and 512)
through `src/ops/linear/gguf/ggml_fattn.cu`, which launches it without ggml's backend context.
The combine-loop barrier in `fattn-mma-f16.cuh` includes the exact condition from upstream
[`1bb2b9fcbe85c35c4e3b787e1c868ef05ca702f3`](https://github.com/ggml-org/llama.cpp/commit/1bb2b9fcbe85c35c4e3b787e1c868ef05ca702f3)
(PR #30103). It synchronizes before reusing the shared combine tile when
`nbatch_combine != DKQ/2`, including the Turing 512-wide 16/32-column configurations.

The local MMQ writeback supports BF16 Store and Accumulate. Split stream-K tiles retain FP32
partials and the existing summation order, adding the residual before one final BF16 rounding.

The tile loop has an asynchronous form at each column tile the bridge launches (J = 16, 32, 64 and
128; `mul_mat_q_process_tile` with `async_tiles`, `ggml_cuda_mmq_async_tiles_supported`,
`mmq_get_nbytes_shared` with its extra argument): the next activation tile and the raw weight tile
are staged through `cp.async` while the current one is multiplied, the weights unpack from shared
memory, and the loop takes three barriers an iteration instead of four. The values are unchanged. It
covers the formats whose 256-value step is a whole number of 8-byte words
(`ggml_cuda_mmq_async_row_bytes`): Q4_0, Q5_0, Q8_0, IQ4_NL, Q4_K, Q5_K and NVFP4 copy 16 bytes at a time,
Q2_0, IQ4_XS and IQ1_M 8 bytes at a time (`cp_async_ca_8`, added to `cp-async.cuh` with a `#pragma
once` so the kernel header can include it). Rows that start on 4 bytes only, whose value count is
not a whole number of 256-value steps (Gemma 4's 704- and 2112-wide rows in the 32-value formats),
copy 4 bytes at a time (`cp_async_ca_4`); the copy width is the kernel's `async_tiles` argument, 0
for the synchronous loop. Q6_K (210-byte blocks), Q3_K, Q2_K, IQ1_S, IQ2_*, IQ3_* and Q1_0 steps
start on 2- or 4-byte boundaries and keep the synchronous loop. The form needs `cp.async` (sm_80 and
later builds), the format's step bytes of shared memory per row on top of the loop's own (Q4_K takes
92.5 KiB at J = 128; Q8_0 fits the 99 KiB of an RTX 3090 only at J = 64 and narrower), and a row
count that is a whole number of row tiles, since the copy has no row clamp. The bridge selects it
per format through the `gguf_matrix/<format>` and `gguf_moe_matrix/<format>` device routes.
