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

The Q4_K tile loop at J = 128 has an asynchronous form (`mul_mat_q_process_tile` with
`async_tiles`, `ggml_cuda_mmq_async_tiles_supported`, `mmq_get_nbytes_shared` with its extra
argument): the next activation tile and the raw weight tile are staged through `cp.async` while the
current one is multiplied, the weights unpack from shared memory, and the loop takes three barriers
an iteration instead of four. The values are unchanged. The form needs `cp.async` (sm_80 and later
builds), 92.5 KiB of shared memory and a row count that is a whole number of row tiles, since the
copy has no row clamp. The bridge selects it through the `gguf_matrix/q4_k` and
`gguf_moe_matrix/q4_k` device routes; `cp-async.cuh` gains a `#pragma once` so the kernel header can
include it.
