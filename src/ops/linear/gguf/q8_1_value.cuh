#pragma once

// One value of the vector kernels' planar q8_1 activation (gguf::vector_activation_bytes): ggml's
// q8_1 numbers, the 32 values of a block held by the 32 lanes of one warp. The quantizers of
// ggml_bridge.cu and the model kernels that write the activation themselves share it, so their
// bytes are the same. Free of the ggml headers, which no NInfer translation unit may see.

#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::gguf {

// Value i of column `column` (k values a column): qs[column][i] = round(x / d) and, from the
// block's first lane, ds[column][i / 32] = (d, sum) with d = max|x| / 127 over the block and sum
// its unquantized total, both reduced over the warp's lanes in ggml's xor order. Every lane of the
// warp takes part.
__device__ __forceinline__ void quantize_q8_1_value(float xi, int i, int column,
                                                    std::int8_t* __restrict__ qs,
                                                    half2* __restrict__ ds, int k) {
    float amax = fabsf(xi);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset, 32));
    }
    float sum = xi;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset, 32);
    }
    const float d = amax / 127.0f;
    qs[std::int64_t(column) * k + i] = amax == 0.0f ? 0 : static_cast<std::int8_t>(roundf(xi / d));
    if (i % 32 == 0) { ds[std::int64_t(column) * (k / 32) + i / 32] = make_half2(d, sum); }
}

} // namespace ninfer::ops::gguf
