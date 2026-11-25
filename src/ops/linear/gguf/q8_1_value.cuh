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
// warp takes part. `transposed` writes PTQ1_0's lane form instead
// (gguf::vector_activation_transposed, k a multiple of 128): value e of 128-value block b at
// qs[column][(e / 16) * (k / 128) * 16 + 16 b + e % 16], and in place of the sum the exact total of
// the block's 32 int8 values, an int16 in the half's bits.
__device__ __forceinline__ void quantize_q8_1_value(float xi, int i, int column,
                                                    std::int8_t* __restrict__ qs,
                                                    half2* __restrict__ ds, int k,
                                                    bool transposed = false) {
    float amax = fabsf(xi);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset, 32));
    }
    const float d = amax / 127.0f;
    const std::int8_t q = amax == 0.0f ? 0 : static_cast<std::int8_t>(roundf(xi / d));
    if (transposed) {
        int total = q;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_xor_sync(0xffffffffu, total, offset, 32);
        }
        const int e = i % 128;
        qs[std::int64_t(column) * k + (e / 16) * (k / 128) * 16 + (i / 128) * 16 + e % 16] = q;
        if (i % 32 == 0) {
            ds[std::int64_t(column) * (k / 32) + i / 32] =
                __halves2half2(__float2half(d), __short_as_half(static_cast<short>(total)));
        }
        return;
    }
    float sum = xi;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset, 32);
    }
    qs[std::int64_t(column) * k + i] = q;
    if (i % 32 == 0) { ds[std::int64_t(column) * (k / 32) + i / 32] = make_half2(d, sum); }
}

} // namespace ninfer::ops::gguf
