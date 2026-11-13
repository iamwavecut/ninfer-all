#pragma once

// The split reducer of the group-16 NVFP4 small-T partials (nvfp4/small_t.cuh): each column's
// splits weighted by their maxima and sums, normalized in FP32, rotated back by the inverse D256
// Hadamard transform the cache's values carry, and stored as BF16 or as a position-split rank's
// partial output.

#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry, bool MultiBatch, bool Masked, bool Offset>
__launch_bounds__(256) __global__ void causal_attention_small_t_nvfp4_reduce_output_kernel(
    const float* partial_acc, const float* partial_m, const float* partial_l,
    const std::int32_t* positions, const std::int32_t* valid_columns, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t batch_size,
    std::int32_t visible_capacity, std::int32_t split_count, __nv_bfloat16* out,
    CausalKeyPartition partition, float* partial_out) {
    // The grid spans the call's query heads.
    const int q_heads     = static_cast<int>(gridDim.x);
    const int q_head      = static_cast<int>(blockIdx.x);
    const int flat_column = static_cast<int>(blockIdx.y);
    int batch             = 0;
    int token             = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / tokens;
        token = flat_column - batch * tokens;
    }
    const int tid = static_cast<int>(threadIdx.x);
    if (token >= tokens) return;
    if constexpr (MultiBatch) {
        if (batch >= batch_size) return;
    }
    if constexpr (Offset) positions += column_begin;
    if constexpr (MultiBatch) positions += static_cast<std::int64_t>(batch) * full_width;
    const int window  = causal_small_t_tile_window(positions[0], tokens, visible_capacity);
    int output_column = token;
    if constexpr (Offset) output_column += column_begin;
    if constexpr (MultiBatch) output_column += batch * full_width;
    if constexpr (Masked) {
        const int absolute_column = token + (Offset ? column_begin : 0);
        if (absolute_column >= valid_columns[batch]) {
            if (partial_out != nullptr) {
                if (tid < kCausalHeadDim) {
                    causal_store_partial_output(partial_out, q_head, tid, output_column, q_heads,
                                                0.0f, -CUDART_INF_F, tid == 0);
                }
                return;
            }
            if (tid < kCausalHeadDim)
                out[causal_q_index(q_head, tid, output_column, q_heads)] = __float2bfloat16(0.0f);
            return;
        }
    }


    if constexpr (MultiBatch) {
        partial_acc +=
            static_cast<std::int64_t>(batch) * kCausalHeadDim * q_heads * tokens * split_count;
        partial_m += static_cast<std::int64_t>(batch) * q_heads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * q_heads * tokens * split_count;
    }
    const int active_splits = causal_small_t_partition_active(
        partition, window,
        causal_small_t_quantized_active_splits<Geometry>(window, split_count, tokens), split_count);
    __shared__ float weights[256], warp_sums[8], scalars[2];
    __shared__ float normalized[256];
    const float head_l = causal_merge_split_statistics<Geometry>(
        partial_m, partial_l, q_head, token, tokens, active_splits, weights, warp_sums, scalars,
        q_heads);

    const float numerator = causal_weighted_split_sum(partial_acc, weights, q_head, tid, token,
                                                      tokens, active_splits, q_heads);
    normalized[tid] = head_l > 0.0F ? numerator / head_l : 0.0F;
    __syncthreads();

    if (tid >= 32) return;
    float values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { values[r] = normalized[tid + 32 * r]; }
    normalized_hadamard_d256_inplace(values, tid);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = tid + 32 * r;
        if (partial_out != nullptr) {
            causal_store_partial_output(partial_out, q_head, d, output_column, q_heads,
                                        values[r],
                                        head_l > 0.0f ? scalars[0] + logf(head_l) : -CUDART_INF_F,
                                        d == 0);
        } else {
            out[causal_q_index(q_head, d, output_column, q_heads)] = __float2bfloat16(values[r]);
        }
    }
}

} // namespace ninfer::ops
