#pragma once

// ninfer::ops - the split reducer of the BF16 and INT8-family small-T partials (and of FP8's): each
// column's splits weighted by their maxima and sums, normalized, optionally gated, and stored as
// BF16 or as a position-split rank's partial output.

#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry, int DChunk, bool Int8, bool MultiBatch, bool Masked, bool Offset>
__launch_bounds__(256) __global__ void causal_attention_small_t_reduce_output_kernel(
    const float* partial_acc, const float* partial_m, const float* partial_l,
    const std::int32_t* positions, const std::int32_t* valid_columns, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t batch_size,
    std::int32_t visible_capacity, std::int32_t split_count, std::int32_t wave_splits,
    __nv_bfloat16* out, const __nv_bfloat16* __restrict__ gate, CausalKeyPartition partition,
    float* partial_out) {
    static_assert(DChunk > 0 && DChunk <= kCausalHeadDim);

    // The grid spans the call's query heads.
    const int q_heads     = static_cast<int>(gridDim.x);
    const int q_head      = static_cast<int>(blockIdx.x);
    const int d_start     = static_cast<int>(blockIdx.y) * DChunk;
    const int flat_column = static_cast<int>(blockIdx.z);
    int batch             = 0;
    int token             = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / tokens;
        token = flat_column - batch * tokens;
    }
    const int tid = threadIdx.x;
    if (token >= tokens) { return; }
    if constexpr (MultiBatch) {
        if (batch >= batch_size) { return; }
    }

    if constexpr (Offset) { positions += column_begin; }
    if constexpr (MultiBatch) { positions += batch * full_width; }
    const int first_pos = positions[0];
    int output_column   = token;
    if constexpr (Offset) { output_column += column_begin; }
    if constexpr (MultiBatch) { output_column += batch * full_width; }
    if constexpr (Masked) {
        const int absolute_column = token + (Offset ? column_begin : 0);
        if (absolute_column >= valid_columns[batch]) {
            if (partial_out != nullptr) {
                if (tid < DChunk && d_start + tid < kCausalHeadDim) {
                    causal_store_partial_output(partial_out, q_head, d_start + tid,
                                                output_column, q_heads, 0.0f, -CUDART_INF_F,
                                                tid == 0 && d_start == 0);
                }
                return;
            }
            if (tid < DChunk && d_start + tid < kCausalHeadDim) {
                const auto index = causal_q_index(q_head, d_start + tid, output_column, q_heads);
                // 0 * sigmoid(g) is 0 for every finite g, but not for a NaN gate, and the
                // standalone multiply would propagate that NaN. Do the multiply either way.
                out[index] =
                    gate == nullptr
                        ? __float2bfloat16(0.0f)
                        : __float2bfloat16_rn(0.0f * sigmoid(__bfloat162float(gate[index])));
            }
            return;
        }
    }


    if constexpr (MultiBatch) {
        const std::int64_t partial_acc_row =
            static_cast<std::int64_t>(batch) * kCausalHeadDim * q_heads * tokens * split_count;
        const std::int64_t partial_stat_row =
            static_cast<std::int64_t>(batch) * q_heads * tokens * split_count;
        partial_acc += partial_acc_row;
        partial_m += partial_stat_row;
        partial_l += partial_stat_row;
    }

    const int window = causal_small_t_tile_window(first_pos, tokens, visible_capacity);
    const int active_split_count = causal_small_t_partition_active(
        partition, window,
        causal_small_t_active_splits<Geometry, Int8>(window, split_count, tokens, wave_splits),
        split_count);

    __shared__ float weights[256], warp_sums[8], scalars[2];
    const float head_l = causal_merge_split_statistics<Geometry>(
        partial_m, partial_l, q_head, token, tokens, active_split_count, weights, warp_sums,
        scalars, q_heads);
    const int d = d_start + tid;
    if (tid >= DChunk || d >= kCausalHeadDim) return;
    const float numerator = causal_weighted_split_sum(partial_acc, weights, q_head, d, token,
                                                      tokens, active_split_count, q_heads);

    const float value    = (head_l > 0.0f) ? numerator / head_l : 0.0f;
    if (partial_out != nullptr) {
        causal_store_partial_output(partial_out, q_head, d, output_column, q_heads, value,
                                    head_l > 0.0f ? scalars[0] + logf(head_l) : -CUDART_INF_F,
                                    tid == 0 && d_start == 0);
        return;
    }
    const auto out_index = causal_q_index(q_head, d, output_column, q_heads);
    if (gate == nullptr) {
        out[out_index] = __float2bfloat16(value);
        return;
    }
    // Bit-exact with the standalone multiply: that kernel reads what attention stored, so it sees
    // the reduce result already rounded to BF16. Round here first, widen back, multiply in FP32,
    // round to nearest. Keeping the FP32 value would be more accurate and would move tokens.
    const __nv_bfloat16 reduced = __float2bfloat16(value);
    out[out_index] =
        __float2bfloat16_rn(__bfloat162float(reduced) * sigmoid(__bfloat162float(gate[out_index])));
}

} // namespace ninfer::ops
