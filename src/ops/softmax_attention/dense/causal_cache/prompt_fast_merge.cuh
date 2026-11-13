#pragma once

// The key-split merge of the fast prompt kernels (INT8 family: int8/prompt_fast.cuh, FP8:
// fp8/prompt_fast.cuh): it combines the splits over the storage-agnostic FP32 partial rows and
// (max, sum) statistics the split launches publish, so both families share this one kernel. The
// partials' layout is here as well, apart from the host's plan (int8/prompt_fast_plan.h), so the
// kernel units do not read the routing.

#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops {

namespace detail {

// Split partial layout for key split s, column c and query head h: the D256 row at
// ((s * width + c) * q_heads + h) * 256 and its (max, sum) pair at that row index.
__host__ __device__ __forceinline__ std::int64_t
causal_prompt_i8_fast_partial_row(int split, int column, int q_head, int width, int q_heads) {
    return (static_cast<std::int64_t>(split) * width + column) * q_heads + q_head;
}

} // namespace detail

// Combines the key splits of one column and query head: every split row is normalized by its own
// sum, so the merged row is sum_s w_s * row_s / sum_s w_s with w_s = sum_s * 2^((m_s - M) * c).
// Columns past the valid count publish zeros.
template <typename Geometry>
__global__ __launch_bounds__(kCausalPromptHeadDim) void causal_attention_prompt_fast_merge_kernel(
    const float* __restrict__ partial_rows, const float2* __restrict__ partial_stats,
    const std::int32_t* __restrict__ valid_columns, std::int32_t width, std::int32_t splits,
    float scale_l2, __nv_bfloat16* __restrict__ out) {
    const int column  = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int q_heads = causal_prompt_q_heads<Geometry>();
    const int d       = static_cast<int>(threadIdx.x);
    const int tokens  = valid_columns == nullptr ? width : max(0, min(width, valid_columns[0]));
    const auto index  = causal_prompt_q_index<Geometry>(q_head, d, column, q_heads);
    if (column >= tokens) {
        out[index] = __float2bfloat16(0.0f);
        return;
    }
    float maximum = -CUDART_INF_F;
    for (int split = 0; split < splits; ++split) {
        const float2 stats =
            partial_stats[detail::causal_prompt_i8_fast_partial_row(split, column, q_head, width, q_heads)];
        if (stats.y > 0.0f) { maximum = fmaxf(maximum, stats.x); }
    }
    float numerator   = 0.0f;
    float denominator = 0.0f;
    for (int split = 0; split < splits; ++split) {
        const std::int64_t row =
            detail::causal_prompt_i8_fast_partial_row(split, column, q_head, width, q_heads);
        const float2 stats = partial_stats[row];
        if (stats.y > 0.0f) {
            const float weight = stats.y * exp2f((stats.x - maximum) * scale_l2);
            numerator = __fmaf_rn(weight, partial_rows[row * kCausalPromptHeadDim + d], numerator);
            denominator += weight;
        }
    }
    out[index] = __float2bfloat16(denominator > 0.0f ? numerator / denominator : 0.0f);
}

} // namespace ninfer::ops
