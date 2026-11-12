// ninfer::ops - the cross-rank merge of position-split causal attention.
#include "ninfer/ops/softmax_attention.h"

#include "core/device_launch.h"
#include "ops/common/math.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr int kHeadDim = 256;

struct MergeSources {
    const float* partial[kCausalAttentionMaximumMergeRanks];
};

// One block per (query head of the share, column), one thread per head dimension. The weights and
// the sums run in rank order, so every rank that merges the same partials computes the same bytes.
__launch_bounds__(kHeadDim) __global__ void causal_attention_merge_partials_kernel(
    MergeSources sources, std::int32_t ranks, std::int32_t q_heads, std::int32_t head_begin,
    const __nv_bfloat16* __restrict__ gate, __nv_bfloat16* out) {
    const int share_head = static_cast<int>(blockIdx.x);
    const int heads      = static_cast<int>(gridDim.x);
    const int column     = static_cast<int>(blockIdx.y);
    const int d          = static_cast<int>(threadIdx.x);
    const int q_head     = head_begin + share_head;
    // The column's block: the outputs [256, q_heads], then the log-sum-exps [q_heads].
    const std::int64_t block    = static_cast<std::int64_t>(column) * (kHeadDim + 1) * q_heads;
    const std::int64_t lse_at   = block + static_cast<std::int64_t>(kHeadDim) * q_heads + q_head;
    const std::int64_t value_at = block + static_cast<std::int64_t>(q_head) * kHeadDim + d;

    float maximum = -CUDART_INF_F;
    for (int rank = 0; rank < ranks; ++rank) {
        maximum = fmaxf(maximum, sources.partial[rank][lse_at]);
    }
    float numerator   = 0.0f;
    float denominator = 0.0f;
    if (maximum > -CUDART_INF_F) {
        for (int rank = 0; rank < ranks; ++rank) {
            const float lse = sources.partial[rank][lse_at];
            if (lse == -CUDART_INF_F) { continue; }
            const float weight = expf(lse - maximum);
            denominator += weight;
            numerator += weight * sources.partial[rank][value_at];
        }
    }
    const float value = denominator > 0.0f ? numerator / denominator : 0.0f;
    const std::int64_t out_at =
        d + static_cast<std::int64_t>(kHeadDim) *
                (share_head + static_cast<std::int64_t>(heads) * column);
    if (gate == nullptr) {
        out[out_at] = __float2bfloat16(value);
        return;
    }
    // As the small-T reducer's epilogue: the standalone multiply's rounding of the stored output.
    const __nv_bfloat16 reduced = __float2bfloat16(value);
    out[out_at] =
        __float2bfloat16_rn(__bfloat162float(reduced) * sigmoid(__bfloat162float(gate[out_at])));
}

} // namespace

std::size_t causal_attention_partial_elements(std::int32_t q_heads, std::int32_t columns) {
    return static_cast<std::size_t>(kHeadDim + 1) * static_cast<std::size_t>(q_heads) *
           static_cast<std::size_t>(columns);
}

void causal_attention_merge_partials(std::span<const float* const> partials, std::int32_t q_heads,
                                     std::int32_t columns, std::int32_t head_begin,
                                     const Tensor* gate, Tensor& out, cudaStream_t stream) {
    const auto ranks = static_cast<std::int32_t>(partials.size());
    if (ranks <= 0 || ranks > kCausalAttentionMaximumMergeRanks || q_heads <= 0 || columns <= 0) {
        throw std::invalid_argument("causal_attention_merge_partials: invalid ranks or columns");
    }
    if (out.dtype != DType::BF16 || out.ne[0] != kHeadDim || out.ne[2] != columns ||
        out.ne[3] != 1 || !out.is_contiguous() || out.data == nullptr) {
        throw std::invalid_argument("causal_attention_merge_partials: invalid out");
    }
    const std::int32_t heads = out.ne[1];
    if (head_begin < 0 || heads <= 0 || head_begin + heads > q_heads) {
        throw std::invalid_argument("causal_attention_merge_partials: invalid head share");
    }
    if (gate != nullptr &&
        (gate->dtype != DType::BF16 || gate->ne[0] != kHeadDim || gate->ne[1] != heads ||
         gate->ne[2] != columns || !gate->is_contiguous())) {
        throw std::invalid_argument("causal_attention_merge_partials: invalid gate");
    }
    MergeSources sources{};
    for (std::int32_t rank = 0; rank < ranks; ++rank) {
        if (partials[static_cast<std::size_t>(rank)] == nullptr) {
            throw std::invalid_argument("causal_attention_merge_partials: null partial");
        }
        sources.partial[rank] = partials[static_cast<std::size_t>(rank)];
    }
    const dim3 grid(static_cast<unsigned>(heads), static_cast<unsigned>(columns));
    causal_attention_merge_partials_kernel<<<grid, kHeadDim, 0, stream>>>(
        sources, ranks, q_heads, head_begin,
        gate == nullptr ? nullptr : static_cast<const __nv_bfloat16*>(gate->data),
        static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
