#pragma once

// Definition of launch_small_t_quantized_sm70 (sm70/small_t_quant_launch.h): Volta's FP8 and NVFP4
// partial kernel (sm70/small_t_quant.cuh) over every query width and batch form, with the tiled
// kernels' grid, splits and partial outputs. Included only by the generated Volta units
// (kernel_unit.cu.in), which compile for sm_70 alone.

#include "core/device_launch.h" // CUDA_CHECK
#include "ops/softmax_attention/dense/causal_cache/sm70/small_t_quant.cuh"
#include "ops/softmax_attention/dense/causal_cache/sm70/small_t_quant_launch.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_width.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

template <typename Geometry, typename CacheInput, bool KeyFp8, bool ValueFp8>
void launch_small_t_quantized_sm70(const SmallTPartialLaunch& launch, CacheInput input) {
    using KeyScale   = std::conditional_t<KeyFp8, __half, std::uint8_t>;
    using ValueScale = std::conditional_t<ValueFp8, __half, std::uint8_t>;
    const dim3 grid(launch.q_heads / Geometry::GroupSize, launch.splits, launch.batch_size);
    const auto issue = [&]<int TokenTile, bool MultiBatch, bool Masked>() {
        constexpr int kBlock = 32 * sm70_attention::Shape<TokenTile * Geometry::GroupSize>::kWarps;
        causal_attention_small_t_sm70_quantized_kernel<Geometry, TokenTile, MultiBatch, Masked,
                                                       CacheInput, KeyFp8, ValueFp8>
            <<<grid, kBlock, 0, launch.stream>>>(
                static_cast<const __nv_bfloat16*>(launch.q), input, launch.positions,
                static_cast<std::uint8_t*>(launch.cache_k),
                static_cast<std::uint8_t*>(launch.cache_v),
                static_cast<KeyScale*>(launch.cache_k_scale),
                static_cast<ValueScale*>(launch.cache_v_scale), launch.block_tables,
                launch.valid_columns, launch.table_rows, launch.table_stride, launch.full_width,
                launch.column_begin, launch.logical_capacity, launch.scale, launch.partial_acc,
                launch.partial_m, launch.partial_l, launch.partition);
        CUDA_CHECK(cudaGetLastError());
    };
    with_small_t_width<Geometry>(launch.width, [&]<int TokenTile>() {
        if (launch.multi_batch) {
            if (launch.masked) {
                issue.template operator()<TokenTile, true, true>();
            } else {
                issue.template operator()<TokenTile, true, false>();
            }
        } else if (launch.masked) {
            issue.template operator()<TokenTile, false, true>();
        } else {
            issue.template operator()<TokenTile, false, false>();
        }
    });
}

} // namespace ninfer::ops::detail
