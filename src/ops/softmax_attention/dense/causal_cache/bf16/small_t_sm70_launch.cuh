#pragma once

// Definition of launch_small_t_bf16_sm70 (bf16/small_t_launch.h): Volta's BF16 partial kernel
// (bf16/small_t_sm70.cuh) over every query width and batch form, with the tiled kernel's grid,
// splits and partial outputs. Included only by the generated Volta units (kernel_unit.cu.in),
// which compile for sm_70 alone.

#include "core/device_launch.h" // CUDA_CHECK
#include "ops/softmax_attention/dense/causal_cache/bf16/small_t_launch.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/small_t_sm70.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_width.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <typename Geometry, typename CacheInput>
void launch_small_t_bf16_sm70(const SmallTPartialLaunch& launch, CacheInput input) {
    static_assert(!kCausalPackedInput<CacheInput>, "bf16p needs sm_75 or newer");
    const dim3 grid(launch.q_heads / Geometry::GroupSize, launch.splits, launch.batch_size);
    const auto issue = [&]<int TokenTile, bool MultiBatch, bool Masked>() {
        constexpr int kBlock = 32 * sm70_attention::Shape<TokenTile * Geometry::GroupSize>::kWarps;
        causal_attention_small_t_sm70_bf16_kernel<Geometry, TokenTile, MultiBatch, Masked,
                                                  CacheInput><<<grid, kBlock, 0, launch.stream>>>(
            static_cast<const __nv_bfloat16*>(launch.q), input, launch.positions,
            static_cast<__nv_bfloat16*>(launch.cache_k),
            static_cast<__nv_bfloat16*>(launch.cache_v), launch.block_tables, launch.valid_columns,
            launch.table_rows, launch.table_stride, launch.width, launch.full_width,
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
