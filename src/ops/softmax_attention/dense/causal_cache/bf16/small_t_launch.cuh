#pragma once

// Definition of launch_small_t_bf16 (bf16/small_t_launch.h): the BF16 partial kernel
// (bf16/small_t.cuh) over every query width, batch form and shape set of a geometry and input.
// Included only by the generated BF16 units (kernel_unit.cu.in).

#include "core/device_launch.h" // CUDA_CHECK
#include "ops/common/device_tier.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/small_t.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/small_t_launch.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_width.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <typename Geometry, typename CacheInput>
void launch_small_t_bf16(const SmallTPartialLaunch& launch, CacheInput input) {
    if constexpr (kBuildPostVolta) {
        const dim3 grid(launch.q_heads / Geometry::GroupSize, launch.splits, launch.batch_size);
        // Every kernel takes the same arguments and uses only static shared memory.
        const auto issue = [&](auto kernel, int block) {
            kernel<<<grid, block, 0, launch.stream>>>(
                static_cast<const __nv_bfloat16*>(launch.q), input, launch.positions,
                static_cast<__nv_bfloat16*>(launch.cache_k),
                static_cast<__nv_bfloat16*>(launch.cache_v), launch.block_tables,
                launch.valid_columns, launch.table_rows, launch.table_stride, launch.width,
                launch.full_width, launch.column_begin, launch.logical_capacity, launch.scale,
                launch.partial_acc, launch.partial_m, launch.partial_l, launch.partition);
            CUDA_CHECK(cudaGetLastError());
        };
        const auto shaped = [&]<int TokenTile, bool MultiBatch, bool Masked>() {
            // BF16 keeps its row-tile warp count.
            constexpr int kWarps = TokenTile == 1 ? 2 : 4;
            with_shape_set([&]<bool TuringShapes>() {
                if constexpr (TuringShapes) {
                    issue(causal_attention_small_t_tc_partial_bf16_turing_kernel<
                              Geometry, TokenTile, kWarps, MultiBatch, Masked, CacheInput>,
                          32 * kWarps);
                } else {
                    issue(causal_attention_small_t_tc_partial_bf16_kernel<
                              Geometry, TokenTile, kWarps, MultiBatch, Masked, CacheInput>,
                          32 * kWarps);
                }
            });
        };
        with_small_t_width<Geometry>(launch.width, [&]<int TokenTile>() {
            if (launch.multi_batch) {
                if (launch.masked) {
                    shaped.template operator()<TokenTile, true, true>();
                } else {
                    shaped.template operator()<TokenTile, true, false>();
                }
            } else if (launch.masked) {
                shaped.template operator()<TokenTile, false, true>();
            } else {
                shaped.template operator()<TokenTile, false, false>();
            }
        });
    }
}

} // namespace ninfer::ops::detail
