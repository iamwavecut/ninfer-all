// ninfer::ops::detail - the split reducer of the group-16 NVFP4 small-T partials
// (small_t_reduce_launch.h).
#include "ops/softmax_attention/dense/causal_cache/small_t_reduce_launch.h"

#include "core/device_launch.h" // CUDA_CHECK
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/small_t_reduce.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

void launch_small_t_nvfp4_reduce(const SmallTReduceLaunch& launch) {
    with_causal_query_geometry(launch.q_heads, [&]<typename Geometry>() {
        constexpr int kBlock = 256;
        const dim3 grid(launch.q_heads, launch.width * launch.batch_size);
        const auto issue = [&]<bool MultiBatch, bool Masked, bool Offset>() {
            causal_attention_small_t_nvfp4_reduce_output_kernel<Geometry, MultiBatch, Masked,
                                                                Offset>
                <<<grid, kBlock, 0, launch.stream>>>(
                    launch.partial_acc, launch.partial_m, launch.partial_l, launch.positions,
                    launch.valid_columns, launch.width, launch.full_width, launch.column_begin,
                    launch.batch_size, launch.logical_capacity, launch.splits,
                    static_cast<__nv_bfloat16*>(launch.out), launch.partition, launch.partial_out);
        };
        const auto offset = [&]<bool MultiBatch, bool Masked>() {
            if (launch.column_begin == 0) {
                issue.template operator()<MultiBatch, Masked, false>();
            } else {
                issue.template operator()<MultiBatch, Masked, true>();
            }
        };
        const bool masked = launch.valid_columns != nullptr;
        if (launch.batch_size == 1) {
            if (masked) {
                offset.template operator()<false, true>();
            } else {
                offset.template operator()<false, false>();
            }
        } else if (masked) {
            offset.template operator()<true, true>();
        } else {
            offset.template operator()<true, false>();
        }
    });
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
