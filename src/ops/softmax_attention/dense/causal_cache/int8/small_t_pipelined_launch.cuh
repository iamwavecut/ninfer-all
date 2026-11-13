#pragma once

// Definition of launch_small_t_i8_pipelined (int8/small_t_launch.h): the pipelined INT8 partial
// kernel (int8/small_t_pipelined.cuh) over every batch form and PV accumulation. Included only by
// the generated pipelined units (kernel_unit.cu.in).

#include "core/device_launch.h" // CUDA_CHECK, raise_dynamic_shared_memory
#include "ops/softmax_attention/dense/causal_cache/int8/small_t_launch.h"
#include "ops/softmax_attention/dense/causal_cache/int8/small_t_pipelined.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// One kernel per unit that names the unit's module to cudaGetKernel.
template <typename Geometry, int TokenTile, typename CacheInput>
__global__ void small_t_i8_pipelined_module_anchor() {}

template <typename Geometry, int TokenTile, typename CacheInput>
void launch_small_t_i8_pipelined(const SmallTI8Launch& launch, CacheInput input) {
    if (launch.load_module_only) {
        cudaKernel_t kernel = nullptr;
        CUDA_CHECK(cudaGetKernel(
            &kernel, small_t_i8_pipelined_module_anchor<Geometry, TokenTile, CacheInput>));
        return;
    }
    const dim3 grid(launch.q_heads / Geometry::GroupSize, launch.splits, launch.batch_size);
    const auto issue = [&]<bool MultiBatch, bool Masked, bool PvF16>() {
        constexpr auto kernel =
            causal_attention_small_t_i8_pipelined_kernel<Geometry, TokenTile, MultiBatch, Masked,
                                                         CacheInput, PvF16>;
        constexpr int kArena = CausalSmallTI8PipelinedShape::kArenaBytes;
        raise_dynamic_shared_memory(kernel, kArena);
        kernel<<<grid, CausalSmallTI8PipelinedShape::kThreads, kArena, launch.stream>>>(
            static_cast<const __nv_bfloat16*>(launch.q), input, launch.positions,
            static_cast<std::int8_t*>(launch.cache_k), static_cast<std::int8_t*>(launch.cache_v),
            static_cast<__half*>(launch.cache_k_scale), static_cast<__half*>(launch.cache_v_scale),
            launch.block_tables, launch.valid_columns, launch.table_rows, launch.table_stride,
            launch.full_width, launch.column_begin, launch.logical_capacity, launch.wave_splits,
            launch.scale, launch.partial_acc, launch.partial_m, launch.partial_l);
    };
    const auto accumulate = [&]<bool MultiBatch, bool Masked>() {
        if (launch.pv_f16) {
            issue.template operator()<MultiBatch, Masked, true>();
        } else {
            issue.template operator()<MultiBatch, Masked, false>();
        }
    };
    if (launch.multi_batch) {
        if (launch.masked) {
            accumulate.template operator()<true, true>();
        } else {
            accumulate.template operator()<true, false>();
        }
    } else if (launch.masked) {
        accumulate.template operator()<false, true>();
    } else {
        accumulate.template operator()<false, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
