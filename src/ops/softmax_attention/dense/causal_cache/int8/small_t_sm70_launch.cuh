#pragma once

// Definition of launch_small_t_i8_sm70 (int8/small_t_launch.h): Volta's INT8-family kernel
// (int8/small_t_sm70.cuh) over every coding and batch form, with the tiled kernel's grid, active
// splits and partial outputs. Included only by the generated Volta units (kernel_unit.cu.in),
// which compile for sm_70 alone.

#include "core/device_launch.h" // CUDA_CHECK
#include "ops/softmax_attention/dense/causal_cache/int8/small_t_launch.h"
#include "ops/softmax_attention/dense/causal_cache/int8/small_t_sm70.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// One kernel per unit that names the unit's module to cudaGetKernel.
template <typename Geometry, int TokenTile, typename CacheInput>
__global__ void small_t_i8_sm70_module_anchor() {}

template <typename Geometry, int TokenTile, typename CacheInput>
void launch_small_t_i8_sm70(const SmallTI8Launch& launch, CacheInput input) {
    if (launch.load_module_only) {
        cudaKernel_t kernel = nullptr;
        CUDA_CHECK(
            cudaGetKernel(&kernel, small_t_i8_sm70_module_anchor<Geometry, TokenTile, CacheInput>));
        return;
    }
    const dim3 grid(launch.q_heads / Geometry::GroupSize, launch.splits, launch.batch_size);
    constexpr int kBlock = 32 * sm70_attention::Shape<TokenTile * Geometry::GroupSize>::kWarps;
    const auto issue = [&]<bool MultiBatch, bool Masked, bool PackedValues, KvKeyCoding Keys>() {
        causal_attention_small_t_sm70_i8_kernel<Geometry, TokenTile, MultiBatch, Masked, CacheInput,
                                                PackedValues, Keys>
            <<<grid, kBlock, 0, launch.stream>>>(
                static_cast<const __nv_bfloat16*>(launch.q), input, launch.positions,
                static_cast<std::int8_t*>(launch.cache_k),
                static_cast<std::int8_t*>(launch.cache_v),
                static_cast<__half*>(launch.cache_k_scale),
                static_cast<__half*>(launch.cache_v_scale), launch.block_tables,
                launch.valid_columns, launch.table_rows, launch.table_stride, launch.full_width,
                launch.column_begin, launch.logical_capacity, launch.wave_splits, launch.scale,
                launch.partial_acc, launch.partial_m, launch.partial_l, launch.partition);
    };
    const auto decode = [&]<bool MultiBatch, bool Masked>() {
        switch (launch.coding) {
        case CausalI8Coding::Lloyd4:
            issue.template operator()<MultiBatch, Masked, true, KvKeyCoding::Lloyd4>();
            break;
        case CausalI8Coding::Int4E8:
            issue.template operator()<MultiBatch, Masked, true, KvKeyCoding::Int4E8>();
            break;
        case CausalI8Coding::RootE8:
            issue.template operator()<MultiBatch, Masked, true, KvKeyCoding::RootE8>();
            break;
        case CausalI8Coding::PackedValues:
            issue.template operator()<MultiBatch, Masked, true, KvKeyCoding::Int8>();
            break;
        case CausalI8Coding::Int8:
            issue.template operator()<MultiBatch, Masked, false, KvKeyCoding::Int8>();
            break;
        }
    };
    if (launch.multi_batch) {
        if (launch.masked) {
            decode.template operator()<true, true>();
        } else {
            decode.template operator()<true, false>();
        }
    } else if (launch.masked) {
        decode.template operator()<false, true>();
    } else {
        decode.template operator()<false, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
