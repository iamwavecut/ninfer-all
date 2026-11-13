#pragma once

// Definition of launch_small_t_i8_tiled (int8/small_t_launch.h): the tiled INT8-family partial
// kernel (int8/small_t.cuh) at one tier, over every coding, batch form and PV accumulation.
// Included only by the generated tier units (kernel_unit.cu.in).

#include "core/device_launch.h" // CUDA_CHECK, raise_dynamic_shared_memory
#include "ops/softmax_attention/dense/causal_cache/int8/small_t.cuh"
#include "ops/softmax_attention/dense/causal_cache/int8/small_t_launch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// The tier's shape at namespace scope: the launch lambdas below read it as a constant expression.
template <SmallTI8Tier Tier>
inline constexpr SmallTI8TierShape kSmallTI8TierShape = small_t_i8_tier_shape(Tier);

// One kernel per unit that names the unit's module to cudaGetKernel: the call parses the module
// without loading code into the context.
template <typename Geometry, int TokenTile, typename CacheInput, SmallTI8Tier Tier>
__global__ void small_t_i8_tiled_module_anchor() {}

template <typename Geometry, int TokenTile, typename CacheInput, SmallTI8Tier Tier>
void launch_small_t_i8_tiled(const SmallTI8Launch& launch, CacheInput input) {
    static_assert(Tier != SmallTI8Tier::Pipelined);
    static_assert(
        small_t_i8_tier_built(small_t_i8_row_tiles(Geometry::GroupSize, TokenTile), Tier),
        "the tier is not built for this query width");
    if (launch.load_module_only) {
        cudaKernel_t kernel = nullptr;
        CUDA_CHECK(cudaGetKernel(
            &kernel, small_t_i8_tiled_module_anchor<Geometry, TokenTile, CacheInput, Tier>));
        return;
    }
    constexpr SmallTI8TierShape kShape = kSmallTI8TierShape<Tier>;
    // The 64-key dynamic arena of 64 KiB beside the kernel's static tiles needs more than 64 KiB of
    // shared memory per block, which Turing does not have: a build of Turing alone launches none.
    if constexpr (kShape.key_block == 64 && !kBuildStandardShapes) {
        return;
    } else {
        constexpr std::size_t kDynamicBytes =
            kShape.dynamic_arena ? static_cast<std::size_t>(4 * kShape.key_block * kCausalHeadDim)
                                 : 0u;
        const dim3 grid(launch.q_heads / Geometry::GroupSize, launch.splits, launch.batch_size);
        const auto issue = [&]<bool MultiBatch, bool Masked, bool PackedValues, KvKeyCoding Keys,
                               bool PvF16>() {
            constexpr SmallTI8TierShape kTier = kSmallTI8TierShape<Tier>;
            constexpr auto kernel             = causal_attention_small_t_i8_tiled_kernel<
                Geometry, TokenTile, kTier.warps, kTier.ctas_per_sm, kTier.key_block,
                kTier.dynamic_arena, MultiBatch, Masked, CacheInput, PackedValues, Keys, PvF16,
                kTier.qk_split, kTier.early_fetch>;
            if constexpr (kTier.dynamic_arena) {
                raise_dynamic_shared_memory(kernel, static_cast<int>(kDynamicBytes));
            }
            kernel<<<grid, kTier.warps * 32, kDynamicBytes, launch.stream>>>(
                static_cast<const __nv_bfloat16*>(launch.q), input, launch.positions,
                static_cast<std::int8_t*>(launch.cache_k),
                static_cast<std::int8_t*>(launch.cache_v),
                static_cast<__half*>(launch.cache_k_scale),
                static_cast<__half*>(launch.cache_v_scale), launch.block_tables,
                launch.valid_columns, launch.table_rows, launch.table_stride, launch.full_width,
                launch.column_begin, launch.logical_capacity, launch.wave_splits, launch.scale,
                launch.partial_acc, launch.partial_m, launch.partial_l, launch.partition);
        };
        const auto accumulate = [&]<bool MultiBatch, bool Masked, bool PackedValues,
                                    KvKeyCoding Keys>() {
            if (launch.pv_f16) {
                issue.template operator()<MultiBatch, Masked, PackedValues, Keys, true>();
            } else {
                issue.template operator()<MultiBatch, Masked, PackedValues, Keys, false>();
            }
        };
        const auto decode = [&]<bool MultiBatch, bool Masked>() {
            switch (launch.coding) {
            case CausalI8Coding::Lloyd4:
                accumulate.template operator()<MultiBatch, Masked, true, KvKeyCoding::Lloyd4>();
                break;
            case CausalI8Coding::Int4E8:
                accumulate.template operator()<MultiBatch, Masked, true, KvKeyCoding::Int4E8>();
                break;
            case CausalI8Coding::RootE8:
                accumulate.template operator()<MultiBatch, Masked, true, KvKeyCoding::RootE8>();
                break;
            case CausalI8Coding::PackedValues:
                accumulate.template operator()<MultiBatch, Masked, true, KvKeyCoding::Int8>();
                break;
            case CausalI8Coding::Int8:
                accumulate.template operator()<MultiBatch, Masked, false, KvKeyCoding::Int8>();
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
}

} // namespace ninfer::ops::detail
