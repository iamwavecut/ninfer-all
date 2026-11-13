#pragma once

// Definition of launch_small_t_nvfp4 (small_t_quantized_launch.h): the group-16 NVFP4 (nvfp4)
// partial kernel (nvfp4/small_t.cuh) over every query width, batch form and shape set of a geometry
// and input.
// Included only by the generated nvfp4 units (kernel_unit.cu.in).

#include "core/device_launch.h" // CUDA_CHECK, raise_dynamic_shared_memory
#include "ninfer/kv_cache_storage.h"
#include "ops/common/device_tier.h"
#include "ops/kv_cache/plane_types.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/small_t.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_quantized_launch.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_width.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

template <typename Geometry, typename CacheInput>
void launch_small_t_nvfp4(const SmallTPartialLaunch& launch, CacheInput input) {
    if constexpr (kBuildPostVolta) {
        const dim3 grid(launch.q_heads / Geometry::GroupSize, launch.splits, launch.batch_size);
        const auto issue = [&]<int TokenTile, bool MultiBatch, bool Masked>() {
            with_shape_set([&]<bool TuringShapes>() {
                constexpr int RowCount  = TokenTile * Geometry::GroupSize;
                constexpr int RowTiles  = (RowCount + 15) / 16;
                constexpr int Warps     = RowTiles == 3 ? 12 : 8;
                constexpr int KeyBlock  = 32;
                constexpr int MinBlocks = RowTiles <= 2 ? 2 : 1;
                constexpr bool Compact =
                    RowTiles <= 2 ||
                    kCausalSmallTCompactKV<RowTiles, KeyBlock, 5 * KeyBlock * kCausalHeadDim,
                                           TuringShapes>;
                constexpr std::size_t DynamicBytes =
                    (Compact ? 3u : 5u) * KeyBlock * kCausalHeadDim;
                static_assert(kCausalSmallTQuantizedFixedBytes<RowTiles, KeyBlock> + DynamicBytes <=
                              shape_max_shared_bytes_per_block(TuringShapes));
                auto kernel = causal_attention_small_t_nvfp4_tiled_kernel<
                    Geometry, TokenTile, Warps, MinBlocks, KeyBlock, true, MultiBatch, Masked,
                    CacheInput, TuringShapes>;
                raise_dynamic_shared_memory(kernel, static_cast<int>(DynamicBytes));
                constexpr auto kStorage = KvCacheStorage::Nvfp4Group16;
                const auto cache_k_ptr  = static_cast<KvKeyCodeT<kStorage>*>(launch.cache_k);
                const auto cache_v_ptr  = static_cast<KvValueCodeT<kStorage>*>(launch.cache_v);
                const auto k_scale_ptr  = static_cast<KvKeyScaleT<kStorage>*>(launch.cache_k_scale);
                const auto v_scale_ptr =
                    static_cast<KvValueScaleT<kStorage>*>(launch.cache_v_scale);
                assert_kv_planes<kStorage, decltype(cache_k_ptr), decltype(cache_v_ptr),
                                 decltype(k_scale_ptr), decltype(v_scale_ptr)>();
                kernel<<<grid, Warps * 32, DynamicBytes, launch.stream>>>(
                    static_cast<const __nv_bfloat16*>(launch.q), input, launch.positions,
                    cache_k_ptr, cache_v_ptr, k_scale_ptr, v_scale_ptr, launch.block_tables,
                    launch.valid_columns, launch.table_rows, launch.table_stride,
                    launch.full_width, launch.column_begin, launch.logical_capacity, launch.scale,
                    launch.partial_acc, launch.partial_m, launch.partial_l, launch.partition);
                CUDA_CHECK(cudaGetLastError());
            });
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
}

} // namespace ninfer::ops::detail
