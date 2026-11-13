// ninfer::ops::detail - the split reducer of the BF16, INT8-family and FP8 small-T partials and the
// tile metadata of parallel query tiles (small_t_reduce_launch.h).
#include "ops/softmax_attention/dense/causal_cache/small_t_reduce_launch.h"

#include "core/device_launch.h" // CUDA_CHECK
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_reduce.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

namespace {

__global__ void causal_small_t_tile_metadata_kernel(const std::int32_t* valid_columns,
                                                    const std::int32_t* table_rows,
                                                    std::int32_t tile_width, std::int32_t tiles,
                                                    std::int32_t* tile_valid,
                                                    std::int32_t* tile_rows) {
    const int tile = static_cast<int>(threadIdx.x);
    if (tile >= tiles) { return; }
    if (tile_valid != nullptr) { tile_valid[tile] = valid_columns[0] - tile * tile_width; }
    tile_rows[tile] = table_rows == nullptr ? 0 : table_rows[0];
}

} // namespace

void launch_small_t_reduce(const SmallTReduceLaunch& launch, bool int8) {
    with_causal_query_geometry(launch.q_heads, [&]<typename Geometry>() {
        constexpr int kBlock  = 256;
        constexpr int kDChunk = Geometry::GroupSize == 6 ? 256 : 64;
        const dim3 grid(launch.q_heads, div_up(kCausalHeadDim, kDChunk),
                        launch.width * launch.batch_size);
        const auto issue = [&]<bool Int8, bool MultiBatch, bool Masked, bool Offset>() {
            causal_attention_small_t_reduce_output_kernel<Geometry, kDChunk, Int8, MultiBatch,
                                                          Masked, Offset>
                <<<grid, kBlock, 0, launch.stream>>>(
                    launch.partial_acc, launch.partial_m, launch.partial_l, launch.positions,
                    launch.valid_columns, launch.width, launch.full_width, launch.column_begin,
                    launch.batch_size, launch.logical_capacity, launch.splits,
                    Int8 ? launch.wave_splits : 0, static_cast<__nv_bfloat16*>(launch.out),
                    static_cast<const __nv_bfloat16*>(launch.gate), launch.partition,
                    launch.partial_out);
        };
        const auto offset = [&]<bool Int8, bool MultiBatch, bool Masked>() {
            if (launch.column_begin == 0) {
                issue.template operator()<Int8, MultiBatch, Masked, false>();
            } else {
                issue.template operator()<Int8, MultiBatch, Masked, true>();
            }
        };
        const auto batch = [&]<bool Int8>() {
            const bool masked = launch.valid_columns != nullptr;
            if (launch.batch_size == 1) {
                if (masked) {
                    offset.template operator()<Int8, false, true>();
                } else {
                    offset.template operator()<Int8, false, false>();
                }
            } else if (masked) {
                offset.template operator()<Int8, true, true>();
            } else {
                offset.template operator()<Int8, true, false>();
            }
        };
        if (int8) {
            batch.template operator()<true>();
        } else {
            batch.template operator()<false>();
        }
    });
    CUDA_CHECK(cudaGetLastError());
}

void launch_small_t_tile_metadata(const std::int32_t* valid_columns,
                                  const std::int32_t* table_rows, std::int32_t tile_width,
                                  std::int32_t tiles, std::int32_t* tile_valid,
                                  std::int32_t* tile_rows, cudaStream_t stream) {
    causal_small_t_tile_metadata_kernel<<<1, 32, 0, stream>>>(valid_columns, table_rows,
                                                               tile_width, tiles, tile_valid,
                                                               tile_rows);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
