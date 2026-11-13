#pragma once

// The split reductions of the causal small-T partials and the tile metadata of parallel query
// tiles, as the host launchers call them: one unit per reducer (small_t_reduce.cu,
// nvfp4/small_t_reduce.cu, k8v4/small_t_reduce.cu), so an edit of a partial kernel or of the
// routing compiles none of them.

#include "ops/softmax_attention/dense/causal_cache/key_partition.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace ninfer::ops::detail {

// One reduction of `splits` partials of each of width x batch_size columns into `out` (BF16
// [D, q_heads, full_width, batch]), or into a position-split rank's partial_out.
struct SmallTReduceLaunch {
    const float* partial_acc          = nullptr;
    const float* partial_m            = nullptr;
    const float* partial_l            = nullptr;
    const std::int32_t* positions     = nullptr;
    const std::int32_t* valid_columns = nullptr; // masked columns when not null
    std::int32_t q_heads              = 0;
    std::int32_t width                = 0;
    std::int32_t full_width           = 0;
    std::int32_t column_begin         = 0;
    std::int32_t batch_size           = 1;
    std::int32_t logical_capacity     = 0;
    std::int32_t splits               = 0;
    // Splits per full wave of a single-row INT8-family launch (causal_small_t_active_splits).
    std::int32_t wave_splits = 0;
    void* out                = nullptr;
    // BF16 gate applied at the store; only the shared reducer takes one.
    const void* gate = nullptr;
    CausalKeyPartition partition{};
    float* partial_out  = nullptr;
    cudaStream_t stream = nullptr;
};

// The reducer of the BF16, INT8-family (int8: their split tiers) and FP8 partials.
void launch_small_t_reduce(const SmallTReduceLaunch& launch, bool int8);

// The NVFP4 and K8V4 reducers, which also rotate the values back.
void launch_small_t_nvfp4_reduce(const SmallTReduceLaunch& launch);
void launch_small_t_k8v4_reduce(const SmallTReduceLaunch& launch);

// Spreads a single row's block-table row and valid-column count over its parallel query tiles:
// tile t covers columns [t * tile_width, (t + 1) * tile_width), so it keeps valid - t * tile_width
// of them. tile_valid is written only with valid_columns.
void launch_small_t_tile_metadata(const std::int32_t* valid_columns,
                                  const std::int32_t* table_rows, std::int32_t tile_width,
                                  std::int32_t tiles, std::int32_t* tile_valid,
                                  std::int32_t* tile_rows, cudaStream_t stream);

} // namespace ninfer::ops::detail
