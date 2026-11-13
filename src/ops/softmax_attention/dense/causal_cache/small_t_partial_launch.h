#pragma once

// The arguments of a causal small-T partial launch as the host launchers (small_t.cpp,
// fp8/small_t.cpp, ...) pass them to the kernel units: plain pointers and counts, so a launcher
// compiles without the kernels and a kernel unit without the routing.

#include "ops/softmax_attention/dense/causal_cache/key_partition.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace ninfer::ops::detail {

// One partial launch over the cache's pages: the kernel arguments, the grid (q_heads / GroupSize
// KV-head groups x splits x batch rows) and the batch form the launcher chose at run time.
struct SmallTPartialLaunch {
    const void* q                     = nullptr; // BF16 [D, q_heads, full_width, batch]
    const std::int32_t* positions     = nullptr;
    void* cache_k                     = nullptr;
    void* cache_v                     = nullptr;
    void* cache_k_scale               = nullptr; // the scale planes of a quantized cache
    void* cache_v_scale               = nullptr;
    const std::int32_t* block_tables  = nullptr;
    const std::int32_t* valid_columns = nullptr;
    const std::int32_t* table_rows    = nullptr;
    std::int32_t table_stride         = 0;
    std::int32_t width                = 0; // query columns of the launch
    std::int32_t full_width           = 0;
    std::int32_t column_begin         = 0;
    std::int32_t logical_capacity     = 0;
    std::int32_t q_heads              = 0;
    std::int32_t splits               = 0;
    std::int32_t batch_size           = 1;
    float scale                       = 0.0F;
    float* partial_acc                = nullptr;
    float* partial_m                  = nullptr;
    float* partial_l                  = nullptr;
    CausalKeyPartition partition{};
    bool multi_batch    = false;
    bool masked         = false;
    cudaStream_t stream = nullptr;
};

} // namespace ninfer::ops::detail
