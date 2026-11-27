#pragma once

// Private workspace layout shared by token sampling and stochastic speculative acceptance.
// The same builder supplies the public sizing query and launcher-side binding.

#include "core/layout.h"
#include "ops/common/math.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr int kSamplerBlock               = 256;
inline constexpr int kSamplerTileItems           = 256;
inline constexpr int kSamplerItemsPerThread      = 2;
inline constexpr int kSamplerPartialTileItems    = kSamplerBlock * kSamplerItemsPerThread;
inline constexpr int kSamplerGroupBlock          = 256;
// Every merge level takes the top `cap` keys of up to one warp's worth of sorted lists: a group
// merges this many partial tiles, and the last group merges the groups.
inline constexpr int kSamplerPartialsPerGroup    = 32;
// The public top-k cap: top_k<=0 or above it keeps this many candidates.
inline constexpr int kSamplerCandidateCap        = 64;
inline constexpr int kSamplerMaxColumns          = 16;
inline constexpr int kSpeculativeSamplerMaxColumns = 64;

// One warp sorts 32 lanes x kSamplerItemsPerThread keys of a partial tile and keeps the first cap.
static_assert(kSamplerCandidateCap <= 32 * kSamplerItemsPerThread,
              "a warp's sorted tile slice must hold the candidate cap");

__host__ __device__ inline int sampler_group_count(int partial_blocks) {
    return div_up(partial_blocks, kSamplerPartialsPerGroup);
}

// The multi-block route is deliberately finite: the last group merges every group's list in one
// warp. Ordinary sampling retains its 16-column route; speculative verification explicitly opts
// into up to 64 columns.
__host__ __device__ inline bool sampler_multiblock_ok(int vocab, int cols, int partial_blocks,
                                                      int group_count,
                                                      int max_columns = kSamplerMaxColumns) {
    return vocab > kSamplerTileItems && cols > 0 && cols <= max_columns && partial_blocks > 0 &&
           group_count > 0 && group_count <= kSamplerPartialsPerGroup;
}

struct SamplingWorkspace {
    unsigned long long* partial_keys         = nullptr;
    std::int32_t* dist_idx                   = nullptr;
    float* dist_prob                         = nullptr;
    std::int32_t* dist_support               = nullptr;
    std::int32_t* group_done                 = nullptr;
    std::int32_t* speculative_finalize_count = nullptr;
    std::int32_t partial_stride              = 0;
};

struct SamplingWorkspaceLayout {
    TensorRegion partial_keys;
    TensorRegion dist_idx;
    TensorRegion dist_prob;
    TensorRegion dist_support;
    TensorRegion group_done;
    TensorRegion speculative_finalize_count;
    std::size_t bytes           = 0;
    std::int32_t partial_stride = 0;
    bool multiblock             = false;

    [[nodiscard]] SamplingWorkspace bind(DeviceSpan backing) const {
        if (!multiblock) { return {}; }
        return {
            static_cast<unsigned long long*>(partial_keys.bind(backing).data),
            static_cast<std::int32_t*>(dist_idx.bind(backing).data),
            static_cast<float*>(dist_prob.bind(backing).data),
            static_cast<std::int32_t*>(dist_support.bind(backing).data),
            static_cast<std::int32_t*>(group_done.bind(backing).data),
            static_cast<std::int32_t*>(speculative_finalize_count.bind(backing).data),
            partial_stride,
        };
    }
};

inline SamplingWorkspaceLayout
make_sampling_workspace_layout(std::int32_t token_domain, std::int32_t columns,
                               std::int32_t max_columns = kSamplerMaxColumns) {
    SamplingWorkspaceLayout out;
    if (token_domain <= 0 || columns <= 0) { return out; }

    const std::int32_t partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const std::int32_t groups         = sampler_group_count(partial_blocks);
    if (!sampler_multiblock_ok(token_domain, columns, partial_blocks, groups, max_columns)) {
        return out;
    }

    out.multiblock     = true;
    out.partial_stride = partial_blocks + groups;

    LayoutBuilder layout;
    out.partial_keys =
        layout.add_tensor(DType::I64, {kSamplerCandidateCap, out.partial_stride, columns}, 256,
                          "sampling partial keys");
    out.dist_idx  = layout.add_tensor(DType::I32, {kSamplerCandidateCap, columns}, 256,
                                      "sampling speculative distribution indices");
    out.dist_prob = layout.add_tensor(DType::FP32, {kSamplerCandidateCap, columns}, 256,
                                      "sampling speculative distribution probabilities");
    out.dist_support =
        layout.add_tensor(DType::I32, {columns}, 256, "sampling speculative support sizes");
    out.group_done = layout.add_tensor(DType::I32, {columns}, 256, "sampling group counters");
    out.speculative_finalize_count =
        layout.add_tensor(DType::I32, {1}, 256, "sampling speculative finalize counter");
    // Speculative acceptance replicates this complete layout once per batch row. Preserve the
    // strongest member alignment in the row stride so every replicated partial-key array remains
    // correctly aligned.
    out.bytes = layout.finish(256, "sampling workspace");
    return out;
}

} // namespace ninfer::ops
