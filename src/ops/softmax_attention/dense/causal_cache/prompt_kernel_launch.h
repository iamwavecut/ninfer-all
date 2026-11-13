#pragma once

// The causal prompt attention launches over BF16, bf16p and INT8-family caches that the host
// launcher (prompt.cpp) calls, and their arguments: plain pointers and counts, so the launcher
// compiles without the kernels and a kernel unit without the routing. Of each geometry and
// metadata form, the fast INT8-family kernels, the standard ones of the codings with packed int4
// values and the BF16 kernels compile in one unit, the standard ones over INT8 values in another
// (kernel_unit.cu.in; sources.cmake gives why).

#include "ops/softmax_attention/dense/causal_cache/int8/coding.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace ninfer::ops::detail {

// How a launch finds its block table and live columns: the one table of a single sequence's layer
// view (PagedKVDirectMetadata), or a batch row's table with every column live or `valid_columns[0]`
// of them (PagedKVBatchMetadata<false> and <true>).
enum class PromptMetadataForm : std::uint8_t { Direct, Batch, BatchMasked };

struct PromptLaunch {
    const void* q                 = nullptr; // BF16 [D, q_heads, tokens]
    const std::int32_t* positions = nullptr;
    float scale                   = 0.0F;
    void* out                     = nullptr; // BF16 [D, q_heads, tokens]
    std::int32_t tokens           = 0;
    std::int32_t q_heads          = 0;
    const void* cache_k           = nullptr;
    const void* cache_v           = nullptr;
    const void* cache_k_scale     = nullptr; // scales, or a bf16p cache's side records
    const void* cache_v_scale     = nullptr;
    // The metadata's fields (PromptMetadataForm): the direct form reads `tables` alone.
    const std::int32_t* tables        = nullptr;
    const std::int32_t* valid_columns = nullptr;
    const std::int32_t* table_rows    = nullptr;
    std::int32_t table_stride         = 0;
    // A bf16p cache: its planes are packed rows over side records.
    bool packed_planes = false;
    // The INT8-family kernels: the planes' coding, FP16 PV sums and packed GQA query heads (the
    // standard kernel), and the fast kernel's plan: warps per CTA and key splits, with the split
    // partials (causal_prompt_i8_fast_plan).
    CausalI8Coding coding = CausalI8Coding::Int8;
    bool pv_f16           = false;
    bool pack_gqa         = false;
    std::int32_t warps    = 4;
    std::int32_t splits   = 1;
    float* split_rows     = nullptr;
    float2* split_stats   = nullptr;
    cudaStream_t stream   = nullptr;
};

// The BF16 kernel (bf16/prompt.cuh) in the current device's shape set, over BF16 or bf16p planes.
template <typename Geometry, PromptMetadataForm Form>
void launch_prompt_bf16(const PromptLaunch& launch);

// The standard INT8-family kernel (int8/prompt.cuh), where the build has the standard shape set:
// the codings with packed int4 values (rk4v4, rk4v4-e8, rk2v4-e8, rk8v4) where PackedValueCodings,
// INT8 values (int8-g64) otherwise. The two compile in separate units (sources.cmake).
template <typename Geometry, PromptMetadataForm Form, bool PackedValueCodings>
void launch_prompt_i8(const PromptLaunch& launch);

// The fast INT8-family kernel (int8/prompt_fast.cuh) at the launch's plan; a split plan leaves its
// partials for launch_prompt_i8_fast_merge.
template <typename Geometry, PromptMetadataForm Form>
void launch_prompt_i8_fast(const PromptLaunch& launch);

// Merges the fast kernel's split partials into `out` (int8/prompt_fast_merge.cu); `valid_columns`
// masks a batch row's columns.
void launch_prompt_i8_fast_merge(const PromptLaunch& launch);

// Whether the fast kernel's eight-warp CTA fits a block of the Turing or the standard shape set
// with packed or INT8 values (causal_prompt_i8_fast_wide in prompt_policy.h asks it).
[[nodiscard]] bool prompt_i8_fast_wide_launchable(bool packed_values, bool turing_shapes);

} // namespace ninfer::ops::detail
