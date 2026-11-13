#pragma once

// The INT8-family small-T partial launches the host router (int8/small_t_dispatch.cpp) calls. Each
// instantiation compiles in a unit of its own, generated per tier, query width, geometry and input
// (kernel_unit.cu.in): an edit of one tier's launch or of the routing compiles no other kernel,
// and the tiers compile in parallel.

#include "ops/softmax_attention/dense/causal_cache/cache_input.h"
#include "ops/softmax_attention/dense/causal_cache/int8/coding.h"
#include "ops/softmax_attention/dense/causal_cache/int8/small_t_tier.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_partial_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

// A partial launch with the INT8 family's own choices, which the router made at run time.
struct SmallTI8Launch : SmallTPartialLaunch {
    // Splits per full wave of a single-row launch (causal_small_t_active_splits), or zero.
    std::int32_t wave_splits = 0;
    CausalI8Coding coding    = CausalI8Coding::Int8;
    // FP16 sums of each key tile's PV products (the device profile's "attn_pv_f16").
    bool pv_f16 = false;
    // Load the unit's CUDA module and return without a launch; every other field is unread.
    bool load_module_only = false;
};

// The tiled kernel at one tier (int8/small_t_tiled.cuh). Defined for the tiers
// small_t_i8_tier_built() lists for the launch's row tiles.
template <typename Geometry, int TokenTile, typename CacheInput, SmallTI8Tier Tier>
void launch_small_t_i8_tiled(const SmallTI8Launch& launch, CacheInput input);

// The pipelined kernel (int8/small_t_pipelined_launch.cuh): INT8 keys and values only.
template <typename Geometry, int TokenTile, typename CacheInput>
void launch_small_t_i8_pipelined(const SmallTI8Launch& launch, CacheInput input);

// Volta's kernel for every INT8-family coding (int8/small_t_sm70_launch.cuh), in builds with sm_70.
template <typename Geometry, int TokenTile, typename CacheInput>
void launch_small_t_i8_sm70(const SmallTI8Launch& launch, CacheInput input);

} // namespace ninfer::ops::detail
