#pragma once

// Host routing of the INT8-family small-T partial launch: the device-profile key of its tiers, the
// tier a launch takes and the launch itself (int8/small_t_dispatch.cpp). No kernel compiles with
// it: each tier compiles in units of its own (int8/small_t_launch.h).

#include "ninfer/kv_cache_storage.h"
#include "ops/common/device_limits.h"
#include "ops/softmax_attention/dense/causal_cache/int8/small_t_launch.h"

#include <cstdint>
#include <string>

namespace ninfer::ops::detail {

// Whether a card of a shape set runs the partial kernel's "12x1x64d" tier: its 64-key dynamic arena
// of 64 KiB beside the kernel's static tiles needs more than 64 KiB of shared memory per block,
// which Turing does not have; there 2K-8K windows take 12x1x32 and the tier is no calibration
// candidate.
[[nodiscard]] constexpr bool small_t_i8_key_block64(bool turing_shapes) {
    return shape_max_shared_bytes_per_block(turing_shapes) > 64 * 1024;
}

// Whether any card this build runs takes it, and whether the current device does.
inline constexpr bool kSmallTI8KeyBlock64Built = kBuildStandardShapes;

[[nodiscard]] inline bool small_t_i8_key_block64_here() {
    return small_t_i8_key_block64(turing_shapes_here());
}

// Device-profile key of the INT8-family partial kernel's launch tier:
// "attn_i8_small/h24/rk8v4/w8". The axis is the launch's implementation window.
std::string small_t_i8_route_key(std::int32_t q_heads, KvCacheStorage storage, std::int32_t width);

// The CTAs per SM of a profile-routed tier at this window, or zero when the profile routes none.
int small_t_i8_routed_ctas_per_sm(std::int32_t q_heads, KvCacheStorage storage, std::int32_t width,
                                  std::int32_t window);

// The tier a launch of `width` query columns over `storage` takes on the current device at
// `window` keys: one 32-key tier per row-tile count for invariant reductions, else the device
// profile's tier for the window, else the compiled default.
[[nodiscard]] SmallTI8Tier small_t_i8_select_tier(std::int32_t group_size,
                                                  std::int32_t model_q_heads, std::int32_t width,
                                                  KvCacheStorage storage,
                                                  CausalKeyPartition partition,
                                                  std::int32_t window);

// Launches the partial kernel of `launch` for `width` columns over `storage` on the current device:
// Volta's kernel there, else the tier small_t_i8_select_tier picks at `window`. The router fills
// launch.pv_f16.
template <typename Geometry, typename CacheInput>
void small_t_i8_partial_launch(std::int32_t width, KvCacheStorage storage, std::int32_t window,
                               SmallTI8Launch launch, CacheInput input);

// Loads the CUDA modules of widths [1, last] for this input that the current device can launch.
template <typename Geometry, typename CacheInput>
void small_t_i8_load_modules(std::int32_t last);

} // namespace ninfer::ops::detail
