// ninfer::ops::detail - host routing of the INT8-family small-T partial launch: its device-profile
// key, the tier each launch takes and the launch of that tier's unit (int8/small_t_launch.h). An
// edit of the routing compiles this unit alone; the kernels compile in the generated tier units.
#include "ops/softmax_attention/dense/causal_cache/int8/small_t_route.h"

#include "ops/common/device_route.h"
#include "ops/common/device_tier.h"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_width.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::ops::detail {

std::string small_t_i8_route_key(std::int32_t q_heads, KvCacheStorage storage,
                                 std::int32_t width) {
    const char* coding = "int8";
    switch (storage) {
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        coding = "rk8v4";
        break;
    case KvCacheStorage::RotatedLloyd4KeyInt4Value:
        coding = "rk4v4";
        break;
    case KvCacheStorage::RotatedInt4KeyInt4ValueE8:
        coding = "rk4v4-e8";
        break;
    case KvCacheStorage::RotatedE8RootKeyInt4Value:
        coding = "rk2v4-e8";
        break;
    default:
        break;
    }
    return "attn_i8_small/h" + std::to_string(q_heads) + "/" + coding + "/w" +
           std::to_string(width);
}

int small_t_i8_routed_ctas_per_sm(std::int32_t q_heads, KvCacheStorage storage,
                                  std::int32_t width, std::int32_t window) {
    const std::string_view routed =
        device_route_schedule(small_t_i8_route_key(q_heads, storage, width), window);
    const std::size_t first = routed.find('x');
    if (first == std::string_view::npos) { return 0; }
    const std::size_t second = routed.find('x', first + 1);
    if (second == std::string_view::npos || second != first + 2) { return 0; }
    const char ctas = routed[first + 1];
    return ctas >= '1' && ctas <= '8' ? ctas - '0' : 0;
}

SmallTI8Tier small_t_i8_select_tier(std::int32_t group_size, std::int32_t model_q_heads,
                                    std::int32_t width, KvCacheStorage storage,
                                    CausalKeyPartition partition, std::int32_t window) {
    using enum SmallTI8Tier;
    const int rows = small_t_i8_row_tiles(group_size, width);
    if (partition.segmented()) {
        // Invariant reductions: one 32-key tile and one producer warp per row tile at every width,
        // so a row's arithmetic does not depend on the rows beside it.
        return rows == 3 ? W6C2K32 : W8C2K32;
    }
    // A device profile names the tier per window (int8/small_t_tier.h).
    const std::string_view routed =
        device_route_schedule(small_t_i8_route_key(model_q_heads, storage, width), window);
    // The pipelined kernel sweeps every page of its split: a position-split rank takes the tiled
    // kernels.
    if (routed == small_t_i8_tier_name(Pipelined) && storage == KvCacheStorage::Int8Group64 &&
        !partition.strided()) {
        return Pipelined;
    }
    if (!routed.empty()) {
        for (int index = 0; index < kSmallTI8TiledTiers; ++index) {
            const auto tier = static_cast<SmallTI8Tier>(index);
            if (routed == small_t_i8_tier_name(tier) && small_t_i8_tier_built(rows, tier) &&
                (tier != W12C1K64D || small_t_i8_key_block64_here())) {
                return tier;
            }
        }
    }
    if (width >= 6) {
        // Small grids need more warps per CTA. From 2K to 8K, Bc=64 halves key loop iterations;
        // dynamic smem avoids penalizing the long-context path.
        if (window > 128 && window <= 160) { return W24C1K32; }
        if (window <= 2054) { return W12C1K32; }
        if (window <= 8198) {
            return kSmallTI8KeyBlock64Built && small_t_i8_key_block64_here() ? W12C1K64D
                                                                              : W12C1K32;
        }
        return W6C2K32;
    }
    if (width == 5) {
        if (group_size == 6) {
            // Two Q row tiles for the 27B group of six.
            if (window > 128 && window <= 512) { return W32C1K32; }
            if (window <= 1029) { return W16C1K32; }
            return W8C2K32;
        }
        // Three Q row tiles for the 35B group of eight. The 24/12-warp routes retain eight/four
        // consumer warps per tile; the 6-warp route is reserved for long windows where CTA
        // residency wins.
        if (window <= 1029) { return W24C1K32; }
        if (window <= 4096) { return W12C1K32; }
        return W6C2K32;
    }
    if (width == 4) { return window <= 1029 ? W16C1K32 : W8C2K32; }
    return W8C2K32;
}

namespace {

// Sums each key tile's PV products in FP16 where the device profile's "attn_pv_f16" says "on"
// (GeForce parts accumulate FP16 at twice their FP32-accumulator rate); NINFER_SMALLT_PV_F16=0|1
// overrides the profile.
bool small_t_pv_f16() {
    static const int forced = [] {
        const char* value = std::getenv("NINFER_SMALLT_PV_F16");
        return value == nullptr ? -1 : (value[0] == '1' ? 1 : 0);
    }();
    if (forced >= 0) { return forced == 1; }
    return device_route_schedule("attn_pv_f16", 1) == "on";
}

// Calls `f.template operator()<Tier>()` for every tiled tier built for the width's row tiles.
template <typename Geometry, int TokenTile, typename F>
void for_each_built_tier(F&& f) {
    constexpr int kRows = small_t_i8_row_tiles(Geometry::GroupSize, TokenTile);
    [&]<int... Index>(std::integer_sequence<int, Index...>) {
        const auto one = [&]<SmallTI8Tier Tier>() {
            if constexpr (small_t_i8_tier_built(kRows, Tier)) { f.template operator()<Tier>(); }
        };
        (one.template operator()<static_cast<SmallTI8Tier>(Index)>(), ...);
    }(std::make_integer_sequence<int, kSmallTI8TiledTiers>{});
}

} // namespace

template <typename Geometry, typename CacheInput>
void small_t_i8_partial_launch(std::int32_t width, KvCacheStorage storage, std::int32_t window,
                               SmallTI8Launch launch, CacheInput input) {
#if defined(NINFER_TIER_VOLTA)
    if (current_device_volta()) {
        // Volta's own kernel (int8/small_t_sm70.cuh) takes every INT8-family storage: the same
        // grid, active splits and partial outputs. The tiled kernels are the other tiers'; a build
        // of Volta alone does not compile them.
        with_small_t_width<Geometry>(width, [&]<int TokenTile>() {
            launch_small_t_i8_sm70<Geometry, TokenTile, CacheInput>(launch, input);
        });
        return;
    }
#endif
    if constexpr (kBuildPostVolta) {
        const SmallTI8Tier tier = small_t_i8_select_tier(Geometry::GroupSize, Geometry::ModelQHeads,
                                                         width, storage, launch.partition, window);
        // Invariant reductions keep the products of every key tile in FP32.
        launch.pv_f16 = small_t_pv_f16() &&
                        (tier == SmallTI8Tier::Pipelined || !launch.partition.segmented());
        with_small_t_width<Geometry>(width, [&]<int TokenTile>() {
            if (tier == SmallTI8Tier::Pipelined) {
                launch_small_t_i8_pipelined<Geometry, TokenTile, CacheInput>(launch, input);
                return;
            }
            for_each_built_tier<Geometry, TokenTile>([&]<SmallTI8Tier Tier>() {
                if (tier == Tier) {
                    launch_small_t_i8_tiled<Geometry, TokenTile, CacheInput, Tier>(launch, input);
                }
            });
        });
    }
}

template <typename Geometry, typename CacheInput>
void small_t_i8_load_modules(std::int32_t last) {
    SmallTI8Launch launch{};
    launch.load_module_only = true;
    const std::int32_t widest = std::min(last, kSmallTWidest<Geometry>);
    for (std::int32_t width = 1; width <= widest; ++width) {
        with_small_t_width<Geometry>(width, [&]<int TokenTile>() {
#if defined(NINFER_TIER_VOLTA)
            // Volta launches its own kernel only; the other units carry no code it runs.
            if (current_device_volta()) {
                launch_small_t_i8_sm70<Geometry, TokenTile, CacheInput>(launch, CacheInput{});
                return;
            }
#endif
            if constexpr (kBuildPostVolta) {
                launch_small_t_i8_pipelined<Geometry, TokenTile, CacheInput>(launch, CacheInput{});
                for_each_built_tier<Geometry, TokenTile>([&]<SmallTI8Tier Tier>() {
                    launch_small_t_i8_tiled<Geometry, TokenTile, CacheInput, Tier>(launch,
                                                                                   CacheInput{});
                });
            }
        });
    }
}

#define NINFER_SMALL_T_I8_DISPATCH(GEOMETRY, INPUT)                                                \
    template void small_t_i8_partial_launch<GEOMETRY, INPUT>(std::int32_t, KvCacheStorage,         \
                                                             std::int32_t, SmallTI8Launch, INPUT); \
    template void small_t_i8_load_modules<GEOMETRY, INPUT>(std::int32_t);
NINFER_SMALL_T_I8_DISPATCH(CausalD256H24Kv4, CausalAppendInput)
NINFER_SMALL_T_I8_DISPATCH(CausalD256H24Kv4, CausalCachedInput)
NINFER_SMALL_T_I8_DISPATCH(CausalD256H16Kv2, CausalAppendInput)
NINFER_SMALL_T_I8_DISPATCH(CausalD256H16Kv2, CausalCachedInput)
#undef NINFER_SMALL_T_I8_DISPATCH

} // namespace ninfer::ops::detail
