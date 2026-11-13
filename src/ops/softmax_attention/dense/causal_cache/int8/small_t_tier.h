#pragma once

// The launch tiers of the INT8-family small-T partial kernels, without the kernels: the host router
// (int8/small_t_dispatch.cpp) picks one per launch, and each tier of each query width, geometry and
// input compiles in a unit of its own (kernel_unit.cu.in), so an edit of the routing compiles
// no kernel and the tiers compile in parallel.

#include <cstdint>
#include <string_view>

namespace ninfer::ops::detail {

// X(id, profile name, warps per CTA, CTAs per SM, keys per tile, dynamic arena, QK split,
//   early fetch) for every shape of the tiled kernel (int8/small_t.cuh). A device profile names a
// shape "<warps>x<CTAs per SM>x<key block>" with "d" for the dynamic arena, "q" for split QK (which
// takes the dynamic arena) and "e" for EarlyFetch.
#define NINFER_SMALL_T_I8_TILED_TIERS(X)                                                           \
    X(W2C4K32, "2x4x32", 2, 4, 32, false, 1, false)                                                \
    X(W4C2K32, "4x2x32", 4, 2, 32, false, 1, false)                                                \
    X(W8C2K32, "8x2x32", 8, 2, 32, false, 1, false)                                                \
    X(W16C1K32, "16x1x32", 16, 1, 32, false, 1, false)                                             \
    X(W32C1K32, "32x1x32", 32, 1, 32, false, 1, false)                                             \
    X(W6C2K32, "6x2x32", 6, 2, 32, false, 1, false)                                                \
    X(W12C1K32, "12x1x32", 12, 1, 32, false, 1, false)                                             \
    X(W24C1K32, "24x1x32", 24, 1, 32, false, 1, false)                                             \
    X(W12C1K64D, "12x1x64d", 12, 1, 64, true, 1, false)                                            \
    X(W2C2K32Q, "2x2x32q", 2, 2, 32, true, 2, false)                                               \
    X(W4C2K32Q, "4x2x32q", 4, 2, 32, true, 2, false)                                               \
    X(W8C2K32Q, "8x2x32q", 8, 2, 32, true, 2, false)                                               \
    X(W6C2K32Q, "6x2x32q", 6, 2, 32, true, 2, false)                                               \
    X(W12C1K32Q, "12x1x32q", 12, 1, 32, true, 2, false)                                            \
    X(W4C2K32E, "4x2x32e", 4, 2, 32, false, 1, true)                                               \
    X(W8C2K32E, "8x2x32e", 8, 2, 32, false, 1, true)                                               \
    X(W16C1K32E, "16x1x32e", 16, 1, 32, false, 1, true)                                            \
    X(W6C2K32E, "6x2x32e", 6, 2, 32, false, 1, true)                                               \
    X(W12C1K32E, "12x1x32e", 12, 1, 32, false, 1, true)                                            \
    X(W4C2K32QE, "4x2x32qe", 4, 2, 32, true, 2, true)                                              \
    X(W8C2K32QE, "8x2x32qe", 8, 2, 32, true, 2, true)                                              \
    X(W6C2K32QE, "6x2x32qe", 6, 2, 32, true, 2, true)                                              \
    X(W12C1K32QE, "12x1x32qe", 12, 1, 32, true, 2, true)

enum class SmallTI8Tier : std::uint8_t {
#define NINFER_SMALL_T_I8_TIER_ID(id, name, warps, ctas, keys, dynamic, qk_split, early) id,
    NINFER_SMALL_T_I8_TILED_TIERS(NINFER_SMALL_T_I8_TIER_ID)
#undef NINFER_SMALL_T_I8_TIER_ID
    // The pipelined kernel (int8/small_t_pipelined.cuh, profile name "8x2x32p"): INT8 keys and
    // values, whole splits of pages.
    Pipelined,
};

inline constexpr int kSmallTI8TiledTiers = static_cast<int>(SmallTI8Tier::Pipelined);

struct SmallTI8TierShape {
    int warps;
    int ctas_per_sm;
    int key_block;
    bool dynamic_arena;
    int qk_split;
    bool early_fetch;
};

[[nodiscard]] constexpr SmallTI8TierShape small_t_i8_tier_shape(SmallTI8Tier tier) {
    switch (tier) {
#define NINFER_SMALL_T_I8_TIER_SHAPE(id, name, warps, ctas, keys, dynamic, qk_split, early)        \
    case SmallTI8Tier::id:                                                                         \
        return {warps, ctas, keys, dynamic, qk_split, early};
        NINFER_SMALL_T_I8_TILED_TIERS(NINFER_SMALL_T_I8_TIER_SHAPE)
#undef NINFER_SMALL_T_I8_TIER_SHAPE
    case SmallTI8Tier::Pipelined:
        break;
    }
    return {8, 2, 32, true, 1, false};
}

[[nodiscard]] constexpr std::string_view small_t_i8_tier_name(SmallTI8Tier tier) {
    switch (tier) {
#define NINFER_SMALL_T_I8_TIER_NAME(id, name, warps, ctas, keys, dynamic, qk_split, early)         \
    case SmallTI8Tier::id:                                                                         \
        return name;
        NINFER_SMALL_T_I8_TILED_TIERS(NINFER_SMALL_T_I8_TIER_NAME)
#undef NINFER_SMALL_T_I8_TIER_NAME
    case SmallTI8Tier::Pipelined:
        return "8x2x32p";
    }
    return {};
}

// The 16-row Q tiles of a launch: query width times the geometry's group of query heads per KV
// head, at most 48 rows.
[[nodiscard]] constexpr int small_t_i8_row_tiles(int group_size, int width) {
    return (width * group_size + 15) / 16;
}

// Whether the tiled kernel has `tier` for a launch of `row_tiles` Q tiles: the shapes whose warps
// divide among the row tiles and that were measured there. Each one compiles for every query width
// of that row-tile count; the pipelined kernel compiles for every width.
[[nodiscard]] constexpr bool small_t_i8_tier_built(int row_tiles, SmallTI8Tier tier) {
    using enum SmallTI8Tier;
    switch (row_tiles) {
    case 1:
        return tier == W2C4K32 || tier == W4C2K32 || tier == W8C2K32 || tier == W16C1K32 ||
               tier == W2C2K32Q || tier == W4C2K32Q || tier == W4C2K32E || tier == W8C2K32E ||
               tier == W16C1K32E || tier == W4C2K32QE || tier == Pipelined;
    case 2:
        return tier == W4C2K32 || tier == W8C2K32 || tier == W16C1K32 || tier == W32C1K32 ||
               tier == W4C2K32Q || tier == W8C2K32Q || tier == W4C2K32E || tier == W8C2K32E ||
               tier == W16C1K32E || tier == W4C2K32QE || tier == W8C2K32QE || tier == Pipelined;
    case 3:
        return tier == W6C2K32 || tier == W12C1K32 || tier == W12C1K64D || tier == W24C1K32 ||
               tier == W6C2K32Q || tier == W12C1K32Q || tier == W6C2K32E || tier == W12C1K32E ||
               tier == W6C2K32QE || tier == W12C1K32QE || tier == Pipelined;
    default:
        return false;
    }
}

} // namespace ninfer::ops::detail
