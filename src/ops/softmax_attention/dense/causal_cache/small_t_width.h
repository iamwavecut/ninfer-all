#pragma once

// The query widths a causal small-T launch takes, as compile-time values: the host routers and
// the kernel units that cover every width of a geometry both select one with with_small_t_width.

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {

// The widest query width a geometry's small-T kernels take: 48 query rows of 16-row Q tiles.
template <typename Geometry>
inline constexpr int kSmallTWidest = Geometry::GroupSize == 6 ? 8 : 6;

// Calls `f.template operator()<Width>()` with `width` as a compile-time value. Throws for a width
// outside [1, kSmallTWidest<Geometry>].
template <typename Geometry, typename F>
void with_small_t_width(std::int32_t width, F&& f) {
    const bool found = [&]<int... Index>(std::integer_sequence<int, Index...>) {
        return ((width == Index + 1 ? (f.template operator()<Index + 1>(), true) : false) || ...);
    }(std::make_integer_sequence<int, kSmallTWidest<Geometry>>{});
    if (!found) { throw std::invalid_argument("causal small-T launch: unsupported query width"); }
}

} // namespace ninfer::ops::detail
