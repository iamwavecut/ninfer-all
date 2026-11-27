#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#    define NINFER_ROUTE_BIAS_HD __host__ __device__
#else
#    define NINFER_ROUTE_BIAS_HD
#endif

namespace ninfer {

// The route bias toward the experts the device holds (HybridExpertOptions::route_bias, lossy): a
// router adds `bias` to the score of each expert whose weights lie in device memory now before it
// ranks the experts, and weights the chosen ones by their unbiased scores. An expert is held when
// its table entry (the address its first projection is read from) lies in [low, high) or in one
// of the `count` device ranges `ranges` ([i][low, high)), which the delivery service may change
// between calls. The entries and ranges a call reads may change while it runs; an expert then
// counts as held or not, either of which routes it to valid weights. A default value (no tables)
// leaves routing as it is.
struct ExpertRouteBias {
    const void* const* tables = nullptr; // [experts]
    std::uintptr_t low = 0, high = 0;
    const std::uintptr_t* ranges = nullptr; // device, [count][2]
    int count  = 0;
    float bias = 0.0f;

    [[nodiscard]] NINFER_ROUTE_BIAS_HD bool enabled() const noexcept { return tables != nullptr; }
};

#if defined(__CUDACC__)
// `bias` when expert `expert` is held, else zero (also with no tables).
__device__ __forceinline__ float expert_route_bonus(const ExpertRouteBias& b, int expert) {
    if (b.tables == nullptr) { return 0.0f; }
    const auto at = reinterpret_cast<std::uintptr_t>(b.tables[expert]);
    bool held     = at >= b.low && at < b.high;
    for (int i = 0; i < b.count; ++i) {
        held = held || (at >= b.ranges[2 * i] && at < b.ranges[2 * i + 1]);
    }
    return held ? b.bias : 0.0f;
}
#endif

} // namespace ninfer
