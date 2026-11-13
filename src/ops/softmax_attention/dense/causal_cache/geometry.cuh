#pragma once

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {

// The head dimension of every causal attention geometry.
inline constexpr int kCausalHeadDim = 256;

// A causal attention geometry fixes the query heads per KV head (GroupSize) and the small-T split
// scale at compile time. How many KV heads a call attends with is a run-time count: the model's,
// or a tensor-parallel rank's run of them, at most ModelKVHeads. Every kernel takes it from its
// launch grid, which spans the call's KV heads (small-T partials, packed-GQA prompts) or query
// heads (reductions, prompts, merges). ModelQHeads names the model geometry whose device-profile
// routes a call takes.
template <int GroupSizeValue, int ModelKVHeadsValue, int SmallTSplitScaleValue>
struct CausalAttentionGeometry {
    static_assert(GroupSizeValue > 0 && ModelKVHeadsValue > 0 && SmallTSplitScaleValue > 0);

    static constexpr int GroupSize           = GroupSizeValue;
    static constexpr int ModelKVHeads        = ModelKVHeadsValue;
    static constexpr int ModelQHeads         = GroupSize * ModelKVHeads;
    static constexpr int SmallTSplitScale    = SmallTSplitScaleValue;
    static constexpr int SmallTMaximumSplits = 85 * SmallTSplitScale;
};

using CausalD256H24Kv4 = CausalAttentionGeometry<6, 4, 1>;
using CausalD256H16Kv2 = CausalAttentionGeometry<8, 2, 2>;

// Whether `query_heads` over `kv_heads` is one of those geometries or a run of its KV heads.
[[nodiscard]] constexpr bool causal_attention_geometry_supported(std::int32_t query_heads,
                                                                 std::int32_t kv_heads) {
    if (kv_heads <= 0 || query_heads % kv_heads != 0) { return false; }
    const std::int32_t group = query_heads / kv_heads;
    return (group == CausalD256H24Kv4::GroupSize && kv_heads <= CausalD256H24Kv4::ModelKVHeads) ||
           (group == CausalD256H16Kv2::GroupSize && kv_heads <= CausalD256H16Kv2::ModelKVHeads);
}

// Calls `f.template operator()<Geometry>()` with the geometry of `query_heads` over `kv_heads`.
// Throws for any other pair.
template <typename F>
decltype(auto) with_causal_geometry(std::int32_t query_heads, std::int32_t kv_heads, F&& f) {
    if (!causal_attention_geometry_supported(query_heads, kv_heads)) {
        throw std::invalid_argument("causal attention: unsupported head geometry");
    }
    if (query_heads / kv_heads == CausalD256H24Kv4::GroupSize) {
        return f.template operator()<CausalD256H24Kv4>();
    }
    return f.template operator()<CausalD256H16Kv2>();
}

// The same for a call named by its query heads alone. A call holds whole groups of query heads, and
// the geometries' counts (6, 12, 18 or 24, and 8 or 16) do not overlap.
template <typename F>
decltype(auto) with_causal_query_geometry(std::int32_t query_heads, F&& f) {
    const std::int32_t group = query_heads % CausalD256H24Kv4::GroupSize == 0
                                   ? CausalD256H24Kv4::GroupSize
                                   : CausalD256H16Kv2::GroupSize;
    return with_causal_geometry(query_heads, query_heads / group, static_cast<F&&>(f));
}

// The query heads of the model whose geometry a call with `query_heads` takes: a tensor-parallel
// share takes its model's measured routes, chunks and tiles.
[[nodiscard]] inline std::int32_t causal_attention_model_query_heads(std::int32_t query_heads) {
    return with_causal_query_geometry(query_heads,
                                      []<typename Geometry>() { return Geometry::ModelQHeads; });
}

} // namespace ninfer::ops
