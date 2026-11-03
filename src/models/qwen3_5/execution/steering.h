#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_5 {

// The edit sites of a model with `layers` layers: the token embedding, then each layer's mixer
// output and FFN output.
[[nodiscard]] constexpr std::uint32_t steering_sites(std::uint32_t layers) noexcept {
    return 1U + 2U * layers;
}
inline constexpr std::uint32_t kEmbeddingSteeringSite = 0;
[[nodiscard]] constexpr std::uint32_t mixer_steering_site(std::uint32_t layer) noexcept {
    return 1U + 2U * layer;
}
[[nodiscard]] constexpr std::uint32_t ffn_steering_site(std::uint32_t layer) noexcept {
    return 2U + 2U * layer;
}

} // namespace ninfer::models::qwen3_5

namespace ninfer::models::qwen3_5::execution {

// The loaded steering (frontend/steering.h) on one rank. Tensors whose part is not loaded are
// empty (null data).
struct SteeringRank {
    // FP32 [hidden, adapters]: the directions the adapters remove.
    Tensor directions;
    // FP32 [adapters, sites]: each adapter's factor at each edit site.
    Tensor site_factors;
    // FP32 [adapters, lanes]: the scales of the request on each lane's KV table row, written when
    // the request takes the lane.
    Tensor lane_scales;
    // FP32 [hidden, layers]: the control vectors' sum, one column per layer.
    Tensor vectors;
    // FP32 [adapters, columns]: what a pass has measured along each direction since the residual's
    // last write (ops::steer_residual), for the widest pass.
    Tensor measures;
    // I32 [columns], rank 0 only: each column's KV table row in a pass over several sequences'
    // prompts (TextContext::prefill_final_batch).
    Tensor column_rows;
};

struct SteeringView {
    std::int32_t adapters = 0;
    // Layers whose control vector column is added after the layer; empty without control vectors.
    std::vector<std::uint8_t> steered_layers;
    // One per rank, in rank order.
    std::vector<SteeringRank> ranks;
};

} // namespace ninfer::models::qwen3_5::execution
