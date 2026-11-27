#pragma once
#include "models/qwen3_5/program/program.h"
#include "ninfer/ops/attention_geometry.h"

namespace ninfer::models::qwen3_5::detail {

// Rounds of at most this many rows replay a CUDA Graph captured for their exact row count. A wider
// ordinary round runs its kernels directly at its exact row count: its kernels are long enough that
// the launches keep ahead of the device, and it needs no capture, padding or graph memory.
inline constexpr std::uint32_t kMaximumGraphRows = 8;

[[nodiscard]] constexpr std::uint32_t graph_rows(std::uint32_t max_concurrency) {
    return max_concurrency < kMaximumGraphRows ? max_concurrency : kMaximumGraphRows;
}

// One-token decode profiles; profiles whose attention takes the same route share an executable.
[[nodiscard]] std::vector<GraphExecutionProfile>
ordinary_graph_profiles(std::uint32_t capacity, ops::AttentionHeadGeometry attention,
                        KvCacheStorage storage);
// An MTP round verifies verify_window drafts and proposes draft_window for the next round. The
// target's attention geometry and KV storage decide which attention route each profile takes,
// and so which profiles can share an executable.
[[nodiscard]] std::vector<GraphExecutionProfile>
mtp_graph_profiles(std::uint32_t capacity, std::uint32_t verify_window, std::uint32_t draft_window,
                   ops::AttentionHeadGeometry attention, KvCacheStorage storage);

[[nodiscard]] inline std::vector<GraphExecutionProfile>
mtp_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window,
                   ops::AttentionHeadGeometry attention, KvCacheStorage storage) {
    return mtp_graph_profiles(capacity, draft_window, draft_window, attention, storage);
}

// The narrowest width adaptive MTP captures: the controller never selects fewer than three while
// three drafts are ready, and a round with fewer ready drafts verifies them at this width.
[[nodiscard]] constexpr std::uint32_t mtp_minimum_adaptive_window(std::uint32_t draft_window) {
    return draft_window < 3 ? draft_window : 3U;
}

// The widths a masked drafter's rounds verify without a copy: the draft window, and under
// --adaptive-mtp also half of it (MaskedWidthController picks one a round), narrowest first.
[[nodiscard]] inline std::vector<std::uint32_t> masked_neural_windows(std::uint32_t draft_window,
                                                                      bool adaptive) {
    const std::uint32_t narrow = draft_window / 2U;
    if (!adaptive || narrow == 0) { return {draft_window}; }
    return {narrow, draft_window};
}

[[nodiscard]] std::vector<GraphExecutionProfile> dflash_graph_profiles(SpeculativeBackend backend,
                                                                       std::uint32_t capacity,
                                                                       std::uint32_t draft_window,
                                                                       std::uint32_t batch_size);

} // namespace ninfer::models::qwen3_5::detail
