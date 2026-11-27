#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/program.h"

#include <cstdint>

namespace ninfer::models::qwen3_5 {

inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 15;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 15;
// Abort salvage publishes the live state only when it covers enough committed work that the
// saved rebuild outweighs the checkpoint's retention cost.
inline constexpr std::uint32_t kSalvageMinFrontier = 1024;

// The MTP draft head's attention window (--mtp-attention-window): its first 64 keys and the newest
// `keys` before its query, for a row whose first query position is `from` or later; zero `keys`
// read the whole history.
struct MtpAttentionWindow {
    std::uint32_t keys = 0;
    std::uint32_t from = 0;
};
// --mtp-attention-window auto: past 32,768 positions the draft reads the newest 4,096 keys. On an
// RTX 3090 with Qwen3.8-27B (rk8v4, three drafts) that decoded 4.6% faster at 106,935 tokens and
// 5.7% at 162,840; at 53,423 tokens 1.0%.
inline constexpr MtpAttentionWindow kAutomaticMtpAttentionWindow{4096, 32768};

// Rows of 17..64 verification columns keep 16-bit activations in FP8 residual projections at any
// batch size; first/last bound the per-row width, not the round's aggregate columns.
[[nodiscard]] inline bool wide_residual_verification(TextPhase phase, std::int32_t first,
                                                     std::int32_t last) {
    return phase == TextPhase::Verify && first > 16 && first <= last && last <= 64;
}

} // namespace ninfer::models::qwen3_5

namespace ninfer::models::qwen3_5::detail {
using ContractAccess = RuntimeContractAccess;

[[nodiscard]] inline std::uint32_t backend_frontier_at(SpeculativeBackend backend,
                                                       std::uint32_t main_frontier) noexcept {
    if (backend == SpeculativeBackend::Mtp) { return main_frontier == 0 ? 0U : main_frontier - 1U; }
    return backend == SpeculativeBackend::DFlash ? main_frontier : 0U;
}

} // namespace ninfer::models::qwen3_5::detail
