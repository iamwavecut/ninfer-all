#pragma once

#include "runtime/contract/request.h"
#include "runtime/contract/timing.h"
#include <array>
#include <compare>
#include <span>
#include <stdexcept>

namespace ninfer::runtime {

// Prefill units a prompt may be passed over for shorter ones before it runs ahead of them: each
// unit goes to the shortest remaining prompt, so a short or prefix-cached request is not held
// behind a long prompt for its whole prefill, and this bound keeps the long one from starving.
inline constexpr std::uint32_t kPrefillMaxSkip = 8;

struct LaneId {
    std::uint32_t value = 0;

    [[nodiscard]] friend constexpr bool operator==(LaneId, LaneId) noexcept  = default;
    [[nodiscard]] friend constexpr auto operator<=>(LaneId, LaneId) noexcept = default;
};

enum class ConsumeStatus : std::uint8_t {
    Consumed,
    InvariantMismatch,
};

enum class CommitDisposition : std::uint8_t {
    Active,
    Finishable,
    CancelledReleased,
};

// The product Engine only needs statistics for rows whose sequence is released by commit.
// Direct diagnostic callers may temporarily request cumulative snapshots for every row.
enum class CommitObservation : std::uint8_t {
    ReleasedRowsOnly,
    AllRows,
};

struct CommitDecision {
    std::uint32_t accepted_tokens = 0;
    bool terminal                 = false;
    bool cancelled                = false;
    // Copied unchanged from the corresponding OutputDecision; still relative to this row's
    // accepted span.
    std::optional<std::uint32_t> prefix_execution_split_after;
    // Copied unchanged from the corresponding OutputDecision; relative to the accepted span.
    std::optional<std::uint32_t> stop_hold_after;
    // The row ends at a stop string: the visible text ends before the tokens the hold covers.
    bool stop_cut = false;
};

struct BeginSummary {
    std::uint32_t prompt_tokens        = 0;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::Root;

    [[nodiscard]] friend constexpr bool operator==(BeginSummary, BeginSummary) noexcept = default;
};

// One generated token's log probability and its top alternatives as the device gathered them,
// before the frontend attaches token bytes.
struct RawTokenLogprob {
    TokenId id        = 0;
    float logprob     = 0.0f;
    std::array<TokenId, kMaximumTokenLogprobs> top_ids{};
    std::array<float, kMaximumTokenLogprobs> top_values{};
    // The typed readout (ExecutionOptions::readout_tokens) on the round's first token: the first
    // readout_count values are the named tokens' log probabilities in the request's order.
    std::uint32_t readout_count = 0;
    std::array<float, kMaximumReadoutTokens> readout_values{};
};

// Thrown by a Program when a prompt's first sampled token is the sampler's non-finite marker or
// lies outside the vocabulary. No logits vouch for what the prompt computed on the way, so the
// Engine recovers as from an out-of-memory failure: it ends the running requests with
// RequestErrorKind::NumericalFailure and drops every cached prefix. A decode round reports such a
// row on its PendingBatch instead, and only that row's request ends.
class NumericalFault final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct GeneratedRound {
    std::span<const TokenId> tokens;
    // Aligned with `tokens` when the request asked for logprobs, else empty.
    std::span<const RawTokenLogprob> logprobs;
    // A branched read's Begin round (ResolvedExecutionOptions::branches): one record per branch,
    // in branch order; `tokens` then holds the first branch's token.
    std::span<const RawTokenLogprob> branches;
};

struct BatchedGeneratedRound {
    std::span<const TokenId> tokens;
    std::span<const std::int32_t> row_counts;
    // Aligned with `tokens` ([rows,row_stride]) when a row asked for logprobs, else empty.
    std::span<const RawTokenLogprob> logprobs;
    std::uint32_t row_stride = 1;
    ExecutionTiming timing;
};

struct PrefillStepResult {
    BeginSummary summary;
    GeneratedRound round;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    ExecutionTiming timing;
};

struct RoundBudget {
    std::uint32_t generated_tokens_remaining = 0;
};

} // namespace ninfer::runtime
