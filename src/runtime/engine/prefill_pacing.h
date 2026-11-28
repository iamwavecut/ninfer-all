#pragma once

#include "runtime/contract/resources.h"
#include "runtime/engine/context_cache/context_cost.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace ninfer::runtime {

// How long a prefill step may hold the GPU while another lane generates
// (EngineOptions::prefill_stall_budget_ms, prefill_stall_round_budget).
//
// The Engine runs a whole prefill step between two decode rounds, so a stream decoding next to a
// long prompt waits one step for each token: about a second per 2048-token chunk of a 27B on an
// RTX 3090. With a budget the step is cut to the largest multiple of 128 tokens, at least 512, that
// the prefill cost model predicts to fit it. The model's static costs are scaled by the ratio of
// observed to predicted step times, so the cap follows the card and model it runs on; a deeper
// prefix, with more attention per token, is cut harder. The round budget is the decode rounds'
// measured time instead, and its steps go down to 256 tokens. Each cut step adds a decode round
// that reads the weights for the few requests then generating: with steps down to 128 tokens,
// Ternary Bonsai 2 at 64 requests and Qwen3.8-27B at 32 delivered 10.7% and 12.5% fewer tokens
// than with 256 on an RTX 3090 while new prompts arrived, and 384 gained under 3% more for a longer
// pause. Nothing here touches the GPU.
inline constexpr std::uint32_t kPacedPrefillAlignment     = 128;
inline constexpr std::uint32_t kMinimumPacedPrefillTokens = 512;
inline constexpr std::uint32_t kMinimumRoundPacedPrefillTokens = 256;
inline constexpr std::uint32_t kMaximumPrefillStallBudgetMs = 60'000;

// Predicted time of one prefill step of `tokens` after `prefix` tokens.
[[nodiscard]] inline std::uint64_t predicted_prefill_step_ns(const ContextMachineCostModel& costs,
                                                             std::uint64_t prefix,
                                                             std::uint64_t tokens) noexcept {
    if (tokens == 0) { return 0; }
    const auto chunk = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(tokens, std::numeric_limits<std::uint32_t>::max()));
    return costs.prefill_ns(make_prefill_work(prefix, tokens, 0, 0, chunk));
}

// The service quanta one prefill step charges against its request's projection, which counts one
// per planned step of `chunk` tokens in each segment of the prompt (a capture frontier ends one).
// `chunk` must be the step the Program runs. A step cut under the budget adds its `tokens` to
// `carried` and charges one for each whole `chunk` they then hold. A planned step (a whole chunk,
// or one that ends at a capture frontier or the prompt) charges one and drops the carry: a short
// planned step already charges a whole chunk for its segment's tail. So a prompt's steps never
// charge more than its planned steps, however cut and planned steps alternate as other lanes start
// and stop generating; the step that ends a prompt is never cut.
[[nodiscard]] inline std::uint64_t prefill_step_quanta(bool cut, std::uint32_t tokens,
                                                       std::uint32_t chunk,
                                                       std::uint32_t& carried) noexcept {
    if (!cut || chunk == 0) {
        carried = 0;
        return 1;
    }
    const std::uint64_t held = std::uint64_t{carried} + tokens;
    carried                  = static_cast<std::uint32_t>(held % chunk);
    return held / chunk;
}

class PrefillPacing {
public:
    PrefillPacing() = default;
    PrefillPacing(std::uint32_t stall_budget_ms, bool round_budget,
                  ContextMachineCostModel costs) noexcept
        : budget_ns_(static_cast<double>(stall_budget_ms) * 1e6),
          round_budget_(stall_budget_ms == 0 && round_budget), costs_(costs) {}

    [[nodiscard]] bool enabled() const noexcept { return budget() > 0.0; }
    [[nodiscard]] bool round_budget() const noexcept { return round_budget_; }

    // The tokens the next step may take, or 0 for the planned step. `planned` is the step the plan
    // would run (already clipped to the rest of the prompt) after `prefix` tokens.
    [[nodiscard]] std::uint32_t step_cap(std::uint64_t prefix, std::uint32_t planned) const noexcept {
        const std::uint32_t minimum =
            round_budget_ ? kMinimumRoundPacedPrefillTokens : kMinimumPacedPrefillTokens;
        if (!enabled() || planned <= minimum) { return 0; }
        const double budget = this->budget();
        const auto fits     = [&](std::uint64_t tokens) {
            return static_cast<double>(predicted_prefill_step_ns(costs_, prefix, tokens)) *
                       correction_ <=
                   budget;
        };
        if (fits(planned)) { return 0; }
        // The prediction grows with the step: the largest affordable multiple of the alignment,
        // by bisection over multiples, and never below the minimum step.
        std::uint32_t low  = minimum / kPacedPrefillAlignment;
        std::uint32_t high = (planned - 1U) / kPacedPrefillAlignment;
        if (high < low) { return 0; }
        while (low < high) {
            const std::uint32_t middle = low + (high - low + 1U) / 2U;
            if (fits(std::uint64_t{middle} * kPacedPrefillAlignment)) {
                low = middle;
            } else {
                high = middle - 1U;
            }
        }
        return low * kPacedPrefillAlignment;
    }

    // Folds in one finished step of `tokens` after `prefix` that took `elapsed_ns`. Every step
    // counts, a lone prompt's too, so the correction is warm when a neighbour starts generating.
    void step_done(std::uint64_t prefix, std::uint64_t tokens, std::uint64_t elapsed_ns) noexcept {
        const std::uint64_t predicted = predicted_prefill_step_ns(costs_, prefix, tokens);
        if (!enabled() || predicted == 0 || elapsed_ns == 0 ||
            predicted == std::numeric_limits<std::uint64_t>::max()) {
            return;
        }
        // A geometric moving average: one odd step (a page-in, a media item's encode) moves the
        // ratio a quarter of the way in log space.
        const double ratio =
            std::clamp(static_cast<double>(elapsed_ns) / static_cast<double>(predicted), 1.0 / 64,
                       64.0);
        correction_ = std::exp(0.75 * std::log(correction_) + 0.25 * std::log(ratio));
    }

    // Folds in one decode round that took `elapsed_ns`, which the round budget follows as a
    // geometric moving average like the correction's.
    void round_done(std::uint64_t elapsed_ns) noexcept {
        if (!round_budget_ || elapsed_ns == 0) { return; }
        const double elapsed = static_cast<double>(elapsed_ns);
        round_ns_ = round_ns_ == 0.0 ? elapsed
                                     : std::exp(0.75 * std::log(round_ns_) + 0.25 * std::log(elapsed));
    }

    [[nodiscard]] double correction() const noexcept { return correction_; }

private:
    [[nodiscard]] double budget() const noexcept { return round_budget_ ? round_ns_ : budget_ns_; }

    double budget_ns_    = 0.0;
    bool round_budget_   = false;
    double round_ns_     = 0.0;
    ContextMachineCostModel costs_;
    double correction_ = 1.0;
};

} // namespace ninfer::runtime
