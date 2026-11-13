#pragma once

// Adaptive MTP verification width (--adaptive-mtp), shared by the Qwen3.5 Programs and the
// Qwen3.8-Flash-Next core, the width of a masked drafter's round in the Qwen3.5 Programs
// (MaskedWidthController), and the per-row extents of a packed verification (MtpPackedPlanner).
// Each request keeps an estimate of how far its drafts survive; a batch
// controller picks the verification width that maximizes expected committed tokens per unit of
// round cost, moving between widths only after a candidate has won several rounds and a width has
// held for a few. Round costs are measured per batch size and width as the rounds run; a width not
// yet run is priced from a relative cost shape scaled by the widths that have.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::runtime {

inline constexpr std::uint32_t kAdaptiveMtpMaximumDrafts = 15;
inline constexpr std::uint32_t kAdaptiveMtpMaximumBatch  = 8;

class MtpAdaptiveSignal final {
public:
    void reset() noexcept { *this = MtpAdaptiveSignal{}; }

    void observe(std::uint32_t proposed, std::uint32_t accepted) noexcept {
        proposed = std::min(proposed, kAdaptiveMtpMaximumDrafts);
        accepted = std::min(accepted, proposed);
        for (std::uint32_t position = 0; position < kAdaptiveMtpMaximumDrafts; ++position) {
            if (position < proposed) {
                const float sample = position < accepted ? 1.0F : 0.0F;
                fast_[position] += kFastAlpha * (sample - fast_[position]);
                slow_[position] += kSlowAlpha * (sample - slow_[position]);
                if (observations_[position] != UINT16_MAX) { ++observations_[position]; }
                if (sample == 1.0F) {
                    if (success_streak_[position] != UINT8_MAX) { ++success_streak_[position]; }
                } else {
                    success_streak_[position] = 0;
                }
                stale_rounds_[position] = 0;
            } else if (stale_rounds_[position] != UINT8_MAX) {
                ++stale_rounds_[position];
            }
        }
    }

    // Expected committed tokens of one round verifying `window` drafts, `available` of them ready.
    [[nodiscard]] float expected_tokens(std::uint32_t window,
                                        std::uint32_t available) const noexcept {
        const std::uint32_t extent = std::min({window, available, kAdaptiveMtpMaximumDrafts});
        float expected             = 1.0F;
        float previous             = 1.0F;
        for (std::uint32_t position = 0; position < extent; ++position) {
            const float evidence =
                std::min(1.0F, static_cast<float>(observations_[position]) / 4.0F);
            float estimate = kPriorSurvival[position] +
                             evidence * (0.75F * fast_[position] + 0.25F * slow_[position] -
                                         kPriorSurvival[position]);
            if ((observations_[position] == 0 || stale_rounds_[position] >= 4) && position != 0) {
                const float continuation = success_streak_[position - 1U] >= 2 ? 0.85F : 0.60F;
                estimate                 = std::max(estimate, previous * continuation);
            }
            estimate = std::clamp(estimate, 0.0F, previous);
            expected += estimate;
            previous = estimate;
        }
        return expected;
    }

    [[nodiscard]] bool confident_tail(std::uint32_t window) const noexcept {
        return window != 0 && window <= kAdaptiveMtpMaximumDrafts &&
               success_streak_[window - 1U] >= kWideningSuccessStreak;
    }

    [[nodiscard]] bool strong_wide_prefix() const noexcept {
        constexpr std::array<float, 3> thresholds{0.75F, 0.60F, 0.45F};
        for (std::size_t position = 0; position < thresholds.size(); ++position) {
            if (observations_[position] < 4 ||
                0.75F * fast_[position] + 0.25F * slow_[position] < thresholds[position]) {
                return false;
            }
        }
        return success_streak_[thresholds.size() - 1U] >= kWideningSuccessStreak;
    }

private:
    static constexpr float kFastAlpha                    = 0.375F;
    static constexpr float kSlowAlpha                    = 0.125F;
    static constexpr std::uint8_t kWideningSuccessStreak = 6;
    static constexpr std::array<float, kAdaptiveMtpMaximumDrafts> kPriorSurvival{
        0.54F,      0.16F,       0.08F,        0.04F,         0.02F,
        0.01F,      0.005F,      0.0025F,      0.00125F,      0.000625F,
        0.0003125F, 0.00015625F, 0.000078125F, 0.0000390625F, 0.00001953125F};

    std::array<float, kAdaptiveMtpMaximumDrafts> fast_ = kPriorSurvival;
    std::array<float, kAdaptiveMtpMaximumDrafts> slow_ = kPriorSurvival;
    std::array<std::uint16_t, kAdaptiveMtpMaximumDrafts> observations_{};
    std::array<std::uint8_t, kAdaptiveMtpMaximumDrafts> stale_rounds_{};
    std::array<std::uint8_t, kAdaptiveMtpMaximumDrafts> success_streak_{};
};

// Round cost by batch size and verification width, in seconds once measured. A width without
// enough rounds is priced from the relative shape below, scaled by the most-run width of its batch
// size; with nothing measured every width is priced in the shape's own units.
class MtpRoundCostModel final {
public:
    void observe(std::size_t batch_size, std::uint32_t window, double seconds) noexcept {
        if (!valid(batch_size, window) || !(seconds > 0.0)) { return; }
        float& cost           = seconds_[batch_size - 1U][window - 1U];
        std::uint16_t& counts = samples_[batch_size - 1U][window - 1U];
        const auto sample     = static_cast<float>(seconds);
        cost                  = counts == 0 ? sample : cost + kAlpha * (sample - cost);
        if (counts != UINT16_MAX) { ++counts; }
    }

    [[nodiscard]] float cost(std::size_t batch_size, std::uint32_t window) const noexcept {
        if (!valid(batch_size, window)) { return 1.0F; }
        const auto& seconds = seconds_[batch_size - 1U];
        const auto& samples = samples_[batch_size - 1U];
        if (samples[window - 1U] >= kMeasuredRounds) { return seconds[window - 1U]; }
        std::size_t reference = kAdaptiveMtpMaximumDrafts;
        for (std::size_t index = 0; index < kAdaptiveMtpMaximumDrafts; ++index) {
            if (samples[index] >= kMeasuredRounds &&
                (reference == kAdaptiveMtpMaximumDrafts || samples[index] > samples[reference])) {
                reference = index;
            }
        }
        const float shape = kRelativeCost[window - 1U];
        if (reference == kAdaptiveMtpMaximumDrafts) { return shape; }
        return shape * seconds[reference] / kRelativeCost[reference];
    }

private:
    [[nodiscard]] static bool valid(std::size_t batch_size, std::uint32_t window) noexcept {
        return batch_size != 0 && batch_size <= kAdaptiveMtpMaximumBatch && window != 0 &&
               window <= kAdaptiveMtpMaximumDrafts;
    }

    static constexpr float kAlpha                  = 0.125F;
    static constexpr std::uint16_t kMeasuredRounds = 4;
    // Verification cost of 1..15 drafts relative to one, from the upstream adaptive-MTP baseline.
    static constexpr std::array<float, kAdaptiveMtpMaximumDrafts> kRelativeCost{
        1.000F, 1.075F, 1.064F, 1.125F, 1.177F, 1.227F, 1.281F, 1.334F,
        1.388F, 1.442F, 1.496F, 1.550F, 1.604F, 1.658F, 1.712F};

    std::array<std::array<float, kAdaptiveMtpMaximumDrafts>, kAdaptiveMtpMaximumBatch> seconds_{};
    std::array<std::array<std::uint16_t, kAdaptiveMtpMaximumDrafts>, kAdaptiveMtpMaximumBatch>
        samples_{};
};

class MtpAdaptiveBatchController final {
public:
    // `floor_window` is the narrowest width the controller settles on while every row has that
    // many drafts ready: 3 where narrower widths cost about as much (the Qwen3.5 Programs, whose
    // drafts are ready before the round), 1 where each draft is a step of its own.
    void reset(std::uint32_t maximum_window, std::uint32_t floor_window = 3) noexcept {
        maximum_window_             = std::clamp(maximum_window, 1U, kAdaptiveMtpMaximumDrafts);
        floor_window_               = std::clamp(floor_window, 1U, maximum_window_);
        selected_window_            = default_startup_window();
        candidate_window_           = selected_window_;
        rounds_at_window_           = 0;
        candidate_rounds_           = 0;
        execution_samples_          = {};
        last_width_execution_round_ = {};
        execution_round_            = 0;
        selection_cohort_key_       = 0;
        last_transition_from_       = selected_window_;
        last_transition_to_         = selected_window_;
    }

    void observe_execution(std::uint32_t batch_size, std::uint32_t window,
                           double seconds) noexcept {
        if (batch_size == 0 || batch_size > kAdaptiveMtpMaximumBatch || window == 0 ||
            window > maximum_window_) {
            return;
        }
        costs_.observe(batch_size, window, seconds);
        const std::size_t batch = batch_size - 1U;
        const std::size_t width = window - 1U;
        if (execution_round_ != UINT64_MAX) { ++execution_round_; }
        if (execution_samples_[batch][width] != UINT16_MAX) { ++execution_samples_[batch][width]; }
        last_width_execution_round_[batch][width] = execution_round_;
    }

    // `available` counts each row's ready drafts, `room` its remaining tokens; a new cohort key (a
    // different set of requests) restarts the selection from a short probe.
    [[nodiscard]] std::uint32_t select(std::span<const MtpAdaptiveSignal* const> signals,
                                       std::span<const std::uint32_t> available,
                                       std::span<const std::uint32_t> room,
                                       std::uint64_t cohort_key) noexcept {
        last_transition_from_ = selected_window_;
        last_transition_to_   = selected_window_;
        if (signals.empty() || signals.size() > kAdaptiveMtpMaximumBatch ||
            signals.size() != available.size() || signals.size() != room.size()) {
            return selected_window_;
        }
        if (selection_cohort_key_ != cohort_key) {
            selected_window_            = startup_window(signals, available, room);
            candidate_window_           = selected_window_;
            rounds_at_window_           = 0;
            candidate_rounds_           = 0;
            execution_samples_          = {};
            last_width_execution_round_ = {};
            execution_round_            = 0;
            selection_cohort_key_       = cohort_key;
        }
        const std::uint32_t round_window = selected_window_;

        const std::uint32_t minimum_window = minimum_selectable_window(available);
        if (selected_window_ < minimum_window) {
            selected_window_  = minimum_window;
            candidate_window_ = minimum_window;
            rounds_at_window_ = 0;
            candidate_rounds_ = 0;
        }

        std::uint32_t best_window = selected_window_;
        float best_score          = score(selected_window_, signals, available, room);
        const std::uint32_t upward_limit =
            selected_window_ == 1 && maximum_window_ >= 3 ? 3U : selected_window_ + 1U;
        for (std::uint32_t window = minimum_window; window <= maximum_window_; ++window) {
            if (window > upward_limit) { continue; }
            const float candidate = score(window, signals, available, room);
            if (candidate > best_score) {
                best_score  = candidate;
                best_window = window;
            }
        }

        if (maximum_window_ >= 6 && best_window < 6) {
            bool all_prefixes_strong = true;
            for (const MtpAdaptiveSignal* signal : signals) {
                all_prefixes_strong = all_prefixes_strong && signal->strong_wide_prefix();
            }
            const std::uint32_t floor_window =
                widest_admissible_width(6, best_window, signals.size());
            if (all_prefixes_strong && floor_window != 0) {
                best_window = floor_window;
                best_score  = score(best_window, signals, available, room);
            }
        }

        const float current_score = score(selected_window_, signals, available, room);
        bool confident_probe      = false;
        const std::size_t batch   = signals.size() - 1U;
        if (selected_window_ < maximum_window_) {
            bool has_continuing_row  = false;
            bool all_tails_confident = true;
            for (std::size_t row = 0; row < signals.size(); ++row) {
                if (available[row] < selected_window_ || room[row] <= selected_window_ + 1U) {
                    continue;
                }
                has_continuing_row = true;
                all_tails_confident =
                    all_tails_confident && signals[row]->confident_tail(selected_window_);
            }
            const std::uint32_t preferred_probe = std::min(selected_window_ + 2U, maximum_window_);
            const std::uint32_t probe_window =
                widest_admissible_width(preferred_probe, selected_window_, signals.size());
            if (probe_window != 0) {
                const std::uint64_t observed_round =
                    last_width_execution_round_[batch][probe_window - 1U];
                const bool probe_due =
                    observed_round == 0 || execution_round_ - observed_round >= 32;
                if (has_continuing_row && all_tails_confident && probe_due) {
                    best_window     = probe_window;
                    best_score      = std::max(score(best_window, signals, available, room),
                                               current_score * 1.011F);
                    confident_probe = true;
                }
            }
        }
        const bool material_gain = best_score > current_score * 1.01F;
        if (best_window != selected_window_ && material_gain) {
            if (candidate_window_ == best_window) {
                if (candidate_rounds_ != UINT8_MAX) { ++candidate_rounds_; }
            } else {
                candidate_window_ = best_window;
                candidate_rounds_ = 1;
            }
            const bool high_width_contraction =
                best_window < selected_window_ && selected_window_ >= 6;
            const bool maximum_width_probation =
                selected_window_ == maximum_window_ &&
                execution_samples_[batch][selected_window_ - 1U] <= 2;
            const std::uint8_t minimum_residency         = confident_probe           ? 2
                                                           : maximum_width_probation ? 2
                                                           : high_width_contraction  ? 12
                                                                                     : 3;
            const std::uint8_t required_candidate_rounds = confident_probe           ? 2
                                                           : maximum_width_probation ? 2
                                                                                     : 4;
            if (candidate_rounds_ >= required_candidate_rounds &&
                rounds_at_window_ >= minimum_residency) {
                selected_window_  = best_window;
                candidate_window_ = best_window;
                rounds_at_window_ = 0;
                candidate_rounds_ = 0;
            }
        } else {
            candidate_window_ = selected_window_;
            candidate_rounds_ = 0;
        }
        if (rounds_at_window_ != UINT8_MAX) { ++rounds_at_window_; }
        last_transition_from_ = round_window;
        last_transition_to_   = selected_window_;
        return selected_window_;
    }

    [[nodiscard]] std::uint32_t selected_window() const noexcept { return selected_window_; }

    [[nodiscard]] bool transitioned() const noexcept {
        return last_transition_from_ != last_transition_to_;
    }

private:
    [[nodiscard]] std::uint32_t default_startup_window() const noexcept {
        return std::min(3U, maximum_window_);
    }

    [[nodiscard]] std::uint32_t
    minimum_selectable_window(std::span<const std::uint32_t> available) const noexcept {
        if (maximum_window_ < floor_window_) { return 1; }
        for (const std::uint32_t row_available : available) {
            if (row_available < floor_window_) { return 1; }
        }
        return floor_window_;
    }

    [[nodiscard]] std::uint32_t startup_window(std::span<const MtpAdaptiveSignal* const> signals,
                                               std::span<const std::uint32_t> available,
                                               std::span<const std::uint32_t> room) const noexcept {
        const std::uint32_t startup = default_startup_window();
        std::uint32_t best_window   = minimum_selectable_window(available);
        float best_score            = score(best_window, signals, available, room);
        for (std::uint32_t window = best_window + 1U; window <= startup; ++window) {
            const float candidate = score(window, signals, available, room);
            if (candidate > best_score) {
                best_score  = candidate;
                best_window = window;
            }
        }
        return best_window;
    }

    // Expected tokens per unit cost over this round and the next one at the same width.
    [[nodiscard]] float score(std::uint32_t window,
                              std::span<const MtpAdaptiveSignal* const> signals,
                              std::span<const std::uint32_t> available,
                              std::span<const std::uint32_t> room) const noexcept {
        float expected           = 0.0F;
        float future_expected    = 0.0F;
        float cost               = costs_.cost(signals.size(), window);
        std::size_t future_batch = 0;
        for (std::size_t row = 0; row < signals.size(); ++row) {
            expected += signals[row]->expected_tokens(window, available[row]);
            const std::uint32_t consumed    = std::min(window, available[row]) + 1U;
            const std::uint32_t future_room = room[row] > consumed ? room[row] - consumed : 0U;
            const std::uint32_t future_extent =
                std::min(window, future_room > 0 ? future_room - 1U : 0U);
            if (future_room != 0) {
                future_expected += signals[row]->expected_tokens(window, future_extent);
                ++future_batch;
            }
        }
        if (future_batch != 0) {
            expected += future_expected;
            cost += costs_.cost(future_batch, window);
        }
        return expected / std::max(cost, 1e-6F);
    }

    // A width whose cost exceeds what even full acceptance could repay against a narrower one.
    [[nodiscard]] bool physically_dominated(std::uint32_t window,
                                            std::size_t batch_size) const noexcept {
        const float candidate_cost = costs_.cost(batch_size, window);
        for (std::uint32_t narrower = 1; narrower < window; ++narrower) {
            const float narrower_cost = costs_.cost(batch_size, narrower);
            const float maximum_yield_ratio =
                static_cast<float>(window + 1U) / static_cast<float>(narrower + 1U);
            if (candidate_cost >= narrower_cost * maximum_yield_ratio) { return true; }
        }
        return false;
    }

    [[nodiscard]] std::uint32_t widest_admissible_width(std::uint32_t preferred,
                                                        std::uint32_t lower_exclusive,
                                                        std::size_t batch_size) const noexcept {
        for (std::uint32_t window = preferred; window > lower_exclusive; --window) {
            if (!physically_dominated(window, batch_size)) { return window; }
        }
        return 0;
    }

    MtpRoundCostModel costs_;
    std::uint32_t maximum_window_   = 1;
    std::uint32_t floor_window_     = 3;
    std::uint32_t selected_window_  = 1;
    std::uint32_t candidate_window_ = 1;
    std::uint8_t rounds_at_window_  = 0;
    std::uint8_t candidate_rounds_  = 0;
    std::array<std::array<std::uint16_t, kAdaptiveMtpMaximumDrafts>, kAdaptiveMtpMaximumBatch>
        execution_samples_{};
    std::array<std::array<std::uint64_t, kAdaptiveMtpMaximumDrafts>, kAdaptiveMtpMaximumBatch>
        last_width_execution_round_{};
    std::uint64_t execution_round_      = 0;
    std::uint64_t selection_cohort_key_ = 0;
    std::uint32_t last_transition_from_ = 1;
    std::uint32_t last_transition_to_   = 1;
};

// A request's acceptance under a masked drafter's two widths, a narrow n and a wide W. With A the
// drafts a wide round accepts, a narrow round commits min(A, n) + 1 tokens and a wide one adds
// (A - n) when A >= n, so E[wide] = E[narrow] + P(A >= n) * E[A - n | A >= n]. Every round
// measures the first two terms; only a wide round measures the tail, whose last value stands
// while narrow rounds run. Moving averages over about ten rounds.
class MaskedWidthSignal final {
public:
    void observe(std::uint32_t window, std::uint32_t narrow, std::uint32_t accepted) noexcept {
        const float head = static_cast<float>(std::min(accepted, narrow)) + 1.0F;
        narrow_tokens_ += kAlpha * (head - narrow_tokens_);
        full_ += kAlpha * ((accepted >= narrow ? 1.0F : 0.0F) - full_);
        if (window > narrow && accepted >= narrow) {
            tail_ += kAlpha * (static_cast<float>(accepted - narrow) - tail_);
        }
    }

    // Expected committed tokens of a round at `window` (the narrow width or wider) with `ready`
    // drafts the request can still take.
    [[nodiscard]] float expected_tokens(std::uint32_t window, std::uint32_t narrow,
                                        std::uint32_t ready) const noexcept {
        const float expected = window > narrow ? narrow_tokens_ + full_ * tail_ : narrow_tokens_;
        return std::min(expected, static_cast<float>(std::min(window, ready)) + 1.0F);
    }

    // A neutral start for widths n and W: half the narrow drafts kept, half the rounds full, and
    // half the drafts past them kept in a full round.
    void reset(std::uint32_t narrow, std::uint32_t wide) noexcept {
        narrow_tokens_ = 1.0F + 0.5F * static_cast<float>(narrow);
        full_          = 0.5F;
        tail_          = 0.5F * static_cast<float>(wide > narrow ? wide - narrow : 0U);
    }

private:
    static constexpr float kAlpha = 0.1F;
    float narrow_tokens_ = 1.0F;
    float full_          = 0.0F;
    float tail_          = 0.0F;
};

// The width of a masked drafter's round (DFlash, DFlash2 and DSpine in the Qwen3.5 Programs, under
// --adaptive-mtp): the draft window W or n = W/2, each with its own CUDA Graphs. A masked drafter
// makes all of a round's drafts in one pass at the round's width, so a narrower round also drafts
// less. The controller keeps the width with the most expected committed tokens of the round's rows
// (MaskedWidthSignal) per second of round time at that batch size; a width changes after it has led
// by kMargin for kSwitchRounds rounds in a row. Round times grow as contexts lengthen, so the width
// not running is priced from the running width's current time and the ratio of the two widths
// measured close together: a batch size runs the other width kProbeRounds rounds once the selected
// width is measured there, and again every kRefreshRounds rounds. A new request starts from the
// pooled estimate of every request's rounds.
class MaskedWidthController final {
public:
    void reset(std::uint32_t wide_window, std::uint32_t narrow_window) noexcept {
        wide_              = std::clamp(wide_window, 1U, kAdaptiveMtpMaximumDrafts);
        narrow_            = std::clamp(narrow_window, 1U, wide_);
        selected_          = wide_;
        challenger_rounds_ = 0;
        probe_rounds_      = 0;
        transitioned_      = false;
        batches_           = {};
        pooled_.reset(narrow_, wide_);
    }

    [[nodiscard]] std::uint32_t wide() const noexcept { return wide_; }
    [[nodiscard]] std::uint32_t narrow() const noexcept { return narrow_; }

    // The estimate a new request starts from.
    [[nodiscard]] const MaskedWidthSignal& pooled() const noexcept { return pooled_; }

    // A round of `window` drafts that one request ran to its full width, `accepted` of them kept.
    void observe_round(MaskedWidthSignal& signal, std::uint32_t window,
                       std::uint32_t accepted) noexcept {
        signal.observe(window, narrow_, accepted);
        pooled_.observe(window, narrow_, accepted);
    }

    // `room` counts each row's remaining tokens, the round's committed bonus token included.
    [[nodiscard]] std::uint32_t select(std::span<const MaskedWidthSignal* const> signals,
                                       std::span<const std::uint32_t> room) noexcept {
        transitioned_ = false;
        if (narrow_ == wide_ || signals.empty() || signals.size() > kAdaptiveMtpMaximumBatch ||
            signals.size() != room.size()) {
            return selected_;
        }
        const Batch& batch        = batches_[signals.size() - 1U];
        const std::uint32_t other = selected_ == wide_ ? narrow_ : wide_;
        if (probe_rounds_ > 0) {
            --probe_rounds_;
            return other;
        }
        const Width& running = batch.widths[index(selected_)];
        const Width& idle    = batch.widths[index(other)];
        if (running.samples >= kMeasuredRounds &&
            (idle.samples == 0 || batch.rounds - idle.last_round > kRefreshRounds)) {
            probe_rounds_ = (idle.samples == 0 ? kProbeRounds : kRefreshProbeRounds) - 1U;
            return other;
        }
        if (score(other, signals, room) > (1.0F + kMargin) * score(selected_, signals, room)) {
            if (++challenger_rounds_ >= kSwitchRounds) {
                selected_          = other;
                challenger_rounds_ = 0;
                transitioned_      = true;
            }
        } else {
            challenger_rounds_ = 0;
        }
        return selected_;
    }

    void observe_execution(std::uint32_t batch_size, std::uint32_t window,
                           double seconds) noexcept {
        if (batch_size == 0 || batch_size > kAdaptiveMtpMaximumBatch ||
            (window != wide_ && window != narrow_) || !(seconds > 0.0)) {
            return;
        }
        Batch& batch       = batches_[batch_size - 1U];
        Width& width       = batch.widths[index(window)];
        const Width& other = batch.widths[index(window == wide_ ? narrow_ : wide_)];
        const auto sample  = static_cast<float>(seconds);
        ++batch.rounds;
        // A width that has not run for a while starts its average again.
        const bool stale = width.samples == 0 || batch.rounds - width.last_round > kFreshRounds;
        width.seconds    = stale ? sample : width.seconds + kCostAlpha * (sample - width.seconds);
        if (stale) { width.samples = 0; }
        if (width.samples != UINT16_MAX) { ++width.samples; }
        width.last_round = batch.rounds;
        if (other.samples != 0 && batch.rounds - other.last_round <= kFreshRounds) {
            batch.wide_over_narrow =
                batch.widths[1].seconds / std::max(batch.widths[0].seconds, 1e-9F);
        }
    }

    // Whether the last select() moved the selected width.
    [[nodiscard]] bool transitioned() const noexcept { return transitioned_; }

private:
    struct Width {
        float seconds          = 0.0F;
        std::uint16_t samples  = 0;
        std::uint32_t last_round = 0;
    };
    struct Batch {
        std::array<Width, 2> widths{}; // narrow, wide
        std::uint32_t rounds   = 0;
        float wide_over_narrow = 0.0F;  // 0 until both widths ran close together
    };

    [[nodiscard]] std::size_t index(std::uint32_t window) const noexcept {
        return window == wide_ ? 1U : 0U;
    }

    // Seconds of a round at `window`: its own current time, else the other width's scaled by their
    // ratio; relative units while neither width has run at this batch size.
    [[nodiscard]] float cost(const Batch& batch, std::uint32_t window) const noexcept {
        const std::size_t own   = index(window);
        const Width& mine       = batch.widths[own];
        const Width& theirs     = batch.widths[1U - own];
        const float ratio       = batch.wide_over_narrow > 0.0F
                                      ? batch.wide_over_narrow
                                      : 1.0F + kRelativeCostPerDraft *
                                                   static_cast<float>(wide_ - narrow_);
        const bool mine_current = mine.samples != 0 &&
                                  batch.rounds - mine.last_round <= kFreshRounds;
        if (mine_current || theirs.samples == 0) {
            if (mine.samples != 0) { return mine.seconds; }
            return own == 1U ? ratio : 1.0F;
        }
        return own == 1U ? theirs.seconds * ratio : theirs.seconds / ratio;
    }

    [[nodiscard]] float score(std::uint32_t window,
                              std::span<const MaskedWidthSignal* const> signals,
                              std::span<const std::uint32_t> room) const noexcept {
        float expected = 0.0F;
        for (std::size_t row = 0; row < signals.size(); ++row) {
            const std::uint32_t ready = room[row] > 0 ? room[row] - 1U : 0U;
            expected += signals[row]->expected_tokens(window, narrow_, ready);
        }
        return expected / cost(batches_[signals.size() - 1U], window);
    }

    static constexpr std::uint32_t kSwitchRounds       = 4;
    static constexpr std::uint32_t kProbeRounds        = 4;
    static constexpr std::uint32_t kRefreshProbeRounds = 2;
    static constexpr std::uint32_t kRefreshRounds      = 96;
    static constexpr std::uint32_t kFreshRounds        = 8;
    static constexpr std::uint16_t kMeasuredRounds     = 4;
    static constexpr float kMargin                     = 0.02F;
    static constexpr float kCostAlpha                  = 0.125F;
    // Round time per extra draft relative to the narrow round, before both widths ran: 15 against 7
    // DSpine drafts took 1.21 times as long for one Qwen3.8-27B request on an RTX 5090, 1.26 for
    // eight.
    static constexpr float kRelativeCostPerDraft = 0.0275F;

    std::uint32_t wide_ = 1, narrow_ = 1, selected_ = 1;
    std::uint32_t challenger_rounds_ = 0, probe_rounds_ = 0;
    bool transitioned_               = false;
    std::array<Batch, kAdaptiveMtpMaximumBatch> batches_{};
    MaskedWidthSignal pooled_;
};

// Draft extents of a packed verification, where each row verifies only its own anchor and drafts
// and the round's cost follows the total rows verified. Given each row's ready drafts, the plan
// keeps the extents that maximize expected committed tokens per unit of that cost: for every total
// it finds the best split over the rows (a knapsack over each row's expected tokens, whose gains
// shrink with depth), then takes the total with the best ratio. Costs are measured per total as
// rounds run; a total not yet measured is priced from the nearest measured totals, or from a cost
// that grows by a twentieth of one row's per extra row while nothing is measured.
class MtpPackedPlanner final {
public:
    static constexpr std::uint32_t kMaximumRows =
        kAdaptiveMtpMaximumBatch * (kAdaptiveMtpMaximumDrafts + 1U);

    void observe(std::uint32_t rows, double seconds) noexcept {
        if (rows == 0 || rows > kMaximumRows || !(seconds > 0.0)) { return; }
        float& cost           = seconds_[rows - 1U];
        std::uint16_t& counts = samples_[rows - 1U];
        const auto sample     = static_cast<float>(seconds);
        cost                  = counts == 0 ? sample : cost + kAlpha * (sample - cost);
        if (counts != UINT16_MAX) { ++counts; }
    }

    [[nodiscard]] float cost(std::uint32_t rows) const noexcept {
        rows = std::clamp(rows, 1U, kMaximumRows);
        if (samples_[rows - 1U] >= kMeasuredRounds) { return seconds_[rows - 1U]; }
        std::uint32_t below = 0;
        std::uint32_t above = 0;
        for (std::uint32_t r = rows; r-- > 1U;) {
            if (samples_[r - 1U] >= kMeasuredRounds) {
                below = r;
                break;
            }
        }
        for (std::uint32_t r = rows + 1U; r <= kMaximumRows; ++r) {
            if (samples_[r - 1U] >= kMeasuredRounds) {
                above = r;
                break;
            }
        }
        const auto shape = [](std::uint32_t r) { return 1.0F + kRowShare * float(r - 1U); };
        if (below != 0 && above != 0) {
            const float t = float(rows - below) / float(above - below);
            return seconds_[below - 1U] + t * (seconds_[above - 1U] - seconds_[below - 1U]);
        }
        if (below != 0) { return seconds_[below - 1U] * shape(rows) / shape(below); }
        if (above != 0) { return seconds_[above - 1U] * shape(rows) / shape(above); }
        return shape(rows);
    }

    // extents[j] in [0, limits[j]]; a row without a signal keeps its limit (a lookup's proposal).
    void plan(std::span<const MtpAdaptiveSignal* const> signals,
              std::span<const std::uint32_t> limits, std::span<std::uint32_t> extents) const noexcept {
        const std::size_t b = signals.size();
        if (b == 0 || b > kAdaptiveMtpMaximumBatch || limits.size() != b || extents.size() != b) {
            return;
        }
        constexpr std::uint32_t kDrafts = kAdaptiveMtpMaximumBatch * kAdaptiveMtpMaximumDrafts;
        constexpr float kNone           = -1.0F;
        // best[j][d]: the most expected tokens of rows [0, j) verifying d drafts in all.
        std::array<std::array<float, kDrafts + 1U>, kAdaptiveMtpMaximumBatch + 1U> best{};
        std::array<std::array<std::uint8_t, kDrafts + 1U>, kAdaptiveMtpMaximumBatch> choice{};
        for (auto& row : best) { row.fill(kNone); }
        best[0][0]          = 0.0F;
        std::uint32_t total = 0;
        for (std::size_t j = 0; j < b; ++j) {
            const std::uint32_t limit = std::min(limits[j], kAdaptiveMtpMaximumDrafts);
            const std::uint32_t first = signals[j] == nullptr ? limit : 0U;
            for (std::uint32_t d = 0; d <= total; ++d) {
                if (best[j][d] == kNone) { continue; }
                for (std::uint32_t e = first; e <= limit; ++e) {
                    const float gain = signals[j] == nullptr
                                           ? 1.0F + kLookupSurvival * float(e)
                                           : signals[j]->expected_tokens(e, limit);
                    if (best[j][d] + gain > best[j + 1U][d + e]) {
                        best[j + 1U][d + e] = best[j][d] + gain;
                        choice[j][d + e]    = static_cast<std::uint8_t>(e);
                    }
                }
            }
            total += limit;
        }
        std::uint32_t chosen = 0;
        float ratio          = kNone;
        for (std::uint32_t d = 0; d <= total; ++d) {
            if (best[b][d] == kNone) { continue; }
            const float candidate =
                best[b][d] / std::max(cost(static_cast<std::uint32_t>(b) + d), 1e-9F);
            if (candidate > ratio * kGain) {
                ratio  = candidate;
                chosen = d;
            }
        }
        for (std::size_t j = b; j-- > 0;) {
            extents[j] = choice[j][chosen];
            chosen -= extents[j];
        }
    }

private:
    static constexpr float kAlpha                  = 0.125F;
    static constexpr std::uint16_t kMeasuredRounds = 4;
    static constexpr float kRowShare               = 0.05F;
    // A deeper plan replaces a shallower one only for a ratio this much better.
    static constexpr float kGain = 1.005F;
    // The share of a lookup's drafts a plan counts on.
    static constexpr float kLookupSurvival = 0.5F;

    std::array<float, kMaximumRows> seconds_{};
    std::array<std::uint16_t, kMaximumRows> samples_{};
};

} // namespace ninfer::runtime
