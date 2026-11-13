// Adaptive MTP only chooses how many drafts a round verifies; greedy output is the same at every
// width. What these checks defend is the choice: the survival estimate, the measured cost model,
// that the controller widens under full acceptance and settles low under none, and that a packed
// round's plan spends its rows where drafts survive, and that a masked drafter's round keeps the
// width whose drafts repay its measured cost.

#include "runtime/contract/mtp_adaptive.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <span>
#include <stdexcept>

namespace {

using ninfer::runtime::MaskedWidthController;
using ninfer::runtime::MaskedWidthSignal;
using ninfer::runtime::MtpAdaptiveBatchController;
using ninfer::runtime::MtpAdaptiveSignal;
using ninfer::runtime::MtpPackedPlanner;
using ninfer::runtime::MtpRoundCostModel;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void survival_follows_acceptance() {
    MtpAdaptiveSignal fresh;
    const float prior = fresh.expected_tokens(3, 3);
    require(prior > 1.5F && prior < 2.5F, "a fresh signal expects the prior survival");
    require(fresh.expected_tokens(3, 0) == 1.0F, "no ready draft expects only the bonus token");

    MtpAdaptiveSignal accepting;
    for (int round = 0; round < 40; ++round) { accepting.observe(5, 5); }
    require(accepting.expected_tokens(5, 5) > 5.5F, "full acceptance expects every draft");
    require(accepting.confident_tail(5), "a long success streak is a confident tail");

    MtpAdaptiveSignal rejecting;
    for (int round = 0; round < 40; ++round) { rejecting.observe(5, 0); }
    require(rejecting.expected_tokens(5, 5) < 1.1F, "no acceptance expects only the bonus token");
}

void costs_are_measured_then_scaled() {
    MtpRoundCostModel costs;
    const float shape_one   = costs.cost(1, 1);
    const float shape_three = costs.cost(1, 3);
    require(shape_three > shape_one, "the relative shape grows with the width");
    for (int round = 0; round < 8; ++round) { costs.observe(1, 3, 0.030); }
    require(std::abs(costs.cost(1, 3) - 0.030F) < 1e-6F, "a measured width reports its seconds");
    require(std::abs(costs.cost(1, 1) - 0.030F * shape_one / shape_three) < 1e-6F,
            "an unmeasured width scales from the measured one");
    require(costs.cost(2, 3) == shape_three, "batch sizes are measured separately");
}

struct Simulation {
    std::uint32_t final_window = 0;
    std::uint32_t widest       = 0;
};

// Rounds of one row with `maximum` drafts always ready; `accept` gives the accepted drafts of a
// round at a width, `seconds` its cost.
template <typename Accept, typename Seconds>
Simulation simulate(std::uint32_t maximum, Accept accept, Seconds seconds,
                    std::uint32_t floor = 3) {
    MtpAdaptiveBatchController controller;
    controller.reset(maximum, floor);
    MtpAdaptiveSignal signal;
    Simulation out;
    const std::array<const MtpAdaptiveSignal*, 1> signals{&signal};
    const std::array<std::uint32_t, 1> available{maximum};
    const std::array<std::uint32_t, 1> room{4096};
    for (int round = 0; round < 300; ++round) {
        const std::uint32_t window = controller.select(signals, available, room, 7);
        require(window >= 1 && window <= maximum, "the selected width is outside the range");
        controller.observe_execution(1, window, seconds(window));
        signal.observe(window, accept(window));
        out.widest       = std::max(out.widest, window);
        out.final_window = window;
    }
    return out;
}

void the_controller_widens_and_narrows() {
    const auto seconds         = [](std::uint32_t window) { return 0.020 + 0.001 * window; };
    const Simulation accepting = simulate(7, [](std::uint32_t window) { return window; }, seconds);
    require(accepting.final_window >= 5, "full acceptance did not widen the window");
    const Simulation rejecting = simulate(7, [](std::uint32_t) { return 0U; }, seconds);
    require(rejecting.final_window == 3, "no acceptance did not settle at the narrowest width");
    const Simulation fixed_short =
        simulate(2, [](std::uint32_t window) { return window; }, seconds);
    require(fixed_short.widest <= 2, "the window exceeded its maximum");
    // With a floor of one (each draft a step of its own, as in Qwen3.8-Flash-Next) rejected drafts
    // settle at a single draft, and drafts that keep surviving still widen the window.
    const Simulation rejecting_floor_one =
        simulate(7, [](std::uint32_t) { return 0U; }, seconds, 1);
    require(rejecting_floor_one.final_window == 1, "a floor of one did not settle at one draft");
    const Simulation accepting_floor_one =
        simulate(7, [](std::uint32_t window) { return window; }, seconds, 1);
    require(accepting_floor_one.final_window >= 5, "a floor of one did not widen under acceptance");
    // Drafts that survive about two positions settle between the extremes when every step costs.
    const auto stepped = [](std::uint32_t window) { return 0.012 + 0.004 * window; };
    const Simulation partial = simulate(
        7, [](std::uint32_t window) { return window < 2 ? window : 2U; }, stepped, 1);
    require(partial.final_window >= 2 && partial.final_window <= 3,
            "two surviving drafts did not settle at two or three");
}

// Rounds of `rows` rows under the masked drafters' controller (widths 15 and 7). `draw` gives the
// drafts a row's wide round would accept (a narrow round keeps at most seven of them), `seconds`
// the round's cost at a width and round index. Returns the width of the last round;
// `narrow_rounds` counts the rounds at seven.
template <typename Draw, typename Seconds>
std::uint32_t simulate_masked(std::size_t rows, Draw draw, Seconds seconds,
                              std::uint32_t& narrow_rounds) {
    MaskedWidthController controller;
    controller.reset(15, 7);
    std::array<MaskedWidthSignal, 8> signals;
    std::array<const MaskedWidthSignal*, 8> views{};
    std::array<std::uint32_t, 8> room{};
    for (std::size_t row = 0; row < rows; ++row) {
        signals[row] = controller.pooled();
        views[row]   = &signals[row];
        room[row]    = 4096;
    }
    std::uint32_t state  = 12345;
    std::uint32_t window = 0;
    narrow_rounds        = 0;
    for (int round = 0; round < 400; ++round) {
        window = controller.select(std::span<const MaskedWidthSignal* const>(views.data(), rows),
                                   std::span<const std::uint32_t>(room.data(), rows));
        require(window == 7 || window == 15, "the masked width is neither width");
        controller.observe_execution(static_cast<std::uint32_t>(rows), window,
                                     seconds(window, round));
        for (std::size_t row = 0; row < rows; ++row) {
            state = state * 1664525U + 1013904223U;
            const std::uint32_t accepted = std::min(draw(state >> 8U), window);
            controller.observe_round(signals[row], window, accepted);
        }
        if (window == 7) { ++narrow_rounds; }
    }
    return window;
}

void the_masked_width_follows_survival_and_cost() {
    // One Qwen3.8-27B request with DSpine on an RTX 5090: 17.2 ms a round of 15 drafts, 14.2 ms
    // of 7. Code: half the rounds keep all 15 drafts, the rest stop anywhere (12.0 tokens a wide
    // round against 7.1 a narrow one).
    const auto single = [](std::uint32_t window, int) { return window == 15 ? 0.0172 : 0.0142; };
    const auto code   = [](std::uint32_t random) {
        return random % 2U == 0 ? 15U : (random >> 1U) % 15U;
    };
    std::uint32_t narrow = 0;
    require(simulate_masked(1, code, single, narrow) == 15 && narrow <= 40,
            "long survival did not keep the wide round");
    require(narrow >= 4, "the narrow width was never measured");
    // Prose: at most three drafts survive, so the narrow round commits as much for less.
    const auto prose = [](std::uint32_t random) { return random % 4U; };
    require(simulate_masked(1, prose, single, narrow) == 7 && narrow >= 360,
            "short survival did not settle on the narrow round");
    // Eight requests: 59.4 ms against 47.0 ms. Each draft survives with probability of about 0.8
    // (4.78 tokens a wide round against 4.17 a narrow one), which does not repay the wide round.
    const auto batched = [](std::uint32_t window, int) { return window == 15 ? 0.0594 : 0.0470; };
    const auto geometric = [](std::uint32_t random) {
        std::uint32_t kept = 0;
        while (kept < 15 && (random % 5U) != 0) {
            random /= 5U;
            random += 7919U * (kept + 1U);
            ++kept;
        }
        return kept;
    };
    require(simulate_masked(8, geometric, batched, narrow) == 7 && narrow >= 340,
            "eight rows of short survival kept the wide round");
    // Rounds that grow 40% slower as contexts lengthen: the wide width, last run in its probe, is
    // priced from the narrow round's current time, not from its own early time.
    const auto growing = [](std::uint32_t window, int round) {
        return (window == 15 ? 0.0594 : 0.0470) * (1.0 + 0.001 * round);
    };
    require(simulate_masked(8, geometric, growing, narrow) == 7 && narrow >= 365,
            "growing round times drew the batch to the wide round");
    // Every draft surviving still pays for the wide round in a batch.
    const auto all = [](std::uint32_t) { return 15U; };
    require(simulate_masked(8, all, batched, narrow) == 15 && narrow <= 16,
            "eight rows of full survival left the wide round");
}

void the_packed_plan_spends_rows_on_survival() {
    MtpAdaptiveSignal accepting;
    MtpAdaptiveSignal rejecting;
    for (int round = 0; round < 40; ++round) {
        accepting.observe(6, 6);
        rejecting.observe(6, 0);
    }
    const std::array<std::uint32_t, 3> limits{6, 6, 4};
    std::array<std::uint32_t, 3> extents{};

    // Nearly free rows: the accepting row verifies all its drafts, the rejecting one none, and a
    // lookup's proposal keeps its length.
    MtpPackedPlanner cheap;
    const std::array<const MtpAdaptiveSignal*, 3> signals{&accepting, &rejecting, nullptr};
    cheap.plan(signals, limits, extents);
    require(extents[0] == 6, "an accepting row did not verify its drafts");
    require(extents[1] == 0, "a rejecting row verified drafts that never survive");
    require(extents[2] == 4, "a lookup's proposal lost drafts");

    // Rows measured as dear as a whole round each: no draft repays its row.
    MtpPackedPlanner dear;
    for (std::uint32_t rows = 1; rows <= MtpPackedPlanner::kMaximumRows; ++rows) {
        for (int sample = 0; sample < 4; ++sample) { dear.observe(rows, 0.01 * rows); }
    }
    const std::array<const MtpAdaptiveSignal*, 3> models{&accepting, &rejecting, &rejecting};
    dear.plan(models, limits, extents);
    require(extents[0] == 0 && extents[1] == 0 && extents[2] == 0,
            "rows that cost a round each still carried drafts");

    // An unmeasured total is priced between its measured neighbours.
    MtpPackedPlanner partial;
    for (int sample = 0; sample < 4; ++sample) {
        partial.observe(2, 0.010);
        partial.observe(6, 0.014);
    }
    require(std::abs(partial.cost(4) - 0.012F) < 1e-6F, "an unmeasured total was not interpolated");
}

} // namespace

int main() {
    try {
        survival_follows_acceptance();
        costs_are_measured_then_scaled();
        the_controller_widens_and_narrows();
        the_masked_width_follows_survival_and_cost();
        the_packed_plan_spends_rows_on_survival();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::puts("adaptive MTP: survival, costs, width selection, masked widths and packed plans ok");
    return 0;
}
