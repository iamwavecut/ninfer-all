#include "runtime/engine/prefill_pacing.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

using ninfer::runtime::ContextMachineCostModel;
using ninfer::runtime::PrefillPacing;
using ninfer::runtime::predicted_prefill_step_ns;
using ninfer::runtime::prefill_step_quanta;

constexpr std::uint64_t kQ32 = std::uint64_t{1} << 32;

// 1 ms per step plus 100 us per token, and 2 ns per attention pair: a 2048-token step from token
// zero is predicted at 1 + 204.8 + 4.2 ms.
ContextMachineCostModel costs() {
    ContextMachineCostModel model;
    model.prefill.chunk_ns              = 1'000'000;
    model.prefill.token_ns_q32          = 100'000 * kQ32;
    model.prefill.attention_pair_ns_q32 = 2 * kQ32;
    return model;
}

} // namespace

int main() {
    int failures = 0;
    const ContextMachineCostModel model = costs();

    failures += check(predicted_prefill_step_ns(model, 0, 0) == 0, "an empty step costs nothing");
    failures += check(predicted_prefill_step_ns(model, 0, 2048) ==
                          1'000'000 + 2048 * 100'000 + 2 * (2048 * 2049 / 2),
                      "a step is one chunk of its tokens and their attention pairs");

    // Off: every step runs as planned.
    failures += check(PrefillPacing().step_cap(0, 4096) == 0 && !PrefillPacing().enabled(),
                      "no budget must leave the step planned");

    // 60 ms fits 1 + 0.1 t + ~0 ms up to 590 tokens: the largest multiple of 128 is 512.
    PrefillPacing pacing(60, false, model);
    failures += check(pacing.step_cap(0, 2048) == 512, "a step is cut to the budget in 128s");
    // A planned step that fits, and one no larger than the minimum, stay as planned.
    failures += check(pacing.step_cap(0, 384) == 0, "a short step must not be cut");
    failures += check(PrefillPacing(1000, false, model).step_cap(0, 2048) == 0,
                      "a step inside the budget must not be cut");
    // Nothing goes below 512 tokens, even when 512 itself overruns the budget.
    failures += check(PrefillPacing(1, false, model).step_cap(0, 2048) == 512,
                      "a step is never cut below the minimum");

    // 120 ms: 1152 tokens from token zero; after 100K tokens each token also attends to the
    // prefix (0.2 ms more per token), so the step shrinks to 384 by time, and to the minimum.
    PrefillPacing deep(120, false, model);
    const std::uint32_t shallow = deep.step_cap(0, 4096);
    const std::uint32_t far     = deep.step_cap(100'000, 4096);
    failures += check(shallow == 1152, "the cap is the largest affordable multiple of 128");
    failures += check(far == 512 && far < shallow, "a deeper prefix must be cut harder");

    // Steps observed at twice the prediction halve what fits: the correction converges on 2.
    PrefillPacing observed(120, false, model);
    for (int step = 0; step < 32; ++step) {
        observed.step_done(0, 1024, 2 * predicted_prefill_step_ns(model, 0, 1024));
    }
    failures += check(observed.correction() > 1.99 && observed.correction() < 2.01,
                      "the correction must follow the observed step times");
    failures += check(observed.step_cap(0, 4096) == 512,
                      "a slower card must cut the step to what fits at its speed");
    // One outlier moves the correction a quarter of the way in log space, bounded at 64x.
    PrefillPacing outlier(120, false, model);
    outlier.step_done(0, 1024, 1'000 * predicted_prefill_step_ns(model, 0, 1024));
    failures += check(outlier.correction() > 2.82 && outlier.correction() < 2.83,
                      "one outlier step must move the correction by 64^(1/4)");
    // An empty or unmeasured step changes nothing.
    outlier.step_done(0, 0, 5);
    outlier.step_done(0, 1024, 0);
    failures += check(outlier.correction() > 2.82 && outlier.correction() < 2.83,
                      "an empty or unmeasured step must not move the correction");

    // The round budget: off until a decode round has been measured, then the rounds' moving
    // average, with steps down to 256 tokens. 50 ms fits 1 + 0.1 t ms up to 490 tokens: 384.
    PrefillPacing rounds(0, true, model);
    failures += check(rounds.round_budget() && !rounds.enabled() &&
                          rounds.step_cap(0, 2048) == 0,
                      "the round budget waits for a measured round");
    rounds.round_done(50'000'000);
    failures += check(rounds.enabled() && rounds.step_cap(0, 2048) == 384,
                      "a step is cut to one decode round in 128s");
    failures += check(rounds.step_cap(0, 256) == 0, "a step of 256 tokens is never cut");
    PrefillPacing lone(0, true, model);
    lone.round_done(1'000'000);
    failures += check(lone.step_cap(0, 2048) == 256,
                      "a round budget never cuts a step below 256 tokens");
    // Rounds of 50 then 100 ms: the average moves a quarter of the way in log space, 59.5 ms,
    // which fits up to 585 tokens: 512.
    rounds.round_done(100'000'000);
    failures += check(rounds.step_cap(0, 2048) == 512, "the round budget follows the rounds");
    // An explicit budget wins over the round budget, and rounds do not move it.
    PrefillPacing fixed(60, true, model);
    fixed.round_done(1'000'000);
    failures += check(!fixed.round_budget() && fixed.step_cap(0, 2048) == 512,
                      "an explicit budget must not follow the rounds");

    // A cut step charges whole chunks only: three 896-token steps of a 2048-token chunk charge
    // 0, 0 and 1, carrying 640 tokens.
    std::uint32_t carried = 0;
    failures += check(prefill_step_quanta(true, 896, 2048, carried) == 0 && carried == 896 &&
                          prefill_step_quanta(true, 896, 2048, carried) == 0 &&
                          prefill_step_quanta(true, 896, 2048, carried) == 1 && carried == 640,
                      "cut steps must charge one quantum per whole chunk");
    failures += check(prefill_step_quanta(false, 100, 2048, carried) == 1 && carried == 0,
                      "a planned step charges one quantum and drops the carry");

    // Two 1025-token segments, each a 1024-token cut step and a planned one-token step to its
    // capture frontier: the projection is one quantum per segment. A carry kept across the first
    // segment's planned step would make the second segment's cut step charge a third.
    carried = 0;
    const std::uint64_t segments = prefill_step_quanta(true, 1024, 2048, carried) +
                                   prefill_step_quanta(false, 1, 2048, carried) +
                                   prefill_step_quanta(true, 1024, 2048, carried) +
                                   prefill_step_quanta(false, 1, 2048, carried);
    failures += check(segments == 2, "a planned step must end its segment's carry");

    // Against the projection, 1 + (segment - 1) / chunk quanta for each segment a capture frontier
    // ends: any mix of planned steps (the chunk, or the rest of the prompt or of the segment) and
    // steps cut to a multiple of 128 from 512 up to below the planned step, which therefore never
    // ends the prompt, charges at least one and never more. Whether a step is cut changes from
    // step to step, as other lanes start and stop generating; a cut step that reaches a frontier
    // counts as cut, as the Engine counts it.
    std::mt19937 random(170);
    bool within = true;
    for (int trial = 0; trial < 50'000 && within; ++trial) {
        const std::uint32_t chunk  = 512U * (2U + random() % 7U) + 128U * (random() % 4U);
        const std::uint32_t prompt = 1U + random() % (8U * chunk);
        std::vector<std::uint32_t> frontiers;
        for (std::uint32_t count = random() % 4U; count != 0; --count) {
            if (prompt > 1U) { frontiers.push_back(1U + random() % (prompt - 1U)); }
        }
        std::sort(frontiers.begin(), frontiers.end());
        frontiers.erase(std::unique(frontiers.begin(), frontiers.end()), frontiers.end());
        frontiers.push_back(prompt);
        std::uint64_t projected = 0;
        std::uint32_t begin     = 0;
        for (const std::uint32_t frontier : frontiers) {
            projected += 1ULL + (frontier - begin - 1ULL) / chunk;
            begin = frontier;
        }
        std::uint32_t position = 0, held = 0;
        std::uint64_t charged  = 0;
        std::size_t segment    = 0;
        while (position != prompt) {
            const std::uint32_t planned = std::min(prompt - position, chunk);
            const std::uint32_t cuts    = planned > 512U ? (planned - 513U) / 128U + 1U : 0U;
            const std::uint32_t cap =
                cuts != 0 && random() % 2U == 0U ? 512U + 128U * (random() % cuts) : 0U;
            const std::uint32_t tokens =
                std::min(cap != 0 ? cap : planned, frontiers[segment] - position);
            charged += prefill_step_quanta(cap != 0 && tokens < planned, tokens, chunk, held);
            position += tokens;
            if (position == frontiers[segment]) { ++segment; }
        }
        within = charged >= 1 && charged <= projected;
    }
    failures += check(within, "a prompt's steps must not charge more than its planned steps");

    if (failures == 0) { std::cout << "prefill pacing tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
