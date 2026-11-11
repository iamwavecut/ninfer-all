#pragma once

#include "ninfer/ops/sampling.h"
#include "ninfer/request.h"

namespace ninfer::runtime {

// A drafting row's sampler. A row whose target draws by Gumbel (--gumbel-sampling) and samples
// keeps its request's filters, seed and draw, without the penalties, masks and counts that belong
// to the target's draws, and names a valid id (0) where the selected logit is not finite, since the
// target verifies every proposal anyway. A drafter that draws with it at the key of the
// verification column it proposes for, (seed, length + i + 1, kSamplePurposeDecode), adds the same
// Gumbel noise to each token as the target's own draw there, so the two draws agree wherever the
// two distributions do; the verification keeps exactly the target's draws either way. Every other
// row (temperature 0, or an inverse-CDF target, whose one uniform per position couples the two
// draws only through their cumulative orders) drafts the argmax: temperature 0.
[[nodiscard]] inline ops::SamplingConfig draft_sampling(const ops::SamplingConfig& target) noexcept {
    ops::SamplingConfig config;
    config.non_finite_token = 0;
    if (target.draw != ops::kSampleDrawGumbel || target.temperature <= 0.0F) { return config; }
    config.temperature = target.temperature;
    config.top_k       = target.top_k;
    config.top_p       = target.top_p;
    config.min_p       = target.min_p;
    config.seed        = target.seed;
    config.draw        = target.draw;
    return config;
}

[[nodiscard]] inline ops::SamplingConfig draft_sampling(
    const ResolvedSamplingParameters& sampling) noexcept {
    ops::SamplingConfig target;
    target.temperature = sampling.temperature;
    target.top_k       = sampling.top_k;
    target.top_p       = sampling.top_p;
    target.min_p       = sampling.min_p;
    target.seed        = sampling.seed;
    target.draw        = sampling.gumbel ? ops::kSampleDrawGumbel : ops::kSampleDrawInverseCdf;
    return draft_sampling(target);
}

} // namespace ninfer::runtime
