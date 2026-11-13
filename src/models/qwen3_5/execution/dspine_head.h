#pragma once

// The head a DSpine drafter's proposals rank in one pass (ops::linear_topk), shared by the
// startup's workspace plan and the Program's drafter target.

#include "core/qtype.h"
#include "core/weight.h"
#include "models/load_options.h"
#include "models/qwen3_5/execution/parameters.h"

namespace ninfer::models::qwen3_5::execution {

// The proposal shortlist when the plan drafts with it, else the full output head, where
// ops::linear_topk takes that head's format and geometry whole on the drafter's device; nullptr
// leaves the drafter's dense logits and ops::candidate_topk. The shortlist ranks only its own
// rows, so a DSpine round drafted with it never proposes a token outside them.
[[nodiscard]] inline const LinearParameters* dspine_ranked_head(const Parameters& p,
                                                                ProposalHead head) noexcept {
    if (head == ProposalHead::Optimized) {
        if (!p.proposal || !p.proposal->token_ids || p.proposal->head_shares.divided()) {
            return nullptr;
        }
        const Weight& w = p.proposal->head.weight;
        const bool form = (w.qtype == QType::Q4_G64_FP16 || w.qtype == QType::T2_G128_FP16) &&
                          w.n == 131072 && w.k == 5120;
        return form ? &p.proposal->head : nullptr;
    }
    const Weight& w = p.text.output_head.weight;
    const bool form = (w.qtype == QType::Q8_G32_FP16 || w.qtype == QType::FP8_E4M3FN_ROW_BF16 ||
                       w.qtype == QType::T2_G128_FP16) &&
                      w.n == 248320 && w.k == 5120 &&
                      p.model.resources().public_token_count == 248077;
    return form && !p.text.output_head_shares.divided() ? &p.text.output_head : nullptr;
}

} // namespace ninfer::models::qwen3_5::execution
