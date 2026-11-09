#pragma once

#include "artifact/schema.h"
#include "models/qwen3_5/config.h"

#include <cstdint>

namespace ninfer::models::qwen3_5 {

// The target dimensions a draft config is checked against: its hidden width (which the draft
// shares), its embedding rows (the mask token's bound) and its decoder blocks (the taps' bound).
struct DraftTarget {
    std::uint32_t hidden_size       = 0;
    std::uint32_t vocab_size        = 0;
    std::uint32_t num_hidden_layers = 0;
};

// A DFlash or DFlash2 component's config (the draft's own fields; hidden width and vocabulary are
// the target's), which another family that drafts with DFlash2 parses too.
[[nodiscard]] DraftConfig parse_draft_config(const artifact::Json& value, const DraftTarget& target,
                                             bool dflash2);

} // namespace ninfer::models::qwen3_5
