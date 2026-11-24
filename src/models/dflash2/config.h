#pragma once

// The `dflash2` component a family's converter writes for a DFlash2 drafter
// (docs/maintainer/dflash.md) that runs beside a family keeping its own sequences, through the
// masked-drafter runtime (models/dspine/runtime.h): sliding-window layers, non-causal inside the
// drafted block, with a dynamic grouped convolution around each branch, over a ring of context
// K/V projected from the target's tapped hidden states; then the target's head, the 16 best
// candidates of each draft row and the conditional candidate selector. The target's token rows
// and head stay the target's.

#include "artifact/schema.h"

#include <cstdint>
#include <vector>

namespace ninfer::models::dflash2 {

struct Config {
    std::uint32_t hidden_size         = 0;
    std::uint32_t intermediate_size   = 0;
    std::uint32_t num_hidden_layers   = 0;
    std::uint32_t num_attention_heads = 0;
    std::uint32_t num_key_value_heads = 0;
    std::uint32_t head_dim            = 0;
    // The selector's codebook rows: the target's token rows.
    std::uint32_t vocab_size = 0;
    // The tap positions the target_layer_ids index (the family states what position p is).
    std::uint32_t num_target_layers = 0;
    // The trained block: the anchor and block_size - 1 masks, the most drafts a round takes.
    std::uint32_t block_size     = 0;
    std::uint32_t sliding_window = 0;
    float rms_norm_eps           = 0;
    float rope_theta             = 0;
    std::vector<std::uint32_t> target_layer_ids;
    std::uint32_t mask_token_id = 0;
    // The convolution: 2 taps over groups of conv_group_size channels; the selector: rank 256 over
    // the head's 16 best candidates of each row.
    std::uint32_t conv_kernel_size = 0;
    std::uint32_t conv_group_size  = 0;
    std::uint32_t selector_rank    = 0;
    std::uint32_t selector_top_k   = 0;
    // > 0: the selector's edges take this soft cap after the correction (the release's
    // selector_softcap_mode after_correction); its unary scores are the head's logits uncapped.
    float selector_softcap = 0;

    [[nodiscard]] std::uint32_t taps() const noexcept {
        return static_cast<std::uint32_t>(target_layer_ids.size());
    }
    [[nodiscard]] std::uint64_t query_width() const noexcept {
        return std::uint64_t(num_attention_heads) * head_dim;
    }
    [[nodiscard]] std::uint64_t key_width() const noexcept {
        return std::uint64_t(num_key_value_heads) * head_dim;
    }
    [[nodiscard]] std::uint32_t conv_groups() const noexcept {
        return conv_group_size == 0 ? 0 : hidden_size / conv_group_size;
    }
};

// The component's config. Throws artifact::ArtifactError naming the field that is missing,
// malformed or outside what the engine implements (the attention Op's heads, head width and
// windows; a two-tap convolution; a rank-256 selector over 16 candidates).
[[nodiscard]] Config parse_config(const artifact::Json& value);

// Refuses a target the drafter was not trained for: another hidden width or token rows, or more
// tap positions than the target has. `family` names the target in the message.
void check_target(const Config& config, std::uint64_t hidden_size, std::uint64_t vocab_size,
                  std::uint64_t tap_positions, const char* family);

} // namespace ninfer::models::dflash2
