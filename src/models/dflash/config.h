#pragma once

// The `dflash` and `dspark` components a family's converter writes for a masked block drafter of
// z-lab's DFlash layout (docs/maintainer/dflash.md#gemma-4) that runs beside a family keeping its
// own sequences, through the masked-drafter runtime (models/dspine/runtime.h): sliding-window and
// full-attention layers, non-causal inside the drafted block, over rings of context K/V projected
// from the target's tapped hidden states; then the target's head under its final soft cap, the 16
// best candidates of each draft row, and either each row's best candidate (DFlash) or the chain
// that a vanilla Markov head of rank 256 adds its transition bias to (DSpark). The target's token
// rows and head stay the target's.

#include "artifact/schema.h"

#include <cstdint>
#include <vector>

namespace ninfer::models::dflash {

struct Config {
    std::uint32_t hidden_size         = 0;
    std::uint32_t intermediate_size   = 0;
    std::uint32_t num_hidden_layers   = 0;
    std::uint32_t num_attention_heads = 0;
    std::uint32_t num_key_value_heads = 0;
    std::uint32_t head_dim            = 0;
    // The target's token rows (the Markov tables' rows).
    std::uint32_t vocab_size = 0;
    // The tap positions the target_layer_ids index (the family states what position p is).
    std::uint32_t num_target_layers = 0;
    // The trained block: the anchor and block_size - 1 masks.
    std::uint32_t block_size = 0;
    // The window of the sliding-window layers; a full-attention layer sees its whole ring.
    std::uint32_t sliding_window = 0;
    float rms_norm_eps           = 0;
    float rope_theta             = 0;
    std::vector<std::uint32_t> target_layer_ids;
    // Per layer: full attention (true) or the sliding window.
    std::vector<bool> full_attention;
    std::uint32_t mask_token_id = 0;
    // > 0: the head's logits take this soft cap (the target's final_logit_softcapping) before the
    // Markov bias is added and the candidates are compared.
    float logit_softcap = 0;
    // 0: DFlash, each draft row's best candidate. 256: DSpark's vanilla Markov head,
    // bias(prev, c) = markov_output[c] . markov_embedding[prev], chained left to right.
    std::uint32_t markov_rank = 0;

    [[nodiscard]] std::uint32_t taps() const noexcept {
        return static_cast<std::uint32_t>(target_layer_ids.size());
    }
    [[nodiscard]] std::uint64_t query_width() const noexcept {
        return std::uint64_t(num_attention_heads) * head_dim;
    }
    [[nodiscard]] std::uint64_t key_width() const noexcept {
        return std::uint64_t(num_key_value_heads) * head_dim;
    }
    // The attention Op's calls a layer makes: each takes 32 query heads over the 8 KV heads.
    [[nodiscard]] std::uint32_t query_groups() const noexcept { return num_attention_heads / 32; }
    [[nodiscard]] bool has_full_attention() const noexcept {
        for (const bool full : full_attention) {
            if (full) { return true; }
        }
        return false;
    }
};

// The component's config. Throws artifact::ArtifactError naming the field that is missing,
// malformed or outside what the engine implements (the attention Op's head width, KV heads and
// windows; 32 or 64 query heads; a Markov head of rank 256 or none).
[[nodiscard]] Config parse_config(const artifact::Json& value);

// Refuses a target the drafter was not trained for: another hidden width or token rows, or other
// tap positions than the target has. `family` names the target in the message.
void check_target(const Config& config, std::uint64_t hidden_size, std::uint64_t vocab_size,
                  std::uint64_t tap_positions, const char* family);

// The smallest of the attention Op's capacities (2048, 4096, 8192) that holds max_context
// positions and a block of `block` columns, at most 8192, and at least the sliding window: the
// full-attention layers' ring. Past 8192 positions a full-attention layer sees the last 8192.
[[nodiscard]] std::uint32_t full_window(const Config& config, std::uint32_t max_context,
                                        std::uint32_t block);

} // namespace ninfer::models::dflash
