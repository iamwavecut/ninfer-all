#pragma once

// The text config of Qwen3.8-Flash-Next (Qwen4ExpForCausalLM) as the converter writes it into the
// artifact (tools/convert/qwen4_exp.py), parsed strictly: every field the mathematics reads is
// required, and the shapes this engine implements are the only ones accepted.

#include "artifact/schema.h"
#include "models/qwen3_5/config.h"
#include "models/qwen4_exp/ngram_hash.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer::models::qwen4_exp {

enum class MixerKind : std::uint8_t { GatedDeltaNet, SparseAttention };

// Edge0's latent routed experts (ops::moe_latent_gguf): each layer's `groups` encoders
// [width, hidden] and decoders [hidden, width], every expert in one group, its gate and up rows
// `width` wide and its down rows `width` long; the per-expert output multipliers of an NVFP4
// release and the encoder's column scale of a Q3_K release when the artifact stores them.
struct LatentMoe {
    std::uint32_t width  = 0;
    std::uint32_t groups = 0;
    bool expert_scales   = false;
    bool column_scale    = false;
};

struct TextConfig {
    std::uint32_t hidden_size             = 0;
    std::uint32_t vocab_size              = 0;
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t max_position_embeddings = 0;
    bool tie_word_embeddings              = false;
    float rms_norm_eps                    = 0;
    std::int32_t eos_token_id             = 0;
    std::vector<MixerKind> layer_types;

    // Qwen Sparse Attention: dense attention over the indexer's selected blocks.
    std::uint32_t num_attention_heads    = 0;
    std::uint32_t num_key_value_heads    = 0;
    std::uint32_t head_dim               = 0;
    float rope_theta                     = 0;
    float partial_rotary_factor          = 0;
    std::array<std::uint32_t, 3> mrope_section{};
    std::uint32_t indexer_n_heads        = 0;
    std::uint32_t indexer_head_dim       = 0;
    std::uint32_t indexer_compress_ratio = 0;
    std::uint32_t indexer_budget         = 0;

    // Gated DeltaNet, with a sigmoid output gate.
    std::uint32_t linear_num_key_heads   = 0;
    std::uint32_t linear_key_head_dim    = 0;
    std::uint32_t linear_num_value_heads = 0;
    std::uint32_t linear_value_head_dim  = 0;
    std::uint32_t linear_conv_kernel_dim = 0;

    // MoE in every block.
    std::uint32_t num_experts                     = 0;
    std::uint32_t num_experts_per_tok             = 0;
    std::uint32_t moe_intermediate_size           = 0;
    std::uint32_t shared_expert_intermediate_size = 0;
    // Absent for Qwen3.8-Flash-Next's own experts.
    std::optional<LatentMoe> latent_moe;

    // Hyper-connection residual.
    std::uint32_t hc_count   = 0;
    std::uint32_t hc_lowrank = 0;

    // PLE: the hashed n-gram embedding, added before each block in `ple_layers`.
    std::vector<std::uint32_t> ple_layers;
    std::uint32_t ple_embed_dim        = 0;
    std::uint32_t ple_conv_kernel_size = 0;
    NgramHashSpec ngram;

    [[nodiscard]] std::uint32_t ngram_heads() const noexcept {
        return (ngram.ngram_size - 1) * ngram.heads_per_ngram;
    }
    [[nodiscard]] std::uint32_t indexer_block_budget() const noexcept {
        return indexer_budget / indexer_compress_ratio;
    }
};

// The MTP block (the artifact's `mtp` component): one sparse-attention layer with the text model's
// geometry and its own MoE, which may keep more experts than an expert-pruned text model.
struct MtpConfig {
    float rope_theta          = 0;
    std::uint32_t num_experts = 0;
};

// The DFlash2 drafter (the artifact's `dflash2` component, docs/maintainer/dflash.md): Qwen3.5's
// draft config, conditioned on this model's completed blocks (the mean of their four streams).
using DFlash2Config = qwen3_5::DraftConfig;

// Throws artifact::ArtifactError naming the field that is missing, malformed or outside the
// implemented geometry.
[[nodiscard]] TextConfig parse_text_config(const artifact::Json& value);
[[nodiscard]] MtpConfig parse_mtp_config(const artifact::Json& value, const TextConfig& text);
[[nodiscard]] DFlash2Config parse_dflash2_config(const artifact::Json& value,
                                                 const TextConfig& text);

} // namespace ninfer::models::qwen4_exp
