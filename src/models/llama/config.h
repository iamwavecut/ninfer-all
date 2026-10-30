#pragma once
#include "artifact/schema.h"
#include <cstdint>

namespace ninfer::models::llama {
// Bias-free dense Llama, unscaled full-head NeoX RoPE, no QK norm, gate or sink.
struct TextConfig {
    std::uint32_t hidden_size = 0, vocab_size = 0, num_hidden_layers = 0,
                  max_position_embeddings = 0;
    std::uint32_t num_attention_heads = 0, num_key_value_heads = 0, head_dim = 0,
                  intermediate_size = 0;
    float rms_norm_eps = 0, rope_theta = 0;
};

TextConfig parse_text_config(const artifact::Json&);
} // namespace ninfer::models::llama
