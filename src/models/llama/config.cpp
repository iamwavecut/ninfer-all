#include "models/llama/config.h"

#include <cmath>
#include <limits>
#include <string>

namespace ninfer::models::llama {
namespace {

using artifact::ArtifactError;
using artifact::Json;

std::uint32_t integer(const Json& config, const char* name) {
    if (!config.contains(name)) {
        throw ArtifactError(std::string("Llama config: missing ") + name);
    }
    const auto n = artifact::require_u64(config.at(name), name, true);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(name) + ": exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

float real(const Json& config, const char* name) {
    if (!config.contains(name) || !config.at(name).is_number()) {
        throw ArtifactError(std::string("Llama config: ") + name + " must be a real value");
    }
    const float value = config.at(name).get<float>();
    if (!std::isfinite(value) || value <= 0) {
        throw ArtifactError(std::string("Llama config: ") + name + " is out of range");
    }
    return value;
}

} // namespace

TextConfig parse_text_config(const Json& value) {
    const auto& architectures = value.at("architectures");
    if (!architectures.is_array() || architectures.size() != 1 ||
        architectures[0] != "LlamaForCausalLM" || value.value("model_type", "") != "llama") {
        throw ArtifactError("Llama text config: architecture/model_type mismatch");
    }
    TextConfig out;
    out.hidden_size             = integer(value, "hidden_size");
    out.vocab_size              = integer(value, "vocab_size");
    out.num_hidden_layers       = integer(value, "num_hidden_layers");
    out.max_position_embeddings = integer(value, "max_position_embeddings");
    out.rms_norm_eps            = real(value, "rms_norm_eps");
    out.rope_theta              = real(value, "rope_theta");
    out.num_attention_heads     = integer(value, "num_attention_heads");
    out.num_key_value_heads     = integer(value, "num_key_value_heads");
    out.head_dim                = integer(value, "head_dim");
    out.intermediate_size       = integer(value, "intermediate_size");
    if (value.value("qk_layout", "") != "neox" || out.head_dim != 128 ||
        out.num_attention_heads % out.num_key_value_heads ||
        out.hidden_size != out.num_attention_heads * out.head_dim || out.hidden_size % 256 ||
        out.intermediate_size % 256 || value.value("attention_bias", false) ||
        value.value("mlp_bias", false) || value.value("tie_word_embeddings", false) ||
        (value.contains("rope_scaling") && !value.at("rope_scaling").is_null())) {
        throw ArtifactError("Llama config: unsupported mathematics or geometry");
    }
    return out;
}
} // namespace ninfer::models::llama
