#include "models/dflash/config.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::dflash {
namespace {

using artifact::ArtifactError;
using artifact::Json;

std::uint32_t integer(const Json& config, const char* name, bool positive = true) {
    if (!config.contains(name)) {
        throw ArtifactError(std::string("DFlash config: missing ") + name);
    }
    const auto n = artifact::require_u64(config.at(name), name, positive);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string("DFlash config: ") + name + " exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

float real(const Json& config, const char* name, bool zero_allowed = false) {
    if (!config.contains(name) || !config.at(name).is_number()) {
        throw ArtifactError(std::string("DFlash config: ") + name + " must be a real value");
    }
    const float value = config.at(name).get<float>();
    if (!std::isfinite(value) || value < 0 || (value == 0 && !zero_allowed)) {
        throw ArtifactError(std::string("DFlash config: ") + name + " is out of range");
    }
    return value;
}

void require(bool condition, const std::string& message) {
    if (!condition) { throw ArtifactError("DFlash config: " + message); }
}

// The windows the sliding-window attention Op registers.
constexpr std::uint32_t kWindows[] = {2048, 4096, 8192};

} // namespace

Config parse_config(const Json& value) {
    require(value.is_object(), "expected an object");
    Config out;
    out.hidden_size         = integer(value, "hidden_size");
    out.intermediate_size   = integer(value, "intermediate_size");
    out.num_hidden_layers   = integer(value, "num_hidden_layers");
    out.num_attention_heads = integer(value, "num_attention_heads");
    out.num_key_value_heads = integer(value, "num_key_value_heads");
    out.head_dim            = integer(value, "head_dim");
    out.vocab_size          = integer(value, "vocab_size");
    out.num_target_layers   = integer(value, "num_target_layers");
    out.block_size          = integer(value, "block_size");
    out.sliding_window      = integer(value, "sliding_window");
    out.rms_norm_eps        = real(value, "rms_norm_eps");
    out.rope_theta          = real(value, "rope_theta");
    out.logit_softcap       = real(value, "logit_softcap", true);
    out.markov_rank         = integer(value, "markov_rank", false);
    out.mask_token_id       = integer(value, "mask_token_id", false);
    const auto& taps = value.at("target_layer_ids");
    require(taps.is_array() && !taps.empty(), "target_layer_ids must name the taps");
    for (const auto& tap : taps) {
        const auto layer = artifact::require_u64(tap, "target_layer_ids", false);
        require(layer < out.num_target_layers, "a tap lies outside the target's tap positions");
        out.target_layer_ids.push_back(static_cast<std::uint32_t>(layer));
    }
    auto sorted = out.target_layer_ids;
    std::sort(sorted.begin(), sorted.end());
    require(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
            "target_layer_ids repeat a tap");
    require(value.contains("layer_types") && value.at("layer_types").is_array() &&
                value.at("layer_types").size() == out.num_hidden_layers,
            "layer_types must name every layer's attention");
    for (const auto& kind : value.at("layer_types")) {
        require(kind == "sliding_attention" || kind == "full_attention",
                "a layer's attention is sliding_attention or full_attention");
        out.full_attention.push_back(kind == "full_attention");
    }
    require(out.mask_token_id < out.vocab_size, "mask token outside the vocabulary");
    // The geometry the Ops implement: the attention profile (D=128 over 8 KV heads, each call 32
    // query heads), a block the verification frame takes, and the selector's rank for the Markov
    // head.
    require(out.head_dim == 128 && out.num_key_value_heads == 8 &&
                (out.num_attention_heads == 32 || out.num_attention_heads == 64),
            "the drafter's attention must be 32 or 64 query heads over 8 KV heads of 128");
    require(out.markov_rank == 0 || out.markov_rank == 256,
            "the Markov head must have rank 256 (DSpark) or be absent (DFlash)");
    require(out.hidden_size % 128 == 0 && out.intermediate_size % 128 == 0,
            "hidden and MLP widths must be multiples of 128");
    require(out.block_size >= 2 && out.block_size <= 16, "the block must be 2..16 wide");
    require(std::find(std::begin(kWindows), std::end(kWindows), out.sliding_window) !=
                std::end(kWindows),
            "the sliding window must be 2048, 4096 or 8192");
    return out;
}

void check_target(const Config& config, std::uint64_t hidden_size, std::uint64_t vocab_size,
                  std::uint64_t tap_positions, const char* family) {
    const auto fail = [&](const std::string& what) {
        throw ArtifactError(std::string("the DFlash drafter was trained for another target than "
                                        "this ") +
                            family + " model (" + what + ")");
    };
    if (config.hidden_size != hidden_size) {
        fail("hidden width " + std::to_string(config.hidden_size) + ", the model's " +
             std::to_string(hidden_size));
    }
    if (config.vocab_size != vocab_size) {
        fail("token rows " + std::to_string(config.vocab_size) + ", the model's " +
             std::to_string(vocab_size));
    }
    if (config.num_target_layers != tap_positions) {
        fail(std::to_string(config.num_target_layers) + " tap positions, the model's " +
             std::to_string(tap_positions));
    }
}

std::uint32_t full_window(const Config& config, std::uint32_t max_context, std::uint32_t block) {
    const std::uint64_t needed = std::uint64_t(max_context) + block;
    for (const std::uint32_t window : kWindows) {
        if (window >= config.sliding_window && window >= needed) { return window; }
    }
    return 8192;
}

} // namespace ninfer::models::dflash
