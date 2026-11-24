#include "models/dflash2/config.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::dflash2 {
namespace {

using artifact::ArtifactError;
using artifact::Json;

std::uint32_t integer(const Json& config, const char* name) {
    if (!config.contains(name)) {
        throw ArtifactError(std::string("DFlash2 config: missing ") + name);
    }
    const auto n = artifact::require_u64(config.at(name), name, true);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string("DFlash2 config: ") + name + " exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

float real(const Json& config, const char* name, bool zero_allowed = false) {
    if (!config.contains(name) || !config.at(name).is_number()) {
        throw ArtifactError(std::string("DFlash2 config: ") + name + " must be a real value");
    }
    const float value = config.at(name).get<float>();
    if (!std::isfinite(value) || value < 0 || (value == 0 && !zero_allowed)) {
        throw ArtifactError(std::string("DFlash2 config: ") + name + " is out of range");
    }
    return value;
}

void require(bool condition, const std::string& message) {
    if (!condition) { throw ArtifactError("DFlash2 config: " + message); }
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
    out.conv_kernel_size    = integer(value, "conv_kernel_size");
    out.conv_group_size     = integer(value, "conv_group_size");
    out.selector_rank       = integer(value, "selector_rank");
    out.selector_top_k      = integer(value, "selector_top_k");
    out.selector_softcap    = real(value, "selector_softcap", true);
    out.mask_token_id       = static_cast<std::uint32_t>(
        artifact::require_u64(value.at("mask_token_id"), "mask_token_id", false));
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
    require(out.mask_token_id < out.vocab_size, "mask token outside the vocabulary");
    // The geometry the Ops implement: the attention profile (D=128, 32 query and 8 KV heads), the
    // two-tap grouped convolution, the selector's rank and candidates, and a block the
    // verification frame takes.
    require(out.head_dim == 128 && out.num_attention_heads == 32 && out.num_key_value_heads == 8,
            "the drafter's attention must be 32 query and 8 KV heads of 128");
    require(out.conv_kernel_size == 2 && out.conv_group_size == 16,
            "the convolution must have two taps over groups of 16 channels");
    require(out.selector_rank == 256 && out.selector_top_k == 16,
            "the selector must be rank 256 over 16 candidates");
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
        throw ArtifactError(std::string("the DFlash2 drafter was trained for another target than "
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

} // namespace ninfer::models::dflash2
