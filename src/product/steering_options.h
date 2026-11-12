#pragma once

#include "ninfer/engine_options.h"
#include "product/option_layers.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::product {

// llama.cpp's LoRA adapter and control vector flags, shared by the CLI, ninfer-serve and
// ninfer-perplexity. Each list flag is repeatable and takes comma-separated entries:
//   --lora FNAME                        adapters at scale 1
//   --lora-scaled FNAME:SCALE           adapters at their own scales
//   --lora-init-without-apply           every adapter at scale 0 until a request names it
//   --control-vector FNAME              control vectors at scale 1
//   --control-vector-scaled FNAME:SCALE control vectors at their own scales
//   --control-vector-layer-range START END
struct SteeringFlags {
    std::vector<LoraAdapterSource> lora_adapters;
    bool lora_init_without_apply = false;
    std::vector<ControlVectorSource> control_vectors;
    std::optional<ControlVectorLayers> control_vector_layers;

    void apply(EngineOptions& options) const {
        options.lora_adapters = lora_adapters;
        if (lora_init_without_apply) {
            for (LoraAdapterSource& adapter : options.lora_adapters) { adapter.scale = 0.0F; }
        }
        options.control_vectors       = control_vectors;
        options.control_vector_layers = control_vector_layers;
    }
};

inline constexpr std::string_view kSteeringHelp =
    "  --lora FNAME[,FNAME...]       load llama.cpp LoRA adapters (GGUF) that remove a\n"
    "                                residual direction, at scale 1; repeatable, at\n"
    "                                most 8 (Qwen3.5-core models)\n"
    "  --lora-scaled FNAME:SCALE[,...]\n"
    "                                load adapters at the given scales\n"
    "  --lora-init-without-apply     run every adapter at scale 0 unless a request\n"
    "                                names it\n"
    "  --control-vector FNAME[,...]  add llama.cpp control vectors (GGUF) to the\n"
    "                                residual, at scale 1; repeatable\n"
    "  --control-vector-scaled FNAME:SCALE[,...]\n"
    "                                add control vectors at the given scales\n"
    "  --control-vector-layer-range START END\n"
    "                                add only the vectors of layers START..END\n";

namespace detail {

inline std::vector<std::string_view> steering_entries(std::string_view flag,
                                                      std::string_view list) {
    std::vector<std::string_view> out;
    for (;;) {
        const std::size_t comma      = list.find(',');
        const std::string_view entry = list.substr(0, comma);
        if (entry.empty()) {
            throw std::invalid_argument(std::string(flag) + " has an empty entry");
        }
        out.push_back(entry);
        if (comma == std::string_view::npos) { return out; }
        list.remove_prefix(comma + 1);
    }
}

// FNAME:SCALE, split at the last colon so a Windows drive letter stays in the path.
inline std::pair<std::string, float> scaled_entry(std::string_view flag, std::string_view entry) {
    const std::size_t colon = entry.rfind(':');
    if (colon == 0 || colon == std::string_view::npos || colon + 1 == entry.size()) {
        throw std::invalid_argument(std::string(flag) + " entries must be FNAME:SCALE");
    }
    const std::string scale_text(entry.substr(colon + 1));
    char* end          = nullptr;
    const double scale = std::strtod(scale_text.c_str(), &end);
    if (end == scale_text.c_str() || *end != '\0' || !std::isfinite(scale)) {
        throw std::invalid_argument("invalid " + std::string(flag) + " scale: " + scale_text);
    }
    return {std::string(entry.substr(0, colon)), static_cast<float>(scale)};
}

inline std::uint32_t steering_layer(std::string_view flag, std::string_view text) {
    const std::string digits(text);
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(digits.c_str(), &end, 10);
    if (end == digits.c_str() || *end != '\0' || digits.front() == '-' || value > 0xffffffffULL) {
        throw std::invalid_argument("invalid " + std::string(flag) + " layer: " + digits);
    }
    return static_cast<std::uint32_t>(value);
}

} // namespace detail

// Takes `flag` and its values into `out` when it is one of the flags above, `next()` returning
// each following argument (and throwing when there is none). Returns false for any other flag.
template <class Next>
bool parse_steering_flag(std::string_view flag, Next&& next, SteeringFlags& out) {
    if (flag == "--lora" || flag == "--control-vector") {
        for (const std::string_view entry : detail::steering_entries(flag, next())) {
            if (flag == "--lora") {
                out.lora_adapters.push_back({.path = std::string(entry), .scale = 1.0F});
            } else {
                out.control_vectors.push_back({.path = std::string(entry), .scale = 1.0F});
            }
        }
        return true;
    }
    if (flag == "--lora-scaled" || flag == "--control-vector-scaled") {
        for (const std::string_view entry : detail::steering_entries(flag, next())) {
            auto [path, scale] = detail::scaled_entry(flag, entry);
            if (flag == "--lora-scaled") {
                out.lora_adapters.push_back({.path = std::move(path), .scale = scale});
            } else {
                out.control_vectors.push_back({.path = std::move(path), .scale = scale});
            }
        }
        return true;
    }
    if (switch_option(flag, "lora-init-without-apply",
                      [&](bool on) { out.lora_init_without_apply = on; })) {
        return true;
    }
    if (flag == "--control-vector-layer-range") {
        // Two arguments, or one "START END" (a configuration file's value).
        std::string first_text(next());
        std::string last_text;
        if (const std::size_t space = first_text.find(' '); space != std::string::npos) {
            const std::size_t end = first_text.find_first_not_of(' ', space);
            if (end == std::string::npos) {
                throw std::invalid_argument(std::string(flag) + " needs START END");
            }
            last_text = first_text.substr(end);
            first_text.resize(space);
        } else {
            last_text = std::string(next());
        }
        const std::uint32_t first = detail::steering_layer(flag, first_text);
        const std::uint32_t last  = detail::steering_layer(flag, last_text);
        if (first == 0 || first > last) {
            throw std::invalid_argument(
                "--control-vector-layer-range needs START END with 1 <= START <= END");
        }
        out.control_vector_layers = ControlVectorLayers{.first = first, .last = last};
        return true;
    }
    return false;
}

} // namespace ninfer::product
