#pragma once

#include "models/qwen3_5/config.h"
#include "models/qwen3_5/execution/steering.h"

#include "ninfer/engine_options.h"
#include "ninfer/request.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

// LoRA adapters that remove a residual direction and control vectors, both in llama.cpp's GGUF
// forms, read into the residual edits the text layers apply (ops::steer_residual).
//
// An edit site is a point where the residual takes a write: the token embedding, and each layer's
// mixer output and FFN output. A direction adapter with unit direction r turns a site's write o
// into o - k r (r^T o), k being the request's scale times the adapter's factor at that site. A
// control vector row is added after its layer's FFN write.
//
// Loading validates every file against the resident model -- the architecture, the hidden width,
// each site's layer kind and input width -- so a file built for another model is refused at
// startup instead of steering a different network.

namespace ninfer::models::qwen3_5 {

struct DirectionAdapter {
    std::filesystem::path path;
    float default_scale = 1.0F;
    // The unit direction r the adapter removes, [hidden_size].
    std::vector<float> direction;
    // What a request's scale is multiplied by at each site (steering_sites): llama.cpp's
    // alpha / rank (one without alpha) times |B|^2, zero where the adapter has no tensor.
    std::vector<float> site_factors;
    // SHA-256 of the file: the adapter's part of a request's cache identity.
    std::array<std::uint8_t, 32> digest{};
};

struct Steering {
    std::vector<DirectionAdapter> adapters;
    // The control vectors' sum, each times its scale: layer l's [hidden_size] row, at
    // l * hidden_size, is added after layer l where steered_layers[l] is set (at least one is).
    // Both empty without control vectors.
    std::vector<float> layer_vectors;
    std::vector<std::uint8_t> steered_layers;
    // SHA-256 over the control vector files, scales and layers: a persisted cache's part of it.
    std::array<std::uint8_t, 32> control_digest{};

    [[nodiscard]] bool empty() const noexcept { return adapters.empty() && layer_vectors.empty(); }
};

// Reads and validates every adapter and control vector against `text`. Throws
// std::invalid_argument naming the file and the first inconsistency found.
[[nodiscard]] Steering load_steering(const std::vector<LoraAdapterSource>& adapters,
                                     const std::vector<ControlVectorSource>& vectors,
                                     const std::optional<ControlVectorLayers>& layers,
                                     const TextConfig& text);

// Each loaded adapter's scale for a request, as llama.cpp reads its `lora` field: the listed ids
// take their scales (a repeated id its last) and the rest zero; unset takes every default. Throws
// std::invalid_argument for an id past the loaded adapters or a scale that is not finite.
[[nodiscard]] std::vector<float> resolve_lora_scales(const Steering& steering,
                                                     const std::optional<std::vector<LoraScale>>& lora);

// What a request's scales add to its prompt's cache identity: zero when every scale is zero, so
// such a request shares its cache with requests on an Engine without adapters.
[[nodiscard]] std::array<std::uint64_t, 2> lora_identity(const Steering& steering,
                                                         std::span<const float> scales);

} // namespace ninfer::models::qwen3_5
