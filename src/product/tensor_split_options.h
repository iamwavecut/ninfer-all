#pragma once

#include "ninfer/model_enums.h"
#include "product/option_layers.h"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <vector>

namespace ninfer::product {

// "3,2": the relative size of each tensor-parallel device's share, one positive weight per device in
// --devices. Whether there is one per device is checked against the parsed --devices.
inline std::vector<double> parse_tensor_split(std::string_view text) {
    std::vector<double> weights;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string_view piece =
            text.substr(start, comma == std::string_view::npos ? std::string_view::npos
                                                               : comma - start);
        double value            = 0.0;
        const auto [end, error] = std::from_chars(piece.data(), piece.data() + piece.size(), value);
        if (piece.empty() || error != std::errc{} || end != piece.data() + piece.size() ||
            !std::isfinite(value) || value <= 0.0) {
            throw std::invalid_argument("--tensor-split entries must be positive numbers");
        }
        weights.push_back(value);
        if (comma == std::string_view::npos) { break; }
        start = comma + 1;
    }
    return weights;
}

// "heads" or "positions" (--tensor-attention).
inline TensorAttention parse_tensor_attention(std::string_view text) {
    if (text == "heads") { return TensorAttention::Heads; }
    if (text == "positions") { return TensorAttention::Positions; }
    throw std::invalid_argument("--tensor-attention must be heads or positions");
}

inline constexpr std::string_view kTensorAttentionHelp =
    "  --tensor-attention heads|positions  with --tensor-parallel, divide each\n"
    "                                attention layer by KV heads (default; devices\n"
    "                                past the KV heads hold none), or by cache pages:\n"
    "                                every device holds every head of its pages and\n"
    "                                the devices merge their partial outputs\n";

// "auto", "host", "peer-read" or "peer-write" (--tensor-parallel-transport).
inline TensorParallelTransport parse_tensor_parallel_transport(std::string_view text) {
    if (text == "auto") { return TensorParallelTransport::Auto; }
    if (text == "host") { return TensorParallelTransport::Host; }
    if (text == "peer-read") { return TensorParallelTransport::PeerRead; }
    if (text == "peer-write") { return TensorParallelTransport::PeerWrite; }
    throw std::invalid_argument(
        "--tensor-parallel-transport must be auto, host, peer-read or peer-write");
}

inline constexpr std::string_view kTensorParallelTransportHelp =
    "  --tensor-parallel-transport auto|host|peer-read|peer-write\n"
    "                                with --tensor-parallel, how the devices exchange\n"
    "                                their partial sums: through host memory, by\n"
    "                                reading or by writing each other's memory (peer\n"
    "                                access), or by whichever was fastest at each\n"
    "                                size when timed at startup (default: auto)\n";

// "fp32", "bf16" or "int8" (--tensor-parallel-prompt-partials).
inline TensorParallelPartials parse_tensor_parallel_partials(std::string_view text) {
    if (text == "fp32") { return TensorParallelPartials::Fp32; }
    if (text == "bf16") { return TensorParallelPartials::Bf16; }
    if (text == "int8") { return TensorParallelPartials::Int8; }
    throw std::invalid_argument("--tensor-parallel-prompt-partials must be fp32, bf16 or int8");
}

inline constexpr std::string_view kTensorParallelPartialsHelp =
    "  --tensor-parallel-prompt-partials fp32|bf16|int8\n"
    "                                with --tensor-parallel on a Qwen3.5-core model or\n"
    "                                Qwen3.8-Flash-Next, how a prompt chunk's partial\n"
    "                                sums cross between the GPUs: bf16 moves half\n"
    "                                their bytes and int8 (blocks of 32 values with a\n"
    "                                scale) 9/32, but each rounds every device's\n"
    "                                partial, so prompt outputs differ (default: fp32)\n";

inline constexpr std::string_view kInvariantAttentionHelp =
    "  --invariant-attention         attention outputs independent of decode width,\n"
    "                                verification window and batch (Qwen3.5-core)\n";

// The checks --tensor-split needs once --devices and --tensor-parallel are known.
inline void check_tensor_split(const std::vector<double>& weights, std::size_t devices,
                               bool tensor_parallel) {
    if (weights.empty()) { return; }
    if (!tensor_parallel) {
        throw OptionConflict({"tensor-split", "tensor-parallel"},
                             "--tensor-split needs --tensor-parallel");
    }
    if (weights.size() != devices) {
        throw OptionConflict({"tensor-split", "devices"},
                             "--tensor-split needs one weight per device in --devices");
    }
}

inline void check_tensor_attention(TensorAttention mode, bool tensor_parallel) {
    if (mode == TensorAttention::Positions && !tensor_parallel) {
        throw OptionConflict({"tensor-attention", "tensor-parallel"},
                             "--tensor-attention needs --tensor-parallel");
    }
}

inline void check_tensor_parallel_transport(TensorParallelTransport transport,
                                            bool tensor_parallel) {
    if (transport != TensorParallelTransport::Auto && !tensor_parallel) {
        throw OptionConflict({"tensor-parallel-transport", "tensor-parallel"},
                             "--tensor-parallel-transport needs --tensor-parallel");
    }
}

inline void check_tensor_parallel_partials(TensorParallelPartials partials, bool tensor_parallel) {
    if (partials != TensorParallelPartials::Fp32 && !tensor_parallel) {
        throw OptionConflict({"tensor-parallel-prompt-partials", "tensor-parallel"},
                             "--tensor-parallel-prompt-partials needs --tensor-parallel");
    }
}

} // namespace ninfer::product
