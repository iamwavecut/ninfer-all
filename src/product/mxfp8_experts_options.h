#pragma once

// --mxfp8-experts, which ninfer and ninfer-serve share: how Inkling 2 holds BF16 routed expert
// banks on the device (ExpertsMxfp8).

#include "ninfer/model_enums.h"

#include <stdexcept>
#include <string_view>

namespace ninfer::product {

// The help lines, in the 32-column layout of ninfer's and ninfer-serve's help.
inline constexpr const char* kMxfp8ExpertsHelp =
    "  --mxfp8-experts auto|on|off   Inkling 2: hold BF16 routed expert banks on\n"
    "                                the device as their exact MXFP8 re-encoding\n"
    "                                (halves their VRAM; output unchanged).\n"
    "                                auto (default): on when BF16 leaves under\n"
    "                                4 GiB free on a device; on: always; off: keep\n"
    "                                BF16\n";

[[nodiscard]] inline const char* mxfp8_experts_name(ExpertsMxfp8 mode) {
    switch (mode) {
    case ExpertsMxfp8::Auto: return "auto";
    case ExpertsMxfp8::On: return "on";
    case ExpertsMxfp8::Off: return "off";
    }
    return "auto";
}

[[nodiscard]] inline ExpertsMxfp8 parse_mxfp8_experts(std::string_view text) {
    if (text == "auto") { return ExpertsMxfp8::Auto; }
    if (text == "on") { return ExpertsMxfp8::On; }
    if (text == "off") { return ExpertsMxfp8::Off; }
    throw std::invalid_argument("--mxfp8-experts must be auto, on or off");
}

// Applies `option` when it is --mxfp8-experts, taking its argument from `value()`; false leaves it
// to the caller.
template <class Value>
bool parse_mxfp8_experts_option(std::string_view option, ExpertsMxfp8& out, Value&& value) {
    if (option != "--mxfp8-experts") { return false; }
    out = parse_mxfp8_experts(std::string_view(value()));
    return true;
}

} // namespace ninfer::product
