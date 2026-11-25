#pragma once

// --loop-kv, which ninfer, ninfer-serve and ninfer-perplexity share: which keys and values the
// passes of Inkling 2's looped core keep (LoopKvCache).

#include "ninfer/model_enums.h"

#include <stdexcept>
#include <string_view>

namespace ninfer::product {

// The help lines, in the 32-column layout of ninfer's and ninfer-serve's help.
inline constexpr const char* kLoopKvHelp =
    "  --loop-kv per-pass|last-full|last-all\n"
    "                                Inkling 2: keys and values of its looped core.\n"
    "                                per-pass (default): every pass keeps its own.\n"
    "                                last-full: the full-attention layers keep only\n"
    "                                the last pass's, which the earlier passes read\n"
    "                                (half the KV that grows with the context);\n"
    "                                last-all: the windowed layers' rings too\n"
    "                                (transformers' last_loop). Lossy: the output\n"
    "                                changes\n";

[[nodiscard]] inline const char* loop_kv_name(LoopKvCache mode) {
    switch (mode) {
    case LoopKvCache::PerPass: return "per-pass";
    case LoopKvCache::LastFull: return "last-full";
    case LoopKvCache::LastAll: return "last-all";
    }
    return "per-pass";
}

[[nodiscard]] inline LoopKvCache parse_loop_kv(std::string_view text) {
    if (text == "per-pass") { return LoopKvCache::PerPass; }
    if (text == "last-full") { return LoopKvCache::LastFull; }
    if (text == "last-all") { return LoopKvCache::LastAll; }
    throw std::invalid_argument("--loop-kv must be per-pass, last-full or last-all");
}

// Applies `option` when it is --loop-kv, taking its argument from `value()`; false leaves it to
// the caller.
template <class Value>
bool parse_loop_kv_option(std::string_view option, LoopKvCache& out, Value&& value) {
    if (option != "--loop-kv") { return false; }
    out = parse_loop_kv(std::string_view(value()));
    return true;
}

} // namespace ninfer::product
