#pragma once

// The prompt-phase route options that ninfer, ninfer-serve and ninfer-perplexity share:
//   --prefill-tail exact|full|N    DeepSeek-V4.1-Flash's decoder blocks over a prompt's tail
//   --iq1m-experts matrix|vector   the kernel of wide calls' IQ1_M routed experts

#include "ninfer/engine_options.h"
#include "ninfer/model_enums.h"

#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace ninfer::product {

struct PromptRouteFlags {
    PrefillTail prefill_tail        = PrefillTail::Exact;
    std::uint32_t prefill_tail_rows = 0;
    bool iq1m_expert_matrix         = true;

    void apply(EngineOptions& options) const {
        options.prefill_tail       = prefill_tail;
        options.prefill_tail_rows  = prefill_tail_rows;
        options.iq1m_expert_matrix = iq1m_expert_matrix;
    }
};

// The help lines of the options, in the 32-column layout of ninfer's and ninfer-serve's help.
inline constexpr const char* kPrefillTailHelp =
    "  --prefill-tail exact|full|N   DeepSeek-V4.1: the prompt rows that blocks\n"
    "                                21-39, which keep no compressed cache, run:\n"
    "                                those the kept state and the logits depend\n"
    "                                on, from the prefill chunk at or below them\n"
    "                                (exact, default; output unchanged), every\n"
    "                                row (full), or the last N >= 128 before each\n"
    "                                kept state (lossy: the output changes)\n";

inline constexpr const char* kIq1mExpertsHelp =
    "  --iq1m-experts matrix|vector  IQ1_M routed experts of calls past eight\n"
    "                                tokens: the integer matrix kernel past 16\n"
    "                                rows an expert for gate and up, from 6 for\n"
    "                                down (default), or the vector kernel; the\n"
    "                                sums round differently (DeepSeek-V4.1,\n"
    "                                Qwen3.8-Flash-Next)\n";

// Applies `option` when it is one of the options above, taking its argument from `value()`;
// false leaves it to the caller. `tail` false: --prefill-tail is not an option of the caller
// (scoring needs every row).
template <class Value>
bool parse_prompt_route_option(std::string_view option, PromptRouteFlags& out, Value&& value,
                               bool tail = true) {
    if (tail && option == "--prefill-tail") {
        const std::string text(value());
        if (text == "exact") {
            out.prefill_tail      = PrefillTail::Exact;
            out.prefill_tail_rows = 0;
        } else if (text == "full") {
            out.prefill_tail      = PrefillTail::Full;
            out.prefill_tail_rows = 0;
        } else {
            std::uint32_t rows      = 0;
            const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), rows);
            if (error != std::errc{} || end != text.data() + text.size() || rows < 128 ||
                rows > (1U << 24U)) {
                throw std::invalid_argument(
                    "--prefill-tail must be exact, full or a row count of at least 128");
            }
            out.prefill_tail      = PrefillTail::Rows;
            out.prefill_tail_rows = rows;
        }
    } else if (option == "--iq1m-experts") {
        const std::string text(value());
        if (text == "matrix") {
            out.iq1m_expert_matrix = true;
        } else if (text == "vector") {
            out.iq1m_expert_matrix = false;
        } else {
            throw std::invalid_argument("--iq1m-experts must be matrix or vector");
        }
    } else {
        return false;
    }
    return true;
}

} // namespace ninfer::product
