#pragma once

#include <string_view>

namespace ninfer::models {

// Qwen4Exp (Qwen3.8-Flash-Next), Gemma 4, Flint, DeepSeek-V4.1 and Inkling 2 are their own
// families (models/qwen4_exp, models/gemma4, models/flint, models/deepseek_v41, models/inkling2);
// the Qwen3.5 config parser never resolves them.
enum class Architecture { Qwen3_5, Qwen3_5Moe, Qwen4Exp, Gemma4, Flint, DeepSeekV41, Inkling2 };

[[nodiscard]] Architecture resolve_architecture(std::string_view architecture,
                                                std::string_view model_type);
[[nodiscard]] std::string_view architecture_name(Architecture architecture) noexcept;

} // namespace ninfer::models
