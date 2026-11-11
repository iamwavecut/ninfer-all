#pragma once

#include "ninfer/base.h"
#include "ninfer/engine_options.h"
#include "ninfer/kv_cache_storage.h"
#include "ninfer/model_enums.h"
#include "ninfer/request.h"
#include "product/logging/logging.h"
#include "product/prompt_route_options.h"
#include "product/steering_options.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::cli {

struct Options {
    bool help_requested = false;
    bool version_requested = false;

    std::filesystem::path artifact_path;
    std::filesystem::path chat_template_path;
    std::string prompt;
    std::filesystem::path messages_path;

    std::uint32_t max_new        = 128;
    std::uint32_t max_context    = 2048;
    KvCapacityPolicy kv_capacity = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t prefill_chunk  = 0; // 0: the Engine's automatic chunk
    int device                   = 0;
    // Several ids split the model's layers into one pipeline stage per device, the first also
    // holding the embedding, head and round state. Empty means use `device`.
    std::vector<int> devices;
    // Layers per pipeline stage, one count per entry of `devices`. Empty lets the engine choose.
    std::vector<std::uint32_t> stage_layers;
    // The `devices` share every layer instead (tensor parallelism).
    bool tensor_parallel = false;
    // The relative size of each device's tensor-parallel share. Empty lets the engine choose.
    std::vector<double> tensor_split;
    ninfer::TensorAttention tensor_attention = ninfer::TensorAttention::Heads;
    ninfer::TensorParallelTransport tensor_parallel_transport =
        ninfer::TensorParallelTransport::Auto;
    ninfer::TensorParallelPartials tensor_parallel_prompt_partials =
        ninfer::TensorParallelPartials::Fp32;
    bool invariant_attention                 = false;
    // Qwen3.8-Flash-Next: where the expert banks live, and where its n-gram table comes from.
    ninfer::ExpertResidency expert_residency = ninfer::ExpertResidency::Device;
    std::optional<std::uint64_t> expert_cache_bytes;
    ninfer::HybridExpertOptions hybrid_experts;
    ninfer::NgramTableOptions ngram_table;
    std::optional<std::uint32_t> kv_resident_positions; // Flash-Next: --kv-resident

    KvCacheStorage kv_cache = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    bool enable_vision  = false;
    VisionResidency vision_residency       = VisionResidency::Resident;
    std::uint32_t vision_max_merged_tokens = 16384;
    bool use_cuda_graph = true;
    std::uint32_t gpu_clock_floor_mhz = 0;
    bool lm_head_q4     = false;
    bool lm_head_q6     = false;
    bool embedding_q4   = false;
    bool embedding_q6   = false;
    ninfer::EmbeddingResidency embedding_residency = ninfer::EmbeddingResidency::Device;
    bool mtp_experts_q4 = false;
    bool gdn_state_fp16 = false;
    bool rope_yarn      = false;
    float rope_yarn_factor = 1.0F;
    float rope_scaling_factor                   = 1.0F;
    std::uint32_t rope_scaling_original_context = 0;
    bool wddm_evictable_budget = false;
    bool mlp_a8_decode  = false;
    bool prefill_a8     = true;
    bool nvfp4_w4a8_int = false;
    bool nvfp4_w4a16    = false;
    bool prefill_cublas = false;
    bool prefill_cublas_projections = true;

    bool raw_output      = false;
    bool print_token_ids = false;
    std::optional<bool> log_colours; // --log-colours on|off: colour the statistics (unset = off)
    std::optional<bool> enable_thinking;
    std::optional<std::uint32_t> thinking_budget;
    ninfer::ReasoningLoopAction reasoning_loop = ninfer::ReasoningLoopAction::Off;
    std::optional<ReasoningEffort> reasoning_effort;
    // LoRA adapters (at their scales) and control vectors the run steers with.
    ninfer::product::SteeringFlags steering;
    // --prefill-tail and --iq1m-experts.
    ninfer::product::PromptRouteFlags prompt_routes;

    std::vector<TokenId> stop_token_ids;
    std::vector<StopString> stop_strings;

    // Omitted fields are resolved from the loaded model and rendered prompt mode by Engine.
    SamplingOverrides sampling;
    // --post-thinking*: a thinking request's answer switches to these once reasoning closes.
    std::optional<SamplingOverrides> post_thinking_sampling;
    StructuredOutputOptions structured_output;
    bool greedy                 = false;
    bool gumbel_sampling        = false; // --gumbel-sampling (EngineOptions::gumbel_sampling)
    product::LogLevel log_level = product::LogLevel::Info;
};

[[nodiscard]] Options parse_options(int argc, char** argv);
[[nodiscard]] std::string usage_text(const char* argv0);

} // namespace ninfer::cli
