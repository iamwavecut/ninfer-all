#pragma once
// The Engine core of a Llama (LlamaForCausalLM) artifact. The model family has its own load and
// execution (models/llama) and serves Nanbeige ChatML through the shared frontend.
//
// One active request with bounded FIFO admission and chunked prefill. Active full-attention KV
// lives in pageable host pages; a bounded device staging buffer keeps all arithmetic on CUDA.
// Snapshots retain an immutable prefix frontier, and bounded pinned images allow prefix reuse
// after the active sequence is reset. Optional context lookup verifies at most seven drafts.
#include "core/device.h"
#include "models/llama/executor.h"
#include "models/llama/model.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "ninfer/observability/resources.h"
#include "ninfer/observability/snapshot.h"
#include "ninfer/request.h"
#include "ninfer/types_fwd.h"
#include "runtime/contract/request.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::runtime {

struct LlamaInstance {
    std::unique_ptr<models::llama::Model> model;
    models::qwen3_5::Frontend frontend;
    std::unique_ptr<models::llama::Executor> executor;
    std::uint32_t capacity         = 0;
    std::size_t free_after_weights = 0;
    std::size_t free_after_startup = 0;
};

struct ConstructedLlama {
    std::unique_ptr<LlamaInstance> instance;
    LoadSummary load;
    ModelMetadata model_metadata;
};

[[nodiscard]] bool is_llama_artifact(const std::filesystem::path& path);
[[nodiscard]] ConstructedLlama construct_llama(const EngineOptions& options, DeviceContext& device);

class LlamaCore {
public:
    using Clock = std::chrono::steady_clock;
    struct Request;

    class Submission {
    public:
        Submission() noexcept = default;
        ~Submission();
        Submission(Submission&& other) noexcept;
        Submission& operator=(Submission&& other) noexcept;
        Submission(const Submission&)            = delete;
        Submission& operator=(const Submission&) = delete;

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation);

        [[nodiscard]] std::optional<std::uint32_t> effective_thinking_budget() const noexcept {
            return effective_thinking_budget_;
        }

    private:
        friend class LlamaCore;
        Submission(LlamaCore& owner, std::shared_ptr<Request> request,
                   std::optional<std::uint32_t> budget) noexcept;
        LlamaCore* owner_ = nullptr;
        std::shared_ptr<Request> request_;
        std::optional<std::uint32_t> effective_thinking_budget_;
    };

    LlamaCore(LlamaInstance& instance, DeviceContext& device, const EngineOptions& options);
    ~LlamaCore();
    LlamaCore(const LlamaCore&)            = delete;
    LlamaCore& operator=(const LlamaCore&) = delete;

    void stop() noexcept;

    Submission submit(models::qwen3_5::PreparedPrompt prompt, PromptSummary prompt_summary,
                      double prepare_seconds, ResolvedRequestOptions options,
                      OutputConsumerMode consumer_mode, GenerationObservationOptions observation,
                      Clock::time_point pending_deadline = {});

    // One request of submit_group: the arguments of submit.
    struct GroupMember {
        models::qwen3_5::PreparedPrompt prompt;
        PromptSummary prompt_summary;
        double prepare_seconds = 0.0;
        ResolvedRequestOptions options;
        OutputConsumerMode consumer_mode = OutputConsumerMode::Aggregate;
        GenerationObservationOptions observation;
        Clock::time_point pending_deadline;
    };

    // Queues the members together, in order, under one queue lock, so the worker admits them at
    // one boundary: the first to reach a prefix they share stores its state there and the others
    // resume from it. Capacity is reserved for the whole group, and an error for one member
    // queues none.
    std::vector<Submission> submit_group(std::vector<GroupMember> group);

    // Natural-log probabilities of tokens [first_target, n) given their prefixes.
    [[nodiscard]] std::vector<float> score(models::qwen3_5::PreparedPrompt prompt,
                                           std::uint32_t first_target);

    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    [[nodiscard]] bool is_available() const;
    [[nodiscard]] bool has_failed() const;

    void reset_memory_peaks() noexcept {}

private:
    struct Impl;
    // A member's request, validated and ready to queue, and its effective thinking budget.
    std::shared_ptr<Request> make_request(GroupMember& member,
                                          std::optional<std::uint32_t>& budget);
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::runtime
