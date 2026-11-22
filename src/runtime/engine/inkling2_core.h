#pragma once
// The Engine core of an Inkling 2 (Inkling2ForCausalLM) artifact. The model family has its own
// load and execution (models/inkling2) and serves prompts through the shared frontend in its
// Inkling dialect.
//
// Up to max_concurrency requests run at once, each on its own executor sequence, admitted in FIFO
// order. A prompt is prefilled in chunks, one request at a time; between its chunks every decoding
// request advances one token, all of them in one batched pass. With the context cache on, a
// sequence keeps its state when its request ends, and a snapshot of its sliding-window rings where
// the prompt's last user turn closes: a later prompt that starts with what the sequence holds, or
// with the prompt up to that point, resumes there instead of prefilling it again. What a sequence
// holds before another request takes it, and its state at the end of a prompt's shared prefixes
// (the leading system messages and tool declarations, a declared cache boundary), go to images in
// pinned host memory within --host-cache-mib; a prompt on any sequence resumes from the longest
// image it starts with, including one stored while it waited for its first prompt chunk.
//
// With --lookup-ngram, a request whose last tokens appeared earlier in its sequence verifies what
// followed them then (up to --draft-tokens) after its next token in one pass; with --spec dspine
// a DSpine drafter proposes, and with --spec dspark the release's DSpark drafter proposes for a
// request decoding alone (the Engine's choice without --spec when the artifact carries it and the
// Engine serves one request at a time), each round's width picked from 1 to --draft-tokens by the
// drafts' survival and the measured round times with --adaptive-mtp. The acceptance keeps drafts
// while they equal the request's own keyed draw, so the output is plain decoding's.
#include "core/device.h"
#include "models/inkling2/executor.h"
#include "models/inkling2/model.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "ninfer/observability/resources.h"
#include "ninfer/observability/snapshot.h"
#include "ninfer/request.h"
#include "ninfer/types_fwd.h"
#include "runtime/contract/request.h"
#include "runtime/engine/session_slot_record.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <span>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::runtime {

struct Inkling2Instance {
    std::unique_ptr<models::inkling2::Model> model;
    models::qwen3_5::Frontend frontend;
    std::unique_ptr<models::inkling2::Executor> executor;
    std::uint32_t capacity = 0;
    std::size_t free_after_weights = 0;
    std::size_t free_after_startup = 0;
    // --expert-profile-out: the routing profile written at request boundaries.
    std::filesystem::path record_profile;
};

struct ConstructedInkling2 {
    std::unique_ptr<Inkling2Instance> instance;
    LoadSummary load;
    ModelMetadata model_metadata;
};

[[nodiscard]] bool is_inkling2_artifact(const std::filesystem::path& path);
[[nodiscard]] ConstructedInkling2 construct_inkling2(const EngineOptions& options,
                                                 DeviceContext& device);

class Inkling2Core {
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
        friend class Inkling2Core;
        Submission(Inkling2Core& owner, std::shared_ptr<Request> request,
                   std::optional<std::uint32_t> budget) noexcept;
        Inkling2Core* owner_ = nullptr;
        std::shared_ptr<Request> request_;
        std::optional<std::uint32_t> effective_thinking_budget_;
    };

    Inkling2Core(Inkling2Instance& instance, DeviceContext& device, const EngineOptions& options);
    ~Inkling2Core();
    Inkling2Core(const Inkling2Core&)            = delete;
    Inkling2Core& operator=(const Inkling2Core&) = delete;

    void stop() noexcept;

    // Session slots (session_slots.h): slot i is sequence i. Each runs between the worker's steps.
    [[nodiscard]] SessionSlotRecord save_slot(std::uint32_t slot,
                                              const std::string& expected_digest);
    [[nodiscard]] std::pair<std::uint32_t, std::string>
    restore_slot(std::uint32_t slot, std::span<const std::uint8_t> record);
    std::uint32_t erase_slot(std::uint32_t slot, const std::string& expected_digest);
    [[nodiscard]] std::vector<SlotState> slot_states() const;

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
