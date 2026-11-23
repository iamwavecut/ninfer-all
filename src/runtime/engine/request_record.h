#pragma once

#include "core/nvtx.h"
#include "ninfer/base.h"
#include "ninfer/observability/execution.h"
#include "ninfer/request.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "runtime/engine/admission_policy.h"
#include "runtime/engine/generation_budget.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::runtime {

enum class RequestEngineHostPhase : std::uint8_t {
    Boundary,
    CommitOutput,
    Maintenance,
};

struct RequestHostTiming {
    std::uint64_t queue_wait_ns                   = 0;
    std::uint64_t engine_boundary_exposed_ns      = 0;
    std::uint64_t program_submit_exposed_ns       = 0;
    std::uint64_t program_post_exposed_ns         = 0;
    std::uint64_t engine_commit_output_exposed_ns = 0;
    std::uint64_t engine_maintenance_exposed_ns   = 0;
    std::uint64_t device_wait_exposed_ns          = 0;
    std::uint64_t decode_host_exposed_ns          = 0;
    std::uint64_t decode_device_wait_exposed_ns   = 0;
    std::uint64_t prefill_units                   = 0;
    std::uint64_t decode_rounds                   = 0;
    std::uint64_t control_units                   = 0;

    void expose_engine(RequestEngineHostPhase phase, std::uint64_t elapsed_ns,
                       bool decode_member) noexcept {
        switch (phase) {
        case RequestEngineHostPhase::Boundary:
            engine_boundary_exposed_ns += elapsed_ns;
            break;
        case RequestEngineHostPhase::CommitOutput:
            engine_commit_output_exposed_ns += elapsed_ns;
            break;
        case RequestEngineHostPhase::Maintenance:
            engine_maintenance_exposed_ns += elapsed_ns;
            break;
        }
        if (decode_member) { decode_host_exposed_ns += elapsed_ns; }
    }

    void expose_program(ExecutionTiming timing, bool decode_member) noexcept {
        program_submit_exposed_ns += timing.submit_host_ns;
        program_post_exposed_ns += timing.post_host_ns;
        device_wait_exposed_ns += timing.device_wait_ns;
        if (decode_member) {
            decode_host_exposed_ns += timing.host_ns();
            decode_device_wait_exposed_ns += timing.device_wait_ns;
        }
    }

    [[nodiscard]] GenerationEngineTiming public_snapshot() const noexcept {
        constexpr double kNanosecondsToSeconds = 1.0e-9;
        return GenerationEngineTiming{
            .queue_wait_seconds = static_cast<double>(queue_wait_ns) * kNanosecondsToSeconds,
            .engine_boundary_exposed_seconds =
                static_cast<double>(engine_boundary_exposed_ns) * kNanosecondsToSeconds,
            .program_submit_exposed_seconds =
                static_cast<double>(program_submit_exposed_ns) * kNanosecondsToSeconds,
            .program_post_exposed_seconds =
                static_cast<double>(program_post_exposed_ns) * kNanosecondsToSeconds,
            .engine_commit_output_exposed_seconds =
                static_cast<double>(engine_commit_output_exposed_ns) * kNanosecondsToSeconds,
            .engine_maintenance_exposed_seconds =
                static_cast<double>(engine_maintenance_exposed_ns) * kNanosecondsToSeconds,
            .device_wait_exposed_seconds =
                static_cast<double>(device_wait_exposed_ns) * kNanosecondsToSeconds,
            .decode_host_exposed_seconds =
                static_cast<double>(decode_host_exposed_ns) * kNanosecondsToSeconds,
            .decode_device_wait_exposed_seconds =
                static_cast<double>(decode_device_wait_exposed_ns) * kNanosecondsToSeconds,
            .prefill_units = prefill_units,
            .decode_rounds = decode_rounds,
            .control_units = control_units,
        };
    }
};

// Adds what one of a request's sequences measured to the sums of its earlier ones (a request
// queued again after a banned-word rewind runs several).
inline void accumulate_measurements(GenerationTimings& into, const GenerationTimings& from) {
    into.vision_seconds += from.vision_seconds;
    into.prefill_seconds += from.prefill_seconds;
    into.overlay_windows += from.overlay_windows;
    into.overlay_window_seconds += from.overlay_window_seconds;
    into.overlay_evict_seconds += from.overlay_evict_seconds;
    into.overlay_restore_seconds += from.overlay_restore_seconds;
    into.overlay_evicted_bytes += from.overlay_evicted_bytes;
    into.overlay_staged_bytes += from.overlay_staged_bytes;
    into.overlay_exclusive_windows += from.overlay_exclusive_windows;
    into.decode_seconds += from.decode_seconds;
}

inline void accumulate_measurements(SpeculativeStats& into, SpeculativeStats&& from) {
    const auto add = [](std::vector<std::uint64_t>& sum, const std::vector<std::uint64_t>& part) {
        if (sum.size() < part.size()) { sum.resize(part.size(), 0); }
        for (std::size_t index = 0; index < part.size(); ++index) { sum[index] += part[index]; }
    };
    if (!into.enabled && into.rounds == 0) {
        into.backend      = from.backend;
        into.enabled      = from.enabled;
        into.draft_window = from.draft_window;
        into.adaptive     = from.adaptive;
    }
    into.rounds += from.rounds;
    into.drafted_tokens += from.drafted_tokens;
    into.accepted_tokens += from.accepted_tokens;
    into.fallback_steps += from.fallback_steps;
    add(into.accepted_per_position, from.accepted_per_position);
    into.window_transitions += from.window_transitions;
    add(into.rounds_per_window, from.rounds_per_window);
    into.ngram_rounds += from.ngram_rounds;
    into.ngram_drafted_tokens += from.ngram_drafted_tokens;
    into.ngram_accepted_tokens += from.ngram_accepted_tokens;
    into.ngram_archive_rounds += from.ngram_archive_rounds;
    into.ngram_archive_drafted_tokens += from.ngram_archive_drafted_tokens;
    into.ngram_archive_accepted_tokens += from.ngram_archive_accepted_tokens;
}

enum class EngineRequestState : std::uint8_t {
    Waiting,
    Materializing,
    Prefill,
    DecodeReady,
    ControlReady,
    ModelFinished,
};

template <class ModelContract>
struct RequestRecord {
    using Clock          = std::chrono::steady_clock;
    using PreparedPrompt = typename ModelContract::PreparedPrompt;
    using OutputSession  = typename ModelContract::OutputSession;
    using BasePlan       = typename ModelContract::RequestBasePlan;
    using SequenceHandle = typename ModelContract::SequenceHandle;
    using StreamEvent    = std::variant<GenerationTimingObservation, OutputDelta>;

    RequestRecord(std::uint64_t request_identity, std::uint64_t publication_sequence,
                  PreparedPrompt input, OutputSession output_session, PromptSummary summary,
                  double frontend_seconds, ResolvedRequestOptions request_options,
                  OutputConsumerMode output_consumer, GenerationObservationOptions observation,
                  Clock::time_point limit, Clock::time_point submit_time)
        : generation_range(nvtx::Name::Generate, nvtx::Category::Runtime, request_identity),
          id(request_identity), publication_order(publication_sequence), prompt(std::move(input)),
          output(std::move(output_session)), prompt_summary(std::move(summary)),
          prepare_seconds(frontend_seconds), options(std::move(request_options)),
          consumer_mode(output_consumer), observation(observation), deadline(limit),
          submitted(submit_time) {}

    RequestRecord(const RequestRecord&)            = delete;
    RequestRecord& operator=(const RequestRecord&) = delete;

    [[nodiscard]] bool is_waiting() const noexcept {
        return model_state == EngineRequestState::Waiting;
    }

    [[nodiscard]] bool is_prefilling() const noexcept {
        return model_state == EngineRequestState::Prefill;
    }

    [[nodiscard]] bool is_materializing() const noexcept {
        return model_state == EngineRequestState::Materializing;
    }

    [[nodiscard]] bool is_decode_ready() const noexcept {
        return model_state == EngineRequestState::DecodeReady;
    }

    [[nodiscard]] bool is_control_ready() const noexcept {
        return model_state == EngineRequestState::ControlReady;
    }

    [[nodiscard]] bool is_model_finished() const noexcept {
        return model_state == EngineRequestState::ModelFinished;
    }

    // The process range follows Request ownership across submit, worker, and consumer threads.
    nvtx::ScopedAsyncRange generation_range;
    const std::uint64_t id;
    const std::uint64_t publication_order;
    PreparedPrompt prompt;
    std::unique_ptr<typename ModelContract::NgramArchive::Request> ngram_archive;
    bool ngram_admission_checked = false;
    OutputSession output;
    PromptSummary prompt_summary;
    double prepare_seconds = 0.0;
    ResolvedRequestOptions options;
    const OutputConsumerMode consumer_mode;
    const GenerationObservationOptions observation;
    Clock::time_point deadline;
    Clock::time_point submitted;
    std::optional<Clock::time_point> admitted_at;
    std::optional<Clock::time_point> first_token;
    std::optional<Clock::time_point> last_token;
    bool queue_wait_recorded = false;
    std::optional<GenerationBudget> budget;
    std::optional<BeginSummary> admitted_begin;
    std::optional<BeginSummary> begin;
    std::vector<TokenId> generated;
    // The content tokens' logprob records in generation order, when the request asked for them.
    std::vector<TokenLogprob> content_logprobs;
    // A branched read's records (Engine::submit_branches), one per branch.
    std::vector<BranchResult> branch_results;
    std::string content;
    std::string reasoning;
    std::optional<LaneId> lane;
    std::optional<SequenceHandle> sequence;
    std::atomic<bool> cancelled{false};
    EngineRequestState model_state        = EngineRequestState::Waiting;
    bool capture_pending                  = false;
    // A Device KV lease that stayed short after retained-cache reclaim was reported once.
    bool lease_shortfall_reported         = false;
    EngineRequestState post_capture_state = EngineRequestState::Prefill;
    std::optional<FinishReason> terminal_reason;

    std::optional<BasePlan> base_plan;
    std::uint64_t remaining_service_work = 0;
    std::uint64_t backfill_epoch         = 0;
    BackfillClass backfill_class         = BackfillClass::None;
    std::uint32_t computed_prompt_tokens = 0;
    // Prompt tokens of steps cut under the stall budget not yet charged as a whole chunk's quantum
    // (prefill_step_quanta).
    std::uint32_t paced_prefill_tokens = 0;
    GenerationTimings generation_timings;
    RequestHostTiming host_timing;
    SpeculativeStats speculative_stats;
    // The switch to post-thinking sampling has been applied to this request's lane.
    bool post_thinking_applied = false;
    MaterializationDiagnostics materialization_diagnostics;
    // The catalog cell and session digest the finished session was retained under, if any.
    std::int32_t retained_slot = -1;
    std::string retained_session_digest;

    // Banned words (bad_words). The Program cannot take a sequence back to an earlier state, so a
    // rewind ends the request's sequence and queues the request again: its prompt is then the
    // submitted one followed by the generated tokens the rewind keeps, and its first draw is the
    // rewound position (EngineCore::requeue_bad_words_rewind).
    struct BadWordsRequeue {
        // The prompt as submitted, which every requeue continues.
        PreparedPrompt prompt;
        // The submitted output limit and sampling, which a requeue resolves again.
        std::uint32_t requested_output_tokens = 0;
        ResolvedSamplingParameters sampling;
        // The generated tokens the pending rewind keeps; set from its commit until the request
        // settles and queues again.
        std::optional<std::uint32_t> target;
        std::uint32_t requeues = 0;
        // The generated tokens before the switch to post-thinking sampling, once it is applied.
        std::optional<std::uint32_t> post_thinking_at;
        // The first admission's Begin, which the result reports, and what the ended sequences
        // measured.
        std::optional<BeginSummary> begin;
        GenerationTimings timings;
        SpeculativeStats speculative;
    };
    std::optional<BadWordsRequeue> bad_words;

    std::mutex mutex;
    std::condition_variable cv;
    std::optional<GenerationStart> stream_start;
    std::optional<PromptProgress> stream_progress;
    std::vector<StreamEvent> events;
    GenerationResult result;
    std::exception_ptr error;
    bool response_done     = false;
    bool consumer_released = false;
    bool capacity_released = false;
};

} // namespace ninfer::runtime
