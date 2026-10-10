#include "runtime/engine/qwen4_exp_core.h"

#include "artifact/formats.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/startup.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/vision_control.h"
#include "models/qwen4_exp/ngram_component.h"
#include "models/qwen4_exp/ngram_profile.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/logprob_topk.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/target_logprobs.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/lookup_draft.h"
#include "runtime/contract/mtp_adaptive.h"
#include "runtime/engine/diagnostics.h"
#include "runtime/engine/effective_thinking_budget.h"
#include "runtime/engine/generation_budget.h"
#include "runtime/engine/host_memory.h"
#include "runtime/engine/model_instance.h"
#include "text/structured_output.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <bit>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <variant>

namespace ninfer::runtime {
namespace {

// Prompt chunks in one prefill step while other requests decode (see prefill_step).
constexpr std::uint32_t kPeerSpanChunks = 2;

using Clock = std::chrono::steady_clock;

// Device memory kept free when the host expert cache grows after warm-up.
constexpr std::uint64_t kGrowthMargin = 640ULL << 20;

std::uint64_t elapsed_ns(Clock::time_point from, Clock::time_point to) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count());
}

double seconds(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double>(to - from).count();
}

// Requests fail alone on their own errors; anything else may have left the device in an unknown
// state, so it fails the Engine.
bool request_error(const std::exception_ptr& error) {
    try {
        std::rethrow_exception(error);
    } catch (const RequestError&) { return true; } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {}
    return false;
}

} // namespace

bool is_qwen4_exp_artifact(const std::filesystem::path& path) {
    if (path.extension() != ".ninfer") { return false; }
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) { return false; }
    std::optional<artifact::Reader> reader;
    try {
        reader.emplace(path);
    } catch (const std::exception&) { return false; }
    if (models::qwen4_exp::is_ngram_table_artifact(*reader)) {
        throw std::invalid_argument(path.string() +
                                    " is a Qwen3.8-Flash-Next n-gram table, not a model; pass it "
                                    "with --ngram-table next to the model");
    }
    return models::qwen4_exp::is_qwen4_exp(*reader);
}

ConstructedQwen4Exp construct_qwen4_exp(const EngineOptions& options, DeviceContext& device) {
    const auto start = Clock::now();
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    if (options.enable_vision && options.vision_residency != VisionResidency::Resident) {
        throw std::invalid_argument(
            "Qwen3.8-Flash-Next runs its Vision tower resident on the device only");
    }
    const SpeculativeOptions& speculative = options.speculative;
    if (speculative.backend == SpeculativeBackend::DFlash ||
        speculative.backend == SpeculativeBackend::DFlash2) {
        throw std::invalid_argument(
            "Qwen3.8-Flash-Next drafts with its MTP block only (--spec mtp)");
    }
    const bool mtp = speculative.backend == SpeculativeBackend::Mtp;
    if (!std::isfinite(speculative.draft_min_p) || speculative.draft_min_p < 0 ||
        speculative.draft_min_p > 1 || (!mtp && speculative.draft_min_p != 0)) {
        throw std::invalid_argument("--draft-min-p requires --spec mtp and a finite value in [0,1]");
    }
    if (mtp && (speculative.draft_tokens == 0 || speculative.draft_tokens > 15)) {
        throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,15]");
    }
    if (!mtp && speculative.ngram_draft_tokens != 0) {
        throw std::invalid_argument("n-gram copy proposals are not available for "
                                    "Qwen3.8-Flash-Next");
    }
    if (speculative.mtp_attention_window != 0 || speculative.proposal_head != ProposalHead::Full ||
        speculative.ngram_archive_bytes != 0) {
        throw std::invalid_argument("Qwen3.8-Flash-Next's MTP drafting has no "
                                    "--mtp-attention-window, --lm-head-draft or n-gram archive");
    }
    if (speculative.lookup_ngram != 0 && !mtp) {
        throw std::invalid_argument("--lookup-ngram requires --spec mtp");
    }
    if (speculative.mtp_policy == MtpDraftPolicy::Adaptive && !mtp) {
        throw std::invalid_argument("--adaptive-mtp requires --spec mtp");
    }
    if (options.context_cache.disk_kv_directstorage) {
        throw std::invalid_argument(
            "Qwen3.8-Flash-Next's context cache reads its disk tier without DirectStorage");
    }
    const NgramTableOptions& table = options.ngram_table;
    const NgramTableOptions defaults;
    if (table.disabled &&
        (!table.path.empty() || table.residency != defaults.residency || table.io != defaults.io ||
         table.ram_budget_bytes || !table.hot_profile.empty() || table.lock ||
         table.io_depth != defaults.io_depth)) {
        throw std::invalid_argument("--no-ngram-table excludes the other n-gram table options");
    }
    if (table.residency != NgramResidency::RamHot && !table.hot_profile.empty()) {
        throw std::invalid_argument(
            "--ngram-hot-profile belongs to --ngram-residency ram-hot");
    }
    if (table.residency == NgramResidency::Ram && table.ram_budget_bytes) {
        throw std::invalid_argument("--ngram-ram-mib needs --ngram-residency disk or ram-hot");
    }
    if (table.lock && table.residency == NgramResidency::Disk) {
        throw std::invalid_argument("--ngram-lock needs --ngram-residency ram or ram-hot");
    }
    if (table.io_depth == 0 || table.io_depth > 1024) {
        throw std::invalid_argument("--ngram-io-depth takes 1..1024 reads");
    }
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    const artifact::Reader reader(options.artifact_path);
    const auto text =
        models::qwen4_exp::parse_text_config(reader.directory().component("text").config);
    const bool has_ple = !text.ple_layers.empty();
    if (!has_ple &&
        (!table.path.empty() || table.residency != defaults.residency || table.io != defaults.io ||
         table.ram_budget_bytes || !table.hot_profile.empty() || table.lock ||
         table.io_depth != defaults.io_depth)) {
        throw std::invalid_argument("n-gram table options do not apply to a slice without PLE layers");
    }
    // The n-gram table, and the hot-row profile of a ram-hot table, are located before anything
    // else starts, so a model without them fails at once.
    std::optional<models::qwen4_exp::NgramTableSource> ngram;
    models::qwen4_exp::NgramReadOptions ngram_read{
        .residency    = table.residency,
        .io           = table.io,
        .budget_bytes = table.ram_budget_bytes.value_or(
            table.residency == NgramResidency::RamHot ? std::uint64_t{4} << 30U : 0),
        .lock         = table.lock,
        .depth        = table.io_depth};
    if (!table.disabled && has_ple) {
        ngram = models::qwen4_exp::ngram_table_source(
            reader, options.artifact_path, text, table.path,
            table.residency == NgramResidency::RamHot && table.hot_profile.empty());
        if (table.residency == NgramResidency::RamHot) {
            if (!table.hot_profile.empty()) {
                auto profile = models::qwen4_exp::read_ngram_profile(table.hot_profile);
                models::qwen4_exp::check_ngram_profile(
                    profile, models::qwen4_exp::derive_ngram_hash_constants(text.ngram),
                    table.hot_profile);
                ngram_read.hot_rows = std::move(profile.rows);
            } else if (ngram->hot_profile) {
                ngram_read.hot_rows = std::move(ngram->hot_profile->rows);
                ngram->hot_profile.reset();
            } else {
                throw std::invalid_argument("--ngram-residency ram-hot needs a profile in the "
                    "table artifact or --ngram-hot-profile PATH (ninfer-ngram-profile makes one)");
            }
        }
    }
    inspect.complete();
    install_device_route_profile_for(options, device);
    models::qwen4_exp::LoadOptions load;
    load.artifact     = options.artifact_path;
    load.ranks        = device.size();
    load.stage_layers = options.stage_layers;
    load.experts      = options.expert_residency;
    load.vision       = options.enable_vision;
    load.mtp          = mtp;
    StartupPhaseScope materialize(options.startup_observer, StartupPhase::TargetPlan);
    auto model = models::qwen4_exp::load_model(reader, load, device, &options.startup_observer);
    device.synchronize();
    materialize.complete();
    const auto free_bytes = [&] {
        RankBinding bind(device, 0);
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        return free;
    };
    const std::size_t free_after_weights = free_bytes();

    StartupPhaseScope frontend_phase(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<Qwen4ExpInstance>(Qwen4ExpInstance{
        .model    = nullptr,
        .frontend = models::qwen3_5::make_frontend(
            model->resources(), {.chat_template_path      = options.chat_template_path,
                                 .architecture            = models::Architecture::Qwen4Exp,
                                 .vision_enabled          = options.enable_vision,
                                 .max_context             = options.max_context,
                                 .media_cache_bytes       = options.media_cache_bytes,
                                 .media_live_bytes        = options.media_live_bytes,
                                 .thinking_budget_message = options.thinking_budget_message}),
        .executor = nullptr,
        .capacity = options.max_context});
    frontend_phase.complete();
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    models::qwen4_exp::ExecutorOptions executor;
    executor.max_context     = options.max_context;
    executor.sequences       = options.max_concurrency;
    executor.draft_tokens    = mtp ? speculative.draft_tokens : 0;
    executor.draft_min_p     = speculative.draft_min_p;
    // A chunk also holds a verification of every sequence at once.
    executor.prefill_chunk   = std::max(std::clamp<std::uint32_t>(options.prefill_chunk, 64, 4096),
                                        (executor.draft_tokens + 1) * executor.sequences);
    executor.ngram           = std::move(ngram);
    executor.ngram_read      = std::move(ngram_read);
    executor.expert_cache_bytes =
        options.expert_cache_bytes.value_or(models::qwen4_exp::ExecutorOptions::kAutoExpertCache);
    executor.cuda_graphs = options.use_cuda_graph;
    executor.hybrid_experts = options.hybrid_experts;
    executor.vision_max_merged_tokens = options.vision_max_merged_tokens;
    executor.kv_cache                 = options.kv_cache;
    instance->executor = std::make_unique<models::qwen4_exp::Executor>(*model, device, executor);
    device.synchronize();
    program.complete();
    {
        // The first request would otherwise load every kernel it reaches.
        StartupPhaseScope warm(options.startup_observer, StartupPhase::CudaGraphPrepare);
        instance->executor->warm_up();
        // What startup left free beyond a margin for lazily loaded kernels and the CUDA Graphs
        // the first requests capture becomes expert slots.
        instance->executor->grow_expert_cache(kGrowthMargin);
        warm.complete();
    }
    instance->free_after_weights = free_after_weights;
    instance->free_after_startup = free_bytes();
    std::string table_place      = has_ple ? "off" : "not used (no PLE layers)";
    if (!table.disabled && has_ple) {
        const std::string file = table.path.empty() ? "the artifact" : "its table artifact";
        const std::string io   = table.io == NgramIo::Direct   ? " (direct I/O)"
                                 : table.io == NgramIo::Mapped ? " (mapped)"
                                                               : "";
        const std::string resident =
            std::to_string(instance->executor->ngram_resident_bytes() >> 20U) + " MiB" +
            (table.lock ? ", locked" : "");
        table_place = table.residency == NgramResidency::Ram ? "in RAM (" + resident + ")"
                      : table.residency == NgramResidency::RamHot
                          ? "hot rows in RAM (" + resident + "), the rest read from " + file + io
                          : "read from " + file + io + ", row cache capacity " + resident;
    }
    publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                       "Qwen3.8-Flash-Next: %zu stage(s), experts in %s memory, n-gram table "
                       "%s, state %.0f MiB, workspace %.0f MiB, expert cache %.0f MiB",
                       model->stages().stages(),
                       options.expert_residency == ExpertResidency::Host   ? "host"
                       : options.expert_residency == ExpertResidency::Disk ? "the artifact's files"
                                                                           : "device",
                       table_place.c_str(),
                       double(instance->executor->memory().state_bytes) / 1048576.0,
                       double(instance->executor->memory().workspace_bytes) / 1048576.0,
                       double(instance->executor->memory().expert_cache_bytes) / 1048576.0);
    if (mtp) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                           "Qwen3.8-Flash-Next: MTP speculative decoding, %u drafts a round",
                           speculative.draft_tokens);
        if (speculative.ngram_draft_tokens != 0) {
            publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                               "Qwen3.8-Flash-Next drafts with its MTP block only; n-gram copy "
                               "proposals (--ngram-draft-tokens) are not available for it");
        }
    }
    if (table.disabled && has_ple) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning,
                           "Qwen3.8-Flash-Next runs WITHOUT its n-gram table (--no-ngram-table): "
                           "a non-standard experimental mode. The model was trained with the "
                           "table and degrades badly without it (WikiText-2 perplexity 2.66 -> "
                           "5.01 on GSQ-RCO Q2_0); use it only for experiments");
    }

    ConstructedQwen4Exp out;
    const auto& stats     = model->storage_stats();
    out.load.architecture = std::string(models::architecture_name(models::Architecture::Qwen4Exp));
    out.load.model_name   = model->info().name;
    out.load.prefill_signature = "qwen4_exp";
    std::set<std::string> formats;
    for (const auto& weight : model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    out.load.weight_formats.assign(formats.begin(), formats.end());
    out.load.load_seconds             = seconds(start, Clock::now());
    out.load.upload_seconds           = stats.upload_seconds;
    out.load.artifact_bytes_read      = stats.read_bytes;
    out.load.host_to_device_bytes     = stats.h2d_bytes;
    out.load.peak_staging_bytes       = stats.peak_staging_bytes;
    out.load.pinned_weight_bytes      = stats.pinned_bytes;
    out.load.device_object_count      = stats.device_object_count;
    out.load.host_object_count        = stats.host_object_count;
    const auto& config                = model->config();
    out.model_metadata.model_id       = out.load.model_name;
    out.model_metadata.vocab_size     = config.vocab_size;
    out.model_metadata.embedding_size = config.hidden_size;
    out.model_metadata.native_context = config.max_position_embeddings;
    std::set<std::string> tensor_formats;
    for (const auto& object : reader.directory().objects) {
        const auto* tensor = std::get_if<artifact::TensorObject>(&object);
        if (tensor == nullptr) { continue; }
        std::uint64_t elements = 1;
        for (const auto dimension : tensor->shape) { elements *= dimension; }
        out.model_metadata.parameters += elements;
        out.model_metadata.weight_bytes += tensor->bytes;
        tensor_formats.emplace(tensor->format);
    }
    for (const auto& format : tensor_formats) {
        if (!out.model_metadata.weights_id.empty()) { out.model_metadata.weights_id += "+"; }
        out.model_metadata.weights_id += format;
    }
    instance->model = std::move(model);
    out.instance    = std::move(instance);
    return out;
}

struct Qwen4ExpCore::Request {
    std::uint64_t id = 0;
    models::qwen3_5::PreparedPrompt prompt;
    std::vector<TokenId> prompt_tokens;
    models::qwen3_5::OutputSession output;
    PromptSummary prompt_summary;
    double prepare_seconds = 0.0;
    double vision_seconds  = 0.0; // the Vision tower's run over the prompt's media
    ResolvedRequestOptions options;
    OutputConsumerMode consumer_mode = OutputConsumerMode::Aggregate;
    GenerationObservationOptions observation;
    Clock::time_point deadline;
    Clock::time_point submitted;
    std::atomic<bool> cancelled{false};

    std::mutex mutex;
    std::condition_variable cv;
    std::optional<GenerationStart> stream_start;
    std::optional<PromptProgress> stream_progress;
    std::vector<std::variant<OutputDelta, GenerationTimingObservation>> events;
    bool response_done = false;
    std::exception_ptr error;
    GenerationResult result;
    std::string content, reasoning;
    std::vector<TokenLogprob> content_logprobs;

    // Where the context cache snapshots the sequence: the prompt's turn closure (the end of the
    // last user turn, which the next turn's prompt repeats although it renders this turn's answer
    // differently), else the prompt's end; and whether the prompt may be reused at all.
    std::uint32_t anchor_at = 0;
    bool reusable           = true;
    bool media              = false; // images or video: encoded at the prompt's first chunk

    // The worker's progress with an admitted request.
    std::uint32_t slot      = 0;
    std::uint32_t reused    = 0; // prompt tokens its sequence already held
    PrefixReusePath reuse_path = PrefixReusePath::Root; // where they came from
    std::uint32_t prefilled = 0; // prompt tokens its sequence holds, the reused ones included
    std::uint32_t prefill_skips = 0; // prefill steps other prompts ran since its last one
    bool decoding           = false;
    bool penalties          = false;
    bool post_thinking      = false;
    std::uint32_t position  = 0; // tokens fed: the logical position of the next sampled token
    std::vector<TokenId> feed;   // what the next decode step feeds
    std::vector<TokenId> generated;
    std::optional<GenerationBudget> budget;
    Clock::time_point admitted, prefill_start, prefill_end, first_token, last_token;
    SpeculativeStats speculative; // MTP rounds, an Engine with drafts only
    runtime::MtpAdaptiveSignal mtp_signal; // how far its drafts survive, with --adaptive-mtp
};

struct Qwen4ExpCore::Impl {
    // One executor sequence, the request it serves and what the context cache keeps of it: the
    // tokens its live state holds, and a snapshot of its state at the end of the last prompt it
    // prefilled, which a later prompt that starts with that one resumes from.
    struct Slot {
        std::shared_ptr<Request> request;
        std::vector<TokenId> fed;
        std::vector<TokenId> anchor;
        models::qwen4_exp::SequenceSnapshot snapshot;
        std::uint64_t last_used = 0;
    };

    // One row of a sampling call: the request's parameters at its position.
    struct SampleRow {
        const ResolvedSamplingParameters* params = nullptr;
        std::uint32_t position                   = 0;
        std::uint32_t slot                       = 0;
        bool counts                              = false;
        bool logprobs                            = false;
        const text::GrammarState* grammar        = nullptr;
    };

    Qwen4ExpInstance& instance;
    DeviceContext& device;
    const std::uint32_t max_context;
    const std::size_t max_outstanding;
    const std::chrono::milliseconds pending_timeout;
    const std::uint32_t domain;
    const bool structured_output;
    const bool reuse_prefixes;
    const std::uint32_t drafts; // MTP drafts a speculative round proposes; 0 without speculation
    const std::uint32_t lookup_ngram; // --lookup-ngram: tokens a context lookup matches; 0 is off
    // With --adaptive-mtp, the steps a round drafts and verifies: from one draft, since each draft
    // is a step of the MTP block of its own here, up to `drafts`.
    std::optional<runtime::MtpAdaptiveBatchController> mtp_controller;

    mutable std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::deque<std::shared_ptr<Request>> pending;
    std::size_t outstanding = 0;
    bool stopping           = false;
    bool failed             = false;
    std::uint64_t next_id   = 1;

    mutable std::mutex stats_mutex;
    RuntimeStats stats;

    // Worker-owned.
    std::vector<Slot> slots; // by executor sequence
    std::uint64_t use_clock = 0;
    // Decode rounds still owed after a prefill chunk before the next chunk may run.
    std::uint32_t decode_rounds_due = 0;
    std::uint32_t decode_rounds_per_prefill = 1;

    // Head-device sampling planes, a row per slot; with structured output the grammars' token
    // bitmasks too. The host side is staged in pinned memory: configs, positions, sampled tokens.
    DeviceBuffer sample_config, sample_position, sample_out, token_counts, score_targets, score_out;
    DeviceBuffer token_mask;
    // The logprob gather of a sampling call whose rows ask for it: [kLogprobTopK, rows] ids and
    // values, [rows] log-sum-exps and the gather's enabling flag, with the host copy.
    DeviceBuffer logprob_ids, logprob_values, logprob_lse, logprob_flag;
    std::unique_ptr<PinnedHostBuffer> host_mask, host_sample, host_logprobs;
    std::unique_ptr<WorkspaceArena> sample_workspace;
    // A speculative round's acceptance on the head device, a row per verified sequence: its
    // drafts, the verification's greedy targets, the live draft count and the tokens fed before
    // the round, and the licensed run (tokens, count, accepted drafts, last token); and every
    // slot's token counts before the round, which a round that keeps fewer tokens than it
    // licensed restores.
    DeviceBuffer spec_drafts, spec_targets, spec_extents, spec_lengths, spec_licensed, spec_counts,
        spec_accepted, spec_anchors, counts_backup;
    std::unique_ptr<PinnedHostBuffer> host_spec;

    // The context cache past the sequences: images of the prefixes they give up (the state a
    // turn closed at, and where a sequence ended), in pinned host memory up to a budget; the least
    // recently used go to files under the disk path when it is set (where a restart finds them)
    // or are dropped. A prompt resumes from the longest stored prefix when no free sequence
    // serves more of it.
    struct Stored {
        std::vector<TokenId> tokens;
        models::qwen4_exp::SequenceImage header;
        PrefixReusePath path = PrefixReusePath::PrivateEndpoint;
        std::unique_ptr<PinnedHostBuffer> host; // null while only on disk
        std::filesystem::path file;             // empty unless on disk
        std::uint64_t bytes     = 0;
        std::uint64_t last_used = 0;
    };

    // Shorter prefixes prefill faster than their recurrent state (about 114 MB) copies back.
    static constexpr std::size_t kMinStoredTokens = 128;
    std::vector<Stored> stored;
    std::uint64_t host_budget = 0, host_used = 0;
    std::filesystem::path disk_dir;
    std::uint64_t disk_budget = 0, disk_used = 0;
    bool disk_restore = false;

    std::thread worker;

    Impl(Qwen4ExpInstance& i, DeviceContext& d, const EngineOptions& options)
        : instance(i), device(d), max_context(options.max_context),
          max_outstanding(std::size_t(options.max_concurrency) + options.max_pending_requests),
          pending_timeout(options.pending_timeout_ms),
          domain(i.model->resources().public_token_count),
          structured_output(options.structured_output),
          reuse_prefixes(options.context_cache.enabled), drafts(i.executor->draft_tokens()),
          lookup_ngram(options.speculative.lookup_ngram),
          slots(i.executor->options().sequences),
          decode_rounds_per_prefill(options.decode_rounds_per_prefill != 0
                                        ? options.decode_rounds_per_prefill
                                        : std::max<std::uint32_t>(
                                              1, i.executor->options().prefill_chunk / 64)) {
        if (drafts > 0 && options.speculative.mtp_policy == MtpDraftPolicy::Adaptive) {
            mtp_controller.emplace();
            mtp_controller->reset(drafts, 1);
        }
        RankBinding bind(device, i.executor->head_rank());
        const std::size_t rows = slots.size();
        // A sampling call's columns per row: one, or a verification's drafts and bonus.
        const std::size_t width = std::size_t(drafts) + 1;
        if (structured_output) {
            token_mask = DeviceBuffer(rows * width * mask_words() * sizeof(std::uint32_t));
            host_mask  = std::make_unique<PinnedHostBuffer>(rows * width * mask_words() *
                                                            sizeof(std::uint32_t));
        }
        sample_config   = DeviceBuffer(rows * sizeof(ops::SamplingConfig));
        sample_position = DeviceBuffer(rows * sizeof(std::int32_t));
        sample_out      = DeviceBuffer(rows * sizeof(std::int32_t));
        host_sample     = std::make_unique<PinnedHostBuffer>(
            rows * (sizeof(ops::SamplingConfig) + 2 * sizeof(std::int32_t)));
        token_counts     = DeviceBuffer(rows * domain * sizeof(std::int32_t));
        score_targets    = DeviceBuffer(4096 * sizeof(std::int32_t));
        score_out        = DeviceBuffer(4096 * sizeof(float));
        const std::size_t top = rows * width * kMaximumTokenLogprobs;
        logprob_ids           = DeviceBuffer(top * sizeof(std::int32_t));
        logprob_values        = DeviceBuffer(top * sizeof(float));
        logprob_lse           = DeviceBuffer(rows * width * sizeof(float));
        logprob_flag          = DeviceBuffer(sizeof(std::int32_t));
        host_logprobs =
            std::make_unique<PinnedHostBuffer>(top * (sizeof(std::int32_t) + sizeof(float)));
        const std::int32_t enabled = 1;
        CUDA_CHECK(cudaMemcpy(logprob_flag.p, &enabled, sizeof(enabled), cudaMemcpyHostToDevice));
        std::size_t workspace = std::max<std::size_t>(
            {ops::sampling_workspace_capacity_bytes(static_cast<std::int32_t>(domain), 1,
                                                    static_cast<std::int32_t>(rows)),
             ops::logprob_topk_workspace_capacity_bytes(static_cast<std::int32_t>(domain),
                                                        static_cast<std::int32_t>(rows * width)),
             std::size_t{256}});
        if (drafts > 0) {
            workspace =
                std::max(workspace, ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                                        static_cast<std::int32_t>(domain), 1,
                                        static_cast<std::int32_t>(drafts), 1,
                                        static_cast<std::int32_t>(rows)));
            const auto plane = [&](std::size_t count) {
                return DeviceBuffer(rows * count * sizeof(std::int32_t));
            };
            spec_drafts   = plane(drafts);
            spec_targets  = plane(width);
            spec_licensed = plane(width);
            spec_extents  = plane(1);
            spec_lengths  = plane(1);
            spec_counts   = plane(1);
            spec_accepted = plane(1);
            spec_anchors  = plane(1);
            counts_backup = DeviceBuffer(rows * domain * sizeof(std::int32_t));
            host_spec     = std::make_unique<PinnedHostBuffer>(rows * (drafts + width + 4) *
                                                               sizeof(std::int32_t));
        }
        sample_workspace                 = std::make_unique<WorkspaceArena>(workspace);
        const ContextCacheOptions& cache = options.context_cache;
        if (reuse_prefixes) {
            // An automatic budget is sized here, once the experts are pinned.
            const ContextCacheOptions sized = resolve_auto_host_cache_now(
                cache, options.enable_vision ? std::uint64_t(options.media_cache_bytes) +
                                                   options.media_live_bytes
                                             : 0U);
            host_budget = sized.host_cache_budget_bytes.value_or(cache.host_kv_capacity_bytes);
            if (!cache.disk_kv_path.empty()) {
                disk_dir     = cache.disk_kv_path / profile_name(options);
                disk_budget  = cache.disk_kv_capacity_bytes != 0 ? cache.disk_kv_capacity_bytes
                                                                 : (64ULL << 30);
                disk_restore = cache.disk_kv_restore;
                std::filesystem::create_directories(disk_dir);
                if (disk_restore) { scan_disk(); }
            }
        }
        worker = std::thread([this] {
            device.bind_to_current_thread();
            loop();
        });
    }

    ~Impl() {
        {
            std::lock_guard lock(queue_mutex);
            stopping = true;
        }
        queue_cv.notify_all();
        if (worker.joinable()) { worker.join(); }
    }

    [[nodiscard]] std::size_t mask_words() const noexcept {
        return (std::size_t(domain) + 31) / 32;
    }

    void complete(const std::shared_ptr<Request>& request, std::exception_ptr error) {
        {
            std::lock_guard lock(request->mutex);
            if (!request->response_done) {
                request->error         = std::move(error);
                request->response_done = true;
            }
        }
        request->cv.notify_all();
        std::lock_guard lock(queue_mutex);
        --outstanding;
    }

    void complete(const std::shared_ptr<Request>& request, GenerationResult result) {
        {
            std::lock_guard lock(request->mutex);
            if (!request->response_done) {
                request->result        = std::move(result);
                request->response_done = true;
            }
        }
        request->cv.notify_all();
        std::lock_guard lock(queue_mutex);
        --outstanding;
    }

    [[nodiscard]] bool any_active() const {
        return std::any_of(slots.begin(), slots.end(),
                           [](const Slot& slot) { return slot.request != nullptr; });
    }

    void fail_engine(std::exception_ptr error) {
        {
            std::lock_guard lock(queue_mutex);
            failed = true;
            stopping = true;
        }
        for (auto& slot : slots) {
            if (slot.request) { complete(std::exchange(slot.request, {}), error); }
        }
        queue_cv.notify_all();
    }

    void loop() {
        for (;;) {
            {
                std::unique_lock lock(queue_mutex);
                queue_cv.wait(lock, [&] { return stopping || !pending.empty() || any_active(); });
                if (stopping) {
                    auto waiting = std::move(pending);
                    pending.clear();
                    lock.unlock();
                    const auto unavailable = std::make_exception_ptr(RequestError(
                        RequestErrorKind::Unavailable, "inference engine is stopping"));
                    for (auto& item : waiting) { complete(item, unavailable); }
                    for (auto& slot : slots) {
                        if (slot.request) { complete(std::exchange(slot.request, {}), unavailable); }
                    }
                    publish_stats();
                    return;
                }
            }
            try {
                admit();
                step();
            } catch (const ops::CpuExpertCancelled&) {
                // Native hybrid execution is C1 and has drained every CPU task before throwing.
                // A partially advanced layer stack is discarded; it must never enter the prefix cache.
                try {
                    auto request = slots.front().request;
                    slots.front().fed.clear();
                    slots.front().anchor.clear();
                    instance.executor->abort(0);
                    // MTP can finish the response before its catch-up observes cancellation.
                    // The sequence still needs cleanup, but no second terminal.
                    if (request) { finish_now(request, FinishReason::Cancelled); }
                } catch (...) { fail_engine(std::current_exception()); }
            } catch (...) {
                // Not a request's own error: the device may be in an unknown state, so the Engine
                // fails with every request it holds.
                fail_engine(std::current_exception());
            }
            publish_stats();
        }
    }

    // The queue depth, from a submitting thread.
    void publish_queue() {
        std::lock_guard lock(queue_mutex);
        std::lock_guard stats_lock(stats_mutex);
        stats.waiting_requests = static_cast<std::uint32_t>(pending.size());
    }

    // Every request gauge, from the worker, which owns the slots and the executor.
    void publish_stats() {
        const NgramTableStats ngram = instance.executor->ngram_stats();
        const auto cache            = instance.executor->expert_cache_stats();
        const ExpertResidencyStats experts{.routes            = cache.routes,
                                           .hits              = cache.hits,
                                           .cpu_routes        = cache.cpu_routes,
                                           .admitted          = cache.admitted,
                                           .transferred_bytes = cache.copied_bytes,
                                           .slots             = cache.slots};
        std::uint32_t running = 0, prefilling = 0, decoding = 0;
        for (const Slot& slot : slots) {
            if (!slot.request) { continue; }
            ++running;
            ++(slot.request->decoding ? decoding : prefilling);
        }
        std::lock_guard lock(queue_mutex);
        std::lock_guard stats_lock(stats_mutex);
        stats.waiting_requests      = static_cast<std::uint32_t>(pending.size());
        stats.running_requests      = running;
        stats.prefilling_requests   = prefilling;
        stats.decode_ready_requests = decoding;
        stats.ngram_table           = ngram;
        stats.experts               = experts;
    }

    // A request failed on its own: it completes with the error, and its sequence keeps nothing
    // for the context cache, since the sequence may have stopped anywhere.
    void fail(const std::shared_ptr<Request>& request, std::exception_ptr error) {
        if (!request_error(error)) { std::rethrow_exception(error); }
        Slot& slot = slots.at(request->slot);
        if (slot.request == request) {
            slot.request = nullptr;
            slot.fed.clear();
            slot.anchor.clear();
        }
        complete(request, std::move(error));
    }

    // Tokens of `held` a prompt can start from: all of them when they are a strict prefix of the
    // prompt (its last token is always fed, since the first sample needs its logits), else none.
    static std::uint32_t prefix_reuse(const std::vector<TokenId>& held,
                                      const std::vector<TokenId>& prompt) {
        if (held.empty() || held.size() >= prompt.size()) { return 0; }
        return std::equal(held.begin(), held.end(), prompt.begin())
                   ? static_cast<std::uint32_t>(held.size())
                   : 0;
    }

    [[nodiscard]] std::uint32_t reuse(const Slot& slot, const std::vector<TokenId>& prompt) const {
        if (!reuse_prefixes) { return 0; }
        return std::max(prefix_reuse(slot.fed, prompt), prefix_reuse(slot.anchor, prompt));
    }

    // ---- The context cache's store ------------------------------------------------------------

    // The disk tier's directory for this model and execution profile: an image holds the KV in its
    // storage format and the MTP block's state only when the block runs. Verify width changes
    // reductions; disabling PLE changes every layer's recurrent/KV state. Each needs its own store.
    [[nodiscard]] std::string profile_name(const EngineOptions& options) const {
        std::string id;
        for (const std::byte b : instance.model->info().artifact_id) {
            char hex[3];
            std::snprintf(hex, sizeof(hex), "%02x", std::to_integer<unsigned>(b));
            id += hex;
        }
        return "qwen4_exp-image1-" + id + "-kv" +
               std::to_string(static_cast<int>(options.kv_cache)) + "-drafts" +
               std::to_string(drafts) + "-draft-min-p" +
               std::to_string(std::bit_cast<std::uint32_t>(options.speculative.draft_min_p)) + "-ngram" +
               std::to_string(options.ngram_table.disabled ? 0 : 1) + "-experts-" +
               instance.executor->expert_execution_profile();
    }

    struct FileHeader {
        char magic[8]                = {'N', 'F', 'N', 'X', 'I', 'M', 'G', '1'};
        std::uint32_t position       = 0;
        std::uint32_t mtp_follows    = 0;
        std::uint32_t context_tokens = 0;
        std::uint32_t prompt_tokens  = 0;
        std::uint32_t path           = 0;
        std::uint32_t reserved       = 0;
        std::uint64_t bytes          = 0;
    };

    static std::string file_name(const std::vector<TokenId>& tokens) {
        std::uint64_t hash = 1469598103934665603ULL; // FNV-1a over the token ids
        for (const TokenId token : tokens) {
            for (int i = 0; i < 4; ++i) {
                hash ^= (std::uint32_t(token) >> (8 * i)) & 0xffU;
                hash *= 1099511628211ULL;
            }
        }
        char name[40];
        std::snprintf(name, sizeof(name), "%016llx-%u.img", static_cast<unsigned long long>(hash),
                      static_cast<unsigned>(tokens.size()));
        return name;
    }

    // Reads the headers of the images a previous run left in the disk tier.
    void scan_disk() {
        std::uint64_t order = 0;
        for (const auto& entry : std::filesystem::directory_iterator(disk_dir)) {
            if (entry.path().extension() != ".img") { continue; }
            std::ifstream file(entry.path(), std::ios::binary);
            FileHeader header;
            if (!file.read(reinterpret_cast<char*>(&header), sizeof(header)) ||
                std::memcmp(header.magic, FileHeader{}.magic, 8) != 0 ||
                header.bytes != instance.executor->image_bytes(header.position)) {
                continue;
            }
            Stored image;
            image.header.position    = header.position;
            image.header.mtp_follows = header.mtp_follows != 0;
            image.header.context.previous.resize(header.context_tokens);
            image.tokens.resize(header.prompt_tokens);
            if (!file.read(reinterpret_cast<char*>(image.header.context.previous.data()),
                           std::streamsize(header.context_tokens) * 4) ||
                !file.read(reinterpret_cast<char*>(image.tokens.data()),
                           std::streamsize(header.prompt_tokens) * 4)) {
                continue;
            }
            image.path      = static_cast<PrefixReusePath>(header.path);
            image.file      = entry.path();
            image.bytes     = header.bytes;
            image.last_used = ++order;
            disk_used += entry.file_size();
            stored.push_back(std::move(image));
        }
    }

    // The stored prefix that serves the most of `prompt`: its index and length.
    [[nodiscard]] std::pair<std::size_t, std::uint32_t>
    find_stored(const std::vector<TokenId>& prompt) const {
        std::pair<std::size_t, std::uint32_t> best{0, 0};
        for (std::size_t i = 0; i < stored.size(); ++i) {
            if (stored[i].host == nullptr && !disk_restore) { continue; }
            const std::uint32_t length = prefix_reuse(stored[i].tokens, prompt);
            if (length > best.second) { best = {i, length}; }
        }
        return best;
    }

    void drop_dead() {
        std::erase_if(stored, [](const Stored& image) {
            return image.host == nullptr && image.file.empty();
        });
    }

    // Writes an image held in host memory to the disk tier, making room by deleting the least
    // recently used files; entries left with neither copy are dropped by the caller.
    void write_file(Stored& image) {
        if (disk_dir.empty() || !image.file.empty() || image.host == nullptr) { return; }
        const std::uint64_t size =
            sizeof(FileHeader) + 4 * (image.header.context.previous.size() + image.tokens.size()) +
            image.bytes;
        if (size > disk_budget) { return; }
        while (disk_used + size > disk_budget) {
            Stored* oldest = nullptr;
            for (Stored& other : stored) {
                if (!other.file.empty() &&
                    (oldest == nullptr || other.last_used < oldest->last_used)) {
                    oldest = &other;
                }
            }
            if (oldest == nullptr) { return; }
            std::error_code ignored;
            const auto bytes = std::filesystem::file_size(oldest->file, ignored);
            disk_used -= std::min<std::uint64_t>(disk_used, ignored ? 0 : bytes);
            std::filesystem::remove(oldest->file, ignored);
            oldest->file.clear();
        }
        FileHeader header;
        header.position       = image.header.position;
        header.mtp_follows    = image.header.mtp_follows ? 1 : 0;
        header.context_tokens = static_cast<std::uint32_t>(image.header.context.previous.size());
        header.prompt_tokens  = static_cast<std::uint32_t>(image.tokens.size());
        header.path           = static_cast<std::uint32_t>(image.path);
        header.bytes          = image.bytes;
        const auto path       = disk_dir / file_name(image.tokens);
        const auto temporary  = std::filesystem::path(path).concat(".tmp");
        {
            std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
            file.write(reinterpret_cast<const char*>(&header), sizeof(header));
            file.write(reinterpret_cast<const char*>(image.header.context.previous.data()),
                       std::streamsize(header.context_tokens) * 4);
            file.write(reinterpret_cast<const char*>(image.tokens.data()),
                       std::streamsize(header.prompt_tokens) * 4);
            file.write(static_cast<const char*>(image.host->data()), std::streamsize(image.bytes));
            if (!file) {
                std::error_code ignored;
                std::filesystem::remove(temporary, ignored);
                return;
            }
        }
        std::error_code failed;
        std::filesystem::rename(temporary, path, failed);
        if (failed) { return; }
        image.file = path;
        disk_used += size;
    }

    // Brings the images in pinned host memory within the budget with `bytes` more: the least
    // recently used go to the disk tier, or are dropped without one. False when the budget
    // cannot hold `bytes` at all.
    bool make_host_room(std::uint64_t bytes) {
        if (bytes > host_budget) { return false; }
        while (host_used + bytes > host_budget) {
            std::optional<std::size_t> oldest;
            for (std::size_t i = 0; i < stored.size(); ++i) {
                if (stored[i].host != nullptr &&
                    (!oldest || stored[i].last_used < stored[*oldest].last_used)) {
                    oldest = i;
                }
            }
            if (!oldest) { return false; }
            write_file(stored[*oldest]);
            stored[*oldest].host.reset();
            host_used -= stored[*oldest].bytes;
            drop_dead();
        }
        drop_dead();
        return true;
    }

    [[nodiscard]] std::optional<std::size_t>
    stored_index(const std::vector<TokenId>& tokens) const {
        for (std::size_t i = 0; i < stored.size(); ++i) {
            if (stored[i].tokens == tokens) { return i; }
        }
        return std::nullopt;
    }

    // Keeps the state of sequence `s` up to `tokens` (its live state, or the snapshot `at`) in
    // the store, unless the store already holds that prefix or it is too short to be worth it.
    void store_image(std::uint32_t s, const std::vector<TokenId>& tokens,
                     const models::qwen4_exp::SequenceSnapshot* at, PrefixReusePath path) {
        if (tokens.size() < kMinStoredTokens || host_budget == 0) { return; }
        if (const auto known = stored_index(tokens)) {
            stored[*known].last_used = ++use_clock;
            return;
        }
        auto& executor      = *instance.executor;
        const auto position = at != nullptr ? at->position : executor.position(s);
        if (position != tokens.size()) { return; } // the state holds other tokens
        const std::uint64_t bytes = executor.image_bytes(position);
        if (!make_host_room(bytes)) { return; }
        Stored image;
        image.tokens = tokens;
        image.path   = path;
        image.bytes  = bytes;
        image.host   = std::make_unique<PinnedHostBuffer>(bytes);
        image.header = executor.save_image(
            s, at, std::span(static_cast<std::byte*>(image.host->data()), bytes));
        image.last_used = ++use_clock;
        host_used += bytes;
        stored.push_back(std::move(image));
        std::lock_guard lock(stats_mutex);
        stats.state_d2h_count += 1;
        stats.state_d2h_bytes += bytes;
        stats.host_kv_occupied_bytes = host_used;
    }

    // Keeps what sequence `s` holds before its state goes: the prefix its last prompt's turn
    // closed at, and where it ended.
    void retire(std::uint32_t s) {
        Slot& slot          = slots[s];
        const bool anchored = !slot.anchor.empty() && slot.anchor.size() <= slot.fed.size() &&
                              std::equal(slot.anchor.begin(), slot.anchor.end(), slot.fed.begin());
        if (anchored) {
            store_image(s, slot.anchor, &slot.snapshot, PrefixReusePath::PrivateTurnClosure);
        }
        if (slot.fed != slot.anchor) {
            store_image(s, slot.fed, nullptr, PrefixReusePath::PrivateEndpoint);
        }
    }

    // Loads the stored image of `tokens` into sequence `s`, reading it from the disk tier when it
    // is only there; false when the store no longer holds it.
    bool restore_stored(std::uint32_t s, const std::vector<TokenId>& tokens) {
        const auto index = stored_index(tokens);
        if (!index) { return false; }
        Stored& image = stored[*index];
        if (image.host == nullptr) {
            image.host = std::make_unique<PinnedHostBuffer>(image.bytes);
            std::ifstream in(image.file, std::ios::binary);
            in.seekg(std::streamoff(sizeof(FileHeader) +
                                    4 * (image.header.context.previous.size() + tokens.size())));
            if (!in.read(static_cast<char*>(image.host->data()), std::streamsize(image.bytes))) {
                image.host.reset();
                return false;
            }
            host_used += image.bytes;
        }
        instance.executor->load_image(
            s, image.header,
            std::span(static_cast<const std::byte*>(image.host->data()), image.bytes));
        image.last_used           = ++use_clock;
        const std::uint64_t bytes = image.bytes;
        // Within the budget again, the image just read in being the most recently used.
        (void)make_host_room(0);
        std::lock_guard lock(stats_mutex);
        stats.state_h2d_count += 1;
        stats.state_h2d_bytes += bytes;
        stats.host_kv_occupied_bytes = host_used;
        return true;
    }

    // Moves queued requests into free sequences, oldest first: each into the free sequence whose
    // cached state serves the most of its prompt, else the least recently used.
    void admit() {
        for (;;) {
            std::optional<std::uint32_t> best;
            std::shared_ptr<Request> request;
            {
                std::lock_guard lock(queue_mutex);
                if (pending.empty()) { return; }
                for (std::uint32_t s = 0; s < slots.size(); ++s) {
                    if (slots[s].request) { continue; }
                    if (!best) {
                        best = s;
                        continue;
                    }
                    const auto& prompt   = pending.front()->prompt_tokens;
                    const auto candidate = reuse(slots[s], prompt), current = reuse(slots[*best], prompt);
                    if (candidate > current ||
                        (candidate == current && slots[s].last_used < slots[*best].last_used)) {
                        best = s;
                    }
                }
                if (!best) { return; }
                request = std::move(pending.front());
                pending.pop_front();
            }
            if (Clock::now() > request->deadline) {
                complete(request, std::make_exception_ptr(
                                      RequestError(RequestErrorKind::QueueTimeout,
                                                   "inference request expired in the queue")));
                continue;
            }
            try {
                begin(request, *best);
            } catch (...) { fail(request, std::current_exception()); }
        }
    }

    void begin(const std::shared_ptr<Request>& request, std::uint32_t s) {
        Request& r          = *request;
        Slot& slot          = slots[s];
        auto& executor      = *instance.executor;
        const auto prompt_n = static_cast<std::uint32_t>(r.prompt_tokens.size());
        r.slot              = s;
        if (prompt_n == 0) { throw std::invalid_argument("prepared prompt is empty"); }
        if (prompt_n > max_context) {
            throw RequestError(RequestErrorKind::ContextLengthExceeded,
                               "prepared prompt exceeds Engine max_context");
        }
        r.admitted = Clock::now();
        // Resume from the sequence's live state when the prompt continues it, else from the
        // snapshot at the end of its last prompt when the prompt continues that, else from the
        // longest prefix the store keeps; what the sequence held goes to the store first.
        std::uint32_t reused = 0;
        if (reuse_prefixes && r.reusable) {
            const std::uint32_t live       = prefix_reuse(slot.fed, r.prompt_tokens);
            const std::uint32_t anchored   = prefix_reuse(slot.anchor, r.prompt_tokens);
            const auto [image, from_store] = find_stored(r.prompt_tokens);
            if (from_store > std::max(live, anchored)) {
                const std::vector<TokenId> tokens = stored[image].tokens;
                const PrefixReusePath path        = stored[image].path;
                retire(s);
                if (restore_stored(s, tokens)) {
                    slot.fed = tokens;
                    slot.anchor.clear();
                    reused       = from_store;
                    r.reuse_path = path;
                }
            } else if (anchored > live) {
                retire(s);
                executor.restore(s, slot.snapshot);
                slot.fed     = slot.anchor;
                reused       = anchored;
                r.reuse_path = PrefixReusePath::PrivateTurnClosure;
            } else if (live > 0) {
                reused       = live;
                r.reuse_path = PrefixReusePath::PrivateEndpoint;
            }
        }
        if (reused == 0) {
            if (reuse_prefixes) { retire(s); }
            executor.reset(s);
            slot.fed.clear();
            // The snapshot's prefix is about to be overwritten.
            slot.anchor.clear();
        }
        slot.request   = request;
        slot.last_used = ++use_clock;
        r.reused = r.prefilled = reused;
        if (drafts > 0) {
            r.speculative.backend      = SpeculativeBackend::Mtp;
            r.speculative.enabled      = true;
            r.speculative.draft_window = drafts;
            r.speculative.accepted_per_position.assign(drafts, 0);
            if (mtp_controller) {
                r.speculative.adaptive = true;
                r.speculative.rounds_per_window.assign(drafts, 0);
                r.mtp_signal.reset();
            }
        }
        r.prefill_start        = Clock::now();
        {
            std::lock_guard lock(stats_mutex);
            stats.reused_prompt_tokens += reused;
        }
        if (r.consumer_mode == OutputConsumerMode::Streaming) {
            {
                std::lock_guard lock(r.mutex);
                r.stream_start =
                    GenerationStart{.prompt = r.prompt_summary, .reused_prompt_tokens = reused};
            }
            r.cv.notify_all();
        }
    }

    // Whether `a` runs the next prefill step before `b`. A media prompt that has begun goes on
    // first, since the executor keeps one prompt's media embeddings; then a prompt passed over
    // kPrefillMaxSkip times (the most passed over first); then the shortest remaining prompt, so
    // a short or cached request is not held behind a long prompt whose chunks take seconds with
    // the experts off the GPU. Ties go to the earlier request.
    static bool prefills_before(const Request& a, const Request& b) {
        const auto media_begun = [](const Request& r) {
            return r.media && r.prefilled > r.reused;
        };
        if (media_begun(a) != media_begun(b)) { return media_begun(a); }
        const bool a_starved = a.prefill_skips >= kPrefillMaxSkip;
        const bool b_starved = b.prefill_skips >= kPrefillMaxSkip;
        if (a_starved != b_starved) { return a_starved; }
        if (a_starved && a.prefill_skips != b.prefill_skips) {
            return a.prefill_skips > b.prefill_skips;
        }
        const auto left = [](const Request& r) {
            const std::size_t n = r.prompt_tokens.size();
            return n - std::min<std::size_t>(r.prefilled, n);
        };
        if (!a_starved && left(a) != left(b)) { return left(a) < left(b); }
        return a.id < b.id;
    }

    void step() {
        auto& executor = *instance.executor;
        const auto cancellation_owner = executor.hybrid_experts() ? slots.front().request : nullptr;
        struct CancellationBorrow {
            models::qwen4_exp::Executor& executor;
            ~CancellationBorrow() { executor.bind_cancellation(nullptr); }
        } cancellation_borrow{executor};
        executor.bind_cancellation(cancellation_owner ? &cancellation_owner->cancelled : nullptr);
        std::shared_ptr<Request> prefilling;
        bool decoding = false;
        for (const Slot& slot : slots) {
            if (!slot.request) { continue; }
            if (slot.request->decoding) {
                decoding = true;
            } else if (!prefilling || prefills_before(*slot.request, *prefilling)) {
                prefilling = slot.request;
            }
        }
        if (prefilling && (decode_rounds_due == 0 || !decoding)) {
            // A long prompt's chunk takes far longer than a decode round: the streams that are
            // generating get several rounds after it rather than one.
            decode_rounds_due = decode_rounds_per_prefill;
            for (const Slot& slot : slots) {
                if (slot.request && !slot.request->decoding && slot.request != prefilling &&
                    slot.request->prefill_skips != UINT32_MAX) {
                    ++slot.request->prefill_skips;
                }
            }
            prefilling->prefill_skips = 0;
            try {
                prefill_step(prefilling);
            } catch (...) { fail(prefilling, std::current_exception()); }
        } else if (decoding) {
            if (decode_rounds_due > 0) { --decode_rounds_due; }
            decode_step();
        }
    }

    // Publishes a committed preview with the logprob records it released, which a streaming
    // request receives on the commit's content delta.
    void push_events(Request& r, models::qwen3_5::PublishedOutput published,
                     std::optional<GenerationTimingObservation> timing) {
        const bool streaming               = r.consumer_mode == OutputConsumerMode::Streaming;
        std::vector<TokenLogprob> logprobs = r.output.take_content_logprobs();
        if (published.empty() && !timing && logprobs.empty()) { return; }
        if (streaming && !logprobs.empty()) {
            OutputDelta* content = nullptr;
            for (OutputDelta& delta : published) {
                if (delta.channel == OutputChannel::Content) { content = &delta; }
            }
            if (content == nullptr) {
                published.push_back(OutputDelta{.channel = OutputChannel::Content});
                content = &published.back();
            }
            content->logprobs = logprobs;
        }
        {
            std::lock_guard lock(r.mutex);
            if (streaming && timing) { r.events.emplace_back(*timing); }
            for (OutputDelta& delta : published) {
                (delta.channel == OutputChannel::Reasoning ? r.reasoning : r.content) += delta.text;
                if (streaming) { r.events.emplace_back(std::move(delta)); }
            }
            r.content_logprobs.insert(r.content_logprobs.end(),
                                      std::make_move_iterator(logprobs.begin()),
                                      std::make_move_iterator(logprobs.end()));
        }
        if (streaming) { r.cv.notify_all(); }
    }

    // Samples one token per row from the first rows.size() columns of the head logits; a grammar
    // restricts its row to the tokens its current state allows. A row that asks for logprobs gets
    // the sampled token's record in `logprobs`, under the distribution its token was drawn from.
    void sample(std::span<const SampleRow> rows, std::int32_t purpose, std::span<TokenId> out,
                std::span<runtime::RawTokenLogprob> logprobs) {
        RankBinding bind(device, instance.executor->head_rank());
        const cudaStream_t stream = instance.executor->head_stream();
        const std::size_t n       = rows.size();
        auto* configs             = static_cast<ops::SamplingConfig*>(host_sample->data());
        auto* positions           = reinterpret_cast<std::int32_t*>(configs + slots.size());
        auto* tokens              = positions + slots.size();
        bool masked               = false;
        for (std::size_t b = 0; b < n; ++b) {
            const SampleRow& row = rows[b];
            ops::SamplingConfig config;
            config.temperature       = row.params->temperature;
            config.top_k             = row.params->top_k;
            config.top_p             = row.params->top_p;
            config.min_p             = row.params->min_p;
            config.presence_penalty  = row.params->presence_penalty;
            config.frequency_penalty = row.params->frequency_penalty;
            config.seed              = row.params->seed;
            config.token_counts =
                row.counts ? static_cast<std::int32_t*>(token_counts.p) + std::size_t(row.slot) * domain
                           : nullptr;
            if (row.grammar != nullptr) {
                auto* words = static_cast<std::uint32_t*>(host_mask->data()) + b * mask_words();
                row.grammar->fill_masks(std::span(words, mask_words()), {});
                config.token_mask =
                    static_cast<const std::uint32_t*>(token_mask.p) + b * mask_words();
                config.token_mask_stride = static_cast<std::int32_t>(mask_words());
                masked                   = true;
            }
            configs[b]   = config;
            positions[b] = static_cast<std::int32_t>(row.position);
        }
        if (masked) {
            CUDA_CHECK(cudaMemcpyAsync(token_mask.p, host_mask->data(),
                                       n * mask_words() * sizeof(std::uint32_t),
                                       cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(sample_config.p, configs, n * sizeof(ops::SamplingConfig),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(sample_position.p, positions, n * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream));
        const auto width = static_cast<std::int32_t>(n);
        Tensor sampled(sample_out.p, DType::I32, {width});
        const Tensor logical(sample_position.p, DType::I32, {width});
        const Tensor logits = instance.executor->logits(static_cast<std::uint32_t>(n));
        const auto* device_configs = static_cast<const ops::SamplingConfig*>(sample_config.p);
        const bool gather = std::any_of(rows.begin(), rows.end(),
                                        [](const SampleRow& row) { return row.logprobs; });
        const std::size_t top = n * kMaximumTokenLogprobs;
        auto* host_ids        = static_cast<std::int32_t*>(host_logprobs->data());
        auto* host_values = reinterpret_cast<float*>(host_ids + slots.size() * kMaximumTokenLogprobs);
        if (gather) {
            // Before sampling adds the tokens to the penalty counts.
            Tensor ids(logprob_ids.p, DType::I32, {ops::kLogprobTopK, 1, width});
            Tensor values(logprob_values.p, DType::FP32, {ops::kLogprobTopK, 1, width});
            Tensor lse(logprob_lse.p, DType::FP32, {1, width});
            const Tensor flag(logprob_flag.p, DType::I32, {1});
            ops::logprob_topk(logits.view({logits.ne[0], 1, width}), device_configs, nullptr,
                              static_cast<std::int32_t>(domain), ids, values, lse, flag,
                              *sample_workspace, stream);
            CUDA_CHECK(cudaMemcpyAsync(host_ids, logprob_ids.p, top * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(host_values, logprob_values.p, top * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
        }
        {
            auto scope = sample_workspace->scope();
            ops::sample(logits, sampled, static_cast<std::int32_t>(domain), device_configs,
                        logical, purpose, *sample_workspace, stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(tokens, sample_out.p, n * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (std::size_t b = 0; b < n; ++b) {
            const std::int32_t token = tokens[b];
            if (token == ops::kSamplerNonFiniteToken || token < 0 ||
                std::uint32_t(token) >= domain) {
                throw std::runtime_error("Qwen3.8-Flash-Next produced non-finite logits");
            }
            out[b] = token;
            if (!rows[b].logprobs) { continue; }
            runtime::RawTokenLogprob& record = logprobs[b];
            record = runtime::RawTokenLogprob{.id = token, .logprob = kLogprobSentinel};
            for (std::size_t k = 0; k < kMaximumTokenLogprobs; ++k) {
                const std::size_t slot = b * kMaximumTokenLogprobs + k;
                record.top_ids[k]      = host_ids[slot];
                record.top_values[k]   = host_values[slot];
                if (host_ids[slot] == token) { record.logprob = host_values[slot]; }
            }
        }
    }

    [[nodiscard]] SampleRow sample_row(Request& r) {
        const auto& exec = r.options.execution;
        if (!r.post_thinking && exec.post_thinking_sampling && r.output.reasoning_closed()) {
            r.post_thinking = true;
        }
        return SampleRow{.params   = r.post_thinking ? &*exec.post_thinking_sampling
                                                     : &exec.sampling,
                         .position = r.position,
                         .slot     = r.slot,
                         .counts   = r.penalties,
                         .logprobs = exec.logprobs,
                         .grammar  = r.output.grammar_state().get()};
    }

    // Frees the request's sequence (keeping its state for the context cache) and completes it.
    void finish(const std::shared_ptr<Request>& request, FinishReason reason) {
        instance.executor->save_expert_profile();
        Request& r = *request;
        GenerationResult result;
        result.prompt              = r.prompt_summary;
        result.generated_token_ids = std::move(r.generated);
        {
            std::lock_guard lock(r.mutex);
            result.content          = std::move(r.content);
            result.reasoning        = std::move(r.reasoning);
            result.content_logprobs = std::move(r.content_logprobs);
        }
        const auto now = Clock::now();
        if (r.first_token == Clock::time_point{}) { r.first_token = r.last_token = now; }
        if (r.prefill_end == Clock::time_point{}) { r.prefill_end = now; }
        result.tool_calls                      = r.output.take_tool_calls();
        result.tool_call_parse                 = r.output.tool_call_parse_diagnostics();
        result.reasoning_tokens                = r.output.reasoning_tokens();
        result.finish_reason                   = reason;
        result.matched_stop_string             = r.output.matched_stop_string();
        result.thinking                        = r.output.thinking_stats();
        result.thinking.post_thinking_sampling = r.post_thinking;
        result.reused_prompt_tokens            = r.reused;
        result.prefix_reuse_path               = r.reuse_path;
        if (drafts > 0) { result.speculative = r.speculative; }
        result.timings.prepare_seconds         = r.prepare_seconds;
        result.timings.vision_seconds          = r.vision_seconds;
        result.timings.prefill_seconds = seconds(r.prefill_start, r.prefill_end) - r.vision_seconds;
        result.timings.decode_seconds          = seconds(r.prefill_end, r.last_token);
        result.timings.first_token_seconds = r.prepare_seconds + seconds(r.submitted, r.first_token);
        if (r.observation.phase_timings) {
            result.timings.prompt_wall_seconds     = seconds(r.admitted, r.first_token);
            result.timings.generation_wall_seconds = seconds(r.first_token, r.last_token);
        }
        result.timings.total_seconds = r.prepare_seconds + seconds(r.submitted, now);
        Slot& slot                   = slots.at(r.slot);
        slot.request                 = nullptr;
        if (r.media) {
            // Its state depends on media its tokens do not identify.
            slot.fed.clear();
            slot.anchor.clear();
        }
        complete(request, std::move(result));
    }

    void finish_now(const std::shared_ptr<Request>& request, FinishReason reason) {
        (void)request->output.preview_terminal(reason);
        push_events(*request, request->output.commit_preview(), std::nullopt);
        finish(request, reason);
    }

    // Feeds the next chunk of a prompt, which ends at the anchor when one lies ahead, where the
    // sequence's state is kept for the context cache; at the prompt's end samples the first token.
    void prefill_step(const std::shared_ptr<Request>& request) {
        Request& r          = *request;
        Slot& slot          = slots[r.slot];
        auto& executor      = *instance.executor;
        const auto prompt_n = static_cast<std::uint32_t>(r.prompt_tokens.size());
        if (r.cancelled.load(std::memory_order_acquire)) {
            finish_now(request, FinishReason::Cancelled);
            return;
        }
        if (r.media && r.prefilled == 0) {
            // The tower's embeddings stay until another prompt's media replace them; prompts
            // prefill one at a time, so this one is done first.
            const auto& data = models::qwen3_5::PreparedPromptAccess::view(r.prompt);
            const auto& vision = *instance.model->vision_config();
            const auto control = models::qwen3_5::build_vision_control(
                data, models::qwen3_5::plan_vision_control(data, vision), 0);
            std::vector<models::qwen4_exp::MediaItem> items;
            for (std::size_t i = 0; i < control.items.size(); ++i) {
                items.push_back({.patches = data.media_payloads.at(i)->span(),
                                 .control = &control.items[i]});
            }
            const auto vision_start = Clock::now();
            executor.set_media(r.slot, items, data.positions, data.rope_delta);
            r.vision_seconds = seconds(vision_start, Clock::now());
        }
        const std::uint32_t until = r.prefilled < r.anchor_at ? r.anchor_at : prompt_n;
        // A media prompt goes chunk by chunk; a text prompt in spans where the executor has them.
        // A span is one step, which with the experts off the GPU lasts seconds, and the requests
        // that are decoding wait for all of it: while any decodes, a step is two chunks at most.
        const std::uint32_t chunk_tokens = executor.options().prefill_chunk;
        std::uint32_t step               = r.media ? chunk_tokens : executor.prompt_step();
        const bool peers_decoding =
            std::any_of(slots.begin(), slots.end(), [&](const Slot& other) {
                return other.request && other.request.get() != &r && other.request->decoding;
            });
        if (peers_decoding) { step = std::min(step, kPeerSpanChunks * chunk_tokens); }
        const std::uint32_t n = std::min(step, until - r.prefilled);
        const auto chunk = std::span<const TokenId>(r.prompt_tokens).subspan(r.prefilled, n);
        executor.forward(r.slot, chunk, 1);
        slot.fed.insert(slot.fed.end(), chunk.begin(), chunk.end());
        r.prefilled += n;
        if (r.prefilled < prompt_n) {
            const auto next = std::span<const TokenId>(r.prompt_tokens).subspan(
                r.prefilled, std::min(executor.options().prefill_chunk, prompt_n - r.prefilled));
            executor.prefetch_ngram(r.slot, next);
        }
        if (reuse_prefixes && r.reusable && r.prefilled == r.anchor_at) {
            executor.snapshot(r.slot, slot.snapshot);
            slot.anchor = slot.fed;
        }
        if (r.observation.prompt_progress) {
            CUDA_CHECK(cudaStreamSynchronize(executor.head_stream()));
            {
                std::lock_guard lock(r.mutex);
                r.stream_progress = PromptProgress{.total_prompt_tokens     = prompt_n,
                                                   .reused_prompt_tokens    = r.reused,
                                                   .processed_prompt_tokens = r.prefilled,
                                                   .elapsed_ns = elapsed_ns(r.admitted, Clock::now())};
            }
            r.cv.notify_all();
        }
        if (r.prefilled < prompt_n) { return; }
        CUDA_CHECK(cudaStreamSynchronize(executor.head_stream()));
        r.prefill_end = Clock::now();
        {
            std::lock_guard lock(stats_mutex);
            stats.computed_prefill_tokens += prompt_n - r.reused;
            stats.prefill_seconds_total += seconds(r.prefill_start, r.prefill_end);
        }
        const auto& exec = r.options.execution;
        const std::uint32_t capacity =
            effective_output_capacity(exec.requested_output_tokens, max_context, prompt_n);
        r.budget.emplace(capacity, exec.requested_output_tokens <= capacity
                                       ? FinishReason::OutputLimit
                                       : FinishReason::ContextCapacity);
        r.penalties = exec.sampling.presence_penalty != 0.0F ||
                      exec.sampling.frequency_penalty != 0.0F ||
                      (exec.post_thinking_sampling &&
                       (exec.post_thinking_sampling->presence_penalty != 0.0F ||
                        exec.post_thinking_sampling->frequency_penalty != 0.0F));
        if (r.penalties) {
            RankBinding bind(device, executor.head_rank());
            CUDA_CHECK(cudaMemsetAsync(static_cast<std::int32_t*>(token_counts.p) +
                                           std::size_t(r.slot) * domain,
                                       0, std::size_t(domain) * sizeof(std::int32_t),
                                       executor.head_stream()));
        }
        r.position         = prompt_n;
        r.decoding         = true;
        const SampleRow row = sample_row(r);
        TokenId token       = 0;
        runtime::RawTokenLogprob logprob;
        sample(std::span(&row, 1), ops::kSamplePurposePrefill, std::span(&token, 1),
               std::span(&logprob, 1));
        accept(request, token, row.logprobs ? &logprob : nullptr);
    }

    // Applies the output policy to a sampled token: publishes what it accepts, then either
    // finishes the request or queues what its next decode step feeds. `logprob` is the token's
    // record when the request asked for logprobs.
    void accept(const std::shared_ptr<Request>& request, TokenId token,
                const runtime::RawTokenLogprob* logprob) {
        (void)accept(request, std::span<const TokenId>(&token, 1),
                     logprob != nullptr ? std::span<const runtime::RawTokenLogprob>(logprob, 1)
                                        : std::span<const runtime::RawTokenLogprob>{});
        check_capacity(request);
    }

    [[nodiscard]] bool active(const std::shared_ptr<Request>& request) const {
        return slots.at(request->slot).request == request;
    }

    // A request whose next feed would pass the context finishes.
    void check_capacity(const std::shared_ptr<Request>& request) {
        if (active(request) && request->position + request->feed.size() > max_context) {
            finish_now(request, FinishReason::ContextCapacity);
        }
    }

    // The same for a run of sampled tokens (a speculative round's licensed tokens), without the
    // capacity check, which the caller makes once the round's position is known: returns how many
    // of them the policy accepted. The next step feeds the last accepted one (the first when none
    // was, as a single sampled token is fed whatever the policy decides).
    std::uint32_t accept(const std::shared_ptr<Request>& request, std::span<const TokenId> tokens,
                         std::span<const runtime::RawTokenLogprob> logprobs) {
        Request& r                    = *request;
        const OutputDecision decision = r.output.preview_model(tokens, r.budget->remaining(),
                                                               r.budget->limit_reason(), logprobs);
        const auto now = Clock::now();
        if (r.first_token == Clock::time_point{}) { r.first_token = now; }
        r.last_token = now;
        if (decision.accepted_tokens > tokens.size()) {
            throw std::logic_error("output policy accepted more than the sampled tokens");
        }
        const std::uint32_t accepted = decision.accepted_tokens;
        if (accepted > 0) {
            r.generated.insert(r.generated.end(), tokens.begin(), tokens.begin() + accepted);
            r.budget->commit(accepted);
        }
        const TokenId token = tokens[accepted > 0 ? accepted - 1 : 0];
        std::optional<GenerationTimingObservation> timing;
        if (r.observation.live_timings) {
            timing = GenerationTimingObservation{
                .generated_tokens      = static_cast<std::uint32_t>(r.generated.size()),
                .prompt_elapsed_ns     = elapsed_ns(r.admitted, r.first_token),
                .generation_elapsed_ns = elapsed_ns(r.first_token, now)};
        }
        push_events(r, r.output.commit_preview(), timing);
        {
            std::lock_guard lock(stats_mutex);
            stats.committed_decode_tokens += decision.accepted_tokens;
        }
        if (decision.finished()) {
            finish(request, decision.finish_reason);
            return accepted;
        }
        r.feed.assign(1, token);
        if (decision.continuation == ContinuationAction::ApplyTargetControl) {
            const auto pending_control = r.output.pending_control_tokens();
            const std::vector<TokenId> control(pending_control.begin(), pending_control.end());
            const OutputDecision forced = r.output.preview_control(control, r.budget->remaining());
            if (forced.accepted_tokens != control.size() || forced.finished()) {
                throw std::logic_error("thinking control preview returned an invalid decision");
            }
            r.generated.insert(r.generated.end(), control.begin(), control.end());
            r.budget->commit(static_cast<std::uint32_t>(control.size()));
            push_events(r, r.output.commit_preview(), std::nullopt);
            r.feed.insert(r.feed.end(), control.begin(), control.end());
        }
        if (r.cancelled.load(std::memory_order_acquire)) {
            finish_now(request, FinishReason::Cancelled);
        }
        return accepted;
    }

    // Whether the request's next step can be a speculative round: an Engine with drafts, one token
    // to feed, an MTP state that follows the sequence, room for the round in the context, and no
    // switch of sampling parameters pending (a reasoning block that has not closed yet when the
    // request samples its answer differently).
    [[nodiscard]] bool speculates(const Request& r) const {
        return drafts > 0 && r.feed.size() == 1 && instance.executor->can_draft(r.slot) &&
               r.position + drafts + 1 <= max_context &&
               r.output.model_token_budget_remaining(r.budget->remaining()) > 1 &&
               !(r.options.execution.post_thinking_sampling && !r.post_thinking);
    }

    // One step of every decoding request: those that speculate run one MTP round together, those
    // that feed one token otherwise run as one batch, whose experts read their weights once; one
    // feeding a thinking-control suffix runs alone.
    void decode_step() {
        auto& executor = *instance.executor;
        std::vector<std::shared_ptr<Request>> batch, alone, speculative;
        for (const Slot& slot : slots) {
            if (!slot.request || !slot.request->decoding) { continue; }
            if (speculates(*slot.request)) {
                speculative.push_back(slot.request);
            } else {
                (slot.request->feed.size() == 1 ? batch : alone).push_back(slot.request);
            }
        }
        const auto start = Clock::now();
        for (const auto& request : alone) {
            Request& r = *request;
            executor.forward(r.slot, r.feed, 1);
            Slot& slot = slots[r.slot];
            slot.fed.insert(slot.fed.end(), r.feed.begin(), r.feed.end());
            r.position += static_cast<std::uint32_t>(r.feed.size());
            const SampleRow row = sample_row(r);
            TokenId token       = 0;
            runtime::RawTokenLogprob logprob;
            sample(std::span(&row, 1), ops::kSamplePurposeDecode, std::span(&token, 1),
                   std::span(&logprob, 1));
            try {
                accept(request, token, row.logprobs ? &logprob : nullptr);
            } catch (...) { fail(request, std::current_exception()); }
        }
        if (!batch.empty()) {
            std::vector<std::uint32_t> sequences;
            std::vector<TokenId> tokens;
            for (const auto& request : batch) {
                sequences.push_back(request->slot);
                tokens.push_back(request->feed.front());
            }
            executor.decode(sequences, tokens);
            std::vector<SampleRow> rows;
            for (const auto& request : batch) {
                slots[request->slot].fed.push_back(request->feed.front());
                request->position += 1;
                rows.push_back(sample_row(*request));
            }
            std::vector<TokenId> sampled(batch.size());
            std::vector<runtime::RawTokenLogprob> logprobs(batch.size());
            sample(rows, ops::kSamplePurposeDecode, sampled, logprobs);
            for (std::size_t b = 0; b < batch.size(); ++b) {
                if (drafts > 0) { ++batch[b]->speculative.fallback_steps; }
                try {
                    accept(batch[b], sampled[b], rows[b].logprobs ? &logprobs[b] : nullptr);
                } catch (...) { fail(batch[b], std::current_exception()); }
            }
        }
        if (!speculative.empty()) { speculative_round(speculative); }
        std::lock_guard lock(stats_mutex);
        stats.decode_rounds += 1;
        stats.decode_row_rounds += batch.size() + alone.size() + speculative.size();
        stats.decode_seconds_total += seconds(start, Clock::now());
    }

    // One MTP round of each request: the MTP block drafts from its anchor, the target verifies the
    // anchor and the drafts at once, the acceptance licenses a run of tokens on the device (the
    // accepted drafts and a correction or bonus, sampled as the request samples, so the output
    // follows the target's distribution), the output policy takes what it accepts of the run, and
    // the executor commits the tokens fed up to the last of them.
    void speculative_round(const std::vector<std::shared_ptr<Request>>& batch) {
        auto& executor      = *instance.executor;
        const std::size_t b = batch.size();
        std::vector<std::uint32_t> sequences;
        std::vector<TokenId> anchors;
        for (const auto& request : batch) {
            sequences.push_back(request->slot);
            anchors.push_back(request->feed.front());
        }
        const auto round_start = Clock::now();
        // With --adaptive-mtp the controller picks the round's steps from each request's draft
        // survival and the round times it has measured; a new set of requests restarts its probe.
        std::uint32_t steps = drafts;
        if (mtp_controller) {
            std::vector<const runtime::MtpAdaptiveSignal*> signals;
            std::vector<std::uint32_t> available, room;
            std::uint64_t cohort = 1469598103934665603ULL;
            for (const auto& request : batch) {
                signals.push_back(&request->mtp_signal);
                const auto budget = request->output.model_token_budget_remaining(
                    request->budget->remaining());
                room.push_back(static_cast<std::uint32_t>(std::min<std::uint64_t>(
                    {static_cast<std::uint64_t>(budget),
                     static_cast<std::uint64_t>(max_context - request->position), 1U << 20})));
                available.push_back(std::min(drafts, room.back() > 0 ? room.back() - 1U : 0U));
                cohort = (cohort ^ (static_cast<std::uint64_t>(request->slot) << 32U |
                                    static_cast<std::uint32_t>(
                                        request->admitted.time_since_epoch().count()))) *
                         1099511628211ULL;
            }
            steps = std::clamp<std::uint32_t>(
                mtp_controller->select(signals, available, room, cohort), 1, drafts);
        }
        std::vector<TokenId> all_drafts(b * drafts);
        std::vector<std::uint32_t> draft_extents(b, steps);
        // Context lookup (--lookup-ngram N): a row whose last N tokens, its anchor last, appeared
        // earlier in its sequence proposes what followed then, up to `drafts` tokens, in place of
        // the MTP block's guess, which is weakest where the output repeats its input; verification
        // keeps it exact. The MTP block drafts the other rows only, and no row when all have one.
        std::vector<char> looked_up(b, 0);
        std::size_t lookups = 0;
        if (lookup_ngram != 0) {
            for (std::size_t j = 0; j < b; ++j) {
                std::vector<TokenId>& ledger = slots[batch[j]->slot].fed;
                ledger.push_back(anchors[j]);
                const std::uint32_t found = runtime::lookup_draft(
                    ledger, lookup_ngram, drafts, all_drafts.data() + j * drafts);
                ledger.pop_back();
                if (found != 0) {
                    looked_up[j]     = 1;
                    draft_extents[j] = found;
                    ++lookups;
                }
            }
        }
        if (lookups < b) {
            std::vector<std::uint32_t> mtp_sequences;
            std::vector<TokenId> mtp_anchors;
            for (std::size_t j = 0; j < b; ++j) {
                if (looked_up[j]) { continue; }
                mtp_sequences.push_back(sequences[j]);
                mtp_anchors.push_back(anchors[j]);
            }
            std::vector<TokenId> mtp_drafts(mtp_sequences.size() * drafts);
            std::vector<std::uint32_t> mtp_extents(mtp_sequences.size(), steps);
            executor.draft(mtp_sequences, mtp_anchors, mtp_drafts, mtp_extents, steps);
            for (std::size_t j = 0, m = 0; j < b; ++j) {
                if (looked_up[j]) { continue; }
                std::copy_n(mtp_drafts.begin() + std::ptrdiff_t(m * drafts), drafts,
                            all_drafts.begin() + std::ptrdiff_t(j * drafts));
                draft_extents[j] = mtp_extents[m++];
            }
        }
        const std::size_t k = *std::max_element(draft_extents.begin(), draft_extents.end());
        const std::size_t w = k + 1;
        const auto columns = static_cast<std::int32_t>(w);
        std::vector<TokenId> proposed;
        proposed.reserve(b * k);
        for (std::size_t j = 0; j < b; ++j) {
            proposed.insert(proposed.end(), all_drafts.begin() + std::ptrdiff_t(j * drafts),
                            all_drafts.begin() + std::ptrdiff_t(j * drafts + k));
        }
        std::vector<TokenId> tokens;
        for (std::size_t j = 0; j < b; ++j) {
            tokens.push_back(anchors[j]);
            tokens.insert(tokens.end(), proposed.begin() + std::ptrdiff_t(j * k),
                          proposed.begin() + std::ptrdiff_t((j + 1) * k));
        }
        executor.verify(sequences, tokens);

        RankBinding bind(device, executor.head_rank());
        const cudaStream_t stream = executor.head_stream();
        std::vector<SampleRow> rows;
        for (const auto& request : batch) { rows.push_back(sample_row(*request)); }
        // The rows' sampling, each column of a grammar's row masked as the drafts before it leave
        // the grammar.
        auto* configs = static_cast<ops::SamplingConfig*>(host_sample->data());
        bool masked   = false;
        for (std::size_t j = 0; j < b; ++j) {
            const SampleRow& row = rows[j];
            ops::SamplingConfig config;
            config.temperature       = row.params->temperature;
            config.top_k             = row.params->top_k;
            config.top_p             = row.params->top_p;
            config.min_p             = row.params->min_p;
            config.presence_penalty  = row.params->presence_penalty;
            config.frequency_penalty = row.params->frequency_penalty;
            config.seed              = row.params->seed;
            config.token_counts      = row.counts ? static_cast<std::int32_t*>(token_counts.p) +
                                                        std::size_t(row.slot) * domain
                                                  : nullptr;
            if (row.grammar != nullptr) {
                auto* words = static_cast<std::uint32_t*>(host_mask->data()) + j * w * mask_words();
                row.grammar->fill_masks(std::span(words, w * mask_words()),
                                        std::span<const TokenId>(proposed).subspan(j * k, k));
                config.token_mask =
                    static_cast<const std::uint32_t*>(token_mask.p) + j * w * mask_words();
                config.token_mask_stride = static_cast<std::int32_t>(mask_words());
                masked                   = true;
            }
            configs[j] = config;
        }
        if (masked) {
            CUDA_CHECK(cudaMemcpyAsync(token_mask.p, host_mask->data(),
                                       b * w * mask_words() * sizeof(std::uint32_t),
                                       cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(sample_config.p, configs, b * sizeof(ops::SamplingConfig),
                                   cudaMemcpyHostToDevice, stream));
        // Drafts, live draft counts and the tokens each request fed before the round.
        auto* staged = static_cast<std::int32_t*>(host_spec->data());
        std::copy(proposed.begin(), proposed.end(), staged);
        for (std::size_t j = 0; j < b; ++j) {
            // The batch uses its largest confidence prefix. Acceptance also respects each
            // sequence's shorter prefix and its output/thinking budget, including correction.
            const auto remaining = batch[j]->output.model_token_budget_remaining(
                batch[j]->budget->remaining());
            staged[b * k + j] = static_cast<std::int32_t>(
                std::min<std::size_t>(draft_extents[j], remaining - 1));
            staged[b * k + b + j] = static_cast<std::int32_t>(batch[j]->position);
        }
        CUDA_CHECK(
            cudaMemcpyAsync(spec_drafts.p, staged, b * k * 4, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(
            cudaMemcpyAsync(spec_extents.p, staged + b * k, b * 4, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(spec_lengths.p, staged + b * k + b, b * 4,
                                   cudaMemcpyHostToDevice, stream));
        const auto rows_n         = static_cast<std::int32_t>(b);
        const Tensor logits       = executor.logits(static_cast<std::uint32_t>(b * w));
        const auto* device_config = static_cast<const ops::SamplingConfig*>(sample_config.p);
        const Tensor drafted(spec_drafts.p, DType::I32, {static_cast<std::int32_t>(k), rows_n});
        // The tokens' logprob records, from the distributions their columns are drawn from,
        // before the acceptance adds the round's tokens to the penalty counts.
        const bool gather     = std::any_of(rows.begin(), rows.end(),
                                            [](const SampleRow& row) { return row.logprobs; });
        const std::size_t top = b * w * kMaximumTokenLogprobs;
        auto* host_ids        = static_cast<std::int32_t*>(host_logprobs->data());
        auto* host_values =
            reinterpret_cast<float*>(host_ids + slots.size() * w * kMaximumTokenLogprobs);
        if (gather) {
            Tensor ids(logprob_ids.p, DType::I32, {ops::kLogprobTopK, columns, rows_n});
            Tensor values(logprob_values.p, DType::FP32, {ops::kLogprobTopK, columns, rows_n});
            Tensor lse(logprob_lse.p, DType::FP32, {columns, rows_n});
            const Tensor flag(logprob_flag.p, DType::I32, {1});
            ops::logprob_topk(logits.view({logits.ne[0], columns, rows_n}), device_config, &drafted,
                              static_cast<std::int32_t>(domain), ids, values, lse, flag,
                              *sample_workspace, stream);
            CUDA_CHECK(cudaMemcpyAsync(host_ids, logprob_ids.p, top * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(host_values, logprob_values.p, top * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
        }
        for (std::size_t j = 0; j < b; ++j) {
            if (!rows[j].counts) { continue; }
            const std::size_t at = std::size_t(rows[j].slot) * domain * sizeof(std::int32_t);
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(counts_backup.p) + at,
                                       static_cast<const std::byte*>(token_counts.p) + at,
                                       std::size_t(domain) * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToDevice, stream));
        }
        Tensor targets(spec_targets.p, DType::I32, {columns * rows_n});
        ops::argmax(logits, targets, static_cast<std::int32_t>(domain), stream);
        Tensor extents(spec_extents.p, DType::I32, {rows_n});
        Tensor lengths(spec_lengths.p, DType::I32, {rows_n});
        Tensor round_anchors(spec_anchors.p, DType::I32, {rows_n});
        Tensor licensed(spec_licensed.p, DType::I32, {columns, rows_n});
        Tensor counts(spec_counts.p, DType::I32, {rows_n});
        Tensor accepted(spec_accepted.p, DType::I32, {rows_n});
        {
            auto scope = sample_workspace->scope();
            ops::speculative_accept_greedy_drafts(
                targets.view({columns, rows_n}), logits.view({logits.ne[0], columns, rows_n}),
                drafted, extents, lengths, round_anchors, licensed, counts, accepted,
                static_cast<std::int32_t>(domain), device_config, *sample_workspace, stream);
        }
        auto* host = staged + b * (k + 2);
        CUDA_CHECK(
            cudaMemcpyAsync(host, spec_licensed.p, b * w * 4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(
            cudaMemcpyAsync(host + b * w, spec_counts.p, b * 4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host + b * w + b, spec_accepted.p, b * 4, cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // The output policy takes its part of each licensed run; the executor then keeps the tokens
        // fed up to the last one taken.
        std::vector<std::uint32_t> kept(b, 1);
        std::vector<char> failed(b, 0);
        for (std::size_t j = 0; j < b; ++j) {
            const auto& request           = batch[j];
            Request& r                    = *request;
            const std::int32_t licensed_n = host[b * w + j];
            const std::int32_t drafts_n   = host[b * w + b + j];
            const std::int32_t* run       = host + j * w;
            if (licensed_n < 1 || licensed_n > columns || run[0] == ops::kSamplerNonFiniteToken) {
                throw std::runtime_error("Qwen3.8-Flash-Next produced non-finite logits");
            }
            for (std::int32_t i = 0; i < licensed_n; ++i) {
                if (run[i] < 0 || std::uint32_t(run[i]) >= domain) {
                    throw std::runtime_error("Qwen3.8-Flash-Next licensed a token outside the "
                                             "vocabulary");
                }
            }
            std::vector<runtime::RawTokenLogprob> records;
            if (rows[j].logprobs) {
                for (std::int32_t i = 0; i < licensed_n; ++i) {
                    runtime::RawTokenLogprob record{.id = run[i], .logprob = kLogprobSentinel};
                    for (std::size_t t = 0; t < kMaximumTokenLogprobs; ++t) {
                        const std::size_t at = (j * w + std::size_t(i)) * kMaximumTokenLogprobs + t;
                        record.top_ids[t]    = host_ids[at];
                        record.top_values[t] = host_values[at];
                        if (host_ids[at] == run[i]) { record.logprob = host_values[at]; }
                    }
                    records.push_back(record);
                }
            }
            r.speculative.rounds += 1;
            // The MTP draft chain runs all its steps; a lookup proposes what it found.
            r.speculative.drafted_tokens += looked_up[j] ? draft_extents[j] : steps;
            r.speculative.accepted_tokens += std::uint32_t(drafts_n);
            if (looked_up[j]) {
                // Reported with the n-gram proposals: what the context, not the model, proposed.
                ++r.speculative.ngram_rounds;
                r.speculative.ngram_drafted_tokens += draft_extents[j];
                r.speculative.ngram_accepted_tokens += std::uint32_t(drafts_n);
            } else if (mtp_controller) {
                r.mtp_signal.observe(std::min<std::uint32_t>(draft_extents[j], steps),
                                     std::uint32_t(drafts_n));
                ++r.speculative.rounds_per_window[steps - 1];
                if (mtp_controller->transitioned()) { ++r.speculative.window_transitions; }
            }
            for (std::int32_t i = 0; i < drafts_n; ++i) {
                ++r.speculative.accepted_per_position[i];
            }
            const std::span<const TokenId> run_tokens(run, std::size_t(licensed_n));
            std::uint32_t taken = 0;
            try {
                taken = accept(request, run_tokens, records);
            } catch (...) {
                fail(request, std::current_exception());
                failed[j] = 1;
            }
            // The tokens counted as sampled: those taken, or the first when none was.
            kept[j] = std::max<std::uint32_t>(taken, 1);
            if (kept[j] < std::uint32_t(licensed_n) && active(request) && rows[j].counts) {
                // The acceptance counted the whole run; the request keeps only its prefix.
                const std::size_t at = std::size_t(r.slot) * domain * sizeof(std::int32_t);
                CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(token_counts.p) + at,
                                           static_cast<const std::byte*>(counts_backup.p) + at,
                                           std::size_t(domain) * sizeof(std::int32_t),
                                           cudaMemcpyDeviceToDevice, stream));
                // The run's tokens are still in the licensed plane, at the row's offset.
                const Tensor ids(static_cast<std::int32_t*>(spec_licensed.p) + j * w, DType::I32,
                                 {static_cast<std::int32_t>(kept[j])});
                Tensor row_counts(static_cast<std::int32_t*>(token_counts.p) +
                                      std::size_t(r.slot) * domain,
                                  DType::I32, {static_cast<std::int32_t>(domain)});
                ops::increment_token_counts(ids, row_counts, stream);
            }
        }
        executor.commit(sequences, kept);
        // The controller measures rounds of MTP drafts only: a lookup changes the round's width.
        if (mtp_controller && lookups == 0) {
            mtp_controller->observe_execution(static_cast<std::uint32_t>(b), steps,
                                              seconds(round_start, Clock::now()));
        }
        for (std::size_t j = 0; j < b; ++j) {
            Request& r = *batch[j];
            if (failed[j]) { continue; } // its sequence keeps nothing for the context cache
            Slot& slot = slots[r.slot];
            // The verification fed the anchor and the drafts; the sequence keeps the first ones.
            slot.fed.insert(slot.fed.end(), tokens.begin() + std::ptrdiff_t(j * w),
                            tokens.begin() + std::ptrdiff_t(j * w + kept[j]));
            r.position += kept[j];
            try {
                check_capacity(batch[j]);
            } catch (...) { fail(batch[j], std::current_exception()); }
        }
    }

    std::vector<float> score(const std::vector<TokenId>& tokens, std::uint32_t first_target) {
        auto& executor        = *instance.executor;
        const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
        if (n < 2 || first_target == 0 || first_target >= n || n > max_context) {
            throw std::invalid_argument("score: invalid token window");
        }
        // Logits for position p predict token p + 1; the executor keeps at most 512 logit rows.
        const std::uint32_t chunk = std::min<std::uint32_t>(executor.options().prefill_chunk, 512);
        executor.reset(0);
        slots[0].fed.clear();
        slots[0].anchor.clear();
        std::vector<float> out;
        out.reserve(n - first_target);
        RankBinding bind(device, executor.head_rank());
        for (std::uint32_t at = 0; at + 1 < n; at += chunk) {
            const std::uint32_t count = std::min(chunk, n - 1 - at);
            executor.forward(0, std::span<const TokenId>(tokens).subspan(at, count), count);
            // Rows predicting targets [at + 1, at + count].
            const std::uint32_t first_row = first_target > at + 1 ? first_target - at - 1 : 0;
            if (first_row >= count) { continue; }
            const std::uint32_t rows = count - first_row;
            std::vector<std::int32_t> targets(rows);
            for (std::uint32_t i = 0; i < rows; ++i) {
                targets[i] = tokens[at + 1 + first_row + i];
            }
            const cudaStream_t stream = executor.head_stream();
            CUDA_CHECK(cudaMemcpyAsync(score_targets.p, targets.data(), rows * 4,
                                       cudaMemcpyHostToDevice, stream));
            const Tensor all = executor.logits(count);
            const Tensor logits(static_cast<__nv_bfloat16*>(all.data) +
                                    std::size_t(first_row) * all.ne[0],
                                DType::BF16, {all.ne[0], static_cast<std::int32_t>(rows)});
            const Tensor target_ids(score_targets.p, DType::I32, {static_cast<std::int32_t>(rows)});
            Tensor result(score_out.p, DType::FP32, {static_cast<std::int32_t>(rows)});
            ops::target_logprobs(logits, target_ids, static_cast<std::int32_t>(domain), result,
                                 stream);
            std::vector<float> host(rows);
            CUDA_CHECK(cudaMemcpyAsync(host.data(), score_out.p, rows * 4, cudaMemcpyDeviceToHost,
                                       stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            out.insert(out.end(), host.begin(), host.end());
        }
        executor.save_expert_profile();
        return out;
    }
};

Qwen4ExpCore::Submission::Submission(Qwen4ExpCore& owner, std::shared_ptr<Request> request,
                                     std::optional<std::uint32_t> budget) noexcept
    : owner_(&owner), request_(std::move(request)), effective_thinking_budget_(budget) {}

Qwen4ExpCore::Submission::~Submission() {
    if (request_ != nullptr) { request_->cancelled.store(true, std::memory_order_release); }
}

Qwen4ExpCore::Submission::Submission(Submission&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), request_(std::move(other.request_)),
      effective_thinking_budget_(other.effective_thinking_budget_) {}

Qwen4ExpCore::Submission& Qwen4ExpCore::Submission::operator=(Submission&& other) noexcept {
    if (this != &other) {
        if (request_ != nullptr) { request_->cancelled.store(true, std::memory_order_release); }
        owner_                     = std::exchange(other.owner_, nullptr);
        request_                   = std::move(other.request_);
        effective_thinking_budget_ = other.effective_thinking_budget_;
    }
    return *this;
}

GenerationResult Qwen4ExpCore::Submission::wait(OutputSink* sink,
                                                const CancellationView& cancellation) {
    if (request_ == nullptr) { throw std::logic_error("submission is empty"); }
    const std::shared_ptr<Request> request = std::move(request_);
    const bool streaming = request->consumer_mode == OutputConsumerMode::Streaming;
    if (streaming != (sink != nullptr)) {
        request->cancelled.store(true, std::memory_order_release);
        throw std::invalid_argument(
            "GenerationHandle wait sink does not match its submitted consumer mode");
    }
    std::exception_ptr caller_error;
    for (;;) {
        std::optional<GenerationStart> start;
        std::optional<PromptProgress> progress;
        std::vector<std::variant<OutputDelta, GenerationTimingObservation>> events;
        bool done = false;
        {
            std::unique_lock lock(request->mutex);
            request->cv.wait_for(lock, std::chrono::milliseconds(10), [&] {
                return request->response_done || request->stream_start.has_value() ||
                       request->stream_progress.has_value() || !request->events.empty();
            });
            start    = std::exchange(request->stream_start, std::nullopt);
            progress = std::exchange(request->stream_progress, std::nullopt);
            events.swap(request->events);
            done = request->response_done;
        }
        if (caller_error == nullptr && sink != nullptr) {
            try {
                if (start) { sink->start(std::move(*start)); }
                if (progress) { sink->progress(std::move(*progress)); }
                for (auto& event : events) {
                    if (auto* timing = std::get_if<GenerationTimingObservation>(&event)) {
                        sink->timing(*timing);
                    } else {
                        sink->publish(std::move(std::get<OutputDelta>(event)));
                    }
                }
            } catch (...) {
                caller_error = std::current_exception();
                request->cancelled.store(true, std::memory_order_release);
            }
        }
        if (caller_error == nullptr) {
            try {
                if (cancellation.requested()) {
                    request->cancelled.store(true, std::memory_order_release);
                }
            } catch (...) {
                caller_error = std::current_exception();
                request->cancelled.store(true, std::memory_order_release);
            }
        }
        if (!done) { continue; }
        if (caller_error != nullptr) { std::rethrow_exception(caller_error); }
        std::lock_guard lock(request->mutex);
        if (request->error != nullptr) { std::rethrow_exception(request->error); }
        return std::move(request->result);
    }
}

Qwen4ExpCore::Qwen4ExpCore(Qwen4ExpInstance& instance, DeviceContext& device,
                           const EngineOptions& options)
    : impl_(std::make_unique<Impl>(instance, device, options)) {
    if (options.max_concurrency == 0 || options.max_pending_requests == 0 ||
        options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine core bounds are invalid");
    }
}

Qwen4ExpCore::~Qwen4ExpCore() = default;

void Qwen4ExpCore::stop() noexcept {
    {
        std::lock_guard lock(impl_->queue_mutex);
        impl_->stopping = true;
    }
    impl_->queue_cv.notify_all();
}

Qwen4ExpCore::Submission Qwen4ExpCore::submit(models::qwen3_5::PreparedPrompt prompt,
                                              PromptSummary prompt_summary, double prepare_seconds,
                                              ResolvedRequestOptions options,
                                              OutputConsumerMode consumer_mode,
                                              GenerationObservationOptions observation,
                                              Clock::time_point pending_deadline) {
    const auto submitted = Clock::now();
    if (options.execution.structured_output.kind != StructuredOutputKind::None &&
        !impl_->structured_output) {
        throw std::invalid_argument(
            "structured output requires an Engine started with structured_output");
    }
    if (pending_deadline == Clock::time_point{}) {
        pending_deadline = submitted + impl_->pending_timeout;
    }
    auto request = std::make_shared<Request>();
    request->prompt_tokens =
        std::vector<TokenId>(prompt.token_ids().begin(), prompt.token_ids().end());
    {
        const auto& data     = models::qwen3_5::PreparedPromptAccess::view(prompt);
        const auto& identity = data.identity;
        const auto prompt_n  = static_cast<std::uint32_t>(request->prompt_tokens.size());
        request->media       = data.has_media();
        if (request->media && !impl_->instance.executor->vision()) {
            throw std::invalid_argument("media need an Engine started with Vision");
        }
        // A prompt's media are not part of its tokens, so a media prompt is never reused.
        request->reusable = identity.reusable && !request->media;
        request->anchor_at   = prompt_n;
        if (identity.rewrite_checkpoint &&
            identity.rewrite_checkpoint->kind ==
                models::qwen3_5::RewriteCheckpointKind::TurnClosure &&
            identity.rewrite_checkpoint->frontier > 0 &&
            identity.rewrite_checkpoint->frontier < prompt_n) {
            request->anchor_at = identity.rewrite_checkpoint->frontier;
        }
    }
    request->prompt_summary  = prompt_summary;
    request->prepare_seconds = prepare_seconds;
    request->consumer_mode   = consumer_mode;
    request->observation     = observation;
    request->deadline        = pending_deadline;
    request->submitted       = submitted;
    apply_effective_thinking_budget(
        options.execution.thinking,
        effective_output_capacity(options.execution.requested_output_tokens, impl_->max_context,
                                  prompt_summary.prompt_tokens),
        impl_->instance.frontend.thinking_control_token_count());
    request->output = impl_->instance.frontend.make_output_session(
        prompt, options.stop, options.output, options.execution.thinking,
        options.execution.structured_output);
    const std::optional<std::uint32_t> budget = request->output.thinking_stats().effective_budget;
    request->prompt                           = std::move(prompt);
    request->options                          = std::move(options);
    {
        std::lock_guard lock(impl_->queue_mutex);
        if (impl_->stopping || impl_->failed) {
            throw RequestError(RequestErrorKind::Unavailable, "inference engine is unavailable");
        }
        if (impl_->outstanding >= impl_->max_outstanding) {
            throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
        }
        ++impl_->outstanding;
        request->id = impl_->next_id++;
        impl_->pending.push_back(request);
    }
    impl_->queue_cv.notify_one();
    impl_->publish_queue();
    return Submission(*this, std::move(request), budget);
}

std::vector<float> Qwen4ExpCore::score(models::qwen3_5::PreparedPrompt prompt,
                                       std::uint32_t first_target) {
    const std::vector<TokenId> tokens(prompt.token_ids().begin(), prompt.token_ids().end());
    {
        std::lock_guard lock(impl_->queue_mutex);
        if (impl_->stopping || impl_->failed) {
            throw RequestError(RequestErrorKind::Unavailable, "inference engine is unavailable");
        }
        if (impl_->outstanding != 0) {
            throw std::logic_error("score requires an idle Qwen3.8-Flash-Next Engine");
        }
        ++impl_->outstanding;
    }

    struct Release {
        Impl& impl;

        ~Release() {
            std::lock_guard lock(impl.queue_mutex);
            --impl.outstanding;
        }
    } release{*impl_};

    impl_->device.bind_to_current_thread();
    return impl_->score(tokens, first_target);
}

MemorySummary Qwen4ExpCore::memory_summary() const {
    const auto& stats  = impl_->instance.model->storage_stats();
    const auto memory  = impl_->instance.executor->memory();
    const auto arena   = [](std::uint64_t bytes) {
        return ArenaMemorySummary{bytes, bytes, bytes};
    };
    std::vector<DeviceMemorySummary> devices;
    for (std::size_t r = 0; r < impl_->device.size(); ++r) {
        const auto& rank = memory.ranks.at(r);
        devices.push_back({.device    = impl_->device.rank(r).device,
                           .weights   = arena(r < stats.device_capacity_by_rank.size()
                                                  ? stats.device_capacity_by_rank[r]
                                                  : 0),
                           .sequence  = arena(rank.state_bytes),
                           .workspace = arena(rank.workspace_bytes),
                           .expert_cache_bytes = rank.expert_cache_bytes});
    }
    MemorySummary out;
    out.device                    = devices.front().device;
    out.max_context               = impl_->max_context;
    out.kv_capacity               = impl_->max_context;
    out.kv_cache                      = impl_->instance.executor->options().kv_cache;
    out.weights                   = devices.front().weights;
    out.sequence                  = devices.front().sequence;
    out.workspace                 = devices.front().workspace;
    out.expert_cache_bytes        = memory.expert_cache_bytes;
    out.kv_payload_bytes          = memory.kv_bytes;
    out.runtime_reservation_bytes = memory.state_bytes + memory.workspace_bytes;
    out.available_after_weights_bytes = impl_->instance.free_after_weights;
    out.available_after_startup_bytes = impl_->instance.free_after_startup;
    if (devices.size() > 1) { out.devices = std::move(devices); }
    return out;
}

RuntimeStats Qwen4ExpCore::runtime_stats() const {
    std::lock_guard lock(impl_->stats_mutex);
    return impl_->stats;
}

bool Qwen4ExpCore::is_available() const {
    std::lock_guard lock(impl_->queue_mutex);
    return !impl_->stopping && !impl_->failed;
}

bool Qwen4ExpCore::has_failed() const {
    std::lock_guard lock(impl_->queue_mutex);
    return impl_->failed;
}

} // namespace ninfer::runtime
