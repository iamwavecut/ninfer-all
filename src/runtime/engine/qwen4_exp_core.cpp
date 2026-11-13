#include "runtime/engine/qwen4_exp_core.h"

#include "artifact/formats.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device_buffer.h"
#include "core/device_wait.h"
#include "core/host_pages.h"
#include "core/file_read_error.h"
#include "core/pinned_memory.h"
#include "core/startup.h"
#include "core/tensor_parallel.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/vision_control.h"
#include "models/qwen4_exp/ngram_component.h"
#include "models/qwen4_exp/ngram_profile.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/logprob_topk.h"
#include "ninfer/ops/readout_logprobs.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/target_logprobs.h"
#include "runtime/contract/draft_sampling.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/grammar_masks.h"
#include "runtime/contract/lookup_draft.h"
#include "runtime/contract/mtp_adaptive.h"
#include "runtime/contract/rope_scaling.h"
#include "runtime/engine/diagnostics.h"
#include "runtime/engine/worker_fault.h"
#include "runtime/engine/worker_recovery.h"
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
#include <optional>
#include <set>
#include <stdexcept>
#include <thread>
#include <variant>

namespace ninfer::runtime {
namespace {

// Prompt chunks in one prefill step while other requests decode (see prompt_step).
constexpr std::uint32_t kPeerSpanChunks = 2;

using Clock = std::chrono::steady_clock;

// Device memory the automatic expert cache leaves free when it takes the rest after startup, for
// what requests still allocate: kernels the warm-up did not reach and library workspaces.
constexpr std::uint64_t kGrowthMargin = 384ULL << 20;

// Tensor parallelism without --prefill-chunk: prompt chunks of this many tokens where every device
// has the room. The executor runs them in pieces of 1024 whose exchanges cross while the next
// piece computes, so a chunk computes what chunks of 1024 do.
constexpr std::uint32_t kTensorPrefillChunk = 2048;

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
    } catch (const text::GrammarViolation&) {
        // Raised by the output policy before anything of the round is committed.
        return true;
    } catch (...) {}
    return false;
}

// A prompt's media unit (an image, or a video's frames between two timestamps): its tokens
// [begin, begin + count) and a key of what they encode, the item's digest, modality, grid and
// timestamps and the unit's index. Media tokens are placeholders, so a prefix is the same only
// with the same units; the context cache stores them beside the tokens (16 bytes each on disk).
struct MediaMark {
    std::uint32_t begin = 0, count = 0;
    std::uint64_t key   = 0;

    friend bool operator==(const MediaMark&, const MediaMark&) = default;
};
static_assert(sizeof(MediaMark) == 16);

std::uint64_t unit_key(const models::qwen3_5::VisionItem& item, std::size_t unit) {
    std::uint64_t key = 1469598103934665603ULL; // FNV-1a
    const auto mix    = [&](std::uint64_t value) {
        for (int i = 0; i < 8; ++i) {
            key ^= (value >> (8 * i)) & 0xFFU;
            key *= 1099511628211ULL;
        }
    };
    for (const std::uint8_t byte : item.content_digest) { mix(byte); }
    mix(static_cast<std::uint64_t>(item.modality));
    mix(std::uint64_t(item.grid.temporal));
    mix(std::uint64_t(item.grid.height));
    mix(std::uint64_t(item.grid.width));
    for (const double stamp : item.timestamps) { mix(std::bit_cast<std::uint64_t>(stamp)); }
    mix(unit);
    return key;
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
    if (options.prefill_stall_budget_ms != 0) {
        throw std::invalid_argument("Qwen3.8-Flash-Next sizes its own prefill steps; --prefill-stall-budget-ms "
                                    "applies to Qwen3.5-family models");
    }
    if (options.enable_vision && options.vision_residency != VisionResidency::Resident) {
        throw std::invalid_argument(
            "Qwen3.8-Flash-Next runs its Vision tower resident on the device only");
    }
    const SpeculativeOptions& speculative = options.speculative;
    if (speculative.backend == SpeculativeBackend::DFlash ||
        speculative.backend == SpeculativeBackend::DSpark) {
        throw std::invalid_argument("Qwen3.8-Flash-Next drafts with its MTP block (--spec mtp), "
                                    "its DFlash2 drafter (--spec dflash2) or a DSpine drafter "
                                    "(--spec dspine)");
    }
    const bool mtp     = speculative.backend == SpeculativeBackend::Mtp;
    const bool dflash2 = speculative.backend == SpeculativeBackend::DFlash2;
    const bool dspine  = speculative.backend == SpeculativeBackend::DSpine;
    const bool drafts  = mtp || dflash2 || dspine;
    const float floor  = speculative.draft_min_p.value_or(0.0F);
    if (!std::isfinite(floor) || floor < 0 || floor > 1 || (!drafts && floor != 0)) {
        throw std::invalid_argument(
            "--draft-min-p requires --spec mtp, dflash2 or dspine and a finite value in [0,1]");
    }
    if (drafts && (speculative.draft_tokens == 0 || speculative.draft_tokens > 15)) {
        throw std::invalid_argument("--spec mtp, dflash2 and dspine require --draft-tokens in "
                                    "[1,15]");
    }
    // The drafter predicts a block of eight positions, the last token and seven masks: incoai's
    // drafter was trained on such blocks (its config's block_size), and a wider round verifies
    // more than eight columns on a slower route.
    if (dflash2 && speculative.draft_tokens > 7) {
        throw std::invalid_argument("--spec dflash2 drafts 1 to 7 tokens a round: the drafter "
                                    "predicts blocks of 8 positions");
    }
    if (!drafts && speculative.ngram_draft_tokens != 0) {
        throw std::invalid_argument("n-gram copy proposals are not available for "
                                    "Qwen3.8-Flash-Next");
    }
    if (speculative.mtp_attention_window != 0 ||
        speculative.proposal_head == ProposalHead::Optimized || speculative.ngram_archive_bytes != 0) {
        throw std::invalid_argument("Qwen3.8-Flash-Next's MTP drafting has no "
                                    "--mtp-attention-window, --lm-head-draft or n-gram archive");
    }
    if (speculative.lookup_ngram != 0 && !drafts) {
        throw std::invalid_argument("--lookup-ngram requires --spec mtp, dflash2 or dspine");
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
         table.io_depth != defaults.io_depth || table.dedupe != defaults.dedupe)) {
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
        .depth        = table.io_depth,
        .dedupe       = table.dedupe};
    if (!table.disabled && has_ple) {
        ngram = models::qwen4_exp::ngram_table_source(
            reader, options.artifact_path, text, table.path,
            table.residency == NgramResidency::RamHot && table.hot_profile.empty());
        if (ngram->table_digest != ngram->model_digest ||
            ngram->table_format != ngram->model_format) {
            publish_diagnostic(
                options.diagnostic_observer, DiagnosticLevel::Warning,
                "Qwen3.8-Flash-Next: n-gram table %s, sha256 %.16s..., in place of the %s table "
                "the model names, sha256 %.16s...; its hash constants and geometry match the "
                "model's, its values are not checked",
                ngram->table_format.c_str(), ngram->table_digest.c_str(),
                ngram->model_format.c_str(), ngram->model_digest.c_str());
        }
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
    load.dflash2      = dflash2;
    if (dspine) { load.dspine = speculative.draft_model; }
    // Tensor parallelism: every layer's mixer heads, routed experts and shared expert divided over
    // the devices, and the head's rows; the MTP block, the DFlash2 drafter and Vision on the first.
    // Without --tensor-split each device takes a share of its speed that fits beside what it holds.
    if (options.tensor_parallel) {
        load.tensor_split = options.tensor_split;
        if (load.tensor_split.empty()) {
            load.tensor_split = models::qwen4_exp::default_tensor_split(
                reader, load, rank_free_bytes(device), rank_copy_bandwidth(device));
        }
    }
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
    executor.draft_tokens    = drafts ? speculative.draft_tokens : 0;
    executor.dflash2         = dflash2;
    executor.dspine          = dspine;
    executor.draft_min_p     = speculative.draft_min_p.value_or(0.0F);
    // The rotation past the native window, as every family resolves it (rope_scaling.h): YaRN
    // reaches every rotation this model makes (sparse attention, its indexer, the MTP block);
    // linear position interpolation is refused.
    const runtime::RopeScalingPlan rope = runtime::resolve_rope_scaling(
        options, text.max_position_embeddings, runtime::kQwen4ExpRopeScaling);
    executor.rope_yarn       = rope.yarn;
    // A chunk also holds a verification of every sequence at once.
    const auto chunk_of = [&](std::uint32_t requested) {
        return std::max(std::clamp<std::uint32_t>(requested, 64, 4096),
                        (executor.draft_tokens + 1) * executor.sequences);
    };
    executor.prefill_chunk = chunk_of(options.prefill_chunk ? options.prefill_chunk : 1024);
    instance->ngram_table_digest = ngram ? ngram->table_digest : std::string();
    executor.ngram           = std::move(ngram);
    executor.ngram_read      = std::move(ngram_read);
    executor.iq1m_matrix     = options.iq1m_expert_matrix;
    executor.expert_cache_bytes =
        options.expert_cache_bytes.value_or(models::qwen4_exp::ExecutorOptions::kAutoExpertCache);
    executor.cuda_graphs = options.use_cuda_graph;
    executor.sampled_drafts = options.gumbel_sampling;
    executor.hybrid_experts = options.hybrid_experts;
    executor.vision_max_merged_tokens = options.vision_max_merged_tokens;
    executor.kv_cache                 = options.kv_cache;
    // Host and disk experts trade device KV for expert slots by default.
    executor.kv_resident = options.kv_resident_positions.value_or(
        options.expert_residency == ExpertResidency::Device ? 0 : kDefaultKvResidentPositions);
    executor.tensor_parallel_transport       = options.tensor_parallel_transport;
    executor.tensor_parallel_prompt_partials = options.tensor_parallel_prompt_partials;
    instance->executor = std::make_unique<models::qwen4_exp::Executor>(*model, device, executor);
    // Tensor parallelism takes chunks of kTensorPrefillChunk when every device keeps room for them:
    // a chunk's buffers grow at most in proportion to it, so by at most the narrower chunk's
    // workspace, beside the margin requests still allocate.
    if (options.tensor_parallel && options.prefill_chunk == 0 &&
        executor.prefill_chunk < chunk_of(kTensorPrefillChunk)) {
        const models::qwen4_exp::ExecutorMemory memory = instance->executor->memory();
        const std::vector<std::uint64_t> free          = rank_free_bytes(device);
        bool room                                      = true;
        for (std::size_t r = 0; r < free.size(); ++r) {
            room = room && free[r] >= memory.ranks.at(r).workspace_bytes + kGrowthMargin;
        }
        if (room) {
            instance->executor.reset();
            executor.prefill_chunk = chunk_of(kTensorPrefillChunk);
            instance->executor =
                std::make_unique<models::qwen4_exp::Executor>(*model, device, executor);
        }
    }
    device.synchronize();
    program.complete();
    if (const std::string routes = instance->executor->tensor_parallel_routes(); !routes.empty()) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info, "%s",
                           routes.c_str());
    }
    {
        // The first request would otherwise load every kernel it reaches, and requests would take
        // device memory for them and for the CUDA Graphs the warm-up captures.
        StartupPhaseScope warm(options.startup_observer, StartupPhase::CudaGraphPrepare);
        instance->executor->warm_up();
        // Every sequence's snapshot, which the context cache takes at a prompt's anchor and a
        // branched read at its shared prefix, is sized now, beside its other state.
        instance->snapshots.resize(executor.sequences);
        for (auto& snapshot : instance->snapshots) { instance->executor->size_snapshot(snapshot); }
        if (options.context_cache.enabled) {
            instance->holds.resize(executor.sequences);
            for (auto& hold : instance->holds) { instance->executor->size_snapshot(hold); }
        }
        // With --expert-cache-mib auto, what startup left free beyond a margin becomes expert
        // slots; an explicit size is the cache's size.
        if (!options.expert_cache_bytes) { instance->executor->grow_expert_cache(kGrowthMargin); }
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
                       "Qwen3.8-Flash-Next: %zu stage(s), prompt chunks of %u tokens, experts in "
                       "%s, n-gram table %s, state %.0f MiB, workspace %.0f MiB, expert "
                       "cache %.0f MiB, CUDA Graphs %.0f MiB",
                       model->stages().stages(), executor.prefill_chunk,
                       options.expert_residency == ExpertResidency::Host   ? "host memory"
                       : options.expert_residency == ExpertResidency::Disk ? "the artifact's files"
                                                                           : "device memory",
                       table_place.c_str(),
                       double(instance->executor->memory().state_bytes) / 1048576.0,
                       double(instance->executor->memory().workspace_bytes) / 1048576.0,
                       double(instance->executor->memory().expert_cache_bytes) / 1048576.0,
                       double(instance->executor->graph_bytes()) / 1048576.0);
    if (model->tensor_parallel()) {
        std::string keys, kv, experts, shared, head;
        const auto range = [](std::string& out, std::size_t rank, std::uint64_t begin,
                              std::uint64_t end) {
            out += (rank == 0 ? "" : ", ") +
                   (begin == end ? std::string("none")
                                 : std::to_string(begin) + "-" + std::to_string(end - 1));
        };
        const auto& head_rows = model->weights().head_rows;
        for (std::size_t rank = 0; rank < device.size(); ++rank) {
            const auto share = model->share(rank);
            range(keys, rank, share.key_head_begin, share.key_head_end);
            range(kv, rank, share.kv_head_begin, share.kv_head_end);
            range(experts, rank, share.expert_begin, share.expert_end);
            range(shared, rank, share.shared_begin, share.shared_end);
            range(head, rank, head_rows.at(rank), head_rows.at(rank + 1));
        }
        publish_diagnostic(
            options.diagnostic_observer, DiagnosticLevel::Info,
            "Qwen3.8-Flash-Next: tensor parallelism, by device Gated DeltaNet key "
            "heads %s, sparse-attention KV heads %s, %s %s, shared-expert "
            "middle %s, head rows %s",
            keys.c_str(), kv.c_str(),
            model->config().latent_moe
                ? "latent groups cut in each layer nearest the routed experts' shares"
                : "routed experts",
            experts.c_str(), shared.c_str(), head.c_str());
    }
    if (const auto memory = instance->executor->memory(); memory.streamed_kv) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                           "Qwen3.8-Flash-Next: sparse-attention KV in pinned host memory "
                           "(%.0f MiB), %.0f MiB of its blocks on the device",
                           double(memory.host_kv_bytes) / 1048576.0,
                           double(memory.kv_bytes) / 1048576.0);
    }
    if (const std::string cpu = instance->executor->expert_cpu_description(); !cpu.empty()) {
        const std::string share =
            !options.hybrid_experts.dma_share
                ? std::string("split by measured speeds")
                : std::to_string(1.0F - *options.hybrid_experts.dma_share) + " of them";
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                           "Qwen3.8-Flash-Next: the CPU computes missing experts (%s; %s)",
                           cpu.c_str(), share.c_str());
    }
    if (drafts) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                           "Qwen3.8-Flash-Next: %s speculative decoding, %u drafts a round",
                           dflash2 ? "DFlash2" : (dspine ? "DSpine" : "MTP"),
                           speculative.draft_tokens);
        if (speculative.ngram_draft_tokens != 0) {
            publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                               "Qwen3.8-Flash-Next drafts with its %s only; n-gram copy "
                               "proposals (--ngram-draft-tokens) are not available for it",
                               dflash2 ? "DFlash2 drafter"
                                       : (dspine ? "DSpine drafter" : "MTP block"));
        }
    }
    if (executor.rope_yarn.active()) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                           "Qwen3.8-Flash-Next: YaRN at factor %.4g past the native %u positions",
                           double(executor.rope_yarn.factor), executor.rope_yarn.native_context);
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
    out.load.rope                     = rope.summary;
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
    // Its images and video units, by position: those past the prefix it resumes from are encoded
    // at its first prompt step.
    std::vector<MediaMark> media;
    // CacheRetentionHint::Disposable: no request continues where this one ends (a decision read),
    // so the store keeps no image of its end.
    bool disposable = false;
    // Engine::submit_branches: the prompt is the branches' shared prefix, and once it is in, each
    // branch runs from the snapshot at its end and its first token is read. branch_fed counts the
    // current branch's tokens fed, branch_results the branches read; the first branch's token and
    // record are what the content publishes.
    std::shared_ptr<const std::vector<runtime::ReadBranch>> branches;
    std::uint32_t branch_fed = 0;
    std::vector<BranchResult> branch_results;
    TokenId branch_token = 0;
    runtime::RawTokenLogprob branch_record;
    // Prefixes other prompts are expected to share (a declared cache boundary, the end of the
    // leading system messages or tool declarations), ascending and below anchor_at: with the context
    // cache on, the state at each goes to the store for them, where a prompt step ends or, with
    // layer-major spans, as a step passes (prompt_step).
    std::vector<std::uint32_t> shared_at;

    // The worker's progress with an admitted request.
    std::uint32_t slot      = 0;
    std::uint32_t reused    = 0; // prompt tokens its sequence already held
    PrefixReusePath reuse_path = PrefixReusePath::Root; // where they came from
    std::uint32_t prefilled = 0; // prompt tokens its sequence holds, the reused ones included
    std::uint32_t prefill_skips = 0; // prefill steps other prompts ran since its last one
    // With a grammar (structured output): the mask of the token it samples next, computed off the
    // worker. A decode step runs without waiting for it: when the mask is late, the row's logits
    // are kept (saved_logits) and the token is sampled once the mask is ready.
    bool constrained = false;
    std::shared_ptr<runtime::GrammarMaskJob> mask;
    runtime::GrammarMaskCost mask_cost; // whether its masks are cheap enough to speculate
    bool awaiting_sample          = false;
    // A stop string's match began at the first token the next step feeds: the step first keeps
    // the sequence's state (Slot::hold). In a speculative round the sequence keeps the verified
    // tokens before that one only (commit_tokens), whether the match ends the turn there or is
    // still open; the next step feeds the rest. And the turn ended at a stop string whose match
    // began in an earlier round: the sequence goes back to the kept state after the round commits.
    bool hold_wanted = false;
    bool stop_rewind = false;
    std::optional<std::uint32_t> commit_tokens;
    std::int32_t awaiting_purpose = 0;
    // Its last accepted token, which the progress watchdog measures a decoding request from.
    Clock::time_point progress;
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
    // prefilled, which a later prompt that starts with that one resumes from; each with the media
    // units among those tokens.
    struct Slot {
        std::shared_ptr<Request> request;
        std::vector<TokenId> fed;
        std::vector<TokenId> anchor;
        // runtime::lookup_draft's copy source, as its distance from the end of `fed`.
        std::size_t lookup_lag = 0;
        std::vector<MediaMark> media, anchor_media;
        models::qwen4_exp::SequenceSnapshot snapshot;
        // The sequence's state after its first hold_tokens tokens, where a stop string's match
        // began in the request it serves (0: none): the visible text of a turn the stop string
        // ends continues that state, not the live one, which has read the tokens the cut removed.
        models::qwen4_exp::SequenceSnapshot hold;
        std::uint32_t hold_tokens = 0;
        std::uint64_t last_used = 0;
        bool disposable         = false; // its last request's
    };

    // One row of a sampling call: the request's parameters at its position.
    struct SampleRow {
        const ResolvedSamplingParameters* params = nullptr;
        std::uint32_t position                   = 0;
        std::uint32_t slot                       = 0;
        bool counts                              = false;
        bool logprobs                            = false;
        const std::uint32_t* mask                = nullptr; // a grammar's next-token mask
        // The typed readout's token ids, on a request's first generated token only.
        const std::vector<TokenId>* readout = nullptr;
    };

    Qwen4ExpInstance& instance;
    DeviceContext& device;
    const std::uint32_t max_context;
    const std::size_t max_outstanding;
    const std::chrono::milliseconds pending_timeout;
    const std::uint32_t domain;
    const bool structured_output;
    const bool reuse_prefixes;
    bool holds_ready = false; // every slot's hold snapshot has its buffers
    const std::uint32_t drafts; // MTP drafts a speculative round proposes; 0 without speculation
    const std::uint32_t lookup_ngram; // --lookup-ngram: tokens a context lookup matches; 0 is off
    // Under the automatic drafting with a CPU share of host experts' misses, the widest
    // verification a round runs: the widest call whose misses the CPU shares. 0: no such limit.
    const std::uint32_t round_tokens;
    const DiagnosticObserver diagnostics;
    // With --adaptive-mtp, the steps a round drafts and verifies: from one draft, since each draft
    // is a step of the MTP block of its own here, up to `drafts`.
    std::optional<runtime::MtpAdaptiveBatchController> mtp_controller;
    runtime::MtpPackedPlanner packed_planner;

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
    // The tokens each decoding request has fed on average since the last prefill step (a plain
    // round one, a speculative round its committed run), and the rounds of one token due before a
    // chunk's worth of prompt while others decode.
    double decode_run = 0;
    std::uint32_t decode_rounds_per_prefill = 1;

    // Head-device sampling planes, a row per slot; with structured output the grammars' token
    // bitmasks too. The host side is staged in pinned memory: configs, positions, sampled tokens.
    DeviceBuffer sample_config, sample_position, sample_out, token_counts, score_targets, score_out;
    DeviceBuffer token_mask;
    // A row's logits kept while its grammar mask is late (one per sequence), and those of the rows
    // sampled together afterwards, side by side.
    DeviceBuffer saved_logits, deferred_logits;
    std::size_t vocab_rows = 0; // logits per row
    // The logprob gather of a sampling call whose rows ask for it: [kLogprobTopK, rows] ids and
    // values, [rows] log-sum-exps and the gather's enabling flag, with the host copy.
    DeviceBuffer logprob_ids, logprob_values, logprob_lse, logprob_flag;
    DeviceBuffer readout_ids, readout_values;
    std::unique_ptr<PinnedHostBuffer> host_readout;
    std::unique_ptr<PinnedHostBuffer> host_mask, host_sample, host_logprobs;
    std::unique_ptr<WorkspaceArena> sample_workspace;
    // A speculative round's acceptance on the head device, a row per verified sequence: its
    // drafts, the verification's greedy targets, the live draft count and the tokens fed before
    // the round, and the licensed run (tokens, count, accepted drafts, last token); and every
    // slot's token counts before the round, which a round that keeps fewer tokens than it
    // licensed restores.
    DeviceBuffer spec_drafts, spec_targets, spec_extents, spec_lengths, spec_licensed, spec_counts,
        spec_accepted, spec_anchors, spec_columns, counts_backup;
    std::unique_ptr<PinnedHostBuffer> host_spec;

    // The context cache past the sequences: images of the prefixes they give up (the state a
    // turn closed at, and where a sequence ended), in host memory up to a budget; the least
    // recently used go to files under the disk path when it is set (where a restart finds them)
    // or are dropped. A prompt resumes from the longest stored prefix when no free sequence
    // serves more of it.
    struct Stored {
        std::vector<TokenId> tokens;
        std::vector<MediaMark> media;
        models::qwen4_exp::SequenceImage header;
        PrefixReusePath path = PrefixReusePath::PrivateEndpoint;
        std::vector<std::byte*> pages; // slices of host_chunks; empty while only on disk
        std::filesystem::path file;    // empty unless on disk
        std::uint64_t bytes     = 0;
        std::uint64_t last_used = 0;
    };

    // Shorter prefixes prefill faster than their recurrent state (about 114 MB) copies back.
    static constexpr std::size_t kMinStoredTokens = 128;
    std::vector<Stored> stored;
    // The images live in pinned pages that stay pinned once the store has needed them: pinning is
    // slow where the experts and the page cache fill the host's memory (1.1 s for the image of a
    // 32K-token prompt on an RTX 4090 host, while the next prompt waited; pageable memory faulted
    // in as slowly, and huge pages took minutes), so later images reuse the pages of those the
    // store let go. host_used counts the pages images hold, within host_budget. Startup pins
    // host_reserve of free pages; while the worker has nothing to run, it pins them back up to
    // host_reserve a chunk at a time (pin_ahead), so the next request seldom pins its own.
    static constexpr std::size_t kHostPageBytes = std::size_t(8) << 20;
    std::vector<std::unique_ptr<HostPages>> host_chunks;
    std::vector<std::byte*> free_pages;
    std::uint64_t host_budget = 0, host_used = 0, host_reserve = 0;
    // A chunk pinned while idle: a request that arrives meanwhile waits for it to finish. Pinning
    // starts once the worker has been idle for the grace period, so a client that sends its next
    // request at once does not wait. A failed pin ends idle pinning; take_pages still pins.
    static constexpr std::size_t kPinAheadPages = 8;
    static constexpr auto kPinAheadGrace        = std::chrono::milliseconds(50);
    bool pin_ahead_failed                       = false;
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
          round_tokens(options.speculative.automatic ? i.executor->cpu_share_tokens() : 0),
          diagnostics(options.diagnostic_observer),
          slots(i.executor->options().sequences),
          decode_rounds_per_prefill(
              options.decode_rounds_per_prefill != 0
                  ? options.decode_rounds_per_prefill
                  : std::max<std::uint32_t>(1, i.executor->options().prefill_chunk / 64)) {
        if (drafts > 0 && options.speculative.mtp_policy == MtpDraftPolicy::Adaptive) {
            mtp_controller.emplace();
            mtp_controller->reset(drafts, 1);
        }
        for (std::size_t s = 0; s < slots.size() && s < i.snapshots.size(); ++s) {
            slots[s].snapshot = std::move(i.snapshots[s]);
        }
        i.snapshots.clear();
        holds_ready = i.holds.size() == slots.size();
        for (std::size_t s = 0; holds_ready && s < slots.size(); ++s) {
            slots[s].hold = std::move(i.holds[s]);
        }
        i.holds.clear();
        RankBinding bind(device, i.executor->head_rank());
        const std::size_t rows = slots.size();
        // A sampling call's columns per row: one, or a verification's drafts and bonus.
        const std::size_t width = std::size_t(drafts) + 1;
        if (structured_output) {
            token_mask = DeviceBuffer(rows * width * mask_words() * sizeof(std::uint32_t));
            host_mask  = std::make_unique<PinnedHostBuffer>(rows * width * mask_words() *
                                                            sizeof(std::uint32_t));
            vocab_rows      = static_cast<std::size_t>(i.executor->logits(1).ne[0]);
            saved_logits    = DeviceBuffer(rows * vocab_rows * sizeof(__nv_bfloat16));
            deferred_logits = DeviceBuffer(rows * vocab_rows * sizeof(__nv_bfloat16));
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
        readout_ids           = DeviceBuffer(rows * kMaximumReadoutTokens * sizeof(std::int32_t));
        readout_values        = DeviceBuffer(rows * kMaximumReadoutTokens * sizeof(float));
        host_readout = std::make_unique<PinnedHostBuffer>(rows * kMaximumReadoutTokens *
                                                          (sizeof(std::int32_t) + sizeof(float)));
        host_logprobs =
            std::make_unique<PinnedHostBuffer>(top * (sizeof(std::int32_t) + sizeof(float)));
        const std::int32_t enabled = 1;
        CUDA_CHECK(ninfer::copy_and_wait(logprob_flag.p, &enabled, sizeof(enabled),
                                         cudaMemcpyHostToDevice));
        std::size_t workspace = std::max<std::size_t>(
            {ops::sampling_workspace_capacity_bytes(static_cast<std::int32_t>(domain), 1,
                                                    static_cast<std::int32_t>(rows)),
             ops::logprob_topk_workspace_capacity_bytes(static_cast<std::int32_t>(domain),
                                                        static_cast<std::int32_t>(rows * width)),
             std::size_t{256}});
        if (drafts > 0) {
            workspace =
                std::max(workspace, ops::speculative_accept_coupled_drafts_workspace_capacity_bytes(
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
            spec_columns  = plane(1);
            counts_backup = DeviceBuffer(rows * domain * sizeof(std::int32_t));
            host_spec     = std::make_unique<PinnedHostBuffer>(rows * (drafts + width + 5) *
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
            // The pages of two images of a whole context (where a prompt's turn closed and where
            // its sequence ended, what a new request stores first), within a quarter of the
            // budget, pinned now rather than while the request that displaces them waits. Without
            // the memory, the store pins its pages as it needs them.
            host_reserve = std::min<std::uint64_t>(
                host_budget / 4 / kHostPageBytes * kHostPageBytes,
                2 * page_rounded(i.executor->image_bytes(max_context)));
            try {
                pin_pages(static_cast<std::size_t>(host_reserve / kHostPageBytes));
            } catch (const std::exception&) { pin_ahead_failed = true; }
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
            const DeviceWaitWatch watched(device, "Qwen3.8-Flash-Next worker");
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

    // A decoding request that accepts no token for this long fails with an explicit error rather
    // than holding its sequence, so a stall surfaces as errors. A prompt waiting for its chunks is
    // not measured: with the experts off the GPU a prompt step lasts seconds.
    static constexpr auto kProgressTimeout = std::chrono::seconds(120);
    Clock::time_point progress_checked{};

    // Starts the mask of a structured-output request's next token, from its grammar's state now.
    void schedule_mask(Request& r) {
        const auto grammar = r.output.grammar_state();
        r.mask = grammar ? runtime::GrammarMaskJob::start(*grammar, mask_words()) : nullptr;
    }

    // After a structured request's accepted tokens: one that decodes plainly starts its next mask
    // now, a step ahead; one that speculates fills its round's columns during the verification.
    void next_mask(const std::shared_ptr<Request>& request) {
        Request& r = *request;
        if (r.constrained && active(request) && !(drafts > 0 && r.mask_cost.speculates())) {
            schedule_mask(r);
        } else {
            r.mask.reset();
        }
    }

    [[nodiscard]] static bool mask_ready(const Request& r) { return !r.mask || r.mask->ready(); }

    // After its mask is ready: false (the request failed) when the grammar had no admissible token
    // or took over the budget.
    bool mask_usable(const std::shared_ptr<Request>& request) {
        const auto& job = request->mask;
        if (!job) { return true; }
        request->mask_cost.observe(job->seconds());
        const std::string problem = runtime::grammar_mask_problem(*job);
        if (problem.empty()) { return true; }
        fail(request, std::make_exception_ptr(RequestError(
                          RequestErrorKind::Unavailable, "structured output: " + problem)));
        return false;
    }

    // Fails each decoding request that has made no progress for kProgressTimeout.
    void check_progress() {
        const auto now = Clock::now();
        if (now - progress_checked < std::chrono::seconds(1)) { return; }
        progress_checked = now;
        for (const Slot& slot : slots) {
            const auto request = slot.request;
            if (!request || !request->decoding || now - request->progress < kProgressTimeout) {
                continue;
            }
            fail(request, std::make_exception_ptr(RequestError(
                              RequestErrorKind::Unavailable,
                              "inference request made no progress for " +
                                  std::to_string(kProgressTimeout.count()) + " s")));
        }
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

    // A failed model file read: every request holding a lane fails with it, their sequences drop
    // what the interrupted round left and keep nothing for the context cache, and the Engine
    // serves the queue on. The next round reads what it needs again.
    void recover_from_read_failure(std::exception_ptr error) {
        std::string what = "a model file read failed";
        try {
            std::rethrow_exception(error);
        } catch (const std::exception& failure) { what = failure.what(); }
        const auto unavailable =
            std::make_exception_ptr(RequestError(RequestErrorKind::Unavailable, what));
        std::uint32_t failed_requests = 0;
        for (std::uint32_t s = 0; s < slots.size(); ++s) {
            Slot& slot = slots[s];
            if (!slot.request) { continue; }
            // A second read failure surfaces while draining the first: the drain completes anyway.
            try {
                instance.executor->abort(s);
            } catch (const FileReadError&) { instance.executor->abort(s); }
            forget(slot);
            complete(std::exchange(slot.request, {}), unavailable);
            ++failed_requests;
        }
        publish_diagnostic(diagnostics, DiagnosticLevel::Warning,
                           "%s; failed %u running request(s), serving on", what.c_str(),
                           failed_requests);
        std::lock_guard lock(stats_mutex);
        ++stats.engine_recoveries;
    }

    // Not a request's own error: the Engine stops serving, and the cause is logged here, where it
    // is the operator's only lead.
    void fail_engine(std::exception_ptr error) {
        const auto running = std::count_if(
            slots.begin(), slots.end(), [](const Slot& slot) { return slot.request != nullptr; });
        publish_diagnostic(diagnostics, DiagnosticLevel::Error,
                           "worker failure: %s - failing %td running request(s), the Engine "
                           "stops serving",
                           exception_text(error).c_str(), running);
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
                const auto ready = [&] { return stopping || !pending.empty() || any_active(); };
                if (!ready() && pins_ahead() && !queue_cv.wait_for(lock, kPinAheadGrace, ready)) {
                    lock.unlock();
                    pin_ahead();
                    continue;
                }
                queue_cv.wait(lock, ready);
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
            bool worked = true;
            try {
                admit();
                worked = step();
                check_progress();
            } catch (const ops::CpuExpertCancelled&) {
                // Native hybrid execution is C1 and has drained every CPU task before throwing.
                // A partially advanced layer stack is discarded; it must never enter the prefix cache.
                try {
                    auto request = slots.front().request;
                    forget(slots.front());
                    instance.executor->abort(0);
                    // MTP can finish the response before its catch-up observes cancellation.
                    // The sequence still needs cleanup, but no second terminal.
                    if (request) { finish_now(request, FinishReason::Cancelled); }
                } catch (...) { fail_engine(std::current_exception()); }
            } catch (const FileReadError&) {
                // A model file read failed (expert or n-gram table bytes): the round cannot finish,
                // but the readers kept nothing the failed read would have supplied.
                try {
                    recover_from_read_failure(std::current_exception());
                } catch (...) { fail_engine(std::current_exception()); }
            } catch (...) {
                // Not a request's own error: the device may be in an unknown state, so the Engine
                // fails with every request it holds.
                fail_engine(std::current_exception());
            }
            publish_stats();
            if (!worked) {
                // Every active request waits for its grammar mask.
                std::unique_lock lock(queue_mutex);
                queue_cv.wait_for(lock, std::chrono::milliseconds(1),
                                  [&] { return stopping || !pending.empty(); });
            }
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

    // The slot's sequence holds nothing the context cache can resume from.
    static void forget(Slot& slot) {
        slot.hold_tokens = 0;
        slot.fed.clear();
        slot.anchor.clear();
        slot.media.clear();
        slot.anchor_media.clear();
    }

    // Drops every stored image of a prefix of the request's prompt.
    void drop_stored_prefixes(const Request& r) {
        for (std::size_t i = stored.size(); i-- > 0;) {
            const std::vector<TokenId>& tokens = stored[i].tokens;
            if (tokens.size() > r.prompt_tokens.size() ||
                !std::equal(tokens.begin(), tokens.end(), r.prompt_tokens.begin())) {
                continue;
            }
            give_back(stored[i]);
            if (!stored[i].file.empty()) {
                std::error_code ignored;
                const auto bytes = std::filesystem::file_size(stored[i].file, ignored);
                disk_used -= std::min<std::uint64_t>(disk_used, ignored ? 0 : bytes);
                std::filesystem::remove(stored[i].file, ignored);
            }
            stored.erase(stored.begin() + static_cast<std::ptrdiff_t>(i));
        }
        std::lock_guard lock(stats_mutex);
        stats.host_kv_occupied_bytes = host_used;
    }

    // Ends one request whose logits came out non-finite, or whose sampled or licensed token lies
    // outside the vocabulary: thrown where the request's own handler (fail) ends it alone, so the
    // other requests and the Engine go on. Before its first token the request may have computed
    // its prompt wrong or resumed from a wrong prefix, and no later logits have vouched for the
    // images it stored on the way: every stored prefix of its prompt goes, so the same prompt
    // computes again. Later the prompt's logits were finite, and its images stay.
    [[noreturn]] void throw_numerical_failure(const Request& r) {
        const std::string where =
            r.generated.empty() ? std::string("at the end of the prompt")
                                : "after " + std::to_string(r.generated.size()) + " generated tokens";
        if (r.generated.empty()) { drop_stored_prefixes(r); }
        publish_diagnostic(diagnostics, DiagnosticLevel::Error,
                           "request %llu: non-finite logits or a token outside the vocabulary %s - "
                           "ending the request",
                           static_cast<unsigned long long>(r.id), where.c_str());
        throw RequestError(RequestErrorKind::NumericalFailure,
                           "the model produced non-finite logits " + where);
    }

    // A request failed on its own: it completes with the error, and its sequence keeps nothing
    // for the context cache, since the sequence may have stopped anywhere.
    void fail(const std::shared_ptr<Request>& request, std::exception_ptr error) {
        if (!request_error(error)) { std::rethrow_exception(error); }
        Slot& slot = slots.at(request->slot);
        if (slot.request == request) {
            slot.request = nullptr;
            forget(slot);
        }
        complete(request, std::move(error));
    }

    // Tokens of `held` a prompt can start from: all of them when they are a strict prefix of the
    // prompt (its last token is always fed, since the first sample needs its logits), or the
    // whole prompt for a branched read (`whole`), whose branches bring their own, and the media
    // units among them (`held_media`) are the prompt's there; else none.
    static std::uint32_t prefix_reuse(const std::vector<TokenId>& held,
                                      const std::vector<MediaMark>& held_media,
                                      const Request& prompt, bool whole = false) {
        const auto& tokens = prompt.prompt_tokens;
        if (held.empty() || tokens.empty() || held.size() > tokens.size() - (whole ? 0 : 1) ||
            !std::equal(held.begin(), held.end(), tokens.begin())) {
            return 0;
        }
        std::size_t units = 0;
        while (units < prompt.media.size() && prompt.media[units].begin < held.size()) { ++units; }
        if (units != held_media.size() ||
            !std::equal(held_media.begin(), held_media.end(), prompt.media.begin())) {
            return 0;
        }
        return static_cast<std::uint32_t>(held.size());
    }

    [[nodiscard]] std::uint32_t reuse(const Slot& slot, const Request& prompt) const {
        if (!reuse_prefixes) { return 0; }
        return std::max(prefix_reuse(slot.fed, slot.media, prompt),
                        prefix_reuse(slot.anchor, slot.anchor_media, prompt));
    }

    // ---- The context cache's store ------------------------------------------------------------

    // The disk tier's directory for this model and execution profile: an image holds the KV in its
    // storage format and the drafter's state (the MTP block's, or the DFlash2 or DSpine ring
    // lane's) only when it runs. Verify width changes reductions; disabling PLE changes every layer's
    // recurrent/KV state. Each needs its own store.
    [[nodiscard]] std::string profile_name(const EngineOptions& options) const {
        std::string id;
        for (const std::byte b : instance.model->info().artifact_id) {
            char hex[3];
            std::snprintf(hex, sizeof(hex), "%02x", std::to_integer<unsigned>(b));
            id += hex;
        }
        return "qwen4_exp-image1-" + id + "-kv" +
               std::to_string(static_cast<int>(options.kv_cache)) +
               (options.speculative.backend == SpeculativeBackend::DFlash2 ? "-dflash2" : "") +
               (instance.model->dspine() != nullptr
                    ? "-dspine" + instance.model->dspine()->artifact_hex() + "-ring" +
                          std::to_string(models::dspine::ring_capacity(
                              instance.model->dspine()->config(), options.max_context))
                    : std::string()) +
               "-drafts" + std::to_string(drafts) + "-draft-min-p" +
               std::to_string(
                   std::bit_cast<std::uint32_t>(options.speculative.draft_min_p.value_or(0.0F))) +
               "-ngram" +
               (instance.ngram_table_digest.empty() ? std::string("0")
                                                    : instance.ngram_table_digest.substr(0, 16)) +
               "-experts-" + instance.executor->expert_execution_profile();
    }

    struct FileHeader {
        char magic[8]                = {'N', 'F', 'N', 'X', 'I', 'M', 'G', '1'};
        std::uint32_t position       = 0;
        std::uint32_t drafter_follows = 0;
        std::uint32_t context_tokens = 0;
        std::uint32_t prompt_tokens  = 0;
        std::uint32_t path           = 0;
        std::uint32_t media_units    = 0; // the marks after the prompt tokens
        std::uint64_t bytes          = 0;
    };

    static std::string file_name(const std::vector<TokenId>& tokens,
                                 const std::vector<MediaMark>& media) {
        std::uint64_t hash = 1469598103934665603ULL; // FNV-1a over the token ids and media units
        const auto mix     = [&](std::uint64_t value, int bytes) {
            for (int i = 0; i < bytes; ++i) {
                hash ^= (value >> (8 * i)) & 0xffU;
                hash *= 1099511628211ULL;
            }
        };
        for (const TokenId token : tokens) { mix(std::uint32_t(token), 4); }
        for (const MediaMark& mark : media) {
            mix(mark.begin, 4);
            mix(mark.count, 4);
            mix(mark.key, 8);
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
            image.header.drafter_follows = header.drafter_follows != 0;
            image.header.context.previous.resize(header.context_tokens);
            image.tokens.resize(header.prompt_tokens);
            image.media.resize(header.media_units);
            if (!file.read(reinterpret_cast<char*>(image.header.context.previous.data()),
                           std::streamsize(header.context_tokens) * 4) ||
                !file.read(reinterpret_cast<char*>(image.tokens.data()),
                           std::streamsize(header.prompt_tokens) * 4) ||
                !file.read(reinterpret_cast<char*>(image.media.data()),
                           std::streamsize(header.media_units) * sizeof(MediaMark))) {
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

    // The stored prefix that serves the most of `prompt`: its index and length. A drafting
    // request takes only a state its MTP block followed.
    [[nodiscard]] std::pair<std::size_t, std::uint32_t>
    find_stored(const Request& prompt, bool drafting, bool whole) const {
        std::pair<std::size_t, std::uint32_t> best{0, 0};
        for (std::size_t i = 0; i < stored.size(); ++i) {
            if (stored[i].pages.empty() && !disk_restore) { continue; }
            if (drafting && !stored[i].header.drafter_follows) { continue; }
            const std::uint32_t length =
                prefix_reuse(stored[i].tokens, stored[i].media, prompt, whole);
            if (length > best.second) { best = {i, length}; }
        }
        return best;
    }

    void drop_dead() {
        std::erase_if(stored, [](const Stored& image) {
            return image.pages.empty() && image.file.empty();
        });
    }

    // Writes an image held in host memory to the disk tier, making room by deleting the least
    // recently used files; entries left with neither copy are dropped by the caller.
    void write_file(Stored& image) {
        if (disk_dir.empty() || !image.file.empty() || image.pages.empty()) { return; }
        const std::uint64_t size = sizeof(FileHeader) +
                                   4 * (image.header.context.previous.size() + image.tokens.size()) +
                                   sizeof(MediaMark) * image.media.size() + image.bytes;
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
        header.drafter_follows = image.header.drafter_follows ? 1 : 0;
        header.context_tokens = static_cast<std::uint32_t>(image.header.context.previous.size());
        header.prompt_tokens  = static_cast<std::uint32_t>(image.tokens.size());
        header.path           = static_cast<std::uint32_t>(image.path);
        header.media_units    = static_cast<std::uint32_t>(image.media.size());
        header.bytes          = image.bytes;
        const auto path       = disk_dir / file_name(image.tokens, image.media);
        const auto temporary  = std::filesystem::path(path).concat(".tmp");
        {
            std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
            file.write(reinterpret_cast<const char*>(&header), sizeof(header));
            file.write(reinterpret_cast<const char*>(image.header.context.previous.data()),
                       std::streamsize(header.context_tokens) * 4);
            file.write(reinterpret_cast<const char*>(image.tokens.data()),
                       std::streamsize(header.prompt_tokens) * 4);
            file.write(reinterpret_cast<const char*>(image.media.data()),
                       std::streamsize(header.media_units) * sizeof(MediaMark));
            for (std::uint64_t at = 0; at < image.bytes; at += kHostPageBytes) {
                file.write(reinterpret_cast<const char*>(image.pages[at / kHostPageBytes]),
                           std::streamsize(std::min<std::uint64_t>(kHostPageBytes,
                                                                   image.bytes - at)));
            }
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

    [[nodiscard]] static std::uint64_t page_rounded(std::uint64_t bytes) {
        return (bytes + kHostPageBytes - 1) / kHostPageBytes * kHostPageBytes;
    }

    // Pins `count` more free pages in one allocation.
    void pin_pages(std::size_t count) {
        if (count == 0) { return; }
        auto chunk = std::make_unique<HostPages>(count * kHostPageBytes);
        chunk->pin();
        auto* base = static_cast<std::byte*>(chunk->data());
        for (std::size_t i = count; i-- > 0;) { free_pages.push_back(base + i * kHostPageBytes); }
        host_chunks.push_back(std::move(chunk));
    }

    // Whether the idle worker pins another chunk: the free pages hold less than host_reserve and
    // one more chunk stays within the budget.
    [[nodiscard]] bool pins_ahead() const noexcept {
        const std::uint64_t free_bytes = std::uint64_t(free_pages.size()) * kHostPageBytes;
        return !pin_ahead_failed && free_bytes < host_reserve &&
               host_used + free_bytes + kPinAheadPages * kHostPageBytes <= host_budget;
    }

    void pin_ahead() {
        try {
            pin_pages(kPinAheadPages);
        } catch (const std::exception&) { pin_ahead_failed = true; }
    }

    // Pages for an image of `bytes`, pinning more when the free ones fall short; the caller has
    // made room for them (make_host_room), so the pinned pages stay within the budget. Empty when
    // more pages cannot be pinned: registering them also takes device memory on every card for
    // their page tables (2 MiB per GiB), which the expert cache may already hold.
    std::vector<std::byte*> take_pages(std::uint64_t bytes) {
        const auto count = static_cast<std::size_t>(page_rounded(bytes) / kHostPageBytes);
        if (free_pages.size() < count) {
            try {
                pin_pages(count - free_pages.size());
            } catch (const std::exception&) {
                return {};
            }
        }
        std::vector<std::byte*> pages(free_pages.end() - std::ptrdiff_t(count), free_pages.end());
        free_pages.resize(free_pages.size() - count);
        host_used += count * kHostPageBytes;
        return pages;
    }

    void give_back(Stored& image) {
        host_used -= image.pages.size() * kHostPageBytes;
        free_pages.insert(free_pages.end(), image.pages.begin(), image.pages.end());
        image.pages.clear();
    }

    // Brings the images in host pages within the budget with `bytes` more (whole pages): the
    // least recently used go to the disk tier, or are dropped without one. False when the budget
    // cannot hold `bytes` at all. Entries may be dropped, so indices into `stored` change.
    bool make_host_room(std::uint64_t bytes) {
        if (bytes > host_budget) { return false; }
        while (host_used + bytes > host_budget) {
            std::optional<std::size_t> oldest;
            for (std::size_t i = 0; i < stored.size(); ++i) {
                if (!stored[i].pages.empty() &&
                    (!oldest || stored[i].last_used < stored[*oldest].last_used)) {
                    oldest = i;
                }
            }
            if (!oldest) { return false; }
            write_file(stored[*oldest]);
            give_back(stored[*oldest]);
            drop_dead();
        }
        drop_dead();
        return true;
    }

    [[nodiscard]] std::optional<std::size_t>
    stored_index(const std::vector<TokenId>& tokens, const std::vector<MediaMark>& media) const {
        for (std::size_t i = 0; i < stored.size(); ++i) {
            if (stored[i].tokens == tokens && stored[i].media == media) { return i; }
        }
        return std::nullopt;
    }

    // Keeps the state of sequence `s` up to `tokens`, with the media units `media` among them
    // (its live state, or the snapshot `at`), in the store, unless the store already holds that
    // prefix or it is too short to be worth it.
    void store_image(std::uint32_t s, const std::vector<TokenId>& tokens,
                     const std::vector<MediaMark>& media,
                     const models::qwen4_exp::SequenceSnapshot* at, PrefixReusePath path) {
        if (tokens.size() < kMinStoredTokens || host_budget == 0) { return; }
        if (const auto known = stored_index(tokens, media)) {
            stored[*known].last_used = ++use_clock;
            return;
        }
        auto& executor      = *instance.executor;
        const auto position = at != nullptr ? at->position : executor.position(s);
        if (position != tokens.size()) { return; } // the state holds other tokens
        const std::uint64_t bytes = executor.image_bytes(position);
        if (!make_host_room(page_rounded(bytes))) { return; }
        Stored image;
        image.tokens = tokens;
        image.media  = media;
        image.path   = path;
        image.bytes  = bytes;
        image.pages  = take_pages(bytes);
        if (image.pages.empty()) { return; }
        try {
            image.header = executor.save_image(s, at, image.pages, kHostPageBytes);
        } catch (...) {
            give_back(image);
            throw;
        }
        image.last_used = ++use_clock;
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
            store_image(s, slot.anchor, slot.anchor_media, &slot.snapshot,
                        PrefixReusePath::PrivateTurnClosure);
        }
        if (slot.fed != slot.anchor && !slot.disposable) {
            store_image(s, slot.fed, slot.media, nullptr, PrefixReusePath::PrivateEndpoint);
        }
    }

    // Loads the stored image of `tokens` and `media` into sequence `s`, reading it from the disk
    // tier when it is only there; false when the store no longer holds it.
    bool restore_stored(std::uint32_t s, const std::vector<TokenId>& tokens,
                        const std::vector<MediaMark>& media) {
        auto index = stored_index(tokens, media);
        if (!index) { return false; }
        if (stored[*index].pages.empty()) {
            if (!make_host_room(page_rounded(stored[*index].bytes)) ||
                !(index = stored_index(tokens, media))) {
                return false;
            }
            Stored& image = stored[*index];
            image.pages   = take_pages(image.bytes);
            if (image.pages.empty()) { return false; }
            std::ifstream in(image.file, std::ios::binary);
            in.seekg(std::streamoff(sizeof(FileHeader) +
                                    4 * (image.header.context.previous.size() + tokens.size()) +
                                    sizeof(MediaMark) * media.size()));
            for (std::uint64_t at = 0; in && at < image.bytes; at += kHostPageBytes) {
                in.read(reinterpret_cast<char*>(image.pages[at / kHostPageBytes]),
                        std::streamsize(std::min<std::uint64_t>(kHostPageBytes, image.bytes - at)));
            }
            if (!in) {
                give_back(image);
                return false;
            }
        }
        Stored& image = stored[*index];
        instance.executor->load_image(s, image.header, image.pages, kHostPageBytes);
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
                    const Request& prompt = *pending.front();
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

    // Whether the request drafts: ExecutionOptions::speculative_drafting off (a decision read)
    // turns the MTP block off for it.
    [[nodiscard]] bool drafts_for(const Request& r) const {
        return drafts > 0 && r.options.execution.speculative_drafting;
    }

    // A request that does not draft runs its prompt without the MTP block, whose catch-up over
    // each prompt step it would never use; its sequence's snapshots and images carry no MTP state.
    void skip_drafter(const Request& r) {
        if (drafts > 0 && !r.options.execution.speculative_drafting) {
            instance.executor->stop_drafting(r.slot);
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
            // A drafting request resumes only from a state its MTP block followed: a read's
            // prompt passes skip the block (stop_drafting).
            const bool drafting          = drafts_for(r);
            const bool whole             = r.branches != nullptr;
            const std::uint32_t live =
                drafting && !executor.can_draft(s) ? 0 : prefix_reuse(slot.fed, slot.media, r, whole);
            const std::uint32_t anchored =
                drafting && !slot.snapshot.drafter_follows
                    ? 0
                    : prefix_reuse(slot.anchor, slot.anchor_media, r, whole);
            const auto [image, from_store] = find_stored(r, drafting, whole);
            if (from_store > std::max(live, anchored)) {
                const std::vector<TokenId> tokens   = stored[image].tokens;
                const std::vector<MediaMark> media = stored[image].media;
                const PrefixReusePath path          = stored[image].path;
                retire(s);
                if (restore_stored(s, tokens, media)) {
                    forget(slot);
                    slot.fed     = tokens;
                    slot.media   = media;
                    reused       = from_store;
                    r.reuse_path = path;
                }
            } else if (anchored > live) {
                retire(s);
                executor.restore(s, slot.snapshot);
                slot.fed     = slot.anchor;
                slot.media   = slot.anchor_media;
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
            // The snapshot's prefix is about to be overwritten.
            forget(slot);
        }
        slot.request     = request;
        slot.hold_tokens = 0;
        slot.last_used   = ++use_clock;
        slot.disposable  = r.disposable;
        r.reused = r.prefilled = reused;
        skip_drafter(r);
        if (drafts_for(r)) {
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
        r.constrained          = r.output.grammar_state() != nullptr;
        if (r.constrained) { schedule_mask(r); } // the first token's, while the prompt prefills
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
            return !r.media.empty() && r.prefilled > r.reused;
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

    // One prompt step or one decode round; false when there was nothing to run (every active
    // request waits for its grammar mask).
    bool step() {
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
                // A request runs a decode step unless its kept logits still wait for their mask.
                decoding = decoding || !slot.request->awaiting_sample || mask_ready(*slot.request);
            } else if (!prefilling || prefills_before(*slot.request, *prefilling)) {
                prefilling = slot.request;
            }
        }
        // A prompt's chunk takes far longer than a decode round, so while others decode a prefill
        // step waits for decode_rounds_per_prefill rounds per chunk it computes: proportionally
        // fewer, at least one, for a shorter step, so a short prompt gets in after one round, and
        // more for a step of several chunks, so the streams keep their share of the GPU. A
        // speculative round counts as the tokens its requests fed on average, so drafting gives
        // the streams the tokens plain rounds would and holds a prompt back no longer.
        if (prefilling &&
            (!decoding ||
             decode_run >=
                 double(decode_rounds_before(prompt_step(*prefilling, true).tokens)))) {
            decode_run = 0;
            for (const Slot& slot : slots) {
                if (slot.request && !slot.request->decoding && slot.request != prefilling &&
                    slot.request->prefill_skips != UINT32_MAX) {
                    ++slot.request->prefill_skips;
                }
            }
            prefilling->prefill_skips = 0;
            const std::uint32_t lead_tokens = prompt_step(*prefilling, true).tokens;
            std::uint32_t tokens            = 0;
            try {
                if (executor.packs_prompts() && plain_prompt(*prefilling)) {
                    tokens = packed_prefill_step(prefilling);
                } else {
                    prefill_step(prefilling);
                }
            } catch (...) { fail(prefilling, std::current_exception()); }
            // A packed step computed more than the lead's tokens, which its wait was sized for:
            // the next prefill step waits for the rest of their decode rounds too.
            if (decoding && tokens > lead_tokens) {
                decode_run = -double(decode_rounds_before(tokens) - decode_rounds_before(lead_tokens));
            }
            return true;
        }
        if (!decoding) { return false; }
        decode_run += decode_step();
        return true;
    }

    [[nodiscard]] std::uint32_t decode_rounds_before(std::uint32_t tokens) const {
        const std::uint64_t chunk = std::max<std::uint32_t>(
            instance.executor->options().prefill_chunk, 1);
        return static_cast<std::uint32_t>(std::max<std::uint64_t>(
            1, (std::uint64_t(decode_rounds_per_prefill) * tokens + chunk - 1) / chunk));
    }

    // A prompt step: the tokens it computes, and the shared frontier inside it whose state it
    // keeps in the slot's snapshot as it passes (Executor::forward's cut; 0: none).
    struct PromptStep {
        std::uint32_t tokens = 0;
        std::uint32_t cut    = 0;
    };

    // The request's next prefill step: a span where the executor has them, else a chunk. A span
    // is one step, which with the experts off the GPU lasts seconds, and the requests that are
    // decoding wait for all of it: while any decodes, a step is two chunks at most. With `cuts`
    // a span may pass a shared frontier (a step that runs alone).
    [[nodiscard]] PromptStep prompt_step(const Request& r, bool cuts) const {
        const auto& executor             = *instance.executor;
        const auto prompt_n              = static_cast<std::uint32_t>(r.prompt_tokens.size());
        if (r.branches && r.prefilled == prompt_n) {
            const std::size_t left =
                (*r.branches)[r.branch_results.size()].tokens.size() - r.branch_fed;
            return {static_cast<std::uint32_t>(
                std::min<std::size_t>(left, executor.prompt_step()))};
        }
        // A step ends where the context cache keeps the state: with host or disk experts every
        // step's wide pass copies the experts, so a prompt the cache does not keep runs whole.
        // `after` is where the cache keeps the state next past the first frontier ahead.
        std::uint32_t until = prompt_n, frontier = 0, after = prompt_n;
        if (reuse_prefixes && r.reusable) {
            if (r.prefilled < r.anchor_at) { until = r.anchor_at; }
            after = until;
            for (std::size_t i = 0; i < r.shared_at.size(); ++i) {
                if (r.shared_at[i] > r.prefilled) {
                    frontier = r.shared_at[i];
                    if (i + 1 < r.shared_at.size()) { after = std::min(after, r.shared_at[i + 1]); }
                    break;
                }
            }
        }
        const std::uint32_t chunk_tokens = executor.options().prefill_chunk;
        std::uint32_t step               = executor.prompt_step();
        const bool peers_decoding =
            std::any_of(slots.begin(), slots.end(), [&](const Slot& other) {
                return other.request && other.request.get() != &r && other.request->decoding;
            });
        if (peers_decoding) { step = std::min(step, kPeerSpanChunks * chunk_tokens); }
        if (frontier != 0 && frontier < until) {
            // A span goes on past the frontier with the chunks the steps from there would run
            // (to `after`, or whole chunks), so that the experts cross the bus once for both
            // sides, and keeps the state at the frontier as it passes. Only where each side
            // starts with a chunk wider than the narrow ones: a step of narrow chunks copies only
            // the experts it misses, so passing the frontier saves no copy there, and a span
            // that passed one into a narrow chunk measured slower than the two steps.
            const std::uint32_t narrow = executor.narrow_prompt_tokens();
            if (cuts && narrow != 0 && !r.branches && frontier - r.prefilled < step &&
                frontier - r.prefilled > narrow) {
                const std::uint32_t room = r.prefilled + step - frontier;
                const std::uint32_t end  = after - frontier <= room
                                               ? after
                                               : frontier + room / chunk_tokens * chunk_tokens;
                if (end - frontier > narrow) { return {end - r.prefilled, frontier}; }
            }
            until = frontier;
        }
        return {std::min(step, until > r.prefilled ? until - r.prefilled : 0U)};
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

    // Samples one token per row from the first rows.size() columns of the head logits (or of
    // `source`); a grammar's mask restricts its row to the tokens its state allows. A row that asks
    // for logprobs gets the sampled token's record in `logprobs`, under the distribution its token
    // was drawn from.
    void sample(std::span<const SampleRow> rows, std::int32_t purpose, std::span<TokenId> out,
                std::span<runtime::RawTokenLogprob> logprobs, const Tensor* source = nullptr) {
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
            config.draw              = row.params->gumbel ? ops::kSampleDrawGumbel
                                                          : ops::kSampleDrawInverseCdf;
            config.token_counts =
                row.counts ? static_cast<std::int32_t*>(token_counts.p) + std::size_t(row.slot) * domain
                           : nullptr;
            if (row.mask != nullptr) {
                auto* words = static_cast<std::uint32_t*>(host_mask->data()) + b * mask_words();
                std::copy_n(row.mask, mask_words(), words);
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
        const Tensor logits =
            source != nullptr ? *source : instance.executor->logits(static_cast<std::uint32_t>(n));
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
        auto* readout_host_ids = static_cast<std::int32_t*>(host_readout->data());
        auto* readout_host_values =
            reinterpret_cast<float*>(readout_host_ids + slots.size() * kMaximumReadoutTokens);
        const bool read = std::any_of(rows.begin(), rows.end(),
                                      [](const SampleRow& row) { return row.readout != nullptr; });
        if (read) {
            for (std::size_t b = 0; b < n; ++b) {
                if (rows[b].readout == nullptr) { continue; }
                std::copy(rows[b].readout->begin(), rows[b].readout->end(),
                          readout_host_ids + b * kMaximumReadoutTokens);
            }
            CUDA_CHECK(cudaMemcpyAsync(readout_ids.p, readout_host_ids,
                                       n * kMaximumReadoutTokens * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, stream));
            for (std::size_t b = 0; b < n; ++b) {
                if (rows[b].readout == nullptr) { continue; }
                const auto offset = b * kMaximumReadoutTokens;
                const Tensor ids(static_cast<std::int32_t*>(readout_ids.p) + offset, DType::I32,
                                 {static_cast<std::int32_t>(kMaximumReadoutTokens)});
                Tensor values(static_cast<float*>(readout_values.p) + offset, DType::FP32,
                              {static_cast<std::int32_t>(kMaximumReadoutTokens)});
                ops::readout_logprobs(logits.slice(1, static_cast<std::int32_t>(b), 1), ids,
                                      static_cast<std::int32_t>(rows[b].readout->size()),
                                      static_cast<std::int32_t>(domain), values, stream);
            }
            CUDA_CHECK(cudaMemcpyAsync(readout_host_values, readout_values.p,
                                       n * kMaximumReadoutTokens * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
        }
        {
            auto scope = sample_workspace->scope();
            ops::sample(logits, sampled, static_cast<std::int32_t>(domain), device_configs,
                        logical, purpose, *sample_workspace, stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(tokens, sample_out.p, n * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(ninfer::wait_for_stream(stream));
        instance.executor->raise_service_failure();
        for (std::size_t b = 0; b < n; ++b) {
            std::int32_t token = tokens[b];
            if (b == 0 && runtime::consume_armed_numerical_fault(
                              purpose == ops::kSamplePurposePrefill
                                  ? runtime::armed_prefill_numerical_faults
                                  : runtime::armed_decode_numerical_faults)) {
                token = ops::kSamplerNonFiniteToken; // the verification seam of worker_fault.h
            }
            if (token < 0 || std::uint32_t(token) >= domain) {
                // Non-finite logits (the sampler's marker) or an id outside the vocabulary: the
                // row's request ends where it takes the token (throw_numerical_failure).
                out[b] = ops::kSamplerNonFiniteToken;
                continue;
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
            if (rows[b].readout != nullptr) {
                record.readout_count = static_cast<std::uint32_t>(rows[b].readout->size());
                std::copy_n(readout_host_values + b * kMaximumReadoutTokens, record.readout_count,
                            record.readout_values.begin());
            }
        }
    }

    [[nodiscard]] SampleRow sample_row(Request& r) {
        const auto& exec = r.options.execution;
        if (!r.post_thinking && exec.post_thinking_sampling && r.output.reasoning_closed()) {
            r.post_thinking = true;
        }
        return SampleRow{.params = r.post_thinking ? &*exec.post_thinking_sampling : &exec.sampling,
                         .position = r.position,
                         .slot     = r.slot,
                         .counts   = r.penalties,
                         .logprobs = exec.logprobs,
                         .mask     = r.mask ? r.mask->masks() : nullptr,
                         .readout  = exec.logprobs && !exec.readout_tokens.empty() &&
                                             r.first_token == Clock::time_point{}
                                         ? &exec.readout_tokens
                                         : nullptr};
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
        result.branches                        = std::move(r.branch_results);
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
        slots.at(r.slot).request = nullptr;
        complete(request, std::move(result));
    }

    void finish_now(const std::shared_ptr<Request>& request, FinishReason reason) {
        (void)request->output.preview_terminal(reason);
        push_events(*request, request->output.commit_preview(), std::nullopt);
        finish(request, reason);
    }

    // Readies the request's next prompt step: a cancelled request finishes; a prompt admitted with
    // nothing to resume may find a prefix another request stored while it waited (the shared state
    // of reads that arrived together) and starts there; a branched read past its prompt runs its
    // branch step, and one whose last prompt step opens its branches runs both; a media prompt
    // runs its tower. The step (prompt_step with `cuts`), or nothing when the step has run or
    // ended the request.
    std::optional<PromptStep> begin_prefill(const std::shared_ptr<Request>& request, bool cuts) {
        Request& r          = *request;
        Slot& slot          = slots[r.slot];
        auto& executor      = *instance.executor;
        const auto prompt_n = static_cast<std::uint32_t>(r.prompt_tokens.size());
        if (r.cancelled.load(std::memory_order_acquire)) {
            finish_now(request, FinishReason::Cancelled);
            return std::nullopt;
        }
        if (reuse_prefixes && r.reusable && r.prefilled == 0) {
            const auto [image, from_store] = find_stored(r, drafts_for(r), r.branches != nullptr);
            if (from_store > 0) {
                const std::vector<TokenId> tokens   = stored[image].tokens;
                const std::vector<MediaMark> media = stored[image].media;
                const PrefixReusePath path          = stored[image].path;
                if (restore_stored(r.slot, tokens, media)) {
                    forget(slot);
                    slot.fed     = tokens;
                    slot.media   = media;
                    r.reuse_path = path;
                    r.reused = r.prefilled = from_store;
                    skip_drafter(r);
                    std::lock_guard lock(stats_mutex);
                    stats.reused_prompt_tokens += from_store;
                }
            }
        }
        if (r.branches && r.prefilled == prompt_n) {
            branch_step(request);
            return std::nullopt;
        }
        if (!r.media.empty() && r.prefilled == r.reused) {
            // The tower encodes the media past the prefix the prompt resumes from. Its embeddings
            // stay until another prompt's media replace them; prompts with media prefill one at a
            // time, so this one is done first.
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
            if (executor.set_media(r.slot, items, data.positions, data.rope_delta) > 0) {
                r.vision_seconds = seconds(vision_start, Clock::now());
            }
        }
        const PromptStep step = prompt_step(r, cuts);
        if (opens_branches(r, step.tokens)) {
            prompt_branch_pass(request, step.tokens);
            return std::nullopt;
        }
        return step;
    }

    // Books a prompt step of `n` tokens that ran for the request, its last token's logits column
    // `column` of logits(rows): the context cache's images at its frontiers (at `cut`, a frontier
    // it passed, from the slot's snapshot), its progress, and at the prompt's end its first token.
    void end_prefill(const std::shared_ptr<Request>& request, std::uint32_t n,
                     std::uint32_t column, std::uint32_t rows, std::uint32_t cut = 0) {
        Request& r          = *request;
        Slot& slot          = slots[r.slot];
        auto& executor      = *instance.executor;
        const auto prompt_n = static_cast<std::uint32_t>(r.prompt_tokens.size());
        const auto chunk    = std::span<const TokenId>(r.prompt_tokens).subspan(r.prefilled, n);
        slot.fed.insert(slot.fed.end(), chunk.begin(), chunk.end());
        for (const MediaMark& mark : r.media) {
            if (mark.begin >= r.prefilled && mark.begin < r.prefilled + n) {
                slot.media.push_back(mark);
            }
        }
        r.prefilled += n;
        if (cut != 0) {
            const std::vector<TokenId> prefix(slot.fed.begin(), slot.fed.begin() + cut);
            std::vector<MediaMark> media;
            bool whole = true; // no media unit straddles the frontier
            for (const MediaMark& mark : slot.media) {
                if (mark.begin >= cut) { continue; }
                whole = whole && mark.begin + mark.count <= cut;
                media.push_back(mark);
            }
            if (whole) {
                store_image(r.slot, prefix, media, &slot.snapshot,
                            PrefixReusePath::SharedStablePrefix);
            }
        }
        if (reuse_prefixes && r.reusable &&
            std::binary_search(r.shared_at.begin(), r.shared_at.end(), r.prefilled)) {
            store_image(r.slot, slot.fed, slot.media, nullptr, PrefixReusePath::SharedStablePrefix);
        }
        if (reuse_prefixes && r.reusable && r.prefilled == r.anchor_at) {
            executor.snapshot(r.slot, slot.snapshot);
            slot.anchor       = slot.fed;
            slot.anchor_media = slot.media;
        }
        if (r.observation.prompt_progress) {
            CUDA_CHECK(ninfer::wait_for_stream(executor.head_stream()));
            observe_prompt(r);
        }
        if (r.prefilled < prompt_n) { return; }
        if (r.branches) { return; } // the next steps run the branches
        finish_prompt(request, column, rows);
    }

    // Feeds the next chunk of a prompt, which ends at the anchor when one lies ahead, where the
    // sequence's state is kept for the context cache; at the prompt's end samples the first token.
    void prefill_step(const std::shared_ptr<Request>& request) {
        const std::optional<PromptStep> step = begin_prefill(request, true);
        if (step) { feed_prompt(request, *step); }
    }

    // Runs the request's prompt step alone.
    void feed_prompt(const std::shared_ptr<Request>& request, PromptStep step) {
        Request& r          = *request;
        Slot& slot          = slots[r.slot];
        auto& executor      = *instance.executor;
        const auto prompt_n = static_cast<std::uint32_t>(r.prompt_tokens.size());
        const auto prompt   = std::span<const TokenId>(r.prompt_tokens);
        const std::uint32_t n = step.tokens;
        // The next step's table rows are read while this one runs.
        const auto next = prompt.subspan(
            r.prefilled + n, std::min(executor.prompt_step(), prompt_n - r.prefilled - n));
        models::qwen4_exp::PromptCut cut;
        if (step.cut != 0) {
            // The slot's snapshot takes the state at the frontier: the anchor it held goes.
            slot.anchor.clear();
            slot.anchor_media.clear();
            cut = {.tokens = step.cut - r.prefilled, .state = &slot.snapshot};
        }
        // Only the prompt's last step leaves logits (end_prefill samples them).
        executor.forward(r.slot, prompt.subspan(r.prefilled, n), r.prefilled + n < prompt_n ? 0 : 1,
                         next, cut);
        end_prefill(request, n, 0, 1, step.cut);
    }

    // A prompt without branches or media, whose steps may share a span with other prompts'.
    [[nodiscard]] static bool plain_prompt(const Request& r) {
        return !r.branches && r.media.empty();
    }

    // A prompt step of `lead` (a plain prompt) that also takes the steps of other plain prompts,
    // in prefills_before order, as long as they fit one span (two chunks while others decode):
    // their chunks reach each layer together, so its uncached experts cross the bus once for all
    // of them. A step that does not end its prompt takes whole chunks, so every prompt runs the
    // chunks, and each piece the arithmetic, it would alone. A prompt that could start from the
    // prefix a packed prompt is about to store waits for it. Returns the tokens computed.
    std::uint32_t packed_prefill_step(const std::shared_ptr<Request>& lead) {
        auto& executor = *instance.executor;
        const std::optional<PromptStep> lead_step = begin_prefill(lead, true);
        if (!lead_step) { return 0; }
        // A step that passes a frontier runs alone, its state there kept in the slot's snapshot.
        if (lead_step->cut != 0) {
            feed_prompt(lead, *lead_step);
            return lead_step->tokens;
        }
        const std::uint32_t lead_n = lead_step->tokens;
        const std::uint32_t chunk = executor.options().prefill_chunk;
        const bool decoding       = std::any_of(slots.begin(), slots.end(), [](const Slot& slot) {
            return slot.request && slot.request->decoding;
        });
        const std::uint32_t budget =
            decoding ? std::min(executor.prompt_step(), kPeerSpanChunks * chunk)
                     : executor.prompt_step();
        struct Packed {
            std::shared_ptr<Request> request;
            std::uint32_t tokens = 0;
        };
        std::vector<Packed> packed{{lead, lead_n}};
        std::uint32_t total = lead_n;
        std::vector<std::shared_ptr<Request>> candidates;
        for (const Slot& slot : slots) {
            if (slot.request && slot.request != lead && !slot.request->decoding &&
                plain_prompt(*slot.request)) {
                candidates.push_back(slot.request);
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) { return prefills_before(*a, *b); });
        const auto shares_store = [&](const Request& r) {
            if (!reuse_prefixes || !r.reusable || r.prefilled != 0 || r.shared_at.empty()) {
                return false;
            }
            const std::uint32_t frontier = r.shared_at.front();
            return std::any_of(packed.begin(), packed.end(), [&](const Packed& p) {
                const Request& q = *p.request;
                return q.prefilled < frontier && q.prompt_tokens.size() >= frontier &&
                       std::equal(r.prompt_tokens.begin(), r.prompt_tokens.begin() + frontier,
                                  q.prompt_tokens.begin());
            });
        };
        for (const auto& request : candidates) {
            if (packed.size() >= executor.max_logit_rows() || budget - total < chunk) { break; }
            std::optional<std::uint32_t> n;
            try {
                if (const auto step = begin_prefill(request, false)) { n = step->tokens; }
            } catch (...) {
                fail(request, std::current_exception());
                continue;
            }
            if (!n || *n == 0 || shares_store(*request)) { continue; }
            if (*n > budget - total) { *n = (budget - total) / chunk * chunk; }
            packed.push_back({request, *n});
            request->prefill_skips = 0;
            total += *n;
        }
        if (packed.size() == 1) {
            feed_prompt(lead, *lead_step);
            return total;
        }
        std::vector<models::qwen4_exp::PromptPiece> pieces;
        for (const Packed& p : packed) {
            pieces.push_back(
                {.sequence = p.request->slot,
                 .tokens   = std::span<const TokenId>(p.request->prompt_tokens)
                                 .subspan(p.request->prefilled, p.tokens),
                 .logits   = p.request->prefilled + p.tokens >= p.request->prompt_tokens.size()});
        }
        // The last piece's next step's table rows are read while this one runs.
        const Request& last  = *packed.back().request;
        const auto last_end  = last.prefilled + packed.back().tokens;
        const auto next      = std::span<const TokenId>(last.prompt_tokens)
                              .subspan(last_end, std::min<std::size_t>(
                                                     executor.prompt_step(),
                                                     last.prompt_tokens.size() - last_end));
        try {
            executor.forward_pieces(pieces, next);
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            for (std::size_t i = 1; i < packed.size(); ++i) { fail(packed[i].request, error); }
            throw;
        }
        const auto rows = static_cast<std::uint32_t>(packed.size());
        for (std::uint32_t i = 0; i < rows; ++i) {
            try {
                end_prefill(packed[i].request, packed[i].tokens, i, rows);
            } catch (...) { fail(packed[i].request, std::current_exception()); }
        }
        return total;
    }

    // One step of a branched read: the current branch's next tokens, from the snapshot at the
    // prompt's end when the branch begins, and once they are in, its first token read. When the
    // last branch is read, the request publishes the first branch's token and finishes.
    void branch_step(const std::shared_ptr<Request>& request) {
        Request& r                        = *request;
        Slot& slot                        = slots[r.slot];
        auto& executor                    = *instance.executor;
        const auto prompt_n               = static_cast<std::uint32_t>(r.prompt_tokens.size());
        const runtime::ReadBranch& branch = (*r.branches)[r.branch_results.size()];
        // Branches that fit a prompt chunk run in shared passes (branch_pass); a longer one runs
        // alone, chunk by chunk.
        const bool shared = r.branch_fed == 0 && executor.runs_branches() &&
                            branch.tokens.size() <= executor.options().prefill_chunk;
        // A branch starts from the state at the prompt's end: the sequence's own at the first
        // branch, kept in its snapshot (the anchor) for the others; a shared pass restores it.
        if (r.branch_fed == 0) {
            if (slot.fed == r.prompt_tokens) {
                if (slot.anchor != slot.fed) {
                    executor.snapshot(r.slot, slot.snapshot);
                    slot.anchor       = slot.fed;
                    slot.anchor_media = slot.media;
                }
            } else {
                if (slot.anchor != r.prompt_tokens) {
                    throw std::logic_error("a branched read lost the state at its prompt's end");
                }
                if (!shared) {
                    executor.restore(r.slot, slot.snapshot);
                    slot.fed   = slot.anchor;
                    slot.media = slot.anchor_media;
                }
            }
        }
        if (shared) {
            branch_pass(request);
            return;
        }
        const std::uint32_t n = prompt_step(r, false).tokens;
        const auto chunk      = std::span<const TokenId>(branch.tokens).subspan(r.branch_fed, n);
        // Only the branch's last step leaves logits, which its first token reads.
        executor.forward(r.slot, chunk, r.branch_fed + n < branch.tokens.size() ? 0 : 1);
        slot.fed.insert(slot.fed.end(), chunk.begin(), chunk.end());
        r.branch_fed += n;
        r.progress = Clock::now();
        if (r.branch_fed < branch.tokens.size()) { return; }
        SampleRow row = sample_row(r);
        row.readout   = branch.readout_tokens.empty() ? nullptr : &branch.readout_tokens;
        TokenId token = 0;
        runtime::RawTokenLogprob record;
        sample(std::span(&row, 1), ops::kSamplePurposePrefill, std::span(&token, 1),
               std::span(&record, 1));
        if (r.branch_results.empty()) {
            r.branch_token  = token;
            r.branch_record = record;
        }
        r.branch_results.push_back(BranchResult{
            .prompt_tokens = prompt_n + static_cast<std::uint32_t>(branch.tokens.size()),
            .first         = instance.frontend.token_logprob(record)});
        r.branch_fed = 0;
        if (r.branch_results.size() < r.branches->size()) { return; }
        finish_prompt(request);
    }

    // Publishes how much of the prompt is in to a request that observes its progress.
    void observe_prompt(Request& r) {
        {
            std::lock_guard lock(r.mutex);
            r.stream_progress = PromptProgress{
                .total_prompt_tokens     = static_cast<std::uint32_t>(r.prompt_tokens.size()),
                .reused_prompt_tokens    = r.reused,
                .processed_prompt_tokens = r.prefilled,
                .elapsed_ns              = elapsed_ns(r.admitted, Clock::now())};
        }
        r.cv.notify_all();
    }

    // Whether a branched read's prompt step of `n` tokens is its last and its first branch fits
    // the chunk beside it, so the step and the branches run in one pass (prompt_branch_pass). A
    // media prompt prefills in passes of its own.
    [[nodiscard]] bool opens_branches(const Request& r, std::uint32_t n) const {
        const auto& executor = *instance.executor;
        const auto prompt_n  = static_cast<std::uint32_t>(r.prompt_tokens.size());
        return r.branches && r.media.empty() && executor.runs_branches() && n > 0 &&
               r.prefilled + n == prompt_n &&
               n + (*r.branches)[0].tokens.size() <= executor.options().prefill_chunk;
    }

    // Appends the next branches of a branched read from `first` to `tokens` while they fit one
    // prompt chunk beside what it holds and their rows the logits; their lengths, in order.
    [[nodiscard]] std::vector<std::uint32_t> pack_branches(const Request& r, std::size_t first,
                                                           std::vector<TokenId>& tokens) const {
        const auto& executor                             = *instance.executor;
        const std::vector<runtime::ReadBranch>& branches = *r.branches;
        const std::size_t room                           = executor.options().prefill_chunk;
        const std::size_t rows                           = executor.max_logit_rows();
        std::vector<std::uint32_t> lengths;
        for (std::size_t b = first; b < branches.size() && lengths.size() < rows &&
                                    tokens.size() + branches[b].tokens.size() <= room;
             ++b) {
            tokens.insert(tokens.end(), branches[b].tokens.begin(), branches[b].tokens.end());
            lengths.push_back(static_cast<std::uint32_t>(branches[b].tokens.size()));
        }
        return lengths;
    }

    // A branched read's last prompt step and its first branches in one pass
    // (Executor::forward_prompt_branches): the state at the prompt's end goes to the slot's
    // snapshot, the anchor the later branches start from, as the pass reaches it, and each layer's
    // experts cross the bus once for the prompt's tokens and the branches'.
    void prompt_branch_pass(const std::shared_ptr<Request>& request, std::uint32_t n) {
        Request& r          = *request;
        Slot& slot          = slots[r.slot];
        auto& executor      = *instance.executor;
        const auto prompt_n = static_cast<std::uint32_t>(r.prompt_tokens.size());
        const auto chunk    = std::span<const TokenId>(r.prompt_tokens).subspan(r.prefilled, n);
        std::vector<TokenId> tokens(chunk.begin(), chunk.end());
        const std::vector<std::uint32_t> lengths = pack_branches(r, 0, tokens);
        executor.forward_prompt_branches(r.slot, n, slot.snapshot, tokens, lengths);
        r.prefilled = prompt_n;
        slot.fed.assign(r.prompt_tokens.begin(), r.prompt_tokens.end());
        slot.media        = r.media;
        slot.anchor       = slot.fed;
        slot.anchor_media = slot.media;
        if (reuse_prefixes && r.reusable &&
            std::binary_search(r.shared_at.begin(), r.shared_at.end(), prompt_n)) {
            store_image(r.slot, slot.anchor, slot.anchor_media, &slot.snapshot,
                        PrefixReusePath::SharedStablePrefix);
        }
        // The pass waited for its copies, so the prompt is in.
        if (r.observation.prompt_progress) { observe_prompt(r); }
        const std::vector<TokenId>& last = (*r.branches)[lengths.size() - 1].tokens;
        slot.fed.insert(slot.fed.end(), last.begin(), last.end());
        read_branches(request, lengths);
    }

    // The next branches of a branched read that fit one prompt chunk together, from the snapshot
    // at the prompt's end in one pass (Executor::forward_branches): each layer's experts cross the
    // bus once for all of them.
    void branch_pass(const std::shared_ptr<Request>& request) {
        Request& r              = *request;
        Slot& slot              = slots[r.slot];
        auto& executor          = *instance.executor;
        const std::size_t first = r.branch_results.size();
        std::vector<TokenId> tokens;
        const std::vector<std::uint32_t> lengths = pack_branches(r, first, tokens);
        executor.forward_branches(r.slot, slot.snapshot, tokens, lengths);
        const std::vector<TokenId>& last = (*r.branches)[first + lengths.size() - 1].tokens;
        slot.fed.assign(r.prompt_tokens.begin(), r.prompt_tokens.end());
        slot.fed.insert(slot.fed.end(), last.begin(), last.end());
        slot.media = r.media;
        read_branches(request, lengths);
    }

    // The first tokens of the branches a pass just ran (`lengths`, after those already read), a
    // lane's worth of logit rows at a time; once the last branch is read, the prompt is in.
    void read_branches(const std::shared_ptr<Request>& request,
                       std::span<const std::uint32_t> lengths) {
        Request& r          = *request;
        auto& executor      = *instance.executor;
        const auto prompt_n = static_cast<std::uint32_t>(r.prompt_tokens.size());
        const std::vector<runtime::ReadBranch>& branches = *r.branches;
        const std::size_t first                          = r.branch_results.size();
        r.progress              = Clock::now();
        const Tensor logits     = executor.logits(static_cast<std::uint32_t>(lengths.size()));
        const std::size_t width = slots.size();
        for (std::size_t at = 0; at < lengths.size(); at += width) {
            const std::size_t n = std::min(width, lengths.size() - at);
            std::vector<SampleRow> rows(n, sample_row(r));
            for (std::size_t k = 0; k < n; ++k) {
                const runtime::ReadBranch& branch = branches[first + at + k];
                rows[k].readout = branch.readout_tokens.empty() ? nullptr : &branch.readout_tokens;
            }
            std::vector<TokenId> sampled(n);
            std::vector<runtime::RawTokenLogprob> records(n);
            const Tensor source =
                logits.slice(1, static_cast<std::int32_t>(at), static_cast<std::int32_t>(n));
            sample(rows, ops::kSamplePurposePrefill, sampled, records, &source);
            for (std::size_t k = 0; k < n; ++k) {
                if (sampled[k] < 0) { throw_numerical_failure(r); }
                if (r.branch_results.empty()) {
                    r.branch_token  = sampled[k];
                    r.branch_record = records[k];
                }
                const auto own = static_cast<std::uint32_t>(branches[first + at + k].tokens.size());
                r.branch_results.push_back(
                    BranchResult{.prompt_tokens = prompt_n + own,
                                 .first         = instance.frontend.token_logprob(records[k])});
            }
        }
        if (r.branch_results.size() < branches.size()) { return; }
        finish_prompt(request);
    }

    // The prompt is in (and a branched read's branches): the request takes its first token, from
    // column `column` of logits(rows), and decodes from there.
    void finish_prompt(const std::shared_ptr<Request>& request, std::uint32_t column = 0,
                       std::uint32_t rows = 1) {
        Request& r          = *request;
        auto& executor      = *instance.executor;
        const auto prompt_n = static_cast<std::uint32_t>(r.prompt_tokens.size());
        CUDA_CHECK(ninfer::wait_for_stream(executor.head_stream()));
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
        r.progress         = Clock::now();
        // The first token's mask began with the request; when it is not done yet, the prompt's
        // last logits wait for it instead of the worker.
        if (r.branches) {
            accept(request, r.branch_token, &r.branch_record);
            return;
        }
        const Tensor logits = executor.logits(rows);
        if (!mask_ready(r)) {
            defer_sample(r, logits, column, ops::kSamplePurposePrefill);
            return;
        }
        if (!mask_usable(request)) { return; }
        const SampleRow row = sample_row(r);
        TokenId token       = 0;
        runtime::RawTokenLogprob logprob;
        const Tensor source(static_cast<std::uint8_t*>(logits.data) +
                                std::size_t(column) * vocab_rows * sizeof(__nv_bfloat16),
                            DType::BF16, {logits.ne[0], 1});
        sample(std::span(&row, 1), ops::kSamplePurposePrefill, std::span(&token, 1),
               std::span(&logprob, 1), &source);
        accept(request, token, row.logprobs ? &logprob : nullptr);
        next_mask(request);
    }

    // Applies the output policy to a sampled token: publishes what it accepts, then either
    // finishes the request or queues what its next decode step feeds. `logprob` is the token's
    // record when the request asked for logprobs.
    void accept(const std::shared_ptr<Request>& request, TokenId token,
                const runtime::RawTokenLogprob* logprob) {
        if (token < 0) { throw_numerical_failure(*request); }
        (void)accept(request, std::span<const TokenId>(&token, 1),
                     logprob != nullptr ? std::span<const runtime::RawTokenLogprob>(logprob, 1)
                                        : std::span<const runtime::RawTokenLogprob>{},
                     true);
        check_capacity(request);
    }

    // Keeps the state of a request whose next step feeds the token that begins a stop string's
    // match: the sequence has read everything before it.
    void keep_hold(Request& r) {
        r.hold_wanted = false;
        Slot& slot    = slots[r.slot];
        if (!holds_ready || !r.decoding || r.feed.empty()) { return; }
        instance.executor->snapshot(r.slot, slot.hold);
        slot.hold_tokens = static_cast<std::uint32_t>(slot.fed.size());
    }

    // A turn that ended at a stop string whose match began before its last token: the sequence
    // goes back to the state kept where the match began, which the visible text continues.
    // Without one it keeps its live state, which no later prompt continues.
    void rewind_to_hold(Request& r) {
        r.stop_rewind = false;
        Slot& slot    = slots[r.slot];
        if (slot.hold_tokens == 0 || slot.hold_tokens > slot.fed.size()) { return; }
        instance.executor->restore(r.slot, slot.hold);
        slot.fed.resize(slot.hold_tokens);
        slot.hold_tokens = 0;
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
    // was, as a single sampled token is fed whatever the policy decides), or the accepted ones
    // from the token that begins a stop string's match.
    // `committed`: the sequence already holds the state the tokens leave; a speculative round
    // commits after the policy has decided, the tokens Request::commit_tokens names when set.
    std::uint32_t accept(const std::shared_ptr<Request>& request, std::span<const TokenId> tokens,
                         std::span<const runtime::RawTokenLogprob> logprobs, bool committed) {
        Request& r                    = *request;
        const OutputDecision decision = r.output.preview_model(tokens, r.budget->remaining(),
                                                               r.budget->limit_reason(), logprobs);
        const auto now = Clock::now();
        if (r.first_token == Clock::time_point{}) { r.first_token = now; }
        r.last_token = now;
        r.progress   = now;
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
        // The match of a stop string began at accepted token `hold`. A sampled token is not read
        // yet; of a speculative round's verified tokens (the anchor, then the accepted ones but
        // the last) the sequence keeps those before it, so its state is where the visible text
        // may end.
        const std::optional<std::uint32_t> hold =
            accepted > 0 && decision.stop_hold_after && *decision.stop_hold_after < accepted
                ? decision.stop_hold_after
                : std::nullopt;
        if (hold && !committed) { r.commit_tokens = *hold + 1; }
        if (decision.finished()) {
            if (decision.stop_cut && !hold) {
                if (committed) {
                    rewind_to_hold(r);
                } else {
                    r.stop_rewind = true;
                }
            }
            finish(request, decision.finish_reason);
            return accepted;
        }
        r.hold_wanted = hold.has_value();
        if (hold) {
            r.feed.assign(tokens.begin() + *hold, tokens.begin() + accepted);
        } else {
            r.feed.assign(1, token);
        }
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

    // The drafts each request of a round of `b` may verify: every draft, or with round_tokens as
    // many as keep the round's verification within it (with 64 tokens, seven each for eight
    // requests); 0 past that, where the requests decode without drafts. A wider verification
    // copies every missing expert to the GPU, and its drafts cost more than they keep.
    [[nodiscard]] std::uint32_t round_drafts(std::size_t b) const {
        if (round_tokens == 0 || b == 0) { return drafts; }
        const auto width = round_tokens / static_cast<std::uint32_t>(b);
        return width > 1 ? std::min(drafts, width - 1) : 0;
    }

    // Whether the request's next step can be a speculative round: an Engine with drafts, one token
    // to feed, an MTP state that follows the sequence, room for the round in the context, and no
    // switch of sampling parameters pending (a reasoning block that has not closed yet when the
    // request samples its answer differently).
    [[nodiscard]] bool speculates(const Request& r) const {
        // A structured row speculates while its masks are cheap enough for the round to fill its
        // columns during the verification; otherwise it decodes plainly, a mask ahead.
        return drafts > 0 && (!r.constrained || r.mask_cost.speculates()) && r.feed.size() == 1 &&
               instance.executor->can_draft(r.slot) &&
               r.position + drafts + 1 <= max_context &&
               r.output.model_token_budget_remaining(r.budget->remaining()) > 1 &&
               !(r.options.execution.post_thinking_sampling && !r.post_thinking);
    }

    // One step of every decoding request: those that speculate run one MTP round together, those
    // that feed one token otherwise run as one batch, whose experts read their weights once; one
    // feeding a thinking-control suffix runs alone. Returns the tokens the requests that ran fed on
    // average, at least one.
    double decode_step() {
        auto& executor = *instance.executor;
        std::vector<std::shared_ptr<Request>> batch, alone, speculative, deferred;
        for (const Slot& slot : slots) {
            if (slot.request && slot.request->hold_wanted) { keep_hold(*slot.request); }
        }
        for (const Slot& slot : slots) {
            // A copy: finishing or failing the request clears its slot.
            const std::shared_ptr<Request> request = slot.request;
            if (!request || !request->decoding) { continue; }
            if (request->awaiting_sample) {
                // Its logits wait for its grammar mask; a cancellation need not.
                if (mask_ready(*request)) {
                    if (mask_usable(request)) { deferred.push_back(request); }
                } else if (request->cancelled.load(std::memory_order_acquire)) {
                    finish_now(request, FinishReason::Cancelled);
                }
                continue;
            }
            if (speculates(*request)) {
                speculative.push_back(request);
            } else {
                (request->feed.size() == 1 ? batch : alone).push_back(request);
            }
        }
        if (!speculative.empty() && round_drafts(speculative.size()) == 0) {
            batch.insert(batch.end(), speculative.begin(), speculative.end());
            speculative.clear();
        }
        if (batch.empty() && alone.empty() && speculative.empty() && deferred.empty()) { return 1; }
        std::vector<std::pair<const Request*, std::uint32_t>> fed_from;
        for (const auto* group : {&alone, &batch, &speculative}) {
            for (const auto& request : *group) {
                fed_from.emplace_back(request.get(), request->position);
            }
        }
        const auto start = Clock::now();
        if (!deferred.empty()) { sample_deferred(deferred); }
        for (const auto& request : alone) {
            Request& r = *request;
            // A structured row without its mask yet starts it now, to fill while the step runs.
            if (r.constrained && !r.mask) { schedule_mask(r); }
            executor.forward(r.slot, r.feed, 1);
            Slot& slot = slots[r.slot];
            slot.fed.insert(slot.fed.end(), r.feed.begin(), r.feed.end());
            r.position += static_cast<std::uint32_t>(r.feed.size());
            if (!mask_ready(r)) {
                defer_sample(r, executor.logits(1), 0, ops::kSamplePurposeDecode);
                continue;
            }
            const SampleRow row = sample_row(r);
            TokenId token       = 0;
            runtime::RawTokenLogprob logprob;
            sample(std::span(&row, 1), ops::kSamplePurposeDecode, std::span(&token, 1),
                   std::span(&logprob, 1));
            try {
                if (!mask_usable(request)) { continue; }
                accept(request, token, row.logprobs ? &logprob : nullptr);
                next_mask(request);
            } catch (...) { fail(request, std::current_exception()); }
        }
        if (!batch.empty()) { plain_step(batch); }
        if (!speculative.empty()) { speculative_round(speculative); }
        {
            std::lock_guard lock(stats_mutex);
            stats.decode_rounds += 1;
            stats.decode_row_rounds += batch.size() + alone.size() + speculative.size();
            stats.decode_seconds_total += seconds(start, Clock::now());
        }
        std::uint64_t fed = 0;
        for (const auto& [request, position] : fed_from) {
            fed += request->position > position ? request->position - position : 0;
        }
        return fed_from.empty() ? 1.0 : std::max(1.0, double(fed) / double(fed_from.size()));
    }

    // Keeps logits row `row` of `source` for the request until its grammar mask is ready.
    void defer_sample(Request& r, const Tensor& source, std::size_t row, std::int32_t purpose) {
        RankBinding bind(device, instance.executor->head_rank());
        const std::size_t bytes = vocab_rows * sizeof(__nv_bfloat16);
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint8_t*>(saved_logits.p) + std::size_t(r.slot) * bytes,
                                   static_cast<const std::uint8_t*>(source.data) + row * bytes, bytes,
                                   cudaMemcpyDeviceToDevice, instance.executor->head_stream()));
        r.awaiting_sample  = true;
        r.awaiting_purpose = purpose;
    }

    // Samples requests whose kept logits now have their masks.
    void sample_deferred(const std::vector<std::shared_ptr<Request>>& ready) {
        RankBinding bind(device, instance.executor->head_rank());
        const std::size_t bytes = vocab_rows * sizeof(__nv_bfloat16);
        for (const std::int32_t purpose : {ops::kSamplePurposePrefill, ops::kSamplePurposeDecode}) {
            std::vector<std::shared_ptr<Request>> group;
            for (const auto& request : ready) {
                if (request->awaiting_purpose == purpose) { group.push_back(request); }
            }
            if (group.empty()) { continue; }
            std::vector<SampleRow> rows;
            for (std::size_t j = 0; j < group.size(); ++j) {
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<std::uint8_t*>(deferred_logits.p) + j * bytes,
                    static_cast<const std::uint8_t*>(saved_logits.p) + std::size_t(group[j]->slot) * bytes,
                    bytes, cudaMemcpyDeviceToDevice, instance.executor->head_stream()));
                rows.push_back(sample_row(*group[j]));
            }
            const Tensor source(deferred_logits.p, DType::BF16,
                                {static_cast<std::int32_t>(vocab_rows), static_cast<std::int32_t>(group.size())});
            std::vector<TokenId> sampled(group.size());
            std::vector<runtime::RawTokenLogprob> logprobs(group.size());
            sample(rows, purpose, sampled, logprobs, &source);
            for (std::size_t j = 0; j < group.size(); ++j) {
                group[j]->awaiting_sample = false;
                if (purpose == ops::kSamplePurposeDecode && drafts > 0) {
                    ++group[j]->speculative.fallback_steps;
                }
                try {
                    accept(group[j], sampled[j], rows[j].logprobs ? &logprobs[j] : nullptr);
                    next_mask(group[j]);
                } catch (...) { fail(group[j], std::current_exception()); }
            }
        }
    }

    // One batched decode step of the requests that feed one token, whose experts read their weights
    // once. A grammar row whose mask is still being computed keeps its logits for later; the step
    // itself does not wait for the mask.
    void plain_step(const std::vector<std::shared_ptr<Request>>& batch) {
        auto& executor = *instance.executor;
        std::vector<std::uint32_t> sequences;
        std::vector<TokenId> tokens;
        for (const auto& request : batch) {
            sequences.push_back(request->slot);
            tokens.push_back(request->feed.front());
            // A structured row without its mask yet starts it now, to fill while the step runs.
            if (request->constrained && !request->mask) { schedule_mask(*request); }
        }
        executor.decode(sequences, tokens);
        const Tensor logits = executor.logits(static_cast<std::uint32_t>(batch.size()));
        std::vector<std::shared_ptr<Request>> now;
        std::vector<std::size_t> columns;
        for (std::size_t b = 0; b < batch.size(); ++b) {
            const auto& request = batch[b];
            slots[request->slot].fed.push_back(request->feed.front());
            request->position += 1;
            if (mask_ready(*request)) {
                now.push_back(request);
                columns.push_back(b);
            } else {
                defer_sample(*request, logits, b, ops::kSamplePurposeDecode);
            }
        }
        std::vector<SampleRow> rows;
        for (const auto& request : now) { rows.push_back(sample_row(*request)); }
        std::optional<Tensor> source;
        if (now.size() != batch.size() && !now.empty()) {
            RankBinding bind(device, executor.head_rank());
            const std::size_t bytes = vocab_rows * sizeof(__nv_bfloat16);
            for (std::size_t j = 0; j < now.size(); ++j) {
                CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint8_t*>(deferred_logits.p) + j * bytes,
                                           static_cast<const std::uint8_t*>(logits.data) + columns[j] * bytes,
                                           bytes, cudaMemcpyDeviceToDevice, executor.head_stream()));
            }
            source.emplace(Tensor(deferred_logits.p, DType::BF16,
                                  {static_cast<std::int32_t>(vocab_rows), static_cast<std::int32_t>(now.size())}));
        }
        std::vector<TokenId> sampled(now.size());
        std::vector<runtime::RawTokenLogprob> logprobs(now.size());
        if (!now.empty()) {
            sample(rows, ops::kSamplePurposeDecode, sampled, logprobs, source ? &*source : nullptr);
        }
        for (std::size_t b = 0; b < now.size(); ++b) {
            if (drafts > 0) { ++now[b]->speculative.fallback_steps; }
            try {
                if (!mask_usable(now[b])) { continue; }
                accept(now[b], sampled[b], rows[b].logprobs ? &logprobs[b] : nullptr);
                next_mask(now[b]);
            } catch (...) { fail(now[b], std::current_exception()); }
        }
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
        // With --adaptive-mtp the controller picks a lone request's steps from its draft survival
        // and the round times it has measured; a new request restarts its probe. A round of
        // several requests drafts kAutomaticDraftTokens steps at most, every request verifying
        // its drafts up to the floor: a batch drafts until each of
        // its requests has reached the floor or the round's width, so a wider round runs nearly
        // every step for all of them, and verifying fewer drafts than ran wastes the rest. Under
        // the automatic drafting with a CPU share, round_drafts() bounds the steps.
        const bool adapts   = mtp_controller && b == 1;
        std::uint32_t steps = std::min(
            round_drafts(b), mtp_controller && b > 1 ? SpeculativeOptions::kAutomaticDraftTokens
                                                     : drafts);
        if (adapts) {
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
        // earlier in its sequence proposes what followed then, in place of the MTP block's guess,
        // which is weakest where the output repeats its input; verification keeps it exact. It
        // proposes up to the round's width (with --adaptive-mtp, the steps the round drafts), since
        // a wider verification also routes to more experts. The MTP block drafts the other rows
        // only, and no row when all have a proposal.
        std::vector<char> looked_up(b, 0);
        std::size_t lookups = 0;
        std::uint32_t mtp_steps = 0;
        if (lookup_ngram != 0) {
            for (std::size_t j = 0; j < b; ++j) {
                std::vector<TokenId>& ledger = slots[batch[j]->slot].fed;
                ledger.push_back(anchors[j]);
                const std::uint32_t found =
                    runtime::lookup_draft(ledger, lookup_ngram, steps, all_drafts.data() + j * drafts,
                                          slots[batch[j]->slot].lookup_lag);
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
            // Each row's MTP block draws its drafts with the keys its verification columns draw
            // the target's tokens with (runtime::draft_sampling).
            std::vector<ops::SamplingConfig> mtp_configs;
            std::vector<std::int32_t> mtp_lengths;
            for (std::size_t j = 0; j < b; ++j) {
                if (looked_up[j]) { continue; }
                mtp_sequences.push_back(sequences[j]);
                mtp_anchors.push_back(anchors[j]);
                mtp_configs.push_back(runtime::draft_sampling(*sample_row(*batch[j]).params));
                mtp_lengths.push_back(static_cast<std::int32_t>(batch[j]->position));
            }
            std::vector<TokenId> mtp_drafts(mtp_sequences.size() * drafts);
            std::vector<std::uint32_t> mtp_extents(mtp_sequences.size(), steps);
            mtp_steps = executor.draft(mtp_sequences, mtp_anchors, mtp_drafts, mtp_extents, steps,
                                       mtp_configs, mtp_lengths);
            for (std::size_t j = 0, m = 0; j < b; ++j) {
                if (looked_up[j]) { continue; }
                std::copy_n(mtp_drafts.begin() + std::ptrdiff_t(m * drafts), drafts,
                            all_drafts.begin() + std::ptrdiff_t(j * drafts));
                draft_extents[j] = mtp_extents[m++];
            }
        }
        // Each row verifies the drafts its output/thinking budget leaves room for (the correction
        // takes one more token); with --adaptive-mtp the plan keeps the extents worth their rows.
        for (std::size_t j = 0; j < b; ++j) {
            const auto remaining = batch[j]->output.model_token_budget_remaining(
                batch[j]->budget->remaining());
            draft_extents[j] =
                static_cast<std::uint32_t>(std::min<std::size_t>(draft_extents[j], remaining - 1));
        }
        if (adapts) {
            std::vector<const runtime::MtpAdaptiveSignal*> signals;
            for (std::size_t j = 0; j < b; ++j) {
                signals.push_back(looked_up[j] ? nullptr : &batch[j]->mtp_signal);
            }
            const std::vector<std::uint32_t> limits = draft_extents;
            packed_planner.plan(signals, limits, draft_extents);
        }
        const std::size_t k = std::max<std::size_t>(
            1, *std::max_element(draft_extents.begin(), draft_extents.end()));
        const std::size_t w = k + 1;
        const auto columns = static_cast<std::int32_t>(w);
        std::vector<TokenId> proposed;
        proposed.reserve(b * k);
        for (std::size_t j = 0; j < b; ++j) {
            proposed.insert(proposed.end(), all_drafts.begin() + std::ptrdiff_t(j * drafts),
                            all_drafts.begin() + std::ptrdiff_t(j * drafts + k));
        }
        // The rows are packed at their own widths, no row verifying columns it cannot take; a row
        // that gathers logprobs keeps every row at the round's width, the layout their gather reads.
        const bool packed = std::none_of(batch.begin(), batch.end(), [](const auto& request) {
            return request->options.execution.logprobs;
        });
        std::vector<std::uint32_t> widths(b);
        std::vector<std::int32_t> starts(b);
        std::vector<TokenId> tokens;
        for (std::size_t j = 0; j < b; ++j) {
            const std::size_t own = packed ? draft_extents[j] : k;
            starts[j]             = static_cast<std::int32_t>(tokens.size());
            widths[j]             = static_cast<std::uint32_t>(own + 1);
            tokens.push_back(anchors[j]);
            tokens.insert(tokens.end(), proposed.begin() + std::ptrdiff_t(j * k),
                          proposed.begin() + std::ptrdiff_t(j * k + own));
        }
        const auto verified_columns = static_cast<std::int32_t>(tokens.size());
        // A structured row's masks, one per column after its own drafts, fill in parallel while
        // the verification runs.
        std::vector<std::shared_ptr<runtime::GrammarMaskJob>> column_masks(b);
        for (std::size_t j = 0; j < b; ++j) {
            if (!batch[j]->constrained) { continue; }
            const std::size_t own = std::min<std::size_t>(draft_extents[j], k);
            column_masks[j] = runtime::GrammarMaskJob::start(
                *batch[j]->output.grammar_state(), mask_words(),
                std::vector<TokenId>(proposed.begin() + std::ptrdiff_t(j * k),
                                     proposed.begin() + std::ptrdiff_t(j * k + own)));
        }
        executor.verify(sequences, tokens, widths);

        RankBinding bind(device, executor.head_rank());
        const cudaStream_t stream = executor.head_stream();
        std::vector<SampleRow> rows;
        for (const auto& request : batch) { rows.push_back(sample_row(*request)); }
        // The rows' sampling, each column of a structured row masked as its drafts before it leave
        // the grammar. A row whose grammar cannot serve samples unmasked and fails after the round.
        auto* configs = static_cast<ops::SamplingConfig*>(host_sample->data());
        std::vector<std::string> mask_failures(b);
        bool masked = false;
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
            config.draw              = row.params->gumbel ? ops::kSampleDrawGumbel
                                                          : ops::kSampleDrawInverseCdf;
            config.token_counts      = row.counts ? static_cast<std::int32_t*>(token_counts.p) +
                                                        std::size_t(row.slot) * domain
                                                  : nullptr;
            if (const auto& job = column_masks[j]) {
                job->wait();
                batch[j]->mask_cost.observe(job->seconds());
                mask_failures[j] = runtime::grammar_mask_problem(*job);
                if (mask_failures[j].empty()) {
                    // Columns past the row's own drafts are never taken; they stay unrestricted.
                    auto* words = static_cast<std::uint32_t*>(host_mask->data()) + j * w * mask_words();
                    std::copy_n(job->masks(), job->columns() * mask_words(), words);
                    std::fill(words + job->columns() * mask_words(), words + w * mask_words(),
                              ~std::uint32_t{0});
                    config.token_mask =
                        static_cast<const std::uint32_t*>(token_mask.p) + j * w * mask_words();
                    config.token_mask_stride = static_cast<std::int32_t>(mask_words());
                    masked                   = true;
                }
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
            staged[b * k + j]     = static_cast<std::int32_t>(draft_extents[j]);
            staged[b * k + b + j] = static_cast<std::int32_t>(batch[j]->position);
        }
        // Past the licensed run (b * (k + 2) + b * w + 2 * b): the rows' first columns.
        auto* staged_starts = staged + b * (k + 4) + b * w;
        std::copy(starts.begin(), starts.end(), staged_starts);
        CUDA_CHECK(
            cudaMemcpyAsync(spec_drafts.p, staged, b * k * 4, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(
            cudaMemcpyAsync(spec_extents.p, staged + b * k, b * 4, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(spec_lengths.p, staged + b * k + b, b * 4,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(spec_columns.p, staged_starts, b * 4, cudaMemcpyHostToDevice,
                                   stream));
        const auto rows_n         = static_cast<std::int32_t>(b);
        const Tensor logits       = executor.logits(static_cast<std::uint32_t>(verified_columns));
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
        Tensor targets(spec_targets.p, DType::I32, {verified_columns});
        ops::argmax(logits, targets, static_cast<std::int32_t>(domain), stream);
        Tensor extents(spec_extents.p, DType::I32, {rows_n});
        Tensor lengths(spec_lengths.p, DType::I32, {rows_n});
        Tensor round_anchors(spec_anchors.p, DType::I32, {rows_n});
        Tensor licensed(spec_licensed.p, DType::I32, {columns, rows_n});
        Tensor counts(spec_counts.p, DType::I32, {rows_n});
        Tensor accepted(spec_accepted.p, DType::I32, {rows_n});
        {
            auto scope = sample_workspace->scope();
            const Tensor column_starts(spec_columns.p, DType::I32, {rows_n});
            ops::speculative_accept_coupled_drafts(
                targets, logits, drafted, extents, lengths, round_anchors, licensed, counts,
                accepted, static_cast<std::int32_t>(domain), device_config, *sample_workspace,
                stream, &column_starts);
        }
        auto* host = staged + b * (k + 2);
        CUDA_CHECK(
            cudaMemcpyAsync(host, spec_licensed.p, b * w * 4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(
            cudaMemcpyAsync(host + b * w, spec_counts.p, b * 4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host + b * w + b, spec_accepted.p, b * 4, cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(ninfer::wait_for_stream(stream));
        instance.executor->raise_service_failure();

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
            if (licensed_n < 1 || licensed_n > columns) {
                throw std::runtime_error("Qwen3.8-Flash-Next verification returned an invalid licensed run");
            }
            // Non-finite logits (the sampler's marker) or an id outside the vocabulary in the run:
            // this request ends, the round's other rows commit.
            if (std::any_of(run, run + licensed_n, [&](std::int32_t token) {
                    return token < 0 || std::uint32_t(token) >= domain;
                })) {
                try {
                    throw_numerical_failure(r);
                } catch (...) { fail(request, std::current_exception()); }
                failed[j] = 1;
                continue;
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
            // The MTP draft chain's steps run for the round; a lookup proposes what it found.
            r.speculative.drafted_tokens += looked_up[j] ? draft_extents[j] : mtp_steps;
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
                if (adapts && mtp_controller->transitioned()) {
                    ++r.speculative.window_transitions;
                }
            }
            for (std::int32_t i = 0; i < drafts_n; ++i) {
                ++r.speculative.accepted_per_position[i];
            }
            const std::span<const TokenId> run_tokens(run, std::size_t(licensed_n));
            std::uint32_t taken = 0;
            if (!mask_failures[j].empty()) {
                fail(request, std::make_exception_ptr(RequestError(
                                  RequestErrorKind::Unavailable, "structured output: " + mask_failures[j])));
                failed[j] = 1;
            } else {
                try {
                    taken = accept(request, run_tokens, records, false);
                } catch (...) {
                    fail(request, std::current_exception());
                    failed[j] = 1;
                }
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
        // A row whose stop string's match began inside its run keeps the tokens before the match.
        std::vector<std::uint32_t> committed = kept;
        for (std::size_t j = 0; j < b; ++j) {
            if (const auto fewer = std::exchange(batch[j]->commit_tokens, std::nullopt);
                fewer && !failed[j]) {
                committed[j] = std::min(*fewer, kept[j]);
            }
        }
        executor.commit(sequences, committed);
        // The controller measures the rounds it picked of MTP drafts only: a lookup changes the
        // round's width. The plan prices the rows verified, whatever proposed them.
        if (adapts) {
            const double round_seconds = seconds(round_start, Clock::now());
            if (lookups == 0) {
                mtp_controller->observe_execution(static_cast<std::uint32_t>(b), steps,
                                                  round_seconds);
            }
            packed_planner.observe(static_cast<std::uint32_t>(verified_columns), round_seconds);
        }
        for (std::size_t j = 0; j < b; ++j) {
            Request& r = *batch[j];
            if (failed[j]) { continue; } // its sequence keeps nothing for the context cache
            Slot& slot = slots[r.slot];
            // The verification fed the anchor and the drafts; the sequence keeps the first ones.
            slot.fed.insert(slot.fed.end(), tokens.begin() + starts[j],
                            tokens.begin() + starts[j] + std::ptrdiff_t(committed[j]));
            r.position += committed[j];
            if (r.stop_rewind) { rewind_to_hold(r); }
            try {
                check_capacity(batch[j]);
            } catch (...) { fail(batch[j], std::current_exception()); }
            if (r.constrained) { next_mask(batch[j]); }
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
        forget(slots[0]);
        std::vector<float> out;
        out.reserve(n - first_target);
        RankBinding bind(device, executor.head_rank());
        for (std::uint32_t at = 0; at + 1 < n; at += chunk) {
            const std::uint32_t count = std::min(chunk, n - 1 - at);
            // A chunk before the first target only extends the context.
            executor.forward(0, std::span<const TokenId>(tokens).subspan(at, count),
                             first_target > at + count ? 0 : count);
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
            CUDA_CHECK(ninfer::wait_for_stream(stream));
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

std::shared_ptr<Qwen4ExpCore::Request>
Qwen4ExpCore::make_request(GroupMember& member, std::optional<std::uint32_t>& budget_out) {
    models::qwen3_5::PreparedPrompt& prompt         = member.prompt;
    const PromptSummary& prompt_summary             = member.prompt_summary;
    const double prepare_seconds                    = member.prepare_seconds;
    ResolvedRequestOptions& options                 = member.options;
    const OutputConsumerMode consumer_mode          = member.consumer_mode;
    const GenerationObservationOptions& observation = member.observation;
    Clock::time_point& pending_deadline             = member.pending_deadline;
    const auto submitted                            = Clock::now();
    for (const TokenId id : options.execution.readout_tokens) {
        if (id < 0 || static_cast<std::uint32_t>(id) >= impl_->domain) {
            throw std::invalid_argument("readout token is outside the public token domain");
        }
    }
    if (const auto& branches = options.execution.branches) {
        if (branches->empty() || options.execution.grammar ||
            prompt.token_ids().size() + 1 > impl_->max_context) {
            throw std::invalid_argument("a branched read needs branches and a text prompt");
        }
        for (const runtime::ReadBranch& branch : *branches) {
            if (branch.tokens.empty() ||
                prompt.token_ids().size() + branch.tokens.size() > impl_->max_context) {
                throw std::invalid_argument("a branch needs tokens within the context");
            }
            for (const TokenId id : branch.readout_tokens) {
                if (id < 0 || static_cast<std::uint32_t>(id) >= impl_->domain) {
                    throw std::invalid_argument("readout token is outside the public token domain");
                }
            }
        }
    }
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
        if (options.execution.branches && data.has_media()) {
            throw std::invalid_argument("a branched read needs a text prompt");
        }
        const auto prompt_n  = static_cast<std::uint32_t>(request->prompt_tokens.size());
        if (data.has_media() && !impl_->instance.executor->vision()) {
            throw std::invalid_argument("media need an Engine started with Vision");
        }
        for (const auto& item : data.vision_items) {
            for (std::size_t unit = 0; unit < item.token_spans.size(); ++unit) {
                const auto& span = item.token_spans[unit];
                request->media.push_back(MediaMark{.begin = std::uint32_t(span.begin),
                                                   .count = std::uint32_t(span.count),
                                                   .key   = unit_key(item, unit)});
            }
        }
        std::sort(request->media.begin(), request->media.end(),
                  [](const MediaMark& a, const MediaMark& b) { return a.begin < b.begin; });
        request->reusable = identity.reusable;
        request->anchor_at   = prompt_n;
        if (identity.rewrite_checkpoint &&
            identity.rewrite_checkpoint->kind ==
                models::qwen3_5::RewriteCheckpointKind::TurnClosure &&
            identity.rewrite_checkpoint->frontier > 0 &&
            identity.rewrite_checkpoint->frontier < prompt_n) {
            request->anchor_at = identity.rewrite_checkpoint->frontier;
        }
        request->disposable = data.tap_hints.disposable;
        // Shared prefixes worth an image of their own: a structural or declared boundary, not one
        // the engine merely observed (the prompt's end, a grid point). A branched read's prompt
        // ends at its declared boundary, the state its branches and the next reads of that state
        // resume from.
        const std::uint32_t shared_end =
            options.execution.branches ? prompt_n + 1 : request->anchor_at;
        for (const auto& opportunity : data.context_cache.opportunities) {
            const bool structural = has_shared_candidate_evidence(
                opportunity.evidence, SharedCandidateEvidence::ExplicitBoundary |
                                          SharedCandidateEvidence::RequestedAutomatic |
                                          SharedCandidateEvidence::DefaultAutomatic |
                                          SharedCandidateEvidence::EngineStructural);
            const std::uint32_t frontier = opportunity.frontier;
            if (request->reusable &&
                opportunity.kind == PromptCacheMarkerKind::SharedStablePrefix && structural &&
                frontier >= Impl::kMinStoredTokens && frontier < shared_end) {
                request->shared_at.push_back(frontier);
            }
        }
        std::sort(request->shared_at.begin(), request->shared_at.end());
        request->shared_at.erase(std::unique(request->shared_at.begin(), request->shared_at.end()),
                                 request->shared_at.end());
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
    // Strict tool arguments bring a grammar without a structured final format.
    if (request->output.grammar_state() && !impl_->structured_output) {
        throw std::invalid_argument(
            "strict tool arguments require an Engine started with structured_output");
    }
    const std::optional<std::uint32_t> budget = request->output.thinking_stats().effective_budget;
    request->prompt                           = std::move(prompt);
    request->options                          = std::move(options);
    request->branches                         = request->options.execution.branches;
    budget_out                                = budget;
    return request;
}

Qwen4ExpCore::Submission Qwen4ExpCore::submit(models::qwen3_5::PreparedPrompt prompt,
                                              PromptSummary prompt_summary, double prepare_seconds,
                                              ResolvedRequestOptions options,
                                              OutputConsumerMode consumer_mode,
                                              GenerationObservationOptions observation,
                                              Clock::time_point pending_deadline) {
    std::vector<GroupMember> one;
    one.push_back(GroupMember{.prompt           = std::move(prompt),
                              .prompt_summary   = prompt_summary,
                              .prepare_seconds  = prepare_seconds,
                              .options          = std::move(options),
                              .consumer_mode    = consumer_mode,
                              .observation      = observation,
                              .pending_deadline = pending_deadline});
    return std::move(submit_group(std::move(one)).front());
}

std::vector<Qwen4ExpCore::Submission> Qwen4ExpCore::submit_group(std::vector<GroupMember> group) {
    std::vector<std::shared_ptr<Request>> requests;
    std::vector<std::optional<std::uint32_t>> budgets(group.size());
    requests.reserve(group.size());
    for (std::size_t k = 0; k < group.size(); ++k) {
        requests.push_back(make_request(group[k], budgets[k]));
    }
    if (requests.empty()) { return {}; }
    {
        std::lock_guard lock(impl_->queue_mutex);
        if (impl_->stopping || impl_->failed) {
            throw RequestError(RequestErrorKind::Unavailable, "inference engine is unavailable");
        }
        if (impl_->outstanding + requests.size() > impl_->max_outstanding) {
            throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
        }
        impl_->outstanding += requests.size();
        for (const auto& request : requests) {
            request->id = impl_->next_id++;
            impl_->pending.push_back(request);
        }
    }
    impl_->queue_cv.notify_one();
    impl_->publish_queue();
    std::vector<Submission> submissions;
    submissions.reserve(requests.size());
    for (std::size_t k = 0; k < requests.size(); ++k) {
        submissions.push_back(Submission(*this, std::move(requests[k]), budgets[k]));
    }
    return submissions;
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

    const DeviceWaitWatch watched(impl_->device, "Qwen3.8-Flash-Next scoring caller");
    impl_->device.bind_to_current_thread();
    return impl_->score(tokens, first_target);
}

MemorySummary Qwen4ExpCore::memory_summary() const {
    const auto& stats  = impl_->instance.model->storage_stats();
    const auto memory  = impl_->instance.executor->memory();
    const auto arena   = [](std::uint64_t bytes) {
        return ArenaMemorySummary{bytes, bytes, bytes};
    };
    // A DSpine drafter's weights load beside the model's on its first device.
    const auto* drafter                 = impl_->instance.model->dspine();
    const std::uint64_t drafter_weights = drafter != nullptr ? drafter->device_bytes() : 0U;
    std::vector<DeviceMemorySummary> devices;
    for (std::size_t r = 0; r < impl_->device.size(); ++r) {
        const auto& rank = memory.ranks.at(r);
        const std::uint64_t rank_weights =
            (r < stats.device_capacity_by_rank.size() ? stats.device_capacity_by_rank[r] : 0) +
            (r == 0 ? drafter_weights : 0U);
        devices.push_back({.device    = impl_->device.rank(r).device,
                           .weights   = arena(rank_weights),
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
