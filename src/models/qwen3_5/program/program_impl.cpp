#include "core/device_wait.h"
#include "models/qwen3_5/program/structured_round.h"
#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/execution/linear.h"
#include "models/qwen3_5/execution/dspine_head.h"
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/speculative_round.h"
#include "core/evictable_kv_pool.h"
#include "core/host_kv_clamp.h"
#include "core/pinned_memory.h"
#include "core/startup.h"
#include "core/device.h"
#include <cuda_runtime.h>
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/target_logprobs.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

static_assert(std::is_nothrow_move_assignable_v<SpeculativeStats>);

namespace {

std::uint32_t normalized_private_capacity(const ContextCacheOptions& options);

std::uint32_t normalized_private_capacity(const ContextCacheOptions& options) {
    if (!options.max_private_continuations || *options.max_private_continuations == 0) {
        throw std::logic_error("Qwen3.5 context cache private capacity is not normalized");
    }
    return *options.max_private_continuations;
}

// The persistent arena is virtual memory whose free KV granules can fund a Vision window (overlay
// Vision residency) or an elastic expert cache (host or disk experts on one device). The lendable
// prefix ends past the last page-major KV plane.
std::unique_ptr<EvictableKVPool> make_kv_arena(DeviceContext& device,
                                               const execution::Parameters& parameters,
                                               const SequencePlanImpl& plan, bool expert_loan) {
    const std::size_t granularity = EvictableKVPool::device_granularity(device);
    if (granularity == 0 || plan.persistent.lendable_kv_end_bytes == 0) { return nullptr; }
    const std::size_t lendable = plan.persistent.lendable_kv_end_bytes / granularity * granularity;
    std::size_t window = 0;
    if (plan.features.overlay_vision()) {
        const EvictableWeightPool* const pool = parameters.model.weight_pool();
        if (pool == nullptr || !pool->mirror_captured()) {
            throw std::logic_error("overlay Vision weight pool has no captured window");
        }
        // A KV cache smaller than one window can never fund a concurrent encode. The Engine still
        // runs: every window then borrows the weight tail.
        window = pool->window_capacity_bytes() <= lendable ? pool->window_capacity_bytes() : 0;
    }
    if (window == 0 && !expert_loan) { return nullptr; }
    return std::make_unique<EvictableKVPool>(
        device, EvictableKVPool::Config{
                    .arena_bytes           = plan.persistent.bytes,
                    .lendable_prefix_bytes = plan.persistent.lendable_kv_end_bytes,
                    .window_capacity_bytes = window,
                });
}

// Devices whose kernels or copies may address memory on `rank` directly: every other rank's device
// that the context enabled peer access from. A suspendable region grants them access explicitly,
// which an ordinary allocation got implicitly when peer access was enabled.
std::vector<int> peers_of(const DeviceContext& device, std::size_t rank) {
    std::vector<int> out;
    const int own = device.rank(rank).device;
    for (std::size_t other = 0; other < device.size(); ++other) {
        const int id = device.rank(other).device;
        if (other == rank || id == own || !device.peer_access(other, rank)) { continue; }
        if (std::find(out.begin(), out.end(), id) == out.end()) { out.push_back(id); }
    }
    return out;
}

VmmRegion make_region(DeviceContext& device, std::size_t rank, std::size_t bytes) {
    RankBinding bind(device, rank);
    return VmmRegion(bytes, VmmRegion::Options{.device = device.rank(rank).device,
                                               .peers  = peers_of(device, rank)});
}

// A Model materialized suspendable makes the Program's device memory suspendable too: persistent
// state and workspace on every rank at fixed addresses (rank 0's persistent state in the KV pool
// when overlay Vision built one, whose pieces release the same way).
ProgramImpl::SuspendableStorage make_suspendable_storage(DeviceContext& device,
                                                         const execution::Parameters& parameters,
                                                         const SequencePlanImpl& plan,
                                                         bool kv_pool_backs_persistent) {
    ProgramImpl::SuspendableStorage out;
    if (!parameters.model.weights_suspendable()) { return out; }
    out.enabled = true;
    if (!kv_pool_backs_persistent) {
        out.persistent.emplace(make_region(device, 0, plan.persistent.bytes));
    }
    for (std::size_t index = 0; index < plan.persistent.extra_rank_bytes.size(); ++index) {
        out.persistent_by_rank.push_back(
            make_region(device, index + 1, plan.persistent.extra_rank_bytes[index]));
    }
    out.workspace.emplace(make_region(device, 0, plan.workspace.capacity));
    if (parameters.text.split_execution()) {
        for (std::size_t rank = 1; rank < parameters.text.rank_count; ++rank) {
            out.workspace_by_rank.push_back(
                make_region(device, rank, plan.workspace.general_capacity));
        }
    }
    return out;
}

// Persistent state for the ranks beyond the first, each allocated while its own device is current
// so it lands in that card's memory. Empty on one device, which leaves that path unchanged. With
// suspendable storage the arenas view its regions instead.
std::vector<DeviceArena> make_rank_persistent(DeviceContext& device,
                                              const std::vector<std::size_t>& bytes_by_rank,
                                              std::span<const VmmRegion> regions) {
    std::vector<DeviceArena> out;
    out.reserve(bytes_by_rank.size());
    for (std::size_t index = 0; index < bytes_by_rank.size(); ++index) {
        if (!regions.empty()) {
            out.emplace_back(regions[index].span());
            continue;
        }
        RankBinding bind(device, index + 1);
        out.emplace_back(bytes_by_rank[index]);
    }
    return out;
}

// The state shards and boundary links a forward pass across pipeline stages uses, sized for the
// widest pass the Program will run.
std::unique_ptr<execution::StageRuntime>
make_stage_runtime(DeviceContext& device, const execution::Parameters& parameters,
                   qwen3_5::StateImageDevicePool& state, const SequencePlanImpl& plan) {
    const auto& text         = parameters.text;
    const std::size_t stages = text.rank_count;
    auto runtime             = std::make_unique<execution::StageRuntime>();
    for (std::size_t shard = 0; shard < state.shard_count(); ++shard) {
        runtime->state.push_back(&state.linear(shard));
        runtime->state_first_layer.push_back(state.shard(shard).first_layer);
    }

    const std::uint64_t columns = stage_boundary_columns(plan);
    const std::size_t hidden = static_cast<std::size_t>(parameters.model.config().text.hidden_size);
    // Tests run the staged boundary protocol between ranks on one card.
    const bool force_staged  = [] {
        const char* value = std::getenv("NINFER_FORCE_STAGED_LINKS");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    const StageLinkOptions residual{.slot_bytes   = columns * hidden * sizeof(std::uint16_t),
                                    .slots        = 2,
                                    .force_staged = force_staged};
    const StageLinkOptions control{.slot_bytes   = stage_control_bytes(columns),
                                   .slots        = 2,
                                   .force_staged = force_staged};
    for (std::size_t stage = 1; stage < stages; ++stage) {
        runtime->control.emplace_back(device, 0, stage, control);
    }
    if (text.tensor_parallel()) {
        // Rank 0 hands every rank the pass's residual; the layers' sums keep the replicas equal.
        for (std::size_t rank = 1; rank < stages; ++rank) {
            runtime->forward.emplace_back(device, 0, rank, residual);
        }
        auto& tp  = runtime->tensor_parallel.emplace();
        tp.shares = text.tensor_shares;
        for (const TensorShare& share : tp.shares) {
            tp.shards.push_back(
                qwen3_5::tensor_parallel_share(parameters.model.config().text, share));
        }
        tp.prompt_partials = plan.features.tensor_parallel_prompt_partials;
        tp.exchange        = std::make_unique<TensorParallelExchange>(
            device,
            TensorParallelExchangeOptions{.slot_bytes = tensor_parallel_slot_bytes(plan),
                                          .transport  = plan.features.tensor_parallel_transport,
                                          .rounded_partials = tp.prompt_partials});
        for (std::size_t rank = 1; rank < stages; ++rank) { tp.joined.emplace_back(device.rank(rank)); }
        // A divided head's rows come back to rank 0, each rank's logits over its own link.
        if (const std::size_t head_bytes = divided_head_link_bytes(plan); head_bytes != 0) {
            const StageLinkOptions heads{
                .slot_bytes = head_bytes, .slots = 2, .force_staged = force_staged};
            for (std::size_t rank = 1; rank < stages; ++rank) {
                tp.heads.emplace_back(device, rank, 0, heads);
            }
        }
        return runtime;
    }
    for (std::size_t stage = 0; stage + 1 < stages; ++stage) {
        runtime->forward.emplace_back(device, stage, stage + 1, residual);
    }
    runtime->back.emplace(device, stages - 1, 0, residual);
    // A masked drafter reads target features from layers that may sit on later stages; such a stage
    // sends each tapped layer's output to rank 0 as soon as the layer has run, on a stream of its
    // own, so the transfer overlaps the stage's later layers.
    const auto& draft = parameters.model.config().draft;
    if (plan.features.masked_draft() && draft) {
        runtime->features.reserve(stages - 1);
        for (std::size_t stage = 1; stage < stages; ++stage) {
            std::vector<std::uint32_t> taps = text.stage_taps(stage, draft->target_layer_ids);
            if (taps.empty()) {
                runtime->features.emplace_back();
                continue;
            }
            const RankContext& rank = device.rank(stage);
            std::vector<StageLink> links;
            std::vector<CudaCompletionEvent> ready;
            links.reserve(taps.size());
            ready.reserve(taps.size());
            for (std::size_t tap = 0; tap < taps.size(); ++tap) {
                links.emplace_back(device, stage, 0, residual);
                ready.emplace_back(rank);
            }
            runtime->features.emplace_back(
                execution::StageRuntime::FeatureLink{.layers = std::move(taps),
                                                     .links  = std::move(links),
                                                     .ready  = std::move(ready),
                                                     .sent   = CudaCompletionEvent(rank),
                                                     .stream = execution::SideStream(rank.device)});
        }
    }
    return runtime;
}

// Scratch for the ranks beyond the first. Each is allocated while its own device is current, so the
// arena lands in that card's memory; `work` then borrows a slice of each and switches between them
// as the layer loop walks stages. Empty without a split, which leaves the single-GPU path unchanged.
//
// Only the general region is duplicated: a stage runs its layers and nothing else, so it never needs
// the Vision, causal-score or bridge regions the primary card's capacity also covers.
std::vector<DeviceArena> make_rank_workspaces(DeviceContext& device,
                                              const execution::TextParameters& text,
                                              std::size_t general_capacity_bytes,
                                              std::span<const VmmRegion> regions) {
    std::vector<DeviceArena> out;
    if (!text.split_execution()) { return out; }
    if (text.rank_count > device.size()) {
        throw std::invalid_argument("Qwen3.5 pipeline split needs more devices than are attached");
    }
    out.reserve(text.rank_count - 1);
    for (std::size_t rank = 1; rank < text.rank_count; ++rank) {
        if (!regions.empty()) {
            out.emplace_back(regions[rank - 1].span());
            continue;
        }
        ScopedDeviceRank guard(device, rank);
        out.emplace_back(general_capacity_bytes);
    }
    return out;
}

} // namespace

ProgramImpl::ProgramImpl(const execution::Parameters& parameters_in, const SequencePlanImpl& plan,
                         DeviceContext& device_in, const StartupObserver& startup_observer,
                         std::unique_ptr<execution::NativeExpertRuntime> expert_runtime,
                         const Steering* steering_source)
    : parameters(parameters_in), device(device_in), capacity(plan.capacity),
      kv_capacity(plan.kv_capacity), max_concurrency(plan.max_concurrency),
      context_cache(plan.context_cache),
      continuation_capacity(normalized_private_capacity(plan.context_cache)),
      shared_prefix_capacity(plan.context_cache.max_shared_prefixes.value_or(0)),
      prefill_chunk(plan.prefill_chunk), fast_prefill_kernel(plan.fast_prefill_kernel),
      draft_window(plan.draft_window), lookup_ngram(plan.lookup_ngram), mtp_policy(plan.mtp_policy),
      ngram_draft_window(plan.ngram_draft_window), ngram_min_match(plan.ngram_min_match),
      speculative_backend(plan.speculative_backend), kv_storage(plan.kv_storage),
      proposal_head(plan.proposal_head), rope_yarn(plan.rope_yarn),
      mtp_attention_window(plan.mtp_attention_window), vision_enabled(plan.features.vision),
      use_cuda_graph(plan.use_cuda_graph), causal_scoring(plan.causal_scoring),
      gumbel_sampling(plan.gumbel_sampling),
      kv_payload_bytes(plan.persistent.kv_payload_bytes),
      graph_allowance_bytes(plan.graph_allowance_bytes), workspace_plan(plan.workspace),
      kv_arena(make_kv_arena(device_in, parameters_in, plan,
                             expert_runtime && expert_runtime->elastic() && device_in.size() == 1)),
      suspendable(make_suspendable_storage(device_in, parameters_in, plan, kv_arena != nullptr)),
      persistent(kv_arena                 ? DeviceArena(kv_arena->arena())
                 : suspendable.persistent ? DeviceArena(suspendable.persistent->span())
                                          : DeviceArena(plan.persistent.bytes)),
      persistent_by_rank(make_rank_persistent(device_in, plan.persistent.extra_rank_bytes,
                                              suspendable.persistent_by_rank)),
      workspace_storage(suspendable.workspace ? DeviceArena(suspendable.workspace->span())
                                              : DeviceArena(plan.workspace.capacity)),
      workspace_storage_by_rank(make_rank_workspaces(device_in, parameters_in.text,
                                                     plan.workspace.general_capacity,
                                                     suspendable.workspace_by_rank)),
      work(DeviceSpan{workspace_storage.base(), plan.workspace.general_capacity}),
      continuation_states(continuation_capacity), continuation_slots(continuation_capacity),
      shared_prefix_states(shared_prefix_capacity), shared_prefix_slots(shared_prefix_capacity),
      experts(std::move(expert_runtime)),
      round_host(plan.causal_scoring
                     ? std::nullopt
                     : std::make_optional<PinnedHostBuffer>(kMaximumConcurrency * sizeof(TokenId))),
      score_logprobs_host(plan.causal_scoring ? std::make_optional<PinnedHostBuffer>(
                                                    kCausalScoreMaximumTile * sizeof(float))
                                              : std::nullopt),
      ordinary_host(
          !plan.causal_scoring && plan.speculative_backend == SpeculativeBackend::None
              ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::OrdinaryDecodeIngress) +
                                                     sizeof(qwen3_5::OrdinaryDecodeEgress))
              : std::nullopt),
      mtp_host(plan.speculative_backend == SpeculativeBackend::Mtp
                   ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::MtpDecodeIngress) +
                                                          sizeof(qwen3_5::MtpDecodeEgress))
                   : std::nullopt),
      dflash_host(is_masked_draft_backend(plan.speculative_backend)
                      ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::DFlashDecodeIngress) +
                                                             sizeof(qwen3_5::DFlashDecodeEgress))
                      : std::nullopt),
      compute_streams(RankStreams::compute(device_in)),
      transfer_streams(RankStreams::transfer(device_in)), context_source_ready_(device_in),
      context_completion_(device_in),
      context_transfer_timers_{CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream)} {
    if (&parameters != plan.parameters || parameters.model.options() != plan.features) {
        throw std::invalid_argument("Program parameters do not match the frozen sequence plan");
    }
    // Hand `work` the extra ranks' storage. From here one arena serves every device: the layer loop
    // switches ranks alongside ScopedDeviceRank and every workspace call site is unchanged.
    for (DeviceArena& rank_storage : workspace_storage_by_rank) {
        work.attach_rank_storage(
            DeviceSpan{rank_storage.base(), workspace_plan.general_capacity});
    }
    if (workspace_plan.general_capacity == 0 ||
        workspace_plan.vision.has_value() != vision_enabled ||
        causal_scoring != plan.persistent.score_hidden.has_value() ||
        causal_scoring != (workspace_plan.causal_score != 0) ||
        workspace_plan.vision_resident == plan.features.host_staged_vision() ||
        (workspace_plan.vision && workspace_plan.vision_resident &&
         workspace_plan.vision->general_capacity_bytes != workspace_plan.general_capacity)) {
        throw std::invalid_argument("Qwen3.5 workspace plan does not match startup features");
    }
    if (plan.features.overlay_vision()) {
        EvictableWeightPool* const pool = parameters.model.weight_pool();
        if (pool == nullptr || !parameters.model.vision_overlay() || !workspace_plan.vision ||
            workspace_plan.vision_bridge_bytes == 0 ||
            workspace_plan.vision_bridge_offset + workspace_plan.vision_bridge_bytes >
                workspace_storage.capacity()) {
            throw std::invalid_argument("overlay Vision assets are incomplete");
        }
        vision_broker.emplace(device, *pool);
        vision_results.emplace(max_concurrency, workspace_plan.vision->handoff_capacity_bytes);
    }
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    // Every rank's persistent memory, in rank order: rank 0's is `backing`.
    std::vector<DeviceSpan> backings{backing};
    for (std::size_t index = 0; index < persistent_by_rank.size(); ++index) {
        backings.push_back(persistent_by_rank[index].alloc_bytes(
            plan.persistent.extra_rank_bytes[index], 256));
    }
    if (!plan.context_cache.max_private_continuations || !plan.context_cache.max_shared_prefixes) {
        throw std::logic_error("Qwen3.5 context cache options are not normalized");
    }
    const std::uint64_t address_capacity64 =
        static_cast<std::uint64_t>(*plan.context_cache.max_private_continuations) +
        *plan.context_cache.max_shared_prefixes;
    if (address_capacity64 == 0 || address_capacity64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.5 KV address-space capacity exceeds uint32");
    }
    // One unpublished descriptor is reserved for the single in-flight active-capture snapshot.
    // Published private/shared address spaces remain bounded by P + S; the transaction slot lets a
    // full shared catalog replace one entry without releasing the old checkpoint before the new
    // snapshot has been prepared.
    if (address_capacity64 == std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.5 KV transaction address capacity exceeds uint32");
    }
    const auto address_capacity      = static_cast<std::uint32_t>(address_capacity64 + 1U);
    const auto logical_page_capacity = [&](const DeviceKVPagePool& pool) {
        const HostKVPageLayout host_layout = plan_host_kv_page_layout(pool.geometry());
        const std::uint64_t host_pages =
            context_cache.host_kv_capacity_bytes / host_layout.page_stride;
        const std::uint64_t total = static_cast<std::uint64_t>(pool.capacity_pages()) + host_pages;
        if (total > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Qwen3.5 logical KV page capacity exceeds uint32");
        }
        return static_cast<std::uint32_t>(total);
    };

    decoder = std::make_unique<qwen3_5::DecoderState>(backings, plan.persistent.decoder);
    text_host_kv_page_stride =
        plan_host_kv_page_layout(decoder->text_kv.page_pool().geometry()).page_stride;
    text_kv_pages = std::make_unique<LogicalKVPageStore>(
        decoder->text_kv.page_pool(), logical_page_capacity(decoder->text_kv.page_pool()));
    text_kv_addresses = std::make_unique<KVAddressSpaceStore>(
        *text_kv_pages, decoder->text_kv.execution_tables(), address_capacity,
        decoder->text_kv.execution_tables().logical_page_capacity());
    if (vision_broker && kv_arena && kv_arena->window_capacity_bytes() != 0) {
        // A loan changes the admission capacity, so it may not race a sealed plan: refuse one while
        // a context transaction or a pressure-planning session is in flight, or while any lane has
        // been provisionally admitted (prefill staged) but hasn't yet bound its own KV entitlement
        // (lifecycle still Empty) -- that lane's future KV demand is real but not yet reflected in
        // the arena's free-page accounting.
        vision_broker->enable_kv_tier(
            *kv_arena, decoder->text_kv.page_pool(),
            [this] {
                if (has_context_transaction() || pressure_planning_active_) { return false; }
                for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
                    if (requests[lane].prefill && requests[lane].lifecycle == Lifecycle::Empty) {
                        return false;
                    }
                }
                return true;
            },
            [this] { advance_resource_revision(); });
    }
    if (kv_arena && experts && experts->elastic() && device.size() == 1) {
        // Free pages kept back from the loan: a few small reservations' worth (prompt chunks,
        // checkpoint tails), and a Vision window's.
        const auto keep = [&](const DeviceKVPagePool& pool) {
            std::size_t page_bytes = 0;
            for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
                page_bytes += static_cast<std::size_t>(pool.plane(plane).nb[3]);
            }
            const std::uint32_t tokens = std::max<std::uint32_t>(8192, 2 * prefill_chunk);
            std::uint64_t pages = (tokens + kPagedKVPageSize - 1) / kPagedKVPageSize;
            if (page_bytes != 0) { pages += (kv_arena->window_capacity_bytes() + page_bytes - 1) / page_bytes; }
            return static_cast<std::uint32_t>(std::min<std::uint64_t>(pages, pool.capacity_pages()));
        };
        std::vector<DeviceKVPagePool*> pools{&decoder->text_kv.page_pool()};
        if (decoder->mtp_kv) { pools.push_back(&decoder->mtp_kv->page_pool()); }
        std::vector<std::uint32_t> margins;
        for (const DeviceKVPagePool* pool : pools) { margins.push_back(keep(*pool)); }
        kv_expert_loan = std::make_unique<KvExpertLoan>(
            *kv_arena, std::move(pools), std::move(margins),
            KvExpertLoan::Cache{[this] { return experts->grow_cache(); },
                                [this](std::uint64_t bytes) { return experts->shrink_cache(bytes); }});
    }
    state_images = std::make_unique<qwen3_5::StateImageDevicePool>(backings,
                                                                    plan.persistent.state_images);
    if (plan.context_cache.host_state_slots != 0) {
        const std::uint64_t host_state_bytes =
            static_cast<std::uint64_t>(state_images->host_layout().image_bytes) *
            context_cache.host_state_slots;
        StartupPhaseScope host_state_phase(startup_observer, StartupPhase::HostStatePin,
                                           StartupProgressUnit::Bytes, host_state_bytes);
        host_state_images = std::make_unique<qwen3_5::HostStatePool>(
            state_images->host_layout(), context_cache.host_state_slots);
        host_state_phase.complete(host_state_bytes, host_state_bytes);
    }
    // The pool's last slots are the lanes' stop-string hold images, outside the catalog.
    stop_hold_slots_ = stop_hold_slots(context_cache, max_concurrency);
    if (state_images->slot_count() <= 0 ||
        stop_hold_slots_ >= static_cast<std::uint32_t>(state_images->slot_count())) {
        throw std::logic_error("Qwen3.5 StateImage pool does not cover its stop-string holds");
    }
    const std::uint32_t catalog_state_slots =
        static_cast<std::uint32_t>(state_images->slot_count()) - stop_hold_slots_;
    const std::uint64_t logical_state_capacity =
        static_cast<std::uint64_t>(catalog_state_slots) + context_cache.host_state_slots;
    if (logical_state_capacity > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.5 logical StateImage capacity exceeds uint32");
    }
    state_store = std::make_unique<StateImageStore>(
        *state_images, host_state_images.get(), static_cast<std::uint32_t>(logical_state_capacity),
        catalog_state_slots);
    pressure_private_owner_scratch_.resize(continuation_capacity);
    pressure_shared_owner_scratch_.resize(shared_prefix_capacity);
    pressure_private_drop_scratch_.resize(continuation_capacity);
    const std::size_t pressure_checkpoint_capacity =
        2U + context_cache.max_long_anchors_per_continuation.value_or(0U);
    for (auto& dropped : pressure_private_drop_scratch_) {
        dropped.reserve(pressure_checkpoint_capacity);
    }
    pressure_state_scratch_.reserve(static_cast<std::size_t>(logical_state_capacity));
    if (plan.persistent.replay_records) {
        replay_records.emplace(backings[state_images->shard(0).rank],
                               *plan.persistent.replay_records);
        replay_fold.emplace(*replay_records, state_images->linear(0).all_layers_view());
        for (std::size_t shard = 1; shard < state_images->shard_count(); ++shard) {
            extra_replay_records.push_back(std::make_unique<GdnReplayRecords>(
                backings[state_images->shard(shard).rank],
                plan.persistent.extra_replay_records.at(shard - 1)));
            extra_replay_fold.push_back(std::make_unique<ops::GdnReplayFoldPlan>(
                *extra_replay_records.back(), state_images->linear(shard).all_layers_view()));
        }
    }
    if (replay_records.has_value() != (speculative_backend != SpeculativeBackend::None) ||
        replay_fold.has_value() != replay_records.has_value()) {
        throw std::logic_error("ReplaySSM records do not match the sequence plan");
    }
    if (parameters.text.split_execution()) {
        stage_runtime = make_stage_runtime(device, parameters, *state_images, plan);
        if (replay_records) {
            stage_runtime->replay.push_back(&*replay_records);
            for (const auto& records : extra_replay_records) {
                stage_runtime->replay.push_back(records.get());
            }
        }
    }
    if (plan.persistent.dflash) {
        CyclicKVCache* local = state_images->dflash_local();
        if (local == nullptr) {
            throw std::logic_error("DFlash StateImage has no local fixed state");
        }
        dflash.emplace(backing, *plan.persistent.dflash, *local);
        if (const dspine::Model* drafter = parameters.dspine) {
            const auto& text   = parameters.text;
            const auto& config = parameters.model.config().text;
            dspine::Target target{
                .hidden = config.hidden_size,
                .vocab  = config.vocab_size,
                .layers = config.num_hidden_layers,
                .domain = static_cast<std::int32_t>(
                    parameters.model.resources().public_token_count),
                .logits = execution::target_logits_dtype(text.output_head),
                .embed  = [&text](const Tensor& ids, Tensor& rows, cudaStream_t stream) {
                    execution::embed_tokens(ids, text.token_embedding, text.token_embedding_signs,
                                            rows, stream);
                },
                .head = [&text](const Tensor& hidden, Tensor& logits, WorkspaceArena& arena,
                                cudaStream_t stream) {
                    execution::project(hidden, text.output_head, logits, arena, stream);
                }};
            // Candidates in one pass where linear_topk takes the head. The proposal head of another
            // form projects dense logits, ranks them and maps its rows to token ids, as DFlash2's.
            const auto* ranked   = execution::dspine_ranked_head(parameters, proposal_head);
            const bool shortlist = proposal_head == ProposalHead::Optimized;
            if (ranked != nullptr || shortlist) {
                const execution::LinearParameters& head =
                    shortlist ? parameters.proposal->head : text.output_head;
                const Tensor* rows =
                    shortlist && parameters.proposal->token_ids ? &*parameters.proposal->token_ids
                                                                : nullptr;
                const auto valid = static_cast<std::int32_t>(
                    parameters.model.resources().public_token_count);
                target.head_topk = [ranked, &head, rows, valid](const Tensor& hidden, Tensor& ids,
                                                                Tensor& scores,
                                                                WorkspaceArena& arena,
                                                                cudaStream_t stream) {
                    auto scope = arena.scope();
                    if (ranked != nullptr) {
                        const Tensor input = execution::rotated_input(
                            hidden, ranked->hadamard_signs, arena, stream);
                        if (rows != nullptr) {
                            ops::linear_topk(input, ranked->weight, *rows, ids, scores, arena,
                                             stream);
                        } else {
                            ops::linear_topk(input, ranked->weight, valid, ids, scores, arena,
                                             stream);
                        }
                        return;
                    }
                    Tensor logits = arena.alloc(DType::BF16, {head.weight.n, hidden.ne[1]});
                    execution::project(hidden, head, logits, arena, stream);
                    ops::candidate_topk(logits, rows != nullptr ? head.weight.n : valid, ids,
                                        scores, arena, stream);
                    if (rows != nullptr) {
                        Tensor every = ids.view({ids.ne[0] * ids.ne[1]});
                        ops::proposal_remap_token_ids(
                            every, static_cast<const std::int32_t*>(rows->data), head.weight.n,
                            stream);
                    }
                };
            }
            // Its widest proposal: a copy round of several rows runs it at the copy window.
            const std::uint32_t proposal =
                masked_proposal_window(draft_window, ngram_draft_window, max_concurrency);
            dspine_drafter.emplace(*drafter, std::move(target), dflash->dspine_head_write,
                                   local->capacity(), capacity,
                                   static_cast<std::int32_t>(max_concurrency),
                                   static_cast<std::int32_t>(proposal + 1U));
            RankBinding bind(device, 0);
            work.reset();
            dspine_drafter->prepare(work, device.stream);
            work.reset();
            dflash->dspine = &*dspine_drafter;
        }
    }
    if (dflash.has_value() != plan.features.masked_draft()) {
        throw std::logic_error("DFlash state does not match the frozen sequence plan");
    }
    if (qwen3_5::PagedKVCache* backend = backend_kv_cache()) {
        backend_host_kv_page_stride =
            plan_host_kv_page_layout(backend->page_pool().geometry()).page_stride;
        backend_kv_pages = std::make_unique<LogicalKVPageStore>(
            backend->page_pool(), logical_page_capacity(backend->page_pool()));
        backend_kv_addresses = std::make_unique<KVAddressSpaceStore>(
            *backend_kv_pages, backend->execution_tables(), address_capacity,
            backend->execution_tables().logical_page_capacity());
    }
    pressure_text_page_scratch_.resize(text_kv_pages->capacity());
    pressure_text_selected_pages_.reserve(text_kv_pages->capacity());
    if (backend_kv_pages) {
        pressure_backend_page_scratch_.resize(backend_kv_pages->capacity());
        pressure_backend_selected_pages_.reserve(backend_kv_pages->capacity());
    }
    if (context_cache.host_kv_capacity_bytes != 0) {
        std::vector<HostKVPageLayout> layouts;
        layouts.push_back(plan_host_kv_page_layout(decoder->text_kv.page_pool().geometry()));
        if (const qwen3_5::PagedKVCache* backend = backend_kv_cache()) {
            HostKVPageLayout backend_layout =
                plan_host_kv_page_layout(backend->page_pool().geometry());
            if (backend_layout != layouts.front()) { layouts.push_back(std::move(backend_layout)); }
        }
        std::size_t minimum_stride = layouts.front().page_stride;
        for (const HostKVPageLayout& layout : layouts) {
            minimum_stride = std::min(minimum_stride, layout.page_stride);
        }

        // The pinned host KV buffer competes with the model for *device* memory, and on Windows
        // that is what actually stops it.
        //
        // Measured on an RTX 3090 (24,576 MiB), one process, allocating N MiB on the device and
        // then finding the largest cudaMallocHost that succeeds:
        //
        //     device resident   VRAM free   largest pin
        //         15,360 MiB     7,972 MiB     8,192 MiB
        //         17,408 MiB     5,924 MiB     6,656 MiB
        //         19,456 MiB     3,876 MiB     3,840 MiB
        //         21,504 MiB     1,828 MiB     2,816 MiB
        //         22,528 MiB       804 MiB     1,536 MiB
        //
        // Resident-device plus pinned-host lands within a few hundred MiB of the card's capacity
        // every time: WDDM maps pinned host memory into the GPU's address space and charges it
        // against the same budget. The default 8 GiB therefore cannot be reserved beside any model
        // this project ships, and the server refused to start.
        //
        // **Asking and backing off does not work.** One failed cudaMallocHost poisons every later
        // one in the process (`cudaErrorAlreadyMapped` persists and `cudaGetLastError` does not
        // clear it), so the size has to be chosen before the first attempt.
        //
        // Windows only: the coupling is a WDDM property and has not been measured on Linux, where
        // shrinking the buffer would cost prefix reuse for no reason. Clamping to zero is a
        // supported outcome -- prefix reuse then works from device pages alone.
        std::size_t reserved_bytes = plan.context_cache.host_kv_capacity_bytes;
#if defined(_WIN32)
        // The 1 GiB floor inside the clamp is memory the run still needs after this point and that
        // no plan accounts for (module loads, graph instantiation, launch-local backing); halving
        // what remains is the conservative split, so the cache never takes more than it leaves.
        // Every GPU of a pipeline can address pinned host memory, so the buffer is sized for the
        // rank with the least device memory free; that a mapping is charged to each of them is an
        // assumption, measured on one card only.
        std::size_t free_device       = std::numeric_limits<std::size_t>::max();
        bool measured                 = false;
        const std::size_t active_rank = device.active_rank();
        for (std::size_t rank = 0; rank < device.size(); ++rank) {
            device.activate_rank(rank);
            std::size_t free_rank = 0, total_rank = 0;
            if (cudaMemGetInfo(&free_rank, &total_rank) == cudaSuccess) {
                free_device = std::min(free_device, free_rank);
                measured    = true;
            }
        }
        device.activate_rank(active_rank);
        if (measured) {
            reserved_bytes =
                clamp_host_kv_reservation_bytes(reserved_bytes, free_device, minimum_stride);
        }
#endif
        StartupPhaseScope host_kv_phase(
            startup_observer, StartupPhase::HostKvPin, StartupProgressUnit::Bytes,
            static_cast<std::uint64_t>(plan.context_cache.host_kv_capacity_bytes));
        if (reserved_bytes >= minimum_stride) {
            try {
                host_kv_arena = std::make_unique<HostKVArena>(
                    reserved_bytes,
                    std::span<const HostKVPageLayout>(layouts.data(), layouts.size()));
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    std::string("failed to reserve the host KV cache: ") + error.what() +
                    "\nThis is the context cache's pinned host buffer, sized by --host-kv-mib "
                    "(default 8192) and clamped at startup to the device memory still free. On "
                    "Windows the allocation is mapped into the GPU's address space and competes "
                    "with the model for VRAM, so this is a full card rather than short system RAM: "
                    "free VRAM, lower --max-context, or pass --no-prefix-reuse to disable the "
                    "context cache entirely.");
            }
        } else {
            // Nothing was pinned. Prefix reuse still works from device pages; only the host-side
            // spill is gone. `host_kv_arena` stays null, which every user of it already handles.
            reserved_bytes = 0;
        }
        host_kv_phase.complete(
            static_cast<std::uint64_t>(reserved_bytes),
            static_cast<std::uint64_t>(plan.context_cache.host_kv_capacity_bytes));
        const std::size_t extent_capacity = reserved_bytes == 0 ? 0 : reserved_bytes / minimum_stride;
        if (extent_capacity > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Qwen3.5 Host KV extent capacity exceeds uint32");
        }
        if (extent_capacity != 0) {
            host_kv_extents = std::make_unique<HostKVExtentStore>(
                *host_kv_arena, static_cast<std::uint32_t>(extent_capacity));
        }
    }

    open_disk_tier(plan);

    io = qwen3_5::RoundState(backing, plan.persistent.round);
    if (io.mtp.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("round-state MTP extension does not match the sequence plan");
    }
    if (io.mtp_decode.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("MTP decode frame does not match the sequence plan");
    }
    if (io.ordinary.has_value() !=
        (!causal_scoring && speculative_backend == SpeculativeBackend::None)) {
        throw std::logic_error("ordinary decode frame does not match the sequence plan");
    }
    if (io.dflash_prefill.has_value() != is_masked_draft_backend(speculative_backend)) {
        throw std::logic_error("DFlash prefill scratch does not match the sequence plan");
    }
    if (io.dflash_decode.has_value() != is_masked_draft_backend(speculative_backend)) {
        throw std::logic_error("DFlash decode frame does not match the sequence plan");
    }
    prefill_hidden = plan.persistent.prefill_hidden.bind(backing);
    bind_steering(plan, steering_source, backings);
    if (plan.persistent.score_hidden) {
        score_hidden = plan.persistent.score_hidden->bind(backing);
    }
    if (plan.persistent.token_counts) {
        token_counts = plan.persistent.token_counts->bind(backing);
    }
    if (plan.persistent.sampling_config) {
        sampling_config = plan.persistent.sampling_config->bind(backing);
    }
    if (plan.persistent.draft_sampling_config) {
        draft_sampling_config = plan.persistent.draft_sampling_config->bind(backing);
    }
    if (plan.persistent.grammar_masks) {
        structured_round = std::make_unique<qwen3_5::StructuredRound>(
            plan.persistent.grammar_masks->bind(backing),
            parameters.model.resources().public_token_count, widest_verify_window() + 1U,
            max_concurrency, is_masked_draft_backend(speculative_backend));
        io.structured = structured_round.get();
    }
    active_continuations.fill(continuation_capacity);
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) { lane_epochs[lane] = 1; }
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        SequenceState& sequence = continuation_states[index];
        sequence.ledger.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_identity.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_digests.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.long_anchors.reserve(context_cache.max_long_anchors_per_continuation.value_or(0));
        // One retained shared resume source can coexist with every fixed per-request candidate
        // and, when a request asks for it, every prefix-grid candidate.
        sequence.shared_prefix_references.reserve(8U + kPrefixGridCandidates);
    }
    materialization_ledger_.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    materialization_identity_.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    materialization_prefix_digests_.reserve(static_cast<std::size_t>(capacity) + 1ULL);

    set_device_i32(io.text_kv_table_row, 0);
    if (!causal_scoring) { set_device_i32(io.backend_kv_table_row, 0); }

    host_tokens = round_host ? static_cast<TokenId*>(round_host->data()) : nullptr;
    if (ordinary_host) {
        ordinary_host_ingress = static_cast<qwen3_5::OrdinaryDecodeIngress*>(ordinary_host->data());
        ordinary_host_egress  = reinterpret_cast<qwen3_5::OrdinaryDecodeEgress*>(
            static_cast<unsigned char*>(ordinary_host->data()) +
            sizeof(qwen3_5::OrdinaryDecodeIngress));
        *ordinary_host_ingress = {};
        *ordinary_host_egress  = {};
    }
    if (mtp_host) {
        mtp_host_ingress = static_cast<qwen3_5::MtpDecodeIngress*>(mtp_host->data());
        mtp_host_egress  = reinterpret_cast<qwen3_5::MtpDecodeEgress*>(
            static_cast<unsigned char*>(mtp_host->data()) + sizeof(qwen3_5::MtpDecodeIngress));
        *mtp_host_ingress = {};
        *mtp_host_egress  = {};
    }
    if (dflash_host) {
        dflash_host_ingress = static_cast<qwen3_5::DFlashDecodeIngress*>(dflash_host->data());
        dflash_host_egress  = reinterpret_cast<qwen3_5::DFlashDecodeEgress*>(
            static_cast<unsigned char*>(dflash_host->data()) +
            sizeof(qwen3_5::DFlashDecodeIngress));
        *dflash_host_ingress = {};
        *dflash_host_egress  = {};
    }
    if (io.dflash_prefill) {
        CUDA_CHECK(cudaMemsetAsync(io.dflash_prefill->produced_count.data, 0,
                                   io.dflash_prefill->produced_count.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(io.rope_delta.data, 0, io.rope_delta.bytes(), device.stream));
    if (io.mtp) {
        CUDA_CHECK(
            cudaMemsetAsync(io.mtp->position.data, 0, io.mtp->position.bytes(), device.stream));
    }
    if (!causal_scoring) {
        CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
        CUDA_CHECK(
            cudaMemsetAsync(sampling_config.data, 0, sampling_config.bytes(), device.stream));
        if (draft_sampling_config.data != nullptr) {
            CUDA_CHECK(cudaMemsetAsync(draft_sampling_config.data, 0,
                                       draft_sampling_config.bytes(), device.stream));
        }
    }
    device.synchronize();
    if (speculative_backend == SpeculativeBackend::Mtp && mtp_policy == MtpDraftPolicy::Adaptive) {
        mtp_controller.emplace();
        mtp_controller->reset(draft_window);
    }
    if (is_masked_draft_backend(speculative_backend) && mtp_policy == MtpDraftPolicy::Adaptive) {
        const auto windows = masked_neural_windows(draft_window, true);
        if (windows.size() == 2) {
            masked_controller.emplace();
            masked_controller->reset(windows[1], windows[0]);
        }
    }
    // CUDA would otherwise load an attention width's module inside the first request that reaches
    // it, behind a context synchronization.
    if (const auto& attention = parameters.model.config().text.attention) {
        const ops::AttentionHeadGeometry geometry{
            .head_dim    = execution::dimension(attention->head_dim),
            .query_heads = execution::dimension(attention->num_attention_heads),
            .kv_heads    = execution::dimension(attention->num_key_value_heads),
        };
        ops::causal_softmax_attention_load_modules(
            geometry, kv_storage, false,
            static_cast<std::int32_t>(std::max(prefill_chunk, widest_verify_window() + 1U)));
        if (speculative_backend == SpeculativeBackend::Mtp) {
            ops::causal_softmax_attention_load_modules(geometry, kv_storage, true, 1);
        }
    }
    if (use_cuda_graph) {
        StartupPhaseScope graph_phase(startup_observer, StartupPhase::CudaGraphPrepare);
        const std::size_t free_before = device.free_bytes();
        prepare_graphs();
        device.synchronize();
        if (experts) { experts->check(); }
        const std::size_t free_after = device.free_bytes();
        graph_measured_bytes =
            free_after == 0 ? std::nullopt
                            : std::optional<std::size_t>(
                                  free_before > free_after ? free_before - free_after : 0);
        graph_phase.complete();
    }
    work.reset();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
    create_hybrid_prefix_cache(startup_observer);
    // Warm-up deliveries, the transport's asynchronous prefetches included, stay out of the
    // request counters, and their routes out of the cache: it starts from the routing profile.
    if (experts) {
        experts->drain();
        experts->prime();
        expert_startup_stats = experts->stats();
    }
    if (kv_expert_loan) {
        // The cache grows only into what loans free: the memory free now stays free.
        experts->set_keep_free_bytes(device.free_bytes());
        lend_kv_to_experts();
    }
}

void ProgramImpl::bind_steering(const SequencePlanImpl& plan, const Steering* source,
                                std::span<const DeviceSpan> backings) {
    const bool loaded = source != nullptr && !source->empty();
    if (plan.persistent.steering.empty()) {
        if (loaded) { throw std::invalid_argument("the sequence plan holds no loaded steering"); }
        return;
    }
    if (!loaded || source->adapters.size() != plan.steering_adapters ||
        source->layer_vectors.empty() == plan.steering_vectors ||
        plan.persistent.steering.size() != backings.size()) {
        throw std::invalid_argument("loaded steering does not match the sequence plan");
    }
    const auto& config       = parameters.model.config().text;
    const std::size_t hidden = config.hidden_size;
    const std::size_t sites  = steering_sites(config.num_hidden_layers);
    const std::size_t count  = source->adapters.size();
    // Column-major images of the device tensors (execution::SteeringRank).
    std::vector<float> directions(hidden * count);
    std::vector<float> factors(count * sites);
    std::vector<float> scales(count * max_concurrency);
    for (std::size_t e = 0; e < count; ++e) {
        const DirectionAdapter& adapter = source->adapters[e];
        std::copy(adapter.direction.begin(), adapter.direction.end(),
                  directions.begin() + static_cast<std::ptrdiff_t>(e * hidden));
        for (std::size_t site = 0; site < sites; ++site) {
            factors[site * count + e] = adapter.site_factors[site];
        }
        // Until a request takes a lane its row runs the default scales.
        for (std::size_t lane = 0; lane < max_concurrency; ++lane) {
            scales[lane * count + e] = adapter.default_scale;
        }
    }
    const auto upload = [](const Tensor& tensor, std::span<const float> values) {
        if (tensor.bytes() != values.size_bytes()) {
            throw std::logic_error("planned steering tensor does not match the loaded steering");
        }
        CUDA_CHECK(ninfer::copy_and_wait(tensor.data, values.data(), values.size_bytes(),
                                         cudaMemcpyHostToDevice));
    };
    steering                 = std::make_unique<execution::SteeringView>();
    steering->adapters       = static_cast<std::int32_t>(count);
    steering->steered_layers = source->steered_layers;
    for (std::size_t rank = 0; rank < backings.size(); ++rank) {
        const SteeringRankLayout& layout = plan.persistent.steering[rank];
        execution::SteeringRank& out     = steering->ranks.emplace_back();
        if (count != 0) {
            out.directions   = layout.directions->bind(backings[rank]);
            out.site_factors = layout.site_factors->bind(backings[rank]);
            out.lane_scales  = layout.lane_scales->bind(backings[rank]);
            out.measures     = layout.measures->bind(backings[rank]);
            if (layout.column_rows) { out.column_rows = layout.column_rows->bind(backings[rank]); }
            upload(out.directions, directions);
            upload(out.site_factors, factors);
            upload(out.lane_scales, scales);
        }
        if (!source->layer_vectors.empty()) {
            out.vectors = layout.vectors->bind(backings[rank]);
            upload(out.vectors, source->layer_vectors);
        }
    }
}

void ProgramImpl::write_lane_scales(const SequenceState& sequence,
                                    const PreparedPromptData& prompt) {
    if (!steering || steering->adapters == 0) { return; }
    const std::int32_t row = text_kv_addresses->bound_row(sequence.kv->text);
    if (row < 0 || static_cast<std::uint32_t>(row) >= max_concurrency) {
        throw std::logic_error("a sequence's KV table row has no adapter scales");
    }
    const auto count = static_cast<std::size_t>(steering->adapters);
    for (std::size_t rank = 0; rank < steering->ranks.size(); ++rank) {
        float* destination = static_cast<float*>(steering->ranks[rank].lane_scales.data) +
                             static_cast<std::size_t>(row) * count;
        CUDA_CHECK(cudaMemcpyAsync(destination, prompt.lora_scales.data(), count * sizeof(float),
                                   cudaMemcpyHostToDevice, device.stream_for_rank(rank)));
    }
}

void ProgramImpl::lend_kv_to_experts() {
    if (!kv_expert_loan) { return; }
    // At least this many pages move at once, so a lend (which drains the expert service) stays
    // rare: 16,384 tokens.
    constexpr std::uint32_t kMinimumPages = 16384 / kPagedKVPageSize;
    kv_expert_loan->lend(kMinimumPages);
}

void ProgramImpl::synchronize_transfer_streams() const {
    for (std::size_t rank = 0; rank < transfer_streams.size(); ++rank) {
        (void)ninfer::wait_for_stream(transfer_streams[rank]);
    }
}

ProgramImpl::~ProgramImpl() noexcept {
    for (std::size_t rank = 0; rank < transfer_streams.size(); ++rank) {
        (void)ninfer::wait_for_stream(transfer_streams[rank]);
        (void)ninfer::wait_for_stream(compute_streams[rank]);
    }
}

std::vector<float> ProgramImpl::causal_score(PreparedPromptData&& prompt,
                                             std::uint32_t first_target) {
    if (!causal_scoring || !score_hidden || !score_logprobs_host ||
        workspace_plan.causal_score == 0) {
        throw std::logic_error("Program was not constructed for causal scoring");
    }
    if (speculative_backend != SpeculativeBackend::None || vision_enabled || use_cuda_graph ||
        context_cache.enabled) {
        throw std::logic_error("causal scoring Program has generation-only startup features");
    }
    const std::size_t token_count_size = prompt.token_ids.size();
    if (token_count_size < 2 || token_count_size > capacity) {
        throw std::invalid_argument("causal score token count must be in [2,capacity]");
    }
    if (first_target == 0 || first_target >= token_count_size) {
        throw std::invalid_argument("causal score first_target is outside the token window");
    }
    if (prompt.has_media()) {
        throw std::invalid_argument("causal scoring accepts text tokens only");
    }

    const auto token_count                     = static_cast<std::uint32_t>(token_count_size);
    const std::uint32_t predictor_count        = token_count - 1U;
    const std::uint32_t scored_predictor_begin = first_target - 1U;
    const std::uint32_t entitlement            = kv_pages_for_frontier(predictor_count);
    if (entitlement == 0) { throw std::logic_error("causal score has no KV entitlement"); }

    std::optional<StateImageHandle> state;
    std::optional<KVAddressSpaceHandle> address;
    const auto cleanup = [&] {
        bool released = true;
        if (address) {
            if (text_kv_addresses->active(*address)) { text_kv_addresses->deactivate(*address); }
            released = text_kv_addresses->release(*address) && released;
            address.reset();
        }
        if (state) {
            released = state_store->release(*state) && released;
            state.reset();
        }
        if (!released) { throw std::logic_error("causal score resources could not be released"); }
    };

    std::vector<float> output;
    output.reserve(token_count_size - first_target);
    const DType logits_dtype       = execution::target_logits_dtype(parameters.text.output_head);
    const std::uint32_t score_tile = causal_score_tile(logits_dtype);
    std::vector<TokenId> staged_targets;
    staged_targets.reserve(score_tile);
    std::uint32_t staged_columns = 0;

    try {
        state = state_store->reserve_reset(compute_streams);
        if (!state) {
            throw ninfer::ContextCacheExhausted(
                "Device StateImage store has no free slot for a fresh sequence");
        }
        address = text_kv_addresses->create_active(entitlement, 0, compute_streams);
        if (!address) {
            throw ninfer::ContextCacheExhausted("text KV address space has no free active entry");
        }
        if (text_kv_addresses->bound_row(*address) != 0) {
            throw std::logic_error("causal score did not bind the unique Main KV row");
        }
        text_kv_addresses->ensure_mapped_to_tokens(*address, predictor_count, compute_streams);

        const std::int32_t state_slot = state_store->physical_slot(*state);
        const auto flush              = [&] {
            if (staged_columns == 0) { return; }
            if (staged_columns != staged_targets.size() || staged_columns > score_tile) {
                throw std::logic_error("causal score staging has an invalid shape");
            }
            work.reset();
            mark_workspace_usage(workspace_plan.causal_score);
            const auto columns = static_cast<std::int32_t>(staged_columns);
            Tensor logits      = work.alloc(
                logits_dtype, {dimension(parameters.model.config().text.vocab_size), columns});
            Tensor target_ids = work.alloc(DType::I32, {columns});
            Tensor logprobs   = work.alloc(DType::FP32, {columns});
            Tensor hidden     = score_hidden->slice(1, 0, columns);
            execution::project(hidden, parameters.text.output_head, logits, work, device.stream);
            CUDA_CHECK(cudaMemcpyAsync(target_ids.data, staged_targets.data(), target_ids.bytes(),
                                                    cudaMemcpyHostToDevice, device.stream));
            ops::target_logprobs(logits, target_ids,
                                              dimension(parameters.model.resources().public_token_count),
                                              logprobs, device.stream);
            CUDA_CHECK(cudaMemcpyAsync(score_logprobs_host->data(), logprobs.data, logprobs.bytes(),
                                                    cudaMemcpyDeviceToHost, device.stream));
            device.synchronize();
            const auto* host = static_cast<const float*>(score_logprobs_host->data());
            output.insert(output.end(), host, host + staged_columns);
            staged_targets.clear();
            staged_columns = 0;
            work.reset();
        };

        std::uint32_t cursor = 0;
        while (cursor < predictor_count) {
            const std::uint32_t nominal = std::min(prefill_chunk, predictor_count - cursor);
            execution::PrefillContext schedule_state{
                {device, parameters, work, state_images->linear(0), nullptr, io, prefill_hidden,
                 prefill_chunk, proposal_head, stage_runtime.get(), rope_yarn, fast_prefill_kernel,
                 mtp_attention_window, experts.get(), steering.get()},
                decoder->text_kv.execution_view(text_kv_addresses->execution_row(*address)),
                {},
                decoder->text_kv,
                nullptr,
                nullptr,
                cursor,
                nullptr,
                nullptr,
                state_slot,
                state_slot,
                0,
                0};
            // A chunk before the first scored predictor only extends the context: nothing reads
            // its final hidden rows.
            schedule_state.last_layer_kv_only = cursor + nominal <= scored_predictor_begin;
            mark_workspace_usage(workspace_plan.text_prefill);
            const execution::PrefillChunkResult result = execution::prefill_text_chunk(
                schedule_state, std::span<const TokenId>(prompt.token_ids), nominal, std::nullopt,
                false);
            if (result.finalized || result.processed_tokens == 0 ||
                result.processed_tokens > nominal) {
                throw std::logic_error("causal score Prefill made invalid progress");
            }
            const std::uint32_t chunk_begin = cursor;
            cursor += result.processed_tokens;
            text_kv_addresses->commit_frontier(*address, cursor);

            std::uint32_t selected = std::max(chunk_begin, scored_predictor_begin);
            while (selected < cursor) {
                const std::uint32_t available = cursor - selected;
                const std::uint32_t room      = score_tile - staged_columns;
                const std::uint32_t count     = std::min(available, room);
                Tensor source =
                    prefill_hidden.slice(1, static_cast<std::int32_t>(selected - chunk_begin),
                                         static_cast<std::int32_t>(count));
                Tensor destination = score_hidden->slice(
                    1, static_cast<std::int32_t>(staged_columns), static_cast<std::int32_t>(count));
                CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, source.bytes(),
                                           cudaMemcpyDeviceToDevice, device.stream));
                for (std::uint32_t column = 0; column < count; ++column) {
                    staged_targets.push_back(prompt.token_ids[selected + column + 1U]);
                }
                selected += count;
                staged_columns += count;
                if (staged_columns == score_tile) { flush(); }
            }
        }
        flush();
        if (output.size() != token_count_size - first_target) {
            throw std::logic_error("causal score produced the wrong number of logprobs");
        }
        cleanup();
        return output;
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        work.reset();
        try {
            cleanup();
        } catch (...) {}
        throw;
    }
}

void ProgramImpl::start_context_transfer_timer(runtime::ContextResourceClass resource) {
    context_transfer_timers_[context_resource_index(resource)].start();
}

void ProgramImpl::stop_context_transfer_timer(runtime::ContextResourceClass resource) {
    context_transfer_timers_[context_resource_index(resource)].record_stop();
}

runtime::ContextTransferObservation ProgramImpl::context_transfer_observation(
    runtime::ContextResourceClass resource, runtime::ContextTransferDirection direction,
    TransferWork work, std::uint32_t page_count, std::uint64_t state_images) const {
    const double elapsed_ns =
        static_cast<double>(
            context_transfer_timers_[context_resource_index(resource)].elapsed_ms()) *
        1'000'000.0;
    const std::uint64_t measured_ns =
        elapsed_ns >= static_cast<double>(std::numeric_limits<std::uint64_t>::max())
            ? std::numeric_limits<std::uint64_t>::max()
            : std::max<std::uint64_t>(1, static_cast<std::uint64_t>(elapsed_ns + 0.5));
    return runtime::ContextTransferObservation{
        .resource  = resource,
        .direction = direction,
        .units =
            resource == runtime::ContextResourceClass::State ? state_images : work.payload_bytes,
        .page_count = page_count,
        .work       = work,
        .elapsed_ns = measured_ns,
    };
}

MemorySummary ProgramImpl::memory_summary() const noexcept {
    MemorySummary out;
    out.device          = device.device;
    out.max_context     = capacity;
    out.kv_capacity     = kv_capacity;
    out.kv_cache        = kv_storage;
    const auto& weights = parameters.model.storage_stats();
    // A DSpine drafter's weights load beside the model's on its first device.
    const std::size_t model_weight_bytes =
        weights.device_capacity_bytes +
        (parameters.dspine != nullptr ? parameters.dspine->device_bytes() : 0U);
    out.weights = ArenaMemorySummary{model_weight_bytes, model_weight_bytes, model_weight_bytes};
    out.sequence =
        ArenaMemorySummary{persistent.capacity(), persistent.used(), persistent.peak_used()};
    std::size_t active_handoff_bytes = 0;
    for (const RequestControl& request : requests) {
        if (request.prefill && request.prefill->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.prefill->vision->active_handoff_bytes());
        }
    }
    std::size_t active_workspace_bytes = work.used();
    if (workspace_plan.vision && active_handoff_bytes != 0) {
        active_workspace_bytes =
            std::max(active_workspace_bytes,
                     workspace_plan.vision->handoff_offset_bytes + active_handoff_bytes);
    }
    out.workspace = ArenaMemorySummary{workspace_storage.capacity(), active_workspace_bytes,
                                       std::max(work.peak_used(), workspace_logical_peak_bytes)};
    if (workspace_plan.vision) {
        out.vision_workspace = VisionWorkspaceMemorySummary{
            .aggregate_prompt_tokens = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(capacity, kMaximumPromptVisionTokens)),
            .max_item_tokens        = workspace_plan.vision->max_merged_tokens,
            .general_capacity_bytes = workspace_plan.vision->general_capacity_bytes,
            .encode_peak_bytes      = workspace_plan.vision->encode_peak_bytes,
            .handoff_offset_bytes   = workspace_plan.vision->handoff_offset_bytes,
            .handoff_capacity_bytes = workspace_plan.vision->handoff_capacity_bytes,
            .handoff_active_bytes   = active_handoff_bytes,
            .handoff_peak_bytes     = vision_handoff_peak_bytes,
            .residency = parameters.model.cpu_vision()
                             ? VisionResidency::Cpu
                             : (workspace_plan.vision_resident ? VisionResidency::Resident
                                                               : VisionResidency::Overlay),
        };
        if (const EvictableWeightPool* const pool = parameters.model.weight_pool();
            !workspace_plan.vision_resident && pool != nullptr) {
            out.vision_workspace->window_capacity_bytes = pool->window_capacity_bytes();
            out.vision_workspace->pinned_weight_bytes   = parameters.model.pinned_weights().size();
            out.vision_workspace->mirror_bytes          = pool->mirror_bytes();
        }
    }
    if (experts) {
        for (std::size_t rank = 0; rank < device.size(); ++rank) {
            out.expert_cache_bytes += experts->memory(rank).cache_bytes;
        }
        const auto m = experts->memory(0);
        const auto overhead = m.bank_bytes + m.metadata_bytes;
        out.workspace.capacity_bytes += overhead;
        out.workspace.used_bytes += overhead;
        out.workspace.peak_used_bytes += overhead;
    }
    out.workspace_logical_peak_bytes = workspace_logical_peak_bytes;
    out.cuda_graph_allowance_bytes   = graph_allowance_bytes;
    out.cuda_graph_measured_bytes    = graph_measured_bytes;
    out.kv_payload_bytes             = kv_payload_bytes;
    out.host_state_image_bytes       = state_images->host_layout().image_bytes;
    out.host_kv_page_group_bytes     = text_host_kv_page_stride;
    out.host_cache_budget_bytes      = context_cache.host_cache_budget_bytes.value_or(0);
    if (host_state_images) {
        out.host_state_capacity_slots = host_state_images->capacity();
        out.host_state_occupied_slots = host_state_images->occupied();
    }
    if (host_kv_arena) {
        out.host_kv_capacity_bytes = host_kv_arena->capacity_bytes();
        out.host_kv_occupied_bytes = host_kv_arena->occupied_bytes();
    }
    if (device.size() > 1) {
        // The primary device's figures are the fields above. The work arena borrows a slice of
        // each further device's scratch and keeps a peak per rank; between passes it holds none.
        out.devices.push_back({.device    = out.device,
                               .weights   = out.weights,
                               .sequence  = out.sequence,
                               .workspace = out.workspace});
        for (std::size_t rank = 1; rank < device.size(); ++rank) {
            const std::size_t weight_bytes =
                rank < weights.device_capacity_by_rank.size()
                    ? weights.device_capacity_by_rank[rank]
                    : 0;
            const DeviceArena& state = persistent_by_rank.at(rank - 1);
            out.devices.push_back(
                {.device    = device.rank(rank).device,
                 .weights   = ArenaMemorySummary{weight_bytes, weight_bytes, weight_bytes},
                 .sequence  = ArenaMemorySummary{state.capacity(), state.used(), state.peak_used()},
                 .workspace = ArenaMemorySummary{
                     workspace_storage_by_rank.at(rank - 1).capacity(), 0,
                     rank < work.rank_count() ? work.peak_used_for_rank(rank) : 0}});
        }
        if (experts) {
            for (std::size_t rank = 0; rank < out.devices.size(); ++rank) {
                const auto m = experts->memory(rank);
                out.devices[rank].expert_cache_bytes = m.cache_bytes;
                if (rank == 0) { continue; }
                const auto overhead = m.bank_bytes + m.metadata_bytes;
                auto& workspace = out.devices[rank].workspace;
                workspace.capacity_bytes += overhead;
                workspace.used_bytes += overhead;
                workspace.peak_used_bytes += overhead;
            }
        }
    }
    return out;
}

void ProgramImpl::reset_memory_peaks() noexcept {
    persistent.reset_peak();
    work.reset_peak();
    std::size_t active_handoff_bytes = 0;
    for (const RequestControl& request : requests) {
        if (request.prefill && request.prefill->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.prefill->vision->active_handoff_bytes());
        }
    }
    vision_handoff_peak_bytes    = active_handoff_bytes;
    workspace_logical_peak_bytes = work.used();
    if (workspace_plan.vision && active_handoff_bytes != 0) {
        workspace_logical_peak_bytes =
            std::max(workspace_logical_peak_bytes,
                     workspace_plan.vision->handoff_offset_bytes + active_handoff_bytes);
    }
}


} // namespace ninfer::models::qwen3_5::detail
