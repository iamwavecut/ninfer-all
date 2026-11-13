#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/dspine_head.h"
#include "models/qwen3_5/execution/rotation.h"
#include "models/qwen3_5/execution/ffn.h"
#include "models/qwen3_5/execution/gdn.h"
#include "models/qwen3_5/execution/linear.h"
#include "models/qwen3_5/execution/mtp.h"
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/frontend/graft.h"
#include "models/qwen3_5/program/prefix/hybrid_host_layout.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/execution/workspace.h"
#include "core/device.h"
#include "core/host_kv_arena.h"
#include "core/tensor_parallel.h"
#include "ops/common/device_tier.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/context_kv_materialize.h"
#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ninfer/ops/logprob_topk.h"
#include "ninfer/ops/readout_logprobs.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "models/dspine/drafter.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/speculative_round.h"
#include "runtime/contract/rope_scaling.h"
#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {
namespace {

using execution::dimension;
namespace workspace = execution::workspace;

constexpr std::size_t kMiB        = 1024ULL * 1024ULL;
constexpr std::size_t kArenaAlign = 256ULL;

enum class GdnWorkspacePath : std::uint8_t {
    Prefill,
    Snapshot,
    ReplayRecord,
};

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::int32_t checked_i32(std::uint64_t value, const char* label) {
    if (value == 0 ||
        value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

// The Main KV pages a plan may hold. The Legacy cache retains context in continuations whose pages
// fit the C active lanes' windows, so C * L pages bound it. The hybrid cache keeps every Device
// page no active lease holds as cached blocks, so only the token capacity representation (pages *
// page size in int32) bounds it and automatic sizing spends the free VRAM on cache.
std::uint64_t maximum_main_page_groups(std::uint32_t concurrency, std::uint32_t logical_pages,
                                       const ContextCacheOptions& cache) {
    std::uint64_t maximum = static_cast<std::uint64_t>(concurrency) * logical_pages;
    if (cache.enabled && cache.mode == ContextCacheMode::Hybrid) {
        maximum = std::max<std::uint64_t>(
            maximum, static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) /
                         static_cast<std::uint64_t>(kPagedKVPageSize));
    }
    if (maximum > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    return maximum;
}

template <class ProfileAllowance>
std::size_t graph_topology_allowance(const std::vector<GraphExecutionProfile>& profiles,
                                     ProfileAllowance&& profile_allowance, const char* label) {
    std::vector<std::pair<std::uint32_t, std::size_t>> classes;
    for (const GraphExecutionProfile profile : profiles) {
        const std::size_t allowance = profile_allowance(profile);
        const auto existing = std::find_if(classes.begin(), classes.end(), [&](const auto& entry) {
            return entry.first == profile.topology_class;
        });
        if (existing == classes.end()) {
            classes.emplace_back(profile.topology_class, allowance);
        } else {
            existing->second = std::max(existing->second, allowance);
        }
    }

    std::size_t total = 0;
    for (const auto& [topology_class, allowance] : classes) {
        (void)topology_class;
        total = checked_add(total, allowance, label);
    }
    return total;
}

TensorLayout add_tensor(LayoutBuilder& builder, DType dtype,
                        std::initializer_list<std::int32_t> shape, const char* label) {
    return builder.add_tensor(dtype, shape, kArenaAlign, label);
}

PersistentLayout persistent_layout(const SequencePlanImpl& plan) {
    const auto& parameters = *plan.parameters;
    const auto& config     = parameters.model.config().text;

    if (!plan.context_cache.device_state_slots) {
        throw std::logic_error("Qwen3.5 context cache options are not normalized");
    }
    const std::int32_t state_image_slots = checked_i32(
        static_cast<std::uint64_t>(plan.max_concurrency) + *plan.context_cache.device_state_slots +
            stop_hold_slots(plan.context_cache, plan.max_concurrency),
        "Qwen3.5 StateImage slot count exceeds int32");
    const auto effective_prefill_chunk =
        static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
    const std::uint32_t logical_pages  = page_count(plan.capacity);
    const std::uint32_t physical_pages = plan.main_page_groups;
    const std::uint64_t mtp_extra_pages =
        plan.features.mtp()
            ? static_cast<std::uint64_t>(plan.max_concurrency) *
                  ((static_cast<std::uint64_t>(plan.draft_window - 1U) + kPagedKVPageSize - 1U) /
                   static_cast<std::uint32_t>(kPagedKVPageSize))
            : 0ULL;
    const std::uint32_t mtp_physical_pages = static_cast<std::uint32_t>(
        checked_i32(static_cast<std::uint64_t>(physical_pages) + mtp_extra_pages,
                    "MTP Paged KV physical pages exceed int32"));
    // One layout per device. A stage's layers keep their KV planes and recurrent state on the
    // stage's device; everything else -- round state, prefill buffers, the head's inputs, MTP and
    // DFlash state -- is rank 0's, so builder 0 carries all of it.
    const std::size_t ranks = parameters.text.rank_count;
    std::vector<LayoutBuilder> builders(ranks);
    std::vector<LayoutBuilder*> builder_pointers;
    for (LayoutBuilder& each : builders) { builder_pointers.push_back(&each); }
    LayoutBuilder& builder = builders.front();

    std::vector<std::size_t> attention_layer_rank;
    std::vector<std::int32_t> kv_head_shares;
    std::uint32_t kv_position_ranks = 0;
    std::vector<qwen3_5::StateImageShard> state_shards;
    if (ranks == 1) {
        state_shards.push_back({.rank = 0, .first_layer = 0, .layers = config.linear_attention_layers});
    } else if (parameters.text.tensor_parallel()) {
        // Every rank holds its KV heads' share of every attention layer's cache, or its pages'
        // share under position-split attention, and its GDN heads' share of every GDN layer's
        // state.
        if (parameters.text.tensor_shares.front().attention_positions) {
            if (plan.kv_storage == KvCacheStorage::Bf16Packed) {
                throw std::invalid_argument(
                    "--kv-dtype bf16p does not take --tensor-attention positions");
            }
            kv_position_ranks = static_cast<std::uint32_t>(ranks);
        } else {
            for (const auto& share : parameters.text.tensor_shares) {
                if (share.kv_head_end > share.kv_head_begin) {
                    kv_head_shares.push_back(
                        static_cast<std::int32_t>(share.kv_head_end - share.kv_head_begin));
                }
            }
        }
        for (std::size_t rank = 0; rank < ranks; ++rank) {
            const TextConfig share =
                qwen3_5::tensor_parallel_share(config, parameters.text.tensor_shares.at(rank));
            state_shards.push_back(
                {.rank          = rank,
                 .first_layer   = 0,
                 .layers        = config.linear_attention_layers,
                 .part          = static_cast<std::uint32_t>(rank),
                 .value_heads   = dimension(share.gdn->linear_num_value_heads),
                 .conv_channels = dimension(share.gdn->conv_channels())});
        }
    } else {
        std::uint32_t gdn_index = 0;
        for (std::size_t layer = 0; layer < parameters.text.layers.size(); ++layer) {
            const std::size_t rank = parameters.text.layers[layer].rank;
            if (config.layer_types[layer] == MixerKind::FullAttention) {
                attention_layer_rank.push_back(rank);
                continue;
            }
            if (!state_shards.empty() && state_shards.back().rank == rank) {
                ++state_shards.back().layers;
            } else {
                state_shards.push_back({.rank = rank, .first_layer = gdn_index, .layers = 1});
            }
            ++gdn_index;
        }
    }

    // A divided MTP attention keeps each rank's KV heads of its cache on that rank, as a text
    // layer's.
    std::vector<std::int32_t> mtp_kv_head_shares;
    if (plan.features.mtp() && parameters.mtp && !parameters.mtp->rank_attention.empty()) {
        mtp_kv_head_shares = kv_head_shares;
    }
    PersistentLayout out;
    out.decoder = qwen3_5::plan_decoder_state(
        builder_pointers, qwen3_5::DecoderStateSpec{
                              .full_attention_layers = config.full_attention_layers,
                              .mtp_layers            = 1,
                              .capacity              = plan.capacity,
                              .kv_heads = dimension(config.attention->num_key_value_heads),
                              .attention_head_dim = dimension(config.attention->head_dim),
                              .kv_storage         = plan.kv_storage,
                              .enable_mtp         = plan.features.mtp(),
                              .kv_table_rows      = static_cast<std::int32_t>(plan.max_concurrency),
                              .text_physical_page_groups = physical_pages,
                              .mtp_physical_page_groups  = mtp_physical_pages,
                              .text_layer_rank           = std::move(attention_layer_rank),
                              .text_kv_head_shares       = std::move(kv_head_shares),
                              .text_kv_position_ranks    = kv_position_ranks,
                              .mtp_kv_head_shares        = std::move(mtp_kv_head_shares),
                              .invariant_attention       = plan.invariant_attention,
                          });
    // The Program binds this pool's own planned geometry, so the Host page cost the RAM budget
    // trades against is priced from the plan rather than recovered from a constructed pool.
    out.host_kv_text_page_stride =
        plan_host_kv_page_layout(out.decoder.text_kv.pages.spec.geometry).page_stride;
    qwen3_5::StateImageSpec state_image_spec{
        .linear =
            {
                .layers        = config.linear_attention_layers,
                .conv_channels = (config.gdn ? dimension(config.gdn->conv_channels()) : 0),
                .conv_width  = (config.gdn ? dimension(config.gdn->linear_conv_kernel_dim - 1) : 0),
                .value_heads = (config.gdn ? dimension(config.gdn->linear_num_value_heads) : 0),
                .value_head_dim = (config.gdn ? dimension(config.gdn->linear_value_head_dim) : 0),
                .key_head_dim   = (config.gdn ? dimension(config.gdn->linear_key_head_dim) : 0),
                .slot_count     = state_image_slots,
                .conv_dtype     = DType::BF16,
                .recurrent_dtype = plan.features.gdn_state_fp16 ? DType::FP16 : DType::FP32,
            },
        .hidden = dimension(config.hidden_size),
    };
    {
        const auto* draft =
            parameters.model.config().draft ? &*parameters.model.config().draft : nullptr;
        if (plan.features.masked_draft()) {
            // DSpine's ring is the narrowest attention window that sees the whole context.
            state_image_spec.dflash_local = qwen3_5::DFlashLocalStateSpec{
                .layers   = draft->local_layer_count(),
                .capacity = draft->dspine ? dspine::ring_capacity(*draft->dspine, plan.capacity)
                                          : draft->sliding_window.value_or(0),
                .kv_heads = dimension(draft->attention.num_key_value_heads),
                .head_dim = dimension(draft->attention.head_dim),
            };
        }
    }
    out.state_images =
        qwen3_5::plan_state_image_device_pool(builder_pointers, state_shards, state_image_spec);
    if (plan.speculative_backend != SpeculativeBackend::None) {
        // A stage records only the GDN layers it holds, in its own device's memory; a
        // tensor-parallel rank only its heads.
        for (std::size_t shard = 0; shard < state_shards.size(); ++shard) {
            // A tensor-parallel shard's own heads and channels, else the whole layer's.
            const StateImageShard& part = state_shards[shard];
            const std::int32_t value_heads =
                part.value_heads != 0
                    ? part.value_heads
                    : (config.gdn ? dimension(config.gdn->linear_num_value_heads) : 0);
            const std::int32_t ratio =
                config.gdn ? dimension(config.gdn->linear_num_value_heads /
                                       config.gdn->linear_num_key_heads)
                           : 1;
            GdnReplayRecordLayout records = plan_gdn_replay_records(
                *builder_pointers[part.rank],
                GdnReplayRecordSpec{
                    .layers          = static_cast<std::int32_t>(part.layers),
                    .record_capacity = static_cast<std::int32_t>(plan.max_concurrency),
                    .width           = static_cast<std::int32_t>(widest_verify_window(plan) + 1U),
                    .conv_channels =
                        part.conv_channels != 0
                            ? part.conv_channels
                            : (config.gdn ? dimension(config.gdn->conv_channels()) : 0),
                    .qk_heads    = value_heads / ratio,
                    .value_heads = value_heads,
                    .key_dim     = (config.gdn ? dimension(config.gdn->linear_key_head_dim) : 0),
                    .value_dim   = (config.gdn ? dimension(config.gdn->linear_value_head_dim) : 0),
                });
            if (shard == 0) {
                out.replay_records = std::move(records);
            } else {
                out.extra_replay_records.push_back(std::move(records));
            }
        }
    }
    {
        const auto* draft =
            parameters.model.config().draft ? &*parameters.model.config().draft : nullptr;
        if (plan.features.masked_draft()) {
            DFlashPersistentLayout& dflash = out.dflash.emplace();
            if (draft->full_layer_count() != 0) {
                const PagedKVStorageLayout full_storage = paged_kv_storage_layout(
                    KvCacheStorage::BFloat16, dimension(draft->attention.head_dim));
                KVPageGeometry full_geometry{
                    .page_tokens        = kPagedKVPageSize,
                    .device_plane_order = PagedKVPlaneOrder::HeadMajor,
                    .planes =
                        {
                            {full_storage.key.data_dtype, full_storage.key.data_leading_extent,
                             dimension(draft->attention.num_key_value_heads), 256},
                            {full_storage.value.data_dtype, full_storage.value.data_leading_extent,
                             dimension(draft->attention.num_key_value_heads), 256},
                        },
                };
                const auto planes = full_geometry.planes;
                for (std::uint32_t layer = 1; layer < draft->full_layer_count(); ++layer) {
                    full_geometry.planes.insert(full_geometry.planes.end(), planes.begin(),
                                                planes.end());
                }
                dflash.full = qwen3_5::PagedKVCacheLayout{
                    .pages = plan_device_kv_page_pool(
                        builder, DeviceKVPagePoolSpec{.page_group_count = physical_pages,
                                                      .geometry = std::move(full_geometry)}),
                    .execution_tables = {plan_kv_execution_tables(
                        builder,
                        KVExecutionTableSpec{
                            .logical_page_capacity = logical_pages,
                            .table_rows = static_cast<std::int32_t>(plan.max_concurrency),
                        })},
                    .layers      = draft->full_layer_count(),
                    .max_context = plan.capacity,
                    .kv_heads    = dimension(draft->attention.num_key_value_heads),
                    .head_dim    = dimension(draft->attention.head_dim),
                    .storage     = KvCacheStorage::BFloat16,
                };
            }
            dflash.prefill_features = add_tensor(
                builder, DType::BF16,
                {dimension(config.hidden_size * std::uint64_t(draft->target_layer_ids.size())),
                 effective_prefill_chunk},
                "DFlash prefill target features");
            dflash.prefill_positions = add_tensor(builder, DType::I32, {effective_prefill_chunk},
                                                  "DFlash prefill target positions");
            dflash.pending_features  = add_tensor(
                builder, DType::BF16,
                {dimension(config.hidden_size * std::uint64_t(draft->target_layer_ids.size())),
                 static_cast<std::int32_t>(widest_verify_window(plan) + 1U),
                 static_cast<std::int32_t>(plan.max_concurrency)},
                "DFlash pending target features");
            if (draft->dspine) {
                dflash.dspine_head_write = add_tensor(
                    builder, DType::BF16,
                    {dimension(draft->dspine->rank), dimension(draft->dspine->vocab_size)},
                    "DSpine head-write table");
            }
        }
    }

    out.round = qwen3_5::begin_round_state_layout(
        builder, qwen3_5::RoundStateSpec{
                     .hidden         = dimension(config.hidden_size),
                     .output_rows    = dimension(config.vocab_size),
                     .batch_capacity = plan.max_concurrency,
                     .draft_window   = plan.draft_window,
                     .verify_window  = widest_verify_window(plan),
                     .backend        = plan.speculative_backend,
                     .causal_scoring = plan.causal_scoring,
                     .logits_dtype = execution::target_logits_dtype(parameters.text.output_head)});
    out.prefill_hidden =
        add_tensor(builder, DType::BF16, {dimension(config.hidden_size), effective_prefill_chunk},
                   "step prefill hidden");
    if (plan.causal_scoring) {
        const auto tile = static_cast<std::int32_t>(
            causal_score_tile(execution::target_logits_dtype(parameters.text.output_head)));
        out.score_hidden = add_tensor(builder, DType::BF16, {dimension(config.hidden_size), tile},
                                      "causal score hidden staging");
    }
    qwen3_5::complete_round_state_layout(builder, out.round);
    if (!plan.causal_scoring) {
        out.token_counts        = add_tensor(builder, DType::I32,
                                             {dimension(parameters.model.resources().public_token_count),
                                              static_cast<std::int32_t>(plan.max_concurrency)},
                                             "sampling token counts");
        if (plan.structured_output) {
            out.grammar_masks =
                add_tensor(builder, DType::I32,
                           {static_cast<std::int32_t>(
                                (parameters.model.resources().public_token_count + 31) / 32),
                            static_cast<std::int32_t>(widest_verify_window(plan) + 1U),
                            static_cast<std::int32_t>(plan.max_concurrency)},
                           "structured output token masks");
        }
        const auto config_words = static_cast<std::int32_t>(
            (sizeof(ops::SamplingConfig) + sizeof(std::int32_t) - 1) / sizeof(std::int32_t));
        out.sampling_config = add_tensor(
            builder, DType::I32, {config_words, static_cast<std::int32_t>(plan.max_concurrency)},
            "sampling config");
        if (plan.features.mtp() && plan.gumbel_sampling) {
            out.draft_sampling_config = add_tensor(
                builder, DType::I32,
                {config_words, static_cast<std::int32_t>(plan.max_concurrency)},
                "MTP proposal sampling config");
        }
    }
    if (plan.steering_adapters != 0 || plan.steering_vectors) {
        // Every rank edits its replica or stage residual itself, so each holds its own copy.
        const auto hidden  = dimension(config.hidden_size);
        const auto columns = checked_i32(stage_boundary_columns(plan),
                                         "steering pass columns exceed int32");
        const auto adapters = static_cast<std::int32_t>(plan.steering_adapters);
        for (std::size_t rank = 0; rank < ranks; ++rank) {
            LayoutBuilder& rank_builder = *builder_pointers[rank];
            SteeringRankLayout& layout  = out.steering.emplace_back();
            if (adapters != 0) {
                layout.directions   = add_tensor(rank_builder, DType::FP32, {hidden, adapters},
                                                 "LoRA adapter directions");
                layout.site_factors = add_tensor(
                    rank_builder, DType::FP32,
                    {adapters, static_cast<std::int32_t>(steering_sites(config.num_hidden_layers))},
                    "LoRA adapter site factors");
                layout.lane_scales =
                    add_tensor(rank_builder, DType::FP32,
                               {adapters, static_cast<std::int32_t>(plan.max_concurrency)},
                               "LoRA adapter lane scales");
                layout.measures = add_tensor(rank_builder, DType::FP32, {adapters, columns},
                                             "LoRA adapter residual measures");
                if (rank == 0) {
                    layout.column_rows = add_tensor(rank_builder, DType::I32, {columns},
                                                    "LoRA adapter column rows");
                }
            }
            if (plan.steering_vectors) {
                layout.vectors = add_tensor(
                    rank_builder, DType::FP32,
                    {hidden, static_cast<std::int32_t>(config.num_hidden_layers)},
                    "control vectors");
            }
        }
    }
    out.bytes = builder.finish(kArenaAlign, "persistent layout");
    for (std::size_t rank = 1; rank < ranks; ++rank) {
        out.extra_rank_bytes.push_back(builders[rank].finish(
            kArenaAlign, ("persistent layout, device " + std::to_string(rank)).c_str()));
    }
    out.kv_payload_bytes =
        out.decoder.kv_payload_bytes() + (out.dflash ? out.dflash->kv_payload_bytes() : 0);
    const auto plane_end = [](const qwen3_5::PagedKVCacheLayout& cache) {
        std::size_t end = 0;
        if (cache.pages.spec.geometry.device_plane_order != PagedKVPlaneOrder::PageMajor) {
            return end;
        }
        for (const DeviceKVPlaneLayout& plane : cache.pages.planes) {
            // Lending is a single-device mechanism: only rank 0's planes are candidates.
            if (plane.rank != 0) { continue; }
            end = std::max(end, plane.storage.region.offset + plane.storage.region.bytes);
        }
        return end;
    };
    out.lendable_kv_end_bytes =
        std::max(plane_end(out.decoder.text_kv),
                 out.decoder.mtp_kv ? plane_end(*out.decoder.mtp_kv) : std::size_t{0});
    return out;
}

WorkspacePlan build_workspace_plan(const SequencePlanImpl& plan) {
    const auto& parameters   = *plan.parameters;
    const auto& config       = parameters.model.config().text;
    const DType logits_dtype = execution::target_logits_dtype(parameters.text.output_head);

    const std::uint32_t chunk_u32 = std::min(plan.prefill_chunk, plan.capacity);
    if (chunk_u32 == 0 ||
        chunk_u32 > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        widest_verify_window(plan) >=
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("sequence workspace dimensions are invalid");
    }
    const auto chunk  = static_cast<std::int32_t>(chunk_u32);
    const auto drafts = static_cast<std::int32_t>(plan.draft_window);
    // The masked drafter's widest proposal (a copy round of several rows runs it at its width).
    const auto proposal_drafts = static_cast<std::int32_t>(
        masked_proposal_window(plan.draft_window, plan.ngram_draft_window, plan.max_concurrency));
    // A copy round verifies up to the widest window; the neural drafter still proposes drafts.
    const auto verify_drafts = static_cast<std::int32_t>(widest_verify_window(plan));
    const auto verify        = verify_drafts + 1;
    // Neural rounds verify the draft window and copy rounds the ngram window.
    std::vector<std::int32_t> round_widths{drafts + 1};
    if (plan.ngram_draft_window != 0 && plan.ngram_draft_window != plan.draft_window) {
        round_widths.push_back(static_cast<std::int32_t>(plan.ngram_draft_window) + 1);
    }
    const std::int32_t narrowest_drafts =
        *std::min_element(round_widths.begin(), round_widths.end()) - 1;
    // Invariant attention reductions plan every call's splits at the context's segment; a plan
    // with them reserves for the tuned splits as well, which the MTP layer keeps.
    const std::uint32_t segment_keys =
        plan.invariant_attention ? ops::causal_attention_invariant_segment_keys(plan.capacity) : 0U;
    const ops::CausalAttentionExecutionEnvelope text_envelope{
        .min_visible_keys = 1, .max_visible_keys = plan.capacity,
        .invariant_segment_keys = segment_keys};
    // Prefill chunks of 17 to 64 rows may take the chunked small-T route over a long context, and
    // wider ones run the selected prompt kernel, whose fast NVFP4 form may split keys into
    // workspace (see execution/text.cpp).
    const ops::CausalAttentionExecutionEnvelope prefill_envelope{
        .min_visible_keys       = 1,
        .max_visible_keys       = plan.capacity,
        .fast_prompt_kernel     = plan.fast_prefill_kernel,
        .small_prefill          = true,
        .invariant_segment_keys = segment_keys};
    const ops::CausalAttentionExecutionEnvelope verify_envelope{
        .min_visible_keys       = 1,
        .max_visible_keys       = plan.capacity,
        .wide_verification      = plan.ngram_draft_window > 15,
        .invariant_segment_keys = segment_keys};

    const auto matrix  = [](WorkspaceLayoutBuilder& layout, DType dtype, std::int32_t rows,
                           std::int32_t tokens) { (void)layout.alloc(dtype, {rows, tokens}); };
    const auto scratch = [](WorkspaceLayoutBuilder& layout, std::size_t bytes) {
        if (bytes == 0) { return; }
        auto scope = layout.scope();
        (void)layout.alloc_bytes(bytes);
    };
    const auto finish = [](const WorkspaceLayoutBuilder& layout) { return layout.peak_bytes(1); };

    const auto text_common_root = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens) {
        (void)workspace::text_prefill_roots(layout, config, tokens, plan.features.vision ? 3 : 0,
                                            plan.features.vision ? tokens : 0,
                                            plan.features.host_staged_vision());
    };
    const auto linear_scratch = [&](WorkspaceLayoutBuilder& layout,
                                    const execution::LinearParameters& p, int first, int last) {
        scratch(layout, execution::rotated_workspace_bytes(
                            p.hadamard_signs, p.weight.k, last,
                            ops::linear_workspace_capacity_bytes(
                                p.weight.qtype, p.weight.n, p.weight.k, p.policy, first, last)));
    };
    // A divided head (tensor parallelism): every rank but 0 holds the hidden it receives and its
    // rows, rank 0 one buffer of the widest rank's rows, each with its share's scratch. Every rank's
    // scratch has rank 0's size, so one layout holds both.
    const auto divided_head_scratch = [&](WorkspaceLayoutBuilder& layout,
                                          const execution::HeadShares& head, DType dtype,
                                          int first, int last) {
        if (!head.divided()) { return; }
        std::int64_t widest = 0;
        for (std::size_t rank = 0; rank < head.shares.size(); ++rank) {
            widest = std::max(widest, head.rows[rank + 1] - head.rows[rank]);
        }
        auto scope = layout.scope();
        matrix(layout, DType::BF16, dimension(config.hidden_size), last);
        matrix(layout, dtype, static_cast<std::int32_t>(widest), last);
        for (const auto& share : head.shares) { linear_scratch(layout, share, first, last); }
    };
    const auto add_scratch = [&](WorkspaceLayoutBuilder& layout,
                                 const execution::LinearParameters& p, int first, int last,
                                 bool wide_verification = false) {
        // A round's narrower families may resolve the ordinary policy at the same columns.
        const auto bytes = [&](bool wide) {
            return execution::rotated_workspace_bytes(
                p.hadamard_signs, p.weight.k, last,
                ops::linear_add_workspace_capacity_bytes(
                    p.weight.qtype, p.weight.n, p.weight.k,
                    execution::residual_projection_policy(p, wide), first, last));
        };
        scratch(layout, wide_verification ? std::max(bytes(true), bytes(false)) : bytes(false));
    };
    const auto target_body = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                 std::int32_t last, TextPhase phase, GdnWorkspacePath path,
                                 std::int32_t batch_size, std::int32_t min_width,
                                 std::int32_t max_width,
                                 ops::CausalAttentionExecutionEnvelope envelope) {
        const bool wide_verification = wide_residual_verification(phase, min_width, max_width);
        // A tensor-parallel rank's share of a layer has attention only where the rank holds KV
        // heads, and its mixer and FFN output projections write FP32 partials instead of adding to
        // the residual.
        const bool partial = parameters.text.tensor_parallel();
        const bool positions =
            partial && parameters.text.tensor_shares.front().attention_positions;
        const auto residual_scratch = [&](const execution::LinearParameters& p) {
            if (partial) {
                linear_scratch(layout, p, first, last);
            } else {
                add_scratch(layout, p, first, last, wide_verification);
            }
        };
        const auto plan_block = [&](const execution::BlockParameters& block, bool attention_runs,
                                    const TextConfig& share) {
            if (attention_runs || std::holds_alternative<execution::GdnParameters>(block.mixer)) {
                auto stage = layout.scope();
                if (const auto* attention =
                        std::get_if<execution::AttentionParameters>(&block.mixer)) {
                    (void)workspace::text_attention_projection(layout, config, last);
                    (void)workspace::matrix(layout, DType::BF16, dimension(config.hidden_size),
                                            last);
                    scratch(layout, execution::attention_projection_workspace_bytes(*attention,
                                                                                    first, last));
                    (void)workspace::text_attention_results(layout, config, last);
                    scratch(layout,
                            ops::causal_softmax_attention_workspace_capacity_bytes(
                                {dimension(share.attention->head_dim),
                                 dimension(share.attention->num_attention_heads),
                                 dimension(share.attention->num_key_value_heads)},
                                plan.kv_storage, envelope, batch_size, min_width, max_width));
                    residual_scratch(attention->output);
                    if (positions) {
                        // The merge of a position-split rank's query heads, a pass of its own.
                        auto merge = layout.scope();
                        (void)workspace::matrix(layout, DType::BF16,
                                                dimension(config.attention->query_width()), last);
                        residual_scratch(attention->output);
                    }
                } else {
                    const auto& gdn = std::get<execution::GdnParameters>(block.mixer);
                    (void)workspace::gdn_control(layout, config, last);
                    scratch(layout, ops::gdn_norm_gating_proj_workspace_capacity_bytes(
                                        dimension(config.gdn->linear_num_value_heads),
                                        dimension(config.hidden_size), first, last));
                    (void)workspace::gdn_projection(layout, config, last);
                    if (path == GdnWorkspacePath::Snapshot) {
                        scratch(layout, execution::gdn_snapshot_workspace_bytes(
                                            gdn, *config.gdn, batch_size, min_width, max_width));
                    } else if (path == GdnWorkspacePath::ReplayRecord) {
                        scratch(layout, execution::gdn_record_workspace_bytes(
                                            gdn, *config.gdn, batch_size, min_width, max_width));
                    } else {
                        (void)workspace::gdn_prefill_conv(layout, config, last);
                        scratch(layout,
                                execution::gdn_projection_workspace_bytes(gdn, first, last));
                    }
                    (void)workspace::gdn_recurrent_output(layout, config, last);
                    if (path == GdnWorkspacePath::Prefill) {
                        scratch(layout, ops::gated_delta_net_workspace_capacity_bytes(
                                            dimension(config.gdn->linear_num_key_heads),
                                            dimension(config.gdn->linear_num_value_heads), true,
                                            first, last));
                    }
                    (void)workspace::gdn_normalized_output(layout, config, last);
                    residual_scratch(gdn.output);
                }
            }
            auto stage = layout.scope();
            (void)workspace::post_mixer_hidden(layout, config, last);
            const bool verify = phase == TextPhase::Verify;
            scratch(layout,
                    std::max(execution::ffn_workspace_bytes(block.ffn, first, last, false, verify,
                                                            wide_verification, partial),
                             execution::ffn_workspace_bytes(block.ffn, first, last, false, verify,
                                                            false, partial)));
        };
        std::vector<TextConfig> shares;
        for (const auto& share : parameters.text.tensor_shares) {
            shares.push_back(qwen3_5::tensor_parallel_share(config, share));
        }
        if (positions) {
            // Each rank keeps its query heads' gates from its attention to its merge for the whole
            // pass (at most every head).
            (void)workspace::matrix(layout, DType::BF16,
                                    dimension(config.attention->query_width()), last);
        }
        const auto holds_attention = [&](std::size_t rank) {
            return shares.empty() || shares[rank].attention->num_key_value_heads != 0;
        };
        for (std::size_t layer = 0; layer < parameters.text.layers.size(); ++layer) {
            plan_block(parameters.text.layers[layer], holds_attention(0),
                       shares.empty() ? config : shares[0]);
            // Every rank's scratch has rank 0's size, so it holds the largest share's needs.
            for (std::size_t rank = 1; rank <= parameters.text.rank_layers.size(); ++rank) {
                plan_block(parameters.text.rank_layers[rank - 1][layer], holds_attention(rank),
                           shares[rank]);
            }
        }
        // A prompt chunk projects only its last column through the head; a verify round projects
        // every column, through the divided head under tensor parallelism.
        if (!plan.causal_scoring) {
            const std::int32_t head_last = phase == TextPhase::Prefill ? 1 : last;
            linear_scratch(layout, parameters.text.output_head, std::min(first, head_last),
                           head_last);
            if (phase != TextPhase::Prefill) {
                divided_head_scratch(layout, parameters.text.output_head_shares, logits_dtype,
                                     first, last);
            }
        }
    };
    const auto mtp_post_mixer = [&](WorkspaceLayoutBuilder& layout, int first, int last) {
        linear_scratch(layout, parameters.mtp->output, first, last);
        if (parameters.mtp->rank_ffn.empty()) {
            scratch(layout, execution::ffn_workspace_bytes(parameters.mtp->ffn, first, last, true));
            return;
        }
        // A divided FFN: each rank's sum, then its share's partial projection. Every rank's
        // scratch has rank 0's size, so this holds the largest share's needs.
        auto scope        = layout.scope();
        std::size_t share = execution::ffn_workspace_bytes(parameters.mtp->ffn, first, last, false,
                                                           false, false, true);
        for (const auto& rank_ffn : parameters.mtp->rank_ffn) {
            share = std::max(share, execution::ffn_workspace_bytes(rank_ffn, first, last, false,
                                                                   false, false, true));
        }
        matrix(layout, DType::BF16, dimension(config.hidden_size), last);
        scratch(layout, share);
    };
    // A proposal's logits and the head's scratch; with --gumbel-sampling also the sampler's, after
    // a subset head's logits are spread over the vocabulary beside the sampled rows' draws
    // (TextContext::proposal_argmax).
    const auto proposal_scratch = [&](WorkspaceLayoutBuilder& layout, std::int32_t columns) {
        const auto vocabulary = dimension(parameters.model.resources().public_token_count);
        if (plan.proposal_head == ProposalHead::Optimized) {
            const auto& head  = parameters.proposal->head;
            const DType dtype = execution::fp32_proposal_logits(head) ? DType::FP32 : DType::BF16;
            matrix(layout, dtype, dimension(parameters.proposal->rows), columns);
            linear_scratch(layout, head, columns, columns);
            divided_head_scratch(layout, parameters.proposal->head_shares, dtype, columns, columns);
            if (plan.gumbel_sampling) {
                if (parameters.proposal->token_ids) {
                    matrix(layout, DType::FP32, vocabulary, columns);
                    (void)layout.alloc(DType::I32, {columns});
                }
                scratch(layout,
                        ops::sampling_workspace_capacity_bytes(vocabulary, columns, columns));
            }
        } else {
            const bool fp32 = execution::fp32_proposal_logits(parameters.mtp->output_head);
            if (fp32) { matrix(layout, DType::FP32, dimension(config.vocab_size), columns); }
            linear_scratch(layout, parameters.mtp->output_head, columns, columns);
            divided_head_scratch(layout, parameters.mtp->output_head_shares,
                                 fp32 ? DType::FP32 : DType::BF16, columns, columns);
            if (plan.gumbel_sampling) {
                scratch(layout,
                        ops::sampling_workspace_capacity_bytes(vocabulary, columns, columns));
            }
        }
    };
    const auto mtp_stem = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                              bool preembedded) {
        (void)workspace::mtp_stem(layout, config, tokens, !preembedded);
        linear_scratch(layout, parameters.mtp->input_projection, 1, tokens);
    };
    // A divided MTP attention (TextContext::mtp_divided_layer): the packed control tensors (cache
    // and rope positions of up to three axes, table rows and valid columns), every other rank's
    // copy of the stem output and of them and each rank's sums; then each rank's normalized input
    // with its share's attention (`attend`) and output partial, and its FFN rows or rank 0's whole
    // FFN. Every rank's scratch has rank 0's size, so one layout holds the largest share's needs.
    const bool mtp_divided = plan.features.mtp() && !parameters.mtp->rank_attention.empty();
    std::vector<TextConfig> mtp_shares;
    if (mtp_divided) {
        for (const auto& share : parameters.text.tensor_shares) {
            mtp_shares.push_back(qwen3_5::tensor_parallel_share(config, share));
        }
    }
    const auto mtp_divided_layer = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                       std::int32_t columns, const auto& attend) {
        auto scope = layout.scope();
        const std::int32_t control =
            4 * tokens + 2 * static_cast<std::int32_t>(plan.max_concurrency) + 8;
        matrix(layout, DType::I32, control, 1);
        matrix(layout, DType::BF16, dimension(config.hidden_size), tokens);
        matrix(layout, DType::I32, control, 1);
        if (columns != 0) { matrix(layout, DType::BF16, dimension(config.hidden_size), columns); }
        {
            auto step = layout.scope();
            matrix(layout, DType::BF16, dimension(config.hidden_size), tokens);
            for (std::size_t rank = 0; rank < parameters.mtp->rank_attention.size(); ++rank) {
                auto share = layout.scope();
                const execution::AttentionParameters& p = parameters.mtp->rank_attention[rank];
                attend(layout, mtp_shares.at(rank), p);
                if (columns != 0) { linear_scratch(layout, p.output, columns, columns); }
            }
        }
        if (columns == 0) { return; }
        auto step = layout.scope();
        matrix(layout, DType::BF16, dimension(config.hidden_size), columns);
        if (parameters.mtp->rank_ffn.empty()) {
            scratch(layout,
                    execution::ffn_workspace_bytes(parameters.mtp->ffn, columns, columns, true));
            return;
        }
        std::size_t share = execution::ffn_workspace_bytes(parameters.mtp->ffn, columns, columns,
                                                           false, false, false, true);
        for (const auto& rank_ffn : parameters.mtp->rank_ffn) {
            share = std::max(share, execution::ffn_workspace_bytes(rank_ffn, columns, columns,
                                                                   false, false, false, true));
        }
        scratch(layout, share);
    };
    // A divided decode attention: a rank's projections, normalized q and k, windowed table and
    // attention, as mtp_full_core's below for its heads.
    const auto mtp_divided_decode = [&](WorkspaceLayoutBuilder& layout, std::int32_t width,
                                        std::int32_t batch,
                                        ops::CausalAttentionExecutionEnvelope envelope) {
        const std::int32_t tokens = width * batch;
        mtp_divided_layer(layout, tokens, tokens,
                          [&](WorkspaceLayoutBuilder& share_layout, const TextConfig& share,
                              const execution::AttentionParameters& p) {
            (void)workspace::mtp_attention_projection(share_layout, share, tokens);
            scratch(share_layout,
                    execution::attention_projection_workspace_bytes(p, tokens, tokens));
            (void)workspace::mtp_attention_results(share_layout, share, tokens);
            if (plan.mtp_attention_window != 0) {
                (void)workspace::paged_kv_window(
                    share_layout, static_cast<std::int32_t>(page_count(plan.capacity)), width,
                    batch);
            }
            scratch(share_layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                                      {dimension(share.attention->head_dim),
                                       dimension(share.attention->num_attention_heads),
                                       dimension(share.attention->num_key_value_heads)},
                                      plan.kv_storage, envelope, batch, width, width));
        });
    };
    const auto mtp_full_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::CausalAttentionExecutionEnvelope envelope) {
        auto core = layout.scope();
        mtp_stem(layout, tokens, false);
        if (mtp_divided) {
            mtp_divided_decode(layout, tokens, 1, envelope);
            return;
        }
        (void)workspace::mtp_attention_projection(layout, config, tokens);
        scratch(layout, execution::mtp_projection_workspace_bytes(parameters.mtp->projection,
                                                                  tokens, tokens));
        (void)workspace::mtp_attention_results(layout, config, tokens);
        if (plan.mtp_attention_window != 0) {
            (void)workspace::paged_kv_window(
                layout, static_cast<std::int32_t>(page_count(plan.capacity)), tokens, 1);
        }
        scratch(layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                            {dimension(config.attention->head_dim),
                             dimension(config.attention->num_attention_heads),
                             dimension(config.attention->num_key_value_heads)},
                            plan.kv_storage, envelope, 1, tokens, tokens));
        (void)workspace::mtp_post_attention(layout, config, tokens);
        mtp_post_mixer(layout, tokens, tokens);
    };
    const auto mtp_full_call = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   bool build_proposal) {
        auto call = layout.scope();
        matrix(layout, DType::I32, 1, tokens);
        mtp_full_core(layout, tokens, envelope);
        if (build_proposal) {
            auto proposal = layout.scope();
            proposal_scratch(layout, 1);
        }
    };
    const auto mtp_prefill_chunk = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                       std::int32_t last, bool preembedded) {
        auto call = layout.scope();
        if (mtp_divided) {
            // Rank 0's whole K/V of the chunk with its normalized keys and the pack of a rank's
            // heads (every other rank receives no more); then the last column's rope positions and
            // its layer on every rank, the final norm's output being the caller's.
            mtp_stem(layout, last, preembedded);
            {
                auto kv = layout.scope();
                matrix(layout, DType::BF16, dimension(config.attention->key_width()), last);
                matrix(layout, DType::BF16, dimension(config.attention->key_width()), last);
                scratch(layout, execution::mtp_kv_rows_workspace_bytes(
                                    parameters.mtp->whole_key, parameters.mtp->whole_value, first,
                                    last));
                matrix(layout, DType::BF16, dimension(config.attention->key_width()), last);
                matrix(layout, DType::BF16, 2 * dimension(config.attention->key_width()), last);
            }
            matrix(layout, DType::I32, 3, 1);
            mtp_divided_layer(layout, 1, 1,
                              [&](WorkspaceLayoutBuilder& share_layout, const TextConfig& share,
                                  const execution::AttentionParameters& p) {
                (void)workspace::mtp_attention_projection(share_layout, share, 1);
                scratch(share_layout, execution::attention_projection_workspace_bytes(p, 1, 1));
                matrix(share_layout, DType::BF16, dimension(share.attention->query_width()), 1);
                matrix(share_layout, DType::BF16, dimension(share.attention->query_width()), 1);
                scratch(share_layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                                          {dimension(share.attention->head_dim),
                                           dimension(share.attention->num_attention_heads),
                                           dimension(share.attention->num_key_value_heads)},
                                          plan.kv_storage, text_envelope, 1, 1, 1));
            });
            proposal_scratch(layout, 1);
            return;
        }
        matrix(layout, DType::BF16, dimension(config.hidden_size), 1);
        matrix(layout, DType::BF16, dimension(config.hidden_size), 1);
        {
            auto bulk = layout.scope();
            mtp_stem(layout, last, preembedded);
            matrix(layout, DType::BF16, dimension(config.attention->key_width()), last);
            matrix(layout, DType::BF16, dimension(config.attention->key_width()), last);
            scratch(layout, execution::mtp_kv_workspace_bytes(parameters.mtp->projection,
                                                              *config.attention, first, last));
            matrix(layout, DType::BF16, dimension(config.attention->key_width()), last);
        }
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        scratch(layout, execution::mtp_query_gate_workspace_bytes(parameters.mtp->projection,
                                                                  *config.attention, 1, 1));
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        matrix(layout, DType::I32, 3, 1);
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        scratch(layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                            {dimension(config.attention->head_dim),
                             dimension(config.attention->num_attention_heads),
                             dimension(config.attention->num_key_value_heads)},
                            plan.kv_storage, text_envelope, 1, 1, 1));
        matrix(layout, DType::BF16, dimension(config.hidden_size), 1);
        matrix(layout, DType::BF16, dimension(config.hidden_size), 1);
        mtp_post_mixer(layout, 1, 1);
        proposal_scratch(layout, 1);
    };

    WorkspacePlan out;
    WorkspaceLayoutBuilder text_prefill;
    text_common_root(text_prefill, chunk);
    if (!plan.causal_scoring && config.gdn) {
        // A branch block's convolution window and recurrent state, held while its layer runs
        // (TextContext::prefill_final_batch).
        matrix(text_prefill, DType::BF16, dimension(config.gdn->conv_channels()),
               dimension(config.gdn->linear_conv_kernel_dim - 1));
        (void)text_prefill.alloc(DType::FP32, {dimension(config.gdn->linear_key_head_dim),
                                               dimension(config.gdn->linear_value_head_dim),
                                               dimension(config.gdn->linear_num_value_heads)});
    }
    target_body(text_prefill, 1, chunk, qwen3_5::TextPhase::Prefill, GdnWorkspacePath::Prefill, 1,
                1, chunk, prefill_envelope);
    if (!plan.causal_scoring) {
        // A shared pass reads up to kMaximumPrefillSegments rows: their hidden, logits, sampling
        // records, logprob gather and readout, and samples up to one row per lane.
        const auto vocabulary = dimension(parameters.model.resources().public_token_count);
        const auto rows       = static_cast<std::int32_t>(execution::kMaximumPrefillSegments);
        const auto lanes      = static_cast<std::int32_t>(plan.max_concurrency);
        auto tail             = text_prefill.scope();
        (void)text_prefill.alloc(DType::BF16, {dimension(config.hidden_size), rows});
        (void)text_prefill.alloc(logits_dtype, {dimension(config.vocab_size), rows});
        (void)text_prefill.alloc_bytes(sizeof(ops::SamplingConfig) *
                                       execution::kMaximumPrefillSegments);
        (void)text_prefill.alloc(DType::I32, {ops::kLogprobTopK, 1, rows});
        (void)text_prefill.alloc(DType::FP32, {ops::kLogprobTopK, 1, rows});
        (void)text_prefill.alloc(DType::FP32, {1, rows});
        (void)text_prefill.alloc(DType::I32, {ops::kReadoutLogprobsMaximum, rows});
        (void)text_prefill.alloc(DType::FP32, {ops::kReadoutLogprobsMaximum, rows});
        (void)text_prefill.alloc(DType::I32, {rows});
        (void)text_prefill.alloc(logits_dtype, {dimension(config.vocab_size), lanes});
        (void)text_prefill.alloc_bytes(sizeof(ops::SamplingConfig) *
                                       static_cast<std::size_t>(lanes));
        (void)text_prefill.alloc(DType::I32, {lanes});
        (void)text_prefill.alloc(DType::I32, {lanes});
        scratch(text_prefill, ops::sampling_workspace_capacity_bytes(vocabulary, 1, lanes));
        scratch(text_prefill, ops::logprob_topk_workspace_capacity_bytes(vocabulary, rows));
    }
    out.text_prefill = finish(text_prefill);

    if (plan.causal_scoring) {
        WorkspaceLayoutBuilder causal_score;
        const auto tile = static_cast<std::int32_t>(causal_score_tile(logits_dtype));
        matrix(causal_score, logits_dtype, dimension(config.vocab_size), tile);
        matrix(causal_score, DType::I32, 1, tile);
        matrix(causal_score, DType::FP32, 1, tile);
        linear_scratch(causal_score, parameters.text.output_head, 1, tile);
        out.causal_score = finish(causal_score);
    }

    if (!plan.causal_scoring) {
        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            WorkspaceLayoutBuilder ordinary;
            matrix(ordinary, DType::BF16, dimension(config.hidden_size), batch);
            target_body(ordinary, batch, batch, qwen3_5::TextPhase::Verify,
                        GdnWorkspacePath::Snapshot, batch, 1, 1, text_envelope);
            scratch(ordinary,
                    ops::sampling_workspace_capacity_bytes(
                        dimension(parameters.model.resources().public_token_count), batch, batch));
            scratch(ordinary,
                    ops::logprob_topk_workspace_capacity_bytes(
                        dimension(parameters.model.resources().public_token_count), batch));
            out.ordinary_round = std::max(out.ordinary_round, finish(ordinary));
        }
    }

    if (plan.features.mtp()) {
        WorkspaceLayoutBuilder mtp_prefill;
        text_common_root(mtp_prefill, chunk);
        target_body(mtp_prefill, 1, chunk, qwen3_5::TextPhase::Prefill, GdnWorkspacePath::Prefill,
                    1, 1, chunk, prefill_envelope);
        matrix(mtp_prefill, DType::I32, 1, chunk);
        if (plan.features.vision) {
            matrix(mtp_prefill, DType::BF16, dimension(config.hidden_size), chunk);
            (void)workspace::visual_scatter_indices(mtp_prefill, chunk);
        }
        mtp_prefill_chunk(mtp_prefill, 1, chunk, plan.features.vision);
        for (std::int32_t i = 1; i < drafts; ++i) {
            matrix(mtp_prefill, DType::BF16, dimension(config.hidden_size), 1);
            mtp_full_call(mtp_prefill, 1, text_envelope, true);
        }
        out.mtp_prefill = finish(mtp_prefill);

        WorkspaceLayoutBuilder mtp_batch;
        mtp_full_call(mtp_batch, verify, verify_envelope, false);
        WorkspaceLayoutBuilder mtp_ar;
        mtp_full_call(mtp_ar, 1, text_envelope, true);
        WorkspaceLayoutBuilder mtp_align;
        mtp_full_call(mtp_align, 1, text_envelope, false);
        WorkspaceLayoutBuilder mtp_proposal;
        proposal_scratch(mtp_proposal, 1);
        const std::size_t accept = ops::speculative_accept_coupled_drafts_workspace_capacity_bytes(
            dimension(parameters.model.resources().public_token_count), narrowest_drafts,
            verify_drafts, 1, 1);
        out.mtp_round = std::max({accept, finish(mtp_batch), finish(mtp_ar), finish(mtp_proposal)});
        out.ordinary_round = std::max(out.ordinary_round, finish(mtp_align));

        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            std::size_t target_bytes = 0;
            for (const std::int32_t width : round_widths) {
                const std::int32_t aggregate = batch * width;
                WorkspaceLayoutBuilder target;
                matrix(target, DType::BF16, dimension(config.hidden_size), aggregate);
                target_body(target, aggregate, aggregate, qwen3_5::TextPhase::Verify,
                            GdnWorkspacePath::ReplayRecord, batch, width, width, verify_envelope);
                target_bytes = std::max(target_bytes, finish(target));
            }

            const auto mtp_decode_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t width) {
                const std::int32_t tokens = batch * width;
                auto core                 = layout.scope();
                mtp_stem(layout, tokens, false);
                if (mtp_divided) {
                    mtp_divided_decode(layout, width, batch, verify_envelope);
                    return;
                }
                (void)workspace::mtp_attention_projection(layout, config, tokens);
                scratch(layout, execution::mtp_projection_workspace_bytes(
                                    parameters.mtp->projection, tokens, tokens));
                (void)workspace::mtp_attention_results(layout, config, tokens);
                if (plan.mtp_attention_window != 0) {
                    (void)workspace::paged_kv_window(
                        layout, static_cast<std::int32_t>(page_count(plan.capacity)), width, batch);
                }
                scratch(layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                                    {dimension(config.attention->head_dim),
                                     dimension(config.attention->num_attention_heads),
                                     dimension(config.attention->num_key_value_heads)},
                                    plan.kv_storage, verify_envelope, batch, width, width));
                (void)workspace::mtp_post_attention(layout, config, tokens);
                mtp_post_mixer(layout, tokens, tokens);
            };

            WorkspaceLayoutBuilder alignment;
            mtp_decode_core(alignment, verify);
            WorkspaceLayoutBuilder ar;
            mtp_decode_core(ar, 1);
            WorkspaceLayoutBuilder proposal;
            proposal_scratch(proposal, batch);
            const std::size_t batch_accept =
                ops::speculative_accept_coupled_drafts_workspace_capacity_bytes(
                    dimension(parameters.model.resources().public_token_count), narrowest_drafts,
                    verify_drafts, batch, batch);
            const std::size_t gather = ops::logprob_topk_workspace_capacity_bytes(
                dimension(parameters.model.resources().public_token_count), batch * verify);
            out.mtp_round = std::max({out.mtp_round, target_bytes, finish(alignment), finish(ar),
                                      finish(proposal), batch_accept, gather});
        }
    }

    if (plan.features.masked_draft()) {
        {
            const auto* draft = &*parameters.model.config().draft;
            // DSpine: its Drafter's own workspace at the ring the state images hold, and the
            // target head's at each call's columns.
            const std::uint32_t dspine_ring =
                draft->dspine ? dspine::ring_capacity(*draft->dspine, plan.capacity) : 0U;
            if (draft->dspine && parameters.dspine == nullptr) {
                throw std::logic_error("--spec dspine plans without its drafter");
            }
            const auto head_bytes = [&](std::int32_t columns) {
                WorkspaceLayoutBuilder head;
                linear_scratch(head, parameters.text.output_head, 1, columns);
                std::size_t bytes = finish(head);
                // A proposal ranks its candidates in one pass where the Op takes the head.
                if (const auto* ranked =
                        execution::dspine_ranked_head(parameters, plan.proposal_head)) {
                    const Weight& w = ranked->weight;
                    bytes           = std::max(
                        bytes, execution::rotated_workspace_bytes(
                                   ranked->hadamard_signs, static_cast<std::int32_t>(w.k), columns,
                                   ops::linear_topk_workspace_capacity_bytes(w.qtype, w.n, w.k, 1,
                                                                             columns)));
                } else if (plan.proposal_head == ProposalHead::Optimized) {
                    // A proposal head of another form: its dense logits, ranked.
                    const auto& proposal = parameters.proposal->head;
                    WorkspaceLayoutBuilder dense;
                    matrix(dense, DType::BF16, proposal.weight.n, columns);
                    linear_scratch(dense, proposal, 1, columns);
                    scratch(dense, ops::candidate_topk_workspace_capacity_bytes(columns));
                    bytes = std::max(bytes, finish(dense));
                }
                return bytes;
            };
            const auto dflash_context_capacity = [&](std::int32_t width, std::int32_t batch,
                                                     bool compact_input) {
                const auto tokens = width * batch;
                WorkspaceLayoutBuilder layout;
                if (compact_input) {
                    matrix(layout, DType::BF16,
                           dimension(config.hidden_size *
                                     std::uint64_t(draft->target_layer_ids.size())),
                           tokens);
                }
                if (draft->dspine.has_value()) {
                    scratch(layout, dspine::Drafter::context_workspace_bytes(
                                        *parameters.dspine, dspine_ring, width, batch));
                    return finish(layout);
                }
                if (draft->dflash2.has_value()) {
                    const auto local_width =
                        std::min(width, dimension(draft->sliding_window.value_or(0)));
                    (void)workspace::dflash_context(layout, config, *draft, local_width * batch);
                    linear_scratch(layout, parameters.draft->feature_projection,
                                   local_width * batch, local_width * batch);
                    if (draft->fused_dflash2(config.hidden_size)) {
                        scratch(layout, ops::context_kv_materialize_workspace_capacity_bytes(
                                            batch, local_width, local_width));
                    } else {
                        // The composed route appends layer by layer, as DFlash does.
                        auto layer = layout.scope();
                        (void)workspace::dflash_context_layer(layout, config, *draft,
                                                              local_width * batch);
                    }
                    return finish(layout);
                }
                (void)workspace::dflash_context(layout, config, *draft, tokens);
                linear_scratch(layout, parameters.draft->feature_projection, tokens, tokens);
                {
                    auto layer = layout.scope();
                    (void)workspace::dflash_context_layer(layout, config, *draft, tokens);
                }
                return finish(layout);
            };
            const auto dflash_proposal_capacity = [&](std::int32_t width, std::int32_t batch) {
                WorkspaceLayoutBuilder layout;
                const std::int32_t tokens          = width * batch;
                const std::int32_t proposal_drafts = width - 1;
                if (draft->dspine.has_value()) {
                    scratch(layout, dspine::Drafter::proposal_workspace_bytes(
                                        *parameters.dspine, logits_dtype, dspine_ring,
                                        plan.capacity, width, batch,
                                        head_bytes(proposal_drafts * batch)));
                    return finish(layout);
                }
                matrix(layout, DType::BF16, dimension(config.hidden_size), tokens);
                if (draft->dflash2.has_value()) {
                    const bool fused = draft->fused_dflash2(config.hidden_size);
                    // The composed route (execution/draft.cpp) keeps a branch's coefficients beside
                    // its prepared input and projects through Linear on both sides.
                    const auto prepare = [&](bool attention_branch) {
                        (void)workspace::dflash2_branch(layout, config, *draft, width, batch);
                        if (fused) {
                            scratch(
                                layout,
                                ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
                                    width, width, batch, batch));
                            return;
                        }
                        matrix(layout, DType::FP32,
                               dimension(2ULL * draft->dflash2->conv_kernel_size *
                                         config.hidden_size / draft->dflash2->conv_group_size),
                               tokens);
                        auto inner = layout.scope();
                        matrix(layout, DType::BF16, dimension(config.hidden_size), tokens);
                        for (const auto& block : parameters.draft->layers) {
                            linear_scratch(layout,
                                           (attention_branch ? block.attention_conv
                                                             : block.mlp_conv)
                                               ->kernel_projection,
                                           tokens, tokens);
                        }
                    };
                    const auto finish_branch = [&](bool attention_branch) {
                        if (fused) {
                            scratch(layout,
                                    ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
                                        dimension(attention_branch
                                                      ? draft->attention.query_width()
                                                      : draft->intermediate_size),
                                        width, width, batch, batch));
                            return;
                        }
                        auto inner = layout.scope();
                        matrix(layout, DType::BF16, dimension(config.hidden_size), tokens);
                        for (const auto& block : parameters.draft->layers) {
                            linear_scratch(layout, attention_branch ? block.output : block.mlp.down,
                                           tokens, tokens);
                        }
                    };
                    {
                        auto attention = layout.scope();
                        prepare(true);
                        matrix(layout, DType::BF16, dimension(draft->attention.query_width()),
                               tokens);
                        matrix(layout, DType::BF16, dimension(draft->attention.key_width()),
                               tokens);
                        matrix(layout, DType::BF16, dimension(draft->attention.key_width()),
                               tokens);
                        matrix(layout, DType::BF16, dimension(draft->attention.query_width()),
                               tokens);
                        scratch(layout, ops::sliding_window_attention_workspace_capacity_bytes(
                                            {dimension(draft->attention.head_dim),
                                             dimension(draft->attention.num_attention_heads),
                                             dimension(draft->attention.num_key_value_heads)},
                                            dimension(draft->sliding_window.value_or(0)),
                                            {0, plan.capacity}, width, width, batch));
                        finish_branch(true);
                    }
                    {
                        auto mlp = layout.scope();
                        prepare(false);
                        matrix(layout, DType::BF16, dimension(draft->intermediate_size), tokens);
                        for (const auto& block : parameters.draft->layers) {
                            const auto& p = block.mlp.gate_up;
                            scratch(layout, execution::rotated_workspace_bytes(
                                                p.hadamard_signs, p.weight.k, tokens,
                                                ops::linear_swiglu_workspace_capacity_bytes(
                                                    p.weight.qtype, p.weight.n, p.weight.k,
                                                    p.policy, tokens, tokens)));
                        }
                        finish_branch(false);
                    }
                    const auto mask_columns = proposal_drafts * batch;
                    matrix(layout, DType::BF16, dimension(config.hidden_size), mask_columns);
                    matrix(layout, DType::FP32, dimension(draft->dflash2->selector_top_k),
                           mask_columns);
                    const auto& head = plan.proposal_head == ProposalHead::Optimized
                                           ? parameters.proposal->head
                                           : parameters.draft->output_head;
                    if (fused) {
                        if (execution::rotated(head.hadamard_signs)) {
                            matrix(layout, DType::BF16, dimension(config.hidden_size),
                                   mask_columns);
                        }
                        scratch(layout, ops::linear_topk_workspace_capacity_bytes(
                                            head.weight.qtype, head.weight.n, head.weight.k,
                                            mask_columns, mask_columns));
                    } else {
                        // The packed mask columns before the final norm, then the dense logits
                        // with the head's scratch or the top-k's.
                        scratch(layout, std::size_t(config.hidden_size) * mask_columns * 2);
                        auto dense = layout.scope();
                        matrix(layout, DType::BF16, head.weight.n, mask_columns);
                        linear_scratch(layout, head, mask_columns, mask_columns);
                        scratch(layout, ops::candidate_topk_workspace_capacity_bytes(mask_columns));
                    }
                    matrix(layout, DType::BF16, dimension(draft->dflash2->selector_rank),
                           mask_columns);
                    linear_scratch(layout, parameters.draft->selector->hidden_projection,
                                   mask_columns, mask_columns);
                    scratch(layout, ops::candidate_selector_path_workspace_capacity_bytes(
                                        proposal_drafts, proposal_drafts, batch, batch));
                    return finish(layout);
                }
                {
                    auto attention = layout.scope();
                    (void)workspace::dflash_attention(layout, config, *draft, tokens);
                    scratch(layout,
                            std::max(ops::sliding_window_attention_workspace_capacity_bytes(
                                         {dimension(draft->attention.head_dim),
                                          dimension(draft->attention.num_attention_heads),
                                          dimension(draft->attention.num_key_value_heads)},
                                         dimension(draft->sliding_window.value_or(0)),
                                         {0, plan.capacity}, width, width, batch),
                                     ops::context_softmax_attention_workspace_capacity_bytes(
                                         {dimension(draft->attention.head_dim),
                                          dimension(draft->attention.num_attention_heads),
                                          dimension(draft->attention.num_key_value_heads)},
                                         {0, plan.capacity}, width, width, batch)));
                    for (const auto& block : parameters.draft->layers) {
                        add_scratch(layout, block.output, tokens, tokens);
                    }
                }
                {
                    auto mlp = layout.scope();
                    (void)workspace::dflash_mlp(layout, config, *draft, tokens);
                    for (const auto& block : parameters.draft->layers) {
                        const auto& p = block.mlp.gate_up;
                        scratch(layout, execution::rotated_workspace_bytes(
                                            p.hadamard_signs, p.weight.k, tokens,
                                            ops::linear_swiglu_workspace_capacity_bytes(
                                                p.weight.qtype, p.weight.n, p.weight.k, p.policy,
                                                tokens, tokens)));
                    }
                    for (const auto& block : parameters.draft->layers) {
                        add_scratch(layout, block.mlp.down, tokens, tokens);
                    }
                }
                matrix(layout, DType::BF16, dimension(config.hidden_size), proposal_drafts * batch);
                matrix(layout, DType::BF16, dimension(config.hidden_size), proposal_drafts * batch);
                if (plan.proposal_head == ProposalHead::Optimized) {
                    matrix(layout, DType::BF16, dimension(parameters.proposal->rows),
                           proposal_drafts * batch);
                } else {
                    matrix(layout, DType::BF16, dimension(config.vocab_size),
                           proposal_drafts * batch);
                }
                const auto& head = plan.proposal_head == ProposalHead::Optimized
                                       ? parameters.proposal->head
                                       : parameters.draft->output_head;
                linear_scratch(layout, head, proposal_drafts * batch, proposal_drafts * batch);
                return finish(layout);
            };

            out.dflash_context = dflash_context_capacity(chunk, 1, false);
            if (draft->dspine.has_value()) {
                // The head-write table's derivation at startup.
                const auto concurrency = static_cast<std::int32_t>(plan.max_concurrency);
                out.dflash_context = std::max(
                    out.dflash_context,
                    dspine::Drafter::prepare_workspace_bytes(
                        *parameters.dspine, logits_dtype, concurrency, proposal_drafts + 1,
                        head_bytes(dspine::Drafter::prepare_columns(
                            *parameters.dspine, concurrency, proposal_drafts + 1))));
            }
            for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
                 ++batch) {
                std::size_t target_bytes = 0;
                for (const std::int32_t width : round_widths) {
                    const std::int32_t aggregate = width * batch;
                    WorkspaceLayoutBuilder target;
                    matrix(target, DType::BF16, dimension(config.hidden_size), aggregate);
                    target_body(target, aggregate, aggregate, qwen3_5::TextPhase::Verify,
                                GdnWorkspacePath::ReplayRecord, batch, width, width,
                                verify_envelope);
                    target_bytes = std::max(target_bytes, finish(target));
                }
                const std::size_t accept =
                    draft->dflash2.has_value() || draft->dspine.has_value()
                        ? ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                              dimension(parameters.model.resources().public_token_count), {false},
                              narrowest_drafts, verify_drafts, batch, batch)
                        : ops::speculative_accept_coupled_drafts_workspace_capacity_bytes(
                              dimension(parameters.model.resources().public_token_count),
                              narrowest_drafts, verify_drafts, batch, batch);
                // A copy round above batch one also runs the drafter, at the copy window, and a
                // round after a copy round appends up to the widest window of target features.
                const std::size_t proposal =
                    dflash_proposal_capacity((batch > 1 ? proposal_drafts : drafts) + 1, batch);
                const std::size_t gather   = ops::logprob_topk_workspace_capacity_bytes(
                    dimension(parameters.model.resources().public_token_count), batch * verify);
                out.dflash_round =
                    std::max({out.dflash_round, target_bytes, accept,
                              dflash_context_capacity(verify, batch, true), proposal, gather});
            }
        }
    }

    out.general_capacity =
        std::max({out.text_prefill, out.ordinary_round, out.mtp_prefill, out.mtp_round,
                  out.dflash_context, out.dflash_round, out.causal_score});
    if (parameters.text.split_execution()) {
        // Around its layers a stage holds the residual it received and its copy of the control
        // block (rank 0 holds the packed block instead, which is smaller), alive for the whole pass
        // and so on top of the layers' own peak. Alignment slack for both allocations.
        const std::uint64_t columns = stage_boundary_columns(plan);
        const std::size_t residual =
            static_cast<std::size_t>(columns) * static_cast<std::size_t>(config.hidden_size) * 2U;
        const auto control = static_cast<std::size_t>(stage_control_bytes(columns));
        out.general_capacity =
            checked_add(out.general_capacity, checked_add(residual, control, "stage boundary") + 1024U,
                        "stage boundary workspace");
        // With a masked drafter a later stage also keeps a copy of each tapped layer's output until
        // it has crossed, and rank 0 receives them one at a time into one residual's room; the
        // stage with the most taps sets the size. Tensor-parallel ranks tap rank 0's replica.
        const auto& draft = parameters.model.config().draft;
        if (plan.features.masked_draft() && draft && !parameters.text.tensor_parallel()) {
            std::size_t widest = 0;
            for (std::size_t stage = 1; stage < parameters.text.rank_count; ++stage) {
                widest = std::max(
                    widest, parameters.text.stage_taps(stage, draft->target_layer_ids).size());
            }
            if (widest != 0) {
                const std::size_t features = checked_mul(residual, widest, "stage feature taps");
                out.general_capacity       = checked_add(out.general_capacity, features + 256U,
                                                         "stage feature tap workspace");
            }
        }
    }
    out.capacity = out.general_capacity;
    if (plan.features.vision) {
        const std::uint32_t merged = vision_item_token_bound(plan.capacity, plan.features);
        if (plan.features.host_staged_vision()) {
            out.vision_resident = false;
            out.vision =
                plan.features.cpu_vision()
                    ? execution::plan_cpu_vision_workspace(
                          static_cast<std::int32_t>(parameters.model.config().text.hidden_size),
                          merged)
                    : execution::plan_vision_window_workspace(parameters, merged);
            out.vision_bridge_offset = checked_add(out.general_capacity, 255, "bridge offset") &
                                       ~std::size_t{255};
            out.vision_bridge_bytes =
                checked_mul(static_cast<std::size_t>(out.vision->output_hidden), 2, "bridge column");
            out.capacity = std::max(out.capacity, checked_add(out.vision_bridge_offset,
                                                              out.vision_bridge_bytes,
                                                              "bridge column extent"));
        } else {
            out.vision = execution::VisionContext::plan_workspace(
                *parameters.model.config().vision, *parameters.vision, merged,
                out.general_capacity);
            out.capacity = std::max(out.capacity, out.vision->capacity_bytes);
        }
    }
    return out;
}

void validate_target_options(const execution::Parameters& parameters, DeviceContext& device,
                             const EngineOptions& options) {
    if (!parameters.model.config().text.attention ||
        parameters.model.config().text.full_attention_layers == 0) {
        throw std::invalid_argument("Qwen3.5 Program requires at least one full-attention layer");
    }
    if (parameters.model.options() != models::load_options(options)) {
        throw std::invalid_argument(
            "loaded components do not match the requested execution options");
    }
    // Past the native window positions run unscaled RoPE or YaRN, to four times the window: the
    // extension Qwen documents for these models (1,048,576 tokens on a 262,144-token window).
    constexpr std::uint64_t kPositionExtension = 4;
    if ((parameters.draft || parameters.dspine) &&
        options.max_context >
            kPositionExtension * parameters.model.config().draft->max_position_embeddings) {
        throw std::invalid_argument("max_context exceeds the selected draft position capacity");
    }
    if (options.max_context == 0 ||
        options.max_context >
            kPositionExtension * parameters.model.config().text.max_position_embeddings) {
        throw std::invalid_argument("max_context exceeds the configured position capacity");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("prefill_chunk must be a nonzero multiple of 128");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("max_concurrency must be in [1,8]");
    }
    // Decode and verify rounds run the target's projections with the prefill's Linear policies, one
    // call over every lane's verify columns. The cuBLAS prefill route has to stay out of them -- it
    // would put a weight-sized scratch into every round workspace and change decode numerics -- so
    // the widest round sits at or below the floor no device profile moves its width gate past.
    static_assert(kMaximumConcurrency * kMtpDecodeMaximumWidth <=
                      static_cast<std::uint32_t>(ops::kCublasPrefillFloorTokens),
                  "the widest decode/verify round must stay below the cuBLAS prefill width gate");
    if (parameters.text.rank_count != device.size()) {
        throw std::invalid_argument("the model is split over " +
                                    std::to_string(parameters.text.rank_count) +
                                    (parameters.text.tensor_parallel() ? " tensor-parallel ranks"
                                                                       : " pipeline stages") +
                                    " but " + std::to_string(device.size()) +
                                    " devices are attached");
    }
    const std::uint32_t logical_pages = page_count(options.max_context);
    const std::uint32_t minimum_pages = std::max(logical_pages, options.max_concurrency);
    const std::uint64_t maximum_pages64 =
        maximum_main_page_groups(options.max_concurrency, logical_pages, options.context_cache);
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit: {
        if (options.kv_capacity.explicit_tokens < options.max_context) {
            throw std::invalid_argument("kv_capacity must be at least max_context");
        }
        const std::uint32_t requested_pages = page_count(options.kv_capacity.explicit_tokens);
        if (requested_pages < minimum_pages || requested_pages > maximum_pages64) {
            throw std::invalid_argument(
                "kv_capacity is outside the usable range for max_context and max_concurrency");
        }
        break;
    }
    case KvCapacityMode::Automatic:
        break;
    default:
        throw std::invalid_argument("unknown kv_capacity policy");
    }
    if (options.speculative.mtp_attention_window != 0 &&
        options.speculative.backend != SpeculativeBackend::Mtp) {
        throw std::invalid_argument("an MTP attention window requires the MTP backend");
    }
    if (options.speculative.mtp_policy == MtpDraftPolicy::Adaptive &&
        options.speculative.backend != SpeculativeBackend::Mtp &&
        !is_masked_draft_backend(options.speculative.backend)) {
        throw std::invalid_argument(
            "--adaptive-mtp requires --spec mtp, dflash, dflash2 or dspine");
    }
    switch (options.speculative.backend) {
    case SpeculativeBackend::None:
        if (options.speculative.draft_tokens != 0 ||
            models::planned_proposal_head(options.speculative.proposal_head) !=
                ProposalHead::Full) {
            throw std::invalid_argument(
                "disabled speculative decoding requires draft_tokens=0 and the full proposal head");
        }
        break;
    case SpeculativeBackend::Mtp:
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens > kMaximumMtpDraftTokens) {
            throw std::invalid_argument("MTP draft window must be in [1,15]");
        }
        if (options.speculative.mtp_attention_window != 0 &&
            options.speculative.mtp_attention_window < options.speculative.draft_tokens + 1) {
            throw std::invalid_argument(
                "MTP attention window must hold at least draft_tokens + 1 keys");
        }
        break;
    case SpeculativeBackend::DFlash:
    case SpeculativeBackend::DFlash2:
        if (!parameters.draft || (options.speculative.backend == SpeculativeBackend::DFlash2) !=
                                     parameters.model.config().draft->dflash2.has_value()) {
            throw std::invalid_argument(
                "selected masked draft backend is not supported by this target");
        }
        if (options.speculative.draft_tokens == 0 || options.speculative.draft_tokens > 15) {
            throw std::invalid_argument("masked draft window must be in [1,15]");
        }
        break;
    case SpeculativeBackend::DSpine: {
        const auto& draft = parameters.model.config().draft;
        if (parameters.dspine == nullptr || !draft || !draft->dspine) {
            throw std::invalid_argument("--spec dspine needs its drafter artifact (--draft-model)");
        }
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens + 1 > draft->dspine->block_size ||
            options.speculative.draft_tokens > 15) {
            throw std::invalid_argument(
                "the DSpine draft window must be in [1, min(15, block_size - 1)]");
        }
        // Copy rounds of several requests run the drafter at the copy window.
        if (masked_proposal_window(options.speculative.draft_tokens,
                                   options.speculative.ngram_draft_tokens,
                                   options.max_concurrency) +
                1 >
            draft->dspine->block_size) {
            throw std::invalid_argument(
                "above one request, --ngram-draft-tokens with DSpine must be at most its "
                "block_size - 1, the widest proposal its drafter makes");
        }
        break;
    }
    case SpeculativeBackend::DSpark:
        throw std::invalid_argument("--spec dspark drafts for DeepSeek-V4.1 only");
    }
    // 12.0 is admitted alongside the qualified capabilities: consumer Blackwell runs the same
    // warp-level mma.sync schedules this family is written against, with the same 100 KiB of shared
    // memory per SM. The occupancy and split-k choices are still the sm_86 ones, so the card is
    // supported here in the sense of running correctly, not of being tuned for. 7.0 (Tesla V100)
    // runs Volta's code, whose kernels emulate the Ampere warp primitives. 7.5 needs Turing's code
    // (a build with 75 among its architectures), whose kernels avoid the sm_80 instructions and fit
    // its 64 KiB of shared memory.
    // Every rank is checked: pipeline stages may mix capabilities (core/device.cu).
    for (std::size_t rank = 0; rank < device.size(); ++rank) {
        const RankContext& context = device.rank(rank);
        const int capability       = context.compute_capability();
        if (capability == 70 || capability == 80 || capability == 86 || capability == 89 ||
            capability == 120 ||
            (capability == 75 && ops::device_tier(context.device) == ops::DeviceTier::Turing)) {
            continue;
        }
        std::string message =
            "Qwen3.5 family runtime requires compute capability 7.0, 7.5, 8.0, 8.6, 8.9 or 12.0 "
            "(7.5 needs a build with 75 among CMAKE_CUDA_ARCHITECTURES)";
        if (device.model_parallel()) {
            message += "; CUDA device " + std::to_string(context.device) + " has " +
                       std::to_string(capability / 10) + "." + std::to_string(capability % 10);
        }
        throw std::invalid_argument(message);
    }
    if (options.speculative.ngram_draft_tokens != 0 &&
        ((options.speculative.backend != SpeculativeBackend::DFlash2 &&
          options.speculative.backend != SpeculativeBackend::DFlash &&
          options.speculative.backend != SpeculativeBackend::DSpine &&
          options.speculative.backend != SpeculativeBackend::Mtp) ||
         options.speculative.ngram_draft_tokens > 63 || options.speculative.ngram_min_match < 4 ||
         options.speculative.ngram_min_match > 64 ||
         (options.speculative.ngram_draft_tokens > 15 && options.max_concurrency != 1))) {
        throw std::invalid_argument(
            "ngram requires MTP/DFlash/DFlash2/DSpine, K1..63 and match 4..64; the GDN conv-record "
            "workspace admits at most 16 verification columns for a multi-request batch, so K "
            "above 15 requires concurrency one");
    }
    // A copy round of a masked drafter verifies the drafter's proposal inside its own width
    // (execution/draft.cpp), so its width is at least the draft window.
    if (options.speculative.ngram_draft_tokens != 0 &&
        options.speculative.backend != SpeculativeBackend::Mtp &&
        options.speculative.ngram_draft_tokens < options.speculative.draft_tokens) {
        throw std::invalid_argument(
            "--ngram-draft-tokens must be at least --draft-tokens with a masked drafter: a copy "
            "round verifies the drafter's proposal inside its own width");
    }
}

std::unique_ptr<SequencePlanImpl> build_sequence_candidate(const SequencePlanningInputs& inputs,
                                                           std::uint32_t main_page_groups) {
    if (main_page_groups == 0) {
        throw std::invalid_argument("Main KV physical page count must be positive");
    }
    auto impl                 = std::make_unique<SequencePlanImpl>();
    impl->parameters          = inputs.parameters;
    impl->capacity            = inputs.capacity;
    impl->main_page_groups    = main_page_groups;
    impl->kv_capacity         = static_cast<std::uint32_t>(checked_i32(
        static_cast<std::uint64_t>(main_page_groups) * static_cast<std::uint32_t>(kPagedKVPageSize),
        "resolved Paged KV capacity exceeds int32"));
    impl->max_concurrency     = inputs.max_concurrency;
    impl->prefill_chunk       = inputs.prefill_chunk;
    impl->fast_prefill_kernel = inputs.fast_prefill_kernel;
    impl->invariant_attention = inputs.invariant_attention;
    impl->draft_window        = inputs.draft_window;
    impl->lookup_ngram        = inputs.lookup_ngram;
    impl->mtp_policy          = inputs.mtp_policy;
    impl->ngram_draft_window  = inputs.ngram_draft_window;
    impl->ngram_min_match     = inputs.ngram_min_match;
    impl->speculative_backend = inputs.speculative_backend;
    impl->proposal_head       = inputs.proposal_head;
    impl->rope_yarn           = inputs.rope_yarn;
    impl->mtp_attention_window = inputs.mtp_attention_window;
    impl->features            = inputs.features;
    impl->expert_cache_bytes  = inputs.expert_cache_bytes;
    impl->use_cuda_graph      = inputs.use_cuda_graph;
    impl->causal_scoring      = inputs.causal_scoring;
    impl->structured_output   = inputs.structured_output;
    impl->gumbel_sampling     = inputs.gumbel_sampling;
    impl->device              = inputs.device;
    impl->context_cache       = inputs.context_cache;
    impl->kv_storage          = inputs.kv_storage;
    impl->steering_adapters   = inputs.steering_adapters;
    impl->steering_vectors    = inputs.steering_vectors;
    impl->persistent          = persistent_layout(*impl);
    if (impl->context_cache.enabled && impl->context_cache.mode == ContextCacheMode::Hybrid) {
        // The whole Host budget is the hybrid slab pool that KV blocks and state snapshots share
        // (docs/maintainer/hybrid-prefix-cache-spec.md §5.4); the Legacy Host pools stay empty.
        // A budget too small for one snapshot is rejected here, before anything is allocated.
        // The backend pool is the MTP KV pool, or the DFlash draft's paged full-attention pool.
        const KVPageGeometry* backend = nullptr;
        if (impl->speculative_backend == SpeculativeBackend::Mtp &&
            impl->persistent.decoder.mtp_kv) {
            backend = &impl->persistent.decoder.mtp_kv->pages.spec.geometry;
        } else if (impl->persistent.dflash && impl->persistent.dflash->full) {
            backend = &impl->persistent.dflash->full->pages.spec.geometry;
        }
        const HybridHostLayout host =
            plan_hybrid_host_layout(impl->persistent.decoder.text_kv.pages.spec.geometry, backend,
                                    impl->persistent.state_images.host.image_bytes);
        (void)hybrid_host_slabs(host, impl->context_cache.host_cache_budget_bytes.value_or(0U));
        impl->context_cache.host_state_slots       = 0;
        impl->context_cache.host_kv_capacity_bytes = 0;
    } else if (impl->context_cache.host_cache_budget_bytes) {
        // The budget is resolved on the finished layout, before anything consumes the plan's
        // context-cache shape: the Program sizes its Host pools from it and the Engine publishes
        // the same resolved counts to its own ResourceManager and frontend, so the anchors the
        // capture path creates always fit the inventory that was paid for.
        if (!impl->context_cache.max_private_continuations ||
            *impl->context_cache.max_private_continuations == 0 ||
            !impl->context_cache.max_shared_prefixes) {
            throw std::logic_error("Qwen3.5 context cache options are not normalized");
        }
        resolve_host_cache_budget(impl->context_cache,
                                  *impl->context_cache.max_private_continuations,
                                  *impl->context_cache.max_shared_prefixes, impl->capacity,
                                  impl->persistent.state_images.host.image_bytes,
                                  impl->persistent.host_kv_text_page_stride);
    }
    impl->workspace           = build_workspace_plan(*impl);
    if (impl->use_cuda_graph) {
        // Definitions remain per execution profile, but only one executable is instantiated for
        // each reachable node-topology class. These bounds cover the largest profile installed in
        // each class and the driver/module state materialized while qualifying all definitions.
        const auto& text_attention = *impl->parameters->model.config().text.attention;
        const ops::AttentionHeadGeometry attention_geometry{
            static_cast<std::int32_t>(text_attention.head_dim),
            static_cast<std::int32_t>(text_attention.num_attention_heads),
            static_cast<std::int32_t>(text_attention.num_key_value_heads)};
        if (impl->speculative_backend == SpeculativeBackend::None) {
            impl->graph_allowance_bytes = checked_mul(
                graph_topology_allowance(
                    ordinary_graph_profiles(impl->capacity, attention_geometry, impl->kv_storage),
                    [](GraphExecutionProfile) { return 12ULL * kMiB; },
                    "ordinary graph allowance"),
                impl->max_concurrency, "ordinary exact-b graph allowance");
        } else if (impl->speculative_backend == SpeculativeBackend::Mtp) {
            // Adaptive MTP captures one graph set per verification width it can select, and copy
            // rounds one more at the ngram window for every batch size they admit.
            const auto family_allowance = [&](std::uint32_t verify_window) {
                const auto profiles =
                    mtp_graph_profiles(impl->capacity, verify_window, impl->draft_window,
                                       attention_geometry, impl->kv_storage);
                return graph_topology_allowance(
                    profiles,
                    [&](GraphExecutionProfile profile) {
                        const std::uint64_t final_visible = std::min<std::uint64_t>(
                            impl->capacity, static_cast<std::uint64_t>(profile.max) +
                                                verify_window + impl->draft_window);
#ifdef NINFER_SM8X_COMPAT
                        if (final_visible <= 4096) {
                            // The reduced-startup graph set still consumes 35.8 MiB at C1/K3 and
                            // 43.1 MiB at C1/K4 on SM86. K2 retains the smaller qualified
                            // allowance; reserve one 64 MiB class for K3 and deeper captures.
                            return (verify_window >= 3 ? 64ULL : 16ULL) * kMiB;
                        }
                        return 86ULL * kMiB;
#else
                        return (final_visible <= 4096 ? 12ULL : 82ULL) * kMiB;
#endif
                    },
                    "MTP graph allowance");
            };
            const std::uint32_t first_window = impl->mtp_policy == MtpDraftPolicy::Adaptive
                                                   ? mtp_minimum_adaptive_window(impl->draft_window)
                                                   : impl->draft_window;
            std::size_t per_batch_allowance  = 0;
            for (std::uint32_t verify_window = first_window; verify_window <= impl->draft_window;
                 ++verify_window) {
                per_batch_allowance = checked_add(
                    per_batch_allowance, family_allowance(verify_window), "MTP graph allowance");
            }
            impl->graph_allowance_bytes = checked_mul(per_batch_allowance, impl->max_concurrency,
                                                      "MTP exact-b graph allowance");
            if (impl->ngram_draft_window != 0) {
                const std::uint32_t copy_batches =
                    impl->ngram_draft_window > 15U ? 1U : impl->max_concurrency;
                impl->graph_allowance_bytes =
                    checked_add(impl->graph_allowance_bytes,
                                checked_mul(family_allowance(impl->ngram_draft_window),
                                            copy_batches, "ngram MTP graph allowance"),
                                "MTP exact-b graph allowance");
            }
        } else {
            // Each DFlash family's profiles are captured at the family's own window for every
            // batch size, on the one frame viewed at that width.
            const auto class_allowance = [&](std::uint32_t batch_size, std::uint32_t window) {
                const std::uint32_t width = window;
                const auto profiles = dflash_graph_profiles(impl->speculative_backend,
                                                            impl->capacity, width, batch_size);
                return graph_topology_allowance(
                    profiles,
                    [&](GraphExecutionProfile profile) {
                        const std::uint64_t final_visible = std::min<std::uint64_t>(
                            impl->capacity, static_cast<std::uint64_t>(profile.max) + width + 1ULL);
                        return (final_visible <= 4096 ? 64ULL : 96ULL) * kMiB;
                    },
                    "DFlash graph allowance");
            };
            // Under adaptive MTP the neural rounds capture one set per width they may select.
            const auto neural_windows = masked_neural_windows(
                impl->draft_window, impl->mtp_policy == MtpDraftPolicy::Adaptive);
            for (std::uint32_t batch_size = 1; batch_size <= impl->max_concurrency; ++batch_size) {
                for (const std::uint32_t window : neural_windows) {
                    impl->graph_allowance_bytes =
                        checked_add(impl->graph_allowance_bytes, class_allowance(batch_size, window),
                                    "DFlash exact-b graph allowance");
                }
                if (impl->ngram_draft_window != 0) {
                    impl->graph_allowance_bytes =
                        checked_add(impl->graph_allowance_bytes,
                                    class_allowance(batch_size, impl->ngram_draft_window),
                                    "DFlash exact-b graph allowance");
                }
            }
        }
        if (inputs.cuda_graph_allowance_bytes != 0) {
            impl->graph_allowance_bytes = inputs.cuda_graph_allowance_bytes;
        }
    }

    impl->device_reservation_bytes = checked_add(
        checked_add(impl->persistent.bytes, impl->workspace.capacity, "sequence memory plan"),
        impl->graph_allowance_bytes, "sequence graph allowance");
    // A further device runs its stage in a workspace of the same general size and holds its own
    // persistent state. Its graph allowance is the primary device's scaled by the share of layers
    // it runs; the primary keeps the whole figure, which only over-reserves it a little.
    const auto& text = impl->parameters->text;
    // A tensor-parallel exchange holds two FP32 partials of the widest pass on every rank, with
    // the buffers of the transports it may take.
    const std::size_t exchange_bytes =
        text.tensor_parallel()
            ? TensorParallelExchange::device_bytes(
                  tensor_parallel_slot_bytes(*impl), text.rank_count, inputs.peer_access,
                  inputs.features.tensor_parallel_transport,
                  inputs.features.tensor_parallel_prompt_partials != TensorParallelPartials::Fp32)
            : 0U;
    impl->device_reservation_bytes =
        checked_add(impl->device_reservation_bytes, exchange_bytes, "tensor-parallel exchange");
    for (std::size_t rank = 1; rank < text.rank_count; ++rank) {
        const std::uint64_t layers_here =
            text.tensor_parallel() ? text.layers.size()
                                   : text.stage_end(rank) - text.stage_begin[rank];
        const std::size_t graph_share   = static_cast<std::size_t>(
            (static_cast<std::uint64_t>(impl->graph_allowance_bytes) * layers_here) /
            std::max<std::size_t>(text.layers.size(), 1));
        impl->extra_rank_reservation_bytes.push_back(checked_add(
            checked_add(checked_add(impl->persistent.extra_rank_bytes[rank - 1],
                                    impl->workspace.general_capacity, "further device memory plan"),
                        exchange_bytes, "tensor-parallel exchange"),
            graph_share, "further device graph allowance"));
    }
    return impl;
}

} // namespace

std::size_t divided_head_link_bytes(const SequencePlanImpl& plan) {
    const auto& parameters = *plan.parameters;
    std::size_t widest     = 0;
    const auto widest_of   = [&](const execution::HeadShares& head, DType dtype) {
        for (std::size_t rank = 1; rank < head.shares.size(); ++rank) {
            widest = std::max(widest, static_cast<std::size_t>(head.rows[rank + 1] - head.rows[rank]) *
                                          dtype_size(dtype));
        }
    };
    const auto draft_dtype = [](const execution::LinearParameters& head) {
        return execution::fp32_proposal_logits(head) ? DType::FP32 : DType::BF16;
    };
    widest_of(parameters.text.output_head_shares,
              execution::target_logits_dtype(parameters.text.output_head));
    if (parameters.proposal) {
        widest_of(parameters.proposal->head_shares, draft_dtype(parameters.proposal->head));
    }
    if (parameters.mtp) {
        widest_of(parameters.mtp->output_head_shares, draft_dtype(parameters.mtp->output_head));
    }
    return checked_mul(widest,
                       static_cast<std::size_t>(plan.max_concurrency) *
                           (widest_verify_window(plan) + 1U),
                       "divided head link");
}

std::size_t tensor_parallel_slot_bytes(const SequencePlanImpl& plan) {
    const auto& text   = plan.parameters->model.config().text;
    std::size_t values = static_cast<std::size_t>(text.hidden_size);
    const auto& shares = plan.parameters->text.tensor_shares;
    if (!shares.empty() && shares.front().attention_positions) {
        values = std::max<std::size_t>(
            values, ops::causal_attention_partial_elements(
                        static_cast<std::int32_t>(text.attention->num_attention_heads), 1));
    }
    return checked_mul(static_cast<std::size_t>(stage_boundary_columns(plan)) * sizeof(float),
                       values, "tensor-parallel exchange");
}

void resolve_host_cache_budget(ContextCacheOptions& cache, std::uint32_t private_capacity,
                               std::uint32_t shared_capacity, std::uint32_t capacity,
                               std::uint64_t state_image_bytes, std::uint64_t host_kv_group_bytes) {
    const std::uint64_t budget     = *cache.host_cache_budget_bytes;
    const std::uint64_t configured = cache.max_long_anchors_per_continuation.value_or(0);
    // The budget is the ceiling for the whole retention tier, so it decides the anchor count only
    // within the inventory it must already hold: every private owner keeps its endpoint, its
    // rewrite checkpoint and up to A long anchors.
    const auto mandatory_images = [&](std::uint64_t anchors) {
        return (2ULL + anchors) * private_capacity + shared_capacity;
    };
    // Only automatic anchoring creates anchors beyond what clients mark, so only it can use more,
    // and an anchor count set to zero stays disabled.
    const std::uint64_t base_anchors = configured;
    std::uint64_t anchors            = configured;
    const std::uint64_t base_images  = mandatory_images(base_anchors);
    if (cache.automatic_long_anchors && configured != 0 && state_image_bytes != 0 &&
        host_kv_group_bytes != 0 && base_images * state_image_bytes <= budget / 2) {
        // One further anchor costs one StateImage per private owner. It pays for itself only while
        // the gap it covers is worth more Main KV than the image costs, so the number of anchors
        // that can ever be useful is bounded by the Main pages the logical capacity admits: a
        // StateImage buys back `buyback_tokens` tokens of re-prefill (page-aligned, so priced in
        // whole page groups).
        const std::uint64_t buyback_tokens =
            std::max<std::uint64_t>(1, state_image_bytes / host_kv_group_bytes) *
            static_cast<std::uint64_t>(kPagedKVPageSize);
        const std::uint64_t capacity_pages =
            1ULL + (capacity - 1ULL) / static_cast<std::uint64_t>(kPagedKVPageSize);
        const std::uint64_t buyback_pages =
            1ULL + (buyback_tokens - 1ULL) / static_cast<std::uint64_t>(kPagedKVPageSize);
        const std::uint64_t ceiling =
            capacity_pages / buyback_pages > 2ULL ? capacity_pages / buyback_pages - 2ULL : 0ULL;
        const std::uint64_t headroom_images =
            (budget / 2 - base_images * state_image_bytes) / state_image_bytes;
        const std::uint64_t grown = base_anchors + headroom_images / private_capacity;
        anchors = std::min(std::max(grown, base_anchors), std::max(ceiling, base_anchors));
    }
    if (anchors > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.5 Host cache budget anchor count exceeds uint32");
    }
    const std::uint64_t state_slots = mandatory_images(anchors);
    if (state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.5 derived Host state capacity exceeds uint32");
    }
    const std::uint64_t state_bytes = state_slots * state_image_bytes;
    if (state_bytes > budget / 2) {
        throw std::invalid_argument(
            "host cache budget is too small for the configured checkpoint inventory: " +
            std::to_string(state_slots) + " Host state images x " +
            std::to_string(state_image_bytes) + " B = " + std::to_string(state_bytes) +
            " B exceeds half the " + std::to_string(budget) + " B budget (private continuations " +
            std::to_string(private_capacity) + ", shared prefixes " +
            std::to_string(shared_capacity) + ", anchors " + std::to_string(anchors) +
            "); reduce --max-private-continuations / --max-long-anchors-per-continuation or "
            "raise --host-cache-mib");
    }
    cache.max_long_anchors_per_continuation = static_cast<std::uint32_t>(anchors);
    cache.host_state_slots                  = static_cast<std::uint32_t>(state_slots);
    cache.host_kv_capacity_bytes            = static_cast<std::size_t>(budget - state_bytes);
}

std::uint32_t vision_item_token_bound(std::uint32_t capacity, const models::LoadOptions& features) {
    // Zero means "no caller-imposed bound", the same meaning FrontendOptions gives it (its
    // bound_merged_tokens helper returns without clamping). Treating it as one token instead sized
    // the Vision workspace for a single merged token, after which request planning rejected every
    // ordinary image.
    const std::uint32_t requested = features.vision_max_merged_tokens == 0
                                        ? kMaximumVisionItemTokens
                                        : features.vision_max_merged_tokens;
    return static_cast<std::uint32_t>(
        std::min<std::uint64_t>({capacity, kMaximumVisionItemTokens, requested}));
}

// The rotation past the native window, as every family resolves it (rope_scaling.h); this family
// takes YaRN and linear position interpolation alike, for its text and MTP layers.
ops::RopeYarn planned_rope_yarn(const execution::Parameters& parameters,
                                const EngineOptions& options) {
    return runtime::resolve_rope_scaling(options,
                                         parameters.model.config().text.max_position_embeddings,
                                         runtime::kQwen35RopeScaling)
        .yarn;
}

// Every chunk but a prompt's last one has the effective width, so it is chosen near the request
// where the prompt-attention grid of this device leaves the least of a last wave idle.
// NINFER_PREFILL_ALIGN=0 keeps the requested chunk.
std::uint32_t effective_prefill_chunk(const execution::Parameters& parameters,
                                      const EngineOptions& options) {
    const std::uint32_t requested = std::min(options.prefill_chunk, options.max_context);
    static const bool align = [] {
        const char* value = std::getenv("NINFER_PREFILL_ALIGN");
        return value == nullptr || value[0] != '0';
    }();
    if (!align) { return requested; }
    const auto& attention         = *parameters.model.config().text.attention;
    const auto aligned            = static_cast<std::uint32_t>(ops::causal_softmax_attention_prompt_aligned_chunk(
        {static_cast<std::int32_t>(attention.head_dim),
         static_cast<std::int32_t>(attention.num_attention_heads),
         static_cast<std::int32_t>(attention.num_key_value_heads)},
        options.kv_cache, options.fast_prefill_kernel, static_cast<std::int32_t>(requested)));
    return std::min(aligned, options.max_context);
}

std::unique_ptr<qwen3_5::detail::SequencePlannerImpl>
make_sequence_planner_impl(const execution::Parameters& parameters, DeviceContext& device,
                           const EngineOptions& options, std::uint32_t resident_main_pages) {
    validate_target_options(parameters, device, options);
    SequencePlanningInputs inputs{
        .parameters                 = &parameters,
        .capacity                   = options.max_context,
        .max_concurrency            = options.max_concurrency,
        .prefill_chunk              = effective_prefill_chunk(parameters, options),
        .fast_prefill_kernel        = options.fast_prefill_kernel,
        .invariant_attention        = options.invariant_attention,
        .draft_window               = options.speculative.draft_tokens,
        .lookup_ngram               = options.speculative.lookup_ngram,
        .mtp_policy                 = options.speculative.mtp_policy,
        .ngram_draft_window         = options.speculative.ngram_draft_tokens,
        .ngram_min_match            = options.speculative.ngram_min_match,
        .speculative_backend        = options.speculative.backend,
        .kv_storage                 = options.kv_cache,
        .proposal_head = models::planned_proposal_head(options.speculative.proposal_head),
        .rope_yarn                  = planned_rope_yarn(parameters, options),
        .mtp_attention_window       = options.speculative.mtp_attention_window,
        .features                   = models::load_options(options),
        .expert_cache_bytes         = options.expert_cache_bytes,
        .use_cuda_graph             = options.use_cuda_graph,
        .cuda_graph_allowance_bytes = options.cuda_graph_allowance_bytes,
        .causal_scoring             = options.purpose == EnginePurpose::CausalScoring,
        .structured_output          = options.structured_output,
        .gumbel_sampling            = options.gumbel_sampling,
        .device                     = options.device,
        .context_cache              = options.context_cache,
        .resident_main_pages        = resident_main_pages,
        .peer_access                = device.peer_access(),
        .steering_adapters          = static_cast<std::uint32_t>(options.lora_adapters.size()),
        .steering_vectors           = !options.control_vectors.empty(),
    };
    const std::uint32_t logical_pages = page_count(inputs.capacity);
    // Injected graft prefixes hold their Main pages for good, on top of what requests use.
    const std::uint64_t minimum_pages64 =
        static_cast<std::uint64_t>(std::max(logical_pages, inputs.max_concurrency)) +
        resident_main_pages;
    const std::uint64_t maximum_pages64 =
        maximum_main_page_groups(inputs.max_concurrency, logical_pages, inputs.context_cache) +
        resident_main_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    const auto minimum_pages = static_cast<std::uint32_t>(minimum_pages64);
    const auto maximum_pages = static_cast<std::uint32_t>(maximum_pages64);

    auto planner     = std::make_unique<qwen3_5::detail::SequencePlannerImpl>();
    planner->inputs  = inputs;
    planner->minimum = build_sequence_candidate(inputs, minimum_pages);
    planner->curve   = runtime::SequenceCapacityCurve{
          .main_page_tokens                     = static_cast<std::uint32_t>(kPagedKVPageSize),
          .minimum_main_page_groups             = minimum_pages,
          .maximum_main_page_groups             = maximum_pages,
          .minimum_device_reservation_bytes     = planner->minimum->device_reservation_bytes,
          .bytes_per_additional_main_page_group = 0,
          .resident_main_pages                  = resident_main_pages,
    };
    for (const std::size_t bytes : planner->minimum->extra_rank_reservation_bytes) {
        planner->curve.extra_ranks.push_back({.minimum_device_reservation_bytes = bytes});
    }
    // The Device state slots beyond one per lane: the hybrid cache's snapshot slots (at least one;
    // a default count yields to the capacity asked for) or the catalog's checkpoint slots that
    // --device-state-slots sets, beside the ones direct grafts hold for good. A plan with one slot
    // fewer prices each.
    const ContextCacheOptions& cache = inputs.context_cache;
    if (cache.enabled && cache.device_state_slots) {
        const bool hybrid          = cache.mode == ContextCacheMode::Hybrid;
        const std::uint32_t slots  = *cache.device_state_slots;
        const std::uint32_t pinned = hybrid ? 0U : count_direct_grafts(options.grafts);
        const std::uint32_t fewest = hybrid ? 1U : 0U;
        if (slots < pinned + fewest) {
            throw std::logic_error("Qwen3.5 context cache options are not normalized");
        }
        runtime::SequenceCapacityCurve& curve = planner->curve;
        curve.state_slots           = slots - pinned;
        curve.minimum_state_slots   = fewest;
        curve.state_slots_automatic = hybrid && cache.hybrid.fit_device_snapshot_slots;
        curve.state_slot_option     = hybrid ? "--device-snapshot-slots" : "--device-state-slots";
        if (curve.state_slots > fewest) {
            SequencePlanningInputs fewer           = inputs;
            fewer.context_cache.device_state_slots = slots - 1U;
            if (hybrid) { fewer.context_cache.hybrid.device_snapshot_slots = slots - 1U; }
            const auto smaller = build_sequence_candidate(fewer, minimum_pages);
            const auto& larger = *planner->minimum;
            if (smaller->device_reservation_bytes >= larger.device_reservation_bytes) {
                throw std::logic_error("Qwen3.5 sequence layout has a nonpositive state slot stride");
            }
            curve.bytes_per_state_slot =
                larger.device_reservation_bytes - smaller->device_reservation_bytes;
            for (std::size_t rank = 0; rank < curve.extra_ranks.size(); ++rank) {
                if (smaller->extra_rank_reservation_bytes[rank] >
                    larger.extra_rank_reservation_bytes[rank]) {
                    throw std::logic_error(
                        "Qwen3.5 sequence layout has a negative state slot stride on a further "
                        "device");
                }
                curve.extra_ranks[rank].bytes_per_state_slot =
                    larger.extra_rank_reservation_bytes[rank] -
                    smaller->extra_rank_reservation_bytes[rank];
            }
        }
    }
    if (minimum_pages < maximum_pages) {
        auto adjacent = build_sequence_candidate(inputs, minimum_pages + 1U);
        if (adjacent->device_reservation_bytes <= planner->minimum->device_reservation_bytes) {
            throw std::logic_error("Qwen3.5 sequence layout has a nonpositive KV capacity stride");
        }
        planner->curve.bytes_per_additional_main_page_group =
            adjacent->device_reservation_bytes - planner->minimum->device_reservation_bytes;
        // A further device's stride may legitimately be zero: a stage with no attention layers holds
        // no KV.
        for (std::size_t rank = 0; rank < planner->curve.extra_ranks.size(); ++rank) {
            planner->curve.extra_ranks[rank].bytes_per_additional_main_page_group =
                adjacent->extra_rank_reservation_bytes[rank] -
                planner->minimum->extra_rank_reservation_bytes[rank];
        }
    }
    return planner;
}

std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_5::detail::SequencePlannerImpl> planner,
                            std::uint32_t main_page_groups) {
    if (planner == nullptr || planner->minimum == nullptr) {
        throw std::invalid_argument("Qwen3.5 sequence planner is empty");
    }
    const std::size_t expected = planner->curve.reservation_bytes(main_page_groups);
    std::unique_ptr<SequencePlanImpl> plan;
    if (main_page_groups == planner->curve.minimum_main_page_groups) {
        plan = std::move(planner->minimum);
    } else {
        plan = build_sequence_candidate(planner->inputs, main_page_groups);
    }
    if (plan->device_reservation_bytes != expected) {
        throw std::logic_error(
            "Qwen3.5 physical sequence layout is not affine in Main KV page capacity");
    }
    for (std::size_t rank = 0; rank < plan->extra_rank_reservation_bytes.size(); ++rank) {
        if (plan->extra_rank_reservation_bytes[rank] !=
            planner->curve.extra_rank_reservation_bytes(rank, main_page_groups)) {
            throw std::logic_error(
                "Qwen3.5 physical sequence layout is not affine in Main KV page capacity on a "
                "further device");
        }
    }
    return plan;
}

} // namespace ninfer::models::qwen3_5::detail
