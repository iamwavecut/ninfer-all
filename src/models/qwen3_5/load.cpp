#include "models/qwen3_5/load.h"

#include "artifact/reader.h"
#include "core/dtype.h"
#include "core/evictable_weight_pool.h"
#include "core/paged_kv_view.h"
#include "core/paged_kv_storage.h"
#include "core/stage_plan.h"
#include "core/tensor_split.h"
#include "models/qwen3_5/load/bindings.h"
#include "models/qwen3_5/load/vision_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5 {

struct LoadPlan::Impl {
    Config config;
    LoadOptions options;
    std::filesystem::path artifact_path;
    ModelWeights weights;
    std::vector<loading::PendingWeight> pending;
    artifact::MaterializationPlan materialization;
    FrontendResources resources;
    InstanceInfo info;
    std::shared_ptr<const CpuVisionWeights> cpu_vision;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

const FrontendResources& LoadPlan::resources() const { return impl_->resources; }

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    return impl_->materialization;
}

const artifact::ParameterReference& LoadPlan::parameter(WeightId id) const {
    return impl_->pending.at(id.index).reference;
}

std::span<const WeightUse> LoadPlan::uses(WeightId id) const {
    return impl_->pending.at(id.index).uses;
}

std::vector<ops::DeviceTier> rank_tiers(const DeviceContext& device) {
    std::vector<ops::DeviceTier> tiers;
    tiers.reserve(device.size());
    for (std::size_t rank = 0; rank < device.size(); ++rank) {
        tiers.push_back(ops::device_tier(device.rank(rank).device));
    }
    return tiers;
}

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options,
                   std::filesystem::path artifact_path, std::vector<ops::DeviceTier> rank_tiers) {
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->options = options;
    out->artifact_path = std::move(artifact_path);
    out->config  = parse_config(reader.directory(), options);
    artifact::Binder binder(reader);
    out->resources = loading::bind_resources(binder, out->config);
    loading::Bindings bindings(binder, options);
    const auto& text = out->config.text;
    if (options.expert_residency == ExpertResidency::Disk && out->artifact_path.empty()) {
        throw std::invalid_argument("disk expert residency requires the artifact entry path");
    }
    if (options.expert_residency != ExpertResidency::Device &&
        !std::holds_alternative<MoeConfig>(text.ffn)) {
        throw std::invalid_argument("--expert-residency host/disk requires a MoE model");
    }
    if (options.ranks > 1 && options.overlay_vision()) {
        // Overlay borrows weight memory from the primary device's evictable tail; a later stage
        // holds whole layers and nothing a Vision window could take. The two residency schemes are
        // answers to the same question -- where the bytes for something else come from -- and no
        // sound combination of them exists today, so say so rather than half-apply one.
        throw std::invalid_argument(
            "--vision-residency overlay and a multi-device --devices split cannot be combined");
    }
    out->weights.text =
        loading::bind_text(bindings, text, options,
                           loading::plan_stage_plan(text.num_hidden_layers, options));
    if (options.overlay_vision() && !out->config.vision) {
        throw std::invalid_argument("--vision-residency overlay requires a Vision artifact");
    }
    if (options.cpu_vision() && !out->config.vision) {
        throw std::invalid_argument("--vision-residency cpu requires a Vision artifact");
    }
    if (options.cpu_vision()) {
        out->cpu_vision = loading::load_cpu_vision(binder, *out->config.vision, text);
    } else if (out->config.vision) {
        out->weights.vision = loading::bind_vision(
            bindings, *out->config.vision, text.hidden_size,
            options.overlay_vision() ? artifact::Residency::Pinned : artifact::Residency::Device);
    }
    std::pair<std::size_t, std::size_t> mtp_parameters{bindings.weights.size(),
                                                       bindings.weights.size()};
    if (out->config.mtp) {
        out->weights.mtp      = loading::bind_mtp(bindings, text, out->weights.text);
        mtp_parameters.second = bindings.weights.size();
    }
    std::pair<std::size_t, std::size_t> draft_parameters{bindings.weights.size(),
                                                         bindings.weights.size()};
    // A DSpine drafter is its own artifact, which materialize_model loads beside this one.
    if (out->config.draft && !out->config.draft->dspine) {
        out->weights.draft =
            loading::bind_draft(bindings, *out->config.draft, text, out->weights.text,
                                std::string(options.speculative_component()));
        draft_parameters.second = bindings.weights.size();
    }
    if (options.proposal_enabled()) {
        const auto& proposal = reader.directory().component("text").proposal;
        if (!proposal) {
            throw artifact::ArtifactError("selected proposal head is absent from artifact");
        }
        out->weights.proposal = loading::bind_proposal(bindings, *proposal, text, options,
                                                       out->resources.public_token_count);
        if (out->config.draft && out->config.draft->dflash2) {
            const auto domain =
                proposal->indexed ? proposal->rows : out->resources.public_token_count;
            if (out->config.draft->dflash2->selector_top_k > domain) {
                throw artifact::ArtifactError(
                    "proposal domain is smaller than DFlash2 selector_top_k");
            }
        }
        if (out->weights.mtp) { out->weights.mtp->output_head = out->weights.proposal->head; }
        if (out->weights.draft) { out->weights.draft->output_head = out->weights.proposal->head; }
    }
    loading::apply_storage_trades(bindings, out->config, out->weights, options);
    if (rank_tiers.empty()) { rank_tiers.assign(options.ranks, ops::current_device_tier()); }
    loading::apply_rank_policies(bindings, rank_tiers);
    loading::apply_device_layouts(bindings, out->weights, rank_tiers);
    if (options.overlay_vision()) {
        loading::apply_vision_overlay_placement(bindings, out->weights, mtp_parameters,
                                                draft_parameters);
    }
    out->weights.text.output_head_use =
        bindings.use(out->weights.text.output_head, "text/final_hidden");
    auto& head_shares = out->weights.text.output_head_shares;
    for (const WeightId share : head_shares.shares) {
        head_shares.uses.push_back(bindings.use(share, "text/final_hidden"));
    }
    if (out->weights.mtp) {
        out->weights.mtp->output_head_use =
            bindings.use(out->weights.mtp->output_head, "mtp/final_hidden");
        // Drafts without a proposal head project the output head's shares through MTP's input.
        if (!out->weights.proposal && !head_shares.shares.empty()) {
            auto& drafts = out->weights.mtp->output_head_shares;
            drafts.shares = head_shares.shares;
            drafts.rows   = head_shares.rows;
            for (const WeightId share : drafts.shares) {
                drafts.uses.push_back(bindings.use(share, "mtp/final_hidden"));
            }
        }
    }
    if (out->weights.draft) {
        out->weights.draft->output_head_use =
            bindings.use(out->weights.draft->output_head,
                         std::string(options.speculative_component()) + "/final_hidden");
    }
    out->pending         = std::move(bindings.weights);
    out->materialization = std::move(binder).finish(
        options.overlay_vision() ? EvictableWeightPool::kChunkBytes : 1);
    out->info.name       = reader.directory().metadata.value(
        "name", std::string(architecture_name(text.architecture)));
    out->info.metadata_json   = reader.directory().metadata.dump();
    out->info.provenance_json = reader.directory().provenance.dump();
    out->info.artifact_id     = reader.artifact_id();
    return LoadPlan(std::move(out));
}

namespace {

// What one stage's device needs whatever layers it owns, beyond weights: the CUDA context, the
// workspace and graph memory of a forward pass over a full chunk. An estimate, since the Program
// plans these later; it only decides how the layers are dealt out, and `--stage-layers` overrides it.
constexpr std::uint64_t kStageFixedBytes = 1536ULL << 20;
// What rank 0 needs on top: the round buffers, logits and sampling state that stay with the head.
constexpr std::uint64_t kHeadStageFixedBytes = 1024ULL << 20;

// A parameter's share of its stored objects: a part that is a range of a grouped parent (the [gate;
// up] rows of an MLP, a GDN block's projections) counts only its elements' share of the parent.
std::uint64_t parameter_bytes(const artifact::Reader& reader, const loading::Bindings& bindings,
                              WeightId id) {
    if (bindings.at(id).reference.residency != artifact::Residency::Device) { return 0; }
    std::uint64_t total = 0;
    for (const auto& part : bindings.at(id).reference.binding.parts) {
        const auto& geometry = reader.geometry(part.object);
        total += geometry.elements == 0 || part.end <= part.begin
                     ? geometry.bytes
                     : static_cast<std::uint64_t>(static_cast<long double>(geometry.bytes) *
                                                  (part.end - part.begin) / geometry.elements);
    }
    return total;
}

// The device bytes of what the head stage (rank 0 under tensor parallelism) holds beside its layers
// or its share: the embedding, the head and the final norm, and as selected a resident Vision
// tower, MTP's layer, a DFlash drafter's own weights and the proposal head the drafts project onto.
// Binds them into `bindings` only to size them. `mtp_ffn` and `mtp_attention` take the parts of
// MTP's layer that are a dense FFN and the attention's query, gate and output projections, which
// tensor parallelism divides; it divides the K and V rows too, which rank 0 keeps whole besides.
std::uint64_t head_stage_bytes(const artifact::Reader& reader, loading::Bindings& bindings,
                               const Config& config, const TextWeights& weights,
                               const LoadOptions& options, std::uint64_t* proposal_head = nullptr,
                               std::uint64_t* mtp_ffn = nullptr,
                               std::uint64_t* mtp_attention = nullptr) {
    const auto& text       = config.text;
    std::uint64_t out      = parameter_bytes(reader, bindings, weights.token_embedding) +
                             parameter_bytes(reader, bindings, weights.output_head) +
                             parameter_bytes(reader, bindings, weights.final_norm);
    const auto bound_since = [&](std::size_t first) {
        std::uint64_t bytes = 0;
        for (std::size_t index = first; index < bindings.weights.size(); ++index) {
            bytes += parameter_bytes(reader, bindings, WeightId{index});
        }
        return bytes;
    };
    if (config.vision && options.vision && !options.host_staged_vision()) {
        // A resident Vision tower runs where the embedding is.
        const std::size_t first = bindings.weights.size();
        (void)loading::bind_vision(bindings, *config.vision, text.hidden_size,
                                   artifact::Residency::Device);
        out += bound_since(first);
    }
    if (config.mtp) {
        // MTP's own parameters; it shares the target's embedding and head.
        const std::size_t first = bindings.weights.size();
        const MtpWeights mtp    = loading::bind_mtp(bindings, text, weights);
        out += bound_since(first);
        if (const auto* dense = std::get_if<DenseWeights>(&mtp.layer.ffn); dense && mtp_ffn) {
            *mtp_ffn = parameter_bytes(reader, bindings, dense->gate) +
                       parameter_bytes(reader, bindings, dense->up) +
                       parameter_bytes(reader, bindings, dense->down);
        }
        if (mtp_attention != nullptr) {
            const auto& attention = std::get<AttentionWeights>(mtp.layer.mixer);
            *mtp_attention = 0;
            for (const WeightId id : {attention.query, attention.gate, attention.output}) {
                *mtp_attention += parameter_bytes(reader, bindings, id);
            }
        }
    }
    if (config.draft && config.draft->dspine) {
        // A DSpine drafter's own artifact and the table it derives from the head.
        out += dspine::weight_bytes(options.draft_model) +
               dspine::head_write_bytes(*config.draft->dspine);
    } else if (config.draft && options.masked_draft()) {
        // The drafter shares the target's embedding and head, so only what it binds anew counts.
        const std::size_t first = bindings.weights.size();
        (void)loading::bind_draft(bindings, *config.draft, text, weights,
                                  std::string(options.speculative_component()));
        out += bound_since(first);
    }
    const auto& proposal = reader.directory().component("text").proposal;
    if (options.proposal_enabled() && proposal) {
        // Sized only: plan_load checks the head's rows against the tokenizer's public tokens.
        const std::size_t first = bindings.weights.size();
        const ProposalWeights bound = loading::bind_proposal(
            bindings, *proposal, text, options, std::numeric_limits<std::uint32_t>::max());
        if (proposal_head != nullptr) {
            *proposal_head = parameter_bytes(reader, bindings, bound.head);
        }
        out += bound_since(first);
    }
    return out;
}

// KV bytes per token of MTP's layer, which keeps every KV head on the head stage; 0 without MTP.
std::uint64_t mtp_kv_bytes_per_token(const Config& config, KvCacheStorage kv_storage) {
    if (!config.mtp || !config.text.attention) { return 0; }
    const auto& attention = *config.text.attention;
    const PagedKVStorageLayout storage =
        paged_kv_storage_layout(kv_storage, static_cast<std::int32_t>(attention.head_dim));
    return std::uint64_t(attention.num_key_value_heads) * storage.physical_bytes_per_token_head();
}

} // namespace

std::vector<std::uint32_t> default_stage_layers(const artifact::Reader& reader,
                                                LoadOptions options, const StageSizing& sizing,
                                                std::span<const std::uint64_t> free_bytes) {
    if (free_bytes.size() < 2) {
        throw std::invalid_argument("default stage layers need at least two devices");
    }
    // Bind the text model on one stage purely to size its layers.
    options.ranks = 1;
    options.stage_layers.clear();
    const Config config = parse_config(reader.directory(), options);
    const auto& text    = config.text;
    artifact::Binder binder(reader);
    loading::Bindings bindings(binder, options);
    const TextWeights weights =
        loading::bind_text(bindings, text, options, StagePlan(text.num_hidden_layers));

    std::vector<LayerCost> layers;
    layers.reserve(weights.layers.size());
    for (std::size_t layer = 0; layer < weights.layers.size(); ++layer) {
        LayerCost cost;
        for (const WeightId id : loading::layer_weights(weights.layers[layer])) {
            cost.weight_bytes += parameter_bytes(reader, bindings, id);
        }
        if (text.layer_types[layer] == MixerKind::FullAttention && text.attention) {
            // One page group of this layer: every plane of a page, for every KV head, as stored.
            const PagedKVStorageLayout storage = paged_kv_storage_layout(
                sizing.kv_storage, static_cast<std::int32_t>(text.attention->head_dim));
            cost.kv_bytes_per_page_group = static_cast<std::uint64_t>(kPagedKVPageSize) *
                                           text.attention->num_key_value_heads *
                                           storage.physical_bytes_per_token_head();
        } else if (text.gdn) {
            // The convolution window and the recurrent matrix of every state slot.
            const auto& gdn = *text.gdn;
            const std::uint64_t conv_bytes =
                gdn.conv_channels() * (gdn.linear_conv_kernel_dim - 1) * dtype_size(DType::BF16);
            const std::uint64_t recurrent_bytes =
                static_cast<std::uint64_t>(gdn.linear_key_head_dim) * gdn.linear_value_head_dim *
                gdn.linear_num_value_heads *
                dtype_size(options.gdn_state_fp16 ? DType::FP16 : DType::FP32);
            cost.state_bytes = sizing.state_slots * (conv_bytes + recurrent_bytes);
        }
        layers.push_back(cost);
    }

    // The head stage keeps the embedding, the head, the final norm and what drafts and Vision add,
    // and MTP's KV for every page group. Automatic KV capacity keeps headroom free on every device.
    const std::uint64_t head_bytes = head_stage_bytes(reader, bindings, config, weights, options);
    const std::uint64_t mtp_kv     = static_cast<std::uint64_t>(kPagedKVPageSize) *
                                     mtp_kv_bytes_per_token(config, sizing.kv_storage);
    const std::uint64_t headroom   = sizing.automatic_kv ? sizing.kv_headroom_bytes : 0;

    std::vector<StageBudget> budgets;
    budgets.reserve(free_bytes.size());
    for (std::size_t stage = 0; stage < free_bytes.size(); ++stage) {
        budgets.push_back(
            {.available_bytes = free_bytes[stage] > headroom ? free_bytes[stage] - headroom : 0,
             .fixed_bytes = kStageFixedBytes + (stage == 0 ? kHeadStageFixedBytes + head_bytes : 0),
             .kv_bytes_per_page_group = stage == 0 ? mtp_kv : 0});
    }
    // The estimates above are rough. When they say nothing fits, deal the layers out evenly and let
    // the Program's exact planning accept the split or report which device runs out.
    const StagePlan plan = [&] {
        try {
            return solve_stage_plan(layers, budgets).plan;
        } catch (const std::runtime_error&) {
            return StagePlan::even(text.num_hidden_layers, free_bytes.size());
        }
    }();
    std::vector<std::uint32_t> counts;
    counts.reserve(plan.stages());
    for (std::size_t stage = 0; stage < plan.stages(); ++stage) {
        counts.push_back(plan.stage_layers(stage));
    }
    return counts;
}

namespace {

// What a tensor-parallel split divides and what it leaves whole, in device bytes.
struct SplitSizes {
    std::uint64_t divided = 0; // GDN and KV heads and FFN rows or experts, with their state and KV
    std::uint64_t every   = 0; // what each rank holds whole: norms, GDN gating, the router
    std::uint64_t first   = 0; // rank 0 alone: embedding, head, shared expert, MTP's KV
};

std::uint64_t gdn_state_bytes(const TextConfig& text, const LoadOptions& options,
                              std::uint32_t state_slots) {
    if (!text.gdn) { return 0; }
    const auto& gdn = *text.gdn;
    const std::uint64_t conv_bytes =
        gdn.conv_channels() * (gdn.linear_conv_kernel_dim - 1) * dtype_size(DType::BF16);
    const std::uint64_t recurrent_bytes =
        static_cast<std::uint64_t>(gdn.linear_key_head_dim) * gdn.linear_value_head_dim *
        gdn.linear_num_value_heads * dtype_size(options.gdn_state_fp16 ? DType::FP16 : DType::FP32);
    return std::uint64_t(text.linear_attention_layers) * state_slots *
           (conv_bytes + recurrent_bytes);
}

std::uint64_t attention_kv_bytes(const TextConfig& text, const StageSizing& sizing,
                                 std::uint64_t kv_heads) {
    if (!text.attention) { return 0; }
    const PagedKVStorageLayout storage = paged_kv_storage_layout(
        sizing.kv_storage, static_cast<std::int32_t>(text.attention->head_dim));
    return sizing.kv_tokens * text.full_attention_layers * kv_heads *
           storage.physical_bytes_per_token_head();
}

SplitSizes split_sizes(const artifact::Reader& reader, LoadOptions options,
                       const StageSizing& sizing) {
    // Bind the text model whole on one device purely to size its parts.
    options.ranks = 1;
    options.tensor_parallel = false;
    options.tensor_split.clear();
    options.stage_layers.clear();
    const Config config = parse_config(reader.directory(), options);
    const auto& text    = config.text;
    artifact::Binder binder(reader);
    loading::Bindings bindings(binder, options);
    const TextWeights weights =
        loading::bind_text(bindings, text, options, StagePlan(text.num_hidden_layers));
    const auto bytes = [&](WeightId id) { return parameter_bytes(reader, bindings, id); };

    SplitSizes out;
    for (std::size_t layer = 0; layer < weights.layers.size(); ++layer) {
        const BlockWeights& block = weights.layers[layer];
        out.every += bytes(block.input_norm) + bytes(block.post_attention_norm);
        if (const auto* gdn = std::get_if<GdnWeights>(&block.mixer)) {
            out.divided += bytes(gdn->query) + bytes(gdn->key) + bytes(gdn->value) +
                           bytes(gdn->z) + bytes(gdn->convolution) + bytes(gdn->output);
            out.every += bytes(gdn->a_projection) + bytes(gdn->b_projection) +
                         bytes(gdn->a_log) + bytes(gdn->dt_bias) + bytes(gdn->norm);
        } else {
            const auto& attention = std::get<AttentionWeights>(block.mixer);
            out.divided += bytes(attention.query) + bytes(attention.key) +
                           bytes(attention.gate) + bytes(attention.value) +
                           bytes(attention.output);
            out.every += bytes(attention.query_norm) + bytes(attention.key_norm);
        }
        if (const auto* dense = std::get_if<DenseWeights>(&block.ffn)) {
            out.divided += bytes(dense->gate) + bytes(dense->up) + bytes(dense->down);
            continue;
        }
        // Expert parallelism: every rank routes, rank 0 holds the shared expert.
        const auto& moe   = std::get<MoeWeights>(block.ffn);
        const auto expert = [&](const DenseWeights& e) {
            return bytes(e.gate) + bytes(e.up) + bytes(e.down);
        };
        std::uint64_t experts = 0;
        for (const DenseWeights& e : moe.experts) { experts += expert(e); }
        out.every += bytes(moe.router) + bytes(moe.shared_score);
        out.first += expert(moe.shared);
        out.divided += experts;
    }
    out.divided += gdn_state_bytes(text, options, sizing.state_slots);
    if (text.attention) {
        out.divided += attention_kv_bytes(text, sizing, text.attention->num_key_value_heads);
    }
    // Rank 0 also holds the head stage's parts.
    std::uint64_t proposal_head = 0;
    std::uint64_t mtp_ffn       = 0;
    std::uint64_t mtp_attention = 0;
    out.first += head_stage_bytes(reader, bindings, config, weights, options, &proposal_head,
                                  &mtp_ffn, &mtp_attention);
    // Every rank holds its rows of the heads besides, which decode projects divided, and of MTP's
    // dense FFN instead of rank 0 the whole. Its attention and KV divide as a text layer's do;
    // position-split attention keeps them whole on rank 0. (The divided K and V rows rank 0 keeps
    // whole besides are small and not counted.)
    if (!options.lm_head_q4 && !options.lm_head_q6) { out.divided += bytes(weights.output_head); }
    out.divided += proposal_head;
    out.first -= mtp_ffn;
    out.divided += mtp_ffn;
    const std::uint64_t mtp_kv = sizing.kv_tokens * mtp_kv_bytes_per_token(config, sizing.kv_storage);
    if (options.tensor_attention == TensorAttention::Positions) {
        out.first += mtp_kv;
    } else {
        out.first -= mtp_attention;
        out.divided += mtp_attention + mtp_kv;
    }
    return out;
}

// What `options.tensor_split` places on each rank of what the split divides: the device weights
// plan_load gives the rank, and the KV cache and GDN state of its heads. Only differences between
// two splits mean anything; what every rank or rank 0 holds whole is the same in both.
struct RankBytes {
    std::vector<std::uint64_t> bytes;
    // The part of `bytes` that is the KV cache of the rank's attention heads.
    std::vector<std::uint64_t> kv;
    std::vector<TensorShare> shares;
};

RankBytes rank_bytes(const artifact::Reader& reader, const LoadOptions& options,
                     const StageSizing& sizing, const std::filesystem::path& artifact_path,
                     const std::vector<ops::DeviceTier>& rank_tiers) {
    const LoadPlan plan = plan_load(reader, options, artifact_path, rank_tiers);
    const auto& text    = plan.config().text;
    RankBytes out;
    out.shares = plan.weights().text.tensor_shares;
    // MTP's KV divides with its attention, by the same heads.
    const bool mtp_divided = plan.weights().mtp && !plan.weights().mtp->rank_attention.empty();
    const std::uint64_t mtp_kv_head =
        mtp_divided ? sizing.kv_tokens * mtp_kv_bytes_per_token(plan.config(), sizing.kv_storage) /
                          text.attention->num_key_value_heads
                    : 0;
    for (std::size_t rank = 0; rank < out.shares.size(); ++rank) {
        const TensorShare& share = out.shares[rank];
        const TextConfig part    = tensor_parallel_share(text, share);
        out.kv.push_back(attention_kv_bytes(text, sizing, share.kv_head_end - share.kv_head_begin) +
                         mtp_kv_head * (share.kv_head_end - share.kv_head_begin));
        out.bytes.push_back(plan.materialization().device_capacity(rank) +
                            gdn_state_bytes(part, options, sizing.state_slots) + out.kv.back());
    }
    return out;
}

} // namespace

std::vector<double> default_tensor_split(const artifact::Reader& reader, LoadOptions options,
                                         const StageSizing& sizing,
                                         std::span<const std::uint64_t> free_bytes,
                                         std::span<const double> speed) {
    if (free_bytes.size() < 2 || speed.size() != free_bytes.size()) {
        throw std::invalid_argument("a tensor split needs the free bytes and speed of each rank");
    }
    const SplitSizes sizes = split_sizes(reader, std::move(options), sizing);
    if (sizes.divided == 0) { return std::vector<double>(speed.begin(), speed.end()); }

    // Automatic KV capacity keeps headroom free on every device, and its KV shrinks to what fits.
    const std::uint64_t headroom = sizing.automatic_kv ? sizing.kv_headroom_bytes : 0;
    std::vector<double> capacity(free_bytes.size());
    for (std::size_t rank = 0; rank < free_bytes.size(); ++rank) {
        const std::uint64_t whole = headroom + sizes.every + (rank == 0 ? sizes.first : 0);
        capacity[rank] = free_bytes[rank] > whole
                             ? static_cast<double>(free_bytes[rank] - whole) /
                                   static_cast<double>(sizes.divided)
                             : 0.0;
    }
    return fill_shares(speed, capacity, sizing.automatic_kv);
}

std::optional<std::vector<double>>
refit_tensor_split(const artifact::Reader& reader, const LoadOptions& options,
                   const StageSizing& sizing, std::span<const std::int64_t> spare,
                   std::span<const double> speed, const std::filesystem::path& artifact_path,
                   const std::vector<ops::DeviceTier>& rank_tiers) {
    const std::size_t ranks = options.tensor_split.size();
    if (ranks < 2 || spare.size() != ranks || speed.size() != ranks) {
        throw std::invalid_argument(
            "a tensor split refit needs the split, spare and speed of each rank");
    }
    const double divided = static_cast<double>(split_sizes(reader, options, sizing).divided);
    if (divided == 0.0) { return std::nullopt; }
    double total = 0.0;
    for (const double weight : options.tensor_split) { total += weight; }
    // The share each rank may take: its own and the spare it kept.
    std::vector<double> room(ranks);
    double rooms = 0.0;
    for (std::size_t rank = 0; rank < ranks; ++rank) {
        room[rank] = std::max(0.0, options.tensor_split[rank] / total +
                                       static_cast<double>(spare[rank]) / divided);
        rooms += room[rank];
    }
    LoadOptions candidate  = options;
    candidate.tensor_split = fill_shares(speed, room, sizing.automatic_kv);
    const bool same = std::ranges::equal(
        candidate.tensor_split, options.tensor_split,
        [&](double a, double b) { return std::abs(a - b / total) <= 1e-9; });
    if (same) { return std::nullopt; }
    // The rooms add up to one exactly when the spares do to zero: only then can a split hold the KV
    // asked for, and each candidate must then fit every rank.
    const bool fits = rooms >= 1.0;
    const auto place = [&](const LoadOptions& at) {
        return rank_bytes(reader, at, sizing, artifact_path, rank_tiers);
    };
    const RankBytes loaded = place(options);
    RankBytes placed;
    // What each rank lacks beside the candidate's shares (zero where it fits).
    const auto lacking = [&](const RankBytes& at) {
        std::vector<double> out(ranks);
        for (std::size_t rank = 0; rank < ranks; ++rank) {
            out[rank] = std::max(0.0, (static_cast<double>(at.bytes[rank]) -
                                       static_cast<double>(loaded.bytes[rank])) -
                                          static_cast<double>(spare[rank]));
        }
        return out;
    };
    const auto fit = [&](const std::vector<double>& lack) {
        return std::ranges::all_of(lack, [](double bytes) { return bytes == 0.0; });
    };
    bool fitted = false;
    if (fits) {
        // A share ends at the nearest end the weights allow, so a candidate can put up to half a
        // unit (a GDN key head of every layer, an FFN tile) back on a rank, or take one more off.
        // Shrink the rooms of the ranks that would lack bytes until a candidate fits, then bisect
        // between the last rooms that did not fit (at first the loaded shares, when a rank fell
        // short there) and the ones that did, so no rank gives up a unit it has room for.
        placed                         = place(candidate);
        std::vector<double> short_room = room;
        if (std::ranges::any_of(spare, [](std::int64_t bytes) { return bytes < 0; })) {
            for (std::size_t rank = 0; rank < ranks; ++rank) {
                short_room[rank] = options.tensor_split[rank] / total;
            }
        }
        std::vector<double> lack = lacking(placed);
        for (int round = 0; round < 16 && !fit(lack); ++round) {
            short_room = room;
            for (std::size_t rank = 0; rank < ranks; ++rank) {
                if (lack[rank] > 0.0) {
                    room[rank] = std::max(0.0, room[rank] - (lack[rank] / divided + 1.0 / 4096.0));
                }
            }
            candidate.tensor_split = fill_shares(speed, room, sizing.automatic_kv);
            placed                 = place(candidate);
            lack                   = lacking(placed);
        }
        if (fit(lack) && short_room != room) {
            std::vector<double> fitting = room;
            for (int step = 0; step < 8; ++step) {
                std::vector<double> middle(ranks);
                for (std::size_t rank = 0; rank < ranks; ++rank) {
                    middle[rank] = (short_room[rank] + fitting[rank]) / 2.0;
                }
                LoadOptions probe  = options;
                probe.tensor_split = fill_shares(speed, middle, sizing.automatic_kv);
                RankBytes at       = place(probe);
                if (fit(lacking(at))) {
                    fitting   = std::move(middle);
                    candidate = std::move(probe);
                    placed    = std::move(at);
                } else {
                    short_room = std::move(middle);
                }
            }
        }
        fitted = fit(lack);
    }
    if (!fitted) {
        // An explicit capacity that no split holds fails at startup whatever the shares.
        if (!sizing.automatic_kv) { return std::nullopt; }
        // Automatic capacity shrinks the KV until the rank short of the largest part of its own KV
        // fits, so the shortfall is best shared in proportion to each rank's KV. A share moves
        // whole units, and a step past where the heads end moves a whole head of every layer, so
        // the search steps from the best split measured so far toward the one that would share
        // the shortfall out: it doubles a step that moved no unit and halves one that left the
        // binding rank short of no less.
        const auto short_of = [&](const RankBytes& at, std::size_t rank) {
            return (static_cast<double>(at.bytes[rank]) -
                    static_cast<double>(loaded.bytes[rank])) -
                   static_cast<double>(spare[rank]);
        };
        const auto binding = [&](const RankBytes& at) {
            double out = -std::numeric_limits<double>::infinity();
            for (std::size_t rank = 0; rank < ranks; ++rank) {
                out = std::max(out, short_of(at, rank) /
                                        std::max(1.0, static_cast<double>(at.kv[rank])));
            }
            return out;
        };
        const auto normalized = [](std::vector<double> split) {
            double sum = 0.0;
            for (double& share : split) {
                share = std::max(0.0, share);
                sum += share;
            }
            for (double& share : split) { share /= sum; }
            return split;
        };
        const auto toward = [&](const std::vector<double>& split, const RankBytes& at) {
            double shortfall = 0.0;
            double kv        = 0.0;
            for (std::size_t rank = 0; rank < ranks; ++rank) {
                shortfall += short_of(at, rank);
                kv += static_cast<double>(at.kv[rank]);
            }
            const double part = shortfall / std::max(1.0, kv);
            std::vector<double> out(ranks);
            for (std::size_t rank = 0; rank < ranks; ++rank) {
                out[rank] = split[rank] - (short_of(at, rank) -
                                           part * static_cast<double>(at.kv[rank])) /
                                              divided;
            }
            return normalized(std::move(out));
        };
        std::vector<double> best(ranks);
        for (std::size_t rank = 0; rank < ranks; ++rank) {
            best[rank] = options.tensor_split[rank] / total;
        }
        placed                     = loaded;
        const double before        = binding(loaded);
        double best_shortfall      = before;
        std::vector<double> target = toward(best, loaded);
        double step                = 1.0;
        for (int round = 0; round < 8; ++round) {
            std::vector<double> trial(ranks);
            for (std::size_t rank = 0; rank < ranks; ++rank) {
                trial[rank] = best[rank] + step * (target[rank] - best[rank]);
            }
            LoadOptions probe  = options;
            probe.tensor_split = normalized(std::move(trial));
            RankBytes at       = place(probe);
            if (const double now = binding(at); now < best_shortfall) {
                best_shortfall = now;
                best           = probe.tensor_split;
                placed         = std::move(at);
                target         = toward(best, placed);
                step           = 1.0;
            } else if (at.shares == placed.shares) {
                step *= 2.0;
            } else {
                step /= 2.0;
            }
        }
        // The binding rank's part times the tokens asked for is the KV it lacks: a gain short of
        // one page leaves the capacity where it is, and a split that holds less than the fewest
        // tokens capacity starts with does not start.
        const auto tokens = static_cast<double>(sizing.kv_tokens);
        if ((before - best_shortfall) * tokens < static_cast<double>(kPagedKVPageSize) ||
            (1.0 - best_shortfall) * tokens < static_cast<double>(sizing.minimum_kv_tokens)) {
            return std::nullopt;
        }
        candidate.tensor_split = std::move(best);
    }
    if (placed.shares == loaded.shares) { return std::nullopt; }
    return candidate.tensor_split;
}

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer,
                                         const artifact::MaterializeOptions& materialize) {
    if (!plan.impl_) { throw artifact::ArtifactError("load plan was already consumed"); }
    auto data = std::move(plan.impl_);
    std::unique_ptr<EvictableWeightPool> pool;
    if (data->options.overlay_vision()) {
        // The tail is sized against the encode window once execution planning knows it; the
        // pool's mirror is captured then (see Model::weight_pool).
        const auto& materialization = data->materialization;
        if (materialization.evictable_tail_bytes == 0 || materialization.pinned_objects.empty()) {
            throw std::logic_error("overlay Vision load plan has no evictable tail or pinned tower");
        }
        if (!EvictableWeightPool::supported(device)) {
            throw std::invalid_argument(
                "--vision-residency overlay requires CUDA virtual memory management support");
        }
        pool = std::make_unique<EvictableWeightPool>(
            device, EvictableWeightPool::Config{
                        .arena_bytes =
                            static_cast<std::size_t>(materialization.device_capacity(0)),
                        .evictable_tail_bytes =
                            static_cast<std::size_t>(materialization.evictable_tail_bytes),
                    });
    }
    const auto& reader = *data->materialization.source;
    auto backing = artifact::materialize(reader,
                                         std::move(data->materialization), device, observer,
                                         std::move(pool), materialize);
    auto experts = loading::expert_sources(reader, data->pending, data->weights, backing,
                                            data->options, data->artifact_path);
    auto bound   = loading::resolve_weights(std::move(data->pending), backing);
    std::optional<VisionOverlayLayout> vision_overlay;
    if (data->options.overlay_vision()) {
        vision_overlay =
            loading::vision_overlay_layout(*data->weights.vision, bound, backing.pinned_block());
    }
    AuxiliaryReplicas replicas(bound, device);
    const bool dspine     = data->options.dspine();
    const auto draft_path = data->options.draft_model;
    auto model            = std::unique_ptr<Model>(
        new Model(std::move(data->config), data->options, std::move(data->weights),
                  std::move(bound), std::move(replicas), std::move(data->resources),
                  std::move(data->info), std::move(backing), std::move(vision_overlay),
                  std::move(data->cpu_vision), std::move(experts)));
    if (dspine) {
        // On the head's device, where the drafts' embedding, head and round state are.
        model->dspine_ = dspine::load_model(
            draft_path, device, observer,
            artifact::MaterializeOptions{.suspendable = materialize.suspendable});
    }
    return model;
}

std::unique_ptr<Model> load_model(const std::filesystem::path& path, LoadOptions options,
                                  DeviceContext& device, const StartupObserver* observer) {
    artifact::Reader reader(path);
    return materialize_model(plan_load(reader, options, path, rank_tiers(device)), device,
                             observer);
}

} // namespace ninfer::models::qwen3_5
