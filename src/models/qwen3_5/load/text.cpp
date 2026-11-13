#include "models/qwen3_5/load/bindings.h"
#include "core/qtype.h"

#include "core/tensor_split.h"
#include "ninfer/ops/softmax_attention_envelope.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>

namespace ninfer::models::qwen3_5::loading {

AttentionWeights bind_attention(Bindings& b, const TextConfig& config, const std::string& p) {
    const auto& a = config.attention.value();
    const auto h  = config.hidden_size;
    const auto q  = a.query_width();
    const auto k  = a.key_width();
    AttentionWeights out;
    out.query      = b.parameter(p + "attention/query", {q, h}, {p + "mixer_input"});
    out.key        = b.parameter(p + "attention/key", {k, h}, {p + "mixer_input"});
    out.gate       = b.parameter(p + "attention/gate", {q, h}, {p + "mixer_input"});
    out.value      = b.parameter(p + "attention/value", {k, h}, {p + "mixer_input"});
    out.query_norm = b.direct(p + "attention/query_norm", {a.head_dim});
    out.key_norm   = b.direct(p + "attention/key_norm", {a.head_dim});
    out.output     = b.parameter(p + "attention/output", {h, q}, {p + "attention/gated_output"});
    return out;
}

DenseWeights bind_dense(Bindings& b, std::uint64_t h, std::uint64_t intermediate,
                        const std::string& p, bool draft) {
    const auto input = p + (draft ? "mlp_input" : "ffn_input");
    return {b.parameter(p + "mlp/gate", {intermediate, h}, {input}),
            b.parameter(p + "mlp/up", {intermediate, h}, {input}),
            b.parameter(p + "mlp/down", {h, intermediate},
                        {p + (draft ? "mlp_product" : "mlp/product")})};
}

namespace {

GdnWeights bind_gdn(Bindings& b, const TextConfig& text, const std::string& p) {
    const auto& g    = text.gdn.value();
    const auto h     = text.hidden_size;
    const auto k     = g.key_width();
    const auto v     = g.value_width();
    const auto heads = g.linear_num_value_heads;
    GdnWeights out;
    out.query        = b.parameter(p + "gdn/query", {k, h}, {p + "mixer_input"});
    out.key          = b.parameter(p + "gdn/key", {k, h}, {p + "mixer_input"});
    out.value        = b.parameter(p + "gdn/value", {v, h}, {p + "mixer_input"});
    out.z            = b.parameter(p + "gdn/z", {v, h}, {p + "mixer_input"});
    out.a_projection = b.parameter(p + "gdn/a_projection", {heads, h}, {p + "mixer_input"});
    out.b_projection = b.parameter(p + "gdn/b_projection", {heads, h}, {p + "mixer_input"});
    out.a_log        = b.direct(p + "gdn/a_log", {heads}, QType::FP32);
    out.dt_bias      = b.direct(p + "gdn/dt_bias", {heads}, QType::FP32);
    out.convolution =
        b.direct(p + "gdn/convolution", {g.linear_conv_kernel_dim, g.conv_channels()});
    out.norm   = b.direct(p + "gdn/norm", {g.linear_value_head_dim});
    out.output = b.parameter(p + "gdn/output", {h, v}, {p + "gdn/gated_output"});
    return out;
}

using Range = std::pair<std::uint64_t, std::uint64_t>;

Range scaled(std::uint32_t begin, std::uint32_t end, std::uint64_t unit) {
    return {std::uint64_t(begin) * unit, std::uint64_t(end) * unit};
}

Range offset(Range range, std::uint64_t by) { return {range.first + by, range.second + by}; }

ParameterPiece rows(Range range) { return {ParameterPiece::Cut::Rows, {range}}; }

ParameterPiece columns(std::vector<Range> ranges) {
    return {ParameterPiece::Cut::Columns, std::move(ranges)};
}

// A copy of a whole parameter for a rank: rank 0 binds it as it is, another rank a piece that
// covers it.
WeightId bind_replica(Bindings& b, const std::string& name, const artifact::Shape& shape,
                      const std::vector<std::string>& inputs, std::optional<QType> format,
                      std::size_t rank) {
    if (rank == 0) { return b.parameter(name, shape, inputs, format); }
    const ParameterPiece whole = shape.size() == 2 ? rows({0, shape[0]}) : columns({{0, shape[0]}});
    return b.piece(name, shape, inputs, format, rank, whole);
}

} // namespace

// A norm every rank reads.
WeightId bind_norm_replica(Bindings& b, const std::string& name, std::uint64_t width,
                           std::size_t rank) {
    return bind_replica(b, name, {width}, {}, QType::BF16, rank);
}

namespace {

// A rank's share of a GDN mixer: its value heads, the key heads they read, and their convolution
// channels; the output projection takes the matching input columns. The gated norm is per head,
// and the gating projection is small, so every rank holds all of both and keeps its heads' gates.
GdnWeights bind_gdn_share(Bindings& b, const TextConfig& text, const std::string& p,
                          std::size_t rank, const TensorShare& share) {
    const auto& g      = text.gdn.value();
    const auto h       = text.hidden_size;
    const auto k       = g.key_width();
    const auto v       = g.value_width();
    const auto heads   = g.linear_num_value_heads;
    const auto ratio   = g.linear_num_value_heads / g.linear_num_key_heads;
    const Range key    = scaled(share.key_head_begin, share.key_head_end, g.linear_key_head_dim);
    const Range value  = scaled(share.key_head_begin * ratio, share.key_head_end * ratio,
                                g.linear_value_head_dim);
    const auto input   = std::vector<std::string>{p + "mixer_input"};
    GdnWeights out;
    out.query        = b.piece(p + "gdn/query", {k, h}, input, {}, rank, rows(key));
    out.key          = b.piece(p + "gdn/key", {k, h}, input, {}, rank, rows(key));
    out.value        = b.piece(p + "gdn/value", {v, h}, input, {}, rank, rows(value));
    out.z            = b.piece(p + "gdn/z", {v, h}, input, {}, rank, rows(value));
    // The gating projections stay one parent where the artifact stores them as one: the gating
    // kernel of some geometries reads them only that way.
    const auto& bindings = b.binder.reader().directory().bindings;
    const auto a         = bindings.find(p + "gdn/a_projection"),
               bb        = bindings.find(p + "gdn/b_projection");
    const bool joint     = rank != 0 && a != bindings.end() && bb != bindings.end() &&
                           a->second.parts.size() == 1 && bb->second.parts.size() == 1 &&
                           a->second.parts[0].object == bb->second.parts[0].object;
    if (joint) {
        const std::array<Bindings::JointPart, 2> gating{
            Bindings::JointPart{p + "gdn/a_projection", {heads, h}, input},
            Bindings::JointPart{p + "gdn/b_projection", {heads, h}, input}};
        const auto ids   = b.joint_piece(gating, rank);
        out.a_projection = ids[0];
        out.b_projection = ids[1];
    } else {
        out.a_projection = bind_replica(b, p + "gdn/a_projection", {heads, h}, input, {}, rank);
        out.b_projection = bind_replica(b, p + "gdn/b_projection", {heads, h}, input, {}, rank);
    }
    out.a_log        = bind_replica(b, p + "gdn/a_log", {heads}, {}, QType::FP32, rank);
    out.dt_bias      = bind_replica(b, p + "gdn/dt_bias", {heads}, {}, QType::FP32, rank);
    out.convolution  = b.piece(p + "gdn/convolution", {g.linear_conv_kernel_dim, g.conv_channels()},
                               {}, QType::BF16, rank,
                               columns({key, offset(key, k), offset(value, 2 * k)}));
    out.norm   = bind_replica(b, p + "gdn/norm", {g.linear_value_head_dim}, {}, QType::BF16, rank);
    out.output = b.piece(p + "gdn/output", {h, v}, {p + "gdn/gated_output"}, {}, rank,
                         columns({value}));
    return out;
}

// A rank's share of an attention mixer: the query and gate rows of the query heads that read its KV
// heads, the key and value rows of those KV heads, and the matching input columns of the output
// projection. The q and k norms are per head dimension, so every rank holds both.
AttentionWeights bind_attention_share(Bindings& b, const TextConfig& config, const std::string& p,
                                      std::size_t rank, const TensorShare& share) {
    const auto& a     = config.attention.value();
    const auto h      = config.hidden_size;
    const auto q      = a.query_width();
    const auto k      = a.key_width();
    const auto group  = a.num_attention_heads / a.num_key_value_heads;
    const Range query = scaled(share.kv_head_begin * group, share.kv_head_end * group, a.head_dim);
    const Range kv    = scaled(share.kv_head_begin, share.kv_head_end, a.head_dim);
    // A position-split rank projects every head (its KV range is all of them) and projects the
    // output of its own query heads.
    const Range output_query =
        share.attention_positions
            ? scaled(share.query_head_begin, share.query_head_end, a.head_dim)
            : query;
    const auto input  = std::vector<std::string>{p + "mixer_input"};
    AttentionWeights out;
    out.query      = b.piece(p + "attention/query", {q, h}, input, {}, rank, rows(query));
    out.key        = b.piece(p + "attention/key", {k, h}, input, {}, rank, rows(kv));
    out.gate       = b.piece(p + "attention/gate", {q, h}, input, {}, rank, rows(query));
    out.value      = b.piece(p + "attention/value", {k, h}, input, {}, rank, rows(kv));
    out.query_norm = bind_norm_replica(b, p + "attention/query_norm", a.head_dim, rank);
    out.key_norm   = bind_norm_replica(b, p + "attention/key_norm", a.head_dim, rank);
    out.output = b.piece(p + "attention/output", {h, q}, {p + "attention/gated_output"}, {}, rank,
                         columns({output_query}));
    return out;
}

// A rank's share of a dense FFN: its intermediate rows of gate and up, and those input columns of
// down.
DenseWeights bind_dense_share(Bindings& b, std::uint64_t h, std::uint64_t intermediate,
                              const std::string& p, std::size_t rank, const TensorShare& share) {
    const Range part  = scaled(share.ffn_begin, share.ffn_end, 1);
    const auto input  = std::vector<std::string>{p + "ffn_input"};
    return {b.piece(p + "mlp/gate", {intermediate, h}, input, {}, rank, rows(part)),
            b.piece(p + "mlp/up", {intermediate, h}, input, {}, rank, rows(part)),
            b.piece(p + "mlp/down", {h, intermediate}, {p + "mlp/product"}, {}, rank,
                    columns({part}))};
}

// A rank's share of a MoE layer: the router (rank 0 binds it as it is, every other rank a replica)
// and either its experts, one device selection of each bank, with the shared expert on rank 0
// alone; or, where the ranks divide the experts' intermediate columns, its columns of every
// expert and of the shared one: one selection of each bank taking each expert's gate rows and up
// rows of those columns, and one taking those columns of every down.
MoeWeights bind_moe_share(Bindings& b, const TextConfig& config, const std::string& prefix,
                          std::size_t rank, const TensorShare& share) {
    const auto& moe = std::get<MoeConfig>(config.ffn);
    const auto p    = prefix + "moe/";
    const std::vector<std::string> input{prefix + "ffn_input"};
    const auto h      = config.hidden_size;
    const auto ir     = moe.moe_intermediate_size;
    const auto shared = moe.shared_expert_intermediate_size;
    const bool divided = share.ffn_end > share.ffn_begin;
    MoeWeights out;
    out.share = MoeWeights::ExpertShare{
        .expert_begin  = share.expert_begin,
        .shared_expert = divided || rank == 0,
        .intermediate  = divided ? share.ffn_end - share.ffn_begin : 0};
    if (divided) {
        if (rank == 0) {
            out.router       = b.parameter(p + "router", {moe.num_experts, h}, input);
            out.shared_score = b.parameter(p + "shared_score", {1, h}, input);
        } else {
            const std::array<Bindings::JointPart, 2> router{
                Bindings::JointPart{p + "router", {moe.num_experts, h}, input},
                Bindings::JointPart{p + "shared_score", {1, h}, input}};
            const auto ids   = b.joint_piece(router, rank);
            out.router       = ids[0];
            out.shared_score = ids[1];
        }
        const ParameterPiece rows{.cut    = ParameterPiece::Cut::Rows,
                                  .ranges = {{share.ffn_begin, share.ffn_end}}};
        const ParameterPiece columns{.cut    = ParameterPiece::Cut::Columns,
                                     .ranges = {{share.ffn_begin, share.ffn_end}}};
        const std::array<Bindings::JointPart, 2> shared_gate_up{
            Bindings::JointPart{p + "shared/gate", {shared, h}, input},
            Bindings::JointPart{p + "shared/up", {shared, h}, input}};
        const auto sgu = b.joint_piece(shared_gate_up, rank, rows);
        const std::array<Bindings::JointPart, 1> shared_down{
            Bindings::JointPart{p + "shared/down", {h, shared}, {p + "shared/product"}}};
        out.shared = {sgu[0], sgu[1], b.joint_piece(shared_down, rank, columns)[0]};
        std::vector<Bindings::JointPart> gate_up, down;
        for (std::uint32_t e = 0; e < moe.num_experts; ++e) {
            const auto ep = p + "experts/" + std::to_string(e) + "/";
            gate_up.push_back({ep + "gate", {ir, h}, input});
            gate_up.push_back({ep + "up", {ir, h}, input});
            down.push_back({ep + "down", {h, ir}, {ep + "product"}});
        }
        const auto gu = b.joint_piece(gate_up, rank, rows);
        const auto d  = b.joint_piece(down, rank, columns);
        for (std::size_t i = 0; i < d.size(); ++i) {
            out.experts.push_back({gu[2 * i], gu[2 * i + 1], d[i]});
        }
        return out;
    }
    if (rank == 0) {
        out.router       = b.parameter(p + "router", {moe.num_experts, h}, input);
        out.shared_score = b.parameter(p + "shared_score", {1, h}, input);
        out.shared       = {b.parameter(p + "shared/gate", {shared, h}, input),
                            b.parameter(p + "shared/up", {shared, h}, input),
                            b.parameter(p + "shared/down", {h, shared}, {p + "shared/product"})};
    } else {
        const std::array<Bindings::JointPart, 2> router{
            Bindings::JointPart{p + "router", {moe.num_experts, h}, input},
            Bindings::JointPart{p + "shared_score", {1, h}, input}};
        const auto ids   = b.joint_piece(router, rank);
        out.router       = ids[0];
        out.shared_score = ids[1];
    }
    std::vector<Bindings::JointPart> gate_up, down;
    for (std::uint32_t e = share.expert_begin; e < share.expert_end; ++e) {
        const auto ep = p + "experts/" + std::to_string(e) + "/";
        gate_up.push_back({ep + "gate", {ir, h}, input});
        gate_up.push_back({ep + "up", {ir, h}, input});
        down.push_back({ep + "down", {h, ir}, {ep + "product"}});
    }
    const auto gu = b.joint_piece(gate_up, rank);
    const auto d  = b.joint_piece(down, rank);
    for (std::size_t i = 0; i < d.size(); ++i) {
        out.experts.push_back({gu[2 * i], gu[2 * i + 1], d[i]});
    }
    return out;
}

MoeWeights bind_moe(Bindings& b, const TextConfig& config, const std::string& prefix) {
    const auto& moe   = std::get<MoeConfig>(config.ffn);
    const auto p      = prefix + "moe/";
    const auto input  = prefix + "ffn_input";
    const auto h      = config.hidden_size;
    const auto ir     = moe.moe_intermediate_size;
    const auto shared = moe.shared_expert_intermediate_size;
    MoeWeights out;
    out.router       = b.parameter(p + "router", {moe.num_experts, h}, {input});
    out.shared_score = b.parameter(p + "shared_score", {1, h}, {input});
    out.experts.reserve(moe.num_experts);
    const auto residency = prefix.starts_with("mtp/") && b.resident_mtp_experts
                               ? artifact::Residency::Device
                           : b.expert_residency == ExpertResidency::Host
                               ? artifact::Residency::Pinned
                           : b.expert_residency == ExpertResidency::Disk
                               ? artifact::Residency::Values
                               : artifact::Residency::Device;
    for (std::uint32_t e = 0; e < moe.num_experts; ++e) {
        const auto ep = p + "experts/" + std::to_string(e) + "/";
        out.experts.push_back({b.parameter(ep + "gate", {ir, h}, {input}, {}, residency),
                               b.parameter(ep + "up", {ir, h}, {input}, {}, residency),
                               b.parameter(ep + "down", {h, ir}, {ep + "product"}, {}, residency)});
    }
    out.shared = {b.parameter(p + "shared/gate", {shared, h}, {input}),
                  b.parameter(p + "shared/up", {shared, h}, {input}),
                  b.parameter(p + "shared/down", {h, shared}, {p + "shared/product"})};
    return out;
}

// A stage owns its layers whole: norms, the attention or GDN projections, and the FFN or experts all
// live on the stage's device, next to the KV cache and recurrent state that layer reads and writes.
// That is what lets the stage run its layers with the same kernels as a single GPU.
void place_layer(Bindings& b, const BlockWeights& block, std::size_t rank) {
    for (const WeightId id : layer_weights(block)) { b.place(id, rank); }
}

} // namespace

std::vector<DenseWeights> bind_dense_shares(Bindings& b, std::uint64_t hidden,
                                            std::uint64_t intermediate, const std::string& prefix,
                                            std::span<const TensorShare> shares) {
    const auto down = b.column_ends(prefix + "mlp/down", {hidden, intermediate},
                                    {prefix + "mlp/product"});
    const auto gate = b.row_ends(prefix + "mlp/gate", {intermediate, hidden});
    const auto up   = b.row_ends(prefix + "mlp/up", {intermediate, hidden});
    for (std::size_t rank = 0; rank + 1 < shares.size(); ++rank) {
        const std::uint64_t end = shares[rank].ffn_end;
        if (end >= intermediate || !down[end] || !gate[end] || !up[end]) { return {}; }
    }
    std::vector<DenseWeights> out;
    for (std::size_t rank = 0; rank < shares.size(); ++rank) {
        out.push_back(bind_dense_share(b, hidden, intermediate, prefix, rank, shares[rank]));
    }
    return out;
}

std::vector<AttentionWeights> bind_attention_shares(Bindings& b, const TextConfig& config,
                                                    const std::string& prefix,
                                                    std::span<const TensorShare> shares) {
    if (shares.size() < 2 || shares.front().attention_positions) { return {}; }
    const auto& a      = config.attention.value();
    const auto h       = config.hidden_size;
    const auto q       = a.query_width();
    const auto k       = a.key_width();
    const auto group   = a.num_attention_heads / a.num_key_value_heads;
    const auto query   = b.row_ends(prefix + "attention/query", {q, h});
    const auto gate    = b.row_ends(prefix + "attention/gate", {q, h});
    const auto key     = b.row_ends(prefix + "attention/key", {k, h});
    const auto value   = b.row_ends(prefix + "attention/value", {k, h});
    const auto output  = b.column_ends(prefix + "attention/output", {h, q},
                                       {prefix + "attention/gated_output"});
    std::size_t ranks = 0;
    while (ranks < shares.size() && shares[ranks].kv_head_end > shares[ranks].kv_head_begin) {
        ++ranks;
    }
    if (ranks < 2) { return {}; }
    for (std::size_t rank = 0; rank + 1 < ranks; ++rank) {
        const std::uint64_t heads = shares[rank].kv_head_end;
        const std::uint64_t rows  = heads * group * a.head_dim;
        const std::uint64_t kv    = heads * a.head_dim;
        if (!query[rows] || !gate[rows] || !output[rows] || !key[kv] || !value[kv]) { return {}; }
    }
    std::vector<AttentionWeights> out;
    for (std::size_t rank = 0; rank < ranks; ++rank) {
        out.push_back(bind_attention_share(b, config, prefix, rank, shares[rank]));
    }
    return out;
}

std::vector<WeightId> layer_weights(const BlockWeights& block) {
    std::vector<WeightId> out{block.input_norm, block.post_attention_norm};
    if (const auto* attention = std::get_if<AttentionWeights>(&block.mixer)) {
        out.insert(out.end(), {attention->query, attention->key, attention->gate, attention->value,
                               attention->query_norm, attention->key_norm, attention->output});
    } else {
        const auto& gdn = std::get<GdnWeights>(block.mixer);
        out.insert(out.end(), {gdn.query, gdn.key, gdn.value, gdn.z, gdn.a_projection,
                               gdn.b_projection, gdn.a_log, gdn.dt_bias, gdn.convolution, gdn.norm,
                               gdn.output});
    }
    const auto add_dense = [&](const DenseWeights& dense) {
        out.insert(out.end(), {dense.gate, dense.up, dense.down});
    };
    if (const auto* dense = std::get_if<DenseWeights>(&block.ffn)) {
        add_dense(*dense);
        return out;
    }
    const auto& moe = std::get<MoeWeights>(block.ffn);
    out.push_back(moe.router);
    out.push_back(moe.shared_score);
    for (const auto& expert : moe.experts) { add_dense(expert); }
    add_dense(moe.shared);
    return out;
}

BlockWeights bind_block(Bindings& b, const TextConfig& config, const std::string& p,
                        MixerKind mixer, const DenseWeights* ffn,
                        const AttentionWeights* attention) {
    BlockWeights out;
    out.input_norm          = b.direct(p + "input_norm", {config.hidden_size});
    out.post_attention_norm = b.direct(p + "post_attention_norm", {config.hidden_size});
    if (attention != nullptr) {
        out.mixer = *attention;
    } else if (mixer == MixerKind::FullAttention) {
        out.mixer = bind_attention(b, config, p);
    } else {
        out.mixer = bind_gdn(b, config, p);
    }
    if (ffn != nullptr) {
        out.ffn = *ffn;
    } else if (const auto* dense = std::get_if<DenseConfig>(&config.ffn)) {
        out.ffn = bind_dense(b, config.hidden_size, dense->intermediate_size, p);
    } else {
        out.ffn = bind_moe(b, config, p);
    }
    return out;
}

StagePlan plan_stage_plan(std::uint32_t layers, const LoadOptions& options) {
    if (options.tensor_parallel) {
        if (!options.stage_layers.empty()) {
            throw std::invalid_argument("--stage-layers and --tensor-parallel cannot be combined");
        }
        return StagePlan(layers);
    }
    if (options.ranks <= 1) {
        if (!options.stage_layers.empty()) {
            throw std::invalid_argument("--stage-layers needs --devices naming more than one device");
        }
        return StagePlan(layers);
    }
    if (options.ranks > layers) {
        throw std::invalid_argument("--devices asks for more devices than the model has layers");
    }
    // The layer counts come from the caller when it has measured device memory (the engine solves
    // them against each card's free bytes), or from `--stage-layers`. Without either, equal counts:
    // the sensible default while layers are near-uniform in size.
    if (!options.stage_layers.empty()) {
        if (options.stage_layers.size() != options.ranks) {
            throw std::invalid_argument("--stage-layers lists " +
                                        std::to_string(options.stage_layers.size()) +
                                        " counts but --devices names " +
                                        std::to_string(options.ranks) + " devices");
        }
        return StagePlan::from_layer_counts(layers, options.stage_layers);
    }
    return StagePlan::even(layers, options.ranks);
}

namespace {

// Tensor parallelism: rank 0's share of a layer, and every other rank's share. A rank without KV
// heads holds no attention.
BlockWeights bind_block_shares(Bindings& b, const TextConfig& config, const std::string& p,
                               MixerKind mixer, std::span<const TensorShare> shares,
                               std::vector<std::vector<RankLayerWeights>>& rank_layers) {
    const auto* dense = std::get_if<DenseConfig>(&config.ffn);
    const auto ffn_share = [&](std::size_t rank) -> std::variant<DenseWeights, MoeWeights> {
        if (dense != nullptr) {
            return bind_dense_share(b, config.hidden_size, dense->intermediate_size, p, rank,
                                    shares[rank]);
        }
        return bind_moe_share(b, config, p, rank, shares[rank]);
    };
    BlockWeights out;
    out.input_norm          = bind_norm_replica(b, p + "input_norm", config.hidden_size, 0);
    out.post_attention_norm = bind_norm_replica(b, p + "post_attention_norm", config.hidden_size, 0);
    if (mixer == MixerKind::FullAttention) {
        out.mixer = bind_attention_share(b, config, p, 0, shares[0]);
    } else {
        out.mixer = bind_gdn_share(b, config, p, 0, shares[0]);
    }
    out.ffn = ffn_share(0);
    for (std::size_t rank = 1; rank < shares.size(); ++rank) {
        RankLayerWeights share;
        share.input_norm = bind_norm_replica(b, p + "input_norm", config.hidden_size, rank);
        share.post_attention_norm =
            bind_norm_replica(b, p + "post_attention_norm", config.hidden_size, rank);
        if (mixer != MixerKind::FullAttention) {
            share.gdn = bind_gdn_share(b, config, p, rank, shares[rank]);
        } else if (shares[rank].kv_head_end > shares[rank].kv_head_begin) {
            share.attention = bind_attention_share(b, config, p, rank, shares[rank]);
        }
        share.ffn = ffn_share(rank);
        rank_layers[rank - 1].push_back(std::move(share));
    }
    return out;
}

// The positions in (0, count) where every layer can end a share: `allowed(layer, position)` for
// each layer that has the dimension.
template <class Allowed>
std::vector<std::uint64_t> common_ends(std::uint64_t count, std::uint32_t layers,
                                       const Allowed& allowed) {
    std::vector<bool> ok(count, true);
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        for (std::uint64_t position = 1; position < count; ++position) {
            if (ok[position] && !allowed(layer, position)) { ok[position] = false; }
        }
    }
    std::vector<std::uint64_t> out;
    for (std::uint64_t position = 1; position < count; ++position) {
        if (ok[position]) { out.push_back(position); }
    }
    return out;
}

// Each rank's GDN key heads, attention KV heads and dense FFN rows or experts, in proportion to
// `options.tensor_split` (evenly without one). A share may end only where every layer's projections
// cut whole stored runs: the column-cut ones (the GDN and attention output projections over the
// value or query heads, the FFN's down projection over its rows) and the row-cut ones (the GDN key
// and value projections, the attention projections, the FFN's gate and up). Experts divide at any
// expert; two ranks of about equal weight (each 40% to 60%) divide every expert's intermediate
// columns in halves instead, routed and shared, where the stored banks cut there, so each rank
// runs every selected expert's half and a token's routed work splits evenly whichever experts it
// selects.
std::vector<TensorShare> plan_tensor_shares(Bindings& b, const TextConfig& config,
                                            const LoadOptions& options) {
    const auto* dense = std::get_if<DenseConfig>(&config.ffn);
    if (!config.gdn) { throw std::invalid_argument("--tensor-parallel supports GDN models"); }
    if (dense == nullptr && options.expert_residency != ExpertResidency::Device) {
        throw std::invalid_argument(
            "--tensor-parallel divides the experts over the devices and keeps them there; it "
            "cannot be combined with --expert-residency host or disk");
    }
    std::vector<double> weights = options.tensor_split;
    if (weights.empty()) { weights.assign(options.ranks, 1.0); }
    if (weights.size() != options.ranks) {
        throw std::invalid_argument("--tensor-split needs one weight per device");
    }
    const auto& g          = *config.gdn;
    const auto& a          = config.attention.value();
    const auto h           = config.hidden_size;
    const auto ratio       = g.linear_num_value_heads / g.linear_num_key_heads;
    const auto layers      = config.num_hidden_layers;
    const auto i           = dense != nullptr ? dense->intermediate_size : 0;
    // Per layer: the value columns of the GDN output projection and the rows of its key and value
    // projections; the FFN's rows of gate and up and columns of down.
    struct Ends {
        std::vector<bool> output, key_rows, value_rows, ffn_rows;
        // Attention: the output projection's query columns, the query and gate rows, the key and
        // value rows.
        std::vector<bool> attention_output, query_rows, kv_rows;
    };
    std::vector<Ends> ends(layers);
    const auto both = [](std::vector<bool> a, const std::vector<bool>& b) {
        for (std::size_t at = 0; at < a.size(); ++at) { a[at] = a[at] && b[at]; }
        return a;
    };
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const std::string p = "text/layers/" + std::to_string(layer) + "/";
        auto& e             = ends[layer];
        if (config.layer_types[layer] != MixerKind::FullAttention) {
            e.output   = b.column_ends(p + "gdn/output", {h, g.value_width()}, {p + "gdn/gated_output"});
            e.key_rows = both(b.row_ends(p + "gdn/query", {g.key_width(), h}),
                              b.row_ends(p + "gdn/key", {g.key_width(), h}));
            e.value_rows = both(b.row_ends(p + "gdn/value", {g.value_width(), h}),
                                b.row_ends(p + "gdn/z", {g.value_width(), h}));
        } else {
            const auto q = a.query_width(), k = a.key_width();
            e.attention_output =
                b.column_ends(p + "attention/output", {h, q}, {p + "attention/gated_output"});
            e.query_rows = both(b.row_ends(p + "attention/query", {q, h}),
                                b.row_ends(p + "attention/gate", {q, h}));
            e.kv_rows    = both(b.row_ends(p + "attention/key", {k, h}),
                                b.row_ends(p + "attention/value", {k, h}));
        }
        if (dense != nullptr) {
            e.ffn_rows = both(both(b.column_ends(p + "mlp/down", {h, i}, {p + "mlp/product"}),
                                   b.row_ends(p + "mlp/gate", {i, h})),
                              b.row_ends(p + "mlp/up", {i, h}));
        }
    }
    const auto key_ends = common_ends(
        g.linear_num_key_heads, layers, [&](std::uint32_t layer, std::uint64_t heads) {
            const auto& e = ends[layer];
            const auto values = heads * ratio * g.linear_value_head_dim;
            return e.output.empty() || (e.output[values] && e.value_rows[values] &&
                                        e.key_rows[heads * g.linear_key_head_dim]);
        });
    const auto heads = split_shares(g.linear_num_key_heads, key_ends, weights);
    // The KV heads go to the first ranks, as many as the heads' allowed ends can give a share, in
    // proportion to their weights: the ranks holding attention are a prefix, which is what the
    // exchange after attention sums.
    const auto group = a.num_attention_heads / a.num_key_value_heads;
    const auto kv_head_ends =
        common_ends(a.num_key_value_heads, layers, [&](std::uint32_t layer, std::uint64_t kv) {
            const auto& e = ends[layer];
            return e.attention_output.empty() ||
                   (e.attention_output[kv * group * a.head_dim] &&
                    e.query_rows[kv * group * a.head_dim] && e.kv_rows[kv * a.head_dim]);
        });
    const auto kv_ranks = std::min<std::size_t>(options.ranks, kv_head_ends.size() + 1);
    // By positions when asked: every rank then attends over its own pages and projects the output
    // of a run of query heads, which may end wherever every layer's output projection cuts whole
    // stored runs. By heads, the ranks past the KV heads' ends hold no attention.
    const bool positions = options.tensor_attention == TensorAttention::Positions;
    std::vector<std::uint64_t> kv_heads(options.ranks + 1, a.num_key_value_heads);
    std::vector<std::uint64_t> query_heads(options.ranks + 1, 0);
    if (positions) {
        if (options.ranks > static_cast<std::size_t>(ops::kCausalAttentionMaximumMergeRanks)) {
            throw std::invalid_argument("--tensor-attention positions supports up to " +
                                        std::to_string(ops::kCausalAttentionMaximumMergeRanks) +
                                        " devices");
        }
        const auto query_ends = common_ends(
            a.num_attention_heads, layers, [&](std::uint32_t layer, std::uint64_t heads) {
                const auto& e = ends[layer];
                return e.attention_output.empty() || e.attention_output[heads * a.head_dim];
            });
        if (query_ends.size() + 1 < options.ranks) {
            throw std::invalid_argument(
                "--tensor-attention positions: the attention output projection cannot be divided "
                "over this many devices");
        }
        query_heads = split_shares(a.num_attention_heads, query_ends, weights);
    } else {
        const std::vector<double> kv_weights(
            weights.begin(), weights.begin() + static_cast<std::ptrdiff_t>(kv_ranks));
        const auto ends_of = split_shares(a.num_key_value_heads, kv_head_ends, kv_weights);
        std::copy(ends_of.begin(), ends_of.end(), kv_heads.begin());
    }
    std::vector<std::uint64_t> rows(options.ranks + 1, 0), experts(options.ranks + 1, 0);
    if (dense != nullptr) {
        const auto row_ends = common_ends(i, layers, [&](std::uint32_t layer, std::uint64_t at) {
            return ends[layer].ffn_rows[at];
        });
        rows                = split_shares(i, row_ends, weights);
    } else {
        const auto& m    = std::get<MoeConfig>(config.ffn);
        const auto count = m.num_experts;
        const auto half  = m.moe_intermediate_size / 2;
        const double first_share =
            options.ranks == 2 ? weights[0] / (weights[0] + weights[1]) : 0.0;
        bool halves = options.ranks == 2 && first_share >= 0.4 && first_share <= 0.6 &&
                      m.moe_intermediate_size == m.shared_expert_intermediate_size;
        for (std::uint32_t layer = 0; halves && layer < layers; ++layer) {
            const std::string e = "text/layers/" + std::to_string(layer) + "/moe/";
            const auto ir       = m.moe_intermediate_size;
            halves = b.row_ends(e + "experts/0/gate", {ir, h})[half] &&
                     b.row_ends(e + "experts/0/up", {ir, h})[half] &&
                     b.column_ends(e + "experts/0/down", {h, ir}, {e + "experts/0/product"})[half] &&
                     b.row_ends(e + "shared/gate", {ir, h})[half] &&
                     b.row_ends(e + "shared/up", {ir, h})[half] &&
                     b.column_ends(e + "shared/down", {h, ir}, {e + "shared/product"})[half];
        }
        if (halves) {
            rows    = {0, half, m.moe_intermediate_size};
            experts = {0, count, count};
        } else {
            // Whole experts: rank 0 also runs the shared expert, so it takes fewer of the routed
            // ones, which a token reaches num_experts_per_tok at a time.
            const double shared = double(m.shared_expert_intermediate_size) /
                                  (double(m.num_experts_per_tok) * m.moe_intermediate_size);
            experts = split_shares(count, unit_positions(count, 1),
                                   shares_beside_first(weights, shared));
        }
    }
    std::vector<TensorShare> out;
    for (std::size_t rank = 0; rank < options.ranks; ++rank) {
        out.push_back({.key_head_begin = static_cast<std::uint32_t>(heads[rank]),
                       .key_head_end   = static_cast<std::uint32_t>(heads[rank + 1]),
                       .kv_head_begin  = positions ? 0U : static_cast<std::uint32_t>(kv_heads[rank]),
                       .kv_head_end    = positions ? a.num_key_value_heads
                                                   : static_cast<std::uint32_t>(kv_heads[rank + 1]),
                       .ffn_begin      = static_cast<std::uint32_t>(rows[rank]),
                       .ffn_end        = static_cast<std::uint32_t>(rows[rank + 1]),
                       .expert_begin   = static_cast<std::uint32_t>(
                           rows[options.ranks] != 0 && dense == nullptr ? 0 : experts[rank]),
                       .expert_end     = static_cast<std::uint32_t>(experts[rank + 1]),
                       .attention_positions = positions,
                       .query_head_begin = static_cast<std::uint32_t>(query_heads[rank]),
                       .query_head_end   = static_cast<std::uint32_t>(query_heads[rank + 1])});
        (void)tensor_parallel_share(config, out.back());
    }
    return out;
}

// Tensor parallelism: the rows of vocabulary head `name` ([count, columns], read through `inputs`)
// each rank projects at decode, in proportion to the split (evenly without one) and ending where
// the stored blocks and the head's Linear route allow, each bound as a piece on its rank.
HeadShareWeights bind_head_shares(Bindings& b, const std::string& name, std::uint64_t count,
                                  std::uint64_t columns, const std::vector<std::string>& inputs,
                                  const LoadOptions& options) {
    std::vector<double> weights = options.tensor_split;
    if (weights.empty()) { weights.assign(options.ranks, 1.0); }
    const std::vector<bool> allowed = b.row_ends(name, {count, columns});
    std::vector<std::uint64_t> ends;
    for (std::uint64_t row = 1; row < count; ++row) {
        if (allowed[row]) { ends.push_back(row); }
    }
    HeadShareWeights out;
    out.rows = split_shares(count, ends, weights);
    for (std::size_t rank = 0; rank < options.ranks; ++rank) {
        out.shares.push_back(b.piece(name, {count, columns}, inputs, {}, rank,
                                     rows({out.rows[rank], out.rows[rank + 1]})));
    }
    return out;
}

} // namespace

TextWeights bind_text(Bindings& b, const TextConfig& config, const LoadOptions& options,
                      StagePlan stages) {
    TextWeights out;
    out.stages = std::move(stages);
    if (options.tensor_parallel) {
        out.tensor_shares = plan_tensor_shares(b, config, options);
        out.rank_layers.resize(options.ranks - 1);
    }
    // The table is only gathered, a row per token, so a host table costs each step's rows over
    // the bus and nothing else; every reader (prefill, decode, MTP, drafters) shares this one.
    out.token_embedding = b.parameter(
        "text/token_embedding", {config.vocab_size, config.hidden_size}, {}, {},
        options.embedding_residency == EmbeddingResidency::Host ? artifact::Residency::Mapped
                                                                : artifact::Residency::Device);
    std::vector<std::string> head_inputs{"text/final_hidden"};
    // A standalone DSpine drafter reads the head through the target's own use of it.
    if (options.speculative != SpeculativeBackend::None && !options.proposal_enabled() &&
        !options.dspine()) {
        head_inputs.push_back(std::string(options.speculative_component()) + "/final_hidden");
    }
    if (options.tensor_parallel && !options.lm_head_q4 && !options.lm_head_q6) {
        out.output_head_shares = bind_head_shares(b, "text/output_head", config.vocab_size,
                                                  config.hidden_size, head_inputs, options);
    }
    out.output_head = b.parameter("text/output_head", {config.vocab_size, config.hidden_size},
                                  std::move(head_inputs));
    out.final_norm  = b.direct("text/final_norm", {config.hidden_size});
    out.layers.reserve(config.num_hidden_layers);
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const std::string prefix = "text/layers/" + std::to_string(i) + "/";
        if (options.tensor_parallel) {
            out.layers.push_back(bind_block_shares(b, config, prefix, config.layer_types[i],
                                                   out.tensor_shares, out.rank_layers));
            continue;
        }
        out.layers.push_back(bind_block(b, config, prefix, config.layer_types[i]));
        // A layer lives on its stage. The embedding, final norm and head stay on rank 0 with the
        // round state, sampling and prefill buffers that reach them directly: the last stage sends
        // the residual back, one extra hop per forward pass.
        // One stage is the single-GPU route and binds exactly what it always bound.
        if (!out.stages.single_stage()) {
            place_layer(b, out.layers.back(), out.stages.placement(i).stage);
        }
    }
    return out;
}

ProposalWeights bind_proposal(Bindings& b, const artifact::Proposal& proposal,
                              const TextConfig& target, const LoadOptions& options,
                              std::uint32_t public_tokens) {
    const auto rows = proposal.indexed ? proposal.rows : target.vocab_size;
    if (rows > std::numeric_limits<std::uint32_t>::max() ||
        (proposal.indexed && rows > public_tokens)) {
        throw artifact::ArtifactError("proposal rows exceed the output domain");
    }
    ProposalWeights out;
    out.rows = static_cast<std::uint32_t>(rows);
    std::vector<std::string> inputs{std::string(options.speculative_component()) +
                                    "/final_hidden"};
    if (options.dspine()) {
        // A DSpine drafter is an artifact of its own, so the model declares no input of the
        // proposal head for it. Its final hidden is a masked drafter's: the head takes it
        // through DFlash2's input, else DFlash's, else MTP's, whichever the artifact declares.
        const auto& uses = b.binder.reader().directory().uses;
        for (const char* drafter : {"dflash2", "dflash", "mtp"}) {
            std::string input = std::string(drafter) + "/final_hidden";
            if (uses.find({"proposal/head", input}) != uses.end()) {
                inputs = {std::move(input)};
                break;
            }
        }
    }
    if (options.tensor_parallel) {
        out.head_shares =
            bind_head_shares(b, "proposal/head", rows, target.hidden_size, inputs, options);
    }
    out.head = b.parameter("proposal/head", {rows, target.hidden_size}, inputs);
    if (proposal.indexed) {
        out.token_ids = b.direct("proposal/token_ids", {rows}, QType::INT32);
        out.global_token_ids =
            b.binder.values(b.at(*out.token_ids).reference.binding, QType::INT32).integers();
        std::set<std::int32_t> unique;
        for (const auto id : out.global_token_ids) {
            if (id < 0 || std::uint32_t(id) >= public_tokens || !unique.insert(id).second) {
                throw artifact::ArtifactError(
                    "proposal token IDs must be unique and in the public domain");
            }
        }
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
