#include "models/qwen4_exp/model.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/views.h"
#include "core/qtype.h"
#include "core/tensor_split.h"
#include "models/expert_parallel.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/load/bindings.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen4_exp {
namespace {

using qwen3_5::loading::Bindings;
using Shape = artifact::Shape;

FrontendResources bind_resources(artifact::Binder& binder, const TextConfig& config,
                                 bool vision) {
    const auto resource = [&](std::string_view role, const char* component = "text") {
        const auto bytes = binder.host_object(binder.resource(component, role));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    FrontendResources out;
    out.tokenizer_json         = resource("tokenizer.json");
    out.tokenizer_config_json  = resource("tokenizer_config.json");
    out.chat_template_jinja    = resource("chat_template.jinja");
    out.generation_config_json = resource("generation_config.json");
    if (vision) {
        out.preprocessor_config_json       = resource("preprocessor_config.json", "vision");
        out.video_preprocessor_config_json = resource("video_preprocessor_config.json", "vision");
    }
    out.tokenizer =
        std::make_shared<const qwen3_5::frontend::Tokenizer>(qwen3_5::frontend::TokenizerResources{
            out.tokenizer_json, out.tokenizer_config_json, out.generation_config_json});
    const auto count = out.tokenizer->vocab_size();
    if (!count || count > config.vocab_size || !out.tokenizer->has_exact_token_domain(count)) {
        throw artifact::ArtifactError(
            "tokenizer must expose a contiguous public domain within embedding rows");
    }
    out.public_token_count = static_cast<std::uint32_t>(count);
    for (const auto token : out.tokenizer->default_stop_token_ids()) {
        if (!out.tokenizer->is_valid_token(token)) {
            throw artifact::ArtifactError("stop token is outside the public token domain");
        }
    }
    return out;
}

HyperConnectionWeights bind_hc(Bindings& b, const TextConfig& c, const std::string& p,
                               bool inject) {
    const std::uint64_t width = std::uint64_t(c.hc_count) * c.hidden_size;
    HyperConnectionWeights out;
    out.norm = b.direct(p + "norm", {width});
    out.down = b.parameter(p + "down", {c.hc_lowrank, width}, {p + "normalized"});
    out.up   = b.parameter(p + "up", {width, c.hc_lowrank}, {p + "low"});
    if (inject) { out.inject = b.parameter(p + "inject", {c.hc_count, width}, {p + "normalized"}); }
    return out;
}

GdnWeights bind_gdn(Bindings& b, const TextConfig& c, const std::string& p) {
    const std::uint64_t h  = c.hidden_size;
    const std::uint64_t kd = std::uint64_t(c.linear_num_key_heads) * c.linear_key_head_dim;
    const std::uint64_t vd = std::uint64_t(c.linear_num_value_heads) * c.linear_value_head_dim;
    const std::string g = p + "gdn/", input = p + "mixer_input";
    GdnWeights out;
    out.a_log        = b.direct(g + "a_log", {c.linear_num_value_heads}, QType::FP32);
    out.dt_bias      = b.direct(g + "dt_bias", {c.linear_num_value_heads}, QType::FP32);
    out.convolution  = b.direct(g + "convolution", {c.linear_conv_kernel_dim, 2 * kd + vd});
    out.a_projection = b.parameter(g + "a_projection", {c.linear_num_value_heads, h}, {input});
    out.b_projection = b.parameter(g + "b_projection", {c.linear_num_value_heads, h}, {input});
    out.query        = b.parameter(g + "query", {kd, h}, {input});
    out.key          = b.parameter(g + "key", {kd, h}, {input});
    out.value        = b.parameter(g + "value", {vd, h}, {input});
    out.z            = b.parameter(g + "z", {vd, h}, {input});
    out.norm         = b.direct(g + "norm", {c.linear_value_head_dim});
    out.output       = b.parameter(g + "output", {h, vd}, {g + "gated_output"});
    return out;
}

SparseAttentionWeights bind_attention(Bindings& b, const TextConfig& c, const std::string& p) {
    const std::uint64_t h  = c.hidden_size;
    const std::uint64_t q  = std::uint64_t(c.num_attention_heads) * c.head_dim;
    const std::uint64_t kv = std::uint64_t(c.num_key_value_heads) * c.head_dim;
    const std::string a = p + "attention/", i = p + "indexer/", input = p + "mixer_input";
    SparseAttentionWeights out;
    out.query       = b.parameter(a + "query", {q, h}, {input});
    out.gate        = b.parameter(a + "gate", {q, h}, {input});
    out.key         = b.parameter(a + "key", {kv, h}, {input});
    out.value       = b.parameter(a + "value", {kv, h}, {input});
    out.query_norm  = b.direct(a + "query_norm", {c.head_dim});
    out.key_norm    = b.direct(a + "key_norm", {c.head_dim});
    out.output      = b.parameter(a + "output", {h, q}, {a + "gated_output"});
    out.index_query = b.parameter(
        i + "query", {std::uint64_t(c.indexer_n_heads) * c.indexer_head_dim, h}, {input});
    out.index_key        = b.parameter(i + "key", {c.indexer_head_dim, h}, {input});
    out.index_query_norm = b.direct(i + "query_norm", {c.indexer_head_dim});
    out.index_key_norm   = b.direct(i + "key_norm", {c.indexer_head_dim});
    return out;
}

using Piece = qwen3_5::loading::ParameterPiece;
using Range = std::pair<std::uint64_t, std::uint64_t>;

Piece row_piece(Range range) { return {Piece::Cut::Rows, {range}}; }

Piece column_piece(std::vector<Range> ranges) { return {Piece::Cut::Columns, std::move(ranges)}; }

Range scaled(std::uint64_t begin, std::uint64_t end, std::uint64_t unit) {
    return {begin * unit, end * unit};
}

Range shifted(Range range, std::uint64_t by) { return {range.first + by, range.second + by}; }

// Tensor parallelism: a parameter a rank after the first holds whole, as a piece of its own; a
// matrix by its rows, a vector by its elements.
WeightId replica(Bindings& b, const std::string& name, const Shape& shape,
                 const std::vector<std::string>& inputs, std::optional<QType> format,
                 std::size_t rank) {
    return b.piece(name, shape, inputs, format, rank,
                   shape.size() == 2 ? row_piece({0, shape[0]}) : column_piece({{0, shape[0]}}));
}

HyperConnectionWeights bind_hc_replica(Bindings& b, const TextConfig& c, const std::string& p,
                                       bool inject, std::size_t rank) {
    const std::uint64_t width = std::uint64_t(c.hc_count) * c.hidden_size;
    HyperConnectionWeights out;
    out.norm = replica(b, p + "norm", {width}, {}, QType::BF16, rank);
    out.down = replica(b, p + "down", {c.hc_lowrank, width}, {p + "normalized"}, {}, rank);
    out.up   = replica(b, p + "up", {width, c.hc_lowrank}, {p + "low"}, {}, rank);
    if (inject) {
        out.inject = replica(b, p + "inject", {c.hc_count, width}, {p + "normalized"}, {}, rank);
    }
    return out;
}

// A rank's share of a Gated DeltaNet mixer: its key heads with the value heads that read them, the
// rows of every projection and gate of those heads, their convolution channels and the matching
// input columns of the output projection. The gated norm is per head dimension: a replica.
GdnWeights bind_gdn_share(Bindings& b, const TextConfig& c, const std::string& p,
                          const TensorShare& share, std::size_t rank) {
    const std::uint64_t h     = c.hidden_size;
    const std::uint64_t kd    = std::uint64_t(c.linear_num_key_heads) * c.linear_key_head_dim;
    const std::uint64_t vd    = std::uint64_t(c.linear_num_value_heads) * c.linear_value_head_dim;
    const std::uint64_t ratio = c.linear_num_value_heads / c.linear_num_key_heads;
    const Range key   = scaled(share.key_head_begin, share.key_head_end, c.linear_key_head_dim);
    const Range heads = scaled(share.key_head_begin, share.key_head_end, ratio);
    const Range value = scaled(heads.first, heads.second, c.linear_value_head_dim);
    const std::string g = p + "gdn/";
    const std::vector<std::string> input{p + "mixer_input"};
    const std::uint64_t nv = c.linear_num_value_heads;
    GdnWeights out;
    out.query        = b.piece(g + "query", {kd, h}, input, {}, rank, row_piece(key));
    out.key          = b.piece(g + "key", {kd, h}, input, {}, rank, row_piece(key));
    out.value        = b.piece(g + "value", {vd, h}, input, {}, rank, row_piece(value));
    out.z            = b.piece(g + "z", {vd, h}, input, {}, rank, row_piece(value));
    out.a_projection = b.piece(g + "a_projection", {nv, h}, input, {}, rank, row_piece(heads));
    out.b_projection = b.piece(g + "b_projection", {nv, h}, input, {}, rank, row_piece(heads));
    out.a_log   = b.piece(g + "a_log", {nv}, {}, QType::FP32, rank, column_piece({heads}));
    out.dt_bias = b.piece(g + "dt_bias", {nv}, {}, QType::FP32, rank, column_piece({heads}));
    out.convolution =
        b.piece(g + "convolution", {c.linear_conv_kernel_dim, 2 * kd + vd}, {}, QType::BF16, rank,
                column_piece({key, shifted(key, kd), shifted(value, 2 * kd)}));
    out.norm   = rank == 0 ? b.direct(g + "norm", {c.linear_value_head_dim})
                           : replica(b, g + "norm", {c.linear_value_head_dim}, {}, QType::BF16, rank);
    out.output = b.piece(g + "output", {h, vd}, {g + "gated_output"}, {}, rank, column_piece({value}));
    return out;
}

// A rank's share of a sparse-attention mixer: the query and gate rows of the query heads that read
// its KV heads, the key and value rows of those heads and the matching input columns of the output
// projection. The norms and the indexer, which every rank holding heads runs whole: replicas.
SparseAttentionWeights bind_attention_share(Bindings& b, const TextConfig& c, const std::string& p,
                                            const TensorShare& share, std::size_t rank) {
    if (rank == 0 && share.kv_head_begin == 0 && share.kv_head_end == c.num_key_value_heads) {
        return bind_attention(b, c, p);
    }
    const std::uint64_t h     = c.hidden_size;
    const std::uint64_t q     = std::uint64_t(c.num_attention_heads) * c.head_dim;
    const std::uint64_t kv    = std::uint64_t(c.num_key_value_heads) * c.head_dim;
    const std::uint64_t group = c.num_attention_heads / c.num_key_value_heads;
    const Range query = scaled(share.kv_head_begin * group, share.kv_head_end * group, c.head_dim);
    const Range heads = scaled(share.kv_head_begin, share.kv_head_end, c.head_dim);
    const std::string a = p + "attention/", i = p + "indexer/";
    const std::vector<std::string> input{p + "mixer_input"};
    const std::uint64_t iq = std::uint64_t(c.indexer_n_heads) * c.indexer_head_dim;
    const auto whole = [&](const std::string& name, const Shape& shape,
                           const std::vector<std::string>& inputs,
                           std::optional<QType> format) {
        if (rank != 0) { return replica(b, name, shape, inputs, format, rank); }
        return shape.size() == 1 ? b.direct(name, shape) : b.parameter(name, shape, inputs);
    };
    SparseAttentionWeights out;
    out.query      = b.piece(a + "query", {q, h}, input, {}, rank, row_piece(query));
    out.gate       = b.piece(a + "gate", {q, h}, input, {}, rank, row_piece(query));
    out.key        = b.piece(a + "key", {kv, h}, input, {}, rank, row_piece(heads));
    out.value      = b.piece(a + "value", {kv, h}, input, {}, rank, row_piece(heads));
    out.output     = b.piece(a + "output", {h, q}, {a + "gated_output"}, {}, rank,
                             column_piece({query}));
    out.query_norm = whole(a + "query_norm", {c.head_dim}, {}, QType::BF16);
    out.key_norm   = whole(a + "key_norm", {c.head_dim}, {}, QType::BF16);
    out.index_query      = whole(i + "query", {iq, h}, input, {});
    out.index_key        = whole(i + "key", {c.indexer_head_dim, h}, input, {});
    out.index_query_norm = whole(i + "query_norm", {c.indexer_head_dim}, {}, QType::BF16);
    out.index_key_norm   = whole(i + "key_norm", {c.indexer_head_dim}, {}, QType::BF16);
    return out;
}

PleWeights bind_ple_replica(Bindings& b, const TextConfig& c, const std::string& p,
                            std::size_t rank) {
    const std::uint64_t width = std::uint64_t(c.hc_count) * c.hidden_size;
    const std::string q = p + "ple/", input = p + "ple/embedding";
    PleWeights out;
    out.key         = replica(b, q + "key", {width, c.ple_embed_dim}, {input}, {}, rank);
    out.value       = replica(b, q + "value", {c.hidden_size, c.ple_embed_dim}, {input}, {}, rank);
    out.norm_key    = replica(b, q + "norm_key", {width}, {}, QType::BF16, rank);
    out.norm_query  = replica(b, q + "norm_query", {width}, {}, QType::BF16, rank);
    out.norm_conv   = replica(b, q + "norm_conv", {width}, {}, QType::BF16, rank);
    out.convolution =
        replica(b, q + "convolution", {width, c.ple_conv_kernel_size}, {}, QType::BF16, rank);
    return out;
}

// The positions in (0, count) where every layer `allowed` lists cuts whole stored runs.
template <class Allowed>
std::vector<std::uint64_t> common_ends(std::uint64_t count, const TextConfig& c, MixerKind kind,
                                       const Allowed& allowed) {
    std::vector<std::uint64_t> out;
    for (std::uint64_t position = 1; position < count; ++position) {
        bool ok = true;
        for (std::uint32_t layer = 0; ok && layer < c.num_hidden_layers; ++layer) {
            if (c.layer_types[layer] == kind) {
                ok = allowed("text/layers/" + std::to_string(layer) + "/", position);
            }
        }
        if (ok) { out.push_back(position); }
    }
    return out;
}

// The shared expert's middle rows each rank takes so that every rank's expert work per token, its
// routed experts' share of num_experts_per_tok * moe_intermediate_size rows and its run of the
// shared expert, is as even as the rows allow: filled from the rank with the least routed work up
// (water filling). A rank whose routed work alone passes the level takes none, which split_shares
// rounds to one allowed run.
std::vector<double> shared_runs(const TextConfig& c, std::span<const std::uint64_t> experts) {
    const std::size_t ranks = experts.size() - 1;
    const double width      = c.shared_expert_intermediate_size;
    const double routed     = double(c.num_experts_per_tok) * c.moe_intermediate_size;
    std::vector<double> work(ranks);
    for (std::size_t r = 0; r < ranks; ++r) {
        work[r] = routed * double(experts[r + 1] - experts[r]) / double(c.num_experts);
    }
    std::vector<bool> open(ranks, true);
    double level = 0.0;
    for (;;) {
        double sum         = width;
        std::size_t filled = 0;
        for (std::size_t r = 0; r < ranks; ++r) {
            if (open[r]) {
                sum += work[r];
                ++filled;
            }
        }
        level       = sum / double(filled);
        bool closed = false;
        for (std::size_t r = 0; r < ranks; ++r) {
            if (open[r] && work[r] > level) {
                open[r] = false;
                closed  = true;
            }
        }
        if (!closed) { break; }
    }
    std::vector<double> out(ranks);
    for (std::size_t r = 0; r < ranks; ++r) {
        out[r] = std::max(open[r] ? level - work[r] : 0.0, width * 1e-6);
    }
    return out;
}

// Each rank's runs of every layer, by `split`: its Gated DeltaNet key heads and routed experts, and
// its sparse-attention KV heads, which go to as many of the first ranks as the two heads allow, in
// proportion to their weights. A run ends only where every layer's projections cut whole stored
// runs: the key, value and gate rows, the convolution channels and the output projection's columns.
std::vector<TensorShare> plan_tensor_shares(Bindings& b, const TextConfig& c,
                                            std::span<const double> split) {
    const std::uint64_t h     = c.hidden_size;
    const std::uint64_t kd    = std::uint64_t(c.linear_num_key_heads) * c.linear_key_head_dim;
    const std::uint64_t vd    = std::uint64_t(c.linear_num_value_heads) * c.linear_value_head_dim;
    const std::uint64_t ratio = c.linear_num_value_heads / c.linear_num_key_heads;
    const std::uint64_t q     = std::uint64_t(c.num_attention_heads) * c.head_dim;
    const std::uint64_t kv    = std::uint64_t(c.num_key_value_heads) * c.head_dim;
    const std::uint64_t group = c.num_attention_heads / c.num_key_value_heads;
    const std::uint64_t nv    = c.linear_num_value_heads;
    const auto key_ends = common_ends(
        c.linear_num_key_heads, c, MixerKind::GatedDeltaNet,
        [&](const std::string& p, std::uint64_t heads) {
            const std::string g  = p + "gdn/";
            const std::vector<std::string> input{p + "mixer_input"};
            const std::uint64_t k = heads * c.linear_key_head_dim;
            const std::uint64_t v = heads * ratio * c.linear_value_head_dim;
            return b.row_ends(g + "query", {kd, h})[k] && b.row_ends(g + "key", {kd, h})[k] &&
                   b.row_ends(g + "value", {vd, h})[v] && b.row_ends(g + "z", {vd, h})[v] &&
                   b.row_ends(g + "a_projection", {nv, h})[heads * ratio] &&
                   b.row_ends(g + "b_projection", {nv, h})[heads * ratio] &&
                   b.column_ends(g + "output", {h, vd}, {g + "gated_output"})[v];
        });
    const auto kv_ends = common_ends(
        c.num_key_value_heads, c, MixerKind::SparseAttention,
        [&](const std::string& p, std::uint64_t heads) {
            const std::string a = p + "attention/";
            const std::uint64_t rows = heads * group * c.head_dim;
            return b.row_ends(a + "query", {q, h})[rows] && b.row_ends(a + "gate", {q, h})[rows] &&
                   b.row_ends(a + "key", {kv, h})[heads * c.head_dim] &&
                   b.row_ends(a + "value", {kv, h})[heads * c.head_dim] &&
                   b.column_ends(a + "output", {h, q}, {a + "gated_output"})[rows];
        });
    // The shared expert's middle ends where every layer's gate and up rows and down columns cut
    // whole stored runs, at whole 128-row tiles of the expert matrix kernel where every rank gets
    // one. Its runs even out each rank's expert work, counted in middle rows a token runs: a rank's
    // routed experts give it num_experts_per_tok * moe_intermediate_size rows per token in
    // proportion to their count, so a rank with fewer experts (rank 0 when its room caps it) takes
    // more of the shared expert (shared_runs). The runs follow the expert ranges, not the split's
    // weights: each rank rounds its run's dot of the down rows on its own, so the sum depends on
    // the runs, and weights measured at startup move from one start to the next (two ranks at an
    // even split hold two and a half of its five tiles, which rounded to three or two by the
    // measurement's noise).
    const std::uint64_t width = c.shared_expert_intermediate_size;
    const std::uint64_t tile  = width / 128 >= split.size() ? 128 : 1;
    std::vector<bool> shared_allowed(width + 1, true);
    for (std::uint32_t layer = 0; layer < c.num_hidden_layers; ++layer) {
        const std::string m  = "text/layers/" + std::to_string(layer) + "/moe/";
        const auto gate_rows = b.row_ends(m + "shared/gate", {width, h});
        const auto up_rows   = b.row_ends(m + "shared/up", {width, h});
        const auto down      = b.column_ends(m + "shared/down", {h, width}, {m + "shared/product"});
        for (std::uint64_t at = 0; at <= width; ++at) {
            shared_allowed[at] = shared_allowed[at] && gate_rows[at] && up_rows[at] && down[at];
        }
    }
    std::vector<std::uint64_t> shared_ends;
    for (std::uint64_t at = tile; at < width; at += tile) {
        if (shared_allowed[at]) { shared_ends.push_back(at); }
    }
    const std::size_t ranks    = split.size();
    // A latent MoE's ranks hold whole groups, which each layer cuts by its labels so that the
    // experts follow the split (latent_group_ends); the shared runs follow that share.
    const auto experts         = expert_ends(c.num_experts, split);
    const auto shared          = split_shares(width, shared_ends, shared_runs(c, experts));
    const auto key_heads       = split_shares(c.linear_num_key_heads, key_ends, split);
    const std::size_t kv_ranks = std::min(ranks, kv_ends.size() + 1);
    const auto kv_heads        = split_shares(c.num_key_value_heads, kv_ends, split.first(kv_ranks));
    std::vector<TensorShare> out(ranks);
    for (std::size_t r = 0; r < ranks; ++r) {
        out[r].key_head_begin = static_cast<std::uint32_t>(key_heads[r]);
        out[r].key_head_end   = static_cast<std::uint32_t>(key_heads[r + 1]);
        out[r].expert_begin   = static_cast<std::uint32_t>(experts[r]);
        out[r].expert_end     = static_cast<std::uint32_t>(experts[r + 1]);
        out[r].shared_begin   = static_cast<std::uint32_t>(shared[r]);
        out[r].shared_end     = static_cast<std::uint32_t>(shared[r + 1]);
        if (r < kv_ranks) {
            out[r].kv_head_begin = static_cast<std::uint32_t>(kv_heads[r]);
            out[r].kv_head_end   = static_cast<std::uint32_t>(kv_heads[r + 1]);
        } else {
            out[r].kv_head_begin = out[r].kv_head_end = c.num_key_value_heads;
        }
    }
    return out;
}

// Where a parameter's rows sit in the artifact's files, without demanding its object.
ExpertLocation locate(const artifact::Reader& reader, const std::string& name, std::uint64_t rows,
                      std::uint64_t k) {
    const auto& bindings = reader.directory().bindings;
    const auto found     = bindings.find(name);
    if (found == bindings.end() || found->second.parts.size() != 1) {
        throw artifact::ArtifactError(name + ": a disk-resident expert must be one stored region");
    }
    const auto& part     = found->second.parts.front();
    const auto& geometry = reader.geometry(part.object);
    if (!is_gguf(geometry.format) || geometry.shape.size() != 2 || geometry.shape[1] != k ||
        part.end - part.begin != rows * k || part.begin % k != 0) {
        throw artifact::ArtifactError(name + ": disk-resident experts must be GGUF block rows");
    }
    const auto block              = gguf_block_shape(geometry.format);
    const std::uint64_t row_bytes = k / block.elements * block.bytes;
    const std::uint64_t logical =
        reader.directory().tensor(part.object).offset + part.begin / k * row_bytes;
    const auto segments = reader.segments(logical, rows * row_bytes);
    if (segments.empty() || segments.size() > 2) {
        throw artifact::ArtifactError(name + ": a disk-resident expert spans more than two files");
    }
    ExpertLocation out{.runs      = {},
                       .bytes     = rows * row_bytes,
                       .format    = geometry.format,
                       .row_bytes = static_cast<std::int64_t>(row_bytes),
                       .rows      = static_cast<std::int32_t>(rows)};
    for (std::size_t i = 0; i < segments.size(); ++i) {
        out.runs[i] = {segments[i].file_index, segments[i].file_offset, segments[i].bytes};
    }
    return out;
}

// Zero bytes after each down bank; see bind_moe.
constexpr std::uint64_t kExpertTail = 256;

// Whether two parameters are stored in one object.
bool one_object(const artifact::Reader& reader, const std::string& a, const std::string& b) {
    const auto& bindings = reader.directory().bindings;
    const auto first = bindings.find(a), second = bindings.find(b);
    if (first == bindings.end() || second == bindings.end()) {
        throw artifact::ArtifactError("missing logical parameter " +
                                      (first == bindings.end() ? a : b));
    }
    return !first->second.parts.empty() && !second->second.parts.empty() &&
           first->second.parts.front().object.index == second->second.parts.front().object.index;
}

// Expert parallelism: rank r's experts [ends[r], ends[r + 1]) of one MoE layer as one device
// selection per stored bank (a [gate; up] bank, or a gate and an up bank, and a down bank), each
// expert's projections views into it. Fills every expert's gate, up and down.
void bind_expert_shares(Bindings& b, const TextConfig& c, const std::string& m,
                        const std::string& input, std::span<const std::uint64_t> ends,
                        MoeWeights& out) {
    const std::uint64_t h = c.hidden_size, w = c.moe_intermediate_size;
    out.gate.resize(c.num_experts);
    out.up.resize(c.num_experts);
    out.down.resize(c.num_experts);
    const auto expert = [&](std::uint64_t e) { return m + "experts/" + std::to_string(e) + "/"; };
    const bool joint  = one_object(b.binder.reader(), expert(0) + "gate", expert(0) + "up");
    for (std::size_t rank = 0; rank + 1 < ends.size(); ++rank) {
        std::vector<Bindings::JointPart> gate_up, gates, ups, downs;
        for (std::uint64_t e = ends[rank]; e < ends[rank + 1]; ++e) {
            const std::string x = expert(e);
            Bindings::JointPart gate{x + "gate", {w, h}, {input}};
            Bindings::JointPart up{x + "up", {w, h}, {input}};
            if (joint) {
                gate_up.push_back(std::move(gate));
                gate_up.push_back(std::move(up));
            } else {
                gates.push_back(std::move(gate));
                ups.push_back(std::move(up));
            }
            downs.push_back({x + "down", {h, w}, {x + "product"}});
        }
        if (joint) {
            const auto ids = b.joint_piece(gate_up, rank);
            for (std::size_t i = 0; i < ids.size() / 2; ++i) {
                out.gate[ends[rank] + i] = ids[2 * i];
                out.up[ends[rank] + i]   = ids[2 * i + 1];
            }
        } else {
            const auto gate_ids = b.joint_piece(gates, rank);
            const auto up_ids   = b.joint_piece(ups, rank);
            std::copy(gate_ids.begin(), gate_ids.end(), out.gate.begin() + ends[rank]);
            std::copy(up_ids.begin(), up_ids.end(), out.up.begin() + ends[rank]);
        }
        const auto down_ids = b.joint_piece(downs, rank);
        std::copy(down_ids.begin(), down_ids.end(), out.down.begin() + ends[rank]);
    }
}

// A latent MoE layer's expert group labels (moe/latent/labels), read before anything is bound:
// under tensor parallelism each expert goes to the rank that holds its group.
std::vector<std::int32_t> read_labels(const artifact::Reader& reader, const std::string& name,
                                      std::uint32_t experts, std::uint32_t groups) {
    const auto& bindings = reader.directory().bindings;
    const auto found     = bindings.find(name);
    if (found == bindings.end() || found->second.parts.size() != 1) {
        throw artifact::ArtifactError(name + ": the labels must be one stored region");
    }
    const auto& part = found->second.parts.front();
    if (reader.geometry(part.object).format != QType::INT32 || part.end - part.begin != experts) {
        throw artifact::ArtifactError(name + ": the labels must be one int32 per expert");
    }
    const auto bytes = reader.read_range(
        reader.directory().tensor(part.object).offset + part.begin * 4, std::uint64_t(experts) * 4);
    std::vector<std::int32_t> out(experts);
    std::memcpy(out.data(), bytes.data(), out.size() * 4);
    for (const std::int32_t label : out) {
        if (label < 0 || std::uint32_t(label) >= groups) {
            throw artifact::ArtifactError(name + ": a label names no group");
        }
    }
    return out;
}

// Tensor parallelism over a latent MoE layer: the ends of each rank's run of whole groups, from 0
// to `groups`. Each cut falls where the experts the labels assign to the groups before it come
// nearest the split's share of them, every rank keeping at least one group: the groups differ in
// size (8 to 200 experts in Edge0's layers), so an even count of groups would give one rank three
// quarters of a layer's experts. Every cut gives the same sums (each group's part is exact).
std::vector<std::uint32_t> latent_group_ends(std::span<const std::int32_t> labels,
                                             std::uint32_t groups, std::span<const double> split) {
    const std::size_t ranks = split.size();
    if (ranks == 0 || ranks > groups) {
        throw std::invalid_argument("tensor parallelism over a latent MoE takes at most one device "
                                    "per expert group");
    }
    std::vector<std::uint64_t> before(groups + 1, 0);
    for (const std::int32_t label : labels) { ++before[std::size_t(label) + 1]; }
    for (std::uint32_t g = 0; g < groups; ++g) { before[g + 1] += before[g]; }
    const double total = std::accumulate(split.begin(), split.end(), 0.0);
    std::vector<std::uint32_t> ends{0};
    double share = 0;
    for (std::size_t r = 0; r + 1 < ranks; ++r) {
        share += split[r];
        const double target       = double(labels.size()) * share / total;
        const std::uint32_t low   = ends.back() + 1;
        const std::uint32_t high  = groups - std::uint32_t(ranks - 1 - r);
        std::uint32_t best        = low;
        for (std::uint32_t at = low + 1; at <= high; ++at) {
            if (std::abs(double(before[at]) - target) < std::abs(double(before[best]) - target)) {
                best = at;
            }
        }
        ends.push_back(best);
    }
    ends.push_back(groups);
    return ends;
}

// The rank ends of layer `p`'s groups under tensor parallelism with `split` (latent_group_ends).
std::vector<std::uint32_t> latent_ends(Bindings& b, const TextConfig& c, const std::string& p,
                                       std::span<const double> split) {
    const auto labels = read_labels(b.binder.reader(), p + "moe/latent/labels", c.num_experts,
                                    c.latent_moe->groups);
    return latent_group_ends(labels, c.latent_moe->groups, split);
}

// Tensor parallelism over a latent MoE: rank `rank`'s groups [begin, end) of layer `p`, their
// encoder and decoder rows, and replicas of the labels and multipliers.
LatentWeights bind_latent_share(Bindings& b, const TextConfig& c, const std::string& p,
                                std::uint32_t begin, std::uint32_t end, std::size_t rank) {
    const auto& latent        = *c.latent_moe;
    const std::uint64_t h     = c.hidden_size, width = latent.width, groups = latent.groups;
    const std::string m       = p + "moe/";
    const auto whole          = [&](const std::string& name, const Shape& shape, QType format) {
        return rank == 0 ? b.direct(name, shape, format)
                         : replica(b, name, shape, {}, format, rank);
    };
    LatentWeights out;
    out.group_begin = begin;
    out.group_end   = end;
    out.encode = b.piece(m + "latent/encode", {groups * width, h}, {p + "ffn_input"}, {}, rank,
                         row_piece(scaled(begin, end, width)));
    out.decode = b.piece(m + "latent/decode", {groups * h, width}, {m + "latent/sum"}, {}, rank,
                         row_piece(scaled(begin, end, h)));
    out.labels = whole(m + "latent/labels", {c.num_experts}, QType::INT32);
    if (latent.expert_scales) {
        out.expert_scales = whole(m + "latent/expert_scales", {c.num_experts, 3}, QType::FP32);
    }
    if (latent.column_scale) {
        out.column_scale = whole(m + "latent/column_scale", {width}, QType::FP32);
    }
    return out;
}

// Tensor parallelism over a latent MoE: every expert of layer `p` on the rank that holds its group
// (`ends`, latent_group_ends), with the zeros the expert matrix kernel reads past each down.
void bind_latent_experts(Bindings& b, const TextConfig& c, const std::string& p,
                         std::span<const std::uint32_t> ends, MoeWeights& out) {
    const std::uint64_t width = c.latent_moe->width, w = c.moe_intermediate_size;
    const std::string m       = p + "moe/";
    const auto labels         = read_labels(b.binder.reader(), m + "latent/labels", c.num_experts,
                                            c.latent_moe->groups);
    out.gate.resize(c.num_experts);
    out.up.resize(c.num_experts);
    out.down.resize(c.num_experts);
    for (std::uint32_t e = 0; e < c.num_experts; ++e) {
        const auto after    = std::upper_bound(ends.begin(), ends.end(), std::uint32_t(labels[e]));
        const auto rank     = static_cast<std::size_t>(after - ends.begin()) - 1;
        const std::string x = m + "experts/" + std::to_string(e) + "/";
        out.gate[e] = b.piece(x + "gate", {w, width}, {m + "latent/input"}, {}, rank,
                              row_piece({0, w}));
        out.up[e]   = b.piece(x + "up", {w, width}, {m + "latent/input"}, {}, rank, row_piece({0, w}));
        out.down[e] = b.piece(x + "down", {width, w}, {x + "product"}, {}, rank,
                              row_piece({0, width}));
        for (const auto& part : b.at(out.down[e]).reference.binding.parts) {
            if (is_gguf(b.binder.geometry(part.object).format)) {
                b.binder.device_tail(part.object, kExpertTail);
            }
        }
    }
}

// `split` divides the routed experts over the ranks (expert parallelism; device residency only),
// a latent MoE's by whole groups (latent_group_ends).
MoeWeights bind_moe(Bindings& b, const TextConfig& c, const std::string& p,
                    ExpertResidency residency, std::span<const double> split = {}) {
    const std::uint64_t h = c.hidden_size, w = c.moe_intermediate_size;
    const std::string m = p + "moe/", input = p + "ffn_input";
    MoeWeights out;
    out.router       = b.parameter(m + "router", {c.num_experts, h}, {input});
    out.shared_score = b.parameter(m + "shared_score", {1, h}, {input});
    // A latent MoE's experts read and write its latent width: gate and up [w, width] from the
    // group's encoded input, down [width, w] into the group's sum.
    std::uint64_t k = h;
    std::string expert_input = input;
    if (c.latent_moe && !split.empty()) {
        const auto ends = latent_ends(b, c, p, split);
        out.latent      = bind_latent_share(b, c, p, ends[0], ends[1], 0);
        bind_latent_experts(b, c, p, ends, out);
    } else if (c.latent_moe) {
        const auto& latent = *c.latent_moe;
        const std::uint64_t width = latent.width, groups = latent.groups;
        LatentWeights l;
        l.encode = b.parameter(m + "latent/encode", {groups * width, h}, {input});
        l.decode = b.parameter(m + "latent/decode", {groups * h, width}, {m + "latent/sum"});
        l.labels = b.direct(m + "latent/labels", {c.num_experts}, QType::INT32);
        if (latent.expert_scales) {
            l.expert_scales = b.direct(m + "latent/expert_scales", {c.num_experts, 3}, QType::FP32);
        }
        if (latent.column_scale) {
            l.column_scale = b.direct(m + "latent/column_scale", {width}, QType::FP32);
        }
        out.latent   = l;
        k            = width;
        expert_input = m + "latent/input";
    }
    if (!split.empty() && !c.latent_moe) {
        bind_expert_shares(b, c, m, input, expert_ends(c.num_experts, split), out);
    }
    for (std::uint32_t e = 0; e < c.num_experts && split.empty(); ++e) {
        const std::string x = m + "experts/" + std::to_string(e) + "/";
        if (residency == ExpertResidency::Disk) {
            const auto& reader = b.binder.reader();
            out.located_gate.push_back(locate(reader, x + "gate", w, k));
            out.located_up.push_back(locate(reader, x + "up", w, k));
            out.located_down.push_back(locate(reader, x + "down", k, w));
            continue;
        }
        const auto storage = [&](const char* role) {
            if (residency != ExpertResidency::Host) { return artifact::Residency::Device; }
            const auto& reader = b.binder.reader();
            const auto found = reader.directory().bindings.find(x + role);
            if (found == reader.directory().bindings.end()) {
                throw artifact::ArtifactError("missing logical parameter " + x + role);
            }
            const auto& parts = found->second.parts;
            const bool blocks = std::all_of(parts.begin(), parts.end(), [&](const auto& part) {
                return is_gguf(reader.geometry(part.object).format);
            });
            if (blocks) { return artifact::Residency::Pinned; }
            // Native banks are registered independently, so each layer's objects stay within
            // one registration. Windows keeps pageable banks and uses bounded copy staging.
#if defined(__linux__)
            return artifact::Residency::Registered;
#else
            return artifact::Residency::Host;
#endif
        };
        out.gate.push_back(b.parameter(x + "gate", {w, k}, {expert_input}, {}, storage("gate")));
        out.up.push_back(b.parameter(x + "up", {w, k}, {expert_input}, {}, storage("up")));
        out.down.push_back(b.parameter(x + "down", {k, w}, {x + "product"}, {}, storage("down")));
    }
    // Under expert parallelism each rank binds its run of the shared expert (bind_shared_run).
    std::vector<WeightId> downs;
    if (split.empty()) {
        const std::uint64_t s = c.shared_expert_intermediate_size;
        out.shared_gate       = b.parameter(m + "shared/gate", {s, h}, {input});
        out.shared_up         = b.parameter(m + "shared/up", {s, h}, {input});
        out.shared_down       = b.parameter(m + "shared/down", {h, s}, {m + "shared/product"});
        downs.push_back(out.shared_down);
    }
    // The expert matrix kernel reads a down row's 640 values in three 256-value steps, past the end
    // of each bank's last row: zeros follow every down bank in device memory.
    if (residency == ExpertResidency::Device) {
        downs.insert(downs.end(), out.down.begin(), out.down.end());
    }
    for (const WeightId id : downs) {
        for (const auto& part : b.at(id).reference.binding.parts) {
            if (is_gguf(b.binder.geometry(part.object).format)) {
                b.binder.device_tail(part.object, kExpertTail);
            }
        }
    }
    return out;
}

// Tensor parallelism: rank `rank`'s run of layer `p`'s shared expert (TensorShare): those rows of
// its gate and up, those values of every down row, and the zeros the expert matrix kernel reads
// past the down piece's last row.
void bind_shared_run(Bindings& b, const TextConfig& c, const std::string& p,
                     const TensorShare& share, std::size_t rank, MoeWeights& out) {
    const std::uint64_t h = c.hidden_size, s = c.shared_expert_intermediate_size;
    const std::string m = p + "moe/", input = p + "ffn_input";
    const Range run{share.shared_begin, share.shared_end};
    out.shared_gate = b.piece(m + "shared/gate", {s, h}, {input}, {}, rank, row_piece(run));
    out.shared_up   = b.piece(m + "shared/up", {s, h}, {input}, {}, rank, row_piece(run));
    out.shared_down =
        b.piece(m + "shared/down", {h, s}, {m + "shared/product"}, {}, rank, column_piece({run}));
    for (const auto& part : b.at(out.shared_down).reference.binding.parts) {
        if (is_gguf(b.binder.geometry(part.object).format)) {
            b.binder.device_tail(part.object, kExpertTail);
        }
    }
}

PleWeights bind_ple(Bindings& b, const TextConfig& c, const std::string& p) {
    const std::uint64_t width = std::uint64_t(c.hc_count) * c.hidden_size;
    const std::string q = p + "ple/", input = p + "ple/embedding";
    PleWeights out;
    out.key         = b.parameter(q + "key", {width, c.ple_embed_dim}, {input});
    out.value       = b.parameter(q + "value", {c.hidden_size, c.ple_embed_dim}, {input});
    out.norm_key    = b.direct(q + "norm_key", {width});
    out.norm_query  = b.direct(q + "norm_query", {width});
    out.norm_conv   = b.direct(q + "norm_conv", {width});
    out.convolution = b.direct(q + "convolution", {width, c.ple_conv_kernel_size});
    return out;
}

MtpWeights bind_mtp(Bindings& b, const TextConfig& c, const MtpConfig& mtp,
                    ExpertResidency residency, std::span<const double> split) {
    const std::uint64_t h = c.hidden_size, width = std::uint64_t(c.hc_count) * h;
    MtpWeights out;
    out.embedding_norm  = b.direct("mtp/embedding_norm", {h});
    out.hidden_norm     = b.direct("mtp/hidden_norm", {width});
    out.fc_embedding    = b.parameter("mtp/fc_embedding", {h, h}, {"mtp/embedding_normalized"});
    out.fc_hidden       = b.parameter("mtp/fc_hidden", {h, h}, {"mtp/hidden_normalized"});
    const std::string p = "mtp/layer/";
    out.layer.attn_hc   = bind_hc(b, c, p + "attn_hc/", true);
    out.layer.mlp_hc    = bind_hc(b, c, p + "mlp_hc/", true);
    out.layer.mixer     = bind_attention(b, c, p);
    TextConfig experts  = c;
    experts.num_experts = mtp.num_experts;
    // The MTP block keeps the base model's experts.
    experts.latent_moe.reset();
    out.layer.moe       = bind_moe(b, experts, p, residency, split);
    out.final_mixer     = bind_hc(b, c, "mtp/final_mixer/", false);
    return out;
}

// Every parameter a layer owns, which moves with it to its stage.
std::vector<WeightId> layer_weights(const LayerWeights& layer, bool with_experts) {
    std::vector<WeightId> out;
    for (const HyperConnectionWeights* hc : {&layer.attn_hc, &layer.mlp_hc}) {
        out.insert(out.end(), {hc->norm, hc->down, hc->up});
        if (hc->inject) { out.push_back(*hc->inject); }
    }
    if (const auto* gdn = std::get_if<GdnWeights>(&layer.mixer)) {
        out.insert(out.end(),
                   {gdn->query, gdn->key, gdn->value, gdn->z, gdn->a_projection, gdn->b_projection,
                    gdn->a_log, gdn->dt_bias, gdn->convolution, gdn->norm, gdn->output});
    } else if (const auto* a = std::get_if<SparseAttentionWeights>(&layer.mixer)) {
        out.insert(out.end(), {a->query, a->gate, a->key, a->value, a->output, a->query_norm,
                               a->key_norm, a->index_query, a->index_key, a->index_query_norm,
                               a->index_key_norm});
    }
    const MoeWeights& m = layer.moe;
    out.insert(out.end(), {m.router, m.shared_score, m.shared_gate, m.shared_up, m.shared_down});
    if (m.latent) {
        out.insert(out.end(), {m.latent->encode, m.latent->decode, m.latent->labels});
        if (m.latent->expert_scales) { out.push_back(*m.latent->expert_scales); }
        if (m.latent->column_scale) { out.push_back(*m.latent->column_scale); }
    }
    if (with_experts) {
        out.insert(out.end(), m.gate.begin(), m.gate.end());
        out.insert(out.end(), m.up.begin(), m.up.end());
        out.insert(out.end(), m.down.begin(), m.down.end());
    }
    if (layer.ple) {
        const PleWeights& p = *layer.ple;
        out.insert(out.end(),
                   {p.key, p.value, p.norm_key, p.norm_query, p.norm_conv, p.convolution});
    }
    return out;
}

DFlash2Weights bind_dflash2(Bindings& b, const TextConfig& c, const DFlash2Config& d) {
    const auto h = std::uint64_t(c.hidden_size);
    const auto q = std::uint64_t(d.attention.query_width());
    const auto k = std::uint64_t(d.attention.key_width());
    const auto i = std::uint64_t(d.intermediate_size);
    const auto& extra   = d.dflash2.value();
    const auto rows     = 2ULL * extra.conv_kernel_size * (h / extra.conv_group_size);
    const std::string n = "dflash2/";
    DFlash2Weights out;
    out.feature_projection =
        b.parameter(n + "feature_projection", {h, d.target_layer_ids.size() * h},
                    {n + "target_features"});
    out.context_norm = b.direct(n + "context_norm", {h});
    out.final_norm   = b.direct(n + "final_norm", {h});
    for (std::uint32_t l = 0; l < d.num_hidden_layers; ++l) {
        const std::string p = n + "layers/" + std::to_string(l) + "/";
        DFlash2LayerWeights layer;
        layer.input_norm          = b.direct(p + "input_norm", {h});
        layer.post_attention_norm = b.direct(p + "post_attention_norm", {h});
        layer.query = b.parameter(p + "attention/query", {q, h}, {p + "query_projection_input"});
        layer.key   = b.parameter(p + "attention/key", {k, h}, {p + "query_projection_input"});
        layer.value = b.parameter(p + "attention/value", {k, h}, {p + "query_projection_input"});
        layer.context_key =
            b.parameter(p + "attention/context_key", {k, h}, {n + "context_input"});
        layer.context_value =
            b.parameter(p + "attention/context_value", {k, h}, {n + "context_input"});
        layer.query_norm = b.direct(p + "attention/query_norm", {d.attention.head_dim});
        layer.key_norm   = b.direct(p + "attention/key_norm", {d.attention.head_dim});
        layer.output = b.parameter(p + "attention/output", {h, q}, {p + "attention_output"});
        layer.gate   = b.parameter(p + "mlp/gate", {i, h}, {p + "mlp_input"});
        layer.up     = b.parameter(p + "mlp/up", {i, h}, {p + "mlp_input"});
        layer.down   = b.parameter(p + "mlp/down", {h, i}, {p + "mlp_product"});
        layer.attention_base = b.direct(p + "attention_conv/base_kernel", {2, 2, h});
        layer.attention_projection = b.parameter(p + "attention_conv/kernel_projection", {rows, h},
                                                 {p + "attention_conv_input"});
        layer.mlp_base       = b.direct(p + "mlp_conv/base_kernel", {2, 2, h});
        layer.mlp_projection = b.parameter(p + "mlp_conv/kernel_projection", {rows, h},
                                           {p + "mlp_conv_input"});
        out.layers.push_back(layer);
    }
    out.selector_projection = b.parameter(n + "candidate_selector/hidden_projection",
                                          {extra.selector_rank, h}, {n + "final_hidden"});
    // The codebooks are rows the selector gathers, BF16 or as a GGUF drafter stores them.
    out.predecessor_codebook = b.parameter(n + "candidate_selector/predecessor_codebook",
                                           {c.vocab_size, extra.selector_rank});
    out.successor_codebook   = b.parameter(n + "candidate_selector/successor_codebook",
                                           {c.vocab_size, extra.selector_rank});
    return out;
}

std::vector<WeightId> dflash2_parameters(const DFlash2Weights& d) {
    std::vector<WeightId> out{d.feature_projection,  d.context_norm,         d.final_norm,
                              d.selector_projection, d.predecessor_codebook, d.successor_codebook};
    for (const auto& l : d.layers) {
        out.insert(out.end(),
                   {l.input_norm, l.post_attention_norm, l.query, l.key, l.value, l.context_key,
                    l.context_value, l.query_norm, l.key_norm, l.output, l.gate, l.up, l.down,
                    l.attention_base, l.attention_projection, l.mlp_base, l.mlp_projection});
    }
    return out;
}

std::vector<WeightId> mtp_parameters(const MtpWeights& mtp, bool with_experts) {
    std::vector<WeightId> out = layer_weights(mtp.layer, with_experts);
    out.insert(out.end(), {mtp.embedding_norm, mtp.hidden_norm, mtp.fc_embedding, mtp.fc_hidden,
                           mtp.final_mixer.norm, mtp.final_mixer.down, mtp.final_mixer.up});
    return out;
}

// Stored bytes behind `ids`, each object counted once.
std::uint64_t stored_bytes(const Bindings& b, std::span<const WeightId> ids,
                           std::set<std::size_t>& seen) {
    std::uint64_t bytes = 0;
    for (const WeightId id : ids) {
        for (const auto& part : b.at(id).reference.binding.parts) {
            if (seen.insert(part.object.index).second) {
                bytes += artifact::object_bytes(b.binder.reader().directory().object(part.object));
            }
        }
    }
    return bytes;
}

// Contiguous layer ranges whose stored bytes are as even as the layer granularity allows; the
// first stage also carries the embedding and the last the final mixer and the head.
StagePlan balanced_stages(std::span<const std::uint64_t> layer_bytes, std::uint64_t first_extra,
                          std::uint64_t last_extra, std::size_t ranks) {
    const auto layers = static_cast<std::uint32_t>(layer_bytes.size());
    std::vector<std::uint64_t> prefix(layers + 1, 0);
    for (std::uint32_t i = 0; i < layers; ++i) { prefix[i + 1] = prefix[i] + layer_bytes[i]; }
    const double total  = double(prefix[layers] + first_extra + last_extra);
    const double target = total / double(ranks);
    std::vector<std::uint32_t> boundaries;
    std::uint32_t begin = 0;
    for (std::size_t stage = 0; stage + 1 < ranks; ++stage) {
        // The end whose cumulative bytes come closest to (stage + 1) shares, leaving at least one
        // layer for each later stage.
        const std::uint32_t last_end = layers - static_cast<std::uint32_t>(ranks - stage - 1);
        std::uint32_t best           = begin + 1;
        double best_gap              = 1e300;
        for (std::uint32_t end = begin + 1; end <= last_end; ++end) {
            const double cumulative = double(prefix[end] + first_extra);
            const double gap        = std::abs(cumulative - target * double(stage + 1));
            if (gap < best_gap) {
                best_gap = gap;
                best     = end;
            }
        }
        boundaries.push_back(best);
        begin = best;
    }
    boundaries.push_back(layers);
    return StagePlan(layers, std::move(boundaries));
}

} // namespace

Model::Model(TextConfig config, LoadOptions options, TextWeights weights, StagePlan stages,
             std::vector<BoundWeight> bound, FrontendResources resources, InstanceInfo info,
             artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), options_(std::move(options)),
      weights_(std::move(weights)), stages_(std::move(stages)), bound_(std::move(bound)),
      resources_(std::move(resources)), info_(std::move(info)) {}

Model::~Model() = default;

std::size_t Model::attention_ranks() const noexcept {
    std::size_t out = 0;
    while (out < shares_.size() && shares_[out].kv_head_end > shares_[out].kv_head_begin) { ++out; }
    return out;
}

const LayerWeights& Model::layer_weights(std::size_t rank, std::size_t layer) const {
    return rank == 0 ? weights_.layers.at(layer) : weights_.rank_layers.at(rank - 1).at(layer);
}

WeightId Model::embedding(std::size_t rank) const {
    return rank == 0 ? weights_.token_embedding : weights_.rank_embeddings.at(rank - 1);
}

std::vector<std::uint64_t> expert_ends(std::uint32_t experts, std::span<const double> split) {
    return split_shares(experts, unit_positions(experts, 1), split);
}

std::vector<double> default_tensor_split(const artifact::Reader& reader, const LoadOptions& options,
                                         std::span<const std::uint64_t> free_bytes,
                                         std::span<const double> speed) {
    // A DSpine drafter and its head-write table also sit on rank 0.
    std::vector<std::uint64_t> rank_free(free_bytes.begin(), free_bytes.end());
    if (!options.dspine.empty() && !rank_free.empty()) {
        const std::uint64_t drafter =
            dspine::weight_bytes(options.dspine) +
            dspine::head_write_bytes(dspine::read_config(options.dspine));
        rank_free.front() -= std::min(rank_free.front(), drafter);
    }
    // The MTP block's experts stay whole on rank 0 with the rest of the block.
    return models::default_expert_split(
        reader,
        [&](std::string_view name) {
            return name.starts_with("text/") || (options.mtp && name.starts_with("mtp/")) ||
                   (options.dflash2 && name.starts_with("dflash2/")) ||
                   (options.vision && name.starts_with("vision/"));
        },
        [](std::string_view name) {
            return name.starts_with("text/") && name.find("/moe/experts/") != std::string_view::npos;
        },
        rank_free, speed);
}

ops::WeightInput Model::input(WeightId id) const {
    const auto& parameter = weight(id);
    if (parameter.uses.size() != 1) {
        throw std::invalid_argument(parameter.name + ": a weight input needs exactly one use");
    }
    return input(id, parameter.uses.front().input);
}

ops::WeightInput Model::input(WeightId id, std::string_view use_name) const {
    const auto& parameter = weight(id);
    const auto use        = std::find_if(parameter.uses.begin(), parameter.uses.end(),
                                         [&](const auto& u) { return u.input == use_name; });
    if (use == parameter.uses.end()) {
        throw std::invalid_argument(parameter.name + ": no use " + std::string(use_name));
    }
    if (use->hadamard_signs) {
        throw std::invalid_argument(parameter.name +
                                    ": Hadamard-rotated matrices are not supported");
    }
    ops::WeightInput result{parameter.view, use->policy, use->activation_input_divisor};
    if (use->input_columns) {
        result.input_columns = replicas_.on_rank(bound_, *use->input_columns, parameter.rank);
    }
    return result;
}

bool is_qwen4_exp(const artifact::Reader& reader) {
    const auto& components = reader.directory().components;
    const auto text        = components.find("text");
    if (text == components.end()) { return false; }
    const auto& config = text->second.config;
    return config.is_object() && config.contains("architectures") &&
           config.at("architectures").is_array() && config.at("architectures").size() == 1 &&
           config.at("architectures")[0] == "Qwen4ExpForCausalLM";
}

std::unique_ptr<Model> load_model(const artifact::Reader& reader, const LoadOptions& options,
                                  DeviceContext& device, const StartupObserver* observer) {
    if (!is_qwen4_exp(reader)) {
        throw artifact::ArtifactError("the artifact's text component is not Qwen4ExpForCausalLM");
    }
    if (options.ranks == 0 || options.ranks != device.size()) {
        throw std::invalid_argument("Qwen3.8-Flash-Next stages must match the device ranks");
    }
    const bool tensor_parallel = !options.tensor_split.empty();
    if (tensor_parallel) {
        if (options.tensor_split.size() != options.ranks || options.ranks < 2) {
            throw std::invalid_argument(
                "tensor parallelism needs one share per device rank and at least two ranks");
        }
        if (options.experts != ExpertResidency::Device) {
            throw std::invalid_argument(
                "--tensor-parallel divides Qwen3.8-Flash-Next's layers over "
                "the devices: it needs --expert-residency device");
        }
        if (!options.stage_layers.empty()) {
            throw std::invalid_argument("--tensor-parallel and --stage-layers exclude each other");
        }
    }
    TextConfig config = parse_text_config(reader.directory().component("text").config);
    artifact::Binder binder(reader);
    std::optional<qwen3_5::VisionConfig> vision_config;
    if (options.vision) {
        const auto& components = reader.directory().components;
        if (!components.contains("vision")) {
            throw std::invalid_argument("Vision needs an artifact converted with its tower "
                                        "(--components text,vision)");
        }
        vision_config = qwen3_5::parse_vision_config(reader.directory());
    }
    std::optional<MtpConfig> mtp_config;
    if (options.mtp) {
        const auto& components = reader.directory().components;
        if (!components.contains("mtp")) {
            throw std::invalid_argument("MTP needs an artifact converted with its MTP block "
                                        "(--components text,mtp)");
        }
        mtp_config = parse_mtp_config(reader.directory().component("mtp").config, config);
    }
    std::optional<DFlash2Config> dflash2_config;
    if (options.dflash2) {
        const auto& components = reader.directory().components;
        if (!components.contains("dflash2")) {
            throw std::invalid_argument("DFlash2 needs an artifact converted with its drafter "
                                        "(--components text,dflash2)");
        }
        dflash2_config = parse_dflash2_config(reader.directory().component("dflash2").config,
                                              config);
    }
    if (!options.dspine.empty()) {
        if (options.ranks > 1 && !tensor_parallel) {
            throw std::invalid_argument("the DSpine drafter runs on one device or under tensor "
                                        "parallelism, not over pipeline stages");
        }
        if (options.mtp || options.dflash2) {
            throw std::invalid_argument("Qwen3.8-Flash-Next drafts with one drafter: its MTP "
                                        "block, its DFlash2 drafter or a DSpine drafter");
        }
        dspine::check_target(dspine::read_config(options.dspine), config.hidden_size,
                             config.vocab_size, config.num_hidden_layers, "Qwen3.8-Flash-Next");
    }
    FrontendResources resources = bind_resources(binder, config, options.vision);
    Bindings b(binder);

    TextWeights weights;
    weights.token_embedding =
        b.parameter("text/token_embedding", {config.vocab_size, config.hidden_size});
    std::vector<std::string> head_uses{"text/final_hidden"};
    if (mtp_config) { head_uses.emplace_back("mtp/final_hidden"); }
    if (tensor_parallel) {
        // Every rank projects its rows of the head, for the text and for the MTP block's drafts,
        // in proportion to the split and ending on the head's stored row bounds and on whole
        // 128-row tiles of a GGUF head's matrix product. A prompt pass with logits of hundreds of
        // positions projects them with the whole head on rank 0 instead (below): gathering the
        // other ranks' rows of them through host memory costs more than their share saves.
        const Shape shape{config.vocab_size, config.hidden_size};
        const std::vector<bool> allowed = b.row_ends("text/output_head", shape);
        std::vector<std::uint64_t> ends;
        for (std::uint64_t row = 128; row < config.vocab_size; row += 128) {
            if (allowed[row]) { ends.push_back(row); }
        }
        weights.head_rows = split_shares(config.vocab_size, ends, options.tensor_split);
        for (std::size_t rank = 0; rank < options.ranks; ++rank) {
            weights.head_shares.push_back(
                b.piece("text/output_head", shape, head_uses, {}, rank,
                        row_piece({weights.head_rows[rank], weights.head_rows[rank + 1]})));
        }
        head_uses = {"text/final_hidden"};
    }
    // DFlash2 drafts project through the whole head, on the first device under tensor parallelism.
    if (dflash2_config) { head_uses.emplace_back("dflash2/final_hidden"); }
    weights.output_head = b.parameter("text/output_head", {config.vocab_size, config.hidden_size},
                                      std::move(head_uses));
    weights.final_mixer = bind_hc(b, config, "text/final_mixer/", false);
    std::vector<TensorShare> shares;
    if (tensor_parallel) { shares = plan_tensor_shares(b, config, options.tensor_split); }
    const auto ple_layer = [&](std::uint32_t i) {
        return std::find(config.ple_layers.begin(), config.ple_layers.end(), i) !=
               config.ple_layers.end();
    };
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const std::string p = "text/layers/" + std::to_string(i) + "/";
        LayerWeights layer{.attn_hc = bind_hc(b, config, p + "attn_hc/", true),
                           .mlp_hc  = bind_hc(b, config, p + "mlp_hc/", true),
                           .mixer   = GdnWeights{},
                           .moe     = {},
                           .ple     = std::nullopt};
        if (config.layer_types[i] == MixerKind::GatedDeltaNet) {
            layer.mixer = tensor_parallel ? bind_gdn_share(b, config, p, shares[0], 0)
                                          : bind_gdn(b, config, p);
        } else {
            layer.mixer = tensor_parallel ? bind_attention_share(b, config, p, shares[0], 0)
                                          : bind_attention(b, config, p);
        }
        layer.moe = bind_moe(b, config, p, options.experts, options.tensor_split);
        if (tensor_parallel) { bind_shared_run(b, config, p, shares[0], 0, layer.moe); }
        if (ple_layer(i)) { layer.ple = bind_ple(b, config, p); }
        weights.layers.push_back(std::move(layer));
    }
    // Every other rank's replicas and runs of each layer.
    for (std::size_t rank = 1; tensor_parallel && rank < options.ranks; ++rank) {
        const TensorShare& share = shares[rank];
        weights.rank_embeddings.push_back(
            replica(b, "text/token_embedding", {config.vocab_size, config.hidden_size}, {}, {}, rank));
        weights.rank_final_mixers.push_back(
            bind_hc_replica(b, config, "text/final_mixer/", false, rank));
        std::vector<LayerWeights> layers;
        for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
            const std::string p = "text/layers/" + std::to_string(i) + "/";
            const std::string m = p + "moe/", input = p + "ffn_input";
            LayerWeights layer{.attn_hc = bind_hc_replica(b, config, p + "attn_hc/", true, rank),
                               .mlp_hc  = bind_hc_replica(b, config, p + "mlp_hc/", true, rank),
                               .mixer   = std::monostate{},
                               .moe     = {},
                               .ple     = std::nullopt};
            if (config.layer_types[i] == MixerKind::GatedDeltaNet) {
                layer.mixer = bind_gdn_share(b, config, p, share, rank);
            } else if (share.kv_head_end > share.kv_head_begin) {
                layer.mixer = bind_attention_share(b, config, p, share, rank);
            }
            layer.moe.router =
                replica(b, m + "router", {config.num_experts, config.hidden_size}, {input}, {}, rank);
            layer.moe.shared_score =
                replica(b, m + "shared_score", {1, config.hidden_size}, {input}, {}, rank);
            if (config.latent_moe) {
                const auto ends  = latent_ends(b, config, p, options.tensor_split);
                layer.moe.latent = bind_latent_share(b, config, p, ends[rank], ends[rank + 1], rank);
            }
            bind_shared_run(b, config, p, share, rank, layer.moe);
            if (ple_layer(i)) { layer.ple = bind_ple_replica(b, config, p, rank); }
            layers.push_back(std::move(layer));
        }
        weights.rank_layers.push_back(std::move(layers));
    }
    std::optional<MtpWeights> mtp_weights;
    if (mtp_config) { mtp_weights = bind_mtp(b, config, *mtp_config, options.experts, {}); }
    std::optional<DFlash2Weights> dflash2_weights;
    if (dflash2_config) { dflash2_weights = bind_dflash2(b, config, *dflash2_config); }
    // The tower stays on rank 0, with the token embedding its output joins.
    std::optional<qwen3_5::VisionWeights> vision_weights;
    if (vision_config) {
        vision_weights = qwen3_5::loading::bind_vision(b, *vision_config, config.hidden_size,
                                                       artifact::Residency::Device);
    }

    // The stage split: what was asked, or the even split of the bytes each device would hold.
    const bool device_experts = options.experts == ExpertResidency::Device;
    StagePlan stages(config.num_hidden_layers);
    if (options.ranks > 1 && !tensor_parallel) {
        if (!options.stage_layers.empty()) {
            if (options.stage_layers.size() != options.ranks) {
                throw std::invalid_argument("--stage-layers must list one count per device");
            }
            stages = StagePlan::from_layer_counts(config.num_hidden_layers, options.stage_layers);
        } else {
            std::set<std::size_t> seen;
            std::vector<std::uint64_t> layer_bytes;
            for (const auto& layer : weights.layers) {
                const auto ids = layer_weights(layer, device_experts);
                layer_bytes.push_back(stored_bytes(b, ids, seen));
            }
            const std::vector<WeightId> first{weights.token_embedding};
            std::vector<WeightId> last{weights.output_head, weights.final_mixer.norm,
                                       weights.final_mixer.down, weights.final_mixer.up};
            if (mtp_weights) {
                const auto mtp = mtp_parameters(*mtp_weights, device_experts);
                last.insert(last.end(), mtp.begin(), mtp.end());
            }
            if (dflash2_weights) {
                const auto drafter = dflash2_parameters(*dflash2_weights);
                last.insert(last.end(), drafter.begin(), drafter.end());
            }
            stages = balanced_stages(layer_bytes, stored_bytes(b, first, seen),
                                     stored_bytes(b, last, seen), options.ranks);
        }
        for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
            const std::size_t rank = stages.placement(i).stage;
            for (const WeightId id : layer_weights(weights.layers[i], device_experts)) {
                b.place(id, rank);
            }
        }
        b.place(weights.token_embedding, 0);
        std::vector<WeightId> head{weights.output_head, weights.final_mixer.norm,
                                   weights.final_mixer.down, weights.final_mixer.up};
        if (mtp_weights) {
            const auto mtp = mtp_parameters(*mtp_weights, device_experts);
            head.insert(head.end(), mtp.begin(), mtp.end());
        }
        if (dflash2_weights) {
            const auto drafter = dflash2_parameters(*dflash2_weights);
            head.insert(head.end(), drafter.begin(), drafter.end());
        }
        for (const WeightId id : head) { b.place(id, stages.stages() - 1); }
    } else if (options.ranks == 1 && !options.stage_layers.empty()) {
        throw std::invalid_argument("--stage-layers needs --devices naming more than one device");
    }

    InstanceInfo info;
    info.name = reader.directory().metadata.value("name", std::string("Qwen3.8-Flash-Next"));
    info.metadata_json   = reader.directory().metadata.dump();
    info.provenance_json = reader.directory().provenance.dump();
    info.artifact_id     = reader.artifact_id();
    std::vector<std::filesystem::path> files;
    for (const auto& record : reader.directory().files) {
        files.push_back(record.path ? options.artifact.parent_path() / *record.path
                                    : options.artifact);
    }
    auto pending         = std::move(b.weights);
    auto materialization = std::move(binder).finish();
    auto backing = artifact::materialize(reader, std::move(materialization), device, observer);
    auto bound   = qwen3_5::loading::resolve_weights(std::move(pending), backing);
    auto model           = std::unique_ptr<Model>(
        new Model(std::move(config), options, std::move(weights), std::move(stages),
                  std::move(bound), std::move(resources), std::move(info), std::move(backing)));
    model->files_          = std::move(files);
    model->replicas_       = qwen3_5::AuxiliaryReplicas(model->bound_, device);
    model->vision_config_  = std::move(vision_config);
    model->vision_weights_ = std::move(vision_weights);
    model->mtp_config_     = std::move(mtp_config);
    model->mtp_weights_    = std::move(mtp_weights);
    model->dflash2_config_  = std::move(dflash2_config);
    model->dflash2_weights_ = std::move(dflash2_weights);
    if (!options.dspine.empty()) {
        model->dspine_ = dspine::load_model(options.dspine, device, observer);
    }
    model->shares_         = std::move(shares);
    return model;
}

} // namespace ninfer::models::qwen4_exp
