#include "models/qwen4_exp/executor.h"

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/decode_graph.h"
#include "core/device_buffer.h"
#include "core/device_wait.h"
#include "core/gdn_replay_records.h"
#include "core/host_worker_pool.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_view.h"
#include "core/paged_kv_storage.h"
#include "core/pinned_memory.h"
#include "core/qtype.h"
#include "core/stage_link.h"
#include "core/tensor_parallel.h"
#include "core/weight_view.h"
#include "ninfer/ops/target_logprobs.h"
#include "models/qwen4_exp/dflash2.h"
#include "models/dspine/drafter.h"
#include "models/qwen4_exp/ngram_hash.h"
#include "models/qwen4_exp/ngram_draft_prefetch.h"
#include "models/expert_profile.h"
#include "models/qwen4_exp/expert_misses.h"
#include "models/qwen4_exp/expert_stream.h"
#include "models/qwen4_exp/ngram_table.h"
#include "models/qwen3_5/execution/vision.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/kv_stream.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/moe_experts.h"
#include "ninfer/ops/moe_route.h"
#include "ninfer/ops/ngram_rows.h"
#include "ninfer/ops/ple_inject.h"
#include "ninfer/ops/qsa_indexer.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/sparse_attention.h"
#include "ninfer/ops/weight_input.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <future>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::models::qwen4_exp {
namespace {

constexpr std::int32_t kAlign = 256;
// Calls wider than this run their experts through the matrix kernel, from device memory.
constexpr std::int32_t kVectorTokens = 8;
// Tensor parallelism: a prompt chunk wider than this runs in pieces of this many columns from its
// first, the last one the rest, so that one piece's exchanges cross while the next piece computes
// (run_tensor_pieces) and the chunk computes what chunks of this width do. Pieces of 512 lost more
// to their narrower projections and expert calls than the overlap gained.
constexpr std::int32_t kTensorPieceColumns = 1024;
// The widest prompt chunk, cut into at most TensorParallelExchange::kPieces pieces.
constexpr std::uint32_t kMaxChunk = 4096;
static_assert((kMaxChunk + kTensorPieceColumns - 1) / kTensorPieceColumns <=
              TensorParallelExchange::kPieces);
// The family name of Flash-Next's built-in routing profiles (models/expert_profile.h), and of a
// latent MoE's (Edge0), whose experts route as they were trained for it.
constexpr std::string_view kExpertProfileFamily = "qwen3.8-flash-next";
constexpr std::string_view kLatentProfileFamily = "edge0";
// Zero bytes after every device slot's down matrix (see expert_stream.cpp and moe_experts_gguf).
constexpr std::uint64_t kSlotTail = 256;

std::uint64_t round_up(std::uint64_t value) { return (value + kAlign - 1) / kAlign * kAlign; }

// A logical projection's output rows, from one or more native operands: contiguous runs of one
// parent are fused into one operand, other parts write their rows separately.
struct Projection {
    struct Piece {
        ops::SingleProjectionWeight weight;
        std::int32_t row  = 0;
        std::int32_t rows = 0;
    };

    std::vector<Piece> pieces;
    std::int32_t rows = 0;
    std::int32_t k    = 0;
};

Projection make_projection(const std::vector<ops::WeightInput>& inputs) {
    Projection out;
    std::size_t begin = 0;
    while (begin < inputs.size()) {
        std::size_t end = begin + 1;
        while (end < inputs.size() && ops::joins(std::span<const ops::WeightInput>(
                                          inputs.data() + begin, end + 1 - begin))) {
            ++end;
        }
        const std::span<const ops::WeightInput> run(inputs.data() + begin, end - begin);
        Projection::Piece piece{ops::prepare_linear_weight(run), out.rows, 0};
        piece.rows = piece.weight.weight.n;
        if (out.k == 0) { out.k = piece.weight.weight.k; }
        if (piece.weight.weight.k != out.k) {
            throw std::invalid_argument("projection parts read different input widths");
        }
        out.rows += piece.rows;
        out.pieces.push_back(std::move(piece));
        begin = end;
    }
    return out;
}

Projection make_projection(const Model& model, std::initializer_list<WeightId> ids) {
    std::vector<ops::WeightInput> inputs;
    for (const WeightId id : ids) { inputs.push_back(model.input(id)); }
    return make_projection(inputs);
}

std::size_t projection_workspace(const Projection& p, std::int32_t tokens) {
    std::size_t bytes = 0;
    for (const auto& piece : p.pieces) {
        std::size_t need = ops::linear_workspace_capacity_bytes(
            piece.weight.weight.qtype, piece.rows, p.k, piece.weight.policy, 1, tokens);
        if (p.pieces.size() > 1) { need += round_up(std::uint64_t(piece.rows) * tokens * 2); }
        bytes = std::max(bytes, need);
    }
    return bytes;
}

// out (BF16 [p.rows, T]) <- the projection of x (BF16 [p.k, T]).
void project(const Projection& p, const Tensor& x, Tensor& out, WorkspaceArena& workspace,
             cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (p.pieces.size() == 1) {
        ops::linear(x, p.pieces.front().weight.weight, out, p.pieces.front().weight.policy,
                    workspace, stream);
        return;
    }
    for (const auto& piece : p.pieces) {
        auto scope  = workspace.scope();
        Tensor part = workspace.alloc(DType::BF16, {piece.rows, tokens});
        ops::linear(x, piece.weight.weight, part, piece.weight.policy, workspace, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::byte*>(out.data) + std::size_t(piece.row) * 2,
                                     std::size_t(p.rows) * 2, part.data,
                                     std::size_t(piece.rows) * 2, std::size_t(piece.rows) * 2,
                                     tokens, cudaMemcpyDeviceToDevice, stream));
    }
}

// One of several projections of one input and the BF16 [p.rows, T] it writes (project_together).
struct ProjectionInto {
    const Projection* projection = nullptr;
    Tensor* out                  = nullptr;
};

[[nodiscard]] bool gguf_projection(const Projection& p) {
    return std::all_of(p.pieces.begin(), p.pieces.end(), [](const Projection::Piece& piece) {
        return is_gguf(piece.weight.weight.qtype);
    });
}

// Each out <- its projection of x (BF16 [k, T]). The GGUF projections run as one gguf_project,
// which quantizes x to q8_1 once for all of their parts and gives each part's rows what a call of
// its own gives them; a projection of another format (a BF16 gate or indexer) takes a call of its
// own.
void project_together(std::span<const ProjectionInto> list, const Tensor& x,
                      WorkspaceArena& workspace, cudaStream_t stream) {
    std::vector<ops::detail::GgufProduct> products;
    for (const ProjectionInto& into : list) {
        if (!gguf_projection(*into.projection)) {
            project(*into.projection, x, *into.out, workspace, stream);
            continue;
        }
        if (into.projection->k != x.ne[0]) {
            throw std::invalid_argument("projections of one input read its width");
        }
        for (const auto& piece : into.projection->pieces) {
            products.push_back(
                {.weight = &piece.weight.weight, .out = into.out, .row = piece.row});
        }
    }
    ops::detail::gguf_project(x, products, workspace, stream);
}

// The workspace project_together needs for these projections at up to `tokens` columns.
std::size_t projections_workspace(std::initializer_list<const Projection*> list,
                                  std::int32_t tokens) {
    std::size_t bytes = 0;
    std::vector<ops::detail::GgufShape> shapes;
    for (const Projection* p : list) {
        bytes = std::max(bytes, projection_workspace(*p, tokens));
        if (!gguf_projection(*p)) { continue; }
        for (const auto& piece : p->pieces) {
            shapes.push_back({piece.weight.weight.qtype, piece.rows, p->k});
        }
    }
    return shapes.empty()
               ? bytes
               : std::max(bytes, ops::detail::gguf_project_workspace_bytes(shapes, 1, tokens));
}

Tensor direct(const Model& model, WeightId id, std::initializer_list<std::int32_t> shape,
              DType dtype = DType::BF16) {
    const auto& view = model.weight(id).view;
    Tensor tensor    = weight_tensor(view, shape);
    if (tensor.dtype != dtype) {
        throw std::invalid_argument(model.weight(id).name + " must be stored as " +
                                    (dtype == DType::BF16  ? "bf16"
                                     : dtype == DType::I32 ? "int32"
                                                           : "fp32"));
    }
    return tensor;
}

// The Vision tower's native operands, prepared as the Qwen3.5 Program prepares them.
qwen3_5::execution::VisionParameters vision_parameters_for(const Model& model,
                                                           const qwen3_5::VisionWeights& w) {
    const auto linear = [&](WeightId id) { return ops::prepare_linear_weight(model.input(id)); };
    const auto tensor = [&](WeightId id) {
        const auto& view = model.weight(id).view;
        if (view.shape.size() == 1) {
            return weight_tensor(view, {static_cast<std::int32_t>(view.shape[0])});
        }
        return weight_tensor(view, {static_cast<std::int32_t>(view.shape[1]),
                                    static_cast<std::int32_t>(view.shape[0])});
    };
    const auto norm = [&](const qwen3_5::NormWeights& n) {
        return qwen3_5::execution::NormParameters{tensor(n.weight), tensor(n.bias)};
    };
    // The query, key and value biases form one bank, as the fused QKV projection reads them.
    const auto joined = [&](std::array<WeightId, 3> ids) {
        WeightView view;
        std::uint64_t count = 0;
        for (const WeightId id : ids) {
            const auto& input = model.weight(id).view;
            count += weight_element_count(input.shape);
            for (const auto& part : input.parts) {
                if (!view.parts.empty() && (view.parts.back().parent != part.parent ||
                                            view.parts.back().end != part.begin)) {
                    throw std::invalid_argument("Vision QKV biases must be one contiguous bank");
                }
                view.parts.push_back(part);
            }
        }
        view.shape = {count};
        return weight_tensor(view, {static_cast<std::int32_t>(count)});
    };
    qwen3_5::execution::VisionParameters out;
    out.patch_embedding      = linear(w.patch_embedding);
    out.patch_embedding_bias = tensor(w.patch_embedding_bias);
    out.position_embedding   = tensor(w.position_embedding);
    for (const auto& layer : w.layers) {
        const std::array qkv{model.input(layer.query), model.input(layer.key),
                             model.input(layer.value)};
        out.layers.push_back({norm(layer.norm1), norm(layer.norm2),
                              ops::prepare_linear_weight(qkv),
                              joined({layer.query_bias, layer.key_bias, layer.value_bias}),
                              linear(layer.output), linear(layer.fc1), linear(layer.fc2),
                              tensor(layer.output_bias), tensor(layer.fc1_bias),
                              tensor(layer.fc2_bias)});
    }
    out.merger_norm     = norm(w.merger_norm);
    out.merger_fc1      = linear(w.merger_fc1);
    out.merger_fc2      = linear(w.merger_fc2);
    out.merger_fc1_bias = tensor(w.merger_fc1_bias);
    out.merger_fc2_bias = tensor(w.merger_fc2_bias);
    return out;
}

struct HcPlan {
    Tensor norm;
    ops::HyperConnectionMatrix down, up, inject;

    [[nodiscard]] ops::HyperConnectionWeights weights() const { return {&norm, down, up, inject}; }
};

// A hyper-connection matrix as stored: ggml Q8_0 blocks, or BF16 words.
ops::HyperConnectionMatrix hc_matrix(const Model& model, WeightId id,
                                     std::initializer_list<std::int32_t> shape) {
    const Weight w = native_weight(model.weight(id).view);
    if (w.qtype == QType::GGUF_Q8_0) {
        return {w.qdata, ops::HyperConnectionMatrix::Format::Q8_0};
    }
    return {direct(model, id, shape).data, ops::HyperConnectionMatrix::Format::BF16};
}

HcPlan make_hc(const Model& model, const HyperConnectionWeights& w, const TextConfig& c) {
    const auto width = static_cast<std::int32_t>(c.hc_count * c.hidden_size);
    const auto low   = static_cast<std::int32_t>(c.hc_lowrank);
    HcPlan out;
    out.norm = direct(model, w.norm, {width});
    out.down = hc_matrix(model, w.down, {width, low});
    out.up   = hc_matrix(model, w.up, {low, width});
    if (w.inject) {
        out.inject = hc_matrix(model, *w.inject, {width, static_cast<std::int32_t>(c.hc_count)});
    }
    return out;
}

// A Gated DeltaNet mixer's operands: every head, or a tensor-parallel rank's key heads with the
// value heads that read them.
struct GdnPlan {
    Projection qkv, z, a, b, output;
    Tensor convolution, a_log, dt_bias, norm;
    std::int32_t key_heads = 0, value_heads = 0;
};

// A sparse-attention mixer's operands: both KV heads, or a tensor-parallel rank's one with its
// query heads (the indexer whole either way).
struct QsaPlan {
    Projection query, gate, key, value, index, output;
    Tensor query_norm, key_norm, index_query_norm, index_key_norm;
    std::int32_t kv_heads = 0;
};

struct PlePlan {
    Projection key, value;
    Tensor norm_key, norm_query, norm_conv, convolution;
};

// One projection of every expert of a layer, as a device table of expert base pointers on the
// layer's device.
struct ExpertTable {
    QType format           = QType::GGUF_Q8_0;
    std::int64_t row_bytes = 0;
    std::int32_t rows      = 0;
    std::vector<const void*> pointers; // each expert's rows where the artifact put them
    DeviceBuffer table;
    bool aligned16 = false;

    // A null entry (another rank's expert) is never read.
    void seal() {
        aligned16 = row_bytes % 16 == 0 &&
                    std::any_of(pointers.begin(), pointers.end(),
                                [](const void* pointer) { return pointer != nullptr; }) &&
                    std::all_of(pointers.begin(), pointers.end(), [](const void* pointer) {
                        return reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0;
                    });
    }

    [[nodiscard]] ops::GgufExpertTable view() const {
        return {format, static_cast<const void* const*>(table.p), row_bytes, aligned16};
    }
};

ExpertTable make_table(const Model& model, std::span<const WeightId> ids) {
    ExpertTable out;
    std::vector<const void*> pointers;
    for (const WeightId id : ids) {
        const Weight w = native_weight(model.weight(id).view);
        if (!is_gguf(w.qtype)) {
            throw std::invalid_argument(model.weight(id).name +
                                        ": the expert kernels take GGUF block banks");
        }
        const auto block       = gguf_block_shape(w.qtype);
        const std::int64_t row = std::int64_t(w.k / block.elements) * block.bytes;
        if (pointers.empty()) {
            out.format    = w.qtype;
            out.row_bytes = row;
        } else if (w.qtype != out.format || row != out.row_bytes) {
            throw std::invalid_argument(model.weight(id).name +
                                        ": every expert of a bank must share its block type");
        }
        pointers.push_back(w.qdata);
        out.rows = w.n;
    }
    out.table = DeviceBuffer(pointers.size() * sizeof(void*));
    out.table.copy_from_host(pointers.data(), pointers.size() * sizeof(void*));
    out.pointers = std::move(pointers);
    out.seal();
    return out;
}

// Tensor parallelism: the table of the experts `rank` holds, the other entries null (never read).
ExpertTable make_rank_table(const Model& model, std::span<const WeightId> ids, std::size_t rank) {
    ExpertTable out;
    std::vector<const void*> pointers(ids.size(), nullptr);
    for (std::size_t e = 0; e < ids.size(); ++e) {
        const auto& bound = model.weight(ids[e]);
        if (bound.rank != rank) { continue; }
        const Weight w = native_weight(bound.view);
        if (!is_gguf(w.qtype)) {
            throw std::invalid_argument(bound.name + ": tensor parallelism takes GGUF block banks");
        }
        const auto block       = gguf_block_shape(w.qtype);
        const std::int64_t row = std::int64_t(w.k / block.elements) * block.bytes;
        if (out.row_bytes == 0) {
            out.format    = w.qtype;
            out.row_bytes = row;
        } else if (w.qtype != out.format || row != out.row_bytes) {
            throw std::invalid_argument(bound.name +
                                        ": every expert of a bank must share its block type");
        }
        pointers[e] = w.qdata;
        out.rows    = w.n;
    }
    out.table = DeviceBuffer(pointers.size() * sizeof(void*));
    out.table.copy_from_host(pointers.data(), pointers.size() * sizeof(void*));
    out.pointers = std::move(pointers);
    out.seal();
    return out;
}

// A layer's table for experts that stay in the artifact's files: every entry null until the
// expert stream makes the expert resident.
ExpertTable make_located_table(std::span<const ExpertLocation> locations) {
    ExpertTable out;
    for (const auto& location : locations) {
        if (out.row_bytes == 0) {
            out.format    = location.format;
            out.row_bytes = location.row_bytes;
            out.rows      = location.rows;
        } else if (location.format != out.format || location.row_bytes != out.row_bytes ||
                   location.rows != out.rows) {
            throw std::invalid_argument("every expert of a bank must share its block type");
        }
    }
    out.pointers.assign(locations.size(), nullptr);
    out.table = DeviceBuffer(out.pointers.size() * sizeof(void*));
    out.table.copy_from_host(out.pointers.data(), out.pointers.size() * sizeof(void*));
    return out;
}

// A latent MoE's encoder or decoder: the groups [begin, end) of `groups` as one bank of their equal
// runs of rows (all of them, or a tensor-parallel rank's), as a table of `groups` entries pointing
// at the runs' first rows, the other groups' null (never read).
ExpertTable make_group_table(const Model& model, WeightId id, std::uint32_t groups,
                             std::uint32_t begin, std::uint32_t end) {
    const Weight w = native_weight(model.weight(id).view);
    if (!is_gguf(w.qtype) || begin >= end || end > groups || w.n % (end - begin) != 0) {
        throw std::invalid_argument(model.weight(id).name +
                                    ": a latent projection takes a GGUF bank of whole groups");
    }
    const auto block = gguf_block_shape(w.qtype);
    ExpertTable out;
    out.format    = w.qtype;
    out.row_bytes = std::int64_t(w.k / block.elements) * block.bytes;
    out.rows      = static_cast<std::int32_t>(w.n / (end - begin));
    out.pointers.assign(groups, nullptr);
    for (std::uint32_t g = begin; g < end; ++g) {
        out.pointers[g] = static_cast<const std::byte*>(w.qdata) +
                          std::size_t(g - begin) * out.rows * out.row_bytes;
    }
    out.table = DeviceBuffer(out.pointers.size() * sizeof(void*));
    out.table.copy_from_host(out.pointers.data(), out.pointers.size() * sizeof(void*));
    out.seal();
    return out;
}

// Native operands retain the artifact's planes. The Program owns only these device tables.
struct NativeExpertTable {
    QType format = QType::BF16;
    bool integer_a8 = false;
    DeviceBuffer table;
    std::vector<Weight> operands;
    [[nodiscard]] ops::NativeExpertTable view() const {
        return {format, static_cast<const Weight*>(table.p), integer_a8};
    }
};

NativeExpertTable make_native_table(const Model& model, std::span<const WeightId> ids,
                                  std::int32_t rows, std::int32_t columns) {
    NativeExpertTable out;
    std::vector<Weight> operands;
    for (const WeightId id : ids) {
        const auto input = model.input(id);
        const auto format = native_weight(input.weight).qtype;
        const bool a8 = format != QType::BF16 && input.policy != ops::LinearPolicy::A16Only;
        Weight operand = ops::prepare_native_expert(input, rows, columns, a8);
        if (operands.empty()) {
            out.format = operand.qtype;
            out.integer_a8 = a8;
        } else if (operand.qtype != out.format || a8 != out.integer_a8) {
            throw std::invalid_argument(model.weight(id).name +
                                        ": native expert bank has mixed formats or policies");
        }
        operands.push_back(operand);
    }
    out.table = DeviceBuffer(operands.size() * sizeof(Weight));
    out.table.copy_from_host(operands.data(), operands.size() * sizeof(Weight));
    out.operands = std::move(operands);
    return out;
}

struct NativeMoePlan {
    NativeExpertTable gate, up, down, shared_gate, shared_up, shared_down;
    std::int32_t experts = 0;
    [[nodiscard]] ops::NativeMoeWeights banks() const {
        return {gate.view(), up.view(), down.view(), shared_gate.view(), shared_up.view(),
                shared_down.view(), experts};
    }
};

// Edge0's latent experts (TextConfig::latent_moe): the groups' encoders and decoders as tables of
// `groups` entries, the experts' group labels and the optional multipliers.
struct LatentPlan {
    ExpertTable encode, decode;
    Tensor labels, expert_scales, column_scale;
    std::int32_t width = 0, groups = 0;
};

struct MoePlan {
    Tensor router, shared_gate;
    ExpertTable gate, up, down, shared_gate_table, shared_up_table, shared_down_table;
    std::optional<NativeMoePlan> native;
    std::optional<LatentPlan> latent;
    // Device banks or disk experts' device slots, with zeros after the down banks; host experts
    // are read across the bus.
    bool device_resident = false;
    // Tensor parallelism: the routed experts [expert_begin, expert_end) the rank holds, the only
    // entries of its tables that are set; the shared expert's tables hold the rank's run of it. A
    // latent MoE's rank holds the layer's groups [group_begin, group_end) and the experts they
    // label (LatentWeights).
    bool share = false;
    std::uint32_t expert_begin = 0, expert_end = 0;
    std::uint32_t group_begin = 0, group_end = 0;
    bool iq1m_matrix = true; // ExecutorOptions::iq1m_matrix

    [[nodiscard]] ops::GgufMoeWeights banks() const {
        return {gate.view(),
                up.view(),
                down.view(),
                shared_gate_table.view(),
                shared_up_table.view(),
                shared_down_table.view(),
                static_cast<std::int32_t>(gate.pointers.size()),
                device_resident,
                shared_gate_table.rows > 0 ? shared_gate_table.rows : 640,
                iq1m_matrix};
    }
    // The latent call over `routed`'s routed and shared tables (banks(), pooled() or cached()).
    [[nodiscard]] ops::GgufLatentWeights latent_banks(const ops::GgufMoeWeights& routed) const {
        ops::GgufLatentWeights out;
        out.gate            = routed.gate;
        out.up              = routed.up;
        out.down            = routed.down;
        out.shared_gate     = routed.shared_gate;
        out.shared_up       = routed.shared_up;
        out.shared_down     = routed.shared_down;
        out.encode          = latent->encode.view();
        out.decode          = latent->decode.view();
        out.labels          = static_cast<const std::int32_t*>(latent->labels.data);
        out.expert_scales   = static_cast<const float*>(latent->expert_scales.data);
        out.column_scale    = static_cast<const float*>(latent->column_scale.data);
        out.width           = latent->width;
        out.groups          = latent->groups;
        out.experts         = routed.experts;
        out.shared_width    = routed.shared_width;
        out.device_resident = routed.device_resident;
        return out;
    }
    [[nodiscard]] ops::GgufLatentWeights latent_banks() const { return latent_banks(banks()); }
    // A wide call over a slot pool's table ([E] gate, [E] up, [E] down). Its entries are device
    // cache slots' planes, which start at 256-byte boundaries, or a pool bank's experts, each at
    // its index from a 256-byte boundary: a row stride of whole 16 bytes aligns every entry.
    [[nodiscard]] ops::GgufMoeWeights pooled(const void* const* table) const {
        ops::GgufMoeWeights out   = cached();
        const std::size_t experts = gate.pointers.size();
        out.gate.experts          = table;
        out.up.experts            = table + experts;
        out.down.experts          = table + 2 * experts;
        return out;
    }
    // A wide call over the expert cache's own tables whose pairs name cached experts only (the
    // staged misses' first stage): every entry it reads is a cache slot's plane, as in pooled().
    [[nodiscard]] ops::GgufMoeWeights cached() const {
        ops::GgufMoeWeights out = banks();
        out.gate.aligned16      = gate.row_bytes % 16 == 0;
        out.up.aligned16        = up.row_bytes % 16 == 0;
        out.down.aligned16      = down.row_bytes % 16 == 0;
        out.device_resident     = true;
        return out;
    }
};

struct LayerPlan {
    std::size_t rank = 0;
    HcPlan attn_hc, mlp_hc;
    std::optional<GdnPlan> gdn;
    std::optional<QsaPlan> qsa;
    std::optional<PlePlan> ple;
    MoePlan moe;
};

// One plane of a sparse-attention layer's paged KV, [leading, 64, heads, pages]: device memory,
// or pinned host memory the device addresses directly when the layer streams.
struct KvPlane {
    DeviceBuffer device;
    std::unique_ptr<PinnedHostBuffer> host;
    void* p             = nullptr;
    std::uint64_t bytes = 0;
};

// One sequence's mutable state of one layer, on that layer's device.
struct LayerState {
    Tensor ssm, conv;          // Gated DeltaNet: the sequence's slot of the pool
    std::array<KvPlane, 4> kv; // sparse attention: K, V and a quantized KV's K and V scales
    DeviceBuffer pooled, tail; // the indexer's pooled keys and incomplete block
    // A streamed layer (ExecutorOptions::kv_resident): `kv` lives in host memory and the device
    // keeps a pool of 4-position pages of the same planes, with its residency map.
    std::array<DeviceBuffer, 4> pool;
    DeviceBuffer map_memory;
    ops::KvStreamMap map;
    DeviceBuffer history; // PLE
};

struct SequenceState {
    std::uint32_t slot     = 0; // its index, and its slot of every Gated DeltaNet state pool
    std::uint32_t position = 0;
    NgramContext context;
    std::vector<LayerState> layers;
    // Tensor parallelism: every rank after the first's state of each layer, [rank - 1][layer]
    // (`layers` is rank 0's).
    std::vector<std::vector<LayerState>> rank_layers;
    // The MTP block (an executor with MTP drafts), on the head rank: its layer's state and the
    // target's stack at the sequence's last position (which the next cell pairs with the next
    // token). Whether the drafter (the MTP block, or the DFlash2 drafter's context ring) follows
    // the sequence's tokens, which a branch or a prompt that ends in media breaks.
    LayerState mtp;
    DeviceBuffer mtp_pending, mtp_scratch_tail;
    bool drafter_follows = true;
    // The last verify(): the n-gram context before it and the tokens it ran.
    NgramContext verify_context;
    std::vector<std::int32_t> verify_tokens;
    // A prompt's media (Vision): for each prompt position the column of its merged embedding, or
    // -1, and the prompt's RoPE positions, axis-major [3, tokens]; positions past them rotate at
    // their index plus rope_delta.
    std::vector<std::int32_t> media_columns, media_rope;
    std::int32_t rope_delta = 0;
};

// The CUDA Graphs of one pass shape of a single sequence, which every sequence shares: a graph per
// segment (a decode step, a verification at one width) or one (a draft chain on one device with
// every expert there, a draft step). The first run of the shape is eager, so every lazy
// initialization of the ops it reaches happens outside a capture; the next run captures the graphs
// for its sequence, and a run of another sequence captures that one's pass and updates them in
// place: its nodes are the same kernels and copies over the other sequence's state. They read the
// tokens, their positions and n-gram rows from the ranks' staged planes, so they stay valid across
// positions and resets.
struct PassGraphs {
    std::vector<DecodeGraphExecutable> graphs;
    const SequenceState* sequence = nullptr; // the sequence the graphs run
    std::uint32_t runs            = 0;
};

// Consecutive layers on one rank, in pass order. The first segment also embeds the tokens, the
// segment on the head rank after the last layer also runs the final mixer and the head. A rank's
// first PLE layer starts a segment (`rows`) before which the pass uploads the n-gram rows, so the
// layers before it run while the rows are read. A segment holds at most kSegmentLayers layers: a
// captured segment is one CUDA Graph, and the device starts a graph only once its launch has
// submitted every node (about 1.8 us a kernel on a 4090: 2.9 ms for a 44-layer verification), so
// shorter graphs let the first run while the host launches the rest.
struct Segment {
    std::size_t rank  = 0;
    std::size_t begin = 0, end = 0; // layers
    bool embed = false, head = false, rows = false;
};

// One wait for the n-gram rows on the stream of the rank that reads them: `wanted` completes when
// the layers before the PLE layer are done, `ready` when the rows are uploaded, so the time between
// them is the time the stream stood still.
struct StallEvents {
    cudaEvent_t wanted = nullptr, ready = nullptr;
    bool pending = false;
};

// One sequence's tokens within a pass: columns [column, column + count) of the activation planes.
// A prefill pass has one part; a batched decode pass has one single-token part per sequence. The
// per-token work (embedding, hyper-connections, MoE, head) runs over every column at once; the
// mixers and the PLE injection, which read and write a sequence's state, run part by part.
struct Part {
    SequenceState* sequence = nullptr;
    std::int32_t column     = 0;
    std::int32_t count      = 0;
    // Its first position past the sequence's: a branch after the prompt part of its pass.
    std::uint32_t offset = 0;
};

// Columns [column, column + count) of a pass, which one or more parts hold.
struct ColumnRun {
    std::int32_t column = 0;
    std::int32_t count  = 0;
};

// What one device holds for the layers it runs: activation planes sized for a full chunk and a
// scoped workspace.
struct RankState {
    std::size_t rank    = 0;
    cudaStream_t stream = nullptr;
    DeviceBuffer activations;
    std::unique_ptr<DeviceArena> workspace;
    DeviceBuffer block_table;
    std::unique_ptr<PinnedHostBuffer> staging;
    cudaEvent_t staged = nullptr;
    cudaEvent_t done   = nullptr;
    cudaEvent_t copied = nullptr; // its stack has been copied in from the previous rank
    cudaEvent_t routes = nullptr; // the last pass's routes reached the host
    // Tensor parallelism: rank 0's work so far, which every other rank's stream waits for at a
    // segment's start (and so joins its capture), and the rank's work, which rank 0's stream waits
    // for at the segment's end.
    cudaEvent_t fork = nullptr, join = nullptr;
    // Tensor parallelism: the rank's uploads of a pass (stage_inputs, stage_rows), which rank 0's
    // stream waits for before it replays the pass's graphs (fence_uploads).
    cudaEvent_t uploaded = nullptr;
    ops::GgufMoeSide side;        // the decode MoE's shared expert runs there concurrently
    // Activation planes (capacity: prefill_chunk tokens).
    float* stack            = nullptr;
    std::int32_t* ids       = nullptr;
    std::int32_t* positions = nullptr; // the tokens' sequence indices: KV slots, causal order
    std::int32_t* rope      = nullptr; // their 1-D RoPE positions
    std::int32_t* mrope     = nullptr; // a media prompt's [T, 3] RoPE positions
    std::int32_t* scatter   = nullptr; // columns that take a Vision embedding
    void* mixed             = nullptr; // BF16 [H, T]
    float* inject           = nullptr; // [4, T]
    void* y                 = nullptr; // BF16 or FP32 [H, T]
    // Tensor parallelism: BF16 [H, T], a prompt piece's exchanged mixer output, which the experts'
    // exchange of an earlier piece (into y) must not overwrite before the piece's MoE reads it.
    void* mixer_out         = nullptr;
    void* a                 = nullptr; // generic planes
    void* b                 = nullptr;
    void* c                 = nullptr;
    void* d                 = nullptr;
    void* e                 = nullptr;
    void* f                 = nullptr;
    void* g                 = nullptr;
    float* route_weights    = nullptr;
    float* route_shared     = nullptr;
    std::int32_t* selected  = nullptr;
    std::int32_t* counts    = nullptr;
    std::int32_t* route_ids = nullptr; // a tensor-parallel rank's routes; rank 0 records its own
    void* rows              = nullptr; // staged n-gram rows
    void* logits            = nullptr; // head rank only
    // The Gated DeltaNet state of the rank's layers, one slot per sequence (gdn_local maps a layer
    // to its pool layer).
    DeviceBuffer gdn_backing;
    std::unique_ptr<LinearAttentionStatePool> gdn;
    DeviceBuffer slot_ids; // I32 [sequences] = 0, 1, ...: a sequence's pool slot on the device
    DeviceBuffer route_done; // the router's completion counter (moe_route), zero between calls
    // Speculative verification (an executor with drafts): the replay records of the rank's Gated
    // DeltaNet layers and their fold, each sparse-attention layer's indexer projections (BF16
    // [640, columns] per layer), the PLE layer's normalised convolution input (FP32
    // [10240, columns]), and scratch for the states a verification reads but must not advance.
    DeviceBuffer record_backing;
    std::optional<GdnReplayRecords> records;
    std::optional<ops::GdnReplayFoldPlan> fold;
    DeviceBuffer index_records, ple_records;
    DeviceBuffer conv_scratch, tail_scratch;
    // Streamed sparse-attention layers: a prompt chunk reads one layer's whole KV from this
    // device copy (the four planes, every page), which holds `kv_staged`'s positions before
    // `kv_staged_end`; the layers of a span's attention phase keep it across the span's chunks.
    std::array<DeviceBuffer, 4> kv_staging;
    const LayerState* kv_staged = nullptr;
    std::uint32_t kv_staged_end = 0;
};

// Where a span keeps a sequence's state (Executor::forward's cut): before chunk `chunk`, the first
// of its sequence past the cut, into `state`; no cut without a state.
struct SpanCut {
    std::size_t chunk       = 0;
    SequenceSnapshot* state = nullptr;
};

// The masked drafter of a DFlash2 or a DSpine round. The two take the same calls over a ring of
// the target's features and leave the same drafts, candidates and probabilities
// (models/dspine/drafter.h), so the executor drives either through this.
class MaskedDrafter {
public:
    MaskedDrafter() = default;
    explicit MaskedDrafter(std::unique_ptr<DFlash2Drafter> dflash2)
        : dflash2_(std::move(dflash2)) {}
    MaskedDrafter(std::unique_ptr<dspine::Drafter> dspine,
                  std::vector<CyclicKVCacheLayerView> ring)
        : dspine_(std::move(dspine)), ring_(std::move(ring)) {}

    explicit operator bool() const noexcept { return dflash2_ != nullptr || dspine_ != nullptr; }
    // The executor calls it as it would a pointer to either drafter.
    const MaskedDrafter* operator->() const noexcept { return this; }

    [[nodiscard]] std::uint32_t taps() const {
        return dflash2_ ? dflash2_->taps() : dspine_->config().taps();
    }

    void materialize(const DFlash2ContextCall& call, WorkspaceArena& work,
                     cudaStream_t stream) const {
        if (dflash2_) {
            dflash2_->materialize(call, work, stream);
            return;
        }
        dspine_->materialize({.features  = call.features,
                              .positions = call.positions,
                              .counts    = call.counts,
                              .lanes     = call.lanes,
                              .min_count = call.min_count,
                              .max_count = call.max_count},
                             ring_, work, stream);
    }

    void propose(const DFlash2ProposalCall& call, WorkspaceArena& work, cudaStream_t stream) const {
        if (dflash2_) {
            dflash2_->propose(call, work, stream);
            return;
        }
        dspine_->propose({.anchors       = call.anchors,
                          .frontiers     = call.frontiers,
                          .valid_columns = call.valid_columns,
                          .lanes         = call.lanes,
                          .configs       = call.configs,
                          .batch         = call.batch,
                          .width         = call.width,
                          .drafts        = call.drafts,
                          .candidates    = call.candidates,
                          .proposal_q    = call.proposal_q,
                          .confidence    = call.confidence},
                         ring_, work, stream);
    }

private:
    std::unique_ptr<DFlash2Drafter> dflash2_;
    std::unique_ptr<dspine::Drafter> dspine_;
    std::vector<CyclicKVCacheLayerView> ring_;
};

} // namespace

struct Executor::Impl {
    // The MTP block's operands on the head rank: its input norms and projections, its layer and
    // final mixer, and the head as its final mixer's output reads it.
    // Under tensor parallelism `head` is rank 0's rows of the head, and each rank after the first
    // projects its rows of a draft with rank_heads[rank - 1] (run_divided_mtp_head).
    struct MtpPlan {
        Tensor embedding_norm, hidden_norm;
        Projection fc_embedding, fc_hidden, head;
        LayerPlan layer;
        HcPlan final_mixer;
        std::vector<Projection> rank_heads;
    };

    // One sequence's cells of an MTP pass: `count` consecutive cells from position `first`, at
    // columns [column, column + count) of the pass.
    struct MtpPart {
        SequenceState* sequence = nullptr;
        std::int32_t column     = 0;
        std::int32_t count      = 0;
        std::uint32_t first     = 0;
    };

    // Tokens a target pass committed for one sequence: `count` of them from column `column` of
    // the pass, the first at `position`.
    struct Committed {
        SequenceState* sequence = nullptr;
        std::int32_t column     = 0;
        std::int32_t count      = 0;
        std::uint32_t position  = 0;
        std::span<const std::int32_t> tokens;
    };

    const Model& model;
    DeviceContext& device;
    ExecutorOptions options;
    TextConfig config;
    NgramHashConstants ngram;
    std::unique_ptr<NgramTableReader> table;
    std::unique_ptr<NgramDraftPrefetch> draft_prefetch;
    // Native host banks use the CPU/GPU scheduler; GGUF banks retain their mapped-host route.
    std::vector<std::size_t> hybrid_layers;
    // The rank of the first PLE layer, whose staging buffer the rows are read into; whether this
    // pass's stage_rows() has waited for them; the waits the stall counters have yet to measure.
    std::size_t rows_rank = 0;
    bool rows_waited      = true;
    bool warming          = false; // warm_up() runs: the expert cache counts no routes
    // The CUDA Graphs of the single-sequence passes: decode steps, verifications by width, draft
    // chains by step count, and the single steps by index of a chain a probability floor can end
    // early. The warm-up captures every shape the options allow, and the device memory they took
    // on the rank that gave the most.
    PassGraphs decode_graphs;
    std::array<PassGraphs, 17> verify_graphs;
    std::array<PassGraphs, 16> draft_graphs;
    std::array<PassGraphs, 15> step_graphs;
    // The same shapes of chains whose rows draw their drafts (draft() with a sampling row).
    std::array<PassGraphs, 16> sampled_draft_graphs;
    std::array<PassGraphs, 15> sampled_step_graphs;
    std::uint64_t graph_bytes = 0;
    // The warm-up runs every draft step of a chain a probability floor would end early.
    bool every_step = false;
    std::array<StallEvents, 8> stall_events{};
    std::size_t stall_next = 0;
    std::uint64_t stalls   = 0;
    double stall_seconds   = 0.0;
    Weight embedding_table;
    Projection head;
    // Tensor parallelism: the whole head on rank 0, for a prompt pass of many logits
    // (run_whole_head); `head` is then rank 0's rows of it, for the divided head.
    Projection whole_head;
    HcPlan final_mixer;
    std::vector<LayerPlan> layers;
    static constexpr std::size_t kSegmentLayers = 6;
    std::vector<Segment> segments;
    std::vector<RankState> ranks;
    std::vector<SequenceState> sequences;
    // A layer's index in its rank's Gated DeltaNet pool, and in its rank's indexer records.
    std::vector<std::uint32_t> gdn_local, qsa_local;
    // MTP speculative decoding (options.draft_tokens > 0). The MTP layer's index in the per-layer
    // route records, expert cache and expert stream is num_hidden_layers.
    std::optional<MtpPlan> mtp;
    // Maximum and current verification widths (drafts plus anchor; the current one the widest
    // sequence's, the record stride), and the public vocabulary.
    std::uint32_t verify_width = 0;
    std::uint32_t active_verify_width = 0;
    std::int32_t domain        = 0;
    bool verifying             = false;
    // A pass of `forward`: a prompt chunk, whose tensor-parallel exchanges take the prompt
    // partials (ExecutorOptions::tensor_parallel_prompt_partials).
    bool prompt_pass = false;
    // A prompt pass that leaves no logits and whose final stacks nothing reads (kv_only_allowed):
    // its last layer, a sparse-attention one, runs only what the sequence keeps -- the KV append
    // and the indexer's append -- and neither its attention, its experts nor the head run.
    bool kv_only_pass = false;
    // The text layers whose experts the pass's wide calls may copy ahead into the slot pools:
    // layers.size(), or one fewer while no call of the last layer runs.
    std::size_t pooled_end = ~std::size_t{0};
    std::vector<SequenceState*> verified; // the last verify()'s sequences, in order
    // Their first columns and widths in that verification.
    std::vector<std::int32_t> verified_columns, verified_widths;
    // While forward_branches() runs: the state its branches start from. Every part of the pass is
    // a branch of one sequence, and each part after the first takes the snapshot's recurrent state
    // of a layer before it runs that layer.
    const SequenceSnapshot* branch_origin = nullptr;
    // While forward_prompt_branches() runs: the snapshot the pass's first part, the prompt's last
    // tokens, leaves its state in, a layer's as soon as that part has run it; branch_origin points
    // to it, and the branches start from it.
    SequenceSnapshot* branch_capture = nullptr;
    // The MTP chain on the head rank: token ids, positions (sequence indices) and 1-D RoPE
    // positions of a pass's cells (a draft step's at step * sequences), and for a catch-up over a
    // media prompt's images or video their three-axis RoPE positions ([3, cells]) and the cells
    // that take a Vision embedding, staged from the host.
    DeviceBuffer mtp_ids, mtp_positions, mtp_rope, mtp_axes, mtp_scatter, ngram_prefetch_tokens;
    DeviceBuffer mtp_logprobs;
    std::unique_ptr<PinnedHostBuffer> mtp_staging;
    // A sampled chain's samplers [sequences] and keys [draft_tokens][rows], and their staging.
    DeviceBuffer mtp_draft_configs, mtp_draft_keys;
    std::unique_ptr<PinnedHostBuffer> mtp_draft_staging;
    cudaEvent_t mtp_staged       = nullptr;
    std::uint32_t mtp_routes     = 0; // MTP route pairs recorded for the expert cache
    // While a draft chain is captured: its replays count no routes for the expert cache (the
    // catch-up passes after a commit still do).
    bool capturing_draft = false;
    // Masked-drafter speculative decoding (options.dflash2, or options.dspine for a standalone
    // DSpine drafter), on the head rank. DFlash2's operands until allocate_dflash2() hands them to
    // its drafter, the head as the drafter's final hidden reads it, DSpine's head-write table, and
    // each text layer's tap: its index among the drafter's features, or -1.
    std::optional<DFlash2Operands> dflash2_operands;
    MaskedDrafter drafter;
    Projection dflash2_head;
    DeviceBuffer dspine_table;
    std::vector<std::int32_t> tap_of;
    // What a pass leaves at the taps: the mean of a tapped layer's four streams of the pass
    // columns [first, first + count), at feature columns from `column` (BF16 [taps * hidden,
    // feature_columns]); count 0 takes none. feature_rows regroups a verification's kept columns
    // sequence by sequence for the ring.
    struct Capture {
        std::int32_t first = 0, count = 0, column = 0;
    };
    Capture capture;
    // Committed positions the ring takes: `count` feature columns from `column`, the first at
    // `position`, into the sequence's lane.
    struct ContextRun {
        SequenceState* sequence = nullptr;
        std::int32_t column     = 0;
        std::int32_t count      = 0;
        std::uint32_t position  = 0;
    };
    std::uint32_t feature_columns = 0;
    DeviceBuffer features, feature_rows;
    // Every sequence's context ring lane (its slot): each drafter layer's K and V of its last
    // window positions, which a proposal attends to.
    DeviceBuffer ring_backing;
    std::unique_ptr<CyclicKVCache> ring;
    // The proposal's and the context calls' device operands, greedy sampling configs, two pinned
    // staging halves that alternate between calls (each with the event of its last upload), and
    // the pinned drafts and confidences a proposal returns.
    struct DFlash2Io {
        std::int32_t *anchors = nullptr, *frontiers = nullptr, *valid = nullptr, *lanes = nullptr;
        std::int32_t *ids = nullptr, *positions = nullptr, *drafts = nullptr;
        std::int32_t *candidates = nullptr;
        float *q = nullptr, *confidence = nullptr;
        std::int32_t *context_positions = nullptr, *context_counts = nullptr;
        std::int32_t *context_lanes = nullptr;
    };
    DFlash2Io io;
    DeviceBuffer dflash2_io, dflash2_configs;
    std::array<std::unique_ptr<PinnedHostBuffer>, 2> dflash2_staging;
    std::array<cudaEvent_t, 2> dflash2_staged{};
    std::size_t dflash2_next = 0, dflash2_next_half = 0;
    std::unique_ptr<PinnedHostBuffer> dflash2_readback;
    // The CUDA Graphs of one sequence's DFlash2 calls (dflash2_replay): a proposal per block
    // width, a context call per run length.
    std::array<PassGraphs, 17> dflash2_proposals, dflash2_contexts;
    std::uint32_t pages          = 0; // KV pages per sequence and attention layer
    PagedKVStorageLayout kv_layout;   // the planes of every sparse-attention layer's KV
    std::uint32_t pooled_slots   = 0; // indexer blocks per sequence and attention layer
    // Streamed sparse-attention layers (options.kv_resident): each sequence's layer keeps
    // stream_slots pages of 4 positions on the device, from which calls of at most stream_width
    // tokens read; wider calls read the rank's staging copy. Zero: every layer resident.
    std::uint32_t stream_slots = 0, stream_width = 0;
    std::uint32_t max_logit_rows = 0;
    ExecutorMemory memory;
    std::vector<std::uint64_t> row_ids;
    std::vector<std::uint8_t> row_bytes_host;
    // Expert cache (host-resident experts): every layer's routes of the last pass, on its device
    // and in pinned host memory, and how many tokens they cover.
    std::unique_ptr<ExpertCache> cache;
    // The routing profile that filled the cache's slots; grow_expert_cache fills its new ones.
    std::optional<ExpertRouteCounts> cache_profile;
    std::unique_ptr<HybridExperts> hybrid;
    const std::atomic<bool>* cancelled = nullptr;
    // Disk-resident experts: made resident before each layer's experts run.
    std::unique_ptr<ExpertStream> stream;
    // GGUF host experts: a decode step's, verification's or short prompt chunk's missing experts,
    // staged (or computed by the CPU: up to kVectorTokens any, up to kCpuMissTokens those with
    // few pairs) while its cached ones run.
    std::unique_ptr<ExpertMisses> misses;
    std::uint32_t miss_tokens = 0; // the widest verification the misses take
    // The widest prompt chunk the misses take, 0 for none: a chunk this narrow stages the experts
    // the cache lacks as a verification does, instead of taking the slot pool back from the cache
    // and copying every routed expert the cache lacks into it.
    std::uint32_t prompt_miss_tokens = 0;
    // Each expert layer's routes of the current pass, [layer][10 * prefill_chunk] in one block per
    // rank (a stage's layers are consecutive), so one copy brings a rank's text layers home.
    std::vector<DeviceBuffer> route_blocks; // by rank
    std::vector<void*> route_records;       // by expert layer
    std::unique_ptr<PinnedHostBuffer> route_host;
    std::uint32_t pending_routes = 0;
    bool pending_decode          = true; // they came from a decode step or verification
    // The text layers whose routes the record holds: all, or all but a K/V-only last layer.
    std::size_t pending_route_layers = 0;

    // Tensor parallelism (Model::tensor_parallel()): every rank runs every layer over its replica
    // of the stack, with its run of each mixer's heads and of the routed experts, and exchanges
    // its partial of each mixer's output projection and of each MoE layer's experts
    // (run_tensor_segment). The plans of every rank after the first, [rank - 1][layer], and each
    // rank's token embedding (rank 0's is embedding_table) and copy of a media prompt's Vision
    // embeddings.
    std::vector<std::vector<LayerPlan>> rank_layers;
    std::vector<Weight> rank_embedding_tables;
    std::vector<DeviceBuffer> rank_vision_embeddings;
    std::unique_ptr<TensorParallelExchange> tensor_exchange;
    std::size_t exchange_next = 0; // the next exchange's slot within a pass
    // The divided head (run_divided_head): `head` holds rank 0's rows, and each rank after the
    // first projects its rows (rank_heads[rank - 1]) from its own final mixer's read of its stack
    // (rank_final_mixers) and sends them to rank 0 over its link (head_links), head_columns
    // columns at a time.
    std::vector<Projection> rank_heads;
    std::vector<HcPlan> rank_final_mixers;
    std::vector<StageLink> head_links;
    std::int32_t head_columns = 0;
    // The columns rank 0's whole head projects at once (run_whole_head): the most, halving from
    // max_logit_rows, that fit in the workspace rank 0 needs anyway, and at least head_columns.
    std::int32_t whole_columns = 0;
    // The MTP block's read of its stack, from rank 0 to each rank after the first, whose rows of
    // the head a draft projects too (run_divided_mtp_head).
    std::vector<StageLink> draft_links;

    // Host-resident experts, wide calls: a layer's routed experts that the cache does not hold are
    // copied into device slots and the matrix kernel reads them there; one pool and one set of
    // tables per rank, whose layers run one at a time.
    // The pool holds a bank per projection laid out as the host's: expert e at e times the
    // layer's matrix bytes, so experts adjacent on the host cross the bus in one copy.
    struct SlotPool {
        struct Buffer {
            DeviceBuffer storage; // gate, up and down banks, zeroed when allocated
            DeviceBuffer tables;  // gate, up and down tables of every expert
        };

        std::array<std::uint64_t, 3> offset{}; // of each bank in a buffer's storage
        std::uint32_t slots = 0;               // experts a buffer holds: all of a layer's, or none
        // One buffer, or two that consecutive layers alternate between, so that a wide call's
        // copies of the next layer run while this layer's experts do.
        std::vector<Buffer> buffers;
    };

    std::vector<SlotPool> slot_pools; // by rank; empty unless the experts are host resident
    // With the automatic expert cache the pool buffers and a span's stack (span_stack, which
    // only run_span writes) are cache slots between prompts (ExpertCache::lend): a wide pass takes
    // them back, the next decode pass returns them. lent_ranks: the ranks with a lent buffer.
    bool pools_lent = false;
    std::vector<unsigned char> lent_ranks;
    std::unique_ptr<PinnedHostBuffer> slot_entries;
    // Wide host calls of at least kPrefetchTokens touch nearly every expert of a layer: every
    // expert the cache lacks is copied into the pool on the transfer stream while the layers
    // before run, without waiting for the layer's routes. Two pinned entry tables alternate.
    // A layer's copies are hundreds of operations, and an enqueue blocks while the stream's queue
    // is full, so a thread of their own (pool_copier) issues them: the layers between keep being
    // launched.
    static constexpr std::int32_t kPrefetchTokens = 256;
    // The widest verification and prompt chunk the staged misses take (served_by_misses):
    // narrower than a prefetched group. A prompt chunk this narrow runs alone at its own width, as
    // device experts run it, while a wider one may join a span's group: ops::moe_experts_gguf
    // divides a call's experts between its two kernels only up to kCpuMissTokens tokens, so a
    // group gives the bits of its chunks.
    static constexpr std::uint32_t kMissTokens = 64;
    static_assert(kMissTokens < std::uint32_t(kPrefetchTokens) &&
                  kNarrowMissTokens == std::uint32_t(kVectorTokens) &&
                  kMissTokens == kCpuMissTokens);
    struct PoolPrefetch {
        // Per pool buffer: the layer whose experts it holds or receives, its copies' completion
        // and the completion of the last call that read it, and the copier's task that issues
        // the copies.
        struct Buffer {
            std::size_t layer = ~std::size_t{0};
            cudaEvent_t ready = nullptr, free = nullptr;
            std::future<void> issued;
        };

        std::array<Buffer, 2> buffers;
        // Pinned table sources: a copy reads its buffer when it runs, so a buffer is rewritten only
        // once the copy that last read it is done (`copied`); the host runs ahead of the device.
        std::array<std::unique_ptr<PinnedHostBuffer>, 2> entries;
        std::array<cudaEvent_t, 2> copied{};
        int next = 0;
    };
    std::vector<PoolPrefetch> prefetches; // by rank
    std::unique_ptr<HostWorkerPool> pool_copier;

    // Layer-major prompt spans (forward_span): the chunks' stacks, staged positions and n-gram
    // rows while every chunk of a span passes a layer before any passes the next. span_tokens is 0
    // when spans are off.
    static constexpr std::uint32_t kSpanChunks = 8;
    // An engine of one sequence keeps no other prompt waiting behind a step, so its spans run to
    // kLoneSpanChunks: a layer's uncached experts then cross the bus half as often in a prompt
    // past kSpanChunks chunks. The further chunks' planes come out of the expert cache (for good,
    // or while a prompt runs when the cache takes the stack between prompts), so they may take at
    // most 1 / kLoneSpanShare of the memory the device has free when the executor starts
    // (plan_span_chunks).
    static constexpr std::uint32_t kLoneSpanChunks = 16;
    static constexpr std::uint64_t kLoneSpanShare  = 24;
    std::uint32_t span_chunks = kSpanChunks;
    std::uint32_t span_tokens = 0;
    DeviceBuffer span_stack, span_positions, span_rope, span_rows;
    // With the Vision tower: a media prompt's three-axis positions, [3, chunk] per chunk.
    DeviceBuffer span_mrope;
    // A span's n-gram rows, read in one batch while its layers before the first PLE layer run.
    std::unique_ptr<PinnedHostBuffer> span_host_rows;
    std::vector<std::uint64_t> span_row_ids;
    // Each chunk's MoE input and hyper-connection injection between a layer's two phases.
    DeviceBuffer span_mixed, span_inject;
    std::size_t span_row_bytes = 0; // n-gram rows of one token
    // A span's expert phase runs up to kGroupTokens of consecutive chunks in one wide call, whose
    // matrix products then take wider column tiles: the group's routes and output. span_group is
    // the chunks a group takes, 1 when groups are off.
    static constexpr std::uint32_t kGroupTokens = 4096;
    std::uint32_t span_group = 1;
    DeviceBuffer span_ids, span_weights, span_shared, span_y;
    // A span's expert call that a wider call of the same layer follows: the pool keeps that
    // layer's experts.
    bool span_hold = false;
    // A span's last expert call of a layer: the disk cache releases the layer's experts the span's
    // earlier calls read and this one does not route to.
    bool span_last = false;

    // Vision (a model loaded with its tower), on rank 0 beside the token embedding: the encoder's
    // workspace, whose handoff region receives one item at a time, and the merged embeddings of
    // the media of the prompt a sequence prefills, item after item.
    std::optional<qwen3_5::execution::VisionParameters> vision_parameters;
    std::unique_ptr<qwen3_5::execution::VisionContext> vision_context;
    qwen3_5::execution::VisionWorkspacePlan vision_plan;
    DeviceBuffer vision_backing, vision_embeddings;

    Impl(const Model& m, DeviceContext& d, ExecutorOptions o)
        : model(m), device(d), options(std::move(o)), config(m.config()),
          ngram(derive_ngram_hash_constants(config.ngram)) {
        if (options.prefill_chunk == 0 || options.prefill_chunk > kMaxChunk || options.sequences == 0 ||
            options.max_context == 0) {
            throw std::invalid_argument("qwen4_exp executor: chunk must be in [1, 4096]");
        }
        if (options.ngram) {
            table = std::make_unique<NgramTableReader>(options.ngram->layout, options.ngram_read);
        }
        max_logit_rows = std::min<std::uint32_t>(options.prefill_chunk, 512);
        if (!std::isfinite(options.draft_min_p) || options.draft_min_p < 0 ||
            options.draft_min_p > 1 || (options.draft_min_p != 0 && options.draft_tokens == 0)) {
            throw std::invalid_argument(
                "qwen4_exp: draft_min_p needs drafts and a finite value in [0,1]");
        }
        if (options.dflash2 && options.dspine) {
            throw std::invalid_argument("qwen4_exp: one masked drafter, DFlash2 or DSpine");
        }
        const bool masked = options.dflash2 || options.dspine;
        if (masked && options.draft_tokens == 0) {
            throw std::invalid_argument("qwen4_exp: the masked drafter needs draft_tokens");
        }
        if (options.draft_tokens > 0) {
            if (options.dflash2 && !model.dflash2_weights()) {
                throw std::invalid_argument("qwen4_exp: DFlash2 drafts need a model loaded with "
                                            "its drafter");
            }
            if (options.dspine && model.dspine() == nullptr) {
                throw std::invalid_argument("qwen4_exp: DSpine drafts need a model loaded with "
                                            "its drafter (--draft-model)");
            }
            if (!masked && !model.mtp_weights()) {
                throw std::invalid_argument("qwen4_exp: drafts need a model loaded with its MTP "
                                            "block");
            }
            if (options.draft_tokens > 15) {
                throw std::invalid_argument("qwen4_exp: a speculative round drafts 1 to 15 tokens");
            }
            if (options.dspine &&
                options.draft_tokens + 1 > model.dspine()->config().block_size) {
                throw std::invalid_argument(
                    "qwen4_exp: the DSpine drafter proposes at most block_size - 1 (" +
                    std::to_string(model.dspine()->config().block_size - 1) + ") drafts a round");
            }
            if (masked && device.size() > 1 && !model.tensor_parallel()) {
                throw std::invalid_argument("qwen4_exp: the masked drafter runs on one device or "
                                            "under tensor parallelism, not over pipeline stages");
            }
            verify_width = options.draft_tokens + 1;
            if (verify_width * options.sequences > options.prefill_chunk) {
                throw std::invalid_argument(
                    "qwen4_exp: the prefill chunk must hold a verification of every sequence, (" +
                    std::to_string(verify_width) + " tokens) x " +
                    std::to_string(options.sequences));
            }
            max_logit_rows = std::max(max_logit_rows, verify_width * options.sequences);
            domain         = static_cast<std::int32_t>(model.resources().public_token_count);
        }
        pages          = (options.max_context + kPagedKVPageSize - 1) / kPagedKVPageSize;
        pooled_slots   = options.max_context / config.indexer_compress_ratio + 1;
        kv_layout =
            paged_kv_storage_layout(options.kv_cache, static_cast<std::int32_t>(config.head_dim));
        if (options.kv_resident != 0) {
            // A decode, verification or MTP call reads at most this many queries' blocks at once.
            const std::uint32_t width = std::max<std::uint32_t>(verify_width, 1);
            const std::uint32_t slots = std::max<std::uint32_t>(
                (options.kv_resident + ops::kKvStreamPageTokens - 1) / ops::kKvStreamPageTokens,
                std::uint32_t(ops::kv_stream_minimum_slots(std::int32_t(width))));
            if (std::uint64_t(slots) * ops::kKvStreamPageTokens < options.max_context) {
                stream_slots = slots;
                stream_width = width;
            }
        }
        if (model.tensor_parallel() && stream_slots != 0) {
            throw std::invalid_argument("qwen4_exp: tensor parallelism keeps every KV position on "
                                        "the devices (no --kv-resident)");
        }
        memory.ranks.resize(device.size());
        plan_span_chunks();
        plan_weights();
        if (verify_width > 0 && options.dflash2) {
            plan_dflash2();
        } else if (verify_width > 0 && options.dspine) {
            plan_dspine();
        } else if (verify_width > 0) {
            plan_mtp();
        }
        const bool native_host = model.options().experts == ExpertResidency::Host &&
            std::any_of(layers.begin(), layers.end(), [](const auto& layer) { return layer.moe.native.has_value(); });
        if (native_host && options.sequences != 1) {
            throw std::invalid_argument("native host experts require concurrency 1");
        }
        if (!native_host && options.hybrid_experts.adaptive_cache) {
            throw std::invalid_argument("--expert-cache-adaptive requires native host experts");
        }
        if (!native_host && model.options().experts != ExpertResidency::Host &&
            (!options.hybrid_experts.routing_profile.empty() ||
             !options.hybrid_experts.record_profile.empty())) {
            throw std::invalid_argument("expert routing profiles require host experts");
        }
        const auto residency = model.options().experts;
        if (residency == ExpertResidency::Device &&
            (options.hybrid_experts.dma_share || options.hybrid_experts.cpu_threads != 0 ||
             options.hybrid_experts.mapped_misses)) {
            throw std::invalid_argument("expert DMA shares, CPU threads and miss paths require "
                                        "host or disk experts");
        }
        if (options.hybrid_experts.mapped_misses &&
            (residency != ExpertResidency::Host || native_host ||
             options.hybrid_experts.dma_share.value_or(1.0F) != 1.0F)) {
            throw std::invalid_argument("mapped misses need GGUF host experts and all misses on the GPU");
        }
        if (options.hybrid_experts.tail_skip_rank != 0 &&
            (residency != ExpertResidency::Host || native_host ||
             options.hybrid_experts.mapped_misses)) {
            throw std::invalid_argument(
                "--expert-tail-skip needs GGUF host experts with staged misses");
        }
        if (options.hybrid_experts.prefill_buffers != 0 &&
            (residency != ExpertResidency::Host || native_host)) {
            throw std::invalid_argument("--expert-prefill-buffers needs GGUF host experts");
        }
        plan_segments();
        allocate_ranks();
        allocate_exchange();
        allocate_vision();
        allocate_sequences();
        allocate_staging();
        if (verify_width > 0) { allocate_speculation(); }
        allocate_span();
        if (dflash2_operands || (verify_width > 0 && options.dspine)) { allocate_drafter(); }
        allocate_cache();
        require_wide_route();
        memory.streamed_kv = stream_slots != 0;
        for (const auto& rank : memory.ranks) {
            memory.state_bytes += rank.state_bytes;
            memory.workspace_bytes += rank.workspace_bytes;
            memory.expert_cache_bytes += rank.expert_cache_bytes;
        }
    }

    ~Impl() {
        draft_prefetch.reset();
        // The copier's queued copies finish before the events and buffers they use go.
        for (auto& prefetch : prefetches) {
            for (auto& buffer : prefetch.buffers) {
                if (!buffer.issued.valid()) { continue; }
                try {
                    buffer.issued.get();
                } catch (...) {}
            }
        }
        pool_copier.reset();
        // A pass that failed between starting the row reads and waiting for them leaves them
        // writing into a staging buffer.
        if (table) {
            try {
                table->wait();
            } catch (...) {}
        }
        const auto reset_segments = [&](PassGraphs& shape) {
            for (std::size_t i = 0; i < shape.graphs.size(); ++i) {
                RankBinding bind(device, segments[i].rank);
                shape.graphs[i].reset();
            }
        };
        reset_segments(decode_graphs);
        for (auto& shape : verify_graphs) { reset_segments(shape); }
        const auto reset_chains = [&](auto& shapes) {
            for (auto& shape : shapes) {
                for (auto& graph : shape.graphs) {
                    RankBinding bind(device, model.head_rank());
                    graph.reset();
                }
            }
        };
        reset_chains(draft_graphs);
        reset_chains(step_graphs);
        reset_chains(sampled_draft_graphs);
        reset_chains(sampled_step_graphs);
        reset_chains(dflash2_proposals);
        reset_chains(dflash2_contexts);
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            if (rank.staged != nullptr) { cudaEventDestroy(rank.staged); }
            if (rank.done != nullptr) { cudaEventDestroy(rank.done); }
            if (rank.copied != nullptr) { cudaEventDestroy(rank.copied); }
            if (rank.routes != nullptr) { cudaEventDestroy(rank.routes); }
            if (rank.fork != nullptr) { cudaEventDestroy(rank.fork); }
            if (rank.join != nullptr) { cudaEventDestroy(rank.join); }
            if (rank.uploaded != nullptr) { cudaEventDestroy(rank.uploaded); }
            if (rank.side.fork != nullptr) { cudaEventDestroy(rank.side.fork); }
            if (rank.side.join != nullptr) { cudaEventDestroy(rank.side.join); }
            if (rank.side.stream != nullptr) { cudaStreamDestroy(rank.side.stream); }
        }
        if (mtp_staged != nullptr) {
            RankBinding bind(device, model.head_rank());
            cudaEventDestroy(mtp_staged);
        }
        for (cudaEvent_t staged : dflash2_staged) {
            if (staged == nullptr) { continue; }
            RankBinding bind(device, model.head_rank());
            cudaEventDestroy(staged);
        }
        for (const StallEvents& events : stall_events) {
            RankBinding bind(device, rows_rank);
            if (events.wanted != nullptr) { cudaEventDestroy(events.wanted); }
            if (events.ready != nullptr) { cudaEventDestroy(events.ready); }
        }
        for (std::size_t r = 0; r < prefetches.size(); ++r) {
            RankBinding bind(device, r);
            for (const auto& buffer : prefetches[r].buffers) {
                if (buffer.ready != nullptr) { cudaEventDestroy(buffer.ready); }
                if (buffer.free != nullptr) { cudaEventDestroy(buffer.free); }
            }
            for (cudaEvent_t copied : prefetches[r].copied) {
                if (copied != nullptr) { cudaEventDestroy(copied); }
            }
        }
    }

    [[nodiscard]] bool tensor() const noexcept { return model.tensor_parallel(); }

    // Tensor parallelism: the head rows rank `rank` projects.
    [[nodiscard]] std::uint64_t head_rows(std::size_t rank) const {
        const auto& rows = model.weights().head_rows;
        return rows.at(rank + 1) - rows.at(rank);
    }

    // A rank's plan and a sequence's state of layer `i`: under tensor parallelism every rank has
    // its own; otherwise the layer's stage's.
    [[nodiscard]] const LayerPlan& plan_of(std::size_t rank, std::size_t i) const {
        return rank == 0 || !tensor() ? layers[i] : rank_layers[rank - 1][i];
    }

    [[nodiscard]] LayerState& state_of(SequenceState& sequence, std::size_t rank,
                                       std::size_t i) const {
        return rank == 0 || !tensor() ? sequence.layers[i] : sequence.rank_layers[rank - 1][i];
    }

    // Calls f(rank, plan, state) for every rank's part of layer `i`: the layer's stage's, or under
    // tensor parallelism each rank's in rank order.
    template <class Sequence, class F>
    void each_part(Sequence& sequence, std::size_t i, F&& f) const {
        f(layers[i].rank, layers[i], sequence.layers[i]);
        for (std::size_t r = 1; r <= rank_layers.size(); ++r) {
            f(r, rank_layers[r - 1][i], sequence.rank_layers[r - 1][i]);
        }
    }

    // The ranks a pass runs a layer on.
    [[nodiscard]] std::size_t parts() const noexcept { return rank_layers.size() + 1; }

    // A sequence's layer states, layer by layer and within a layer rank by rank: a snapshot's
    // layers.
    [[nodiscard]] std::size_t state_count() const noexcept { return layers.size() * parts(); }

    // The snapshot layer of rank `rank`'s state of layer `i`.
    [[nodiscard]] std::size_t state_slot(std::size_t i, std::size_t rank) const noexcept {
        return i * parts() + (tensor() ? rank : 0);
    }

    // Whether `rank` runs a PLE layer: its stage's, or under tensor parallelism every rank.
    [[nodiscard]] bool ple_on(std::size_t rank) const {
        return std::any_of(layers.begin(), layers.end(), [&](const LayerPlan& p) {
            return p.ple && (p.rank == rank || tensor());
        });
    }

    void plan_weights() {
        const auto& w   = model.weights();
        embedding_table = native_weight(model.weight(w.token_embedding).view);
        {
            RankBinding bind(device, model.head_rank());
            head = make_projection(std::vector{
                model.input(tensor() ? w.head_shares[0] : w.output_head, "text/final_hidden")});
            if (tensor()) {
                whole_head =
                    make_projection(std::vector{model.input(w.output_head, "text/final_hidden")});
            }
            final_mixer = make_hc(model, w.final_mixer, config);
        }
        for (std::size_t r = 1; tensor() && r < device.size(); ++r) {
            RankBinding bind(device, r);
            rank_heads.push_back(
                make_projection(std::vector{model.input(w.head_shares[r], "text/final_hidden")}));
            rank_final_mixers.push_back(make_hc(model, w.rank_final_mixers[r - 1], config));
        }
        if (tensor()) {
            // A gather's columns fill at most kHeadGatherBytes of the widest share's rows.
            constexpr std::uint64_t kHeadGatherBytes = 16U << 20;
            std::uint64_t widest                     = 0;
            for (std::size_t r = 0; r < device.size(); ++r) {
                widest = std::max(widest, head_rows(r));
            }
            head_columns = static_cast<std::int32_t>(
                std::clamp<std::uint64_t>(kHeadGatherBytes / (widest * 2), 1, max_logit_rows));
        }
        for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
            layers.push_back(make_layer_plan(w.layers[i], model.stages().placement(i).stage,
                                             config.num_experts,
                                             tensor() ? &w.layers[i].moe : nullptr));
        }
        for (std::size_t r = 1; tensor() && r < device.size(); ++r) {
            rank_embedding_tables.push_back(native_weight(model.weight(model.embedding(r)).view));
            std::vector<LayerPlan> plans;
            for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
                plans.push_back(make_layer_plan(model.layer_weights(r, i), r, config.num_experts,
                                                &w.layers[i].moe));
            }
            rank_layers.push_back(std::move(plans));
        }
    }

    void plan_mtp() {
        const MtpWeights& w = *model.mtp_weights();
        RankBinding bind(device, model.head_rank());
        const auto h = static_cast<std::int32_t>(config.hidden_size);
        MtpPlan plan;
        plan.embedding_norm = direct(model, w.embedding_norm, {h});
        plan.hidden_norm =
            direct(model, w.hidden_norm, {static_cast<std::int32_t>(config.hc_count) * h});
        plan.fc_embedding = make_projection(model, {w.fc_embedding});
        plan.fc_hidden    = make_projection(model, {w.fc_hidden});
        const auto& weights = model.weights();
        plan.head           = make_projection(std::vector{model.input(
            tensor() ? weights.head_shares[0] : weights.output_head, "mtp/final_hidden")});
        for (std::size_t r = 1; tensor() && r < device.size(); ++r) {
            RankBinding on_rank(device, r);
            plan.rank_heads.push_back(make_projection(
                std::vector{model.input(weights.head_shares[r], "mtp/final_hidden")}));
        }
        plan.layer = make_layer_plan(w.layer, model.head_rank(), model.mtp_config()->num_experts);
        plan.final_mixer = make_hc(model, w.final_mixer, config);
        mtp              = std::move(plan);
    }

    // The DFlash2 drafter's operands on the head rank (it reads the token embedding there, which
    // is rank 0's: the drafter runs on one device or under tensor parallelism), its head, and the
    // text layers it taps.
    void plan_dflash2() {
        const DFlash2Weights& w = *model.dflash2_weights();
        const DFlash2Config& c  = *model.dflash2_config();
        RankBinding bind(device, model.head_rank());
        const auto h      = static_cast<std::int32_t>(config.hidden_size);
        const auto d      = static_cast<std::int32_t>(c.attention.head_dim);
        const auto linear = [&](WeightId id) { return ops::prepare_linear_weight(model.input(id)); };
        DFlash2Operands p;
        p.context_norm         = direct(model, w.context_norm, {h});
        p.final_norm           = direct(model, w.final_norm, {h});
        p.feature_projection   = linear(w.feature_projection);
        p.selector_projection  = linear(w.selector_projection);
        p.predecessor_codebook = native_weight(model.weight(w.predecessor_codebook).view);
        p.successor_codebook   = native_weight(model.weight(w.successor_codebook).view);
        p.embedding            = embedding_table;
        for (const DFlash2LayerWeights& lw : w.layers) {
            DFlash2LayerOperands layer;
            layer.input_norm           = direct(model, lw.input_norm, {h});
            layer.post_attention_norm  = direct(model, lw.post_attention_norm, {h});
            layer.query_norm           = direct(model, lw.query_norm, {d});
            layer.key_norm             = direct(model, lw.key_norm, {d});
            layer.attention_base       = direct(model, lw.attention_base, {h, 2, 2});
            layer.mlp_base             = direct(model, lw.mlp_base, {h, 2, 2});
            layer.query                = linear(lw.query);
            layer.key                  = linear(lw.key);
            layer.value                = linear(lw.value);
            layer.context_key          = linear(lw.context_key);
            layer.context_value        = linear(lw.context_value);
            layer.output               = linear(lw.output);
            layer.gate                 = linear(lw.gate);
            layer.up                   = linear(lw.up);
            layer.down                 = linear(lw.down);
            layer.attention_projection = linear(lw.attention_projection);
            layer.mlp_projection       = linear(lw.mlp_projection);
            p.layers.push_back(std::move(layer));
        }
        dflash2_head = make_projection(
            std::vector{model.input(model.weights().output_head, "dflash2/final_hidden")});
        tap_of.assign(layers.size(), -1);
        for (std::size_t t = 0; t < c.target_layer_ids.size(); ++t) {
            tap_of.at(c.target_layer_ids[t]) = static_cast<std::int32_t>(t);
        }
        dflash2_operands = std::move(p);
    }

    // The standalone DSpine drafter's head (the text's own use of the whole head on the head
    // rank) and the text layers it taps: the mean of a tapped layer's streams, as DFlash2's.
    void plan_dspine() {
        const dspine::Config& c = model.dspine()->config();
        RankBinding bind(device, model.head_rank());
        dflash2_head = make_projection(
            std::vector{model.input(model.weights().output_head, "text/final_hidden")});
        tap_of.assign(layers.size(), -1);
        for (std::size_t t = 0; t < c.target_layer_ids.size(); ++t) {
            tap_of.at(c.target_layer_ids[t]) = static_cast<std::int32_t>(t);
        }
    }

    // The feature columns a pass captures: a verification of every sequence, each sequence's
    // last window positions of the widest prompt call (a chunk, or a span packing every sequence's
    // prompt), whose context the ring takes, and a span's two windows of one sequence, before its
    // cut and at its end (Executor::forward).
    [[nodiscard]] std::uint32_t planned_feature_columns() const {
        const std::uint32_t window = drafter_window();
        const std::uint32_t span   = planned_span_tokens();
        const std::uint32_t prompt = std::max(options.prefill_chunk, span);
        return std::max({verify_width * options.sequences,
                         std::min(prompt, window * options.sequences), std::min(span, 2 * window)});
    }

    // A layer's operands on its rank; `experts` its routed experts (the MTP block may keep more
    // than an expert-pruned text model). Under tensor parallelism `banks` lists every rank's routed
    // experts of the layer, and the plan is the rank's share.
    LayerPlan make_layer_plan(const LayerWeights& lw, std::size_t rank, std::uint32_t experts,
                              const MoeWeights* banks = nullptr) {
        LayerPlan plan;
        plan.rank = rank;
        RankBinding bind(device, plan.rank);
        plan.attn_hc = make_hc(model, lw.attn_hc, config);
        plan.mlp_hc  = make_hc(model, lw.mlp_hc, config);
        const TensorShare share = banks != nullptr ? model.share(rank) : TensorShare{};
        if (const auto* g = std::get_if<GdnWeights>(&lw.mixer)) {
            GdnPlan gdn;
            gdn.key_heads = static_cast<std::int32_t>(
                banks != nullptr ? share.key_head_end - share.key_head_begin
                                 : config.linear_num_key_heads);
            gdn.value_heads = gdn.key_heads * static_cast<std::int32_t>(
                                                  config.linear_num_value_heads /
                                                  config.linear_num_key_heads);
            gdn.qkv             = make_projection(model, {g->query, g->key, g->value});
            gdn.z               = make_projection(model, {g->z});
            gdn.a               = make_projection(model, {g->a_projection});
            gdn.b               = make_projection(model, {g->b_projection});
            gdn.output          = make_projection(model, {g->output});
            const auto channels = static_cast<std::int32_t>(gdn.qkv.rows);
            gdn.convolution =
                direct(model, g->convolution,
                       {channels, static_cast<std::int32_t>(config.linear_conv_kernel_dim)});
            gdn.a_log   = direct(model, g->a_log, {gdn.value_heads}, DType::FP32);
            gdn.dt_bias = direct(model, g->dt_bias, {gdn.value_heads}, DType::FP32);
            gdn.norm =
                direct(model, g->norm, {static_cast<std::int32_t>(config.linear_value_head_dim)});
            plan.gdn = std::move(gdn);
        } else if (const auto* sparse = std::get_if<SparseAttentionWeights>(&lw.mixer)) {
            const auto& a = *sparse;
            QsaPlan qsa;
            qsa.kv_heads = static_cast<std::int32_t>(
                banks != nullptr ? share.kv_head_end - share.kv_head_begin
                                 : config.num_key_value_heads);
            qsa.query            = make_projection(model, {a.query});
            qsa.gate             = make_projection(model, {a.gate});
            qsa.key              = make_projection(model, {a.key});
            qsa.value            = make_projection(model, {a.value});
            qsa.index            = make_projection(model, {a.index_query, a.index_key});
            qsa.output           = make_projection(model, {a.output});
            const auto d         = static_cast<std::int32_t>(config.head_dim);
            const auto di        = static_cast<std::int32_t>(config.indexer_head_dim);
            qsa.query_norm       = direct(model, a.query_norm, {d});
            qsa.key_norm         = direct(model, a.key_norm, {d});
            qsa.index_query_norm = direct(model, a.index_query_norm, {di});
            qsa.index_key_norm   = direct(model, a.index_key_norm, {di});
            plan.qsa             = std::move(qsa);
        }
        // Without the n-gram table the injection adds nothing (an all-zero embedding projects
        // to zero keys and values), so the layer runs without it.
        if (lw.ple && options.ngram) {
            PlePlan ple;
            ple.key          = make_projection(model, {lw.ple->key});
            ple.value        = make_projection(model, {lw.ple->value});
            const auto width = static_cast<std::int32_t>(config.hc_count * config.hidden_size);
            ple.norm_key     = direct(model, lw.ple->norm_key, {width});
            ple.norm_query   = direct(model, lw.ple->norm_query, {width});
            ple.norm_conv    = direct(model, lw.ple->norm_conv, {width});
            ple.convolution =
                direct(model, lw.ple->convolution,
                       {static_cast<std::int32_t>(config.ple_conv_kernel_size), width});
            plan.ple = std::move(ple);
        }
        const auto h    = static_cast<std::int32_t>(config.hidden_size);
        plan.moe.router = direct(model, lw.moe.router, {h, static_cast<std::int32_t>(experts)});
        plan.moe.shared_gate = direct(model, lw.moe.shared_score, {h});
        plan.moe.iq1m_matrix = options.iq1m_matrix;
        if (banks != nullptr) {
            plan.moe.share           = true;
            plan.moe.expert_begin    = share.expert_begin;
            plan.moe.expert_end      = share.expert_end;
            plan.moe.gate            = make_rank_table(model, banks->gate, rank);
            plan.moe.up              = make_rank_table(model, banks->up, rank);
            plan.moe.down            = make_rank_table(model, banks->down, rank);
            plan.moe.device_resident   = true;
            plan.moe.shared_gate_table = make_table(model, std::span(&lw.moe.shared_gate, 1));
            plan.moe.shared_up_table   = make_table(model, std::span(&lw.moe.shared_up, 1));
            plan.moe.shared_down_table = make_table(model, std::span(&lw.moe.shared_down, 1));
            if (lw.moe.latent) {
                plan.moe.group_begin = lw.moe.latent->group_begin;
                plan.moe.group_end   = lw.moe.latent->group_end;
                plan.moe.latent      = make_latent_plan(*lw.moe.latent, experts,
                                                        plan.moe.group_begin, plan.moe.group_end);
            }
            return plan;
        }
        if (!lw.moe.gate.empty() &&
            !is_gguf(native_weight(model.weight(lw.moe.gate.front()).view).qtype)) {
            const auto width = static_cast<std::int32_t>(config.moe_intermediate_size);
            const auto shared_width = static_cast<std::int32_t>(config.shared_expert_intermediate_size);
            NativeMoePlan native;
            native.gate = make_native_table(model, lw.moe.gate, width, h);
            native.up = make_native_table(model, lw.moe.up, width, h);
            native.down = make_native_table(model, lw.moe.down, h, width);
            native.shared_gate = make_native_table(model, std::span(&lw.moe.shared_gate, 1), shared_width, h);
            native.shared_up = make_native_table(model, std::span(&lw.moe.shared_up, 1), shared_width, h);
            native.shared_down = make_native_table(model, std::span(&lw.moe.shared_down, 1), h, shared_width);
            native.experts = static_cast<std::int32_t>(lw.moe.gate.size());
            plan.moe.native = std::move(native);
            return plan;
        }
        if (lw.moe.latent) {
            plan.moe.latent =
                make_latent_plan(*lw.moe.latent, experts, 0, config.latent_moe->groups);
        }
        if (lw.moe.gate.empty()) {
            plan.moe.gate = make_located_table(lw.moe.located_gate);
            plan.moe.up   = make_located_table(lw.moe.located_up);
            plan.moe.down = make_located_table(lw.moe.located_down);
        } else {
            plan.moe.gate = make_table(model, lw.moe.gate);
            plan.moe.up   = make_table(model, lw.moe.up);
            plan.moe.down = make_table(model, lw.moe.down);
        }
        plan.moe.device_resident   = model.options().experts != ExpertResidency::Host;
        plan.moe.shared_gate_table = make_table(model, std::span(&lw.moe.shared_gate, 1));
        plan.moe.shared_up_table   = make_table(model, std::span(&lw.moe.shared_up, 1));
        plan.moe.shared_down_table = make_table(model, std::span(&lw.moe.shared_down, 1));
        return plan;
    }

    // A layer's latent operands on its device: the groups [begin, end) of its encoder and decoder
    // (all of them, or a tensor-parallel rank's), the labels and the multipliers.
    [[nodiscard]] LatentPlan make_latent_plan(const LatentWeights& l, std::uint32_t experts,
                                              std::uint32_t begin, std::uint32_t end) const {
        const auto& latent = *config.latent_moe;
        LatentPlan out;
        out.width  = static_cast<std::int32_t>(latent.width);
        out.groups = static_cast<std::int32_t>(latent.groups);
        out.encode = make_group_table(model, l.encode, latent.groups, begin, end);
        out.decode = make_group_table(model, l.decode, latent.groups, begin, end);
        out.labels = direct(model, l.labels, {static_cast<std::int32_t>(experts)}, DType::I32);
        if (l.expert_scales) {
            out.expert_scales = direct(model, *l.expert_scales,
                                       {3, static_cast<std::int32_t>(experts)}, DType::FP32);
        }
        if (l.column_scale) {
            out.column_scale = direct(model, *l.column_scale, {out.width}, DType::FP32);
        }
        return out;
    }

    void plan_segments() {
        std::vector<bool> ple_seen(device.size(), false);
        bool first_ple = true;
        Segment current{.rank = 0, .begin = 0, .end = 0, .embed = true};
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const std::size_t rank = layers[i].rank;
            const bool rows        = layers[i].ple.has_value() && !ple_seen[rank];
            if (rows) {
                ple_seen[rank] = true;
                if (std::exchange(first_ple, false)) { rows_rank = rank; }
            }
            if (rank != current.rank || rows) {
                if (current.end > current.begin || current.embed) { segments.push_back(current); }
                current = Segment{.rank = rank, .begin = i, .end = i, .rows = rows};
            }
            current.end = i + 1;
        }
        if (current.rank != model.head_rank()) {
            segments.push_back(current);
            current =
                Segment{.rank = model.head_rank(), .begin = layers.size(), .end = layers.size()};
        }
        current.head = true;
        segments.push_back(current);
        // Launch-sized pieces; a boundary writes the stack before the next layer reads it, the
        // rounding the fused write-read takes (hyper_connection_write_read).
        std::vector<Segment> pieces;
        for (const Segment& segment : segments) {
            Segment piece = segment;
            while (piece.end - piece.begin > kSegmentLayers) {
                Segment first = piece;
                first.end     = piece.begin + kSegmentLayers;
                first.head    = false;
                pieces.push_back(first);
                piece.begin = first.end;
                piece.embed = false;
                piece.rows  = false;
            }
            pieces.push_back(piece);
        }
        segments = std::move(pieces);
    }

    [[nodiscard]] std::size_t workspace_bytes(std::size_t rank) const {
        const std::int32_t t = static_cast<std::int32_t>(options.prefill_chunk);
        std::size_t bytes    = ops::hyper_connection_read_workspace_bytes(
            static_cast<std::int32_t>(config.hc_count),
            static_cast<std::int32_t>(config.hidden_size),
            static_cast<std::int32_t>(config.hc_lowrank), t);
        bytes = std::max(
            {bytes, ops::moe_experts_gguf_workspace_bytes(t),
             ops::moe_route_workspace_bytes(t, static_cast<std::int32_t>(config.num_experts))});
        if (tensor()) {
            // A rank's experts' fixed-point part, held beside the part's own workspace.
            bytes = std::max(bytes, ops::moe_experts_gguf_part_workspace_bytes(t) +
                                        round_up(std::uint64_t(config.hidden_size) * t * 8));
        }
        if (const std::uint32_t group = planned_group_chunks(); group > 1 && rank == 0) {
            bytes = std::max(bytes, ops::moe_experts_gguf_workspace_bytes(
                                        static_cast<std::int32_t>(group) * t));
        }
        std::vector<const LayerPlan*> planned;
        for (const auto& plan : layers) {
            if (plan.rank == rank) { planned.push_back(&plan); }
        }
        if (tensor() && rank > 0) {
            for (const auto& plan : rank_layers[rank - 1]) { planned.push_back(&plan); }
        }
        if (mtp && rank == model.head_rank()) {
            planned.push_back(&mtp->layer);
            bytes = std::max(
                {bytes,
                 ops::moe_route_workspace_bytes(
                     t, static_cast<std::int32_t>(model.mtp_config()->num_experts)),
                 projection_workspace(mtp->fc_embedding, t),
                 projection_workspace(mtp->fc_hidden,
                                      t * static_cast<std::int32_t>(config.hc_count)),
                 projection_workspace(mtp->head, static_cast<std::int32_t>(max_logit_rows)),
                 // A sampled draft step's sampler, beside the step's argmax pair for the prefetch.
                 options.sampled_drafts
                     ? ops::sampling_workspace_capacity_bytes(
                           static_cast<std::int32_t>(model.resources().public_token_count), 1,
                           static_cast<std::int32_t>(options.sequences)) +
                           round_up(std::uint64_t(options.sequences) * 4)
                     : 0,
                 // The MTP block's calls over cached experts only: their kept ids and weights.
                 ops::moe_experts_gguf_workspace_bytes(t) +
                     2 * ((std::size_t(config.num_experts_per_tok) * t * 4 + 255) / 256 * 256)});
        }
        for (const LayerPlan* layer : planned) {
            const LayerPlan& plan = *layer;
            if (plan.moe.native) {
                bytes = std::max(bytes, ops::moe_experts_native_workspace_bytes(t));
            }
            if (plan.moe.latent) {
                // The staged misses take a whole chunk; a single call, a span group's included,
                // a slice of at most kGgufLatentSliceTokens.
                const std::int32_t group =
                    rank == 0 ? static_cast<std::int32_t>(planned_group_chunks()) * t : t;
                const std::int32_t call  = std::min(group, ops::kGgufLatentSliceTokens);
                const auto& latent       = *plan.moe.latent;
                bytes = std::max({bytes,
                                  ops::moe_latent_gguf_workspace_bytes(t, latent.width, latent.groups),
                                  ops::moe_latent_gguf_workspace_bytes(call, latent.width,
                                                                       latent.groups)});
                if (tensor()) {
                    // A rank's groups' fixed-point part, held beside the part's own workspace.
                    bytes = std::max(bytes, ops::moe_latent_gguf_workspace_bytes(t, latent.width,
                                                                                 latent.groups) +
                                                round_up(std::uint64_t(config.hidden_size) * t * 8));
                }
            }
            if (plan.gdn) {
                bytes = std::max({bytes,
                                  projections_workspace({&plan.gdn->qkv, &plan.gdn->z,
                                                         &plan.gdn->a, &plan.gdn->b},
                                                        t),
                                  projection_workspace(plan.gdn->output, t)});
                bytes =
                    std::max(bytes, ops::gated_delta_net_workspace_capacity_bytes(
                                        static_cast<std::int32_t>(config.linear_num_key_heads),
                                        static_cast<std::int32_t>(config.linear_num_value_heads),
                                        true, 1, t));
            }
            if (plan.qsa) {
                bytes = std::max({bytes,
                                  projections_workspace({&plan.qsa->query, &plan.qsa->gate,
                                                         &plan.qsa->key, &plan.qsa->value,
                                                         &plan.qsa->index},
                                                        t),
                                  projection_workspace(plan.qsa->output, t)});
                bytes = std::max({bytes,
                                  ops::qsa_indexer_select_workspace_bytes(
                                      t, static_cast<std::int32_t>(pooled_slots)),
                                  ops::sparse_softmax_attention_workspace_bytes(t)});
            }
            if (plan.ple) {
                bytes = std::max({bytes,
                                  projections_workspace({&plan.ple->key, &plan.ple->value}, t),
                                  ops::ple_inject_workspace_bytes(t)});
            }
        }
        if (tensor()) {
            // A gather's columns of the rank's head rows, of the text or of a draft; rank 0 places
            // every rank's rows of them into the logits from one buffer.
            const Projection& own = rank == 0 ? head : rank_heads.at(rank - 1);
            std::size_t gather    = 0;
            for (std::size_t r = 0; rank == 0 && r < device.size(); ++r) {
                gather = std::max<std::size_t>(
                    gather, round_up(head_rows(r) * std::uint64_t(head_columns) * 2));
            }
            std::size_t projected = projection_workspace(own, head_columns);
            if (mtp) {
                projected = std::max(
                    projected,
                    projection_workspace(rank == 0 ? mtp->head : mtp->rank_heads.at(rank - 1),
                                         head_columns));
            }
            bytes = std::max(bytes, projected + gather);
            if (rank == 0) {
                bytes = std::max(bytes, projection_workspace(whole_head,
                                                             std::max(head_columns, whole_columns)));
            }
        } else if (rank == model.head_rank()) {
            bytes = std::max(bytes,
                             projection_workspace(head, static_cast<std::int32_t>(max_logit_rows)));
        }
        if (options.dspine && verify_width > 0 && rank == model.head_rank()) {
            const dspine::Model& spine = *model.dspine();
            const auto s               = static_cast<std::int32_t>(options.sequences);
            const auto w               = static_cast<std::int32_t>(verify_width);
            const std::uint32_t window = drafter_window();
            const auto prepare = dspine::Drafter::prepare_columns(spine, s, w);
            bytes              = std::max(
                {bytes,
                 dspine::Drafter::proposal_workspace_bytes(
                     spine, DType::BF16, window, options.max_context, w, s,
                     projection_workspace(dflash2_head, (w - 1) * s)),
                 dspine::Drafter::context_workspace_bytes(
                     spine, window, static_cast<std::int32_t>(planned_feature_columns()), 1),
                 dspine::Drafter::context_workspace_bytes(spine, window, w, s),
                 dspine::Drafter::prepare_workspace_bytes(
                     spine, DType::BF16, s, w, projection_workspace(dflash2_head, prepare))});
        }
        if (dflash2_operands && rank == model.head_rank()) {
            const auto masks = static_cast<std::int32_t>(options.draft_tokens * options.sequences);
            bytes = std::max(bytes, DFlash2Drafter::workspace_bytes(
                                        *dflash2_operands, *model.dflash2_config(),
                                        config.hidden_size, config.vocab_size,
                                        static_cast<std::int32_t>(options.sequences),
                                        static_cast<std::int32_t>(verify_width),
                                        static_cast<std::int32_t>(planned_feature_columns()),
                                        options.max_context,
                                        projection_workspace(dflash2_head, masks)));
        }
        return bytes + (16U << 20);
    }

    void allocate_ranks() {
        const std::uint64_t t     = options.prefill_chunk;
        // Prompt pieces (tensor_pieces) exchange FP32 prompt partials; BF16 and 8-bit partials
        // cross the bus fast enough that a whole chunk prefilled faster than its pieces.
        const bool pieces = tensor() && t > std::uint64_t(kTensorPieceColumns) &&
                            options.tensor_parallel_prompt_partials == TensorParallelPartials::Fp32;
        const std::uint64_t h     = config.hidden_size;
        const std::uint64_t width = std::uint64_t(config.hc_count) * h;
        const std::uint64_t plane = std::max<std::uint64_t>(width, 6144); // widest BF16 plane rows
        const std::uint64_t row_b = options.ngram ? ops::ngram_row_bytes(options.ngram->format) : 0;
        if (tensor()) {
            RankBinding bind(device, 0);
            const std::size_t room = workspace_bytes(0) - (16U << 20);
            whole_columns          = head_columns;
            for (auto c = static_cast<std::int32_t>(max_logit_rows); c > head_columns; c /= 2) {
                if (projection_workspace(whole_head, c) <= room) {
                    whole_columns = c;
                    break;
                }
            }
        }
        for (std::size_t r = 0; r < device.size(); ++r) {
            RankBinding bind(device, r);
            RankState rank;
            rank.rank                                            = r;
            rank.stream                                          = device.rank(r).stream;
            std::vector<std::pair<void**, std::uint64_t>> planes = {
                {reinterpret_cast<void**>(&rank.stack), width * t * 4},
                {reinterpret_cast<void**>(&rank.ids), t * 4},
                {reinterpret_cast<void**>(&rank.positions), t * 4},
                {reinterpret_cast<void**>(&rank.rope), t * 4},
                {reinterpret_cast<void**>(&rank.mrope), 3 * t * 4},
                {reinterpret_cast<void**>(&rank.scatter), t * 4},
                {&rank.mixed, h * t * 2},
                {reinterpret_cast<void**>(&rank.inject), std::uint64_t(config.hc_count) * t * 4},
                {&rank.y, h * t * 4},
                {&rank.mixer_out, pieces ? h * t * 2 : 0},
                {&rank.a, plane * t * 2},
                {&rank.b, plane * t * 2},
                {&rank.c, plane * t * 2},
                {&rank.d, plane * t * 2},
                {&rank.e, plane * t * 2},
                {&rank.f, plane * t * 2},
                {&rank.g, plane * t * 4},
                {reinterpret_cast<void**>(&rank.route_weights),
                 std::uint64_t(config.num_experts_per_tok) * t * 4},
                {reinterpret_cast<void**>(&rank.route_shared), t * 4},
                {reinterpret_cast<void**>(&rank.selected),
                 std::uint64_t(config.indexer_block_budget()) * t * 4},
                {reinterpret_cast<void**>(&rank.counts), t * 4},
                {reinterpret_cast<void**>(&rank.route_ids),
                 std::uint64_t(config.num_experts_per_tok) * t * 4},
                {&rank.rows, std::uint64_t(config.ngram_heads()) * t * row_b},
            };
            if (r == model.head_rank()) {
                planes.push_back(
                    {&rank.logits, std::uint64_t(config.vocab_size) * max_logit_rows * 2});
            } else if (tensor()) {
                // The rank's head rows of a gather's columns.
                planes.push_back({&rank.logits, head_rows(r) * std::uint64_t(head_columns) * 2});
            }
            std::uint64_t total = 0;
            for (const auto& [slot, bytes] : planes) { total += round_up(bytes); }
            rank.activations     = DeviceBuffer(total);
            std::uint64_t offset = 0;
            for (const auto& [slot, bytes] : planes) {
                *slot = static_cast<std::byte*>(rank.activations.p) + offset;
                offset += round_up(bytes);
            }
            // Without prompt pieces (tensor_pieces) there is no exchanged mixer output plane.
            if (!pieces) {
                rank.mixer_out = nullptr;
            }
            rank.workspace = std::make_unique<DeviceArena>(workspace_bytes(r));
            std::vector<std::int32_t> identity(std::max<std::uint32_t>(pages, options.sequences));
            std::iota(identity.begin(), identity.end(), 0);
            rank.block_table = DeviceBuffer(std::size_t(pages) * 4);
            rank.block_table.copy_from_host(identity.data(), std::size_t(pages) * 4);
            rank.slot_ids = DeviceBuffer(std::size_t(options.sequences) * 4);
            rank.slot_ids.copy_from_host(identity.data(), std::size_t(options.sequences) * 4);
            rank.route_done = DeviceBuffer(4);
            rank.route_done.fill(0);
            // ids, positions, RoPE positions, scatter columns, three-axis RoPE positions, rows.
            rank.staging = std::make_unique<PinnedHostBuffer>(
                round_up(t * 28) + round_up(std::uint64_t(config.ngram_heads()) * t * row_b));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.staged, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.done, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.copied, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.routes, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.fork, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.join, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.uploaded, cudaEventDisableTiming));
            CUDA_CHECK(cudaStreamCreateWithFlags(&rank.side.stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.side.fork, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.side.join, cudaEventDisableTiming));
            memory.ranks[r].workspace_bytes += total + rank.workspace->capacity();
            ranks.push_back(std::move(rank));
        }
    }

    // Tensor parallelism: the exchange of a pass's partials, one FP32 [hidden, prefill_chunk]
    // per rank and slot.
    void allocate_exchange() {
        if (!tensor()) { return; }
        const std::size_t bytes = std::size_t(config.hidden_size) * options.prefill_chunk * 4;
        tensor_exchange         = std::make_unique<TensorParallelExchange>(
            device, TensorParallelExchangeOptions{.slot_bytes = bytes,
                                                  .transport  = options.tensor_parallel_transport,
                                                  .rounded_partials =
                                                      options.tensor_parallel_prompt_partials});
        for (std::size_t r = 0; r < device.size(); ++r) {
            memory.ranks[r].workspace_bytes += tensor_exchange->device_bytes();
        }
        for (std::size_t r = 1; r < device.size(); ++r) {
            head_links.emplace_back(
                device, r, 0,
                StageLinkOptions{.slot_bytes = head_rows(r) * std::size_t(head_columns) * 2});
            if (mtp) {
                draft_links.emplace_back(
                    device, 0, r,
                    StageLinkOptions{.slot_bytes =
                                         std::size_t(config.hidden_size) * options.sequences * 2});
            }
        }
    }

    void allocate_vision() {
        const auto& vision = model.vision_config();
        if (!vision) { return; }
        if (options.vision_max_merged_tokens == 0) {
            throw std::invalid_argument("qwen4_exp: Vision needs a positive merged-token limit");
        }
        RankBinding bind(device, 0);
        vision_parameters.emplace(vision_parameters_of(*model.vision_weights()));
        vision_plan = qwen3_5::execution::VisionContext::plan_workspace(
            *vision, *vision_parameters, options.vision_max_merged_tokens, 1);
        if (vision_plan.output_hidden != static_cast<std::int32_t>(config.hidden_size)) {
            throw std::invalid_argument("qwen4_exp: the Vision merger's width differs from the "
                                        "text model's");
        }
        vision_backing    = DeviceBuffer(vision_plan.capacity_bytes);
        vision_embeddings = DeviceBuffer(std::size_t(config.hidden_size) *
                                         options.vision_max_merged_tokens * 2);
        vision_context    = std::make_unique<qwen3_5::execution::VisionContext>(
            device, *vision, *vision_parameters, ranks[0].stream);
        memory.ranks[0].workspace_bytes += vision_backing.bytes + vision_embeddings.bytes;
        for (std::size_t r = 1; tensor() && r < device.size(); ++r) {
            RankBinding peer(device, r);
            rank_vision_embeddings.emplace_back(vision_embeddings.bytes);
            memory.ranks[r].workspace_bytes += vision_embeddings.bytes;
        }
    }

    [[nodiscard]] qwen3_5::execution::VisionParameters
    vision_parameters_of(const qwen3_5::VisionWeights& w) const {
        return vision_parameters_for(model, w);
    }

    std::size_t set_media(std::uint32_t s, std::span<const MediaItem> items,
                          std::vector<std::int32_t> rope_positions, std::int32_t rope_delta) {
        if (!vision_context) {
            throw std::invalid_argument("the model was loaded without its Vision tower");
        }
        SequenceState& sequence = sequences.at(s);
        if (rope_positions.empty() || rope_positions.size() % 3) {
            throw std::invalid_argument("qwen4_exp: RoPE positions must cover three axes");
        }
        const std::size_t tokens = rope_positions.size() / 3;
        if (sequence.position >= tokens) {
            throw std::invalid_argument("qwen4_exp: media start before the prompt's last token");
        }
        std::vector<std::int32_t> columns(tokens, -1);
        RankBinding bind(device, 0);
        const DeviceSpan backing{vision_backing.p, vision_backing.bytes};
        const std::size_t h = config.hidden_size;
        std::size_t column  = 0, encoded = 0;
        for (const MediaItem& item : items) {
            const auto& control = *item.control;
            // An item wholly inside the prefix the sequence holds is in its state already.
            const auto& scatter = control.scatter_indices;
            if (scatter.empty() || std::uint32_t(scatter.back()) < sequence.position) { continue; }
            if (column + control.merged_count > options.vision_max_merged_tokens) {
                throw std::invalid_argument("the prompt's media exceed the Vision merged-token "
                                            "limit (--vision-max-merged)");
            }
            Tensor output = qwen3_5::execution::VisionContext::bind_output(backing, vision_plan,
                                                                           control.merged_count);
            vision_context->encode(qwen3_5::execution::VisionItemView{item.patches, &control},
                                   output, backing, vision_plan);
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(vision_embeddings.p) +
                                           column * h * 2,
                                       output.data, output.bytes(), cudaMemcpyDeviceToDevice,
                                       ranks[0].stream));
            for (std::size_t j = 0; j < control.scatter_indices.size(); ++j) {
                const std::int32_t at = control.scatter_indices[j];
                if (at < 0 || std::size_t(at) >= tokens) {
                    throw std::invalid_argument("qwen4_exp: a media token lies outside the prompt");
                }
                columns[std::size_t(at)] = static_cast<std::int32_t>(column + j);
            }
            column += control.merged_count;
            ++encoded;
        }
        // A draft's first cell rotates at the last prompt token's position on one axis, which a
        // media token does not have: a prompt that ends in media does not draft with MTP.
        if (mtp && columns.back() >= 0) { sequence.drafter_follows = false; }
        sequence.media_columns = std::move(columns);
        sequence.media_rope    = std::move(rope_positions);
        sequence.rope_delta    = rope_delta;
        CUDA_CHECK(ninfer::wait_for_stream(ranks[0].stream));
        // Under tensor parallelism every rank embeds the prompt itself, from its own copy.
        for (std::size_t r = 1; r <= rank_vision_embeddings.size(); ++r) {
            RankBinding peer(device, r);
            CUDA_CHECK(cudaMemcpyPeerAsync(rank_vision_embeddings[r - 1].p, device.rank(r).device,
                                           vision_embeddings.p, device.rank(0).device,
                                           column * h * 2, ranks[r].stream));
            CUDA_CHECK(ninfer::wait_for_stream(ranks[r].stream));
        }
        return encoded;
    }

    // The planes of the sparse-attention layers' KV storage, in LayerState::kv's order (K, V,
    // K scales, V scales), each a row width and dtype; absent scale planes have width 0.
    [[nodiscard]] std::array<std::pair<std::int32_t, DType>, 4> kv_planes() const {
        return {{{kv_layout.key.data_leading_extent, kv_layout.key.data_dtype},
                 {kv_layout.value.data_leading_extent, kv_layout.value.data_dtype},
                 {kv_layout.key.has_scale() ? kv_layout.key.scale_leading_extent : 0,
                  kv_layout.key.scale_dtype},
                 {kv_layout.value.has_scale() ? kv_layout.value.scale_leading_extent : 0,
                  kv_layout.value.scale_dtype}}};
    }

    // One plane's bytes for `pages` pages of `page_tokens` positions of `heads` KV heads (both,
    // or a tensor-parallel rank's).
    [[nodiscard]] std::uint64_t kv_plane_bytes(std::size_t plane, std::uint64_t page_tokens,
                                               std::uint64_t count, std::uint64_t heads = 0) const {
        const auto [leading, dtype] = kv_planes()[plane];
        return std::uint64_t(leading) * page_tokens *
               (heads != 0 ? heads : config.num_key_value_heads) * count * dtype_size(dtype);
    }

    // A view of a sparse-attention layer's KV planes `data(i)`: pages of `page_tokens` positions,
    // `count` of them, read through `table` (`entries` I32).
    template <class Data>
    [[nodiscard]] PagedKVLayerView kv_view(Data&& data, std::int32_t page_tokens,
                                           std::int32_t count, void* table,
                                           std::int32_t entries, std::int32_t heads = 0) const {
        if (heads == 0) { heads = static_cast<std::int32_t>(config.num_key_value_heads); }
        PagedKVLayerView cache{};
        Tensor* targets[4] = {&cache.k_pages, &cache.v_pages, &cache.k_scale_pages,
                              &cache.v_scale_pages};
        const auto planes = kv_planes();
        for (std::size_t i = 0; i < planes.size(); ++i) {
            if (planes[i].first == 0) { continue; }
            *targets[i] =
                Tensor(data(i), planes[i].second, {planes[i].first, page_tokens, heads, count});
        }
        cache.block_table  = Tensor(table, DType::I32, {entries});
        cache.head_dim     = static_cast<std::int32_t>(config.head_dim);
        cache.num_kv_heads = heads;
        cache.storage      = options.kv_cache;
        return cache;
    }

    // A sparse-attention layer's per-sequence state: paged KV (one plane per K or V vector and
    // per scale, each [leading, page, head, page]), pooled keys and the indexer's tail. A
    // streamed layer keeps the paged KV in pinned host memory, and its pool and map on the device.
    void allocate_attention_state(LayerState& state, std::size_t rank, std::uint32_t heads = 0) {
        std::uint64_t device_kv = 0, bytes = 0;
        const auto planes       = kv_planes();
        for (std::size_t i = 0; i < planes.size(); ++i) {
            if (planes[i].first == 0) { continue; }
            KvPlane& plane = state.kv[i];
            plane.bytes    = kv_plane_bytes(i, kPagedKVPageSize, pages, heads);
            if (stream_slots != 0) {
                plane.host    = std::make_unique<PinnedHostBuffer>(plane.bytes);
                plane.p       = plane.host->data();
                state.pool[i] = DeviceBuffer(kv_plane_bytes(i, ops::kKvStreamPageTokens,
                                                            stream_slots));
                device_kv += state.pool[i].bytes;
                memory.host_kv_bytes += plane.bytes;
            } else {
                plane.device = DeviceBuffer(plane.bytes);
                plane.p      = plane.device.p;
                device_kv += plane.bytes;
            }
        }
        if (stream_slots != 0) {
            const auto blocks = static_cast<std::int32_t>(
                std::uint64_t(pages) * (kPagedKVPageSize / ops::kKvStreamPageTokens));
            const auto slots  = static_cast<std::int32_t>(stream_slots);
            state.map_memory  = DeviceBuffer(ops::kv_stream_map_bytes(blocks, slots));
            state.map         = ops::kv_stream_map(state.map_memory.p, blocks, slots);
            bytes += state.map_memory.bytes;
        }
        state.pooled = DeviceBuffer(std::uint64_t(config.indexer_head_dim) * pooled_slots * 4);
        state.tail   = DeviceBuffer(std::uint64_t(config.indexer_head_dim) *
                                    (config.indexer_compress_ratio - 1) * 4);
        memory.ranks[rank].state_bytes += device_kv + bytes + state.pooled.bytes + state.tail.bytes;
        memory.kv_bytes += device_kv;
    }

    // Streamed layers: each rank that runs one gets the staging copy a prompt chunk reads.
    void allocate_staging() {
        if (stream_slots == 0) { return; }
        for (RankState& rank : ranks) {
            const bool attention =
                std::any_of(layers.begin(), layers.end(), [&](const LayerPlan& plan) {
                    return plan.qsa && plan.rank == rank.rank;
                }) ||
                (mtp && rank.rank == model.head_rank());
            if (!attention) { continue; }
            RankBinding bind(device, rank.rank);
            const auto planes = kv_planes();
            for (std::size_t i = 0; i < planes.size(); ++i) {
                if (planes[i].first == 0) { continue; }
                rank.kv_staging[i] = DeviceBuffer(kv_plane_bytes(i, kPagedKVPageSize, pages));
                memory.ranks[rank.rank].workspace_bytes += rank.kv_staging[i].bytes;
            }
        }
    }

    // Before a prompt chunk of a streamed layer: the rank's staging copy holds the layer's
    // positions before `first`, copied from its host copy unless the copy is still there from the
    // layer's previous chunk.
    void stage(const LayerState& state, RankState& rank, std::uint32_t first) {
        if (rank.kv_staged == &state && rank.kv_staged_end >= first) { return; }
        const std::uint64_t used = (std::uint64_t(first) + kPagedKVPageSize - 1) / kPagedKVPageSize;
        for (std::size_t i = 0; i < state.kv.size(); ++i) {
            const KvPlane& plane = state.kv[i];
            if (plane.p == nullptr || used == 0) { continue; }
            CUDA_CHECK(cudaMemcpyAsync(rank.kv_staging[i].p, plane.p, plane.bytes / pages * used,
                                       cudaMemcpyDefault, rank.stream));
        }
        rank.kv_staged = &state;
    }

    // Each rank's Gated DeltaNet layers keep their state in one pool with a slot per sequence,
    // the layout the replay fold of a speculative commit reads.
    void allocate_gdn_pools() {
        gdn_local.assign(layers.size(), 0);
        std::vector<std::uint32_t> count(ranks.size(), 0);
        std::vector<const GdnPlan*> geometry(ranks.size(), nullptr);
        for (std::size_t i = 0; i < layers.size(); ++i) {
            if (!layers[i].gdn) { continue; }
            gdn_local[i] = count[layers[i].rank]++;
            for (std::size_t r = 0; r < ranks.size(); ++r) {
                if (plan_of(r, i).rank == r) { geometry[r] = &*plan_of(r, i).gdn; }
            }
        }
        // Under tensor parallelism every rank pools its run of every layer, at rank 0's indices.
        if (tensor()) { std::fill(count.begin(), count.end(), count[0]); }
        for (RankState& rank : ranks) {
            if (count[rank.rank] == 0) { continue; }
            RankBinding bind(device, rank.rank);
            const GdnPlan& g = *geometry[rank.rank];
            const LinearAttentionStatePoolSpec spec{
                .layers         = count[rank.rank],
                .conv_channels  = g.qkv.rows,
                .conv_width     = static_cast<std::int32_t>(config.linear_conv_kernel_dim - 1),
                .value_heads    = g.value_heads,
                .value_head_dim = static_cast<std::int32_t>(config.linear_value_head_dim),
                .key_head_dim   = static_cast<std::int32_t>(config.linear_key_head_dim),
                .slot_count     = static_cast<std::int32_t>(options.sequences),
            };
            LayoutBuilder builder;
            const auto layout = plan_linear_attention_state_pool(builder, spec);
            rank.gdn_backing  = DeviceBuffer(builder.finish(256, "Gated DeltaNet state"));
            rank.gdn          = std::make_unique<LinearAttentionStatePool>(
                DeviceSpan{rank.gdn_backing.p, rank.gdn_backing.bytes}, layout);
            memory.ranks[rank.rank].state_bytes += rank.gdn_backing.bytes;
        }
    }

    void allocate_sequences() {
        const std::uint64_t width = std::uint64_t(config.hc_count) * config.hidden_size;
        allocate_gdn_pools();
        // One rank's state of layer i for sequence s.
        const auto layer_state = [&](const LayerPlan& plan, std::size_t i, std::uint32_t s) {
            RankBinding bind(device, plan.rank);
            LayerState state;
            if (plan.gdn) {
                const auto slot = static_cast<std::int32_t>(s);
                state.ssm       = ranks[plan.rank].gdn->recurrent_slot(gdn_local[i], slot);
                state.conv      = ranks[plan.rank].gdn->conv_slot(gdn_local[i], slot);
            }
            if (plan.qsa) {
                allocate_attention_state(state, plan.rank,
                                         static_cast<std::uint32_t>(plan.qsa->kv_heads));
            }
            if (plan.ple) {
                state.history = DeviceBuffer(width * (config.ple_conv_kernel_size - 1) *
                                             config.ngram.ngram_size * 4);
                memory.ranks[plan.rank].state_bytes += state.history.bytes;
            }
            return state;
        };
        for (std::uint32_t s = 0; s < options.sequences; ++s) {
            SequenceState sequence;
            sequence.slot = s;
            for (std::size_t i = 0; i < layers.size(); ++i) {
                sequence.layers.push_back(layer_state(layers[i], i, s));
            }
            for (const auto& plans : rank_layers) {
                std::vector<LayerState> states;
                for (std::size_t i = 0; i < plans.size(); ++i) {
                    states.push_back(layer_state(plans[i], i, s));
                }
                sequence.rank_layers.push_back(std::move(states));
            }
            if (mtp) {
                const std::size_t head = model.head_rank();
                RankBinding bind(device, head);
                allocate_attention_state(sequence.mtp, head);
                sequence.mtp_pending = DeviceBuffer(width * 4);
                memory.ranks[head].state_bytes += sequence.mtp_pending.bytes;
            }
            sequences.push_back(std::move(sequence));
            reset(s);
        }
    }

    // What a speculative round needs beyond the sequences' state: on every rank the replay records
    // of its Gated DeltaNet layers with their fold, its sparse-attention layers' indexer
    // projections, the PLE layer's normalised convolution input and scratch states; on the head
    // rank the MTP chain's tokens and positions.
    void allocate_speculation() {
        const std::uint64_t columns = std::uint64_t(verify_width) * options.sequences;
        const std::uint64_t index_rows =
            std::uint64_t(config.indexer_n_heads + 1) * config.indexer_head_dim;
        const std::uint64_t stack_width = std::uint64_t(config.hc_count) * config.hidden_size;
        qsa_local.assign(layers.size(), 0);
        std::vector<std::uint32_t> attention(ranks.size(), 0);
        for (std::size_t i = 0; i < layers.size(); ++i) {
            if (layers[i].qsa) { qsa_local[i] = attention[layers[i].rank]++; }
        }
        // Under tensor parallelism each attention rank records every sparse-attention layer at
        // rank 0's indices.
        for (std::size_t r = 1; tensor() && r < ranks.size(); ++r) {
            attention[r] = r < model.attention_ranks() ? attention[0] : 0;
        }
        for (RankState& rank : ranks) {
            RankBinding bind(device, rank.rank);
            std::uint64_t bytes = 0;
            if (rank.gdn) {
                const auto& pool = rank.gdn->spec();
                const GdnReplayRecordSpec spec{
                    .layers          = static_cast<std::int32_t>(pool.layers),
                    .record_capacity = static_cast<std::int32_t>(options.sequences),
                    .width           = static_cast<std::int32_t>(verify_width),
                    .conv_channels   = pool.conv_channels,
                    .qk_heads        = static_cast<std::int32_t>(
                        std::uint64_t(pool.value_heads) * config.linear_num_key_heads /
                        config.linear_num_value_heads),
                    .value_heads     = pool.value_heads,
                    .key_dim         = pool.key_head_dim,
                    .value_dim       = pool.value_head_dim,
                };
                LayoutBuilder builder;
                const auto layout   = plan_gdn_replay_records(builder, spec);
                rank.record_backing = DeviceBuffer(builder.finish(256, "GDN replay records"));
                rank.records.emplace(DeviceSpan{rank.record_backing.p, rank.record_backing.bytes},
                                     layout);
                rank.fold.emplace(*rank.records, rank.gdn->all_layers_view());
                rank.conv_scratch = DeviceBuffer(std::uint64_t(pool.conv_channels) *
                                                 (config.linear_conv_kernel_dim - 1) * 2);
                bytes += rank.record_backing.bytes + rank.conv_scratch.bytes;
            }
            if (attention[rank.rank] > 0) {
                rank.index_records = DeviceBuffer(attention[rank.rank] * index_rows * columns * 2);
                rank.tail_scratch  = DeviceBuffer(std::uint64_t(config.indexer_head_dim) *
                                                  (config.indexer_compress_ratio - 1) * 4);
                bytes += rank.index_records.bytes + rank.tail_scratch.bytes;
            }
            if (ple_on(rank.rank)) {
                rank.ple_records = DeviceBuffer(stack_width * columns * 4);
                bytes += rank.ple_records.bytes;
            }
            memory.ranks[rank.rank].workspace_bytes += bytes;
        }
        const std::size_t head = model.head_rank();
        RankBinding bind(device, head);
        if (mtp) {
            const std::uint64_t cells = std::max<std::uint64_t>(
                options.prefill_chunk, std::uint64_t(verify_width) * options.sequences);
            mtp_ids       = DeviceBuffer(cells * 4);
            mtp_positions = DeviceBuffer(cells * 4);
            mtp_rope      = DeviceBuffer(cells * 4);
            mtp_axes      = DeviceBuffer(cells * 3 * 4);
            mtp_scatter   = DeviceBuffer(cells * 4);
            mtp_staging   = std::make_unique<PinnedHostBuffer>(cells * 7 * 4);
            if (options.draft_min_p > 0) {
                mtp_logprobs =
                    DeviceBuffer(std::uint64_t(options.draft_tokens) * options.sequences * 4);
            }
            if (options.sampled_drafts) {
                const std::uint64_t config_bytes =
                    options.sequences * sizeof(ops::SamplingConfig);
                const std::uint64_t key_bytes =
                    std::uint64_t(options.draft_tokens) * options.sequences * 4;
                mtp_draft_configs = DeviceBuffer(config_bytes);
                mtp_draft_keys    = DeviceBuffer(key_bytes);
                mtp_draft_staging = std::make_unique<PinnedHostBuffer>(config_bytes + key_bytes);
            }
            CUDA_CHECK(cudaEventCreateWithFlags(&mtp_staged, cudaEventDisableTiming));
        }
        if (table && options.ngram_read.residency != NgramResidency::Ram) {
            draft_prefetch = std::make_unique<NgramDraftPrefetch>(
                device.rank(head), *table, ngram, config.eos_token_id, config.vocab_size,
                options.sequences, options.draft_tokens);
            ngram_prefetch_tokens = DeviceBuffer(2 * std::uint64_t(options.sequences) *
                                                  verify_width * sizeof(std::int32_t));
        }
        // Each sequence's MTP tail as the draft steps advance it (a catch-up redoes those cells).
        for (SequenceState& sequence : sequences) {
            if (mtp) { sequence.mtp_scratch_tail = DeviceBuffer(sequence.mtp.tail.bytes); }
        }
        memory.ranks[head].workspace_bytes += mtp_ids.bytes + mtp_positions.bytes + mtp_rope.bytes +
                                              mtp_axes.bytes + mtp_scatter.bytes +
                                              mtp_logprobs.bytes + ngram_prefetch_tokens.bytes +
                                              mtp_draft_configs.bytes + mtp_draft_keys.bytes +
                                              sequences.size() * sequences.front().mtp.tail.bytes;
    }

    // Bytes of one region of the DFlash2 operands that holds `count` words.
    [[nodiscard]] static std::size_t io_region(std::size_t count) { return round_up(count * 4); }

    // The masked drafter on the head rank: its context ring (every sequence a lane, zeroed once,
    // so that a lane's slots hold finite values before its positions reach them), the captured
    // features, the proposal's and context calls' operands with their staging, and the drafter
    // (DFlash2's, or DSpine's with the head-write table it derives from the head here).
    void allocate_drafter() {
        const std::size_t head = model.head_rank();
        RankBinding bind(device, head);
        const std::size_t s    = options.sequences;
        const std::size_t k    = options.draft_tokens;
        const std::size_t w    = verify_width;
        const DrafterGeometry g = drafter_geometry();
        const std::size_t rows = std::size_t(g.taps) * std::size_t(config.hidden_size);
        feature_columns        = planned_feature_columns();
        features               = DeviceBuffer(rows * feature_columns * 2);
        feature_rows           = DeviceBuffer(rows * w * s * 2);
        LayoutBuilder builder;
        const auto layout = plan_cyclic_kv_cache(builder, g.layers, drafter_window(), g.kv_heads,
                                                 g.head_dim, static_cast<std::int32_t>(s));
        ring_backing = DeviceBuffer(builder.finish(256, "masked drafter context ring"));
        ring_backing.fill(0);
        ring = std::make_unique<CyclicKVCache>(DeviceSpan{ring_backing.p, ring_backing.bytes},
                                               layout);
        // Proposal: anchors, frontiers, live widths and lanes (one region each, staged together),
        // ids and positions [W, S], drafts [K, S], candidates and their probabilities [16, K, S],
        // confidences [K, S]; context calls: positions, then counts and lanes (staged together).
        const std::size_t r = io_region(s);
        const std::array<std::size_t, 13> regions{
            r, r, r, r, io_region(w * s), io_region(w * s), io_region(k * s),
            io_region(16 * k * s), io_region(16 * k * s), io_region(k * s),
            io_region(feature_columns), r, r};
        std::size_t total = 0;
        for (const std::size_t bytes : regions) { total += bytes; }
        dflash2_io = DeviceBuffer(total);
        std::array<std::byte*, 13> at{};
        std::size_t offset = 0;
        for (std::size_t i = 0; i < regions.size(); ++i) {
            at[i] = static_cast<std::byte*>(dflash2_io.p) + offset;
            offset += regions[i];
        }
        const auto words = [&](std::size_t i) { return reinterpret_cast<std::int32_t*>(at[i]); };
        io = {.anchors           = words(0),
              .frontiers         = words(1),
              .valid             = words(2),
              .lanes             = words(3),
              .ids               = words(4),
              .positions         = words(5),
              .drafts            = words(6),
              .candidates        = words(7),
              .q                 = reinterpret_cast<float*>(at[8]),
              .confidence        = reinterpret_cast<float*>(at[9]),
              .context_positions = words(10),
              .context_counts    = words(11),
              .context_lanes     = words(12)};
        // Greedy drafts: the default sampling config draws the highest edge.
        const std::vector<ops::SamplingConfig> greedy(s);
        dflash2_configs = DeviceBuffer(s * sizeof(ops::SamplingConfig));
        dflash2_configs.copy_from_host(greedy.data(), dflash2_configs.bytes);
        const std::size_t half = 4 * r + io_region(feature_columns) + 2 * r;
        for (std::size_t i = 0; i < 2; ++i) {
            dflash2_staging[i] = std::make_unique<PinnedHostBuffer>(half);
            CUDA_CHECK(cudaEventCreateWithFlags(&dflash2_staged[i], cudaEventDisableTiming));
        }
        dflash2_readback = std::make_unique<PinnedHostBuffer>(2 * k * s * 4);
        std::vector<CyclicKVCacheLayerView> views;
        for (std::uint32_t l = 0; l < ring->layer_count(); ++l) {
            views.push_back(ring->layer_view(l));
        }
        const auto head_projection = [this](const Tensor& hidden, Tensor& logits,
                                            WorkspaceArena& ws, cudaStream_t stream) {
            project(dflash2_head, hidden, logits, ws, stream);
        };
        if (options.dspine) {
            const dspine::Config& c = model.dspine()->config();
            dspine_table            = DeviceBuffer(dspine::head_write_bytes(c));
            dspine::Target target{
                .hidden = config.hidden_size,
                .vocab  = config.vocab_size,
                .layers = config.num_hidden_layers,
                .domain = domain,
                .logits = DType::BF16,
                .embed  = [this](const Tensor& ids, Tensor& rows, cudaStream_t stream) {
                    ops::embedding(ids, embedding_table, rows, stream);
                },
                .head = head_projection};
            auto spine = std::make_unique<dspine::Drafter>(
                *model.dspine(), std::move(target),
                Tensor(dspine_table.p, DType::BF16,
                       {static_cast<std::int32_t>(c.rank), static_cast<std::int32_t>(c.vocab_size)}),
                drafter_window(), options.max_context, static_cast<std::int32_t>(s),
                static_cast<std::int32_t>(w));
            RankState& rank = ranks[head];
            spine->prepare(*rank.workspace, rank.stream);
            CUDA_CHECK(ninfer::wait_for_stream(rank.stream));
            drafter = MaskedDrafter(std::move(spine), std::move(views));
            memory.ranks[head].state_bytes += dspine_table.bytes;
        } else {
            drafter = MaskedDrafter(std::make_unique<DFlash2Drafter>(
                std::move(*dflash2_operands), *model.dflash2_config(), config.hidden_size,
                config.vocab_size, domain, std::move(views), options.max_context,
                head_projection));
            dflash2_operands.reset();
        }
        memory.ranks[head].state_bytes += ring_backing.bytes;
        memory.ranks[head].workspace_bytes +=
            features.bytes + feature_rows.bytes + dflash2_io.bytes + dflash2_configs.bytes;
    }

    // The masked drafter's ring and feature geometry: DFlash2's config, or the DSpine drafter's.
    struct DrafterGeometry {
        std::uint32_t layers = 0, taps = 0;
        std::int32_t kv_heads = 0, head_dim = 0;
    };
    [[nodiscard]] DrafterGeometry drafter_geometry() const {
        if (options.dspine) {
            const dspine::Config& c = model.dspine()->config();
            return {c.num_hidden_layers, c.taps(), static_cast<std::int32_t>(c.num_key_value_heads),
                    static_cast<std::int32_t>(c.head_dim)};
        }
        const DFlash2Config& c = *model.dflash2_config();
        return {c.num_hidden_layers, static_cast<std::uint32_t>(c.target_layer_ids.size()),
                static_cast<std::int32_t>(c.attention.num_key_value_heads),
                static_cast<std::int32_t>(c.attention.head_dim)};
    }

    // The ring's capacity and the drafter's attention window: DFlash2's trained window, or the
    // narrowest DSpine window that sees the whole context (dspine::ring_capacity).
    [[nodiscard]] std::uint32_t drafter_window() const {
        return options.dspine ? dspine::ring_capacity(model.dspine()->config(), options.max_context)
                              : *model.dflash2_config()->sliding_window;
    }

    // The layers whose experts the route records, the expert cache and the expert stream serve:
    // the text layers, then the MTP block's.
    [[nodiscard]] std::vector<std::pair<const LayerPlan*, const MoeWeights*>>
    expert_layers() const {
        std::vector<std::pair<const LayerPlan*, const MoeWeights*>> out;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            out.emplace_back(&layers[i], &model.weights().layers[i].moe);
        }
        if (mtp) { out.emplace_back(&mtp->layer, &model.mtp_weights()->layer.moe); }
        return out;
    }

    void allocate_cache() {
        const std::uint64_t pairs =
            std::uint64_t(config.num_experts_per_tok) * options.prefill_chunk;
        const auto expert = expert_layers();
        std::vector<std::size_t> on_rank(device.size(), 0);
        for (const auto& [plan, moe] : expert) { ++on_rank[plan->rank]; }
        route_blocks.resize(device.size());
        for (std::size_t r = 0; r < device.size(); ++r) {
            if (on_rank[r] == 0) { continue; }
            RankBinding bind(device, r);
            route_blocks[r] = DeviceBuffer(on_rank[r] * pairs * sizeof(std::int32_t));
        }
        std::fill(on_rank.begin(), on_rank.end(), 0);
        for (const auto& [plan, moe] : expert) {
            route_records.push_back(static_cast<std::byte*>(route_blocks[plan->rank].p) +
                                    on_rank[plan->rank]++ * pairs * sizeof(std::int32_t));
        }
        route_host =
            std::make_unique<PinnedHostBuffer>(expert.size() * pairs * sizeof(std::int32_t));
        const ExpertResidency residency = model.options().experts;
        const bool native_host = residency == ExpertResidency::Host &&
            std::any_of(expert.begin(), expert.end(), [](const auto& item) { return item.first->moe.native.has_value(); });
        if (native_host) {
            std::vector<HybridExpertLayer> host_layers;
            hybrid_layers.resize(expert.size());
            for (std::size_t index = 0; index < expert.size(); ++index) {
                const auto& [plan, moe] = expert[index];
                if (!plan->moe.native) { continue; }
                hybrid_layers[index] = host_layers.size();
                const auto& native = *plan->moe.native;
                HybridExpertLayer layer;
                layer.rank = plan->rank;
#if defined(__linux__)
                layer.registered = true; // native Host bindings use Residency::Registered
#endif
                for (std::int32_t e = 0; e < native.experts; ++e) {
                    layer.experts.push_back({{native.gate.operands[e], native.gate.integer_a8},
                                             {native.up.operands[e], native.up.integer_a8},
                                             {native.down.operands[e], native.down.integer_a8}});
                }
                host_layers.push_back(std::move(layer));
            }
            if (!options.hybrid_experts.routing_profile.empty()) {
                std::vector<std::size_t> widths;
                for (const auto& layer : host_layers) { widths.push_back(layer.experts.size()); }
                auto counts = read_expert_profile(options.hybrid_experts.routing_profile,
                                                  model.info().artifact_id, widths);
                for (std::size_t layer = 0; layer < host_layers.size(); ++layer) {
                    host_layers[layer].route_counts = std::move(counts[layer]);
                }
            }
            hybrid = std::make_unique<HybridExperts>(device, std::move(host_layers),
                options.prefill_chunk, options.expert_cache_bytes, options.hybrid_experts);
            for (std::size_t r = 0; r < ranks.size(); ++r) {
                memory.ranks[r].expert_cache_bytes += hybrid->cache_bytes(r);
                memory.ranks[r].workspace_bytes += hybrid->working_bytes(r);
            }
            return;
        }
        if (residency == ExpertResidency::Host) { allocate_slot_pools(); }
        if (residency == ExpertResidency::Device ||
            (residency == ExpertResidency::Host && options.expert_cache_bytes == 0)) {
            return;
        }
        // What each rank lends: the request split evenly, or what is free less a margin for the
        // allocations later startup makes (sampling, CUDA context growth).
        constexpr std::uint64_t kMargin = 1536ULL << 20;
        std::vector<std::uint64_t> bytes(device.size(), 0);
        for (std::size_t r = 0; r < device.size(); ++r) {
            RankBinding bind(device, r);
            std::size_t free_bytes = 0, total = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total));
            const std::uint64_t available = free_bytes > kMargin ? free_bytes - kMargin : 0;
            bytes[r] = options.expert_cache_bytes == ExecutorOptions::kAutoExpertCache
                           ? available
                           : std::min<std::uint64_t>(available,
                                                     options.expert_cache_bytes / device.size());
        }
        // Disk experts keep their per-layer residency until staged experts can enter its slots.
        const bool staged = residency == ExpertResidency::Host && !options.hybrid_experts.mapped_misses;
        if (staged) {
            for (auto& b : bytes) { b = b > kStagingBytes ? b - kStagingBytes : 0; }
        }
        // The automatic cache starts at half (the disk stream's pools at least at their widest
        // call): grow_expert_cache completes it once the warm-up and the rest of startup have
        // allocated what requests use, and can only add.
        const bool grows = options.expert_cache_bytes == ExecutorOptions::kAutoExpertCache;
        if (residency == ExpertResidency::Disk) {
            if (options.expert_cache_bytes == 0) {
                throw std::invalid_argument("disk-resident experts need a device expert cache");
            }
            // One call routes at most a chunk's tokens: a span's chunks prepare one by one. The MTP
            // block's calls are draft steps, a token for each of up to kVectorTokens sequences; its
            // slots hold one such step's experts, and one sequence's working set.
            const std::size_t widest_call =
                std::size_t(config.num_experts_per_tok) * options.prefill_chunk;
            const std::size_t widest_draft =
                std::size_t(config.num_experts_per_tok) * kVectorTokens;
            std::vector<ExpertStreamLayer> stream_layers;
            for (std::size_t i = 0; i < expert.size(); ++i) {
                const auto& [plan, moe] = expert[i];
                stream_layers.push_back(ExpertStreamLayer{
                    .rank        = plan->rank,
                    .stream      = ranks[plan->rank].stream,
                    .experts     = {moe->located_gate, moe->located_up, moe->located_down},
                    .tables      = {plan->moe.gate.table.p, plan->moe.up.table.p,
                                    plan->moe.down.table.p},
                    .widest_call = i >= layers.size() ? widest_draft : widest_call,
                    .own_pool    = i >= layers.size()});
            }
            stream = std::make_unique<ExpertStream>(
                device, model.files(), std::move(stream_layers), bytes, grows,
                ExpertDiskReads{options.hybrid_experts.mirror, options.hybrid_experts.direct});
            for (std::size_t r = 0; r < bytes.size(); ++r) {
                memory.ranks[r].expert_cache_bytes += stream->pool_bytes(r);
            }
            return;
        }
        if (grows) {
            for (auto& b : bytes) { b /= 2; }
        }
        std::vector<ExpertCacheLayer> cache_layers;
        for (const auto& [plan, moe] : expert) {
            const auto bank = [](const ExpertTable& table) {
                return ExpertBank{.expert_bytes = std::int64_t(table.rows) * table.row_bytes,
                                  .host         = table.pointers,
                                  .table        = table.table.p};
            };
            cache_layers.push_back(ExpertCacheLayer{.rank   = plan->rank,
                                                    .stream = ranks[plan->rank].stream,
                                                    .gate   = bank(plan->moe.gate),
                                                    .up     = bank(plan->moe.up),
                                                    .down   = bank(plan->moe.down)});
        }
        // The routing profile: the one given, else the one recorded on this family's release.
        std::vector<std::size_t> widths;
        for (const auto& [plan, moe] : expert) { widths.push_back(plan->moe.gate.pointers.size()); }
        auto& profile = cache_profile;
        if (!options.hybrid_experts.routing_profile.empty()) {
            profile = read_expert_profile(options.hybrid_experts.routing_profile,
                                          model.info().artifact_id, widths);
        } else {
            profile = builtin_expert_profile(
                config.latent_moe ? kLatentProfileFamily : kExpertProfileFamily, widths);
        }
        cache = std::make_unique<ExpertCache>(device, std::move(cache_layers), bytes,
                                              profile ? &*profile : nullptr,
                                              expert.size() > layers.size());
        for (std::size_t r = 0; r < bytes.size(); ++r) {
            memory.ranks[r].expert_cache_bytes += bytes[r];
        }
        if (profile) { cache->seed(*profile); }
        if (staged) { allocate_misses(); }
    }

    // A text layer's call of more than eight tokens over GGUF host experts runs from device memory
    // (ops::moe_experts_gguf): through the misses' rounds or the rank's slot pool. Mapped misses
    // and a 0 MiB expert cache have no staged misses, so every such rank needs its pool.
    void require_wide_route() const {
        if (model.options().experts != ExpertResidency::Host || misses) { return; }
        for (const auto& plan : layers) {
            if (plan.moe.native || plan.moe.device_resident) { continue; }
            if (plan.rank >= slot_pools.size() || slot_pools[plan.rank].slots == 0) {
                throw std::invalid_argument(
                    "GGUF host experts without staged misses (--expert-misses mapped or "
                    "--expert-cache-mib 0) need a slot pool for calls of more than eight tokens, "
                    "and half the device's free memory cannot hold one");
            }
        }
    }

    // Device staging of the missing experts, taken from the expert cache's share of each device.
    static constexpr std::uint64_t kStagingBytes = 128ULL << 20;

    void allocate_misses() {
        std::vector<MissLayer> out;
        const auto expert = expert_layers();
        for (std::size_t i = 0; i < expert.size(); ++i) {
            const auto& [plan, moe] = expert[i];
            MissLayer layer;
            layer.rank    = plan->rank;
            layer.experts = plan->moe.gate.pointers.size();
            const ExpertTable* tables[3] = {&plan->moe.gate, &plan->moe.up, &plan->moe.down};
            const std::span<const ExpertLocation> located[3] = {moe->located_gate, moe->located_up,
                                                                moe->located_down};
            for (int k = 0; k < 3; ++k) {
                auto& p        = layer.projections[std::size_t(k)];
                p.format       = tables[k]->format;
                p.row_bytes    = tables[k]->row_bytes;
                p.rows         = tables[k]->rows;
                // A latent layer's gate and up read its latent width (TextConfig::latent_moe).
                p.k            = std::int32_t(k == 2              ? config.moe_intermediate_size
                                              : plan->moe.latent ? std::uint32_t(plan->moe.latent->width)
                                                                 : config.hidden_size);
                p.expert_bytes = std::int64_t(p.rows) * p.row_bytes;
                if (stream) {
                    p.located = located[k];
                } else {
                    p.host = tables[k]->pointers;
                }
            }
            if (const auto& latent = plan->moe.latent) {
                layer.latent_width = latent->width;
                if (latent->expert_scales.data != nullptr) {
                    RankBinding bind(device, plan->rank);
                    layer.latent_scales.resize(std::size_t(latent->expert_scales.numel()));
                    CUDA_CHECK(cudaMemcpy(layer.latent_scales.data(), latent->expert_scales.data,
                                          layer.latent_scales.size() * sizeof(float),
                                          cudaMemcpyDeviceToHost));
                }
            }
            layer.gate_table = static_cast<const void* const*>(plan->moe.gate.table.p);
            // The text layers' slots follow their misses; the MTP block's keep the cache's own
            // policy.
            if (cache && !stream && i < layers.size()) {
                cache->hand_over(i);
                layer.admit = true;
            }
            layer.resident = resident_blocks(i);
            out.push_back(std::move(layer));
        }
        // Every decode step, and every verification up to the calls' limit, the joint one of all
        // sequences included; without the lossy tail skip, prompt chunks of up to kMissTokens
        // too. The CPU takes a share of the missing experts of calls up to kCpuMissTokens, in a
        // wide call of those with few pairs (ExpertMisses). A layer whose rank has no slot pool
        // (half the free memory could not hold one) has every verification and prompt chunk take
        // the misses, the tail skip included: their rounds bring the missing experts to the
        // device, as a pool would.
        MissOptions miss;
        miss.cpu_balance    = !options.hybrid_experts.dma_share && !options.hybrid_experts.mapped_misses;
        miss.cpu_share      = miss.cpu_balance ? 0.0 : 1.0 - double(options.hybrid_experts.dma_share.value_or(1.0F));
        miss.cpu_threads    = options.hybrid_experts.cpu_threads;
        miss.staging_bytes  = kStagingBytes;
        miss.tail_skip_rank = options.hybrid_experts.tail_skip_rank;
        miss.disk = ExpertDiskReads{options.hybrid_experts.mirror, options.hybrid_experts.direct};
        const std::uint32_t joint = verify_width * options.sequences;
        const bool pooled = std::all_of(layers.begin(), layers.end(), [&](const auto& plan) {
            return plan.rank < slot_pools.size() && slot_pools[plan.rank].slots != 0;
        });
        miss_tokens = std::max<std::uint32_t>(kVectorTokens, pooled ? std::min(kMissTokens, joint)
                                                                    : joint);
        prompt_miss_tokens = !pooled                    ? options.prefill_chunk
                             : miss.tail_skip_rank != 0 ? 0
                                                        : kMissTokens;
        miss.token_capacity = std::max(miss_tokens, prompt_miss_tokens);
        misses              = std::make_unique<ExpertMisses>(
            device, std::move(out),
            stream ? model.files() : std::vector<std::filesystem::path>{}, miss, cache.get());
        for (auto& rank : memory.ranks) { rank.workspace_bytes += kStagingBytes; }
    }

    // One slot per expert on every rank, sized for its layers' widest projections, when half the
    // free memory holds them, and a second such buffer when half holds both and two are wanted:
    // --expert-prefill-buffers 2, or by default with the automatic expert cache, which holds both
    // between prompts. The expert cache takes what is left.
    void allocate_slot_pools() {
        const std::size_t experts = config.num_experts;
        const std::uint32_t wanted =
            options.hybrid_experts.prefill_buffers != 0 ? options.hybrid_experts.prefill_buffers
            : options.expert_cache_bytes == ExecutorOptions::kAutoExpertCache ? 2
                                                                               : 1;
        slot_pools.resize(device.size());
        for (std::size_t r = 0; r < device.size(); ++r) {
            std::array<std::uint64_t, 3> widest{};
            for (const auto& plan : layers) {
                if (plan.rank != r) { continue; }
                const ExpertTable* tables[3] = {&plan.moe.gate, &plan.moe.up, &plan.moe.down};
                for (int k = 0; k < 3; ++k) {
                    widest[k] = std::max<std::uint64_t>(widest[k], std::uint64_t(tables[k]->rows) *
                                                                       tables[k]->row_bytes);
                }
            }
            if (widest[0] == 0) { continue; }
            SlotPool& pool = slot_pools[r];
            pool.offset    = {0, round_up(experts * widest[0]),
                              round_up(experts * widest[0]) + round_up(experts * widest[1])};
            RankBinding bind(device, r);
            std::size_t free_bytes = 0, total = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total));
            // Zeros follow the last down in its bank: the matrix kernel reads past a down.
            const std::uint64_t bytes = pool.offset[2] + round_up(experts * widest[2] + kSlotTail);
            if (bytes > free_bytes / 2) { continue; }
            pool.slots = static_cast<std::uint32_t>(experts);
            pool.buffers.resize(wanted == 2 && 2 * bytes <= free_bytes / 2 ? 2 : 1);
            for (auto& buffer : pool.buffers) {
                buffer.storage = DeviceBuffer(bytes);
                CUDA_CHECK(ninfer::fill_and_wait(buffer.storage.p, 0, buffer.storage.bytes));
                buffer.tables = DeviceBuffer(3 * experts * sizeof(void*));
                memory.ranks[r].workspace_bytes += buffer.storage.bytes + buffer.tables.bytes;
            }
        }
        slot_entries = std::make_unique<PinnedHostBuffer>(3 * experts * sizeof(void*));
        prefetches.resize(device.size());
        for (std::size_t r = 0; r < device.size(); ++r) {
            if (slot_pools[r].slots == 0) { continue; }
            RankBinding bind(device, r);
            auto& prefetch = prefetches[r];
            for (std::size_t b = 0; b < slot_pools[r].buffers.size(); ++b) {
                CUDA_CHECK(
                    cudaEventCreateWithFlags(&prefetch.buffers[b].ready, cudaEventDisableTiming));
                CUDA_CHECK(
                    cudaEventCreateWithFlags(&prefetch.buffers[b].free, cudaEventDisableTiming));
            }
            for (auto& entries : prefetch.entries) {
                entries = std::make_unique<PinnedHostBuffer>(3 * experts * sizeof(void*));
            }
            for (cudaEvent_t& copied : prefetch.copied) {
                CUDA_CHECK(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
                CUDA_CHECK(cudaEventRecord(copied, device.rank(r).transfer_stream));
            }
            if (!pool_copier) { pool_copier = std::make_unique<HostWorkerPool>(1, 4); }
        }
    }

    // Copies the experts of text layer `index` that `wanted` selects and the cache lacks into the
    // pool's banks, each at its own index, and points `entries` (the gate, up and down tables) at
    // the cache or the copies; unselected experts get none. While the layer's cache slots are not
    // all taken (a handed-over layer, ExpertCache::admit_free), such an expert goes into a free
    // slot instead and the cache's tables point at it behind the copy, so later calls find it
    // cached, as the staged misses leave theirs. A run of pool copies goes in one copy per
    // projection wherever the host's are adjacent, and zeros follow the run's last down, which the
    // matrix kernel reads past: inside a run the bytes after a down are the next expert's down, as
    // in a host bank.
    template <class Wanted>
    void stage_pool(const SlotPool& pool, const SlotPool::Buffer& buffer, std::size_t index,
                    Wanted&& wanted, const void** entries, cudaStream_t stream) {
        const MoePlan& m             = layers[index].moe;
        const std::size_t experts    = m.gate.pointers.size();
        const ExpertTable* tables[3] = {&m.gate, &m.up, &m.down};
        std::array<std::uint64_t, 3> bytes{};
        for (int k = 0; k < 3; ++k) {
            bytes[k] = std::uint64_t(tables[k]->rows) * tables[k]->row_bytes;
        }
        auto* storage     = static_cast<std::byte*>(buffer.storage.p);
        const auto cached = [&](std::size_t e) {
            return cache && cache->cached(index, 0, std::int32_t(e)) != nullptr;
        };
        const std::array<void*, 3> cache_tables =
            cache ? cache->tables(index) : std::array<void*, 3>{};
        const auto admitted = [&](std::size_t e) {
            ExpertCache::Admitted slot;
            if (!cache || !cache->admit_free(index, std::int32_t(e), slot)) { return false; }
            for (int k = 0; k < 3; ++k) {
                CUDA_CHECK(cudaMemcpyAsync(slot.slot[k], tables[k]->pointers[e],
                                           std::size_t(bytes[k]), cudaMemcpyHostToDevice, stream));
                entries[k * experts + e] = slot.slot[k];
                CUDA_CHECK(cudaMemcpyAsync(static_cast<const void**>(cache_tables[k]) + e,
                                           entries + k * experts + e, sizeof(void*),
                                           cudaMemcpyHostToDevice, stream));
            }
            return true;
        };
        std::size_t run = experts; // the first expert of the pending run of pool copies
        const auto flush = [&](std::size_t end) {
            if (run == experts) { return; }
            for (int k = 0; k < 3; ++k) {
                std::byte* bank = storage + pool.offset[k];
                for (std::size_t a = run; a < end;) {
                    std::size_t b = a + 1;
                    while (b < end && static_cast<const std::byte*>(tables[k]->pointers[b - 1]) +
                                              bytes[k] ==
                                          tables[k]->pointers[b]) {
                        ++b;
                    }
                    CUDA_CHECK(cudaMemcpyAsync(bank + a * bytes[k], tables[k]->pointers[a],
                                               std::size_t((b - a) * bytes[k]),
                                               cudaMemcpyHostToDevice, stream));
                    a = b;
                }
                for (std::size_t x = run; x < end; ++x) {
                    entries[k * experts + x] = bank + x * bytes[k];
                }
            }
            CUDA_CHECK(cudaMemsetAsync(storage + pool.offset[2] + end * bytes[2], 0, kSlotTail,
                                       stream));
            run = experts;
        };
        for (std::size_t e = 0; e < experts; ++e) {
            if (!wanted(e) || cached(e)) {
                flush(e);
                for (int k = 0; k < 3; ++k) {
                    entries[k * experts + e] =
                        wanted(e) ? cache->cached(index, k, std::int32_t(e)) : nullptr;
                }
                continue;
            }
            if (admitted(e)) {
                flush(e);
                continue;
            }
            if (run == experts) { run = e; }
        }
        flush(experts);
    }

    // The pool buffer that text layer `index`'s wide calls read: with two, consecutive layers
    // alternate.
    [[nodiscard]] std::size_t pool_buffer(std::size_t index) const {
        return slot_pools[layers[index].rank].buffers.size() == 2 ? index % 2 : 0;
    }

    // Whether text layer `index` exists on `rank` and runs its wide calls through the pool.
    [[nodiscard]] bool pooled_layer(std::size_t index, std::size_t rank) const {
        return index < layers.size() && index < pooled_end && layers[index].rank == rank &&
               !layers[index].moe.native;
    }

    // Has the copier copy every expert of text layer `index` that the cache lacks into its pool
    // buffer (issue_prefetch), which then receives the layer.
    void prefetch_layer(std::size_t index) {
        auto& target  = prefetches[layers[index].rank].buffers[pool_buffer(index)];
        target.layer  = index;
        target.issued = pool_copier->submit([this, index] { issue_prefetch(index); });
    }

    // Before a call waits for a buffer's ready event: the copier has recorded it. Rethrows the
    // copier's failure.
    static void await_prefetch(PoolPrefetch::Buffer& buffer) {
        if (buffer.issued.valid()) { buffer.issued.get(); }
    }

    // Waits until the copier has issued every queued layer.
    void drain_prefetches() {
        for (auto& prefetch : prefetches) {
            for (auto& buffer : prefetch.buffers) { await_prefetch(buffer); }
        }
    }

    // On the copier thread: once the last call that read layer `index`'s pool buffer is done with
    // it, copies every expert of the layer that the cache lacks into the buffer on the transfer
    // stream and points the buffer's tables at the cache or the copies. Waiting on the host keeps
    // copies that cannot run yet out of the stream's queue.
    void issue_prefetch(std::size_t index) {
        const LayerPlan& plan = layers[index];
        SlotPool& pool        = slot_pools[plan.rank];
        PoolPrefetch& state   = prefetches[plan.rank];
        const std::size_t b   = pool_buffer(index);
        auto& target          = state.buffers[b];
        RankBinding bind(device, plan.rank);
        const cudaStream_t t  = device.rank(plan.rank).transfer_stream;
        CUDA_CHECK(ninfer::wait_for_event(target.free));
        const MoePlan& m          = plan.moe;
        const std::size_t experts = m.gate.pointers.size();
        const auto source         = std::size_t(state.next);
        state.next ^= 1;
        CUDA_CHECK(ninfer::wait_for_event(state.copied[source]));
        auto* entries = static_cast<const void**>(state.entries[source]->data());
        stage_pool(pool, pool.buffers[b], index, [](std::size_t) { return true; }, entries, t);
        CUDA_CHECK(cudaMemcpyAsync(pool.buffers[b].tables.p, entries, 3 * experts * sizeof(void*),
                                   cudaMemcpyHostToDevice, t));
        CUDA_CHECK(cudaEventRecord(state.copied[source], t));
        CUDA_CHECK(cudaEventRecord(target.ready, t));
    }

    // Before a pass: a wide one takes the pool buffers back from the expert cache, a decode or
    // verification pass returns them.
    void place_pools(bool wide) {
        if (!pools_lent) { return; }
        for (std::size_t r = 0; r < lent_ranks.size(); ++r) {
            if (!lent_ranks[r]) { continue; }
            const cudaStream_t transfer = device.rank(r).transfer_stream;
            if (wide) {
                cache->reclaim(r, transfer);
            } else {
                cache->give_back(r, transfer);
            }
        }
    }

    // Forgets every prefetched layer, so the next wide call copies its own.
    void forget_prefetches() {
        drain_prefetches();
        for (auto& prefetch : prefetches) {
            for (auto& buffer : prefetch.buffers) { buffer.layer = ~std::size_t{0}; }
        }
    }

    // Feeds the last pass's routes to the expert cache and lets it swap experts, before the next
    // pass reads the tables.
    void settle_routes() {
        if (pending_routes == 0 && mtp_routes == 0) { return; }
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            CUDA_CHECK(ninfer::wait_for_event(rank.routes));
        }
        const std::uint64_t pairs =
            std::uint64_t(config.num_experts_per_tok) * options.prefill_chunk;
        const auto* host = static_cast<const std::int32_t*>(route_host->data());
        if (cache) {
            const std::size_t used = std::size_t(config.num_experts_per_tok) * pending_routes;
            // The tier learns from decode and verification passes; a prompt chunk, which routes
            // nearly every expert, only counts toward the statistics.
            const bool decode = pending_decode;
            if (pending_routes > 0) {
                for (std::size_t i = 0; i < pending_route_layers; ++i) {
                    cache->observe(i, std::span(host + i * pairs, used), decode);
                }
            }
            if (mtp_routes > 0) {
                cache->observe(layers.size(),
                               std::span(host + layers.size() * pairs,
                                         std::size_t(config.num_experts_per_tok) * mtp_routes),
                               true);
            }
            if (decode) { cache->adapt(); }
        }
        pending_routes = 0;
        pending_decode = true;
        mtp_routes     = 0;
    }

    // Brings the pass's routes to the host for the expert cache's counts; nothing else reads them.
    // `routed`: the text layers that routed the pass's tokens, all but a K/V-only last layer.
    void record_routes(std::uint32_t tokens, std::size_t routed) {
        if (!cache || warming) { return; }
        const std::uint64_t pairs =
            std::uint64_t(config.num_experts_per_tok) * options.prefill_chunk;
        auto* host = static_cast<std::int32_t*>(route_host->data());
        for (std::size_t i = 0; i < routed;) {
            std::size_t end = i;
            while (end < routed && layers[end].rank == layers[i].rank) { ++end; }
            RankBinding bind(device, layers[i].rank);
            CUDA_CHECK(cudaMemcpy2DAsync(host + i * pairs, pairs * 4, route_records[i], pairs * 4,
                                         std::size_t(config.num_experts_per_tok) * tokens * 4,
                                         end - i, cudaMemcpyDeviceToHost,
                                         ranks[layers[i].rank].stream));
            i = end;
        }
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            CUDA_CHECK(cudaEventRecord(rank.routes, rank.stream));
        }
        pending_routes       = tokens;
        pending_route_layers = routed;
        pending_decode       = tokens <= std::uint32_t(kVectorTokens) || verifying;
    }

    // The state regions a reset zeroes, a snapshot keeps and a restore puts back: the recurrent
    // state, the convolution window, the indexer's tail and the PLE history, where the layer has
    // them; null where it does not.
    static std::array<std::pair<void*, std::size_t>, 4> recurrent_regions(const LayerState& state) {
        return {{{state.ssm.data, state.ssm.data != nullptr ? state.ssm.bytes() : 0},
                 {state.conv.data, state.conv.data != nullptr ? state.conv.bytes() : 0},
                 {state.tail.p, state.tail.bytes},
                 {state.history.p, state.history.bytes}}};
    }

    void reset(std::uint32_t s) {
        auto& sequence    = sequences.at(s);
        if (std::find(verified.begin(), verified.end(), &sequence) != verified.end()) { verified.clear(); }
        sequence.position = 0;
        sequence.context  = NgramContext::sequence_start(ngram, config.eos_token_id);
        sequence.media_columns.clear();
        sequence.media_rope.clear();
        sequence.rope_delta  = 0;
        sequence.drafter_follows = true;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            each_part(sequence, i, [&](std::size_t r, const LayerPlan&, LayerState& state) {
                RankBinding bind(device, r);
                const cudaStream_t stream = ranks[r].stream;
                for (const auto& [data, bytes] : recurrent_regions(state)) {
                    if (data != nullptr) { CUDA_CHECK(cudaMemsetAsync(data, 0, bytes, stream)); }
                }
                if (state.pooled.p != nullptr) {
                    CUDA_CHECK(cudaMemsetAsync(state.pooled.p, 0, state.pooled.bytes, stream));
                }
            });
        }
        if (mtp) {
            RankBinding bind(device, model.head_rank());
            const cudaStream_t stream = ranks[model.head_rank()].stream;
            for (DeviceBuffer* buffer : {&sequence.mtp.pooled, &sequence.mtp.tail}) {
                CUDA_CHECK(cudaMemsetAsync(buffer->p, 0, buffer->bytes, stream));
            }
        }
        forget_streamed_kv(sequence);
    }

    // A sequence's streamed layers: every block leaves their pools, and no rank's staging copy
    // stands for them, after a reset or a loaded image rewrote what their positions hold.
    void forget_streamed_kv(SequenceState& sequence) {
        if (stream_slots == 0) { return; }
        const auto forget = [&](LayerState& state, std::size_t rank) {
            if (state.map.table == nullptr) { return; }
            RankBinding bind(device, rank);
            ops::kv_stream_reset(state.map, ranks[rank].stream);
            if (ranks[rank].kv_staged == &state) { ranks[rank].kv_staged = nullptr; }
        };
        for (std::size_t i = 0; i < layers.size(); ++i) {
            forget(sequence.layers[i], layers[i].rank);
        }
        if (mtp) { forget(sequence.mtp, model.head_rank()); }
    }

    // Moves the stack's first `tokens` columns from one rank to another, ordered after the
    // source's work. The source's next pass (the next prompt chunk, queued before the target gets
    // to this copy) rewrites the source's stack, so it waits until the copy has read it.
    void cross(std::size_t from, std::size_t to, std::int32_t tokens) {
        RankState& source = ranks[from];
        RankState& target = ranks[to];
        {
            RankBinding bind(device, from);
            CUDA_CHECK(cudaEventRecord(source.done, source.stream));
        }
        {
            RankBinding bind(device, to);
            CUDA_CHECK(cudaStreamWaitEvent(target.stream, source.done, 0));
            const std::size_t bytes =
                std::size_t(config.hc_count) * config.hidden_size * std::size_t(tokens) * 4;
            CUDA_CHECK(cudaMemcpyPeerAsync(target.stack, device.rank(to).device, source.stack,
                                           device.rank(from).device, bytes, target.stream));
            CUDA_CHECK(cudaEventRecord(target.copied, target.stream));
        }
        RankBinding bind(device, from);
        CUDA_CHECK(cudaStreamWaitEvent(source.stream, target.copied, 0));
    }

    // A pass's Vision work: the runs of consecutive embedding columns its tokens take (each run's
    // destination columns sit in the scatter plane at `offset`), and whether its one part rotates
    // at a media prompt's three-axis positions.
    struct MediaRun {
        std::int32_t source = 0, count = 0, offset = 0;
    };
    std::vector<MediaRun> media_runs;
    bool mrope = false;

    // Hints the table rows of `next`, the known tokens after the sequence's hashed ones, from a
    // copy of its hash context: forward() calls it once its own rows are read.
    void hint_next(const SequenceState& sequence, std::span<const std::int32_t> next) {
        if (next.empty() || !table || options.ngram_read.residency == NgramResidency::Ram ||
            options.ngram_read.io == NgramIo::Direct) { return; }
        NgramContext context = sequence.context;
        std::vector<std::uint64_t> rows(next.size() * config.ngram_heads());
        ngram_row_ids(ngram, next, config.eos_token_id, config.vocab_size, context, rows);
        table->prefetch(rows);
    }

    // The RoPE position of the sequence's position `at` on `axis`: within a media prompt the
    // prompt's own, past it the index plus the prompt's offset.
    [[nodiscard]] static std::int32_t rope_at(const SequenceState& sequence, std::size_t at,
                                              std::size_t axis) {
        const std::size_t prompt = sequence.media_rope.size() / 3;
        return at < prompt ? sequence.media_rope[axis * prompt + at]
                           : static_cast<std::int32_t>(at) + sequence.rope_delta;
    }

    // Stages a pass's inputs. With `read_rows` it also starts reading the tokens' n-gram rows into
    // the rows' staging (stage_rows uploads them); without, the rows' ids stay in row_ids.
    void stage_inputs(std::span<const Part> parts, std::span<const std::int32_t> tokens,
                      bool read_rows = true) {
        const auto t            = static_cast<std::int32_t>(tokens.size());
        const std::size_t c     = options.prefill_chunk;
        // Host side first: positions (sequence indices and RoPE positions), the n-gram rows of the
        // PLE layer's tokens, each part from its own sequence's position and n-gram context, and
        // where a media prompt's tokens take their embeddings and rotate.
        std::vector<std::int32_t> positions(tokens.size()), rope(tokens.size()), scatter;
        std::vector<std::int32_t> mrope_positions;
        std::size_t row_b       = 0;
        const std::size_t heads = config.ngram_heads();
        if (table) {
            row_ids.resize(tokens.size() * heads);
            row_b = ops::ngram_row_bytes(options.ngram->format);
        }
        media_runs.clear();
        mrope = false;
        for (const Part& part : parts) {
            const SequenceState& sequence = *part.sequence;
            const auto first = positions.begin() + part.column;
            const auto start = static_cast<std::int32_t>(sequence.position + part.offset);
            std::iota(first, first + part.count, start);
            for (std::int32_t i = 0; i < part.count; ++i) {
                rope[std::size_t(part.column + i)] = start + i + sequence.rope_delta;
            }
            if (table) {
                // A verification hashes from a copy, which commit() replaces by the context
                // advanced over the tokens it keeps. Each branch hashes from the origin's context,
                // which the sequence holds: the last branch advances it, the others a copy. A
                // pass's prompt part advances it first, and the origin takes it there.
                NgramContext branch_context;
                const bool opening = branch_capture != nullptr && &part == &parts.front();
                const bool copied  = branch_origin != nullptr && !opening && &part != &parts.back();
                if (copied) { branch_context = branch_origin->context; }
                NgramContext& context = copied      ? branch_context
                                        : verifying ? part.sequence->verify_context
                                                    : part.sequence->context;
                ngram_row_ids(ngram,
                              tokens.subspan(std::size_t(part.column), std::size_t(part.count)),
                              config.eos_token_id, config.vocab_size, context,
                              std::span(row_ids).subspan(std::size_t(part.column) * heads,
                                                         std::size_t(part.count) * heads));
                if (opening) { branch_capture->context = context; }
            }
            const std::size_t prompt = sequence.media_rope.size() / 3;
            if (prompt == 0 || sequence.position >= prompt) { continue; }
            if (parts.size() != 1) {
                throw std::logic_error("qwen4_exp: a media prompt prefills in a pass of its own");
            }
            mrope = true;
            mrope_positions.resize(3 * std::size_t(t));
            for (std::int32_t i = 0; i < t; ++i) {
                const std::size_t at = sequence.position + std::size_t(i);
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    mrope_positions[axis * std::size_t(t) + std::size_t(i)] =
                        rope_at(sequence, at, axis);
                }
                const std::int32_t column = at < prompt ? sequence.media_columns[at] : -1;
                if (column < 0) { continue; }
                if (!media_runs.empty() &&
                    media_runs.back().source + media_runs.back().count == column) {
                    ++media_runs.back().count;
                } else {
                    media_runs.push_back({.source = column,
                                          .count  = 1,
                                          .offset = static_cast<std::int32_t>(scatter.size())});
                }
                scatter.push_back(i);
            }
        }
        if (table) {
            // Rows a failed pass left reading go nowhere now.
            try {
                table->wait();
            } catch (...) {}
        }
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            // The staging buffer is rewritten only once the previous upload has left it.
            CUDA_CHECK(ninfer::wait_for_event(rank.staged));
            auto* base = static_cast<std::byte*>(rank.staging->data());
            const auto upload = [&](void* target, std::size_t at,
                                    const std::vector<std::int32_t>& values) {
                std::memcpy(base + at, values.data(), values.size() * 4);
                CUDA_CHECK(cudaMemcpyAsync(target, base + at, values.size() * 4,
                                           cudaMemcpyHostToDevice, rank.stream));
            };
            std::memcpy(base, tokens.data(), tokens.size() * 4);
            CUDA_CHECK(cudaMemcpyAsync(rank.ids, base, tokens.size() * 4, cudaMemcpyHostToDevice,
                                       rank.stream));
            upload(rank.positions, 4 * c, positions);
            upload(rank.rope, 8 * c, rope);
            if (!scatter.empty() && (rank.rank == 0 || tensor())) {
                upload(rank.scatter, 12 * c, scatter);
            }
            if (mrope) { upload(rank.mrope, 16 * c, mrope_positions); }
            if (table && read_rows && rank.rank == rows_rank) {
                // The rows are read while the layers before the PLE layer run; stage_rows()
                // uploads them and marks the staging buffer free.
                table->submit(row_ids, staged_rows(rank, row_ids.size() * row_b));
                rows_waited = false;
            }
            if (!ple_on(rank.rank) || !read_rows) {
                CUDA_CHECK(cudaEventRecord(rank.staged, rank.stream));
            }
        }
    }

    // The n-gram rows' place in a rank's staging buffer.
    std::span<std::uint8_t> staged_rows(RankState& rank, std::size_t bytes) const {
        auto* base = static_cast<std::byte*>(rank.staging->data());
        return {reinterpret_cast<std::uint8_t*>(
                    base + round_up(28 * std::uint64_t(options.prefill_chunk))),
                bytes};
    }

    // Before a segment that starts with its rank's first PLE layer: the rows stage_inputs() started
    // reading, uploaded to the rank once read. The pass's first upload waits for the read between
    // a pair of stall events.
    void stage_rows(RankState& rank) {
        const std::size_t bytes = row_ids.size() * ops::ngram_row_bytes(options.ngram->format);
        RankState& reader       = ranks[rows_rank];
        await_rows(reader);
        std::span<std::uint8_t> rows = staged_rows(reader, bytes);
        if (&rank != &reader) {
            const auto own = staged_rows(rank, bytes);
            std::memcpy(own.data(), rows.data(), bytes);
            rows = own;
        }
        CUDA_CHECK(
            cudaMemcpyAsync(rank.rows, rows.data(), bytes, cudaMemcpyHostToDevice, rank.stream));
        CUDA_CHECK(cudaEventRecord(rank.staged, rank.stream));
    }

    // Waits for the rows being read, unless already waited for, between a pair of stall events on
    // the reading rank's stream.
    void await_rows(RankState& reader) {
        if (rows_waited) { return; }
        StallEvents& events = stall_events[stall_next];
        stall_next          = (stall_next + 1) % stall_events.size();
        if (events.wanted == nullptr) {
            CUDA_CHECK(cudaEventCreate(&events.wanted));
            CUDA_CHECK(cudaEventCreate(&events.ready));
        }
        measure_stall(events, true);
        CUDA_CHECK(cudaEventRecord(events.wanted, reader.stream));
        table->wait();
        rows_waited = true;
        CUDA_CHECK(cudaEventRecord(events.ready, reader.stream));
        events.pending = true;
    }

    // Adds a finished wait to the stall counters; `block` waits for it to finish.
    void measure_stall(StallEvents& events, bool block) {
        if (!events.pending) { return; }
        if (block) {
            CUDA_CHECK(ninfer::wait_for_event(events.ready));
        } else {
            const cudaError_t status = cudaEventQuery(events.ready);
            if (status == cudaErrorNotReady) { return; }
            CUDA_CHECK(status);
        }
        float ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms, events.wanted, events.ready));
        events.pending = false;
        // An upload queued behind unfinished layers completes right after them; anything beyond
        // a few microseconds is time the stream waited for the read.
        if (ms > 0.002F) {
            ++stalls;
            stall_seconds += double(ms) * 1e-3;
        }
    }

    NgramTableStats ngram_stats() {
        if (!table) { return {}; }
        RankBinding bind(device, rows_rank);
        for (StallEvents& events : stall_events) { measure_stall(events, false); }
        NgramTableStats out = table->counters();
        out.stalls          = stalls;
        out.stall_seconds   = stall_seconds;
        return out;
    }

    // The part's columns of a BF16 [rows, *] plane.
    static void* columns(void* plane, std::int32_t rows, std::int32_t column, std::size_t bytes) {
        return static_cast<std::byte*>(plane) + std::size_t(rows) * std::size_t(column) * bytes;
    }

    // The columns from the parts' first to the end of their last: what a mixer's steps over every
    // column of the call take (a tensor-parallel prompt piece is a part of its own).
    [[nodiscard]] static ColumnRun covered(std::span<const Part> parts) {
        std::int32_t first = parts.front().column, end = first;
        for (const Part& part : parts) {
            first = std::min(first, part.column);
            end   = std::max(end, part.column + part.count);
        }
        return {.column = first, .count = end - first};
    }

    // A verification's record row of the part: its position among the verified sequences.
    [[nodiscard]] std::int32_t record_row(const Part& part) const {
        const auto at = std::find(verified.begin(), verified.end(), part.sequence);
        return static_cast<std::int32_t>(at - verified.begin());
    }

    // Columns that one call of a mixer projection takes: consecutive parts of up to kVectorTokens
    // tokens together, kVectorTokens columns at most and every part whole, a wider part alone. A
    // GGUF projection's vector products give a column the same bits at any width up to
    // kVectorTokens, so each part gets what a call of its own gives it, and a batch of decode steps
    // reads the weights once instead of once a sequence; `together` false (a projection of another
    // format) keeps a call per part.
    [[nodiscard]] static std::vector<ColumnRun> projection_groups(std::span<const Part> parts,
                                                                  bool together) {
        std::vector<ColumnRun> groups;
        for (const Part& part : parts) {
            if (together && !groups.empty()) {
                ColumnRun& last = groups.back();
                if (last.column + last.count == part.column &&
                    last.count + part.count <= kVectorTokens) {
                    last.count += part.count;
                    continue;
                }
            }
            groups.push_back({.column = part.column, .count = part.count});
        }
        return groups;
    }

    [[nodiscard]] static bool gguf_only(const Projection& p) { return gguf_projection(p); }

    // The parts' columns of `out` (BF16 [p.rows, *]) <- p's projection of those of `x` (BF16
    // [p.k, *]), a call per group of projection_groups.
    static void project_columns(const Projection& p, void* x, void* out,
                                std::span<const Part> parts, WorkspaceArena& ws, cudaStream_t s) {
        for (const ColumnRun& group : projection_groups(parts, gguf_only(p))) {
            const Tensor in(columns(x, p.k, group.column, 2), DType::BF16, {p.k, group.count});
            Tensor o(columns(out, p.rows, group.column, 2), DType::BF16, {p.rows, group.count});
            project(p, in, o, ws, s);
        }
    }

    // Several projections of one input x (BF16 [k, *]) into their planes, over the parts'
    // columns: the GGUF projections share each column group's q8_1 activation (project_together);
    // the one-operand projections of another format take one linear_together a part, which runs
    // the BF16 ones that a call of their own runs as a GEMV (a Gated DeltaNet layer's a and b) in
    // one launch; a projection of several operands takes project_columns of its own.
    struct ColumnsInto {
        const Projection* projection = nullptr;
        void* out                    = nullptr;
    };
    static void project_columns(std::initializer_list<ColumnsInto> list, void* x,
                                std::span<const Part> parts, WorkspaceArena& ws, cudaStream_t s) {
        std::vector<const ColumnsInto*> gguf;
        std::vector<const ColumnsInto*> single;
        gguf.reserve(list.size());
        single.reserve(list.size());
        for (const ColumnsInto& into : list) {
            if (gguf_only(*into.projection)) {
                gguf.push_back(&into);
            } else if (into.projection->pieces.size() == 1) {
                single.push_back(&into);
            } else {
                project_columns(*into.projection, x, into.out, parts, ws, s);
            }
        }
        if (!single.empty()) {
            const std::int32_t k = single.front()->projection->k;
            std::vector<Tensor> outs;
            std::vector<ops::LinearInto> into;
            outs.reserve(single.size());
            into.reserve(single.size());
            for (const ColumnRun& group : projection_groups(parts, false)) {
                const Tensor in(columns(x, k, group.column, 2), DType::BF16, {k, group.count});
                outs.clear();
                into.clear();
                for (const ColumnsInto* c : single) {
                    if (c->projection->k != k) {
                        throw std::invalid_argument("projections of one input read its width");
                    }
                    const std::int32_t rows = c->projection->rows;
                    outs.push_back(Tensor(columns(c->out, rows, group.column, 2), DType::BF16,
                                          {rows, group.count}));
                }
                for (std::size_t i = 0; i < outs.size(); ++i) {
                    const auto& operand = single[i]->projection->pieces.front().weight;
                    into.push_back({.weight = &operand.weight, .out = &outs[i],
                                    .policy = operand.policy});
                }
                ops::linear_together(in, into, ws, s);
            }
        }
        if (gguf.empty()) { return; }
        const std::int32_t k = gguf.front()->projection->k;
        std::vector<Tensor> outs;
        std::vector<ProjectionInto> into;
        outs.reserve(gguf.size());
        into.reserve(gguf.size());
        for (const ColumnRun& group : projection_groups(parts, true)) {
            const Tensor in(columns(x, k, group.column, 2), DType::BF16, {k, group.count});
            outs.clear();
            into.clear();
            for (const ColumnsInto* c : gguf) {
                const std::int32_t rows = c->projection->rows;
                outs.push_back(Tensor(columns(c->out, rows, group.column, 2), DType::BF16,
                                      {rows, group.count}));
            }
            for (std::size_t i = 0; i < outs.size(); ++i) {
                into.push_back({gguf[i]->projection, &outs[i]});
            }
            project_together(into, in, ws, s);
        }
    }

    // A mixer's output into the columns of rank.y (BF16), or under tensor parallelism of the rank's
    // FP32 `partial` [hidden, *] of the exchange.
    [[nodiscard]] Tensor mixer_output(RankState& rank, const ColumnRun& run, float* partial) const {
        const auto h = static_cast<std::int32_t>(config.hidden_size);
        if (partial != nullptr) {
            return Tensor(partial + std::size_t(h) * std::size_t(run.column), DType::FP32,
                          {h, run.count});
        }
        return Tensor(columns(rank.y, h, run.column, 2), DType::BF16, {h, run.count});
    }

    // Where a Gated DeltaNet layer keeps a pass's `t` columns in a rank's planes; a part's columns
    // are its own columns of every plane. The heads are the plan's.
    struct GdnPlanes {
        void* qkv   = nullptr;
        void* z     = nullptr;
        void* a     = nullptr;
        void* b     = nullptr;
        void* q     = nullptr;
        void* k     = nullptr;
        void* v     = nullptr;
        void* o     = nullptr;
        void* gated = nullptr;
        float* gate = nullptr;
        float* beta = nullptr;
    };

    [[nodiscard]] GdnPlanes gdn_planes(const GdnPlan& g, const RankState& rank,
                                       std::int32_t t) const {
        const auto kd = g.key_heads * static_cast<std::int32_t>(config.linear_key_head_dim);
        const std::uint64_t nv = std::uint64_t(g.value_heads);
        auto* c = static_cast<std::byte*>(rank.c);
        auto* d = static_cast<std::byte*>(rank.d);
        auto* f = static_cast<std::byte*>(rank.g);
        return {.qkv   = rank.a,
                .z     = rank.b,
                .a     = c,
                .b     = c + round_up(nv * std::uint64_t(t) * 2),
                .q     = d,
                .k     = d + round_up(std::uint64_t(kd) * std::uint64_t(t) * 2),
                .v     = rank.e,
                .o     = rank.f,
                .gated = rank.a,
                .gate  = reinterpret_cast<float*>(f),
                .beta  = reinterpret_cast<float*>(f + round_up(nv * std::uint64_t(t) * 4))};
    }

    // A Gated DeltaNet layer runs in three steps over a pass of `t` columns, `parts` holding them:
    // gdn_project, its projections and gates over every column (projection_groups); gdn_part, the
    // convolution and recurrence of one part, which read and advance its sequence's state; and
    // gdn_output, the gated norm and the output projection over every column.
    void gdn_project(const GdnPlan& g, RankState& rank, std::span<const Part> parts,
                     std::int32_t t) {
        WorkspaceArena& ws   = *rank.workspace;
        const cudaStream_t s = rank.stream;
        const GdnPlanes p    = gdn_planes(g, rank, t);
        project_columns({{&g.qkv, p.qkv}, {&g.z, p.z}, {&g.a, p.a}, {&g.b, p.b}}, rank.mixed, parts,
                        ws, s);
        const auto nv        = g.value_heads;
        const ColumnRun span = covered(parts);
        const std::size_t at = std::size_t(nv) * std::size_t(span.column);
        const Tensor ga(columns(p.a, nv, span.column, 2), DType::BF16, {nv, span.count}),
            gb(columns(p.b, nv, span.column, 2), DType::BF16, {nv, span.count});
        Tensor gate(p.gate + at, DType::FP32, {nv, span.count}),
            beta(p.beta + at, DType::FP32, {nv, span.count});
        ops::gdn_gating(ga, gb, g.a_log, g.dt_bias, gate, beta, s);
    }

    void gdn_part(const LayerPlan& plan, std::size_t index, LayerState& state, RankState& rank,
                  const Part& part, std::int32_t t) {
        const GdnPlan& g     = *plan.gdn;
        WorkspaceArena& ws   = *rank.workspace;
        const cudaStream_t s = rank.stream;
        const GdnPlanes p    = gdn_planes(g, rank, t);
        const auto n         = part.count;
        const auto column    = part.column;
        const auto nk        = g.key_heads;
        const auto nv        = g.value_heads;
        const auto kd = nk * static_cast<std::int32_t>(config.linear_key_head_dim);
        const auto vd = nv * static_cast<std::int32_t>(config.linear_value_head_dim);
        const auto dh = static_cast<std::int32_t>(config.linear_value_head_dim);
        Tensor qkv(columns(p.qkv, g.qkv.rows, column, 2), DType::BF16, {g.qkv.rows, n});
        // A verification records its raw projection, the transitions and the gates for the fold
        // that commits a prefix, and leaves the sequence's states as they were. The record rows
        // are the round's widest width apart; a narrower row fills the first columns of its own.
        std::optional<GdnReplayRecordLayer> record;
        if (verifying) {
            const auto row = record_row(part);
            const auto all = rank.records->layer(static_cast<std::int32_t>(gdn_local[index]),
                                                 static_cast<std::int32_t>(verified.size()),
                                                 static_cast<std::int32_t>(active_verify_width));
            record = GdnReplayRecordLayer{all.conv.slice(2, row, 1).slice(1, 0, n),
                                          all.key.slice(3, row, 1).slice(2, 0, n),
                                          all.value.slice(3, row, 1).slice(2, 0, n),
                                          all.gate.slice(3, row, 1).slice(2, 0, n)};
            CUDA_CHECK(cudaMemcpyAsync(record->conv.data, qkv.data,
                                       std::size_t(g.qkv.rows) * std::size_t(n) * 2,
                                       cudaMemcpyDeviceToDevice, s));
            qkv = Tensor(record->conv.data, DType::BF16, {g.qkv.rows, n});
        }
        // Convolution of the projections' q/k/v planes into the part's columns of its own planes.
        Tensor q(columns(p.q, kd, column, 2), DType::BF16, {kd, n});
        Tensor k(columns(p.k, kd, column, 2), DType::BF16, {kd, n});
        Tensor v(columns(p.v, vd, column, 2), DType::BF16, {vd, n});
        const auto window = static_cast<std::int32_t>(config.linear_conv_kernel_dim - 1);
        Tensor conv_state(state.conv.data, DType::BF16, {g.qkv.rows, window});
        Tensor conv_next =
            record ? Tensor(rank.conv_scratch.p, DType::BF16, {g.qkv.rows, window}) : conv_state;
        ops::causal_conv1d_silu_split(qkv, g.convolution, conv_state, conv_next, q, k, v, s);
        const Tensor gate(p.gate + std::size_t(nv) * std::size_t(column), DType::FP32, {nv, n});
        const Tensor beta(p.beta + std::size_t(nv) * std::size_t(column), DType::FP32, {nv, n});
        Tensor q3(q.data, DType::BF16, {dh, nk, n}), k3(k.data, DType::BF16, {dh, nk, n});
        Tensor v3(v.data, DType::BF16, {dh, nv, n});
        Tensor o(columns(p.o, vd, column, 2), DType::BF16, {dh, nv, n});
        const float scale = 1.0F / std::sqrt(float(dh));
        if (record) {
            const Tensor states = rank.gdn->layer_view(gdn_local[index]).recurrent;
            const Tensor slot(static_cast<std::int32_t*>(rank.slot_ids.p) + part.sequence->slot,
                              DType::I32, {1});
            Tensor o4 = o.view({dh, nv, n, 1});
            ops::gated_delta_net_replay_record(q3.view({dh, nk, n, 1}), k3.view({dh, nk, n, 1}),
                                               v3.view({dh, nv, n, 1}), gate.view({nv, n, 1}),
                                               beta.view({nv, n, 1}), scale, states, Tensor{}, slot,
                                               record->key, record->value, record->gate, o4, s);
        } else {
            Tensor ssm(state.ssm.data, DType::FP32, {dh, dh, nv});
            auto scope = ws.scope();
            ops::gated_delta_net(q3, k3, v3, gate, beta, scale, true, ws, ssm, o, s);
        }
    }

    void gdn_output(const GdnPlan& g, RankState& rank, std::span<const Part> parts, std::int32_t t,
                    float* partial) {
        WorkspaceArena& ws   = *rank.workspace;
        const cudaStream_t s = rank.stream;
        const GdnPlanes p    = gdn_planes(g, rank, t);
        const auto nv        = g.value_heads;
        const auto dh = static_cast<std::int32_t>(config.linear_value_head_dim);
        const auto vd = nv * dh;
        const ColumnRun span = covered(parts);
        const Tensor rows_o(columns(p.o, vd, span.column, 2), DType::BF16, {dh, nv * span.count});
        const Tensor rows_z(columns(p.z, vd, span.column, 2), DType::BF16, {dh, nv * span.count});
        Tensor gated_rows(columns(p.gated, vd, span.column, 2), DType::BF16,
                          {dh, nv * span.count});
        ops::gated_rmsnorm(rows_o, g.norm, rows_z, ops::GateActivation::Sigmoid,
                           config.rms_norm_eps, gated_rows, s);
        for (const ColumnRun& group : projection_groups(parts, gguf_only(g.output))) {
            const Tensor gated(columns(p.gated, vd, group.column, 2), DType::BF16,
                               {vd, group.count});
            Tensor y = mixer_output(rank, group, partial);
            project(g.output, gated, y, ws, s);
        }
    }

    // One part's Gated DeltaNet in a pass of its own.
    void run_gdn(const LayerPlan& plan, std::size_t index, LayerState& state, RankState& rank,
                 const Part& part) {
        const std::span<const Part> parts(&part, 1);
        const std::int32_t t = part.column + part.count;
        gdn_project(*plan.gdn, rank, parts, t);
        gdn_part(plan, index, state, rank, part, t);
        gdn_output(*plan.gdn, rank, parts, t, nullptr);
    }

    // Where a sparse-attention layer finds its positions and keeps its indexer state. By default
    // the staged planes and the sequence's live tail; a verification projects the indexer into its
    // record row and runs a scratch copy of the tail (commit() advances the live one); an MTP pass
    // reads its cells' positions and, drafting, its scratch tail.
    struct AttentionInputs {
        const std::int32_t* positions = nullptr; // sequence indices: KV slots, causal order
        const std::int32_t* rope      = nullptr; // 1-D RoPE positions
        void* index                   = nullptr; // BF16 [640, count]
        void* tail                    = nullptr; // FP32 [128, 3]
        std::uint32_t first           = 0;       // the first position, as the host knows it
    };

    // A verification's indexer record of sparse-attention layer `index` from column `column` of
    // the pass, else null (the rank's plane).
    [[nodiscard]] void* indexer_records(std::size_t index, const RankState& rank,
                                        std::int32_t column) const {
        if (!verifying) { return nullptr; }
        const std::uint64_t rows =
            std::uint64_t(config.indexer_n_heads + 1) * config.indexer_head_dim;
        const std::uint64_t columns = std::uint64_t(verify_width) * options.sequences;
        return static_cast<std::byte*>(rank.index_records.p) +
               (qsa_local[index] * columns + std::uint64_t(column)) * rows * 2;
    }

    [[nodiscard]] AttentionInputs target_inputs(std::size_t index, LayerState& state,
                                                RankState& rank, const Part& part) {
        AttentionInputs in{.positions = rank.positions + part.column,
                           .rope      = rank.rope + part.column,
                           .index     = indexer_records(index, rank, part.column),
                           .tail      = state.tail.p,
                           .first     = part.sequence->position + part.offset};
        if (verifying) {
            CUDA_CHECK(cudaMemcpyAsync(rank.tail_scratch.p, state.tail.p, state.tail.bytes,
                                       cudaMemcpyDeviceToDevice, rank.stream));
            in.tail = rank.tail_scratch.p;
        }
        return in;
    }

    // Where a sparse-attention layer keeps a pass's `t` columns in a rank's planes, its indexer
    // projection at `index` (a verification's records) or in the rank's plane; a part's columns
    // are its own columns of every plane. A tensor-parallel rank's KV heads with their query
    // heads.
    struct QsaPlanes {
        void* q         = nullptr;
        void* gate      = nullptr;
        void* k         = nullptr;
        void* v         = nullptr;
        void* index     = nullptr;
        void* qo        = nullptr;
        void* ko        = nullptr;
        void* attention = nullptr;
    };

    [[nodiscard]] QsaPlanes qsa_planes(const QsaPlan& a, const RankState& rank, std::int32_t t,
                                       void* index) const {
        const std::uint64_t kv_bytes =
            round_up(std::uint64_t(config.head_dim) * std::uint64_t(a.kv_heads) * std::uint64_t(t) * 2);
        auto* kv = static_cast<std::byte*>(rank.c);
        return {.q         = rank.a,
                .gate      = rank.b,
                .k         = kv,
                .v         = kv + kv_bytes,
                .index     = index != nullptr ? index : kv + 2 * kv_bytes,
                .qo        = rank.d,
                .ko        = rank.e,
                .attention = rank.f};
    }

    [[nodiscard]] std::int32_t query_heads(const QsaPlan& a) const {
        return a.kv_heads *
               static_cast<std::int32_t>(config.num_attention_heads / config.num_key_value_heads);
    }

    // A sparse-attention layer runs in three steps over a pass of `t` columns, `parts` holding
    // them: qsa_project, its projections and the query and key norms and rotation over every
    // column (projection_groups), from the 1-D RoPE positions at `rope` (the pass's first column)
    // or a media prompt's; qsa_part, the KV append, indexer and attention of one part, which read
    // and advance its sequence's state; and qsa_output, the output gate and projection over every
    // column.
    void qsa_project(const QsaPlan& a, RankState& rank, std::span<const Part> parts,
                     std::int32_t t, void* index, const std::int32_t* rope) {
        WorkspaceArena& ws   = *rank.workspace;
        const cudaStream_t s = rank.stream;
        const QsaPlanes p    = qsa_planes(a, rank, t, index);
        const auto d         = static_cast<std::int32_t>(config.head_dim);
        const auto nk        = a.kv_heads;
        const auto nq        = query_heads(a);
        project_columns({{&a.query, p.q},
                         {&a.gate, p.gate},
                         {&a.key, p.k},
                         {&a.value, p.v},
                         {&a.index, p.index}},
                        rank.mixed, parts, ws, s);
        // The parts' columns; a media prompt's three-axis positions hold every column of the pass.
        const ColumnRun span = mrope ? ColumnRun{.column = 0, .count = t} : covered(parts);
        Tensor q3(columns(p.q, d * nq, span.column, 2), DType::BF16, {d, nq, span.count});
        Tensor k3(columns(p.k, d * nk, span.column, 2), DType::BF16, {d, nk, span.count});
        Tensor qo(columns(p.qo, d * nq, span.column, 2), DType::BF16, {d, nq, span.count});
        Tensor ko(columns(p.ko, d * nk, span.column, 2), DType::BF16, {d, nk, span.count});
        const ops::RopeYarn& yarn = options.rope_yarn;
        // A tensor-parallel share of the heads leaves the fused norm and rotation's head counts.
        const bool share = nk != static_cast<std::int32_t>(config.num_key_value_heads);
        if (mrope || yarn.active() || share) {
            // A media prompt: the three-axis positions of its images and video, each pair of
            // rotated dimensions on axis i % 3 (the interleaved MRoPE sections). Under YaRN, text
            // too: the fused norm and rotation fixes the unscaled one, so the rotation is its own
            // Op, one tensor at a time (its YaRN forms cover 24 query and 2 key heads that way).
            const Tensor axes(rank.mrope, DType::I32, {t, 3});
            const Tensor line(const_cast<std::int32_t*>(rope) + span.column, DType::I32,
                              {span.count});
            const int rotary = static_cast<int>(float(d) * config.partial_rotary_factor);
            ops::rmsnorm(q3, a.query_norm, config.rms_norm_eps, true, qo, s);
            ops::rmsnorm(k3, a.key_norm, config.rms_norm_eps, true, ko, s);
            if (yarn.active()) {
                ops::rope(mrope ? axes : line, rotary, config.rope_theta, yarn, qo, s);
                ops::rope(mrope ? axes : line, rotary, config.rope_theta, yarn, ko, s);
            } else {
                ops::rope(mrope ? axes : line, rotary, config.rope_theta, qo, ko, s);
            }
        } else {
            const Tensor line(const_cast<std::int32_t*>(rope) + span.column, DType::I32,
                              {span.count});
            ops::rmsnorm_rope(line, a.query_norm, a.key_norm, q3, k3, qo, ko, s);
        }
    }

    // With `kv_only` (a K/V-only last layer, kv_only_pass), only the part's KV and indexer appends
    // run, which leave the sequence's state as the whole step leaves it.
    void qsa_part(const QsaPlan& a, LayerState& state, RankState& rank, const Part& part,
                  std::int32_t t, const AttentionInputs& in, bool kv_only = false) {
        WorkspaceArena& ws   = *rank.workspace;
        const cudaStream_t s = rank.stream;
        const QsaPlanes p    = qsa_planes(a, rank, t, nullptr);
        const auto n         = part.count;
        const auto column    = part.column;
        const auto d         = static_cast<std::int32_t>(config.head_dim);
        const auto nk        = a.kv_heads;
        const auto nq        = query_heads(a);
        const Tensor qo(columns(p.qo, d * nq, column, 2), DType::BF16, {d, nq, n});
        const Tensor ko(columns(p.ko, d * nk, column, 2), DType::BF16, {d, nk, n});
        const Tensor v3(columns(p.v, d * nk, column, 2), DType::BF16, {d, nk, n});
        Tensor index(in.index != nullptr ? in.index : columns(p.index, a.index.rows, column, 2),
                     DType::BF16, {a.index.rows, n});
        auto* const part_positions = const_cast<std::int32_t*>(in.positions);
        const Tensor positions(part_positions, DType::I32, {n});
        const ops::RopeYarn& yarn = options.rope_yarn;
        // The paged KV: device pages, or a streamed layer's host copy. A streamed layer's call of
        // at most stream_width tokens reads its device pool once kv_stream_resolve has made the
        // call's blocks resident there; a wider call (a prompt chunk) reads the rank's staging
        // copy, into which it appends as well, and then drops the blocks it wrote from the pool.
        const auto paged_count = static_cast<std::int32_t>(pages);
        const PagedKVLayerView cache =
            kv_view([&](std::size_t i) { return state.kv[i].p; }, kPagedKVPageSize, paged_count,
                    rank.block_table.p, paged_count, nk);
        const bool streamed = state.map.table != nullptr;
        const bool pooled_read = streamed && n <= static_cast<std::int32_t>(stream_width);
        const auto block       = [](std::uint32_t position) {
            return static_cast<std::int32_t>(position / ops::kKvStreamPageTokens);
        };
        ops::kv_cache_append(ko, v3, positions, cache, s);
        const auto di = static_cast<std::int32_t>(config.indexer_head_dim);
        Tensor pooled(state.pooled.p, DType::FP32, {di, static_cast<std::int32_t>(pooled_slots)});
        Tensor tail(in.tail, DType::FP32,
                    {di, static_cast<std::int32_t>(config.indexer_compress_ratio - 1)});
        const ops::QsaIndexerWeights iw{&a.index_query_norm, &a.index_key_norm};
        // The call's first position is the part's first staged one, read on the device.
        const Tensor first(part_positions, DType::I32, {1});
        if (kv_only) {
            if (streamed) {
                // As the attention's appends leave them: a staging copy that holds the layer's
                // positions before the part takes the part's too, and no pool slot keeps a block
                // older than the host copy.
                if (pooled_read) {
                    if (rank.kv_staged == &state) { rank.kv_staged = nullptr; }
                } else if (rank.kv_staged == &state && rank.kv_staged_end >= in.first) {
                    const PagedKVLayerView staged =
                        kv_view([&](std::size_t i) { return rank.kv_staging[i].p; },
                                kPagedKVPageSize, paged_count, rank.block_table.p, paged_count);
                    ops::kv_cache_append(ko, v3, positions, staged, s);
                    rank.kv_staged_end = in.first + std::uint32_t(n);
                }
                ops::kv_stream_evict(state.map, block(in.first),
                                     block(in.first + std::uint32_t(n) - 1) + 1, s);
            }
            ops::qsa_indexer_append(index, first, iw, config.rms_norm_eps, yarn, pooled, tail, s);
            return;
        }
        PagedKVLayerView read = cache;
        if (pooled_read) {
            if (rank.kv_staged == &state) { rank.kv_staged = nullptr; }
            read = kv_view([&](std::size_t i) { return state.pool[i].p; }, ops::kKvStreamPageTokens,
                           static_cast<std::int32_t>(stream_slots), state.map.table,
                           state.map.blocks);
        } else if (streamed) {
            stage(state, rank, in.first);
            read = kv_view([&](std::size_t i) { return rank.kv_staging[i].p; }, kPagedKVPageSize,
                           paged_count, rank.block_table.p, paged_count);
            ops::kv_cache_append(ko, v3, positions, read, s);
            rank.kv_staged_end = in.first + std::uint32_t(n);
        }
        ops::qsa_indexer_append(index, first, iw, config.rms_norm_eps, yarn, pooled, tail, s);
        Tensor selected(rank.selected, DType::I32,
                        {static_cast<std::int32_t>(config.indexer_block_budget()), n});
        Tensor counts(rank.counts, DType::I32, {n});
        {
            auto scope = ws.scope();
            ops::qsa_indexer_select(index, first, iw, config.rms_norm_eps, yarn, pooled, ws,
                                    selected, counts, s);
        }
        if (pooled_read) {
            ops::kv_stream_resolve(state.map, first, selected, counts, cache, read, s);
        }
        Tensor attention(columns(p.attention, d * nq, column, 2), DType::BF16, {d, nq, n});
        {
            auto scope = ws.scope();
            ops::sparse_softmax_attention(qo, first, selected, counts, read,
                                          1.0F / std::sqrt(float(d)), ws, attention, s);
        }
        if (streamed && !pooled_read) {
            ops::kv_stream_evict(state.map, block(in.first),
                                 block(in.first + std::uint32_t(n) - 1) + 1, s);
        }
    }

    void qsa_output(const QsaPlan& a, RankState& rank, std::span<const Part> parts,
                    std::int32_t t, float* partial) {
        WorkspaceArena& ws   = *rank.workspace;
        const cudaStream_t s = rank.stream;
        const QsaPlanes p    = qsa_planes(a, rank, t, nullptr);
        const auto width     = static_cast<std::int32_t>(config.head_dim) * query_heads(a);
        const ColumnRun span = covered(parts);
        const Tensor gate(columns(p.gate, width, span.column, 2), DType::BF16, {width, span.count});
        Tensor flat(columns(p.attention, width, span.column, 2), DType::BF16, {width, span.count});
        ops::sigmoid_mul(gate, flat, s);
        for (const ColumnRun& group : projection_groups(parts, gguf_only(a.output))) {
            const Tensor in(columns(p.attention, width, group.column, 2), DType::BF16,
                            {width, group.count});
            Tensor y = mixer_output(rank, group, partial);
            project(a.output, in, y, ws, s);
        }
    }

    // One part's sparse attention in a pass of its own; with `kv_only`, its appends alone.
    void run_qsa(const LayerPlan& plan, LayerState& state, RankState& rank, const Part& part,
                 const AttentionInputs& in, bool kv_only = false) {
        const std::span<const Part> parts(&part, 1);
        const std::int32_t t = part.column + part.count;
        // The pass's indexer plane, which the part's columns continue.
        void* index = in.index == nullptr
                          ? nullptr
                          : static_cast<std::byte*>(in.index) -
                                std::size_t(plan.qsa->index.rows) * std::size_t(part.column) * 2;
        qsa_project(*plan.qsa, rank, parts, t, index, in.rope - part.column);
        qsa_part(*plan.qsa, state, rank, part, t, in, kv_only);
        if (!kv_only) { qsa_output(*plan.qsa, rank, parts, t, nullptr); }
    }

    // Layer `i`'s mixer over the pass's `parts` (its `t` columns), part k reading and advancing
    // the state `state_of(part)`: the projections and the steps without state over every column at
    // once, the others part by part. In a branch pass the first branch continues the state its
    // sequence holds and leaves a copy in `slot`, from which each later branch starts the layer.
    // Under tensor parallelism into the rank's FP32 `partial`. With `kv_only`, a sparse-attention
    // layer's projections and appends alone (qsa_part), without an output.
    template <typename StateOf>
    void run_mixer(const LayerPlan& plan, std::size_t i, RankState& rank,
                   std::span<const Part> parts, std::int32_t t, std::size_t slot,
                   const StateOf& state_of, float* partial, bool kv_only = false) {
        const std::size_t first_branch = branch_capture != nullptr ? 1 : 0;
        if (kv_only && plan.gdn) {
            throw std::logic_error("qwen4_exp: a K/V-only layer is a sparse-attention one");
        }
        if (plan.gdn) {
            gdn_project(*plan.gdn, rank, parts, t);
        } else {
            qsa_project(*plan.qsa, rank, parts, t, indexer_records(i, rank, 0), rank.rope);
        }
        for (std::size_t k = 0; k < parts.size(); ++k) {
            const Part& part  = parts[k];
            LayerState& state = state_of(part);
            if (k > first_branch) { restore_branch_layer(state, slot, rank, kMixerRegions); }
            if (plan.gdn) {
                gdn_part(plan, i, state, rank, part, t);
            } else {
                qsa_part(*plan.qsa, state, rank, part, t, target_inputs(i, state, rank, part),
                         kv_only);
            }
            if (k == 0) { capture_branch_layer(state, slot, rank, kMixerRegions); }
        }
        if (kv_only) { return; }
        if (plan.gdn) {
            gdn_output(*plan.gdn, rank, parts, t, partial);
        } else {
            qsa_output(*plan.qsa, rank, parts, t, partial);
        }
    }

    void run_ple(const LayerPlan& plan, LayerState& state, RankState& rank, const Part& part) {
        const PlePlan& p     = *plan.ple;
        WorkspaceArena& ws   = *rank.workspace;
        const cudaStream_t s = rank.stream;
        const auto t         = part.count;
        const auto heads     = static_cast<std::int32_t>(config.ngram_heads());
        const auto h         = static_cast<std::int32_t>(config.hidden_size);
        const auto width     = static_cast<std::int32_t>(config.hc_count * config.hidden_size);
        const auto row_b = static_cast<std::int32_t>(ops::ngram_row_bytes(options.ngram->format));
        const Tensor rows(columns(rank.rows, row_b * heads, part.column, 1), DType::U8,
                          {row_b, heads * t});
        Tensor stack(rank.stack + std::size_t(width) * std::size_t(part.column), DType::FP32,
                     {h, static_cast<std::int32_t>(config.hc_count), t});
        Tensor embedding(rank.a, DType::BF16, {static_cast<std::int32_t>(config.ple_embed_dim), t});
        ops::ngram_embed_rows(rows, options.ngram->format, heads, embedding, s);
        Tensor key(rank.g, DType::BF16, {width, t});
        Tensor value(rank.b, DType::BF16, {h, t});
        const ProjectionInto both[] = {{&p.key, &key}, {&p.value, &value}};
        project_together(both, embedding, ws, s);
        Tensor history(state.history.p, DType::FP32,
                       {width, static_cast<std::int32_t>((config.ple_conv_kernel_size - 1) *
                                                         config.ngram.ngram_size)});
        const ops::PleInjectWeights weights{&p.norm_key, &p.norm_query, &p.norm_conv,
                                            &p.convolution};
        auto scope = ws.scope();
        if (verifying) {
            // The history stays; commit() advances it over the kept positions' recorded input.
            Tensor normalized(static_cast<float*>(rank.ple_records.p) +
                                  std::size_t(width) * std::size_t(part.column),
                              DType::FP32, {width, t});
            ops::ple_inject_record(stack, key, value, weights, config.rms_norm_eps, history,
                                   normalized, ws, s);
            return;
        }
        ops::ple_inject(stack, key, value, weights, config.rms_norm_eps, history, ws, s);
    }

    // A wide call over host-resident experts: once its routes reach the host, the routed experts
    // the cache does not hold are copied into the rank's slots, and the experts run from device
    // memory through tables that point at cache slots and pool slots. False, leaving the call to
    // the vector products over the cache's tables, when the pool cannot hold them.
    bool run_through_slots(const LayerPlan& plan, RankState& rank, std::int32_t t,
                           std::size_t index, Tensor& ids, Tensor& weights, Tensor& shared) {
        SlotPool& pool = slot_pools[rank.rank];
        if (pool.slots == 0) { return false; }
        const cudaStream_t s = rank.stream;
        const std::size_t b        = pool_buffer(index);
        const bool double_buffered = pool.buffers.size() == 2;
        // A span's last chunk may be shorter than kPrefetchTokens, but the pool still holds the
        // layer's experts that its earlier chunks prefetched: it runs from the pool too, rather
        // than copying its routed experts over the pool again, and the next layer's copies start
        // behind it as behind any prefetched call.
        const bool prefetched =
            !prefetches.empty() && prefetches[rank.rank].buffers[b].layer == index;
        if ((t >= kPrefetchTokens || prefetched) && pool.slots >= plan.moe.gate.pointers.size()) {
            PoolPrefetch& state = prefetches[rank.rank];
            auto& buffer        = state.buffers[b];
            if (buffer.layer != index) { prefetch_layer(index); }
            // With two buffers the next layer's copies need only the previous layer's calls done:
            // they run while this layer's experts do.
            if (double_buffered && pooled_layer(index + 1, rank.rank) &&
                state.buffers[1 - b].layer != index + 1) {
                prefetch_layer(index + 1);
            }
            await_prefetch(buffer);
            CUDA_CHECK(cudaStreamWaitEvent(s, buffer.ready, 0));
            const ops::GgufMoeWeights banks =
                plan.moe.pooled(static_cast<const void* const*>(pool.buffers[b].tables.p));
            const auto h = static_cast<std::int32_t>(config.hidden_size);
            const Tensor mixed(rank.mixed, DType::BF16, {h, t});
            Tensor y(rank.y, DType::FP32, {h, t});
            {
                auto scope = rank.workspace->scope();
                run_gguf(plan.moe, mixed, ids, weights, shared, banks, *rank.workspace, y, s);
            }
            CUDA_CHECK(cudaEventRecord(buffer.free, s));
            // A span's later chunks run this layer's experts again from the pool; otherwise the
            // buffer takes the copies of the next text layer on this device that reads it.
            if (span_hold) { return true; }
            const std::size_t next = index + (double_buffered ? 2 : 1);
            if (pooled_layer(next, rank.rank)) {
                prefetch_layer(next);
            } else {
                buffer.layer = ~std::size_t{0};
            }
            return true;
        }
        const auto top       = static_cast<std::int32_t>(config.num_experts_per_tok);
        auto* routes         = static_cast<std::int32_t*>(route_host->data()) +
                               index * std::size_t(top) * options.prefill_chunk;
        CUDA_CHECK(
            cudaMemcpyAsync(routes, ids.data, std::size_t(top) * t * 4, cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(ninfer::wait_for_stream(s));
        const MoePlan& m          = plan.moe;
        const std::size_t experts = m.gate.pointers.size();
        // The copies below overwrite the first buffer, so no buffer holds a prefetched layer. They
        // and the call wait for the prefetches' copies, which may have put experts into cache
        // slots that this call reads.
        if (!prefetches.empty()) {
            drain_prefetches();
            // A pool of one buffer (an explicit expert cache, --expert-prefill-buffers 1, or too
            // little free memory for two) leaves the second buffer without events.
            for (auto& buffer : prefetches[rank.rank].buffers) {
                if (buffer.ready != nullptr) {
                    CUDA_CHECK(cudaStreamWaitEvent(s, buffer.ready, 0));
                }
                buffer.layer = ~std::size_t{0};
            }
        }
        std::vector<char> routed(experts, 0);
        for (std::int32_t i = 0; i < top * t; ++i) {
            const std::int32_t e = routes[i];
            if (e >= 0 && std::size_t(e) < experts) { routed[e] = 1; }
        }
        if (pool.slots < experts) { return false; }
        auto* entries     = static_cast<const void**>(slot_entries->data());
        const auto wanted = [&](std::size_t e) { return routed[e] != 0; };
        stage_pool(pool, pool.buffers[0], index, wanted, entries, s);
        CUDA_CHECK(cudaMemcpyAsync(pool.buffers[0].tables.p, entries, 3 * experts * sizeof(void*),
                                   cudaMemcpyHostToDevice, s));
        const ops::GgufMoeWeights banks =
            m.pooled(static_cast<const void* const*>(pool.buffers[0].tables.p));
        const auto h = static_cast<std::int32_t>(config.hidden_size);
        const Tensor mixed(rank.mixed, DType::BF16, {h, t});
        Tensor y(rank.y, DType::FP32, {h, t});
        auto scope = rank.workspace->scope();
        run_gguf(m, mixed, ids, weights, shared, banks, *rank.workspace, y, s);
        return true;
    }

    // One call over `banks` (device banks, cache slots or a slot pool): Flash-Next's experts, or a
    // latent MoE's through the same routed tables.
    static void run_gguf(const MoePlan& m, const Tensor& mixed, const Tensor& ids,
                         const Tensor& weights, const Tensor& shared,
                         const ops::GgufMoeWeights& banks, WorkspaceArena& ws, Tensor& y,
                         cudaStream_t s) {
        if (m.latent) {
            ops::moe_latent_gguf(mixed, ids, weights, shared, m.latent_banks(banks), ws, y, s);
            return;
        }
        ops::moe_experts_gguf(mixed, ids, weights, shared, banks, ws, y, s);
    }

    // A latent layer's call over host or disk experts through the misses (served_by_misses): D and
    // the shared expert, then the cached pairs while the missing experts arrive or CPU threads
    // compute them from the pairs' latent inputs, then U.
    void run_latent_misses(const MoePlan& m, RankState& rank, std::int32_t t, std::size_t index,
                           const Tensor& mixed, const Tensor& ids, const Tensor& weights,
                           const Tensor& shared, Tensor& y) {
        WorkspaceArena& ws       = *rank.workspace;
        const cudaStream_t s     = rank.stream;
        const bool wide          = t > kVectorTokens;
        // A wide call's stages read device memory only: cache slots, staging and the dense banks.
        const auto all           = m.latent_banks(wide ? m.cached() : m.banks());
        const auto stage         = ops::moe_latent_gguf_begin(mixed, ids, shared, all, ws, s,
                                                              wide ? nullptr : &rank.side);
        const Tensor inputs =
            stage.inputs != nullptr
                ? Tensor(stage.inputs, DType::U8,
                         {static_cast<std::int32_t>(
                             ops::moe_latent_gguf_input_bytes(stage.pairs, m.latent->width))})
                : mixed;
        const auto call = misses->begin(index, ids, inputs, t, m.banks(), s);
        ops::moe_latent_gguf_add(stage, call.cached_ids, weights, all, ws, s);
        const auto missing = m.latent_banks(call.missing_banks);
        std::int32_t products = 0;
        if (wide) {
            products = misses->stage_wide(index, s, [&](const Tensor& missing_ids) {
                ops::moe_latent_gguf_add(stage, missing_ids, weights, missing, ws, s);
            });
        } else {
            misses->await(index, s);
            ops::moe_latent_gguf_add(stage, call.missing_ids, weights, missing, ws, s);
            products = call.max_products;
        }
        if (products > 0) {
            ops::moe_latent_gguf_add_products(stage, call.products, call.pairs, call.count,
                                              products, ids, weights, all, s);
        }
        ops::moe_latent_gguf_finish(stage, all, ws, y, s);
    }

    // The rank's router counter: every router call of a rank runs on its stream, one at a time.
    [[nodiscard]] static std::uint32_t* route_done(RankState& rank) {
        return static_cast<std::uint32_t*>(rank.route_done.p);
    }

    // Routes the `t` tokens in the rank's mixed plane: their experts into the layer's route record,
    // their weights and the shared expert's gate into the rank's route planes.
    void route(const MoePlan& m, RankState& rank, std::int32_t t, std::size_t index) {
        const auto h   = static_cast<std::int32_t>(config.hidden_size);
        const auto top = static_cast<std::int32_t>(config.num_experts_per_tok);
        Tensor ids(route_records[index], DType::I32, {top, t});
        Tensor weights(rank.route_weights, DType::FP32, {top, t});
        Tensor shared(rank.route_shared, DType::FP32, {t});
        auto scope = rank.workspace->scope();
        ops::moe_route(Tensor(rank.mixed, DType::BF16, {h, t}), m.router, m.shared_gate,
                       *rank.workspace, ids, weights, shared, rank.stream, route_done(rank));
    }

    void run_moe(const LayerPlan& plan, RankState& rank, std::int32_t t, std::size_t index) {
        const MoePlan& m     = plan.moe;
        WorkspaceArena& ws   = *rank.workspace;
        const cudaStream_t s = rank.stream;
        const auto h         = static_cast<std::int32_t>(config.hidden_size);
        const auto top       = static_cast<std::int32_t>(config.num_experts_per_tok);
        const Tensor mixed(rank.mixed, DType::BF16, {h, t});
        Tensor ids(route_records[index], DType::I32, {top, t});
        Tensor weights(rank.route_weights, DType::FP32, {top, t});
        Tensor shared(rank.route_shared, DType::FP32, {t});
        route(m, rank, t, index);
        if (m.native) {
            Tensor y(rank.y, DType::FP32, {h, t});
            if (hybrid) {
                hybrid->run(hybrid_layers[index], mixed, ids, weights, shared, m.native->banks(), t == 1 || verifying,
                            ws, y);
                return;
            }
            auto scope = ws.scope();
            ops::moe_experts_native(mixed, ids, weights, shared, m.native->banks(), nullptr, ws, y, s);
            return;
        }
        if (served_by_misses(t)) {
            // The cached pairs run at once; the missing experts arrive (or the CPU computes them)
            // meanwhile, and run in a second stage.
            Tensor y(rank.y, DType::FP32, {h, t});
            auto scope      = ws.scope();
            if (m.latent) {
                run_latent_misses(m, rank, t, index, mixed, ids, weights, shared, y);
                return;
            }
            const auto call = misses->begin(index, ids, mixed, t, m.banks(), s);
            if (t > kVectorTokens) {
                // As over device banks: the cached experts' slots while the missing experts
                // cross the bus, then each round of them, and the CPU's products of the few-pair
                // experts it took (stage_wide).
                const auto stage = ops::moe_experts_gguf_begin(mixed, ws, s);
                ops::moe_experts_gguf_add(stage, mixed, call.cached_ids, weights, &shared,
                                          m.cached(), ws, s);
                const std::int32_t computed =
                    misses->stage_wide(index, s, [&](const Tensor& missing_ids) {
                        ops::moe_experts_gguf_add(stage, mixed, missing_ids, weights, nullptr,
                                                  call.missing_banks, ws, s);
                    });
                if (computed > 0) {
                    ops::moe_experts_gguf_add_products(stage, call.products, call.pairs, call.count,
                                                       computed, weights, s);
                }
                ops::moe_experts_gguf_finish(stage, y, s);
                return;
            }
            if (ops::moe_experts_gguf_decode_supported(m.banks(), t)) {
                const auto state = ops::moe_experts_gguf_decode_begin(
                    mixed, call.cached_ids, shared, m.banks(), ws, s, &rank.side);
                misses->await(index, s);
                ops::moe_experts_gguf_decode_missing(state, mixed, call.missing_ids,
                                                     call.missing_banks, s);
                ops::moe_experts_gguf_decode_finish(state, call.cached_ids, &call.missing_ids,
                                                    weights, m.banks(), &call.missing_banks,
                                                    call.products, call.pairs, call.count, y, s);
                return;
            }
            const auto stage = ops::moe_experts_gguf_begin(mixed, ws, s);
            ops::moe_experts_gguf_add(stage, mixed, call.cached_ids, weights, &shared, m.banks(),
                                      ws, s);
            misses->await(index, s);
            ops::moe_experts_gguf_add(stage, mixed, call.missing_ids, weights, nullptr,
                                      call.missing_banks, ws, s);
            if (call.max_products > 0) {
                ops::moe_experts_gguf_add_products(stage, call.products, call.pairs, call.count,
                                                   call.max_products, weights, s);
            }
            ops::moe_experts_gguf_finish(stage, y, s);
            return;
        }
        // The slot pools are sized for the text layers' experts (the MTP block's calls, a draft
        // step's, take the vector products).
        if (t > kVectorTokens && !slot_pools.empty() && index < layers.size() &&
            run_through_slots(plan, rank, t, index, ids, weights, shared)) {
            return;
        }
        if (stream) {
            // The routes reach the host before the layer's experts can be made resident.
            auto* host = static_cast<std::int32_t*>(route_host->data()) +
                         index * std::size_t(top) * options.prefill_chunk;
            CUDA_CHECK(cudaMemcpyAsync(host, ids.data, std::size_t(top) * t * 4,
                                       cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(ninfer::wait_for_stream(s));
            stream->prepare(index, std::span<const std::int32_t>(host, std::size_t(top) * t),
                            span_last);
        }
        Tensor y(rank.y, DType::FP32, {h, t});
        auto scope = ws.scope();
        if (m.latent) {
            // The shared expert on the side stream beside the latent work at decode widths.
            const ops::GgufLatentWeights banks = m.latent_banks();
            const auto stage = ops::moe_latent_gguf_begin(
                mixed, ids, shared, banks, ws, s, t <= kVectorTokens ? &rank.side : nullptr);
            ops::moe_latent_gguf_add(stage, ids, weights, banks, ws, s);
            ops::moe_latent_gguf_finish(stage, banks, ws, y, s);
            return;
        }
        if (ops::moe_experts_gguf_decode_supported(m.banks(), t)) {
            // The shared expert on the side stream beside the routed pairs, as with staged misses.
            const auto state = ops::moe_experts_gguf_decode_begin(mixed, ids, shared, m.banks(), ws,
                                                                  s, &rank.side);
            ops::moe_experts_gguf_decode_finish(state, ids, nullptr, weights, m.banks(), nullptr,
                                                nullptr, nullptr, nullptr, y, s);
            return;
        }
        ops::moe_experts_gguf(mixed, ids, weights, shared, m.banks(), ws, y, s);
    }

    // The staged tokens' embeddings into `mixed` [hidden, t]: a media prompt's image and video
    // tokens take the tower's merged embeddings (media_runs, as stage_inputs() left them). Under
    // tensor parallelism each rank embeds from its own table and copy of the media's.
    void embed(RankState& rank, std::int32_t t, Tensor& mixed) {
        const auto h = static_cast<std::int32_t>(config.hidden_size);
        const bool own = tensor() && rank.rank > 0;
        ops::embedding(Tensor(rank.ids, DType::I32, {t}),
                       own ? rank_embedding_tables[rank.rank - 1] : embedding_table, mixed,
                       rank.stream);
        // A model without a Vision tower has no media copies on the other ranks.
        if (media_runs.empty()) { return; }
        void* media = own ? rank_vision_embeddings[rank.rank - 1].p : vision_embeddings.p;
        for (const MediaRun& run : media_runs) {
            const Tensor source(static_cast<std::byte*>(media) +
                                    std::size_t(run.source) * std::size_t(h) * 2,
                                DType::BF16, {h, run.count});
            const Tensor columns(rank.scatter + run.offset, DType::I32, {run.count});
            ops::scatter(source, columns, mixed, rank.stream);
        }
    }

    // Queues one segment of the pass for the parts' `t` tokens on its rank's stream; with `head`,
    // the logits of the last `logit_rows` tokens (none for 0). In a K/V-only pass the last layer
    // ends after its appends (kv_only_pass).
    void run_segment(std::span<const Part> parts, const Segment& segment, std::int32_t t,
                     std::uint32_t logit_rows) {
        if (tensor()) {
            run_tensor_segment(parts, segment, t, logit_rows);
            return;
        }
        RankState& rank = ranks[segment.rank];
        const auto h    = static_cast<std::int32_t>(config.hidden_size);
        const auto hc   = static_cast<std::int32_t>(config.hc_count);
        Tensor stack(rank.stack, DType::FP32, {h, hc, t});
        Tensor mixed(rank.mixed, DType::BF16, {h, t});
        if (segment.embed) {
            embed(rank, t, mixed);
            ops::hyper_connection_expand(mixed, stack, rank.stream);
        }
        // A layer's MoE output is written into the stack by the next layer's read, which takes
        // the same rounding, unless the PLE comes between.
        Tensor inject(rank.inject, DType::FP32, {hc, t});
        // In a branch pass the first branch continues the state its sequence holds and each later
        // one starts a layer from the origin's; a prompt part ahead of the branches leaves its
        // state there as it goes.
        const std::size_t first_branch = branch_capture != nullptr ? 1 : 0;
        const Tensor moe_out(rank.y, DType::FP32, {h, t});
        bool pending = false;
        for (std::size_t i = segment.begin; i < segment.end; ++i) {
            const LayerPlan& plan = layers[i];
            if (plan.ple) {
                if (pending) { ops::hyper_connection_write(stack, moe_out, inject, rank.stream); }
                pending = false;
                for (std::size_t k = 0; k < parts.size(); ++k) {
                    LayerState& state = parts[k].sequence->layers[i];
                    if (k > first_branch) {
                        restore_branch_layer(state, state_slot(i, rank.rank), rank, kPleRegions);
                    }
                    run_ple(plan, state, rank, parts[k]);
                    if (k == 0) {
                        capture_branch_layer(state, state_slot(i, rank.rank), rank, kPleRegions);
                    }
                }
            }
            {
                auto scope = rank.workspace->scope();
                if (pending) {
                    ops::hyper_connection_write_read(stack, moe_out, inject, plan.attn_hc.weights(),
                                                     config.rms_norm_eps, *rank.workspace, mixed,
                                                     &inject, rank.stream);
                } else {
                    ops::hyper_connection_read(stack, plan.attn_hc.weights(), config.rms_norm_eps,
                                               *rank.workspace, mixed, &inject, rank.stream);
                }
                pending = false;
            }
            const bool kv_only = kv_only_pass && i + 1 == layers.size();
            run_mixer(
                plan, i, rank, parts, t, state_slot(i, rank.rank),
                [&](const Part& part) -> LayerState& { return part.sequence->layers[i]; }, nullptr,
                kv_only);
            if (kv_only) { break; }
            {
                auto scope = rank.workspace->scope();
                ops::hyper_connection_write_read(
                    stack, Tensor(rank.y, DType::BF16, {h, t}), inject, plan.mlp_hc.weights(),
                    config.rms_norm_eps, *rank.workspace, mixed, &inject, rank.stream);
            }
            run_moe(plan, rank, t, i);
            pending = true;
            if (taps(i)) {
                // The DFlash2 tap reads the completed stack, which takes the deferred write's
                // rounding now.
                ops::hyper_connection_write(stack, moe_out, inject, rank.stream);
                pending = false;
                capture_tap(rank, rank.stack, i);
            }
        }
        if (pending) { ops::hyper_connection_write(stack, moe_out, inject, rank.stream); }
        if (segment.head && logit_rows > 0) { run_head(parts, rank, t, logit_rows); }
    }

    // Whether the pass captures layer i's DFlash2 features.
    [[nodiscard]] bool taps(std::size_t i) const {
        return drafter && capture.count > 0 && tap_of[i] >= 0;
    }

    // Layer i's DFlash2 features of the pass columns `capture` names, from the completed stack at
    // `stack` (FP32 [hidden, hc_count, columns]): the mean of each column's four streams, into the
    // tap's rows of the feature columns from capture.column.
    void capture_tap(RankState& rank, const float* stack, std::size_t i) {
        const auto h           = static_cast<std::int32_t>(config.hidden_size);
        const auto hc          = static_cast<std::int32_t>(config.hc_count);
        const std::size_t rows = std::size_t(drafter->taps()) * config.hidden_size;
        const Tensor columns(const_cast<float*>(stack) + std::size_t(capture.first) * hc * h,
                             DType::FP32, {h, hc, capture.count});
        Tensor out(static_cast<std::uint16_t*>(features.p) + std::size_t(capture.column) * rows +
                       std::size_t(tap_of[i]) * config.hidden_size,
                   DType::BF16, {h, capture.count});
        out.nb[1] = std::int64_t(rows) * 2;
        out.nb[2] = out.nb[1] * capture.count;
        out.nb[3] = out.nb[2];
        ops::hyper_connection_stream_mean(columns, out, rank.stream);
    }

    // The logits of the last `logit_rows` columns of the head rank's stack, or of each branch's
    // last column in a branch pass.
    void run_head(std::span<const Part> parts, RankState& rank, std::int32_t t,
                  std::uint32_t logit_rows) {
        const Tensor head_mixed = head_input(parts, rank, final_mixer, t, logit_rows);
        Tensor logits(rank.logits, DType::BF16,
                      {static_cast<std::int32_t>(config.vocab_size), head_mixed.ne[1]});
        project(head, head_mixed, logits, *rank.workspace, rank.stream);
    }

    // Tensor parallelism: a prompt pass's logits of more than one gather's columns, which rank 0
    // projects with its whole head, whole_columns columns at a time: gathering the other ranks'
    // rows of so many columns through host memory costs more than their share of the projection.
    void run_whole_head(std::span<const Part> parts, RankState& rank, std::int32_t t,
                        std::uint32_t logit_rows) {
        const auto h            = static_cast<std::int32_t>(config.hidden_size);
        const auto vocab        = static_cast<std::int32_t>(config.vocab_size);
        const Tensor head_mixed = head_input(parts, rank, final_mixer, t, logit_rows);
        const auto columns      = static_cast<std::int32_t>(logit_rows);
        for (std::int32_t c = 0; c < columns; c += whole_columns) {
            const std::int32_t count = std::min(whole_columns, columns - c);
            const Tensor x(static_cast<std::byte*>(head_mixed.data) + std::size_t(c) * h * 2,
                           DType::BF16, {h, count});
            Tensor logits(static_cast<std::byte*>(rank.logits) + std::size_t(c) * vocab * 2,
                          DType::BF16, {vocab, count});
            project(whole_head, x, logits, *rank.workspace, rank.stream);
        }
    }

    // Tensor parallelism: the head divided over the ranks, in every pass but a prompt's of more
    // than head_columns logits (run_whole_head). Each rank reads its own stack (the replicas are
    // equal) through its final mixer and projects its rows of the head; rank 0 places
    // its rows and every other rank's, which come over that rank's link, into the logits,
    // head_columns columns at a time.
    void run_divided_head(std::span<const Part> parts, std::int32_t t, std::uint32_t logit_rows) {
        const std::size_t n = device.size();
        const auto h        = static_cast<std::int32_t>(config.hidden_size);
        const auto vocab    = static_cast<std::size_t>(config.vocab_size);
        const auto columns  = static_cast<std::int32_t>(logit_rows);
        const auto& bounds  = model.weights().head_rows;
        const auto rows_of = [&](std::size_t r) { return static_cast<std::int32_t>(head_rows(r)); };
        const auto on      = [&](std::size_t r, const auto& f) {
            RankBinding bind(device, r);
            f(ranks[r]);
        };
        std::vector<Tensor> mixed(n);
        for (std::size_t r = 0; r < n; ++r) {
            on(r, [&](RankState& rank) {
                mixed[r] = head_input(parts, rank, r == 0 ? final_mixer : rank_final_mixers[r - 1],
                                      t, logit_rows);
            });
        }
        RankState& first    = ranks[0];
        auto scope          = first.workspace->scope();
        std::int32_t widest = 0;
        for (std::size_t r = 0; r < n; ++r) { widest = std::max(widest, rows_of(r)); }
        const Tensor buffer = first.workspace->alloc(DType::BF16, {widest, head_columns});
        std::size_t slot    = 0;
        for (std::int32_t c = 0; c < columns; c += head_columns, slot ^= 1U) {
            const std::int32_t count = std::min(head_columns, columns - c);
            const auto input         = [&](std::size_t r) {
                return Tensor(static_cast<std::byte*>(mixed[r].data) + std::size_t(c) * h * 2,
                              DType::BF16, {h, count});
            };
            for (std::size_t r = 1; r < n; ++r) {
                on(r, [&](RankState& rank) {
                    Tensor part(rank.logits, DType::BF16, {rows_of(r), count});
                    project(rank_heads[r - 1], input(r), part, *rank.workspace, rank.stream);
                    head_links[r - 1].send(part.data, part.bytes(), slot, rank.stream);
                });
            }
            on(0, [&](RankState& rank) {
                const auto place = [&](std::size_t r) {
                    const std::size_t row_bytes = std::size_t(rows_of(r)) * 2;
                    CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::byte*>(rank.logits) +
                                                     (std::size_t(c) * vocab + bounds[r]) * 2,
                                                 vocab * 2, buffer.data, row_bytes, row_bytes,
                                                 std::size_t(count), cudaMemcpyDeviceToDevice,
                                                 rank.stream));
                };
                Tensor own(buffer.data, DType::BF16, {rows_of(0), count});
                project(head, input(0), own, *rank.workspace, rank.stream);
                place(0);
                for (std::size_t r = 1; r < n; ++r) {
                    head_links[r - 1].recv(buffer.data,
                                           std::size_t(rows_of(r)) * std::size_t(count) * 2, slot,
                                           rank.stream);
                    place(r);
                }
            });
        }
    }

    // The read of the last `logit_rows` columns of `rank`'s stack through `mixer` (the rank's
    // final mixer), or of each branch's last column in a branch pass, which copies those columns
    // to the front of the stack: BF16 [hidden, columns] in rank.mixed.
    Tensor head_input(std::span<const Part> parts, RankState& rank, const HcPlan& mixer,
                      std::int32_t t, std::uint32_t logit_rows) {
        const auto h                   = static_cast<std::int32_t>(config.hidden_size);
        const auto hc                  = static_cast<std::int32_t>(config.hc_count);
        const std::size_t first_branch = branch_capture != nullptr ? 1 : 0;
        const auto n                   = static_cast<std::int32_t>(logit_rows);
        if (branch_origin != nullptr) {
            // Each branch's last column, to the front of the stack in branch order. A branch's
            // last column lies at or past its index, so no copy overwrites a later one's source.
            const std::size_t bytes = std::size_t(hc) * h * 4;
            for (std::size_t b = 0; b + first_branch < parts.size(); ++b) {
                const Part& part       = parts[b + first_branch];
                const std::size_t last = std::size_t(part.column + part.count - 1);
                if (last == b) { continue; }
                CUDA_CHECK(cudaMemcpyAsync(rank.stack + b * std::size_t(hc) * h,
                                           rank.stack + last * std::size_t(hc) * h, bytes,
                                           cudaMemcpyDeviceToDevice, rank.stream));
            }
        }
        const std::size_t first_row = branch_origin != nullptr ? 0 : std::size_t(t - n);
        const Tensor last(rank.stack + std::size_t(hc) * h * first_row, DType::FP32, {h, hc, n});
        Tensor head_mixed(rank.mixed, DType::BF16, {h, n});
        auto scope = rank.workspace->scope();
        ops::hyper_connection_read(last, mixer.weights(), config.rms_norm_eps, *rank.workspace,
                                   head_mixed, nullptr, rank.stream);
        return head_mixed;
    }

    // Tensor parallelism: one segment on every rank. Each rank runs every layer over its replica
    // of the stack. A layer's mixer and its experts each end in an exchange of the ranks' FP32
    // partials (TensorParallelExchange), whose rank-order sum every rank stores in rank.y: the
    // mixer's rounded to BF16, as one device's output projection stores it, the experts' in FP32.
    // The hyper-connections, the PLE and the router run on every rank from equal inputs, so the
    // replicas stay equal; every rank runs its run of the shared expert and projects its rows of
    // the head (run_divided_head). The other ranks' streams fork from rank 0's and join it again,
    // so a captured segment spans every rank.
    void run_tensor_segment(std::span<const Part> parts, const Segment& segment, std::int32_t t,
                            std::uint32_t logit_rows) {
        if (const std::vector<ColumnRun> pieces = tensor_pieces(parts, t); !pieces.empty()) {
            run_tensor_pieces(parts.front(), segment, t, logit_rows, pieces);
            return;
        }
        const auto h                     = static_cast<std::int32_t>(config.hidden_size);
        const auto hc                    = static_cast<std::int32_t>(config.hc_count);
        const std::size_t n              = this->parts();
        const std::size_t bytes          = std::size_t(h) * std::size_t(t) * 4;
        const std::size_t first_branch   = branch_capture != nullptr ? 1 : 0;
        TensorParallelExchange& exchange = *tensor_exchange;
        const auto on = [&](std::size_t r, const auto& f) {
            RankBinding bind(device, r);
            f(ranks[r]);
        };
        on(0, [&](RankState& rank) { CUDA_CHECK(cudaEventRecord(rank.fork, rank.stream)); });
        for (std::size_t r = 1; r < n; ++r) {
            on(r, [&](RankState& rank) {
                CUDA_CHECK(cudaStreamWaitEvent(rank.stream, ranks[0].fork, 0));
            });
        }
        // One exchange: ranks [0, contributors) write their partials, every rank stores the sum.
        const TensorParallelPartials partials =
            prompt_pass ? options.tensor_parallel_prompt_partials : TensorParallelPartials::Fp32;
        const auto exchange_step = [&](std::size_t contributors, TensorParallelSum sum,
                                       const auto& partial) {
            const std::size_t slot = exchange_next;
            exchange_next ^= 1U;
            for (std::size_t r = 0; r < n; ++r) {
                on(r, [&](RankState& rank) {
                    const bool contributes = r < contributors;
                    if (contributes) { partial(r, rank, exchange.partial(r, slot)); }
                    exchange.publish(r, slot, contributes ? bytes : 0, rank.y, rank.stream, sum, {},
                                     partials);
                });
            }
            for (std::size_t r = 0; r < n; ++r) {
                on(r, [&](RankState& rank) {
                    exchange.reduce(r, slot, rank.y, std::int64_t(h) * t, contributors, rank.stream,
                                    sum, {}, partials);
                });
            }
            for (std::size_t r = 0; r < n; ++r) {
                on(r, [&](RankState& rank) {
                    exchange.complete(r, slot, rank.y, std::int64_t(h) * t, rank.stream, sum, {},
                                      partials);
                });
            }
        };
        const auto stack_of = [&](RankState& rank) {
            return Tensor(rank.stack, DType::FP32, {h, hc, t});
        };
        const auto mixed_of = [&](RankState& rank) {
            return Tensor(rank.mixed, DType::BF16, {h, t});
        };
        const auto inject_of = [&](RankState& rank) {
            return Tensor(rank.inject, DType::FP32, {hc, t});
        };
        if (segment.embed) {
            for (std::size_t r = 0; r < n; ++r) {
                on(r, [&](RankState& rank) {
                    Tensor mixed = mixed_of(rank);
                    Tensor stack = stack_of(rank);
                    embed(rank, t, mixed);
                    ops::hyper_connection_expand(mixed, stack, rank.stream);
                });
            }
        }
        // A layer's MoE output is written into the stack by the next layer's read, unless the PLE
        // comes between.
        bool pending = false;
        for (std::size_t i = segment.begin; i < segment.end; ++i) {
            for (std::size_t r = 0; r < n; ++r) {
                on(r, [&](RankState& rank) {
                    const LayerPlan& plan = plan_of(r, i);
                    Tensor stack          = stack_of(rank);
                    Tensor mixed          = mixed_of(rank);
                    Tensor inject         = inject_of(rank);
                    const Tensor moe_out(rank.y, DType::FP32, {h, t});
                    bool write_read = pending;
                    if (plan.ple) {
                        if (pending) {
                            ops::hyper_connection_write(stack, moe_out, inject, rank.stream);
                        }
                        write_read = false;
                        for (std::size_t k = 0; k < parts.size(); ++k) {
                            LayerState& state = state_of(*parts[k].sequence, r, i);
                            if (k > first_branch) {
                                restore_branch_layer(state, state_slot(i, r), rank, kPleRegions);
                            }
                            run_ple(plan, state, rank, parts[k]);
                            if (k == 0) {
                                capture_branch_layer(state, state_slot(i, r), rank, kPleRegions);
                            }
                        }
                    }
                    auto scope = rank.workspace->scope();
                    if (write_read) {
                        ops::hyper_connection_write_read(stack, moe_out, inject,
                                                         plan.attn_hc.weights(), config.rms_norm_eps,
                                                         *rank.workspace, mixed, &inject, rank.stream);
                    } else {
                        ops::hyper_connection_read(stack, plan.attn_hc.weights(), config.rms_norm_eps,
                                                   *rank.workspace, mixed, &inject, rank.stream);
                    }
                });
            }
            // Every rank runs its key heads of a Gated DeltaNet layer; the ranks holding KV heads
            // of a sparse-attention layer run those.
            const std::size_t mixers = layers[i].gdn ? n : model.attention_ranks();
            if (kv_only_pass && i + 1 == layers.size()) {
                // A K/V-only last layer: the ranks holding its KV heads append the pass's keys,
                // values and indexer keys, and no exchange follows.
                for (std::size_t r = 0; r < mixers; ++r) {
                    on(r, [&](RankState& rank) {
                        run_mixer(
                            plan_of(r, i), i, rank, parts, t, state_slot(i, r),
                            [&](const Part& part) -> LayerState& {
                                return state_of(*part.sequence, r, i);
                            },
                            nullptr, true);
                    });
                }
                pending = false;
                break;
            }
            exchange_step(mixers, TensorParallelSum::StoreBf16,
                          [&](std::size_t r, RankState& rank, float* partial) {
                              run_mixer(plan_of(r, i), i, rank, parts, t, state_slot(i, r),
                                        [&](const Part& part) -> LayerState& {
                                            return state_of(*part.sequence, r, i);
                                        },
                                        partial);
                          });
            for (std::size_t r = 0; r < n; ++r) {
                on(r, [&](RankState& rank) {
                    Tensor stack  = stack_of(rank);
                    Tensor mixed  = mixed_of(rank);
                    Tensor inject = inject_of(rank);
                    auto scope    = rank.workspace->scope();
                    ops::hyper_connection_write_read(
                        stack, Tensor(rank.y, DType::BF16, {h, t}), inject,
                        plan_of(r, i).mlp_hc.weights(), config.rms_norm_eps, *rank.workspace, mixed,
                        &inject, rank.stream);
                });
            }
            exchange_step(n, TensorParallelSum::StoreFp32,
                          [&](std::size_t r, RankState& rank, float* partial) {
                              run_moe_share(plan_of(r, i), rank, i, partial,
                                            ColumnRun{.column = 0, .count = t});
                          });
            pending = true;
            if (taps(i)) {
                // Every replica completes its stack; rank 0, the head rank, keeps the features.
                for (std::size_t r = 0; r < n; ++r) {
                    on(r, [&](RankState& rank) {
                        Tensor stack  = stack_of(rank);
                        Tensor inject = inject_of(rank);
                        ops::hyper_connection_write(stack, Tensor(rank.y, DType::FP32, {h, t}),
                                                    inject, rank.stream);
                    });
                }
                pending = false;
                on(0, [&](RankState& rank) { capture_tap(rank, rank.stack, i); });
            }
        }
        for (std::size_t r = 0; r < n && pending; ++r) {
            on(r, [&](RankState& rank) {
                Tensor stack  = stack_of(rank);
                Tensor inject = inject_of(rank);
                ops::hyper_connection_write(stack, Tensor(rank.y, DType::FP32, {h, t}), inject,
                                            rank.stream);
            });
        }
        if (segment.head && prompt_pass && logit_rows > std::uint32_t(head_columns)) {
            on(0, [&](RankState& rank) { run_whole_head(parts, rank, t, logit_rows); });
        } else if (segment.head && logit_rows > 0) {
            run_divided_head(parts, t, logit_rows);
        }
        for (std::size_t r = 1; r < n; ++r) {
            on(r, [&](RankState& rank) { CUDA_CHECK(cudaEventRecord(rank.join, rank.stream)); });
            on(0, [&](RankState& rank) {
                CUDA_CHECK(cudaStreamWaitEvent(rank.stream, ranks[r].join, 0));
            });
        }
    }

    // The pieces of a tensor-parallel pass over one prompt chunk of `t` columns: kTensorPieceColumns
    // each from the chunk's first column, the last one the rest. None for a chunk no wider than one
    // piece, BF16 or 8-bit prompt partials (no mixer_out plane), several parts, a verification, a
    // branch pass or a media prompt (whose three-axis positions hold the whole chunk).
    [[nodiscard]] std::vector<ColumnRun> tensor_pieces(std::span<const Part> parts,
                                                       std::int32_t t) const {
        if (parts.size() != 1 || verifying || branch_origin != nullptr ||
            branch_capture != nullptr || mrope || ranks.front().mixer_out == nullptr ||
            t <= kTensorPieceColumns) {
            return {};
        }
        std::vector<ColumnRun> pieces;
        for (std::int32_t column = 0; column < t; column += kTensorPieceColumns) {
            pieces.push_back(
                {.column = column, .count = std::min(kTensorPieceColumns, t - column)});
        }
        return pieces;
    }

    // run_tensor_segment over one prompt chunk in `pieces` (tensor_pieces): each exchange crosses a
    // piece at a time on the ranks' side streams while their compute streams run the next piece
    // (Pieces in core/tensor_parallel.h), and a rank's next step on a piece first takes back that
    // piece's exchange. A piece runs as a prompt chunk of its own columns does: its mixers carry the
    // Gated DeltaNet, convolution and indexer states and the PLE history on to the next piece, its
    // sparse attention reads the keys up to its own last position, and its experts route and run its
    // own tokens; so the chunk gives what chunks of the pieces' widths give. The mixer's exchange
    // stores into rank.mixer_out, the experts' into rank.y, which a later piece's mixer output
    // there would overlap.
    void run_tensor_pieces(const Part& whole, const Segment& segment, std::int32_t t,
                           std::uint32_t logit_rows, std::span<const ColumnRun> pieces) {
        const auto h                     = static_cast<std::int32_t>(config.hidden_size);
        const auto hc                    = static_cast<std::int32_t>(config.hc_count);
        const std::size_t n              = this->parts();
        TensorParallelExchange& exchange = *tensor_exchange;
        const auto on = [&](std::size_t r, const auto& f) {
            RankBinding bind(device, r);
            f(ranks[r]);
        };
        std::vector<Part> parts;
        for (const ColumnRun& run : pieces) {
            parts.push_back(Part{.sequence = whole.sequence,
                                 .column   = run.column,
                                 .count    = run.count,
                                 .offset   = whole.offset + std::uint32_t(run.column)});
        }
        const auto stack_of = [&](RankState& rank, const ColumnRun& run) {
            return Tensor(rank.stack + std::size_t(hc) * std::size_t(h) * std::size_t(run.column),
                          DType::FP32, {h, hc, run.count});
        };
        const auto mixed_of = [&](RankState& rank, const ColumnRun& run) {
            return Tensor(columns(rank.mixed, h, run.column, 2), DType::BF16, {h, run.count});
        };
        const auto inject_of = [&](RankState& rank, const ColumnRun& run) {
            return Tensor(rank.inject + std::size_t(hc) * std::size_t(run.column), DType::FP32,
                          {hc, run.count});
        };
        const auto moe_out_of = [&](RankState& rank, const ColumnRun& run) {
            return Tensor(static_cast<float*>(rank.y) + std::size_t(h) * std::size_t(run.column),
                          DType::FP32, {h, run.count});
        };
        const auto mixer_out_of = [&](RankState& rank, const ColumnRun& run) {
            return Tensor(columns(rank.mixer_out, h, run.column, 2), DType::BF16, {h, run.count});
        };
        on(0, [&](RankState& rank) { CUDA_CHECK(cudaEventRecord(rank.fork, rank.stream)); });
        for (std::size_t r = 1; r < n; ++r) {
            on(r, [&](RankState& rank) {
                CUDA_CHECK(cudaStreamWaitEvent(rank.stream, ranks[0].fork, 0));
            });
        }
        // Whether the pieces' last exchanges may still be crossing.
        bool crossing        = false;
        const auto take_back = [&](std::size_t r, RankState& rank, std::size_t p) {
            if (crossing) {
                CUDA_CHECK(cudaStreamWaitEvent(rank.stream, exchange.piece_done(r, p), 0));
            }
        };
        // One exchange in pieces: every rank takes back piece p's last exchange and runs
        // `compute(r, rank, p, partial)`, ranks [0, contributors) their share into `partial`, then
        // hands the piece to its side stream, which publishes, reduces and completes it into the
        // piece's columns of `out(rank, run)`, the output plane.
        const TensorParallelPartials partials =
            prompt_pass ? options.tensor_parallel_prompt_partials : TensorParallelPartials::Fp32;
        const auto piece_step = [&](std::size_t contributors, TensorParallelSum sum,
                                    const auto& out, const auto& compute) {
            const std::size_t slot = exchange_next;
            exchange_next ^= 1U;
            for (std::size_t p = 0; p < pieces.size(); ++p) {
                const ColumnRun& run      = pieces[p];
                const std::int64_t values = std::int64_t(h) * run.count;
                const TensorParallelPiece fence{.index  = p,
                                                .offset = std::int64_t(h) * run.column};
                for (std::size_t r = 0; r < n; ++r) {
                    on(r, [&](RankState& rank) {
                        take_back(r, rank, p);
                        const bool contributes = r < contributors;
                        compute(r, rank, p, contributes ? exchange.partial(r, slot) : nullptr);
                        const cudaStream_t side = exchange.side_stream(r);
                        CUDA_CHECK(cudaEventRecord(exchange.piece_ready(r, p), rank.stream));
                        CUDA_CHECK(cudaStreamWaitEvent(side, exchange.piece_ready(r, p), 0));
                        exchange.publish(r, slot, contributes ? std::size_t(values) * 4 : 0,
                                         out(rank, run), side, sum, fence, partials);
                    });
                }
                for (std::size_t r = 0; r < n; ++r) {
                    on(r, [&](RankState& rank) {
                        exchange.reduce(r, slot, out(rank, run), values, contributors,
                                        exchange.side_stream(r), sum, fence, partials);
                    });
                }
                for (std::size_t r = 0; r < n; ++r) {
                    on(r, [&](RankState& rank) {
                        const cudaStream_t side = exchange.side_stream(r);
                        exchange.complete(r, slot, out(rank, run), values, side, sum, fence,
                                          partials);
                        CUDA_CHECK(cudaEventRecord(exchange.piece_done(r, p), side));
                    });
                }
            }
            crossing = true;
        };
        const auto mixer_plane = [&](RankState& rank, const ColumnRun& run) -> void* {
            return mixer_out_of(rank, run).data;
        };
        const auto moe_plane = [&](RankState& rank, const ColumnRun& run) -> void* {
            return moe_out_of(rank, run).data;
        };
        if (segment.embed) {
            for (std::size_t r = 0; r < n; ++r) {
                on(r, [&](RankState& rank) {
                    Tensor mixed(rank.mixed, DType::BF16, {h, t});
                    Tensor stack(rank.stack, DType::FP32, {h, hc, t});
                    embed(rank, t, mixed);
                    ops::hyper_connection_expand(mixed, stack, rank.stream);
                });
            }
        }
        // Every piece's experts' output, which the next layer's read writes into its stack unless
        // the PLE comes between.
        bool pending = false;
        for (std::size_t i = segment.begin; i < segment.end; ++i) {
            // Every rank runs its key heads of a Gated DeltaNet layer; the ranks holding KV heads
            // of a sparse-attention layer run those.
            const std::size_t mixers = layers[i].gdn ? n : model.attention_ranks();
            if (kv_only_pass && i + 1 == layers.size()) {
                // A K/V-only last layer: piece by piece, once its experts' exchange is back, the
                // ranks holding the layer's KV heads read the piece's MoE input and append its
                // keys, values and indexer keys; no exchange follows.
                for (std::size_t p = 0; p < pieces.size(); ++p) {
                    for (std::size_t r = 0; r < mixers; ++r) {
                        on(r, [&](RankState& rank) {
                            const ColumnRun& run  = pieces[p];
                            const LayerPlan& plan = plan_of(r, i);
                            Tensor stack          = stack_of(rank, run);
                            Tensor mixed          = mixed_of(rank, run);
                            Tensor inject         = inject_of(rank, run);
                            take_back(r, rank, p);
                            {
                                auto scope = rank.workspace->scope();
                                if (pending) {
                                    ops::hyper_connection_write_read(
                                        stack, moe_out_of(rank, run), inject,
                                        plan.attn_hc.weights(), config.rms_norm_eps,
                                        *rank.workspace, mixed, &inject, rank.stream);
                                } else {
                                    ops::hyper_connection_read(stack, plan.attn_hc.weights(),
                                                               config.rms_norm_eps, *rank.workspace,
                                                               mixed, &inject, rank.stream);
                                }
                            }
                            run_mixer(
                                plan, i, rank, std::span<const Part>(&parts[p], 1), t,
                                state_slot(i, r),
                                [&](const Part& part) -> LayerState& {
                                    return state_of(*part.sequence, r, i);
                                },
                                nullptr, true);
                        });
                    }
                }
                pending = false;
                break;
            }
            piece_step(
                mixers, TensorParallelSum::StoreBf16, mixer_plane,
                [&](std::size_t r, RankState& rank, std::size_t p, float* partial) {
                    const ColumnRun& run  = pieces[p];
                    const LayerPlan& plan = plan_of(r, i);
                    Tensor stack          = stack_of(rank, run);
                    Tensor mixed          = mixed_of(rank, run);
                    Tensor inject         = inject_of(rank, run);
                    bool write_read       = pending;
                    if (plan.ple) {
                        if (pending) {
                            ops::hyper_connection_write(stack, moe_out_of(rank, run), inject,
                                                        rank.stream);
                        }
                        write_read = false;
                        run_ple(plan, state_of(*whole.sequence, r, i), rank, parts[p]);
                    }
                    {
                        auto scope = rank.workspace->scope();
                        if (write_read) {
                            ops::hyper_connection_write_read(
                                stack, moe_out_of(rank, run), inject, plan.attn_hc.weights(),
                                config.rms_norm_eps, *rank.workspace, mixed, &inject, rank.stream);
                        } else {
                            ops::hyper_connection_read(stack, plan.attn_hc.weights(),
                                                       config.rms_norm_eps, *rank.workspace, mixed,
                                                       &inject, rank.stream);
                        }
                    }
                    if (partial == nullptr) { return; }
                    run_mixer(plan, i, rank, std::span<const Part>(&parts[p], 1), t,
                              state_slot(i, r),
                              [&](const Part& part) -> LayerState& {
                                  return state_of(*part.sequence, r, i);
                              },
                              partial);
                });
            piece_step(n, TensorParallelSum::StoreFp32, moe_plane,
                       [&](std::size_t r, RankState& rank, std::size_t p, float* partial) {
                           const ColumnRun& run = pieces[p];
                           Tensor stack         = stack_of(rank, run);
                           Tensor mixed         = mixed_of(rank, run);
                           Tensor inject        = inject_of(rank, run);
                           {
                               auto scope = rank.workspace->scope();
                               ops::hyper_connection_write_read(
                                   stack, mixer_out_of(rank, run), inject,
                                   plan_of(r, i).mlp_hc.weights(), config.rms_norm_eps,
                                   *rank.workspace, mixed, &inject, rank.stream);
                           }
                           run_moe_share(plan_of(r, i), rank, i, partial, run);
                       });
            pending = true;
            if (taps(i)) {
                // Every replica completes its stack; rank 0, the head rank, keeps the features.
                for (std::size_t p = 0; p < pieces.size(); ++p) {
                    for (std::size_t r = 0; r < n; ++r) {
                        on(r, [&](RankState& rank) {
                            take_back(r, rank, p);
                            Tensor stack  = stack_of(rank, pieces[p]);
                            Tensor inject = inject_of(rank, pieces[p]);
                            ops::hyper_connection_write(stack, moe_out_of(rank, pieces[p]), inject,
                                                        rank.stream);
                        });
                    }
                }
                pending = false;
                on(0, [&](RankState& rank) { capture_tap(rank, rank.stack, i); });
            }
        }
        // Each compute stream takes its pieces' last exchanges back before the segment ends.
        for (std::size_t p = 0; p < pieces.size(); ++p) {
            for (std::size_t r = 0; r < n; ++r) {
                on(r, [&](RankState& rank) {
                    take_back(r, rank, p);
                    if (!pending) { return; }
                    Tensor stack  = stack_of(rank, pieces[p]);
                    Tensor inject = inject_of(rank, pieces[p]);
                    ops::hyper_connection_write(stack, moe_out_of(rank, pieces[p]), inject,
                                                rank.stream);
                });
            }
        }
        const std::span<const Part> chunk(&whole, 1);
        if (segment.head && prompt_pass && logit_rows > std::uint32_t(head_columns)) {
            on(0, [&](RankState& rank) { run_whole_head(chunk, rank, t, logit_rows); });
        } else if (segment.head && logit_rows > 0) {
            run_divided_head(chunk, t, logit_rows);
        }
        for (std::size_t r = 1; r < n; ++r) {
            on(r, [&](RankState& rank) { CUDA_CHECK(cudaEventRecord(rank.join, rank.stream)); });
            on(0, [&](RankState& rank) {
                CUDA_CHECK(cudaStreamWaitEvent(rank.stream, ranks[r].join, 0));
            });
        }
    }

    // A rank's share of a layer's experts under tensor parallelism, for the pass's columns `run`
    // (every column, or a prompt piece's). Every rank routes the call itself from the same MoE
    // input, runs the routed pairs of its experts and its run of the shared expert into a 2^-32
    // fixed-point part, and stores the part in those columns of its FP32 `partial` [hidden, t].
    void run_moe_share(const LayerPlan& plan, RankState& rank, std::size_t index, float* partial,
                       const ColumnRun& run) {
        const MoePlan& m     = plan.moe;
        WorkspaceArena& ws   = *rank.workspace;
        const cudaStream_t s = rank.stream;
        const auto h         = static_cast<std::int32_t>(config.hidden_size);
        const auto top       = static_cast<std::int32_t>(config.num_experts_per_tok);
        const bool first     = rank.rank == 0;
        const std::int32_t n = run.count;
        const Tensor mixed(columns(rank.mixed, h, run.column, 2), DType::BF16, {h, n});
        // Rank 0's routes are the ones the route records keep.
        Tensor ids(columns(first ? route_records[index] : static_cast<void*>(rank.route_ids), top,
                           run.column, 4),
                   DType::I32, {top, n});
        Tensor weights(columns(rank.route_weights, top, run.column, 4), DType::FP32, {top, n});
        Tensor shared(rank.route_shared + run.column, DType::FP32, {n});
        {
            auto scope = ws.scope();
            ops::moe_route(mixed, m.router, m.shared_gate, ws, ids, weights, shared, s,
                           route_done(rank));
        }
        auto scope   = ws.scope();
        Tensor fixed = ws.alloc(DType::I64, {h, n});
        if (m.latent) {
            ops::moe_latent_gguf_part(mixed, ids, weights, &shared, m.latent_banks(),
                                      static_cast<std::int32_t>(m.group_begin),
                                      static_cast<std::int32_t>(m.group_end), ws, fixed, s,
                                      &rank.side);
        } else {
            ops::moe_experts_gguf_part(
                mixed, ids, weights, &shared, m.banks(), static_cast<std::int32_t>(m.expert_begin),
                static_cast<std::int32_t>(m.expert_end), ws, fixed, s, &rank.side);
        }
        Tensor y(partial + std::size_t(h) * std::size_t(run.column), DType::FP32, {h, n});
        ops::moe_experts_gguf_combine(std::span<const Tensor>(&fixed, 1), y, s);
    }

    // Whether a pass over one part of `t` tokens of `sequence` (a decode step, or a verification
    // while `verifying`) runs from the graphs of `shape`, a graph per segment: false at the
    // shape's first run, else they are captured, or updated, for the sequence (PassGraphs).
    bool bind_segments(PassGraphs& shape, SequenceState& sequence, std::int32_t t) {
        if (shape.graphs.empty() && shape.runs++ == 0) { return false; }
        if (!shape.graphs.empty() && shape.sequence == &sequence) { return true; }
        shape.graphs.resize(segments.size());
        for (std::size_t i = 0; i < segments.size(); ++i) {
            const Segment& segment = segments[i];
            RankBinding bind(device, segment.rank);
            DecodeGraphDefinition definition;
            const Part part{.sequence = &sequence, .column = 0, .count = t};
            definition.capture(ranks[segment.rank].stream, [&] {
                run_segment(std::span(&part, 1), segment, t, std::uint32_t(t));
            });
            if (shape.graphs[i].ready()) {
                shape.graphs[i].update(definition);
            } else {
                shape.graphs[i].instantiate(definition);
            }
        }
        shape.sequence = &sequence;
        return true;
    }

    // Whether a prompt pass that leaves no logits may run its last layer K/V-only: the layer is a
    // sparse-attention one, no DFlash2 tap reads it, and no MTP block follows the prompt over the
    // final stacks (mtp_catch_up reads every chunk's).
    [[nodiscard]] bool kv_only_allowed() const noexcept {
        return !mtp && !layers.empty() && layers.back().qsa &&
               (!drafter || tap_of[layers.size() - 1] < 0);
    }

    // `none`: the call may leave no logits (a prompt step's).
    void check_tokens(std::span<const std::int32_t> tokens, std::uint32_t logit_rows,
                      bool none = false) const {
        if (tokens.empty() || tokens.size() > options.prefill_chunk) {
            throw std::invalid_argument("qwen4_exp forward: 1..prefill_chunk tokens per call");
        }
        if ((logit_rows == 0 && !none) || logit_rows > tokens.size() ||
            logit_rows > max_logit_rows) {
            throw std::invalid_argument("qwen4_exp forward: invalid logit rows");
        }
        for (const auto token : tokens) {
            if (token < 0 || std::uint32_t(token) >= config.vocab_size) {
                throw std::invalid_argument("qwen4_exp forward: token outside the vocabulary");
            }
        }
    }

    // One pass over `parts`, whose tokens are `tokens` in column order. A pass of one part, a
    // decode step or a verification of up to kVectorTokens tokens, runs from its shape's graphs.
    void pass(std::span<const Part> parts, std::span<const std::int32_t> tokens,
              std::uint32_t logit_rows) {
        const auto t = static_cast<std::int32_t>(tokens.size());
        for (const Part& part : parts) {
            if (part.sequence->position + part.offset + std::uint32_t(part.count) >
                options.max_context) {
                throw std::invalid_argument("qwen4_exp forward: the sequence exceeds max_context");
            }
        }
        settle_routes();
        // A copy queued for an earlier pass may predate the cache's last swaps.
        forget_prefetches();
        place_pools(t > kVectorTokens && !served_by_misses(t));
        if (hybrid) { hybrid->begin(cancelled, !warming); }
        stage_inputs(parts, tokens);
        // Disk-resident experts need the host between a layer's routing and its experts, so their
        // passes stay eager, and so does a batch of several sequences. A verification of one
        // sequence replays its graphs while its experts take the vector products. Native host
        // experts coordinate with their supervisor through captured handshakes.
        SequenceState& first = *parts.front().sequence;
        // Staged misses keep disk experts' decode and verification free of host synchronization.
        const bool host_free = !stream || misses;
        // A K/V-only pass runs eagerly: the decode graphs end in the head.
        const bool decode_graph = options.cuda_graphs && t == 1 && host_free && !verifying &&
                                  branch_origin == nullptr && !kv_only_pass;
        const bool verify_graph = options.cuda_graphs && verifying && parts.size() == 1 &&
                                  host_free && t <= kVectorTokens && branch_origin == nullptr;
        if (misses) { misses->keep_alive(); }
        // Every segment of a tensor-parallel pass makes an even number of exchanges, so each one,
        // captured or not, starts in slot 0.
        exchange_next     = 0;
        PassGraphs* shape = decode_graph   ? &decode_graphs
                            : verify_graph ? &verify_graphs.at(std::size_t(t))
                                           : nullptr;
        const bool replay = shape != nullptr && bind_segments(*shape, first, t);
        for (std::size_t i = 0; i < segments.size(); ++i) {
            const Segment& segment = segments[i];
            if (i > 0 && segments[i - 1].rank != segment.rank) {
                cross(segments[i - 1].rank, segment.rank, t);
            }
            RankBinding bind(device, segment.rank);
            // Under tensor parallelism every rank runs the PLE layer and takes its rows.
            for (std::size_t r = 0; segment.rows && table && r < ranks.size(); ++r) {
                if (r != segment.rank && !tensor()) { continue; }
                RankBinding rows(device, r);
                stage_rows(ranks[r]);
            }
            if (replay) {
                if (tensor()) { fence_uploads(segment.rank); }
                shape->graphs[i].launch(ranks[segment.rank].stream);
            } else {
                run_segment(parts, segment, t, logit_rows);
            }
        }
        if (hybrid) { hybrid->finish(); }
        if (exchange_next != 0) {
            throw std::logic_error("qwen4_exp: a tensor-parallel segment left an exchange unpaired");
        }
        // A verification commits its positions later (commit()).
        if (!verifying) {
            for (const Part& part : parts) {
                part.sequence->position += static_cast<std::uint32_t>(part.count);
            }
        }
        record_routes(static_cast<std::uint32_t>(t), layers.size() - (kv_only_pass ? 1 : 0));
        // While a staged layer waits on the device for its missing experts, no other launch may
        // load a kernel lazily: loading can wait for the device to go idle, the waiting kernel for
        // the misses' service, and the service's own calls for the loader. The caller's next
        // kernels therefore follow a pass that may have waited.
        if (served_by_misses(t)) { synchronize_ranks(); }
    }

    // Whether the staged misses serve a pass of `t` tokens: a decode step, a verification up to
    // their widest call, the joint one of all sequences included, or a prompt chunk up to
    // prompt_miss_tokens. A wider pass copies its layers' missing experts into the slot pools
    // instead. Either way a call past kVectorTokens takes the matrix kernel, as over device banks.
    [[nodiscard]] bool served_by_misses(std::int32_t t) const {
        return misses && (t <= kVectorTokens || (verifying && std::uint32_t(t) <= miss_tokens) ||
                          std::uint32_t(t) <= prompt_miss_tokens);
    }

    // Whether a prompt chunk of `t` tokens takes the staged misses rather than the slot pool.
    [[nodiscard]] bool narrow_chunk(std::int32_t t) const {
        return t <= kVectorTokens || std::uint32_t(t) <= prompt_miss_tokens;
    }

    // A graph replays after the work of the stream it is launched on, not after that of the other
    // ranks' streams, whose work in it hangs off that stream's fork: the inputs each other rank's
    // stream uploaded for the pass (ids, positions, RoPE positions, PLE rows) must be fenced in
    // first, or the graph can read the previous pass's.
    void fence_uploads(std::size_t launcher) {
        for (std::size_t r = 0; r < ranks.size(); ++r) {
            if (r == launcher) { continue; }
            {
                RankBinding bind(device, r);
                CUDA_CHECK(cudaEventRecord(ranks[r].uploaded, ranks[r].stream));
            }
            RankBinding bind(device, launcher);
            CUDA_CHECK(cudaStreamWaitEvent(ranks[launcher].stream, ranks[r].uploaded, 0));
        }
    }

    void synchronize_ranks() {
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            CUDA_CHECK(ninfer::wait_for_stream(rank.stream));
        }
    }

    // The tokens a prompt span takes, 0 without spans. Spans need one device and GGUF experts off
    // the GPU: a host layer's wide calls copy every uncached expert of the layer into the pool,
    // and a disk layer's read the ones its slots lack, which the chunks before them in the span
    // left there; per span instead of per chunk.
    [[nodiscard]] std::uint32_t planned_span_tokens() const {
        const bool native = std::any_of(layers.begin(), layers.end(),
                                        [](const LayerPlan& plan) { return plan.moe.native.has_value(); });
        // One device: its segments split only where the PLE layer's rows arrive.
        const bool one_rank = std::all_of(segments.begin(), segments.end(),
                                          [](const Segment& segment) { return segment.rank == 0; });
        if (model.options().experts == ExpertResidency::Device || native || !one_rank) { return 0; }
        const std::uint32_t tokens =
            std::min<std::uint32_t>(options.max_context, span_chunks * options.prefill_chunk);
        return tokens > options.prefill_chunk ? tokens : 0;
    }

    // The chunks a span takes (span_chunks), before anything is sized by it.
    void plan_span_chunks() {
        span_chunks = kSpanChunks;
        const std::uint64_t further =
            std::uint64_t(kLoneSpanChunks - kSpanChunks) * options.prefill_chunk;
        if (options.sequences != 1 ||
            options.max_context <= kSpanChunks * options.prefill_chunk) {
            return;
        }
        // A token's stack, MoE input and injection: nearly all of its planes (allocate_span).
        const std::uint64_t token_bytes =
            std::uint64_t(config.hc_count) * config.hidden_size * 4 + config.hidden_size * 2 +
            config.hc_count * 4;
        RankBinding bind(device, 0);
        std::size_t free_bytes = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total));
        if (further * token_bytes <= free_bytes / kLoneSpanShare) { span_chunks = kLoneSpanChunks; }
    }

    // The chunks one expert call of a span's expert phase takes: as many as kGroupTokens holds,
    // with host experts, whose groups run from the pool.
    [[nodiscard]] std::uint32_t planned_group_chunks() const {
        if (planned_span_tokens() == 0 || model.options().experts != ExpertResidency::Host) {
            return 1;
        }
        return std::clamp<std::uint32_t>(kGroupTokens / options.prefill_chunk, 1, kSpanChunks);
    }

    void allocate_span() {
        const std::uint32_t tokens = planned_span_tokens();
        if (tokens == 0) { return; }
        span_tokens       = tokens;
        const auto width  = std::size_t(config.hc_count) * config.hidden_size;
        RankBinding bind(device, 0);
        span_stack     = DeviceBuffer(std::size_t(tokens) * width * 4);
        span_positions = DeviceBuffer(std::size_t(tokens) * 4);
        span_rope      = DeviceBuffer(std::size_t(tokens) * 4);
        span_mixed     = DeviceBuffer(std::size_t(tokens) * config.hidden_size * 2);
        span_inject    = DeviceBuffer(std::size_t(tokens) * config.hc_count * 4);
        std::uint64_t bytes = span_stack.bytes + span_positions.bytes + span_rope.bytes +
                              span_mixed.bytes + span_inject.bytes;
        if (vision_context) {
            span_mrope = DeviceBuffer(std::size_t(tokens) * 3 * 4);
            bytes += span_mrope.bytes;
        }
        if (table) {
            span_row_bytes = std::size_t(config.ngram_heads()) *
                             ops::ngram_row_bytes(options.ngram->format);
            span_rows      = DeviceBuffer(std::size_t(tokens) * span_row_bytes);
            span_host_rows = std::make_unique<PinnedHostBuffer>(span_rows.bytes);
            bytes += span_rows.bytes;
        }
        span_group = planned_group_chunks();
        if (span_group > 1) {
            const std::size_t group = std::size_t(span_group) * options.prefill_chunk;
            const std::size_t pairs = group * config.num_experts_per_tok;
            span_ids     = DeviceBuffer(pairs * 4);
            span_weights = DeviceBuffer(pairs * 4);
            span_shared  = DeviceBuffer(group * 4);
            span_y       = DeviceBuffer(group * config.hidden_size * 4);
            bytes += span_ids.bytes + span_weights.bytes + span_shared.bytes + span_y.bytes;
        }
        memory.ranks[0].workspace_bytes += bytes;
    }

    // One chunk of a prompt span: `count` tokens of `sequence` from its position `first`, kept from
    // token `at` of the span planes. A span holds its chunks one after another, those of one
    // sequence in order.
    struct SpanChunk {
        SequenceState* sequence = nullptr;
        std::size_t at          = 0;
        std::int32_t count      = 0;
        std::uint32_t first     = 0;
        std::span<const std::int32_t> tokens;
    };

    // The chunk after the expert group that starts at chunk `first` of a span. A group needs the
    // full pool and kPrefetchTokens or more, so that it runs from the layer's prefetched pool copy;
    // a narrow chunk (narrow_chunk) runs alone, as forward() runs it.
    [[nodiscard]] std::size_t group_end(std::span<const SpanChunk> chunks, std::size_t first) const {
        if (span_group == 1 || slot_pools.empty() || slot_pools[0].slots < config.num_experts ||
            narrow_chunk(chunks[first].count)) {
            return first + 1;
        }
        std::size_t last = std::min<std::size_t>(chunks.size(), first + span_group);
        for (std::size_t j = first + 1; j < last; ++j) {
            if (narrow_chunk(chunks[j].count)) {
                last = j;
                break;
            }
        }
        const std::size_t tokens = chunks[last - 1].at + std::size_t(chunks[last - 1].count) -
                                   chunks[first].at;
        return tokens >= std::size_t(kPrefetchTokens) ? last : first + 1;
    }

    // The experts of a span's chunks first..last-1 in one wide call through the pool, into
    // span_y. rank.mixed holds the first chunk's MoE input and the others' follow it. Each chunk
    // is routed on its own, as forward() routes it, into the group's route planes.
    void run_group(const LayerPlan& plan, RankState& rank, std::size_t index,
                   std::span<const SpanChunk> chunks, std::size_t first, std::size_t last) {
        const MoePlan& m      = plan.moe;
        const cudaStream_t st = rank.stream;
        const auto h          = static_cast<std::int32_t>(config.hidden_size);
        const auto top        = static_cast<std::int32_t>(config.num_experts_per_tok);
        auto* const ids_p     = static_cast<std::int32_t*>(span_ids.p);
        auto* const weights_p = static_cast<float*>(span_weights.p);
        auto* const shared_p  = static_cast<float*>(span_shared.p);
        for (std::size_t j = first; j < last; ++j) {
            const auto t         = chunks[j].count;
            const std::size_t at = chunks[j].at - chunks[first].at;
            const Tensor mixed(static_cast<std::byte*>(rank.mixed) + at * std::size_t(h) * 2,
                               DType::BF16, {h, t});
            Tensor ids(ids_p + at * std::size_t(top), DType::I32, {top, t});
            Tensor weights(weights_p + at * std::size_t(top), DType::FP32, {top, t});
            Tensor shared(shared_p + at, DType::FP32, {t});
            auto scope = rank.workspace->scope();
            ops::moe_route(mixed, m.router, m.shared_gate, *rank.workspace, ids, weights, shared,
                           st, route_done(rank));
        }
        const auto t = static_cast<std::int32_t>(chunks[last - 1].at +
                                                 std::size_t(chunks[last - 1].count) -
                                                 chunks[first].at);
        Tensor ids(ids_p, DType::I32, {top, t});
        Tensor weights(weights_p, DType::FP32, {top, t});
        Tensor shared(shared_p, DType::FP32, {t});
        void* const own_y = rank.y;
        rank.y            = span_y.p;
        bool pooled       = false;
        try {
            pooled = run_through_slots(plan, rank, t, index, ids, weights, shared);
        } catch (...) {
            rank.y = own_y;
            throw;
        }
        rank.y = own_y;
        if (!pooled) { throw std::logic_error("qwen4_exp: a span group ran outside the pool"); }
    }

    // A prompt span of one sequence's chunks (prompt_step() tokens at most), layer by layer; with
    // a cut, its chunks restart there and cut.state takes the state there (Executor::forward).
    void forward_span(std::uint32_t s, std::span<const std::int32_t> tokens,
                      std::uint32_t logit_rows, std::span<const std::int32_t> next,
                      PromptCut cut) {
        const std::size_t c = options.prefill_chunk;
        if (span_tokens == 0 || tokens.size() > span_tokens) {
            throw std::invalid_argument("qwen4_exp forward: tokens exceed the prompt span");
        }
        SequenceState& sequence = sequences.at(s);
        if (sequence.position + tokens.size() > options.max_context) {
            throw std::invalid_argument("qwen4_exp forward: the sequence exceeds max_context");
        }
        std::vector<SpanChunk> chunks;
        const auto chunk_up = [&](std::size_t begin, std::size_t end) {
            for (std::size_t at = begin; at < end; at += c) {
                const auto t = static_cast<std::int32_t>(std::min(c, end - at));
                chunks.push_back({.sequence = &sequence,
                                  .at       = at,
                                  .count    = t,
                                  .first    = sequence.position + std::uint32_t(at),
                                  .tokens   = tokens.subspan(at, std::size_t(t))});
                check_tokens(chunks.back().tokens,
                             at + std::size_t(t) == tokens.size() ? logit_rows : 1, true);
            }
        };
        SpanCut span_cut;
        if (cut.state != nullptr) {
            chunk_up(0, cut.tokens);
            span_cut = {.chunk = chunks.size(), .state = cut.state};
            chunk_up(cut.tokens, tokens.size());
        } else {
            chunk_up(0, tokens.size());
        }
        const std::size_t last = chunks.size() - 1;
        const char wanted      = logit_rows > 0 ? 1 : 0;
        run_span(chunks, std::span(&last, 1), std::span(&wanted, 1), logit_rows, &sequence, next,
                 span_cut);
    }

    // Several sequences' prompt tokens in one span (Executor::forward_pieces).
    void forward_pieces(std::span<const PromptPiece> pieces, std::span<const std::int32_t> next) {
        const std::size_t c = options.prefill_chunk;
        if (span_tokens == 0) {
            throw std::invalid_argument("qwen4_exp forward_pieces: prompt spans are off");
        }
        if (pieces.empty() || pieces.size() > max_logit_rows) {
            throw std::invalid_argument("qwen4_exp forward_pieces: 1..max_logit_rows pieces");
        }
        std::size_t total = 0;
        std::vector<SpanChunk> chunks;
        std::vector<std::size_t> lasts;
        std::vector<char> wanted;
        for (const PromptPiece& piece : pieces) {
            SequenceState& sequence = sequences.at(piece.sequence);
            for (const PromptPiece& other : pieces) {
                if (&other != &piece && other.sequence == piece.sequence) {
                    throw std::invalid_argument("qwen4_exp forward_pieces: a sequence in two pieces");
                }
            }
            if (piece.tokens.empty() || sequence.position + piece.tokens.size() > options.max_context) {
                throw std::invalid_argument(
                    "qwen4_exp forward_pieces: a piece is empty or exceeds max_context");
            }
            const std::size_t prompt = sequence.media_rope.size() / 3;
            if (pieces.size() > 1 && prompt != 0 && sequence.position < prompt) {
                throw std::invalid_argument(
                    "qwen4_exp forward_pieces: a media prompt prefills in a span of its own");
            }
            for (std::size_t at = 0; at < piece.tokens.size(); at += c) {
                const auto t = static_cast<std::int32_t>(std::min(c, piece.tokens.size() - at));
                chunks.push_back({.sequence = &sequence,
                                  .at       = total + at,
                                  .count    = t,
                                  .first    = sequence.position + std::uint32_t(at),
                                  .tokens   = piece.tokens.subspan(at, std::size_t(t))});
                check_tokens(chunks.back().tokens, 1);
            }
            total += piece.tokens.size();
            lasts.push_back(chunks.size() - 1);
            wanted.push_back(piece.logits ? 1 : 0);
        }
        if (total > span_tokens || next.size() > prompt_step()) {
            throw std::invalid_argument("qwen4_exp forward_pieces: tokens exceed the prompt span");
        }
        run_span(chunks, lasts, wanted, 1, &sequences.at(pieces.back().sequence), next);
    }

    // A prompt span, layer by layer: every chunk passes a layer before any passes the next, so a
    // wide call's pool copy of the layer's uncached experts serves every chunk. The arithmetic of
    // each chunk is that of forward() over its sequence. The logits are the last `logit_rows` rows
    // of each chunk `lasts` names whose `wanted` entry is set, the chunk's columns in order, chunk
    // after chunk (several chunks take one row each, and an unwanted chunk's place stays as it
    // was); `hinted` then hints the rows of `next`. A `cut` keeps its sequence's state between two
    // of its chunks as each layer, the ring and the MTP block pass there. Where kv_only_allowed(),
    // the last layer's expert calls without a wanted chunk do not run, and their chunks run the
    // layer's appends alone.
    void run_span(std::span<const SpanChunk> chunks, std::span<const std::size_t> lasts,
                  std::span<const char> wanted, std::uint32_t logit_rows,
                  const SequenceState* hinted, std::span<const std::int32_t> next,
                  SpanCut cut = {}) {
        const std::size_t n = chunks.back().at + std::size_t(chunks.back().count);
        if (cut.state != nullptr) {
            if (cut.chunk == 0 || cut.chunk >= chunks.size() ||
                chunks[cut.chunk - 1].sequence != chunks[cut.chunk].sequence) {
                throw std::logic_error("qwen4_exp: a span's cut lies outside its sequence");
            }
            size_snapshot(*chunks[cut.chunk].sequence, *cut.state);
        }
        RankBinding bind(device, 0);
        RankState& rank      = ranks[0];
        const cudaStream_t st = rank.stream;
        const auto h         = static_cast<std::int32_t>(config.hidden_size);
        const auto hc        = static_cast<std::int32_t>(config.hc_count);
        const std::size_t width = std::size_t(hc) * std::size_t(h);
        const auto chunk_stack  = [&](std::size_t k) {
            return static_cast<float*>(span_stack.p) + chunks[k].at * width;
        };
        settle_routes();
        forget_prefetches();
        place_pools(true);
        float* const own_stack = rank.stack;
        // The chunks' inputs, staged one after another as forward() stages them, kept in the
        // span planes; their embeddings start their stacks. The span's n-gram rows are read in
        // one batch while the layers before the first PLE layer run.
        const bool reads_rows =
            table && std::any_of(segments.begin(), segments.end(),
                                 [](const Segment& segment) { return segment.rows; });
        const std::size_t heads = config.ngram_heads();
        if (reads_rows) { span_row_ids.resize(n * heads); }
        // Which chunks of a media prompt rotate at its three-axis positions (span_mrope).
        std::vector<char> chunk_mrope(chunks.size(), 0);
        for (std::size_t k = 0; k < chunks.size(); ++k) {
            const SpanChunk& chunk = chunks[k];
            const std::int32_t t   = chunk.count;
            const std::size_t at   = chunk.at;
            const Part part{.sequence = chunk.sequence, .column = 0, .count = t};
            chunk.sequence->position = chunk.first;
            stage_inputs(std::span(&part, 1), chunk.tokens, false);
            if (cut.state != nullptr && k + 1 == cut.chunk) {
                cut.state->context = chunk.sequence->context;
            }
            if (reads_rows) {
                std::copy(row_ids.begin(), row_ids.end(),
                          span_row_ids.begin() + std::ptrdiff_t(at * heads));
            }
            Tensor mixed(rank.mixed, DType::BF16, {h, t});
            embed(rank, t, mixed);
            Tensor stack(chunk_stack(k), DType::FP32, {h, hc, t});
            ops::hyper_connection_expand(mixed, stack, st);
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int32_t*>(span_positions.p) + at,
                                       rank.positions, std::size_t(t) * 4,
                                       cudaMemcpyDeviceToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int32_t*>(span_rope.p) + at, rank.rope,
                                       std::size_t(t) * 4, cudaMemcpyDeviceToDevice, st));
            if (mrope) {
                CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int32_t*>(span_mrope.p) + 3 * at,
                                           rank.mrope, 3 * std::size_t(t) * 4,
                                           cudaMemcpyDeviceToDevice, st));
                chunk_mrope[k] = 1;
            }
        }
        // Each sequence stands at its first chunk's position until the span ends.
        for (std::size_t k = chunks.size(); k-- > 0;) {
            chunks[k].sequence->position = chunks[k].first;
        }
        if (reads_rows) {
            table->submit(span_row_ids,
                          std::span(static_cast<std::uint8_t*>(span_host_rows->data()),
                                    n * span_row_bytes));
            rows_waited = false;
        }
        // The next step's rows are read while the device runs this span.
        if (hinted != nullptr) { hint_next(*hinted, next); }
        // The span's expert calls, the same in every layer: the groups and the wider chunks in span
        // order, then the narrow ones (narrow_chunk), whose misses enter cache slots that the
        // layer's pool table may name for the wider calls' experts.
        struct ExpertCall {
            std::size_t first = 0, last = 0;
        };
        std::vector<ExpertCall> calls;
        for (std::size_t k = 0; k < chunks.size();) {
            const std::size_t last = group_end(chunks, k);
            if (!narrow_chunk(chunks[k].count)) { calls.push_back({k, last}); }
            k = last;
        }
        const std::size_t wide = calls.size();
        for (std::size_t k = 0; k < chunks.size(); ++k) {
            if (narrow_chunk(chunks[k].count)) { calls.push_back({k, k + 1}); }
        }
        const SpanChunk& recorded = chunks[calls.back().last - 1];
        // The last layer's calls that run, and the chunks that run the whole layer: all, or with a
        // K/V-only last layer those that share a call with a chunk whose logits are wanted.
        std::vector<char> last_runs(calls.size(), 1), last_whole(chunks.size(), 1);
        if (kv_only_allowed()) {
            std::vector<char> read(chunks.size(), 0);
            for (std::size_t p = 0; p < lasts.size(); ++p) {
                if (wanted[p] != 0 && logit_rows > 0) { read[lasts[p]] = 1; }
            }
            for (std::size_t c = 0; c < calls.size(); ++c) {
                const auto first = read.begin() + std::ptrdiff_t(calls[c].first);
                const auto end   = read.begin() + std::ptrdiff_t(calls[c].last);
                last_runs[c]     = std::find(first, end, 1) != end ? 1 : 0;
                std::fill(last_whole.begin() + std::ptrdiff_t(calls[c].first),
                          last_whole.begin() + std::ptrdiff_t(calls[c].last), last_runs[c]);
            }
        }
        const std::size_t span_pooled_end =
            layers.size() -
            (std::find(last_runs.begin(), last_runs.end(), 1) == last_runs.end() ? 1 : 0);
        // DFlash2: each following sequence's last window positions of the span, chunk by chunk;
        // their features take consecutive columns, sequence after sequence. A cut's sequence
        // first takes its last window positions before the cut, which the ring holds there.
        std::vector<Capture> span_capture(chunks.size());
        std::vector<ContextRun> span_runs;
        std::size_t cut_runs = 0; // the runs that end at the cut
        if (drafter) {
            const auto window = static_cast<std::int32_t>(drafter_window());
            std::int32_t column = 0;
            // The last window positions of the sequence's chunks among chunks[begin, end).
            const auto keep_last = [&](SequenceState* sequence, std::size_t begin,
                                       std::size_t end) {
                std::int32_t total = 0;
                std::uint32_t last = 0;
                for (std::size_t k = begin; k < end; ++k) {
                    if (chunks[k].sequence != sequence) { continue; }
                    total += chunks[k].count;
                    last = chunks[k].first + std::uint32_t(chunks[k].count);
                }
                const std::int32_t keep = std::min(total, window);
                std::int32_t skip       = total - keep;
                span_runs.push_back({sequence, column, keep, last - std::uint32_t(keep)});
                for (std::size_t k = begin; k < end; ++k) {
                    if (chunks[k].sequence != sequence) { continue; }
                    const std::int32_t skipped = std::min(skip, chunks[k].count);
                    skip -= skipped;
                    const std::int32_t count = chunks[k].count - skipped;
                    if (count == 0) { continue; }
                    span_capture[k] = {.first = skipped, .count = count, .column = column};
                    column += count;
                }
            };
            std::vector<SequenceState*> seen;
            if (cut.state != nullptr && chunks[cut.chunk].sequence->drafter_follows) {
                SequenceState* sequence = chunks[cut.chunk].sequence;
                seen.push_back(sequence);
                keep_last(sequence, 0, cut.chunk);
                cut_runs = span_runs.size();
                keep_last(sequence, cut.chunk, chunks.size());
            }
            for (const SpanChunk& chunk : chunks) {
                SequenceState* sequence = chunk.sequence;
                if (!sequence->drafter_follows ||
                    std::find(seen.begin(), seen.end(), sequence) != seen.end()) {
                    continue;
                }
                seen.push_back(sequence);
                keep_last(sequence, 0, chunks.size());
            }
            if (std::uint32_t(column) > feature_columns) {
                throw std::logic_error("qwen4_exp: a span's DFlash2 features exceed their columns");
            }
        }
        void* const own_mixed   = rank.mixed;
        float* const own_inject = rank.inject;
        // Chunk k's MoE input and injection, kept between a layer's two phases.
        const auto chunk_planes = [&](std::size_t k) {
            rank.mixed  = static_cast<std::byte*>(span_mixed.p) + chunks[k].at * std::size_t(h) * 2;
            rank.inject = static_cast<float*>(span_inject.p) + chunks[k].at * std::size_t(hc);
        };
        try {
            pooled_end = span_pooled_end;
            // Each layer runs in two phases: attention over every chunk of the span, then the
            // experts over every chunk. The next layer's experts are copied into the pool while its
            // attention phase runs, so its first expert call rarely waits for them.
            for (std::size_t i = 0; i < layers.size(); ++i) {
                const LayerPlan& plan = layers[i];
                if (plan.ple && !rows_waited) {
                    await_rows(rank);
                    CUDA_CHECK(cudaMemcpyAsync(span_rows.p, span_host_rows->data(),
                                               n * span_row_bytes, cudaMemcpyHostToDevice, st));
                }
                for (std::size_t k = 0; k < chunks.size(); ++k) {
                    const SpanChunk& chunk = chunks[k];
                    const std::int32_t t   = chunk.count;
                    LayerState& state      = chunk.sequence->layers[i];
                    const Part part{.sequence = chunk.sequence, .column = 0, .count = t};
                    CUDA_CHECK(cudaMemcpyAsync(rank.positions,
                                               static_cast<std::int32_t*>(span_positions.p) +
                                                   chunk.at,
                                               std::size_t(t) * 4, cudaMemcpyDeviceToDevice, st));
                    CUDA_CHECK(cudaMemcpyAsync(rank.rope,
                                               static_cast<std::int32_t*>(span_rope.p) + chunk.at,
                                               std::size_t(t) * 4, cudaMemcpyDeviceToDevice, st));
                    mrope = chunk_mrope[k] != 0;
                    if (mrope) {
                        CUDA_CHECK(cudaMemcpyAsync(
                            rank.mrope, static_cast<std::int32_t*>(span_mrope.p) + 3 * chunk.at,
                            3 * std::size_t(t) * 4, cudaMemcpyDeviceToDevice, st));
                    }
                    if (plan.ple) {
                        CUDA_CHECK(cudaMemcpyAsync(rank.rows,
                                                   static_cast<std::byte*>(span_rows.p) +
                                                       chunk.at * span_row_bytes,
                                                   std::size_t(t) * span_row_bytes,
                                                   cudaMemcpyDeviceToDevice, st));
                    }
                    rank.stack = chunk_stack(k);
                    chunk_planes(k);
                    Tensor stack(rank.stack, DType::FP32, {h, hc, t});
                    Tensor mixed(rank.mixed, DType::BF16, {h, t});
                    Tensor inject(rank.inject, DType::FP32, {hc, t});
                    if (plan.ple) { run_ple(plan, state, rank, part); }
                    {
                        auto scope = rank.workspace->scope();
                        ops::hyper_connection_read(stack, plan.attn_hc.weights(),
                                                   config.rms_norm_eps, *rank.workspace, mixed,
                                                   &inject, st);
                    }
                    const bool kv_only = i + 1 == layers.size() && last_whole[k] == 0;
                    if (plan.gdn) {
                        run_gdn(plan, i, state, rank, part);
                    } else {
                        // The sequence stands at its first chunk; this chunk begins at its own.
                        AttentionInputs in = target_inputs(i, state, rank, part);
                        in.first           = chunk.first;
                        run_qsa(plan, state, rank, part, in, kv_only);
                    }
                    if (!kv_only) {
                        auto scope = rank.workspace->scope();
                        ops::hyper_connection_write_read(
                            stack, Tensor(rank.y, DType::BF16, {h, t}), inject,
                            plan.mlp_hc.weights(), config.rms_norm_eps, *rank.workspace, mixed,
                            &inject, st);
                    }
                    // The layer's state at the cut, before the next chunk moves it on.
                    if (cut.state != nullptr && k + 1 == cut.chunk) {
                        keep_layer(state, cut.state->layers.at(state_slot(i, 0)), st);
                    }
                }
                for (std::size_t c = 0; c < calls.size(); ++c) {
                    if (i + 1 == layers.size() && last_runs[c] == 0) { continue; }
                    const auto [k, last] = calls[c];
                    const auto t         = static_cast<std::int32_t>(
                        chunks[last - 1].at + std::size_t(chunks[last - 1].count) - chunks[k].at);
                    rank.stack = chunk_stack(k);
                    chunk_planes(k);
                    span_hold = c + 1 < wide;
                    span_last = c + 1 == calls.size();
                    Tensor stack(rank.stack, DType::FP32, {h, hc, t});
                    Tensor inject(rank.inject, DType::FP32, {hc, t});
                    if (last == k + 1) {
                        run_moe(plan, rank, t, i);
                        ops::hyper_connection_write(stack, Tensor(rank.y, DType::FP32, {h, t}),
                                                    inject, st);
                    } else {
                        run_group(plan, rank, i, chunks, k, last);
                        ops::hyper_connection_write(stack, Tensor(span_y.p, DType::FP32, {h, t}),
                                                    inject, st);
                        if (span_last) {
                            // record_routes() reads the last call's last chunk's routes from the
                            // layer's record, where run_moe() leaves them.
                            const auto top = std::size_t(config.num_experts_per_tok);
                            CUDA_CHECK(cudaMemcpyAsync(
                                route_records[i],
                                static_cast<std::int32_t*>(span_ids.p) +
                                    (recorded.at - chunks[k].at) * top,
                                top * std::size_t(recorded.count) * 4, cudaMemcpyDeviceToDevice,
                                st));
                        }
                    }
                }
                // Every chunk's stack has completed the layer: its DFlash2 features.
                if (drafter && tap_of[i] >= 0) {
                    for (std::size_t k = 0; k < chunks.size(); ++k) {
                        if (span_capture[k].count == 0) { continue; }
                        capture = span_capture[k];
                        capture_tap(rank, chunk_stack(k), i);
                    }
                    capture = {};
                }
            }
            span_hold   = false;
            span_last   = false;
            rank.mixed  = own_mixed;
            rank.inject = own_inject;
            // The head reads the last rows of each named chunk, chunk after chunk, so that each
            // gets the logits forward() leaves for it.
            const auto rows  = static_cast<std::int32_t>(logit_rows);
            const auto vocab = static_cast<std::int32_t>(config.vocab_size);
            for (std::size_t p = 0; p < lasts.size(); ++p) {
                if (rows == 0 || wanted[p] == 0) { continue; }
                const SpanChunk& chunk = chunks[lasts[p]];
                const Tensor last(chunk_stack(lasts[p]) + width * std::size_t(chunk.count - rows),
                                  DType::FP32, {h, hc, rows});
                Tensor head_mixed(rank.mixed, DType::BF16, {h, rows});
                {
                    auto scope = rank.workspace->scope();
                    ops::hyper_connection_read(last, final_mixer.weights(), config.rms_norm_eps,
                                               *rank.workspace, head_mixed, nullptr, st);
                }
                Tensor logits(static_cast<std::byte*>(rank.logits) +
                                  p * std::size_t(rows) * std::size_t(vocab) * 2,
                              DType::BF16, {vocab, rows});
                project(head, head_mixed, logits, *rank.workspace, st);
            }
            for (const SpanChunk& chunk : chunks) {
                chunk.sequence->position = chunk.first + std::uint32_t(chunk.count);
            }
            pooled_end = ~std::size_t{0};
            record_routes(std::uint32_t(recorded.count),
                          layers.size() - (last_runs.back() != 0 ? 0 : 1));
            // A cut's ring lane is kept once its run before the cut is in.
            const std::span<const ContextRun> runs(span_runs);
            dflash2_context(runs.first(cut_runs), nullptr, 0);
            if (drafter && cut.state != nullptr) {
                copy_ring(*chunks[cut.chunk].sequence, *cut.state, true);
            }
            dflash2_context(runs.subspan(cut_runs), nullptr, 0);
            // The MTP block catches up chunk by chunk over each chunk's own final stacks; the
            // next pass's MTP cells of each sequence start from its last chunk's stack, which it
            // copied. A cut keeps the block's state after the chunk before it.
            if (mtp) {
                for (std::size_t k = 0; k < chunks.size(); ++k) {
                    rank.stack = chunk_stack(k);
                    const Committed committed{chunks[k].sequence, 0, chunks[k].count,
                                              chunks[k].first, chunks[k].tokens};
                    mtp_catch_up(std::span(&committed, 1));
                    if (cut.state != nullptr && k + 1 == cut.chunk) {
                        copy_mtp(*chunks[k].sequence, *cut.state, true);
                    }
                }
            }
            if (cut.state != nullptr) {
                const SequenceState& sequence = *chunks[cut.chunk].sequence;
                cut.state->position           = chunks[cut.chunk].first;
                cut.state->drafter_follows    = (mtp || drafter) && sequence.drafter_follows;
            }
        } catch (...) {
            rank.stack  = own_stack;
            rank.mixed  = own_mixed;
            rank.inject = own_inject;
            span_hold   = false;
            span_last   = false;
            pooled_end  = ~std::size_t{0};
            throw;
        }
        rank.stack = own_stack;
    }

    // Tokens one prompt call may take: a span with spans on, else a chunk.
    [[nodiscard]] std::uint32_t prompt_step() const noexcept {
        return span_tokens != 0 ? span_tokens : options.prefill_chunk;
    }

    void forward(std::uint32_t s, std::span<const std::int32_t> tokens, std::uint32_t logit_rows,
                 std::span<const std::int32_t> next, PromptCut cut = {}) {
        if (next.size() > prompt_step()) {
            throw std::invalid_argument("qwen4_exp forward: more next tokens than a prompt step");
        }
        if (cut.state != nullptr &&
            (span_tokens == 0 || cut.tokens == 0 || cut.tokens >= tokens.size())) {
            throw std::invalid_argument(
                "qwen4_exp forward: a cut needs prompt spans and tokens on both of its sides");
        }
        if (tokens.size() > options.prefill_chunk || cut.state != nullptr) {
            forward_span(s, tokens, logit_rows, next, cut);
            return;
        }
        check_tokens(tokens, logit_rows, true);
        SequenceState& sequence      = sequences.at(s);
        const std::uint32_t position = sequence.position;
        const Part part{
            .sequence = &sequence, .column = 0, .count = static_cast<std::int32_t>(tokens.size())};
        // DFlash2: the chunk's last window positions, which the ring keeps. A one-token pass
        // runs the decode graphs, which capture its column whether or not the drafter follows.
        const std::int32_t kept =
            drafter ? std::min(part.count, static_cast<std::int32_t>(drafter_window())) : 0;
        capture             = {.first = part.count - kept, .count = kept, .column = 0};
        prompt_pass         = true;
        kv_only_pass        = logit_rows == 0 && kv_only_allowed();
        pooled_end          = layers.size() - (kv_only_pass ? 1 : 0);
        const auto end_pass = [&] {
            capture      = {};
            prompt_pass  = false;
            kv_only_pass = false;
            pooled_end   = ~std::size_t{0};
        };
        try {
            pass(std::span(&part, 1), tokens, logit_rows);
        } catch (...) {
            end_pass();
            throw;
        }
        end_pass();
        hint_next(sequence, next);
        if (kept > 0 && sequence.drafter_follows) {
            const ContextRun run{&sequence, 0, kept,
                                 position + std::uint32_t(part.count - kept)};
            dflash2_context(std::span(&run, 1), nullptr, 0);
        }
        if (mtp) {
            const Committed committed{&sequence, 0, part.count, position, tokens};
            mtp_catch_up(std::span(&committed, 1));
        }
    }

    // Whether forward_branches() runs: every sparse-attention layer keeps its KV on the device.
    [[nodiscard]] bool runs_branches() const noexcept { return stream_slots == 0; }

    void forward_branches(std::uint32_t s, const SequenceSnapshot& at,
                          std::span<const std::int32_t> tokens,
                          std::span<const std::uint32_t> lengths) {
        if (!runs_branches()) {
            throw std::logic_error("qwen4_exp forward_branches: a streamed KV layer");
        }
        check_tokens(tokens, static_cast<std::uint32_t>(lengths.size()));
        std::vector<Part> parts;
        parts.reserve(lengths.size());
        std::uint32_t column = 0;
        for (const std::uint32_t length : lengths) {
            if (length == 0) {
                throw std::invalid_argument("qwen4_exp forward_branches: an empty branch");
            }
            parts.push_back({.sequence = &sequences.at(s),
                             .column   = static_cast<std::int32_t>(column),
                             .count    = static_cast<std::int32_t>(length)});
            column += length;
        }
        if (column != tokens.size()) {
            throw std::invalid_argument("qwen4_exp forward_branches: lengths unlike the tokens");
        }
        restore(s, at);
        branch_origin = &at;
        try {
            pass(parts, tokens, static_cast<std::uint32_t>(parts.size()));
        } catch (...) {
            branch_origin = nullptr;
            throw;
        }
        branch_origin = nullptr;
        // The pass advanced the position by every branch; the state is the last branch's, which
        // the MTP block did not follow.
        sequences.at(s).position    = at.position + lengths.back();
        sequences.at(s).drafter_follows = false;
    }

    void forward_prompt_branches(std::uint32_t s, std::uint32_t prompt, SequenceSnapshot& at,
                                 std::span<const std::int32_t> tokens,
                                 std::span<const std::uint32_t> lengths) {
        if (!runs_branches()) {
            throw std::logic_error("qwen4_exp forward_prompt_branches: a streamed KV layer");
        }
        check_tokens(tokens, static_cast<std::uint32_t>(lengths.size()));
        SequenceState& sequence = sequences.at(s);
        if (prompt == 0 || lengths.empty()) {
            throw std::invalid_argument("qwen4_exp forward_prompt_branches: no prompt or branch");
        }
        std::vector<Part> parts;
        parts.reserve(lengths.size() + 1);
        parts.push_back(
            {.sequence = &sequence, .column = 0, .count = static_cast<std::int32_t>(prompt)});
        std::uint32_t column = prompt;
        for (const std::uint32_t length : lengths) {
            if (length == 0) {
                throw std::invalid_argument("qwen4_exp forward_prompt_branches: an empty branch");
            }
            parts.push_back({.sequence = &sequence,
                             .column   = static_cast<std::int32_t>(column),
                             .count    = static_cast<std::int32_t>(length),
                             .offset   = prompt});
            column += length;
        }
        if (column != tokens.size()) {
            throw std::invalid_argument(
                "qwen4_exp forward_prompt_branches: lengths unlike the tokens");
        }
        size_snapshot(sequence, at);
        const std::uint32_t position = sequence.position;
        at.context                   = sequence.context;
        branch_origin                = &at;
        branch_capture               = &at;
        try {
            pass(parts, tokens, static_cast<std::uint32_t>(lengths.size()));
        } catch (...) {
            branch_origin  = nullptr;
            branch_capture = nullptr;
            throw;
        }
        branch_origin  = nullptr;
        branch_capture = nullptr;
        // The MTP block followed none of the pass's tokens; its state goes along as snapshot()
        // keeps it.
        if (mtp) {
            RankBinding bind(device, model.head_rank());
            const cudaStream_t stream = ranks[model.head_rank()].stream;
            CUDA_CHECK(cudaMemcpyAsync(at.mtp[0].p, sequence.mtp.tail.p, at.mtp[0].bytes,
                                       cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(at.mtp[1].p, sequence.mtp_pending.p, at.mtp[1].bytes,
                                       cudaMemcpyDeviceToDevice, stream));
        }
        synchronize_ranks();
        at.position          = position + prompt;
        at.drafter_follows       = false;
        sequence.position    = at.position + lengths.back();
        sequence.drafter_follows = false;
    }

    // Each part after a branch pass's first starts a layer from the origin's state of it: its
    // mixers' (Gated DeltaNet, convolution, indexer tail) before them, its PLE history before the
    // PLE (recurrent_regions order).
    static constexpr std::array<bool, 4> kMixerRegions{true, true, true, false};
    static constexpr std::array<bool, 4> kPleRegions{false, false, false, true};

    // The prompt part's state of a layer on a rank (snapshot layer `slot`, state_slot()), into the
    // snapshot the pass's branches start from.
    void capture_branch_layer(const LayerState& state, std::size_t slot, RankState& rank,
                              const std::array<bool, 4>& which) {
        if (branch_capture == nullptr) { return; }
        const auto regions = recurrent_regions(state);
        auto& kept         = branch_capture->layers.at(slot).buffers;
        for (std::size_t k = 0; k < regions.size(); ++k) {
            if (!which[k] || regions[k].first == nullptr || kept[k].bytes == 0) { continue; }
            CUDA_CHECK(cudaMemcpyAsync(kept[k].p, regions[k].first, kept[k].bytes,
                                       cudaMemcpyDeviceToDevice, rank.stream));
        }
    }

    void restore_branch_layer(const LayerState& state, std::size_t slot, RankState& rank,
                              const std::array<bool, 4>& which) {
        if (branch_origin == nullptr) { return; }
        const auto regions = recurrent_regions(state);
        const auto& kept   = branch_origin->layers.at(slot).buffers;
        for (std::size_t k = 0; k < regions.size(); ++k) {
            if (!which[k] || regions[k].first == nullptr || kept[k].bytes == 0) { continue; }
            CUDA_CHECK(cudaMemcpyAsync(regions[k].first, kept[k].p, kept[k].bytes,
                                       cudaMemcpyDeviceToDevice, rank.stream));
        }
    }

    void decode(std::span<const std::uint32_t> batch, std::span<const std::int32_t> tokens) {
        if (batch.empty() || batch.size() != tokens.size()) {
            throw std::invalid_argument("qwen4_exp decode: one token per sequence");
        }
        check_tokens(tokens, static_cast<std::uint32_t>(tokens.size()));
        std::vector<Part> parts;
        parts.reserve(batch.size());
        for (std::size_t j = 0; j < batch.size(); ++j) {
            SequenceState* sequence = &sequences.at(batch[j]);
            for (const Part& other : parts) {
                if (other.sequence == sequence) {
                    throw std::invalid_argument("qwen4_exp decode: a sequence appears twice");
                }
            }
            parts.push_back({.sequence = sequence, .column = std::int32_t(j), .count = 1});
        }
        std::vector<Committed> committed;
        std::vector<ContextRun> runs;
        for (const Part& part : parts) {
            committed.push_back({part.sequence, part.column, 1, part.sequence->position,
                                 tokens.subspan(std::size_t(part.column), 1)});
            runs.push_back({part.sequence, part.column,
                            part.sequence->drafter_follows ? 1 : 0, part.sequence->position});
        }
        // The decode graphs capture every column's DFlash2 features.
        capture = {.first = 0, .count = drafter ? static_cast<std::int32_t>(parts.size()) : 0};
        try {
            pass(parts, tokens, static_cast<std::uint32_t>(tokens.size()));
        } catch (...) {
            capture = {};
            throw;
        }
        capture = {};
        if (drafter) { dflash2_context(runs, features.p, 1); }
        if (mtp) { mtp_catch_up(committed); }
    }

    void warm_up() {
        // Any token; the n-gram rows it addresses are read as a request's would be. The expert
        // cache does not count the warm-up's routes, which would fill it with one token's experts.
        constexpr std::int32_t kToken = 13;
        const bool device_experts     = model.options().experts == ExpertResidency::Device;
        const std::uint32_t room      = options.max_context > 1 ? options.max_context - 1 : 1;
        const std::uint32_t chunk = std::min(device_experts ? options.prefill_chunk : 16U, room);
        std::vector<std::int32_t> tokens(std::max<std::uint32_t>(chunk, 16), kToken);
        const auto prefill = [&](std::uint32_t count) {
            reset(0);
            count = std::min(count, room);
            forward(0, std::span(tokens).first(count), 1, {});
        };
        warming = true;
        prefill(1);
        if (verify_width > 1) { prefill(verify_width); }
        prefill(chunk);
        reset(0);
        prefill(2);
        // Every graph a single sequence's passes may run is captured now, so that requests take
        // no device memory for them: the decode step's, a verification's at each width up to
        // kVectorTokens, and the draft's chains by step count, or its steps under a probability
        // floor, or DFlash2's proposal and its context calls by run length. Each shape's first run
        // is eager (the decode step's was prefill(1)) and its second captures; the device memory
        // the second runs take is the graphs'. The first run also verifies the wider widths, which
        // stay eager, and commits a token.
        const std::uint32_t sequence = 0;
        const auto fits = [&](std::uint32_t s, std::uint32_t count) {
            return sequences.at(s).position + count <= options.max_context;
        };
        // Room for both runs: the commit and the decode step between them take a position each.
        const bool drafting =
            (mtp || drafter) && can_draft(sequence) && fits(sequence, verify_width + 2);
        const auto widest   = std::min<std::uint32_t>(verify_width, kVectorTokens);
        std::vector<std::int32_t> drafts(options.draft_tokens);
        std::vector<std::uint32_t> extents(1);
        const auto shapes = [&](bool first) {
            if (!drafting) { return; }
            for (std::uint32_t w = 1; w <= (first ? verify_width : widest); ++w) {
                verify(std::span(&sequence, 1), std::span(tokens).first(w));
            }
            if (first) {
                const std::uint32_t kept = 1;
                commit(std::span(&sequence, 1), std::span(&kept, 1));
            }
            if (drafter) {
                // One proposal runs every column at once. A commit's context call runs its run
                // length's graph; the verifications above left features in every column, and the
                // reset after the warm-up drops what these calls put in the ring.
                draft(std::span(&sequence, 1), std::span(tokens).first(1), drafts, extents);
                SequenceState& state = sequences.at(sequence);
                for (std::uint32_t count = 1; count <= verify_width; ++count) {
                    const ContextRun run{&state, 0, std::int32_t(count), state.position};
                    dflash2_context(std::span(&run, 1), nullptr, 0);
                }
            } else {
                // The MTP chains of greedy rows, then of sampling ones (their own graphs).
                ops::SamplingConfig sampler;
                sampler.temperature       = 1.0F;
                const std::int32_t length = std::int32_t(sequences.at(sequence).position);
                for (const bool sampling : {false, true}) {
                    const std::span<const ops::SamplingConfig> configs =
                        sampling ? std::span<const ops::SamplingConfig>(&sampler, 1)
                                 : std::span<const ops::SamplingConfig>{};
                    const std::span<const std::int32_t> lengths =
                        sampling ? std::span<const std::int32_t>(&length, 1)
                                 : std::span<const std::int32_t>{};
                    if (options.draft_min_p > 0) {
                        every_step = true;
                        draft(std::span(&sequence, 1), std::span(tokens).first(1), drafts, extents,
                              0, configs, lengths);
                        every_step = false;
                    } else {
                        for (std::uint32_t k = 1; k <= options.draft_tokens; ++k) {
                            draft(std::span(&sequence, 1), std::span(tokens).first(1), drafts, {},
                                  k, configs, lengths);
                        }
                    }
                }
            }
        };
        const auto free_bytes = [&] {
            std::vector<std::uint64_t> out(ranks.size());
            for (std::size_t r = 0; r < ranks.size(); ++r) {
                RankBinding bind(device, r);
                CUDA_CHECK(ninfer::wait_for_device());
                std::size_t free = 0, total = 0;
                CUDA_CHECK(cudaMemGetInfo(&free, &total));
                out[r] = free;
            }
            return out;
        };
        shapes(true);
        if (options.cuda_graphs && fits(sequence, 1)) {
            const auto before = free_bytes();
            decode(std::span(&sequence, 1), std::span(tokens).first(1));
            shapes(false);
            const auto after = free_bytes();
            for (std::size_t r = 0; r < ranks.size(); ++r) {
                const std::uint64_t taken = before[r] > after[r] ? before[r] - after[r] : 0;
                graph_bytes               = std::max(graph_bytes, taken);
            }
        }
        // Batches of several sequences run eagerly: their first passes load the kernels of their
        // widths now, a plain step of every batch size, then a verification of every sequence at
        // the full width with the drafts before it.
        std::vector<std::uint32_t> batch{sequence};
        for (std::uint32_t s = 1; s < options.sequences; ++s) {
            reset(s);
            forward(s, std::span(tokens).first(1), 1, {});
            batch.push_back(s);
            if (std::all_of(batch.begin(), batch.end(), [&](auto b) { return fits(b, 1); })) {
                decode(batch, std::vector<std::int32_t>(batch.size(), kToken));
            }
        }
        const bool round = batch.size() > 1 && (mtp || drafter) &&
                           std::all_of(batch.begin(), batch.end(), [&](std::uint32_t s) {
                               return can_draft(s) && fits(s, verify_width);
                           });
        if (round) {
            std::vector<std::int32_t> batch_drafts(batch.size() * options.draft_tokens);
            draft(batch, std::vector<std::int32_t>(batch.size(), kToken), batch_drafts);
            verify(batch, std::vector<std::int32_t>(batch.size() * verify_width, kToken));
            commit(batch, std::vector<std::uint32_t>(batch.size(), 1));
        }
        for (const std::uint32_t s : batch) { reset(s); }
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            CUDA_CHECK(ninfer::wait_for_stream(rank.stream));
        }
        warming = false;
    }

    void grow_expert_cache(std::uint64_t keep) {
        if (!cache && !stream && !hybrid) { return; }
        std::vector<std::uint64_t> bytes(device.size(), 0);
        for (std::size_t r = 0; r < device.size(); ++r) {
            RankBinding bind(device, r);
            CUDA_CHECK(ninfer::wait_for_device());
            std::size_t free_bytes = 0, total = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total));
            bytes[r] = free_bytes > keep ? free_bytes - keep : 0;
        }
        const auto added = cache    ? cache->grow(bytes)
                           : stream ? stream->grow(bytes)
                                    : hybrid->grow(bytes);
        for (std::size_t r = 0; r < added.size(); ++r) {
            memory.ranks[r].expert_cache_bytes += added[r];
            memory.expert_cache_bytes += added[r];
        }
        if (cache) {
            // The new slots take the next most routed experts of the profile, as construction's
            // did.
            if (cache_profile) { cache->seed(*cache_profile); }
            std::vector<ExpertCache::Loan> loans;
            for (std::size_t r = 0; r < slot_pools.size(); ++r) {
                for (const auto& buffer : slot_pools[r].buffers) {
                    loans.push_back(
                        {.rank = r, .base = buffer.storage.p, .bytes = buffer.storage.bytes});
                }
            }
            if (span_stack.p != nullptr) {
                loans.push_back({.rank = 0, .base = span_stack.p, .bytes = span_stack.bytes});
            }
            if (!loans.empty()) {
                lent_ranks.assign(device.size(), 0);
                for (const auto& loan : loans) { lent_ranks[loan.rank] = 1; }
                const auto lent = cache->lend(loans);
                for (std::size_t r = 0; r < lent.size(); ++r) {
                    memory.ranks[r].expert_cache_bytes += lent[r];
                    memory.ranks[r].workspace_bytes -= lent[r];
                    memory.expert_cache_bytes += lent[r];
                    memory.workspace_bytes -= lent[r];
                }
                pools_lent = true;
            }
        }
        // The kernels that split a call's routes read the new blocks as cached.
        if (misses) {
            for (std::size_t i = 0; i < expert_layers().size(); ++i) {
                misses->set_resident(i, resident_blocks(i));
            }
        }
    }

    // The device blocks that hold layer i's cached experts: the host cache's or the disk stream's.
    [[nodiscard]] ExpertCache::Ranges resident_blocks(std::size_t i) const {
        if (cache) { return cache->storage(i); }
        ExpertCache::Ranges out{};
        if (stream) {
            const auto blocks = stream->storage(i);
            std::copy(blocks.begin(), blocks.end(), out.begin());
        }
        return out;
    }

    // ---- MTP speculative decoding ------------------------------------------------------------

    void require_drafts(const char* what) const {
        if (!mtp && !drafter) {
            throw std::logic_error(std::string("qwen4_exp ") + what +
                                   ": the executor has no drafts");
        }
    }

    [[nodiscard]] bool can_draft(std::uint32_t s) const {
        const SequenceState& sequence = sequences.at(s);
        return (mtp || drafter) && sequence.drafter_follows && sequence.position > 0;
    }

    // ---- DFlash2 ------------------------------------------------------------------------------

    // The next pinned staging half, once the upload that last read it is done.
    std::int32_t* dflash2_stage() {
        const std::size_t half = std::exchange(dflash2_next, dflash2_next ^ 1U);
        CUDA_CHECK(ninfer::wait_for_event(dflash2_staged[half]));
        dflash2_next_half = half;
        return static_cast<std::int32_t*>(dflash2_staging[half]->data());
    }

    // Runs one sequence's DFlash2 call on the head rank's stream from its shape's graph: the first
    // run of a shape is eager, so every lazy initialization of its ops happens outside a capture,
    // and the second captures it. Its operands are the staged planes and the ring lanes they name,
    // so one graph serves every sequence.
    template <class Body> void dflash2_replay(PassGraphs& shape, const Body& body) {
        RankState& rank = ranks[model.head_rank()];
        if (!options.cuda_graphs || (shape.graphs.empty() && shape.runs++ == 0)) {
            body();
            return;
        }
        if (shape.graphs.empty()) {
            DecodeGraphDefinition definition;
            definition.capture(rank.stream, body);
            shape.graphs.emplace_back().instantiate(definition);
        }
        shape.graphs.front().launch(rank.stream);
    }

    // Uploads `bytes` of the half dflash2_stage() returned to `to` on the head rank's stream.
    void dflash2_upload(void* to, std::size_t bytes) {
        RankState& rank = ranks[model.head_rank()];
        CUDA_CHECK(cudaMemcpyAsync(to, dflash2_staging[dflash2_next_half]->data(), bytes,
                                   cudaMemcpyHostToDevice, rank.stream));
        CUDA_CHECK(cudaEventRecord(dflash2_staged[dflash2_next_half], rank.stream));
    }

    // Projects committed positions' features into their sequences' ring lanes. With `width`,
    // one context call reads the runs as rows of `width` columns from `base` (run j at column
    // j * width, its other columns inert); else one call per run reads its columns of `features`.
    void dflash2_context(std::span<const ContextRun> runs, const void* base, std::int32_t width) {
        if (runs.empty()) { return; }
        RankBinding bind(device, model.head_rank());
        RankState& rank        = ranks[model.head_rank()];
        const std::size_t r    = io_region(options.sequences) / 4;
        const std::size_t rows = std::size_t(drafter->taps()) * config.hidden_size;
        const std::size_t positions_words = io_region(feature_columns) / 4;
        std::int32_t* host     = dflash2_stage();
        std::int32_t* counts   = host + positions_words;
        std::int32_t* lanes    = counts + r;
        std::size_t at         = 0;
        std::uint32_t min_count = ~0U, max_count = 0;
        for (std::size_t j = 0; j < runs.size(); ++j) {
            const ContextRun& run = runs[j];
            const std::int32_t columns = width > 0 ? width : run.count;
            for (std::int32_t i = 0; i < columns; ++i) {
                host[at + std::size_t(i)] = static_cast<std::int32_t>(run.position) + i;
            }
            at += std::size_t(columns);
            counts[j] = run.count;
            lanes[j]  = static_cast<std::int32_t>(run.sequence->slot);
            min_count = std::min(min_count, std::uint32_t(run.count));
            max_count = std::max(max_count, std::uint32_t(run.count));
        }
        dflash2_upload(io.context_positions, (positions_words + 2 * r) * 4);
        if (width > 0) {
            if (max_count == 0) { return; }
            const auto b = static_cast<std::int32_t>(runs.size());
            const DFlash2ContextCall call{
                .features  = Tensor(const_cast<void*>(base), DType::BF16,
                                    {static_cast<std::int32_t>(rows), width, b}),
                .positions = Tensor(io.context_positions, DType::I32, {width, b}),
                .counts    = Tensor(io.context_counts, DType::I32, {b}),
                .lanes     = Tensor(io.context_lanes, DType::I32, {b}),
                .min_count = min_count,
                .max_count = max_count};
            drafter->materialize(call, *rank.workspace, rank.stream);
            return;
        }
        at = 0;
        for (std::size_t j = 0; j < runs.size(); ++j) {
            const ContextRun& run = runs[j];
            if (run.count > 0) {
                const DFlash2ContextCall call{
                    .features =
                        Tensor(static_cast<std::uint16_t*>(features.p) +
                                   std::size_t(run.column) * rows,
                               DType::BF16, {static_cast<std::int32_t>(rows), run.count, 1}),
                    .positions = Tensor(io.context_positions + at, DType::I32, {run.count, 1}),
                    .counts    = Tensor(io.context_counts + j, DType::I32, {1}),
                    .lanes     = Tensor(io.context_lanes + j, DType::I32, {1}),
                    .min_count = std::uint32_t(run.count),
                    .max_count = std::uint32_t(run.count)};
                const auto body = [&] { drafter->materialize(call, *rank.workspace, rank.stream); };
                // A lone run from the first feature column (a verification's or a decode step's
                // commit) replays its length's graph.
                if (runs.size() == 1 && run.column == 0 &&
                    std::size_t(run.count) < dflash2_contexts.size()) {
                    dflash2_replay(dflash2_contexts[std::size_t(run.count)], body);
                } else {
                    body();
                }
            }
            at += std::size_t(run.count);
        }
    }

    // One DFlash2 proposal for each of `drafting` (their anchors `anchors`): the block of the
    // anchor and draft_tokens masks against each ring lane, greedy; the first `k` drafts of each
    // row into `out` (rows `stride` long) and, with `extents`, each row's length up to its first
    // draft whose selector probability is at or below the floor.
    void draft_dflash2(std::span<SequenceState* const> drafting,
                       std::span<const std::int32_t> anchors, std::span<std::int32_t> out,
                       std::span<std::uint32_t> extents, std::size_t k) {
        RankBinding bind(device, model.head_rank());
        RankState& rank        = ranks[model.head_rank()];
        const std::size_t b    = drafting.size();
        const std::size_t kk   = options.draft_tokens;
        const auto w           = static_cast<std::int32_t>(verify_width);
        const std::size_t r    = io_region(options.sequences) / 4;
        std::int32_t* host     = dflash2_stage();
        for (std::size_t j = 0; j < b; ++j) {
            host[j]         = anchors[j];
            host[r + j]     = static_cast<std::int32_t>(drafting[j]->position);
            host[2 * r + j] = w;
            host[3 * r + j] = static_cast<std::int32_t>(drafting[j]->slot);
        }
        dflash2_upload(io.anchors, 4 * r * 4);
        const auto n  = static_cast<std::int32_t>(b);
        const auto ki = static_cast<std::int32_t>(kk);
        const DFlash2ProposalCall call{
            .anchors       = Tensor(io.anchors, DType::I32, {n}),
            .frontiers     = Tensor(io.frontiers, DType::I32, {n}),
            .valid_columns = Tensor(io.valid, DType::I32, {n}),
            .lanes         = Tensor(io.lanes, DType::I32, {n}),
            .configs       = static_cast<const ops::SamplingConfig*>(dflash2_configs.p),
            .batch         = n,
            .width         = w,
            .ids           = Tensor(io.ids, DType::I32, {w, n}),
            .positions     = Tensor(io.positions, DType::I32, {w, n}),
            .drafts        = Tensor(io.drafts, DType::I32, {ki, n}),
            .candidates    = Tensor(io.candidates, DType::I32, {16, ki, n}),
            .proposal_q    = Tensor(io.q, DType::FP32, {16, ki, n}),
            .confidence    = Tensor(io.confidence, DType::FP32, {ki, n})};
        const auto body = [&] { drafter->propose(call, *rank.workspace, rank.stream); };
        if (b == 1) {
            dflash2_replay(dflash2_proposals.at(std::size_t(w)), body);
        } else {
            body();
        }
        auto* drafts      = static_cast<std::int32_t*>(dflash2_readback->data());
        auto* confidences = reinterpret_cast<float*>(drafts + kk * b);
        CUDA_CHECK(cudaMemcpyAsync(drafts, io.drafts, kk * b * 4, cudaMemcpyDeviceToHost,
                                   rank.stream));
        CUDA_CHECK(cudaMemcpyAsync(confidences, io.confidence, kk * b * 4,
                                   cudaMemcpyDeviceToHost, rank.stream));
        // The expert cache takes the last verification's routes while the device drafts.
        settle_routes();
        CUDA_CHECK(ninfer::wait_for_stream(rank.stream));
        const std::size_t stride = options.draft_tokens;
        for (std::size_t j = 0; j < b; ++j) {
            std::uint32_t length = static_cast<std::uint32_t>(k);
            for (std::size_t i = 0; i < k; ++i) {
                out[j * stride + i] = drafts[j * kk + i];
                const float p       = confidences[j * kk + i];
                if (!std::isfinite(p)) {
                    throw std::runtime_error("qwen4_exp: non-finite masked draft confidence");
                }
                if (options.draft_min_p > 0 && length == k && p <= options.draft_min_p) {
                    length = static_cast<std::uint32_t>(i + 1);
                }
            }
            for (std::size_t i = k; i < stride; ++i) { out[j * stride + i] = -1; }
            if (!extents.empty()) { extents[j] = length; }
        }
        if (draft_prefetch) {
            // The verification's n-gram rows of the anchors and the drafts it can verify: up to
            // the longest row's floor.
            std::vector<NgramContext> contexts;
            for (const auto* sequence : drafting) { contexts.push_back(sequence->context); }
            const std::size_t hinted =
                extents.empty() ? k : *std::max_element(extents.begin(), extents.end());
            draft_prefetch->begin(std::move(contexts), anchors);
            draft_prefetch->hint(std::span<const std::int32_t>(out.data(), out.size()), stride,
                                 hinted);
        }
    }

    // Uploads an MTP pass's cells to the head rank's chain buffers: their token ids, positions
    // and 1-D RoPE positions, and for a catch-up over media their three-axis RoPE positions and
    // the cells that take a Vision embedding (empty otherwise).
    void stage_mtp(std::span<const std::int32_t> ids, std::span<const std::int32_t> positions,
                   std::span<const std::int32_t> rope, std::span<const std::int32_t> axes = {},
                   std::span<const std::int32_t> scatter = {}) {
        RankState& rank = ranks[model.head_rank()];
        CUDA_CHECK(ninfer::wait_for_event(mtp_staged));
        auto* host                 = static_cast<std::int32_t*>(mtp_staging->data());
        const std::size_t capacity = mtp_ids.bytes / 4;
        const auto upload = [&](const DeviceBuffer& target, std::size_t at,
                                std::span<const std::int32_t> values) {
            if (values.empty()) { return; }
            std::memcpy(host + at, values.data(), values.size() * 4);
            CUDA_CHECK(cudaMemcpyAsync(target.p, host + at, values.size() * 4,
                                       cudaMemcpyHostToDevice, rank.stream));
        };
        upload(mtp_ids, 0, ids);
        upload(mtp_positions, capacity, positions);
        upload(mtp_rope, 2 * capacity, rope);
        upload(mtp_axes, 3 * capacity, axes);
        upload(mtp_scatter, 6 * capacity, scatter);
        CUDA_CHECK(cudaEventRecord(mtp_staged, rank.stream));
    }

    // Puts `count` FP32 stacks into an MTP pass's input plane (BF16, plane a of the head rank) from
    // column `column` on.
    void stage_hidden(const void* source, std::int32_t count, std::int32_t column) {
        RankState& rank  = ranks[model.head_rank()];
        const auto width = static_cast<std::int32_t>(config.hc_count * config.hidden_size);
        const Tensor from(const_cast<void*>(source), DType::FP32, {width, count});
        Tensor to(columns(rank.a, width, column, 2), DType::BF16, {width, count});
        ops::cast_fp32_to_bf16(from, to, rank.stream);
    }

    // The token embeddings of `n` cells into `out` (BF16 [hidden, n] on the head rank) from their
    // ids at `ids` (device I32 [n], head rank); a catch-up's `media` runs take the prompt's Vision
    // embeddings into the cells mtp_scatter names. The table and the embeddings live on rank 0, so
    // a split model looks the rows up there and copies them over.
    void embed_cells(const std::int32_t* ids, std::int32_t n, void* out,
                     std::span<const MediaRun> media) {
        const std::size_t head = model.head_rank();
        const auto h           = static_cast<std::int32_t>(config.hidden_size);
        RankState& last        = ranks[head];
        const auto take_media  = [&](void* rows, std::int32_t* cells, cudaStream_t stream) {
            Tensor to(rows, DType::BF16, {h, n});
            for (const MediaRun& run : media) {
                const Tensor source(static_cast<std::byte*>(vision_embeddings.p) +
                                        std::size_t(run.source) * std::size_t(h) * 2,
                                    DType::BF16, {h, run.count});
                ops::scatter(source, Tensor(cells + run.offset, DType::I32, {run.count}), to,
                             stream);
            }
        };
        if (head == 0) {
            Tensor rows(out, DType::BF16, {h, n});
            ops::embedding(Tensor(const_cast<std::int32_t*>(ids), DType::I32, {n}), embedding_table,
                           rows, last.stream);
            take_media(out, static_cast<std::int32_t*>(mtp_scatter.p), last.stream);
            return;
        }
        RankState& first = ranks[0];
        CUDA_CHECK(cudaEventRecord(last.done, last.stream));
        {
            RankBinding bind(device, 0);
            CUDA_CHECK(cudaStreamWaitEvent(first.stream, last.done, 0));
            CUDA_CHECK(cudaMemcpyPeerAsync(first.ids, device.rank(0).device, ids,
                                           device.rank(head).device, std::size_t(n) * 4,
                                           first.stream));
            if (!media.empty()) {
                const auto cells = std::size_t(media.back().offset + media.back().count);
                CUDA_CHECK(cudaMemcpyPeerAsync(first.scatter, device.rank(0).device,
                                               mtp_scatter.p, device.rank(head).device, cells * 4,
                                               first.stream));
            }
            Tensor rows(first.mixed, DType::BF16, {h, n});
            ops::embedding(Tensor(first.ids, DType::I32, {n}), embedding_table, rows, first.stream);
            take_media(first.mixed, first.scatter, first.stream);
            CUDA_CHECK(cudaEventRecord(first.done, first.stream));
        }
        CUDA_CHECK(cudaStreamWaitEvent(last.stream, first.done, 0));
        CUDA_CHECK(cudaMemcpyPeerAsync(out, device.rank(head).device, first.mixed,
                                       device.rank(0).device, std::size_t(h) * n * 2, last.stream));
        // Catch-up returns with this copy queued. The next prompt chunk can start on rank 0
        // before the head consumes its embeddings, so protect the source until the copy ends.
        CUDA_CHECK(cudaEventRecord(last.done, last.stream));
        {
            RankBinding bind(device, 0);
            CUDA_CHECK(cudaStreamWaitEvent(first.stream, last.done, 0));
        }
    }

    // One MTP pass on the head rank over `n` cells whose starting stacks are in its input plane,
    // their token ids at `ids` and positions at `positions` (device, head rank): the token's
    // normalised embedding, projected, joins every stream of the normalised stack (one norm over
    // all of a cell's streams, then a projection stream by stream) with unit weight, and the
    // block's sparse-attention layer advances that stack, which stays in the head rank's stack
    // plane. Drafting cells run the indexer on the sequences' scratch tails. A catch-up (not
    // drafting) keeps only what the cells leave in the sequences' state, the layer's KV, pooled
    // keys and indexer tail, which its attention input determines; no draft reads the stack it
    // would leave, so it runs the router, whose routes the expert cache counts, but no expert.
    // A catch-up over a media prompt's images or video takes their Vision embeddings (`media`)
    // and rotates its cells at the three-axis positions in mtp_axes (`axes`); the cells' 1-D RoPE
    // positions are at `rope`.
    void mtp_pass(std::span<const MtpPart> parts, std::int32_t n, const std::int32_t* ids,
                  const std::int32_t* positions, const std::int32_t* rope, bool drafting,
                  std::span<const MediaRun> media = {}, bool axes = false) {
        RankState& rank      = ranks[model.head_rank()];
        const cudaStream_t s = rank.stream;
        WorkspaceArena& ws   = *rank.workspace;
        const auto h         = static_cast<std::int32_t>(config.hidden_size);
        const auto hc        = static_cast<std::int32_t>(config.hc_count);
        const float eps      = config.rms_norm_eps;
        embed_cells(ids, n, rank.mixed, media);
        Tensor embedded(rank.mixed, DType::BF16, {h, n});
        Tensor embedded_norm(rank.d, DType::BF16, {h, n});
        ops::rmsnorm(embedded, mtp->embedding_norm, eps, true, embedded_norm, s);
        Tensor token(rank.e, DType::BF16, {h, n});
        project(mtp->fc_embedding, embedded_norm, token, ws, s);
        Tensor hidden(rank.a, DType::BF16, {hc * h, n});
        Tensor hidden_norm(rank.b, DType::BF16, {hc * h, n});
        ops::rmsnorm(hidden, mtp->hidden_norm, eps, true, hidden_norm, s);
        Tensor streams(rank.c, DType::BF16, {h, hc * n});
        project(mtp->fc_hidden, Tensor(rank.b, DType::BF16, {h, hc * n}), streams, ws, s);
        Tensor stack(rank.stack, DType::FP32, {h, hc, n});
        ops::hyper_connection_expand(token, Tensor(rank.c, DType::BF16, {h, hc, n}), stack, s);

        const LayerPlan& plan = mtp->layer;
        Tensor mixed(rank.mixed, DType::BF16, {h, n});
        Tensor inject(rank.inject, DType::FP32, {hc, n});
        {
            auto scope = ws.scope();
            ops::hyper_connection_read(stack, plan.attn_hc.weights(), eps, ws, mixed, &inject, s);
        }
        // The target pass's rotation state, which the cells' replaces until their attention ends.
        const bool target_axes          = std::exchange(mrope, axes);
        std::int32_t* const target_rope = rank.mrope;
        if (axes) { rank.mrope = static_cast<std::int32_t*>(mtp_axes.p); }
        try {
            std::vector<Part> cells;
            for (const MtpPart& part : parts) {
                cells.push_back(
                    {.sequence = part.sequence, .column = part.column, .count = part.count});
            }
            qsa_project(*plan.qsa, rank, cells, n, nullptr, rope);
            for (std::size_t k = 0; k < parts.size(); ++k) {
                const MtpPart& part     = parts[k];
                SequenceState& sequence = *part.sequence;
                qsa_part(*plan.qsa, sequence.mtp, rank, cells[k], n,
                         {.positions = positions + part.column,
                          .rope      = rope + part.column,
                          .index     = nullptr,
                          .tail      = drafting ? sequence.mtp_scratch_tail.p : sequence.mtp.tail.p,
                          .first     = part.first});
            }
            qsa_output(*plan.qsa, rank, cells, n, nullptr);
        } catch (...) {
            mrope      = target_axes;
            rank.mrope = target_rope;
            throw;
        }
        mrope      = target_axes;
        rank.mrope = target_rope;
        {
            auto scope = ws.scope();
            ops::hyper_connection_write_read(stack, Tensor(rank.y, DType::BF16, {h, n}), inject,
                                             plan.mlp_hc.weights(), eps, ws, mixed, &inject, s);
        }
        if (drafting) {
            run_moe(plan, rank, n, layers.size());
            ops::hyper_connection_write(stack, Tensor(rank.y, DType::FP32, {h, n}), inject, s);
        } else {
            route(plan.moe, rank, n, layers.size());
        }
        record_mtp_routes(n);
    }

    // The MTP layer's routes for the expert cache, gathered over the passes between two target
    // passes (beyond a chunk of them the rest go uncounted).
    void record_mtp_routes(std::int32_t n) {
        if (!cache || warming || capturing_draft ||
            mtp_routes + std::uint32_t(n) > options.prefill_chunk) {
            return;
        }
        RankState& rank           = ranks[model.head_rank()];
        const std::uint64_t top   = config.num_experts_per_tok;
        const std::uint64_t pairs = top * options.prefill_chunk;
        auto* host = static_cast<std::int32_t*>(route_host->data()) + layers.size() * pairs +
                     top * mtp_routes;
        CUDA_CHECK(cudaMemcpyAsync(host, route_records[layers.size()], std::size_t(top) * n * 4,
                                   cudaMemcpyDeviceToHost, rank.stream));
        CUDA_CHECK(cudaEventRecord(rank.routes, rank.stream));
        mtp_routes += std::uint32_t(n);
    }

    // The draft of each of the `n` single-cell parts of the last MTP pass: the block's final mixer,
    // the head, and the argmax over the public tokens into `out` (device I32 [n], head rank), or
    // with `sampled_step` (>= 0) the draw of each row's sampler at the step's keys
    // (mtp_draft_configs, mtp_draft_keys).
    void mtp_head(std::int32_t n, std::int32_t* out, std::int32_t* second = nullptr,
                  float* logprobs = nullptr, std::int32_t sampled_step = -1) {
        RankState& rank      = ranks[model.head_rank()];
        const cudaStream_t s = rank.stream;
        WorkspaceArena& ws   = *rank.workspace;
        const auto h         = static_cast<std::int32_t>(config.hidden_size);
        const Tensor stack(rank.stack, DType::FP32,
                           {h, static_cast<std::int32_t>(config.hc_count), n});
        Tensor mixed(rank.mixed, DType::BF16, {h, n});
        {
            auto scope = ws.scope();
            ops::hyper_connection_read(stack, mtp->final_mixer.weights(), config.rms_norm_eps, ws,
                                       mixed, nullptr, s);
        }
        Tensor logits(rank.logits, DType::BF16, {static_cast<std::int32_t>(config.vocab_size), n});
        if (tensor()) {
            run_divided_mtp_head(mixed);
        } else {
            project(mtp->head, mixed, logits, ws, s);
        }
        Tensor tokens(out, DType::I32, {n});
        if (sampled_step >= 0) {
            auto scope = ws.scope();
            if (second) {
                // The prefetch reads the head's two best tokens beside the drawn one.
                Tensor best = ws.alloc(DType::I32, {n});
                Tensor alternative(second, DType::I32, {n});
                ops::argmax_top2(logits, best, alternative, domain, s);
            }
            const Tensor keys(static_cast<std::int32_t*>(mtp_draft_keys.p) +
                                  std::int64_t(sampled_step) * n,
                              DType::I32, {n});
            ops::sample(logits, tokens, domain,
                        static_cast<const ops::SamplingConfig*>(mtp_draft_configs.p), keys,
                        ops::kSamplePurposeDecode, ws, s);
        } else if (second) {
            Tensor alternative(second, DType::I32, {n});
            ops::argmax_top2(logits, tokens, alternative, domain, s);
        } else {
            ops::argmax(logits, tokens, domain, s);
        }
        if (logprobs) {
            Tensor probabilities(logprobs, DType::FP32, {n});
            ops::target_logprobs(logits, tokens, domain, probabilities, s);
        }
    }

    // Tensor parallelism: the head of the MTP block's drafts divided over the ranks. Rank 0 sends
    // its read of the block's stack (`mixed`, BF16 [hidden, n]) to every other rank, each rank
    // projects its rows of the head, and rank 0 places its rows and every other rank's, which come
    // over that rank's head link, into the logits, head_columns columns at a time. The other
    // ranks' streams fork from rank 0's and join it again, so a captured draft spans every rank.
    void run_divided_mtp_head(const Tensor& mixed) {
        const std::size_t n        = device.size();
        const auto h               = static_cast<std::int32_t>(config.hidden_size);
        const auto vocab           = static_cast<std::size_t>(config.vocab_size);
        const std::int32_t columns = mixed.ne[1];
        const auto& bounds         = model.weights().head_rows;
        const auto rows_of = [&](std::size_t r) { return static_cast<std::int32_t>(head_rows(r)); };
        const auto on      = [&](std::size_t r, const auto& f) {
            RankBinding bind(device, r);
            f(ranks[r]);
        };
        if (std::size_t(columns) > options.sequences) {
            throw std::logic_error("qwen4_exp: a divided draft head reads one column a sequence");
        }
        on(0, [&](RankState& rank) { CUDA_CHECK(cudaEventRecord(rank.fork, rank.stream)); });
        for (std::size_t r = 1; r < n; ++r) {
            on(0, [&](RankState& rank) {
                draft_links[r - 1].send(mixed.data, mixed.bytes(), 0, rank.stream);
            });
            on(r, [&](RankState& rank) {
                CUDA_CHECK(cudaStreamWaitEvent(rank.stream, ranks[0].fork, 0));
                draft_links[r - 1].recv(rank.mixed, mixed.bytes(), 0, rank.stream);
            });
        }
        RankState& first    = ranks[0];
        auto scope          = first.workspace->scope();
        std::int32_t widest = 0;
        for (std::size_t r = 0; r < n; ++r) { widest = std::max(widest, rows_of(r)); }
        const Tensor buffer = first.workspace->alloc(DType::BF16, {widest, head_columns});
        std::size_t slot    = 0;
        for (std::int32_t c = 0; c < columns; c += head_columns, slot ^= 1U) {
            const std::int32_t count = std::min(head_columns, columns - c);
            const auto input         = [&](std::size_t r) {
                auto* base = static_cast<std::byte*>(r == 0 ? mixed.data : ranks[r].mixed);
                return Tensor(base + std::size_t(c) * h * 2, DType::BF16, {h, count});
            };
            for (std::size_t r = 1; r < n; ++r) {
                on(r, [&](RankState& rank) {
                    Tensor part(rank.logits, DType::BF16, {rows_of(r), count});
                    project(mtp->rank_heads[r - 1], input(r), part, *rank.workspace, rank.stream);
                    head_links[r - 1].send(part.data, part.bytes(), slot, rank.stream);
                });
            }
            on(0, [&](RankState& rank) {
                const auto place = [&](std::size_t r) {
                    const std::size_t row_bytes = std::size_t(rows_of(r)) * 2;
                    CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::byte*>(rank.logits) +
                                                     (std::size_t(c) * vocab + bounds[r]) * 2,
                                                 vocab * 2, buffer.data, row_bytes, row_bytes,
                                                 std::size_t(count), cudaMemcpyDeviceToDevice,
                                                 rank.stream));
                };
                Tensor own(buffer.data, DType::BF16, {rows_of(0), count});
                project(mtp->head, input(0), own, *rank.workspace, rank.stream);
                place(0);
                for (std::size_t r = 1; r < n; ++r) {
                    head_links[r - 1].recv(buffer.data,
                                           std::size_t(rows_of(r)) * std::size_t(count) * 2, slot,
                                           rank.stream);
                    place(r);
                }
            });
        }
        for (std::size_t r = 1; r < n; ++r) {
            on(r, [&](RankState& rank) { CUDA_CHECK(cudaEventRecord(rank.join, rank.stream)); });
            on(0, [&](RankState& rank) {
                CUDA_CHECK(cudaStreamWaitEvent(rank.stream, ranks[r].join, 0));
            });
        }
    }

    // Advances the MTP block over what a target pass committed: for each sequence the cells from
    // its last position before the pass (whose stack it kept) to the second to last committed
    // position, each pairing that position's stack with the next token, and keeps the last
    // committed position's stack for the next cell. The target's stacks are still in the head
    // rank's stack plane.
    void mtp_catch_up(std::span<const Committed> committed) {
        RankBinding bind(device, model.head_rank());
        RankState& rank  = ranks[model.head_rank()];
        const auto width = std::size_t(config.hc_count) * config.hidden_size;
        std::vector<MtpPart> parts;
        std::vector<std::int32_t> ids, positions, rope, scatter;
        // A cell rotates at its position's RoPE position as the target does, on three axes in a
        // pass that holds media positions; the cell before a media token takes that token's
        // Vision embedding.
        std::vector<MediaRun> media;
        bool axes      = false;
        std::int32_t n = 0;
        for (const Committed& c : committed) {
            SequenceState& sequence = *c.sequence;
            if (!sequence.drafter_follows || c.count == 0) { continue; }
            // A sequence's first position has no stack before it.
            const bool kept           = c.position > 0;
            const std::int32_t cells  = c.count - (kept ? 0 : 1);
            const std::uint32_t first = kept ? c.position - 1 : 0;
            if (cells == 0) { continue; }
            if (kept) { stage_hidden(sequence.mtp_pending.p, 1, n); }
            if (c.count > 1) {
                stage_hidden(rank.stack + width * std::size_t(c.column), c.count - 1,
                             n + (kept ? 1 : 0));
            }
            parts.push_back({&sequence, n, cells, first});
            ids.insert(ids.end(), c.tokens.begin() + (kept ? 0 : 1), c.tokens.end());
            for (std::int32_t q = 0; q < cells; ++q) {
                const std::size_t at = first + std::size_t(q);
                positions.push_back(static_cast<std::int32_t>(at));
                rope.push_back(rope_at(sequence, at, 0));
                axes = axes || rope_at(sequence, at, 1) != rope.back() ||
                       rope_at(sequence, at, 2) != rope.back();
                const std::int32_t column = at + 1 < sequence.media_columns.size()
                                                ? sequence.media_columns[at + 1]
                                                : -1;
                if (column < 0) { continue; }
                if (!media.empty() && media.back().source + media.back().count == column) {
                    ++media.back().count;
                } else {
                    media.push_back({.source = column,
                                     .count  = 1,
                                     .offset = static_cast<std::int32_t>(scatter.size())});
                }
                scatter.push_back(n + q);
            }
            n += cells;
        }
        // The stacks the next cells start from, once the casts above have read the old ones.
        for (const Committed& c : committed) {
            if (!c.sequence->drafter_follows || c.count == 0) { continue; }
            CUDA_CHECK(cudaMemcpyAsync(c.sequence->mtp_pending.p,
                                       rank.stack + width * std::size_t(c.column + c.count - 1),
                                       width * 4, cudaMemcpyDeviceToDevice, rank.stream));
        }
        if (n == 0) { return; }
        std::vector<std::int32_t> three;
        if (axes) {
            three.resize(3 * std::size_t(n));
            for (const MtpPart& part : parts) {
                for (std::int32_t q = 0; q < part.count; ++q) {
                    for (std::size_t axis = 0; axis < 3; ++axis) {
                        three[axis * std::size_t(n) + std::size_t(part.column + q)] =
                            rope_at(*part.sequence, part.first + std::size_t(q), axis);
                    }
                }
            }
        }
        stage_mtp(ids, positions, rope, three, scatter);
        mtp_pass(parts, n, static_cast<const std::int32_t*>(mtp_ids.p),
                 static_cast<const std::int32_t*>(mtp_positions.p),
                 static_cast<const std::int32_t*>(mtp_rope.p), false, media, axes);
    }

    std::uint32_t draft(std::span<const std::uint32_t> batch, std::span<const std::int32_t> anchors,
                        std::span<std::int32_t> out, std::span<std::uint32_t> extents = {},
                        std::uint32_t steps                         = 0,
                        std::span<const ops::SamplingConfig> configs = {},
                        std::span<const std::int32_t> verify_lengths = {}) {
        require_drafts("draft");
        const std::size_t b      = batch.size();
        const std::size_t stride = options.draft_tokens;
        const std::size_t k      = steps == 0 ? stride : steps;
        if (b == 0 || b > sequences.size() || anchors.size() != b || out.size() != b * stride ||
            (!extents.empty() && extents.size() != b) || k > stride ||
            (!configs.empty() && (configs.size() != b || verify_lengths.size() != b))) {
            throw std::invalid_argument("qwen4_exp draft: one anchor and its drafts per sequence");
        }
        std::vector<SequenceState*> drafting;
        for (std::size_t j = 0; j < b; ++j) {
            SequenceState* sequence = &sequences.at(batch[j]);
            if (std::find(drafting.begin(), drafting.end(), sequence) != drafting.end()) {
                throw std::invalid_argument("qwen4_exp draft: a sequence appears twice");
            }
            if (!can_draft(batch[j])) {
                throw std::invalid_argument("qwen4_exp draft: the sequence's drafter does not "
                                            "follow its tokens");
            }
            if (sequence->position + verify_width > options.max_context) {
                throw std::invalid_argument("qwen4_exp draft: the round passes max_context");
            }
            if (anchors[j] < 0 || std::uint32_t(anchors[j]) >= config.vocab_size) {
                throw std::invalid_argument("qwen4_exp draft: anchor outside the vocabulary");
            }
            drafting.push_back(sequence);
        }
        if (drafter) {
            draft_dflash2(drafting, anchors, out, extents, k);
            return static_cast<std::uint32_t>(k);
        }
        RankBinding bind(device, model.head_rank());
        RankState& rank = ranks[model.head_rank()];
        // A chain whose rows sample draws each row's drafts with its sampler at the keys of the
        // verification columns they propose for; an all-greedy chain keeps the argmax.
        const bool sampled = options.sampled_drafts &&
                             std::any_of(configs.begin(), configs.end(),
                                         [](const ops::SamplingConfig& c) {
                                             return c.temperature > 0.0F;
                                         });
        if (sampled) {
            auto* host_configs = static_cast<ops::SamplingConfig*>(mtp_draft_staging->data());
            auto* host_keys    = reinterpret_cast<std::int32_t*>(
                static_cast<std::byte*>(mtp_draft_staging->data()) +
                std::size_t(options.sequences) * sizeof(ops::SamplingConfig));
            for (std::size_t j = 0; j < b; ++j) {
                ops::SamplingConfig c = configs[j];
                c.presence_penalty    = 0.0F;
                c.frequency_penalty   = 0.0F;
                c.token_counts        = nullptr;
                c.token_mask          = nullptr;
                c.token_mask_stride   = 0;
                c.non_finite_token    = 0;
                host_configs[j]       = c;
                for (std::size_t i = 0; i < k; ++i) {
                    host_keys[i * b + j] = verify_lengths[j] + static_cast<std::int32_t>(i) + 1;
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(mtp_draft_configs.p, host_configs,
                                       b * sizeof(ops::SamplingConfig), cudaMemcpyHostToDevice,
                                       rank.stream));
            CUDA_CHECK(cudaMemcpyAsync(mtp_draft_keys.p, host_keys, k * b * 4,
                                       cudaMemcpyHostToDevice, rank.stream));
        }
        // Step i's cell of a sequence sits at its position - 1 + i and pairs the stack before it
        // (the kept one at step 0) with the step's token (the anchor at step 0); it rotates at
        // that position's RoPE position, on one axis past a media prompt's last media token.
        std::vector<std::int32_t> positions(k * b), rope(k * b);
        std::vector<MtpPart> parts;
        for (std::size_t j = 0; j < b; ++j) {
            const std::uint32_t first = drafting[j]->position - 1;
            for (std::size_t i = 0; i < k; ++i) {
                positions[i * b + j] = static_cast<std::int32_t>(first + i);
                rope[i * b + j]      = rope_at(*drafting[j], first + i, 0);
            }
            parts.push_back({drafting[j], std::int32_t(j), 1, first});
        }
        if (hybrid) { hybrid->begin(cancelled, !warming); }
        stage_mtp(anchors, positions, rope);
        if (draft_prefetch) {
            std::vector<NgramContext> contexts;
            for (const auto* sequence : drafting) { contexts.push_back(sequence->context); }
            draft_prefetch->begin(std::move(contexts), anchors);
        }
        auto* ids        = static_cast<std::int32_t*>(mtp_ids.p);
        const auto* at   = static_cast<const std::int32_t*>(mtp_positions.p);
        const auto* turn = static_cast<const std::int32_t*>(mtp_rope.p);
        const auto n     = static_cast<std::int32_t>(b);
        const auto second = [&](std::size_t i) {
            return draft_prefetch ? static_cast<std::int32_t*>(ngram_prefetch_tokens.p) + i * b
                                  : nullptr;
        };
        auto* confidence = static_cast<float*>(mtp_logprobs.p);
        // Step i: the MTP pass over the step's cells (the anchors at step 0, with the stacks the
        // sequences kept) and the head's pick for the next one.
        const auto step = [&](std::size_t i) {
            if (i == 0) {
                for (std::size_t j = 0; j < b; ++j) {
                    CUDA_CHECK(cudaMemcpyAsync(drafting[j]->mtp_scratch_tail.p,
                                               drafting[j]->mtp.tail.p, drafting[j]->mtp.tail.bytes,
                                               cudaMemcpyDeviceToDevice, rank.stream));
                    stage_hidden(drafting[j]->mtp_pending.p, 1, std::int32_t(j));
                }
            } else {
                stage_hidden(rank.stack, n, 0);
            }
            mtp_pass(parts, n, ids + i * b, at + i * b, turn + i * b, true);
            mtp_head(n, ids + (i + 1) * b, second(i), confidence ? confidence + i * b : nullptr,
                     sampled ? static_cast<std::int32_t>(i) : -1);
        };
        const auto prefetch = [&](std::size_t i) {
            if (draft_prefetch) {
                draft_prefetch->enqueue(i, ids + (i + 1) * b, second(i), rank.stream);
            }
        };
        // Tokens and positions stay in the staged buffers. Native hybrid experts use the same
        // captured exchange as text decode; GGUF host experts replay too, without counting the
        // draft steps' routes. A chain or step of one sequence runs from its shape's graph
        // (PassGraphs). Under tensor parallelism the MTP block runs on rank 0 alone.
        const bool graph  = options.cuda_graphs && b == 1 &&
                            (device.size() == 1 || tensor()) && !stream;
        const auto replay = [&](PassGraphs& shape, const auto& body) {
            const bool run = graph && (!shape.graphs.empty() || shape.runs++ > 0);
            if (run && (shape.graphs.empty() || shape.sequence != drafting.front())) {
                DecodeGraphDefinition definition;
                capturing_draft = true;
                try {
                    definition.capture(rank.stream, body);
                } catch (...) {
                    capturing_draft = false;
                    throw;
                }
                capturing_draft = false;
                if (shape.graphs.empty()) {
                    shape.graphs.emplace_back().instantiate(definition);
                } else {
                    shape.graphs.front().update(definition);
                }
                shape.sequence = drafting.front();
            }
            if (run) {
                shape.graphs.front().launch(rank.stream);
            } else {
                body();
            }
        };
        if (misses) { misses->keep_alive(); }
        auto* host     = static_cast<std::int32_t*>(mtp_staging->data());
        auto* logprobs = reinterpret_cast<float*>(host + mtp_ids.bytes / 4);
        std::vector<std::uint32_t> lengths(b, static_cast<std::uint32_t>(k));
        std::size_t ran = k;
        if (confidence == nullptr || extents.empty()) {
            replay((sampled ? sampled_draft_graphs : draft_graphs).at(k), [&] {
                for (std::size_t i = 0; i < k; ++i) {
                    step(i);
                    prefetch(i);
                }
                if (draft_prefetch) { draft_prefetch->join(rank.stream); }
            });
            // The expert cache takes the last verification's routes while the device drafts; the
            // table uploads it queues follow the draft on the stream, before the next pass.
            settle_routes();
        } else {
            // The chain ends at the first step after which every sequence has drafted at or below
            // the floor: each step runs from its own graph, and the host reads the step's
            // confidences before it starts the next.
            const float floor = std::log(options.draft_min_p);
            std::vector<char> open(b, 1);
            for (std::size_t i = 0; i < k; ++i) {
                replay((sampled ? sampled_step_graphs : step_graphs).at(i), [&] { step(i); });
                prefetch(i);
                CUDA_CHECK(cudaMemcpyAsync(logprobs + i * b, confidence + i * b,
                                           b * sizeof(float), cudaMemcpyDeviceToHost, rank.stream));
                // The expert cache takes the last verification's routes while the first step
                // runs; the MTP block's table uploads it queues precede the next step.
                if (i == 0) { settle_routes(); }
                CUDA_CHECK(ninfer::wait_for_stream(rank.stream));
                bool more = false;
                for (std::size_t j = 0; j < b; ++j) {
                    if (!std::isfinite(logprobs[i * b + j])) {
                        throw std::runtime_error("qwen4_exp: non-finite MTP draft confidence");
                    }
                    if (open[j] && logprobs[i * b + j] <= floor) {
                        open[j]    = 0;
                        lengths[j] = static_cast<std::uint32_t>(i + 1);
                    }
                    more = more || open[j];
                }
                if (!more && !every_step) {
                    ran = i + 1;
                    break;
                }
            }
            if (draft_prefetch) { draft_prefetch->join(rank.stream); }
        }
        CUDA_CHECK(cudaMemcpyAsync(host, ids + b, ran * b * 4, cudaMemcpyDeviceToHost, rank.stream));
        CUDA_CHECK(ninfer::wait_for_stream(rank.stream));
        if (hybrid) { hybrid->finish(); }
        if (draft_prefetch) { draft_prefetch->check(); }
        for (std::size_t j = 0; j < b; ++j) {
            // Past the steps run no draft exists; no verification reaches those columns.
            for (std::size_t i = 0; i < k; ++i) {
                out[j * stride + i] = i < ran ? host[i * b + j] : -1;
            }
            if (!extents.empty()) { extents[j] = lengths[j]; }
        }
        return static_cast<std::uint32_t>(ran);
    }

    void verify(std::span<const std::uint32_t> batch, std::span<const std::int32_t> tokens,
                std::span<const std::uint32_t> widths = {}) {
        require_drafts("verify");
        const std::size_t b = batch.size();
        std::vector<std::uint32_t> row_widths(widths.begin(), widths.end());
        if (row_widths.empty() && b != 0) {
            row_widths.assign(b, static_cast<std::uint32_t>(tokens.size() / b));
        }
        std::size_t total = 0;
        std::uint32_t w   = 0;
        for (const std::uint32_t width : row_widths) {
            total += width;
            w = std::max(w, width);
        }
        if (b == 0 || b > sequences.size() || row_widths.size() != b || w > verify_width ||
            tokens.size() != total ||
            std::find(row_widths.begin(), row_widths.end(), 0U) != row_widths.end()) {
            throw std::invalid_argument("qwen4_exp verify: an anchor and its drafts per sequence");
        }
        check_tokens(tokens, static_cast<std::uint32_t>(tokens.size()));
        active_verify_width = w;
        std::vector<Part> parts;
        verified.clear();
        verified_columns.clear();
        verified_widths.clear();
        std::int32_t column = 0;
        for (std::size_t j = 0; j < b; ++j) {
            SequenceState* sequence = &sequences.at(batch[j]);
            if (std::find(verified.begin(), verified.end(), sequence) != verified.end()) {
                verified.clear();
                throw std::invalid_argument("qwen4_exp verify: a sequence appears twice");
            }
            const auto width = static_cast<std::int32_t>(row_widths[j]);
            parts.push_back({.sequence = sequence, .column = column, .count = width});
            sequence->verify_context = sequence->context;
            sequence->verify_tokens.assign(tokens.begin() + column,
                                           tokens.begin() + column + width);
            verified.push_back(sequence);
            verified_columns.push_back(column);
            verified_widths.push_back(width);
            column += width;
        }
        verifying = true;
        // The verification graphs capture every column's DFlash2 features; commit() takes the
        // kept ones into the ring.
        capture = {.first = 0, .count = drafter ? static_cast<std::int32_t>(tokens.size()) : 0};
        try {
            pass(parts, tokens, static_cast<std::uint32_t>(tokens.size()));
            if (draft_prefetch) {
                RankBinding bind(device, model.head_rank());
                RankState& rank = ranks[model.head_rank()];
                std::vector<NgramContext> contexts;
                for (const auto* sequence : verified) { contexts.push_back(sequence->context); }
                draft_prefetch->begin_verification(std::move(contexts), tokens, verified_widths);
                const auto count = static_cast<std::int32_t>(tokens.size());
                auto* first = static_cast<std::int32_t*>(ngram_prefetch_tokens.p);
                auto* second = first + std::size_t(options.sequences) * w;
                Tensor logits(rank.logits, DType::BF16,
                              {static_cast<std::int32_t>(config.vocab_size), count});
                Tensor winners(first, DType::I32, {count}), alternatives(second, DType::I32, {count});
                ops::argmax_top2(logits, winners, alternatives, domain, rank.stream);
                draft_prefetch->enqueue_verification(first, second, rank.stream);
                draft_prefetch->join(rank.stream);
            }
        } catch (...) {
            capture   = {};
            verifying = false;
            verified.clear();
            throw;
        }
        capture   = {};
        verifying = false;
    }

    void commit(std::span<const std::uint32_t> batch, std::span<const std::uint32_t> kept) {
        require_drafts("commit");
        const std::uint32_t w = active_verify_width;
        if (batch.size() != verified.size() || kept.size() != batch.size()) {
            throw std::invalid_argument("qwen4_exp commit: the last verification's sequences");
        }
        for (std::size_t j = 0; j < batch.size(); ++j) {
            if (&sequences.at(batch[j]) != verified[j] || kept[j] == 0 ||
                kept[j] > std::uint32_t(verified_widths[j])) {
                throw std::invalid_argument("qwen4_exp commit: the last verification's sequences, "
                                            "1 to their verified width each");
            }
        }
        if (draft_prefetch) {
            RankBinding bind(device, model.head_rank());
            // Captured events encode graph dependencies; wait on the joined stream before
            // reading host callback state. Host synchronization of a captured event is invalid.
            CUDA_CHECK(ninfer::wait_for_stream(ranks[model.head_rank()].stream));
            draft_prefetch->check();
        }
        const auto di = static_cast<std::int32_t>(config.indexer_head_dim);
        const std::uint64_t index_rows =
            std::uint64_t(config.indexer_n_heads + 1) * config.indexer_head_dim;
        const std::uint64_t columns = std::uint64_t(verify_width) * options.sequences;
        const auto stack_width      = std::int32_t(config.hc_count * config.hidden_size);
        const auto advance = [&](std::size_t i, std::size_t r, const LayerPlan& plan,
                                 std::size_t j, LayerState& state) {
            RankBinding bind(device, r);
            RankState& rank = ranks[r];
            const auto m = static_cast<std::int32_t>(kept[j]);
            if (plan.qsa) {
                // The indexer's tail advances over the kept positions' recorded projections;
                // the verification's first position is still staged.
                const QsaPlan& a = *plan.qsa;
                const Tensor projection(static_cast<std::byte*>(rank.index_records.p) +
                                            (qsa_local[i] * columns + verified_columns[j]) *
                                                index_rows * 2,
                                        DType::BF16, {a.index.rows, m});
                const Tensor first(rank.positions + verified_columns[j], DType::I32, {1});
                Tensor pooled(state.pooled.p, DType::FP32,
                              {di, static_cast<std::int32_t>(pooled_slots)});
                Tensor tail(state.tail.p, DType::FP32,
                            {di, static_cast<std::int32_t>(config.indexer_compress_ratio - 1)});
                const ops::QsaIndexerWeights iw{&a.index_query_norm, &a.index_key_norm};
                ops::qsa_indexer_append(projection, first, iw, config.rms_norm_eps,
                                        options.rope_yarn, pooled, tail, rank.stream);
            }
            if (plan.ple) {
                // The PLE history advances over the kept positions' recorded input.
                Tensor history(
                    state.history.p, DType::FP32,
                    {stack_width, static_cast<std::int32_t>((config.ple_conv_kernel_size - 1) *
                                                            config.ngram.ngram_size)});
                const Tensor normalized(static_cast<float*>(rank.ple_records.p) +
                                            std::size_t(stack_width) * verified_columns[j],
                                        DType::FP32, {stack_width, m});
                ops::ple_history_advance(history, normalized, rank.stream);
            }
        };
        for (std::size_t i = 0; i < layers.size(); ++i) {
            for (std::size_t j = 0; j < verified.size(); ++j) {
                each_part(*verified[j], i,
                          [&](std::size_t r, const LayerPlan& plan, LayerState& state) {
                              advance(i, r, plan, j, state);
                          });
            }
        }
        // The Gated DeltaNet layers replay the kept transitions into each sequence's slot.
        std::vector<ops::GdnReplayFoldRow> rows;
        for (std::size_t j = 0; j < verified.size(); ++j) {
            const auto slot = static_cast<std::int32_t>(verified[j]->slot);
            rows.push_back({slot, slot, static_cast<std::int32_t>(kept[j])});
        }
        for (RankState& rank : ranks) {
            if (!rank.fold) { continue; }
            RankBinding bind(device, rank.rank);
            rank.fold->execute(rows, static_cast<std::int32_t>(w), rank.stream);
        }
        std::vector<Committed> committed;
        std::vector<std::uint64_t> scratch;
        for (std::size_t j = 0; j < verified.size(); ++j) {
            SequenceState& sequence = *verified[j];
            const std::span<const std::int32_t> tokens(sequence.verify_tokens.data(), kept[j]);
            if (table) {
                scratch.resize(tokens.size() * config.ngram_heads());
                ngram_row_ids(ngram, tokens, config.eos_token_id, config.vocab_size,
                              sequence.context, scratch);
            }
            committed.push_back({&sequence, verified_columns[j],
                                 static_cast<std::int32_t>(kept[j]), sequence.position, tokens});
            sequence.position += kept[j];
        }
        verified.clear();
        if (drafter) {
            std::vector<ContextRun> runs;
            for (const Committed& c : committed) {
                runs.push_back({c.sequence, c.column, c.sequence->drafter_follows ? c.count : 0,
                                c.position});
            }
            if (runs.size() == 1) {
                dflash2_context(runs, nullptr, 0);
            } else {
                // Each sequence's kept columns as a row of the verification's width.
                RankBinding bind(device, model.head_rank());
                const cudaStream_t stream = ranks[model.head_rank()].stream;
                const std::size_t rows = std::size_t(drafter->taps()) * config.hidden_size;
                for (std::size_t j = 0; j < runs.size(); ++j) {
                    if (runs[j].count == 0) { continue; }
                    CUDA_CHECK(cudaMemcpyAsync(
                        static_cast<std::uint16_t*>(feature_rows.p) + j * w * rows,
                        static_cast<const std::uint16_t*>(features.p) +
                            std::size_t(runs[j].column) * rows,
                        std::size_t(runs[j].count) * rows * 2, cudaMemcpyDeviceToDevice, stream));
                }
                dflash2_context(runs, feature_rows.p, static_cast<std::int32_t>(w));
            }
        }
        if (mtp) { mtp_catch_up(committed); }
    }

    void snapshot(std::uint32_t s, SequenceSnapshot& out) {
        SequenceState& sequence = sequences.at(s);
        size_snapshot(sequence, out);
        copy_state(sequence, out, true);
        out.position        = sequence.position;
        out.context         = sequence.context;
        out.drafter_follows = (mtp || drafter) && sequence.drafter_follows;
    }

    // Gives `out` its buffers, sized as `sequence`'s regions, unless it has them.
    void size_snapshot(const SequenceState& sequence, SequenceSnapshot& out) {
        if (out.layers.size() != state_count()) {
            out.layers.clear();
            for (std::size_t i = 0; i < layers.size(); ++i) {
                each_part(sequence, i,
                          [&](std::size_t r, const LayerPlan&, const LayerState& state) {
                              RankBinding bind(device, r);
                              SequenceSnapshot::Layer layer;
                              for (const auto& [data, bytes] : recurrent_regions(state)) {
                                  layer.buffers.push_back(data != nullptr ? DeviceBuffer(bytes)
                                                                          : DeviceBuffer());
                              }
                              out.layers.push_back(std::move(layer));
                          });
            }
        }
        if (mtp && out.mtp.empty()) {
            RankBinding bind(device, model.head_rank());
            out.mtp.emplace_back(sequence.mtp.tail.bytes);
            out.mtp.emplace_back(sequence.mtp_pending.bytes);
        }
        if (drafter && out.ring.bytes == 0) {
            RankBinding bind(device, model.head_rank());
            const auto lane = ring->slot_view(0);
            out.ring = DeviceBuffer((lane.k_layer_bytes + lane.v_layer_bytes) * lane.layers);
        }
    }

    void restore(std::uint32_t s, const SequenceSnapshot& from) {
        if (from.layers.size() != state_count()) {
            throw std::invalid_argument("qwen4_exp restore: the snapshot holds no state");
        }
        SequenceState& sequence = sequences.at(s);
        copy_state(sequence, const_cast<SequenceSnapshot&>(from), false);
        sequence.position = from.position;
        sequence.context  = from.context;
        // A snapshot taken before the drafter followed the sequence leaves it behind.
        sequence.drafter_follows =
            ((mtp && !from.mtp.empty()) || (drafter && from.ring.bytes != 0)) &&
            from.drafter_follows;
        // A media prompt that resumes from the state sets its media again (set_media), which
        // skips the items the state already holds.
        sequence.media_columns.clear();
        sequence.media_rope.clear();
        sequence.rope_delta = 0;
    }

    // One region of a sequence image, on its rank: device memory, or a streamed layer's host copy.
    struct ImageRegion {
        void* device      = nullptr;
        std::size_t bytes = 0;
        std::size_t rank  = 0;
    };

    // A sequence image's regions at `position`, in the image's byte order: each layer's recurrent
    // state (taken from `at` when given), its KV pages and pooled keys of the positions before
    // `position`, then the MTP block's. Positions past `position` in its last page and pooled
    // block come along; the sequence rewrites them before it reads them.
    [[nodiscard]] std::vector<ImageRegion> image_regions(const SequenceState& sequence,
                                                         const SequenceSnapshot* at,
                                                         std::uint32_t position) const {
        std::vector<ImageRegion> out;
        const std::uint64_t used =
            (std::uint64_t(position) + kPagedKVPageSize - 1) / kPagedKVPageSize;
        const std::uint64_t blocks = position / config.indexer_compress_ratio;
        const auto attention       = [&](const LayerState& state, std::size_t rank) {
            for (const KvPlane& plane : state.kv) {
                if (plane.p == nullptr) { continue; }
                out.push_back({plane.p, std::size_t(plane.bytes / pages * used), rank});
            }
            out.push_back(
                {state.pooled.p, std::size_t(config.indexer_head_dim) * 4 * blocks, rank});
        };
        std::size_t kept = 0;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            each_part(sequence, i,
                      [&](std::size_t r, const LayerPlan& plan, const LayerState& state) {
                          const auto regions = recurrent_regions(state);
                          for (std::size_t k = 0; k < regions.size(); ++k) {
                              if (regions[k].first == nullptr) { continue; }
                              void* source =
                                  at != nullptr ? at->layers[kept].buffers[k].p : regions[k].first;
                              out.push_back({source, regions[k].second, r});
                          }
                          if (plan.qsa) { attention(state, r); }
                          ++kept;
                      });
        }
        if (mtp) {
            const std::size_t head = model.head_rank();
            attention(sequence.mtp, head);
            out.push_back({at != nullptr ? at->mtp[0].p : sequence.mtp.tail.p,
                           sequence.mtp.tail.bytes, head});
            out.push_back({at != nullptr ? at->mtp[1].p : sequence.mtp_pending.p,
                           sequence.mtp_pending.bytes, head});
        }
        if (drafter) {
            // The ring lane's slots of the positions before `position` (all of them past the
            // window), each layer's K then V, head by head while they are partly written.
            const std::size_t head = model.head_rank();
            const auto lane        = ring->slot_view(static_cast<std::int32_t>(sequence.slot));
            const std::uint32_t used = std::min(position, ring->capacity());
            const std::size_t row    = std::size_t(ring->head_dim()) * 2;
            const std::size_t pitch  = row * ring->padded_capacity();
            auto* kept               = at != nullptr ? static_cast<std::byte*>(at->ring.p) : nullptr;
            for (std::uint32_t l = 0; l < lane.layers; ++l) {
                const std::array<std::pair<std::byte*, std::size_t>, 2> planes{{
                    {static_cast<std::byte*>(lane.k_layer0.data) + l * lane.k_layer_pitch_bytes,
                     lane.k_layer_bytes},
                    {static_cast<std::byte*>(lane.v_layer0.data) + l * lane.v_layer_pitch_bytes,
                     lane.v_layer_bytes},
                }};
                for (const auto& [live, bytes] : planes) {
                    std::byte* base = kept != nullptr ? kept : live;
                    if (kept != nullptr) { kept += bytes; }
                    if (used == ring->capacity()) {
                        out.push_back({base, bytes, head});
                        continue;
                    }
                    for (std::int32_t h = 0; h < ring->num_kv_heads(); ++h) {
                        if (used == 0) { break; }
                        out.push_back({base + std::size_t(h) * pitch, used * row, head});
                    }
                }
            }
        }
        return out;
    }

    [[nodiscard]] std::uint64_t image_bytes(std::uint32_t position) const {
        std::uint64_t bytes = 0;
        for (const ImageRegion& region : image_regions(sequences.front(), nullptr, position)) {
            bytes += region.bytes;
        }
        return bytes;
    }

    // Copies between a sequence image's regions and host pages (to the host when `save`) on the
    // regions' streams, and waits for the copies. The image runs through the pages in order, each
    // holding `page_bytes` of it.
    void copy_image(const std::vector<ImageRegion>& regions, std::span<std::byte* const> pages,
                    std::size_t page_bytes, bool save) {
        std::uint64_t total = 0;
        for (const ImageRegion& region : regions) { total += region.bytes; }
        if (page_bytes == 0 || pages.size() < (total + page_bytes - 1) / page_bytes) {
            throw std::invalid_argument("qwen4_exp image: the pages do not hold the image");
        }
        std::uint64_t at = 0;
        for (const ImageRegion& region : regions) {
            RankBinding bind(device, region.rank);
            auto* device_bytes = static_cast<std::byte*>(region.device);
            for (std::size_t left = region.bytes; left > 0;) {
                const std::size_t in_page = std::min<std::size_t>(left, page_bytes - at % page_bytes);
                std::byte* host           = pages[at / page_bytes] + at % page_bytes;
                CUDA_CHECK(cudaMemcpyAsync(save ? static_cast<void*>(host) : device_bytes,
                                           save ? static_cast<void*>(device_bytes) : host, in_page,
                                           cudaMemcpyDefault, ranks[region.rank].stream));
                device_bytes += in_page;
                left -= in_page;
                at += in_page;
            }
        }
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            CUDA_CHECK(ninfer::wait_for_stream(rank.stream));
        }
    }

    SequenceImage save_image(std::uint32_t s, const SequenceSnapshot* at,
                             std::span<std::byte* const> pages, std::size_t page_bytes) {
        const SequenceState& sequence = sequences.at(s);
        if (at != nullptr && (at->layers.size() != state_count() || (mtp && at->mtp.size() != 2) ||
                              (drafter && at->ring.bytes == 0))) {
            throw std::invalid_argument("qwen4_exp image: the snapshot holds no state");
        }
        const std::uint32_t position = at != nullptr ? at->position : sequence.position;
        copy_image(image_regions(sequence, at, position), pages, page_bytes, true);
        return SequenceImage{.position = position,
                             .context  = at != nullptr ? at->context : sequence.context,
                             .drafter_follows = (mtp || drafter) &&
                                                (at != nullptr ? at->drafter_follows
                                                               : sequence.drafter_follows)};
    }

    void load_image(std::uint32_t s, const SequenceImage& image,
                    std::span<std::byte* const> pages, std::size_t page_bytes) {
        SequenceState& sequence = sequences.at(s);
        if (image.position > options.max_context) {
            throw std::invalid_argument("qwen4_exp image: the image does not fit the sequence");
        }
        copy_image(image_regions(sequence, nullptr, image.position), pages, page_bytes, false);
        forget_streamed_kv(sequence);
        sequence.position    = image.position;
        sequence.context     = image.context;
        sequence.drafter_follows = (mtp || drafter) && image.drafter_follows;
        sequence.media_columns.clear();
        sequence.media_rope.clear();
        sequence.rope_delta = 0;
    }

    // A layer's recurrent state into a snapshot's layer, on `stream`.
    static void keep_layer(const LayerState& state, SequenceSnapshot::Layer& kept,
                           cudaStream_t stream) {
        const auto regions = recurrent_regions(state);
        for (std::size_t k = 0; k < regions.size(); ++k) {
            if (regions[k].first == nullptr || kept.buffers[k].bytes == 0) { continue; }
            CUDA_CHECK(cudaMemcpyAsync(kept.buffers[k].p, regions[k].first, kept.buffers[k].bytes,
                                       cudaMemcpyDeviceToDevice, stream));
        }
    }

    // The MTP block's tail and kept stack between a sequence and a snapshot, on the head rank's
    // stream.
    void copy_mtp(SequenceState& sequence, SequenceSnapshot& snapshot, bool save) {
        if (!mtp || snapshot.mtp.size() != 2) { return; }
        RankBinding bind(device, model.head_rank());
        const cudaStream_t stream = ranks[model.head_rank()].stream;
        const std::array<std::pair<void*, DeviceBuffer*>, 2> pairs{
            {{sequence.mtp.tail.p, &snapshot.mtp[0]}, {sequence.mtp_pending.p, &snapshot.mtp[1]}}};
        for (const auto& [live, kept] : pairs) {
            if (live == nullptr || kept->bytes == 0) { continue; }
            CUDA_CHECK(cudaMemcpyAsync(save ? kept->p : live, save ? live : kept->p, kept->bytes,
                                       cudaMemcpyDeviceToDevice, stream));
        }
    }

    // The sequence's DFlash2 ring lane, whose slots the positions past the snapshot overwrite,
    // between the sequence and a snapshot, on the head rank's stream.
    void copy_ring(SequenceState& sequence, SequenceSnapshot& snapshot, bool save) {
        if (!drafter || snapshot.ring.bytes == 0) { return; }
        RankBinding bind(device, model.head_rank());
        const cudaStream_t stream = ranks[model.head_rank()].stream;
        const auto lane = ring->slot_view(static_cast<std::int32_t>(sequence.slot));
        auto* kept      = static_cast<std::byte*>(snapshot.ring.p);
        for (std::uint32_t l = 0; l < lane.layers; ++l) {
            const std::array<std::pair<std::byte*, std::size_t>, 2> planes{{
                {static_cast<std::byte*>(lane.k_layer0.data) + l * lane.k_layer_pitch_bytes,
                 lane.k_layer_bytes},
                {static_cast<std::byte*>(lane.v_layer0.data) + l * lane.v_layer_pitch_bytes,
                 lane.v_layer_bytes},
            }};
            for (const auto& [live, bytes] : planes) {
                CUDA_CHECK(cudaMemcpyAsync(save ? kept : live, save ? live : kept, bytes,
                                           cudaMemcpyDeviceToDevice, stream));
                kept += bytes;
            }
        }
    }

    // Copies the recurrent state between a sequence and a snapshot on the layers' streams, and
    // waits for the copies: the paged KV and the indexer's pooled keys stay where they are, since
    // a sequence only writes positions at or past its own. The MTP block's tail and kept stack
    // go with them.
    void copy_state(SequenceState& sequence, SequenceSnapshot& snapshot, bool save) {
        const auto copy = [&](void* live, DeviceBuffer& kept, cudaStream_t stream) {
            if (live == nullptr || kept.bytes == 0) { return; }
            CUDA_CHECK(cudaMemcpyAsync(save ? kept.p : live, save ? live : kept.p, kept.bytes,
                                       cudaMemcpyDeviceToDevice, stream));
        };
        std::size_t kept = 0;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            each_part(sequence, i, [&](std::size_t r, const LayerPlan&, LayerState& state) {
                RankBinding bind(device, r);
                const auto regions = recurrent_regions(state);
                for (std::size_t k = 0; k < regions.size(); ++k) {
                    copy(regions[k].first, snapshot.layers[kept].buffers[k], ranks[r].stream);
                }
                ++kept;
            });
        }
        copy_mtp(sequence, snapshot, save);
        copy_ring(sequence, snapshot, save);
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            CUDA_CHECK(ninfer::wait_for_stream(rank.stream));
        }
    }
};

Executor::Executor(const Model& model, DeviceContext& device, ExecutorOptions options)
    : impl_(std::make_unique<Impl>(model, device, std::move(options))) {}

Executor::~Executor() = default;

const ExecutorOptions& Executor::options() const noexcept { return impl_->options; }

ExecutorMemory Executor::memory() const noexcept { return impl_->memory; }

NgramTableStats Executor::ngram_stats() { return impl_->ngram_stats(); }

std::uint64_t Executor::ngram_resident_bytes() const noexcept {
    return impl_->table ? impl_->table->resident_bytes() : 0;
}

std::string Executor::tensor_parallel_routes() const {
    return impl_->tensor_exchange ? impl_->tensor_exchange->describe() : std::string();
}

void Executor::reset(std::uint32_t sequence) { impl_->reset(sequence); }

void Executor::warm_up() { impl_->warm_up(); }
std::uint32_t Executor::prompt_step() const noexcept { return impl_->prompt_step(); }
void Executor::grow_expert_cache(std::uint64_t keep_free_bytes) {
    impl_->grow_expert_cache(keep_free_bytes);
}

std::uint32_t Executor::position(std::uint32_t sequence) const {
    return impl_->sequences.at(sequence).position;
}

void Executor::forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                       std::uint32_t logit_rows, std::span<const std::int32_t> next,
                       PromptCut cut) {
    impl_->forward(sequence, tokens, logit_rows, next, cut);
}

bool Executor::packs_prompts() const noexcept { return impl_->span_tokens != 0; }
std::uint32_t Executor::narrow_prompt_tokens() const noexcept {
    return impl_->span_tokens == 0
               ? 0
               : std::max(std::uint32_t(kVectorTokens), impl_->prompt_miss_tokens);
}

void Executor::forward_pieces(std::span<const PromptPiece> pieces,
                              std::span<const std::int32_t> next) {
    impl_->forward_pieces(pieces, next);
}

void Executor::forward_branches(std::uint32_t sequence, const SequenceSnapshot& at,
                                std::span<const std::int32_t> tokens,
                                std::span<const std::uint32_t> lengths) {
    impl_->forward_branches(sequence, at, tokens, lengths);
}

void Executor::forward_prompt_branches(std::uint32_t sequence, std::uint32_t prompt,
                                       SequenceSnapshot& at, std::span<const std::int32_t> tokens,
                                       std::span<const std::uint32_t> lengths) {
    impl_->forward_prompt_branches(sequence, prompt, at, tokens, lengths);
}

bool Executor::runs_branches() const noexcept { return impl_->runs_branches(); }

std::uint32_t Executor::max_logit_rows() const noexcept { return impl_->max_logit_rows; }

void Executor::decode(std::span<const std::uint32_t> sequences,
                      std::span<const std::int32_t> tokens) {
    impl_->decode(sequences, tokens);
}

void Executor::snapshot(std::uint32_t sequence, SequenceSnapshot& out) {
    impl_->snapshot(sequence, out);
}

std::uint64_t Executor::graph_bytes() const { return impl_->graph_bytes; }

void Executor::size_snapshot(SequenceSnapshot& out) {
    impl_->size_snapshot(impl_->sequences.at(0), out);
}

void Executor::restore(std::uint32_t sequence, const SequenceSnapshot& from) {
    impl_->restore(sequence, from);
}

std::uint64_t Executor::image_bytes(std::uint32_t position) const {
    return impl_->image_bytes(position);
}

SequenceImage Executor::save_image(std::uint32_t sequence, const SequenceSnapshot* at,
                                   std::span<std::byte* const> pages, std::size_t page_bytes) {
    return impl_->save_image(sequence, at, pages, page_bytes);
}

void Executor::load_image(std::uint32_t sequence, const SequenceImage& image,
                          std::span<std::byte* const> pages, std::size_t page_bytes) {
    impl_->load_image(sequence, image, pages, page_bytes);
}

std::size_t Executor::set_media(std::uint32_t sequence, std::span<const MediaItem> items,
                                std::vector<std::int32_t> rope_positions,
                                std::int32_t rope_delta) {
    return impl_->set_media(sequence, items, std::move(rope_positions), rope_delta);
}

bool Executor::vision() const noexcept { return impl_->vision_context != nullptr; }

std::uint32_t Executor::draft_tokens() const noexcept {
    return impl_->mtp || impl_->drafter ? impl_->options.draft_tokens : 0;
}

bool Executor::can_draft(std::uint32_t sequence) const { return impl_->can_draft(sequence); }

void Executor::stop_drafting(std::uint32_t sequence) {
    impl_->sequences.at(sequence).drafter_follows = false;
}

std::uint32_t Executor::draft(std::span<const std::uint32_t> sequences,
                              std::span<const std::int32_t> anchors, std::span<std::int32_t> out,
                              std::span<std::uint32_t> extents, std::uint32_t steps,
                              std::span<const ops::SamplingConfig> configs,
                              std::span<const std::int32_t> lengths) {
    return impl_->draft(sequences, anchors, out, extents, steps, configs, lengths);
}

void Executor::verify(std::span<const std::uint32_t> sequences,
                      std::span<const std::int32_t> tokens, std::span<const std::uint32_t> widths) {
    impl_->verify(sequences, tokens, widths);
}

void Executor::commit(std::span<const std::uint32_t> sequences,
                      std::span<const std::uint32_t> columns) {
    impl_->commit(sequences, columns);
}

Tensor Executor::logits(std::uint32_t rows) const {
    return Tensor(
        impl_->ranks[impl_->model.head_rank()].logits, DType::BF16,
        {static_cast<std::int32_t>(impl_->config.vocab_size), static_cast<std::int32_t>(rows)});
}

std::size_t Executor::head_rank() const noexcept { return impl_->model.head_rank(); }

ExpertCacheStats Executor::expert_cache_stats() const noexcept {
    if (impl_->hybrid) { return impl_->hybrid->stats(); }
    if (impl_->stream) {
        const auto stream = impl_->stream->stats();
        return ExpertCacheStats{.routes       = stream.routes,
                                .hits         = stream.hits,
                                .admitted     = stream.routes - stream.hits,
                                .copied_bytes = stream.read_bytes,
                                .slots        = stream.slots};
    }
    ExpertCacheStats out = impl_->cache ? impl_->cache->stats() : ExpertCacheStats{};
    if (impl_->misses) {
        const auto staged = impl_->misses->stats();
        out.cpu_routes += staged.cpu_pairs;
        out.copied_bytes += staged.staged_bytes;
    }
    return out;
}

cudaStream_t Executor::head_stream() const noexcept {
    return impl_->ranks[impl_->model.head_rank()].stream;
}

std::string Executor::expert_execution_profile() const {
    // Native hybrid banks compute some routed pairs in other arithmetic; a GGUF CPU share computes
    // the GPU's bits (gguf_expert_cpu), so its states are the GPU's.
    if (impl_->hybrid) { return impl_->hybrid->execution_profile(); }
    return "gpu";
}

bool Executor::hybrid_experts() const noexcept { return impl_->hybrid != nullptr; }
std::string Executor::expert_cpu_description() const {
    return impl_->misses ? impl_->misses->cpu_description() : std::string();
}
std::uint32_t Executor::cpu_share_tokens() const noexcept {
    if (!impl_->misses || !impl_->misses->cpu_enabled()) { return 0; }
    // Disk experts' wide calls copy every miss (ExpertMisses::stage_wide).
    if (impl_->model.options().experts != ExpertResidency::Host) { return kNarrowMissTokens; }
    return std::max(kNarrowMissTokens, std::min(impl_->miss_tokens, kCpuMissTokens));
}
void Executor::save_expert_profile() const {
    const auto& path = impl_->options.hybrid_experts.record_profile;
    if (path.empty()) { return; }
    if (impl_->hybrid) {
        write_expert_profile(path, impl_->model.info().artifact_id, impl_->hybrid->route_counts());
    } else if (impl_->cache) {
        write_expert_profile(path, impl_->model.info().artifact_id, impl_->cache->totals());
    }
}
void Executor::bind_cancellation(const std::atomic<bool>* cancelled) noexcept {
    // Also drains partially enqueued work on exception paths before the request's borrowed flag
    // can die. Normal pass/draft boundaries have already drained and propagated any failure.
    if (!cancelled && impl_->hybrid) { try { impl_->hybrid->finish(); } catch (...) {} }
    impl_->cancelled = cancelled;
}

void Executor::raise_service_failure() {
    if (impl_->misses) { impl_->misses->raise_failure(); }
}

void Executor::abort(std::uint32_t sequence) {
    if (impl_->table) { impl_->table->wait(); }
    impl_->rows_waited = true;
    impl_->device.synchronize();
    impl_->pending_routes = 0;
    impl_->pending_decode = true;
    impl_->mtp_routes = 0;
    impl_->verifying = false;
    impl_->verified.clear();
    impl_->reset(sequence);
}

} // namespace ninfer::models::qwen4_exp
