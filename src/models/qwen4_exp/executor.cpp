#include "models/qwen4_exp/executor.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/gdn_replay_records.h"
#include "core/host_worker_pool.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "core/weight_view.h"
#include "ninfer/ops/target_logprobs.h"
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
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/sparse_attention.h"
#include "ninfer/ops/weight_input.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
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
// The family name of Flash-Next's built-in routing profiles (models/expert_profile.h).
constexpr std::string_view kExpertProfileFamily = "qwen3.8-flash-next";
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

Tensor direct(const Model& model, WeightId id, std::initializer_list<std::int32_t> shape,
              DType dtype = DType::BF16) {
    const auto& view = model.weight(id).view;
    Tensor tensor    = weight_tensor(view, shape);
    if (tensor.dtype != dtype) {
        throw std::invalid_argument(model.weight(id).name + " must be stored as " +
                                    (dtype == DType::BF16 ? "bf16" : "fp32"));
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

struct GdnPlan {
    Projection qkv, z, a, b, output;
    Tensor convolution, a_log, dt_bias, norm;
};

struct QsaPlan {
    Projection query, gate, key, value, index, output;
    Tensor query_norm, key_norm, index_query_norm, index_key_norm;
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

    [[nodiscard]] ops::GgufExpertTable view() const {
        return {format, static_cast<const void* const*>(table.p), row_bytes};
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

struct MoePlan {
    Tensor router, shared_gate;
    ExpertTable gate, up, down, shared_gate_table, shared_up_table, shared_down_table;
    std::optional<NativeMoePlan> native;
    // Device banks or disk experts' device slots, with zeros after the down banks; host experts
    // are read across the bus.
    bool device_resident = false;

    [[nodiscard]] ops::GgufMoeWeights banks() const {
        return {gate.view(),
                up.view(),
                down.view(),
                shared_gate_table.view(),
                shared_up_table.view(),
                shared_down_table.view(),
                static_cast<std::int32_t>(gate.pointers.size()),
                device_resident};
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
    // The MTP block (an executor with drafts), on the head rank: its layer's state, the target's
    // stack at the sequence's last position (which the next cell pairs with the next token), and
    // whether the block follows the sequence's tokens, which a media prompt breaks.
    LayerState mtp;
    DeviceBuffer mtp_pending, mtp_scratch_tail;
    bool mtp_follows = true;
    // The last verify(): the n-gram context before it and the tokens it ran.
    NgramContext verify_context;
    std::vector<std::int32_t> verify_tokens;
    // A prompt's media (Vision): for each prompt position the column of its merged embedding, or
    // -1, and the prompt's RoPE positions, axis-major [3, tokens]; positions past them rotate at
    // their index plus rope_delta.
    std::vector<std::int32_t> media_columns, media_rope;
    std::int32_t rope_delta = 0;
    // One decode graph per segment, captured at the sequence's second decode step: the first runs
    // eagerly, so every lazy initialization of the ops it reaches happens outside capture. The
    // graphs read the token, its position and n-gram rows from the ranks' staged planes and stay
    // valid for the sequence's lifetime, across resets.
    std::vector<DecodeGraphExecutable> decode;
    std::uint32_t decode_steps = 0;
    // The same for a verification of this sequence alone (a graph per segment) and for its draft
    // chain on one device with every expert there (one graph), captured at the second of each.
    struct VerifyGraphs {
        std::vector<DecodeGraphExecutable> segments;
        std::uint32_t runs = 0;
    };
    // The captured record strides and kernels depend on the actual verification width.
    std::array<VerifyGraphs, 17> verify;
    // Pipeline windows: verifications on the rank threads (every segment) and guessed windows (the
    // front stages' segments), per record set and width.
    std::array<std::array<VerifyGraphs, 17>, 2> pipelined_verify, pipelined_ahead;
    // Draft chains by step count (adaptive MTP drafts fewer than draft_tokens).
    struct DraftGraph {
        DecodeGraphExecutable chain;
        std::uint32_t runs = 0;
    };
    std::array<DraftGraph, 16> draft;
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
};

// What one device holds for the layers it runs: activation planes sized for a full chunk and a
// scoped workspace.
struct RankState {
    std::size_t rank    = 0;
    cudaStream_t stream = nullptr;
    DeviceBuffer activations;
    std::unique_ptr<DeviceArena> workspace;
    DeviceBuffer block_table;
    // A ring of host staging buffers, one per pass in flight (Impl::staging_slot picks the
    // pass's), each with the event that marks its uploads done: a pass rewrites a buffer only
    // once the pass that used it ring-size passes ago has uploaded it, so prompt chunks can fill
    // every stage of a pipeline instead of the host waiting for the previous pass on all ranks.
    std::vector<std::unique_ptr<PinnedHostBuffer>> staging;
    std::vector<cudaEvent_t> staged;
    cudaEvent_t done   = nullptr;
    cudaEvent_t copied = nullptr; // its stack has been copied in from the previous rank
    cudaEvent_t routes = nullptr; // the last pass's routes reached the host
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
    void* rows              = nullptr; // staged n-gram rows
    void* logits            = nullptr; // head rank only
    // The Gated DeltaNet state of the rank's layers, one slot per sequence (gdn_local maps a layer
    // to its pool layer).
    DeviceBuffer gdn_backing;
    std::unique_ptr<LinearAttentionStatePool> gdn;
    DeviceBuffer slot_ids; // I32 [sequences] = 0, 1, ...: a sequence's pool slot on the device
    // Speculative verification (an executor with drafts): the replay records of the rank's Gated
    // DeltaNet layers and their fold, each sparse-attention layer's indexer projections (BF16
    // [640, columns] per layer), the PLE layer's normalised convolution input (FP32
    // [10240, columns]), and scratch for the states a verification reads but must not advance.
    // Pipeline windows keep a second record set, so a guessed window can record while the window
    // before it still awaits its commit, and each set's positions for that commit.
    std::array<DeviceBuffer, 2> record_backing;
    std::array<std::optional<GdnReplayRecords>, 2> records;
    std::array<std::optional<ops::GdnReplayFoldPlan>, 2> fold;
    std::array<DeviceBuffer, 2> index_records, ple_records, window_positions;
    DeviceBuffer conv_scratch, tail_scratch;
    // Streamed sparse-attention layers: a prompt chunk reads one layer's whole KV from this
    // device copy (the four planes, every page), which holds `kv_staged`'s positions before
    // `kv_staged_end`; the layers of a span's attention phase keep it across the span's chunks.
    std::array<DeviceBuffer, 4> kv_staging;
    const LayerState* kv_staged = nullptr;
    std::uint32_t kv_staged_end = 0;
};

} // namespace

struct Executor::Impl {
    // The MTP block's operands on the head rank: its input norms and projections, its layer and
    // final mixer, and the head as its final mixer's output reads it.
    struct MtpPlan {
        Tensor embedding_norm, hidden_norm;
        Projection fc_embedding, fc_hidden, head;
        LayerPlan layer;
        HcPlan final_mixer;
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
    // The staging buffer of every rank the current pass uses (RankState::staging).
    std::size_t staging_slot = 0;
    std::array<StallEvents, 8> stall_events{};
    std::size_t stall_next = 0;
    std::uint64_t stalls   = 0;
    double stall_seconds   = 0.0;
    Weight embedding_table;
    Projection head;
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
    // Maximum and current verification widths (drafts plus anchor), and the public vocabulary.
    std::uint32_t verify_width = 0;
    std::uint32_t active_verify_width = 0;
    std::int32_t domain        = 0;
    bool verifying             = false;
    std::vector<SequenceState*> verified; // the last verify()'s sequences, in order
    // What the segments a thread queues compute: a verification (its record set, verified rows and
    // width), or a guessed window ahead of the verdict on the front ranks (pipeline windows), which
    // reads the sequence's shadow states. The host thread sets it around verify(); a rank thread
    // takes it from its task.
    struct PassMode {
        bool verify         = false;
        bool ahead          = false;
        std::uint32_t set   = 0;
        std::int32_t rows   = 0;
        std::uint32_t width = 0;
    };
    static inline thread_local PassMode pass_mode{};
    std::uint32_t verified_set = 0; // the record set of the last verification
    // The MTP chain on the head rank: token ids and positions of a pass's cells (a draft step's at
    // step * sequences), staged from the host.
    DeviceBuffer mtp_ids, mtp_positions, ngram_prefetch_tokens;
    DeviceBuffer mtp_logprobs;
    std::unique_ptr<PinnedHostBuffer> mtp_staging;
    cudaEvent_t mtp_staged       = nullptr;
    std::uint32_t mtp_routes     = 0; // MTP route pairs recorded for the expert cache
    // While a draft chain is captured: its replays count no routes for the expert cache (the
    // catch-up passes after a commit still do).
    bool capturing_draft = false;
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
    std::unique_ptr<HybridExperts> hybrid;
    const std::atomic<bool>* cancelled = nullptr;
    // Disk-resident experts: made resident before each layer's experts run.
    std::unique_ptr<ExpertStream> stream;
    // GGUF host or disk experts: a decode or verification call's missing experts, staged (or
    // computed by the CPU) while its cached ones run.
    std::unique_ptr<ExpertMisses> misses;
    // Each expert layer's routes of the current pass, [layer][10 * prefill_chunk] in one block per
    // rank (a stage's layers are consecutive), so one copy brings a rank's text layers home.
    std::vector<DeviceBuffer> route_blocks; // by rank
    std::vector<void*> route_records;       // by expert layer
    std::unique_ptr<PinnedHostBuffer> route_host;
    std::uint32_t pending_routes = 0;

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
    // With the automatic expert cache the pool buffers are cache slots between prompts
    // (ExpertCache::lend): a wide pass takes them back, the next decode pass returns them.
    bool pools_lent = false;
    std::unique_ptr<PinnedHostBuffer> slot_entries;
    // Wide host calls of at least kPrefetchTokens touch nearly every expert of a layer: every
    // expert the cache lacks is copied into the pool on the transfer stream while the layers
    // before run, without waiting for the layer's routes. Two pinned entry tables alternate.
    // A layer's copies are hundreds of operations, and an enqueue blocks while the stream's queue
    // is full, so a thread of their own (pool_copier) issues them: the layers between keep being
    // launched.
    static constexpr std::int32_t kPrefetchTokens = 256;
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
    std::uint32_t span_tokens = 0;
    DeviceBuffer span_stack, span_positions, span_rope, span_rows;
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
    // A span's chunk before its last runs a layer's experts: the pool keeps that layer's experts.
    bool span_hold = false;

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
        if (options.prefill_chunk == 0 || options.prefill_chunk > 4096 || options.sequences == 0 ||
            options.max_context == 0) {
            throw std::invalid_argument("qwen4_exp executor: chunk must be in [1, 4096]");
        }
        if (options.ngram) {
            table = std::make_unique<NgramTableReader>(options.ngram->layout, options.ngram_read);
        }
        max_logit_rows = std::min<std::uint32_t>(options.prefill_chunk, 512);
        if (!std::isfinite(options.draft_min_p) || options.draft_min_p < 0 ||
            options.draft_min_p > 1 || (options.draft_min_p != 0 && options.draft_tokens == 0)) {
            throw std::invalid_argument("qwen4_exp: draft_min_p needs MTP and a finite value in [0,1]");
        }
        if (options.draft_tokens > 0) {
            if (!model.mtp_weights()) {
                throw std::invalid_argument("qwen4_exp: drafts need a model loaded with its MTP "
                                            "block");
            }
            if (options.draft_tokens > 15) {
                throw std::invalid_argument("qwen4_exp: an MTP round drafts 1 to 15 tokens");
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
        if (options.kv_pool_tokens != 0) {
            const std::uint64_t asked =
                (std::uint64_t(options.kv_pool_tokens) + kPagedKVPageSize - 1) / kPagedKVPageSize;
            // A pool that holds every sequence's whole window is no pool.
            if (asked < std::uint64_t(pages) * options.sequences) {
                pool_pages = static_cast<std::uint32_t>(asked);
            }
        }
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
        if (stream_slots != 0 && pool_pages != 0) {
            // Streamed layers keep each sequence's own host copy of its KV.
            std::fprintf(stderr, "qwen4_exp: streamed KV (--kv-resident) keeps a window per sequence; "
                                 "no shared KV pool\n");
            pool_pages = 0;
        }
        memory.ranks.resize(device.size());
        plan_weights();
        if (verify_width > 0) { plan_mtp(); }
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
        pipeline = pipeline_requested();
        allocate_ranks();
        allocate_vision();
        allocate_sequences();
        allocate_staging();
        if (verify_width > 0) { allocate_speculation(); }
        allocate_span();
        allocate_cache();
        memory.streamed_kv = stream_slots != 0;
        for (const auto& rank : memory.ranks) {
            memory.state_bytes += rank.state_bytes;
            memory.workspace_bytes += rank.workspace_bytes;
            memory.expert_cache_bytes += rank.expert_cache_bytes;
        }
    }

    ~Impl() {
        // Every queued pass reaches its device before the buffers it reads go away.
        drain_quietly();
        stop_rank_threads();
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
        for (std::size_t i = 0; i < window_timing.events.size(); ++i) {
            RankBinding bind(device, segments[i].rank);
            for (cudaEvent_t event : window_timing.events[i]) {
                if (event != nullptr) { cudaEventDestroy(event); }
            }
        }
        // A pass that failed between starting the row reads and waiting for them leaves them
        // writing into a staging buffer.
        if (table) {
            try {
                table->wait();
            } catch (...) {}
        }
        for (auto& sequence : sequences) {
            for (std::size_t i = 0; i < sequence.decode.size(); ++i) {
                RankBinding bind(device, segments[i].rank);
                sequence.decode[i].reset();
            }
            for (auto& width : sequence.verify) {
                for (std::size_t i = 0; i < width.segments.size(); ++i) {
                    RankBinding bind(device, segments[i].rank);
                    width.segments[i].reset();
                }
            }
            for (auto& chain : sequence.draft) {
                if (chain.chain.ready()) {
                    RankBinding bind(device, model.head_rank());
                    chain.chain.reset();
                }
            }
        }
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            for (cudaEvent_t event : rank.staged) { cudaEventDestroy(event); }
            if (rank.done != nullptr) { cudaEventDestroy(rank.done); }
            if (rank.copied != nullptr) { cudaEventDestroy(rank.copied); }
            if (rank.routes != nullptr) { cudaEventDestroy(rank.routes); }
            if (rank.side.fork != nullptr) { cudaEventDestroy(rank.side.fork); }
            if (rank.side.join != nullptr) { cudaEventDestroy(rank.side.join); }
            if (rank.side.stream != nullptr) { cudaStreamDestroy(rank.side.stream); }
        }
        if (mtp_staged != nullptr) {
            RankBinding bind(device, model.head_rank());
            cudaEventDestroy(mtp_staged);
        }
        for (const CrossRing& ring : crossings) {
            for (cudaEvent_t event : ring.written) { cudaEventDestroy(event); }
            for (cudaEvent_t event : ring.read) { cudaEventDestroy(event); }
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

    void plan_weights() {
        const auto& w   = model.weights();
        embedding_table = native_weight(model.weight(w.token_embedding).view);
        {
            RankBinding bind(device, model.head_rank());
            head = make_projection(std::vector{model.input(w.output_head, "text/final_hidden")});
            final_mixer = make_hc(model, w.final_mixer, config);
        }
        for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
            layers.push_back(make_layer_plan(w.layers[i], model.stages().placement(i).stage,
                                             config.num_experts));
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
        plan.head         = make_projection(
            std::vector{model.input(model.weights().output_head, "mtp/final_hidden")});
        plan.layer = make_layer_plan(w.layer, model.head_rank(), model.mtp_config()->num_experts);
        plan.final_mixer = make_hc(model, w.final_mixer, config);
        mtp              = std::move(plan);
    }

    // A layer's operands on its rank; `experts` its routed experts (the MTP block may keep more
    // than an expert-pruned text model).
    LayerPlan make_layer_plan(const LayerWeights& lw, std::size_t rank, std::uint32_t experts) {
        LayerPlan plan;
        plan.rank = rank;
        RankBinding bind(device, plan.rank);
        plan.attn_hc = make_hc(model, lw.attn_hc, config);
        plan.mlp_hc  = make_hc(model, lw.mlp_hc, config);
        if (const auto* g = std::get_if<GdnWeights>(&lw.mixer)) {
            GdnPlan gdn;
            gdn.qkv             = make_projection(model, {g->query, g->key, g->value});
            gdn.z               = make_projection(model, {g->z});
            gdn.a               = make_projection(model, {g->a_projection});
            gdn.b               = make_projection(model, {g->b_projection});
            gdn.output          = make_projection(model, {g->output});
            const auto channels = static_cast<std::int32_t>(gdn.qkv.rows);
            gdn.convolution =
                direct(model, g->convolution,
                       {channels, static_cast<std::int32_t>(config.linear_conv_kernel_dim)});
            const auto heads = static_cast<std::int32_t>(config.linear_num_value_heads);
            gdn.a_log        = direct(model, g->a_log, {heads}, DType::FP32);
            gdn.dt_bias      = direct(model, g->dt_bias, {heads}, DType::FP32);
            gdn.norm =
                direct(model, g->norm, {static_cast<std::int32_t>(config.linear_value_head_dim)});
            plan.gdn = std::move(gdn);
        } else {
            const auto& a = std::get<SparseAttentionWeights>(lw.mixer);
            QsaPlan qsa;
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
        if (const std::uint32_t group = planned_group_chunks(); group > 1 && rank == 0) {
            bytes = std::max(bytes, ops::moe_experts_gguf_workspace_bytes(
                                        static_cast<std::int32_t>(group) * t));
        }
        std::vector<const LayerPlan*> planned;
        for (const auto& plan : layers) {
            if (plan.rank == rank) { planned.push_back(&plan); }
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
                 // The MTP block's calls over cached experts only: their kept ids and weights.
                 ops::moe_experts_gguf_workspace_bytes(t) +
                     2 * ((std::size_t(config.num_experts_per_tok) * t * 4 + 255) / 256 * 256)});
        }
        for (const LayerPlan* layer : planned) {
            const LayerPlan& plan = *layer;
            if (plan.moe.native) {
                bytes = std::max(bytes, ops::moe_experts_native_workspace_bytes(t));
            }
            if (plan.gdn) {
                for (const Projection* p : {&plan.gdn->qkv, &plan.gdn->z, &plan.gdn->a,
                                            &plan.gdn->b, &plan.gdn->output}) {
                    bytes = std::max(bytes, projection_workspace(*p, t));
                }
                bytes =
                    std::max(bytes, ops::gated_delta_net_workspace_capacity_bytes(
                                        static_cast<std::int32_t>(config.linear_num_key_heads),
                                        static_cast<std::int32_t>(config.linear_num_value_heads),
                                        true, 1, t));
            }
            if (plan.qsa) {
                for (const Projection* p :
                     {&plan.qsa->query, &plan.qsa->gate, &plan.qsa->key, &plan.qsa->value,
                      &plan.qsa->index, &plan.qsa->output}) {
                    bytes = std::max(bytes, projection_workspace(*p, t));
                }
                bytes = std::max({bytes,
                                  ops::qsa_indexer_select_workspace_bytes(
                                      t, static_cast<std::int32_t>(pooled_slots)),
                                  ops::sparse_softmax_attention_workspace_bytes(t)});
            }
            if (plan.ple) {
                bytes = std::max({bytes, projection_workspace(plan.ple->key, t),
                                  projection_workspace(plan.ple->value, t),
                                  ops::ple_inject_workspace_bytes(t)});
            }
        }
        if (rank == model.head_rank()) {
            bytes = std::max(bytes,
                             projection_workspace(head, static_cast<std::int32_t>(max_logit_rows)));
        }
        return bytes + (16U << 20);
    }

    void allocate_ranks() {
        const std::uint64_t t     = options.prefill_chunk;
        const std::uint64_t h     = config.hidden_size;
        const std::uint64_t width = std::uint64_t(config.hc_count) * h;
        const std::uint64_t plane = std::max<std::uint64_t>(width, 6144); // widest BF16 plane rows
        const std::uint64_t row_b = options.ngram ? ops::ngram_row_bytes(options.ngram->format) : 0;
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
                {&rank.rows, std::uint64_t(config.ngram_heads()) * t * row_b},
            };
            if (r == model.head_rank()) {
                planes.push_back(
                    {&rank.logits, std::uint64_t(config.vocab_size) * max_logit_rows * 2});
            }
            std::uint64_t total = 0;
            for (const auto& [slot, bytes] : planes) { total += round_up(bytes); }
            rank.activations     = DeviceBuffer(total);
            std::uint64_t offset = 0;
            for (const auto& [slot, bytes] : planes) {
                *slot = static_cast<std::byte*>(rank.activations.p) + offset;
                offset += round_up(bytes);
            }
            rank.workspace = std::make_unique<DeviceArena>(workspace_bytes(r));
            // Pool slots: one per sequence, and with pipeline windows the shadow slot after them.
            const std::uint32_t slots = options.sequences + (pipeline ? 1 : 0);
            std::vector<std::int32_t> identity(std::max<std::uint32_t>(pages, slots));
            std::iota(identity.begin(), identity.end(), 0);
            if (pool_pages == 0) {
                rank.block_table = DeviceBuffer(std::size_t(pages) * 4);
                rank.block_table.copy_from_host(identity.data(), std::size_t(pages) * 4);
            } else {
                // A table per sequence into the shared pool, filled as its pages are taken.
                const std::vector<std::int32_t> zeros(std::size_t(pages) * options.sequences, 0);
                rank.block_table = DeviceBuffer(zeros.size() * 4);
                rank.block_table.copy_from_host(zeros.data(), zeros.size() * 4);
            }
            rank.slot_ids = DeviceBuffer(std::size_t(slots) * 4);
            rank.slot_ids.copy_from_host(identity.data(), std::size_t(slots) * 4);
            // ids, positions, RoPE positions, scatter columns, three-axis RoPE positions, rows.
            // One more buffer than there are stages: every stage can hold a pass while the host
            // stages the next. NINFER_QWEN4_STAGING_RING=N sets the ring (1: the host waits for
            // the previous pass on every rank, as before the ring), for A/B measurements.
            std::size_t ring = device.size() + 1;
            if (const char* value = std::getenv("NINFER_QWEN4_STAGING_RING")) {
                ring = std::clamp<std::size_t>(std::strtoull(value, nullptr, 10), 1, 16);
            }
            for (std::size_t s = 0; s < ring; ++s) {
                rank.staging.push_back(std::make_unique<PinnedHostBuffer>(
                    round_up(t * 28) + round_up(std::uint64_t(config.ngram_heads()) * t * row_b)));
                cudaEvent_t event = nullptr;
                CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
                rank.staged.push_back(event);
            }
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.done, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.copied, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.routes, cudaEventDisableTiming));
            CUDA_CHECK(cudaStreamCreateWithFlags(&rank.side.stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.side.fork, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.side.join, cudaEventDisableTiming));
            memory.ranks[r].workspace_bytes += total + rank.workspace->capacity();
            ranks.push_back(std::move(rank));
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
    }

    [[nodiscard]] qwen3_5::execution::VisionParameters
    vision_parameters_of(const qwen3_5::VisionWeights& w) const {
        return vision_parameters_for(model, w);
    }

    void set_media(std::uint32_t s, std::span<const MediaItem> items,
                   std::vector<std::int32_t> rope_positions, std::int32_t rope_delta) {
        if (!vision_context) {
            throw std::invalid_argument("the model was loaded without its Vision tower");
        }
        SequenceState& sequence = sequences.at(s);
        if (sequence.position != 0) {
            throw std::invalid_argument("qwen4_exp: media start a prompt at position zero");
        }
        if (rope_positions.empty() || rope_positions.size() % 3) {
            throw std::invalid_argument("qwen4_exp: RoPE positions must cover three axes");
        }
        const std::size_t tokens = rope_positions.size() / 3;
        std::vector<std::int32_t> columns(tokens, -1);
        RankBinding bind(device, 0);
        const DeviceSpan backing{vision_backing.p, vision_backing.bytes};
        const std::size_t h = config.hidden_size;
        std::size_t column  = 0;
        for (const MediaItem& item : items) {
            const auto& control = *item.control;
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
        }
        sequence.media_columns = std::move(columns);
        sequence.media_rope    = std::move(rope_positions);
        sequence.rope_delta    = rope_delta;
        // The MTP block drafts from text cells only; the sequence stops drafting until a reset.
        sequence.mtp_follows = false;
        CUDA_CHECK(cudaStreamSynchronize(ranks[0].stream));
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

    // One plane's bytes for `pages` pages of `page_tokens` positions.
    [[nodiscard]] std::uint64_t kv_plane_bytes(std::size_t plane, std::uint64_t page_tokens,
                                               std::uint64_t count) const {
        const auto [leading, dtype] = kv_planes()[plane];
        return std::uint64_t(leading) * page_tokens * config.num_key_value_heads * count *
               dtype_size(dtype);
    }

    // A view of a sparse-attention layer's KV planes `data(i)`: pages of `page_tokens` positions,
    // `count` of them, read through `table` (`entries` I32).
    template <class Data>
    [[nodiscard]] PagedKVLayerView kv_view(Data&& data, std::int32_t page_tokens,
                                           std::int32_t count, void* table,
                                           std::int32_t entries) const {
        PagedKVLayerView cache{};
        Tensor* targets[4] = {&cache.k_pages, &cache.v_pages, &cache.k_scale_pages,
                              &cache.v_scale_pages};
        const auto planes = kv_planes();
        for (std::size_t i = 0; i < planes.size(); ++i) {
            if (planes[i].first == 0) { continue; }
            *targets[i] = Tensor(data(i), planes[i].second,
                                 {planes[i].first, page_tokens,
                                  static_cast<std::int32_t>(config.num_key_value_heads), count});
        }
        cache.block_table  = Tensor(table, DType::I32, {entries});
        cache.head_dim     = static_cast<std::int32_t>(config.head_dim);
        cache.num_kv_heads = static_cast<std::int32_t>(config.num_key_value_heads);
        cache.storage      = options.kv_cache;
        return cache;
    }

    // A sparse-attention layer's KV planes of `count` pages in device memory (a shared pool's).
    void allocate_device_kv(LayerState& state, std::size_t rank, std::uint32_t count) {
        std::uint64_t device_kv = 0;
        const auto planes       = kv_planes();
        for (std::size_t i = 0; i < planes.size(); ++i) {
            if (planes[i].first == 0) { continue; }
            KvPlane& plane = state.kv[i];
            plane.bytes    = kv_plane_bytes(i, kPagedKVPageSize, count);
            plane.device   = DeviceBuffer(plane.bytes);
            plane.p        = plane.device.p;
            device_kv += plane.bytes;
        }
        memory.ranks[rank].state_bytes += device_kv;
        memory.kv_bytes += device_kv;
    }

    // A sparse-attention layer's per-sequence state: paged KV (one plane per K or V vector and
    // per scale, each [leading, page, head, page]), pooled keys and the indexer's tail. A
    // streamed layer keeps the paged KV in pinned host memory, and its pool and map on the device.
    void allocate_attention_state(LayerState& state, std::size_t rank) {
        std::uint64_t device_kv = 0, bytes = 0;
        const auto planes       = kv_planes();
        // A shared KV pool keeps the planes once, in pool_kv.
        for (std::size_t i = 0; pool_pages == 0 && i < planes.size(); ++i) {
            if (planes[i].first == 0) { continue; }
            KvPlane& plane = state.kv[i];
            plane.bytes    = kv_plane_bytes(i, kPagedKVPageSize, pages);
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
    }

    // Each rank's Gated DeltaNet layers keep their state in one pool with a slot per sequence,
    // the layout the replay fold of a speculative commit reads.
    void allocate_gdn_pools() {
        gdn_local.assign(layers.size(), 0);
        std::vector<std::uint32_t> count(ranks.size(), 0);
        std::int32_t channels = 0;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            if (!layers[i].gdn) { continue; }
            gdn_local[i] = count[layers[i].rank]++;
            channels     = layers[i].gdn->qkv.rows;
        }
        for (RankState& rank : ranks) {
            if (count[rank.rank] == 0) { continue; }
            RankBinding bind(device, rank.rank);
            const LinearAttentionStatePoolSpec spec{
                .layers         = count[rank.rank],
                .conv_channels  = channels,
                .conv_width     = static_cast<std::int32_t>(config.linear_conv_kernel_dim - 1),
                .value_heads    = static_cast<std::int32_t>(config.linear_num_value_heads),
                .value_head_dim = static_cast<std::int32_t>(config.linear_value_head_dim),
                .key_head_dim   = static_cast<std::int32_t>(config.linear_key_head_dim),
                .slot_count     = static_cast<std::int32_t>(options.sequences + (pipeline ? 1 : 0)),
            };
            LayoutBuilder builder;
            const auto layout = plan_linear_attention_state_pool(builder, spec);
            rank.gdn_backing  = DeviceBuffer(builder.finish(256, "Gated DeltaNet state"));
            rank.gdn          = std::make_unique<LinearAttentionStatePool>(
                DeviceSpan{rank.gdn_backing.p, rank.gdn_backing.bytes}, layout);
            memory.ranks[rank.rank].state_bytes += rank.gdn_backing.bytes;
        }
    }

    // ---- Shared KV pool (ExecutorOptions::kv_pool_tokens) ----
    // One set of KV planes per sparse-attention layer (and the MTP layer) of pool_pages pages; a
    // sequence's block table maps its positions' pages into them, taken from the free list as the
    // sequence grows and returned at reset. The host keeps the tables in pinned memory and copies
    // a sequence's new entries to every rank's table on that rank's stream before the pass that
    // first writes them.
    std::uint32_t pool_pages = 0;
    std::vector<LayerState> pool_kv; // by layer; index layers.size() is the MTP layer
    std::vector<std::int32_t> free_pages;
    std::vector<std::vector<std::int32_t>> page_of; // by sequence
    std::unique_ptr<PinnedHostBuffer> table_host;   // [sequences][pages]

    void allocate_pool() {
        pool_kv.resize(layers.size() + 1);
        for (std::size_t i = 0; i < layers.size(); ++i) {
            if (!layers[i].qsa) { continue; }
            RankBinding bind(device, layers[i].rank);
            allocate_device_kv(pool_kv[i], layers[i].rank, pool_pages);
        }
        if (mtp) {
            RankBinding bind(device, model.head_rank());
            allocate_device_kv(pool_kv[layers.size()], model.head_rank(), pool_pages);
        }
        free_pages.resize(pool_pages);
        // Taken from the back: the lowest pages first.
        for (std::uint32_t p = 0; p < pool_pages; ++p) {
            free_pages[p] = static_cast<std::int32_t>(pool_pages - 1 - p);
        }
        page_of.assign(options.sequences, {});
        table_host = std::make_unique<PinnedHostBuffer>(std::size_t(pages) * options.sequences * 4);
        std::memset(table_host->data(), 0, table_host->size());
        std::fprintf(stderr, "qwen4_exp: %u sequences share a KV pool of %llu tokens\n",
                     options.sequences,
                     static_cast<unsigned long long>(pool_pages) * kPagedKVPageSize);
    }

    // Pages for `sequence`'s positions up to `tokens` (at most max_context): true when it holds
    // them now; false, taking none, when the pool lacks some.
    bool reserve_pages(const SequenceState& sequence, std::uint64_t tokens) {
        if (pool_pages == 0) { return true; }
        const std::uint32_t need = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            (tokens + kPagedKVPageSize - 1) / kPagedKVPageSize, pages));
        auto& held = page_of[sequence.slot];
        if (need <= held.size()) { return true; }
        const std::size_t more = need - held.size();
        if (free_pages.size() < more) { return false; }
        auto* table = static_cast<std::int32_t*>(table_host->data()) +
                      std::size_t(sequence.slot) * pages;
        const std::size_t first = held.size();
        for (std::size_t k = 0; k < more; ++k) {
            held.push_back(free_pages.back());
            free_pages.pop_back();
            table[first + k] = held.back();
        }
        for (RankState& rank : ranks) {
            RankBinding bind(device, rank.rank);
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int32_t*>(rank.block_table.p) +
                                           std::size_t(sequence.slot) * pages + first,
                                       table + first, more * 4, cudaMemcpyHostToDevice,
                                       rank.stream));
        }
        return true;
    }

    void require_pages(const SequenceState& sequence, std::uint64_t tokens) {
        if (!reserve_pages(sequence, tokens)) {
            throw KvPoolExhausted("qwen4_exp: the shared KV pool has no pages left for sequence " +
                                  std::to_string(sequence.slot));
        }
    }

    // Returns a sequence's pages; the caller has waited for every pass that reads them.
    void release_pages(const SequenceState& sequence) {
        if (pool_pages == 0) { return; }
        auto& held = page_of[sequence.slot];
        free_pages.insert(free_pages.end(), held.rbegin(), held.rend());
        held.clear();
    }

    [[nodiscard]] std::uint64_t free_pool_tokens() const noexcept {
        return std::uint64_t(free_pages.size()) * kPagedKVPageSize;
    }
    [[nodiscard]] std::uint64_t held_pool_tokens(std::uint32_t s) const noexcept {
        return pool_pages == 0 ? 0 : std::uint64_t(page_of.at(s).size()) * kPagedKVPageSize;
    }

    void allocate_sequences() {
        const std::uint64_t width = std::uint64_t(config.hc_count) * config.hidden_size;
        allocate_gdn_pools();
        if (pool_pages != 0) { allocate_pool(); }
        for (std::uint32_t s = 0; s < options.sequences; ++s) {
            SequenceState sequence;
            sequence.slot = s;
            for (std::size_t i = 0; i < layers.size(); ++i) {
                const LayerPlan& plan = layers[i];
                RankBinding bind(device, plan.rank);
                LayerState state;
                if (plan.gdn) {
                    const auto slot = static_cast<std::int32_t>(s);
                    state.ssm       = ranks[plan.rank].gdn->recurrent_slot(gdn_local[i], slot);
                    state.conv      = ranks[plan.rank].gdn->conv_slot(gdn_local[i], slot);
                }
                if (plan.qsa) { allocate_attention_state(state, plan.rank); }
                if (plan.ple) {
                    state.history = DeviceBuffer(width * (config.ple_conv_kernel_size - 1) *
                                                 config.ngram.ngram_size * 4);
                    memory.ranks[plan.rank].state_bytes += state.history.bytes;
                }
                sequence.layers.push_back(std::move(state));
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
        if (!pipeline) { return; }
        // The shadow of the sequence's states on the front ranks: its states as if the window in
        // flight were kept whole, which a guessed window reads. Its KV and pooled keys are the
        // sequence's own (a guessed window writes positions after the verified one).
        ahead_layers.resize(layers.size());
        const auto shadow = static_cast<std::int32_t>(options.sequences);
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const LayerPlan& plan = layers[i];
            if (plan.rank == model.head_rank()) { continue; }
            RankBinding bind(device, plan.rank);
            LayerState& state = ahead_layers[i];
            if (plan.gdn) {
                state.ssm  = ranks[plan.rank].gdn->recurrent_slot(gdn_local[i], shadow);
                state.conv = ranks[plan.rank].gdn->conv_slot(gdn_local[i], shadow);
            }
            if (plan.qsa) {
                state.tail = DeviceBuffer(sequences.front().layers[i].tail.bytes);
                memory.ranks[plan.rank].state_bytes += state.tail.bytes;
            }
            if (plan.ple) {
                state.history = DeviceBuffer(sequences.front().layers[i].history.bytes);
                memory.ranks[plan.rank].state_bytes += state.history.bytes;
            }
        }
    }

    // Pipeline windows (NINFER_QWEN4_PIPELINE_WINDOWS=1, with rank threads): while the head rank
    // verifies a window, the front ranks run the next one, guessed by the MTP block on the
    // assumption that this window is kept whole with the block's bonus; a wrong guess is dropped.
    // One sequence, MTP drafts, device experts and CUDA Graphs.
    bool pipeline = false;
    std::vector<LayerState> ahead_layers; // the shadow states, by layer (front ranks)
    [[nodiscard]] bool pipeline_requested() const {
        const auto on = [](const char* name) {
            const char* value = std::getenv(name);
            return value != nullptr && value[0] == '1';
        };
        const bool split = std::any_of(segments.begin(), segments.end(), [&](const Segment& s) {
            return s.rank != model.head_rank();
        });
        return on("NINFER_QWEN4_PIPELINE_WINDOWS") && on("NINFER_QWEN4_RANK_THREADS") && split &&
               verify_width > 0 && options.cuda_graphs &&
               model.options().experts == ExpertResidency::Device;
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
        const std::size_t sets = pipeline ? 2 : 1;
        for (RankState& rank : ranks) {
            RankBinding bind(device, rank.rank);
            std::uint64_t bytes = 0;
            const bool ple_here =
                std::any_of(layers.begin(), layers.end(),
                            [&](const LayerPlan& p) { return p.ple && p.rank == rank.rank; });
            for (std::size_t set = 0; set < sets; ++set) {
                if (rank.gdn) {
                    const auto& pool = rank.gdn->spec();
                    const GdnReplayRecordSpec spec{
                        .layers          = static_cast<std::int32_t>(pool.layers),
                        .record_capacity = static_cast<std::int32_t>(options.sequences),
                        .width           = static_cast<std::int32_t>(verify_width),
                        .conv_channels   = pool.conv_channels,
                        .qk_heads        = static_cast<std::int32_t>(config.linear_num_key_heads),
                        .value_heads     = pool.value_heads,
                        .key_dim         = pool.key_head_dim,
                        .value_dim       = pool.value_head_dim,
                    };
                    LayoutBuilder builder;
                    const auto layout = plan_gdn_replay_records(builder, spec);
                    rank.record_backing[set] =
                        DeviceBuffer(builder.finish(256, "GDN replay records"));
                    rank.records[set].emplace(DeviceSpan{rank.record_backing[set].p,
                                                         rank.record_backing[set].bytes},
                                              layout);
                    rank.fold[set].emplace(*rank.records[set], rank.gdn->all_layers_view());
                    bytes += rank.record_backing[set].bytes;
                }
                if (attention[rank.rank] > 0) {
                    rank.index_records[set] =
                        DeviceBuffer(attention[rank.rank] * index_rows * columns * 2);
                    bytes += rank.index_records[set].bytes;
                }
                if (ple_here) {
                    rank.ple_records[set] = DeviceBuffer(stack_width * columns * 4);
                    bytes += rank.ple_records[set].bytes;
                }
                if (pipeline) {
                    rank.window_positions[set] = DeviceBuffer(columns * 4);
                    bytes += rank.window_positions[set].bytes;
                }
            }
            if (rank.gdn) {
                const auto& pool  = rank.gdn->spec();
                rank.conv_scratch = DeviceBuffer(std::uint64_t(pool.conv_channels) *
                                                 (config.linear_conv_kernel_dim - 1) * 2);
                bytes += rank.conv_scratch.bytes;
            }
            if (attention[rank.rank] > 0) {
                rank.tail_scratch = DeviceBuffer(std::uint64_t(config.indexer_head_dim) *
                                                 (config.indexer_compress_ratio - 1) * 4);
                bytes += rank.tail_scratch.bytes;
            }
            memory.ranks[rank.rank].workspace_bytes += bytes;
        }
        const std::size_t head = model.head_rank();
        RankBinding bind(device, head);
        const std::uint64_t cells = std::max<std::uint64_t>(
            options.prefill_chunk, std::uint64_t(verify_width) * options.sequences);
        mtp_ids       = DeviceBuffer(cells * 4);
        mtp_positions = DeviceBuffer(cells * 4);
        mtp_staging   = std::make_unique<PinnedHostBuffer>(cells * 8);
        if (options.draft_min_p > 0) {
            mtp_logprobs = DeviceBuffer(std::uint64_t(options.draft_tokens) * options.sequences * 4);
        }
        // Pipeline windows read a verification's rows on the host thread as they stage it.
        if (table && options.ngram_read.residency != NgramResidency::Ram && !pipeline) {
            draft_prefetch = std::make_unique<NgramDraftPrefetch>(
                device.rank(head), *table, ngram, config.eos_token_id, config.vocab_size,
                options.sequences, options.draft_tokens);
            ngram_prefetch_tokens = DeviceBuffer(2 * std::uint64_t(options.sequences) *
                                                  verify_width * sizeof(std::int32_t));
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&mtp_staged, cudaEventDisableTiming));
        // Each sequence's MTP tail as the draft steps advance it (a catch-up redoes those cells).
        for (SequenceState& sequence : sequences) {
            sequence.mtp_scratch_tail = DeviceBuffer(sequence.mtp.tail.bytes);
        }
        memory.ranks[head].workspace_bytes += mtp_ids.bytes + mtp_positions.bytes + mtp_logprobs.bytes + ngram_prefetch_tokens.bytes +
                                              sequences.size() * sequences.front().mtp.tail.bytes;
        mirror_embedding_for_head();
    }

    // A split model's MTP cells take their token embeddings from rank 0's table; without peer
    // access each draft step's lookup is a round trip head -> rank 0 -> head that the host waits
    // on twice. The head instead reads the rows from a mapped copy of the table's payload in pinned
    // host memory (the same bytes through the same lookup, a row of the bus per cell), which also
    // keeps the draft chain on one device, where it can be captured.
    std::unique_ptr<PinnedHostBuffer> head_embedding_host;
    Weight head_embedding; // payload == nullptr: look the rows up on rank 0
    void mirror_embedding_for_head() {
        const std::size_t head = model.head_rank();
        const Weight& table    = embedding_table;
        if (head == 0 || table.payload == nullptr || table.payload_bytes == 0) { return; }
        if (const char* value = std::getenv("NINFER_QWEN4_HEAD_EMBEDDING"); value && value[0] == '0') {
            return;
        }
        const auto* begin = static_cast<const std::byte*>(table.payload);
        const auto* end   = begin + table.payload_bytes;
        const auto inside = [&](const void* p) {
            return p == nullptr || (static_cast<const std::byte*>(p) >= begin &&
                                    static_cast<const std::byte*>(p) < end);
        };
        if (!inside(table.qdata) || !inside(table.qhigh) || !inside(table.scales) ||
            !inside(table.weight_divisors) || !inside(table.input_columns)) {
            return;
        }
        auto host = std::make_unique<PinnedHostBuffer>(table.payload_bytes);
        {
            RankBinding bind(device, 0);
            CUDA_CHECK(cudaMemcpy(host->data(), table.payload, table.payload_bytes,
                                  cudaMemcpyDeviceToHost));
        }
        void* mapped = nullptr;
        {
            RankBinding bind(device, head);
            if (cudaHostGetDevicePointer(&mapped, host->data(), 0) != cudaSuccess) {
                (void)cudaGetLastError();
                return;
            }
        }
        const auto rebase = [&](const void* p) -> const void* {
            if (p == nullptr) { return nullptr; }
            return static_cast<const std::byte*>(mapped) + (static_cast<const std::byte*>(p) - begin);
        };
        head_embedding               = table;
        head_embedding.payload       = mapped;
        head_embedding.qdata         = rebase(table.qdata);
        head_embedding.qhigh         = rebase(table.qhigh);
        head_embedding.scales        = rebase(table.scales);
        head_embedding.weight_divisors = rebase(table.weight_divisors);
        head_embedding.input_columns =
            static_cast<const std::int32_t*>(rebase(table.input_columns));
        head_embedding_host = std::move(host);
        std::fprintf(stderr, "qwen4_exp: MTP embeddings read from a %.1f MiB host copy of the table\n",
                     double(table.payload_bytes) / (1024.0 * 1024.0));
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
        if (residency == ExpertResidency::Disk) {
            if (options.expert_cache_bytes == 0) {
                throw std::invalid_argument("disk-resident experts need a device expert cache");
            }
            std::vector<ExpertStreamLayer> stream_layers;
            for (const auto& [plan, moe] : expert) {
                stream_layers.push_back(ExpertStreamLayer{
                    .rank    = plan->rank,
                    .stream  = ranks[plan->rank].stream,
                    .experts = {moe->located_gate, moe->located_up, moe->located_down},
                    .tables  = {plan->moe.gate.table.p, plan->moe.up.table.p,
                                plan->moe.down.table.p}});
            }
            stream = std::make_unique<ExpertStream>(device, model.files(), std::move(stream_layers),
                                                    bytes);
            for (std::size_t r = 0; r < bytes.size(); ++r) {
                memory.ranks[r].expert_cache_bytes += bytes[r];
            }
            return;
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
        std::optional<ExpertRouteCounts> profile;
        if (!options.hybrid_experts.routing_profile.empty()) {
            profile = read_expert_profile(options.hybrid_experts.routing_profile,
                                          model.info().artifact_id, widths);
        } else {
            profile = builtin_expert_profile(kExpertProfileFamily, widths);
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
                p.k            = std::int32_t(k == 2 ? config.moe_intermediate_size
                                                     : config.hidden_size);
                p.expert_bytes = std::int64_t(p.rows) * p.row_bytes;
                if (stream) {
                    p.located = located[k];
                } else {
                    p.host = tables[k]->pointers;
                }
            }
            layer.gate_table = static_cast<const void* const*>(plan->moe.gate.table.p);
            // The text layers' slots follow their misses; the MTP block's keep the cache's own
            // policy.
            if (cache && !stream && i < layers.size()) {
                cache->hand_over(i);
                layer.admit = true;
            }
            if (cache) {
                layer.resident = cache->storage(i);
            } else if (stream) {
                layer.resident[0] = stream->storage(plan->rank);
            }
            out.push_back(std::move(layer));
        }
        MissOptions miss;
        miss.token_capacity = kVectorTokens;
        miss.cpu_balance    = !options.hybrid_experts.dma_share && !options.hybrid_experts.mapped_misses;
        miss.cpu_share      = miss.cpu_balance ? 0.0 : 1.0 - double(options.hybrid_experts.dma_share.value_or(1.0F));
        miss.cpu_threads    = options.hybrid_experts.cpu_threads;
        miss.staging_bytes  = kStagingBytes;
        miss.tail_skip_rank = options.hybrid_experts.tail_skip_rank;
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
                CUDA_CHECK(cudaMemset(buffer.storage.p, 0, buffer.storage.bytes));
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
    // the cache or the copies; unselected experts get none. A run of such experts goes in one copy
    // per projection wherever the host's are adjacent, and zeros follow the run's last down, which
    // the matrix kernel reads past: inside a run the bytes after a down are the next expert's
    // down, as in a host bank.
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
        std::size_t e = 0;
        while (e < experts) {
            if (!wanted(e) || cached(e)) {
                for (int k = 0; k < 3; ++k) {
                    entries[k * experts + e] =
                        wanted(e) ? cache->cached(index, k, std::int32_t(e)) : nullptr;
                }
                ++e;
                continue;
            }
            std::size_t end = e + 1;
            while (end < experts && wanted(end) && !cached(end)) { ++end; }
            for (int k = 0; k < 3; ++k) {
                std::byte* bank = storage + pool.offset[k];
                for (std::size_t a = e; a < end;) {
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
                for (std::size_t x = e; x < end; ++x) {
                    entries[k * experts + x] = bank + x * bytes[k];
                }
            }
            CUDA_CHECK(cudaMemsetAsync(storage + pool.offset[2] + end * bytes[2], 0, kSlotTail,
                                       stream));
            e = end;
        }
    }

    // The pool buffer that text layer `index`'s wide calls read: with two, consecutive layers
    // alternate.
    [[nodiscard]] std::size_t pool_buffer(std::size_t index) const {
        return slot_pools[layers[index].rank].buffers.size() == 2 ? index % 2 : 0;
    }

    // Whether text layer `index` exists on `rank` and runs its wide calls through the pool.
    [[nodiscard]] bool pooled_layer(std::size_t index, std::size_t rank) const {
        return index < layers.size() && layers[index].rank == rank && !layers[index].moe.native;
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
        CUDA_CHECK(cudaEventSynchronize(target.free));
        const MoePlan& m          = plan.moe;
        const std::size_t experts = m.gate.pointers.size();
        const auto source         = std::size_t(state.next);
        state.next ^= 1;
        CUDA_CHECK(cudaEventSynchronize(state.copied[source]));
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
        for (std::size_t r = 0; r < slot_pools.size(); ++r) {
            if (slot_pools[r].slots == 0) { continue; }
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
            CUDA_CHECK(cudaEventSynchronize(rank.routes));
        }
        const std::uint64_t pairs =
            std::uint64_t(config.num_experts_per_tok) * options.prefill_chunk;
        const auto* host = static_cast<const std::int32_t*>(route_host->data());
        if (cache) {
            const std::size_t used = std::size_t(config.num_experts_per_tok) * pending_routes;
            // The tier learns from decode and verification passes; a prompt chunk, which routes
            // nearly every expert, only counts toward the statistics.
            const bool decode = pending_routes <= std::uint32_t(kVectorTokens);
            if (pending_routes > 0) {
                for (std::size_t i = 0; i < layers.size(); ++i) {
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
        mtp_routes     = 0;
    }

    // Brings the pass's routes to the host for the expert cache's counts; nothing else reads them.
    void record_routes(std::uint32_t tokens) {
        if (!cache || warming) { return; }
        const std::uint64_t pairs =
            std::uint64_t(config.num_experts_per_tok) * options.prefill_chunk;
        auto* host = static_cast<std::int32_t*>(route_host->data());
        for (std::size_t i = 0; i < layers.size();) {
            std::size_t end = i;
            while (end < layers.size() && layers[end].rank == layers[i].rank) { ++end; }
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
        pending_routes = tokens;
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
        reset_pipeline();
        drain();
        auto& sequence    = sequences.at(s);
        if (pool_pages != 0 && !page_of[s].empty()) {
            // Another sequence may take the pages (and their table entries) next.
            synchronize_ranks();
            release_pages(sequence);
        }
        if (std::find(verified.begin(), verified.end(), &sequence) != verified.end()) { verified.clear(); }
        sequence.position = 0;
        sequence.context  = NgramContext::sequence_start(ngram, config.eos_token_id);
        sequence.media_columns.clear();
        sequence.media_rope.clear();
        sequence.rope_delta  = 0;
        sequence.mtp_follows = true;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            RankBinding bind(device, layers[i].rank);
            const cudaStream_t stream = ranks[layers[i].rank].stream;
            LayerState& state         = sequence.layers[i];
            for (const auto& [data, bytes] : recurrent_regions(state)) {
                if (data != nullptr) { CUDA_CHECK(cudaMemsetAsync(data, 0, bytes, stream)); }
            }
            if (state.pooled.p != nullptr) {
                CUDA_CHECK(cudaMemsetAsync(state.pooled.p, 0, state.pooled.bytes, stream));
            }
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

    // A stage boundary's host bounce buffers (rank threads): the source's thread copies its stack
    // into one on its stream, the target's thread uploads it on its own, each copy ordered by
    // events. A buffer is written again only once the target has read it.
    struct CrossRing {
        std::vector<std::unique_ptr<PinnedHostBuffer>> buffers;
        std::vector<cudaEvent_t> written; // on the source's stream
        std::vector<cudaEvent_t> read;    // on the target's stream
    };
    std::vector<CrossRing> crossings; // by target rank
    static constexpr std::size_t kCrossRing = 2;

    // ---- Rank threads (NINFER_QWEN4_RANK_THREADS=1) ----
    //
    // Between devices without peer access, a stream waiting on another device's event or a staged
    // peer copy blocks the calling host thread until the other device gets there (Windows/WDDM).
    // One host thread queuing every rank's work therefore walks a pass through the pipeline at
    // device speed, and the next pass starts only once the last rank has it: the stages take
    // turns. With rank threads the main thread prepares a pass's inputs and posts it; each rank's
    // thread uploads them, waits on the host for its stack from the rank before (the only thread
    // that blocks), runs its segments and hands the stack on through pinned host memory. Prompt
    // chunks then fill every stage. The head rank's thread runs the MTP catch-up, whose cells take
    // the pass's embeddings from host memory (rank 0 copied them there) instead of a round trip to
    // rank 0, which would order rank 0's next chunk after the head's work.
    //
    // Prompt chunks and plain decode passes of device-resident experts without media take this
    // route; anything else, and every call that reads results or state, first waits until the
    // threads have queued all posted passes (drain()).
    struct RankTask {
        std::uint64_t id    = 0;
        std::size_t slot    = 0; // staging ring entry
        std::int32_t t      = 0; // 0: no pass, only `work`
        std::uint32_t logit_rows = 0;
        std::size_t row_bytes    = 0;
        std::vector<Part> parts;
        SequenceState* graph_sequence = nullptr; // replay its decode graphs
        bool catch_up = false;                   // run the MTP catch-up on the head rank
        std::vector<Committed> committed;        // its tokens point into `tokens`
        std::vector<std::int32_t> tokens;
        Executor::AfterHead after_head;          // on the head rank's thread, after the head
        // Pipeline windows: the pass's kind, the segments it runs, per-segment graphs to replay,
        // whether its stack crosses to the head rank through the guessed window's buffer (a
        // guessed window) or the head rank's segments take it from there (its verification), and
        // work each rank's thread does first (a commit, the shadow states, the guess chain).
        PassMode mode;
        std::size_t first_segment = 0, end_segment = ~std::size_t{0};
        std::vector<DecodeGraphExecutable>* graphs = nullptr;
        bool to_ahead = false, from_ahead = false;
        std::uint64_t ahead_number = 0;
        std::function<void(std::size_t)> work;
    };
    struct RankWorker {
        std::thread thread;
        std::deque<std::shared_ptr<const RankTask>> queue;
    };
    bool rank_threads = false;
    std::vector<RankWorker> workers;
    std::mutex worker_mutex;
    std::condition_variable worker_cv;
    std::vector<std::uint64_t> done_upto; // by rank, task ids
    // Stage boundaries, by target rank: stacks sent and received so far. Both ends count the
    // passes that cross the boundary in the same order, so a pass's crossing number picks its
    // buffer without the ids of passes that do not cross there.
    std::vector<std::uint64_t> cross_sent, cross_received;
    std::vector<std::uint64_t> send_count, receive_count; // each touched by one thread only
    std::vector<std::uint64_t> slot_owner;                // the pass task that last staged a slot
    std::uint64_t staged_passes = 0;
    std::uint64_t posted          = 0;
    bool workers_stop             = false;
    std::exception_ptr worker_error;
    std::vector<std::unique_ptr<PinnedHostBuffer>> embedded_host; // rank 0's, by staging entry
    DeviceBuffer embed_scratch; // rank 0: graph passes' embeddings for the MTP catch-up

    void start_rank_threads() {
        const char* value = std::getenv("NINFER_QWEN4_RANK_THREADS");
        if (value == nullptr || value[0] != '1' || ranks.size() < 2) { return; }
        if (model.options().experts != ExpertResidency::Device || hybrid || misses || stream) {
            return;
        }
        rank_threads = true;
        const std::size_t ring = ranks.front().staging.size();
        const std::size_t capacity =
            std::size_t(config.hc_count) * config.hidden_size * options.prefill_chunk * 4;
        crossings.resize(ranks.size());
        for (std::size_t to = 1; to < ranks.size(); ++to) {
            CrossRing& cross = crossings[to];
            for (std::size_t i = 0; i < kCrossRing; ++i) {
                cross.buffers.push_back(std::make_unique<PinnedHostBuffer>(capacity));
                cudaEvent_t written = nullptr, read = nullptr;
                {
                    RankBinding bind(device, to - 1);
                    CUDA_CHECK(cudaEventCreateWithFlags(&written, cudaEventDisableTiming));
                }
                {
                    RankBinding bind(device, to);
                    CUDA_CHECK(cudaEventCreateWithFlags(&read, cudaEventDisableTiming));
                }
                cross.written.push_back(written);
                cross.read.push_back(read);
            }
        }
        for (std::size_t s = 0; s < ring; ++s) {
            embedded_host.push_back(std::make_unique<PinnedHostBuffer>(
                std::size_t(config.hidden_size) * options.prefill_chunk * 2));
        }
        if (mtp && options.cuda_graphs) {
            RankBinding bind(device, 0);
            embed_scratch = DeviceBuffer(std::size_t(config.hidden_size) * kVectorTokens * 2);
        }
        done_upto.assign(ranks.size(), 0);
        cross_sent.assign(ranks.size(), 0);
        cross_received.assign(ranks.size(), 0);
        send_count.assign(ranks.size(), 0);
        receive_count.assign(ranks.size(), 0);
        slot_owner.assign(ring, 0);
        if (pipeline) { start_pipeline(); }
        workers = std::vector<RankWorker>(ranks.size());
        for (std::size_t r = 0; r < ranks.size(); ++r) {
            workers[r].thread = std::thread([this, r] { worker_loop(r); });
        }
    }

    void stop_rank_threads() noexcept {
        if (!rank_threads) { return; }
        {
            std::lock_guard lock(worker_mutex);
            workers_stop = true;
        }
        worker_cv.notify_all();
        for (RankWorker& worker : workers) {
            if (worker.thread.joinable()) { worker.thread.join(); }
        }
    }

    // Waits until `counter[rank]` reaches `value`; a failed thread ends every wait with its error.
    void wait_for(const std::vector<std::uint64_t>& counter, std::size_t rank, std::uint64_t value) {
        std::unique_lock lock(worker_mutex);
        worker_cv.wait(lock, [&] { return counter[rank] >= value || worker_error != nullptr; });
        if (worker_error) { std::rethrow_exception(worker_error); }
    }

    // Waits until the rank threads have queued every posted pass on their devices.
    void drain() {
        if (!rank_threads) { return; }
        for (std::size_t r = 0; r < ranks.size(); ++r) { wait_for(done_upto, r, posted); }
    }
    // The same for the head rank's thread alone: with pipeline windows the front ranks may still
    // be queuing a guessed window while the host reads the verdict or drafts on the head rank.
    void drain_head() {
        if (!rank_threads) { return; }
        if (!pipeline) {
            drain();
            return;
        }
        wait_for(done_upto, model.head_rank(), posted);
    }
    void drain_quietly() noexcept {
        try {
            drain();
        } catch (...) {}
    }
    void drain_head_quietly() noexcept {
        try {
            drain_head();
        } catch (...) {}
    }

    void worker_loop(std::size_t r) {
        for (;;) {
            std::shared_ptr<const RankTask> task;
            {
                std::unique_lock lock(worker_mutex);
                worker_cv.wait(lock, [&] { return workers_stop || !workers[r].queue.empty(); });
                if (workers[r].queue.empty()) { return; }
                task = workers[r].queue.front();
                workers[r].queue.pop_front();
            }
            try {
                bool failed = false;
                {
                    std::lock_guard lock(worker_mutex);
                    failed = worker_error != nullptr;
                }
                if (!failed) { run_rank_task(r, *task); }
            } catch (...) {
                std::lock_guard lock(worker_mutex);
                if (!worker_error) { worker_error = std::current_exception(); }
            }
            {
                std::lock_guard lock(worker_mutex);
                done_upto[r] = task->id;
            }
            worker_cv.notify_all();
        }
    }

    [[nodiscard]] std::size_t stack_bytes(std::int32_t t) const {
        return std::size_t(config.hc_count) * config.hidden_size * std::size_t(t) * 4;
    }

    // The source half of a stage boundary, on the source's thread: the stack goes to the
    // boundary's host buffer for this task once the target has read what it held before.
    void send_stack(std::size_t from, std::size_t to, const RankTask& task) {
        CrossRing& ring       = crossings[to];
        const std::uint64_t n = ++send_count[to];
        const std::size_t s   = n % kCrossRing;
        if (n > kCrossRing) {
            wait_for(cross_received, to, n - kCrossRing);
            CUDA_CHECK(cudaEventSynchronize(ring.read[s]));
        }
        RankState& source = ranks[from];
        CUDA_CHECK(cudaMemcpyAsync(ring.buffers[s]->data(), source.stack, stack_bytes(task.t),
                                   cudaMemcpyDeviceToHost, source.stream));
        CUDA_CHECK(cudaEventRecord(ring.written[s], source.stream));
        {
            std::lock_guard lock(worker_mutex);
            cross_sent[to] = n;
        }
        worker_cv.notify_all();
    }

    // The target half, on the target's thread: waits on the host for the source's copy, then
    // uploads the stack on its own stream.
    void receive_stack(std::size_t to, const RankTask& task) {
        CrossRing& ring       = crossings[to];
        const std::uint64_t n = ++receive_count[to];
        const std::size_t s   = n % kCrossRing;
        wait_for(cross_sent, to, n);
        CUDA_CHECK(cudaEventSynchronize(ring.written[s]));
        RankState& target = ranks[to];
        CUDA_CHECK(cudaMemcpyAsync(target.stack, ring.buffers[s]->data(), stack_bytes(task.t),
                                   cudaMemcpyHostToDevice, target.stream));
        CUDA_CHECK(cudaEventRecord(ring.read[s], target.stream));
        {
            std::lock_guard lock(worker_mutex);
            cross_received[to] = n;
        }
        worker_cv.notify_all();
    }

    // The guessed window's crossing to the head rank (pipeline windows): one buffer, written by
    // the last front rank once the head rank has read, or the host has dropped, the guess before;
    // read by the head rank only when the guess holds.
    std::unique_ptr<PinnedHostBuffer> ahead_buffer;
    cudaEvent_t ahead_written = nullptr, ahead_read = nullptr;
    std::uint64_t ahead_sent = 0, ahead_resolved = 0; // guessed windows (worker_mutex)
    std::uint64_t ahead_count = 0;                    // the host's count of guessed windows

    void send_ahead(std::size_t from, const RankTask& task, std::uint64_t number) {
        {
            std::unique_lock lock(worker_mutex);
            worker_cv.wait(lock, [&] { return ahead_resolved + 1 >= number || worker_error; });
            if (worker_error) { std::rethrow_exception(worker_error); }
        }
        RankState& source = ranks[from];
        CUDA_CHECK(cudaMemcpyAsync(ahead_buffer->data(), source.stack, stack_bytes(task.t),
                                   cudaMemcpyDeviceToHost, source.stream));
        CUDA_CHECK(cudaEventRecord(ahead_written, source.stream));
        {
            std::lock_guard lock(worker_mutex);
            ahead_sent = number;
        }
        worker_cv.notify_all();
    }

    void receive_ahead(std::size_t to, const RankTask& task, std::uint64_t number) {
        {
            std::unique_lock lock(worker_mutex);
            worker_cv.wait(lock, [&] { return ahead_sent >= number || worker_error; });
            if (worker_error) { std::rethrow_exception(worker_error); }
        }
        CUDA_CHECK(cudaEventSynchronize(ahead_written));
        RankState& target = ranks[to];
        CUDA_CHECK(cudaMemcpyAsync(target.stack, ahead_buffer->data(), stack_bytes(task.t),
                                   cudaMemcpyHostToDevice, target.stream));
        // The host waits for this upload before the next guess overwrites the buffer.
        CUDA_CHECK(cudaEventRecord(ahead_read, target.stream));
        CUDA_CHECK(cudaEventSynchronize(ahead_read));
        {
            std::lock_guard lock(worker_mutex);
            ahead_resolved = std::max(ahead_resolved, number);
        }
        worker_cv.notify_all();
    }

    void run_rank_task(std::size_t r, const RankTask& task) {
        RankBinding bind(device, r);
        RankState& rank = ranks[r];
        pass_mode       = task.mode;
        struct ModeReset {
            ~ModeReset() { pass_mode = {}; }
        } mode_reset;
        if (task.work) { task.work(r); }
        if (task.t == 0) { return; }
        const std::size_t end = std::min(task.end_segment, segments.size());
        bool runs             = false;
        for (std::size_t i = task.first_segment; i < end; ++i) { runs = runs || segments[i].rank == r; }
        if (!runs) { return; }
        const std::size_t c = options.prefill_chunk;
        auto* base          = static_cast<std::byte*>(rank.staging[task.slot]->data());
        const std::size_t bytes = std::size_t(task.t) * 4;
        CUDA_CHECK(cudaMemcpyAsync(rank.ids, base, bytes, cudaMemcpyHostToDevice, rank.stream));
        CUDA_CHECK(cudaMemcpyAsync(rank.positions, base + 4 * c, bytes, cudaMemcpyHostToDevice,
                                   rank.stream));
        CUDA_CHECK(cudaMemcpyAsync(rank.rope, base + 8 * c, bytes, cudaMemcpyHostToDevice,
                                   rank.stream));
        if (task.mode.verify && pipeline) {
            // The window's positions for its commit, which later passes' staging overwrites.
            CUDA_CHECK(cudaMemcpyAsync(rank.window_positions[task.mode.set].p, rank.positions, bytes,
                                       cudaMemcpyDeviceToDevice, rank.stream));
        }
        for (std::size_t i = task.first_segment; i < end; ++i) {
            const Segment& segment = segments[i];
            if (segment.rank != r) { continue; }
            if (i > 0 && segments[i - 1].rank != r) {
                if (task.from_ahead && i == task.first_segment) {
                    receive_ahead(r, task, task.ahead_number);
                } else {
                    receive_stack(r, task);
                }
            }
            if (segment.rows && table) {
                CUDA_CHECK(cudaMemcpyAsync(rank.rows, staged_rows(rank, task.slot, task.row_bytes).data(),
                                           task.row_bytes, cudaMemcpyHostToDevice, rank.stream));
            }
            if (task.graphs != nullptr) {
                (*task.graphs)[i].launch(rank.stream);
            } else if (task.graph_sequence != nullptr) {
                if (segment.embed && task.catch_up) {
                    // The graph's embedding plane is rewritten by its layers; the MTP catch-up
                    // takes the tokens' embeddings from a lookup of their own.
                    const auto h = static_cast<std::int32_t>(config.hidden_size);
                    Tensor rows(embed_scratch.p, DType::BF16, {h, task.t});
                    ops::embedding(Tensor(rank.ids, DType::I32, {task.t}), embedding_table, rows,
                                   rank.stream);
                    CUDA_CHECK(cudaMemcpyAsync(embedded_host[task.slot]->data(), embed_scratch.p,
                                               std::size_t(h) * task.t * 2, cudaMemcpyDeviceToHost,
                                               rank.stream));
                }
                task.graph_sequence->decode[i].launch(rank.stream);
            } else {
                void* embedded = segment.embed && task.catch_up ? embedded_host[task.slot]->data()
                                                                 : nullptr;
                run_segment(task.parts, segment, task.t, task.logit_rows, embedded);
            }
            if (segment.head && task.catch_up) {
                mtp_catch_up(task.committed,
                             static_cast<const std::byte*>(embedded_host[task.slot]->data()));
            }
            if (segment.head && task.after_head) {
                task.after_head(rank.stream, head_logits(std::size_t(task.logit_rows)));
            }
            if (i + 1 < segments.size() && segments[i + 1].rank != r) {
                if (task.to_ahead && i + 1 == end) {
                    send_ahead(r, task, task.ahead_number);
                } else if (i + 1 < end) {
                    send_stack(r, segments[i + 1].rank, task);
                }
            }
        }
        // The staging entry (and rank 0's embeddings in it) may be rewritten once this is done.
        CUDA_CHECK(cudaEventRecord(rank.staged[task.slot], rank.stream));
    }

    // Whether a pass over `parts` can go to the rank threads.
    [[nodiscard]] bool threadable(std::span<const Part> parts, std::int32_t t) const {
        if (!rank_threads || verifying) { return false; }
        for (const Part& part : parts) {
            if (!part.sequence->media_rope.empty() &&
                part.sequence->position < part.sequence->media_rope.size() / 3) {
                return false;
            }
        }
        (void)t;
        return true;
    }

    // Prepares a pass's inputs on the host and posts it to the rank threads; the sequences'
    // positions advance at once (the threads read positions from the staged planes).
    void threaded_pass(std::span<const Part> parts, std::span<const std::int32_t> tokens,
                       std::uint32_t logit_rows, Executor::AfterHead after_head = {}) {
        const auto t = static_cast<std::int32_t>(tokens.size());
        for (const Part& part : parts) {
            if (part.sequence->position + std::uint32_t(part.count) > options.max_context) {
                throw std::invalid_argument("qwen4_exp forward: the sequence exceeds max_context");
            }
        }
        std::vector<NgramContext*> contexts;
        for (const Part& part : parts) { contexts.push_back(&part.sequence->context); }
        auto task        = stage_task(parts, tokens, logit_rows, contexts, 0, segments.size());
        task->after_head = std::move(after_head);
        // Decode graphs: the first single-token pass runs eagerly, the second captures them (with
        // the threads drained, since a capture forbids other threads' launches).
        SequenceState& first = *parts.front().sequence;
        if (options.cuda_graphs && t == 1 && parts.size() == 1) {
            if (first.decode.empty() && first.decode_steps++ > 0) {
                drain();
                capture_decode(first);
            }
            if (!first.decode.empty()) { task->graph_sequence = &first; }
        }
        if (mtp) {
            task->catch_up = true;
            task->tokens.assign(tokens.begin(), tokens.end());
            for (const Part& part : parts) {
                task->committed.push_back(
                    {part.sequence, part.column, part.count, part.sequence->position,
                     std::span<const std::int32_t>(task->tokens)
                         .subspan(std::size_t(part.column), std::size_t(part.count))});
            }
        }
        post(task);
        for (const Part& part : parts) {
            part.sequence->position += static_cast<std::uint32_t>(part.count);
        }
    }

    // A task id for the next post.
    std::shared_ptr<RankTask> new_task() {
        auto task = std::make_shared<RankTask>();
        task->id  = posted + 1;
        return task;
    }

    void post(const std::shared_ptr<RankTask>& task) {
        {
            std::lock_guard lock(worker_mutex);
            if (worker_error) { std::rethrow_exception(worker_error); }
            if (task->id != posted + 1) { throw std::logic_error("qwen4_exp: rank tasks out of order"); }
            for (RankWorker& worker : workers) { worker.queue.push_back(task); }
            posted = task->id;
        }
        worker_cv.notify_all();
    }

    // The host half of a pass to the rank threads: a staging entry (once every rank has queued the
    // pass that staged it last) with the tokens, their positions from each part's sequence position
    // plus `offset`, their RoPE positions and, when a segment in [first, end) reads them, their
    // n-gram rows, hashed from `contexts` (one per part, advanced over the part's tokens) and read
    // on this thread.
    std::shared_ptr<RankTask> stage_task(std::span<const Part> parts,
                                         std::span<const std::int32_t> tokens,
                                         std::uint32_t logit_rows,
                                         std::span<NgramContext* const> contexts,
                                         std::uint32_t offset, std::size_t end,
                                         std::size_t first_segment = 0) {
        for (const Part& part : parts) {
            require_pages(*part.sequence, std::uint64_t(part.sequence->position) + offset +
                                              std::uint64_t(part.count));
        }
        const std::size_t ring = ranks.front().staging.size();
        auto task              = new_task();
        const std::size_t slot = ++staged_passes % ring;
        if (slot_owner[slot] != 0) {
            for (std::size_t r = 0; r < ranks.size(); ++r) { wait_for(done_upto, r, slot_owner[slot]); }
            for (auto& rank : ranks) { CUDA_CHECK(cudaEventSynchronize(rank.staged[slot])); }
        }
        slot_owner[slot]    = task->id;
        task->slot          = slot;
        task->t             = static_cast<std::int32_t>(tokens.size());
        task->logit_rows    = logit_rows;
        task->first_segment = first_segment;
        task->end_segment   = end;
        task->parts.assign(parts.begin(), parts.end());
        std::vector<std::int32_t> positions(tokens.size()), rope(tokens.size());
        const std::size_t heads = config.ngram_heads();
        bool rows_read          = false;
        for (std::size_t i = first_segment; i < end; ++i) { rows_read = rows_read || segments[i].rows; }
        rows_read         = rows_read && table;
        std::size_t row_b = 0;
        if (rows_read) {
            row_ids.resize(tokens.size() * heads);
            row_b = ops::ngram_row_bytes(options.ngram->format);
        }
        for (std::size_t k = 0; k < parts.size(); ++k) {
            const Part& part              = parts[k];
            const SequenceState& sequence = *part.sequence;
            const auto start = static_cast<std::int32_t>(sequence.position + offset);
            const auto at    = positions.begin() + part.column;
            std::iota(at, at + part.count, start);
            for (std::int32_t i = 0; i < part.count; ++i) {
                rope[std::size_t(part.column + i)] = start + i + sequence.rope_delta;
            }
            const auto own = tokens.subspan(std::size_t(part.column), std::size_t(part.count));
            if (rows_read) {
                ngram_row_ids(ngram, own, config.eos_token_id, config.vocab_size, *contexts[k],
                              std::span(row_ids).subspan(std::size_t(part.column) * heads,
                                                         std::size_t(part.count) * heads));
            } else if (table) {
                std::vector<std::uint64_t> scratch(own.size() * heads);
                ngram_row_ids(ngram, own, config.eos_token_id, config.vocab_size, *contexts[k],
                              scratch);
            }
        }
        const std::size_t c = options.prefill_chunk;
        if (rows_read) {
            task->row_bytes = row_ids.size() * row_b;
            // Read on this thread: the threads only upload the rows.
            table->submit(row_ids, staged_rows(ranks[rows_rank], slot, task->row_bytes));
            table->wait();
            const auto rows = staged_rows(ranks[rows_rank], slot, task->row_bytes);
            for (auto& rank : ranks) {
                if (&rank == &ranks[rows_rank]) { continue; }
                const bool consumes = std::any_of(segments.begin(), segments.end(), [&](const Segment& s) {
                    return s.rows && s.rank == rank.rank;
                });
                if (consumes) {
                    std::memcpy(staged_rows(rank, slot, task->row_bytes).data(), rows.data(),
                                task->row_bytes);
                }
            }
        }
        for (auto& rank : ranks) {
            auto* base = static_cast<std::byte*>(rank.staging[slot]->data());
            std::memcpy(base, tokens.data(), tokens.size() * 4);
            std::memcpy(base + 4 * c, positions.data(), positions.size() * 4);
            std::memcpy(base + 8 * c, rope.data(), rope.size() * 4);
        }
        return task;
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
            std::iota(first, first + part.count, static_cast<std::int32_t>(sequence.position));
            for (std::int32_t i = 0; i < part.count; ++i) {
                rope[std::size_t(part.column + i)] =
                    static_cast<std::int32_t>(sequence.position) + i + sequence.rope_delta;
            }
            if (table) {
                // A verification hashes from a copy, which commit() replaces by the context
                // advanced over the tokens it keeps.
                NgramContext& context =
                    verifying ? part.sequence->verify_context : part.sequence->context;
                ngram_row_ids(ngram,
                              tokens.subspan(std::size_t(part.column), std::size_t(part.count)),
                              config.eos_token_id, config.vocab_size, context,
                              std::span(row_ids).subspan(std::size_t(part.column) * heads,
                                                         std::size_t(part.count) * heads));
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
                        at < prompt ? sequence.media_rope[axis * prompt + at]
                                    : static_cast<std::int32_t>(at) + sequence.rope_delta;
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
        staging_slot = (staging_slot + 1) % ranks.front().staging.size();
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            // The staging buffer is rewritten only once its last upload, ring-size passes ago, has
            // left it.
            CUDA_CHECK(cudaEventSynchronize(rank.staged[staging_slot]));
            auto* base = static_cast<std::byte*>(rank.staging[staging_slot]->data());
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
            if (!scatter.empty() && rank.rank == 0) { upload(rank.scatter, 12 * c, scatter); }
            if (mrope) { upload(rank.mrope, 16 * c, mrope_positions); }
            if (table && read_rows && rank.rank == rows_rank) {
                // The rows are read while the layers before the PLE layer run; stage_rows()
                // uploads them and marks the staging buffer free.
                table->submit(row_ids, staged_rows(rank, row_ids.size() * row_b));
                rows_waited = false;
            }
            const bool ple_here =
                std::any_of(layers.begin(), layers.end(),
                            [&](const LayerPlan& p) { return p.ple && p.rank == rank.rank; });
            if (!ple_here || !read_rows) { CUDA_CHECK(cudaEventRecord(rank.staged[staging_slot], rank.stream)); }
        }
    }

    // The n-gram rows' place in a rank's staging buffer.
    std::span<std::uint8_t> staged_rows(RankState& rank, std::size_t bytes) const {
        return staged_rows(rank, staging_slot, bytes);
    }
    std::span<std::uint8_t> staged_rows(RankState& rank, std::size_t slot, std::size_t bytes) const {
        auto* base = static_cast<std::byte*>(rank.staging[slot]->data());
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
        CUDA_CHECK(cudaEventRecord(rank.staged[staging_slot], rank.stream));
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
            CUDA_CHECK(cudaEventSynchronize(events.ready));
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

    // A verification's record row of the part: its position among the verified sequences.
    [[nodiscard]] static std::int32_t record_row(const Part& part) {
        return part.column / static_cast<std::int32_t>(pass_mode.width);
    }

    void run_gdn(const LayerPlan& plan, std::size_t index, LayerState& state, RankState& rank,
                 const Part& part) {
        const GdnPlan& g     = *plan.gdn;
        WorkspaceArena& ws   = *rank.workspace;
        const cudaStream_t s = rank.stream;
        const auto t         = part.count;
        const auto h         = static_cast<std::int32_t>(config.hidden_size);
        const auto kd =
            static_cast<std::int32_t>(config.linear_num_key_heads * config.linear_key_head_dim);
        const auto vd =
            static_cast<std::int32_t>(config.linear_num_value_heads * config.linear_value_head_dim);
        const auto nk = static_cast<std::int32_t>(config.linear_num_key_heads);
        const auto nv = static_cast<std::int32_t>(config.linear_num_value_heads);
        const auto dh = static_cast<std::int32_t>(config.linear_value_head_dim);
        const Tensor mixed(columns(rank.mixed, h, part.column, 2), DType::BF16, {h, t});
        // A verification records its raw projection, the transitions and the gates for the fold
        // that commits a prefix, and leaves the sequence's states as they were.
        std::optional<GdnReplayRecordLayer> record;
        if (pass_mode.verify) {
            const auto row = record_row(part);
            const auto all = rank.records[pass_mode.set]->layer(
                static_cast<std::int32_t>(gdn_local[index]), pass_mode.rows,
                static_cast<std::int32_t>(pass_mode.width));
            record = GdnReplayRecordLayer{all.conv.slice(2, row, 1), all.key.slice(3, row, 1),
                                          all.value.slice(3, row, 1), all.gate.slice(3, row, 1)};
        }
        // A guessed window reads the shadow slot.
        const LayerState& source = pass_mode.ahead ? ahead_layers[index] : state;
        const std::uint32_t source_slot =
            pass_mode.ahead ? options.sequences : part.sequence->slot;
        Tensor qkv = record ? Tensor(record->conv.data, DType::BF16, {g.qkv.rows, t})
                            : Tensor(rank.a, DType::BF16, {g.qkv.rows, t});
        Tensor z(rank.b, DType::BF16, {vd, t});
        Tensor ga(rank.c, DType::BF16, {nv, t});
        Tensor gb(static_cast<std::byte*>(rank.c) + round_up(std::uint64_t(nv) * t * 2),
                  DType::BF16, {nv, t});
        project(g.qkv, mixed, qkv, ws, s);
        project(g.z, mixed, z, ws, s);
        project(g.a, mixed, ga, ws, s);
        project(g.b, mixed, gb, ws, s);
        // Convolution in place of the projections' q/k/v planes.
        auto* conv_out = static_cast<std::byte*>(rank.d);
        Tensor q(conv_out, DType::BF16, {kd, t});
        Tensor k(conv_out + round_up(std::uint64_t(kd) * t * 2), DType::BF16, {kd, t});
        Tensor v(rank.e, DType::BF16, {vd, t});
        const auto window = static_cast<std::int32_t>(config.linear_conv_kernel_dim - 1);
        Tensor conv_state(source.conv.data, DType::BF16, {g.qkv.rows, window});
        Tensor conv_next =
            record ? Tensor(rank.conv_scratch.p, DType::BF16, {g.qkv.rows, window}) : conv_state;
        ops::causal_conv1d_silu_split(qkv, g.convolution, conv_state, conv_next, q, k, v, s);
        Tensor gate(rank.g, DType::FP32, {nv, t});
        Tensor beta(static_cast<std::byte*>(rank.g) + round_up(std::uint64_t(nv) * t * 4),
                    DType::FP32, {nv, t});
        ops::gdn_gating(ga, gb, g.a_log, g.dt_bias, gate, beta, s);
        Tensor q3(q.data, DType::BF16, {dh, nk, t}), k3(k.data, DType::BF16, {dh, nk, t});
        Tensor v3(v.data, DType::BF16, {dh, nv, t});
        Tensor o(rank.f, DType::BF16, {dh, nv, t});
        const float scale = 1.0F / std::sqrt(float(dh));
        if (record) {
            const Tensor states = rank.gdn->layer_view(gdn_local[index]).recurrent;
            const Tensor slot(static_cast<std::int32_t*>(rank.slot_ids.p) + source_slot,
                              DType::I32, {1});
            Tensor o4 = o.view({dh, nv, t, 1});
            ops::gated_delta_net_replay_record(q3.view({dh, nk, t, 1}), k3.view({dh, nk, t, 1}),
                                               v3.view({dh, nv, t, 1}), gate.view({nv, t, 1}),
                                               beta.view({nv, t, 1}), scale, states, Tensor{}, slot,
                                               record->key, record->value, record->gate, o4, s);
        } else {
            Tensor ssm(state.ssm.data, DType::FP32, {dh, dh, nv});
            auto scope = ws.scope();
            ops::gated_delta_net(q3, k3, v3, gate, beta, scale, true, ws, ssm, o, s);
        }
        Tensor rows_o(rank.f, DType::BF16, {dh, nv * t});
        Tensor rows_z(rank.b, DType::BF16, {dh, nv * t});
        Tensor gated_rows(rank.a, DType::BF16, {dh, nv * t});
        ops::gated_rmsnorm(rows_o, g.norm, rows_z, ops::GateActivation::Sigmoid,
                           config.rms_norm_eps, gated_rows, s);
        const Tensor gated(rank.a, DType::BF16, {vd, t});
        Tensor y(columns(rank.y, h, part.column, 2), DType::BF16, {h, t});
        project(g.output, gated, y, ws, s);
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

    [[nodiscard]] AttentionInputs target_inputs(std::size_t index, LayerState& state,
                                                RankState& rank, const Part& part) {
        AttentionInputs in{.positions = rank.positions + part.column,
                           .rope      = rank.rope + part.column,
                           .index     = nullptr,
                           .tail      = state.tail.p,
                           .first     = part.sequence->position};
        if (pass_mode.verify) {
            const std::uint64_t rows =
                std::uint64_t(config.indexer_n_heads + 1) * config.indexer_head_dim;
            const std::uint64_t columns = std::uint64_t(verify_width) * options.sequences;
            in.index = static_cast<std::byte*>(rank.index_records[pass_mode.set].p) +
                       (qsa_local[index] * columns + std::uint64_t(part.column)) * rows * 2;
            const DeviceBuffer& tail = pass_mode.ahead ? ahead_layers[index].tail : state.tail;
            CUDA_CHECK(cudaMemcpyAsync(rank.tail_scratch.p, tail.p, tail.bytes,
                                       cudaMemcpyDeviceToDevice, rank.stream));
            in.tail = rank.tail_scratch.p;
        }
        return in;
    }

    void run_qsa(const LayerPlan& plan, std::size_t kv_index, LayerState& state, RankState& rank,
                 const Part& part,
                 const AttentionInputs& in) {
        const QsaPlan& a     = *plan.qsa;
        WorkspaceArena& ws   = *rank.workspace;
        const cudaStream_t s = rank.stream;
        const auto t         = part.count;
        const auto h         = static_cast<std::int32_t>(config.hidden_size);
        const auto d         = static_cast<std::int32_t>(config.head_dim);
        const auto nq        = static_cast<std::int32_t>(config.num_attention_heads);
        const auto nk        = static_cast<std::int32_t>(config.num_key_value_heads);
        const Tensor mixed(columns(rank.mixed, h, part.column, 2), DType::BF16, {h, t});
        Tensor q(rank.a, DType::BF16, {d * nq, t});
        Tensor gate(rank.b, DType::BF16, {d * nq, t});
        auto* kv = static_cast<std::byte*>(rank.c);
        Tensor k(kv, DType::BF16, {d * nk, t});
        Tensor v(kv + round_up(std::uint64_t(d) * nk * t * 2), DType::BF16, {d * nk, t});
        Tensor index(in.index != nullptr ? in.index
                                         : kv + 2 * round_up(std::uint64_t(d) * nk * t * 2),
                     DType::BF16, {a.index.rows, t});
        project(a.query, mixed, q, ws, s);
        project(a.gate, mixed, gate, ws, s);
        project(a.key, mixed, k, ws, s);
        project(a.value, mixed, v, ws, s);
        project(a.index, mixed, index, ws, s);
        Tensor q3(rank.a, DType::BF16, {d, nq, t}), k3(k.data, DType::BF16, {d, nk, t});
        Tensor qo(rank.d, DType::BF16, {d, nq, t});
        Tensor ko(rank.e, DType::BF16, {d, nk, t});
        auto* const part_positions = const_cast<std::int32_t*>(in.positions);
        const Tensor positions(part_positions, DType::I32, {t});
        if (mrope) {
            // A media prompt: the three-axis positions of its images and video, each pair of
            // rotated dimensions on axis i % 3 (the interleaved MRoPE sections).
            const Tensor axes(rank.mrope, DType::I32, {t, 3});
            ops::rmsnorm(q3, a.query_norm, config.rms_norm_eps, true, qo, s);
            ops::rmsnorm(k3, a.key_norm, config.rms_norm_eps, true, ko, s);
            ops::rope(axes, static_cast<int>(float(d) * config.partial_rotary_factor),
                      config.rope_theta, qo, ko, s);
        } else {
            const Tensor rope(const_cast<std::int32_t*>(in.rope), DType::I32, {t});
            ops::rmsnorm_rope(rope, a.query_norm, a.key_norm, q3, k3, qo, ko, s);
        }
        // The paged KV: device pages (a shared KV pool's through the sequence's block table), or a
        // streamed layer's host copy. A streamed layer's call of at most stream_width tokens reads its
        // device pool once kv_stream_resolve has made the call's blocks resident there; a wider call
        // (a prompt chunk) reads the rank's staging copy, into which it appends as well, and then drops
        // the blocks it wrote from the pool.
        const auto paged_count = static_cast<std::int32_t>(pages);
        const LayerState& planes_of = pool_pages != 0 ? pool_kv[kv_index] : state;
        auto* const table = static_cast<std::int32_t*>(rank.block_table.p) +
                            (pool_pages != 0 ? std::size_t(part.sequence->slot) * pages : 0);
        const PagedKVLayerView cache = kv_view(
            [&](std::size_t i) { return planes_of.kv[i].p; }, kPagedKVPageSize,
            static_cast<std::int32_t>(pool_pages != 0 ? pool_pages : pages), table, paged_count);
        const bool streamed = state.map.table != nullptr;
        const bool pooled_read = streamed && t <= static_cast<std::int32_t>(stream_width);
        const Tensor v3(v.data, DType::BF16, {d, nk, t});
        ops::kv_cache_append(ko, v3, positions, cache, s);
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
            rank.kv_staged_end = in.first + std::uint32_t(t);
        }
        const auto di = static_cast<std::int32_t>(config.indexer_head_dim);
        Tensor pooled(state.pooled.p, DType::FP32, {di, static_cast<std::int32_t>(pooled_slots)});
        Tensor tail(in.tail, DType::FP32,
                    {di, static_cast<std::int32_t>(config.indexer_compress_ratio - 1)});
        const ops::QsaIndexerWeights iw{&a.index_query_norm, &a.index_key_norm};
        // The call's first position is the part's first staged one, read on the device.
        const Tensor first(part_positions, DType::I32, {1});
        ops::qsa_indexer_append(index, first, iw, config.rms_norm_eps, pooled, tail, s);
        Tensor selected(rank.selected, DType::I32,
                        {static_cast<std::int32_t>(config.indexer_block_budget()), t});
        Tensor counts(rank.counts, DType::I32, {t});
        {
            auto scope = ws.scope();
            ops::qsa_indexer_select(index, first, iw, config.rms_norm_eps, pooled, ws, selected,
                                    counts, s);
        }
        if (pooled_read) {
            ops::kv_stream_resolve(state.map, first, selected, counts, cache, read, s);
        }
        Tensor attention(rank.f, DType::BF16, {d, nq, t});
        {
            auto scope = ws.scope();
            ops::sparse_softmax_attention(qo, first, selected, counts, read,
                                          1.0F / std::sqrt(float(d)), ws, attention, s);
        }
        if (streamed && !pooled_read) {
            const auto block = [](std::uint32_t position) {
                return static_cast<std::int32_t>(position / ops::kKvStreamPageTokens);
            };
            ops::kv_stream_evict(state.map, block(in.first),
                                 block(in.first + std::uint32_t(t) - 1) + 1, s);
        }
        Tensor flat(rank.f, DType::BF16, {d * nq, t});
        ops::sigmoid_mul(gate, flat, s);
        Tensor y(columns(rank.y, h, part.column, 2), DType::BF16, {h, t});
        project(a.output, flat, y, ws, s);
    }

    void run_ple(const LayerPlan& plan, std::size_t index, LayerState& state, RankState& rank,
                 const Part& part) {
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
        project(p.key, embedding, key, ws, s);
        project(p.value, embedding, value, ws, s);
        const DeviceBuffer& own = pass_mode.ahead ? ahead_layers[index].history : state.history;
        Tensor history(own.p, DType::FP32,
                       {width, static_cast<std::int32_t>((config.ple_conv_kernel_size - 1) *
                                                         config.ngram.ngram_size)});
        const ops::PleInjectWeights weights{&p.norm_key, &p.norm_query, &p.norm_conv,
                                            &p.convolution};
        auto scope = ws.scope();
        if (pass_mode.verify) {
            // The history stays; commit() advances it over the kept positions' recorded input.
            Tensor normalized(static_cast<float*>(rank.ple_records[pass_mode.set].p) +
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
            ops::GgufMoeWeights banks = plan.moe.banks();
            const auto* table_p       = static_cast<const void* const*>(pool.buffers[b].tables.p);
            const std::size_t experts = plan.moe.gate.pointers.size();
            banks.gate.experts        = table_p;
            banks.up.experts          = table_p + experts;
            banks.down.experts        = table_p + 2 * experts;
            banks.device_resident     = true;
            const auto h              = static_cast<std::int32_t>(config.hidden_size);
            const Tensor mixed(rank.mixed, DType::BF16, {h, t});
            Tensor y(rank.y, DType::FP32, {h, t});
            {
                auto scope = rank.workspace->scope();
                ops::moe_experts_gguf(mixed, ids, weights, shared, banks, *rank.workspace, y, s);
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
        CUDA_CHECK(cudaStreamSynchronize(s));
        const MoePlan& m          = plan.moe;
        const std::size_t experts = m.gate.pointers.size();
        // The copies below overwrite the first buffer, so no buffer holds a prefetched layer.
        if (!prefetches.empty()) {
            drain_prefetches();
            for (auto& buffer : prefetches[rank.rank].buffers) { buffer.layer = ~std::size_t{0}; }
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
        ops::GgufMoeWeights banks = m.banks();
        const auto* table_p       = static_cast<const void* const*>(pool.buffers[0].tables.p);
        banks.gate.experts        = table_p;
        banks.up.experts          = table_p + experts;
        banks.down.experts        = table_p + 2 * experts;
        banks.device_resident     = true;
        const auto h              = static_cast<std::int32_t>(config.hidden_size);
        const Tensor mixed(rank.mixed, DType::BF16, {h, t});
        Tensor y(rank.y, DType::FP32, {h, t});
        auto scope = rank.workspace->scope();
        ops::moe_experts_gguf(mixed, ids, weights, shared, banks, *rank.workspace, y, s);
        return true;
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
        {
            auto scope = ws.scope();
            ops::moe_route(mixed, m.router, m.shared_gate, ws, ids, weights, shared, s);
        }
        if (m.native) {
            Tensor y(rank.y, DType::FP32, {h, t});
            if (hybrid) {
                hybrid->run(hybrid_layers[index], mixed, ids, weights, shared, m.native->banks(), t == 1 || pass_mode.verify,
                            ws, y);
                return;
            }
            auto scope = ws.scope();
            ops::moe_experts_native(mixed, ids, weights, shared, m.native->banks(), nullptr, ws, y, s);
            return;
        }
        if (misses && index == layers.size()) {
            // The MTP block only steers drafts, which verification checks: its missing experts
            // are dropped rather than fetched.
            Tensor y(rank.y, DType::FP32, {h, t});
            auto scope = ws.scope();
            Tensor kept_ids     = ws.alloc(DType::I32, {top, t});
            Tensor kept_weights = ws.alloc(DType::FP32, {top, t});
            misses->drop_missing(index, ids, weights, kept_ids, kept_weights, s);
            if (ops::moe_experts_gguf_decode_supported(m.banks(), t)) {
                const auto state = ops::moe_experts_gguf_decode_begin(mixed, kept_ids, shared,
                                                                      m.banks(), ws, s, &rank.side);
                ops::moe_experts_gguf_decode_finish(state, kept_ids, nullptr, kept_weights,
                                                    m.banks(), nullptr, nullptr, nullptr, nullptr,
                                                    y, s);
                return;
            }
            const auto stage = ops::moe_experts_gguf_begin(mixed, ws, s);
            ops::moe_experts_gguf_add(stage, mixed, kept_ids, kept_weights, &shared, m.banks(), ws,
                                      s);
            ops::moe_experts_gguf_finish(stage, y, s);
            return;
        }
        if (misses && t <= kVectorTokens) {
            // The cached pairs run at once; the missing experts arrive (or the CPU computes them)
            // meanwhile, and run in a second stage.
            Tensor y(rank.y, DType::FP32, {h, t});
            auto scope      = ws.scope();
            const auto call = misses->begin(index, ids, mixed, t, m.banks(), s);
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
        // The slot pools are sized for the text layers' experts; the MTP block's wide calls stay
        // on the vector products over its host banks.
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
            CUDA_CHECK(cudaStreamSynchronize(s));
            stream->prepare(index, std::span<const std::int32_t>(host, std::size_t(top) * t));
        }
        Tensor y(rank.y, DType::FP32, {h, t});
        auto scope = ws.scope();
        ops::moe_experts_gguf(mixed, ids, weights, shared, m.banks(), ws, y, s);
    }

    // Queues one segment of the pass for the parts' `t` tokens on its rank's stream; with `head`,
    // the logits of the last `logit_rows` tokens.
    // `embedded_host` (rank threads with MTP): the pass's token embeddings are also copied there,
    // pinned host memory the head rank reads for the MTP catch-up instead of asking rank 0 again.
    void run_segment(std::span<const Part> parts, const Segment& segment, std::int32_t t,
                     std::uint32_t logit_rows, void* embedded_host = nullptr) {
        RankState& rank = ranks[segment.rank];
        const auto h    = static_cast<std::int32_t>(config.hidden_size);
        const auto hc   = static_cast<std::int32_t>(config.hc_count);
        Tensor stack(rank.stack, DType::FP32, {h, hc, t});
        Tensor mixed(rank.mixed, DType::BF16, {h, t});
        if (segment.embed) {
            const Tensor ids(rank.ids, DType::I32, {t});
            ops::embedding(ids, embedding_table, mixed, rank.stream);
            // A media prompt's image and video tokens take the tower's merged embeddings.
            for (const MediaRun& run : media_runs) {
                const Tensor source(static_cast<std::byte*>(vision_embeddings.p) +
                                        std::size_t(run.source) * std::size_t(h) * 2,
                                    DType::BF16, {h, run.count});
                const Tensor columns(rank.scatter + run.offset, DType::I32, {run.count});
                ops::scatter(source, columns, mixed, rank.stream);
            }
            if (embedded_host != nullptr) {
                CUDA_CHECK(cudaMemcpyAsync(embedded_host, rank.mixed, std::size_t(h) * t * 2,
                                           cudaMemcpyDeviceToHost, rank.stream));
            }
            ops::hyper_connection_expand(mixed, stack, rank.stream);
        }
        // A layer's MoE output is written into the stack by the next layer's read, which takes
        // the same rounding, unless the PLE comes between.
        Tensor inject(rank.inject, DType::FP32, {hc, t});
        const Tensor moe_out(rank.y, DType::FP32, {h, t});
        bool pending = false;
        for (std::size_t i = segment.begin; i < segment.end; ++i) {
            const LayerPlan& plan = layers[i];
            if (plan.ple) {
                if (pending) { ops::hyper_connection_write(stack, moe_out, inject, rank.stream); }
                pending = false;
                for (const Part& part : parts) {
                    run_ple(plan, i, part.sequence->layers[i], rank, part);
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
            for (const Part& part : parts) {
                LayerState& state = part.sequence->layers[i];
                if (plan.gdn) {
                    run_gdn(plan, i, state, rank, part);
                } else {
                    run_qsa(plan, i, state, rank, part, target_inputs(i, state, rank, part));
                }
            }
            {
                auto scope = rank.workspace->scope();
                ops::hyper_connection_write_read(
                    stack, Tensor(rank.y, DType::BF16, {h, t}), inject, plan.mlp_hc.weights(),
                    config.rms_norm_eps, *rank.workspace, mixed, &inject, rank.stream);
            }
            run_moe(plan, rank, t, i);
            pending = true;
        }
        if (pending) { ops::hyper_connection_write(stack, moe_out, inject, rank.stream); }
        if (!segment.head) { return; }
        const auto n = static_cast<std::int32_t>(logit_rows);
        const Tensor last(rank.stack + std::size_t(hc) * h * (t - n), DType::FP32, {h, hc, n});
        Tensor head_mixed(rank.mixed, DType::BF16, {h, n});
        {
            auto scope = rank.workspace->scope();
            ops::hyper_connection_read(last, final_mixer.weights(), config.rms_norm_eps,
                                       *rank.workspace, head_mixed, nullptr, rank.stream);
        }
        Tensor logits(rank.logits, DType::BF16, {static_cast<std::int32_t>(config.vocab_size), n});
        project(head, head_mixed, logits, *rank.workspace, rank.stream);
    }

    // A graph per segment of a pass over one part of `t` tokens of `sequence` (a decode step, or a
    // verification while `verifying`).
    // Pipeline windows capture segments [first, end) only; the others stay empty.
    std::vector<DecodeGraphExecutable> capture_segments(SequenceState& sequence, std::int32_t t,
                                                        std::size_t first = 0,
                                                        std::size_t end   = ~std::size_t{0}) {
        std::vector<DecodeGraphExecutable> graphs;
        for (std::size_t i = 0; i < segments.size(); ++i) {
            const Segment& segment = segments[i];
            if (i < first || i >= end) {
                graphs.emplace_back();
                continue;
            }
            RankBinding bind(device, segment.rank);
            DecodeGraphDefinition definition;
            const Part part{.sequence = &sequence, .column = 0, .count = t};
            definition.capture(ranks[segment.rank].stream, [&] {
                run_segment(std::span(&part, 1), segment, t, std::uint32_t(t));
            });
            DecodeGraphExecutable graph;
            graph.instantiate(definition);
            graphs.push_back(std::move(graph));
        }
        return graphs;
    }

    void capture_decode(SequenceState& sequence) {
        sequence.decode = capture_segments(sequence, 1);
    }

    void check_tokens(std::span<const std::int32_t> tokens, std::uint32_t logit_rows) const {
        if (tokens.empty() || tokens.size() > options.prefill_chunk) {
            throw std::invalid_argument("qwen4_exp forward: 1..prefill_chunk tokens per call");
        }
        if (logit_rows == 0 || logit_rows > tokens.size() || logit_rows > max_logit_rows) {
            throw std::invalid_argument("qwen4_exp forward: invalid logit rows");
        }
        for (const auto token : tokens) {
            if (token < 0 || std::uint32_t(token) >= config.vocab_size) {
                throw std::invalid_argument("qwen4_exp forward: token outside the vocabulary");
            }
        }
    }

    // One pass over `parts`, whose tokens are `tokens` in column order. A pass of one single-token
    // part replays its sequence's decode graphs when it has them.
    void pass(std::span<const Part> parts, std::span<const std::int32_t> tokens,
              std::uint32_t logit_rows) {
        const auto t = static_cast<std::int32_t>(tokens.size());
        for (const Part& part : parts) {
            if (part.sequence->position + std::uint32_t(part.count) > options.max_context) {
                throw std::invalid_argument("qwen4_exp forward: the sequence exceeds max_context");
            }
            require_pages(*part.sequence, part.sequence->position + std::uint32_t(part.count));
        }
        settle_routes();
        // A copy queued for an earlier pass may predate the cache's last swaps.
        forget_prefetches();
        place_pools(t > kVectorTokens);
        if (hybrid) { hybrid->begin(cancelled, !warming); }
        stage_inputs(parts, tokens);
        // Disk-resident experts need the host between a layer's routing and its experts, so their
        // passes stay eager, and so does a batch of several sequences. A verification of one
        // sequence replays its graphs while its experts take the vector products. Native host
        // experts coordinate with their supervisor through captured handshakes.
        SequenceState& first = *parts.front().sequence;
        // Staged misses keep disk experts' decode and verification free of host synchronization.
        const bool host_free = !stream || misses;
        const bool graph     = options.cuda_graphs && t == 1 && host_free && !verifying;
        const bool verify_graph = options.cuda_graphs && verifying && parts.size() == 1 &&
                                  host_free && t <= kVectorTokens;
        if (misses) { misses->keep_alive(); }
        if (graph && first.decode.empty() && first.decode_steps++ > 0) { capture_decode(first); }
        auto* verification = verify_graph ? &first.verify.at(std::size_t(t)) : nullptr;
        if (verification && verification->segments.empty() && verification->runs++ > 0) {
            verification->segments = capture_segments(first, t);
        }
        const bool timed_window = verifying && window_timing_enabled();
        if (timed_window) { collect_window_timing(); }
        for (std::size_t i = 0; i < segments.size(); ++i) {
            const Segment& segment = segments[i];
            if (i > 0 && segments[i - 1].rank != segment.rank) {
                cross(segments[i - 1].rank, segment.rank, t);
            }
            RankBinding bind(device, segment.rank);
            if (timed_window) { record_window_event(i, 0, ranks[segment.rank].stream); }
            if (segment.rows && table) { stage_rows(ranks[segment.rank]); }
            if (graph && !first.decode.empty()) {
                first.decode[i].launch(ranks[segment.rank].stream);
            } else if (verification && !verification->segments.empty()) {
                verification->segments[i].launch(ranks[segment.rank].stream);
            } else {
                run_segment(parts, segment, t, logit_rows);
            }
            if (timed_window) { record_window_event(i, 1, ranks[segment.rank].stream); }
        }
        if (timed_window) { window_timing.pending = true; }
        if (hybrid) { hybrid->finish(); }
        // A verification commits its positions later (commit()).
        if (!verifying) {
            for (const Part& part : parts) {
                part.sequence->position += static_cast<std::uint32_t>(part.count);
            }
        }
        record_routes(static_cast<std::uint32_t>(t));
        // While a staged layer waits on the device for its missing experts, no other launch may
        // load a kernel lazily: loading can wait for the device to go idle, the waiting kernel for
        // the misses' service, and the service's own calls for the loader. The caller's next
        // kernels therefore follow a pass that may have waited.
        if (misses && t <= kVectorTokens) { synchronize_ranks(); }
    }

    // NINFER_QWEN4_ROUND_TIMING=1: each stage's device time in a verification window (events around
    // its segments, read at the next window), printed every 100 windows.
    struct WindowTiming {
        std::vector<std::array<cudaEvent_t, 2>> events; // by segment
        std::array<double, 8> stage{};
        std::uint64_t windows = 0;
        bool pending          = false;
    };
    WindowTiming window_timing;
    static bool window_timing_enabled() {
        static const bool enabled = [] {
            const char* value = std::getenv("NINFER_QWEN4_ROUND_TIMING");
            return value != nullptr && value[0] == '1';
        }();
        return enabled;
    }
    void record_window_event(std::size_t segment, int end, cudaStream_t stream) {
        auto& events = window_timing.events;
        if (events.size() != segments.size()) {
            events.resize(segments.size(), {nullptr, nullptr});
        }
        auto& event = events[segment][std::size_t(end)];
        if (event == nullptr) { CUDA_CHECK(cudaEventCreate(&event)); }
        CUDA_CHECK(cudaEventRecord(event, stream));
    }
    void collect_window_timing() {
        WindowTiming& w = window_timing;
        if (!w.pending) { return; }
        w.pending = false;
        for (std::size_t i = 0; i < w.events.size(); ++i) {
            RankBinding bind(device, segments[i].rank);
            CUDA_CHECK(cudaEventSynchronize(w.events[i][1]));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, w.events[i][0], w.events[i][1]));
            w.stage[std::min<std::size_t>(segments[i].rank, 7)] += ms;
        }
        if (++w.windows % 100 != 0) { return; }
        std::fprintf(stderr, "window timing | %llu windows | device ms/window:",
                     static_cast<unsigned long long>(w.windows));
        for (std::size_t r = 0; r < ranks.size() && r < w.stage.size(); ++r) {
            std::fprintf(stderr, " r%zu %.2f", r, w.stage[r] / 100.0);
        }
        std::fprintf(stderr, "\n");
        w.stage = {};
    }

    void synchronize_ranks() {
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            CUDA_CHECK(cudaStreamSynchronize(rank.stream));
        }
    }

    // The tokens a prompt span takes, 0 without spans. Spans need one device and GGUF host
    // experts, whose wide calls copy every uncached expert of a layer into the pool: per span
    // instead of per chunk.
    [[nodiscard]] std::uint32_t planned_span_tokens() const {
        const bool native = std::any_of(layers.begin(), layers.end(),
                                        [](const LayerPlan& plan) { return plan.moe.native.has_value(); });
        // One device: its segments split only where the PLE layer's rows arrive.
        const bool one_rank = std::all_of(segments.begin(), segments.end(),
                                          [](const Segment& segment) { return segment.rank == 0; });
        if (model.options().experts != ExpertResidency::Host || native || !one_rank) { return 0; }
        const std::uint32_t tokens =
            std::min<std::uint32_t>(options.max_context, kSpanChunks * options.prefill_chunk);
        return tokens > options.prefill_chunk ? tokens : 0;
    }

    // The chunks one expert call of a span's expert phase takes: as many as kGroupTokens holds.
    [[nodiscard]] std::uint32_t planned_group_chunks() const {
        if (planned_span_tokens() == 0) { return 1; }
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

    // The chunk after the expert group of a span's `n` tokens that starts at chunk `first` of
    // `chunks` (c tokens each but the last). A group needs the full pool and kPrefetchTokens or
    // more, so that it runs from the layer's prefetched pool copy; a last chunk of
    // kVectorTokens or fewer runs alone, as forward() runs it.
    [[nodiscard]] std::size_t group_end(std::size_t first, std::size_t chunks, std::size_t n,
                                        std::size_t c) const {
        if (span_group == 1 || slot_pools.empty() || slot_pools[0].slots < config.num_experts) {
            return first + 1;
        }
        std::size_t last = std::min<std::size_t>(chunks, first + span_group);
        if (last > first + 1 && n - (last - 1) * c <= std::size_t(kVectorTokens)) { --last; }
        return std::min(last * c, n) - first * c >= std::size_t(kPrefetchTokens) ? last
                                                                                 : first + 1;
    }

    // The experts of a span's chunks first..last-1 in one wide call through the pool, into
    // span_y. rank.mixed holds the first chunk's MoE input and the others' follow it. Each chunk
    // is routed on its own, as forward() routes it, into the group's route planes.
    void run_group(const LayerPlan& plan, RankState& rank, std::size_t index, std::size_t first,
                   std::size_t last, std::size_t n, std::size_t c) {
        const MoePlan& m      = plan.moe;
        const cudaStream_t st = rank.stream;
        const auto h          = static_cast<std::int32_t>(config.hidden_size);
        const auto top        = static_cast<std::int32_t>(config.num_experts_per_tok);
        auto* const ids_p     = static_cast<std::int32_t*>(span_ids.p);
        auto* const weights_p = static_cast<float*>(span_weights.p);
        auto* const shared_p  = static_cast<float*>(span_shared.p);
        for (std::size_t j = first; j < last; ++j) {
            const auto t         = static_cast<std::int32_t>(std::min(c, n - j * c));
            const std::size_t at = (j - first) * c;
            const Tensor mixed(static_cast<std::byte*>(rank.mixed) + at * std::size_t(h) * 2,
                               DType::BF16, {h, t});
            Tensor ids(ids_p + at * std::size_t(top), DType::I32, {top, t});
            Tensor weights(weights_p + at * std::size_t(top), DType::FP32, {top, t});
            Tensor shared(shared_p + at, DType::FP32, {t});
            auto scope = rank.workspace->scope();
            ops::moe_route(mixed, m.router, m.shared_gate, *rank.workspace, ids, weights, shared,
                           st);
        }
        const auto t = static_cast<std::int32_t>(std::min(last * c, n) - first * c);
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

    // A prompt span of several chunks, layer by layer: every chunk passes a layer before any passes
    // the next, so a wide call's pool copy of the layer's uncached experts serves every chunk. The
    // arithmetic of each chunk is that of forward(); the logits are the last chunk's.
    void forward_span(std::uint32_t s, std::span<const std::int32_t> tokens,
                      std::uint32_t logit_rows, std::span<const std::int32_t> next) {
        const std::size_t n = tokens.size();
        const std::size_t c = options.prefill_chunk;
        if (span_tokens == 0 || n > span_tokens) {
            throw std::invalid_argument("qwen4_exp forward: tokens exceed the prompt span");
        }
        SequenceState& sequence = sequences.at(s);
        const std::uint32_t start = sequence.position;
        if (start + n > options.max_context) {
            throw std::invalid_argument("qwen4_exp forward: the sequence exceeds max_context");
        }
        require_pages(sequence, std::uint64_t(start) + n);
        const std::size_t prompt = sequence.media_rope.size() / 3;
        if (prompt != 0 && start < prompt) {
            throw std::invalid_argument("qwen4_exp forward: a media prompt prefills chunk by chunk");
        }
        const std::size_t chunks = (n + c - 1) / c;
        const auto last_count    = static_cast<std::uint32_t>(n - (chunks - 1) * c);
        check_tokens(tokens.subspan((chunks - 1) * c), logit_rows);
        for (std::size_t k = 0; k + 1 < chunks; ++k) { check_tokens(tokens.subspan(k * c, c), 1); }
        RankBinding bind(device, 0);
        RankState& rank      = ranks[0];
        const cudaStream_t st = rank.stream;
        const auto h         = static_cast<std::int32_t>(config.hidden_size);
        const auto hc        = static_cast<std::int32_t>(config.hc_count);
        const std::size_t width = std::size_t(hc) * std::size_t(h);
        const auto chunk_stack  = [&](std::size_t k) {
            return static_cast<float*>(span_stack.p) + k * c * width;
        };
        const auto count_of = [&](std::size_t k) {
            return static_cast<std::int32_t>(std::min(c, n - k * c));
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
        for (std::size_t k = 0; k < chunks; ++k) {
            const std::int32_t t = count_of(k);
            const Part part{.sequence = &sequence, .column = 0, .count = t};
            stage_inputs(std::span(&part, 1), tokens.subspan(k * c, std::size_t(t)), false);
            if (reads_rows) {
                std::copy(row_ids.begin(), row_ids.end(),
                          span_row_ids.begin() + std::ptrdiff_t(k * c * heads));
            }
            Tensor mixed(rank.mixed, DType::BF16, {h, t});
            ops::embedding(Tensor(rank.ids, DType::I32, {t}), embedding_table, mixed, st);
            Tensor stack(chunk_stack(k), DType::FP32, {h, hc, t});
            ops::hyper_connection_expand(mixed, stack, st);
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int32_t*>(span_positions.p) + k * c,
                                       rank.positions, std::size_t(t) * 4,
                                       cudaMemcpyDeviceToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int32_t*>(span_rope.p) + k * c, rank.rope,
                                       std::size_t(t) * 4, cudaMemcpyDeviceToDevice, st));
            sequence.position += std::uint32_t(t);
        }
        sequence.position = start;
        if (reads_rows) {
            table->submit(span_row_ids,
                          std::span(static_cast<std::uint8_t*>(span_host_rows->data()),
                                    n * span_row_bytes));
            rows_waited = false;
        }
        // The next step's rows are read while the device runs this span.
        hint_next(sequence, next);
        void* const own_mixed   = rank.mixed;
        float* const own_inject = rank.inject;
        // Chunk k's MoE input and injection, kept between a layer's two phases.
        const auto chunk_planes = [&](std::size_t k) {
            rank.mixed  = static_cast<std::byte*>(span_mixed.p) + k * c * std::size_t(h) * 2;
            rank.inject = static_cast<float*>(span_inject.p) + k * c * std::size_t(hc);
        };
        try {
            // Each layer runs in two phases: attention over every chunk of the span, then the
            // experts over every chunk. The next layer's experts are copied into the pool while its
            // attention phase runs, so its first expert call rarely waits for them.
            for (std::size_t i = 0; i < layers.size(); ++i) {
                const LayerPlan& plan = layers[i];
                LayerState& state     = sequence.layers[i];
                if (plan.ple && !rows_waited) {
                    await_rows(rank);
                    CUDA_CHECK(cudaMemcpyAsync(span_rows.p, span_host_rows->data(),
                                               n * span_row_bytes, cudaMemcpyHostToDevice, st));
                }
                for (std::size_t k = 0; k < chunks; ++k) {
                    const std::int32_t t = count_of(k);
                    const Part part{.sequence = &sequence, .column = 0, .count = t};
                    CUDA_CHECK(cudaMemcpyAsync(rank.positions,
                                               static_cast<std::int32_t*>(span_positions.p) + k * c,
                                               std::size_t(t) * 4, cudaMemcpyDeviceToDevice, st));
                    CUDA_CHECK(cudaMemcpyAsync(rank.rope,
                                               static_cast<std::int32_t*>(span_rope.p) + k * c,
                                               std::size_t(t) * 4, cudaMemcpyDeviceToDevice, st));
                    if (plan.ple) {
                        CUDA_CHECK(cudaMemcpyAsync(rank.rows,
                                                   static_cast<std::byte*>(span_rows.p) +
                                                       k * c * span_row_bytes,
                                                   std::size_t(t) * span_row_bytes,
                                                   cudaMemcpyDeviceToDevice, st));
                    }
                    rank.stack = chunk_stack(k);
                    chunk_planes(k);
                    Tensor stack(rank.stack, DType::FP32, {h, hc, t});
                    Tensor mixed(rank.mixed, DType::BF16, {h, t});
                    Tensor inject(rank.inject, DType::FP32, {hc, t});
                    if (plan.ple) { run_ple(plan, i, state, rank, part); }
                    {
                        auto scope = rank.workspace->scope();
                        ops::hyper_connection_read(stack, plan.attn_hc.weights(),
                                                   config.rms_norm_eps, *rank.workspace, mixed,
                                                   &inject, st);
                    }
                    if (plan.gdn) {
                        run_gdn(plan, i, state, rank, part);
                    } else {
                        // The span holds the sequence at its start; chunk k begins k chunks on.
                        AttentionInputs in = target_inputs(i, state, rank, part);
                        in.first           = start + std::uint32_t(k * c);
                        run_qsa(plan, i, state, rank, part, in);
                    }
                    {
                        auto scope = rank.workspace->scope();
                        ops::hyper_connection_write_read(
                            stack, Tensor(rank.y, DType::BF16, {h, t}), inject,
                            plan.mlp_hc.weights(), config.rms_norm_eps, *rank.workspace, mixed,
                            &inject, st);
                    }
                }
                for (std::size_t k = 0; k < chunks;) {
                    const std::size_t last = group_end(k, chunks, n, c);
                    const auto t = static_cast<std::int32_t>(std::min(last * c, n) - k * c);
                    rank.stack = chunk_stack(k);
                    chunk_planes(k);
                    span_hold = last < chunks;
                    Tensor stack(rank.stack, DType::FP32, {h, hc, t});
                    Tensor inject(rank.inject, DType::FP32, {hc, t});
                    if (last == k + 1) {
                        run_moe(plan, rank, t, i);
                        ops::hyper_connection_write(stack, Tensor(rank.y, DType::FP32, {h, t}),
                                                    inject, st);
                    } else {
                        run_group(plan, rank, i, k, last, n, c);
                        ops::hyper_connection_write(stack, Tensor(span_y.p, DType::FP32, {h, t}),
                                                    inject, st);
                        if (last == chunks) {
                            // record_routes() reads the last chunk's routes from the layer's
                            // record, where run_moe() leaves them.
                            const auto top = std::size_t(config.num_experts_per_tok);
                            CUDA_CHECK(cudaMemcpyAsync(
                                route_records[i],
                                static_cast<std::int32_t*>(span_ids.p) + (chunks - 1 - k) * c * top,
                                top * std::size_t(count_of(chunks - 1)) * 4,
                                cudaMemcpyDeviceToDevice, st));
                        }
                    }
                    k = last;
                }
            }
            span_hold   = false;
            rank.mixed  = own_mixed;
            rank.inject = own_inject;
            // The head reads the last chunk's last rows.
            rank.stack       = chunk_stack(chunks - 1);
            const auto rows  = static_cast<std::int32_t>(logit_rows);
            const Tensor last(rank.stack + width * std::size_t(std::int32_t(last_count) - rows),
                              DType::FP32, {h, hc, rows});
            Tensor head_mixed(rank.mixed, DType::BF16, {h, rows});
            {
                auto scope = rank.workspace->scope();
                ops::hyper_connection_read(last, final_mixer.weights(), config.rms_norm_eps,
                                           *rank.workspace, head_mixed, nullptr, st);
            }
            Tensor logits(rank.logits, DType::BF16,
                          {static_cast<std::int32_t>(config.vocab_size), rows});
            project(head, head_mixed, logits, *rank.workspace, st);
            sequence.position = start + std::uint32_t(n);
            record_routes(last_count);
            // The MTP block catches up chunk by chunk over each chunk's own final stacks.
            if (mtp) {
                for (std::size_t k = 0; k < chunks; ++k) {
                    rank.stack = chunk_stack(k);
                    const auto t = count_of(k);
                    const Committed committed{&sequence, 0, t, start + std::uint32_t(k * c),
                                              tokens.subspan(k * c, std::size_t(t))};
                    mtp_catch_up(std::span(&committed, 1));
                }
                // The next pass's MTP cells start from the last chunk's stack, which it copied.
            }
        } catch (...) {
            rank.stack  = own_stack;
            rank.mixed  = own_mixed;
            rank.inject = own_inject;
            span_hold   = false;
            throw;
        }
        rank.stack = own_stack;
    }

    // Tokens one prompt call may take: a span with spans on, else a chunk.
    [[nodiscard]] std::uint32_t prompt_step() const noexcept {
        return span_tokens != 0 ? span_tokens : options.prefill_chunk;
    }

    void forward(std::uint32_t s, std::span<const std::int32_t> tokens, std::uint32_t logit_rows,
                 std::span<const std::int32_t> next) {
        if (next.size() > prompt_step()) {
            throw std::invalid_argument("qwen4_exp forward: more next tokens than a prompt step");
        }
        if (tokens.size() > options.prefill_chunk) {
            drain();
            forward_span(s, tokens, logit_rows, next);
            return;
        }
        check_tokens(tokens, logit_rows);
        SequenceState& sequence      = sequences.at(s);
        const std::uint32_t position = sequence.position;
        const Part part{
            .sequence = &sequence, .column = 0, .count = static_cast<std::int32_t>(tokens.size())};
        if (threadable(std::span(&part, 1), part.count)) {
            threaded_pass(std::span(&part, 1), tokens, logit_rows);
            hint_next(sequence, next);
            return;
        }
        drain();
        pass(std::span(&part, 1), tokens, logit_rows);
        hint_next(sequence, next);
        if (mtp) {
            const Committed committed{&sequence, 0, part.count, position, tokens};
            mtp_catch_up(std::span(&committed, 1));
        }
    }

    void decode(std::span<const std::uint32_t> batch, std::span<const std::int32_t> tokens,
                Executor::AfterHead after_head) {
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
        if (threadable(parts, static_cast<std::int32_t>(tokens.size()))) {
            threaded_pass(parts, tokens, static_cast<std::uint32_t>(tokens.size()),
                          std::move(after_head));
            return;
        }
        drain();
        std::vector<Committed> committed;
        for (const Part& part : parts) {
            committed.push_back({part.sequence, part.column, 1, part.sequence->position,
                                 tokens.subspan(std::size_t(part.column), 1)});
        }
        pass(parts, tokens, static_cast<std::uint32_t>(tokens.size()));
        if (mtp) { mtp_catch_up(committed); }
        if (after_head) {
            RankBinding bind(device, model.head_rank());
            after_head(ranks[model.head_rank()].stream, head_logits(tokens.size()));
        }
    }

    [[nodiscard]] Tensor head_logits(std::size_t rows) const {
        return Tensor(ranks[model.head_rank()].logits, DType::BF16,
                      {static_cast<std::int32_t>(config.vocab_size), static_cast<std::int32_t>(rows)});
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
        const std::uint32_t sequence = 0;
        if (sequences.front().position + 1 < options.max_context) {
            decode(std::span(&sequence, 1), std::span(tokens).first(1), {});
        }
        if (mtp && can_draft(0) &&
            sequences.front().position + verify_width <= options.max_context) {
            std::vector<std::int32_t> drafts(options.draft_tokens);
            draft(std::span(&sequence, 1), std::span(tokens).first(1), drafts);
            std::vector<std::int32_t> round{kToken};
            round.insert(round.end(), drafts.begin(), drafts.end());
            verify(std::span(&sequence, 1), round);
            const std::uint32_t kept = 1;
            commit(std::span(&sequence, 1), std::span(&kept, 1));
        }
        reset(0);
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            CUDA_CHECK(cudaStreamSynchronize(rank.stream));
        }
        warming = false;
    }

    void grow_expert_cache(std::uint64_t keep) {
        if (!cache) { return; }
        std::vector<std::uint64_t> bytes(device.size(), 0);
        for (std::size_t r = 0; r < device.size(); ++r) {
            RankBinding bind(device, r);
            CUDA_CHECK(cudaDeviceSynchronize());
            std::size_t free_bytes = 0, total = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total));
            bytes[r] = free_bytes > keep ? free_bytes - keep : 0;
        }
        const auto added = cache->grow(bytes);
        for (std::size_t r = 0; r < added.size(); ++r) {
            memory.ranks[r].expert_cache_bytes += added[r];
            memory.expert_cache_bytes += added[r];
        }
        std::vector<ExpertCache::Loan> loans;
        for (std::size_t r = 0; r < slot_pools.size(); ++r) {
            for (const auto& buffer : slot_pools[r].buffers) {
                loans.push_back({.rank = r, .base = buffer.storage.p, .bytes = buffer.storage.bytes});
            }
        }
        if (!loans.empty()) {
            const auto lent = cache->lend(loans);
            for (std::size_t r = 0; r < lent.size(); ++r) {
                memory.ranks[r].expert_cache_bytes += lent[r];
                memory.ranks[r].workspace_bytes -= lent[r];
                memory.expert_cache_bytes += lent[r];
                memory.workspace_bytes -= lent[r];
            }
            pools_lent = true;
        }
        if (misses) {
            for (std::size_t i = 0; i < expert_layers().size(); ++i) {
                misses->set_resident(i, cache->storage(i));
            }
        }
    }

    // ---- MTP speculative decoding ------------------------------------------------------------

    void require_mtp(const char* what) const {
        if (!mtp) {
            throw std::logic_error(std::string("qwen4_exp ") + what +
                                   ": the executor has no drafts");
        }
    }

    [[nodiscard]] bool can_draft(std::uint32_t s) const {
        const SequenceState& sequence = sequences.at(s);
        return mtp && sequence.mtp_follows && sequence.position > 0;
    }

    // Uploads an MTP pass's token ids and cell positions to the head rank's chain buffers, at the
    // given offsets.
    void stage_mtp(std::span<const std::int32_t> ids, std::size_t ids_at,
                   std::span<const std::int32_t> positions, std::size_t positions_at) {
        RankState& rank = ranks[model.head_rank()];
        CUDA_CHECK(cudaEventSynchronize(mtp_staged));
        auto* host                 = static_cast<std::int32_t*>(mtp_staging->data());
        const std::size_t capacity = mtp_ids.bytes / 4;
        std::memcpy(host, ids.data(), ids.size() * 4);
        std::memcpy(host + capacity, positions.data(), positions.size() * 4);
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int32_t*>(mtp_ids.p) + ids_at, host,
                                   ids.size() * 4, cudaMemcpyHostToDevice, rank.stream));
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int32_t*>(mtp_positions.p) + positions_at,
                                   host + capacity, positions.size() * 4, cudaMemcpyHostToDevice,
                                   rank.stream));
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
    // ids at `ids` (device I32 [n], head rank). The table lives on rank 0, so a split model looks
    // the rows up there and copies them over.
    void embed_cells(const std::int32_t* ids, std::int32_t n, void* out) {
        const std::size_t head = model.head_rank();
        const auto h           = static_cast<std::int32_t>(config.hidden_size);
        RankState& last        = ranks[head];
        if (head == 0 || head_embedding.payload != nullptr) {
            Tensor rows(out, DType::BF16, {h, n});
            ops::embedding(Tensor(const_cast<std::int32_t*>(ids), DType::I32, {n}),
                           head == 0 ? embedding_table : head_embedding, rows, last.stream);
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
            Tensor rows(first.mixed, DType::BF16, {h, n});
            ops::embedding(Tensor(first.ids, DType::I32, {n}), embedding_table, rows, first.stream);
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
    // plane. Drafting cells run the indexer on the sequences' scratch tails.
    void mtp_pass(std::span<const MtpPart> parts, std::int32_t n, const std::int32_t* ids,
                  const std::int32_t* positions, bool drafting) {
        RankState& rank      = ranks[model.head_rank()];
        const cudaStream_t s = rank.stream;
        WorkspaceArena& ws   = *rank.workspace;
        const auto h         = static_cast<std::int32_t>(config.hidden_size);
        const auto hc        = static_cast<std::int32_t>(config.hc_count);
        const float eps      = config.rms_norm_eps;
        if (catch_up_host != nullptr) {
            // A rank-thread catch-up: the cells' embeddings are the pass's own, which rank 0 left
            // in pinned host memory before it handed the stack on.
            for (const CatchUpRows& rows : catch_up_rows) {
                CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(rank.mixed) +
                                               std::size_t(rows.cell) * h * 2,
                                           catch_up_host + std::size_t(rows.column) * h * 2,
                                           std::size_t(rows.count) * h * 2,
                                           cudaMemcpyHostToDevice, s));
            }
        } else {
            embed_cells(ids, n, rank.mixed);
        }
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
        // MTP cells rotate on the one text axis.
        const bool media = std::exchange(mrope, false);
        for (const MtpPart& part : parts) {
            SequenceState& sequence = *part.sequence;
            const Part cells{.sequence = &sequence, .column = part.column, .count = part.count};
            run_qsa(plan, layers.size(), sequence.mtp, rank, cells,
                    {.positions = positions + part.column,
                     .rope      = positions + part.column,
                     .index     = nullptr,
                     .tail      = drafting ? sequence.mtp_scratch_tail.p : sequence.mtp.tail.p,
                     .first     = part.first});
        }
        mrope = media;
        {
            auto scope = ws.scope();
            ops::hyper_connection_write_read(stack, Tensor(rank.y, DType::BF16, {h, n}), inject,
                                             plan.mlp_hc.weights(), eps, ws, mixed, &inject, s);
        }
        run_moe(plan, rank, n, layers.size());
        ops::hyper_connection_write(stack, Tensor(rank.y, DType::FP32, {h, n}), inject, s);
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

    // The greedy draft of each of the `n` single-cell parts of the last MTP pass: the block's
    // final mixer, the head, and the argmax over the public tokens into `out` (device I32 [n],
    // head rank).
    void mtp_head(std::int32_t n, std::int32_t* out, std::int32_t* second = nullptr,
                  float* logprobs = nullptr) {
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
        project(mtp->head, mixed, logits, ws, s);
        Tensor tokens(out, DType::I32, {n});
        if (second) {
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

    // Advances the MTP block over what a target pass committed: for each sequence the cells from
    // its last position before the pass (whose stack it kept) to the second to last committed
    // position, each pairing that position's stack with the next token, and keeps the last
    // committed position's stack for the next cell. The target's stacks are still in the head
    // rank's stack plane.
    // `embedded` (rank threads): the pass's token embeddings in pinned host memory, one column per
    // pass column, which the cells take instead of a round trip to rank 0.
    void mtp_catch_up(std::span<const Committed> committed, const std::byte* embedded = nullptr) {
        RankBinding bind(device, model.head_rank());
        RankState& rank  = ranks[model.head_rank()];
        const auto width = std::size_t(config.hc_count) * config.hidden_size;
        std::vector<MtpPart> parts;
        std::vector<std::int32_t> ids, positions;
        std::int32_t n = 0;
        catch_up_rows.clear();
        for (const Committed& c : committed) {
            SequenceState& sequence = *c.sequence;
            if (!sequence.mtp_follows || c.count == 0) { continue; }
            // A sequence's first position has no stack before it.
            const bool kept           = c.position > 0;
            const std::int32_t cells  = c.count - (kept ? 0 : 1);
            const std::uint32_t first = kept ? c.position - 1 : 0;
            if (cells == 0) { continue; }
            catch_up_rows.push_back(
                {.cell = n, .column = c.column + (kept ? 0 : 1), .count = cells});
            if (kept) { stage_hidden(sequence.mtp_pending.p, 1, n); }
            if (c.count > 1) {
                stage_hidden(rank.stack + width * std::size_t(c.column), c.count - 1,
                             n + (kept ? 1 : 0));
            }
            parts.push_back({&sequence, n, cells, first});
            ids.insert(ids.end(), c.tokens.begin() + (kept ? 0 : 1), c.tokens.end());
            for (std::int32_t q = 0; q < cells; ++q) {
                positions.push_back(static_cast<std::int32_t>(first) + q);
            }
            n += cells;
        }
        // The stacks the next cells start from, once the casts above have read the old ones.
        for (const Committed& c : committed) {
            if (!c.sequence->mtp_follows || c.count == 0) { continue; }
            CUDA_CHECK(cudaMemcpyAsync(c.sequence->mtp_pending.p,
                                       rank.stack + width * std::size_t(c.column + c.count - 1),
                                       width * 4, cudaMemcpyDeviceToDevice, rank.stream));
        }
        if (n == 0) { return; }
        if (hybrid) { hybrid->begin(cancelled, !warming); }
        stage_mtp(ids, 0, positions, 0);
        if (misses) { misses->keep_alive(); }
        catch_up_host = embedded;
        try {
            mtp_pass(parts, n, static_cast<const std::int32_t*>(mtp_ids.p),
                     static_cast<const std::int32_t*>(mtp_positions.p), false);
        } catch (...) {
            catch_up_host = nullptr;
            throw;
        }
        catch_up_host = nullptr;
        if (hybrid) { hybrid->finish(); }
    }

    // The rows of a rank-thread catch-up's cells in its pass's host embeddings.
    struct CatchUpRows {
        std::int32_t cell = 0, column = 0, count = 0;
    };
    std::vector<CatchUpRows> catch_up_rows;
    const std::byte* catch_up_host = nullptr;

    void draft(std::span<const std::uint32_t> batch, std::span<const std::int32_t> anchors,
               std::span<std::int32_t> out, std::span<std::uint32_t> extents = {},
               std::uint32_t steps = 0) {
        require_mtp("draft");
        const std::size_t b      = batch.size();
        const std::size_t stride = options.draft_tokens;
        const std::size_t k      = steps == 0 ? stride : steps;
        if (b == 0 || b > sequences.size() || anchors.size() != b || out.size() != b * stride ||
            (!extents.empty() && extents.size() != b) || k > stride) {
            throw std::invalid_argument("qwen4_exp draft: one anchor and its drafts per sequence");
        }
        std::vector<SequenceState*> drafting;
        for (std::size_t j = 0; j < b; ++j) {
            SequenceState* sequence = &sequences.at(batch[j]);
            if (std::find(drafting.begin(), drafting.end(), sequence) != drafting.end()) {
                throw std::invalid_argument("qwen4_exp draft: a sequence appears twice");
            }
            if (!can_draft(batch[j])) {
                throw std::invalid_argument("qwen4_exp draft: the sequence's MTP state does not "
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
        drain_head();
        RankBinding bind(device, model.head_rank());
        RankState& rank = ranks[model.head_rank()];
        // Step i's cell of a sequence sits at its position - 1 + i and pairs the stack before it
        // (the kept one at step 0) with the step's token (the anchor at step 0). Pipeline windows
        // stage the positions of the guess chain's steps after them too.
        const std::size_t staged = pipeline && b == 1 ? 2 * k + 1 : k;
        std::vector<std::int32_t> positions(staged * b);
        std::vector<MtpPart> parts;
        for (std::size_t j = 0; j < b; ++j) {
            const std::uint32_t first = drafting[j]->position - 1;
            for (std::size_t i = 0; i < staged; ++i) {
                positions[i * b + j] = static_cast<std::int32_t>(first + i);
            }
            parts.push_back({drafting[j], std::int32_t(j), 1, first});
        }
        draft_sequence = drafting.front();
        draft_first = drafting.front()->position - 1;
        draft_steps = static_cast<std::uint32_t>(k);
        // The MTP layer writes its cells' KV at the steps' positions.
        for (const SequenceState* sequence : drafting) {
            require_pages(*sequence, std::uint64_t(sequence->position) + staged);
        }
        if (hybrid) { hybrid->begin(cancelled, !warming); }
        stage_mtp(anchors, 0, positions, 0);
        if (draft_prefetch) {
            std::vector<NgramContext> contexts;
            for (const auto* sequence : drafting) { contexts.push_back(sequence->context); }
            draft_prefetch->begin(std::move(contexts), anchors);
        }
        auto* ids        = static_cast<std::int32_t*>(mtp_ids.p);
        const auto* at   = static_cast<const std::int32_t*>(mtp_positions.p);
        const auto n     = static_cast<std::int32_t>(b);
        const auto chain = [&] {
            for (std::size_t j = 0; j < b; ++j) {
                CUDA_CHECK(cudaMemcpyAsync(drafting[j]->mtp_scratch_tail.p, drafting[j]->mtp.tail.p,
                                           drafting[j]->mtp.tail.bytes, cudaMemcpyDeviceToDevice,
                                           rank.stream));
                stage_hidden(drafting[j]->mtp_pending.p, 1, std::int32_t(j));
            }
            for (std::size_t i = 0; i < k; ++i) {
                if (i > 0) { stage_hidden(rank.stack, n, 0); }
                mtp_pass(parts, n, ids + i * b, at + i * b, true);
                auto* second = draft_prefetch ?
                    static_cast<std::int32_t*>(ngram_prefetch_tokens.p) + i * b : nullptr;
                auto* logprobs = mtp_logprobs.p ? static_cast<float*>(mtp_logprobs.p) + i * b : nullptr;
                mtp_head(n, ids + (i + 1) * b, second, logprobs);
                if (draft_prefetch) {
                    draft_prefetch->enqueue(i, ids + (i + 1) * b, second, rank.stream);
                }
            }
            if (draft_prefetch) { draft_prefetch->join(rank.stream); }
        };
        // Tokens and positions stay in the staged buffers. Native hybrid experts use the same
        // captured exchange as text decode; GGUF host experts replay too, without counting the
        // draft steps' routes.
        auto& only       = drafting.front()->draft.at(k);
        // A split model's chain stays on the head rank once its embeddings come from the host copy.
        const bool graph = options.cuda_graphs && b == 1 &&
                           (device.size() == 1 || head_embedding.payload != nullptr) && !stream;
        if (misses) { misses->keep_alive(); }
        if (graph && !only.chain.ready() && only.runs++ > 0) {
            DecodeGraphDefinition definition;
            capturing_draft = true;
            try {
                definition.capture(rank.stream, chain);
            } catch (...) {
                capturing_draft = false;
                throw;
            }
            capturing_draft = false;
            only.chain.instantiate(definition);
        }
        if (graph && only.chain.ready()) {
            only.chain.launch(rank.stream);
        } else {
            chain();
        }
        // The expert cache takes the last verification's routes while the device drafts; the
        // table uploads it queues follow the draft on the stream, before the next pass.
        settle_routes();
        auto* host = static_cast<std::int32_t*>(mtp_staging->data());
        CUDA_CHECK(cudaMemcpyAsync(host, ids + b, k * b * 4, cudaMemcpyDeviceToHost, rank.stream));
        auto* logprobs = reinterpret_cast<float*>(host + mtp_ids.bytes / 4);
        if (mtp_logprobs.p) {
            CUDA_CHECK(cudaMemcpyAsync(logprobs, mtp_logprobs.p, k * b * sizeof(float),
                                       cudaMemcpyDeviceToHost, rank.stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(rank.stream));
        if (hybrid) { hybrid->finish(); }
        if (draft_prefetch) { draft_prefetch->check(); }
        for (std::size_t j = 0; j < b; ++j) {
            for (std::size_t i = 0; i < k; ++i) { out[j * stride + i] = host[i * b + j]; }
            std::uint32_t extent = static_cast<std::uint32_t>(k);
            if (mtp_logprobs.p) {
                const float floor = std::log(options.draft_min_p);
                for (std::size_t i = 0; i < k; ++i) {
                    if (!std::isfinite(logprobs[i * b + j])) {
                        throw std::runtime_error("qwen4_exp: non-finite MTP draft confidence");
                    }
                    if (logprobs[i * b + j] <= floor && extent == k) {
                        extent = static_cast<std::uint32_t>(i + 1);
                    }
                }
            }
            if (!extents.empty()) { extents[j] = extent; }
        }
    }

    void verify(std::span<const std::uint32_t> batch, std::span<const std::int32_t> tokens) {
        require_mtp("verify");
        const std::size_t b = batch.size();
        const std::size_t w = b == 0 ? 0 : tokens.size() / b;
        if (b == 0 || b > sequences.size() || w < 2 || w > verify_width || tokens.size() != b * w) {
            throw std::invalid_argument("qwen4_exp verify: an anchor and its drafts per sequence");
        }
        check_tokens(tokens, static_cast<std::uint32_t>(tokens.size()));
        active_verify_width = static_cast<std::uint32_t>(w);
        std::vector<Part> parts;
        verified.clear();
        for (std::size_t j = 0; j < b; ++j) {
            SequenceState* sequence = &sequences.at(batch[j]);
            if (std::find(verified.begin(), verified.end(), sequence) != verified.end()) {
                verified.clear();
                throw std::invalid_argument("qwen4_exp verify: a sequence appears twice");
            }
            parts.push_back({.sequence = sequence,
                             .column   = static_cast<std::int32_t>(j * w),
                             .count    = static_cast<std::int32_t>(w)});
            sequence->verify_context = sequence->context;
            sequence->verify_tokens.assign(tokens.begin() + std::ptrdiff_t(j * w),
                                           tokens.begin() + std::ptrdiff_t((j + 1) * w));
            verified.push_back(sequence);
        }
        // The host's pass orders itself after every rank thread's queued work.
        if (!pipeline) { drain(); }
        verifying = true;
        verified_set = pipeline ? next_set() : 0;
        pass_mode    = {.verify = true,
                        .ahead  = false,
                        .set    = verified_set,
                        .rows   = static_cast<std::int32_t>(b),
                        .width  = static_cast<std::uint32_t>(w)};
        if (pipeline) {
            try {
                threaded_verify(parts, tokens);
            } catch (...) {
                pass_mode = {};
                verifying = false;
                verified.clear();
                throw;
            }
            pass_mode = {};
            verifying = false;
            return;
        }
        try {
            pass(parts, tokens, static_cast<std::uint32_t>(tokens.size()));
            if (draft_prefetch) {
                RankBinding bind(device, model.head_rank());
                RankState& rank = ranks[model.head_rank()];
                std::vector<NgramContext> contexts;
                for (const auto* sequence : verified) { contexts.push_back(sequence->context); }
                draft_prefetch->begin_verification(std::move(contexts), tokens);
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
            pass_mode = {};
            verifying = false;
            verified.clear();
            throw;
        }
        pass_mode = {};
        verifying = false;
    }

    // One rank's part of a commit, on its stream: its sparse-attention layers' indexer tails and
    // its PLE histories advance over the kept positions' records of `set` (a window of width `w`),
    // and its Gated DeltaNet layers replay the kept transitions. `shadow` (pipeline windows): the
    // window is taken whole into the shadow states, which start from the sequence's own.
    void commit_rank(std::size_t r, std::span<SequenceState* const> rows_of,
                     std::span<const std::uint32_t> kept, std::uint32_t set, std::uint32_t w,
                     bool shadow) {
        RankState& rank = ranks[r];
        const auto di   = static_cast<std::int32_t>(config.indexer_head_dim);
        const std::uint64_t index_rows =
            std::uint64_t(config.indexer_n_heads + 1) * config.indexer_head_dim;
        const std::uint64_t columns = std::uint64_t(verify_width) * options.sequences;
        const auto stack_width      = std::int32_t(config.hc_count * config.hidden_size);
        // The window's positions: still staged, unless a later pass has been staged since.
        const auto* positions = pipeline ? static_cast<const std::int32_t*>(rank.window_positions[set].p)
                                         : rank.positions;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const LayerPlan& plan = layers[i];
            if (plan.rank != r) { continue; }
            for (std::size_t j = 0; j < rows_of.size(); ++j) {
                LayerState& state  = rows_of[j]->layers[i];
                LayerState& target = shadow ? ahead_layers[i] : state;
                const auto m       = static_cast<std::int32_t>(kept[j]);
                if (plan.qsa) {
                    // The indexer's tail advances over the kept positions' recorded projections.
                    if (shadow) {
                        CUDA_CHECK(cudaMemcpyAsync(target.tail.p, state.tail.p, state.tail.bytes,
                                                   cudaMemcpyDeviceToDevice, rank.stream));
                    }
                    const QsaPlan& a = *plan.qsa;
                    const Tensor projection(static_cast<std::byte*>(rank.index_records[set].p) +
                                                (qsa_local[i] * columns + j * w) * index_rows * 2,
                                            DType::BF16, {a.index.rows, m});
                    const Tensor first(const_cast<std::int32_t*>(positions) + j * w, DType::I32, {1});
                    Tensor pooled(state.pooled.p, DType::FP32,
                                  {di, static_cast<std::int32_t>(pooled_slots)});
                    Tensor tail(target.tail.p, DType::FP32,
                                {di, static_cast<std::int32_t>(config.indexer_compress_ratio - 1)});
                    const ops::QsaIndexerWeights iw{&a.index_query_norm, &a.index_key_norm};
                    ops::qsa_indexer_append(projection, first, iw, config.rms_norm_eps, pooled,
                                            tail, rank.stream);
                }
                if (plan.ple) {
                    // The PLE history advances over the kept positions' recorded input.
                    if (shadow) {
                        CUDA_CHECK(cudaMemcpyAsync(target.history.p, state.history.p,
                                                   state.history.bytes, cudaMemcpyDeviceToDevice,
                                                   rank.stream));
                    }
                    Tensor history(
                        target.history.p, DType::FP32,
                        {stack_width, static_cast<std::int32_t>((config.ple_conv_kernel_size - 1) *
                                                                config.ngram.ngram_size)});
                    const Tensor normalized(static_cast<float*>(rank.ple_records[set].p) +
                                                std::size_t(stack_width) * j * w,
                                            DType::FP32, {stack_width, m});
                    ops::ple_history_advance(history, normalized, rank.stream);
                }
            }
        }
        // The Gated DeltaNet layers replay the kept transitions into each sequence's slot.
        if (!rank.fold[set]) { return; }
        std::vector<ops::GdnReplayFoldRow> rows;
        for (std::size_t j = 0; j < rows_of.size(); ++j) {
            const auto slot = static_cast<std::int32_t>(rows_of[j]->slot);
            const auto to   = shadow ? static_cast<std::int32_t>(options.sequences) : slot;
            rows.push_back({slot, to, static_cast<std::int32_t>(kept[j])});
        }
        rank.fold[set]->execute(rows, static_cast<std::int32_t>(w), rank.stream);
    }

    void commit(std::span<const std::uint32_t> batch, std::span<const std::uint32_t> kept) {
        require_mtp("commit");
        const std::uint32_t w = active_verify_width;
        if (batch.size() != verified.size() || kept.size() != batch.size()) {
            throw std::invalid_argument("qwen4_exp commit: the last verification's sequences");
        }
        for (std::size_t j = 0; j < batch.size(); ++j) {
            if (&sequences.at(batch[j]) != verified[j] || kept[j] == 0 || kept[j] > w) {
                throw std::invalid_argument("qwen4_exp commit: the last verification's sequences, "
                                            "1 to draft_tokens + 1 tokens each");
            }
        }
        if (pipeline) {
            pipelined_commit(kept);
            return;
        }
        drain();
        if (draft_prefetch) {
            RankBinding bind(device, model.head_rank());
            // Captured events encode graph dependencies; wait on the joined stream before
            // reading host callback state. Host synchronization of a captured event is invalid.
            CUDA_CHECK(cudaStreamSynchronize(ranks[model.head_rank()].stream));
            draft_prefetch->check();
        }
        for (std::size_t r = 0; r < ranks.size(); ++r) {
            RankBinding bind(device, r);
            commit_rank(r, verified, kept, verified_set, w, false);
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
            committed.push_back({&sequence, static_cast<std::int32_t>(j * w),
                                 static_cast<std::int32_t>(kept[j]), sequence.position, tokens});
            sequence.position += kept[j];
        }
        verified.clear();
        mtp_catch_up(committed);
    }

    // ---- Pipeline windows ----
    // A round's verification goes to the rank threads. Before it, the head rank's thread runs the
    // MTP chain on past the round's drafts to the window that would follow if this one is kept
    // whole: its bonus token and drafts after it (the guess). The host posts that window to the
    // front ranks at once; they run it on shadow states (theirs advanced over the window in flight,
    // taken whole) while the head rank verifies. When the verdict keeps the window whole and its
    // bonus is the guessed one, the head rank completes the guessed window from the front ranks'
    // stack; otherwise the guess is dropped and the next round drafts as before. Every token is
    // still the verification's, so the output is the serial loop's.
    std::size_t head_first = 0;      // the first segment of the head rank
    std::uint32_t set_flip = 0;      // the record set of the last window
    struct Window {                  // the last window posted to the front ranks
        SequenceState* sequence = nullptr;
        std::uint32_t set = 0, width = 0, position = 0;
        std::vector<std::int32_t> tokens;
        NgramContext context;        // the n-gram context before it
    } window;
    struct Ahead {                   // the guessed window the front ranks run
        SequenceState* sequence = nullptr;
        bool live            = false;
        std::uint64_t number = 0;
        std::uint32_t set = 0, width = 0, position = 0;
        std::vector<std::int32_t> tokens;
    } ahead;
    std::unique_ptr<PinnedHostBuffer> guess_host;
    cudaEvent_t guess_event      = nullptr;
    std::uint64_t guess_posted   = 0; // the host's count of guess chains
    std::uint64_t guess_recorded = 0; // the head thread's (worker_mutex)
    std::uint32_t draft_first = 0, draft_steps = 0; // the last draft chain
    SequenceState* draft_sequence = nullptr;
    static constexpr std::size_t kDiscardCell = 512; // where forced chain steps write their argmax

    void start_pipeline() {
        const std::size_t head = model.head_rank();
        head_first = 0;
        while (head_first < segments.size() && segments[head_first].rank != head) { ++head_first; }
        bool layout = head_first > 0 && head_first < segments.size();
        for (std::size_t i = head_first; layout && i < segments.size(); ++i) {
            layout = segments[i].rank == head;
        }
        if (!layout) {
            pipeline = false;
            return;
        }
        ahead_buffer = std::make_unique<PinnedHostBuffer>(stack_bytes(std::int32_t(verify_width)));
        {
            RankBinding bind(device, segments[head_first - 1].rank);
            CUDA_CHECK(cudaEventCreateWithFlags(&ahead_written, cudaEventDisableTiming));
        }
        RankBinding bind(device, head);
        CUDA_CHECK(cudaEventCreateWithFlags(&ahead_read, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&guess_event, cudaEventDisableTiming));
        guess_host = std::make_unique<PinnedHostBuffer>(64 * sizeof(std::int32_t));
        std::fprintf(stderr, "qwen4_exp: pipeline windows on (%zu front segments)\n", head_first);
    }

    std::uint32_t next_set() {
        set_flip ^= 1U;
        return set_flip;
    }

    // A verification posted to the rank threads (the host's pass_mode describes it).
    void threaded_verify(std::span<const Part> parts, std::span<const std::int32_t> tokens) {
        SequenceState& sequence = *parts.front().sequence;
        const auto w            = static_cast<std::uint32_t>(tokens.size() / parts.size());
        // Each part hashes from a copy of its context: the commit advances the live one.
        std::vector<NgramContext> copies;
        for (const Part& part : parts) { copies.push_back(part.sequence->context); }
        std::vector<NgramContext*> contexts;
        for (NgramContext& copy : copies) { contexts.push_back(&copy); }
        auto task  = stage_task(parts, tokens, static_cast<std::uint32_t>(tokens.size()), contexts,
                                0, segments.size());
        task->mode = pass_mode;
        auto& graphs = sequence.pipelined_verify[pass_mode.set][w];
        if (parts.size() == 1 && graphs.segments.empty() && graphs.runs++ > 0) {
            drain();
            graphs.segments = capture_segments(sequence, std::int32_t(w), 0, segments.size());
        }
        if (parts.size() == 1 && !graphs.segments.empty()) { task->graphs = &graphs.segments; }
        post(task);
        if (parts.size() != 1) { return; }
        window = {.sequence = &sequence,
                  .set      = pass_mode.set,
                  .width    = w,
                  .position = sequence.position,
                  .tokens   = std::vector<std::int32_t>(tokens.begin(), tokens.end()),
                  .context  = sequence.context};
    }

    // The MTP chain on the head rank's thread from step `from` to step `to` of a chain whose cells
    // start at position `first`; steps before `forced` take the staged tokens (their argmax is
    // dropped). The argmax of steps forced - 1 .. to - the guessed bonus and drafts - goes to the
    // host.
    void run_guess_chain(SequenceState* sequence, std::uint32_t first, std::uint32_t from,
                         std::uint32_t to, std::uint32_t forced, std::uint64_t number) {
        RankState& rank = ranks[model.head_rank()];
        auto* ids       = static_cast<std::int32_t*>(mtp_ids.p);
        const auto* at  = static_cast<const std::int32_t*>(mtp_positions.p);
        const MtpPart part{sequence, 0, 1, first};
        for (std::uint32_t i = from; i <= to; ++i) {
            if (i == 0) {
                CUDA_CHECK(cudaMemcpyAsync(sequence->mtp_scratch_tail.p, sequence->mtp.tail.p,
                                           sequence->mtp.tail.bytes, cudaMemcpyDeviceToDevice,
                                           rank.stream));
                stage_hidden(sequence->mtp_pending.p, 1, 0);
            } else {
                stage_hidden(rank.stack, 1, 0);
            }
            mtp_pass(std::span(&part, 1), 1, ids + i, at + i, true);
            mtp_head(1, i + 1 < forced ? ids + kDiscardCell : ids + i + 1);
        }
        const std::uint32_t k = to / 2; // the guess: steps k .. 2k write cells k + 1 .. 2k + 1
        CUDA_CHECK(cudaMemcpyAsync(guess_host->data(), ids + k + 1, (k + 1) * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, rank.stream));
        CUDA_CHECK(cudaEventRecord(guess_event, rank.stream));
        {
            std::lock_guard lock(worker_mutex);
            guess_recorded = number;
        }
        worker_cv.notify_all();
    }

    void post_head_work(std::function<void()> work) {
        auto task  = new_task();
        task->work = [this, work = std::move(work)](std::size_t r) {
            if (r == model.head_rank()) { work(); }
        };
        post(task);
    }

    // After draft(): the head rank's thread runs the chain on from the drafts (before it takes
    // the verification's stack).
    void guess_after_draft() {
        const std::uint64_t number = ++guess_posted;
        SequenceState* sequence    = draft_sequence;
        const std::uint32_t k = draft_steps, first = draft_first;
        post_head_work([=, this] { run_guess_chain(sequence, first, k, 2 * k, 0, number); });
    }

    // The guess of the last chain: the bonus token and the drafts after it (width tokens).
    void take_guess(std::span<std::int32_t> out) {
        {
            std::unique_lock lock(worker_mutex);
            worker_cv.wait(lock, [&] { return guess_recorded >= guess_posted || worker_error; });
            if (worker_error) { std::rethrow_exception(worker_error); }
        }
        RankBinding bind(device, model.head_rank());
        CUDA_CHECK(cudaEventSynchronize(guess_event));
        std::memcpy(out.data(), guess_host->data(), out.size() * sizeof(std::int32_t));
    }

    // The front ranks run `tokens` as the window after the one in flight, taken whole.
    void verify_ahead(std::span<const std::int32_t> tokens) {
        if (window.sequence == nullptr) { throw std::logic_error("qwen4_exp: no window in flight"); }
        SequenceState& sequence = *window.sequence;
        const auto w            = static_cast<std::uint32_t>(tokens.size());
        NgramContext start      = window.context;
        if (table) {
            std::vector<std::uint64_t> scratch(window.tokens.size() * config.ngram_heads());
            ngram_row_ids(ngram, window.tokens, config.eos_token_id, config.vocab_size, start,
                          scratch);
        }
        NgramContext context     = start;
        NgramContext* contexts[] = {&context};
        const std::uint32_t position = window.position + window.width;
        if (position + w > options.max_context) {
            throw std::invalid_argument("qwen4_exp: a guessed window past max_context");
        }
        const Part part{.sequence = &sequence, .column = 0, .count = std::int32_t(w)};
        const std::uint32_t set = next_set();
        auto task = stage_task(std::span(&part, 1), tokens, w, contexts,
                               position - sequence.position, head_first);
        task->mode         = {.verify = true, .ahead = true, .set = set, .rows = 1, .width = w};
        task->to_ahead     = true;
        task->ahead_number = ++ahead_count;
        const std::uint32_t source = window.set, width = window.width;
        SequenceState* rows_of     = &sequence;
        task->work = [this, source, width, rows_of](std::size_t r) {
            if (r == model.head_rank()) { return; }
            SequenceState* const of[] = {rows_of};
            const std::uint32_t kept[] = {width};
            commit_rank(r, of, kept, source, width, true);
        };
        auto& graphs = sequence.pipelined_ahead[set][w];
        if (graphs.segments.empty() && graphs.runs++ > 0) {
            drain();
            const PassMode host = pass_mode;
            pass_mode           = task->mode;
            try {
                graphs.segments = capture_segments(sequence, std::int32_t(w), 0, head_first);
            } catch (...) {
                pass_mode = host;
                throw;
            }
            pass_mode = host;
        }
        if (!graphs.segments.empty()) { task->graphs = &graphs.segments; }
        post(task);
        ahead  = {.sequence = &sequence,
                  .live     = true,
                  .number   = task->ahead_number,
                  .set      = set,
                  .width    = w,
                  .position = position,
                  .tokens   = std::vector<std::int32_t>(tokens.begin(), tokens.end())};
        window = {.sequence = &sequence,
                  .set      = set,
                  .width    = w,
                  .position = position,
                  .tokens   = ahead.tokens,
                  .context  = start};
    }

    [[nodiscard]] bool guess_live() const noexcept { return ahead.live; }

    // The verdict dropped the guessed window.
    void drop_guess() {
        if (!ahead.live) { return; }
        {
            std::lock_guard lock(worker_mutex);
            ahead_resolved = std::max(ahead_resolved, ahead.number);
        }
        worker_cv.notify_all();
        ahead.live = false;
    }

    // The verdict kept the window in flight whole with the guessed bonus: the guessed window is
    // the next verification. The head rank's thread first runs the next guess chain (through the
    // guessed window's tokens, then past them), then completes the window from the front ranks'
    // stack.
    void verify_guessed(std::span<const std::int32_t> tokens) {
        if (ahead.sequence == nullptr) { throw std::logic_error("qwen4_exp: no guessed window"); }
        SequenceState& sequence = *ahead.sequence;
        const auto w            = static_cast<std::uint32_t>(tokens.size());
        if (!ahead.live || w != ahead.width || sequence.position != ahead.position ||
            !std::equal(tokens.begin(), tokens.end(), ahead.tokens.begin())) {
            throw std::logic_error("qwen4_exp: the verification is not the guessed window");
        }
        verified.assign(1, &sequence);
        sequence.verify_tokens.assign(tokens.begin(), tokens.end());
        sequence.verify_context = sequence.context;
        active_verify_width     = w;
        verified_set            = ahead.set;
        const std::uint32_t k     = w - 1;
        const std::uint32_t first = sequence.position - 1;
        // The chain's MTP cells reach position first + 2k.
        require_pages(sequence, std::uint64_t(first) + 2 * k + 1);
        const std::uint64_t number = ++guess_posted;
        std::vector<std::int32_t> forced(tokens.begin(), tokens.end());
        std::vector<std::int32_t> positions(2 * k + 1);
        std::iota(positions.begin(), positions.end(), static_cast<std::int32_t>(first));
        SequenceState* of = &sequence;
        post_head_work([=, this] {
            stage_mtp(forced, 0, positions, 0);
            run_guess_chain(of, first, 0, 2 * k, k + 1, number);
        });
        NgramContext context     = sequence.context;
        NgramContext* contexts[] = {&context};
        const Part part{.sequence = &sequence, .column = 0, .count = std::int32_t(w)};
        auto task = stage_task(std::span(&part, 1), tokens, w, contexts, 0, segments.size(),
                               head_first);
        task->mode = {.verify = true, .ahead = false, .set = ahead.set, .rows = 1, .width = w};
        task->from_ahead   = true;
        task->ahead_number = ahead.number;
        auto& graphs       = sequence.pipelined_verify[ahead.set][w];
        if (!graphs.segments.empty()) { task->graphs = &graphs.segments; }
        post(task);
        ahead.live = false;
    }

    // A commit on the rank threads: each rank's part, and on the head rank the MTP catch-up.
    void pipelined_commit(std::span<const std::uint32_t> kept) {
        const std::uint32_t w = active_verify_width;
        auto task             = new_task();
        std::vector<SequenceState*> rows_of(verified.begin(), verified.end());
        std::vector<std::uint32_t> counts(kept.begin(), kept.end());
        for (SequenceState* sequence : rows_of) {
            task->tokens.insert(task->tokens.end(), sequence->verify_tokens.begin(),
                                sequence->verify_tokens.end());
        }
        std::vector<std::uint64_t> scratch;
        for (std::size_t j = 0; j < rows_of.size(); ++j) {
            SequenceState& sequence = *rows_of[j];
            const std::span<const std::int32_t> tokens(task->tokens.data() + j * w, kept[j]);
            if (table) {
                scratch.resize(tokens.size() * config.ngram_heads());
                ngram_row_ids(ngram, tokens, config.eos_token_id, config.vocab_size,
                              sequence.context, scratch);
            }
            task->committed.push_back({&sequence, static_cast<std::int32_t>(j * w),
                                       static_cast<std::int32_t>(kept[j]), sequence.position,
                                       tokens});
            sequence.position += kept[j];
        }
        const std::uint32_t set = verified_set;
        RankTask* own           = task.get();
        task->work = [this, rows_of, counts, set, w, own](std::size_t r) {
            commit_rank(r, rows_of, counts, set, w, false);
            if (r == model.head_rank()) { mtp_catch_up(own->committed); }
        };
        verified.clear();
        post(task);
    }

    void reset_pipeline() {
        if (!pipeline) { return; }
        {
            std::lock_guard lock(worker_mutex);
            ahead_resolved = ahead_count;
        }
        worker_cv.notify_all();
        ahead.live = false;
    }

    std::uint64_t snapshot_bytes() const {
        const SequenceState& sequence = sequences.at(0);
        std::uint64_t bytes = 0;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            for (const auto& region : recurrent_regions(sequence.layers[i])) {
                if (region.first != nullptr) { bytes += region.second; }
            }
        }
        if (mtp) { bytes += sequence.mtp.tail.bytes + sequence.mtp_pending.bytes; }
        return bytes;
    }

    void snapshot(std::uint32_t s, SequenceSnapshot& out) {
        SequenceState& sequence = sequences.at(s);
        if (out.layers.size() != layers.size()) {
            out.layers.clear();
            for (std::size_t i = 0; i < layers.size(); ++i) {
                RankBinding bind(device, layers[i].rank);
                SequenceSnapshot::Layer layer;
                for (const auto& [data, bytes] : recurrent_regions(sequence.layers[i])) {
                    layer.buffers.push_back(data != nullptr ? DeviceBuffer(bytes) : DeviceBuffer());
                }
                out.layers.push_back(std::move(layer));
            }
        }
        if (mtp && out.mtp.empty()) {
            RankBinding bind(device, model.head_rank());
            out.mtp.emplace_back(sequence.mtp.tail.bytes);
            out.mtp.emplace_back(sequence.mtp_pending.bytes);
        }
        copy_state(sequence, out, true);
        out.position    = sequence.position;
        out.context     = sequence.context;
        out.mtp_follows = mtp && sequence.mtp_follows;
    }

    void restore(std::uint32_t s, const SequenceSnapshot& from) {
        if (from.layers.size() != layers.size()) {
            throw std::invalid_argument("qwen4_exp restore: the snapshot holds no state");
        }
        SequenceState& sequence = sequences.at(s);
        copy_state(sequence, const_cast<SequenceSnapshot&>(from), false);
        sequence.position = from.position;
        sequence.context  = from.context;
        // A snapshot taken before the MTP block followed the sequence leaves it behind.
        sequence.mtp_follows = mtp && !from.mtp.empty() && from.mtp_follows;
        // Snapshots hold text prompts only: a media prompt is not reused.
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
        if (pool_pages != 0) {
            throw std::invalid_argument("qwen4_exp: sequence images need per-sequence KV windows, "
                                        "not a shared KV pool");
        }
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
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const LayerState& state = sequence.layers[i];
            const auto regions      = recurrent_regions(state);
            for (std::size_t k = 0; k < regions.size(); ++k) {
                if (regions[k].first == nullptr) { continue; }
                void* source = at != nullptr ? at->layers[i].buffers[k].p : regions[k].first;
                out.push_back({source, regions[k].second, layers[i].rank});
            }
            if (layers[i].qsa) { attention(state, layers[i].rank); }
        }
        if (mtp) {
            const std::size_t head = model.head_rank();
            attention(sequence.mtp, head);
            out.push_back({at != nullptr ? at->mtp[0].p : sequence.mtp.tail.p,
                           sequence.mtp.tail.bytes, head});
            out.push_back({at != nullptr ? at->mtp[1].p : sequence.mtp_pending.p,
                           sequence.mtp_pending.bytes, head});
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
            CUDA_CHECK(cudaStreamSynchronize(rank.stream));
        }
    }

    SequenceImage save_image(std::uint32_t s, const SequenceSnapshot* at,
                             std::span<std::byte* const> pages, std::size_t page_bytes) {
        const SequenceState& sequence = sequences.at(s);
        if (at != nullptr && (at->layers.size() != layers.size() || (mtp && at->mtp.size() != 2))) {
            throw std::invalid_argument("qwen4_exp image: the snapshot holds no state");
        }
        const std::uint32_t position = at != nullptr ? at->position : sequence.position;
        copy_image(image_regions(sequence, at, position), pages, page_bytes, true);
        return SequenceImage{.position = position,
                             .context  = at != nullptr ? at->context : sequence.context,
                             .mtp_follows =
                                 mtp && (at != nullptr ? at->mtp_follows : sequence.mtp_follows)};
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
        sequence.mtp_follows = mtp && image.mtp_follows;
        sequence.media_columns.clear();
        sequence.media_rope.clear();
        sequence.rope_delta = 0;
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
        for (std::size_t i = 0; i < layers.size(); ++i) {
            RankBinding bind(device, layers[i].rank);
            const auto regions = recurrent_regions(sequence.layers[i]);
            for (std::size_t k = 0; k < regions.size(); ++k) {
                copy(regions[k].first, snapshot.layers[i].buffers[k], ranks[layers[i].rank].stream);
            }
        }
        if (mtp && snapshot.mtp.size() == 2) {
            RankBinding bind(device, model.head_rank());
            const cudaStream_t stream = ranks[model.head_rank()].stream;
            copy(sequence.mtp.tail.p, snapshot.mtp[0], stream);
            copy(sequence.mtp_pending.p, snapshot.mtp[1], stream);
        }
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            CUDA_CHECK(cudaStreamSynchronize(rank.stream));
        }
    }
};

Executor::Executor(const Model& model, DeviceContext& device, ExecutorOptions options)
    : impl_(std::make_unique<Impl>(model, device, std::move(options))) {
    impl_->start_rank_threads();
    // Pipeline windows run on the rank threads.
    if (!impl_->rank_threads) { impl_->pipeline = false; }
}

Executor::~Executor() = default;

const ExecutorOptions& Executor::options() const noexcept { return impl_->options; }

ExecutorMemory Executor::memory() const noexcept { return impl_->memory; }

NgramTableStats Executor::ngram_stats() { return impl_->ngram_stats(); }

std::uint64_t Executor::ngram_resident_bytes() const noexcept {
    return impl_->table ? impl_->table->resident_bytes() : 0;
}

void Executor::reset(std::uint32_t sequence) { impl_->reset(sequence); }

void Executor::warm_up() {
    impl_->drain();
    impl_->warm_up();
    impl_->drain();
}
std::uint32_t Executor::prompt_step() const noexcept { return impl_->prompt_step(); }
void Executor::grow_expert_cache(std::uint64_t keep_free_bytes) {
    impl_->drain();
    impl_->grow_expert_cache(keep_free_bytes);
}

std::uint32_t Executor::position(std::uint32_t sequence) const {
    return impl_->sequences.at(sequence).position;
}

void Executor::forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                       std::uint32_t logit_rows, std::span<const std::int32_t> next) {
    impl_->forward(sequence, tokens, logit_rows, next);
}

void Executor::decode(std::span<const std::uint32_t> sequences,
                      std::span<const std::int32_t> tokens) {
    impl_->decode(sequences, tokens, {});
}

void Executor::decode(std::span<const std::uint32_t> sequences,
                      std::span<const std::int32_t> tokens, AfterHead after_head) {
    impl_->decode(sequences, tokens, std::move(after_head));
}

void Executor::snapshot(std::uint32_t sequence, SequenceSnapshot& out) {
    impl_->drain();
    impl_->snapshot(sequence, out);
}

std::uint64_t Executor::snapshot_bytes() const { return impl_->snapshot_bytes(); }

void Executor::restore(std::uint32_t sequence, const SequenceSnapshot& from) {
    impl_->drain();
    impl_->restore(sequence, from);
}

std::uint64_t Executor::image_bytes(std::uint32_t position) const {
    return impl_->image_bytes(position);
}

SequenceImage Executor::save_image(std::uint32_t sequence, const SequenceSnapshot* at,
                                   std::span<std::byte* const> pages, std::size_t page_bytes) {
    impl_->drain();
    return impl_->save_image(sequence, at, pages, page_bytes);
}

void Executor::load_image(std::uint32_t sequence, const SequenceImage& image,
                          std::span<std::byte* const> pages, std::size_t page_bytes) {
    impl_->drain();
    impl_->load_image(sequence, image, pages, page_bytes);
}

void Executor::set_media(std::uint32_t sequence, std::span<const MediaItem> items,
                         std::vector<std::int32_t> rope_positions, std::int32_t rope_delta) {
    impl_->drain();
    impl_->set_media(sequence, items, std::move(rope_positions), rope_delta);
}

bool Executor::vision() const noexcept { return impl_->vision_context != nullptr; }

std::uint32_t Executor::draft_tokens() const noexcept {
    return impl_->mtp ? impl_->options.draft_tokens : 0;
}

bool Executor::can_draft(std::uint32_t sequence) const { return impl_->can_draft(sequence); }

void Executor::draft(std::span<const std::uint32_t> sequences,
                     std::span<const std::int32_t> anchors, std::span<std::int32_t> out,
                     std::span<std::uint32_t> extents, std::uint32_t steps) {
    impl_->draft(sequences, anchors, out, extents, steps);
}

void Executor::verify(std::span<const std::uint32_t> sequences,
                      std::span<const std::int32_t> tokens) {
    impl_->verify(sequences, tokens);
}

void Executor::commit(std::span<const std::uint32_t> sequences,
                      std::span<const std::uint32_t> columns) {
    impl_->commit(sequences, columns);
}

bool Executor::kv_pooled() const noexcept { return impl_->pool_pages != 0; }

std::uint64_t Executor::kv_free_tokens() const noexcept { return impl_->free_pool_tokens(); }

std::uint64_t Executor::kv_held_tokens(std::uint32_t sequence) const noexcept {
    return impl_->held_pool_tokens(sequence);
}

bool Executor::reserve_kv(std::uint32_t sequence, std::uint32_t tokens) {
    return impl_->reserve_pages(impl_->sequences.at(sequence), tokens);
}

bool Executor::pipeline_windows() const noexcept { return impl_->pipeline; }

void Executor::guess_after_draft() { impl_->guess_after_draft(); }

void Executor::take_guess(std::span<std::int32_t> out) { impl_->take_guess(out); }

void Executor::verify_ahead(std::span<const std::int32_t> tokens) { impl_->verify_ahead(tokens); }

bool Executor::guess_live() const noexcept { return impl_->guess_live(); }

void Executor::drop_guess() { impl_->drop_guess(); }

void Executor::verify_guessed(std::span<const std::int32_t> tokens) {
    impl_->verify_guessed(tokens);
}

Tensor Executor::logits(std::uint32_t rows) const {
    impl_->drain_head_quietly();
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
    impl_->drain_head_quietly();
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
    impl_->drain_quietly();
    if (impl_->table) { impl_->table->wait(); }
    impl_->rows_waited = true;
    impl_->device.synchronize();
    impl_->pending_routes = 0;
    impl_->mtp_routes = 0;
    impl_->verifying = false;
    impl_->verified.clear();
    impl_->reset(sequence);
}

} // namespace ninfer::models::qwen4_exp
