#include "models/flint/executor.h"

#include "core/decode_graph.h"
#include "core/device_buffer.h"
#include "core/device_wait.h"
#include "core/expert_parallel.h"
#include "core/growable_region.h"
#include "core/pinned_memory.h"
#include "core/qtype.h"
#include "core/tensor_parallel.h"
#include "core/vmm.h"
#include "core/weight_view.h"
#include "models/flint/cpu_experts.h"
#include "models/divided_head.h"
#include "models/dspine/runtime.h"
#include "models/flint/kernels.h"
#include "ninfer/ops/linear_policy.h"
#include "ninfer/ops/weight_input.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/ggml_fattn.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::flint {
namespace {

namespace gguf = ops::gguf;

constexpr std::size_t kAlign = 256;

std::size_t round_up(std::size_t value, std::size_t to) { return (value + to - 1) / to * to; }

const float* fp32(const Model& model, WeightId id, std::int32_t count) {
    const Tensor tensor = weight_tensor(model.weight(id).view, {count});
    if (tensor.dtype != DType::FP32) {
        throw std::invalid_argument(model.weight(id).name + " must be stored as fp32");
    }
    return static_cast<const float*>(tensor.data);
}

Weight matrix(const Model& model, std::initializer_list<WeightId> ids) {
    std::vector<ops::WeightInput> inputs;
    for (const WeightId id : ids) { inputs.push_back(model.input(id)); }
    if (inputs.size() == 1) { return ops::prepare_linear_weight(inputs.front()).weight; }
    if (!ops::joins(inputs)) {
        throw std::invalid_argument("flint: projection parts are not one stored parent");
    }
    return ops::prepare_linear_weight(std::span<const ops::WeightInput>(inputs)).weight;
}

// The matrices of `ids`: one when they are rows of one stored parent, else each on its own (a
// mixed-format GGUF stores a block's Q, K and V in different formats).
std::vector<Weight> matrices(const Model& model, std::initializer_list<WeightId> ids) {
    std::vector<ops::WeightInput> inputs;
    for (const WeightId id : ids) { inputs.push_back(model.input(id)); }
    if (inputs.size() > 1 && ops::joins(inputs)) {
        return {ops::prepare_linear_weight(std::span<const ops::WeightInput>(inputs)).weight};
    }
    std::vector<Weight> out;
    for (const auto& input : inputs) { out.push_back(ops::prepare_linear_weight(input).weight); }
    return out;
}

std::int64_t row_bytes(const Weight& w) {
    const auto block = gguf_block_shape(w.qtype);
    return std::int64_t(w.k / block.elements) * block.bytes;
}

// A context-sized cache plane: rows of `row_bytes`, `rows` of them for each owner (a sequence's
// KV heads), `position_bytes` a position. Owned whole, or, beside an elastic expert cache, a
// growable region whose rows are backed only as far as their owner has reached (Impl::back), so
// the expert cache holds the rest of the device until then.
struct KvArea {
    DeviceBuffer owned;
    GrowableRegion region;
    void* p = nullptr;
    std::size_t bytes = 0, row_bytes = 0, position_bytes = 0;
    int rows = 0;
    std::vector<std::uint32_t> backed; // positions backed, by owner

    KvArea() = default;
    KvArea(std::size_t owners, int rows_per_owner, std::size_t row, std::size_t position,
           bool growing, int device)
        : bytes(owners * std::size_t(rows_per_owner) * row), row_bytes(row), position_bytes(position),
          rows(rows_per_owner), backed(owners, 0) {
        if (growing) {
            region = GrowableRegion(bytes, {.device = device});
            p      = region.data();
        } else {
            owned = DeviceBuffer(bytes);
            owned.fill(0);
            p = owned.p;
        }
    }
};

struct LayerPlan {
    bool full = false, rope = false, moe = false;
    int qkv_rows = 0;
    std::vector<Weight> qkv; // one [Q; K; V] parent, or the parts of a mixed-format block
    // The dense MLP's, or a MoE block's shared expert's, [gate; up] parent and down rows.
    Weight output, gate_up, down;
    int width = 0;
    const float *input_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr,
                *sinks = nullptr;
    // MoE.
    int delivery_layer = -1;
    const float* router = nullptr;
    gguf::GgmlType gate_up_type{}, down_type{};
    std::int64_t gate_up_row_bytes = 0, down_row_bytes = 0;
    DeviceBuffer tables; // pointers: [E] gate, [E] up, [E] gate_up, [E] down
    bool gate_up_aligned16 = true, down_aligned16 = true;
    // Expert parallelism: rank r's tables (r > 0) in its device memory, entries of the experts it
    // holds (`tables` holds rank 0's); the other entries are null.
    std::vector<DeviceBuffer> peer_tables;
    // The F16 K and V caches: [sequences][kv heads][slots][D].
    KvArea k, v;
    int slots = 0, ring = 0;
    std::int64_t seq_stride = 0, head_stride = 0; // elements of a cache (and of its staging)
    // A sliding block's checkpoint of every sequence's rings (snapshot()), allocated with the
    // cache: K, then V.
    std::vector<DeviceBuffer> checkpoint;
};

// Keys of the decode graphs: the batch size.
struct GraphEntry {
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    bool warmed = false; // ran once eagerly, so every kernel is loaded before the capture
};

} // namespace

struct Executor::Impl {
    const Model& model;
    DeviceContext& device;
    ExecutorOptions options;
    const TextConfig& config;
    cudaStream_t stream = nullptr;
    int hidden = 0, query_heads = 0, kv_heads = 0, head_dim = 0, vocab = 0, ring = 0,
        global_slots = 0, window = 0;
    // A prompt chunk's sliding-window attention needs the window before it and the chunk itself:
    // a ring of prefill_ring slots that every sequence's chunks share, one block at a time
    // (staging_k/v), while each sequence keeps only its last `ring` (the window) positions.
    int prefill_ring = 0;
    DeviceBuffer staging_k, staging_v;
    KvArea mask_full; // a prompt chunk's full-attention mask, [rows][full_slots]
    // Beside an elastic expert cache, the full blocks' caches (a row per sequence and KV head) and
    // the prompt chunk's mask grow as far as the sequences reach: the expert cache gives memory
    // back when they need it and takes it again when a sequence is reset.
    bool growing = false;
    std::vector<KvArea*> sequence_areas, chunk_areas;
    std::atomic<std::uint64_t> kv_backed{0};
    std::uint64_t kv_fixed = 0; // the caches' bytes that do not grow
    float eps = 1e-6f;
    std::vector<LayerPlan> layers;
    // The CPU's share of missing experts; it outlives the delivery that calls it.
    std::unique_ptr<CpuExperts> cpu_experts;
    std::unique_ptr<ExpertDelivery> delivery;

    // Expert parallelism (model.expert_parallel()): every rank holds a range of each MoE block's
    // routed experts and rank 0 everything else. A MoE call's input goes to every other rank, which
    // runs its part on its own stream and workspace and sends it back (experts_parallel()).
    struct ExpertPeer {
        cudaStream_t stream = nullptr;
        std::unique_ptr<WorkspaceArena> workspace;
    };

    std::vector<ExpertPeer> expert_peers; // by rank; rank 0's is unused
    std::unique_ptr<ExpertParallelExchange> expert_exchange;
    DeviceBuffer expert_pack; // a call's packed input, on rank 0
    // The head divided over the ranks for calls of up to kMaxVectorColumns columns
    // (divide_head()); null keeps it whole on rank 0.
    std::unique_ptr<models::DividedHead> divided_head;
    mutable std::mutex expert_stats_mutex;
    ExpertDeliveryStats expert_startup_stats;
    Weight embedding, head;
    const float* final_norm = nullptr;
    DeviceBuffer inv_freq;
    std::vector<std::uint32_t> positions;
    // The position at which each sequence's rings last came from a prompt, a checkpoint or an
    // image: they hold the window before it and what the sequence wrote since (rewind()).
    std::vector<std::uint32_t> ring_floor;
    // Device staging: tokens, positions, sequences for up to max(prefill_chunk, sequences) rows. A
    // branch pass's rows: d_slots, the position whose cache slot each takes; d_starts, its branch's
    // first row.
    DeviceBuffer d_tokens, d_positions, d_sequences, d_slots, d_starts;
    DeviceBuffer route_done; // kernels::route_select's block count
    std::unique_ptr<PinnedHostBuffer> h_staging;
    // Recorded after the staging copies: the next stage() rewrites the pinned buffer only once
    // the previous copies have read it (a prompt chunk that keeps no logits is not synchronized).
    std::unique_ptr<CudaCompletionEvent> staged;
    DeviceBuffer logits_bf16;
    std::unique_ptr<WorkspaceArena> workspace;
    std::size_t workspace_bytes = 0;
    std::uint64_t kv_bytes      = 0;
    int decode_splits_full = 1, decode_splits_sliding = 1;
    // Decode graphs by row count, verification graphs by 16 + rows.
    std::map<std::uint32_t, GraphEntry> graphs;
    std::vector<std::uint32_t> verify_widths; // rows per sequence of the last verify()
    // The DSpine drafter's ring, table and features (ExecutorOptions::dspine): every pass stores
    // its taps of the rows it computes; a sequence's kept rows go into its lane.
    std::unique_ptr<dspine::Runtime> spine;

    Impl(const Model& m, DeviceContext& d, ExecutorOptions o)
        : model(m), device(d), options(o), config(m.config()) {
        if (options.sequences == 0 || options.prefill_chunk == 0 || options.max_context == 0) {
            throw std::invalid_argument("flint executor: invalid options");
        }
        if (device.size() != 1 && !model.expert_parallel()) {
            throw std::invalid_argument("Flint runs on one device, or divides its experts over "
                                        "several");
        }
        if (options.draft_tokens > std::uint32_t(gguf::kMaxVectorColumns) - 1) {
            throw std::invalid_argument("flint executor: at most 7 drafts a round");
        }
        stream = device.stream;
        // Beside an automatic cache of host or disk experts, the full-attention caches grow with
        // their sequences and the cache is elastic (ExpertDeliveryOptions::elastic).
        const int device_id = device.rank(0).device;
        growing = !model.expert_banks().empty() && !options.expert_cache_bytes &&
                  vmm::granularity(device_id) != 0;
        hidden      = int(config.hidden_size);
        query_heads = int(config.num_attention_heads);
        kv_heads    = int(config.num_key_value_heads);
        head_dim    = int(config.head_dim);
        vocab       = int(config.vocab_size);
        window      = int(config.sliding_window);
        eps         = config.rms_norm_eps;
        options.prefill_chunk = std::min<std::uint32_t>(options.prefill_chunk, 4096);
        // A verification writes a round's drafts past the window before keeping only some: the
        // ring holds them without overwriting a key the window still needs.
        ring         = int(round_up(config.sliding_window + options.draft_tokens, 256));
        prefill_ring = int(round_up(config.sliding_window + options.prefill_chunk, 256));
        global_slots = int(round_up(options.max_context + 1, 256));
        positions.assign(options.sequences, 0);
        ring_floor.assign(options.sequences, 0);
        const auto& w = model.weights();
        embedding     = native_weight(model.weight(w.token_embedding).view);
        head          = matrix(model, {w.output_head});
        final_norm    = fp32(model, w.final_norm, hidden);
        if (!is_gguf(embedding.qtype) || !is_gguf(head.qtype)) {
            throw std::invalid_argument("Flint runs GGUF block matrices");
        }
        if (options.dspine && (model.dspine() == nullptr || options.draft_tokens == 0 ||
                               options.draft_tokens + 1 > model.dspine()->config().block_size)) {
            throw std::invalid_argument("flint executor: DSpine drafting needs the model's drafter "
                                        "and 1 to its block less one drafts a round");
        }
        {
            // The rotation of the sliding-window blocks, as the reference computes it in FP32.
            std::vector<float> freq(head_dim / 2);
            for (std::size_t i = 0; i < freq.size(); ++i) {
                freq[i] = float(1.0 / std::pow(double(config.rope_theta),
                                               double(2 * i) / double(head_dim)));
            }
            inv_freq = DeviceBuffer(freq.size() * 4);
            inv_freq.copy_from_host(freq.data(), freq.size() * 4);
        }
        const int sequences = int(options.sequences);
        const std::size_t row_f16 = std::size_t(head_dim) * sizeof(__half);
        {
            // The caches are sized up front for every sequence's whole context: refuse with the
            // numbers rather than fail inside an allocation. Counted here: what grows with the
            // context or the sequences; the reserve covers the rest of the workspace, the CUDA
            // graphs (some captured while serving) and a margin.
            std::size_t needed = 0;
            for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
                const std::size_t kv = 2 * std::size_t(kv_heads) * row_f16 * std::size_t(sequences);
                needed += config.full_attention[i]
                              ? std::size_t(global_slots) * kv
                              : (options.checkpoints ? 2 : 1) * std::size_t(ring) * kv;
            }
            needed += std::size_t(prefill_ring) * 2 * kv_heads * row_f16 +
                      std::size_t(options.prefill_chunk) * global_slots * 2;
            if (options.dspine) {
                needed += dspine::Runtime::planned_bytes(*model.dspine(), DType::FP32,
                                                         spine_options());
            }
            std::size_t free_bytes = 0, total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            const std::size_t reserve = start_reserve();
            if (needed + reserve > free_bytes) {
                throw InsufficientMemory(
                    "Flint: the KV cache, checkpoints, prompt buffers and drafter state of " +
                    std::to_string(sequences) + " sequence(s) of " +
                    std::to_string(options.max_context) + " tokens need " +
                    std::to_string(needed >> 20) + " MiB beside " + std::to_string(reserve >> 20) +
                    " MiB for the workspace and CUDA graphs, and " +
                    std::to_string(free_bytes >> 20) +
                    " MiB are free after the weights; lower --max-concurrency or --max-context");
            }
        }
        for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
            const LayerWeights& lw = w.layers[i];
            LayerPlan p;
            p.full = config.full_attention[i];
            p.rope = config.rope[i];
            p.moe  = lw.moe.has_value();
            p.qkv  = matrices(model, {lw.attention.query, lw.attention.key, lw.attention.value});
            for (const Weight& part : p.qkv) { p.qkv_rows += part.n; }
            p.output     = matrix(model, {lw.attention.output});
            p.input_norm = fp32(model, lw.input_norm, hidden);
            p.ffn_norm   = fp32(model, lw.ffn_norm, hidden);
            p.q_norm     = fp32(model, lw.attention.query_norm, head_dim);
            p.k_norm     = fp32(model, lw.attention.key_norm, head_dim);
            p.sinks      = fp32(model, lw.attention.sinks, query_heads);
            if (!p.moe) {
                p.gate_up = matrix(model, {*lw.mlp_gate, *lw.mlp_up});
                p.down    = matrix(model, {*lw.mlp_down});
                p.width   = int(config.intermediate_size);
            } else {
                const MoeWeights& mw = *lw.moe;
                const int e = int(config.num_experts), width = int(config.moe_intermediate_size);
                p.gate_up = matrix(model, {mw.shared_gate, mw.shared_up});
                p.down    = matrix(model, {mw.shared_down});
                p.width   = int(config.shared_intermediate_size);
                p.router  = fp32(model, mw.router, e * hidden);
                if (model.expert_residency() == ExpertResidency::Device) {
                    // The tables of experts [first, last), held from `first` on by the two banks.
                    const auto bank_tables = [&](WeightId gate_up_id, WeightId down_id,
                                                 std::uint32_t first, std::uint32_t last) {
                        const Weight gate_up = matrix(model, {gate_up_id});
                        const Weight down    = matrix(model, {down_id});
                        if (!is_gguf(gate_up.qtype) || !is_gguf(down.qtype)) {
                            throw std::invalid_argument(
                                "Flint experts must be GGUF block matrices");
                        }
                        const auto gate_up_type = ops::detail::gguf_type(gate_up.qtype);
                        const auto down_type    = ops::detail::gguf_type(down.qtype);
                        if (p.gate_up_row_bytes == 0) {
                            p.gate_up_type      = gate_up_type;
                            p.down_type         = down_type;
                            p.gate_up_row_bytes = row_bytes(gate_up);
                            p.down_row_bytes    = row_bytes(down);
                        } else if (gate_up_type != p.gate_up_type || down_type != p.down_type) {
                            throw std::invalid_argument("Flint: one block's experts in different "
                                                        "formats");
                        }
                        std::vector<const void*> table(4 * std::size_t(e), nullptr);
                        for (std::uint32_t x = first; x < last; ++x) {
                            const auto* gu =
                                static_cast<const std::byte*>(gate_up.qdata) +
                                std::int64_t(x - first) * 2 * width * p.gate_up_row_bytes;
                            table[x]         = gu;
                            table[e + x]     = gu + std::int64_t(width) * p.gate_up_row_bytes;
                            table[2 * e + x] = gu;
                            table[3 * e + x] = static_cast<const std::byte*>(down.qdata) +
                                               std::int64_t(x - first) * hidden * p.down_row_bytes;
                        }
                        bool gate_up_aligned = p.gate_up_row_bytes % 16 == 0;
                        bool down_aligned    = p.down_row_bytes % 16 == 0;
                        for (std::size_t i = 0; i < table.size(); ++i) {
                            const auto address = reinterpret_cast<std::uintptr_t>(table[i]);
                            const bool aligned = table[i] == nullptr || address % 16 == 0;
                            (i < 3 * std::size_t(e) ? gate_up_aligned : down_aligned) &=
                                aligned;
                        }
                        p.gate_up_aligned16 = p.gate_up_aligned16 && gate_up_aligned;
                        p.down_aligned16    = p.down_aligned16 && down_aligned;
                        DeviceBuffer buffer(table.size() * sizeof(void*));
                        buffer.copy_from_host(table.data(), table.size() * sizeof(void*));
                        return buffer;
                    };
                    if (!model.expert_parallel()) {
                        p.tables = bank_tables(*mw.gate_up, *mw.down, 0, std::uint32_t(e));
                    } else {
                        p.peer_tables.resize(device.size());
                        for (std::size_t r = 0; r < device.size(); ++r) {
                            const auto [first, last] = model.expert_range(r);
                            RankBinding bind(device, r);
                            (r == 0 ? p.tables : p.peer_tables[r]) = bank_tables(
                                mw.gate_up_shares.at(r), mw.down_shares.at(r), first, last);
                        }
                    }
                } else {
                    p.delivery_layer    = int(i - config.dense_layers);
                    const auto& bank    = model.expert_banks()[std::size_t(p.delivery_layer)];
                    p.gate_up_type      = ops::detail::gguf_type(bank.gate_up_type);
                    p.down_type         = ops::detail::gguf_type(bank.down_type);
                    p.gate_up_row_bytes = std::int64_t(bank.gate_up_row_bytes);
                    p.down_row_bytes    = std::int64_t(bank.down_row_bytes);
                }
                if (!gguf::moe_matrix_supported(p.gate_up_type, 2 * width, hidden, true) ||
                    !gguf::moe_matrix_supported(p.down_type, hidden, width, true)) {
                    throw std::invalid_argument(
                        "Flint: no integer expert kernel for these expert formats");
                }
            }
            for (const Weight* m : {&p.output, &p.gate_up, &p.down}) {
                if (!is_gguf(m->qtype)) { throw std::invalid_argument("Flint runs GGUF block matrices"); }
            }
            for (const Weight& m : p.qkv) {
                if (!is_gguf(m.qtype)) { throw std::invalid_argument("Flint runs GGUF block matrices"); }
            }
            p.slots       = p.full ? global_slots : ring;
            p.ring        = p.full ? 0 : ring;
            p.head_stride = std::int64_t(p.slots) * head_dim;
            p.seq_stride  = p.head_stride * kv_heads;
            const bool grows = growing && p.full;
            for (KvArea* area : {&p.k, &p.v}) {
                *area = KvArea(std::size_t(sequences), kv_heads,
                               std::size_t(p.head_stride) * sizeof(__half), row_f16, grows,
                               device_id);
                kv_bytes += area->bytes;
            }
            if (!p.full && options.checkpoints) {
                // Sized up front with the cache, so a configuration that cannot hold them is refused
                // at startup instead of failing a request's first checkpoint.
                for (int c = 0; c < 2; ++c) {
                    p.checkpoint.emplace_back(std::size_t(p.seq_stride) * sizeof(__half) * sequences);
                    kv_bytes += p.checkpoint.back().bytes;
                }
            }
            layers.push_back(std::move(p));
        }
        {
            const std::size_t staging = std::size_t(prefill_ring) * kv_heads * row_f16;
            staging_k = DeviceBuffer(staging);
            staging_v = DeviceBuffer(staging);
            staging_k.fill(0);
            staging_v.fill(0);
            kv_bytes += 2 * staging;
            mask_full = KvArea(1, 1, std::size_t(options.prefill_chunk) * global_slots * 2,
                               std::size_t(options.prefill_chunk) * 2, growing, device_id);
        }
        const int sms = device.multiprocessor_count();
        // Upper bounds: a step uses as many as its keys fill (a tile of 16 each), so the sliding
        // window's 513 keys take 16 splits of two tiles and a long full-attention context fills a
        // wave of the kernel.
        decode_splits_full    = std::clamp(2 * sms / std::max(1, kv_heads), 1, 128);
        decode_splits_sliding = 16;
        const std::size_t rows = std::max<std::size_t>(options.prefill_chunk, sequences);
        d_tokens    = DeviceBuffer(rows * 4);
        d_positions = DeviceBuffer(rows * 4);
        d_sequences = DeviceBuffer(rows * 4);
        d_slots     = DeviceBuffer(rows * 4);
        d_starts    = DeviceBuffer(rows * 4);
        route_done             = DeviceBuffer(4);
        route_done.fill();
        h_staging   = std::make_unique<PinnedHostBuffer>(kStagingRegions * rows * 4);
        staged      = std::make_unique<CudaCompletionEvent>(device);
        logits_bf16 = DeviceBuffer(std::size_t(vocab) * kMaxLogitRows * 2);
        for (LayerPlan& p : layers) {
            if (!p.full) { continue; }
            sequence_areas.push_back(&p.k);
            sequence_areas.push_back(&p.v);
        }
        chunk_areas.push_back(&mask_full);
        kv_fixed = kv_bytes;
        if (growing) {
            for (KvArea* area : sequence_areas) { kv_fixed -= area->bytes; }
        }
        kv_backed       = kv_fixed;
        workspace_bytes = plan_workspace();
        {
            std::size_t free_bytes = 0, total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            if (options.prefill_chunk < kFullChunk &&
                workspace_bytes + kGraphReserve + unbacked() > free_bytes) {
                throw InsufficientMemory(
                    "Flint: the workspace of " + std::to_string(workspace_bytes >> 20) +
                    " MiB and " + std::to_string(kGraphReserve >> 20) +
                    " MiB for the CUDA graphs need more than the " +
                    std::to_string(free_bytes >> 20) +
                    " MiB free after the caches; lower --max-concurrency or --max-context");
            }
        }
        workspace = std::make_unique<WorkspaceArena>(workspace_bytes);
        if (options.dspine) { make_spine(); }
        allocate_expert_peers();
        if (!model.expert_banks().empty()) { plan_delivery(); }
    }

    // The DSpine drafter's state: a lane per sequence and one per checkpoint; a prompt chunk's
    // kept rows (the last window of them) or a decode or verification call's rows of taps; and
    // proposals for up to eight sequences at once.
    [[nodiscard]] dspine::RuntimeOptions spine_options() const {
        const dspine::Config& c     = model.dspine()->config();
        const std::uint32_t window  = dspine::ring_capacity(c, options.max_context);
        dspine::RuntimeOptions o;
        o.sequences       = options.sequences;
        o.kept_states     = options.checkpoints ? 1 : 0;
        o.max_context     = options.max_context;
        o.draft_tokens    = options.draft_tokens;
        o.draft_rows      = std::min<std::uint32_t>(options.sequences, 8);
        o.commit_rows     = std::min<std::uint32_t>(options.sequences, 8);
        o.feature_columns = std::max<std::uint32_t>(std::min(options.prefill_chunk, window),
                                                    std::uint32_t(decode_rows()));
        const ops::detail::GgufShape shape{head.qtype, head.n, head.k};
        o.head_bytes = ops::detail::gguf_project_workspace_bytes(
            {&shape, 1}, 1, static_cast<std::int32_t>(o.draft_tokens * o.draft_rows));
        return o;
    }

    // The drafter reads the target's token rows (its GGUF embedding, as the first block does) and
    // its head as FP32 logits, on rank 0.
    void make_spine() {
        dspine::Target target;
        target.hidden = std::uint32_t(hidden);
        target.vocab  = std::uint32_t(vocab);
        target.layers = config.num_hidden_layers;
        target.domain = static_cast<std::int32_t>(model.resources().public_token_count);
        target.logits = DType::FP32;
        target.embed  = [this](const Tensor& ids, Tensor& rows, cudaStream_t on) {
            gguf::dequantize_rows(ops::detail::gguf_type(embedding.qtype), embedding.qdata,
                                  row_bytes(embedding), hidden,
                                  static_cast<const std::int32_t*>(ids.data), ids.ne[0],
                                  static_cast<__nv_bfloat16*>(rows.data), hidden, on);
        };
        target.head = [this](const Tensor& x, Tensor& logits, WorkspaceArena& work,
                             cudaStream_t on) {
            ops::detail::GgufProduct product;
            product.weight   = &head;
            product.f32      = static_cast<float*>(logits.data);
            product.f32_rows = vocab;
            ops::detail::gguf_project(x, {&product, 1}, work, on);
        };
        spine = std::make_unique<dspine::Runtime>(*model.dspine(), std::move(target),
                                                  spine_options());
    }

    // Tap `li` of the drafter, when it reads that block's output: rows [first, first + n) of the
    // residual h [T][H] into feature columns [0, n).
    void store_tap(std::size_t li, const float* h, int first, int n) {
        if (!spine || n <= 0) { return; }
        const int tap = spine->tap_of(std::uint32_t(li));
        if (tap < 0) { return; }
        spine->store_tap(std::uint32_t(tap),
                         Tensor(const_cast<float*>(h) + std::int64_t(first) * hidden, DType::FP32,
                                {hidden, n}),
                         0, stream);
    }

    void plan_delivery() {
        std::vector<ExpertDeliveryLayer> sources;
        std::vector<std::uint64_t> sizes;
        for (const auto& bank : model.expert_banks()) {
            sources.push_back(bank.delivery);
            for (const auto& expert : bank.delivery.experts) {
                sizes.resize(std::max(sizes.size(), expert.size()), 0);
                for (std::size_t b = 0; b < expert.size(); ++b) {
                    sizes[b] = std::max(sizes[b], std::uint64_t(round_up(
                        expert[b].bytes + expert[b].zero_tail, kAlign)));
                }
            }
        }
        // One whole-layer bank at least (the transport adds a second where memory allows).
        std::uint64_t slot = 0;
        for (auto bytes : sizes) { slot += bytes; }
        const auto bank     = slot * config.num_experts;
        const auto metadata = std::uint64_t(options.prefill_chunk) * config.top_k_experts * 8 +
            std::uint64_t(config.moe_layers()) * config.num_experts * 4 * sizeof(void*) +
            config.num_experts * 4;
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        const auto cache = options.expert_cache_bytes.value_or(0);
        // The caches that grow later need their memory too: the elastic cache gives it back.
        if (bank + metadata + cache + unbacked() + kGraphReserve > free) {
            throw InsufficientMemory("Flint: expert delivery needs " +
                std::to_string((bank + metadata + cache) >> 20) + " MiB beside " +
                std::to_string(unbacked() >> 20) +
                " MiB for the growing caches and the CUDA Graph reserve after workspace allocation");
        }
        ExpertDeliveryOptions transport;
        transport.pair_capacity = options.prefill_chunk * config.top_k_experts;
        transport.cache_bytes   = options.expert_cache_bytes;
        // A growing cache's next positions come from the headroom before the expert cache has to
        // give memory back.
        transport.keep_free_bytes = kGraphReserve + (growing ? kKvHeadroom : 0);
        transport.elastic         = growing;
        transport.disk            = options.expert_disk;
        const float dma_share     = options.expert_dma_share.value_or(0.0F);
        if (!(dma_share >= 0 && dma_share <= 1) || options.expert_cpu_threads > 256) {
            throw std::invalid_argument("Flint: the DMA share must be 0..1 and CPU threads 0..256");
        }
        std::vector<CpuExpertLayer> cpu_layers;
        bool computes = false;
        for (const auto& bank : model.expert_banks()) {
            cpu_layers.push_back({bank.gate_up_type, bank.down_type,
                                  std::int64_t(bank.gate_up_row_bytes),
                                  std::int64_t(bank.down_row_bytes)});
            computes = computes || CpuExperts::computes(cpu_layers.back(), hidden,
                                                        int(config.moe_intermediate_size));
        }
        if (computes && dma_share < 1) {
            cpu_experts = std::make_unique<CpuExperts>(std::move(cpu_layers), hidden,
                int(config.moe_intermediate_size), int(config.top_k_experts));
            transport.cpu.products      = cpu_experts.get();
            transport.cpu.input_bytes   = cpu_experts->input_bytes();
            transport.cpu.product_bytes = cpu_experts->product_bytes();
            transport.cpu.share         = 1.0 - double(dma_share);
            transport.cpu.balance       = !options.expert_dma_share;
            transport.cpu.threads       = options.expert_cpu_threads;
        }
        delivery = std::make_unique<ExpertDelivery>(device.rank(0), model.expert_files(),
                                                    std::move(sources), transport);
        // Aliases 0 to 2 (gate, up, gate_up) point into the gate/up blob, 3 into the down blob.
        for (auto& p : layers) {
            if (p.delivery_layer < 0) { continue; }
            const auto l       = std::size_t(p.delivery_layer);
            const auto aligned = [&](std::size_t alias) {
                return delivery->alias_aligned(l, alias, 16);
            };
            p.gate_up_aligned16 =
                p.gate_up_row_bytes % 16 == 0 && aligned(0) && aligned(1) && aligned(2);
            p.down_aligned16 = p.down_row_bytes % 16 == 0 && aligned(3);
        }
    }

    // The startup check's room for the workspace and the CUDA graphs: 1.5 GiB at the full
    // 2048-token chunk; a narrower chunk keeps the least its workspace takes and then checks the
    // planned workspace with kGraphReserve beside it.
    static constexpr std::uint32_t kFullChunk  = 2048;
    static constexpr std::size_t kGraphReserve = std::size_t(320) << 20;
    static constexpr std::size_t kKvHeadroom   = std::size_t(256) << 20;

    // ---- Growing caches -------------------------------------------------------------------------

    // Device bytes the growing caches may still take.
    [[nodiscard]] std::size_t unbacked() const {
        std::size_t bytes = 0;
        for (const auto* list : {&sequence_areas, &chunk_areas}) {
            for (const KvArea* area : *list) {
                if (area->region) { bytes += area->region.bytes() - area->region.mapped_bytes(); }
            }
        }
        return bytes;
    }

    void count_backed() {
        std::uint64_t bytes = kv_fixed;
        for (const KvArea* area : sequence_areas) { bytes += area->region.mapped_bytes(); }
        kv_backed.store(bytes, std::memory_order_relaxed);
    }

    // A growing area's rows of `owner` up to `positions`: the bytes still to back, and backing them
    // (each newly backed granule zeroed on the stream). False when the device has no room.
    [[nodiscard]] std::size_t missing(const KvArea& a, std::size_t owner, std::uint32_t positions) const {
        if (!a.region || positions <= a.backed[owner]) { return 0; }
        std::size_t bytes = 0;
        for (int r = 0; r < a.rows; ++r) {
            bytes += a.region.unmapped_bytes((owner * a.rows + r) * a.row_bytes,
                                             std::size_t(positions) * a.position_bytes);
        }
        return bytes;
    }
    [[nodiscard]] bool extend(KvArea& a, std::size_t owner, std::uint32_t positions) {
        if (!a.region || positions <= a.backed[owner]) { return true; }
        for (int r = 0; r < a.rows; ++r) {
            if (!a.region.map((owner * a.rows + r) * a.row_bytes,
                              std::size_t(positions) * a.position_bytes, stream)) {
                return false;
            }
        }
        a.backed[owner] = positions;
        return true;
    }

    // Unbacks `owner`'s rows past `positions`, except granules another row still reaches into.
    // The stream has finished with them.
    void trim(KvArea& a, std::size_t owner, std::uint32_t positions) {
        if (!a.region || a.backed[owner] <= positions) { return; }
        const std::size_t g     = a.region.granule();
        const std::uint32_t was = a.backed[owner];
        a.backed[owner]         = positions;
        const auto reached = [&](std::size_t lo, std::size_t hi) {
            for (std::size_t o = 0; o < a.backed.size(); ++o) {
                for (int r = 0; r < a.rows; ++r) {
                    const std::size_t row = (o * a.rows + r) * a.row_bytes;
                    if (row < hi && lo < row + std::size_t(a.backed[o]) * a.position_bytes) { return true; }
                }
            }
            return false;
        };
        for (int r = 0; r < a.rows; ++r) {
            const std::size_t row = (owner * a.rows + r) * a.row_bytes;
            const std::size_t hi  = row + std::size_t(was) * a.position_bytes;
            for (std::size_t at = (row + std::size_t(positions) * a.position_bytes) / g * g; at < hi; at += g) {
                if (!reached(at, at + g)) { a.region.unmap(at, g); }
            }
        }
    }

    // Before a call that writes or reads `sequence`'s positions below `end`: its full-attention
    // rows (and with `chunk`, the prompt chunk's mask) are backed up to the 256-slot tile that
    // holds `end`, the expert cache giving memory back when the device keeps less than the CUDA
    // Graph reserve beside them. Outside capture.
    void back(std::uint32_t sequence, std::uint32_t end, bool chunk) {
        if (!growing) { return; }
        const auto positions = std::uint32_t(round_up(end, 256));
        std::size_t need = 0;
        for (const KvArea* area : sequence_areas) { need += missing(*area, sequence, positions); }
        if (chunk) {
            for (const KvArea* area : chunk_areas) { need += missing(*area, 0, positions); }
        }
        const auto apply = [&] {
            bool ok = true;
            for (KvArea* area : sequence_areas) { ok = ok && extend(*area, sequence, positions); }
            if (chunk) {
                for (KvArea* area : chunk_areas) { ok = ok && extend(*area, 0, positions); }
            }
            return ok;
        };
        if (need == 0) {
            (void)apply();
            return;
        }
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        if (free < need + kGraphReserve) { delivery->shrink_cache(need + kGraphReserve - free); }
        while (!apply()) {
            if (delivery->shrink_cache(std::max(need, std::size_t(64) << 20)) == 0) {
                count_backed();
                throw InsufficientMemory("Flint: no device memory left to grow the KV cache to " +
                                         std::to_string(positions) + " positions");
            }
        }
        count_backed();
    }

    // A reset sequence's rows go back to the device, the prompt mask to what the longest remaining
    // sequence reaches, and the expert cache takes what is free beyond its headroom.
    void release(std::uint32_t sequence) {
        if (!growing || sequence_areas.empty() || sequence_areas.front()->backed[sequence] == 0) { return; }
        CUDA_CHECK(ninfer::wait_for_stream(stream));
        for (KvArea* area : sequence_areas) { trim(*area, sequence, 0); }
        std::uint32_t longest = 0;
        for (const std::uint32_t backed : sequence_areas.front()->backed) { longest = std::max(longest, backed); }
        for (KvArea* area : chunk_areas) { trim(*area, 0, longest); }
        count_backed();
        delivery->grow_cache();
    }

    std::size_t start_reserve() const {
        return options.prefill_chunk >= kFullChunk ? std::size_t(1536) << 20
                                                   : std::size_t(512) << 20;
    }

    ~Impl() {
        if (delivery || growing) { ninfer::wait_for_stream(stream); }
        for (std::size_t r = 1; r < expert_peers.size(); ++r) {
            if (expert_peers[r].stream == nullptr) { continue; }
            RankBinding bind(device, r);
            ninfer::wait_for_stream(expert_peers[r].stream);
            expert_peers[r].workspace.reset();
            cudaStreamDestroy(expert_peers[r].stream);
        }
        divided_head.reset();
    }
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;

    // ---- Workspace -----------------------------------------------------------------------------

    static constexpr int kHeadRows = 64; // LM-head columns per FP32 plane pass

    // Rows of a decode or verification call.
    [[nodiscard]] int decode_rows() const {
        return std::min<int>(gguf::kMaxVectorColumns,
                             std::max<int>(int(options.sequences), int(options.draft_tokens) + 1));
    }

    std::size_t plan_workspace() const {
        const std::size_t t = options.prefill_chunk;
        const std::size_t h = hidden;
        const std::size_t q_floats = std::size_t(query_heads) * head_dim;
        std::size_t qkv_rows = 0, width = 0, gguf_scratch = 0;
        for (const LayerPlan& p : layers) {
            qkv_rows = std::max<std::size_t>(qkv_rows, p.qkv_rows);
            width    = std::max<std::size_t>(width, p.width);
            std::vector<const Weight*> products{&p.output, &p.down};
            for (const Weight& part : p.qkv) { products.push_back(&part); }
            for (const Weight* m : products) {
                const ops::detail::GgufShape shape{m->qtype, m->n, m->k};
                gguf_scratch = std::max(gguf_scratch, ops::detail::gguf_project_workspace_bytes(
                                                          {&shape, 1}, 1, int(t)));
            }
            const ops::detail::GgufShape gu{p.gate_up.qtype, p.gate_up.n, p.gate_up.k};
            gguf_scratch = std::max(gguf_scratch,
                                    ops::detail::gguf_swiglu_workspace_bytes(gu, nullptr, 1, int(t)));
        }
        const ops::detail::GgufShape hs{head.qtype, std::min<int>(head.n, vocab), head.k};
        gguf_scratch = std::max(gguf_scratch,
                                ops::detail::gguf_project_workspace_bytes({&hs, 1}, 1, kHeadRows));
        std::size_t bytes = 0;
        const auto add    = [&](std::size_t n) { bytes += round_up(n, kAlign); };
        add(t * h * 4);              // residual
        add(t * h * 2);              // normalized input
        add(t * qkv_rows * 4);       // qkv FP32
        add(t * q_floats * 4);       // q
        add(t * q_floats * 4);       // attention out FP32
        add(t * q_floats * 2);       // attention out BF16
        add(t * h * 2);              // FFN input
        add(t * h * 4);              // router input
        add(t * width * 2);          // middle
        add(t * std::size_t(prefill_ring) * 2); // sliding-window mask
        add(gguf::flash_attention_workspace_bytes(head_dim, query_heads, kv_heads, int(t)));
        add(std::size_t(kHeadRows) * vocab * 4); // head FP32 plane
        add(kHeadRows * h * 2);
        std::size_t moe = 0;
        {
            const std::size_t pairs = t * config.top_k_experts;
            const std::size_t w     = config.moe_intermediate_size;
            moe += round_up(pairs * 4, kAlign) * 2;                   // ids, weights
            moe += round_up(t * config.num_experts * 4, kAlign);      // logits
            moe += round_up(gguf::moe_routing_bytes(int(config.num_experts), int(pairs)), kAlign);
            moe += round_up(gguf::moe_matrix_activation_bytes(int(h), int(pairs)), kAlign);
            moe += round_up(pairs * 2 * w * 4, kAlign);
            moe += round_up(pairs * w * 2, kAlign);
            moe += round_up(gguf::moe_matrix_activation_bytes(int(w), int(pairs)), kAlign);
            moe += round_up(pairs * h * 4, kAlign);
            if (model.expert_parallel()) {
                // Rank 0's part: its pairs' ids and routing, and its fixed-point sum.
                moe +=
                    round_up(pairs * 4, kAlign) +
                    round_up(gguf::moe_routing_bytes(int(config.num_experts), int(pairs)), kAlign) +
                    round_up(t * h * 8, kAlign);
            }
        }
        const std::size_t decode = kernels::decode_attention_workspace_floats(
            decode_rows(), query_heads, head_dim, std::max(decode_splits_full, decode_splits_sliding)) * 4;
        return bytes + moe + round_up(decode, kAlign) + gguf_scratch + (16u << 20);
    }

    // ---- Expert parallelism
    // ----------------------------------------------------------------------

    // The most tokens one MoE call takes: a prompt chunk, or a decode or verification call.
    [[nodiscard]] int max_moe_tokens() const {
        return std::max<int>(int(options.prefill_chunk), gguf::kMaxVectorColumns);
    }

    // The byte layout of a MoE call's packed input: the activation the experts read (up to 8
    // tokens: their q8_1 vector form; wider: BF16 [T][H]), then the ids and weights, 256-aligned.
    struct ExpertInput {
        std::size_t activation = 0, ids = 0, weights = 0, bytes = 0;
    };

    [[nodiscard]] ExpertInput expert_input(int t) const {
        const std::size_t pairs = std::size_t(t) * config.top_k_experts;
        ExpertInput out;
        out.activation = t <= gguf::kMaxVectorColumns ? gguf::vector_activation_bytes(hidden, t)
                                                      : std::size_t(t) * hidden * 2;
        out.ids        = round_up(out.activation, kAlign);
        out.weights    = out.ids + round_up(pairs * 4, kAlign);
        out.bytes      = out.weights + pairs * 4;
        return out;
    }

    // Workspace of one rank's part of a call of up to `t` tokens (expert_part()).
    [[nodiscard]] std::size_t expert_part_bytes(int t) const {
        const std::size_t pairs = std::size_t(t) * config.top_k_experts;
        const std::size_t vector_pairs =
            std::size_t(gguf::kMaxVectorColumns) * config.top_k_experts;
        const std::size_t h = hidden, w = config.moe_intermediate_size;
        std::size_t bytes =
            round_up(pairs * 4, kAlign) +
            round_up(gguf::moe_routing_bytes(int(config.num_experts), int(pairs)), kAlign) +
            round_up(pairs * w * 2, kAlign);
        const std::size_t vector =
            round_up(gguf::vector_activation_bytes(int(w), int(vector_pairs)), kAlign);
        const std::size_t matrix =
            round_up(gguf::moe_matrix_activation_bytes(int(h), int(pairs)), kAlign) +
            round_up(pairs * 2 * w * 4, kAlign) +
            round_up(gguf::moe_matrix_activation_bytes(int(w), int(pairs)), kAlign) +
            round_up(pairs * h * 4, kAlign);
        return bytes + std::max(vector, matrix) + (std::size_t(1) << 20);
    }

    void allocate_expert_peers() {
        if (!model.expert_parallel()) { return; }
        const int t = max_moe_tokens();
        const std::size_t input =
            std::max(expert_input(t).bytes, expert_input(gguf::kMaxVectorColumns).bytes);
        const std::size_t part = std::size_t(t) * hidden * 8;
        expert_pack            = DeviceBuffer(input);
        expert_exchange        = std::make_unique<ExpertParallelExchange>(device, input, part);
        expert_peers.resize(device.size());
        for (std::size_t r = 1; r < device.size(); ++r) {
            RankBinding bind(device, r);
            CUDA_CHECK(cudaStreamCreateWithFlags(&expert_peers[r].stream, cudaStreamNonBlocking));
            expert_peers[r].workspace = std::make_unique<WorkspaceArena>(expert_part_bytes(t));
        }
        divide_head();
    }

    // Decode and verification calls of up to kMaxVectorColumns columns project the head divided
    // over the ranks, which otherwise wait for rank 0: each rank a range of its rows in proportion
    // to its copy bandwidth, where every other rank has room for its range beside a margin for the
    // CUDA graphs.
    void divide_head() {
        const int columns = gguf::kMaxVectorColumns;
        if (!models::DividedHead::divides(head, device.size())) { return; }
        const auto speed        = rank_copy_bandwidth(device);
        const auto free         = rank_free_bytes(device);
        const std::size_t input = gguf::vector_activation_bytes(hidden, columns);
        double speeds           = 0;
        for (const double v : speed) { speeds += v; }
        // A rank's rows, an FP32 plane and two BF16 copies of them per column, and the input.
        for (std::size_t r = 1; r < device.size(); ++r) {
            const double rows = double(head.n) * speed[r] / speeds;
            const double need = rows * double(row_bytes(head) + 8 * columns) + double(input) +
                                double(kHeadRoom);
            if (double(free[r]) < need) { return; }
        }
        divided_head = std::make_unique<models::DividedHead>(device, head, row_bytes(head), speed,
                                                             input, columns);
    }
    static constexpr std::size_t kHeadRoom = std::size_t(512) << 20;

    // One rank's part of a MoE call of block `p`: the call's pairs routed to the experts the rank
    // holds (`range`), through its `tables`, weighted into the 2^-32 fixed-point sum `fixed`
    // ([T][H]). Up to 8 tokens read the call's q8_1 activation `act`, wider calls its BF16 rows
    // `x`. The F16-sum guards may scale `weights` in place. Each contribution is the one a single
    // device computes and rounds, so the parts add up to its sum exactly.
    void expert_part(const LayerPlan& p, const void* const* tables,
                     std::pair<std::uint32_t, std::uint32_t> range, const void* act,
                     const __nv_bfloat16* x, const std::int32_t* ids, float* weights, int t,
                     unsigned long long* fixed, WorkspaceArena& space, cudaStream_t on) {
        auto scope  = space.scope();
        const int e = int(config.num_experts), k = int(config.top_k_experts);
        const int width = int(config.moe_intermediate_size);
        const int pairs = t * k;
        auto* mine = static_cast<std::int32_t*>(space.alloc_bytes(std::size_t(pairs) * 4).data);
        gguf::moe_keep_experts(ids, pairs, int(range.first), int(range.second), mine, on);
        const gguf::MoeRouting routing = gguf::moe_sort_routes(
            mine, pairs, e, space.alloc_bytes(gguf::moe_routing_bytes(e, pairs)).data, on);
        const bool gate_up_aligned = p.gate_up_aligned16, down_aligned = p.down_aligned16;
        const gguf::MoeTable gate{tables, p.gate_up_row_bytes, width, hidden, gate_up_aligned};
        const gguf::MoeTable up{tables + e, p.gate_up_row_bytes, width, hidden, gate_up_aligned};
        const gguf::MoeTable gate_up{tables + 2 * e, p.gate_up_row_bytes, 2 * width, hidden,
                                     gate_up_aligned};
        const gguf::MoeTable down{tables + 3 * e, p.down_row_bytes, hidden, width, down_aligned};
        const int max_active = std::max(1, std::min(pairs, int(range.second - range.first)));
        auto* middle =
            static_cast<__nv_bfloat16*>(space.alloc_bytes(std::size_t(pairs) * width * 2).data);
        if (t <= gguf::kMaxVectorColumns) {
            const int chunk = t <= 1 ? 1 : t <= 2 ? 2 : t <= 4 ? 4 : 8;
            void* act2      = space.alloc_bytes(gguf::vector_activation_bytes(width, pairs)).data;
            CUDA_CHECK(cudaMemsetAsync(fixed, 0, std::size_t(t) * hidden * 8, on));
            gguf::MoeOutput out;
            out.weighted = fixed;
            out.weights  = weights;
            gguf::moe_vector_swiglu(p.gate_up_type, gate, up, routing, max_active, k, act, t,
                                    middle, chunk, on);
            // Every pair's middle is quantized; another rank's pairs are never read.
            if (gguf::vector_activation_f16_sums(p.down_type)) {
                const kernels::Q8Rows middle_q{
                    static_cast<std::int8_t*>(act2),
                    reinterpret_cast<half2*>(static_cast<std::byte*>(act2) +
                                             gguf::vector_activation_scales_offset(width, pairs))};
                kernels::quantize_pairs(middle, pairs, width, weights, middle_q, on);
            } else {
                gguf::quantize_vector_activation(middle, width, pairs, nullptr, act2, on);
            }
            gguf::moe_vector_product(p.down_type, down, routing, max_active, 1, k, act2, pairs, out,
                                     chunk, on);
            return;
        }
        void* rows_act = space.alloc_bytes(gguf::moe_matrix_activation_bytes(hidden, pairs)).data;
        gguf::quantize_moe_matrix_activation(p.gate_up_type, x, hidden, routing, pairs, k, rows_act,
                                             on);
        auto* gu = static_cast<float*>(space.alloc_bytes(std::size_t(pairs) * 2 * width * 4).data);
        gguf::moe_matrix_product(p.gate_up_type, gate_up, routing, max_active, pairs, true,
                                 rows_act, gu, on);
        kernels::swiglu_rows(gu, pairs, width, middle,
                             gguf::matrix_activation_f16_sums(p.down_type) ? weights : nullptr, on);
        void* act2 = space.alloc_bytes(gguf::moe_matrix_activation_bytes(width, pairs)).data;
        gguf::quantize_moe_matrix_activation(p.down_type, middle, width, routing, pairs, 1, act2,
                                             on);
        auto* rows = static_cast<float*>(space.alloc_bytes(std::size_t(pairs) * hidden * 4).data);
        gguf::moe_matrix_product(p.down_type, down, routing, max_active, pairs, true, act2, rows,
                                 on);
        gguf::moe_weighted_rows(rows, mine, weights, t, k, hidden, fixed, on);
    }

    // Expert parallelism's MoE call after rank 0 has routed it: the input goes to every other
    // rank, each runs its part while rank 0 runs its own and the shared expert, and the parts' sum
    // lands where `combine` reads it. Up to 8 tokens this is the sum one device stores, bit for
    // bit; a wider call sums in the same fixed point where one device weighs FP32 rows in
    // ffn_combine.
    template <typename Shared>
    void experts_parallel(const LayerPlan& p, const __nv_bfloat16* x, const void* act,
                          const std::int32_t* ids, float* weights, int t,
                          kernels::FfnCombineArgs& combine, Shared&& shared) {
        const std::size_t pairs  = std::size_t(t) * config.top_k_experts;
        const std::size_t part   = std::size_t(t) * hidden * 8;
        const ExpertInput layout = expert_input(t);
        auto* pack               = static_cast<std::byte*>(expert_pack.p);
        CUDA_CHECK(cudaMemcpyAsync(pack, t <= gguf::kMaxVectorColumns ? act : x, layout.activation,
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(
            cudaMemcpyAsync(pack + layout.ids, ids, pairs * 4, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(pack + layout.weights, weights, pairs * 4,
                                   cudaMemcpyDeviceToDevice, stream));
        const std::size_t slot = expert_exchange->send(pack, layout.bytes, stream);
        for (std::size_t r = 1; r < device.size(); ++r) {
            RankBinding bind(device, r);
            ExpertPeer& peer = expert_peers[r];
            auto* input      = static_cast<std::byte*>(
                expert_exchange->receive(r, layout.bytes, slot, peer.stream));
            expert_part(p, static_cast<const void* const*>(p.peer_tables[r].p),
                        model.expert_range(r), input, reinterpret_cast<const __nv_bfloat16*>(input),
                        reinterpret_cast<const std::int32_t*>(input + layout.ids),
                        reinterpret_cast<float*>(input + layout.weights), t,
                        static_cast<unsigned long long*>(expert_exchange->part(r)), *peer.workspace,
                        peer.stream);
            expert_exchange->reply(r, part, slot, peer.stream);
        }
        auto* fixed = static_cast<unsigned long long*>(workspace->alloc_bytes(part).data);
        expert_part(p, static_cast<const void* const*>(p.tables.p), model.expert_range(0), act, x,
                    ids, weights, t, fixed, *workspace, stream);
        shared();
        std::vector<const unsigned long long*> parts;
        for (void* received : expert_exchange->collect(part, slot, stream)) {
            parts.push_back(static_cast<const unsigned long long*>(received));
        }
        gguf::add_fixed_parts(fixed, parts.data(), int(parts.size()), std::int64_t(t) * hidden,
                              stream);
        combine.moe_fixed = fixed;
    }

    // ---- Shared pieces ---------------------------------------------------------------------------

    // Up to kMaxVectorColumns rows of width k in the vector kernels' q8_1 form, written by the
    // kernel that produces them instead of a BF16 row and a separate quantization.
    struct Q8Act {
        void* base = nullptr;
        kernels::Q8Rows rows;
    };
    Q8Act alloc_q8(int k, int t) {
        Q8Act q;
        q.base    = workspace->alloc_bytes(gguf::vector_activation_bytes(k, t)).data;
        q.rows.qs = static_cast<std::int8_t*>(q.base);
        q.rows.ds = reinterpret_cast<half2*>(static_cast<std::uint8_t*>(q.base) +
                                             gguf::vector_activation_scales_offset(k, t));
        return q;
    }

    // out[:, c] (+)= W x_c for the q8_1 rows x: FP32 columns `column_stride` apart, from row `row`.
    void vector_f32(const Q8Act& x, const Weight& w, int t, float* out, int column_stride,
                    int row = 0, bool accumulate = false) {
        gguf::VectorOutput o;
        o.f32           = out + row;
        o.column_stride = column_stride;
        o.epilogue      = accumulate ? gguf::Epilogue::Accumulate : gguf::Epilogue::Store;
        gguf::vector_product(ops::detail::gguf_type(w.qtype), w.qdata, row_bytes(w), w.n, w.k,
                             x.base, t, o, stream);
    }

    // out[:, c] (+)= W x_c for BF16 rows x [K, T]: FP32 columns `column_stride` apart, from `row`.
    void project_f32(const Tensor& x, const Weight& w, float* out, int column_stride, int row = 0,
                     bool accumulate = false) {
        ops::detail::GgufProduct product;
        product.weight   = &w;
        product.f32      = out;
        product.f32_rows = column_stride;
        product.row      = row;
        product.epilogue = accumulate ? ops::detail::GgufEpilogue::Accumulate
                                      : ops::detail::GgufEpilogue::Store;
        ops::detail::gguf_project(x, {&product, 1}, *workspace, stream);
    }

    // h[:, c] += W x_c for BF16 rows x [K, T]. The vector and the matrix kernels' q8_1 rows keep
    // each 32 values' sum in F16 for some types (Q5_K among Flint's): there the columns whose sums
    // could pass F16's range are scaled by a power of two, with their h columns, and h's columns
    // restored after the product (kernels::scale_columns).
    void project_residual(const Tensor& x, const Weight& w, float* h, int t) {
        const auto type = ops::detail::gguf_type(w.qtype);
        if (!(t <= gguf::kMaxVectorColumns ? gguf::vector_activation_f16_sums(type)
                                           : gguf::matrix_activation_f16_sums(type))) {
            project_f32(x, w, h, hidden, 0, true);
            return;
        }
        auto scope  = workspace->scope();
        auto* scale = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * 4).data);
        kernels::scale_columns(static_cast<__nv_bfloat16*>(x.data), w.k, t, h, hidden, scale,
                               stream);
        project_f32(x, w, h, hidden, 0, true);
        kernels::unscale_columns(h, hidden, t, scale, stream);
    }

    // A block's Q, K and V rows into FP32 [T][qkv_rows], part by part when they are not one parent.
    void project_qkv(const Tensor* x, const Q8Act* xq, const LayerPlan& p, int t, float* out) {
        if (xq != nullptr) {
            int row = 0;
            for (const Weight& part : p.qkv) {
                vector_f32(*xq, part, t, out, p.qkv_rows, row);
                row += part.n;
            }
            return;
        }
        std::vector<ops::detail::GgufProduct> parts(p.qkv.size());
        for (std::size_t i = 0, row = 0; i < parts.size(); row += p.qkv[i].n, ++i) {
            parts[i].weight   = &p.qkv[i];
            parts[i].f32      = out;
            parts[i].f32_rows = p.qkv_rows;
            parts[i].row      = std::int32_t(row);
        }
        ops::detail::gguf_project(*x, parts, *workspace, stream);
    }

    static constexpr std::size_t kStagingRegions = 5;

    // Tokens, positions, and when given sequences, a branch pass's slots and branch starts, to the
    // device.
    void stage(std::span<const std::int32_t> a, std::span<const std::int32_t> b,
               std::span<const std::int32_t> c, std::span<const std::int32_t> g = {},
               std::span<const std::int32_t> k = {}) {
        staged->synchronize();
        auto* host = static_cast<std::int32_t*>(h_staging->data());
        const std::size_t rows = std::max<std::size_t>(options.prefill_chunk, options.sequences);
        if (a.size() > rows || b.size() > rows || c.size() > rows || g.size() > rows ||
            k.size() > rows) {
            throw std::invalid_argument("flint executor: staging overflow");
        }
        std::copy(a.begin(), a.end(), host);
        std::copy(b.begin(), b.end(), host + rows);
        std::copy(c.begin(), c.end(), host + 2 * rows);
        std::copy(g.begin(), g.end(), host + 3 * rows);
        std::copy(k.begin(), k.end(), host + 4 * rows);
        CUDA_CHECK(cudaMemcpyAsync(d_tokens.p, host, a.size() * 4, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_positions.p, host + rows, b.size() * 4,
                                   cudaMemcpyHostToDevice, stream));
        if (!c.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(d_sequences.p, host + 2 * rows, c.size() * 4,
                                       cudaMemcpyHostToDevice, stream));
        }
        if (!g.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(d_slots.p, host + 3 * rows, g.size() * 4,
                                       cudaMemcpyHostToDevice, stream));
        }
        if (!k.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(d_starts.p, host + 4 * rows, k.size() * 4,
                                       cudaMemcpyHostToDevice, stream));
        }
        staged->record(stream);
    }

    // The embeddings of the staged tokens into FP32 h [T][H].
    void embed(int tokens, float* h) {
        gguf::dequantize_rows(ops::detail::gguf_type(embedding.qtype), embedding.qdata,
                              row_bytes(embedding), hidden,
                              static_cast<const std::int32_t*>(d_tokens.p), tokens, h, hidden,
                              stream);
    }

    // The FFN of block `p` after its attention has been added to the residual h: the dense MLP or
    // the routed and shared experts into h, and the normalized input of what follows (`next_norm`:
    // the next block's input norm, or the final norm) into `next` (BF16) and/or `next_q`.
    void block_tail(const LayerPlan& p, float* h, int t, const float* next_norm,
                    __nv_bfloat16* next, const Q8Act* next_q) {
        auto scope    = workspace->scope();
        const bool q8 = t <= gguf::kMaxVectorColumns;
        Tensor y      = workspace->alloc(DType::BF16, {hidden, t});
        Q8Act yq;
        float* router_x = nullptr;
        if (q8) { yq = alloc_q8(hidden, t); }
        if (p.moe) {
            router_x = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * hidden * 4).data);
        }
        kernels::rms_norm(h, p.ffn_norm, hidden, t, eps,
                          q8 ? nullptr : static_cast<__nv_bfloat16*>(y.data), stream, yq.rows,
                          router_x);
        // The dense MLP or the shared expert: SwiGLU over the [gate; up] parent, then the down rows
        // added to the residual.
        const auto mlp = [&] {
            auto mlp_scope = workspace->scope();
            Tensor middle  = workspace->alloc(DType::BF16, {p.width, t});
            if (q8) {
                gguf::vector_swiglu(ops::detail::gguf_type(p.gate_up.qtype), p.gate_up.qdata,
                                    row_bytes(p.gate_up), p.width, hidden, yq.base, t,
                                    static_cast<__nv_bfloat16*>(middle.data), p.width, stream);
            } else {
                ops::detail::gguf_swiglu(y, p.gate_up, nullptr, middle, *workspace, stream);
            }
            project_residual(middle, p.down, h, t);
        };
        kernels::FfnCombineArgs combine;
        combine.h         = h;
        combine.hidden    = hidden;
        combine.tokens    = t;
        combine.eps       = eps;
        combine.next_norm = next_norm;
        combine.next_out  = next;
        if (next_q != nullptr) { combine.next_q = next_q->rows; }
        if (p.moe) {
            experts(p, static_cast<const __nv_bfloat16*>(y.data), q8 ? &yq : nullptr, router_x, t,
                    combine, mlp);
        } else {
            mlp();
        }
        kernels::ffn_combine(combine, stream);
    }

    // The routed experts of block `p` (their output lands where `combine` reads it) and `shared`,
    // the shared expert, run while missing experts are copied in.
    template <typename Shared>
    void experts(const LayerPlan& p, const __nv_bfloat16* x, const Q8Act* xq, const float* router_x,
                 int t, kernels::FfnCombineArgs& combine, Shared&& shared) {
        const int e = int(config.num_experts), k = int(config.top_k_experts);
        const int width = int(config.moe_intermediate_size);
        const int pairs = t * k;
        auto* ids      = static_cast<std::int32_t*>(workspace->alloc_bytes(std::size_t(pairs) * 4).data);
        auto* weights  = static_cast<float*>(workspace->alloc_bytes(std::size_t(pairs) * 4).data);
        auto* logits   = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * e * 4).data);
        void* routing_space = workspace->alloc_bytes(gguf::moe_routing_bytes(e, pairs)).data;
        const bool vector   = t <= gguf::kMaxVectorColumns;
        gguf::MoeRouting routing;
        if (vector) {
            auto* base = static_cast<std::int32_t*>(routing_space);
            routing.bounds       = base;
            routing.sorted       = base + e + 1;
            routing.active       = base + e + 1 + pairs;
            routing.active_count = base + e + 1 + pairs + e;
            kernels::route_select(router_x, hidden, t, p.router, e, k, logits, ids, weights, base,
                                  base + e + 1, base + e + 1 + pairs, base + e + 1 + pairs + e,
                                  static_cast<std::uint32_t*>(route_done.p), stream);
        } else {
            kernels::route(router_x, hidden, t, p.router, e, k, logits, ids, weights, stream);
            if (!model.expert_parallel()) {
                routing = gguf::moe_sort_routes(ids, pairs, e, routing_space, stream);
            }
        }
        // A decode or verification call may leave misses to the CPU, which reads the quantized
        // activation and the route weights.
        const bool cpu = cpu_experts && t <= CpuExperts::kMaxTokens && vector;
        const void* act = vector ? xq->base : nullptr;
        if (model.expert_parallel()) {
            experts_parallel(p, x, act, ids, weights, t, combine, shared);
            return;
        }
        std::optional<ExpertDelivery::Call> call;
        if (delivery) {
            const std::array<ExpertDelivery::Input, 2> inputs{
                ExpertDelivery::Input{act, cpu ? cpu_experts->activation_bytes(t) : 0},
                ExpertDelivery::Input{weights, cpu ? cpu_experts->weight_bytes(t) : 0}};
            call = delivery->begin(std::size_t(p.delivery_layer), ids, std::uint32_t(pairs), stream,
                                   cpu ? std::span<const ExpertDelivery::Input>(inputs)
                                       : std::span<const ExpertDelivery::Input>());
        }
        const auto* tables = delivery ? delivery->tables(std::size_t(p.delivery_layer))
                                      : static_cast<const void* const*>(p.tables.p);
        const bool gate_up_aligned = p.gate_up_aligned16, down_aligned = p.down_aligned16;
        const gguf::MoeTable gate{tables, p.gate_up_row_bytes, width, hidden, gate_up_aligned};
        const gguf::MoeTable up{tables + e, p.gate_up_row_bytes, width, hidden, gate_up_aligned};
        const gguf::MoeTable gate_up{tables + 2 * e, p.gate_up_row_bytes, 2 * width, hidden,
                                     gate_up_aligned};
        const gguf::MoeTable down{tables + 3 * e, p.down_row_bytes, hidden, width, down_aligned};
        const int max_active = std::min(pairs, e);
        auto* middle = static_cast<__nv_bfloat16*>(
            workspace->alloc_bytes(std::size_t(pairs) * width * 2).data);
        if (vector) {
            const int chunk = t <= 1 ? 1 : t <= 2 ? 2 : t <= 4 ? 4 : 8;
            void* act2  = workspace->alloc_bytes(gguf::vector_activation_bytes(width, pairs)).data;
            auto* fixed = static_cast<unsigned long long*>(
                workspace->alloc_bytes(std::size_t(t) * hidden * 8).data);
            CUDA_CHECK(cudaMemsetAsync(fixed, 0, std::size_t(t) * hidden * 8, stream));
            gguf::MoeOutput out;
            out.weighted = fixed;
            out.weights  = weights;
            // Down rows whose dot reads the activation's F16 sums take the middles' guard, which
            // may scale a pair's weight by a power of two.
            const bool guard = gguf::vector_activation_f16_sums(p.down_type);
            const kernels::Q8Rows middle_q{
                static_cast<std::int8_t*>(act2),
                reinterpret_cast<half2*>(static_cast<std::byte*>(act2) +
                                         gguf::vector_activation_scales_offset(width, pairs))};
            const auto compute = [&](const gguf::MoeRouting& part) {
                gguf::moe_vector_swiglu(p.gate_up_type, gate, up, part, max_active, k, act, t,
                                        middle, chunk, stream);
                if (guard) {
                    kernels::quantize_pairs(middle, pairs, width, weights, middle_q, stream);
                } else {
                    gguf::quantize_vector_activation(middle, width, pairs, nullptr, act2, stream);
                }
                gguf::moe_vector_product(p.down_type, down, part, max_active, 1, k, act2, pairs,
                                         out, chunk, stream);
            };
            if (call) {
                // Both stages add to the same exact fixed-point sum, zeroed once above, each over
                // its experts of the call's sorted routes; the shared expert runs between them,
                // while the missing experts arrive. Quantization reads every pair: the cached stage
                // finds the other pairs cleared, the missing stage the cached pairs' middles.
                const auto stage = [&](const std::int32_t* set) {
                    gguf::MoeRouting part = routing;
                    part.active_count     = set;
                    part.active           = set + 1;
                    return part;
                };
                CUDA_CHECK(cudaMemsetAsync(middle, 0, std::size_t(pairs) * width * 2, stream));
                compute(stage(call->cached_experts));
                shared();
                delivery->await(stream);
                compute(stage(call->missing_experts));
                if (call->cpu != nullptr) {
                    kernels::add_fixed(fixed, call->products, call->cpu, std::int64_t(t) * hidden, stream);
                }
            } else {
                compute(routing);
                shared();
            }
            combine.moe_fixed = fixed;
            return;
        }
        shared();
        if (delivery) { delivery->await(stream); }
        void* rows_act = workspace->alloc_bytes(gguf::moe_matrix_activation_bytes(hidden, pairs)).data;
        gguf::quantize_moe_matrix_activation(p.gate_up_type, x, hidden, routing, pairs, k, rows_act,
                                             stream);
        auto* gu = static_cast<float*>(
            workspace->alloc_bytes(std::size_t(pairs) * 2 * width * 4).data);
        gguf::moe_matrix_product(p.gate_up_type, gate_up, routing, max_active, pairs, true, rows_act,
                                 gu, stream);
        // The route weights take the power-of-two factors of rows scaled for the down product.
        kernels::swiglu_rows(gu, pairs, width, middle,
                             gguf::matrix_activation_f16_sums(p.down_type) ? weights : nullptr,
                             stream);
        void* act2 = workspace->alloc_bytes(gguf::moe_matrix_activation_bytes(width, pairs)).data;
        gguf::quantize_moe_matrix_activation(p.down_type, middle, width, routing, pairs, 1, act2,
                                             stream);
        auto* rows = static_cast<float*>(
            workspace->alloc_bytes(std::size_t(pairs) * hidden * 4).data);
        gguf::moe_matrix_product(p.down_type, down, routing, max_active, pairs, true, act2, rows,
                                 stream);
        combine.moe_rows    = rows;
        combine.moe_weights = weights;
        combine.top_k       = k;
    }

    // The head of `rows` final-normed rows x into logits_bf16 columns [first, first + rows).
    void head_logits(const __nv_bfloat16* rows_x, int rows, int first) {
        for (int at = 0; at < rows; at += kHeadRows) {
            const int n  = std::min(kHeadRows, rows - at);
            auto scope   = workspace->scope();
            Tensor x(const_cast<__nv_bfloat16*>(rows_x) + std::int64_t(at) * hidden, DType::BF16,
                     {hidden, n});
            auto* plane = static_cast<float*>(
                workspace->alloc_bytes(std::size_t(n) * vocab * 4).data);
            project_f32(x, head, plane, vocab);
            kernels::f32_to_bf16(plane, static_cast<__nv_bfloat16*>(logits_bf16.p) +
                                            std::int64_t(first + at) * vocab,
                                 std::int64_t(n) * vocab, stream);
        }
    }

    // The head of the last `rows` of t <= kMaxVectorColumns final-normed q8_1 rows into
    // logits_bf16 columns [0, rows).
    void head_logits_q8(const Q8Act& x, int t, int rows) {
        if (divided_head) {
            std::vector<cudaStream_t> streams{stream};
            for (std::size_t r = 1; r < expert_peers.size(); ++r) {
                streams.push_back(expert_peers[r].stream);
            }
            divided_head->project(
                x.base, gguf::vector_activation_bytes(hidden, t), rows,
                static_cast<__nv_bfloat16*>(logits_bf16.p), streams,
                [&](std::size_t, const void* input, const Weight& piece, int, float* plane,
                    __nv_bfloat16* out, cudaStream_t on) {
                    gguf::VectorOutput o;
                    o.f32           = plane;
                    o.column_stride = piece.n;
                    gguf::vector_product(ops::detail::gguf_type(piece.qtype), piece.qdata,
                                         row_bytes(piece), piece.n, piece.k, input, t, o, on);
                    kernels::f32_to_bf16(plane + std::int64_t(t - rows) * piece.n, out,
                                         std::int64_t(rows) * piece.n, on);
                });
            return;
        }
        auto scope  = workspace->scope();
        auto* plane = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * vocab * 4).data);
        vector_f32(x, head, t, plane, vocab);
        kernels::f32_to_bf16(plane + std::int64_t(t - rows) * vocab,
                             static_cast<__nv_bfloat16*>(logits_bf16.p),
                             std::int64_t(rows) * vocab, stream);
    }

    // ---- Prompt chunks -----------------------------------------------------------------------------

    // With `branches` (forward_branches): the chunk is those branches' tokens one after another,
    // each continuing the sequence at its position, and `logit_rows` is their count. With `shared`
    // (forward_prompt_branches) the chunk's first `shared` tokens continue the sequence ahead of the
    // branches, which continue it from there; the sequence keeps those rows and stops there.
    void forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                 std::uint32_t logit_rows, std::span<const std::uint32_t> branches = {},
                 std::uint32_t shared = 0) {
        const int t = int(tokens.size());
        if (sequence >= options.sequences || t <= 0 || std::uint32_t(t) > options.prefill_chunk ||
            logit_rows > std::min<std::uint32_t>(t, kMaxLogitRows) ||
            (logit_rows == 0 && !branches.empty())) {
            throw std::invalid_argument("flint forward: invalid call");
        }
        // No logits: a full last block only writes its keys and values (Executor::forward), unless
        // the drafter reads its output.
        const bool kv_only   = logit_rows == 0;
        const bool branching = !branches.empty();
        const bool last_tap  = spine && spine->tap_of(std::uint32_t(layers.size() - 1)) >= 0;
        if (branching && (logit_rows != branches.size() || shared >= std::uint32_t(t) ||
                          std::accumulate(branches.begin(), branches.end(), std::uint64_t{0}) !=
                              std::uint64_t(t) - shared ||
                          std::find(branches.begin(), branches.end(), 0U) != branches.end())) {
            throw std::invalid_argument("flint forward_branches: invalid branches");
        }
        const std::uint32_t first = positions[sequence];
        if (first + std::uint32_t(t) > options.max_context) {
            throw std::invalid_argument("flint forward: context capacity exceeded");
        }
        back(sequence, first + std::uint32_t(t), true);
        std::vector<std::int32_t> pos(t);
        for (int i = 0; i < t; ++i) { pos[i] = std::int32_t(first) + i; }
        // A branch pass: each branch row rotates at its branch's position, and takes the cache
        // slot of its place in the chunk (`slots`, the positions a plain chunk has); `starts` its
        // branch's first row, for the masks. The shared rows keep their own positions.
        std::vector<std::int32_t> slots, starts;
        if (branching) {
            slots = pos;
            starts.resize(t);
            for (std::size_t b = 0, row = shared; b < branches.size(); row += branches[b++]) {
                for (std::uint32_t j = 0; j < branches[b]; ++j) {
                    pos[row + j]    = std::int32_t(first + shared + j);
                    starts[row + j] = std::int32_t(row);
                }
            }
        }
        stage(tokens, pos, {}, slots, starts);
        const auto* d_branch_starts =
            branching ? static_cast<const std::int32_t*>(d_starts.p) : nullptr;
        workspace->reset();
        auto scope = workspace->scope();
        auto* h    = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * hidden * 4).data);
        embed(t, h);
        // Masks: the sliding ring and the full-attention prefix.
        const int full_slots = int(round_up(first + t, 256));
        auto* mask_ring = static_cast<__half*>(
            workspace->alloc_bytes(std::size_t(t) * prefill_ring * 2).data);
        auto* mask_full = static_cast<__half*>(this->mask_full.p);
        kernels::build_mask(mask_ring, t, prefill_ring, int(first), prefill_ring, window, stream,
                            d_branch_starts, int(shared));
        kernels::build_mask(mask_full, t, full_slots, int(first), 0, 0, stream, d_branch_starts,
                            int(shared));
        // Each block's normalized input rows; after the last block, the final-normed rows (q8_1
        // for a vector-width chunk).
        const bool q8 = t <= gguf::kMaxVectorColumns;
        Tensor x      = workspace->alloc(DType::BF16, {hidden, t});
        Q8Act xq;
        if (q8) { xq = alloc_q8(hidden, t); }
        kernels::rms_norm(h, layers.front().input_norm, hidden, t, eps,
                          q8 ? nullptr : static_cast<__nv_bfloat16*>(x.data), stream, xq.rows);
        const int q_floats = query_heads * head_dim;
        const std::int64_t stage_stride = std::int64_t(prefill_ring) * head_dim;
        // The drafter's taps of the rows the sequence keeps (a branch pass's shared rows), the
        // last window of them: no earlier one is visible past the chunk.
        const int kept_rows = branching ? int(shared) : t;
        const int tap_rows  = spine ? std::min(kept_rows, int(spine->window())) : 0;
        const int tap_first = kept_rows - tap_rows;
        for (std::size_t li = 0; li < layers.size(); ++li) {
            const LayerPlan& p = layers[li];
            auto layer_scope   = workspace->scope();
            auto* qkv = static_cast<float*>(
                workspace->alloc_bytes(std::size_t(t) * p.qkv_rows * 4).data);
            project_qkv(q8 ? nullptr : &x, q8 ? &xq : nullptr, p, t, qkv);
            auto* q = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * q_floats * 4).data);
            kernels::QkvArgs qa;
            qa.qkv         = qkv;
            qa.tokens      = t;
            qa.query_heads = query_heads;
            qa.kv_heads    = kv_heads;
            qa.head_dim    = head_dim;
            qa.q_norm      = p.q_norm;
            qa.k_norm      = p.k_norm;
            qa.eps         = eps;
            qa.positions   = static_cast<const std::int32_t*>(d_positions.p);
            qa.inv_freq    = p.rope ? static_cast<const float*>(inv_freq.p) : nullptr;
            qa.q_out       = q;
            qa.slots       = branching ? static_cast<const std::int32_t*>(d_slots.p) : nullptr;
            // A sliding block attends over the staging ring: the window before the chunk first
            // comes over from the sequence's ring, the chunk's keys land there, and the window that
            // ends the chunk goes back. A full block's keys and values land in the sequence's cache.
            const std::int64_t seq_offset = std::int64_t(sequence) * p.seq_stride;
            kernels::RingCopyArgs keep;
            keep.kv_heads = kv_heads;
            keep.head_dim = head_dim;
            if (!p.full) {
                keep.src_k           = static_cast<const __half*>(p.k.p) + seq_offset;
                keep.src_v           = static_cast<const __half*>(p.v.p) + seq_offset;
                keep.dst_k           = static_cast<__half*>(staging_k.p);
                keep.dst_v           = static_cast<__half*>(staging_v.p);
                keep.src_head_stride = p.head_stride;
                keep.dst_head_stride = stage_stride;
                keep.src_ring        = p.ring;
                keep.dst_ring        = prefill_ring;
                keep.begin           = std::max(0, int(first) - window);
                keep.end             = int(first);
                kernels::ring_copy(keep, stream);
                qa.k_cache     = static_cast<__half*>(staging_k.p);
                qa.v_cache     = static_cast<__half*>(staging_v.p);
                qa.head_stride = stage_stride;
                qa.ring        = prefill_ring;
            } else {
                qa.k_cache     = static_cast<__half*>(p.k.p);
                qa.v_cache     = static_cast<__half*>(p.v.p);
                qa.seq_stride  = p.seq_stride;
                qa.head_stride = p.head_stride;
                qa.sequence    = int(sequence);
            }
            kernels::qkv_post(qa, stream);
            if (kv_only && p.full && li + 1 == layers.size() && !last_tap) { break; }
            auto* attn = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * q_floats * 4).data);
            const std::size_t fa_bytes =
                gguf::flash_attention_workspace_bytes(head_dim, query_heads, kv_heads, t);
            void* fa_space = workspace->alloc_bytes(fa_bytes).data;
            gguf::FlashAttentionArgs fa;
            fa.head_dim       = head_dim;
            fa.query_heads    = query_heads;
            fa.kv_heads       = kv_heads;
            fa.tokens         = t;
            fa.q              = q;
            fa.q_token_stride = q_floats;
            fa.k = p.full ? static_cast<const __half*>(p.k.p) + seq_offset
                          : static_cast<const __half*>(staging_k.p);
            fa.v = p.full ? static_cast<const __half*>(p.v.p) + seq_offset
                          : static_cast<const __half*>(staging_v.p);
            fa.row_bytes       = std::int64_t(head_dim) * 2;
            fa.head_bytes      = (p.full ? p.head_stride : stage_stride) * 2;
            fa.slots           = p.full ? full_slots : prefill_ring;
            fa.mask            = p.full ? mask_full : mask_ring;
            fa.mask_row_halves = fa.slots;
            fa.out             = attn;
            fa.scale           = 1.0f / std::sqrt(float(head_dim));
            fa.sinks           = p.sinks;
            gguf::flash_attention(fa, fa_space, fa_bytes, stream);
            // The window that ends the sequence's rows of the chunk goes back: all of them, or a
            // branch pass's shared rows (the branches' rows lie past them, later in the ring).
            const int kept = int(first) + (branching ? int(shared) : t);
            if (!p.full && kept > int(first)) {
                kernels::RingCopyArgs back = keep;
                std::swap(back.src_head_stride, back.dst_head_stride);
                std::swap(back.src_ring, back.dst_ring);
                back.src_k = static_cast<const __half*>(staging_k.p);
                back.src_v = static_cast<const __half*>(staging_v.p);
                back.dst_k = static_cast<__half*>(p.k.p) + seq_offset;
                back.dst_v = static_cast<__half*>(p.v.p) + seq_offset;
                back.begin = std::max(0, kept - window);
                back.end   = kept;
                kernels::ring_copy(back, stream);
            }
            Tensor attn_bf16 = workspace->alloc(DType::BF16, {q_floats, t});
            kernels::f32_to_bf16(attn, static_cast<__nv_bfloat16*>(attn_bf16.data),
                                 std::int64_t(t) * q_floats, stream);
            project_residual(attn_bf16, p.output, h, t);
            const bool last = li + 1 == layers.size();
            // The final rows in BF16 too when a branch pass gathers its branches' last ones.
            block_tail(p, h, t, last ? final_norm : layers[li + 1].input_norm,
                       q8 && !(last && branching) ? nullptr : static_cast<__nv_bfloat16*>(x.data),
                       q8 ? &xq : nullptr);
            store_tap(li, h, tap_first, tap_rows);
        }
        if (tap_rows > 0) {
            spine->commit(spine->lane(sequence), first + std::uint32_t(tap_first), 0,
                          std::uint32_t(tap_rows), stream);
        }
        if (branching) {
            // Each branch's last row to the front, in branch order; a branch's last row lies at or
            // past its index, so no copy overwrites a later one's source. The sequence ends after
            // its shared rows, its window there.
            auto* rows      = static_cast<__nv_bfloat16*>(x.data);
            std::size_t row = shared;
            for (std::size_t b = 0; b < branches.size(); ++b) {
                row += branches[b];
                if (row - 1 == b) { continue; }
                CUDA_CHECK(cudaMemcpyAsync(
                    rows + std::int64_t(b) * hidden, rows + std::int64_t(row - 1) * hidden,
                    std::size_t(hidden) * 2, cudaMemcpyDeviceToDevice, stream));
            }
            head_logits(rows, int(logit_rows), 0);
            positions[sequence]  = first + shared;
            ring_floor[sequence] = first + shared;
            return;
        }
        if (kv_only) {
            positions[sequence]  = first + std::uint32_t(t);
            ring_floor[sequence] = first + std::uint32_t(t);
            return;
        }
        if (q8) {
            head_logits_q8(xq, t, int(logit_rows));
        } else {
            head_logits(static_cast<const __nv_bfloat16*>(x.data) +
                            std::int64_t(t - int(logit_rows)) * hidden,
                        int(logit_rows), 0);
        }
        positions[sequence]  = first + std::uint32_t(t);
        ring_floor[sequence] = first + std::uint32_t(t);
    }

    // ---- Decode ---------------------------------------------------------------------------------------

    // One token per row; tokens, positions and sequences already staged on the device. With
    // `keep` (a decode step) each row's drafter taps go into its sequence's lane; a verification
    // keeps them for commit().
    void decode_body(int b, bool keep) {
        workspace->reset();
        auto scope = workspace->scope();
        auto* h    = static_cast<float*>(workspace->alloc_bytes(std::size_t(b) * hidden * 4).data);
        embed(b, h);
        // Each block's normalized input rows in q8_1 form; after the last block, the final-normed
        // rows.
        const Q8Act xq = alloc_q8(hidden, b);
        kernels::rms_norm(h, layers.front().input_norm, hidden, b, eps, nullptr, stream, xq.rows);
        const int q_floats = query_heads * head_dim;
        for (std::size_t li = 0; li < layers.size(); ++li) {
            const LayerPlan& p = layers[li];
            auto layer_scope   = workspace->scope();
            auto* qkv = static_cast<float*>(
                workspace->alloc_bytes(std::size_t(b) * p.qkv_rows * 4).data);
            project_qkv(nullptr, &xq, p, b, qkv);
            auto* q = static_cast<float*>(workspace->alloc_bytes(std::size_t(b) * q_floats * 4).data);
            kernels::QkvArgs qa;
            qa.qkv         = qkv;
            qa.tokens      = b;
            qa.query_heads = query_heads;
            qa.kv_heads    = kv_heads;
            qa.head_dim    = head_dim;
            qa.q_norm      = p.q_norm;
            qa.k_norm      = p.k_norm;
            qa.eps         = eps;
            qa.positions   = static_cast<const std::int32_t*>(d_positions.p);
            qa.inv_freq    = p.rope ? static_cast<const float*>(inv_freq.p) : nullptr;
            qa.q_out       = q;
            qa.sequences   = static_cast<const std::int32_t*>(d_sequences.p);
            qa.k_cache     = static_cast<__half*>(p.k.p);
            qa.v_cache     = static_cast<__half*>(p.v.p);
            qa.seq_stride  = p.seq_stride;
            qa.head_stride = p.head_stride;
            qa.ring        = p.ring;
            kernels::qkv_post(qa, stream);
            const int splits = p.full ? decode_splits_full : decode_splits_sliding;
            auto* partial    = static_cast<float*>(
                workspace->alloc_bytes(kernels::decode_attention_workspace_floats(
                                           b, query_heads, head_dim, splits) * 4).data);
            const Q8Act attn = alloc_q8(q_floats, b);
            kernels::DecodeAttentionArgs da;
            da.q           = q;
            da.k_cache     = static_cast<const __half*>(p.k.p);
            da.v_cache     = static_cast<const __half*>(p.v.p);
            da.sinks       = p.sinks;
            da.seq_stride  = p.seq_stride;
            da.head_stride = p.head_stride;
            da.sequences   = static_cast<const std::int32_t*>(d_sequences.p);
            da.positions   = static_cast<const std::int32_t*>(d_positions.p);
            da.batch       = b;
            da.query_heads = query_heads;
            da.kv_heads    = kv_heads;
            da.head_dim    = head_dim;
            da.ring        = p.ring;
            da.window      = window;
            da.splits      = splits;
            da.scale       = 1.0f / std::sqrt(float(head_dim));
            da.partial     = partial;
            da.out_q       = attn.rows;
            kernels::decode_attention(da, stream);
            vector_f32(attn, p.output, b, h, hidden, 0, true);
            const bool last = li + 1 == layers.size();
            block_tail(p, h, b, last ? final_norm : layers[li + 1].input_norm, nullptr, &xq);
            store_tap(li, h, 0, b);
        }
        head_logits_q8(xq, b, b);
        if (spine && keep) {
            spine->commit_rows(Tensor(d_sequences.p, DType::I32, {b}),
                               Tensor(d_positions.p, DType::I32, {b}), 0, stream);
        }
    }

    // Runs `body` from the graph of `key`: eagerly the first time, so every kernel is loaded
    // before the capture, then captured and replayed.
    template <typename Body>
    void run_graph(std::uint32_t key, Body&& body) {
        if (!options.cuda_graphs) {
            body();
            return;
        }
        GraphEntry& entry = graphs[key];
        if (!entry.warmed) {
            // A graph launched before this may still wait on the device for the experts' host
            // service, and a kernel that loads at its first launch during that wait never
            // returns: the stream is idle before a body's kernels load.
            CUDA_CHECK(ninfer::wait_for_stream(stream));
            body();
            entry.warmed = true;
            return;
        }
        if (!entry.executable.ready()) {
            entry.definition.capture(stream, [&] { body(); });
            entry.executable.instantiate(entry.definition);
        }
        entry.executable.launch(stream);
    }

    void decode(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens) {
        const int b = int(sequences.size());
        if (b <= 0 || std::uint32_t(b) > options.sequences || tokens.size() != sequences.size() ||
            b > gguf::kMaxVectorColumns) {
            throw std::invalid_argument("flint decode: invalid batch");
        }
        std::vector<std::int32_t> pos(b), seq(b);
        for (int i = 0; i < b; ++i) {
            if (sequences[i] >= options.sequences) {
                throw std::invalid_argument("flint decode: invalid sequence");
            }
            if (positions[sequences[i]] + 1 > options.max_context) {
                throw std::invalid_argument("flint decode: context capacity exceeded");
            }
            pos[i] = std::int32_t(positions[sequences[i]]);
            seq[i] = std::int32_t(sequences[i]);
        }
        for (int i = 0; i < b; ++i) { back(sequences[i], positions[sequences[i]] + 1, false); }
        stage(tokens, pos, seq);
        run_graph(std::uint32_t(b), [&] { decode_body(b, true); });
        for (int i = 0; i < b; ++i) { positions[sequences[i]] += 1; }
    }

    // ---- Speculative rounds ------------------------------------------------------------------------

    void verify(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens,
                std::span<const std::uint32_t> widths = {}) {
        const int n = int(sequences.size());
        if (options.draft_tokens == 0 || n <= 0 ||
            (widths.empty() && tokens.size() % sequences.size() != 0) ||
            (!widths.empty() && widths.size() != sequences.size())) {
            throw std::invalid_argument("flint verify: invalid call");
        }
        std::vector<std::uint32_t> row_widths(widths.begin(), widths.end());
        if (row_widths.empty()) { row_widths.assign(n, std::uint32_t(tokens.size()) / n); }
        int rows = 0;
        for (const std::uint32_t w : row_widths) {
            if (w < 1 || w > options.draft_tokens + 1) {
                throw std::invalid_argument("flint verify: invalid width");
            }
            rows += int(w);
        }
        if (std::size_t(rows) != tokens.size() || rows > gguf::kMaxVectorColumns) {
            throw std::invalid_argument("flint verify: invalid width");
        }
        std::vector<std::int32_t> pos(rows), seq(rows);
        for (int j = 0, at = 0; j < n; ++j) {
            const std::uint32_t s = sequences[j];
            if (s >= options.sequences ||
                std::count(sequences.begin(), sequences.begin() + j, s) != 0) {
                throw std::invalid_argument("flint verify: invalid sequence");
            }
            if (positions[s] + row_widths[j] > options.max_context) {
                throw std::invalid_argument("flint verify: context capacity exceeded");
            }
            for (std::uint32_t i = 0; i < row_widths[j]; ++i, ++at) {
                pos[at] = std::int32_t(positions[s] + i);
                seq[at] = std::int32_t(s);
            }
        }
        for (int j = 0; j < n; ++j) {
            back(sequences[j], positions[sequences[j]] + row_widths[j], false);
        }
        stage(tokens, pos, seq);
        run_graph(16 + std::uint32_t(rows), [&] { decode_body(rows, false); });
        verify_widths = std::move(row_widths);
    }

    void commit(std::span<const std::uint32_t> sequences, std::span<const std::uint32_t> kept) {
        if (kept.size() != sequences.size() || verify_widths.size() != sequences.size()) {
            throw std::invalid_argument("flint commit: invalid call");
        }
        for (std::size_t j = 0; j < sequences.size(); ++j) {
            const std::uint32_t s = sequences[j];
            if (s >= options.sequences || kept[j] < 1 || kept[j] > verify_widths[j]) {
                throw std::invalid_argument("flint commit: invalid row");
            }
        }
        // The verification left each sequence's taps at its rows' columns, in call order.
        for (std::size_t j = 0, column = 0; j < sequences.size(); column += verify_widths[j++]) {
            const std::uint32_t s = sequences[j];
            if (spine) {
                spine->commit(spine->lane(s), positions[s], std::uint32_t(column), kept[j], stream);
            }
            positions[s] += kept[j];
        }
        verify_widths.clear();
    }

    void draft(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> anchors,
               std::uint32_t k, std::span<std::int32_t> drafts, std::span<float> confidence) {
        if (!spine) { throw std::logic_error("flint: an executor without a DSpine drafter"); }
        std::vector<std::uint32_t> lanes, at;
        for (std::size_t j = 0; j < sequences.size(); ++j) {
            const std::uint32_t s = sequences[j];
            if (s >= options.sequences ||
                std::count(sequences.begin(), sequences.begin() + std::ptrdiff_t(j), s) != 0) {
                throw std::invalid_argument("flint draft: invalid sequence");
            }
            lanes.push_back(spine->lane(s));
            at.push_back(positions[s]);
        }
        spine->draft(lanes, anchors, at, k, drafts, confidence, stream);
    }

    void warm_up() {
        // The prompt route once (kernels load at first launch), then each decode batch's graph and
        // each verification width's. A DSpine drafter derives its table, then drafts at every
        // width once.
        if (spine) { spine->prepare(stream); }
        const std::vector<std::int32_t> prompt(std::min<std::uint32_t>(options.prefill_chunk, 16), 2);
        forward(0, prompt, 1);
        if (spine) {
            const std::uint32_t s = 0;
            const std::int32_t anchor = 2;
            for (std::uint32_t k = 1; k <= options.draft_tokens; ++k) {
                std::vector<std::int32_t> drafts(k);
                std::vector<float> confidence(k);
                draft({&s, 1}, {&anchor, 1}, k, drafts, confidence);
            }
        }
        positions[0] = 0;
        std::vector<std::uint32_t> sequences;
        std::vector<std::int32_t> tokens;
        for (std::uint32_t b = 1; b <= std::min<std::uint32_t>(options.sequences, 8); ++b) {
            sequences.push_back(b - 1);
            tokens.push_back(2);
            for (int pass = 0; pass < 2; ++pass) { // eagerly, then captured
                decode(sequences, tokens);
                for (const std::uint32_t s : sequences) { positions[s] = 0; }
            }
        }
        if (options.draft_tokens > 0) {
            const std::uint32_t s = 0;
            std::vector<std::int32_t> row(options.draft_tokens + 1, 2);
            for (int pass = 0; pass < 2; ++pass) {
                verify({&s, 1}, row);
                const std::uint32_t one = 1;
                commit({&s, 1}, {&one, 1});
                positions[0] = 0;
            }
        }
        CUDA_CHECK(ninfer::wait_for_stream(stream));
        if (delivery) {
            // Drained, so warm-up deliveries and their asynchronous prefetches stay out of the
            // request counters, and the cache starts from the routing profile rather than the
            // warm-up's routes. Runtime readers use the same mutex and report request work only.
            delivery->drain();
            delivery->prime(options.expert_profile ? &*options.expert_profile : nullptr);
            std::lock_guard lock(expert_stats_mutex);
            expert_startup_stats = delivery->stats();
        }
    }

    // A sliding block's ring buffers of one sequence, (start, bytes): K and V.
    std::array<std::pair<std::uint8_t*, std::size_t>, 2> ring_buffers(const LayerPlan& p,
                                                                      std::uint32_t sequence) const {
        const std::size_t bytes = std::size_t(p.seq_stride) * sizeof(__half);
        return {{{static_cast<std::uint8_t*>(p.k.p) + sequence * bytes, bytes},
                 {static_cast<std::uint8_t*>(p.v.p) + sequence * bytes, bytes}}};
    }

    void copy_rings(std::uint32_t sequence, bool save) {
        if (!options.checkpoints) { throw std::logic_error("flint: an executor without checkpoints"); }
        if (spine) {
            const std::uint32_t live = spine->lane(sequence), kept = spine->lane(sequence, 1);
            spine->copy_lane(save ? live : kept, save ? kept : live, stream);
        }
        for (const LayerPlan& p : layers) {
            if (p.full) { continue; }
            const auto rings = ring_buffers(p, sequence);
            for (std::size_t i = 0; i < rings.size(); ++i) {
                auto* kept = static_cast<std::uint8_t*>(p.checkpoint[i].p) + sequence * rings[i].second;
                CUDA_CHECK(cudaMemcpyAsync(save ? kept : rings[i].first, save ? rings[i].first : kept,
                                           rings[i].second, cudaMemcpyDeviceToDevice, stream));
            }
        }
    }

    [[nodiscard]] std::uint64_t image_bytes(std::uint32_t position) const {
        std::uint64_t bytes = 0;
        for (const LayerPlan& p : layers) {
            if (p.full) {
                bytes += 2 * std::uint64_t(kv_heads) * position * head_dim * sizeof(__half);
            } else {
                for (const auto& ring : ring_buffers(p, 0)) { bytes += ring.second; }
            }
        }
        return bytes + (spine ? spine->image_bytes(position) : 0);
    }

    // Copies a sequence's image between the device and the host pages: the rings (from or to its
    // checkpoint with `checkpoint`) and the full-attention rows below `position`.
    void copy_image(std::uint32_t sequence, std::uint32_t position, std::span<std::byte* const> pages,
                    std::size_t page_bytes, bool save, bool checkpoint) {
        if (page_bytes == 0 || pages.size() < (image_bytes(position) + page_bytes - 1) / page_bytes) {
            throw std::invalid_argument("flint image: pages do not hold the image");
        }
        std::size_t at = 0;
        const auto copy = [&](std::uint8_t* device, std::size_t bytes) {
            while (bytes > 0) {
                const std::size_t in_page = std::min(bytes, page_bytes - at % page_bytes);
                std::byte* host           = pages[at / page_bytes] + at % page_bytes;
                if (save) {
                    CUDA_CHECK(cudaMemcpyAsync(host, device, in_page, cudaMemcpyDeviceToHost, stream));
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(device, host, in_page, cudaMemcpyHostToDevice, stream));
                }
                device += in_page;
                bytes -= in_page;
                at += in_page;
            }
        };
        const std::size_t head_bytes = std::size_t(global_slots) * head_dim * sizeof(__half);
        for (const LayerPlan& p : layers) {
            if (p.full) {
                for (const KvArea* area : {&p.k, &p.v}) {
                    auto* base = static_cast<std::uint8_t*>(area->p) + sequence * head_bytes * kv_heads;
                    for (int h = 0; h < kv_heads; ++h) {
                        copy(base + h * head_bytes, std::size_t(position) * head_dim * sizeof(__half));
                    }
                }
                continue;
            }
            const auto rings = ring_buffers(p, sequence);
            for (std::size_t i = 0; i < rings.size(); ++i) {
                std::uint8_t* device =
                    checkpoint ? static_cast<std::uint8_t*>(p.checkpoint[i].p) + sequence * rings[i].second
                               : rings[i].first;
                copy(device, rings[i].second);
            }
        }
        if (spine) { spine->copy_image(spine->lane(sequence, checkpoint ? 1 : 0), position, copy); }
    }

    SequenceImage save_image(std::uint32_t sequence, const SequenceSnapshot* at,
                             std::span<std::byte* const> pages, std::size_t page_bytes) {
        const std::uint32_t position = at != nullptr ? at->position : positions.at(sequence);
        copy_image(sequence, position, pages, page_bytes, true, at != nullptr);
        return {position};
    }

    void load_image(std::uint32_t sequence, const SequenceImage& image,
                    std::span<std::byte* const> pages, std::size_t page_bytes) {
        if (image.position > options.max_context) {
            throw std::invalid_argument("flint load_image: invalid image");
        }
        back(sequence, image.position, false);
        copy_image(sequence, image.position, pages, page_bytes, false, false);
        positions.at(sequence)  = image.position;
        ring_floor.at(sequence) = image.position;
    }

    void snapshot(std::uint32_t sequence, SequenceSnapshot& out) {
        copy_rings(sequence, true);
        out.position = positions[sequence];
    }

    void restore(std::uint32_t sequence, const SequenceSnapshot& from) {
        copy_rings(sequence, false);
        positions[sequence]  = from.position;
        ring_floor[sequence] = from.position;
    }

    // A position's window lies in the rings while the sequence has written fewer positions past
    // it than the rings hold beyond the window and a round's drafts, and the rings held the
    // window before it when they last came whole. The full-attention rows are kept by position.
    bool rewind(std::uint32_t sequence, std::uint32_t tokens) {
        const std::uint32_t at = positions.at(sequence);
        const std::int64_t slack = std::int64_t(ring) - window - std::int64_t(options.draft_tokens);
        if (tokens > at || at - tokens < ring_floor[sequence] || std::int64_t(tokens) > slack) {
            return false;
        }
        positions[sequence] = at - tokens;
        return true;
    }
};

Executor::Executor(const Model& model, DeviceContext& device, ExecutorOptions options)
    : impl_(std::make_unique<Impl>(model, device, options)) {}

Executor::~Executor() = default;

const ExecutorOptions& Executor::options() const noexcept { return impl_->options; }

ExecutorMemory Executor::memory() const noexcept {
    return ExecutorMemory{.kv_bytes        = impl_->kv_bytes,
                          .kv_backed_bytes = impl_->kv_backed.load(std::memory_order_relaxed),
                          .kv_grows        = impl_->growing,
                          .workspace_bytes = impl_->workspace_bytes + impl_->mask_full.bytes,
                          .drafter_bytes   = impl_->spine ? impl_->spine->device_bytes() : 0,
                          .experts = impl_->delivery ? impl_->delivery->memory() : ExpertDeliveryMemory{}};
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

void Executor::reset(std::uint32_t sequence) {
    impl_->positions.at(sequence)  = 0;
    impl_->ring_floor.at(sequence) = 0;
    impl_->release(sequence);
}

std::uint32_t Executor::position(std::uint32_t sequence) const {
    return impl_->positions.at(sequence);
}

void Executor::warm_up() {
    impl_->warm_up();
    check_experts();
}

void Executor::check_experts() const {
    if (impl_->delivery) { impl_->delivery->check(); }
}
ExpertDeliveryStats Executor::expert_stats() const {
    if (!impl_->delivery) { return {}; }
    std::lock_guard lock(impl_->expert_stats_mutex);
    return impl_->delivery->stats() - impl_->expert_startup_stats;
}
ExpertRouteCounts Executor::expert_totals() const {
    return impl_->delivery ? impl_->delivery->totals() : ExpertRouteCounts{};
}
std::string Executor::expert_cpu_description() const {
    return impl_->delivery ? impl_->delivery->cpu_description() : std::string();
}

void Executor::forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                       std::uint32_t logit_rows) {
    impl_->forward(sequence, tokens, logit_rows);
}

void Executor::forward_branches(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                                std::span<const std::uint32_t> lengths) {
    impl_->forward(sequence, tokens, static_cast<std::uint32_t>(lengths.size()), lengths);
}

void Executor::forward_prompt_branches(std::uint32_t sequence,
                                       std::span<const std::int32_t> prompt,
                                       std::span<const std::int32_t> tokens,
                                       std::span<const std::uint32_t> lengths) {
    std::vector<std::int32_t> chunk(prompt.begin(), prompt.end());
    chunk.insert(chunk.end(), tokens.begin(), tokens.end());
    impl_->forward(sequence, chunk, static_cast<std::uint32_t>(lengths.size()), lengths,
                   static_cast<std::uint32_t>(prompt.size()));
}

void Executor::decode(std::span<const std::uint32_t> sequences,
                      std::span<const std::int32_t> tokens) {
    impl_->decode(sequences, tokens);
}

Tensor Executor::logits(std::uint32_t rows) const {
    if (rows == 0 || rows > kMaxLogitRows) {
        throw std::invalid_argument("flint logits: invalid row count");
    }
    return Tensor(impl_->logits_bf16.p, DType::BF16,
                  {impl_->vocab, static_cast<std::int32_t>(rows)});
}

cudaStream_t Executor::head_stream() const noexcept { return impl_->stream; }

std::uint32_t Executor::draft_tokens() const noexcept { return impl_->options.draft_tokens; }

void Executor::verify(std::span<const std::uint32_t> sequences,
                      std::span<const std::int32_t> tokens, std::span<const std::uint32_t> widths) {
    impl_->verify(sequences, tokens, widths);
}

void Executor::commit(std::span<const std::uint32_t> sequences,
                      std::span<const std::uint32_t> kept) {
    impl_->commit(sequences, kept);
}

void Executor::draft(std::span<const std::uint32_t> sequences,
                     std::span<const std::int32_t> anchors, std::uint32_t k,
                     std::span<std::int32_t> drafts, std::span<float> confidence) {
    impl_->draft(sequences, anchors, k, drafts, confidence);
}

void Executor::snapshot(std::uint32_t sequence, SequenceSnapshot& out) {
    impl_->snapshot(sequence, out);
}

void Executor::restore(std::uint32_t sequence, const SequenceSnapshot& from) {
    impl_->restore(sequence, from);
}

bool Executor::rewind(std::uint32_t sequence, std::uint32_t tokens) {
    return impl_->rewind(sequence, tokens);
}

} // namespace ninfer::models::flint
