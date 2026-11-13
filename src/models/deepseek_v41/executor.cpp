#include "models/deepseek_v41/executor.h"

#include "core/decode_graph.h"
#include "core/device_buffer.h"
#include "core/device_wait.h"
#include "core/expert_parallel.h"
#include "core/growable_region.h"
#include "core/pinned_memory.h"
#include "core/qtype.h"
#include "core/tensor_parallel.h"
#include "core/tensor_split.h"
#include "core/vmm.h"
#include "core/weight_view.h"
#include "models/deepseek_v41/cpu_experts.h"
#include "models/deepseek_v41/kernels.h"
#include "models/deepseek_v41/vision.h"
#include "models/divided_head.h"
#include "models/dspine/runtime.h"
#include "ninfer/ops/linear_policy.h"
#include "ninfer/ops/weight_input.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"
#include "ops/linear/stq/stq_linear.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <numbers>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::deepseek_v41 {
namespace {

namespace gguf = ops::gguf;

constexpr std::size_t kAlign = 256;
constexpr int kLatent        = 512; // the attention latent: key and value of every head
constexpr int kIndexDim      = 128; // the indexer's head width
constexpr int kTailSlots     = 16;  // a ratio-2 compressor's kept rows per sequence
constexpr int kRing          = 256; // a sequence's window ring: the window and a round's drafts
// A sequence's window ring, and a compressed position's latent, key and key scales (kernels.h).
constexpr std::size_t kRingBytes          = std::size_t(kRing) * kernels::kWindowRowBytes;
constexpr std::size_t kCompressedRowBytes = std::size_t(kernels::kLatentRowBytes) +
                                            kernels::kKeyRowBytes + kernels::kKeyScaleBytes;
constexpr int kCandidateBits = 32;
// Index scores of a prompt chunk are computed for as many rows at once as this many bytes hold.
constexpr std::size_t kScoreBudget = std::size_t(64) << 20;

std::size_t round_up(std::size_t value, std::size_t to) { return (value + to - 1) / to * to; }

const float* fp32(const Model& model, WeightId id, std::int64_t count) {
    const Tensor tensor = weight_tensor(model.weight(id).view, {std::int32_t(count)});
    if (tensor.dtype != DType::FP32) {
        throw std::invalid_argument(model.weight(id).name + " must be stored as fp32");
    }
    return static_cast<const float*>(tensor.data);
}

const __nv_bfloat16* bf16_table(const Model& model, WeightId id, std::int64_t rows, std::int64_t columns) {
    const Tensor tensor = weight_tensor(model.weight(id).view, {std::int32_t(columns), std::int32_t(rows)});
    if (tensor.dtype != DType::BF16) {
        throw std::invalid_argument(model.weight(id).name + " must be stored as bf16");
    }
    return static_cast<const __nv_bfloat16*>(tensor.data);
}

// The device address of a direct weight's first element, checked to be stored as `format`.
const std::byte* direct_words(const Model& model, WeightId id, QType format) {
    const auto region = contiguous_weight_region(model.weight(id).view);
    const auto& g     = region.parent->geometry;
    if (g.layout != QuantLayout::Contiguous || g.format != format || !region.parent->data) {
        throw std::invalid_argument(model.weight(id).name + ": unexpected direct representation");
    }
    return region.parent->data + region.begin * 2;
}

// An STQ drafter's matrices of k inputs and n outputs (`count` of them, each its own trellis and
// signs, or one).
ops::stq::Matrices stq_matrices(const Model& model, const StqWeights& w, int k, int n,
                                ops::stq::Codebook codebook, int count = 1) {
    ops::stq::Matrices m;
    m.trellis  = reinterpret_cast<const std::uint16_t*>(direct_words(model, w.trellis, QType::UINT16));
    m.suh      = reinterpret_cast<const __half*>(direct_words(model, w.suh, QType::FP16));
    m.svh      = reinterpret_cast<const __half*>(direct_words(model, w.svh, QType::FP16));
    m.k        = k;
    m.n        = n;
    m.codebook = codebook;
    m.trellis_stride = std::int64_t(k / 16) * (n / 16) * ops::stq::tile_words(codebook);
    m.suh_stride     = k;
    m.svh_stride     = n;
    for (const auto& [id, elements] : {std::pair{w.trellis, m.trellis_stride * count},
                                       std::pair{w.suh, std::int64_t(k) * count},
                                       std::pair{w.svh, std::int64_t(n) * count}}) {
        if (weight_element_count(model.weight(id).view.shape) != std::uint64_t(elements)) {
            throw std::invalid_argument(model.weight(id).name + ": STQ shape mismatch");
        }
    }
    return m;
}

Weight matrix(const Model& model, WeightId id) {
    const Weight w = ops::prepare_linear_weight(model.input(id)).weight;
    if (!is_gguf(w.qtype)) {
        throw std::invalid_argument(model.weight(id).name + ": DeepSeek-V4.1 runs GGUF block matrices");
    }
    return w;
}

// The matrices of `ids`: one when they are rows of one stored parent, else each on its own.
std::vector<Weight> matrices(const Model& model, std::initializer_list<WeightId> ids) {
    std::vector<ops::WeightInput> inputs;
    for (const WeightId id : ids) { inputs.push_back(model.input(id)); }
    std::vector<Weight> out;
    if (inputs.size() > 1 && ops::joins(inputs)) {
        out.push_back(ops::prepare_linear_weight(std::span<const ops::WeightInput>(inputs)).weight);
    } else {
        for (const auto& input : inputs) { out.push_back(ops::prepare_linear_weight(input).weight); }
    }
    for (const Weight& w : out) {
        if (!is_gguf(w.qtype)) { throw std::invalid_argument("DeepSeek-V4.1 runs GGUF block matrices"); }
    }
    return out;
}

std::int64_t row_bytes(const Weight& w) {
    const auto block = gguf_block_shape(w.qtype);
    return std::int64_t(w.k / block.elements) * block.bytes;
}

// The rotation frequencies of the reference (precompute_freqs_cis) in FP32: base^(-2i/dim), and
// with YaRN the long wavelengths divided by `factor`, a linear ramp between the dimensions whose
// wavelengths complete beta_fast and beta_slow turns over the original context.
std::vector<float> rope_freq(int dim, float base, std::uint32_t original, float factor,
                             float beta_fast, float beta_slow) {
    std::vector<float> f(std::size_t(dim) / 2);
    for (std::size_t i = 0; i < f.size(); ++i) {
        f[i] = 1.0f / std::pow(base, float(2 * i) / float(dim));
    }
    if (original == 0) { return f; }
    const auto corrected = [&](double rotations) {
        return dim * std::log(double(original) / (rotations * 2 * std::numbers::pi)) /
               (2 * std::log(double(base)));
    };
    const double low  = std::max(std::floor(corrected(beta_fast)), 0.0);
    const double high = std::min(std::ceil(corrected(beta_slow)), double(dim - 1));
    const float span  = float(std::max(high - low, 1e-3));
    for (std::size_t i = 0; i < f.size(); ++i) {
        const float ramp   = std::clamp((float(i) - float(low)) / span, 0.0f, 1.0f);
        const float smooth = 1.0f - ramp;
        f[i]               = f[i] / factor * (1.0f - smooth) + f[i] * smooth;
    }
    return f;
}

// A context-sized cache plane: a row of `row_bytes` per sequence, `entry_bytes` for every `ratio`
// token positions. Owned whole, or, beside an elastic expert cache, a growable region whose rows
// are backed only as far as their sequence has reached (Impl::back).
struct KvArea {
    DeviceBuffer owned;
    GrowableRegion region;
    void* p = nullptr;
    std::size_t bytes = 0, row_bytes = 0, entry_bytes = 0, ratio = 1;
    std::vector<std::uint32_t> backed; // positions backed, by sequence

    // The bytes of a row that its first `positions` positions reach.
    [[nodiscard]] std::size_t extent(std::uint32_t positions) const {
        return std::min(row_bytes, (std::size_t(positions) + ratio - 1) / ratio * entry_bytes);
    }

    KvArea() = default;
    KvArea(std::size_t owners, std::size_t row, std::size_t entry, std::size_t per, bool growing,
           int device)
        : bytes(owners * row), row_bytes(row), entry_bytes(entry), ratio(per), backed(owners, 0) {
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

// One hyper-connection site's weights.
struct Site {
    const float *fn = nullptr, *base = nullptr, *scale = nullptr;
};

// An STQ drafter block's matrices (DSparkConfig::stq), in place of the GGUF ones.
struct StqPlan {
    ops::stq::Matrices query_a, kv, query_b, output_a, output_b, shared_gate, shared_up, shared_down;
    ops::stq::Matrices gate, up, down; // the routed experts
};

struct LayerPlan {
    int ratio = 0;
    int experts = 0, top_k = 0; // the block's routed experts and how many a row takes
    bool kv_source = false, index_source = false, candidate_source = false, uses_candidates = false;
    int kv_from = -1; // the kv source whose caches the block reads
    bool decoder = false; // after the last kv source: the prefill tail may skip its rows
    Site hc_attention, hc_ffn;
    std::vector<Weight> query_kv; // [query_a; kv] as one parent, or the two
    Weight query_b, output_a, output_b, shared_down;
    DeviceBuffer output_groups; // output_a's row groups: the experts of the grouped decode product
    std::vector<Weight> shared_gate_up;
    const float *attention_norm = nullptr, *ffn_norm = nullptr, *query_norm = nullptr,
                *kv_norm = nullptr, *sinks = nullptr, *freq = nullptr;
    // kv source: compressor and indexer key; index source: indexer query and head weights.
    const float *compressor_kv = nullptr, *compressor_gate = nullptr, *compressor_norm = nullptr,
                *index_key = nullptr, *index_key_norm = nullptr, *index_query = nullptr,
                *index_weights = nullptr;
    const float *router = nullptr, *router_bias = nullptr;
    const float* image_bias = nullptr; // with the vision tower: the router's bias of image tokens
    gguf::GgmlType gate_up_type{}, down_type{};
    std::int64_t gate_up_row_bytes = 0, down_row_bytes = 0;
    DeviceBuffer tables; // device experts: [E] gate, [E] up, [E] gate_up, [E] down
    // A drafter block under --draft-experts: its kept experts' weights, router rows and biases.
    DeviceBuffer kept_experts, kept_router;
    int delivery_layer = -1;
    int engram         = -1; // index among the Engram blocks
    Weight engram_projection;
    const float* engram_gate = nullptr;
    // Every sequence's window ring, [sequences][kRing] rows of kWindowRowBytes (kernels.h).
    DeviceBuffer ring;
    // kv source: the compressed latents [sequences][rows] of kLatentRowBytes, the index keys'
    // codes [sequences][rows] of kKeyRowBytes and their scales of kKeyScaleBytes, rows =
    // max_context / ratio; ratio 2: the tails [sequences][kTailSlots][512] FP32 of kv and score.
    KvArea latents, keys, key_scales;
    int rows = 0;
    DeviceBuffer tail_kv, tail_score;
    // The checkpoint of every sequence's ring and tails (snapshot()).
    DeviceBuffer checkpoint_ring, checkpoint_kv, checkpoint_score;
    std::optional<StqPlan> stq;
};

struct GraphEntry {
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    bool warmed = false; // ran once eagerly, so every kernel is loaded before the capture
};

// The rows of a pass: a prompt chunk of one sequence, or decode rows of one sequence each. A
// chunk of a prompt span reads its staged rows from `row` of the span's `staged` rows.
struct Pass {
    int t     = 0;
    bool chunk = false;
    int row = 0, staged = 0; // staged 0: the pass's own t rows
    std::uint32_t sequence = 0, first = 0;
    // The rows whose logits the pass keeps (the last ones), and the logit column of the first.
    int logit_rows = 0, logit_column = 0;
    // With the drafter: whether the pass's tapped rows seed its rings now (a verification's wait for
    // commit()). A draft block's rows attend as if at `visible` (each the block's last position) over
    // `extra_window` keys past the window.
    bool seed = true;
    const std::int32_t* visible = nullptr;
    int extra_window = 0;
    // A decode or verification pass's listed candidate blocks (candidate_lists, a row alone a
    // group), which its candidate source lists for the later index sources; a chunk lists its own
    // rows' at each of them.
    std::int32_t* candidate_list   = nullptr;
    std::int32_t* candidate_counts = nullptr;
    // A prompt call's first row the decoder blocks run (t: none, under a prefill tail), and its
    // first row the drafter taps (-1: its last window of rows).
    int decoder_row = 0, tap_row = -1;
    // A prompt call without logit rows: its last block leaves only its window latents (and a kv
    // source's compressed rows), since no row reads that block's output.
    bool kv_only = false;
    // A prompt call's first row a DSpine drafter keeps (t: none).
    int spine_row = 0;
};

} // namespace

struct Executor::Impl {
    const Model& model;
    DeviceContext& device;
    ExecutorOptions options;
    const TextConfig& config;
    cudaStream_t stream = nullptr;
    int hidden = 0, streams = 0, heads = 0, vocab = 0, window = 0, query_rank = 0, groups = 0,
        output_rank = 0, index_heads = 0, index_topk = 0, experts = 0, top_k = 0, width = 0;
    float eps = 0;
    int sms   = 0; // the device's multiprocessors, the STQ products' split
    // A prompt chunk attends over a staging ring of the window before it and the chunk itself,
    // which every block and sequence shares in turn.
    int prefill_ring = 0;
    DeviceBuffer staging;
    // Beside an elastic expert cache, the compressed caches grow as far as the sequences reach.
    bool growing = false;
    std::vector<KvArea*> sequence_areas;
    std::atomic<std::uint64_t> kv_backed{0};
    std::uint64_t kv_fixed = 0, kv_bytes = 0;
    std::uint64_t drafter_expert_bytes = 0; // see ExecutorMemory
    std::vector<LayerPlan> layers;
    // The CPU's share of missing experts; it outlives the delivery that calls it.
    std::unique_ptr<CpuExperts> cpu_experts;
    std::unique_ptr<ExpertDelivery> delivery;
    // Expert parallelism (several device ranks): rank r serves experts expert_ranges[r] of every
    // block through a delivery of its own (rank 0's is `delivery`); every other rank has a stream
    // and workspace for its part of a MoE call, and the exchange carries a call's input there and
    // the part back (routed_parallel()). Without options.expert_split the ranges start equal and
    // size_expert_ranges() sets them before the deliveries are planned.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> expert_ranges;
    bool split_by_room = false;

    struct ExpertPeer {
        cudaStream_t stream = nullptr;
        std::unique_ptr<WorkspaceArena> workspace;
        std::unique_ptr<ExpertDelivery> delivery;
        ExpertDeliveryStats startup;
    };

    std::vector<ExpertPeer> expert_peers; // by rank; rank 0's is unused
    std::unique_ptr<ExpertParallelExchange> expert_exchange;
    DeviceBuffer expert_pack; // a call's packed input, on rank 0
    // The head divided over the ranks for calls of up to kMaxVectorColumns logit columns; every
    // rank holds a range of its rows in proportion to its copy bandwidth.
    std::unique_ptr<models::DividedHead> divided_head;
    mutable std::mutex expert_stats_mutex;
    ExpertDeliveryStats expert_startup_stats;
    Weight embedding, head;
    const float* final_norm = nullptr;
    DeviceBuffer freq_window, freq_compressed;
    // The grouped output projection's routing for each decode width t (group_routing(t)).
    DeviceBuffer group_routes;
    std::vector<std::size_t> group_routes_at;
    std::vector<std::uint32_t> positions;
    // The prefill tail (ExecutorOptions::prefill_tail): the first decoder block, the rows before a
    // position that decide the decoder blocks' state there (exact_rows), the rows a prompt call
    // runs before its kept state (tail_rows, 0: every row) and the rows after which a run's state
    // counts as complete. Per sequence: the first position of its decoder blocks' current run, the
    // window floor of their chunks (0: a complete state), and whether a prompt call left them
    // behind its position.
    int decoder_block = 0;
    std::uint32_t exact_rows = 0, tail_rows = 0, complete_rows = 0;
    std::vector<std::uint32_t> decoder_from;
    std::vector<std::uint8_t> decoder_behind;
    DeviceBuffer d_tokens, d_positions, d_sequences;
    std::unique_ptr<PinnedHostBuffer> h_staging;
    std::unique_ptr<CudaCompletionEvent> staged;
    DeviceBuffer logits_bf16;
    std::unique_ptr<WorkspaceArena> workspace;
    std::size_t workspace_bytes = 0;
    // Layer-major prompt spans (forward_span): with host or disk experts, every chunk of a span
    // passes a block before any passes the next, so the block's wide calls copy the experts the
    // cache lacks once per span instead of once per chunk. span_tokens is 0 when spans are off.
    // Beside an elastic expert cache a span borrows its streams, coefficients and selections from
    // the cache (span_buffers) and gives them back after it, so decode keeps the whole cache; a
    // fixed cache leaves them in the workspace. Under expert parallelism the span's buffers stay on
    // rank 0 and its MoE calls keep the chunk's width, so every rank's delivery fills its banks once
    // per block of the span.
    static constexpr std::uint32_t kSpanChunks = 8;
    std::uint32_t span_tokens = 0;
    std::size_t span_borrow_bytes = 0;
    std::map<std::uint32_t, GraphEntry> graphs;
    // Rows per sequence of the last verify() and each sequence's first row.
    std::vector<std::uint32_t> verify_widths;
    std::vector<int> verify_starts;
    int capacity_rows = 0;   // the largest compressed cache, a decode pass's score width
    int candidate_words = 0; // words of a row's candidate mask
    int candidate_keep = 0, candidate_block = 0;
    int candidate_rows = 0; // the positions a row's candidate mask covers
    // The Engram: the hash, each sequence's history, and the rows of a pass (pinned ids and rows,
    // then on the device: [Engram block][rows of the pass][columns] rows of 144 bytes).
    EngramHasher hasher;
    std::unique_ptr<qwen4_exp::NgramTableReader> engram;
    std::vector<EngramHasher::History> histories;
    std::vector<std::vector<std::int32_t>> verified; // a verification's tokens until commit
    // A pass's ids and rows go through one of two pinned sets (the second only with
    // ExecutorOptions::engram_read_ahead), each with its last copy to the device; engram_set is
    // the set the passes read into, the other the read-ahead's.
    struct EngramSet {
        std::unique_ptr<PinnedHostBuffer> ids, rows;
        std::unique_ptr<CudaCompletionEvent> copied;
    };
    std::array<EngramSet, 2> engram_sets;
    int engram_set = 0;
    DeviceBuffer d_engram;
    std::vector<std::uint64_t> engram_scratch;
    // The read-ahead: a batch in the reader (pending) or read (ready) for the prompt call of
    // `sequence` at `first` with these tokens, image flags and history before them.
    struct EngramAhead {
        bool pending = false, ready = false;
        std::uint32_t sequence = 0, first = 0;
        std::vector<std::int32_t> tokens;
        std::vector<std::uint8_t> dead;
        EngramHasher::History history;
    } ahead;
    // One sequence's rings and tails, the state a branch pass returns to.
    DeviceBuffer branch_state;
    // Every sequence's held state (hold()): its rings and tails, state() order, branch_bytes each.
    DeviceBuffer hold_state;
    std::size_t hold_bytes = 0;
    // The vision tower, and a prompt step's images: their rows' flags (pinned, then on the
    // device, one byte per staged row), the pieces of their span rows a step places, the device
    // rows a piece passes through (one image span at most, also the tower's output), the patches
    // of an image being encoded; the current pass's flags (null without image rows).
    std::unique_ptr<VisionTower> tower;
    std::unique_ptr<PinnedHostBuffer> h_image;
    std::unique_ptr<CudaCompletionEvent> image_staged;
    DeviceBuffer d_image, d_image_rows, d_patches;
    std::vector<ImageRows> step_images;
    std::vector<std::uint8_t> step_dead;
    const std::uint8_t* image_mask = nullptr;
    // Decode attention splits: one count for every row count, so a row sums its keys in the same
    // order however many rows its pass decodes (a verified draft's values are plain decoding's).
    int decode_splits = 1;
    // The DSpark drafter (Model::dspark()): its blocks (each with every sequence's window ring),
    // the backbone blocks it taps (target_of: a block's index among the targets, -1 for none), the
    // taps of a pass's last rows ([window][targets][hidden] FP32, the streams' mean entering each
    // target), the seed's kv rows of each block, and a draft's buffers: the anchor and the drafts,
    // their confidences and the Markov steps' scratch on the device, the drafts and confidences in
    // pinned memory.
    struct Drafter {
        std::vector<LayerPlan> layers;
        std::vector<int> target_of;
        int targets = 0, block = 0, domain = 0;
        std::int32_t noise = 0;
        Weight fc;
        std::vector<Weight> kv;
        std::optional<ops::stq::Matrices> stq_fc; // an STQ drafter's projection (d.layers' stq: kv)
        const float *tap_norm = nullptr, *final_norm = nullptr, *confidence = nullptr;
        const __nv_bfloat16 *markov_embed = nullptr, *markov_head = nullptr;
        DeviceBuffer taps, ids, values, scratch;
        std::unique_ptr<PinnedHostBuffer> host;
        std::unique_ptr<CudaCompletionEvent> copied;
    };
    std::optional<Drafter> drafter;
    // A standalone DSpine drafter (ExecutorOptions::dspine): its ring, table and features, and the
    // FP32 rows a tap's streams' mean passes through on its way there. Branch passes keep nothing
    // in it (in_branches).
    std::unique_ptr<dspine::Runtime> spine;
    DeviceBuffer spine_stage;
    bool in_branches = false;

    Impl(const Model& m, DeviceContext& d, ExecutorOptions o)
        : model(m), device(d), options(std::move(o)), config(m.config()),
          hasher(m.config().engram, m.engram_token_map()) {
        if (options.sequences == 0 || options.prefill_chunk == 0 || options.max_context == 0) {
            throw std::invalid_argument("deepseek_v41 executor: invalid options");
        }
        if (!options.expert_split.empty() || device.size() > 1) {
            if (model.expert_banks().empty()) {
                throw std::invalid_argument("DeepSeek-V4.1 divides its experts over the devices "
                                            "from host memory or the files "
                                            "(--expert-residency host|disk)");
            }
            if (device.size() < 2 ||
                (!options.expert_split.empty() && options.expert_split.size() != device.size())) {
                throw std::invalid_argument("expert parallelism needs one share per device rank");
            }
            split_by_room = options.expert_split.empty();
            set_expert_ranges(split_by_room ? std::vector<double>(device.size(), 1.0)
                                            : options.expert_split);
        }
        if (options.draft_tokens > std::uint32_t(gguf::kMaxVectorColumns) - 1) {
            throw std::invalid_argument("deepseek_v41 executor: at most 7 drafts a round");
        }
        stream              = device.stream;
        sms                 = device.multiprocessor_count();
        const int device_id = device.rank(0).device;
        growing = !model.expert_banks().empty() && !options.expert_cache_bytes &&
                  vmm::granularity(device_id) != 0;
        hidden      = int(config.hidden_size);
        streams     = int(config.hc_mult);
        heads       = int(config.num_attention_heads);
        vocab       = int(config.vocab_size);
        window      = int(config.sliding_window);
        query_rank  = int(config.q_lora_rank);
        groups      = int(config.o_groups);
        output_rank = int(config.o_lora_rank);
        index_heads = int(config.index_heads);
        index_topk  = int(config.index_topk);
        experts     = int(config.num_experts);
        top_k       = int(config.top_k_experts);
        width       = int(config.moe_intermediate_size);
        eps         = config.rms_norm_eps;
        options.prefill_chunk = std::min<std::uint32_t>(options.prefill_chunk, 4096);
        prefill_ring          = int(round_up(std::size_t(window) + options.prefill_chunk, 256));
        positions.assign(options.sequences, 0);
        histories.assign(options.sequences, {});
        verified.assign(options.sequences, {});
        const auto& w = model.weights();
        embedding     = native_weight(model.weight(w.token_embedding).view);
        head          = matrix(model, w.output_head);
        final_norm    = fp32(model, w.final_norm, hidden);
        if (!is_gguf(embedding.qtype)) { throw std::invalid_argument("DeepSeek-V4.1: GGUF token table"); }
        if (options.dspine && (model.dspine() == nullptr || model.dspark() ||
                               options.draft_tokens == 0 ||
                               options.draft_tokens + 1 > model.dspine()->config().block_size)) {
            throw std::invalid_argument("deepseek_v41 executor: DSpine drafting needs the model's "
                                        "DSpine drafter alone and 1 to its block less one drafts "
                                        "a round");
        }
        {
            const auto window_freq = rope_freq(int(config.rope_dim), config.rope_theta, 0, 1, 0, 0);
            const auto yarn_freq   = rope_freq(int(config.rope_dim), config.compress_rope_theta,
                                               config.yarn_original, config.yarn_factor,
                                               config.yarn_beta_fast, config.yarn_beta_slow);
            freq_window     = DeviceBuffer(window_freq.size() * 4);
            freq_compressed = DeviceBuffer(yarn_freq.size() * 4);
            freq_window.copy_from_host(window_freq.data(), window_freq.size() * 4);
            freq_compressed.copy_from_host(yarn_freq.data(), yarn_freq.size() * 4);
        }
        {
            // Per width t: bounds, sorted pairs, active groups and their count (see group_routing).
            std::vector<std::int32_t> words;
            group_routes_at.assign(std::size_t(gguf::kMaxVectorColumns) + 1, 0);
            for (int t = 1; t <= gguf::kMaxVectorColumns; ++t) {
                group_routes_at[std::size_t(t)] = words.size();
                for (int g = 0; g <= groups; ++g) { words.push_back(g * t); }
                for (int g = 0; g < groups; ++g) {
                    for (int j = 0; j < t; ++j) { words.push_back(j * groups + g); }
                }
                for (int g = 0; g < groups; ++g) { words.push_back(g); }
                words.push_back(groups);
            }
            group_routes = DeviceBuffer(words.size() * sizeof(std::int32_t));
            group_routes.copy_from_host(words.data(), words.size() * sizeof(std::int32_t));
        }
        const int sequences = int(options.sequences);
        {
            // The caches are sized up front for every sequence's whole context: refuse with the
            // numbers rather than fail inside an allocation.
            std::size_t needed = 0;
            for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
                needed += (options.checkpoints ? 2 : 1) * std::size_t(sequences) * kRingBytes;
                if (config.kv_source[i]) {
                    const std::size_t rows = options.max_context / config.compress_ratios[i];
                    needed += std::size_t(sequences) * rows * kCompressedRowBytes;
                }
            }
            if (model.dspark()) {
                needed += model.dspark()->layers.size() * (options.checkpoints ? 2 : 1) *
                              std::size_t(sequences) * kRingBytes +
                          std::size_t(window) * model.dspark_config()->target_layers.size() * hidden * 4;
            }
            needed += std::size_t(prefill_ring) * kernels::kWindowRowBytes;
            if (options.dspine) {
                needed += dspine::Runtime::planned_bytes(*model.dspine(), DType::FP32,
                                                         spine_options()) +
                          spine_stage_bytes();
            }
            if (options.checkpoints) {
                // The held state (hold()): another copy of every sequence's rings.
                needed += (config.num_hidden_layers +
                           (model.dspark() ? model.dspark()->layers.size() : 0)) *
                          std::size_t(sequences) * kRingBytes;
            }
            std::size_t free_bytes = 0, total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            const std::size_t reserve = start_reserve();
            if (needed + reserve > free_bytes) {
                throw InsufficientMemory(
                    "DeepSeek-V4.1: the caches and checkpoints of " + std::to_string(sequences) +
                    " sequence(s) of " + std::to_string(options.max_context) + " tokens need " +
                    std::to_string(needed >> 20) + " MiB beside " + std::to_string(reserve >> 20) +
                    " MiB for the workspace and CUDA graphs, and " +
                    std::to_string(free_bytes >> 20) +
                    " MiB are free after the weights; lower --max-concurrency or --max-context");
            }
        }
        const std::size_t ring_bytes = kRingBytes;
        const std::size_t tail_bytes = std::size_t(kTailSlots) * kLatent * 4;
        std::size_t branch_bytes     = 0;
        for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
            const LayerWeights& lw = w.layers[i];
            LayerPlan p = plan_block(lw, int(config.compress_ratios[i]), experts, top_k);
            p.kv_source        = config.kv_source[i];
            p.index_source     = config.index_source[i];
            p.candidate_source = config.candidate_source == int(i);
            p.uses_candidates  = p.index_source && config.candidate_source >= 0 &&
                                config.candidate_source < int(i);
            p.kv_from = p.ratio > 0 ? config.kv_source_of(i) : -1;
            if (p.ratio > 0 && (p.kv_from < 0 || config.index_source_of(i) < 0)) {
                throw std::invalid_argument("DeepSeek-V4.1: a compressed block before its sources");
            }
            const auto& at = lw.attention;
            if (p.kv_source) {
                p.compressor_kv = fp32(model, *at.compressor_kv, std::int64_t(kLatent) * hidden);
                if (p.ratio > 1) {
                    p.compressor_gate = fp32(model, *at.compressor_gate, std::int64_t(kLatent) * hidden);
                }
                p.compressor_norm = fp32(model, *at.compressor_norm, kLatent);
                p.index_key       = fp32(model, *at.index_key, std::int64_t(kIndexDim) * kLatent);
                p.index_key_norm  = fp32(model, *at.index_key_norm, kIndexDim);
            }
            if (p.index_source) {
                p.index_query   = fp32(model, *at.index_query,
                                       std::int64_t(index_heads) * kIndexDim * query_rank);
                p.index_weights = fp32(model, *at.index_weights, std::int64_t(index_heads) * hidden);
            }
            if (model.expert_residency() == ExpertResidency::Device) {
                plan_resident(p, lw.moe);
            } else {
                p.delivery_layer    = int(i);
                const auto& bank    = model.expert_banks()[i];
                p.gate_up_type      = ops::detail::gguf_type(bank.gate_up_type);
                p.down_type         = ops::detail::gguf_type(bank.down_type);
                p.gate_up_row_bytes = std::int64_t(bank.gate_up_row_bytes);
                p.down_row_bytes    = std::int64_t(bank.down_row_bytes);
            }
            if (lw.engram) {
                p.engram            = config.engram.index(i);
                p.engram_projection = matrix(model, lw.engram->projection);
                p.engram_gate       = fp32(model, lw.engram->gate, std::int64_t(streams) * hidden);
            }
            p.ring = DeviceBuffer(ring_bytes * sequences);
            p.ring.fill(0);
            kv_bytes += p.ring.bytes;
            branch_bytes += ring_bytes;
            if (options.checkpoints) {
                p.checkpoint_ring = DeviceBuffer(ring_bytes * sequences);
                kv_bytes += p.checkpoint_ring.bytes;
            }
            if (p.kv_source) {
                p.rows = int(options.max_context / std::uint32_t(p.ratio));
                const auto area = [&](int entry) {
                    return KvArea(std::size_t(sequences), std::size_t(p.rows) * entry, std::size_t(entry),
                                  std::size_t(p.ratio), growing, device_id);
                };
                p.latents    = area(kernels::kLatentRowBytes);
                p.keys       = area(kernels::kKeyRowBytes);
                p.key_scales = area(kernels::kKeyScaleBytes);
                kv_bytes += p.latents.bytes + p.keys.bytes + p.key_scales.bytes;
                capacity_rows = std::max(capacity_rows, p.rows);
                if (p.ratio > 1) {
                    p.tail_kv    = DeviceBuffer(tail_bytes * sequences);
                    p.tail_score = DeviceBuffer(tail_bytes * sequences);
                    p.tail_kv.fill(0);
                    p.tail_score.fill(0);
                    kv_bytes += 2 * p.tail_kv.bytes;
                    branch_bytes += 2 * tail_bytes;
                    if (options.checkpoints) {
                        p.checkpoint_kv    = DeviceBuffer(tail_bytes * sequences);
                        p.checkpoint_score = DeviceBuffer(tail_bytes * sequences);
                        kv_bytes += 2 * p.checkpoint_kv.bytes;
                    }
                }
            }
            if (p.candidate_source) {
                candidate_block = int(config.candidate_block_size);
                candidate_keep  = int(config.candidate_blocks);
                const int blocks = (p.rows + candidate_block - 1) / candidate_block;
                candidate_words  = (blocks + kCandidateBits - 1) / kCandidateBits;
                candidate_rows   = p.rows;
            }
            layers.push_back(std::move(p));
        }
        plan_prefill_tail();
        if (model.dspark()) { plan_drafter(ring_bytes, branch_bytes); }
        staging = DeviceBuffer(std::size_t(prefill_ring) * kernels::kWindowRowBytes);
        staging.fill(0);
        kv_bytes += staging.bytes;
        branch_state = DeviceBuffer(branch_bytes);
        kv_bytes += branch_state.bytes;
        if (options.checkpoints) {
            hold_bytes = branch_bytes;
            hold_state = DeviceBuffer(branch_bytes * std::size_t(sequences));
            kv_bytes += hold_state.bytes;
        }
        for (LayerPlan& p : layers) {
            if (!p.kv_source) { continue; }
            sequence_areas.push_back(&p.latents);
            sequence_areas.push_back(&p.keys);
            sequence_areas.push_back(&p.key_scales);
        }
        kv_fixed = kv_bytes;
        if (growing) {
            for (KvArea* area : sequence_areas) { kv_fixed -= area->bytes; }
        }
        kv_backed = kv_fixed;
        decode_splits = std::clamp(2 * sms / (heads / 16), 1, 40);
        span_tokens = plan_span();
        const std::size_t rows = staging_rows();
        d_tokens    = DeviceBuffer(rows * 4);
        d_positions = DeviceBuffer(rows * 4);
        d_sequences = DeviceBuffer(rows * 4);
        h_staging   = std::make_unique<PinnedHostBuffer>(3 * rows * 4);
        staged      = std::make_unique<CudaCompletionEvent>(device);
        logits_bf16 = DeviceBuffer(std::size_t(vocab) * kMaxLogitRows * 2);
        if (options.engram_table) {
            const std::size_t engram_rows = rows * hasher.rows_per_token();
            engram = std::make_unique<qwen4_exp::NgramTableReader>(*options.engram_table,
                                                                   options.engram_read);
            for (int i = 0; i < (options.engram_read_ahead ? 2 : 1); ++i) {
                engram_sets[i].ids    = std::make_unique<PinnedHostBuffer>(engram_rows * 8);
                engram_sets[i].rows   = std::make_unique<PinnedHostBuffer>(engram_rows * kEngramRowBytes);
                engram_sets[i].copied = std::make_unique<CudaCompletionEvent>(device);
            }
            d_engram = DeviceBuffer(engram_rows * kEngramRowBytes);
            engram_scratch.resize(engram_rows);
        }
        if (options.vision_max_patches > 0) { plan_vision(rows); }
        span_borrow_bytes = growing && span_tokens != 0 ? span_buffers(span_tokens) : 0;
        workspace_bytes   = plan_workspace(growing ? 0 : span_tokens);
        if (tower) { workspace_bytes = std::max(workspace_bytes, tower->workspace_bytes()); }
        {
            std::size_t free_bytes = 0, total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            if (workspace_bytes + kGraphReserve + unbacked() > free_bytes) {
                throw InsufficientMemory(
                    "DeepSeek-V4.1: the workspace of " + std::to_string(workspace_bytes >> 20) +
                    " MiB and " + std::to_string(kGraphReserve >> 20) +
                    " MiB for the CUDA graphs need more than the " + std::to_string(free_bytes >> 20) +
                    " MiB free after the caches; lower --prefill-chunk, --max-concurrency or "
                    "--max-context");
            }
        }
        workspace = std::make_unique<WorkspaceArena>(workspace_bytes);
        if (options.dspine) { make_spine(); }
        allocate_expert_peers();
        if (split_by_room) { size_expert_ranges(); }
        if (!model.expert_banks().empty()) { plan_delivery(); }
    }

    // ---- A DSpine drafter ----------------------------------------------------------------------

    // Its state: a lane per sequence and one per checkpoint; the taps of a prompt step's last
    // window of rows (a span's when spans may run) or of a decode or verification call's rows; and
    // proposals for up to eight sequences at once.
    [[nodiscard]] dspine::RuntimeOptions spine_options() const {
        const dspine::Config& c    = model.dspine()->config();
        const std::uint32_t window = dspine::ring_capacity(c, options.max_context);
        const std::uint32_t step =
            model.expert_banks().empty()
                ? options.prefill_chunk
                : std::max(options.prefill_chunk,
                           std::min(options.max_context, kSpanChunks * options.prefill_chunk));
        dspine::RuntimeOptions o;
        o.sequences       = options.sequences;
        o.kept_states     = options.checkpoints ? 1 : 0;
        o.max_context     = options.max_context;
        o.draft_tokens    = options.draft_tokens;
        o.draft_rows      = std::min<std::uint32_t>(options.sequences, 8);
        o.commit_rows     = std::min<std::uint32_t>(options.sequences, 8);
        o.feature_columns = std::max<std::uint32_t>(std::min(step, window),
                                                    std::uint32_t(gguf::kMaxVectorColumns));
        const ops::detail::GgufShape shape{head.qtype, head.n, head.k};
        o.head_bytes = ops::detail::gguf_project_workspace_bytes(
                           {&shape, 1}, 1, static_cast<std::int32_t>(o.draft_tokens * o.draft_rows)) +
                       kAlign;
        return o;
    }

    // The FP32 rows a tap's streams' mean passes through: a store's piece of columns.
    [[nodiscard]] std::size_t spine_stage_bytes() const {
        const dspine::RuntimeOptions o = spine_options();
        return std::size_t(hidden) * std::min(o.piece_columns, o.feature_columns) * 4;
    }

    // The drafter reads the target's token rows (its GGUF embedding, before the streams) and its
    // head as FP32 logits, on rank 0.
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
            Tensor rows = x; // scaled in place where the head's activation keeps F16 sums
            project_on(rows, head, static_cast<float*>(logits.data), work, on);
        };
        spine_stage = DeviceBuffer(spine_stage_bytes());
        spine = std::make_unique<dspine::Runtime>(*model.dspine(), std::move(target),
                                                  spine_options());
    }

    // Whether the drafter reads the last block's output (a pass that keeps only the last block's
    // window latents would not compute it).
    [[nodiscard]] bool spine_taps_last() const {
        return spine && spine->tap_of(std::uint32_t(layers.size() - 1)) >= 0;
    }

    // Block `layer`'s output for rows [lo, hi) of the streams h ([T][streams][H]), when the drafter
    // taps it: their streams' mean into feature columns [lo - base, hi - base).
    void spine_tap(std::size_t layer, const float* h, int lo, int hi, int base) {
        if (!spine || lo >= hi) { return; }
        const int tap = spine->tap_of(std::uint32_t(layer));
        if (tap < 0) { return; }
        const int piece = int(spine_stage.bytes / (std::size_t(hidden) * 4));
        for (int row = lo; row < hi; row += piece) {
            const int m = std::min(piece, hi - row);
            kernels::stream_mean(h + std::size_t(row) * streams * hidden, m, streams, hidden,
                                 static_cast<float*>(spine_stage.p), hidden, stream);
            spine->store_tap(std::uint32_t(tap),
                             Tensor(spine_stage.p, DType::FP32, {hidden, m}),
                             std::uint32_t(row - base), stream);
        }
    }

    // The first of a prompt call's t rows at `first` that a DSpine drafter keeps: those within its
    // window before the state the caller keeps (tap_start()); none in a branch pass, which the
    // sequence does not keep.
    [[nodiscard]] int spine_start(std::uint32_t first, int t, std::uint32_t state_at) const {
        if (!spine || in_branches) { return t; }
        const std::int64_t at = state_at == 0 ? std::int64_t(first) + t : std::int64_t(state_at);
        return int(std::clamp<std::int64_t>(at - std::int64_t(spine->window()) - std::int64_t(first),
                                            0, t));
    }

    void draft(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> anchors,
               std::uint32_t k, std::span<std::int32_t> drafts, std::span<float> confidence) {
        if (!spine) { throw std::logic_error("deepseek_v41: an executor without a DSpine drafter"); }
        std::vector<std::uint32_t> lanes, at;
        for (std::size_t j = 0; j < sequences.size(); ++j) {
            const std::uint32_t s = sequences[j];
            if (s >= options.sequences ||
                std::count(sequences.begin(), sequences.begin() + std::ptrdiff_t(j), s) != 0) {
                throw std::invalid_argument("deepseek_v41 draft: invalid sequence");
            }
            require_resumable(s, "draft");
            lanes.push_back(spine->lane(s));
            at.push_back(positions[s]);
        }
        spine->draft(lanes, anchors, at, k, drafts, confidence, stream);
    }

    // The vision tower and the image buffers: a step's flags, a span's rows, an image's patches.
    void plan_vision(std::size_t rows) {
        if (!model.vision()) {
            throw std::invalid_argument("deepseek_v41 executor: images need the model's vision tower");
        }
        tower = std::make_unique<VisionTower>(model, options.vision_max_patches);
        const VisionConfig& vc = tower->config();
        const float* bias = fp32(model, model.vision()->image_router_bias,
                                 std::int64_t(config.num_hidden_layers) * experts);
        for (std::size_t i = 0; i < layers.size(); ++i) {
            layers[i].image_bias = bias + i * std::size_t(experts);
        }
        h_image      = std::make_unique<PinnedHostBuffer>(rows);
        image_staged = std::make_unique<CudaCompletionEvent>(device);
        d_image      = DeviceBuffer(rows);
        d_image_rows = DeviceBuffer(std::size_t(vc.max_image_tokens) * hidden * 2);
        d_patches    = DeviceBuffer(std::size_t(options.vision_max_patches) * vc.patch_features() * 2);
    }

    [[nodiscard]] std::uint64_t vision_owned_bytes() const noexcept {
        if (!tower) { return 0; }
        return tower->owned_bytes() + d_image.bytes + d_image_rows.bytes + d_patches.bytes;
    }

    // What every block plans, the backbone's and the drafter's: hyper-connections, the window
    // attention, the norms, the router and the shared expert; an STQ drafter block's matrices from
    // `stq` (its routed experts too).
    LayerPlan plan_block(const LayerWeights& lw, int ratio, int block_experts, int block_top_k,
                         const StqLayerWeights* stq = nullptr) {
        LayerPlan p;
        p.ratio                  = ratio;
        p.experts                = block_experts;
        p.top_k                  = block_top_k;
        const std::int64_t mixes = config.hc_mixes(), flat = std::int64_t(streams) * hidden;
        p.hc_attention = {fp32(model, lw.hc_attention.fn, mixes * flat),
                          fp32(model, lw.hc_attention.base, mixes), fp32(model, lw.hc_attention.scale, 3)};
        p.hc_ffn       = {fp32(model, lw.hc_ffn.fn, mixes * flat), fp32(model, lw.hc_ffn.base, mixes),
                          fp32(model, lw.hc_ffn.scale, 3)};
        const auto& at   = lw.attention;
        if (stq != nullptr) {
            using ops::stq::Codebook;
            constexpr auto trunk = Codebook::Spec1W15N64;
            const int latent = heads * kLatent, latent_rows = groups * output_rank;
            StqPlan& m       = p.stq.emplace();
            m.query_a     = stq_matrices(model, stq->query_a, hidden, query_rank, trunk);
            m.kv          = stq_matrices(model, stq->kv, hidden, kLatent, trunk);
            m.query_b     = stq_matrices(model, stq->query_b, query_rank, latent, trunk);
            m.output_a    = stq_matrices(model, stq->output_a, latent / groups, output_rank, trunk, groups);
            m.output_b    = stq_matrices(model, stq->output_b, latent_rows, hidden, trunk);
            m.shared_gate = stq_matrices(model, stq->shared_gate, hidden, width, trunk);
            m.shared_up   = stq_matrices(model, stq->shared_up, hidden, width, trunk);
            m.shared_down = stq_matrices(model, stq->shared_down, width, hidden, trunk);
            m.gate = stq_matrices(model, stq->gate, hidden, width, Codebook::Pair1W16N32, block_experts);
            m.up   = stq_matrices(model, stq->up, hidden, width, Codebook::Pair1W16N32, block_experts);
            m.down = stq_matrices(model, stq->down, width, hidden, Codebook::Pair1W16N32, block_experts);
        } else {
            p.query_kv = matrices(model, {at.query_a, at.kv});
            p.query_b  = matrix(model, at.query_b);
            p.output_a = matrix(model, at.output_a);
            p.output_b = matrix(model, at.output_b);
        }
        if (stq == nullptr) {
            std::vector<const void*> parts(static_cast<std::size_t>(groups));
            for (int g = 0; g < groups; ++g) {
                parts[std::size_t(g)] = static_cast<const std::byte*>(p.output_a.qdata) +
                                        std::int64_t(g) * output_rank * row_bytes(p.output_a);
            }
            p.output_groups = DeviceBuffer(parts.size() * sizeof(void*));
            p.output_groups.copy_from_host(parts.data(), parts.size() * sizeof(void*));
        }
        p.attention_norm = fp32(model, lw.attention_norm, hidden);
        p.ffn_norm       = fp32(model, lw.ffn_norm, hidden);
        p.query_norm     = fp32(model, at.query_a_norm, query_rank);
        p.kv_norm        = fp32(model, at.kv_norm, kLatent);
        p.sinks          = fp32(model, at.sinks, heads);
        p.freq = static_cast<const float*>(p.ratio > 0 ? freq_compressed.p : freq_window.p);
        const MoeWeights& mw = lw.moe;
        p.router             = fp32(model, mw.router, std::int64_t(block_experts) * hidden);
        p.router_bias        = fp32(model, mw.router_bias, block_experts);
        if (stq == nullptr) {
            p.shared_gate_up = matrices(model, {mw.shared_gate, mw.shared_up});
            p.shared_down    = matrix(model, mw.shared_down);
        }
        return p;
    }

    // Device experts: the tables of each expert's gate, up, gate_up and down rows.
    void plan_resident(LayerPlan& p, const MoeWeights& mw) {
        const Weight gate_up = matrix(model, *mw.gate_up);
        const Weight down    = matrix(model, *mw.down);
        p.gate_up_type       = ops::detail::gguf_type(gate_up.qtype);
        p.down_type          = ops::detail::gguf_type(down.qtype);
        p.gate_up_row_bytes  = row_bytes(gate_up);
        p.down_row_bytes     = row_bytes(down);
        const int e          = p.experts;
        std::vector<const void*> table(4 * std::size_t(e));
        for (int x = 0; x < e; ++x) {
            const auto* gu = static_cast<const std::byte*>(gate_up.qdata) +
                             std::int64_t(x) * 2 * width * p.gate_up_row_bytes;
            table[x]         = gu;
            table[e + x]     = gu + std::int64_t(width) * p.gate_up_row_bytes;
            table[2 * e + x] = gu;
            table[3 * e + x] = static_cast<const std::byte*>(down.qdata) +
                               std::int64_t(x) * hidden * p.down_row_bytes;
        }
        p.tables = DeviceBuffer(table.size() * sizeof(void*));
        p.tables.copy_from_host(table.data(), table.size() * sizeof(void*));
    }

    // --draft-experts: a drafter block's kept experts (Model::dspark_kept, ascending), copied from
    // the pinned host weights to one device buffer with the kernels' zero tails, and their router
    // rows and biases, so the block routes among them as experts 0..N-1. Startup only.
    void keep_drafter_experts(LayerPlan& p, const LayerWeights& lw, std::span<const std::int32_t> kept) {
        const int n           = int(kept.size());
        constexpr std::size_t kAlign = 256, kTail = 512;
        const auto aligned    = [](std::size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; };
        p.kept_router         = DeviceBuffer(aligned(std::size_t(n) * hidden * 4) + std::size_t(n) * 4);
        auto* router          = static_cast<float*>(p.kept_router.p);
        auto* bias            = reinterpret_cast<float*>(static_cast<std::byte*>(p.kept_router.p) +
                                                         aligned(std::size_t(n) * hidden * 4));
        for (int i = 0; i < n; ++i) {
            const auto x = std::size_t(kept[std::size_t(i)]);
            CUDA_CHECK(cudaMemcpy(router + std::size_t(i) * hidden, p.router + x * hidden,
                                  std::size_t(hidden) * 4, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(bias + i, p.router_bias + x, 4, cudaMemcpyDeviceToDevice));
        }
        p.router      = router;
        p.router_bias = bias;
        p.experts     = n;
        // Each run: `bytes` per expert at `source` (host), expert x at source + x * bytes.
        struct Run {
            const std::byte* source;
            std::size_t bytes;
            std::byte* target = nullptr;
        };
        std::vector<Run> runs;
        if (p.stq) {
            for (ops::stq::Matrices* m : {&p.stq->gate, &p.stq->up, &p.stq->down}) {
                runs.push_back({reinterpret_cast<const std::byte*>(m->trellis),
                                std::size_t(m->trellis_stride) * 2});
                runs.push_back({reinterpret_cast<const std::byte*>(m->suh), std::size_t(m->k) * 2});
                runs.push_back({reinterpret_cast<const std::byte*>(m->svh), std::size_t(m->n) * 2});
            }
        } else {
            const Weight gate_up = matrix(model, *lw.moe.gate_up);
            const Weight down    = matrix(model, *lw.moe.down);
            p.gate_up_type       = ops::detail::gguf_type(gate_up.qtype);
            p.down_type          = ops::detail::gguf_type(down.qtype);
            p.gate_up_row_bytes  = row_bytes(gate_up);
            p.down_row_bytes     = row_bytes(down);
            runs.push_back({static_cast<const std::byte*>(gate_up.qdata),
                            std::size_t(2 * width) * std::size_t(p.gate_up_row_bytes)});
            runs.push_back({static_cast<const std::byte*>(down.qdata),
                            std::size_t(hidden) * std::size_t(p.down_row_bytes)});
        }
        std::size_t total = 0;
        for (const Run& run : runs) { total += aligned(run.bytes * std::size_t(n) + kTail); }
        p.kept_experts = DeviceBuffer(total);
        p.kept_experts.fill(0);
        auto* at = static_cast<std::byte*>(p.kept_experts.p);
        for (Run& run : runs) {
            run.target = at;
            for (int i = 0; i < n; ++i) {
                CUDA_CHECK(cudaMemcpy(at + std::size_t(i) * run.bytes,
                                      run.source + std::size_t(kept[std::size_t(i)]) * run.bytes,
                                      run.bytes, cudaMemcpyHostToDevice));
            }
            at += aligned(run.bytes * std::size_t(n) + kTail);
        }
        if (p.stq) {
            std::size_t r = 0;
            for (ops::stq::Matrices* m : {&p.stq->gate, &p.stq->up, &p.stq->down}) {
                m->trellis = reinterpret_cast<const std::uint16_t*>(runs[r++].target);
                m->suh     = reinterpret_cast<const __half*>(runs[r++].target);
                m->svh     = reinterpret_cast<const __half*>(runs[r++].target);
            }
            return;
        }
        std::vector<const void*> table(4 * std::size_t(n));
        for (int x = 0; x < n; ++x) {
            const std::byte* gu = runs[0].target + std::size_t(x) * runs[0].bytes;
            table[std::size_t(x)]         = gu;
            table[std::size_t(n + x)]     = gu + std::int64_t(width) * p.gate_up_row_bytes;
            table[std::size_t(2 * n + x)] = gu;
            table[std::size_t(3 * n + x)] = runs[1].target + std::size_t(x) * runs[1].bytes;
        }
        p.tables = DeviceBuffer(table.size() * sizeof(void*));
        p.tables.copy_from_host(table.data(), table.size() * sizeof(void*));
    }

    // The drafter's blocks with their rings (and checkpoints), its tables and its buffers.
    void plan_drafter(std::size_t ring_bytes, std::size_t& branch_bytes) {
        const DSparkConfig& dc = *model.dspark_config();
        const DSparkWeights& dw = *model.dspark();
        Drafter& d              = drafter.emplace();
        d.targets               = int(dc.target_layers.size());
        d.block                 = int(dc.block_size);
        d.noise                 = dc.noise_token;
        d.domain                = int(model.resources().public_token_count);
        d.target_of.assign(config.num_hidden_layers, -1);
        for (int j = 0; j < d.targets; ++j) { d.target_of[dc.target_layers[std::size_t(j)]] = j; }
        if (d.block < 1 || std::uint32_t(d.block) > std::uint32_t(gguf::kMaxVectorColumns) ||
            d.domain <= 0 || d.domain > vocab) {
            throw std::invalid_argument("DeepSeek-V4.1: the DSpark block must be 1..8 rows over the vocabulary");
        }
        if (dc.stq) {
            d.stq_fc = stq_matrices(model, *dw.stq_fc, d.targets * hidden, hidden,
                                    ops::stq::Codebook::Spec1W15N64);
        } else {
            d.fc = matrix(model, dw.fc);
        }
        d.tap_norm   = fp32(model, dw.tap_norm, hidden);
        d.final_norm = fp32(model, dw.final_norm, hidden);
        d.confidence = fp32(model, dw.confidence, std::int64_t(hidden) + kernels::kMarkovRank);
        d.markov_embed = bf16_table(model, dw.markov_embed, vocab, kernels::kMarkovRank);
        d.markov_head  = bf16_table(model, dw.markov_head, vocab, kernels::kMarkovRank);
        const int sequences = int(options.sequences);
        for (std::size_t b = 0; b < dw.layers.size(); ++b) {
            const LayerWeights& lw = dw.layers[b];
            LayerPlan p = plan_block(lw, 0, int(dc.experts), int(dc.top_k),
                                     dc.stq ? &dw.stq_layers.at(b) : nullptr);
            const bool kept = !model.dspark_kept().empty();
            if (kept) {
                keep_drafter_experts(p, lw, model.dspark_kept().at(b));
                drafter_expert_bytes += p.kept_experts.bytes + p.kept_router.bytes;
            }
            if (!dc.stq) {
                if (!kept) { plan_resident(p, lw.moe); }
                // The seed's kv rows: of [query_a; kv] when the two are one parent.
                Weight kv = p.query_kv.back();
                if (p.query_kv.size() == 1) {
                    kv.qdata = static_cast<const std::byte*>(kv.qdata) +
                               std::int64_t(query_rank) * row_bytes(kv);
                    kv.n = kLatent;
                }
                d.kv.push_back(kv);
            }
            p.ring = DeviceBuffer(ring_bytes * sequences);
            p.ring.fill(0);
            kv_bytes += p.ring.bytes;
            branch_bytes += ring_bytes;
            if (options.checkpoints) {
                p.checkpoint_ring = DeviceBuffer(ring_bytes * sequences);
                kv_bytes += p.checkpoint_ring.bytes;
            }
            d.layers.push_back(std::move(p));
        }
        d.taps    = DeviceBuffer(std::size_t(window) * d.targets * hidden * 4);
        d.ids     = DeviceBuffer(std::size_t(d.block + 1) * 4);
        d.values  = DeviceBuffer(std::size_t(d.block) * 4);
        d.scratch = DeviceBuffer(kernels::markov_scratch_bytes());
        d.host    = std::make_unique<PinnedHostBuffer>(std::size_t(2 * d.block) * 4);
        d.copied  = std::make_unique<CudaCompletionEvent>(device);
        kv_bytes += d.taps.bytes;
    }

    // A whole-layer bank of the routed experts (`experts` of them: a rank's share under expert
    // parallelism), each blob at its largest aligned size.
    [[nodiscard]] std::uint64_t bank_bytes(std::uint32_t experts = 0) const {
        std::vector<std::uint64_t> sizes;
        for (const auto& bank : model.expert_banks()) {
            for (const auto& expert : bank.delivery.experts) {
                sizes.resize(std::max(sizes.size(), expert.size()), 0);
                for (std::size_t b = 0; b < expert.size(); ++b) {
                    sizes[b] = std::max(sizes[b], std::uint64_t(round_up(
                        expert[b].bytes + expert[b].zero_tail, kAlign)));
                }
            }
        }
        std::uint64_t slot = 0;
        for (auto bytes : sizes) { slot += bytes; }
        return slot * (experts != 0 ? experts : config.num_experts);
    }
    [[nodiscard]] std::uint64_t delivery_metadata() const {
        return std::uint64_t(options.prefill_chunk) * config.top_k_experts * 8 +
               std::uint64_t(config.num_hidden_layers) * config.num_experts * 4 * sizeof(void*) +
               config.num_experts * 4;
    }

    // The banks' experts [first, last) of every block: one rank's delivery sources.
    [[nodiscard]] std::vector<ExpertDeliveryLayer> delivery_sources(std::uint32_t first,
                                                                    std::uint32_t last) const {
        std::vector<ExpertDeliveryLayer> sources;
        for (const auto& bank : model.expert_banks()) {
            ExpertDeliveryLayer layer;
            layer.aliases = bank.delivery.aliases;
            layer.experts.assign(bank.delivery.experts.begin() + first,
                                 bank.delivery.experts.begin() + last);
            sources.push_back(std::move(layer));
        }
        return sources;
    }

    void plan_delivery() {
        const auto [first, last] =
            expert_ranges.empty() ? std::pair<std::uint32_t, std::uint32_t>{0, config.num_experts}
                                  : expert_ranges[0];
        std::vector<ExpertDeliveryLayer> sources = delivery_sources(first, last);
        // One whole-layer bank at least (the transport adds a second where memory allows).
        const auto bank     = bank_bytes(last - first);
        const auto metadata = delivery_metadata();
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        const auto cache = options.expert_cache_bytes.value_or(0);
        if (bank + metadata + cache + unbacked() + kGraphReserve > free) {
            throw InsufficientMemory("DeepSeek-V4.1: expert delivery needs " +
                std::to_string((bank + metadata + cache) >> 20) + " MiB beside " +
                std::to_string(unbacked() >> 20) +
                " MiB for the growing caches and the CUDA Graph reserve after workspace allocation");
        }
        ExpertDeliveryOptions transport;
        transport.pair_capacity   = options.prefill_chunk * config.top_k_experts;
        transport.cache_bytes     = options.expert_cache_bytes;
        transport.keep_free_bytes = kGraphReserve + (growing ? kKvHeadroom : 0);
        transport.elastic         = growing;
        transport.disk            = options.expert_disk;
        const float dma_share     = options.expert_dma_share.value_or(0.0F);
        if (!(dma_share >= 0 && dma_share <= 1) || options.expert_cpu_threads > 256) {
            throw std::invalid_argument("DeepSeek-V4.1: the DMA share must be 0..1 and CPU threads 0..256");
        }
        std::vector<CpuExpertLayer> cpu_layers;
        bool computes = false;
        for (const auto& b : model.expert_banks()) {
            cpu_layers.push_back({b.gate_up_type, b.down_type, std::int64_t(b.gate_up_row_bytes),
                                  std::int64_t(b.down_row_bytes)});
            computes = computes || CpuExperts::computes(cpu_layers.back(), hidden, width);
        }
        // A disk expert the CPU computes needs the same file read as a copied one; on an RTX 4090
        // host every split of disk experts decoded slower than copying every miss, so the CPU
        // computes disk experts only at an explicit share.
        // Under expert parallelism every rank copies all of its misses.
        const bool cpu_share =
            expert_ranges.empty() &&
            (options.expert_dma_share ? dma_share < 1
                                      : model.expert_residency() == ExpertResidency::Host);
        if (computes && cpu_share) {
            cpu_experts = std::make_unique<CpuExperts>(std::move(cpu_layers), hidden, width,
                int(config.top_k_experts), config.swiglu_limit);
            transport.cpu.products      = cpu_experts.get();
            transport.cpu.input_bytes   = cpu_experts->input_bytes();
            transport.cpu.product_bytes = cpu_experts->product_bytes();
            transport.cpu.share         = 1.0 - double(dma_share);
            transport.cpu.balance       = !options.expert_dma_share;
            transport.cpu.threads       = options.expert_cpu_threads;
        }
        delivery = std::make_unique<ExpertDelivery>(device.rank(0), model.expert_files(),
                                                    std::move(sources), transport);
        for (std::size_t r = 1; r < expert_peers.size(); ++r) {
            const auto [begin, end] = expert_ranges[r];
            RankBinding bind(device, r);
            std::size_t free_r = 0, total_r = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_r, &total_r));
            if (bank_bytes(end - begin) + metadata + cache + kGraphReserve > free_r) {
                throw InsufficientMemory(
                    "DeepSeek-V4.1: expert delivery of device rank " + std::to_string(r) +
                    " needs " + std::to_string((bank_bytes(end - begin) + metadata + cache) >> 20) +
                    " MiB beside the CUDA Graph reserve, and " + std::to_string(free_r >> 20) +
                    " MiB are free");
            }
            ExpertDeliveryOptions peer;
            peer.pair_capacity   = transport.pair_capacity;
            peer.cache_bytes     = options.expert_cache_bytes;
            peer.keep_free_bytes = kGraphReserve;
            peer.disk            = options.expert_disk;
            // The delivery serves the calls on the rank's expert-parallel stream.
            RankContext rank         = device.rank(r);
            rank.stream              = expert_peers[r].stream;
            expert_peers[r].delivery = std::make_unique<ExpertDelivery>(
                rank, model.expert_files(), delivery_sources(begin, end), peer);
        }
    }

    void set_expert_ranges(std::span<const double> weights) {
        const auto ends =
            split_shares(config.num_experts, unit_positions(config.num_experts, 1), weights);
        expert_ranges.clear();
        for (std::size_t r = 0; r + 1 < ends.size(); ++r) {
            expert_ranges.emplace_back(std::uint32_t(ends[r]), std::uint32_t(ends[r + 1]));
        }
    }

    // Every routed expert's bytes in a device slot: what a cache of all of them would hold.
    [[nodiscard]] std::uint64_t expert_bytes() const {
        std::uint64_t bytes = 0;
        for (const auto& bank : model.expert_banks()) {
            for (const auto& expert : bank.delivery.experts) {
                for (const auto& blob : expert) {
                    bytes += round_up(blob.bytes + blob.zero_tail, kAlign);
                }
            }
        }
        return bytes;
    }

    // Expert parallelism without --tensor-split: each rank's share of the experts follows the
    // memory its cache gets beside what the rank holds now (rank 0 the model, its caches and
    // workspace; every rank the exchange) and its bus speed, so that every rank's misses take the
    // same time (delivery_shares). Rank 0's cache counts as the delivery starts it, before the
    // growing caches take memory back; the layer banks are lent to the cache while decoding.
    void size_expert_ranges() {
        const std::uint64_t metadata = delivery_metadata();
        const auto free_bytes        = rank_free_bytes(device);
        std::vector<double> room(device.size());
        for (std::size_t r = 0; r < device.size(); ++r) {
            const std::uint64_t keep =
                kGraphReserve + metadata + (r == 0 && growing ? kKvHeadroom : 0);
            room[r] = options.expert_cache_bytes ? double(*options.expert_cache_bytes)
                      : free_bytes[r] > keep      ? double(free_bytes[r] - keep)
                                                  : 0.0;
        }
        set_expert_ranges(delivery_shares(room, rank_host_copy_bandwidth(device),
                                          double(expert_bytes()), rank_copy_bandwidth(device)));
    }

    // A rank's slice [first, last) of a routing profile ([layer][expert]).
    [[nodiscard]] std::optional<ExpertRouteCounts> profile_slice(std::uint32_t first,
                                                                 std::uint32_t last) const {
        if (!options.expert_profile) { return std::nullopt; }
        ExpertRouteCounts out;
        for (const auto& layer : *options.expert_profile) {
            out.emplace_back(layer.begin() + first, layer.begin() + last);
        }
        return out;
    }

    // ---- Expert parallelism
    // ----------------------------------------------------------------------

    // The byte layout of a MoE call's packed input: the activation the experts read (up to 8
    // tokens: their q8_1 vector form; wider: BF16 [T][H]), the guard's factors [T], then the ids
    // and weights, 256-aligned.
    struct ExpertInput {
        std::size_t activation = 0, scale = 0, ids = 0, weights = 0, bytes = 0;
    };

    [[nodiscard]] ExpertInput expert_input(int t) const {
        const std::size_t pairs = std::size_t(t) * config.top_k_experts;
        ExpertInput out;
        out.activation = t <= gguf::kMaxVectorColumns ? gguf::vector_activation_bytes(hidden, t)
                                                      : std::size_t(t) * hidden * 2;
        out.scale      = round_up(out.activation, kAlign);
        out.ids        = out.scale + round_up(std::size_t(t) * 4, kAlign);
        out.weights    = out.ids + round_up(pairs * 4, kAlign);
        out.bytes      = out.weights + pairs * 4;
        return out;
    }

    // The most tokens one MoE call takes (a span's calls take a chunk at a time).
    [[nodiscard]] int max_moe_tokens() const {
        return std::max<int>(int(options.prefill_chunk), gguf::kMaxVectorColumns);
    }

    // Workspace of one rank's part of a call of up to `t` tokens (expert_part()).
    [[nodiscard]] std::size_t expert_part_bytes(int t) const {
        const std::size_t pairs = std::size_t(t) * config.top_k_experts;
        const std::size_t vector_pairs =
            std::size_t(gguf::kMaxVectorColumns) * config.top_k_experts;
        const std::size_t h = hidden, w = width;
        const std::size_t bytes =
            round_up(pairs * 4, kAlign) +
            round_up(gguf::moe_routing_bytes(int(config.num_experts), int(pairs)), kAlign) +
            round_up(pairs * 2 * w * 4, kAlign) + round_up(pairs * w * 2, kAlign);
        const std::size_t vector =
            round_up(gguf::vector_activation_bytes(int(w), int(vector_pairs)), kAlign);
        const std::size_t matrix =
            round_up(std::max(gguf::moe_matrix_activation_bytes(int(h), int(pairs)),
                              gguf::vector_activation_bytes(int(h), t)),
                     kAlign) +
            round_up(std::max(gguf::moe_matrix_activation_bytes(int(w), int(pairs)),
                              gguf::vector_activation_bytes(int(w), int(pairs))),
                     kAlign) +
            round_up(pairs * h * 4, kAlign);
        return bytes + std::max(vector, matrix) + (std::size_t(1) << 20);
    }

    // The exchange and every other rank's stream and workspace, before the deliveries take what
    // memory is left for their caches.
    void allocate_expert_peers() {
        if (expert_ranges.empty()) { return; }
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
        // Before the deliveries take what memory is left for their caches.
        if (models::DividedHead::divides(head, device.size())) {
            const int columns = gguf::kMaxVectorColumns;
            divided_head      = std::make_unique<models::DividedHead>(
                device, head, row_bytes(head), rank_copy_bandwidth(device),
                std::size_t(columns) * hidden * 2, columns);
        }
    }

    static constexpr std::size_t kGraphReserve = std::size_t(320) << 20;
    static constexpr std::size_t kKvHeadroom   = std::size_t(256) << 20;

    std::size_t start_reserve() const {
        return options.prefill_chunk >= 2048 ? std::size_t(2048) << 20 : std::size_t(768) << 20;
    }

    // Rows the staging buffers hold: a prompt step's tokens or one row per sequence.
    [[nodiscard]] std::size_t staging_rows() const {
        return std::max<std::size_t>(std::max(options.prefill_chunk, span_tokens), options.sequences);
    }

    // The tokens of a prompt span (0: spans off): with host or disk experts, the most chunks
    // (kSpanChunks, 4 or 2, within max_context) whose span buffers (in rank 0's workspace, or
    // borrowed from its elastic cache for each span) and staging take no more than the memory
    // rank 0's expert cache would get beside two whole-layer banks, and at most an eighth of that
    // memory and the banks together, so the cache keeps most of its slots while a span runs.
    [[nodiscard]] std::uint32_t plan_span() const {
        if (model.expert_banks().empty()) { return 0; }
        const std::uint64_t chunk = plan_workspace(0), bank = bank_bytes();
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        const std::uint64_t fixed = chunk + kGraphReserve + unbacked() + (growing ? kKvHeadroom : 0) +
                                    delivery_metadata() + options.expert_cache_bytes.value_or(0);
        if (free <= fixed + 2 * bank) { return 0; }
        const std::uint64_t room = free - fixed - 2 * bank;
        const std::uint64_t row  = 12 + (options.engram_table ? std::uint64_t(hasher.rows_per_token()) *
                                                                    kEngramRowBytes : 0);
        for (std::uint32_t chunks = kSpanChunks; chunks >= 2; chunks /= 2) {
            const auto tokens = std::uint32_t(std::min<std::uint64_t>(
                options.max_context, std::uint64_t(chunks) * options.prefill_chunk));
            if (tokens <= options.prefill_chunk) { continue; }
            const std::uint64_t extra =
                plan_workspace(tokens) - chunk + std::uint64_t(tokens - options.prefill_chunk) * row;
            if (extra <= room && 8 * extra <= room + 2 * bank) { return tokens; }
        }
        return 0;
    }

    // ---- The prefill tail ----------------------------------------------------------------------

    // The blocks after the last kv source keep no compressed state: each reads that source's
    // caches and its own window ring, so the state they leave at a position depends on their rows
    // within decoder blocks x (window - 1) + 1 before it, and a call's logit row on as many before
    // it.
    void plan_prefill_tail() {
        int last_source = -1;
        for (std::size_t l = 0; l < layers.size(); ++l) {
            if (layers[l].kv_source) { last_source = int(l); }
        }
        decoder_block = last_source + 1;
        for (std::size_t l = std::size_t(decoder_block); l < layers.size(); ++l) {
            layers[l].decoder = true;
        }
        const auto decoders = std::uint32_t(int(layers.size()) - decoder_block);
        exact_rows          = decoders == 0 ? 0 : decoders * std::uint32_t(window - 1) + 1;
        // A DSpine drafter that taps a decoder block reads its output at every kept row.
        bool taps_decoder = false;
        if (options.dspine) {
            for (const std::uint32_t l : model.dspine()->config().target_layer_ids) {
                taps_decoder = taps_decoder || int(l) >= decoder_block;
            }
        }
        switch (options.prefill_tail) {
        case PrefillTail::Full: tail_rows = 0; break;
        case PrefillTail::Exact: tail_rows = exact_rows; break;
        case PrefillTail::Rows:
            if (options.prefill_tail_rows < std::uint32_t(window)) {
                throw std::invalid_argument("DeepSeek-V4.1: a prefill tail of " +
                                            std::to_string(options.prefill_tail_rows) +
                                            " rows is shorter than the attention window (" +
                                            std::to_string(window) + ")");
            }
            tail_rows = options.prefill_tail_rows;
            break;
        }
        if (decoders == 0 || taps_decoder) { tail_rows = 0; }
        complete_rows = std::min(tail_rows, exact_rows);
        decoder_from.assign(options.sequences, 0);
        decoder_behind.assign(options.sequences, 0);
    }

    // The first of a prompt call's t rows (at positions first..) that the decoder blocks run: under
    // the tail, the rows from need = min(state_at, the first logit row + 1) - tail_rows on, from the
    // chunk boundary of the call at or below it (Exact: every chunk keeps a full run's shapes) or
    // from that row (Rows); t when the call needs none of its rows there. Sets the run's window
    // floor: a run starts where the call needs rows past its first, or where the decoder blocks are
    // behind the call's first position, which only a need from there on allows.
    int decoder_start(std::uint32_t sequence, std::uint32_t first, int t, std::uint32_t logit_rows,
                      std::uint32_t state_at) {
        const std::uint32_t end = first + std::uint32_t(t);
        if (state_at != 0 && state_at < end) {
            throw std::invalid_argument("deepseek_v41 forward: a kept state before the call's end");
        }
        int row          = 0;
        std::int64_t need = 0;
        if (tail_rows != 0) {
            std::uint32_t at = state_at == 0 ? end : state_at;
            if (logit_rows > 0) { at = std::min(at, end - logit_rows + 1); }
            need = std::int64_t(at) - std::int64_t(tail_rows);
            if (need >= std::int64_t(end)) {
                row = t;
            } else if (need > std::int64_t(first)) {
                row = int(need - std::int64_t(first));
                if (options.prefill_tail == PrefillTail::Exact) {
                    const int c = int(options.prefill_chunk);
                    row         = row / c * c;
                }
            }
        }
        if (row < t && decoder_behind[sequence]) {
            if (need < std::int64_t(first)) {
                throw std::logic_error("deepseek_v41: a prompt call needs decoder rows that an "
                                       "earlier call of its prefill tail skipped");
            }
            decoder_from[sequence] = first + std::uint32_t(row);
        } else if (row > 0 && row < t) {
            decoder_from[sequence] = first + std::uint32_t(row);
        }
        return row;
    }

    // After a prompt call whose decoder blocks ran its rows from `row` on (t: none).
    void decoder_ran(std::uint32_t sequence, std::uint32_t end, int row, int t) {
        if (row >= t) {
            decoder_behind[sequence] = 1;
            return;
        }
        decoder_behind[sequence] = 0;
        if (decoder_from[sequence] != 0 && end - decoder_from[sequence] >= complete_rows) {
            decoder_from[sequence] = 0;
        }
    }

    [[nodiscard]] bool resumable(std::uint32_t sequence) const {
        return decoder_behind.at(sequence) == 0 && decoder_from.at(sequence) == 0;
    }

    void require_resumable(std::uint32_t sequence, const char* what) const {
        if (!resumable(sequence)) {
            throw std::logic_error(std::string("deepseek_v41 ") + what +
                                   ": the sequence's state is mid-way through a prefill tail");
        }
    }

    // The first row of a prompt call at `first` that the drafter taps: its rows within the window
    // before the state the caller keeps (the call's end by default).
    [[nodiscard]] int tap_start(std::uint32_t first, int t, std::uint32_t state_at) const {
        const std::int64_t at = state_at == 0 ? std::int64_t(first) + t : std::int64_t(state_at);
        return int(std::clamp<std::int64_t>(at - window - std::int64_t(first), 0, t));
    }

    void reset_decoder(std::uint32_t sequence) {
        decoder_from.at(sequence)   = 0;
        decoder_behind.at(sequence) = 0;
    }

    // ---- Growing caches -------------------------------------------------------------------------

    [[nodiscard]] std::size_t unbacked() const {
        std::size_t bytes = 0;
        for (const KvArea* area : sequence_areas) {
            if (area->region) { bytes += area->region.bytes() - area->region.mapped_bytes(); }
        }
        return bytes;
    }

    void count_backed() {
        std::uint64_t bytes = kv_fixed;
        for (const KvArea* area : sequence_areas) {
            if (area->region) { bytes += area->region.mapped_bytes(); }
        }
        kv_backed.store(bytes, std::memory_order_relaxed);
    }

    [[nodiscard]] std::size_t missing(const KvArea& a, std::size_t owner, std::uint32_t positions) const {
        if (!a.region || positions <= a.backed[owner]) { return 0; }
        return a.region.unmapped_bytes(owner * a.row_bytes, a.extent(positions));
    }
    [[nodiscard]] bool extend(KvArea& a, std::size_t owner, std::uint32_t positions) {
        if (!a.region || positions <= a.backed[owner]) { return true; }
        if (!a.region.map(owner * a.row_bytes, a.extent(positions), stream)) {
            return false;
        }
        a.backed[owner] = positions;
        return true;
    }

    // Unbacks `owner`'s row past `positions`, except granules another row still reaches into.
    void trim(KvArea& a, std::size_t owner, std::uint32_t positions) {
        if (!a.region || a.backed[owner] <= positions) { return; }
        const std::size_t g     = a.region.granule();
        const std::uint32_t was = a.backed[owner];
        a.backed[owner]         = positions;
        const auto reached = [&](std::size_t lo, std::size_t hi) {
            for (std::size_t o = 0; o < a.backed.size(); ++o) {
                const std::size_t row = o * a.row_bytes;
                const std::size_t end = row + a.extent(a.backed[o]);
                if (row < hi && lo < end) { return true; }
            }
            return false;
        };
        const std::size_t row = owner * a.row_bytes;
        const std::size_t hi  = row + a.extent(was);
        for (std::size_t at = (row + a.extent(positions)) / g * g; at < hi; at += g) {
            if (!reached(at, at + g)) { a.region.unmap(at, g); }
        }
    }

    // Before a call that writes or reads `sequence`'s positions below `end`: its compressed rows
    // are backed up to the 256-position tile that holds `end`, the expert cache giving memory back
    // when the device keeps less than the CUDA Graph reserve beside them. Outside capture.
    void back(std::uint32_t sequence, std::uint32_t end) {
        if (!growing) { return; }
        const auto upto = std::uint32_t(std::min<std::size_t>(round_up(end, 256), options.max_context));
        std::size_t need = 0;
        for (const KvArea* area : sequence_areas) { need += missing(*area, sequence, upto); }
        const auto apply = [&] {
            bool ok = true;
            for (KvArea* area : sequence_areas) { ok = ok && extend(*area, sequence, upto); }
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
                throw InsufficientMemory("DeepSeek-V4.1: no device memory left to grow the "
                                         "compressed caches to " + std::to_string(upto) + " positions");
            }
        }
        count_backed();
    }

    // Beside an elastic expert cache, before a span of `n` tokens: its span_buffers, the cache
    // giving memory back when the device keeps less than them and the CUDA Graph reserve. Empty
    // when the cache has no more to give. Outside capture.
    DeviceBuffer borrow_span(std::uint32_t n) {
        const std::size_t bytes = span_buffers(n), need = bytes + kGraphReserve;
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        while (free < need) {
            if (delivery->shrink_cache(std::max(need - free, std::size_t(64) << 20)) == 0) { return {}; }
            CUDA_CHECK(cudaMemGetInfo(&free, &total));
        }
        try {
            return DeviceBuffer(bytes);
        } catch (const std::runtime_error&) {
            (void)cudaGetLastError(); // the failed allocation's error
            return {};
        }
    }

    // After the span: its buffers back to the expert cache, once the span's work is done.
    void give_back(DeviceBuffer& borrowed) {
        if (!borrowed.p) { return; }
        CUDA_CHECK(ninfer::wait_for_stream(stream));
        borrowed = DeviceBuffer();
        delivery->grow_cache();
    }

    void release(std::uint32_t sequence) {
        if (!growing || sequence_areas.empty() || sequence_areas.front()->backed[sequence] == 0) { return; }
        CUDA_CHECK(ninfer::wait_for_stream(stream));
        for (KvArea* area : sequence_areas) { trim(*area, sequence, 0); }
        count_backed();
        delivery->grow_cache();
    }

    ~Impl() {
        settle_ahead(); // the reader's threads write no more into the pinned sets
        if (delivery || growing) { ninfer::wait_for_stream(stream); }
        for (std::size_t r = 1; r < expert_peers.size(); ++r) {
            if (expert_peers[r].stream == nullptr) { continue; }
            RankBinding bind(device, r);
            ninfer::wait_for_stream(expert_peers[r].stream);
            expert_peers[r].delivery.reset();
            expert_peers[r].workspace.reset();
            cudaStreamDestroy(expert_peers[r].stream);
        }
        divided_head.reset();
    }
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;

    // ---- Workspace -----------------------------------------------------------------------------

    static constexpr int kHeadRows = 64; // LM-head columns per FP32 plane pass

    [[nodiscard]] int decode_rows() const {
        return std::min<int>(gguf::kMaxVectorColumns,
                             int(options.sequences) * (int(options.draft_tokens) + 1));
    }

    // A span of `n` tokens' residual streams, both coefficient sites and index selections, as
    // forward_span allocates them.
    [[nodiscard]] std::size_t span_buffers(std::size_t n) const {
        const std::size_t s = streams, site = (2 * s + s * s) * 4;
        return round_up(n * s * hidden * 4, kAlign) + 2 * round_up(n * site, kAlign) +
               round_up(n * std::size_t(index_topk) * 4, kAlign) +
               (candidate_words > 0 ? round_up(n * std::size_t(candidate_words) * 4, kAlign) : 0);
    }

    // The workspace of a prompt chunk's pass, and with `span` tokens a span's residual streams,
    // coefficients and index selections beside one chunk's temporaries.
    std::size_t plan_workspace(std::uint32_t span) const {
        const std::size_t t = options.prefill_chunk, n = span, h = hidden, s = streams;
        const std::size_t latent_rows = std::size_t(heads) * kLatent;
        const std::size_t pairs = t * top_k, w = width;
        std::size_t gguf_scratch = 0;
        const auto product = [&](const Weight& m) {
            const ops::detail::GgufShape shape{m.qtype, m.n, m.k};
            gguf_scratch = std::max(gguf_scratch,
                                    ops::detail::gguf_project_workspace_bytes({&shape, 1}, 1, int(t)));
        };
        const auto block_products = [&](const LayerPlan& p) {
            for (const Weight& m : p.query_kv) { product(m); }
            for (const Weight& m : p.shared_gate_up) { product(m); }
            Weight group = p.output_a;
            group.n      = output_rank;
            for (const Weight* m : {&p.query_b, &p.output_b, &p.shared_down, &std::as_const(group)}) {
                product(*m);
            }
            if (p.engram >= 0) { product(p.engram_projection); }
        };
        for (const LayerPlan& p : layers) { block_products(p); }
        // An STQ drafter's products: its trunk's at a draft's rows (the seed's in chunks of as
        // many), its grouped output projection's groups, and a pair per routed expert of a row.
        std::size_t stq_scratch = 0;
        const auto stq_product = [&](const ops::stq::Matrices& m, int items, int rows) {
            ops::stq::Product q;
            q.m     = m;
            q.items = items;
            q.rows  = rows;
            stq_scratch = std::max(stq_scratch, ops::stq::workspace_bytes(q, sms));
        };
        if (drafter) {
            for (const LayerPlan& p : drafter->layers) {
                if (!p.stq) {
                    block_products(p);
                    continue;
                }
                const StqPlan& m = *p.stq;
                constexpr int rows = ops::stq::kMaxRows;
                for (const auto* q : {&m.query_a, &m.kv, &m.query_b, &m.output_b, &m.shared_gate,
                                      &m.shared_up, &m.shared_down}) {
                    stq_product(*q, 1, rows);
                }
                stq_product(m.output_a, groups, rows);
                for (const auto* q : {&m.gate, &m.up, &m.down}) { stq_product(*q, rows * p.top_k, 1); }
            }
            if (drafter->stq_fc) {
                stq_product(*drafter->stq_fc, 1, ops::stq::kMaxRows);
            } else {
                product(drafter->fc);
            }
        }
        {
            Weight rows = head;
            rows.n      = std::min<int>(head.n, vocab);
            const ops::detail::GgufShape hs{rows.qtype, rows.n, rows.k};
            gguf_scratch = std::max(gguf_scratch,
                                    ops::detail::gguf_project_workspace_bytes({&hs, 1}, 1, kHeadRows));
        }
        std::size_t allocations = 0;
        const auto sum = [&](std::initializer_list<std::size_t> sizes) {
            std::size_t bytes = 0;
            for (const std::size_t n : sizes) {
                bytes += round_up(n, kAlign);
                ++allocations;
            }
            return bytes;
        };
        const std::size_t mixes = config.hc_mixes();
        const std::size_t sites = 2 * (2 * s + s * s) * 4;
        // A decode or verification pass's listed candidate blocks (index_block) beside its masks.
        const std::size_t listed = std::size_t(std::max(candidate_keep, 0));
        std::size_t pass = sum({t * s * h * 4, t * sites, t * mixes * 4, t * std::size_t(index_topk) * 4,
                                t * std::size_t(candidate_words) * 4, t * listed * 4, t * 4});
        if (n > t) { pass = std::max(pass, span_buffers(n) + sum({t * mixes * 4})); }
        const std::size_t columns = std::size_t(config.engram.columns()) * config.engram.head_dim;
        const std::size_t engram_bytes = sum({t * columns * 2, t * 4, t * (s + 1) * h * 4});
        const std::size_t decode_width = std::size_t(decode_rows()) * std::size_t(capacity_rows) * 4;
        const std::size_t scores = std::max(kScoreBudget, decode_width);
        // The candidate source's block maxima (index_block): for each row of scores, which holds
        // the `width` positions the row sees, a row of ceil(width / block) floats, so at most the
        // scores' bytes over the block and 4 bytes a row (a chunk's or a decode pass's) for the
        // last block.
        const std::size_t maxima =
            candidate_block > 0
                ? scores / std::size_t(candidate_block) +
                      4 * std::max(t, std::size_t(decode_rows()))
                : 0;
        const std::size_t partial =
            kernels::sparse_attention_workspace_floats(decode_rows(), heads, decode_splits) * 4;
        const std::size_t attention = sum({
            t * h * 2, t * h * 4, t * 1792 * 4, t * std::size_t(query_rank) * 2,
            t * std::size_t(query_rank) * 4, t * 4, t * 4, 8 * t * 4,
            t * latent_rows * 2, t * latent_rows * 2,
            t * std::size_t(groups) * output_rank * 4, t * std::size_t(groups) * output_rank * 2,
            gguf::vector_activation_bytes(heads / groups * kLatent, groups * gguf::kMaxVectorColumns),
            std::size_t(groups) * gguf::kMaxVectorColumns * 4, t * h * 4,
            // compressor
            t * kLatent * 4, t * kLatent * 4, t * kLatent * 4, t * kIndexDim * 4,
            // indexer
            t * std::size_t(index_heads) * kIndexDim * 4, t * std::size_t(index_heads) * kIndexDim * 2,
            t * std::size_t(index_heads) * 4, scores, maxima,
            // a chunk's listed candidate blocks, four rows a group
            (t + 3) * listed * 4, t * 4, partial});
        const std::size_t ffn =
            sum({t * h * 2, t * h * 4, t * 4, t * h * 4, t * std::size_t(experts) * 4, pairs * 4,
                 pairs * 4, gguf::moe_routing_bytes(experts, int(pairs)), pairs * 2 * w * 4,
                 pairs * w * 2,
                 std::max(gguf::moe_matrix_activation_bytes(int(h), int(pairs)),
                          gguf::vector_activation_bytes(int(h), int(t))),
                 std::max(gguf::moe_matrix_activation_bytes(int(w), int(pairs)),
                          gguf::vector_activation_bytes(int(w), int(pairs))),
                 pairs * h * 4, t * h * 8, t * 2 * w * 4, t * w * 2,
                 expert_ranges.empty() ? 0 : pairs * 4}); // expert parallelism: rank 0's own ids
        const std::size_t head_bytes = sum({t * h * 2, std::size_t(kHeadRows) * vocab * 4, kHeadRows * 4});
        // A seed of the drafter's rings from a pass's last rows (at most a window of them).
        std::size_t seed_bytes = 0;
        if (drafter) {
            const std::size_t r = std::min<std::size_t>(window, std::max<std::size_t>(t, n));
            seed_bytes = sum({r * std::size_t(drafter->targets) * h * 2, r * h * 4, r * h * 2,
                              r * drafter->layers.size() * kLatent * 4, r * 4});
        }
        return pass + std::max({engram_bytes, attention, ffn, head_bytes}) + seed_bytes + gguf_scratch +
               stq_scratch + allocations * kAlign + (32u << 20);
    }

    // ---- Shared pieces ---------------------------------------------------------------------------

    // A call of t <= kMaxVectorColumns tokens through the grouped output projection: group g of
    // token j is pair j * groups + g, the groups the experts.
    [[nodiscard]] gguf::MoeRouting group_routing(int t) const {
        const auto* base = static_cast<const std::int32_t*>(group_routes.p) + group_routes_at.at(std::size_t(t));
        gguf::MoeRouting routing;
        routing.bounds       = base;
        routing.sorted       = base + groups + 1;
        routing.active       = routing.sorted + std::int64_t(groups) * t;
        routing.active_count = routing.active + groups;
        return routing;
    }

    template <typename T>
    T* alloc(std::size_t count) {
        return static_cast<T*>(workspace->alloc_bytes(count * sizeof(T)).data);
    }
    Tensor alloc_bf16(int rows, int t) { return workspace->alloc(DType::BF16, {rows, t}); }

    [[nodiscard]] static bool f16_sums(const Weight& w, int t) {
        const auto type = ops::detail::gguf_type(w.qtype);
        return t <= gguf::kMaxVectorColumns ? gguf::vector_activation_f16_sums(type)
                                            : gguf::matrix_activation_f16_sums(type);
    }

    // The products of `parts` (rows one after another) of the BF16 x [K, T] into FP32 [T][rows]
    // (column stride `stride`), x guarded for F16 sums where a part's activation keeps them
    // (kernels::guard_columns), its product's columns unscaled after. x may be scaled in place.
    void project(Tensor& x, std::span<const Weight> parts, float* out, int stride, int row = 0) {
        const int t = x.ne[1];
        bool guard  = false;
        for (const Weight& w : parts) { guard = guard || f16_sums(w, t); }
        float* scale = nullptr;
        if (guard) {
            scale = alloc<float>(t);
            kernels::guard_columns(static_cast<__nv_bfloat16*>(x.data), x.ne[0], t, scale, stream);
        }
        std::vector<ops::detail::GgufProduct> products(parts.size());
        int rows = 0;
        for (std::size_t i = 0; i < parts.size(); ++i) {
            products[i].weight   = &parts[i];
            products[i].f32      = out;
            products[i].f32_rows = stride;
            products[i].row      = row + rows;
            rows += parts[i].n;
        }
        ops::detail::gguf_project(x, products, *workspace, stream);
        if (guard) { kernels::unscale_columns(out + row, stride, rows, t, scale, stream); }
    }
    void project(Tensor& x, const Weight& w, float* out, int stride, int row = 0) {
        project(x, std::span<const Weight>(&w, 1), out, stride, row);
    }
    // The same on another rank's workspace and stream: FP32 columns of w.n rows from `out`.
    void project_on(Tensor& x, const Weight& w, float* out, WorkspaceArena& space,
                    cudaStream_t on) const {
        const int t  = x.ne[1];
        float* scale = nullptr;
        if (f16_sums(w, t)) {
            scale = static_cast<float*>(space.alloc_bytes(std::size_t(t) * 4).data);
            kernels::guard_columns(static_cast<__nv_bfloat16*>(x.data), x.ne[0], t, scale, on);
        }
        ops::detail::GgufProduct product;
        product.weight   = &w;
        product.f32      = out;
        product.f32_rows = w.n;
        ops::detail::gguf_project(x, {&product, 1}, space, on);
        if (scale != nullptr) { kernels::unscale_columns(out, w.n, w.n, t, scale, on); }
    }
    // The same into BF16 rows [T][w.n].
    void project_bf16(Tensor& x, const Weight& w, Tensor& out) {
        const int t      = x.ne[1];
        const bool guard = f16_sums(w, t);
        float* scale     = nullptr;
        if (guard) {
            scale = alloc<float>(t);
            kernels::guard_columns(static_cast<__nv_bfloat16*>(x.data), x.ne[0], t, scale, stream);
        }
        ops::detail::GgufProduct product;
        product.weight = &w;
        product.out    = &out;
        ops::detail::gguf_project(x, {&product, 1}, *workspace, stream);
        if (guard) {
            kernels::unscale_columns(static_cast<__nv_bfloat16*>(out.data), w.n, w.n, t, scale, stream);
        }
    }

    void stq_run(const ops::stq::Product& q) {
        auto scope = workspace->scope();
        ops::stq::product(q, workspace->alloc_bytes(ops::stq::workspace_bytes(q, sms)).data, sms,
                          stream);
    }

    // An STQ matrix's products of the BF16 x [K, T] into FP32 [T][stride] from column `row`, or
    // with `out16` into BF16 [T][stride], kMaxRows rows at a time.
    void stq_project(const ops::stq::Matrices& m, const Tensor& x, float* out, int stride, int row = 0,
                     __nv_bfloat16* out16 = nullptr) {
        const int t = x.ne[1];
        if (x.ne[0] != m.k) { throw std::invalid_argument("deepseek_v41: STQ input width mismatch"); }
        for (int at = 0; at < t; at += ops::stq::kMaxRows) {
            ops::stq::Product q;
            q.m            = m;
            q.rows         = std::min(ops::stq::kMaxRows, t - at);
            q.x            = static_cast<const __nv_bfloat16*>(x.data) + std::int64_t(at) * m.k;
            q.x_row_stride = m.k;
            const std::int64_t first = std::int64_t(at) * stride + row;
            if (out16 != nullptr) {
                q.out_bf16 = out16 + first;
            } else {
                q.out = out + first;
            }
            q.out_row_stride = stride;
            stq_run(q);
        }
    }

    void encode_image(std::span<const std::uint16_t> patches, int grid_w, int grid_h,
                      std::span<std::uint16_t> rows) {
        if (!tower) { throw std::invalid_argument("deepseek_v41 encode_image: no vision tower"); }
        const VisionConfig& vc = tower->config();
        const std::size_t span =
            VisionConfig::span_tokens(tower->aligner_rows(grid_h), tower->aligner_columns(grid_w));
        if (grid_w <= 0 || grid_h <= 0 ||
            std::uint64_t(grid_w) * std::uint64_t(grid_h) > options.vision_max_patches ||
            patches.size() != std::size_t(grid_w) * grid_h * vc.patch_features() ||
            span > vc.max_image_tokens || rows.size() != span * std::size_t(hidden)) {
            throw std::invalid_argument("deepseek_v41 encode_image: invalid image");
        }
        CUDA_CHECK(cudaMemcpyAsync(d_patches.p, patches.data(), patches.size() * 2,
                                   cudaMemcpyHostToDevice, stream));
        workspace->reset();
        tower->encode(static_cast<const __nv_bfloat16*>(d_patches.p), grid_w, grid_h,
                      static_cast<__nv_bfloat16*>(d_image_rows.p), *workspace, stream);
        CUDA_CHECK(cudaMemcpyAsync(rows.data(), d_image_rows.p, rows.size() * 2,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(ninfer::wait_for_stream(stream));
    }

    // Tokens, positions, and when given sequences, to the device.
    void stage(std::span<const std::int32_t> a, std::span<const std::int32_t> b,
               std::span<const std::int32_t> c) {
        staged->synchronize();
        auto* host = static_cast<std::int32_t*>(h_staging->data());
        const std::size_t rows = staging_rows();
        if (a.size() > rows || b.size() > rows || c.size() > rows) {
            throw std::invalid_argument("deepseek_v41 executor: staging overflow");
        }
        std::copy(a.begin(), a.end(), host);
        std::copy(b.begin(), b.end(), host + rows);
        std::copy(c.begin(), c.end(), host + 2 * rows);
        CUDA_CHECK(cudaMemcpyAsync(d_tokens.p, host, a.size() * 4, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_positions.p, host + rows, b.size() * 4,
                                   cudaMemcpyHostToDevice, stream));
        if (!c.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(d_sequences.p, host + 2 * rows, c.size() * 4,
                                       cudaMemcpyHostToDevice, stream));
        }
        staged->record(stream);
    }

    // ---- Engram rows -----------------------------------------------------------------------------

    // The table rows of a pass's rows to the device, block-major: rows[r] is (sequence, token)
    // after the sequence's `history` (advanced when `advance`).
    void read_engram(std::span<const std::int32_t> tokens, std::span<const std::uint32_t> owners,
                     std::span<EngramHasher::History> history_of, bool advance,
                     std::span<const std::uint8_t> dead = {}) {
        if (!engram) { return; }
        const std::size_t t     = tokens.size();
        const std::uint32_t per = hasher.rows_per_token();
        for (std::size_t r = 0; r < t;) {
            // Consecutive rows of one sequence hash together.
            std::size_t end = r + 1;
            while (end < t && owners[end] == owners[r]) { ++end; }
            hasher.rows(history_of[r], tokens.subspan(r, end - r), engram_scratch.data() + r * per,
                        advance, dead.empty() ? dead : dead.subspan(r, end - r));
            r = end;
        }
        settle_ahead();
        EngramSet& set = engram_sets[std::size_t(engram_set)];
        set.copied->synchronize();
        auto* ids = static_cast<std::uint64_t*>(set.ids->data());
        engram_ids(t, ids);
        const std::size_t n = t * per;
        engram->read_rows({ids, n}, {static_cast<std::uint8_t*>(set.rows->data()), n * kEngramRowBytes});
        CUDA_CHECK(cudaMemcpyAsync(d_engram.p, set.rows->data(), n * kEngramRowBytes,
                                   cudaMemcpyHostToDevice, stream));
        set.copied->record(stream);
    }

    // The table rows of t hashed positions (engram_scratch, [position][block][column]) in the order
    // the blocks read them: [block][position][column].
    void engram_ids(std::size_t t, std::uint64_t* ids) const {
        const std::uint32_t per = hasher.rows_per_token(), columns = hasher.columns();
        for (std::size_t b = 0; b < per / columns; ++b) {
            for (std::size_t r = 0; r < t; ++r) {
                std::memcpy(ids + (b * t + r) * columns, engram_scratch.data() + r * per + b * columns,
                            columns * 8);
            }
        }
    }

    // Completes a read-ahead batch in flight; its rows stay for the call they were read for. A
    // read that failed leaves none, and that call then reads (and reports) its rows itself.
    void settle_ahead() {
        if (!ahead.pending) { return; }
        ahead.pending = false;
        try {
            engram->wait();
            ahead.ready = true;
        } catch (const std::runtime_error&) { ahead.ready = false; }
    }

    // Starts reading the Engram rows of `next`, the prompt call of `sequence` at `first` that
    // follows the current one (whose tokens have advanced the sequence's history), into the set
    // the passes do not read into.
    void read_ahead(std::uint32_t sequence, std::uint32_t first, NextPromptCall next) {
        if (!engram || !options.engram_read_ahead || next.tokens.empty()) { return; }
        const std::size_t t = next.tokens.size();
        if (t > staging_rows() || (!next.image_rows.empty() && next.image_rows.size() != t)) {
            throw std::invalid_argument("deepseek_v41 forward: invalid next prompt call");
        }
        settle_ahead();
        ahead.ready    = false;
        EngramSet& set = engram_sets[std::size_t(1 - engram_set)];
        set.copied->synchronize();
        EngramHasher::History history = histories[sequence];
        ahead.history                 = history;
        hasher.rows(history, next.tokens, engram_scratch.data(), false, next.image_rows);
        auto* ids = static_cast<std::uint64_t*>(set.ids->data());
        engram_ids(t, ids);
        const std::size_t n = t * hasher.rows_per_token();
        engram->submit({ids, n}, {static_cast<std::uint8_t*>(set.rows->data()), n * kEngramRowBytes});
        ahead.pending  = true;
        ahead.sequence = sequence;
        ahead.first    = first;
        ahead.tokens.assign(next.tokens.begin(), next.tokens.end());
        ahead.dead.assign(next.image_rows.begin(), next.image_rows.end());
    }

    // Whether the read-ahead holds the rows of this prompt call of `sequence` (its tokens, image
    // flags `dead`, position and history): then they go to the device from its set, which the
    // passes read into from now on.
    bool use_ahead(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                   std::span<const std::uint8_t> dead) {
        settle_ahead();
        const EngramHasher::History& history = histories[sequence];
        const bool dead_matches = dead.empty()
                                      ? std::find(ahead.dead.begin(), ahead.dead.end(), 1) == ahead.dead.end()
                                      : std::equal(dead.begin(), dead.end(), ahead.dead.begin(), ahead.dead.end());
        if (!ahead.ready || ahead.sequence != sequence || ahead.first != positions[sequence] ||
            !std::equal(tokens.begin(), tokens.end(), ahead.tokens.begin(), ahead.tokens.end()) ||
            !dead_matches || ahead.history.count != history.count ||
            !std::equal(history.ids, history.ids + history.count, ahead.history.ids)) {
            return false;
        }
        ahead.ready         = false;
        engram_set          = 1 - engram_set;
        EngramSet& set      = engram_sets[std::size_t(engram_set)];
        const std::size_t n = tokens.size() * hasher.rows_per_token();
        CUDA_CHECK(cudaMemcpyAsync(d_engram.p, set.rows->data(), n * kEngramRowBytes,
                                   cudaMemcpyHostToDevice, stream));
        set.copied->record(stream);
        return true;
    }

    void advance_history(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                         std::span<const std::uint8_t> dead = {}) {
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            histories[sequence].push(!dead.empty() && dead[i] ? EngramHasher::kDead
                                                              : hasher.compressed(tokens[i]),
                                     hasher.hash().ngram_size - 1);
        }
    }

    // ---- Blocks ----------------------------------------------------------------------------------

    void embed(int t, float* h) {
        gguf::dequantize_rows(ops::detail::gguf_type(embedding.qtype), embedding.qdata,
                              row_bytes(embedding), hidden,
                              static_cast<const std::int32_t*>(d_tokens.p), t, h,
                              std::int64_t(streams) * hidden, stream);
        // The step's image rows replace their embeddings, a piece at a time through the device
        // rows (each copy ordered after the placement before it).
        const std::int64_t stride = std::int64_t(streams) * hidden;
        for (const ImageRows& piece : step_images) {
            const int count = int(piece.embeddings.size() / std::size_t(hidden));
            CUDA_CHECK(cudaMemcpyAsync(d_image_rows.p, piece.embeddings.data(),
                                       piece.embeddings.size() * 2, cudaMemcpyHostToDevice, stream));
            kernels::place_rows(static_cast<const __nv_bfloat16*>(d_image_rows.p), count, hidden,
                                h + std::int64_t(piece.row) * stride, stride, stream);
        }
        kernels::expand_streams(h, t, streams, hidden, stream);
    }

    struct Coefficients {
        float *pre = nullptr, *post = nullptr, *comb = nullptr;
    };

    void mixes(const Site& site, const float* h, int t, float* scratch, const Coefficients& out) {
        const int m = int(config.hc_mixes());
        kernels::gemm_f32(m, t, streams * hidden, site.fn, h, scratch, stream);
        kernels::HcSplitArgs a;
        a.h          = h;
        a.mixes      = scratch;
        a.tokens     = t;
        a.streams    = streams;
        a.hidden     = hidden;
        a.scale      = site.scale;
        a.base       = site.base;
        a.norm_eps   = eps;
        a.hc_eps     = config.hc_eps;
        a.iterations = int(config.hc_sinkhorn_iters);
        a.pre        = out.pre;
        a.post       = out.post;
        a.comb       = out.comb;
        kernels::hc_split(a, stream);
    }

    void engram_block(const LayerPlan& p, float* h, int t, const Pass& s) {
        if (!engram) { return; }
        auto scope        = workspace->scope();
        const int columns = int(hasher.columns());
        const int rows    = t * columns;
        const std::size_t block = std::size_t(s.staged ? s.staged : t) * columns;
        Tensor e          = alloc_bf16(columns * int(config.engram.head_dim), t);
        kernels::engram_rows(static_cast<const std::uint8_t*>(d_engram.p) +
                                 (std::size_t(p.engram) * block + std::size_t(s.row) * columns) *
                                     kEngramRowBytes,
                             rows, static_cast<__nv_bfloat16*>(e.data), stream);
        const int kv_rows = (streams + 1) * hidden;
        auto* kv          = alloc<float>(std::size_t(t) * kv_rows);
        project(e, p.engram_projection, kv, kv_rows);
        kernels::engram_gate(h, kv, p.engram_gate, t, streams, hidden, eps, stream, image_mask);
    }

    // The block's attention: its output y added to the streams through `post` and `comb`. With
    // `kv_only`, only the state the block leaves for later rows: its window latents, and a kv
    // source's compressed rows (the same projection, so the same bytes); the streams stay.
    void attention(const LayerPlan& p, float* h, const float* pre, int t, const Pass& s,
                   std::int32_t* topk, std::uint32_t* candidates, const Coefficients& c,
                   bool kv_only = false) {
        auto scope      = workspace->scope();
        const auto* pos = static_cast<const std::int32_t*>(d_positions.p) + s.row;
        const auto* seq = s.chunk ? nullptr : static_cast<const std::int32_t*>(d_sequences.p);
        Tensor x        = alloc_bf16(hidden, t);
        auto* x32       = alloc<float>(std::size_t(t) * hidden);
        kernels::hc_pre_norm(h, pre, p.attention_norm, t, streams, hidden, eps,
                             static_cast<__nv_bfloat16*>(x.data), x32, stream);
        const int qkv_rows = query_rank + kLatent;
        auto* qkv          = alloc<float>(std::size_t(t) * qkv_rows);
        if (p.stq) {
            stq_project(p.stq->query_a, x, qkv, qkv_rows);
            stq_project(p.stq->kv, x, qkv, qkv_rows, query_rank);
        } else {
            project(x, p.query_kv, qkv, qkv_rows);
        }
        Tensor qr  = alloc_bf16(query_rank, t);
        auto* qr32 = alloc<float>(std::size_t(t) * query_rank);
        kernels::rms_rows(qkv, qkv_rows, query_rank, t, p.query_norm, eps,
                          static_cast<__nv_bfloat16*>(qr.data), qr32, stream);
        // The window latents: a chunk's into the staging ring after the window before it, decode
        // rows' into their sequences' rings.
        const auto ring_stride = std::int64_t(kRingBytes);
        auto* ring             = static_cast<std::uint8_t*>(p.ring.p);
        auto* own              = ring + std::int64_t(s.sequence) * ring_stride;
        auto* stage_ring       = static_cast<std::uint8_t*>(staging.p);
        if (s.chunk) {
            kernels::ring_copy(own, kRing, stage_ring, prefill_ring,
                               std::max(0, int(s.first) - window + 1), int(s.first), stream);
        }
        kernels::KvPostArgs kv;
        kv.kv        = qkv + query_rank;
        kv.kv_stride = qkv_rows;
        kv.tokens    = t;
        kv.norm      = p.kv_norm;
        kv.eps       = eps;
        kv.positions = pos;
        kv.freq      = p.freq;
        kv.cache     = s.chunk ? stage_ring : ring;
        kv.seq_stride = s.chunk ? 0 : ring_stride;
        kv.sequences = seq;
        kv.ring      = s.chunk ? prefill_ring : kRing;
        kernels::kv_post(kv, stream);
        if (kv_only) {
            if (!s.chunk) { throw std::logic_error("deepseek_v41: a key-value-only decode row"); }
            if (p.ratio > 0 && p.kv_source) { compress_block(p, x32, t, s); }
            const int end = int(s.first) + t;
            kernels::ring_copy(stage_ring, prefill_ring, own, kRing, std::max(0, end - window), end,
                               stream);
            return;
        }
        Tensor q = alloc_bf16(heads * kLatent, t);
        if (p.stq) {
            stq_project(p.stq->query_b, qr, nullptr, heads * kLatent, 0,
                        static_cast<__nv_bfloat16*>(q.data));
        } else {
            project_bf16(qr, p.query_b, q);
        }
        kernels::rotate_heads(static_cast<const __nv_bfloat16*>(q.data), t, heads, kLatent, pos,
                              p.freq, false, static_cast<__nv_bfloat16*>(q.data), stream);
        kernels::AttentionArgs a;
        if (p.ratio > 0) {
            if (p.kv_source) { compress_block(p, x32, t, s); }
            const LayerPlan& source = layers[std::size_t(p.kv_from)];
            if (p.index_source) { index_block(p, source, qr32, x32, t, s, topk, candidates); }
            a.compressed            = static_cast<const std::uint8_t*>(source.latents.p);
            a.compressed_seq_stride = std::int64_t(source.latents.row_bytes);
            a.topk                  = topk;
            a.k                     = index_topk;
        }
        Tensor o            = alloc_bf16(heads * kLatent, t);
        a.q                 = static_cast<const __nv_bfloat16*>(q.data);
        a.tokens            = t;
        a.heads             = heads;
        a.positions         = s.visible != nullptr ? s.visible : pos;
        a.sequences         = seq;
        a.sequence          = int(s.sequence);
        a.window            = s.chunk ? stage_ring : ring;
        a.window_seq_stride = s.chunk ? 0 : ring_stride;
        a.ring              = s.chunk ? prefill_ring : kRing;
        a.window_size       = window + s.extra_window;
        // A prefill tail's run: its chunks' decoder blocks read no window position before it.
        a.window_floor      = s.chunk && p.decoder ? int(decoder_from[s.sequence]) : 0;
        a.sinks             = p.sinks;
        a.scale             = 1.0f / std::sqrt(float(kLatent));
        a.splits            = s.chunk ? 1 : decode_splits;
        if (a.splits > 1) {
            a.partial = alloc<float>(kernels::sparse_attention_workspace_floats(t, heads, a.splits));
        }
        a.out = static_cast<__nv_bfloat16*>(o.data);
        kernels::sparse_attention(a, stream);
        if (s.chunk) {
            const int end = int(s.first) + t;
            kernels::ring_copy(stage_ring, prefill_ring, own, kRing, std::max(0, end - window), end,
                               stream);
        }
        // The grouped output projection: group g's heads (rounded as the reference rounds them)
        // through rows [g * o_rank, (g + 1) * o_rank) of output_a, into the latent's rows there. A
        // decode or verification call runs the groups as the experts of one vector product, its
        // pairs token by token (group_routing), so one guard, quantization, product and unscale
        // serve every group: each pair's arithmetic is that of its group's own product, whose eight
        // launches of 1,024 rows read output_a at about 470 GB/s on an RTX 4090.
        auto* grouped         = static_cast<__nv_bfloat16*>(q.data); // q is spent
        const bool vector     = !p.stq && t <= gguf::kMaxVectorColumns;
        kernels::attention_output(static_cast<const __nv_bfloat16*>(o.data), t, heads, groups, pos,
                                  p.freq, vector, grouped, stream);
        const int latent_rows = groups * output_rank;
        const int group_k     = heads / groups * kLatent;
        auto* latent          = alloc<float>(std::size_t(t) * latent_rows);
        if (p.stq) {
            // The groups as the items of one STQ product, group g's rows [g][t][group_k].
            if (t > ops::stq::kMaxRows) {
                throw std::invalid_argument("deepseek_v41: an STQ block runs at most 8 rows");
            }
            ops::stq::Product g;
            g.m               = p.stq->output_a;
            g.items           = groups;
            g.rows            = t;
            g.x               = grouped;
            g.x_item_stride   = std::int64_t(t) * group_k;
            g.x_row_stride    = group_k;
            g.out             = latent;
            g.out_item_stride = output_rank;
            g.out_row_stride  = latent_rows;
            stq_run(g);
        } else if (vector) {
            const int pairs = t * groups;
            const auto type = ops::detail::gguf_type(p.output_a.qtype);
            float* scale    = nullptr;
            if (gguf::vector_activation_f16_sums(type)) {
                scale = alloc<float>(pairs);
                kernels::guard_columns(grouped, group_k, pairs, scale, stream);
            }
            void* act = workspace->alloc_bytes(gguf::vector_activation_bytes(group_k, pairs)).data;
            gguf::quantize_vector_activation(grouped, group_k, pairs, nullptr, act, stream);
            const gguf::MoeTable table{static_cast<const void* const*>(p.output_groups.p),
                                       row_bytes(p.output_a), output_rank, group_k};
            gguf::MoeOutput out;
            out.f32         = latent;
            const int chunk = t <= 1 ? 1 : t <= 2 ? 2 : t <= 4 ? 4 : 8;
            gguf::moe_vector_product(type, table, group_routing(t), groups, 1, 1, act, pairs, out,
                                     chunk, stream);
            if (scale != nullptr) {
                kernels::unscale_columns(latent, output_rank, output_rank, pairs, scale, stream);
            }
        }
        for (int g = 0; g < groups && !vector && !p.stq; ++g) {
            Weight part = p.output_a;
            part.qdata  = static_cast<const std::byte*>(p.output_a.qdata) +
                         std::int64_t(g) * output_rank * row_bytes(p.output_a);
            part.n      = output_rank;
            Tensor xg(grouped + std::int64_t(g) * t * group_k, DType::BF16, {group_k, t});
            project(xg, part, latent, latent_rows, g * output_rank);
        }
        Tensor latent16 = alloc_bf16(latent_rows, t);
        kernels::f32_to_bf16(latent, static_cast<__nv_bfloat16*>(latent16.data),
                             std::int64_t(t) * latent_rows, stream);
        auto* y = alloc<float>(std::size_t(t) * hidden);
        if (p.stq) {
            stq_project(p.stq->output_b, latent16, y, hidden);
        } else {
            project(latent16, p.output_b, y, hidden);
        }
        kernels::hc_post(h, y, c.post, c.comb, t, streams, hidden, stream);
    }

    // A kv source's compressor: the compressed latents of the groups its rows complete, and their
    // indexer keys, into its caches.
    void compress_block(const LayerPlan& p, const float* x32, int t, const Pass& s) {
        auto scope      = workspace->scope();
        const auto* pos = static_cast<const std::int32_t*>(d_positions.p) + s.row;
        const auto* seq = s.chunk ? nullptr : static_cast<const std::int32_t*>(d_sequences.p);
        auto* kv        = alloc<float>(std::size_t(t) * kLatent);
        kernels::gemm_f32(kLatent, t, hidden, p.compressor_kv, x32, kv, stream);
        float* score = nullptr;
        if (p.ratio > 1) {
            score = alloc<float>(std::size_t(t) * kLatent);
            kernels::gemm_f32(kLatent, t, hidden, p.compressor_gate, x32, score, stream);
        }
        auto* latent = alloc<float>(std::size_t(t) * kLatent);
        kernels::CompressArgs c;
        c.kv              = kv;
        c.score           = score;
        c.tokens          = t;
        c.ratio           = p.ratio;
        c.positions       = pos;
        c.sequences       = seq;
        c.sequence        = int(s.sequence);
        c.tail_kv         = static_cast<float*>(p.tail_kv.p);
        c.tail_score      = static_cast<float*>(p.tail_score.p);
        c.tail_seq_stride = std::int64_t(kTailSlots) * kLatent;
        c.slots           = kTailSlots;
        c.tail_from       = s.chunk ? std::max(0, t - kTailSlots) : 0;
        c.norm            = p.compressor_norm;
        c.eps             = eps;
        c.latent          = latent;
        kernels::compress(c, stream);
        kernels::compress_tail(c, stream);
        auto* key = alloc<float>(std::size_t(t) * kIndexDim);
        kernels::gemm_f32(kIndexDim, t, kLatent, p.index_key, latent, key, stream);
        kernels::LatentArgs l;
        l.latent           = latent;
        l.key              = key;
        l.tokens           = t;
        l.ratio            = p.ratio;
        l.positions        = pos;
        l.sequences        = seq;
        l.sequence         = int(s.sequence);
        l.key_norm         = p.index_key_norm;
        l.eps              = eps;
        l.freq             = p.freq;
        l.index_cache            = static_cast<std::uint8_t*>(p.keys.p);
        l.index_seq_stride       = std::int64_t(p.keys.row_bytes);
        l.index_scales           = static_cast<std::uint8_t*>(p.key_scales.p);
        l.index_scale_seq_stride = std::int64_t(p.key_scales.row_bytes);
        l.kv_cache               = static_cast<std::uint8_t*>(p.latents.p);
        l.kv_seq_stride          = std::int64_t(p.latents.row_bytes);
        kernels::latent_post(l, stream);
    }

    // The listed blocks one group of a pass's rows (a decode or verification row alone, a chunk's
    // rows in fours) keeps at most where its rows see `width` positions, or 0 where a row keeps every
    // block it sees, so that a later index source scores every visible position.
    [[nodiscard]] int listed_capacity(int width, bool chunk) const {
        if (candidate_keep <= 0) { return 0; }
        const int blocks = (width + candidate_block - 1) / candidate_block;
        if (blocks <= candidate_keep) { return 0; }
        return std::min((chunk ? 4 : 1) * candidate_keep, blocks);
    }

    // An index source: each row's index_topk compressed positions (topk), and at the candidate
    // source the candidate blocks the later index sources keep to. Where the rows see more blocks
    // than they keep (listed_capacity), a later index source scores only the blocks its rows' group
    // keeps, in a row of those blocks' positions (candidate_lists), and selects among them: every
    // other position of the row would score -inf, which the selection never takes.
    void index_block(const LayerPlan& p, const LayerPlan& source, const float* qr32,
                     const float* x32, int t, const Pass& s, std::int32_t* topk,
                     std::uint32_t* candidates) {
        auto scope      = workspace->scope();
        const auto* pos = static_cast<const std::int32_t*>(d_positions.p) + s.row;
        const auto* seq = s.chunk ? nullptr : static_cast<const std::int32_t*>(d_sequences.p);
        const int q_rows = index_heads * kIndexDim;
        auto* q32        = alloc<float>(std::size_t(t) * q_rows);
        kernels::gemm_f32(q_rows, t, query_rank, p.index_query, qr32, q32, stream);
        auto* q = alloc<__nv_bfloat16>(std::size_t(t) * q_rows);
        kernels::rotate_heads(q32, t, index_heads, kIndexDim, pos, p.freq, true, q, stream);
        auto* weights = alloc<float>(std::size_t(t) * index_heads);
        kernels::gemm_f32(index_heads, t, hidden, p.index_weights, x32, weights, stream);
        const int width = s.chunk ? std::max(1, int(s.first + std::uint32_t(t)) / p.ratio)
                                  : source.rows;
        const int group    = s.chunk ? 4 : 1;
        // A decode pass's rows list their blocks against the mask's every position.
        const int capacity = listed_capacity(s.chunk ? width : candidate_rows, s.chunk);
        const bool listed  = p.uses_candidates && capacity > 0 &&
                            (s.chunk || s.candidate_list != nullptr);
        // A row of scores: every visible position, or the listed blocks' positions.
        const int columns = listed ? capacity * candidate_block : width;
        int sub = t;
        if (s.chunk) {
            sub = int(std::clamp<std::size_t>(kScoreBudget / (std::size_t(columns) * 4), 4, std::size_t(t)));
            sub = std::max(4, sub / 4 * 4);
        }
        auto* scores = alloc<float>(std::size_t(std::min(sub, t)) * columns);
        // The candidate source's block maxima: a row for each row of scores, as many as the blocks
        // of the positions this pass sees (not the whole context's mask).
        const int maxima_stride =
            p.candidate_source ? (width + candidate_block - 1) / candidate_block : 0;
        float* maxima = p.candidate_source
                            ? alloc<float>(std::size_t(std::min(sub, t)) * maxima_stride)
                            : nullptr;
        // A chunk's groups list their blocks here, a decode pass's rows at its candidate source.
        const int list_groups = (std::min(sub, t) + group - 1) / group;
        std::int32_t* list    = listed && s.chunk ? alloc<std::int32_t>(std::size_t(list_groups) * capacity)
                                                  : nullptr;
        std::int32_t* counts  = listed && s.chunk ? alloc<std::int32_t>(std::size_t(list_groups)) : nullptr;
        for (int r0 = 0; r0 < t; r0 += sub) {
            const int n = std::min(sub, t - r0);
            kernels::ScoreArgs a;
            a.q            = q + std::int64_t(r0) * q_rows;
            a.weights      = weights + std::int64_t(r0) * index_heads;
            a.weight_scale = 1.0f / std::sqrt(float(kIndexDim)) / std::sqrt(float(index_heads));
            a.heads        = index_heads;
            a.tokens       = n;
            a.ratio        = p.ratio;
            a.width        = columns;
            a.group        = group;
            a.positions    = pos + r0;
            a.sequences    = seq != nullptr ? seq + r0 : nullptr;
            a.sequence     = int(s.sequence);
            a.keys                 = static_cast<const std::uint8_t*>(source.keys.p);
            a.key_seq_stride       = std::int64_t(source.keys.row_bytes);
            a.key_scales           = static_cast<const std::uint8_t*>(source.key_scales.p);
            a.key_scale_seq_stride = std::int64_t(source.key_scales.row_bytes);
            kernels::ListedRows rows;
            if (p.uses_candidates) {
                a.candidates      = candidates + std::int64_t(r0) * candidate_words;
                a.candidate_words = candidate_words;
                a.block           = candidate_block;
                if (listed) {
                    rows = kernels::ListedRows{
                        .list     = s.chunk ? list : s.candidate_list + std::int64_t(r0) * capacity,
                        .counts   = s.chunk ? counts : s.candidate_counts + r0,
                        .capacity = capacity,
                        .group    = group,
                        .block    = candidate_block};
                    if (s.chunk) {
                        kernels::candidate_lists(a.candidates, candidate_words, n, group, capacity,
                                                 list, counts, stream);
                    }
                    a.list     = rows.list;
                    a.counts   = rows.counts;
                    a.capacity = capacity;
                }
            }
            a.out = scores;
            kernels::index_scores(a, stream);
            if (p.candidate_source) {
                std::uint32_t* mask = candidates + std::int64_t(r0) * candidate_words;
                kernels::select_candidates(scores, width, n, pos + r0, p.ratio, candidate_block,
                                           candidate_keep, mask, candidate_words, maxima,
                                           maxima_stride, stream);
                // A decode or verification pass lists its rows' blocks once, for every later
                // index source.
                if (!s.chunk && s.candidate_list != nullptr && capacity > 0) {
                    kernels::candidate_lists(mask, candidate_words, n, 1, capacity,
                                             s.candidate_list + std::int64_t(r0) * capacity,
                                             s.candidate_counts + r0, stream);
                }
            }
            kernels::select_topk(scores, columns, n, pos + r0, p.ratio, index_topk, rows,
                                 topk + std::int64_t(r0) * index_topk, stream);
        }
    }

    // Whether a wide call of t tokens runs block p's routed experts of `type`, their gate and up
    // (`gate_up`) or their down products, on the integer matrix kernel (else the vector kernel in
    // groups of eight columns): IQ1_M's only with ExecutorOptions::iq1m_matrix and once the call
    // routes enough rows to each expert (gguf::iq1m_moe_matrix_pays).
    [[nodiscard]] bool matrix_route(const LayerPlan& p, bool gate_up, gguf::GgmlType type,
                                    int rows, int k, int t) const {
        if (type == gguf::GgmlType::IQ1_M &&
            (!options.iq1m_matrix ||
             !gguf::iq1m_moe_matrix_pays(gate_up, std::int64_t(t) * p.top_k, p.experts))) {
            return false;
        }
        return gguf::moe_matrix_supported(type, rows, k, true);
    }

    // The block's MoE: routed experts and the shared expert into y, added to the streams.
    void ffn(const LayerPlan& p, float* h, const float* pre, int t, const Coefficients& c) {
        auto scope = workspace->scope();
        Tensor x   = alloc_bf16(hidden, t);
        auto* x32  = alloc<float>(std::size_t(t) * hidden);
        kernels::hc_pre_norm(h, pre, p.ffn_norm, t, streams, hidden, eps,
                             static_cast<__nv_bfloat16*>(x.data), x32, stream);
        // Every product of x (the routed and the shared experts' gate and up rows) sees it guarded
        // for F16 sums; the clamped SwiGLU takes the factors back before its nonlinearity.
        auto* scale = alloc<float>(t);
        kernels::guard_columns(static_cast<__nv_bfloat16*>(x.data), hidden, t, scale, stream);
        auto* y = alloc<float>(std::size_t(t) * hidden);
        const auto shared = [&] {
            auto shared_scope = workspace->scope();
            auto* gu          = alloc<float>(std::size_t(t) * 2 * width);
            if (p.stq) {
                stq_project(p.stq->shared_gate, x, gu, 2 * width);
                stq_project(p.stq->shared_up, x, gu, 2 * width, width);
                Tensor middle = alloc_bf16(width, t);
                kernels::clamped_swiglu(gu, 2 * width, t, width, config.swiglu_limit, scale, 1,
                                        static_cast<__nv_bfloat16*>(middle.data), width, stream);
                stq_project(p.stq->shared_down, middle, y, hidden);
                return;
            }
            ops::detail::GgufProduct parts[2];
            int rows = 0;
            for (std::size_t i = 0; i < p.shared_gate_up.size(); ++i) {
                parts[i].weight   = &p.shared_gate_up[i];
                parts[i].f32      = gu;
                parts[i].f32_rows = 2 * width;
                parts[i].row      = rows;
                rows += p.shared_gate_up[i].n;
            }
            ops::detail::gguf_project(x, {parts, p.shared_gate_up.size()}, *workspace, stream);
            Tensor middle = alloc_bf16(width, t);
            kernels::clamped_swiglu(gu, 2 * width, t, width, config.swiglu_limit, scale, 1,
                                    static_cast<__nv_bfloat16*>(middle.data), width, stream);
            project(middle, p.shared_down, y, hidden);
        };
        routed(p, x, scale, x32, t, y, shared);
        kernels::hc_post(h, y, c.post, c.comb, t, streams, hidden, stream);
    }

    // One rank's part of a MoE call of block `p` under expert parallelism: the call's pairs routed
    // to the experts `range` the rank serves, through its delivery `from` (whose tables hold those
    // experts alone), weighted into the 2^-32 fixed-point sum `fixed` ([T][H]). Up to 8 tokens read
    // the call's q8_1 activation `xq`, wider calls its BF16 rows `xb`; `scale` holds the guard's
    // factors. Each contribution is the one a single device computes and rounds.
    void expert_part(const LayerPlan& p, ExpertDelivery& from,
                     std::pair<std::uint32_t, std::uint32_t> range, const void* xq,
                     const __nv_bfloat16* xb, const float* scale, const std::int32_t* ids,
                     const float* weights, int t, unsigned long long* fixed, WorkspaceArena& space,
                     cudaStream_t on) {
        auto scope  = space.scope();
        const int k = p.top_k, w = width, pairs = t * k;
        const int n     = int(range.second - range.first);
        const auto take = [&](std::size_t bytes) { return space.alloc_bytes(bytes).data; };
        auto* mine      = static_cast<std::int32_t*>(take(std::size_t(pairs) * 4));
        gguf::moe_local_experts(ids, pairs, int(range.first), int(range.second), mine, on);
        const gguf::MoeRouting routing =
            gguf::moe_sort_routes(mine, pairs, n, take(gguf::moe_routing_bytes(n, pairs)), on);
        const ExpertDelivery::Call call =
            from.begin(std::size_t(p.delivery_layer), mine, std::uint32_t(pairs), on);
        const auto* tables = from.tables(std::size_t(p.delivery_layer));
        const gguf::MoeTable gate_up{tables + 2 * n, p.gate_up_row_bytes, 2 * w, hidden};
        const gguf::MoeTable down{tables + 3 * n, p.down_row_bytes, hidden, w};
        const int max_active = std::max(1, std::min(pairs, n));
        auto* gu             = static_cast<float*>(take(std::size_t(pairs) * 2 * w * 4));
        auto* middle         = static_cast<__nv_bfloat16*>(take(std::size_t(pairs) * w * 2));
        if (t <= gguf::kMaxVectorColumns) {
            const int chunk = t <= 1 ? 1 : t <= 2 ? 2 : t <= 4 ? 4 : 8;
            void* act2      = take(gguf::vector_activation_bytes(w, pairs));
            CUDA_CHECK(cudaMemsetAsync(fixed, 0, std::size_t(t) * hidden * 8, on));
            CUDA_CHECK(cudaMemsetAsync(gu, 0, std::size_t(pairs) * 2 * w * 4, on));
            gguf::MoeOutput rows;
            rows.f32 = gu;
            gguf::MoeOutput out;
            out.weighted       = fixed;
            out.weights        = weights;
            const auto compute = [&](const std::int32_t* set) {
                gguf::MoeRouting part = routing;
                part.active_count     = set;
                part.active           = set + 1;
                gguf::moe_vector_product(p.gate_up_type, gate_up, part, max_active, k, k, xq, t,
                                         rows, chunk, on);
                kernels::clamped_swiglu(gu, 2 * w, pairs, w, config.swiglu_limit, scale, k, middle,
                                        w, on);
                gguf::quantize_vector_activation(middle, w, pairs, nullptr, act2, on);
                gguf::moe_vector_product(p.down_type, down, part, max_active, 1, k, act2, pairs,
                                         out, chunk, on);
            };
            compute(call.cached_experts);
            from.await(on);
            compute(call.missing_experts);
            return;
        }
        from.await(on);
        if (matrix_route(p, true, p.gate_up_type, 2 * w, hidden, t)) {
            void* act = take(gguf::moe_matrix_activation_bytes(hidden, pairs));
            gguf::quantize_moe_matrix_activation(p.gate_up_type, xb, hidden, routing, pairs, k, act,
                                                 on);
            gguf::moe_matrix_product(p.gate_up_type, gate_up, routing, max_active, pairs, true, act,
                                     gu, on);
        } else {
            void* act = take(gguf::vector_activation_bytes(hidden, t));
            gguf::quantize_vector_activation(xb, hidden, t, nullptr, act, on);
            gguf::MoeOutput out;
            out.f32 = gu;
            gguf::moe_vector_product(p.gate_up_type, gate_up, routing, max_active, k, k, act, t,
                                     out, 8, on);
        }
        kernels::clamped_swiglu(gu, 2 * w, pairs, w, config.swiglu_limit, scale, k, middle, w, on);
        auto* rows = static_cast<float*>(take(std::size_t(pairs) * hidden * 4));
        if (matrix_route(p, false, p.down_type, hidden, w, t)) {
            void* act = take(gguf::moe_matrix_activation_bytes(w, pairs));
            gguf::quantize_moe_matrix_activation(p.down_type, middle, w, routing, pairs, 1, act,
                                                 on);
            gguf::moe_matrix_product(p.down_type, down, routing, max_active, pairs, true, act, rows,
                                     on);
        } else {
            void* act = take(gguf::vector_activation_bytes(w, pairs));
            gguf::quantize_vector_activation(middle, w, pairs, nullptr, act, on);
            gguf::MoeOutput out;
            out.f32 = rows;
            gguf::moe_vector_product(p.down_type, down, routing, max_active, 1, k, act, pairs, out,
                                     8, on);
        }
        gguf::moe_weighted_rows(rows, mine, weights, t, k, hidden, fixed, on);
    }

    // Expert parallelism's routed experts after rank 0 has routed the call: the input goes to every
    // other rank, each runs its part while rank 0 runs its own and the shared expert, and the
    // parts' sum is added to y. Up to 8 tokens this is the sum one device adds, bit for bit; a
    // wider call sums in the same fixed point where one device weighs FP32 rows in moe_combine.
    template <typename Shared>
    void routed_parallel(const LayerPlan& p, const __nv_bfloat16* xb, const void* xq,
                         const float* scale, const std::int32_t* ids, const float* weights, int t,
                         float* y, Shared&& shared) {
        const std::size_t pairs  = std::size_t(t) * p.top_k;
        const std::size_t part   = std::size_t(t) * hidden * 8;
        const ExpertInput layout = expert_input(t);
        auto* pack               = static_cast<std::byte*>(expert_pack.p);
        CUDA_CHECK(cudaMemcpyAsync(pack, t <= gguf::kMaxVectorColumns ? xq : xb, layout.activation,
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(pack + layout.scale, scale, std::size_t(t) * 4,
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
            expert_part(p, *peer.delivery, expert_ranges[r], input,
                        reinterpret_cast<const __nv_bfloat16*>(input),
                        reinterpret_cast<const float*>(input + layout.scale),
                        reinterpret_cast<const std::int32_t*>(input + layout.ids),
                        reinterpret_cast<const float*>(input + layout.weights), t,
                        static_cast<unsigned long long*>(expert_exchange->part(r)), *peer.workspace,
                        peer.stream);
            expert_exchange->reply(r, part, slot, peer.stream);
        }
        auto* fixed = alloc<unsigned long long>(std::size_t(t) * hidden);
        expert_part(p, *delivery, expert_ranges[0], xq, xb, scale, ids, weights, t, fixed,
                    *workspace, stream);
        shared();
        std::vector<const unsigned long long*> parts;
        for (void* received : expert_exchange->collect(part, slot, stream)) {
            parts.push_back(static_cast<const unsigned long long*>(received));
        }
        gguf::add_fixed_parts(fixed, parts.data(), int(parts.size()), std::int64_t(t) * hidden,
                              stream);
        kernels::MoeCombineArgs combine;
        combine.y      = y;
        combine.tokens = t;
        combine.hidden = hidden;
        combine.fixed  = fixed;
        kernels::moe_combine(combine, stream);
    }

    // The routed experts of the block (their output added to y after `shared` stores the shared
    // expert's there), run while missing experts are copied in.
    template <typename Shared>
    void routed(const LayerPlan& p, const Tensor& x, const float* scale, const float* x32, int t,
                float* y, Shared&& shared) {
        const int e = p.experts, k = p.top_k, w = width, pairs = t * k;
        auto* logits = alloc<float>(std::size_t(t) * e);
        kernels::gemm_f32(e, t, hidden, p.router, x32, logits, stream);
        auto* ids     = alloc<std::int32_t>(pairs);
        auto* weights = alloc<float>(pairs);
        kernels::route(logits, p.router_bias, t, e, k, config.routed_scaling_factor, ids, weights,
                       stream, p.image_bias, p.image_bias != nullptr ? image_mask : nullptr);
        const auto* xb   = static_cast<const __nv_bfloat16*>(x.data);
        if (p.stq) {
            // An STQ drafter block's resident experts: a pair per routed expert of a row through its
            // gate, up and down matrices, the rows weighted into y.
            shared();
            auto* gu     = alloc<float>(std::size_t(pairs) * 2 * w);
            auto* middle = alloc<__nv_bfloat16>(std::size_t(pairs) * w);
            auto* rows   = alloc<float>(std::size_t(pairs) * hidden);
            ops::stq::Product gate;
            gate.m               = p.stq->gate;
            gate.ids             = ids;
            gate.items           = pairs;
            gate.x               = xb;
            gate.x_item_stride   = hidden;
            gate.x_divisor       = k;
            gate.out             = gu;
            gate.out_item_stride = 2 * w;
            stq_run(gate);
            ops::stq::Product up = gate;
            up.m                 = p.stq->up;
            up.out               = gu + w;
            stq_run(up);
            kernels::clamped_swiglu(gu, 2 * w, pairs, w, config.swiglu_limit, scale, k, middle, w,
                                    stream);
            ops::stq::Product down;
            down.m               = p.stq->down;
            down.ids             = ids;
            down.items           = pairs;
            down.x               = middle;
            down.x_item_stride   = w;
            down.out             = rows;
            down.out_item_stride = hidden;
            stq_run(down);
            kernels::MoeCombineArgs combine;
            combine.y       = y;
            combine.tokens  = t;
            combine.hidden  = hidden;
            combine.rows    = rows;
            combine.weights = weights;
            combine.top_k   = k;
            kernels::moe_combine(combine, stream);
            return;
        }
        const bool vector = t <= gguf::kMaxVectorColumns;
        void* xq          = nullptr; // the vector kernels' activation of x
        if (vector) {
            xq = workspace->alloc_bytes(gguf::vector_activation_bytes(hidden, t)).data;
            gguf::quantize_vector_activation(xb, hidden, t, nullptr, xq, stream);
        }
        if (!expert_ranges.empty() && p.delivery_layer >= 0) {
            routed_parallel(p, xb, xq, scale, ids, weights, t, y, shared);
            return;
        }
        void* routing_space = workspace->alloc_bytes(gguf::moe_routing_bytes(e, pairs)).data;
        const gguf::MoeRouting routing =
            gguf::moe_sort_routes(ids, pairs, e, routing_space, stream);
        // A decode or verification call may leave misses to the CPU, which reads the quantized
        // activation, the route weights and the guard's factors.
        const bool cpu = cpu_experts && vector && t <= CpuExperts::kMaxTokens;
        std::optional<ExpertDelivery::Call> call;
        if (p.delivery_layer >= 0) {
            const std::array<ExpertDelivery::Input, 3> inputs{
                ExpertDelivery::Input{xq, cpu ? cpu_experts->activation_bytes(t) : 0},
                ExpertDelivery::Input{weights, cpu ? cpu_experts->weight_bytes(t) : 0},
                ExpertDelivery::Input{scale, cpu ? CpuExperts::scale_bytes(t) : 0}};
            call = delivery->begin(std::size_t(p.delivery_layer), ids, std::uint32_t(pairs), stream,
                                   cpu ? std::span<const ExpertDelivery::Input>(inputs)
                                       : std::span<const ExpertDelivery::Input>());
        }
        const auto* tables = p.delivery_layer >= 0 ? delivery->tables(std::size_t(p.delivery_layer))
                                                   : static_cast<const void* const*>(p.tables.p);
        const gguf::MoeTable gate_up{tables + 2 * e, p.gate_up_row_bytes, 2 * w, hidden};
        const gguf::MoeTable down{tables + 3 * e, p.down_row_bytes, hidden, w};
        const int max_active = std::min(pairs, e);
        auto* gu             = alloc<float>(std::size_t(pairs) * 2 * w);
        auto* middle         = alloc<__nv_bfloat16>(std::size_t(pairs) * w);
        kernels::MoeCombineArgs combine;
        combine.y      = y;
        combine.tokens = t;
        combine.hidden = hidden;
        if (vector) {
            const int chunk = t <= 1 ? 1 : t <= 2 ? 2 : t <= 4 ? 4 : 8;
            void* act2 = workspace->alloc_bytes(gguf::vector_activation_bytes(w, pairs)).data;
            auto* fixed = alloc<unsigned long long>(std::size_t(t) * hidden);
            CUDA_CHECK(cudaMemsetAsync(fixed, 0, std::size_t(t) * hidden * 8, stream));
            CUDA_CHECK(cudaMemsetAsync(gu, 0, std::size_t(pairs) * 2 * w * 4, stream));
            gguf::MoeOutput rows;
            rows.f32 = gu;
            gguf::MoeOutput out;
            out.weighted = fixed;
            out.weights  = weights;
            const auto compute = [&](const gguf::MoeRouting& part) {
                gguf::moe_vector_product(p.gate_up_type, gate_up, part, max_active, k, k, xq, t,
                                         rows, chunk, stream);
                kernels::clamped_swiglu(gu, 2 * w, pairs, w, config.swiglu_limit, scale, k, middle,
                                        w, stream);
                gguf::quantize_vector_activation(middle, w, pairs, nullptr, act2, stream);
                gguf::moe_vector_product(p.down_type, down, part, max_active, 1, k, act2, pairs,
                                         out, chunk, stream);
            };
            if (call) {
                // Both stages add to the same exact fixed-point sum, each over its experts of the
                // call's sorted routes; the shared expert runs between them, while the missing
                // experts arrive. The SwiGLU and its quantization read every pair: the cached stage
                // finds the other pairs' rows zero, the missing stage the cached pairs' rows.
                const auto stage = [&](const std::int32_t* set) {
                    gguf::MoeRouting part = routing;
                    part.active_count     = set;
                    part.active           = set + 1;
                    return part;
                };
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
            combine.fixed = fixed;
            kernels::moe_combine(combine, stream);
            return;
        }
        shared();
        if (call) { delivery->await(stream); }
        if (matrix_route(p, true, p.gate_up_type, 2 * w, hidden, t)) {
            void* act = workspace->alloc_bytes(gguf::moe_matrix_activation_bytes(hidden, pairs)).data;
            gguf::quantize_moe_matrix_activation(p.gate_up_type, xb, hidden, routing, pairs, k, act,
                                                 stream);
            gguf::moe_matrix_product(p.gate_up_type, gate_up, routing, max_active, pairs, true, act,
                                     gu, stream);
        } else {
            void* act = workspace->alloc_bytes(gguf::vector_activation_bytes(hidden, t)).data;
            gguf::quantize_vector_activation(xb, hidden, t, nullptr, act, stream);
            gguf::MoeOutput out;
            out.f32 = gu;
            gguf::moe_vector_product(p.gate_up_type, gate_up, routing, max_active, k, k, act, t, out,
                                     8, stream);
        }
        kernels::clamped_swiglu(gu, 2 * w, pairs, w, config.swiglu_limit, scale, k, middle, w,
                                stream);
        auto* rows = alloc<float>(std::size_t(pairs) * hidden);
        if (matrix_route(p, false, p.down_type, hidden, w, t)) {
            void* act = workspace->alloc_bytes(gguf::moe_matrix_activation_bytes(w, pairs)).data;
            gguf::quantize_moe_matrix_activation(p.down_type, middle, w, routing, pairs, 1, act,
                                                 stream);
            gguf::moe_matrix_product(p.down_type, down, routing, max_active, pairs, true, act, rows,
                                     stream);
        } else {
            void* act = workspace->alloc_bytes(gguf::vector_activation_bytes(w, pairs)).data;
            gguf::quantize_vector_activation(middle, w, pairs, nullptr, act, stream);
            gguf::MoeOutput out;
            out.f32 = rows;
            gguf::moe_vector_product(p.down_type, down, routing, max_active, 1, k, act, pairs, out, 8,
                                     stream);
        }
        combine.rows    = rows;
        combine.weights = weights;
        combine.top_k   = k;
        kernels::moe_combine(combine, stream);
    }

    // The head of `rows` final-normed rows into logits_bf16 columns [first, first + rows).
    void head_logits(const __nv_bfloat16* rows_x, int rows, int first) {
        if (divided_head && rows <= gguf::kMaxVectorColumns) {
            auto scope = workspace->scope();
            Tensor x   = alloc_bf16(hidden, rows);
            CUDA_CHECK(cudaMemcpyAsync(x.data, rows_x, std::size_t(rows) * hidden * 2,
                                       cudaMemcpyDeviceToDevice, stream));
            std::vector<cudaStream_t> streams{stream};
            for (std::size_t r = 1; r < expert_peers.size(); ++r) {
                streams.push_back(expert_peers[r].stream);
            }
            divided_head->project(
                x.data, std::size_t(rows) * hidden * 2, rows,
                static_cast<__nv_bfloat16*>(logits_bf16.p) + std::int64_t(first) * vocab, streams,
                [&](std::size_t r, const void* input, const Weight& piece, int columns,
                    float* plane, __nv_bfloat16* out, cudaStream_t on) {
                    WorkspaceArena& space = r == 0 ? *workspace : *expert_peers[r].workspace;
                    auto inner            = space.scope();
                    Tensor rows_in(const_cast<void*>(input), DType::BF16, {hidden, columns});
                    project_on(rows_in, piece, plane, space, on);
                    kernels::f32_to_bf16(plane, out, std::int64_t(columns) * piece.n, on);
                });
            return;
        }
        for (int at = 0; at < rows; at += kHeadRows) {
            const int n = std::min(kHeadRows, rows - at);
            auto scope  = workspace->scope();
            Tensor x    = alloc_bf16(hidden, n);
            CUDA_CHECK(cudaMemcpyAsync(x.data, rows_x + std::int64_t(at) * hidden,
                                       std::size_t(n) * hidden * 2, cudaMemcpyDeviceToDevice, stream));
            auto* plane = alloc<float>(std::size_t(n) * vocab);
            project(x, head, plane, vocab);
            kernels::f32_to_bf16(plane,
                                 static_cast<__nv_bfloat16*>(logits_bf16.p) + std::int64_t(first + at) * vocab,
                                 std::int64_t(n) * vocab, stream);
        }
    }

    // ---- The DSpark drafter -----------------------------------------------------------------------

    // The streams' mean of `n` rows of h entering backbone block l, when the drafter taps it, into
    // the taps from row `at`.
    void tap(std::size_t l, const float* rows, int n, int at) {
        const int j = drafter->target_of[l];
        if (j < 0 || n <= 0) { return; }
        kernels::stream_mean(rows, n, streams, hidden,
                             static_cast<float*>(drafter->taps.p) +
                                 (std::size_t(at) * drafter->targets + std::size_t(j)) * hidden,
                             std::int64_t(drafter->targets) * hidden, stream);
    }

    // The drafter's ring rows of `n` taps from row `at`: the taps through the projection and its
    // norm, then each block's kv of that, normed, turned and rounded as the window latents are, at
    // positions pos of `sequence` (of seq[r] each when given).
    void seed(int n, int at, const std::int32_t* pos, const std::int32_t* seq, std::uint32_t sequence) {
        Drafter& d       = *drafter;
        auto scope       = workspace->scope();
        const int inputs = d.targets * hidden;
        Tensor x         = alloc_bf16(inputs, n);
        kernels::f32_to_bf16(static_cast<const float*>(d.taps.p) + std::size_t(at) * inputs,
                             static_cast<__nv_bfloat16*>(x.data), std::int64_t(n) * inputs, stream);
        auto* main = alloc<float>(std::size_t(n) * hidden);
        if (d.stq_fc) {
            stq_project(*d.stq_fc, x, main, hidden);
        } else {
            project(x, d.fc, main, hidden);
        }
        Tensor normed = alloc_bf16(hidden, n);
        kernels::rms_rows(main, hidden, hidden, n, d.tap_norm, eps,
                          static_cast<__nv_bfloat16*>(normed.data), nullptr, stream);
        const int kv_rows = int(d.layers.size()) * kLatent;
        auto* kv          = alloc<float>(std::size_t(n) * kv_rows);
        if (d.stq_fc) {
            for (std::size_t b = 0; b < d.layers.size(); ++b) {
                stq_project(d.layers[b].stq->kv, normed, kv, kv_rows, int(b) * kLatent);
            }
        } else {
            project(normed, d.kv, kv, kv_rows);
        }
        for (std::size_t b = 0; b < d.layers.size(); ++b) {
            const LayerPlan& p = d.layers[b];
            kernels::KvPostArgs a;
            a.kv         = kv + b * kLatent;
            a.kv_stride  = kv_rows;
            a.tokens     = n;
            a.norm       = p.kv_norm;
            a.eps        = eps;
            a.positions  = pos;
            a.freq       = p.freq;
            a.cache      = static_cast<std::uint8_t*>(p.ring.p);
            a.seq_stride = std::int64_t(kRingBytes);
            a.sequences  = seq;
            a.sequence   = int(sequence);
            a.ring       = kRing;
            kernels::kv_post(a, stream);
        }
    }

    // The drafter's block over staged rows (the anchor, then the noise token, at the sequence's
    // next positions; then each row's visible position, the block's last): its blocks, the head
    // over the final norm of the collapsed streams, the Markov chain's greedy drafts and their
    // confidences, copied to the pinned buffer.
    void draft_pass(int b) {
        Drafter& d = *drafter;
        workspace->reset();
        auto scope     = workspace->scope();
        auto* h        = alloc<float>(std::size_t(b) * streams * hidden);
        embed(b, h);
        const int site = 2 * streams + streams * streams;
        Coefficients attention_site, ffn_site;
        for (Coefficients* c : {&attention_site, &ffn_site}) {
            auto* base = alloc<float>(std::size_t(b) * site);
            c->pre     = base;
            c->post    = base + std::size_t(b) * streams;
            c->comb    = base + std::size_t(b) * 2 * streams;
        }
        auto* mix_scratch = alloc<float>(std::size_t(b) * config.hc_mixes());
        Pass s;
        s.t              = b;
        s.visible        = static_cast<const std::int32_t*>(d_positions.p) + b;
        s.extra_window   = b;
        const float* pre = nullptr;
        for (const LayerPlan& p : d.layers) {
            mixes(p.hc_attention, h, b, mix_scratch, attention_site);
            attention(p, h, pre, b, s, nullptr, nullptr, attention_site);
            mixes(p.hc_ffn, h, b, mix_scratch, ffn_site);
            ffn(p, h, attention_site.pre, b, ffn_site);
            pre = ffn_site.pre;
        }
        Tensor x = alloc_bf16(hidden, b);
        kernels::hc_pre_norm(h, pre, d.final_norm, b, streams, hidden, eps,
                             static_cast<__nv_bfloat16*>(x.data), nullptr, stream);
        auto* plane = alloc<float>(std::size_t(b) * vocab);
        project(x, head, plane, vocab);
        auto* ids = static_cast<std::int32_t*>(d.ids.p);
        CUDA_CHECK(cudaMemcpyAsync(ids, d_tokens.p, 4, cudaMemcpyDeviceToDevice, stream));
        for (int i = 0; i < b; ++i) {
            kernels::markov_step(plane + std::size_t(i) * vocab, d.markov_head, d.markov_embed, d.domain,
                                 ids, i, d.scratch.p, stream);
        }
        kernels::dspark_confidence(h, pre, b, streams, hidden, d.confidence, d.markov_embed, ids,
                                   static_cast<float*>(d.values.p), stream);
        auto* host = static_cast<std::int32_t*>(d.host->data());
        CUDA_CHECK(cudaMemcpyAsync(host, ids + 1, std::size_t(b) * 4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host + b, d.values.p, std::size_t(b) * 4, cudaMemcpyDeviceToHost,
                                   stream));
    }

    void draft(std::uint32_t sequence, std::int32_t anchor, std::span<std::int32_t> drafts,
               std::span<float> confidence) {
        if (!drafter) { throw std::logic_error("deepseek_v41 draft: no DSpark drafter"); }
        Drafter& d  = *drafter;
        const int b = d.block;
        if (sequence >= options.sequences || drafts.size() != std::size_t(b) ||
            confidence.size() != std::size_t(b) || anchor < 0 || anchor >= vocab) {
            throw std::invalid_argument("deepseek_v41 draft: invalid call");
        }
        require_resumable(sequence, "draft");
        const std::uint32_t first = positions[sequence];
        if (first + std::uint32_t(b) > options.max_context) {
            throw std::invalid_argument("deepseek_v41 draft: context capacity exceeded");
        }
        std::vector<std::int32_t> tokens(b, d.noise), pos(2 * b), seq(b, std::int32_t(sequence));
        tokens[0] = anchor;
        for (int i = 0; i < b; ++i) {
            pos[std::size_t(i)]     = std::int32_t(first) + i;
            pos[std::size_t(b + i)] = std::int32_t(first) + b - 1;
        }
        stage(tokens, pos, seq);
        run_graph(32 + std::uint32_t(b), [&] { draft_pass(b); });
        d.copied->record(stream);
        d.copied->synchronize();
        const auto* host = static_cast<const std::int32_t*>(d.host->data());
        std::copy_n(host, b, drafts.begin());
        std::memcpy(confidence.data(), host + b, std::size_t(b) * 4);
    }

    // A pass over staged rows (tokens, positions and, for decode rows, sequences on the device; the
    // Engram rows copied).
    void run(const Pass& s) {
        const int t = s.t;
        image_mask  = step_images.empty() ? nullptr : static_cast<const std::uint8_t*>(d_image.p) + s.row;
        workspace->reset();
        auto scope = workspace->scope();
        auto* h    = alloc<float>(std::size_t(t) * streams * hidden);
        embed(t, h);
        const int site = 2 * streams + streams * streams;
        Coefficients attention_site, ffn_site;
        for (Coefficients* c : {&attention_site, &ffn_site}) {
            auto* base = alloc<float>(std::size_t(t) * site);
            c->pre     = base;
            c->post    = base + std::size_t(t) * streams;
            c->comb    = base + std::size_t(t) * 2 * streams;
        }
        auto* mix_scratch = alloc<float>(std::size_t(t) * config.hc_mixes());
        auto* topk        = alloc<std::int32_t>(std::size_t(t) * index_topk);
        auto* candidates  = candidate_words > 0 ? alloc<std::uint32_t>(std::size_t(t) * candidate_words)
                                                : nullptr;
        // A decode or verification pass's rows list their candidate blocks once.
        Pass listed = s;
        if (const int capacity = candidates != nullptr && !s.chunk
                                     ? listed_capacity(candidate_rows, false)
                                     : 0;
            capacity > 0) {
            listed.candidate_list   = alloc<std::int32_t>(std::size_t(t) * capacity);
            listed.candidate_counts = alloc<std::int32_t>(std::size_t(t));
        }
        const float* pre = nullptr; // the first block reads stream 0
        // With the drafter, the pass's last rows (a window at most, or a prompt call's within the
        // window before the state its caller keeps) tap its target blocks.
        const int tap_row = !drafter ? t : s.tap_row >= 0 ? s.tap_row : t - std::min(t, window);
        const int tapped  = t - tap_row;
        const auto rows_of = [&](const Coefficients& co, int row) {
            return Coefficients{co.pre + std::size_t(row) * streams, co.post + std::size_t(row) * streams,
                                co.comb + std::size_t(row) * streams * streams};
        };
        const std::uint8_t* const images = image_mask;
        // A DSpine drafter's taps: a prompt call's rows it keeps, a decode or verification pass's
        // every row.
        const int spine_row = !spine ? t : s.chunk ? s.spine_row : 0;
        for (std::size_t l = 0; l < layers.size(); ++l) {
            const LayerPlan& p = layers[l];
            if (l > 0) { spine_tap(l - 1, h, spine_row, t, spine_row); }
            // A prompt call under a prefill tail runs its decoder blocks over its rows from
            // decoder_row on; every other block, and every other pass, over all of them.
            const int r = p.decoder ? s.decoder_row : 0;
            if (r >= t) { continue; }
            if (tapped > 0) { tap(l, h + std::size_t(tap_row) * streams * hidden, tapped, 0); }
            Pass part = listed;
            if (r > 0) {
                part.t      = t - r;
                part.row    = s.row + r;
                part.staged = s.staged != 0 ? s.staged : t;
                part.first  = s.first + std::uint32_t(r);
            }
            float* hr            = h + std::size_t(r) * streams * hidden;
            const Coefficients a = rows_of(attention_site, r), f = rows_of(ffn_site, r);
            image_mask           = images == nullptr ? nullptr : images + r;
            if (p.engram >= 0) { engram_block(p, hr, part.t, part); }
            mixes(p.hc_attention, hr, part.t, mix_scratch, a);
            const bool kv_only = s.kv_only && l + 1 == layers.size();
            attention(p, hr, pre == nullptr ? nullptr : pre + std::size_t(r) * streams, part.t,
                      part, topk + std::size_t(r) * index_topk,
                      candidates == nullptr ? nullptr
                                            : candidates + std::size_t(r) * candidate_words,
                      a, kv_only);
            if (kv_only) { break; }
            mixes(p.hc_ffn, hr, part.t, mix_scratch, f);
            ffn(p, hr, a.pre, part.t, f);
            pre = ffn_site.pre;
        }
        image_mask = images;
        if (!s.kv_only) { spine_tap(layers.size() - 1, h, spine_row, t, spine_row); }
        if (spine && !s.chunk && s.seed) {
            // A decode step's rows continue their sequences' lanes (a verification's wait for
            // commit()).
            spine->commit_rows(Tensor(d_sequences.p, DType::I32, {t}),
                               Tensor(d_positions.p, DType::I32, {t}), 0, stream);
        }
        if (tapped > 0 && s.seed) {
            const int from = s.row + tap_row;
            seed(tapped, 0, static_cast<const std::int32_t*>(d_positions.p) + from,
                 s.chunk ? nullptr : static_cast<const std::int32_t*>(d_sequences.p) + from, s.sequence);
        }
        if (s.logit_rows == 0) { return; }
        const int first = t - s.logit_rows;
        Tensor x        = alloc_bf16(hidden, s.logit_rows);
        kernels::hc_pre_norm(h + std::int64_t(first) * streams * hidden, pre + std::int64_t(first) * streams,
                             final_norm, s.logit_rows, streams, hidden, eps,
                             static_cast<__nv_bfloat16*>(x.data), nullptr, stream);
        head_logits(static_cast<const __nv_bfloat16*>(x.data), s.logit_rows, s.logit_column);
    }

    // ---- Calls -----------------------------------------------------------------------------------

    [[nodiscard]] std::uint32_t prompt_step() const noexcept {
        return span_tokens != 0 ? span_tokens : options.prefill_chunk;
    }

    // A step's images, kept while it runs.
    struct StepImages {
        Impl& impl;
        ~StepImages() {
            impl.step_images.clear();
            impl.step_dead.clear();
            impl.image_mask = nullptr;
        }
    };

    // Checks a step's images against its t rows and stages their flags.
    void stage_images(std::span<const ImageRows> images, int t) {
        if (images.empty()) { return; }
        if (!tower) { throw std::invalid_argument("deepseek_v41 forward: images need the vision tower"); }
        step_dead.assign(std::size_t(t), 0);
        std::size_t next = 0;
        for (const ImageRows& piece : images) {
            const std::size_t count = piece.embeddings.size() / std::size_t(hidden);
            if (count == 0 || piece.embeddings.size() % std::size_t(hidden) != 0 || piece.row < next ||
                piece.row + count > std::size_t(t) || count > tower->config().max_image_tokens) {
                throw std::invalid_argument("deepseek_v41 forward: image rows outside the call or "
                                            "out of order");
            }
            std::fill_n(step_dead.begin() + piece.row, count, std::uint8_t(1));
            next = piece.row + count;
        }
        step_images.assign(images.begin(), images.end());
        image_staged->synchronize();
        std::memcpy(h_image->data(), step_dead.data(), step_dead.size());
        CUDA_CHECK(cudaMemcpyAsync(d_image.p, h_image->data(), step_dead.size(),
                                   cudaMemcpyHostToDevice, stream));
        image_staged->record(stream);
    }

    // A flag for each of t rows an image of `images` takes (empty: none does).
    [[nodiscard]] std::vector<std::uint8_t> image_flags(std::span<const ImageRows> images,
                                                        std::size_t t) const {
        std::vector<std::uint8_t> flags;
        for (const ImageRows& piece : images) {
            if (flags.empty()) { flags.assign(t, 0); }
            const std::size_t from  = std::min<std::size_t>(piece.row, t);
            const std::size_t count = piece.embeddings.size() / std::size_t(hidden);
            std::fill_n(flags.begin() + std::ptrdiff_t(from), std::min(count, t - from), std::uint8_t(1));
        }
        return flags;
    }

    // The pieces of `images` within rows [begin, end), rows counted from begin.
    std::vector<ImageRows> slice_images(std::span<const ImageRows> images, std::size_t begin,
                                        std::size_t end) const {
        std::vector<ImageRows> out;
        for (const ImageRows& piece : images) {
            const std::size_t count = piece.embeddings.size() / std::size_t(hidden);
            const std::size_t from = std::max<std::size_t>(piece.row, begin);
            const std::size_t to   = std::min<std::size_t>(piece.row + count, end);
            if (from >= to) { continue; }
            out.push_back({std::uint32_t(from - begin),
                           piece.embeddings.subspan((from - piece.row) * std::size_t(hidden),
                                                    (to - from) * std::size_t(hidden))});
        }
        return out;
    }

    void forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                 std::uint32_t logit_rows, std::uint32_t logit_column = 0,
                 std::span<const ImageRows> images = {}, std::uint32_t state_at = 0,
                 NextPromptCall next = {}) {
        const int t = int(tokens.size());
        if (std::uint32_t(t) > options.prefill_chunk && logit_column == 0) {
            forward_span(sequence, tokens, logit_rows, images, state_at, next);
            return;
        }
        if (sequence >= options.sequences || t <= 0 || std::uint32_t(t) > options.prefill_chunk ||
            logit_rows > std::min<std::uint32_t>(t, kMaxLogitRows) ||
            logit_column + logit_rows > kMaxLogitRows) {
            throw std::invalid_argument("deepseek_v41 forward: invalid call");
        }
        const std::uint32_t first = positions[sequence];
        if (first + std::uint32_t(t) > options.max_context) {
            throw std::invalid_argument("deepseek_v41 forward: context capacity exceeded");
        }
        const int decoder_row = decoder_start(sequence, first, t, logit_rows, state_at);
        back(sequence, first + std::uint32_t(t));
        std::vector<std::int32_t> pos(t);
        for (int i = 0; i < t; ++i) { pos[i] = std::int32_t(first) + i; }
        StepImages kept{*this};
        stage(tokens, pos, {});
        stage_images(images, t);
        read_engram_rows(tokens, std::vector<std::uint32_t>(std::size_t(t), sequence), true, step_dead,
                         true);
        read_ahead(sequence, first + std::uint32_t(t), next);
        Pass s;
        s.t            = t;
        s.chunk        = true;
        s.sequence     = sequence;
        s.first        = first;
        s.logit_rows   = int(logit_rows);
        s.logit_column = int(logit_column);
        s.decoder_row  = decoder_row;
        s.tap_row      = tap_start(first, t, state_at);
        s.kv_only      = logit_rows == 0 && !spine_taps_last();
        s.spine_row    = spine_start(first, t, state_at);
        run(s);
        if (s.spine_row < t) {
            spine->commit(spine->lane(sequence), first + std::uint32_t(s.spine_row), 0,
                          std::uint32_t(t - s.spine_row), stream);
        }
        decoder_ran(sequence, first + std::uint32_t(t), decoder_row, t);
        positions[sequence] = first + std::uint32_t(t);
    }

    // A prompt span of several chunks, layer by layer: every chunk passes a block before any
    // passes the next, so the block's wide calls bring the experts the cache lacks into one layer
    // bank for every chunk of the span while the delivery prefetches the next block's bank. The
    // span's residual streams, its blocks' coefficients and its index selections stay between
    // blocks in memory borrowed from an elastic expert cache (borrow_span), else in the workspace;
    // each chunk's arithmetic is forward()'s, and the logits are the span's last rows.
    void forward_span(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                      std::uint32_t logit_rows, std::span<const ImageRows> images = {},
                      std::uint32_t state_at = 0, NextPromptCall next = {}) {
        const int n = int(tokens.size()), c = int(options.prefill_chunk);
        if (sequence >= options.sequences || std::uint32_t(n) > span_tokens ||
            logit_rows > std::min<std::uint32_t>(n, kMaxLogitRows)) {
            throw std::invalid_argument("deepseek_v41 forward: invalid prompt span");
        }
        const std::uint32_t first = positions[sequence];
        if (first + std::uint32_t(n) > options.max_context) {
            throw std::invalid_argument("deepseek_v41 forward: context capacity exceeded");
        }
        back(sequence, first + std::uint32_t(n));
        DeviceBuffer borrowed;
        if (span_borrow_bytes != 0) {
            borrowed = borrow_span(std::uint32_t(n));
            if (!borrowed.p) {
                // The cache cannot lend the span's buffers now: its halves (whole chunks) as spans
                // whose buffers it may lend, down to chunk by chunk.
                const int half = (n / 2 + c - 1) / c * c;
                const auto first_images = slice_images(images, 0, std::size_t(half));
                const auto last_images  = slice_images(images, std::size_t(half), std::size_t(n));
                const auto last_flags   = image_flags(last_images, std::size_t(n - half));
                forward(sequence, tokens.first(std::size_t(half)), 0, 0, first_images,
                        state_at != 0 ? state_at : first + std::uint32_t(n),
                        {tokens.subspan(std::size_t(half)), last_flags});
                forward(sequence, tokens.subspan(std::size_t(half)), logit_rows, 0, last_images,
                        state_at, next);
                return;
            }
        }
        // The span's rows the decoder blocks run (all of them but under a prefill tail) and the
        // rows the drafter taps: those within the window before the state the caller keeps.
        const int decoder_row = decoder_start(sequence, first, n, logit_rows, state_at);
        const int tap_row     = drafter ? tap_start(first, n, state_at) : n;
        const int tapped      = n - tap_row;
        std::size_t taken = 0;
        // The span's buffers: from the borrowed memory (in span_buffers' order), else the workspace.
        const auto span_alloc = [&](std::size_t bytes) -> void* {
            if (!borrowed.p) { return workspace->alloc_bytes(bytes).data; }
            void* p = static_cast<std::byte*>(borrowed.p) + taken;
            taken += round_up(bytes, kAlign);
            return p;
        };
        std::vector<std::int32_t> pos(n);
        for (int i = 0; i < n; ++i) { pos[i] = std::int32_t(first) + i; }
        StepImages kept{*this};
        stage(tokens, pos, {});
        stage_images(images, n);
        read_engram_rows(tokens, std::vector<std::uint32_t>(std::size_t(n), sequence), true, step_dead,
                         true);
        read_ahead(sequence, first + std::uint32_t(n), next);
        workspace->reset();
        auto scope = workspace->scope();
        auto* h    = static_cast<float*>(span_alloc(std::size_t(n) * streams * hidden * 4));
        embed(n, h);
        const int site = 2 * streams + streams * streams;
        Coefficients attention_site, ffn_site;
        for (Coefficients* co : {&attention_site, &ffn_site}) {
            auto* base = static_cast<float*>(span_alloc(std::size_t(n) * site * 4));
            co->pre    = base;
            co->post   = base + std::size_t(n) * streams;
            co->comb   = base + std::size_t(n) * 2 * streams;
        }
        auto* topk       = static_cast<std::int32_t*>(span_alloc(std::size_t(n) * index_topk * 4));
        auto* candidates = candidate_words > 0
                               ? static_cast<std::uint32_t*>(span_alloc(std::size_t(n) * candidate_words * 4))
                               : nullptr;
        auto* mix_scratch = alloc<float>(std::size_t(c) * config.hc_mixes());
        const auto rows_of = [&](const Coefficients& co, int row) {
            return Coefficients{co.pre + std::size_t(row) * streams, co.post + std::size_t(row) * streams,
                                co.comb + std::size_t(row) * streams * streams};
        };
        // The span's rows a DSpine drafter keeps.
        const int spine_row = spine_start(first, n, state_at);
        for (std::size_t l = 0; l < layers.size(); ++l) {
            const LayerPlan& p = layers[l];
            const int from     = p.decoder ? decoder_row : 0;
            for (int row = 0; row < n; row += c) {
                if (l > 0) {
                    spine_tap(l - 1, h, std::max(row, spine_row), std::min(row + c, n), spine_row);
                }
                // A chunk's rows from `from` on (under the Exact tail, whole chunks).
                const int lo = std::max(row, from), hi = std::min(row + c, n);
                if (lo >= hi) { continue; }
                Pass s;
                s.t        = hi - lo;
                s.chunk    = true;
                s.row      = lo;
                s.staged   = n;
                s.sequence = sequence;
                s.first    = first + std::uint32_t(lo);
                float* hc  = h + std::size_t(lo) * streams * hidden;
                if (const int at = std::max(lo, tap_row); tapped > 0 && at < hi) {
                    tap(l, h + std::size_t(at) * streams * hidden, hi - at, at - tap_row);
                }
                const Coefficients a = rows_of(attention_site, lo), f = rows_of(ffn_site, lo);
                image_mask = step_images.empty()
                                 ? nullptr
                                 : static_cast<const std::uint8_t*>(d_image.p) + lo;
                if (p.engram >= 0) { engram_block(p, hc, s.t, s); }
                mixes(p.hc_attention, hc, s.t, mix_scratch, a);
                // The last block of a chunk before the span's logit rows: its window latents alone
                // (unless a DSpine drafter reads its output).
                const bool kv_only =
                    l + 1 == layers.size() && hi <= n - int(logit_rows) && !spine_taps_last();
                // The first block reads stream 0; a later one the previous block's ffn site.
                attention(p, hc, l == 0 ? nullptr : f.pre, s.t, s,
                          topk + std::size_t(lo) * index_topk,
                          candidates ? candidates + std::size_t(lo) * candidate_words : nullptr, a,
                          kv_only);
                if (kv_only) { continue; }
                mixes(p.hc_ffn, hc, s.t, mix_scratch, f);
                ffn(p, hc, a.pre, s.t, f);
            }
        }
        if (tapped > 0) {
            seed(tapped, 0, static_cast<const std::int32_t*>(d_positions.p) + tap_row, nullptr,
                 sequence);
        }
        if (spine_row < n) {
            spine_tap(layers.size() - 1, h, spine_row, n, spine_row);
            spine->commit(spine->lane(sequence), first + std::uint32_t(spine_row), 0,
                          std::uint32_t(n - spine_row), stream);
        }
        if (logit_rows > 0) {
            const int last = n - int(logit_rows);
            Tensor x       = alloc_bf16(hidden, int(logit_rows));
            kernels::hc_pre_norm(h + std::int64_t(last) * streams * hidden,
                                 ffn_site.pre + std::int64_t(last) * streams, final_norm,
                                 int(logit_rows), streams, hidden, eps,
                                 static_cast<__nv_bfloat16*>(x.data), nullptr, stream);
            head_logits(static_cast<const __nv_bfloat16*>(x.data), int(logit_rows), 0);
        }
        give_back(borrowed);
        decoder_ran(sequence, first + std::uint32_t(n), decoder_row, n);
        positions[sequence] = first + std::uint32_t(n);
    }

    // The Engram rows of a pass whose row r belongs to owners[r], from that sequence's history.
    void read_engram_rows(std::span<const std::int32_t> tokens, std::span<const std::uint32_t> owners,
                          bool advance, std::span<const std::uint8_t> dead = {}, bool prompt = false) {
        if (!engram) { return; }
        if (prompt && use_ahead(owners.front(), tokens, dead)) {
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                advance_history(owners[i], tokens.subspan(i, 1),
                                dead.empty() ? dead : dead.subspan(i, 1));
            }
            return;
        }
        std::vector<EngramHasher::History> view(tokens.size());
        // Each run of rows of one sequence continues that sequence's history; a verification's
        // rows hash from a copy (advance false) and commit() advances the history.
        std::size_t r = 0;
        while (r < tokens.size()) {
            std::size_t end = r + 1;
            while (end < tokens.size() && owners[end] == owners[r]) { ++end; }
            view[r] = histories[owners[r]];
            r       = end;
        }
        read_engram(tokens, owners, view, false, dead);
        if (advance) {
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                advance_history(owners[i], tokens.subspan(i, 1),
                                dead.empty() ? dead : dead.subspan(i, 1));
            }
        }
    }

    void decode(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens) {
        const int b = int(sequences.size());
        if (b <= 0 || std::uint32_t(b) > options.sequences || tokens.size() != sequences.size() ||
            b > gguf::kMaxVectorColumns) {
            throw std::invalid_argument("deepseek_v41 decode: invalid batch");
        }
        std::vector<std::int32_t> pos(b), seq(b);
        for (int i = 0; i < b; ++i) {
            if (sequences[i] >= options.sequences ||
                std::count(sequences.begin(), sequences.begin() + i, sequences[i]) != 0) {
                throw std::invalid_argument("deepseek_v41 decode: invalid sequence");
            }
            if (positions[sequences[i]] + 1 > options.max_context) {
                throw std::invalid_argument("deepseek_v41 decode: context capacity exceeded");
            }
            require_resumable(sequences[i], "decode");
            pos[i] = std::int32_t(positions[sequences[i]]);
            seq[i] = std::int32_t(sequences[i]);
        }
        for (int i = 0; i < b; ++i) { back(sequences[i], positions[sequences[i]] + 1); }
        stage(tokens, pos, seq);
        read_engram_rows(tokens, sequences, true);
        Pass s;
        s.t          = b;
        s.logit_rows = b;
        run_graph(std::uint32_t(b), [&] { run(s); });
        for (int i = 0; i < b; ++i) { positions[sequences[i]] += 1; }
    }

    template <typename Body>
    void run_graph(std::uint32_t key, Body&& body) {
        if (!options.cuda_graphs) {
            body();
            return;
        }
        GraphEntry& entry = graphs[key];
        if (!entry.warmed) {
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

    void verify(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens,
                std::span<const std::uint32_t> widths = {}) {
        const int n = int(sequences.size());
        if (options.draft_tokens == 0 || n <= 0 ||
            (widths.empty() && tokens.size() % sequences.size() != 0) ||
            (!widths.empty() && widths.size() != sequences.size())) {
            throw std::invalid_argument("deepseek_v41 verify: invalid call");
        }
        std::vector<std::uint32_t> row_widths(widths.begin(), widths.end());
        if (row_widths.empty()) { row_widths.assign(n, std::uint32_t(tokens.size()) / n); }
        int rows = 0;
        for (const std::uint32_t w : row_widths) {
            if (w < 1 || w > options.draft_tokens + 1) {
                throw std::invalid_argument("deepseek_v41 verify: invalid width");
            }
            rows += int(w);
        }
        if (std::size_t(rows) != tokens.size() || rows > gguf::kMaxVectorColumns) {
            throw std::invalid_argument("deepseek_v41 verify: invalid width");
        }
        std::vector<std::int32_t> pos(rows), seq(rows);
        std::vector<std::uint32_t> owners(rows);
        verify_starts.assign(n, 0);
        for (int j = 0, at = 0; j < n; ++j) {
            const std::uint32_t s = sequences[j];
            const std::uint32_t w = row_widths[j];
            if (s >= options.sequences || std::count(sequences.begin(), sequences.begin() + j, s) != 0) {
                throw std::invalid_argument("deepseek_v41 verify: invalid sequence");
            }
            if (positions[s] + w > options.max_context) {
                throw std::invalid_argument("deepseek_v41 verify: context capacity exceeded");
            }
            require_resumable(s, "verify");
            verify_starts[j] = at;
            verified[s].assign(tokens.begin() + at, tokens.begin() + at + w);
            for (std::uint32_t i = 0; i < w; ++i, ++at) {
                pos[at]    = std::int32_t(positions[s] + i);
                seq[at]    = std::int32_t(s);
                owners[at] = s;
            }
        }
        for (int j = 0; j < n; ++j) { back(sequences[j], positions[sequences[j]] + row_widths[j]); }
        stage(tokens, pos, seq);
        read_engram_rows(tokens, owners, false);
        Pass s;
        s.t          = rows;
        s.logit_rows = rows;
        s.seed       = false; // commit() seeds the rows it keeps
        run_graph(16 + std::uint32_t(rows), [&] { run(s); });
        verify_widths = std::move(row_widths);
    }

    // `hold_after[j]` (0: none), at most kept[j]: `holds[j]` also receives the state after that
    // many of sequence j's tokens, as hold() keeps it. The verification wrote every row by its
    // position, so the device state a commit leaves serves each position of the kept rows: a
    // commit of fewer rows leaves the same bytes. Returns the rows whose state was kept, a bit
    // each.
    std::uint32_t commit(std::span<const std::uint32_t> sequences,
                         std::span<const std::uint32_t> kept,
                         std::span<const std::uint32_t> hold_after = {},
                         std::span<SequenceSnapshot* const> holds = {}) {
        if (kept.size() != sequences.size() || verify_widths.size() != sequences.size() ||
            hold_after.size() != holds.size() ||
            (!hold_after.empty() && hold_after.size() != sequences.size())) {
            throw std::invalid_argument("deepseek_v41 commit: invalid call");
        }
        std::uint32_t held = 0;
        for (std::size_t j = 0; j < sequences.size(); ++j) {
            const std::uint32_t s = sequences[j];
            if (s >= options.sequences || kept[j] < 1 || kept[j] > verify_widths[j] ||
                verified[s].size() != verify_widths[j] ||
                (!hold_after.empty() && hold_after[j] > kept[j])) {
                throw std::invalid_argument("deepseek_v41 commit: invalid row");
            }
            const std::span<const std::int32_t> tokens(verified[s]);
            const std::uint32_t before = hold_after.empty() ? 0 : hold_after[j];
            if (before != 0 && holds[j] != nullptr && hold_bytes != 0 && resumable(s)) {
                advance_history(s, tokens.first(before));
                holds[j]->position = positions[s] + before;
                holds[j]->history  = histories[s];
                advance_history(s, tokens.subspan(before, kept[j] - before));
                held |= 1U << j;
            } else {
                advance_history(s, tokens.first(kept[j]));
            }
            verified[s].clear();
            positions[s] += kept[j];
        }
        if (spine) {
            // The kept rows' taps (each sequence's from its first row) into its lane.
            for (std::size_t j = 0; j < sequences.size(); ++j) {
                const std::uint32_t s = sequences[j];
                spine->commit(spine->lane(s), positions[s] - kept[j],
                              std::uint32_t(verify_starts[j]), kept[j], stream);
            }
        }
        if (drafter) {
            // The kept rows' taps into the drafter's rings, at the positions and sequences the
            // verification staged.
            workspace->reset();
            const auto row = [&](std::size_t j) { return std::int32_t(verify_starts[j]); };
            for (std::size_t j = 0; j < sequences.size(); ++j) {
                seed(int(kept[j]), row(j), static_cast<const std::int32_t*>(d_positions.p) + row(j),
                     static_cast<const std::int32_t*>(d_sequences.p) + row(j), sequences[j]);
            }
        }
        for (std::size_t j = 0; j < sequences.size(); ++j) {
            if ((held >> j & 1U) != 0) { copy_hold_state(sequences[j], true); }
        }
        verify_widths.clear();
        return held;
    }

    // ---- Branches -------------------------------------------------------------------------------

    // One sequence's rings and tails, (device, bytes) in order: the live state or its checkpoint.
    std::vector<std::pair<std::uint8_t*, std::size_t>> state(std::uint32_t sequence, bool checkpoint) const {
        std::vector<std::pair<std::uint8_t*, std::size_t>> out;
        const std::size_t ring_bytes = kRingBytes;
        const std::size_t tail_bytes = std::size_t(kTailSlots) * kLatent * 4;
        for (const LayerPlan& p : layers) {
            const auto& ring = checkpoint ? p.checkpoint_ring : p.ring;
            out.emplace_back(static_cast<std::uint8_t*>(ring.p) + sequence * ring_bytes, ring_bytes);
            if (p.kv_source && p.ratio > 1) {
                const auto& kv    = checkpoint ? p.checkpoint_kv : p.tail_kv;
                const auto& score = checkpoint ? p.checkpoint_score : p.tail_score;
                out.emplace_back(static_cast<std::uint8_t*>(kv.p) + sequence * tail_bytes, tail_bytes);
                out.emplace_back(static_cast<std::uint8_t*>(score.p) + sequence * tail_bytes, tail_bytes);
            }
        }
        if (drafter) {
            for (const LayerPlan& p : drafter->layers) {
                const auto& ring = checkpoint ? p.checkpoint_ring : p.ring;
                out.emplace_back(static_cast<std::uint8_t*>(ring.p) + sequence * ring_bytes, ring_bytes);
            }
        }
        return out;
    }

    void copy_state(std::uint32_t sequence, bool to_checkpoint, bool from_checkpoint) {
        const auto from = state(sequence, from_checkpoint);
        const auto to   = state(sequence, to_checkpoint);
        for (std::size_t i = 0; i < from.size(); ++i) {
            CUDA_CHECK(cudaMemcpyAsync(to[i].first, from[i].first, from[i].second,
                                       cudaMemcpyDeviceToDevice, stream));
        }
        if (spine) {
            spine->copy_lane(spine->lane(sequence, from_checkpoint ? 1 : 0),
                             spine->lane(sequence, to_checkpoint ? 1 : 0), stream);
        }
    }

    void copy_hold_state(std::uint32_t sequence, bool save) {
        auto* kept = static_cast<std::uint8_t*>(hold_state.p) + sequence * hold_bytes;
        for (const auto& [device_state, bytes] : state(sequence, false)) {
            CUDA_CHECK(cudaMemcpyAsync(save ? kept : device_state, save ? device_state : kept, bytes,
                                       cudaMemcpyDeviceToDevice, stream));
            kept += bytes;
        }
    }

    void copy_branch_state(std::uint32_t sequence, bool save) {
        auto* kept = static_cast<std::uint8_t*>(branch_state.p);
        for (const auto& [device_state, bytes] : state(sequence, false)) {
            CUDA_CHECK(cudaMemcpyAsync(save ? kept : device_state, save ? device_state : kept, bytes,
                                       cudaMemcpyDeviceToDevice, stream));
            kept += bytes;
        }
    }

    void branches(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                  std::span<const std::uint32_t> lengths) {
        if (sequence >= options.sequences || lengths.empty() || lengths.size() > kMaxLogitRows ||
            std::find(lengths.begin(), lengths.end(), 0U) != lengths.end() ||
            std::accumulate(lengths.begin(), lengths.end(), std::uint64_t{0}) != tokens.size()) {
            throw std::invalid_argument("deepseek_v41 forward_branches: invalid branches");
        }
        for (const std::uint32_t length : lengths) {
            if (length > options.prefill_chunk) {
                throw std::invalid_argument("deepseek_v41 forward_branches: a branch longer than a chunk");
            }
        }
        const std::uint32_t position         = positions[sequence];
        const EngramHasher::History history  = histories[sequence];
        const std::uint32_t from             = decoder_from[sequence];
        const std::uint8_t behind            = decoder_behind[sequence];
        copy_branch_state(sequence, true);
        // A branch's rows stay out of the DSpine ring: the sequence keeps none of them.
        in_branches = true;
        struct Kept {
            bool& flag;
            ~Kept() { flag = false; }
        } kept_flag{in_branches};
        std::size_t at = 0;
        for (std::size_t b = 0; b < lengths.size(); ++b) {
            forward(sequence, tokens.subspan(at, lengths[b]), 1, std::uint32_t(b));
            at += lengths[b];
            copy_branch_state(sequence, false);
            positions[sequence]      = position;
            histories[sequence]      = history;
            decoder_from[sequence]   = from;
            decoder_behind[sequence] = behind;
        }
    }

    void warm_up() {
        // The prompt route once (kernels load at first launch), then every decode batch and every
        // verification's row count eagerly, then their graphs. Under lazy module loading a kernel
        // loaded while a graph waits on the device for the experts' host service deadlocks the
        // two, so every route has loaded its kernels before the first graph runs, and each graph
        // has finished before the next call.
        const std::vector<std::int32_t> prompt(std::min<std::uint32_t>(options.prefill_chunk, 16), 2);
        // A DSpine drafter derives its table first, and drafts at every width once.
        if (spine) { spine->prepare(stream); }
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
        reset_state(0);
        const std::uint32_t batches = std::min<std::uint32_t>(options.sequences, 8);
        // A verification's graph is keyed by its rows: one (sequences, width) shape for each.
        std::vector<std::pair<std::uint32_t, std::uint32_t>> rounds;
        for (std::uint32_t rows = 1; options.draft_tokens > 0 && rows <= 8; ++rows) {
            for (std::uint32_t w = std::min(options.draft_tokens + 1, rows); w >= 1; --w) {
                if (rows % w == 0 && rows / w <= batches) {
                    rounds.emplace_back(rows / w, w);
                    break;
                }
            }
        }
        for (int pass = 0; pass < 2; ++pass) { // eagerly, then captured
            std::vector<std::uint32_t> sequences;
            std::vector<std::int32_t> tokens;
            for (std::uint32_t b = 1; b <= batches; ++b) {
                sequences.push_back(b - 1);
                tokens.push_back(2);
                decode(sequences, tokens);
                CUDA_CHECK(ninfer::wait_for_stream(stream));
                for (const std::uint32_t s : sequences) { reset_state(s); }
            }
            for (const auto& [n, w] : rounds) {
                const std::vector<std::uint32_t> owners(sequences.begin(), sequences.begin() + n);
                verify(owners, std::vector<std::int32_t>(std::size_t(n) * w, 2));
                commit(owners, std::vector<std::uint32_t>(n, 1));
                CUDA_CHECK(ninfer::wait_for_stream(stream));
                for (const std::uint32_t s : owners) { reset_state(s); }
            }
            if (drafter) {
                std::vector<std::int32_t> drafts(std::size_t(drafter->block));
                std::vector<float> confidence(drafts.size());
                draft(0, 2, drafts, confidence);
                reset_state(0);
            }
        }
        if (delivery) {
            const auto first =
                profile_slice(expert_ranges.empty() ? 0 : expert_ranges[0].first,
                              expert_ranges.empty() ? config.num_experts : expert_ranges[0].second);
            delivery->drain();
            delivery->prime(first ? &*first : nullptr);
            for (std::size_t r = 1; r < expert_peers.size(); ++r) {
                const auto slice = profile_slice(expert_ranges[r].first, expert_ranges[r].second);
                RankBinding bind(device, r);
                expert_peers[r].delivery->drain();
                expert_peers[r].delivery->prime(slice ? &*slice : nullptr);
            }
            std::lock_guard lock(expert_stats_mutex);
            expert_startup_stats = delivery->stats();
            for (std::size_t r = 1; r < expert_peers.size(); ++r) {
                expert_peers[r].startup = expert_peers[r].delivery->stats();
            }
        }
    }

    void reset_state(std::uint32_t sequence) {
        positions.at(sequence) = 0;
        histories.at(sequence) = {};
        verified.at(sequence).clear();
        reset_decoder(sequence);
    }

    // ---- Snapshots and images ----------------------------------------------------------------------

    [[nodiscard]] std::uint64_t image_bytes(std::uint32_t position) const {
        std::uint64_t bytes = 0;
        for (const auto& [unused, n] : state(0, false)) { bytes += n; }
        for (const LayerPlan& p : layers) {
            if (!p.kv_source) { continue; }
            bytes += std::uint64_t(position / std::uint32_t(p.ratio)) * kCompressedRowBytes;
        }
        return bytes + (spine ? spine->image_bytes(position) : 0);
    }

    // Copies a sequence's image between the device and the host pages: its rings and tails (from
    // or to its checkpoint with `checkpoint`), then the compressed rows below `position`.
    void copy_image(std::uint32_t sequence, std::uint32_t position, std::span<std::byte* const> pages,
                    std::size_t page_bytes, bool save, bool checkpoint) {
        if (page_bytes == 0 || pages.size() < (image_bytes(position) + page_bytes - 1) / page_bytes) {
            throw std::invalid_argument("deepseek_v41 image: pages do not hold the image");
        }
        std::size_t at = 0;
        const auto copy = [&](std::uint8_t* device_bytes, std::size_t bytes) {
            while (bytes > 0) {
                const std::size_t in_page = std::min(bytes, page_bytes - at % page_bytes);
                std::byte* host           = pages[at / page_bytes] + at % page_bytes;
                if (save) {
                    CUDA_CHECK(cudaMemcpyAsync(host, device_bytes, in_page, cudaMemcpyDeviceToHost, stream));
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(device_bytes, host, in_page, cudaMemcpyHostToDevice, stream));
                }
                device_bytes += in_page;
                bytes -= in_page;
                at += in_page;
            }
        };
        for (const auto& [device_state, bytes] : state(sequence, checkpoint)) { copy(device_state, bytes); }
        for (const LayerPlan& p : layers) {
            if (!p.kv_source) { continue; }
            const std::size_t rows = position / std::uint32_t(p.ratio);
            for (const KvArea* area : {&p.latents, &p.keys, &p.key_scales}) {
                copy(static_cast<std::uint8_t*>(area->p) + std::size_t(sequence) * area->row_bytes,
                     rows * area->entry_bytes);
            }
        }
        if (spine) { spine->copy_image(spine->lane(sequence, checkpoint ? 1 : 0), position, copy); }
    }

    SequenceImage save_image(std::uint32_t sequence, const SequenceSnapshot* at,
                             std::span<std::byte* const> pages, std::size_t page_bytes) {
        if (at == nullptr) { require_resumable(sequence, "save_image"); }
        const std::uint32_t position = at != nullptr ? at->position : positions.at(sequence);
        copy_image(sequence, position, pages, page_bytes, true, at != nullptr);
        return {position, at != nullptr ? at->history : histories.at(sequence)};
    }

    void load_image(std::uint32_t sequence, const SequenceImage& image,
                    std::span<std::byte* const> pages, std::size_t page_bytes) {
        if (image.position > options.max_context) {
            throw std::invalid_argument("deepseek_v41 load_image: invalid image");
        }
        back(sequence, image.position);
        copy_image(sequence, image.position, pages, page_bytes, false, false);
        positions.at(sequence) = image.position;
        histories.at(sequence) = image.history;
        verified.at(sequence).clear();
        reset_decoder(sequence);
    }

    void snapshot(std::uint32_t sequence, SequenceSnapshot& out) {
        if (!options.checkpoints) { throw std::logic_error("deepseek_v41: an executor without checkpoints"); }
        require_resumable(sequence, "snapshot");
        copy_state(sequence, true, false);
        out.position = positions[sequence];
        out.history  = histories[sequence];
    }

    void restore(std::uint32_t sequence, const SequenceSnapshot& from) {
        if (!options.checkpoints) { throw std::logic_error("deepseek_v41: an executor without checkpoints"); }
        copy_state(sequence, false, true);
        positions[sequence] = from.position;
        histories[sequence] = from.history;
        verified[sequence].clear();
        reset_decoder(sequence); // a snapshot holds a complete state
    }

    bool hold(std::uint32_t sequence, SequenceSnapshot& out) {
        if (hold_bytes == 0 || !resumable(sequence)) { return false; }
        copy_hold_state(sequence, true);
        out.position = positions[sequence];
        out.history  = histories[sequence];
        return true;
    }

    void restore_hold(std::uint32_t sequence, const SequenceSnapshot& from) {
        if (hold_bytes == 0) { throw std::logic_error("deepseek_v41: an executor without checkpoints"); }
        copy_hold_state(sequence, false);
        positions[sequence] = from.position;
        histories[sequence] = from.history;
        verified[sequence].clear();
        reset_decoder(sequence);
    }
};

Executor::Executor(const Model& model, DeviceContext& device, ExecutorOptions options)
    : impl_(std::make_unique<Impl>(model, device, std::move(options))) {}

Executor::~Executor() = default;

const ExecutorOptions& Executor::options() const noexcept { return impl_->options; }

ExecutorMemory Executor::memory() const noexcept {
    return ExecutorMemory{.kv_bytes        = impl_->kv_bytes,
                          .kv_backed_bytes = impl_->kv_backed.load(std::memory_order_relaxed),
                          .kv_grows        = impl_->growing,
                          .workspace_bytes = impl_->workspace_bytes,
                          .span_borrow_bytes = impl_->span_borrow_bytes,
                          .engram_resident_bytes = impl_->engram ? impl_->engram->resident_bytes() : 0,
                          .vision_encode_bytes = impl_->tower ? impl_->tower->workspace_bytes() : 0,
                          .vision_owned_bytes  = impl_->vision_owned_bytes(),
                          .drafter_expert_bytes = impl_->drafter_expert_bytes,
                          .dspine_bytes = impl_->spine ? impl_->spine->device_bytes() +
                                                             impl_->spine_stage.bytes
                                                       : 0,
                          .experts = impl_->delivery ? impl_->delivery->memory() : ExpertDeliveryMemory{}};
}

std::uint64_t Executor::image_bytes(std::uint32_t position) const { return impl_->image_bytes(position); }

SequenceImage Executor::save_image(std::uint32_t sequence, const SequenceSnapshot* at,
                                   std::span<std::byte* const> pages, std::size_t page_bytes) {
    return impl_->save_image(sequence, at, pages, page_bytes);
}

void Executor::load_image(std::uint32_t sequence, const SequenceImage& image,
                          std::span<std::byte* const> pages, std::size_t page_bytes) {
    impl_->load_image(sequence, image, pages, page_bytes);
}

void Executor::reset(std::uint32_t sequence) {
    impl_->reset_state(sequence);
    impl_->release(sequence);
}

std::uint32_t Executor::position(std::uint32_t sequence) const { return impl_->positions.at(sequence); }

void Executor::warm_up() {
    impl_->warm_up();
    check_experts();
}

void Executor::check_experts() const {
    if (impl_->delivery) { impl_->delivery->check(); }
    for (std::size_t r = 1; r < impl_->expert_peers.size(); ++r) {
        impl_->expert_peers[r].delivery->check();
    }
}
ExpertDeliveryStats Executor::expert_stats() const {
    if (!impl_->delivery) { return {}; }
    std::lock_guard lock(impl_->expert_stats_mutex);
    ExpertDeliveryStats out = impl_->delivery->stats() - impl_->expert_startup_stats;
    for (std::size_t r = 1; r < impl_->expert_peers.size(); ++r) {
        const auto& peer = impl_->expert_peers[r];
        out              = out + (peer.delivery->stats() - peer.startup);
    }
    return out;
}
ExpertRouteCounts Executor::expert_totals() const {
    if (!impl_->delivery) { return {}; }
    ExpertRouteCounts out = impl_->delivery->totals();
    // Expert parallelism: every rank's counts at its experts' places.
    for (std::size_t r = 1; r < impl_->expert_peers.size(); ++r) {
        const auto first  = impl_->expert_ranges[r].first;
        const auto counts = impl_->expert_peers[r].delivery->totals();
        for (std::size_t layer = 0; layer < counts.size(); ++layer) {
            out[layer].resize(
                std::max<std::size_t>(out[layer].size(), first + counts[layer].size()), 0);
            for (std::size_t e = 0; e < counts[layer].size(); ++e) {
                out[layer][first + e] += counts[layer][e];
            }
        }
    }
    return out;
}

std::optional<NgramTableStats> Executor::engram_stats() const {
    if (!impl_->engram) { return std::nullopt; }
    return impl_->engram->counters();
}

std::string Executor::engram_description() const {
    if (!impl_->engram) { return {}; }
    return impl_->engram->io_backend();
}

std::string Executor::expert_cpu_description() const {
    return impl_->delivery ? impl_->delivery->cpu_description() : std::string();
}

std::uint32_t Executor::prompt_step() const noexcept { return impl_->prompt_step(); }

void Executor::forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                       std::uint32_t logit_rows, std::span<const ImageRows> images,
                       std::uint32_t state_at, NextPromptCall next) {
    impl_->forward(sequence, tokens, logit_rows, 0, images, state_at, next);
}

bool Executor::resumable(std::uint32_t sequence) const { return impl_->resumable(sequence); }

std::uint32_t Executor::exact_tail_rows() const noexcept { return impl_->exact_rows; }

std::uint32_t Executor::prefill_tail_rows() const noexcept { return impl_->tail_rows; }

bool Executor::vision() const noexcept { return impl_->tower != nullptr; }

void Executor::encode_image(std::span<const std::uint16_t> patches, int grid_w, int grid_h,
                            std::span<std::uint16_t> rows) {
    impl_->encode_image(patches, grid_w, grid_h, rows);
}

void Executor::forward_branches(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                                std::span<const std::uint32_t> lengths) {
    impl_->branches(sequence, tokens, lengths);
}

void Executor::forward_prompt_branches(std::uint32_t sequence, std::span<const std::int32_t> prompt,
                                       std::span<const std::int32_t> tokens,
                                       std::span<const std::uint32_t> lengths) {
    if (!prompt.empty()) { impl_->forward(sequence, prompt, 0); }
    impl_->branches(sequence, tokens, lengths);
}

void Executor::decode(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens) {
    impl_->decode(sequences, tokens);
}

Tensor Executor::logits(std::uint32_t rows) const {
    if (rows == 0 || rows > kMaxLogitRows) {
        throw std::invalid_argument("deepseek_v41 logits: invalid row count");
    }
    return Tensor(impl_->logits_bf16.p, DType::BF16, {impl_->vocab, static_cast<std::int32_t>(rows)});
}

cudaStream_t Executor::head_stream() const noexcept { return impl_->stream; }

std::uint32_t Executor::draft_tokens() const noexcept { return impl_->options.draft_tokens; }

std::uint32_t Executor::dspark_block() const noexcept {
    return impl_->drafter ? std::uint32_t(impl_->drafter->block) : 0;
}

void Executor::draft(std::uint32_t sequence, std::int32_t anchor, std::span<std::int32_t> drafts,
                     std::span<float> confidence) {
    impl_->draft(sequence, anchor, drafts, confidence);
}

void Executor::draft(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> anchors,
                     std::uint32_t k, std::span<std::int32_t> drafts, std::span<float> confidence) {
    impl_->draft(sequences, anchors, k, drafts, confidence);
}

void Executor::verify(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens,
                      std::span<const std::uint32_t> widths) {
    impl_->verify(sequences, tokens, widths);
}

std::uint32_t Executor::commit(std::span<const std::uint32_t> sequences,
                               std::span<const std::uint32_t> kept,
                               std::span<const std::uint32_t> hold_after,
                               std::span<SequenceSnapshot* const> holds) {
    return impl_->commit(sequences, kept, hold_after, holds);
}

void Executor::snapshot(std::uint32_t sequence, SequenceSnapshot& out) { impl_->snapshot(sequence, out); }

void Executor::restore(std::uint32_t sequence, const SequenceSnapshot& from) {
    impl_->restore(sequence, from);
}

bool Executor::hold(std::uint32_t sequence, SequenceSnapshot& out) { return impl_->hold(sequence, out); }

void Executor::restore_hold(std::uint32_t sequence, const SequenceSnapshot& from) {
    impl_->restore_hold(sequence, from);
}

} // namespace ninfer::models::deepseek_v41
