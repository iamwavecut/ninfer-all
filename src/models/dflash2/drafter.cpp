#include "models/dflash2/drafter.h"

#include "core/device_launch.h"
#include "core/nvtx.h"
#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/sliding_window_attention.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::dflash2 {
namespace {

constexpr std::int32_t kCandidates = 16;
constexpr std::size_t kSlack       = 256; // per arena allocation

std::int32_t dim(std::uint64_t value) { return static_cast<std::int32_t>(value); }

std::size_t plane(std::uint64_t rows, std::uint64_t columns, std::size_t element) {
    return rows * columns * element + kSlack;
}

void project(const ops::SingleProjectionWeight& w, const Tensor& x, Tensor& out,
             WorkspaceArena& work, cudaStream_t stream) {
    ops::linear(x, w.weight, out, w.policy, work, stream);
}

std::size_t linear_bytes(const ops::SingleProjectionWeight& w, std::int32_t columns) {
    return ops::linear_workspace_capacity_bytes(w.weight.qtype, w.weight.n, w.weight.k, w.policy,
                                                1, std::max(columns, 1));
}

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::invalid_argument("DFlash2 drafter: " + message); }
}

} // namespace

Drafter::Drafter(const Config& config, Operands operands, Target target,
                 std::uint32_t max_context, std::int32_t max_batch, std::int32_t max_width)
    : config_(config), p_(std::move(operands)), target_(std::move(target)),
      max_context_(max_context), max_batch_(max_batch), max_width_(max_width) {
    const Config& c = config_;
    require(target_.hidden == c.hidden_size && target_.vocab == c.vocab_size,
            "the target's width and token rows must be the drafter's");
    require(target_.domain >= kCandidates && std::uint32_t(target_.domain) <= c.vocab_size,
            "the candidates' vocabulary must hold 16 ids and lie within the codebooks' rows");
    require(target_.embed != nullptr && target_.head != nullptr, "the target's rows and head");
    require(target_.logits == DType::BF16 || target_.logits == DType::FP32,
            "the head writes BF16 or FP32 logits");
    require(p_.layers.size() == c.num_hidden_layers, "one operand set per layer");
    for (const Tensor* codebook : {&p_.predecessor_codebook, &p_.successor_codebook}) {
        require(codebook->dtype == DType::BF16 && codebook->ne[0] == dim(c.selector_rank) &&
                    codebook->ne[1] == dim(c.vocab_size),
                "the selector's codebooks must be BF16 [rank, vocab]");
    }
    require(max_batch_ >= 1 && max_batch_ <= 8, "at most eight rows a call");
    require(max_width_ >= 2 && std::uint32_t(max_width_) <= c.block_size,
            "a block of 2..block_size columns");
}

void Drafter::materialize(const ContextCall& call, std::span<const CyclicKVCacheLayerView> ring,
                          WorkspaceArena& work, cudaStream_t stream) const {
    const Config& c = config_;
    const std::int32_t h = dim(c.hidden_size), w = call.features.ne[1], b = call.features.ne[2];
    const std::int32_t d = dim(c.head_dim), kv = dim(c.num_key_value_heads);
    const std::uint32_t window = c.sliding_window;
    require(call.features.dtype == DType::BF16 &&
                call.features.ne[0] == dim(std::uint64_t(c.taps()) * c.hidden_size) && w > 0 &&
                b > 0 && b <= 8 && call.positions.dtype == DType::I32 &&
                call.positions.ne[0] == w && call.positions.ne[1] == b &&
                call.counts.dtype == DType::I32 && call.counts.ne[0] == b &&
                call.lanes.dtype == DType::I32 && call.lanes.ne[0] == b &&
                call.max_count <= std::uint32_t(w) && call.min_count <= call.max_count,
            "the context call's geometry");
    require(ring.size() == p_.layers.size(), "one ring layer per drafter layer");
    for (const auto& layer : ring) { require(layer.capacity == window, "the ring's capacity"); }
    nvtx::ScopedRange range(nvtx::Name::DFlashContextAppend, nvtx::Category::DFlash,
                            static_cast<std::uint64_t>(w) * b);
    auto scope = work.scope();
    // A single row's columns past max_count are inert. A single row of known count wider than the
    // ring stores its last window of columns, which every later block sees in full.
    std::int32_t first = 0, width = b == 1 ? std::min(w, dim(call.max_count)) : w;
    if (width == 0) { return; }
    Tensor counts = call.counts;
    ops::KVCacheAppendPrefixExecutionEnvelope envelope{call.min_count, call.max_count};
    if (call.max_count > window) {
        require(b == 1 && call.min_count == call.max_count,
                "a commit wider than the ring must be one row of known count");
        first    = width - dim(window);
        width    = dim(window);
        counts   = work.alloc(DType::I32, {1});
        ops::set_i32_scalar(counts, width, stream);
        envelope = {window, window};
    }
    const std::int32_t columns = width * b;
    const Tensor features  = width == w ? call.features : call.features.slice(1, first, width);
    const Tensor positions = width == w ? call.positions : call.positions.slice(0, first, width);
    Tensor projected       = work.alloc(DType::BF16, {h, columns});
    project(p_.feature_projection, features.view({features.ne[0], columns}), projected, work,
            stream);
    Tensor context = work.alloc(DType::BF16, {h, columns});
    ops::rmsnorm(projected, p_.context_norm, c.rms_norm_eps, false, context, stream);
    const Tensor flat_positions = positions.view({columns});
    for (std::size_t l = 0; l < p_.layers.size(); ++l) {
        auto layer_scope  = work.scope();
        const auto& layer = p_.layers[l];
        Tensor key_raw    = work.alloc(DType::BF16, {d * kv, columns});
        Tensor value      = work.alloc(DType::BF16, {d, kv, width, b});
        Tensor value_flat = value.view({d * kv, columns});
        Tensor key        = work.alloc(DType::BF16, {d, kv, width, b});
        project(layer.key, context, key_raw, work, stream);
        project(layer.value, context, value_flat, work, stream);
        Tensor key_rows = key.view({d, kv * columns});
        ops::rmsnorm(key_raw.view({d, kv * columns}), layer.key_norm, c.rms_norm_eps, false,
                     key_rows, stream);
        Tensor rotated = key.view({d, kv, columns});
        ops::rope(flat_positions, d, c.rope_theta, rotated, stream);
        ops::kv_cache_append_prefix(key, value, positions.view({width, b}), counts, call.lanes,
                                    envelope, ring[l], stream);
    }
}

void Drafter::propose(const ProposalCall& call, std::span<const CyclicKVCacheLayerView> ring,
                      WorkspaceArena& work, cudaStream_t stream) const {
    const Config& c = config_;
    const std::int32_t h = dim(c.hidden_size), w = call.width, b = call.batch, k = w - 1;
    const std::int32_t columns = w * b, masks = k * b;
    const std::int32_t d = dim(c.head_dim), hq = dim(c.num_attention_heads),
                       kv = dim(c.num_key_value_heads);
    const std::int32_t inner  = dim(c.intermediate_size);
    const std::int32_t groups = dim(c.conv_groups());
    const std::int32_t rank   = dim(c.selector_rank);
    const float eps           = c.rms_norm_eps;
    require(w >= 2 && w <= max_width_ && b >= 1 && b <= max_batch_ && call.configs != nullptr,
            "the proposal's geometry");
    require(ring.size() == p_.layers.size(), "one ring layer per drafter layer");
    for (const auto& layer : ring) {
        require(layer.capacity == c.sliding_window, "the ring's capacity");
    }
    nvtx::ScopedRange range(nvtx::Name::DFlashProposal, nvtx::Category::DFlash,
                            static_cast<std::uint64_t>(columns));
    auto scope       = work.scope();
    Tensor ids       = work.alloc(DType::I32, {w, b});
    Tensor positions = work.alloc(DType::I32, {w, b});
    ops::prepare_masked_block(call.anchors, call.frontiers, call.valid_columns,
                              dim(c.mask_token_id), ids, positions, stream);
    const Tensor flat_positions = positions.view({columns});
    Tensor residual             = work.alloc(DType::BF16, {h, w, b});
    {
        Tensor flat = residual.view({h, columns});
        target_.embed(ids.view({columns}), flat, stream);
    }
    const ops::SlidingWindowAttentionExecutionEnvelope envelope{0, max_context_};
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(d)));
    // A branch's normalized input and its convolution's coefficients (both sides), then side 0:
    // the prepared input of the branch's projections.
    const auto prepare = [&](const Tensor& norm, const ops::SingleProjectionWeight& projection,
                             const Tensor& base, Tensor& coefficients, Tensor& prepared) {
        auto inner_scope   = work.scope();
        Tensor normed      = work.alloc(DType::BF16, {h, w, b});
        Tensor normed_flat = normed.view({h, columns});
        ops::rmsnorm(residual.view({h, columns}), norm, eps, false, normed_flat, stream);
        Tensor flat = coefficients.view({4 * groups, columns});
        project(projection, normed_flat, flat, work, stream);
        ops::dynamic_grouped_conv(normed, coefficients, base, 0, prepared, stream);
    };
    for (std::size_t l = 0; l < p_.layers.size(); ++l) {
        const auto& layer = p_.layers[l];
        nvtx::ScopedRange layer_range(nvtx::Name::DFlashLayer, nvtx::Category::DFlash, l);
        {
            auto branch         = work.scope();
            Tensor coefficients = work.alloc(DType::FP32, {4 * groups, w, b});
            Tensor prepared     = work.alloc(DType::BF16, {h, w, b});
            prepare(layer.input_norm, layer.attention_projection, layer.attention_base,
                    coefficients, prepared);
            const Tensor x = prepared.view({h, columns});
            Tensor q_raw   = work.alloc(DType::BF16, {d * hq, columns});
            Tensor k_raw   = work.alloc(DType::BF16, {d * kv, columns});
            Tensor v       = work.alloc(DType::BF16, {d, kv, w, b});
            Tensor v_flat  = v.view({d * kv, columns});
            project(layer.query, x, q_raw, work, stream);
            project(layer.key, x, k_raw, work, stream);
            project(layer.value, x, v_flat, work, stream);
            Tensor q      = work.alloc(DType::BF16, {d, hq, w, b});
            Tensor key    = work.alloc(DType::BF16, {d, kv, w, b});
            Tensor q_rows = q.view({d, hq * columns});
            Tensor k_rows = key.view({d, kv * columns});
            ops::rmsnorm(q_raw.view({d, hq * columns}), layer.query_norm, eps, false, q_rows,
                         stream);
            ops::rmsnorm(k_raw.view({d, kv * columns}), layer.key_norm, eps, false, k_rows,
                         stream);
            Tensor q_heads = q.view({d, hq, columns});
            Tensor k_heads = key.view({d, kv, columns});
            ops::rope(flat_positions, d, c.rope_theta, q_heads, k_heads, stream);
            Tensor attended = work.alloc(DType::BF16, {d, hq, w, b});
            ops::sliding_window_attention(q, key, v, positions, call.valid_columns, call.lanes,
                                          {d, hq, kv}, c.sliding_window,
                                          ops::SlidingWindowMask::Symmetric, scale, ring[l],
                                          envelope, work, attended, stream);
            Tensor output   = work.alloc(DType::BF16, {h, w, b});
            Tensor out_flat = output.view({h, columns});
            project(layer.output, attended.view({d * hq, columns}), out_flat, work, stream);
            ops::dynamic_grouped_conv(output, coefficients, layer.attention_base, 1, residual,
                                      stream);
        }
        {
            auto branch         = work.scope();
            Tensor coefficients = work.alloc(DType::FP32, {4 * groups, w, b});
            Tensor prepared     = work.alloc(DType::BF16, {h, w, b});
            prepare(layer.post_attention_norm, layer.mlp_projection, layer.mlp_base, coefficients,
                    prepared);
            const Tensor x = prepared.view({h, columns});
            Tensor gate    = work.alloc(DType::FP32, {inner, columns});
            Tensor up      = work.alloc(DType::FP32, {inner, columns});
            project(layer.gate, x, gate, work, stream);
            project(layer.up, x, up, work, stream);
            Tensor product = work.alloc(DType::BF16, {inner, columns});
            ops::silu_mul(gate, up, product, stream);
            Tensor down      = work.alloc(DType::BF16, {h, w, b});
            Tensor down_flat = down.view({h, columns});
            project(layer.down, product, down_flat, work, stream);
            ops::dynamic_grouped_conv(down, coefficients, layer.mlp_base, 1, residual, stream);
        }
    }
    // The mask columns' final hidden, packed row after row: their head candidates and the
    // selector's conditional path.
    Tensor normed = work.alloc(DType::BF16, {h, w, b});
    {
        Tensor flat = normed.view({h, columns});
        ops::rmsnorm(residual.view({h, columns}), p_.final_norm, eps, false, flat, stream);
    }
    Tensor hidden         = work.alloc(DType::BF16, {h, masks});
    const std::size_t row = std::size_t(h) * dtype_size(DType::BF16);
    CUDA_CHECK(cudaMemcpy2DAsync(hidden.data, std::size_t(k) * row,
                                 static_cast<const std::byte*>(normed.data) + row,
                                 std::size_t(w) * row, std::size_t(k) * row, std::size_t(b),
                                 cudaMemcpyDeviceToDevice, stream));
    Tensor scores = work.alloc(DType::FP32, {kCandidates, k, b});
    {
        auto head_scope   = work.scope();
        Tensor logits     = work.alloc(target_.logits, {dim(target_.vocab), masks});
        target_.head(hidden, logits, work, stream);
        Tensor candidates = call.candidates;
        ops::candidate_topk(logits, target_.domain, candidates, scores, work, stream);
    }
    Tensor projected = work.alloc(DType::BF16, {rank, k, b});
    {
        Tensor flat = projected.view({rank, masks});
        project(p_.selector_projection, hidden, flat, work, stream);
    }
    Tensor drafts     = call.drafts;
    Tensor proposal_q = call.proposal_q;
    Tensor confidence = call.confidence;
    ops::candidate_selector_path(call.candidates, scores, projected, call.anchors,
                                 p_.predecessor_codebook, p_.successor_codebook, call.frontiers,
                                 call.configs, drafts, proposal_q, work, stream,
                                 confidence.data != nullptr ? &confidence : nullptr,
                                 c.selector_softcap);
}

std::size_t Drafter::context_workspace_bytes(const Config& c, const Operands& p,
                                             std::int32_t width, std::int32_t batch) {
    // A single row wider than the ring keeps only its last window of columns.
    const std::uint64_t columns =
        batch == 1 ? std::min<std::uint64_t>(std::uint64_t(std::max(width, 1)), c.sliding_window)
                   : std::uint64_t(std::max(width, 1)) * std::uint64_t(batch);
    std::size_t layer_linear = 0;
    for (const auto& layer : p.layers) {
        layer_linear = std::max({layer_linear, linear_bytes(layer.key, dim(columns)),
                                 linear_bytes(layer.value, dim(columns))});
    }
    const std::size_t layer = 3 * plane(c.key_width(), columns, 2) + layer_linear;
    return plane(1, 1, 4) + 2 * plane(c.hidden_size, columns, 2) +
           std::max(linear_bytes(p.feature_projection, dim(columns)), layer);
}

std::size_t Drafter::proposal_workspace_bytes(const Config& c, const Operands& p, DType logits,
                                              std::uint32_t max_context, std::int32_t width,
                                              std::int32_t batch, std::size_t head_bytes) {
    const std::uint64_t h = c.hidden_size, q = c.query_width(), kv = c.key_width();
    const std::uint64_t inner  = c.intermediate_size;
    const std::uint64_t groups = c.conv_groups();
    const std::uint64_t rank   = c.selector_rank;
    const std::uint64_t cells  = std::uint64_t(width) * batch;
    const std::uint64_t masks  = std::uint64_t(width - 1) * batch;
    std::size_t projections    = 0;
    for (const auto& layer : p.layers) {
        for (const auto* w : {&layer.query, &layer.key, &layer.value, &layer.output, &layer.gate,
                              &layer.up, &layer.down, &layer.attention_projection,
                              &layer.mlp_projection}) {
            projections = std::max(projections, linear_bytes(*w, dim(cells)));
        }
    }
    const std::size_t window = ops::sliding_window_attention_workspace_capacity_bytes(
        {dim(c.head_dim), dim(c.num_attention_heads), dim(c.num_key_value_heads)},
        c.sliding_window, {0, max_context}, 1, width, batch);
    const std::size_t base   = 2 * plane(1, cells, 4) + plane(h, cells, 2);
    const std::size_t branch = plane(4 * groups, cells, 4) + plane(h, cells, 2);
    const std::size_t prepare = plane(h, cells, 2) + projections;
    const std::size_t attention =
        branch + std::max(prepare, 2 * plane(q, cells, 2) + 4 * plane(kv, cells, 2) +
                                       std::max({projections, window}) + plane(q, cells, 2) +
                                       plane(h, cells, 2) + projections);
    const std::size_t mlp = branch + std::max(prepare, 2 * plane(inner, cells, 4) +
                                                           plane(inner, cells, 2) +
                                                           plane(h, cells, 2) + projections);
    std::size_t selector = 0;
    for (std::int32_t steps = 1; steps < width; ++steps) {
        selector = std::max(selector, ops::candidate_selector_path_workspace_capacity_bytes(
                                          steps, steps, 1, batch));
    }
    const std::size_t tail =
        plane(h, cells, 2) + plane(h, masks, 2) + plane(kCandidates, masks, 4) +
        std::max(plane(c.vocab_size, masks, dtype_size(logits)) +
                     std::max(head_bytes, ops::candidate_topk_workspace_capacity_bytes(dim(masks))),
                 plane(rank, masks, 2) +
                     std::max(linear_bytes(p.selector_projection, dim(masks)), selector));
    return base + std::max({attention, mlp, tail}) + kSlack;
}

} // namespace ninfer::models::dflash2
