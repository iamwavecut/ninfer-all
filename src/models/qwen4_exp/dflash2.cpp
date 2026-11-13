#include "models/qwen4_exp/dflash2.h"

#include "core/device_launch.h"
#include "core/nvtx.h"
#include "core/qtype.h"
#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/sliding_window_attention.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen4_exp {
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
    if (!condition) { throw std::invalid_argument("Qwen4Exp DFlash2: " + message); }
}

} // namespace

DFlash2Drafter::DFlash2Drafter(DFlash2Operands operands, const DFlash2Config& config,
                               std::uint32_t hidden, std::uint32_t vocab, std::int32_t domain,
                               std::vector<CyclicKVCacheLayerView> ring, std::uint32_t max_context,
                               DFlash2Head head)
    : p_(std::move(operands)), config_(config), hidden_(dim(hidden)), vocab_(dim(vocab)),
      domain_(domain), ring_(std::move(ring)), max_context_(max_context), head_(std::move(head)) {
    require(ring_.size() == config_.num_hidden_layers && p_.layers.size() == ring_.size(),
            "one context ring layer per drafter layer");
    require(config_.dflash2.has_value() && config_.sliding_window.has_value(),
            "the drafter needs its DFlash2 fields and a sliding window");
    for (const Weight* codebook : {&p_.predecessor_codebook, &p_.successor_codebook}) {
        require(codebook->qtype == QType::GGUF_Q4_0,
                "the selector's codebooks must be stored as GGUF Q4_0 rows");
    }
    require(domain_ >= kCandidates && domain_ <= vocab_ && head_ != nullptr,
            "the head and its vocabulary");
}

void DFlash2Drafter::materialize(const DFlash2ContextCall& call, WorkspaceArena& work,
                                 cudaStream_t stream) const {
    const std::int32_t h = hidden_, w = call.features.ne[1], b = call.features.ne[2];
    const std::int32_t columns = w * b;
    const auto& attention      = config_.attention;
    const std::int32_t d = dim(attention.head_dim), kv = dim(attention.num_key_value_heads);
    require(call.features.dtype == DType::BF16 &&
                call.features.ne[0] == dim(std::uint64_t(taps()) * hidden_) && w > 0 && b > 0 &&
                b <= 8 && call.max_count <= std::uint32_t(w) && call.min_count <= call.max_count,
            "the context call's geometry");
    nvtx::ScopedRange range(nvtx::Name::DFlashContextAppend, nvtx::Category::DFlash,
                            static_cast<std::uint64_t>(columns));
    auto scope       = work.scope();
    Tensor projected = work.alloc(DType::BF16, {h, columns});
    project(p_.feature_projection, call.features.view({call.features.ne[0], columns}), projected,
            work, stream);
    Tensor context = work.alloc(DType::BF16, {h, columns});
    ops::rmsnorm(projected, p_.context_norm, config_.rms_norm_eps, false, context, stream);
    const Tensor flat_positions = call.positions.view({columns});
    for (std::size_t l = 0; l < p_.layers.size(); ++l) {
        auto layer_scope    = work.scope();
        const auto& layer   = p_.layers[l];
        Tensor key_raw      = work.alloc(DType::BF16, {d * kv, columns});
        Tensor value        = work.alloc(DType::BF16, {d, kv, w, b});
        Tensor value_flat   = value.view({d * kv, columns});
        Tensor key          = work.alloc(DType::BF16, {d, kv, w, b});
        project(layer.context_key, context, key_raw, work, stream);
        project(layer.context_value, context, value_flat, work, stream);
        Tensor key_rows = key.view({d, kv * columns});
        ops::rmsnorm(key_raw.view({d, kv * columns}), layer.key_norm, config_.rms_norm_eps, false,
                     key_rows, stream);
        Tensor rotated = key.view({d, kv, columns});
        ops::rope(flat_positions, d, config_.rope_theta, rotated, stream);
        ops::kv_cache_append_prefix(key, value, call.positions, call.counts, call.lanes,
                                    {call.min_count, call.max_count}, ring_[l], stream);
    }
}

void DFlash2Drafter::propose(const DFlash2ProposalCall& call, WorkspaceArena& work,
                             cudaStream_t stream) const {
    const std::int32_t h = hidden_, w = call.width, b = call.batch, k = w - 1;
    const std::int32_t columns = w * b, masks = k * b;
    const auto& attention      = config_.attention;
    const std::int32_t d = dim(attention.head_dim), hq = dim(attention.num_attention_heads),
                       kv = dim(attention.num_key_value_heads);
    const std::int32_t inner  = dim(config_.intermediate_size);
    const std::int32_t groups = h / dim(config_.dflash2->conv_group_size);
    const std::int32_t rank   = dim(config_.dflash2->selector_rank);
    const float eps           = config_.rms_norm_eps;
    require(w >= 2 && w <= 16 && b >= 1 && b <= 8 && call.configs != nullptr,
            "the proposal's geometry");
    nvtx::ScopedRange range(nvtx::Name::DFlashProposal, nvtx::Category::DFlash,
                            static_cast<std::uint64_t>(columns));
    auto scope       = work.scope();
    Tensor ids       = call.ids;
    Tensor positions = call.positions;
    ops::prepare_masked_block(call.anchors, call.frontiers, call.valid_columns,
                              dim(config_.mask_token_id), ids, positions, stream);
    const Tensor flat_positions = positions.view({columns});
    Tensor residual             = work.alloc(DType::BF16, {h, w, b});
    {
        Tensor flat = residual.view({h, columns});
        ops::embedding(ids.view({columns}), p_.embedding, flat, stream);
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
            Tensor k      = work.alloc(DType::BF16, {d, kv, w, b});
            Tensor q_rows = q.view({d, hq * columns});
            Tensor k_rows = k.view({d, kv * columns});
            ops::rmsnorm(q_raw.view({d, hq * columns}), layer.query_norm, eps, false, q_rows,
                         stream);
            ops::rmsnorm(k_raw.view({d, kv * columns}), layer.key_norm, eps, false, k_rows,
                         stream);
            Tensor q_heads = q.view({d, hq, columns});
            Tensor k_heads = k.view({d, kv, columns});
            ops::rope(flat_positions, d, config_.rope_theta, q_heads, k_heads, stream);
            Tensor attended = work.alloc(DType::BF16, {d, hq, w, b});
            ops::sliding_window_attention(q, k, v, positions, call.valid_columns, call.lanes,
                                          {d, hq, kv}, *config_.sliding_window,
                                          ops::SlidingWindowMask::Symmetric, scale, ring_[l],
                                          envelope, work, attended, stream);
            Tensor output = work.alloc(DType::BF16, {h, w, b});
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
            Tensor down     = work.alloc(DType::BF16, {h, w, b});
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
    Tensor hidden = work.alloc(DType::BF16, {h, masks});
    CUDA_CHECK(cudaMemcpy2DAsync(hidden.data, std::size_t(k) * h * 2,
                                 static_cast<const std::byte*>(normed.data) + std::size_t(h) * 2,
                                 std::size_t(w) * h * 2, std::size_t(k) * h * 2, std::size_t(b),
                                 cudaMemcpyDeviceToDevice, stream));
    Tensor scores = work.alloc(DType::FP32, {kCandidates, k, b});
    {
        auto head_scope = work.scope();
        Tensor logits   = work.alloc(DType::BF16, {vocab_, masks});
        head_(hidden, logits, work, stream);
        Tensor candidates = call.candidates;
        ops::candidate_topk(logits, domain_, candidates, scores, work, stream);
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
                                 call.configs, drafts, proposal_q, work, stream, &confidence);
}

std::size_t DFlash2Drafter::workspace_bytes(const DFlash2Operands& p, const DFlash2Config& config,
                                            std::uint32_t hidden, std::uint32_t vocab,
                                            std::int32_t max_batch, std::int32_t max_width,
                                            std::int32_t context_columns,
                                            std::uint32_t max_context, std::size_t head_bytes) {
    const std::uint64_t h = hidden, d = config.attention.head_dim;
    const std::uint64_t q = config.attention.query_width(), kv = config.attention.key_width();
    const std::uint64_t inner  = config.intermediate_size;
    const std::uint64_t groups = h / config.dflash2->conv_group_size;
    const std::uint64_t rank   = config.dflash2->selector_rank;
    // materialize: the projected and normalized context, then one layer's K/V at a time.
    const auto c       = static_cast<std::uint64_t>(std::max(context_columns, 1));
    std::size_t linear = linear_bytes(p.feature_projection, dim(c));
    for (const auto& layer : p.layers) {
        linear = std::max({linear, linear_bytes(layer.context_key, dim(c)),
                           linear_bytes(layer.context_value, dim(c))});
    }
    const std::size_t context = 2 * plane(h, c, 2) + 3 * plane(kv, c, 2) + linear;
    // propose at the widest block: the residual, a branch's planes and scratch, then the tail.
    const std::uint64_t cells = std::uint64_t(max_width) * max_batch;
    const std::uint64_t masks = std::uint64_t(max_width - 1) * max_batch;
    std::size_t projections   = 0;
    for (const auto& layer : p.layers) {
        for (const auto* w : {&layer.query, &layer.key, &layer.value, &layer.output, &layer.gate,
                              &layer.up, &layer.down, &layer.attention_projection,
                              &layer.mlp_projection}) {
            projections = std::max(projections, linear_bytes(*w, dim(cells)));
        }
    }
    std::size_t window = 0;
    for (std::int32_t width = 2; width <= max_width; ++width) {
        for (std::int32_t batch = 1; batch <= max_batch; ++batch) {
            window = std::max(window, ops::sliding_window_attention_workspace_capacity_bytes(
                                          {dim(d), dim(config.attention.num_attention_heads),
                                           dim(config.attention.num_key_value_heads)},
                                          *config.sliding_window, {0, max_context}, width, width,
                                          batch));
        }
    }
    const std::size_t branch = plane(4 * groups, cells, 4) + plane(h, cells, 2);
    const std::size_t attention =
        branch + std::max(plane(h, cells, 2) + projections,
                          plane(q, cells, 2) + 2 * plane(kv, cells, 2) + plane(q, cells, 2) +
                              plane(kv, cells, 2) + std::max({projections, window}) +
                              plane(q, cells, 2) + plane(h, cells, 2));
    const std::size_t mlp = branch + std::max(plane(h, cells, 2) + projections,
                                              2 * plane(inner, cells, 4) + plane(inner, cells, 2) +
                                                  plane(h, cells, 2) + projections);
    std::size_t selector = 0;
    for (std::int32_t width = 2; width <= max_width; ++width) {
        selector = std::max(selector, ops::candidate_selector_path_workspace_capacity_bytes(
                                          width - 1, width - 1, 1, max_batch));
    }
    const std::size_t tail =
        plane(h, cells, 2) + plane(h, masks, 2) + plane(kCandidates, masks, 4) +
        std::max(plane(vocab, masks, 2) +
                     std::max(head_bytes, ops::candidate_topk_workspace_capacity_bytes(dim(masks))),
                 plane(rank, masks, 2) +
                     std::max(linear_bytes(p.selector_projection, dim(masks)), selector));
    const std::size_t propose = plane(h, cells, 2) + std::max({attention, mlp, tail});
    return std::max(context, propose) + (1U << 20);
}

} // namespace ninfer::models::qwen4_exp
