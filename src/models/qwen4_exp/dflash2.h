#pragma once

// The DFlash2 drafter of Qwen3.8-Flash-Next (docs/maintainer/dflash.md), from a llama.cpp GGUF with
// its ggml blocks kept as stored: one masked block of K + 1 columns (the anchor and K masks)
// through sliding-window layers with a dynamic grouped convolution around each branch, conditioned
// on a per-sequence ring of context K/V projected from the target's features (the mean of the four
// hyper-connection streams after each tapped block), then the target's head, the dense logits'
// top 16 and the conditional candidate selector. Every projection runs through Linear in its
// stored format, so this route is the reference shape of the fused Q8 Ops. The executor owns the
// ring, the features and every input and output word; this class holds operands only.

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/qwen4_exp/config.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/weight_input.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace ninfer::models::qwen4_exp {

struct DFlash2LayerOperands {
    Tensor input_norm, post_attention_norm, query_norm, key_norm; // BF16
    Tensor attention_base, mlp_base;                              // BF16 [H, 2, 2]
    ops::SingleProjectionWeight query, key, value, context_key, context_value, output;
    ops::SingleProjectionWeight gate, up, down, attention_projection, mlp_projection;
};

struct DFlash2Operands {
    Tensor context_norm, final_norm; // BF16 [H]
    ops::SingleProjectionWeight feature_projection, selector_projection;
    Weight predecessor_codebook, successor_codebook; // GGUF Q4_0 [vocab, 256]
    Weight embedding;                                // the text model's token rows
    std::vector<DFlash2LayerOperands> layers;
};

// The target's head over `hidden` (BF16 [H, U]) into `logits` (BF16 [vocab, U]).
using DFlash2Head =
    std::function<void(const Tensor& hidden, Tensor& logits, WorkspaceArena&, cudaStream_t)>;

// Committed target features into the ring: row b's first counts[b] columns (positions[i, b], a
// sequential run) are stored in lane lanes[b]; its other columns are inert.
struct DFlash2ContextCall {
    Tensor features;  // BF16 [taps * H, W, B]
    Tensor positions; // I32 [W, B]
    Tensor counts;    // I32 [B]
    Tensor lanes;     // I32 [B]
    // The host's promise on counts (min <= counts[b] <= max <= W).
    std::uint32_t min_count = 0, max_count = 0;
};

// One proposal per row: the block [anchor, mask x K] at positions frontier .. frontier + K
// against the ring's committed context [frontier - window, frontier).
struct DFlash2ProposalCall {
    Tensor anchors;       // I32 [B]: each row's anchor token
    Tensor frontiers;     // I32 [B]: the anchor's position
    Tensor valid_columns; // I32 [B]: live block columns 1..W
    Tensor lanes;         // I32 [B]: ring lanes
    const ops::SamplingConfig* configs = nullptr; // device [B]
    std::int32_t batch = 0, width = 0;             // W = K + 1 in [2, 16]
    Tensor ids, positions;                         // I32 [W, B] scratch
    Tensor drafts;                                 // I32 [K, B]
    Tensor candidates;                             // I32 [16, K, B]
    Tensor proposal_q;                             // FP32 [16, K, B]
    Tensor confidence;                             // FP32 [K, B]
};

class DFlash2Drafter {
public:
    // `ring` holds one layer view per drafter layer (capacity the window); `domain` bounds the
    // candidates (the public vocabulary); max_context bounds every frontier.
    DFlash2Drafter(DFlash2Operands operands, const DFlash2Config& config, std::uint32_t hidden,
                   std::uint32_t vocab, std::int32_t domain,
                   std::vector<CyclicKVCacheLayerView> ring, std::uint32_t max_context,
                   DFlash2Head head);

    void materialize(const DFlash2ContextCall& call, WorkspaceArena& work,
                     cudaStream_t stream) const;
    void propose(const DFlash2ProposalCall& call, WorkspaceArena& work, cudaStream_t stream) const;

    // Workspace of materialize over up to `context_columns` columns in rows of up to max_batch and
    // of propose at widths [2, max_width] over up to max_batch rows, beside `head_bytes` (the
    // head's own workspace at (max_width - 1) * max_batch columns).
    [[nodiscard]] static std::size_t workspace_bytes(const DFlash2Operands& operands,
                                                     const DFlash2Config& config,
                                                     std::uint32_t hidden, std::uint32_t vocab,
                                                     std::int32_t max_batch, std::int32_t max_width,
                                                     std::int32_t context_columns,
                                                     std::uint32_t max_context,
                                                     std::size_t head_bytes);

    [[nodiscard]] std::uint32_t taps() const noexcept {
        return static_cast<std::uint32_t>(config_.target_layer_ids.size());
    }

private:
    DFlash2Operands p_;
    const DFlash2Config& config_;
    std::int32_t hidden_ = 0, vocab_ = 0, domain_ = 0;
    std::vector<CyclicKVCacheLayerView> ring_;
    std::uint32_t max_context_ = 0;
    DFlash2Head head_;
};

} // namespace ninfer::models::qwen4_exp
