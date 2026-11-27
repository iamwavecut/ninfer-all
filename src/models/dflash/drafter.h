#pragma once

// A DFlash or DSpark drafter of z-lab's DFlash layout beside a family that keeps its own sequences
// (docs/maintainer/dflash.md#gemma-4), composed from the runtime-shape Ops: one masked block of
// K + 1 columns (the anchor and K masks) at a sequence's next positions through Qwen 3 layers,
// non-causal inside the block, against per-sequence rings of context K/V projected from the
// target's tapped hidden states (each layer's own key and value matrices over the normalized
// feature projection); a sliding-window layer reads a ring of its window, a full-attention layer
// one of the full window (dflash::full_window). Then the target's head under its final soft cap
// over the block's final hidden, the 16 best candidates of each draft row, and either each row's
// best candidate (DFlash) or the chain a vanilla Markov head adds its transition bias to (DSpark,
// ops::candidate_selector_path with unit hidden projections).
//
// A layer of 64 query heads runs the attention Op twice, over 32 heads each: call g takes heads
// 8 j + 4 g .. 8 j + 4 g + 3 of KV head j, so the Op's group of four reads the KV head the
// drafter's group of eight does.
//
// The family supplies the operands from its artifact's `dflash` or `dspark` component, its token
// rows, its head and the rings (one layer view per drafter layer, of layer_capacity());
// models/dspine/runtime.h holds the rings, the features and the calls' operands. Every call
// enqueues on the given stream and allocates only from the given workspace, so a CUDA graph may
// capture it.

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/tensor.h"
#include "models/dflash/config.h"
#include "models/dspine/drafter.h"
#include "ninfer/ops/weight_input.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::dflash {

struct LayerOperands {
    Tensor input_norm, post_attention_norm; // BF16 [H]
    Tensor query_norm, key_norm;            // BF16 [D]
    // The block's Q, K, V and output; the context's K and V go through the same key and value.
    ops::SingleProjectionWeight query, key, value, output;
    ops::SingleProjectionWeight gate, up, down;
};

struct Operands {
    Tensor context_norm, final_norm;                // BF16 [H]
    ops::SingleProjectionWeight feature_projection; // [H, taps * H]
    // DSpark's Markov head: BF16 [256, vocab] each, the predecessor's row and the candidate's.
    Tensor markov_embedding, markov_output;
    std::vector<LayerOperands> layers;
};

using Target       = dspine::Target;
using ContextCall  = dspine::ContextCall;
using ProposalCall = dspine::ProposalCall;

// Layer `layer`'s ring capacity: the sliding window, or the full window of a full-attention layer.
[[nodiscard]] std::uint32_t layer_capacity(const Config& config, std::uint32_t layer,
                                           std::uint32_t full_window) noexcept;

class Drafter {
public:
    // full_window is dflash::full_window(config, max_context, max_width); max_context bounds every
    // frontier, max_batch every proposal's rows (a commit takes up to eight) and max_width
    // (2..block_size) every block.
    Drafter(const Config& config, Operands operands, Target target, std::uint32_t full_window,
            std::uint32_t max_context, std::int32_t max_batch, std::int32_t max_width);

    void materialize(const ContextCall& call, std::span<const CyclicKVCacheLayerView> ring,
                     WorkspaceArena& work, cudaStream_t stream) const;
    void propose(const ProposalCall& call, std::span<const CyclicKVCacheLayerView> ring,
                 WorkspaceArena& work, cudaStream_t stream) const;

    // Workspace of materialize() over calls of up to `width` columns in `batch` rows.
    [[nodiscard]] static std::size_t context_workspace_bytes(const Config& config,
                                                             const Operands& operands,
                                                             std::uint32_t full_window,
                                                             std::int32_t width,
                                                             std::int32_t batch);
    // Workspace of propose() at widths 2..`width` over `batch` rows beside `head_bytes`, the
    // head's own workspace at up to (width - 1) * batch columns.
    [[nodiscard]] static std::size_t proposal_workspace_bytes(
        const Config& config, const Operands& operands, DType logits, std::uint32_t full_window,
        std::uint32_t max_context, std::int32_t width, std::int32_t batch, std::size_t head_bytes);

    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] std::uint32_t full_window() const noexcept { return full_window_; }

private:
    const Config& config_;
    Operands p_;
    Target target_;
    std::uint32_t full_window_ = 0, max_context_ = 0;
    std::int32_t max_batch_ = 0, max_width_ = 0;
};

} // namespace ninfer::models::dflash
