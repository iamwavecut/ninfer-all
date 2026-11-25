#pragma once

// A DFlash2 drafter beside a family that keeps its own sequences (docs/maintainer/dflash.md),
// composed from the runtime-shape Ops: one masked block of K + 1 columns (the anchor and K masks)
// at a sequence's next positions through sliding-window layers, non-causal inside the block, with
// a dynamic grouped convolution around each branch, against a per-sequence ring of context K/V
// projected from the target's tapped hidden states (the drafter's own key and value matrices over
// the normalized feature projection); then the target's head over the block's final hidden, the
// 16 best candidates of each draft row and the conditional candidate selector that chains them.
//
// The family supplies the operands from its artifact's `dflash2` component, its token rows, its
// head and the ring (one layer view per drafter layer); models/dspine/runtime.h holds the ring,
// the features and the calls' operands. Every call enqueues on the given stream and allocates only
// from the given workspace, so a CUDA graph may capture it.

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/tensor.h"
#include "models/dflash2/config.h"
#include "models/dspine/drafter.h"
#include "ninfer/ops/weight_input.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::dflash2 {

struct LayerOperands {
    Tensor input_norm, post_attention_norm; // BF16 [H]
    Tensor query_norm, key_norm;            // BF16 [D]
    Tensor attention_base, mlp_base;        // BF16 [H, 2, 2]: [channel, tap, side]
    // The block's Q, K, V and output; the context's K and V go through the same key and value.
    ops::SingleProjectionWeight query, key, value, output;
    ops::SingleProjectionWeight gate, up, down;
    // The convolutions' coefficient projections [4 G, H] (rows (2 side + tap) G + group).
    ops::SingleProjectionWeight attention_projection, mlp_projection;
};

struct Operands {
    Tensor context_norm, final_norm;                 // BF16 [H]
    ops::SingleProjectionWeight feature_projection;  // [H, taps * H]
    ops::SingleProjectionWeight selector_projection; // [256, H]
    Tensor predecessor_codebook, successor_codebook; // BF16 [256, vocab]
    std::vector<LayerOperands> layers;
};

using Target       = dspine::Target;
using ContextCall  = dspine::ContextCall;
using ProposalCall = dspine::ProposalCall;

class Drafter {
public:
    // The ring's capacity is the trained window; max_context bounds every frontier, max_batch
    // every proposal's rows (a commit takes up to eight) and max_width (2..block_size) every
    // block.
    Drafter(const Config& config, Operands operands, Target target, std::uint32_t max_context,
            std::int32_t max_batch, std::int32_t max_width);

    void materialize(const ContextCall& call, std::span<const CyclicKVCacheLayerView> ring,
                     WorkspaceArena& work, cudaStream_t stream) const;
    void propose(const ProposalCall& call, std::span<const CyclicKVCacheLayerView> ring,
                 WorkspaceArena& work, cudaStream_t stream) const;

    // Workspace of materialize() over calls of up to `width` columns in `batch` rows.
    [[nodiscard]] static std::size_t context_workspace_bytes(const Config& config,
                                                             const Operands& operands,
                                                             std::int32_t width,
                                                             std::int32_t batch);
    // Workspace of propose() at widths 2..`width` over `batch` rows beside `head_bytes`, the
    // head's own workspace at up to (width - 1) * batch columns.
    [[nodiscard]] static std::size_t proposal_workspace_bytes(const Config& config,
                                                              const Operands& operands,
                                                              DType logits,
                                                              std::uint32_t max_context,
                                                              std::int32_t width,
                                                              std::int32_t batch,
                                                              std::size_t head_bytes);

    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] const Operands& operands() const noexcept { return p_; }

private:
    const Config& config_;
    Operands p_;
    Target target_;
    std::uint32_t max_context_ = 0;
    std::int32_t max_batch_ = 0, max_width_ = 0;
};

} // namespace ninfer::models::dflash2
