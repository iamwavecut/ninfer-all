#pragma once

// A masked drafter beside a family whose engine keeps its own sequences (Gemma 4, Flint,
// DeepSeek-V4.1, Inkling 2; docs/maintainer/dspine.md): a standalone DSpine drafter, or a DFlash2
// drafter the family's artifact carries (models/dflash2/drafter.h). The runtime holds the context
// ring with a lane per sequence and per state the family keeps of it (a checkpoint, a held state),
// DSpine's head-write table, the target features a pass leaves for the ring, a proposal's device
// operands and a workspace of its own. The family stores its taps after each pass, commits the
// kept columns into a lane, copies lanes with its own snapshots, and carries a lane in its
// context-cache images. Every call enqueues on the given stream, on the device that was current
// at construction (the target's head device).

#include "core/arena.h"
#include "core/device_buffer.h"
#include "core/cyclic_kv_cache.h"
#include "core/tensor.h"
#include "models/dflash2/drafter.h"
#include "models/dspine/drafter.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models::dspine {

struct RuntimeOptions {
    std::uint32_t sequences = 1;
    // Lanes per sequence beside its live one: the states the family keeps of it.
    std::uint32_t kept_states = 0;
    std::uint32_t max_context = 0;
    // Drafts a proposal makes, 1..block_size - 1.
    std::uint32_t draft_tokens = 0;
    // Rows a draft() call proposes for at once, 1..8.
    std::uint32_t draft_rows = 1;
    // Rows a commit_rows() call commits at once (a column each), 1..8.
    std::uint32_t commit_rows = 1;
    // The widest run of taps a pass leaves (feature columns).
    std::uint32_t feature_columns = 0;
    // The most columns one cast or context projection takes: a wider store or commit runs in
    // pieces, which bounds the workspace.
    std::uint32_t piece_columns = 2048;
    // The head's own workspace at up to draft_tokens * draft_rows columns.
    std::size_t head_bytes = 0;
};

class Runtime {
public:
    // A standalone DSpine drafter.
    Runtime(const Model& model, Target target, RuntimeOptions options);
    // A DFlash2 drafter of the family's artifact; `config` outlives the runtime.
    Runtime(const dflash2::Config& config, dflash2::Operands operands, Target target,
            RuntimeOptions options);
    ~Runtime();
    Runtime(const Runtime&)            = delete;
    Runtime& operator=(const Runtime&) = delete;

    [[nodiscard]] const RuntimeOptions& options() const noexcept { return options_; }
    // The ring's capacity and the drafter's attention window: ring_capacity() of a DSpine
    // drafter, a DFlash2 drafter's trained window.
    [[nodiscard]] std::uint32_t window() const noexcept { return window_; }
    [[nodiscard]] std::uint32_t taps() const noexcept {
        return static_cast<std::uint32_t>(geometry_.target_layer_ids.size());
    }
    // The trained block: the anchor and block_size() - 1 masks.
    [[nodiscard]] std::uint32_t block_size() const noexcept { return geometry_.block_size; }
    [[nodiscard]] bool dflash2() const noexcept { return dflash2_ != nullptr; }
    // The tap index of target layer `layer`, or -1 when the drafter does not read it.
    [[nodiscard]] int tap_of(std::uint32_t layer) const noexcept;
    // The lane of `sequence`'s state `state` (0: live; 1..kept_states: the family's kept states).
    [[nodiscard]] std::uint32_t lane(std::uint32_t sequence, std::uint32_t state = 0) const;
    // Device bytes the runtime owns: the ring, the table, the features and operands, the
    // workspace.
    [[nodiscard]] std::uint64_t device_bytes() const noexcept;
    // The bytes the constructor allocates for these options (device_bytes() of the result).
    [[nodiscard]] static std::uint64_t planned_bytes(const Model& model, DType logits,
                                                     const RuntimeOptions& options);
    [[nodiscard]] static std::uint64_t planned_bytes(const dflash2::Config& config,
                                                     const dflash2::Operands& operands,
                                                     DType logits, const RuntimeOptions& options);

    // Derives a DSpine drafter's head-write table from the head (Drafter::prepare): once, before
    // the first draft. A DFlash2 drafter has none.
    void prepare(cudaStream_t stream);

    // Stores tap `tap` of `rows` (FP32 or BF16 [H, n], contiguous) at feature columns
    // [column, column + n).
    void store_tap(std::uint32_t tap, const Tensor& rows, std::uint32_t column,
                   cudaStream_t stream);
    // The target's features of `count` committed positions from `position` on, held at feature
    // columns [column, column + count), into `lane`: a sequential run that continues the lane's
    // live interval (a commit wider than the ring keeps its last window, the only positions a
    // later block sees). Count 0 does nothing.
    void commit(std::uint32_t lane, std::uint32_t position, std::uint32_t column,
                std::uint32_t count, cudaStream_t stream);
    // One committed position a row (a decode step's sequences, distinct lanes): row i's position
    // positions[i] continues lane lanes[i] (device I32 [n] each, n <= commit_rows) with feature
    // column column + i. Only device work, so a CUDA graph may capture it.
    void commit_rows(const Tensor& lanes, const Tensor& positions, std::uint32_t column,
                     cudaStream_t stream);
    // One proposal per row (lanes.size() rows, 1..draft_rows): the greedy drafts of the k
    // (1..draft_tokens) positions after the row's anchor, the token at its position, against the
    // context its lane holds before that position, and each draft's probability under the
    // drafter's refined law, k a row in `drafts` and `confidence`; returns once both are on the
    // host.
    void draft(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> anchors,
               std::span<const std::uint32_t> positions, std::uint32_t k,
               std::span<std::int32_t> drafts, std::span<float> confidence, cudaStream_t stream);
    // Copies one lane's whole ring into another.
    void copy_lane(std::uint32_t from, std::uint32_t to, cudaStream_t stream);

    // A lane's part of a context-cache image at `position`: each layer's K and V slots of the
    // positions before it that the ring holds, head by head.
    [[nodiscard]] std::uint64_t image_bytes(std::uint32_t position) const noexcept;
    // Calls copy(device, bytes) over the lane's image regions in the order image_bytes() counts
    // them; the family copies each one to or from its pages.
    template <class Copy>
    void copy_image(std::uint32_t lane, std::uint32_t position, Copy&& copy) const {
        const std::size_t used = std::min<std::size_t>(position, window_);
        if (used == 0) { return; }
        const std::size_t head_dim = static_cast<std::size_t>(ring_->head_dim());
        const std::size_t pitch    = std::size_t(ring_->padded_capacity()) * head_dim * 2;
        for (std::uint32_t layer = 0; layer < ring_->layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = ring_->layer_view(layer);
            for (const Tensor* plane : {&view.k, &view.v}) {
                auto* base = static_cast<std::uint8_t*>(plane->data) +
                             std::size_t(lane) * std::size_t(ring_->num_kv_heads()) * pitch;
                for (std::int32_t h = 0; h < ring_->num_kv_heads(); ++h) {
                    copy(base + std::size_t(h) * pitch, used * head_dim * 2);
                }
            }
        }
    }

    // What the runtime needs of its drafter: the features' and ring's geometry, and the
    // drafter's workspace and table sizes at these options.
    struct Geometry {
        std::uint32_t hidden = 0, vocab = 0, layers = 0, block_size = 0, num_target_layers = 0;
        std::int32_t kv_heads = 0, head_dim = 0;
        std::vector<std::uint32_t> target_layer_ids;
    };
    struct Plan {
        Geometry geometry;
        std::uint32_t window = 0;
        std::size_t table = 0, proposal = 0, prepare = 0;
        // Workspace of a commit of `width` columns in `batch` rows.
        std::size_t context_piece = 0, context_rows = 0;
    };

private:
    void allocate(const Plan& plan);
    void materialize(const ContextCall& call, cudaStream_t stream);

    Geometry geometry_;
    RuntimeOptions options_;
    std::uint32_t window_ = 0;
    std::vector<int> tap_of_;
    DeviceBuffer ring_backing_, table_, features_, operands_, workspace_backing_;
    std::unique_ptr<CyclicKVCache> ring_;
    std::vector<CyclicKVCacheLayerView> views_;
    std::unique_ptr<WorkspaceArena> work_;
    // A proposal's inputs on their way to the device, then its drafts and confidences back.
    std::unique_ptr<PinnedHostBuffer> staging_;
    // The drafter: DSpine's or DFlash2's.
    std::unique_ptr<Drafter> drafter_;
    std::unique_ptr<dflash2::Drafter> dflash2_;
};

} // namespace ninfer::models::dspine
