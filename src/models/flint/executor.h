#pragma once

// The forward pass of Flint over a loaded Model. Each sequence owns an F16 cache per block: keys
// and values in a ring of the last window (and a round's drafts) for the sliding-window blocks,
// and of every position for the full-attention blocks. Prompt chunks run llama.cpp's tensor-core
// flash attention; decode steps run one token of each of several sequences at once, replayed from
// a CUDA graph per batch size. Both add each head's learned sink to its softmax denominator.
//
// With draft_tokens, verify() runs a sequence's next token (its anchor) and up to draft_tokens
// proposed tokens after it (a context lookup's or a DSpine drafter's) as decode rows, and commit()
// keeps the first ones. With a DSpine drafter (ExecutorOptions::dspine) every pass stores the
// drafter's taps of the rows it computes, the rows a sequence keeps go into the drafter's context
// ring (a lane per sequence and one per checkpoint), and draft() proposes.

#include "core/device.h"
#include "core/expert_disk.h"
#include "core/tensor.h"
#include "models/expert_profile.h"
#include "models/flint/model.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::flint {

// The device lacks the memory the options need (the caches, or the workspace and CUDA graphs
// beside them); the message names the amounts.
struct InsufficientMemory : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct ExecutorOptions {
    std::uint32_t max_context   = 32768;
    std::uint32_t prefill_chunk = 2048; // prompt tokens per forward()
    std::uint32_t sequences     = 1;
    bool cuda_graphs            = true;
    // Proposed tokens a verification may carry per sequence after its anchor (0: no
    // verification); a verification's rows, the anchors and drafts of all its sequences, are at
    // most 8.
    std::uint32_t draft_tokens = 0;
    // One checkpoint of every sequence's rings (snapshot/restore), allocated with the caches.
    bool checkpoints = true;
    // Drafts with the model's DSpine drafter (Model::dspine(), draft_tokens of them at most).
    bool dspine = false;
    std::optional<std::uint64_t> expert_cache_bytes;
    // Host or disk experts: the cache's first contents, the most routed of these counts
    // ([expert layer][expert]) once warm-up is done; none leaves the cache empty then.
    std::optional<ExpertRouteCounts> expert_profile;
    // The share 0..1 of a decode or verification call's missing experts the GPU copies; the CPU
    // computes the rest (cpu_experts.h). Unset: each call splits its misses by measured CPU and
    // bus times. CPU threads: 0 for one per physical core.
    std::optional<float> expert_dma_share;
    std::uint32_t expert_cpu_threads = 0;
    // Disk experts: a drive mirror and the direct path (core/expert_disk.h).
    ExpertDiskReads expert_disk{};
};

struct ExecutorMemory {
    std::uint64_t kv_bytes        = 0;
    // Beside an elastic expert cache, the full-attention caches grow with their sequences: the
    // bytes backed now, of kv_bytes at most.
    std::uint64_t kv_backed_bytes = 0;
    bool kv_grows                 = false;
    std::uint64_t workspace_bytes = 0;
    // A DSpine drafter's ring, table, features and workspace (its weights are the model's).
    std::uint64_t drafter_bytes = 0;
    ExpertDeliveryMemory experts;
};

// A sequence's checkpoint: its sliding-window rings, kept in the executor's checkpoint storage of
// that sequence (one per sequence), and its position, which Executor::restore resumes from. The
// full-attention caches of earlier positions are not copied, since a sequence only writes positions
// at or past its own.
struct SequenceSnapshot {
    std::uint32_t position = 0;
};

// A sequence's state in host memory, laid out as image_bytes() sizes it: its sliding-window rings,
// its full-attention rows below `position` and a DSpine drafter's ring slots of the positions
// before it. The context cache keeps prefixes this way beyond the device's sequences; the header
// travels beside the bytes.
struct SequenceImage {
    std::uint32_t position = 0;
};

class Executor {
public:
    // Logit rows forward() may leave at once.
    static constexpr std::uint32_t kMaxLogitRows = 256;

    Executor(const Model& model, DeviceContext& device, ExecutorOptions options);
    ~Executor();
    Executor(const Executor&)            = delete;
    Executor& operator=(const Executor&) = delete;

    [[nodiscard]] const ExecutorOptions& options() const noexcept;
    [[nodiscard]] ExecutorMemory memory() const noexcept;
    // Position 0; growing caches give the sequence's rows back (ExecutorMemory::kv_grows).
    void reset(std::uint32_t sequence);
    [[nodiscard]] std::uint32_t position(std::uint32_t sequence) const;
    // Loads the kernels every route reaches, and captures the decode graphs.
    void warm_up();
    // Check after a stream completion and before output publication.
    void check_experts() const;
    [[nodiscard]] ExpertDeliveryStats expert_stats() const;
    // Every route the decode and verification calls took through each expert layer's experts
    // since warm-up, the form a recorded routing profile takes.
    [[nodiscard]] ExpertRouteCounts expert_totals() const;
    // The CPU's share of missing experts: its threads and backend, or empty without one.
    [[nodiscard]] std::string expert_cpu_description() const;
    // Runs up to prefill_chunk tokens at the sequence's next positions and leaves the logits of the
    // last `logit_rows` (at most kMaxLogitRows) in logits(). With no logit rows (a chunk whose
    // output nothing reads), a full-attention last block only writes the chunk's keys and values,
    // the bytes the whole block writes.
    void forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                 std::uint32_t logit_rows);
    // Branches of a prompt in one pass, each continuing the sequence from its position: branch b
    // runs `lengths[b]` of `tokens` (in branch order, at most prefill_chunk in all, at most
    // kMaxLogitRows branches) at the positions after the sequence's, seeing the sequence's keys
    // and its own, no other branch's. Leaves the logits of each branch's last token in logits(),
    // one column per branch in order. The sequence keeps its position and window.
    void forward_branches(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                          std::span<const std::uint32_t> lengths);
    // A prompt's last chunk and the first branches of a read in one pass: the sequence runs
    // `prompt` at its next positions, then the branches continue it from there as
    // forward_branches has them (prompt.size() + tokens.size() at most prefill_chunk). The
    // sequence ends at the prompt's end with its window there, as forward leaves it.
    void forward_prompt_branches(std::uint32_t sequence, std::span<const std::int32_t> prompt,
                                 std::span<const std::int32_t> tokens,
                                 std::span<const std::uint32_t> lengths);
    // One token of each of `sequences` (distinct) at its next position; logits one column each.
    void decode(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens);
    // BF16 [vocab, rows], ready on head_stream().
    [[nodiscard]] Tensor logits(std::uint32_t rows) const;

    [[nodiscard]] std::uint32_t draft_tokens() const noexcept;
    // widths[j] tokens of sequence j (its anchor, then its drafts; without widths, tokens.size() /
    // sequences.size() each) at its next positions, as decode rows: logits one column each,
    // sequence by sequence. The positions advance only with commit().
    void verify(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens,
                std::span<const std::uint32_t> widths = {});
    // Keeps the first kept[j] (1..its width) of sequence j's verified tokens.
    void commit(std::span<const std::uint32_t> sequences, std::span<const std::uint32_t> kept);
    // DSpine (ExecutorOptions::dspine): for each of `sequences` (distinct, at most 8), the greedy
    // drafts of the `k` (1..draft_tokens) positions after its anchor, the token it reads next,
    // and each draft's probability under the drafter, k a sequence in `drafts` and `confidence`;
    // returns once they are on the host.
    void draft(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> anchors,
               std::uint32_t k, std::span<std::int32_t> drafts, std::span<float> confidence);
    [[nodiscard]] cudaStream_t head_stream() const noexcept;
    void snapshot(std::uint32_t sequence, SequenceSnapshot& out);
    void restore(std::uint32_t sequence, const SequenceSnapshot& from);
    // Takes the sequence back `tokens` positions, to the state it had before it read them: false,
    // changing nothing, when its sliding-window rings no longer hold the window before that
    // position (they hold it for as many positions as their slots exceed the window and a
    // round's drafts, and not before the sequence's last prompt, checkpoint or image). A DSpine
    // ring that has wrapped keeps, in the slots of the window's oldest `tokens` positions, the
    // keys of the positions given back: later drafts see those slots as context, which changes
    // the drafts only (the verification keeps the output exact).
    bool rewind(std::uint32_t sequence, std::uint32_t tokens);
    // Bytes of an image at `position`.
    [[nodiscard]] std::uint64_t image_bytes(std::uint32_t position) const;
    // Copies the sequence's state into `pages`, pinned host memory that holds image_bytes() of its
    // position in consecutive slices of `page_bytes` (the last one may be partly used): its live
    // state, or the state at `at`, its checkpoint, whose positions it has not written since.
    // Snapshots, restores and image copies are enqueued on the executor's stream, ordered with
    // the work around them, and the host does not wait for them: device work enqueued later sees
    // them done, and pages are only to be rewritten by later image copies.
    SequenceImage save_image(std::uint32_t sequence, const SequenceSnapshot* at,
                             std::span<std::byte* const> pages, std::size_t page_bytes);
    // Makes the sequence hold the image's state.
    void load_image(std::uint32_t sequence, const SequenceImage& image,
                    std::span<std::byte* const> pages, std::size_t page_bytes);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::flint
