#pragma once

// The forward pass of DeepSeek-V4.1-Flash over a loaded Model. Each sequence owns, per block, a
// ring of its last window of attention latents (and a round's drafts), and per kv source block
// the compressed latents and the indexer's keys of every compressed position, with the ratio-2
// compressor's incomplete group in a small tail. Prompt chunks attend through a staging ring that
// every sequence's chunks share, a span of several chunks block by block when experts stay off
// the device; decode steps run one token of each of several sequences at once,
// replayed from a CUDA graph per batch size. The Engram rows a pass addresses are read before it
// from the table's files (or RAM) and copied to the device.
//
// With draft_tokens, verify() runs a sequence's next token (its anchor) and up to draft_tokens
// proposed tokens after it (a context lookup's or the drafter's) as decode rows, and commit() keeps
// the first ones. Branches (forward_branches) run one after another from a copy of the sequence's
// state.
//
// With the model's DSpark drafter, every pass taps the streams entering the drafter's target
// blocks for its last rows (a window at most), and those rows seed each sequence's drafter rings
// (a verification's at commit(), the rows kept). draft() runs the drafter's block after a
// sequence's anchor at its next positions, from a CUDA graph.
//
// With a standalone DSpine drafter (ExecutorOptions::dspine), every pass stores the streams' mean
// after each block the drafter taps for the rows a sequence keeps (a prompt call's last window of
// them), those rows go into the sequence's lane of the drafter's context ring (a verification's at
// commit()), and draft() proposes for several sequences at once. Branch passes keep nothing there.
//
// With the model's vision tower (vision_max_patches), encode_image() turns an image's patches into
// its span's rows, and a prompt call's image rows take the place of their tokens' embeddings: they
// route with the image routing bias, and the Engram neither adds to them nor looks back past them.
//
// The blocks after the last kv source block (the decoder blocks) keep no compressed state: each
// reads that block's caches and its own window ring. Under a prefill tail (ExecutorOptions::
// prefill_tail) a prompt call runs them only over the rows that the state at the caller's next kept
// position (forward's state_at) and the call's logit rows depend on; their first rows attend no
// further back than the run's first position (the window floor).

#include "core/device.h"
#include "core/expert_disk.h"
#include "core/tensor.h"
#include "models/deepseek_v41/engram.h"
#include "models/deepseek_v41/model.h"
#include "models/expert_profile.h"
#include "models/qwen4_exp/ngram_table.h"
#include "ninfer/model_enums.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::deepseek_v41 {

// The device lacks the memory the options need (the caches, or the workspace and CUDA graphs
// beside them); the message names the amounts.
struct InsufficientMemory : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct ExecutorOptions {
    std::uint32_t max_context   = 32768;
    std::uint32_t prefill_chunk = 2048; // prompt tokens per forward()
    // The rows of a prompt call the decoder blocks run (Rows: the last prefill_tail_rows, at least
    // the window, before each kept position).
    PrefillTail prefill_tail        = PrefillTail::Full;
    std::uint32_t prefill_tail_rows = 0;
    // GGUF IQ1_M routed experts of a call wider than eight tokens: the integer matrix kernel once
    // the call routes enough rows to each expert (gguf::iq1m_moe_matrix_pays; the vector kernel in
    // groups of eight columns below), or (false) always the vector kernel. Both multiply 8-bit
    // activations, the matrix route's scales in FP32 and the vector route's in binary16; the sums
    // differ in rounding.
    bool iq1m_matrix = true;
    std::uint32_t sequences     = 1;
    bool cuda_graphs            = true;
    // Proposed tokens a verification may carry per sequence after its anchor (0: no
    // verification); a verification's rows, the anchors and drafts of all its sequences, are at
    // most 8.
    std::uint32_t draft_tokens = 0;
    // One checkpoint of every sequence's rings and tails (snapshot/restore) and one held state
    // (hold/restore_hold), allocated with the caches.
    bool checkpoints = true;
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
    // Drafts with the model's DSpine drafter (Model::dspine(), draft_tokens of them at most). When
    // it taps a decoder block, every prompt row runs every block (no prefill tail).
    bool dspine = false;
    // The Engram table's rows and how they are read; none runs the model without its Engram (a
    // non-standard mode that degrades the output).
    std::optional<qwen4_exp::NgramTableLayout> engram_table;
    qwen4_exp::NgramReadOptions engram_read;
    // A prompt call given its sequence's next prompt call (forward's `next`) reads that call's
    // Engram rows into a second pinned buffer while it computes.
    bool engram_read_ahead = true;
    // Prompt calls may carry images (a model loaded with its vision tower): the most patches an
    // image has, which sizes the tower's scratch. 0: text only.
    std::uint32_t vision_max_patches = 0;
    // Expert parallelism (two or more device ranks): each rank's relative share of every block's
    // routed experts, which it serves from host memory or the files through a delivery and device
    // cache of its own; everything else stays on rank 0. Empty with several ranks: shares that give
    // every rank's misses the same bus time from the memory left for its cache (delivery_shares).
    std::vector<double> expert_split;
};

// An image's rows in a prompt call: the call's rows from `row` on, as many as `embeddings` holds
// (BF16 [rows][hidden] in host memory, a slice of the span encode_image() wrote).
struct ImageRows {
    std::uint32_t row = 0;
    std::span<const std::uint16_t> embeddings;
};

// The prompt call that will follow a prompt call of the same sequence: its tokens, and a flag for
// each of its rows an image takes (empty: none).
struct NextPromptCall {
    std::span<const std::int32_t> tokens;
    std::span<const std::uint8_t> image_rows;
};

struct ExecutorMemory {
    std::uint64_t kv_bytes        = 0;
    // Beside an elastic expert cache, the compressed caches grow with their sequences: the bytes
    // backed now, of kv_bytes at most.
    std::uint64_t kv_backed_bytes = 0;
    bool kv_grows                 = false;
    std::uint64_t workspace_bytes = 0;
    // The most a prompt span borrows from an elastic expert cache while it runs (0: none).
    std::uint64_t span_borrow_bytes = 0;
    // Engram rows held in RAM (a table read whole, or the reader's row cache).
    std::uint64_t engram_resident_bytes = 0;
    // The vision tower: the scratch of one encode within workspace_bytes, and the device bytes the
    // tower and its image buffers own beside the model's weights.
    std::uint64_t vision_encode_bytes = 0;
    std::uint64_t vision_owned_bytes  = 0;
    // --draft-experts: the device copy of the drafter's kept experts and their router rows.
    std::uint64_t drafter_expert_bytes = 0;
    // A DSpine drafter's ring, table, features, staging and workspace (its weights are the
    // model's).
    std::uint64_t dspine_bytes = 0;
    ExpertDeliveryMemory experts;
};

// A sequence's checkpoint: its window rings and compressor tails, kept in the executor's
// checkpoint storage of that sequence (one per sequence), and its position and Engram history,
// which Executor::restore resumes from. The compressed caches of earlier positions are not
// copied, since a sequence only rewrites the compressed rows of positions at or past its own.
struct SequenceSnapshot {
    std::uint32_t position = 0;
    EngramHasher::History history;
};

// A sequence's state in host memory, laid out as image_bytes() sizes it: its rings, tails,
// compressed rows below `position` and a DSpine drafter's ring slots of the positions before it. The context cache keeps prefixes this way beyond the device's
// sequences; the header (position and Engram history) travels beside the bytes.
struct SequenceImage {
    std::uint32_t position = 0;
    EngramHasher::History history;
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
    [[nodiscard]] ExpertRouteCounts expert_totals() const;
    // The Engram reader's counters, or empty without the Engram.
    [[nodiscard]] std::optional<NgramTableStats> engram_stats() const;
    [[nodiscard]] std::string engram_description() const;
    // The CPU's share of missing experts (its threads and kernels), or empty when the GPU copies
    // every miss.
    [[nodiscard]] std::string expert_cpu_description() const;
    // Prompt tokens one forward() may take: prefill_chunk, or with host or disk experts a span of
    // several chunks (as many as memory allows beside the expert cache), which forward() runs
    // layer by layer so that each block's missing experts cross the bus once for all of them.
    [[nodiscard]] std::uint32_t prompt_step() const noexcept;
    // Runs up to prompt_step() tokens at the sequence's next positions and leaves the logits of
    // the last `logit_rows` (at most kMaxLogitRows; 0: none, and the last block of a chunk without
    // logit rows only keeps its window latents) in logits(). `images` (in row order,
    // apart) replace their rows' embeddings. `state_at` (0: the call's end) is the position of the
    // next state the caller keeps or resumes from (an anchor, a stored prefix, the prompt's end),
    // at or past the call's end; every prompt call up to it passes the same one. Under a prefill
    // tail a call that ends before it may leave a state that only those calls continue
    // (resumable() false); one whose rows a later call needs but an earlier call skipped throws
    // std::logic_error. `next`, when the caller knows it, is the sequence's next prompt call
    // (ExecutorOptions::engram_read_ahead).
    void forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                 std::uint32_t logit_rows, std::span<const ImageRows> images = {},
                 std::uint32_t state_at = 0, NextPromptCall next = {});
    // Whether the sequence's state at its position is complete. Decode, verify, draft, snapshot()
    // and save_image() of the live state need a complete state; reset(), restore() and
    // load_image() leave one.
    [[nodiscard]] bool resumable(std::uint32_t sequence) const;
    // The rows of a prompt before a position that decide its decoder blocks' state there (the
    // decoder blocks times window - 1, plus one), and the rows a prefill tail runs (0: every row).
    [[nodiscard]] std::uint32_t exact_tail_rows() const noexcept;
    [[nodiscard]] std::uint32_t prefill_tail_rows() const noexcept;
    // Whether prompt calls may carry images.
    [[nodiscard]] bool vision() const noexcept;
    // The span rows of an image of grid_w x grid_h patches (BF16 [grid_h * grid_w][patch features]
    // in host memory, row by row) into `rows` (BF16 [span tokens][hidden]); returns once they are
    // there.
    void encode_image(std::span<const std::uint16_t> patches, int grid_w, int grid_h,
                      std::span<std::uint16_t> rows);
    // Branches of a prompt, each continuing the sequence from its position: branch b runs
    // `lengths[b]` of `tokens` (in branch order, each at most prefill_chunk, at most kMaxLogitRows
    // branches). Leaves the logits of each branch's last token in logits(), one column per branch
    // in order. The sequence keeps its position and state.
    void forward_branches(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                          std::span<const std::uint32_t> lengths);
    // A prompt's last chunk, then branches continuing it from there as forward_branches has
    // them. The sequence ends at the prompt's end.
    void forward_prompt_branches(std::uint32_t sequence, std::span<const std::int32_t> prompt,
                                 std::span<const std::int32_t> tokens,
                                 std::span<const std::uint32_t> lengths);
    // One token of each of `sequences` (distinct) at its next position; logits one column each.
    void decode(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens);
    // BF16 [vocab, rows], ready on head_stream().
    [[nodiscard]] Tensor logits(std::uint32_t rows) const;

    [[nodiscard]] std::uint32_t draft_tokens() const noexcept;
    // The DSpark drafter's block, the drafts a draft() proposes; 0 without the drafter.
    [[nodiscard]] std::uint32_t dspark_block() const noexcept;
    // The drafter's greedy drafts of the dspark_block() positions after the sequence's next token
    // `anchor` (not yet run), and the confidence of each. The sequence's position and backbone
    // state do not change.
    void draft(std::uint32_t sequence, std::int32_t anchor, std::span<std::int32_t> drafts,
               std::span<float> confidence);
    // DSpine (ExecutorOptions::dspine): for each of `sequences` (distinct, at most 8), the greedy
    // drafts of the `k` (1..draft_tokens) positions after its anchor, the token it reads next,
    // and each draft's probability under the drafter, k a sequence in `drafts` and `confidence`;
    // returns once they are on the host.
    void draft(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> anchors,
               std::uint32_t k, std::span<std::int32_t> drafts, std::span<float> confidence);
    // widths[j] tokens of sequence j (its anchor, then its drafts; without widths, tokens.size() /
    // sequences.size() each) at its next positions, as decode rows: logits one column each,
    // sequence by sequence. The positions advance only with commit().
    void verify(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens,
                std::span<const std::uint32_t> widths = {});
    // Keeps the first kept[j] (1..its width) of sequence j's verified tokens.
    // `hold_after[j]` (0: none; at most kept[j]): `holds[j]` also receives the state after that
    // many of sequence j's tokens, as hold() keeps it and restore_hold() returns to. Returns the
    // rows whose state was kept, a bit each (none in an executor without checkpoints).
    std::uint32_t commit(std::span<const std::uint32_t> sequences,
                         std::span<const std::uint32_t> kept,
                         std::span<const std::uint32_t> hold_after = {},
                         std::span<SequenceSnapshot* const> holds = {});
    [[nodiscard]] cudaStream_t head_stream() const noexcept;
    void snapshot(std::uint32_t sequence, SequenceSnapshot& out);
    void restore(std::uint32_t sequence, const SequenceSnapshot& from);
    // A second kept state beside the checkpoint, of an executor with checkpoints: hold() keeps the
    // sequence's live state (false when it is mid-way through a prefill tail, keeping nothing),
    // restore_hold() takes the sequence back to it. The engine keeps the state before the token
    // that begins a stop string's match this way. A DSpine ring keeps no held copy: once it has
    // wrapped, the slots of its window's oldest positions hold the keys of the positions given
    // back, which changes later drafts only.
    bool hold(std::uint32_t sequence, SequenceSnapshot& out);
    void restore_hold(std::uint32_t sequence, const SequenceSnapshot& from);
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

} // namespace ninfer::models::deepseek_v41
