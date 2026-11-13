#pragma once

// The execution envelopes and limits of Softmax Attention, which its launch code reads; the Op entry
// points are in ninfer/ops/softmax_attention.h, which includes this header.

#include <cstdint>

namespace ninfer::ops {

inline constexpr std::uint32_t kCausalAttentionMaximumVisibleKeys = 1048576;

struct CausalAttentionExecutionEnvelope {
    std::uint32_t min_visible_keys = 0;
    std::uint32_t max_visible_keys = 0;
    // Select the fast prompt kernel for prompt-route launches over an INT8-G64 or row-scaled
    // FP8-E4M3FN cache (the INT8 family: FP16 per-tile PV accumulation; FP8: in-register dequant
    // over the same Flash-2 schedule). Other routes and cache formats ignore it; it never changes
    // the route or workspace.
    bool fast_prompt_kernel = false;
    // Opt into chunked single-row verification through width 64; ordinary prompts keep
    // their existing route. Workspace planning and execution must use the same hint.
    bool wide_verification = false;
    // Single-row prefill: widths 17 through 64 take the chunked small-T route once the visible
    // keys make it faster than the prompt route. The prompt route runs one CTA per query head and
    // row block, so a few query rows over a long context leave most SMs idle; small-T splits the
    // keys across CTAs instead. Workspace planning and execution must use the same hint.
    bool small_prefill = false;
    // Invariant reductions (--invariant-attention) when nonzero: every call of 1 to 16 columns
    // per row, at any batch size, takes the small-T routes, each split covers the keys
    // [s * N, (s + 1) * N) for this N whatever the window, width or batch, and each storage runs
    // one key tile and one reduction order. A live column's output is then the same bytes in every
    // such call that holds it. N comes from causal_attention_invariant_segment_keys for the
    // engine's whole context, so every envelope of one engine carries the same value. Workspace
    // planned with it reserves for calls of either schedule.
    std::uint32_t invariant_segment_keys = 0;
};

struct ContextAttentionExecutionEnvelope {
    std::uint32_t min_context = 0;
    std::uint32_t max_context = 0;
};

// The most partials causal_attention_merge_partials merges.
inline constexpr std::int32_t kCausalAttentionMaximumMergeRanks = 8;

} // namespace ninfer::ops
