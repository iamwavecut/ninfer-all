#pragma once

#include "runtime/prefix_cache/block_hash.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::runtime::prefix_cache {

// Prompt boundary facts supplied by the Frontend (docs/maintainer/hybrid-prefix-cache-spec.md
// §7.1). Positions are exact token frontiers.
enum class TapHintKind : std::uint8_t {
    Explicit,         // client-named breakpoint (explicit-evidence protocol / PromptInput marker)
    GenerationOpener, // start of the final assistant generation opener
    Structural,       // end of tools, end of the leading System/Developer block
    // A protocol-automatic marker (OpenAI default prompt caching, Anthropic request-level
    // cache_control). Planned like a structural boundary, but it marks the conversation's latest
    // turn rather than a prefix conversations share, so its snapshot can be superseded.
    Automatic,
    MessageBoundary, // any message boundary; used to snap ladder taps
};

struct TapHint {
    std::uint32_t position = 0;
    TapHintKind kind       = TapHintKind::MessageBoundary;
};

// Half-open token span in which no resume point may lie (a media item's tokens, media_exclusion).
struct TapExclusion {
    std::uint32_t begin = 0;
    std::uint32_t end   = 0;
};

// The exclusion of a media item whose tokens are [begin, end): the item's span, extended to the end
// of the block the item begins in. A block's key carries every item that begins before the block's
// end, and a snapshot's tail, the tokens past its last full block, is compared as tokens alone,
// where media tokens are placeholders. A frontier inside the block an item begins in would leave
// that item in no key, so two prompts that differ only in that item would share the snapshot.
[[nodiscard]] constexpr TapExclusion media_exclusion(std::uint32_t begin,
                                                     std::uint32_t end) noexcept {
    const std::uint32_t block_end = (begin / kBlockTokens + 1U) * kBlockTokens;
    return TapExclusion{begin, end > block_end ? end : block_end};
}

// Whether a frontier lies inside an exclusion. Both ends of a span are valid frontiers.
[[nodiscard]] constexpr bool inside_exclusion(std::uint32_t frontier,
                                              std::span<const TapExclusion> exclusions) noexcept {
    for (const TapExclusion& span : exclusions) {
        if (span.begin < frontier && frontier < span.end) { return true; }
    }
    return false;
}

// How the prefill loop realizes a tap.
enum class TapPlacement : std::uint8_t {
    // Snapshot exactly at `position`: semantic boundaries where a later prompt is expected to
    // diverge. A text chunk on one device writes the state there in its pass (up to four taps a
    // chunk); otherwise the chunk is split at `position`.
    Exact,
    // Snapshot at a prefill chunk boundary instead of splitting: at the end of the chunk that
    // reaches `position`, or at the start of the prompt's final chunk when `position` lies in it.
    // A split costs one more forward pass, an in-pass tap one more Gated DeltaNet call a layer,
    // a chunk boundary nothing.
    Flexible,
};

struct PlannedTap {
    std::uint32_t position = 0;
    TapPlacement placement = TapPlacement::Exact;
    // A boundary later prompts may share across conversations (a client breakpoint or a
    // structural boundary): its snapshot is published as SnapshotKind::Boundary, superseded only
    // while no other conversation has continued from it.
    bool boundary = false;

    [[nodiscard]] friend bool operator==(PlannedTap, PlannedTap) noexcept = default;
};

// Two taps closer than this cover too little to pay for a second snapshot and split.
inline constexpr std::uint32_t kMinimumTapSeparation = 64;

struct TapPlannerConfig {
    std::uint32_t max_new_taps   = 8;
    std::uint32_t ladder_tokens  = 4096;
    std::uint32_t min_gap_tokens = 1024;
};

// Returns taps sorted by position with base < position <= prompt_tokens - 1, outside every
// exclusion and not near an existing snapshot frontier on the matched path. Priority: Explicit,
// GenerationOpener, Structural, then the flexible prompt tail and geometric history ladder
// (prompt_tokens - G * 2^k, snapped back to a message boundary when one lies within the gap).
// A snapshot serves every prompt diverging after it, so proximity only makes a later tap redundant:
// semantic boundaries (opener, structural) closer than kMinimumTapSeparation keep only the
// earliest, and a non-Explicit, non-ladder candidate is dropped when an accepted tap, an existing
// snapshot or the base lies at most kMinimumTapSeparation before it. Ladder taps keep
// min_gap_tokens spacing on both sides.
[[nodiscard]] std::vector<PlannedTap> plan_taps(std::uint32_t prompt_tokens, std::uint32_t base,
                                                std::span<const TapHint> hints,
                                                std::span<const std::uint32_t> existing_frontiers,
                                                std::span<const TapExclusion> exclusions,
                                                const TapPlannerConfig& config);

} // namespace ninfer::runtime::prefix_cache
