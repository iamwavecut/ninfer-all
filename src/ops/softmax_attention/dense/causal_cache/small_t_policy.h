#pragma once

// The split bound the causal small-T host launchers (small_t.cpp) and the partial kernels
// (small_t.cuh) share.

namespace ninfer::ops {

// The most keys one split may cover: a split stages at most 64 physical-page IDs, and two 64-key
// pages go to key-tile rounding and page alignment. A split that must cover more, past the maximum
// split count, reads its pages from the block table instead.
inline constexpr int kCausalSmallTSplitKeyLimit = 3968;

} // namespace ninfer::ops
