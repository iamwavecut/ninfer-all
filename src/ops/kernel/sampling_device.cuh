#pragma once

// Shared implementation primitives for include/ninfer/ops/sampling.h and
// include/ninfer/ops/speculative_round.h. The ordering key is exact for finite BF16
// logits (including numeric-zero ties); candidate storage is bounded by the
// public top-k cap (kSamplerCandidateCap) and all global staging is supplied by the caller.

#include "ops/common/math.h"
#include "ops/common/sampling_workspace.h"
#include "ops/common/score_id_order.cuh"
#include "ops/common/pre_sm80.cuh"
#include "ninfer/ops/sampling.h"

#include <cub/block/block_merge_sort.cuh>
#include <cub/warp/warp_merge_sort.cuh>

#include <cuda_bf16.h>
#include <climits>
#include <cstdint>
#include <math_constants.h>

namespace ninfer::ops {

// The candidate merges reduce over the first kSamplingTileWarps lanes; redux.sync is sm_80.
template <int Width>
__device__ __forceinline__ unsigned int sampling_lane_max(unsigned int mask, unsigned int value) {
#if defined(NINFER_PRE_SM80_DEVICE)
    return ops::pre_sm80::reduce_max<Width>(mask, value);
#else
    return __reduce_max_sync(mask, value);
#endif
}


using SamplingPartialSort =
    cub::BlockMergeSort<unsigned long long, kSamplerBlock, kSamplerItemsPerThread>;
inline constexpr int kSamplingTileWarps              = kSamplerBlock / 32;
inline constexpr int kSamplingTileIndexBits          = 9;
inline constexpr unsigned int kSamplingTileIndexMask = (1u << kSamplingTileIndexBits) - 1u;
static_assert(kSamplerPartialTileItems == (1 << kSamplingTileIndexBits));

using SamplingTileWarpSort     = cub::WarpMergeSort<unsigned long long, kSamplerItemsPerThread>;
using SamplingBf16TileWarpSort = cub::WarpMergeSort<unsigned int, kSamplerItemsPerThread>;

struct SamplingTileTopKStorage {
    typename SamplingTileWarpSort::TempStorage warp[kSamplingTileWarps];
    unsigned long long candidates[kSamplingTileWarps * kSamplerCandidateCap];
};

struct SamplingBf16TileTopKStorage {
    typename SamplingBf16TileWarpSort::TempStorage warp[kSamplingTileWarps];
    unsigned int candidates[kSamplingTileWarps * kSamplerCandidateCap];
};

union SamplingPartialTopKStorage {
    SamplingTileTopKStorage fp32;
    SamplingBf16TileTopKStorage bf16;
};

// True when the value is a real number. Used to fail a round whose logits diverged instead of
// sampling a plausible-looking token out of NaN (see kSamplerNonFiniteToken).
__device__ __forceinline__ bool sampling_value_is_finite(float value) {
    return value == value && value < CUDART_INF_F && value > -CUDART_INF_F;
}

// Logits are BF16, or FP32 where the head writes its unrounded sums; every reader widens to FP32.
__device__ __forceinline__ float sampling_logit(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ __forceinline__ float sampling_logit(float value) { return value; }

// True when `index` names a token whose logit is finite. A reduction that met only NaN keeps its
// empty sentinel, a negative index or INT_MAX, which names no token and reads nothing.
template <typename Logit>
__device__ __forceinline__ bool
sampling_selected_logit_is_finite(const Logit* logits, std::int64_t base, std::int32_t index) {
    return index >= 0 && index != INT_MAX &&
           sampling_value_is_finite(sampling_logit(logits[base + index]));
}

struct SamplingKeyGreater {
    __device__ __forceinline__ bool operator()(unsigned long long a, unsigned long long b) const {
        return a > b;
    }
};

struct SamplingUintGreater {
    __device__ __forceinline__ bool operator()(unsigned int a, unsigned int b) const {
        return a > b;
    }
};

__device__ inline unsigned long long sampling_block_max_key(unsigned long long key,
                                                            unsigned long long* warp_keys) {
    constexpr unsigned int kMask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const unsigned long long other = __shfl_down_sync(kMask, key, offset);
        if (other > key) { key = other; }
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) { warp_keys[warp] = key; }
    __syncthreads();
    key = (warp == 0 && lane < (blockDim.x >> 5)) ? warp_keys[lane] : 0ull;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            const unsigned long long other = __shfl_down_sync(kMask, key, offset);
            if (other > key) { key = other; }
        }
    }
    return key;
}

__device__ __forceinline__ unsigned long long sampling_splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Uniform float in [0,1) from (seed, position, purpose, sub). Pure function of
// its inputs so it is safe under CUDA-graph replay (no mutable RNG state).
__device__ __forceinline__ float sampling_uniform(unsigned long long seed, int position,
                                                  int purpose, unsigned int sub) {
    unsigned long long key = seed;
    key                    = sampling_splitmix64(
        key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(position)) *
               0xD1B54A32D192ED03ull));
    key = sampling_splitmix64(
        key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(purpose)) << 21) ^
        (static_cast<unsigned long long>(sub) * 0x2545F4914F6CDD1Dull));
    const unsigned int bits = static_cast<unsigned int>(key >> 40); // 24 bits
    return static_cast<float>(bits) * (1.0f / 16777216.0f);
}

// The same keyed uniform in (0,1), for the Gumbel noise of token `sub - 1`.
__device__ __forceinline__ float sampling_uniform_open(unsigned long long seed, int position,
                                                       int purpose, unsigned int sub) {
    return sampling_uniform(seed, position, purpose, sub) + 0.5f / 16777216.0f;
}

// Candidate ordering: higher value wins, ties broken by lower vocab index.
__device__ __forceinline__ bool sampling_better(float v, int i, float bv, int bi) {
    return v > bv || (v == bv && i < bi);
}

// True when (v,i) ranks strictly below pivot (pv,pi) in the ordering above.
__device__ __forceinline__ bool sampling_worse_than(float v, int i, float pv, int pi) {
    return pv > v || (pv == v && pi < i);
}

__device__ __forceinline__ unsigned long long sampling_sort_key(float v, int idx) {
    return score_id_order_key(v, idx);
}

__device__ __forceinline__ unsigned int sampling_bf16_tile_sort_key(__nv_bfloat16 v,
                                                                    int tile_index) {
    unsigned int bits = __bfloat16_as_ushort(v);
    if ((bits & 0x7fffu) == 0u) { bits = 0u; }
    const unsigned int ordered = (bits & 0x8000u) ? ((~bits) & 0xffffu) : (bits ^ 0x8000u);
    return (ordered << kSamplingTileIndexBits) |
           (kSamplingTileIndexMask - static_cast<unsigned int>(tile_index));
}

__device__ __forceinline__ unsigned long long sampling_expand_bf16_tile_sort_key(unsigned int key,
                                                                                 int tile_start) {
    if (key == 0u) { return 0ull; }
    const unsigned int ordered = key >> kSamplingTileIndexBits;
    const unsigned int bits    = (ordered & 0x8000u) ? (ordered ^ 0x8000u) : ((~ordered) & 0xffffu);
    const int token            = tile_start + static_cast<int>(kSamplingTileIndexMask) -
                      static_cast<int>(key & kSamplingTileIndexMask);
    return sampling_sort_key(__uint_as_float(bits << 16), token);
}

__device__ __forceinline__ float sampling_key_float(unsigned long long key) {
    return score_from_order_key(key);
}

__device__ __forceinline__ int sampling_key_index(unsigned long long key) {
    return id_from_order_key(key);
}

__device__ __forceinline__ int sampling_candidate_cap(const SamplingConfig& cfg,
                                                      std::int32_t vocab) {
    // top_k is clamped to the pipeline cap: top_k <= 0 (no explicit limit) or a
    // top_k larger than the cap both select the full kSamplerCandidateCap set.
    int cap = kSamplerCandidateCap;
    if (cfg.top_k > 0 && cfg.top_k < cap) { cap = cfg.top_k; }
    if (vocab < cap) { cap = vocab; }
    return cap;
}

__device__ __forceinline__ int sampling_partial_offset(const SamplingWorkspace& workspace, int col,
                                                       int partial, int j) {
    return ((col * workspace.partial_stride + partial) * kSamplerCandidateCap) + j;
}

__device__ inline void sampling_store_tile_topk(unsigned long long (&keys)[kSamplerItemsPerThread],
                                                int cap, SamplingWorkspace workspace, int col,
                                                int partial, SamplingTileTopKStorage& storage) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    SamplingTileWarpSort(storage.warp[warp]).Sort(keys, SamplingKeyGreater{});
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int rank = lane * kSamplerItemsPerThread + item;
        if (rank < kSamplerCandidateCap) {
            storage.candidates[warp * kSamplerCandidateCap + rank] = keys[item];
        }
    }
    __syncthreads();

    if (warp != 0) { return; }
    if (lane < kSamplingTileWarps) {
        constexpr unsigned int kMergeMask = (1u << kSamplingTileWarps) - 1u;
        int position                      = 0;
        for (int rank = 0; rank < cap; ++rank) {
            const unsigned long long key =
                storage.candidates[lane * kSamplerCandidateCap + position];
            const unsigned int high     = static_cast<unsigned int>(key >> 32);
            const unsigned int max_high = sampling_lane_max<kSamplingTileWarps>(kMergeMask, high);
            const unsigned int low      = high == max_high ? static_cast<unsigned int>(key) : 0u;
            const unsigned int max_low  = sampling_lane_max<kSamplingTileWarps>(kMergeMask, low);
            const unsigned int winners =
                __ballot_sync(kMergeMask, high == max_high && low == max_low);
            const int source = __ffs(static_cast<int>(winners)) - 1;
            if (lane == 0) {
                const int off = sampling_partial_offset(workspace, col, partial, rank);
                workspace.partial_keys[off] =
                    (static_cast<unsigned long long>(max_high) << 32) | max_low;
            }
            if (lane == source) { ++position; }
        }
    }
}

__device__ inline void sampling_store_bf16_tile_topk(unsigned int (&keys)[kSamplerItemsPerThread],
                                                     int cap, int tile_start,
                                                     SamplingWorkspace workspace, int col,
                                                     int partial,
                                                     SamplingBf16TileTopKStorage& storage) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    SamplingBf16TileWarpSort(storage.warp[warp]).Sort(keys, SamplingUintGreater{});
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int rank = lane * kSamplerItemsPerThread + item;
        if (rank < kSamplerCandidateCap) {
            storage.candidates[warp * kSamplerCandidateCap + rank] = keys[item];
        }
    }
    __syncthreads();

    if (warp != 0) { return; }
    if (lane < kSamplingTileWarps) {
        constexpr unsigned int kMergeMask = (1u << kSamplingTileWarps) - 1u;
        int position                      = 0;
        for (int rank = 0; rank < cap; ++rank) {
            const unsigned int key     = storage.candidates[lane * kSamplerCandidateCap + position];
            const unsigned int best    = sampling_lane_max<kSamplingTileWarps>(kMergeMask, key);
            const unsigned int winners = __ballot_sync(kMergeMask, key == best);
            const int source           = __ffs(static_cast<int>(winners)) - 1;
            if (lane == 0) {
                const int off = sampling_partial_offset(workspace, col, partial, rank);
                workspace.partial_keys[off] = sampling_expand_bf16_tile_sort_key(best, tile_start);
            }
            if (lane == source) { ++position; }
        }
    }
}

__device__ __forceinline__ int sampling_dist_offset(int col, int j) {
    return col * kSamplerCandidateCap + j;
}

// Applies presence/frequency penalties to a raw logit. `overlay`/`overlay_len`
// carry a round-local count overlay: tokens already committed earlier in the
// current speculative round but not yet flushed to the global `token_counts`. For speculative
// verify column `col` the overlay is exactly drafts[0..col-1] (statically known,
// since column `col` is only consumed when every earlier draft was accepted), so
// the penalty at each column sees the same prefix a per-token sampler would.
// Non-speculative callers pass no overlay. The scan is bounded by k and
// only runs when penalties are active, so it is free on the no-penalty path.
__device__ __forceinline__ float sampling_adjusted_logit(float raw, int v, const SamplingConfig& c,
                                                         const std::int32_t* overlay = nullptr,
                                                         int overlay_len             = 0) {
    if (c.token_mask != nullptr &&
        (c.token_mask[overlay_len * c.token_mask_stride + v / 32] & (1U << (v % 32))) == 0) {
        return -CUDART_INF_F;
    }
    // A non-finite raw logit must sort as itself, unperturbed. Penalty subtraction on NaN is legal
    // IEEE-754 but not required to preserve the operand's bit pattern, and CUDA's hardware NaN
    // canonicalization does not: `NaN - presence_penalty` comes back as a different NaN encoding
    // than an untouched NaN, which score_id_order_key's total-order key treats as a *different,
    // higher* score. Since the penalty overlay is round-local (only already-drafted tokens get
    // subtracted), a poisoned verify column would then rank its drafted token above every other
    // (untouched-NaN) candidate in that same column -- letting a diverged forward pass spuriously
    // "match" the draft and licensing it as accepted instead of tripping the finiteness guard.
    // Returning raw unchanged keeps every non-finite entry in a column bit-identical, so ties fall
    // back to the id tie-break and the guard sees whichever one wins.
    if (!sampling_value_is_finite(raw)) { return raw; }
    float x = raw;
    if (c.presence_penalty == 0.0f && c.frequency_penalty == 0.0f) { return x; }
    int cnt = c.token_counts != nullptr ? c.token_counts[v] : 0;
    for (int j = 0; j < overlay_len; ++j) {
        if (overlay[j] == v) { ++cnt; }
    }
    if (cnt > 0) { x -= c.presence_penalty; }
    if (c.frequency_penalty != 0.0f) { x -= c.frequency_penalty * static_cast<float>(cnt); }
    return x;
}

__device__ __forceinline__ void sampling_insert_candidate(float* vals, int* idxs, int cap, float v,
                                                          int idx) {
    if (cap <= 0 || !sampling_better(v, idx, vals[cap - 1], idxs[cap - 1])) { return; }
    int pos = cap - 1;
    while (pos > 0 && sampling_better(v, idx, vals[pos - 1], idxs[pos - 1])) {
        vals[pos] = vals[pos - 1];
        idxs[pos] = idxs[pos - 1];
        --pos;
    }
    vals[pos] = v;
    idxs[pos] = idx;
}

__device__ __forceinline__ void sampling_sort_tile_desc(float* vals, int* idxs) {
    const int tid = threadIdx.x;
    for (int size = 2; size <= kSamplerTileItems; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const int other = tid ^ stride;
            __syncthreads();
            if (other > tid) {
                const float ov        = vals[other];
                const int oi          = idxs[other];
                const bool descending = ((tid & size) == 0);
                const bool swap       = descending ? sampling_better(ov, oi, vals[tid], idxs[tid])
                                                   : sampling_better(vals[tid], idxs[tid], ov, oi);
                if (swap) {
                    const float tv = vals[tid];
                    const int ti   = idxs[tid];
                    vals[tid]      = ov;
                    idxs[tid]      = oi;
                    vals[other]    = tv;
                    idxs[other]    = ti;
                }
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void sampling_normalize_support(const SamplingConfig& cfg, float* cand_val,
                                                          int* cand_idx, float* prob, int* n_support,
                                                          int n) {
    const int tid = threadIdx.x;
    if (tid == 0) {
        const float inv_temp = 1.0f / cfg.temperature;
        const float m        = cand_val[0] * inv_temp;
        float sum            = 0.0f;
        for (int j = 0; j < n; ++j) {
            const float e = __expf(cand_val[j] * inv_temp - m);
            prob[j]       = e;
            sum += e;
        }
        const float e0           = prob[0];
        const float min_p_thresh = (cfg.min_p > 0.0f) ? cfg.min_p * e0 : -1.0f;
        const bool top_p_active  = (cfg.top_p < 1.0f);
        const float top_p_target = cfg.top_p * sum;
        float cum                = 0.0f;
        int support              = 0;
        for (int j = 0; j < n; ++j) {
            if (prob[j] == 0.0f || (min_p_thresh >= 0.0f && prob[j] < min_p_thresh)) { break; }
            cum += prob[j];
            support = j + 1;
            if (top_p_active && cum >= top_p_target) { break; }
        }
        if (support < 1) { support = 1; }
        float ssum = 0.0f;
        for (int j = 0; j < support; ++j) { ssum += prob[j]; }
        const float inv = 1.0f / ssum;
        for (int j = 0; j < support; ++j) { prob[j] *= inv; }
        *n_support = support;
    }
    __syncthreads();
}

// All threads of the block must call. Small-column exact truncated support
// builder used by unit tests and any <=256-token column. It sorts one shared
// tile and then applies the same top-p/min-p renormalization as the large path.
template <typename Logit>
__device__ __forceinline__ void
sampling_build_truncated_small(const Logit* logits, std::int64_t base, std::int32_t vocab,
                               const SamplingConfig& cfg, float* tile_val, int* tile_idx,
                               float* cand_val, int* cand_idx, float* prob, int* n_support,
                               const std::int32_t* overlay = nullptr, int overlay_len = 0) {
    const int tid = threadIdx.x;
    const int cap = sampling_candidate_cap(cfg, vocab);
    if (tid < kSamplerTileItems) {
        if (tid < vocab) {
            const float x = sampling_adjusted_logit(sampling_logit(logits[base + tid]), tid, cfg,
                                                    overlay, overlay_len);
            tile_val[tid] = x;
            tile_idx[tid] = tid;
        } else {
            tile_val[tid] = -CUDART_INF_F;
            tile_idx[tid] = INT_MAX;
        }
    }
    __syncthreads();
    sampling_sort_tile_desc(tile_val, tile_idx);
    if (tid < cap) {
        cand_val[tid] = tile_val[tid];
        cand_idx[tid] = tile_idx[tid];
    }
    __syncthreads();
    sampling_normalize_support(cfg, cand_val, cand_idx, prob, n_support, cap);
}

// Warp-level merge of sorted candidate lists. The calling warp (all 32 lanes) writes to out[0..cap)
// the first `cap` keys, descending, of `list_count` <= 32 lists sorted descending, list l at
// lists[l * stride] with `cap` keys. Empty keys (0) rank last.
__device__ inline void sampling_merge_sorted_lists(const unsigned long long* lists, int list_count,
                                                   int stride, int cap, unsigned long long* out) {
    constexpr unsigned int kFull = 0xffffffffu;
    const int lane               = static_cast<int>(threadIdx.x) & 31;
    int position                 = 0;
    for (int rank = 0; rank < cap; ++rank) {
        const unsigned long long key =
            lane < list_count && position < cap ? lists[lane * stride + position] : 0ull;
        const unsigned int high     = static_cast<unsigned int>(key >> 32);
        const unsigned int max_high = sampling_lane_max<32>(kFull, high);
        const unsigned int low      = high == max_high ? static_cast<unsigned int>(key) : 0u;
        const unsigned int max_low  = sampling_lane_max<32>(kFull, low);
        const unsigned int winners  = __ballot_sync(kFull, high == max_high && low == max_low);
        const int source            = __ffs(static_cast<int>(winners)) - 1;
        if (lane == 0) { out[rank] = (static_cast<unsigned long long>(max_high) << 32) | max_low; }
        if (lane == source) { ++position; }
    }
}

struct SamplingMergeStorage {
    unsigned long long lists[kSamplerPartialsPerGroup * kSamplerCandidateCap];
    unsigned long long merged[kSamplerCandidateCap];
};

// All threads of the block must call. Merges the top-cap lists of `count` <= 32 consecutive
// workspace slots from `first` into storage.merged[0..cap).
__device__ inline void sampling_merge_workspace_lists(const SamplingWorkspace& workspace, int col,
                                                      int first, int count, int cap,
                                                      SamplingMergeStorage& storage) {
    const unsigned long long* source =
        workspace.partial_keys + sampling_partial_offset(workspace, col, first, 0);
    for (int i = static_cast<int>(threadIdx.x); i < count * kSamplerCandidateCap;
         i += static_cast<int>(blockDim.x)) {
        if (i % kSamplerCandidateCap < cap) { storage.lists[i] = source[i]; }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        sampling_merge_sorted_lists(storage.lists, count, kSamplerCandidateCap, cap,
                                    storage.merged);
    }
    __syncthreads();
}

// All threads of the block must call. Writes storage.merged[0..cap) to workspace slot `slot`.
__device__ inline void sampling_store_merged(const SamplingWorkspace& workspace, int col, int slot,
                                             int cap, const SamplingMergeStorage& storage) {
    for (int rank = static_cast<int>(threadIdx.x); rank < cap;
         rank += static_cast<int>(blockDim.x)) {
        workspace.partial_keys[sampling_partial_offset(workspace, col, slot, rank)] =
            storage.merged[rank];
    }
}

struct SamplingBlockTopKStorage {
    typename SamplingTileWarpSort::TempStorage warp[kSamplingTileWarps];
    unsigned long long chunk[kSamplingTileWarps][32 * kSamplerItemsPerThread];
    unsigned long long list[2][kSamplingTileWarps][kSamplerCandidateCap];
    unsigned long long merged[kSamplerCandidateCap];
};

// Number of keys in the descending list[0..n) greater than key.
__device__ __forceinline__ int sampling_count_greater(const unsigned long long* list, int n,
                                                      unsigned long long key) {
    int lo = 0;
    int hi = n;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (list[mid] > key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Single-block fallback for large columns when the finite multi-block workspace route cannot
// represent the launch. Each warp reads its share of the vocabulary once, 64 tokens at a time,
// and keeps its own exact top-cap list: a slice none of whose keys beats the list's last is
// skipped, any other is sorted and merged in. Warp 0 then merges the eight lists.
template <typename Logit>
__device__ __forceinline__ void
sampling_build_truncated_block_fast(const Logit* logits, std::int64_t base, std::int32_t vocab,
                                    const SamplingConfig& cfg, SamplingBlockTopKStorage& storage,
                                    float* cand_val, int* cand_idx, float* prob, int* n_support,
                                    const std::int32_t* overlay = nullptr, int overlay_len = 0) {
    constexpr int kSlice         = 32 * kSamplerItemsPerThread;
    constexpr unsigned int kFull = 0xffffffffu;
    const int warp               = static_cast<int>(threadIdx.x) >> 5;
    const int lane               = static_cast<int>(threadIdx.x) & 31;
    const int warps              = static_cast<int>(blockDim.x) >> 5;
    const int cap                = sampling_candidate_cap(cfg, vocab);

    int current = 0;
    for (int j = lane; j < cap; j += 32) { storage.list[0][warp][j] = 0ull; }
    __syncwarp();
    unsigned long long threshold = 0ull;
    for (int slice = warp * kSlice; slice < vocab; slice += warps * kSlice) {
        unsigned long long keys[kSamplerItemsPerThread];
        bool enters = false;
#pragma unroll
        for (int item = 0; item < kSamplerItemsPerThread; ++item) {
            const int v = slice + item * 32 + lane;
            keys[item]  = v < vocab ? sampling_sort_key(sampling_adjusted_logit(
                                                           sampling_logit(logits[base + v]), v, cfg,
                                                           overlay, overlay_len),
                                                       v)
                                    : 0ull;
            enters      = enters || keys[item] > threshold;
        }
        if (!__any_sync(kFull, enters)) { continue; }
        SamplingTileWarpSort(storage.warp[warp]).Sort(keys, SamplingKeyGreater{});
#pragma unroll
        for (int item = 0; item < kSamplerItemsPerThread; ++item) {
            storage.chunk[warp][lane * kSamplerItemsPerThread + item] = keys[item];
        }
        __syncwarp();
        const unsigned long long* list = storage.list[current][warp];
        unsigned long long* next       = storage.list[current ^ 1][warp];
#pragma unroll
        for (int item = 0; item < kSamplerItemsPerThread; ++item) {
            const int rank = lane * kSamplerItemsPerThread + item +
                             sampling_count_greater(list, cap, keys[item]);
            if (rank < cap) { next[rank] = keys[item]; }
        }
        for (int j = lane; j < cap; j += 32) {
            const int rank = j + sampling_count_greater(storage.chunk[warp], kSlice, list[j]);
            if (rank < cap) { next[rank] = list[j]; }
        }
        __syncwarp();
        current ^= 1;
        threshold = storage.list[current][warp][cap - 1];
    }
    if (current != 0) {
        for (int j = lane; j < cap; j += 32) {
            storage.list[0][warp][j] = storage.list[1][warp][j];
        }
    }
    __syncthreads();
    if (warp == 0) {
        sampling_merge_sorted_lists(&storage.list[0][0][0], warps, kSamplerCandidateCap, cap,
                                    storage.merged);
    }
    __syncthreads();
    for (int rank = static_cast<int>(threadIdx.x); rank < cap;
         rank += static_cast<int>(blockDim.x)) {
        cand_val[rank] = sampling_key_float(storage.merged[rank]);
        cand_idx[rank] = sampling_key_index(storage.merged[rank]);
    }
    __syncthreads();
    sampling_normalize_support(cfg, cand_val, cand_idx, prob, n_support, cap);
}

// thread-0 helper: the inverse-CDF draw of every keyed sample -- the rank of the first candidate
// whose cumulative normalized weight exceeds `u`, the last one when rounding leaves the sum at or
// below `u`. sample() and coupled speculative verification share it, so the same key and the
// same distribution give the same token on both routes.
__device__ __forceinline__ int sampling_pick_rank(const float* prob, int n, float u) {
    float acc = 0.0f;
    for (int j = 0; j < n; ++j) {
        acc += prob[j];
        if (u < acc) { return j; }
    }
    return n - 1;
}

// thread-0 helper: the rank of the keyed draw at (position, purpose) from the n candidates `idx`
// with normalized weights `prob`, as cfg.draw selects: the inverse-CDF rank of
// u(seed, position, purpose, 0), or the rank whose log weight plus the Gumbel noise of its token,
// -log(-log u(seed, position, purpose, id + 1)), is largest (the lower rank on a tie). Drafters
// drawing from their own distribution with the target's key share it with sample() and coupled
// verification.
__device__ __forceinline__ int sampling_pick(const SamplingConfig& cfg, const float* prob,
                                             const int* idx, int n, int position, int purpose) {
    if (cfg.draw == kSampleDrawGumbel) {
        int best         = 0;
        float best_score = -CUDART_INF_F;
        for (int j = 0; j < n; ++j) {
            const float u = sampling_uniform_open(cfg.seed, position, purpose,
                                                  static_cast<unsigned int>(idx[j]) + 1u);
            const float score = logf(prob[j]) - logf(-logf(u));
            if (score > best_score) {
                best_score = score;
                best       = j;
            }
        }
        return best;
    }
    return sampling_pick_rank(prob, n, sampling_uniform(cfg.seed, position, purpose, 0u));
}

} // namespace ninfer::ops
