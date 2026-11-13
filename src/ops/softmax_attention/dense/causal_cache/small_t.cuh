#pragma once

// ninfer::ops - split-KV causal small-T attention shared scaffolding. The partial kernels of each
// KV format live in their own headers (bf16/small_t.cuh, int8/small_t.cuh, fp8/small_t.cuh, ...);
// they are fully separate kernels (no shared body) so each format can be optimized independently.
// This header owns only what they share: layout constants, split policies and device helpers. The
// BF16 and INT8 split reducer is in small_t_reduce.cuh, the split bound the host launchers share
// in small_t_policy.h.

#include "ops/common/device_limits.h"
#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/dense/causal_cache/cache_input.h"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/softmax_attention/dense/causal_cache/key_partition.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_policy.h"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

// The shared memory of a quantized-cache (FP8, NVFP4, K8V4) small-T CTA besides its key/value
// stage: Q and the probabilities of `RowTiles` 16-row tiles, their row rescales, the key and value
// scales of `KeyBlock` keys (at most 16 bytes a key each) and 64 staged page IDs.
template <int RowTiles, int KeyBlock>
inline constexpr int kCausalSmallTQuantizedFixedBytes =
    RowTiles * 16 * (kCausalHeadDim * 2 + (KeyBlock < 64 ? 64 : KeyBlock) * 2 + 4) +
    2 * KeyBlock * 16 + 64 * 4;

// Whether a quantized-cache small-T tile widens K and V into one shared tile, V overwriting K once
// QK is done, instead of widening V into its own tile during QK: where the stage with both tiles
// (`SeparateStageBytes`) does not fit a block of the shape set beside the rest. On Turing (64 KiB)
// that is every shape of two and three 16-row tiles. The NVFP4 kernel also stages its one- and
// two-row-tile shapes so on every architecture.
template <int RowTiles, int KeyBlock, int SeparateStageBytes, bool TuringShapes>
inline constexpr bool
    kCausalSmallTCompactKV = kCausalSmallTQuantizedFixedBytes<RowTiles, KeyBlock> +
                                 SeparateStageBytes >
                             shape_max_shared_bytes_per_block(TuringShapes);

// The indexes below take the call's head counts at run time (geometry.cuh): `kv_heads` KV heads
// and `q_heads` = kv_heads * GroupSize query heads.
__device__ __forceinline__ std::int64_t causal_cache_index(int physical_page, int kv_head, int d,
                                                           int page_offset, int kv_heads) {
    return paged_kv_element_offset<kCausalHeadDim>(physical_page, kv_head, page_offset, d,
                                                   kv_heads);
}

__device__ __forceinline__ std::int64_t causal_q_index(int q_head, int d, int token, int q_heads) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kCausalHeadDim) *
               (static_cast<std::int64_t>(q_head) + static_cast<std::int64_t>(q_heads) * token);
}

__device__ __forceinline__ std::int64_t kv_cache_int8_new_index(int kv_head, int d, int token,
                                                                int kv_heads) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kCausalHeadDim) *
               (static_cast<std::int64_t>(kv_head) + static_cast<std::int64_t>(kv_heads) * token);
}

__device__ __forceinline__ std::int64_t
causal_partial_acc_index(int q_head, int d, int token, int split, int tokens, int q_heads) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kCausalHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(q_heads) *
                    (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split));
}

__device__ __forceinline__ std::int64_t causal_partial_stat_index(int q_head, int token, int split,
                                                                  int tokens, int q_heads) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(q_heads) *
               (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split);
}

template <typename Geometry>
__device__ __forceinline__ bool causal_valid_q_head(int kv_head, int q_head, int kv_heads) {
    return kv_head >= 0 && kv_head < kv_heads && q_head >= kv_head * Geometry::GroupSize &&
           q_head < (kv_head + 1) * Geometry::GroupSize;
}

// The key window a tile of verification columns partitions by: its first position plus its
// physical width, as if every column were live, so masking trailing columns never moves the split
// boundaries of the live ones. A round whose drafts the output budget capped then rounds the
// committed KV and GDN state exactly as the uncapped round does. Live positions are sequential
// from the first, so a dense tile's window is its last position plus one. The envelope bounds the
// window where the physical width reaches past it. Masked columns repeat the last live position,
// and no kernel loads a key at or past it plus one (the live end).
__host__ __device__ __forceinline__ int causal_small_t_tile_window(int first_position, int width,
                                                                   int visible_capacity) {
    const int window = first_position + width;
    return window < visible_capacity ? window : visible_capacity;
}

template <typename Geometry>
__device__ __forceinline__ int causal_small_t_default_splits(int window) {
    int target_keys_per_split = 480 / Geometry::SmallTSplitScale;
    if (window <= 4096) {
        target_keys_per_split = 64 / Geometry::SmallTSplitScale;
    } else if (window <= 8198) {
        target_keys_per_split = 128 / Geometry::SmallTSplitScale;
    } else if (window <= 16390) {
        target_keys_per_split = 256 / Geometry::SmallTSplitScale;
    }
    constexpr int kMinSplits = 4 * Geometry::SmallTSplitScale;
    int splits               = div_up(window, target_keys_per_split);
    splits                   = splits > kMinSplits ? splits : kMinSplits;
    return splits < Geometry::SmallTMaximumSplits ? splits : Geometry::SmallTMaximumSplits;
}

// wave_splits, when positive, is the launch's split count per full wave of CTAs: a count above it
// drops to the fewest whole waves that still keep every split within kCausalSmallTSplitKeyLimit.
template <typename Geometry, bool Int8>
__device__ __forceinline__ int causal_small_t_active_splits(int window, int launch_capacity,
                                                            int tokens, int wave_splits = 0) {
    if (window <= 0) { return launch_capacity; }
    int splits = 0;
    if constexpr (Int8) {
        if (tokens == 5 && window > 128 && window <= 512) {
            splits = div_up(window, 32 / Geometry::SmallTSplitScale);
        } else if (tokens >= 6 && window > 128 && window <= 160) {
            constexpr int kKeysPerSplit = Geometry::SmallTSplitScale == 2 ? 17 : 24;
            splits                      = div_up(window, kKeysPerSplit);
        } else if (tokens >= 6 && window > 5000 && window <= 8198) {
            splits             = div_up(window, 192 / Geometry::SmallTSplitScale);
            constexpr int kMin = 4 * Geometry::SmallTSplitScale;
            constexpr int kMax = 42 * Geometry::SmallTSplitScale;
            splits             = splits > kMin ? splits : kMin;
            splits             = splits < kMax ? splits : kMax;
        } else {
            splits = causal_small_t_default_splits<Geometry>(window);
        }
    } else {
        splits = causal_small_t_default_splits<Geometry>(window);
    }
    if (wave_splits > 0 && splits > wave_splits) {
        const int whole_waves =
            wave_splits * div_up(div_up(window, kCausalSmallTSplitKeyLimit), wave_splits);
        splits = splits < whole_waves ? splits : whole_waves;
    }
    return splits < launch_capacity ? splits : launch_capacity;
}

// The active splits of a window under `partition`: one per segment of a segmented partition,
// otherwise the kernel's tuned count. The host sizes a segmented launch's capacity for its whole
// envelope, so the clamp never binds there.
__device__ __forceinline__ int causal_small_t_partition_active(CausalKeyPartition partition,
                                                               int window, int tuned,
                                                               int launch_capacity) {
    if (!partition.segmented()) { return tuned; }
    const int segments = div_up(window, partition.segment_keys);
    return segments < launch_capacity ? segments : launch_capacity;
}

// The keys [start, end) of split `split` under a segmented or strided partition: a segment, or the
// tuned split count cut at whole pages, so no tile of a strided split straddles a page another rank
// holds. The tuned counts keep every split within kCausalSmallTSplitKeyLimit keys, which page
// rounding keeps within the 64 staged page IDs.
__device__ __forceinline__ void causal_small_t_partition_range(CausalKeyPartition partition,
                                                               int window, int split, int active,
                                                               int& start, int& end) {
    if (partition.segmented()) {
        start = split * partition.segment_keys;
        end   = start + partition.segment_keys;
    } else {
        const int pages = div_up(div_up(window, kPagedKVPageSize), active);
        start           = split * pages * kPagedKVPageSize;
        end             = start + pages * kPagedKVPageSize;
    }
    end = end < window ? end : window;
}

// The page of this rank's planes that holds a row of `physical_page`, or -1 where another rank's
// planes hold it (CausalKeyPartition).
__device__ __forceinline__ int causal_strided_page(int physical_page,
                                                   CausalKeyPartition partition) {
    return physical_page % partition.page_stride == partition.page_residue
               ? physical_page / partition.page_stride
               : -1;
}

// A strided split's staged physical page IDs replaced in place by the pages of the rank's planes
// that hold them, or -1 where another rank's planes do; kernels that test each tile instead of
// compacting their pages (causal_small_t_own_pages) take this. Every thread of the block calls it
// once the IDs are staged.
__device__ __forceinline__ void causal_small_t_localize_pages(std::int32_t* pages, int page_count,
                                                              CausalKeyPartition partition) {
    __syncthreads();
    for (int page = static_cast<int>(threadIdx.x); page < page_count;
         page += static_cast<int>(blockDim.x)) {
        pages[page] = causal_strided_page(pages[page], partition);
    }
    __syncthreads();
}

// A strided split's own pages. `pages` holds the physical IDs of the split's `page_count` pages
// from its first (at most 64); every thread of the block calls this once they are staged. It
// leaves in `pages` one entry per page this rank holds, in key order, packing the page of the
// rank's planes with the page's index in the split ((page << 7) | index), and returns how many
// tiles of Bc keys below `split_end` those pages hold. The split starts at a page boundary.
template <int Bc>
__device__ __forceinline__ int causal_small_t_own_pages(std::int32_t* pages, int page_count,
                                                        int split_end,
                                                        CausalKeyPartition partition) {
    static_assert(kPagedKVPageSize % Bc == 0);
    constexpr int kTilesPerPage = kPagedKVPageSize / Bc;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    __syncthreads();
    const int first  = lane < page_count ? causal_strided_page(pages[lane], partition) : -1;
    const int second = lane + 32 < page_count ? causal_strided_page(pages[lane + 32], partition)
                                              : -1;
    const unsigned low  = __ballot_sync(0xffffffffu, first >= 0);
    const unsigned high = __ballot_sync(0xffffffffu, second >= 0);
    __syncthreads();
    if (warp == 0) {
        const unsigned below = (1u << lane) - 1u;
        if (first >= 0) { pages[__popc(low & below)] = (first << 7) | lane; }
        if (second >= 0) { pages[__popc(low) + __popc(high & below)] = (second << 7) | (lane + 32); }
    }
    __syncthreads();
    const int owned = __popc(low) + __popc(high);
    const int last  = page_count - 1;
    const bool last_owned =
        last < 32 ? ((low >> last) & 1u) != 0u : ((high >> (last - 32)) & 1u) != 0u;
    int tiles = owned * kTilesPerPage;
    if (owned > 0 && last_owned) {
        tiles -= kTilesPerPage - div_up(((split_end - 1) & kPagedKVPageMask) + 1, Bc);
    }
    return tiles;
}

// Tile `tile` of a strided split whose first page starts at key `first_key`: its first key, and the
// page of the rank's planes holding it (causal_small_t_own_pages' entries).
template <int Bc>
__device__ __forceinline__ int causal_small_t_owned_tile_key(const std::int32_t* pages,
                                                             int first_key, int tile) {
    constexpr int kTilesPerPage = kPagedKVPageSize / Bc;
    return first_key + ((pages[tile / kTilesPerPage] & 127) << kPagedKVPageShift) +
           (tile % kTilesPerPage) * Bc;
}

template <int Bc>
__device__ __forceinline__ int causal_small_t_owned_tile_page(const std::int32_t* pages,
                                                              int tile) {
    return pages[tile / (kPagedKVPageSize / Bc)] >> 7;
}

// Quantized storages take the plain default tier. This used to ask for SmallTMaximumSplits at
// tokens==1 and window>8198 while the host granted that capacity to fp8 only, so nvfp4 and k8v4
// silently ran fewer splits than this function returned and the two sides disagreed about intent.
// Measurement said the host's default was the better number for all of them (see
// causal_small_t_split_count in small_t.cpp), so the request is gone rather than the grant
// extended, and this now agrees with the host by construction.
template <typename Geometry>
__device__ __forceinline__ int
causal_small_t_quantized_active_splits(int window, int launch_capacity, int tokens) {
    (void)tokens;
    const int splits = causal_small_t_default_splits<Geometry>(window);
    return splits < launch_capacity ? splits : launch_capacity;
}

__device__ __forceinline__ int causal_small_t_tc_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

template <typename Byte>
__device__ __forceinline__ void causal_small_t_store_byte_swizzled(Byte* tile, int row, int d,
                                                                   int d_b16_stride, Byte code) {
    const int col_b16 = d >> 1;
    const int byte    = d & 1;
    const int off     = (row * d_b16_stride + causal_small_t_tc_swz(row, col_b16)) * 2 + byte;
    tile[off]         = code;
}

__device__ __forceinline__ int causal_small_t_tc_swz32(int row, int col) {
    return (((col >> 3) ^ (row & 3)) << 3) | (col & 7);
}

// Signed int8 QK MMA, k=32 contraction. A = 16x32 s8 (4 regs/thread, 4 s8 each),
// B = 8x32 s8 col-major (2 regs/thread), D = 16x8 s32 (4 regs/thread). The A/B
// register byte layout is identical to the m16n8k16 bf16 fragments loaded by
// ldmatrix_x4/x2 over a d-contiguous int8 tile reinterpreted as
// b16 (two packed int8 per 16-bit lane), so the same ldmatrix helpers and XOR
// swizzle feed this MMA. The s32 accumulator layout matches the bf16 f32
// accumulator (c0/c1 -> row groupID, c2/c3 -> row groupID+8), so score
// consumption is unchanged; only per-64-group scale rescale differs.
template <typename Geometry>
__device__ __forceinline__ void causal_small_t_tc_row_to_qt(int row, int tokens, int kv_head,
                                                            int& q_head, int& token) {
    token             = row / Geometry::GroupSize;
    const int local_q = row - token * Geometry::GroupSize;
    q_head            = kv_head * Geometry::GroupSize + local_q;
}

// Merge one query/head's split statistics once per CTA. Published scalars are separate
// from the reduction/weight storage, so later writes cannot race another warp's scalar read.
template <class Geometry>
__device__ __forceinline__ float
causal_merge_split_statistics(const float* partial_m, const float* partial_l, int q_head, int token,
                              int tokens, int splits, float* weights, float* warp_sums,
                              float* scalars, int q_heads) {
    static_assert(Geometry::SmallTMaximumSplits <= 256);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const auto index   = causal_partial_stat_index(q_head, token, tid, tokens, q_heads);
    const float m      = tid < splits ? partial_m[index] : -CUDART_INF_F;
    const float warp_m = warp_max(m);
    if (lane == 0) warp_sums[warp] = warp_m;
    __syncthreads();
    if (warp == 0) {
        const float maximum = warp_max(tid < 8 ? warp_sums[tid] : -CUDART_INF_F);
        if (tid == 0) scalars[0] = maximum;
    }
    __syncthreads();
    const float maximum     = scalars[0];
    const float l           = tid < splits ? partial_l[index] : 0.0f;
    const float weight      = l > 0.0f && maximum > -CUDART_INF_F ? expf(m - maximum) : 0.0f;
    const float denominator = block_reduce_sum<256>(l * weight, warp_sums);
    if (tid == 0) scalars[1] = denominator;
    __syncthreads();
    const float total = scalars[1];
    if (tid < splits) weights[tid] = total > 0.0f ? weight : 0.0f;
    __syncthreads();
    return total;
}

// One column's split partials weighted and summed in split order. Eight splits' partials are read
// before any of them is added, so the reads overlap; the adds keep the original order and a split of
// zero weight is still skipped, so the sum is the one the plain loop gives.
__device__ __forceinline__ float causal_weighted_split_sum(const float* partial_acc,
                                                           const float* weights, int q_head, int d,
                                                           int token, int tokens, int splits,
                                                           int q_heads) {
    constexpr int kBatch = 8;
    float numerator      = 0.0f;
    for (int first = 0; first < splits; first += kBatch) {
        float value[kBatch];
#pragma unroll
        for (int j = 0; j < kBatch; ++j) {
            const int split = first + j;
            value[j] = split < splits && weights[split] != 0.0f
                           ? partial_acc[causal_partial_acc_index(q_head, d, token, split, tokens,
                                                                  q_heads)]
                           : 0.0f;
        }
#pragma unroll
        for (int j = 0; j < kBatch; ++j) {
            const int split = first + j;
            if (split < splits && weights[split] != 0.0f) numerator += value[j] * weights[split];
        }
    }
    return numerator;
}

// A column's output in the cross-rank form a position-split rank exchanges
// (causal_attention_merge_partials): per column, a block of the FP32 outputs [256, H], each
// normalized over the rank's keys alone, followed by their log-sum-exps [H] in natural units of the
// scaled scores (minus infinity where the rank holds none of the column's keys). Columns are
// consecutive blocks, so a range of a call's columns is a partial of its own.
__device__ __forceinline__ void causal_store_partial_output(float* partial_out, int q_head, int d,
                                                            int column, int q_heads, float value,
                                                            float lse, bool lead) {
    const std::int64_t block =
        static_cast<std::int64_t>(column) * (kCausalHeadDim + 1) * static_cast<std::int64_t>(q_heads);
    partial_out[block + static_cast<std::int64_t>(q_head) * kCausalHeadDim + d] = value;
    if (lead) {
        partial_out[block + static_cast<std::int64_t>(kCausalHeadDim) * q_heads + q_head] = lse;
    }
}

} // namespace ninfer::ops
