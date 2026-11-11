#pragma once

// Shared causal-attention dimensions and leaf PTX helpers used by the independently tuned
// BF16 and INT8 prompt kernels. This file deliberately owns no staging policy,
// shared-memory arena, warp schedule, or kernel body.

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/common/history_rescale.cuh"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kCausalPromptHeadDim = 256;

// A prompt CTA's query row block and grid y index (its query head, or its KV head under
// PackGqa) in longest-first issue order: the first gridDim.y CTAs take the last row block of every
// y index, whose rows see the most keys, so a causal launch's last wave holds its shortest CTAs.
// Every (row block, y) pair still runs in exactly one CTA, which does the work blockIdx.x and
// blockIdx.y would give it: the same bits.
struct CausalPromptBlock {
    int row_block = 0;
    int y         = 0;
};

__device__ __forceinline__ CausalPromptBlock causal_prompt_block_longest_first() {
    const int columns = static_cast<int>(gridDim.y);
    const int linear  = static_cast<int>(blockIdx.x + blockIdx.y * gridDim.x);
    return {static_cast<int>(gridDim.x) - 1 - linear / columns, linear % columns};
}

inline constexpr int kCausalPromptBr        = 64;
inline constexpr int kCausalPromptBc        = 64;
inline constexpr int kCausalPromptThreads   = 128;
inline constexpr int kCausalPromptSmemBytes = (kCausalPromptBr + 2 * kCausalPromptBc) *
                                              kCausalPromptHeadDim *
                                              static_cast<int>(sizeof(__nv_bfloat16));

template <typename Geometry>
__device__ __forceinline__ std::int64_t causal_prompt_q_index(int q_head, int d, int token,
                                                              int q_heads) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kCausalPromptHeadDim) *
               (static_cast<std::int64_t>(q_head) + static_cast<std::int64_t>(q_heads) * token);
}

// The query heads of a prompt launch: its grid's y extent spans them, or their KV heads when the
// kernel packs a KV head's query heads into one CTA (PackGqa).
template <typename Geometry, bool PackGqa = false>
__device__ __forceinline__ int causal_prompt_q_heads() {
    return static_cast<int>(gridDim.y) * (PackGqa ? Geometry::GroupSize : 1);
}

template <typename Geometry>
__device__ __forceinline__ void causal_prompt_zero_output_rows(__nv_bfloat16* out, int q_head,
                                                               int row_begin, int row_end, int tid,
                                                               int threads, int q_heads) {
    if (row_begin >= row_end) { return; }
    const int elements = (row_end - row_begin) * kCausalPromptHeadDim;
    for (int element = tid; element < elements; element += threads) {
        const int row = row_begin + element / kCausalPromptHeadDim;
        const int d   = element - (row - row_begin) * kCausalPromptHeadDim;
        out[causal_prompt_q_index<Geometry>(q_head, d, row, q_heads)] = __float2bfloat16(0.0f);
    }
}

// XOR-swizzled b16 element address. INT8 operands use the same layout by packing
// two consecutive signed bytes into each b16 lane before ldmatrix.
__device__ __forceinline__ int causal_prompt_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

template <typename Byte>
__device__ __forceinline__ void causal_prompt_store_byte_swizzled(Byte* tile, int row, int d,
                                                                  Byte code) {
    const int col_b16 = d >> 1;
    const int byte    = d & 1;
    const int off = (row * (kCausalPromptHeadDim / 2) + causal_prompt_swz(row, col_b16)) * 2 + byte;
    tile[off]     = code;
}

template <int Columns>
__device__ __forceinline__ int causal_prompt_p_swz(int row, int col) {
    if constexpr (Columns == 16) { return (((col >> 3) ^ (row & 1)) << 3) | (col & 7); }
    if constexpr (Columns == 32) { return (((col >> 3) ^ (row & 3)) << 3) | (col & 7); }
    return causal_prompt_swz(row, col);
}

__device__ __forceinline__ unsigned causal_prompt_swz_addr(unsigned lane_base, unsigned ck,
                                                           unsigned as, unsigned r) {
    return lane_base + ((ck | as) ^ r);
}

} // namespace ninfer::ops
