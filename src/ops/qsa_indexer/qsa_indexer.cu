// ninfer::ops - the block indexer of Qwen3.8-Flash-Next's sparse attention (contract in
// include/ninfer/ops/qsa_indexer.h). Pooling runs one CTA per completed block. Selection scores the
// blocks into workspace, a few tokens with one CTA per token and 64 blocks, prompts in tiles of
// tokens x blocks (grids sized by the capacity, so a graph replays them at any position), then one
// CTA per query runs a four-pass radix select of the 512th key and two ordered passes that keep
// ties at the lower block index and write the blocks in increasing order.
#include "ninfer/ops/qsa_indexer.h"
#include "ninfer/ops/rope.h"

#include "core/device_launch.h"
#include "core/layout.h"
#include "ops/common/device_multiprocessors.h"
#include "ops/common/device_tier.h"
#include "ops/common/math.h"
#include "ops/common/mma.cuh"
#include "ops/common/pre_sm80.cuh"

#include <cub/block/block_scan.cuh>
#include <cuda_bf16.h>
#include <mma.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kHeadDim     = 128;
constexpr int kHeads       = 4;
constexpr int kRotaryPairs = 32;
constexpr int kBlock       = 4;
constexpr int kTopBlocks   = 512;
constexpr int kProjection  = kHeads * kHeadDim + kHeadDim;
constexpr int kThreads     = 512;
constexpr int kScoreBlocks = 64; // blocks one scoring CTA covers
// Prompt tiles: queries prepared once per token, then scored in tiles of tokens x blocks.
constexpr int kTileThreads  = 256;
constexpr int kTileBlocks   = 64;
constexpr int kStageDims    = 16;
constexpr int kWideTokens   = 32;
constexpr int kNarrowTokens = 16;
constexpr int kTensorTokens = 32;
constexpr int kTensorBlocks = 128;
// Below these token counts the per-token scalar kernel is faster: the tiles' separate query
// kernel and eight dependent stages cost 9-25 us, which a context of 525-700 blocks does not repay.
constexpr int kTileMinTokens   = 64;
constexpr int kTensorMinTokens = 80;

// fl32(1e7^(-2i/64)) for i < 32: the text RoPE frequencies of theta 1e7 over 64 rotary dims.
__device__ __forceinline__ float rope_frequency(int pair) {
    return static_cast<float>(exp(-(2.0 * pair / 64.0) * 16.11809565095832)); // ln(1e7)
}

// YaRN's rotation: each pair's inverse frequency and the attention factor on cos and sin. A kernel
// parameter, so a captured graph keeps it. The kernels take it as a template choice: their unscaled
// instantiation is the original code, so it rounds as before.
struct QsaRope {
    float inverse_frequency[kRotaryPairs];
    float attention_factor;
    int scaled;
};

// cos and sin of pair `pair` at `position` under YaRN.
__device__ __forceinline__ void yarn_sincos(const QsaRope& rope, int pair, int position,
                                            float* sine, float* cosine) {
    sincosf(static_cast<float>(position) * rope.inverse_frequency[pair], sine, cosine);
    *sine *= rope.attention_factor;
    *cosine *= rope.attention_factor;
}

// Zero-centred RMSNorm of a 128-vector held one value per thread of a 128-thread group, then the
// rotate-half rotation of its first 64 dims at `position`. `values` is the group's shared row.
template <bool Yarn>
__device__ void norm_rope_row(float* values, const __nv_bfloat16* weight, float eps, int position,
                              int lane128, float* reduce, const QsaRope& rope) {
    const float x = values[lane128];
    float square  = x * x;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) square += __shfl_xor_sync(0xffffffffu, square, offset);
    const int warp = lane128 >> 5;
    if ((lane128 & 31) == 0) reduce[warp] = square;
    __syncwarp();
    __syncthreads();
    const float total = reduce[0] + reduce[1] + reduce[2] + reduce[3];
    const float normed =
        x * rsqrtf(total / kHeadDim + eps) * (1.0f + __bfloat162float(weight[lane128]));
    __syncthreads();
    values[lane128] = normed;
    __syncthreads();
    if (lane128 < kRotaryPairs) {
        float sine, cosine;
        if constexpr (Yarn) {
            yarn_sincos(rope, lane128, position, &sine, &cosine);
        } else {
            const float angle = static_cast<float>(position) * rope_frequency(lane128);
            sincosf(angle, &sine, &cosine);
        }
        const float x1 = values[lane128], x2 = values[lane128 + kRotaryPairs];
        values[lane128]                = x1 * cosine - x2 * sine;
        values[lane128 + kRotaryPairs] = x2 * cosine + x1 * sine;
    }
    __syncthreads();
}

// One 128-thread CTA per block completing in the call: mean of its four raw keys (from the tail
// or the projection), norm, rope at the block's first position. The grid covers every block a call
// of `tokens` can complete; a CTA past the call's completed blocks has nothing to do.
template <bool Yarn>
__global__ void __launch_bounds__(kHeadDim)
    qsa_pool_kernel(const __nv_bfloat16* __restrict__ projection, const int* __restrict__ first,
                    int tokens, const __nv_bfloat16* __restrict__ key_norm, float eps,
                    const float* __restrict__ tail, int capacity, float* __restrict__ pooled,
                    const QsaRope rope) {
    __shared__ float row[kHeadDim];
    __shared__ float reduce[4];
    const int first_position = *first;
    const int block          = first_position / kBlock + static_cast<int>(blockIdx.x);
    if (block >= (first_position + tokens) / kBlock || block >= capacity) { return; }
    const int d     = threadIdx.x;
    float sum       = 0.0f;
#pragma unroll
    for (int i = 0; i < kBlock; ++i) {
        const int position = block * kBlock + i;
        sum += position < first_position
                   ? tail[(position % kBlock) * kHeadDim + d]
                   : __bfloat162float(projection[static_cast<std::int64_t>(position - first_position) *
                                                     kProjection +
                                                 kHeads * kHeadDim + d]);
    }
    row[d] = sum * 0.25f;
    __syncthreads();
    norm_rope_row<Yarn>(row, key_norm, eps, block * kBlock, d, reduce, rope);
    pooled[static_cast<std::int64_t>(block) * kHeadDim + d] = row[d];
}

// The raw keys of the incomplete block after the call, at slot position mod 4. Positions of that
// block before the call are already in place.
__global__ void qsa_tail_kernel(const __nv_bfloat16* __restrict__ projection,
                                const int* __restrict__ first, int tokens,
                                float* __restrict__ tail) {
    const int first_position = *first;
    const int last      = first_position + tokens - 1;
    const int start     = (last + 1) / kBlock * kBlock; // first position of the incomplete block
    const int d         = threadIdx.x;
    for (int position = max(start, first_position); position <= last; ++position) {
        tail[(position % kBlock) * kHeadDim + d] = __bfloat162float(
            projection[static_cast<std::int64_t>(position - first_position) * kProjection +
                       kHeads * kHeadDim + d]);
    }
}

// Orderable key of a non-negative or negative float: larger floats give larger keys.
__device__ __forceinline__ unsigned order_key(float value) {
    const unsigned bits = __float_as_uint(value);
    return bits & 0x80000000u ? ~bits : bits | 0x80000000u;
}

// One query preparation per token, shared by every block tile. Keep the original FP32
// normalization/reduction and angle boundaries; only its materialization location changes.
template <bool Yarn>
__global__ void __launch_bounds__(kThreads)
    qsa_query_kernel(const __nv_bfloat16* projection, const int* first,
                     const __nv_bfloat16* query_norm, float eps, float* queries,
                     const QsaRope rope) {
    __shared__ float row[kHeads * kHeadDim];
    __shared__ float reduce[kHeads][4];
    const int tid = threadIdx.x, head = tid / kHeadDim, lane = tid % kHeadDim;
    const int t = blockIdx.x, position = *first + t;
    if ((position + 1) / kBlock <= kTopBlocks) { return; }
    row[tid] = __bfloat162float(projection[static_cast<std::int64_t>(t) * kProjection + tid]);
    __syncthreads();
    norm_rope_row<Yarn>(row + head * kHeadDim, query_norm, eps, position, lane, reduce[head], rope);
    queries[static_cast<std::int64_t>(t) * kHeads * kHeadDim + tid] = row[tid];
}

// Prompt scoring tiles: a 256-thread CTA scores Tok tokens (4 Tok query rows, one per head) against
// 64 blocks in FP32 FMA, a 16 x 16 thread grid where each thread owns Tok / 16 tokens and four
// blocks 1 apart. Dims arrive in eight 16-dim stages through double-buffered shared memory, the
// next stage's loads in flight while the current one is multiplied. Each stage's 16 products run in
// a fresh partial that is then added to the total: one FMA chain over all 128 dims erred up to
// 9.8e-8 of the products' absolute sum, the partials 4.6e-8 (the scalar route 4.4e-8) on an RTX
// 3090, 4090, 5090, 5060 Ti, 2080 Ti, Tesla V100 and A100. GeForce parts multiply FP32 as fast as
// TF32 on their tensor cores, which need three products for FP32 accuracy: at 512 tokens and 32768
// blocks the wide tile took 0.28-0.35 of the time of the earlier 3xTF32 MMA route on the RTX 3090,
// 4090, 5090 and 5060 Ti, and 0.33-0.42 of its FMA form on the 2080 Ti and V100.
template <int Tok>
__global__ void __launch_bounds__(kTileThreads, 2)
    qsa_tile_score_kernel(const float* __restrict__ queries, const int* __restrict__ first,
                          const float* __restrict__ pooled, unsigned* __restrict__ keys_workspace,
                          int tokens, int capacity) {
    constexpr int kTokensPerThread = Tok / 16;
    constexpr int kRows            = Tok * kHeads;
    constexpr int kParts           = kStageDims / 4;          // float4 loads per row and stage
    constexpr int kLines           = kTileThreads / kParts;   // rows one pass of loads covers
    constexpr int kQueryLoads      = kRows / kLines;
    constexpr int kKeyLoads        = kTileBlocks / kLines;
    constexpr int kStages          = kHeadDim / kStageDims;
    static_assert(Tok % 16 == 0 && kRows % kLines == 0 && kTileBlocks % kLines == 0,
                  "whole passes of loads");
    // Rows padded by four floats, which keeps the float4 reads aligned.
    __shared__ __align__(16) float query_tile[2][kStageDims][kRows + 4];
    __shared__ __align__(16) float key_tile[2][kStageDims][kTileBlocks + 4];
    const int tid = threadIdx.x;
    const int t0 = blockIdx.x * Tok, b0 = blockIdx.y * kTileBlocks;
    const int start  = *first;
    const int blocks = min((start + min(t0 + Tok, tokens)) / kBlock, capacity);
    if (blocks <= kTopBlocks || b0 >= blocks) { return; }
    const int part = tid % kParts, line = tid / kParts;
    const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float4* query_src[kQueryLoads];
    bool query_ok[kQueryLoads];
#pragma unroll
    for (int l = 0; l < kQueryLoads; ++l) {
        const int row   = line + kLines * l;
        const int token = t0 + row / kHeads;
        query_ok[l]     = token < tokens && (start + token + 1) / kBlock > kTopBlocks;
        query_src[l]    = reinterpret_cast<const float4*>(
                           queries + (static_cast<std::int64_t>(t0) * kHeads + row) * kHeadDim) +
                       part;
    }
    const float4* key_src[kKeyLoads];
    bool key_ok[kKeyLoads];
#pragma unroll
    for (int l = 0; l < kKeyLoads; ++l) {
        const int row = line + kLines * l;
        key_ok[l]     = b0 + row < blocks;
        key_src[l]    = reinterpret_cast<const float4*>(
                         pooled + static_cast<std::int64_t>(b0 + row) * kHeadDim) +
                     part;
    }
    float4 query_v[kQueryLoads], key_v[kKeyLoads];
    const auto load = [&](int stage) {
#pragma unroll
        for (int l = 0; l < kQueryLoads; ++l) {
            query_v[l] = query_ok[l] ? query_src[l][kParts * stage] : zero;
        }
#pragma unroll
        for (int l = 0; l < kKeyLoads; ++l) {
            key_v[l] = key_ok[l] ? key_src[l][kParts * stage] : zero;
        }
    };
    const auto put = [&](int buffer) {
#pragma unroll
        for (int l = 0; l < kQueryLoads; ++l) {
            const int row = line + kLines * l;
            query_tile[buffer][4 * part + 0][row] = query_v[l].x;
            query_tile[buffer][4 * part + 1][row] = query_v[l].y;
            query_tile[buffer][4 * part + 2][row] = query_v[l].z;
            query_tile[buffer][4 * part + 3][row] = query_v[l].w;
        }
#pragma unroll
        for (int l = 0; l < kKeyLoads; ++l) {
            const int row = line + kLines * l;
            key_tile[buffer][4 * part + 0][row] = key_v[l].x;
            key_tile[buffer][4 * part + 1][row] = key_v[l].y;
            key_tile[buffer][4 * part + 2][row] = key_v[l].z;
            key_tile[buffer][4 * part + 3][row] = key_v[l].w;
        }
    };
    load(0);
    put(0);
    __syncthreads();
    const int tx = tid & 15, ty = tid >> 4;
    constexpr int kOwnRows = 4 * kTokensPerThread;
    float total[kOwnRows][4];
#pragma unroll
    for (int i = 0; i < kOwnRows; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) total[i][j] = 0.0f;
    }
#pragma unroll 1
    for (int stage = 0; stage < kStages; ++stage) {
        const int buffer = stage & 1;
        if (stage + 1 < kStages) { load(stage + 1); }
        float partial[kOwnRows][4];
#pragma unroll
        for (int i = 0; i < kOwnRows; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) partial[i][j] = 0.0f;
        }
#pragma unroll
        for (int d = 0; d < kStageDims; ++d) {
            float q[kOwnRows];
#pragma unroll
            for (int k = 0; k < kTokensPerThread; ++k) {
                const float4 x =
                    *reinterpret_cast<const float4*>(&query_tile[buffer][d][4 * (ty + 16 * k)]);
                q[4 * k + 0] = x.x;
                q[4 * k + 1] = x.y;
                q[4 * k + 2] = x.z;
                q[4 * k + 3] = x.w;
            }
            const float4 key = *reinterpret_cast<const float4*>(&key_tile[buffer][d][4 * tx]);
            const float kv[4] = {key.x, key.y, key.z, key.w};
#pragma unroll
            for (int i = 0; i < kOwnRows; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) partial[i][j] = fmaf(q[i], kv[j], partial[i][j]);
            }
        }
#pragma unroll
        for (int i = 0; i < kOwnRows; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) total[i][j] += partial[i][j];
        }
        if (stage + 1 < kStages) { put(buffer ^ 1); }
        __syncthreads();
    }
#pragma unroll
    for (int k = 0; k < kTokensPerThread; ++k) {
        const int t       = t0 + ty + 16 * k;
        const int visible = min((start + t + 1) / kBlock, capacity);
        if (t < tokens && visible > kTopBlocks) {
            unsigned* keys = keys_workspace + static_cast<std::int64_t>(t) * capacity;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int b = b0 + 4 * tx + j;
                if (b < visible) {
                    float score = 0.0f;
#pragma unroll
                    for (int h = 0; h < kHeads; ++h) score += fmaxf(total[4 * k + h][j], 0.0f);
                    keys[b] = order_key(score);
                }
            }
        }
    }
}

// The A100 class's prompt tile: 32 tokens x 128 blocks per 256-thread CTA in m16n8k8 TF32 MMA,
// whose TF32 rate there is eight times its FP32 FMA rate (0.63 of the wide SIMT tile's time and
// 0.39 of the earlier route's at 512 tokens and 32768 blocks). Each FP32 operand splits into a TF32
// high part and its TF32 residual; three products omit low*low. Each 8-dim step starts the high
// product and the two cross products in zeroed accumulators and adds them to the FP32 total in
// registers: the tensor cores lose bits when they add products into a larger accumulator, and one
// fragment accumulating all 16 steps erred up to 6.7e-7 of the products' absolute sum and selected
// blocks outside the contract's 1e-5 tie band on an RTX 5060 Ti. Per-step accumulators err at most
// 4.7e-8. Eight warps: warp & 1 takes tokens 16 (warp & 1) to 16 (warp & 1) + 15, warp >> 1 blocks
// 32 (warp >> 1) to 32 (warp >> 1) + 31; MMA k index c is dim 2c of the step's eight and c + 4 dim
// 2c + 1, for queries and keys alike. Only an sm_80 image runs it (tensor_scoring_here); the other
// images hold a trap in its place.
__global__ void __launch_bounds__(kTileThreads, 1)
    qsa_tensor_score_kernel(const float* __restrict__ queries, const int* __restrict__ first,
                            const float* __restrict__ pooled, unsigned* __restrict__ keys_workspace,
                            int tokens, int capacity) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 800
    (void)queries, (void)first, (void)pooled, (void)keys_workspace, (void)tokens, (void)capacity;
    __trap();
#else
    namespace wmma = nvcuda::wmma;
    constexpr int kTiles  = kTensorBlocks / 32; // n8 tiles per warp
    constexpr int kSteps  = kHeadDim / 8;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int wm = warp & 1, wn = warp >> 1;
    const int t0 = blockIdx.x * kTensorTokens, b0 = blockIdx.y * kTensorBlocks;
    const int start  = *first;
    const int blocks = min((start + min(t0 + kTensorTokens, tokens)) / kBlock, capacity);
    if (blocks <= kTopBlocks || b0 >= blocks) { return; }
    const float* query_base[2];
    bool query_ok[2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int token  = t0 + 16 * wm + g + 8 * half;
        query_ok[half]   = token < tokens && (start + token + 1) / kBlock > kTopBlocks;
        query_base[half] = queries + static_cast<std::int64_t>(token) * kHeads * kHeadDim + 2 * c;
    }
    const float* key_base[kTiles];
    bool key_ok[kTiles];
#pragma unroll
    for (int n = 0; n < kTiles; ++n) {
        const int b  = b0 + 8 * kTiles * wn + 8 * n + g;
        key_ok[n]    = b < blocks;
        key_base[n]  = pooled + static_cast<std::int64_t>(b) * kHeadDim + 2 * c;
    }
    const float2 zero = make_float2(0.0f, 0.0f);
    const auto load = [&](int step, float2 (&query_v)[2][kHeads], float2 (&key_v)[kTiles]) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int h = 0; h < kHeads; ++h) {
                query_v[half][h] = query_ok[half] ? *reinterpret_cast<const float2*>(
                                                        query_base[half] + h * kHeadDim + 8 * step)
                                                  : zero;
            }
        }
#pragma unroll
        for (int n = 0; n < kTiles; ++n) {
            key_v[n] = key_ok[n] ? *reinterpret_cast<const float2*>(key_base[n] + 8 * step) : zero;
        }
    };
    const auto split = [](float x, unsigned& high, unsigned& low) {
        const float h = wmma::__float_to_tf32(x);
        high          = __float_as_uint(h);
        low           = __float_as_uint(wmma::__float_to_tf32(x - h));
    };
    float2 query_v[2][kHeads], key_v[kTiles];
    load(0, query_v, key_v);
    float total[kHeads][kTiles][4];
#pragma unroll
    for (int h = 0; h < kHeads; ++h) {
#pragma unroll
        for (int n = 0; n < kTiles; ++n) {
#pragma unroll
            for (int e = 0; e < 4; ++e) total[h][n][e] = 0.0f;
        }
    }
#pragma unroll 1
    for (int step = 0; step < kSteps; ++step) {
        float2 query_next[2][kHeads], key_next[kTiles];
        if (step + 1 < kSteps) { load(step + 1, query_next, key_next); }
        unsigned key_high[kTiles][2], key_low[kTiles][2];
#pragma unroll
        for (int n = 0; n < kTiles; ++n) {
            split(key_v[n].x, key_high[n][0], key_low[n][0]);
            split(key_v[n].y, key_high[n][1], key_low[n][1]);
        }
#pragma unroll
        for (int h = 0; h < kHeads; ++h) {
            const float x[4] = {query_v[0][h].x, query_v[1][h].x, query_v[0][h].y,
                                query_v[1][h].y};
            unsigned high[4], low[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) split(x[i], high[i], low[i]);
#pragma unroll
            for (int n = 0; n < kTiles; ++n) {
                float product[4] = {0.0f, 0.0f, 0.0f, 0.0f}, cross[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                mma_tf32_bits(product[0], product[1], product[2], product[3], high[0], high[1],
                              high[2], high[3], key_high[n][0], key_high[n][1]);
                mma_tf32_bits(cross[0], cross[1], cross[2], cross[3], high[0], high[1], high[2],
                              high[3], key_low[n][0], key_low[n][1]);
                mma_tf32_bits(cross[0], cross[1], cross[2], cross[3], low[0], low[1], low[2],
                              low[3], key_high[n][0], key_high[n][1]);
#pragma unroll
                for (int e = 0; e < 4; ++e) total[h][n][e] += product[e] + cross[e];
            }
        }
        if (step + 1 < kSteps) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
#pragma unroll
                for (int h = 0; h < kHeads; ++h) query_v[half][h] = query_next[half][h];
            }
#pragma unroll
            for (int n = 0; n < kTiles; ++n) key_v[n] = key_next[n];
        }
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int t       = t0 + 16 * wm + g + 8 * half;
        const int visible = min((start + t + 1) / kBlock, capacity);
        if (t < tokens && visible > kTopBlocks) {
            unsigned* keys = keys_workspace + static_cast<std::int64_t>(t) * capacity;
#pragma unroll
            for (int n = 0; n < kTiles; ++n) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int b = b0 + 8 * kTiles * wn + 8 * n + 2 * c + e;
                    if (b < visible) {
                        float score = 0.0f;
#pragma unroll
                        for (int h = 0; h < kHeads; ++h) {
                            score += fmaxf(total[h][n][2 * half + e], 0.0f);
                        }
                        keys[b] = order_key(score);
                    }
                }
            }
        }
    }
#endif
}

// The scores of blocks [64 x, 64 x + 64) for query y (the 1/sqrt(128) scale does not change the
// order), as orderable keys. A query that sees at most 512 blocks needs none.
template <bool Yarn>
__global__ void __launch_bounds__(kThreads)
    qsa_score_kernel(const __nv_bfloat16* __restrict__ projection, const int* __restrict__ first,
                     const __nv_bfloat16* __restrict__ query_norm, float eps,
                     const float* __restrict__ pooled, unsigned* __restrict__ keys_workspace,
                     int capacity, const QsaRope rope) {
    __shared__ __align__(16) float query[kHeads * kHeadDim];
    __shared__ float reduce[kHeads][4];
    const int t        = blockIdx.y;
    const int position = *first + t;
    const int blocks   = min((position + 1) / kBlock, capacity);
    const int b0       = blockIdx.x * kScoreBlocks;
    if (blocks <= kTopBlocks || b0 >= blocks) { return; }
    const int tid = threadIdx.x;

    // Normalised, rotated query heads: four 128-thread groups.
    const int head = tid / kHeadDim, lane = tid % kHeadDim;
    query[tid] = __bfloat162float(projection[static_cast<std::int64_t>(t) * kProjection + tid]);
    __syncthreads();
    {
        // norm_rope_row synchronises the whole CTA; every group runs it together.
        float* row = query + head * kHeadDim;
        const float x = row[lane];
        float square  = x * x;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) square += __shfl_xor_sync(0xffffffffu, square, offset);
        if ((lane & 31) == 0) reduce[head][lane >> 5] = square;
        __syncthreads();
        const float total = reduce[head][0] + reduce[head][1] + reduce[head][2] + reduce[head][3];
        const float normed =
            x * rsqrtf(total / kHeadDim + eps) * (1.0f + __bfloat162float(query_norm[lane]));
        __syncthreads();
        row[lane] = normed;
        __syncthreads();
        if (lane < kRotaryPairs) {
            float sine, cosine;
            if constexpr (Yarn) {
                yarn_sincos(rope, lane, position, &sine, &cosine);
            } else {
                const float angle = static_cast<float>(position) * rope_frequency(lane);
                sincosf(angle, &sine, &cosine);
            }
            const float x1 = row[lane], x2 = row[lane + kRotaryPairs];
            row[lane]                = x1 * cosine - x2 * sine;
            row[lane + kRotaryPairs] = x2 * cosine + x1 * sine;
        }
        __syncthreads();
    }

    // One warp per block, four dims per lane against every head.
    unsigned* keys = keys_workspace + static_cast<std::int64_t>(t) * capacity;
    const int warp = tid >> 5, warp_lane = tid & 31;
    float4 q[kHeads];
#pragma unroll
    for (int h = 0; h < kHeads; ++h) {
        q[h] = reinterpret_cast<const float4*>(query + h * kHeadDim)[warp_lane];
    }
    for (int b = b0 + warp; b < min(b0 + kScoreBlocks, blocks); b += kThreads / 32) {
        const float4 k = reinterpret_cast<const float4*>(pooled + static_cast<std::int64_t>(b) *
                                                                      kHeadDim)[warp_lane];
        float score    = 0.0f;
#pragma unroll
        for (int h = 0; h < kHeads; ++h) {
            float dot = q[h].x * k.x + q[h].y * k.y + q[h].z * k.z + q[h].w * k.w;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, offset);
            score += fmaxf(dot, 0.0f);
        }
        if (warp_lane == 0) keys[b] = order_key(score);
    }
}

__global__ void __launch_bounds__(kThreads)
    qsa_select_kernel(const int* __restrict__ first, const unsigned* __restrict__ keys_workspace,
                      int capacity, int* __restrict__ selected, int* __restrict__ counts) {
    using Scan = cub::BlockScan<int, kThreads>;
    __shared__ typename Scan::TempStorage scan_storage;
    __shared__ int histogram[256];
    __shared__ unsigned prefix_shared;
    __shared__ int remaining_shared;
    __shared__ int carry;

    const int t        = blockIdx.x;
    const int position = *first + t;
    const int blocks   = min((position + 1) / kBlock, capacity);
    int* out           = selected + static_cast<std::int64_t>(t) * kTopBlocks;
    const int tid      = threadIdx.x;
    if (blocks <= kTopBlocks) {
        for (int b = tid; b < blocks; b += kThreads) out[b] = b;
        if (tid == 0) counts[t] = blocks;
        return;
    }
    const unsigned* keys = keys_workspace + static_cast<std::int64_t>(t) * capacity;

    // Radix select of the kTopBlocks-th largest key, eight bits at a time from the top.
    if (tid == 0) {
        prefix_shared    = 0;
        remaining_shared = kTopBlocks;
    }
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += kThreads) histogram[i] = 0;
        __syncthreads();
        const unsigned prefix   = prefix_shared;
        const unsigned high_mask = shift == 24 ? 0u : ~0u << (shift + 8);
        for (int b = tid; b < blocks; b += kThreads) {
            const unsigned key = keys[b];
            if ((key & high_mask) == prefix) atomicAdd(&histogram[(key >> shift) & 0xff], 1);
        }
        __syncthreads();
        if (tid == 0) {
            int remaining = remaining_shared;
            int digit     = 255;
            for (; digit > 0; --digit) {
                if (histogram[digit] >= remaining) break;
                remaining -= histogram[digit];
            }
            prefix_shared    = prefix | (static_cast<unsigned>(digit) << shift);
            remaining_shared = remaining; // how many keys equal to the threshold to keep
        }
        __syncthreads();
    }
    const unsigned threshold = prefix_shared;
    const int keep_equal     = remaining_shared;

    // Ordered compaction: every key above the threshold, and the first keep_equal keys equal to
    // it by block index, written in increasing block order.
    if (tid == 0) carry = 0;
    int equal_seen = 0; // equal keys before this chunk (uniform across the CTA)
    __syncthreads();
    for (int base = 0; base < blocks; base += kThreads) {
        const int b         = base + tid;
        const unsigned key  = b < blocks ? keys[b] : 0u;
        const int is_equal  = b < blocks && key == threshold;
        int equal_rank      = 0, equal_total = 0;
        Scan(scan_storage).ExclusiveSum(is_equal, equal_rank, equal_total);
        __syncthreads();
        const int take = b < blocks && (key > threshold || (is_equal && equal_seen + equal_rank < keep_equal));
        int slot = 0, taken = 0;
        Scan(scan_storage).ExclusiveSum(take, slot, taken);
        if (take) out[carry + slot] = b;
        __syncthreads();
        if (tid == 0) carry += taken;
        equal_seen += equal_total;
        __syncthreads();
    }
    if (tid == 0) counts[t] = carry;
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("qsa_indexer: ") + message); }
}

void require_projection(const Tensor& projection) {
    require(projection.dtype == DType::BF16 && projection.is_contiguous() &&
                projection.data != nullptr && projection.ne[0] == kProjection &&
                projection.ne[1] > 0 && projection.ne[2] == 1 && projection.ne[3] == 1,
            "projection must be contiguous BF16 [640, tokens]");
}

const int* require_position(const Tensor& first_position) {
    require(first_position.dtype == DType::I32 && first_position.data != nullptr &&
                first_position.numel() >= 1,
            "first_position must be a device I32 word");
    return static_cast<const int*>(first_position.data);
}

void require_norm(const Tensor* norm, const char* message) {
    require(norm != nullptr && norm->data != nullptr && norm->dtype == DType::BF16 &&
                norm->is_contiguous() && norm->ne[0] == kHeadDim && norm->ne[1] == 1,
            message);
}

void require_pooled(const Tensor& pooled) {
    require(pooled.dtype == DType::FP32 && pooled.is_contiguous() && pooled.data != nullptr &&
                pooled.ne[0] == kHeadDim && pooled.ne[1] > 0 && pooled.ne[2] == 1,
            "pooled must be contiguous FP32 [128, capacity]");
}

// Scoring routes. The per-token scalar kernel prepares its query in every CTA and reads the keys
// once per token; the tiles read them once per 16 or 32 tokens after one query kernel.
enum class ScoreRoute : std::uint8_t { Scalar, Narrow, Wide, Tensor };

// Whether the current device is an A100-class part (compute capability 8.0) running this binary's
// sm_80 code, whose TF32 tensor cores outrun its FP32 FMA (qsa_tensor_score_kernel).
bool tensor_scoring_here() {
    if (current_device_tier() != DeviceTier::Ampere) { return false; }
    return detail::current_device_compute_capability(0) == 80;
}

// The route of `tokens` queries over `capacity` blocks. The grid is sized by the capacity (a graph
// replays it at any position), so the choice is too. Wide tiles halve the key reads of narrow ones
// and win once their CTAs fill the SMs; narrow tiles keep more CTAs at work below that. The A100's
// tensor tile wins at half a wave of its CTAs. Measured on the cards of the score kernels' comments
// at 1-2048 tokens and 700-114688 blocks, full and 525-block contexts.
ScoreRoute score_route(std::int32_t tokens, std::int32_t capacity) {
    const std::int64_t multiprocessors = detail::current_device_multiprocessors(82);
    const auto tiles = [&](int tile_tokens, int tile_blocks) {
        return static_cast<std::int64_t>((tokens + tile_tokens - 1) / tile_tokens) *
               ((capacity + tile_blocks - 1) / tile_blocks);
    };
    if (tensor_scoring_here()) {
        if (tokens < kTensorMinTokens) { return ScoreRoute::Scalar; }
        return 2 * tiles(kTensorTokens, kTensorBlocks) >= multiprocessors ? ScoreRoute::Tensor
                                                                          : ScoreRoute::Narrow;
    }
    if (tokens < kTileMinTokens) { return ScoreRoute::Scalar; }
    return tiles(kWideTokens, kTileBlocks) >= multiprocessors ? ScoreRoute::Wide
                                                              : ScoreRoute::Narrow;
}

QsaRope make_rope(const RopeYarn& yarn) {
    require(!(yarn.interpolation_factor > 1.0F), "linear position interpolation is not supported");
    QsaRope rope{};
    if (yarn.factor > 1.0F) {
        const RopeYarnFrequencies f =
            rope_yarn_frequencies(2 * kRotaryPairs, kRotaryPairs, 1.0e7F, yarn);
        for (int i = 0; i < kRotaryPairs; ++i) rope.inverse_frequency[i] = f.inverse_frequency[i];
        rope.attention_factor = f.attention_factor;
        rope.scaled           = 1;
    }
    return rope;
}

} // namespace

void qsa_indexer_append(const Tensor& projection, const Tensor& first_position,
                        const QsaIndexerWeights& weights, float eps, const RopeYarn& yarn,
                        Tensor& pooled, Tensor& tail, cudaStream_t stream) {
    require_projection(projection);
    const int* first = require_position(first_position);
    require_norm(weights.key_norm, "key_norm must be BF16 [128]");
    require(eps > 0.0f, "eps must be positive");
    require_pooled(pooled);
    require(tail.dtype == DType::FP32 && tail.is_contiguous() && tail.data != nullptr &&
                tail.ne[0] == kHeadDim && tail.ne[1] == kBlock - 1,
            "tail must be contiguous FP32 [128, 3]");
    const QsaRope rope        = make_rope(yarn);
    const std::int32_t tokens = projection.ne[1];
    // A call of `tokens` positions completes at most tokens / 4 + 1 blocks, wherever it starts.
    auto* const pool_kernel = rope.scaled ? qsa_pool_kernel<true> : qsa_pool_kernel<false>;
    pool_kernel<<<tokens / kBlock + 1, kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projection.data), first, tokens,
        static_cast<const __nv_bfloat16*>(weights.key_norm->data), eps,
        static_cast<const float*>(tail.data), pooled.ne[1], static_cast<float*>(pooled.data),
        rope);
    CUDA_CHECK(cudaGetLastError());
    qsa_tail_kernel<<<1, kHeadDim, 0, stream>>>(static_cast<const __nv_bfloat16*>(projection.data),
                                                first, tokens, static_cast<float*>(tail.data));
    CUDA_CHECK(cudaGetLastError());
}

std::size_t qsa_indexer_select_workspace_bytes(std::int32_t tokens, std::int32_t capacity) {
    require(tokens > 0 && capacity > 0, "tokens and capacity must be positive");
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::I32, {capacity, tokens});
    // The prepared queries of a tiled route, on any device (kTensorMinTokens > kTileMinTokens).
    if (tokens >= kTileMinTokens && capacity > kTopBlocks) {
        (void)layout.alloc(DType::FP32, {kHeads * kHeadDim, tokens});
    }
    return layout.peak_bytes(1);
}

void qsa_indexer_select(const Tensor& projection, const Tensor& first_position,
                        const QsaIndexerWeights& weights, float eps, const RopeYarn& yarn,
                        const Tensor& pooled, WorkspaceArena& workspace, Tensor& selected,
                        Tensor& counts, cudaStream_t stream) {
    require_projection(projection);
    const int* first = require_position(first_position);
    require_norm(weights.query_norm, "query_norm must be BF16 [128]");
    require(eps > 0.0f, "eps must be positive");
    require_pooled(pooled);
    const QsaRope rope = make_rope(yarn);
    const std::int32_t tokens = projection.ne[1];
    require(selected.dtype == DType::I32 && selected.is_contiguous() && selected.data != nullptr &&
                selected.ne[0] == kTopBlocks && selected.ne[1] == tokens,
            "selected must be contiguous I32 [512, tokens]");
    require(counts.dtype == DType::I32 && counts.is_contiguous() && counts.data != nullptr &&
                counts.ne[0] == tokens,
            "counts must be contiguous I32 [tokens]");
    const std::int32_t capacity = pooled.ne[1];
    // The tiles' blocks run along the grid's y dimension, at most 65,535 tiles of 64.
    require(capacity <= 65535 * kTileBlocks, "pooled capacity must be at most 4,194,240 blocks");
    auto scope                  = workspace.scope();
    Tensor keys                 = workspace.alloc(DType::I32, {capacity, tokens});
    if (capacity > kTopBlocks) {
        const ScoreRoute route = score_route(tokens, capacity);
        if (route == ScoreRoute::Scalar) {
            const dim3 grid(static_cast<unsigned>((capacity + kScoreBlocks - 1) / kScoreBlocks),
                            static_cast<unsigned>(tokens));
            auto* const score_kernel =
                rope.scaled ? qsa_score_kernel<true> : qsa_score_kernel<false>;
            score_kernel<<<grid, kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(projection.data), first,
                static_cast<const __nv_bfloat16*>(weights.query_norm->data), eps,
                static_cast<const float*>(pooled.data), static_cast<unsigned*>(keys.data), capacity,
                rope);
        } else {
            Tensor queries = workspace.alloc(DType::FP32, {kHeads * kHeadDim, tokens});
            auto* const query_kernel =
                rope.scaled ? qsa_query_kernel<true> : qsa_query_kernel<false>;
            query_kernel<<<tokens, kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(projection.data), first,
                static_cast<const __nv_bfloat16*>(weights.query_norm->data), eps,
                static_cast<float*>(queries.data), rope);
            CUDA_CHECK(cudaGetLastError());
            const int tile_tokens = route == ScoreRoute::Narrow ? kNarrowTokens : kWideTokens;
            const int tile_blocks = route == ScoreRoute::Tensor ? kTensorBlocks : kTileBlocks;
            const dim3 grid(static_cast<unsigned>((tokens + tile_tokens - 1) / tile_tokens),
                            static_cast<unsigned>((capacity + tile_blocks - 1) / tile_blocks));
            auto* const score_kernel = route == ScoreRoute::Tensor ? qsa_tensor_score_kernel
                                       : route == ScoreRoute::Wide
                                           ? qsa_tile_score_kernel<kWideTokens>
                                           : qsa_tile_score_kernel<kNarrowTokens>;
            score_kernel<<<grid, kTileThreads, 0, stream>>>(
                static_cast<const float*>(queries.data), first,
                static_cast<const float*>(pooled.data), static_cast<unsigned*>(keys.data), tokens,
                capacity);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    qsa_select_kernel<<<tokens, kThreads, 0, stream>>>(
        first, static_cast<const unsigned*>(keys.data), capacity, static_cast<int*>(selected.data),
        static_cast<int*>(counts.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
