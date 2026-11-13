#pragma once

// BF16-cache causal prompt kernel. INT8 has an independent kernel body and resource
// policy in causal_attention_prompt_i8.cuh.
//
//   * Br = 64 query rows and Bc = 64 key columns per CTA tile.
//   * 4 warps / 128 threads; each warp owns 16 query rows of the tile.
//   * Q, K, V staged in 96 KiB of dynamic shared memory (single-buffered), with
//     the cp.async of the next K/V tile overlapped against the current
//     QK / PV tensor-core work (exactly FA's single-buffer overlap pattern).
//   * m16n8k16 bf16 MMA for both S = Q Kᵀ and O += P V, online softmax in exp2.
//
// The op first writes the new chunk K/V into absolute positions in the paged cache,
// then computes causal attention for
// every chunk token over all cached history using bottom-right causal alignment
// (query row i attends to keys [0, base_pos + i]).

#include <math_constants.h>

#include <type_traits>

#include "ops/common/device_tier.h"
#include "ops/kv_cache/bf16p_codec.cuh"
#include "ops/softmax_attention/common/bf16_attention.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"

namespace ninfer::ops {

// Stage one [Bc, D] K or V tile from the per-kv-head contiguous cache into the
// swizzled smem buffer. Keys beyond max_query_abs (which the causal mask always
// drops) are zeroed so the padded/uninitialized cache tail never feeds NaNs into
// the tensor cores. Mirrors FA's predicated K/V cp.async + Clear_OOB path.
template <typename Geometry>
__device__ __forceinline__ void
causal_prompt_stage_kv(__nv_bfloat16* dst, const __nv_bfloat16* cache, int kv_head, int k0,
                       int max_query_abs, int physical_page, int tid) {
    constexpr int D         = kCausalPromptHeadDim;
    constexpr int Bc        = kCausalPromptBc;
    constexpr int Threads   = kCausalPromptThreads;
    constexpr int VecPerRow = D / 8; // 8 bf16 per 16B cp.async
    const bool full_tile    = (k0 + Bc - 1) <= max_query_abs;
    // Block base pointer computed once (int64); per-element offsets stay 32-bit.
    const __nv_bfloat16* cache_block =
        cache + paged_kv_element_offset<kCausalPromptHeadDim>(
                    physical_page, kv_head, k0 & kPagedKVPageMask, 0,
                    causal_prompt_q_heads<Geometry>() / Geometry::GroupSize);
    if (full_tile) {
#pragma unroll
        for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
            const int key_l  = chunk >> 5;        // / VecPerRow (32)
            const int d      = (chunk & 31) << 3; // (chunk % 32) * 8
            __nv_bfloat16* p = &dst[key_l * D + causal_prompt_swz(key_l, d)];
            cp_async<16, Cache::cg>(p, &cache_block[key_l * D + d]);
        }
    } else {
#pragma unroll
        for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
            const int key_l  = chunk >> 5;        // / VecPerRow (32)
            const int d      = (chunk & 31) << 3; // (chunk % 32) * 8
            __nv_bfloat16* p = &dst[key_l * D + causal_prompt_swz(key_l, d)];
            if ((k0 + key_l) <= max_query_abs) {
                cp_async<16, Cache::cg>(p, &cache_block[key_l * D + d]);
            } else {
                store_vec(p, make_int4(0, 0, 0, 0));
            }
        }
    }
}

// A packed cache's K or V plane (KvCacheStorage::Bf16Packed): its rows and their side records.
struct CausalPromptPackedPlane {
    const std::uint8_t* rows;
    const std::uint8_t* side;
};

// A packed tile in shared memory: Bc rows of kBf16pRowBytes, then their Bc side records. One
// 64-key tile lies in one page, where a head's rows and side records are each contiguous.
inline constexpr int kCausalPromptPackedRowsBytes = kCausalPromptBc * kBf16pRowBytes;
inline constexpr int kCausalPromptPackedBytes =
    kCausalPromptPackedRowsBytes + kCausalPromptBc * kBf16pSideBytes;
static_assert(kCausalPromptPackedBytes <= kCausalPromptBc * kCausalPromptHeadDim * 2);

// Starts the cp.async copies of the packed tile at key k0 (a multiple of Bc) into `raw`; the
// caller commits and waits. Rows past the visible keys are copied too: they lie in the same
// page, and the decode zeroes them.
template <typename Geometry>
__device__ __forceinline__ void causal_prompt_stage_packed(std::uint8_t* raw,
                                                           CausalPromptPackedPlane plane,
                                                           int kv_head, int physical_page,
                                                           int tid) {
    constexpr int RowChunks  = kCausalPromptPackedRowsBytes / 16;
    constexpr int SideChunks = kCausalPromptBc * kBf16pSideBytes / 16;
    const int kv_heads       = causal_prompt_q_heads<Geometry>() / Geometry::GroupSize;
    const std::uint8_t* rows = plane.rows + bf16p_row_offset(physical_page, kv_head, 0, kv_heads);
    const std::uint8_t* side = plane.side + bf16p_side_offset(physical_page, kv_head, 0, kv_heads);
#pragma unroll
    for (int chunk = tid; chunk < RowChunks + SideChunks; chunk += kCausalPromptThreads) {
        if (chunk < RowChunks) {
            cp_async<16, Cache::cg>(raw + 16 * chunk, rows + 16 * chunk);
        } else {
            cp_async<16, Cache::cg>(raw + 16 * chunk, side + 16 * (chunk - RowChunks));
        }
    }
}

// Decodes the packed tile at key k0 from `raw` into the swizzled BF16 tile `dst`, zeroing keys
// past max_query_abs. A warp decodes one row at a time, lane L its values 8 L .. 8 L + 7.
__device__ __forceinline__ void causal_prompt_decode_packed(__nv_bfloat16* dst,
                                                            const std::uint8_t* raw, int k0,
                                                            int max_query_abs, int tid) {
    constexpr int D       = kCausalPromptHeadDim;
    constexpr int Bc      = kCausalPromptBc;
    constexpr int Threads = kCausalPromptThreads;
    static_assert(kBf16pRowValues == D && D / 8 == 32);
    const int lane = tid & 31;
#pragma unroll 4
    for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
        const int key_l  = chunk >> 5; // uniform in a warp
        __nv_bfloat16* p = &dst[key_l * D + causal_prompt_swz(key_l, lane << 3)];
        if ((k0 + key_l) <= max_query_abs) {
            const std::uint8_t* row = raw + key_l * kBf16pRowBytes;
            const uint2 code        = *reinterpret_cast<const uint2*>(row + 8 * lane);
            const unsigned nib =
                *reinterpret_cast<const unsigned*>(row + kBf16pNibbles + 4 * lane);
            const uint2 record = *reinterpret_cast<const uint2*>(
                raw + kCausalPromptPackedRowsBytes + key_l * kBf16pSideBytes);
            store_vec(p, bf16p_decode_regs(code, nib, record, lane));
        } else {
            store_vec(p, make_int4(0, 0, 0, 0));
        }
    }
}

// causal_prompt_stage_kv over packed rows, for the Turing kernel's exact sweep
// (bf16/prompt_sm75.cuh): each thread decodes its chunks to the BF16 values the bf16 storage would
// hold, with plain stores (the caller's barrier publishes them).
template <typename Geometry>
__device__ __forceinline__ void causal_prompt_stage_kv(__nv_bfloat16* dst,
                                                       CausalPromptPackedPlane plane, int kv_head,
                                                       int k0, int max_query_abs,
                                                       int physical_page, int tid) {
    constexpr int D         = kCausalPromptHeadDim;
    constexpr int Bc        = kCausalPromptBc;
    constexpr int Threads   = kCausalPromptThreads;
    constexpr int VecPerRow = D / 8;
    static_assert(kBf16pRowValues == D);
    const int kv_heads = causal_prompt_q_heads<Geometry>() / Geometry::GroupSize;
    const int offset0  = k0 & kPagedKVPageMask;
#pragma unroll 4
    for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
        const int key_l  = chunk >> 5;
        const int c      = chunk & 31;
        __nv_bfloat16* p = &dst[key_l * D + causal_prompt_swz(key_l, c << 3)];
        if ((k0 + key_l) <= max_query_abs) {
            const int offset = offset0 + key_l;
            const std::uint8_t* row =
                plane.rows + bf16p_row_offset(physical_page, kv_head, offset, kv_heads);
            const std::uint8_t* side =
                plane.side + bf16p_side_offset(physical_page, kv_head, offset, kv_heads);
            store_vec(p, bf16p_decode8(row, side, c));
        } else {
            store_vec(p, make_int4(0, 0, 0, 0));
        }
    }
}

// The tile math of the BF16 prompt kernels: this file's kernel and a Turing build's
// (bf16/prompt_sm75.cuh) stage their tiles differently and run the same steps on them. `mma`
// multiplies one m16n8k16 fragment pair into FP32 accumulators.

// One warp's ldmatrix bases over the Q tile and a K and a V tile (causal_prompt_swz_addr); the warp
// owns query rows [warp_row0, warp_row0 + 16).
struct CausalPromptWarpTiles {
    unsigned q_base, q_as, q_r;
    unsigned k_base, k_as, k_r;
    unsigned v_base, v_as, v_r;
};

__device__ __forceinline__ CausalPromptWarpTiles causal_prompt_warp_tiles(unsigned q_sbase,
                                                                          unsigned k_sbase,
                                                                          unsigned v_sbase,
                                                                          int warp_row0, int lane) {
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;
    CausalPromptWarpTiles tiles;
    // Q A-fragment: row = warp_row0 + a_rowoff, col = k*16 + a_coloff.
    tiles.q_base = q_sbase + static_cast<unsigned>((warp_row0 + a_rowoff) * 512);
    tiles.q_as   = static_cast<unsigned>((a_mat >> 1) << 4);
    tiles.q_r    = static_cast<unsigned>(a_rin << 4);
    // K B-fragment via ldmatrix.x4 (2 n-tiles/instr): lanes 16-31 fetch the +8-key
    // half (extra 4096 bytes), lanes with bit3 set fetch the +8 d-contract half.
    tiles.k_base =
        k_sbase + static_cast<unsigned>(b_rin * 512) + (static_cast<unsigned>(lane >> 4) << 12);
    tiles.k_as = static_cast<unsigned>((b_koff >> 3) << 4);
    tiles.k_r  = static_cast<unsigned>(b_rin << 4);
    // V B-fragment via ldmatrix.x4.trans (2 n-tiles/instr): row = k*16 + (bit3)*8 + b_rin,
    // col = n*8 + (lane>>4)*8.
    tiles.v_base = v_sbase + static_cast<unsigned>(((lane >> 3) & 1) * 4096) +
                   static_cast<unsigned>(b_rin * 512);
    tiles.v_as = static_cast<unsigned>((lane >> 4) << 4);
    tiles.v_r  = static_cast<unsigned>(b_rin << 4);
    return tiles;
}

inline constexpr int kCausalPromptQKNt = kCausalPromptBc / 8;      // QK score n-tiles
inline constexpr int kCausalPromptPVNt = kCausalPromptHeadDim / 8; // PV output n-tiles
inline constexpr int kCausalPromptPVKs = kCausalPromptBc / 16;     // PV contraction steps

// S = Q Kᵀ for the warp's 16 rows over all Bc keys, in registers. Software-pipelined like cute's
// gemm: the ldmatrix for contraction step k+1 is issued while the m16n8k16 MMAs for step k run,
// so the LSU (ldmatrix) and tensor pipes overlap instead of stalling on each other.
template <class Mma>
__device__ __forceinline__ void causal_prompt_qk(float (&score)[kCausalPromptQKNt][4],
                                                 const CausalPromptWarpTiles& tiles, Mma&& mma) {
    constexpr int QKNt = kCausalPromptQKNt;
    constexpr int QKKs = kCausalPromptHeadDim / 16; // QK contraction steps over head_dim
#pragma unroll
    for (int nt = 0; nt < QKNt; ++nt) {
        score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
    }
    // Swizzled ldmatrix addresses via precomputed per-lane bases + immediates.
    unsigned af[2][4];
    unsigned bf[2][QKNt][2];
    ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                causal_prompt_swz_addr(tiles.q_base, 0u, tiles.q_as, tiles.q_r));
#pragma unroll
    for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
        ldmatrix_x4(bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                    causal_prompt_swz_addr(tiles.k_base + static_cast<unsigned>(nt2 * 4096), 0u,
                                           tiles.k_as, tiles.k_r));
    }
#pragma unroll
    for (int k = 0; k < QKKs; ++k) {
        const int cur = k & 1;
        const int nxt = cur ^ 1;
        if (k + 1 < QKKs) {
            const unsigned ck = static_cast<unsigned>((k + 1) << 5);
            ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                        causal_prompt_swz_addr(tiles.q_base, ck, tiles.q_as, tiles.q_r));
#pragma unroll
            for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                ldmatrix_x4(bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0],
                            bf[nxt][nt2 + 1][1],
                            causal_prompt_swz_addr(tiles.k_base + static_cast<unsigned>(nt2 * 4096),
                                                   ck, tiles.k_as, tiles.k_r));
            }
        }
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            mma(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0], af[cur][1],
                af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
        }
    }
}

// The online-softmax step of key tile k0 for the warp's rows: the causal mask, the block row max
// on raw scores (the scale folds into exp2), P = exp2(S - m) as the PV A fragment
// (pack_attention_p) with the lane's row sums, and the running max, sum and accumulators rescaled.
// The row-sum allreduce is deferred to the epilogue; only the row max is reduced per tile.
__device__ __forceinline__ void causal_prompt_softmax(float (&score)[kCausalPromptQKNt][4],
                                                      unsigned (&p_frag)[kCausalPromptPVKs][4],
                                                      float (&acc)[kCausalPromptPVNt][4], float& m0,
                                                      float& m1, float& l0, float& l1, int q0,
                                                      int k0, int tokens, int base_pos,
                                                      float scale_l2, int warp_row0, int lane) {
    constexpr int Br            = kCausalPromptBr;
    constexpr int Bc            = kCausalPromptBc;
    constexpr int QKNt          = kCausalPromptQKNt;
    constexpr int PVNt          = kCausalPromptPVNt;
    constexpr unsigned FullMask = 0xffffffffu;
    const int gid               = lane >> 2;
    const int lid               = lane & 3;
    const int row0              = warp_row0 + gid;
    const int row1              = warp_row0 + gid + 8;
    const int qrow0             = q0 + row0;
    const int qrow1             = q0 + row1;
    const int qabs0             = (qrow0 < tokens) ? base_pos + qrow0 : -1;
    const int qabs1             = (qrow1 < tokens) ? base_pos + qrow1 : -1;
    const bool full_score_tile  = (q0 + Br <= tokens) && ((k0 + Bc - 1) <= (base_pos + q0));

    // block row-max on raw (unscaled) scores; scale is folded into exp2 below
    float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
    if (full_score_tile) {
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
    } else {
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int key0 = k0 + nt * 8 + 2 * lid;
            const int key1 = key0 + 1;
            score[nt][0]   = (qrow0 < tokens && key0 <= qabs0) ? score[nt][0] : -CUDART_INF_F;
            score[nt][1]   = (qrow0 < tokens && key1 <= qabs0) ? score[nt][1] : -CUDART_INF_F;
            score[nt][2]   = (qrow1 < tokens && key0 <= qabs1) ? score[nt][2] : -CUDART_INF_F;
            score[nt][3]   = (qrow1 < tokens && key1 <= qabs1) ? score[nt][3] : -CUDART_INF_F;
            bm0            = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1            = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
    }
    bm0 = warp_max<4>(bm0, FullMask);
    bm1 = warp_max<4>(bm1, FullMask);

    const float nm0        = fmaxf(m0, bm0);
    const float nm1        = fmaxf(m1, bm1);
    const float nm0_scaled = nm0 * scale_l2;
    const float nm1_scaled = nm1 * scale_l2;
    const float alpha0     = softmax_history_rescale(m0, scale_l2, nm0_scaled);
    const float alpha1     = softmax_history_rescale(m1, scale_l2, nm1_scaled);

    // P = exp2(S - m), repacked into the PV A-fragment layout, plus local block row-sum.
    float bl0 = 0.0f, bl1 = 0.0f;
    if (full_score_tile) {
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const float p00 = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled));
            const float p01 = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled));
            const float p10 = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled));
            const float p11 = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled));
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            const int pk = nt >> 1;
            if ((nt & 1) == 0) {
                p_frag[pk][0] = pack_attention_p(p00, p01);
                p_frag[pk][1] = pack_attention_p(p10, p11);
            } else {
                p_frag[pk][2] = pack_attention_p(p00, p01);
                p_frag[pk][3] = pack_attention_p(p10, p11);
            }
        }
    } else {
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const float p00 = (score[nt][0] > -CUDART_INF_F)
                                  ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                                  : 0.0f;
            const float p01 = (score[nt][1] > -CUDART_INF_F)
                                  ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                                  : 0.0f;
            const float p10 = (score[nt][2] > -CUDART_INF_F)
                                  ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                                  : 0.0f;
            const float p11 = (score[nt][3] > -CUDART_INF_F)
                                  ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                                  : 0.0f;
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            const int pk = nt >> 1;
            if ((nt & 1) == 0) {
                p_frag[pk][0] = pack_attention_p(p00, p01);
                p_frag[pk][1] = pack_attention_p(p10, p11);
            } else {
                p_frag[pk][2] = pack_attention_p(p00, p01);
                p_frag[pk][3] = pack_attention_p(p10, p11);
            }
        }
    }

    l0 = __fmaf_rn(l0, alpha0, bl0);
    l1 = __fmaf_rn(l1, alpha1, bl1);
    m0 = nm0;
    m1 = nm1;
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        acc[n][0] *= alpha0;
        acc[n][1] *= alpha0;
        acc[n][2] *= alpha1;
        acc[n][3] *= alpha1;
    }
}

// O += P V, contracting over the Bc keys. The (k, n) iteration space is flattened and
// software-pipelined: the transposed ldmatrix for the next V fragment is issued while the current
// MMA runs. Each x4.trans load covers 2 output n-tiles (16 dims); the next load is pipelined
// against the current pair of MMAs.
template <class Mma>
__device__ __forceinline__ void causal_prompt_pv(float (&acc)[kCausalPromptPVNt][4],
                                                 const unsigned (&p_frag)[kCausalPromptPVKs][4],
                                                 const CausalPromptWarpTiles& tiles, Mma&& mma) {
    constexpr int PVHalf  = kCausalPromptPVNt / 2;      // 16 n-tile pairs
    constexpr int PVLoads = kCausalPromptPVKs * PVHalf; // 64 x4.trans loads
    // Swizzled V x4.trans addresses via precomputed per-lane base + immediates.
    unsigned vf[2][4];
    ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                  causal_prompt_swz_addr(tiles.v_base, 0u, tiles.v_as, tiles.v_r));
#pragma unroll
    for (int li = 0; li < PVLoads; ++li) {
        const int k   = li / PVHalf;
        const int n2  = (li % PVHalf) * 2;
        const int cur = li & 1;
        const int nxt = cur ^ 1;
        if (li + 1 < PVLoads) {
            const int k2       = (li + 1) / PVHalf;
            const int n2b      = ((li + 1) % PVHalf) * 2;
            const unsigned ckv = static_cast<unsigned>(n2b << 4);
            ldmatrix_x4_t(vf[nxt][0], vf[nxt][1], vf[nxt][2], vf[nxt][3],
                          causal_prompt_swz_addr(tiles.v_base + static_cast<unsigned>(k2 * 8192),
                                                 ckv, tiles.v_as, tiles.v_r));
        }
        mma(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_frag[k][0], p_frag[k][1],
            p_frag[k][2], p_frag[k][3], vf[cur][0], vf[cur][1]);
        mma(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3], p_frag[k][0],
            p_frag[k][1], p_frag[k][2], p_frag[k][3], vf[cur][2], vf[cur][3]);
    }
}

// The rows' outputs: the row sums reduced over the quad, normalized once per row via
// reciprocal-multiply instead of 128 IEEE divides, stored as BF16, and the rows past the valid
// tokens up to `width` zeroed.
template <typename Geometry>
__device__ __forceinline__ void causal_prompt_store(float (&acc)[kCausalPromptPVNt][4], float l0,
                                                    float l1, __nv_bfloat16* __restrict__ out,
                                                    int q_head, int q0, int tokens, int width,
                                                    int warp_row0, int lane, int tid) {
    constexpr unsigned FullMask = 0xffffffffu;
    const int q_heads           = causal_prompt_q_heads<Geometry>();
    const int gid               = lane >> 2;
    const int lid               = lane & 3;
    l0                          = warp_sum<4>(l0, FullMask);
    l1                          = warp_sum<4>(l1, FullMask);
    const float inv_l0          = (l0 > 0.0f) ? __frcp_rn(l0) : 0.0f;
    const float inv_l1          = (l1 > 0.0f) ? __frcp_rn(l1) : 0.0f;
#pragma unroll
    for (int n = 0; n < kCausalPromptPVNt; ++n) {
        const int d0    = n * 8 + 2 * lid;
        const int qrow0 = q0 + warp_row0 + gid;
        const int qrow1 = q0 + warp_row0 + gid + 8;
        if (qrow0 < tokens) {
            *reinterpret_cast<unsigned*>(
                &out[causal_prompt_q_index<Geometry>(q_head, d0, qrow0, q_heads)]) =
                pack_bf16x2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
        }
        if (qrow1 < tokens) {
            *reinterpret_cast<unsigned*>(
                &out[causal_prompt_q_index<Geometry>(q_head, d0, qrow1, q_heads)]) =
                pack_bf16x2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
        }
    }
    causal_prompt_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + kCausalPromptBr, width),
                                             tid, kCausalPromptThreads, q_heads);
}

inline constexpr int kCausalPromptBf16SmemBytes = kCausalPromptSmemBytes;

// FlashAttention-2 forward, one CTA per (query 64-row block, query head). Grid is
// (ceil(tokens/64), q_heads). seqlen_q = tokens, seqlen_k = base_pos + tokens, with
// bottom-right causal alignment (query row i sees keys [0, base_pos + i]).
// Plane: a BF16 cache's `const __nv_bfloat16*` pages or a packed cache's CausalPromptPackedPlane.
template <typename Geometry, typename Metadata, typename Plane = const __nv_bfloat16*>
__launch_bounds__(kCausalPromptThreads, 1) __global__
    void causal_attention_prompt_bf16_kernel(const __nv_bfloat16* __restrict__ q, Plane cache_k,
                                             Plane cache_v, Metadata metadata,
                                             const std::int32_t* __restrict__ positions,
                                             float scale, __nv_bfloat16* __restrict__ out,
                                             std::int32_t width) {
    if constexpr (!shape_set_in_image(false)) { return; } // not this image's set
    constexpr int D       = kCausalPromptHeadDim; // 256
    constexpr int Br      = kCausalPromptBr;      // 64 query rows
    constexpr int Bc      = kCausalPromptBc;      // 64 key cols
    constexpr int Threads = kCausalPromptThreads; // 128
    constexpr float Log2E = 1.4426950408889634074f;

    static_assert(Threads == 128);

    extern __shared__ __align__(16) __nv_bfloat16 causal_smem[];
    __nv_bfloat16* q_s = causal_smem;  // [Br, D] swizzled
    __nv_bfloat16* k_s = q_s + Br * D; // [Bc, D] swizzled
    __nv_bfloat16* v_s = k_s + Bc * D; // [Bc, D] swizzled

    const CausalPromptBlock block = causal_prompt_block_longest_first();

    const int q_block = block.row_block;
    const int q_head  = block.y;
    const int q_heads = causal_prompt_q_heads<Geometry>();
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);

    if (q_head >= q_heads || q0 >= width) { return; }
    if (q0 >= tokens) {
        causal_prompt_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid, Threads,
                                                 q_heads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();
    const int warp_row0             = warp * 16; // this warp owns rows [warp_row0, warp_row0+16)
    const CausalPromptWarpTiles tiles =
        causal_prompt_warp_tiles(smem_addr(q_s), smem_addr(k_s), smem_addr(v_s), warp_row0, lane);
    const auto bf16_mma = [](float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1,
                             unsigned a2, unsigned a3, unsigned b0,
                             unsigned b1) { mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1); };

    // Stage Q into smem once via cp.async (overlaps with the K(0) prologue load
    // below); it stays resident for the whole key loop. Global Q rows are 256 bf16
    // contiguous, with a token stride of 256 * q_heads.
    {
        constexpr int VecPerRow      = D / 8;
        const int q_row_stride       = D * q_heads; // global stride between tokens
        const __nv_bfloat16* q_block = q + causal_prompt_q_index<Geometry>(q_head, 0, q0, q_heads);
        if (q0 + Br <= tokens) {
#    pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk >> 5;
                const int d      = (chunk & 31) << 3;
                __nv_bfloat16* p = &q_s[row * D + causal_prompt_swz(row, d)];
                cp_async<16, Cache::cg>(p, &q_block[row * q_row_stride + d]);
            }
        } else {
#    pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk >> 5;
                const int d      = (chunk & 31) << 3;
                __nv_bfloat16* p = &q_s[row * D + causal_prompt_swz(row, d)];
                if (q0 + row < tokens) {
                    cp_async<16, Cache::cg>(p, &q_block[row * q_row_stride + d]);
                } else {
                    store_vec(p, make_int4(0, 0, 0, 0));
                }
            }
        }
    }

    float acc[kCausalPromptPVNt][4];
#    pragma unroll
    for (int n = 0; n < kCausalPromptPVNt; ++n) {
#    pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int n_block_max   = (max_query_abs / Bc) + 1; // n_block_min == 0

    // Fold softmax_scale into the exp2 (FA-style): scores stay raw, so the
    // per-element "* scale" multiply drops out of the QK epilogue entirely.
    const float scale_l2 = scale * Log2E;
    int physical_page    = block_table[0];

    if constexpr (std::is_same_v<Plane, CausalPromptPackedPlane>) {
        // A packed cache: cp.async brings each tile's packed bytes into v_s while the MMAs run on
        // k_s, and the decode turns them into k_s's BF16 tile between the MMAs. k_s holds K for
        // the QK product, then V for the PV product.
        auto* raw = reinterpret_cast<std::uint8_t*>(v_s);
        const CausalPromptWarpTiles packed_tiles =
            causal_prompt_warp_tiles(smem_addr(q_s), smem_addr(k_s), smem_addr(k_s), warp_row0,
                                     lane);
        ninfer::ops::cp_commit();
        causal_prompt_stage_packed<Geometry>(raw, cache_k, kv_head, physical_page, tid);
        ninfer::ops::cp_commit();
        ninfer::ops::cp_wait<0>(); // Q and raw K(0)
        __syncthreads();
        causal_prompt_decode_packed(k_s, raw, 0, max_query_abs, tid);
        __syncthreads();
        for (int kb = 0; kb < n_block_max; ++kb) {
            const int k0 = kb * Bc;
            causal_prompt_stage_packed<Geometry>(raw, cache_v, kv_head, physical_page, tid);
            ninfer::ops::cp_commit();

            float score[kCausalPromptQKNt][4];
            causal_prompt_qk(score, packed_tiles, bf16_mma);
            unsigned p_frag[kCausalPromptPVKs][4];
            causal_prompt_softmax(score, p_frag, acc, m0, m1, l0, l1, q0, k0, tokens, base_pos,
                                  scale_l2, warp_row0, lane);

            ninfer::ops::cp_wait<0>(); // raw V(kb) landed
            __syncthreads();           // and the QK product is done with k_s
            causal_prompt_decode_packed(k_s, raw, k0, max_query_abs, tid);
            __syncthreads();
            if (kb + 1 < n_block_max) {
                physical_page = block_table[kb + 1];
                causal_prompt_stage_packed<Geometry>(raw, cache_k, kv_head, physical_page, tid);
                ninfer::ops::cp_commit();
            }

            causal_prompt_pv(acc, p_frag, packed_tiles, bf16_mma);

            if (kb + 1 < n_block_max) {
                ninfer::ops::cp_wait<0>(); // raw K(kb+1) landed
                __syncthreads();           // and the PV product is done with k_s
                causal_prompt_decode_packed(k_s, raw, (kb + 1) * Bc, max_query_abs, tid);
                __syncthreads();
            }
        }
    } else {
        // Prologue: commit Q, then kick off K(0). The loop's wait<0> below drains both.
        ninfer::ops::cp_commit();
        causal_prompt_stage_kv<Geometry>(k_s, cache_k, kv_head, 0, max_query_abs, physical_page,
                                         tid);
        ninfer::ops::cp_commit();

        for (int kb = 0; kb < n_block_max; ++kb) {
            const int k0                 = kb * Bc;
            const int next_physical_page =
                (kb + 1 < n_block_max) ? block_table[kb + 1] : physical_page;

            ninfer::ops::cp_wait<0>(); // K(kb) landed (also publishes q_s / prev PV done)
            __syncthreads();

            // Overlap V(kb) load against the QK MMA below.
            causal_prompt_stage_kv<Geometry>(v_s, cache_v, kv_head, k0, max_query_abs,
                                             physical_page, tid);
            ninfer::ops::cp_commit();

            float score[kCausalPromptQKNt][4];
            causal_prompt_qk(score, tiles, bf16_mma);
            unsigned p_frag[kCausalPromptPVKs][4];
            causal_prompt_softmax(score, p_frag, acc, m0, m1, l0, l1, q0, k0, tokens, base_pos,
                                  scale_l2, warp_row0, lane);

            ninfer::ops::cp_wait<0>(); // V(kb) landed; QK done reading k_s
            __syncthreads();

            // Prefetch K(kb+1) into the (now-free) K buffer, overlapping the PV MMA.
            if (kb + 1 < n_block_max) {
                physical_page = next_physical_page;
                causal_prompt_stage_kv<Geometry>(k_s, cache_k, kv_head, (kb + 1) * Bc,
                                                 max_query_abs, physical_page, tid);
                ninfer::ops::cp_commit();
            }

            causal_prompt_pv(acc, p_frag, tiles, bf16_mma);
        }
    }

    causal_prompt_store<Geometry>(acc, l0, l1, out, q_head, q0, tokens, width, warp_row0, lane,
                                  tid);
}

} // namespace ninfer::ops

// The Turing shape set's kernel of the same contract (ops/common/device_tier.h).
#include "ops/softmax_attention/dense/causal_cache/bf16/prompt_sm75.cuh"
