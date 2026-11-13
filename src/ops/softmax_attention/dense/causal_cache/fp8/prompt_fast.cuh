#pragma once

// Fast row-scaled E4M3-cache causal prompt kernel for the registered head geometries, selected per
// launch by CausalAttentionExecutionEnvelope::fast_prompt_kernel (ninfer-serve
// --fast-prefill-kernel); the default route keeps the kernel in fp8/prompt.cuh.
//
//   * Each warp owns 16 query rows of one head for the whole key sweep (four warps per CTA), so
//     scores, probabilities and the D256 output accumulator never leave registers and no warp
//     waits for another between QK and PV. The only CTA barrier is the one that publishes a key
//     tile.
//   * Unlike this fork's INT8 fast kernel, which multiplies QK on the INT8 tensor cores, the
//     E4M3 codes here have no tensor-core path compiled for sm_86 or sm_89 (fp8/prompt.cuh), so
//     both planes decode in registers: K widens exactly to BF16 (the code
//     converted, one FP32 product with the key's scale, one BF16 rounding -- the standard kernel's
//     staging tile bit for bit) and feeds the BF16 Tensor Cores from two 16-bit shared-memory
//     loads per fragment register; V is read by ldmatrix.trans as byte pairs and widens exactly
//     to FP16, its key's scale applied once, the form the standard kernel's FP16 staging tile held.
//   * A key's scale is one value shared by its 256 dimensions. It is read straight into lane
//     registers at decode time (two-byte global loads that stay L2-resident) instead of staging a
//     scale plane, so a stage holds only the raw E4M3 codes of K and V: 32 KiB.
//   * PV accumulates in FP32 through every tile (no INT8 fast kernel's FP16 partials, so no
//     power-of-two rescale guard); the standard kernel's running-max softmax applies, the output
//     accumulator rescaled by the tile's alpha before the tile's PV contributes.
//   * CTAs are issued longest-first so a causal prompt's heaviest row blocks do not form the tail.
//   * A split launch (gridDim.z > 1, int8/prompt_fast_plan.h) gives each CTA a contiguous run of a
//     row block's key pages and publishes, per row, the normalized FP32 row and its (max, sum)
//     statistics; the shared fast-prompt merge kernel (prompt_fast_merge.cuh) combines the splits.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/common/device_limits.h"
#include "ops/common/mma.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_fast_merge.cuh"

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kCausalPromptFp8FastBc        = 64;
inline constexpr int kCausalPromptFp8FastTileBytes = kCausalPromptFp8FastBc * kKVCacheFp8HeadDim;

static_assert(kCausalPromptFp8FastBc == kPagedKVPageSize);

// One pipeline stage: the raw E4M3 K codes, then the raw V codes. The per-key scales decode to
// lane registers (file comment), so a stage holds no scale plane: 32 KiB.
struct CausalPromptFp8FastStage {
    static constexpr int TileBytes = kCausalPromptFp8FastTileBytes;
    static constexpr int VOffset   = TileBytes;
    static constexpr int Bytes     = 2 * TileBytes;
};

// K/V tiles are double-buffered so the next tile's cp.async overlaps this tile's math, where two
// stages beside the four-warp Q tile fit a block. Turing's 64 KiB hold one stage at the exact
// block limit, and its copies are synchronous, so a second stage would overlap nothing there.
template <bool TuringShapes>
inline constexpr int kCausalPromptFp8FastStages =
    4 * 16 * kKVCacheFp8HeadDim * 2 + 2 * CausalPromptFp8FastStage::Bytes <=
            shape_max_shared_bytes_per_block(TuringShapes)
        ? 2
        : 1;

// Each warp owns 16 query rows. The fast FP8 CTA is always four warps: an eight-warp form doubles
// the BF16 Q tile to 64 KiB and fits no stage beside it in either shape set.
template <bool TuringShapes = false>
struct CausalPromptFp8FastShape {
    static constexpr int Threads   = 4 * 32;
    static constexpr int Br        = 4 * 16;
    static constexpr int QBytes    = Br * kCausalPromptHeadDim * 2;
    static constexpr int Stages    = kCausalPromptFp8FastStages<TuringShapes>;
    static constexpr int SmemBytes = QBytes + Stages * CausalPromptFp8FastStage::Bytes;
    // Whether this shape can launch in its shape set.
    static constexpr bool Launchable = SmemBytes <= shape_max_shared_bytes_per_block(TuringShapes);
};

static_assert(CausalPromptFp8FastShape<>::Stages == 2 &&
              CausalPromptFp8FastShape<>::SmemBytes == 98304);
static_assert(CausalPromptFp8FastShape<true>::Stages == 1 &&
              CausalPromptFp8FastShape<true>::SmemBytes == 65536);
static_assert(CausalPromptFp8FastShape<>::Launchable &&
              CausalPromptFp8FastShape<true>::Launchable);

// Two adjacent E4M3 codes and their key's scale to one BF16x2, widened exactly as
// kv_cache_fp8_dequant_bf16x8 does: the vendor pair conversion is exact, the scale rides the FP32
// product (the pre-sm_89 integer expansion carries code times 2^-8 exactly, so the scale absorbs
// the 2^8), and one BF16 rounding is the standard kernel's staging tile.
__device__ __forceinline__ unsigned causal_fp8_fast_kpair_bf16(std::uint16_t codes, float scale) {
    __nv_bfloat162 value;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    __nv_fp8x2_e4m3 storage;
    storage.__x = codes;
    const float2 decoded = static_cast<float2>(storage);
    value = __floats2bfloat162_rn(decoded.x * scale, decoded.y * scale);
#else
    const float2 decoded = __half22float2(kv_cache_fp8_code2_to_half2_x2m8(codes));
    value = __floats2bfloat162_rn(decoded.x * (scale * 256.0f), decoded.y * (scale * 256.0f));
#endif
    return load_vec<unsigned>(&value);
}

// One ldmatrix.trans b16 lane of a raw E4M3 [key][d] tile holds the codes
// {V[k][d], V[k][d+1], V[k+1][d], V[k+1][d+1]}. Returns the FP16 B-fragment halves
// {V[k][d], V[k+1][d]} and {V[k][d+1], V[k+1][d+1]}, each code widened exactly
// (kv_cache_fp8_dequant_code2_to_half2's form) and multiplied once by its key's scale.
__device__ __forceinline__ void causal_fp8_value_pair_fragments(unsigned codes, __half scale_lo,
                                                                __half scale_hi, unsigned& even,
                                                                unsigned& odd) {
    const __half2 lo =
        kv_cache_fp8_dequant_code2_to_half2(static_cast<std::uint16_t>(codes & 0xffffu), scale_lo);
    const __half2 hi =
        kv_cache_fp8_dequant_code2_to_half2(static_cast<std::uint16_t>(codes >> 16), scale_hi);
    const __half2 e = __halves2half2(lo.x, hi.x);
    const __half2 o = __halves2half2(lo.y, hi.y);
    even = load_vec<unsigned>(&e);
    odd  = load_vec<unsigned>(&o);
}

template <typename Geometry, typename Metadata, bool TuringShapes, bool Split>
__global__ __launch_bounds__(CausalPromptFp8FastShape<TuringShapes>::Threads, 1)
void causal_attention_prompt_fp8_fast_kernel(const __nv_bfloat16* __restrict__ q,
                                             const std::uint8_t* __restrict__ cache_k,
                                             const std::uint8_t* __restrict__ cache_v,
                                             const __half* __restrict__ cache_k_scale,
                                             const __half* __restrict__ cache_v_scale,
                                             Metadata metadata,
                                             const std::int32_t* __restrict__ positions,
                                             float scale, __nv_bfloat16* __restrict__ out,
                                             std::int32_t width,
                                             float* __restrict__ partial_rows,
                                             float2* __restrict__ partial_stats) {
    if constexpr (!shape_set_in_image(TuringShapes)) { return; } // not this image's set
    // A Volta card runs its own prompt route (sm70/prompt.cu) or the standard kernel: a build of
    // several architectures compiles this kernel empty in the Volta image.
#if !defined(NINFER_SM70_DEVICE)
    constexpr int D          = kCausalPromptHeadDim;
    using Shape              = CausalPromptFp8FastShape<TuringShapes>;
    using Stage              = CausalPromptFp8FastStage;
    constexpr int Threads    = Shape::Threads;
    constexpr int Br         = Shape::Br;
    constexpr int Bc         = kCausalPromptFp8FastBc;
    constexpr int QKNt       = Bc / 8;
    constexpr int PVKs       = Bc / 16;
    constexpr int DBlocks    = D / 16;
    constexpr float Log2E    = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    // Longest-first issue order: the first q_heads CTAs take the last row block of every head.
    const int row_blocks = static_cast<int>(gridDim.x);
    const int q_heads    = causal_prompt_q_heads<Geometry>();
    const int kv_heads   = q_heads / Geometry::GroupSize;
    const int linear     = static_cast<int>(blockIdx.x + blockIdx.y * gridDim.x);
    const int q_head     = linear % q_heads;
    const int q0         = (row_blocks - 1 - linear / q_heads) * Br;
    const int kv_head    = q_head / Geometry::GroupSize;
    const int tokens     = metadata.valid_tokens(width);

    // Rows past the valid token count are inactive columns and publish zeros.
    const auto store_row = [&](int row, int d0, float v0, float v1, float v2, float v3) {
        const int token = q0 + row;
        if (token >= width) { return; }
        const bool valid   = token < tokens;
        const uint2 packed = make_uint2(pack_bf16x2(valid ? v0 : 0.0f, valid ? v1 : 0.0f),
                                        pack_bf16x2(valid ? v2 : 0.0f, valid ? v3 : 0.0f));
        store_vec(&out[causal_prompt_q_index<Geometry>(q_head, d0, token, q_heads)], packed);
    };

    if (q0 >= tokens) {
        // The merge publishes a split launch's inactive columns.
        if constexpr (!Split) {
            for (int element = tid; element < Br * (D / 4); element += Threads) {
                const int row = element / (D / 4);
                store_row(row, (element - row * (D / 4)) * 4, 0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        return;
    }

    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();
    const int rows                  = min(Br, tokens - q0);
    const int max_query_abs         = base_pos + q0 + rows - 1;
    const int key_blocks            = max_query_abs / Bc + 1;
    // Key pages [kb_begin, kb_end) of this split; an empty run publishes neutral statistics.
    const int split = Split ? static_cast<int>(blockIdx.z) : 0;
    const int pages_per_split =
        Split ? (key_blocks + static_cast<int>(gridDim.z) - 1) / static_cast<int>(gridDim.z) : 0;
    const int kb_begin = Split ? min(key_blocks, split * pages_per_split) : 0;
    const int kb_end   = Split ? min(key_blocks, kb_begin + pages_per_split) : key_blocks;

    // Q is never quantized -- only the cache is FP8 -- but the D256 rotation is a property of the
    // stored (permanently rotated) K, so Q must be rotated the same way for Q.K to be correct
    // (fp8/prompt.cuh).
    const int row_base = warp * 16;
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int row    = row_base + r;
        const int token  = q0 + row;
        const bool valid = row < rows;
        float q_values[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            q_values[k] = valid
                              ? __bfloat162float(
                                    q[causal_prompt_q_index<Geometry>(q_head, lane + 32 * k, token,
                                                                      q_heads)])
                              : 0.0f;
        }
        normalized_hadamard_d256_inplace(q_values, lane);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            q_b16[row * D + causal_prompt_swz(row, lane + 32 * k)] =
                __float2bfloat16(q_values[k]);
        }
    }

    const auto stage_base = [&](int stage) {
        return smem_raw + Shape::QBytes + stage * Stage::Bytes;
    };

    // One tile is one physical page of this KV head: 16 KiB of K codes and 16 KiB of V codes.
    // Keys past the CTA's last visible key are zero-filled so masked columns stay finite.
    const auto issue_tile = [&](int kb) {
        unsigned char* base = stage_base(kb % Shape::Stages);
        std::uint8_t* k_s   = reinterpret_cast<std::uint8_t*>(base);
        std::uint8_t* v_s   = k_s + Stage::TileBytes;
        const int page      = block_table[kb];
        const int valid     = min(Bc, max_query_abs + 1 - kb * Bc);
        const std::int64_t code_base =
            kv_cache_fp8_code_index(page, kv_head, 0, 0, kv_heads);
#pragma unroll
        for (int i = 0; i < Stage::TileBytes / 16 / Threads; ++i) {
            const int chunk = tid + i * Threads;
            const int key   = chunk >> 4;
            const int c     = chunk & 15;
            const int dst   = key * D + ((c ^ (key & 7)) << 4);
            const int bytes = key < valid ? 16 : 0;
            cp_async_zfill<16, Cache::cg>(k_s + dst, cache_k + code_base + key * D + c * 16,
                                          bytes);
            cp_async_zfill<16, Cache::cg>(v_s + dst, cache_v + code_base + key * D + c * 16,
                                          bytes);
        }
        cp_commit();
    };

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;

    // Rows this warp owns, for warp-uniform tile skipping and mask elision.
    const bool warp_active  = row_base < rows;
    const int warp_min_qabs = base_pos + q0 + row_base;
    const int warp_max_qabs = base_pos + q0 + min(row_base + 15, rows - 1);
    const int row0          = row_base + gid;
    const int row1          = row0 + 8;
    const int qabs0         = row0 < rows ? base_pos + q0 + row0 : -1;
    const int qabs1         = row1 < rows ? base_pos + q0 + row1 : -1;

    // [16-dimension block][even/odd n8 tile][fragment slot]: 128 floats per lane, accumulated in
    // FP32 through every tile.
    float acc[DBlocks][2][4];
#pragma unroll
    for (int b = 0; b < DBlocks; ++b) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[b][p][i] = 0.0f; }
        }
    }
    float running_m0     = -CUDART_INF_F;
    float running_m1     = -CUDART_INF_F;
    float running_l0     = 0.0f;
    float running_l1     = 0.0f;
    const float scale_l2 = scale * Log2E;

    // What QK hands to PV for one tile: the probability A fragments and the row rescale factors.
    unsigned pa[PVKs][4];
    float tile_alpha0 = 0.0f;
    float tile_alpha1 = 0.0f;
    bool tile_live    = false;

    const auto qk_softmax = [&](int kb) {
        tile_live = false;
        const int k0 = kb * Bc;
        if (!warp_active || k0 > warp_max_qabs) { return; }
        tile_live = true;
        const std::uint8_t* k_s =
            reinterpret_cast<const std::uint8_t*>(stage_base(kb % Shape::Stages));
        const int page        = block_table[kb];
        const std::int64_t scale_base = kv_cache_fp8_scale_index(page, kv_head, 0, kv_heads);
        const int valid              = min(Bc, max_query_abs + 1 - k0);

        // The per-key scales of the keys each lane supplies (key = nt*8 + gid), straight from the
        // cache: an invalid key reads zero, so its zero-filled codes decode to zero.
        float ks[QKNt];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int key_l = nt * 8 + gid;
            ks[nt]          = key_l < valid ? __half2float(cache_k_scale[scale_base + key_l])
                                            : 0.0f;
        }

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
#pragma unroll
        for (int kk = 0; kk < DBlocks; ++kk) {
            unsigned af[4];
            ldmatrix_x4(af[0], af[1], af[2], af[3],
                        smem_addr(&q_b16[(row_base + a_rowoff) * D +
                                         causal_prompt_swz(row_base + a_rowoff,
                                                           kk * 16 + a_coloff)]));
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                // The lane's two code pairs are adjacent u16s of its key's swizzled 16-byte chunk:
                // dimensions kk*16 + 2t .. 2t + 1 and kk*16 + 8 + 2t .. 8 + 2t + 1.
                const int key_l      = nt * 8 + gid;
                const std::uint16_t* krow =
                    reinterpret_cast<const std::uint16_t*>(k_s + key_l * D +
                                                           (((kk ^ (key_l & 7)) << 4)));
                unsigned b0 = causal_fp8_fast_kpair_bf16(krow[lid], ks[nt]);
                unsigned b1 = causal_fp8_fast_kpair_bf16(krow[4 + lid], ks[nt]);
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[0], af[1],
                         af[2], af[3], b0, b1);
            }
        }

        const bool full_tile = k0 + Bc - 1 <= warp_min_qabs;
        float bm0            = -CUDART_INF_F;
        float bm1            = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            if (!full_tile) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                score[nt][0]   = key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
                score[nt][1]   = key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
                score[nt][2]   = key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
                score[nt][3]   = key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
            }
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0        = fmaxf(running_m0, bm0);
        const float nm1        = fmaxf(running_m1, bm1);
        const float nm0_scaled = nm0 == -CUDART_INF_F ? 0.0f : nm0 * scale_l2;
        const float nm1_scaled = nm1 == -CUDART_INF_F ? 0.0f : nm1 * scale_l2;
        tile_alpha0            = running_m0 == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2_approx(__fmaf_rn(running_m0, scale_l2, -nm0_scaled));
        tile_alpha1            = running_m1 == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2_approx(__fmaf_rn(running_m1, scale_l2, -nm1_scaled));
        running_m0             = nm0;
        running_m1             = nm1;

        // Probabilities become the PV A fragments directly: k-step j covers keys 16j..16j+15,
        // which are score tiles 2j (keys 2t, 2t+1) and 2j+1 (keys 8+2t, 9+2t).
        float bl0 = 0.0f;
        float bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const float p00 = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled));
            const float p01 = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled));
            const float p10 = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled));
            const float p11 = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled));
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            pa[nt >> 1][(nt & 1) * 2 + 0] = pack_f16x2(p00, p01);
            pa[nt >> 1][(nt & 1) * 2 + 1] = pack_f16x2(p10, p11);
        }
        // Row sums stay lane-partial; the quad is reduced once after the sweep.
        running_l0 = __fmaf_rn(running_l0, tile_alpha0, bl0);
        running_l1 = __fmaf_rn(running_l1, tile_alpha1, bl1);
    };

    const auto pv = [&](int kb) {
        if (!tile_live) { return; }
        const std::uint8_t* v_s =
            reinterpret_cast<const std::uint8_t*>(stage_base(kb % Shape::Stages)) + Stage::VOffset;
        const int page        = block_table[kb];
        const std::int64_t scale_base = kv_cache_fp8_scale_index(page, kv_head, 0, kv_heads);
        const int valid              = min(Bc, max_query_abs + 1 - kb * Bc);

        // The per-key scales of the keys each lane supplies: (16j + 2t, 16j + 2t + 1) and
        // (16j + 8 + 2t, 16j + 9 + 2t) per k-step j; an invalid key reads zero.
        __half vs0[PVKs], vs1[PVKs], vs2[PVKs], vs3[PVKs];
#pragma unroll
        for (int j = 0; j < PVKs; ++j) {
            const int key_l = j * 16 + 2 * lid;
            vs0[j] = key_l < valid ? cache_v_scale[scale_base + key_l] : __float2half_rn(0.0f);
            vs1[j] = (key_l + 1) < valid ? cache_v_scale[scale_base + key_l + 1]
                                         : __float2half_rn(0.0f);
            vs2[j] = (key_l + 8) < valid ? cache_v_scale[scale_base + key_l + 8]
                                         : __float2half_rn(0.0f);
            vs3[j] = (key_l + 9) < valid ? cache_v_scale[scale_base + key_l + 9]
                                         : __float2half_rn(0.0f);
        }

        // The FP32 accumulator takes the tile's alpha before the tile's PV contributes (the
        // running-max rescale); the MMAs accumulate on top of it.
#pragma unroll
        for (int b = 0; b < DBlocks; ++b) {
            acc[b][0][0] *= tile_alpha0;
            acc[b][0][1] *= tile_alpha0;
            acc[b][1][0] *= tile_alpha0;
            acc[b][1][1] *= tile_alpha0;
            acc[b][0][2] *= tile_alpha1;
            acc[b][0][3] *= tile_alpha1;
            acc[b][1][2] *= tile_alpha1;
            acc[b][1][3] *= tile_alpha1;
        }

#pragma unroll
        for (int db = 0; db < DBlocks; db += 2) {
#pragma unroll
            for (int j = 0; j < PVKs; ++j) {
                // Matrices: keys 16j+0..7 and 16j+8..15 of d-block db, then of db + 1.
                const int key   = j * 16 + ((a_mat & 1) << 3) + a_rin;
                const int chunk = db + (a_mat >> 1);
                unsigned r[4];
                ldmatrix_x4_t(r[0], r[1], r[2], r[3],
                              smem_addr(&v_s[key * D + ((chunk ^ (key & 7)) << 4)]));
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    unsigned even_lo, odd_lo, even_hi, odd_hi;
                    causal_fp8_value_pair_fragments(r[2 * b], vs0[j], vs1[j], even_lo, odd_lo);
                    causal_fp8_value_pair_fragments(r[2 * b + 1], vs2[j], vs3[j], even_hi, odd_hi);
                    mma_f16(acc[db + b][0][0], acc[db + b][0][1], acc[db + b][0][2],
                            acc[db + b][0][3], pa[j][0], pa[j][1], pa[j][2], pa[j][3], even_lo,
                            even_hi);
                    mma_f16(acc[db + b][1][0], acc[db + b][1][1], acc[db + b][1][2],
                            acc[db + b][1][3], pa[j][0], pa[j][1], pa[j][2], pa[j][3], odd_lo,
                            odd_hi);
                }
            }
        }
    };

    if constexpr (Shape::Stages == 1) {
        // One stage: a tile is staged once every warp is done with the previous one, then
        // published.
#pragma unroll 1
        for (int kb = kb_begin; kb < kb_end; ++kb) {
            if (kb > kb_begin) { __syncthreads(); }
            issue_tile(kb);
            cp_wait<0>();
            __syncthreads();
            qk_softmax(kb);
            pv(kb);
        }
    } else {
        // Every warp publishes one tile per barrier; the next tile's copy overlaps this tile's
        // math.
        if (kb_begin < kb_end) { issue_tile(kb_begin); }
#pragma unroll 1
        for (int kb = kb_begin; kb < kb_end; ++kb) {
            cp_wait<0>();
            __syncthreads();
            if (kb + 1 < kb_end) { issue_tile(kb + 1); }
            qk_softmax(kb);
            pv(kb);
        }
    }

    running_l0         = warp_sum<4>(running_l0, FullMask);
    running_l1         = warp_sum<4>(running_l1, FullMask);
    const float inv_l0 = running_l0 > 0.0f ? __frcp_rn(running_l0) : 0.0f;
    const float inv_l1 = running_l1 > 0.0f ? __frcp_rn(running_l1) : 0.0f;
    if constexpr (Split) {
        // Each split publishes its rows normalized by its own sum, beside (max, sum).
        const auto publish = [&](int row, float maximum, float sum, float inverse, int e0) {
            if (row >= rows) { return; }
            const std::int64_t index =
                detail::causal_prompt_i8_fast_partial_row(split, q0 + row, q_head, width,
                                                          q_heads);
            if (lid == 0) { partial_stats[index] = make_float2(maximum, sum); }
            float* target = partial_rows + index * D;
#pragma unroll
            for (int b = 0; b < DBlocks; ++b) {
                store_vec(target + b * 16 + 4 * lid,
                          make_float4(acc[b][0][e0] * inverse, acc[b][1][e0] * inverse,
                                      acc[b][0][e0 + 1] * inverse,
                                      acc[b][1][e0 + 1] * inverse));
            }
        };
        publish(row0, running_m0, running_l0, inv_l0, 0);
        publish(row1, running_m1, running_l1, inv_l1, 2);
    } else {
        // Even n8 tiles hold dimensions 16b + 2n and odd tiles 16b + 2n + 1, so lane (g, t) owns
        // the four contiguous dimensions 16b + 4t .. 16b + 4t + 3 of its two rows.
#pragma unroll
        for (int b = 0; b < DBlocks; ++b) {
            const int d0 = b * 16 + 4 * lid;
            store_row(row0, d0, acc[b][0][0] * inv_l0, acc[b][1][0] * inv_l0,
                      acc[b][0][1] * inv_l0, acc[b][1][1] * inv_l0);
            store_row(row1, d0, acc[b][0][2] * inv_l1, acc[b][1][2] * inv_l1,
                      acc[b][0][3] * inv_l1, acc[b][1][3] * inv_l1);
        }
    }
#endif
}

} // namespace ninfer::ops
