#pragma once

// ninfer::ops - the products of the split-KV causal small-T kernels on sm_70 (Tesla V100). Volta's
// tensor cores multiply FP16 operands only, as mma.m8n8k4; these kernels keep the other tiers'
// grids, split ranges, cache writes and partial outputs, so the split reducers are shared.
//
// A warp owns eight query rows (a row group: G query heads times T tokens, eight at a time) and
// walks its split's keys in tiles of 32: S = Q K^T as m8n8k4.row.col products (the four quad pairs
// take eight keys each, Q's fragment mirrored), the online softmax in FP32, P through shared memory
// into m8n8k4.row.row products with V (the quad pairs take eight of each 32 head dimensions). Each
// lane reads its key's K row and the V rows of its products through the kernel's Rows type, which
// decodes the cache's coding to FP16 as it loads. One row group splits its tiles over up to four
// warps, combined in shared memory before the partial is written. A tile whose keys (or the CTA's
// queries) hold a BF16 value beyond FP16 forms its scores again in FP32 from the BF16 values
// (exact_scores), and one whose values do adds what their saturation took from P V
// (add_value_excess), so the products keep BF16's range.
//
// The kernels: bf16/small_t_sm70.cuh (BF16 caches), int8/small_t_sm70.cuh (the INT8 family) and
// sm70/small_t_quant.cuh (FP8, NVFP4 and k8v4); sm70/prompt.cu shares the products.

#include "ops/linear/sm70/sm70_bf16_f16.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"
#include "ops/softmax_attention/dense/causal_cache/sm70/small_t_shape.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops {

namespace sm70_attention {

// Volta's m8n8k4 product. Kernels launched on Volta only emit it in the 70 image; a portable
// caller (the position-split prompt partials, sm70/prompt.cu) emits it on every architecture,
// where the later tensor cores still execute it.
template <bool RowB, bool Portable = false>
__device__ __forceinline__ void hmma(float (&c)[8], unsigned a0, unsigned a1, unsigned b0,
                                     unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    constexpr bool kEmit = Portable || __CUDA_ARCH__ == 700;
    if constexpr (kEmit && RowB) {
        asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]),
                       "+f"(c[6]), "+f"(c[7])
                     : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
        return;
    } else if constexpr (kEmit) {
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]),
                       "+f"(c[6]), "+f"(c[7])
                     : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
        return;
    }
#endif
    (void)c;
    (void)a0;
    (void)a1;
    (void)b0;
    (void)b1;
}

__device__ __forceinline__ unsigned to_f16x2(unsigned bf16x2) {
    return detail::sm70::bf16x2_to_f16x2(bf16x2);
}

// Four signed int8 codes as two FP16 pairs (code 0 in the low lane of the first), times `scale`.
__device__ __forceinline__ void int8x4_to_f16x2(unsigned codes, __half2 scale, unsigned& lo,
                                                unsigned& hi) {
    const unsigned offset = codes ^ 0x80808080u; // 128 + code per byte
    const __half2 bias    = __float2half2_rn(1152.0f);
    const unsigned m0     = __byte_perm(offset, 0x64646464u, 0x4140u);
    const unsigned m1     = __byte_perm(offset, 0x64646464u, 0x4342u);
    const __half2 v0      = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&m0), bias), scale);
    const __half2 v1      = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&m1), bias), scale);
    lo                    = *reinterpret_cast<const unsigned*>(&v0);
    hi                    = *reinterpret_cast<const unsigned*>(&v1);
}

// Flags BF16 values beyond FP16 in `bits` (detail::sm70::bf16x2_beyond_f16, accumulated): a
// value is beyond where bits & kBeyondBits is nonzero.
inline constexpr unsigned kBeyondBits = 0x80008000u;
__device__ __forceinline__ unsigned beyond_bits(unsigned bf16x2) {
    return (bf16x2 & 0x7f807f80u) + 0x38803880u;
}

// Rows whose lanes read 32 dimensions of a row at a time (Int8Rows::kChunks).
template <class Rows, class = void>
struct chunked_rows : std::false_type {};
template <class Rows>
struct chunked_rows<Rows, std::void_t<decltype(Rows::kChunks)>>
    : std::bool_constant<Rows::kChunks> {};

// Rows of BF16 values that FP16 may not hold (Bf16Rows::kExact).
template <class Rows, class = void>
struct exact_rows : std::false_type {};
template <class Rows>
struct exact_rows<Rows, std::void_t<decltype(Rows::kExact)>> : std::bool_constant<Rows::kExact> {};

// Rows whose products run on every architecture (hmma's Portable form; Rows::kPortableMma).
template <class Rows, class = void>
struct portable_mma_rows : std::false_type {};
template <class Rows>
struct portable_mma_rows<Rows, std::void_t<decltype(Rows::kPortableMma)>>
    : std::bool_constant<Rows::kPortableMma> {};

// Eight BF16 values of a and of b: sum a_e b_e in FP32.
__device__ __forceinline__ float dot8_bf16(uint4 a, uint4 b) {
    const unsigned aw[4] = {a.x, a.y, a.z, a.w};
    const unsigned bw[4] = {b.x, b.y, b.z, b.w};
    float sum            = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum = fmaf(__uint_as_float(aw[i] << 16), __uint_as_float(bw[i] << 16), sum);
        sum = fmaf(__uint_as_float(aw[i] & 0xffff0000u), __uint_as_float(bw[i] & 0xffff0000u), sum);
    }
    return sum;
}

// The tile's unscaled scores in FP32 from the BF16 queries and keys, in place of the FP16 products:
// element i is row (lane & 1) + (i & 2) + 4h of the warp's rows from q_row0 and key k0 + 8 qp +
// (i & 4) + (lane & 2) + (i & 1). Keys outside [key_lo, key_hi) and rows without a query score 0
// (the caller masks them).
template <class Rows>
__device__ __forceinline__ void exact_scores(const Rows& rows, int k0, int key_lo, int key_hi,
                                          int q_row0, float (&s)[8]) {
    constexpr int D = kCausalHeadDim;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int qp    = (lane >> 2) & 3;
    const int h     = lane >> 4;
    const __nv_bfloat16* query[2] = {rows.query(q_row0 + (lane & 1) + 4 * h),
                                     rows.query(q_row0 + (lane & 1) + 4 * h + 2)};
    const __nv_bfloat16* key[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int k = k0 + qp * 8 + 4 * (e >> 1) + (lane & 2) + (e & 1);
        key[e]      = k >= key_lo && k < key_hi ? rows.key_bf16(k) : nullptr;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) { s[i] = 0.0f; }
#pragma unroll 1
    for (int d = 0; d < D; d += 8) {
        uint4 q8[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            q8[r] = query[r] != nullptr ? __ldg(reinterpret_cast<const uint4*>(query[r] + d))
                                        : make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            if (key[e] == nullptr) { continue; }
            const uint4 k8 = __ldg(reinterpret_cast<const uint4*>(key[e] + d));
            const int i    = 4 * (e >> 1) + (e & 1);
            s[i] += dot8_bf16(q8[0], k8);
            s[i + 2] += dot8_bf16(q8[1], k8);
        }
    }
}

// Adds to the warp's outputs what saturating the tile's BF16 values to FP16 took from P V: the
// FP16 probability in p_tile times the value's excess (detail::sm70::saturated_excess), for each
// output element of the lane (row (lane & 1) + (i & 2) + 4h, dimension 32n + 8qp + (i & 4) +
// (lane & 2) + (i & 1)). A zero probability adds nothing, whatever the value.
template <class Rows>
__device__ __forceinline__ void add_value_excess(const Rows& rows, const __half* p_tile, int k0,
                                              int key_lo, int key_hi, float (&acc)[8][8]) {
    constexpr int Bc      = 32;
    constexpr int PStride = kPStride;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int qp          = (lane >> 2) & 3;
    const int row_a       = (lane & 1) + 4 * (lane >> 4);
#pragma unroll 1
    for (int kk = 0; kk < Bc; ++kk) {
        const int k = k0 + kk;
        if (k < key_lo || k >= key_hi) { continue; }
        const float p[2] = {__half2float(p_tile[row_a * PStride + kk]),
                            __half2float(p_tile[(row_a + 2) * PStride + kk])};
        const __nv_bfloat16* v = rows.value_bf16(k);
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int d         = 32 * n + 8 * qp + 4 * half + (lane & 2);
                const unsigned word = *reinterpret_cast<const unsigned*>(v + d);
                const float lo      = detail::sm70::saturated_excess(__uint_as_float(word << 16));
                const float hi =
                    detail::sm70::saturated_excess(__uint_as_float(word & 0xffff0000u));
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    if (p[r] == 0.0f) { continue; }
                    const int i = 4 * half + 2 * r;
                    acc[n][i] = fmaf(p[r], lo, acc[n][i]);
                    acc[n][i + 1] = fmaf(p[r], hi, acc[n][i + 1]);
                }
            }
        }
    }
}

// The value dimension of accumulator element i of product n in a lane: column c = (i & 4) +
// (lane & 2) + (i & 1) of product n is dimension 32n + 8qp + c, or, for chunked rows, 64qp +
// 32(c / 4) + 4n + c % 4, which gives each lane's B operands 32 consecutive dimensions.
template <bool Chunks>
__device__ __forceinline__ int acc_dim(int lane, int n, int i) {
    const int qp     = (lane >> 2) & 3;
    const int column = (lane & 2) + (i & 1);
    return Chunks ? 64 * qp + 8 * (i & 4) + 4 * n + column : 32 * n + 8 * qp + (i & 4) + column;
}

// The online-softmax state of a warp's eight query rows in one lane: rows a ((lane & 1) + 4h)
// and b (a + 2) of the group, and the lane's share of their FP32 outputs (element i of product n is
// row (lane & 1) + (i & 2) + 4h, dimension acc_dim(lane, n, i)).
struct RowState {
    float m_a;
    float m_b;
    float l_a;
    float l_b;
    float acc[8][8];

    __device__ __forceinline__ RowState() : l_a(0.0f), l_b(0.0f) {
        m_a = -CUDART_INF_F;
        m_b = -CUDART_INF_F;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { acc[n][i] = 0.0f; }
        }
    }
};

// One key tile [k0, k0 + 32) of a warp's eight query rows: S = Q K^T as m8n8k4.row.col products
// (q_row is this lane's A row, Q in FP16 QStride halves apart), the masked online softmax in FP32,
// P through the warp's p_tile into m8n8k4.row.row products with V. Keys outside [key_lo, key_hi)
// and past a row's position (qabs_a, qabs_b; -1 for no row) are masked. `rows` supplies K and V in
// FP16: key(k) and key8(row, d) the eight dimensions from d of key k (chunked rows: key_chunk(row,
// d0) the stored bytes of 32 and key8(row, chunk, d0, j) the eight from d0 + 8j), value(k, dim0)
// and value4(row, n) the four from dim0 + 32 n (chunked rows: from dim0 + 4n). Exact rows
// (exact_rows) also flag saturated keys and values (keys_beyond, values_beyond, query_beyond) and
// give their BF16 values, the warp's rows numbered from q_row0.
template <class Rows>
__device__ __forceinline__ void attend_tile(const Rows& rows, const __half* q_row, __half* p_tile,
                                            int k0, int key_lo, int key_hi, int qabs_a, int qabs_b,
                                            float scale, RowState& st, int q_row0 = 0) {
    constexpr int D             = kCausalHeadDim;
    constexpr int Bc            = 32;
    constexpr int PStride       = kPStride;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int qp                = (lane >> 2) & 3;
    const int h                 = lane >> 4;
    const int a_row             = (lane & 3) + 4 * h;

    constexpr bool Chunks   = chunked_rows<Rows>::value;
    constexpr bool Exact    = exact_rows<Rows>::value;
    constexpr bool Portable = portable_mma_rows<Rows>::value;

    // S = Q K^T: this lane supplies the K row of key k0 + 8 qp + (lane % 4) + 4h.
    float s[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    unsigned key_bits = 0u;
    {
        const int key    = k0 + qp * 8 + (lane & 3) + 4 * h;
        const bool live  = key >= key_lo && key < key_hi;
        const auto k_row = rows.key(live ? key : k0);
        if constexpr (Chunks) {
#pragma unroll (Rows::kKeyChunksInFlight)
            for (int d0 = 0; d0 < D; d0 += 32) {
                const auto chunk = rows.key_chunk(k_row, d0);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint4 k8 = live ? rows.key8(k_row, chunk, d0, j) : make_uint4(0, 0, 0, 0);
                    const uint4 q8 = *reinterpret_cast<const uint4*>(q_row + d0 + 8 * j);
                    hmma<false, Portable>(s, q8.x, q8.y, k8.x, k8.y);
                    hmma<false, Portable>(s, q8.z, q8.w, k8.z, k8.w);
                }
            }
        } else {
#pragma unroll 8
            for (int d = 0; d < D; d += 8) {
                uint4 k8 = make_uint4(0, 0, 0, 0);
                if (live) {
                    if constexpr (Exact) {
                        k8 = rows.key8(k_row, d, key_bits);
                    } else {
                        k8 = rows.key8(k_row, d);
                    }
                }
                const uint4 q8 = *reinterpret_cast<const uint4*>(q_row + d);
                hmma<false, Portable>(s, q8.x, q8.y, k8.x, k8.y);
                hmma<false, Portable>(s, q8.z, q8.w, k8.z, k8.w);
            }
        }
    }
    if constexpr (Exact) {
        if (rows.keys_beyond(key_bits) || rows.query_beyond) {
            exact_scores(rows, k0, key_lo, key_hi, q_row0, s);
        }
    }
    // Masked, scaled scores and the tile's row maxima (rows a: i & 2 == 0, b: i & 2 == 2).
    float bm_a = -CUDART_INF_F, bm_b = -CUDART_INF_F;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int key    = k0 + qp * 8 + (i & 4) + (lane & 2) + (i & 1);
        const bool upper = (i & 2) != 0;
        const int qabs   = upper ? qabs_b : qabs_a;
        const bool keep  = qabs >= 0 && key >= key_lo && key < key_hi && key <= qabs;
        s[i]             = keep ? s[i] * scale : -CUDART_INF_F;
        if (upper) {
            bm_b = fmaxf(bm_b, s[i]);
        } else {
            bm_a = fmaxf(bm_a, s[i]);
        }
    }
#pragma unroll
    for (int mask = 2; mask <= 8; mask <<= 1) {
        bm_a = fmaxf(bm_a, __shfl_xor_sync(FullMask, bm_a, mask));
        bm_b = fmaxf(bm_b, __shfl_xor_sync(FullMask, bm_b, mask));
    }
    const float nm_a    = fmaxf(st.m_a, bm_a);
    const float nm_b    = fmaxf(st.m_b, bm_b);
    const float alpha_a = st.m_a == -CUDART_INF_F ? 0.0f : exp2_approx((st.m_a - nm_a) * Log2E);
    const float alpha_b = st.m_b == -CUDART_INF_F ? 0.0f : exp2_approx((st.m_b - nm_b) * Log2E);
    float bl_a = 0.0f, bl_b = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool upper = (i & 2) != 0;
        const float nm   = upper ? nm_b : nm_a;
        const float p =
            (nm > -CUDART_INF_F && s[i] > -CUDART_INF_F) ? exp2_approx((s[i] - nm) * Log2E) : 0.0f;
        if (upper) {
            bl_b += p;
        } else {
            bl_a += p;
        }
        const int row               = (lane & 1) + (i & 2) + 4 * h;
        const int col               = qp * 8 + (i & 4) + (lane & 2) + (i & 1);
        p_tile[row * PStride + col] = __float2half_rn(p);
    }
#pragma unroll
    for (int mask = 2; mask <= 8; mask <<= 1) {
        bl_a += __shfl_xor_sync(FullMask, bl_a, mask);
        bl_b += __shfl_xor_sync(FullMask, bl_b, mask);
    }
    st.l_a = st.l_a * alpha_a + bl_a;
    st.l_b = st.l_b * alpha_b + bl_b;
    st.m_a = nm_a;
    st.m_b = nm_b;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { st.acc[n][i] *= (i & 2) ? alpha_b : alpha_a; }
    }
    __syncwarp();
    // O += P V: four keys per product; the quad pairs take eight of each 32 dimensions, or, for
    // chunked rows, 64 consecutive dimensions each.
    const int value_dim0 = Chunks ? 64 * qp + 32 * h : 8 * qp + 4 * h;
    unsigned value_bits  = 0u;
#pragma unroll 2
    for (int step = 0; step < Bc / 4; ++step) {
        const uint2 p4   = *reinterpret_cast<const uint2*>(p_tile + a_row * PStride + 4 * step);
        const int key    = k0 + 4 * step + (lane & 3);
        const bool live  = key >= key_lo && key < key_hi;
        const auto v_row = rows.value(live ? key : k0, value_dim0);
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            uint2 v4 = make_uint2(0, 0);
            if (live) {
                if constexpr (Exact) {
                    v4 = rows.value4(v_row, n, value_bits);
                } else {
                    v4 = rows.value4(v_row, n);
                }
            }
            hmma<true, Portable>(st.acc[n], p4.x, p4.y, v4.x, v4.y);
        }
    }
    if constexpr (Exact) {
        if (rows.values_beyond(value_bits)) {
            add_value_excess(rows, p_tile, k0, key_lo, key_hi, st.acc);
        }
    }
    __syncwarp();
}

// The products, softmax and partial outputs of one CTA, after its query rows are in q_s (FP16,
// QStride halves apart, zero past row_count). `rows` supplies K and V as attend_tile reads them.
template <typename Geometry, int TokenTile, class Rows>
__device__ __forceinline__ void
attend(const Rows& rows, const __half* q_s, __half* p_s, float (*stat_m)[8], float (*stat_l)[8],
       float* combined, int tokens, int row_count, int kv_head, int split, int split_start,
       int split_end, int first_tile, int key_blocks, const std::int32_t* pos, float scale,
       float* partial_acc, float* partial_m, float* partial_l, CausalKeyPartition partition,
       const std::int32_t* pages, int first_page) {
    using Dims             = Shape<TokenTile * Geometry::GroupSize>;
    constexpr int D        = kCausalHeadDim;
    constexpr int Bc       = 32;
    constexpr int Groups   = Dims::kRowGroups;
    constexpr int KeyWarps = Dims::kKeyWarps;
    constexpr int Threads  = 32 * Dims::kWarps;
    constexpr int QStride  = kQStride;
    constexpr int PStride  = kPStride;
    constexpr float Log2E  = 1.4426950408889634074f;
    constexpr bool Chunks  = chunked_rows<Rows>::value;
    // The grid spans the call's KV heads.
    const int kv_heads     = static_cast<int>(gridDim.x);
    const int q_heads      = kv_heads * Geometry::GroupSize;
    const int tid          = static_cast<int>(threadIdx.x);
    const int warp         = tid >> 5;
    const int lane         = tid & 31;

    const int group    = warp / KeyWarps;
    const int key_warp = warp % KeyWarps;
    const int h        = lane >> 4;
    const int a_row    = (lane & 3) + 4 * h;
    // The accumulator rows of this lane: element i is row (lane & 1) + (i & 2) + 4h of the group.
    const int row_a           = group * 8 + (lane & 1) + 4 * h;
    const int row_b           = row_a + 2;
    const auto query_position = [&](int row) {
        return row < row_count ? pos[row / Geometry::GroupSize] : -1;
    };
    const int qabs_a    = query_position(row_a);
    const int qabs_b    = query_position(row_b);
    const __half* q_row = q_s + (group * 8 + a_row) * QStride;
    __half* p_tile      = p_s + warp * 8 * PStride;

    RowState st;
    // A segmented partition sweeps each row group's tiles on its first key warp alone, so a row's
    // arithmetic is the same however many row groups share the CTA; the combine below then adds
    // the other warps' empty shares exactly. A strided one skips the tiles of other ranks' pages
    // (causal_small_t_localize_pages).
    const int first_block = partition.segmented() ? (key_warp == 0 ? 0 : key_blocks) : key_warp;
    const int block_step  = partition.segmented() ? 1 : KeyWarps;
    for (int block = first_block; block < key_blocks; block += block_step) {
        const int k0 = first_tile + block * Bc;
        if (partition.strided() && pages[(k0 >> kPagedKVPageShift) - first_page] < 0) { continue; }
        attend_tile(rows, q_row, p_tile, k0, split_start, split_end, qabs_a, qabs_b, scale, st,
                    group * 8);
    }
    float& m_a         = st.m_a;
    float& m_b         = st.m_b;
    float& l_a         = st.l_a;
    float& l_b         = st.l_b;
    float (&acc)[8][8] = st.acc;

    // Element i of product n: row (lane & 1) + (i & 2) + 4h, dimension acc_dim(lane, n, i).
    const auto write_row = [&](int row, float m, float l) {
        int q_head = 0;
        int token  = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        partial_m[causal_partial_stat_index(q_head, token, split, tokens, q_heads)] = m;
        partial_l[causal_partial_stat_index(q_head, token, split, tokens, q_heads)] = l;
    };
    if constexpr (KeyWarps == 1) {
        if ((lane & 14) == 0) {
            if (row_a < row_count) { write_row(row_a, m_a, l_a); }
            if (row_b < row_count) { write_row(row_b, m_b, l_b); }
        }
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
            for (int i = 0; i < 8; i += 2) {
                const int row = (i & 2) ? row_b : row_a;
                if (row >= row_count) { continue; }
                int q_head = 0;
                int token  = 0;
                causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
                const int d                            = acc_dim<Chunks>(lane, n, i);
                *reinterpret_cast<float2*>(&partial_acc[causal_partial_acc_index(
                    q_head, d, token, split, tokens, q_heads)]) =
                    make_float2(acc[n][i], acc[n][i + 1]);
            }
        }
    } else {
        // Combine the row group's key warps: their maxima, then each warp's share rescaled to
        // the common maximum and added in shared memory, warp by warp in a fixed order (a repeated
        // call reproduces every bit).
        if ((lane & 14) == 0) {
            const int local_a         = (lane & 1) + 4 * h;
            stat_m[warp][local_a]     = m_a;
            stat_m[warp][local_a + 2] = m_b;
            stat_l[warp][local_a]     = l_a;
            stat_l[warp][local_a + 2] = l_b;
        }
        __syncthreads();
        const auto common_max = [&](int local) {
            float m = -CUDART_INF_F;
#pragma unroll
            for (int w = 0; w < KeyWarps; ++w) {
                m = fmaxf(m, stat_m[group * KeyWarps + w][local]);
            }
            return m;
        };
        const int local_a = (lane & 1) + 4 * h;
        const float max_a = common_max(local_a);
        const float max_b = common_max(local_a + 2);
        const float f_a   = m_a == -CUDART_INF_F ? 0.0f : exp2_approx((m_a - max_a) * Log2E);
        const float f_b   = m_b == -CUDART_INF_F ? 0.0f : exp2_approx((m_b - max_b) * Log2E);
        float* group_acc  = combined + group * 8 * D;
        // Chunked rows keep a row's dimension d at slot_of(d, row): their quad pairs' dimensions
        // are 64 apart, one bank, so bits 6-7 move to 3-4, and rows are D apart, so odd rows flip
        // bit 2. A warp's adds then meet at most two to a bank.
        const auto slot_of = [](int d, int row) {
            return Chunks ? d ^ ((row & 1) << 2) ^ (((d >> 6) & 3) << 3) : d;
        };
        for (int w = 0; w < KeyWarps; ++w) {
            if (key_warp == w) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int local   = (lane & 1) + (i & 2) + 4 * h;
                        const int d       = acc_dim<Chunks>(lane, n, i);
                        const float share = acc[n][i] * ((i & 2) ? f_b : f_a);
                        float& slot       = group_acc[local * D + slot_of(d, local)];
                        slot              = w == 0 ? share : slot + share;
                    }
                }
            }
            __syncthreads();
        }
        // One thread per (row, two dimensions) writes the combined partial.
        for (int idx = tid; idx < Groups * 8 * (D / 2); idx += Threads) {
            const int row = idx / (D / 2);
            const int d   = (idx - row * (D / 2)) * 2;
            if (row >= row_count) { continue; }
            const int g     = row / 8;
            const int local = row % 8;
            int q_head      = 0;
            int token       = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (!causal_valid_q_head<Geometry>(kv_head, q_head, kv_heads)) { continue; }
            float m = -CUDART_INF_F;
#pragma unroll
            for (int w = 0; w < KeyWarps; ++w) { m = fmaxf(m, stat_m[g * KeyWarps + w][local]); }
            if (d == 0) {
                float l = 0.0f;
#pragma unroll
                for (int w = 0; w < KeyWarps; ++w) {
                    const float mw = stat_m[g * KeyWarps + w][local];
                    l += mw == -CUDART_INF_F
                             ? 0.0f
                             : stat_l[g * KeyWarps + w][local] * exp2_approx((mw - m) * Log2E);
                }
                partial_m[causal_partial_stat_index(q_head, token, split, tokens, q_heads)] = m;
                partial_l[causal_partial_stat_index(q_head, token, split, tokens, q_heads)] = l;
            }
            const float* src = combined + (g * 8 + local) * D + slot_of(d, local);
            *reinterpret_cast<float2*>(
                &partial_acc[causal_partial_acc_index(q_head, d, token, split, tokens, q_heads)]) =
                make_float2(src[0], src[1]);
        }
    }
}

} // namespace sm70_attention

} // namespace ninfer::ops
