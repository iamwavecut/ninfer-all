#pragma once

// PTQ1_0's products of 2 to 20 q8 activation columns on the integer tensor cores (sm_80 and newer;
// ggml_bridge_ptq1.cuh's lane-per-block kernel takes one column, and up to eight on the older
// targets). A warp multiplies one or two 16-row tiles (16 or 32 consecutive rows, or a fused
// product's 16 gate rows and their 16 up rows) by up to three 8-column tiles with
// mma.m16n8k32.s8, one 32-value slice of a block at a time; the eight warps of a CTA split the
// rows' blocks, and their sums meet in shared memory.
//
// A slice's 32 k slots may hold its values in any order the A and B fragments share. Lane
// (g, c) of the m16n8k32 fragments holds slots 4c .. 4c + 3 and 16 + 4c .. 16 + 4c + 3 of row g
// and row g + 8 (A) and of column g (B). Here lane c reads three of a block's seven words: qs word c,
// qs word 4 + (c >> 1) and the qh/d word, and takes these values, each group of four consecutive:
//
//   slice 0: qs[c] digit 0 (values 4c ..)            and digit 1 (16 + 4c ..)
//   slice 1: qs[c] digit 2 (32 + 4c ..)              and digit 3 (48 + 4c ..)
//   slice 2: qs[c] digit 4 (64 + 4c ..)              and qs[4 + h] digit o (80 + 8o + 4h ..)
//   slice 3: qs[4 + h] digit 2 + o (96 + 8o + 4h ..) and o = 0: qs[4 + h] digit 4 (112 + 4h ..),
//                                                         o = 1: qh digits 2h, 2h + 1 (120 + 4h ..)
//
// with h = c >> 1 and o = c & 1; the four lanes of a quad cover each slice's 32 values once. The
// B fragment loads the activation's int8 words at the same offsets. Each slice's dot is an exact
// integer, scaled by the activation's slice scale; a block's four slices are summed, then scaled by
// the row's d. A column's arithmetic does not depend on the other columns, so a product's columns
// come out the same at every column count.

#include "ggml_bridge_ptq1.cuh"

namespace ninfer::ops::gguf::detail::ptq1::tc {

// d += a b over one m16n8k32 tile of signed bytes. Only sm_80 and newer images run the kernels that
// call it (takes() below); the older images keep an empty body so the unit builds for them.
__device__ __forceinline__ void mma(int (&d)[4], std::uint32_t a0, std::uint32_t a1,
                                    std::uint32_t a2, std::uint32_t a3, std::uint32_t b0,
                                    std::uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)d;
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)b0;
    (void)b1;
#endif
}

inline constexpr int kWarps          = 8;
inline constexpr int kThreads        = kWarps * 32;
inline constexpr int kMaxColumnTiles = 3;
// Columns the tensor cores take; wider products take the matrix kernel. Up to 20 columns they beat
// it at the 27B's GDN layer shapes on an RTX 4090, 5090 and PRO 6000 (the layer's five products at
// 20 columns: 246.9 us against 252.1, 194.6 against 200.7, 159.8 against 184.3); at 24 they lose on
// the 4090 and 5090 (263.3 against 251.9, 217.1 against 200.7).
inline constexpr int kMaxColumns = 20;
static_assert(kMaxColumns <= 8 * kMaxColumnTiles);

// The words of one block of a row that lane c reads: qs word c, qs word 4 + (c >> 1), qh and d.
struct Words {
    std::uint32_t own, side, tail;
};

__device__ __forceinline__ Words load_words(const std::uint8_t* row, int block, int c, int h) {
    const auto* w = reinterpret_cast<const unsigned int*>(row + std::int64_t(block) * kBlockBytes);
    return {__ldg(w + c), __ldg(w + 4 + h), __ldg(w + 6)};
}

// The row's A words of the block's four slices (a: first half, b: second half), as signed bytes.
__device__ __forceinline__ void decode(const Words& w, int h, int o, std::uint32_t (&a)[4],
                                       std::uint32_t (&b)[4]) {
    Walk own = start(w.own);
    a[0]     = next(own);
    b[0]     = next(own);
    a[1]     = next(own);
    b[1]     = next(own);
    a[2]     = next<true>(own);
    Walk side              = start(w.side);
    const std::uint32_t x0 = next(side);
    const std::uint32_t x1 = next(side);
    const std::uint32_t x2 = next(side);
    const std::uint32_t x3 = next(side);
    const std::uint32_t x4 = next<true>(side);
    // Digits 2h and 2h + 1 of qh bytes 0 and 1 (values 120 + 4h ..): value 120 + 2t + j is digit t
    // of qh byte j. Times 9 skips two digits; a 16-bit lane stays below 2^16.
    const std::uint32_t q  = (__byte_perm(w.tail, 0, 0x4140) * (h != 0 ? 9u : 1u)) & 0x00FF00FFu;
    const std::uint32_t p0 = q * 3u;
    const std::uint32_t p1 = (p0 & 0x00FF00FFu) * 3u;
    const std::uint32_t qh = __byte_perm(p0, p1, 0x7531);
    b[2]                   = o != 0 ? x1 : x0;
    a[3]                   = o != 0 ? x3 : x2;
    b[3]                   = o != 0 ? qh : x4;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        a[s] = std::uint32_t(to_signed(a[s]));
        b[s] = std::uint32_t(to_signed(b[s]));
    }
}

__device__ __forceinline__ float block_scale(const Words& w) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(w.tail >> 16)));
}

// Output groups of a product: 32 rows, or 16 rows of a fused [gate; up] product.
[[nodiscard]] __host__ __device__ inline int groups(int rows, bool fused, int tiles) {
    return fused ? (rows + 15) / 16 : (rows + 16 * tiles - 1) / (16 * tiles);
}

template <int NT, bool Fused, int D, int kTiles>
__device__ __forceinline__ void product(const VecArgs& p, int group, int columns) {
    static_assert(!Fused || kTiles == 2, "a fused group is a gate tile and its up tile");
    __shared__ float partial[kWarps][kTiles][NT][32][4];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, c = lane & 3, h = c >> 1, o = c & 1;
    const int nb     = p.k / kBlockValues;
    const int slices = p.k / 32;
    const int first  = int(std::int64_t(warp) * nb / kWarps);
    const int last   = int(std::int64_t(warp + 1) * nb / kWarps);

    // Weight rows: tile t, row g (r = 0) and row g + 8 (r = 1).
    const std::uint8_t* row[kTiles][2];
#pragma unroll
    for (int t = 0; t < kTiles; ++t) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            int at;
            if constexpr (Fused) {
                at = min(16 * group + g + 8 * r, p.rows - 1) + t * p.rows;
            } else {
                at = min(16 * kTiles * group + 16 * t + g + 8 * r, p.rows - 1);
            }
            row[t][r] = p.weight + std::int64_t(at) * p.row_bytes;
        }
    }
    // Offsets of lane c's activation words in a block, per slice.
    const int off_a[4] = {4 * c, 32 + 4 * c, 64 + 4 * c, 96 + 8 * o + 4 * h};
    const int off_b[4] = {16 + 4 * c, 48 + 4 * c, 80 + 8 * o + 4 * h, 112 + 8 * o + 4 * h};
    const std::int8_t* bq[NT];
    const half2* sd[NT][2];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        bq[n] = p.qs + std::int64_t(min(8 * n + g, columns - 1)) * p.k;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            sd[n][j] = p.ds + std::int64_t(min(8 * n + 2 * c + j, columns - 1)) * slices;
        }
    }

    float acc[kTiles][NT][4];
#pragma unroll
    for (int t = 0; t < kTiles; ++t) {
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[t][n][i] = 0.0f; }
        }
    }
    // D blocks in flight; the weights do not depend on an earlier launch, so the first ones are
    // read before a dependent launch waits.
    Words ring[D][kTiles][2];
#pragma unroll
    for (int q = 0; q < D; ++q) {
        if (first + q < last) {
#pragma unroll
            for (int t = 0; t < kTiles; ++t) {
#pragma unroll
                for (int r = 0; r < 2; ++r) { ring[q][t][r] = load_words(row[t][r], first + q, c, h); }
            }
        }
    }
    ninfer::pdl::arch_enter();
    for (int b0 = first; b0 < last; b0 += D) {
#pragma unroll
        for (int q = 0; q < D; ++q) {
            const int b = b0 + q;
            if (b >= last) { break; }
            Words cur[kTiles][2];
#pragma unroll
            for (int t = 0; t < kTiles; ++t) {
#pragma unroll
                for (int r = 0; r < 2; ++r) { cur[t][r] = ring[q][t][r]; }
            }
            if (b + D < last) {
#pragma unroll
                for (int t = 0; t < kTiles; ++t) {
#pragma unroll
                    for (int r = 0; r < 2; ++r) { ring[q][t][r] = load_words(row[t][r], b + D, c, h); }
                }
            }
            // The block's activation words and slice scales, per column tile.
            std::uint32_t xa[NT][4], xb[NT][4];
            float dx[NT][2][4];
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const std::int8_t* at = bq[n] + kBlockValues * b;
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    xa[n][s] = __ldg(reinterpret_cast<const unsigned int*>(at + off_a[s]));
                    xb[n][s] = __ldg(reinterpret_cast<const unsigned int*>(at + off_b[s]));
                }
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const uint4 v = __ldg(reinterpret_cast<const uint4*>(sd[n][j] + 4 * b));
                    dx[n][j][0]   = __low2float(*reinterpret_cast<const half2*>(&v.x));
                    dx[n][j][1]   = __low2float(*reinterpret_cast<const half2*>(&v.y));
                    dx[n][j][2]   = __low2float(*reinterpret_cast<const half2*>(&v.z));
                    dx[n][j][3]   = __low2float(*reinterpret_cast<const half2*>(&v.w));
                }
            }
#pragma unroll
            for (int t = 0; t < kTiles; ++t) {
                std::uint32_t a0[4], a1[4], a2[4], a3[4];
                decode(cur[t][0], h, o, a0, a2);
                decode(cur[t][1], h, o, a1, a3);
                const float d0 = block_scale(cur[t][0]), d1 = block_scale(cur[t][1]);
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        int q4[4] = {0, 0, 0, 0};
                        mma(q4, a0[s], a1[s], a2[s], a3[s], xa[n][s], xb[n][s]);
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            part[i] = fmaf(float(q4[i]), dx[n][i & 1][s], part[i]);
                        }
                    }
                    acc[t][n][0] = fmaf(d0, part[0], acc[t][n][0]);
                    acc[t][n][1] = fmaf(d0, part[1], acc[t][n][1]);
                    acc[t][n][2] = fmaf(d1, part[2], acc[t][n][2]);
                    acc[t][n][3] = fmaf(d1, part[3], acc[t][n][3]);
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < kTiles; ++t) {
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { partial[warp][t][n][lane][i] = acc[t][n][i]; }
        }
    }
    __syncthreads();
    // Thread e sums output e of the group over the warps in order and stores it.
    constexpr int kOutputs = (Fused ? 1 : kTiles) * NT * 32 * 4;
    for (int e = threadIdx.x; e < kOutputs; e += kThreads) {
        const int i   = e & 3;
        const int ln  = (e >> 2) & 31;
        const int n   = (e >> 7) % NT;
        const int t   = (e >> 7) / NT;
        const int col = 8 * n + 2 * (ln & 3) + (i & 1);
        const int r   = (ln >> 2) + 8 * (i >> 1);
        if (col >= columns) { continue; }
        float x = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) { x += partial[w][t][n][ln][i]; }
        if constexpr (Fused) {
            const int out_row = 16 * group + r;
            if (out_row >= p.rows) { continue; }
            float up = 0.0f;
#pragma unroll
            for (int w = 0; w < kWarps; ++w) { up += partial[w][1][n][ln][i]; }
            vec_store(p.out, out_row, col, vec_gate(x, p.out.activation) * up);
        } else {
            const int out_row = 16 * kTiles * group + 16 * t + r;
            if (out_row >= p.rows) { continue; }
            vec_store(p.out, out_row, col, x);
        }
    }
}

// Up to kMaxVecJobs products of one K: CTAs [p.first[j], p.first[j + 1]) take product j's groups.
// Two or more column tiles, or four blocks in flight, hold more than 128 registers a thread: one CTA
// an SM keeps them out of local memory.
template <int NT, bool Fused, int D, int Tiles>
__launch_bounds__(kThreads, NT == 1 && D <= 2 ? 2 : 1) __global__ void kernel(VecJobs p,
                                                                              int columns) {
    const int b = blockIdx.x;
    int j       = 0;
#pragma unroll
    for (int i = 1; i < kMaxVecJobs; ++i) {
        if (i < p.count && b >= p.first[i]) { j = i; }
    }
    VecArgs job = p.job[0];
    int first   = p.first[0];
#pragma unroll
    for (int i = 1; i < kMaxVecJobs; ++i) {
        if (i == j) {
            job   = p.job[i];
            first = p.first[i];
        }
    }
    product<NT, Fused, D, Tiles>(job, b - first, columns);
}

template <int NT, bool Fused, int D, int Tiles>
void launch(const VecJobs& args, int columns, cudaStream_t stream) {
    VecJobs part  = args;
    part.first[0] = 0;
    for (int i = 0; i < part.count; ++i) {
        part.job[i].prefetch = false;
        part.first[i + 1]    = part.first[i] + groups(part.job[i].rows, Fused, Tiles);
    }
    ninfer::pdl::launch_arch(part.first[part.count], kThreads, 0, stream,
                             kernel<NT, Fused, D, Tiles>, part, columns);
    check(cudaGetLastError(), "ptq1 tensor-core product launch");
}

// The forms measured fastest on an RTX 5090 (kernel alone, L2 evicted first): a fused [gate; up]
// product keeps four blocks in flight up to 16 columns (2 x 17408 rows over K = 5120 at 8 columns:
// 36.5 us, 47.2 us with two), and an unfused product of up to 8 columns takes 16 rows a warp (5120
// rows over K = 17408 at 2 columns: 21.1 us, 22.9 us with 32 rows; at 16 columns 16 rows a warp
// take 51.8 us, 32 rows 33.8 us).
template <int NT>
void launch_form(const VecJobs& args, int columns, bool fused, cudaStream_t stream) {
    if (fused) {
        if constexpr (NT <= 2) { return launch<NT, true, 4, 2>(args, columns, stream); }
        return launch<NT, true, 2, 2>(args, columns, stream);
    }
    if constexpr (NT == 1) { return launch<1, false, 2, 1>(args, columns, stream); }
    return launch<NT, false, 2, 2>(args, columns, stream);
}

// Whether the device takes the tensor-core form (mma.m16n8k32.s8: sm_80 and newer).
[[nodiscard]] inline bool takes() { return device_facts().cc >= 800; }

inline void launch_columns(const VecJobs& args, int columns, bool fused, cudaStream_t stream) {
    switch ((columns + 7) / 8) {
    case 1: return launch_form<1>(args, columns, fused, stream);
    case 2: return launch_form<2>(args, columns, fused, stream);
    case 3: return launch_form<3>(args, columns, fused, stream);
    default: break;
    }
    throw std::invalid_argument("ptq1 tensor-core product: columns must be 1..20");
}

} // namespace ninfer::ops::gguf::detail::ptq1::tc
