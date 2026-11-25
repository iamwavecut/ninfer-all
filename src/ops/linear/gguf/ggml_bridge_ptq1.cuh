#pragma once

// PTQ1_0's vector products: one to eight q8 activation columns times a matrix of PrismML's ternary
// blocks (ggml-common.h block_ptq1_0). A lane takes whole 128-value blocks: its qs words hold their
// values in value order (word g of qs[0..15], digit t: values 16 t + 4 g ..; word g of qs[16..23]:
// values 80 + 8 t + 4 g ..), so walking a word's five base-3 digits in turn yields the words of the
// activation's four 32-value slices in turn, and no lane ever decodes a digit it does not use. The
// vector kernel's slice decoders (ggml_bridge_vec.cuh) give each lane one slice; here a slice's
// eight words would come from different words and digits per slice, which a warp would run as four
// divergent paths.
//
// A row set is two weight rows (two consecutive rows, or a fused product's gate row and its up row)
// that share their activation loads; G lanes take a row set, lane q its blocks q, q + G, ..., where
// G is 8, or 16 or 32 when a product's row sets are too few to keep the SMs busy or a lane would walk
// too many blocks (group_lanes). The
// activation is transposed (gguf::vector_activation_transposed): the lanes of a row set read
// neighbouring 16 bytes of one plane, and each slice's (d, sum) pair comes from one load of a block's
// four (16 bytes) or, on Blackwell, from its own (4 bytes; kSliceScales). Each
// slice's integer dot is exact: the {0, 1, 2} digits' dot less the slice's exact sum of
// activations, which the activation carries, equal to the dot of the digits less one. The float
// sums then run slice by slice and block by block, each lane its blocks in order, and the G lanes'
// sums meet in a butterfly.

#include "ggml_bridge_vec_launch.cuh"
#include "../../../core/pdl_arch.cuh"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::gguf::detail::ptq1 {

inline constexpr int kWarps       = 4;
// Blocks per SM that one launch of several products may hold (ggml_bridge_vec.cuh
// kVecJobsBlocksPerSm, whose measurements set it).
inline constexpr int kJobsBlocksPerSm = 10;
inline constexpr int kBlockValues = 128;
inline constexpr int kBlockBytes  = 28;
inline constexpr int kBlockWords  = kBlockBytes / 4;
// Columns this kernel takes in one launch.
inline constexpr int kMaxColumns  = 8;

// The digits still to come of four bytes, each byte as ((byte * 3^t) mod 256) in a 16-bit lane:
// bytes 0 and 1 in `lo`, bytes 2 and 3 in `hi`. A lane times three stays below 2^16.
struct Walk {
    std::uint32_t lo, hi;
};

__device__ __forceinline__ Walk start(std::uint32_t word) {
    return {__byte_perm(word, 0, 0x4140), __byte_perm(word, 0, 0x4342)};
}

// The next digit {0, 1, 2} of each byte, in the bytes' order. `Last`: the walk ends here.
template <bool Last = false>
__device__ __forceinline__ std::uint32_t next(Walk& w) {
    const std::uint32_t lo = w.lo * 3u, hi = w.hi * 3u;
    if constexpr (!Last) {
        w.lo = lo & 0x00FF00FFu;
        w.hi = hi & 0x00FF00FFu;
    }
    return __byte_perm(lo, hi, 0x7531);
}

// Digits {0, 1, 2} as the signed bytes {-1, 0, 1} (the tensor-core kernel's A fragments): no byte
// carries, since d + 0x7F <= 0x81.
__device__ __forceinline__ int to_signed(std::uint32_t digits) {
    return int((digits + 0x7F7F7F7Fu) ^ 0x80808080u);
}

// One row's block: the walks of its six qs words and of its two qh bytes, and its scale d.
struct Block {
    Walk a[4], b[2];
    std::uint32_t h; // qh0 and qh1 in 16-bit lanes
    float d;
};

__device__ __forceinline__ void load_block(const std::uint8_t* at, std::uint32_t (&w)[kBlockWords]) {
    const auto* words = reinterpret_cast<const unsigned int*>(at);
#pragma unroll
    for (int i = 0; i < kBlockWords; ++i) { w[i] = __ldg(words + i); }
}

__device__ __forceinline__ Block begin_block(const std::uint32_t (&w)[kBlockWords]) {
    Block blk;
#pragma unroll
    for (int g = 0; g < 4; ++g) { blk.a[g] = start(w[g]); }
#pragma unroll
    for (int g = 0; g < 2; ++g) { blk.b[g] = start(w[4 + g]); }
    blk.h = __byte_perm(w[6], 0, 0x4140);
    blk.d = __half2float(__ushort_as_half(static_cast<unsigned short>(w[6] >> 16)));
    return blk;
}

// The eight words of slice S (values 32 S .. 32 S + 31) of a block, digits {0, 1, 2} in value order.
// Slices are taken in order 0, 1, 2, 3: each continues the walks the one before left.
template <int S>
__device__ __forceinline__ void slice_words(Block& blk, std::uint32_t (&w)[8]) {
    if constexpr (S == 0 || S == 1) {
        // qs[0..15] digits 2 S and 2 S + 1: values 32 S + 4 g .. and 32 S + 16 + 4 g ..
#pragma unroll
        for (int g = 0; g < 4; ++g) { w[g] = next(blk.a[g]); }
#pragma unroll
        for (int g = 0; g < 4; ++g) { w[4 + g] = next(blk.a[g]); }
    } else if constexpr (S == 2) {
        // qs[0..15] digit 4 (values 64 ..), then qs[16..23] digits 0 and 1 (values 80 .. 95).
#pragma unroll
        for (int g = 0; g < 4; ++g) { w[g] = next<true>(blk.a[g]); }
        w[4] = next(blk.b[0]);
        w[5] = next(blk.b[1]);
        w[6] = next(blk.b[0]);
        w[7] = next(blk.b[1]);
    } else {
        // qs[16..23] digits 2, 3 and 4 (values 96 .. 119), then qh digits 0 .. 3 (values 120 ..):
        // value 120 + 2 t + h is digit t of qh byte h.
        w[0] = next(blk.b[0]);
        w[1] = next(blk.b[1]);
        w[2] = next(blk.b[0]);
        w[3] = next(blk.b[1]);
        w[4] = next<true>(blk.b[0]);
        w[5] = next<true>(blk.b[1]);
        std::uint32_t h      = blk.h;
        const std::uint32_t p0 = h * 3u;
        h                      = p0 & 0x00FF00FFu;
        const std::uint32_t p1 = h * 3u;
        h                      = p1 & 0x00FF00FFu;
        const std::uint32_t p2 = h * 3u;
        h                      = p2 & 0x00FF00FFu;
        const std::uint32_t p3 = h * 3u;
        w[6]                   = __byte_perm(p0, p1, 0x7531);
        w[7]                   = __byte_perm(p2, p3, 0x7531);
    }
}

__device__ __forceinline__ int dot8(const std::uint32_t (&w)[8], const int (&a)[8]) {
    int acc = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) { acc = __dp4a(int(w[i]), a[i], acc); }
    return acc;
}

__device__ __forceinline__ int component(const int4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// Whether each slice loads its own (d, sum) pair rather than taking it from one load of the block's
// four. Bonsai 2.5 PTQ1_0's short chats with no drafter, a slice's own pairs against a block's: RTX
// 5090 188.5 against 180.3 tok/s, RTX PRO 6000 187.6 against 186.6, RTX 4090 125.6 either way,
// RTX 3090 90.1 against 91.6.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
inline constexpr bool kSliceScales = true;
#else
inline constexpr bool kSliceScales = false;
#endif

// Slice S of block `blk_index` for both rows of the set, into part[r][j]. Without kSliceScales,
// scales[j] holds the block's four (d, exact sum) pairs of column j.
template <int T, int S>
__device__ __forceinline__ void slice(const VecArgs& p, int planes, int blk_index, Block (&blk)[2],
                                      const int4 (&scales)[T], float (&part)[2][T]) {
    int a[T][8];
    float d[T];
    int sum[T];
#pragma unroll
    for (int j = 0; j < T; ++j) {
        // The slice's (d, exact sum) pair: d in the low 16 bits, the sum (int16) in the high ones.
        int pair;
        if constexpr (kSliceScales) {
            pair = __ldg(reinterpret_cast<const int*>(p.ds) + std::int64_t(j) * 4 * planes +
                         4 * blk_index + S);
        } else {
            pair = component(scales[j], S);
        }
        d[j]   = __half2float(__ushort_as_half(static_cast<unsigned short>(pair & 0xFFFF)));
        sum[j] = pair >> 16;
        // Values 32 S .. 32 S + 15 and 32 S + 16 .. 32 S + 31: planes 2 S and 2 S + 1.
        const auto* q = reinterpret_cast<const int4*>(p.qs + std::int64_t(j) * p.k) + blk_index;
        const int4 v0 = __ldg(q + (2 * S) * planes), v1 = __ldg(q + (2 * S + 1) * planes);
        a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
        a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        std::uint32_t w[8];
        slice_words<S>(blk[r], w);
#pragma unroll
        for (int j = 0; j < T; ++j) {
            part[r][j] = fmaf(d[j], float(dot8(w, a[j]) - sum[j]), part[r][j]);
        }
    }
}

// Warps an SM below which a product's row sets take more lanes each: a lane's blocks follow one
// another, and too few warps leave the SM waiting on them (RTX 5090, the untransposed activation:
// K = 17408 over 5120 rows took 14.9 us at 16 lanes a row set, 20.9 us at 8, 16.1 us at 32; 10240
// rows over K = 5120, 9.9 us at 8 and 10.1 us at 16).
inline constexpr int kMinWarpsPerSm = 6;
// Blocks a lane takes in turn above which a row set takes more lanes, whatever the warps (RTX 3090:
// K = 17408 over 5120 rows, 17 blocks a lane at 8 lanes and 7.8 warps an SM, takes 41.0 us against
// 35.8 at 16 lanes).
inline constexpr int kMaxLaneBlocks = 12;

// Lanes a row set takes: 8, doubled up to 32 (and at most the row's blocks) while the product's row
// sets would fill fewer than kMinWarpsPerSm warps an SM or a lane would take more than
// kMaxLaneBlocks blocks. A product takes the same lanes alone and in a launch of several
// (jobs_launch), so its sums keep their bits.
[[nodiscard]] inline int group_lanes(int k, std::int64_t sets) {
    const int blocks        = k / kBlockValues;
    const std::int64_t want = std::int64_t(kMinWarpsPerSm) * device_facts().sm_count;
    int group               = 8;
    while (group < 32 && 2 * group <= blocks &&
           (sets * group / 32 < want || blocks > kMaxLaneBlocks * group)) {
        group *= 2;
    }
    return group;
}

[[nodiscard]] __host__ __device__ inline int row_sets(int rows, bool fused) {
    return fused ? rows : (rows + 1) / 2;
}

// The two weight rows of row set `set`: a fused product's gate row and up row, else rows 2 set and
// 2 set + 1 (the last row twice when the rows are odd).
template <bool Fused>
__device__ __forceinline__ void set_rows(const VecArgs& p, int set, int (&row)[2]) {
    if constexpr (Fused) {
        row[0] = set;
        row[1] = p.rows + set;
    } else {
        row[0] = 2 * set;
        row[1] = min(2 * set + 1, p.rows - 1);
    }
}

// A dependent launch's first weight lines before it waits (sm_90+), as vec::prefetch_first_rows.
template <bool Fused>
__device__ __forceinline__ void prefetch_first_rows(const VecArgs& p, int block, int group) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int per  = 32 / group;
    const int sets = row_sets(p.rows, Fused);
    const int set0 = (block * kWarps + warp) * per;
    for (int set = set0; set < min(set0 + per, sets); ++set) {
        int row[2];
        set_rows<Fused>(p, set, row);
#    pragma unroll
        for (int r = 0; r < 2; ++r) {
            const std::uint8_t* base = p.weight + std::int64_t(row[r]) * p.row_bytes;
            for (std::int64_t o = std::int64_t(lane) * 128; o < p.row_bytes; o += 32 * 128) {
                asm volatile("prefetch.global.L2 [%0];" ::"l"(base + o));
            }
        }
    }
#else
    (void)p;
    (void)block;
    (void)group;
#endif
}

// The row sets block `block` of `blocks` takes: each warp 32 / group of them at a time.
template <int T, bool Fused>
__device__ __forceinline__ void rows(const VecArgs& p, int block, int blocks, int group) {
    const int warp   = threadIdx.x >> 5;
    const int lane   = threadIdx.x & 31;
    const int per    = 32 / group;
    const int q      = lane % group;
    const int nb     = p.k / kBlockValues;
    const int sets   = row_sets(p.rows, Fused);
    for (int set0 = (block * kWarps + warp) * per; set0 < sets; set0 += blocks * kWarps * per) {
        const int mine = set0 + lane / group;
        int row[2];
        set_rows<Fused>(p, min(mine, sets - 1), row);
        const std::uint8_t* base[2] = {p.weight + std::int64_t(row[0]) * p.row_bytes,
                                       p.weight + std::int64_t(row[1]) * p.row_bytes};
        float acc[2][T];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) { acc[r][j] = 0.0f; }
        }
        std::uint32_t cur[2][kBlockWords];
        if (q < nb) {
#pragma unroll
            for (int r = 0; r < 2; ++r) { load_block(base[r] + std::int64_t(q) * kBlockBytes, cur[r]); }
        }
        for (int b = q; b < nb; b += group) {
            // The next block's words are in flight while this one is decoded.
            std::uint32_t nxt[2][kBlockWords];
            if (b + group < nb) {
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    load_block(base[r] + std::int64_t(b + group) * kBlockBytes, nxt[r]);
                }
            }
            Block blk[2] = {begin_block(cur[0]), begin_block(cur[1])};
            float part[2][T];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
#pragma unroll
                for (int j = 0; j < T; ++j) { part[r][j] = 0.0f; }
            }
            int4 scales[T];
#pragma unroll
            for (int j = 0; j < T; ++j) {
                scales[j] = kSliceScales
                                ? make_int4(0, 0, 0, 0)
                                : __ldg(reinterpret_cast<const int4*>(p.ds) + std::int64_t(j) * nb + b);
            }
            slice<T, 0>(p, nb, b, blk, scales, part);
            slice<T, 1>(p, nb, b, blk, scales, part);
            slice<T, 2>(p, nb, b, blk, scales, part);
            slice<T, 3>(p, nb, b, blk, scales, part);
#pragma unroll
            for (int r = 0; r < 2; ++r) {
#pragma unroll
                for (int j = 0; j < T; ++j) { acc[r][j] = fmaf(blk[r].d, part[r][j], acc[r][j]); }
            }
            if (b + group < nb) {
#pragma unroll
                for (int r = 0; r < 2; ++r) {
#pragma unroll
                    for (int i = 0; i < kBlockWords; ++i) { cur[r][i] = nxt[r][i]; }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) {
                for (int o = group / 2; o > 0; o >>= 1) {
                    acc[r][j] += __shfl_xor_sync(0xFFFFFFFFu, acc[r][j], o);
                }
            }
        }
        if (mine >= sets) { continue; }
        // Lane q of the set stores the columns j = q mod group.
#pragma unroll
        for (int j = 0; j < T; ++j) {
            if (j % group != q) { continue; }
            if constexpr (Fused) {
                vec_store(p.out, mine, j, vec_gate(acc[0][j], p.out.activation) * acc[1][j]);
            } else {
                vec_store(p.out, 2 * mine, j, acc[0][j]);
                if (2 * mine + 1 < p.rows) { vec_store(p.out, 2 * mine + 1, j, acc[1][j]); }
            }
        }
    }
}

template <int T, bool Fused>
__launch_bounds__(kWarps * 32) __global__ void kernel(VecArgs p, int group) {
    if (p.prefetch) { prefetch_first_rows<Fused>(p, int(blockIdx.x), group); }
    ninfer::pdl::arch_enter();
    rows<T, Fused>(p, blockIdx.x, gridDim.x, group);
}

// Each product's lanes a row set, as it would take them alone.
struct Groups {
    int lanes[kMaxVecJobs];
};

// Up to kMaxVecJobs products of one K in one launch: blocks [p.first[j], p.first[j + 1]) take
// product j's row sets as kernel() takes a product's.
template <int T>
__launch_bounds__(kWarps * 32) __global__ void jobs_kernel(VecJobs p, Groups groups) {
    const int b = blockIdx.x;
    int j       = 0;
#pragma unroll
    for (int i = 1; i < kMaxVecJobs; ++i) {
        if (i < p.count && b >= p.first[i]) { j = i; }
    }
    VecArgs job = p.job[0];
    int first = p.first[0], last = p.first[1], group = groups.lanes[0];
#pragma unroll
    for (int i = 1; i < kMaxVecJobs; ++i) {
        if (i == j) {
            job   = p.job[i];
            first = p.first[i];
            last  = p.first[i + 1];
            group = groups.lanes[i];
        }
    }
    if (job.prefetch) { prefetch_first_rows<false>(job, b - first, group); }
    ninfer::pdl::arch_enter();
    rows<T, false>(job, b - first, last - first, group);
}

template <class Kernel>
int resident_blocks(Kernel kernel) {
    int blocks = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, kWarps * 32, 0),
          "ptq1 vector occupancy");
    return std::max(blocks, 1);
}

[[nodiscard]] inline int launch_blocks(int rows, bool fused, int group) {
    const int per_block = kWarps * (32 / group);
    return (row_sets(rows, fused) + per_block - 1) / per_block;
}

template <int T, bool Fused>
void launch(const VecArgs& args, cudaStream_t stream) {
    static const int resident = resident_blocks(kernel<T, Fused>);
    const int group           = group_lanes(args.k, row_sets(args.rows, Fused));
    const int blocks =
        std::min(launch_blocks(args.rows, Fused, group), resident * device_facts().sm_count);
    VecArgs a   = args;
    a.prefetch  = ninfer::pdl::arch_dependent(stream);
    ninfer::pdl::launch_arch(blocks, kWarps * 32, 0, stream, kernel<T, Fused>, a, group);
    check(cudaGetLastError(), "ptq1 vector product launch");
}

// Products share a launch while their blocks fit kJobsBlocksPerSm per SM, as the vector
// kernel's jobs_launch; a product left alone takes the one-product kernel.
template <int T>
void jobs_launch(const VecJobs& args, cudaStream_t stream) {
    static const int resident = resident_blocks(jobs_kernel<T>);
    int group[kMaxVecJobs]    = {};
    for (int i = 0; i < args.count; ++i) {
        group[i] = group_lanes(args.job[i].k, row_sets(args.job[i].rows, false));
    }
    const int budget          = kJobsBlocksPerSm * device_facts().sm_count;
    const int cap             = resident * device_facts().sm_count;
    const bool prefetch       = ninfer::pdl::arch_dependent(stream);
    for (int begin = 0; begin < args.count;) {
        int end    = begin + 1;
        int blocks = launch_blocks(args.job[begin].rows, false, group[begin]);
        while (end < args.count) {
            const int more = launch_blocks(args.job[end].rows, false, group[end]);
            if (blocks + more > budget) { break; }
            blocks += more;
            ++end;
        }
        if (end - begin == 1) {
            launch<T, false>(args.job[begin], stream);
            begin = end;
            continue;
        }
        VecJobs part{};
        Groups lanes{};
        part.count = end - begin;
        int want[kMaxVecJobs] = {}, total = 0;
        for (int i = 0; i < part.count; ++i) {
            part.job[i]          = args.job[begin + i];
            part.job[i].prefetch = prefetch;
            lanes.lanes[i]       = group[begin + i];
            want[i]              = launch_blocks(part.job[i].rows, false, lanes.lanes[i]);
            total += want[i];
        }
        part.first[0] = 0;
        for (int i = 0; i < part.count; ++i) {
            const int share = total <= cap ? want[i]
                                           : std::max(1, int(std::int64_t(want[i]) * cap / total));
            part.first[i + 1] = part.first[i] + share;
        }
        ninfer::pdl::launch_arch(part.first[part.count], kWarps * 32, 0, stream, jobs_kernel<T>,
                                 part, lanes);
        check(cudaGetLastError(), "ptq1 vector products launch");
        begin = end;
    }
}

template <int T>
void launch_columns(const VecJobs& args, bool fused, cudaStream_t stream) {
    if (fused) {
        launch<T, true>(args.job[0], stream);
    } else if (args.count > 1) {
        jobs_launch<T>(args, stream);
    } else {
        launch<T, false>(args.job[0], stream);
    }
}

} // namespace ninfer::ops::gguf::detail::ptq1
