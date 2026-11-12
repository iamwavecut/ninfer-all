#include "ops/linear/stq/stq_linear.h"

#include "core/device_launch.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::stq {
namespace {

constexpr int kTransform      = 128;
constexpr float kNorm         = 0.08838834764831845F; // 1 / sqrt(128)
constexpr int kTransformWarps = 4;                    // transform blocks of a prepare/finish CTA
constexpr int kTileWarps      = 8;                    // column tiles of a partial CTA
constexpr unsigned kFull      = 0xFFFFFFFFU;

std::size_t round_up(std::size_t value) { return (value + 255) / 256 * 256; }

// The binary16 sum of a word's two halves (rounded once), times 4096: the codebooks' weight.
__device__ __forceinline__ float half_sum_scaled(std::uint32_t halves) {
    const float lo = __half2float(__ushort_as_half(static_cast<unsigned short>(halves & 0xFFFFU)));
    const float hi = __half2float(__ushort_as_half(static_cast<unsigned short>(halves >> 16)));
    return __half2float(__float2half_rn(lo + hi)) * 4096.0F;
}

// The unnormalized Walsh-Hadamard transform of 128 values a warp holds four to a lane (lane l
// elements 4l..4l+3), in Sylvester order.
__device__ __forceinline__ void transform(float v[4], int lane) {
    const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float other = __shfl_xor_sync(kFull, v[q], m);
            v[q]              = (lane & m) ? other - v[q] : v[q] + other;
        }
    }
}

__device__ __forceinline__ int matrix_of(const Product& p, int item) {
    return p.ids != nullptr ? p.ids[item] : item;
}

// xh [item][row][k] = H(suh * x) / sqrt(128).
__global__ void prepare_kernel(Product p, float* xh) {
    const int warp = int(threadIdx.x) >> 5, lane = int(threadIdx.x) & 31;
    const int block = int(blockIdx.x) * kTransformWarps + warp;
    if (block * kTransform >= p.m.k) { return; }
    const int row = int(blockIdx.y), item = int(blockIdx.z);
    const int column = block * kTransform + 4 * lane;
    const __nv_bfloat16* x = p.x + std::int64_t(item / p.x_divisor) * p.x_item_stride +
                             std::int64_t(row) * p.x_row_stride + column;
    const __half* suh = p.m.suh + std::int64_t(matrix_of(p, item)) * p.m.suh_stride + column;
    float v[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) { v[q] = __bfloat162float(x[q]) * __half2float(suh[q]); }
    transform(v, lane);
    float* out = xh + (std::int64_t(item) * p.rows + row) * p.m.k + column;
    *reinterpret_cast<float4*>(out) = make_float4(v[0] * kNorm, v[1] * kNorm, v[2] * kNorm, v[3] * kNorm);
}

// partial [split][item][row][n]: the inner products of xh with the decoded tiles of input tiles
// [split * per_split, (split + 1) * per_split). A warp owns one column tile; lane l decodes its
// eight slots of each tile and the four lanes of a column pair sum their rows at the end.
template <int Words, bool Pair, int Rows>
__global__ void __launch_bounds__(kTileWarps * 32)
    partial_kernel(Product p, const float* xh, float* partial, int per_split) {
    const int warp = int(threadIdx.x) >> 5, lane = int(threadIdx.x) & 31;
    const int column_tiles = p.m.n / 16;
    const int tile_column  = int(blockIdx.x) * kTileWarps + warp;
    if (tile_column >= column_tiles) { return; }
    const int split = int(blockIdx.y), item = int(blockIdx.z);
    const int first = split * per_split, last = min(p.m.k / 16, first + per_split);
    constexpr int kPairs = Words / 2; // u32 words of a tile
    const auto* tiles    = reinterpret_cast<const std::uint32_t*>(
        p.m.trellis + std::int64_t(matrix_of(p, item)) * p.m.trellis_stride);
    const float* xi = xh + std::int64_t(item) * Rows * p.m.k;
    // The lane's windows lie in the 64 bits of the u32 words lo - 1 and lo (lo wrapping to the
    // stream's end), the first bit of word lo - 1 at stream bit 32 (lo - 1).
    const int lo       = (lane * Words) / 64;
    const int hi       = (lo + kPairs - 1) % kPairs;
    const int base_bit = 32 * (lo - 1);
    const int r0       = 2 * (lane & 3);
    float acc0[Rows], acc1[Rows];
#pragma unroll
    for (int r = 0; r < Rows; ++r) { acc0[r] = acc1[r] = 0.0F; }
    for (int tile = first; tile < last; ++tile) {
        const std::uint32_t* words = tiles + (std::int64_t(tile) * column_tiles + tile_column) * kPairs;
        const std::uint32_t own    = lane < kPairs ? __ldg(words + lane) : 0U;
        const std::uint64_t bits   = (std::uint64_t(__shfl_sync(kFull, own, hi)) << 32) |
                                   __shfl_sync(kFull, own, lo);
        float w[8];
        if constexpr (!Pair) {
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const int end         = ((8 * lane + s + 1) * Words) >> 4;
                const std::uint32_t c = std::uint32_t(bits >> (64 - end + base_bit)) & 0x7FFFU;
                w[s]                  = half_sum_scaled((c * 0x3FED4CEFU) & 0x8FF08FF0U);
            }
        } else {
#pragma unroll
            for (int s = 1; s < 8; s += 2) {
                const int end         = ((8 * lane + s + 1) * Words) >> 4;
                const std::uint32_t c = std::uint32_t(bits >> (64 - end + base_bit)) & 0xFFFFU;
                const std::uint32_t m = c * 0x3B22BE15U;
                w[s - 1]              = half_sum_scaled(m & 0x8ED78C13U);
                w[s]                  = half_sum_scaled(__funnelshift_l(m, m, 3) & 0x8ECE8C03U);
            }
        }
        const float* xt = xi + tile * 16 + r0;
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            const float2 a = *reinterpret_cast<const float2*>(xt + std::int64_t(r) * p.m.k);
            const float2 b = *reinterpret_cast<const float2*>(xt + std::int64_t(r) * p.m.k + 8);
            acc0[r] += a.x * w[0] + a.y * w[1] + b.x * w[2] + b.y * w[3];
            acc1[r] += a.x * w[4] + a.y * w[5] + b.x * w[6] + b.y * w[7];
        }
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        acc0[r] += __shfl_xor_sync(kFull, acc0[r], 1);
        acc0[r] += __shfl_xor_sync(kFull, acc0[r], 2);
        acc1[r] += __shfl_xor_sync(kFull, acc1[r], 1);
        acc1[r] += __shfl_xor_sync(kFull, acc1[r], 2);
    }
    if ((lane & 3) == 0) {
        float* out = partial + (std::int64_t(split) * p.items + item) * Rows * p.m.n +
                     tile_column * 16 + (lane >> 2);
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            out[std::int64_t(r) * p.m.n]     = acc0[r];
            out[std::int64_t(r) * p.m.n + 8] = acc1[r];
        }
    }
}

// out = svh * H(sum of the splits' partials) / sqrt(128), the splits summed in order.
__global__ void finish_kernel(Product p, const float* partial, int splits) {
    const int warp = int(threadIdx.x) >> 5, lane = int(threadIdx.x) & 31;
    const int block = int(blockIdx.x) * kTransformWarps + warp;
    if (block * kTransform >= p.m.n) { return; }
    const int row = int(blockIdx.y), item = int(blockIdx.z);
    const int column = block * kTransform + 4 * lane;
    float v[4]       = {0.0F, 0.0F, 0.0F, 0.0F};
    for (int s = 0; s < splits; ++s) {
        const float4 z = *reinterpret_cast<const float4*>(
            partial + ((std::int64_t(s) * p.items + item) * p.rows + row) * p.m.n + column);
        v[0] += z.x;
        v[1] += z.y;
        v[2] += z.z;
        v[3] += z.w;
    }
    transform(v, lane);
    const __half* svh = p.m.svh + std::int64_t(matrix_of(p, item)) * p.m.svh_stride + column;
    const std::int64_t at = std::int64_t(item) * p.out_item_stride +
                            std::int64_t(row) * p.out_row_stride + column;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float y = v[q] * kNorm * __half2float(svh[q]);
        if (p.out_bf16 != nullptr) {
            p.out_bf16[at + q] = __float2bfloat16(y);
        } else {
            p.out[at + q] = y;
        }
    }
}

struct Split {
    int per_split = 0, splits = 0;
};

// Enough warps to fill the device: input tiles split over CTAs while the column tiles alone leave
// fewer than 16 warps per multiprocessor, at most 16 ways.
Split split_of(const Product& p, int sms) {
    const long long tiles  = p.m.k / 16;
    const long long warps  = (long long)(p.m.n / 16) * p.items;
    const long long target = (long long)std::max(sms, 1) * 16;
    const long long ways   = std::clamp<long long>((target + warps - 1) / warps, 1,
                                                   std::min<long long>(16, tiles));
    Split out;
    out.per_split = int((tiles + ways - 1) / ways);
    out.splits    = int((tiles + out.per_split - 1) / out.per_split);
    return out;
}

template <int Words, bool Pair>
void launch_partial(const Product& p, const float* xh, float* partial, const Split& split,
                    cudaStream_t stream) {
    const dim3 grid((unsigned)((p.m.n / 16 + kTileWarps - 1) / kTileWarps), (unsigned)split.splits,
                    (unsigned)p.items);
    const dim3 block(kTileWarps * 32);
    switch (p.rows) {
#define NINFER_STQ_ROWS(R)                                                                         \
    case R:                                                                                         \
        partial_kernel<Words, Pair, R><<<grid, block, 0, stream>>>(p, xh, partial, split.per_split); \
        break;
        NINFER_STQ_ROWS(1)
        NINFER_STQ_ROWS(2)
        NINFER_STQ_ROWS(3)
        NINFER_STQ_ROWS(4)
        NINFER_STQ_ROWS(5)
        NINFER_STQ_ROWS(6)
        NINFER_STQ_ROWS(7)
        NINFER_STQ_ROWS(8)
#undef NINFER_STQ_ROWS
    default:
        throw std::invalid_argument("STQ product: 1..8 rows per item");
    }
}

} // namespace

void validate(const Product& p) {
    if (p.m.trellis == nullptr || p.m.suh == nullptr || p.m.svh == nullptr || p.x == nullptr ||
        (p.out == nullptr) == (p.out_bf16 == nullptr)) {
        throw std::invalid_argument("STQ product: missing matrices, input or output");
    }
    if (p.m.k <= 0 || p.m.n <= 0 || p.m.k % kTransform || p.m.n % kTransform) {
        throw std::invalid_argument("STQ product: k and n must be whole 128-element blocks");
    }
    if (p.rows < 1 || p.rows > kMaxRows || p.items < 1 || p.x_divisor < 1) {
        throw std::invalid_argument("STQ product: 1..8 rows of at least one item");
    }
    if (reinterpret_cast<std::uintptr_t>(p.m.trellis) % 4 || p.m.trellis_stride % 2) {
        throw std::invalid_argument("STQ product: trellis words must start on 4-byte bounds");
    }
}

std::size_t workspace_bytes(const Product& p, int sms) {
    const Split split = split_of(p, sms);
    return round_up(std::size_t(p.items) * p.rows * p.m.k * 4) +
           round_up(std::size_t(split.splits) * p.items * p.rows * p.m.n * 4);
}

void product(const Product& p, void* workspace, int sms, cudaStream_t stream) {
    validate(p);
    const Split split = split_of(p, sms);
    auto* xh          = static_cast<float*>(workspace);
    auto* partial = reinterpret_cast<float*>(static_cast<std::byte*>(workspace) +
                                             round_up(std::size_t(p.items) * p.rows * p.m.k * 4));
    const dim3 block(kTransformWarps * 32);
    const auto blocks = [](int n) {
        return (unsigned)((n / kTransform + kTransformWarps - 1) / kTransformWarps);
    };
    prepare_kernel<<<dim3(blocks(p.m.k), (unsigned)p.rows, (unsigned)p.items), block, 0, stream>>>(p, xh);
    CUDA_CHECK(cudaGetLastError());
    if (p.m.codebook == Codebook::Spec1W15N64) {
        launch_partial<64, false>(p, xh, partial, split, stream);
    } else {
        launch_partial<32, true>(p, xh, partial, split, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    finish_kernel<<<dim3(blocks(p.m.n), (unsigned)p.rows, (unsigned)p.items), block, 0, stream>>>(
        p, partial, split.splits);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::stq
