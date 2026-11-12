#include "models/deepseek_v41/kernels.h"

#include "core/device_launch.h"
#include "ops/common/route_order.cuh"
#include "ops/common/silu.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::models::deepseek_v41::kernels {
namespace {

void check_launch(const char* what) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("deepseek_v41 ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

unsigned blocks_for(std::int64_t n, int threads) {
    return static_cast<unsigned>((n + threads - 1) / threads);
}

constexpr int kLatent = 512; // the attention latent's width
constexpr int kRope   = 64;  // its rotated tail
constexpr int kIndex  = 128; // the indexer's head width

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFu, v, o); }
    return v;
}
__device__ __forceinline__ int warp_count(int v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFu, v, o); }
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o)); }
    return v;
}

// Sum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets the total.
__device__ float block_sum(float v) {
    __shared__ float partial[32];
    __shared__ float total;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) { partial[warp] = v; }
    __syncthreads();
    if (warp == 0) {
        float s = lane < int(blockDim.x >> 5) ? partial[lane] : 0.0f;
        s = warp_sum(s);
        if (lane == 0) { total = s; }
    }
    __syncthreads();
    return total;
}

__device__ float block_max(float v) {
    __shared__ float partial[32];
    __shared__ float total;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) { partial[warp] = v; }
    __syncthreads();
    if (warp == 0) {
        float m = lane < int(blockDim.x >> 5) ? partial[lane] : -INFINITY;
        m = warp_max(m);
        if (lane == 0) { total = m; }
    }
    __syncthreads();
    return total;
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }

__device__ __forceinline__ float4 bf16r4(float4 v) {
    return make_float4(bf16r(v.x), bf16r(v.y), bf16r(v.z), bf16r(v.w));
}

__device__ __forceinline__ void st_bf16x4(__nv_bfloat16* p, float4 v) {
    const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y);
    const __nv_bfloat162 hi = __floats2bfloat162_rn(v.z, v.w);
    uint2 w;
    w.x = *reinterpret_cast<const unsigned*>(&lo);
    w.y = *reinterpret_cast<const unsigned*>(&hi);
    *reinterpret_cast<uint2*>(p) = w;
}

__device__ __forceinline__ float4 ld_bf16x4(const __nv_bfloat16* p) {
    const uint2 w = *reinterpret_cast<const uint2*>(p);
    const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.x));
    const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.y));
    return make_float4(lo.x, lo.y, hi.x, hi.y);
}

__device__ __forceinline__ float amax4(float4 v) {
    return fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
}

// The maximum over `lanes` consecutive lanes (aligned groups; every lane of the warp calls it).
template <int Lanes>
__device__ __forceinline__ float group_max(float v) {
#pragma unroll
    for (int o = 1; o < Lanes; o <<= 1) { v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o)); }
    return v;
}

__device__ __forceinline__ float pow2_ceil(float x) {
    const unsigned b = __float_as_uint(x);
    const int e      = int((b >> 23) & 0xFFu) - 127 + ((b & 0x7FFFFFu) != 0 ? 1 : 0);
    return __uint_as_float(unsigned(e + 127) << 23);
}

// fp8 (E4M3, ue8m0 scale) of a 32-value group held by 8 aligned lanes, four values each: the
// rounded codes' values in `r` and the group's scale in `s`; the result is r * s.
__device__ __forceinline__ float4 fp8_group(float4 v, float4& r, float& s) {
    const float amax = group_max<8>(amax4(v));
    s                = pow2_ceil(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
    const float d    = s;
    const auto q = [d](float x) { return round_e4m3(fminf(fmaxf(x / d, -448.0f), 448.0f)); };
    r = make_float4(q(v.x), q(v.y), q(v.z), q(v.w));
    return make_float4(r.x * s, r.y * s, r.z * s, r.w * s);
}
__device__ __forceinline__ float4 fp8_group(float4 v) {
    float4 r;
    float s;
    return fp8_group(v, r, s);
}

// fp4 (E2M1, E8M0 scale) of a 32-value group held by 8 aligned lanes.
__device__ __forceinline__ float4 fp4_e8m0_group(float4 v, float4& r, float& s) {
    const float amax = group_max<8>(amax4(v));
    s                = pow2_ceil(fmaxf(amax, 6.0f * 1.17549435e-38f) * (1.0f / 6.0f));
    const float d    = s;
    const auto q = [d](float x) { return round_e2m1(fminf(fmaxf(x / d, -6.0f), 6.0f)); };
    r = make_float4(q(v.x), q(v.y), q(v.z), q(v.w));
    return make_float4(r.x * s, r.y * s, r.z * s, r.w * s);
}
__device__ __forceinline__ float4 fp4_e8m0_group(float4 v) {
    float4 r;
    float s;
    return fp4_e8m0_group(v, r, s);
}

// fp4 (E2M1, E4M3 scale) of a 16-value group held by 4 aligned lanes.
__device__ __forceinline__ float4 fp4_e4m3_group(float4 v, float4& r, float& s) {
    const float amax = group_max<4>(amax4(v));
    s = round_e4m3(fminf(fmaxf(amax, 6.0f * 0.001953125f) / 6.0f, 448.0f));
    const float d = s;
    const auto q = [d](float x) { return round_e2m1(fminf(fmaxf(x / d, -6.0f), 6.0f)); };
    r = make_float4(q(v.x), q(v.y), q(v.z), q(v.w));
    return make_float4(r.x * s, r.y * s, r.z * s, r.w * s);
}

// ---- Packed caches ------------------------------------------------------------------------------
// The caches keep the codes of the roundings above and their scales; a value is its code's value
// times its group's scale, exactly the BF16 value the rounding gave (an E4M3 or E2M1 value times a
// power of two, or an E2M1 value times an E4M3 scale, has at most six significant bits).

// The E4M3 code of an E4M3 value (|r| <= 448), the sign kept for zeros.
__device__ __forceinline__ unsigned e4m3_code(float r) {
    const unsigned bits = __float_as_uint(r);
    const unsigned sign = (bits >> 24) & 0x80u;
    const float a       = fabsf(r);
    if (a < 0.015625f) { return sign | unsigned(a * 512.0f); } // subnormal: m * 2^-9
    const unsigned e = ((bits >> 23) & 0xFFu) - 127u + 7u;
    return sign | (e << 3) | ((bits >> 20) & 7u);
}
__device__ __forceinline__ float e4m3_value(unsigned c) {
    const unsigned e = (c >> 3) & 15u, m = c & 7u;
    const float a    = e == 0 ? float(m) * 0.001953125f
                              : __uint_as_float(((e - 7u + 127u) << 23) | (m << 20));
    return (c & 0x80u) != 0 ? -a : a;
}

// The E2M1 code of an E2M1 value (|r| <= 6).
__device__ __forceinline__ unsigned e2m1_code(float r) {
    const unsigned bits = __float_as_uint(r);
    const unsigned sign = (bits >> 28) & 8u;
    const float a       = fabsf(r);
    if (a < 1.0f) { return sign | (a == 0.5f ? 1u : 0u); }
    const unsigned e = ((bits >> 23) & 0xFFu) - 127u + 1u;
    return sign | (e << 1) | ((bits >> 22) & 1u);
}
__device__ __forceinline__ float e2m1_value(unsigned n) {
    const unsigned e = (n >> 1) & 3u, m = n & 1u;
    const float a = e == 0 ? (m != 0 ? 0.5f : 0.0f) : __uint_as_float(((e - 1u + 127u) << 23) | (m << 22));
    return (n & 8u) != 0 ? -a : a;
}

// A power-of-two scale as its E8M0 code (the float's exponent field) and back.
__device__ __forceinline__ unsigned e8m0_code(float s) { return __float_as_uint(s) >> 23; }
__device__ __forceinline__ float e8m0_value(unsigned c) { return __uint_as_float(c << 23); }

__device__ __forceinline__ unsigned e4m3_codes4(float4 r) {
    return e4m3_code(r.x) | (e4m3_code(r.y) << 8) | (e4m3_code(r.z) << 16) | (e4m3_code(r.w) << 24);
}
__device__ __forceinline__ unsigned e2m1_codes4(float4 r) {
    return e2m1_code(r.x) | (e2m1_code(r.y) << 4) | (e2m1_code(r.z) << 8) | (e2m1_code(r.w) << 12);
}

// Eight BF16 values of E4M3 codes (two words, value 0 in the low byte) times scale s.
__device__ __forceinline__ uint4 e4m3_bf16x8(uint2 codes, float s) {
    uint4 out;
    unsigned* w = &out.x;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned word = i < 2 ? codes.x : codes.y;
        const int shift     = (i & 1) * 16;
        const __nv_bfloat162 pair = __floats2bfloat162_rn(e4m3_value((word >> shift) & 0xFFu) * s,
                                                          e4m3_value((word >> (shift + 8)) & 0xFFu) * s);
        w[i] = *reinterpret_cast<const unsigned*>(&pair);
    }
    return out;
}
// Eight BF16 values of E2M1 codes (one word, value 0 in the low nibble) times scale s.
__device__ __forceinline__ uint4 e2m1_bf16x8(unsigned codes, float s) {
    uint4 out;
    unsigned* w = &out.x;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __nv_bfloat162 pair = __floats2bfloat162_rn(e2m1_value((codes >> (8 * i)) & 15u) * s,
                                                          e2m1_value((codes >> (8 * i + 4)) & 15u) * s);
        w[i] = *reinterpret_cast<const unsigned*>(&pair);
    }
    return out;
}

// The rotation of four values at offset `at` (a multiple of 4) of a row whose last kRope values of
// `dim` turn: pairs (2i, 2i + 1) of the tail by position * freq[i]; `inverse` turns back.
__device__ __forceinline__ float4 rope4(float4 v, int at, int dim, float position,
                                        const float* __restrict__ freq, bool inverse) {
    const int tail = at - (dim - kRope);
    if (tail < 0) { return v; }
    float s0, c0, s1, c1;
    sincosf(position * freq[tail / 2], &s0, &c0);
    sincosf(position * freq[tail / 2 + 1], &s1, &c1);
    if (inverse) {
        s0 = -s0;
        s1 = -s1;
    }
    return make_float4(v.x * c0 - v.y * s0, v.x * s0 + v.y * c0, v.z * c1 - v.w * s1,
                       v.z * s1 + v.w * c1);
}

// ---- Products -------------------------------------------------------------------------------

constexpr int kGemvThreads = 256;
constexpr int kGemvRows    = 8;
constexpr int kGemvAhead   = 4; // float4 steps whose loads a thread issues before their products

// Block m: output row m of N tokens (at most kGemvRows), its k values split over the block. A
// thread loads kGemvAhead steps of the row and the tokens before it multiplies them, in its own
// order: with one step in flight a block read a hyper-connection row of 20,480 values at about
// 8 GB/s, so the 24 rows of a mix took 10.4 us on an RTX 4090.
template <int N>
__global__ void gemv_kernel(const float* __restrict__ w, const float* __restrict__ x, int m, int k,
                            float* __restrict__ out) {
    __shared__ float part[kGemvThreads / 32][N];
    const int row = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float* wr = w + std::int64_t(row) * k;
    float dot[N] = {};
    if (k % 4 == 0) {
        const int quads = k / 4;
        const auto* w4  = reinterpret_cast<const float4*>(wr);
        int i           = threadIdx.x;
        for (; i + (kGemvAhead - 1) * kGemvThreads < quads; i += kGemvAhead * kGemvThreads) {
            float4 wv[kGemvAhead], xv[kGemvAhead][N];
#pragma unroll
            for (int a = 0; a < kGemvAhead; ++a) {
                wv[a] = w4[i + a * kGemvThreads];
#pragma unroll
                for (int t = 0; t < N; ++t) {
                    xv[a][t] =
                        reinterpret_cast<const float4*>(x + std::int64_t(t) * k)[i + a * kGemvThreads];
                }
            }
#pragma unroll
            for (int a = 0; a < kGemvAhead; ++a) {
#pragma unroll
                for (int t = 0; t < N; ++t) {
                    dot[t] += wv[a].x * xv[a][t].x + wv[a].y * xv[a][t].y + wv[a].z * xv[a][t].z +
                              wv[a].w * xv[a][t].w;
                }
            }
        }
        for (; i < quads; i += kGemvThreads) {
            const float4 wv = w4[i];
#pragma unroll
            for (int t = 0; t < N; ++t) {
                const float4 xv = reinterpret_cast<const float4*>(x + std::int64_t(t) * k)[i];
                dot[t] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
            }
        }
    } else {
        for (int i = threadIdx.x; i < k; i += kGemvThreads) {
#pragma unroll
            for (int t = 0; t < N; ++t) { dot[t] += wr[i] * x[std::int64_t(t) * k + i]; }
        }
    }
#pragma unroll
    for (int t = 0; t < N; ++t) {
        const float v = warp_sum(dot[t]);
        if (lane == 0) { part[warp][t] = v; }
    }
    __syncthreads();
    if (threadIdx.x < N) {
        float v = 0.0f;
#pragma unroll
        for (int w8 = 0; w8 < kGemvThreads / 32; ++w8) { v += part[w8][threadIdx.x]; }
        out[std::int64_t(threadIdx.x) * m + row] = v;
    }
}

void check_blas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("deepseek_v41: cuBLAS ") + what + " failed (" +
                                 std::to_string(static_cast<int>(status)) + ")");
    }
}

// One handle per device, created on first use and kept: the stream is set per call.
cublasHandle_t blas_handle() {
    static constexpr int kMaxDevices = 16;
    static cublasHandle_t handles[kMaxDevices]{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxDevices) {
        throw std::runtime_error("deepseek_v41: device index out of range");
    }
    if (handles[device] == nullptr) {
        check_blas(cublasCreate(&handles[device]), "create");
        // FP32 products: no TF32.
        check_blas(cublasSetMathMode(handles[device], CUBLAS_PEDANTIC_MATH), "math mode");
    }
    return handles[device];
}

// ---- Hyper-connections ------------------------------------------------------------------------

__global__ void place_rows_kernel(const __nv_bfloat16* __restrict__ rows, std::int64_t n,
                                  int hidden, float* __restrict__ h, std::int64_t stride) {
    const std::int64_t k = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= n) { return; }
    h[k / hidden * stride + k % hidden] = __bfloat162float(rows[k]);
}

__global__ void expand_streams_kernel(float* h, int streams, int hidden) {
    const std::int64_t row = std::int64_t(blockIdx.x) * streams * hidden;
    const int s            = blockIdx.y + 1;
    const float4* src      = reinterpret_cast<const float4*>(h + row);
    float4* dst            = reinterpret_cast<float4*>(h + row + std::int64_t(s) * hidden);
    for (int i = threadIdx.x; i < hidden / 4; i += blockDim.x) { dst[i] = src[i]; }
}

constexpr int kMaxStreams = 4;

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void hc_split_kernel(HcSplitArgs a) {
    const int t = blockIdx.x, S = a.streams;
    const std::int64_t width = std::int64_t(S) * a.hidden;
    const float* h           = a.h + t * width;
    const std::int64_t quads = width / 4, step = blockDim.x;
    float ss                 = 0.0f;
    // kGemvAhead steps loaded before they are added, in the thread's order (see gemv_kernel).
    std::int64_t i = threadIdx.x;
    for (; i + (kGemvAhead - 1) * step < quads; i += kGemvAhead * step) {
        float4 v[kGemvAhead];
#pragma unroll
        for (int u = 0; u < kGemvAhead; ++u) { v[u] = ld4(h + 4 * (i + u * step)); }
#pragma unroll
        for (int u = 0; u < kGemvAhead; ++u) {
            ss += v[u].x * v[u].x + v[u].y * v[u].y + v[u].z * v[u].z + v[u].w * v[u].w;
        }
    }
    for (; i < quads; i += step) {
        const float4 v = ld4(h + 4 * i);
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    const float total = block_sum(ss);
    if (threadIdx.x >= 32) { return; }
    // Warp 0: lane s < S writes pre[s] and post[s], and lane j * S + k holds entry (j, k) of the
    // Sinkhorn matrix. A lane adds its row's or column's entries in index order, from zero, as one
    // thread looping over the matrix would, so every entry has that loop's bits.
    const int lane = int(threadIdx.x);
    const float r  = rsqrtf(total / float(width) + a.norm_eps);
    const int M    = (2 + S) * S;
    const float* m = a.mixes + std::int64_t(t) * M;
    if (lane < S) {
        a.pre[t * S + lane]  = sigmoid(m[lane] * r * a.scale[0] + a.base[lane]) + a.hc_eps;
        a.post[t * S + lane] = 2.0f * sigmoid(m[S + lane] * r * a.scale[1] + a.base[S + lane]);
    }
    const bool entry = lane < S * S;
    const int j = entry ? lane / S : 0, k = entry ? lane % S : 0;
    const auto at = [&](float v, int row, int column) {
        return __shfl_sync(0xFFFFFFFFu, v, row * S + column);
    };
    float c  = entry ? m[2 * S + lane] * r * a.scale[2] + a.base[2 * S + lane] : 0.0f;
    float mx = -INFINITY;
    for (int q = 0; q < S; ++q) { mx = fmaxf(mx, at(c, j, q)); }
    c         = expf(c - mx);
    float sum = 0.0f;
    for (int q = 0; q < S; ++q) { sum += at(c, j, q); }
    c = c / sum + a.hc_eps;
    const auto columns = [&] {
        float column = 0.0f;
        for (int q = 0; q < S; ++q) { column += at(c, q, k); }
        c = c / (column + a.hc_eps);
    };
    columns();
    for (int it = 1; it < a.iterations; ++it) {
        float row = 0.0f;
        for (int q = 0; q < S; ++q) { row += at(c, j, q); }
        c = c / (row + a.hc_eps);
        columns();
    }
    if (entry) { a.comb[std::int64_t(t) * S * S + lane] = c; }
}

__device__ __forceinline__ float4 mix_streams(const float* h, const float* mix, int streams,
                                              int hidden, int i) {
    if (mix == nullptr) { return bf16r4(ld4(h + 4 * i)); }
    float4 y = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int s = 0; s < streams; ++s) {
        const float4 v = ld4(h + std::int64_t(s) * hidden + 4 * i);
        const float w  = mix[s];
        y.x += w * v.x;
        y.y += w * v.y;
        y.z += w * v.z;
        y.w += w * v.w;
    }
    return bf16r4(y);
}

// The mixed row (up to kPreNormThreads * kPreNormSteps float4s) stays in registers between the
// sum and the output, its weights loaded before the sum: one block per token re-reading every
// stream's row after the sum took 5.7 us at 5,120 values and four streams on an RTX 4090.
constexpr int kPreNormThreads = 256, kPreNormSteps = 8;

__global__ void hc_pre_norm_kernel(const float* __restrict__ h, const float* __restrict__ mix,
                                   const float* __restrict__ w, int streams, int hidden, float eps,
                                   __nv_bfloat16* __restrict__ out, float* __restrict__ out_f32) {
    const int t     = blockIdx.x;
    const float* ht = h + std::int64_t(t) * streams * hidden;
    const float* mt = mix != nullptr ? mix + std::int64_t(t) * streams : nullptr;
    const int quads = hidden / 4;
    float4 y[kPreNormSteps], wv[kPreNormSteps];
    float ss = 0.0f;
#pragma unroll
    for (int s = 0; s < kPreNormSteps; ++s) {
        const int i = threadIdx.x + s * kPreNormThreads;
        if (i < quads) {
            y[s]  = mix_streams(ht, mt, streams, hidden, i);
            wv[s] = ld4(w + 4 * i);
            ss += y[s].x * y[s].x + y[s].y * y[s].y + y[s].z * y[s].z + y[s].w * y[s].w;
        }
    }
    const float inv = rsqrtf(block_sum(ss) / float(hidden) + eps);
#pragma unroll
    for (int s = 0; s < kPreNormSteps; ++s) {
        const int i = threadIdx.x + s * kPreNormThreads;
        if (i >= quads) { continue; }
        const float4 o = bf16r4(make_float4(y[s].x * inv * wv[s].x, y[s].y * inv * wv[s].y,
                                            y[s].z * inv * wv[s].z, y[s].w * inv * wv[s].w));
        if (out != nullptr) { st_bf16x4(out + std::int64_t(t) * hidden + 4 * i, o); }
        if (out_f32 != nullptr) {
            *reinterpret_cast<float4*>(out_f32 + std::int64_t(t) * hidden + 4 * i) = o;
        }
    }
}

__global__ void hc_post_kernel(float* __restrict__ h, const float* __restrict__ y,
                               const float* __restrict__ post, const float* __restrict__ comb,
                               int streams, int hidden) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden / 4) { return; }
    float* ht    = h + std::int64_t(t) * streams * hidden;
    const float4 yv = ld4(y + std::int64_t(t) * hidden + 4 * i);
    float4 old[kMaxStreams];
    for (int s = 0; s < streams; ++s) { old[s] = ld4(ht + std::int64_t(s) * hidden + 4 * i); }
    const float* ct = comb + std::int64_t(t) * streams * streams;
    for (int k = 0; k < streams; ++k) {
        const float p = post[t * streams + k];
        float4 o      = make_float4(p * yv.x, p * yv.y, p * yv.z, p * yv.w);
        for (int j = 0; j < streams; ++j) {
            const float c = ct[j * streams + k];
            o.x += c * old[j].x;
            o.y += c * old[j].y;
            o.z += c * old[j].z;
            o.w += c * old[j].w;
        }
        *reinterpret_cast<float4*>(ht + std::int64_t(k) * hidden + 4 * i) = o;
    }
}

__global__ void rms_rows_kernel(const float* __restrict__ x, std::int64_t x_stride, int n,
                                const float* __restrict__ w, float eps,
                                __nv_bfloat16* __restrict__ out, float* __restrict__ out_f32) {
    const int t     = blockIdx.x;
    const float* xt = x + t * x_stride;
    float ss        = 0.0f;
    for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
        const float4 v = bf16r4(ld4(xt + 4 * i));
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    const float inv = rsqrtf(block_sum(ss) / float(n) + eps);
    for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
        const float4 v  = bf16r4(ld4(xt + 4 * i));
        const float4 wv = ld4(w + 4 * i);
        const float4 o =
            bf16r4(make_float4(v.x * inv * wv.x, v.y * inv * wv.y, v.z * inv * wv.z, v.w * inv * wv.w));
        if (out != nullptr) { st_bf16x4(out + std::int64_t(t) * n + 4 * i, o); }
        if (out_f32 != nullptr) { *reinterpret_cast<float4*>(out_f32 + std::int64_t(t) * n + 4 * i) = o; }
    }
}

// The bf16(rms(bf16(v)) * w) of a 512-value row held by a block of 128 threads, four values each.
__device__ __forceinline__ float4 latent_norm(float4 v, const float* __restrict__ w, float eps) {
    v               = bf16r4(v);
    const float inv = rsqrtf(block_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w) / float(kLatent) + eps);
    const float4 wv = ld4(w + 4 * threadIdx.x);
    return bf16r4(make_float4(v.x * inv * wv.x, v.y * inv * wv.y, v.z * inv * wv.z, v.w * inv * wv.w));
}

constexpr int kLatentThreads = kLatent / 4;

__global__ void kv_post_kernel(KvPostArgs a) {
    const int t   = blockIdx.x;
    const int at  = 4 * threadIdx.x;
    float4 v      = latent_norm(ld4(a.kv + t * a.kv_stride + at), a.norm, a.eps);
    v             = bf16r4(rope4(v, at, kLatent, float(a.positions[t]), a.freq, false));
    float4 r;
    float s;
    (void)fp8_group(v, r, s);
    const int seq = a.sequences != nullptr ? a.sequences[t] : a.sequence;
    const int pos = a.slots != nullptr ? a.slots[t] : a.positions[t];
    std::uint8_t* row = a.cache + seq * a.seq_stride + std::int64_t(pos % a.ring) * kWindowRowBytes;
    *reinterpret_cast<unsigned*>(row + at) = e4m3_codes4(r);
    if ((at & 31) == 0) { row[kLatent + at / 32] = std::uint8_t(e8m0_code(s)); }
}

__device__ __forceinline__ float4 load4(const float* p) { return ld4(p); }
__device__ __forceinline__ float4 load4(const __nv_bfloat16* p) { return ld_bf16x4(p); }

// Block (t, head) of dim / 4 threads.
template <typename In>
__global__ void rotate_heads_kernel(const In* x, int heads, int dim,
                                    const std::int32_t* __restrict__ positions,
                                    const float* __restrict__ freq, bool fp4, __nv_bfloat16* out) {
    const int t = blockIdx.x, head = blockIdx.y, at = 4 * threadIdx.x;
    const std::int64_t row = (std::int64_t(t) * heads + head) * dim;
    float4 v               = bf16r4(load4(x + row + at));
    v                      = bf16r4(rope4(v, at, dim, float(positions[t]), freq, false));
    if (fp4) { v = fp4_e8m0_group(v); }
    st_bf16x4(out + row + at, v);
}

// ---- Compressor ---------------------------------------------------------------------------------

__global__ void compress_kernel(CompressArgs a) {
    const int t  = blockIdx.x;
    const int at = 4 * threadIdx.x;
    const int p  = a.positions[t];
    float4 v;
    if (a.ratio == 1) {
        v = ld4(a.kv + std::int64_t(t) * kLatent + at);
    } else {
        if ((p + 1) % a.ratio != 0) { return; }
        const int seq = a.sequences != nullptr ? a.sequences[t] : a.sequence;
        const bool in_call =
            t > 0 && a.positions[t - 1] == p - 1 &&
            (a.sequences != nullptr ? a.sequences[t - 1] : a.sequence) == seq;
        const float *kv0, *sc0;
        if (in_call) {
            kv0 = a.kv + std::int64_t(t - 1) * kLatent;
            sc0 = a.score + std::int64_t(t - 1) * kLatent;
        } else {
            const std::int64_t slot = seq * a.tail_seq_stride + std::int64_t((p - 1) % a.slots) * kLatent;
            kv0 = a.tail_kv + slot;
            sc0 = a.tail_score + slot;
        }
        const float4 k0 = ld4(kv0 + at), s0 = ld4(sc0 + at);
        const float4 k1 = ld4(a.kv + std::int64_t(t) * kLatent + at);
        const float4 s1 = ld4(a.score + std::int64_t(t) * kLatent + at);
        const auto pool = [](float ka, float sa, float kb, float sb) {
            const float m = fmaxf(sa, sb);
            const float ea = expf(sa - m), eb = expf(sb - m);
            const float sum = ea + eb;
            return ka * (ea / sum) + kb * (eb / sum);
        };
        v = make_float4(pool(k0.x, s0.x, k1.x, s1.x), pool(k0.y, s0.y, k1.y, s1.y),
                        pool(k0.z, s0.z, k1.z, s1.z), pool(k0.w, s0.w, k1.w, s1.w));
    }
    v = latent_norm(v, a.norm, a.eps);
    *reinterpret_cast<float4*>(a.latent + std::int64_t(t) * kLatent + at) = v;
}

__global__ void compress_tail_kernel(CompressArgs a) {
    const int t = a.tail_from + blockIdx.x;
    const int at = 4 * threadIdx.x;
    const int seq = a.sequences != nullptr ? a.sequences[t] : a.sequence;
    const std::int64_t slot =
        seq * a.tail_seq_stride + std::int64_t(a.positions[t] % a.slots) * kLatent + at;
    *reinterpret_cast<float4*>(a.tail_kv + slot)    = ld4(a.kv + std::int64_t(t) * kLatent + at);
    *reinterpret_cast<float4*>(a.tail_score + slot) = ld4(a.score + std::int64_t(t) * kLatent + at);
}

__global__ void latent_post_kernel(LatentArgs a) {
    const int t = blockIdx.x;
    const int p = a.positions[t];
    if ((p + 1) % a.ratio != 0) { return; }
    const int j     = (p + 1) / a.ratio - 1;
    const float g   = float(j * a.ratio);
    const int seq   = a.sequences != nullptr ? a.sequences[t] : a.sequence;
    const int at    = 4 * threadIdx.x;
    // Warp 0: the indexer key, 128 values.
    if (threadIdx.x < 32) {
        float4 k = bf16r4(ld4(a.key + std::int64_t(t) * kIndex + at));
        const float inv = rsqrtf(warp_sum(k.x * k.x + k.y * k.y + k.z * k.z + k.w * k.w) / float(kIndex) + a.eps);
        const float4 wv = ld4(a.key_norm + at);
        k = bf16r4(make_float4(k.x * inv * wv.x, k.y * inv * wv.y, k.z * inv * wv.z, k.w * inv * wv.w));
        k = bf16r4(rope4(k, at, kIndex, g, a.freq, false));
        float4 r;
        float s;
        (void)fp4_e8m0_group(k, r, s);
        *reinterpret_cast<std::uint16_t*>(a.index_cache + seq * a.index_seq_stride +
                                          std::int64_t(j) * kKeyRowBytes + at / 2) =
            std::uint16_t(e2m1_codes4(r));
        if ((at & 31) == 0) {
            a.index_scales[seq * a.index_scale_seq_stride + std::int64_t(j) * kKeyScaleBytes + at / 32] =
                std::uint8_t(e8m0_code(s));
        }
    }
    float4 v = ld4(a.latent + std::int64_t(t) * kLatent + at);
    v        = bf16r4(rope4(v, at, kLatent, g, a.freq, false));
    float4 r;
    float s;
    (void)fp4_e4m3_group(v, r, s);
    std::uint8_t* row = a.kv_cache + seq * a.kv_seq_stride + std::int64_t(j) * kLatentRowBytes;
    *reinterpret_cast<std::uint16_t*>(row + at / 2) = std::uint16_t(e2m1_codes4(r));
    if ((at & 15) == 0) { row[kLatent / 2 + at / 16] = std::uint8_t(e4m3_code(s)); }
}

// ---- Indexer ------------------------------------------------------------------------------------

constexpr int kScoreKeys  = 128; // keys per block, one per thread
constexpr int kScoreHeads = 32;

template <int G>
__global__ void __launch_bounds__(kScoreKeys) index_scores_kernel(ScoreArgs a) {
    __shared__ __align__(16) __nv_bfloat162 q_s[G][kScoreHeads * kIndex / 2];
    __shared__ float w_s[G][kScoreHeads];
    const int t0  = blockIdx.y * G;
    const int rows = min(G, a.tokens - t0);
    int visible[G];
    int most = 0;
#pragma unroll
    for (int r = 0; r < G; ++r) {
        visible[r] = r < rows ? (a.positions[t0 + r] + 1) / a.ratio : 0;
        most       = max(most, visible[r]);
    }
    // Listed rows: column s of group blockIdx.y stands for the position its list gives.
    const bool listed = a.list != nullptr;
    const std::int32_t* list = listed ? a.list + std::int64_t(blockIdx.y) * a.capacity : nullptr;
    const int columns = listed ? a.counts[blockIdx.y] * a.block : most;
    const int s0      = blockIdx.x * kScoreKeys;
    if (s0 >= columns) { return; }
    for (int r = 0; r < rows; ++r) {
        const auto* src = reinterpret_cast<const uint4*>(a.q + std::int64_t(t0 + r) * a.heads * kIndex);
        auto* dst       = reinterpret_cast<uint4*>(q_s[r]);
        for (int i = threadIdx.x; i < a.heads * kIndex / 8; i += kScoreKeys) { dst[i] = src[i]; }
        for (int h = threadIdx.x; h < a.heads; h += kScoreKeys) {
            w_s[r][h] = a.weights[std::int64_t(t0 + r) * a.heads + h] * a.weight_scale;
        }
    }
    __syncthreads();
    const int s = s0 + threadIdx.x;
    if (s >= columns) { return; }
    const int j = listed ? list[s / a.block] * a.block + s % a.block : s;
    if (j >= most) { return; }
    // Rows that do not keep the position's block take -inf; the key is read only for a row that
    // does.
    bool needed = false;
#pragma unroll
    for (int r = 0; r < G; ++r) {
        if (j >= visible[r]) { continue; }
        if (a.candidates != nullptr) {
            const int b = j / a.block;
            const std::uint32_t word =
                a.candidates[std::int64_t(t0 + r) * a.candidate_words + b / 32];
            if (((word >> (b % 32)) & 1U) == 0) {
                a.out[std::int64_t(t0 + r) * a.width + s] = -INFINITY;
                continue;
            }
        }
        needed = true;
    }
    if (!needed) { return; }
    const int seq = a.sequences != nullptr ? a.sequences[t0] : a.sequence;
    __nv_bfloat162 k[kIndex / 2];
    {
        const auto* src = reinterpret_cast<const uint4*>(a.keys + seq * a.key_seq_stride +
                                                         std::int64_t(j) * kKeyRowBytes);
        const unsigned scales = *reinterpret_cast<const unsigned*>(
            a.key_scales + seq * a.key_scale_seq_stride + std::int64_t(j) * kKeyScaleBytes);
#pragma unroll
        for (int i = 0; i < kIndex / 32; ++i) {
            const uint4 w = src[i];
            const float s = e8m0_value((scales >> (8 * i)) & 0xFFu);
            const unsigned words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint4 v = e2m1_bf16x8(words[q], s);
                k[16 * i + 4 * q]     = *reinterpret_cast<const __nv_bfloat162*>(&v.x);
                k[16 * i + 4 * q + 1] = *reinterpret_cast<const __nv_bfloat162*>(&v.y);
                k[16 * i + 4 * q + 2] = *reinterpret_cast<const __nv_bfloat162*>(&v.z);
                k[16 * i + 4 * q + 3] = *reinterpret_cast<const __nv_bfloat162*>(&v.w);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < G; ++r) {
        if (j >= visible[r]) { continue; }
        const int t = t0 + r;
        if (a.candidates != nullptr) {
            const int b = j / a.block;
            const std::uint32_t word = a.candidates[std::int64_t(t) * a.candidate_words + b / 32];
            if (((word >> (b % 32)) & 1U) == 0) { continue; }
        }
        float acc = 0.0f;
        for (int h = 0; h < a.heads; ++h) {
            const __nv_bfloat162* qh = q_s[r] + h * (kIndex / 2);
            float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
            for (int i = 0; i < kIndex / 2; ++i) {
                const float2 qv = __bfloat1622float2(qh[i]);
                const float2 kv = __bfloat1622float2(k[i]);
                d0              = fmaf(qv.x, kv.x, d0);
                d1              = fmaf(qv.y, kv.y, d1);
            }
            acc = fmaf(fmaxf(d0 + d1, 0.0f), w_s[r][h], acc);
        }
        a.out[std::int64_t(t) * a.width + s] = acc;
    }
}

// ---- Selection ----------------------------------------------------------------------------------

constexpr int kSelectThreads = 1024;

// The order-preserving key of a score: larger scores have larger keys, both zeros the same one.
__device__ __forceinline__ unsigned order_key(float v) {
    const unsigned u = v == 0.0f ? 0u : __float_as_uint(v);
    return (u & 0x80000000u) != 0 ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ bool selectable(float v) { return v != -INFINITY && !isnan(v); }

// Exclusive prefix sum over the block (kSelectThreads threads) of one value per thread, and the
// block's total.
__device__ int block_scan(int v, int& total) {
    __shared__ int warp_totals[kSelectThreads / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int inclusive = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xFFFFFFFFu, inclusive, o);
        if (lane >= o) { inclusive += u; }
    }
    __syncthreads();
    if (lane == 31) { warp_totals[warp] = inclusive; }
    __syncthreads();
    int before = 0, all = 0;
    for (int w = 0; w < kSelectThreads / 32; ++w) {
        const int c = warp_totals[w];
        if (w < warp) { before += c; }
        all += c;
    }
    total = all;
    return before + inclusive - v;
}

// The k-th largest selectable key of values[0, n) by a radix select over the block, and how many
// values equal to it the k largest take (lower index first): `threshold` and `equal`. With fewer
// than k selectable values every one of them is taken (equal = -1).
__device__ void radix_threshold(const float* values, int n, int k, unsigned& threshold, int& equal) {
    __shared__ int histogram[256];
    __shared__ unsigned s_prefix;
    __shared__ int s_remaining, s_count;
    if (threadIdx.x == 0) { s_count = 0; }
    __syncthreads();
    int count = 0;
    for (int i = threadIdx.x; i < n; i += kSelectThreads) { count += selectable(values[i]) ? 1 : 0; }
    count = warp_count(count);
    if ((threadIdx.x & 31) == 0) { atomicAdd(&s_count, count); }
    __syncthreads();
    if (s_count <= k) {
        threshold = 0;
        equal     = -1;
        return;
    }
    unsigned prefix = 0, mask = 0;
    int remaining   = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += kSelectThreads) { histogram[i] = 0; }
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += kSelectThreads) {
            const float v = values[i];
            if (!selectable(v)) { continue; }
            const unsigned key = order_key(v);
            if ((key & mask) == prefix) { atomicAdd(&histogram[(key >> shift) & 0xFFu], 1); }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int above = 0, digit = 255;
            for (; digit > 0; --digit) {
                if (above + histogram[digit] >= remaining) { break; }
                above += histogram[digit];
            }
            s_prefix    = prefix | (unsigned(digit) << shift);
            s_remaining = remaining - above;
        }
        __syncthreads();
        prefix    = s_prefix;
        remaining = s_remaining;
        mask |= 0xFFu << shift;
        __syncthreads();
    }
    threshold = prefix;
    equal     = remaining;
}

// Whether value i is among the k largest, given radix_threshold's result: called by every thread
// for consecutive chunks of kSelectThreads indices in order; `equal_seen` counts the equal values
// of the earlier chunks.
__device__ __forceinline__ bool chosen(float v, bool in_range, unsigned threshold, int equal,
                                       int& equal_seen) {
    const bool ok = in_range && selectable(v);
    if (equal < 0) { return ok; }
    const unsigned key = ok ? order_key(v) : 0;
    const bool same    = ok && key == threshold;
    int total          = 0;
    const int before   = block_scan(same ? 1 : 0, total) + equal_seen;
    equal_seen += total;
    return ok && (key > threshold || (same && before < equal));
}

// How many of the listed columns (`blocks` ascending blocks of `block` positions) stand for
// positions below n: a prefix of them, through the last listed block that starts below n.
__device__ int listed_below(const std::int32_t* list, int blocks, int block, int n) {
    __shared__ int s_last;
    if (threadIdx.x == 0) { s_last = -1; }
    __syncthreads();
    for (int i = threadIdx.x; i < blocks; i += kSelectThreads) {
        if (list[i] * block < n && (i + 1 == blocks || list[i + 1] * block >= n)) { s_last = i; }
    }
    __syncthreads();
    const int last = s_last;
    __syncthreads();
    return last < 0 ? 0 : last * block + min(block, n - list[last] * block);
}

__global__ void __launch_bounds__(kSelectThreads)
    select_topk_kernel(const float* __restrict__ scores, int width, const std::int32_t* positions,
                       int ratio, int k, ListedRows listed, std::int32_t* __restrict__ out) {
    const int t      = blockIdx.x;
    const int n      = (positions[t] + 1) / ratio;
    const float* row = scores + std::int64_t(t) * width;
    // Listed rows: the columns that stand for visible positions, and the positions they stand for.
    const std::int32_t* list =
        listed.list != nullptr ? listed.list + std::int64_t(t / listed.group) * listed.capacity
                               : nullptr;
    const int columns =
        list != nullptr ? listed_below(list, listed.counts[t / listed.group], listed.block, n) : n;
    unsigned threshold = 0;
    int equal          = 0;
    radix_threshold(row, columns, k, threshold, equal);
    int equal_seen = 0, written = 0;
    for (int base = 0; base < columns; base += kSelectThreads) {
        const int i   = base + threadIdx.x;
        const float v = i < columns ? row[i] : -INFINITY;
        const bool c  = chosen(v, i < columns, threshold, equal, equal_seen);
        int total     = 0;
        const int at  = block_scan(c ? 1 : 0, total) + written;
        if (c && at < k) {
            out[std::int64_t(t) * k + at] =
                list != nullptr ? list[i / listed.block] * listed.block + i % listed.block : i;
        }
        written += total;
    }
    for (int i = written + threadIdx.x; i < k; i += kSelectThreads) {
        out[std::int64_t(t) * k + i] = -1;
    }
}

__global__ void __launch_bounds__(kSelectThreads)
    select_candidates_kernel(const float* __restrict__ scores, int width,
                             const std::int32_t* positions, int ratio, int block, int keep,
                             std::uint32_t* __restrict__ mask, int words,
                             float* __restrict__ scratch, int stride) {
    const int t      = blockIdx.x;
    const int n      = (positions[t] + 1) / ratio;
    const int blocks = (n + block - 1) / block;
    const float* row = scores + std::int64_t(t) * width;
    float* maxima    = scratch + std::int64_t(t) * stride;
    std::uint32_t* bits = mask + std::int64_t(t) * words;
    for (int w = threadIdx.x; w < words; w += kSelectThreads) { bits[w] = 0; }
    for (int b = threadIdx.x; b < blocks; b += kSelectThreads) {
        float m = -INFINITY;
        for (int i = b * block; i < min(n, (b + 1) * block); ++i) { m = fmaxf(m, row[i]); }
        maxima[b] = b == blocks - 1 ? INFINITY : m;
    }
    __syncthreads();
    unsigned threshold = 0;
    int equal          = 0;
    radix_threshold(maxima, blocks, keep, threshold, equal);
    int equal_seen = 0;
    for (int base = 0; base < blocks; base += kSelectThreads) {
        const int b   = base + threadIdx.x;
        const float v = b < blocks ? maxima[b] : -INFINITY;
        if (chosen(v, b < blocks, threshold, equal, equal_seen)) {
            atomicOr(bits + b / 32, 1U << (b % 32));
        }
    }
}

// Group blockIdx.x's listed blocks: the set bits of its rows' masks, in ascending order.
__global__ void __launch_bounds__(kSelectThreads)
    candidate_lists_kernel(const std::uint32_t* __restrict__ mask, int words, int tokens, int group,
                           int capacity, std::int32_t* __restrict__ list,
                           std::int32_t* __restrict__ counts) {
    const int g    = blockIdx.x;
    const int t0   = g * group;
    const int rows = min(group, tokens - t0);
    std::int32_t* own = list + std::int64_t(g) * capacity;
    int written       = 0;
    for (int base = 0; base < words; base += kSelectThreads) {
        const int w        = base + threadIdx.x;
        std::uint32_t bits = 0;
        for (int r = 0; w < words && r < rows; ++r) { bits |= mask[std::int64_t(t0 + r) * words + w]; }
        int total = 0;
        int at    = block_scan(__popc(bits), total) + written;
        for (; bits != 0; bits &= bits - 1U, ++at) {
            if (at < capacity) { own[at] = w * 32 + __ffs(int(bits)) - 1; }
        }
        written += total;
    }
    if (threadIdx.x == 0) { counts[g] = min(written, capacity); }
}

// ---- Sparse attention ---------------------------------------------------------------------------
// Block (row, group of 16 heads, split) of eight warps, warp w the heads 2w and 2w + 1 of the group;
// lane l holds dimensions 128 m + 4 l .. 128 m + 4 l + 3 (m = 0..3) of both, so a key row's reads
// are 8-byte words consecutive over the lanes. A tile of 16 keys passes through shared memory once
// for all sixteen heads; every head keeps its own online softmax.
constexpr int kAttnThreads = 256;
constexpr int kAttnHeads   = 16;
constexpr int kTile        = 16;
constexpr int kMaxKeys     = 128 + 1024; // window and selected latents a row may have

__device__ __forceinline__ float key_reduce16(float (&v)[16], int lane) {
#pragma unroll
    for (int width = 8, offset = 16; width >= 1; width >>= 1, offset >>= 1) {
        const bool upper = (lane & offset) != 0;
#pragma unroll
        for (int i = 0; i < width; ++i) {
            const float send = upper ? v[i] : v[i + width];
            const float keep = upper ? v[i + width] : v[i];
            v[i]             = keep + __shfl_xor_sync(0xFFFFFFFFu, send, offset);
        }
    }
    return v[0] + __shfl_xor_sync(0xFFFFFFFFu, v[0], 1);
}

// The keys a row attends to: window slots (>= 0) then compressed rows (-1 - row), and their count.
__device__ int attention_keys(const AttentionArgs& a, int t, int* keys) {
    __shared__ int s_compressed;
    const int p   = a.positions[t];
    const int lo  = max(max(0, min(a.window_floor, p)), p - a.window_size + 1);
    const int win = p + 1 - lo;
    for (int i = threadIdx.x; i < win; i += blockDim.x) { keys[i] = (lo + i) % a.ring; }
    if (threadIdx.x == 0) { s_compressed = 0; }
    __syncthreads();
    int count = 0;
    if (a.compressed != nullptr) {
        for (int i = threadIdx.x; i < a.k; i += blockDim.x) {
            const int j = a.topk[std::int64_t(t) * a.k + i];
            if (j >= 0) {
                keys[win + i] = -1 - j;
                ++count;
            }
        }
    }
    count = warp_count(count);
    if ((threadIdx.x & 31) == 0 && count != 0) { atomicAdd(&s_compressed, count); }
    __syncthreads();
    return win + s_compressed;
}

__device__ __forceinline__ int active_splits(int keys, int splits) {
    return min(splits, max(1, (keys + kTile - 1) / kTile));
}

__global__ void __launch_bounds__(kAttnThreads) sparse_attention_kernel(AttentionArgs a) {
    __shared__ int keys[kMaxKeys];
    __shared__ __align__(16) __nv_bfloat16 kv_s[kTile][kLatent];
    const int t = blockIdx.x, group = blockIdx.y, split = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n      = attention_keys(a, t, keys);
    const int active = active_splits(n, a.splits);
    if (split >= active) { return; }
    const int chunk = ((n + active - 1) / active + kTile - 1) / kTile * kTile;
    const int lo = split * chunk, hi = min(n, lo + chunk);
    const int seq = a.sequences != nullptr ? a.sequences[t] : a.sequence;
    const std::uint8_t* window     = a.window + seq * a.window_seq_stride;
    const std::uint8_t* compressed = a.compressed != nullptr ? a.compressed + seq * a.compressed_seq_stride : nullptr;

    const int head0 = group * kAttnHeads + 2 * warp;
    float q[2][16], acc[2][16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const __nv_bfloat16* qh = a.q + (std::int64_t(t) * a.heads + head0 + h) * kLatent;
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const float4 v = ld_bf16x4(qh + 128 * m + 4 * lane);
            q[h][4 * m]     = v.x * a.scale;
            q[h][4 * m + 1] = v.y * a.scale;
            q[h][4 * m + 2] = v.z * a.scale;
            q[h][4 * m + 3] = v.w * a.scale;
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) { acc[h][i] = 0.0f; }
    }
    float mx[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};

    // Each thread moves four 16-byte words (eight values each) of a tile: row r = threadIdx.x /
    // 16, words c + 16 i. It loads their codes and scales (a window row's E4M3 codes and E8M0
    // scales, a compressed row's E2M1 codes and E4M3 scales) and stores their BF16 values.
    constexpr int kWords = kLatent / 8;
    const int r = threadIdx.x / 16, c = threadIdx.x % 16;
    uint2 next[4];
    unsigned next_scales = 0;
    int next_kind        = 0; // 0: no row, 1: a window row, 2: a compressed row
    const auto fetch = [&](int tile, int count) {
        next_kind = 0;
        if (r < count) {
            const int key = keys[tile + r];
            if (key >= 0) {
                const std::uint8_t* row = window + std::int64_t(key) * kWindowRowBytes;
                next_scales             = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int w = c + 16 * i;
                    next[i]     = *reinterpret_cast<const uint2*>(row + 8 * w);
                    next_scales |= unsigned(row[kLatent + w / 4]) << (8 * i);
                }
                next_kind = 1;
            } else {
                const std::uint8_t* row = compressed + std::int64_t(-1 - key) * kLatentRowBytes;
                next_scales             = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int w = c + 16 * i;
                    next[i].x   = *reinterpret_cast<const unsigned*>(row + 4 * w);
                    next_scales |= unsigned(row[kLatent / 2 + w / 2]) << (8 * i);
                }
                next_kind = 2;
            }
        }
    };
    const int lane_key = ((lane & 16) ? 8 : 0) + ((lane & 8) ? 4 : 0) + ((lane & 4) ? 2 : 0) +
                         ((lane & 2) ? 1 : 0);
    if (lo < hi) { fetch(lo, min(kTile, hi - lo)); }
    for (int tile = lo; tile < hi; tile += kTile) {
        const int count = min(kTile, hi - tile);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned code = (next_scales >> (8 * i)) & 0xFFu;
            reinterpret_cast<uint4*>(kv_s[r])[c + 16 * i] =
                next_kind == 1   ? e4m3_bf16x8(next[i], e8m0_value(code))
                : next_kind == 2 ? e2m1_bf16x8(next[i].x, e4m3_value(code))
                                 : make_uint4(0, 0, 0, 0);
        }
        static_assert(kTile * kWords == 4 * kAttnThreads, "four words per thread and tile");
        __syncthreads();
        if (tile + kTile < hi) { fetch(tile + kTile, min(kTile, hi - tile - kTile)); }
        float p[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float part[kTile];
#pragma unroll
            for (int kk = 0; kk < kTile; ++kk) {
                float s = 0.0f;
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const float4 v = ld_bf16x4(&kv_s[kk][128 * m + 4 * lane]);
                    s += q[h][4 * m] * v.x + q[h][4 * m + 1] * v.y + q[h][4 * m + 2] * v.z +
                         q[h][4 * m + 3] * v.w;
                }
                part[kk] = s;
            }
            const float score = key_reduce16(part, lane);
            const float sc    = lane_key < count ? score : -INFINITY;
            const float m_new = fmaxf(mx[h], warp_max(sc));
            const float scale = expf(mx[h] - m_new);
            p[h]              = lane_key < count ? expf(sc - m_new) : 0.0f;
            l[h]              = l[h] * scale + 0.5f * warp_sum(p[h]);
            mx[h]             = m_new;
#pragma unroll
            for (int i = 0; i < 16; ++i) { acc[h][i] *= scale; }
        }
#pragma unroll
        for (int kk = 0; kk < kTile; ++kk) {
            const int src = ((kk & 8) ? 16 : 0) | ((kk & 4) ? 8 : 0) | ((kk & 2) ? 4 : 0) |
                            ((kk & 1) ? 2 : 0);
            const float p0 = __shfl_sync(0xFFFFFFFFu, p[0], src);
            const float p1 = __shfl_sync(0xFFFFFFFFu, p[1], src);
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const float4 v = ld_bf16x4(&kv_s[kk][128 * m + 4 * lane]);
                acc[0][4 * m] += p0 * v.x;
                acc[0][4 * m + 1] += p0 * v.y;
                acc[0][4 * m + 2] += p0 * v.z;
                acc[0][4 * m + 3] += p0 * v.w;
                acc[1][4 * m] += p1 * v.x;
                acc[1][4 * m + 1] += p1 * v.y;
                acc[1][4 * m + 2] += p1 * v.z;
                acc[1][4 * m + 3] += p1 * v.w;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int head = head0 + h;
        if (a.splits == 1) {
            const float sink = a.sinks[head];
            const float m    = fmaxf(mx[h], sink);
            const float w    = expf(mx[h] - m);
            const float inv  = 1.0f / (l[h] * w + expf(sink - m));
            __nv_bfloat16* o = a.out + (std::int64_t(t) * a.heads + head) * kLatent;
#pragma unroll
            for (int mm = 0; mm < 4; ++mm) {
                st_bf16x4(o + 128 * mm + 4 * lane,
                          make_float4(acc[h][4 * mm] * w * inv, acc[h][4 * mm + 1] * w * inv,
                                      acc[h][4 * mm + 2] * w * inv, acc[h][4 * mm + 3] * w * inv));
            }
            continue;
        }
        // Partials: [T][heads][splits] rows of 512, then [T][heads][splits] (m, l) pairs.
        const std::int64_t rows = std::int64_t(a.tokens) * a.heads * a.splits;
        const std::int64_t row  = (std::int64_t(t) * a.heads + head) * a.splits + split;
#pragma unroll
        for (int mm = 0; mm < 4; ++mm) {
            *reinterpret_cast<float4*>(a.partial + row * kLatent + 128 * mm + 4 * lane) =
                make_float4(acc[h][4 * mm], acc[h][4 * mm + 1], acc[h][4 * mm + 2], acc[h][4 * mm + 3]);
        }
        if (lane == 0) {
            a.partial[rows * kLatent + 2 * row]     = mx[h];
            a.partial[rows * kLatent + 2 * row + 1] = l[h];
        }
    }
}

// Block (head, row) of 128 threads, four dimensions each: merges the splits and the sink.
__global__ void attention_combine_kernel(AttentionArgs a) {
    __shared__ int keys[kMaxKeys];
    __shared__ float weight[64];
    __shared__ float s_inv;
    const int head = blockIdx.x, t = blockIdx.y;
    const int n      = attention_keys(a, t, keys);
    const int active = active_splits(n, a.splits);
    const std::int64_t rows  = std::int64_t(a.tokens) * a.heads * a.splits;
    const std::int64_t first = (std::int64_t(t) * a.heads + head) * a.splits;
    if (threadIdx.x == 0) {
        const float sink = a.sinks[head];
        float m          = sink;
        for (int s = 0; s < active; ++s) { m = fmaxf(m, a.partial[rows * kLatent + 2 * (first + s)]); }
        float l = expf(sink - m);
        for (int s = 0; s < active; ++s) {
            const float ms = a.partial[rows * kLatent + 2 * (first + s)];
            const float w  = ms != -INFINITY ? expf(ms - m) : 0.0f;
            weight[s]      = w;
            l += a.partial[rows * kLatent + 2 * (first + s) + 1] * w;
        }
        s_inv = 1.0f / l;
    }
    __syncthreads();
    float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int s = 0; s < active; ++s) {
        const float w = weight[s];
        if (w == 0.0f) { continue; }
        const float4 v = ld4(a.partial + (first + s) * kLatent + 4 * threadIdx.x);
        o.x += v.x * w;
        o.y += v.y * w;
        o.z += v.z * w;
        o.w += v.w * w;
    }
    const float inv = s_inv;
    st_bf16x4(a.out + (std::int64_t(t) * a.heads + head) * kLatent + 4 * threadIdx.x,
              make_float4(o.x * inv, o.y * inv, o.z * inv, o.w * inv));
}

// Block (t, head) of 128 threads.
__global__ void attention_output_kernel(const __nv_bfloat16* __restrict__ o, int heads, int groups,
                                        const std::int32_t* __restrict__ positions,
                                        const float* __restrict__ freq, int tokens, bool token_major,
                                        __nv_bfloat16* __restrict__ out) {
    const int t = blockIdx.x, head = blockIdx.y, at = 4 * threadIdx.x;
    float4 v = ld_bf16x4(o + (std::int64_t(t) * heads + head) * kLatent + at);
    v        = bf16r4(rope4(v, at, kLatent, float(positions[t]), freq, true));
    v        = fp8_group(v);
    if (token_major) {
        st_bf16x4(out + (std::int64_t(t) * heads + head) * kLatent + at, v);
        return;
    }
    const int per = heads / groups;
    const int g = head / per, in_group = head % per;
    st_bf16x4(out + (std::int64_t(g) * tokens + t) * per * kLatent + in_group * kLatent + at, v);
}

// ---- Routing and experts ------------------------------------------------------------------------

constexpr int kMaxExperts = 512;
constexpr int kMaxTopK    = 8;

__global__ void route_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                             int experts, int top_k, float scale, std::int32_t* __restrict__ ids,
                             float* __restrict__ weights, const float* __restrict__ image_bias,
                             const std::uint8_t* __restrict__ image) {
    __shared__ float sel[kMaxExperts];
    __shared__ float chosen_s[kMaxTopK];
    __shared__ int chosen_id[kMaxTopK];
    const int t = blockIdx.x, e = threadIdx.x;
    if (image != nullptr && image[t]) { bias = image_bias; }
    float s = 0.0f;
    if (e < experts) {
        const float x  = logits[std::int64_t(t) * experts + e];
        const float sp = x > 20.0f ? x : log1pf(expf(x));
        s              = sqrtf(sp);
        sel[e]         = ops::route_order(s + bias[e]);
    }
    __syncthreads();
    if (e < experts) {
        const float mine = sel[e];
        int rank         = 0;
        for (int j = 0; j < experts; ++j) {
            const float other = sel[j];
            rank += (other > mine || (other == mine && j < e)) ? 1 : 0;
        }
        if (rank < top_k) {
            chosen_s[rank]  = s;
            chosen_id[rank] = e;
        }
    }
    __syncthreads();
    if (e < top_k) {
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k) { sum += chosen_s[k]; }
        ids[std::int64_t(t) * top_k + e]     = chosen_id[e];
        weights[std::int64_t(t) * top_k + e] = chosen_s[e] / (sum + 1e-20f) * scale;
    }
}

__global__ void clamped_swiglu_kernel(const float* __restrict__ x, std::int64_t x_stride, int width,
                                      float limit, const float* __restrict__ scale, int per_token,
                                      __nv_bfloat16* __restrict__ out, std::int64_t out_stride) {
    const std::int64_t r = blockIdx.y;
    const int i          = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) { return; }
    const float sc = scale != nullptr ? scale[r / per_token] : 1.0f;
    const float g  = bf16r(x[r * x_stride + i] * sc);
    const float u  = bf16r(x[r * x_stride + width + i] * sc);
    out[r * out_stride + i] =
        __float2bfloat16(ops::portable_silu(fminf(g, limit)) * fminf(fmaxf(u, -limit), limit));
}

__global__ void moe_combine_kernel(MoeCombineArgs a) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.hidden) { return; }
    const std::int64_t at = std::int64_t(t) * a.hidden + i;
    float v               = a.y[at];
    if (a.rows != nullptr) {
        for (int k = 0; k < a.top_k; ++k) {
            const std::int64_t pair = std::int64_t(t) * a.top_k + k;
            v += a.weights[pair] * a.rows[pair * a.hidden + i];
        }
    } else {
        v += float(double(static_cast<long long>(a.fixed[at])) * (1.0 / 4294967296.0));
    }
    a.y[at] = v;
}

__global__ void add_fixed_kernel(unsigned long long* sums, const unsigned long long* products,
                                 const std::int32_t* flag, std::int64_t n) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n || *flag == 0) { return; }
    sums[i] += products[i];
}

// ---- Engram -------------------------------------------------------------------------------------

__constant__ signed char kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                       1,    13,   25,  38,  53,  69,  89,  113};

// Thread per value: block b of 32 values is 18 bytes (F16 d, then 16 bytes of nibbles: value j
// the low nibble of byte j, value j + 16 its high nibble).
__global__ void engram_rows_kernel(const std::uint8_t* __restrict__ rows, std::int64_t values,
                                   __nv_bfloat16* __restrict__ out) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= values) { return; }
    const std::int64_t block = i / 32;
    const int j              = int(i % 32);
    const std::uint8_t* b    = rows + block * 18;
    const float d            = __half2float(__ushort_as_half(std::uint16_t(b[0] | (b[1] << 8))));
    const std::uint8_t q     = b[2 + (j % 16)];
    const int code           = j < 16 ? (q & 15) : (q >> 4);
    out[i]                   = __float2bfloat16(d * float(kIq4nl[code]));
}

// Block (t, s).
__global__ void engram_gate_kernel(float* __restrict__ h, const float* __restrict__ kv,
                                   const float* __restrict__ w, int streams, int hidden, float eps,
                                   const std::uint8_t* __restrict__ image) {
    const int t = blockIdx.x, s = blockIdx.y;
    if (image != nullptr && image[t]) { return; }
    float* hs       = h + (std::int64_t(t) * streams + s) * hidden;
    const float* kt = kv + std::int64_t(t) * (streams + 1) * hidden;
    const float* key = kt + std::int64_t(s) * hidden;
    const float* value = kt + std::int64_t(streams) * hidden;
    const float* ws = w + std::int64_t(s) * hidden;
    float dot = 0.0f, hh = 0.0f, kk = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float hv = hs[i], kv_ = bf16r(key[i]);
        dot += hv * ws[i] * kv_;
        hh += hv * hv;
        kk += kv_ * kv_;
    }
    dot = block_sum(dot);
    hh  = block_sum(hh);
    kk  = block_sum(kk);
    const float d = dot * rsqrtf(hh / float(hidden) + eps) * rsqrtf(kk / float(hidden) + eps) *
                    rsqrtf(float(hidden));
    const float gate = sigmoid(copysignf(sqrtf(fmaxf(fabsf(d), 1e-6f)), d));
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) { hs[i] += gate * bf16r(value[i]); }
}

// ---- Guards and copies --------------------------------------------------------------------------

constexpr int kGuardThreads = 128, kGuardWideThreads = 1024, kGuardNarrowColumns = 8;

__global__ void guard_columns_kernel(__nv_bfloat16* __restrict__ x, int rows, float* __restrict__ scale) {
    __nv_bfloat16* column = x + std::int64_t(blockIdx.x) * rows;
    float largest         = 0.0f;
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
        largest = fmaxf(largest, fabsf(warp_sum(__bfloat162float(column[r]))));
    }
    const int k = f16_sum_exponent(block_max(largest));
    if (threadIdx.x == 0) { scale[blockIdx.x] = ldexpf(1.0f, k); }
    if (k == 0) { return; }
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
        column[r] = __float2bfloat16(ldexpf(__bfloat162float(column[r]), -k));
    }
}

__device__ __forceinline__ void scale_value(float& v, float s) { v *= s; }
__device__ __forceinline__ void scale_value(__nv_bfloat16& v, float s) {
    v = __float2bfloat16(__bfloat162float(v) * s);
}

template <typename Out>
__global__ void unscale_columns_kernel(Out* __restrict__ out, std::int64_t stride, int rows,
                                       const float* __restrict__ scale) {
    const float s = scale[blockIdx.x];
    if (s == 1.0f) { return; }
    Out* column = out + std::int64_t(blockIdx.x) * stride;
    for (int r = threadIdx.x; r < rows; r += blockDim.x) { scale_value(column[r], s); }
}

// Block per position, one 16-byte word of the packed row per thread.
__global__ void ring_copy_kernel(const std::uint8_t* __restrict__ src, int src_ring,
                                 std::uint8_t* __restrict__ dst, int dst_ring, int begin) {
    const int p = begin + blockIdx.x;
    reinterpret_cast<uint4*>(dst + std::int64_t(p % dst_ring) * kWindowRowBytes)[threadIdx.x] =
        reinterpret_cast<const uint4*>(src + std::int64_t(p % src_ring) * kWindowRowBytes)[threadIdx.x];
}

__global__ void f32_to_bf16_kernel(const float* in, __nv_bfloat16* out, std::int64_t n) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = __float2bfloat16(in[i]); }
}

__global__ void stream_mean_kernel(const float* __restrict__ h, int streams, int hidden,
                                   float* __restrict__ out, std::int64_t out_stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) { return; }
    const float* row = h + std::int64_t(blockIdx.y) * streams * hidden + i;
    float sum        = 0.0f;
    for (int s = 0; s < streams; ++s) { sum += row[std::int64_t(s) * hidden]; }
    out[std::int64_t(blockIdx.y) * out_stride + i] = sum / float(streams);
}

// A Markov step: each warp a vocabulary row at a time (its lanes eight values of the rank each), each
// block its best (value, index); markov_pick_kernel the best of the blocks.
constexpr int kMarkovThreads = 256, kMarkovBlocks = 512;

struct Best {
    float value;
    int index;
};

__device__ __forceinline__ bool better(float v, int i, float bv, int bi) {
    return v > bv || (v == bv && i < bi);
}

__device__ Best block_best(float v, int i) {
    __shared__ float values[32];
    __shared__ int indices[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFFu, v, o);
        const int oi   = __shfl_xor_sync(0xFFFFFFFFu, i, o);
        if (better(ov, oi, v, i)) {
            v = ov;
            i = oi;
        }
    }
    __syncthreads();
    if (lane == 0) {
        values[warp]  = v;
        indices[warp] = i;
    }
    __syncthreads();
    Best best{-INFINITY, INT_MAX};
    if (threadIdx.x == 0) {
        for (int w = 0; w < int(blockDim.x >> 5); ++w) {
            if (better(values[w], indices[w], best.value, best.index)) { best = {values[w], indices[w]}; }
        }
    }
    return best;
}

__global__ void __launch_bounds__(kMarkovThreads)
    markov_best_kernel(const float* __restrict__ logits, const __nv_bfloat16* __restrict__ head,
                       const __nv_bfloat16* __restrict__ embed, const std::int32_t* __restrict__ ids,
                       int step, int domain, Best* __restrict__ out) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = kMarkovThreads / 32;
    const __nv_bfloat16* e_row = embed + std::int64_t(ids[step]) * kMarkovRank + lane * 8;
    float e[8];
    {
        const float4 a = ld_bf16x4(e_row), b = ld_bf16x4(e_row + 4);
        e[0] = a.x, e[1] = a.y, e[2] = a.z, e[3] = a.w, e[4] = b.x, e[5] = b.y, e[6] = b.z, e[7] = b.w;
    }
    float bv = -INFINITY;
    int bi   = INT_MAX;
    for (int v = blockIdx.x * warps + warp; v < domain; v += gridDim.x * warps) {
        const __nv_bfloat16* h_row = head + std::int64_t(v) * kMarkovRank + lane * 8;
        const float4 a = ld_bf16x4(h_row), b = ld_bf16x4(h_row + 4);
        float dot = a.x * e[0] + a.y * e[1] + a.z * e[2] + a.w * e[3] + b.x * e[4] + b.y * e[5] +
                    b.z * e[6] + b.w * e[7];
        dot           = warp_sum(dot);
        const float x = logits[v] + dot;
        if (better(x, v, bv, bi)) {
            bv = x;
            bi = v;
        }
    }
    const Best best = block_best(bv, bi);
    if (threadIdx.x == 0) { out[blockIdx.x] = best; }
}

__global__ void __launch_bounds__(kMarkovThreads)
    markov_pick_kernel(const Best* __restrict__ blocks, std::int32_t* __restrict__ ids, int step) {
    float bv = -INFINITY;
    int bi   = INT_MAX;
    for (int b = threadIdx.x; b < kMarkovBlocks; b += blockDim.x) {
        if (better(blocks[b].value, blocks[b].index, bv, bi)) {
            bv = blocks[b].value;
            bi = blocks[b].index;
        }
    }
    const Best best = block_best(bv, bi);
    if (threadIdx.x == 0) { ids[step + 1] = best.index == INT_MAX ? 0 : best.index; }
}

__global__ void dspark_confidence_kernel(const float* __restrict__ h, const float* __restrict__ pre,
                                         int streams, int hidden, const float* __restrict__ w,
                                         const __nv_bfloat16* __restrict__ embed,
                                         const std::int32_t* __restrict__ ids, float* __restrict__ out) {
    const int t      = blockIdx.x;
    const float* row = h + std::int64_t(t) * streams * hidden;
    float sum        = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float x = 0.0f;
        for (int s = 0; s < streams; ++s) { x += pre[t * streams + s] * row[std::int64_t(s) * hidden + i]; }
        sum += x * w[i];
    }
    const __nv_bfloat16* e = embed + std::int64_t(ids[t]) * kMarkovRank;
    for (int r = threadIdx.x; r < kMarkovRank; r += blockDim.x) {
        sum += __bfloat162float(e[r]) * w[hidden + r];
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0) { out[t] = 1.0f / (1.0f + expf(-sum)); }
}

} // namespace

void gemm_f32(int m, int n, int k, const float* w, const float* x, float* out, cudaStream_t stream) {
    if (m <= 0 || n <= 0) { return; }
    if (n <= kGemvRows) {
        const auto gemv = [&](auto tokens) {
            gemv_kernel<decltype(tokens)::value><<<m, kGemvThreads, 0, stream>>>(w, x, m, k, out);
        };
        switch (n) {
        case 1: gemv(std::integral_constant<int, 1>{}); break;
        case 2: gemv(std::integral_constant<int, 2>{}); break;
        case 3: gemv(std::integral_constant<int, 3>{}); break;
        case 4: gemv(std::integral_constant<int, 4>{}); break;
        case 5: gemv(std::integral_constant<int, 5>{}); break;
        case 6: gemv(std::integral_constant<int, 6>{}); break;
        case 7: gemv(std::integral_constant<int, 7>{}); break;
        default: gemv(std::integral_constant<int, 8>{}); break;
        }
        check_launch("gemv");
        return;
    }
    const cublasHandle_t handle = blas_handle();
    check_blas(cublasSetStream(handle, stream), "set stream");
    const float one = 1.0f, zero = 0.0f;
    check_blas(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, w, k, x, k, &zero, out, m),
               "SGEMM");
}

void place_rows(const __nv_bfloat16* rows, int count, int hidden, float* h, std::int64_t stride,
                cudaStream_t stream) {
    const std::int64_t n = std::int64_t(count) * hidden;
    if (n <= 0) { return; }
    place_rows_kernel<<<int((n + 255) / 256), 256, 0, stream>>>(rows, n, hidden, h, stride);
    check_launch("place_rows");
}

void expand_streams(float* h, int tokens, int streams, int hidden, cudaStream_t stream) {
    if (tokens <= 0 || streams <= 1) { return; }
    expand_streams_kernel<<<dim3(tokens, streams - 1), 256, 0, stream>>>(h, streams, hidden);
    check_launch("expand_streams");
}

void hc_split(const HcSplitArgs& args, cudaStream_t stream) {
    if (args.streams < 1 || args.streams > kMaxStreams || args.hidden % 4 != 0) {
        throw std::invalid_argument("deepseek_v41 hc_split: 1..4 streams of whole float4s");
    }
    if (args.tokens <= 0) { return; }
    hc_split_kernel<<<args.tokens, 256, 0, stream>>>(args);
    check_launch("hc_split");
}

void hc_pre_norm(const float* h, const float* mix, const float* w, int tokens, int streams,
                 int hidden, float eps, __nv_bfloat16* out, float* out_f32, cudaStream_t stream) {
    if (hidden % 4 != 0 || hidden / 4 > kPreNormThreads * kPreNormSteps) {
        throw std::invalid_argument("deepseek_v41 hc_pre_norm: at most 8192 values in whole float4s");
    }
    if (tokens <= 0) { return; }
    hc_pre_norm_kernel<<<tokens, kPreNormThreads, 0, stream>>>(h, mix, w, streams, hidden, eps, out,
                                                               out_f32);
    check_launch("hc_pre_norm");
}

void hc_post(float* h, const float* y, const float* post, const float* comb, int tokens,
             int streams, int hidden, cudaStream_t stream) {
    if (tokens <= 0) { return; }
    hc_post_kernel<<<dim3(blocks_for(hidden / 4, 256), tokens), 256, 0, stream>>>(h, y, post, comb,
                                                                                streams, hidden);
    check_launch("hc_post");
}

void rms_rows(const float* x, std::int64_t x_stride, int n, int tokens, const float* w, float eps,
              __nv_bfloat16* out, float* out_f32, cudaStream_t stream) {
    if (tokens <= 0) { return; }
    if (n % 4 != 0 || x_stride % 4 != 0) { throw std::invalid_argument("deepseek_v41 rms_rows: float4s"); }
    rms_rows_kernel<<<tokens, 256, 0, stream>>>(x, x_stride, n, w, eps, out, out_f32);
    check_launch("rms_rows");
}

void kv_post(const KvPostArgs& args, cudaStream_t stream) {
    if (args.tokens <= 0) { return; }
    kv_post_kernel<<<args.tokens, kLatentThreads, 0, stream>>>(args);
    check_launch("kv_post");
}

template <typename In>
void rotate_heads_launch(const In* x, int tokens, int heads, int dim, const std::int32_t* positions,
                         const float* freq, bool fp4, __nv_bfloat16* out, cudaStream_t stream) {
    if (tokens <= 0) { return; }
    if (dim % 32 != 0 || dim < kRope || dim > 1024) {
        throw std::invalid_argument("deepseek_v41 rotate_heads: whole 32-value groups past the rope");
    }
    rotate_heads_kernel<In><<<dim3(tokens, heads), dim / 4, 0, stream>>>(x, heads, dim, positions,
                                                                        freq, fp4, out);
    check_launch("rotate_heads");
}

void rotate_heads(const float* x, int tokens, int heads, int dim, const std::int32_t* positions,
                  const float* freq, bool fp4, __nv_bfloat16* out, cudaStream_t stream) {
    rotate_heads_launch(x, tokens, heads, dim, positions, freq, fp4, out, stream);
}

void rotate_heads(const __nv_bfloat16* x, int tokens, int heads, int dim,
                  const std::int32_t* positions, const float* freq, bool fp4, __nv_bfloat16* out,
                  cudaStream_t stream) {
    rotate_heads_launch(x, tokens, heads, dim, positions, freq, fp4, out, stream);
}

void compress(const CompressArgs& args, cudaStream_t stream) {
    if (args.tokens <= 0) { return; }
    if (args.ratio != 1 && args.ratio != 2) { throw std::invalid_argument("deepseek_v41 compress: ratio 1 or 2"); }
    compress_kernel<<<args.tokens, kLatentThreads, 0, stream>>>(args);
    check_launch("compress");
}

void compress_tail(const CompressArgs& args, cudaStream_t stream) {
    const int rows = args.tokens - args.tail_from;
    if (rows <= 0 || args.ratio == 1) { return; }
    compress_tail_kernel<<<rows, kLatentThreads, 0, stream>>>(args);
    check_launch("compress_tail");
}

void latent_post(const LatentArgs& args, cudaStream_t stream) {
    if (args.tokens <= 0) { return; }
    latent_post_kernel<<<args.tokens, kLatentThreads, 0, stream>>>(args);
    check_launch("latent_post");
}

void index_scores(const ScoreArgs& args, cudaStream_t stream) {
    if (args.tokens <= 0 || args.width <= 0) { return; }
    if (args.heads > kScoreHeads || (args.group != 1 && args.group != 4)) {
        throw std::invalid_argument("deepseek_v41 index_scores: at most 32 heads, groups of 1 or 4");
    }
    if (args.list != nullptr &&
        (args.candidates == nullptr || args.counts == nullptr || args.capacity <= 0 ||
         std::int64_t(args.capacity) * args.block > args.width)) {
        throw std::invalid_argument("deepseek_v41 index_scores: listed rows need candidates and "
                                    "a width of their capacity");
    }
    const std::int64_t columns =
        args.list != nullptr ? std::int64_t(args.capacity) * args.block : args.width;
    const dim3 grid(blocks_for(columns, kScoreKeys), blocks_for(args.tokens, args.group));
    if (args.group == 4) {
        index_scores_kernel<4><<<grid, kScoreKeys, 0, stream>>>(args);
    } else {
        index_scores_kernel<1><<<grid, kScoreKeys, 0, stream>>>(args);
    }
    check_launch("index_scores");
}

void select_candidates(const float* scores, int width, int tokens, const std::int32_t* positions,
                       int ratio, int block, int keep, std::uint32_t* mask, int words,
                       float* scratch, int stride, cudaStream_t stream) {
    if (tokens <= 0) { return; }
    if (block <= 0 || stride < (width + block - 1) / block) {
        throw std::invalid_argument(
            "deepseek_v41 select_candidates: the scratch stride must hold a row's blocks");
    }
    select_candidates_kernel<<<tokens, kSelectThreads, 0, stream>>>(
        scores, width, positions, ratio, block, keep, mask, words, scratch, stride);
    check_launch("select_candidates");
}

void candidate_lists(const std::uint32_t* mask, int words, int tokens, int group, int capacity,
                     std::int32_t* list, std::int32_t* counts, cudaStream_t stream) {
    if (tokens <= 0) { return; }
    if (group <= 0 || capacity <= 0) {
        throw std::invalid_argument("deepseek_v41 candidate_lists: a positive group and capacity");
    }
    candidate_lists_kernel<<<blocks_for(tokens, group), kSelectThreads, 0, stream>>>(
        mask, words, tokens, group, capacity, list, counts);
    check_launch("candidate_lists");
}

void select_topk(const float* scores, int width, int tokens, const std::int32_t* positions,
                 int ratio, int k, std::int32_t* out, cudaStream_t stream) {
    select_topk(scores, width, tokens, positions, ratio, k, ListedRows{}, out, stream);
}

void select_topk(const float* scores, int width, int tokens, const std::int32_t* positions,
                 int ratio, int k, const ListedRows& listed, std::int32_t* out,
                 cudaStream_t stream) {
    if (tokens <= 0) { return; }
    if (listed.list != nullptr && (listed.counts == nullptr || listed.group <= 0 ||
                                   listed.block <= 0 || listed.capacity <= 0)) {
        throw std::invalid_argument("deepseek_v41 select_topk: listed rows need their counts");
    }
    select_topk_kernel<<<tokens, kSelectThreads, 0, stream>>>(scores, width, positions, ratio, k,
                                                              listed, out);
    check_launch("select_topk");
}

std::size_t sparse_attention_workspace_floats(int tokens, int heads, int splits) {
    if (splits <= 1) { return 0; }
    const std::size_t rows = std::size_t(tokens) * heads * splits;
    return rows * kLatent + 2 * rows;
}

void sparse_attention(const AttentionArgs& args, cudaStream_t stream) {
    const int selected = args.compressed != nullptr ? args.k : 0;
    if (args.heads % kAttnHeads != 0 || args.splits < 1 || args.splits > 64 || args.window_size < 1 ||
        args.window_size > args.ring || selected < 0 || args.window_size + selected > kMaxKeys) {
        throw std::invalid_argument("deepseek_v41 sparse_attention: heads in groups of 16, 1..64 "
                                    "splits, a window within the ring and at most 1152 keys");
    }
    if (args.tokens <= 0) { return; }
    const dim3 grid(args.tokens, args.heads / kAttnHeads, args.splits);
    sparse_attention_kernel<<<grid, kAttnThreads, 0, stream>>>(args);
    check_launch("sparse_attention");
    if (args.splits > 1) {
        attention_combine_kernel<<<dim3(args.heads, args.tokens), kLatentThreads, 0, stream>>>(args);
        check_launch("attention_combine");
    }
}

void attention_output(const __nv_bfloat16* o, int tokens, int heads, int groups,
                      const std::int32_t* positions, const float* freq, bool token_major,
                      __nv_bfloat16* out, cudaStream_t stream) {
    if (tokens <= 0) { return; }
    if (groups <= 0 || heads % groups != 0) { throw std::invalid_argument("deepseek_v41 attention_output: groups"); }
    attention_output_kernel<<<dim3(tokens, heads), kLatentThreads, 0, stream>>>(
        o, heads, groups, positions, freq, tokens, token_major, out);
    check_launch("attention_output");
}

void route(const float* logits, const float* bias, int tokens, int experts, int top_k, float scale,
           std::int32_t* ids, float* weights, cudaStream_t stream, const float* image_bias,
           const std::uint8_t* image) {
    if (experts > kMaxExperts || top_k > kMaxTopK || top_k > experts) {
        throw std::invalid_argument("deepseek_v41 route: at most 8 of 512 experts");
    }
    if (image != nullptr && image_bias == nullptr) {
        throw std::invalid_argument("deepseek_v41 route: image tokens need the image bias");
    }
    if (tokens <= 0) { return; }
    route_kernel<<<tokens, (experts + 31) / 32 * 32, 0, stream>>>(logits, bias, experts, top_k, scale,
                                                                  ids, weights, image_bias, image);
    check_launch("route");
}

void clamped_swiglu(const float* x, std::int64_t x_stride, int rows, int width, float limit,
                    const float* scale, int per_token, __nv_bfloat16* out, std::int64_t out_stride,
                    cudaStream_t stream) {
    if (rows <= 0 || width <= 0) { return; }
    clamped_swiglu_kernel<<<dim3(blocks_for(width, 256), rows), 256, 0, stream>>>(
        x, x_stride, width, limit, scale, std::max(per_token, 1), out, out_stride);
    check_launch("clamped_swiglu");
}

void add_fixed(unsigned long long* sums, const void* products, const std::int32_t* flag,
               std::int64_t n, cudaStream_t stream) {
    if (n <= 0) { return; }
    add_fixed_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(
        sums, static_cast<const unsigned long long*>(products), flag, n);
    check_launch("add_fixed");
}

void moe_combine(const MoeCombineArgs& args, cudaStream_t stream) {
    if (args.tokens <= 0) { return; }
    moe_combine_kernel<<<dim3(blocks_for(args.hidden, 256), args.tokens), 256, 0, stream>>>(args);
    check_launch("moe_combine");
}

void engram_rows(const std::uint8_t* rows, int count, __nv_bfloat16* out, cudaStream_t stream) {
    if (count <= 0) { return; }
    const std::int64_t values = std::int64_t(count) * 256;
    engram_rows_kernel<<<blocks_for(values, 256), 256, 0, stream>>>(rows, values, out);
    check_launch("engram_rows");
}

void engram_gate(float* h, const float* kv, const float* w, int tokens, int streams, int hidden,
                 float eps, cudaStream_t stream, const std::uint8_t* image) {
    if (tokens <= 0) { return; }
    engram_gate_kernel<<<dim3(tokens, streams), 256, 0, stream>>>(h, kv, w, streams, hidden, eps,
                                                                  image);
    check_launch("engram_gate");
}

void guard_columns(__nv_bfloat16* x, int rows, int columns, float* scale, cudaStream_t stream) {
    if (rows % 32 != 0) { throw std::invalid_argument("deepseek_v41 guard_columns: rows % 32 != 0"); }
    if (columns <= 0) { return; }
    // A decode or verification call's few columns would leave all but a few SMs idle: one wide
    // block per column instead. A slice's sum and the maximum do not depend on the block's width.
    const int threads =
        columns <= kGuardNarrowColumns ? std::min(rows, kGuardWideThreads) : kGuardThreads;
    guard_columns_kernel<<<columns, threads, 0, stream>>>(x, rows, scale);
    check_launch("guard_columns");
}

void unscale_columns(float* out, std::int64_t out_stride, int out_rows, int columns,
                     const float* scale, cudaStream_t stream) {
    if (columns <= 0) { return; }
    unscale_columns_kernel<float><<<columns, kGuardThreads, 0, stream>>>(out, out_stride, out_rows, scale);
    check_launch("unscale_columns");
}

void unscale_columns(__nv_bfloat16* out, std::int64_t out_stride, int out_rows, int columns,
                     const float* scale, cudaStream_t stream) {
    if (columns <= 0) { return; }
    unscale_columns_kernel<__nv_bfloat16><<<columns, kGuardThreads, 0, stream>>>(out, out_stride,
                                                                                out_rows, scale);
    check_launch("unscale_columns");
}

void ring_copy(const std::uint8_t* src, int src_ring, std::uint8_t* dst, int dst_ring, int begin,
               int end, cudaStream_t stream) {
    if (end <= begin) { return; }
    static_assert(kWindowRowBytes % 16 == 0, "whole 16-byte words per window row");
    ring_copy_kernel<<<end - begin, kWindowRowBytes / 16, 0, stream>>>(src, src_ring, dst, dst_ring, begin);
    check_launch("ring_copy");
}

void f32_to_bf16(const float* in, __nv_bfloat16* out, std::int64_t n, cudaStream_t stream) {
    if (n <= 0) { return; }
    f32_to_bf16_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(in, out, n);
    check_launch("f32_to_bf16");
}

void stream_mean(const float* h, int tokens, int streams, int hidden, float* out,
                 std::int64_t out_stride, cudaStream_t stream) {
    if (tokens <= 0) { return; }
    stream_mean_kernel<<<dim3(blocks_for(hidden, 256), tokens), 256, 0, stream>>>(h, streams, hidden,
                                                                                 out, out_stride);
    check_launch("stream_mean");
}

std::size_t markov_scratch_bytes() { return std::size_t(kMarkovBlocks) * sizeof(Best); }

void markov_step(const float* logits, const __nv_bfloat16* head, const __nv_bfloat16* embed,
                 int domain, std::int32_t* ids, int step, void* scratch, cudaStream_t stream) {
    if (domain <= 0) { throw std::invalid_argument("deepseek_v41 markov_step: an empty domain"); }
    auto* blocks = static_cast<Best*>(scratch);
    markov_best_kernel<<<kMarkovBlocks, kMarkovThreads, 0, stream>>>(logits, head, embed, ids, step,
                                                                     domain, blocks);
    check_launch("markov_best");
    markov_pick_kernel<<<1, kMarkovThreads, 0, stream>>>(blocks, ids, step);
    check_launch("markov_pick");
}

void dspark_confidence(const float* h, const float* pre, int tokens, int streams, int hidden,
                       const float* w, const __nv_bfloat16* embed, const std::int32_t* ids,
                       float* out, cudaStream_t stream) {
    if (tokens <= 0) { return; }
    dspark_confidence_kernel<<<tokens, 256, 0, stream>>>(h, pre, streams, hidden, w, embed, ids, out);
    check_launch("dspark_confidence");
}

} // namespace ninfer::models::deepseek_v41::kernels
