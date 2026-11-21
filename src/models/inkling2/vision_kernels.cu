#include "models/inkling2/vision_kernels.h"

#include <cuda_bf16.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::models::inkling2::vision_kernels {
namespace {

void check_launch(const char* what) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("inkling2 vision ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

int blocks_for(std::int64_t n, int threads) { return int((n + threads - 1) / threads); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFu, v, o); }
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o)); }
    return v;
}

// Over the block (blockDim.x a multiple of 32, at most 1024); every thread gets the result.
template <bool Max>
__device__ float block_reduce(float v) {
    __shared__ float partial[32];
    __shared__ float total;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = Max ? warp_max(v) : warp_sum(v);
    __syncthreads();
    if (lane == 0) { partial[warp] = v; }
    __syncthreads();
    if (warp == 0) {
        float s = lane < int(blockDim.x >> 5) ? partial[lane] : (Max ? -INFINITY : 0.0f);
        s       = Max ? warp_max(s) : warp_sum(s);
        if (lane == 0) { total = s; }
    }
    __syncthreads();
    return total;
}

int row_threads(int hidden) { return (hidden / 4 + 31) / 32 * 32; }

// Block r (the tower's row r), thread over channels.
__global__ void embed_kernel(const float* __restrict__ proj, const float* __restrict__ positions,
                             int grid, int grid_w, int grid_h,
                             const float* __restrict__ class_embedding,
                             const float* __restrict__ class_position, int hidden,
                             float* __restrict__ h) {
    const int r = blockIdx.x;
    float* out  = h + std::int64_t(r) * hidden;
    if (r == 0) {
        for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
            out[c] = class_embedding[c] + class_position[c];
        }
        return;
    }
    const int p = r - 1, x = p % grid_w, y = p / grid_w;
    const float* row = proj + std::int64_t(p) * hidden;
    if (grid_w == grid && grid_h == grid) {
        const float* pos = positions + std::int64_t(p) * hidden;
        for (int c = threadIdx.x; c < hidden; c += blockDim.x) { out[c] = row[c] + pos[c]; }
        return;
    }
    // ggml_interpolate (BILINEAR, no flags): the scale factor out / in per axis, half-pixel
    // centers, the lower neighbour clamped before its weight is taken, the weight clamped to [0, 1].
    const float sf0 = float(grid_w) / float(grid), sf1 = float(grid_h) / float(grid);
    const float fy = (float(y) + 0.5f) / sf1 - 0.5f, fx = (float(x) + 0.5f) / sf0 - 0.5f;
    int y0 = int(floorf(fy)), x0 = int(floorf(fx));
    int y1 = y0 + 1, x1 = x0 + 1;
    y0 = max(0, min(y0, grid - 1));
    y1 = max(0, min(y1, grid - 1));
    x0 = max(0, min(x0, grid - 1));
    x1 = max(0, min(x1, grid - 1));
    const float dy = fmaxf(0.0f, fminf(fy - float(y0), 1.0f));
    const float dx = fmaxf(0.0f, fminf(fx - float(x0), 1.0f));
    const float* a = positions + (std::int64_t(y0) * grid + x0) * hidden;
    const float* b = positions + (std::int64_t(y0) * grid + x1) * hidden;
    const float* c = positions + (std::int64_t(y1) * grid + x0) * hidden;
    const float* d = positions + (std::int64_t(y1) * grid + x1) * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float v = a[i] * (1 - dx) * (1 - dy) + b[i] * dx * (1 - dy) + c[i] * (1 - dx) * dy +
                        d[i] * dx * dy;
        out[i] = row[i] + v;
    }
}

// Block r, one float4 of the row per thread.
__global__ void layer_norm_kernel(const float* x, const float* __restrict__ w,
                                  const float* __restrict__ b, int hidden, float eps,
                                  __nv_bfloat16* __restrict__ out, float* f32) {
    const std::int64_t row = std::int64_t(blockIdx.x) * hidden;
    const int i = threadIdx.x * 4;
    const bool valid = i < hidden;
    float4 v = valid ? *reinterpret_cast<const float4*>(x + row + i) : make_float4(0, 0, 0, 0);
    const float mean = block_reduce<false>(v.x + v.y + v.z + v.w) / float(hidden);
    const float4 d = make_float4(v.x - mean, v.y - mean, v.z - mean, v.w - mean);
    const float var = block_reduce<false>(valid ? d.x * d.x + d.y * d.y + d.z * d.z + d.w * d.w : 0.0f) /
                      float(hidden);
    if (!valid) { return; }
    const float inv = 1.0f / sqrtf(var + eps);
    const float4 wv = *reinterpret_cast<const float4*>(w + i);
    const float4 bv = *reinterpret_cast<const float4*>(b + i);
    const float4 y  = make_float4(d.x * inv * wv.x + bv.x, d.y * inv * wv.y + bv.y,
                                  d.z * inv * wv.z + bv.z, d.w * inv * wv.w + bv.w);
    if (f32 != nullptr) { *reinterpret_cast<float4*>(f32 + row + i) = y; }
    if (out == nullptr) { return; }
    out[row + i]     = __float2bfloat16_rn(y.x);
    out[row + i + 1] = __float2bfloat16_rn(y.y);
    out[row + i + 2] = __float2bfloat16_rn(y.z);
    out[row + i + 3] = __float2bfloat16_rn(y.w);
}

// Block r; thread j over the row's 3 hidden / 2 pairs: section (q, k, v), head and pair i < D/2.
__global__ void rotate_qkv_kernel(const float* __restrict__ qkv, const float* __restrict__ q_bias,
                                  const float* __restrict__ k_bias, const float* __restrict__ v_bias,
                                  int grid_w, int hidden, int head_dim,
                                  const float* __restrict__ inv_freq,
                                  __nv_bfloat16* __restrict__ out) {
    const int r = blockIdx.x, pairs = head_dim / 2, quarter = head_dim / 4;
    const float px = r == 0 ? 0.0f : float((r - 1) % grid_w + 1);
    const float py = r == 0 ? 0.0f : float((r - 1) / grid_w + 1);
    const std::int64_t base = std::int64_t(r) * 3 * hidden;
    for (int j = threadIdx.x; j < 3 * hidden / 2; j += blockDim.x) {
        const int section = j / (hidden / 2);
        const int within  = j % (hidden / 2);
        const int head = within / pairs, i = within % pairs;
        const int at = head * head_dim + 2 * i; // within the section
        const float* bias = section == 0 ? q_bias : section == 1 ? k_bias : v_bias;
        const float x1 = qkv[base + section * hidden + at] + bias[at];
        const float x2 = qkv[base + section * hidden + at + 1] + bias[at + 1];
        __nv_bfloat16* o = out + base + section * hidden + at;
        if (section == 2) {
            o[0] = __float2bfloat16_rn(x1);
            o[1] = __float2bfloat16_rn(x2);
            continue;
        }
        const float angle = (i < quarter ? px : py) * inv_freq[i % quarter];
        float s, c;
        sincosf(angle, &s, &c);
        o[0] = __float2bfloat16_rn(x1 * c - x2 * s);
        o[1] = __float2bfloat16_rn(x1 * s + x2 * c);
    }
}

// Block r.
__global__ void softmax_rows_kernel(const float* __restrict__ s, int keys,
                                    __nv_bfloat16* __restrict__ p) {
    const float* row   = s + std::int64_t(blockIdx.x) * keys;
    __nv_bfloat16* out = p + std::int64_t(blockIdx.x) * keys;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < keys; j += blockDim.x) { m = fmaxf(m, row[j]); }
    m = block_reduce<true>(m);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < keys; j += blockDim.x) { sum += expf(row[j] - m); }
    const float inv = 1.0f / block_reduce<false>(sum);
    for (int j = threadIdx.x; j < keys; j += blockDim.x) {
        out[j] = __float2bfloat16_rn(expf(row[j] - m) * inv);
    }
}

__global__ void residual_kernel(float* __restrict__ h, const float* __restrict__ y,
                                const float* __restrict__ bias, const float* __restrict__ scale,
                                std::int64_t n, int hidden) {
    const std::int64_t k = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= n) { return; }
    const int c = int(k % hidden);
    h[k] += (y[k] + bias[c]) * scale[c];
}

__global__ void gelu_kernel(const float* __restrict__ x, const float* __restrict__ bias,
                            std::int64_t n, int width, __nv_bfloat16* __restrict__ mid) {
    const std::int64_t k = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= n) { return; }
    const float v = bias != nullptr ? x[k] + bias[k % width] : x[k];
    // 1 + erf(v / sqrt(2)) as erfc(-v / sqrt(2)), which keeps the negative tail's small values
    // instead of cancelling them.
    mid[k] = __float2bfloat16_rn(0.5f * v * erfcf(v * -0.70710678118654752f));
}

__global__ void merge_kernel(const float* __restrict__ h, int grid_w, int hidden, std::int64_t n,
                             __nv_bfloat16* __restrict__ cells) {
    const std::int64_t k = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= n) { return; }
    const int features = 4 * hidden, cells_w = grid_w / 2;
    const std::int64_t cell = k / features;
    const int f  = int(k % features);
    const int dy = f / (2 * hidden), dx = f % (2 * hidden) / hidden, c = f % hidden;
    const std::int64_t y = (cell / cells_w) * 2 + dy, x = (cell % cells_w) * 2 + dx;
    cells[k] = __float2bfloat16_rn(h[(1 + y * grid_w + x) * hidden + c]);
}

__global__ void f32_to_bf16_kernel(const float* __restrict__ x, std::int64_t n,
                                   __nv_bfloat16* __restrict__ out) {
    const std::int64_t k = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < n) { out[k] = __float2bfloat16_rn(x[k]); }
}

} // namespace

void embed(const float* proj, const float* positions, int grid, int grid_w, int grid_h,
           const float* class_embedding, const float* class_position, int hidden, float* h,
           cudaStream_t stream) {
    if (grid_w <= 0 || grid_h <= 0) { return; }
    embed_kernel<<<1 + grid_w * grid_h, 256, 0, stream>>>(proj, positions, grid, grid_w, grid_h,
                                                          class_embedding, class_position, hidden, h);
    check_launch("embed");
}

void layer_norm(const float* x, const float* w, const float* b, int rows, int hidden, float eps,
                __nv_bfloat16* out, cudaStream_t stream, float* f32) {
    if (hidden % 4 != 0 || hidden > 4096) {
        throw std::invalid_argument("inkling2 vision layer_norm: hidden % 4 != 0 or > 4096");
    }
    if (rows <= 0) { return; }
    layer_norm_kernel<<<rows, row_threads(hidden), 0, stream>>>(x, w, b, hidden, eps, out, f32);
    check_launch("layer_norm");
}

void rotate_qkv(const float* qkv, const float* q_bias, const float* k_bias, const float* v_bias,
                int rows, int grid_w, int hidden, int head_dim, const float* inv_freq,
                __nv_bfloat16* out, cudaStream_t stream) {
    if (head_dim % 4 != 0 || hidden % head_dim != 0) {
        throw std::invalid_argument("inkling2 vision rotate_qkv: heads of whole rotated pairs");
    }
    if (rows <= 0) { return; }
    rotate_qkv_kernel<<<rows, 512, 0, stream>>>(qkv, q_bias, k_bias, v_bias, grid_w, hidden,
                                                head_dim, inv_freq, out);
    check_launch("rotate_qkv");
}

void softmax_rows(const float* s, int rows, int keys, __nv_bfloat16* p, cudaStream_t stream) {
    if (rows <= 0 || keys <= 0) { return; }
    softmax_rows_kernel<<<rows, 256, 0, stream>>>(s, keys, p);
    check_launch("softmax_rows");
}

void residual(float* h, const float* y, const float* bias, const float* scale, int rows,
              int hidden, cudaStream_t stream) {
    const std::int64_t n = std::int64_t(rows) * hidden;
    if (n <= 0) { return; }
    residual_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(h, y, bias, scale, n, hidden);
    check_launch("residual");
}

void gelu(const float* x, const float* bias, int rows, int width, __nv_bfloat16* mid,
          cudaStream_t stream) {
    const std::int64_t n = std::int64_t(rows) * width;
    if (n <= 0) { return; }
    gelu_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(x, bias, n, width, mid);
    check_launch("gelu");
}

void merge(const float* h, int grid_w, int grid_h, int hidden, __nv_bfloat16* cells,
           cudaStream_t stream) {
    if (grid_w % 2 != 0 || grid_h % 2 != 0) {
        throw std::invalid_argument("inkling2 vision merge: a grid of whole 2 x 2 cells");
    }
    const std::int64_t n = std::int64_t(grid_w / 2) * (grid_h / 2) * 4 * hidden;
    if (n <= 0) { return; }
    merge_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(h, grid_w, hidden, n, cells);
    check_launch("merge");
}

void f32_to_bf16(const float* x, std::int64_t n, __nv_bfloat16* out, cudaStream_t stream) {
    if (n <= 0) { return; }
    f32_to_bf16_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(x, n, out);
    check_launch("f32_to_bf16");
}

} // namespace ninfer::models::inkling2::vision_kernels
