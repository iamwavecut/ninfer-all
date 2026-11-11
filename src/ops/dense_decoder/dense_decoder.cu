#include "ninfer/ops/dense_decoder.h"
#include "core/device.h"
#include <cmath>
#include <stdexcept>

namespace ninfer::ops {
namespace {
__device__ float warp_sum(float x) {
    for (int d = 16; d; d >>= 1) x += __shfl_down_sync(0xffffffff, x, d);
    return __shfl_sync(0xffffffff, x, 0);
}

__global__ void norm(const float* x, const float* w, int h, float eps, __nv_bfloat16* y) {
    __shared__ float sums[8];
    int t = blockIdx.x, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    float s = 0;
    for (int i = threadIdx.x; i < h; i += 256) {
        float z = x[t * h + i];
        s       = fmaf(z, z, s);
    }
    s = warp_sum(s);
    if (lane == 0) sums[warp] = s;
    __syncthreads();
    float all = threadIdx.x < 8 ? sums[threadIdx.x] : 0;
    all       = warp_sum(all);
    if (threadIdx.x == 0) sums[0] = rsqrtf(all / h + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < h; i += 256)
        y[t * h + i] = __float2bfloat16_rn(x[t * h + i] * sums[0] * w[i]);
}

__global__ void rotary(const float* in, const int* pos, int t, int hq, int hk, float theta,
                       float* q, __half* k, __half* v) {
    int head = blockIdx.x, row = blockIdx.y, d = threadIdx.x;
    int stride = (hq + 2 * hk) * 128;
    if (d < 64 && head < hq + hk) {
        int offset = head < hq ? head * 128 : hq * 128 + (head - hq) * 128;
        float a = in[row * stride + offset + d], b = in[row * stride + offset + d + 64];
        double angle = double(pos[row]) * pow(double(theta), -double(d) / 64.0), ss, cc;
        sincos(angle, &ss, &cc);
        float s = float(ss), c = float(cc);
        float lo = fmaf(a, c, -b * s), hi = fmaf(a, s, b * c);
        if (head < hq) {
            q[(row * hq + head) * 128 + d]      = lo;
            q[(row * hq + head) * 128 + d + 64] = hi;
        } else {
            int o         = ((head - hq) * t + row) * 128;
            k[o + d]      = __float2half_rn(lo);
            k[o + d + 64] = __float2half_rn(hi);
        }
    }
    if (head < hk && d < 128)
        v[(head * t + row) * 128 + d] =
            __float2half_rn(in[row * stride + (hq + hk + head) * 128 + d]);
}

// Each warp owns a query/head and one 256-key segment. No cross-warp dependencies.
__global__ void partial_attention(const float* q, const __half* k, const __half* v, const int* pos,
                                  int hq, int hk, int first, int n, int stride, int splits,
                                  float* partial) {
    int head = blockIdx.x, row = blockIdx.y, split = blockIdx.z, lane = threadIdx.x;
    int kv = head / (hq / hk), end = min(n, min((split + 1) * 256, pos[row] - first + 1));
    float query[4], acc[4]         = {0, 0, 0, 0};
    for (int i = 0; i < 4; ++i) query[i] = q[(row * hq + head) * 128 + lane + 32 * i];
    float m = -INFINITY, sum = 0;
    for (int j = split * 256; j < end; ++j) {
        float dot = 0;
        for (int i = 0; i < 4; ++i)
            dot = fmaf(query[i], __half2float(k[(kv * stride + j) * 128 + lane + 32 * i]), dot);
        float score = warp_sum(dot) * 0.08838834764831845f;
        float next = fmaxf(m, score), a = expf(m - next), b = expf(score - next);
        for (int i = 0; i < 4; ++i)
            acc[i] = fmaf(b, __half2float(v[(kv * stride + j) * 128 + lane + 32 * i]), a * acc[i]);
        sum = a * sum + b;
        m   = next;
    }
    std::size_t base = ((std::size_t(row) * hq + head) * splits + split) * 130;
    for (int i = 0; i < 4; ++i) partial[base + lane + 32 * i] = acc[i];
    if (lane == 0) {
        partial[base + 128] = m;
        partial[base + 129] = sum;
    }
}

__global__ void merge_attention(const float* part, int hq, int splits, float* state, bool init,
                                bool finish, __nv_bfloat16* out) {
    int head = blockIdx.x, row = blockIdx.y, lane = threadIdx.x;
    std::size_t base = (std::size_t(row) * hq + head) * 130;
    float m = init ? -INFINITY : state[base + 128], s = init ? 0 : state[base + 129];
    float acc[4];
    for (int i = 0; i < 4; ++i) acc[i] = init ? 0 : state[base + lane + 32 * i];
    for (int p = 0; p < splits; ++p) {
        std::size_t at = ((std::size_t(row) * hq + head) * splits + p) * 130;
        float ps       = part[at + 129];
        if (ps == 0) continue;
        float pm = part[at + 128], nm = fmaxf(m, pm), a = s == 0 ? 0 : expf(m - nm),
              b = expf(pm - nm);
        for (int i = 0; i < 4; ++i) acc[i] = fmaf(b, part[at + lane + 32 * i], a * acc[i]);
        s = a * s + b * ps;
        m = nm;
    }
    for (int i = 0; i < 4; ++i) {
        state[base + lane + 32 * i] = acc[i];
        if (finish)
            out[(std::size_t(row) * hq + head) * 128 + lane + 32 * i] =
                __float2bfloat16_rn(acc[i] / s);
    }
    if (lane == 0) {
        state[base + 128] = m;
        state[base + 129] = s;
    }
}
} // namespace

void rmsnorm_f32(const float* x, const float* w, int h, int t, float eps, __nv_bfloat16* y,
                 cudaStream_t s) {
    if (!x || !w || !y || h <= 0 || h % 32 || t <= 0 || !(eps > 0) || !std::isfinite(eps))
        throw std::invalid_argument("rmsnorm_f32 geometry");
    norm<<<t, 256, 0, s>>>(x, w, h, eps, y);
    CUDA_CHECK(cudaGetLastError());
}

void rotary_kv_f16(const float* x, const int* pos, int t, int hq, int hk, float theta, float* q,
                   __half* k, __half* v, cudaStream_t s) {
    if (!x || !pos || !q || !k || !v || t <= 0 || hq <= 0 || hk <= 0 || hq % hk || !(theta > 0) ||
        !std::isfinite(theta))
        throw std::invalid_argument("rotary_kv_f16 geometry");
    rotary<<<dim3(hq + hk, t), 128, 0, s>>>(x, pos, t, hq, hk, theta, q, k, v);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t segment_attention_workspace_bytes(int t, int h, int n) {
    if (t <= 0 || h <= 0 || n <= 0)
        throw std::invalid_argument("segment attention workspace geometry");
    return std::size_t(t) * h * ((n + 255) / 256) * 130 * sizeof(float);
}

void segment_attention_f16(const float* q, const __half* k, const __half* v, const int* pos, int t,
                           int hq, int hk, int first, int n, int stride, float* state, void* work,
                           bool init, bool finish, __nv_bfloat16* out, cudaStream_t s) {
    if (!q || !k || !v || !pos || !state || !work || !out || t <= 0 || hk <= 0 || hq <= 0 ||
        hq % hk || n <= 0 || stride < n || first < 0)
        throw std::invalid_argument("segment attention geometry");
    int splits = (n + 255) / 256;
    partial_attention<<<dim3(hq, t, splits), 32, 0, s>>>(q, k, v, pos, hq, hk, first, n, stride,
                                                         splits, static_cast<float*>(work));
    merge_attention<<<dim3(hq, t), 32, 0, s>>>(static_cast<float*>(work), hq, splits, state, init,
                                               finish, out);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
