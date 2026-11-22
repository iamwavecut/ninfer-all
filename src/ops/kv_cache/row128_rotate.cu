#include "ops/kv_cache/row128_rotate.h"

#include "ops/kv_cache/row128_codec.cuh"

#include <stdexcept>
#include <string>

namespace ninfer::ops::kv128 {
namespace {

constexpr int kWarps = 4; // warps per block, one row each

__launch_bounds__(kWarps * 32) __global__ void rotate_queries_kernel(float* q, std::int64_t count) {
    const int lane = threadIdx.x & 31;
    const std::int64_t row = std::int64_t(blockIdx.x) * kWarps + (threadIdx.x >> 5);
    if (row >= count) { return; }
    float* x = q + row * kKvRowDims;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { v[i] = x[lane + 32 * i]; }
    hadamard128(v, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) { x[lane + 32 * i] = __fmul_rn(v[i], kRotatedQueryScale); }
}

} // namespace

void rotate_queries(float* rows, std::int64_t count, cudaStream_t stream) {
    if (count <= 0) { return; }
    const auto blocks = static_cast<unsigned>((count + kWarps - 1) / kWarps);
    rotate_queries_kernel<<<blocks, kWarps * 32, 0, stream>>>(rows, count);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("kv128 rotate_queries: ") +
                                 cudaGetErrorString(status));
    }
}

} // namespace ninfer::ops::kv128
