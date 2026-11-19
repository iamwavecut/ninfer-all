#pragma once

// ninfer::ops - gdn_gating kernel: elementwise GDN gate prep over [H,T].
// Transcendentals use fp32 CUDA math functions, not polynomial approximations.

#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops {

__global__ void gdn_gating_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b,
                                  const float* A_log, const float* dt_bias, float* g, float* beta,
                                  std::int64_t n, std::int32_t heads, bool a_is_log) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const int h    = static_cast<int>(i % heads);
        const float av = __bfloat162float(a[i]);
        const float bv = __bfloat162float(b[i]);
        const float sp = softplus(av + dt_bias[h]);
        g[i]           = (a_is_log ? -expf(A_log[h]) : A_log[h]) * sp;
        beta[i]        = sigmoid(bv);
    }
}

} // namespace ninfer::ops
