#include "ops/linear/fp8/fp8_fused_norm_quantize.h"

#include "core/device.h"
#include "ops/common/blackwell_routes.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_fused_norm_quantize.cuh"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kThreads5120 = 256;  // 5120 / 2 / 256 = 10 bf16x2 per thread
constexpr int kThreads6144 = 768;  // 24 warps, two 128-element heads of the 6144 row per warp

} // namespace

void launch_fp8_fused_rmsnorm_quantize(const Tensor& x, const Tensor& norm_weight, float eps,
                                       std::int32_t input_rows, Fp8A8Workspace workspace,
                                       cudaStream_t stream) {
    require_fp8_a8_card();
    if (workspace.codes == nullptr || workspace.scales == nullptr) {
        throw std::invalid_argument("fp8 fused rmsnorm quantize: requires caller workspace");
    }
    if (input_rows != Fp8Activation5120Geometry::kInputRows) {
        throw std::invalid_argument("fp8 fused rmsnorm quantize: unsupported K");
    }
    CUDA_CHECK(pdl::launch_consumer(
        {dim3(x.ne[1]), dim3(kThreads5120), 0, stream},
        fp8_fused_norm_quantize_kernel<RmsEpilogue::Offset, kThreads5120,
                                       Fp8Activation5120Geometry::kInputRows>,
        static_cast<const __nv_bfloat162*>(x.data),
        static_cast<const __nv_bfloat162*>(norm_weight.data),
        static_cast<const __nv_bfloat162*>(nullptr), workspace.codes, workspace.scales, eps));
}

void launch_fp8_fused_gated_rmsnorm_quantize(const Tensor& x, const Tensor& norm_weight,
                                             const Tensor& z, float eps, std::int32_t input_rows,
                                             Fp8A8Workspace workspace, cudaStream_t stream) {
    require_fp8_a8_card();
    if (workspace.codes == nullptr || workspace.scales == nullptr) {
        throw std::invalid_argument("fp8 fused gated rmsnorm quantize: requires caller workspace");
    }
    if (input_rows != Fp8Activation6144Geometry::kInputRows) {
        throw std::invalid_argument("fp8 fused gated rmsnorm quantize: unsupported K");
    }
    CUDA_CHECK(pdl::launch_consumer(
        {dim3(x.ne[1]), dim3(kThreads6144), 0, stream},
        fp8_fused_gated_head_rmsnorm_quantize_kernel<kThreads6144>,
        static_cast<const __nv_bfloat162*>(x.data),
        static_cast<const __nv_bfloat162*>(norm_weight.data),
        static_cast<const __nv_bfloat162*>(z.data), workspace.codes, workspace.scales, eps));
}

} // namespace ninfer::ops::detail
