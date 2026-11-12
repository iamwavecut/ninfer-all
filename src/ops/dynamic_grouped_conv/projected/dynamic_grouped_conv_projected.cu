// ninfer::ops - one side of a DFlash2 dynamic grouped convolution over coefficients a Linear call
// projected (contract in include/ninfer/ops/dynamic_grouped_conv.h). One thread per output element:
// the convolution reads two inputs and two coefficients of its column, so the Op is bound by its
// handful of BF16/FP32 loads and needs no staging.
#include "ninfer/ops/dynamic_grouped_conv.h"

#include "core/device_launch.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kGroupWidth = 16;

template <typename T>
__device__ __forceinline__ float load(const T* p, std::int64_t i) {
    if constexpr (std::is_same_v<T, float>) {
        return p[i];
    } else {
        return __bfloat162float(p[i]);
    }
}

// Element e = (b * W + i) * H + h of the [H, W, B] planes.
template <typename X, typename C>
__global__ void __launch_bounds__(256)
    projected_conv_kernel(const X* __restrict__ x, const C* __restrict__ coefficients,
                          const __nv_bfloat16* __restrict__ base, __nv_bfloat16* __restrict__ out,
                          std::int32_t hidden, std::int32_t width, std::int32_t side,
                          bool accumulate, std::int64_t elements) {
    const std::int64_t e = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= elements) return;
    const std::int64_t column = e / hidden;
    const std::int32_t h      = static_cast<std::int32_t>(e % hidden);
    const std::int32_t i      = static_cast<std::int32_t>(column % width);
    const std::int32_t groups = hidden / kGroupWidth;
    const std::int32_t g      = h / kGroupWidth;
    // base_kernel [channel, tap, side]: tap t of side s at (s * 2 + t) * H + h.
    const float b0 = __bfloat162float(base[std::int64_t(side * 2 + 0) * hidden + h]);
    const C* row   = coefficients + column * 4 * groups;
    float y        = (b0 + load(row, std::int64_t(side * 2 + 0) * groups + g)) * load(x, e);
    if (i > 0) {
        const float b1 = __bfloat162float(base[std::int64_t(side * 2 + 1) * hidden + h]);
        y = fmaf(b1 + load(row, std::int64_t(side * 2 + 1) * groups + g), load(x, e - hidden), y);
    }
    out[e] = __float2bfloat16_rn(accumulate ? __bfloat162float(out[e]) + y : y);
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(std::string("dynamic_grouped_conv: ") + message);
    }
}

bool contiguous(const Tensor& t, std::int32_t d0, std::int32_t d1, std::int32_t d2) {
    return t.data != nullptr && t.is_contiguous() && t.ne[0] == d0 && t.ne[1] == d1 &&
           t.ne[2] == d2 && t.ne[3] == 1;
}

template <typename X, typename C>
void launch(const Tensor& x, const Tensor& coefficients, const Tensor& base_kernel,
            std::int32_t side, Tensor& out, cudaStream_t stream) {
    const std::int64_t elements = std::int64_t(x.ne[0]) * x.ne[1] * x.ne[2];
    projected_conv_kernel<X, C><<<static_cast<unsigned>((elements + 255) / 256), 256, 0, stream>>>(
        static_cast<const X*>(x.data), static_cast<const C*>(coefficients.data),
        static_cast<const __nv_bfloat16*>(base_kernel.data), static_cast<__nv_bfloat16*>(out.data),
        x.ne[0], x.ne[1], side, side == 1, elements);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void dynamic_grouped_conv(const Tensor& x, const Tensor& coefficients, const Tensor& base_kernel,
                          std::int32_t side, Tensor& out, cudaStream_t stream) {
    const std::int32_t hidden = x.ne[0], width = x.ne[1], batch = x.ne[2];
    require(side == 0 || side == 1, "side must be 0 or 1");
    require(hidden > 0 && hidden % kGroupWidth == 0, "H must be a positive multiple of 16");
    require(width >= 1 && width <= 16 && batch >= 1 && batch <= 8,
            "W must be in [1,16] and B in [1,8]");
    require((x.dtype == DType::BF16 || x.dtype == DType::FP32) &&
                contiguous(x, hidden, width, batch),
            "x must be contiguous BF16 or FP32 [H,W,B]");
    const std::int32_t rows = 4 * (hidden / kGroupWidth);
    require((coefficients.dtype == DType::FP32 || coefficients.dtype == DType::BF16) &&
                contiguous(coefficients, rows, width, batch),
            "coefficients must be contiguous FP32 or BF16 [4H/16,W,B]");
    require(base_kernel.dtype == DType::BF16 && contiguous(base_kernel, hidden, 2, 2),
            "base_kernel must be contiguous BF16 [H,2,2]");
    require(out.dtype == DType::BF16 && contiguous(out, hidden, width, batch),
            "out must be contiguous BF16 [H,W,B]");
    const auto begin = [](const Tensor& t) { return reinterpret_cast<std::uintptr_t>(t.data); };
    const auto end   = [&](const Tensor& t) { return begin(t) + t.bytes(); };
    for (const Tensor* input : {&x, &coefficients, &base_kernel}) {
        require(end(*input) <= begin(out) || end(out) <= begin(*input),
                "out must not overlap an input");
    }
    const bool fx = x.dtype == DType::FP32, fc = coefficients.dtype == DType::FP32;
    if (fx && fc) {
        launch<float, float>(x, coefficients, base_kernel, side, out, stream);
    } else if (fx) {
        launch<float, __nv_bfloat16>(x, coefficients, base_kernel, side, out, stream);
    } else if (fc) {
        launch<__nv_bfloat16, float>(x, coefficients, base_kernel, side, out, stream);
    } else {
        launch<__nv_bfloat16, __nv_bfloat16>(x, coefficients, base_kernel, side, out, stream);
    }
}

} // namespace ninfer::ops
