#include "ops/linear/bf16/bf16_general.cuh"
#include "core/device_launch.h"
#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/bf16/bf16_vector.h"

#include <cublas_v2.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// From this width a runtime-shape product takes cuBLAS's tensor-core BF16 GEMM (FP32 accumulator)
// instead of the SIMT tile below; narrower ones, the widths decode graphs capture, keep the kernels
// of this file.
constexpr std::int32_t kCublasMinTokens = 64;

cublasHandle_t blas_handle() {
    static std::array<cublasHandle_t, 16> handles{};
    static std::array<std::once_flag, 16> once;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= int(handles.size())) {
        throw std::invalid_argument("bf16 linear: device ordinal out of range");
    }
    std::call_once(once[device], [&] {
        // The first product may come inside a stream capture: a DSpine drafter's bridge at eight
        // rows of eight or more columns. The handle allocates its workspace as it is created,
        // which the capture's global mode forbids, so this thread relaxes the mode for the call;
        // captured products then use that workspace and allocate nothing.
        cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
        CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        const cublasStatus_t created = cublasCreate(&handles[device]);
        CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        if (created != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("bf16 linear: cublasCreate failed");
        }
    });
    return handles[device];
}

// out[T, N] = x[T, K] * weight[N, K]^T: column-major, the weight is a K x N matrix read
// transposed, x a K x T one and out an N x T one, BF16 or FP32.
void bf16_cublas_gemm(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const cublasHandle_t blas = blas_handle();
    if (cublasSetStream(blas, stream) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("bf16 linear: cublasSetStream failed");
    }
    const float alpha = 1.0f, beta = 0.0f;
    const cublasStatus_t status = cublasGemmEx(
        blas, CUBLAS_OP_T, CUBLAS_OP_N, weight.n, x.ne[1], weight.k, &alpha, weight.qdata,
        CUDA_R_16BF, weight.k, x.data, CUDA_R_16BF, weight.k, &beta, out.data,
        out.dtype == DType::FP32 ? CUDA_R_32F : CUDA_R_16BF, weight.n, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("bf16 linear: cublasGemmEx failed");
    }
}

template <int T, class Out>
void launch_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t N = weight.n;
    bf16_general_gemv_kernel<T, Out>
        <<<static_cast<unsigned>((N + kGemvWarps - 1) / kGemvWarps), kGemvWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.qdata), static_cast<Out*>(out.data), N,
            weight.k);
    CUDA_CHECK(cudaGetLastError());
}

template <int T>
void launch_gemv_together(const Tensor& x, const GemvProducts<__nv_bfloat16>& products,
                          std::int32_t blocks, cudaStream_t stream) {
    bf16_general_gemv_together_kernel<T, __nv_bfloat16>
        <<<static_cast<unsigned>(blocks), kGemvWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), products, x.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

// The GEMV form at a runtime width 1..kGemvMaxTokens, each width its own instantiation.
template <class Out, int T = 1>
void launch_gemv_width(std::int32_t tokens, const Tensor& x, const Weight& weight, Tensor& out,
                       cudaStream_t stream) {
    if constexpr (T < kGemvMaxTokens) {
        if (tokens != T) { return launch_gemv_width<Out, T + 1>(tokens, x, weight, out, stream); }
    }
    launch_gemv<T, Out>(x, weight, out, stream);
}

template <int T = 1>
void launch_gemv_together_width(std::int32_t tokens, const Tensor& x,
                                const GemvProducts<__nv_bfloat16>& products, std::int32_t blocks,
                                cudaStream_t stream) {
    if constexpr (T < kGemvMaxTokens) {
        if (tokens != T) {
            return launch_gemv_together_width<T + 1>(tokens, x, products, blocks, stream);
        }
    }
    launch_gemv_together<T>(x, products, blocks, stream);
}

bool aligned16(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0; }

// Runtime-shape bf16 GEMM used for every linear projection the specialised bf16 kernels do not
// tile and every FP32 result. x is [tokens, k], weight is [n, k] row-major, and out is [tokens, n]
// token-major, matching the contiguous layout the specialised kernels produce.
template <class Out>
void bf16_general_gemm_as(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t N = weight.n;
    const std::int32_t K = weight.k;
    const std::int32_t T = x.ne[1];
    if (T >= kCublasMinTokens) { return bf16_cublas_gemm(x, weight, out, stream); }
    if (T <= kGemvMaxTokens && K % 8 == 0 && aligned16(x.data) && aligned16(weight.qdata)) {
        return launch_gemv_width<Out>(T, x, weight, out, stream);
    }
    const dim3 grid(static_cast<unsigned>((N + kTileN - 1) / kTileN),
                    static_cast<unsigned>((T + kTileT - 1) / kTileT));
    bf16_general_gemm_kernel<Out><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.qdata),
        static_cast<Out*>(out.data), N, K, T);
    CUDA_CHECK(cudaGetLastError());
}

void bf16_general_gemm(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (out.dtype == DType::FP32) { return bf16_general_gemm_as<float>(x, weight, out, stream); }
    bf16_general_gemm_as<__nv_bfloat16>(x, weight, out, stream);
}

} // namespace

bool bf16_general_vector(const Tensor& x, const Weight& weight) {
    const std::int32_t T = x.ne[1];
    return T < kCublasMinTokens && T <= kGemvMaxTokens && weight.k % 8 == 0 &&
           aligned16(x.data) && aligned16(weight.qdata);
}

void bf16_general_vector_together(const Tensor& x, std::span<const Bf16VectorProduct> products,
                                  cudaStream_t stream) {
    for (const Bf16VectorProduct& product : products) {
        if (product.out->dtype != DType::BF16 || product.weight->k != x.ne[0] ||
            !bf16_general_vector(x, *product.weight)) {
            throw std::invalid_argument("bf16 vector products: a product outside the GEMV form");
        }
    }
    for (std::size_t first = 0; first < products.size(); first += kGemvTogether) {
        GemvProducts<__nv_bfloat16> launch{};
        std::int32_t blocks = 0;
        for (std::size_t i = first; i < products.size() && launch.count < kGemvTogether; ++i) {
            const Bf16VectorProduct& product = products[i];
            launch.weight[launch.count] = static_cast<const __nv_bfloat16*>(product.weight->qdata);
            launch.out[launch.count]    = static_cast<__nv_bfloat16*>(product.out->data);
            launch.n[launch.count]      = product.weight->n;
            launch.first_block[launch.count] = blocks;
            blocks += (product.weight->n + kGemvWarps - 1) / kGemvWarps;
            ++launch.count;
        }
        launch_gemv_together_width(x.ne[1], x, launch, blocks, stream);
    }
}

[[nodiscard]] Bf16Launch select_bf16_general_launch(std::int32_t tokens) {
    (void)tokens;
    return &bf16_general_gemm;
}

} // namespace ninfer::ops::detail
