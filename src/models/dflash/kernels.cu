#include "models/dflash/kernels.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::models::dflash::kernels {
namespace {

constexpr int kThreads    = 256;
constexpr int kCandidates = 16;

void check_launch(const char* what) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("dflash ") + what + ": " + cudaGetErrorString(status));
    }
}

std::int64_t elements(const Tensor& t) {
    std::int64_t n = 1;
    for (int d = 0; d < 4; ++d) { n *= t.ne[d] > 0 ? t.ne[d] : 1; }
    return n;
}

__global__ void softcap_kernel(float* scores, std::int64_t n, float cap) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { scores[i] = cap * tanhf(scores[i] / cap); }
}

__global__ void best_candidate_kernel(const std::int32_t* __restrict__ ids,
                                      const float* __restrict__ scores, int columns,
                                      std::int32_t* __restrict__ drafts,
                                      float* __restrict__ confidence, float* __restrict__ q) {
    const int u = int(blockIdx.x) * blockDim.x + int(threadIdx.x);
    if (u >= columns) { return; }
    const float* s = scores + std::int64_t(u) * kCandidates;
    drafts[u]      = ids[std::int64_t(u) * kCandidates];
    float* row     = q + std::int64_t(u) * kCandidates;
    row[0]         = 1.0F;
    for (int c = 1; c < kCandidates; ++c) { row[c] = 0.0F; }
    if (confidence == nullptr) { return; }
    const float top = s[0];
    if (!isfinite(top)) {
        confidence[u] = 0.0F;
        return;
    }
    // The candidates descend, so s[0] is the maximum; -inf slots add nothing.
    float sum = 0.0F;
    for (int c = 0; c < kCandidates; ++c) { sum += expf(s[c] - top); }
    confidence[u] = 1.0F / sum;
}

__global__ void fill_bf16_kernel(__nv_bfloat16* out, std::int64_t n, __nv_bfloat16 value) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = value; }
}

unsigned blocks(std::int64_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

} // namespace

void softcap(Tensor& scores, float cap, cudaStream_t stream) {
    if (scores.dtype != DType::FP32 || !scores.is_contiguous() || !(cap > 0) ||
        !std::isfinite(cap)) {
        throw std::invalid_argument("dflash softcap: contiguous FP32 scores and a positive cap");
    }
    const std::int64_t n = elements(scores);
    if (n == 0) { return; }
    softcap_kernel<<<blocks(n), kThreads, 0, stream>>>(static_cast<float*>(scores.data), n, cap);
    check_launch("softcap");
}

void best_candidate(const Tensor& ids, const Tensor& scores, Tensor& drafts, Tensor& confidence,
                    Tensor& proposal_q, cudaStream_t stream) {
    const std::int64_t cells = elements(ids);
    const std::int64_t u     = cells / kCandidates;
    if (ids.dtype != DType::I32 || scores.dtype != DType::FP32 || drafts.dtype != DType::I32 ||
        proposal_q.dtype != DType::FP32 || ids.ne[0] != kCandidates || cells % kCandidates ||
        elements(scores) != cells || elements(proposal_q) != cells || elements(drafts) != u ||
        (confidence.data != nullptr &&
         (confidence.dtype != DType::FP32 || elements(confidence) != u)) ||
        !ids.is_contiguous() || !scores.is_contiguous() || !drafts.is_contiguous() ||
        !proposal_q.is_contiguous()) {
        throw std::invalid_argument("dflash best_candidate: ids I32 and scores FP32 [16, U], "
                                    "drafts I32 [U], q FP32 [16, U], confidence FP32 [U]");
    }
    if (u == 0) { return; }
    best_candidate_kernel<<<blocks(u), kThreads, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const float*>(scores.data),
        static_cast<int>(u), static_cast<std::int32_t*>(drafts.data),
        confidence.data != nullptr ? static_cast<float*>(confidence.data) : nullptr,
        static_cast<float*>(proposal_q.data));
    check_launch("best_candidate");
}

void fill_bf16(Tensor& out, float value, cudaStream_t stream) {
    if (out.dtype != DType::BF16 || !out.is_contiguous()) {
        throw std::invalid_argument("dflash fill_bf16: a contiguous BF16 tensor");
    }
    const std::int64_t n = elements(out);
    if (n == 0) { return; }
    fill_bf16_kernel<<<blocks(n), kThreads, 0, stream>>>(static_cast<__nv_bfloat16*>(out.data), n,
                                                         __float2bfloat16(value));
    check_launch("fill_bf16");
}

} // namespace ninfer::models::dflash::kernels
