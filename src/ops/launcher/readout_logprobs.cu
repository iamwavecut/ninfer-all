// Implements: include/ninfer/ops/readout_logprobs.h
// Match: wrapper-validated contiguous tensors and valid vocabulary rows.
// Algorithm assumptions: one CTA reads the whole column; no global workspace.
#include "ops/launcher/readout_logprobs.h"

#include "core/device_launch.h"
#include "ops/kernel/readout_logprobs.cuh"

namespace ninfer::ops::detail {

void readout_logprobs_launch(const Tensor& logits, const Tensor& token_ids, std::int32_t count,
                             std::int32_t valid_rows, Tensor& output, cudaStream_t stream) {
    const auto launch = [&]<typename Logit>() {
        readout_logprobs_kernel<kReadoutLogprobsBlock, Logit>
            <<<1, kReadoutLogprobsBlock, 0, stream>>>(
                static_cast<const Logit*>(logits.data),
                static_cast<const std::int32_t*>(token_ids.data), static_cast<float*>(output.data),
                count, valid_rows);
    };
    if (logits.dtype == DType::FP32) {
        launch.template operator()<float>();
    } else {
        launch.template operator()<__nv_bfloat16>();
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
