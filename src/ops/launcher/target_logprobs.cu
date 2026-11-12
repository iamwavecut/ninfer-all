// Implements: include/ninfer/ops/target_logprobs.h
// Match: wrapper-validated contiguous tensors and valid vocabulary rows.
// Algorithm assumptions: one independent CTA per column; no global workspace.
#include "ops/launcher/target_logprobs.h"

#include "core/device_launch.h"
#include "ops/kernel/target_logprobs.cuh"

namespace ninfer::ops::detail {

void target_logprobs_launch(const Tensor& logits, const Tensor& target_ids, std::int32_t valid_rows,
                            Tensor& output, cudaStream_t stream) {
    const auto columns = static_cast<unsigned int>(logits.ne[1]);
    const auto launch  = [&]<typename Logit>() {
        target_logprobs_kernel<kTargetLogprobsBlock, Logit>
            <<<columns, kTargetLogprobsBlock, 0, stream>>>(
                static_cast<const Logit*>(logits.data),
                static_cast<const std::int32_t*>(target_ids.data), static_cast<float*>(output.data),
                valid_rows, logits.ne[0]);
    };
    if (logits.dtype == DType::FP32) {
        launch.template operator()<float>();
    } else {
        launch.template operator()<__nv_bfloat16>();
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
