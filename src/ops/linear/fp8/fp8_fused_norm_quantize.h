#pragma once

#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cstdint>

namespace ninfer::ops::detail {

// The A8 quantiser with the producer norm folded in: one launch writes the per-token E4M3 codes
// and scale for `x`'s normalized rows, the bit-for-bit output of running the named norm to BF16
// storage and launch_fp8_a8_quantize on it. The normalized row is never materialized. The caller
// owns the Fp8A8Workspace, sized for x.ne[1] tokens and input_rows.
//
// launch_fp8_fused_rmsnorm_quantize: the Offset norm (gain 1 + weight) at K=5120, the layer
// input and post-attention norms' shape.
// launch_fp8_fused_gated_rmsnorm_quantize: the per-head SiLU-gated norm (no unit offset) at
// K=6144, the gated delta net's output norm shape: the 6144 row holds 48 heads of 128 and
// `norm_weight` is the head's 128 BF16 gain, `z` the gate row.
void launch_fp8_fused_rmsnorm_quantize(const Tensor& x, const Tensor& norm_weight, float eps,
                                       std::int32_t input_rows, Fp8A8Workspace workspace,
                                       cudaStream_t stream);

void launch_fp8_fused_gated_rmsnorm_quantize(const Tensor& x, const Tensor& norm_weight,
                                             const Tensor& z, float eps, std::int32_t input_rows,
                                             Fp8A8Workspace workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
