#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kFp8LinearAddChunkTokens = 24;

[[nodiscard]] std::size_t fp8_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                                  std::int32_t input_rows,
                                                                  LinearPolicy policy,
                                                                  std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

void fp8_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  cudaStream_t stream);
void fp8_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                   cudaStream_t stream);
void fp8_linear_add_a8_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                              WorkspaceArena& workspace, cudaStream_t stream);

// The A8 fused-norm producer's answer: true exactly when the 5120x6144 problem's legacy route is
// the A8 MMA route the fused entry feeds (the unified chain's A8 schedule is its TMA route, which
// the entry does not serve). A permissive policy that the resolver would answer A16 here is not
// eligible. The 5120x17408 problem has no fused entry.
[[nodiscard]] bool fp8_linear_add_fused_gated_eligible(LinearPolicy policy, std::int32_t tokens);

// The A8 route with the SiLU-gated RMSNorm folded into the quantize; 5120x6144 only. `o` is the
// gated delta net's output row, `z` its gate row; the fused producer writes the same
// Fp8A8Workspace the separate launch_fp8_a8_quantize would, so the result is the composition of
// gated_rmsnorm() into BF16 storage and this route.
void fp8_linear_add_fused_gated_launch(const Tensor& o, const Tensor& norm_weight, const Tensor& z,
                                       float eps, const Weight& weight, Tensor& residual,
                                       WorkspaceArena& workspace, cudaStream_t stream);

void fp8_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                             LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
