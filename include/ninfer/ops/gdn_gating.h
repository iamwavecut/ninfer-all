#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Prepares Gated DeltaNet decay and update gates:
 *
 *   g[h,t]    = -exp(A_log[h]) * softplus(a[h,t] + dt_bias[h])
 *   beta[h,t] = sigmoid(b[h,t]).
 * With a_is_log=false, A_log stores the negative decay coefficient directly and g uses
 * A_log[h] * softplus(a[h,t] + dt_bias[h]), preserving an already exponentiated weight.
 *
 * `a` and `b` are contiguous BF16 [H,T] for any H > 0 (48 for a whole layer, a tensor-parallel
 * rank's share of its value heads otherwise), `A_log` and `dt_bias` are contiguous FP32 [H], and
 * `g` and `beta` are contiguous FP32 [H,T]. The oracle evaluates the formula naively in FP64;
 * transcendental implementation and intermediate precision are private kernel choices. Inputs and
 * the two outputs must be mutually non-overlapping. There is no workspace or persistent state side
 * effect.
 */
void gdn_gating(const Tensor& a, const Tensor& b, const Tensor& A_log, const Tensor& dt_bias,
                Tensor& g, Tensor& beta, cudaStream_t stream, bool a_is_log = true);

} // namespace ninfer::ops
