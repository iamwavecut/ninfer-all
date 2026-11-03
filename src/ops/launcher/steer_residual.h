#pragma once

// ninfer::ops::detail - private launch prototype for steer_residual. Included by the wrapper
// (host) and defined by the launcher (.cu). Not part of the public api.
// See docs/op-development.md §2.

#include "ninfer/ops/steer_residual.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Host entry; assumes inputs already validated by the wrapper. `directions_count` is E.
void steer_residual_launch(SteerResidualStep step, Tensor& x, std::int32_t directions_count,
                           const Tensor& directions, const Tensor& factors,
                           const SteerResidualLanes& lanes, Tensor& measures, const Tensor& vector,
                           cudaStream_t stream);

} // namespace ninfer::ops::detail
