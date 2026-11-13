#pragma once

// The small-T partial launches over the FP8 and NVFP4 caches (fp8, nvfp4, k8v4) that their host
// launchers (small_t_fp8.cpp, small_t_nvfp4.cpp, small_t_k8v4.cpp) call, apart from those
// launchers' Tensor interfaces: a kernel unit reaches neither the routing nor the envelope.

#include "ops/softmax_attention/dense/causal_cache/small_t_partial_launch.h"

namespace ninfer::ops::detail {

// The partial kernels (small_t_fp8.cuh, small_t_nvfp4.cuh, small_t_k8v4.cuh) at launch.width
// columns, in their Turing or standard shape. Each format, geometry and input compiles in a unit
// of its own (small_t_fp8_launch.cuh, small_t_nvfp4_launch.cuh, small_t_k8v4_launch.cuh).
template <typename Geometry, typename CacheInput>
void launch_small_t_fp8(const SmallTPartialLaunch& launch, CacheInput input);

template <typename Geometry, typename CacheInput>
void launch_small_t_nvfp4(const SmallTPartialLaunch& launch, CacheInput input);

template <typename Geometry, typename CacheInput>
void launch_small_t_k8v4(const SmallTPartialLaunch& launch, CacheInput input);

} // namespace ninfer::ops::detail
