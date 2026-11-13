#pragma once

// The small-T partial launches over the FP8 and NVFP4 caches (fp8, nvfp4, k8v4) that their host
// launchers (fp8/small_t.cpp, nvfp4/small_t.cpp, k8v4/small_t.cpp) call, apart from those
// launchers' Tensor interfaces: a kernel unit reaches neither the routing nor the envelope.

#include "ops/softmax_attention/dense/causal_cache/small_t_partial_launch.h"

namespace ninfer::ops::detail {

// The partial kernels (fp8/small_t.cuh, nvfp4/small_t.cuh, k8v4/small_t.cuh) at launch.width
// columns, in their Turing or standard shape. Each format, geometry and input compiles in a unit
// of its own (fp8/small_t_launch.cuh, nvfp4/small_t_launch.cuh, k8v4/small_t_launch.cuh).
template <typename Geometry, typename CacheInput>
void launch_small_t_fp8(const SmallTPartialLaunch& launch, CacheInput input);

template <typename Geometry, typename CacheInput>
void launch_small_t_nvfp4(const SmallTPartialLaunch& launch, CacheInput input);

template <typename Geometry, typename CacheInput>
void launch_small_t_k8v4(const SmallTPartialLaunch& launch, CacheInput input);

} // namespace ninfer::ops::detail
