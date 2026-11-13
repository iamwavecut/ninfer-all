#pragma once

// Volta's small-T partial launch over the FP8 and NVFP4 caches (fp8, nvfp4, k8v4), which their host
// launchers call on an sm_70 card. Each geometry, input and coding compiles in a unit of its own
// (kernel_unit.cu.in) for sm_70 alone.

#include "ops/softmax_attention/dense/causal_cache/cache_input.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_partial_launch.h"

namespace ninfer::ops::detail {

// The kernel of sm70/small_t_quant.cuh at launch.width columns: FP8 or NVFP4 keys (KeyFp8) and
// values (ValueFp8).
template <typename Geometry, typename CacheInput, bool KeyFp8, bool ValueFp8>
void launch_small_t_quantized_sm70(const SmallTPartialLaunch& launch, CacheInput input);

} // namespace ninfer::ops::detail
