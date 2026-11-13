#pragma once

// The BF16 and bf16p small-T partial launches the host launcher (small_t.cpp) calls. Each geometry
// and input compiles in a unit of its own (kernel_unit.cu.in), over every query width it takes:
// an edit of the routing compiles no kernel, and an edit of one kernel only its units.

#include "ops/softmax_attention/dense/causal_cache/cache_input.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_partial_launch.h"

namespace ninfer::ops::detail {

// The tiled kernel (bf16/small_t.cuh) at launch.width columns, in its Turing or standard shape.
template <typename Geometry, typename CacheInput>
void launch_small_t_bf16(const SmallTPartialLaunch& launch, CacheInput input);

// Volta's kernel (bf16/small_t_sm70.cuh), in builds with sm_70: appending and cached inputs.
template <typename Geometry, typename CacheInput>
void launch_small_t_bf16_sm70(const SmallTPartialLaunch& launch, CacheInput input);

} // namespace ninfer::ops::detail
