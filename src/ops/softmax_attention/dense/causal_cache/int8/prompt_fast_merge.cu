// ninfer::ops::detail - the fast INT8-family prompt kernel's split merge (the merge kernel the fast
// FP8 kernel shares, prompt_fast_merge.cuh) and the CTA shapes its plan may take
// (prompt_kernel_launch.h).
#include "ops/softmax_attention/dense/causal_cache/prompt_kernel_launch.h"

#include "core/device_launch.h" // CUDA_CHECK
#include "ops/common/device_tier.h"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_fast_merge.cuh"
#include "ops/softmax_attention/dense/causal_cache/int8/prompt_fast.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

void launch_prompt_i8_fast_merge(const PromptLaunch& launch) {
    // A build for Volta alone launches no fast kernel (causal_attention_prompt_fast_kernel).
    if constexpr (kBuildPostVolta) {
        constexpr float kLog2E = 1.4426950408889634074f;
        with_causal_query_geometry(launch.q_heads, [&]<typename Geometry>() {
            causal_attention_prompt_fast_merge_kernel<Geometry>
                <<<dim3(static_cast<unsigned>(launch.tokens),
                        static_cast<unsigned>(launch.q_heads)),
                   kCausalPromptHeadDim, 0, launch.stream>>>(
                    launch.split_rows, launch.split_stats, launch.valid_columns, launch.tokens,
                    launch.splits, launch.scale * kLog2E,
                    static_cast<__nv_bfloat16*>(launch.out));
        });
        CUDA_CHECK(cudaGetLastError());
    }
}

bool prompt_i8_fast_wide_launchable(bool packed, bool turing_shapes) {
    if (turing_shapes) {
        return packed ? CausalPromptI8FastShape<8, true, true>::Launchable
                      : CausalPromptI8FastShape<8, false, true>::Launchable;
    }
    return packed ? CausalPromptI8FastShape<8, true, false>::Launchable
                  : CausalPromptI8FastShape<8, false, false>::Launchable;
}

} // namespace ninfer::ops::detail
