#pragma once

// Definition of launch_prompt_bf16 (prompt_kernel_launch.h): the BF16 prompt kernel
// (bf16/prompt.cuh) in the current device's shape set, over BF16 or bf16p planes. Included only by
// the generated prompt units (kernel_unit.cu.in).

#include "core/device_launch.h" // CUDA_CHECK, raise_dynamic_shared_memory
#include "ops/common/device_tier.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/prompt.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_kernel_launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_metadata.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_shape.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

static_assert(kPromptWaveRows % kCausalPromptBr == 0);

template <typename Geometry, PromptMetadataForm Form>
void launch_prompt_bf16(const PromptLaunch& launch) {
    using Metadata          = PromptMetadata<Form>;
    const Metadata metadata = prompt_metadata<Form>(launch);
    const dim3 grid(static_cast<unsigned>(div_up(launch.tokens, kCausalPromptBr)),
                    static_cast<unsigned>(launch.q_heads), 1u);
    // The kernel of the current device's shape set, past the default 48 KiB dynamic-smem ceiling in
    // both.
    const auto issue = [&]<typename Plane>(auto kernel, int smem, Plane plane_k, Plane plane_v) {
        raise_dynamic_shared_memory(kernel, smem);
        kernel<<<grid, kCausalPromptThreads, smem, launch.stream>>>(
            static_cast<const __nv_bfloat16*>(launch.q), plane_k, plane_v, metadata,
            launch.positions, launch.scale, static_cast<__nv_bfloat16*>(launch.out),
            launch.tokens);
    };
    // A bf16p cache's kernels decode its rows to the BF16 a bf16 cache holds.
    const auto planes = [&]<typename Plane>(Plane plane_k, Plane plane_v) {
        with_shape_set([&]<bool TuringShapes>() {
            if constexpr (TuringShapes) {
                issue(causal_attention_prompt_bf16_turing_kernel<Geometry, Metadata, Plane>,
                      kCausalPromptBf16TuringSmemBytes, plane_k, plane_v);
            } else {
                issue(causal_attention_prompt_bf16_kernel<Geometry, Metadata, Plane>,
                      kCausalPromptBf16SmemBytes, plane_k, plane_v);
            }
        });
    };
    if (launch.packed_planes) {
        planes(CausalPromptPackedPlane{static_cast<const std::uint8_t*>(launch.cache_k),
                                       static_cast<const std::uint8_t*>(launch.cache_k_scale)},
               CausalPromptPackedPlane{static_cast<const std::uint8_t*>(launch.cache_v),
                                       static_cast<const std::uint8_t*>(launch.cache_v_scale)});
    } else {
        planes(static_cast<const __nv_bfloat16*>(launch.cache_k),
               static_cast<const __nv_bfloat16*>(launch.cache_v));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
