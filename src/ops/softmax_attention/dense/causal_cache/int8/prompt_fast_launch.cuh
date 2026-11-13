#pragma once

// Definition of launch_prompt_i8_fast (prompt_kernel_launch.h): the fast INT8-family prompt kernel
// (int8/prompt_fast.cuh) over every coding and CTA shape of its plan. Included only by the
// generated prompt units (kernel_unit.cu.in).

#include "core/device_launch.h" // CUDA_CHECK, raise_dynamic_shared_memory
#include "ops/common/device_tier.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/int8/prompt_fast.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_kernel_launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_metadata.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_shape.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

static_assert(kPromptWaveRows == CausalPromptI8FastShape<8>::Br);
static_assert(kPromptWaveRows % CausalPromptI8FastShape<4>::Br == 0);

// The int8 cache and rk8v4 share the fast kernel: their keys are the same INT8 G64 codes, and
// PackedValues decodes rk8v4's packed int4 value plane; Keys expands the packed key codings. The
// plan (int8/prompt_fast_plan.h) picks the CTA shape and, past the envelope's key count, may split
// every row block's keys over several CTAs whose partial rows the merge kernel combines.
template <typename Geometry, PromptMetadataForm Form>
void launch_prompt_i8_fast(const PromptLaunch& launch) {
    using Metadata          = PromptMetadata<Form>;
    const Metadata metadata = prompt_metadata<Form>(launch);
    const auto shaped       = [&]<bool PackedValues, KvKeyCoding Keys>() {
        with_shape_set([&]<bool TuringShapes>() {
            // Eight warps with INT8 values do not fit Turing's shared memory; those launches take
            // four.
            constexpr bool kWide =
                CausalPromptI8FastShape<8, PackedValues, TuringShapes>::Launchable;
            const auto issue = [&]<int Warps, bool Split>() {
                using Shape = CausalPromptI8FastShape<Warps, PackedValues, TuringShapes>;
                constexpr auto kernel =
                    causal_attention_prompt_i8_fast_kernel<Geometry, Metadata, Warps, PackedValues,
                                                           Keys, TuringShapes, Split>;
                raise_dynamic_shared_memory(kernel, Shape::SmemBytes);
                const dim3 grid(static_cast<unsigned>(div_up(launch.tokens, Shape::Br)),
                                static_cast<unsigned>(launch.q_heads),
                                static_cast<unsigned>(launch.splits));
                kernel<<<grid, Shape::Threads, Shape::SmemBytes, launch.stream>>>(
                    static_cast<const __nv_bfloat16*>(launch.q),
                    static_cast<const std::int8_t*>(launch.cache_k),
                    static_cast<const std::int8_t*>(launch.cache_v),
                    static_cast<const __half*>(launch.cache_k_scale),
                    static_cast<const __half*>(launch.cache_v_scale), metadata, launch.positions,
                    launch.scale, static_cast<__nv_bfloat16*>(launch.out), launch.tokens,
                    launch.split_rows, launch.split_stats);
            };
            if (launch.splits == 1) {
                if constexpr (kWide) {
                    if (launch.warps == 8) {
                        issue.template operator()<8, false>();
                        return;
                    }
                }
                issue.template operator()<4, false>();
                return;
            }
            // A split launch takes the widest CTA the shape set launches
            // (causal_prompt_i8_fast_plan).
            issue.template operator()<kWide ? 8 : 4, true>();
        });
    };
    switch (launch.coding) {
    case CausalI8Coding::Int8:
        shaped.template operator()<false, KvKeyCoding::Int8>();
        break;
    case CausalI8Coding::PackedValues:
        shaped.template operator()<true, KvKeyCoding::Int8>();
        break;
    case CausalI8Coding::Lloyd4:
        shaped.template operator()<true, KvKeyCoding::Lloyd4>();
        break;
    case CausalI8Coding::Int4E8:
        shaped.template operator()<true, KvKeyCoding::Int4E8>();
        break;
    case CausalI8Coding::RootE8:
        shaped.template operator()<true, KvKeyCoding::RootE8>();
        break;
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
