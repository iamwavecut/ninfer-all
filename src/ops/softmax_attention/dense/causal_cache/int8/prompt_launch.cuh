#pragma once

// Definition of launch_prompt_i8 (prompt_kernel_launch.h): the standard INT8-family prompt kernel
// (int8/prompt.cuh) over the codings with packed int4 values or over INT8 values, every PV
// accumulation and GQA packing. Included only by the generated prompt units (kernel_unit.cu.in).

#include "core/device_launch.h" // CUDA_CHECK, raise_dynamic_shared_memory
#include "ops/common/device_tier.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/int8/prompt.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_kernel_launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_metadata.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_shape.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

static_assert(kPromptWaveRows % kCausalPromptI8Br == 0);

template <typename Geometry, PromptMetadataForm Form, bool PackedValueCodings>
void launch_prompt_i8(const PromptLaunch& launch) {
    // A build of Turing's shape set alone does not compile the standard kernel.
    if constexpr (kBuildStandardShapes) {
        using Metadata          = PromptMetadata<Form>;
        const Metadata metadata = prompt_metadata<Form>(launch);
        const std::int32_t tokens = launch.tokens;
        // The INT8 family is the one place that deliberately does not take its plane types from
        // ops/kv_cache/plane_types.h. One kernel serves every coding, so its value parameter is
        // std::int8_t* for int8-g64 and for rk8v4 alike, and the packed-int4 path re-casts to
        // std::uint8_t where it unpacks (causal_prompt_i4_dequant_f16x8).
        const auto issue = [&]<bool PackedValues, KvKeyCoding Keys, bool PvF16, bool PackGqa>() {
            const auto kernel = causal_attention_prompt_i8_kernel<Geometry, Metadata, PackedValues,
                                                                  Keys, PvF16, PackGqa>;
            // The kernel exceeds the default 48 KiB dynamic-smem ceiling.
            raise_dynamic_shared_memory(kernel, kCausalPromptI8SmemBytes);
            const dim3 grid =
                PackGqa ? dim3(static_cast<unsigned>(
                                   div_up(tokens * Geometry::GroupSize, kCausalPromptI8Br)),
                               static_cast<unsigned>(launch.q_heads / Geometry::GroupSize), 1u)
                        : dim3(static_cast<unsigned>(div_up(tokens, kCausalPromptI8Br)),
                               static_cast<unsigned>(launch.q_heads), 1u);
            kernel<<<grid, kCausalPromptI8Threads, kCausalPromptI8SmemBytes, launch.stream>>>(
                static_cast<const __nv_bfloat16*>(launch.q),
                static_cast<const std::int8_t*>(launch.cache_k),
                static_cast<const std::int8_t*>(launch.cache_v),
                static_cast<const __half*>(launch.cache_k_scale),
                static_cast<const __half*>(launch.cache_v_scale), metadata, launch.positions,
                launch.scale, static_cast<__nv_bfloat16*>(launch.out), tokens);
        };
        // A Volta card emulates these tiles: one variant per coding keeps ptxas within bounds, and
        // a build for Volta alone compiles no other. The launcher clears pack_gqa and pv_f16 there.
        const auto pack = [&]<bool PackedValues, KvKeyCoding Keys, bool PvF16>() {
            if constexpr (kBuildPostVolta) {
                if (launch.pack_gqa) {
                    issue.template operator()<PackedValues, Keys, PvF16, true>();
                    return;
                }
            }
            issue.template operator()<PackedValues, Keys, PvF16, false>();
        };
        // FP16 PV sums where the launcher asks for them.
        const auto with_pv = [&]<bool PackedValues, KvKeyCoding Keys>() {
            if constexpr (kBuildPostVolta) {
                if (launch.pv_f16) {
                    pack.template operator()<PackedValues, Keys, true>();
                    return;
                }
            }
            pack.template operator()<PackedValues, Keys, false>();
        };
        if constexpr (PackedValueCodings) {
            switch (launch.coding) {
            case CausalI8Coding::Lloyd4:
                with_pv.template operator()<true, KvKeyCoding::Lloyd4>();
                break;
            case CausalI8Coding::Int4E8:
                with_pv.template operator()<true, KvKeyCoding::Int4E8>();
                break;
            case CausalI8Coding::RootE8:
                pack.template operator()<true, KvKeyCoding::RootE8, false>();
                break;
            case CausalI8Coding::PackedValues:
                with_pv.template operator()<true, KvKeyCoding::Int8>();
                break;
            case CausalI8Coding::Int8:
                throw std::logic_error("INT8 values reached the packed codings' prompt launch");
            }
        } else {
            pack.template operator()<false, KvKeyCoding::Int8, false>();
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
