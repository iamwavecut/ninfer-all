#pragma once

// ninfer::ops::detail - which prompt kernels a card and a storage take: the standard INT8-family
// kernel where its tile fits, the fast kernel's switch and its key splits. Host code only; the
// shape facts the kernel units share are in prompt_shape.h.

#include "ninfer/kv_cache_storage.h"
#include "ops/common/device_limits.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_shape.h"

#include <cstdint>

namespace ninfer::ops::detail {

// Whether the current device's blocks hold the standard INT8-family prompt kernel's tile.
[[nodiscard]] inline bool causal_prompt_i8_standard_here() {
    return causal_prompt_i8_standard(turing_shapes_here());
}

// What a launch may split: `maximum` 1 turns splits off; `forced` (with a maximum above 1) takes
// exactly that many splits where the width and the budget allow, whatever the cost model says.
struct CausalPromptI8FastSplitPolicy {
    std::int32_t maximum = kCausalPromptI8FastMaxSplits;
    std::int32_t forced  = 0;
};

// Whether a prompt call on this storage takes the fast kernel (register-resident rows, FP16 PV per
// 64-key tile for the INT8 family and the row-scaled FP8 cache, fp8/prompt_fast.cuh): when the
// caller asks for it (--fast-prefill-kernel), else as NINFER_PROMPT_FAST=0|1 or the device
// profile's "attn_prompt_fast" says. The FP8 cache stops short of the profile step: that key was
// measured on the INT8-family kernel only and the fast kernel changes the output, so a default
// --kv-dtype fp8 run keeps the standard kernel, byte-identical to master. Of the other storages
// only NVFP4-G16 has one (nvfp4/prompt_fast.cuh, Blackwell builds, over more than 2048 visible
// keys).
[[nodiscard]] bool causal_attention_prompt_fast_kernel(KvCacheStorage storage, bool requested);

// Whether the fast INT8-family prompt kernel may split a launch's keys over several CTAs
// (int8/prompt_fast_plan.h): unless the device profile's "attn_prompt_split" says "off".
// NINFER_PROMPT_SPLITS=0 turns splits off, and N > 0 forces N splits wherever a launch may split.
[[nodiscard]] CausalPromptI8FastSplitPolicy causal_attention_prompt_fast_split_policy();

// Whether the fast INT8-family prompt kernel's eight-warp CTA launches for this storage on the
// current device's shape set (on Turing, INT8 values leave room for four warps only).
[[nodiscard]] bool causal_prompt_i8_fast_wide(KvCacheStorage storage);

} // namespace ninfer::ops::detail
