#pragma once

// ninfer::ops::detail - the prompt kernels' shape facts that both the kernel units and the host
// launcher read: the route's row block, the standard INT8-family kernel's shared memory and the
// fast kernel's split limit. The launcher's choices are in prompt_policy.h.

#include "ops/common/device_limits.h"

#include <cstdint>

namespace ninfer::ops::detail {

// The widest prompt-route row block (the eight-warp fast INT8 kernel); every prompt kernel's row
// block divides it.
inline constexpr std::int32_t kPromptWaveRows = 128;

// Shared memory of the standard INT8-family prompt kernel's tile (int8/prompt.cuh), and whether the
// blocks of a shape set hold it (ops/common/device_tier.h). Turing's 64 KiB do not: there every
// INT8-family prompt runs on the fast kernel, and the "attn_pack_gqa" and "attn_prompt_fast"
// switches have nothing to choose.
inline constexpr int kCausalPromptI8StandardSmemBytes = 93184;

[[nodiscard]] constexpr bool causal_prompt_i8_standard(bool turing_shapes) {
    return kCausalPromptI8StandardSmemBytes <= shape_max_shared_bytes_per_block(turing_shapes);
}

// The most CTAs the fast INT8-family prompt kernel splits one launch's keys over
// (int8/prompt_fast_plan.h).
inline constexpr std::int32_t kCausalPromptI8FastMaxSplits = 8;

} // namespace ninfer::ops::detail
