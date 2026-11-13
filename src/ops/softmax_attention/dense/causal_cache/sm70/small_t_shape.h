#pragma once

// The block shape of the Volta small-T kernels (small_t_sm70_*.cuh) as compile-time numbers that host
// code reads too: warps per row group and per CTA, the staged row strides, and the CTAs of the INT8
// kernel an SM holds, from which the host sizes its whole-wave split counts.

#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"

namespace ninfer::ops::sm70_attention {

template <int Rows>
struct Shape {
    static constexpr int kRowGroups = (Rows + 7) / 8;
    // Warps splitting one row group's key tiles; more row groups share the keys' loads instead.
    static constexpr int kKeyWarps = kRowGroups == 1 ? 4 : (kRowGroups == 2 ? 2 : 1);
    static constexpr int kWarps    = kRowGroups * kKeyWarps;
};

inline constexpr int kQStride = kCausalHeadDim + 8; // halves; pads rows apart by 16 bytes
inline constexpr int kPStride = 32 + 8;             // halves

// The shared memory of causal_attention_small_t_sm70_i8_kernel for TokenTile tokens (its appending
// form, the larger), and the CTAs of it an SM holds: the kernel is bounded to that many, which keeps
// it within 128 registers a thread at four warps, and sm_70 has 96 KiB of shared memory an SM. The
// host sizes its whole-wave split counts from the same number (i8_partial_ctas_per_sm).
template <int TokenTile, int Rows>
constexpr int i8_shared_bytes() {
    using S = Shape<Rows>;
    return S::kRowGroups * 8 * kQStride * 2 + S::kWarps * 8 * kPStride * 2 + 64 * 4 +
           2 * S::kWarps * 8 * 4 + (S::kKeyWarps > 1 ? S::kRowGroups * 8 * kCausalHeadDim * 4 : 4) +
           TokenTile * kCausalHeadDim * 4;
}

template <int TokenTile, int Rows>
constexpr int i8_resident_ctas() {
    const int by_registers = 16 / Shape<Rows>::kWarps;
    const int by_shared    = 96 * 1024 / i8_shared_bytes<TokenTile, Rows>();
    const int ctas         = by_registers < by_shared ? by_registers : by_shared;
    return ctas > 0 ? ctas : 1;
}

} // namespace ninfer::ops::sm70_attention
