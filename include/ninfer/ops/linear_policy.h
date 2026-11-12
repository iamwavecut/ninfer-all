#pragma once

// The activation-compute policies of the linear projections, their width gates and the share units
// of tensor-parallel matrices, which the route code of every quantised format reads; the Op entry
// points are in ninfer/ops/linear.h, which includes this header.

#include <cstdint>

namespace ninfer::ops {

/**
 * @brief Permitted private activation-compute profiles for a linear projection.
 *
 * The policy constrains private route selection; it does not select a kernel or prescribe a
 * particular MMA instruction. The public activation and output tensors remain BF16 for every
 * policy.
 */
enum class LinearPolicy : std::uint8_t {
    A16Only, ///< Admit only A16 compute profiles.
    AllowA8, ///< Admit either A16 or A8 compute profiles.
    AllowA4, ///< Admit A16, A8 or A4 compute profiles.
    /// Admit either A16 or integer-A8 compute profiles. Distinct from AllowA8, which selects the
    /// FP8 activation path against FP8 weights; this one quantises activations to s8 with one
    /// scale per weight group and feeds groupwise-int weights to the integer tensor cores. Held
    /// to the same A8 activation allowance.
    AllowA8Int,
    /// AllowA8Int plus the integer small-T route for the 27B gate_up at decode and verify widths
    /// (ops/linear/q4/q4_small_t_mma_i8.cuh). Separate because that route is a quality trade the
    /// prefill one is not asked to carry: it applies s8 activation quantisation to every decode
    /// step rather than to full prefill tiles only, so it is opt-in per engine.
    AllowA8IntDecode,
    /// AllowA8Int plus the cuBLAS prefill route at wide token counts: the weight is materialised as
    /// int8 with one scale per row and the GEMM handed to cuBLAS, which runs it about twice as fast
    /// as this fork's own integer mainloop can. It is a further quality trade -- per-row weight
    /// scales and per-token activation scales, where the integer route carries per-group ones -- so
    /// it is opt-in per engine, and it falls back to AllowA8Int below the width where it pays.
    AllowPrefillCublas,
    /// AllowA8Int's counterpart for block-scaled NVFP4 weights: the stored E2M1 codes widen
    /// losslessly to s8 and feed the integer tensor cores with s8-quantised activations, one FP32
    /// scale per (token, 32 columns) and the per-16 E4M3 weight scales rescaling in FP32. It is the
    /// same A8-activation quality trade the Q4/Q5 integer routes carry, opt-in per engine with
    /// --nvfp4-w4a8-int. Which mainloop answers follows the route table -- the own table runs MMQ,
    /// the unified table runs the grouped ladder, and T <= 8 runs the MMVQ-shaped row-form kernel
    /// under either -- so a calibrated band chooses it (the nvfp4_int8/* keys).
    AllowNvfp4Int8,
};

[[nodiscard]] constexpr bool valid_linear_policy(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8 ||
           policy == LinearPolicy::AllowA4 || policy == LinearPolicy::AllowA8Int ||
           policy == LinearPolicy::AllowA8IntDecode || policy == LinearPolicy::AllowPrefillCublas ||
           policy == LinearPolicy::AllowNvfp4Int8;
}

[[nodiscard]] constexpr bool allows_a8(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowA8 || policy == LinearPolicy::AllowA4;
}

[[nodiscard]] constexpr bool allows_a4(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowA4;
}

/// Integer-A8 (s8 activation, groupwise-int weight) profiles. Never implies the FP8 A8 path.
[[nodiscard]] constexpr bool allows_a8_int(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowA8Int || policy == LinearPolicy::AllowA8IntDecode ||
           policy == LinearPolicy::AllowPrefillCublas;
}

/// The NVFP4 int8-grouped route: the same activation trade as allows_a8_int, but for block-scaled
/// NVFP4 weights against their stored E2M1/E4M3 planes, and never for any other weight format.
[[nodiscard]] constexpr bool allows_nvfp4_int8(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowNvfp4Int8;
}

/// The cuBLAS prefill route. Only ever admitted past a width gate, because its dequantise pass is
/// weight-sized and has to be amortised over the tokens in the call.
[[nodiscard]] constexpr bool allows_cublas_prefill(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowPrefillCublas;
}

/// The route's width gate where the device profile names none: from here it beats the integer
/// mainloop it replaces on every card measured. The dequantise pass costs the same whatever the
/// token count, so a narrow call pays for it without the GEMM to amortise it; where the narrower
/// gate lies depends on the card and on whether the width is a multiple of the integer route's
/// 128-column tile, so a calibrated profile places it per Op (`w4_cublas/...` keys).
inline constexpr std::int32_t kCublasPrefillMinTokens = 512;

/// No profile moves the gate to this width or below it: a decode or verify round runs up to
/// kMaximumConcurrency lanes of a 16-token window, 8 x 16 = 128 columns, through the same Ops and
/// policies, and those rounds stay off the route.
inline constexpr std::int32_t kCublasPrefillFloorTokens = 128;

// The rows and K columns that a share of a matrix (a tensor-parallel rank's piece) is a whole
// number of in its format, so that a runtime-shape route takes it, beyond the stored layout's units:
// T2 runs 32-row tiles over 512-column K slabs, FP8 64 rows over 512 and NVFP4 128 over 256; every
// other format adds none.
struct LinearShareUnits {
    std::uint64_t rows    = 1;
    std::uint64_t columns = 1;
};

} // namespace ninfer::ops
