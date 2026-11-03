#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstdint>

namespace ninfer::ops {

// The most directions one call removes.
inline constexpr std::int32_t kSteerResidualMaximumDirections = 8;

enum class SteerResidualStep : std::uint8_t {
    // m_e(t) = r_e . x(t); x unchanged.
    Measure,
    // x(t) -= sum_e k_e(t) (r_e . x(t)) r_e; m unchanged.
    Embedding,
    // d_e(t) = r_e . x(t) - m_e(t); x(t) <- bf16(x(t) - sum_e k_e(t) d_e(t) r_e + v);
    // m_e(t) <- r_e . x(t), read from the stored BF16 x.
    Write,
};

/**
 * Residual steering: removes directions from what was written to a BF16 residual since the last
 * call, and adds a vector.
 *
 * `x` is BF16 [H, T], contiguous. `directions` is FP32 [H, E] (direction e is column e; unit length
 * is the caller's convention, not a requirement), 0 <= E <= kSteerResidualMaximumDirections; an
 * empty `directions` has E = 0. Column t belongs to lane L(t) = rows[t / row_width], `rows` I32
 * [ceil(T / row_width)]. `lane_scales` is FP32 [E, lanes]; `factors` FP32 [E] (Embedding, Write).
 * The coefficient is k_e(t) = lane_scales[e, L(t)] * factors[e], where a lane outside
 * [0, lanes) has every scale zero. A column is active when some lane_scales[e, L(t)] is nonzero.
 * `measures` is FP32 [E, >= T]: m_e(t) at element (e, t).
 *
 * Measure and Embedding touch only active columns. Write adds `v` (FP32 [H], or empty for none) to
 * every column and, in active columns, removes and measures as stated above; m of an inactive
 * column is left as it was. Measure, then Write after each write to x, keep m_e(t) equal to
 * r_e . x(t), so d_e(t) is the component of that write along r_e as x stores it.
 *
 * The oracle evaluates every dot product, d and the updated x in FP64 from the represented inputs
 * (the BF16 x, the FP32 directions, scales, factors, measures and v). The updated x is promoted
 * and compared with that value, whose BF16 rounding belongs to the Op's numerical criterion; the
 * new m is compared with r_e . x evaluated in FP64 from the stored BF16 output. x and measures are
 * updated in place; no other input is modified, and none may alias x or measures. The Op uses no
 * workspace or other state.
 */
struct SteerResidualLanes {
    Tensor lane_scales;
    Tensor rows;
    std::int32_t row_width = 1;
};

void steer_residual(SteerResidualStep step, Tensor& x, const Tensor& directions,
                    const Tensor& factors, const SteerResidualLanes& lanes, Tensor& measures,
                    const Tensor& vector, cudaStream_t stream);

} // namespace ninfer::ops
