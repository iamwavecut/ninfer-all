#pragma once

#include "ops/common/math.cuh"

namespace ninfer::ops {

// Online softmax holds a row's running sums relative to its rounded scaled maximum: a probability
// is exp2(fma(score, scale, -rn(maximum * scale))). The factor that moves the sums from the previous
// maximum to the new one is 2^(rn(previous * scale) - scaled_maximum), the difference of the two
// rounded products (never contracted into an FMA), which is exactly 1 while the maximum holds. An
// FMA against the new product leaves 2^(that product's rounding residual) instead, and the
// residual compounds over a row's key tiles.
__device__ __forceinline__ float softmax_history_rescale(float previous, float scale,
                                                         float scaled_maximum) {
    return exp2_approx(__fsub_rn(__fmul_rn(previous, scale), scaled_maximum));
}

} // namespace ninfer::ops
