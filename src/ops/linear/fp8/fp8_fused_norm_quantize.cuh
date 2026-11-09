#pragma once

// ninfer::ops - FP8 A8 fused producer kernels: one pass over a residual row computes the
// normalized activation's per-token E4M3 codes and scale, replacing the caller's separate norm
// and quantize launches. The normalized BF16 row is never written; the codes and scales are
// bit-for-bit the two-step composition's (norm to BF16 storage, then the A8 quantiser on it).

#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/rmsnorm.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::detail {

// One CTA owns one residual row of D elements (D fixed). Pass one reads the row, its norm
// weight and, for the gated epilogue, its gate row into registers and accumulates the sum of
// squares in the rmsnorm CTA kernel's pair order; pass two rounds the normalized values to BF16
// once, exactly as that kernel's epilogue does, and folds the per-token maximum over those
// rounded values; pass three converts them to E4M3 with the row's scale, the same formula
// fp8_a8_quantize_kernel applies to the stored row. The sum, the reduction, the single BF16
// rounding and the quantiser's formula each reproduce the standalone kernel, so the output is
// the two-step composition bit for bit.
template <RmsEpilogue Epilogue, int Threads, int D>
__global__ __launch_bounds__(Threads)
    void fp8_fused_norm_quantize_kernel(const __nv_bfloat162* __restrict__ x,
                                        const __nv_bfloat162* __restrict__ weight,
                                        const __nv_bfloat162* __restrict__ z,
                                        std::uint8_t* __restrict__ codes,
                                        float* __restrict__ scales, float eps) {
    static_assert((D % (Threads * 2)) == 0);
    constexpr int kPairs        = D / 2;
    constexpr int kPairsPerThread = kPairs / Threads;
    constexpr int kWarps        = Threads / kWarpSize;
    __shared__ float warp_sums[kWarps];
    __shared__ float warp_maxima[kWarps];
    __shared__ float inv_shared;
    __shared__ float scale_shared;

    pdl::enter();
    const int tid          = static_cast<int>(threadIdx.x);
    const int lane         = tid & (kWarpSize - 1);
    const int warp         = tid >> 5;
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    const std::int64_t row_base = row * static_cast<std::int64_t>(kPairs);

    __nv_bfloat162 values[kPairsPerThread];
    __nv_bfloat162 weights[kPairsPerThread];
    __nv_bfloat162 gates[kPairsPerThread];
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kPairsPerThread; ++k) {
        const int pair = tid + k * Threads;
        values[k]      = x[row_base + pair];
        weights[k]     = weight[pair];
        if constexpr (rms_has_gate(Epilogue)) { gates[k] = z[row_base + pair]; }
        const float2 xf = __bfloat1622float2(values[k]);
        sum += xf.x * xf.x + xf.y * xf.y;
    }

    const float block_sum = block_reduce_sum<Threads>(sum, warp_sums);
    if (tid == 0) { inv_shared = rsqrtf(block_sum / static_cast<float>(D) + eps); }
    __syncthreads();
    const float inv = inv_shared;

    float maximum = 0.0f;
#pragma unroll
    for (int k = 0; k < kPairsPerThread; ++k) {
        const float2 xf = __bfloat1622float2(values[k]);
        const float2 wf = __bfloat1622float2(weights[k]);
        float2 zf{0.0f, 0.0f};
        if constexpr (rms_has_gate(Epilogue)) { zf = __bfloat1622float2(gates[k]); }
        const __nv_bfloat162 stored = __floats2bfloat162_rn(
            rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, zf.x),
            rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, zf.y));
        const float2 rounded = __bfloat1622float2(stored);
        maximum              = fmaxf(maximum, fabsf(rounded.x));
        maximum              = fmaxf(maximum, fabsf(rounded.y));
    }

    maximum = warp_max(maximum);
    if (lane == 0) { warp_maxima[warp] = maximum; }
    __syncthreads();
    if (warp == 0) {
        maximum = lane < kWarps ? warp_maxima[lane] : 0.0f;
        maximum = warp_max(maximum);
        if (lane == 0) { scale_shared = maximum > 0.0f ? maximum / 448.0f : 0.0f; }
    }
    __syncthreads();
    const float scale   = scale_shared;
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;

    auto* output_pairs = reinterpret_cast<std::uint16_t*>(
        codes + static_cast<std::int64_t>(row) * D);
#pragma unroll
    for (int k = 0; k < kPairsPerThread; ++k) {
        const int pair = tid + k * Threads;
        const float2 xf = __bfloat1622float2(values[k]);
        const float2 wf = __bfloat1622float2(weights[k]);
        float2 zf{0.0f, 0.0f};
        if constexpr (rms_has_gate(Epilogue)) { zf = __bfloat1622float2(gates[k]); }
        const __nv_bfloat162 stored = __floats2bfloat162_rn(
            rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, zf.x),
            rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, zf.y));
        const float2 rounded = __bfloat1622float2(stored);
        const float2 scaled  = make_float2(rounded.x * inverse, rounded.y * inverse);
        output_pairs[pair]   = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
    }
    if (tid == 0) { scales[row] = scale; }
}

// Per-head SiLU-gated variant for the gated delta net's output norm: a 6144 row holds 48 heads
// of 128, and the 128 gain normalizes its own head only. The fused CTA (768 threads, 24 warps,
// two heads per warp) must keep the rmsnorm warp kernel's per-row geometry for the bit-for-bit
// match: within a head each lane owns pairs lane + k*32 (guarded slots 0..3), so each lane's
// partial sum of squares is the value the standalone one-warp-per-head kernel would hold, and the
// per-head warp reductions reuse its butterfly tree. The per-token E4M3 scale still spans the
// whole row: maximum is order-independent, so the row maximum may reduce across the 48 head
// maxima in any order.
template <int Threads>
__global__ __launch_bounds__(Threads)
    void fp8_fused_gated_head_rmsnorm_quantize_kernel(const __nv_bfloat162* __restrict__ x,
                                                      const __nv_bfloat162* __restrict__ weight,
                                                      const __nv_bfloat162* __restrict__ z,
                                                      std::uint8_t* __restrict__ codes,
                                                      float* __restrict__ scales, float eps) {
    constexpr int kRowPairs  = 6144 / 2;
    constexpr int kHeadPairs = 128 / 2; // 64
    constexpr int kHeads     = kRowPairs / kHeadPairs; // 48
    constexpr int kWarps     = Threads / kWarpSize;
    constexpr int kSlots     = 4; // the rmsnorm warp kernel's kMaxPairsPerLane
    static_assert(kHeads % 2 == 0, "heads pair two per warp");
    static_assert(kHeads / 2 == kWarps, "two heads per warp");
    static_assert(kHeadPairs % (2 * kWarpSize) == 0, "head pairs span two warp passes");
    __shared__ float head_maxima[kHeads];
    __shared__ float scale_shared;

    pdl::enter();
    const int lane             = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp             = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t row        = static_cast<std::int64_t>(blockIdx.x);
    const std::int64_t row_base   = row * static_cast<std::int64_t>(kRowPairs);
    const std::int64_t headA_base = row_base + static_cast<std::int64_t>(2 * warp) * kHeadPairs;
    const std::int64_t headB_base = headA_base + kHeadPairs;

    __nv_bfloat162 valuesA[kSlots];
    __nv_bfloat162 gatesA[kSlots];
    __nv_bfloat162 valuesB[kSlots];
    __nv_bfloat162 gatesB[kSlots];
    float sumA = 0.0f;
    float sumB = 0.0f;
    #pragma unroll
    for (int k = 0; k < kSlots; ++k) {
        const int pair = lane + k * kWarpSize;
        if (pair < kHeadPairs) {
            valuesA[k] = x[headA_base + pair];
            gatesA[k]  = z[headA_base + pair];
            const float2 xfa = __bfloat1622float2(valuesA[k]);
            sumA += xfa.x * xfa.x + xfa.y * xfa.y;

            valuesB[k] = x[headB_base + pair];
            gatesB[k]  = z[headB_base + pair];
            const float2 xfb = __bfloat1622float2(valuesB[k]);
            sumB += xfb.x * xfb.x + xfb.y * xfb.y;
        }
    }

    const float headA_sum = warp_reduce_sum(sumA);
    const float headB_sum = warp_reduce_sum(sumB);
    float invA           = lane == 0 ? rsqrtf(headA_sum / 128.0f + eps) : 0.0f;
    float invB           = lane == 0 ? rsqrtf(headB_sum / 128.0f + eps) : 0.0f;
    invA                 = __shfl_sync(kFullWarpMask, invA, 0);
    invB                 = __shfl_sync(kFullWarpMask, invB, 0);

    float maxA = 0.0f;
    float maxB = 0.0f;
    #pragma unroll
    for (int k = 0; k < kSlots; ++k) {
        const int pair = lane + k * kWarpSize;
        if (pair < kHeadPairs) {
            const float2 xfa = __bfloat1622float2(valuesA[k]);
            const float2 wfa = __bfloat1622float2(weight[pair]);
            const float2 zfa = __bfloat1622float2(gatesA[k]);
            const __nv_bfloat162 sa = __floats2bfloat162_rn(
                rmsnorm_epilogue<RmsEpilogue::Gated>(xfa.x, invA, wfa.x, zfa.x),
                rmsnorm_epilogue<RmsEpilogue::Gated>(xfa.y, invA, wfa.y, zfa.y));
            const float2 ra = __bfloat1622float2(sa);
            maxA = fmaxf(maxA, fmaxf(fabsf(ra.x), fabsf(ra.y)));

            const float2 xfb = __bfloat1622float2(valuesB[k]);
            const float2 wfb = __bfloat1622float2(weight[pair]);
            const float2 zfb = __bfloat1622float2(gatesB[k]);
            const __nv_bfloat162 sb = __floats2bfloat162_rn(
                rmsnorm_epilogue<RmsEpilogue::Gated>(xfb.x, invB, wfb.x, zfb.x),
                rmsnorm_epilogue<RmsEpilogue::Gated>(xfb.y, invB, wfb.y, zfb.y));
            const float2 rb = __bfloat1622float2(sb);
            maxB = fmaxf(maxB, fmaxf(fabsf(rb.x), fabsf(rb.y)));
        }
    }
    // warp_max is a full-warp collective: every lane must execute it; only lane 0 stores.
    const float headA_max = warp_max(maxA);
    const float headB_max = warp_max(maxB);
    if (lane == 0) {
        head_maxima[2 * warp]     = headA_max;
        head_maxima[2 * warp + 1] = headB_max;
    }
    __syncthreads();
    if (warp == 0) {
        float m = head_maxima[lane];
        if (lane < kHeads - kWarpSize) { m = fmaxf(m, head_maxima[kWarpSize + lane]); }
        m = warp_max(m);
        if (lane == 0) { scale_shared = m > 0.0f ? m / 448.0f : 0.0f; }
    }
    __syncthreads();
    const float scale   = scale_shared;
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;

    auto* outputA =
        reinterpret_cast<std::uint16_t*>(codes + row * 6144) + static_cast<std::uint64_t>(2 * warp) * kHeadPairs;
    auto* outputB = outputA + kHeadPairs;
    #pragma unroll
    for (int k = 0; k < kSlots; ++k) {
        const int pair = lane + k * kWarpSize;
        if (pair < kHeadPairs) {
            const float2 xfa = __bfloat1622float2(valuesA[k]);
            const float2 wfa = __bfloat1622float2(weight[pair]);
            const float2 zfa = __bfloat1622float2(gatesA[k]);
            const __nv_bfloat162 sa = __floats2bfloat162_rn(
                rmsnorm_epilogue<RmsEpilogue::Gated>(xfa.x, invA, wfa.x, zfa.x),
                rmsnorm_epilogue<RmsEpilogue::Gated>(xfa.y, invA, wfa.y, zfa.y));
            const float2 ra = __bfloat1622float2(sa);
            outputA[pair] = __nv_cvt_float2_to_fp8x2(make_float2(ra.x * inverse, ra.y * inverse),
                                                     __NV_SATFINITE, __NV_E4M3);

            const float2 xfb = __bfloat1622float2(valuesB[k]);
            const float2 wfb = __bfloat1622float2(weight[pair]);
            const float2 zfb = __bfloat1622float2(gatesB[k]);
            const __nv_bfloat162 sb = __floats2bfloat162_rn(
                rmsnorm_epilogue<RmsEpilogue::Gated>(xfb.x, invB, wfb.x, zfb.x),
                rmsnorm_epilogue<RmsEpilogue::Gated>(xfb.y, invB, wfb.y, zfb.y));
            const float2 rb = __bfloat1622float2(sb);
            outputB[pair] = __nv_cvt_float2_to_fp8x2(make_float2(rb.x * inverse, rb.y * inverse),
                                                     __NV_SATFINITE, __NV_E4M3);
        }
    }
    if (threadIdx.x == 0) { scales[row] = scale; }
}

} // namespace ninfer::ops::detail
