#pragma once

// Implements: include/ninfer/ops/steer_residual.h
// One block per residual column: the column's dot products with the directions are block sums,
// then one pass removes the directions, adds the vector, stores BF16 and sums the new dots. A
// width that is a multiple of 8 and fits the block's registers takes the chunked kernel, which
// reads and writes the column once in 16-byte chunks; a decode column is latency-bound, and the
// strided element loop took about three times as long there.

#include "ninfer/ops/steer_residual.h"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr int kSteerResidualThreads = 256;

// Sums values[e] over the block for e < count into totals[e], visible to every thread.
template <int Threads>
__device__ __forceinline__ void steer_residual_block_sums(
    float (&values)[kSteerResidualMaximumDirections], int count,
    float (*partial)[kSteerResidualMaximumDirections], float* totals) {
    constexpr int Warps = Threads / kWarpSize;
    const int lane      = threadIdx.x & (kWarpSize - 1);
    const int warp      = threadIdx.x / kWarpSize;
#pragma unroll
    for (int e = 0; e < kSteerResidualMaximumDirections; ++e) {
        if (e < count) {
            const float sum = warp_sum(values[e]);
            if (lane == 0) { partial[warp][e] = sum; }
        }
    }
    __syncthreads();
    if (static_cast<int>(threadIdx.x) < count) {
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < Warps; ++w) { total += partial[w][threadIdx.x]; }
        totals[threadIdx.x] = total;
    }
    __syncthreads();
}

// The coefficients a column's lane gives its directions, and whether any is nonzero.
__device__ __forceinline__ bool steer_residual_scales(
    float (&scale)[kSteerResidualMaximumDirections], int count, const float* lane_scales,
    int lanes, const std::int32_t* rows, int row_width, int column) {
    const int lane = count > 0 ? rows[column / row_width] : -1;
    bool active    = false;
#pragma unroll
    for (int e = 0; e < kSteerResidualMaximumDirections; ++e) {
        scale[e] = e < count && lane >= 0 && lane < lanes ? lane_scales[lane * count + e] : 0.0f;
        active   = active || scale[e] != 0.0f;
    }
    return active;
}

template <int Threads>
__launch_bounds__(Threads) __global__ void steer_residual_kernel(
    int step, __nv_bfloat16* __restrict__ x, int hidden, int count,
    const float* __restrict__ directions, const float* __restrict__ factors,
    const float* __restrict__ lane_scales, int lanes, const std::int32_t* __restrict__ rows,
    int row_width, float* __restrict__ measures, const float* __restrict__ vector) {
    __shared__ float partial[Threads / kWarpSize][kSteerResidualMaximumDirections];
    __shared__ float totals[kSteerResidualMaximumDirections];
    const int column = blockIdx.x;
    float scale[kSteerResidualMaximumDirections];
    const bool active =
        steer_residual_scales(scale, count, lane_scales, lanes, rows, row_width, column);
    if (!active && vector == nullptr) { return; }

    __nv_bfloat16* values = x + static_cast<std::size_t>(column) * hidden;
    float* measured       = measures + static_cast<std::size_t>(column) * count;
    float coefficient[kSteerResidualMaximumDirections] = {};
    if (active) {
        float dot[kSteerResidualMaximumDirections] = {};
        for (int i = threadIdx.x; i < hidden; i += Threads) {
            const float value = __bfloat162float(values[i]);
#pragma unroll
            for (int e = 0; e < kSteerResidualMaximumDirections; ++e) {
                if (e < count) { dot[e] += directions[static_cast<std::size_t>(e) * hidden + i] * value; }
            }
        }
        steer_residual_block_sums<Threads>(dot, count, partial, totals);
        if (step == static_cast<int>(SteerResidualStep::Measure)) {
            if (static_cast<int>(threadIdx.x) < count) { measured[threadIdx.x] = totals[threadIdx.x]; }
            return;
        }
#pragma unroll
        for (int e = 0; e < kSteerResidualMaximumDirections; ++e) {
            if (e < count) {
                const float written = step == static_cast<int>(SteerResidualStep::Write)
                                          ? totals[e] - measured[e]
                                          : totals[e];
                coefficient[e] = scale[e] * factors[e] * written;
            }
        }
        // Every thread has read totals before the second sum reuses it.
        __syncthreads();
    }

    const bool measure = active && step == static_cast<int>(SteerResidualStep::Write);
    float stored_dot[kSteerResidualMaximumDirections] = {};
    for (int i = threadIdx.x; i < hidden; i += Threads) {
        float value = __bfloat162float(values[i]);
        if (active) {
#pragma unroll
            for (int e = 0; e < kSteerResidualMaximumDirections; ++e) {
                if (e < count) {
                    value -= coefficient[e] * directions[static_cast<std::size_t>(e) * hidden + i];
                }
            }
        }
        if (vector != nullptr) { value += vector[i]; }
        const __nv_bfloat16 stored = __float2bfloat16_rn(value);
        values[i]                  = stored;
        if (measure) {
            const float rounded = __bfloat162float(stored);
#pragma unroll
            for (int e = 0; e < kSteerResidualMaximumDirections; ++e) {
                if (e < count) {
                    stored_dot[e] += directions[static_cast<std::size_t>(e) * hidden + i] * rounded;
                }
            }
        }
    }
    if (measure) {
        steer_residual_block_sums<Threads>(stored_dot, count, partial, totals);
        if (static_cast<int>(threadIdx.x) < count) { measured[threadIdx.x] = totals[threadIdx.x]; }
    }
}

// Eight consecutive FP32 values from a 32-byte-aligned address.
__device__ __forceinline__ void steer_residual_load8(const float* source, float (&out)[8]) {
    const float4 low  = reinterpret_cast<const float4*>(source)[0];
    const float4 high = reinterpret_cast<const float4*>(source)[1];
    out[0] = low.x;
    out[1] = low.y;
    out[2] = low.z;
    out[3] = low.w;
    out[4] = high.x;
    out[5] = high.y;
    out[6] = high.z;
    out[7] = high.w;
}

inline constexpr int kSteerResidualChunks = 4;

// hidden % 8 == 0 and hidden <= Threads * Chunks * 8; x, directions and vector 16-byte aligned.
// Thread t holds chunks t, t + Threads, ... of the column in registers.
template <int Threads, int Chunks>
__launch_bounds__(Threads) __global__ void steer_residual_chunked_kernel(
    int step, __nv_bfloat16* __restrict__ x, int hidden, int count,
    const float* __restrict__ directions, const float* __restrict__ factors,
    const float* __restrict__ lane_scales, int lanes, const std::int32_t* __restrict__ rows,
    int row_width, float* __restrict__ measures, const float* __restrict__ vector) {
    __shared__ float partial[Threads / kWarpSize][kSteerResidualMaximumDirections];
    __shared__ float totals[kSteerResidualMaximumDirections];
    const int column = blockIdx.x;
    float scale[kSteerResidualMaximumDirections];
    const bool active =
        steer_residual_scales(scale, count, lane_scales, lanes, rows, row_width, column);
    if (!active && vector == nullptr) { return; }

    const int chunks = hidden / 8;
    uint4* stored    = reinterpret_cast<uint4*>(x + static_cast<std::size_t>(column) * hidden);
    float* measured  = measures + static_cast<std::size_t>(column) * count;
    float value[Chunks][8];
#pragma unroll
    for (int k = 0; k < Chunks; ++k) {
        const int chunk = static_cast<int>(threadIdx.x) + k * Threads;
        if (chunk < chunks) {
            const uint4 raw              = stored[chunk];
            const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float2 pair   = __bfloat1622float2(pairs[p]);
                value[k][2 * p]     = pair.x;
                value[k][2 * p + 1] = pair.y;
            }
        }
    }

    float coefficient[kSteerResidualMaximumDirections] = {};
    if (active) {
        float dot[kSteerResidualMaximumDirections] = {};
#pragma unroll
        for (int k = 0; k < Chunks; ++k) {
            const int chunk = static_cast<int>(threadIdx.x) + k * Threads;
            if (chunk >= chunks) { continue; }
#pragma unroll
            for (int e = 0; e < kSteerResidualMaximumDirections; ++e) {
                if (e >= count) { continue; }
                float direction[8];
                steer_residual_load8(directions + static_cast<std::size_t>(e) * hidden + chunk * 8,
                                     direction);
#pragma unroll
                for (int j = 0; j < 8; ++j) { dot[e] += direction[j] * value[k][j]; }
            }
        }
        steer_residual_block_sums<Threads>(dot, count, partial, totals);
        if (step == static_cast<int>(SteerResidualStep::Measure)) {
            if (static_cast<int>(threadIdx.x) < count) { measured[threadIdx.x] = totals[threadIdx.x]; }
            return;
        }
#pragma unroll
        for (int e = 0; e < kSteerResidualMaximumDirections; ++e) {
            if (e < count) {
                const float written = step == static_cast<int>(SteerResidualStep::Write)
                                          ? totals[e] - measured[e]
                                          : totals[e];
                coefficient[e] = scale[e] * factors[e] * written;
            }
        }
        // Every thread has read totals before the second sum reuses it.
        __syncthreads();
    }

    const bool measure = active && step == static_cast<int>(SteerResidualStep::Write);
    float stored_dot[kSteerResidualMaximumDirections] = {};
#pragma unroll
    for (int k = 0; k < Chunks; ++k) {
        const int chunk = static_cast<int>(threadIdx.x) + k * Threads;
        if (chunk >= chunks) { continue; }
        float* updated = value[k];
        if (active) {
#pragma unroll
            for (int e = 0; e < kSteerResidualMaximumDirections; ++e) {
                if (e >= count) { continue; }
                float direction[8];
                steer_residual_load8(directions + static_cast<std::size_t>(e) * hidden + chunk * 8,
                                     direction);
#pragma unroll
                for (int j = 0; j < 8; ++j) { updated[j] -= coefficient[e] * direction[j]; }
            }
        }
        if (vector != nullptr) {
            float added[8];
            steer_residual_load8(vector + chunk * 8, added);
#pragma unroll
            for (int j = 0; j < 8; ++j) { updated[j] += added[j]; }
        }
        uint4 raw;
        __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            pairs[p] = __floats2bfloat162_rn(updated[2 * p], updated[2 * p + 1]);
        }
        stored[chunk] = raw;
        if (measure) {
#pragma unroll
            for (int e = 0; e < kSteerResidualMaximumDirections; ++e) {
                if (e >= count) { continue; }
                float direction[8];
                steer_residual_load8(directions + static_cast<std::size_t>(e) * hidden + chunk * 8,
                                     direction);
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const float2 rounded = __bfloat1622float2(pairs[p]);
                    stored_dot[e] += direction[2 * p] * rounded.x + direction[2 * p + 1] * rounded.y;
                }
            }
        }
    }
    if (measure) {
        steer_residual_block_sums<Threads>(stored_dot, count, partial, totals);
        if (static_cast<int>(threadIdx.x) < count) { measured[threadIdx.x] = totals[threadIdx.x]; }
    }
}

} // namespace ninfer::ops
