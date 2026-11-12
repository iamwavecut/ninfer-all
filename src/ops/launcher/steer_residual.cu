// Implements: include/ninfer/ops/steer_residual.h
// One dispatch: a block of kSteerResidualThreads per residual column, chunked when the width
// and alignment allow it.
#include "ops/launcher/steer_residual.h"

#include "ops/kernel/steer_residual.cuh"
#include "core/device_launch.h" // CUDA_CHECK

#include <cstdint>

namespace ninfer::ops::detail {

namespace {

bool aligned16(const void* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0;
}

} // namespace

void steer_residual_launch(SteerResidualStep step, Tensor& x, std::int32_t directions_count,
                           const Tensor& directions, const Tensor& factors,
                           const SteerResidualLanes& lanes, Tensor& measures, const Tensor& vector,
                           cudaStream_t stream) {
    const bool directed       = directions_count > 0;
    const std::int32_t hidden = x.ne[0];
    const auto* direction_data =
        directed ? static_cast<const float*>(directions.data) : nullptr;
    const auto* vector_data = static_cast<const float*>(vector.data);
    const bool chunked =
        hidden % 8 == 0 && hidden <= kSteerResidualThreads * kSteerResidualChunks * 8 &&
        aligned16(x.data) && (direction_data == nullptr || aligned16(direction_data)) &&
        (vector_data == nullptr || aligned16(vector_data));
    auto* kernel = chunked
                       ? &steer_residual_chunked_kernel<kSteerResidualThreads, kSteerResidualChunks>
                       : &steer_residual_kernel<kSteerResidualThreads>;
    kernel<<<x.ne[1], kSteerResidualThreads, 0, stream>>>(
        static_cast<int>(step), static_cast<__nv_bfloat16*>(x.data), hidden, directions_count,
        direction_data,
        directed && step != SteerResidualStep::Measure ? static_cast<const float*>(factors.data)
                                                       : nullptr,
        directed ? static_cast<const float*>(lanes.lane_scales.data) : nullptr,
        directed ? lanes.lane_scales.ne[1] : 0,
        directed ? static_cast<const std::int32_t*>(lanes.rows.data) : nullptr, lanes.row_width,
        directed ? static_cast<float*>(measures.data) : nullptr, vector_data);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
