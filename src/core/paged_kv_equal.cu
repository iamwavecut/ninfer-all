// Byte equality of two page groups' planes (DeviceKVPagePool::pages_equal): one kernel per
// plane counts mismatching bytes atomically; the host reads the verdict after the stream drains.

#include "core/device_launch.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer {
namespace {

__global__ void pages_equal_kernel(const std::uint8_t* a, const std::uint8_t* b,
                                   std::size_t bytes, std::uint32_t* mismatches) {
    const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
    for (std::size_t at = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; at < bytes;
         at += stride) {
        if (a[at] != b[at]) { atomicAdd(mismatches, 1u); }
    }
}

} // namespace

void launch_pages_equal(const void* a, const void* b, std::size_t bytes, void* mismatches,
                        cudaStream_t stream) {
    constexpr int kBlock = 256;
    const auto blocks    = (bytes + kBlock - 1) / kBlock;
    const int grid       = static_cast<int>(blocks < 4096 ? blocks : 4096);
    pages_equal_kernel<<<grid, kBlock, 0, stream>>>(static_cast<const std::uint8_t*>(a),
                                                    static_cast<const std::uint8_t*>(b), bytes,
                                                    static_cast<std::uint32_t*>(mismatches));
}

} // namespace ninfer
