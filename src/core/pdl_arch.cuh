#pragma once

// Programmatic dependent launch in a build of several architectures: the device side compiles the
// grid dependency instructions into the sm_90+ images only (elsewhere nothing), and launch_arch
// adds the programmatic attribute only to a launch on an sm_90+ device inside a CUDA Graph capture.
// core/pdl.cuh holds the single-architecture form the native routes use.

#include "pdl_scope.h"

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <utility>

namespace ninfer::pdl {

__device__ __forceinline__ unsigned arch_sm_count() {
    unsigned count;
    asm volatile("mov.u32 %0, %%nsmid;" : "=r"(count));
    return count;
}

// The forms below work in a build of several architectures: the device side compiles the grid
// dependency instructions into the sm_90+ images only (elsewhere nothing), and launch_arch adds the
// programmatic attribute only to a launch on an sm_90+ device inside a CUDA Graph capture. A kernel
// launched by launch_arch calls arch_enter() at the top of every control path, before it touches
// anything; without the attribute arch_enter() returns at once.

__device__ __forceinline__ void arch_enter() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// arch_enter_glue(): arch_enter() for a short kernel. A grid covering at most half the SMs first
// lets the next captured kernel begin launching once every CTA of this grid is resident. Not for a
// kernel that streams weights: the release instruction in its code slows it even where the branch
// is not taken (an expert product by ~15% on an RTX 5090).
__device__ __forceinline__ void arch_enter_glue() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (2U * gridDim.x * gridDim.y * gridDim.z <= arch_sm_count()) {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// Whether launches on the current device may become programmatic dependents: an sm_90+ device.
[[nodiscard]] inline bool arch_device() {
    // Per device: 0 unknown, 1 before sm_90, 2 sm_90 or newer.
    static std::array<std::atomic<int>, 64> tier{};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= int(tier.size())) {
        (void)cudaGetLastError();
        return false;
    }
    int t = tier[std::size_t(device)].load(std::memory_order_relaxed);
    if (t == 0) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) !=
            cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
        t = major >= 9 ? 2 : 1;
        tier[std::size_t(device)].store(t, std::memory_order_relaxed);
    }
    return t == 2;
}

// Whether a launch on `stream` becomes a programmatic dependent (see above): inside an ArchScope
// (core/pdl_scope.h), on an sm_90+ device, while the stream captures a CUDA Graph.
[[nodiscard]] inline bool arch_dependent(cudaStream_t stream) {
    if (arch_scope_depth == 0 || !arch_device()) { return false; }
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return status == cudaStreamCaptureStatusActive;
}

template <class... KernelArgs, class... CallArgs>
inline void launch_arch(dim3 grid, dim3 block, std::size_t smem, cudaStream_t stream,
                        void (*kernel)(KernelArgs...), CallArgs&&... args) {
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim          = grid;
    config.blockDim         = block;
    config.dynamicSmemBytes = smem;
    config.stream           = stream;
    if (arch_dependent(stream)) {
        config.attrs    = &attribute;
        config.numAttrs = 1;
    }
    (void)cudaLaunchKernelEx(&config, kernel, std::forward<CallArgs>(args)...);
}

} // namespace ninfer::pdl
