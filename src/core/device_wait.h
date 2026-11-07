#pragma once

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>

namespace ninfer {

struct DeviceContext;
struct DeviceWaitState;

// Device waits that a DeviceWaitWatch sees. CUDA reports a device that stops finishing work as no
// error: the thread that waits for it stays inside the driver for good, while every other thread of
// the process keeps answering (on 2026-11-07 an Engine worker waited 46 minutes in
// cudaEventSynchronize with the GPU idle and /health answering ok). Code that waits for the device
// calls these in place of cudaStreamSynchronize, cudaEventSynchronize, cudaDeviceSynchronize,
// cudaMemcpy and cudaMemset. Each returns what its CUDA call returns; on a thread no watch is bound
// to, it is that call.
cudaError_t wait_for_stream(cudaStream_t stream) noexcept;
cudaError_t wait_for_event(cudaEvent_t event) noexcept;
cudaError_t wait_for_device() noexcept;
cudaError_t copy_and_wait(void* destination, const void* source, std::size_t bytes,
                          cudaMemcpyKind kind) noexcept;
cudaError_t fill_and_wait(void* destination, int value, std::size_t bytes) noexcept;

// A host service answered a request of CUDA device `device` (CudaHostHandshake: host or disk
// experts, a CPU share of expert products). The device may wait for such answers inside a watched
// wait, so the wait is not a stall while they continue.
void note_host_service_progress(int device) noexcept;

// While alive, watches the device waits of the thread that constructed it. A wait that lasts
// `limit` while no host service of the context's devices answers ends the process: the device
// stopped finishing work, the waiting thread never returns, and only a new process recovers the
// device, so the stall ends the process the way CUDA_CHECK ends it on a CUDA error, with a message
// on stderr that names `thread`, the devices and the kind of wait. Watches nest on one thread; the
// innermost one governs until it ends. One thread of the process checks every watch.
class DeviceWaitWatch {
public:
    // No device wait of an Engine lasts this long without a host service answering: the longest
    // device work between two host waits is one prompt step, seconds on the slowest card.
    static constexpr std::chrono::seconds kStallLimit{120};

    DeviceWaitWatch(const DeviceContext& context, const char* thread,
                    std::chrono::milliseconds limit = kStallLimit);
    ~DeviceWaitWatch();
    DeviceWaitWatch(const DeviceWaitWatch&)            = delete;
    DeviceWaitWatch& operator=(const DeviceWaitWatch&) = delete;

private:
    DeviceWaitState* state_ = nullptr;
    DeviceWaitState* outer_ = nullptr;
};

} // namespace ninfer
