#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ninfer {

class PinnedHostBuffer;

// Whether the driver model queues a device's memory copies behind one another whatever their
// streams: WDDM, which is every Windows build and, through /dev/dxg, a Linux build under WSL2.
// There a copy a stream queued behind a kernel that waits for a host worker also holds back that
// worker's own copies on another stream, so the two wait for each other. Work that puts a
// host-to-device copy behind such a wait uses MappedHostBuffer instead where this is true.
[[nodiscard]] bool serialized_copy_queue() noexcept;

// Pinned host memory that every device also maps into its address space, so a kernel reads it in
// place: its copy to device memory is an exact, capturable kernel launch on the compute queue.
class MappedHostBuffer {
public:
    explicit MappedHostBuffer(std::size_t bytes);
    ~MappedHostBuffer();
    MappedHostBuffer(const MappedHostBuffer&)            = delete;
    MappedHostBuffer& operator=(const MappedHostBuffer&) = delete;

    [[nodiscard]] void* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    // Copies the first `bytes` to `destination`, memory of the current device, with a kernel on
    // `stream`. The kernel reads the host bytes when it runs, as a queued cudaMemcpyAsync does.
    void copy_to_device(void* destination, std::size_t bytes, cudaStream_t stream) const;

private:
    void* data_       = nullptr;
    std::size_t size_ = 0;
};

// Whether a host service that a replayed graph's device-side wait depends on copies with
// SegmentCopies instead of the copy engine: NINFER_MISS_COPY_KERNEL=0|1 forces either copy for A/B
// runs and diagnosis, else serialized_copy_queue().
[[nodiscard]] bool copy_kernel_default();

// A non-blocking stream for such a service's copies on the current device. With `kernel` (its
// copies are SegmentCopies launches) it has the device's greatest priority, so the copies' few
// blocks start ahead of the waiting blocks of a prompt pass's kernels instead of behind them; else
// the default priority.
[[nodiscard]] cudaStream_t create_copy_stream(bool kernel);

// Copies queued for one stream that run as kernel launches on it, outside the copy queue: where
// serialized_copy_queue() holds, a copy-engine copy that a replayed graph's device-side wait
// depends on can sit behind the copies the graph queued after that wait, and the wait never ends.
// A launch reads its sources in place when it runs: device memory, or pinned host memory at its
// device address. Each launch is a small fixed grid whose blocks stride over its segments, so it
// leaves the rest of the device to the kernels beside it. The segments of one launch run in no
// particular order; launches run in stream order. The segment list is pinned host memory the
// launches read; one thread at a time uses the queue.
class SegmentCopies {
public:
    // Up to `capacity` segments between restarts; queuing one more first waits for the stream.
    // Launches the kernel once (empty), so a lazy load never waits for a graph's device-side wait.
    SegmentCopies(cudaStream_t stream, std::size_t capacity);
    ~SegmentCopies();
    SegmentCopies(const SegmentCopies&)            = delete;
    SegmentCopies& operator=(const SegmentCopies&) = delete;

    // `bytes` from `source` to `target`, or zeros without a source, in the next launch.
    void queue(void* target, const void* source, std::uint64_t bytes);
    // Launches the segments queued since the last launch.
    void launch();
    // Launches, then waits for the stream; the list fills from its start again.
    void drain();
    // Once the stream's work has ended (a wait of the caller's), the list fills from its start.
    void restart() noexcept;
    // Restarts the list if every segment was launched and the stream is idle, without waiting.
    void restart_if_idle();
    // Drops the segments queued since the last launch.
    void discard() noexcept;
    // The segment list's pinned bytes.
    [[nodiscard]] std::size_t pinned_bytes() const noexcept;

private:
    cudaStream_t stream_ = nullptr;
    std::size_t capacity_ = 0, count_ = 0, launched_ = 0;
    std::uint64_t widest_ = 0;
    std::unique_ptr<PinnedHostBuffer> segments_;
};

} // namespace ninfer
