#include "core/mapped_host_copy.h"

#include "core/device_launch.h"
#include "core/device_wait.h"
#include "core/pinned_memory.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace ninfer {
namespace {

constexpr unsigned kThreads = 256;
constexpr std::size_t kMaximumBlocks = 4096;

__global__ void copy_mapped_bytes(unsigned char* destination, const unsigned char* source,
                                  std::size_t bytes) {
    const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
    for (std::size_t at = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; at < bytes;
         at += stride) {
        destination[at] = source[at];
    }
}

// One copy of a copy_segments_kernel launch: `bytes` from `source` (device memory, or pinned host
// memory the device reads in place) to `target`, or zeros without a source.
struct CopySegment {
    std::byte* target       = nullptr;
    const std::byte* source = nullptr;
    std::uint64_t bytes     = 0;
};

constexpr int kCopyThreads = 256;
// A block takes eight 16-byte words a thread of a pass over the widest segment.
constexpr std::uint64_t kCopyBlockBytes = std::uint64_t(kCopyThreads) * 16 * 8;
// Every launch's blocks. On an RTX 4090 (PCIe 4.0 x16) 32 blocks at the greatest priority copied
// pinned host memory at 26.1-26.3 GB/s beside kernels that kept every SM busy, the copy engine's
// rate, and slowed those kernels by at most 1%; a row of 8 blocks for each of 512 segments slowed
// them by 191%.
constexpr std::uint64_t kCopyGridBlocks = 32;

template <class Word>
__device__ void copy_words(std::byte* target, const std::byte* source, std::uint64_t words,
                           std::uint64_t first, std::uint64_t stride) {
    auto* to         = reinterpret_cast<Word*>(target);
    const auto* from = reinterpret_cast<const Word*>(source);
    for (std::uint64_t i = first; i < words; i += stride) { to[i] = from ? from[i] : Word{}; }
}

// The blocks of a grid row stride over one segment's bytes: 16-byte words where target and source
// share their offset in a 16-byte word, else 4-byte words, else bytes, the bytes before the
// target's first whole word and after its last taken one at a time.
__device__ void copy_segment(const CopySegment segment, std::uint64_t first, std::uint64_t stride) {
    std::byte* target         = segment.target;
    const std::byte* source   = segment.source;
    const std::uint64_t bytes = segment.bytes;
    const auto at              = reinterpret_cast<std::uintptr_t>(target);
    const std::uintptr_t skew  = source ? at ^ reinterpret_cast<std::uintptr_t>(source) : 0;
    const std::uint64_t word   = (skew & 15U) == 0 ? 16 : (skew & 3U) == 0 ? 4 : 1;
    const std::uint64_t lead   = (word - (at & (word - 1))) & (word - 1);
    const std::uint64_t head   = lead < bytes ? lead : bytes;
    const std::uint64_t words  = (bytes - head) / word;
    const auto byte_at = [&](std::uint64_t i) { target[i] = source ? source[i] : std::byte{0}; };
    for (std::uint64_t i = first; i < head; i += stride) { byte_at(i); }
    const std::byte* from = source ? source + head : nullptr;
    if (word == 16) {
        copy_words<uint4>(target + head, from, words, first, stride);
    } else if (word == 4) {
        copy_words<std::uint32_t>(target + head, from, words, first, stride);
    } else {
        copy_words<unsigned char>(target + head, from, words, first, stride);
    }
    for (std::uint64_t i = head + words * word + first; i < bytes; i += stride) { byte_at(i); }
}

// The grid's rows stride over the `count` segments.
__global__ void __launch_bounds__(kCopyThreads)
    copy_segments_kernel(const CopySegment* segments, unsigned count) {
    const std::uint64_t first  = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride = std::uint64_t(gridDim.x) * blockDim.x;
    for (unsigned s = blockIdx.y; s < count; s += gridDim.y) {
        copy_segment(segments[s], first, stride);
    }
}

} // namespace

bool serialized_copy_queue() noexcept {
#ifdef _WIN32
    return true;
#else
    // WSL2 reaches the GPU through the dxgkrnl device; a native Linux driver has no such node.
    static const bool wsl = ::access("/dev/dxg", F_OK) == 0;
    return wsl;
#endif
}

MappedHostBuffer::MappedHostBuffer(std::size_t bytes) : size_(bytes) {
    if (bytes == 0) { throw std::invalid_argument("MappedHostBuffer size must be nonzero"); }
    CUDA_CHECK(cudaHostAlloc(&data_, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
}

MappedHostBuffer::~MappedHostBuffer() {
    if (data_ != nullptr) { ninfer::free_host_memory(data_); }
}

void MappedHostBuffer::copy_to_device(void* destination, std::size_t bytes,
                                      cudaStream_t stream) const {
    if (bytes > size_) { throw std::out_of_range("mapped host copy exceeds its source"); }
    if (bytes == 0) { return; }
    if (destination == nullptr) { throw std::invalid_argument("mapped host copy has no target"); }
    void* source = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&source, data_, 0));
    const auto blocks =
        static_cast<unsigned>(std::min((bytes - 1) / kThreads + 1, kMaximumBlocks));
    copy_mapped_bytes<<<blocks, kThreads, 0, stream>>>(
        static_cast<unsigned char*>(destination), static_cast<const unsigned char*>(source), bytes);
    CUDA_CHECK(cudaGetLastError());
}

bool copy_kernel_default() {
    const char* value = std::getenv("NINFER_MISS_COPY_KERNEL");
    if (value != nullptr && (value[0] == '0' || value[0] == '1') && value[1] == '\0') {
        return value[0] == '1';
    }
    return serialized_copy_queue();
}

cudaStream_t create_copy_stream(bool kernel) {
    int least    = 0;
    int greatest = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking,
                                            kernel ? greatest : least));
    return stream;
}

SegmentCopies::SegmentCopies(cudaStream_t stream, std::size_t capacity)
    : stream_(stream), capacity_(capacity),
      segments_(std::make_unique<PinnedHostBuffer>(capacity * sizeof(CopySegment))) {
    if (capacity == 0) { throw std::invalid_argument("SegmentCopies capacity must be nonzero"); }
    copy_segments_kernel<<<1, kCopyThreads, 0, stream_>>>(
        static_cast<const CopySegment*>(segments_->data()), 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(ninfer::wait_for_stream(stream_));
}

SegmentCopies::~SegmentCopies() = default;

void SegmentCopies::queue(void* target, const void* source, std::uint64_t bytes) {
    if (bytes == 0) { return; }
    if (count_ == capacity_) { drain(); }
    static_cast<CopySegment*>(segments_->data())[count_++] = CopySegment{
        static_cast<std::byte*>(target), static_cast<const std::byte*>(source), bytes};
    widest_ = std::max(widest_, bytes);
}

void SegmentCopies::launch() {
    const std::size_t count = count_ - launched_;
    if (count == 0) { return; }
    // Columns for the widest segment, rows for the rest of the grid.
    const auto columns = std::clamp<std::uint64_t>(
        (widest_ + kCopyBlockBytes - 1) / kCopyBlockBytes, 1, kCopyGridBlocks);
    const auto rows = std::clamp<std::uint64_t>(kCopyGridBlocks / columns, 1, count);
    copy_segments_kernel<<<dim3(unsigned(columns), unsigned(rows)), kCopyThreads, 0, stream_>>>(
        static_cast<const CopySegment*>(segments_->data()) + launched_, unsigned(count));
    CUDA_CHECK(cudaGetLastError());
    launched_ = count_;
    widest_   = 0;
}

void SegmentCopies::drain() {
    launch();
    CUDA_CHECK(ninfer::wait_for_stream(stream_));
    restart();
}

void SegmentCopies::restart() noexcept { count_ = launched_ = 0; }

void SegmentCopies::restart_if_idle() {
    if (count_ == 0 || launched_ != count_) { return; }
    const cudaError_t status = cudaStreamQuery(stream_);
    if (status == cudaErrorNotReady) { return; }
    CUDA_CHECK(status);
    restart();
}

void SegmentCopies::discard() noexcept {
    count_  = launched_;
    widest_ = 0;
}

std::size_t SegmentCopies::pinned_bytes() const noexcept { return segments_->size(); }

} // namespace ninfer
