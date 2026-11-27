#include "core/host_pages.h"

#include "core/host_memory.h"
#include "core/pinned_memory.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <sys/mman.h>
#    include <unistd.h>
#endif

namespace ninfer {
namespace {

std::string mib(std::size_t bytes) { return std::to_string((bytes + (1ULL << 20) - 1) >> 20); }

std::size_t page_bytes() noexcept {
#ifdef _WIN32
    return 4096;
#else
    const long page = sysconf(_SC_PAGESIZE);
    return page > 0 ? static_cast<std::size_t>(page) : 4096;
#endif
}

// Rewrites a byte of every page in [first, last) on several threads, which faults in the pages
// nobody wrote; a page a writer filled keeps its bytes. Registration would fault them in on its
// one thread.
void touch_pages(std::byte* data, std::size_t first, std::size_t last) {
    const std::size_t page  = page_bytes();
    const std::size_t begin = first / page, pages = (last + page - 1) / page - begin;
    const std::size_t count = std::clamp<std::size_t>(
        std::min<std::size_t>(std::thread::hardware_concurrency(), (last - first) >> 26U), 1, 16);
    const auto touch = [&](std::size_t from, std::size_t to) {
        for (std::size_t index = begin + from; index < begin + to; ++index) {
            auto* const byte = reinterpret_cast<volatile unsigned char*>(data + index * page);
            *byte            = *byte;
        }
    };
    std::vector<std::thread> threads;
    try {
        for (std::size_t i = 1; i < count; ++i) {
            threads.emplace_back(touch, pages * i / count, pages * (i + 1) / count);
        }
    } catch (...) {
        // Fewer threads only slow the touch; this thread finishes the share it could not hand out.
        touch(pages * (threads.size() + 1) / count, pages);
    }
    touch(0, pages / count);
    for (auto& thread : threads) { thread.join(); }
}

} // namespace

HostPages::HostPages(std::size_t bytes) {
    if (bytes == 0) { throw std::invalid_argument("HostPages size must be nonzero"); }
    if (const auto available = available_host_memory_bytes(); available && bytes > *available) {
        throw std::runtime_error("failed to pin " + mib(bytes) +
                                 " MiB of host memory: the host has " +
                                 std::to_string(*available >> 20) + " MiB available");
    }
#ifdef _WIN32
    void* mapping = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (mapping == nullptr) { throw std::bad_alloc(); }
#else
    void* mapping =
        mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED) { throw std::bad_alloc(); }
#endif
    data_          = static_cast<std::byte*>(mapping);
    bytes_         = bytes;
    mapping_bytes_ = bytes;
}

HostPages::~HostPages() {
    registrations_.clear();
    if (data_ == nullptr) { return; }
#ifdef _WIN32
    (void)VirtualFree(data_, 0, MEM_RELEASE);
#else
    (void)munmap(data_, mapping_bytes_);
#endif
}

void HostPages::pin(bool device_mapped) {
    if (!registrations_.empty()) { return; }
    touch_pages(data_, 0, bytes_);
    ranges_.reserve(1);
    try {
        registrations_.push_back(
            std::make_unique<HostMemoryRegistration>(data_, bytes_, device_mapped));
        ranges_.push_back({0, bytes_});
    } catch (const std::exception& error) {
        throw std::runtime_error("failed to pin " + mib(bytes_) + " MiB of host memory (" +
                                 error.what() + ")");
    }
}

std::size_t HostPages::pin_prefix(std::span<const std::size_t> ends, std::size_t limit,
                                  bool device_mapped) {
    if (!registrations_.empty()) { throw std::logic_error("HostPages: the block is pinned"); }
    // Two registrations may not share a page, so each part but the block's last ends on one.
    const std::size_t page = page_bytes();
    ranges_.reserve(ends.size());
    std::size_t pinned = 0, registered = 0;
    for (std::size_t i = 0; i < ends.size(); ++i) {
        const std::size_t end = ends[i];
        if (end <= pinned || end > bytes_) {
            throw std::invalid_argument("HostPages: part ends must ascend within the block");
        }
        if (end > limit) { break; }
        const std::size_t through = std::min((end + page - 1) / page * page, bytes_);
        if (through > registered) {
            touch_pages(data_, registered, through);
            try {
                registrations_.push_back(std::make_unique<HostMemoryRegistration>(
                    data_ + registered, through - registered, device_mapped));
                ranges_.push_back({registered, through});
            } catch (const std::exception&) { break; }
            registered = through;
        }
        pinned = end;
    }
    // The pages past the pinned bytes go back; one a pinned part shares stays.
    const std::size_t kept = (pinned + page - 1) / page * page;
    if (kept < mapping_bytes_) {
#ifdef _WIN32
        (void)VirtualFree(data_ + kept, mapping_bytes_ - kept, MEM_DECOMMIT);
#else
        (void)madvise(data_ + kept, mapping_bytes_ - kept, MADV_DONTNEED);
#endif
    }
    bytes_ = pinned;
    return pinned;
}

const std::byte* HostPages::device_alias(std::size_t offset, std::size_t bytes) const {
    if (bytes == 0 || offset >= bytes_ || bytes > bytes_ - offset) {
        throw std::out_of_range("HostPages: device alias range is outside pinned bytes");
    }
    const auto it = std::upper_bound(ranges_.begin(), ranges_.end(), offset,
                                     [](std::size_t value, const RegistrationRange& range) {
                                         return value < range.begin;
                                     });
    if (it == ranges_.begin()) { throw std::out_of_range("HostPages: range is not registered"); }
    const auto& range = *std::prev(it);
    if (offset < range.begin || offset >= range.end || bytes > range.end - offset) {
        throw std::out_of_range("HostPages: device alias crosses registrations");
    }
    void* alias = nullptr;
    const cudaError_t error = cudaHostGetDevicePointer(&alias, data_ + range.begin, 0);
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("HostPages: cudaHostGetDevicePointer: ") +
                                 cudaGetErrorString(error));
    }
    return static_cast<const std::byte*>(alias) + (offset - range.begin);
}

} // namespace ninfer
