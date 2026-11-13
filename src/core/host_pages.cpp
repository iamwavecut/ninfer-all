#include "core/host_pages.h"

#include "core/host_memory.h"
#include "core/pinned_memory.h"

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
    data_  = static_cast<std::byte*>(mapping);
    bytes_ = bytes;
}

HostPages::~HostPages() {
    registration_.reset();
    if (data_ == nullptr) { return; }
#ifdef _WIN32
    (void)VirtualFree(data_, 0, MEM_RELEASE);
#else
    (void)munmap(data_, bytes_);
#endif
}

void HostPages::pin(bool device_mapped) {
    if (registration_) { return; }
    // Registration faults in every page nobody wrote on its one thread; rewriting a byte of each
    // page first does that on several. A page a writer filled keeps its bytes.
    const std::size_t page  = page_bytes();
    const std::size_t pages = (bytes_ + page - 1) / page;
    const std::size_t count = std::clamp<std::size_t>(
        std::min<std::size_t>(std::thread::hardware_concurrency(), bytes_ >> 26U), 1, 16);
    const auto touch = [&](std::size_t first, std::size_t last) {
        for (std::size_t index = first; index < last; ++index) {
            auto* const byte = reinterpret_cast<volatile unsigned char*>(data_ + index * page);
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
    try {
        registration_ = std::make_unique<HostMemoryRegistration>(data_, bytes_, device_mapped);
    } catch (const std::exception& error) {
        throw std::runtime_error("failed to pin " + mib(bytes_) + " MiB of host memory (" +
                                 error.what() + ")");
    }
}

} // namespace ninfer
