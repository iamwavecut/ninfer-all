#pragma once

#include "core/pinned_memory.h"

#include <cstddef>
#include <memory>

namespace ninfer {

// Large page-locked host memory: ordinary anonymous pages, faulted in on several threads (by the
// writers that fill them, or by pin() itself), then registered with every device context at once.
// A large block pins several times faster this way than through cudaHostAlloc, which zeroes and
// locks it on one thread (16 GiB on a Threadripper 3960X: 10.5 s for cudaHostAlloc, 2.9 s for an
// 8-thread first touch and cudaHostRegister), and is released faster. A first touch cannot fail
// cleanly, so a block larger than the memory the host still has available is refused before it is
// mapped. The pages start zeroed.
class HostPages {
public:
    explicit HostPages(std::size_t bytes);
    ~HostPages();
    HostPages(const HostPages&)            = delete;
    HostPages& operator=(const HostPages&) = delete;

    // Faults in the pages nobody wrote yet, on several threads, keeping what writers put there,
    // then locks and registers them; transfers may treat the block as pinned from then on.
    // `device_mapped` also maps the block into the device address space (see
    // HostMemoryRegistration).
    void pin(bool device_mapped = false);

    [[nodiscard]] std::byte* data() const noexcept { return data_; }

    [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

private:
    std::byte* data_   = nullptr;
    std::size_t bytes_ = 0;
    std::unique_ptr<HostMemoryRegistration> registration_;
};

} // namespace ninfer
