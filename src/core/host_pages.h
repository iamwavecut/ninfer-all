#pragma once

#include "core/pinned_memory.h"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

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

    // Pins the block from its start in parts that end at `ends` (ascending, past each other, the
    // last at most size()), one registration each, until a part would end past `limit` or the host
    // refuses one (Windows charges page-locked memory to a GPU's shared segment, about half the
    // RAM). Returns the bytes pinned: the end of the last pinned part, or 0. The pages past them
    // go back to the system and size() becomes that end; their bytes are lost. Faults in the pages
    // nobody wrote first, as pin().
    std::size_t pin_prefix(std::span<const std::size_t> ends, std::size_t limit,
                           bool device_mapped = false);

    [[nodiscard]] std::byte* data() const noexcept { return data_; }

    [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

    // Translate a range inside one independently registered part to its device address.
    [[nodiscard]] const std::byte* device_alias(std::size_t offset, std::size_t bytes) const;

private:
    struct RegistrationRange { std::size_t begin, end; };
    std::vector<RegistrationRange> ranges_;
    std::byte* data_   = nullptr;
    std::size_t bytes_ = 0;
    std::size_t mapping_bytes_ = 0; // the reservation, which size() may end before
    std::vector<std::unique_ptr<HostMemoryRegistration>> registrations_;
};

} // namespace ninfer
