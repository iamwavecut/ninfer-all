#include "runtime/engine/host_memory.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::runtime {

ContextCacheOptions resolve_auto_host_cache(const ContextCacheOptions& requested,
                                            std::uint64_t available_bytes,
                                            std::optional<std::uint64_t> total_bytes,
                                            std::uint64_t extra_reserve_bytes) {
    if (!requested.host_cache_auto) {
        throw std::logic_error("host cache sizing was requested without host_cache_auto");
    }
    // Pinned pages cannot be reclaimed, so the reserve for what still grows (request buffers, the
    // response store, graph instantiation, the media caches) is a fixed amount, not a fraction.
    const std::uint64_t max  = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t base = requested.host_cache_reserve_bytes;
    const std::uint64_t reserve =
        base > max - extra_reserve_bytes ? max : base + extra_reserve_bytes;
    std::uint64_t budget = available_bytes > reserve ? available_bytes - reserve : 0;
    // The reserve leaves memory for what still grows; a cap leaves it for the machine's other
    // users, which the available memory cannot know will want it.
    if (requested.host_cache_max_bytes) {
        budget = std::min<std::uint64_t>(budget, *requested.host_cache_max_bytes);
    }
    if (requested.host_cache_percent) {
        const std::uint64_t percent = *requested.host_cache_percent;
        if (percent == 0 || percent > 100) {
            throw std::invalid_argument("host cache percent must be in [1,100]");
        }
        if (!total_bytes) {
            throw std::invalid_argument("host cache percent needs the machine's total memory");
        }
        const std::uint64_t total = *total_bytes;
        budget =
            std::min<std::uint64_t>(budget, total / 100 * percent + total % 100 * percent / 100);
    }
    budget = std::min<std::uint64_t>(budget, std::numeric_limits<std::size_t>::max());
    ContextCacheOptions resolved     = requested;
    resolved.host_cache_auto         = false;
    resolved.host_cache_budget_bytes = static_cast<std::size_t>(budget);
    return resolved;
}

ContextCacheOptions resolve_auto_host_cache_now(const ContextCacheOptions& requested,
                                                std::uint64_t extra_reserve_bytes) {
    if (!requested.host_cache_auto) { return requested; }
    std::optional<std::uint64_t> available = available_host_memory_bytes();
    if (!available) {
        throw std::runtime_error(
            "--host-cache-mib auto needs the host's available memory, which this platform does "
            "not report; give the budget in MiB");
    }
#if defined(_WIN32)
    // The tier is pinned, which Windows charges to the GPU's shared memory segment.
    int device = 0;
    cudaDeviceProp props{};
    if (cudaGetDevice(&device) == cudaSuccess &&
        cudaGetDeviceProperties(&props, device) == cudaSuccess) {
        if (const auto pinnable = pinnable_host_memory_bytes(props.luid)) {
            available = std::min(*available, *pinnable);
        }
    } else {
        (void)cudaGetLastError();
    }
#endif
    return resolve_auto_host_cache(requested, *available, total_host_memory_bytes(),
                                   extra_reserve_bytes);
}

} // namespace ninfer::runtime
