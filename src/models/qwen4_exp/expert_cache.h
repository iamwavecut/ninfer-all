#pragma once

// A device cache of the routed experts of host-resident banks. Every expert of a layer is reached
// through device tables of base pointers (one per projection) that the expert kernels read; a
// cached expert's entries point at its copy in a device slot, every other entry at its bytes in the
// pinned host block. Between forward passes the cache counts the routes the last pass took, with an
// exponential decay per token, and swaps the experts it would most often have needed into the
// slots of the ones it needed least, by copy and table update on the layer's stream. The kernels
// never see the difference, so no pass waits on the cache.

#include "core/arena.h"
#include "core/device.h"
#include "core/weight.h"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace ninfer::models::qwen4_exp {

struct ExpertBank {
    std::int64_t expert_bytes = 0; // one expert's rows
    std::vector<const void*> host; // each expert's rows in the pinned host block
    void* table = nullptr;         // the device table the kernels read
};

struct ExpertCacheLayer {
    std::size_t rank    = 0;
    cudaStream_t stream = nullptr;
    ExpertBank gate, up, down;
};

struct ExpertCacheStats {
    std::uint64_t routes       = 0; // (token, expert) pairs observed
    std::uint64_t hits         = 0; // of them, served from a slot
    std::uint64_t admitted     = 0; // experts copied into slots
    std::uint64_t copied_bytes = 0;
    std::uint32_t slots        = 0;
    std::uint64_t cpu_routes   = 0; // native hybrid: routed (token, expert) pairs computed on CPU
    std::uint64_t dma_routes   = 0; // native hybrid: uncached pairs staged to GPU slots
};

class ExpertCache {
public:
    // `bytes_by_rank` is the device memory each rank lends its layers' slots.
    ExpertCache(DeviceContext& device, std::vector<ExpertCacheLayer> layers,
                std::span<const std::uint64_t> bytes_by_rank);
    ~ExpertCache();
    ExpertCache(const ExpertCache&)            = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;

    // The routes one pass took through `layer`: `ids` holds every (token, slot) pair's expert.
    void observe(std::size_t layer, std::span<const std::int32_t> ids, std::uint32_t tokens);
    // Copies up to `byte_budget` bytes of experts into slots and updates the tables, on each
    // layer's stream; handed-over layers only with `handed_over_too` (their admitting calls are
    // done, as after a prompt chunk).
    void rebalance(std::uint64_t byte_budget, bool handed_over_too = false);
    // Projection k (0 gate, 1 up, 2 down) of `expert` in its device slot, or null when the expert
    // is not cached. Slots are zeroed when allocated and 256 zero bytes follow the last, as the
    // expert matrix kernel needs (see moe_experts_gguf).
    [[nodiscard]] const void* cached(std::size_t layer, int k, std::int32_t expert) const;
    [[nodiscard]] ExpertCacheStats stats() const noexcept;
    // Fills every layer's slots with its most routed experts of `counts` (a recorded profile of
    // the same layers) and starts their scores from it, before the first pass.
    void seed(const std::vector<std::vector<std::uint64_t>>& counts);
    // Every route each layer's experts took since startup.
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> totals() const;
    // The device memory of `layer`'s slots, in up to two blocks (the second empty until grow):
    // a table entry inside either is a cached expert.
    using Ranges = std::array<std::pair<std::uintptr_t, std::uintptr_t>, 2>;
    [[nodiscard]] Ranges storage(std::size_t layer) const;
    // Once startup's allocations are done: more slots for every layer that is not full, from
    // `bytes_by_rank`, as free slots. Returns the bytes each rank gave. At most once.
    std::vector<std::uint64_t> grow(std::span<const std::uint64_t> bytes_by_rank);

    // Slot admission by the expert misses: from now on `layer` changes only through admit, which
    // the misses' service thread calls while the layer's call runs, and rebalance leaves it alone.
    void hand_over(std::size_t layer);
    struct Admitted {
        std::array<void*, 3> slot{};              // the expert's projections in its slot
        std::int32_t victim = -1;                 // the expert that left the slot, or -1
        std::array<const void*, 3> victim_host{}; // the victim's projections in host memory
    };
    // Takes a slot of a handed-over `layer` for the uncached `expert`: a free one, else, among
    // the experts `in_use` (a flag per expert) does not mark, the one `last_use` stamps least
    // recently in the lowest tier of decayed route counts (about once lately, a few times, often),
    // so a passing decode does not push out a working set other prompts keep routing to. False
    // when the layer has no slot to give. The caller copies the expert in and
    // updates the device tables before the layer's next call; a route of the expert counts as a
    // miss until the next observe.
    bool admit(std::size_t layer, std::int32_t expert, std::span<const std::uint32_t> last_use,
               std::span<const unsigned char> in_use, Admitted& out);
    // The device tables (gate, up, down) of `layer`.
    [[nodiscard]] std::array<void*, 3> tables(std::size_t layer) const;

private:
    struct Layer;
    DeviceContext& device_;
    std::vector<Layer> layers_;
    std::unique_ptr<PinnedHostBuffer> staging_;
    ExpertCacheStats stats_;
    // Guards the bookkeeping, which the misses' service thread changes through admit.
    mutable std::mutex mutex_;
};

} // namespace ninfer::models::qwen4_exp
