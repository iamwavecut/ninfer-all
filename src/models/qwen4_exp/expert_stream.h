#pragma once

// Disk-resident experts: they stay in the artifact's files and a device pool of slots holds the
// ones passes route to. Before a layer's experts run, prepare() makes every expert the layer's
// tokens routed to resident: the slots of the least recently used experts (never one the same call
// needs) take the missing ones, read from the files through the OS page cache into a page-locked
// staging ring (several reads in flight at once) and copied on the layer's stream, and the layer's
// expert tables are pointed at them. Empty slots are taken first, then released ones, then by
// CLOCK. Each device's layers share one pool whose slots fit their
// widest expert, except a layer marked own_pool whose experts are wider: it has slots of its own
// when the device's budget holds them beside the widest call of the others.
// The host memory this mode needs is the staging ring; the page cache keeps what the system can
// spare.

#include "core/device.h"
#include "core/expert_disk.h"
#include "models/qwen4_exp/model.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace ninfer::models::qwen4_exp {

struct ExpertStreamLayer {
    std::size_t rank    = 0;
    cudaStream_t stream = nullptr;
    // Gate, up and down: every expert's location, and the device table the kernels read.
    std::array<std::span<const ExpertLocation>, 3> experts;
    std::array<void*, 3> tables{};
    // The most experts one call routes to, 0 for every expert of the layer.
    std::size_t widest_call = 0;
    // The MTP block, whose experts may be stored wider than the text layers': with slots of their
    // own, the text layers' slots are not sized for them.
    bool own_pool = false;
};

struct ExpertStreamStats {
    std::uint64_t routes     = 0; // distinct (call, expert) needs
    std::uint64_t hits       = 0; // of them, already resident
    std::uint64_t read_bytes = 0;
    std::uint32_t slots      = 0;
    double read_seconds      = 0.0;
};

class ExpertStream {
public:
    // With `grows` (the automatic cache) the pools are laid out as `bytes_by_rank` lays them, but
    // start at half of it, or at the widest call of their layers if that is more; grow() adds to
    // them once startup is done. `disk`: a drive mirror whose copies the reads of odd stripes go
    // to, and the direct path (core/expert_disk.h).
    ExpertStream(DeviceContext& device, const std::vector<std::filesystem::path>& files,
                 std::vector<ExpertStreamLayer> layers,
                 std::span<const std::uint64_t> bytes_by_rank, bool grows = false,
                 const ExpertDiskReads& disk = {});
    ~ExpertStream();
    ExpertStream(const ExpertStream&)            = delete;
    ExpertStream& operator=(const ExpertStream&) = delete;

    // Makes every expert in `ids` resident for `layer` and its tables point at them, on the
    // layer's stream. Returns once the copies are queued; the staging ring is reused only after
    // the copies that read it finish. With `release_others` (a prompt span's last chunk), the
    // layer's residents `ids` lacks become the first slots later calls take once every slot holds
    // an expert, oldest release first, unless a call routes to them before: the earlier layers keep
    // their last chunk's experts while the later layers of the span read theirs.
    void prepare(std::size_t layer, std::span<const std::int32_t> ids, bool release_others);
    [[nodiscard]] ExpertStreamStats stats() const noexcept;
    // The device memory of `layer`'s slots, its pool's own block and grow's (empty until then):
    // a table entry inside either is a resident expert.
    using Ranges = std::array<std::pair<std::uintptr_t, std::uintptr_t>, 2>;
    [[nodiscard]] Ranges storage(std::size_t layer) const;
    // Once startup's allocations are done: more empty slots for the pool of each rank that the
    // budget sizes (not a layer's own slots, which hold its widest call), from `bytes_by_rank`,
    // taken before any slot that holds an expert. Returns the bytes each rank gave. At most once.
    std::vector<std::uint64_t> grow(std::span<const std::uint64_t> bytes_by_rank);
    // The device memory the pools of `rank` hold.
    [[nodiscard]] std::uint64_t pool_bytes(std::size_t rank) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen4_exp
