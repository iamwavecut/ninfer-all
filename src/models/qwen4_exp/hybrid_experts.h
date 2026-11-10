#pragma once

#include "core/device.h"
#include "models/qwen4_exp/expert_cache.h"
#include "models/expert_profile.h"
#include "ninfer/ops/moe_expert_cpu.h"
#include "ninfer/ops/moe_experts.h"
#include "ninfer/engine_options.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen4_exp {

struct HybridExpertLayer {
    std::size_t rank = 0;
    // Host-readable operands borrowed from the immutable Model, including each Use's policy.
    std::vector<ops::CpuExpertWeights> experts;
    bool registered = false; // every plane belongs to a Model-owned DMA registration
    std::vector<std::uint64_t> route_counts; // optional recorded initial cache ranking
};

// Program-owned native expert cache, bounded DMA staging and CPU workers. One sequence may
// execute at a time. GPU slots and CPU workspaces are allocated at construction (and by grow),
// never by Ops. The automatic cache (`cache_bytes` ~0) starts at half of what each device has
// free less a margin.
class HybridExperts {
public:
    HybridExperts(DeviceContext& device, std::vector<HybridExpertLayer> layers,
                  std::uint32_t token_capacity, std::uint64_t cache_bytes,
                  HybridExpertOptions options,
                  // Internal injection point for a delayed-worker cancellation check.
                  std::function<void()> before_cpu = {},
                  // Whether the stream takes the supervisor's operands through mapped memory and a
                  // kernel instead of a queued copy; unset follows serialized_copy_queue(). A test
                  // sets it to run either exchange on any driver.
                  std::optional<bool> mapped_exchange = {});
    ~HybridExperts();
    HybridExperts(const HybridExperts&) = delete;
    HybridExperts& operator=(const HybridExperts&) = delete;

    // Queues a capturable exchange with the CPU supervisor. The same schedule runs eagerly and
    // in graphs; finish() drains it and propagates worker failures before results are published.
    // The cancellation flag must outlive finish(). Bind it outside graph capture/replay.
    void begin(const std::atomic<bool>* cancelled, bool observe);
    void run(std::size_t layer, const Tensor& x, const Tensor& ids, const Tensor& weights,
             const Tensor& shared, ops::NativeMoeWeights banks, bool cpu_allowed,
             WorkspaceArena& workspace, Tensor& output);
    void finish();
    [[nodiscard]] ExpertRouteCounts route_counts() const;
    [[nodiscard]] ExpertCacheStats stats() const noexcept;
    [[nodiscard]] std::uint64_t cache_bytes(std::size_t rank) const;
    // Once startup's allocations are done, before the first run: more slots for every layer
    // from `bytes_by_rank`, shared by a rank's layers as at construction, which take the most
    // routed experts the cache lacks. Returns the bytes each rank gave. At most once.
    std::vector<std::uint64_t> grow(std::span<const std::uint64_t> bytes_by_rank);
    [[nodiscard]] std::uint64_t working_bytes(std::size_t rank) const;
    [[nodiscard]] std::string execution_profile() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen4_exp
