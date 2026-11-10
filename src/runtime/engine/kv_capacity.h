#pragma once

#include "ninfer/types_fwd.h"
#include "runtime/contract/resources.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::runtime {

// `available_runtime_bytes` is the primary device's free memory after weights;
// `extra_rank_available_bytes` is the same for each further device, in the order of
// `curve.extra_ranks`. The capacity is the largest that fits every device, and a failure names the
// device that could not hold it.
[[nodiscard]] KvCapacityResolution
resolve_kv_capacity(const KvCapacityPolicy& policy, const SequenceCapacityCurve& curve,
                    std::size_t available_runtime_bytes,
                    std::span<const std::size_t> extra_rank_available_bytes = {});

// Whether the capacity asked for fits every device: the explicit capacity, or under automatic
// capacity the smallest reservation beside the headroom.
[[nodiscard]] bool kv_capacity_fits(const KvCapacityPolicy& policy,
                                    const SequenceCapacityCurve& curve,
                                    std::size_t available_runtime_bytes,
                                    std::span<const std::size_t> extra_rank_available_bytes = {});

// What each device (device 0 first) keeps free beyond the Engine's reservation for the KV cache
// asked for, negative where it falls short: the explicit capacity, or `automatic_page_groups` Main
// page groups (and the resident ones) under automatic capacity, which also keeps its headroom free
// on every device. The page count is held to the curve's range.
[[nodiscard]] std::vector<std::int64_t>
kv_capacity_spare(const KvCapacityPolicy& policy, const SequenceCapacityCurve& curve,
                  std::uint32_t automatic_page_groups, std::size_t available_runtime_bytes,
                  std::span<const std::size_t> extra_rank_available_bytes = {});

} // namespace ninfer::runtime
