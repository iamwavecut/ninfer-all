#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>

namespace ninfer {

// Memory this process may still take from the host: the smaller of the system's available memory
// and what remains under its memory cgroup. Empty when the platform gives no answer.
[[nodiscard]] std::optional<std::uint64_t> available_host_memory_bytes() noexcept;

// The machine's memory as this process may use it: the smaller of physical memory and the tightest
// cgroup limit above the process. Empty when the platform gives no answer.
[[nodiscard]] std::optional<std::uint64_t> total_host_memory_bytes() noexcept;

// What the memory cgroup of this process still allows, from the contents of `/proc/self/cgroup`
// and the cgroup mount `cgroup_root` (normally /sys/fs/cgroup). Every level from the process's own
// cgroup up to the root is read and the tightest remainder wins, because a limit may sit on any
// ancestor (a container runtime's or a systemd scope's cgroup). Handles cgroup v2, v1 and hybrid;
// reclaimable file cache does not count as use. Empty when no level sets a limit.
[[nodiscard]] std::optional<std::uint64_t>
cgroup_remaining_bytes(const std::filesystem::path& cgroup_root, std::string_view proc_self_cgroup);

// Page-locked memory this process may still register for the GPU whose adapter LUID is the 8 bytes
// at `luid` (cudaDeviceProp::luid). On Windows a GPU's page-locked host memory is charged to its
// shared WDDM segment, about half the RAM: that segment's budget less this process's use of it
// (DXGI), else half the physical memory less 8 GiB when DXGI gives no answer. WSL2 charges the
// same segment, about half of Windows' memory whatever the guest's: the same budget less use, read
// through /dev/dxg, and empty when the driver gives none. Empty elsewhere, where a process locks
// what the host has available.
[[nodiscard]] std::optional<std::uint64_t> pinnable_host_memory_bytes(const void* luid) noexcept;

// The tightest memory limit (not the remainder) any cgroup level above the process sets.
[[nodiscard]] std::optional<std::uint64_t>
cgroup_limit_bytes(const std::filesystem::path& cgroup_root, std::string_view proc_self_cgroup);

} // namespace ninfer
