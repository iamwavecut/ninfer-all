#include "core/host_memory.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>

#include <cstring>
#endif

#if defined(__linux__)
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <array>
#include <cstring>
#endif

namespace ninfer {
namespace {

std::optional<std::uint64_t> read_u64_file(const std::filesystem::path& path) {
    std::ifstream file(path);
    std::string text;
    if (!file || !(file >> text) || text == "max") { return std::nullopt; }
    try {
        return std::stoull(text);
    } catch (...) { return std::nullopt; }
}

// File-backed page cache charged to a cgroup (active and inactive lists) is reclaimable, so it does
// not count against what remains: pages read more than once, such as an artifact just loaded, sit
// on the active list.
std::uint64_t reclaimable_file_bytes(const std::filesystem::path& path, const char* active_key,
                                     const char* inactive_key) {
    std::ifstream file(path);
    std::string name;
    std::uint64_t value = 0;
    std::uint64_t total = 0;
    while (file >> name >> value) {
        if (name == active_key || name == inactive_key) { total += value; }
    }
    return total;
}

enum class CgroupMeasure { Remaining, Limit };

// One cgroup directory's limit, or its limit less its non-reclaimable use. Empty when the directory
// sets no limit (or does not exist), which leaves the decision to its ancestors.
std::optional<std::uint64_t> measure_under(const std::filesystem::path& dir, bool v2,
                                           CgroupMeasure measure) {
    const auto limit = read_u64_file(dir / (v2 ? "memory.max" : "memory.limit_in_bytes"));
    // cgroup v1 reports "no limit" as a value near the top of the page-aligned 63-bit range.
    if (!limit || (!v2 && *limit >= (1ULL << 60))) { return std::nullopt; }
    if (measure == CgroupMeasure::Limit) { return limit; }
    const auto used = read_u64_file(dir / (v2 ? "memory.current" : "memory.usage_in_bytes"));
    if (!used) { return std::nullopt; }
    const std::uint64_t reclaimable = reclaimable_file_bytes(
        dir / "memory.stat", v2 ? "active_file" : "total_active_file",
        v2 ? "inactive_file" : "total_inactive_file");
    const std::uint64_t charged = *used > reclaimable ? *used - reclaimable : 0;
    return *limit > charged ? *limit - charged : 0;
}

std::vector<std::string> path_components(std::string path) {
    if (const auto deleted = path.find(" (deleted)"); deleted != std::string::npos) {
        path.erase(deleted);
    }
    std::vector<std::string> parts;
    std::string part;
    std::istringstream stream(path);
    while (std::getline(stream, part, '/')) {
        if (!part.empty()) { parts.push_back(part); }
    }
    return parts;
}

// The tightest value over `base/<path>` and every ancestor up to `base`.
std::optional<std::uint64_t> tightest(const std::filesystem::path& base,
                                      const std::vector<std::string>& parts, bool v2,
                                      CgroupMeasure measure) {
    std::optional<std::uint64_t> best;
    for (std::size_t depth = parts.size() + 1; depth-- > 0;) {
        std::filesystem::path dir = base;
        for (std::size_t index = 0; index < depth; ++index) { dir /= parts[index]; }
        if (const auto value = measure_under(dir, v2, measure)) {
            best = best ? std::min(*best, *value) : *value;
        }
    }
    return best;
}

std::optional<std::uint64_t> cgroup_tightest(const std::filesystem::path& cgroup_root,
                                             std::string_view proc_self_cgroup,
                                             CgroupMeasure measure) {
    std::optional<std::uint64_t> best;
    const auto take = [&](std::optional<std::uint64_t> value) {
        if (value) { best = best ? std::min(*best, *value) : *value; }
    };
    std::istringstream lines{std::string(proc_self_cgroup)};
    std::string line;
    while (std::getline(lines, line)) {
        // "<hierarchy>:<controllers>:<path>"; the path may itself contain colons.
        const std::size_t first  = line.find(':');
        const std::size_t second = first == std::string::npos ? first : line.find(':', first + 1);
        if (second == std::string::npos) { continue; }
        const std::string controllers        = line.substr(first + 1, second - first - 1);
        const std::vector<std::string> parts = path_components(line.substr(second + 1));
        if (line.compare(0, first, "0") == 0 && controllers.empty()) {
            take(tightest(cgroup_root, parts, true, measure));
        } else if ((',' + controllers + ',').find(",memory,") != std::string::npos) {
            take(tightest(cgroup_root / "memory", parts, false, measure));
        }
    }
    return best;
}

#if defined(__linux__)
std::optional<std::uint64_t> meminfo_bytes(std::string_view wanted) {
    std::ifstream file("/proc/meminfo");
    std::string key;
    std::uint64_t kib = 0;
    std::string unit;
    while (file >> key >> kib) {
        std::getline(file, unit);
        if (key == wanted) { return kib * 1024ULL; }
    }
    return std::nullopt;
}

std::string proc_self_cgroup() {
    std::ifstream self("/proc/self/cgroup");
    std::ostringstream text;
    text << self.rdbuf();
    return text.str();
}

// WSL2's /dev/dxg answers the D3DKMT calls DXGI makes on Windows. The layouts and request codes
// are those of include/uapi/misc/d3dkmthk.h in Microsoft's WSL2-Linux-Kernel, unchanged from 5.10
// to 6.18.
struct DxgAdapterInfo {           // d3dkmt_adapterinfo
    std::uint32_t adapter;        // d3dkmthandle
    std::uint32_t luid[2];        // winluid: LowPart, HighPart
    std::uint32_t sources;
    std::uint32_t move_regions_preferred;
};
struct DxgEnumAdapters3 {         // d3dkmt_enumadapters3
    std::uint64_t filter;
    std::uint32_t count;
    std::uint32_t reserved;
    std::uint64_t adapters;       // DxgAdapterInfo[count]
};
struct DxgQueryVideoMemoryInfo {  // d3dkmt_queryvideomemoryinfo
    std::uint64_t process;        // 0: this process
    std::uint32_t adapter;
    std::uint32_t segment_group;
    std::uint64_t budget;
    std::uint64_t current_usage;
    std::uint64_t current_reservation;
    std::uint64_t available_for_reservation;
    std::uint32_t physical_adapter_index;
};
static_assert(sizeof(DxgAdapterInfo) == 20 && sizeof(DxgEnumAdapters3) == 24 &&
                  sizeof(DxgQueryVideoMemoryInfo) == 56,
              "d3dkmthk.h layouts");
constexpr unsigned long kDxgEnumAdapters3        = _IOWR(0x47, 0x3e, DxgEnumAdapters3);
constexpr unsigned long kDxgQueryVideoMemoryInfo = _IOWR(0x47, 0x0a, DxgQueryVideoMemoryInfo);
constexpr unsigned long kDxgCloseAdapter         = _IOWR(0x47, 0x15, std::uint32_t);
constexpr std::uint64_t kDxgIncludeComputeOnly   = 1;
constexpr std::uint32_t kDxgSegmentGroupNonLocal = 1;

// What the shared WDDM segment of the GPU whose LUID is at `luid` can still lock for this process:
// its budget less this process's use, as DXGI reports it on Windows. When no adapter has that
// LUID, the tightest adapter answers: this process's page-locked memory is charged to the CUDA
// device's adapter. Empty when the driver gives no budget.
std::optional<std::uint64_t> wsl_pinnable_bytes(const void* luid) {
    const int dxg = ::open("/dev/dxg", O_RDWR | O_CLOEXEC);
    if (dxg < 0) { return std::nullopt; }
    std::array<DxgAdapterInfo, 64> adapters{};
    DxgEnumAdapters3 list{.filter   = kDxgIncludeComputeOnly,
                          .count    = static_cast<std::uint32_t>(adapters.size()),
                          .reserved = 0,
                          .adapters = reinterpret_cast<std::uintptr_t>(adapters.data())};
    std::optional<std::uint64_t> matched, tightest_adapter;
    if (::ioctl(dxg, kDxgEnumAdapters3, &list) >= 0) {
        const std::uint32_t count = std::min<std::uint32_t>(list.count, adapters.size());
        for (std::uint32_t i = 0; i < count; ++i) {
            DxgQueryVideoMemoryInfo info{};
            info.adapter       = adapters[i].adapter;
            info.segment_group = kDxgSegmentGroupNonLocal;
            if (::ioctl(dxg, kDxgQueryVideoMemoryInfo, &info) >= 0 && info.budget > 0) {
                const std::uint64_t remaining =
                    info.budget > info.current_usage ? info.budget - info.current_usage : 0;
                if (luid != nullptr && std::memcmp(adapters[i].luid, luid, 8) == 0) {
                    matched = remaining;
                }
                tightest_adapter =
                    tightest_adapter ? std::min(*tightest_adapter, remaining) : remaining;
            }
            std::uint32_t handle = adapters[i].adapter;
            (void)::ioctl(dxg, kDxgCloseAdapter, &handle);
        }
    }
    ::close(dxg);
    return matched ? matched : tightest_adapter;
}
#endif

} // namespace

std::optional<std::uint64_t> cgroup_remaining_bytes(const std::filesystem::path& cgroup_root,
                                                    std::string_view proc_self_cgroup) {
    return cgroup_tightest(cgroup_root, proc_self_cgroup, CgroupMeasure::Remaining);
}

std::optional<std::uint64_t> cgroup_limit_bytes(const std::filesystem::path& cgroup_root,
                                                std::string_view proc_self_cgroup) {
    return cgroup_tightest(cgroup_root, proc_self_cgroup, CgroupMeasure::Limit);
}

std::optional<std::uint64_t> available_host_memory_bytes() noexcept {
    try {
#if defined(_WIN32)
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (!GlobalMemoryStatusEx(&status)) { return std::nullopt; }
        return static_cast<std::uint64_t>(status.ullAvailPhys);
#elif defined(__linux__)
        const std::optional<std::uint64_t> system = meminfo_bytes("MemAvailable:");
        const std::optional<std::uint64_t> cgroup =
            cgroup_remaining_bytes("/sys/fs/cgroup", proc_self_cgroup());
        if (system && cgroup) { return std::min(*system, *cgroup); }
        return system ? system : cgroup;
#else
        return std::nullopt;
#endif
    } catch (...) { return std::nullopt; }
}

std::optional<std::uint64_t> pinnable_host_memory_bytes(const void* luid) noexcept {
#if defined(_WIN32)
    try {
        // dxgi.dll is loaded when asked, not linked.
        if (HMODULE dxgi = LoadLibraryA("dxgi.dll"); dxgi != nullptr) {
            using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
            const auto create = reinterpret_cast<CreateFactory>(
                reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1")));
            IDXGIFactory1* factory = nullptr;
            std::optional<std::uint64_t> remaining;
            if (create != nullptr && luid != nullptr &&
                SUCCEEDED(create(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))) &&
                factory != nullptr) {
                IDXGIAdapter1* adapter = nullptr;
                for (UINT i = 0; !remaining && factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND;
                     ++i) {
                    DXGI_ADAPTER_DESC1 desc{};
                    IDXGIAdapter3* adapter3 = nullptr;
                    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
                    if (adapter != nullptr && SUCCEEDED(adapter->GetDesc1(&desc)) &&
                        std::memcmp(&desc.AdapterLuid, luid, sizeof(desc.AdapterLuid)) == 0 &&
                        SUCCEEDED(adapter->QueryInterface(__uuidof(IDXGIAdapter3),
                                                          reinterpret_cast<void**>(&adapter3))) &&
                        adapter3 != nullptr &&
                        SUCCEEDED(adapter3->QueryVideoMemoryInfo(
                            0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info)) &&
                        info.Budget > 0) {
                        remaining = info.Budget > info.CurrentUsage ? info.Budget - info.CurrentUsage : 0;
                    }
                    if (adapter3 != nullptr) { adapter3->Release(); }
                    if (adapter != nullptr) { adapter->Release(); }
                    adapter = nullptr;
                }
                factory->Release();
            }
            FreeLibrary(dxgi);
            if (remaining) { return remaining; }
        }
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (!GlobalMemoryStatusEx(&status)) { return std::nullopt; }
        const std::uint64_t half = status.ullTotalPhys / 2;
        return half > (8ULL << 30) ? half - (8ULL << 30) : 0;
    } catch (...) { return std::nullopt; }
#elif defined(__linux__)
    // WSL2 (/dev/dxg) charges a GPU's page-locked memory to the same WDDM segment, about half of
    // Windows' memory, whatever memory the guest has; without a budget from the driver the pins
    // stop at the first part the host refuses.
    return wsl_pinnable_bytes(luid);
#else
    (void)luid;
    return std::nullopt;
#endif
}

std::optional<std::uint64_t> total_host_memory_bytes() noexcept {
    try {
#if defined(_WIN32)
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (!GlobalMemoryStatusEx(&status)) { return std::nullopt; }
        return static_cast<std::uint64_t>(status.ullTotalPhys);
#elif defined(__linux__)
        const std::optional<std::uint64_t> system = meminfo_bytes("MemTotal:");
        const std::optional<std::uint64_t> cgroup =
            cgroup_limit_bytes("/sys/fs/cgroup", proc_self_cgroup());
        if (system && cgroup) { return std::min(*system, *cgroup); }
        return system ? system : cgroup;
#else
        return std::nullopt;
#endif
    } catch (...) { return std::nullopt; }
}

} // namespace ninfer
