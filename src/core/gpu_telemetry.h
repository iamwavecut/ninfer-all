#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer {

// One NVML reading of a CUDA device this process sees. A field NVML could not read stays empty
// rather than zero.
struct GpuReading {
    int device = 0; // CUDA ordinal
    std::optional<unsigned> utilization_percent;
    std::optional<unsigned> memory_utilization_percent;
    std::optional<unsigned> temperature_celsius;
    std::optional<unsigned> sm_clock_mhz;
    std::optional<unsigned> memory_clock_mhz;
    std::optional<double> power_watts;
    std::optional<double> power_limit_watts;
    std::optional<std::uint64_t> memory_used_bytes;
    std::optional<std::uint64_t> memory_total_bytes;
};

// Reads every CUDA device of this process through NVML, matched by PCI bus id, so
// CUDA_VISIBLE_DEVICES and CUDA's ordering hold. NVML starts on the first call and stays loaded.
// Empty when NVML or the CUDA runtime is unavailable. Safe to call from any thread; a reading takes
// a few NVML calls per device.
[[nodiscard]] std::vector<GpuReading> read_gpu_telemetry();

} // namespace ninfer
