#include "core/gpu_telemetry.h"

#include "core/nvml_library.h"

#include <cuda_runtime_api.h>

#include <utility>

namespace ninfer {
namespace {

// NVML's stable C ABI (nvml.h): every entry point below exists since driver 450.
using NvmlReturn = int;
using NvmlDevice = void*;
struct NvmlUtilization {
    unsigned int gpu;
    unsigned int memory;
};
struct NvmlMemory {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};
// nvmlMemory_v2_t (driver 510 and newer): `used` leaves out the memory the driver reserves, as
// nvidia-smi reports it; the first form counts the reservation (about 450 MiB on an idle RTX 3090).
struct NvmlMemoryV2 {
    unsigned int version;
    unsigned long long total;
    unsigned long long reserved;
    unsigned long long free;
    unsigned long long used;
};
constexpr unsigned int kMemoryV2 = static_cast<unsigned int>(sizeof(NvmlMemoryV2)) | (2U << 24U);
using InitFn        = NvmlReturn (*)();
using ByPciFn       = NvmlReturn (*)(const char*, NvmlDevice*);
using UtilizationFn = NvmlReturn (*)(NvmlDevice, NvmlUtilization*);
using MemoryFn      = NvmlReturn (*)(NvmlDevice, NvmlMemory*);
using MemoryV2Fn    = NvmlReturn (*)(NvmlDevice, NvmlMemoryV2*);
using TemperatureFn = NvmlReturn (*)(NvmlDevice, int, unsigned int*);
using UnsignedFn    = NvmlReturn (*)(NvmlDevice, unsigned int*);
using ClockFn       = NvmlReturn (*)(NvmlDevice, int, unsigned int*);
constexpr NvmlReturn kSuccess       = 0;
constexpr int kTemperatureGpu       = 0; // NVML_TEMPERATURE_GPU
constexpr int kClockSm              = 1; // NVML_CLOCK_SM
constexpr int kClockMemory          = 2; // NVML_CLOCK_MEM

class Sampler {
public:
    static const Sampler& instance() {
        static const Sampler sampler;
        return sampler;
    }

    [[nodiscard]] std::vector<GpuReading> read() const {
        std::vector<GpuReading> out;
        out.reserve(devices_.size());
        for (const auto& [ordinal, handle] : devices_) {
            GpuReading r;
            r.device = ordinal;
            NvmlUtilization rates{};
            if (utilization_ != nullptr && utilization_(handle, &rates) == kSuccess) {
                r.utilization_percent        = rates.gpu;
                r.memory_utilization_percent = rates.memory;
            }
            NvmlMemoryV2 memory2{.version = kMemoryV2};
            NvmlMemory memory{};
            if (memory_v2_ != nullptr && memory_v2_(handle, &memory2) == kSuccess) {
                r.memory_used_bytes  = memory2.used;
                r.memory_total_bytes = memory2.total;
            } else if (memory_ != nullptr && memory_(handle, &memory) == kSuccess) {
                r.memory_used_bytes  = memory.used;
                r.memory_total_bytes = memory.total;
            }
            unsigned int value = 0;
            if (temperature_ != nullptr && temperature_(handle, kTemperatureGpu, &value) == kSuccess) {
                r.temperature_celsius = value;
            }
            if (power_ != nullptr && power_(handle, &value) == kSuccess) { r.power_watts = value * 1e-3; }
            if (power_limit_ != nullptr && power_limit_(handle, &value) == kSuccess) {
                r.power_limit_watts = value * 1e-3;
            }
            if (clock_ != nullptr && clock_(handle, kClockSm, &value) == kSuccess) { r.sm_clock_mhz = value; }
            if (clock_ != nullptr && clock_(handle, kClockMemory, &value) == kSuccess) {
                r.memory_clock_mhz = value;
            }
            out.push_back(r);
        }
        return out;
    }

private:
    // NVML stays loaded for the process: a reading is a few calls, and nvmlShutdown at exit would
    // race the HTTP threads that still read.
    Sampler() {
        void* library = open_nvml_library();
        const auto init = reinterpret_cast<InitFn>(nvml_symbol(library, "nvmlInit_v2"));
        const auto by_pci =
            reinterpret_cast<ByPciFn>(nvml_symbol(library, "nvmlDeviceGetHandleByPciBusId_v2"));
        if (init == nullptr || by_pci == nullptr || init() != kSuccess) {
            close_nvml_library(library);
            return;
        }
        utilization_ = reinterpret_cast<UtilizationFn>(
            nvml_symbol(library, "nvmlDeviceGetUtilizationRates"));
        memory_      = reinterpret_cast<MemoryFn>(nvml_symbol(library, "nvmlDeviceGetMemoryInfo"));
        memory_v2_ = reinterpret_cast<MemoryV2Fn>(nvml_symbol(library, "nvmlDeviceGetMemoryInfo_v2"));
        temperature_ = reinterpret_cast<TemperatureFn>(nvml_symbol(library, "nvmlDeviceGetTemperature"));
        power_       = reinterpret_cast<UnsignedFn>(nvml_symbol(library, "nvmlDeviceGetPowerUsage"));
        power_limit_ = reinterpret_cast<UnsignedFn>(
            nvml_symbol(library, "nvmlDeviceGetEnforcedPowerLimit"));
        clock_       = reinterpret_cast<ClockFn>(nvml_symbol(library, "nvmlDeviceGetClockInfo"));
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess) {
            (void)cudaGetLastError();
            return;
        }
        for (int ordinal = 0; ordinal < count; ++ordinal) {
            char bus[32] = {};
            NvmlDevice handle = nullptr;
            if (cudaDeviceGetPCIBusId(bus, static_cast<int>(sizeof(bus)), ordinal) != cudaSuccess) {
                (void)cudaGetLastError();
                continue;
            }
            if (by_pci(bus, &handle) == kSuccess) { devices_.emplace_back(ordinal, handle); }
        }
    }

    std::vector<std::pair<int, NvmlDevice>> devices_;
    UtilizationFn utilization_ = nullptr;
    MemoryFn memory_           = nullptr;
    MemoryV2Fn memory_v2_      = nullptr;
    TemperatureFn temperature_ = nullptr;
    UnsignedFn power_          = nullptr;
    UnsignedFn power_limit_    = nullptr;
    ClockFn clock_             = nullptr;
};

} // namespace

std::vector<GpuReading> read_gpu_telemetry() { return Sampler::instance().read(); }

} // namespace ninfer
