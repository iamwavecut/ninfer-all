#include "core/device.h"
#include "cuda_availability.h"
#include "ops/common/device_tier.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

using ninfer::test::cuda_unavailable;

int expect_throws_device(int device_id) {
    try {
        ninfer::DeviceContext invalid(device_id);
    } catch (const std::runtime_error&) { return 0; }
    std::cerr << "DeviceContext(" << device_id << ") did not throw\n";
    return 1;
}

__global__ void store_one(int* value) { *value = 1; }

// Whether this binary's code runs on `device`, by launching a kernel there.
bool kernel_runs_on(int device) {
    CUDA_CHECK(cudaSetDevice(device));
    int* value = nullptr;
    CUDA_CHECK(cudaMalloc(&value, sizeof(int)));
    store_one<<<1, 1>>>(value);
    const cudaError_t launched = cudaGetLastError();
    const cudaError_t finished = launched == cudaSuccess ? cudaDeviceSynchronize() : launched;
    CUDA_CHECK(cudaFree(value));
    return launched == cudaSuccess && finished == cudaSuccess;
}

// The code tier each device runs: the newest tier this build carries that the device's compute
// capability reaches, since CUDA loads the newest image a card can run. The current-device
// predicates and the shape set follow it.
int check_tiers(int count) {
    namespace ops = ninfer::ops;
    using ops::DeviceTier;
    for (int id = 0; id < count; ++id) {
        if (!kernel_runs_on(id)) { continue; }
        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDeviceProperties(&props, id));
        const int capability      = props.major * 10 + props.minor;
        const DeviceTier expected = ops::kBuildBlackwell && capability >= 120
                                        ? DeviceTier::Blackwell
                                    : ops::kBuildAmpere && capability >= 80 ? DeviceTier::Ampere
                                    : ops::kBuildTuring && capability >= 75 ? DeviceTier::Turing
                                                                            : DeviceTier::Volta;
        if (ops::device_tier(id) != expected) {
            std::cerr << "CUDA device " << id << " (" << props.name << ")\n";
            return fail("a device runs another tier than its compute capability selects");
        }
        CUDA_CHECK(cudaSetDevice(id));
        if (ops::current_device_tier() != expected ||
            ops::current_device_volta() != (expected == DeviceTier::Volta) ||
            ops::current_device_turing() != (expected == DeviceTier::Turing) ||
            ops::current_device_blackwell() != (expected == DeviceTier::Blackwell) ||
            ops::turing_shapes_here() != (expected == DeviceTier::Turing)) {
            return fail("the current-device tier predicates disagree with the device's tier");
        }
    }
    return 0;
}

// Every visible device in one context: it opens exactly when the build's code runs on each of
// them, whatever their compute capabilities, and each rank reports its own device's.
int check_all_devices(int count) {
    const int ranks = std::min<int>(count, int(ninfer::kMaxRanks));
    std::vector<int> ids(std::size_t(ranks), 0);
    bool runs = true;
    for (int id = 0; id < ranks; ++id) {
        ids[std::size_t(id)] = id;
        runs                 = kernel_runs_on(id) && runs;
    }
    try {
        const ninfer::DeviceContext all{std::span<const int>(ids)};
        if (!runs) { return fail("a context opened on a device this build cannot run"); }
        for (int id = 0; id < ranks; ++id) {
            cudaDeviceProp props{};
            CUDA_CHECK(cudaGetDeviceProperties(&props, id));
            if (all.rank(std::size_t(id)).compute_capability() != props.major * 10 + props.minor) {
                return fail("a rank reports another device's compute capability");
            }
        }
    } catch (const std::invalid_argument& error) {
        if (runs) {
            std::cerr << error.what() << '\n';
            return fail("devices this build runs on were refused together");
        }
    }
    return 0;
}

int check_context(const ninfer::DeviceContext& ctx, const char* label) {
    int failures = 0;
    if (ctx.stream == nullptr) {
        std::cerr << label << " compute stream is null\n";
        ++failures;
    }
    if (ctx.transfer_stream == nullptr) {
        std::cerr << label << " load stream is null\n";
        ++failures;
    }
    if (ctx.compute_capability() <= 0) {
        std::cerr << label << " compute capability is not positive\n";
        ++failures;
    }
    if (ctx.total_vram() == 0) {
        std::cerr << label << " total_vram is zero\n";
        ++failures;
    }
    return failures;
}

// Whether device 0's primary context exists, or -1 when the driver cannot say.
int device0_context_active() {
    CUdevice device = 0;
    unsigned int flags = 0;
    int active         = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxGetState(device, &flags, &active) != CUDA_SUCCESS) {
        return -1;
    }
    return active;
}

// A new thread reads device 0 without having used it. A binding to device 1 there must leave device
// 0 alone when it ends: returning to it would create its primary context, which on a full card
// fails with cudaErrorMemoryAllocation and leaves the error for the next cudaGetLastError() (#279).
// A DeviceContext on device 1 must leave it alone too, with the stall probe's thread, which makes
// CUDA calls from its first line. A thread that did use device 0 gets it back. Runs in a process of
// its own, before anything there makes device 0 current.
int check_new_thread_binding() {
    const int before = device0_context_active();
    if (before < 0) { return fail("cuDevicePrimaryCtxGetState failed"); }
    if (before != 0) {
        std::cout << "new-thread binding not checked: device 0's context already exists\n";
        return 0;
    }
    int failures = 0;
    const auto expect = [&failures](bool held, const char* message) {
        if (!held) {
            ++failures;
            std::cerr << message << '\n';
        }
    };
    std::thread([&] {
        {
            ninfer::DeviceBinding bind(1);
            expect(cudaFree(nullptr) == cudaSuccess, "cudaFree on device 1 failed");
        }
        int current = -1;
        expect(cudaGetDevice(&current) == cudaSuccess && current == 1,
               "a new thread's binding did not keep device 1");
        expect(cudaGetLastError() == cudaSuccess, "a new thread's binding left a CUDA error");
    }).join();
    expect(device0_context_active() == 0, "a new thread's binding created device 0's context");
    {
        const ninfer::DeviceContext context(1);
        // The probe's thread starts with the context and probes after its first 250 ms poll.
        std::this_thread::sleep_for(std::chrono::seconds(1));
        expect(device0_context_active() == 0,
               "a DeviceContext on device 1 created device 0's context");
    }
    std::thread([&] {
        expect(cudaSetDevice(0) == cudaSuccess && cudaFree(nullptr) == cudaSuccess,
               "device 0 did not initialize");
        {
            ninfer::DeviceBinding bind(1);
            expect(cudaFree(nullptr) == cudaSuccess, "cudaFree on device 1 failed");
        }
        int current = -1;
        expect(cudaGetDevice(&current) == cudaSuccess && current == 0,
               "a binding did not return its thread to device 0");
    }).join();
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--invalid-sync") {
        try {
            ninfer::DeviceContext ctx(0);
        } catch (const std::invalid_argument& error) {
            return std::string_view(error.what()).find("NINFER_CUDA_SYNC") != std::string_view::npos
                       ? 0
                       : fail("invalid sync setting has no configuration diagnostic");
        }
        return fail("invalid sync setting did not fail before CUDA initialization");
    }
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 77;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--new-thread-binding") {
        if (count < 2) {
            std::cout << "SKIP: the new-thread binding check needs two CUDA devices\n";
            return 77;
        }
        return check_new_thread_binding() == 0 ? 0 : fail("new-thread binding check failed");
    }
    const unsigned int expected_flags =
        argc == 2 ? static_cast<unsigned int>(std::stoul(argv[1])) : cudaDeviceScheduleAuto;

    int failures = 0;

    ninfer::DeviceContext ctx(0);
    unsigned int actual_flags = 0;
    CUDA_CHECK(cudaGetDeviceFlags(&actual_flags));
    if ((actual_flags & cudaDeviceScheduleMask) != expected_flags) {
        return fail("CUDA did not apply the requested synchronization schedule");
    }
    if (std::string_view(ctx.sync_mode()).empty()) {
        return fail("sync_mode() returned an empty name");
    }
    if (ctx.device != 0) {
        ++failures;
        std::cerr << "ctx.device expected 0, got " << ctx.device << '\n';
    }
    failures += check_context(ctx, "ctx");
    int* device_value = nullptr;
    int* host_value   = nullptr;
    CUDA_CHECK(cudaMalloc(&device_value, sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&host_value, sizeof(int)));
    *host_value = 0;
    CUDA_CHECK(cudaMemsetAsync(device_value, 0x5a, sizeof(int), ctx.stream));
    CUDA_CHECK(
        cudaMemcpyAsync(host_value, device_value, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream));
    ctx.synchronize();
    const bool transfer_complete = *host_value == 0x5a5a5a5a;
    CUDA_CHECK(cudaFreeHost(host_value));
    CUDA_CHECK(cudaFree(device_value));
    if (!transfer_complete) {
        return fail("stream synchronization returned before transfer completed");
    }

    const cudaStream_t original_stream = ctx.stream;
    ninfer::DeviceContext moved(std::move(ctx));
    if (ctx.stream != nullptr || ctx.transfer_stream != nullptr) {
        ++failures;
        std::cerr << "move construction did not null source streams\n";
    }
    if (moved.stream != original_stream) {
        ++failures;
        std::cerr << "move construction did not transfer compute stream\n";
    }
    failures += check_context(moved, "moved");

    failures += expect_throws_device(count);

    ninfer::CudaEventTimer timer(moved);
    timer.start();
    moved.synchronize();
    const float elapsed_ms = timer.stop_ms();
    if (elapsed_ms < 0.0f) {
        ++failures;
        std::cerr << "timer elapsed time was negative\n";
    }

    // Last: they switch the calling thread's current device.
    failures += check_tiers(count);
    if (count > 1) { failures += check_all_devices(count); }

    return failures == 0 ? 0 : fail("device test failed");
}
