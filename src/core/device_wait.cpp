#include "core/device_wait.h"

#include "core/device.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ninfer {

struct DeviceWaitState {
    const char* thread = "";
    std::vector<int> devices;
    std::int64_t limit_ns = 0;
    // Steady-clock nanoseconds when the thread's outermost watched wait began; 0 between waits.
    std::atomic<std::int64_t> waiting_since{0};
    std::atomic<const char*> kind{""};
    // Watched waits the thread is inside (a wait may call another); read by that thread only.
    int depth = 0;
};

namespace {

using Clock = std::chrono::steady_clock;

// The watch checks every bound thread this often; a stall is reported this much past its limit.
constexpr auto kPoll = std::chrono::milliseconds(250);
// CUDA device ordinals whose host-service answers are counted; a larger one's are not.
constexpr std::size_t kTrackedDevices = 64;

std::int64_t now_ns() noexcept {
    const auto ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch());
    return std::max<std::int64_t>(1, ns.count());
}

struct Registry {
    // When a host service last answered each device; 0 before its first answer.
    std::array<std::atomic<std::int64_t>, kTrackedDevices> answered{};
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<const DeviceWaitState*> states;
    bool monitoring = false;
};

// Never destroyed: the monitoring thread runs until the process ends, past static destruction.
Registry& registry() {
    static auto* const instance = new Registry();
    return *instance;
}

thread_local DeviceWaitState* t_watched = nullptr;

class WaitScope {
public:
    explicit WaitScope(const char* kind) noexcept : state_(t_watched) {
        if (state_ != nullptr && state_->depth++ == 0) {
            state_->kind.store(kind, std::memory_order_relaxed);
            state_->waiting_since.store(now_ns(), std::memory_order_release);
        }
    }

    ~WaitScope() {
        if (state_ != nullptr && --state_->depth == 0) {
            state_->waiting_since.store(0, std::memory_order_release);
        }
    }

    WaitScope(const WaitScope&)            = delete;
    WaitScope& operator=(const WaitScope&) = delete;

private:
    DeviceWaitState* state_;
};

[[noreturn]] void end_process(const DeviceWaitState& state, std::int64_t waited_ns) {
    std::string devices;
    for (const int device : state.devices) {
        devices += (devices.empty() ? "" : ", ") + std::to_string(device);
    }
    std::fprintf(stderr,
                 "ninfer: the %s has waited %lld s for CUDA device %s (%s) while no host service "
                 "answered it: the device stopped finishing work, which CUDA reports as no error. "
                 "Ending the process; a new one recovers the device.\n",
                 state.thread, static_cast<long long>(waited_ns / 1'000'000'000), devices.c_str(),
                 state.kind.load(std::memory_order_relaxed));
    std::fflush(stderr);
    std::abort();
}

void monitor(Registry& watched) {
    std::unique_lock lock(watched.mutex);
    for (;;) {
        watched.changed.wait(lock, [&] { return !watched.states.empty(); });
        watched.changed.wait_for(lock, kPoll);
        const std::int64_t now = now_ns();
        for (const DeviceWaitState* state : watched.states) {
            const std::int64_t since = state->waiting_since.load(std::memory_order_acquire);
            if (since == 0) { continue; }
            // An answer the device received after the wait began restarts the quiet period.
            std::int64_t quiet = since;
            for (const int device : state->devices) {
                if (device >= 0 && std::size_t(device) < kTrackedDevices) {
                    quiet =
                        std::max(quiet, watched.answered[device].load(std::memory_order_acquire));
                }
            }
            if (now - quiet >= state->limit_ns) { end_process(*state, now - since); }
        }
    }
}

} // namespace

cudaError_t wait_for_stream(cudaStream_t stream) noexcept {
    const WaitScope scope("a stream");
    return cudaStreamSynchronize(stream);
}

cudaError_t wait_for_event(cudaEvent_t event) noexcept {
    const WaitScope scope("an event");
    return cudaEventSynchronize(event);
}

cudaError_t wait_for_device() noexcept {
    const WaitScope scope("the device");
    return cudaDeviceSynchronize();
}

cudaError_t copy_and_wait(void* destination, const void* source, std::size_t bytes,
                          cudaMemcpyKind kind) noexcept {
    const WaitScope scope("a copy");
    return cudaMemcpy(destination, source, bytes, kind);
}

cudaError_t fill_and_wait(void* destination, int value, std::size_t bytes) noexcept {
    const WaitScope scope("a fill");
    return cudaMemset(destination, value, bytes);
}

void note_host_service_progress(int device) noexcept {
    if (device < 0 || std::size_t(device) >= kTrackedDevices) { return; }
    registry().answered[device].store(now_ns(), std::memory_order_release);
}

DeviceWaitWatch::DeviceWaitWatch(const DeviceContext& context, const char* thread,
                                 std::chrono::milliseconds limit) {
    auto state      = std::make_unique<DeviceWaitState>();
    state->thread   = thread;
    state->limit_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(limit).count();
    for (std::size_t rank = 0; rank < context.size(); ++rank) {
        state->devices.push_back(context.rank(rank).device);
    }
    Registry& watched = registry();
    {
        std::lock_guard lock(watched.mutex);
        if (!watched.monitoring) {
            std::thread([&watched] { monitor(watched); }).detach();
            watched.monitoring = true;
        }
        watched.states.push_back(state.get());
    }
    watched.changed.notify_all();
    state_    = state.release();
    outer_    = t_watched;
    t_watched = state_;
}

DeviceWaitWatch::~DeviceWaitWatch() {
    t_watched         = outer_;
    Registry& watched = registry();
    {
        std::lock_guard lock(watched.mutex);
        std::erase(watched.states, state_);
    }
    delete state_;
}

} // namespace ninfer
