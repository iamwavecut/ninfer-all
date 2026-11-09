// A device that never finishes work ends the process: a wait on a thread with a watch, a wait on a
// thread without one, and, with no thread waiting at all, the probe the device's DeviceContext
// starts. A long wait the device's host service keeps answering, host work on a watched thread, a
// stopped process and NINFER_DEVICE_STALL_SECONDS=0 do not end it. The device is held by kernels
// that spin until a host-mapped flag is set, so its work does not finish, with no CUDA error, until
// the test lets it go -- the state of the 2026-11-07 AI Farm incident. Each case runs in a child
// process (the stall ends its process), started before the parent touches CUDA.

#include "core/device.h"
#include "core/device_wait.h"
#include "cuda_availability.h"

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

#ifndef _WIN32
#    include <csignal>
#    include <sys/wait.h>
#    include <unistd.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
using std::chrono::milliseconds;

constexpr int kSkip = 77;
constexpr const char* kLimitVariable = "NINFER_DEVICE_STALL_SECONDS";

// Sets the stall limit of this process before anything reads it.
void set_limit(const char* seconds) {
#ifdef _WIN32
    (void)_putenv_s(kLimitVariable, seconds);
#else
    setenv(kLimitVariable, seconds, 1);
#endif
}

__global__ void spin_until_set(const unsigned int* flag) {
    cuda::atomic_ref<const unsigned int, cuda::thread_scope_system> released(*flag);
    while (released.load(cuda::memory_order_acquire) == 0) { __nanosleep(1000); }
}

// The device's stream held by spin_until_set until release(): one block, or as many as every
// multiprocessor can hold at once, which leaves no room for another kernel's block anywhere.
class HeldDevice {
public:
    enum class Extent { one_block, every_block_slot };

    explicit HeldDevice(const ninfer::DeviceContext& device, Extent extent = Extent::one_block)
        : stream_(device.stream) {
        CUDA_CHECK(cudaHostAlloc(&flag_, sizeof(*flag_), cudaHostAllocMapped));
        *flag_               = 0;
        unsigned int* mapped = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped), flag_, 0));
        int blocks = 1;
        if (extent == Extent::every_block_slot) {
            int per_multiprocessor = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_multiprocessor,
                                                                     spin_until_set, 32, 0));
            blocks = per_multiprocessor * device.multiprocessor_count();
        }
        spin_until_set<<<blocks, 32, 0, stream_>>>(mapped);
        CUDA_CHECK(cudaGetLastError());
    }

    ~HeldDevice() {
        release();
        (void)cudaStreamSynchronize(stream_);
        (void)cudaFreeHost(flag_);
    }

    HeldDevice(const HeldDevice&)            = delete;
    HeldDevice& operator=(const HeldDevice&) = delete;

    void release() { std::atomic_ref<unsigned int>(*flag_).store(1, std::memory_order_release); }

    [[nodiscard]] cudaStream_t stream() const { return stream_; }

private:
    cudaStream_t stream_;
    unsigned int* flag_ = nullptr;
};

// Child cases: each returns its exit status, unless a watch ends the process first.

// The stream never finishes: the watch must end the process past its one-second limit.
int stall() {
    ninfer::DeviceContext device(0);
    const ninfer::DeviceWaitWatch watch(device, "test worker", milliseconds(1000));
    const HeldDevice held(device);
    (void)ninfer::wait_for_stream(held.stream());
    std::fprintf(stderr, "the stalled wait returned\n");
    return 3;
}

// The same wait on a thread without a watch, under a one-second limit from the environment.
int unnamed_stall() {
    set_limit("1");
    ninfer::DeviceContext device(0);
    const HeldDevice held(device);
    std::thread waiter([&] {
        device.bind_to_current_thread();
        (void)ninfer::wait_for_stream(held.stream());
    });
    waiter.join();
    std::fprintf(stderr, "the stalled wait returned\n");
    return 3;
}

// No thread waits for the device; its kernels hold every block slot, so the probe's kernel never
// runs. The probe must end the process past the one-second limit.
int probe_stall() {
    set_limit("1");
    ninfer::DeviceContext device(0);
    const HeldDevice held(device, HeldDevice::Extent::every_block_slot);
    std::this_thread::sleep_for(milliseconds(10000));
    std::fprintf(stderr, "the probe let a held device pass\n");
    return 3;
}

// The wait lasts three limits while a host service answers the device every 100 ms, then the
// device is let go: the wait returns and the process lives.
int answered() {
    ninfer::DeviceContext device(0);
    const ninfer::DeviceWaitWatch watch(device, "test worker", milliseconds(1000));
    HeldDevice held(device);
    std::thread service([&] {
        const auto until = Clock::now() + milliseconds(3000);
        while (Clock::now() < until) {
            ninfer::note_host_service_progress(device.device);
            std::this_thread::sleep_for(milliseconds(100));
        }
        held.release();
    });
    const cudaError_t status = ninfer::wait_for_stream(held.stream());
    service.join();
    return status == cudaSuccess ? 0 : 4;
}

// Host work on a watched thread lasts two limits, then a thread without a watch waits two limits
// of the other thread's watch, far within its own: neither is a stall.
int unwatched() {
    ninfer::DeviceContext device(0);
    {
        const ninfer::DeviceWaitWatch watch(device, "test worker", milliseconds(1000));
        std::this_thread::sleep_for(milliseconds(2000));
    }
    const ninfer::DeviceWaitWatch other(device, "test worker", milliseconds(1000));
    HeldDevice held(device);
    cudaError_t status = cudaErrorUnknown;
    std::thread waiter([&] {
        device.bind_to_current_thread();
        status = ninfer::wait_for_stream(held.stream());
    });
    std::this_thread::sleep_for(milliseconds(2000));
    held.release();
    waiter.join();
    return status == cudaSuccess ? 0 : 5;
}

#ifndef _WIN32
// A two-second wait stops the whole process for three seconds (the parent continues it), then
// lasts one more second: the time stopped is not a stall.
int stopped() {
    ninfer::DeviceContext device(0);
    const ninfer::DeviceWaitWatch watch(device, "test worker", milliseconds(2000));
    HeldDevice held(device);
    std::thread stopper([&] {
        std::this_thread::sleep_for(milliseconds(300));
        raise(SIGSTOP);
        std::this_thread::sleep_for(milliseconds(1000));
        held.release();
    });
    const cudaError_t status = ninfer::wait_for_stream(held.stream());
    stopper.join();
    return status == cudaSuccess ? 0 : 7;
}
#endif

// Zero turns both watches off: a wait for, and kernels on, every block slot of the device for
// three seconds end nothing.
int switched_off() {
    set_limit("0");
    ninfer::DeviceContext device(0);
    const ninfer::DeviceWaitWatch watch(device, "test worker");
    HeldDevice held(device, HeldDevice::Extent::every_block_slot);
    std::thread releaser([&] {
        std::this_thread::sleep_for(milliseconds(3000));
        held.release();
    });
    const cudaError_t status = ninfer::wait_for_stream(held.stream());
    releaser.join();
    return status == cudaSuccess ? 0 : 8;
}

// A limit that is not a whole number of seconds refuses the DeviceContext.
int refused_limit() {
    set_limit("two");
    try {
        ninfer::DeviceContext device(0);
    } catch (const std::invalid_argument& error) {
        return std::strstr(error.what(), kLimitVariable) != nullptr ? 0 : 9;
    }
    return 10;
}

#ifndef _WIN32
struct Outcome {
    bool exited     = false;
    int status      = -1;
    int signal      = 0;
    double seconds  = 0.0;
    std::string err = {};
};

// Runs `body` in a child process. With `stop_for` set, the child is continued that long after it
// stops itself.
Outcome run_child(int (*body)(), milliseconds stop_for = milliseconds(0)) {
    int pipe_fds[2];
    if (pipe(pipe_fds) != 0) { return {}; }
    const auto started = Clock::now();
    const pid_t child  = fork();
    if (child == 0) {
        close(pipe_fds[0]);
        dup2(pipe_fds[1], STDERR_FILENO);
        unsetenv(kLimitVariable);
        _exit(body());
    }
    close(pipe_fds[1]);
    Outcome outcome;
    int status    = 0;
    bool finished = false;
    if (stop_for.count() > 0 && waitpid(child, &status, WUNTRACED) == child) {
        if (WIFSTOPPED(status)) {
            std::this_thread::sleep_for(stop_for);
            kill(child, SIGCONT);
        } else {
            finished = true;
        }
    }
    char buffer[512];
    ssize_t read_bytes = 0;
    while ((read_bytes = read(pipe_fds[0], buffer, sizeof(buffer))) > 0) {
        outcome.err.append(buffer, std::size_t(read_bytes));
    }
    close(pipe_fds[0]);
    if (!finished) { waitpid(child, &status, 0); }
    outcome.seconds = std::chrono::duration<double>(Clock::now() - started).count();
    if (WIFEXITED(status)) {
        outcome.exited = true;
        outcome.status = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        outcome.signal = WTERMSIG(status);
    }
    return outcome;
}

// A child the watch fails to end would hang the test; CTest's timeout catches that.
int check(bool condition, const char* what, const Outcome& outcome) {
    if (condition) { return 0; }
    std::fprintf(stderr, "%s (exited %d, status %d, signal %d, %.1f s)\n%s\n", what,
                 int(outcome.exited), outcome.status, outcome.signal, outcome.seconds,
                 outcome.err.c_str());
    return 1;
}

bool says(const Outcome& outcome, const char* text) {
    return outcome.err.find(text) != std::string::npos;
}
#endif

} // namespace

int main() {
#ifdef _WIN32
    std::fprintf(stderr, "skip: the stall cases need fork()\n");
    return kSkip;
#else
    // The parent stays free of CUDA: a child forked from a process that initialized it cannot use
    // the device. The first child answers whether there is one.
    const Outcome probe = run_child([] {
        int count               = 0;
        const cudaError_t error = cudaGetDeviceCount(&count);
        if (ninfer::test::cuda_unavailable(error) || (error == cudaSuccess && count == 0)) {
            return kSkip;
        }
        return error == cudaSuccess ? 0 : 6;
    });
    if (probe.exited && probe.status == kSkip) {
        std::fprintf(stderr, "skip: no CUDA device\n");
        return kSkip;
    }
    int failures = check(probe.exited && probe.status == 0, "the device query failed", probe);

    const Outcome stalled = run_child(stall);
    failures += check(stalled.signal == SIGABRT, "a stalled wait did not end the process", stalled);
    failures += check(says(stalled, "ninfer: the test worker (thread ") &&
                          says(stalled, ") has waited 1 s for CUDA device 0 (a stream)"),
                      "the stall message does not name the thread, device and wait", stalled);
    failures +=
        check(stalled.seconds < 10.0, "the stall took too long to end the process", stalled);

    const Outcome unnamed = run_child(unnamed_stall);
    failures +=
        check(unnamed.signal == SIGABRT,
              "a stalled wait on a thread without a watch did not end the process", unnamed);
    failures += check(says(unnamed, "ninfer: thread ") &&
                          says(unnamed, " has waited 1 s for CUDA device 0 (a stream)"),
                      "the stall message does not name the thread id, device and wait", unnamed);

    const Outcome probed = run_child(probe_stall);
    failures += check(probed.signal == SIGABRT,
                      "a held device that no thread waits for did not end the process", probed);
    failures += check(says(probed, "ninfer: CUDA device 0 has not finished a probe"),
                      "the probe message does not name the device", probed);
    failures += check(probed.seconds < 9.0, "the probe took too long to end the process", probed);

    const Outcome served = run_child(answered);
    failures += check(served.exited && served.status == 0,
                      "a wait the host service kept answering ended the process", served);
    failures +=
        check(served.seconds >= 3.0, "the answered wait did not last its three limits", served);

    const Outcome idle = run_child(unwatched);
    failures += check(idle.exited && idle.status == 0,
                      "host work or a wait on a thread without a watch ended the process", idle);

    const Outcome paused = run_child(stopped, milliseconds(3000));
    failures += check(paused.exited && paused.status == 0,
                      "the time a stopped process stood still counted as a stall", paused);
    failures += check(paused.seconds >= 4.0, "the stopped process was not stopped", paused);

    const Outcome off = run_child(switched_off);
    failures += check(off.exited && off.status == 0,
                      "a zero limit did not turn the watches off", off);

    const Outcome refused = run_child(refused_limit);
    failures += check(refused.exited && refused.status == 0,
                      "a limit that is not a number of seconds was not refused", refused);
    return failures == 0 ? 0 : 1;
#endif
}
