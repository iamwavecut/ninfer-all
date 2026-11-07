// A device wait that never finishes ends the process; a long wait the device's host service keeps
// answering, host work on a watched thread and a wait on an unwatched thread do not. The device
// is held by a kernel that spins until a host-mapped flag is set, so its stream does not finish,
// with no CUDA error, until the test lets it go -- the state of the 2026-11-07 AI Farm incident.
// Each case runs in a child process (the stall ends its process), started before the parent
// touches CUDA.

#include "core/device.h"
#include "core/device_wait.h"
#include "cuda_availability.h"

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
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

__global__ void spin_until_set(const unsigned int* flag) {
    cuda::atomic_ref<const unsigned int, cuda::thread_scope_system> released(*flag);
    while (released.load(cuda::memory_order_acquire) == 0) { __nanosleep(1000); }
}

// A device stream held by spin_until_set until release().
class HeldStream {
public:
    explicit HeldStream(const ninfer::DeviceContext& device) : stream_(device.stream) {
        CUDA_CHECK(cudaHostAlloc(&flag_, sizeof(*flag_), cudaHostAllocMapped));
        *flag_               = 0;
        unsigned int* mapped = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped), flag_, 0));
        spin_until_set<<<1, 1, 0, stream_>>>(mapped);
        CUDA_CHECK(cudaGetLastError());
    }

    ~HeldStream() {
        release();
        (void)cudaStreamSynchronize(stream_);
        (void)cudaFreeHost(flag_);
    }

    HeldStream(const HeldStream&)            = delete;
    HeldStream& operator=(const HeldStream&) = delete;

    void release() { std::atomic_ref<unsigned int>(*flag_).store(1, std::memory_order_release); }

    [[nodiscard]] cudaStream_t stream() const { return stream_; }

private:
    cudaStream_t stream_;
    unsigned int* flag_ = nullptr;
};

// Child cases: each returns its exit status, unless the watch ends the process first.

// The stream never finishes: the watch must end the process past its one-second limit.
int stall() {
    ninfer::DeviceContext device(0);
    const ninfer::DeviceWaitWatch watch(device, "test worker", milliseconds(1000));
    const HeldStream held(device);
    (void)ninfer::wait_for_stream(held.stream());
    std::fprintf(stderr, "the stalled wait returned\n");
    return 3;
}

// The wait lasts three limits while a host service answers the device every 100 ms, then the
// device is let go: the wait returns and the process lives.
int answered() {
    ninfer::DeviceContext device(0);
    const ninfer::DeviceWaitWatch watch(device, "test worker", milliseconds(1000));
    HeldStream held(device);
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

// Host work on a watched thread lasts two limits, then a thread with no watch waits two limits
// for the device: neither is a stall.
int unwatched() {
    ninfer::DeviceContext device(0);
    {
        const ninfer::DeviceWaitWatch watch(device, "test worker", milliseconds(1000));
        std::this_thread::sleep_for(milliseconds(2000));
    }
    const ninfer::DeviceWaitWatch other(device, "test worker", milliseconds(1000));
    HeldStream held(device);
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
struct Outcome {
    bool exited     = false;
    int status      = -1;
    int signal      = 0;
    double seconds  = 0.0;
    std::string err = {};
};

Outcome run_child(int (*body)()) {
    int pipe_fds[2];
    if (pipe(pipe_fds) != 0) { return {}; }
    const auto started = Clock::now();
    const pid_t child  = fork();
    if (child == 0) {
        close(pipe_fds[0]);
        dup2(pipe_fds[1], STDERR_FILENO);
        _exit(body());
    }
    close(pipe_fds[1]);
    Outcome outcome;
    char buffer[512];
    ssize_t read_bytes = 0;
    while ((read_bytes = read(pipe_fds[0], buffer, sizeof(buffer))) > 0) {
        outcome.err.append(buffer, std::size_t(read_bytes));
    }
    close(pipe_fds[0]);
    int status = 0;
    waitpid(child, &status, 0);
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
    int failures = check(probe.exited && probe.status == 0, "the device probe failed", probe);

    const Outcome stalled = run_child(stall);
    failures += check(stalled.signal == SIGABRT, "a stalled wait did not end the process", stalled);
    failures += check(stalled.err.find("the test worker has waited 1 s for CUDA device 0 (a "
                                       "stream)") != std::string::npos,
                      "the stall message does not name the thread, device and wait", stalled);
    failures +=
        check(stalled.seconds < 10.0, "the stall took too long to end the process", stalled);

    const Outcome served = run_child(answered);
    failures += check(served.exited && served.status == 0,
                      "a wait the host service kept answering ended the process", served);
    failures +=
        check(served.seconds >= 3.0, "the answered wait did not last its three limits", served);

    const Outcome idle = run_child(unwatched);
    failures += check(idle.exited && idle.status == 0,
                      "host work or an unwatched wait ended the process", idle);
    return failures == 0 ? 0 : 1;
#endif
}
