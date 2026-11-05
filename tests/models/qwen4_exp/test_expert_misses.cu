// The expert-miss service (qwen4_exp ExpertMisses) serves a call whose request the device posts
// after the call's keep-alive has lapsed and the service has gone to sleep: eagerly, the host
// waiting for the call (a prompt span's narrow chunk behind the span's attention and wide calls),
// and replayed from a CUDA graph, only the device waiting. Without and with a CPU share. Every call
// routes one token to ten uncached Q8_0 experts at Flash-Next's shapes; the service stages them (or
// the CPU computes its share) and releases the call, which must return within seconds. On Linux an
// idle service and its CPU workers sleep: the process then uses almost no CPU time.

#include "models/qwen4_exp/expert_misses.h"

#include "core/arena.h"
#include "core/device.h"

#include "cuda_availability.h"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef __linux__
#    include <sys/resource.h>
#endif

namespace {

using namespace ninfer;
using namespace ninfer::models::qwen4_exp;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

constexpr int kHidden = 2560, kWidth = 640, kTop = 10, kExperts = 16;
// The device stays busy this long before a call posts its request: past the call's keep-alive.
constexpr auto kLate = kMissKeepAlive + 300ms;
// A call that has not returned by then has lost its request.
constexpr auto kHang = 20s;

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (condition) { return; }
    std::cerr << "expectation failed: " << label << '\n';
    ++failures;
}

// Holds the device for `nanoseconds` of its global timer.
__global__ void busy_kernel(unsigned long long nanoseconds) {
    unsigned long long start = 0, now = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do {
        __nanosleep(100000);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (now - start < nanoseconds);
}

void busy(cudaStream_t stream) {
    busy_kernel<<<1, 1, 0, stream>>>(
        static_cast<unsigned long long>(std::chrono::nanoseconds(kLate).count()));
    CUDA_CHECK(cudaGetLastError());
}

// Ends the process when a step does not finish in time: a lost request waits forever, on the host
// or on the device.
class Watchdog {
public:
    explicit Watchdog(std::string step) : step_(std::move(step)), thread_([this] { watch(); }) {}
    ~Watchdog() {
        {
            std::lock_guard lock(mutex_);
            done_ = true;
        }
        done_changed_.notify_all();
        thread_.join();
    }
    Watchdog(const Watchdog&)            = delete;
    Watchdog& operator=(const Watchdog&) = delete;

private:
    void watch() {
        std::unique_lock lock(mutex_);
        if (!done_changed_.wait_for(lock, kHang, [this] { return done_; })) {
            std::cerr << "FAIL: " << step_ << " did not return within " << kHang.count()
                      << " s: the service lost the request\n";
            std::_Exit(1);
        }
    }

    std::string step_;
    std::mutex mutex_;
    std::condition_variable done_changed_;
    bool done_ = false;
    std::thread thread_; // last: the members it reads exist before it starts
};

// One layer of 16 Q8_0 experts at Flash-Next's shapes. Every byte is zero, so the CPU's products
// are zero; every expert shares one matrix per projection. The gate table names no cached storage,
// so all routed experts are missing.
struct Fixture {
    std::array<std::unique_ptr<PinnedHostBuffer>, 3> matrices;
    DeviceBuffer gate_table, ids, input;
    std::unique_ptr<ExpertMisses> misses;

    Fixture(DeviceContext& device, double cpu_share) {
        MissLayer layer;
        layer.rank    = 0;
        layer.experts = kExperts;
        const std::array<int, 3> rows = {kWidth, kWidth, kHidden}, ks = {kHidden, kHidden, kWidth};
        for (std::size_t p = 0; p < 3; ++p) {
            auto& projection        = layer.projections[p];
            projection.format       = QType::GGUF_Q8_0; // 34 bytes a block of 32 values
            projection.row_bytes    = std::int64_t(ks[p] / 32) * 34;
            projection.rows         = rows[p];
            projection.k            = ks[p];
            projection.expert_bytes = projection.row_bytes * rows[p];
            matrices[p] = std::make_unique<PinnedHostBuffer>(std::size_t(projection.expert_bytes));
            std::memset(matrices[p]->data(), 0, matrices[p]->size());
            projection.host.assign(kExperts, matrices[p]->data());
        }
        gate_table = DeviceBuffer(kExperts * sizeof(void*));
        gate_table.fill(0);
        layer.gate_table = static_cast<const void* const*>(gate_table.p);
        std::array<std::int32_t, kTop> routed{};
        for (int i = 0; i < kTop; ++i) { routed[std::size_t(i)] = i; }
        ids = DeviceBuffer(sizeof(routed));
        ids.copy_from_host(routed.data(), sizeof(routed));
        input = DeviceBuffer(kHidden * 2);
        input.fill(0);
        MissOptions options;
        options.cpu_share   = cpu_share;
        options.cpu_threads = 2;
        std::vector<MissLayer> layers;
        layers.push_back(std::move(layer));
        misses = std::make_unique<ExpertMisses>(device, std::move(layers),
                                                std::vector<std::filesystem::path>{}, options,
                                                nullptr);
    }

    ExpertMisses::Call begin(cudaStream_t stream) {
        const Tensor routed(ids.p, DType::I32, {kTop, 1});
        const Tensor m(input.p, DType::BF16, {kHidden, 1});
        return misses->begin(0, routed, m, 1, ops::GgufMoeWeights{}, stream);
    }
};

// What the service did for the last call: the second stage's missing pairs (the GPU's share) and
// the count of CPU products.
void check_served(const Fixture& f, const ExpertMisses::Call& call, const MissStats& before,
                  bool cpu, const std::string& label) {
    const MissStats after = f.misses->stats();
    const auto computed   = after.computed - before.computed;
    const auto staged     = after.staged + after.mapped - before.staged - before.mapped;
    expect(after.requests == before.requests + 1, label + ": the request was served");
    expect(computed == (cpu ? 5U : 0U), label + ": the CPU computed its share");
    expect(computed + staged == std::uint64_t(kTop), label + ": every missing expert was served");
    std::array<std::int32_t, kTop> missing{};
    CUDA_CHECK(cudaMemcpy(missing.data(), call.missing_ids.data, sizeof(missing),
                          cudaMemcpyDeviceToHost));
    std::uint64_t on_gpu = 0;
    for (int p = 0; p < kTop; ++p) {
        const std::int32_t id = missing[std::size_t(p)];
        expect(id == -1 || id == p, label + ": a missing pair keeps its expert");
        on_gpu += id == p;
    }
    expect(on_gpu == staged, label + ": the second stage takes the GPU's share");
    expect(*call.count == std::int32_t(computed), label + ": one CPU product a computed pair");
    f.misses->raise_failure();
}

// The host waits for the call, whose request the device posts after the keep-alive begin() armed.
void eager_late(Fixture& f, cudaStream_t stream, bool cpu, const std::string& label) {
    const MissStats before = f.misses->stats();
    busy(stream);
    const auto started = Clock::now();
    const auto call    = f.begin(stream);
    {
        Watchdog guard(label + ": an eager await");
        f.misses->await(0, stream);
    }
    const auto waited = Clock::now() - started;
    expect(waited > kMissKeepAlive, label + ": the request came after the keep-alive");
    expect(waited < kLate + 1s, label + ": the service took the request at once");
    check_served(f, call, before, cpu, label);
}

// Only the device waits: a replayed graph posts the request after the keep-alive its launch armed.
void replay_late(Fixture& f, cudaStream_t stream, bool cpu, const std::string& label) {
    cudaGraph_t graph     = nullptr;
    cudaGraphExec_t exec  = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    busy(stream);
    const auto call = f.begin(stream);
    f.misses->await(0, stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    const MissStats before = f.misses->stats();
    // As a pass does before it launches its graphs.
    f.misses->keep_alive();
    const auto started = Clock::now();
    CUDA_CHECK(cudaGraphLaunch(exec, stream));
    {
        Watchdog guard(label + ": a replayed wait");
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const auto waited = Clock::now() - started;
    expect(waited > kMissKeepAlive, label + ": the request came after the keep-alive");
    expect(waited < kLate + 1s, label + ": the sleeping service found the request");
    check_served(f, call, before, cpu, label);
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
}

#ifdef __linux__
double process_cpu_seconds() {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return double(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) +
           1e-6 * double(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec);
}

// Past the keep-alive the service and its workers sleep, looking for requests now and then.
void idle_sleeps(const std::string& label) {
    std::this_thread::sleep_for(kMissKeepAlive + 200ms);
    const double before = process_cpu_seconds();
    std::this_thread::sleep_for(1s);
    const double used = process_cpu_seconds() - before;
    expect(used < 0.1, label + ": an idle service sleeps (" + std::to_string(used) +
                           " s of CPU time in a second)");
}
#endif

int run() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (ninfer::test::cuda_unavailable(status) || (status == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(status);
    DeviceContext device;
    RankBinding bind(device, 0);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (const bool cpu : {false, true}) {
        const std::string label = cpu ? "CPU share" : "GPU only";
        Fixture f(device, cpu ? 0.5 : 0.0);
        expect(f.misses->cpu_enabled() == cpu, label + ": the CPU share as configured");
        eager_late(f, stream, cpu, label + ", eager");
        replay_late(f, stream, cpu, label + ", replayed");
#ifdef __linux__
        idle_sleeps(label);
#endif
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    if (failures != 0) { return 1; }
    std::cout << "PASS qwen4_exp expert misses\n";
    return 0;
}

} // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
