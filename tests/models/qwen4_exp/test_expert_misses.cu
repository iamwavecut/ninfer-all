// The expert-miss service (qwen4_exp ExpertMisses) serves a call whose request the device posts
// after the call's keep-alive has lapsed and the service has gone to sleep: eagerly, the host
// waiting for the call (a prompt span's narrow chunk behind the span's attention and wide calls),
// and replayed from a CUDA graph, only the device waiting. Without and with a CPU share. Every call
// routes one token to ten uncached Q8_0 experts at Flash-Next's shapes; the service stages them (or
// the CPU computes its share) and releases the call, which must return within seconds. On Linux an
// idle service and its CPU workers sleep: the process then uses almost no CPU time.
//
// A wide call (more than kNarrowMissTokens tokens) brings the missing experts the GPU runs into
// device memory in rounds that alternate between the staging's halves, with zeros after each down.
// With a CPU share, the CPU takes its share of the missing experts with at most
// ops::kGgufMoeVectorPairs pairs (the vector products' experts, which its products reproduce) and
// none of the others (the matrix kernel's): a call whose experts all have more takes no CPU share.
// A narrow call after it is served as before. Disk experts take the same wide call (and the narrow
// one), read from a file whose first expert starts inside a 4 KiB block of the second 1 MiB stripe:
// through host staging, striped over a drive mirror, and by the direct path's host window, which
// runs every step of direct reads but the kernel modules' DMA; each staged projection must hold
// its expert's bytes.

#include "models/qwen4_exp/expert_misses.h"

#include "core/device.h"
#include "core/device_buffer.h"
#include "core/pinned_memory.h"
#include "core/qtype.h"
#include "core/tensor.h"

#include "cuda_availability.h"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
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

// Where a fixture's experts are: in pinned host memory, or in a file (disk), read through host
// staging or the direct path's host window, with or without a drive mirror.
struct DiskMode {
    bool disk   = false;
    bool mirror = false;
    bool direct = false;
};

// One layer of 16 Q8_0 experts at Flash-Next's shapes, each its own bytes (every byte of expert
// e's projection k is 1 + 3e + k), behind a staging of two experts a half: a wide call of 16
// tokens routes all 16, none cached. Uniform: token j to experts j..j + 9 mod 16, ten pairs each.
// Mixed: every token to experts 0..5 (16 pairs each) and four of experts 6..15 in turn (six or
// seven pairs each).
struct WideFixture {
    static constexpr int kTokens = 16;
    static constexpr int kHot    = 6; // mixed: the experts every token routes
    [[nodiscard]] static std::int32_t route(bool mixed, int token, int slot) {
        if (!mixed) { return (token + slot) % kExperts; }
        return slot < kHot ? slot : kHot + (token * (kTop - kHot) + slot - kHot) % (kExperts - kHot);
    }
    std::vector<std::unique_ptr<PinnedHostBuffer>> matrices; // [expert][projection]
    // Disk experts: the file, the mirror directory and each projection's locations.
    std::vector<std::filesystem::path> paths;
    std::filesystem::path mirror;
    std::array<std::vector<ExpertLocation>, 3> locations;
    DeviceBuffer gate_table, ids, input;
    std::unique_ptr<ExpertMisses> misses;
    std::array<std::int64_t, 3> bytes{};

    WideFixture(DeviceContext& device, double cpu_share, bool mixed, const DiskMode& disk = {}) {
        MissLayer layer;
        layer.rank    = 0;
        layer.experts = kExperts;
        const std::array<int, 3> rows = {kWidth, kWidth, kHidden}, ks = {kHidden, kHidden, kWidth};
        for (std::size_t p = 0; p < 3; ++p) {
            auto& projection        = layer.projections[p];
            projection.format       = QType::GGUF_Q8_0;
            projection.row_bytes    = std::int64_t(ks[p] / 32) * 34;
            projection.rows         = rows[p];
            projection.k            = ks[p];
            projection.expert_bytes = projection.row_bytes * rows[p];
            bytes[p]                = projection.expert_bytes;
        }
        if (disk.disk) {
            write_experts(layer, disk);
        } else {
            for (int e = 0; e < kExperts; ++e) {
                for (std::size_t p = 0; p < 3; ++p) {
                    auto matrix = std::make_unique<PinnedHostBuffer>(std::size_t(bytes[p]));
                    std::memset(matrix->data(), 1 + 3 * e + int(p), matrix->size());
                    layer.projections[p].host.push_back(matrix->data());
                    matrices.push_back(std::move(matrix));
                }
            }
        }
        gate_table = DeviceBuffer(kExperts * sizeof(void*));
        gate_table.fill(0);
        layer.gate_table = static_cast<const void* const*>(gate_table.p);
        std::vector<std::int32_t> routed(std::size_t(kTop) * kTokens);
        for (int j = 0; j < kTokens; ++j) {
            for (int i = 0; i < kTop; ++i) { routed[std::size_t(j * kTop + i)] = route(mixed, j, i); }
        }
        ids = DeviceBuffer(routed.size() * 4);
        ids.copy_from_host(routed.data(), routed.size() * 4);
        input = DeviceBuffer(std::size_t(kHidden) * 2 * kTokens);
        input.fill(0);
        MissOptions options;
        options.token_capacity = kTokens;
        options.cpu_share      = cpu_share;
        options.cpu_threads    = 2;
        options.staging_bytes  = 0; // the least: two of the largest experts a half
        if (disk.mirror) { options.disk.mirror = mirror; }
        options.disk.direct      = disk.direct ? ExpertDirect::On : ExpertDirect::Off;
        options.disk.host_window = disk.direct;
        std::vector<MissLayer> layers;
        layers.push_back(std::move(layer));
        misses = std::make_unique<ExpertMisses>(device, std::move(layers), paths, options, nullptr);
    }

    ~WideFixture() {
        misses.reset();
        std::error_code error;
        for (const auto& path : paths) { std::filesystem::remove(path, error); }
        if (!mirror.empty()) { std::filesystem::remove_all(mirror, error); }
    }

    WideFixture(const WideFixture&)            = delete;
    WideFixture& operator=(const WideFixture&) = delete;

    // The experts in one file, in order, after 1 MiB + 123 bytes of zeros, every byte of expert
    // e's projection k 1 + 3e + k; with a mirror, a copy in a directory beside it.
    void write_experts(MissLayer& layer, const DiskMode& disk) {
        const auto stem = std::filesystem::current_path() /
                          ("ninfer_misses_" + std::to_string(std::random_device{}()));
        paths.push_back(stem.string() + ".experts");
        std::uint64_t offset = (1ULL << 20) + 123;
        {
            std::ofstream file(paths[0], std::ios::binary);
            const std::vector<char> zeros(std::size_t(offset), 0);
            file.write(zeros.data(), std::streamsize(zeros.size()));
            for (int e = 0; e < kExperts; ++e) {
                for (std::size_t p = 0; p < 3; ++p) {
                    const std::vector<char> data(std::size_t(bytes[p]), char(1 + 3 * e + int(p)));
                    file.write(data.data(), std::streamsize(data.size()));
                    ExpertLocation location;
                    location.runs[0]   = {0, offset, std::uint64_t(bytes[p])};
                    location.bytes     = std::uint64_t(bytes[p]);
                    location.format    = layer.projections[p].format;
                    location.row_bytes = layer.projections[p].row_bytes;
                    location.rows      = layer.projections[p].rows;
                    locations[p].push_back(location);
                    offset += std::uint64_t(bytes[p]);
                }
            }
            if (!file) { throw std::runtime_error("cannot write the disk experts"); }
        }
        for (std::size_t p = 0; p < 3; ++p) { layer.projections[p].located = locations[p]; }
        if (disk.mirror) {
            mirror = stem.string() + ".mirror";
            std::filesystem::create_directory(mirror);
            std::filesystem::copy_file(paths[0], mirror / paths[0].filename());
        }
    }
};

void wide_call(DeviceContext& device, cudaStream_t stream, double cpu_share, bool mixed,
               const std::string& label, const DiskMode& disk = {}) {
    WideFixture f(device, cpu_share, mixed, disk);
    const int tokens = WideFixture::kTokens;
    const MissStats before = f.misses->stats();
    const Tensor routed(f.ids.p, DType::I32, {kTop, tokens});
    const Tensor m(f.input.p, DType::BF16, {kHidden, tokens});
    const auto call = f.misses->begin(0, routed, m, tokens, ops::GgufMoeWeights{}, stream);
    expect(call.missing_banks.device_resident, label + ": the missing banks are device banks");
    expect(call.max_products == 0, label + ": products are counted by stage_wide");
    std::array<int, kExperts> uses{};
    for (int j = 0; j < tokens; ++j) {
        for (int i = 0; i < kTop; ++i) { ++uses[std::size_t(WideFixture::route(mixed, j, i))]; }
    }
    int few = 0;
    for (const int n : uses) { few += n <= ops::kGgufMoeVectorPairs; }
    expect(few == (mixed ? kExperts - WideFixture::kHot : 0), label + ": the fixture's few-pair experts");
    // The fixed share: half of the few-pair experts, rounded.
    const int cpu_experts = cpu_share > 0 ? int(std::lround(few * cpu_share)) : 0;
    const int gpu_experts = kExperts - cpu_experts;
    std::vector<std::int32_t> cached(std::size_t(kTop) * tokens);
    std::vector<int> served(std::size_t(kTop) * tokens, 0);
    std::array<int, kExperts> rounds_of{};
    int rounds = 0;
    std::int32_t computed = 0;
    {
        Watchdog guard(label + ": the rounds");
        computed = f.misses->stage_wide(0, stream, [&](const Tensor& missing_ids) {
            CUDA_CHECK(cudaStreamSynchronize(stream)); // the round's copies have landed
            std::vector<std::int32_t> round(cached.size());
            CUDA_CHECK(cudaMemcpy(round.data(), missing_ids.data, round.size() * 4,
                                  cudaMemcpyDeviceToHost));
            std::array<bool, kExperts> in_round{};
            for (std::size_t p = 0; p < round.size(); ++p) {
                const std::int32_t e = round[p];
                if (e < 0) { continue; }
                expect(e == WideFixture::route(mixed, int(p / kTop), int(p % kTop)),
                       label + ": a round's pair keeps its expert");
                ++served[p];
                in_round[std::size_t(e)] = true;
            }
            for (int e = 0; e < kExperts; ++e) {
                if (!in_round[std::size_t(e)]) { continue; }
                ++rounds_of[std::size_t(e)];
                for (int k = 0; k < 3; ++k) {
                    const auto* tables = k == 0   ? call.missing_banks.gate.experts
                                         : k == 1 ? call.missing_banks.up.experts
                                                  : call.missing_banks.down.experts;
                    const void* at     = tables[e];
                    cudaPointerAttributes attributes{};
                    CUDA_CHECK(cudaPointerGetAttributes(&attributes, at));
                    expect(attributes.type == cudaMemoryTypeDevice,
                           label + ": a staged expert sits in device memory");
                    const auto expected = static_cast<unsigned char>(1 + 3 * e + k);
                    const std::size_t size = std::size_t(f.bytes[std::size_t(k)]);
                    std::array<unsigned char, 2> ends{};
                    CUDA_CHECK(cudaMemcpy(&ends[0], at, 1, cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(&ends[1], static_cast<const unsigned char*>(at) + size - 1,
                                          1, cudaMemcpyDeviceToHost));
                    expect(ends[0] == expected && ends[1] == expected,
                           label + ": a staged projection holds its expert's bytes");
                    if (k == 2) {
                        std::array<unsigned char, 256> tail{};
                        CUDA_CHECK(cudaMemcpy(tail.data(),
                                              static_cast<const unsigned char*>(at) + size,
                                              tail.size(), cudaMemcpyDeviceToHost));
                        bool zero = true;
                        for (const auto byte : tail) { zero = zero && byte == 0; }
                        expect(zero, label + ": zeros follow a staged down");
                    }
                }
            }
            ++rounds;
        });
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(cached.data(), call.cached_ids.data, cached.size() * 4,
                          cudaMemcpyDeviceToHost));
    // The CPU's pairs: those of its experts, each listed once beside its product.
    std::array<bool, kExperts> on_cpu{};
    expect(computed == (cpu_experts == 0 ? 0 : *call.count), label + ": the products' count");
    for (std::int32_t i = 0; i < computed; ++i) {
        const std::int32_t p = call.pairs[i];
        expect(p >= 0 && p < kTop * tokens, label + ": a product names a pair");
        if (p < 0 || p >= kTop * tokens) { continue; }
        const std::int32_t e = WideFixture::route(mixed, p / kTop, p % kTop);
        expect(uses[std::size_t(e)] <= ops::kGgufMoeVectorPairs,
               label + ": the CPU computes only few-pair experts");
        on_cpu[std::size_t(e)] = true;
        ++served[std::size_t(p)];
    }
    int cpu_count = 0;
    for (const bool b : on_cpu) { cpu_count += b; }
    expect(cpu_count == cpu_experts, label + ": the CPU's share, " + std::to_string(cpu_count) +
                                         " experts");
    for (std::size_t p = 0; p < served.size(); ++p) {
        expect(cached[p] == -1, label + ": no pair is cached");
        expect(served[p] == 1, label + ": every pair runs in exactly one round or on the CPU");
    }
    for (int e = 0; e < kExperts; ++e) {
        expect(rounds_of[std::size_t(e)] == (on_cpu[std::size_t(e)] ? 0 : 1),
               label + ": a GPU expert takes one round");
    }
    expect(rounds == (gpu_experts + 1) / 2, label + ": two experts a round, " +
                                                std::to_string(rounds) + " rounds");
    const MissStats after = f.misses->stats();
    expect(after.requests == before.requests + 1, label + ": the call counts as a request");
    expect(after.staged == before.staged + std::uint64_t(gpu_experts),
           label + ": the GPU's missing experts were staged");
    expect(after.mapped == before.mapped, label + ": nothing read across the bus");
    expect(after.computed == before.computed + std::uint64_t(cpu_experts),
           label + ": the CPU computed its share");
    f.misses->raise_failure();

    // A narrow call after the wide one: token 0's ten experts, served by the service.
    const Tensor one(f.ids.p, DType::I32, {kTop, 1});
    const Tensor one_m(f.input.p, DType::BF16, {kHidden, 1});
    const MissStats narrow_before = f.misses->stats();
    const auto narrow = f.misses->begin(0, one, one_m, 1, ops::GgufMoeWeights{}, stream);
    {
        Watchdog guard(label + ": a narrow call after a wide one");
        f.misses->await(0, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const MissStats narrow_after = f.misses->stats();
    expect(!narrow.missing_banks.device_resident, label + ": a narrow call reads its banks as mapped");
    expect(narrow_after.requests == narrow_before.requests + 1, label + ": the narrow call was served");
    expect(narrow_after.computed - narrow_before.computed == (cpu_share > 0 ? 5U : 0U),
           label + ": the CPU takes its share of the narrow call");
    f.misses->raise_failure();
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
        wide_call(device, stream, cpu ? 0.5 : 0.0, false, label + ", wide");
        wide_call(device, stream, cpu ? 0.5 : 0.0, true, label + ", wide with few-pair experts");
    }
    // Disk experts: a wide call of disk experts runs every missing expert on the GPU.
    for (const DiskMode mode : {DiskMode{true, false, false}, DiskMode{true, true, false},
                                DiskMode{true, false, true}, DiskMode{true, true, true}}) {
        const std::string label = std::string("disk") + (mode.mirror ? ", mirror" : "") +
                                  (mode.direct ? ", direct (host window)" : "");
        wide_call(device, stream, 0.0, false, label + ", wide", mode);
        wide_call(device, stream, 0.0, true, label + ", wide with few-pair experts", mode);
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
