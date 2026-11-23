// ninfer::ops - the L2 read-ahead of guessed experts (contract in
// include/ninfer/ops/moe_expert_prefetch.h).
#include "ninfer/ops/moe_expert_prefetch.h"

#include "core/device_launch.h"

#include <algorithm>
#include <array>
#include <mutex>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr int kGuessExperts = 512;
constexpr int kGuessThreads = 256;
constexpr int kMaxPerRow    = 16;
constexpr int kMaxRows      = 8;
constexpr int kMaxIds       = 128;
constexpr int kReadThreads  = 256;
constexpr int kInFlight     = 4;

// Block per row; thread e scores experts e and e + 256. An expert's rank is how many experts
// precede it (higher score, or the same score and a lower index).
__global__ void __launch_bounds__(kGuessThreads)
    guess_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int experts,
                 int per_row, std::int32_t* __restrict__ ids) {
    __shared__ float score[kGuessExperts];
    const int row = int(blockIdx.x);
    for (int e = int(threadIdx.x); e < experts; e += kGuessThreads) {
        const float l = logits[std::int64_t(row) * experts + e];
        score[e]      = bias != nullptr ? 1.0f / (1.0f + expf(-l)) + bias[e] : l;
    }
    __syncthreads();
    for (int e = int(threadIdx.x); e < experts; e += kGuessThreads) {
        const float s = score[e];
        int rank      = 0;
        for (int j = 0; j < experts && rank < per_row; ++j) {
            const float o = score[j];
            rank += o > s || (o == s && j < e) ? 1 : 0;
        }
        if (rank < per_row) { ids[row * per_row + rank] = e; }
    }
}

struct Spans {
    const void* const* tables[ExpertPrefetch::kMaxProjections];
    std::int64_t bytes[ExpertPrefetch::kMaxProjections];
};

// Each block lists the distinct experts, then the grid reads every listed expert's spans as
// 16-byte words, kInFlight loads a thread at a time, through L2 only (ld.global.cg).
__global__ void __launch_bounds__(kReadThreads)
    read_kernel(Spans s, int projections, int experts, const std::int32_t* __restrict__ ids,
                int count, unsigned* __restrict__ sink) {
    __shared__ const unsigned char* begin[kMaxIds * ExpertPrefetch::kMaxProjections];
    __shared__ std::int64_t words[kMaxIds * ExpertPrefetch::kMaxProjections + 1];
    __shared__ int spans;
    if (threadIdx.x == 0) {
        int n            = 0;
        std::int64_t sum = 0;
        for (int i = 0; i < count; ++i) {
            const int x = ids[i];
            if (x < 0 || x >= experts) { continue; }
            bool seen = false;
            for (int j = 0; j < i && !seen; ++j) { seen = ids[j] == x; }
            if (seen) { continue; }
            for (int p = 0; p < projections; ++p) {
                const auto* base = static_cast<const unsigned char*>(s.tables[p][x]);
                if (base == nullptr || s.bytes[p] < 16) { continue; }
                const auto first =
                    (reinterpret_cast<std::uintptr_t>(base) + 15) & ~std::uintptr_t(15);
                const auto last =
                    (reinterpret_cast<std::uintptr_t>(base) + std::uintptr_t(s.bytes[p])) &
                    ~std::uintptr_t(15);
                if (last <= first) { continue; }
                begin[n] = reinterpret_cast<const unsigned char*>(first);
                words[n] = sum;
                sum += std::int64_t(last - first) / 16;
                ++n;
            }
        }
        words[n] = sum;
        spans    = n;
    }
    __syncthreads();
    const int n             = spans;
    const std::int64_t all  = words[n];
    const std::int64_t step = std::int64_t(gridDim.x) * blockDim.x;
    unsigned acc            = 0;
    int span                = 0;
    for (std::int64_t w = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x; w < all;
         w += kInFlight * step) {
        uint4 v[kInFlight];
#pragma unroll
        for (int u = 0; u < kInFlight; ++u) {
            const std::int64_t i = w + u * step;
            v[u]                 = make_uint4(0, 0, 0, 0);
            if (i < all) {
                while (span + 1 < n && words[span + 1] <= i) { ++span; }
                v[u] = __ldcg(reinterpret_cast<const uint4*>(begin[span]) + (i - words[span]));
            }
        }
#pragma unroll
        for (int u = 0; u < kInFlight; ++u) { acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w; }
    }
    // A data-dependent store the compiler cannot prove dead keeps the loads.
    if (acc == 0x9E3779B9u) { *sink = acc; }
}

struct DeviceFacts {
    int multiprocessors = 0;
    int l2_bytes        = 0;
    unsigned* sink      = nullptr;
};

// Per device: its multiprocessors, its L2 size and a word for read_kernel's sink, kept for the
// process's life.
const DeviceFacts& facts(int device) {
    static std::mutex mutex;
    static std::array<DeviceFacts, 64> table{};
    if (device < 0 || device >= int(table.size())) {
        throw std::invalid_argument("moe_expert_prefetch: device index out of range");
    }
    const std::lock_guard<std::mutex> lock(mutex);
    DeviceFacts& f = table[std::size_t(device)];
    if (f.sink == nullptr) {
        CUDA_CHECK(
            cudaDeviceGetAttribute(&f.multiprocessors, cudaDevAttrMultiProcessorCount, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&f.l2_bytes, cudaDevAttrL2CacheSize, device));
        int current = 0;
        CUDA_CHECK(cudaGetDevice(&current));
        CUDA_CHECK(cudaSetDevice(device));
        void* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, 256));
        CUDA_CHECK(cudaSetDevice(current));
        f.sink = static_cast<unsigned*>(p);
    }
    return f;
}

} // namespace

void moe_guess_experts(const float* logits, const float* bias, int rows, int experts, int per_row,
                       std::int32_t* ids, cudaStream_t stream) {
    if (rows <= 0) { return; }
    if (rows > kMaxRows || experts <= 0 || experts > kGuessExperts || per_row <= 0 ||
        per_row > kMaxPerRow || per_row > experts || logits == nullptr || ids == nullptr) {
        throw std::invalid_argument("moe_guess_experts: invalid rows, experts or per_row");
    }
    guess_kernel<<<rows, kGuessThreads, 0, stream>>>(logits, bias, experts, per_row, ids);
    CUDA_CHECK(cudaGetLastError());
}

void moe_expert_prefetch(const ExpertPrefetch& args, cudaStream_t stream) {
    if (args.count <= 0 || args.projections <= 0) { return; }
    if (args.count > kMaxIds || args.projections > ExpertPrefetch::kMaxProjections ||
        args.experts <= 0 || args.ids == nullptr) {
        throw std::invalid_argument("moe_expert_prefetch: invalid ids or projections");
    }
    Spans s{};
    for (int p = 0; p < args.projections; ++p) {
        if (args.tables[p] == nullptr || args.bytes[p] < 0) {
            throw std::invalid_argument("moe_expert_prefetch: a projection without a table");
        }
        s.tables[p] = args.tables[p];
        s.bytes[p]  = args.bytes[p];
    }
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    const DeviceFacts& f = facts(device);
    read_kernel<<<f.multiprocessors, kReadThreads, 0, stream>>>(s, args.projections, args.experts,
                                                                args.ids, args.count, f.sink);
    CUDA_CHECK(cudaGetLastError());
}

int moe_prefetch_budget(int device, std::size_t expert_bytes) {
    if (expert_bytes == 0) { return 0; }
    const DeviceFacts& f = facts(device);
    const std::size_t n  = std::size_t(f.l2_bytes) / 4 / expert_bytes;
    return n < 2 ? 0 : int(std::min<std::size_t>(n, kMaxPerRow));
}

} // namespace ninfer::ops
