#include "core/tensor_parallel.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

void check(cudaError_t err, const char* what) {
    if (err == cudaSuccess) { return; }
    throw std::runtime_error(std::string("TensorParallelExchange ") + what + ": " +
                             cudaGetErrorName(err) + ": " + cudaGetErrorString(err));
}

// Value i of partial r is part[r][i - first[r]]: an inbox row holds an owner's range from its start.
struct PartialSet {
    const float* part[kMaxRanks];
    std::int64_t first[kMaxRanks];
    int count;
};

// One value's sum into the output, as TensorParallelSum says.
template <TensorParallelSum Sum>
__device__ __forceinline__ void store_sum(void* out, std::int64_t i, float sum) {
    if constexpr (Sum == TensorParallelSum::AddBf16) {
        auto* residual = static_cast<__nv_bfloat16*>(out);
        residual[i]    = __float2bfloat16(__bfloat162float(residual[i]) + sum);
    } else if constexpr (Sum == TensorParallelSum::StoreBf16) {
        static_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16(sum);
    } else {
        static_cast<float*>(out)[i] = sum;
    }
}

struct alignas(8) Bf16x4 {
    __nv_bfloat16 v[4];
};

// Four consecutive values' sums, from a multiple of four, each rounded as store_sum rounds it.
template <TensorParallelSum Sum>
__device__ __forceinline__ void store_sum4(void* out, std::int64_t i, float4 sum) {
    if constexpr (Sum == TensorParallelSum::AddBf16) {
        auto* residual = reinterpret_cast<Bf16x4*>(static_cast<__nv_bfloat16*>(out) + i);
        Bf16x4 word    = *residual;
        word.v[0]      = __float2bfloat16(__bfloat162float(word.v[0]) + sum.x);
        word.v[1]      = __float2bfloat16(__bfloat162float(word.v[1]) + sum.y);
        word.v[2]      = __float2bfloat16(__bfloat162float(word.v[2]) + sum.z);
        word.v[3]      = __float2bfloat16(__bfloat162float(word.v[3]) + sum.w);
        *residual      = word;
    } else if constexpr (Sum == TensorParallelSum::StoreBf16) {
        const Bf16x4 word{{__float2bfloat16(sum.x), __float2bfloat16(sum.y),
                           __float2bfloat16(sum.z), __float2bfloat16(sum.w)}};
        *reinterpret_cast<Bf16x4*>(static_cast<__nv_bfloat16*>(out) + i) = word;
    } else {
        *reinterpret_cast<float4*>(static_cast<float*>(out) + i) = sum;
    }
}

// Values [begin, end); the adds run in rank order. `vector` (begin a multiple of four and every
// pointer and first aligned for it) reads each partial four values at a time, which crosses the bus
// in fewer, larger requests; the arithmetic per value is the same.
template <TensorParallelSum Sum>
__global__ void tensor_parallel_reduce_kernel(void* __restrict__ out, PartialSet parts,
                                              std::int64_t begin, std::int64_t end, bool vector) {
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t thread = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::int64_t tail         = begin;
    if (vector) {
        const std::int64_t quads = (end - begin) / 4;
        for (std::int64_t q = thread; q < quads; q += stride) {
            const std::int64_t i = begin + 4 * q;
            float4 sum = *reinterpret_cast<const float4*>(parts.part[0] + (i - parts.first[0]));
            for (int r = 1; r < parts.count; ++r) {
                const float4 part =
                    *reinterpret_cast<const float4*>(parts.part[r] + (i - parts.first[r]));
                sum.x += part.x;
                sum.y += part.y;
                sum.z += part.z;
                sum.w += part.w;
            }
            store_sum4<Sum>(out, i, sum);
        }
        tail = begin + 4 * quads;
    }
    for (std::int64_t i = tail + thread; i < end; i += stride) {
        float sum = parts.part[0][i - parts.first[0]];
        for (int r = 1; r < parts.count; ++r) { sum += parts.part[r][i - parts.first[r]]; }
        store_sum<Sum>(out, i, sum);
    }
}

struct GatherSet {
    const unsigned char* source[kMaxRanks];
    std::int64_t offset[kMaxRanks];
    std::int64_t bytes[kMaxRanks];
};

// Row y of the grid copies `bytes[y]` from `source[y]` to `out + offset[y]`, 16 bytes at a time
// where `vector` says every pointer and offset allows it.
__global__ void tensor_parallel_gather_kernel(unsigned char* __restrict__ out, GatherSet set,
                                              bool vector) {
    const unsigned char* source = set.source[blockIdx.y];
    unsigned char* destination  = out + set.offset[blockIdx.y];
    const std::int64_t bytes    = set.bytes[blockIdx.y];
    const std::int64_t stride   = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t thread   = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::int64_t tail           = 0;
    if (vector) {
        const std::int64_t words = bytes / 16;
        for (std::int64_t w = thread; w < words; w += stride) {
            reinterpret_cast<uint4*>(destination)[w] = reinterpret_cast<const uint4*>(source)[w];
        }
        tail = words * 16;
    }
    for (std::int64_t b = tail + thread; b < bytes; b += stride) { destination[b] = source[b]; }
}

struct CopySet {
    unsigned char* destination[kMaxRanks];
    const unsigned char* source[kMaxRanks];
    std::int64_t bytes[kMaxRanks];
};

// Row y of the grid copies `bytes[y]` from `source[y]` to `destination[y]`, 16 bytes at a time
// where `vector` says every pointer allows it.
__global__ void tensor_parallel_copies_kernel(CopySet set, bool vector) {
    const unsigned char* source = set.source[blockIdx.y];
    unsigned char* destination  = set.destination[blockIdx.y];
    const std::int64_t bytes    = set.bytes[blockIdx.y];
    const std::int64_t stride   = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t thread   = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::int64_t tail           = 0;
    if (vector) {
        const std::int64_t words = bytes / 16;
        for (std::int64_t w = thread; w < words; w += stride) {
            reinterpret_cast<uint4*>(destination)[w] = reinterpret_cast<const uint4*>(source)[w];
        }
        tail = words * 16;
    }
    for (std::int64_t b = tail + thread; b < bytes; b += stride) { destination[b] = source[b]; }
}

bool aligned(const void* pointer, std::size_t alignment) {
    return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

std::size_t output_bytes(TensorParallelSum sum) {
    return sum == TensorParallelSum::StoreFp32 ? sizeof(float) : sizeof(__nv_bfloat16);
}

struct WordsArgs {
    void* out;
    const float* mine;
    unsigned long long* targets[kMaxRanks];
    const unsigned long long* sources[kMaxRanks];
    std::uint32_t* epoch;
    std::int64_t values;
    std::int64_t capacity;
    int target_count;
    int ranks;
    int rank;
    int contributors;
};

__device__ __forceinline__ unsigned long long words_clock() {
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

// The value of a word once it carries `epoch`.
__device__ __forceinline__ std::uint32_t await_word(const unsigned long long* word,
                                                    std::uint32_t epoch,
                                                    unsigned long long begun) {
    unsigned long long seen = *reinterpret_cast<const volatile unsigned long long*>(word);
    for (std::uint32_t spins = 0; static_cast<std::uint32_t>(seen >> 32U) != epoch;) {
        if ((++spins & 1023U) == 0 &&
            words_clock() - begun > TensorParallelExchange::kWordsTimeoutNs) {
            printf("ninfer: a tensor-parallel exchange waited %llu s for epoch %u\n",
                   TensorParallelExchange::kWordsTimeoutNs / 1'000'000'000ULL, epoch);
            __trap();
        }
        seen = *reinterpret_cast<const volatile unsigned long long*>(word);
    }
    return static_cast<std::uint32_t>(seen);
}

// The epoch of the exchange after one of `epoch`. Rows start zeroed, as if every word carried
// epoch 0, so after 2^32 - 1 exchanges the count goes on at 2: a word nobody stored never matches,
// and the parity, the row's slot, still alternates.
__device__ __forceinline__ std::uint32_t next_words_epoch(std::uint32_t epoch) {
    return epoch == 0xFFFFFFFFU ? 2U : epoch + 1U;
}

// One rank's words exchange: stores its words into every row of `targets` (its own host row, or
// its row in every other rank's memory), then reads rank r's words from `sources[r]`.
template <TensorParallelSum Sum>
__global__ void tensor_parallel_words_kernel(WordsArgs a) {
    const std::uint32_t epoch =
        next_words_epoch(*reinterpret_cast<const volatile std::uint32_t*>(a.epoch));
    const std::int64_t base   = static_cast<std::int64_t>(epoch & 1U) * a.capacity;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t first  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool contributes    = a.rank < a.contributors;
    const unsigned long long tag = static_cast<unsigned long long>(epoch) << 32U;
    if (contributes) {
        for (std::int64_t i = first; i < a.values; i += stride) {
            const unsigned long long word = tag | __float_as_uint(a.mine[i]);
            for (int t = 0; t < a.target_count; ++t) {
                reinterpret_cast<volatile unsigned long long*>(a.targets[t])[base + i] = word;
            }
        }
    } else if (first == 0) {
        for (int t = 0; t < a.target_count; ++t) {
            reinterpret_cast<volatile unsigned long long*>(a.targets[t])[base] = tag;
        }
    }
    const unsigned long long begun = words_clock();
    for (std::int64_t i = first; i < a.values; i += stride) {
        float sum = 0.0F;
        for (int r = 0; r < a.contributors; ++r) {
            const float part =
                r == a.rank ? a.mine[i]
                            : __uint_as_float(await_word(a.sources[r] + base + i, epoch, begun));
            sum = r == 0 ? part : sum + part;
        }
        store_sum<Sum>(a.out, i, sum);
    }
    if (first == 0) {
        for (int r = a.contributors; r < a.ranks; ++r) {
            if (r != a.rank) { (void)await_word(a.sources[r] + base, epoch, begun); }
        }
    }
    // Every block read the count before it finished, so the last one to finish advances it.
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(a.epoch + 1, 1U) + 1U == gridDim.x) {
            a.epoch[1] = 0;
            a.epoch[0] = epoch;
        }
    }
}

// Values [begin, end).
struct ValueRange {
    std::int64_t begin = 0;
    std::int64_t end   = 0;
};

// Rank `rank`'s range of a reduce-scatter of `values` over `ranks` ranks. The ranks before the
// remainder take one block more, so rank 0's range is the largest.
ValueRange scatter_range(std::size_t rank, std::size_t ranks, std::int64_t values) noexcept {
    constexpr std::int64_t kBlock = 64;
    const auto n                  = static_cast<std::int64_t>(ranks);
    const auto r                  = static_cast<std::int64_t>(rank);
    const std::int64_t blocks     = (values + kBlock - 1) / kBlock;
    const std::int64_t per        = blocks / n;
    const std::int64_t extra      = blocks % n;
    const std::int64_t first      = r * per + std::min(r, extra);
    const std::int64_t last       = first + per + (r < extra ? 1 : 0);
    return {std::min(first * kBlock, values), std::min(last * kBlock, values)};
}

// A piece's reduce-scatter range in an inbox row begins past every earlier piece's: at a sixty-fourth
// of its offset over the ranks rounded up, plus kPieceInboxStep floats per earlier piece, which
// covers both roundings of the ranges (a range is at most values / ranks + 128 floats long).
constexpr std::int64_t kPieceInboxStep = 192;
// What the pieces of a full slot take in a row beyond its owner's range of the whole slot.
constexpr std::size_t kPieceInboxSlack =
    TensorParallelExchange::kPieces * static_cast<std::size_t>(kPieceInboxStep) + 256;

// Floats of one PeerWrite inbox row: an owner's range of a full slot, or a whole partial of up to
// kPeerWriteWholeBytes, whichever is more, but never more than the slot; whole 64-value blocks, so
// every row starts aligned.
std::size_t inbox_values(std::size_t slot_values, std::size_t ranks) {
    const ValueRange owned = scatter_range(0, ranks, static_cast<std::int64_t>(slot_values));
    const std::size_t whole = TensorParallelExchange::kPeerWriteWholeBytes / sizeof(float);
    const std::size_t row   = std::min(
        slot_values,
        std::max(static_cast<std::size_t>(owned.end - owned.begin) + kPieceInboxSlack, whole));
    return (row + 63) / 64 * 64;
}

std::size_t words_values(std::size_t words_bytes, std::size_t slot_bytes) {
    return std::min(words_bytes, slot_bytes) / sizeof(float);
}

// out[i] <- bf16(float(out[i]) + ((part[0][i] + part[1][i]) + ...)) for values [begin, end), or the
// sum stored as BF16 or FP32 (`sum`); value i of partial r is parts[r][i - firsts[r]].
void reduce_values(void* out, const float* const* parts, const std::int64_t* firsts,
                   std::size_t count, std::int64_t begin, std::int64_t end, cudaStream_t stream,
                   TensorParallelSum sum) {
    if (end == begin) { return; }
    PartialSet set{};
    bool vector = begin % 4 == 0 && aligned(out, 4 * output_bytes(sum));
    for (std::size_t r = 0; r < count; ++r) {
        set.part[r]  = parts[r];
        set.first[r] = firsts[r];
        vector       = vector && aligned(parts[r], sizeof(float4)) && firsts[r] % 4 == 0;
    }
    set.count                 = static_cast<int>(count);
    constexpr int kBlock      = 256;
    const std::int64_t items  = vector ? (end - begin + 3) / 4 : end - begin;
    const std::int64_t blocks = std::min<std::int64_t>((items + kBlock - 1) / kBlock, 4096);
    const auto grid           = static_cast<unsigned>(blocks);
    switch (sum) {
    case TensorParallelSum::AddBf16:
        tensor_parallel_reduce_kernel<TensorParallelSum::AddBf16>
            <<<grid, kBlock, 0, stream>>>(out, set, begin, end, vector);
        break;
    case TensorParallelSum::StoreBf16:
        tensor_parallel_reduce_kernel<TensorParallelSum::StoreBf16>
            <<<grid, kBlock, 0, stream>>>(out, set, begin, end, vector);
        break;
    case TensorParallelSum::StoreFp32:
        tensor_parallel_reduce_kernel<TensorParallelSum::StoreFp32>
            <<<grid, kBlock, 0, stream>>>(out, set, begin, end, vector);
        break;
    }
    check(cudaGetLastError(), "reduce launch");
}

// Byte copies into `out`: `bytes[i]` from `sources[i]` to `out + offsets[i]`, for i < `count`.
void gather_ranges(void* out, const void* const* sources, const std::int64_t* offsets,
                   const std::int64_t* bytes, std::size_t count, cudaStream_t stream) {
    GatherSet set{};
    std::size_t rows     = 0;
    std::int64_t largest = 0;
    bool vector          = aligned(out, 16);
    for (std::size_t i = 0; i < count; ++i) {
        if (bytes[i] == 0) { continue; }
        set.source[rows] = static_cast<const unsigned char*>(sources[i]);
        set.offset[rows] = offsets[i];
        set.bytes[rows]  = bytes[i];
        vector           = vector && aligned(sources[i], 16) && offsets[i] % 16 == 0;
        largest          = std::max(largest, bytes[i]);
        ++rows;
    }
    if (rows == 0) { return; }
    constexpr int kBlock      = 256;
    const std::int64_t items  = vector ? (largest + 15) / 16 : largest;
    const std::int64_t blocks = std::min<std::int64_t>((items + kBlock - 1) / kBlock, 1024);
    tensor_parallel_gather_kernel<<<
        dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(rows)), kBlock, 0, stream>>>(
        static_cast<unsigned char*>(out), set, vector);
    check(cudaGetLastError(), "gather launch");
}

// Byte copies `bytes[i]` from `sources[i]` to `destinations[i]`, for i < `count`: the copy engines
// take them where each is at least kEngineBytes, which they stream faster than kernels store across
// the bus (4x RTX A4000, 2026-11-02); one kernel takes smaller ones, all at once, without a copy's
// fixed cost each.
void copy_rows(void* const* destinations, const void* const* sources, const std::int64_t* bytes,
               std::size_t count, cudaStream_t stream) {
    constexpr std::int64_t kEngineBytes = 1LL << 20;
    bool engine                         = true;
    for (std::size_t i = 0; i < count; ++i) {
        engine = engine && (bytes[i] == 0 || bytes[i] >= kEngineBytes);
    }
    if (engine) {
        for (std::size_t i = 0; i < count; ++i) {
            if (bytes[i] == 0) { continue; }
            check(cudaMemcpyAsync(destinations[i], sources[i], static_cast<std::size_t>(bytes[i]),
                                  cudaMemcpyDeviceToDevice, stream),
                  "peer copy");
        }
        return;
    }
    CopySet set{};
    std::size_t rows     = 0;
    std::int64_t largest = 0;
    bool vector          = true;
    for (std::size_t i = 0; i < count; ++i) {
        if (bytes[i] == 0) { continue; }
        set.destination[rows] = static_cast<unsigned char*>(destinations[i]);
        set.source[rows]      = static_cast<const unsigned char*>(sources[i]);
        set.bytes[rows]       = bytes[i];
        vector  = vector && aligned(destinations[i], 16) && aligned(sources[i], 16);
        largest = std::max(largest, bytes[i]);
        ++rows;
    }
    if (rows == 0) { return; }
    constexpr int kBlock      = 256;
    const std::int64_t items  = vector ? (largest + 15) / 16 : largest;
    const std::int64_t blocks = std::min<std::int64_t>((items + kBlock - 1) / kBlock, 1024);
    tensor_parallel_copies_kernel<<<
        dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(rows)), kBlock, 0, stream>>>(
        set, vector);
    check(cudaGetLastError(), "copies launch");
}

void exchange_words(const WordsArgs& args, cudaStream_t stream, TensorParallelSum sum) {
    // About two values per thread: a poll is a bus round trip, so the waits overlap across blocks.
    constexpr int kBlock      = 256;
    const std::int64_t blocks = std::clamp<std::int64_t>(args.values / (2 * kBlock), 1, 64);
    const auto grid           = static_cast<unsigned>(blocks);
    switch (sum) {
    case TensorParallelSum::AddBf16:
        tensor_parallel_words_kernel<TensorParallelSum::AddBf16><<<grid, kBlock, 0, stream>>>(args);
        break;
    case TensorParallelSum::StoreBf16:
        tensor_parallel_words_kernel<TensorParallelSum::StoreBf16><<<grid, kBlock, 0, stream>>>(args);
        break;
    case TensorParallelSum::StoreFp32:
        tensor_parallel_words_kernel<TensorParallelSum::StoreFp32><<<grid, kBlock, 0, stream>>>(args);
        break;
    }
    check(cudaGetLastError(), "words exchange launch");
}

const TensorParallelBand& band_of(const std::vector<TensorParallelBand>& bands, std::size_t bytes) {
    for (const TensorParallelBand& band : bands) {
        if (bytes <= band.bytes) { return band; }
    }
    return bands.back();
}

// Consecutive bands with one route as one, up to the last one's size.
std::vector<TensorParallelBand> merged(const std::vector<TensorParallelBand>& bands) {
    std::vector<TensorParallelBand> out;
    for (const TensorParallelBand& band : bands) {
        if (!out.empty() && out.back().route == band.route) {
            out.back() = band;
        } else {
            out.push_back(band);
        }
    }
    return out;
}

std::string size_text(std::size_t bytes) {
    char text[32];
    if (bytes >= (1U << 20U)) {
        std::snprintf(text, sizeof(text), bytes % (1U << 20U) == 0 ? "%.0f MiB" : "%.2f MiB",
                      static_cast<double>(bytes) / 1048576.0);
    } else if (bytes >= 1024U) {
        std::snprintf(text, sizeof(text), bytes % 1024U == 0 ? "%.0f KiB" : "%.1f KiB",
                      static_cast<double>(bytes) / 1024.0);
    } else {
        std::snprintf(text, sizeof(text), "%zu B", bytes);
    }
    return text;
}

constexpr TensorParallelForm kForms[] = {TensorParallelForm::Words, TensorParallelForm::Whole,
                                         TensorParallelForm::Scatter};

} // namespace

const char* tensor_parallel_transport_name(TensorParallelTransport transport) noexcept {
    switch (transport) {
    case TensorParallelTransport::Auto:
        return "auto";
    case TensorParallelTransport::Host:
        return "host";
    case TensorParallelTransport::PeerRead:
        return "peer-read";
    case TensorParallelTransport::PeerWrite:
        return "peer-write";
    }
    return "unknown";
}

const char* tensor_parallel_form_name(TensorParallelForm form) noexcept {
    switch (form) {
    case TensorParallelForm::Words:
        return "words";
    case TensorParallelForm::Whole:
        return "whole";
    case TensorParallelForm::Scatter:
        return "scatter";
    }
    return "unknown";
}

std::vector<std::uint64_t> rank_free_bytes(const DeviceContext& context) {
    std::vector<std::uint64_t> out(context.size(), 0);
    for (std::size_t rank = 0; rank < context.size(); ++rank) {
        std::size_t sharing = 0;
        for (std::size_t other = 0; other < context.size(); ++other) {
            if (context.same_physical_device(rank, other)) { ++sharing; }
        }
        DeviceBinding bind(context.rank(rank).device);
        std::size_t free = 0, total = 0;
        check(cudaMemGetInfo(&free, &total), "free device memory");
        out[rank] = free / sharing;
    }
    return out;
}

std::vector<double> rank_copy_bandwidth(const DeviceContext& context) {
    // Large enough to stream past the L2 of every supported card, small enough to fit beside a
    // CUDA context on any of them.
    constexpr std::size_t kBytes  = 128ULL << 20;
    constexpr int kTimedCopies    = 3;
    std::vector<double> out(context.size(), 0.0);
    for (std::size_t rank = 0; rank < context.size(); ++rank) {
        std::size_t first = rank;
        for (std::size_t other = 0; other < rank; ++other) {
            if (context.same_physical_device(rank, other)) {
                first = other;
                break;
            }
        }
        if (first != rank) {
            out[rank] = out[first];
            continue;
        }
        DeviceBinding bind(context.rank(rank).device);
        void* source      = nullptr;
        void* destination = nullptr;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop  = nullptr;
        cudaStream_t stream = nullptr;
        const auto release = [&] {
            if (stream != nullptr) { (void)cudaStreamDestroy(stream); }
            if (start != nullptr) { (void)cudaEventDestroy(start); }
            if (stop != nullptr) { (void)cudaEventDestroy(stop); }
            if (source != nullptr) { (void)cudaFree(source); }
            if (destination != nullptr) { (void)cudaFree(destination); }
        };
        try {
            check(cudaMalloc(&source, kBytes), "bandwidth cudaMalloc");
            check(cudaMalloc(&destination, kBytes), "bandwidth cudaMalloc");
            check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "bandwidth stream");
            check(cudaEventCreate(&start), "bandwidth event");
            check(cudaEventCreate(&stop), "bandwidth event");
            check(cudaMemsetAsync(source, 0, kBytes, stream), "bandwidth memset");
            check(cudaMemcpyAsync(destination, source, kBytes, cudaMemcpyDeviceToDevice, stream),
                  "bandwidth warm-up");
            check(cudaEventRecord(start, stream), "bandwidth record");
            for (int copy = 0; copy < kTimedCopies; ++copy) {
                check(cudaMemcpyAsync(destination, source, kBytes, cudaMemcpyDeviceToDevice, stream),
                      "bandwidth copy");
            }
            check(cudaEventRecord(stop, stream), "bandwidth record");
            check(cudaEventSynchronize(stop), "bandwidth synchronize");
            float milliseconds = 0.0F;
            check(cudaEventElapsedTime(&milliseconds, start, stop), "bandwidth elapsed");
            // A copy reads and writes every byte.
            const double moved = 2.0 * static_cast<double>(kBytes) * kTimedCopies;
            out[rank] = moved / (std::max(static_cast<double>(milliseconds), 1e-3) * 1e-3);
        } catch (...) {
            release();
            throw;
        }
        release();
    }
    for (std::size_t rank = 0; rank < context.size(); ++rank) {
        std::size_t sharing = 0;
        for (std::size_t other = 0; other < context.size(); ++other) {
            if (context.same_physical_device(rank, other)) { ++sharing; }
        }
        out[rank] /= static_cast<double>(sharing);
    }
    return out;
}

std::size_t TensorParallelExchange::device_bytes(std::size_t slot_bytes, std::size_t ranks,
                                                 bool peer, TensorParallelTransport transport) {
    const std::size_t values = slot_bytes / sizeof(float);
    std::size_t bytes        = kSlots * slot_bytes + 2 * sizeof(std::uint32_t);
    if (peer && ranks > 1 &&
        (transport == TensorParallelTransport::Auto ||
         transport == TensorParallelTransport::PeerWrite)) {
        const std::size_t words = words_values(TensorParallelExchangeOptions{}.words_bytes, slot_bytes);
        bytes += kSlots * (ranks - 1) * inbox_values(values, ranks) * sizeof(float) +
                 ranks * kSlots * words * sizeof(unsigned long long);
    }
    // The BF16 replica the measurement reduces into.
    return bytes + values * sizeof(__nv_bfloat16);
}

std::size_t TensorParallelExchange::device_bytes() const noexcept {
    if (ranks_.empty()) { return 0; }
    const Rank& rank  = ranks_.front();
    std::size_t bytes = kSlots * slot_bytes_ + 2 * sizeof(std::uint32_t);
    if (rank.inbox[0] != nullptr) {
        bytes += kSlots * (ranks_.size() - 1) * inbox_values_ * sizeof(float);
    }
    if (rank.peer_words != nullptr) {
        bytes += ranks_.size() * kSlots * words_values_ * sizeof(unsigned long long);
    }
    return bytes;
}

TensorParallelExchange::TensorParallelExchange(const DeviceContext& context,
                                               TensorParallelExchangeOptions options)
    : slot_bytes_(options.slot_bytes) {
    if (context.size() < 2) {
        throw std::invalid_argument("a tensor-parallel exchange needs at least two ranks");
    }
    if (slot_bytes_ == 0 || slot_bytes_ % sizeof(float) != 0) {
        throw std::invalid_argument("a tensor-parallel slot holds a positive number of floats");
    }
    peer_ = context.peer_access();
    const bool peer_transport = options.transport == TensorParallelTransport::PeerRead ||
                                options.transport == TensorParallelTransport::PeerWrite;
    if (peer_transport && !peer_) {
        throw std::invalid_argument(std::string("tensor-parallel transport ") +
                                    tensor_parallel_transport_name(options.transport) +
                                    " needs peer access between every pair of devices");
    }
    // Words need every rank on its own card.
    bool own_cards = true;
    for (std::size_t a = 0; a < context.size(); ++a) {
        for (std::size_t b = a + 1; b < context.size(); ++b) {
            if (context.same_physical_device(a, b)) { own_cards = false; }
        }
    }
    words_values_ = own_cards ? words_values(options.words_bytes, slot_bytes_) : 0;
    inbox_values_ = inbox_values(slot_bytes_ / sizeof(float), context.size());
    ranks_.resize(context.size());
    try {
        for (std::size_t r = 0; r < ranks_.size(); ++r) {
            Rank& rank  = ranks_[r];
            rank.device = context.rank(r).device;
            DeviceBinding bind(rank.device);
            for (std::size_t slot = 0; slot < kSlots; ++slot) {
                check(cudaMalloc(reinterpret_cast<void**>(&rank.device_slot[slot]), slot_bytes_),
                      "cudaMalloc");
                // Events cannot be created while a stream captures, so all exist up front. A
                // rank records its own events.
                for (std::size_t piece = 0; piece < kPieces; ++piece) {
                    check(cudaEventCreateWithFlags(&rank.published[slot][piece],
                                                   cudaEventDisableTiming),
                          "cudaEventCreate");
                    check(cudaEventCreateWithFlags(&rank.reduced[slot][piece],
                                                   cudaEventDisableTiming),
                          "cudaEventCreate");
                }
            }
            check(cudaStreamCreateWithFlags(&rank.side, cudaStreamNonBlocking), "cudaStreamCreate");
            for (std::size_t piece = 0; piece < kPieces; ++piece) {
                check(cudaEventCreateWithFlags(&rank.piece_ready[piece], cudaEventDisableTiming),
                      "cudaEventCreate");
                check(cudaEventCreateWithFlags(&rank.piece_done[piece], cudaEventDisableTiming),
                      "cudaEventCreate");
            }
            check(cudaMalloc(reinterpret_cast<void**>(&rank.epoch), 2 * sizeof(std::uint32_t)),
                  "cudaMalloc");
            const std::uint32_t count[2] = {options.words_count, 0};
            check(cudaMemcpy(rank.epoch, count, sizeof(count), cudaMemcpyHostToDevice),
                  "cudaMemcpy");
        }
        if (!options.routes.empty()) {
            for (std::size_t i = 0; i < options.routes.size(); ++i) {
                const TensorParallelTransport transport = options.routes[i].route.transport;
                if (transport == TensorParallelTransport::Auto ||
                    (transport != TensorParallelTransport::Host && !peer_) ||
                    (i > 0 && options.routes[i].bytes <= options.routes[i - 1].bytes)) {
                    throw std::invalid_argument(
                        "tensor-parallel routes need increasing sizes and transports the devices "
                        "allow");
                }
            }
            bands_ = options.routes;
            for (TensorParallelBand band : bands_) {
                band.route.form = TensorParallelForm::Whole;
                whole_bands_.push_back(band);
            }
            bands_       = merged(bands_);
            whole_bands_ = merged(whole_bands_);
            provide(needs(bands_));
            provide(needs(whole_bands_));
        } else {
            std::vector<TensorParallelTransport> transports;
            if (options.transport != TensorParallelTransport::Auto) {
                transports.push_back(options.transport);
            } else {
                transports.push_back(TensorParallelTransport::Host);
                if (peer_) {
                    transports.push_back(TensorParallelTransport::PeerRead);
                    transports.push_back(TensorParallelTransport::PeerWrite);
                }
            }
            measure(transports);
        }
    } catch (...) {
        release();
        throw;
    }
}

TensorParallelExchange::~TensorParallelExchange() { release(); }

void TensorParallelExchange::release() noexcept {
    for (Rank& rank : ranks_) {
        DeviceBinding bind(rank.device);
        for (std::size_t slot = 0; slot < kSlots; ++slot) {
            for (std::size_t piece = 0; piece < kPieces; ++piece) {
                for (cudaEvent_t* event : {&rank.published[slot][piece], &rank.reduced[slot][piece]}) {
                    if (*event != nullptr) { (void)cudaEventDestroy(*event); }
                    *event = nullptr;
                }
            }
            if (rank.device_slot[slot] != nullptr) { (void)cudaFree(rank.device_slot[slot]); }
            if (rank.host_slot[slot] != nullptr) { (void)cudaFreeHost(rank.host_slot[slot]); }
            if (rank.inbox[slot] != nullptr) { (void)cudaFree(rank.inbox[slot]); }
            rank.device_slot[slot] = nullptr;
            rank.host_slot[slot]   = nullptr;
            rank.inbox[slot]       = nullptr;
        }
        for (std::size_t piece = 0; piece < kPieces; ++piece) {
            for (cudaEvent_t* event : {&rank.piece_ready[piece], &rank.piece_done[piece]}) {
                if (*event != nullptr) { (void)cudaEventDestroy(*event); }
                *event = nullptr;
            }
        }
        if (rank.side != nullptr) { (void)cudaStreamDestroy(rank.side); }
        rank.side = nullptr;
        if (rank.host_words != nullptr) { (void)cudaFreeHost(rank.host_words); }
        if (rank.peer_words != nullptr) { (void)cudaFree(rank.peer_words); }
        if (rank.epoch != nullptr) { (void)cudaFree(rank.epoch); }
        rank.host_words = nullptr;
        rank.peer_words = nullptr;
        rank.epoch      = nullptr;
    }
    ranks_.clear();
}

bool TensorParallelExchange::legal(TensorParallelRoute route, std::size_t bytes) const noexcept {
    switch (route.form) {
    case TensorParallelForm::Words:
        return route.transport != TensorParallelTransport::PeerRead && words_values_ != 0 &&
               bytes <= words_values_ * sizeof(float);
    case TensorParallelForm::Whole:
        return route.transport != TensorParallelTransport::PeerWrite ||
               bytes <= inbox_values_ * sizeof(float);
    case TensorParallelForm::Scatter:
        return true;
    }
    return false;
}

TensorParallelExchange::Needs
TensorParallelExchange::needs(const std::vector<TensorParallelBand>& bands) const noexcept {
    Needs out;
    for (const TensorParallelBand& band : bands) {
        const bool words = band.route.form == TensorParallelForm::Words && words_values_ != 0;
        if (band.route.transport == TensorParallelTransport::Host) {
            (words ? out.host_words : out.host_slots) = true;
        } else if (band.route.transport == TensorParallelTransport::PeerWrite) {
            (words ? out.peer_words : out.inbox) = true;
        }
    }
    return out;
}

void TensorParallelExchange::provide(Needs needs) {
    const std::size_t row = kSlots * words_values_;
    for (std::size_t r = 0; r < ranks_.size(); ++r) {
        Rank& rank = ranks_[r];
        DeviceBinding bind(rank.device);
        for (std::size_t slot = 0; slot < kSlots; ++slot) {
            if (needs.host_slots && rank.host_slot[slot] == nullptr) {
                // Mapped, so a reduce on any rank reads it in place; portable, so every rank's
                // context treats it as pinned.
                check(cudaHostAlloc(reinterpret_cast<void**>(&rank.host_slot[slot]), slot_bytes_,
                                    cudaHostAllocMapped | cudaHostAllocPortable),
                      "cudaHostAlloc");
            }
            if (needs.inbox && rank.inbox[slot] == nullptr) {
                check(cudaMalloc(reinterpret_cast<void**>(&rank.inbox[slot]),
                                 (ranks_.size() - 1) * inbox_values_ * sizeof(float)),
                      "cudaMalloc");
            }
        }
        const bool host_words = needs.host_words && rank.host_words == nullptr;
        const bool peer_words = needs.peer_words && rank.peer_words == nullptr;
        if (host_words) {
            check(cudaHostAlloc(reinterpret_cast<void**>(&rank.host_words),
                                row * sizeof(unsigned long long),
                                cudaHostAllocMapped | cudaHostAllocPortable),
                  "cudaHostAlloc");
            std::fill_n(rank.host_words, row, 0ULL);
        }
        if (peer_words) {
            const std::size_t bytes = ranks_.size() * row * sizeof(unsigned long long);
            check(cudaMalloc(reinterpret_cast<void**>(&rank.peer_words), bytes), "cudaMalloc");
            check(cudaMemset(rank.peer_words, 0, bytes), "cudaMemset");
        }
        if (host_words || peer_words) {
            // Loaded now, so no first launch waits on module loading while a peer spins.
            cudaFuncAttributes attributes{};
            for (const auto kernel : {tensor_parallel_words_kernel<TensorParallelSum::AddBf16>,
                                      tensor_parallel_words_kernel<TensorParallelSum::StoreBf16>,
                                      tensor_parallel_words_kernel<TensorParallelSum::StoreFp32>}) {
                check(cudaFuncGetAttributes(&attributes, kernel), "words kernel");
            }
        }
    }
}

void TensorParallelExchange::measure(const std::vector<TensorParallelTransport>& transports) {
    constexpr std::size_t kFirstProbeBytes = 4U << 10U;
    constexpr int kTrials                  = 5;
    std::vector<std::size_t> probes;
    for (std::size_t bytes = kFirstProbeBytes; bytes < slot_bytes_; bytes *= 2) {
        probes.push_back(bytes);
    }
    probes.push_back(slot_bytes_);
    std::vector<std::vector<TensorParallelRoute>> candidates(probes.size());
    std::vector<TensorParallelBand> every;
    bool choice = false;
    for (std::size_t i = 0; i < probes.size(); ++i) {
        for (const TensorParallelTransport transport : transports) {
            for (const TensorParallelForm form : kForms) {
                const TensorParallelRoute route{transport, form};
                if (!legal(route, probes[i])) { continue; }
                candidates[i].push_back(route);
                every.push_back({probes[i], route});
            }
        }
        choice = choice || candidates[i].size() > 1;
    }

    // The fastest route at every probe and, between two probes, at sixteenths of the step, where a
    // route's time is the line through its times at both (an exchange costs a fixed part and a part
    // per byte; one octave apart the probes alone would misplace a crossover by up to half the
    // size). Beside it the fastest whole route: PeerWrite past its inbox rows has none, and its
    // whole reduces read.
    const auto choose = [&](const std::vector<double>& microseconds) {
        constexpr double kNone = std::numeric_limits<double>::infinity();
        constexpr int kSteps   = 16;
        std::vector<std::vector<double>> times(probes.size());
        std::size_t at = 0;
        for (std::size_t i = 0; i < probes.size(); ++i) {
            for (std::size_t k = 0; k < candidates[i].size(); ++k) {
                times[i].push_back(microseconds.empty() ? 0.0 : microseconds[at++]);
            }
        }
        std::vector<TensorParallelBand> bands;
        std::vector<TensorParallelBand> whole;
        const auto add = [&](std::size_t bytes, const auto& time_of) {
            TensorParallelBand best{bytes, {}, kNone};
            TensorParallelBand best_whole{
                bytes, {TensorParallelTransport::PeerRead, TensorParallelForm::Whole}, kNone};
            for (const auto& [route, taken] : time_of) {
                if (taken < best.microseconds) {
                    best.route        = route;
                    best.microseconds = taken;
                }
                if (route.form == TensorParallelForm::Whole && taken < best_whole.microseconds) {
                    best_whole.route        = route;
                    best_whole.microseconds = taken;
                }
            }
            if (best.microseconds == kNone) { return; }
            if (best_whole.microseconds == kNone) { best_whole.microseconds = 0.0; }
            bands.push_back(best);
            whole.push_back(best_whole);
        };
        std::vector<std::pair<TensorParallelRoute, double>> time_of;
        for (std::size_t i = 0; i < probes.size(); ++i) {
            for (int step = 1; i > 0 && !microseconds.empty() && step < kSteps; ++step) {
                const double share = static_cast<double>(step) / kSteps;
                time_of.clear();
                for (std::size_t k = 0; k < candidates[i].size(); ++k) {
                    for (std::size_t j = 0; j < candidates[i - 1].size(); ++j) {
                        if (candidates[i - 1][j] == candidates[i][k]) {
                            time_of.emplace_back(candidates[i][k],
                                                 times[i - 1][j] +
                                                     share * (times[i][k] - times[i - 1][j]));
                        }
                    }
                }
                add(probes[i - 1] +
                        static_cast<std::size_t>(share *
                                                 static_cast<double>(probes[i] - probes[i - 1])),
                    time_of);
            }
            time_of.clear();
            for (std::size_t k = 0; k < candidates[i].size(); ++k) {
                time_of.emplace_back(candidates[i][k], times[i][k]);
            }
            add(probes[i], time_of);
        }
        bands_       = merged(bands);
        whole_bands_ = merged(whole);
    };
    if (!choice) {
        choose({});
        provide(needs(bands_));
        provide(needs(whole_bands_));
        return;
    }
    provide(needs(every));

    // Every rank's stream, the fork and join of a pass, and the BF16 replica it reduces into.
    const std::size_t n = ranks_.size();
    std::vector<cudaStream_t> streams(n, nullptr);
    std::vector<cudaEvent_t> fork(n, nullptr);
    std::vector<cudaEvent_t> join(n, nullptr);
    std::vector<void*> replicas(n, nullptr);
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    const auto discard = [&] {
        for (std::size_t r = 0; r < n; ++r) {
            DeviceBinding bind(ranks_[r].device);
            if (streams[r] != nullptr) { (void)cudaStreamDestroy(streams[r]); }
            if (fork[r] != nullptr) { (void)cudaEventDestroy(fork[r]); }
            if (join[r] != nullptr) { (void)cudaEventDestroy(join[r]); }
            if (replicas[r] != nullptr) { (void)cudaFree(replicas[r]); }
        }
        DeviceBinding bind(ranks_[0].device);
        if (start != nullptr) { (void)cudaEventDestroy(start); }
        if (stop != nullptr) { (void)cudaEventDestroy(stop); }
    };
    const std::size_t replica_bytes = slot_bytes_ / sizeof(float) * sizeof(__nv_bfloat16);
    // Milliseconds of the fastest of kTrials replays of `exchanges` exchanges of `bytes` on `route`.
    const auto time_route = [&](TensorParallelRoute route, std::size_t bytes, int exchanges) {
        const auto values  = static_cast<std::int64_t>(bytes / sizeof(float));
        DeviceBinding bind(ranks_[0].device);
        cudaGraph_t graph          = nullptr;
        cudaGraphExec_t executable = nullptr;
        check(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeThreadLocal),
              "measurement capture");
        try {
            check(cudaEventRecord(fork[0], streams[0]), "measurement fork");
            for (std::size_t r = 1; r < n; ++r) {
                check(cudaStreamWaitEvent(streams[r], fork[0], 0), "measurement fork");
            }
            for (int e = 0; e < exchanges; ++e) {
                const auto slot = static_cast<std::size_t>(e) % kSlots;
                for (std::size_t r = 0; r < n; ++r) {
                    publish_on(route, r, slot, bytes, replicas[r], streams[r],
                               TensorParallelSum::AddBf16);
                }
                for (std::size_t r = 0; r < n; ++r) {
                    reduce_on(route, r, slot, replicas[r], values, n, streams[r],
                              TensorParallelSum::AddBf16);
                }
                for (std::size_t r = 0; r < n; ++r) {
                    complete_on(route, r, slot, replicas[r], values, streams[r],
                                TensorParallelSum::AddBf16);
                }
            }
            for (std::size_t r = 1; r < n; ++r) {
                check(cudaEventRecord(join[r], streams[r]), "measurement join");
                check(cudaStreamWaitEvent(streams[0], join[r], 0), "measurement join");
            }
        } catch (...) {
            (void)cudaStreamEndCapture(streams[0], &graph);
            if (graph != nullptr) { (void)cudaGraphDestroy(graph); }
            throw;
        }
        check(cudaStreamEndCapture(streams[0], &graph), "measurement capture");
        const cudaError_t instantiated = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
        (void)cudaGraphDestroy(graph);
        check(instantiated, "measurement instantiate");
        float best = std::numeric_limits<float>::infinity();
        try {
            check(cudaGraphLaunch(executable, streams[0]), "measurement launch");
            check(cudaStreamSynchronize(streams[0]), "measurement synchronize");
            for (int trial = 0; trial < kTrials; ++trial) {
                check(cudaEventRecord(start, streams[0]), "measurement record");
                check(cudaGraphLaunch(executable, streams[0]), "measurement launch");
                check(cudaEventRecord(stop, streams[0]), "measurement record");
                check(cudaEventSynchronize(stop), "measurement synchronize");
                float milliseconds = 0.0F;
                check(cudaEventElapsedTime(&milliseconds, start, stop), "measurement elapsed");
                best = std::min(best, milliseconds);
            }
        } catch (...) {
            (void)cudaGraphExecDestroy(executable);
            throw;
        }
        (void)cudaGraphExecDestroy(executable);
        return 1000.0 * static_cast<double>(best) / exchanges;
    };

    std::vector<double> microseconds;
    try {
        for (std::size_t r = 0; r < n; ++r) {
            DeviceBinding bind(ranks_[r].device);
            check(cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking), "measurement stream");
            check(cudaEventCreateWithFlags(&fork[r], cudaEventDisableTiming), "measurement event");
            check(cudaEventCreateWithFlags(&join[r], cudaEventDisableTiming), "measurement event");
            check(cudaMalloc(&replicas[r], replica_bytes), "measurement cudaMalloc");
            check(cudaMemsetAsync(replicas[r], 0, replica_bytes, streams[r]), "measurement memset");
            for (std::size_t slot = 0; slot < kSlots; ++slot) {
                check(cudaMemsetAsync(ranks_[r].device_slot[slot], 0, slot_bytes_, streams[r]),
                      "measurement memset");
            }
            check(cudaStreamSynchronize(streams[r]), "measurement synchronize");
        }
        {
            DeviceBinding bind(ranks_[0].device);
            check(cudaEventCreate(&start), "measurement event");
            check(cudaEventCreate(&stop), "measurement event");
        }
        for (std::size_t i = 0; i < probes.size(); ++i) {
            // Enough exchanges per replay that its launch is a small part of it.
            const int exchanges = static_cast<int>(
                std::clamp<std::size_t>((4U << 20U) / probes[i], 2, 32));
            for (const TensorParallelRoute& route : candidates[i]) {
                microseconds.push_back(time_route(route, probes[i], exchanges));
                timings_.push_back({probes[i], route, microseconds.back()});
            }
        }
    } catch (...) {
        discard();
        throw;
    }
    discard();
    measured_ = true;
    choose(microseconds);

    // Free what no band takes.
    Needs keep          = needs(bands_);
    const Needs whole   = needs(whole_bands_);
    keep.host_slots     = keep.host_slots || whole.host_slots;
    keep.inbox          = keep.inbox || whole.inbox;
    for (Rank& rank : ranks_) {
        DeviceBinding bind(rank.device);
        for (std::size_t slot = 0; slot < kSlots; ++slot) {
            if (!keep.host_slots && rank.host_slot[slot] != nullptr) {
                check(cudaFreeHost(rank.host_slot[slot]), "cudaFreeHost");
                rank.host_slot[slot] = nullptr;
            }
            if (!keep.inbox && rank.inbox[slot] != nullptr) {
                check(cudaFree(rank.inbox[slot]), "cudaFree");
                rank.inbox[slot] = nullptr;
            }
        }
        if (!keep.host_words && rank.host_words != nullptr) {
            check(cudaFreeHost(rank.host_words), "cudaFreeHost");
            rank.host_words = nullptr;
        }
        if (!keep.peer_words && rank.peer_words != nullptr) {
            check(cudaFree(rank.peer_words), "cudaFree");
            rank.peer_words = nullptr;
        }
    }
}

std::string TensorParallelExchange::describe() const {
    std::string out = "tensor-parallel exchange, " + std::to_string(ranks_.size()) + " ranks" +
                      (measured_ ? ", measured at startup:" : ":");
    for (std::size_t i = 0; i < bands_.size(); ++i) {
        const TensorParallelBand& band = bands_[i];
        out += std::string(i == 0 ? " " : ", ") +
               tensor_parallel_transport_name(band.route.transport) + " " +
               tensor_parallel_form_name(band.route.form) + " to " + size_text(band.bytes);
        if (measured_) {
            char text[32];
            std::snprintf(text, sizeof(text), " (%.1f us)", band.microseconds);
            out += text;
        }
    }
    return out;
}

TensorParallelRoute TensorParallelExchange::route(std::size_t bytes,
                                                  TensorParallelSum sum) const noexcept {
    const TensorParallelRoute chosen = band_of(bands_, bytes).route;
    // Two ranks storing an FP32 sum reduce whole (see Scatter above).
    const bool fp32_pair = chosen.form == TensorParallelForm::Scatter &&
                           sum == TensorParallelSum::StoreFp32 && ranks_.size() == 2;
    if (!fp32_pair && legal(chosen, bytes)) { return chosen; }
    return gather_route(bytes);
}

TensorParallelRoute TensorParallelExchange::gather_route(std::size_t bytes) const noexcept {
    const TensorParallelRoute chosen = band_of(whole_bands_, bytes).route;
    if (legal(chosen, bytes)) { return chosen; }
    // PeerWrite past its inbox rows: the same peer access, read.
    return {TensorParallelTransport::PeerRead, TensorParallelForm::Whole};
}

void TensorParallelExchange::check_slot(std::size_t rank, std::size_t slot,
                                        std::size_t bytes) const {
    if (rank >= ranks_.size() || slot >= kSlots) {
        throw std::out_of_range("tensor-parallel rank or slot is out of range");
    }
    if (bytes > slot_bytes_) {
        throw std::invalid_argument("tensor-parallel partial exceeds its slot");
    }
}

float* TensorParallelExchange::inbox_row(std::size_t rank, std::size_t sender,
                                         std::size_t slot) const {
    return ranks_[rank].inbox[slot] + (sender < rank ? sender : sender - 1) * inbox_values_;
}

float* TensorParallelExchange::partial(std::size_t rank, std::size_t slot) const {
    check_slot(rank, slot, 0);
    return ranks_[rank].device_slot[slot];
}

void TensorParallelExchange::publish(std::size_t rank, std::size_t slot, std::size_t bytes,
                                     void* out, cudaStream_t stream, TensorParallelSum sum,
                                     TensorParallelPiece piece) {
    check_piece(rank, slot, bytes, piece);
    publish_on(piece_route(bytes, sum, piece), rank, slot, bytes, out, stream, sum, piece);
}

void TensorParallelExchange::reduce(std::size_t rank, std::size_t slot, void* out,
                                    std::int64_t count, std::size_t contributors,
                                    cudaStream_t stream, TensorParallelSum sum,
                                    TensorParallelPiece piece) {
    const auto bytes = static_cast<std::size_t>(count) * sizeof(float);
    check_piece(rank, slot, bytes, piece);
    if (count <= 0 || contributors == 0 || contributors > ranks_.size()) {
        throw std::invalid_argument(
            "tensor-parallel reduce needs values and one to all ranks as contributors");
    }
    reduce_on(piece_route(bytes, sum, piece), rank, slot, out, count, contributors, stream, sum,
              piece);
}

void TensorParallelExchange::complete(std::size_t rank, std::size_t slot, void* out,
                                      std::int64_t count, cudaStream_t stream,
                                      TensorParallelSum sum, TensorParallelPiece piece) {
    const auto bytes = static_cast<std::size_t>(count) * sizeof(float);
    check_piece(rank, slot, bytes, piece);
    complete_on(piece_route(bytes, sum, piece), rank, slot, out, count, stream, sum, piece);
}

cudaStream_t TensorParallelExchange::side_stream(std::size_t rank) const {
    check_slot(rank, 0, 0);
    return ranks_[rank].side;
}

cudaEvent_t TensorParallelExchange::piece_ready(std::size_t rank, std::size_t piece) const {
    check_slot(rank, 0, 0);
    return ranks_[rank].piece_ready.at(piece);
}

cudaEvent_t TensorParallelExchange::piece_done(std::size_t rank, std::size_t piece) const {
    check_slot(rank, 0, 0);
    return ranks_[rank].piece_done.at(piece);
}

void TensorParallelExchange::check_piece(std::size_t rank, std::size_t slot, std::size_t bytes,
                                         TensorParallelPiece piece) const {
    check_slot(rank, slot, 0);
    if (piece.index >= kPieces || piece.offset < 0 ||
        static_cast<std::size_t>(piece.offset) * sizeof(float) + bytes > slot_bytes_) {
        throw std::invalid_argument("a tensor-parallel piece lies outside its slot");
    }
}

std::int64_t TensorParallelExchange::inbox_offset(TensorParallelPiece piece,
                                                  TensorParallelForm form) const noexcept {
    if (form != TensorParallelForm::Scatter) { return piece.offset; }
    const auto n = static_cast<std::int64_t>(ranks_.size());
    return (piece.offset + n * 64 - 1) / (n * 64) * 64 +
           static_cast<std::int64_t>(piece.index) * kPieceInboxStep;
}

TensorParallelRoute TensorParallelExchange::piece_route(std::size_t bytes, TensorParallelSum sum,
                                                        TensorParallelPiece piece) const {
    const TensorParallelRoute chosen = route(bytes, sum);
    if (piece.index == 0 && piece.offset == 0) { return chosen; }
    if (chosen.form == TensorParallelForm::Words) {
        throw std::logic_error("a tensor-parallel piece cannot be exchanged as words");
    }
    if (chosen.transport != TensorParallelTransport::PeerWrite) { return chosen; }
    // Every owner's range of the piece, at the piece's place, within an inbox row.
    const auto values = static_cast<std::int64_t>(bytes / sizeof(float));
    std::int64_t widest = values;
    if (chosen.form == TensorParallelForm::Scatter) {
        const ValueRange first = scatter_range(0, ranks_.size(), values);
        widest                 = first.end - first.begin;
    }
    if (inbox_offset(piece, chosen.form) + widest <= static_cast<std::int64_t>(inbox_values_)) {
        return chosen;
    }
    return {TensorParallelTransport::PeerRead, TensorParallelForm::Whole};
}

void TensorParallelExchange::publish_on(TensorParallelRoute route, std::size_t rank,
                                        std::size_t slot, std::size_t bytes, void* out,
                                        cudaStream_t stream, TensorParallelSum sum,
                                        TensorParallelPiece piece) {
    Rank& self = ranks_[rank];
    DeviceBinding bind(self.device);
    self.output[slot][piece.index]        = out;
    self.reduced_ready[slot][piece.index] = false;
    // Words leave the slot where it is: the reduce's kernel stores them itself. PeerRead leaves it
    // for the others to read.
    const auto values  = static_cast<std::int64_t>(bytes / sizeof(float));
    float* const local = self.device_slot[slot] + piece.offset;
    if (bytes != 0 && route.transport == TensorParallelTransport::Host &&
        route.form != TensorParallelForm::Words) {
        // A reduce-scatter's own range of the partial stays on the device, since no other rank
        // reads it.
        std::size_t own_begin = bytes;
        std::size_t own_end   = bytes;
        if (route.form == TensorParallelForm::Scatter) {
            const ValueRange range = scatter_range(rank, ranks_.size(), values);
            own_begin              = static_cast<std::size_t>(range.begin) * sizeof(float);
            own_end                = static_cast<std::size_t>(range.end) * sizeof(float);
        }
        auto* host         = reinterpret_cast<unsigned char*>(self.host_slot[slot] + piece.offset);
        const auto* device = reinterpret_cast<const unsigned char*>(local);
        if (own_begin != 0) {
            check(cudaMemcpyAsync(host, device, own_begin, cudaMemcpyDeviceToHost, stream),
                  "publish copy");
        }
        if (own_end != bytes) {
            check(cudaMemcpyAsync(host + own_end, device + own_end, bytes - own_end,
                                  cudaMemcpyDeviceToHost, stream),
                  "publish copy");
        }
    }
    if (bytes != 0 && route.transport == TensorParallelTransport::PeerWrite &&
        route.form != TensorParallelForm::Words) {
        // The whole partial into the rank's row of every other inbox, or each owner's range of it
        // into the rank's row of the owner's.
        std::array<void*, kMaxRanks> destinations{};
        std::array<const void*, kMaxRanks> sources{};
        std::array<std::int64_t, kMaxRanks> sizes{};
        std::size_t copies       = 0;
        const std::int64_t place = inbox_offset(piece, route.form);
        for (std::size_t other = 0; other < ranks_.size(); ++other) {
            if (other == rank) { continue; }
            const ValueRange range = route.form == TensorParallelForm::Scatter
                                         ? scatter_range(other, ranks_.size(), values)
                                         : ValueRange{0, values};
            destinations[copies] = inbox_row(other, rank, slot) + place;
            sources[copies]      = local + range.begin;
            sizes[copies]        = (range.end - range.begin) * static_cast<std::int64_t>(sizeof(float));
            ++copies;
        }
        copy_rows(destinations.data(), sources.data(), sizes.data(), copies, stream);
    }
    check(cudaEventRecord(self.published[slot][piece.index], stream), "publish fence");
}

void TensorParallelExchange::reduce_on(TensorParallelRoute route, std::size_t rank,
                                       std::size_t slot, void* out, std::int64_t count,
                                       std::size_t contributors, cudaStream_t stream,
                                       TensorParallelSum sum, TensorParallelPiece piece) {
    Rank& self = ranks_[rank];
    DeviceBinding bind(self.device);
    if (route.form == TensorParallelForm::Words) {
        WordsArgs args{};
        args.out              = out;
        args.mine             = self.device_slot[slot];
        const std::size_t row = kSlots * words_values_;
        for (std::size_t r = 0; r < ranks_.size(); ++r) {
            args.sources[r] = route.transport == TensorParallelTransport::Host
                                  ? ranks_[r].host_words
                                  : self.peer_words + r * row;
            if (route.transport == TensorParallelTransport::PeerWrite && r != rank) {
                args.targets[args.target_count++] = ranks_[r].peer_words + rank * row;
            }
        }
        if (route.transport == TensorParallelTransport::Host) {
            args.targets[0]   = self.host_words;
            args.target_count = 1;
        }
        args.epoch        = self.epoch;
        args.values       = count;
        args.capacity     = static_cast<std::int64_t>(words_values_);
        args.ranks        = static_cast<int>(ranks_.size());
        args.rank         = static_cast<int>(rank);
        args.contributors = static_cast<int>(contributors);
        exchange_words(args, stream, sum);
        return;
    }
    const bool scatter     = route.form == TensorParallelForm::Scatter;
    const ValueRange range = scatter ? scatter_range(rank, ranks_.size(), count)
                                     : ValueRange{0, count};
    const std::int64_t place = inbox_offset(piece, route.form);
    std::array<const float*, kMaxRanks> parts{};
    std::array<std::int64_t, kMaxRanks> firsts{};
    for (std::size_t r = 0; r < ranks_.size(); ++r) {
        if (r == rank) {
            parts[r] = self.device_slot[slot] + piece.offset;
            continue;
        }
        check(cudaStreamWaitEvent(stream, ranks_[r].published[slot][piece.index], 0),
              "reduce wait");
        switch (route.transport) {
        case TensorParallelTransport::PeerRead:
            parts[r] = ranks_[r].device_slot[slot] + piece.offset;
            break;
        case TensorParallelTransport::PeerWrite:
            parts[r]  = inbox_row(rank, r, slot) + place;
            firsts[r] = range.begin;
            break;
        default:
            parts[r] = ranks_[r].host_slot[slot] + piece.offset;
            break;
        }
    }
    reduce_values(out, parts.data(), firsts.data(), contributors, range.begin, range.end, stream,
                  sum);
    if (!scatter) { return; }
    if (out != self.output[slot][piece.index]) {
        throw std::logic_error("a tensor-parallel reduce writes another output than its publish named");
    }
    self.reduced_ready[slot][piece.index] = true;
    const std::size_t size   = output_bytes(sum);
    const auto begin         = static_cast<std::size_t>(range.begin);
    const auto length        = static_cast<std::size_t>(range.end - range.begin);
    if (length != 0 && route.transport == TensorParallelTransport::PeerWrite) {
        // The finished range into every other rank's output.
        std::array<void*, kMaxRanks> destinations{};
        std::array<const void*, kMaxRanks> sources{};
        std::array<std::int64_t, kMaxRanks> sizes{};
        std::size_t copies = 0;
        for (std::size_t other = 0; other < ranks_.size(); ++other) {
            if (other == rank) { continue; }
            destinations[copies] =
                static_cast<unsigned char*>(ranks_[other].output[slot][piece.index]) + begin * size;
            sources[copies]      = static_cast<const unsigned char*>(out) + begin * size;
            sizes[copies]        = static_cast<std::int64_t>(length * size);
            ++copies;
        }
        copy_rows(destinations.data(), sources.data(), sizes.data(), copies, stream);
    }
    if (length != 0 && route.transport == TensorParallelTransport::Host) {
        check(cudaMemcpyAsync(self.host_slot[slot] + piece.offset + range.begin,
                              static_cast<const unsigned char*>(out) + begin * size, length * size,
                              cudaMemcpyDeviceToHost, stream),
              "reduce copy");
    }
    check(cudaEventRecord(self.reduced[slot][piece.index], stream), "reduce fence");
}

void TensorParallelExchange::complete_on(TensorParallelRoute route, std::size_t rank,
                                         std::size_t slot, void* out, std::int64_t count,
                                         cudaStream_t stream, TensorParallelSum sum,
                                         TensorParallelPiece piece) {
    if (route.form != TensorParallelForm::Scatter) { return; }
    DeviceBinding bind(ranks_[rank].device);
    const std::size_t size = output_bytes(sum);
    std::array<const void*, kMaxRanks> sources{};
    std::array<std::int64_t, kMaxRanks> offsets{};
    std::array<std::int64_t, kMaxRanks> bytes{};
    std::size_t copies = 0;
    for (std::size_t owner = 0; owner < ranks_.size(); ++owner) {
        const ValueRange range = scatter_range(owner, ranks_.size(), count);
        if (owner == rank || range.end == range.begin) { continue; }
        const Rank& from = ranks_[owner];
        if (!from.reduced_ready[slot][piece.index]) {
            throw std::logic_error("tensor-parallel complete before every rank's reduce");
        }
        check(cudaStreamWaitEvent(stream, from.reduced[slot][piece.index], 0), "complete wait");
        if (route.transport == TensorParallelTransport::PeerWrite) { continue; }
        const auto begin = static_cast<std::size_t>(range.begin);
        sources[copies]  = route.transport == TensorParallelTransport::PeerRead
                               ? static_cast<const unsigned char*>(from.output[slot][piece.index]) +
                                    begin * size
                               : static_cast<const void*>(from.host_slot[slot] + piece.offset +
                                                          range.begin);
        offsets[copies] = range.begin * static_cast<std::int64_t>(size);
        bytes[copies]   = (range.end - range.begin) * static_cast<std::int64_t>(size);
        ++copies;
    }
    gather_ranges(out, sources.data(), offsets.data(), bytes.data(), copies, stream);
}

void TensorParallelExchange::publish_whole(std::size_t rank, std::size_t slot, std::size_t bytes,
                                           cudaStream_t stream) {
    check_slot(rank, slot, bytes);
    publish_on(gather_route(bytes), rank, slot, bytes, nullptr, stream,
               TensorParallelSum::AddBf16);
}

void TensorParallelExchange::gather(std::size_t rank, std::size_t slot, std::size_t bytes,
                                    std::span<const float*> parts, cudaStream_t stream) {
    check_slot(rank, slot, bytes);
    if (parts.size() != ranks_.size()) {
        throw std::invalid_argument("tensor-parallel gather needs one pointer per rank");
    }
    const TensorParallelRoute route = gather_route(bytes);
    DeviceBinding bind(ranks_[rank].device);
    for (std::size_t r = 0; r < ranks_.size(); ++r) {
        if (r == rank) {
            parts[r] = ranks_[r].device_slot[slot];
            continue;
        }
        check(cudaStreamWaitEvent(stream, ranks_[r].published[slot][0], 0), "gather wait");
        parts[r] = route.transport == TensorParallelTransport::PeerRead ? ranks_[r].device_slot[slot]
                   : route.transport == TensorParallelTransport::PeerWrite
                       ? inbox_row(rank, r, slot)
                       : ranks_[r].host_slot[slot];
    }
}

} // namespace ninfer
