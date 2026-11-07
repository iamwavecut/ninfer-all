#include "core/tensor_parallel.h"

#include "core/device_wait.h"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ninfer {
namespace {

void check(cudaError_t err, const char* what) {
    if (err == cudaSuccess) { return; }
    throw std::runtime_error(std::string("TensorParallelExchange ") + what + ": " +
                             cudaGetErrorName(err) + ": " + cudaGetErrorString(err));
}

// Value i of partial r is value i - first[r] of part[r]: an inbox row holds an owner's range from
// its start. The partials are FP32, or rounded to BF16 or 8-bit blocks (see Rounded partials).
template <class Part>
struct PartialSet {
    const Part* part[kMaxRanks];
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

// 32 values of an 8-bit partial: value k is code[k] * scale.
struct Q8Block {
    float scale;
    std::int8_t code[32];
};

static_assert(sizeof(Q8Block) == 36 && alignof(Q8Block) == 4);
constexpr std::int64_t kQ8Values = 32;

// Bytes of the first `values` values of a partial stored as `partials`: of every value for FP32
// and BF16, of every 32-value block begun for 8-bit blocks. From a multiple of 32 it is also where
// that value's bytes begin.
std::size_t partial_bytes(TensorParallelPartials partials, std::int64_t values) noexcept {
    const auto n = static_cast<std::size_t>(values);
    switch (partials) {
    case TensorParallelPartials::Bf16:
        return n * sizeof(__nv_bfloat16);
    case TensorParallelPartials::Int8:
        return (n + kQ8Values - 1) / kQ8Values * sizeof(Q8Block);
    default:
        return n * sizeof(float);
    }
}

// The bytes of a rank's rounded copy of a slot of `values` FP32 values: room for either rounding.
std::size_t rounded_bytes(std::size_t values) noexcept {
    return std::max(partial_bytes(TensorParallelPartials::Bf16, static_cast<std::int64_t>(values)),
                    partial_bytes(TensorParallelPartials::Int8, static_cast<std::int64_t>(values)));
}

// Value k of a partial.
__device__ __forceinline__ float load_value(const float* part, std::int64_t k) { return part[k]; }

__device__ __forceinline__ float load_value(const __nv_bfloat16* part, std::int64_t k) {
    return __bfloat162float(part[k]);
}

// An 8-bit value is one correctly rounded product, never contracted into the sum that adds it.
__device__ __forceinline__ float load_value(const Q8Block* part, std::int64_t k) {
    const Q8Block& block = part[k / kQ8Values];
    return __fmul_rn(static_cast<float>(block.code[k % kQ8Values]), block.scale);
}

// Four values from a multiple of four, aligned for one load.
__device__ __forceinline__ float4 load_value4(const float* part, std::int64_t k) {
    return *reinterpret_cast<const float4*>(part + k);
}

__device__ __forceinline__ float4 load_value4(const __nv_bfloat16* part, std::int64_t k) {
    const Bf16x4 word = *reinterpret_cast<const Bf16x4*>(part + k);
    return {__bfloat162float(word.v[0]), __bfloat162float(word.v[1]), __bfloat162float(word.v[2]),
            __bfloat162float(word.v[3])};
}

__device__ __forceinline__ float4 load_value4(const Q8Block* part, std::int64_t k) {
    const Q8Block& block = part[k / kQ8Values];
    const char4 codes    = *reinterpret_cast<const char4*>(block.code + k % kQ8Values);
    return {__fmul_rn(static_cast<float>(codes.x), block.scale),
            __fmul_rn(static_cast<float>(codes.y), block.scale),
            __fmul_rn(static_cast<float>(codes.z), block.scale),
            __fmul_rn(static_cast<float>(codes.w), block.scale)};
}

// The alignment a partial's pointer needs for load_value4.
template <class Part>
constexpr std::size_t vector_alignment() {
    if constexpr (std::is_same_v<Part, Q8Block>) {
        return alignof(Q8Block);
    } else {
        return 4 * sizeof(Part);
    }
}

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
template <TensorParallelSum Sum, class Part>
__global__ void tensor_parallel_reduce_kernel(void* __restrict__ out, PartialSet<Part> parts,
                                              std::int64_t begin, std::int64_t end, bool vector) {
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t thread = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::int64_t tail         = begin;
    if (vector) {
        const std::int64_t quads = (end - begin) / 4;
        for (std::int64_t q = thread; q < quads; q += stride) {
            const std::int64_t i = begin + 4 * q;
            float4 sum           = load_value4(parts.part[0], i - parts.first[0]);
            for (int r = 1; r < parts.count; ++r) {
                const float4 part = load_value4(parts.part[r], i - parts.first[r]);
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
        float sum = load_value(parts.part[0], i - parts.first[0]);
        for (int r = 1; r < parts.count; ++r) {
            sum += load_value(parts.part[r], i - parts.first[r]);
        }
        store_sum<Sum>(out, i, sum);
    }
}

// out[i] <- bf16(in[i]) for i < count, four values at a time where `vector` says both allow it.
__global__ void tensor_parallel_round_kernel(const float* __restrict__ in,
                                             __nv_bfloat16* __restrict__ out, std::int64_t count,
                                             bool vector) {
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t thread = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::int64_t tail         = 0;
    if (vector) {
        const std::int64_t quads = count / 4;
        for (std::int64_t q = thread; q < quads; q += stride) {
            const float4 value = reinterpret_cast<const float4*>(in)[q];
            reinterpret_cast<Bf16x4*>(out)[q] =
                Bf16x4{{__float2bfloat16(value.x), __float2bfloat16(value.y),
                        __float2bfloat16(value.z), __float2bfloat16(value.w)}};
        }
        tail = 4 * quads;
    }
    for (std::int64_t i = tail + thread; i < count; i += stride) {
        out[i] = __float2bfloat16(in[i]);
    }
}

// Each 32-value block of in[0, count) as 8-bit codes and a scale, a warp a block: the scale is the
// block's largest magnitude over 127 and each code its value over the scale, rounded to the
// nearest integer (ties to even), both by correctly rounded division; values past `count` are
// zeros. A block holding an infinity or a NaN takes a NaN scale, which its sums carry on.
__global__ void tensor_parallel_quantize_kernel(const float* __restrict__ in,
                                                Q8Block* __restrict__ out, std::int64_t count) {
    const int lane            = static_cast<int>(threadIdx.x % 32U);
    const std::int64_t warps  = static_cast<std::int64_t>(gridDim.x) * blockDim.x / 32;
    const std::int64_t blocks = (count + kQ8Values - 1) / kQ8Values;
    for (std::int64_t b = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
         b < blocks; b += warps) {
        const std::int64_t i = b * kQ8Values + lane;
        const float value    = i < count ? in[i] : 0.0f;
        float largest        = fabsf(value);
        for (int offset = 16; offset > 0; offset /= 2) {
            largest = fmaxf(largest, __shfl_xor_sync(0xffffffffU, largest, offset));
        }
        const bool finite = __all_sync(0xffffffffU, isfinite(value));
        const float scale = finite ? __fdiv_rn(largest, 127.0f) : __int_as_float(0x7fc00000);
        int code          = 0;
        if (finite && largest != 0.0f) {
            code = max(-127, min(127, __float2int_rn(__fdiv_rn(value, scale))));
        }
        out[b].code[lane] = static_cast<std::int8_t>(code);
        if (lane == 0) { out[b].scale = scale; }
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

// The blocks of a scatter-flags kernel, at every size: on four RTX 5060 Ti without peer access
// (2026-11-03) two or three took 40.5-41.9 us at 80 KiB and 202-204 us at 640 KiB, one 49.0 and
// 255.8 us, a block per 4096 values (up to 16) 46.0 and 228.3 us, and Scatter 50 and 220 us; the
// reads across the bus go no faster from more blocks, which only raise and poll more flags.
constexpr int kFlagBlocks = 2;

// A scatter-flags area (Rank::scatter_flags) holds per slot a row of the rank's finished outputs,
// `row` floats long, then the flags of both phases: one per slot and block.
__device__ __forceinline__ unsigned char* flags_finished_row(unsigned char* area, std::int64_t row,
                                                             std::int64_t slot) {
    return area + slot * row * static_cast<std::int64_t>(sizeof(float));
}

__device__ __forceinline__ std::uint32_t* flags_of(unsigned char* area, std::int64_t row,
                                                   int phase, std::int64_t slot) {
    auto* flags = reinterpret_cast<std::uint32_t*>(
        area + static_cast<std::int64_t>(TensorParallelExchange::kSlots) * row *
                   static_cast<std::int64_t>(sizeof(float)));
    return flags + (phase * static_cast<std::int64_t>(TensorParallelExchange::kSlots) + slot) *
                       kFlagBlocks;
}

struct ScatterFlagsArgs {
    void* out;
    // Every rank's partial: this rank's in its device slot, the others' in their host slots.
    const float* parts[kMaxRanks];
    // Every rank's mapped host area, this rank's at `rank`.
    unsigned char* areas[kMaxRanks];
    std::uint32_t* epoch;
    // Every rank's range of the values.
    std::int64_t begin[kMaxRanks];
    std::int64_t end[kMaxRanks];
    std::int64_t row;
    int ranks;
    int rank;
    int contributors;
    // `out` is aligned to 16 bytes; the slots and areas always are.
    bool vector;
};

// Block `block` of `blocks`' part of the values [begin, end): whole 64-value blocks, as even as
// they divide, so every part starts aligned.
__device__ __forceinline__ void block_part(std::int64_t begin, std::int64_t end, int block,
                                           int blocks, std::int64_t& from, std::int64_t& to) {
    const std::int64_t step = ((end - begin + blocks - 1) / blocks + 63) / 64 * 64;
    from                    = begin + block * step < end ? begin + block * step : end;
    to                      = from + step < end ? from + step : end;
}

// Thread r of the block waits until rank r's flag of this block in `phase` carries `epoch`, for
// every rank r in `ranks`; then every thread of the block may read what those ranks stored before
// raising their flags.
__device__ __forceinline__ void await_flags(unsigned char* const* areas, std::int64_t row,
                                            int phase, std::int64_t slot, std::uint32_t epoch,
                                            unsigned ranks) {
    const unsigned r = threadIdx.x;
    if (r < kMaxRanks && ((ranks >> r) & 1U) != 0) {
        const volatile std::uint32_t* flag = flags_of(areas[r], row, phase, slot) + blockIdx.x;
        const unsigned long long begun     = words_clock();
        for (std::uint32_t spins = 0; *flag != epoch;) {
            // A poll is a read across the bus; spaced out, the polls leave it to the data
            // (41.0-41.9 us at 80 KiB against 42.5-43.0 without the pause).
            __nanosleep(200);
            if ((++spins & 1023U) == 0 &&
                words_clock() - begun > TensorParallelExchange::kWordsTimeoutNs) {
                printf("ninfer: a tensor-parallel exchange waited %llu s for epoch %u\n",
                       TensorParallelExchange::kWordsTimeoutNs / 1'000'000'000ULL, epoch);
                __trap();
            }
        }
        __threadfence_system();
    }
    __syncthreads();
}

// Raises this block's flag in `phase` once every thread's stores before it have reached host
// memory.
__device__ __forceinline__ void raise_flag(unsigned char* area, std::int64_t row, int phase,
                                           std::int64_t slot, std::uint32_t epoch) {
    __syncthreads();
    if (threadIdx.x == 0) {
        // Cumulative: it orders every thread's stores before the barrier.
        __threadfence_system();
        *reinterpret_cast<volatile std::uint32_t*>(flags_of(area, row, phase, slot) + blockIdx.x) =
            epoch;
    }
}

// Values [begin, end) of `Size` bytes each from host memory into `to`, fetched past every cache,
// sixteen bytes at a time where `vector` allows.
template <int Size>
__device__ __forceinline__ void block_fetch(unsigned char* to, const unsigned char* from,
                                            std::int64_t begin, std::int64_t end, bool vector) {
    const std::int64_t first = begin * Size;
    const std::int64_t last  = end * Size;
    std::int64_t tail        = first;
    if (vector) {
        const std::int64_t lines = (last - first) / 16;
        for (std::int64_t q = threadIdx.x; q < lines; q += blockDim.x) {
            *reinterpret_cast<uint4*>(to + first + 16 * q) =
                __ldcv(reinterpret_cast<const uint4*>(from + first + 16 * q));
        }
        tail = first + 16 * lines;
    }
    for (std::int64_t b = tail + Size * static_cast<std::int64_t>(threadIdx.x); b < last;
         b += Size * static_cast<std::int64_t>(blockDim.x)) {
        if constexpr (Size == 2) {
            *reinterpret_cast<unsigned short*>(to + b) =
                __ldcv(reinterpret_cast<const unsigned short*>(from + b));
        } else {
            *reinterpret_cast<unsigned int*>(to + b) =
                __ldcv(reinterpret_cast<const unsigned int*>(from + b));
        }
    }
}

// Value i's sum stored into the output as store_sum stores it, and into the finished row.
template <TensorParallelSum Sum>
__device__ __forceinline__ void finish_sum(void* out, unsigned char* finished, std::int64_t i,
                                           float sum) {
    if constexpr (Sum == TensorParallelSum::StoreFp32) {
        static_cast<float*>(out)[i]           = sum;
        reinterpret_cast<float*>(finished)[i] = sum;
    } else {
        auto* values = static_cast<__nv_bfloat16*>(out);
        const __nv_bfloat16 value =
            Sum == TensorParallelSum::AddBf16 ? __float2bfloat16(__bfloat162float(values[i]) + sum)
                                              : __float2bfloat16(sum);
        values[i]                                     = value;
        reinterpret_cast<__nv_bfloat16*>(finished)[i] = value;
    }
}

// Four values' sums from a multiple of four, as finish_sum stores them.
template <TensorParallelSum Sum>
__device__ __forceinline__ void finish_sum4(void* out, unsigned char* finished, std::int64_t i,
                                            float4 sum) {
    if constexpr (Sum == TensorParallelSum::StoreFp32) {
        *reinterpret_cast<float4*>(static_cast<float*>(out) + i)           = sum;
        *reinterpret_cast<float4*>(reinterpret_cast<float*>(finished) + i) = sum;
    } else {
        auto* residual = reinterpret_cast<Bf16x4*>(static_cast<__nv_bfloat16*>(out) + i);
        Bf16x4 word;
        if constexpr (Sum == TensorParallelSum::AddBf16) {
            word      = *residual;
            word.v[0] = __float2bfloat16(__bfloat162float(word.v[0]) + sum.x);
            word.v[1] = __float2bfloat16(__bfloat162float(word.v[1]) + sum.y);
            word.v[2] = __float2bfloat16(__bfloat162float(word.v[2]) + sum.z);
            word.v[3] = __float2bfloat16(__bfloat162float(word.v[3]) + sum.w);
        } else {
            word = Bf16x4{{__float2bfloat16(sum.x), __float2bfloat16(sum.y),
                           __float2bfloat16(sum.z), __float2bfloat16(sum.w)}};
        }
        *residual                                                                  = word;
        *reinterpret_cast<Bf16x4*>(reinterpret_cast<__nv_bfloat16*>(finished) + i) = word;
    }
}

// One rank's scatter-flags exchange; block b of every rank handles part b of every range (see
// ScatterFlags in core/tensor_parallel.h). The publish's copy of a contributor's partial into its
// host slot ended before the kernel began, so every block raises its first flag at once; the rank
// then sums its own part from the contributors' partials in rank order into `out` and its finished
// row, raises its second flag, and fetches the other ranks' finished parts into `out`.
template <TensorParallelSum Sum>
__global__ void tensor_parallel_scatter_flags_kernel(ScatterFlagsArgs a) {
    constexpr int kSize = Sum == TensorParallelSum::StoreFp32 ? 4 : 2;
    const std::uint32_t epoch =
        next_words_epoch(*reinterpret_cast<const volatile std::uint32_t*>(a.epoch));
    const auto slot           = static_cast<std::int64_t>(epoch & 1U);
    const int blocks          = static_cast<int>(gridDim.x);
    const int block           = static_cast<int>(blockIdx.x);
    unsigned char* const area = a.areas[a.rank];
    std::int64_t from         = 0;
    std::int64_t to           = 0;
    if (a.rank < a.contributors && threadIdx.x == 0) {
        *reinterpret_cast<volatile std::uint32_t*>(flags_of(area, a.row, 0, slot) + blockIdx.x) =
            epoch;
    }
    unsigned contributors = 0;
    unsigned others       = 0;
    for (int r = 0; r < a.ranks; ++r) {
        if (r == a.rank) { continue; }
        others |= 1U << r;
        if (r < a.contributors) { contributors |= 1U << r; }
    }
    await_flags(a.areas, a.row, 0, slot, epoch, contributors);
    block_part(a.begin[a.rank], a.end[a.rank], block, blocks, from, to);
    unsigned char* const finished = flags_finished_row(area, a.row, slot);
    const std::int64_t quads      = a.vector ? (to - from) / 4 : 0;
    for (std::int64_t q = threadIdx.x; q < quads; q += blockDim.x) {
        const std::int64_t i = from + 4 * q;
        float4 sum{};
        for (int r = 0; r < a.contributors; ++r) {
            const auto* at    = reinterpret_cast<const float4*>(a.parts[r] + i);
            const float4 part = r == a.rank ? *at : __ldcv(at);
            if (r == 0) {
                sum = part;
            } else {
                sum.x += part.x;
                sum.y += part.y;
                sum.z += part.z;
                sum.w += part.w;
            }
        }
        finish_sum4<Sum>(a.out, finished, i, sum);
    }
    for (std::int64_t i = from + 4 * quads + threadIdx.x; i < to; i += blockDim.x) {
        float sum = 0.0F;
        for (int r = 0; r < a.contributors; ++r) {
            const float part = r == a.rank ? a.parts[r][i] : __ldcv(a.parts[r] + i);
            sum = r == 0 ? part : sum + part;
        }
        finish_sum<Sum>(a.out, finished, i, sum);
    }
    raise_flag(area, a.row, 1, slot, epoch);
    await_flags(a.areas, a.row, 1, slot, epoch, others);
    for (int o = 0; o < a.ranks; ++o) {
        if (o == a.rank) { continue; }
        block_part(a.begin[o], a.end[o], block, blocks, from, to);
        block_fetch<kSize>(static_cast<unsigned char*>(a.out),
                           flags_finished_row(a.areas[o], a.row, slot), from, to, a.vector);
    }
    // As in the words kernel: the last block to finish advances the count.
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(a.epoch + 1, 1U) + 1U == gridDim.x) {
            a.epoch[1] = 0;
            a.epoch[0] = epoch;
        }
    }
}

// The most blocks of a peer scatter-flags kernel: its stores and reads cross the links from as
// many SMs, and its polls stay on the rank's own card.
constexpr int kPeerFlagBlocks = 16;

// A peer scatter-flags area (Rank::peer_flags), in the rank's device memory, holds per slot a row
// of the rank's finished outputs, `row` floats long (PeerRead), then the flags the other ranks
// raise in it: per phase, slot, sender and block.
__device__ __forceinline__ std::uint32_t* peer_flags_of(unsigned char* area, std::int64_t row,
                                                        int phase, std::int64_t slot, int sender) {
    auto* flags = reinterpret_cast<std::uint32_t*>(
        area + static_cast<std::int64_t>(TensorParallelExchange::kSlots) * row *
                   static_cast<std::int64_t>(sizeof(float)));
    return flags +
           ((phase * static_cast<std::int64_t>(TensorParallelExchange::kSlots) + slot) * kMaxRanks +
            sender) *
               kPeerFlagBlocks;
}

struct PeerFlagsArgs {
    void* out;
    // PeerWrite: every rank's output, which its publish named.
    void* outs[kMaxRanks];
    // Every contributor's partial: this rank's in its device slot; under PeerRead the others' in
    // their device slots, under PeerWrite their owner's ranges in this rank's inbox, from value
    // first[r].
    const float* parts[kMaxRanks];
    std::int64_t first[kMaxRanks];
    // PeerWrite: this rank's row in every other rank's inbox.
    float* rows[kMaxRanks];
    // Every rank's peer flags area.
    unsigned char* areas[kMaxRanks];
    std::uint32_t* epoch;
    // Every rank's range of the values.
    std::int64_t begin[kMaxRanks];
    std::int64_t end[kMaxRanks];
    std::int64_t row;
    int ranks;
    int rank;
    int contributors;
    // Every output and partial is aligned to 16 bytes and every range start to four values.
    bool vector;
};

// Raises this block's flag in `phase` in the area of every rank in `ranks`, once every thread's
// stores before it have reached those ranks: thread t raises rank t's.
__device__ __forceinline__ void raise_peer_flags(unsigned char* const* areas, std::int64_t row,
                                                 int phase, std::int64_t slot, int sender,
                                                 std::uint32_t epoch, unsigned ranks) {
    __syncthreads();
    const unsigned t = threadIdx.x;
    if (t < kMaxRanks && ((ranks >> t) & 1U) != 0) {
        // Cumulative: it orders every thread's stores before the barrier.
        __threadfence_system();
        *reinterpret_cast<volatile std::uint32_t*>(
            peer_flags_of(areas[t], row, phase, slot, sender) + blockIdx.x) = epoch;
    }
}

// Thread r of the block waits until rank r's flag of this block in `phase` in this rank's own
// area carries `epoch`, for every rank r in `ranks`; then every thread of the block may read what
// those ranks stored before raising their flags.
__device__ __forceinline__ void await_peer_flags(unsigned char* area, std::int64_t row, int phase,
                                                 std::int64_t slot, std::uint32_t epoch,
                                                 unsigned ranks) {
    const unsigned r = threadIdx.x;
    if (r < kMaxRanks && ((ranks >> r) & 1U) != 0) {
        const volatile std::uint32_t* flag =
            peer_flags_of(area, row, phase, slot, static_cast<int>(r)) + blockIdx.x;
        const unsigned long long begun = words_clock();
        for (std::uint32_t spins = 0; *flag != epoch;) {
            if ((++spins & 1023U) == 0 &&
                words_clock() - begun > TensorParallelExchange::kWordsTimeoutNs) {
                printf("ninfer: a tensor-parallel exchange waited %llu s for epoch %u\n",
                       TensorParallelExchange::kWordsTimeoutNs / 1'000'000'000ULL, epoch);
                __trap();
            }
        }
        __threadfence_system();
    }
    __syncthreads();
}

// One rank's scatter-flags exchange with peer access (see ScatterFlags in core/tensor_parallel.h);
// block b of every rank handles part b of every range. Under PeerWrite (`Write`) a contributor
// first stores its part of every other owner's range of its partial into its row of that owner's
// inbox; every block then raises its first flag in every other rank's area, which says that its
// rank's kernel has begun and its partial (under PeerRead) or its stores (under PeerWrite) are
// there, waits there for its own flags from the other ranks (under PeerRead the contributors',
// under PeerWrite all, since it writes their outputs next), and sums its part of its rank's range
// from the contributors' partials in rank order into `out` and either every other rank's output
// (PeerWrite) or its own finished row (PeerRead). It raises its second flag in every other rank's
// area, waits for theirs and, under PeerRead, reads their finished parts into `out`.
template <TensorParallelSum Sum, bool Write>
__global__ void tensor_parallel_peer_flags_kernel(PeerFlagsArgs a) {
    constexpr int kSize = Sum == TensorParallelSum::StoreFp32 ? 4 : 2;
    const std::uint32_t epoch =
        next_words_epoch(*reinterpret_cast<const volatile std::uint32_t*>(a.epoch));
    const auto slot       = static_cast<std::int64_t>(epoch & 1U);
    const int blocks      = static_cast<int>(gridDim.x);
    const int block       = static_cast<int>(blockIdx.x);
    unsigned char* own    = a.areas[a.rank];
    unsigned contributors = 0;
    unsigned others       = 0;
    for (int r = 0; r < a.ranks; ++r) {
        if (r == a.rank) { continue; }
        others |= 1U << r;
        if (r < a.contributors) { contributors |= 1U << r; }
    }
    std::int64_t from = 0;
    std::int64_t to   = 0;
    if constexpr (Write) {
        if (a.rank < a.contributors) {
            const float* mine = a.parts[a.rank];
            for (int o = 0; o < a.ranks; ++o) {
                if (o == a.rank) { continue; }
                block_part(a.begin[o], a.end[o], block, blocks, from, to);
                float* row               = a.rows[o];
                const std::int64_t first = a.begin[o];
                const std::int64_t quads = a.vector ? (to - from) / 4 : 0;
                for (std::int64_t q = threadIdx.x; q < quads; q += blockDim.x) {
                    const std::int64_t i = from + 4 * q;
                    *reinterpret_cast<float4*>(row + (i - first)) =
                        *reinterpret_cast<const float4*>(mine + i);
                }
                for (std::int64_t i = from + 4 * quads + threadIdx.x; i < to; i += blockDim.x) {
                    row[i - first] = mine[i];
                }
            }
        }
    }
    raise_peer_flags(a.areas, a.row, 0, slot, a.rank, epoch, others);
    await_peer_flags(own, a.row, 0, slot, epoch, Write ? others : contributors);
    block_part(a.begin[a.rank], a.end[a.rank], block, blocks, from, to);
    unsigned char* const finished = own + slot * a.row * static_cast<std::int64_t>(sizeof(float));
    const auto keep               = [&](std::int64_t i, float sum) {
        finish_sum<Sum>(a.out, finished, i, sum);
        if constexpr (Write) {
            const unsigned char* value = finished + i * kSize;
            for (int o = 0; o < a.ranks; ++o) {
                if (o == a.rank) { continue; }
                if constexpr (kSize == 4) {
                    static_cast<float*>(a.outs[o])[i] = *reinterpret_cast<const float*>(value);
                } else {
                    static_cast<__nv_bfloat16*>(a.outs[o])[i] =
                        *reinterpret_cast<const __nv_bfloat16*>(value);
                }
            }
        }
    };
    const std::int64_t quads = a.vector ? (to - from) / 4 : 0;
    for (std::int64_t q = threadIdx.x; q < quads; q += blockDim.x) {
        const std::int64_t i = from + 4 * q;
        float4 sum{};
        for (int r = 0; r < a.contributors; ++r) {
            const auto* at    = reinterpret_cast<const float4*>(a.parts[r] + (i - a.first[r]));
            const float4 part = r == a.rank ? *at : __ldcv(at);
            if (r == 0) {
                sum = part;
            } else {
                sum.x += part.x;
                sum.y += part.y;
                sum.z += part.z;
                sum.w += part.w;
            }
        }
        finish_sum4<Sum>(a.out, finished, i, sum);
        if constexpr (Write) {
            for (int o = 0; o < a.ranks; ++o) {
                if (o == a.rank) { continue; }
                if constexpr (kSize == 4) {
                    *reinterpret_cast<float4*>(static_cast<float*>(a.outs[o]) + i) =
                        *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(finished) +
                                                         i);
                } else {
                    *reinterpret_cast<Bf16x4*>(static_cast<__nv_bfloat16*>(a.outs[o]) + i) =
                        *reinterpret_cast<const Bf16x4*>(
                            reinterpret_cast<const __nv_bfloat16*>(finished) + i);
                }
            }
        }
    }
    for (std::int64_t i = from + 4 * quads + threadIdx.x; i < to; i += blockDim.x) {
        float sum = 0.0F;
        for (int r = 0; r < a.contributors; ++r) {
            const float* at  = a.parts[r] + (i - a.first[r]);
            const float part = r == a.rank ? *at : __ldcv(at);
            sum              = r == 0 ? part : sum + part;
        }
        keep(i, sum);
    }
    raise_peer_flags(a.areas, a.row, 1, slot, a.rank, epoch, others);
    await_peer_flags(own, a.row, 1, slot, epoch, others);
    if constexpr (!Write) {
        for (int o = 0; o < a.ranks; ++o) {
            if (o == a.rank) { continue; }
            block_part(a.begin[o], a.end[o], block, blocks, from, to);
            block_fetch<kSize>(static_cast<unsigned char*>(a.out),
                               a.areas[o] + slot * a.row * static_cast<std::int64_t>(sizeof(float)),
                               from, to, a.vector);
        }
    }
    // As in the words kernel: the last block to finish advances the count.
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
// sum stored as BF16 or FP32 (`sum`); value i of partial r is parts[r][i - firsts[r]], FP32 or
// rounded to BF16 (`Part`).
template <class Part>
void reduce_values(void* out, const Part* const* parts, const std::int64_t* firsts,
                   std::size_t count, std::int64_t begin, std::int64_t end, cudaStream_t stream,
                   TensorParallelSum sum) {
    if (end == begin) { return; }
    PartialSet<Part> set{};
    bool vector = begin % 4 == 0 && aligned(out, 4 * output_bytes(sum));
    for (std::size_t r = 0; r < count; ++r) {
        set.part[r]  = parts[r];
        set.first[r] = firsts[r];
        vector       = vector && aligned(parts[r], vector_alignment<Part>()) && firsts[r] % 4 == 0;
    }
    set.count                 = static_cast<int>(count);
    constexpr int kBlock      = 256;
    const std::int64_t items  = vector ? (end - begin + 3) / 4 : end - begin;
    const std::int64_t blocks = std::min<std::int64_t>((items + kBlock - 1) / kBlock, 4096);
    const auto grid           = static_cast<unsigned>(blocks);
    switch (sum) {
    case TensorParallelSum::AddBf16:
        tensor_parallel_reduce_kernel<TensorParallelSum::AddBf16, Part>
            <<<grid, kBlock, 0, stream>>>(out, set, begin, end, vector);
        break;
    case TensorParallelSum::StoreBf16:
        tensor_parallel_reduce_kernel<TensorParallelSum::StoreBf16, Part>
            <<<grid, kBlock, 0, stream>>>(out, set, begin, end, vector);
        break;
    case TensorParallelSum::StoreFp32:
        tensor_parallel_reduce_kernel<TensorParallelSum::StoreFp32, Part>
            <<<grid, kBlock, 0, stream>>>(out, set, begin, end, vector);
        break;
    }
    check(cudaGetLastError(), "reduce launch");
}

// in[0, count) rounded as `partials` (BF16 or 8-bit blocks) into `out`.
void round_values(const float* in, void* out, std::int64_t count, TensorParallelPartials partials,
                  cudaStream_t stream) {
    if (count == 0) { return; }
    if (partials == TensorParallelPartials::Int8) {
        constexpr int kThreads   = 256;
        const std::int64_t warps = (count + kQ8Values - 1) / kQ8Values;
        const std::int64_t blocks =
            std::min<std::int64_t>((warps * 32 + kThreads - 1) / kThreads, 4096);
        tensor_parallel_quantize_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
            in, static_cast<Q8Block*>(out), count);
        check(cudaGetLastError(), "quantize launch");
        return;
    }
    const bool vector         = aligned(in, sizeof(float4)) && aligned(out, sizeof(Bf16x4));
    constexpr int kBlock      = 256;
    const std::int64_t items  = vector ? (count + 3) / 4 : count;
    const std::int64_t blocks = std::min<std::int64_t>((items + kBlock - 1) / kBlock, 4096);
    tensor_parallel_round_kernel<<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(
        in, static_cast<__nv_bfloat16*>(out), count, vector);
    check(cudaGetLastError(), "round launch");
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
// take them where each is at least `engine_bytes`, from which they stream faster than kernels store
// across the links; one kernel takes smaller ones, all at once, without a copy's fixed cost each.
void copy_rows(void* const* destinations, const void* const* sources, const std::int64_t* bytes,
               std::size_t count, cudaStream_t stream, std::int64_t engine_bytes) {
    bool engine = true;
    for (std::size_t i = 0; i < count; ++i) {
        engine = engine && (bytes[i] == 0 || bytes[i] >= engine_bytes);
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

void exchange_scatter_flags(const ScatterFlagsArgs& args, cudaStream_t stream,
                            TensorParallelSum sum) {
    constexpr int kBlock = 256;
    const auto grid      = static_cast<unsigned>(kFlagBlocks);
    switch (sum) {
    case TensorParallelSum::AddBf16:
        tensor_parallel_scatter_flags_kernel<TensorParallelSum::AddBf16>
            <<<grid, kBlock, 0, stream>>>(args);
        break;
    case TensorParallelSum::StoreBf16:
        tensor_parallel_scatter_flags_kernel<TensorParallelSum::StoreBf16>
            <<<grid, kBlock, 0, stream>>>(args);
        break;
    case TensorParallelSum::StoreFp32:
        tensor_parallel_scatter_flags_kernel<TensorParallelSum::StoreFp32>
            <<<grid, kBlock, 0, stream>>>(args);
        break;
    }
    check(cudaGetLastError(), "scatter flags exchange launch");
}

// The blocks of a peer scatter-flags exchange of `count` values over `ranks` ranks: one per 1024
// values of a range, at least one and at most kPeerFlagBlocks; every rank derives the same.
int peer_flag_blocks(std::int64_t count, std::size_t ranks) {
    const std::int64_t range =
        (count + static_cast<std::int64_t>(ranks) - 1) / static_cast<std::int64_t>(ranks);
    return static_cast<int>(std::clamp<std::int64_t>((range + 1023) / 1024, 1, kPeerFlagBlocks));
}

template <bool Write>
void exchange_peer_flags(const PeerFlagsArgs& args, int blocks, cudaStream_t stream,
                         TensorParallelSum sum) {
    constexpr int kBlock = 256;
    const auto grid      = static_cast<unsigned>(blocks);
    switch (sum) {
    case TensorParallelSum::AddBf16:
        tensor_parallel_peer_flags_kernel<TensorParallelSum::AddBf16, Write>
            <<<grid, kBlock, 0, stream>>>(args);
        break;
    case TensorParallelSum::StoreBf16:
        tensor_parallel_peer_flags_kernel<TensorParallelSum::StoreBf16, Write>
            <<<grid, kBlock, 0, stream>>>(args);
        break;
    case TensorParallelSum::StoreFp32:
        tensor_parallel_peer_flags_kernel<TensorParallelSum::StoreFp32, Write>
            <<<grid, kBlock, 0, stream>>>(args);
        break;
    }
    check(cudaGetLastError(), "peer scatter flags exchange launch");
}

// Floats of one scatter-flags row: whole 64-value blocks, so every row starts aligned.
std::size_t flags_row(std::size_t values) { return (values + 63) / 64 * 64; }

// Bytes of one rank's scatter-flags area.
std::size_t flags_area_bytes(std::size_t row) {
    return TensorParallelExchange::kSlots * row * sizeof(float) +
           2 * TensorParallelExchange::kSlots * kFlagBlocks * sizeof(std::uint32_t);
}

// Bytes of one rank's peer scatter-flags area.
std::size_t peer_flags_area_bytes(std::size_t row) {
    return TensorParallelExchange::kSlots * row * sizeof(float) +
           2 * TensorParallelExchange::kSlots * kMaxRanks * kPeerFlagBlocks * sizeof(std::uint32_t);
}

// Whether a form's reduce is one kernel that waits on the other ranks itself, with nothing to
// complete.
bool one_kernel(TensorParallelForm form) {
    return form == TensorParallelForm::Words || form == TensorParallelForm::ScatterFlags;
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
                                         TensorParallelForm::Scatter,
                                         TensorParallelForm::ScatterFlags};

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
    case TensorParallelForm::ScatterFlags:
        return "scatter-flags";
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

namespace {

// Whether two ranks' cards are one model: the same name, memory, multiprocessors and capability.
bool same_card_model(const DeviceContext& context, std::size_t a, std::size_t b) {
    const cudaDeviceProp& x = context.rank(a).props;
    const cudaDeviceProp& y = context.rank(b).props;
    return std::strncmp(x.name, y.name, sizeof(x.name)) == 0 &&
           x.totalGlobalMem == y.totalGlobalMem &&
           x.multiProcessorCount == y.multiProcessorCount && x.major == y.major &&
           x.minor == y.minor;
}

// Each rank's copy bandwidth from a buffer on its card (from_host false: the card's memory) or in
// pinned host memory (from_host true: its bus), shared evenly by the ranks one card holds. Cards of
// one model have one memory system, so their memory bandwidths are the mean of their measurements:
// a share that followed the noise between them would move from start to start, and with it the
// partial sums the ranks add, so the same request would not give the same numbers. Bus speeds stay
// measured per card, since one model of card can sit in slots with different links.
std::vector<double> measured_copy_bandwidth(const DeviceContext& context, bool from_host) {
    // Large enough to stream past the L2 of every supported card, small enough to fit beside a
    // CUDA context on any of them.
    const std::size_t kBytes   = from_host ? 64ULL << 20 : 128ULL << 20;
    constexpr int kTimedCopies = 3;
    const cudaMemcpyKind kind  = from_host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
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
            if (source != nullptr) {
                (void)(from_host ? ninfer::free_host_memory(source)
                                 : ninfer::free_device_memory(source));
            }
            if (destination != nullptr) { (void)ninfer::free_device_memory(destination); }
        };
        try {
            if (from_host) {
                check(cudaHostAlloc(&source, kBytes, cudaHostAllocPortable), "bandwidth cudaHostAlloc");
                std::memset(source, 0, kBytes);
            } else {
                check(cudaMalloc(&source, kBytes), "bandwidth cudaMalloc");
            }
            check(cudaMalloc(&destination, kBytes), "bandwidth cudaMalloc");
            check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "bandwidth stream");
            check(cudaEventCreate(&start), "bandwidth event");
            check(cudaEventCreate(&stop), "bandwidth event");
            if (!from_host) { check(cudaMemsetAsync(source, 0, kBytes, stream), "bandwidth memset"); }
            check(cudaMemcpyAsync(destination, source, kBytes, kind, stream), "bandwidth warm-up");
            check(cudaEventRecord(start, stream), "bandwidth record");
            for (int copy = 0; copy < kTimedCopies; ++copy) {
                check(cudaMemcpyAsync(destination, source, kBytes, kind, stream), "bandwidth copy");
            }
            check(cudaEventRecord(stop, stream), "bandwidth record");
            check(ninfer::wait_for_event(stop), "bandwidth synchronize");
            float milliseconds = 0.0F;
            check(cudaEventElapsedTime(&milliseconds, start, stop), "bandwidth elapsed");
            // A device copy reads and writes every byte; a host copy crosses the bus once.
            const double moved = (from_host ? 1.0 : 2.0) * static_cast<double>(kBytes) * kTimedCopies;
            out[rank] = moved / (std::max(static_cast<double>(milliseconds), 1e-3) * 1e-3);
        } catch (...) {
            release();
            throw;
        }
        release();
    }
    if (!from_host) {
        std::vector<double> mean(out.size(), 0.0);
        for (std::size_t rank = 0; rank < context.size(); ++rank) {
            std::size_t cards = 0;
            for (std::size_t other = 0; other < context.size(); ++other) {
                if (same_card_model(context, rank, other)) {
                    mean[rank] += out[other];
                    ++cards;
                }
            }
            mean[rank] /= static_cast<double>(cards);
        }
        out = std::move(mean);
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

} // namespace

std::vector<double> rank_copy_bandwidth(const DeviceContext& context) {
    return measured_copy_bandwidth(context, false);
}

std::vector<double> rank_host_copy_bandwidth(const DeviceContext& context) {
    return measured_copy_bandwidth(context, true);
}

std::size_t TensorParallelExchange::device_bytes(std::size_t slot_bytes, std::size_t ranks,
                                                 bool peer, TensorParallelTransport transport,
                                                 bool rounded_partials) {
    const std::size_t values = slot_bytes / sizeof(float);
    std::size_t bytes        = kSlots * slot_bytes + 2 * sizeof(std::uint32_t);
    if (rounded_partials) { bytes += kSlots * rounded_bytes(values); }
    if (peer && ranks > 1 &&
        (transport == TensorParallelTransport::Auto ||
         transport == TensorParallelTransport::PeerWrite)) {
        const std::size_t words = words_values(TensorParallelExchangeOptions{}.words_bytes, slot_bytes);
        bytes += kSlots * (ranks - 1) * inbox_values(values, ranks) * sizeof(float) +
                 ranks * kSlots * words * sizeof(unsigned long long);
    }
    if (peer && ranks > 1 && transport != TensorParallelTransport::Host) {
        bytes += peer_flags_area_bytes(
            flags_row(std::min(kScatterFlagsBytes, slot_bytes) / sizeof(float)));
    }
    // The BF16 replica the measurement reduces into.
    return bytes + values * sizeof(__nv_bfloat16);
}

std::size_t TensorParallelExchange::device_bytes() const noexcept {
    if (ranks_.empty()) { return 0; }
    const Rank& rank  = ranks_.front();
    std::size_t bytes = kSlots * slot_bytes_ + 2 * sizeof(std::uint32_t);
    if (rank.rounded[0] != nullptr) {
        bytes += kSlots * rounded_bytes(slot_bytes_ / sizeof(float));
    }
    if (rank.inbox[0] != nullptr) {
        bytes += kSlots * (ranks_.size() - 1) * inbox_values_ * sizeof(float);
    }
    if (rank.peer_words != nullptr) {
        bytes += ranks_.size() * kSlots * words_values_ * sizeof(unsigned long long);
    }
    if (rank.peer_flags != nullptr) {
        bytes += peer_flags_area_bytes(flags_row(scatter_flags_values_));
    }
    return bytes;
}

TensorParallelExchange::TensorParallelExchange(const DeviceContext& context,
                                               TensorParallelExchangeOptions options)
    : slot_bytes_(options.slot_bytes), rounded_(options.rounded_partials) {
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
    scatter_flags_values_ =
        own_cards ? std::min(kScatterFlagsBytes, slot_bytes_) / sizeof(float) : 0;
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
                if (options.rounded_partials != TensorParallelPartials::Fp32) {
                    check(cudaMalloc(reinterpret_cast<void**>(&rank.rounded[slot]),
                                     rounded_bytes(slot_bytes_ / sizeof(float))),
                          "cudaMalloc");
                }
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
            check(ninfer::copy_and_wait(rank.epoch, count, sizeof(count), cudaMemcpyHostToDevice),
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
            if (rank.device_slot[slot] != nullptr) {
                (void)ninfer::free_device_memory(rank.device_slot[slot]);
            }
            if (rank.rounded[slot] != nullptr) {
                (void)ninfer::free_device_memory(rank.rounded[slot]);
            }
            if (rank.host_slot[slot] != nullptr) {
                (void)ninfer::free_host_memory(rank.host_slot[slot]);
            }
            if (rank.inbox[slot] != nullptr) { (void)ninfer::free_device_memory(rank.inbox[slot]); }
            rank.device_slot[slot] = nullptr;
            rank.rounded[slot]     = nullptr;
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
        if (rank.host_words != nullptr) { (void)ninfer::free_host_memory(rank.host_words); }
        if (rank.peer_words != nullptr) { (void)ninfer::free_device_memory(rank.peer_words); }
        if (rank.epoch != nullptr) { (void)ninfer::free_device_memory(rank.epoch); }
        if (rank.scatter_flags != nullptr) { (void)ninfer::free_host_memory(rank.scatter_flags); }
        if (rank.peer_flags != nullptr) { (void)ninfer::free_device_memory(rank.peer_flags); }
        rank.peer_flags    = nullptr;
        rank.host_words    = nullptr;
        rank.peer_words    = nullptr;
        rank.epoch         = nullptr;
        rank.scatter_flags = nullptr;
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
    case TensorParallelForm::ScatterFlags:
        return (route.transport == TensorParallelTransport::Host || peer_) &&
               scatter_flags_values_ != 0 && bytes <= scatter_flags_values_ * sizeof(float);
    }
    return false;
}

TensorParallelExchange::Needs
TensorParallelExchange::needs(const std::vector<TensorParallelBand>& bands) const noexcept {
    Needs out;
    for (const TensorParallelBand& band : bands) {
        if (band.route.form == TensorParallelForm::ScatterFlags && scatter_flags_values_ != 0) {
            (band.route.transport == TensorParallelTransport::Host ? out.scatter_flags
                                                                   : out.peer_flags) = true;
        }
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
                check(cudaHostAlloc(reinterpret_cast<void**>(&rank.host_slot[slot]),
                                    host_slot_bytes(), cudaHostAllocMapped | cudaHostAllocPortable),
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
            check(ninfer::fill_and_wait(rank.peer_words, 0, bytes), "cudaMemset");
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
        if (needs.scatter_flags && rank.scatter_flags == nullptr) {
            const std::size_t bytes = flags_area_bytes(flags_row(scatter_flags_values_));
            check(cudaHostAlloc(reinterpret_cast<void**>(&rank.scatter_flags), bytes,
                                cudaHostAllocMapped | cudaHostAllocPortable),
                  "cudaHostAlloc");
            // Zeroed flags carry epoch 0, which no exchange waits for.
            std::fill_n(rank.scatter_flags, bytes, static_cast<unsigned char>(0));
            cudaFuncAttributes attributes{};
            for (const auto kernel :
                 {tensor_parallel_scatter_flags_kernel<TensorParallelSum::AddBf16>,
                  tensor_parallel_scatter_flags_kernel<TensorParallelSum::StoreBf16>,
                  tensor_parallel_scatter_flags_kernel<TensorParallelSum::StoreFp32>}) {
                check(cudaFuncGetAttributes(&attributes, kernel), "scatter flags kernel");
            }
        }
        if (needs.peer_flags && rank.peer_flags == nullptr) {
            const std::size_t bytes = peer_flags_area_bytes(flags_row(scatter_flags_values_));
            check(cudaMalloc(reinterpret_cast<void**>(&rank.peer_flags), bytes), "cudaMalloc");
            // Zeroed flags carry epoch 0, which no exchange waits for.
            check(ninfer::fill_and_wait(rank.peer_flags, 0, bytes), "cudaMemset");
            cudaFuncAttributes attributes{};
            for (const auto kernel :
                 {tensor_parallel_peer_flags_kernel<TensorParallelSum::AddBf16, false>,
                  tensor_parallel_peer_flags_kernel<TensorParallelSum::StoreBf16, false>,
                  tensor_parallel_peer_flags_kernel<TensorParallelSum::StoreFp32, false>,
                  tensor_parallel_peer_flags_kernel<TensorParallelSum::AddBf16, true>,
                  tensor_parallel_peer_flags_kernel<TensorParallelSum::StoreBf16, true>,
                  tensor_parallel_peer_flags_kernel<TensorParallelSum::StoreFp32, true>}) {
                check(cudaFuncGetAttributes(&attributes, kernel), "peer scatter flags kernel");
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
    // Also the largest partial of each route that stops short of the slot (words, scatter flags,
    // a whole reduce into PeerWrite's inbox rows), so that route's band reaches its last size
    // rather than the power of two below it.
    for (const std::size_t limit :
         {words_values_ * sizeof(float), scatter_flags_values_ * sizeof(float),
          inbox_values_ * sizeof(float)}) {
        if (limit > kFirstProbeBytes && limit < slot_bytes_ &&
            std::find(probes.begin(), probes.end(), limit) == probes.end()) {
            probes.insert(std::upper_bound(probes.begin(), probes.end(), limit), limit);
        }
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
    const auto choose = [&](const std::vector<double>& microseconds)
        -> std::pair<std::vector<TensorParallelBand>, std::vector<TensorParallelBand>> {
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
        return {merged(bands), merged(whole)};
    };
    if (!choice) {
        std::tie(bands_, whole_bands_) = choose({});
        provide(needs(bands_));
        provide(needs(whole_bands_));
        return;
    }
    provide(needs(every));
    if (std::find(transports.begin(), transports.end(), TensorParallelTransport::PeerWrite) !=
        transports.end()) {
        measure_engine_bytes();
    }

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
            if (replicas[r] != nullptr) { (void)ninfer::free_device_memory(replicas[r]); }
        }
        DeviceBinding bind(ranks_[0].device);
        if (start != nullptr) { (void)cudaEventDestroy(start); }
        if (stop != nullptr) { (void)cudaEventDestroy(stop); }
    };
    const std::size_t replica_bytes = slot_bytes_ / sizeof(float) * sizeof(__nv_bfloat16);
    // Microseconds of one exchange of `bytes` on `route` with `partials`: the fastest of kTrials
    // replays of `exchanges` exchanges.
    const auto time_route = [&](TensorParallelRoute route, std::size_t bytes, int exchanges,
                                TensorParallelPartials partials) {
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
                               TensorParallelSum::AddBf16, {}, partials);
                }
                for (std::size_t r = 0; r < n; ++r) {
                    reduce_on(route, r, slot, replicas[r], values, n, streams[r],
                              TensorParallelSum::AddBf16, {}, partials);
                }
                for (std::size_t r = 0; r < n; ++r) {
                    complete_on(route, r, slot, replicas[r], values, streams[r],
                                TensorParallelSum::AddBf16, {}, partials);
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
            check(ninfer::launch_graph(executable, streams[0]), "measurement launch");
            check(ninfer::wait_for_stream(streams[0]), "measurement synchronize");
            for (int trial = 0; trial < kTrials; ++trial) {
                check(cudaEventRecord(start, streams[0]), "measurement record");
                check(ninfer::launch_graph(executable, streams[0]), "measurement launch");
                check(cudaEventRecord(stop, streams[0]), "measurement record");
                check(ninfer::wait_for_event(stop), "measurement synchronize");
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
    std::vector<double> rounded;
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
            check(ninfer::wait_for_stream(streams[r]), "measurement synchronize");
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
                microseconds.push_back(
                    time_route(route, probes[i], exchanges, TensorParallelPartials::Fp32));
                timings_.push_back({probes[i], route, microseconds.back()});
                // Words and scatter flags move the partials as computed whatever the call asks.
                if (rounded_ != TensorParallelPartials::Fp32) {
                    const bool rounds = route.form == TensorParallelForm::Whole ||
                                        route.form == TensorParallelForm::Scatter;
                    rounded.push_back(rounds ? time_route(route, probes[i], exchanges, rounded_)
                                             : microseconds.back());
                }
            }
        }
    } catch (...) {
        discard();
        throw;
    }
    discard();
    measured_                      = true;
    std::tie(bands_, whole_bands_) = choose(microseconds);
    if (!rounded.empty()) { std::tie(rounded_bands_, rounded_whole_bands_) = choose(rounded); }

    // Free what no band takes.
    Needs keep = needs(bands_);
    for (const auto* bands : {&whole_bands_, &rounded_bands_, &rounded_whole_bands_}) {
        const Needs also   = needs(*bands);
        keep.host_slots    = keep.host_slots || also.host_slots;
        keep.inbox         = keep.inbox || also.inbox;
        keep.host_words    = keep.host_words || also.host_words;
        keep.peer_words    = keep.peer_words || also.peer_words;
        keep.scatter_flags = keep.scatter_flags || also.scatter_flags;
        keep.peer_flags    = keep.peer_flags || also.peer_flags;
    }
    for (Rank& rank : ranks_) {
        DeviceBinding bind(rank.device);
        for (std::size_t slot = 0; slot < kSlots; ++slot) {
            if (!keep.host_slots && rank.host_slot[slot] != nullptr) {
                check(ninfer::free_host_memory(rank.host_slot[slot]), "cudaFreeHost");
                rank.host_slot[slot] = nullptr;
            }
            if (!keep.inbox && rank.inbox[slot] != nullptr) {
                check(ninfer::free_device_memory(rank.inbox[slot]), "cudaFree");
                rank.inbox[slot] = nullptr;
            }
        }
        if (!keep.host_words && rank.host_words != nullptr) {
            check(ninfer::free_host_memory(rank.host_words), "cudaFreeHost");
            rank.host_words = nullptr;
        }
        if (!keep.peer_words && rank.peer_words != nullptr) {
            check(ninfer::free_device_memory(rank.peer_words), "cudaFree");
            rank.peer_words = nullptr;
        }
        if (!keep.scatter_flags && rank.scatter_flags != nullptr) {
            check(ninfer::free_host_memory(rank.scatter_flags), "cudaFreeHost");
            rank.scatter_flags = nullptr;
        }
        if (!keep.peer_flags && rank.peer_flags != nullptr) {
            check(ninfer::free_device_memory(rank.peer_flags), "cudaFree");
            rank.peer_flags = nullptr;
        }
    }
}

void TensorParallelExchange::measure_engine_bytes() {
    // Rank 0's rows of every other rank's inbox, as a PeerWrite publish stores them: one kernel for
    // all rows, against one copy-engine copy per row, at sizes doubling up to the row or the slot.
    // The threshold is the smallest size from which the copy engines win at every size timed.
    constexpr std::int64_t kFirstBytes = 64 << 10;
    constexpr std::int64_t kLastBytes  = 8 << 20;
    constexpr int kTrials              = 3;
    const auto widest                  = std::min<std::int64_t>(
        {kLastBytes, static_cast<std::int64_t>(inbox_values_ * sizeof(float)),
         static_cast<std::int64_t>(slot_bytes_)});
    if (widest < kFirstBytes || ranks_.front().inbox[0] == nullptr) { return; }
    const std::size_t n = ranks_.size();
    DeviceBinding bind(ranks_[0].device);
    cudaStream_t stream = nullptr;
    cudaEvent_t start   = nullptr;
    cudaEvent_t stop    = nullptr;
    const auto discard  = [&] {
        if (stream != nullptr) { (void)cudaStreamDestroy(stream); }
        if (start != nullptr) { (void)cudaEventDestroy(start); }
        if (stop != nullptr) { (void)cudaEventDestroy(stop); }
    };
    std::vector<std::pair<std::int64_t, bool>> engine_wins;
    try {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "threshold stream");
        check(cudaEventCreate(&start), "threshold event");
        check(cudaEventCreate(&stop), "threshold event");
        std::array<void*, kMaxRanks> destinations{};
        std::array<const void*, kMaxRanks> sources{};
        std::array<std::int64_t, kMaxRanks> sizes{};
        for (std::size_t other = 1; other < n; ++other) {
            destinations[other - 1] = inbox_row(other, 0, 0);
            sources[other - 1]      = ranks_[0].device_slot[0];
        }
        const auto time = [&](std::int64_t bytes, std::int64_t engine_bytes) {
            std::fill(sizes.begin(), sizes.begin() + static_cast<std::ptrdiff_t>(n - 1), bytes);
            copy_rows(destinations.data(), sources.data(), sizes.data(), n - 1, stream,
                      engine_bytes);
            float best = std::numeric_limits<float>::infinity();
            for (int trial = 0; trial < kTrials; ++trial) {
                check(cudaEventRecord(start, stream), "threshold record");
                copy_rows(destinations.data(), sources.data(), sizes.data(), n - 1, stream,
                          engine_bytes);
                check(cudaEventRecord(stop, stream), "threshold record");
                check(ninfer::wait_for_event(stop), "threshold synchronize");
                float milliseconds = 0.0F;
                check(cudaEventElapsedTime(&milliseconds, start, stop), "threshold elapsed");
                best = std::min(best, milliseconds);
            }
            return best;
        };
        for (std::int64_t bytes = kFirstBytes; bytes <= widest; bytes *= 2) {
            const float kernel = time(bytes, std::numeric_limits<std::int64_t>::max());
            const float engine = time(bytes, 0);
            engine_wins.emplace_back(bytes, engine < kernel);
        }
    } catch (...) {
        discard();
        throw;
    }
    discard();
    // Past the widest size timed, whichever won there keeps winning.
    engine_bytes_ = std::numeric_limits<std::int64_t>::max();
    for (auto it = engine_wins.rbegin(); it != engine_wins.rend() && it->second; ++it) {
        engine_bytes_ = it->first;
    }
    engine_measured_ = true;
}

std::string TensorParallelExchange::describe() const {
    std::string out = "tensor-parallel exchange, " + std::to_string(ranks_.size()) + " ranks" +
                      (measured_ ? ", measured at startup:" : ":");
    const auto list = [&](const std::vector<TensorParallelBand>& bands) {
        for (std::size_t i = 0; i < bands.size(); ++i) {
            const TensorParallelBand& band = bands[i];
            out += std::string(i == 0 ? " " : ", ") +
                   tensor_parallel_transport_name(band.route.transport) + " " +
                   tensor_parallel_form_name(band.route.form) + " to " + size_text(band.bytes);
            if (measured_) {
                char text[32];
                std::snprintf(text, sizeof(text), " (%.1f us)", band.microseconds);
                out += text;
            }
        }
    };
    list(bands_);
    if (!rounded_bands_.empty()) {
        out += rounded_ == TensorParallelPartials::Bf16 ? "; bf16 partials:" : "; int8 partials:";
        list(rounded_bands_);
    }
    if (engine_measured_) {
        out += engine_bytes_ == std::numeric_limits<std::int64_t>::max()
                   ? "; peer-write rows by kernel stores at every size"
                   : "; peer-write rows by the copy engines from " +
                         size_text(static_cast<std::size_t>(engine_bytes_));
    }
    return out;
}

TensorParallelRoute TensorParallelExchange::route(std::size_t bytes, TensorParallelSum sum,
                                                  TensorParallelPartials partials) const noexcept {
    const bool rounded = partials != TensorParallelPartials::Fp32 && !rounded_bands_.empty();
    const TensorParallelRoute chosen = band_of(rounded ? rounded_bands_ : bands_, bytes).route;
    // Two ranks storing an FP32 sum reduce whole (see Scatter above).
    const bool fp32_pair = (chosen.form == TensorParallelForm::Scatter ||
                            chosen.form == TensorParallelForm::ScatterFlags) &&
                           sum == TensorParallelSum::StoreFp32 && ranks_.size() == 2;
    if (!fp32_pair && legal(chosen, bytes)) { return chosen; }
    return whole_route(bytes, partials);
}

TensorParallelRoute TensorParallelExchange::gather_route(std::size_t bytes) const noexcept {
    return whole_route(bytes, TensorParallelPartials::Fp32);
}

TensorParallelRoute
TensorParallelExchange::whole_route(std::size_t bytes,
                                    TensorParallelPartials partials) const noexcept {
    const bool rounded = partials != TensorParallelPartials::Fp32 && !rounded_whole_bands_.empty();
    const TensorParallelRoute chosen =
        band_of(rounded ? rounded_whole_bands_ : whole_bands_, bytes).route;
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

TensorParallelPartials TensorParallelExchange::moved(TensorParallelRoute route,
                                                     TensorParallelPartials partials) const {
    if (partials == TensorParallelPartials::Fp32 ||
        (route.form != TensorParallelForm::Whole && route.form != TensorParallelForm::Scatter)) {
        return TensorParallelPartials::Fp32;
    }
    if (ranks_.front().rounded[0] == nullptr) {
        throw std::logic_error(
            "rounded tensor-parallel partials need an exchange with rounded_partials");
    }
    return partials;
}

unsigned char* TensorParallelExchange::finished_host(std::size_t owner, std::size_t slot,
                                                     TensorParallelPiece piece, std::int64_t begin,
                                                     bool rounded,
                                                     std::size_t size) const noexcept {
    // Unrounded, the owner's own range of its partial, which it never publishes; rounded, past
    // the slot's first half, which holds every rounded partial: up to half a slot more where the
    // output is FP32 (see host_slot_bytes).
    auto* host        = reinterpret_cast<unsigned char*>(ranks_[owner].host_slot[slot]);
    const auto values = static_cast<std::size_t>(piece.offset + begin);
    return rounded ? host + slot_bytes_ / 2 + values * size : host + values * sizeof(float);
}

std::size_t TensorParallelExchange::host_slot_bytes() const noexcept {
    return ranks_.front().rounded[0] != nullptr ? slot_bytes_ + slot_bytes_ / 2 : slot_bytes_;
}

void TensorParallelExchange::publish(std::size_t rank, std::size_t slot, std::size_t bytes,
                                     void* out, cudaStream_t stream, TensorParallelSum sum,
                                     TensorParallelPiece piece, TensorParallelPartials partials) {
    check_piece(rank, slot, bytes, piece);
    const TensorParallelRoute route = piece_route(rank, slot, bytes, sum, piece, partials);
    publish_on(route, rank, slot, bytes, out, stream, sum, piece, moved(route, partials));
}

void TensorParallelExchange::reduce(std::size_t rank, std::size_t slot, void* out,
                                    std::int64_t count, std::size_t contributors,
                                    cudaStream_t stream, TensorParallelSum sum,
                                    TensorParallelPiece piece, TensorParallelPartials partials) {
    const auto bytes = static_cast<std::size_t>(count) * sizeof(float);
    check_piece(rank, slot, bytes, piece);
    if (count <= 0 || contributors == 0 || contributors > ranks_.size()) {
        throw std::invalid_argument(
            "tensor-parallel reduce needs values and one to all ranks as contributors");
    }
    const TensorParallelRoute route = piece_route(rank, slot, bytes, sum, piece, partials);
    ranks_[rank].piece_route[slot][piece.index] = route;
    reduce_on(route, rank, slot, out, count, contributors, stream, sum, piece,
              moved(route, partials));
}

void TensorParallelExchange::complete(std::size_t rank, std::size_t slot, void* out,
                                      std::int64_t count, cudaStream_t stream,
                                      TensorParallelSum sum, TensorParallelPiece piece,
                                      TensorParallelPartials partials) {
    const auto bytes = static_cast<std::size_t>(count) * sizeof(float);
    check_piece(rank, slot, bytes, piece);
    const TensorParallelRoute route = piece_route(rank, slot, bytes, sum, piece, partials);
    complete_on(route, rank, slot, out, count, stream, sum, piece, moved(route, partials));
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

TensorParallelRoute TensorParallelExchange::piece_route(std::size_t rank, std::size_t slot,
                                                        std::size_t bytes, TensorParallelSum sum,
                                                        TensorParallelPiece piece,
                                                        TensorParallelPartials partials) const {
    TensorParallelRoute chosen = route(bytes, sum, partials);
    if (piece.index == 0 && piece.offset == 0) { return chosen; }
    if (one_kernel(chosen.form)) { chosen = whole_route(bytes, partials); }
    if (chosen.transport != TensorParallelTransport::PeerWrite) { return chosen; }
    // A reduce-scatter's place lies below the values of an earlier piece reduced whole over peer
    // writes, so after one every later piece reduces whole too.
    if (chosen.form == TensorParallelForm::Scatter) {
        const auto& earlier = ranks_[rank].piece_route[slot];
        const TensorParallelRoute whole{TensorParallelTransport::PeerWrite,
                                        TensorParallelForm::Whole};
        if (std::find(earlier.begin(), earlier.begin() + static_cast<std::ptrdiff_t>(piece.index),
                      whole) != earlier.begin() + static_cast<std::ptrdiff_t>(piece.index)) {
            chosen = whole;
        }
    }
    // Every owner's range of the piece, at the piece's place, within an inbox row, in the bytes
    // the partials move as: rounded, fewer than their FP32 values.
    const auto values = static_cast<std::int64_t>(bytes / sizeof(float));
    std::int64_t widest = values;
    if (chosen.form == TensorParallelForm::Scatter) {
        const ValueRange first = scatter_range(0, ranks_.size(), values);
        widest                 = first.end - first.begin;
    }
    if (partial_bytes(moved(chosen, partials), inbox_offset(piece, chosen.form) + widest) <=
        inbox_values_ * sizeof(float)) {
        return chosen;
    }
    return {TensorParallelTransport::PeerRead, TensorParallelForm::Whole};
}

void TensorParallelExchange::publish_on(TensorParallelRoute route, std::size_t rank,
                                        std::size_t slot, std::size_t bytes, void* out,
                                        cudaStream_t stream, TensorParallelSum sum,
                                        TensorParallelPiece piece,
                                        TensorParallelPartials partials) {
    Rank& self = ranks_[rank];
    DeviceBinding bind(self.device);
    self.output[slot][piece.index]        = out;
    self.reduced_ready[slot][piece.index] = false;
    // Words leave the slot where it is: the reduce's kernel stores them itself. PeerRead leaves it
    // for the others to read.
    const auto values  = static_cast<std::int64_t>(bytes / sizeof(float));
    float* const local = self.device_slot[slot] + piece.offset;
    const bool rounded = partials != TensorParallelPartials::Fp32;
    if (partials == TensorParallelPartials::Int8 && piece.offset % kQ8Values != 0) {
        throw std::logic_error("8-bit tensor-parallel partials need pieces from a multiple of 32 "
                               "values");
    }
    // Where value v of the piece begins in a slot, or in the rounded copy, as the partials move.
    const auto at = [&](std::int64_t v) { return partial_bytes(partials, piece.offset + v); };
    const auto* moved_source = rounded
                                   ? self.rounded[slot]
                                   : reinterpret_cast<const unsigned char*>(self.device_slot[slot]);
    // Rounded, every value, the rank's own range too, which its reduce adds from the device.
    if (bytes != 0 && rounded) {
        round_values(local, self.rounded[slot] + at(0), values, partials, stream);
    }
    if (bytes != 0 && route.transport == TensorParallelTransport::Host &&
        route.form != TensorParallelForm::Words) {
        // A reduce-scatter's own range of the partial stays on the device, since no other rank
        // reads it.
        const std::size_t size = at(values) - at(0);
        std::size_t own_begin  = size;
        std::size_t own_end    = size;
        if (route.form != TensorParallelForm::Whole) {
            const ValueRange range = scatter_range(rank, ranks_.size(), values);
            own_begin              = at(range.begin) - at(0);
            own_end                = at(range.end) - at(0);
        }
        auto* host         = reinterpret_cast<unsigned char*>(self.host_slot[slot]) + at(0);
        const auto* device = moved_source + at(0);
        if (own_begin != 0) {
            check(cudaMemcpyAsync(host, device, own_begin, cudaMemcpyDeviceToHost, stream),
                  "publish copy");
        }
        if (own_end != size) {
            check(cudaMemcpyAsync(host + own_end, device + own_end, size - own_end,
                                  cudaMemcpyDeviceToHost, stream),
                  "publish copy");
        }
    }
    if (bytes != 0 && route.transport == TensorParallelTransport::PeerWrite &&
        !one_kernel(route.form)) {
        // The whole partial into the rank's row of every other inbox, or each owner's range of it
        // into the rank's row of the owner's; rounded, a row holds the range's rounded values from
        // the byte its place gives them.
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
            destinations[copies] = reinterpret_cast<unsigned char*>(inbox_row(other, rank, slot)) +
                                   partial_bytes(partials, place);
            sources[copies]      = moved_source + at(range.begin);
            sizes[copies]        = static_cast<std::int64_t>(at(range.end) - at(range.begin));
            ++copies;
        }
        copy_rows(destinations.data(), sources.data(), sizes.data(), copies, stream, engine_bytes_);
    }
    check(cudaEventRecord(self.published[slot][piece.index], stream), "publish fence");
}

void TensorParallelExchange::reduce_on(TensorParallelRoute route, std::size_t rank,
                                       std::size_t slot, void* out, std::int64_t count,
                                       std::size_t contributors, cudaStream_t stream,
                                       TensorParallelSum sum, TensorParallelPiece piece,
                                       TensorParallelPartials partials) {
    Rank& self = ranks_[rank];
    DeviceBinding bind(self.device);
    const bool rounded = partials != TensorParallelPartials::Fp32;
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
    if (route.form == TensorParallelForm::ScatterFlags &&
        route.transport != TensorParallelTransport::Host) {
        const bool write = route.transport == TensorParallelTransport::PeerWrite;
        PeerFlagsArgs args{};
        args.out    = out;
        args.vector = aligned(out, 16);
        for (std::size_t r = 0; r < ranks_.size(); ++r) {
            const ValueRange range = scatter_range(r, ranks_.size(), count);
            args.areas[r]          = ranks_[r].peer_flags;
            args.begin[r]          = range.begin;
            args.end[r]            = range.end;
            if (r == rank) {
                args.parts[r] = self.device_slot[slot];
                continue;
            }
            if (write) {
                args.parts[r] = inbox_row(rank, r, slot);
                args.first[r] = scatter_range(rank, ranks_.size(), count).begin;
                args.rows[r]  = inbox_row(r, rank, slot);
                args.outs[r]  = ranks_[r].output[slot][0];
                args.vector   = args.vector && aligned(args.outs[r], 16);
            } else {
                args.parts[r] = ranks_[r].device_slot[slot];
            }
        }
        args.epoch        = self.epoch;
        args.row          = static_cast<std::int64_t>(flags_row(scatter_flags_values_));
        args.ranks        = static_cast<int>(ranks_.size());
        args.rank         = static_cast<int>(rank);
        args.contributors = static_cast<int>(contributors);
        const int blocks  = peer_flag_blocks(count, ranks_.size());
        if (write) {
            exchange_peer_flags<true>(args, blocks, stream, sum);
        } else {
            exchange_peer_flags<false>(args, blocks, stream, sum);
        }
        return;
    }
    if (route.form == TensorParallelForm::ScatterFlags) {
        ScatterFlagsArgs args{};
        args.out = out;
        for (std::size_t r = 0; r < ranks_.size(); ++r) {
            const ValueRange range = scatter_range(r, ranks_.size(), count);
            args.parts[r]          = r == rank ? self.device_slot[slot] : ranks_[r].host_slot[slot];
            args.areas[r]          = ranks_[r].scatter_flags;
            args.begin[r]          = range.begin;
            args.end[r]            = range.end;
        }
        args.epoch        = self.epoch;
        args.row          = static_cast<std::int64_t>(flags_row(scatter_flags_values_));
        args.ranks        = static_cast<int>(ranks_.size());
        args.rank         = static_cast<int>(rank);
        args.contributors = static_cast<int>(contributors);
        args.vector       = aligned(out, 16);
        exchange_scatter_flags(args, stream, sum);
        return;
    }
    const bool scatter     = route.form == TensorParallelForm::Scatter;
    const ValueRange range = scatter ? scatter_range(rank, ranks_.size(), count)
                                     : ValueRange{0, count};
    const std::int64_t place = inbox_offset(piece, route.form);
    std::array<const float*, kMaxRanks> parts{};
    std::array<const unsigned char*, kMaxRanks> rounded_parts{};
    std::array<std::int64_t, kMaxRanks> firsts{};
    const std::size_t start = partial_bytes(partials, piece.offset);
    for (std::size_t r = 0; r < ranks_.size(); ++r) {
        if (rounded) {
            // The rank's own rounded partial on its device; the others' where the transport puts
            // them.
            if (r == rank) {
                rounded_parts[r] = self.rounded[slot] + start;
                continue;
            }
            check(cudaStreamWaitEvent(stream, ranks_[r].published[slot][piece.index], 0),
                  "reduce wait");
            switch (route.transport) {
            case TensorParallelTransport::PeerRead:
                rounded_parts[r] = ranks_[r].rounded[slot] + start;
                break;
            case TensorParallelTransport::PeerWrite:
                rounded_parts[r] =
                    reinterpret_cast<const unsigned char*>(inbox_row(rank, r, slot)) +
                    partial_bytes(partials, place);
                firsts[r] = range.begin;
                break;
            default:
                rounded_parts[r] =
                    reinterpret_cast<const unsigned char*>(ranks_[r].host_slot[slot]) + start;
                break;
            }
            continue;
        }
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
    if (partials == TensorParallelPartials::Bf16) {
        std::array<const __nv_bfloat16*, kMaxRanks> typed{};
        for (std::size_t r = 0; r < ranks_.size(); ++r) {
            typed[r] = reinterpret_cast<const __nv_bfloat16*>(rounded_parts[r]);
        }
        reduce_values(out, typed.data(), firsts.data(), contributors, range.begin, range.end,
                      stream, sum);
    } else if (partials == TensorParallelPartials::Int8) {
        std::array<const Q8Block*, kMaxRanks> typed{};
        for (std::size_t r = 0; r < ranks_.size(); ++r) {
            typed[r] = reinterpret_cast<const Q8Block*>(rounded_parts[r]);
        }
        reduce_values(out, typed.data(), firsts.data(), contributors, range.begin, range.end,
                      stream, sum);
    } else {
        reduce_values(out, parts.data(), firsts.data(), contributors, range.begin, range.end,
                      stream, sum);
    }
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
        copy_rows(destinations.data(), sources.data(), sizes.data(), copies, stream, engine_bytes_);
    }
    if (length != 0 && route.transport == TensorParallelTransport::Host) {
        check(cudaMemcpyAsync(finished_host(rank, slot, piece, range.begin, rounded, size),
                              static_cast<const unsigned char*>(out) + begin * size, length * size,
                              cudaMemcpyDeviceToHost, stream),
              "reduce copy");
    }
    check(cudaEventRecord(self.reduced[slot][piece.index], stream), "reduce fence");
}

void TensorParallelExchange::complete_on(TensorParallelRoute route, std::size_t rank,
                                         std::size_t slot, void* out, std::int64_t count,
                                         cudaStream_t stream, TensorParallelSum sum,
                                         TensorParallelPiece piece,
                                         TensorParallelPartials partials) {
    const bool rounded = partials != TensorParallelPartials::Fp32;
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
        sources[copies] =
            route.transport == TensorParallelTransport::PeerRead
                ? static_cast<const unsigned char*>(from.output[slot][piece.index]) + begin * size
                : static_cast<const void*>(
                      finished_host(owner, slot, piece, range.begin, rounded, size));
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
