#pragma once

// Inkling 2's KV-cache storage (--kv-dtype). A cache row is one KV head of one position, its
// 128 values in one block: codes and scales in two planes of a visit (executor.cpp). The codings
// are the engine's own codecs (ops/kv_cache/*_codec.cuh) over groups of the row, and MXFP8:
//
//   Int8G64   signed int8, one FP16 scale per 64 dimensions, FP16-RNE(absmax / 127)
//   Int4G32   two signed 4-bit codes per byte, one FP16 scale per 32, FP16-RNE(absmax / 7)
//   Lloyd4    4-bit Lloyd-Max indices with a G64 FP16 least-squares scale (rk4v4 keys)
//   Mxfp8     E4M3 codes, one UE8M0 power-of-two scale per 32 (kv_codec.cuh, the release's rule)
//
// Only the eight full-attention visits store planes (rk8v4: Int8G64 keys and Int4G32 values;
// rk4v4: Lloyd4 keys and Int4G32 values; mxfp8: Mxfp8 keys and values); the windowed visits'
// rings, the convolution histories and the DSpark drafter's caches stay F16 whatever the flag
// says. Rounding the rings' rows to MXFP8 as well raised WikiText's mean NLL (+0.029% against
// +0.024% over 67,977 tokens, +0.091% against +0.015% over 9,718) and saves no memory, since a
// ring's size does not grow with the context. Unlike Gemma 4's sliding blocks the
// rows here are stored unrotated: Inkling's K and V come out of their convolutions clipped and
// RMS-normed, and Gemma 4's full-attention blocks store the same codings unrotated (its rk8v4
// quality numbers cover exactly this). The codings otherwise need no rotation to reach their
// measured error.

#include "ninfer/kv_cache_storage.h"

#include <cstdint>

// The sizes below are also compile-time constants of the CUDA kernels.
#if defined(__CUDACC__)
#    define NINFER_INKLING2_KV_HOST_DEVICE __host__ __device__
#else
#    define NINFER_INKLING2_KV_HOST_DEVICE
#endif

namespace ninfer::models::inkling2 {

enum class KvCoding : std::uint8_t { F16, Int8G64, Int4G32, Lloyd4, Mxfp8 };

// One row: one KV head's 128 values, so 128-wide blocks.
inline constexpr int kKvRowDims = 128;

struct KvPlane {
    KvCoding coding = KvCoding::F16;
};

struct KvFormat {
    KvCacheStorage storage = KvCacheStorage::BFloat16;
    KvPlane key, value;
    [[nodiscard]] bool quantized() const noexcept {
        return storage != KvCacheStorage::BFloat16 && storage != KvCacheStorage::Float16;
    }
    // MXFP8: kv_post encodes the full visits' planes itself; the other quantized formats stage
    // F16 rows for kv_store.
    [[nodiscard]] bool mxfp8() const noexcept { return storage == KvCacheStorage::Mxfp8; }
    [[nodiscard]] bool staged() const noexcept { return quantized() && !mxfp8(); }
};

[[nodiscard]] KvFormat kv_format(KvCacheStorage storage);

// Bytes of one row's codes and of its scales.
[[nodiscard]] NINFER_INKLING2_KV_HOST_DEVICE constexpr int kv_code_bytes(KvCoding c) noexcept {
    switch (c) {
    case KvCoding::F16: return 2 * kKvRowDims;
    case KvCoding::Int8G64:
    case KvCoding::Mxfp8: return kKvRowDims;
    case KvCoding::Int4G32:
    case KvCoding::Lloyd4: return kKvRowDims / 2;
    }
    return 0;
}

[[nodiscard]] NINFER_INKLING2_KV_HOST_DEVICE constexpr int kv_scale_bytes(KvCoding c) noexcept {
    switch (c) {
    case KvCoding::F16: return 0;
    case KvCoding::Int8G64:
    case KvCoding::Lloyd4: return 2 * 2; // two groups of 64
    case KvCoding::Int4G32: return 4 * 2; // four groups of 32
    case KvCoding::Mxfp8: return 4;       // a UE8M0 exponent byte per 32
    }
    return 0;
}

// A quantized plane of a visit: one row's codes at codes + sequence * code_seq +
// head * code_head + slot * code_row, its scales likewise (executor.cpp's code and scale areas).
struct KvPlaneView {
    std::uint8_t* codes  = nullptr;
    std::uint8_t* scales = nullptr;
    std::int64_t code_row = 0, scale_row = 0, code_head = 0, scale_head = 0, code_seq = 0,
                 scale_seq = 0;
    KvCoding coding = KvCoding::F16;
};

} // namespace ninfer::models::inkling2
