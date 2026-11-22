#pragma once

// Quantized KV planes of 128-wide rows (one KV head of one position), as Inkling 2's full-attention
// visits and Flint's full-attention blocks store them: a row's codes and scales lie in two planes.
// The codings are the engine's codecs (int8_g64_codec.cuh) over groups of the row, and MXFP8:
//
//   Int8G64   signed int8, one FP16 scale per 64 dimensions, FP16-RNE(absmax / 127)
//   Int4G32   two signed 4-bit codes per byte, one FP16 scale per 32, FP16-RNE(absmax / 7)
//   Lloyd4    4-bit Lloyd-Max indices with a G64 FP16 least-squares scale (rk4v4 keys)
//   Mxfp8     E4M3 codes, one UE8M0 power-of-two scale per 32 (row128_codec.cuh)
//
// A plane stores its rows as they are, or rotated: each row as H x / 8, H the unnormalized
// 128-point Sylvester-Hadamard transform, the attention turning its queries by H / 16 instead (the
// two scales multiply to 1/128, so the scores keep their values). Device side: row128_codec.cuh.

#include <cstdint>

// The sizes below are also compile-time constants of the CUDA kernels.
#if defined(__CUDACC__)
#    define NINFER_KV128_HOST_DEVICE __host__ __device__
#else
#    define NINFER_KV128_HOST_DEVICE
#endif

namespace ninfer::ops::kv128 {

enum class KvCoding : std::uint8_t { F16, Int8G64, Int4G32, Lloyd4, Mxfp8 };

// One row: one KV head's 128 values, so 128-wide blocks.
inline constexpr int kKvRowDims = 128;

// Bytes of one row's codes and of its scales.
[[nodiscard]] NINFER_KV128_HOST_DEVICE constexpr int kv_code_bytes(KvCoding c) noexcept {
    switch (c) {
    case KvCoding::F16: return 2 * kKvRowDims;
    case KvCoding::Int8G64:
    case KvCoding::Mxfp8: return kKvRowDims;
    case KvCoding::Int4G32:
    case KvCoding::Lloyd4: return kKvRowDims / 2;
    }
    return 0;
}

[[nodiscard]] NINFER_KV128_HOST_DEVICE constexpr int kv_scale_bytes(KvCoding c) noexcept {
    switch (c) {
    case KvCoding::F16: return 0;
    case KvCoding::Int8G64:
    case KvCoding::Lloyd4: return 2 * 2; // two groups of 64
    case KvCoding::Int4G32: return 4 * 2; // four groups of 32
    case KvCoding::Mxfp8: return 4;       // a UE8M0 exponent byte per 32
    }
    return 0;
}

// A rotated plane stores H x times kRotatedRowScale; its queries are turned by H times
// kRotatedQueryScale (row128_rotate.h).
inline constexpr float kRotatedRowScale = 0.125f, kRotatedQueryScale = 0.0625f;

// A quantized plane: one row's codes at codes + sequence * code_seq + head * code_head +
// slot * code_row, its scales likewise.
struct KvPlaneView {
    std::uint8_t* codes  = nullptr;
    std::uint8_t* scales = nullptr;
    std::int64_t code_row = 0, scale_row = 0, code_head = 0, scale_head = 0, code_seq = 0,
                 scale_seq = 0;
    KvCoding coding = KvCoding::F16;
    bool rotated    = false;
};

} // namespace ninfer::ops::kv128
