#pragma once

// Gemma 4's KV-cache storage (--kv-dtype). A cache row is one KV head of one position, in blocks of
// 256 dimensions (one for a sliding-window block, two for a full-attention block), each block's
// codes and scales in two planes. The codings are the engine's row codings at D = 256
// (ops/kv_cache/kv_row.h): Int8G64, Int4G32, Fp8Row, Nvfp4G16, Lloyd4 (rk4v4 keys), Int4E8
// (rk4v4-e8 keys) and RootE8 (rk2v4-e8 keys).
//
// A rotated plane stores the normalized Sylvester-Hadamard transform of each block, as the engine's
// formats store keys (and NVFP4 values): the attention turns its queries (and its output, for
// rotated values) instead of every row.
//
// A sliding-window block keeps K and V in the format's own codings. A full-attention block keeps one
// plane, its values, and derives each key from it (RoPE(v * k_norm)): the per-position rotation
// rules out a Hadamard-rotated plane, so the plane takes an unrotated coding at least as fine as the
// format's keys -- INT8 for int8 and rk8v4, FP8 for fp8 and k8v4, NVFP4 for nvfp4, and the packed
// int4 value coding for the 4-bit and 2-bit key formats (rk4v4, rk4v4-e8, rk2v4-e8).

#include "ninfer/kv_cache_storage.h"
#include "ops/kv_cache/kv_row.h"

namespace ninfer::models::gemma4 {

using ops::kv_row::KvCoding;

struct KvPlane {
    KvCoding coding = KvCoding::F16;
    bool rotated    = false;
};

struct KvFormat {
    KvCacheStorage storage = KvCacheStorage::BFloat16;
    KvPlane key, value; // sliding-window blocks
    KvPlane full;       // the full-attention blocks' one plane
    [[nodiscard]] bool quantized() const noexcept {
        return storage != KvCacheStorage::BFloat16 && storage != KvCacheStorage::Float16;
    }
};

[[nodiscard]] KvFormat kv_format(KvCacheStorage storage);

inline constexpr int kKvBlockDims = 256;

// Bytes of one block's codes and of its scales.
[[nodiscard]] NINFER_KV_ROW_HOST_DEVICE constexpr int kv_code_bytes(KvCoding c) noexcept {
    return ops::kv_row::code_bytes(c, kKvBlockDims);
}

[[nodiscard]] NINFER_KV_ROW_HOST_DEVICE constexpr int kv_scale_bytes(KvCoding c) noexcept {
    return ops::kv_row::scale_bytes(c, kKvBlockDims);
}

} // namespace ninfer::models::gemma4
