#pragma once
#include "ops/linear/common/output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail::unified {
using Nvfp4GdnInputOutput = LinearBf16SegmentedOutput<10240, 6144>;

// This call's share of the fused [qkv; z] rows. A tensor-parallel rank holds whole GDN key heads
// (core/tensor_split.h divides GDN key heads), so the two segments keep their order and only shrink
// -- and the two output tensors already carry their row counts, so nothing is baked here. Row tiles
// are 128 wide.
struct GdnInputShare {
    std::int32_t qkv;
    std::int32_t z;

    [[nodiscard]] std::int32_t total() const { return qkv + z; }
};

inline bool gdn_input_share_admitted(const GdnInputShare& share) {
    constexpr std::int32_t kTile  = 128;
    constexpr std::int32_t kPlane = 16384;
    if (share.qkv <= 0 || share.z <= 0) { return false; }
    if (share.qkv % kTile != 0 || share.z % kTile != 0) { return false; }
    // Never wider than the registered plane the workspace query is sized for.
    return share.total() <= kPlane;
}

// The runtime-segment form of Nvfp4GdnInputOutput: same order and same round-once store, with the
// segment sizes the call brings instead of the compile-time constants. Every route that reads it
// stores through `store` alone, so no vector form is needed.
struct Nvfp4GdnInputShareOutput {
    __nv_bfloat16* qkv;
    __nv_bfloat16* z;
    GdnInputShare rows;

    __device__ __forceinline__ void store(std::int32_t row, std::int32_t token,
                                          float value) const {
        const __nv_bfloat16 bits  = __float2bfloat16_rn(value);
        const std::int64_t column = token;
        if (row < rows.qkv) {
            qkv[column * rows.qkv + row] = bits;
            return;
        }
        row -= rows.qkv;
        z[column * rows.z + row] = bits;
    }
};
} // namespace ninfer::ops::detail::unified
