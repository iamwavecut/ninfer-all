#pragma once
#include "ops/linear/common/output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail::unified {
using Nvfp4AttentionInputOutput = LinearBf16SegmentedOutput<6144, 1024, 6144, 1024>;

// This call's share of the fused [query; key; gate; value] rows. A tensor-parallel rank holds whole
// KV heads and their query heads (core/weight_input.cpp), so a share keeps every segment's order and
// only shrinks it -- and the four output tensors already carry those row counts, which is why the
// geometry is read from the call instead of baked. Row tiles are 128 wide; K and V carry the same
// head count.
struct AttentionInputShare {
    std::int32_t query;
    std::int32_t key;
    std::int32_t gate;
    std::int32_t value;

    [[nodiscard]] std::int32_t total() const { return query + key + gate + value; }
};

inline bool attention_input_share_admitted(const AttentionInputShare& share) {
    constexpr std::int32_t kTile  = 128;
    constexpr std::int32_t kPlane = 14336;
    if (share.query <= 0 || share.key <= 0 || share.gate <= 0 || share.value <= 0) { return false; }
    if (share.key != share.value) { return false; }
    if (share.query % kTile != 0 || share.key % kTile != 0 || share.gate % kTile != 0) {
        return false;
    }
    // Never wider than the registered plane the workspace query is sized for.
    return share.total() <= kPlane;
}

// A segmented store whose segment rows are the call's rather than a template's. Four named pointers
// keep the segment choice off a runtime array index, the way LinearBf16SegmentedOutput's constant
// indices do; every route that reads it stores through `store` alone, so no vector form is needed.
struct Nvfp4AttentionInputShareOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;
    AttentionInputShare rows;

    __device__ __forceinline__ void store(std::int32_t row, std::int32_t token,
                                          float result) const {
        const __nv_bfloat16 bits  = __float2bfloat16_rn(result);
        const std::int64_t column = token;
        if (row < rows.query) {
            query[column * rows.query + row] = bits;
            return;
        }
        row -= rows.query;
        if (row < rows.key) {
            key[column * rows.key + row] = bits;
            return;
        }
        row -= rows.key;
        if (row < rows.gate) {
            gate[column * rows.gate + row] = bits;
            return;
        }
        row -= rows.gate;
        value[column * rows.value + row] = bits;
    }
};
} // namespace ninfer::ops::detail::unified
