#pragma once

#include "ops/kernel/paged_kv_address.cuh"

#include <cstdint>

namespace ninfer::ops {

// One sequence of a batched append: a direct cache has one, every row of it the table's.
__device__ __forceinline__ PagedKVDirectMetadata
paged_kv_sequence_row(PagedKVDirectMetadata metadata, int) {
    return metadata;
}

// Batch row `row`: its table row and valid columns.
template <bool Masked>
__device__ __forceinline__ PagedKVBatchMetadata<Masked>
paged_kv_sequence_row(PagedKVBatchMetadata<Masked> metadata, int row) {
    metadata.table_rows += row;
    if constexpr (Masked) { metadata.valid_columns += row; }
    return metadata;
}

// A batched append's sequence blockIdx.y: its metadata row, its `width` positions and its K and
// V columns of `column_elements` values each. A single-sequence launch has the one. The pointers
// are deduced with their qualifiers, __restrict__ included.
template <typename Pointer, typename Positions, typename Metadata>
__device__ __forceinline__ void paged_kv_select_sequence(Pointer& k, Pointer& v,
                                                         Positions& positions, Metadata& metadata,
                                                         std::int32_t width,
                                                         std::int32_t column_elements) {
    const int sequence = static_cast<int>(blockIdx.y);
    if (sequence == 0) { return; }
    metadata = paged_kv_sequence_row(metadata, sequence);
    positions += static_cast<std::int64_t>(width) * sequence;
    const std::int64_t offset = static_cast<std::int64_t>(width) * sequence * column_elements;
    k += offset;
    v += offset;
}

} // namespace ninfer::ops
