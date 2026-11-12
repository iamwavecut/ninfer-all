// ninfer::ops::detail - packed BF16 (bf16p) append: each K and V row packed by itself, or stored as
// the poison row when it has no packed form (ops/kv_cache/bf16p_codec.cuh).
#include "ops/kv_cache/append/launch.h"

#include "core/device_launch.h"
#include "ops/common/math.h"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/append/geometry.cuh"
#include "ops/kv_cache/bf16p_codec.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kWarps = 8;

// The metadata of a batched call's sequence `row` (blockIdx.y): its table row and valid columns.
__device__ __forceinline__ PagedKVDirectMetadata bf16p_sequence_metadata(PagedKVDirectMetadata metadata,
                                                                        int) {
    return metadata;
}

template <bool Masked>
__device__ __forceinline__ PagedKVBatchMetadata<Masked>
bf16p_sequence_metadata(PagedKVBatchMetadata<Masked> metadata, int row) {
    metadata.table_rows += row;
    if constexpr (Masked) { metadata.valid_columns += row; }
    return metadata;
}

// One warp a (token, KV head) of sequence blockIdx.y: its K row, then its V row. Lane L packs
// values 8 L .. 8 L + 7.
template <typename Geometry, typename Metadata>
__launch_bounds__(kWarps * 32) __global__
    void kv_cache_append_bf16p_kernel(const __nv_bfloat16* __restrict__ k,
                                      const __nv_bfloat16* __restrict__ v,
                                      const std::int32_t* __restrict__ positions, Metadata metadata,
                                      std::uint8_t* __restrict__ cache_k,
                                      std::uint8_t* __restrict__ cache_v,
                                      std::uint8_t* __restrict__ side_k,
                                      std::uint8_t* __restrict__ side_v, std::int32_t width) {
    __shared__ __align__(16) unsigned scratch[kWarps][kBf16pRowValues];
    const int sequence = static_cast<int>(blockIdx.y);
    metadata           = bf16p_sequence_metadata(metadata, sequence);
    positions += static_cast<std::int64_t>(width) * sequence;
    k += static_cast<std::int64_t>(width) * sequence * Geometry::KVHeads * kBf16pRowValues;
    v += static_cast<std::int64_t>(width) * sequence * Geometry::KVHeads * kBf16pRowValues;
    const int tokens = metadata.valid_tokens(width);
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int unit   = static_cast<int>(blockIdx.x) * kWarps + warp;
    if (unit >= tokens * Geometry::KVHeads) { return; }
    const int head     = unit % Geometry::KVHeads;
    const int token    = unit / Geometry::KVHeads;
    const int position = positions[0] + token;
    const int page     = metadata.local_page(paged_kv_physical_page(metadata.block_table(), position));
    if (page < 0) { return; } // another rank's page (a position-split cache)
    const int offset         = position & kPagedKVPageMask;
    const std::int64_t row   = bf16p_row_offset(page, head, offset, Geometry::KVHeads);
    const std::int64_t side  = bf16p_side_offset(page, head, offset, Geometry::KVHeads);
    const std::int64_t value = (static_cast<std::int64_t>(token) * Geometry::KVHeads + head) *
                                   kBf16pRowValues +
                               8 * lane;
    const uint4 key_bits   = *reinterpret_cast<const uint4*>(k + value);
    const uint4 value_bits = *reinterpret_cast<const uint4*>(v + value);
    (void)bf16p_encode_row(key_bits, cache_k + row, side_k + side, scratch[warp], lane);
    (void)bf16p_encode_row(value_bits, cache_v + row, side_v + side, scratch[warp], lane);
}

template <typename CacheView, typename Metadata>
void launch_bf16p(const Tensor& k, const Tensor& v, const Tensor& positions, CacheView cache,
                  Metadata metadata, cudaStream_t stream) {
    const auto tokens    = static_cast<std::int32_t>(k.ne[2]);
    const auto sequences = static_cast<unsigned>(k.ne[3] < 1 ? 1 : k.ne[3]);
    if (tokens < 1) { return; }
    with_kv_append_geometry(k.ne[1], [&]<typename Geometry>() {
        const std::int64_t units = static_cast<std::int64_t>(tokens) * Geometry::KVHeads;
        const dim3 grid(static_cast<unsigned>(div_up(units, std::int64_t{kWarps})), sequences);
        kv_cache_append_bf16p_kernel<Geometry, Metadata><<<grid, kWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data), metadata,
            static_cast<std::uint8_t*>(cache.k_pages.data),
            static_cast<std::uint8_t*>(cache.v_pages.data),
            static_cast<std::uint8_t*>(cache.k_scale_pages.data),
            static_cast<std::uint8_t*>(cache.v_scale_pages.data), tokens);
    });
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void kv_cache_append_bf16p_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                  PagedKVLayerView cache, cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data),
                                         cache.page_stride, cache.page_residue};
    launch_bf16p(k, v, positions, cache, metadata, stream);
}

void kv_cache_append_bf16p_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                        const Tensor& valid_columns, const Tensor& table_rows,
                                        PagedKVBatchLayerView cache, cudaStream_t stream) {
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
            .page_stride  = cache.page_stride,
            .page_residue = cache.page_residue,
        };
        launch_bf16p(k, v, positions, cache, metadata, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
