#pragma once

// The non-owning page-table views that the attention and cache kernels read; the pools that own the
// pages are in core/paged_kv_cache.h, which includes this header.

#include "core/paged_kv_page.h"
#include "core/paged_kv_storage.h"
#include "core/tensor.h"

#include <cstdint>

namespace ninfer {

/** Non-owning, single-sequence view consumed by growing-cache Ops. */
struct PagedKVLayerView {
    Tensor k_pages;
    Tensor v_pages;
    Tensor k_scale_pages;
    Tensor v_scale_pages;
    Tensor block_table;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    KvCacheStorage storage    = KvCacheStorage::BFloat16;
    // A rank of a position-split pool (DeviceKVPagePool) holds the physical pages p with
    // p % page_stride == page_residue, at page p / page_stride of its planes; 1 and 0 hold them all.
    std::int32_t page_stride  = 1;
    std::int32_t page_residue = 0;
};

/** Non-owning multi-sequence view consumed by batched growing-cache Ops. */
struct PagedKVBatchLayerView {
    Tensor k_pages;
    Tensor v_pages;
    Tensor k_scale_pages;
    Tensor v_scale_pages;
    Tensor block_tables;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    KvCacheStorage storage    = KvCacheStorage::BFloat16;
    // As PagedKVLayerView.
    std::int32_t page_stride  = 1;
    std::int32_t page_residue = 0;
};

/** Rebinds one checked single-sequence table row as a one-row batched view. */
[[nodiscard]] PagedKVBatchLayerView single_row_paged_kv_batch_view(const PagedKVLayerView& cache);

} // namespace ninfer
