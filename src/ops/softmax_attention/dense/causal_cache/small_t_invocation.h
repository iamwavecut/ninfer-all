#pragma once

// ninfer::ops::detail - what a causal small-T partial launch reads besides its tensors: the call's
// columns, batch rows and key partition; and the launch arguments the host launchers make of them
// for the kernel units.

#include "core/paged_kv_view.h"
#include "ninfer/ops/softmax_attention_envelope.h"
#include "ops/softmax_attention/dense/causal_cache/key_partition.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_partial_launch.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_reduce_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

struct CausalSmallTInvocation {
    const Tensor* valid_columns = nullptr;
    const Tensor* table_rows    = nullptr;
    std::int32_t full_width     = 0;
    std::int32_t column_begin   = 0;
    std::int32_t width          = 0;
    std::int32_t batch_size     = 1;
    // How the launch divides and holds keys, and where a position-split rank's reducer leaves the
    // partial output it exchanges (causal_store_partial_output) in place of `out`.
    CausalKeyPartition partition{};
    float* partial_out = nullptr;
};

// The partition a launch over `cache` within `envelope` takes.
[[nodiscard]] inline CausalKeyPartition
causal_attention_key_partition(CausalAttentionExecutionEnvelope envelope,
                               const PagedKVBatchLayerView& cache) {
    return {.segment_keys = static_cast<std::int32_t>(envelope.invariant_segment_keys),
            .page_stride  = cache.page_stride,
            .page_residue = cache.page_residue};
}

// The partial launch of `invocation` over `cache` into `splits` partials of each column.
[[nodiscard]] inline SmallTPartialLaunch
small_t_partial_launch(const Tensor& q, const Tensor& positions, float scale,
                       const PagedKVBatchLayerView& cache, const CausalSmallTInvocation& invocation,
                       std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                       Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    SmallTPartialLaunch launch{};
    launch.q             = q.data;
    launch.positions     = static_cast<const std::int32_t*>(positions.data);
    launch.cache_k       = cache.k_pages.data;
    launch.cache_v       = cache.v_pages.data;
    launch.cache_k_scale = cache.k_scale_pages.data;
    launch.cache_v_scale = cache.v_scale_pages.data;
    launch.block_tables  = static_cast<const std::int32_t*>(cache.block_tables.data);
    launch.valid_columns = invocation.valid_columns == nullptr
                               ? nullptr
                               : static_cast<const std::int32_t*>(invocation.valid_columns->data);
    launch.table_rows    = invocation.table_rows == nullptr
                               ? nullptr
                               : static_cast<const std::int32_t*>(invocation.table_rows->data);
    launch.table_stride     = cache.block_tables.ne[0];
    launch.width            = invocation.width;
    launch.full_width       = invocation.full_width;
    launch.column_begin     = invocation.column_begin;
    launch.logical_capacity = logical_capacity;
    launch.q_heads          = q.ne[1];
    launch.splits           = splits;
    launch.batch_size       = invocation.batch_size;
    launch.scale            = scale;
    launch.partial_acc      = static_cast<float*>(partial_acc.data);
    launch.partial_m        = static_cast<float*>(partial_m.data);
    launch.partial_l        = static_cast<float*>(partial_l.data);
    launch.partition        = invocation.partition;
    launch.multi_batch      = invocation.batch_size != 1;
    launch.masked           = invocation.valid_columns != nullptr;
    launch.stream           = stream;
    return launch;
}

// The reduction of those partials of `q_heads` query heads into `out`, or into the invocation's
// partial_out.
[[nodiscard]] inline SmallTReduceLaunch
small_t_reduce_launch(const Tensor& positions, const CausalSmallTInvocation& invocation,
                      std::int32_t q_heads, std::int32_t logical_capacity, std::int32_t splits,
                      const Tensor& partial_acc, const Tensor& partial_m, const Tensor& partial_l,
                      Tensor& out, cudaStream_t stream) {
    SmallTReduceLaunch launch{};
    launch.partial_acc   = static_cast<const float*>(partial_acc.data);
    launch.partial_m     = static_cast<const float*>(partial_m.data);
    launch.partial_l     = static_cast<const float*>(partial_l.data);
    launch.positions     = static_cast<const std::int32_t*>(positions.data);
    launch.valid_columns = invocation.valid_columns == nullptr
                               ? nullptr
                               : static_cast<const std::int32_t*>(invocation.valid_columns->data);
    launch.q_heads          = q_heads;
    launch.width            = invocation.width;
    launch.full_width       = invocation.full_width;
    launch.column_begin     = invocation.column_begin;
    launch.batch_size       = invocation.batch_size;
    launch.logical_capacity = logical_capacity;
    launch.splits           = splits;
    launch.out              = out.data;
    launch.partition        = invocation.partition;
    launch.partial_out      = invocation.partial_out;
    launch.stream           = stream;
    return launch;
}

} // namespace ninfer::ops::detail
