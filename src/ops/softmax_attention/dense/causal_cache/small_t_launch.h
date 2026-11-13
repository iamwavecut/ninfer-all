#pragma once

// ninfer::ops::detail - the causal small-T launches the attention dispatcher
// (causal_softmax_attention.cpp) calls: BF16, bf16p and the INT8 family (small_t.cpp, which hands
// FP8, NVFP4 and K8V4 caches to their own launchers), parallel query tiles and module loading.

#include "core/paged_kv_view.h"
#include "ninfer/kv_cache_storage.h"
#include "ninfer/ops/softmax_attention_envelope.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_invocation.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace ninfer::ops::detail {

// A non-null gate is applied by the reduce epilogue at the store. Only the shared BF16/INT8
// reducer accepts one; the caller keeps the standalone multiply for every other storage.
void causal_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                     const Tensor& positions, const Tensor& valid_columns,
                                     const Tensor& table_rows, float scale,
                                     PagedKVBatchLayerView cache,
                                     CausalAttentionExecutionEnvelope envelope,
                                     std::int32_t column_begin, std::int32_t width,
                                     Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                     Tensor& out, cudaStream_t stream, const void* gate = nullptr,
                                     float* partial_out = nullptr);

// Parallel query tiles for a single row's chunked small-T width over an INT8-family cache: the
// tile width (0 when off) whose tiles run as the batch rows of one split-KV launch and one reduce,
// after one batched append, instead of the serial fused chunks. Opt-in: NINFER_ATTN_PARALLEL_TILES
// or the device profile's "attn_parallel_tiles".
[[nodiscard]] std::int32_t causal_attention_parallel_tile_width(std::int32_t q_heads,
                                                                std::int32_t width,
                                                                std::int32_t batch_size,
                                                                KvCacheStorage storage);

// Runs those tiles over keys already in the cache. tile_valid (only with valid_columns) and
// tile_rows hold one I32 per tile; the partials are sized for tile_width columns, the split
// capacity of that width at batch = tiles, and tiles batch rows.
void causal_attention_small_t_tiles_launch(const Tensor& q, const Tensor& positions,
                                           const Tensor* valid_columns, const Tensor* table_rows,
                                           float scale, PagedKVBatchLayerView cache,
                                           CausalAttentionExecutionEnvelope envelope,
                                           std::int32_t tile_width, Tensor& tile_valid,
                                           Tensor& tile_rows, Tensor& partial_acc,
                                           Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                           cudaStream_t stream, const void* gate = nullptr,
                                           float* partial_out = nullptr);

// Loads the INT8-family small-T modules of widths [1, max_width] (clamped to the geometry's widest
// small-T tile) for appending or cached calls; with parallel tiles enabled, appending calls also
// reach the cached modules up to their widest tile. Other storages have none to load.
void causal_attention_small_t_load_modules(std::int32_t q_heads, std::int32_t kv_heads,
                                           KvCacheStorage storage, bool cached_input,
                                           std::int32_t max_width);

void causal_attention_cached_small_t_launch(const Tensor& q, const Tensor& positions, float scale,
                                            const PagedKVLayerView& cache,
                                            CausalAttentionExecutionEnvelope envelope,
                                            Tensor& partial_acc, Tensor& partial_m,
                                            Tensor& partial_l, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
