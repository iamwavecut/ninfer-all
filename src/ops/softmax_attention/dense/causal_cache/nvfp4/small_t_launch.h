#pragma once

// ninfer::ops::detail - the causal small-T launch over NVFP4 G16 caches (nvfp4), nvfp4/small_t.cpp.
// small_t.cpp hands such caches to it.

#include "ninfer/ops/softmax_attention_envelope.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace ninfer {
struct Tensor;
struct PagedKVLayerView;
struct PagedKVBatchLayerView;
} // namespace ninfer

namespace ninfer::ops::detail {

void causal_attention_small_t_nvfp4_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream,
    float* partial_out = nullptr);

void causal_attention_cached_small_t_nvfp4_launch(const Tensor& q, const Tensor& positions,
                                             float scale, const PagedKVLayerView& cache,
                                             CausalAttentionExecutionEnvelope envelope,
                                             Tensor& partial_acc, Tensor& partial_m,
                                             Tensor& partial_l, Tensor& out, cudaStream_t stream);


} // namespace ninfer::ops::detail
