#pragma once

// ninfer::ops::detail - the causal prompt launch over FP8-key, NVFP4-value caches (k8v4),
// k8v4/prompt.cu. prompt.cpp hands such caches to it.

#include <cuda_runtime_api.h>

namespace ninfer {
struct Tensor;
struct PagedKVLayerView;
struct PagedKVBatchLayerView;
} // namespace ninfer

namespace ninfer::ops::detail {

void causal_attention_prompt_k8v4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                        const Tensor& positions, const Tensor& valid_columns,
                                        const Tensor& table_rows, float scale,
                                        PagedKVBatchLayerView cache, Tensor& out,
                                        cudaStream_t stream);

void causal_attention_prompt_k8v4_attention_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
