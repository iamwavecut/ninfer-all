#pragma once

// ninfer::ops::detail - the causal prompt launch over FP8 E4M3 row-scaled caches (fp8),
// fp8/prompt.cu. prompt.cpp hands such caches to it.

#include <cuda_runtime_api.h>

#include <cstdint>

namespace ninfer {
struct Tensor;
struct PagedKVLayerView;
struct PagedKVBatchLayerView;
class DeviceArena;
using WorkspaceArena = DeviceArena;
} // namespace ninfer

namespace ninfer::ops::detail {

// The fast FP8 prompt kernel (fp8/prompt_fast.cuh) plans its key splits at `max_visible_keys` and
// takes their partials from `workspace` (causal_softmax_attention_workspace_capacity_bytes
// reserves them).
void causal_attention_prompt_fp8_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                        const Tensor& positions, const Tensor& valid_columns,
                                        const Tensor& table_rows, float scale,
                                        PagedKVBatchLayerView cache, std::uint32_t max_visible_keys,
                                        WorkspaceArena& workspace, Tensor& out, bool fast,
                                        cudaStream_t stream);

void causal_attention_prompt_fp8_attention_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  std::uint32_t max_visible_keys,
                                                  WorkspaceArena& workspace, Tensor& out, bool fast,
                                                  cudaStream_t stream);

} // namespace ninfer::ops::detail
