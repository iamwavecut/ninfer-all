#pragma once

// ninfer::ops::detail - the causal prompt launches the attention dispatcher
// (causal_softmax_attention.cpp) calls: BF16 and the INT8 family (prompt.cpp, which hands FP8,
// NVFP4 and K8V4 caches to their own launchers) and a position-split rank's partial prompt.

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

// A position-split rank's prompt attention over the keys of its own pages, every storage: each
// column's output normalized over those keys and its log-sum-exp, into `partial_out`
// (causal_attention_merge_partials). The caller has appended the call's rows. One sequence; the
// Volta prompt kernel's partial form on every architecture (sm70/prompt.h).
void causal_attention_prompt_partial_launch(const Tensor& q, const Tensor& positions,
                                            const Tensor& valid_columns, const Tensor& table_rows,
                                            float scale, PagedKVBatchLayerView cache,
                                            float* partial_out, cudaStream_t stream);

// The fast INT8-family kernel plans its key splits at `max_visible_keys` and takes their partials
// from `workspace` (causal_softmax_attention_workspace_capacity_bytes reserves them).
void causal_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& positions, const Tensor& valid_columns,
                                    const Tensor& table_rows, float scale,
                                    PagedKVBatchLayerView cache, std::uint32_t max_visible_keys,
                                    WorkspaceArena& workspace, Tensor& out, bool fast,
                                    cudaStream_t stream);

void causal_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                              const PagedKVLayerView& cache,
                                              std::uint32_t max_visible_keys,
                                              WorkspaceArena& workspace, Tensor& out, bool fast,
                                              cudaStream_t stream);

} // namespace ninfer::ops::detail
