#pragma once

// ninfer::ops::detail - the fast NVFP4 prompt kernel's launch (nvfp4/prompt_fast.cu).

#include <cuda_runtime_api.h>

#include <cstdint>

namespace ninfer {
struct Tensor;
struct PagedKVBatchLayerView;
class DeviceArena;
using WorkspaceArena = DeviceArena;
} // namespace ninfer

namespace ninfer::ops::detail {

// The fast NVFP4 prompt kernel (block-scaled FP4 QK, V decoded in registers; Ian Ranson's
// nvfp4/prompt_fast.cuh) over keys already in the cache. A single-row launch whose row blocks alone
// would leave SMs idle splits its keys across CTAs into `workspace`. Blackwell (120a) builds only;
// causal_attention_prompt_nvfp4_fast_applies() is false everywhere else.
void causal_attention_prompt_nvfp4_fast_launch(const Tensor& q, const Tensor& positions,
                                               const Tensor* valid_columns,
                                               const Tensor* table_rows, float scale,
                                               PagedKVBatchLayerView cache,
                                               std::uint32_t max_visible_keys,
                                               WorkspaceArena& workspace, Tensor& out,
                                               cudaStream_t stream);

} // namespace ninfer::ops::detail
