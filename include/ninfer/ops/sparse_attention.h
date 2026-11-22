#pragma once

#include "core/arena.h"
#include "core/paged_kv_view.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * The sparse attention of Qwen3.8-Flash-Next and dense Qwen4 for one sequence: query p attends
 * only to the 4-token blocks its indexer selected (qsa_indexer_select) and to its own incomplete
 * block, positions 4 floor((p + 1) / 4) .. p:
 *
 *   out[h] = sum_{j in S_p} softmax_j(scale * q[h] . k[kv(h), j]) v[kv(h), j],   kv(h) = h / G
 *
 * with G query heads per KV head: 12 (Flash-Next, two KV heads) or 6 (dense Qwen4, four), the
 * ratio of q's heads to the cache's, at most 24 query heads in all.
 * `q` is BF16 [256, G K, tokens] (normalised and rotated) for the consecutive positions
 * `first_position` .., read from a device I32 word so that a call captured in a graph replays at
 * whatever position the word holds; `selected` I32 [512, tokens] and `counts` I32 [tokens] are the
 * indexer's blocks; `cache` is the sequence's paged KV layer of K heads of 256 (every KV head of
 * the layer, or a tensor-parallel share of whole KV heads and their query heads) holding every
 * selected and tail position, its block table covering every position the caller reaches, in any
 * storage kv_cache_append writes (BF16, native FP16, the INT8 family, FP8 rows, NVFP4, K8V4), in
 * pages of 64 positions or, for a streamed layer, its device pool of 4-position pages after
 * kv_stream_resolve made every block the call reads resident (kv_stream.h); `out` is BF16 [256, G
 * K, tokens]. The oracle evaluates the softmax and the weighted sum in FP64 from the represented
 * cache and queries: a quantized cache's decoded keys lie in the rotated basis kv_cache_append
 * stores them in, against the query under the same normalized Hadamard transform, and its decoded
 * values as stored, NVFP4 and K8V4 values rotated as well with the transform applied back to the
 * result. The kernel stages a quantized cache's decoded rows in FP16; the output is compared after
 * its BF16 store. Native FP16 rows are read directly without a BF16 staging cast or Hadamard
 * transform, with FP32 accumulation. Up to eight queries keep partial softmaxes in FP32 workspace;
 * no state.
 */
[[nodiscard]] std::size_t sparse_softmax_attention_workspace_bytes(std::int32_t tokens);

void sparse_softmax_attention(const Tensor& q, const Tensor& first_position, const Tensor& selected,
                              const Tensor& counts, const PagedKVLayerView& cache, float scale,
                              WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
