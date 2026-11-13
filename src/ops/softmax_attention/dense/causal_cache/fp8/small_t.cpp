// ninfer::ops::detail - host launch of the FP8 E4M3 row-scaled split-KV small-T attention. The
// partial kernels compile in the generated units of fp8/small_t_launch.cuh (and of Volta's
// sm70/small_t_quant_launch.cuh), the reducer in small_t_reduce.cu.
#include "ops/softmax_attention/dense/causal_cache/fp8/small_t_launch.h"

#include "core/paged_kv_view.h"
#include "ops/common/device_tier.h"
#include "ops/softmax_attention/dense/causal_cache/cache_input.h"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/softmax_attention/dense/causal_cache/route.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_invocation.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_quantized_launch.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_reduce_launch.h"
#include "ops/softmax_attention/dense/causal_cache/sm70/small_t_quant_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, typename CacheInput>
void causal_attention_small_t_fp8_launch_for(const Tensor& q, CacheInput input,
                                             const Tensor& positions, float scale,
                                             PagedKVBatchLayerView cache,
                                             const CausalSmallTInvocation& invocation,
                                             CausalAttentionExecutionEnvelope envelope,
                                             Tensor& partial_acc, Tensor& partial_m,
                                             Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    const auto logical_capacity = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto splits = causal_attention_split_capacity(q.ne[1], invocation.width, cache.storage,
                                                        envelope, invocation.batch_size);
    const SmallTPartialLaunch partial =
        small_t_partial_launch(q, positions, scale, cache, invocation, logical_capacity, splits,
                               partial_acc, partial_m, partial_l, stream);
    const auto launch_partial = [&] {
#if defined(NINFER_TIER_VOLTA)
        if (current_device_volta()) {
            // Volta's own kernel (sm70/small_t_quant.cuh): the same grid, splits and partial
            // outputs. The tiled kernel is the other tiers'; a build of Volta alone does not
            // compile it.
            launch_small_t_quantized_sm70<Geometry, CacheInput, true, true>(partial, input);
            return;
        }
#endif
        if constexpr (kBuildPostVolta) {
            launch_small_t_fp8<Geometry, CacheInput>(partial, input);
        }
    };
    launch_partial();
    const SmallTReduceLaunch reduce =
        small_t_reduce_launch(positions, invocation, out.ne[1], logical_capacity, splits,
                              partial_acc, partial_m, partial_l, out, stream);
    launch_small_t_reduce(reduce, false);
}

} // namespace

void causal_attention_small_t_fp8_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream,
    float* partial_out) {
    const CausalAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                                  static_cast<const __nv_bfloat16*>(v.data)};
    const CausalSmallTInvocation invocation{
        .valid_columns = valid_columns.data == nullptr ? nullptr : &valid_columns,
        .table_rows    = &table_rows,
        .full_width    = q.ne[2],
        .column_begin  = column_begin,
        .width         = width,
        .batch_size    = q.ne[3],
        .partition     = causal_attention_key_partition(envelope, cache),
        .partial_out   = partial_out,
    };
    with_causal_query_geometry(q.ne[1], [&]<typename Geometry>() {
        causal_attention_small_t_fp8_launch_for<Geometry>(q, input, positions, scale, cache,
                                                          invocation, envelope, partial_acc,
                                                          partial_m, partial_l, out, stream);
    });
}

void causal_attention_cached_small_t_fp8_launch(const Tensor& q, const Tensor& positions,
                                                float scale, const PagedKVLayerView& cache,
                                                CausalAttentionExecutionEnvelope envelope,
                                                Tensor& partial_acc, Tensor& partial_m,
                                                Tensor& partial_l, Tensor& out,
                                                cudaStream_t stream) {
    const CausalCachedInput input{};
    PagedKVBatchLayerView batch_cache = single_row_paged_kv_batch_view(cache);
    const CausalSmallTInvocation invocation{
        .valid_columns = nullptr,
        .table_rows    = nullptr,
        .full_width    = q.ne[2],
        .column_begin  = 0,
        .width         = q.ne[2],
        .batch_size    = 1,
        .partition     = causal_attention_key_partition(envelope, batch_cache),
    };
    with_causal_query_geometry(q.ne[1], [&]<typename Geometry>() {
        causal_attention_small_t_fp8_launch_for<Geometry>(q, input, positions, scale, batch_cache,
                                                          invocation, envelope, partial_acc,
                                                          partial_m, partial_l, out, stream);
    });
}

} // namespace ninfer::ops::detail
