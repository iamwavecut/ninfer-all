// ninfer::ops::detail - row-scaled E4M3FN-cache causal prompt launch ownership.
#include "ops/softmax_attention/dense/causal_cache/fp8/prompt_launch.h"

#include "core/device_launch.h"
#include "core/paged_kv_view.h"
#include "ops/common/device_multiprocessors.h"
#include "ops/common/math.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/kv_cache/plane_types.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_fast_merge.cuh"
#include "ops/softmax_attention/dense/causal_cache/fp8/prompt.cuh"
#include "ops/softmax_attention/dense/causal_cache/fp8/prompt_fast.cuh"
#include "ops/softmax_attention/dense/causal_cache/int8/prompt_fast_plan.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_policy.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

static_assert(kPromptWaveRows % CausalPromptFp8FastShape<>::Br == 0);

template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_fp8_fast_launch_for(const Tensor& q, const Tensor& positions,
                                                 float scale, const CacheView& cache,
                                                 Metadata metadata, const std::int32_t* valid,
                                                 std::uint32_t max_visible_keys,
                                                 WorkspaceArena& workspace, Tensor& out,
                                                 cudaStream_t stream) {
    constexpr auto kStorage = KvCacheStorage::Fp8E4M3Row256;
    const auto* cache_k_ptr = static_cast<const KvKeyCodeT<kStorage>*>(cache.k_pages.data);
    const auto* cache_v_ptr = static_cast<const KvValueCodeT<kStorage>*>(cache.v_pages.data);
    const auto* k_scale_ptr = static_cast<const KvKeyScaleT<kStorage>*>(cache.k_scale_pages.data);
    const auto* v_scale_ptr = static_cast<const KvValueScaleT<kStorage>*>(cache.v_scale_pages.data);
    assert_kv_planes<kStorage, decltype(cache_k_ptr), decltype(cache_v_ptr), decltype(k_scale_ptr),
                     decltype(v_scale_ptr)>();
    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    with_shape_set([&]<bool TuringShapes>() {
        using Shape = CausalPromptFp8FastShape<TuringShapes>;
        if constexpr (!Shape::Launchable) { return; }
        // The fast FP8 CTA is four warps (fp8/prompt_fast.cuh), so the plan's wide candidate never
        // applies.
        const CausalPromptI8FastPlan plan = causal_prompt_i8_fast_plan(
            q.ne[1], tokens, max_visible_keys, current_device_multiprocessors(170), false,
            causal_attention_prompt_fast_split_policy());
        auto scope = workspace.scope();
        CausalPromptI8FastPartials partials{};
        if (plan.splits > 1) {
            partials =
                allocate_causal_prompt_i8_fast_partials(workspace, q.ne[1], tokens, plan.splits);
        }
        const auto launch = [&]<bool Split>() {
            const auto kernel =
                causal_attention_prompt_fp8_fast_kernel<Geometry, Metadata, TuringShapes, Split>;
            raise_dynamic_shared_memory(kernel, Shape::SmemBytes);
            const dim3 grid(static_cast<unsigned>(div_up(tokens, Shape::Br)),
                            static_cast<unsigned>(q.ne[1]), static_cast<unsigned>(plan.splits));
            kernel<<<grid, Shape::Threads, Shape::SmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), cache_k_ptr, cache_v_ptr, k_scale_ptr,
                v_scale_ptr, metadata, static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens,
                static_cast<float*>(partials.rows.data),
                static_cast<float2*>(partials.stats.data));
        };
        if (plan.splits == 1) {
            launch.template operator()<false>();
        } else {
            launch.template operator()<true>();
            CUDA_CHECK(cudaGetLastError());
            constexpr float Log2E = 1.4426950408889634074f;
            causal_attention_prompt_fast_merge_kernel<Geometry>
                <<<dim3(static_cast<unsigned>(tokens), static_cast<unsigned>(q.ne[1])),
                   kCausalPromptHeadDim, 0, stream>>>(
                    static_cast<const float*>(partials.rows.data),
                    static_cast<const float2*>(partials.stats.data), valid, tokens, plan.splits,
                    scale * Log2E, static_cast<__nv_bfloat16*>(out.data));
        }
    });
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_fp8_attention_launch_for(const Tensor& q, const Tensor& positions,
                                                      float scale, const CacheView& cache,
                                                      Metadata metadata,
                                                      const std::int32_t* valid,
                                                      std::uint32_t max_visible_keys,
                                                      WorkspaceArena& workspace, Tensor& out,
                                                      bool fast, cudaStream_t stream) {
    constexpr auto kStorage = KvCacheStorage::Fp8E4M3Row256;
    const auto* cache_k_ptr = static_cast<const KvKeyCodeT<kStorage>*>(cache.k_pages.data);
    const auto* cache_v_ptr = static_cast<const KvValueCodeT<kStorage>*>(cache.v_pages.data);
    const auto* k_scale_ptr = static_cast<const KvKeyScaleT<kStorage>*>(cache.k_scale_pages.data);
    const auto* v_scale_ptr = static_cast<const KvValueScaleT<kStorage>*>(cache.v_scale_pages.data);
    assert_kv_planes<kStorage, decltype(cache_k_ptr), decltype(cache_v_ptr), decltype(k_scale_ptr),
                     decltype(v_scale_ptr)>();
    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    if (causal_attention_prompt_fast_kernel(kStorage, fast)) {
        bool taken = false;
        with_shape_set([&]<bool TuringShapes>() {
            if constexpr (CausalPromptFp8FastShape<TuringShapes>::Launchable) {
                causal_attention_prompt_fp8_fast_launch_for<Geometry>(
                    q, positions, scale, cache, metadata, valid, max_visible_keys, workspace, out,
                    stream);
                taken = true;
            }
        });
        if (taken) { return; }
    }
    with_shape_set([&]<bool TuringShapes>() {
        using Shape       = CausalPromptFp8Shape<TuringShapes>;
        const auto kernel = causal_attention_prompt_fp8_kernel<Geometry, Metadata, TuringShapes>;
        raise_dynamic_shared_memory(kernel, Shape::SmemBytes);
        const dim3 grid(static_cast<unsigned>(div_up(tokens, Shape::Br)),
                        static_cast<unsigned>(q.ne[1]), 1u);
        kernel<<<grid, kCausalPromptFp8Threads, Shape::SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), cache_k_ptr, cache_v_ptr, k_scale_ptr,
            v_scale_ptr, metadata, static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens);
    });
    CUDA_CHECK(cudaGetLastError());
}

template <typename CacheView, typename Metadata>
void causal_attention_prompt_fp8_attention_dispatch(const Tensor& q, const Tensor& positions,
                                                    float scale, const CacheView& cache,
                                                    Metadata metadata,
                                                    const std::int32_t* valid,
                                                    std::uint32_t max_visible_keys,
                                                    WorkspaceArena& workspace, Tensor& out,
                                                    bool fast, cudaStream_t stream) {
    with_causal_query_geometry(q.ne[1], [&]<typename Geometry>() {
        causal_attention_prompt_fp8_attention_launch_for<Geometry>(
            q, positions, scale, cache, metadata, valid, max_visible_keys, workspace, out, fast,
            stream);
    });
}

} // namespace

void causal_attention_prompt_fp8_attention_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  std::uint32_t max_visible_keys,
                                                  WorkspaceArena& workspace, Tensor& out, bool fast,
                                                  cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    causal_attention_prompt_fp8_attention_dispatch(q, positions, scale, cache, metadata,
                                                   static_cast<const std::int32_t*>(nullptr),
                                                   max_visible_keys, workspace, out, fast, stream);
}

void causal_attention_prompt_fp8_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                        const Tensor& positions, const Tensor& valid_columns,
                                        const Tensor& table_rows, float scale,
                                        PagedKVBatchLayerView cache, std::uint32_t max_visible_keys,
                                        WorkspaceArena& workspace, Tensor& out, bool fast,
                                        cudaStream_t stream) {
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        const std::int32_t* valid = metadata.valid_columns;
        causal_attention_prompt_fp8_attention_dispatch(q, positions, scale, cache, metadata, valid,
                                                       max_visible_keys, workspace, out, fast,
                                                       stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
