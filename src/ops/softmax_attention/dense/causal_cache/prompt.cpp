// ninfer::ops - causal_softmax_attention prompt-scale launcher: fill k/v at device positions then
// launch causal attention over absolute cached history. The kernels compile in generated units
// (prompt_kernel_launch.h), the fast kernel's split merge in int8/prompt_fast_merge.cu.
#include "ops/softmax_attention/dense/causal_cache/prompt_launch.h"

#include "core/arena.h"
#include "core/paged_kv_view.h"
#include "ops/common/device_multiprocessors.h"
#include "ops/common/device_route.h"
#include "ops/common/device_tier.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/kv_cache/d256_profile.h"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/softmax_attention/dense/causal_cache/fp8/prompt_launch.h"
#include "ops/softmax_attention/dense/causal_cache/int8/prompt_fast_plan.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/prompt_launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_kernel_launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/prompt_launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_policy.h"
#include "ops/softmax_attention/dense/causal_cache/route.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops::detail {

namespace {

// Sums each key tile's PV products in FP16 on the packed-value INT8 prompt kernels where the
// device profile's "attn_pv_f16" says "on"; NINFER_PROMPT_PV_F16=0|1 overrides the profile.
[[maybe_unused]] bool prompt_pv_f16() {
    static const int forced = [] {
        const char* value = std::getenv("NINFER_PROMPT_PV_F16");
        return value == nullptr ? -1 : (value[0] == '1' ? 1 : 0);
    }();
    if (forced >= 0) { return forced == 1; }
    return device_route_schedule("attn_pv_f16", 1) == "on";
}

// Packs one KV head's query heads into each INT8 prompt tile (causal_attention_prompt_i8_kernel's
// PackGqa) where the device profile's "attn_pack_gqa" says "on"; NINFER_PROMPT_PACK_GQA=0|1
// overrides the profile. The fast kernel already issues a row block's heads together.
[[maybe_unused]] bool prompt_pack_gqa() {
    static const int forced = [] {
        const char* value = std::getenv("NINFER_PROMPT_PACK_GQA");
        return value == nullptr ? -1 : (value[0] == '1' ? 1 : 0);
    }();
    if (forced >= 0) { return forced == 1; }
    return device_route_schedule("attn_pack_gqa", 1) == "on";
}

// The launch arguments every prompt kernel takes; the caller adds the block-table metadata.
template <typename CacheView>
PromptLaunch prompt_launch(const Tensor& q, const Tensor& positions, float scale,
                           const CacheView& cache, Tensor& out, cudaStream_t stream) {
    PromptLaunch launch{};
    launch.q             = q.data;
    launch.positions     = static_cast<const std::int32_t*>(positions.data);
    launch.scale         = scale;
    launch.out           = out.data;
    launch.tokens        = static_cast<std::int32_t>(q.ne[2]);
    launch.q_heads       = static_cast<std::int32_t>(q.ne[1]);
    launch.cache_k       = cache.k_pages.data;
    launch.cache_v       = cache.v_pages.data;
    launch.cache_k_scale = cache.k_scale_pages.data;
    launch.cache_v_scale = cache.v_scale_pages.data;
    launch.stream        = stream;
    return launch;
}

template <typename Geometry, PromptMetadataForm Form, typename CacheView>
void causal_attention_prompt_attention_launch_for(const CacheView& cache, PromptLaunch launch,
                                                  std::uint32_t max_visible_keys,
                                                  WorkspaceArena& workspace, bool fast) {
    fast = causal_attention_prompt_fast_kernel(cache.storage, fast);
    // The fast kernels hold 100 KiB of shared memory, past sm_70's 96 KiB: a Volta card never takes
    // them (causal_attention_prompt_fast_kernel), and a build for Volta alone does not compile
    // them.
    if constexpr (kBuildPostVolta) {
        if (fast && kv_cache_is_int8_family(cache.storage)) {
            // The int8 cache and rk8v4 share the fast kernel's INT8 G64 keys; rk8v4 and the packed
            // key codings decode packed int4 values.
            launch.coding = causal_i8_coding(cache.storage,
                                             cache.storage != KvCacheStorage::Int8Group64);
            const CausalPromptI8FastPlan plan = causal_prompt_i8_fast_plan(
                launch.q_heads, launch.tokens, max_visible_keys,
                current_device_multiprocessors(170), causal_prompt_i8_fast_wide(cache.storage),
                causal_attention_prompt_fast_split_policy());
            auto scope = workspace.scope();
            CausalPromptI8FastPartials partials{};
            if (plan.splits > 1) {
                partials = allocate_causal_prompt_i8_fast_partials(workspace, launch.q_heads,
                                                                   launch.tokens, plan.splits);
            }
            launch.warps       = plan.warps;
            launch.splits      = plan.splits;
            launch.split_rows  = static_cast<float*>(partials.rows.data);
            launch.split_stats = static_cast<float2*>(partials.stats.data);
            launch_prompt_i8_fast<Geometry, Form>(launch);
            if (plan.splits > 1) { launch_prompt_i8_fast_merge(launch); }
            return;
        }
    }
    if (kv_cache_is_int8_family(cache.storage)) {
        if (!causal_prompt_i8_standard_here()) {
            // Every INT8-family coding took the fast kernel above
            // (causal_attention_prompt_fast_kernel).
            throw std::logic_error(
                "INT8-family prompt attention reached the standard kernel, which this card's shape "
                "set lacks");
        }
        // A build of Turing's shape set alone does not compile the standard kernel.
        if constexpr (kBuildStandardShapes) {
            // A U8 key plane is a packed key coding (rk4v4, rk4v4-e8 or rk2v4-e8), told apart by
            // storage; a U8 value plane is the packed signed int4 coding they share with rk8v4.
            launch.coding = causal_i8_coding(cache.storage, cache.v_pages.dtype == DType::U8);
            // A Volta card emulates these tiles and takes one variant per coding: no packed GQA
            // and no FP16 PV sums. rk2v4-e8 and INT8 values sum in FP32 everywhere.
            launch.pack_gqa =
                Geometry::GroupSize > 1 && prompt_pack_gqa() && !current_device_volta();
            const bool pv_coding = launch.coding == CausalI8Coding::Lloyd4 ||
                                   launch.coding == CausalI8Coding::Int4E8 ||
                                   launch.coding == CausalI8Coding::PackedValues;
            launch.pv_f16 = pv_coding && !current_device_volta() && prompt_pv_f16();
            if (launch.coding == CausalI8Coding::Int8) {
                launch_prompt_i8<Geometry, Form, false>(launch);
            } else {
                launch_prompt_i8<Geometry, Form, true>(launch);
            }
        }
    } else {
        launch.packed_planes = cache.storage == KvCacheStorage::Bf16Packed;
        launch_prompt_bf16<Geometry, Form>(launch);
    }
}

} // namespace

bool causal_prompt_i8_fast_wide(KvCacheStorage storage) {
    return prompt_i8_fast_wide_launchable(storage != KvCacheStorage::Int8Group64,
                                          turing_shapes_here());
}

bool causal_attention_prompt_fast_kernel(KvCacheStorage storage, bool requested) {
    // The INT8-family and nvfp4 fast kernels hold 100 KiB of shared memory, past a Volta card's
    // 96 KiB, and the FP8 fast kernel compiles empty for Volta.
    if (current_device_volta()) { return false; }
    const bool supported = storage == KvCacheStorage::Int8Group64 ||
                           storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64 ||
                           storage == KvCacheStorage::RotatedLloyd4KeyInt4Value ||
                           storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8 ||
                           storage == KvCacheStorage::RotatedE8RootKeyInt4Value ||
                           storage == KvCacheStorage::Nvfp4Group16 ||
                           storage == KvCacheStorage::Fp8E4M3Row256;
    if (!supported) { return false; }
    // A card without the standard INT8-family prompt kernel (causal_prompt_i8_standard) takes the
    // fast one for every INT8-family coding; the FP8 cache keeps its standard kernel there, so it
    // follows the usual selection.
    if (!causal_prompt_i8_standard_here() && storage != KvCacheStorage::Nvfp4Group16 &&
        storage != KvCacheStorage::Fp8E4M3Row256) {
        return true;
    }
    if (requested) { return true; }
    static const int forced = [] {
        const char* value = std::getenv("NINFER_PROMPT_FAST");
        return value == nullptr ? -1 : (value[0] == '1' ? 1 : 0);
    }();
    if (forced >= 0) { return forced == 1; }
    // The profile key was measured on the INT8-family kernel only, and the FP8 fast kernel changes
    // the output: a default --kv-dtype fp8 run must stay byte-identical to the standard route, so
    // the profile never selects it. The flag or NINFER_PROMPT_FAST=1 (diagnosis) is its only route.
    if (storage == KvCacheStorage::Fp8E4M3Row256) { return false; }
    return device_route_schedule("attn_prompt_fast", 1) == "on";
}

CausalPromptI8FastSplitPolicy causal_attention_prompt_fast_split_policy() {
    static const int forced = [] {
        const char* value = std::getenv("NINFER_PROMPT_SPLITS");
        return value == nullptr ? -1 : std::max(0, std::atoi(value));
    }();
    if (forced == 0) { return {.maximum = 1}; }
    if (forced > 0) { return {.maximum = kCausalPromptI8FastMaxSplits, .forced = forced}; }
    if (device_route_schedule("attn_prompt_split", 1) == "off") { return {.maximum = 1}; }
    return {};
}

void causal_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                              const PagedKVLayerView& cache,
                                              std::uint32_t max_visible_keys,
                                              WorkspaceArena& workspace, Tensor& out, bool fast,
                                              cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_prompt_k8v4_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_prompt_nvfp4_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_prompt_fp8_attention_launch(q, positions, scale, cache, max_visible_keys,
                                                     workspace, out, fast, stream);
        return;
    }
    PromptLaunch launch = prompt_launch(q, positions, scale, cache, out, stream);
    launch.tables       = static_cast<const std::int32_t*>(cache.block_table.data);
    with_causal_query_geometry(q.ne[1], [&]<typename Geometry>() {
        causal_attention_prompt_attention_launch_for<Geometry, PromptMetadataForm::Direct>(
            cache, launch, max_visible_keys, workspace, fast);
    });
}

void causal_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& positions, const Tensor& valid_columns,
                                    const Tensor& table_rows, float scale,
                                    PagedKVBatchLayerView cache, std::uint32_t max_visible_keys,
                                    WorkspaceArena& workspace, Tensor& out, bool fast,
                                    cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_prompt_k8v4_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                            cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_prompt_nvfp4_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                             cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_prompt_fp8_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                           cache, max_visible_keys, workspace, out, fast, stream);
        return;
    }
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    PromptLaunch launch = prompt_launch(q, positions, scale, cache, out, stream);
    launch.tables       = static_cast<const std::int32_t*>(cache.block_tables.data);
    launch.table_rows   = static_cast<const std::int32_t*>(table_rows.data);
    launch.table_stride = cache.block_tables.ne[0];
    const auto attend   = [&]<PromptMetadataForm Form>() {
        with_causal_query_geometry(q.ne[1], [&]<typename Geometry>() {
            causal_attention_prompt_attention_launch_for<Geometry, Form>(cache, launch,
                                                                         max_visible_keys,
                                                                         workspace, fast);
        });
    };
    if (valid_columns.data == nullptr) {
        attend.template operator()<PromptMetadataForm::Batch>();
    } else {
        launch.valid_columns = static_cast<const std::int32_t*>(valid_columns.data);
        attend.template operator()<PromptMetadataForm::BatchMasked>();
    }
}

} // namespace ninfer::ops::detail
