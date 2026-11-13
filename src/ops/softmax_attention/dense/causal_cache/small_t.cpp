// ninfer::ops - host launch of the split-KV causal small-T attention over BF16, bf16p and
// INT8-family caches, and the unified route dispatcher that hands FP8, NVFP4 and K8V4 caches to
// their own launchers. The partial kernels compile in generated units (small_t_bf16_launch.h,
// small_t_i8_launch.h), the reducer and the tile metadata in small_t_reduce.cu.
#include "ops/softmax_attention/dense/causal_cache/small_t_launch.h"

#include "core/paged_kv_storage.h"
#include "ninfer/ops/softmax_attention_envelope.h"
#include "ops/common/device_multiprocessors.h"
#include "ops/common/device_route.h"
#include "ops/common/device_tier.h"
#include "ops/common/math.h"
#include "ops/kv_cache/d256_profile.h"
#include "ops/softmax_attention/dense/causal_cache/cache_input.h"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/softmax_attention/dense/causal_cache/route.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_bf16_launch.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_fp8_launch.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_i8_route.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_invocation.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_k8v4_launch.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_nvfp4_launch.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_policy.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_reduce_launch.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_sm70_shape.h"

#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::ops::detail {

namespace {

// Supplies an upper bound for the device-side active-split policy over one explicit execution
// envelope. Eager calls normally pass an exact window; graph calls pass their target-private
// replay interval. The dtype-aware wrapper below adds the measured INT8 specializations.
template <typename Geometry>
std::int32_t causal_small_t_split_upper_bound(std::int32_t window) {
    if (window <= 0) { return Geometry::SmallTMaximumSplits; }

    constexpr std::int32_t kMinSplits = 4 * Geometry::SmallTSplitScale;
    std::int32_t splits               = kMinSplits;

    const auto include_tier = [&](std::int32_t window_limit, std::int32_t target_keys_per_split) {
        const std::int32_t tier_window = (window < window_limit) ? window : window_limit;
        if (tier_window > 0) {
            const std::int32_t tier_splits = div_up(tier_window, target_keys_per_split);
            splits                         = (splits > tier_splits) ? splits : tier_splits;
        }
    };

    include_tier(4096, 64 / Geometry::SmallTSplitScale);
    if (window > 4096) { include_tier(8198, 128 / Geometry::SmallTSplitScale); }
    if (window > 8198) { include_tier(16390, 256 / Geometry::SmallTSplitScale); }
    if (window > 16390) { include_tier(window, 480 / Geometry::SmallTSplitScale); }

    return (splits < Geometry::SmallTMaximumSplits) ? splits : Geometry::SmallTMaximumSplits;
}

template <typename Geometry>
std::int32_t causal_small_t_split_count(std::int32_t window, std::int32_t tokens,
                                        KvCacheStorage storage) {
    // There used to be a SmallTMaximumSplits bump here for Fp8E4M3Row256 at tokens==1 and
    // window>8198, and the device asked for that bump for *every* quantized storage while only
    // fp8 was granted it. The asymmetry read as the host shortchanging nvfp4 and k8v4 -- 69 splits
    // where their kernels asked for 85 -- and extending the grant was proposed.
    //
    // Measured on this 3090, and it is the other way round: more splits at depth is worse.
    // Extending the grant to nvfp4 and k8v4 cost them 2.3-3.1% on the 35B at every depth, against
    // an fp8 control that moved 0.3%. Removing it from fp8 as well gained 0.4/0.5/1.2% at
    // 4,096/16,384/32,768 over two samples per variant, with within-variant spread well under the
    // difference at the two deeper points.
    //
    // So the bump is gone from both sides -- here, and from
    // causal_small_t_quantized_active_splits -- and every quantized storage now takes the same
    // default tier. That also retires the invariant section 2c flagged as held by coincidence:
    // the partial kernel and the reducer no longer depend on two call sites happening to agree
    // about a special case, because there is no special case.
    //
    // The 27B was not conclusive either way (its fp8 control moved +-3% between identical runs,
    // which is worth knowing before trusting any single-sample result on that model).

    // A 64-key default split just above a 32-key boundary makes the partial kernel execute a
    // nearly empty second tile. T=5 uses one 32-key tile per split; the short T>=6 profile keeps
    // all newly appended rows in one tail split while retaining a useful B=8 grid.
    const bool i8_family = kv_cache_is_int8_family(storage);
    if (i8_family && tokens == 5 && window > 128 && window <= 512) {
        return div_up(window, 32 / Geometry::SmallTSplitScale);
    }
    if (i8_family && tokens >= 6 && window > 128 && window <= 160) {
        constexpr std::int32_t kKeysPerSplit = Geometry::SmallTSplitScale == 2 ? 17 : 24;
        return div_up(window, kKeysPerSplit);
    }
    // Bc=64 is one CTA/SM on these model shapes. Keep the 8K grid at or below
    // one 170-SM wave after accounting for the geometry's KV-head count.
    if (i8_family && tokens >= 6 && window > 5000 && window <= 8198) {
        const std::int32_t splits   = div_up(window, 192 / Geometry::SmallTSplitScale);
        constexpr std::int32_t kMin = 4 * Geometry::SmallTSplitScale;
        constexpr std::int32_t kMax = 42 * Geometry::SmallTSplitScale;
        const std::int32_t clamped  = (splits > kMin) ? splits : kMin;
        return (clamped < kMax) ? clamped : kMax;
    }
    return causal_small_t_split_upper_bound<Geometry>(window);
}

template <typename Geometry>
std::int32_t causal_small_t_launch_capacity(CausalAttentionExecutionEnvelope envelope,
                                            std::int32_t tokens, KvCacheStorage storage) {
    std::int32_t capacity = 0;
    const auto include    = [&](std::uint32_t window) {
        if (window < envelope.min_visible_keys || window > envelope.max_visible_keys) { return; }
        const auto splits = causal_small_t_split_count<Geometry>(static_cast<std::int32_t>(window),
                                                                    tokens, storage);
        capacity          = capacity > splits ? capacity : splits;
    };
    include(envelope.min_visible_keys);
    include(envelope.max_visible_keys);
    // The policy is monotonic inside these finite segments and may drop when crossing a boundary.
    // Evaluating every segment end plus both interval ends gives the exact interval maximum.
    constexpr std::uint32_t ends[] = {128, 160, 512, 4096, 5000, 8198, 16390};
    for (const std::uint32_t end : ends) { include(end); }
    return capacity;
}

// CTAs per SM of the partial kernel launch_tc_partial_i8 selects for this width and window (its
// MinBlocksPerSm): one for the wide-CTA short-window routes, two otherwise, or the device profile's
// tier when it routes one.
template <typename Geometry>
int i8_partial_ctas_per_sm(std::int32_t tokens, std::int32_t window, KvCacheStorage storage) {
#if defined(NINFER_TIER_VOLTA)
    if (current_device_volta()) {
        // Volta's kernel (small_t_sm70_i8.cuh) holds its own count, from its registers and shared
        // memory.
        int ctas = 1;
        [&]<int... T>(std::integer_sequence<int, T...>) {
            ((tokens == T + 1
                  ? (ctas =
                         sm70_attention::i8_resident_ctas<T + 1, (T + 1) * Geometry::GroupSize>())
                  : 0),
             ...);
        }(std::make_integer_sequence<int, 8>{});
        return ctas;
    }
#endif
    if (const int routed =
            small_t_i8_routed_ctas_per_sm(Geometry::ModelQHeads, storage, tokens, window);
        routed > 0) {
        return routed;
    }
    if (tokens >= 6) { return window > 8198 ? 2 : 1; }
    if (tokens == 5) { return window > (Geometry::GroupSize == 6 ? 1029 : 4096) ? 2 : 1; }
    if (tokens == 4) { return window > 1029 ? 2 : 1; }
    return 2;
}

// Splits per full wave of a single-row INT8-family launch. The split tiers were set where their
// largest count (85 splits of four KV heads at two CTAs per SM) is exactly one wave of a 170-SM
// part; on a part with fewer SMs the same counts leave a nearly empty last wave that costs a full
// CTA time. The partial kernel and the reducer take this count and hold their per-window split
// count to whole waves (causal_small_t_active_splits); zero leaves the tiers alone.
template <typename Geometry>
std::int32_t causal_small_t_wave_splits(std::int32_t kv_heads, std::int32_t tokens,
                                        std::int32_t implementation_window,
                                        KvCacheStorage storage) {
    return current_device_multiprocessors(0) *
           i8_partial_ctas_per_sm<Geometry>(tokens, implementation_window, storage) / kv_heads;
}

// The launch capacity the whole-wave policy can reach over the envelope.
template <typename Geometry>
std::int32_t
causal_small_t_wave_capacity(std::int32_t capacity, std::int32_t kv_heads, std::int32_t tokens,
                             CausalAttentionExecutionEnvelope envelope, KvCacheStorage storage) {
    const auto window = static_cast<std::int32_t>(envelope.max_visible_keys);
    const int wave    = causal_small_t_wave_splits<Geometry>(kv_heads, tokens, window, storage);
    if (wave <= 0 || capacity <= wave) { return capacity; }
    return std::min(capacity, wave * div_up(div_up(window, kCausalSmallTSplitKeyLimit), wave));
}

} // namespace

std::int32_t causal_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                             KvCacheStorage cache_storage,
                                             CausalAttentionExecutionEnvelope envelope,
                                             std::int32_t batch_size) {
    return with_causal_query_geometry(q_heads, [&]<typename Geometry>() -> std::int32_t {
        const std::int32_t kv_heads = q_heads / Geometry::GroupSize;
        if (tokens < 1 || tokens > (Geometry::GroupSize == 6 ? 8 : 6) ||
            envelope.min_visible_keys == 0 ||
            envelope.min_visible_keys > envelope.max_visible_keys) {
            throw std::invalid_argument("causal_softmax_attention split capacity: invalid profile");
        }
        (void)paged_kv_storage_layout(cache_storage, kCausalHeadDim);
        if (envelope.invariant_segment_keys != 0) {
            // One split per segment of the envelope's longest window.
            return static_cast<std::int32_t>(
                div_up(envelope.max_visible_keys, envelope.invariant_segment_keys));
        }
        const bool i8_family = kv_cache_is_int8_family(cache_storage);
        const int capacity =
            causal_small_t_launch_capacity<Geometry>(envelope, tokens, cache_storage);
        if constexpr (Geometry::GroupSize == 6) {
            if (batch_size > 1) {
                // Keep complete grids within one or two waves of the current device. Rounding
                // from sms - sms/16 CTAs (160 on the 170-SM part these targets were set on) leaves
                // room for the indivisible KV-heads*B group, including B=3/5/6/7. Targets sized
                // for 170 SMs left a smaller part with partial waves and extra merge traffic.
                const int sms     = current_device_multiprocessors(170);
                const int wave    = sms - sms / 16;
                const bool narrow = tokens <= 5;
                int target_ctas   = wave;
                if (cache_storage == KvCacheStorage::BFloat16 ||
                    cache_storage == KvCacheStorage::Bf16Packed)
                    target_ctas = narrow || batch_size >= 5 || envelope.max_visible_keys > 4096
                                      ? 2 * wave
                                      : wave;
                else if (cache_storage == KvCacheStorage::Int8Group64)
                    target_ctas = narrow || envelope.max_visible_keys > 4096 ? 2 * wave : wave;
                else if (cache_storage == KvCacheStorage::Nvfp4Group16)
                    target_ctas = narrow ? 2 * wave : wave;
                const int grid_limit = div_up(target_ctas, kv_heads * batch_size);
                // A split stages at most 64 physical-page IDs, two of them for key-tile rounding
                // and page alignment; one that must span more reads the block table directly.
                const int page_limit =
                    div_up(static_cast<int>(envelope.max_visible_keys), kCausalSmallTSplitKeyLimit);
                return std::min(capacity, std::max({4, grid_limit, page_limit}));
            }
            return i8_family ? causal_small_t_wave_capacity<Geometry>(capacity, kv_heads, tokens,
                                                                      envelope, cache_storage)
                             : capacity;
        } else {
            return i8_family && batch_size == 1
                       ? causal_small_t_wave_capacity<Geometry>(capacity, kv_heads, tokens,
                                                                envelope, cache_storage)
                       : capacity;
        }
    });
}

template <typename Geometry, typename CacheInput>
void causal_attention_small_t_launch_for(const Tensor& q, CacheInput input, const Tensor& pos,
                                         float scale, PagedKVBatchLayerView cache,
                                         const CausalSmallTInvocation& invocation,
                                         CausalAttentionExecutionEnvelope envelope,
                                         Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                         Tensor& out, cudaStream_t stream,
                                         const void* gate = nullptr) {
    const auto logical_capacity      = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto implementation_window = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto splits    = causal_attention_split_capacity(q.ne[1], invocation.width, cache.storage,
                                                           envelope, invocation.batch_size);
    const bool i8_family = kv_cache_is_int8_family(cache.storage);
    const std::int32_t wave_splits =
        i8_family && invocation.batch_size == 1 && !invocation.partition.segmented()
            ? causal_small_t_wave_splits<Geometry>(q.ne[1] / Geometry::GroupSize, invocation.width,
                                                   implementation_window, cache.storage)
            : 0;

    const SmallTPartialLaunch partial =
        small_t_partial_launch(q, pos, scale, cache, invocation, logical_capacity, splits,
                               partial_acc, partial_m, partial_l, stream);
    if constexpr (!kCausalPackedInput<CacheInput>) {
        if (i8_family) {
            // The INT8 family's tier, coding and PV accumulation are routed on the host
            // (small_t_i8_dispatch.cpp); each tier's kernels compile in units of their own.
            SmallTI8Launch launch{partial};
            launch.wave_splits = wave_splits;
            // A U8 value plane is the packed signed int4 coding.
            launch.coding = causal_i8_coding(cache.storage, cache.v_pages.dtype == DType::U8);
            small_t_i8_partial_launch<Geometry>(invocation.width, cache.storage,
                                                implementation_window, launch, input);
        }
    }
    if (!i8_family || kCausalPackedInput<CacheInput>) {
#if defined(NINFER_TIER_VOLTA)
        if (current_device_volta()) {
            if constexpr (kCausalPackedInput<CacheInput>) {
                throw std::invalid_argument(
                    "causal_softmax_attention: --kv-dtype bf16p needs sm_75 or newer");
            } else {
                // Volta's own kernel (small_t_sm70_bf16.cuh): same grid, splits and partial
                // outputs.
                launch_small_t_bf16_sm70<Geometry, CacheInput>(partial, input);
            }
        } else
#endif
        {
            if constexpr (kBuildPostVolta) {
                launch_small_t_bf16<Geometry, CacheInput>(partial, input);
            }
        }
    }

    SmallTReduceLaunch reduce =
        small_t_reduce_launch(pos, invocation, q.ne[1], logical_capacity, splits, partial_acc,
                              partial_m, partial_l, out, stream);
    reduce.wave_splits = wave_splits;
    reduce.gate        = gate;
    // rk8v4 and rk4v4 are int8-family caches on this fork and take the same path.
    launch_small_t_reduce(reduce, kv_cache_is_int8_family(cache.storage));
}

void causal_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                     const Tensor& pos, const Tensor& valid_columns,
                                     const Tensor& table_rows, float scale,
                                     PagedKVBatchLayerView cache,
                                     CausalAttentionExecutionEnvelope envelope,
                                     std::int32_t column_begin, std::int32_t width,
                                     Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                     Tensor& out, cudaStream_t stream, const void* gate,
                                     float* partial_out) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_small_t_k8v4_launch(q, k, v, pos, valid_columns, table_rows, scale, cache,
                                             envelope, column_begin, width, partial_acc, partial_m,
                                             partial_l, out, stream, partial_out);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_small_t_fp8_launch(q, k, v, pos, valid_columns, table_rows, scale, cache,
                                            envelope, column_begin, width, partial_acc, partial_m,
                                            partial_l, out, stream, partial_out);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_small_t_nvfp4_launch(q, k, v, pos, valid_columns, table_rows, scale, cache,
                                              envelope, column_begin, width, partial_acc, partial_m,
                                              partial_l, out, stream, partial_out);
        return;
    }
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
    if (cache.storage == KvCacheStorage::Bf16Packed) {
        // The call's rows were appended before it (causal_softmax_attention).
        const CausalPackedInput input{static_cast<const std::uint8_t*>(cache.k_scale_pages.data),
                                      static_cast<const std::uint8_t*>(cache.v_scale_pages.data)};
        with_causal_query_geometry(q.ne[1], [&]<typename Geometry>() {
            causal_attention_small_t_launch_for<Geometry>(q, input, pos, scale, cache, invocation,
                                                          envelope, partial_acc, partial_m,
                                                          partial_l, out, stream, gate);
        });
        return;
    }
    const CausalAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                                  static_cast<const __nv_bfloat16*>(v.data)};
    with_causal_query_geometry(q.ne[1], [&]<typename Geometry>() {
        causal_attention_small_t_launch_for<Geometry>(q, input, pos, scale, cache, invocation,
                                                      envelope, partial_acc, partial_m, partial_l,
                                                      out, stream, gate);
    });
}

void causal_attention_cached_small_t_launch(const Tensor& q, const Tensor& pos, float scale,
                                            const PagedKVLayerView& cache,
                                            CausalAttentionExecutionEnvelope envelope,
                                            Tensor& partial_acc, Tensor& partial_m,
                                            Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_cached_small_t_k8v4_launch(q, pos, scale, cache, envelope, partial_acc,
                                                    partial_m, partial_l, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_cached_small_t_fp8_launch(q, pos, scale, cache, envelope, partial_acc,
                                                   partial_m, partial_l, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_cached_small_t_nvfp4_launch(q, pos, scale, cache, envelope, partial_acc,
                                                     partial_m, partial_l, out, stream);
        return;
    }
    const PagedKVBatchLayerView batch_cache = single_row_paged_kv_batch_view(cache);
    const CausalSmallTInvocation invocation{
        .valid_columns = nullptr,
        .table_rows    = nullptr,
        .full_width    = q.ne[2],
        .column_begin  = 0,
        .width         = q.ne[2],
        .batch_size    = 1,
        .partition     = causal_attention_key_partition(envelope, batch_cache),
    };
    const auto launch = [&](auto input) {
        with_causal_query_geometry(q.ne[1], [&]<typename Geometry>() {
            causal_attention_small_t_launch_for<Geometry>(q, input, pos, scale, batch_cache,
                                                          invocation, envelope, partial_acc,
                                                          partial_m, partial_l, out, stream);
        });
    };
    if (cache.storage == KvCacheStorage::Bf16Packed) {
        launch(CausalPackedInput{static_cast<const std::uint8_t*>(cache.k_scale_pages.data),
                                 static_cast<const std::uint8_t*>(cache.v_scale_pages.data)});
    } else {
        launch(CausalCachedInput{});
    }
}

void causal_attention_small_t_load_modules(std::int32_t q_heads, std::int32_t kv_heads,
                                           KvCacheStorage storage, bool cached_input,
                                           std::int32_t max_width) {
    if (!kv_cache_is_int8_family(storage)) { return; }
    // Every INT8-family storage runs the same units (one per tier, width, geometry and input;
    // small_t_i8_launch.h), so loading them serves them all.
    with_causal_geometry(q_heads, kv_heads, [&]<typename Geometry>() {
        constexpr int kWidest = Geometry::ModelQHeads == 24 ? 8 : 6;
        if (cached_input) {
            small_t_i8_load_modules<Geometry, CausalCachedInput>(std::min(max_width, kWidest));
            return;
        }
        small_t_i8_load_modules<Geometry, CausalAppendInput>(std::min(max_width, kWidest));
        // Parallel query tiles run a chunked width as cached launches of its tile width.
        std::int32_t widest_tile = 0;
        for (std::int32_t width = 4; width <= std::min(max_width, 2 * kWidest); ++width) {
            widest_tile = std::max(widest_tile, causal_attention_parallel_tile_width(
                                                    Geometry::ModelQHeads, width, 1, storage));
        }
        small_t_i8_load_modules<Geometry, CausalCachedInput>(widest_tile);
    });
}

std::int32_t causal_attention_parallel_tile_width(std::int32_t q_heads, std::int32_t width,
                                                  std::int32_t batch_size,
                                                  KvCacheStorage storage) {
    if (batch_size != 1 || !kv_cache_is_int8_family(storage)) { return 0; }
    static const int forced = [] {
        const char* value = std::getenv("NINFER_ATTN_PARALLEL_TILES");
        return value == nullptr ? -1 : (value[0] == '1' ? 1 : 0);
    }();
    const bool enabled =
        forced >= 0 ? forced == 1 : device_route_schedule("attn_parallel_tiles", 1) == "on";
    if (!enabled) { return 0; }
    // The widest tile the small-T kernel instantiates for this geometry that divides the width
    // exactly: the tiles are views of one contiguous [D, H, width] block, so a ragged last tile
    // would read past it. A width without such a divisor keeps the serial chunks.
    const std::int32_t widest = causal_attention_model_query_heads(q_heads) == 24 ? 8 : 6;
    for (std::int32_t tile = widest; tile >= 2; --tile) {
        if (width % tile == 0 && width / tile >= 2) { return tile; }
    }
    return 0;
}

void causal_attention_small_t_tiles_launch(const Tensor& q, const Tensor& positions,
                                           const Tensor* valid_columns, const Tensor* table_rows,
                                           float scale, PagedKVBatchLayerView cache,
                                           CausalAttentionExecutionEnvelope envelope,
                                           std::int32_t tile_width, Tensor& tile_valid,
                                           Tensor& tile_rows, Tensor& partial_acc,
                                           Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                           cudaStream_t stream, const void* gate,
                                           float* partial_out) {
    const std::int32_t tiles = q.ne[2] / tile_width;
    launch_small_t_tile_metadata(
        valid_columns == nullptr ? nullptr
                                 : static_cast<const std::int32_t*>(valid_columns->data),
        table_rows == nullptr ? nullptr : static_cast<const std::int32_t*>(table_rows->data),
        tile_width, tiles,
        valid_columns == nullptr ? nullptr : static_cast<std::int32_t*>(tile_valid.data),
        static_cast<std::int32_t*>(tile_rows.data), stream);
    // Each tile is one batch row of the multi-batch kernel over the cache the caller already
    // appended: batch t starts at column t * full_width, the same column t * tile_width of q,
    // positions and out.
    const CausalSmallTInvocation invocation{
        .valid_columns = valid_columns == nullptr ? nullptr : &tile_valid,
        .table_rows    = &tile_rows,
        .full_width    = tile_width,
        .column_begin  = 0,
        .width         = tile_width,
        .batch_size    = tiles,
        .partition     = causal_attention_key_partition(envelope, cache),
        .partial_out   = partial_out,
    };
    const CausalCachedInput input{};
    with_causal_query_geometry(q.ne[1], [&]<typename Geometry>() {
        causal_attention_small_t_launch_for<Geometry>(q, input, positions, scale, cache, invocation,
                                                      envelope, partial_acc, partial_m, partial_l,
                                                      out, stream, gate);
    });
}
} // namespace ninfer::ops::detail
