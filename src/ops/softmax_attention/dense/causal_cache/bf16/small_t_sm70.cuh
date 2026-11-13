#pragma once

// ninfer::ops - split-KV causal small-T attention over a BF16 KV cache on sm_70 (Tesla V100), the
// partial kernel of a build with 70. The Ampere kernel (bf16/small_t.cuh) runs there on emulated
// ldmatrix and m16n8k16 tiles; this one uses Volta's mma.m8n8k4 on FP16 operands
// (sm70/small_t_common.cuh), converting BF16 to FP16 (saturated at +-65504) as it loads K and V.

#include "ops/softmax_attention/dense/causal_cache/sm70/small_t_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

namespace sm70_attention {

// K and V of a BF16 cache: the current step's rows from the append input, the rest from the pages.
// kExact: the rows also give the BF16 values of the CTA's queries (query) and keys and values
// (key_bf16, value_bf16), and key8 and value4 flag what they saturate.
template <typename Geometry, typename CacheInput>
struct Bf16Rows {
    static constexpr bool kExact = true;
    const __nv_bfloat16* cache_k;
    const __nv_bfloat16* cache_v;
    CacheInput input;
    const std::int32_t* pages;
    const std::int32_t* block_table;
    int first_page;
    bool staged_pages;
    int kv_head;
    int kv_heads; // the call's KV heads
    int first_pos;
    int valid_tokens;
    const __nv_bfloat16* q; // the call's queries from the CTA's column on
    int tokens;
    int row_count;
    bool query_beyond; // some query of the CTA is beyond FP16

    __device__ __forceinline__ const __nv_bfloat16* query(int row) const {
        int q_head = 0;
        int token  = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        return row < row_count && causal_valid_q_head<Geometry>(kv_head, q_head, kv_heads)
                   ? q + causal_q_index(q_head, 0, token, kv_heads * Geometry::GroupSize)
                   : nullptr;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_bf16(int k) const { return key(k); }
    __device__ __forceinline__ const __nv_bfloat16* value_bf16(int k) const {
        return value(k, 0);
    }
    __device__ __forceinline__ bool keys_beyond(unsigned bits) const {
        return __any_sync(0xffffffffu, (bits & kBeyondBits) != 0u);
    }
    __device__ __forceinline__ bool values_beyond(unsigned bits) const {
        return __any_sync(0xffffffffu, (bits & kBeyondBits) != 0u);
    }

    __device__ __forceinline__ const __nv_bfloat16* address(const __nv_bfloat16* plane, bool keys,
                                                            int key, int d) const {
        if constexpr (CacheInput::writes_cache) {
            const int new_token = key - first_pos;
            if (key >= first_pos && new_token < valid_tokens) {
                return (keys ? input.k : input.v) +
                       kv_cache_int8_new_index(kv_head, d, new_token, kv_heads);
            }
        }
        const int page = staged_pages ? pages[(key >> kPagedKVPageShift) - first_page]
                                      : block_table[key >> kPagedKVPageShift];
        return plane + causal_cache_index(page, kv_head, d, key & kPagedKVPageMask, kv_heads);
    }

    __device__ __forceinline__ const __nv_bfloat16* key(int k) const {
        return address(cache_k, true, k, 0);
    }

    __device__ __forceinline__ uint4 key8(const __nv_bfloat16* row, int d, unsigned& bits) const {
        const uint4 b = __ldg(reinterpret_cast<const uint4*>(row + d));
        bits |= beyond_bits(b.x) | beyond_bits(b.y) | beyond_bits(b.z) | beyond_bits(b.w);
        return make_uint4(to_f16x2(b.x), to_f16x2(b.y), to_f16x2(b.z), to_f16x2(b.w));
    }

    __device__ __forceinline__ const __nv_bfloat16* value(int k, int dim0) const {
        return address(cache_v, false, k, dim0);
    }

    __device__ __forceinline__ uint2 value4(const __nv_bfloat16* row, int n, unsigned& bits) const {
        const uint2 b = __ldg(reinterpret_cast<const uint2*>(row + 32 * n));
        bits |= beyond_bits(b.x) | beyond_bits(b.y);
        return make_uint2(to_f16x2(b.x), to_f16x2(b.y));
    }
};

} // namespace sm70_attention

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
__launch_bounds__(32 * sm70_attention::Shape<TokenTile * Geometry::GroupSize>::kWarps) __global__
    void causal_attention_small_t_sm70_bf16_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, __nv_bfloat16* cache_k,
        __nv_bfloat16* cache_v, const std::int32_t* block_tables, const std::int32_t* valid_columns,
        const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t tokens,
        std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity,
        float scale, float* partial_acc, float* partial_m, float* partial_l,
        CausalKeyPartition partition) {
    using Shape            = sm70_attention::Shape<TokenTile * Geometry::GroupSize>;
    constexpr int D        = kCausalHeadDim;
    constexpr int Bc       = 32;
    constexpr int Groups   = Shape::kRowGroups;
    constexpr int KeyWarps = Shape::kKeyWarps;
    constexpr int Threads  = 32 * Shape::kWarps;
    constexpr int QStride  = sm70_attention::kQStride;
    constexpr int PStride  = sm70_attention::kPStride;
    constexpr int PageIds  = 64;

    __shared__ __align__(16) __half q_s[Groups * 8 * QStride];
    __shared__ __align__(16) __half p_s[Shape::kWarps * 8 * PStride];
    __shared__ std::int32_t physical_pages_s[PageIds];
    __shared__ float stat_m[Shape::kWarps][8];
    __shared__ float stat_l[Shape::kWarps][8];
    __shared__ float combined[KeyWarps > 1 ? Groups * 8 * D : 1];

    const int kv_head     = static_cast<int>(blockIdx.x);
    // The grid spans the call's KV heads.
    const int kv_heads    = static_cast<int>(gridDim.x);
    const int q_heads     = kv_heads * Geometry::GroupSize;
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    int valid_tokens      = tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }
    const int row_count = tokens * Geometry::GroupSize;

    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(D) * q_heads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(D) * kv_heads * column_base;
        input.v += static_cast<std::int64_t>(D) * kv_heads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * D * q_heads * tokens * split_count;
        partial_m += static_cast<std::int64_t>(batch) * q_heads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * q_heads * tokens * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head, kv_heads)) {
                partial_m[causal_partial_stat_index(q_head, token, split, tokens, q_heads)] =
                    -CUDART_INF_F;
                partial_l[causal_partial_stat_index(q_head, token, split, tokens, q_heads)] = 0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head, kv_heads)) {
                partial_acc[causal_partial_acc_index(q_head, d, token, split, tokens, q_heads)] =
                    0.0f;
            }
        }
    };

    if (kv_head < 0 || kv_head >= kv_heads || tokens < 1 || tokens > TokenTile ||
        row_count > Groups * 8 || split_count <= 0) {
        return;
    }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int live_end = last_pos + 1;
    const int window   = causal_small_t_tile_window(first_pos, tokens, logical_capacity);
    const int active_split_count = causal_small_t_partition_active(
        partition, window,
        causal_small_t_active_splits<Geometry, false>(window, split_count, TokenTile),
        split_count);
    if (split >= active_split_count) { return; }

    int split_start = 0;
    int split_end   = 0;
    if (partition.segmented() || partition.strided()) {
        causal_small_t_partition_range(partition, window, split, active_split_count, split_start,
                                       split_end);
    } else {
        const int logical_tiles = div_up(window, Bc);
        const bool tile_split   = logical_tiles >= active_split_count;
        const int units_per_split = tile_split ? div_up(logical_tiles, active_split_count)
                                               : div_up(window, active_split_count);
        split_start           = split * units_per_split * (tile_split ? Bc : 1);
        const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
        split_end             = (split_limit < window) ? split_limit : window;
    }
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    // Keys at or past the live end are beyond every live column: none is loaded.
    const int load_end = split_end < live_end ? split_end : live_end;

    const int first_tile    = (split_start / Bc) * Bc;
    const int key_blocks    = div_up(split_end - first_tile, Bc);
    const int first_page    = first_tile >> kPagedKVPageShift;
    const int page_count    = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    const bool staged_pages = page_count <= PageIds;
    if (staged_pages) {
        for (int page = tid; page < page_count; page += Threads) {
            physical_pages_s[page] = block_table[first_page + page];
        }
    }
    __syncthreads();
    // A strided split's staged pages become the rank's own, -1 where another rank holds one.
    if (partition.strided()) {
        causal_small_t_localize_pages(physical_pages_s, page_count, partition);
    }

    if constexpr (CacheInput::writes_cache) {
        // The owning split writes each new row. Current attention reads those rows directly from
        // input below, so no split depends on another split's cache write.
        for (int chunk = tid; chunk < valid_tokens * (D / 8); chunk += Threads) {
            const int token = chunk / (D / 8);
            const int d     = (chunk - token * (D / 8)) * 8;
            const int p_tok = pos[token];
            if (p_tok >= split_start && p_tok < split_end && p_tok >= 0 &&
                p_tok < logical_capacity) {
                const int page = partition.strided()
                                     ? physical_pages_s[(p_tok >> kPagedKVPageShift) - first_page]
                                     : paged_kv_physical_page(block_table, p_tok);
                if (page < 0) { continue; } // another rank's planes hold this row
                const std::int64_t new_off = kv_cache_int8_new_index(kv_head, d, token, kv_heads);
                const std::int64_t cache_off =
                    causal_cache_index(page, kv_head, d, p_tok & kPagedKVPageMask, kv_heads);
                store_vec(&cache_k[cache_off], load_vec<int4>(&input.k[new_off]));
                store_vec(&cache_v[cache_off], load_vec<int4>(&input.v[new_off]));
            }
        }
    }

    // The CTA's query rows in FP16, zero past row_count.
    unsigned query_bits = 0u;
    for (int idx = tid; idx < Groups * 8 * (D / 2); idx += Threads) {
        const int row = idx / (D / 2);
        const int d   = (idx - row * (D / 2)) * 2;
        int q_head    = 0;
        int token     = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        unsigned value = 0;
        if (row < row_count && causal_valid_q_head<Geometry>(kv_head, q_head, kv_heads)) {
            const unsigned word =
                *reinterpret_cast<const unsigned*>(&q[causal_q_index(q_head, d, token, q_heads)]);
            query_bits |= sm70_attention::beyond_bits(word);
            value = sm70_attention::to_f16x2(word);
        }
        *reinterpret_cast<unsigned*>(&q_s[row * QStride + d]) = value;
    }
    const bool query_beyond = __syncthreads_or((query_bits & sm70_attention::kBeyondBits) != 0u);

    const sm70_attention::Bf16Rows<Geometry, CacheInput> rows{
        cache_k,      cache_v,   input,       physical_pages_s, block_table,  first_page,
        staged_pages, kv_head,   kv_heads,    first_pos,        valid_tokens, q,
        tokens,       row_count, query_beyond};
    sm70_attention::attend<Geometry, TokenTile>(
        rows, q_s, p_s, stat_m, stat_l, combined, tokens, row_count, kv_head, split, split_start,
        load_end, first_tile, key_blocks, pos, scale, partial_acc, partial_m, partial_l, partition,
        physical_pages_s, first_page);
}

} // namespace ninfer::ops
