#pragma once

// ninfer::ops - split-KV causal small-T attention over the INT8-family caches on sm_70 (int8,
// rk8v4, rk4v4, rk4v4-e8, rk2v4-e8): the fused append rotates and quantizes the new rows exactly as
// int8/small_t.cuh does, Q is rotated by the same normalized D256 Hadamard transform and kept in
// FP16, and the key codings and value codes are decoded to FP16 with their group scales as they
// load into Volta's products (sm70/small_t_common.cuh).

#include "ops/kv_cache/e8_root_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/sm70/small_t_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

namespace sm70_attention {

// Eight int8 codes (dimension order) of eight key dimensions from their stored bytes in `w`: eight
// INT8 codes, a packed 4-bit word (the Lloyd-Max levels or the E8-snapped signed int4 codes) in
// w.x, or one E8 root block in the low half of w.x.
template <KvKeyCoding Keys>
__device__ __forceinline__ uint2 key_codes8(uint2 w) {
    if constexpr (Keys == KvKeyCoding::Int8) {
        return w;
    } else if constexpr (Keys == KvKeyCoding::RootE8) {
        return kv_cache_e8_root_decode_block(w.x & 0xffu, (w.x >> 8) & 0xffu);
    } else {
        const unsigned word = w.x;
        if constexpr (Keys == KvKeyCoding::Lloyd4) {
            return make_uint2(kv_cache_lloyd4_expand4(word), kv_cache_lloyd4_expand4(word >> 16));
        } else {
            const unsigned low  = __vsub4((word & 0x0f0f0f0fu) ^ 0x08080808u, 0x08080808u);
            const unsigned high = __vsub4(((word >> 4) & 0x0f0f0f0fu) ^ 0x08080808u, 0x08080808u);
            return make_uint2(__byte_perm(low, high, 0x5140), __byte_perm(low, high, 0x7362));
        }
    }
}

// K and V of an INT8-family cache (rotated keys): INT8 G64 codes or a packed key coding over the
// G64 key scales, INT8 G64 values or packed signed int4 values over G32 scales. A lane reads 32
// dimensions of a row at a time in one or two 16-byte loads (kChunks): attend_tile's key chunks,
// and the 32 consecutive value dimensions of its value products.
template <typename Geometry, KvKeyCoding Keys, bool PackedValues>
struct Int8Rows {
    static constexpr bool kChunks = true;
    static constexpr int kKeyChunksInFlight = 8;
    const std::uint8_t* cache_k;
    const std::uint8_t* cache_v;
    const __half* scale_k;
    const __half* scale_v;
    const std::int32_t* pages;
    const std::int32_t* block_table;
    int first_page;
    bool staged_pages;
    int kv_head;
    int kv_heads; // the call's KV heads

    struct KeyRow {
        const std::uint8_t* codes;
        uint2 scales; // the row's four G64 scales
    };

    // The stored bytes of 32 key dimensions: 32 INT8 codes, 16 bytes of a 4-bit coding, or four
    // E8 root blocks (8 bytes); `hi` holds INT8 keys' second 16 bytes only.
    struct KeyChunk {
        uint4 lo;
        uint4 hi;
    };

    // A lane's 32 value dimensions [dim0, dim0 + 32) of one key: 32 INT8 codes or 16 bytes of
    // packed int4 codes, and their one scale (G64 or G32).
    struct ValueRow {
        uint4 lo;
        uint4 hi;
        __half2 scale;
    };

    __device__ __forceinline__ int page_of(int k) const {
        return staged_pages ? pages[(k >> kPagedKVPageShift) - first_page]
                            : block_table[k >> kPagedKVPageShift];
    }

    __device__ __forceinline__ static __half2 half_of(unsigned word, int upper) {
        return __half2half2(
            __ushort_as_half(static_cast<unsigned short>(upper ? (word >> 16) : (word & 0xffffu))));
    }

    __device__ __forceinline__ KeyRow key(int k) const {
        const int page   = page_of(k);
        const int offset = k & kPagedKVPageMask;
        std::int64_t at  = 0;
        if constexpr (Keys == KvKeyCoding::Int8) {
            at = kv_cache_int8_quant_code_index(page, kv_head, 0, offset, kv_heads);
        } else {
            at = kv_cache_packed_key_chunk_index<Keys>(page, kv_head, 0, offset, kv_heads);
        }
        return {cache_k + at,
                *reinterpret_cast<const uint2*>(
                    scale_k + kv_cache_int8_quant_scale_index(page, kv_head, 0, offset, kv_heads))};
    }

    // The stored bytes of dimensions [d0, d0 + 32), d0 a multiple of 32.
    __device__ __forceinline__ KeyChunk key_chunk(const KeyRow& r, int d0) const {
        KeyChunk c{};
        if constexpr (Keys == KvKeyCoding::Int8) {
            const auto* p = reinterpret_cast<const uint4*>(r.codes + d0);
            c.lo          = p[0];
            c.hi          = p[1];
        } else if constexpr (Keys == KvKeyCoding::RootE8) {
            const uint2 v = *reinterpret_cast<const uint2*>(r.codes + d0 / 4);
            c.lo          = make_uint4(v.x, v.y, 0, 0);
        } else {
            c.lo = *reinterpret_cast<const uint4*>(r.codes + d0 / 2);
        }
        return c;
    }

    // Dimensions d0 + 8j..+7 of a chunk in FP16.
    __device__ __forceinline__ uint4 key8(const KeyRow& r, const KeyChunk& c, int d0, int j) const {
        uint2 w;
        if constexpr (Keys == KvKeyCoding::Int8) {
            const uint4& half = j < 2 ? c.lo : c.hi;
            w                 = (j & 1) ? make_uint2(half.z, half.w) : make_uint2(half.x, half.y);
        } else if constexpr (Keys == KvKeyCoding::RootE8) {
            const unsigned word = j < 2 ? c.lo.x : c.lo.y;
            w                   = make_uint2((j & 1) ? word >> 16 : word & 0xffffu, 0u);
        } else {
            w = make_uint2(j == 0 ? c.lo.x : j == 1 ? c.lo.y : j == 2 ? c.lo.z : c.lo.w, 0u);
        }
        const int group = (d0 + 8 * j) / kKVCacheInt8Group;
        const __half2 s = half_of(group < 2 ? r.scales.x : r.scales.y, group & 1);
        const uint2 codes = key_codes8<Keys>(w);
        uint4 out;
        int8x4_to_f16x2(codes.x, s, out.x, out.y);
        int8x4_to_f16x2(codes.y, s, out.z, out.w);
        return out;
    }

    // Dimensions [dim0, dim0 + 32) of key k, dim0 a multiple of 32.
    __device__ __forceinline__ ValueRow value(int k, int dim0) const {
        const int page   = page_of(k);
        const int offset = k & kPagedKVPageMask;
        ValueRow out{};
        if constexpr (PackedValues) {
            out.lo = *reinterpret_cast<const uint4*>(
                cache_v +
                kv_cache_int4_value_code_index(page, kv_head, dim0 / 2, offset, kv_heads));
            out.scale = __half2half2(scale_v[kv_cache_int4_value_scale_index(
                page, kv_head, dim0 / 32, offset, kv_heads)]);
        } else {
            const auto* p = reinterpret_cast<const uint4*>(
                cache_v + kv_cache_int8_quant_code_index(page, kv_head, dim0, offset, kv_heads));
            out.lo    = p[0];
            out.hi    = p[1];
            out.scale = __half2half2(scale_v[kv_cache_int8_quant_scale_index(
                page, kv_head, dim0 / kKVCacheInt8Group, offset, kv_heads)]);
        }
        return out;
    }

    // Dimensions dim0 + 4n..+3 of a value row.
    __device__ __forceinline__ uint2 value4(const ValueRow& r, int n) const {
        unsigned codes = 0;
        if constexpr (PackedValues) {
            const unsigned word = n < 2 ? r.lo.x : n < 4 ? r.lo.y : n < 6 ? r.lo.z : r.lo.w;
            const unsigned pair = (n & 1) ? word >> 16 : word & 0xffffu;
            const unsigned low  = __vsub4((pair & 0x0f0fu) ^ 0x0808u, 0x0808u);
            const unsigned high = __vsub4(((pair >> 4) & 0x0f0fu) ^ 0x0808u, 0x0808u);
            codes               = __byte_perm(low, high, 0x5140);
        } else {
            const uint4& half = n < 4 ? r.lo : r.hi;
            const int m       = n & 3;
            codes = m == 0 ? half.x : m == 1 ? half.y : m == 2 ? half.z : half.w;
        }
        uint2 out;
        int8x4_to_f16x2(codes, r.scale, out.x, out.y);
        return out;
    }
};

} // namespace sm70_attention

// The INT8-family caches (rotated keys in any key coding, INT8 or packed int4 values), launched by
// int8/small_t_sm70_launch.cuh in place of the tiled kernel: the same grid, active splits (wave_splits
// included), fused append and partial outputs, with this file's products.
template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput,
          bool PackedValues, KvKeyCoding Keys>
__launch_bounds__(32 * sm70_attention::Shape<TokenTile * Geometry::GroupSize>::kWarps,
                  sm70_attention::i8_resident_ctas<TokenTile, TokenTile * Geometry::GroupSize>())
    __global__ void causal_attention_small_t_sm70_i8_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, std::int8_t* cache_k_i8,
        std::int8_t* cache_v_i8, __half* cache_k_scale, __half* cache_v_scale,
        const std::int32_t* block_tables, const std::int32_t* valid_columns,
        const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
        std::int32_t column_begin, std::int32_t logical_capacity, std::int32_t wave_splits,
        float scale, float* partial_acc, float* partial_m, float* partial_l,
        CausalKeyPartition partition) {
    using Shape                 = sm70_attention::Shape<TokenTile * Geometry::GroupSize>;
    constexpr int D             = kCausalHeadDim;
    constexpr int Bc            = 32;
    constexpr int Groups        = Shape::kRowGroups;
    constexpr int KeyWarps      = Shape::kKeyWarps;
    constexpr int Warps         = Shape::kWarps;
    constexpr int Threads       = 32 * Warps;
    constexpr int QStride       = sm70_attention::kQStride;
    constexpr int PStride       = sm70_attention::kPStride;
    constexpr int RowCount      = TokenTile * Geometry::GroupSize;
    constexpr int KeyGroups     = kKVCacheInt8Groups;
    constexpr int PageIds       = 64;
    constexpr unsigned FullMask = 0xffffffffu;

    __shared__ __align__(16) __half q_s[Groups * 8 * QStride];
    __shared__ __align__(16) __half p_s[Warps * 8 * PStride];
    __shared__ std::int32_t physical_pages_s[PageIds];
    __shared__ float stat_m[Warps][8];
    __shared__ float stat_l[Warps][8];
    __shared__ float combined[KeyWarps > 1 ? Groups * 8 * D : 1];
    __shared__ float k_h64_s[CacheInput::writes_cache ? TokenTile * D : 1];

    const int kv_head     = static_cast<int>(blockIdx.x);
    // The grid spans the call's KV heads.
    const int kv_heads    = static_cast<int>(gridDim.x);
    const int q_heads     = kv_heads * Geometry::GroupSize;
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    int valid_tokens      = TokenTile;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < TokenTile ? remaining : TokenTile);
    }

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
        partial_acc += static_cast<std::int64_t>(batch) * D * q_heads * TokenTile * split_count;
        partial_m += static_cast<std::int64_t>(batch) * q_heads * TokenTile * split_count;
        partial_l += static_cast<std::int64_t>(batch) * q_heads * TokenTile * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < RowCount; row += Threads) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head, kv_heads)) {
                partial_m[causal_partial_stat_index(q_head, token, split, TokenTile, q_heads)] =
                    -CUDART_INF_F;
                partial_l[causal_partial_stat_index(q_head, token, split, TokenTile, q_heads)] =
                    0.0f;
            }
        }
        for (int idx = tid; idx < RowCount * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head, kv_heads)) {
                partial_acc[causal_partial_acc_index(q_head, d, token, split, TokenTile, q_heads)] =
                    0.0f;
            }
        }
    };

    if (kv_head < 0 || kv_head >= kv_heads || split_count <= 0) { return; }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[TokenTile - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int live_end = last_pos + 1;
    const int window   = causal_small_t_tile_window(first_pos, TokenTile, logical_capacity);
    const int active_split_count = causal_small_t_partition_active(
        partition, window,
        causal_small_t_active_splits<Geometry, true>(window, split_count, TokenTile, wave_splits),
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
    const auto page_at = [&](int key) {
        return staged_pages ? physical_pages_s[(key >> kPagedKVPageShift) - first_page]
                            : block_table[key >> kPagedKVPageShift];
    };

    if constexpr (CacheInput::writes_cache) {
        // The fused append of int8/small_t.cuh: the owning split rotates (H64 fragments, then the
        // H4 leaf) and quantizes each new key row and quantizes its value row; every key, the new
        // ones included, is then read back from the cache.
        for (int pair = warp; pair < valid_tokens * KeyGroups; pair += Warps) {
            const int token    = pair / KeyGroups;
            const int grp      = pair - token * KeyGroups;
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            const int d0            = grp * kKVCacheInt8Group + lane;
            const int d1            = d0 + 32;
            const std::int64_t src0 = kv_cache_int8_new_index(kv_head, d0, token, kv_heads);
            const std::int64_t src1 = kv_cache_int8_new_index(kv_head, d1, token, kv_heads);
            float k_h64[2] = {__bfloat162float(input.k[src0]), __bfloat162float(input.k[src1])};
            hadamard_d64_fragment_inplace(k_h64, lane);
            k_h64_s[token * D + d0] = k_h64[0];
            k_h64_s[token * D + d1] = k_h64[1];
        }
        __syncthreads();

        for (int pair = warp; pair < valid_tokens * KeyGroups; pair += Warps) {
            const int token    = pair / KeyGroups;
            const int grp      = pair - token * KeyGroups;
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            const int physical_page = page_at(position);
            if (physical_page < 0) { continue; } // another rank's planes hold this row
            const int page_offset   = position & kPagedKVPageMask;
            const int d0            = grp * kKVCacheInt8Group + lane;
            const int d1            = d0 + 32;
            const std::int64_t src0 = kv_cache_int8_new_index(kv_head, d0, token, kv_heads);
            const std::int64_t src1 = kv_cache_int8_new_index(kv_head, d1, token, kv_heads);
            float k_out[2];
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int dh   = lane + half * 32;
                const float x0 = k_h64_s[token * D + dh];
                const float x1 = k_h64_s[token * D + kKVCacheInt8Group + dh];
                const float x2 = k_h64_s[token * D + 2 * kKVCacheInt8Group + dh];
                const float x3 = k_h64_s[token * D + 3 * kKVCacheInt8Group + dh];
                k_out[half]    = normalized_hadamard_d256_group_value_from_h64(x0, x1, x2, x3, grp);
            }
            kv_cache_i8_family_store_key_group<Keys>(cache_k_i8, cache_k_scale, physical_page,
                                                     kv_head, grp, page_offset, lane, k_out[0],
                                                     k_out[1], kv_heads);
            // The vv0 lanes span dimensions [64g, 64g+32) and the vv1 lanes the next 32, so the
            // packed coding's two G32 groups fall out of the lane assignment directly.
            const float vv0       = __bfloat162float(input.v[src0]);
            const float vv1       = __bfloat162float(input.v[src1]);
            const float vamax_lo  = warp_max(fabsf(vv0), FullMask);
            const float vamax_hi  = warp_max(fabsf(vv1), FullMask);
            const auto v_quant    = PackedValues
                                        ? kv_cache_int4_quant_params(vamax_lo)
                                        : kv_cache_int8_quant_params(fmaxf(vamax_lo, vamax_hi));
            const auto v_quant_hi = PackedValues ? kv_cache_int4_quant_params(vamax_hi) : v_quant;
            if constexpr (PackedValues) {
                // A packed byte holds the adjacent dimension pair, which spans lanes l and l^1.
                const std::int8_t c0 = kv_cache_int4_quant_code(vv0, v_quant.inverse_scale);
                const std::int8_t c1 = kv_cache_int4_quant_code(vv1, v_quant_hi.inverse_scale);
                const int partner0   = __shfl_xor_sync(FullMask, static_cast<int>(c0), 1);
                const int partner1   = __shfl_xor_sync(FullMask, static_cast<int>(c1), 1);
                if ((lane & 1) == 0) {
                    auto* packed_v                 = reinterpret_cast<std::uint8_t*>(cache_v_i8);
                    const std::int64_t packed_base = kv_cache_int4_value_code_index(
                        physical_page, kv_head, grp * (kKVCacheInt8Group / 2), page_offset,
                        kv_heads);
                    packed_v[packed_base + (lane >> 1)] =
                        kv_cache_int4_pack(c0, static_cast<std::int8_t>(partner0));
                    packed_v[packed_base + (lane >> 1) + 16] =
                        kv_cache_int4_pack(c1, static_cast<std::int8_t>(partner1));
                }
            } else {
                cache_v_i8[kv_cache_int8_quant_code_index(physical_page, kv_head, d0, page_offset,
                                                          kv_heads)] =
                    kv_cache_int8_quant_code(vv0, v_quant.inverse_scale);
                cache_v_i8[kv_cache_int8_quant_code_index(physical_page, kv_head, d1, page_offset,
                                                          kv_heads)] =
                    kv_cache_int8_quant_code(vv1, v_quant.inverse_scale);
            }
            if (lane == 0) {
                if constexpr (PackedValues) {
                    cache_v_scale[kv_cache_int4_value_scale_index(
                        physical_page, kv_head, 2 * grp, page_offset, kv_heads)] = v_quant.scale;
                    cache_v_scale[kv_cache_int4_value_scale_index(
                        physical_page, kv_head, 2 * grp + 1, page_offset, kv_heads)] =
                        v_quant_hi.scale;
                } else {
                    cache_v_scale[kv_cache_int8_quant_scale_index(
                        physical_page, kv_head, grp, page_offset, kv_heads)] = v_quant.scale;
                }
            }
        }
        __syncthreads();
    }

    // Q rotated by the keys' normalized D256 transform, in FP16 (zero past RowCount).
    for (int row = warp; row < Groups * 8; row += Warps) {
        float values[8];
        int q_head = 0;
        int token  = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
        const bool live =
            row < RowCount && causal_valid_q_head<Geometry>(kv_head, q_head, kv_heads);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            values[r] =
                live ? __bfloat162float(q[causal_q_index(q_head, lane + 32 * r, token, q_heads)])
                     : 0.0f;
        }
        normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            q_s[row * QStride + lane + 32 * r] = __float2half_rn(live ? values[r] : 0.0f);
        }
    }
    __syncthreads();

    const sm70_attention::Int8Rows<Geometry, Keys, PackedValues> rows{
        reinterpret_cast<const std::uint8_t*>(cache_k_i8),
        reinterpret_cast<const std::uint8_t*>(cache_v_i8),
        cache_k_scale,
        cache_v_scale,
        physical_pages_s,
        block_table,
        first_page,
        staged_pages,
        kv_head,
        kv_heads};
    sm70_attention::attend<Geometry, TokenTile>(
        rows, q_s, p_s, stat_m, stat_l, combined, TokenTile, RowCount, kv_head, split, split_start,
        load_end, first_tile, key_blocks, pos, scale, partial_acc, partial_m, partial_l, partition,
        physical_pages_s, first_page);
}

} // namespace ninfer::ops
