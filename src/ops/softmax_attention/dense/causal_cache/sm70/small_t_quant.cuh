#pragma once

// ninfer::ops - split-KV causal small-T attention over the FP8 and NVFP4 caches on sm_70 (fp8,
// nvfp4, k8v4), decoded to FP16 with their scales as they load into Volta's products
// (sm70/small_t_common.cuh).

#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/sm70/small_t_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops {

namespace sm70_attention {

// K and V of the FP8 (row-scaled E4M3) and NVFP4 (G16 E2M1 over E4M3 scales) caches: fp8, nvfp4,
// and k8v4's FP8 keys with NVFP4 values. Keys are stored rotated in both codings, NVFP4 values too
// (their reducer rotates the output back). Chunked as Int8Rows: 32 dimensions a load.
template <typename Geometry, bool KeyFp8, bool ValueFp8, int KeyChunksInFlight = 8>
struct QuantRows {
    static constexpr bool kChunks = true;
    static constexpr int kKeyChunksInFlight = KeyChunksInFlight;
    const std::uint8_t* cache_k;
    const std::uint8_t* cache_v;
    const void* scale_k;
    const void* scale_v;
    const std::int32_t* pages;
    const std::int32_t* block_table;
    int first_page;
    bool staged_pages;
    int kv_head;
    int kv_heads; // the call's KV heads

    struct KeyRow {
        const std::uint8_t* codes;
        uint4 scales; // FP8: the row's FP16 scale in the low half of x; NVFP4: its 16 E4M3 bytes
    };

    // The stored bytes of 32 dimensions: 32 FP8 codes or 16 bytes of E2M1 pairs (`hi` FP8 only).
    struct KeyChunk {
        uint4 lo;
        uint4 hi;
    };

    // A lane's 32 value dimensions [dim0, dim0 + 32) of one key and the row's scales.
    struct ValueRow {
        uint4 lo;
        uint4 hi;
        uint4 scales;
        int group0; // NVFP4: the G16 group of dim0
    };

    __device__ __forceinline__ int page_of(int k) const {
        return staged_pages ? pages[(k >> kPagedKVPageShift) - first_page]
                            : block_table[k >> kPagedKVPageShift];
    }

    __device__ __forceinline__ static unsigned bits(__half2 v) {
        return *reinterpret_cast<const unsigned*>(&v);
    }

    __device__ __forceinline__ static unsigned fp8x2(unsigned codes, __half2 s) {
        return bits(__hmul2(kv_cache_fp8_code2_to_half2_x2m8(codes), s));
    }

    __device__ __forceinline__ static unsigned e2m1x2(unsigned byte, __half2 s) {
        return bits(__hmul2(kv_cache_e2m1x2_to_half2(byte), s));
    }

    // The row's FP16 scale times 2^8 for fp8x2 (exact, and at most 65504 / 448 * 256).
    __device__ __forceinline__ static __half2 fp16_scale(const uint4& scales) {
        const __half s = __ushort_as_half(static_cast<unsigned short>(scales.x & 0xffffu));
        return __half2half2(__hmul(s, __float2half_rn(256.0f)));
    }

    __device__ __forceinline__ static __half2 e4m3_scale(const uint4& scales, int group) {
        const unsigned word =
            group < 8 ? (group < 4 ? scales.x : scales.y) : (group < 12 ? scales.z : scales.w);
        const unsigned code = (word >> (8 * (group & 3))) & 0xffu;
        // Every E4M3 scale is an exact FP16, and the bit-level decode times 2^8 restores it.
        return __hmul2(kv_cache_fp8_code2_to_half2_x2m8(code | (code << 8)),
                       __float2half2_rn(256.0f));
    }

    template <bool Fp8>
    __device__ __forceinline__ static uint4 load_scales(const void* plane, int page, int kv_head,
                                                        int offset, int kv_heads) {
        if constexpr (Fp8) {
            const __half s = static_cast<const __half*>(
                plane)[kv_cache_fp8_scale_index(page, kv_head, offset, kv_heads)];
            return make_uint4(__half_as_ushort(s), 0, 0, 0);
        } else {
            return *reinterpret_cast<const uint4*>(
                static_cast<const std::uint8_t*>(plane) +
                kv_cache_nvfp4_scale_index(page, kv_head, 0, offset, kv_heads));
        }
    }

    __device__ __forceinline__ KeyRow key(int k) const {
        const int page        = page_of(k);
        const int offset      = k & kPagedKVPageMask;
        const std::int64_t at = KeyFp8
                                    ? kv_cache_fp8_code_index(page, kv_head, 0, offset, kv_heads)
                                    : kv_cache_nvfp4_code_index(page, kv_head, 0, offset, kv_heads);
        return {cache_k + at, load_scales<KeyFp8>(scale_k, page, kv_head, offset, kv_heads)};
    }

    // The stored bytes of dimensions [d0, d0 + 32), d0 a multiple of 32.
    __device__ __forceinline__ KeyChunk key_chunk(const KeyRow& r, int d0) const {
        KeyChunk c{};
        if constexpr (KeyFp8) {
            const auto* p = reinterpret_cast<const uint4*>(r.codes + d0);
            c.lo          = p[0];
            c.hi          = p[1];
        } else {
            c.lo = *reinterpret_cast<const uint4*>(r.codes + d0 / 2);
        }
        return c;
    }

    // Dimensions d0 + 8j..+7 of a chunk in FP16.
    __device__ __forceinline__ uint4 key8(const KeyRow& r, const KeyChunk& c, int d0, int j) const {
        if constexpr (KeyFp8) {
            const uint4& half = j < 2 ? c.lo : c.hi;
            const uint2 w = (j & 1) ? make_uint2(half.z, half.w) : make_uint2(half.x, half.y);
            const __half2 s = fp16_scale(r.scales);
            return make_uint4(fp8x2(w.x, s), fp8x2(w.x >> 16, s), fp8x2(w.y, s),
                              fp8x2(w.y >> 16, s));
        } else {
            const unsigned w = j == 0 ? c.lo.x : j == 1 ? c.lo.y : j == 2 ? c.lo.z : c.lo.w;
            const __half2 s  = e4m3_scale(r.scales, (d0 + 8 * j) / kKVCacheNvfp4Group);
            return make_uint4(e2m1x2(w, s), e2m1x2(w >> 8, s), e2m1x2(w >> 16, s),
                              e2m1x2(w >> 24, s));
        }
    }

    // Dimensions [dim0, dim0 + 32) of key k, dim0 a multiple of 32.
    __device__ __forceinline__ ValueRow value(int k, int dim0) const {
        const int page   = page_of(k);
        const int offset = k & kPagedKVPageMask;
        ValueRow out{};
        if constexpr (ValueFp8) {
            const auto* p = reinterpret_cast<const uint4*>(
                cache_v + kv_cache_fp8_code_index(page, kv_head, dim0, offset, kv_heads));
            out.lo = p[0];
            out.hi = p[1];
        } else {
            out.lo = *reinterpret_cast<const uint4*>(
                cache_v + kv_cache_nvfp4_code_index(page, kv_head, dim0, offset, kv_heads));
        }
        out.scales = load_scales<ValueFp8>(scale_v, page, kv_head, offset, kv_heads);
        out.group0 = dim0 / kKVCacheNvfp4Group;
        return out;
    }

    // Dimensions dim0 + 4n..+3 of a value row: NVFP4 group group0 + n / 4.
    __device__ __forceinline__ uint2 value4(const ValueRow& r, int n) const {
        if constexpr (ValueFp8) {
            const uint4& half = n < 4 ? r.lo : r.hi;
            const int m       = n & 3;
            const unsigned c  = m == 0 ? half.x : m == 1 ? half.y : m == 2 ? half.z : half.w;
            const __half2 s   = fp16_scale(r.scales);
            return make_uint2(fp8x2(c, s), fp8x2(c >> 16, s));
        } else {
            const unsigned word = n < 2 ? r.lo.x : n < 4 ? r.lo.y : n < 6 ? r.lo.z : r.lo.w;
            const unsigned c    = (n & 1) ? word >> 16 : word & 0xffffu;
            const __half2 s     = e4m3_scale(r.scales, r.group0 + n / 4);
            return make_uint2(e2m1x2(c, s), e2m1x2(c >> 8, s));
        }
    }
};

// NVFP4's scalar row access for the one/two-tile H16 verification splits. Keep the same
// represented FP16 operands, but read each MMA fragment separately and use its logical output
// dimensions. The chunked provider still owns the common pages and stored-scale decoding.
template <typename Geometry>
struct Nvfp4ScalarRows {
    static constexpr bool kChunks = false;
    using PackedRows = QuantRows<Geometry, false, false>;
    using KeyRow = typename PackedRows::KeyRow;
    PackedRows packed;

    struct ValueRow {
        const std::uint8_t* codes;
        uint4 scales;
        int group0;
    };

    __device__ __forceinline__ KeyRow key(int k) const { return packed.key(k); }

    __device__ __forceinline__ uint4 key8(const KeyRow& row, int d) const {
        const unsigned codes = *reinterpret_cast<const unsigned*>(row.codes + d / 2);
        const __half2 scale = PackedRows::e4m3_scale(row.scales, d / kKVCacheNvfp4Group);
        return make_uint4(PackedRows::e2m1x2(codes, scale),
                          PackedRows::e2m1x2(codes >> 8, scale),
                          PackedRows::e2m1x2(codes >> 16, scale),
                          PackedRows::e2m1x2(codes >> 24, scale));
    }

    __device__ __forceinline__ ValueRow value(int k, int dim0) const {
        const int page = packed.page_of(k);
        const int offset = k & kPagedKVPageMask;
        const auto at =
            kv_cache_nvfp4_code_index(page, packed.kv_head, dim0, offset, packed.kv_heads);
        return {packed.cache_v + at,
                PackedRows::template load_scales<false>(packed.scale_v, page, packed.kv_head,
                                                        offset, packed.kv_heads),
                dim0 / kKVCacheNvfp4Group};
    }

    __device__ __forceinline__ uint2 value4(const ValueRow& row, int n) const {
        const unsigned codes = *reinterpret_cast<const std::uint16_t*>(row.codes + 16 * n);
        const __half2 scale = PackedRows::e4m3_scale(row.scales, row.group0 + 2 * n);
        return make_uint2(PackedRows::e2m1x2(codes, scale),
                          PackedRows::e2m1x2(codes >> 8, scale));
    }
};

} // namespace sm70_attention

// The FP8 and NVFP4 caches (fp8: FP8 keys and values; nvfp4: NVFP4 keys and values; k8v4: FP8
// keys, NVFP4 values), launched by fp8/small_t.cpp, nvfp4/small_t.cpp and k8v4/small_t.cpp in
// place of their tiled kernels (sm70/small_t_quant_launch.cuh): the same grid, quantized active
// splits and split ranges, the fused append of those kernels (one warp per new row, the standalone
// append's operations and rounding order) and partial outputs, NVFP4 values left rotated for their
// reducers.
template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput,
          bool KeyFp8, bool ValueFp8>
__launch_bounds__(32 * sm70_attention::Shape<TokenTile * Geometry::GroupSize>::kWarps) __global__
    void causal_attention_small_t_sm70_quantized_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* positions,
        std::uint8_t* cache_k, std::uint8_t* cache_v,
        std::conditional_t<KeyFp8, __half, std::uint8_t>* cache_k_scale,
        std::conditional_t<ValueFp8, __half, std::uint8_t>* cache_v_scale,
        const std::int32_t* block_tables, const std::int32_t* valid_columns,
        const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
        std::int32_t column_begin, std::int32_t logical_capacity, float attention_scale,
        float* partial_acc, float* partial_m, float* partial_l, CausalKeyPartition partition) {
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
    constexpr int PageIds       = 64;
    constexpr bool Nvfp4Append  = !KeyFp8 || !ValueFp8;
    constexpr unsigned FullMask = 0xffffffffu;

    __shared__ __align__(16) __half q_s[Groups * 8 * QStride];
    __shared__ __align__(16) __half p_s[Warps * 8 * PStride];
    __shared__ std::int32_t physical_pages_s[PageIds];
    __shared__ float stat_m[Warps][8];
    __shared__ float stat_l[Warps][8];
    __shared__ float combined[KeyWarps > 1 ? Groups * 8 * D : 1];
    __shared__ float quant_scratch[CacheInput::writes_cache && Nvfp4Append ? Warps * D : 1];

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
    positions += column_base;
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

    const std::int32_t first_pos = positions[0];
    const std::int32_t last_pos  = positions[TokenTile - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int live_end = last_pos + 1;
    const int window   = causal_small_t_tile_window(first_pos, TokenTile, logical_capacity);
    const int active_split_count = causal_small_t_partition_active(
        partition, window,
        causal_small_t_quantized_active_splits<Geometry>(window, split_count, TokenTile),
        split_count);
    if (split >= active_split_count) { return; }

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    int split_start         = 0;
    int split_end           = 0;
    if (partition.segmented() || partition.strided()) {
        causal_small_t_partition_range(partition, window, split, active_split_count, split_start,
                                       split_end);
    } else if constexpr (TokenTile == 1) {
        const int first_owned_tile = split * logical_tiles / active_split_count;
        const int end_owned_tile   = (split + 1) * logical_tiles / active_split_count;
        split_start                = first_owned_tile * Bc;
        split_end                  = min(end_owned_tile * Bc, window);
    } else {
        const int units_per_split = tile_split ? div_up(logical_tiles, active_split_count)
                                               : div_up(window, active_split_count);
        split_start               = split * units_per_split * (tile_split ? Bc : 1);
        split_end = min(split_start + units_per_split * (tile_split ? Bc : 1), window);
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
        // One warp owns the complete D256 K row and then the complete V row, with the standalone
        // append's operations and rounding order (FP8: the represented FP16 scale feeds back
        // before the codes; NVFP4: G16 groups of the rotated row).
        for (int token = warp; token < valid_tokens; token += Warps) {
            const int position = positions[token];
            if (position < split_start || position >= split_end) { continue; }
            const int physical_page = page_at(position);
            if (physical_page < 0) { continue; } // another rank's planes hold this row
            const int page_offset   = position & kPagedKVPageMask;
            float* row_s            = Nvfp4Append ? quant_scratch + warp * D : nullptr;
            const auto store_fp8    = [&](std::uint8_t* plane, __half* scales,
                                          const float (&values)[8]) {
                float local_absmax = 0.0f;
#pragma unroll
                for (float value : values) { local_absmax = fmaxf(local_absmax, fabsf(value)); }
                const auto quant = kv_cache_fp8_quant_params(warp_max(local_absmax, FullMask));
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    plane[kv_cache_fp8_code_index(physical_page, kv_head, lane + 32 * r,
                                                  page_offset, kv_heads)] =
                        kv_cache_fp8_quant_code(values[r], quant.inverse_scale);
                }
                if (lane == 0) {
                    scales[kv_cache_fp8_scale_index(physical_page, kv_head, page_offset,
                                                    kv_heads)] = quant.scale;
                }
            };
            const auto store_nvfp4 = [&](std::uint8_t* plane, std::uint8_t* scales,
                                         const float (&values)[8]) {
#pragma unroll
                for (int r = 0; r < 8; ++r) { row_s[lane + 32 * r] = values[r]; }
                __syncwarp();
                if (lane < kKVCacheNvfp4Groups) {
                    const auto quantized =
                        kv_cache_nvfp4_quantize_group16(row_s + lane * kKVCacheNvfp4Group);
                    const std::int64_t code_offset = kv_cache_nvfp4_code_index(
                        physical_page, kv_head, lane * kKVCacheNvfp4Group, page_offset, kv_heads);
                    store_vec(plane + code_offset,
                              make_uint2(quantized.codes_lo, quantized.codes_hi));
                    scales[kv_cache_nvfp4_scale_index(physical_page, kv_head, lane, page_offset,
                                                      kv_heads)] = quantized.scale;
                }
                __syncwarp();
            };
            float values[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                values[r] = __bfloat162float(
                    input.k[kv_cache_fp8_src_index(kv_head, lane + 32 * r, token, kv_heads)]);
            }
            normalized_hadamard_d256_inplace(values, lane);
            if constexpr (KeyFp8) {
                store_fp8(cache_k, cache_k_scale, values);
            } else {
                store_nvfp4(cache_k, cache_k_scale, values);
            }
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                values[r] = __bfloat162float(
                    input.v[kv_cache_fp8_src_index(kv_head, lane + 32 * r, token, kv_heads)]);
            }
            if constexpr (ValueFp8) {
                store_fp8(cache_v, cache_v_scale, values);
            } else {
                normalized_hadamard_d256_inplace(values, lane);
                store_nvfp4(cache_v, cache_v_scale, values);
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

    const sm70_attention::QuantRows<Geometry, KeyFp8, ValueFp8> rows{
        cache_k,     cache_v,    cache_k_scale, cache_v_scale, physical_pages_s,
        block_table, first_page, staged_pages,  kv_head,       kv_heads};
    // A segmented partition keeps every width on the generic rows, whose arithmetic it fixes.
    if constexpr (!MultiBatch && !KeyFp8 && !ValueFp8 && TokenTile == 4) {
      if (!partition.segmented()) {
        // These choices are measured over the complete Op on V100. Use the actual split work,
        // not the host envelope: one captured graph serves every position inside that envelope.
        if constexpr (Geometry::GroupSize == 8) {
            if (key_blocks <= 2) {
                const sm70_attention::Nvfp4ScalarRows<Geometry> scalar{rows};
                sm70_attention::attend<Geometry, TokenTile>(
                    scalar, q_s, p_s, stat_m, stat_l, combined, TokenTile, RowCount, kv_head, split,
                    split_start, load_end, first_tile, key_blocks, positions, attention_scale,
                    partial_acc, partial_m, partial_l, partition, physical_pages_s, first_page);
                return;
            }
        }
        // Fused append benefits from the shorter QK unroll. Cached-only H16 splits at 9K/12K/
        // 24K keep eight chunks in flight; above 256 active CTAs, two chunks win again.
        if (CacheInput::writes_cache || kv_heads * active_split_count > 256) {
            const sm70_attention::QuantRows<Geometry, false, false, 2> limited{
                cache_k,     cache_v,    cache_k_scale, cache_v_scale, physical_pages_s,
                block_table, first_page, staged_pages,  kv_head,       kv_heads};
            sm70_attention::attend<Geometry, TokenTile>(
                limited, q_s, p_s, stat_m, stat_l, combined, TokenTile, RowCount, kv_head, split,
                split_start, load_end, first_tile, key_blocks, positions, attention_scale,
                partial_acc, partial_m, partial_l, partition, physical_pages_s, first_page);
            return;
        }
      }
    }
    sm70_attention::attend<Geometry, TokenTile>(rows, q_s, p_s, stat_m, stat_l, combined, TokenTile,
                                                RowCount, kv_head, split, split_start, load_end,
                                                first_tile, key_blocks, positions, attention_scale,
                                                partial_acc, partial_m, partial_l, partition,
                                                physical_pages_s, first_page);
}

} // namespace ninfer::ops
