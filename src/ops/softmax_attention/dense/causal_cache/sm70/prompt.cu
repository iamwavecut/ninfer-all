#include "ops/softmax_attention/dense/causal_cache/sm70/prompt.h"

#include "core/device_launch.h"
#include "ops/common/device_tier.h"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/e8_root_codec.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/sm70/small_t_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHeadDim = 256;
constexpr int kTile    = 32;                       // keys per tile
constexpr int kStride  = sm70_attention::kQStride; // halves per staged Q, K or V row
constexpr int kPStride = sm70_attention::kPStride;

// A CTA's warps take eight query rows each. The standard set's eight warps (64 rows) stage 71 KiB,
// past the 64 KiB a Turing block may hold, and their rotated epilogue's FP32 rows alone take 64.25
// KiB; Turing's set runs four warps (32 rows), whose threads load two (key, part) slots of a tile.
template <bool TuringShapes>
struct PromptShape {
    static constexpr int kWarps   = TuringShapes ? 4 : 8;
    static constexpr int kRows    = kWarps * 8; // query rows per CTA
    static constexpr int kThreads = kWarps * 32;
    static constexpr int kLoads   = kTile * 8 / kThreads; // (key, 32-dimension part) slots
    static constexpr int kSharedBytes =
        (kRows * kStride + 2 * kTile * kStride + kWarps * 8 * kPStride) *
        static_cast<int>(sizeof(__half));
    static_assert(kLoads * kThreads == kTile * 8, "every slot of a tile has one loader");
    static_assert(kSharedBytes <= shape_max_shared_bytes_per_block(TuringShapes));
    static_assert((kRows * kHeadDim + kRows) * sizeof(float) <=
                      (kRows + 2 * kTile) * kStride * sizeof(__half),
                  "the rotated epilogue's FP32 rows and log-sum-exps reuse the Q, K and V stages");
};

// How a cache stores its K or V plane. Every plane is [page][head][page offset][row], so a token's
// row (page * kv_heads + head) * page size + offset addresses each of them.
enum class Coding : int { Bf16, Int8, Lloyd4, Int4E8, RootE8, Int4, Fp8, Nvfp4 };

template <Coding C>
__host__ __device__ constexpr int row_bytes() {
    if constexpr (C == Coding::Bf16) {
        return kHeadDim * 2;
    } else if constexpr (C == Coding::Int8 || C == Coding::Fp8) {
        return kHeadDim;
    } else if constexpr (C == Coding::RootE8) {
        return kKVCacheE8RootRowBytes;
    } else {
        return kHeadDim / 2;
    }
}

// One loader thread's 32 dimensions of one key row as stored: up to 64 bytes and the scale bits
// that cover them (an FP16 scale, or two E4M3 bytes for NVFP4).
struct Raw {
    uint4 w[4];
    unsigned scale;
};

// Dimensions [32 part, 32 part + 32) of plane row `row`.
template <Coding C>
__device__ __forceinline__ Raw read_raw(const void* plane, const void* scales, std::int64_t row,
                                        int part) {
    Raw r{};
    const auto* base = static_cast<const std::uint8_t*>(plane) + row * row_bytes<C>();
    if constexpr (C == Coding::Bf16) {
        const auto* p = reinterpret_cast<const uint4*>(base + part * 64);
#pragma unroll
        for (int i = 0; i < 4; ++i) { r.w[i] = p[i]; }
    } else if constexpr (C == Coding::Int8 || C == Coding::Fp8) {
        const auto* p = reinterpret_cast<const uint4*>(base + part * 32);
        r.w[0]        = p[0];
        r.w[1]        = p[1];
    } else if constexpr (C == Coding::RootE8) {
        const uint2 v = *reinterpret_cast<const uint2*>(base + part * 8);
        r.w[0]        = make_uint4(v.x, v.y, 0u, 0u);
    } else {
        r.w[0] = *reinterpret_cast<const uint4*>(base + part * 16);
    }
    const auto* half_scales = static_cast<const __half*>(scales);
    if constexpr (C == Coding::Int8 || C == Coding::Lloyd4 || C == Coding::Int4E8 ||
                  C == Coding::RootE8) {
        r.scale = __half_as_ushort(half_scales[row * kKVCacheInt8Groups + part / 2]);
    } else if constexpr (C == Coding::Int4) {
        r.scale = __half_as_ushort(half_scales[row * kKVCacheInt4ValueGroups + part]);
    } else if constexpr (C == Coding::Fp8) {
        r.scale = __half_as_ushort(half_scales[row]);
    } else if constexpr (C == Coding::Nvfp4) {
        r.scale = *reinterpret_cast<const std::uint16_t*>(static_cast<const std::uint8_t*>(scales) +
                                                          row * kKVCacheNvfp4Groups + 2 * part);
    }
    return r;
}

__device__ __forceinline__ unsigned clamped_f16x2(unsigned bf16x2) {
    const float lo  = fminf(fmaxf(__uint_as_float(bf16x2 << 16), -65504.0f), 65504.0f);
    const float hi  = fminf(fmaxf(__uint_as_float(bf16x2 & 0xffff0000u), -65504.0f), 65504.0f);
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}

__device__ __forceinline__ __half2 fp16_scale(unsigned bits) {
    return __half2half2(__ushort_as_half(static_cast<unsigned short>(bits)));
}

// Eight words of four signed int8 codes times `scale` as sixteen FP16 pairs.
__device__ __forceinline__ void codes_to_f16(const unsigned (&codes)[8], __half2 scale,
                                             unsigned (&out)[16]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sm70_attention::int8x4_to_f16x2(codes[i], scale, out[2 * i], out[2 * i + 1]);
    }
}

// The 32 dimensions of a Raw in FP16 pairs, dimension order.
template <Coding C>
__device__ __forceinline__ void decode_raw(const Raw& r, unsigned (&out)[16]) {
    if constexpr (C == Coding::Bf16) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            out[4 * i + 0] = clamped_f16x2(r.w[i].x);
            out[4 * i + 1] = clamped_f16x2(r.w[i].y);
            out[4 * i + 2] = clamped_f16x2(r.w[i].z);
            out[4 * i + 3] = clamped_f16x2(r.w[i].w);
        }
    } else if constexpr (C == Coding::Int8) {
        const unsigned codes[8] = {r.w[0].x, r.w[0].y, r.w[0].z, r.w[0].w,
                                   r.w[1].x, r.w[1].y, r.w[1].z, r.w[1].w};
        codes_to_f16(codes, fp16_scale(r.scale), out);
    } else if constexpr (C == Coding::Lloyd4 || C == Coding::Int4E8 || C == Coding::Int4) {
        const uint2 first       = make_uint2(r.w[0].x, r.w[0].y);
        const uint2 second      = make_uint2(r.w[0].z, r.w[0].w);
        const int4 lo           = C == Coding::Lloyd4 ? kv_cache_lloyd4_expand16(first)
                                                      : kv_cache_int4_unpack_i8x16(first);
        const int4 hi           = C == Coding::Lloyd4 ? kv_cache_lloyd4_expand16(second)
                                                      : kv_cache_int4_unpack_i8x16(second);
        const unsigned codes[8] = {static_cast<unsigned>(lo.x), static_cast<unsigned>(lo.y),
                                   static_cast<unsigned>(lo.z), static_cast<unsigned>(lo.w),
                                   static_cast<unsigned>(hi.x), static_cast<unsigned>(hi.y),
                                   static_cast<unsigned>(hi.z), static_cast<unsigned>(hi.w)};
        codes_to_f16(codes, fp16_scale(r.scale), out);
    } else if constexpr (C == Coding::RootE8) {
        unsigned codes[8];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const unsigned word  = b < 2 ? r.w[0].x : r.w[0].y;
            const unsigned block = word >> (16 * (b & 1));
            const uint2 decoded =
                kv_cache_e8_root_decode_block(block & 0xffu, (block >> 8) & 0xffu);
            codes[2 * b]     = decoded.x;
            codes[2 * b + 1] = decoded.y;
        }
        codes_to_f16(codes, fp16_scale(r.scale), out);
    } else if constexpr (C == Coding::Fp8) {
        // The bit-level decode is the code times 2^-8; the scale carries the 2^8.
        const __half2 scale     = __hmul2(fp16_scale(r.scale), __float2half2_rn(256.0f));
        const unsigned words[8] = {r.w[0].x, r.w[0].y, r.w[0].z, r.w[0].w,
                                   r.w[1].x, r.w[1].y, r.w[1].z, r.w[1].w};
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const __half2 v =
                __hmul2(kv_cache_fp8_code2_to_half2_x2m8(words[i >> 1] >> (16 * (i & 1))), scale);
            out[i] = *reinterpret_cast<const unsigned*>(&v);
        }
    } else {
        // NVFP4: E2M1 pairs over one E4M3 scale per 16 dimensions, exact in FP16 (every E4M3 scale
        // is an FP16, and the bit-level decode times 2^8 restores it).
        const unsigned words[4] = {r.w[0].x, r.w[0].y, r.w[0].z, r.w[0].w};
        const __half2 two8      = __float2half2_rn(256.0f);
        const unsigned s0       = r.scale & 0xffu;
        const unsigned s1       = (r.scale >> 8) & 0xffu;
        const __half2 scale0    = __hmul2(kv_cache_fp8_code2_to_half2_x2m8(s0 | (s0 << 8)), two8);
        const __half2 scale1    = __hmul2(kv_cache_fp8_code2_to_half2_x2m8(s1 | (s1 << 8)), two8);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const __half2 v = __hmul2(kv_cache_e2m1x2_to_half2(words[i >> 2] >> (8 * (i & 3))),
                                      i < 8 ? scale0 : scale1);
            out[i]          = *reinterpret_cast<const unsigned*>(&v);
        }
    }
}

// K and V of the CTA's current key tile [k0, k0 + 32), in shared memory, as attend_tile reads
// them. Portable rows take the products on every architecture (the partial form).
template <bool Portable>
struct SharedRows {
    static constexpr bool kPortableMma = Portable;
    const __half* k_s;
    const __half* v_s;
    int k0;

    __device__ __forceinline__ const __half* key(int k) const { return k_s + (k - k0) * kStride; }

    __device__ __forceinline__ uint4 key8(const __half* row, int d) const {
        return *reinterpret_cast<const uint4*>(row + d);
    }

    __device__ __forceinline__ const __half* value(int k, int dim0) const {
        return v_s + (k - k0) * kStride + dim0;
    }

    __device__ __forceinline__ uint2 value4(const __half* row, int n) const {
        return *reinterpret_cast<const uint2*>(row + 32 * n);
    }
};

// A BF16 cache's tile as SharedRows reads it, and the BF16 values its FP16 stage saturates: the
// queries of the CTA's rows (numbered from 0 at the call's first), and the keys and values of the
// pages. The flags say whether the tile's keys or values, or the CTA's queries, hold a value
// beyond FP16 (sm70_attention::attend_tile's exact rows). A position-split rank's pages are at
// physical / page_stride of its planes (only the tiles of its own pages are attended).
template <bool Portable>
struct SharedBf16Rows : SharedRows<Portable> {
    static constexpr bool kExact = true;
    const __nv_bfloat16* cache_k;
    const __nv_bfloat16* cache_v;
    const std::int32_t* table;
    std::int32_t kv_heads;
    int kv_head;
    const __nv_bfloat16* q;
    std::int32_t q_heads;
    int group;
    int row_count;
    bool tile_beyond;
    bool query_beyond;
    std::int32_t page_stride;

    __device__ __forceinline__ uint4 key8(const __half* row, int d, unsigned&) const {
        return SharedRows<Portable>::key8(row, d);
    }
    __device__ __forceinline__ uint2 value4(const __half* row, int n, unsigned&) const {
        return SharedRows<Portable>::value4(row, n);
    }
    __device__ __forceinline__ bool keys_beyond(unsigned) const { return tile_beyond; }
    __device__ __forceinline__ bool values_beyond(unsigned) const { return tile_beyond; }

    __device__ __forceinline__ const __nv_bfloat16* query(int row) const {
        return row < row_count ? q + (static_cast<std::int64_t>(row / group) * q_heads +
                                      kv_head * group + row % group) *
                                         kHeadDim
                               : nullptr;
    }
    __device__ __forceinline__ const __nv_bfloat16* page_row(const __nv_bfloat16* plane,
                                                             int k) const {
        const std::int64_t page = table[k >> kPagedKVPageShift] / page_stride;
        return plane +
               ((page * kv_heads + kv_head) * kPagedKVPageSize + (k & kPagedKVPageMask)) * kHeadDim;
    }
    __device__ __forceinline__ const __nv_bfloat16* key_bf16(int k) const {
        return page_row(cache_k, k);
    }
    __device__ __forceinline__ const __nv_bfloat16* value_bf16(int k) const {
        return page_row(cache_v, k);
    }
};

// Whether any of a BF16 Raw's 32 values is beyond FP16 (zero for the other codings).
template <Coding C>
__device__ __forceinline__ bool raw_beyond(const Raw& r) {
    if constexpr (C == Coding::Bf16) {
        unsigned bits = 0u;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            bits |= sm70_attention::beyond_bits(r.w[i].x) | sm70_attention::beyond_bits(r.w[i].y) |
                    sm70_attention::beyond_bits(r.w[i].z) | sm70_attention::beyond_bits(r.w[i].w);
        }
        return (bits & sm70_attention::kBeyondBits) != 0u;
    } else {
        (void)r;
        return false;
    }
}

__device__ __forceinline__ void store_row(__half* dst, const unsigned (&words)[16]) {
    auto* p = reinterpret_cast<uint4*>(dst);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        p[i] = make_uint4(words[4 * i], words[4 * i + 1], words[4 * i + 2], words[4 * i + 3]);
    }
}

// The rows that carry a query: all of them, or the first valid_columns[0] of a masked row.
__device__ __forceinline__ std::int32_t valid_tokens(const std::int32_t* valid_columns,
                                                     std::int32_t tokens) {
    return valid_columns == nullptr ? tokens : max(0, min(tokens, valid_columns[0]));
}

// Packed row r of a KV head is token r / G and query head kv_head * G + r % G (G = the group's
// query heads). Rows of the tokens past the valid ones are written as zeros. The Partial form
// attends only over the tiles of the pages `partition` gives this rank and writes the rows'
// normalized FP32 outputs and log-sum-exps to `partial_out` (causal_attention_partial_elements)
// in place of `out`; a row without a key here, or past the valid tokens, gets 0 and -inf.
template <bool TuringShapes, class Metadata, Coding K, Coding V, bool RotatedKeys,
          bool RotatedValues, bool Partial>
__launch_bounds__(PromptShape<TuringShapes>::kThreads) __global__
    void prompt_kernel(const __nv_bfloat16* __restrict__ q,
                       const std::int32_t* __restrict__ positions,
                       const std::int32_t* __restrict__ valid_columns, Metadata metadata,
                       const void* __restrict__ cache_k, const void* __restrict__ cache_v,
                       const void* __restrict__ scale_k, const void* __restrict__ scale_v,
                       std::int32_t tokens, std::int32_t q_heads, std::int32_t kv_heads,
                       float scale, __nv_bfloat16* __restrict__ out, CausalKeyPartition partition,
                       float* __restrict__ partial_out) {
    if constexpr (!shape_set_in_image(TuringShapes)) { return; } // not this image's set
    using Shape            = PromptShape<TuringShapes>;
    constexpr int kRows    = Shape::kRows;
    constexpr int kWarps   = Shape::kWarps;
    constexpr int kThreads = Shape::kThreads;
    constexpr int kLoads   = Shape::kLoads;
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    __half* q_s = reinterpret_cast<__half*>(shared_bytes);
    __half* k_s = q_s + kRows * kStride;
    __half* v_s = k_s + kTile * kStride;
    __half* p_s = v_s + kTile * kStride;

    const int group   = q_heads / kv_heads;
    const int kv_head = static_cast<int>(blockIdx.y);
    // Later blocks see more keys; they start first.
    const int block     = static_cast<int>(gridDim.x - 1 - blockIdx.x);
    const int row0      = block * kRows;
    const int tid       = static_cast<int>(threadIdx.x);
    const int warp      = tid >> 5;
    const int lane      = tid & 31;
    const int valid     = valid_tokens(valid_columns, tokens);
    const int row_count = valid * group;
    const auto q_index  = [&](int row, int d) {
        return (static_cast<std::int64_t>(row / group) * q_heads + kv_head * group + row % group) *
                   kHeadDim +
               d;
    };

    // The CTA's query rows in FP16 (rotated like the keys), zero past the valid ones.
    constexpr bool Exact = K == Coding::Bf16 && V == Coding::Bf16;
    bool query_beyond    = false;
    for (int r = warp; r < kRows; r += kWarps) {
        const int row   = row0 + r;
        const bool live = row < row_count;
        float values[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            values[i] = live ? __bfloat162float(q[q_index(row, lane + 32 * i)]) : 0.0f;
        }
        if constexpr (RotatedKeys) { normalized_hadamard_d256_inplace(values, lane); }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            query_beyond |= !(fabsf(values[i]) <= 65504.0f);
            q_s[r * kStride + lane + 32 * i] =
                __float2half_rn(fminf(fmaxf(values[i], -65504.0f), 65504.0f));
        }
    }
    // The keys the block's rows see: up to the largest position among its valid tokens.
    int window = 0;
    for (int t = row0 / group; t <= (min(row0 + kRows, row_count) - 1) / group && t < valid; ++t) {
        window = max(window, positions[t] + 1);
    }
    const int lane_row_a = (lane & 1) + 4 * (lane >> 4);
    const int row_a      = row0 + warp * 8 + lane_row_a;
    const int row_b      = row_a + 2;
    const int qabs_a     = row_a < row_count ? positions[row_a / group] : -1;
    const int qabs_b     = row_b < row_count ? positions[row_b / group] : -1;
    int warp_position    = max(qabs_a, qabs_b);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        warp_position = max(warp_position, __shfl_xor_sync(0xffffffffu, warp_position, mask));
    }
    const __half* q_row = q_s + (warp * 8 + (lane & 3) + 4 * (lane >> 4)) * kStride;
    __half* p_tile      = p_s + warp * 8 * kPStride;
    sm70_attention::RowState st;

    // A thread's loader slot s: key k0 + tid / 8 + s * kThreads / 8, dimensions
    // [32 (tid % 8), +32).
    const int load_key        = tid >> 3;
    const int part            = tid & 7;
    const std::int32_t* table = metadata.block_table();
    // The page of the planes that holds tile k0 (a tile lies within one page), or -1 for a page
    // another rank holds.
    const auto tile_page = [&](int k0) -> int {
        const int physical = table[k0 >> kPagedKVPageShift];
        if constexpr (Partial) {
            return causal_strided_page(physical, partition);
        } else {
            return physical;
        }
    };
    const auto read = [&](int k0, Raw (&key_raw)[kLoads], Raw (&value_raw)[kLoads]) {
#pragma unroll
        for (int s = 0; s < kLoads; ++s) {
            const int key  = k0 + load_key + s * (kThreads / 8);
            const int page = key < window ? tile_page(k0) : -1;
            if (page >= 0) {
                const std::int64_t row = (static_cast<std::int64_t>(page) * kv_heads + kv_head) *
                                             kPagedKVPageSize +
                                         (key & kPagedKVPageMask);
                key_raw[s]   = read_raw<K>(cache_k, scale_k, row, part);
                value_raw[s] = read_raw<V>(cache_v, scale_v, row, part);
            } else {
                key_raw[s]   = Raw{};
                value_raw[s] = Raw{};
            }
        }
    };
    const int tiles = (window + kTile - 1) / kTile;
    Raw key_raw[kLoads]{};
    Raw value_raw[kLoads]{};
    if (tiles > 0) { read(0, key_raw, value_raw); }
    // A BF16 query beyond FP16 sends every tile through the exact scores; the others ignore it.
    query_beyond = __syncthreads_or(query_beyond) != 0;
    for (int tile = 0; tile < tiles; ++tile) {
        const int k0 = tile * kTile;
        if constexpr (Partial) {
            // Uniform over the CTA: every thread's key of the tile is on one page.
            if (tile_page(k0) < 0) {
                if (tile + 1 < tiles) { read(k0 + kTile, key_raw, value_raw); }
                continue;
            }
        }
        unsigned words[16];
        bool raw_saturates = false;
#pragma unroll
        for (int s = 0; s < kLoads; ++s) {
            const int key_row = load_key + s * (kThreads / 8);
            raw_saturates |= raw_beyond<K>(key_raw[s]) || raw_beyond<V>(value_raw[s]);
            decode_raw<K>(key_raw[s], words);
            store_row(k_s + key_row * kStride + 32 * part, words);
            decode_raw<V>(value_raw[s], words);
            store_row(v_s + key_row * kStride + 32 * part, words);
        }
        const bool tile_beyond = __syncthreads_or(raw_saturates) != 0;
        if (tile + 1 < tiles) { read(k0 + kTile, key_raw, value_raw); }
        if (warp_position >= k0) {
            if constexpr (Exact) {
                const SharedBf16Rows<Partial> rows{{k_s, v_s, k0},
                                                   static_cast<const __nv_bfloat16*>(cache_k),
                                                   static_cast<const __nv_bfloat16*>(cache_v),
                                                   table,
                                                   kv_heads,
                                                   kv_head,
                                                   q,
                                                   q_heads,
                                                   group,
                                                   row_count,
                                                   tile_beyond,
                                                   query_beyond,
                                                   Partial ? partition.page_stride : 1};
                sm70_attention::attend_tile(rows, q_row, p_tile, k0, 0, window, qabs_a, qabs_b,
                                            scale, st, row0 + warp * 8);
            } else {
                sm70_attention::attend_tile(SharedRows<Partial>{k_s, v_s, k0}, q_row, p_tile, k0,
                                            0, window, qabs_a, qabs_b, scale, st);
            }
        }
        __syncthreads();
    }

    // Element i of product n: row (lane & 1) + (i & 2) + 4h, dimension 32n + 8qp + (i & 4) +
    // (lane & 2) + (i & 1).
    const float inv_a = st.l_a > 0.0f ? 1.0f / st.l_a : 0.0f;
    const float inv_b = st.l_b > 0.0f ? 1.0f / st.l_b : 0.0f;
    const int qp      = (lane >> 2) & 3;
    const int rows    = tokens * group;
    // The partial form's log-sum-exp of a row (natural log units: the scores are scaled), -inf
    // without a key here or a query.
    const auto row_lse = [&](int row, float m, float l) {
        return row < row_count && l > 0.0f ? m + logf(l) : -CUDART_INF_F;
    };
    const float lse_a = row_lse(row_a, st.m_a, st.l_a);
    const float lse_b = row_lse(row_b, st.m_b, st.l_b);
    if constexpr (RotatedValues) {
        // The values were stored rotated: each row goes back through the (self-inverse)
        // transform, from FP32 rows in the stages the tiles no longer need.
        float* rows_s = reinterpret_cast<float*>(shared_bytes) + warp * 8 * kHeadDim;
        float* lse_s  = reinterpret_cast<float*>(shared_bytes) + kRows * kHeadDim + warp * 8;
        if constexpr (Partial) {
            if (qp == 0 && (lane & 2) == 0) {
                lse_s[lane_row_a]     = lse_a;
                lse_s[lane_row_a + 2] = lse_b;
            }
        }
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int local              = lane_row_a + (i & 2);
                const int d                  = 32 * n + 8 * qp + (i & 4) + (lane & 2) + (i & 1);
                rows_s[local * kHeadDim + d] = st.acc[n][i] * ((i & 2) ? inv_b : inv_a);
            }
        }
        __syncwarp();
        for (int local = 0; local < 8; ++local) {
            const int row = row0 + warp * 8 + local;
            if (row >= rows) { break; }
            float values[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) { values[i] = rows_s[local * kHeadDim + lane + 32 * i]; }
            normalized_hadamard_d256_inplace(values, lane);
            const bool live = row < row_count;
            if constexpr (Partial) {
                const int q_head = kv_head * group + row % group;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    causal_store_partial_output(partial_out, q_head, lane + 32 * i, row / group,
                                                q_heads, live ? values[i] : 0.0f, lse_s[local],
                                                lane == 0 && i == 0);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    out[q_index(row, lane + 32 * i)] =
                        __float2bfloat16_rn(live ? values[i] : 0.0f);
                }
            }
        }
    } else {
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
            for (int i = 0; i < 8; i += 2) {
                const int row = (i & 2) ? row_b : row_a;
                if (row >= rows) { continue; }
                const float inv = row < row_count ? ((i & 2) ? inv_b : inv_a) : 0.0f;
                const int d     = 32 * n + 8 * qp + (i & 4) + (lane & 2);
                if constexpr (Partial) {
                    const int q_head = kv_head * group + row % group;
                    const float lse  = (i & 2) ? lse_b : lse_a;
                    const bool lead  = n == 0 && qp == 0 && (lane & 2) == 0 && (i & 4) == 0;
                    causal_store_partial_output(partial_out, q_head, d, row / group, q_heads,
                                                st.acc[n][i] * inv, lse, lead);
                    causal_store_partial_output(partial_out, q_head, d + 1, row / group, q_heads,
                                                st.acc[n][i + 1] * inv, lse, false);
                } else {
                    *reinterpret_cast<__nv_bfloat162*>(out + q_index(row, d)) =
                        __floats2bfloat162_rn(st.acc[n][i] * inv, st.acc[n][i + 1] * inv);
                }
            }
        }
    }
}

// The plane codings of each storage, and which planes it stores rotated (every quantized key
// coding; NVFP4 values).
template <class F>
void with_codings(KvCacheStorage storage, F&& f) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return f.template operator()<Coding::Bf16, Coding::Bf16, false, false>();
    case KvCacheStorage::Int8Group64:
        return f.template operator()<Coding::Int8, Coding::Int8, true, false>();
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return f.template operator()<Coding::Int8, Coding::Int4, true, false>();
    case KvCacheStorage::RotatedLloyd4KeyInt4Value:
        return f.template operator()<Coding::Lloyd4, Coding::Int4, true, false>();
    case KvCacheStorage::RotatedInt4KeyInt4ValueE8:
        return f.template operator()<Coding::Int4E8, Coding::Int4, true, false>();
    case KvCacheStorage::RotatedE8RootKeyInt4Value:
        return f.template operator()<Coding::RootE8, Coding::Int4, true, false>();
    case KvCacheStorage::Fp8E4M3Row256:
        return f.template operator()<Coding::Fp8, Coding::Fp8, true, false>();
    case KvCacheStorage::Nvfp4Group16:
        return f.template operator()<Coding::Nvfp4, Coding::Nvfp4, true, true>();
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return f.template operator()<Coding::Fp8, Coding::Nvfp4, true, true>();
    case KvCacheStorage::Bf16Packed:
        throw std::invalid_argument(
            "causal_attention_prompt_sm70: a bf16p KV cache takes neither the Volta prompt nor "
            "position-split attention");
    }
    throw std::invalid_argument("causal_attention_prompt_sm70: unknown KV storage");
}

template <bool Partial, class View, class Metadata>
void launch(const Tensor& q, const Tensor& positions, const std::int32_t* valid_columns,
            float scale, const View& cache, Metadata metadata, Tensor* out,
            CausalKeyPartition partition, float* partial_out, cudaStream_t stream) {
    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t tokens   = q.ne[2];
    const std::int32_t kv_heads = cache.num_kv_heads;
    if (cache.head_dim != kHeadDim || q.ne[0] != kHeadDim || kv_heads <= 0 ||
        q_heads % kv_heads != 0 || tokens <= 0) {
        throw std::invalid_argument("causal_attention_prompt_sm70: unsupported geometry");
    }
    const std::int32_t rows = tokens * (q_heads / kv_heads);
    with_shape_set([&]<bool TuringShapes>() {
        using Shape = PromptShape<TuringShapes>;
        const dim3 grid(static_cast<unsigned>((rows + Shape::kRows - 1) / Shape::kRows),
                        static_cast<unsigned>(kv_heads));
        with_codings(cache.storage,
                     [&]<Coding K, Coding V, bool RotatedKeys, bool RotatedValues>() {
            auto kernel =
                prompt_kernel<TuringShapes, Metadata, K, V, RotatedKeys, RotatedValues, Partial>;
            raise_dynamic_shared_memory(kernel, Shape::kSharedBytes);
            kernel<<<grid, Shape::kThreads, Shape::kSharedBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::int32_t*>(positions.data), valid_columns, metadata,
                cache.k_pages.data, cache.v_pages.data, cache.k_scale_pages.data,
                cache.v_scale_pages.data, tokens, q_heads, kv_heads, scale,
                out == nullptr ? nullptr : static_cast<__nv_bfloat16*>(out->data), partition,
                partial_out);
            CUDA_CHECK(cudaGetLastError());
        });
    });
}

} // namespace

bool causal_attention_prompt_sm70_applies(KvCacheStorage storage, std::int32_t batch) {
    (void)storage; // every KV storage has a coding here
    return batch == 1 && current_device_volta();
}

void causal_attention_prompt_sm70_launch(const Tensor& q, const Tensor& positions, float scale,
                                         const PagedKVLayerView& cache, Tensor& out,
                                         cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    launch<false>(q, positions, nullptr, scale, cache, metadata, &out, {}, nullptr, stream);
}

void causal_attention_prompt_sm70_launch(const Tensor& q, const Tensor& positions,
                                         const Tensor& valid_columns, const Tensor& table_rows,
                                         float scale, const PagedKVBatchLayerView& cache,
                                         Tensor& out, cudaStream_t stream) {
    const PagedKVBatchMetadata<false> metadata{
        .tables        = static_cast<const std::int32_t*>(cache.block_tables.data),
        .valid_columns = nullptr,
        .table_rows    = static_cast<const std::int32_t*>(table_rows.data),
        .table_stride  = cache.block_tables.ne[0],
    };
    launch<false>(q, positions, static_cast<const std::int32_t*>(valid_columns.data), scale,
                  cache, metadata, &out, {}, nullptr, stream);
}

void causal_attention_prompt_partial_launch(const Tensor& q, const Tensor& positions,
                                            const Tensor& valid_columns, const Tensor& table_rows,
                                            float scale, PagedKVBatchLayerView cache,
                                            float* partial_out, cudaStream_t stream) {
    if (q.ne[3] != 1 || partial_out == nullptr) {
        throw std::invalid_argument(
            "causal_attention_prompt_partial: one sequence and a partial output are required");
    }
    const PagedKVBatchMetadata<false> metadata{
        .tables        = static_cast<const std::int32_t*>(cache.block_tables.data),
        .valid_columns = nullptr,
        .table_rows    = static_cast<const std::int32_t*>(table_rows.data),
        .table_stride  = cache.block_tables.ne[0],
    };
    const CausalKeyPartition partition{
        .segment_keys = 0, .page_stride = cache.page_stride, .page_residue = cache.page_residue};
    launch<true>(q, positions, static_cast<const std::int32_t*>(valid_columns.data), scale, cache,
                 metadata, nullptr, partition, partial_out, stream);
}

} // namespace ninfer::ops::detail
