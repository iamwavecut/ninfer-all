// ninfer::ops - Qwen3.8-Flash-Next sparse attention over selected blocks and the query's own
// incomplete block (contract in include/ninfer/ops/sparse_attention.h), over every KV head or a
// tensor-parallel share of them. A CTA takes one (query, KV head): its query heads (twelve of each
// of two KV heads, or six of each of four), one warp each, stream positions in 32-key tiles staged
// in shared memory, each lane scoring one key and accumulating eight output dims online. Up to
// eight queries split their positions over CTAs of 64 positions each (a grid fixed for any
// position, so a graph replays it), and a second launch merges the partial softmaxes; wider calls
// run the whole list in one CTA per (query, KV head), on tensor cores from Ampere on
// (sparse_attention_tc_kernel, below).
//
// A quantized cache is decoded while a tile is staged, sixteen dimensions per thread, to FP16 in
// shared memory: keys in the rotated basis the append stored them in, against a query rotated by
// the same normalized Hadamard transform, and values as stored -- rotated too for NVFP4 and K8V4,
// whose reduction then takes the transform back.
#include "ninfer/ops/sparse_attention.h"

#include "ninfer/ops/kv_stream.h"

#include "core/device_launch.h"
#include "core/layout.h"
#include "core/paged_kv_storage.h"
#include "ops/common/device_tier.h"
#include "ops/common/math.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/kv_cache/bf16p_codec.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops {
namespace {

constexpr int kHeadDim     = 256;
constexpr int kQueryHeads  = 24;
// Query heads per KV head: 12 of each of two KV heads (Qwen3.8-Flash-Next) or 6 of each of four
// (dense Qwen4); a tensor-parallel share holds whole KV heads with their queries.
constexpr int kWideGroup   = 12;
constexpr int kNarrowGroup = 6;
constexpr int kTopBlocks   = 512;
constexpr int kTile        = 32;
constexpr int kRowStride   = kHeadDim + 2; // BF16 elements; 129 words per row spreads the banks
constexpr int kSplitTokens    = 8;
constexpr int kSplitPositions = 2 * kTile;
constexpr int kMaxPositions   = 4 * kTopBlocks + 3; // the selected blocks and the query's own
constexpr int kSplits         = (kMaxPositions + kSplitPositions - 1) / kSplitPositions;
constexpr int kPartialFloats  = kHeadDim + 2; // unnormalised output, running max, running sum

static_assert(kPagedKVPageSize == 1 << 6 && kKvStreamPageTokens == 1 << 2);

constexpr int kChunk         = 16; // dimensions a thread decodes at once from a quantized row
constexpr int kChunks        = kHeadDim / kChunk;
constexpr int kWordsPerChunk = kChunk / 2;

__device__ __forceinline__ int position_at(const int* blocks, int count, int tail_begin, int i) {
    return i < 4 * count ? blocks[i >> 2] * 4 + (i & 3) : tail_begin + (i - 4 * count);
}

// The four planes of one paged KV layer; the scale planes are null where the storage has none.
// `heads` KV heads per page: every one, or a tensor-parallel share's.
struct KvPlanes {
    const void* k       = nullptr;
    const void* v       = nullptr;
    const void* k_scale = nullptr;
    const void* v_scale = nullptr;
    int heads           = 1;
};

// Element offset of `leading` in the row of position `p` and KV head `head`, for a plane whose
// rows are `Leading` elements wide and whose pages hold 2^Shift positions: 64 for a paged layer,
// 4 for a streamed layer's pool (kv_stream.h).
template <int Leading, int Shift>
__device__ __forceinline__ std::int64_t plane_offset(const std::int32_t* table, int heads,
                                                     int head, int p, int leading) {
    const std::int64_t page = table[p >> Shift];
    return static_cast<std::int64_t>(Leading) * ((page * heads + head) << Shift) +
           static_cast<std::int64_t>(Leading) * (p & ((1 << Shift) - 1)) + leading;
}

__device__ __forceinline__ float scale_at(const void* plane, std::int64_t offset) {
    return __half2float(static_cast<const __half*>(plane)[offset]);
}

// Sixteen int8 codes in dimension order (byte 0 of x is the first) times `s`, as eight FP16 pairs.
__device__ __forceinline__ void codes_to_half(int4 codes, float s,
                                              unsigned (&out)[kWordsPerChunk]) {
    const unsigned words[4] = {static_cast<unsigned>(codes.x), static_cast<unsigned>(codes.y),
                               static_cast<unsigned>(codes.z), static_cast<unsigned>(codes.w)};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const auto code = [&](int byte) {
            return static_cast<float>(static_cast<std::int8_t>((words[w] >> (8 * byte)) & 0xffu));
        };
        const __half2 low  = __floats2half2_rn(code(0) * s, code(1) * s);
        const __half2 high = __floats2half2_rn(code(2) * s, code(3) * s);
        out[2 * w]         = *reinterpret_cast<const unsigned*>(&low);
        out[2 * w + 1]     = *reinterpret_cast<const unsigned*>(&high);
    }
}

__device__ __forceinline__ void halves_to_words(int4 low, int4 high,
                                                unsigned (&out)[kWordsPerChunk]) {
    out[0] = static_cast<unsigned>(low.x);
    out[1] = static_cast<unsigned>(low.y);
    out[2] = static_cast<unsigned>(low.z);
    out[3] = static_cast<unsigned>(low.w);
    out[4] = static_cast<unsigned>(high.x);
    out[5] = static_cast<unsigned>(high.y);
    out[6] = static_cast<unsigned>(high.z);
    out[7] = static_cast<unsigned>(high.w);
}

// Key dimensions [16 c, 16 c + 16) of one row, in the rotated basis, as FP16 pairs.
template <KvCacheStorage Storage, int Shift>
__device__ __forceinline__ void decode_key(const KvPlanes& planes, const std::int32_t* table,
                                           int head, int p, int c,
                                           unsigned (&out)[kWordsPerChunk]) {
    const int d = c * kChunk;
    if constexpr (Storage == KvCacheStorage::Int8Group64 ||
                  Storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64) {
        const auto* codes =
            static_cast<const std::int8_t*>(planes.k) + plane_offset<256, Shift>(table, planes.heads, head, p, d);
        codes_to_half(load_vec<int4>(codes),
                      scale_at(planes.k_scale, plane_offset<4, Shift>(table, planes.heads, head, p, d / 64)), out);
    } else if constexpr (Storage == KvCacheStorage::RotatedLloyd4KeyInt4Value ||
                         Storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8) {
        const auto* bytes =
            static_cast<const std::uint8_t*>(planes.k) + plane_offset<128, Shift>(table, planes.heads, head, p, d / 2);
        const uint2 packed = load_vec<uint2>(bytes);
        const int4 codes   = Storage == KvCacheStorage::RotatedLloyd4KeyInt4Value
                                 ? kv_cache_lloyd4_expand16(packed)
                                 : kv_cache_int4_unpack_i8x16(packed);
        codes_to_half(codes, scale_at(planes.k_scale, plane_offset<4, Shift>(table, planes.heads, head, p, d / 64)),
                      out);
    } else if constexpr (Storage == KvCacheStorage::RotatedE8RootKeyInt4Value) {
        const auto* bytes =
            static_cast<const std::uint8_t*>(planes.k) + plane_offset<64, Shift>(table, planes.heads, head, p, d / 4);
        codes_to_half(kv_cache_e8_root_unpack_i8x16(load_vec<std::uint32_t>(bytes)),
                      scale_at(planes.k_scale, plane_offset<4, Shift>(table, planes.heads, head, p, d / 64)), out);
    } else if constexpr (Storage == KvCacheStorage::Fp8E4M3Row256 ||
                         Storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        const auto* codes =
            static_cast<const std::uint8_t*>(planes.k) + plane_offset<256, Shift>(table, planes.heads, head, p, d);
        const __half s =
            static_cast<const __half*>(planes.k_scale)[plane_offset<1, Shift>(table, planes.heads, head, p, 0)];
        halves_to_words(kv_cache_fp8_dequant_f16x8(codes, s),
                        kv_cache_fp8_dequant_f16x8(codes + 8, s), out);
    } else {
        static_assert(Storage == KvCacheStorage::Nvfp4Group16);
        const auto* codes =
            static_cast<const std::uint8_t*>(planes.k) + plane_offset<128, Shift>(table, planes.heads, head, p, d / 2);
        const std::uint8_t s =
            static_cast<const std::uint8_t*>(planes.k_scale)[plane_offset<16, Shift>(table, planes.heads, head, p, c)];
        halves_to_words(kv_cache_nvfp4_dequant_f16x8(codes, s),
                        kv_cache_nvfp4_dequant_f16x8(codes + 4, s), out);
    }
}

// Value dimensions [16 c, 16 c + 16) of one row as FP16 pairs.
template <KvCacheStorage Storage, int Shift>
__device__ __forceinline__ void decode_value(const KvPlanes& planes, const std::int32_t* table,
                                             int head, int p, int c,
                                             unsigned (&out)[kWordsPerChunk]) {
    const int d = c * kChunk;
    if constexpr (Storage == KvCacheStorage::Int8Group64) {
        const auto* codes =
            static_cast<const std::int8_t*>(planes.v) + plane_offset<256, Shift>(table, planes.heads, head, p, d);
        codes_to_half(load_vec<int4>(codes),
                      scale_at(planes.v_scale, plane_offset<4, Shift>(table, planes.heads, head, p, d / 64)), out);
    } else if constexpr (Storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64 ||
                         Storage == KvCacheStorage::RotatedLloyd4KeyInt4Value ||
                         Storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8 ||
                         Storage == KvCacheStorage::RotatedE8RootKeyInt4Value) {
        const auto* bytes =
            static_cast<const std::uint8_t*>(planes.v) + plane_offset<128, Shift>(table, planes.heads, head, p, d / 2);
        codes_to_half(kv_cache_int4_unpack_i8x16(load_vec<uint2>(bytes)),
                      scale_at(planes.v_scale, plane_offset<8, Shift>(table, planes.heads, head, p, d / 32)), out);
    } else if constexpr (Storage == KvCacheStorage::Fp8E4M3Row256) {
        const auto* codes =
            static_cast<const std::uint8_t*>(planes.v) + plane_offset<256, Shift>(table, planes.heads, head, p, d);
        const __half s =
            static_cast<const __half*>(planes.v_scale)[plane_offset<1, Shift>(table, planes.heads, head, p, 0)];
        halves_to_words(kv_cache_fp8_dequant_f16x8(codes, s),
                        kv_cache_fp8_dequant_f16x8(codes + 8, s), out);
    } else {
        static_assert(Storage == KvCacheStorage::Nvfp4Group16 ||
                      Storage == KvCacheStorage::Fp8KeyNvfp4Value);
        const auto* codes =
            static_cast<const std::uint8_t*>(planes.v) + plane_offset<128, Shift>(table, planes.heads, head, p, d / 2);
        const std::uint8_t s =
            static_cast<const std::uint8_t*>(planes.v_scale)[plane_offset<16, Shift>(table, planes.heads, head, p, c)];
        halves_to_words(kv_cache_nvfp4_dequant_f16x8(codes, s),
                        kv_cache_nvfp4_dequant_f16x8(codes + 4, s), out);
    }
}

// The staged tiles hold the cache's own BF16 rows, or a quantized cache's rows decoded to FP16.
template <KvCacheStorage Storage>
using TileElement = std::conditional_t<Storage == KvCacheStorage::BFloat16 ||
                                           Storage == KvCacheStorage::Bf16Packed,
                                       __nv_bfloat16, __half>;

__device__ __forceinline__ float2 to_float2(const __nv_bfloat162& pair) {
    return __bfloat1622float2(pair);
}

__device__ __forceinline__ float2 to_float2(const __half2& pair) { return __half22float2(pair); }

// Split: this CTA takes positions [64 z, 64 z + 64) of the list and leaves its partial softmax in
// `partial`; otherwise the whole list, normalised into `out`.
template <int Group, bool Split, KvCacheStorage Storage, int Shift>
__global__ void __launch_bounds__(Group * 32)
    sparse_attention_kernel(const __nv_bfloat16* __restrict__ q, const int* __restrict__ first,
                            const int* __restrict__ selected, const int* __restrict__ counts,
                            KvPlanes planes, const std::int32_t* __restrict__ block_table,
                            float scale, __nv_bfloat16* __restrict__ out,
                            float* __restrict__ partial, int q_heads) {
    using Element = TileElement<Storage>;
    using Pair    = std::conditional_t<std::is_same_v<Element, __half>, __half2, __nv_bfloat162>;
    // bf16p rows decode to the BF16 a bf16 cache holds: neither quantized nor rotated.
    constexpr bool kPacked    = Storage == KvCacheStorage::Bf16Packed;
    constexpr bool kQuantized =
        Storage != KvCacheStorage::BFloat16 && Storage != KvCacheStorage::Float16 && !kPacked;
    // NVFP4 and K8V4 store their values rotated as well as their keys.
    constexpr bool kRotatedValues =
        Storage == KvCacheStorage::Nvfp4Group16 || Storage == KvCacheStorage::Fp8KeyNvfp4Value;
    constexpr int kThreads = Group * 32;
    __shared__ float query[Group][kHeadDim];
    __shared__ Element keys[kTile * kRowStride];
    __shared__ Element values[kTile * kRowStride];

    const int t        = blockIdx.x;
    const int kv_head  = blockIdx.y;
    const int warp     = threadIdx.x >> 5;
    const int lane     = threadIdx.x & 31;
    const int q_head   = kv_head * Group + warp;
    const int position   = *first + t;
    const int count    = counts[t];
    const int tail_begin = (position + 1) / 4 * 4;
    const int total      = 4 * count + (position + 1 - tail_begin);
    const int* blocks    = selected + static_cast<std::int64_t>(t) * kTopBlocks;
    const int begin      = Split ? static_cast<int>(blockIdx.z) * kSplitPositions : 0;
    const int end        = Split ? min(total, begin + kSplitPositions) : total;

    {
        // Lane l holds dimensions l + 32 r, the layout the key rotation used.
        float row[kHeadDim / 32];
#pragma unroll
        for (int r = 0; r < kHeadDim / 32; ++r) {
            row[r] = __bfloat162float(
                q[(static_cast<std::int64_t>(t) * q_heads + q_head) * kHeadDim + lane +
                  32 * r]);
        }
        if constexpr (kQuantized) { normalized_hadamard_d256_inplace(row, lane); }
#pragma unroll
        for (int r = 0; r < kHeadDim / 32; ++r) { query[warp][lane + 32 * r] = row[r]; }
    }

    float running_max = -INFINITY, running_sum = 0.0f;
    float acc[8]      = {};
    for (int base = begin; base < end; base += kTile) {
        __syncthreads();
        // Stage the tile: 32 rows of K and V, 4-byte words (the padded rows are not 16B aligned).
        if constexpr (kQuantized) {
            for (int i = threadIdx.x; i < kTile * kChunks; i += kThreads) {
                const int row = i / kChunks, c = i % kChunks;
                const int index             = base + row;
                unsigned kw[kWordsPerChunk] = {}, vw[kWordsPerChunk] = {};
                if (index < end) {
                    const int p = position_at(blocks, count, tail_begin, index);
                    decode_key<Storage, Shift>(planes, block_table, kv_head, p, c, kw);
                    decode_value<Storage, Shift>(planes, block_table, kv_head, p, c, vw);
                }
                auto* key_words = reinterpret_cast<unsigned*>(&keys[row * kRowStride + c * kChunk]);
                auto* value_words =
                    reinterpret_cast<unsigned*>(&values[row * kRowStride + c * kChunk]);
#pragma unroll
                for (int w = 0; w < kWordsPerChunk; ++w) {
                    key_words[w]   = kw[w];
                    value_words[w] = vw[w];
                }
            }
        } else if constexpr (kPacked) {
            constexpr int kChunksPerRow = kHeadDim / 8;
            for (int i = threadIdx.x; i < kTile * kChunksPerRow; i += kThreads) {
                const int row = i / kChunksPerRow, c = i % kChunksPerRow;
                const int index = base + row;
                uint4 kw = make_uint4(0, 0, 0, 0), vw = make_uint4(0, 0, 0, 0);
                if (index < end) {
                    const int p = position_at(blocks, count, tail_begin, index);
                    const std::int64_t row_offset = plane_offset<kBf16pRowBytes, Shift>(
                        block_table, planes.heads, kv_head, p, 0);
                    const std::int64_t side_offset = plane_offset<kBf16pSideBytes, Shift>(
                        block_table, planes.heads, kv_head, p, 0);
                    kw = bf16p_decode8(static_cast<const std::uint8_t*>(planes.k) + row_offset,
                                       static_cast<const std::uint8_t*>(planes.k_scale) +
                                           side_offset,
                                       c);
                    vw = bf16p_decode8(static_cast<const std::uint8_t*>(planes.v) + row_offset,
                                       static_cast<const std::uint8_t*>(planes.v_scale) +
                                           side_offset,
                                       c);
                }
                auto* key_words   = reinterpret_cast<unsigned*>(&keys[row * kRowStride + 8 * c]);
                auto* value_words = reinterpret_cast<unsigned*>(&values[row * kRowStride + 8 * c]);
                key_words[0]      = kw.x;
                key_words[1]      = kw.y;
                key_words[2]      = kw.z;
                key_words[3]      = kw.w;
                value_words[0]    = vw.x;
                value_words[1]    = vw.y;
                value_words[2]    = vw.z;
                value_words[3]    = vw.w;
            }
        } else {
            const auto* k_pages = static_cast<const Element*>(planes.k);
            const auto* v_pages = static_cast<const Element*>(planes.v);
            for (int i = threadIdx.x; i < kTile * (kHeadDim / 2); i += kThreads) {
                const int row = i / (kHeadDim / 2), pair = i % (kHeadDim / 2);
                const int index = base + row;
                unsigned kw = 0, vw = 0;
                if (index < end) {
                    const int p               = position_at(blocks, count, tail_begin, index);
                    const std::int64_t offset =
                        plane_offset<kHeadDim, Shift>(block_table, planes.heads, kv_head, p,
                                                      2 * pair);
                    kw = *reinterpret_cast<const unsigned*>(k_pages + offset);
                    vw = *reinterpret_cast<const unsigned*>(v_pages + offset);
                }
                *reinterpret_cast<unsigned*>(&keys[row * kRowStride + 2 * pair])   = kw;
                *reinterpret_cast<unsigned*>(&values[row * kRowStride + 2 * pair]) = vw;
            }
        }
        __syncthreads();
        // Lane j scores key base + j for this warp's head.
        float score = -INFINITY;
        if (base + lane < end) {
            float dot       = 0.0f;
            const Pair* row = reinterpret_cast<const Pair*>(&keys[lane * kRowStride]);
#pragma unroll 8
            for (int pair = 0; pair < kHeadDim / 2; ++pair) {
                const float2 k = to_float2(row[pair]);
                dot = fmaf(query[warp][2 * pair], k.x, fmaf(query[warp][2 * pair + 1], k.y, dot));
            }
            score = dot * scale;
        }
        float tile_max = score;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, offset));
        }
        const float new_max    = fmaxf(running_max, tile_max);
        const float correction = __expf(running_max - new_max);
        const float weight     = base + lane < end ? __expf(score - new_max) : 0.0f;
        float weight_sum       = weight;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            weight_sum += __shfl_xor_sync(0xffffffffu, weight_sum, offset);
        }
        running_sum = running_sum * correction + weight_sum;
        running_max = new_max;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] *= correction;
        const int keys_in_tile = min(kTile, end - base);
        for (int j = 0; j < keys_in_tile; ++j) {
            const float p   = __shfl_sync(0xffffffffu, weight, j);
            const Pair* row = reinterpret_cast<const Pair*>(&values[j * kRowStride + lane * 8]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 v = to_float2(row[i]);
                acc[2 * i]     = fmaf(p, v.x, acc[2 * i]);
                acc[2 * i + 1] = fmaf(p, v.y, acc[2 * i + 1]);
            }
        }
    }
    if constexpr (kRotatedValues) {
        // The values were stored rotated, so the reduction is too: the same transform takes it
        // back, through this warp's query row, which scoring no longer needs.
        float* scratch = query[warp];
        __syncwarp();
#pragma unroll
        for (int i = 0; i < 8; ++i) scratch[lane * 8 + i] = acc[i];
        __syncwarp();
        float row[kHeadDim / 32];
#pragma unroll
        for (int r = 0; r < kHeadDim / 32; ++r) row[r] = scratch[lane + 32 * r];
        normalized_hadamard_d256_inplace(row, lane);
#pragma unroll
        for (int r = 0; r < kHeadDim / 32; ++r) scratch[lane + 32 * r] = row[r];
        __syncwarp();
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = scratch[lane * 8 + i];
    }
    if constexpr (Split) {
        float* slot =
            partial +
            ((static_cast<std::int64_t>(t) * kSplits + blockIdx.z) * q_heads + q_head) *
                kPartialFloats;
#pragma unroll
        for (int i = 0; i < 8; ++i) slot[lane * 8 + i] = acc[i];
        if (lane == 0) {
            slot[kHeadDim]     = running_max;
            slot[kHeadDim + 1] = running_sum;
        }
    } else {
        const float inverse = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
        __nv_bfloat16* destination =
            out + (static_cast<std::int64_t>(t) * q_heads + q_head) * kHeadDim + lane * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) destination[i] = __float2bfloat16_rn(acc[i] * inverse);
    }
}

// ---- Prompt calls on tensor cores ----------------------------------------------------------------
// Block (query, KV head) of four warps. The KV head's query heads, twelve or six, are the first
// rows of a 16-row tile (the other rows are zero). Positions pass through shared memory in tiles of
// 32: warp w scores keys 8 w .. 8 w + 7 of a tile on m16n8k16 MMAs against the whole query, the
// block shares each row's tile maximum, and warp w then adds P V for dimensions 64 w .. 64 w + 63
// over all 32 keys.
// A BF16 cache's query is BF16 already, so a score sums exact products of Q and K in FP32; FP16
// tiles (an F16 cache, or a quantized one decoded as the SIMT kernel decodes it) take Q in two FP16
// parts, Q rounded and its remainder times 2^12 rounded again (about 22 bits of Q). The
// probabilities enter P V in parts of the tile's type: two FP16 parts in the same way (22 bits),
// or three BF16 parts (24 bits, all of an FP32 probability). Each tile's P V starts from zero on
// the tensor cores, which round their sums toward zero, and joins the running sum in FP32; masks,
// the online softmax and the sums stay FP32. Rows stored as BF16 or FP16 are copied with cp.async,
// the K of tile c + 1 while tile c's softmax and P V run and its V while tile c + 1 is scored;
// other storages are decoded into the tiles at the same points.
constexpr int kTcWarps       = 4;
constexpr int kTcThreads     = kTcWarps * 32;
constexpr int kTcRows        = 16;
constexpr int kTcTile        = 32;
constexpr int kTcStride      = kHeadDim + 8; // elements: 528-byte rows put ldmatrix's rows on distinct banks
constexpr int kTcPStride     = kTcTile + 8;
constexpr float kTcRemainder = 4096.0f;

// Dynamic shared memory of the tensor-core kernel: the query's parts, one K and one V tile, the
// probabilities' parts, and each warp's row maxima and row sums.
template <bool TwoPartQuery>
struct TcLayout {
    static constexpr int kQueryBytes = (TwoPartQuery ? 2 : 1) * kTcRows * kTcStride * 2;
    static constexpr int kTileBytes  = kTcTile * kTcStride * 2;
    static constexpr int kPParts     = TwoPartQuery ? 2 : 3;
    static constexpr int kPBytes     = kPParts * kTcRows * kTcPStride * 2;
    static constexpr int kRowBytes   = kTcWarps * kTcRows * 4;
    static constexpr int kBytes      = kQueryBytes + 2 * kTileBytes + kPBytes + 2 * kRowBytes;
};

template <class Element>
__device__ __forceinline__ unsigned pack_pair(float a, float b) {
    if constexpr (std::is_same_v<Element, __nv_bfloat16>) {
        const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
        return *reinterpret_cast<const unsigned*>(&v);
    } else {
        const __half2 v = __floats2half2_rn(a, b);
        return *reinterpret_cast<const unsigned*>(&v);
    }
}

template <class Element>
__device__ __forceinline__ float2 unpack_pair(unsigned word) {
    if constexpr (std::is_same_v<Element, __nv_bfloat16>) {
        return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&word));
    } else {
        return __half22float2(*reinterpret_cast<const __half2*>(&word));
    }
}

template <class Element>
__device__ __forceinline__ void tc_mma(float (&c)[4], unsigned a0, unsigned a1, unsigned a2,
                                       unsigned a3, unsigned b0, unsigned b1) {
    if constexpr (std::is_same_v<Element, __nv_bfloat16>) {
        mma_bf16(c[0], c[1], c[2], c[3], a0, a1, a2, a3, b0, b1);
    } else {
        mma_f16(c[0], c[1], c[2], c[3], a0, a1, a2, a3, b0, b1);
    }
}

template <int Group, KvCacheStorage Storage, int Shift>
__global__ void __launch_bounds__(kTcThreads)
    sparse_attention_tc_kernel(const __nv_bfloat16* __restrict__ q, const int* __restrict__ first,
                               const int* __restrict__ selected, const int* __restrict__ counts,
                               KvPlanes planes, const std::int32_t* __restrict__ block_table,
                               float scale, __nv_bfloat16* __restrict__ out, int q_heads) {
    using Element = TileElement<Storage>;
    constexpr bool kBf16   = std::is_same_v<Element, __nv_bfloat16>;
    constexpr bool kPacked = Storage == KvCacheStorage::Bf16Packed;
    constexpr bool kCopied = Storage == KvCacheStorage::BFloat16 || Storage == KvCacheStorage::Float16;
    constexpr bool kQuantized = !kCopied && !kPacked;
    constexpr bool kRotatedValues =
        Storage == KvCacheStorage::Nvfp4Group16 || Storage == KvCacheStorage::Fp8KeyNvfp4Value;
    // FP16 tiles take the query and the probabilities in two parts, the second scaled by 2^12.
    constexpr bool kTwoPart  = !kBf16;
    constexpr float kPScale  = kTwoPart ? kTcRemainder : 1.0f;
    using Layout             = TcLayout<kTwoPart>;
    static_assert(Group <= kTcRows && Group * kHeadDim * 4 <= 2 * Layout::kTileBytes,
                  "the group's query rows fit the tile, and its rotated output rows the two tiles");
    extern __shared__ __align__(16) unsigned char tc_shared[];
    auto* queries = reinterpret_cast<Element*>(tc_shared);
    auto* keys    = reinterpret_cast<Element*>(tc_shared + Layout::kQueryBytes);
    auto* values  = keys + kTcTile * kTcStride;
    auto* probs   = values + kTcTile * kTcStride; // [part][row][key]
    constexpr int kPart = kTcRows * kTcPStride; // elements between the probabilities' parts
    auto* maxima  = reinterpret_cast<float*>(probs + Layout::kPParts * kTcRows * kTcPStride);
    auto* sums    = maxima + kTcWarps * kTcRows;

    const int t          = blockIdx.x;
    const int kv_head    = blockIdx.y;
    const int warp       = threadIdx.x >> 5;
    const int lane       = threadIdx.x & 31;
    const int g          = lane >> 2;
    const int tq         = lane & 3;
    const int position   = *first + t;
    const int count      = counts[t];
    const int tail_begin = (position + 1) / 4 * 4;
    const int total      = 4 * count + (position + 1 - tail_begin);
    const int* blocks    = selected + static_cast<std::int64_t>(t) * kTopBlocks;
    const int tiles      = (total + kTcTile - 1) / kTcTile;

    // Tile c's K (Value false) or V rows into their buffer: row i is position c * 32 + i, zero past
    // the list's end.
    const auto stage = [&](int c, bool value) {
        Element* tile = value ? values : keys;
        if constexpr (kCopied) {
            // Warp w copies rows w, w + 4, ...; lane l the row's 16-byte piece l.
            const auto* pages = static_cast<const Element*>(value ? planes.v : planes.k);
#pragma unroll
            for (int row = warp; row < kTcTile; row += kTcWarps) {
                const int index = c * kTcTile + row;
                Element* dst    = tile + row * kTcStride + lane * 8;
                if (index < total) {
                    const int p = position_at(blocks, count, tail_begin, index);
                    cp_async<16, Cache::cg>(
                        dst, pages + plane_offset<kHeadDim, Shift>(block_table, planes.heads,
                                                                   kv_head, p, lane * 8));
                } else {
                    cp_async_zfill<16, Cache::cg>(dst, pages, 0);
                }
            }
        } else if constexpr (kPacked) {
            const auto* rows  = static_cast<const std::uint8_t*>(value ? planes.v : planes.k);
            const auto* sides = static_cast<const std::uint8_t*>(value ? planes.v_scale : planes.k_scale);
#pragma unroll
            for (int row = warp; row < kTcTile; row += kTcWarps) {
                const int index = c * kTcTile + row;
                uint4 words     = make_uint4(0, 0, 0, 0);
                if (index < total) {
                    const int p = position_at(blocks, count, tail_begin, index);
                    words       = bf16p_decode8(
                        rows + plane_offset<kBf16pRowBytes, Shift>(block_table, planes.heads,
                                                                   kv_head, p, 0),
                        sides + plane_offset<kBf16pSideBytes, Shift>(block_table, planes.heads,
                                                                     kv_head, p, 0),
                        lane);
                }
                *reinterpret_cast<uint4*>(tile + row * kTcStride + lane * 8) = words;
            }
        } else {
#pragma unroll
            for (int i = threadIdx.x; i < kTcTile * kChunks; i += kTcThreads) {
                const int row = i / kChunks, chunk = i % kChunks;
                const int index = c * kTcTile + row;
                unsigned w[kWordsPerChunk] = {};
                if (index < total) {
                    const int p = position_at(blocks, count, tail_begin, index);
                    if (value) {
                        decode_value<Storage, Shift>(planes, block_table, kv_head, p, chunk, w);
                    } else {
                        decode_key<Storage, Shift>(planes, block_table, kv_head, p, chunk, w);
                    }
                }
                auto* dst = reinterpret_cast<uint4*>(tile + row * kTcStride + chunk * kChunk);
                dst[0]    = make_uint4(w[0], w[1], w[2], w[3]);
                dst[1]    = make_uint4(w[4], w[5], w[6], w[7]);
            }
        }
    };

    stage(0, false);
    cp_commit();
    stage(0, true);
    cp_commit();
    // The query rows: BF16 as given, or two FP16 parts of the (rotated) query.
    if constexpr (!kTwoPart) {
        for (int i = threadIdx.x; i < kTcRows * (kHeadDim / 8); i += kTcThreads) {
            const int row = i / (kHeadDim / 8), piece = i % (kHeadDim / 8);
            uint4 words   = make_uint4(0, 0, 0, 0);
            if (row < Group) {
                words = *reinterpret_cast<const uint4*>(
                    q +
                    (static_cast<std::int64_t>(t) * q_heads + kv_head * Group + row) *
                        kHeadDim +
                    piece * 8);
            }
            *reinterpret_cast<uint4*>(queries + row * kTcStride + piece * 8) = words;
        }
    } else {
        for (int row = warp; row < kTcRows; row += kTcWarps) {
            float values_of_row[kHeadDim / 32];
#pragma unroll
            for (int r = 0; r < kHeadDim / 32; ++r) {
                values_of_row[r] =
                    row < Group
                        ? __bfloat162float(q[(static_cast<std::int64_t>(t) * q_heads +
                                              kv_head * Group + row) *
                                                 kHeadDim +
                                             lane + 32 * r])
                        : 0.0f;
            }
            if constexpr (kQuantized) { normalized_hadamard_d256_inplace(values_of_row, lane); }
#pragma unroll
            for (int r = 0; r < kHeadDim / 32; ++r) {
                const __half high = __float2half_rn(values_of_row[r]);
                const __half low =
                    __float2half_rn((values_of_row[r] - __half2float(high)) * kTcRemainder);
                queries[row * kTcStride + lane + 32 * r]                       = high;
                queries[(kTcRows + row) * kTcStride + lane + 32 * r] = low;
            }
        }
    }

    float running_max[2] = {-INFINITY, -INFINITY};
    float running_sum[2] = {0.0f, 0.0f}; // this warp's keys only
    float acc[8][4]      = {};
    float acc_low[kTwoPart ? 8 : 1][4] = {};
    const unsigned query_address = smem_addr(queries + (lane & 15) * kTcStride + (lane >> 4) * 8);
    const unsigned key_address =
        smem_addr(keys + (8 * warp + (lane & 7)) * kTcStride + ((lane >> 3) & 1) * 8);
    const unsigned prob_address = smem_addr(probs + (lane & 15) * kTcPStride + (lane >> 4) * 8);
    const unsigned value_address = smem_addr(
        values + ((lane & 7) + ((lane >> 3) & 1) * 8) * kTcStride + 64 * warp + (lane >> 4) * 8);
    for (int c = 0; c < tiles; ++c) {
        cp_wait<1>();
        __syncthreads(); // K(c) has landed, and the query rows with the first
        float score[4] = {}, score_low[4] = {};
#pragma unroll
        for (int k = 0; k < kHeadDim / 16; ++k) {
            unsigned a0, a1, a2, a3, b0, b1;
            ldmatrix_x2(b0, b1, key_address + 2 * 16 * k);
            ldmatrix_x4(a0, a1, a2, a3, query_address + 2 * 16 * k);
            tc_mma<Element>(score, a0, a1, a2, a3, b0, b1);
            if constexpr (kTwoPart) {
                ldmatrix_x4(a0, a1, a2, a3,
                            query_address + 2 * (kTcRows * kTcStride + 16 * k));
                tc_mma<Element>(score_low, a0, a1, a2, a3, b0, b1);
            }
        }
        const int key0 = c * kTcTile + 8 * warp + 2 * tq;
        float row_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float dot = kTwoPart ? score[i] + score_low[i] * (1.0f / kTcRemainder) : score[i];
            score[i]        = key0 + (i & 1) < total ? dot * scale : -INFINITY;
            row_max[i >> 1] = fmaxf(row_max[i >> 1], score[i]);
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            row_max[r] = fmaxf(row_max[r], __shfl_xor_sync(0xffffffffu, row_max[r], 1));
            row_max[r] = fmaxf(row_max[r], __shfl_xor_sync(0xffffffffu, row_max[r], 2));
        }
        if (tq == 0) {
            maxima[warp * kTcRows + g]     = row_max[0];
            maxima[warp * kTcRows + g + 8] = row_max[1];
        }
        __syncthreads(); // every warp has scored K(c) and shared its maxima
        if (c + 1 < tiles) { stage(c + 1, false); }
        cp_commit();
        float correction[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float tile_max = maxima[g + 8 * r];
#pragma unroll
            for (int w = 1; w < kTcWarps; ++w) {
                tile_max = fmaxf(tile_max, maxima[w * kTcRows + g + 8 * r]);
            }
            const float new_max = fmaxf(running_max[r], tile_max);
            correction[r]       = __expf(running_max[r] - new_max);
            running_max[r]      = new_max;
        }
        float p[4], weight_sum[2] = {0.0f, 0.0f};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            p[i] = key0 + (i & 1) < total ? __expf(score[i] - running_max[i >> 1]) : 0.0f;
            weight_sum[i >> 1] += p[i];
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            weight_sum[r] += __shfl_xor_sync(0xffffffffu, weight_sum[r], 1);
            weight_sum[r] += __shfl_xor_sync(0xffffffffu, weight_sum[r], 2);
            running_sum[r] = running_sum[r] * correction[r] + weight_sum[r];
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            // FP16: p and its remainder times 2^12 (22 bits). BF16: p, its remainder and the
            // remainder's (24 bits, all of p's).
            const unsigned high = pack_pair<Element>(p[2 * r], p[2 * r + 1]);
            const float2 rounded = unpack_pair<Element>(high);
            const float rest[2] = {p[2 * r] - rounded.x, p[2 * r + 1] - rounded.y};
            const unsigned low  = pack_pair<Element>(rest[0] * kPScale, rest[1] * kPScale);
            const int at        = (g + 8 * r) * kTcPStride + 8 * warp + 2 * tq;
            *reinterpret_cast<unsigned*>(probs + at)         = high;
            *reinterpret_cast<unsigned*>(probs + kPart + at) = low;
            if constexpr (!kTwoPart) {
                const float2 middle = unpack_pair<Element>(low);
                *reinterpret_cast<unsigned*>(probs + 2 * kPart + at) =
                    pack_pair<Element>(rest[0] - middle.x, rest[1] - middle.y);
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if constexpr (kTwoPart) { acc_low[j][i] *= correction[i >> 1]; }
            }
        }
        cp_wait<1>();
        __syncthreads(); // V(c) has landed and P is complete
        float tile[8][4] = {};
#pragma unroll
        for (int k = 0; k < kTcTile / 16; ++k) {
            unsigned h0, h1, h2, h3, l0, l1, l2, l3, n0 = 0, n1 = 0, n2 = 0, n3 = 0;
            ldmatrix_x4(h0, h1, h2, h3, prob_address + 2 * 16 * k);
            ldmatrix_x4(l0, l1, l2, l3, prob_address + 2 * (kPart + 16 * k));
            if constexpr (!kTwoPart) {
                ldmatrix_x4(n0, n1, n2, n3, prob_address + 2 * (2 * kPart + 16 * k));
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                unsigned b0, b1, b2, b3;
                ldmatrix_x4_t(b0, b1, b2, b3, value_address + 2 * (16 * k * kTcStride + 16 * j));
                tc_mma<Element>(tile[2 * j], h0, h1, h2, h3, b0, b1);
                tc_mma<Element>(tile[2 * j + 1], h0, h1, h2, h3, b2, b3);
                if constexpr (kTwoPart) {
                    tc_mma<Element>(acc_low[2 * j], l0, l1, l2, l3, b0, b1);
                    tc_mma<Element>(acc_low[2 * j + 1], l0, l1, l2, l3, b2, b3);
                } else {
                    tc_mma<Element>(tile[2 * j], l0, l1, l2, l3, b0, b1);
                    tc_mma<Element>(tile[2 * j + 1], l0, l1, l2, l3, b2, b3);
                    tc_mma<Element>(tile[2 * j], n0, n1, n2, n3, b0, b1);
                    tc_mma<Element>(tile[2 * j + 1], n0, n1, n2, n3, b2, b3);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[j][i] = __fadd_rn(__fmul_rn(acc[j][i], correction[i >> 1]), tile[j][i]);
            }
        }
        __syncthreads(); // every warp has read V(c), P and the maxima
        if (c + 1 < tiles) { stage(c + 1, true); }
        cp_commit();
    }
    cp_wait<0>();
    if (tq == 0) {
        sums[warp * kTcRows + g]     = running_sum[0];
        sums[warp * kTcRows + g + 8] = running_sum[1];
    }
    if constexpr (kTwoPart) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[j][i] += acc_low[j][i] * (1.0f / kTcRemainder); }
        }
    }
    const auto inverse_of = [&](int row) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < kTcWarps; ++w) { sum += sums[w * kTcRows + row]; }
        return sum > 0.0f ? 1.0f / sum : 0.0f;
    };
    const std::int64_t out_row0 = static_cast<std::int64_t>(t) * q_heads + kv_head * Group;
    if constexpr (kRotatedValues) {
        // The values were stored rotated, so the reduction is too: the group's rows meet in shared
        // memory (over the tiles) and the warps take them back through the transform in turn.
        auto* rows_out = reinterpret_cast<float*>(keys);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int d = 64 * warp + 8 * j + 2 * tq;
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int row = g + 8 * r;
                if (row < Group) {
                    rows_out[row * kHeadDim + d]     = acc[j][2 * r];
                    rows_out[row * kHeadDim + d + 1] = acc[j][2 * r + 1];
                }
            }
        }
        __syncthreads();
        for (int row = warp; row < Group; row += kTcWarps) {
            float values_of_row[kHeadDim / 32];
#pragma unroll
            for (int r = 0; r < kHeadDim / 32; ++r) {
                values_of_row[r] = rows_out[row * kHeadDim + lane + 32 * r];
            }
            normalized_hadamard_d256_inplace(values_of_row, lane);
            const float inverse = inverse_of(row);
#pragma unroll
            for (int r = 0; r < kHeadDim / 32; ++r) {
                out[(out_row0 + row) * kHeadDim + lane + 32 * r] =
                    __float2bfloat16_rn(values_of_row[r] * inverse);
            }
        }
    } else {
        __syncthreads();
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = g + 8 * r;
            if (row >= Group) { continue; }
            const float inverse = inverse_of(row);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int d = 64 * warp + 8 * j + 2 * tq;
                *reinterpret_cast<__nv_bfloat162*>(out + (out_row0 + row) * kHeadDim + d) =
                    __floats2bfloat162_rn(acc[j][2 * r] * inverse, acc[j][2 * r + 1] * inverse);
            }
        }
    }
}

// One CTA per (query, query head), one thread per output dim: the splits' partial softmaxes
// rescaled to the common maximum. A split with no positions has a zero sum and adds nothing.
// The call's query heads are a template argument: with them at run time, nvcc 12.8 (sm_70 and
// sm_86) addressed each split's maximum at slot[2 * kHeadDim + 1] instead of slot[kHeadDim].
template <int QHeads>
__global__ void __launch_bounds__(kHeadDim)
    sparse_attention_merge_kernel(const float* __restrict__ partial,
                                  __nv_bfloat16* __restrict__ out) {
    const int t = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const float* first_split =
        partial + (static_cast<std::int64_t>(t) * kSplits * QHeads + head) * kPartialFloats;
    constexpr std::int64_t kStride = std::int64_t(QHeads) * kPartialFloats;
    float maximum                  = -INFINITY;
    for (int s = 0; s < kSplits; ++s) {
        const float* slot = first_split + s * kStride;
        if (slot[kHeadDim + 1] > 0.0f) maximum = fmaxf(maximum, slot[kHeadDim]);
    }
    float sum = 0.0f, value = 0.0f;
    for (int s = 0; s < kSplits; ++s) {
        const float* slot = first_split + s * kStride;
        if (slot[kHeadDim + 1] > 0.0f) {
            const float weight = __expf(slot[kHeadDim] - maximum);
            sum += slot[kHeadDim + 1] * weight;
            value += slot[d] * weight;
        }
    }
    out[(static_cast<std::int64_t>(t) * QHeads + head) * kHeadDim + d] =
        __float2bfloat16_rn(sum > 0.0f ? value / sum : 0.0f);
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(std::string("sparse_softmax_attention: ") + message);
    }
}

// One plane of the cache against the storage's layout: its dtype, row width and page size.
void require_plane(const Tensor& plane, DType dtype, std::int32_t leading,
                   std::int32_t page_tokens, std::int32_t heads, const char* message) {
    require(plane.data != nullptr && plane.dtype == dtype && plane.is_contiguous() &&
                plane.ne[0] == leading && plane.ne[1] == page_tokens && plane.ne[2] == heads,
            message);
}

// Wider calls run on tensor cores where they measured faster than the SIMT kernel for the call's
// query heads per KV head, the device's tier and the cache's storage, at 1,024 queries over 2,051
// positions.
template <int Group, KvCacheStorage Storage>
bool tensor_cores_faster(DeviceTier tier) {
    if constexpr (Group == kWideGroup) {
        // Twelve query heads (Qwen3.8-Flash-Next, dense Qwen4 with two KV heads): an RTX PRO 4500
        // Blackwell ran every storage faster here (1.07-3.15 times). An RTX 3090 ran BF16 2.76
        // times, FP16 1.67, bf16p 1.18 and the 4-bit keys (nvfp4, rk4v4, rk4v4-e8, rk2v4-e8)
        // 1.04-1.07 times, but the 8-bit keys (int8, rk8v4, fp8, k8v4) at 0.94-0.99, whose decode
        // the four warps here hide worse than the SIMT kernel's twelve. An RTX 2080 Ti ran the
        // FP16 tiles (FP16 and every decoded storage) 1.21-1.73 times, but BF16 and bf16p at 0.32
        // and 0.20 on its emulated BF16 MMAs. Volta emulates every MMA.
        constexpr bool kBf16Tiles = std::is_same_v<TileElement<Storage>, __nv_bfloat16>;
        constexpr bool kByteKeys  = Storage == KvCacheStorage::Int8Group64 ||
                                   Storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64 ||
                                   Storage == KvCacheStorage::Fp8E4M3Row256 ||
                                   Storage == KvCacheStorage::Fp8KeyNvfp4Value;
        switch (tier) {
        case DeviceTier::Blackwell: return true;
        case DeviceTier::Ampere: return !kByteKeys;
        case DeviceTier::Turing: return !kBf16Tiles;
        case DeviceTier::Volta: return false;
        }
        return false;
    } else {
        // Six query heads (dense Qwen4 with four KV heads): the 16-row tile holds six queries, so
        // each staged K and V tile serves half as many rows, while the SIMT kernel's work follows
        // the query heads. An RTX 3090 ran BF16 2.54 times, FP16 1.51 and bf16p 1.09 here, every
        // other storage 0.79-0.86; an RTX 5080 ran BF16 2.83 and FP16 1.46, bf16p 0.98 and the
        // others 0.66-0.72. Turing (not measured for six) and Volta keep the SIMT kernel.
        constexpr bool kHalfTiles =
            Storage == KvCacheStorage::BFloat16 || Storage == KvCacheStorage::Float16;
        switch (tier) {
        case DeviceTier::Blackwell: return kHalfTiles;
        case DeviceTier::Ampere: return kHalfTiles || Storage == KvCacheStorage::Bf16Packed;
        case DeviceTier::Turing:
        case DeviceTier::Volta: return false;
        }
        return false;
    }
}

template <int Group, KvCacheStorage Storage, int Shift>
bool launch_tensor_cores(const __nv_bfloat16* q, const int* first, const int* selected,
                         const int* counts, const KvPlanes& planes, const std::int32_t* table,
                         float scale, std::int32_t tokens, __nv_bfloat16* out, cudaStream_t stream) {
    if (!tensor_cores_faster<Group, Storage>(current_device_tier())) { return false; }
    constexpr int kBytes = TcLayout<!std::is_same_v<TileElement<Storage>, __nv_bfloat16>>::kBytes;
    constexpr int kDevices = 64;
    static std::array<std::once_flag, kDevices> opted;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    require(device >= 0 && device < kDevices, "device index out of range");
    std::call_once(opted[static_cast<std::size_t>(device)], [] {
        CUDA_CHECK(cudaFuncSetAttribute(sparse_attention_tc_kernel<Group, Storage, Shift>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes));
    });
    sparse_attention_tc_kernel<Group, Storage, Shift>
        <<<dim3(tokens, planes.heads), kTcThreads, kBytes, stream>>>(
            q, first, selected, counts, planes, table, scale, out, planes.heads * Group);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

template <int Group, KvCacheStorage Storage, int Shift>
void launch_group(const __nv_bfloat16* q, const int* first, const int* selected,
                  const int* counts, const KvPlanes& planes, const std::int32_t* table,
                  float scale, std::int32_t tokens, float* partial, __nv_bfloat16* out,
                  cudaStream_t stream) {
    const int q_heads = planes.heads * Group;
    if (partial == nullptr) {
        if (launch_tensor_cores<Group, Storage, Shift>(q, first, selected, counts, planes, table,
                                                       scale, tokens, out, stream)) {
            return;
        }
        sparse_attention_kernel<Group, false, Storage, Shift>
            <<<dim3(tokens, planes.heads), Group * 32, 0, stream>>>(
                q, first, selected, counts, planes, table, scale, out, nullptr, q_heads);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    sparse_attention_kernel<Group, true, Storage, Shift>
        <<<dim3(tokens, planes.heads, kSplits), Group * 32, 0, stream>>>(
            q, first, selected, counts, planes, table, scale, out, partial, q_heads);
    CUDA_CHECK(cudaGetLastError());
    const auto merge = q_heads == 6    ? sparse_attention_merge_kernel<6>
                       : q_heads == 12 ? sparse_attention_merge_kernel<12>
                       : q_heads == 18 ? sparse_attention_merge_kernel<18>
                                       : sparse_attention_merge_kernel<kQueryHeads>;
    merge<<<dim3(tokens, q_heads), kHeadDim, 0, stream>>>(partial, out);
    CUDA_CHECK(cudaGetLastError());
}

template <KvCacheStorage Storage, int Shift>
void launch(int group, const __nv_bfloat16* q, const int* first, const int* selected,
            const int* counts, const KvPlanes& planes, const std::int32_t* table, float scale,
            std::int32_t tokens, float* partial, __nv_bfloat16* out, cudaStream_t stream) {
    (group == kWideGroup ? launch_group<kWideGroup, Storage, Shift>
                         : launch_group<kNarrowGroup, Storage, Shift>)(
        q, first, selected, counts, planes, table, scale, tokens, partial, out, stream);
}

} // namespace

std::size_t sparse_softmax_attention_workspace_bytes(std::int32_t tokens) {
    require(tokens > 0, "tokens must be positive");
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP32,
                       {kPartialFloats, kQueryHeads, kSplits, std::min(tokens, kSplitTokens)});
    return layout.peak_bytes(1);
}

void sparse_softmax_attention(const Tensor& q, const Tensor& first_position, const Tensor& selected,
                              const Tensor& counts, const PagedKVLayerView& cache, float scale,
                              WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    require(cache.head_dim == kHeadDim && cache.num_kv_heads >= 1 &&
                cache.block_table.dtype == DType::I32 && cache.block_table.data != nullptr,
            "cache must be a paged layer of KV heads of 256");
    const std::int32_t heads   = cache.num_kv_heads;
    const std::int32_t q_heads = q.ne[1];
    // The query heads per KV head: 12 for two KV heads in all, 6 for four.
    const std::int32_t group = heads > 0 && q_heads % heads == 0 ? q_heads / heads : 0;
    require(q.dtype == DType::BF16 && q.is_contiguous() && q.data != nullptr &&
                q.ne[0] == kHeadDim && (group == kWideGroup || group == kNarrowGroup) &&
                q_heads <= kQueryHeads && q.ne[2] > 0 && q.ne[3] == 1,
            "q must be contiguous BF16 [256, 12 or 6 per KV head, tokens], at most 24 heads");
    const std::int32_t tokens = q.ne[2];
    require(first_position.dtype == DType::I32 && first_position.data != nullptr &&
                first_position.numel() >= 1,
            "first_position must be a device I32 word");
    require(selected.dtype == DType::I32 && selected.is_contiguous() && selected.data != nullptr &&
                selected.ne[0] == kTopBlocks && selected.ne[1] == tokens,
            "selected must be contiguous I32 [512, tokens]");
    require(counts.dtype == DType::I32 && counts.is_contiguous() && counts.data != nullptr &&
                counts.ne[0] == tokens,
            "counts must be contiguous I32 [tokens]");
    require(out.dtype == DType::BF16 && out.is_contiguous() && out.data != nullptr &&
                out.ne[0] == kHeadDim && out.ne[1] == q_heads && out.ne[2] == tokens,
            "out must be contiguous BF16 [256, q heads, tokens]");
    const PagedKVStorageLayout layout = paged_kv_storage_layout(cache.storage, kHeadDim);
    const std::int32_t page_tokens   = cache.k_pages.ne[1];
    require(page_tokens == kPagedKVPageSize || page_tokens == kKvStreamPageTokens,
            "pages must hold 64 positions, or 4 in a streamed layer's pool");
    require_plane(cache.k_pages, layout.key.data_dtype, layout.key.data_leading_extent,
                  page_tokens, heads, "the key plane does not match the cache storage");
    require_plane(cache.v_pages, layout.value.data_dtype, layout.value.data_leading_extent,
                  page_tokens, heads, "the value plane does not match the cache storage");
    if (layout.key.has_scale()) {
        require_plane(cache.k_scale_pages, layout.key.scale_dtype, layout.key.scale_leading_extent,
                      page_tokens, heads, "the key scale plane does not match the cache storage");
    }
    if (layout.value.has_scale()) {
        require_plane(cache.v_scale_pages, layout.value.scale_dtype,
                      layout.value.scale_leading_extent, page_tokens, heads,
                      "the value scale plane does not match the cache storage");
    }
    const bool streamed = page_tokens == kKvStreamPageTokens;
    require(std::isfinite(scale) && scale > 0.0f, "scale must be positive and finite");
    const auto* q_p      = static_cast<const __nv_bfloat16*>(q.data);
    const auto* first_p  = static_cast<const int*>(first_position.data);
    const auto* select_p = static_cast<const int*>(selected.data);
    const auto* counts_p = static_cast<const int*>(counts.data);
    const auto* table_p  = static_cast<const std::int32_t*>(cache.block_table.data);
    auto* out_p          = static_cast<__nv_bfloat16*>(out.data);
    const KvPlanes planes{.k       = cache.k_pages.data,
                          .v       = cache.v_pages.data,
                          .k_scale = layout.key.has_scale() ? cache.k_scale_pages.data : nullptr,
                          .v_scale = layout.value.has_scale() ? cache.v_scale_pages.data : nullptr,
                          .heads   = heads};
    auto scope       = workspace.scope();
    float* partial_p = nullptr;
    if (tokens <= kSplitTokens) {
        Tensor partial =
            workspace.alloc(DType::FP32, {kPartialFloats, q_heads, kSplits, tokens});
        partial_p = static_cast<float*>(partial.data);
    }
    switch (cache.storage) {
#define NINFER_SPARSE_ATTENTION_CASE(STORAGE)                                                      \
    case KvCacheStorage::STORAGE:                                                                  \
        (streamed ? launch<KvCacheStorage::STORAGE, 2> : launch<KvCacheStorage::STORAGE, 6>)(      \
            group, q_p, first_p, select_p, counts_p, planes, table_p, scale, tokens, partial_p,    \
            out_p, stream);                                                                        \
        return;
        NINFER_SPARSE_ATTENTION_CASE(BFloat16)
        NINFER_SPARSE_ATTENTION_CASE(Float16)
        NINFER_SPARSE_ATTENTION_CASE(Int8Group64)
        NINFER_SPARSE_ATTENTION_CASE(Fp8E4M3Row256)
        NINFER_SPARSE_ATTENTION_CASE(RotatedInt8KeyInt4ValueGroup64)
        NINFER_SPARSE_ATTENTION_CASE(Nvfp4Group16)
        NINFER_SPARSE_ATTENTION_CASE(Fp8KeyNvfp4Value)
        NINFER_SPARSE_ATTENTION_CASE(RotatedLloyd4KeyInt4Value)
        NINFER_SPARSE_ATTENTION_CASE(RotatedInt4KeyInt4ValueE8)
        NINFER_SPARSE_ATTENTION_CASE(RotatedE8RootKeyInt4Value)
        NINFER_SPARSE_ATTENTION_CASE(Bf16Packed)
#undef NINFER_SPARSE_ATTENTION_CASE
    case KvCacheStorage::Mxfp8: // Inkling 2's own cache
        break;
    }
    require(false, "unsupported cache storage");
}

} // namespace ninfer::ops
