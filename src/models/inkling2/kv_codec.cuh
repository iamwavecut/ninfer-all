#pragma once

// Device side of Inkling 2's quantized KV planes (kv_format.h): a warp encoding or decoding one
// 128-wide row (one KV head of one position), and the eight-value chunk reads the attention's tile
// loader makes. The codings are the engine's codecs (ops/kv_cache/int8_g64_codec.cuh); only the
// addressing is Inkling's own: a row's codes and scales live in separate planes, and chunk c
// (dimensions [8c, 8c + 8)) of a row has its codes at byte c * bytes-per-chunk and its scale at
// index c / chunks-per-group of the row's scales.

#include "models/inkling2/kv_format.h"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::models::inkling2::kv {

// MXFP8, the release's rule (vLLM's mxfp8_quantize for Inkling 2): a group of 32 values has the
// scale 2^(e - 127), e the exponent of amax / 448 rounded up to a power of two (0 when amax / 448
// is at most 2^-127, at most 254, 255 for NaN); a value's code is x * 2^(127 - e) clamped to
// +-448 and rounded to E4M3, nearest even. Inkling's keys are clipped to qk_clip and its values
// RMS-normed, so their exponents stay far below 246, the largest the decode below represents.
__device__ __forceinline__ unsigned mxfp8_exponent(float amax) {
    const float raw = amax * (1.0f / 448.0f);
    if (raw != raw) { return 255u; }
    if (raw <= 0x1p-127f) { return 0u; }
    const unsigned e = ((__float_as_uint(raw) + 0x007FFFFFu) & 0x7F800000u) >> 23;
    return min(e, 254u);
}

// The code of x in a group of exponent e.
__device__ __forceinline__ std::uint8_t mxfp8_code(float x, unsigned e) {
    // 2^(127 - e): e 0 gives 2^127, as the rule's inverse does; NaN groups encode zeros times NaN.
    const float inverse = e == 255u ? 0.0f : __uint_as_float((254u - e) << 23);
    const float v       = fminf(fmaxf(x * inverse, -448.0f), 448.0f);
    return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

// The group's scale times 2^8, for codes read as FP16 times 2^-8 (ops' x2m8 expansion): exact
// for e up to 246; 255 is NaN.
__device__ __forceinline__ float mxfp8_scale_x256(unsigned e) {
    return e == 255u ? __uint_as_float(0x7FC00000u) : __uint_as_float((e + 8u) << 23);
}


// Chunks of eight values: eight int8 codes (one uint2), eight packed nibbles or eight Lloyd-Max
// indices (one word). Per coding: Raw, the bytes of one chunk; Scale, the chunk's group scale;
// load(), the chunk's bytes from its row; scale(), the chunk's scale; decode(), the eight values.
template <KvCoding C>
struct Chunk;

template <>
struct Chunk<KvCoding::Int8G64> {
    using Raw = uint2;
    using Scale = __half;
    static constexpr int kBytes = 8, kPerGroup = 8; // 64-dimension groups
    __device__ static Raw load(const std::uint8_t* codes, int c) {
        return *reinterpret_cast<const uint2*>(codes + c * kBytes);
    }
    __device__ static Scale scale(const std::uint8_t* scales, int c) {
        return reinterpret_cast<const __half*>(scales)[c / kPerGroup];
    }
    __device__ static void decode(Raw r, Scale s, float (&f)[8]) {
        const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&r);
        const float sf       = __half2float(s);
#pragma unroll
        for (int j = 0; j < 8; ++j) { f[j] = float(c[j]) * sf; }
    }
};

template <>
struct Chunk<KvCoding::Int4G32> {
    using Raw = unsigned;
    using Scale = __half;
    static constexpr int kBytes = 4, kPerGroup = 4; // 32-dimension groups
    __device__ static Raw load(const std::uint8_t* codes, int c) {
        return *reinterpret_cast<const unsigned*>(codes + c * kBytes);
    }
    __device__ static Scale scale(const std::uint8_t* scales, int c) {
        return reinterpret_cast<const __half*>(scales)[c / kPerGroup];
    }
    __device__ static void decode(Raw r, Scale s, float (&f)[8]) {
        const float sf = __half2float(s);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int nibble = int((r >> (4 * j)) & 0xfu);
            f[j]             = float((nibble ^ 8) - 8) * sf;
        }
    }
};

template <>
struct Chunk<KvCoding::Lloyd4> {
    using Raw = unsigned;
    using Scale = __half;
    static constexpr int kBytes = 4, kPerGroup = 8; // 64-dimension groups
    __device__ static Raw load(const std::uint8_t* codes, int c) {
        return *reinterpret_cast<const unsigned*>(codes + c * kBytes);
    }
    __device__ static Scale scale(const std::uint8_t* scales, int c) {
        return reinterpret_cast<const __half*>(scales)[c / kPerGroup];
    }
    __device__ static void decode(Raw r, Scale s, float (&f)[8]) {
        const uint2 codes = make_uint2(ops::kv_cache_lloyd4_expand4(r & 0xffffu),
                                       ops::kv_cache_lloyd4_expand4(r >> 16));
        const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&codes);
        const float sf       = __half2float(s);
#pragma unroll
        for (int j = 0; j < 8; ++j) { f[j] = float(c[j]) * sf; }
    }
};

template <>
struct Chunk<KvCoding::Mxfp8> {
    using Raw = uint2;
    using Scale = std::uint8_t;
    static constexpr int kBytes = 8, kPerGroup = 4; // 32-dimension groups
    __device__ static Raw load(const std::uint8_t* codes, int c) {
        return *reinterpret_cast<const uint2*>(codes + c * kBytes);
    }
    __device__ static Scale scale(const std::uint8_t* scales, int c) { return scales[c / kPerGroup]; }
    __device__ static void decode(Raw r, Scale s, float (&f)[8]) {
        // Codes times 2^-8 as FP16 pairs, then times the group's scale times 2^8: exact in FP32.
        const float sf = mxfp8_scale_x256(s);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned word = j < 2 ? r.x : r.y;
            const float2 pair   = __half22float2(
                ops::kv_cache_fp8_code2_to_half2_x2m8(word >> (16 * (j & 1))));
            f[2 * j]     = pair.x * sf;
            f[2 * j + 1] = pair.y * sf;
        }
    }
};

// One 128-wide row per warp: lane l holds dimensions l + 32 r of the row in x[r] (the layout
// kv_post writes). Encodes x into the row's codes and scales; reading rows back happens in the
// attention's tile loader, which decodes eight-value chunks (Chunk above) as it needs them.
template <KvCoding C>
__device__ __forceinline__ void encode_row(const float (&x)[4], std::uint8_t* codes,
                                           std::uint8_t* scales, int lane) {
    constexpr unsigned kAll = 0xffffffffu;
    if constexpr (C == KvCoding::Int8G64) {
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const float a = ops::warp_max(fmaxf(fabsf(x[2 * g]), fabsf(x[2 * g + 1])), kAll);
            const auto q  = ops::kv_cache_int8_quant_params(a);
            auto* c       = reinterpret_cast<std::int8_t*>(codes);
            c[64 * g + lane]      = ops::kv_cache_int8_quant_code(x[2 * g], q.inverse_scale);
            c[64 * g + 32 + lane] = ops::kv_cache_int8_quant_code(x[2 * g + 1], q.inverse_scale);
            if (lane == 0) { reinterpret_cast<__half*>(scales)[g] = q.scale; }
        }
    } else if constexpr (C == KvCoding::Int4G32) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const auto q        = ops::kv_cache_int4_quant_params(ops::warp_max(fabsf(x[r]), kAll));
            const std::int8_t c = ops::kv_cache_int4_quant_code(x[r], q.inverse_scale);
            const int partner   = __shfl_xor_sync(kAll, int(c), 1);
            if ((lane & 1) == 0) {
                codes[16 * r + (lane >> 1)] = ops::kv_cache_int4_pack(c, std::int8_t(partner));
            }
            if (lane == 0) { reinterpret_cast<__half*>(scales)[r] = q.scale; }
        }
    } else if constexpr (C == KvCoding::Mxfp8) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const unsigned e = mxfp8_exponent(ops::warp_max(fabsf(x[r]), kAll));
            codes[32 * r + lane] = mxfp8_code(x[r], e);
            if (lane == 0) { scales[r] = std::uint8_t(e); }
        }
    } else if constexpr (C == KvCoding::Lloyd4) {
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const auto e       = ops::kv_cache_lloyd4_encode_group(x[2 * g], x[2 * g + 1]);
            const int partner0 = __shfl_xor_sync(kAll, e.index0, 1);
            const int partner1 = __shfl_xor_sync(kAll, e.index1, 1);
            if ((lane & 1) == 0) {
                codes[32 * g + (lane >> 1)]      = std::uint8_t(e.index0 | (partner0 << 4));
                codes[32 * g + 16 + (lane >> 1)] = std::uint8_t(e.index1 | (partner1 << 4));
            }
            if (lane == 0) { reinterpret_cast<__half*>(scales)[g] = e.scale; }
        }
    }
}

} // namespace ninfer::models::inkling2::kv

namespace ninfer::models::inkling2 {

// Eight values of chunk `c` of the row whose codes and scales begin at `codes` / `scales`, as one
// 16-byte word of halves: the attention's tile loader (kernels.cu). F16 never reaches here.
__device__ __forceinline__ uint4 chunk_word(const KvPlaneView& plane, const std::uint8_t* codes,
                                            const std::uint8_t* scales, int c) {
    float f[8] = {};
    switch (plane.coding) {
    case KvCoding::Int8G64:
        kv::Chunk<KvCoding::Int8G64>::decode(kv::Chunk<KvCoding::Int8G64>::load(codes, c),
                                             kv::Chunk<KvCoding::Int8G64>::scale(scales, c), f);
        break;
    case KvCoding::Int4G32:
        kv::Chunk<KvCoding::Int4G32>::decode(kv::Chunk<KvCoding::Int4G32>::load(codes, c),
                                             kv::Chunk<KvCoding::Int4G32>::scale(scales, c), f);
        break;
    case KvCoding::Lloyd4:
        kv::Chunk<KvCoding::Lloyd4>::decode(kv::Chunk<KvCoding::Lloyd4>::load(codes, c),
                                            kv::Chunk<KvCoding::Lloyd4>::scale(scales, c), f);
        break;
    case KvCoding::Mxfp8:
        kv::Chunk<KvCoding::Mxfp8>::decode(kv::Chunk<KvCoding::Mxfp8>::load(codes, c),
                                           kv::Chunk<KvCoding::Mxfp8>::scale(scales, c), f);
        break;
    case KvCoding::F16: break;
    }
    uint4 w;
    __half2* h = reinterpret_cast<__half2*>(&w);
#pragma unroll
    for (int j = 0; j < 4; ++j) { h[j] = __floats2half2_rn(f[2 * j], f[2 * j + 1]); }
    return w;
}

} // namespace ninfer::models::inkling2
