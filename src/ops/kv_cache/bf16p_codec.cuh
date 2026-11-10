#pragma once

// The packed BF16 KV row (KvCacheStorage::Bf16Packed, --kv-dtype bf16p): a D256 K or V row of BF16
// values stored losslessly in 384 bytes and an 8-byte side record, 392 bytes against BF16's 512.
// The form is Infernix's bf16p (Apache-2.0), itself UZ-40's bf16_p12; the reader and the poison
// row are this tree's. A row:
//
//   [0, 256)    code[d] = sign << 7 | the seven fraction bits of value d
//   [256, 384)  nibble[d] (d even: the low nibble of byte 256 + d / 2):
//                 0..13  exponent base + nibble
//                 14     exponent 0 (a zero or a subnormal: the code byte is the value)
//                 15     an escape: its exponent is in the side record
//
// and the side record, in the storage's scale plane:
//
//   0           base (the window of normal exponents is [base, base + 13])
//   1           bits 0-2: n, the escapes (0..kBf16pEscapes); bit 7: some value has exponent 0
//   [2, 8)      the escapes' exponents in column order; unused bytes zero
//
// An escape is a value outside the window: an infinity or NaN, or a normal value too large or too
// small for it. The window: a row whose normal exponents span at most 14 binades takes
// base = clamp(lowest, 1, 240); a wider row takes the lowest base of the windows holding most of
// them. A row with more than kBf16pEscapes escapes has no packed form. The append stores such a row
// as the poison row instead: every nibble 15, no escapes recorded and base 240, so every value
// decodes to exponent 255 with all seven fraction bits set, a NaN. Attention over it gives NaN,
// and the request that holds the row ends as one with non-finite logits. A record of zero bytes
// (a page never written) decodes to zeros.

#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kBf16pRowValues       = 256;
inline constexpr int kBf16pRowBytes        = 384;
inline constexpr int kBf16pNibbles         = 256; // the nibbles' offset in a row
inline constexpr int kBf16pSideBytes       = 8;
inline constexpr int kBf16pEscapes         = 6;
inline constexpr unsigned kBf16pWindow     = 14; // binades of normal exponents
inline constexpr unsigned kBf16pZeroCode   = 14;
inline constexpr unsigned kBf16pEscapeCode = 15;
inline constexpr unsigned kBf16pPoisonBase = 240;

// A row's bytes and its side record in a paged plane of kv_heads heads.
__device__ __forceinline__ std::int64_t bf16p_row_offset(int physical_page, int kv_head,
                                                         int page_offset, int kv_heads) {
    return paged_kv_element_offset<kBf16pRowBytes>(physical_page, kv_head, page_offset, 0,
                                                   kv_heads);
}
__device__ __forceinline__ std::int64_t bf16p_side_offset(int physical_page, int kv_head,
                                                          int page_offset, int kv_heads) {
    return paged_kv_element_offset<kBf16pSideBytes>(physical_page, kv_head, page_offset, 0,
                                                    kv_heads);
}

namespace detail {

__device__ __forceinline__ unsigned bf16p_prmt(unsigned a, unsigned b, unsigned selector) {
    unsigned r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(selector));
    return r;
}

// Warp reductions over all 32 lanes: redux.sync from sm_80, a butterfly before it (same values).
__device__ __forceinline__ unsigned bf16p_warp_min(unsigned v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __reduce_min_sync(0xFFFFFFFFU, v);
#else
#    pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v = min(v, __shfl_xor_sync(0xFFFFFFFFU, v, o)); }
    return v;
#endif
}
__device__ __forceinline__ unsigned bf16p_warp_max(unsigned v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __reduce_max_sync(0xFFFFFFFFU, v);
#else
#    pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v = max(v, __shfl_xor_sync(0xFFFFFFFFU, v, o)); }
    return v;
#endif
}
__device__ __forceinline__ unsigned bf16p_warp_add(unsigned v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __reduce_add_sync(0xFFFFFFFFU, v);
#else
#    pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, o); }
    return v;
#endif
}
__device__ __forceinline__ unsigned bf16p_warp_or(unsigned v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __reduce_or_sync(0xFFFFFFFFU, v);
#else
#    pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v |= __shfl_xor_sync(0xFFFFFFFFU, v, o); }
    return v;
#endif
}

// An exclusive prefix sum over the lanes.
__device__ __forceinline__ unsigned bf16p_lanes_before(unsigned mine, int lane) {
    unsigned sum = mine;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const unsigned n = __shfl_up_sync(0xFFFFFFFFU, sum, d);
        if (lane >= d) { sum += n; }
    }
    return sum - mine;
}

// Value pair j of a lane's chunk (values 2 j, 2 j + 1, the low and high halves of output word j):
// its code bytes as [code, sign, code, sign] (bytes 0, 1 or 2, 3 of code word j / 2; a selector
// nibble with bit 3 set gives the selected byte's sign in all eight bits), and its exponent
// nibbles as [nibble, 0, nibble, 0] from lo (the even values' nibbles a byte) and hi (the odd
// values').
template <int J> inline constexpr unsigned kBf16pCodeSelector = J % 2 == 0 ? 0x9180U : 0xB3A2U;
template <int J>
inline constexpr unsigned kBf16pNibbleSelector = (J | (J | 8) << 4 | (J + 4) << 8 | (J + 12) << 12);

// The values of a lane's chunk its base does not give: a value of exponent 0 (nibble 14) keeps its
// code byte alone, and an escape (nibble 15) takes the side's next exponent in column order. The
// whole warp calls it for one row (the escapes' order is a warp scan).
__device__ __forceinline__ void bf16p_patch(unsigned (&w)[4], unsigned nib, uint2 side, int lane) {
    // Bit 4 i: value i is an escape (nibble 15) or of exponent 0 (nibble 14).
    const unsigned high    = (nib >> 1) & (nib >> 2) & (nib >> 3) & 0x11111111U;
    const unsigned escaped = nib & high, zero = ~nib & high;
    if (zero != 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if ((zero >> (4 * i)) & 1U) { w[i >> 1] &= ~(0xFFU << (7 + 16 * (i & 1))); }
        }
    }
    if (((side.x >> 8) & 7U) == 0) { return; } // uniform: the row's record
    unsigned index = bf16p_lanes_before(__popc(escaped), lane);
    const unsigned long long exponents =
        (static_cast<unsigned long long>(side.y) << 32 | side.x) >> 16;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if ((escaped >> (4 * i)) & 1U) {
            const unsigned e = index < static_cast<unsigned>(kBf16pEscapes)
                                   ? static_cast<unsigned>(exponents >> (8 * index)) & 0xFFU
                                   : 0U;
            ++index;
            const unsigned shift = 7 + 16 * (i & 1);
            w[i >> 1]            = (w[i >> 1] & ~(0xFFU << shift)) | (e << shift);
        }
    }
}

} // namespace detail

// Lane L's chunk (values 8 L .. 8 L + 7) of a packed row, as the eight BF16 values the row was
// appended from: its code bytes `code` (row bytes 8 L ..), its exponent nibbles `nib` (row bytes
// 256 + 4 L ..) and the row's side record `side`. The whole warp calls it for one row, lane L
// taking values 8 L ... A word: (code bytes with their signs) & 0x807F807F | (nibbles + base)
// << 7; nibbles 14 and 15 give base + 14 or 15 <= 255, which bf16p_patch replaces (none in the
// poison row: its record has no escapes, so base 240 + 15 = 255 stays).
__device__ __forceinline__ uint4 bf16p_decode_regs(uint2 code, unsigned nib, uint2 side, int lane) {
    const unsigned lo = nib & 0x0F0F0F0FU, hi = (nib >> 4) & 0x0F0F0F0FU;
    const unsigned base7 = (side.x & 0xFFU) * 0x00800080U;
    const auto pair      = [&](unsigned word, unsigned code_selector, unsigned nibble_selector) {
        return (detail::bf16p_prmt(word, 0, code_selector) & 0x807F807FU) |
               ((detail::bf16p_prmt(lo, hi, nibble_selector) << 7) + base7);
    };
    unsigned w[4] = {pair(code.x, detail::kBf16pCodeSelector<0>, detail::kBf16pNibbleSelector<0>),
                     pair(code.x, detail::kBf16pCodeSelector<1>, detail::kBf16pNibbleSelector<1>),
                     pair(code.y, detail::kBf16pCodeSelector<2>, detail::kBf16pNibbleSelector<2>),
                     pair(code.y, detail::kBf16pCodeSelector<3>, detail::kBf16pNibbleSelector<3>)};
    if (((side.x >> 8) & 0xFFU) != 0) { detail::bf16p_patch(w, nib, side, lane); }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

// Lane L's chunk of the row at `row` (its 384 bytes) with side record `side`, decoded; the whole
// warp calls it for one row.
__device__ __forceinline__ uint4 bf16p_load_chunk(const std::uint8_t* __restrict__ row,
                                                  const std::uint8_t* __restrict__ side, int lane) {
    const uint2 code   = *reinterpret_cast<const uint2*>(row + 8 * lane);
    const unsigned nib = *reinterpret_cast<const unsigned*>(row + kBf16pNibbles + 4 * lane);
    const uint2 record = *reinterpret_cast<const uint2*>(side);
    return bf16p_decode_regs(code, nib, record, lane);
}

// Chunk c (values 8 c .. 8 c + 7) of the row at `row` with side record `side`, decoded by one
// thread: the same bits as bf16p_decode_regs. An escape's exponent is the side's next one after the
// escapes of chunks 0 .. c - 1, which a row with escapes counts from their nibble words.
__device__ __forceinline__ uint4 bf16p_decode8(const std::uint8_t* __restrict__ row,
                                               const std::uint8_t* __restrict__ side, int c) {
    const uint2 code   = *reinterpret_cast<const uint2*>(row + 8 * c);
    const unsigned nib = *reinterpret_cast<const unsigned*>(row + kBf16pNibbles + 4 * c);
    const uint2 record = *reinterpret_cast<const uint2*>(side);
    const unsigned lo = nib & 0x0F0F0F0FU, hi = (nib >> 4) & 0x0F0F0F0FU;
    const unsigned base7 = (record.x & 0xFFU) * 0x00800080U;
    const auto pair      = [&](unsigned word, unsigned code_selector, unsigned nibble_selector) {
        return (detail::bf16p_prmt(word, 0, code_selector) & 0x807F807FU) |
               ((detail::bf16p_prmt(lo, hi, nibble_selector) << 7) + base7);
    };
    unsigned w[4] = {pair(code.x, detail::kBf16pCodeSelector<0>, detail::kBf16pNibbleSelector<0>),
                     pair(code.x, detail::kBf16pCodeSelector<1>, detail::kBf16pNibbleSelector<1>),
                     pair(code.y, detail::kBf16pCodeSelector<2>, detail::kBf16pNibbleSelector<2>),
                     pair(code.y, detail::kBf16pCodeSelector<3>, detail::kBf16pNibbleSelector<3>)};
    if (((record.x >> 8) & 0xFFU) != 0) {
        const auto escapes = [](unsigned n) {
            return (n & (n >> 1) & (n >> 2) & (n >> 3) & 0x11111111U);
        };
        const unsigned high    = (nib >> 1) & (nib >> 2) & (nib >> 3) & 0x11111111U;
        const unsigned escaped = nib & high, zero = ~nib & high;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if ((zero >> (4 * i)) & 1U) { w[i >> 1] &= ~(0xFFU << (7 + 16 * (i & 1))); }
        }
        if (((record.x >> 8) & 7U) != 0 && escaped != 0) {
            unsigned index = 0;
            for (int j = 0; j < c; ++j) {
                index += __popc(escapes(
                    *reinterpret_cast<const unsigned*>(row + kBf16pNibbles + 4 * j)));
            }
            const unsigned long long exponents =
                (static_cast<unsigned long long>(record.y) << 32 | record.x) >> 16;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if ((escaped >> (4 * i)) & 1U) {
                    const unsigned e = index < static_cast<unsigned>(kBf16pEscapes)
                                           ? static_cast<unsigned>(exponents >> (8 * index)) & 0xFFU
                                           : 0U;
                    ++index;
                    const unsigned shift = 7 + 16 * (i & 1);
                    w[i >> 1]            = (w[i >> 1] & ~(0xFFU << shift)) | (e << shift);
                }
            }
        }
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

// Packs one row: lane L holds values 8 L .. 8 L + 7 (`bits`, BF16); `scratch` is the warp's 256
// words of shared memory. The whole warp calls it; it writes the row's 384 bytes and its 8-byte
// side record (the poison row when the row has no packed form) and returns whether it had one
// (uniform).
__device__ __forceinline__ bool bf16p_encode_row(uint4 bits, std::uint8_t* __restrict__ row,
                                                 std::uint8_t* __restrict__ side, unsigned* scratch,
                                                 int lane) {
    const unsigned word[4] = {bits.x, bits.y, bits.z, bits.w};
    unsigned e[8];
    unsigned lowest = 255, highest = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        e[i] = (word[i >> 1] >> (7 + 16 * (i & 1))) & 0xFFU;
        if (e[i] != 0 && e[i] != 255) {
            lowest  = min(lowest, e[i]);
            highest = max(highest, e[i]);
        }
    }
    lowest  = detail::bf16p_warp_min(lowest);
    highest = detail::bf16p_warp_max(highest);
    unsigned base;
    if (highest < lowest || highest - lowest < kBf16pWindow) {
        // Every normal exponent fits one window (or there is none).
        base = highest < lowest ? 1U : min(max(lowest, 1U), 240U);
    } else {
        // The lowest base of the windows holding most normal exponents: count[j] = exponents <= j,
        // then window(b) = count[b + 13] - count[b - 1] for b in 1..240 (lane L takes b = 8 L + 1
        // .. 8 L + 8).
        *reinterpret_cast<uint4*>(scratch + 8 * lane)     = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(scratch + 8 * lane + 4) = make_uint4(0, 0, 0, 0);
        __syncwarp();
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (e[i] != 0 && e[i] != 255) { atomicAdd(scratch + e[i], 1U); }
        }
        __syncwarp();
        unsigned bins[8], run = 0;
        {
            const uint4 lo      = *reinterpret_cast<const uint4*>(scratch + 8 * lane);
            const uint4 hi      = *reinterpret_cast<const uint4*>(scratch + 8 * lane + 4);
            const unsigned h[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                run += h[i];
                bins[i] = run;
            }
        }
        const unsigned offset = detail::bf16p_lanes_before(run, lane);
        __syncwarp();
        *reinterpret_cast<uint4*>(scratch + 8 * lane) =
            make_uint4(bins[0] + offset, bins[1] + offset, bins[2] + offset, bins[3] + offset);
        *reinterpret_cast<uint4*>(scratch + 8 * lane + 4) =
            make_uint4(bins[4] + offset, bins[5] + offset, bins[6] + offset, bins[7] + offset);
        __syncwarp();
        unsigned best = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int b = 8 * lane + 1 + i;
            if (b <= 240) {
                const unsigned count = scratch[b + kBf16pWindow - 1] - scratch[b - 1];
                best                 = max(best, count << 8 | static_cast<unsigned>(255 - b));
            }
        }
        best = detail::bf16p_warp_max(best);
        base = 255U - (best & 0xFFU);
        __syncwarp(); // the scratch is free for the warp's next row
    }

    unsigned nibbles = 0, escaped = 0, zero = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned offset = e[i] - base; // wraps for e < base
        const bool inside     = e[i] != 0 && e[i] != 255 && offset < kBf16pWindow;
        const unsigned nibble = e[i] == 0 ? kBf16pZeroCode : inside ? offset : kBf16pEscapeCode;
        nibbles |= nibble << (4 * i);
        escaped |= (nibble == kBf16pEscapeCode ? 1U : 0U) << i;
        zero |= e[i] == 0 ? 1U : 0U;
    }
    const unsigned mine  = __popc(escaped);
    const unsigned total = detail::bf16p_warp_add(mine);
    if (total > static_cast<unsigned>(kBf16pEscapes)) {
        // No packed form: the poison row (uniform).
        *reinterpret_cast<uint2*>(row + 8 * lane) = make_uint2(0x7F7F7F7FU, 0x7F7F7F7FU);
        *reinterpret_cast<unsigned*>(row + kBf16pNibbles + 4 * lane) = 0xFFFFFFFFU;
        if (lane == 0) { *reinterpret_cast<uint2*>(side) = make_uint2(kBf16pPoisonBase, 0); }
        return false;
    }
    const bool zeros = __any_sync(0xFFFFFFFFU, zero != 0);
    unsigned code[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned h = (word[i >> 1] >> (16 * (i & 1))) & 0xFFFFU;
        code[i >> 2] |= (((h >> 8) & 0x80U) | (h & 0x7FU)) << (8 * (i & 3));
    }
    *reinterpret_cast<uint2*>(row + 8 * lane)                    = make_uint2(code[0], code[1]);
    *reinterpret_cast<unsigned*>(row + kBf16pNibbles + 4 * lane) = nibbles;
    // The side record: this lane's escape exponents after the lower lanes' (bytes 2 + index),
    // OR-combined.
    unsigned before           = detail::bf16p_lanes_before(mine, lane);
    unsigned long long record = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if ((escaped >> i) & 1U) {
            record |= static_cast<unsigned long long>(e[i]) << (8 * (2 + before));
            ++before;
        }
    }
    const unsigned lo    = detail::bf16p_warp_or(static_cast<unsigned>(record));
    const unsigned hi    = detail::bf16p_warp_or(static_cast<unsigned>(record >> 32));
    const unsigned count = total | (zeros ? 0x80U : 0U);
    if (lane == 0) { *reinterpret_cast<uint2*>(side) = make_uint2(lo | base | count << 8, hi); }
    return true;
}

} // namespace ninfer::ops
