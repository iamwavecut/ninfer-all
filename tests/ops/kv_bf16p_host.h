#pragma once

// Host decoder of the bf16p KV row (--kv-dtype bf16p), written from the format in
// src/ops/kv_cache/bf16p_codec.cuh rather than from the device code: a test's independent oracle.

#include <cstdint>

namespace ninfer::test::bf16p {

inline constexpr int kRowValues = 256;
inline constexpr int kRowBytes  = 384;
inline constexpr int kSideBytes = 8;

// The exponent nibble of value d: the low nibble of byte 256 + d / 2 for even d, the high for odd.
inline unsigned nibble(const std::uint8_t* row, int d) {
    return (row[kRowValues + d / 2] >> (4 * (d & 1))) & 0xFu;
}

// Value d of a row as BF16 bits. Code byte: sign and seven fraction bits. Nibble 0..13: exponent
// base + nibble; 14: exponent 0; 15: the side record's next escape exponent in column order, or
// base + 15 past the recorded escapes (the poison row).
inline std::uint16_t decode(const std::uint8_t* row, const std::uint8_t* side, int d) {
    const unsigned code     = row[d];
    const unsigned base     = side[0];
    const unsigned recorded = side[1] & 0x7u;
    const unsigned n        = nibble(row, d);
    unsigned exponent       = base + n;
    if (n == 14u) {
        exponent = 0u;
    } else if (n == 15u) {
        unsigned before = 0;
        for (int e = 0; e < d; ++e) before += nibble(row, e) == 15u ? 1u : 0u;
        if (before < recorded) { exponent = side[2 + before]; }
    }
    return static_cast<std::uint16_t>((code >> 7) << 15 | (exponent & 0xFFu) << 7 | (code & 0x7Fu));
}

} // namespace ninfer::test::bf16p
