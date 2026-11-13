#pragma once

// The stored formats of a Weight's planes (QType) and the ggml block shapes among them. A Weight
// holds its format as an opaque QType (core/weight.h); code that names a format includes this
// header, so a new format recompiles that code alone and not every kernel that reads a Weight.

#include <cstdint>

namespace ninfer {

enum class QType : std::uint16_t {
    Q4_G64_FP16         = 0,
    Q5_G64_FP16         = 1,
    Q6_G64_FP16         = 2,
    Q8_G32_FP16         = 3,
    BF16                = 4,
    FP32                = 5,
    INT32               = 6,
    NVFP4               = 7,
    FP8_E4M3FN_ROW_BF16 = 8,
    // Ternary codes {-1, 0, +1} as two-bit two's complement (0b10 unused), four per byte in
    // column order, one binary16 multiplier per 128 columns. Row-split only.
    T2_G128_FP16 = 9,
    // ggml block types kept as a GGUF stored them (GgufBlocks layout only): each row is its K /
    // block_elements blocks, byte for byte. Their products quantize the activation to ggml's q8_1.
    GGUF_Q8_0    = 10,
    GGUF_Q2_K    = 11,
    GGUF_Q3_K    = 12,
    GGUF_Q4_K    = 13,
    GGUF_Q5_K    = 14,
    GGUF_Q6_K    = 15,
    GGUF_IQ2_XXS = 16,
    GGUF_IQ2_XS  = 17,
    GGUF_IQ2_S   = 18,
    GGUF_IQ3_XXS = 19,
    GGUF_IQ3_S   = 20,
    GGUF_IQ1_S   = 21,
    GGUF_IQ1_M   = 22,
    GGUF_IQ4_NL  = 23,
    GGUF_IQ4_XS  = 24,
    GGUF_Q4_0    = 25,
    GGUF_Q5_0    = 26,
    // Two-bit codes over {-1, 0, 1, 2} times one binary16 d per 64 columns (GSQ-RCO's Q2_0).
    GGUF_Q2_0 = 27,
    // Four adjacent offset-binary codes per byte: (code - 1) times a signed binary16 scale
    // per 64 columns. Row-split planes preserve the GSQ-RCO Q2_0 grid without requantization.
    Q2_G64_FP16 = 28,
    // E4M3FN row codes followed by one binary16 multiplier (RowInterleaved only).
    FP8_E4M3FN_ROW_FP16 = 29,
    // ggml's MXFP4 blocks (GgufBlocks): an E8M0 exponent byte, then 32 E2M1 codes, value j < 16 the
    // low nibble of byte j, value j + 16 its high nibble.
    GGUF_MXFP4 = 30,
    // Direct 16-bit words (Contiguous only): binary16 values, and unsigned words a consumer decodes
    // itself (an STQ matrix's trellis; its sign vectors are FP16).
    FP16   = 31,
    UINT16 = 32,
    // ggml's NVFP4 blocks (GgufBlocks) of 64 values: four UE4M3 scale bytes, one per 16 values,
    // then 32 bytes of E2M1 codes; value j of sub-block s is the low nibble of code byte 8 s + j for
    // j < 8 and the high nibble of code byte 8 s + j - 8 otherwise.
    GGUF_NVFP4 = 33,
};

[[nodiscard]] constexpr bool is_gguf(QType format) {
    return (format >= QType::GGUF_Q8_0 && format <= QType::GGUF_Q2_0) ||
           format == QType::GGUF_MXFP4 || format == QType::GGUF_NVFP4;
}

struct GgufBlockShape {
    int elements = 0;
    int bytes    = 0;
};

// ggml-common.h's block sizes; {0, 0} for every non-GGUF format.
[[nodiscard]] constexpr GgufBlockShape gguf_block_shape(QType format) {
    switch (format) {
    case QType::GGUF_Q8_0: return {32, 34};
    case QType::GGUF_Q2_K: return {256, 84};
    case QType::GGUF_Q3_K: return {256, 110};
    case QType::GGUF_Q4_K: return {256, 144};
    case QType::GGUF_Q5_K: return {256, 176};
    case QType::GGUF_Q6_K: return {256, 210};
    case QType::GGUF_IQ2_XXS: return {256, 66};
    case QType::GGUF_IQ2_XS: return {256, 74};
    case QType::GGUF_IQ2_S: return {256, 82};
    case QType::GGUF_IQ3_XXS: return {256, 98};
    case QType::GGUF_IQ3_S: return {256, 110};
    case QType::GGUF_IQ1_S: return {256, 50};
    case QType::GGUF_IQ1_M: return {256, 56};
    case QType::GGUF_IQ4_NL: return {32, 18};
    case QType::GGUF_IQ4_XS: return {256, 136};
    case QType::GGUF_Q4_0:
        return {32, 18};
    case QType::GGUF_Q5_0:
        return {32, 22};
    case QType::GGUF_Q2_0:
        return {64, 18};
    case QType::GGUF_MXFP4: return {32, 17};
    case QType::GGUF_NVFP4: return {64, 36};
    default: return {};
    }
}

} // namespace ninfer
