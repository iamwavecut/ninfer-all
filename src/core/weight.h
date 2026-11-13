#pragma once

#include "core/dtype.h"

#include <cstdint>

namespace ninfer {

// The planes' stored format. Its formats and their ggml block shapes are in core/qtype.h, which a
// kernel that only reads the planes does not need.
enum class QType : std::uint16_t;

enum class QuantLayout : std::uint16_t {
    RowSplit            = 0,
    Contiguous          = 1,
    BlockScaleK16M128x4 = 2,
    RowScale            = 3,
    // RowSplit's bytes, permuted so that the code records of kRowSplitPanelRows consecutive rows
    // for one k-group are contiguous. Same size, same bytes, same scales: a block streaming a row
    // tile reads whole cache lines instead of one 32-byte record per row per group. Device-only --
    // it is produced by a load-time permute, never stored in a `.ninfer`.
    RowSplitPanel = 4,
    // Rows of whole ggml blocks, as a GGUF stores them.
    GgufBlocks = 5,
    RowInterleaved = 6,
};

// Rows per stored panel. Four 32-byte records is exactly one 128-byte line, which is all the
// prefill kernels need (measured flat from 2 to 64 in tools/w4a8_marlin_probe.cu), and it is the
// least disruptive value for the GEMV decode kernels, whose blocks own four consecutive rows.
inline constexpr int kRowSplitPanelRows  = 4;
inline constexpr int kRowSplitPanelShift = 2;
static_assert((1 << kRowSplitPanelShift) == kRowSplitPanelRows,
              "the panel row count is addressed by shifting, so it must be a power of two");

// Kernels address both layouts with one expression, where a shift of zero collapses the panel form
// to the row-major one. Nothing but this decides which a kernel reads.
[[nodiscard]] constexpr int row_split_panel_shift(QuantLayout layout) {
    return layout == QuantLayout::RowSplitPanel ? kRowSplitPanelShift : 0;
}

struct Weight {
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = {}; // Q4_G64_FP16
    std::uint32_t group_size       = 0;
    std::int32_t shape[4]          = {1, 1, 1, 1};
    std::int32_t padded_shape[4]   = {1, 1, 1, 1};
    std::uint32_t ndim             = 0;

    const void* qdata          = nullptr;
    const void* qhigh          = nullptr;
    const void* scales         = nullptr;
    std::int32_t n             = 0;
    std::int32_t k             = 0;
    std::int32_t group         = 0;
    QuantLayout layout         = QuantLayout::RowSplit;
    DType scale_dtype          = DType::FP32;
    std::int32_t scale_ne[4]   = {1, 1, 1, 1};
    std::int64_t scale_nb[4]   = {0, 0, 0, 0};
    float weight_scale_divisor = 0.0F;
    float input_scale_divisor  = 0.0F;

    // An NVFP4 plane assembled from several source matrices carries one divisor per source, in the
    // payload after the scales. `weight_divisors` addresses them and `weight_divisor_rows` says how
    // many consecutive rows each covers, so the divisor of row r is element r /
    // weight_divisor_rows. A plane with one source sets the rows to its own row count, which makes
    // that index zero for every row, and `weight_scale_divisor` is then the whole story. On a stack
    // `weight_scale_divisor` holds only the first source's word, so a route that reads it for any
    // other row is silently wrong by a scale factor.
    const void* weight_divisors      = nullptr;
    std::int32_t weight_divisor_rows = 0;

    // INT32 [K] input gather of a GGUF matrix whose stored columns are a permutation of its
    // input's: column c multiplies input element input_columns[c]. Null for every other matrix.
    const std::int32_t* input_columns = nullptr;
};

} // namespace ninfer
