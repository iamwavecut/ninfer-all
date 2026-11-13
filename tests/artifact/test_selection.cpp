// Pieces of stored objects (artifact/selection.h), the weights a tensor-parallel rank uploads. The
// oracle is exact: the piece's bytes, gathered by the runs from a stored object whose every byte
// names its row and column unit, must be exactly the selected units in row-major order, and the
// piece's geometry must be what a stored object of the piece's shape would have.

#include "artifact/selection.h"

#include "artifact/schema.h"
#include "core/qtype.h"
#include "core/weight.h"
#include "core/weight_view.h"

#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::artifact;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

WeightGeometry stored(QType format, QuantLayout layout, std::vector<std::uint64_t> shape) {
    return weight_geometry(format, layout, std::span<const std::uint64_t>(shape));
}

std::uint64_t piece_rows(const ObjectSelection& selection) {
    std::uint64_t rows = 0;
    for (const auto& [begin, end] : selection.rows) { rows += end - begin; }
    return rows;
}

// The stored row that row `row` of the piece holds.
std::uint64_t source_row(const ObjectSelection& selection, std::uint64_t row) {
    for (const auto& [begin, end] : selection.rows) {
        if (row < end - begin) { return begin + row; }
        row -= end - begin;
    }
    throw std::runtime_error("a piece row outside the selection");
}

// Gathers the piece through `runs` from a stored object whose unit (u, row) holds the bytes
// `row * 100000 + u` in a fixed pattern, then checks each piece unit against its source.
void check_gather(const WeightGeometry& whole, const ObjectSelection& selection,
                  std::uint64_t rows, std::uint64_t units, std::uint64_t unit_bytes,
                  std::uint64_t column_unit) {
    std::vector<std::uint32_t> source(rows * units * unit_bytes);
    for (std::uint64_t r = 0; r < rows; ++r) {
        for (std::uint64_t u = 0; u < units; ++u) {
            for (std::uint64_t b = 0; b < unit_bytes; ++b) {
                source[(r * units + u) * unit_bytes + b] =
                    static_cast<std::uint32_t>((r * 100000 + u) * 64 + b);
            }
        }
    }
    const auto piece = selected_geometry(whole, selection);
    std::uint64_t picked_units = 0;
    for (const auto& [begin, end] : selection.columns) { picked_units += (end - begin) / column_unit; }
    const std::uint64_t rows_picked = piece_rows(selection);
    std::vector<std::uint32_t> got(rows_picked * picked_units * unit_bytes, 0xffffffffU);
    std::uint64_t covered = 0;
    for (const auto& run : selected_runs(whole, selection)) {
        require(run.source % sizeof(std::uint32_t) == 0 && run.bytes % sizeof(std::uint32_t) == 0,
                "runs stay on the test's 4-byte cells");
        for (std::uint64_t i = 0; i < run.bytes / 4; ++i) {
            got[run.destination / 4 + i] = source[run.source / 4 + i];
        }
        covered += run.bytes;
    }
    require(covered == got.size() * 4, "the runs cover the piece exactly once");
    for (std::uint64_t r = 0; r < rows_picked; ++r) {
        std::uint64_t unit = 0;
        for (const auto& [begin, end] : selection.columns) {
            for (std::uint64_t u = begin / column_unit; u < end / column_unit; ++u, ++unit) {
                for (std::uint64_t b = 0; b < unit_bytes; ++b) {
                    const std::uint64_t at = (r * picked_units + unit) * unit_bytes + b;
                    require(got[at] == source[(source_row(selection, r) * units + u) * unit_bytes + b],
                            "a piece unit holds its source unit");
                }
            }
        }
    }
    require(piece.bytes == got.size() * 4, "the piece geometry counts the gathered bytes");
}

void contiguous_matrix() {
    // FP32 [6, 8]: one unit is one column of four bytes, cut as a cell of the gather.
    const auto whole = stored(QType::FP32, QuantLayout::Contiguous, {6, 8});
    const ObjectSelection rows_only{.rows = {{2, 5}}, .columns = {{0, 8}}};
    check_gather(whole, rows_only, 6, 8, 1, 1);
    require(selected_runs(whole, rows_only).size() == 1, "whole rows gather in one run");
    const ObjectSelection columns{.rows = {{0, 6}}, .columns = {{1, 3}, {5, 8}}};
    check_gather(whole, columns, 6, 8, 1, 1);
    const auto piece = selected_geometry(whole, columns);
    require(piece.shape == std::vector<std::uint64_t>({6, 5}), "a column piece keeps the rows");
    // Two row ranges, as one part of each of two experts: whole rows gather in one run per range.
    const ObjectSelection ranges{.rows = {{0, 2}, {3, 5}}, .columns = {{0, 8}}};
    check_gather(whole, ranges, 6, 8, 1, 1);
    require(selected_runs(whole, ranges).size() == 2, "each row range of whole rows is one run");
    require(selected_geometry(whole, ranges).shape == std::vector<std::uint64_t>({4, 8}),
            "row ranges stack");
    check_gather(whole, {.rows = {{1, 2}, {4, 6}}, .columns = {{2, 6}}}, 6, 8, 1, 1);
}

void vector_object() {
    const auto whole = stored(QType::FP32, QuantLayout::Contiguous, {12});
    const ObjectSelection part{.rows = {{0, 1}}, .columns = {{4, 8}}};
    check_gather(whole, part, 1, 12, 1, 1);
    require(selected_geometry(whole, part).shape == std::vector<std::uint64_t>({4}),
            "a vector piece is a vector");
}

void ggml_blocks() {
    // Q8_0 [4, 256]: blocks of 32 values in 34 bytes; the test cells are bytes / 2 wide, so the
    // gather check runs on 2-byte multiples via a Q8_0 row of 8 blocks = 272 bytes = 68 cells.
    const auto whole = stored(QType::GGUF_Q8_0, QuantLayout::GgufBlocks, {4, 256});
    require(whole.code_bytes_per_row == 272, "a Q8_0 row of 256 values is 272 bytes");
    const ObjectSelection blocks{.rows = {{1, 3}}, .columns = {{32, 96}, {192, 256}}};
    const auto runs = selected_runs(whole, blocks);
    require(runs.size() == 4, "two rows of two column ranges");
    require(runs[0].source == 272 + 34 && runs[0].bytes == 68 && runs[0].destination == 0,
            "the first run starts at row 1, block 1");
    require(runs[1].source == 272 + 6 * 34 && runs[1].bytes == 68 && runs[1].destination == 68,
            "the second range follows the first");
    require(runs[2].source == 2 * 272 + 34 && runs[2].destination == 136, "row 2 follows row 1");
    const auto piece = selected_geometry(whole, blocks);
    require(piece.shape == std::vector<std::uint64_t>({2, 128}) && piece.bytes == 2 * 136,
            "the piece is a Q8_0 [2, 128]");
    bool refused = false;
    try {
        (void)selected_runs(whole, {.rows = {{0, 1}}, .columns = {{16, 64}}});
    } catch (const ArtifactError&) { refused = true; }
    require(refused, "a column bound inside a block is refused");
}

// A stored object whose every byte is a function of its offset, and the piece gathered from it.
struct Gathered {
    std::vector<std::uint8_t> stored;
    std::vector<std::uint8_t> piece;
};

std::uint8_t pattern(std::uint64_t offset) {
    return static_cast<std::uint8_t>((offset * 2654435761ULL) >> 13);
}

Gathered gather(const WeightGeometry& whole, const ObjectSelection& selection) {
    Gathered out;
    out.stored.resize(whole.bytes);
    for (std::uint64_t i = 0; i < whole.bytes; ++i) { out.stored[i] = pattern(i); }
    const auto piece = selected_geometry(whole, selection);
    out.piece.assign(piece.bytes, 0);
    std::vector<std::uint8_t> written(piece.bytes, 0);
    for (const auto& run : selected_runs(whole, selection)) {
        require(run.source + run.bytes <= whole.bytes && run.destination + run.bytes <= piece.bytes,
                "a run stays inside both objects");
        for (std::uint64_t i = 0; i < run.bytes; ++i) {
            require(written[run.destination + i]++ == 0, "no piece byte is written twice");
            out.piece[run.destination + i] = out.stored[run.source + i];
        }
    }
    return out;
}

// The stored column a piece column reads.
std::uint64_t source_column(const ObjectSelection& selection, std::uint64_t column) {
    for (const auto& [begin, end] : selection.columns) {
        if (column < end - begin) { return begin + column; }
        column -= end - begin;
    }
    throw std::runtime_error("a piece column outside the selection");
}

// Row-split: every code byte, high-bit byte and group scale of the piece is the stored one of its
// row and group, addressed independently by the planes' definitions.
void row_split(QType format, std::uint64_t group, std::uint64_t code_bytes, std::uint64_t high_bytes,
               std::vector<std::uint64_t> shape, const ObjectSelection& selection) {
    const auto whole = stored(format, QuantLayout::RowSplit, shape);
    const auto piece = selected_geometry(whole, selection);
    const auto bytes = gather(whole, selection);
    const std::uint64_t rows = piece_rows(selection);
    for (std::uint64_t r = 0; r < rows; ++r) {
        for (std::uint64_t column = 0; column < piece.shape[1]; column += group) {
            const std::uint64_t g  = column / group;
            const std::uint64_t sg = source_column(selection, column) / group;
            const std::uint64_t sr = source_row(selection, r);
            for (std::uint64_t b = 0; b < code_bytes; ++b) {
                require(bytes.piece[r * piece.code_bytes_per_row + g * code_bytes + b] ==
                            bytes.stored[sr * whole.code_bytes_per_row + sg * code_bytes + b],
                        "a row-split code byte holds its source");
            }
            for (std::uint64_t b = 0; b < high_bytes; ++b) {
                require(bytes.piece[piece.high_offset + r * piece.high_bytes_per_row +
                                    g * high_bytes + b] ==
                            bytes.stored[whole.high_offset + sr * whole.high_bytes_per_row +
                                         sg * high_bytes + b],
                        "a row-split high-bit byte holds its source");
            }
            for (std::uint64_t b = 0; b < 2; ++b) {
                require(bytes.piece[weight_scale_offset(piece, r, g) + b] ==
                            bytes.stored[weight_scale_offset(whole, sr, sg) + b],
                        "a row-split group scale holds its source");
            }
        }
    }
}

void row_split_layouts() {
    // Q5: codes, high bits and scales; rows 1..5 of 6, the second and third 128 columns.
    row_split(QType::Q5_G64_FP16, 64, 32, 8, {6, 384},
              {.rows = {{1, 5}}, .columns = {{128, 384}}});
    // Q8 groups of 32, two ranges; T2 groups of 128.
    row_split(QType::Q8_G32_FP16, 32, 32, 0, {3, 512},
              {.rows = {{0, 3}}, .columns = {{0, 128}, {256, 512}}});
    row_split(QType::T2_G128_FP16, 128, 32, 0, {4, 384},
              {.rows = {{2, 4}}, .columns = {{256, 384}}});
    // Q4 whole rows; a K that is not a multiple of 128 cuts only at its end.
    row_split(QType::Q4_G64_FP16, 64, 32, 0, {5, 320},
              {.rows = {{1, 3}}, .columns = {{0, 320}}});
    row_split(QType::Q4_G64_FP16, 64, 32, 0, {5, 320},
              {.rows = {{0, 5}}, .columns = {{128, 320}}});
    // Half of each of three experts' rows, whole and by columns.
    row_split(QType::Q5_G64_FP16, 64, 32, 8, {12, 512},
              {.rows = {{0, 2}, {4, 6}, {8, 10}}, .columns = {{0, 512}}});
    row_split(QType::Q6_G64_FP16, 64, 32, 16, {6, 512},
              {.rows = {{1, 3}, {4, 6}}, .columns = {{256, 512}}});
    bool refused = false;
    try {
        (void)selected_runs(stored(QType::Q4_G64_FP16, QuantLayout::RowSplit, {2, 256}),
                            {.rows = {{0, 2}}, .columns = {{64, 256}}});
    } catch (const ArtifactError&) { refused = true; }
    require(refused, "a row-split bound inside 128 columns is refused");
}

// FP8 with a row scale (a separate plane) or with the scale after each row's codes: a code byte
// per column, and the row's scale whole.
void fp8_layouts() {
    for (const auto layout : {QuantLayout::RowScale, QuantLayout::RowInterleaved}) {
        const auto format = layout == QuantLayout::RowScale ? QType::FP8_E4M3FN_ROW_BF16
                                                            : QType::FP8_E4M3FN_ROW_FP16;
        const auto whole  = stored(format, layout, {4, 100});
        const ObjectSelection selection{.rows = {{1, 4}}, .columns = {{10, 50}, {60, 100}}};
        const auto piece = selected_geometry(whole, selection);
        const auto bytes = gather(whole, selection);
        const auto code  = [](const WeightGeometry& g, std::uint64_t row, std::uint64_t column) {
            return row * (g.row_stride_bytes ? g.row_stride_bytes : g.code_bytes_per_row) + column;
        };
        for (std::uint64_t r = 0; r < 3; ++r) {
            for (std::uint64_t column = 0; column < 80; ++column) {
                require(bytes.piece[code(piece, r, column)] ==
                            bytes.stored[code(whole, r + 1, source_column(selection, column))],
                        "an FP8 code byte holds its source");
            }
            for (std::uint64_t b = 0; b < 2; ++b) {
                require(bytes.piece[weight_scale_offset(piece, r, 0) + b] ==
                            bytes.stored[weight_scale_offset(whole, r + 1, 0) + b],
                        "an FP8 row keeps its scale");
            }
        }
    }
}

// NVFP4: codes by row, swizzled scale tiles of 128 rows and four groups of 16, one divisor per
// divisor's rows.
void nvfp4_layout() {
    const std::uint64_t shape[] = {384, 256};
    const auto whole = weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4, shape, 3);
    const ObjectSelection selection{.rows = {{128, 384}}, .columns = {{64, 192}}};
    const auto piece = selected_geometry(whole, selection);
    require(piece.shape == std::vector<std::uint64_t>({256, 128}) && piece.divisor_count == 2,
            "an NVFP4 piece keeps its divisors' rows");
    const auto bytes = gather(whole, selection);
    for (std::uint64_t r = 0; r < 256; ++r) {
        const std::uint64_t sr = r + 128;
        for (std::uint64_t column = 0; column < 128; column += 2) {
            require(bytes.piece[r * 64 + column / 2] == bytes.stored[sr * 128 + (column + 64) / 2],
                    "an NVFP4 code byte holds its source");
        }
        for (std::uint64_t g = 0; g < 8; ++g) {
            require(bytes.piece[weight_scale_offset(piece, r, g)] ==
                        bytes.stored[weight_scale_offset(whole, sr, g + 4)],
                    "an NVFP4 scale holds its source");
        }
    }
    for (std::uint64_t d = 0; d < 2; ++d) {
        for (std::uint64_t b = 0; b < 4; ++b) {
            require(bytes.piece[piece.divisor_offset + d * 4 + b] ==
                        bytes.stored[whole.divisor_offset + (d + 1) * 4 + b],
                    "an NVFP4 piece keeps its rows' divisors");
        }
    }
    bool refused = false;
    try {
        (void)selected_geometry(whole, {.rows = {{64, 384}}, .columns = {{0, 256}}});
    } catch (const ArtifactError&) { refused = true; }
    require(refused, "an NVFP4 row bound inside a scale tile is refused");
}

// Several row ranges of one NVFP4 object, as a tensor-parallel rank takes its rows of q, k and v
// from one stored object: a divisor covers the rows of every range it spans once, so a piece of a
// single-divisor object holds that divisor once, and one of a stacked object one word per range.
void nvfp4_row_ranges() {
    const std::uint64_t shape[] = {512, 128};
    const ObjectSelection selection{.rows = {{0, 128}, {256, 384}}, .columns = {{0, 128}}};
    for (const std::uint64_t divisors : {1U, 4U}) {
        const auto whole = weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4, shape,
                                           divisors);
        const auto piece = selected_geometry(whole, selection);
        const std::uint64_t kept = divisors == 1 ? 1 : 2;
        require(piece.shape == std::vector<std::uint64_t>({256, 128}) && piece.divisor_count == kept,
                "an NVFP4 piece of two row ranges keeps one divisor per stored divisor it spans");
        const auto bytes = gather(whole, selection);
        for (std::uint64_t r = 0; r < 256; ++r) {
            const std::uint64_t sr = source_row(selection, r);
            for (std::uint64_t b = 0; b < 64; ++b) {
                require(bytes.piece[r * 64 + b] == bytes.stored[sr * 64 + b],
                        "an NVFP4 code byte of a row range holds its source");
            }
            for (std::uint64_t g = 0; g < 8; ++g) {
                require(bytes.piece[weight_scale_offset(piece, r, g)] ==
                            bytes.stored[weight_scale_offset(whole, sr, g)],
                        "an NVFP4 scale of a row range holds its source");
            }
        }
        for (std::uint64_t d = 0; d < kept; ++d) {
            const std::uint64_t source = divisors == 1 ? 0 : 2 * d;
            for (std::uint64_t b = 0; b < 4; ++b) {
                require(bytes.piece[piece.divisor_offset + d * 4 + b] ==
                            bytes.stored[whole.divisor_offset + source * 4 + b],
                        "an NVFP4 piece of two row ranges holds the divisors of its rows");
            }
        }
    }
}

void refusals() {
    const auto whole = stored(QType::FP32, QuantLayout::Contiguous, {6, 8});
    const ObjectSelection bad[] = {
        {.rows = {{3, 3}}, .columns = {{0, 8}}},
        {.rows = {{0, 7}}, .columns = {{0, 8}}},
        {.rows = {{0, 2}}, .columns = {}},
        {.rows = {{0, 2}}, .columns = {{4, 6}, {2, 3}}},
        {.rows = {{0, 2}}, .columns = {{0, 9}}},
        {.rows = {}, .columns = {{0, 8}}},
        {.rows = {{2, 4}, {0, 1}}, .columns = {{0, 8}}},
        {.rows = {{0, 3}, {2, 4}}, .columns = {{0, 8}}},
    };
    for (const auto& selection : bad) {
        bool refused = false;
        try {
            (void)selected_geometry(whole, selection);
        } catch (const ArtifactError&) { refused = true; }
        require(refused, "an empty, outside or unordered selection is refused");
    }
    // Only an INT32 vector's values can be rebased.
    const auto indices = stored(QType::INT32, QuantLayout::Contiguous, {16});
    const ObjectSelection rebased{.rows = {{0, 1}}, .columns = {{4, 8}}, .rebase = 4};
    require(selected_geometry(indices, rebased).shape == std::vector<std::uint64_t>({4}),
            "an INT32 vector piece can be rebased");
    bool refused = false;
    try {
        (void)selected_geometry(stored(QType::FP32, QuantLayout::Contiguous, {16}), rebased);
    } catch (const ArtifactError&) { refused = true; }
    require(refused, "a rebased FP32 piece is refused");
}

} // namespace

int main() {
    try {
        contiguous_matrix();
        vector_object();
        ggml_blocks();
        row_split_layouts();
        fp8_layouts();
        nvfp4_layout();
        nvfp4_row_ranges();
        refusals();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "OK artifact selection\n";
    return 0;
}
