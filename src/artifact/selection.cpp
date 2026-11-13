#include "artifact/selection.h"

#include "artifact/schema.h"
#include "core/qtype.h"

#include <algorithm>

namespace ninfer::artifact {
namespace {

// One plane of a stored object as lines of bytes. A line covers `rows` rows of the object (one,
// or an NVFP4 scale tile's 128, or an NVFP4 divisor's rows) and holds `unit_bytes` per column unit,
// or, for a per-line record such as a row scale, `record_bytes` that every piece of the line
// repeats whole.
struct Plane {
    std::uint64_t offset       = 0;
    std::uint64_t stride       = 0; // bytes from one line to the next
    std::uint64_t rows         = 1; // object rows per line
    std::uint64_t unit_bytes   = 0;
    std::uint64_t record_bytes = 0;
};

struct Layout {
    std::uint64_t rows        = 0;
    std::uint64_t columns     = 0;
    std::uint64_t column_unit = 1; // columns per indivisible run of a plane's bytes
    std::uint64_t row_unit    = 1; // rows a piece's bounds fall on
    std::vector<Plane> planes;
};

Layout describe(const WeightGeometry& g) {
    if (g.shape.empty() || g.shape.size() > 2) {
        throw ArtifactError("a selection needs a vector or a matrix");
    }
    Layout out;
    out.rows    = g.shape.size() == 1 ? 1 : g.shape[0];
    out.columns = g.shape.back();
    const auto units = [&](std::uint64_t unit) {
        out.column_unit = unit;
        return (std::max(g.padded_columns, out.columns) + unit - 1) / unit;
    };
    if (g.layout == QuantLayout::Contiguous) {
        const std::uint64_t row = g.shape.size() == 1 ? g.bytes : g.code_bytes_per_row;
        out.planes.push_back({.offset = 0, .stride = row, .unit_bytes = row / units(1)});
    } else if (g.layout == QuantLayout::GgufBlocks) {
        const std::uint64_t count = units(g.group_size);
        out.planes.push_back(
            {.offset = 0, .stride = g.code_bytes_per_row, .unit_bytes = g.code_bytes_per_row / count});
    } else if (g.layout == QuantLayout::RowSplit) {
        // Whole 128-column runs: the rows are padded to 128 columns, which every group divides.
        const std::uint64_t count = units(128);
        out.planes.push_back({.offset     = 0,
                              .stride     = g.code_bytes_per_row,
                              .unit_bytes = g.code_bytes_per_row / count});
        if (g.high_bytes_per_row != 0) {
            out.planes.push_back({.offset     = g.high_offset,
                                  .stride     = g.high_bytes_per_row,
                                  .unit_bytes = g.high_bytes_per_row / count});
        }
        out.planes.push_back({.offset     = g.scale_offset,
                              .stride     = g.scale_bytes_per_row,
                              .unit_bytes = g.scale_bytes_per_row / count});
    } else if (g.layout == QuantLayout::RowScale) {
        (void)units(1);
        out.planes.push_back({.offset = 0, .stride = g.code_bytes_per_row, .unit_bytes = 1});
        out.planes.push_back({.offset       = g.scale_offset,
                              .stride       = g.scale_bytes_per_row,
                              .record_bytes = g.scale_bytes_per_row});
    } else if (g.layout == QuantLayout::RowInterleaved) {
        (void)units(1);
        out.planes.push_back({.offset = 0, .stride = g.row_stride_bytes, .unit_bytes = 1});
        out.planes.push_back({.offset       = g.scale_offset,
                              .stride       = g.row_stride_bytes,
                              .record_bytes = g.scale_bytes_per_row});
    } else if (g.layout == QuantLayout::BlockScaleK16M128x4) {
        // Codes by row; scales in tiles of 128 rows by 64 columns (512 bytes), row tile by row
        // tile; one FP32 divisor per divisor's rows.
        const std::uint64_t count = units(64);
        out.row_unit = g.divisor_count == 1 ? 128 : std::max<std::uint64_t>(128, out.rows / g.divisor_count);
        out.planes.push_back({.offset     = 0,
                              .stride     = g.code_bytes_per_row,
                              .unit_bytes = g.code_bytes_per_row / count});
        out.planes.push_back({.offset     = g.scale_offset,
                              .stride     = count * 512,
                              .rows       = 128,
                              .unit_bytes = 512});
        out.planes.push_back({.offset       = g.divisor_offset,
                              .stride       = 4,
                              .rows         = out.rows / g.divisor_count,
                              .record_bytes = 4});
    } else {
        throw ArtifactError("a selection cuts no row-split panel");
    }
    return out;
}

void check(const Layout& stored, const ObjectSelection& selection) {
    if (selection.rows.empty() || selection.columns.empty()) {
        throw ArtifactError("a selection names no rows or no columns");
    }
    std::uint64_t previous_row = 0;
    for (const auto& [begin, end] : selection.rows) {
        if (begin >= end || end > stored.rows || begin < previous_row) {
            throw ArtifactError("selection rows must be ordered, disjoint and inside the object");
        }
        if (begin % stored.row_unit != 0 || (end % stored.row_unit != 0 && end != stored.rows)) {
            throw ArtifactError("selection rows must fall on the layout's whole row tiles");
        }
        previous_row = end;
    }
    std::uint64_t previous_end = 0;
    for (const auto& [begin, end] : selection.columns) {
        if (begin >= end || end > stored.columns || begin < previous_end) {
            throw ArtifactError("selection columns must be ordered, disjoint and inside the object");
        }
        if (begin % stored.column_unit != 0 ||
            (end % stored.column_unit != 0 && end != stored.columns)) {
            throw ArtifactError("selection columns must fall on whole stored blocks");
        }
        previous_end = end;
    }
}

std::uint64_t selected_rows(const ObjectSelection& selection) {
    std::uint64_t rows = 0;
    for (const auto& [begin, end] : selection.rows) {
        rows = checked_add(rows, end - begin, "selected rows");
    }
    return rows;
}

std::uint64_t selected_columns(const ObjectSelection& selection) {
    std::uint64_t columns = 0;
    for (const auto& [begin, end] : selection.columns) {
        columns = checked_add(columns, end - begin, "selected columns");
    }
    return columns;
}

std::uint64_t unit_index(const Layout& layout, std::uint64_t column) {
    return (column + layout.column_unit - 1) / layout.column_unit;
}

} // namespace

std::uint64_t selection_column_unit(const WeightGeometry& stored) {
    return describe(stored).column_unit;
}

std::uint64_t selection_row_unit(const WeightGeometry& stored) {
    return describe(stored).row_unit;
}

WeightGeometry selected_geometry(const WeightGeometry& stored, const ObjectSelection& selection) {
    const Layout layout = describe(stored);
    check(layout, selection);
    if (selection.rebase != 0 &&
        (stored.format != QType::INT32 || stored.layout != QuantLayout::Contiguous ||
         stored.shape.size() != 1)) {
        throw ArtifactError("only a piece of an INT32 vector is rebased");
    }
    const std::uint64_t columns = selected_columns(selection);
    if (stored.shape.size() == 1) {
        const std::uint64_t shape[] = {columns};
        return weight_geometry(stored.format, stored.layout, shape, stored.divisor_count);
    }
    const std::uint64_t rows    = selected_rows(selection);
    const std::uint64_t shape[] = {rows, columns};
    const std::uint64_t divisor_rows = stored.shape[0] / stored.divisor_count;
    return weight_geometry(stored.format, stored.layout, shape,
                           stored.divisor_count == 1 ? 1 : (rows + divisor_rows - 1) / divisor_rows);
}

std::vector<SelectedRun> selected_runs(const WeightGeometry& stored,
                                       const ObjectSelection& selection) {
    const Layout layout = describe(stored);
    const Layout piece  = describe(selected_geometry(stored, selection));
    std::vector<SelectedRun> out;
    const bool whole_rows = selection.columns.size() == 1 && selection.columns.front().first == 0 &&
                            selection.columns.front().second == layout.columns;
    for (std::size_t p = 0; p < layout.planes.size(); ++p) {
        const Plane& from = layout.planes[p];
        const Plane& to   = piece.planes[p];
        // Lines the plane's own bytes fill, taken whole, follow each other: one run per row range.
        const bool dense_lines = from.record_bytes != 0
                                     ? from.stride == from.record_bytes
                                     : whole_rows && from.stride == to.stride &&
                                           from.stride == unit_index(layout, layout.columns) *
                                                              from.unit_bytes;
        std::uint64_t placed = 0; // lines of the piece's plane the earlier row ranges filled
        std::uint64_t taken  = 0; // the stored line after the last one the earlier ranges took
        for (const auto& [row_begin, row_end] : selection.rows) {
            // A line can hold rows of several ranges, as the single divisor of a whole NVFP4 object
            // does; the piece holds it once.
            const std::uint64_t first = std::max(row_begin / from.rows, taken);
            const std::uint64_t last  = (row_end + from.rows - 1) / from.rows;
            if (first >= last) { continue; }
            taken = last;
            if (dense_lines) {
                out.push_back(
                    {.source      = checked_add(from.offset, checked_mul(first, from.stride, "line"),
                                                "plane line"),
                     .bytes       = checked_mul(last - first, from.stride, "lines"),
                     .destination = checked_add(to.offset, checked_mul(placed, to.stride, "line"),
                                                "plane line")});
                placed += last - first;
                continue;
            }
            for (std::uint64_t line = first; line < last; ++line) {
                const std::uint64_t source_line =
                    checked_add(from.offset, checked_mul(line, from.stride, "line"), "plane line");
                std::uint64_t destination = checked_add(
                    to.offset, checked_mul(placed + line - first, to.stride, "line"), "plane line");
                if (from.record_bytes != 0) {
                    out.push_back({.source      = source_line,
                                   .bytes       = from.record_bytes,
                                   .destination = destination});
                    continue;
                }
                for (const auto& [begin, end] : selection.columns) {
                    const std::uint64_t at    = unit_index(layout, begin);
                    const std::uint64_t count = unit_index(layout, end) - at;
                    const std::uint64_t bytes = checked_mul(count, from.unit_bytes, "selection run");
                    out.push_back(
                        {.source      = source_line + checked_mul(at, from.unit_bytes, "unit"),
                         .bytes       = bytes,
                         .destination = destination});
                    destination += bytes;
                }
            }
            placed += last - first;
        }
    }
    return out;
}

} // namespace ninfer::artifact
