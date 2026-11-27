#pragma once

#include <cstdint>

namespace ninfer::ops {

// Rows of a batched sequence call. An ordinary decode round gives each row one column and takes up
// to kMaximumDecodeRows rows; a call of wider rows (verification, replay records, drafts) takes up
// to kMaximumVerifyRows. The Engine's limits (include/ninfer/base.h) stay within these.
inline constexpr std::int32_t kMaximumDecodeRows = 64;
inline constexpr std::int32_t kMaximumVerifyRows = 8;

// True when a batched call of `batch` rows, each `width` columns wide, is within the row domain.
[[nodiscard]] constexpr bool batch_rows_admitted(std::int32_t width, std::int32_t batch) {
    return width >= 1 && batch >= 1 &&
           batch <= (width == 1 ? kMaximumDecodeRows : kMaximumVerifyRows);
}

} // namespace ninfer::ops
