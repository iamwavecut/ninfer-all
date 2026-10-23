#pragma once

#include <cstdint>
#include <array>
#include <climits>
#include <stdexcept>

namespace ninfer::ops::gguf::detail {

inline int matrix_work_blocks(int row_tiles, int column_tiles, int k_blocks) {
    constexpr std::int64_t limit = std::int64_t(1) << 30;
    if (row_tiles <= 0 || column_tiles <= 0 || k_blocks <= 0) {
        throw std::invalid_argument("gguf matrix product: nonpositive geometry");
    }
    const std::int64_t tiles = std::int64_t(row_tiles) * column_tiles;
    if (tiles >= limit || k_blocks > (limit - 1) / tiles) {
        throw std::invalid_argument("gguf matrix product: stream-k index range exceeded");
    }
    return static_cast<int>(tiles * k_blocks);
}

inline void matrix_index_span(std::int64_t count, std::int64_t stride, std::int64_t extent) {
    if (count <= 0 || stride <= 0 || stride > INT_MAX || extent <= 0 || extent > INT_MAX ||
        count - 1 > (INT_MAX - (extent - 1)) / stride) {
        throw std::invalid_argument("gguf matrix product: index span exceeds int range");
    }
}

inline void matrix_grid(int x, int y, int z, const std::array<int, 3>& limits) {
    const int dimensions[] = {x, y, z};
    for (int axis = 0; axis != 3; ++axis) {
        if (dimensions[axis] <= 0 || dimensions[axis] > limits[axis]) {
            throw std::invalid_argument("gguf matrix product: grid exceeds device limit");
        }
    }
}

inline int matrix_padded_width(int k) {
    if (k <= 0 || (std::int64_t(k) + 255) / 256 * 256 > INT_MAX) {
        throw std::invalid_argument("gguf matrix product: padded K exceeds int range");
    }
    return static_cast<int>((std::int64_t(k) + 255) / 256 * 256);
}

}
