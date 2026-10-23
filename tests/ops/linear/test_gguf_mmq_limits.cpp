#include "ops/linear/gguf/mmq_limits.h"

#include <climits>
#include <iostream>
#include <stdexcept>

int main() {
    using ninfer::ops::gguf::detail::matrix_work_blocks;
    constexpr int limit = 1 << 30;
    struct Case { int rows; int columns; int k; int expected; };
    const Case cases[] = {
        {1, 1, 1, 1}, {40, 1, 68, 2720},
        {1, 1, limit - 1, limit - 1}, {3, 1, 357913941, limit - 1},
        {limit - 1, 1, 1, limit - 1},
        {1, 1, limit, 0}, {1, 1, limit + 1, 0},
        {32768, 32768, 1, 0}, {3, 1, 357913942, 0},
        {INT_MAX, INT_MAX, INT_MAX, 0}, {1, 1, INT_MAX, 0},
        {0, 1, 1, 0}, {1, 0, 1, 0}, {1, 1, 0, 0},
        {-1, 1, 1, 0}, {1, -1, 1, 0}, {1, 1, -1, 0},
        {INT_MIN, INT_MAX, INT_MAX, 0},
    };
    try {
        for (const auto& c : cases) {
            int actual = 0;
            bool rejected = false;
            try {
                actual = matrix_work_blocks(c.rows, c.columns, c.k);
            } catch (const std::invalid_argument&) {
                rejected = true;
                if (c.expected != 0) { throw; }
            }
            if (actual != c.expected || rejected != (c.expected == 0)) {
                throw std::runtime_error("MMQ work bound mismatch");
            }
        }
        struct Span { std::int64_t count; std::int64_t stride; std::int64_t extent; bool valid; };
        const Span spans[] = {
            {2, INT_MAX, 1, true}, {2, INT_MAX, 2, false},
            {32768, 65536, 65536, true}, {32769, 65536, 65536, false},
            {128, 16909320, 8, true}, {128, 16909320, 9, false},
            {262140, 8172, 9216, true}, {262140, 8208, 9216, false},
            {1, INT_MAX, 1, true}, {1, std::int64_t(INT_MAX) + 1, 1, false},
            {1, LLONG_MAX, 1, false}, {LLONG_MAX, 1, 1, false},
            {1, 1, LLONG_MAX, false}, {0, 1, 1, false},
            {1, 0, 1, false}, {1, 1, 0, false},
            {-1, 1, 1, false}, {1, -1, 1, false}, {1, 1, -1, false},
        };
        for (const auto& c : spans) {
            bool accepted = false;
            try {
                ninfer::ops::gguf::detail::matrix_index_span(c.count, c.stride, c.extent);
                accepted = true;
            } catch (const std::invalid_argument&) {}
            if (accepted != c.valid) { throw std::runtime_error("MMQ index span mismatch"); }
        }
        struct Grid { int x; int y; int z; bool valid; };
        const Grid grids[] = {
            {7, 5, 3, true}, {7, 1, 1, true}, {1, 5, 1, true}, {1, 1, 3, true},
            {8, 1, 1, false}, {1, 6, 1, false}, {1, 1, 4, false},
            {0, 1, 1, false}, {1, -1, 1, false}, {1, 1, 0, false},
            {INT_MAX, INT_MAX, INT_MAX, false},
        };
        for (const auto& c : grids) {
            bool accepted = false;
            try {
                ninfer::ops::gguf::detail::matrix_grid(c.x, c.y, c.z, {7, 5, 3});
                accepted = true;
            } catch (const std::invalid_argument&) {}
            if (accepted != c.valid) { throw std::runtime_error("MMQ grid bound mismatch"); }
        }
        struct Padding { int k; int expected; };
        for (const Padding c : {Padding{1, 256}, {256, 256}, {257, 512},
                                {INT_MAX - 255, INT_MAX - 255}, {INT_MAX - 254, 0},
                                {INT_MAX, 0}, {0, 0}, {-256, 0}, {INT_MIN, 0}}) {
            int actual = 0;
            bool rejected = false;
            try {
                actual = ninfer::ops::gguf::detail::matrix_padded_width(c.k);
            } catch (const std::invalid_argument&) { rejected = true; }
            if (actual != c.expected || rejected != (c.expected == 0)) {
                throw std::runtime_error("MMQ padded K mismatch");
            }
        }
        std::cout << "OK GGUF MMQ index limits\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
