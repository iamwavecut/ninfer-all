#pragma once

// FP64 oracle of the 128-wide rotated KV planes (ops/kv_cache/row128.h): H x for the 128-point
// Sylvester-Hadamard matrix H[i][j] = (-1)^popcount(i & j), evaluated naively.

#include <vector>

namespace ninfer::test {

inline std::vector<double> hadamard128(const double* x) {
    std::vector<double> y(128, 0.0);
    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < 128; ++j) {
            y[i] += (__builtin_popcount(unsigned(i & j)) & 1) != 0 ? -x[j] : x[j];
        }
    }
    return y;
}

} // namespace ninfer::test
