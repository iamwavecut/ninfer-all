#pragma once

// The queries of an attention over rotated 128-wide key planes (row128.h): each 128-value FP32 row
// turned in place by H times kRotatedQueryScale, H the unnormalized Sylvester-Hadamard matrix, the
// keys having been stored as H k times kRotatedRowScale.

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::kv128 {

// Rows [0, count) of 128 floats at rows + r * 128, one warp a row.
void rotate_queries(float* rows, std::int64_t count, cudaStream_t stream);

} // namespace ninfer::ops::kv128
