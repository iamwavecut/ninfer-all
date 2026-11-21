#pragma once

// The Inkling 2 vision tower's own kernels. Activations are row-major over the tower's sequence:
// the class row first, then the image's patches row by row of its grid_w x grid_h grid (patch p at
// row 1 + p). The residual is FP32, projection inputs BF16, projection outputs FP32.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::models::inkling2::vision_kernels {

// The tower's input rows h [1 + grid_w * grid_h][hidden]: row 0 the class embedding plus its
// position row; row 1 + p the patch projection proj[p] plus the learned position rows
// [grid][grid][hidden] resized to grid_w x grid_h (as they are when the sizes match, else
// bilinearly with half-pixel centers and clamped edges, ggml_interpolate's BILINEAR mode).
void embed(const float* proj, const float* positions, int grid, int grid_w, int grid_h,
           const float* class_embedding, const float* class_position, int hidden, float* h,
           cudaStream_t stream);

// LayerNorm of `rows` FP32 rows, y = (x - mean) / sqrt(var + eps) * w + b (biased variance), as
// BF16 rows (out) and FP32 rows (f32, which may be x itself), each when given. hidden % 4 == 0, at
// most 4096.
void layer_norm(const float* x, const float* w, const float* b, int rows, int hidden, float eps,
                __nv_bfloat16* out, cudaStream_t stream, float* f32 = nullptr);

// The attention's inputs from the Q, K and V products qkv [rows][3 hidden] (in that order) and
// their biases: BF16 [rows][3 hidden] in the same places, each head (head_dim D values) of Q and K
// turned by the 2D RoPE: the pairs (2i, 2i + 1) of its first half by x * inv_freq[i] and those of its
// second half by y * inv_freq[i], i < D/4, where (x, y) is (0, 0) for the class row and (column + 1,
// row + 1) for a patch (inv_freq: [D/4]).
void rotate_qkv(const float* qkv, const float* q_bias, const float* k_bias, const float* v_bias,
                int rows, int grid_w, int hidden, int head_dim, const float* inv_freq,
                __nv_bfloat16* out, cudaStream_t stream);

// p[r][j] = bf16(softmax_j(s[r][j])) over `rows` rows of `keys` FP32 scores.
void softmax_rows(const float* s, int rows, int keys, __nv_bfloat16* p, cudaStream_t stream);

// h[r] += (y[r] + bias) * scale: a block's output projection or MLP back into the residual.
void residual(float* h, const float* y, const float* bias, const float* scale, int rows,
              int hidden, cudaStream_t stream);

// mid[r][i] = bf16(gelu(x[r][i] + bias[i])) with the erf GELU (bias null: none).
void gelu(const float* x, const float* bias, int rows, int width, __nv_bfloat16* mid,
          cudaStream_t stream);

// The projector's input from the tower's output h (the class row first): one row of 4 hidden values
// per 2 x 2 cell of patches, cells row by row, value dy * 2 hidden + dx * hidden + c of cell (Y, X)
// being h at patch (2 Y + dy, 2 X + dx), channel c (grid_w and grid_h even).
void merge(const float* h, int grid_w, int grid_h, int hidden, __nv_bfloat16* cells,
           cudaStream_t stream);

void f32_to_bf16(const float* x, std::int64_t n, __nv_bfloat16* out, cudaStream_t stream);

} // namespace ninfer::models::inkling2::vision_kernels
