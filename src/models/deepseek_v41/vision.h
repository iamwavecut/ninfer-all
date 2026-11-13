#pragma once

// DeepSeek-V4.1's vision tower on the device: an image's patches in, its prompt span's input rows
// out (inference/vision.py). The patch projection adds its bias; every block adds to an FP32
// residual the attention of its RMS-normed input (the fused QKV projection with bias, the 2D RoPE,
// softmax attention with scale 1/sqrt(D) over the image's patches, the output projection with
// bias) and its SiLU-gated MLP; the final RMS norm leads into the aligner, which projects each
// downsample_ratio x downsample_ratio cell of the grid (zero past its edge) through the erf GELU
// into the text model's width. Matrices multiply BF16 inputs into FP32 results (cuBLAS).
//
// The span: the learned start row, each aligner row of the grid row by row with the learned
// newline row after each grid row, the learned end row, rounded to BF16 as the release keeps them.
//
// The matrices are resident on the device, or (VisionPlacement::Pinned) the blocks' stay in pinned
// host memory and pass through a window of two device slots, a block's matrices uploading on a copy
// stream while the block before it computes; the patch projection and the aligner stay resident.

#include "core/arena.h"
#include "core/device_buffer.h"
#include "core/device_launch.h"
#include "models/deepseek_v41/model.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::models::deepseek_v41 {

class VisionTower {
public:
    // `max_patches`: the most patches one image may have, which sizes the scratch.
    VisionTower(const Model& model, std::uint32_t max_patches);
    ~VisionTower();
    VisionTower(const VisionTower&)            = delete;
    VisionTower& operator=(const VisionTower&) = delete;

    [[nodiscard]] const VisionConfig& config() const noexcept { return config_; }
    [[nodiscard]] std::uint32_t max_patches() const noexcept { return max_patches_; }
    [[nodiscard]] bool streamed() const noexcept { return streamed_; }
    // Device bytes the tower owns: the window and, streamed, its resident copies.
    [[nodiscard]] std::size_t owned_bytes() const noexcept { return owned_bytes_; }
    // Scratch one encode() of up to max_patches patches takes from its workspace.
    [[nodiscard]] std::size_t workspace_bytes() const;
    // The aligner's grid (rows, columns) of an image of grid_w x grid_h patches.
    [[nodiscard]] std::uint32_t aligner_rows(int grid_h) const noexcept;
    [[nodiscard]] std::uint32_t aligner_columns(int grid_w) const noexcept;

    // The patches (BF16 [grid_h * grid_w][patch features] on the device) to the span's rows, BF16
    // [span_tokens(aligner rows, aligner columns)][projection_dim] in `out` on the device.
    void encode(const __nv_bfloat16* patches, int grid_w, int grid_h, __nv_bfloat16* out,
                WorkspaceArena& workspace, cudaStream_t stream) const;

private:
    struct Matrix {
        const void* data = nullptr; // BF16 [n][k], device or pinned host
        int n = 0, k = 0;
        [[nodiscard]] std::size_t bytes() const noexcept { return std::size_t(n) * k * 2; }
    };
    struct Layer {
        Matrix qkv, output, gate, up, down;
        const float *attention_norm = nullptr, *qkv_bias = nullptr, *output_bias = nullptr,
                    *mlp_norm = nullptr;
    };

    // out[t][column + j] = sum_k w[j][k] x[t][k] (FP32, row stride ldc) for the w rows.
    void gemm(const __nv_bfloat16* x, int tokens, const Matrix& w, float* out, int ldc,
              cudaStream_t stream) const;
    // Streamed: layer i's matrices in window slot `slot`, uploading them first.
    [[nodiscard]] Layer staged(std::size_t i, int slot) const;
    void upload(std::size_t i, int slot) const;
    [[nodiscard]] const void* own(const void* host, std::size_t bytes);

    VisionConfig config_;
    std::uint32_t max_patches_ = 0;
    bool streamed_             = false;
    Matrix patch_embedding_, aligner_up_, aligner_out_;
    const float *patch_bias_ = nullptr, *final_norm_ = nullptr, *aligner_up_bias_ = nullptr,
                *aligner_out_bias_ = nullptr, *image_start_ = nullptr, *image_newline_ = nullptr,
                *image_end_ = nullptr;
    std::vector<Layer> layers_;
    DeviceBuffer inv_freq_; // [head_dim / 4]
    std::vector<DeviceBuffer> owned_;
    std::size_t owned_bytes_ = 0;
    cublasHandle_t blas_     = nullptr;
    DeviceBuffer slots_[2];
    std::size_t slot_bytes_   = 0;
    cudaStream_t copy_stream_ = nullptr;
    cudaEvent_t copied_[2]    = {nullptr, nullptr};
    cudaEvent_t used_[2]      = {nullptr, nullptr};
};

} // namespace ninfer::models::deepseek_v41
