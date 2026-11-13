#include "models/deepseek_v41/vision.h"

#include "core/device_buffer.h"
#include "core/weight_view.h"
#include "models/deepseek_v41/vision_kernels.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::models::deepseek_v41 {
namespace {

namespace vk = vision_kernels;

constexpr std::size_t kAlign = 256;
// The attention's scores and probabilities of a slice of query rows, all heads at once.
constexpr std::size_t kAttentionBytes = std::size_t(128) << 20;

std::size_t round_up(std::size_t value) { return (value + kAlign - 1) / kAlign * kAlign; }

const float* fp32(const Model& model, WeightId id, std::int32_t count) {
    const Tensor tensor = weight_tensor(model.weight(id).view, {count});
    if (tensor.dtype != DType::FP32) {
        throw std::invalid_argument(model.weight(id).name + " must be stored as fp32");
    }
    return static_cast<const float*>(tensor.data);
}

void check_blas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("deepseek_v41 vision: ") + what + " failed (" +
                                 std::to_string(int(status)) + ")");
    }
}

// Query rows per attention slice of an image of n patches.
int attention_rows(std::size_t n, std::size_t heads) {
    return int(std::clamp<std::size_t>(kAttentionBytes / (heads * n * 6), 1, n));
}

// The scratch of one encode, as encode() allocates it: the residual and a block's input, then the
// largest of its phases (each in its own scope).
std::size_t scratch_bytes(const VisionConfig& c, std::size_t n) {
    const std::size_t h = c.hidden_size, w = c.intermediate_size, heads = c.num_attention_heads;
    const std::size_t big = c.projection_dim;
    const std::size_t cells = std::min<std::size_t>(n, c.max_image_tokens);
    const std::size_t rows  = std::size_t(attention_rows(n, heads));
    const std::size_t embed     = round_up(n * h * 4);
    const std::size_t attention = round_up(n * 3 * h * 4) + round_up(n * 3 * h * 2) +
                                  round_up(n * h * 2) + round_up(heads * rows * n * 4) +
                                  round_up(heads * rows * n * 2);
    const std::size_t mlp     = round_up(n * 2 * w * 4) + round_up(n * w * 2);
    const std::size_t aligner = round_up(cells * c.merged_features() * 2) + round_up(cells * big * 4) +
                                round_up(cells * big * 2);
    return round_up(n * h * 4) + round_up(n * h * 2) +
           std::max({embed, attention, mlp, aligner});
}

} // namespace

VisionTower::VisionTower(const Model& model, std::uint32_t max_patches)
    : config_(*model.vision_config()), max_patches_(max_patches),
      streamed_(model.vision_placement() == VisionPlacement::Pinned) {
    if (!model.vision() || (model.vision_placement() != VisionPlacement::Device && !streamed_)) {
        throw std::invalid_argument("deepseek_v41 vision: the model holds no vision tower");
    }
    if (max_patches == 0) { throw std::invalid_argument("deepseek_v41 vision: no patch capacity"); }
    const VisionWeights& w = *model.vision();
    const int h = int(config_.hidden_size), big = int(config_.projection_dim);
    const auto matrix = [&](WeightId id, int n, int k) {
        const Tensor tensor = weight_tensor(model.weight(id).view, {k, n});
        if (tensor.dtype != DType::BF16) {
            throw std::invalid_argument(model.weight(id).name + ": the tower runs BF16 matrices");
        }
        return Matrix{tensor.data, n, k};
    };
    patch_embedding_ = matrix(w.patch_embedding, h, int(config_.patch_features()));
    aligner_up_      = matrix(w.aligner_up, big, int(config_.merged_features()));
    aligner_out_     = matrix(w.aligner_out, big, big);
    if (streamed_) {
        // The patch projection and the aligner stay resident: copies, which the same cuBLAS calls
        // multiply as the resident tower does, so both give the same rows.
        for (Matrix* m : {&patch_embedding_, &aligner_up_, &aligner_out_}) {
            m->data = own(m->data, m->bytes());
        }
    }
    patch_bias_       = fp32(model, w.patch_bias, h);
    final_norm_       = fp32(model, w.final_norm, h);
    aligner_up_bias_  = fp32(model, w.aligner_up_bias, big);
    aligner_out_bias_ = fp32(model, w.aligner_out_bias, big);
    image_start_      = fp32(model, w.image_start, big);
    image_newline_    = fp32(model, w.image_newline, big);
    image_end_        = fp32(model, w.image_end, big);
    const int width   = int(config_.intermediate_size);
    for (const VisionLayerWeights& lw : w.layers) {
        Layer layer;
        layer.qkv            = matrix(lw.qkv, 3 * h, h);
        layer.output         = matrix(lw.output, h, h);
        layer.gate           = matrix(lw.gate, width, h);
        layer.up             = matrix(lw.up, width, h);
        layer.down           = matrix(lw.down, h, width);
        layer.attention_norm = fp32(model, lw.attention_norm, h);
        layer.qkv_bias       = fp32(model, lw.qkv_bias, 3 * h);
        layer.output_bias    = fp32(model, lw.output_bias, h);
        layer.mlp_norm       = fp32(model, lw.mlp_norm, h);
        layers_.push_back(layer);
    }
    if (streamed_) {
        const Layer& first = layers_.front();
        for (const Matrix* m : {&first.qkv, &first.output, &first.gate, &first.up, &first.down}) {
            slot_bytes_ += round_up(m->bytes());
        }
        for (int i = 0; i < 2; ++i) {
            slots_[i] = DeviceBuffer(slot_bytes_);
            CUDA_CHECK(cudaEventCreateWithFlags(&copied_[i], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&used_[i], cudaEventDisableTiming));
        }
        owned_bytes_ += 2 * slot_bytes_;
        CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
    }
    check_blas(cublasCreate(&blas_), "cublasCreate");
    // The RoPE frequencies over a quarter of the head: theta^(-2i / (D/2)), as vision.py computes
    // them in FP32.
    std::vector<float> freq(config_.head_dim() / 4);
    const double rotated = double(config_.head_dim() / 2);
    for (std::size_t i = 0; i < freq.size(); ++i) {
        freq[i] = float(1.0 / std::pow(double(config_.rope_theta), double(2 * i) / rotated));
    }
    inv_freq_ = DeviceBuffer(freq.size() * sizeof(float));
    inv_freq_.copy_from_host(freq.data(), freq.size() * sizeof(float));
}

VisionTower::~VisionTower() {
    for (int i = 0; i < 2; ++i) {
        if (copied_[i] != nullptr) { cudaEventDestroy(copied_[i]); }
        if (used_[i] != nullptr) { cudaEventDestroy(used_[i]); }
    }
    if (copy_stream_ != nullptr) { cudaStreamDestroy(copy_stream_); }
    if (blas_ != nullptr) { cublasDestroy(blas_); }
}

const void* VisionTower::own(const void* host, std::size_t bytes) {
    DeviceBuffer& buffer = owned_.emplace_back(bytes);
    buffer.copy_from_host(host, bytes);
    owned_bytes_ += bytes;
    return buffer.p;
}

std::size_t VisionTower::workspace_bytes() const {
    return scratch_bytes(config_, max_patches_) + (std::size_t(4) << 20);
}

std::uint32_t VisionTower::aligner_rows(int grid_h) const noexcept {
    return std::uint32_t((grid_h + int(config_.downsample_ratio) - 1) / int(config_.downsample_ratio));
}

std::uint32_t VisionTower::aligner_columns(int grid_w) const noexcept {
    return std::uint32_t((grid_w + int(config_.downsample_ratio) - 1) / int(config_.downsample_ratio));
}

void VisionTower::gemm(const __nv_bfloat16* x, int tokens, const Matrix& w, float* out, int ldc,
                       cudaStream_t stream) const {
    check_blas(cublasSetStream(blas_, stream), "cublasSetStream");
    const float alpha = 1.0f, beta = 0.0f;
    check_blas(cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, w.n, tokens, w.k, &alpha, w.data,
                            CUDA_R_16BF, w.k, x, CUDA_R_16BF, w.k, &beta, out, CUDA_R_32F, ldc,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
               "cublasGemmEx");
}

VisionTower::Layer VisionTower::staged(std::size_t i, int slot) const {
    Layer layer  = layers_[i];
    auto* cursor = static_cast<std::byte*>(slots_[slot].p);
    for (Matrix* m : {&layer.qkv, &layer.output, &layer.gate, &layer.up, &layer.down}) {
        m->data = cursor;
        cursor += round_up(m->bytes());
    }
    return layer;
}

void VisionTower::upload(std::size_t i, int slot) const {
    // The slot's last user must be done with it; its next user waits for the upload.
    CUDA_CHECK(cudaStreamWaitEvent(copy_stream_, used_[slot]));
    const Layer& from = layers_[i];
    const Layer to    = staged(i, slot);
    const Matrix* src[] = {&from.qkv, &from.output, &from.gate, &from.up, &from.down};
    const Matrix* dst[] = {&to.qkv, &to.output, &to.gate, &to.up, &to.down};
    for (int m = 0; m < 5; ++m) {
        CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(dst[m]->data), src[m]->data, src[m]->bytes(),
                                   cudaMemcpyHostToDevice, copy_stream_));
    }
    CUDA_CHECK(cudaEventRecord(copied_[slot], copy_stream_));
}

void VisionTower::encode(const __nv_bfloat16* patches, int grid_w, int grid_h, __nv_bfloat16* out,
                         WorkspaceArena& workspace, cudaStream_t stream) const {
    const int n = grid_w * grid_h;
    if (grid_w <= 0 || grid_h <= 0 || std::uint32_t(n) > max_patches_) {
        throw std::invalid_argument("deepseek_v41 vision: invalid image grid");
    }
    const int h = int(config_.hidden_size), width = int(config_.intermediate_size);
    const int heads = int(config_.num_attention_heads), d = int(config_.head_dim());
    const int big = int(config_.projection_dim), r = int(config_.downsample_ratio);
    const float eps = config_.rms_norm_eps;
    const int rows = int(aligner_rows(grid_h)), columns = int(aligner_columns(grid_w));
    const int cells = rows * columns;
    if (std::uint32_t(cells) > std::min(max_patches_, config_.max_image_tokens)) {
        throw std::invalid_argument("deepseek_v41 vision: the image exceeds its span budget");
    }
    auto scope = workspace.scope();
    const auto bf16 = [&](std::size_t count) {
        return static_cast<__nv_bfloat16*>(workspace.alloc_bytes(count * 2).data);
    };
    const auto f32 = [&](std::size_t count) {
        return static_cast<float*>(workspace.alloc_bytes(count * 4).data);
    };
    float* residual    = f32(std::size_t(n) * h);
    __nv_bfloat16* x   = bf16(std::size_t(n) * h);
    {
        auto embed_scope = workspace.scope();
        float* projected = f32(std::size_t(n) * h);
        gemm(patches, n, patch_embedding_, projected, h, stream);
        vk::residual(residual, projected, patch_bias_, true, n, h, layers_.front().attention_norm,
                     eps, x, stream);
    }
    const int slice    = attention_rows(std::size_t(n), std::size_t(heads));
    const float scale  = 1.0f / std::sqrt(float(d));
    const float one = 1.0f, zero = 0.0f;
    if (streamed_) { upload(0, 0); }
    for (std::size_t li = 0; li < layers_.size(); ++li) {
        const int slot = int(li % 2);
        if (streamed_) {
            // The next block uploads while this one computes.
            if (li + 1 < layers_.size()) { upload(li + 1, 1 - slot); }
            CUDA_CHECK(cudaStreamWaitEvent(stream, copied_[slot]));
        }
        const Layer layer = streamed_ ? staged(li, slot) : layers_[li];
        {
            auto attention_scope = workspace.scope();
            float* qkv           = f32(std::size_t(n) * 3 * h);
            gemm(x, n, layer.qkv, qkv, 3 * h, stream);
            __nv_bfloat16* rotated = bf16(std::size_t(n) * 3 * h);
            vk::rotate_qkv(qkv, layer.qkv_bias, n, grid_w, h, d,
                           static_cast<const float*>(inv_freq_.p), rotated, stream);
            __nv_bfloat16* attended = bf16(std::size_t(n) * h);
            float* scores           = f32(std::size_t(heads) * slice * n);
            __nv_bfloat16* p        = bf16(std::size_t(heads) * slice * n);
            check_blas(cublasSetStream(blas_, stream), "cublasSetStream");
            for (int first = 0; first < n; first += slice) {
                const int m = std::min(slice, n - first);
                // scores[head][i][j] = scale q_i . k_j over the head's D values.
                check_blas(cublasGemmStridedBatchedEx(
                               blas_, CUBLAS_OP_T, CUBLAS_OP_N, n, m, d, &scale, rotated + h,
                               CUDA_R_16BF, 3 * h, d, rotated + std::size_t(first) * 3 * h,
                               CUDA_R_16BF, 3 * h, d, &zero, scores, CUDA_R_32F, n,
                               std::int64_t(m) * n, heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                           "cublasGemmStridedBatchedEx (scores)");
                vk::softmax_rows(scores, heads * m, n, p, stream);
                // attended[first + i][head * D + c] = sum_j p[head][i][j] v_j[c].
                check_blas(cublasGemmStridedBatchedEx(
                               blas_, CUBLAS_OP_N, CUBLAS_OP_N, d, m, n, &one, rotated + 2 * h,
                               CUDA_R_16BF, 3 * h, d, p, CUDA_R_16BF, n, std::int64_t(m) * n,
                               &zero, attended + std::size_t(first) * h, CUDA_R_16BF, h, d, heads,
                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                           "cublasGemmStridedBatchedEx (values)");
            }
            gemm(attended, n, layer.output, qkv, h, stream);
            vk::residual(residual, qkv, layer.output_bias, false, n, h, layer.mlp_norm, eps, x,
                         stream);
        }
        {
            auto mlp_scope    = workspace.scope();
            float* gu         = f32(std::size_t(n) * 2 * width);
            gemm(x, n, layer.gate, gu, 2 * width, stream);
            gemm(x, n, layer.up, gu + width, 2 * width, stream);
            __nv_bfloat16* mid = bf16(std::size_t(n) * width);
            vk::swiglu(gu, n, width, mid, stream);
            gemm(mid, n, layer.down, gu, h, stream);
            const float* next = li + 1 < layers_.size() ? layers_[li + 1].attention_norm : final_norm_;
            vk::residual(residual, gu, nullptr, false, n, h, next, eps, x, stream);
        }
        if (streamed_) { CUDA_CHECK(cudaEventRecord(used_[slot], stream)); }
    }
    // The aligner over the normed output x.
    __nv_bfloat16* merged = bf16(std::size_t(cells) * config_.merged_features());
    vk::unfold(x, grid_w, grid_h, h, r, merged, stream);
    float* up          = f32(std::size_t(cells) * big);
    __nv_bfloat16* mid = bf16(std::size_t(cells) * big);
    gemm(merged, cells, aligner_up_, up, big, stream);
    vk::gelu_bias(up, aligner_up_bias_, cells, big, mid, stream);
    gemm(mid, cells, aligner_out_, up, big, stream);
    vk::span(up, aligner_out_bias_, image_start_, image_newline_, image_end_, rows, columns, big,
             out, stream);
}

} // namespace ninfer::models::deepseek_v41
