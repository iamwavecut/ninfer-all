#include "models/inkling2/vision.h"

#include "core/device_buffer.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "models/inkling2/vision_kernels.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::models::inkling2 {
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
        throw std::runtime_error(std::string("inkling2 vision: ") + what + " failed (" +
                                 std::to_string(int(status)) + ")");
    }
}

// Query rows per attention slice over n rows.
int attention_rows(std::size_t n, std::size_t heads) {
    return int(std::clamp<std::size_t>(kAttentionBytes / (heads * n * 6), 1, n));
}

// The scratch of one encode of up to `patches` patches, as encode() allocates it: the residual and
// a block's input, then the largest of its phases (each in its own scope).
std::size_t scratch_bytes(const VisionConfig& c, std::size_t patches) {
    const std::size_t n = patches + 1;
    const std::size_t h = c.hidden_size, w = c.intermediate_size, heads = c.num_attention_heads;
    const std::size_t cells = patches / (c.merge_size * c.merge_size);
    const std::size_t rows  = std::size_t(attention_rows(n, heads));
    const std::size_t embed     = round_up(patches * h * 4);
    const std::size_t attention = round_up(n * 3 * h * 4) + round_up(n * 3 * h * 2) +
                                  round_up(n * h * 2) + round_up(heads * rows * n * 4) +
                                  round_up(heads * rows * n * 2);
    const std::size_t mlp       = round_up(n * w * 4) + round_up(n * w * 2);
    const std::size_t big       = c.projection_hidden;
    const std::size_t projector = round_up(cells * c.merged_features() * 2) +
                                  round_up(cells * std::max<std::size_t>(big, c.text_hidden_size) * 4) +
                                  round_up(cells * big * 2);
    return round_up(n * h * 4) + round_up(n * h * 2) +
           std::max({embed, attention, mlp, projector});
}

} // namespace

VisionTower::VisionTower(const Model& model, std::uint32_t max_patches)
    : config_(*model.vision_config()), max_patches_(max_patches),
      streamed_(model.vision_placement() == VisionPlacement::Pinned) {
    if (!model.vision() || (model.vision_placement() != VisionPlacement::Device && !streamed_)) {
        throw std::invalid_argument("inkling2 vision: the model holds no vision tower");
    }
    if (max_patches == 0) { throw std::invalid_argument("inkling2 vision: no patch capacity"); }
    if (config_.merge_size != 2) {
        throw std::invalid_argument("inkling2 vision: the tower merges 2 x 2 cells");
    }
    const VisionWeights& w = *model.vision();
    const int h = int(config_.hidden_size), width = int(config_.intermediate_size);
    const int big = int(config_.projection_hidden), text = int(config_.text_hidden_size);
    const auto matrix = [&](WeightId id, int n, int k) {
        const BoundWeight& bound = model.weight(id);
        const QType format = contiguous_weight_region(bound.view).parent->geometry.format;
        if (is_gguf(format)) {
            const Weight weight = native_weight(bound.view);
            ops::detail::require_gguf(weight, bound.name.c_str());
            if (weight.n != n || weight.k != k) {
                throw std::invalid_argument(bound.name + ": stored rows differ from the tower's");
            }
            return Matrix{weight.qdata, n, k, format};
        }
        const Tensor tensor = weight_tensor(bound.view, {k, n});
        if (tensor.dtype != DType::BF16) {
            throw std::invalid_argument(bound.name +
                                        ": the tower runs BF16 matrices or GGUF block rows");
        }
        return Matrix{tensor.data, n, k, QType::BF16};
    };
    patch_embedding_ = matrix(w.patch_embedding, h, int(config_.patch_features()));
    projector_up_    = matrix(w.projector_up, big, int(config_.merged_features()));
    projector_out_   = matrix(w.projector_out, text, big);
    if (streamed_) {
        // The patch projection and the projector stay resident: copies, which the same cuBLAS
        // calls multiply as the resident tower does, so both give the same rows.
        for (Matrix* m : {&patch_embedding_, &projector_up_, &projector_out_}) {
            m->data = own(*m);
        }
    }
    class_embedding_ = fp32(model, w.class_embedding, h);
    class_position_  = fp32(model, w.class_position, h);
    positions_       = fp32(model, w.positions, std::int32_t(config_.posemb_grid * config_.posemb_grid) * h);
    pre_norm_        = fp32(model, w.pre_norm, h);
    pre_norm_bias_   = fp32(model, w.pre_norm_bias, h);
    for (const VisionLayerWeights& lw : w.layers) {
        Layer layer;
        layer.query               = matrix(lw.query, h, h);
        layer.key                 = matrix(lw.key, h, h);
        layer.value               = matrix(lw.value, h, h);
        layer.output              = matrix(lw.output, h, h);
        layer.up                  = matrix(lw.up, width, h);
        layer.down                = matrix(lw.down, h, width);
        layer.attention_norm      = fp32(model, lw.attention_norm, h);
        layer.attention_norm_bias = fp32(model, lw.attention_norm_bias, h);
        layer.query_bias          = fp32(model, lw.query_bias, h);
        layer.key_bias            = fp32(model, lw.key_bias, h);
        layer.value_bias          = fp32(model, lw.value_bias, h);
        layer.output_bias         = fp32(model, lw.output_bias, h);
        layer.attention_scale     = fp32(model, lw.attention_scale, h);
        layer.mlp_norm            = fp32(model, lw.mlp_norm, h);
        layer.mlp_norm_bias       = fp32(model, lw.mlp_norm_bias, h);
        layer.up_bias             = fp32(model, lw.up_bias, width);
        layer.down_bias           = fp32(model, lw.down_bias, h);
        layer.mlp_scale           = fp32(model, lw.mlp_scale, h);
        layers_.push_back(layer);
    }
    if (streamed_) {
        // A slot holds the largest block.
        for (const Layer& layer : layers_) {
            std::size_t bytes = 0;
            for (const Matrix* m :
                 {&layer.query, &layer.key, &layer.value, &layer.output, &layer.up, &layer.down}) {
                bytes += round_up(m->bytes());
            }
            slot_bytes_ = std::max(slot_bytes_, bytes);
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
    // The RoPE frequencies of each half of a head: theta^(-2i / (D/2)), i < D/4.
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

const void* VisionTower::own(const Matrix& host) {
    DeviceBuffer& buffer = owned_.emplace_back(host.bytes());
    buffer.copy_from_host(host.data, host.bytes());
    owned_bytes_ += host.bytes();
    return buffer.p;
}

std::size_t VisionTower::workspace_bytes() const {
    // A GGUF matrix's BF16 rows come on top of the phase that multiplies it.
    std::size_t decoded = 0;
    const auto add = [&](const Matrix& m) {
        if (m.blocks()) { decoded = std::max(decoded, round_up(std::size_t(m.n) * m.k * 2)); }
    };
    for (const Matrix* m : {&patch_embedding_, &projector_up_, &projector_out_}) { add(*m); }
    for (const Layer& layer : layers_) {
        for (const Matrix* m :
             {&layer.query, &layer.key, &layer.value, &layer.output, &layer.up, &layer.down}) {
            add(*m);
        }
    }
    return scratch_bytes(config_, max_patches_) + decoded + (std::size_t(4) << 20);
}

void VisionTower::gemm(const __nv_bfloat16* x, int tokens, const Matrix& w, float* out, int ldc,
                       WorkspaceArena& workspace, cudaStream_t stream) const {
    auto decode_scope = workspace.scope();
    const void* rows  = w.data;
    if (w.blocks()) {
        // GGUF rows decode to BF16 and multiply as a BF16 matrix does, so the input stays BF16:
        // ggml's q8_1 input (the GGUF product's) moved the tower's rows by 2.7% RMS.
        auto* decoded = static_cast<__nv_bfloat16*>(
            workspace.alloc_bytes(std::size_t(w.n) * w.k * 2).data);
        ops::gguf::dequantize_rows(ops::detail::gguf_type(w.format), w.data,
                                   std::int64_t(w.row_bytes()), w.k, nullptr, w.n, decoded, w.k,
                                   stream);
        rows = decoded;
    }
    check_blas(cublasSetStream(blas_, stream), "cublasSetStream");
    const float alpha = 1.0f, beta = 0.0f;
    check_blas(cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, w.n, tokens, w.k, &alpha, rows,
                            CUDA_R_16BF, w.k, x, CUDA_R_16BF, w.k, &beta, out, CUDA_R_32F, ldc,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
               "cublasGemmEx");
}

VisionTower::Layer VisionTower::staged(std::size_t i, int slot) const {
    Layer layer  = layers_[i];
    auto* cursor = static_cast<std::byte*>(slots_[slot].p);
    for (Matrix* m : {&layer.query, &layer.key, &layer.value, &layer.output, &layer.up, &layer.down}) {
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
    const Matrix* src[] = {&from.query, &from.key, &from.value, &from.output, &from.up, &from.down};
    const Matrix* dst[] = {&to.query, &to.key, &to.value, &to.output, &to.up, &to.down};
    for (int m = 0; m < 6; ++m) {
        CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(dst[m]->data), src[m]->data, src[m]->bytes(),
                                   cudaMemcpyHostToDevice, copy_stream_));
    }
    CUDA_CHECK(cudaEventRecord(copied_[slot], copy_stream_));
}

void VisionTower::encode(const __nv_bfloat16* patches, int grid_w, int grid_h, __nv_bfloat16* out,
                         WorkspaceArena& workspace, cudaStream_t stream) const {
    const int patches_n = grid_w * grid_h, n = patches_n + 1;
    if (grid_w <= 0 || grid_h <= 0 || grid_w % 2 != 0 || grid_h % 2 != 0 ||
        std::uint32_t(patches_n) > max_patches_ ||
        span_rows(grid_w, grid_h) > config_.max_image_tokens) {
        throw std::invalid_argument("inkling2 vision: invalid image grid");
    }
    const int h = int(config_.hidden_size), width = int(config_.intermediate_size);
    const int heads = int(config_.num_attention_heads), d = int(config_.head_dim());
    const int big = int(config_.projection_hidden), text = int(config_.text_hidden_size);
    const int cells = int(span_rows(grid_w, grid_h));
    const float eps = config_.layer_norm_eps;
    auto scope = workspace.scope();
    const auto bf16 = [&](std::size_t count) {
        return static_cast<__nv_bfloat16*>(workspace.alloc_bytes(count * 2).data);
    };
    const auto f32 = [&](std::size_t count) {
        return static_cast<float*>(workspace.alloc_bytes(count * 4).data);
    };
    float* residual  = f32(std::size_t(n) * h);
    __nv_bfloat16* x = bf16(std::size_t(n) * h);
    {
        auto embed_scope = workspace.scope();
        float* projected = f32(std::size_t(patches_n) * h);
        gemm(patches, patches_n, patch_embedding_, projected, h, workspace, stream);
        vk::embed(projected, positions_, int(config_.posemb_grid), grid_w, grid_h, class_embedding_,
                  class_position_, h, residual, stream);
    }
    vk::layer_norm(residual, pre_norm_, pre_norm_bias_, n, h, eps, nullptr, stream, residual);
    const int slice   = attention_rows(std::size_t(n), std::size_t(heads));
    const float scale = 1.0f / std::sqrt(float(d));
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
            vk::layer_norm(residual, layer.attention_norm, layer.attention_norm_bias, n, h, eps, x,
                           stream);
            float* qkv = f32(std::size_t(n) * 3 * h);
            gemm(x, n, layer.query, qkv, 3 * h, workspace, stream);
            gemm(x, n, layer.key, qkv + h, 3 * h, workspace, stream);
            gemm(x, n, layer.value, qkv + 2 * h, 3 * h, workspace, stream);
            __nv_bfloat16* rotated = bf16(std::size_t(n) * 3 * h);
            vk::rotate_qkv(qkv, layer.query_bias, layer.key_bias, layer.value_bias, n, grid_w, h, d,
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
            gemm(attended, n, layer.output, qkv, h, workspace, stream);
            vk::residual(residual, qkv, layer.output_bias, layer.attention_scale, n, h, stream);
        }
        {
            auto mlp_scope = workspace.scope();
            vk::layer_norm(residual, layer.mlp_norm, layer.mlp_norm_bias, n, h, eps, x, stream);
            float* up = f32(std::size_t(n) * width);
            gemm(x, n, layer.up, up, width, workspace, stream);
            __nv_bfloat16* mid = bf16(std::size_t(n) * width);
            vk::gelu(up, layer.up_bias, n, width, mid, stream);
            gemm(mid, n, layer.down, up, h, workspace, stream);
            vk::residual(residual, up, layer.down_bias, layer.mlp_scale, n, h, stream);
        }
        if (streamed_) { CUDA_CHECK(cudaEventRecord(used_[slot], stream)); }
    }
    // The projector over the merged cells of the output, the class row left out.
    __nv_bfloat16* merged = bf16(std::size_t(cells) * config_.merged_features());
    vk::merge(residual, grid_w, grid_h, h, merged, stream);
    float* up          = f32(std::size_t(cells) * std::max(big, text));
    __nv_bfloat16* mid = bf16(std::size_t(cells) * big);
    gemm(merged, cells, projector_up_, up, big, workspace, stream);
    vk::gelu(up, nullptr, cells, big, mid, stream);
    gemm(mid, cells, projector_out_, up, text, workspace, stream);
    vk::f32_to_bf16(up, std::int64_t(cells) * text, out, stream);
}

} // namespace ninfer::models::inkling2
