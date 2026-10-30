#include "models/llama/executor.h"
#include "core/weight_view.h"
#include "ninfer/ops/dense_decoder.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <vector>

namespace ninfer::models::llama {
namespace {
constexpr int kPage          = 1024;
constexpr int kStage         = 8192;
constexpr int kAttentionRows = 128;

const float* vector(const Model& m, WeightId id, int count) {
    auto t = weight_tensor(m.weight(id).view, {count});
    if (t.dtype != DType::FP32) throw std::invalid_argument("Llama vector must be FP32");
    return static_cast<const float*>(t.data);
}

Weight matrix(const Model& m, WeightId id) {
    return ops::prepare_linear_weight(m.input(id)).weight;
}

std::int64_t row_bytes(const Weight& w) {
    auto b = gguf_block_shape(w.qtype);
    return std::int64_t(w.k / b.elements) * b.bytes;
}

struct Layer {
    Weight q, k, v, o, gate, up, down;
    const float *input_norm = nullptr, *ffn_norm = nullptr;
    // A prefix sized to free VRAM stays on device. Remaining pages stream through one
    // bounded transfer buffer; host pages are allocated only when a sequence reaches them.
    DeviceBuffer resident;
    std::vector<std::unique_ptr<std::byte[]>> pages;
};
} // namespace

struct Executor::Impl {
    const Model& model;
    ExecutorOptions options;
    cudaStream_t stream = nullptr;
    int h, hq, hk, vocab, width, layers_count, kv;
    std::atomic<std::uint64_t> host_bytes{0};
    std::uint32_t pos = 0, verified_base = 0, verified_width = 0;
    Weight embedding, head;
    const float* final_norm;
    std::vector<Layer> layers;
    std::unique_ptr<WorkspaceArena> workspace;
    DeviceBuffer ids, positions, logit_buffer, segment;
    std::unique_ptr<PinnedHostBuffer> updates, upload;
    std::size_t plane_bytes, page_bytes, update_plane, resident_plane_bytes = 0, resident_bytes = 0;
    int resident_slots = 0;

    Impl(const Model& m, DeviceContext&, ExecutorOptions o)
        : model(m), options(o), h(m.config().hidden_size), hq(m.config().num_attention_heads),
          hk(m.config().num_key_value_heads), vocab(m.config().vocab_size),
          width(m.config().intermediate_size), layers_count(m.config().num_hidden_layers),
          kv(hk * 128), embedding(native_weight(m.weight(m.weights().token_embedding).view)),
          head(matrix(m, m.weights().output_head)),
          final_norm(vector(m, m.weights().final_norm, h)) {
        if (o.sequences != 1 || o.max_context == 0 ||
            o.max_context > m.config().max_position_embeddings || o.prefill_chunk == 0 ||
            o.prefill_chunk > 512 || o.draft_tokens > 7)
            throw std::invalid_argument(
                "Llama: one sequence, chunks 1..512 and context within native capacity required");
        plane_bytes  = std::size_t(kPage) * kv * 2;
        page_bytes   = 2 * plane_bytes;
        update_plane = std::size_t(o.prefill_chunk) * kv * 2;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        try {
            workspace    = std::make_unique<WorkspaceArena>(256ull << 20);
            ids          = DeviceBuffer(o.prefill_chunk * 4);
            positions    = DeviceBuffer(o.prefill_chunk * 4);
            logit_buffer = DeviceBuffer(std::size_t(kMaxLogitRows) * vocab * 2);
            segment      = DeviceBuffer(page_bytes * (kStage / kPage));
            updates =
                std::make_unique<PinnedHostBuffer>(std::size_t(layers_count) * 2 * update_plane);
            upload = std::make_unique<PinnedHostBuffer>(page_bytes * (kStage / kPage));
            std::size_t free_bytes = 0, total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            constexpr std::size_t reserve = 256ull << 20;
            const std::size_t page_cost   = page_bytes * layers_count;
            const std::size_t fit_pages =
                free_bytes > reserve ? (free_bytes - reserve) / page_cost : 0;
            const std::size_t requested = (std::size_t(o.max_context) + kPage - 1) / kPage;
            std::size_t pages           = std::min(fit_pages, requested);
            if (o.resident_tokens) {
                if (o.resident_tokens % kPage)
                    throw std::invalid_argument(
                        "Llama resident KV must use whole 1024-token pages");
                pages = o.resident_tokens / kPage;
                if (pages > fit_pages || pages > requested)
                    throw std::invalid_argument(
                        "Llama resident KV exceeds context or free device memory");
            }
            if (!pages)
                throw std::runtime_error(
                    "Llama: insufficient device memory for resident KV and 256 MiB headroom");
            resident_slots          = int(pages * kPage);
            options.resident_tokens = resident_slots;
            resident_plane_bytes    = std::size_t(resident_slots) * kv * 2;
            resident_bytes          = 2 * resident_plane_bytes;
            for (const auto& w : m.weights().layers) {
                Layer p{.q          = matrix(m, w.attention.query),
                        .k          = matrix(m, w.attention.key),
                        .v          = matrix(m, w.attention.value),
                        .o          = matrix(m, w.attention.output),
                        .gate       = matrix(m, w.mlp_gate),
                        .up         = matrix(m, w.mlp_up),
                        .down       = matrix(m, w.mlp_down),
                        .input_norm = vector(m, w.input_norm, h),
                        .ffn_norm   = vector(m, w.ffn_norm, h)};
                p.resident = DeviceBuffer(resident_bytes);
                p.resident.fill(0);
                layers.push_back(std::move(p));
            }
        } catch (...) {
            cudaStreamDestroy(stream);
            stream = nullptr;
            throw;
        }
    }

    ~Impl() {
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
    }

    void sequence(std::uint32_t s) const {
        if (s != 0) throw std::out_of_range("Llama sequence");
    }

    void project(const Tensor& x, const Weight& w, float* out, int stride, int row = 0,
                 bool add = false) {
        auto scope = workspace->scope();
        ops::detail::GgufProduct p;
        p.weight   = &w;
        p.f32      = out;
        p.f32_rows = stride;
        p.row      = row;
        p.epilogue = add ? ops::detail::GgufEpilogue::Accumulate : ops::detail::GgufEpilogue::Store;
        ops::detail::gguf_project(x, {&p, 1}, *workspace, stream);
    }

    void norm(const float* x, const float* weight, int t, Tensor& out) {
        ops::rmsnorm_f32(x, weight, h, t, model.config().rms_norm_eps,
                         static_cast<__nv_bfloat16*>(out.data), stream);
    }

    std::byte* page(Layer& l, std::size_t b) {
        if (l.pages.size() <= b) l.pages.resize(b + 1);
        if (!l.pages[b]) {
            l.pages[b] = std::make_unique<std::byte[]>(page_bytes);
            host_bytes.fetch_add(page_bytes, std::memory_order_relaxed);
        }
        return l.pages[b].get();
    }

    void attend(Layer& l, const float* q, const __half* k, const __half* v, int t, Tensor& out) {
        auto scope = workspace->scope();
        auto* state =
            static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * hq * 130 * 4).data);
        auto* partial  = workspace
                             ->alloc_bytes(ops::segment_attention_workspace_bytes(
                                 std::min(t, kAttentionRows), hq, kStage))
                             .data;
        const auto run = [&](const __half* keys, const __half* values, int first_key, int count,
                             int stride, bool initialize, bool finish) {
            for (int row = 0; row < t; row += kAttentionRows) {
                const int rows = std::min(kAttentionRows, t - row);
                ops::segment_attention_f16(
                    q + std::size_t(row) * hq * 128, keys, values,
                    static_cast<const int*>(positions.p) + row, rows, hq, hk, first_key, count,
                    stride, state + std::size_t(row) * hq * 130, partial, initialize, finish,
                    static_cast<__nv_bfloat16*>(out.data) + std::size_t(row) * hq * 128, stream);
            }
        };
        bool first                = true;
        const auto resident_count = std::min<std::uint32_t>(pos, resident_slots);
        const auto* resident      = static_cast<const std::byte*>(l.resident.p);
        for (std::uint32_t at = 0; at < resident_count; at += kStage) {
            const int n = std::min<std::uint32_t>(kStage, resident_count - at);
            run(reinterpret_cast<const __half*>(resident) + std::size_t(at) * 128,
                reinterpret_cast<const __half*>(resident + resident_plane_bytes) +
                    std::size_t(at) * 128,
                at, n, resident_slots, first, false);
            first = false;
        }
        const std::size_t stage_plane = std::size_t(kStage) * kv * 2;
        for (std::uint32_t at = resident_slots; at < pos; at += kStage) {
            const int n = std::min<std::uint32_t>(kStage, pos - at);
            // A whole transfer tile amortizes stream/driver overhead. Query subtiles keep the
            // split-softmax workspace bounded independently of the projection's prompt chunk.
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto* host = static_cast<std::byte*>(upload->data());
            for (int plane = 0; plane < 2; ++plane)
                for (int head = 0; head < hk; ++head) {
                    int done = 0;
                    while (done < n) {
                        const std::uint32_t absolute = at + done;
                        const int slot = absolute % kPage, take = std::min(n - done, kPage - slot);
                        const auto* src = page(l, absolute / kPage) + plane * plane_bytes +
                                          (std::size_t(head) * kPage + slot) * 256;
                        std::memcpy(host + plane * stage_plane +
                                        (std::size_t(head) * kStage + done) * 256,
                                    src, std::size_t(take) * 256);
                        done += take;
                    }
                }
            CUDA_CHECK(cudaMemcpy2DAsync(segment.p, kStage * 256, host, kStage * 256, n * 256,
                                         2 * hk, cudaMemcpyHostToDevice, stream));
            auto* base = static_cast<const std::byte*>(segment.p);
            run(reinterpret_cast<const __half*>(base),
                reinterpret_cast<const __half*>(base + stage_plane), at, n, kStage, first, false);
            first = false;
        }
        run(k, v, pos, t, t, first, true);
    }

    void retain_updates(int t) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int i = 0; i < layers_count; ++i) {
            auto* src =
                static_cast<const std::byte*>(updates->data()) + std::size_t(i) * 2 * update_plane;
            for (int plane = 0; plane < 2; ++plane)
                for (int head = 0; head < hk; ++head) {
                    int done = 0;
                    while (done < t) {
                        auto absolute = pos + done;
                        int slot = absolute % kPage, n = std::min(t - done, kPage - slot);
                        auto* dst = page(layers[i], absolute / kPage) + plane * plane_bytes +
                                    (std::size_t(head) * kPage + slot) * 256;
                        std::memcpy(
                            dst, src + plane * update_plane + (std::size_t(head) * t + done) * 256,
                            std::size_t(n) * 256);
                        done += n;
                    }
                }
        }
    }

    void run(std::span<const int> tokens, std::uint32_t logit_rows) {
        const int t = tokens.size();
        if (t <= 0 || t > int(options.prefill_chunk) || logit_rows > std::uint32_t(t) ||
            logit_rows > kMaxLogitRows || std::uint64_t(pos) + t > options.max_context)
            throw std::invalid_argument("Llama forward extents");
        for (auto id : tokens)
            if (id < 0 || id >= vocab) throw std::invalid_argument("Llama token outside embedding");
        // cudaMemcpy (not Async) completes the borrowed host inputs before returning.
        CUDA_CHECK(cudaMemcpy(ids.p, tokens.data(), t * 4, cudaMemcpyHostToDevice));
        std::vector<int> p(t);
        for (int j = 0; j < t; ++j) p[j] = pos + j;
        CUDA_CHECK(cudaMemcpy(positions.p, p.data(), t * 4, cudaMemcpyHostToDevice));
        workspace->reset();
        auto* residual = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * h * 4).data);
        Tensor x       = workspace->alloc(DType::BF16, {h, t});
        ops::gguf::dequantize_rows(ops::detail::gguf_type(embedding.qtype), embedding.qdata,
                                   row_bytes(embedding), h, static_cast<const int*>(ids.p), t,
                                   residual, h, stream);
        for (int i = 0; i < layers_count; ++i) {
            auto scope = workspace->scope();
            auto& l    = layers[i];
            norm(residual, l.input_norm, t, x);
            int qkv_width = (hq + 2 * hk) * 128;
            auto* qkv =
                static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * qkv_width * 4).data);
            project(x, l.q, qkv, qkv_width);
            project(x, l.k, qkv, qkv_width, hq * 128);
            project(x, l.v, qkv, qkv_width, (hq + hk) * 128);
            auto* q =
                static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * hq * 128 * 4).data);
            auto* k = static_cast<__half*>(workspace->alloc_bytes(std::size_t(t) * kv * 2).data);
            auto* v = static_cast<__half*>(workspace->alloc_bytes(std::size_t(t) * kv * 2).data);
            ops::rotary_kv_f16(qkv, static_cast<const int*>(positions.p), t, hq, hk,
                               model.config().rope_theta, q, k, v, stream);
            Tensor attended = workspace->alloc(DType::BF16, {hq * 128, t});
            attend(l, q, k, v, t, attended);
            project(attended, l.o, residual, h, 0, true);
            norm(residual, l.ffn_norm, t, x);
            Tensor middle = workspace->alloc(DType::BF16, {width, t});
            {
                auto scope = workspace->scope();
                ops::detail::gguf_swiglu(x, l.gate, &l.up, middle, *workspace, stream);
            }
            project(middle, l.down, residual, h, 0, true);
            auto* update =
                static_cast<std::byte*>(updates->data()) + std::size_t(i) * 2 * update_plane;
            CUDA_CHECK(cudaMemcpyAsync(update, k, std::size_t(t) * kv * 2, cudaMemcpyDeviceToHost,
                                       stream));
            CUDA_CHECK(cudaMemcpyAsync(update + update_plane, v, std::size_t(t) * kv * 2,
                                       cudaMemcpyDeviceToHost, stream));
            if (pos < std::uint32_t(resident_slots)) {
                int n          = std::min<std::uint32_t>(t, resident_slots - pos);
                auto* resident = static_cast<std::byte*>(l.resident.p);
                CUDA_CHECK(cudaMemcpy2DAsync(resident + pos * 256, resident_slots * 256, k, t * 256,
                                             n * 256, hk, cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaMemcpy2DAsync(resident + resident_plane_bytes + pos * 256,
                                             resident_slots * 256, v, t * 256, n * 256, hk,
                                             cudaMemcpyDeviceToDevice, stream));
            }
        }
        if (logit_rows) {
            norm(residual, final_norm, t, x);
            for (int at = t - logit_rows; at < t; at += 8) {
                auto scope = workspace->scope();
                int n      = std::min(8, t - at);
                Tensor last(static_cast<__nv_bfloat16*>(x.data) + std::size_t(at) * h, DType::BF16,
                            {h, n});
                auto* f =
                    static_cast<float*>(workspace->alloc_bytes(std::size_t(n) * vocab * 4).data);
                project(last, head, f, vocab);
                Tensor source(f, DType::FP32, {vocab, n});
                Tensor destination(static_cast<__nv_bfloat16*>(logit_buffer.p) +
                                       std::size_t(at - (t - logit_rows)) * vocab,
                                   DType::BF16, {vocab, n});
                ops::cast_fp32_to_bf16(source, destination, stream);
            }
        }
        retain_updates(t);
        pos += t;
    }

    void rebuild_resident() {
        const auto count = std::min<std::uint32_t>(pos, resident_slots);
        for (auto& l : layers)
            for (std::uint32_t at = 0; at < count; at += kPage) {
                const int n = std::min<std::uint32_t>(kPage, count - at);
                auto* src   = page(l, at / kPage);
                auto* dst   = static_cast<std::byte*>(l.resident.p);
                for (int plane = 0; plane < 2; ++plane)
                    CUDA_CHECK(cudaMemcpy2DAsync(
                        dst + plane * resident_plane_bytes + std::size_t(at) * 256,
                        resident_slots * 256, src + plane * plane_bytes, kPage * 256, n * 256, hk,
                        cudaMemcpyHostToDevice, stream));
            }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
};

Executor::Executor(const Model& m, DeviceContext& d, ExecutorOptions o)
    : impl_(std::make_unique<Impl>(m, d, o)) {}

Executor::~Executor() = default;

const ExecutorOptions& Executor::options() const noexcept { return impl_->options; }

ExecutorMemory Executor::memory() const noexcept {
    ExecutorMemory m;
    m.kv_bytes        = impl_->resident_bytes * impl_->layers.size();
    m.kv_backed_bytes = m.kv_bytes;
    m.workspace_bytes = impl_->workspace->capacity() + impl_->segment.bytes + impl_->ids.bytes +
                        impl_->positions.bytes + impl_->logit_buffer.bytes;
    m.host_kv_bytes   = impl_->host_bytes.load(std::memory_order_relaxed);
    return m;
}

void Executor::reset(std::uint32_t s) {
    impl_->sequence(s);
    impl_->pos            = 0;
    impl_->verified_width = 0;
    for (auto& l : impl_->layers) l.pages.clear();
    impl_->host_bytes.store(0, std::memory_order_relaxed);
}

std::uint32_t Executor::position(std::uint32_t s) const {
    impl_->sequence(s);
    return impl_->pos;
}

void Executor::warm_up() {
    const int zero = 0;
    forward(0, {&zero, 1}, 1);
    reset(0);
}

void Executor::forward(std::uint32_t s, std::span<const int> t, std::uint32_t rows) {
    impl_->sequence(s);
    impl_->run(t, rows);
}

void Executor::decode(std::span<const std::uint32_t> s, std::span<const int> t) {
    if (s.size() != 1 || t.size() != 1) throw std::invalid_argument("Llama decode batch");
    forward(s[0], t, 1);
}

Tensor Executor::logits(std::uint32_t rows) const {
    if (!rows || rows > kMaxLogitRows) throw std::invalid_argument("Llama logits rows");
    return Tensor(impl_->logit_buffer.p, DType::BF16, {impl_->vocab, int(rows)});
}

std::uint32_t Executor::draft_tokens() const noexcept { return impl_->options.draft_tokens; }

cudaStream_t Executor::head_stream() const noexcept { return impl_->stream; }

void Executor::snapshot(std::uint32_t s, SequenceSnapshot& out) { out.position = position(s); }

void Executor::restore(std::uint32_t s, const SequenceSnapshot& from) {
    impl_->sequence(s);
    if (from.position > impl_->pos) throw std::invalid_argument("Llama restore frontier");
    impl_->pos = from.position;
}

void Executor::verify(std::span<const std::uint32_t> s, std::span<const int> t,
                      std::span<const std::uint32_t> widths) {
    if (s.size() != 1 || (!widths.empty() && (widths.size() != 1 || widths[0] != t.size())) ||
        t.empty() || t.size() > 1 + draft_tokens())
        throw std::invalid_argument("Llama verification extents");
    impl_->verified_base = position(s[0]);
    impl_->run(t, t.size());
    impl_->verified_width = t.size();
    impl_->pos            = impl_->verified_base;
}

void Executor::commit(std::span<const std::uint32_t> s, std::span<const std::uint32_t> kept) {
    if (s.size() != 1 || kept.size() != 1 || !kept[0] || kept[0] > impl_->verified_width)
        throw std::invalid_argument("Llama commit extent");
    impl_->sequence(s[0]);
    impl_->pos            = impl_->verified_base + kept[0];
    impl_->verified_width = 0;
}

void Executor::forward_branches(std::uint32_t s, std::span<const int> tokens,
                                std::span<const std::uint32_t> lengths) {
    impl_->sequence(s);
    if (lengths.empty() || lengths.size() > kMaxLogitRows)
        throw std::invalid_argument("Llama branches");
    std::size_t sum = 0;
    for (auto n : lengths) {
        if (!n || n > options().prefill_chunk) throw std::invalid_argument("Llama branch length");
        sum += n;
    }
    if (sum != tokens.size()) throw std::invalid_argument("Llama branch tokens");
    DeviceBuffer rows(lengths.size() * impl_->vocab * 2);
    auto base      = impl_->pos;
    std::size_t at = 0;
    try {
        for (std::size_t b = 0; b < lengths.size(); ++b) {
            impl_->pos = base;
            impl_->run(tokens.subspan(at, lengths[b]), 1);
            at += lengths[b];
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(rows.p) + b * impl_->vocab * 2,
                                       impl_->logit_buffer.p, impl_->vocab * 2,
                                       cudaMemcpyDeviceToDevice, impl_->stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(impl_->logit_buffer.p, rows.p, rows.bytes,
                                   cudaMemcpyDeviceToDevice, impl_->stream));
        CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
        impl_->pos = base;
    } catch (...) {
        impl_->pos = base;
        throw;
    }
}

void Executor::forward_prompt_branches(std::uint32_t s, std::span<const int> prompt,
                                       std::span<const int> tokens,
                                       std::span<const std::uint32_t> lengths) {
    if (!prompt.empty()) forward(s, prompt, 0);
    forward_branches(s, tokens, lengths);
}

std::uint64_t Executor::image_bytes(std::uint32_t pos) const {
    if (pos > options().max_context) throw std::invalid_argument("Llama image position");
    return std::uint64_t(pos) * impl_->kv * 2 * 2 * impl_->layers_count;
}

namespace {
void copy_pages(std::span<std::byte* const> pages, std::size_t page_bytes, std::uint64_t& offset,
                void* buffer, std::size_t count, bool save) {
    if (!page_bytes || offset + count > std::uint64_t(pages.size()) * page_bytes)
        throw std::invalid_argument("Llama image capacity");
    auto* data = static_cast<std::byte*>(buffer);
    while (count) {
        std::size_t in = offset % page_bytes, n = std::min(count, page_bytes - in);
        auto* p = pages[offset / page_bytes] + in;
        if (save)
            std::memcpy(p, data, n);
        else
            std::memcpy(data, p, n);
        data += n;
        offset += n;
        count -= n;
    }
}
} // namespace

SequenceImage Executor::save_image(std::uint32_t s, const SequenceSnapshot* at,
                                   std::span<std::byte* const> pages, std::size_t bytes) {
    impl_->sequence(s);
    auto p = at ? at->position : impl_->pos;
    if (p > impl_->pos) throw std::invalid_argument("Llama image frontier");
    std::uint64_t offset = 0;
    for (auto& l : impl_->layers)
        for (std::uint32_t start = 0; start < p; start += kPage) {
            auto* data = impl_->page(l, start / kPage);
            int n      = std::min<std::uint32_t>(kPage, p - start);
            for (int plane = 0; plane < 2; ++plane)
                for (int head = 0; head < impl_->hk; ++head)
                    copy_pages(pages, bytes, offset,
                               data + plane * impl_->plane_bytes + std::size_t(head) * kPage * 256,
                               n * 256, true);
        }
    return {.position = p};
}

void Executor::load_image(std::uint32_t s, const SequenceImage& image,
                          std::span<std::byte* const> pages, std::size_t bytes) {
    impl_->sequence(s);
    (void)image_bytes(image.position);
    reset(s);
    std::uint64_t offset = 0;
    for (auto& l : impl_->layers)
        for (std::uint32_t start = 0; start < image.position; start += kPage) {
            auto* data = impl_->page(l, start / kPage);
            int n      = std::min<std::uint32_t>(kPage, image.position - start);
            for (int plane = 0; plane < 2; ++plane)
                for (int head = 0; head < impl_->hk; ++head)
                    copy_pages(pages, bytes, offset,
                               data + plane * impl_->plane_bytes + std::size_t(head) * kPage * 256,
                               n * 256, false);
        }
    impl_->pos = image.position;
    impl_->rebuild_resident();
}
} // namespace ninfer::models::llama
