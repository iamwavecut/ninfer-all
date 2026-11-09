#include "models/qwen4_exp/ngram_draft_prefetch.h"

#include "core/device_wait.h"

#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen4_exp {

NgramDraftPrefetch::NgramDraftPrefetch(const RankContext& rank, NgramTableReader& table,
                                     const NgramHashConstants& hash, std::int32_t eos,
                                     std::uint32_t vocab, std::size_t sequences,
                                     std::size_t steps)
    : device_(rank.device), table_(table), hash_(hash), eos_(eos), vocab_(vocab),
      capacity_(sequences), tokens_(2 * sequences * (steps + 1) * sizeof(std::int32_t)), done_(rank) {
    if (sequences == 0 || steps == 0) {
        throw std::invalid_argument("n-gram draft prefetch needs sequences and steps");
    }
    steps_.reserve(steps);
    for (std::size_t i = 0; i < steps; ++i) {
        steps_.push_back({this, i, CudaCompletionEvent(rank)});
    }
    DeviceBinding bind(device_);
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

NgramDraftPrefetch::~NgramDraftPrefetch() {
    // A captured graph launches on the compute stream, so synchronizing only stream_ would not
    // drain its host nodes. No callback may outlive this object or the referenced table.
    try {
        DeviceBinding bind(device_);
        ninfer::wait_for_device();
        cudaStreamDestroy(stream_);
    } catch (...) {}
}

void NgramDraftPrefetch::begin(std::vector<NgramContext> contexts,
                              std::span<const std::int32_t> anchors) {
    if (contexts.empty() || contexts.size() != anchors.size() || contexts.size() > capacity_) {
        throw std::invalid_argument("n-gram draft prefetch batch does not match its contexts");
    }
    contexts_ = std::move(contexts);
    batch_ = contexts_.size();
    rows_.resize(2 * batch_ * hash_.heads());
    failure_ = nullptr;
    issue(anchors);
}

void NgramDraftPrefetch::issue(std::span<const std::int32_t> tokens) {
    for (std::size_t i = 0; i < batch_; ++i) {
        ngram_row_ids(hash_, tokens.subspan(i, 1), eos_, vocab_, contexts_[i],
                       std::span(rows_).subspan(i * hash_.heads(), hash_.heads()));
    }
    table_.prefetch(std::span(rows_).first(batch_ * hash_.heads()));
}

void NgramDraftPrefetch::issue_candidates(const std::int32_t* first, const std::int32_t* second) {
    const auto heads = hash_.heads();
    for (std::size_t i = 0; i < batch_; ++i) {
        auto alternative = contexts_[i];
        ngram_row_ids(hash_, {second + i, 1}, eos_, vocab_, alternative,
                       std::span(rows_).subspan((batch_ + i) * heads, heads));
        ngram_row_ids(hash_, {first + i, 1}, eos_, vocab_, contexts_[i],
                       std::span(rows_).subspan(i * heads, heads));
    }
    table_.prefetch(rows_);
}

void NgramDraftPrefetch::hint(std::span<const std::int32_t> drafts, std::size_t stride,
                              std::size_t steps) {
    if (steps > steps_.size() || drafts.size() < batch_ * stride || steps > stride) {
        throw std::invalid_argument("n-gram draft prefetch: drafts do not match the batch");
    }
    const auto heads = hash_.heads();
    for (std::size_t i = 0; i < steps; ++i) {
        for (std::size_t j = 0; j < batch_; ++j) {
            ngram_row_ids(hash_, drafts.subspan(j * stride + i, 1), eos_, vocab_, contexts_[j],
                          std::span(rows_).subspan(j * heads, heads));
        }
        table_.prefetch(std::span(rows_).first(batch_ * heads));
    }
}

void CUDART_CB NgramDraftPrefetch::prefetch(void* data) noexcept {
    const auto& step = *static_cast<Step*>(data);
    auto& self = *step.owner;
    if (self.failure_) { return; }
    try {
        const auto* ids = static_cast<const std::int32_t*>(self.tokens_.data()) +
                          2 * step.index * self.capacity_;
        self.issue_candidates(ids, ids + self.capacity_);
    } catch (...) { self.failure_ = std::current_exception(); }
}

void NgramDraftPrefetch::enqueue(std::size_t step, const std::int32_t* first,
                                const std::int32_t* second, cudaStream_t compute) {
    auto& entry = steps_.at(step);
    entry.ready.record(compute);
    entry.ready.wait(stream_);
    auto* out = static_cast<std::int32_t*>(tokens_.data()) + 2 * step * capacity_;
    CUDA_CHECK(cudaMemcpyAsync(out, first, batch_ * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(out + capacity_, second, batch_ * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream_));
    // This callback hashes host tokens and issues OS hints; it calls no CUDA API.
    CUDA_CHECK(cudaLaunchHostFunc(stream_, &prefetch, &entry));
}

void NgramDraftPrefetch::begin_verification(std::vector<NgramContext> contexts,
                                           std::span<const std::int32_t> prefix,
                                           std::span<const std::int32_t> widths) {
    std::size_t columns = 0;
    for (const std::int32_t width : widths) {
        if (width < 1 || std::size_t(width) > steps_.size() + 1) { columns = 0; break; }
        columns += std::size_t(width);
    }
    if (contexts.empty() || contexts.size() > capacity_ || widths.size() != contexts.size() ||
        columns == 0 || prefix.size() != columns) {
        throw std::invalid_argument("n-gram verification prefetch prefix does not match its contexts");
    }
    contexts_ = std::move(contexts);
    batch_ = contexts_.size();
    verification_widths_.assign(widths.begin(), widths.end());
    verification_prefix_.assign(prefix.begin(), prefix.end());
    rows_.resize(2 * prefix.size() * hash_.heads());
    failure_ = nullptr;
}

void CUDART_CB NgramDraftPrefetch::prefetch_verification(void* data) noexcept {
    auto& self = *static_cast<NgramDraftPrefetch*>(data);
    try {
        const auto count = self.verification_prefix_.size();
        const auto heads = self.hash_.heads();
        const auto* first = static_cast<const std::int32_t*>(self.tokens_.data());
        const auto* second = first + self.capacity_ * (self.steps_.size() + 1);
        std::vector<std::uint64_t> prefix_rows(heads);
        std::size_t at = 0;
        for (std::size_t sequence = 0; sequence < self.batch_; ++sequence) {
            auto context = self.contexts_[sequence];
            const std::int32_t width = self.verification_widths_[sequence];
            for (std::int32_t column = 0; column < width; ++column, ++at) {
                ngram_row_ids(self.hash_, {self.verification_prefix_.data() + at, 1}, self.eos_,
                               self.vocab_, context, prefix_rows);
                auto winner_context = context, alternative_context = context;
                ngram_row_ids(self.hash_, {first + at, 1}, self.eos_, self.vocab_, winner_context,
                               std::span(self.rows_).subspan(at * heads, heads));
                ngram_row_ids(self.hash_, {second + at, 1}, self.eos_, self.vocab_, alternative_context,
                               std::span(self.rows_).subspan((count + at) * heads, heads));
            }
        }
        self.table_.prefetch(self.rows_);
    } catch (...) { self.failure_ = std::current_exception(); }
}

void NgramDraftPrefetch::enqueue_verification(const std::int32_t* first,
                                              const std::int32_t* second, cudaStream_t compute) {
    auto& ready = steps_.front().ready;
    ready.record(compute);
    ready.wait(stream_);
    const auto count = verification_prefix_.size();
    auto* out = static_cast<std::int32_t*>(tokens_.data());
    CUDA_CHECK(cudaMemcpyAsync(out, first, count * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(out + capacity_ * (steps_.size() + 1), second,
                               count * sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaLaunchHostFunc(stream_, &prefetch_verification, this));
}

void NgramDraftPrefetch::join(cudaStream_t compute) {
    done_.record(stream_);
    done_.wait(compute);
}

void NgramDraftPrefetch::check() const {
    if (failure_) { std::rethrow_exception(failure_); }
}

} // namespace ninfer::models::qwen4_exp
