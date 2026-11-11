#pragma once
#include "models/llama/model.h"
#include "core/arena.h"
#include <span>
#include <stdexcept>

namespace ninfer::models::llama {
struct ExecutorOptions {
    std::uint32_t max_context = 262144, prefill_chunk = 512, sequences = 1, draft_tokens = 0;
    // Zero selects as many whole host pages as fit after workspace, retaining device headroom.
    // Explicit sizes are useful for qualifying the resident/streamed boundary.
    std::uint32_t resident_tokens = 0;
    bool cuda_graphs = false, checkpoints = true;
};

struct ExecutorMemory {
    std::uint64_t kv_bytes = 0, kv_backed_bytes = 0, workspace_bytes = 0, host_kv_bytes = 0;
    bool kv_grows = false;
};

struct SequenceSnapshot {
    std::uint32_t position = 0;
};

struct SequenceImage {
    std::uint32_t position = 0;
};

class Executor {
public:
    static constexpr std::uint32_t kMaxLogitRows = 256;
    Executor(const Model&, DeviceContext&, ExecutorOptions);
    ~Executor();
    const ExecutorOptions& options() const noexcept;
    ExecutorMemory memory() const noexcept;
    void reset(std::uint32_t);
    std::uint32_t position(std::uint32_t) const;
    void warm_up();
    void forward(std::uint32_t, std::span<const std::int32_t>, std::uint32_t);
    void forward_branches(std::uint32_t, std::span<const std::int32_t>,
                          std::span<const std::uint32_t>);
    void forward_prompt_branches(std::uint32_t, std::span<const std::int32_t>,
                                 std::span<const std::int32_t>, std::span<const std::uint32_t>);
    void decode(std::span<const std::uint32_t>, std::span<const std::int32_t>);
    Tensor logits(std::uint32_t) const;
    std::uint32_t draft_tokens() const noexcept;
    void verify(std::span<const std::uint32_t>, std::span<const std::int32_t>,
                std::span<const std::uint32_t> = {});
    void commit(std::span<const std::uint32_t>, std::span<const std::uint32_t>);
    cudaStream_t head_stream() const noexcept;
    void snapshot(std::uint32_t, SequenceSnapshot&);
    void restore(std::uint32_t, const SequenceSnapshot&);
    std::uint64_t image_bytes(std::uint32_t) const;
    SequenceImage save_image(std::uint32_t, const SequenceSnapshot*, std::span<std::byte* const>,
                             std::size_t);
    void load_image(std::uint32_t, const SequenceImage&, std::span<std::byte* const>, std::size_t);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ninfer::models::llama
