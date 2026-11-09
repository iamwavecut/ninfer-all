#pragma once
#include "core/tensor.h"
#include "core/layout.h"
#include "core/arena.h"
#include "ninfer/ops/sampling.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
enum class SelectorRoute { Direct, Lattice };

struct SelectorWorkspace {
    Tensor edges;
};

// The two codebooks' rows: BF16, or (q4) ggml Q4_0 rows of 256 values in 8 blocks of 18 bytes.
struct SelectorCodebooks {
    const void* predecessor = nullptr;
    const void* successor   = nullptr;
    std::size_t bytes       = 0; // each codebook's
    bool q4                 = false;
};

SelectorRoute candidate_selector_path_route(int steps, int batch);
const char* candidate_selector_path_route_name(int steps, int batch);
const char* selector_route_name(SelectorRoute route);

template <class Allocator>
SelectorWorkspace allocate_selector_workspace(Allocator& allocator, SelectorRoute route, int steps,
                                              int batch) {
    SelectorWorkspace out;
    if (route == SelectorRoute::Lattice)
        out.edges = allocator.alloc(DType::FP32, {16, 16, steps, batch}, 256);
    return out;
}

void candidate_selector_path_dispatch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                      const Tensor& projected_hidden, const Tensor& anchors,
                                      const SelectorCodebooks& codebooks,
                                      const Tensor& base_positions, const SamplingConfig* configs,
                                      Tensor& drafts, Tensor& proposal_q, float* confidence,
                                      WorkspaceArena& workspace, cudaStream_t stream);
} // namespace ninfer::ops::detail
