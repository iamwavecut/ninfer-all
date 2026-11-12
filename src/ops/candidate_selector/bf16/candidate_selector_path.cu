#include "ops/candidate_selector/bf16/candidate_selector_path_kernels.h"
#include "core/device_launch.h"
#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/sampling_device.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kCandidates = 16, kRank = 256;

struct DeviceArgs {
    const std::int32_t* ids;
    const float* unary;
    const __nv_bfloat16* hidden;
    const std::int32_t* anchors;
    // BF16 rows of kRank values, or (q4) ggml Q4_0 rows of kRank / 32 blocks: a binary16 scale d
    // and 16 bytes of codes, value j = d * (q_j - 8), q_j the low nibble of byte j (j < 16) or the
    // high nibble of byte j - 16.
    const void* predecessor;
    const void* successor;
    bool q4;
    const std::int32_t* positions;
    const SamplingConfig* configs;
    std::int32_t* drafts;
    float* q;
    float* confidence; // FP32 [K,B] or null: the selected candidate's softmax(edge) probability
    int steps;
};

constexpr int kQ4BlockBytes = 18, kQ4RowBytes = kRank / 32 * kQ4BlockBytes;

struct alignas(16) SelectorShared {
    union {
        __nv_bfloat16 successors[kCandidates * kRank];
        float decoded[kCandidates * kRank]; // a Q4_0 codebook's rows, decoded exactly
    };
    float product[kRank];
    float edge[kCandidates];
    int predecessor, base_position;
    float temperature;
    unsigned long long seed;
};

// The probabilities written here are the same FP32 values consumed by the draw.
__device__ int draw_from(float edge, float maximum, float temperature, unsigned long long seed,
                         int position, float* q) {
    const int lane = threadIdx.x & 31;
    if (temperature <= 0.0F) {
        const unsigned winners =
            __ballot_sync(kFullWarpMask, lane < kCandidates && edge == maximum);
        const int selected = winners == 0 ? 0 : __ffs(winners) - 1;
        if (lane < kCandidates) q[lane] = lane == selected ? 1.0F : 0.0F;
        return selected;
    }
    const float weight      = lane < kCandidates ? __expf((edge - maximum) / temperature) : 0.0F;
    const float probability = weight / warp_sum(weight);
    if (lane < kCandidates) q[lane] = probability;
    float uniform =
        lane == 0 ? sampling_uniform(seed, position, kSamplePurposeDFlash2Proposal, 0U) : 0.0F;
    uniform          = __shfl_sync(kFullWarpMask, uniform, 0);
    float cumulative = probability;
#pragma unroll
    for (int offset = 1; offset < kCandidates; offset *= 2) {
        const float previous = __shfl_up_sync(kFullWarpMask, cumulative, offset);
        if (lane >= offset) cumulative += previous;
    }
    const unsigned hits = __ballot_sync(kFullWarpMask, lane < kCandidates && uniform < cumulative);
    return hits ? __ffs(hits) - 1 : kCandidates - 1;
}

// A warp's draw; with `confidence`, lane 0 also stores the drawn candidate's probability under
// the softmax of the edges at temperature 1.
__device__ int draw_rank(float edge, float temperature, unsigned long long seed, int position,
                         float* q, float* confidence) {
    const int lane      = threadIdx.x & 31;
    const float maximum = warp_max(edge);
    const int selected  = draw_from(edge, maximum, temperature, seed, position, q);
    if (confidence != nullptr) {
        const float weight = lane < kCandidates ? __expf(edge - maximum) : 0.0F;
        const float total  = warp_sum(weight);
        const float mine   = __shfl_sync(kFullWarpMask, weight, selected);
        if (lane == 0) *confidence = mine / total;
    }
    return selected;
}

// Value r of a Q4_0 codebook row, exact in FP32.
__device__ __forceinline__ float q4_value(const unsigned char* row, int r) {
    const unsigned char* block = row + (r >> 5) * kQ4BlockBytes;
    const float d              = __half2float(*reinterpret_cast<const __half*>(block));
    const int j                = r & 31;
    const int code = j < 16 ? (block[2 + j] & 0x0F) : (block[2 + j - 16] >> 4);
    return d * static_cast<float>(code - 8);
}

__device__ void score_row(const DeviceArgs& a, int column, SelectorShared& shared) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int token = lane == 0 ? a.ids[column * kCandidates + warp] : 0;
    token     = __shfl_sync(kFullWarpMask, token, 0);
    if (a.q4) {
        // Lane l decodes values [8 (l % 4), 8 (l % 4) + 8) of block l / 4 of its warp's candidate.
        const auto* row = static_cast<const unsigned char*>(a.successor) +
                          static_cast<std::int64_t>(token) * kQ4RowBytes;
        const int first = (lane >> 2) * 32 + (lane & 3) * 8;
#pragma unroll
        for (int k = 0; k < 8; ++k)
            shared.decoded[warp * kRank + first + k] = q4_value(row, first + k);
    } else {
        cp_async<16, Cache::cg>(&shared.successors[warp * kRank + lane * 8],
                                static_cast<const __nv_bfloat16*>(a.successor) +
                                    static_cast<std::int64_t>(token) * kRank + lane * 8);
        cp_commit();
    }
    // Publish the preceding draw before reading its token. Successor prefetch is independent.
    __syncthreads();
    if (tid < kRank) {
        const std::int64_t predecessor = shared.predecessor;
        const float value =
            a.q4 ? q4_value(static_cast<const unsigned char*>(a.predecessor) +
                                predecessor * kQ4RowBytes,
                            tid)
                 : __bfloat162float(static_cast<const __nv_bfloat16*>(
                       a.predecessor)[predecessor * kRank + tid]);
        shared.product[tid] =
            value * __bfloat162float(a.hidden[static_cast<std::int64_t>(column) * kRank + tid]);
    }
    if (!a.q4) cp_wait<0>();
    __syncthreads();
    {
        const int c = warp;
        float sum   = 0;
#pragma unroll
        for (int r = lane; r < kRank; r += 32)
            sum = fmaf(shared.product[r],
                       a.q4 ? shared.decoded[c * kRank + r]
                            : __bfloat162float(shared.successors[c * kRank + r]),
                       sum);
        sum = warp_reduce_sum(sum);
        if (lane == 0) shared.edge[c] = a.unary[column * kCandidates + c] + sum;
    }
    __syncthreads();
}

__global__ __launch_bounds__(512, 1) void selector_walk_kernel(DeviceArgs a) {
    pdl::enter();
    __shared__ SelectorShared shared;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, batch = blockIdx.x;
    if (tid == 0) {
        shared.predecessor   = a.anchors[batch];
        shared.base_position = a.positions[batch];
        shared.temperature   = a.configs[batch].temperature;
        shared.seed          = a.configs[batch].seed;
    }
#pragma unroll 1
    for (int step = 0; step < a.steps; ++step) {
        const int column = batch * a.steps + step;
        score_row(a, column, shared);
        if (warp == 0) {
            const float edge   = lane < kCandidates ? shared.edge[lane] : -CUDART_INF_F;
            const int selected = draw_rank(edge, shared.temperature, shared.seed,
                                           shared.base_position + step, a.q + column * kCandidates,
                                           a.confidence ? a.confidence + column : nullptr);
            if (lane == 0) {
                shared.predecessor = a.ids[column * kCandidates + selected];
                a.drafts[column]   = shared.predecessor;
            }
        }
    }
}

__global__ __launch_bounds__(512, 2) void selector_lattice_kernel(DeviceArgs a, float* edges) {
    pdl::enter();
    const int column = blockIdx.x, p = blockIdx.y;
    const int step = column % a.steps, batch = column / a.steps;
    if (step == 0 && p != 0) return;
    __shared__ SelectorShared shared;
    if (threadIdx.x == 0)
        shared.predecessor = step == 0 ? a.anchors[batch] : a.ids[(column - 1) * kCandidates + p];
    score_row(a, column, shared);
    if (threadIdx.x < kCandidates)
        edges[(static_cast<std::int64_t>(column) * kCandidates + p) * kCandidates + threadIdx.x] =
            shared.edge[threadIdx.x];
}

__global__ __launch_bounds__(32) void selector_lattice_walk_kernel(DeviceArgs a,
                                                                   const float* edges) {
    pdl::enter();
    const int lane = threadIdx.x, batch = blockIdx.x;
    const auto seed         = a.configs[batch].seed;
    const float temperature = a.configs[batch].temperature;
    const int position      = a.positions[batch];
    int predecessor_rank    = 0;
#pragma unroll 1
    for (int step = 0; step < a.steps; ++step) {
        const int column = batch * a.steps + step;
        const float edge =
            lane < kCandidates
                ? edges[(static_cast<std::int64_t>(column) * kCandidates + predecessor_rank) *
                            kCandidates +
                        lane]
                : -CUDART_INF_F;
        int selected = draw_rank(edge, temperature, seed, position + step,
                                 a.q + column * kCandidates,
                                 a.confidence ? a.confidence + column : nullptr);
        predecessor_rank = selected;
        if (lane == 0) a.drafts[column] = a.ids[column * kCandidates + selected];
    }
}

} // namespace

void candidate_selector_path_launch(SelectorRoute route, const Tensor& candidate_ids,
                                    const Tensor& unary_scores, const Tensor& projected_hidden,
                                    const Tensor& anchors, const SelectorCodebooks& codebooks,
                                    const Tensor& base_positions,
                                    const SamplingConfig* configs, Tensor& drafts,
                                    Tensor& proposal_q, float* confidence,
                                    const SelectorWorkspace& workspace, cudaStream_t stream) {
    const DeviceArgs args{static_cast<const std::int32_t*>(candidate_ids.data),
                          static_cast<const float*>(unary_scores.data),
                          static_cast<const __nv_bfloat16*>(projected_hidden.data),
                          static_cast<const std::int32_t*>(anchors.data),
                          codebooks.predecessor,
                          codebooks.successor,
                          codebooks.q4,
                          static_cast<const std::int32_t*>(base_positions.data),
                          configs,
                          static_cast<std::int32_t*>(drafts.data),
                          static_cast<float*>(proposal_q.data),
                          confidence,
                          candidate_ids.ne[1]};
    if (route == SelectorRoute::Direct) {
        CUDA_CHECK(pdl::launch_consumer({dim3(candidate_ids.ne[2]), dim3(512), 0, stream},
                                        selector_walk_kernel, args));
    } else {
        auto* edges = static_cast<float*>(workspace.edges.data);
        CUDA_CHECK(pdl::launch_consumer(
            {dim3(args.steps * candidate_ids.ne[2], kCandidates), dim3(512), 0, stream},
            selector_lattice_kernel, args, edges));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(pdl::launch_consumer({dim3(candidate_ids.ne[2]), dim3(32), 0, stream},
                                        selector_lattice_walk_kernel, args, edges));
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
