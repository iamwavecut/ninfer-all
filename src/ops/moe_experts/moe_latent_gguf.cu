// ninfer::ops - Edge0's latent MoE over GGUF block banks (contract in
// include/ninfer/ops/moe_experts.h). Three MoE products of the GGUF bridge: the encoder D and the
// decoder U run as banks of `groups` experts, a token's pair (t, g) present when one of its routed
// experts belongs to group g; the routed experts read the encoded column of their group and add
// their weighted down rows into the 2^-32 fixed-point group sums, which U then reads. The shared
// expert is moe_experts_gguf's.
#include "ninfer/ops/moe_experts.h"

#include "core/device_launch.h"
#include "core/layout.h"
#include "ops/common/fixed_point.cuh"
#include "ops/common/silu.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kHidden       = 2560;
constexpr int kMiddle       = 640;
constexpr int kTopK         = 10;
constexpr int kVectorTokens = 8;
constexpr int kMaxGroups    = 16;
// The vector kernel's columns of one expert per pass in a divided call (moe_experts_gguf.cu).
constexpr int kDividedVectorChunk = 4;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("moe_latent_gguf: ") + message); }
}

bool shaped(const Tensor& tensor, DType dtype, std::int32_t n0, std::int32_t n1) {
    return tensor.dtype == dtype && tensor.is_contiguous() && tensor.data != nullptr &&
           tensor.ne[0] == n0 && tensor.ne[1] == n1 && tensor.ne[2] == 1 && tensor.ne[3] == 1;
}

int chunk_for(std::int64_t pairs, std::int64_t experts) {
    const std::int64_t average = (pairs + experts - 1) / experts;
    return average >= 6 ? 8 : average >= 3 ? 4 : average >= 2 ? 2 : 1;
}

gguf::MoeTable table(const GgufExpertTable& bank, int rows, int k) {
    require(bank.experts != nullptr && bank.row_bytes > 0, "an expert table is empty");
    return gguf::MoeTable{bank.experts, bank.row_bytes, rows, k, bank.aligned16};
}

gguf::GgmlType type_of(const GgufExpertTable& bank) { return detail::gguf_type(bank.format); }

// A wide call's projection on the matrix kernel, as moe_experts_gguf chooses (IQ1_M stays on the
// vector products here: no Edge0 build stores it).
bool matrix_capable(std::int32_t tokens, const GgufExpertTable& bank, int rows, int k, bool tail) {
    const auto type = type_of(bank);
    return tokens > kVectorTokens && type != gguf::GgmlType::IQ1_M &&
           gguf::moe_matrix_supported(type, rows, k, tail);
}

void check_banks(const GgufLatentWeights& banks) {
    require(banks.width > 0 && banks.width % 256 == 0, "the latent width must be a multiple of 256");
    require(banks.groups > 0 && banks.groups <= kMaxGroups, "one to sixteen groups");
    require(banks.experts >= kTopK && banks.experts <= 512, "experts must be in [10, 512]");
    require(banks.labels != nullptr, "the experts' group labels are required");
}

GgufMoeWeights shared_banks(const GgufLatentWeights& banks) {
    GgufMoeWeights out;
    out.shared_gate     = banks.shared_gate;
    out.shared_up       = banks.shared_up;
    out.shared_down     = banks.shared_down;
    out.shared_width    = banks.shared_width;
    out.device_resident = banks.device_resident;
    return out;
}

// One thread per token: the groups it routes to within [group_begin, group_end) (group_ids, with
// weight one), and the encoded column each of its pairs reads (sources; -1 for an empty pair).
__global__ void latent_routes_kernel(const std::int32_t* __restrict__ ids,
                                     const std::int32_t* __restrict__ labels, int tokens,
                                     int groups, int group_begin, int group_end,
                                     std::int32_t* __restrict__ group_ids,
                                     float* __restrict__ group_weights,
                                     std::int32_t* __restrict__ sources) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tokens) { return; }
    unsigned used = 0;
    for (int j = 0; j < kTopK; ++j) {
        const int p = t * kTopK + j;
        const int e = ids[p];
        if (e < 0) {
            sources[p] = -1;
            continue;
        }
        const int g = labels[e];
        if (g >= group_begin && g < group_end) {
            sources[p] = t * groups + g;
            used |= 1u << g;
        } else {
            sources[p] = -1;
        }
    }
    for (int g = 0; g < groups; ++g) {
        group_ids[t * groups + g]     = (used >> g) & 1u ? g : -1;
        group_weights[t * groups + g] = 1.0f;
    }
}

// out[p] = ids[p] when its expert's group lies in [group_begin, group_end), else -1.
__global__ void keep_groups_kernel(const std::int32_t* __restrict__ ids,
                                   const std::int32_t* __restrict__ labels, int pairs,
                                   int group_begin, int group_end, std::int32_t* __restrict__ out) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pairs) { return; }
    const int e = ids[p];
    const int g = e >= 0 ? labels[e] : -1;
    out[p]      = g >= group_begin && g < group_end ? e : -1;
}

// middle[p][r] = BF16(silu(s_gate * gate) * (s_up * up)) for the pairs with an expert, each
// multiplier rounded onto its product (ones without scales).
__global__ void latent_swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                     const std::int32_t* __restrict__ ids,
                                     const float* __restrict__ scales, std::int64_t count,
                                     __nv_bfloat16* __restrict__ middle) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const int e = ids[i / kMiddle];
    if (e < 0) { return; }
    float g = gate[i], u = up[i];
    if (scales != nullptr) {
        g = __fmul_rn(scales[3 * e + 0], g);
        u = __fmul_rn(scales[3 * e + 1], u);
    }
    middle[i] = __float2bfloat16(portable_silu(g) * u);
}

// S[t * groups + label][r] += fixed(s_down * down[p][r], weights[p]) over token t's pairs with an
// expert. One thread per (token, row): it owns every group sum of that row, so no atomics.
__global__ void latent_accumulate_kernel(const float* __restrict__ down,
                                         const float* __restrict__ weights,
                                         const std::int32_t* __restrict__ ids,
                                         const std::int32_t* __restrict__ labels,
                                         const float* __restrict__ scales, int groups, int width,
                                         std::int64_t count, unsigned long long* __restrict__ latent) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const std::int64_t t = i / width, r = i % width;
    for (int j = 0; j < kTopK; ++j) {
        const std::int64_t p = t * kTopK + j;
        const int e          = ids[p];
        if (e < 0) { continue; }
        float value = down[p * width + r];
        if (scales != nullptr) { value = __fmul_rn(scales[3 * e + 2], value); }
        latent[(t * groups + labels[e]) * width + r] +=
            static_cast<unsigned long long>(fixed_product(value, weights[p]));
    }
}

// The listed external down products, as latent_accumulate_kernel adds a kernel's.
__global__ void latent_products_kernel(const float* __restrict__ values,
                                       const std::int32_t* __restrict__ pairs,
                                       const std::int32_t* __restrict__ count,
                                       const std::int32_t* __restrict__ ids,
                                       const std::int32_t* __restrict__ labels,
                                       const float* __restrict__ scales,
                                       const float* __restrict__ weights, int groups, int width,
                                       unsigned long long* __restrict__ latent) {
    const int i = blockIdx.y;
    if (i >= *count) { return; }
    const int pair = pairs[i];
    const int e    = ids[pair];
    if (e < 0) { return; }
    unsigned long long* row =
        latent + (std::int64_t(pair / kTopK) * groups + labels[e]) * width;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < width; r += gridDim.x * blockDim.x) {
        float value = values[std::int64_t(i) * width + r];
        if (scales != nullptr) { value = __fmul_rn(scales[3 * e + 2], value); }
        atomicAdd(row + r, static_cast<unsigned long long>(fixed_product(value, weights[pair])));
    }
}

// fixed[t][r] += each routed group's U row r of token t, weight one, rounded as the vector
// kernel's weighted output rounds it. One thread per (token, row).
__global__ void add_group_rows_kernel(const float* __restrict__ rows,
                                      const std::int32_t* __restrict__ group_ids, int groups,
                                      std::int64_t count, unsigned long long* __restrict__ fixed) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const std::int64_t t = i / kHidden, r = i % kHidden;
    unsigned long long sum = 0;
    for (int g = 0; g < groups; ++g) {
        const std::int64_t c = t * groups + g;
        if (group_ids[c] < 0) { continue; }
        sum += static_cast<unsigned long long>(fixed_product(rows[c * kHidden + r], 1.0f));
    }
    fixed[i] += sum;
}

__global__ void fixed_to_float_kernel(const unsigned long long* __restrict__ fixed,
                                      float* __restrict__ out, std::int64_t count) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    out[i] = static_cast<float>(static_cast<double>(static_cast<long long>(fixed[i])) *
                                (1.0 / 4294967296.0));
}

unsigned blocks_for(std::int64_t count, int threads = 256) {
    return static_cast<unsigned>((count + threads - 1) / threads);
}

// The stage's internal begin: the groups [group_begin, group_end) and the shared expert when given.
GgufLatentStage latent_begin(const Tensor& m, const Tensor& ids, const Tensor* shared,
                             const GgufLatentWeights& banks, std::int32_t group_begin,
                             std::int32_t group_end, WorkspaceArena& workspace,
                             cudaStream_t stream, const GgufMoeSide* side) {
    check_banks(banks);
    const std::int32_t tokens = m.ne[1];
    require(tokens > 0 && shaped(m, DType::BF16, kHidden, tokens), "m must be BF16 [2560, tokens]");
    require(tokens * kTopK <= 65535, "at most 6553 tokens per call");
    require(shaped(ids, DType::I32, kTopK, tokens), "ids must be I32 [10, tokens]");
    require(tokens <= kVectorTokens || banks.device_resident,
            "a call of more than eight tokens needs device-resident banks");
    const int G = banks.groups, L = banks.width;
    GgufLatentStage stage;
    stage.tokens       = tokens;
    stage.pairs        = tokens * kTopK;
    const int columns  = tokens * G;
    stage.fixed        = static_cast<unsigned long long*>(
        workspace.alloc(DType::I64, {kHidden, tokens}).data);
    stage.latent       = static_cast<unsigned long long*>(workspace.alloc(DType::I64, {L, columns}).data);
    stage.encoded      = static_cast<float*>(workspace.alloc(DType::FP32, {L, columns}).data);
    stage.sources      = static_cast<std::int32_t*>(workspace.alloc(DType::I32, {stage.pairs}).data);
    stage.group_ids    = static_cast<std::int32_t*>(workspace.alloc(DType::I32, {columns}).data);
    stage.group_weights = static_cast<float*>(workspace.alloc(DType::FP32, {columns}).data);
    stage.group_routing = workspace.alloc(
        DType::U8, {static_cast<std::int32_t>(gguf::moe_routing_bytes(G, columns))}).data;
    if (tokens <= kGgufMoeSplitTokens) {
        const std::size_t bytes = gguf::vector_activation_bytes(L, stage.pairs);
        require(bytes == moe_latent_gguf_input_bytes(stage.pairs, L) &&
                    gguf::vector_activation_scales_offset(L, stage.pairs) ==
                        moe_latent_gguf_input_scales(stage.pairs, L),
                "the latent inputs' layout differs from moe_latent_gguf_input_bytes");
        stage.inputs = workspace.alloc(DType::U8, {static_cast<std::int32_t>(bytes)}).data;
    }
    CUDA_CHECK(cudaMemsetAsync(stage.fixed, 0, sizeof(unsigned long long) * kHidden * tokens, stream));
    CUDA_CHECK(cudaMemsetAsync(stage.latent, 0, sizeof(unsigned long long) * L * columns, stream));
    latent_routes_kernel<<<blocks_for(tokens, 128), 128, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), banks.labels, tokens, G, group_begin,
        group_end, stage.group_ids, stage.group_weights, stage.sources);
    CUDA_CHECK(cudaGetLastError());
    if (shared != nullptr) {
        stage.join = moe_experts_gguf_shared(m, *shared, shared_banks(banks), workspace,
                                             stage.fixed, stream, side);
    }
    const gguf::MoeRouting routing =
        gguf::moe_sort_routes(stage.group_ids, columns, G, stage.group_routing, stream);
    const int active     = std::min(columns, G);
    const auto* m_p      = static_cast<const __nv_bfloat16*>(m.data);
    const auto encode    = type_of(banks.encode);
    const auto d_table   = table(banks.encode, L, kHidden);
    if (matrix_capable(tokens, banks.encode, L, kHidden, false)) {
        auto scope = workspace.scope();
        Tensor x   = workspace.alloc(
            DType::U8,
            {static_cast<std::int32_t>(gguf::moe_matrix_activation_bytes(kHidden, columns))});
        gguf::quantize_moe_matrix_activation(encode, m_p, kHidden, routing, columns, G, x.data,
                                             stream);
        gguf::moe_matrix_product(encode, d_table, routing, active, columns, false, x.data,
                                 stage.encoded, stream);
    } else {
        auto scope = workspace.scope();
        Tensor x   = workspace.alloc(
            DType::U8, {static_cast<std::int32_t>(gguf::vector_activation_bytes(kHidden, tokens))});
        gguf::quantize_vector_activation(m_p, kHidden, tokens, nullptr, x.data, stream);
        gguf::moe_vector_product(encode, d_table, routing, active, G, G, x.data, tokens,
                                 gguf::MoeOutput{.f32 = stage.encoded}, chunk_for(columns, active),
                                 stream);
    }
    if (stage.inputs != nullptr) {
        gguf::quantize_vector_activation_f32(stage.encoded, L, stage.pairs, stage.sources,
                                             banks.column_scale, stage.inputs, stream);
    }
    return stage;
}

// The routed pairs of `ids` (a subset of the stage's) through `banks`' routed tables into S.
void latent_add(const GgufLatentStage& stage, const Tensor& ids, const Tensor& weights,
                const GgufLatentWeights& banks, WorkspaceArena& workspace, cudaStream_t stream) {
    check_banks(banks);
    const std::int32_t tokens = stage.tokens, pairs = stage.pairs;
    require(stage.latent != nullptr, "the stage has not begun");
    require(shaped(ids, DType::I32, kTopK, tokens), "ids must be I32 [10, tokens]");
    require(shaped(weights, DType::FP32, kTopK, tokens), "weights must be FP32 [10, tokens]");
    require(tokens <= kVectorTokens || banks.device_resident,
            "a call of more than eight tokens needs device-resident banks");
    const int L = banks.width, experts = banks.experts;
    auto scope            = workspace.scope();
    const auto* ids_p     = static_cast<const std::int32_t*>(ids.data);
    const bool gate_up_matrix = matrix_capable(tokens, banks.gate, kMiddle, L, false) &&
                                matrix_capable(tokens, banks.up, kMiddle, L, false);
    const bool down_matrix = matrix_capable(tokens, banks.down, L, kMiddle, true);
    const bool divided     = (gate_up_matrix || down_matrix) && tokens <= kGgufMoeSplitTokens;
    gguf::MoeSplitRouting split;
    if (divided) {
        Tensor routing_bytes = workspace.alloc(
            DType::U8,
            {static_cast<std::int32_t>(gguf::moe_split_routing_bytes(experts, pairs))});
        split = gguf::moe_sort_routes_split(ids_p, pairs, experts, kGgufMoeVectorPairs,
                                            routing_bytes.data, stream);
    } else {
        Tensor routing_bytes = workspace.alloc(
            DType::U8, {static_cast<std::int32_t>(gguf::moe_routing_bytes(experts, pairs))});
        split.all    = gguf::moe_sort_routes(ids_p, pairs, experts, routing_bytes.data, stream);
        split.vector = split.all;
        split.matrix = split.all;
    }
    require((gate_up_matrix && !divided) || stage.inputs != nullptr,
            "the stage holds no vector activations");
    const int active   = std::min(pairs, experts);
    const int chunk    = divided ? kDividedVectorChunk : chunk_for(pairs, active);
    const auto gate_t  = type_of(banks.gate), up_t = type_of(banks.up), down_t = type_of(banks.down);
    const auto gate_table = table(banks.gate, kMiddle, L);
    const auto up_table   = table(banks.up, kMiddle, L);
    const bool fused      = banks.expert_scales == nullptr && gate_t == up_t &&
                       banks.gate.row_bytes == banks.up.row_bytes;
    Tensor middle  = workspace.alloc(DType::BF16, {kMiddle, pairs});
    auto* middle_p = static_cast<__nv_bfloat16*>(middle.data);
    {
        auto planes   = workspace.scope();
        Tensor gate_p = workspace.alloc(DType::FP32, {kMiddle, pairs});
        Tensor up_p   = workspace.alloc(DType::FP32, {kMiddle, pairs});
        auto* gate_f  = static_cast<float*>(gate_p.data);
        auto* up_f    = static_cast<float*>(up_p.data);
        // The vector kernel's gate and up products of `routing`'s experts into the planes.
        const auto vector_planes = [&](const gguf::MoeRouting& routing) {
            gguf::moe_vector_product(gate_t, gate_table, routing, active, 1, kTopK, stage.inputs,
                                     pairs, gguf::MoeOutput{.f32 = gate_f}, chunk, stream);
            gguf::moe_vector_product(up_t, up_table, routing, active, 1, kTopK, stage.inputs,
                                     pairs, gguf::MoeOutput{.f32 = up_f}, chunk, stream);
        };
        if (gate_up_matrix) {
            const auto bytes = static_cast<std::int32_t>(gguf::moe_matrix_activation_bytes(L, pairs));
            Tensor gate_x    = workspace.alloc(DType::U8, {bytes});
            gguf::quantize_moe_matrix_activation_f32(gate_t, stage.encoded, L, split.all, pairs,
                                                     stage.sources, banks.column_scale,
                                                     gate_x.data, stream);
            Tensor up_x = gate_x;
            if (gguf::matrix_activation_layout(up_t) != gguf::matrix_activation_layout(gate_t)) {
                up_x = workspace.alloc(DType::U8, {bytes});
                gguf::quantize_moe_matrix_activation_f32(up_t, stage.encoded, L, split.all, pairs,
                                                         stage.sources, banks.column_scale,
                                                         up_x.data, stream);
            }
            gguf::moe_matrix_product(gate_t, gate_table, split.matrix, active, pairs, false,
                                     gate_x.data, gate_f, stream);
            gguf::moe_matrix_product(up_t, up_table, split.matrix, active, pairs, false,
                                     up_x.data, up_f, stream);
            if (divided && !fused) { vector_planes(split.vector); }
        } else if (!fused) {
            vector_planes(split.all);
        }
        if (gate_up_matrix || !fused) {
            const std::int64_t count = std::int64_t(kMiddle) * pairs;
            latent_swiglu_kernel<<<blocks_for(count), 256, 0, stream>>>(
                gate_f, up_f, ids_p, banks.expert_scales, count, middle_p);
            CUDA_CHECK(cudaGetLastError());
        }
        if (fused && (!gate_up_matrix || divided)) {
            gguf::moe_vector_swiglu(gate_t, gate_table, up_table,
                                    gate_up_matrix ? split.vector : split.all, active, 1,
                                    stage.inputs, pairs, middle_p, chunk, stream);
        }
    }
    const auto down_table = table(banks.down, L, kMiddle);
    Tensor down           = workspace.alloc(DType::FP32, {L, pairs});
    auto* down_p          = static_cast<float*>(down.data);
    const auto down_vector = [&](const gguf::MoeRouting& routing) {
        Tensor middle_q = workspace.alloc(
            DType::U8, {static_cast<std::int32_t>(gguf::vector_activation_bytes(kMiddle, pairs))});
        gguf::quantize_vector_activation(middle_p, kMiddle, pairs, nullptr, middle_q.data, stream);
        gguf::moe_vector_product(down_t, down_table, routing, active, 1, kTopK, middle_q.data,
                                 pairs, gguf::MoeOutput{.f32 = down_p}, chunk, stream);
    };
    if (down_matrix) {
        Tensor middle_x = workspace.alloc(
            DType::U8,
            {static_cast<std::int32_t>(gguf::moe_matrix_activation_bytes(kMiddle, pairs))});
        gguf::quantize_moe_matrix_activation(down_t, middle_p, kMiddle, split.all, pairs, 1,
                                             middle_x.data, stream);
        gguf::moe_matrix_product(down_t, down_table, split.matrix, active, pairs, true,
                                 middle_x.data, down_p, stream);
        if (divided) { down_vector(split.vector); }
    } else {
        down_vector(split.all);
    }
    const std::int64_t count = std::int64_t(L) * tokens;
    latent_accumulate_kernel<<<blocks_for(count), 256, 0, stream>>>(
        down_p, static_cast<const float*>(weights.data), ids_p, banks.labels, banks.expert_scales,
        banks.groups, L, count, stage.latent);
    CUDA_CHECK(cudaGetLastError());
}

// U over the group sums into the stage's output sum; the caller stores or copies it.
void latent_decode(const GgufLatentStage& stage, const gguf::MoeRouting& routing,
                   const GgufLatentWeights& banks, WorkspaceArena& workspace, cudaStream_t stream) {
    const int G = banks.groups, L = banks.width;
    const std::int32_t tokens = stage.tokens, columns = tokens * G;
    auto scope = workspace.scope();
    // The sums as FP32 over the encoded plane, which no stage reads any more.
    const std::int64_t values = std::int64_t(L) * columns;
    fixed_to_float_kernel<<<blocks_for(values), 256, 0, stream>>>(stage.latent, stage.encoded,
                                                                  values);
    CUDA_CHECK(cudaGetLastError());
    const int active     = std::min(columns, G);
    const auto decode    = type_of(banks.decode);
    const auto u_table   = table(banks.decode, kHidden, L);
    if (matrix_capable(tokens, banks.decode, kHidden, L, false)) {
        Tensor x = workspace.alloc(
            DType::U8, {static_cast<std::int32_t>(gguf::moe_matrix_activation_bytes(L, columns))});
        gguf::quantize_moe_matrix_activation_f32(decode, stage.encoded, L, routing, columns,
                                                 nullptr, nullptr, x.data, stream);
        Tensor rows = workspace.alloc(DType::FP32, {kHidden, columns});
        gguf::moe_matrix_product(decode, u_table, routing, active, columns, false, x.data,
                                 static_cast<float*>(rows.data), stream);
        const std::int64_t count = std::int64_t(kHidden) * tokens;
        add_group_rows_kernel<<<blocks_for(count), 256, 0, stream>>>(
            static_cast<const float*>(rows.data), stage.group_ids, G, count, stage.fixed);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    Tensor x = workspace.alloc(
        DType::U8, {static_cast<std::int32_t>(gguf::vector_activation_bytes(L, columns))});
    gguf::quantize_vector_activation_f32(stage.encoded, L, columns, nullptr, nullptr, x.data,
                                         stream);
    gguf::moe_vector_product(decode, u_table, routing, active, 1, G, x.data, columns,
                             gguf::MoeOutput{.weighted = stage.fixed,
                                             .weights  = stage.group_weights},
                             chunk_for(columns, active), stream);
}

__global__ void store_kernel(const unsigned long long* __restrict__ fixed, float* __restrict__ y,
                             std::int64_t count) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    y[i] = static_cast<float>(static_cast<double>(static_cast<long long>(fixed[i])) *
                              (1.0 / 4294967296.0));
}

// The routing begin sorted, kept beside the stage: finish runs U over the same groups.
gguf::MoeRouting stage_routing(const GgufLatentStage& stage, const GgufLatentWeights& banks) {
    return gguf::moe_sorted_routes(stage.group_routing, stage.tokens * banks.groups, banks.groups);
}

} // namespace

std::size_t moe_latent_gguf_workspace_bytes(std::int32_t tokens, std::int32_t width,
                                            std::int32_t groups) {
    require(tokens > 0 && width > 0 && groups > 0, "invalid latent shape");
    const std::int32_t pairs = tokens * kTopK, columns = tokens * groups;
    const bool wide = tokens > kVectorTokens;
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::I64, {kHidden, tokens});
    (void)layout.alloc(DType::I64, {width, columns});
    (void)layout.alloc(DType::FP32, {width, columns});
    (void)layout.alloc(DType::I32, {pairs});
    (void)layout.alloc(DType::I32, {columns});
    (void)layout.alloc(DType::FP32, {columns});
    (void)layout.alloc(DType::U8, {static_cast<std::int32_t>(gguf::moe_routing_bytes(groups, columns))});
    // A group-parallel part's routed ids.
    (void)layout.alloc(DType::I32, {pairs});
    if (tokens <= kGgufMoeSplitTokens) {
        (void)layout.alloc(DType::U8, {static_cast<std::int32_t>(
                                          gguf::vector_activation_bytes(width, pairs))});
    }
    // The shared expert's allocations stay to the end of the call (a side stream may run them).
    (void)layout.alloc(DType::U8,
                       {static_cast<std::int32_t>(moe_experts_gguf_shared_workspace_bytes(tokens))});
    {
        // D's activation.
        auto d = layout.scope();
        (void)layout.alloc(DType::U8, {static_cast<std::int32_t>(
                                          wide ? gguf::moe_matrix_activation_bytes(kHidden, columns)
                                               : gguf::vector_activation_bytes(kHidden, tokens))});
    }
    {
        // An add: routing, middle, the gate and up planes with their activations, then the down
        // plane and its inputs.
        auto add = layout.scope();
        (void)layout.alloc(DType::U8, {static_cast<std::int32_t>(
                                          wide && tokens <= kGgufMoeSplitTokens
                                              ? gguf::moe_split_routing_bytes(512, pairs)
                                              : gguf::moe_routing_bytes(512, pairs))});
        (void)layout.alloc(DType::BF16, {kMiddle, pairs});
        {
            auto planes = layout.scope();
            (void)layout.alloc(DType::FP32, {kMiddle, pairs});
            (void)layout.alloc(DType::FP32, {kMiddle, pairs});
            if (wide) {
                const auto bytes =
                    static_cast<std::int32_t>(gguf::moe_matrix_activation_bytes(width, pairs));
                (void)layout.alloc(DType::U8, {bytes});
                (void)layout.alloc(DType::U8, {bytes});
            }
        }
        (void)layout.alloc(DType::FP32, {width, pairs});
        if (wide) {
            (void)layout.alloc(DType::U8, {static_cast<std::int32_t>(
                                              gguf::moe_matrix_activation_bytes(kMiddle, pairs))});
        }
        if (!wide || tokens <= kGgufMoeSplitTokens) {
            (void)layout.alloc(DType::U8, {static_cast<std::int32_t>(
                                              gguf::vector_activation_bytes(kMiddle, pairs))});
        }
    }
    {
        // U: its activation and, on the matrix kernel, the rows before their weighted sum.
        auto u = layout.scope();
        if (wide) {
            (void)layout.alloc(DType::U8, {static_cast<std::int32_t>(
                                              gguf::moe_matrix_activation_bytes(width, columns))});
            (void)layout.alloc(DType::FP32, {kHidden, columns});
        } else {
            (void)layout.alloc(DType::U8, {static_cast<std::int32_t>(
                                              gguf::vector_activation_bytes(width, columns))});
        }
    }
    return layout.peak_bytes(1);
}

GgufLatentStage moe_latent_gguf_begin(const Tensor& m, const Tensor& ids, const Tensor& shared,
                                      const GgufLatentWeights& banks, WorkspaceArena& workspace,
                                      cudaStream_t stream, const GgufMoeSide* side) {
    return latent_begin(m, ids, &shared, banks, 0, banks.groups, workspace, stream, side);
}

void moe_latent_gguf_add(const GgufLatentStage& stage, const Tensor& ids, const Tensor& weights,
                         const GgufLatentWeights& banks, WorkspaceArena& workspace,
                         cudaStream_t stream) {
    latent_add(stage, ids, weights, banks, workspace, stream);
}

void moe_latent_gguf_add_products(const GgufLatentStage& stage, const float* values,
                                  const std::int32_t* pairs, const std::int32_t* count,
                                  std::int32_t max_count, const Tensor& ids,
                                  const Tensor& weights, const GgufLatentWeights& banks,
                                  cudaStream_t stream) {
    require(stage.latent != nullptr && max_count >= 0 && max_count <= stage.pairs,
            "invalid product count");
    require(shaped(ids, DType::I32, kTopK, stage.tokens), "ids must be I32 [10, tokens]");
    require(shaped(weights, DType::FP32, kTopK, stage.tokens), "weights must be FP32 [10, tokens]");
    if (max_count == 0) { return; }
    latent_products_kernel<<<dim3(4, static_cast<unsigned>(max_count)), 512, 0, stream>>>(
        values, pairs, count, static_cast<const std::int32_t*>(ids.data), banks.labels,
        banks.expert_scales, static_cast<const float*>(weights.data), banks.groups, banks.width,
        stage.latent);
    CUDA_CHECK(cudaGetLastError());
}

void moe_latent_gguf_finish(const GgufLatentStage& stage, const GgufLatentWeights& banks,
                            WorkspaceArena& workspace, Tensor& y, cudaStream_t stream) {
    require(stage.fixed != nullptr && shaped(y, DType::FP32, kHidden, stage.tokens),
            "y must be FP32 [2560, tokens]");
    check_banks(banks);
    latent_decode(stage, stage_routing(stage, banks), banks, workspace, stream);
    if (stage.join != nullptr) { CUDA_CHECK(cudaStreamWaitEvent(stream, stage.join, 0)); }
    const std::int64_t count = std::int64_t(kHidden) * stage.tokens;
    store_kernel<<<blocks_for(count), 256, 0, stream>>>(stage.fixed, static_cast<float*>(y.data),
                                                        count);
    CUDA_CHECK(cudaGetLastError());
}

void moe_latent_gguf(const Tensor& m, const Tensor& ids, const Tensor& weights,
                     const Tensor& shared, const GgufLatentWeights& banks,
                     WorkspaceArena& workspace, Tensor& y, cudaStream_t stream) {
    const std::int32_t tokens = m.ne[1];
    require(tokens > 0 && shaped(m, DType::BF16, kHidden, tokens), "m must be BF16 [2560, tokens]");
    require(shaped(ids, DType::I32, kTopK, tokens) && shaped(weights, DType::FP32, kTopK, tokens) &&
                shared.dtype == DType::FP32 && shared.numel() == tokens &&
                shaped(y, DType::FP32, kHidden, tokens),
            "ids, weights, shared and y must hold the call's tokens");
    const std::int32_t slices = (tokens + kGgufLatentSliceTokens - 1) / kGgufLatentSliceTokens;
    const std::int32_t step   = (tokens + slices - 1) / slices;
    for (std::int32_t first = 0; first < tokens; first += step) {
        const std::int32_t n = std::min(step, tokens - first);
        const Tensor m_s(static_cast<std::byte*>(m.data) + std::size_t(first) * kHidden * 2,
                         DType::BF16, {kHidden, n});
        const Tensor ids_s(static_cast<std::int32_t*>(ids.data) + std::size_t(first) * kTopK,
                           DType::I32, {kTopK, n});
        const Tensor weights_s(static_cast<float*>(weights.data) + std::size_t(first) * kTopK,
                               DType::FP32, {kTopK, n});
        const Tensor shared_s(static_cast<float*>(shared.data) + first, DType::FP32, {n});
        Tensor y_s(static_cast<float*>(y.data) + std::size_t(first) * kHidden, DType::FP32,
                   {kHidden, n});
        auto scope       = workspace.scope();
        const auto stage = moe_latent_gguf_begin(m_s, ids_s, shared_s, banks, workspace, stream);
        moe_latent_gguf_add(stage, ids_s, weights_s, banks, workspace, stream);
        moe_latent_gguf_finish(stage, banks, workspace, y_s, stream);
    }
}

void moe_latent_gguf_part(const Tensor& m, const Tensor& ids, const Tensor& weights,
                          const Tensor* shared, const GgufLatentWeights& banks,
                          std::int32_t group_begin, std::int32_t group_end,
                          WorkspaceArena& workspace, Tensor& fixed, cudaStream_t stream,
                          const GgufMoeSide* side) {
    check_banks(banks);
    const std::int32_t tokens = m.ne[1];
    require(group_begin >= 0 && group_begin <= group_end && group_end <= banks.groups,
            "the group range must lie in the layer");
    require(fixed.dtype == DType::I64 && fixed.is_contiguous() && fixed.data != nullptr &&
                fixed.numel() == std::int64_t(kHidden) * tokens,
            "fixed must be I64 [2560, tokens]");
    require(shaped(ids, DType::I32, kTopK, tokens), "ids must be I32 [10, tokens]");
    auto scope = workspace.scope();
    const auto stage = latent_begin(m, ids, shared, banks, group_begin, group_end, workspace,
                                    stream, side);
    Tensor kept = workspace.alloc(DType::I32, {kTopK, tokens});
    keep_groups_kernel<<<blocks_for(stage.pairs), 256, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), banks.labels, stage.pairs, group_begin,
        group_end, static_cast<std::int32_t*>(kept.data));
    CUDA_CHECK(cudaGetLastError());
    latent_add(stage, kept, weights, banks, workspace, stream);
    latent_decode(stage, stage_routing(stage, banks), banks, workspace, stream);
    if (stage.join != nullptr) { CUDA_CHECK(cudaStreamWaitEvent(stream, stage.join, 0)); }
    CUDA_CHECK(cudaMemcpyAsync(fixed.data, stage.fixed,
                               sizeof(unsigned long long) * kHidden * tokens,
                               cudaMemcpyDeviceToDevice, stream));
}

} // namespace ninfer::ops
