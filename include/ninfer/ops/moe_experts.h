#pragma once

#include "core/arena.h"
#include "core/qtype.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/weight_input.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops {

/**
 * The routed and shared experts of Qwen3.8-Flash-Next's MoE over BF16 weights (the debug and
 * oracle representation), after moe_route:
 *
 *   e_j(x) = down_j . (silu(gate_j . x) * (up_j . x))          expert width 640
 *   y      = sum_{k<10} weights[k] e_{ids[k]}(m) + shared * e_shared(m)
 *
 * `m` BF16 [2560, tokens]; `ids` I32 [10, tokens], `weights` FP32 [10, tokens], `shared` FP32
 * [tokens] from moe_route; `gate_up` BF16 [2560, 1280, 512] (each expert's 640 gate rows then 640
 * up rows), `down` BF16 [640, 2560, 512], `shared_gate_up` BF16 [2560, 1280], `shared_down` BF16
 * [640, 2560]; `y` FP32 [2560, tokens]. The oracle evaluates in FP64 from the represented inputs;
 * y is compared as FP32 by relative L2 and gross error.
 */
[[nodiscard]] std::size_t moe_experts_bf16_workspace_bytes(std::int32_t tokens);

void moe_experts_bf16(const Tensor& m, const Tensor& ids, const Tensor& weights,
                      const Tensor& shared, const Tensor& gate_up, const Tensor& down,
                      const Tensor& shared_gate_up, const Tensor& shared_down,
                      WorkspaceArena& workspace, Tensor& y, cudaStream_t stream);

/**
 * The same experts over GGUF block banks (an imported GSQ-RCO release), reached through tables of
 * expert base pointers, so an expert may sit in a device bank, a cache slot or mapped host memory.
 * `gate` and `up` hold each expert's 640 rows of 2560 values, `down` its 2560 rows of 640; the
 * routed tables have `experts` entries (up to 512), the shared expert's tables one. Every entry
 * points at device-readable memory in the table's block type, rows `row_bytes` apart.
 *
 * The products quantize their activation to ggml's q8_1 as llama.cpp does (the format's
 * arithmetic): m once per token, the middle silu(gate . m) * (up . m) after its BF16 store. Each
 * expert's contribution weights[k] * e (shared * e_shared for the shared expert) is accumulated in
 * 2^-32 fixed point, so y does not depend on the order the experts run in, and stored as FP32.
 * The oracle independently decodes stored blocks and evaluates the complete expert mathematics
 * in FP64 from the public BF16 input, without the implementation's internal q8_1 quantization or
 * BF16 middle cast; y is compared as FP32 by relative L2 and gross error. Q2_0 has an independent
 * CPU decoder in its qualification; other formats' existing fixtures still use the GPU decoder.
 *
 * Up to eight tokens run ggml-style vector products. A wider call needs `device_resident` banks
 * and runs all three projections through ggml's integer tensor-core matrix kernel (with the same
 * q8_1 arithmetic) where the block type has one; in a call of at most kGgufMoeSplitTokens tokens,
 * an expert with at most kGgufMoeVectorPairs pairs keeps the vector products instead. The choice
 * depends on the call's width and the expert's pairs only, so a placement that stages the missing
 * experts, or computes some on CPU threads (gguf_expert_cpu, which reproduces the vector
 * products), gives the bits of device banks. The tables of a wide call point at device memory
 * only, and at least 256 zero bytes follow the last expert of every down bank (and of a slot pool),
 * since the matrix kernel reads a 640-value row in three 256-value steps.
 */
// In a call of 9 to kGgufMoeSplitTokens tokens, an expert with at most this many pairs takes the
// vector products. Wider calls are prompt chunks, which host experts' slot pools may group into
// calls of other widths than device experts run: there every expert takes the matrix kernel,
// whose result for a pair does not depend on the call's other pairs.
inline constexpr std::int32_t kGgufMoeVectorPairs = 8;
inline constexpr std::int32_t kGgufMoeSplitTokens = 64;

struct GgufExpertTable {
    QType format               = QType::GGUF_Q8_0;
    const void* const* experts = nullptr; // device array of expert base pointers
    std::int64_t row_bytes     = 0;
    bool aligned16             = false;
};

struct GgufMoeWeights {
    GgufExpertTable gate, up, down;
    GgufExpertTable shared_gate, shared_up, shared_down;
    // Entries of the routed tables: 512, or the 256 an expert-pruned release keeps.
    std::int32_t experts = 512;
    // Every entry in device memory, zeros past each down bank (see above); required of a call of
    // more than eight tokens.
    bool device_resident = false;
    // The shared expert's width: its 640 middle values, or under tensor parallelism a device's
    // run of them, a multiple of 32 (the rows of its gate and up tables and the values of each down
    // row, whose rows are then `shared_down.row_bytes` apart). A narrower shared expert takes the
    // vector products at decode widths.
    std::int32_t shared_width = 640;
    // IQ1_M tables of a wide call take the matrix kernel too once the call routes enough rows to
    // each expert; false keeps them on the vector products (EngineOptions::iq1m_expert_matrix).
    bool iq1m_matrix = true;
};

[[nodiscard]] std::size_t moe_experts_gguf_workspace_bytes(std::int32_t tokens);
void moe_experts_gguf(const Tensor& m, const Tensor& ids, const Tensor& weights,
                      const Tensor& shared, const GgufMoeWeights& banks, WorkspaceArena& workspace,
                      Tensor& y, cudaStream_t stream);

/**
 * The same sum in stages, for banks whose missing experts arrive in device memory, or are computed
 * elsewhere, while the resident ones run. begin quantizes m and zeroes the 2^-32 fixed-point sum in
 * `workspace`; the caller keeps that workspace scope open until finish. Each add sums the routed
 * pairs whose `ids` entry is non-negative (a negative entry is a pair another stage adds) through
 * `banks`, and the shared expert when `shared` is given. add_products adds `*count` (at most
 * `max_count`) externally computed pair values, FP32 [count][2560] with their pair indices in
 * `pairs`, weighted by `weights` and rounded per pair as the kernels round theirs; its arrays may be
 * mapped host memory. finish stores y. Adds that divide the experts (every pair of an expert in the
 * same add) give exactly the single call's sum; external products carry their own arithmetic.
 */
struct GgufMoeStage {
    std::int32_t tokens       = 0;
    void* activation          = nullptr;
    unsigned long long* fixed = nullptr;
};

[[nodiscard]] GgufMoeStage moe_experts_gguf_begin(const Tensor& m, WorkspaceArena& workspace,
                                                  cudaStream_t stream);
void moe_experts_gguf_add(const GgufMoeStage& stage, const Tensor& m, const Tensor& ids,
                          const Tensor& weights, const Tensor* shared,
                          const GgufMoeWeights& banks, WorkspaceArena& workspace,
                          cudaStream_t stream);
void moe_experts_gguf_add_products(const GgufMoeStage& stage, const float* values,
                                   const std::int32_t* pairs, const std::int32_t* count,
                                   std::int32_t max_count, const Tensor& weights,
                                   cudaStream_t stream);
void moe_experts_gguf_finish(const GgufMoeStage& stage, Tensor& y, cudaStream_t stream);

// A second stream on which the decode form runs the shared expert concurrently with the routed
// experts, and the events that fork it from the call's stream and join it back before the sum is
// read. All three are the caller's, reused call after call.
struct GgufMoeSide {
    cudaStream_t stream = nullptr;
    cudaEvent_t fork    = nullptr;
    cudaEvent_t join    = nullptr;
};

/**
 * The decode form, up to eight tokens over banks whose gate and up share a block type: the same
 * sum in a few launches instead of a sort, memsets, activation passes and a store per bank, with
 * every row's arithmetic unchanged (moe_experts_gguf takes it whenever it applies). begin runs the
 * pairs whose `ids` entry is non-negative and the shared expert; decode_missing adds the up rows of
 * the pairs `ids` of a second stage lists, through `banks` (missing experts staged elsewhere);
 * finish runs every down row from the two id sets and their banks, adds external products and
 * stores y. The workspace scope stays open from begin to finish. Given a side stream, begin runs
 * the shared expert there and finish waits for it.
 */
struct GgufMoeDecode {
    std::int32_t tokens       = 0;
    void* middle              = nullptr; // BF16 [tokens * 10][640]
    void* shared_middle       = nullptr; // BF16 [tokens][640]
    unsigned long long* fixed = nullptr; // [tokens][2560] and a completion counter
    // The call's preparation: every slot's pairs, the tokens' inputs and the pairs' middles
    // quantized once for all blocks.
    void* records = nullptr;
    void* inputs  = nullptr;
    void* middles = nullptr;
    cudaEvent_t join = nullptr; // the shared expert's side stream, when it ran on one
};
[[nodiscard]] bool moe_experts_gguf_decode_supported(const GgufMoeWeights& banks,
                                                     std::int32_t tokens);
[[nodiscard]] std::size_t moe_experts_gguf_decode_workspace_bytes(std::int32_t tokens);
[[nodiscard]] GgufMoeDecode moe_experts_gguf_decode_begin(const Tensor& m, const Tensor& ids,
                                                          const Tensor& shared,
                                                          const GgufMoeWeights& banks,
                                                          WorkspaceArena& workspace,
                                                          cudaStream_t stream,
                                                          const GgufMoeSide* side = nullptr);
void moe_experts_gguf_decode_missing(const GgufMoeDecode& state, const Tensor& m,
                                     const Tensor& ids, const GgufMoeWeights& banks,
                                     cudaStream_t stream);
void moe_experts_gguf_decode_finish(const GgufMoeDecode& state, const Tensor& ids,
                                    const Tensor* missing_ids, const Tensor& weights,
                                    const GgufMoeWeights& banks, const GgufMoeWeights* missing_banks,
                                    const float* products, const std::int32_t* product_pairs,
                                    const std::int32_t* product_count, Tensor& y,
                                    cudaStream_t stream);

/**
 * Expert parallelism over GGUF banks: one device's part of moe_experts_gguf's sum. The part runs
 * the routed pairs whose `ids` entry lies in [expert_begin, expert_end), through `banks`, whose
 * other routed entries are never read, and the shared expert when `shared` is given; `fixed` I64
 * [2560, tokens] receives the 2^-32 fixed-point sum of their contributions. The routes, weights
 * and m are the whole call's, the same on every device. With parts whose ranges partition the
 * routed experts and the shared expert in exactly one, moe_experts_gguf_combine stores the y
 * moe_experts_gguf stores, bit for bit: each contribution is computed and rounded on its own, and
 * the parts add as integers. A part may instead hold a run of the shared expert's width
 * (`shared_width`: its gate and up rows, the matching values of every down row), each part its own
 * run; the middle values are then the whole expert's, and each part rounds its down rows' dot over
 * its run on its own, so the combined y is close to the whole call's, not identical. `parts` of
 * combine are device memory of the stream's device.
 */
[[nodiscard]] std::size_t moe_experts_gguf_part_workspace_bytes(std::int32_t tokens);
void moe_experts_gguf_part(const Tensor& m, const Tensor& ids, const Tensor& weights,
                           const Tensor* shared, const GgufMoeWeights& banks,
                           std::int32_t expert_begin, std::int32_t expert_end,
                           WorkspaceArena& workspace, Tensor& fixed, cudaStream_t stream,
                           const GgufMoeSide* side = nullptr);
void moe_experts_gguf_combine(std::span<const Tensor> parts, Tensor& y, cudaStream_t stream);

/**
 * The shared expert of moe_experts_gguf alone, with its arithmetic there: shared * e_shared(m)
 * added into `fixed` I64 [2560, tokens] (2^-32 fixed point, zeroed by the caller), through the
 * decode kernels at up to eight tokens where its banks allow them and the bank pass otherwise
 * (device-resident shared banks past eight tokens). Allocations stay in the caller's workspace
 * scope until the sum is read. Given `side`, a call of up to eight tokens runs there and returns
 * side->join, recorded after its work, for the caller to wait on before it reads the sum; otherwise
 * it returns nullptr.
 */
[[nodiscard]] cudaEvent_t moe_experts_gguf_shared(const Tensor& m, const Tensor& shared,
                                                  const GgufMoeWeights& banks,
                                                  WorkspaceArena& workspace,
                                                  unsigned long long* fixed, cudaStream_t stream,
                                                  const GgufMoeSide* side = nullptr);
[[nodiscard]] std::size_t moe_experts_gguf_shared_workspace_bytes(std::int32_t tokens);

/**
 * Edge0's latent MoE over GGUF block banks: Qwen3.8-Flash-Next's router and shared expert, with
 * routed experts that work in a latent width L (1792 or 1280) of one of G = 8 projection groups:
 *
 *   xl[t, g] = c * (D_g . q8_1(m[t]))                       D_g [L, 2560], c FP32 [L] or ones
 *   a_p      = q8_1(xl[t, g_e])                             g_e = labels[e], pair p = (t, e, w)
 *   h_p      = silu(s_gate[e] * (gate_e . a_p)) * (s_up[e] * (up_e . a_p))   gate_e, up_e [640, L]
 *   S[t, g]  = sum of w * (s_down[e] * (down_e . q8_1(h_p))) over token t's pairs with g_e = g
 *   y[t]     = sum over the groups g token t routes to of U_g . q8_1(S[t, g]) + shared * e_shared(m)
 *
 * with down_e [L, 640], U_g [2560, L], and s_* FP32 per-expert multipliers (ones when absent).
 * Each product rounds once in FP32; the multipliers apply to its FP32 result, h_p is stored as
 * BF16 before its q8_1 quantization, and S and y are 2^-32 fixed-point sums as in
 * moe_experts_gguf, so y does not depend on the order or placement in which the pairs run.
 * edge0-strata applies U to each pair's q8_1(s_down * down_e . q8_1(h_p)) before the weighted sum;
 * the two agree in exact arithmetic, and the group sum needs at most 8 U products per token
 * instead of 10. A group no pair of a token routes to costs no D or U work.
 *
 * Up to eight tokens run the vector products; a wider call needs `device_resident` banks and runs
 * D, U and each routed projection on the matrix kernel where the block type has one (the routed
 * experts with at most kGgufMoeVectorPairs pairs of a call of up to kGgufMoeSplitTokens tokens keep
 * the vector products), as moe_experts_gguf chooses. The choice depends on the call's width and
 * routing only. Encoder, decoder, labels and scales are device memory; routed tables may point at
 * banks, cache slots or mapped host memory as in moe_experts_gguf (device memory past eight tokens,
 * with 256 zero bytes after the last expert of the down bank).
 *
 * The independent oracle decodes every stored block and the scales, and evaluates the formula in
 * FP64 from the public BF16 input with the q8_1 points above; y is compared as FP32 by relative L2
 * and gross error.
 */
struct GgufLatentWeights {
    // Routed experts: gate and up 640 rows of `width` values, down `width` rows of 640.
    GgufExpertTable gate, up, down;
    // `groups` entries each: D_g `width` rows of 2560 values, U_g 2560 rows of `width`.
    GgufExpertTable encode, decode;
    GgufExpertTable shared_gate, shared_up, shared_down;
    const std::int32_t* labels  = nullptr; // device I32 [experts], each below `groups`
    const float* expert_scales  = nullptr; // device FP32 [experts][3] (gate, up, down), or ones
    const float* column_scale   = nullptr; // device FP32 [width], or ones
    std::int32_t width          = 0;       // L: a multiple of 256
    std::int32_t groups         = 8;
    std::int32_t experts        = 512;
    std::int32_t shared_width   = 640;
    bool device_resident        = false;
};

// moe_latent_gguf runs a call of more than kGgufLatentSliceTokens tokens in equal slices of at most
// that many, each wider than kGgufMoeSplitTokens, so that every product takes the route the whole
// call's would: the slices store the unsliced call's y, bit for bit, in the workspace of one slice.
// The staged form below runs its call whole.
inline constexpr std::int32_t kGgufLatentSliceTokens = 2048;

// A call of `tokens` tokens through the staged form; moe_latent_gguf needs that of
// min(tokens, kGgufLatentSliceTokens).
[[nodiscard]] std::size_t moe_latent_gguf_workspace_bytes(std::int32_t tokens, std::int32_t width,
                                                          std::int32_t groups);
void moe_latent_gguf(const Tensor& m, const Tensor& ids, const Tensor& weights,
                     const Tensor& shared, const GgufLatentWeights& banks,
                     WorkspaceArena& workspace, Tensor& y, cudaStream_t stream);

/**
 * The same sum in stages, for routed experts whose missing ones arrive in device memory or are
 * computed elsewhere. begin takes the whole call's `ids`: it runs the shared expert (on `side`
 * when given), D for every group the tokens route to, and zeroes the latent and output sums; the
 * workspace scope stays open until finish. Each add runs the routed pairs whose entry of its `ids`
 * is non-negative through `banks`' routed tables. add_products adds `*count` (at most `max_count`)
 * externally computed down products, FP32 [count][width] before s_down, with their pair indices
 * in `pairs`; they take s_down, the pair weight and the fixed-point rounding here, as the kernels'
 * products do. finish runs U and stores y. Adds that divide the experts (every pair of an expert in
 * one add) give exactly the single call's sum.
 *
 * In a call of up to kGgufMoeSplitTokens tokens `inputs` holds the latent activations of every
 * pair as the vector kernel reads them (planar q8_1, [tokens * 10][width]), so a CPU product of a
 * pair reads the GPU's quantized input; wider calls leave it null.
 */
// The layout of GgufLatentStage::inputs over `pairs` pairs: int8 codes [pairs][width] from byte 0,
// then binary16 (d, s) pairs [pairs][width / 32] from moe_latent_gguf_input_scales.
[[nodiscard]] constexpr std::size_t moe_latent_gguf_input_scales(std::int32_t pairs,
                                                                 std::int32_t width) {
    return (std::size_t(width) * std::size_t(pairs) + 15) / 16 * 16;
}
[[nodiscard]] constexpr std::size_t moe_latent_gguf_input_bytes(std::int32_t pairs,
                                                                std::int32_t width) {
    return moe_latent_gguf_input_scales(pairs, width) + std::size_t(pairs) * (width / 32) * 4;
}

struct GgufLatentStage {
    std::int32_t tokens          = 0;
    std::int32_t pairs           = 0;    // routed pairs: tokens * 10
    unsigned long long* fixed    = nullptr; // [tokens][2560]
    unsigned long long* latent   = nullptr; // S [tokens * groups][width]
    float* encoded               = nullptr; // xl [tokens * groups][width]
    std::int32_t* sources        = nullptr; // [pairs]: the xl column of each pair, or -1
    std::int32_t* group_ids      = nullptr; // [tokens * groups]: g, or -1 for a group not routed to
    float* group_weights         = nullptr; // [tokens * groups]: ones
    void* group_routing          = nullptr; // the groups' MoeRouting arrays
    void* inputs                 = nullptr; // the pairs' latent activations (vector layout)
    cudaEvent_t join             = nullptr; // the shared expert's side stream, when it ran on one
};

[[nodiscard]] GgufLatentStage moe_latent_gguf_begin(const Tensor& m, const Tensor& ids,
                                                    const Tensor& shared,
                                                    const GgufLatentWeights& banks,
                                                    WorkspaceArena& workspace, cudaStream_t stream,
                                                    const GgufMoeSide* side = nullptr);
void moe_latent_gguf_add(const GgufLatentStage& stage, const Tensor& ids, const Tensor& weights,
                         const GgufLatentWeights& banks, WorkspaceArena& workspace,
                         cudaStream_t stream);
void moe_latent_gguf_add_products(const GgufLatentStage& stage, const float* values,
                                  const std::int32_t* pairs, const std::int32_t* count,
                                  std::int32_t max_count, const Tensor& ids,
                                  const Tensor& weights, const GgufLatentWeights& banks,
                                  cudaStream_t stream);
void moe_latent_gguf_finish(const GgufLatentStage& stage, const GgufLatentWeights& banks,
                            WorkspaceArena& workspace, Tensor& y, cudaStream_t stream);

/**
 * Group parallelism: one device's part of moe_latent_gguf's sum, the groups [group_begin,
 * group_end) with their D_g, U_g and routed experts, and the shared expert when `shared` is given.
 * `fixed` I64 [2560, tokens] receives the part's 2^-32 fixed-point sum; with parts whose ranges
 * partition the groups and the shared expert in exactly one, moe_experts_gguf_combine stores the y
 * moe_latent_gguf stores, bit for bit. `banks` tables of other groups and their experts are never
 * read. A part may hold a run of the shared expert's width, as moe_experts_gguf_part's may; given
 * `side`, its shared expert runs there at up to eight tokens.
 */
void moe_latent_gguf_part(const Tensor& m, const Tensor& ids, const Tensor& weights,
                          const Tensor* shared, const GgufLatentWeights& banks,
                          std::int32_t group_begin, std::int32_t group_end,
                          WorkspaceArena& workspace, Tensor& fixed, cudaStream_t stream,
                          const GgufMoeSide* side = nullptr);

/**
 * One native expert projection bank. `experts` is a caller-owned device array of prepared
 * Weight operands, all in `format`: BF16 or Q2/Q4/Q5/Q6/Q8 row-split weights. The operands
 * retain their actual code/high/scale planes and layout; no selected-weight repack is needed.
 * `integer_a8` selects private group-32 activation quantization and signed-byte dot products;
 * BF16 banks take A16 only. Preparation validates the represented shapes and storage.
 */
struct NativeExpertTable {
    QType format = QType::BF16;
    const Weight* experts = nullptr;
    bool integer_a8 = false;
};

struct NativeMoeWeights {
    NativeExpertTable gate, up, down;
    NativeExpertTable shared_gate, shared_up, shared_down;
    std::int32_t experts = 512;
};

[[nodiscard]] Weight prepare_native_expert(const WeightInput& input, std::int32_t rows,
                                           std::int32_t columns, bool integer_a8);
[[nodiscard]] std::size_t moe_experts_native_workspace_bytes(std::int32_t tokens);

/**
 * Residual-free Flash-Next experts with the same represented inputs, top-10 routing and FP32
 * output as moe_experts_bf16. Routed banks have `experts` entries (10..512), shared banks one;
 * gate/up have 640 rows of 2560 values, down 2560 rows of 640. Each projection keeps its own
 * format and compute profile. Each call supports 1..65535 tokens and enqueues without host synchronization.
 *
 * The independent oracle decodes the original stored words and evaluates complete SwiGLU and
 * weighted merge in FP64 from public BF16 m. Middle products, activation quantization and
 * reduction order are private arithmetic, not oracle boundaries. Repeats with fixed operands
 * and profile are bit-exact. Optional `parts` is U8 [10,tokens]: zero excludes that routed
 * contribution, nonzero evaluates it; null includes every route. The shared expert always
 * contributes. Excluded experts' weight operands are never read. A caller may compute those
 * experts on the CPU and add their weighted sum after both parts finish.
 * Caller-owned workspace and operand tables are graph-stable;
 * live inputs, weights, output and workspace do not overlap.
 */
void moe_experts_native(const Tensor& m, const Tensor& ids, const Tensor& weights,
                        const Tensor& shared, const NativeMoeWeights& banks,
                        const Tensor* parts, WorkspaceArena& workspace, Tensor& y, cudaStream_t stream);

} // namespace ninfer::ops
