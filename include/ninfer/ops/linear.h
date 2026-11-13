#pragma once

#include "core/weight.h"
#include "ninfer/ops/linear_policy.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops {

/**
 * Returns the caller-owned transient capacity required by Linear for every T in the inclusive
 * `[min_tokens,max_tokens]` interval. Invalid registered profiles, policies, or intervals throw;
 * a legal route that requires no transient storage returns zero.
 */
[[nodiscard]] std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                          std::int32_t input_rows,
                                                          LinearPolicy policy,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens);

/**
 * @brief Applies a bias-free matrix projection independently to every input column.
 *
 * @details The ideal mathematical result is
 *
 * @f[
 *   \mathrm{ideal}_{n,t} =
 *   \sum_{k=0}^{K-1}
 *     \mathrm{FP32Dequant}(w)_{n,k}\,\mathrm{FP32}(x_{k,t}).
 * @f]
 *
 * `out` stores a BF16 approximation of this ideal result under the named numerical criterion for
 * the selected private activation-compute path; with a weight linear_writes_fp32() names it may
 * instead be FP32, the unrounded FP32 accumulation.
 *
 * @par Logical tensors and layout
 * `x` is contiguous, non-null, 16-byte-aligned BF16 `[K,T]`, `w` has logical shape `[N,K]`, and
 * `out` is contiguous, non-null, 16-byte-aligned BF16 `[N,T]` (FP32 `[N,T]` is accepted for the
 * formats linear_writes_fp32() names only). Every logical extent is positive;
 * in particular, `T=0` is invalid rather than a no-op. Dimension zero is stored fastest. The Op has
 * no bias, activation, residual addition, or transpose mode.
 *
 * @par Supported execution domain
 * Registered execution uses RowSplit Q4_G64_FP16, Q5_G64_FP16, Q6_G64_FP16, or Q8_G32_FP16 weights
 * with FP16 scales, block-scaled NVFP4 weights, row-scaled FP8_E4M3FN_ROW_BF16 weights, plus
 * registered contiguous BF16 problems. Each format owns a finite registry of exact physical
 * weight problems and selects its kernel internally. A RowSplit Q4/Q5/Q6/Q8 problem no registry
 * names, and every one with an FP32 `out`, runs a runtime-shape route over the same templates (any
 * N, K a multiple of 128): a tensor-parallel rank's share of a projection, for instance. The
 * 248320x5120 vocabulary head is the exception for FP32 `out`: its Q4 and Q8 registered routes up
 * to T=32 (Q8: up to its K-split widths) and a Q6 device-profile route store their sums unrounded
 * instead. So does a
 * T2, FP8 or NVFP4 problem, in whole tiles of linear_share_units(): with A16 compute for T2 and for
 * a registered problem's FP32 result whatever the policy; for an FP8 problem no registry names
 * with A8 compute from T=17 where the policy permits it, and for such an NVFP4 problem with K of
 * 512 or more with A4 compute from T=9 where AllowA4 permits it. FP8 currently registers `[N,K]`
 * in `{[14336,5120], [16384,5120],
 * [34816,5120], [248320,5120], [5120,6144], [5120,17408]}` at every positive T. The current NVFP4
 * problems register the five non-vocabulary FP8 geometries and accept every positive T. Q8 also
 * registers `[5120,25600]` at every positive T. BF16 registers `[14336,5120]`,
 * `[5120,6144]`, and `[256,5120]` at every positive T; any other BF16 `[N,K]` (the Gemma 4 vision
 * tower's projections, for instance), and every BF16 problem with an FP32 `out`, runs a
 * runtime-shape GEMM, cuBLAS's tensor-core BF16 GEMM with an FP32 accumulator from T=64 on. Text and MTP packed-weight problems accept every positive
 * column extent T. Registered Vision problems accept raw-patch P in
 * `{4,8,...,131072}` or merged-token V in `[1,32768]`; a matrix column does not inherently
 * represent a text token. FP32 `x`, and FP32 `out` with another weight format, are unsupported.
 *
 * @par Numerical contract
 * Test fixture code materializes the persistent weight as its logical FP32 dequantized matrix.
 * The one Linear oracle accepts that matrix and the FP32 values represented by the BF16 activation,
 * evaluates every complete dot product with naive FP64 accumulation, and retains the FP64 result.
 * The BF16 or FP32 output is promoted and compared against that result. Output representation,
 * accumulator precision, activation quantization, staging, reduction order, and kernel schedule
 * are private implementation effects covered by the named tolerance for the selected
 * activation-compute path; none is copied into the oracle. Kernel, schedule, template instance,
 * host launcher, and T region do not create separate criteria inside one path.
 *
 * @par Compute policy
 * `policy` specifies the permitted private activation-compute set. A permission does not require a
 * corresponding low-precision route: the resolved plan may remain A16 when that is the qualified
 * choice. Every policy permits the existing A16 implementations of BF16 and Q4/Q5/Q6/Q8.
 * FP8 accepts all three policies; AllowA8 and AllowA4 permit the private resolver to select a
 * qualified A16 or A8 route for the problem and T. FP8 `[248320,5120]` retains A16
 * compute under every policy at every positive T. NVFP4 uses A16 for A16Only and
 * AllowA8; AllowA4 permits the private resolver to select either a qualified A16 route or
 * activation quantization to NVFP4 at every positive T. The selected route depends only on the
 * problem, T and whether `out` is FP32.
 *
 * @par Workspace
 * `workspace` is caller-owned call-scoped transient storage sized by
 * linear_workspace_capacity_bytes(). It must not overlap x, any weight plane, or out. Linear does
 * not allocate device memory internally.
 *
 * @param[in] x Contiguous, non-null, 16-byte-aligned BF16 input matrix `[K,T]`.
 * @param[in] w Logical weight matrix `[N,K]` in a registered persistent format and layout.
 * @param[out] out Contiguous, non-null, 16-byte-aligned BF16 output matrix `[N,T]`. It must not
 * overlap `x` or any weight plane.
 * @param[in] policy Permitted private activation-compute profiles.
 * @param[in,out] workspace Caller-owned transient arena.
 * @param[in] stream CUDA stream on which execution is enqueued.
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream);

/**
 * @brief Applies the A16-only form of the bias-free matrix projection.
 *
 * @details This overload admits only A16 compute and requires no transient workspace. All tensor,
 * weight, aliasing, and execution-domain requirements of the policy-bearing overload apply.
 *
 * @param[in] x Contiguous BF16 input matrix `[K,T]`.
 * @param[in] w Logical weight matrix `[N,K]` in a registered persistent format and layout.
 * @param[out] out Contiguous BF16 output matrix `[N,T]`.
 * @param[in] stream CUDA stream on which execution is enqueued.
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

/// One projection of linear_together: its weight, its output and the compute policy of its call.
struct LinearInto {
    const Weight* weight = nullptr;
    Tensor* out          = nullptr;
    LinearPolicy policy  = LinearPolicy::A16Only;
};

/**
 * @brief Applies several bias-free projections of one input, each as linear() applies it.
 *
 * @details Every `out` receives the bytes that `linear(x, *weight, *out, policy, workspace,
 * stream)` writes, under the same requirements on each problem. The BF16 problems whose own call
 * runs the runtime-shape route's GEMV form (no registered shape, T up to 16, K a multiple of 8, a
 * BF16 `out`) share one launch, four at a time, and every row runs that form's arithmetic; every
 * other problem takes a call of its own. The outputs must not overlap each other, `x` or any
 * weight plane. `workspace` is sized for the largest problem's linear_workspace_capacity_bytes().
 */
void linear_together(const Tensor& x, std::span<const LinearInto> list, WorkspaceArena& workspace,
                     cudaStream_t stream);

// Whether Linear writes an FP32 `out` for `format` and takes it at runtime shapes: BF16, GGUF and
// the RowSplit, FP8 and NVFP4 formats; a tensor-parallel rank's piece of a projection is one.
[[nodiscard]] bool linear_writes_fp32(QType format) noexcept;

// LinearShareUnits (ninfer/ops/linear_policy.h) of a `format` matrix.
[[nodiscard]] LinearShareUnits linear_share_units(QType format) noexcept;

} // namespace ninfer::ops
