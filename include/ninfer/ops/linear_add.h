#pragma once

// ninfer::ops - fused residual += W @ x.

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the A16-only transient capacity required by LinearAdd for every T in the inclusive
 * [min_tokens,max_tokens] interval. The QType and dimensions are the fixed implementation profile.
 * Invalid profiles or intervals throw; a legal static-zero route returns zero.
 */
[[nodiscard]] std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

[[nodiscard]] std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              LinearPolicy policy,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

/**
 * Op: linear_add
 *
 * Math / indexing:
 *   ideal[:,t] = residual[:,t] + Linear(x,w)[:,t].
 *
 * Logical shapes:
 *   Contiguous BF16 x [K,T] and residual [N,T]. Registered weights are Q4_G64_FP16 RowSplit
 *   [5120,6144], Q5_G64_FP16 RowSplit [5120,17408] or [5120,6144], Q8_G32_FP16 RowSplit
 *   [2048,4096], [2048,6144], [5120,6144] or [5120,17408], NVFP4
 *   BlockScaleK16M128x4 [5120,6144] or [5120,17408], row-scaled
 *   FP8_E4M3FN_ROW_BF16 [5120,6144] or [5120,17408], or BF16 Contiguous [5120,6144]. T may
 *   be any positive value.
 *
 * Numeric:
 *   The oracle reads a registered BF16 weight directly or exact-decodes a registered packed
 *   weight, then evaluates `ideal` naively in FP64 from the represented inputs. The updated BF16
 *   residual is promoted and compared directly with that result; output storage rounding belongs
 *   to LinearAdd's selected A16, A8, or A4 criterion, not the oracle. Production routes may fuse
 *   or materialize the projection and may choose their natural accumulator, activation
 *   quantization, staging, and workspace precision; those private choices are not semantic
 *   rounding boundaries.
 *   Within an arithmetic profile, full versus partial private token tiles must preserve each
 *   column's residual result; a live-column predicate must not change scaling/addition rounding.
 *
 * Compute policy:
 *   All policies permit the A16 implementations of Q4, Q5, Q8 and BF16. NVFP4 uses A16 for
 *   A16Only/AllowA8 and may use A4 under AllowA4. FP8 may use A8 under AllowA8/AllowA4.
 *   Each registration owns its production plan. A permissive policy
 *   allows the private resolver to select either qualified
 *   arithmetic profile; it does not itself prescribe a kernel.
 *
 * Effects:
 *   Updates the full residual tensor in place; x/weight must not alias residual.
 *
 * Workspace:
 *   Caller-owned transient storage reported by linear_add_workspace_capacity_bytes(), scoped to
 *   the call. A16 routes require no storage; quantized-activation routes use the reported capacity.
 *   There is no persistent state side effect.
 */
void linear_add(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena& ws,
                cudaStream_t stream);

void linear_add(const Tensor& x, const Weight& w, Tensor& residual, LinearPolicy policy,
                WorkspaceArena& ws, cudaStream_t stream);

/**
 * Whether the row-scaled FP8 [5120,6144] projection takes the fused norm for this weight at this
 * width. The answer is the FP8 A8 plan's route: true exactly when the legacy chain answers the
 * call with its A8 Tensor Core mainloop (T >= 17 under an A8-permitting policy that is not an
 * integer-A8 policy). A16Only and integer-A8 policies take the A16 route and are not eligible;
 * the unified chain's A8 is its TMA route, which this entry does not serve. The 5120x17408 problem
 * has no fused entry. This is a host-side route predicate. A true result does not validate the
 * weight payload.
 */
[[nodiscard]] bool linear_add_fused_gated_rmsnorm_fp8_eligible(const Weight& weight,
                                                               LinearPolicy policy,
                                                               std::int32_t tokens);

/**
 * Computes the SiLU-gated RMSNorm of `o` and the [5120,6144] projection of the result, adding the
 * product into `residual`, with one activation pass: a single kernel turns each `o` row into the
 * per-token E4M3 codes and FP32 scale the A8 mainloop then consumes, replacing the caller's
 * separate gated_rmsnorm and this route's separate quantise. `o` is BF16 [6144, tokens], `z` its
 * BF16 [6144, tokens] gate row, `residual` BF16 [5120, tokens], the weight
 * FP8_E4M3FN_ROW_BF16 [5120,6144], and `norm_weight` the 128 BF16 gain of the row's 128-wide
 * head: the 6144 row holds 48 heads and each head normalizes itself, x * rsqrt(sum(x*x)/128 +
 * eps) * norm_weight, then * silu(z) (no unit offset), rounded to BF16 before quantization. The
 * per-token E4M3 scale spans the whole 6144 row. The workspace capacity is the ordinary A8 call's
 * for this width (this entry allocates strictly less: the caller's BF16 normed row never exists).
 * Bit-for-bit the composition of gated_rmsnorm() on the head-grouped view followed by
 * linear_add() with the same policy: the norm half reproduces the D=128 launcher's per-head pair
 * ownership, accumulation order, warp reduction and single BF16 rounding, and the quantise half
 * applies the per-token amax/448 scale with SATFINITE E4M3 conversion the standalone quantiser
 * uses.
 */
void linear_add_fused_gated_rmsnorm_fp8(const Tensor& o, const Tensor& norm_weight, const Tensor& z,
                                        float eps, const Weight& weight, Tensor& residual,
                                        LinearPolicy policy, WorkspaceArena& workspace,
                                        cudaStream_t stream);

} // namespace ninfer::ops
