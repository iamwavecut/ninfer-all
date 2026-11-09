#include "core/weight.h"
#include "ops/common/blackwell_routes.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_a8_schedule.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_fused_norm_quantize.h"
#include "ops/linear/fp8/fp8_output.cuh"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Geometry = Fp8N34816K5120;
using A8Small  = Fp8A8Rows64Schedule;
using A8Mid    = Fp8A8K256Schedule;
using A8Large  = Fp8A8Tokens128Schedule;

constexpr int kIntermediate = Geometry::kOutputRows / 2;

template <class Schedule, bool FullTokens>
void launch_mma(const Weight& weight, Tensor& out, Fp8A8Workspace workspace, std::int32_t tokens,
                cudaStream_t stream) {
    static_assert((Schedule::kBlockRows % 2) == 0);
    using Rows = Fp8SwiGluRows<Schedule::kBlockRows / 2, kIntermediate>;
    constexpr int kRowTiles = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles   = (tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
    const int blocks        = kRowTiles * token_tiles;
    const Rows rows{};
    const Fp8SwiGluOutput output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};

    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        raise_dynamic_shared_memory(
            fp8_mma_kernel<Geometry, Schedule, FullTokens, Fp8IdentityEpilogue, Fp8SwiGluOutput,
                           Rows, true>,
            Schedule::kSharedBytes);
    }
    fp8_mma_kernel<Geometry, Schedule, FullTokens, Fp8IdentityEpilogue, Fp8SwiGluOutput, Rows, true>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), tokens, Fp8IdentityEpilogue{}, output,
            rows);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_problem(const Weight& weight, Tensor& out, Fp8A8Workspace workspace, std::int32_t tokens,
                    cudaStream_t stream) {
    if ((tokens % Schedule::kBlockTokens) == 0) {
        launch_mma<Schedule, true>(weight, out, workspace, tokens, stream);
    } else {
        launch_mma<Schedule, false>(weight, out, workspace, tokens, stream);
    }
}

} // namespace

void fp8_linear_swiglu_a8_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                 WorkspaceArena& workspace, cudaStream_t stream) {
    require_fp8_a8_card();
    auto scope = workspace.scope();
    const Fp8A8Workspace scratch =
        allocate_fp8_a8_workspace(workspace, x.ne[1], Geometry::kInputRows);
    launch_fp8_a8_quantize(x, weight, scratch, stream);
    // Per-T winners on the unlocked RTX 4090 (bench/ops/fp8_a8_schedule_bench.cu, suite
    // 20261105-185306, measured on the plain 34816x5120 A8 GEMM this epilogue shares; cold L2,
    // median of 9): the 64-row 4-stage tile leads T 1-128 (204.8-214.1 us vs the 120a-tuned
    // default's 229.4-236.5), the 256-deep K tile leads T 129-192 (236.5 vs 245.8) and the
    // 128-token tile leads from T=256 (280.4/481.3/889.9/1753.1 at T=256/512/1024/2048, the
    // default at 292.9/518.1/1003.5/1945.6).
    if (x.ne[1] <= 128) {
        launch_problem<A8Small>(weight, out, scratch, x.ne[1], stream);
    } else if (x.ne[1] <= 192) {
        launch_problem<A8Mid>(weight, out, scratch, x.ne[1], stream);
    } else {
        launch_problem<A8Large>(weight, out, scratch, x.ne[1], stream);
    }
}

// The A8 route with the Offset RMSNorm folded into the quantize: the fused producer writes the
// same Fp8A8Workspace the separate launch_fp8_a8_quantize would (bit for bit) and the MMA above
// runs unchanged, so the result is the composition of rmsnorm() into BF16 storage and this route.
void fp8_linear_swiglu_fused_norm_launch(const Tensor& residual, const Tensor& norm_weight,
                                         float eps, const Weight& weight, Tensor& out,
                                         WorkspaceArena& workspace, cudaStream_t stream) {
    require_fp8_a8_card();
    auto scope = workspace.scope();
    const Fp8A8Workspace scratch =
        allocate_fp8_a8_workspace(workspace, residual.ne[1], Geometry::kInputRows);
    launch_fp8_fused_rmsnorm_quantize(residual, norm_weight, eps, Geometry::kInputRows, scratch,
                                      stream);
    if (residual.ne[1] <= 128) {
        launch_problem<A8Small>(weight, out, scratch, residual.ne[1], stream);
    } else if (residual.ne[1] <= 192) {
        launch_problem<A8Mid>(weight, out, scratch, residual.ne[1], stream);
    } else {
        launch_problem<A8Large>(weight, out, scratch, residual.ne[1], stream);
    }
}

} // namespace ninfer::ops::detail
