#include "core/weight.h"
#include "ops/common/blackwell_routes.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_a8_schedule.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"
#include "ops/linear_add/fp8/fp8_linear_add_epilogue.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Geometry, class Schedule, bool FullTokens>
void launch_mma(const Weight& weight, Tensor& residual, Fp8A8Workspace workspace,
                std::int32_t tokens, cudaStream_t stream) {
    constexpr int kRowTiles = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles   = (tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
    const int blocks        = kRowTiles * token_tiles;
    auto* output            = static_cast<__nv_bfloat16*>(residual.data);
    const Fp8AddResidualEpilogue epilogue{output, Geometry::kOutputRows};
    const Fp8ContiguousOutput destination{output, Geometry::kOutputRows};

    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        raise_dynamic_shared_memory(fp8_mma_kernel<Geometry, Schedule, FullTokens,
                                                   Fp8AddResidualEpilogue, Fp8ContiguousOutput>,
                                    Schedule::kSharedBytes);
    }
    fp8_mma_kernel<Geometry, Schedule, FullTokens>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), tokens, epilogue, destination);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Schedule>
void launch_region(const Weight& weight, Tensor& residual, Fp8A8Workspace workspace,
                   std::int32_t tokens, cudaStream_t stream) {
    if ((tokens % Schedule::kBlockTokens) == 0) {
        launch_mma<Geometry, Schedule, true>(weight, residual, workspace, tokens, stream);
    } else {
        launch_mma<Geometry, Schedule, false>(weight, residual, workspace, tokens, stream);
    }
}

// Per-T winners on the unlocked RTX 4090 (bench/ops/fp8_a8_schedule_bench.cu, suite
// 20261106-054406; cold L2, median of 9). The 64-row 4-stage tile leads T 17-64 (42.0-43.0 us,
// the 120a-tuned default at 46.1) but loses badly from T=96 (58.4); the 256-deep K tile leads
// T 96-192 (43.0-47.1, the default at 46.1-48.1) and T=512 (120.8 vs 133.1); the one-token-warp
// tile leads T=256 (60.4-61.5 vs 67.6); the 128-token tile leads T=384 (73.7 vs 82.9) and from
// T=1024 (198.7/332.8, the default at 206.8/379.9).
void launch_5120_6144(const Weight& weight, Tensor& residual, Fp8A8Workspace workspace,
                      std::int32_t tokens, cudaStream_t stream) {
    if (tokens <= 64) {
        launch_region<Fp8N5120K6144, Fp8A8Rows64Schedule>(weight, residual, workspace, tokens, stream);
    } else if (tokens <= 192) {
        launch_region<Fp8N5120K6144, Fp8A8K256Schedule>(weight, residual, workspace, tokens, stream);
    } else if (tokens <= 256) {
        launch_region<Fp8N5120K6144, Fp8A8Warps1Schedule>(weight, residual, workspace, tokens, stream);
    } else if (tokens <= 384) {
        launch_region<Fp8N5120K6144, Fp8A8Tokens128Schedule>(weight, residual, workspace, tokens,
                                                             stream);
    } else if (tokens <= 512) {
        launch_region<Fp8N5120K6144, Fp8A8K256Schedule>(weight, residual, workspace, tokens, stream);
    } else {
        launch_region<Fp8N5120K6144, Fp8A8Tokens128Schedule>(weight, residual, workspace, tokens,
                                                             stream);
    }
}

// The 256-deep K tile leads T 20-192 (105.5-123.0 us, the default at 126.0-136.2) and T=512
// (313.4 vs 356.4); the one-token-warp tile leads T=256 (163.8 vs 184.2); the 128-token tile
// leads T=384 (194.6 vs 219.1) and from T=1024 (538.6/928.8, the default at 573.4/1078.3).
void launch_5120_17408(const Weight& weight, Tensor& residual, Fp8A8Workspace workspace,
                       std::int32_t tokens, cudaStream_t stream) {
    if (tokens <= 192) {
        launch_region<Fp8N5120K17408, Fp8A8K256Schedule>(weight, residual, workspace, tokens, stream);
    } else if (tokens <= 256) {
        launch_region<Fp8N5120K17408, Fp8A8Warps1Schedule>(weight, residual, workspace, tokens, stream);
    } else if (tokens <= 384) {
        launch_region<Fp8N5120K17408, Fp8A8Tokens128Schedule>(weight, residual, workspace, tokens,
                                                              stream);
    } else if (tokens <= 512) {
        launch_region<Fp8N5120K17408, Fp8A8K256Schedule>(weight, residual, workspace, tokens, stream);
    } else {
        launch_region<Fp8N5120K17408, Fp8A8Tokens128Schedule>(weight, residual, workspace, tokens,
                                                              stream);
    }
}

} // namespace

void fp8_linear_add_a8_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                              WorkspaceArena& workspace, cudaStream_t stream) {
    require_fp8_a8_card();
    auto scope                   = workspace.scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(workspace, x.ne[1], weight.k);
    launch_fp8_a8_quantize(x, weight, scratch, stream);
    switch (resolve_fp8_geometry(weight.n, weight.k)) {
    case Fp8GeometryId::N5120K6144:
        launch_5120_6144(weight, residual, scratch, x.ne[1], stream);
        return;
    case Fp8GeometryId::N5120K17408:
        launch_5120_17408(weight, residual, scratch, x.ne[1], stream);
        return;
    case Fp8GeometryId::N14336K5120:
    case Fp8GeometryId::N16384K5120:
    case Fp8GeometryId::N34816K5120:
    case Fp8GeometryId::N248320K5120:
        break;
    }
    throw std::invalid_argument("fp8 linear_add: unsupported problem");
}

} // namespace ninfer::ops::detail
