#include "core/weight.h"
#include "ops/common/blackwell_routes.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/linear/fp8/fp8_a8_schedule.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Geometry = Fp8N16384K5120;

// 4090-measured per-token dispatch (fp8_a8_schedule_bench 16384x5120, cold L2, median of 9):
//   T <= 192:  bk256  96.3-111.6 us  (dft 101.4-126.0)
//   T >= 256:  t128   122.9-836.6 us (dft 138.2-948.3)
using A8Mid   = Fp8A8K256Schedule;
using A8Large = Fp8A8Tokens128Schedule;

static_assert((Fp8GdnInputOutput::kQkvRows % A8Mid::kBlockRows) == 0);
static_assert((Fp8GdnInputOutput::kZRows % A8Mid::kBlockRows) == 0);
static_assert((Fp8GdnInputOutput::kQkvRows % A8Large::kBlockRows) == 0);
static_assert((Fp8GdnInputOutput::kZRows % A8Large::kBlockRows) == 0);

template <class Schedule, bool FullTokens>
void launch_mma(const Weight& weight, Tensor& qkv, Tensor& z, Fp8A8Workspace workspace,
                std::int32_t tokens, cudaStream_t stream) {
    constexpr int kRowTiles = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles   = (tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
    const int blocks        = kRowTiles * token_tiles;
    const Fp8GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv.data),
                                   static_cast<__nv_bfloat16*>(z.data)};

    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        raise_dynamic_shared_memory(
            fp8_mma_kernel<Geometry, Schedule, FullTokens, Fp8IdentityEpilogue, Fp8GdnInputOutput>,
            Schedule::kSharedBytes);
    }
    fp8_mma_kernel<Geometry, Schedule, FullTokens>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), tokens, Fp8IdentityEpilogue{},
            output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_problem(const Weight& weight, Tensor& qkv, Tensor& z, Fp8A8Workspace workspace,
                    std::int32_t tokens, cudaStream_t stream) {
    if ((tokens % Schedule::kBlockTokens) == 0) {
        launch_mma<Schedule, true>(weight, qkv, z, workspace, tokens, stream);
    } else {
        launch_mma<Schedule, false>(weight, qkv, z, workspace, tokens, stream);
    }
}

} // namespace

void fp8_gdn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             Fp8A8Workspace workspace, cudaStream_t stream) {
    require_fp8_a8_card();
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const std::int32_t tokens = x.ne[1];
    if (tokens <= 192) {
        launch_problem<A8Mid>(weight, qkv, z, workspace, tokens, stream);
    } else {
        launch_problem<A8Large>(weight, qkv, z, workspace, tokens, stream);
    }
}

} // namespace ninfer::ops::detail
