#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Fp8Geometry<34816, 5120>;
using Gemv     = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using A8Small  = Fp8A8Rows64Schedule;
using A8Mid    = Fp8A8K256Schedule;
using A8Large  = Fp8A8Tokens128Schedule;
using C2        = Fp8SimtSchedule<8, 2, 16, 2, 1, Fp8SimtActivationAccess::SharedPhase,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C4        = Fp8SimtSchedule<8, 2, 16, 4, 1, Fp8SimtActivationAccess::SharedPhase,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using FullChunk = Fp8SimtSchedule<8, 2, 8, 4, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;

Fp8Launch select_a16(std::int32_t tokens) {
    if (tokens == 1) return launch_fp8_gemv<Geometry, Gemv>;
    if (tokens == 4) return launch_fp8_simt<Geometry, 4, FullChunk, true>;
    if (tokens <= 2) return launch_fp8_simt<Geometry, 2, C2>;
    if (tokens <= 4) return launch_fp8_simt<Geometry, 4, C4>;
    throw std::logic_error("fp8 A16 chunk exceeds shape capacity");
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 5; }

// Per-T A8 winners on the unlocked RTX 4090 (bench/ops/fp8_a8_schedule_bench.cu, suite
// 20261105-185306; cold L2, median of 9): the 64-row 4-stage tile leads T 5-128 (204.8-214.1 us,
// the 120a-tuned default at 229.4-236.5), the 256-deep K tile leads T 129-192 (236.5, the default
// at 245.8 and the 64-row tile at 279.6) and the 128-token tile leads from T=256 (280.4/481.3/
// 889.9/1753.1 at T=256/512/1024/2048, the default at 292.9/518.1/1003.5/1945.6).
void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    if (x.ne[1] <= 128) {
        launch_fp8_a8<Geometry, A8Small>(x, weight, out, scratch, stream);
    } else if (x.ne[1] <= 192) {
        launch_fp8_a8<Geometry, A8Mid>(x, weight, out, scratch, stream);
    } else {
        launch_fp8_a8<Geometry, A8Large>(x, weight, out, scratch, stream);
    }
}
} // namespace

const Fp8LinearShape kFp8N34816K5120{34816, 5120, launch_fp8_a16_chunks<4, select_a16>, launch_a8,
                                     uses_a8};
} // namespace ninfer::ops::detail
