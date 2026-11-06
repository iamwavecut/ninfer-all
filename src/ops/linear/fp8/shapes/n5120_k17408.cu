#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Fp8Geometry<5120, 17408>;
using Gemv     = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using A8Mid    = Fp8A8K256Schedule;
using A8T256   = Fp8A8Warps1Schedule;
using A8Large  = Fp8A8Tokens128Schedule;
using C2        = Fp8SimtSchedule<8, 2, 16, 2, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C4        = Fp8SimtSchedule<8, 2, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C8        = Fp8SimtSchedule<8, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C12       = Fp8SimtSchedule<8, 2, 16, 12, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::TokenTilesContiguous, 2>;
using C16       = Fp8SimtSchedule<8, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::TokenTilesContiguous, 2>;
using C24       = Fp8SimtSchedule<8, 2, 16, 12, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::TokenTilesContiguous, 2>;
using FullChunk = Fp8SimtSchedule<8, 2, 8, 24, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using Full17    = Fp8SimtSchedule<8, 2, 16, 17, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using Full18    = Fp8SimtSchedule<8, 2, 8, 18, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using Full19    = Fp8SimtSchedule<8, 2, 8, 19, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using Full20    = Fp8SimtSchedule<8, 2, 8, 20, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;

Fp8Launch select_a16(std::int32_t tokens) {
    if (tokens == 1) return launch_fp8_gemv<Geometry, Gemv>;
    if (tokens == 24) return launch_fp8_simt<Geometry, 24, FullChunk, true>;
    // These short SIMT extents retain a measured advantage over padded capacities.
    if (tokens == 17) return launch_fp8_simt<Geometry, 17, Full17, true>;
    if (tokens == 18) return launch_fp8_simt<Geometry, 18, Full18, true>;
    if (tokens == 19) return launch_fp8_simt<Geometry, 19, Full19, true>;
    if (tokens == 20) return launch_fp8_simt<Geometry, 20, Full20, true>;
    if (tokens <= 2) return launch_fp8_simt<Geometry, 2, C2>;
    if (tokens <= 4) return launch_fp8_simt<Geometry, 4, C4>;
    if (tokens <= 8) return launch_fp8_simt<Geometry, 8, C8>;
    if (tokens <= 12) return launch_fp8_simt<Geometry, 12, C12>;
    if (tokens <= 16) return launch_fp8_simt<Geometry, 16, C16>;
    if (tokens <= 24) return launch_fp8_simt<Geometry, 24, C24>;
    throw std::logic_error("fp8 A16 chunk exceeds shape capacity");
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

// Per-T A8 winners on the unlocked RTX 4090 (bench/ops/fp8_a8_schedule_bench.cu, suite
// 20261105-185306; cold L2, median of 9): the 256-deep K tile leads T 17-192 (105.5-123.0 us,
// the 120a-tuned default at 126.0-136.2) and T=512 (315.4 vs 365.6); the one-token-warp tile
// leads T=256 (163.8 vs 184.2); the 128-token tile leads from T=1024 (538.6/928.8 at T=1024/2048,
// the default at 573.4/1078.3).
void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    if (x.ne[1] <= 192) {
        launch_fp8_a8<Geometry, A8Mid>(x, weight, out, scratch, stream);
    } else if (x.ne[1] <= 256) {
        launch_fp8_a8<Geometry, A8T256>(x, weight, out, scratch, stream);
    } else if (x.ne[1] <= 512) {
        launch_fp8_a8<Geometry, A8Mid>(x, weight, out, scratch, stream);
    } else {
        launch_fp8_a8<Geometry, A8Large>(x, weight, out, scratch, stream);
    }
}
} // namespace

const Fp8LinearShape kFp8N5120K17408{5120, 17408, launch_fp8_a16_chunks<24, select_a16>,
                                     launch_a8, uses_a8};
} // namespace ninfer::ops::detail
