#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Fp8Geometry<16384, 5120>;
using Gemv     = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using A8Mid    = Fp8A8K256Schedule;
using A8Large  = Fp8A8Tokens128Schedule;
using C2        = Fp8SimtSchedule<8, 2, 16, 2, 1, Fp8SimtActivationAccess::SharedPhase,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C4        = Fp8SimtSchedule<8, 2, 16, 4, 1, Fp8SimtActivationAccess::SharedPhase,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C8        = Fp8SimtSchedule<8, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C10       = Fp8SimtSchedule<8, 2, 16, 10, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 2>;
using FullChunk = Fp8SimtSchedule<8, 2, 16, 10, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;

Fp8Launch select_a16(std::int32_t tokens) {
    if (tokens == 1) return launch_fp8_gemv<Geometry, Gemv>;
    if (tokens == 10) return launch_fp8_simt<Geometry, 10, FullChunk, true>;
    if (tokens <= 2) return launch_fp8_simt<Geometry, 2, C2>;
    if (tokens <= 4) return launch_fp8_simt<Geometry, 4, C4>;
    if (tokens <= 8) return launch_fp8_simt<Geometry, 8, C8>;
    if (tokens <= 10) return launch_fp8_simt<Geometry, 10, C10>;
    throw std::logic_error("fp8 A16 chunk exceeds shape capacity");
}

// 4090-measured (forced-policy T-curve, suite 20261106-054406): A8 wins at T=17 (the A16 route
// is pathologically slow exactly at T=17: 184 us vs A8 100 us), A16 wins by 3-6% at T=20-25,
// and A8 wins by >= 5% from T=32 (19% at T=48, growing).
bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens == 17 || max_tokens >= 32; }

// Per-T A8 winners on the unlocked RTX 4090 (bench/ops/fp8_a8_schedule_bench.cu, suite
// 20261106-054406; cold L2, median of 9): the 256-deep K tile leads T 17-192 (96.3-111.6 us,
// the 120a-tuned default at 101.4-126.0) and the 128-token tile leads from T=256
// (122.9/219.1/324.6/836.6 at T=256/512/1024/2048, the default at 138.2/249.9/366.6/948.3).
void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    if (x.ne[1] <= 192) {
        launch_fp8_a8<Geometry, A8Mid>(x, weight, out, scratch, stream);
    } else {
        launch_fp8_a8<Geometry, A8Large>(x, weight, out, scratch, stream);
    }
}
} // namespace

const Fp8LinearShape kFp8N16384K5120{16384, 5120, launch_fp8_a16_chunks<10, select_a16>, launch_a8,
                                     uses_a8};
} // namespace ninfer::ops::detail
