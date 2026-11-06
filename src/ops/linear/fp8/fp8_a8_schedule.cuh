#pragma once
#include "ops/linear/fp8/fp8_a8_mma.cuh"

namespace ninfer::ops::detail {
using Fp8A8DefaultSchedule =
    Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg, Fp8MmaFragmentPipeline::PingPong,
                   Fp8MmaRaster::TokenFast>;

// Per-range A8 winners measured on the unlocked RTX 4090 (bench/ops/fp8_a8_schedule_bench.cu,
// suite 20261105-185306: cold L2, median of 9, 11-candidate sweep). The 120a-tuned default is
// not the fastest at any of this model's shapes on this card; the shape launchers select among
// these by token count. Names follow the knob moved from the default.
using Fp8A8Rows64Schedule    = Fp8MmaSchedule<64, 64, 128, 2, 4, 4, 1, Cache::cg, Cache::cg,
                                              Fp8MmaFragmentPipeline::PingPong,
                                              Fp8MmaRaster::TokenFast>;
using Fp8A8K256Schedule      = Fp8MmaSchedule<64, 128, 256, 2, 4, 2, 1, Cache::cg, Cache::cg,
                                              Fp8MmaFragmentPipeline::PingPong,
                                              Fp8MmaRaster::TokenFast>;
using Fp8A8Tokens128Schedule = Fp8MmaSchedule<128, 128, 128, 2, 4, 2, 1, Cache::cg, Cache::cg,
                                              Fp8MmaFragmentPipeline::PingPong,
                                              Fp8MmaRaster::TokenFast>;
using Fp8A8Warps1Schedule    = Fp8MmaSchedule<64, 128, 128, 1, 4, 2, 2, Cache::cg, Cache::cg,
                                              Fp8MmaFragmentPipeline::PingPong,
                                              Fp8MmaRaster::TokenFast>;
}
