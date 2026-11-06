// Route-boundary sweep for the sm_89 plain-E4M3 FP8 A8 route (schedule_sweep.cuh).
//
// M2.2 of the fp8-dev plan: the plain m16n8k32 tile runs the Fp8A8DefaultSchedule (a 120a-tuned
// constant) and the uses_a8 thresholds that upstream published; neither was ever measured on the
// 4090. This bench times the candidate plain-mma schedules at every column count the boundary sits
// in, with the production A8/A16 dispatch alongside, so the schedule and the threshold both come
// from the card.
//
// The candidates are Fp8MmaSchedule instantiations launched through launch_fp8_a8 -- quantize plus
// mma, exactly the registered shapes' A8 route with the schedule swapped. Every candidate must clear
// the kernel's static asserts (K % BK, N % BR, TILES_K >= S, warp tiling, the 128x128 bf16
// output-staging floor kSharedBytes >= 34816) and Ada's 99,328-byte dynamic shared-memory budget at
// its MinBlocksPerSm.
//
// The 5120x6144 and 5120x17408 geometries run twice in production: as these plain `linear` shapes
// (measured here, no residual) and, in the Qwen3.8-27B prefill, as the fused `linear_add` with its
// residual epilogue (the public_op column uses linear_add for exactly those two shapes, so the
// A8-vs-A16 decision column stays the production one).
//
// Domains: the plain-mma kernel masks its final token tile, so every T >= 1 is in domain for every
// candidate (no max_cols). A schedule that faults takes the process with it; --only narrows the
// candidate set for that case.

#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"

#include "ops/linear/fp8/fp8_geometry.h"
#include "ops/linear/fp8/fp8_launch.cuh"
#include "ops/linear/fp8/fp8_shapes.h"
#include "quantized_weight.cuh"
#include "schedule_sweep.cuh"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// Optional name filter (`--only sub[,sub...]`).
std::vector<std::string> g_only;

bool selected(const char* name) {
    if (g_only.empty()) { return true; }
    for (const std::string& part : g_only) {
        if (std::string(name).find(part) != std::string::npos) { return true; }
    }
    return false;
}

using ninfer::bench::SweepEntry;
using ninfer::bench::SweepOptions;
namespace detail = ninfer::ops::detail;
using Cache = ninfer::ops::Cache;
using Fp8MmaFragmentPipeline = detail::Fp8MmaFragmentPipeline;
using Fp8MmaRaster           = detail::Fp8MmaRaster;

// Template alias (forwarding the default), so the candidate set below stays readable.
template <int BlockTokens, int BlockRows, int BlockK, int WarpsTokens, int WarpsRows, int Stages,
          int MinBlocksPerSm, Cache WeightCache, Cache ActivationCache,
          Fp8MmaFragmentPipeline FragmentPipeline, Fp8MmaRaster Raster, int RasterGroupRows = 1>
using Fp8MmaSchedule =
    detail::Fp8MmaSchedule<BlockTokens, BlockRows, BlockK, WarpsTokens, WarpsRows, Stages,
                           MinBlocksPerSm, WeightCache, ActivationCache, FragmentPipeline, Raster,
                           RasterGroupRows>;

// --- the candidate set ---------------------------------------------------------------------------
//
// dft is the shipped Fp8A8DefaultSchedule; the rest move one knob at a time (pipeline, raster,
// tile shape, stages, occupancy) from it.

using Dft     = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                               Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Serial  = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                               Fp8MmaFragmentPipeline::Serial, Fp8MmaRaster::TokenFast>;
using RowFast = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                               Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::RowFast>;
using Group8  = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                               Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::Grouped, 8>;
using Br64S4  = Fp8MmaSchedule<64, 64, 128, 2, 4, 4, 1, Cache::cg, Cache::cg,
                               Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Bk64S4  = Fp8MmaSchedule<64, 128, 64, 2, 4, 4, 2, Cache::cg, Cache::cg,
                               Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using T32     = Fp8MmaSchedule<32, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                               Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using T128    = Fp8MmaSchedule<128, 128, 128, 2, 4, 2, 1, Cache::cg, Cache::cg,
                               Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using Bk256   = Fp8MmaSchedule<64, 128, 256, 2, 4, 2, 1, Cache::cg, Cache::cg,
                               Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using W44     = Fp8MmaSchedule<64, 128, 128, 4, 4, 2, 2, Cache::cg, Cache::cg,
                               Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
using W14     = Fp8MmaSchedule<64, 128, 128, 1, 4, 2, 2, Cache::cg, Cache::cg,
                               Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;

// --- the sweep -------------------------------------------------------------------------------------

struct ShapeSpec {
    const char* key;
    // Fused linear_add with the residual epilogue in production (Qwen3.8-27B down / out).
    bool fused;
    // The registered shape's own A8/A16 admission, for the routed-name column.
    const detail::Fp8LinearShape* shape;
    void (*sweep)(const ShapeSpec&, const SweepOptions&);
};

template <class Geometry, bool Fused>
void sweep_shape_t(const ShapeSpec& spec, const SweepOptions& base) {
    constexpr std::int32_t N = Geometry::kOutputRows;
    constexpr std::int32_t K = Geometry::kInputRows;
    const std::int32_t n = N, k = K;
    const std::int32_t max_tokens = *std::max_element(base.tokens.begin(), base.tokens.end());

    ninfer::bench::PackedQuantizedWeight packed = ninfer::bench::make_fp8_weight(n, k);
    // Random, not zero: constant data draws less power and raises clocks (quantized_weight.cuh).
    ninfer::DeviceBuffer input =
        ninfer::bench::make_bf16(static_cast<std::size_t>(k) * max_tokens);
    ninfer::DeviceBuffer output(static_cast<std::size_t>(n) * max_tokens * 2);
    ninfer::DeviceBuffer residual(static_cast<std::size_t>(n) * max_tokens * 2);

    const std::size_t a8_capacity = detail::fp8_a8_workspace_capacity_bytes(max_tokens, k);
    ninfer::WorkspaceArena a8_arena(std::max<std::size_t>(a8_capacity, 256));
    const std::size_t public_capacity = Fused
        ? ninfer::ops::linear_add_workspace_capacity_bytes(ninfer::QType::FP8_E4M3FN_ROW_BF16, n,
                                                           k, ninfer::ops::LinearPolicy::AllowA8,
                                                           1, max_tokens)
        : ninfer::ops::linear_workspace_capacity_bytes(ninfer::QType::FP8_E4M3FN_ROW_BF16, n, k,
                                                       ninfer::ops::LinearPolicy::AllowA8, 1,
                                                       max_tokens);
    ninfer::WorkspaceArena public_arena(std::max<std::size_t>(public_capacity, 256));

    auto make_entry = [&](const char* name, auto launch) {
        return SweepEntry{
            name,
            [&, launch](std::int32_t tokens, cudaStream_t stream) {
                a8_arena.reset();
                const detail::Fp8A8Workspace workspace =
                    detail::allocate_fp8_a8_workspace(a8_arena, tokens, k);
                ninfer::Tensor x(input.p, ninfer::DType::BF16, {k, tokens});
                ninfer::Tensor out(output.p, ninfer::DType::BF16, {n, tokens});
                launch(x, packed.weight, out, workspace, stream);
            },
            0};
    };

    std::vector<SweepEntry> entries;
    if (selected("dft")) { entries.push_back(make_entry("dft", &detail::launch_fp8_a8<Geometry, Dft>)); }
    if (selected("serial")) { entries.push_back(make_entry("serial", &detail::launch_fp8_a8<Geometry, Serial>)); }
    if (selected("rowfast")) { entries.push_back(make_entry("rowfast", &detail::launch_fp8_a8<Geometry, RowFast>)); }
    if (selected("group8")) { entries.push_back(make_entry("group8", &detail::launch_fp8_a8<Geometry, Group8>)); }
    if (selected("br64s4")) { entries.push_back(make_entry("br64s4", &detail::launch_fp8_a8<Geometry, Br64S4>)); }
    if (selected("bk64s4")) { entries.push_back(make_entry("bk64s4", &detail::launch_fp8_a8<Geometry, Bk64S4>)); }
    if (selected("t32")) { entries.push_back(make_entry("t32", &detail::launch_fp8_a8<Geometry, T32>)); }
    if (selected("t128")) { entries.push_back(make_entry("t128", &detail::launch_fp8_a8<Geometry, T128>)); }
    if (selected("bk256")) { entries.push_back(make_entry("bk256", &detail::launch_fp8_a8<Geometry, Bk256>)); }
    if (selected("w44")) { entries.push_back(make_entry("w44", &detail::launch_fp8_a8<Geometry, W44>)); }
    if (selected("w14")) { entries.push_back(make_entry("w14", &detail::launch_fp8_a8<Geometry, W14>)); }

    SweepOptions options = base;
    // title is a const char*: keep the composed string alive for the duration of the sweep.
    static std::string title;
    title = "fp8 a8 plain-mma schedules n=" + std::to_string(n) + " k=" + std::to_string(k);
    options.title = title.c_str();
    options.routed_name  = [spec](std::int32_t tokens) -> const char* {
        return spec.shape->uses_a8(tokens, tokens) ? "a8:default" : "a16";
    };
    options.public_op    = [&](std::int32_t tokens, cudaStream_t stream) {
        public_arena.reset();
        ninfer::Tensor x(input.p, ninfer::DType::BF16, {k, tokens});
        if constexpr (Fused) {
            ninfer::Tensor out(residual.p, ninfer::DType::BF16, {n, tokens});
            ninfer::ops::linear_add(x, packed.weight, out, ninfer::ops::LinearPolicy::AllowA8,
                                    public_arena, stream);
        } else {
            ninfer::Tensor out(output.p, ninfer::DType::BF16, {n, tokens});
            ninfer::ops::linear(x, packed.weight, out, ninfer::ops::LinearPolicy::AllowA8,
                                public_arena, stream);
        }
    };
    ninfer::bench::run_sweep(options, entries);
    std::printf("\n");
}

// --- per-shape instantiations ----------------------------------------------------------------------

void sweep_14336x5120(const ShapeSpec& spec, const SweepOptions& base) {
    sweep_shape_t<detail::Fp8N14336K5120, false>(spec, base);
}
void sweep_16384x5120(const ShapeSpec& spec, const SweepOptions& base) {
    sweep_shape_t<detail::Fp8N16384K5120, false>(spec, base);
}
void sweep_34816x5120(const ShapeSpec& spec, const SweepOptions& base) {
    sweep_shape_t<detail::Fp8N34816K5120, false>(spec, base);
}
void sweep_5120x6144(const ShapeSpec& spec, const SweepOptions& base) {
    sweep_shape_t<detail::Fp8N5120K6144, true>(spec, base);
}
void sweep_5120x17408(const ShapeSpec& spec, const SweepOptions& base) {
    sweep_shape_t<detail::Fp8N5120K17408, true>(spec, base);
}

const ShapeSpec kShapes[] = {
    {"14336x5120", false, &detail::kFp8N14336K5120, sweep_14336x5120},
    {"16384x5120", false, &detail::kFp8N16384K5120, sweep_16384x5120},
    {"34816x5120", false, &detail::kFp8N34816K5120, sweep_34816x5120},
    {"5120x6144", true, &detail::kFp8N5120K6144, sweep_5120x6144},
    {"5120x17408", true, &detail::kFp8N5120K17408, sweep_5120x17408},
};

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <shape> [--tokens T,...] [--repeat N] [--warmup N]"
                             " [--spread] [--only sub,...]\nshapes:\n",
                     argv[0]);
        for (const ShapeSpec& spec : kShapes) { std::fprintf(stderr, "  %s\n", spec.key); }
        return 2;
    }
    const std::string key(argv[1]);
    --argc;
    ++argv;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) != "--only") { continue; }
        const std::string value(argv[i + 1]);
        std::size_t start = 0;
        while (start <= value.size()) {
            const std::size_t comma = value.find(',', start);
            const std::string item  = value.substr(start, comma - start);
            if (!item.empty()) { g_only.push_back(item); }
            if (comma == std::string::npos) { break; }
            start = comma + 1;
        }
        for (int j = i; j + 2 <= argc; ++j) { argv[j] = argv[j + 2]; }
        argc -= 2;
        break;
    }

    ninfer::bench::SweepOptions options;
    options.tokens = {1,  2,  4,  8,  12, 16, 17, 20, 24, 32, 48, 64,  96,
                      128, 192, 256, 384, 512, 768, 1024};
    if (!ninfer::bench::parse_sweep_args(argc, argv, options)) {
        std::fprintf(stderr, "usage: %s <shape> [--tokens T,...] [--repeat N] [--warmup N]"
                             " [--spread] [--only sub,...]\n",
                     argv[0]);
        return 2;
    }
    for (const ShapeSpec& spec : kShapes) {
        if (key == spec.key) {
            spec.sweep(spec, options);
            return 0;
        }
    }
    std::fprintf(stderr, "unknown shape '%s'\n", key.c_str());
    return 2;
}
