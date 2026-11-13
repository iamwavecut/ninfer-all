// Small-T schedules of a tensor-parallel rank's share of a Q4/Q5 projection.
//
// A share's shape is no shape table's, so `linear` takes the runtime-shape routes of q4_general.cu
// and q5_general.cu, writing BF16 (a column share) or FP32 (a row share's partial sum). This times
// those routes against neighbouring runtime-K schedules at a share's exact rows and K, cold, so a
// decode width's route is chosen by measurement on the card that runs it.
//
//   tp_share_linear_bench q4|q5 <rows> <k> bf16|fp32 [--tokens 1,2,4] [--repeat N] [--spread]

#include "ninfer/ops/linear.h"

#include "core/device_buffer.h"
#include "core/qtype.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/common/small_t_split_store.cuh"
#include "ops/linear/q4/q4_a16_simt_launch.cuh"
#include "ops/linear/q4/q4_ksplit_mma.cuh"
#include "ops/linear/q4/q4_operands.h"
#include "ops/linear/q4/q4_sliced_k_launch.cuh"
#include "ops/linear/q5/q5_operands.h"
#include "ops/linear/q5/q5_simt_launch.cuh"
#include "ops/linear/q5/q5_small_t_mma.cuh"
#include "ops/linear/q5/q5_sliced_k_launch.cuh"
#include "quantized_weight.cuh"
#include "schedule_sweep.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
using ninfer::Weight;
using ninfer::ops::Cache;
namespace detail = ninfer::ops::detail;

template <class Output>
Output result(Tensor& out, std::int32_t rows);

template <>
detail::LinearBf16Output result(Tensor& out, std::int32_t rows) {
    return {static_cast<__nv_bfloat16*>(out.data), rows};
}

template <>
detail::LinearFp32Output result(Tensor& out, std::int32_t rows) {
    return {static_cast<float*>(out.data), rows};
}

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <class Output>
struct SmallTStore;
template <>
struct SmallTStore<detail::LinearFp32Output> {
    static detail::SmallTFp32Store make(Tensor& out, std::int32_t rows, int columns) {
        return {static_cast<float*>(out.data), rows, columns};
    }
};
template <>
struct SmallTStore<detail::LinearBf16Output> {
    static detail::SmallTSplitStore make(Tensor& out, std::int32_t rows, int columns) {
        auto* p = static_cast<__nv_bfloat16*>(out.data);
        return {p, p, rows, rows, rows, columns};
    }
};

// The small-T MMA kernels with K fixed at compile time and the rows carried by the grid and store.
// K = 0: the runtime-K form, K read from the operands.
template <int K, int KWarps, int Stages, class O>
void q4_ksplit(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if ((K != 0 && x.ne[0] != K) || w.padded_shape[1] != x.ne[0] || x.ne[0] % (64 * KWarps) != 0)
        throw std::invalid_argument("K differs");
    constexpr int kRowsPerCta = detail::SmallTLayout<KWarps>::kRowsPerCta;
    if (w.n % kRowsPerCta != 0) throw std::invalid_argument("rows differ");
    const auto store = SmallTStore<O>::make(out, w.n, x.ne[1]);
    detail::q4_ksplit_mma_launch<detail::Q4LinearGeometry<kRowsPerCta, K>, 8, 8,
                                 std::remove_const_t<decltype(store)>, detail::Q4KSplitIdentityRows,
                                 true, KWarps, Stages>(
        w.n / kRowsPerCta, stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales),
        nullptr, store, detail::Q4KSplitIdentityRows{}, x.ne[1], x.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

template <int K, int KWarps, int Stages, int XCols, class O>
void q5_small_t(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if ((K != 0 && x.ne[0] != K) || w.padded_shape[1] != x.ne[0] || x.ne[0] % (64 * KWarps) != 0)
        throw std::invalid_argument("K differs");
    using Schedule = detail::Q5SmallTSchedule<KWarps>;
    if (w.n % Schedule::kRowsPerCta != 0) throw std::invalid_argument("rows differ");
    const auto store  = SmallTStore<O>::make(out, w.n, x.ne[1]);
    using Store       = std::remove_const_t<decltype(store)>;
    constexpr auto kernel =
        detail::q5_small_t_mma_kernel<Schedule::kRowsPerCta, K, XCols, Stages, Store, KWarps>;
    kernel<<<w.n / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        store, x.ne[1], x.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

struct Candidate {
    const char* name;
    Launch launch;
    std::int32_t max_cols;
};

template <class Schedule, class Output>
void q4_sliced(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    detail::launch_q4_a16_sliced_k_mma<Schedule>(detail::q4_linear_operands(x, w),
                                                 result<Output>(out, w.n),
                                                 detail::LinearIdentityEpilogue{}, stream);
}

template <class Schedule, class Output>
void q4_simt(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    detail::launch_q4_a16_simt<Schedule>(detail::q4_linear_operands(x, w), result<Output>(out, w.n),
                                         detail::LinearIdentityEpilogue{}, stream);
}

template <class Schedule, class Output>
void q5_direct(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    detail::launch_q5_a16_direct_simt<Schedule>(detail::q5_linear_operands(x, w),
                                                result<Output>(out, w.n),
                                                detail::LinearIdentityEpilogue{}, stream);
}

template <class Schedule, class Output>
void q5_sliced(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    detail::launch_q5_a16_sliced_k_mma<Schedule>(detail::q5_linear_operands(x, w),
                                                 result<Output>(out, w.n),
                                                 detail::LinearIdentityEpilogue{}, stream);
}

template <class Schedule, class Output>
void q5_simt(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    detail::launch_q5_a16_simt<Schedule>(detail::q5_linear_operands(x, w), result<Output>(out, w.n),
                                         detail::LinearIdentityEpilogue{}, stream);
}

// <rows, tokens, k warps, stages, min blocks, static K, capacity>
template <int R, int T, int W, int S, int M, int K = 0, int C = T>
using Q4Sliced = detail::Q4A16SlicedKMmaSchedule<R, T, W, S, Cache::cg, Cache::ca, M, K, C>;
template <int R, int T, int W, int S, int M, int K = 0, int C = T>
using Q5Sliced = detail::Q5A16SlicedKMmaSchedule<R, T, W, S, Cache::cg, Cache::ca, M, K, C>;
// <rows, tokens, warps per row, groups per warp tile, min blocks, unroll>
template <int R, int T, int W, int G, int M, int U = 1>
using Q5Direct = detail::Q5A16DirectSimtSchedule<R, T, W, G, M, 0, true, U>;

template <class O>
std::vector<Candidate> q4_candidates() {
    return {
        {"s16w4s2m2", q4_sliced<Q4Sliced<16, 8, 4, 2, 2, 0, 4>, O>, 4},
        {"s16w4s2m4", q4_sliced<Q4Sliced<16, 8, 4, 2, 4, 0, 4>, O>, 4},
        {"s16w4s1m6", q4_sliced<Q4Sliced<16, 8, 4, 1, 6, 0, 4>, O>, 4},
        {"s16w8s1m6", q4_sliced<Q4Sliced<16, 8, 8, 1, 6, 0, 4>, O>, 4},
        {"s16w8s2m3", q4_sliced<Q4Sliced<16, 8, 8, 2, 3, 0, 4>, O>, 4},
        {"s16w2s2m8", q4_sliced<Q4Sliced<16, 8, 2, 2, 8, 0, 4>, O>, 4},
        {"s32w4s2m2", q4_sliced<Q4Sliced<32, 8, 4, 2, 2, 0, 4>, O>, 4},
        {"s16w8k5120", q4_sliced<Q4Sliced<16, 8, 8, 1, 6, 5120, 4>, O>, 4},
        {"s16t8w4s2", q4_sliced<Q4Sliced<16, 8, 4, 2, 2>, O>, 8},
        {"simt_r4t4", q4_simt<detail::Q4A16SimtSchedule<4, 4, 2, 8, 2, Cache::ca, 1>, O>, 4},
        {"simt_r2t4w4", q4_simt<detail::Q4A16SimtSchedule<2, 4, 4, 8, 2, Cache::ca, 2>, O>, 4},
        {"simt_r8t4", q4_simt<detail::Q4A16SimtSchedule<8, 4, 2, 8, 2, Cache::ca, 1>, O>, 4},
        {"ks5120w8s1", q4_ksplit<5120, 8, 1, O>, 8},
        {"ks5120w8s2", q4_ksplit<5120, 8, 2, O>, 8},
        {"ks5120w4s2", q4_ksplit<5120, 4, 2, O>, 8},
        {"ksRw8s1", q4_ksplit<0, 8, 1, O>, 8},
        {"ksRw8s2", q4_ksplit<0, 8, 2, O>, 8},
        {"ksRw4s2", q4_ksplit<0, 4, 2, O>, 8},
        {"ksRw4s1", q4_ksplit<0, 4, 1, O>, 8},
    };
}

template <class O>
std::vector<Candidate> q5_candidates() {
    return {
        {"d1t1w2u8", q5_direct<Q5Direct<1, 1, 2, 8, 16, 8>, O>, 1},
        {"d1t2w2", q5_direct<Q5Direct<1, 2, 2, 8, 16>, O>, 2},
        {"d1t3w2", q5_direct<Q5Direct<1, 3, 2, 8, 16>, O>, 3},
        {"d1t4w2", q5_direct<Q5Direct<1, 4, 2, 8, 16>, O>, 4},
        {"d1t4w2u2", q5_direct<Q5Direct<1, 4, 2, 8, 16, 2>, O>, 4},
        {"d1t4w2u4", q5_direct<Q5Direct<1, 4, 2, 8, 16, 4>, O>, 4},
        {"d1t4w4g4", q5_direct<Q5Direct<1, 4, 4, 4, 10>, O>, 4},
        {"d1t4w4g4u4", q5_direct<Q5Direct<1, 4, 4, 4, 10, 4>, O>, 4},
        {"d2t4w2", q5_direct<Q5Direct<2, 4, 2, 8, 8>, O>, 4},
        {"d1t4w1g16u2", q5_direct<Q5Direct<1, 4, 1, 16, 16, 2>, O>, 4},
        {"s16w4s2m2", q5_sliced<Q5Sliced<16, 8, 4, 2, 2, 0, 4>, O>, 4},
        {"s16w4s2m4", q5_sliced<Q5Sliced<16, 8, 4, 2, 4, 0, 4>, O>, 4},
        {"s16w8s1m6", q5_sliced<Q5Sliced<16, 8, 8, 1, 6, 0, 4>, O>, 4},
        {"s16w8s2m3", q5_sliced<Q5Sliced<16, 8, 8, 2, 3, 0, 4>, O>, 4},
        {"s32w4s2m2", q5_sliced<Q5Sliced<32, 8, 4, 2, 2, 0, 4>, O>, 4},
        {"s16t8w4s2", q5_sliced<Q5Sliced<16, 8, 4, 2, 2>, O>, 8},
        {"simt_r4t4", q5_simt<detail::Q5A16SimtSchedule<4, 4, 2, 8, 2, Cache::ca, 1>, O>, 4},
        {"st4352w4x4s2", q5_small_t<4352, 4, 2, 4, O>, 4},
        {"st4352w4x8s1", q5_small_t<4352, 4, 1, 8, O>, 8},
        {"st1536w8x4s2", q5_small_t<1536, 8, 2, 4, O>, 4},
        {"st1536w8x8s1", q5_small_t<1536, 8, 1, 8, O>, 8},
        {"st5120w8x4s2", q5_small_t<5120, 8, 2, 4, O>, 4},
        {"st5120w8x8s1", q5_small_t<5120, 8, 1, 8, O>, 8},
        {"stRw8x4s2", q5_small_t<0, 8, 2, 4, O>, 4},
        {"stRw8x8s1", q5_small_t<0, 8, 1, 8, O>, 8},
        {"stRw4x4s2", q5_small_t<0, 4, 2, 4, O>, 4},
        {"stRw4x8s1", q5_small_t<0, 4, 1, 8, O>, 8},
        {"stRw4x8s2", q5_small_t<0, 4, 2, 8, O>, 8},
    };
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
                     "usage: %s q4|q5 <rows> <k> bf16|fp32 [--tokens T,...] [--repeat N]"
                     " [--warmup N] [--spread]\n",
                     argv[0]);
        return 2;
    }
    const std::string format(argv[1]);
    const std::int32_t n = std::atoi(argv[2]);
    const std::int32_t k = std::atoi(argv[3]);
    const bool fp32      = std::string(argv[4]) == "fp32";
    argc -= 4;
    argv += 4;
    ninfer::bench::SweepOptions options;
    options.tokens = {1, 2, 3, 4, 8};
    if (!ninfer::bench::parse_sweep_args(argc, argv, options) && argc > 1) {
        std::fprintf(stderr, "bad options\n");
        return 2;
    }
    const QType qtype = format == "q4" ? QType::Q4_G64_FP16 : QType::Q5_G64_FP16;
    const std::vector<Candidate> candidates =
        format == "q4" ? (fp32 ? q4_candidates<detail::LinearFp32Output>()
                               : q4_candidates<detail::LinearBf16Output>())
                       : (fp32 ? q5_candidates<detail::LinearFp32Output>()
                               : q5_candidates<detail::LinearBf16Output>());
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    ninfer::bench::PackedQuantizedWeight packed =
        ninfer::bench::make_row_split_weight(qtype, n, k, k);
    ninfer::DeviceBuffer input(static_cast<std::size_t>(k) * max_tokens * 2);
    ninfer::DeviceBuffer output(static_cast<std::size_t>(n) * max_tokens * 4);
    const DType out_dtype = fp32 ? DType::FP32 : DType::BF16;

    std::vector<ninfer::bench::SweepEntry> schedules;
    for (const Candidate& candidate : candidates) {
        const Launch launch = candidate.launch;
        schedules.push_back({candidate.name,
                             [&, launch](std::int32_t tokens, cudaStream_t stream) {
                                 Tensor x(input.p, DType::BF16, {k, tokens});
                                 Tensor out(output.p, out_dtype, {n, tokens});
                                 launch(x, packed.weight, out, stream);
                             },
                             candidate.max_cols});
    }
    const std::string title = format + " share n=" + std::to_string(n) + " k=" +
                              std::to_string(k) + (fp32 ? " fp32" : " bf16") + " weight_MB=" +
                              std::to_string(packed.weight.payload_bytes / 1e6);
    options.title       = title.c_str();
    options.routed_name = [](std::int32_t) { return "linear"; };
    options.public_op   = [&](std::int32_t tokens, cudaStream_t stream) {
        Tensor x(input.p, DType::BF16, {k, tokens});
        Tensor out(output.p, out_dtype, {n, tokens});
        ninfer::ops::linear(x, packed.weight, out, stream);
    };
    ninfer::bench::run_sweep(options, schedules);
    return 0;
}
