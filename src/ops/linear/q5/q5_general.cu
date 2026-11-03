#include "ops/common/small_t_split_store.cuh"
#include "ops/linear/q5/q5_gemv_launch.cuh"
#include "ops/linear/q5/q5_simt_launch.cuh"
#include "ops/linear/q5/q5_mma_launch.cuh"
#include "ops/linear/q5/q5_sliced_k_launch.cuh"
#include "ops/linear/q5/q5_instances.cuh"
#include "ops/linear/q5/q5_shapes.h"
#include "ops/linear/q5/q5_small_t_mma.cuh"

#include <type_traits>

namespace ninfer::ops::detail {
namespace {

// The runtime-shape routes of a product no shape table names, such as a tensor-parallel rank's
// share of a projection: the shape tables' schedules with the rows and K read from the operands,
// writing BF16 or FP32 (a rank's partial sum).
template <class Output>
Output result(Tensor& out, std::int32_t rows);

template <>
LinearBf16Output result(Tensor& out, std::int32_t rows) {
    return {static_cast<__nv_bfloat16*>(out.data), rows};
}

template <>
LinearFp32Output result(Tensor& out, std::int32_t rows) {
    return {static_cast<float*>(out.data), rows};
}

template <class Schedule, class Output>
void direct(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q5_a16_direct_simt<Schedule>(q5_linear_operands(x, w), result<Output>(out, w.n),
                                        LinearIdentityEpilogue{}, stream);
}

template <class Schedule, class Output>
void sliced(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q5_a16_sliced_k_mma<Schedule>(q5_linear_operands(x, w), result<Output>(out, w.n),
                                         LinearIdentityEpilogue{}, stream);
}

template <class Schedule, class Output>
void mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q5_a16_mma<Schedule>(q5_linear_operands(x, w), result<Output>(out, w.n),
                                LinearIdentityEpilogue{}, stream);
}

// The small-T MMA's store of the rows the weight holds.
template <class Output>
struct SmallTStore;

template <>
struct SmallTStore<LinearBf16Output> {
    static SmallTSplitStore of(Tensor& out, std::int32_t rows, int columns) {
        auto* data = static_cast<__nv_bfloat16*>(out.data);
        return {data, data, rows, rows, rows, columns};
    }
};

template <>
struct SmallTStore<LinearFp32Output> {
    static SmallTFp32Store of(Tensor& out, std::int32_t rows, int columns) {
        return {static_cast<float*>(out.data), rows, columns};
    }
};

// Up to XCols columns over the small-T MMA with K read from the operands: one 64-k group per K
// warp, a CTA of 128 / KWarps rows. A weight whose rows are padded past K takes `Wider`.
template <int KWarps, int XCols, int Stages, class Output, Q5Launch Wider>
void small_t(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.padded_shape[1] != x.ne[0]) {
        Wider(x, w, out, stream);
        return;
    }
    using Schedule   = Q5SmallTSchedule<KWarps>;
    const auto store = SmallTStore<Output>::of(out, w.n, x.ne[1]);
    q5_small_t_mma_kernel<Schedule::kRowsPerCta, 0, XCols, Stages,
                          std::remove_const_t<decltype(store)>, KWarps>
        <<<w.n / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            store, x.ne[1], x.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

// The small-T MMA where the rows and K fit it, `Wider` elsewhere: eight K warps where K is whole
// 512-k slabs (four columns in two stages up to T4, eight in one past it), four K warps in two
// stages over eight columns where K is whole 256-k ones. The direct route keeps T1 except over a
// short K (a 64-thread CTA per row has little to read), and T2-T3 where only four K warps fit.
// Tensor-parallel shares of Qwen3.6-27B on the RTX 5060 Ti, one kernel, cold L2, median of 21, us
// (direct routes to T4, the sliced-K tile at T8, then the small-T MMA):
//
//                          T1            T2            T4            T8
//   down 5120x4352 FP32   41.0          43.0 / 47.1   55.3 / 47.1   49.1 / 43.0
//   out 5120x1536 FP32    22.5 / 16.4   22.5 / 16.4   30.7 / 16.4   20.5 / 16.4
//   1536x5120 BF16        16.4 / 16.4   18.4 / 16.4   22.5 / 18.4   22.5 / 18.4
//   256x5120 BF16          8.2 /  8.2    8.2 /  8.2    8.2 /  8.2   12.3 / 12.3
template <int Tokens, class Output, Q5Launch Wider>
Q5Launch narrow(std::int32_t rows, std::int32_t k) {
    if (k % 512 == 0 && rows % Q5SmallTSchedule<8>::kRowsPerCta == 0 && (Tokens > 1 || k <= 2048)) {
        return Tokens <= 4 ? small_t<8, 4, 2, Output, Wider> : small_t<8, 8, 1, Output, Wider>;
    }
    if (k % 256 == 0 && rows % Q5SmallTSchedule<4>::kRowsPerCta == 0 && Tokens >= 4) {
        return small_t<4, 8, 2, Output, Wider>;
    }
    return Wider;
}

template <class Output>
Q5Launch select(std::int32_t rows, std::int32_t k, std::int32_t tokens) {
    using namespace q5_instances;
    if (tokens <= 1) return narrow<1, Output, direct<DirectR1T1W2U8, Output>>(rows, k);
    if (tokens <= 2) return narrow<2, Output, direct<DirectR1T2W2, Output>>(rows, k);
    if (tokens <= 3) return narrow<3, Output, direct<DirectR1T3W2, Output>>(rows, k);
    if (tokens <= 4) return narrow<4, Output, direct<DirectR1T4W2, Output>>(rows, k);
    if (tokens <= 8) return narrow<8, Output, sliced<SlicedR16T8W4S2, Output>>(rows, k);
    if (tokens <= 16) return sliced<SlicedR16T16W4S2, Output>;
    if (tokens <= 64) return sliced<SlicedR32T32W4S2, Output>;
    return mma<MmaR64T128, Output>;
}

} // namespace

Q5Launch select_q5_general_launch(std::int32_t rows, std::int32_t k, std::int32_t tokens,
                                  DType out) {
    return out == DType::FP32 ? select<LinearFp32Output>(rows, k, tokens)
                              : select<LinearBf16Output>(rows, k, tokens);
}

} // namespace ninfer::ops::detail
