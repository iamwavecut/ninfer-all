#include "ops/common/small_t_layout.cuh"
#include "ops/common/small_t_split_store.cuh"
#include "ops/linear/q4/q4_instance_launch.cuh"
#include "ops/linear/q4/q4_ksplit_mma.cuh"
#include "ops/linear/q4/q4_shapes.h"

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
void gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_gemv<Schedule>(q4_linear_operands(x, w), result<Output>(out, w.n),
                                 LinearIdentityEpilogue{}, stream);
}

template <class Schedule, class Output>
void sliced(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_sliced_k_mma<Schedule>(q4_linear_operands(x, w), result<Output>(out, w.n),
                                         LinearIdentityEpilogue{}, stream);
}

template <class Schedule, class Output>
void mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_mma<Schedule>(q4_linear_operands(x, w), result<Output>(out, w.n),
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

// Up to eight columns over the small-T MMA with K read from the operands: one 64-k group per K
// warp, a CTA of 128 / KWarps rows. A weight whose rows are padded past K takes `Wider`.
template <int KWarps, int Stages, class Output, Q4Launch Wider>
void small_t(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.padded_shape[1] != x.ne[0]) {
        Wider(x, w, out, stream);
        return;
    }
    using Layout     = SmallTLayout<KWarps>;
    const auto store = SmallTStore<Output>::of(out, w.n, x.ne[1]);
    q4_ksplit_mma_launch<Q4LinearGeometry<Layout::kRowsPerCta, 0>, 8, 8,
                         std::remove_const_t<decltype(store)>, Q4KSplitIdentityRows, true, KWarps,
                         Stages>(w.n / Layout::kRowsPerCta, stream,
                                 static_cast<const __nv_bfloat16*>(x.data),
                                 static_cast<const std::uint8_t*>(w.qdata),
                                 static_cast<const std::uint8_t*>(w.scales), nullptr, store,
                                 Q4KSplitIdentityRows{}, x.ne[1], x.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

// The small-T MMA where the rows and K fit it: eight K warps where K is whole 512-k slabs, four
// where it is whole 256-k ones; `Wider` elsewhere. Tensor-parallel shares of Qwen3.6-27B on the
// RTX 5060 Ti, one kernel, cold L2, median of 21, us (the GEMV at T1, sliced-K tiles past it, then
// eight K warps in two stages):
//
//                         T1            T4            T8
//   FFN 4352x5120 FP32   51.2 / 32.7   43.0 / 34.8   47.1 / 34.8
//   3072x5120 BF16       38.9 / 24.5   32.7 / 24.5   36.8 / 26.6
//   256x5120 BF16        10.2 /  8.2   10.2 /  8.2   12.3 /  8.2
template <class Output, Q4Launch Wider>
Q4Launch narrow(std::int32_t rows, std::int32_t k) {
    if (k % 512 == 0 && rows % SmallTLayout<8>::kRowsPerCta == 0) {
        return small_t<8, 2, Output, Wider>;
    }
    if (k % 256 == 0 && rows % SmallTLayout<4>::kRowsPerCta == 0) {
        return small_t<4, 2, Output, Wider>;
    }
    return Wider;
}

template <class Output>
Q4Launch select(std::int32_t rows, std::int32_t k, std::int32_t tokens) {
    using namespace q4_instances;
    if (tokens <= 1) return narrow<Output, gemv<GemvR1W8, Output>>(rows, k);
    if (tokens <= 4) return narrow<Output, sliced<SlicedR16T8Capacity4, Output>>(rows, k);
    if (tokens <= 8) return narrow<Output, sliced<SlicedR16T8W4S2, Output>>(rows, k);
    if (tokens <= 16) return sliced<SlicedR32T16W4S2, Output>;
    if (tokens <= 64) return sliced<SlicedR32T32W4S2, Output>;
    return mma<MmaR64T128K64Wr64Wt32S2A2B1, Output>;
}

} // namespace

Q4Launch select_q4_general_launch(std::int32_t rows, std::int32_t k, std::int32_t tokens,
                                  DType out) {
    return out == DType::FP32 ? select<LinearFp32Output>(rows, k, tokens)
                              : select<LinearBf16Output>(rows, k, tokens);
}

} // namespace ninfer::ops::detail
