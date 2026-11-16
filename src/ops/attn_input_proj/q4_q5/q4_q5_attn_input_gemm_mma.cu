#include "core/weight.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/rowsplit_grouped_mma.cuh"
#include "ops/common/token_slices.h"

#include <cuda_fp16.h>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

inline bool debug_weights() {
    static const bool enabled = [] {
        const char* v = std::getenv("NINFER_DEBUG_WEIGHTS");
        return v != nullptr && *v != '\0';
    }();
    return enabled;
}

RowSplitGroupedMmaJob make_job(const Weight& weight, std::int32_t row_begin, std::int32_t row_count,
                               Tensor& out) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > weight.n ||
        out.ne[0] != row_count) {
        throw std::invalid_argument("Q4/Q5 attention input grouped MMA row view is invalid");
    }
    // The Weight struct's qtype field may be incorrect for multi-part weights (it reflects
    // the first part's qtype, which can be Q4 even when the slice is Q5/Q6). Use the qhigh
    // pointer as the authoritative signal: non-null qhigh means Q5 or Q6.
    QType effective_qtype = weight.qtype;
    if (effective_qtype == QType::Q4_G64_FP16 && weight.qhigh != nullptr) {
        effective_qtype = QType::Q5_G64_FP16;
    }
    if (debug_weights() && row_begin >= 6000 && row_begin < 7200) {
        fprintf(stderr, "[debug] make_job POINTERS: row_begin=%d qdata=%p qhigh=%p scales=%p n=%d k=%d\n",
                row_begin, weight.qdata, weight.qhigh, weight.scales, weight.n, weight.k);
    }
    if (effective_qtype == QType::Q6_G64_FP16 && debug_weights()) {
        fprintf(stderr, "[debug] MMA make_job Q6: row_begin=%d row_count=%d n=%d\n",
                row_begin, row_count, weight.n);
    }
    if (debug_weights() && row_begin >= 6000 && row_begin < 7200) {
        fprintf(stderr, "[debug] make_job attn row_begin=%d qtype=%d second_sub_row=%d second_sub_qtype=%d effective=%d\n",
                row_begin, (int)weight.qtype, weight.second_sub_row, (int)weight.second_sub_qtype, (int)effective_qtype);
    }
    const std::int64_t groups = weight.padded_shape[1] / 64;
    const auto* codes         = static_cast<const std::uint8_t*>(weight.qdata) +
                        static_cast<std::int64_t>(row_begin) * groups * 32;
    const auto* high   = (effective_qtype == QType::Q5_G64_FP16 && weight.qhigh)
                             ? static_cast<const std::uint8_t*>(weight.qhigh) +
                                 static_cast<std::int64_t>(row_begin) * groups * 8
                             : (effective_qtype == QType::Q6_G64_FP16 && weight.qhigh)
                                   ? static_cast<const std::uint8_t*>(weight.qhigh) +
                                         static_cast<std::int64_t>(row_begin) * groups * 16
                                   : nullptr;
    const auto* scales = static_cast<const std::uint8_t*>(weight.scales) +
                         static_cast<std::int64_t>(row_begin) * groups * 2;
    return RowSplitGroupedMmaJob{
        codes,     high,      scales, static_cast<__nv_bfloat16*>(out.data),
        row_count, out.ne[0], 0,      effective_qtype == QType::Q5_G64_FP16,
        effective_qtype == QType::Q6_G64_FP16,
    };
}

template <class Schedule, RowSplitGroupedMmaCodec Codec>
void launch_pair(bool full, const Tensor& x, RowSplitGroupedMmaJob first,
                 RowSplitGroupedMmaJob second, cudaStream_t stream) {
    if (debug_weights()) {
        auto check_ptr = [](const void* p, const char* name) {
            if (p == nullptr) return;
            cudaPointerAttributes attr;
            cudaError_t err = cudaPointerGetAttributes(&attr, p);
            if (err != cudaSuccess) {
                fprintf(stderr, "[debug] launch_pair BAD PTR: %s=%p err=%s\n",
                        name, p, cudaGetErrorString(err));
            } else {
                int current_device = -1;
                cudaGetDevice(&current_device);
                fprintf(stderr, "[debug] launch_pair PTR: %s=%p dev=%d current_dev=%d type=%d\n",
                        name, p, attr.device, current_device, (int)attr.type);
            }
        };
        check_ptr(x.data, "x");
        check_ptr(first.codes, "first.codes");
        check_ptr(first.high, "first.high");
        check_ptr(first.scales, "first.scales");
        check_ptr(first.out, "first.out");
        check_ptr(second.codes, "second.codes");
        check_ptr(second.high, "second.high");
        check_ptr(second.scales, "second.scales");
        check_ptr(second.out, "second.out");
    }

    const int tiles = div_up(first.n, Schedule::BM) + div_up(second.n, Schedule::BM);
    const int cols  = x.ne[1];
    const dim3 grid(static_cast<unsigned>(tiles),
                    static_cast<unsigned>(div_up(cols, Schedule::BN)));
    RowSplitGroupedMmaJob empty{};

    if (full) {
        rowsplit_grouped_mma_kernel<Schedule, true, Codec, 2>
            <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                     first, second, empty, empty, x.ne[0], cols,
                                                     x.ne[0]);
    } else {
        rowsplit_grouped_mma_kernel<Schedule, false, Codec, 2>
            <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                     first, second, empty, empty, x.ne[0], cols,
                                                     x.ne[0]);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_slice(const Tensor& x, const Weight& query_key_weight, const Weight& gate_value_weight,
                  Tensor& q, Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    const bool full = (x.ne[1] % Schedule::BN) == 0;
    if (query_key_weight.qtype == QType::Q6_G64_FP16) {
        launch_pair<Schedule, RowSplitGroupedMmaCodec::Q6>(
            full, x, make_job(query_key_weight, 0, 6144, q),
            make_job(query_key_weight, 6144, 1024, k), stream);
    } else if (query_key_weight.qtype == QType::Q5_G64_FP16) {
        launch_pair<Schedule, RowSplitGroupedMmaCodec::Q5>(
            full, x, make_job(query_key_weight, 0, 6144, q),
            make_job(query_key_weight, 6144, 1024, k), stream);
    } else {
        launch_pair<Schedule, RowSplitGroupedMmaCodec::Q4>(
            full, x, make_job(query_key_weight, 0, 6144, q),
            make_job(query_key_weight, 6144, 1024, k), stream);
    }
    if (gate_value_weight.qtype == QType::Q6_G64_FP16) {
        launch_pair<Schedule, RowSplitGroupedMmaCodec::Q6>(
            full, x, make_job(gate_value_weight, 0, 6144, gate),
            make_job(gate_value_weight, 6144, 1024, v), stream);
    } else if (gate_value_weight.qtype == QType::Q5_G64_FP16) {
        launch_pair<Schedule, RowSplitGroupedMmaCodec::Q5>(
            full, x, make_job(gate_value_weight, 0, 6144, gate),
            make_job(gate_value_weight, 6144, 1024, v), stream);
    } else {
        launch_pair<Schedule, RowSplitGroupedMmaCodec::Q4>(
            full, x, make_job(gate_value_weight, 0, 6144, gate),
            make_job(gate_value_weight, 6144, 1024, v), stream);
    }
}

template <class Schedule>
void launch(const Tensor& x, const Weight& query_key_weight, const Weight& gate_value_weight,
            Tensor& q, Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    if (debug_weights() && gate_value_weight.n == 7168) {
        fprintf(stderr, "[debug] launch GV: qtype=%d qdata=%p qhigh=%p scales=%p n=%d k=%d\n",
                (int)gate_value_weight.qtype, gate_value_weight.qdata, gate_value_weight.qhigh,
                gate_value_weight.scales, gate_value_weight.n, gate_value_weight.k);
    }
    constexpr std::int32_t kSliceCols = Schedule::BN;
    for_each_token_slice(x.ne[1], kSliceCols, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor q_slice       = q.slice(1, offset, count);
        Tensor gate_slice    = gate.slice(1, offset, count);
        Tensor k_slice       = k.slice(1, offset, count);
        Tensor v_slice       = v.slice(1, offset, count);
        launch_slice<Schedule>(x_slice, query_key_weight, gate_value_weight, q_slice, gate_slice,
                               k_slice, v_slice, stream);
    });
}

// The pair tile pads every column group to its full width, so a 64-wide tile does twice the
// MMA work a 32-column decode round needs. The 32-wide tile covers every decode extent; the
// 64-wide tile stays for prefill chunks.
using MmaR32C32S4 = GemmCfg<32, 32, 64, 16, 16, 2, 1, false, true, true>;
using MmaR32C64S4 = GemmCfg<32, 64, 64, 16, 16, 2, 1, false, true, true>;

template <class S, bool Full>
void mixed_slice(const Tensor& x, const Weight& w0, const Weight& w1, Tensor& q, Tensor& g,
                 Tensor& k, Tensor& v, cudaStream_t stream) {
    const dim3 grid(14336 / S::BM, (x.ne[1] + S::BN - 1) / S::BN);
    rowsplit_grouped_mma_kernel<S, Full, RowSplitGroupedMmaCodec::Mixed, 4>
        <<<grid, S::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                          make_job(w0, 0, 6144, q), make_job(w0, 6144, 1024, k),
                                          make_job(w1, 0, 6144, g), make_job(w1, 6144, 1024, v),
                                          5120, x.ne[1], 5120);
    CUDA_CHECK(cudaGetLastError());
}

template <class S>
void launch_mixed(const Tensor& x, const Weight& w0, const Weight& w1, Tensor& q, Tensor& g,
                  Tensor& k, Tensor& v, cudaStream_t stream) {
    for_each_token_slice(x.ne[1], S::BN, [&](int begin, int count) {
        const Tensor xs = x.slice(1, begin, count);
        Tensor qs = q.slice(1, begin, count), gs = g.slice(1, begin, count),
               ks = k.slice(1, begin, count), vs = v.slice(1, begin, count);
        if (count % S::BN == 0)
            mixed_slice<S, true>(xs, w0, w1, qs, gs, ks, vs, stream);
        else
            mixed_slice<S, false>(xs, w0, w1, qs, gs, ks, vs, stream);
    });
}

} // namespace

void q4_q5_attn_input_grouped_mma_r32_c32_s4_launch(const Tensor& x, const Weight& query_key_weight,
                                                    const Weight& gate_value_weight, Tensor& q,
                                                    Tensor& gate, Tensor& k, Tensor& v,
                                                    cudaStream_t stream) {
    launch<MmaR32C32S4>(x, query_key_weight, gate_value_weight, q, gate, k, v, stream);
}

void q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(const Tensor& x, const Weight& query_key_weight,
                                                    const Weight& gate_value_weight, Tensor& q,
                                                    Tensor& gate, Tensor& k, Tensor& v,
                                                    cudaStream_t stream) {
    launch<MmaR32C64S4>(x, query_key_weight, gate_value_weight, q, gate, k, v, stream);
}

void q4_q5_attn_input_mixed_r32_c32_s2_launch(const Tensor& x, const Weight& w0, const Weight& w1,
                                              Tensor& q, Tensor& g, Tensor& k, Tensor& v,
                                              cudaStream_t stream) {
    launch_mixed<GemmCfg<32, 32, 64, 16, 16, 2, 1, false, true, true>>(x, w0, w1, q, g, k, v,
                                                                       stream);
}

void q4_q5_attn_input_mixed_r32_c32_k128_s2_launch(const Tensor& x, const Weight& w0,
                                                   const Weight& w1, Tensor& q, Tensor& g,
                                                   Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_mixed<GemmCfg<32, 32, 128, 16, 16, 2, 1, false, true, true>>(x, w0, w1, q, g, k, v,
                                                                        stream);
}

void q4_q5_attn_input_mixed_r32_c64_s3_launch(const Tensor& x, const Weight& w0, const Weight& w1,
                                              Tensor& q, Tensor& g, Tensor& k, Tensor& v,
                                              cudaStream_t stream) {
    launch_mixed<GemmCfg<32, 64, 64, 16, 16, 3, 3, false, true, true>>(x, w0, w1, q, g, k, v,
                                                                       stream);
}

void q4_q5_attn_input_pair_r32_c64_s3_launch(const Tensor& x, const Weight& w0, const Weight& w1,
                                             Tensor& q, Tensor& g, Tensor& k, Tensor& v,
                                             cudaStream_t stream) {
    launch<GemmCfg<32, 64, 64, 32, 16, 3, 2, false, true, true>>(x, w0, w1, q, g, k, v, stream);
}

void q4_q5_attn_input_mixed_r64_c128_s2_launch(const Tensor& x, const Weight& w0, const Weight& w1,
                                               Tensor& q, Tensor& g, Tensor& k, Tensor& v,
                                               cudaStream_t stream) {
    launch_mixed<GemmCfg<64, 128, 64, 64, 16, 2, 2, false, true, true>>(x, w0, w1, q, g, k, v,
                                                                        stream);
}
} // namespace ninfer::ops::detail
