// The two-parent LinearSwiGLU over BF16 and row-split parents: a tensor-parallel rank's rows of gate
// and up, two device objects at a shape outside the tuned tables. Oracle: silu(G x) * (U x) in FP64
// from the independently decoded weights; the output is rounded once to BF16.

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/device_buffer.h"
#include "core/qtype.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
namespace qw  = test::quantized_weight;
namespace bfw = test::direct_bf16_weight;

// One BF16 storage step for the whole tensor and two for its worst element: both products are FP32
// and SwiGLU rounds once.
constexpr test::ReductionCriterion kCriterion{1.0 / 256.0, 0.0, test::kBf16GrossRelativeFloor};

// A host matrix with its FP64 decode and a device copy.
struct Parent {
    std::vector<std::uint8_t> payload;
    std::vector<double> logical; // [n, k]
    Weight weight{};
    DeviceBuffer device{};

    Parent(QType qtype, std::int32_t n, std::int32_t k, std::uint32_t seed) {
        logical.resize(static_cast<std::size_t>(n) * k);
        if (qtype == QType::BF16) {
            const bfw::HostWeight host = bfw::make_patterned(n, k, seed);
            payload.resize(host.bits.size() * sizeof(std::uint16_t));
            std::copy_n(reinterpret_cast<const std::uint8_t*>(host.bits.data()), payload.size(),
                        payload.begin());
            for (std::size_t i = 0; i < host.bits.size(); ++i) {
                logical[i] = test::bf16_to_f32(host.bits[i]);
            }
            device = DeviceBuffer(payload.size());
            device.copy_from_host(payload.data(), payload.size());
            weight = host.device_weight(device.p);
            return;
        }
        const qw::PackedWeight host = qw::make_patterned_weight(
            qtype, n, k, seed, {qw::RowSplitScalePattern::Small, qw::RowSplitCodePattern::Hashed});
        for (std::int32_t row = 0; row < n; ++row) {
            for (std::int32_t column = 0; column < k; ++column) {
                logical[static_cast<std::size_t>(row) * k + column] =
                    qw::logical_weight_fp64(host, row, column);
            }
        }
        payload = host.payload;
        device  = DeviceBuffer(payload.size());
        device.copy_from_host(payload.data(), payload.size());
        weight = host.device_weight(device.p);
    }
};

double silu(double value) { return value / (1.0 + std::exp(-value)); }

const char* format(QType qtype) {
    switch (qtype) {
    case QType::BF16:
        return "BF16";
    case QType::Q4_G64_FP16:
        return "Q4";
    case QType::Q5_G64_FP16:
        return "Q5";
    case QType::Q6_G64_FP16:
        return "Q6";
    case QType::Q8_G32_FP16:
        return "Q8";
    case QType::T2_G128_FP16:
        return "T2";
    default:
        return "?";
    }
}

int run_pair(QType gate_qtype, QType up_qtype, std::int32_t n, std::int32_t k,
             std::uint32_t seed) {
    const std::string label = std::string("LinearSwiGLU pair ") + format(gate_qtype) + "/" +
                              format(up_qtype) + " [" + std::to_string(n) + "," +
                              std::to_string(k) + "]";
    constexpr std::int32_t kWidths[] = {1, 2, 5, 9, 17, 33, 65, 129};
    constexpr std::int32_t kMaxT     = 129;
    const Parent gate(gate_qtype, n, k, seed);
    const Parent up(up_qtype, n, k, seed + 1);

    std::vector<std::uint16_t> x_bits(static_cast<std::size_t>(k) * kMaxT);
    for (std::size_t i = 0; i < x_bits.size(); ++i) {
        const std::uint64_t mixed = qw::detail::mix64((std::uint64_t{seed} << 32) | i);
        x_bits[i] = test::f32_to_bf16(static_cast<float>(static_cast<int>(mixed & 0xffU) - 128) *
                                      (1.0F / 512.0F));
    }
    std::vector<double> reference(static_cast<std::size_t>(n) * kMaxT);
    for (std::int32_t t = 0; t < kMaxT; ++t) {
        for (std::int32_t row = 0; row < n; ++row) {
            double g = 0.0;
            double u = 0.0;
            for (std::int32_t column = 0; column < k; ++column) {
                const double x = test::bf16_to_f32(x_bits[static_cast<std::size_t>(t) * k + column]);
                g += gate.logical[static_cast<std::size_t>(row) * k + column] * x;
                u += up.logical[static_cast<std::size_t>(row) * k + column] * x;
            }
            reference[static_cast<std::size_t>(t) * n + row] = silu(g) * u;
        }
    }

    DeviceBuffer x_device(x_bits.size() * sizeof(std::uint16_t));
    x_device.copy_from_host(x_bits.data(), x_device.bytes);
    DeviceBuffer out_device(static_cast<std::size_t>(n) * kMaxT * sizeof(std::uint16_t));
    const std::size_t capacity = ops::linear_swiglu_pair_workspace_capacity_bytes(
        gate_qtype, up_qtype, n, k, ops::LinearPolicy::A16Only, 1, kMaxT);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));

    int failures = 0;
    for (const std::int32_t t : kWidths) {
        for (const bool replay : {false, true}) {
            const std::string case_label =
                label + " T=" + std::to_string(t) + (replay ? " graph" : "");
            Tensor x(x_device.p, DType::BF16, {k, t});
            Tensor out(out_device.p, DType::BF16, {n, t});
            workspace.reset();
            workspace.reset_peak();
            DeviceContext context;
            const auto launch = [&] {
                ops::linear_swiglu(x, gate.weight, up.weight, out, ops::LinearPolicy::A16Only,
                                   workspace, context.stream);
            };
            test::cuda_check(cudaMemset(out_device.p, 0xff, out_device.bytes), "poison output");
            test::cuda_check(cudaDeviceSynchronize(), "finish setup");
            if (replay) {
                DecodeGraphDefinition definition;
                DecodeGraphExecutable graph;
                definition.capture(context.stream, launch);
                graph.instantiate(definition);
                graph.launch(context.stream);
            } else {
                launch();
            }
            test::cuda_check(cudaStreamSynchronize(context.stream), "run pair");
            const std::size_t exact = ops::linear_swiglu_pair_workspace_capacity_bytes(
                gate_qtype, up_qtype, n, k, ops::LinearPolicy::A16Only, t, t);
            if (workspace.used() != 0 || workspace.peak_used() > exact) {
                std::cerr << case_label << ": workspace query/execution mismatch\n";
                ++failures;
            }
            std::vector<std::uint16_t> got_bits(static_cast<std::size_t>(n) * t);
            out_device.copy_to_host(got_bits.data(), got_bits.size() * sizeof(std::uint16_t));
            std::vector<double> got(got_bits.size());
            for (std::size_t i = 0; i < got.size(); ++i) { got[i] = test::bf16_to_f32(got_bits[i]); }
            failures += test::verify_reduction(
                case_label, got, std::span<const double>(reference.data(), got.size()), kCriterion);
        }
    }
    std::cout << (failures == 0 ? "OK " : "FAIL ") << label << '\n';
    return failures;
}

} // namespace

int main() {
    if (test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = 0;
        for (const QType qtype : {QType::BF16, QType::Q4_G64_FP16, QType::Q5_G64_FP16,
                                  QType::Q6_G64_FP16, QType::Q8_G32_FP16, QType::T2_G128_FP16}) {
            failures += run_pair(qtype, qtype, 640, 5120, 701U);
        }
        // A recipe may store the two halves in different formats; and a small K.
        failures += run_pair(QType::Q4_G64_FP16, QType::Q8_G32_FP16, 384, 1536, 709U);
        failures += run_pair(QType::Q6_G64_FP16, QType::BF16, 384, 1536, 719U);
        // Q4's one-column kernel at a row count that leaves its last block part empty.
        failures += run_pair(QType::Q4_G64_FP16, QType::Q4_G64_FP16, 330, 3072, 727U);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU pair: " << error.what() << '\n';
        return 1;
    }
}
