#include "core/decode_graph.h"
#include "core/device.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/rmsnorm.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_fused_norm_quantize.h"
#include "ops/input_projection_test_common.h"
#include "ops/op_tester.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

constexpr int kRows   = 34816;
constexpr int kHidden = 5120;
constexpr int kHalf   = kRows / 2;
constexpr float kEps  = 1.0e-6F;

template <class T>
int compare_bytes(const std::string& label, const T* reference, const T* fused, std::size_t count) {
    const auto expected = from_device<std::uint8_t>(reference, count * sizeof(T));
    const auto actual   = from_device<std::uint8_t>(fused, count * sizeof(T));
    if (actual == expected) { return 0; }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::cerr << label << ": byte mismatch at " << i << " (reference "
                      << unsigned(expected[i]) << ", fused " << unsigned(actual[i]) << ")\n";
            break;
        }
    }
    return 1;
}

// Below the A8 gate the route carries BF16 activations (no quantizer): the fused entry must
// refuse rather than misroute, and the unfused pair must run clean.
int check_unfused_fallback(const Weight& weight, int tokens) {
    if (ops::linear_swiglu_fused_rmsnorm_fp8_eligible(weight, ops::LinearPolicy::AllowA8, tokens)) {
        std::cerr << "T=" << tokens << " must stay on the unfused path\n";
        return 1;
    }
    const auto values             = make_bf16_activation(kHidden, tokens, 871U + tokens);
    const auto gains              = make_bf16_activation(kHidden, 1, 119U);
    DeviceBuffer residual_buffer  = to_device(bf16_bits(values));
    DeviceBuffer gain_buffer      = to_device(bf16_bits(gains));
    DeviceBuffer hidden_buffer(static_cast<std::size_t>(kHidden) * tokens * 2);
    DeviceBuffer output_buffer(static_cast<std::size_t>(kHalf) * tokens * 2);
    Tensor residual(residual_buffer.p, DType::BF16, {kHidden, tokens});
    Tensor gain(gain_buffer.p, DType::BF16, {kHidden});
    Tensor hidden(hidden_buffer.p, DType::BF16, {kHidden, tokens});
    Tensor output(output_buffer.p, DType::BF16, {kHalf, tokens});
    DeviceArena workspace(1);
    try {
        ops::linear_swiglu_fused_rmsnorm_fp8(residual, gain, kEps, weight, output,
                                             ops::LinearPolicy::AllowA8, workspace, nullptr);
    } catch (const std::invalid_argument&) {
        ops::rmsnorm(residual, gain, kEps, true, hidden, nullptr);
        ops::linear_swiglu(hidden, weight, output, ops::LinearPolicy::AllowA8, workspace, nullptr);
        cuda_synchronize();
        for (const double v : from_device_bf16(output_buffer.p, output_buffer.bytes / 2)) {
            if (!std::isfinite(v)) {
                std::cerr << "T=" << tokens << " unfused fallback produced non-finite output\n";
                return 1;
            }
        }
        return 0;
    }
    std::cerr << "T=" << tokens << " fused entry must throw\n";
    return 1;
}

int run_case(const Weight& weight, int tokens) {
    const auto values            = make_bf16_activation(kHidden, tokens, 871U + tokens);
    const auto gains             = make_bf16_activation(kHidden, 1, 119U);
    DeviceBuffer residual_buffer = to_device(bf16_bits(values));
    DeviceBuffer gain_buffer     = to_device(bf16_bits(gains));
    DeviceBuffer hidden_buffer(static_cast<std::size_t>(kHidden) * tokens * 2);
    Tensor residual(residual_buffer.p, DType::BF16, {kHidden, tokens});
    Tensor gain(gain_buffer.p, DType::BF16, {kHidden});
    Tensor hidden(hidden_buffer.p, DType::BF16, {kHidden, tokens});
    ops::rmsnorm(residual, gain, kEps, true, hidden, nullptr);

    int failures = 0;
    // Quantiser boundary: the fused launch writes the same codes and per-token scale the
    // separate quantize launch would have read out of the stored normalized row.
    {
        DeviceArena reference_workspace(ops::detail::fp8_a8_workspace_capacity_bytes(tokens, kHidden));
        DeviceArena fused_workspace(ops::detail::fp8_a8_workspace_capacity_bytes(tokens, kHidden));
        auto reference_scope = reference_workspace.scope();
        auto fused_scope     = fused_workspace.scope();
        const auto reference_planes =
            ops::detail::allocate_fp8_a8_workspace(reference_workspace, tokens, kHidden);
        const auto fused_planes =
            ops::detail::allocate_fp8_a8_workspace(fused_workspace, tokens, kHidden);
        ops::detail::launch_fp8_a8_quantize(hidden, weight, reference_planes, nullptr);
        ops::detail::launch_fp8_fused_rmsnorm_quantize(residual, gain, kEps, kHidden, fused_planes,
                                                  nullptr);
        cuda_synchronize();
        const std::string suffix = " T=" + std::to_string(tokens);
        failures += compare_bytes("codes" + suffix, reference_planes.codes, fused_planes.codes,
                                  static_cast<std::size_t>(tokens) * kHidden);
        failures += compare_bytes("scales" + suffix, reference_planes.scales, fused_planes.scales,
                                  static_cast<std::size_t>(tokens));
    }

    // Composition: the public fused entry against the standalone norm + A8 route.
    const std::size_t capacity = ops::linear_swiglu_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, ops::LinearPolicy::AllowA8, tokens, tokens);
    DeviceArena reference_workspace(capacity);
    DeviceArena fused_workspace(capacity);
    DeviceBuffer reference_output(static_cast<std::size_t>(kHalf) * tokens * 2);
    DeviceBuffer fused_output(static_cast<std::size_t>(kHalf) * tokens * 2);
    Tensor reference_out(reference_output.p, DType::BF16, {kHalf, tokens});
    Tensor fused_out(fused_output.p, DType::BF16, {kHalf, tokens});
    {
        auto reference_scope = reference_workspace.scope();
        auto fused_scope     = fused_workspace.scope();
        ops::linear_swiglu(hidden, weight, reference_out, ops::LinearPolicy::AllowA8,
                           reference_workspace, nullptr);
        ops::linear_swiglu_fused_rmsnorm_fp8(residual, gain, kEps, weight, fused_out,
                                             ops::LinearPolicy::AllowA8, fused_workspace, nullptr);
    }
    cuda_synchronize();
    const std::string suffix = " T=" + std::to_string(tokens);
    failures += compare_bytes("output" + suffix,
                              static_cast<const std::uint8_t*>(reference_output.p),
                              static_cast<const std::uint8_t*>(fused_output.p),
                              reference_output.bytes);
    if (reference_workspace.used() != 0 || fused_workspace.used() != 0) {
        std::cerr << "linear_swiglu workspace scope was not released\n";
        ++failures;
    }
    if ((tokens == 16 || tokens == 2048) && failures == 0) {
        DeviceContext device;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(device.stream, [&] {
            ops::linear_swiglu_fused_rmsnorm_fp8(residual, gain, kEps, weight, fused_out,
                                                 ops::LinearPolicy::AllowA8, fused_workspace,
                                                 device.stream);
        });
        graph.instantiate(definition);
        graph.launch(device.stream);
        cuda_synchronize(device.stream);
        failures += compare_bytes("graph output" + suffix,
                                  static_cast<const std::uint8_t*>(reference_output.p),
                                  static_cast<const std::uint8_t*>(fused_output.p),
                                  reference_output.bytes);
    }
    return failures;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) { return 77; }
        quantized_weight::PatternedWeightOptions options;
        DevicePackedWeight parent(
            quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden,
                                                    2467U, options));
        const Weight weight = parent.view();
        // Eligibility reads qtype/layout/shape only (never the payload), so field-override
        // copies are safe negatives here.
        Weight wrong_qtype  = weight;
        Weight wrong_layout = weight;
        Weight wrong_n      = weight;
        Weight wrong_k      = weight;
        wrong_qtype.qtype   = QType::BF16;
        wrong_layout.layout = QuantLayout::RowSplit;
        wrong_n.n           = kRows - 1;
        wrong_k.k           = kHidden - 1;
        if (ops::linear_swiglu_fused_rmsnorm_fp8_eligible(weight, ops::LinearPolicy::A16Only, 1024) ||
            ops::linear_swiglu_fused_rmsnorm_fp8_eligible(weight, ops::LinearPolicy::AllowA8Int,
                                                          1024) ||
            ops::linear_swiglu_fused_rmsnorm_fp8_eligible(weight, ops::LinearPolicy::AllowA8, 0) ||
            ops::linear_swiglu_fused_rmsnorm_fp8_eligible(weight, ops::LinearPolicy::AllowA8, 1) ||
            ops::linear_swiglu_fused_rmsnorm_fp8_eligible(weight, ops::LinearPolicy::AllowA8, 2) ||
            ops::linear_swiglu_fused_rmsnorm_fp8_eligible(wrong_qtype,
                                                          ops::LinearPolicy::AllowA8, 1024) ||
            ops::linear_swiglu_fused_rmsnorm_fp8_eligible(wrong_layout,
                                                          ops::LinearPolicy::AllowA8, 1024) ||
            ops::linear_swiglu_fused_rmsnorm_fp8_eligible(wrong_n, ops::LinearPolicy::AllowA8,
                                                          1024) ||
            ops::linear_swiglu_fused_rmsnorm_fp8_eligible(wrong_k, ops::LinearPolicy::AllowA8,
                                                          1024) ||
            ops::linear_swiglu_fused_rmsnorm_fp8_eligible(wrong_n, ops::LinearPolicy::AllowA8, 1) ||
            !ops::linear_swiglu_fused_rmsnorm_fp8_eligible(weight,
                                                           ops::LinearPolicy::AllowA8, 3) ||
            !ops::linear_swiglu_fused_rmsnorm_fp8_eligible(weight,
                                                           ops::LinearPolicy::AllowA8, 1024)) {
            std::cerr << "fused route eligibility mismatch\n";
            return 1;
        }
        int failures = 0;
        for (const int tokens : {1, 2}) { failures += check_unfused_fallback(weight, tokens); }
        for (const int tokens : {3,   4,   5,   6,   8,   16,  32,  33,  64,  65,   96,   128,
                                  129, 192, 193, 255, 256, 257, 512, 513, 1024, 2048}) {
            failures +=
                run_case_allowing_arch_skip("fused rmsnorm fp8 linear_swiglu T=" +
                                                std::to_string(tokens),
                                            [&] { return run_case(weight, tokens); });
        }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "fused rmsnorm fp8 linear_swiglu test: " << error.what() << '\n';
        return 1;
    }
}
