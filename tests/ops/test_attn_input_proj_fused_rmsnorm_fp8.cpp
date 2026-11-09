#include "core/decode_graph.h"
#include "core/device.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/rmsnorm.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_fused_norm_quantize.h"
#include "ops/input_projection_test_common.h"
#include "ops/op_tester.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

constexpr int kHidden     = 5120;
constexpr int kParentRows = 14336;
constexpr int kQRows      = 6144;
constexpr int kKvRows     = 1024;
constexpr float kEps      = 1.0e-6F;

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
    if (ops::attn_input_proj_fused_rmsnorm_fp8_eligible(weight, ops::LinearPolicy::AllowA8,
                                                        tokens)) {
        std::cerr << "T=" << tokens << " must stay on the unfused path\n";
        return 1;
    }
    const auto values             = make_bf16_activation(kHidden, tokens, 871U + tokens);
    const auto gains              = make_bf16_activation(kHidden, 1, 119U);
    DeviceBuffer residual_buffer  = to_device(bf16_bits(values));
    DeviceBuffer gain_buffer      = to_device(bf16_bits(gains));
    DeviceBuffer hidden_buffer(static_cast<std::size_t>(kHidden) * tokens * 2);
    DeviceBuffer output_q(static_cast<std::size_t>(kQRows) * tokens * 2);
    DeviceBuffer output_g(static_cast<std::size_t>(kQRows) * tokens * 2);
    DeviceBuffer output_k(static_cast<std::size_t>(kKvRows) * tokens * 2);
    DeviceBuffer output_v(static_cast<std::size_t>(kKvRows) * tokens * 2);
    Tensor residual(residual_buffer.p, DType::BF16, {kHidden, tokens});
    Tensor gain(gain_buffer.p, DType::BF16, {kHidden});
    Tensor hidden(hidden_buffer.p, DType::BF16, {kHidden, tokens});
    Tensor q(output_q.p, DType::BF16, {kQRows, tokens});
    Tensor gate(output_g.p, DType::BF16, {kQRows, tokens});
    Tensor k(output_k.p, DType::BF16, {kKvRows, tokens});
    Tensor v(output_v.p, DType::BF16, {kKvRows, tokens});
    DeviceArena workspace(1);
    try {
        ops::attn_input_proj_fused_rmsnorm_fp8(residual, gain, kEps, weight, q, gate, k, v,
                                               ops::LinearPolicy::AllowA8, workspace, nullptr);
    } catch (const std::invalid_argument&) {
        ops::rmsnorm(residual, gain, kEps, true, hidden, nullptr);
        ops::attn_input_proj(hidden, weight, q, gate, k, v, ops::LinearPolicy::AllowA8, workspace,
                             nullptr);
        cuda_synchronize();
        for (const double value :
             from_device_bf16(output_q.p, output_q.bytes / 2)) {
            if (!std::isfinite(value)) {
                std::cerr << "T=" << tokens
                          << " unfused fallback produced non-finite output\n";
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
    const std::size_t capacity = ops::attn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kParentRows, kHidden, ops::LinearPolicy::AllowA8, tokens,
        tokens);
    DeviceArena reference_workspace(capacity);
    DeviceArena fused_workspace(capacity);
    DeviceBuffer reference_q(static_cast<std::size_t>(kQRows) * tokens * 2);
    DeviceBuffer reference_gate(static_cast<std::size_t>(kQRows) * tokens * 2);
    DeviceBuffer reference_k(static_cast<std::size_t>(kKvRows) * tokens * 2);
    DeviceBuffer reference_v(static_cast<std::size_t>(kKvRows) * tokens * 2);
    DeviceBuffer fused_q(reference_q.bytes), fused_gate(reference_gate.bytes);
    DeviceBuffer fused_k(reference_k.bytes), fused_v(reference_v.bytes);
    Tensor rq(reference_q.p, DType::BF16, {kQRows, tokens});
    Tensor rg(reference_gate.p, DType::BF16, {kQRows, tokens});
    Tensor rk(reference_k.p, DType::BF16, {kKvRows, tokens});
    Tensor rv(reference_v.p, DType::BF16, {kKvRows, tokens});
    Tensor fq(fused_q.p, DType::BF16, {kQRows, tokens});
    Tensor fg(fused_gate.p, DType::BF16, {kQRows, tokens});
    Tensor fk(fused_k.p, DType::BF16, {kKvRows, tokens});
    Tensor fv(fused_v.p, DType::BF16, {kKvRows, tokens});
    {
        auto reference_scope = reference_workspace.scope();
        auto fused_scope     = fused_workspace.scope();
        ops::attn_input_proj(hidden, weight, rq, rg, rk, rv, ops::LinearPolicy::AllowA8,
                             reference_workspace, nullptr);
        ops::attn_input_proj_fused_rmsnorm_fp8(residual, gain, kEps, weight, fq, fg, fk, fv,
                                               ops::LinearPolicy::AllowA8, fused_workspace,
                                               nullptr);
    }
    cuda_synchronize();
    const std::string suffix = " T=" + std::to_string(tokens);
    failures += compare_bytes("q" + suffix,
                              static_cast<const std::uint8_t*>(reference_q.p),
                              static_cast<const std::uint8_t*>(fused_q.p), reference_q.bytes);
    failures += compare_bytes("gate" + suffix,
                              static_cast<const std::uint8_t*>(reference_gate.p),
                              static_cast<const std::uint8_t*>(fused_gate.p),
                              reference_gate.bytes);
    failures += compare_bytes("k" + suffix,
                              static_cast<const std::uint8_t*>(reference_k.p),
                              static_cast<const std::uint8_t*>(fused_k.p), reference_k.bytes);
    failures += compare_bytes("v" + suffix,
                              static_cast<const std::uint8_t*>(reference_v.p),
                              static_cast<const std::uint8_t*>(fused_v.p), reference_v.bytes);
    if (reference_workspace.used() != 0 || fused_workspace.used() != 0) {
        std::cerr << "attention input workspace scope was not released\n";
        ++failures;
    }
    if ((tokens == 16 || tokens == 2048) && failures == 0) {
        DeviceContext device;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(device.stream, [&] {
            ops::attn_input_proj_fused_rmsnorm_fp8(residual, gain, kEps, weight, fq, fg, fk, fv,
                                                   ops::LinearPolicy::AllowA8, fused_workspace,
                                                   device.stream);
        });
        graph.instantiate(definition);
        graph.launch(device.stream);
        cuda_synchronize(device.stream);
        failures += compare_bytes("graph q" + suffix,
                                  static_cast<const std::uint8_t*>(reference_q.p),
                                  static_cast<const std::uint8_t*>(fused_q.p), reference_q.bytes);
        failures += compare_bytes("graph gate" + suffix,
                                  static_cast<const std::uint8_t*>(reference_gate.p),
                                  static_cast<const std::uint8_t*>(fused_gate.p),
                                  reference_gate.bytes);
        failures += compare_bytes("graph k" + suffix,
                                  static_cast<const std::uint8_t*>(reference_k.p),
                                  static_cast<const std::uint8_t*>(fused_k.p),
                                  reference_k.bytes);
        failures += compare_bytes("graph v" + suffix,
                                  static_cast<const std::uint8_t*>(reference_v.p),
                                  static_cast<const std::uint8_t*>(fused_v.p),
                                  reference_v.bytes);
    }
    return failures;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) { return 77; }
        quantized_weight::PatternedWeightOptions options;
        DevicePackedWeight parent(
            quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16, kParentRows,
                                                    kHidden, 2469U, options));
        const Weight weight = parent.view();
        // Eligibility reads qtype/layout/shape only (never the payload), so field-override
        // copies are safe negatives here.
        Weight wrong_qtype  = weight;
        Weight wrong_layout = weight;
        Weight wrong_n      = weight;
        Weight wrong_k      = weight;
        wrong_qtype.qtype   = QType::BF16;
        wrong_layout.layout = QuantLayout::RowSplit;
        wrong_n.n           = kParentRows - 1;
        wrong_k.k           = kHidden - 1;
        if (ops::attn_input_proj_fused_rmsnorm_fp8_eligible(weight, ops::LinearPolicy::A16Only,
                                                            1024) ||
            ops::attn_input_proj_fused_rmsnorm_fp8_eligible(weight, ops::LinearPolicy::AllowA8,
                                                            0) ||
            ops::attn_input_proj_fused_rmsnorm_fp8_eligible(weight, ops::LinearPolicy::AllowA8,
                                                            1) ||
            ops::attn_input_proj_fused_rmsnorm_fp8_eligible(weight, ops::LinearPolicy::AllowA8,
                                                            4) ||
            ops::attn_input_proj_fused_rmsnorm_fp8_eligible(wrong_qtype,
                                                            ops::LinearPolicy::AllowA8, 1024) ||
            ops::attn_input_proj_fused_rmsnorm_fp8_eligible(wrong_layout,
                                                            ops::LinearPolicy::AllowA8, 1024) ||
            ops::attn_input_proj_fused_rmsnorm_fp8_eligible(wrong_n, ops::LinearPolicy::AllowA8,
                                                            1024) ||
            ops::attn_input_proj_fused_rmsnorm_fp8_eligible(wrong_k, ops::LinearPolicy::AllowA8,
                                                            1024) ||
            ops::attn_input_proj_fused_rmsnorm_fp8_eligible(wrong_n, ops::LinearPolicy::AllowA8,
                                                            1) ||
            !ops::attn_input_proj_fused_rmsnorm_fp8_eligible(weight,
                                                             ops::LinearPolicy::AllowA8, 5) ||
            !ops::attn_input_proj_fused_rmsnorm_fp8_eligible(weight,
                                                             ops::LinearPolicy::AllowA8, 1024)) {
            std::cerr << "fused route eligibility mismatch\n";
            return 1;
        }
        int failures = 0;
        for (const int tokens : {1, 2, 3, 4}) {
            failures += check_unfused_fallback(weight, tokens);
        }
        for (const int tokens : {5,   6,   8,   16,  32,  33,  64,  65,   96,   97,   128, 129,
                                  192, 193, 256, 512, 513, 1024, 2048}) {
            failures +=
                run_case_allowing_arch_skip("fused rmsnorm fp8 attn_input_proj T=" +
                                                std::to_string(tokens),
                                            [&] { return run_case(weight, tokens); });
        }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "fused rmsnorm fp8 attention input test: " << error.what() << '\n';
        return 1;
    }
}
