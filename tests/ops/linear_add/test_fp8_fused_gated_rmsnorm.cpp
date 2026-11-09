#include "core/decode_graph.h"
#include "core/device.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/linear_add.h"
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

constexpr int kInputRows = 6144;
constexpr int kOutputRows = 5120;
constexpr int kHeadDim    = 128; // the gated delta net's output norm head width (48 x 128 = 6144)
constexpr int kHeads      = 48;  // kInputRows / kHeadDim
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
    if (ops::linear_add_fused_gated_rmsnorm_fp8_eligible(weight, ops::LinearPolicy::AllowA8,
                                                         tokens)) {
        std::cerr << "T=" << tokens << " must stay on the unfused path\n";
        return 1;
    }
    const auto values             = make_bf16_activation(kInputRows, tokens, 871U + tokens);
    const auto gates              = make_bf16_activation(kInputRows, tokens, 2047U + tokens);
    const auto gains              = make_bf16_activation(kHeadDim, 1, 119U);
    DeviceBuffer o_buffer         = to_device(bf16_bits(values));
    DeviceBuffer z_buffer         = to_device(bf16_bits(gates));
    DeviceBuffer gain_buffer      = to_device(bf16_bits(gains));
    DeviceBuffer hidden_buffer(static_cast<std::size_t>(kInputRows) * tokens * 2);
    DeviceBuffer residual_buffer(static_cast<std::size_t>(kOutputRows) * tokens * 2);
    Tensor o(o_buffer.p, DType::BF16, {kInputRows, tokens});
    Tensor z(z_buffer.p, DType::BF16, {kInputRows, tokens});
    Tensor o3d(o_buffer.p, DType::BF16, {kHeadDim, kHeads, tokens});
    Tensor z3d(z_buffer.p, DType::BF16, {kHeadDim, kHeads, tokens});
    Tensor gain(gain_buffer.p, DType::BF16, {kHeadDim});
    Tensor hidden(hidden_buffer.p, DType::BF16, {kInputRows, tokens});
    Tensor hidden3d(hidden_buffer.p, DType::BF16, {kHeadDim, kHeads, tokens});
    Tensor residual(residual_buffer.p, DType::BF16, {kOutputRows, tokens});
    DeviceArena workspace(1);
    try {
        ops::linear_add_fused_gated_rmsnorm_fp8(o, gain, z, kEps, weight, residual,
                                                ops::LinearPolicy::AllowA8, workspace, nullptr);
    } catch (const std::invalid_argument&) {
        ops::gated_rmsnorm(o3d, gain, z3d, ops::GateActivation::Silu, kEps, hidden3d, nullptr);
        ops::linear_add(hidden, weight, residual, ops::LinearPolicy::AllowA8, workspace, nullptr);
        cuda_synchronize();
        for (const double value :
             from_device_bf16(residual_buffer.p, residual_buffer.bytes / 2)) {
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
    const auto values            = make_bf16_activation(kInputRows, tokens, 871U + tokens);
    const auto gates             = make_bf16_activation(kInputRows, tokens, 2047U + tokens);
    const auto gains             = make_bf16_activation(kHeadDim, 1, 119U);
    DeviceBuffer o_buffer        = to_device(bf16_bits(values));
    DeviceBuffer z_buffer        = to_device(bf16_bits(gates));
    DeviceBuffer gain_buffer     = to_device(bf16_bits(gains));
    DeviceBuffer hidden_buffer(static_cast<std::size_t>(kInputRows) * tokens * 2);
    Tensor o(o_buffer.p, DType::BF16, {kInputRows, tokens});
    Tensor z(z_buffer.p, DType::BF16, {kInputRows, tokens});
    Tensor o3d(o_buffer.p, DType::BF16, {kHeadDim, kHeads, tokens});
    Tensor z3d(z_buffer.p, DType::BF16, {kHeadDim, kHeads, tokens});
    Tensor gain(gain_buffer.p, DType::BF16, {kHeadDim});
    Tensor hidden(hidden_buffer.p, DType::BF16, {kInputRows, tokens});
    Tensor hidden3d(hidden_buffer.p, DType::BF16, {kHeadDim, kHeads, tokens});
    // The GDN output norm is per-head: each 192 head normalizes itself with the 192 gain.
    ops::gated_rmsnorm(o3d, gain, z3d, ops::GateActivation::Silu, kEps, hidden3d, nullptr);

    int failures = 0;
    // Quantiser boundary: the fused launch writes the same codes and per-token scale the
    // separate quantize launch would have read out of the stored gated row.
    {
        DeviceArena reference_workspace(ops::detail::fp8_a8_workspace_capacity_bytes(tokens,
                                                                               kInputRows));
        DeviceArena fused_workspace(ops::detail::fp8_a8_workspace_capacity_bytes(tokens, kInputRows));
        auto reference_scope = reference_workspace.scope();
        auto fused_scope     = fused_workspace.scope();
        const auto reference_planes =
            ops::detail::allocate_fp8_a8_workspace(reference_workspace, tokens, kInputRows);
        const auto fused_planes =
            ops::detail::allocate_fp8_a8_workspace(fused_workspace, tokens, kInputRows);
        ops::detail::launch_fp8_a8_quantize(hidden, weight, reference_planes, nullptr);
        ops::detail::launch_fp8_fused_gated_rmsnorm_quantize(o, gain, z, kEps, kInputRows,
                                                        fused_planes, nullptr);
        cuda_synchronize();
        const std::string suffix = " T=" + std::to_string(tokens);
        failures += compare_bytes("codes" + suffix, reference_planes.codes, fused_planes.codes,
                                  static_cast<std::size_t>(tokens) * kInputRows);
        failures += compare_bytes("scales" + suffix, reference_planes.scales,
                                  fused_planes.scales, static_cast<std::size_t>(tokens));
    }

    // Composition: the public fused entry against the standalone gated norm + A8 route. The
    // residual-add epilogue reads its own destination, so every destination is seeded with the
    // same deterministic state: the eager comparison then holds exactly (no dependence on fresh
    // buffer contents), and the graph replay below must reproduce one application of the op to
    // that state rather than accumulate on the eager fused output.
    const std::size_t capacity = ops::linear_add_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kOutputRows, kInputRows, ops::LinearPolicy::AllowA8, tokens,
        tokens);
    DeviceArena reference_workspace(capacity);
    DeviceArena fused_workspace(capacity);
    const auto residual_init = make_bf16_activation(kOutputRows, tokens, 3071U + tokens);
    const auto residual_bits = bf16_bits(residual_init);
    const std::size_t residual_bytes = static_cast<std::size_t>(kOutputRows) * tokens * 2;
    DeviceBuffer reference_residual(residual_bytes);
    DeviceBuffer fused_residual(residual_bytes);
    DeviceBuffer graph_residual(residual_bytes);
    cuda_check(cudaMemcpy(reference_residual.p, residual_bits.data(), residual_bytes,
                          cudaMemcpyHostToDevice),
               "seed reference residual");
    cuda_check(cudaMemcpy(fused_residual.p, residual_bits.data(), residual_bytes,
                          cudaMemcpyHostToDevice),
               "seed fused residual");
    cuda_check(cudaMemcpy(graph_residual.p, residual_bits.data(), residual_bytes,
                          cudaMemcpyHostToDevice),
               "seed graph residual");
    Tensor reference_out(reference_residual.p, DType::BF16, {kOutputRows, tokens});
    Tensor fused_out(fused_residual.p, DType::BF16, {kOutputRows, tokens});
    Tensor graph_out(graph_residual.p, DType::BF16, {kOutputRows, tokens});
    {
        auto reference_scope = reference_workspace.scope();
        auto fused_scope     = fused_workspace.scope();
        ops::linear_add(hidden, weight, reference_out, ops::LinearPolicy::AllowA8,
                        reference_workspace, nullptr);
        ops::linear_add_fused_gated_rmsnorm_fp8(o, gain, z, kEps, weight, fused_out,
                                                ops::LinearPolicy::AllowA8, fused_workspace,
                                                nullptr);
    }
    cuda_synchronize();
    const std::string suffix = " T=" + std::to_string(tokens);
    failures += compare_bytes("residual" + suffix,
                              static_cast<const std::uint8_t*>(reference_residual.p),
                              static_cast<const std::uint8_t*>(fused_residual.p),
                              reference_residual.bytes);
    if (reference_workspace.used() != 0 || fused_workspace.used() != 0) {
        std::cerr << "linear_add workspace scope was not released\n";
        ++failures;
    }
    if ((tokens == 17 || tokens == 2048) && failures == 0) {
        DeviceContext device;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(device.stream, [&] {
            ops::linear_add_fused_gated_rmsnorm_fp8(o, gain, z, kEps, weight, graph_out,
                                                    ops::LinearPolicy::AllowA8, fused_workspace,
                                                    device.stream);
        });
        graph.instantiate(definition);
        graph.launch(device.stream);
        cuda_synchronize(device.stream);
        failures += compare_bytes("graph residual" + suffix,
                                  static_cast<const std::uint8_t*>(reference_residual.p),
                                  static_cast<const std::uint8_t*>(graph_residual.p),
                                  reference_residual.bytes);
    }
    return failures;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) { return 77; }
        quantized_weight::PatternedWeightOptions options;
        DevicePackedWeight parent(
            quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16, kOutputRows,
                                                    kInputRows, 2471U, options));
        const Weight weight = parent.view();
        // Eligibility reads qtype/layout/shape only (never the payload), so field-override
        // copies are safe negatives here.
        Weight wrong_qtype  = weight;
        Weight wrong_layout = weight;
        Weight wrong_n      = weight;
        Weight wrong_k      = weight;
        wrong_qtype.qtype   = QType::BF16;
        wrong_layout.layout = QuantLayout::RowSplit;
        wrong_n.n           = kOutputRows - 1;
        wrong_k.k           = kInputRows - 1;
        if (ops::linear_add_fused_gated_rmsnorm_fp8_eligible(weight, ops::LinearPolicy::A16Only,
                                                             1024) ||
            ops::linear_add_fused_gated_rmsnorm_fp8_eligible(weight,
                                                             ops::LinearPolicy::AllowA8Int,
                                                             1024) ||
            ops::linear_add_fused_gated_rmsnorm_fp8_eligible(weight,
                                                             ops::LinearPolicy::AllowA8, 0) ||
            ops::linear_add_fused_gated_rmsnorm_fp8_eligible(weight,
                                                             ops::LinearPolicy::AllowA8, 16) ||
            ops::linear_add_fused_gated_rmsnorm_fp8_eligible(wrong_qtype,
                                                             ops::LinearPolicy::AllowA8, 1024) ||
            ops::linear_add_fused_gated_rmsnorm_fp8_eligible(wrong_layout,
                                                             ops::LinearPolicy::AllowA8, 1024) ||
            ops::linear_add_fused_gated_rmsnorm_fp8_eligible(wrong_n,
                                                             ops::LinearPolicy::AllowA8, 1024) ||
            ops::linear_add_fused_gated_rmsnorm_fp8_eligible(wrong_k,
                                                             ops::LinearPolicy::AllowA8, 1024) ||
            ops::linear_add_fused_gated_rmsnorm_fp8_eligible(wrong_n,
                                                             ops::LinearPolicy::AllowA8, 1) ||
            !ops::linear_add_fused_gated_rmsnorm_fp8_eligible(weight,
                                                              ops::LinearPolicy::AllowA8, 17) ||
            !ops::linear_add_fused_gated_rmsnorm_fp8_eligible(weight,
                                                              ops::LinearPolicy::AllowA8, 1024)) {
            std::cerr << "fused route eligibility mismatch\n";
            return 1;
        }
        int failures = 0;
        for (const int tokens : {1, 16}) { failures += check_unfused_fallback(weight, tokens); }
        for (const int tokens : {17, 18, 32, 33, 64, 65, 96, 128, 129, 192, 255, 256, 512, 513,
                                  1024, 2048}) {
            failures +=
                run_case_allowing_arch_skip("fused gated rmsnorm fp8 linear_add T=" +
                                                std::to_string(tokens),
                                            [&] { return run_case(weight, tokens); });
        }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "fused gated rmsnorm fp8 linear_add test: " << error.what() << '\n';
        return 1;
    }
}
