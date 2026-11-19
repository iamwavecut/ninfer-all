#include "ninfer/ops/rmsnorm.h"
#include "ops/norm_test_common.h"
#include "core/device.h"
#include "core/decode_graph.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::norm;

namespace {

// BF16 RNE alone can incur almost 2^-8 relative error; the gross bound must
// contain that storage error as well as FP32 reduction/rsqrt error. It used to
// be 3.95e-3 -- 1.01 of that storage step, so it contained the first term and
// almost none of the second, and 168 of 734 cases sat above 0.9 of it. It now
// takes the shared floor; see kBf16GrossRelativeFloor in op_check.h.
constexpr ReductionCriterion rmsnorm_bf16_criterion() {
    return {/*relative_l2*/ 1.85e-3, /*gross_absolute*/ 1.0e-5,
            /*gross_relative_to_max_reference*/ kBf16GrossRelativeFloor};
}

// FP32 reduction/rsqrt and products, without a BF16 output storage boundary.
constexpr ReductionCriterion rmsnorm_fp32_criterion() { return {2.e-6, 1.e-7, 4.e-6}; }

std::vector<double> rmsnorm_oracle(const std::vector<float>& input,
                                   const std::vector<float>& weight, const Shape& shape,
                                   bool unit_offset) {
    std::vector<double> output(input.size());
    const auto row_count = static_cast<std::int64_t>(shape.rows) * shape.tokens;
    for (std::int64_t row = 0; row < row_count; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * shape.d;
        double sum_squares     = 0.0;
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double value = input[base + column];
            sum_squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sum_squares / static_cast<double>(shape.d) + kEps);
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double gain     = static_cast<double>(weight[column]) + (unit_offset ? 1.0 : 0.0);
            output[base + column] = static_cast<double>(input[base + column]) * inverse * gain;
        }
    }
    return output;
}

int run_fp32_case(const Shape& shape, bool unit_offset, bool fp32_weight) {
    std::vector<float> input(shape.elements()), weight(shape.d);
    fill_uniform(input, 1381U, -4.f, 4.f);
    fill_uniform(weight, 1382U, -1.5f, 1.5f);
    if (!fp32_weight) round_to_bf16(weight);
    weight[0] = unit_offset ? -1.f : 0.f;
    std::vector<std::uint16_t> weight_bits(weight.size());
    for (std::size_t i = 0; i < weight.size(); ++i) weight_bits[i] = f32_to_bf16(weight[i]);
    GuardedDeviceBuffer dx(input.size() * sizeof(float)), dw(weight.size() * (fp32_weight ? 4 : 2)),
        dy(dx.bytes());
    dx.copy_from_host(input.data(), dx.bytes());
    dw.copy_from_host(fp32_weight ? static_cast<const void*>(weight.data()) : weight_bits.data(),
                      dw.bytes());
    const Tensor x(dx.data(), DType::FP32, {shape.d, shape.rows, shape.tokens});
    const Tensor w(dw.data(), fp32_weight ? DType::FP32 : DType::BF16, {shape.d});
    Tensor y(dy.data(), DType::FP32, {shape.d, shape.rows, shape.tokens});
    DeviceContext device;
    DecodeGraphDefinition definition;
    definition.capture(device.stream,
                       [&] { ops::rmsnorm(x, w, kEps, unit_offset, y, device.stream); });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    graph.launch(device.stream);
    cuda_synchronize(device.stream);
    const auto expected = rmsnorm_oracle(input, weight, shape, unit_offset);
    const auto actual   = from_device<float>(dy.data(), input.size());
    int failures =
        verify_reduction("rmsnorm FP32", std::vector<double>(actual.begin(), actual.end()),
                         expected, rmsnorm_fp32_criterion());
    failures +=
        verify_exact("rmsnorm FP32 input", from_device<float>(dx.data(), input.size()), input);
    if (fp32_weight)
        failures += verify_exact("rmsnorm FP32 weight",
                                 from_device<float>(dw.data(), weight.size()), weight);
    else
        failures += verify_exact("rmsnorm BF16 weight",
                                 from_device<std::uint16_t>(dw.data(), weight.size()), weight_bits);
    return failures + dx.verify_guards("rmsnorm FP32 input") +
           dw.verify_guards("rmsnorm FP32 weight") + dy.verify_guards("rmsnorm FP32 output");
}

int run_case(const char* label, const Shape& shape, bool unit_offset, std::uint32_t seed,
             float input_scale = 4.0F, bool bf16x2_unaligned = false, bool replay = false,
             int width = 0, bool fp32_weight = false) {
    const std::size_t count = shape.elements();
    std::vector<float> input(count), weight(shape.d);
    fill_uniform(input, seed, -input_scale, input_scale);
    fill_uniform(weight, seed + 1U, -1.5F, 1.5F);
    round_to_bf16(input);
    if (!fp32_weight) round_to_bf16(weight);
    weight[0] =
        unit_offset ? -1.0F : 0.0F; // Exact zero gain, without an intermediate BF16 gain cast.

    DeviceInput device_input  = make_input(input, bf16x2_unaligned);
    DeviceInput device_weight = make_input(weight, bf16x2_unaligned);
    DeviceBuffer direct_weight;
    if (fp32_weight) direct_weight = to_device(weight);
    const std::size_t leading = bf16x2_unaligned ? sizeof(std::uint16_t) : 0;
    GuardedDeviceBuffer output(leading + count * sizeof(std::uint16_t));
    output.fill(0xff);
    void* output_data = static_cast<std::uint8_t*>(output.data()) + leading;

    Tensor input_tensor = tensor_for(device_input.data, shape);
    Tensor weight_tensor(fp32_weight ? direct_weight.p : device_weight.data,
                         fp32_weight ? DType::FP32 : DType::BF16, {shape.d});
    Tensor output_tensor = tensor_for(output_data, shape);
    if (width) {
        input_tensor = Tensor(device_input.data, DType::BF16,
                              {shape.d, shape.rows, width, shape.tokens / width});
        output_tensor =
            Tensor(output_data, DType::BF16, {shape.d, shape.rows, width, shape.tokens / width});
    }
    DeviceContext device;
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    const auto launch = [&] {
        ops::rmsnorm(input_tensor, weight_tensor, kEps, unit_offset, output_tensor, device.stream);
    };
    int failures = 0;
    for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
        if (phase) {
            for (auto& value : input) value = -value;
            for (auto& value : weight) value = -value;
            const std::size_t offset = bf16x2_unaligned ? 1 : 0;
            for (std::size_t i = 0; i < input.size(); ++i)
                device_input.expected[i + offset] = f32_to_bf16(input[i]);
            for (std::size_t i = 0; i < weight.size(); ++i)
                device_weight.expected[i + offset] = f32_to_bf16(weight[i]);
            CUDA_CHECK(cudaMemcpy(device_input.storage.p, device_input.expected.data(),
                                  device_input.expected.size() * 2, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(device_weight.storage.p, device_weight.expected.data(),
                                  device_weight.expected.size() * 2, cudaMemcpyHostToDevice));
            if (fp32_weight)
                CUDA_CHECK(cudaMemcpy(direct_weight.p, weight.data(), weight.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));
        }
        output.fill(0xff);
        cuda_synchronize();
        if (replay && phase == 0) {
            definition.capture(device.stream, launch);
            graph.instantiate(definition);
        }
        if (replay)
            graph.launch(device.stream);
        else
            launch();
        cuda_synchronize(device.stream);
        const auto reference = rmsnorm_oracle(input, weight, shape, unit_offset);
        failures += verify_reduction(label, from_device_bf16(output_data, count), reference,
                                     rmsnorm_bf16_criterion());
        failures += verify_output_storage(std::string(label) + " output", output, bf16x2_unaligned);
        failures += verify_preserved(std::string(label) + " preserves input", device_input);
        failures += verify_preserved(std::string(label) + " preserves weight", device_weight);
        if (fp32_weight)
            failures += verify_exact((std::string(label) + " FP32 weight").c_str(),
                                     from_device<float>(direct_weight, weight.size()), weight);
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    for (int tokens : {1, 8, 9, 128, 1024})
        for (bool offset : {false, true})
            for (bool fp32_weight : {false, true})
                failures += run_fp32_case({256, tokens == 1 ? 1 : 2, tokens}, offset, fp32_weight);
    failures += run_case("rmsnorm offset [5120,1]", {5120, 1}, true, 1101U);
    failures += run_case("rmsnorm offset [5120,128]", {5120, 128}, true, 1102U);
    failures += run_case("rmsnorm offset [5120,1024]", {5120, 1024}, true, 1107U);
    failures += run_case("rmsnorm offset [2048,7]", {2048, 7}, true, 1103U);
    failures += run_case("rmsnorm offset [256,24,7]", {256, 24, 7}, true, 1104U);
    failures += run_case("rmsnorm offset [256,2,1]", {256, 2}, true, 1105U);
    failures += run_case("rmsnorm offset [256,4,48]", {256, 4, 48}, true, 1106U);
    failures += run_case("rmsnorm plain [2048,1]", {2048, 1}, false, 1201U);
    failures += run_case("rmsnorm plain [2048,128]", {2048, 128}, false, 1202U);
    failures += run_case("rmsnorm plain [128,32,7]", {128, 32, 7}, false, 1203U);
    failures += run_case("rmsnorm plain [128,8,128]", {128, 8, 128}, false, 1204U);
    failures += run_case("rmsnorm offset unaligned [128,32]", {128, 32}, true, 1301U, 4.0F, true);
    failures += run_case("rmsnorm plain unaligned [128,8]", {128, 8}, false, 1302U, 4.0F, true);
    failures += run_case("rmsnorm plain near-zero [128,32]", {128, 32}, false, 1303U, 1.0e-5F);
    for (bool offset : {false, true}) {
        failures += run_case("rmsnorm FP32 gamma hidden", {5120, 1, 9}, offset, 1310U, 4.0F, false,
                             true, 0, true);
        failures += run_case("rmsnorm FP32 gamma QK", {256, 4, 17}, offset, 1311U, 4.0F, true, true,
                             0, true);
    }
    for (int t = 1; t <= 128; ++t) {
        const bool graph = t == 1 || t == 2 || t == 7 || t == 8 || t == 15 || t == 16 || t == 31 ||
                           t == 32 || t == 63 || t == 64 || t == 65 || t == 96 || t == 127 ||
                           t == 128;
        for (bool offset : {false, true}) {
            const auto label =
                "rmsnorm hidden5120 T=" + std::to_string(t) + (offset ? " offset" : " plain");
            failures += run_case(label.c_str(), {5120, 1, t}, offset, 1400U + t, 4.f, false, graph);
        }
        for (int heads : {4, 24}) {
            const auto label =
                "rmsnorm QK256 H=" + std::to_string(heads) + " T=" + std::to_string(t);
            failures +=
                run_case(label.c_str(), {256, heads, t}, true, 1600U + t, 4.f, false, graph);
        }
    }
    for (int t : {129, 256, 1024, 2048})
        for (bool offset : {false, true})
            failures += run_case("rmsnorm hidden prefill", {5120, 1, t}, offset, 1700U + t, 4.f,
                                 false, true);
    for (int t : {129, 256, 1024, 2048})
        for (int heads : {4, 24})
            failures +=
                run_case("rmsnorm QK prefill", {256, heads, t}, true, 1750U + t, 4.f, false, true);
    for (int width : {2, 3, 8, 9, 15, 16})
        for (int batch : {1, 8}) {
            for (bool offset : {false, true})
                failures += run_case("rmsnorm hidden W/B", {5120, 1, width * batch}, offset,
                                     1800U + width, 4.f, false, true, width);
            for (int heads : {4, 24})
                failures += run_case("rmsnorm QK W/B", {256, heads, width * batch}, true,
                                     1900U + width, 4.f, false, true, width);
        }
    for (float scale : {0.0f, 1.e-5f, 4096.f})
        for (bool offset : {false, true}) {
            failures += run_case("rmsnorm hidden scale/unaligned", {5120, 1, 17}, offset, 2001U,
                                 scale, true, true);
            failures += run_case("rmsnorm QK scale/unaligned", {256, 4, 17}, offset, 2002U, scale,
                                 true, true);
        }
    std::cout << (failures ? "FAIL" : "OK") << " rmsnorm\n";
    return failures ? 1 : 0;
}
