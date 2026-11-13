// The GDN input projection's part form over BF16 and row-split parts: a tensor-parallel rank's q, k,
// value and z rows, each its own device object of any of those formats. Oracle: each output row is
// the FP64 product of its part's decoded row; the output is BF16. One column writes a part's rows in
// place; wider calls reach rows inside the q/k/value plane through a copy.

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/device_buffer.h"
#include "core/qtype.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
namespace qw  = test::quantized_weight;
namespace bfw = test::direct_bf16_weight;

constexpr test::ReductionCriterion kCriterion{1.0 / 256.0, 0.0, test::kBf16GrossRelativeFloor};
constexpr std::int32_t kHidden = 2048;

struct Part {
    std::vector<double> logical; // [n, kHidden]
    DeviceBuffer device;
    ops::ProjectionPart part;

    Part(QType qtype, std::int32_t n, std::uint32_t seed, std::int32_t output, std::int32_t row) {
        logical.resize(static_cast<std::size_t>(n) * kHidden);
        part.output = output;
        part.row    = row;
        if (qtype == QType::BF16) {
            const bfw::HostWeight host = bfw::make_patterned(n, kHidden, seed);
            for (std::size_t i = 0; i < host.bits.size(); ++i) {
                logical[i] = test::bf16_to_f32(host.bits[i]);
            }
            device = DeviceBuffer(host.bits.size() * sizeof(std::uint16_t));
            device.copy_from_host(host.bits.data(), device.bytes);
            part.weight = host.device_weight(device.p);
            return;
        }
        const qw::PackedWeight host =
            qw::make_patterned_weight(qtype, n, kHidden, seed,
                                      {qw::RowSplitScalePattern::Small, qw::RowSplitCodePattern::Hashed});
        for (std::int32_t r = 0; r < n; ++r) {
            for (std::int32_t column = 0; column < kHidden; ++column) {
                logical[static_cast<std::size_t>(r) * kHidden + column] =
                    qw::logical_weight_fp64(host, r, column);
            }
        }
        device = DeviceBuffer(host.payload.size());
        device.copy_from_host(host.payload.data(), device.bytes);
        part.weight = host.device_weight(device.p);
    }
};

} // namespace

int main() {
    if (test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        // A rank's 4 key heads of 128 (q and k 512 rows), 12 value heads (1536 rows, split in two
        // parts of different formats) and z (two parts, one ternary).
        std::vector<Part> parts;
        parts.reserve(6);
        parts.emplace_back(QType::Q4_G64_FP16, 512, 11U, 0, 0);
        parts.emplace_back(QType::Q5_G64_FP16, 512, 13U, 0, 512);
        parts.emplace_back(QType::BF16, 768, 17U, 0, 1024);
        parts.emplace_back(QType::Q6_G64_FP16, 768, 19U, 0, 1792);
        parts.emplace_back(QType::Q8_G32_FP16, 768, 23U, 1, 0);
        parts.emplace_back(QType::T2_G128_FP16, 768, 29U, 1, 768);
        constexpr std::int32_t kQkv = 2560;
        constexpr std::int32_t kZ   = 1536;
        ops::PartProjectionWeights weights;
        for (const Part& part : parts) { weights.parts.push_back(part.part); }

        constexpr std::int32_t kMaxT = 33;
        std::vector<std::uint16_t> x_bits(static_cast<std::size_t>(kHidden) * kMaxT);
        for (std::size_t i = 0; i < x_bits.size(); ++i) {
            const std::uint64_t mixed = qw::detail::mix64(0x6d0ULL << 32 | i);
            x_bits[i] = test::f32_to_bf16(static_cast<float>(static_cast<int>(mixed & 0xffU) - 128) *
                                          (1.0F / 256.0F));
        }
        std::vector<double> qkv_reference(static_cast<std::size_t>(kQkv) * kMaxT);
        std::vector<double> z_reference(static_cast<std::size_t>(kZ) * kMaxT);
        for (const Part& part : parts) {
            auto& reference      = part.part.output == 0 ? qkv_reference : z_reference;
            const std::int32_t n = part.part.output == 0 ? kQkv : kZ;
            for (std::int32_t t = 0; t < kMaxT; ++t) {
                for (std::int32_t r = 0; r < part.part.weight.n; ++r) {
                    double sum = 0.0;
                    for (std::int32_t column = 0; column < kHidden; ++column) {
                        sum += part.logical[static_cast<std::size_t>(r) * kHidden + column] *
                               test::bf16_to_f32(
                                   x_bits[static_cast<std::size_t>(t) * kHidden + column]);
                    }
                    reference[static_cast<std::size_t>(t) * n + part.part.row + r] = sum;
                }
            }
        }

        DeviceBuffer x_device(x_bits.size() * sizeof(std::uint16_t));
        x_device.copy_from_host(x_bits.data(), x_device.bytes);
        DeviceBuffer qkv_device(static_cast<std::size_t>(kQkv) * kMaxT * sizeof(std::uint16_t));
        DeviceBuffer z_device(static_cast<std::size_t>(kZ) * kMaxT * sizeof(std::uint16_t));
        WorkspaceArena workspace(
            std::max<std::size_t>(ops::gdn_input_proj_workspace_capacity_bytes(weights, 1, kMaxT),
                                  256));

        int failures = 0;
        for (const std::int32_t t : {1, 2, 7, 33}) {
            for (const bool replay : {false, true}) {
                const std::string label = "gdn_input_proj parts T=" + std::to_string(t) +
                                          (replay ? " graph" : "");
                Tensor x(x_device.p, DType::BF16, {kHidden, t});
                Tensor qkv(qkv_device.p, DType::BF16, {kQkv, t});
                Tensor z(z_device.p, DType::BF16, {kZ, t});
                qkv_device.fill(0xff);
                z_device.fill(0xff);
                workspace.reset();
                workspace.reset_peak();
                DeviceContext context;
                const auto launch = [&] {
                    ops::gdn_input_proj(x, weights, qkv, z, workspace, context.stream);
                };
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
                test::cuda_check(cudaStreamSynchronize(context.stream), "project parts");
                if (workspace.used() != 0 ||
                    workspace.peak_used() > ops::gdn_input_proj_workspace_capacity_bytes(weights, t, t)) {
                    std::cerr << label << ": workspace query/execution mismatch\n";
                    ++failures;
                }
                const auto compare = [&](const DeviceBuffer& device, std::int32_t n,
                                         const std::vector<double>& reference, const char* name) {
                    std::vector<std::uint16_t> bits(static_cast<std::size_t>(n) * t);
                    device.copy_to_host(bits.data(), bits.size() * sizeof(std::uint16_t));
                    std::vector<double> got(bits.size());
                    for (std::size_t i = 0; i < got.size(); ++i) { got[i] = test::bf16_to_f32(bits[i]); }
                    return test::verify_reduction(
                        label + " " + name, got,
                        std::span<const double>(reference.data(), got.size()), kCriterion);
                };
                failures += compare(qkv_device, kQkv, qkv_reference, "qkv");
                failures += compare(z_device, kZ, z_reference, "z");
            }
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_input_proj native parts\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "gdn_input_proj native parts: " << error.what() << '\n';
        return 1;
    }
}
