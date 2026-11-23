// The vector kernels' products against the CPU's scalar expert rows (gguf_cpu_rows), which compute
// the GPU's decode arithmetic bit for bit and which test_gguf_expert_cpu holds against FP64 from
// independently decoded blocks. The routed experts' one-column form (moe_vector_product at chunk 1,
// a decode step's): FP32 rows, gated BF16 rows and the weighted 2^-32 fixed-point sums. The dense
// form (vector_product, vector_swiglu) at one, three and eight columns: FP32 rows and fused BF16
// middles. Every value must equal the CPU's to the bit, for every format with a CPU product. K
// covers whole lane passes (K/32 a multiple of 32) and K whose last pass leaves lanes idle, whose
// last slices the kernels spread over the lanes in one to three rounds (K 640, 704, 1152, 1280,
// 1792, 2304, 2560, 2816); the row counts end in part of a warp's rows.

#include "core/device.h"
#include "core/device_buffer.h"
#include "core/qtype.h"
#include "ops/common/fixed_point.cuh"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"
#include "ops/moe_experts/gguf_expert_cpu.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gguf = ops::gguf;

constexpr int kExperts = 8, kTopK = 4, kDistinct = 3;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void require(bool ok, const std::string& message) {
    if (!ok) { throw std::runtime_error(message); }
}

std::uint16_t half_bits(float value) {
    const __half h = __float2half(value);
    std::uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// The binary16 multipliers of a block, as test_moe_experts_gguf sets them.
std::vector<int> half_fields(QType q) {
    switch (q) {
    case QType::GGUF_Q3_K: return {108};
    case QType::GGUF_Q4_K:
    case QType::GGUF_Q5_K: return {0, 2};
    case QType::GGUF_Q6_K: return {208};
    case QType::GGUF_IQ1_M:
    case QType::GGUF_NVFP4: return {};
    default: return {0};
    }
}

// Random codes; the multipliers at sizes a quantizer writes (Q6_K's a hundred times smaller), IQ1_M's
// spread over its scale words, NVFP4's UE4M3 scale bytes 1..31.
std::vector<std::uint8_t> random_blocks(QType q, int rows, int k, std::mt19937& rng) {
    const auto block         = gguf_block_shape(q);
    const std::int64_t count = std::int64_t(rows) * (k / block.elements);
    std::vector<std::uint8_t> bytes(count * block.bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    const float least = q == QType::GGUF_Q6_K ? 2e-5f : 0.002f;
    std::uniform_real_distribution<float> scale(least, 10 * least);
    for (auto& b : bytes) { b = static_cast<std::uint8_t>(byte(rng)); }
    for (std::int64_t i = 0; i < count; ++i) {
        std::uint8_t* base = bytes.data() + i * block.bytes;
        for (const int offset : half_fields(q)) {
            const std::uint16_t bits = half_bits(scale(rng));
            std::memcpy(base + offset, &bits, sizeof(bits));
        }
        if (q == QType::GGUF_IQ1_M) {
            const std::uint16_t d = half_bits(scale(rng));
            for (int j = 0; j < 4; ++j) {
                std::uint16_t word;
                std::memcpy(&word, base + 48 + 2 * j, 2);
                word = static_cast<std::uint16_t>((word & 0x0fff) | (((d >> (4 * j)) & 0xf) << 12));
                std::memcpy(base + 48 + 2 * j, &word, 2);
            }
        }
        if (q == QType::GGUF_NVFP4) {
            for (int s = 0; s < 4; ++s) { base[s] = static_cast<std::uint8_t>(1 + byte(rng) % 31); }
        }
    }
    return bytes;
}

std::uint32_t bits(float value) { return std::bit_cast<std::uint32_t>(value); }

void run_case(const char* name, QType format, int rows, int k, std::mt19937& rng) {
    const auto block             = gguf_block_shape(format);
    const std::int64_t row_bytes = std::int64_t(k / block.elements) * block.bytes;
    const gguf::GgmlType type    = ops::detail::gguf_type(format);
    // Two projections of the same shape: gate rows and up (or plain) rows.
    std::vector<std::vector<std::uint8_t>> gate, up;
    std::vector<DeviceBuffer> banks;
    for (int i = 0; i < kDistinct; ++i) {
        for (auto* set : {&gate, &up}) {
            set->push_back(random_blocks(format, rows, k, rng));
            banks.emplace_back(set->back().size() + 256);
            check(cudaMemset(banks.back().p, 0, set->back().size() + 256), "zero a bank");
            banks.back().copy_from_host(set->back().data(), set->back().size());
        }
    }
    std::vector<const void*> table(2 * std::size_t(kExperts));
    for (int e = 0; e < kExperts; ++e) {
        table[std::size_t(e)]            = banks[2 * std::size_t(e % kDistinct)].p;
        table[std::size_t(kExperts + e)] = banks[2 * std::size_t(e % kDistinct) + 1].p;
    }
    DeviceBuffer d_table(table.size() * sizeof(void*));
    d_table.copy_from_host(table.data(), table.size() * sizeof(void*));
    const auto* tables = static_cast<const void* const*>(d_table.p);
    const gguf::MoeTable gate_table{tables, row_bytes, rows, k};
    const gguf::MoeTable up_table{tables + kExperts, row_bytes, rows, k};

    // One token routed to kTopK distinct experts (two of them sharing a bank).
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16> x(static_cast<std::size_t>(k));
    for (auto& v : x) { v = __float2bfloat16(2.0f * normal(rng)); }
    const std::vector<std::int32_t> ids{5, 1, 7, 2};
    std::uniform_real_distribution<float> unit(0.05f, 2.5f);
    std::vector<float> weights(kTopK);
    for (auto& w : weights) { w = unit(rng); }
    DeviceBuffer d_x(x.size() * 2), d_ids(ids.size() * 4), d_w(weights.size() * 4);
    d_x.copy_from_host(x.data(), x.size() * 2);
    d_ids.copy_from_host(ids.data(), ids.size() * 4);
    d_w.copy_from_host(weights.data(), weights.size() * 4);
    DeviceBuffer d_xq(gguf::vector_activation_bytes(k, 1));
    gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(d_x.p), k, 1, nullptr,
                                     d_xq.p, nullptr);
    DeviceBuffer d_routing(gguf::moe_routing_bytes(kExperts, kTopK));
    const gguf::MoeRouting routing = gguf::moe_sort_routes(static_cast<const std::int32_t*>(d_ids.p),
                                                           kTopK, kExperts, d_routing.p, nullptr);

    const std::size_t plane = std::size_t(kTopK) * rows;
    DeviceBuffer d_gate(plane * 4), d_up(plane * 2), d_fixed(std::size_t(rows) * 8);
    check(cudaMemset(d_fixed.p, 0, std::size_t(rows) * 8), "zero the sums");
    gguf::MoeOutput gate_out;
    gate_out.f32 = static_cast<float*>(d_gate.p);
    gguf::moe_vector_product(type, gate_table, routing, kTopK, kTopK, kTopK, d_xq.p, 1, gate_out, 1,
                             nullptr);
    gguf::MoeOutput gated;
    gated.bf16 = static_cast<__nv_bfloat16*>(d_up.p);
    gated.gate = static_cast<const float*>(d_gate.p);
    gguf::moe_vector_product(type, up_table, routing, kTopK, kTopK, kTopK, d_xq.p, 1, gated, 1,
                             nullptr);
    gguf::MoeOutput weighted;
    weighted.weighted = static_cast<unsigned long long*>(d_fixed.p);
    weighted.weights  = static_cast<const float*>(d_w.p);
    gguf::moe_vector_product(type, up_table, routing, kTopK, kTopK, kTopK, d_xq.p, 1, weighted, 1,
                             nullptr);
    check(cudaDeviceSynchronize(), "the products");
    std::vector<float> got_gate(plane);
    std::vector<std::uint16_t> got_gated(plane);
    std::vector<unsigned long long> got_fixed(static_cast<std::size_t>(rows));
    d_gate.copy_to_host(got_gate.data(), plane * 4);
    d_up.copy_to_host(got_gated.data(), plane * 2);
    d_fixed.copy_to_host(got_fixed.data(), std::size_t(rows) * 8);

    // The CPU's products from the GPU's own quantized activation.
    std::vector<std::uint8_t> q(gguf::vector_activation_bytes(k, 1));
    d_xq.copy_to_host(q.data(), q.size());
    ops::GgufCpuActivation in(1, k);
    const std::vector<int> column{0};
    in.load_q8_1(format, reinterpret_cast<const std::int8_t*>(q.data()),
                 reinterpret_cast<const std::uint16_t*>(q.data() +
                                                        gguf::vector_activation_scales_offset(k, 1)),
                 column);
    std::vector<unsigned long long> want_fixed(static_cast<std::size_t>(rows), 0);
    std::size_t wrong_gate = 0, wrong_gated = 0, wrong_fixed = 0;
    for (int p = 0; p < kTopK; ++p) {
        const auto bank = std::size_t(ids[std::size_t(p)] % kDistinct);
        std::vector<float> g(static_cast<std::size_t>(rows)), u(static_cast<std::size_t>(rows));
        ops::gguf_cpu_rows({format, gate[bank].data(), row_bytes, rows, k}, in, 0, rows, g.data(),
                           std::size_t(rows), ops::GgufCpuBackend::Scalar);
        ops::gguf_cpu_rows({format, up[bank].data(), row_bytes, rows, k}, in, 0, rows, u.data(),
                           std::size_t(rows), ops::GgufCpuBackend::Scalar);
        std::vector<std::uint16_t> m(static_cast<std::size_t>(rows));
        ops::gguf_cpu_swiglu_bf16(g.data(), u.data(), std::size_t(rows), m.data(),
                                  ops::GgufCpuBackend::Scalar);
        for (int r = 0; r < rows; ++r) {
            const std::size_t at = std::size_t(p) * rows + r;
            wrong_gate += bits(got_gate[at]) != bits(g[std::size_t(r)]) ? 1 : 0;
            wrong_gated += got_gated[at] != m[std::size_t(r)] ? 1 : 0;
            want_fixed[std::size_t(r)] += static_cast<unsigned long long>(
                ops::fixed_product(u[std::size_t(r)], weights[std::size_t(p)]));
        }
    }
    for (int r = 0; r < rows; ++r) {
        wrong_fixed += got_fixed[std::size_t(r)] != want_fixed[std::size_t(r)] ? 1 : 0;
    }
    std::cout << "  " << name << " rows " << rows << " k " << k << ": "
              << wrong_gate << " FP32, " << wrong_gated << " gated BF16, " << wrong_fixed
              << " fixed-point rows differ\n";
    require(wrong_gate == 0 && wrong_gated == 0 && wrong_fixed == 0,
            "a one-column MoE vector product differs from the CPU's rows");
}

// vector_product (FP32 rows) and vector_swiglu (BF16 middles) of `columns` columns against the
// CPU's rows of the same quantized activation.
void dense_case(const char* name, QType format, int rows, int k, int columns, std::mt19937& rng) {
    const auto block             = gguf_block_shape(format);
    const std::int64_t row_bytes = std::int64_t(k / block.elements) * block.bytes;
    const gguf::GgmlType type    = ops::detail::gguf_type(format);
    // [gate; up]: 2 rows rows; the plain product takes the first `rows` of them.
    const auto weight = random_blocks(format, 2 * rows, k, rng);
    DeviceBuffer d_weight(weight.size() + 256);
    check(cudaMemset(d_weight.p, 0, weight.size() + 256), "zero the weight");
    d_weight.copy_from_host(weight.data(), weight.size());
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16> x(std::size_t(k) * columns);
    for (auto& v : x) { v = __float2bfloat16(2.0f * normal(rng)); }
    DeviceBuffer d_x(x.size() * 2), d_xq(gguf::vector_activation_bytes(k, columns));
    d_x.copy_from_host(x.data(), x.size() * 2);
    gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(d_x.p), k, columns, nullptr,
                                     d_xq.p, nullptr);
    const std::size_t plane = std::size_t(rows) * columns;
    DeviceBuffer d_y(plane * 4), d_m(plane * 2);
    gguf::VectorOutput out;
    out.f32           = static_cast<float*>(d_y.p);
    out.column_stride = rows;
    gguf::vector_product(type, d_weight.p, row_bytes, rows, k, d_xq.p, columns, out, nullptr);
    gguf::vector_swiglu(type, d_weight.p, row_bytes, rows, k, d_xq.p, columns,
                        static_cast<__nv_bfloat16*>(d_m.p), rows, nullptr);
    check(cudaDeviceSynchronize(), "the dense products");
    std::vector<float> got_y(plane);
    std::vector<std::uint16_t> got_m(plane);
    d_y.copy_to_host(got_y.data(), plane * 4);
    d_m.copy_to_host(got_m.data(), plane * 2);

    std::vector<std::uint8_t> q(gguf::vector_activation_bytes(k, columns));
    d_xq.copy_to_host(q.data(), q.size());
    ops::GgufCpuActivation in(columns, k);
    std::vector<int> order(static_cast<std::size_t>(columns));
    for (int c = 0; c < columns; ++c) { order[std::size_t(c)] = c; }
    in.load_q8_1(format, reinterpret_cast<const std::int8_t*>(q.data()),
                 reinterpret_cast<const std::uint16_t*>(
                     q.data() + gguf::vector_activation_scales_offset(k, columns)),
                 order);
    std::vector<float> g(plane), u(plane);
    ops::gguf_cpu_rows({format, weight.data(), row_bytes, rows, k}, in, 0, rows, g.data(),
                       std::size_t(rows), ops::GgufCpuBackend::Scalar);
    ops::gguf_cpu_rows({format, weight.data() + std::size_t(rows) * row_bytes, row_bytes, rows, k},
                       in, 0, rows, u.data(), std::size_t(rows), ops::GgufCpuBackend::Scalar);
    std::vector<std::uint16_t> m(plane);
    ops::gguf_cpu_swiglu_bf16(g.data(), u.data(), plane, m.data(), ops::GgufCpuBackend::Scalar);
    std::size_t wrong_y = 0, wrong_m = 0;
    for (std::size_t i = 0; i < plane; ++i) {
        wrong_y += bits(got_y[i]) != bits(g[i]) ? 1 : 0;
        wrong_m += got_m[i] != m[i] ? 1 : 0;
    }
    std::cout << "  dense " << name << " rows " << rows << " k " << k << " columns " << columns
              << ": " << wrong_y << " FP32, " << wrong_m << " fused BF16 values differ\n";
    require(wrong_y == 0 && wrong_m == 0, "a dense vector product differs from the CPU's rows");
}

} // namespace

int main() {
    try {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            std::cout << "gguf_vector_rows: no CUDA device, skipped\n";
            return 0;
        }
        std::mt19937 rng(20261123);
        const struct {
            const char* name;
            QType format;
        } formats[] = {{"Q2_0", QType::GGUF_Q2_0},       {"Q4_0", QType::GGUF_Q4_0},
                       {"Q5_0", QType::GGUF_Q5_0},       {"Q8_0", QType::GGUF_Q8_0},
                       {"Q3_K", QType::GGUF_Q3_K},       {"Q4_K", QType::GGUF_Q4_K},
                       {"Q5_K", QType::GGUF_Q5_K},       {"Q6_K", QType::GGUF_Q6_K},
                       {"IQ4_NL", QType::GGUF_IQ4_NL},   {"IQ1_S", QType::GGUF_IQ1_S},
                       {"IQ1_M", QType::GGUF_IQ1_M},     {"IQ2_XXS", QType::GGUF_IQ2_XXS},
                       {"IQ2_S", QType::GGUF_IQ2_S},     {"IQ3_XXS", QType::GGUF_IQ3_XXS},
                       {"IQ3_S", QType::GGUF_IQ3_S},     {"IQ4_XS", QType::GGUF_IQ4_XS},
                       {"NVFP4", QType::GGUF_NVFP4}};
        const int ks[] = {640, 704, 1024, 1152, 1280, 1792, 2048, 2304, 2560, 2816};
        for (const auto& [name, format] : formats) {
            require(ops::gguf_cpu_supported(format), "a format has no CPU product");
            for (const int k : ks) {
                if (k % gguf_block_shape(format).elements != 0) { continue; }
                run_case(name, format, 70, k, rng);
                for (const int columns : {1, 3, 8}) { dense_case(name, format, 70, k, columns, rng); }
            }
        }
        std::cout << "gguf_vector_rows: OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gguf_vector_rows: " << error.what() << '\n';
        return 1;
    }
}
