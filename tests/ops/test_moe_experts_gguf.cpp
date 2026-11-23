// Qwen3.8-Flash-Next's MoE experts over GGUF block banks (moe_experts_gguf) against an FP64 product
// of the exactly dequantized weights. The tables point several experts at one buffer, so a few
// distinct banks cover all 512 experts while the routing still groups by expert id. The products
// quantize their activations to q8_1 as llama.cpp does, so they are held to that arithmetic's
// error; on a Tesla V100 the matrix kernel's FP16 route (FP16 activations and weights) stays well
// within it. Up to eight tokens take the vector kernel. Wider calls need device-resident banks
// (each matrix followed by 256 zero bytes, which the matrix kernel's last K step of a 640-value
// down row reads; banks not declared device resident are refused) take the matrix kernel
// wherever the block type has one (MXFP4 stays on the vector kernel, and IQ1_M gate and up
// products until a call routes more than 16 rows to each expert: 260 tokens stay on the vector
// kernel, 820 take the matrix kernel's 32-wide column tiles), with one or several column tiles per
// expert; up to kGgufMoeSplitTokens tokens, experts with at most kGgufMoeVectorPairs pairs keep
// the vector kernel. The mixed-routing cases route both kinds in one call. Where the CPU has the
// bank's block types, the decode form's second stage, and the few-pair experts of a wide call of
// up to kGgufMoeSplitTokens tokens, also come from CPU products (gguf_expert_cpu, its
// automatically selected backend), as a miss service with a CPU share supplies them, and the sum
// must equal the GPU's bit for bit; so must a wide call whose experts two stages divide (as
// staged misses run them).
// Expert-parallel parts over tables that hold only their own experts
// (the others null) combine to the whole call's y bit for bit; parts that also hold a run of the
// shared expert's width each combine close to it and within the oracle's bounds.

#include "core/arena.h"
#include "core/device_buffer.h"
#include "core/qtype.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/moe_experts.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"
#include "ops/moe_experts/gguf_expert_cpu.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

constexpr int kHidden = 2560, kWidth = 640, kExperts = 512, kTopK = 10;

// Independent stored Q2_0 decoder: binary16 multiplier and four adjacent 2-bit codes per byte.
// Do not use the production GPU decoder to construct this format's mathematical oracle.
float exact_q2_0(const std::uint8_t* block, int element) {
    const std::uint16_t scale = std::uint16_t(block[0]) | (std::uint16_t(block[1]) << 8U);
    const int exponent = (scale >> 10U) & 31, fraction = scale & 1023;
    if (exponent == 31) { throw std::runtime_error("nonfinite Q2_0 scale in fixture"); }
    const double magnitude = exponent ? std::ldexp(1024.0 + fraction, exponent - 25)
                                      : std::ldexp(double(fraction), -24);
    const double multiplier = scale & 0x8000 ? -magnitude : magnitude;
    const int code = (block[2 + element / 4] >> (2 * (element % 4))) & 3;
    return float(multiplier * (code - 1));
}

// OCP E2M1 values and unsigned E4M3 scale per 16 values, decoded without a production kernel.
float exact_nvfp4(const std::uint8_t* block, int element) {
    constexpr double values[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    const unsigned scale      = block[element / 16];
    if (scale & 128U) { throw std::runtime_error("invalid NVFP4 scale in fixture"); }
    const unsigned exponent = scale >> 3, fraction = scale & 7U;
    const double multiplier = scale == 127 ? 0
                              : exponent   ? std::ldexp(1.0 + fraction / 8.0, int(exponent) - 7)
                                           : std::ldexp(double(fraction), -9);
    const unsigned byte     = block[4 + (element / 16) * 8 + element % 8];
    const unsigned code     = (element % 16 < 8 ? byte : byte >> 4) & 15U;
    return float(multiplier * values[code & 7U] * (code & 8U ? -1 : 1));
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

std::uint16_t half_bits(float value) {
    const __half h = __float2half(value);
    std::uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// Byte offsets of each type's binary16 multipliers, set to finite values (gguf linear test).
std::vector<int> half_fields(QType q) {
    switch (q) {
    case QType::GGUF_Q2_K:
        return {80, 82};
    case QType::GGUF_Q3_K:
        return {108};
    case QType::GGUF_Q4_K:
    case QType::GGUF_Q5_K:
        return {0, 2};
    case QType::GGUF_Q6_K:
        return {208};
    case QType::GGUF_IQ1_M:
    case QType::GGUF_MXFP4:
    case QType::GGUF_NVFP4:
        return {};
    default:
        return {0};
    }
}

// Q6_K multiplies its binary16 factor by an int8 sub-scale and a 6-bit code (up to 128 x 32), so
// its factors are a hundred times smaller than the other types', as a quantizer writes them (the
// factor is the largest sub-scale's 1/128), and its weights stay within a real model's. With the
// others' factors its weights reach 82 and a wide call's middles exceed what q8_1's binary16
// scale holds (127 x 65504 per 32 values), the vector products' and llama.cpp's limit.
std::vector<std::uint8_t> random_blocks(QType q, std::int32_t rows, std::int32_t k,
                                        std::mt19937& rng) {
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
            auto* sc              = reinterpret_cast<std::uint16_t*>(base + 48);
            for (int j = 0; j < 4; ++j) {
                sc[j] =
                    static_cast<std::uint16_t>((sc[j] & 0x0fff) | (((d >> (4 * j)) & 0xf) << 12));
            }
        }
        // MXFP4's exponent byte: scales 2^-12 .. 2^-7.
        if (q == QType::GGUF_MXFP4) { base[0] = static_cast<std::uint8_t>(115 + byte(rng) % 6); }
        // NVFP4's four UE4M3 scale bytes: halved scales 2^-10 .. 2^-5 (subnormals included).
        if (q == QType::GGUF_NVFP4) {
            for (int s = 0; s < 4; ++s) { base[s] = static_cast<std::uint8_t>(1 + byte(rng) % 31); }
        }
    }
    return bytes;
}

// `distinct` random matrices of one projection, the device table that maps expert e to matrix
// e % distinct, and their exact values.
// Zero bytes after every matrix, as a device-resident bank keeps them.
constexpr std::size_t kTail = 256;

struct Bank {
    QType format;
    std::int64_t row_bytes = 0;
    std::vector<DeviceBuffer> matrices;
    DeviceBuffer table;
    std::vector<std::vector<float>> exact; // [distinct][rows * k]
    std::vector<std::vector<std::uint8_t>> host; // [distinct] stored blocks
    std::vector<const void*> pointers;           // the table's entries
    int distinct = 1;

    Bank(QType q, int rows, int k, int count, int experts, std::mt19937& rng)
        : format(q), table(std::size_t(experts) * sizeof(void*)), distinct(count) {
        const auto block = gguf_block_shape(q);
        row_bytes        = std::int64_t(k / block.elements) * block.bytes;
        pointers.assign(experts, nullptr);
        for (int i = 0; i < count; ++i) {
            const auto bytes = random_blocks(q, rows, k, rng);
            matrices.emplace_back(bytes.size() + kTail);
            check(cudaMemset(matrices.back().p, 0, bytes.size() + kTail), "zero a bank");
            matrices.back().copy_from_host(bytes.data(), bytes.size());
            host.push_back(bytes);
            exact.emplace_back(std::size_t(rows) * k);
            if (q == QType::GGUF_Q2_0 || q == QType::GGUF_NVFP4) {
                for (std::size_t element = 0; element < exact.back().size(); ++element) {
                    const auto* packed    = bytes.data() + (element / 64) * block.bytes;
                    exact.back()[element] = q == QType::GGUF_Q2_0
                                                ? exact_q2_0(packed, int(element % 64))
                                                : exact_nvfp4(packed, int(element % 64));
                }
            } else {
                DeviceBuffer values(std::size_t(rows) * k * sizeof(float));
                ops::gguf::dequantize_rows(ops::detail::gguf_type(q), matrices.back().p, row_bytes, k,
                                           nullptr, rows, static_cast<float*>(values.p), k, nullptr);
                check(cudaDeviceSynchronize(), "dequantize");
                values.copy_to_host(exact.back().data(), exact.back().size() * sizeof(float));
            }
        }
        for (int e = 0; e < experts; ++e) { pointers[e] = matrices[e % count].p; }
        table.copy_from_host(pointers.data(), pointers.size() * sizeof(void*));
    }

    [[nodiscard]] ops::GgufExpertTable view() const {
        return {format, static_cast<const void* const*>(table.p), row_bytes};
    }

    [[nodiscard]] const float* rows_of(int expert) const { return exact[expert % distinct].data(); }

    // A device's table under expert parallelism: experts [begin, end), the other entries null.
    [[nodiscard]] DeviceBuffer range_table(int begin, int end) const {
        std::vector<const void*> entries(pointers.size(), nullptr);
        for (int e = begin; e < end; ++e) { entries[e] = pointers[e]; }
        DeviceBuffer out(entries.size() * sizeof(void*));
        out.copy_from_host(entries.data(), entries.size() * sizeof(void*));
        return out;
    }
};

double silu(double x) { return x / (1.0 + std::exp(-x)); }

// One expert's product for one token (BF16 words) on the CPU, as the miss service computes it.
std::vector<float> cpu_expert(const Bank& gate, const Bank& up, const Bank& down, int expert,
                              const std::uint16_t* x, const float* scales = nullptr) {
    const auto matrix = [&](const Bank& bank, int rows, int k) {
        return ops::GgufCpuMatrix{bank.format, bank.host[expert % bank.distinct].data(),
                                  bank.row_bytes, rows, k};
    };
    constexpr auto backend = ops::GgufCpuBackend::Automatic;
    ops::GgufCpuActivation in(1, kHidden), in_up(1, kHidden), middle(1, kWidth);
    in.prepare_bf16(gate.format, x, kHidden, 1);
    in_up.prepare_bf16(up.format, x, kHidden, 1);
    std::vector<float> g(kWidth), u(kWidth), y(kHidden);
    ops::gguf_cpu_rows(matrix(gate, kWidth, kHidden), in, 0, kWidth, g.data(), kWidth, backend);
    ops::gguf_cpu_rows(matrix(up, kWidth, kHidden), in_up, 0, kWidth, u.data(), kWidth, backend);
    if (scales != nullptr) {
        for (int r = 0; r < kWidth; ++r) {
            g[r] *= scales[0];
            u[r] *= scales[1];
        }
    }
    std::vector<std::uint16_t> m(kWidth);
    ops::gguf_cpu_swiglu_bf16(g.data(), u.data(), kWidth, m.data(), backend);
    middle.prepare_bf16(down.format, m.data(), kWidth, 1);
    ops::gguf_cpu_rows(matrix(down, kHidden, kWidth), middle, 0, kHidden, y.data(), kHidden,
                       backend);
    if (scales != nullptr) {
        for (auto& value : y) { value *= scales[2]; }
    }
    return y;
}

// down . (silu(gate . x) * (up . x)) in FP64.
std::vector<double> expert_output(const float* gate, const float* up, const float* down,
                                  const std::vector<float>& x, const float* scales = nullptr) {
    std::vector<double> middle(kWidth);
    for (int r = 0; r < kWidth; ++r) {
        double g = 0, u = 0;
        const float* gr = gate + std::size_t(r) * kHidden;
        const float* ur = up + std::size_t(r) * kHidden;
        for (int i = 0; i < kHidden; ++i) {
            g += double(gr[i]) * x[i];
            u += double(ur[i]) * x[i];
        }
        middle[r] = silu(g * (scales ? scales[0] : 1)) * u * (scales ? scales[1] : 1);
    }
    std::vector<double> out(kHidden);
    for (int r = 0; r < kHidden; ++r) {
        double acc      = 0;
        const float* dr = down + std::size_t(r) * kWidth;
        for (int i = 0; i < kWidth; ++i) { acc += double(dr[i]) * middle[i]; }
        out[r] = acc * (scales ? scales[2] : 1);
    }
    return out;
}

struct Case {
    const char* name;
    QType gate, up, down, shared_gate, shared_up, shared_down;
    int distinct;
    std::vector<int> tokens;
    int expert_pool; // routes draw from experts [0, expert_pool)
    bool scaled = false;
};

int run_case(const Case& c, std::uint32_t seed) {
    std::mt19937 rng(seed);
    const Bank gate(c.gate, kWidth, kHidden, c.distinct, kExperts, rng);
    const Bank up(c.up, kWidth, kHidden, c.distinct, kExperts, rng);
    const Bank down(c.down, kHidden, kWidth, c.distinct, kExperts, rng);
    const Bank sgate(c.shared_gate, kWidth, kHidden, 1, 1, rng);
    const Bank sup(c.shared_up, kWidth, kHidden, 1, 1, rng);
    const Bank sdown(c.shared_down, kHidden, kWidth, 1, 1, rng);
    std::vector<float> scales(kExperts * 3);
    for (int e = 0; e < kExperts; ++e) {
        scales[3 * e]     = 0.7F + float(e % 7) * 0.1F;
        scales[3 * e + 1] = 0.8F + float(e % 5) * 0.1F;
        scales[3 * e + 2] = 0.6F + float(e % 11) * 0.05F;
    }
    DeviceBuffer d_scales(c.scaled ? scales.size() * sizeof(float) : 0);
    if (c.scaled) { d_scales.copy_from_host(scales.data(), scales.size() * sizeof(float)); }
    int failures = 0;
    for (const int tokens : c.tokens) {
        const bool device_resident = true;
        const ops::GgufMoeWeights banks{gate.view(),
                                        up.view(),
                                        down.view(),
                                        sgate.view(),
                                        sup.view(),
                                        sdown.view(),
                                        kExperts,
                                        device_resident,
                                        kWidth,
                                        true,
                                        static_cast<const float*>(d_scales.p)};
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::uniform_real_distribution<float> unit(0.05f, 1.0f);
        std::vector<__nv_bfloat16> m(std::size_t(kHidden) * tokens);
        for (auto& v : m) { v = __float2bfloat16(normal(rng)); }
        std::vector<std::int32_t> ids(std::size_t(kTopK) * tokens);
        std::vector<float> weights(ids.size()), shared(tokens);
        std::vector<int> pool(c.expert_pool);
        std::iota(pool.begin(), pool.end(), 0);
        for (int t = 0; t < tokens; ++t) {
            std::shuffle(pool.begin(), pool.end(), rng);
            float total = 0;
            for (int j = 0; j < kTopK; ++j) {
                ids[t * kTopK + j]     = pool[j];
                weights[t * kTopK + j] = unit(rng);
                total += weights[t * kTopK + j];
            }
            for (int j = 0; j < kTopK; ++j) { weights[t * kTopK + j] /= total; }
            shared[t] = unit(rng);
        }
        DeviceBuffer d_m(m.size() * 2), d_ids(ids.size() * 4), d_w(weights.size() * 4),
            d_s(shared.size() * 4), d_y(std::size_t(kHidden) * tokens * 4);
        d_m.copy_from_host(m.data(), m.size() * 2);
        d_ids.copy_from_host(ids.data(), ids.size() * 4);
        d_w.copy_from_host(weights.data(), weights.size() * 4);
        d_s.copy_from_host(shared.data(), shared.size() * 4);
        Tensor t_m(d_m.p, DType::BF16, {kHidden, tokens}),
            t_ids(d_ids.p, DType::I32, {kTopK, tokens}), t_w(d_w.p, DType::FP32, {kTopK, tokens}),
            t_s(d_s.p, DType::FP32, {tokens}), t_y(d_y.p, DType::FP32, {kHidden, tokens});
        WorkspaceArena workspace(ops::moe_experts_gguf_workspace_bytes(tokens, c.scaled));
        ops::moe_experts_gguf(t_m, t_ids, t_w, t_s, banks, workspace, t_y, nullptr);
        check(cudaDeviceSynchronize(), "moe_experts_gguf");
        std::vector<float> got(std::size_t(kHidden) * tokens);
        d_y.copy_to_host(got.data(), got.size() * 4);
        // A second run must reproduce every bit: the accumulation is order independent.
        ops::moe_experts_gguf(t_m, t_ids, t_w, t_s, banks, workspace, t_y, nullptr);
        check(cudaDeviceSynchronize(), "moe_experts_gguf repeat");
        std::vector<float> again(got.size());
        d_y.copy_to_host(again.data(), again.size() * 4);
        if (std::memcmp(got.data(), again.data(), got.size() * 4) != 0) {
            std::cerr << "FAIL " << c.name << " T=" << tokens << ": a repeat differs\n";
            ++failures;
        }
        // The FP64 oracle of every output, and a y's relative L2 and gross error against it.
        std::vector<double> oracle(std::size_t(kHidden) * tokens, 0.0);
        for (int t = 0; t < tokens; ++t) {
            std::vector<float> x(kHidden);
            for (int i = 0; i < kHidden; ++i) {
                x[i] = __bfloat162float(m[std::size_t(t) * kHidden + i]);
            }
            double* want = oracle.data() + std::size_t(t) * kHidden;
            // Scaled experts may share blocks but keep different external multipliers.
            std::vector<std::vector<double>> outputs(c.scaled ? c.expert_pool : c.distinct);
            for (int j = 0; j < kTopK; ++j) {
                const int e = ids[t * kTopK + j];
                auto& o     = outputs[c.scaled ? e : e % c.distinct];
                if (o.empty()) {
                    o = expert_output(gate.rows_of(e), up.rows_of(e), down.rows_of(e), x,
                                      c.scaled ? scales.data() + 3 * e : nullptr);
                }
                for (int r = 0; r < kHidden; ++r) { want[r] += weights[t * kTopK + j] * o[r]; }
            }
            const auto s = expert_output(sgate.rows_of(0), sup.rows_of(0), sdown.rows_of(0), x);
            for (int r = 0; r < kHidden; ++r) { want[r] += shared[t] * s[r]; }
        }
        const auto against_oracle = [&](const std::vector<float>& y, const std::string& what) {
            double num = 0, den = 0, worst = 0, scale = 0;
            for (std::size_t i = 0; i < oracle.size(); ++i) {
                const double d = double(y[i]) - oracle[i];
                num += d * d;
                den += oracle[i] * oracle[i];
                worst = std::max(worst, std::abs(d));
                scale = std::max(scale, std::abs(oracle[i]));
            }
            const double rms = std::sqrt(num / std::max(den, 1e-30)),
                         max = worst / std::max(scale, 1e-30);
            // Two q8_1 activations (the token and the middle) per product chain.
            const bool ok = std::isfinite(rms) && rms < 4e-2 && max < 1.6e-1;
            std::cout << (ok ? "OK   " : "FAIL ") << c.name << " T=" << tokens
                      << what
                      << " rms=" << rms << " max=" << max << '\n';
            return ok ? 0 : 1;
        };
        {
            // Expert parallelism: parts over range tables, the shared expert in the first, or in
            // a later one, combine to the whole call's y bit for bit.
            const std::vector<std::vector<int>> splits = {{0, 200, 512}, {0, 1, 300, 511, 512}};
            for (std::size_t s = 0; s < splits.size(); ++s) {
                const auto& ends     = splits[s];
                const std::size_t n  = ends.size() - 1;
                const std::size_t at = s == 0 ? 0 : n - 1; // the part with the shared expert
                std::vector<DeviceBuffer> tables, fixed;
                std::vector<Tensor> parts;
                WorkspaceArena part_workspace(
                    ops::moe_experts_gguf_part_workspace_bytes(tokens, c.scaled));
                for (std::size_t p = 0; p < n; ++p) {
                    const int begin = ends[p], end = ends[p + 1];
                    tables.push_back(gate.range_table(begin, end));
                    tables.push_back(up.range_table(begin, end));
                    tables.push_back(down.range_table(begin, end));
                    const auto view = [&](const Bank& bank, std::size_t i) {
                        return ops::GgufExpertTable{bank.format,
                                                    static_cast<const void* const*>(tables[i].p),
                                                    bank.row_bytes};
                    };
                    const std::size_t base = tables.size() - 3;
                    const ops::GgufMoeWeights part_banks{view(gate, base),
                                                         view(up, base + 1),
                                                         view(down, base + 2),
                                                         sgate.view(),
                                                         sup.view(),
                                                         sdown.view(),
                                                         kExperts,
                                                         device_resident,
                                                         kWidth,
                                                         true,
                                                         static_cast<const float*>(d_scales.p)};
                    fixed.emplace_back(std::size_t(kHidden) * tokens * 8);
                    parts.push_back(Tensor(fixed.back().p, DType::I64, {kHidden, tokens}));
                    ops::moe_experts_gguf_part(t_m, t_ids, t_w, p == at ? &t_s : nullptr,
                                               part_banks, begin, end, part_workspace, parts.back(),
                                               nullptr);
                    if (part_workspace.used() != 0) {
                        std::cerr << "FAIL " << c.name << " T=" << tokens
                                  << ": a part's workspace scope leaked\n";
                        ++failures;
                    }
                }
                ops::moe_experts_gguf_combine(parts, t_y, nullptr);
                check(cudaDeviceSynchronize(), "expert-parallel parts");
                std::vector<float> combined(got.size());
                d_y.copy_to_host(combined.data(), combined.size() * 4);
                const bool same = std::memcmp(got.data(), combined.data(), got.size() * 4) == 0;
                std::cout << (same ? "OK   " : "FAIL ") << c.name << " T=" << tokens
                          << ": " << n << " expert-parallel parts " << (same ? "equal" : "differ from")
                          << " the whole call\n";
                failures += same ? 0 : 1;
            }
        }
        {
            // Tensor parallelism: parts that each hold a run of the shared expert's width as banks
            // of their own (those gate and up rows, those values of every down row, zeros after
            // the down piece) combine close to the whole call, each part rounding its run's down
            // dot on its own, and within the oracle's bounds.
            struct Layout {
                std::vector<int> experts, shared;
            };

            const std::vector<Layout> layouts = {{{0, 200, 512}, {0, 256, 640}},
                                                 {{0, 1, 300, 511, 512}, {0, 128, 256, 384, 640}}};
            const auto block                  = gguf_block_shape(sdown.format);
            for (const Layout& layout : layouts) {
                const std::size_t n = layout.experts.size() - 1;
                std::vector<DeviceBuffer> buffers, fixed;
                std::vector<Tensor> parts;
                WorkspaceArena part_workspace(
                    ops::moe_experts_gguf_part_workspace_bytes(tokens, c.scaled));
                const auto table_of = [&](const DeviceBuffer& run) {
                    DeviceBuffer table(sizeof(void*));
                    table.copy_from_host(&run.p, sizeof(void*));
                    return table;
                };
                const auto view = [](QType format, const DeviceBuffer& table,
                                     std::int64_t row_bytes) {
                    return ops::GgufExpertTable{format, static_cast<const void* const*>(table.p),
                                                row_bytes};
                };
                for (std::size_t p = 0; p < n; ++p) {
                    const int begin = layout.experts[p], end = layout.experts[p + 1];
                    const int first = layout.shared[p], width = layout.shared[p + 1] - first;
                    const std::int64_t down_bytes =
                        std::int64_t(width / block.elements) * block.bytes;
                    DeviceBuffer gate_run(std::size_t(width) * sgate.row_bytes + kTail),
                        up_run(std::size_t(width) * sup.row_bytes + kTail),
                        down_run(std::size_t(kHidden) * down_bytes + kTail);
                    check(cudaMemset(gate_run.p, 0, std::size_t(width) * sgate.row_bytes + kTail),
                          "zero a gate run");
                    check(cudaMemset(up_run.p, 0, std::size_t(width) * sup.row_bytes + kTail),
                          "zero an up run");
                    check(cudaMemset(down_run.p, 0, std::size_t(kHidden) * down_bytes + kTail),
                          "zero a down run");
                    check(cudaMemcpy(gate_run.p,
                                     static_cast<const std::uint8_t*>(sgate.matrices[0].p) +
                                         std::size_t(first) * sgate.row_bytes,
                                     std::size_t(width) * sgate.row_bytes,
                                     cudaMemcpyDeviceToDevice),
                          "copy a gate run");
                    check(cudaMemcpy(up_run.p,
                                     static_cast<const std::uint8_t*>(sup.matrices[0].p) +
                                         std::size_t(first) * sup.row_bytes,
                                     std::size_t(width) * sup.row_bytes, cudaMemcpyDeviceToDevice),
                          "copy an up run");
                    check(cudaMemcpy2D(down_run.p, std::size_t(down_bytes),
                                       static_cast<const std::uint8_t*>(sdown.matrices[0].p) +
                                           std::size_t(first / block.elements) * block.bytes,
                                       std::size_t(sdown.row_bytes), std::size_t(down_bytes),
                                       kHidden, cudaMemcpyDeviceToDevice),
                          "copy a down run");
                    buffers.push_back(gate.range_table(begin, end));
                    buffers.push_back(up.range_table(begin, end));
                    buffers.push_back(down.range_table(begin, end));
                    buffers.push_back(table_of(gate_run));
                    buffers.push_back(table_of(up_run));
                    buffers.push_back(table_of(down_run));
                    const std::size_t base = buffers.size() - 6;
                    const ops::GgufMoeWeights part_banks{
                        view(gate.format, buffers[base], gate.row_bytes),
                        view(up.format, buffers[base + 1], up.row_bytes),
                        view(down.format, buffers[base + 2], down.row_bytes),
                        view(sgate.format, buffers[base + 3], sgate.row_bytes),
                        view(sup.format, buffers[base + 4], sup.row_bytes),
                        view(sdown.format, buffers[base + 5], down_bytes),
                        kExperts,
                        device_resident,
                        width,
                        true,
                        static_cast<const float*>(d_scales.p)};
                    buffers.push_back(std::move(gate_run));
                    buffers.push_back(std::move(up_run));
                    buffers.push_back(std::move(down_run));
                    fixed.emplace_back(std::size_t(kHidden) * tokens * 8);
                    parts.push_back(Tensor(fixed.back().p, DType::I64, {kHidden, tokens}));
                    ops::moe_experts_gguf_part(t_m, t_ids, t_w, &t_s, part_banks, begin, end,
                                               part_workspace, parts.back(), nullptr);
                    check(cudaDeviceSynchronize(), "a part with a shared run");
                    if (part_workspace.used() != 0) {
                        std::cerr << "FAIL " << c.name << " T=" << tokens
                                  << ": a part's workspace scope leaked\n";
                        ++failures;
                    }
                }
                ops::moe_experts_gguf_combine(parts, t_y, nullptr);
                check(cudaDeviceSynchronize(), "parts with shared runs");
                std::vector<float> combined(got.size());
                d_y.copy_to_host(combined.data(), combined.size() * 4);
                double worst = 0, scale = 0;
                for (std::size_t i = 0; i < got.size(); ++i) {
                    worst = std::max(worst, std::abs(double(combined[i]) - double(got[i])));
                    scale = std::max(scale, std::abs(double(got[i])));
                }
                const bool close = std::isfinite(worst) && worst <= 1e-4 * scale;
                std::cout << (close ? "OK   " : "FAIL ") << c.name << " T=" << tokens
                          << ": " << n << " parts with shared runs differ from the whole call by "
                          << worst / std::max(scale, 1e-30) << " of its largest output\n";
                failures += close ? 0 : 1;
                failures +=
                    against_oracle(combined, ", " + std::to_string(n) + " parts with shared runs,");
            }
        }
        if (ops::moe_experts_gguf_decode_supported(banks, tokens)) {
            // The decode form must equal the separate kernels bit for bit, whole and split into a
            // first stage and a second (the even and odd pairs) as missing experts are served.
            const auto compare = [&](const char* what) {
                check(cudaDeviceSynchronize(), what);
                std::vector<float> other(got.size());
                d_y.copy_to_host(other.data(), other.size() * 4);
                if (std::memcmp(got.data(), other.data(), got.size() * 4) != 0) {
                    std::cerr << "FAIL " << c.name << " T=" << tokens << ": " << what
                              << " differs from the decode form\n";
                    ++failures;
                }
            };
            {
                auto scope       = workspace.scope();
                const auto stage = ops::moe_experts_gguf_begin(t_m, workspace, nullptr);
                ops::moe_experts_gguf_add(stage, t_m, t_ids, t_w, &t_s, banks, workspace, nullptr);
                ops::moe_experts_gguf_finish(stage, t_y, nullptr);
            }
            compare("the separate kernels");
            std::vector<std::int32_t> first(ids.size()), second(ids.size());
            for (std::size_t p = 0; p < ids.size(); ++p) {
                const bool odd = ids[p] % 2 != 0;
                first[p]       = odd ? -1 : ids[p];
                second[p]      = odd ? ids[p] : -1;
            }
            DeviceBuffer d_first(ids.size() * 4), d_second(ids.size() * 4);
            d_first.copy_from_host(first.data(), first.size() * 4);
            d_second.copy_from_host(second.data(), second.size() * 4);
            const Tensor t_first(d_first.p, DType::I32, {kTopK, tokens});
            const Tensor t_second(d_second.p, DType::I32, {kTopK, tokens});
            {
                auto scope       = workspace.scope();
                const auto state = ops::moe_experts_gguf_decode_begin(t_m, t_first, t_s, banks,
                                                                      workspace, nullptr);
                ops::moe_experts_gguf_decode_missing(state, t_m, t_second, banks, nullptr);
                ops::moe_experts_gguf_decode_finish(state, t_first, &t_second, t_w, banks, &banks,
                                                    nullptr, nullptr, nullptr, t_y, nullptr);
            }
            compare("the two-stage decode form");
            {
                // The shared expert on a side stream, joined before the sum is read.
                ops::GgufMoeSide side;
                check(cudaStreamCreateWithFlags(&side.stream, cudaStreamNonBlocking), "side stream");
                check(cudaEventCreateWithFlags(&side.fork, cudaEventDisableTiming), "fork event");
                check(cudaEventCreateWithFlags(&side.join, cudaEventDisableTiming), "join event");
                {
                    auto scope       = workspace.scope();
                    const auto state = ops::moe_experts_gguf_decode_begin(
                        t_m, t_first, t_s, banks, workspace, nullptr, &side);
                    ops::moe_experts_gguf_decode_missing(state, t_m, t_second, banks, nullptr);
                    ops::moe_experts_gguf_decode_finish(state, t_first, &t_second, t_w, banks,
                                                        &banks, nullptr, nullptr, nullptr, t_y,
                                                        nullptr);
                }
                compare("the decode form with a side stream");
                check(cudaEventDestroy(side.fork), "fork event");
                check(cudaEventDestroy(side.join), "join event");
                check(cudaStreamDestroy(side.stream), "side stream");
            }
            {
                auto scope       = workspace.scope();
                const auto stage = ops::moe_experts_gguf_begin(t_m, workspace, nullptr);
                ops::moe_experts_gguf_add(stage, t_m, t_first, t_w, &t_s, banks, workspace, nullptr);
                ops::moe_experts_gguf_add(stage, t_m, t_second, t_w, nullptr, banks, workspace,
                                          nullptr);
                ops::moe_experts_gguf_finish(stage, t_y, nullptr);
            }
            compare("the two-stage separate kernels");
            if (ops::gguf_cpu_supported(c.gate) && ops::gguf_cpu_supported(c.up) &&
                ops::gguf_cpu_supported(c.down)) {
                // The second stage's pairs as CPU products, the GPU running only the first.
                std::vector<float> products;
                std::vector<std::int32_t> pairs;
                for (std::size_t p = 0; p < ids.size(); ++p) {
                    if (second[p] < 0) { continue; }
                    const auto* x = reinterpret_cast<const std::uint16_t*>(m.data()) +
                                    (p / kTopK) * std::size_t(kHidden);
                    const auto y = cpu_expert(gate, up, down, second[p], x);
                    products.insert(products.end(), y.begin(), y.end());
                    pairs.push_back(std::int32_t(p));
                }
                const std::int32_t count = std::int32_t(pairs.size());
                const std::vector<std::int32_t> none(ids.size(), -1);
                DeviceBuffer d_products(std::max<std::size_t>(products.size(), 1) * 4),
                    d_pairs(std::max<std::size_t>(pairs.size(), 1) * 4), d_count(4),
                    d_none(ids.size() * 4);
                if (count > 0) {
                    d_products.copy_from_host(products.data(), products.size() * 4);
                    d_pairs.copy_from_host(pairs.data(), pairs.size() * 4);
                }
                d_count.copy_from_host(&count, 4);
                d_none.copy_from_host(none.data(), none.size() * 4);
                const Tensor t_none(d_none.p, DType::I32, {kTopK, tokens});
                {
                    auto scope       = workspace.scope();
                    const auto state = ops::moe_experts_gguf_decode_begin(t_m, t_first, t_s, banks,
                                                                          workspace, nullptr);
                    ops::moe_experts_gguf_decode_missing(state, t_m, t_none, banks, nullptr);
                    ops::moe_experts_gguf_decode_finish(
                        state, t_first, &t_none, t_w, banks, &banks,
                        static_cast<const float*>(d_products.p),
                        static_cast<const std::int32_t*>(d_pairs.p),
                        static_cast<const std::int32_t*>(d_count.p), t_y, nullptr);
                }
                check(cudaDeviceSynchronize(), "the decode form with CPU products");
                std::vector<float> hybrid(got.size());
                d_y.copy_to_host(hybrid.data(), hybrid.size() * 4);
                std::size_t differ = 0;
                for (std::size_t i = 0; i < got.size(); ++i) {
                    differ += std::memcmp(&hybrid[i], &got[i], 4) != 0 ? 1 : 0;
                }
                std::cout << (differ == 0 ? "OK   " : "FAIL ") << c.name << " T=" << tokens
                          << " with " << count << " CPU products ("
                          << ops::gguf_cpu_backend_name(ops::GgufCpuBackend::Automatic)
                          << "): " << differ << " of " << got.size() << " outputs differ from the GPU's\n";
                failures += differ == 0 ? 0 : 1;
            }
        }
        if (tokens > 8 || c.scaled) {
            // Banks not declared device resident are refused past eight tokens.
            if (tokens > 8) {
                ops::GgufMoeWeights mapped = banks;
                mapped.device_resident     = false;
                bool refused               = false;
                try {
                    ops::moe_experts_gguf(t_m, t_ids, t_w, t_s, mapped, workspace, t_y, nullptr);
                } catch (const std::invalid_argument&) { refused = true; }
                std::cout << (refused ? "OK   " : "FAIL ") << c.name << " T=" << tokens
                          << ": banks read across the bus are " << (refused ? "" : "not ")
                          << "refused\n";
                failures += refused ? 0 : 1;
            }
            // The experts of each kind: a mixed call routes both.
            std::vector<int> uses(kExperts, 0);
            for (const std::int32_t e : ids) { ++uses[e]; }
            int few = 0, many = 0;
            for (const int n : uses) {
                few += n > 0 && n <= ops::kGgufMoeVectorPairs;
                many += n > ops::kGgufMoeVectorPairs;
            }
            const auto stages = [&](const std::vector<std::int32_t>& first,
                                    const std::vector<std::int32_t>& second,
                                    const std::vector<float>* products,
                                    const std::vector<std::int32_t>* pairs) {
                DeviceBuffer d_first(ids.size() * 4), d_second(ids.size() * 4), d_count(4),
                    d_products(std::max<std::size_t>(products ? products->size() : 0, 1) * 4),
                    d_pairs(std::max<std::size_t>(pairs ? pairs->size() : 0, 1) * 4);
                d_first.copy_from_host(first.data(), first.size() * 4);
                d_second.copy_from_host(second.data(), second.size() * 4);
                const std::int32_t count = pairs ? std::int32_t(pairs->size()) : 0;
                if (count > 0) {
                    d_products.copy_from_host(products->data(), products->size() * 4);
                    d_pairs.copy_from_host(pairs->data(), pairs->size() * 4);
                }
                d_count.copy_from_host(&count, 4);
                const Tensor t_first(d_first.p, DType::I32, {kTopK, tokens});
                const Tensor t_second(d_second.p, DType::I32, {kTopK, tokens});
                {
                    auto scope       = workspace.scope();
                    const auto stage = ops::moe_experts_gguf_begin(t_m, workspace, nullptr);
                    ops::moe_experts_gguf_add(stage, t_m, t_first, t_w, &t_s, banks, workspace,
                                              nullptr);
                    ops::moe_experts_gguf_add(stage, t_m, t_second, t_w, nullptr, banks, workspace,
                                              nullptr);
                    if (count > 0) {
                        ops::moe_experts_gguf_add_products(
                            stage, static_cast<const float*>(d_products.p),
                            static_cast<const std::int32_t*>(d_pairs.p),
                            static_cast<const std::int32_t*>(d_count.p), count, t_w, nullptr);
                    }
                    ops::moe_experts_gguf_finish(stage, t_y, nullptr);
                }
                check(cudaDeviceSynchronize(), "a staged wide call");
                std::vector<float> staged(got.size());
                d_y.copy_to_host(staged.data(), staged.size() * 4);
                std::size_t differ = 0;
                for (std::size_t i = 0; i < got.size(); ++i) {
                    differ += std::memcmp(&staged[i], &got[i], 4) != 0 ? 1 : 0;
                }
                return differ;
            };
            {
                // Two stages that divide the experts (the even ones, then the odd), as the staged
                // misses' cached stage and rounds do: every expert's pairs in one stage.
                std::vector<std::int32_t> first(ids.size(), -1), second(ids.size(), -1);
                for (std::size_t p = 0; p < ids.size(); ++p) {
                    (ids[p] % 2 == 0 ? first : second)[p] = ids[p];
                }
                const std::size_t differ = stages(first, second, nullptr, nullptr);
                std::cout << (differ == 0 ? "OK   " : "FAIL ") << c.name << " T=" << tokens
                          << " (" << few << " few-pair, " << many
                          << " many-pair experts) in two stages: " << differ << " of "
                          << got.size() << " outputs differ from the whole call\n";
                failures += differ == 0 ? 0 : 1;
            }
            if (tokens <= ops::kGgufMoeSplitTokens && ops::gguf_cpu_supported(c.gate) &&
                ops::gguf_cpu_supported(c.up) && ops::gguf_cpu_supported(c.down)) {
                // The odd few-pair experts as CPU products, as a miss service with a CPU share
                // supplies them; the GPU runs the rest.
                std::vector<std::int32_t> first(ids.size(), -1), second(ids.size(), -1), pairs;
                std::vector<float> products;
                for (std::size_t p = 0; p < ids.size(); ++p) {
                    const std::int32_t e = ids[p];
                    if (e % 2 != 0 && uses[e] <= ops::kGgufMoeVectorPairs) {
                        const auto* x = reinterpret_cast<const std::uint16_t*>(m.data()) +
                                        (p / kTopK) * std::size_t(kHidden);
                        const auto y = cpu_expert(gate, up, down, e, x,
                                                  c.scaled ? scales.data() + 3 * e : nullptr);
                        products.insert(products.end(), y.begin(), y.end());
                        pairs.push_back(std::int32_t(p));
                    } else {
                        (e % 2 == 0 ? first : second)[p] = e;
                    }
                }
                const std::size_t differ = stages(first, second, &products, &pairs);
                std::cout << (differ == 0 ? "OK   " : "FAIL ") << c.name << " T=" << tokens
                          << " with " << pairs.size() << " CPU products of few-pair experts ("
                          << ops::gguf_cpu_backend_name(ops::GgufCpuBackend::Automatic)
                          << "): " << differ << " of " << got.size()
                          << " outputs differ from the GPU's\n";
                failures += differ == 0 ? 0 : 1;
            }
        }
        failures += against_oracle(got, "");
    }
    return failures;
}

// A call of more than 6,553 tokens (more pairs than a grid has rows) equals the same tokens in two
// calls bit for bit: Flash-Next's IQ3_XXS gate and up with IQ4_NL down on the matrix kernel, and
// MXFP4, which only the vector kernel takes, its down products' input quantized over every pair.
int wide_call(const char* name, QType gate_up, QType down_format, std::uint32_t seed) {
    constexpr int kTokens = 6600, kHalf = kTokens / 2;
    std::mt19937 rng(seed);
    using Q = QType;
    const Bank gate(gate_up, kWidth, kHidden, 4, kExperts, rng);
    const Bank up(gate_up, kWidth, kHidden, 4, kExperts, rng);
    const Bank down(down_format, kHidden, kWidth, 4, kExperts, rng);
    const Bank sgate(Q::GGUF_Q6_K, kWidth, kHidden, 1, 1, rng);
    const Bank sup(Q::GGUF_Q6_K, kWidth, kHidden, 1, 1, rng);
    const Bank sdown(Q::GGUF_IQ4_NL, kHidden, kWidth, 1, 1, rng);
    const ops::GgufMoeWeights banks{gate.view(), up.view(),    down.view(), sgate.view(),
                                    sup.view(),  sdown.view(), kExperts,    true,
                                    kWidth,      true,         nullptr};
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> unit(0.05f, 1.0f);
    std::vector<__nv_bfloat16> m(std::size_t(kHidden) * kTokens);
    for (auto& v : m) { v = __float2bfloat16(normal(rng)); }
    std::vector<std::int32_t> ids(std::size_t(kTopK) * kTokens);
    std::vector<float> weights(ids.size()), shared(kTokens);
    std::vector<int> pool(kExperts);
    std::iota(pool.begin(), pool.end(), 0);
    for (int t = 0; t < kTokens; ++t) {
        std::shuffle(pool.begin(), pool.end(), rng);
        for (int j = 0; j < kTopK; ++j) {
            ids[t * kTopK + j]     = pool[j];
            weights[t * kTopK + j] = unit(rng);
        }
        shared[t] = unit(rng);
    }
    DeviceBuffer d_m(m.size() * 2), d_ids(ids.size() * 4), d_w(weights.size() * 4),
        d_s(shared.size() * 4), d_y(std::size_t(kHidden) * kTokens * 4),
        d_halves(std::size_t(kHidden) * kTokens * 4);
    d_m.copy_from_host(m.data(), m.size() * 2);
    d_ids.copy_from_host(ids.data(), ids.size() * 4);
    d_w.copy_from_host(weights.data(), weights.size() * 4);
    d_s.copy_from_host(shared.data(), shared.size() * 4);
    WorkspaceArena workspace(ops::moe_experts_gguf_workspace_bytes(kTokens, false));
    const auto run = [&](int first, int tokens, void* y) {
        const auto at = [&](const DeviceBuffer& b, std::size_t per_token, std::size_t bytes) {
            return static_cast<std::uint8_t*>(b.p) + std::size_t(first) * per_token * bytes;
        };
        Tensor t_m(at(d_m, kHidden, 2), DType::BF16, {kHidden, tokens}),
            t_ids(at(d_ids, kTopK, 4), DType::I32, {kTopK, tokens}),
            t_w(at(d_w, kTopK, 4), DType::FP32, {kTopK, tokens}),
            t_s(at(d_s, 1, 4), DType::FP32, {tokens}),
            t_y(static_cast<std::uint8_t*>(y) + std::size_t(first) * kHidden * 4, DType::FP32,
                {kHidden, tokens});
        ops::moe_experts_gguf(t_m, t_ids, t_w, t_s, banks, workspace, t_y, nullptr);
        check(cudaDeviceSynchronize(), "moe_experts_gguf wide");
    };
    run(0, kTokens, d_y.p);
    run(0, kHalf, d_halves.p);
    run(kHalf, kTokens - kHalf, d_halves.p);
    std::vector<float> whole(std::size_t(kHidden) * kTokens), halves(whole.size());
    d_y.copy_to_host(whole.data(), whole.size() * 4);
    d_halves.copy_to_host(halves.data(), halves.size() * 4);
    bool finite = true;
    for (const float v : whole) { finite = finite && std::isfinite(v); }
    const bool same = finite && std::memcmp(whole.data(), halves.data(), whole.size() * 4) == 0;
    std::cout << (same ? "OK   " : "FAIL ") << name << " wide call T=" << kTokens
              << ": equals two calls of " << kHalf << '\n';
    return same ? 0 : 1;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        using Q                       = QType;
        const std::vector<Case> cases = {
            // The GSQ-RCO Q2_0 release: fused Q2_0 gate/up, Q2_0 down, a mixed shared expert.
            {"q2_0 bank",
             Q::GGUF_Q2_0,
             Q::GGUF_Q2_0,
             Q::GGUF_Q2_0,
             Q::GGUF_IQ4_XS,
             Q::GGUF_Q3_K,
             Q::GGUF_Q5_0,
             8,
             {1, 2, 3, 8, 9, 16, 24, 40, 48},
             kExperts},
            // Routes over 30 experts: past eight tokens a call holds experts of up to eight pairs
            // (the vector kernel's) beside experts of more (the matrix kernel's).
            {"q2_0 mixed routing",
             Q::GGUF_Q2_0,
             Q::GGUF_Q2_0,
             Q::GGUF_Q2_0,
             Q::GGUF_IQ4_XS,
             Q::GGUF_Q3_K,
             Q::GGUF_Q5_0,
             8,
             {9, 24, 64},
             30},
            // Experts concentrated on a few ids: several columns per pass (chunks 4 and 8).
            {"q2_0 dense routing",
             Q::GGUF_Q2_0,
             Q::GGUF_Q2_0,
             Q::GGUF_Q2_0,
             Q::GGUF_Q4_K,
             Q::GGUF_Q4_K,
             Q::GGUF_Q4_0,
             2,
             {160, 260},
             12},
            // Gate and up in different types take the unfused path.
            {"iq2_s/iq2_xxs unfused",
             Q::GGUF_IQ2_S,
             Q::GGUF_IQ2_XXS,
             Q::GGUF_IQ4_NL,
             Q::GGUF_IQ3_S,
             Q::GGUF_IQ3_S,
             Q::GGUF_Q8_0,
             4,
             {1, 5, 24},
             kExperts},
            // IQ2_S's two half-slice factors on the decode form, against the CPU's products.
            {"iq2_s bank",
             Q::GGUF_IQ2_S,
             Q::GGUF_IQ2_S,
             Q::GGUF_Q8_0,
             Q::GGUF_IQ4_XS,
             Q::GGUF_IQ4_XS,
             Q::GGUF_IQ4_NL,
             4,
             {1, 2, 8},
             kExperts},
            {"iq1_m/iq3_xxs",
             Q::GGUF_IQ1_M,
             Q::GGUF_IQ1_M,
             Q::GGUF_Q2_0,
             Q::GGUF_IQ3_XXS,
             Q::GGUF_IQ3_XXS,
             Q::GGUF_IQ4_NL,
             4,
             {1, 4, 16, 260, 820},
             kExperts},
            // The GSQ-RCO IQ3_S release's routed types on the matrix path: its IQ3_S layers, and
            // routes concentrated on a few experts (several columns per expert) in the IQ3_XXS
            // and IQ4_XS layers, with both of its down types.
            {"iq3_s layer",
             Q::GGUF_IQ3_S,
             Q::GGUF_IQ3_S,
             Q::GGUF_IQ4_NL,
             Q::GGUF_Q6_K,
             Q::GGUF_Q6_K,
             Q::GGUF_IQ4_NL,
             4,
             {1, 8, 9, 24, 72},
             kExperts},
            {"iq3_s mixed routing",
             Q::GGUF_IQ3_S,
             Q::GGUF_IQ3_S,
             Q::GGUF_IQ4_NL,
             Q::GGUF_Q6_K,
             Q::GGUF_Q6_K,
             Q::GGUF_IQ4_NL,
             4,
             {24, 64},
             30},
            {"iq3_xxs dense routing",
             Q::GGUF_IQ3_XXS,
             Q::GGUF_IQ3_XXS,
             Q::GGUF_Q2_0,
             Q::GGUF_Q4_K,
             Q::GGUF_Q4_K,
             Q::GGUF_IQ4_NL,
             2,
             {3, 24, 160},
             12},
            // DeepSeek-V4.1's DSpark drafter: MXFP4 experts beside a Q8_0 shared expert, on the
            // vector kernel (no matrix kernel takes MXFP4).
            {"mxfp4 bank",
             Q::GGUF_MXFP4,
             Q::GGUF_MXFP4,
             Q::GGUF_MXFP4,
             Q::GGUF_Q8_0,
             Q::GGUF_Q8_0,
             Q::GGUF_Q8_0,
             4,
             {1, 3, 5, 8, 9},
             kExperts},
            {"iq4_xs dense routing",
             Q::GGUF_IQ4_XS,
             Q::GGUF_IQ4_XS,
             Q::GGUF_IQ4_NL,
             Q::GGUF_Q5_K,
             Q::GGUF_Q5_K,
             Q::GGUF_IQ4_NL,
             2,
             {8, 24},
             12},
            // Edge0's expert formats: the NVFP4 builds' three projections, and the Q3_K builds'
            // Q3_K gate and up with IQ4_NL down, on both kernels and the CPU products.
            {"nvfp4 bank",
             Q::GGUF_NVFP4,
             Q::GGUF_NVFP4,
             Q::GGUF_NVFP4,
             Q::GGUF_Q8_0,
             Q::GGUF_Q8_0,
             Q::GGUF_Q8_0,
             4,
             {1, 3, 8, 9, 64, 260},
             kExperts},
            {"nvfp4 scaled bank",
             Q::GGUF_NVFP4,
             Q::GGUF_NVFP4,
             Q::GGUF_NVFP4,
             Q::GGUF_Q8_0,
             Q::GGUF_Q8_0,
             Q::GGUF_Q8_0,
             4,
             {1, 3, 8, 9, 24, 65},
             30,
             true},
            {"q3_k bank",
             Q::GGUF_Q3_K,
             Q::GGUF_Q3_K,
             Q::GGUF_IQ4_NL,
             Q::GGUF_Q8_0,
             Q::GGUF_Q8_0,
             Q::GGUF_Q8_0,
             4,
             {1, 8, 24, 260},
             kExperts},
        };
        int failures       = 0;
        std::uint32_t seed = 4242;
        for (const auto& c : cases) { failures += run_case(c, seed++); }
        failures += wide_call("iq3_xxs/iq4_nl", QType::GGUF_IQ3_XXS, QType::GGUF_IQ4_NL, seed++);
        failures += wide_call("mxfp4", QType::GGUF_MXFP4, QType::GGUF_MXFP4, seed++);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " MoE experts GGUF\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "MoE experts GGUF: " << error.what() << '\n';
        return 1;
    }
}
