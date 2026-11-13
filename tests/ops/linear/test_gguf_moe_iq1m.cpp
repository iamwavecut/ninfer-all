// IQ1_M routed experts at DeepSeek-V4.1-Flash's expert shapes (a [gate; up] parent of 2 x 2304 rows
// over 5120 inputs, a down projection of 5120 rows over 2304) through both routes of a call wider
// than eight tokens: the integer matrix kernel (quantize_moe_matrix_activation, moe_matrix_product)
// and the vector kernel in groups of eight columns (quantize_vector_activation, moe_vector_product).
// Each route is held against an FP64 product of the weights ggml's dequantizer decodes exactly and
// of the 8-bit activation the route defines for each 32 values: the matrix route's x * (127 /
// max|x|) rounded half away from zero, times 1 / (127 / max|x|) in FP32; the vector route's x /
// (max|x| / 127) rounded the same way, times that scale rounded to binary16. Only a kernel's FP32
// sums separate a route from its product. Both are also held, more loosely, against the product of
// the BF16 activation itself, the 8-bit activation's own error; the report prints the two routes'
// errors side by side. Off sm_70 only: there the experts take the FP16 route, which has no IQ1_M.

#include "core/device_buffer.h"
#include "ops/linear/gguf/ggml_bridge.h"

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
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gguf = ops::gguf;

constexpr int kHidden = 5120, kWidth = 2304, kExperts = 16, kTopK = 6;
constexpr int kBanks       = 3;  // expert e's matrices are bank e % kBanks's
constexpr int kSample      = 32; // one row in kSample is held against the oracle
constexpr int kBlockBytes  = 56, kBlockValues = 256;
constexpr std::size_t kTail = 4096;

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

// Random IQ1_M rows: random codes, the block's binary16 multiplier (spread over the top nibbles of
// its four scale words) finite.
std::vector<std::uint8_t> iq1m_rows(int rows, int k, std::mt19937& rng) {
    std::vector<std::uint8_t> bytes(std::size_t(rows) * (k / kBlockValues) * kBlockBytes);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    for (auto& b : bytes) { b = static_cast<std::uint8_t>(byte(rng)); }
    for (std::size_t at = 0; at < bytes.size(); at += kBlockBytes) {
        const std::uint16_t d = half_bits(scale(rng));
        for (int j = 0; j < 4; ++j) {
            std::uint16_t w = 0;
            std::memcpy(&w, &bytes[at + 48 + 2 * j], 2);
            w = static_cast<std::uint16_t>((w & 0x0FFF) | (((d >> (4 * j)) & 0xF) << 12));
            std::memcpy(&bytes[at + 48 + 2 * j], &w, 2);
        }
    }
    return bytes;
}

enum class Activation { Matrix, Vector, Exact };

// The values a route multiplies for one column of k values.
std::vector<double> activation(const float* x, int k, Activation kind) {
    std::vector<double> out(static_cast<std::size_t>(k));
    for (int b = 0; b < k; b += 32) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) { amax = std::max(amax, std::fabs(x[b + i])); }
        for (int i = 0; i < 32; ++i) {
            const float v = x[b + i];
            if (kind == Activation::Exact) {
                out[std::size_t(b + i)] = v;
            } else if (kind == Activation::Matrix) {
                const float d_inv       = 127.0f / amax;
                out[std::size_t(b + i)] = double(std::round(v * d_inv)) * double(1.0f / d_inv);
            } else {
                const float d           = amax / 127.0f;
                out[std::size_t(b + i)] = double(std::round(v / d)) *
                                          double(__half2float(__float2half(d)));
            }
        }
    }
    return out;
}

struct Error {
    double rms = 0.0, max = 0.0; // relative to the oracle's root mean square
};

Error compare(const std::vector<double>& got, const std::vector<double>& want) {
    double num = 0.0, den = 0.0, worst = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double e = got[i] - want[i];
        num += e * e;
        den += want[i] * want[i];
        worst = std::max(worst, std::fabs(e));
    }
    const double rms = std::sqrt(den / double(want.size()));
    return {std::sqrt(num / den), worst / rms};
}

int report(const std::string& what, const Error& e, double rms_bound, double max_bound) {
    const bool ok = std::isfinite(e.rms) && e.rms < rms_bound && e.max < max_bound;
    std::cout << (ok ? "OK   " : "FAIL ") << what << ": rms " << e.rms << ", max " << e.max
              << " (bounds " << rms_bound << ", " << max_bound << ")\n";
    return ok ? 0 : 1;
}

// One projection of every expert (rows x k), its call over `columns` activation columns (the
// tokens with column_group kTopK, the pairs with 1) and `pairs` routes.
int run(const char* name, int rows, int k, int tokens, int column_group, std::mt19937& rng) {
    const int pairs   = tokens * kTopK;
    const int columns = pairs / column_group;
    const std::int64_t row_bytes = std::int64_t(k / kBlockValues) * kBlockBytes;
    std::vector<std::vector<std::uint8_t>> host(kBanks);
    std::vector<DeviceBuffer> banks;
    for (int b = 0; b < kBanks; ++b) {
        host[std::size_t(b)] = iq1m_rows(rows, k, rng);
        banks.emplace_back(host[std::size_t(b)].size() + kTail);
        banks.back().fill(0);
        banks.back().copy_from_host(host[std::size_t(b)].data(), host[std::size_t(b)].size());
    }
    std::vector<const void*> table(kExperts);
    for (int e = 0; e < kExperts; ++e) { table[std::size_t(e)] = banks[std::size_t(e % kBanks)].p; }
    DeviceBuffer d_table(table.size() * sizeof(void*));
    d_table.copy_from_host(table.data(), table.size() * sizeof(void*));
    const gguf::MoeTable moe{static_cast<const void* const*>(d_table.p), row_bytes, rows, k};

    // The sampled rows of every bank, decoded exactly by ggml's dequantizer.
    std::vector<std::int32_t> sampled;
    for (int i = 0; i * kSample < rows; ++i) { sampled.push_back(i * kSample + i % kSample); }
    const int n = int(sampled.size());
    DeviceBuffer d_sampled(sampled.size() * 4), d_exact(std::size_t(n) * k * 4);
    d_sampled.copy_from_host(sampled.data(), sampled.size() * 4);
    std::vector<std::vector<float>> exact(kBanks, std::vector<float>(std::size_t(n) * k));
    for (int b = 0; b < kBanks; ++b) {
        gguf::dequantize_rows(gguf::GgmlType::IQ1_M, banks[std::size_t(b)].p, row_bytes, k,
                              static_cast<const std::int32_t*>(d_sampled.p), n,
                              static_cast<float*>(d_exact.p), k, nullptr);
        check(cudaDeviceSynchronize(), "dequantize");
        d_exact.copy_to_host(exact[std::size_t(b)].data(), exact[std::size_t(b)].size() * 4);
    }

    // Each token's kTopK distinct experts; the activation columns.
    std::vector<std::int32_t> ids(static_cast<std::size_t>(pairs));
    std::vector<int> order(kExperts);
    for (int t = 0; t < tokens; ++t) {
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        std::copy_n(order.begin(), kTopK, ids.begin() + std::ptrdiff_t(t) * kTopK);
    }
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16> x(std::size_t(columns) * k);
    std::vector<float> xf(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i]  = __float2bfloat16(normal(rng));
        xf[i] = __bfloat162float(x[i]);
    }
    DeviceBuffer d_ids(ids.size() * 4), d_x(x.size() * 2),
        routing_space(gguf::moe_routing_bytes(kExperts, pairs)),
        matrix_act(gguf::moe_matrix_activation_bytes(k, pairs)),
        vector_act(gguf::vector_activation_bytes(k, columns)),
        d_matrix(std::size_t(pairs) * rows * 4), d_vector(std::size_t(pairs) * rows * 4);
    d_ids.copy_from_host(ids.data(), ids.size() * 4);
    d_x.copy_from_host(x.data(), x.size() * 2);
    d_vector.fill(0);
    const auto* xb       = static_cast<const __nv_bfloat16*>(d_x.p);
    const auto routing   = gguf::moe_sort_routes(static_cast<const std::int32_t*>(d_ids.p), pairs,
                                                 kExperts, routing_space.p, nullptr);
    const int max_active = std::min(pairs, kExperts);
    gguf::quantize_moe_matrix_activation(gguf::GgmlType::IQ1_M, xb, k, routing, pairs, column_group,
                                         matrix_act.p, nullptr);
    gguf::moe_matrix_product(gguf::GgmlType::IQ1_M, moe, routing, max_active, pairs, true,
                             matrix_act.p, static_cast<float*>(d_matrix.p), nullptr);
    gguf::quantize_vector_activation(xb, k, columns, nullptr, vector_act.p, nullptr);
    gguf::MoeOutput out;
    out.f32 = static_cast<float*>(d_vector.p);
    gguf::moe_vector_product(gguf::GgmlType::IQ1_M, moe, routing, max_active, column_group, kTopK,
                             vector_act.p, columns, out, 8, nullptr);
    check(cudaDeviceSynchronize(), "products");
    std::vector<float> matrix(std::size_t(pairs) * rows), vector(matrix.size());
    d_matrix.copy_to_host(matrix.data(), matrix.size() * 4);
    d_vector.copy_to_host(vector.data(), vector.size() * 4);

    // The FP64 products at the sampled rows.
    std::vector<double> got_matrix, got_vector, want_matrix, want_vector, want_exact;
    std::vector<std::vector<double>> acts[3];
    for (int kind = 0; kind < 3; ++kind) {
        for (int c = 0; c < columns; ++c) {
            acts[kind].push_back(activation(xf.data() + std::size_t(c) * k, k, Activation(kind)));
        }
    }
    for (int p = 0; p < pairs; ++p) {
        const int column = p / column_group;
        const auto& w    = exact[std::size_t(ids[std::size_t(p)] % kBanks)];
        for (int s = 0; s < n; ++s) {
            const float* row = w.data() + std::size_t(s) * k;
            double sums[3]   = {0.0, 0.0, 0.0};
            for (int kind = 0; kind < 3; ++kind) {
                const auto& a = acts[kind][std::size_t(column)];
                for (int i = 0; i < k; ++i) { sums[kind] += double(row[i]) * a[std::size_t(i)]; }
            }
            const std::size_t at = std::size_t(p) * rows + std::size_t(sampled[std::size_t(s)]);
            got_matrix.push_back(matrix[at]);
            got_vector.push_back(vector[at]);
            want_matrix.push_back(sums[0]);
            want_vector.push_back(sums[1]);
            want_exact.push_back(sums[2]);
        }
    }
    const std::string what = std::string(name) + " (" + std::to_string(rows) + " x " +
                             std::to_string(k) + "), " + std::to_string(tokens) + " tokens";
    int failures = 0;
    failures += report(what + ", matrix route vs its activation", compare(got_matrix, want_matrix),
                       1e-5, 2e-4);
    failures += report(what + ", vector route vs its activation", compare(got_vector, want_vector),
                       1e-5, 2e-4);
    failures += report(what + ", matrix route vs the BF16 activation",
                       compare(got_matrix, want_exact), 1e-2, 8e-2);
    failures += report(what + ", vector route vs the BF16 activation",
                       compare(got_vector, want_exact), 1e-2, 8e-2);
    return failures;
}

} // namespace

int main() {
    try {
        int device = 0;
        check(cudaGetDevice(&device), "device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "device properties");
        if (properties.major * 10 + properties.minor == 70) {
            std::cout << "SKIP: sm_70 runs the routed experts' FP16 route, which has no IQ1_M\n";
            return 77;
        }
        if (!gguf::moe_matrix_supported(gguf::GgmlType::IQ1_M, 2 * kWidth, kHidden, true) ||
            !gguf::moe_matrix_supported(gguf::GgmlType::IQ1_M, kHidden, kWidth, true)) {
            std::cout << "FAIL: no IQ1_M matrix route at DeepSeek-V4.1's expert shapes\n";
            return 1;
        }
        std::mt19937 rng(20261104);
        int failures = 0;
        for (const int tokens : {12, 128}) {
            failures += run("gate_up", 2 * kWidth, kHidden, tokens, kTopK, rng);
            failures += run("down", kHidden, kWidth, tokens, 1, rng);
        }
        std::cout << (failures == 0 ? "PASS" : "FAIL") << '\n';
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
