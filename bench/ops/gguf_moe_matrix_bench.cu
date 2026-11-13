// A wide call's routed experts, by default at DeepSeek-V4.1-Flash's shapes (a [gate; up] parent of
// 2 x 2304 rows over 5120 inputs and a down projection of 5120 rows over 2304, six experts a token;
// Qwen3.8-Flash-Next's are --hidden 2560 --width 640 --top-k 10 --experts 512), on the two routes a
// call wider than eight tokens can take: the integer matrix kernel (quantize_moe_matrix_activation,
// moe_matrix_product) and the vector kernel in groups of eight columns (quantize_vector_activation,
// moe_vector_product). Each projection is timed alone, its activation's quantization included, over
// random routes of each --tokens count to distinct experts (the routes' sort is outside the timing).
// The bytes are the active experts' weights, each read once. The products themselves are checked by
// test_gguf_moe_iq1m (IQ1_M) and the models' tests.
//
// usage: ninfer_benches ninfer_gguf_moe_matrix_bench [--formats IQ1_M,IQ1_S,IQ2_XXS,IQ2_S|all]
//        [--tokens 64,512,2048] [--experts 384] [--hidden 5120] [--width 2304] [--top-k 6]

#include "core/device_buffer.h"
#include "ninfer_bench_common.h"
#include "ops/linear/gguf/ggml_bridge.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gguf = ops::gguf;

struct Shape {
    int hidden = 5120, width = 2304, top_k = 6;
};

struct Format {
    const char* name;
    gguf::GgmlType type;
};
// Every block type with both expert routes (MXFP4 has no matrix kernel).
constexpr Format kFormats[] = {
    {"Q4_0", gguf::GgmlType::Q4_0},       {"Q5_0", gguf::GgmlType::Q5_0},
    {"Q8_0", gguf::GgmlType::Q8_0},       {"Q2_0", gguf::GgmlType::Q2_0},
    {"Q2_K", gguf::GgmlType::Q2_K},       {"Q3_K", gguf::GgmlType::Q3_K},
    {"Q4_K", gguf::GgmlType::Q4_K},       {"Q5_K", gguf::GgmlType::Q5_K},
    {"Q6_K", gguf::GgmlType::Q6_K},       {"IQ1_S", gguf::GgmlType::IQ1_S},
    {"IQ1_M", gguf::GgmlType::IQ1_M},     {"IQ2_XXS", gguf::GgmlType::IQ2_XXS},
    {"IQ2_XS", gguf::GgmlType::IQ2_XS},   {"IQ2_S", gguf::GgmlType::IQ2_S},
    {"IQ3_XXS", gguf::GgmlType::IQ3_XXS}, {"IQ3_S", gguf::GgmlType::IQ3_S},
    {"IQ4_NL", gguf::GgmlType::IQ4_NL},   {"IQ4_XS", gguf::GgmlType::IQ4_XS},
};

std::vector<std::string> split(const std::string& text) {
    std::vector<std::string> out;
    std::stringstream stream(text);
    for (std::string item; std::getline(stream, item, ',');) {
        if (!item.empty()) { out.push_back(item); }
    }
    return out;
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

// Every expert's rows of random codes, each block's binary16 multiplier finite (IQ1_M's spread over
// the top nibbles of its four scale words, the others' the block's first two bytes), then zeros.
DeviceBuffer random_bank(gguf::GgmlType type, int experts, int rows, int k, std::mt19937& rng) {
    const auto shape = gguf::block_shape(type);
    const std::size_t row_bytes = std::size_t(k / shape.elements) * shape.bytes;
    std::vector<std::uint8_t> host(std::size_t(rows) * row_bytes);
    DeviceBuffer bank(host.size() * std::size_t(experts) + 4096);
    bank.fill(0);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    for (int e = 0; e < experts; ++e) {
        for (std::size_t i = 0; i + 4 <= host.size(); i += 4) {
            const std::uint32_t word = rng();
            std::memcpy(host.data() + i, &word, 4);
        }
        for (std::size_t at = 0; at < host.size(); at += std::size_t(shape.bytes)) {
            const std::uint16_t d = half_bits(scale(rng));
            if (type != gguf::GgmlType::IQ1_M) {
                std::memcpy(&host[at], &d, 2);
                continue;
            }
            for (int j = 0; j < 4; ++j) {
                std::uint16_t w = 0;
                std::memcpy(&w, &host[at + 48 + 2 * j], 2);
                w = static_cast<std::uint16_t>((w & 0x0FFF) | (((d >> (4 * j)) & 0xF) << 12));
                std::memcpy(&host[at + 48 + 2 * j], &w, 2);
            }
        }
        bank.copy_from_host(host.data(), host.size(), std::size_t(e) * host.size());
    }
    return bank;
}

struct Projection {
    const char* name;
    int rows, k, column_group;
};

void run(const Format& format, const Shape& shape_of, int experts, int tokens, std::mt19937& rng) {
    const int top_k      = shape_of.top_k;
    const int pairs      = tokens * top_k;
    const int max_active = std::min(pairs, experts);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(pairs));
    std::vector<int> order(static_cast<std::size_t>(experts));
    std::vector<char> active(std::size_t(experts), 0);
    for (int t = 0; t < tokens; ++t) {
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        for (int j = 0; j < top_k; ++j) {
            ids[std::size_t(t) * top_k + std::size_t(j)] = order[std::size_t(j)];
            active[std::size_t(order[std::size_t(j)])] = 1;
        }
    }
    const int active_experts = int(std::count(active.begin(), active.end(), char(1)));
    DeviceBuffer d_ids(ids.size() * 4), routing_space(gguf::moe_routing_bytes(experts, pairs));
    d_ids.copy_from_host(ids.data(), ids.size() * 4);
    const auto routing = gguf::moe_sort_routes(static_cast<const std::int32_t*>(d_ids.p), pairs,
                                               experts, routing_space.p, nullptr);
    for (const Projection& proj :
         {Projection{"gate_up", 2 * shape_of.width, shape_of.hidden, top_k},
          Projection{"down", shape_of.hidden, shape_of.width, 1}}) {
        if (!gguf::moe_matrix_supported(format.type, proj.rows, proj.k, true)) {
            std::printf("%-8s %-7s T=%-5d no matrix route on this device\n", format.name, proj.name,
                        tokens);
            continue;
        }
        const auto shape = gguf::block_shape(format.type);
        const std::int64_t row_bytes = std::int64_t(proj.k / shape.elements) * shape.bytes;
        DeviceBuffer bank = random_bank(format.type, experts, proj.rows, proj.k, rng);
        std::vector<const void*> table(static_cast<std::size_t>(experts));
        for (int e = 0; e < experts; ++e) {
            table[std::size_t(e)] = static_cast<const std::uint8_t*>(bank.p) +
                                    std::size_t(e) * std::size_t(proj.rows) * row_bytes;
        }
        DeviceBuffer d_table(table.size() * sizeof(void*));
        d_table.copy_from_host(table.data(), table.size() * sizeof(void*));
        const gguf::MoeTable moe{static_cast<const void* const*>(d_table.p), row_bytes, proj.rows,
                                 proj.k};
        const int columns = pairs / proj.column_group;
        DeviceBuffer x     = bench::make_bf16(std::size_t(columns) * proj.k, 0x1234567u);
        DeviceBuffer matrix_act(gguf::moe_matrix_activation_bytes(proj.k, pairs)),
            vector_act(gguf::vector_activation_bytes(proj.k, columns)),
            out(std::size_t(pairs) * proj.rows * 4);
        const auto* xb = static_cast<const __nv_bfloat16*>(x.p);
        const double bytes = double(active_experts) * double(proj.rows) * double(row_bytes);
        const auto matrix  = bench::bench_loop(
            [&](cudaStream_t stream) {
                gguf::quantize_moe_matrix_activation(format.type, xb, proj.k, routing, pairs,
                                                     proj.column_group, matrix_act.p, stream);
                gguf::moe_matrix_product(format.type, moe, routing, max_active, pairs, true,
                                         matrix_act.p, static_cast<float*>(out.p), stream);
            },
            bytes, 3, 20, 300);
        gguf::MoeOutput rows;
        rows.f32           = static_cast<float*>(out.p);
        const auto vector = bench::bench_loop(
            [&](cudaStream_t stream) {
                gguf::quantize_vector_activation(xb, proj.k, columns, nullptr, vector_act.p, stream);
                gguf::moe_vector_product(format.type, moe, routing, max_active, proj.column_group,
                                         top_k, vector_act.p, columns, rows, 8, stream);
            },
            bytes, 3, 20, 300);
        check(cudaDeviceSynchronize(), "products");
        std::printf("%-8s %-7s T=%-5d experts %3d: matrix %9.1f us (%6.1f GB/s), vector %9.1f us "
                    "(%6.1f GB/s), vector/matrix %.2fx\n",
                    format.name, proj.name, tokens, active_experts, matrix.median_us, matrix.gbs,
                    vector.median_us, vector.gbs, vector.median_us / matrix.median_us);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string> formats{"IQ1_M", "IQ1_S", "IQ2_XXS", "IQ2_S"};
        std::vector<int> tokens{64, 512, 2048};
        int experts = 384;
        Shape shape;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            const auto value      = [&]() -> std::string {
                if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
                return argv[++i];
            };
            if (arg == "--formats") {
                formats = split(value());
                if (formats.size() == 1 && formats[0] == "all") {
                    formats.clear();
                    for (const Format& f : kFormats) { formats.emplace_back(f.name); }
                }
            } else if (arg == "--tokens") {
                tokens.clear();
                for (const auto& t : split(value())) { tokens.push_back(std::stoi(t)); }
            } else if (arg == "--experts") {
                experts = std::stoi(value());
            } else if (arg == "--hidden") {
                shape.hidden = std::stoi(value());
            } else if (arg == "--width") {
                shape.width = std::stoi(value());
            } else if (arg == "--top-k") {
                shape.top_k = std::stoi(value());
            } else {
                throw std::invalid_argument("unknown argument " + arg);
            }
        }
        if (shape.top_k < 1 || experts < shape.top_k || shape.hidden < 1 || shape.width < 1) {
            throw std::invalid_argument("--top-k, --hidden and --width must be positive and "
                                        "--experts at least --top-k");
        }
        std::mt19937 rng(20261104);
        for (const auto& name : formats) {
            const Format* format = nullptr;
            for (const Format& f : kFormats) {
                if (name == f.name) { format = &f; }
            }
            if (format == nullptr) { throw std::invalid_argument("unknown format " + name); }
            for (const int t : tokens) { run(*format, shape, experts, t, rng); }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ninfer_gguf_moe_matrix_bench: %s\n", e.what());
        return 1;
    }
}
