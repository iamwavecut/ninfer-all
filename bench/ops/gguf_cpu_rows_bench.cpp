// Throughput of the GGUF CPU expert rows (ops::gguf_cpu_rows) on this host: random stored blocks of
// each format in a working set far larger than the caches, read once per pass in row order by one
// thread or by several that split the rows, for one to eight tokens. Reports rows a second, the
// stored bytes read a second and nanoseconds a row; the products themselves are checked by
// test_gguf_expert_cpu and test_moe_experts_gguf.
//
// usage: ninfer_benches ninfer_gguf_cpu_rows_bench [--formats IQ3_S,Q4_0,...] [--k 2560]
//        [--tokens 1,2,4,8] [--threads 1,8] [--megabytes 1024] [--seconds 2]
//        [--backend automatic|scalar|avx2]
#include "core/qtype.h"
#include "ops/moe_experts/gguf_expert_cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::ops;

struct Format {
    const char* name;
    QType type;
};
constexpr Format kFormats[] = {
    {"Q2_0", QType::GGUF_Q2_0},       {"Q4_0", QType::GGUF_Q4_0},       {"Q5_0", QType::GGUF_Q5_0},
    {"Q8_0", QType::GGUF_Q8_0},       {"Q4_K", QType::GGUF_Q4_K},       {"Q5_K", QType::GGUF_Q5_K},
    {"Q6_K", QType::GGUF_Q6_K},       {"IQ4_NL", QType::GGUF_IQ4_NL},   {"IQ1_S", QType::GGUF_IQ1_S},
    {"IQ1_M", QType::GGUF_IQ1_M},     {"IQ2_XXS", QType::GGUF_IQ2_XXS}, {"IQ2_S", QType::GGUF_IQ2_S},
    {"IQ3_XXS", QType::GGUF_IQ3_XXS}, {"IQ3_S", QType::GGUF_IQ3_S},     {"IQ4_XS", QType::GGUF_IQ4_XS},
};

std::vector<std::string> split(const std::string& text) {
    std::vector<std::string> out;
    std::stringstream stream(text);
    for (std::string item; std::getline(stream, item, ',');) {
        if (!item.empty()) { out.push_back(item); }
    }
    return out;
}

std::vector<int> split_ints(const std::string& text) {
    std::vector<int> out;
    for (const auto& item : split(text)) { out.push_back(std::stoi(item)); }
    return out;
}

// A positive normal binary16 near `value`.
std::uint16_t half_bits(float value) {
    std::uint32_t word;
    std::memcpy(&word, &value, 4);
    const int exponent = int((word >> 23) & 255) - 127 + 15;
    return static_cast<std::uint16_t>((exponent << 10) | ((word & 0x7fffff) >> 13));
}

// Random codes; each block's multipliers set to sizes whose products stay finite (every byte
// pattern is otherwise a valid block of these formats).
std::vector<std::uint8_t> random_rows(QType format, int k, std::size_t rows, std::mt19937& rng) {
    const auto [group, block] = gguf_block_shape(format);
    const std::size_t row_bytes = std::size_t(k / group) * block;
    std::vector<std::uint8_t> bytes(rows * row_bytes);
    for (std::size_t i = 0; i + 4 <= bytes.size(); i += 4) {
        const std::uint32_t word = rng();
        std::memcpy(bytes.data() + i, &word, 4);
    }
    const std::uint16_t d = half_bits(0.002F), dmin = half_bits(0.0004F);
    for (std::size_t b = 0; b < bytes.size() / block; ++b) {
        std::uint8_t* p = bytes.data() + b * block;
        if (format == QType::GGUF_IQ1_M) {
            for (int w = 0; w < 4; ++w) {
                std::uint16_t word;
                std::memcpy(&word, p + 48 + 2 * w, 2);
                word = std::uint16_t((word & 0x0FFF) | (((d >> (4 * w)) & 0xF) << 12));
                std::memcpy(p + 48 + 2 * w, &word, 2);
            }
        } else {
            std::memcpy(p + (format == QType::GGUF_Q6_K ? 208 : 0), &d, 2);
            if (format == QType::GGUF_Q4_K || format == QType::GGUF_Q5_K) { std::memcpy(p + 2, &dmin, 2); }
        }
    }
    return bytes;
}

GgufCpuBackend parse_backend(const std::string& name) {
    if (name == "automatic") { return GgufCpuBackend::Automatic; }
    if (name == "scalar") { return GgufCpuBackend::Scalar; }
    if (name == "avx2") { return GgufCpuBackend::Avx2; }
    throw std::invalid_argument("unknown backend " + name);
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string> formats = {"Q4_0", "Q4_K", "Q6_K", "IQ4_NL", "IQ4_XS", "IQ3_S",
                                            "IQ3_XXS", "IQ2_S", "IQ1_M", "IQ2_XXS"};
        std::vector<int> tokens_list = {1, 2, 4, 8}, threads_list = {1};
        int k = 2560;
        double megabytes = 1024, seconds = 2;
        std::string backend_name = "automatic";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            const auto value = [&]() -> std::string {
                if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
                return argv[++i];
            };
            if (arg == "--formats") { formats = split(value()); }
            else if (arg == "--tokens") { tokens_list = split_ints(value()); }
            else if (arg == "--threads") { threads_list = split_ints(value()); }
            else if (arg == "--k") { k = std::stoi(value()); }
            else if (arg == "--megabytes") { megabytes = std::stod(value()); }
            else if (arg == "--seconds") { seconds = std::stod(value()); }
            else if (arg == "--backend") { backend_name = value(); }
            else { throw std::invalid_argument("unknown argument " + arg); }
        }
        const GgufCpuBackend backend = gguf_cpu_backend_selected(parse_backend(backend_name));
        std::cout << "backend " << gguf_cpu_backend_name(backend) << ", k " << k << ", working set "
                  << megabytes << " MB\n";
        std::printf("%-8s %6s %7s %12s %10s %9s\n", "format", "tokens", "threads", "rows/s", "GB/s",
                    "ns/row");
        std::mt19937 rng(20261104);
        for (const auto& name : formats) {
            const Format* format = nullptr;
            for (const auto& f : kFormats) {
                if (name == f.name) { format = &f; }
            }
            if (format == nullptr) { throw std::invalid_argument("unknown format " + name); }
            if (!gguf_cpu_supported(format->type)) {
                std::cout << name << ": no CPU product\n";
                continue;
            }
            const auto [group, block] = gguf_block_shape(format->type);
            if (k % group != 0) {
                std::cout << name << ": k " << k << " is not a multiple of " << group << '\n';
                continue;
            }
            const std::int64_t row_bytes = std::int64_t(k / group) * block;
            const auto rows = std::size_t(std::max(1.0, megabytes * 1e6 / double(row_bytes)));
            const std::vector<std::uint8_t> bytes = random_rows(format->type, k, rows, rng);
            const GgufCpuMatrix w{format->type, bytes.data(), row_bytes, int(rows), k};
            for (const int tokens : tokens_list) {
                std::vector<std::uint16_t> x(std::size_t(tokens) * k);
                std::uniform_int_distribution<int> mantissa(0, 127);
                for (std::size_t i = 0; i < x.size(); ++i) {
                    // BF16 values in [-2, 2): sign, exponent 126..127, random mantissa.
                    x[i] = std::uint16_t((rng() & 1U) << 15 | (126U + (rng() & 1U)) << 7 |
                                         unsigned(mantissa(rng)));
                }
                GgufCpuActivation activation(tokens, k);
                activation.prepare_bf16(format->type, x.data(), std::size_t(k), tokens);
                for (const int threads : threads_list) {
                    // Each thread owns rows [rows t / n, rows (t + 1) / n) and writes its own outputs.
                    constexpr int kChunk = 64;
                    std::vector<std::vector<float>> outputs(
                        std::size_t(threads), std::vector<float>(std::size_t(tokens) * kChunk));
                    std::atomic<bool> stop{false};
                    std::vector<std::uint64_t> done(std::size_t(threads), 0);
                    std::vector<std::thread> workers;
                    const auto started = std::chrono::steady_clock::now();
                    for (int t = 0; t < threads; ++t) {
                        workers.emplace_back([&, t] {
                            const std::size_t begin = rows * t / threads, end = rows * (t + 1) / threads;
                            std::uint64_t count = 0;
                            while (!stop.load(std::memory_order_relaxed)) {
                                for (std::size_t r = begin; r < end && !stop.load(std::memory_order_relaxed);
                                     r += kChunk) {
                                    const int n = int(std::min(end - r, std::size_t(kChunk)));
                                    const GgufCpuMatrix piece{w.format, w.data + r * std::size_t(row_bytes),
                                                              row_bytes, n, k};
                                    gguf_cpu_rows(piece, activation, 0, n, outputs[t].data(),
                                                  std::size_t(kChunk), backend);
                                    count += std::uint64_t(n);
                                }
                            }
                            done[std::size_t(t)] = count;
                        });
                    }
                    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
                    stop.store(true);
                    for (auto& worker : workers) { worker.join(); }
                    const double elapsed =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                    std::uint64_t total = 0;
                    for (const auto count : done) { total += count; }
                    const double rows_per_second = double(total) / elapsed;
                    std::printf("%-8s %6d %7d %12.0f %10.2f %9.1f\n", name.c_str(), tokens, threads,
                                rows_per_second, rows_per_second * double(row_bytes) / 1e9,
                                1e9 * double(threads) / rows_per_second);
                    std::fflush(stdout);
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gguf_cpu_rows_bench: " << error.what() << '\n';
        return 1;
    }
}
