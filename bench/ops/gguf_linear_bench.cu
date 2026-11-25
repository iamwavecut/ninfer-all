// Dense GGUF products through the public Linear and LinearSwiGLU Ops (production dispatch: the
// vector kernels up to their width, the integer matrix kernel above), by default PTQ1_0 at the
// Qwen3.8-27B text tower's shapes: the [gate; up] parent of 2 x 17408 rows and the down product of
// 5120 rows over 17408, the GDN input rows (10240 and 6144 over 5120) and the 5120-row output
// products over 6144. Each point quantizes its activation inside the timing, as a decode or
// verification call does, and starts cold: a read of 256 MiB evicts L2 before each sample, as a
// decode step reads each layer's weights from DRAM. Bytes are the weight's, read once.
//
// usage: ninfer_benches ninfer_gguf_linear_bench [--format ptq1_0|q4_0|...] [--tokens 1,2,4,8,16,32]
//        [--shapes swiglu:17408x5120,linear:5120x17408,...]

#include "core/arena.h"
#include "core/device_buffer.h"
#include "core/qtype.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer_bench_common.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

struct Format {
    const char* name;
    QType qtype;
    int half_offset; // byte offset of the block's binary16 multiplier
};
constexpr Format kFormats[] = {
    {"ptq1_0", QType::GGUF_PTQ1_0, 26}, {"q2_0", QType::GGUF_Q2_0, 0},
    {"q4_0", QType::GGUF_Q4_0, 0},      {"q8_0", QType::GGUF_Q8_0, 0},
};

struct Shape {
    bool swiglu = false;
    int rows = 0, k = 0; // swiglu: rows of each of gate and up
};

std::vector<std::string> split(const std::string& text, char by) {
    std::vector<std::string> out;
    std::stringstream stream(text);
    for (std::string item; std::getline(stream, item, by);) {
        if (!item.empty()) { out.push_back(item); }
    }
    return out;
}

Shape parse_shape(const std::string& text) {
    const auto parts = split(text, ':');
    const auto dims  = parts.size() == 2 ? split(parts[1], 'x') : std::vector<std::string>{};
    if (parts.size() != 2 || dims.size() != 2 || (parts[0] != "linear" && parts[0] != "swiglu")) {
        throw std::invalid_argument("shape " + text + ": expected linear:RxK or swiglu:RxK");
    }
    return {parts[0] == "swiglu", std::stoi(dims[0]), std::stoi(dims[1])};
}

DeviceBuffer random_blocks(const Format& f, std::int64_t rows, int k, std::mt19937& rng) {
    const auto block = gguf_block_shape(f.qtype);
    const std::size_t bytes = std::size_t(rows) * (k / block.elements) * block.bytes;
    std::vector<std::uint8_t> host(bytes);
    for (std::size_t i = 0; i + 4 <= bytes; i += 4) {
        const std::uint32_t w = rng();
        std::memcpy(host.data() + i, &w, 4);
    }
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    for (std::size_t at = 0; at + block.bytes <= bytes; at += block.bytes) {
        const __half h = __float2half(scale(rng));
        std::memcpy(host.data() + at + f.half_offset, &h, 2);
    }
    DeviceBuffer d(bytes);
    d.copy_from_host(host.data(), bytes);
    return d;
}

Weight gguf_weight(QType qtype, const void* data, std::int32_t rows, std::int32_t k) {
    Weight w;
    w.qtype   = qtype;
    w.layout  = QuantLayout::GgufBlocks;
    w.qdata   = data;
    w.payload = data;
    w.n = w.shape[0] = w.padded_shape[0] = rows;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    w.ndim       = 2;
    w.group_size = static_cast<std::uint32_t>(gguf_block_shape(qtype).elements);
    w.group      = gguf_block_shape(qtype).elements;
    return w;
}

constexpr std::size_t kFlushBytes = 256ULL << 20;

void run(const Format& f, const Shape& s, const std::vector<int>& tokens, std::mt19937& rng,
         DeviceBuffer& flush) {
    const int max_t          = *std::max_element(tokens.begin(), tokens.end());
    const std::int64_t rows  = s.swiglu ? 2 * std::int64_t(s.rows) : s.rows;
    DeviceBuffer blocks      = random_blocks(f, rows, s.k, rng);
    const Weight w           = gguf_weight(f.qtype, blocks.p, std::int32_t(rows), s.k);
    DeviceBuffer x           = bench::make_bf16(std::size_t(s.k) * max_t, 0x1234567u);
    DeviceBuffer out(std::size_t(s.rows) * max_t * 2);
    const std::size_t capacity =
        s.swiglu ? ops::linear_swiglu_workspace_capacity_bytes(f.qtype, std::int32_t(rows), s.k,
                                                               ops::LinearPolicy::A16Only, 1, max_t)
                 : ops::linear_workspace_capacity_bytes(f.qtype, s.rows, s.k,
                                                        ops::LinearPolicy::A16Only, 1, max_t);
    WorkspaceArena workspace(capacity + (1u << 20));
    const auto block   = gguf_block_shape(f.qtype);
    const double bytes = double(rows) * (s.k / block.elements) * block.bytes;
    const int vector   = ops::gguf::vector_columns(ops::detail::gguf_type(f.qtype));
    for (const int t : tokens) {
        Tensor xt(x.p, DType::BF16, {s.k, t});
        Tensor ot(out.p, DType::BF16, {s.rows, t});
        const auto r = bench::measure_cold_launch(
            [&](cudaStream_t stream) {
                if (s.swiglu) {
                    ops::linear_swiglu(xt, w, ot, ops::LinearPolicy::A16Only, workspace, stream);
                } else {
                    ops::linear(xt, w, ot, ops::LinearPolicy::A16Only, workspace, stream);
                }
            },
            flush, nullptr, 3, 30);
        std::printf("%-7s %-6s %6dx%-6d T=%-3d %-6s %8.1f us %7.1f GB/s\n", f.name,
                    s.swiglu ? "swiglu" : "linear", s.rows, s.k, t, t <= vector ? "vector" : "matrix",
                    r.median_us, bytes / (r.median_us * 1e3));
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string format = "ptq1_0";
        std::vector<int> tokens{1, 2, 3, 4, 5, 8, 9, 12, 16, 20, 24, 28, 32, 40, 48, 64};
        std::vector<std::string> shapes{"swiglu:17408x5120", "linear:5120x17408",
                                        "linear:10240x5120", "linear:6144x5120",
                                        "linear:5120x6144"};
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            const auto value      = [&]() -> std::string {
                if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
                return argv[++i];
            };
            if (arg == "--format") {
                format = value();
            } else if (arg == "--tokens") {
                tokens.clear();
                for (const auto& t : split(value(), ',')) { tokens.push_back(std::stoi(t)); }
            } else if (arg == "--shapes") {
                shapes = split(value(), ',');
            } else {
                throw std::invalid_argument("unknown argument " + arg);
            }
        }
        const Format* f = nullptr;
        for (const Format& c : kFormats) {
            if (format == c.name) { f = &c; }
        }
        if (f == nullptr) { throw std::invalid_argument("unknown format " + format); }
        std::mt19937 rng(20261125);
        DeviceBuffer flush(kFlushBytes);
        for (const auto& s : shapes) { run(*f, parse_shape(s), tokens, rng, flush); }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ninfer_gguf_linear_bench: %s\n", e.what());
        return 1;
    }
}
