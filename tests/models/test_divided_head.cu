// A vocabulary head divided over expert-parallel ranks (models::DividedHead) gives the logits one
// device's whole head gives, byte for byte: each rank's rows are the same GGUF vector products.
// Q8_0 and Q6_K heads whose row count is not a multiple of the 256-row unit, one to eight
// columns, even and uneven speeds, eagerly and replayed from a CUDA graph; two ranks on device 0,
// and on devices 0 and 1 where the host has two.

#include "core/device.h"
#include "core/device_buffer.h"
#include "core/qtype.h"
#include "models/divided_head.h"
#include "ops/linear/gguf/ggml_bridge.h"

#include "cuda_availability.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace {

namespace gguf = ninfer::ops::gguf;

int failures = 0;

void expect(bool condition, const char* label) {
    if (condition) { return; }
    std::cerr << "expectation failed: " << label << '\n';
    ++failures;
}

// Random stored blocks with finite FP16 scales: Q8_0 is [fp16 d][32 x int8]; Q6_K's 210 bytes
// end in its FP16 super-scale.
std::vector<std::uint8_t> random_rows(gguf::GgmlType type, int rows, int k, std::mt19937& rng) {
    const auto shape = gguf::block_shape(type);
    const std::size_t blocks = std::size_t(rows) * std::size_t(k / shape.elements);
    std::vector<std::uint8_t> bytes(blocks * std::size_t(shape.bytes));
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.001F, 0.02F);
    for (auto& b : bytes) { b = std::uint8_t(byte(rng)); }
    for (std::size_t i = 0; i < blocks; ++i) {
        const __half d = __float2half(scale(rng));
        std::uint8_t* block = bytes.data() + i * std::size_t(shape.bytes);
        std::uint8_t* at = type == gguf::GgmlType::Q8_0 ? block : block + shape.bytes - 2;
        std::memcpy(at, &d, 2);
    }
    return bytes;
}

void run_case(const ninfer::DeviceContext& device, gguf::GgmlType type, ninfer::QType qtype,
              int rows, int k, std::vector<double> speed, const char* label) {
    std::mt19937 rng(1234);
    const auto shape         = gguf::block_shape(type);
    const std::int64_t row_b = std::int64_t(k / shape.elements) * shape.bytes;
    const auto host_rows     = random_rows(type, rows, k, rng);
    const int columns        = gguf::kMaxVectorColumns;
    ninfer::RankBinding bind(device, 0);
    const cudaStream_t stream = device.rank(0).stream;
    ninfer::DeviceBuffer weights(host_rows.size());
    weights.copy_from_host(host_rows.data(), host_rows.size());
    std::vector<__nv_bfloat16> x(std::size_t(columns) * k);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (auto& v : x) { v = __float2bfloat16(normal(rng)); }
    ninfer::DeviceBuffer x_dev(x.size() * 2);
    x_dev.copy_from_host(x.data(), x.size() * 2);
    const std::size_t input_bytes = gguf::vector_activation_bytes(k, columns);
    ninfer::DeviceBuffer act(input_bytes);

    ninfer::Weight head;
    head.qtype   = qtype;
    head.qdata   = weights.p;
    head.payload = weights.p;
    head.n       = rows;
    head.k       = k;
    ninfer::models::DividedHead divided(device, head, row_b, speed, input_bytes, columns);
    std::vector<cudaStream_t> streams;
    for (std::size_t r = 0; r < device.size(); ++r) { streams.push_back(device.rank(r).stream); }

    ninfer::DeviceBuffer whole(std::size_t(columns) * rows * 2), parts(std::size_t(columns) * rows * 2);
    for (int t = 1; t <= columns; ++t) {
        gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(x_dev.p), k, t, nullptr,
                                         act.p, stream);
        gguf::VectorOutput o;
        o.bf16          = static_cast<__nv_bfloat16*>(whole.p);
        o.column_stride = rows;
        gguf::vector_product(type, weights.p, row_b, rows, k, act.p, t, o, stream);
        const auto project = [&](std::size_t, const void* input, const ninfer::Weight& piece, int,
                                 float*, __nv_bfloat16* out, cudaStream_t on) {
            gguf::VectorOutput p;
            p.bf16          = out;
            p.column_stride = piece.n;
            gguf::vector_product(type, piece.qdata, row_b, piece.n, k, input, t, p, on);
        };
        for (const bool graph : {false, true}) {
            CUDA_CHECK(cudaMemsetAsync(parts.p, 0, parts.bytes, stream));
            if (graph) {
                cudaGraph_t captured = nullptr;
                cudaGraphExec_t exec = nullptr;
                CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                divided.project(act.p, gguf::vector_activation_bytes(k, t), t,
                                static_cast<__nv_bfloat16*>(parts.p), streams, project);
                CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
                CUDA_CHECK(cudaGraphInstantiate(&exec, captured, 0));
                for (int replay = 0; replay < 3; ++replay) { CUDA_CHECK(cudaGraphLaunch(exec, stream)); }
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaGraphExecDestroy(exec));
                CUDA_CHECK(cudaGraphDestroy(captured));
            } else {
                divided.project(act.p, gguf::vector_activation_bytes(k, t), t,
                                static_cast<__nv_bfloat16*>(parts.p), streams, project);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<std::uint16_t> a(std::size_t(t) * rows), b(std::size_t(t) * rows);
            whole.copy_to_host(a.data(), a.size() * 2);
            parts.copy_to_host(b.data(), b.size() * 2);
            if (a != b) {
                std::cerr << label << ": columns " << t << (graph ? " (graph)" : "") << '\n';
                expect(false, "divided logits equal the whole head's");
            }
        }
    }
}

int run() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (ninfer::test::cuda_unavailable(status) || (status == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(status);
    const int one_card[2] = {0, 0};
    const ninfer::DeviceContext shared(std::span<const int>(one_card, 2));
    run_case(shared, gguf::GgmlType::Q8_0, ninfer::QType::GGUF_Q8_0, 4096 + 300, 512, {1, 1},
             "Q8_0 even");
    run_case(shared, gguf::GgmlType::Q6_K, ninfer::QType::GGUF_Q6_K, 8192 + 77, 1024, {3, 1},
             "Q6_K 3:1");
    if (count >= 2) {
        const int two_cards[2] = {0, 1};
        const ninfer::DeviceContext pair(std::span<const int>(two_cards, 2));
        run_case(pair, gguf::GgmlType::Q6_K, ninfer::QType::GGUF_Q6_K, 8192 + 77, 1024, {1, 1},
                 "Q6_K two cards");
    }
    if (failures != 0) { return 1; }
    std::cout << "divided head tests passed\n";
    return 0;
}

} // namespace

int main() { return run(); }
