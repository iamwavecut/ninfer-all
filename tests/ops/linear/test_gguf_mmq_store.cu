#include "core/arena.h"
#include "core/device_buffer.h"
#include "core/qtype.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/common/device_route.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/ggml_bridge_mmq.cuh"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gguf = ninfer::ops::gguf;
namespace detail = ninfer::ops::detail;

struct TypeCase {
    QType qtype;
    gguf::GgmlType type;
    const char* name;
    std::vector<int> half_fields;
};

const TypeCase kCases[] = {
    {QType::GGUF_Q8_0, gguf::GgmlType::Q8_0, "q8_0", {0}},
    {QType::GGUF_Q2_K, gguf::GgmlType::Q2_K, "q2_k", {80, 82}},
    {QType::GGUF_Q3_K, gguf::GgmlType::Q3_K, "q3_k", {108}},
    {QType::GGUF_Q4_K, gguf::GgmlType::Q4_K, "q4_k", {0, 2}},
    {QType::GGUF_Q5_K, gguf::GgmlType::Q5_K, "q5_k", {0, 2}},
    {QType::GGUF_Q6_K, gguf::GgmlType::Q6_K, "q6_k", {208}},
    {QType::GGUF_IQ2_XXS, gguf::GgmlType::IQ2_XXS, "iq2_xxs", {0}},
    {QType::GGUF_IQ2_XS, gguf::GgmlType::IQ2_XS, "iq2_xs", {0}},
    {QType::GGUF_IQ2_S, gguf::GgmlType::IQ2_S, "iq2_s", {0}},
    {QType::GGUF_IQ3_XXS, gguf::GgmlType::IQ3_XXS, "iq3_xxs", {0}},
    {QType::GGUF_IQ3_S, gguf::GgmlType::IQ3_S, "iq3_s", {0}},
    {QType::GGUF_IQ1_S, gguf::GgmlType::IQ1_S, "iq1_s", {0}},
    {QType::GGUF_IQ4_NL, gguf::GgmlType::IQ4_NL, "iq4_nl", {0}},
    {QType::GGUF_IQ4_XS, gguf::GgmlType::IQ4_XS, "iq4_xs", {0}},
    {QType::GGUF_Q4_0, gguf::GgmlType::Q4_0, "q4_0", {0}},
    {QType::GGUF_Q5_0, gguf::GgmlType::Q5_0, "q5_0", {0}},
    {QType::GGUF_Q2_0, gguf::GgmlType::Q2_0, "q2_0", {0}},
};

// IQ1_M has the routed-expert matrix kernel only, so it is not among the dense cases above.
const TypeCase kIq1m = {QType::GGUF_IQ1_M, gguf::GgmlType::IQ1_M, "iq1_m", {}};
// NVFP4 (four UE4M3 scales, then 32 bytes of E2M1 pairs a 64-value block) runs the async
// comparisons only.
const TypeCase kNvfp4 = {QType::GGUF_NVFP4, gguf::GgmlType::NVFP4, "nvfp4", {}};

void check(cudaError_t status, const char* label) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
    }
}

std::uint16_t half_bits(float value) {
    const __half converted = __float2half(value);
    std::uint16_t bits = 0;
    std::memcpy(&bits, &converted, sizeof(bits));
    return bits;
}

std::vector<std::uint8_t> make_weight(const TypeCase& c, int rows, int k, std::uint32_t seed) {
    const gguf::BlockShape block = gguf::block_shape(c.type);
    const std::size_t count = std::size_t(rows) * (k / block.elements);
    std::vector<std::uint8_t> bytes(count * block.bytes);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    for (auto& value : bytes) {
        value = static_cast<std::uint8_t>(byte(rng));
    }
    for (std::size_t i = 0; i < count; ++i) {
        std::uint8_t* base = bytes.data() + i * block.bytes;
        for (const int offset : c.half_fields) {
            const std::uint16_t bits = half_bits(scale(rng));
            std::memcpy(base + offset, &bits, sizeof(bits));
        }
        if (c.type == gguf::GgmlType::NVFP4) {
            // Finite UE4M3 scales from 0.125 to 0.234375 (exponent field 4).
            for (int s = 0; s < 4; ++s) {
                base[s] = static_cast<std::uint8_t>(0x20 | (byte(rng) & 0x07));
            }
        }
        if (c.type == gguf::GgmlType::IQ1_M) {
            // The FP16 scale is spread over the top nibbles of the four scale words (bytes 48-55).
            const std::uint16_t bits = half_bits(scale(rng));
            for (int w = 0; w < 4; ++w) {
                std::uint8_t& high = base[48 + 2 * w + 1];
                high = static_cast<std::uint8_t>((high & 0x0f) | (((bits >> (4 * w)) & 0xf) << 4));
            }
        }
    }
    return bytes;
}

int rows_for_fixup(const TypeCase& c, int columns, bool wanted) {
    const int sm_count = gguf::multiprocessor_count();
    const int max_rows = 128 * sm_count * 4;
    for (int rows = 128; rows <= max_rows; rows += 128) {
        const bool has_fixup = gguf::matrix_fixup_bytes(c.type, rows, columns) != 0;
        if (has_fixup == wanted) {
            return rows;
        }
    }
    return 0;
}

int padded_tail_k(const TypeCase& c) {
    const int qk = gguf::block_shape(c.type).elements;
    for (int k = qk * 9; k <= qk * 16; k += qk) {
        if (k % 256 != 0) {
            return k;
        }
    }
    return 0;
}

void check_rejected_geometry(const TypeCase& c) {
    const auto reject = [](auto call, const char* message) {
        try {
            call();
        } catch (const std::invalid_argument& error) {
            if (std::string(error.what()).find(message) != std::string::npos) { return; }
            throw;
        }
        throw std::runtime_error("GGUF MMQ accepted unsupported geometry");
    };
    for (const int value : {0, -128}) {
        reject([&] { (void)gguf::matrix_fixup_bytes(c.type, value, 16); }, "nonpositive geometry");
        reject([&] { (void)gguf::matrix_fixup_bytes(c.type, 128, value); }, "nonpositive geometry");
    }
    reject([&] { (void)gguf::matrix_fixup_bytes(c.type, INT_MAX, INT_MAX); }, "index range exceeded");
    (void)gguf::matrix_fixup_bytes(c.type, 128, INT_MAX);
    const auto block = gguf::block_shape(c.type);
    const int k = INT_MAX - 255;
    const std::int64_t row_bytes = std::int64_t(k / block.elements) * block.bytes;
    reject([&] {
        gguf::matrix_product(c.type, nullptr, row_bytes, 128, k, nullptr, 65536,
                             nullptr, 128, nullptr, nullptr);
    }, "index range exceeded");
    for (const bool accumulate : {false, true}) {
        reject([&] {
            gguf::matrix_product_bf16(c.type, nullptr, row_bytes, 128, k, nullptr, 65536,
                                      nullptr, 128, nullptr, nullptr, nullptr, accumulate);
        }, "index range exceeded");
    }
    for (const int invalid_k : {0, -256}) {
        reject([&] {
            gguf::matrix_product(c.type, nullptr, 0, 128, invalid_k, nullptr, 16,
                                 nullptr, 128, nullptr, nullptr);
        }, "unsupported geometry");
    }
}

void compare_outputs(const TypeCase& c, int k, int columns, int rows, int padding,
                     bool accumulate, const std::vector<__nv_bfloat16>& initial,
                     const std::vector<__nv_bfloat16>& expected,
                     const std::vector<__nv_bfloat16>& actual) {
    const int output_rows = rows + 2 * padding;
    for (int column = 0; column < columns; ++column) {
        for (int row = 0; row < output_rows; ++row) {
            const std::size_t at = std::size_t(column) * output_rows + row;
            std::uint16_t expected_bits = 0;
            std::uint16_t actual_bits = 0;
            std::memcpy(&expected_bits, &expected[at], sizeof(expected_bits));
            std::memcpy(&actual_bits, &actual[at], sizeof(actual_bits));
            if (expected_bits != actual_bits) {
                throw std::runtime_error(std::string(c.name) + " K=" + std::to_string(k) +
                    " T=" + std::to_string(columns) + " row=" + std::to_string(row) +
                    " accumulate=" + std::to_string(accumulate) + " differs from native FP32 route");
            }
            if ((row < padding || row >= padding + rows) &&
                std::memcmp(&actual[at], &initial[at], sizeof(std::uint16_t)) != 0) {
                throw std::runtime_error(std::string(c.name) + " modified output guard rows");
            }
        }
    }
}

void run_direct_case(const TypeCase& c, int k, int columns, int rows, int padding,
                     bool accumulate, bool require_fixup, bool tail = false) {
    const gguf::BlockShape block = gguf::block_shape(c.type);
    if (rows % 128 != 0 || k % block.elements != 0) {
        throw std::runtime_error(std::string(c.name) + " has invalid test geometry");
    }
    const std::size_t fixup_bytes = gguf::matrix_fixup_bytes(c.type, rows, columns);
    if ((fixup_bytes != 0) != require_fixup) {
        throw std::runtime_error(std::string(c.name) + " did not select the requested fixup geometry");
    }

    const int output_rows = rows + 2 * padding;
    const std::int64_t row_bytes = std::int64_t(k / block.elements) * block.bytes;
    const auto host_weight = make_weight(c, rows, k, 0xE190u + static_cast<unsigned>(c.qtype));
    DeviceBuffer weight(host_weight.size() + (tail ? 4096 : 0));
    weight.copy_from_host(host_weight.data(), host_weight.size());
    if (tail) {
        check(cudaMemset(static_cast<char*>(weight.p) + host_weight.size(), 0, 4096),
              "padded matrix weight tail");
    }

    std::mt19937 rng(0xA119u + static_cast<unsigned>(c.qtype) +
                     static_cast<unsigned>(k + columns + rows));
    std::uniform_real_distribution<float> sample(-0.25f, 0.25f);
    std::vector<__nv_bfloat16> host_input(std::size_t(k) * columns);
    std::vector<__nv_bfloat16> initial(std::size_t(output_rows) * columns);
    for (auto& value : host_input) { value = __float2bfloat16(sample(rng)); }
    for (auto& value : initial) { value = __float2bfloat16(sample(rng)); }

    DeviceBuffer input(host_input.size() * sizeof(__nv_bfloat16));
    input.copy_from_host(host_input.data(), input.bytes);
    const int activation_k = tail ? gguf::matrix_padded_k(k) : k;
    DeviceBuffer activation(gguf::matrix_activation_bytes(activation_k, columns));
    if (tail) {
        gguf::quantize_matrix_activation_padded(c.type,
            static_cast<const __nv_bfloat16*>(input.p), k, columns, nullptr, activation.p, nullptr);
    } else {
        gguf::quantize_matrix_activation(c.type, static_cast<const __nv_bfloat16*>(input.p), k,
                                         columns, nullptr, activation.p, nullptr);
    }

    DeviceBuffer native_plane(std::size_t(rows) * columns * sizeof(float));
    DeviceBuffer partial(std::size_t(rows) * columns * sizeof(float));
    DeviceBuffer fixup(std::max<std::size_t>(fixup_bytes, 1));
    DeviceBuffer native_output(initial.size() * sizeof(__nv_bfloat16));
    DeviceBuffer bf16_output(initial.size() * sizeof(__nv_bfloat16));
    native_output.copy_from_host(initial.data(), native_output.bytes);
    bf16_output.copy_from_host(initial.data(), bf16_output.bytes);

    gguf::matrix_product(c.type, weight.p, row_bytes, rows, k, activation.p, columns,
                         static_cast<float*>(native_plane.p), rows, fixup.p, nullptr, tail);
    gguf::VectorOutput destination;
    destination.bf16 = static_cast<__nv_bfloat16*>(native_output.p) + padding;
    destination.column_stride = output_rows;
    destination.epilogue = accumulate ? gguf::Epilogue::Accumulate : gguf::Epilogue::Store;
    gguf::store_plane(static_cast<const float*>(native_plane.p), rows, rows, columns,
                      destination, nullptr);

    gguf::matrix_product_bf16(c.type, weight.p, row_bytes, rows, k, activation.p, columns,
                              static_cast<__nv_bfloat16*>(bf16_output.p) + padding,
                              output_rows, static_cast<float*>(partial.p), fixup.p, nullptr,
                              accumulate, tail);
    check(cudaDeviceSynchronize(), "GGUF MMQ BF16 store synchronization");

    std::vector<__nv_bfloat16> expected(initial.size());
    std::vector<__nv_bfloat16> actual(initial.size());
    native_output.copy_to_host(expected.data(), native_output.bytes);
    bf16_output.copy_to_host(actual.data(), bf16_output.bytes);
    compare_outputs(c, k, columns, rows, padding, accumulate, initial, expected, actual);
}

void run_public_gather_case(const TypeCase& c, bool accumulate) {
    constexpr int k = 512;
    constexpr int rows = 128;
    constexpr int columns = 17;
    constexpr int padding = 7;
    const gguf::BlockShape block = gguf::block_shape(c.type);
    const std::int64_t row_bytes = std::int64_t(k / block.elements) * block.bytes;
    const int output_rows = rows + 2 * padding;
    const auto host_weight = make_weight(c, rows, k, 0xB410u + static_cast<unsigned>(c.qtype));
    DeviceBuffer weight(host_weight.size());
    weight.copy_from_host(host_weight.data(), host_weight.size());

    std::mt19937 rng(0xC318u + static_cast<unsigned>(c.qtype));
    std::uniform_real_distribution<float> sample(-0.25f, 0.25f);
    std::vector<__nv_bfloat16> host_input(std::size_t(k) * columns);
    std::vector<__nv_bfloat16> initial(std::size_t(output_rows) * columns);
    std::vector<std::int32_t> host_columns(k);
    for (int i = 0; i < k; ++i) { host_columns[i] = (i * 7) % k; }
    for (auto& value : host_input) { value = __float2bfloat16(sample(rng)); }
    for (auto& value : initial) { value = __float2bfloat16(sample(rng)); }

    DeviceBuffer input(host_input.size() * sizeof(__nv_bfloat16));
    input.copy_from_host(host_input.data(), input.bytes);
    DeviceBuffer columns_device(host_columns.size() * sizeof(std::int32_t));
    columns_device.copy_from_host(host_columns.data(), columns_device.bytes);
    DeviceBuffer activation(gguf::matrix_activation_bytes(k, columns));
    gguf::quantize_matrix_activation(c.type, static_cast<const __nv_bfloat16*>(input.p), k,
                                     columns, static_cast<const std::int32_t*>(columns_device.p),
                                     activation.p, nullptr);

    DeviceBuffer native_plane(std::size_t(rows) * columns * sizeof(float));
    DeviceBuffer fixup(std::max<std::size_t>(
        gguf::matrix_fixup_bytes(c.type, rows, columns), 1));
    DeviceBuffer native_output(initial.size() * sizeof(__nv_bfloat16));
    DeviceBuffer public_output(initial.size() * sizeof(__nv_bfloat16));
    native_output.copy_from_host(initial.data(), native_output.bytes);
    public_output.copy_from_host(initial.data(), public_output.bytes);
    gguf::matrix_product(c.type, weight.p, row_bytes, rows, k, activation.p, columns,
                         static_cast<float*>(native_plane.p), rows, fixup.p, nullptr);
    gguf::VectorOutput destination;
    destination.bf16 = static_cast<__nv_bfloat16*>(native_output.p) + padding;
    destination.column_stride = output_rows;
    destination.epilogue = accumulate ? gguf::Epilogue::Accumulate : gguf::Epilogue::Store;
    gguf::store_plane(static_cast<const float*>(native_plane.p), rows, rows, columns,
                      destination, nullptr);

    Tensor x(input.p, DType::BF16, {k, columns});
    Tensor y(public_output.p, DType::BF16, {output_rows, columns});
    Weight w;
    w.qtype = c.qtype;
    w.layout = QuantLayout::GgufBlocks;
    w.qdata = weight.p;
    w.payload = weight.p;
    w.n = w.shape[0] = w.padded_shape[0] = rows;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    w.ndim = 2;
    w.group_size = static_cast<std::uint32_t>(block.elements);
    w.group = block.elements;
    w.input_columns = static_cast<const std::int32_t*>(columns_device.p);
    const detail::GgufProduct product{
        &w, &y, padding,
        accumulate ? detail::GgufEpilogue::Accumulate : detail::GgufEpilogue::Store};
    const detail::GgufShape shape{c.qtype, rows, k};
    WorkspaceArena workspace(detail::gguf_project_workspace_bytes({&shape, 1}, columns, columns) +
                             (1U << 20));
    detail::gguf_project(x, {&product, 1}, workspace, nullptr);
    check(cudaDeviceSynchronize(), "public GGUF project synchronization");

    std::vector<__nv_bfloat16> expected(initial.size());
    std::vector<__nv_bfloat16> actual(initial.size());
    native_output.copy_to_host(expected.data(), native_output.bytes);
    public_output.copy_to_host(actual.data(), public_output.bytes);
    compare_outputs(c, k, columns, rows, padding, accumulate, initial, expected, actual);
}

// The formats with the asynchronous tile loop, their device-route keys and launch choices: the
// staging copy's width for rows of `row_bytes` (0 for the compiled loop).
template <ggml_type T>
int dense_async_offered(int rows, int columns, std::int64_t row_bytes) {
    return gguf::detail::select_matrix_launch<T>(
               rows, columns, gguf::detail::async_copy_width<T>(row_bytes, 0))
        .async;
}
template <ggml_type T>
int moe_async_offered(int rows, int k, int columns, std::int64_t row_bytes) {
    return gguf::detail::select_moe_launch<T>(rows, k, columns, true,
                                              gguf::detail::async_copy_width<T>(row_bytes, 0))
        .async;
}
struct AsyncCase {
    gguf::GgmlType type;
    const char* dense_key;
    const char* moe_key;
    int (*dense)(int, int, std::int64_t);
    int (*moe)(int, int, int, std::int64_t);
};
const AsyncCase kAsyncCases[] = {
    {gguf::GgmlType::Q4_K, "gguf_matrix/q4_k", "gguf_moe_matrix/q4_k",
     dense_async_offered<GGML_TYPE_Q4_K>, moe_async_offered<GGML_TYPE_Q4_K>},
    {gguf::GgmlType::Q5_K, "gguf_matrix/q5_k", "gguf_moe_matrix/q5_k",
     dense_async_offered<GGML_TYPE_Q5_K>, moe_async_offered<GGML_TYPE_Q5_K>},
    {gguf::GgmlType::Q4_0, "gguf_matrix/q4_0", "gguf_moe_matrix/q4_0",
     dense_async_offered<GGML_TYPE_Q4_0>, moe_async_offered<GGML_TYPE_Q4_0>},
    {gguf::GgmlType::Q5_0, "gguf_matrix/q5_0", "gguf_moe_matrix/q5_0",
     dense_async_offered<GGML_TYPE_Q5_0>, moe_async_offered<GGML_TYPE_Q5_0>},
    {gguf::GgmlType::Q8_0, "gguf_matrix/q8_0", "gguf_moe_matrix/q8_0",
     dense_async_offered<GGML_TYPE_Q8_0>, moe_async_offered<GGML_TYPE_Q8_0>},
    {gguf::GgmlType::IQ4_NL, "gguf_matrix/iq4_nl", "gguf_moe_matrix/iq4_nl",
     dense_async_offered<GGML_TYPE_IQ4_NL>, moe_async_offered<GGML_TYPE_IQ4_NL>},
    // Steps of whole 8-byte words: staged in 8-byte copies.
    {gguf::GgmlType::Q2_0, "gguf_matrix/q2_0", "gguf_moe_matrix/q2_0",
     dense_async_offered<GGML_TYPE_Q2_0>, moe_async_offered<GGML_TYPE_Q2_0>},
    {gguf::GgmlType::IQ4_XS, "gguf_matrix/iq4_xs", "gguf_moe_matrix/iq4_xs",
     dense_async_offered<GGML_TYPE_IQ4_XS>, moe_async_offered<GGML_TYPE_IQ4_XS>},
    {gguf::GgmlType::IQ1_M, nullptr, "gguf_moe_matrix/iq1_m", nullptr,
     moe_async_offered<GGML_TYPE_IQ1_M>},
    {gguf::GgmlType::NVFP4, "gguf_matrix/nvfp4", "gguf_moe_matrix/nvfp4",
     dense_async_offered<GGML_TYPE_NVFP4>, moe_async_offered<GGML_TYPE_NVFP4>},
};

std::int64_t row_bytes_of(const TypeCase& c, int k) {
    const auto block = ninfer::gguf_block_shape(c.qtype);
    return std::int64_t(k / block.elements) * block.bytes;
}

struct RouteRun {
    std::vector<std::vector<std::uint8_t>> parts;
    int async = 0;
    std::size_t fixup_bytes = 0;
};

std::vector<std::uint8_t> device_bytes(const void* device, std::size_t bytes) {
    std::vector<std::uint8_t> host(bytes);
    check(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost), "async route copy");
    return host;
}

RouteRun run_route(const char* schedule, const TypeCase& c, const AsyncCase& a, int rows, int k,
                   int columns, const DeviceBuffer& weight, const DeviceBuffer& activation,
                   const std::vector<__nv_bfloat16>& initial, bool tail) {
    const ninfer::ops::DeviceRouteForce force(a.dense_key, schedule);
    RouteRun run;
    run.fixup_bytes = gguf::matrix_fixup_bytes(c.type, rows, columns);
    const std::int64_t row_bytes = row_bytes_of(c, k);
    run.async = a.dense(rows, columns, row_bytes);
    const std::size_t plane_bytes = std::size_t(rows) * columns * sizeof(float);
    const std::size_t fixup_bytes = std::max<std::size_t>(run.fixup_bytes, 1);
    DeviceBuffer plane(plane_bytes);
    DeviceBuffer partial(plane_bytes);
    DeviceBuffer fixup(fixup_bytes);
    DeviceBuffer stored(initial.size() * sizeof(__nv_bfloat16));
    DeviceBuffer accumulated(initial.size() * sizeof(__nv_bfloat16));
    check(cudaMemset(plane.p, 0xA5, plane_bytes), "async route plane");
    check(cudaMemset(partial.p, 0xA5, plane_bytes), "async route partial");
    check(cudaMemset(fixup.p, 0xA5, fixup_bytes), "async route fixup");
    stored.copy_from_host(initial.data(), stored.bytes);
    accumulated.copy_from_host(initial.data(), accumulated.bytes);

    gguf::matrix_product(c.type, weight.p, row_bytes, rows, k, activation.p, columns,
                         static_cast<float*>(plane.p), rows, fixup.p, nullptr, tail);
    check(cudaDeviceSynchronize(), "async route float product");
    run.parts.push_back(device_bytes(plane.p, plane_bytes));
    run.parts.push_back(device_bytes(fixup.p, fixup_bytes));

    for (const bool accumulate : {false, true}) {
        DeviceBuffer& destination = accumulate ? accumulated : stored;
        gguf::matrix_product_bf16(c.type, weight.p, row_bytes, rows, k, activation.p, columns,
                                  static_cast<__nv_bfloat16*>(destination.p), rows,
                                  static_cast<float*>(partial.p), fixup.p, nullptr, accumulate,
                                  tail);
        check(cudaDeviceSynchronize(), "async route BF16 product");
        run.parts.push_back(device_bytes(destination.p, destination.bytes));
        run.parts.push_back(device_bytes(partial.p, plane_bytes));
        run.parts.push_back(device_bytes(fixup.p, fixup_bytes));
    }
    return run;
}

// Eight experts of 640 rows in one bank followed by zeros (a row's last 256-value step reads into
// the next row, past the last one into the zeros); `expect_copy` is the async route's staging copy.
void compare_moe_routes(const TypeCase& c, const AsyncCase& a, int pairs, int expect_copy,
                        int k = 2560) {
    constexpr int rows = 640;
    constexpr int experts = 8;
    constexpr int per_token = 2;
    const std::int64_t row_bytes = row_bytes_of(c, k);
    const std::size_t expert_bytes = std::size_t(rows) * std::size_t(row_bytes);
    DeviceBuffer bank(experts * expert_bytes + 4096);
    check(cudaMemset(bank.p, 0, bank.bytes), "moe route bank");
    std::vector<const void*> pointers;
    for (int expert = 0; expert < experts; ++expert) {
        const auto host = make_weight(c, rows, k, 0x7C11u + static_cast<unsigned>(expert));
        auto* base = static_cast<char*>(bank.p) + expert * expert_bytes;
        check(cudaMemcpy(base, host.data(), host.size(), cudaMemcpyHostToDevice),
              "moe route expert");
        pointers.push_back(base);
    }
    DeviceBuffer table_storage(experts * sizeof(const void*));
    table_storage.copy_from_host(pointers.data(), table_storage.bytes);
    std::vector<std::int32_t> ids(pairs);
    for (int p = 0; p < pairs; ++p) { ids[p] = (p * 5 + p / experts) % experts; }
    DeviceBuffer ids_device(ids.size() * sizeof(std::int32_t));
    ids_device.copy_from_host(ids.data(), ids_device.bytes);
    DeviceBuffer routing_workspace(gguf::moe_routing_bytes(experts, pairs));
    const gguf::MoeRouting routing = gguf::moe_sort_routes(
        static_cast<const std::int32_t*>(ids_device.p), pairs, experts, routing_workspace.p,
        nullptr);
    std::mt19937 rng(0x4D0Eu + static_cast<unsigned>(pairs));
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16> input_host(static_cast<std::size_t>(k) * (pairs / per_token));
    for (auto& v : input_host) { v = __float2bfloat16(normal(rng)); }
    DeviceBuffer input(input_host.size() * sizeof(__nv_bfloat16));
    input.copy_from_host(input_host.data(), input.bytes);
    DeviceBuffer activation(gguf::moe_matrix_activation_bytes(k, pairs));
    gguf::quantize_moe_matrix_activation(c.type, static_cast<const __nv_bfloat16*>(input.p), k,
                                         routing, pairs, per_token, activation.p, nullptr);
    check(cudaDeviceSynchronize(), "moe route activation");
    const std::size_t plane_bytes = std::size_t(rows) * pairs * sizeof(float);
    std::vector<std::uint8_t> results[2];
    int offered[2] = {0, 0};
    int index = 0;
    for (const char* schedule : {"", "async"}) {
        const ninfer::ops::DeviceRouteForce force(a.moe_key, schedule);
        offered[index] = a.moe(rows, k, pairs / experts, row_bytes);
        DeviceBuffer plane(plane_bytes);
        check(cudaMemset(plane.p, 0xA5, plane_bytes), "moe route plane");
        const gguf::MoeTable table{reinterpret_cast<const void* const*>(table_storage.p), row_bytes,
                                   rows, k, true};
        gguf::moe_matrix_product(c.type, table, routing, experts, pairs, true, activation.p,
                                 static_cast<float*>(plane.p), nullptr);
        check(cudaDeviceSynchronize(), "moe route product");
        results[index++] = device_bytes(plane.p, plane_bytes);
    }
    const std::string label = std::string(c.name) + " K=" + std::to_string(k) +
                              " pairs=" + std::to_string(pairs);
    if (offered[0]) {
        throw std::runtime_error(label + ": the compiled MoE route took the async tiles");
    }
    if (offered[1] != expect_copy) {
        throw std::runtime_error(label + ": the async MoE route copies " +
                                 std::to_string(offered[1]) + " bytes, expected " +
                                 std::to_string(expect_copy));
    }
    if (results[0] != results[1]) {
        throw std::runtime_error(label +
                                 ": the MoE plane differs between the compiled and async routes");
    }
    std::cout << "OK GGUF MoE MMQ async " << label << " copy=" << offered[1] << '\n';
}

// The column tile a product of `columns` columns (or pairs an expert) takes: the narrowest that
// holds them, at most 128.
int column_tile(int columns) {
    return columns <= 16 ? 16 : columns <= 32 ? 32 : columns <= 64 ? 64 : 128;
}

// Whether the device offers the async loop at each column tile under a forced `async` schedule. A
// narrower tile stages less shared memory, so it is offered wherever a wider one is.
template <typename Offer>
std::map<int, int> offered_tiles(const TypeCase& c, const char* key, Offer offer) {
    std::map<int, int> offered;
    const ninfer::ops::DeviceRouteForce force(key, "async");
    for (const int tile : {16, 32, 64, 128}) { offered[tile] = offer(tile); }
    for (const int tile : {16, 32, 64}) {
        if (offered[2 * tile] && !offered[tile]) {
            throw std::runtime_error(std::string(c.name) + " " + key + ": async tiles offered at " +
                                     std::to_string(2 * tile) + " columns but not at " +
                                     std::to_string(tile));
        }
    }
    return offered;
}

// Eight experts with 8 to 1,024 pairs each, every column tile at a full and a partial width.
void compare_moe_widths(const TypeCase& c, const AsyncCase& a) {
    const auto offered =
        offered_tiles(c, a.moe_key, [&](int columns) { return a.moe(640, 2560, columns, 0); });
    for (const int pairs : {64, 128, 192, 256, 384, 512, 1024, 4096, 8192}) {
        compare_moe_routes(c, a, pairs, offered.at(column_tile(pairs / 8)));
    }
    // Rows of 704 values (Gemma 4's expert down projections) are whole 32-value blocks but not
    // whole 256-value steps: for the 32-value formats they start on 4 bytes only, and the async
    // loop copies 4 bytes at a time.
    if (gguf::block_shape(c.type).elements == 32) {
        for (const int pairs : {128, 512, 1024}) {
            compare_moe_routes(c, a, pairs, offered.at(column_tile(pairs / 8)) != 0 ? 4 : 0, 704);
        }
    }
}

// `expect_copy`: the async route's staging copy in bytes (0: the compiled loop). With `tail`, k is
// not a whole number of 256-value steps and zeros follow the last row.
void compare_routes(const TypeCase& c, const AsyncCase& a,
                    const std::vector<std::uint8_t>& host_weight, int rows, int k, int columns,
                    int expect_copy, bool tail = false) {
    DeviceBuffer weight(host_weight.size() + 4096);
    check(cudaMemset(weight.p, 0, weight.bytes), "async route weight");
    weight.copy_from_host(host_weight.data(), host_weight.size());
    std::mt19937 rng(0x5A17u + static_cast<unsigned>(k + columns + rows));
    std::uniform_real_distribution<float> sample(-0.25f, 0.25f);
    std::vector<__nv_bfloat16> host_input(std::size_t(k) * columns);
    std::vector<__nv_bfloat16> initial(std::size_t(rows) * columns);
    for (auto& value : host_input) { value = __float2bfloat16(sample(rng)); }
    for (auto& value : initial) { value = __float2bfloat16(sample(rng)); }
    DeviceBuffer input(host_input.size() * sizeof(__nv_bfloat16));
    input.copy_from_host(host_input.data(), input.bytes);
    DeviceBuffer activation(
        gguf::matrix_activation_bytes(tail ? gguf::matrix_padded_k(k) : k, columns));
    if (tail) {
        gguf::quantize_matrix_activation_padded(c.type, static_cast<const __nv_bfloat16*>(input.p),
                                                k, columns, nullptr, activation.p, nullptr);
    } else {
        gguf::quantize_matrix_activation(c.type, static_cast<const __nv_bfloat16*>(input.p), k,
                                         columns, nullptr, activation.p, nullptr);
    }

    const RouteRun compiled =
        run_route("", c, a, rows, k, columns, weight, activation, initial, tail);
    const RouteRun async =
        run_route("async", c, a, rows, k, columns, weight, activation, initial, tail);
    const std::string label = std::string(c.name) + " rows=" + std::to_string(rows) + " K=" +
                              std::to_string(k) + " T=" + std::to_string(columns);
    if (compiled.async) {
        throw std::runtime_error(label + ": the compiled route took the async tiles");
    }
    if (async.async != expect_copy) {
        throw std::runtime_error(label + ": the async route copies " +
                                 std::to_string(async.async) + " bytes, expected " +
                                 std::to_string(expect_copy));
    }
    if (compiled.fixup_bytes != async.fixup_bytes) {
        throw std::runtime_error(label + ": the route changed the stream-K fixup geometry");
    }
    for (std::size_t part = 0; part < compiled.parts.size(); ++part) {
        if (compiled.parts[part] != async.parts[part]) {
            throw std::runtime_error(label + ": output part " + std::to_string(part) +
                                     " differs between the compiled and async routes");
        }
    }
    std::cout << "OK GGUF MMQ async " << label << " fixup=" << (compiled.fixup_bytes != 0)
              << " copy=" << async.async << '\n';
}

}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int sm_count = gguf::multiprocessor_count();
        for (const TypeCase& c : kCases) {
            check_rejected_geometry(c);
            const int nofix_rows = rows_for_fixup(c, 9, false);
            if (nofix_rows == 0) {
                throw std::runtime_error(std::string(c.name) + " has no non-fixup geometry");
            }
            run_direct_case(c, 512, 9, nofix_rows, 7, false, false);
            run_direct_case(c, 512, 9, nofix_rows, 7, true, false);

            const int split_rows = rows_for_fixup(c, 9, true);
            if (split_rows == 0) {
                throw std::runtime_error(std::string(c.name) + " did not expose a stream-K fixup shape");
            }
            run_direct_case(c, 512, 9, split_rows, 7, false, true);
            run_direct_case(c, 512, 9, split_rows, 7, true, true);
            run_direct_case(c, 256, 9, split_rows, 7, false, true, true);
            run_direct_case(c, 256, 9, split_rows, 7, true, true, true);
            const int tail_k = padded_tail_k(c);
            if (tail_k != 0) {
                run_direct_case(c, tail_k, 9, nofix_rows, 7, false, false, true);
                run_direct_case(c, tail_k, 9, nofix_rows, 7, true, false, true);
            }
            std::cout << "OK GGUF MMQ BF16 " << c.name << " nonfixup/split="
                      << (split_rows != 0) << '\n';
        }

        const TypeCase& q4 = kCases[3];
        const TypeCase& q6 = kCases[5];
        const bool q6_t16_fixup = gguf::matrix_fixup_bytes(q6.type, 5120, 16) != 0;
        for (const bool accumulate : {false, true}) {
            run_direct_case(q6, 17408, 16, 5120, 7, accumulate, q6_t16_fixup);
        }
        for (const TypeCase* c : {&q4, &q6}) {
            if (gguf::matrix_fixup_bytes(c->type, 128, 9) == 0) {
                throw std::runtime_error(std::string(c->name) + " has no multi-contributor fixup geometry");
            }
            for (const int columns : {9, 17, 65, 128, 129, 512, 1024}) {
                const int rows = 128;
                const bool fixup = gguf::matrix_fixup_bytes(c->type, rows, columns) != 0;
                if (columns == 9 && !fixup) {
                    throw std::runtime_error(std::string(c->name) + " lost the stream-K fixup route");
                }
                for (const bool accumulate : {false, true}) {
                    run_direct_case(*c, 5120, columns, rows, 7, accumulate, fixup);
                }
            }
            for (const bool accumulate : {false, true}) {
                run_public_gather_case(*c, accumulate);
            }
        }
        std::cout << "OK GGUF MMQ BF16 store, sm=" << sm_count << '\n';

        for (const AsyncCase& a : kAsyncCases) {
            const TypeCase& c =
                a.type == kIq1m.type    ? kIq1m
                : a.type == kNvfp4.type ? kNvfp4
                                        : *std::find_if(std::begin(kCases), std::end(kCases),
                                                        [&](const TypeCase& t) {
                                                            return t.type == a.type;
                                                        });
            const bool q4k = a.type == gguf::GgmlType::Q4_K;
            if (a.dense_key == nullptr) {
                compare_moe_widths(c, a);
                continue;
            }
            const auto offered =
                offered_tiles(c, a.dense_key,
                              [&](int columns) { return a.dense(128, columns, 0); });
            if (!offered.at(128)) {
                std::cout << "NOTE: " << c.name << " async tiles are not offered at 128 columns on "
                          << "this device; those arms ran the compiled route\n";
            }
            bool fixup_seen = false;
            bool plain_seen = false;
            for (const auto& shape :
                 {std::pair{128, 5120}, std::pair{17408, 5120}, std::pair{5120, 17408}}) {
                const int rows = shape.first;
                const int k = shape.second;
                const auto host_weight =
                    make_weight(c, rows, k, 0x3A11u + static_cast<unsigned>(rows));
                const std::vector<int> widths =
                    q4k ? std::vector<int>{9, 16, 17, 32, 33, 48, 64, 65, 128, 129, 256, 1280, 4096}
                        : std::vector<int>{9, 24, 48, 64, 65, 129, 1280};
                for (const int columns : widths) {
                    const bool fixup = gguf::matrix_fixup_bytes(c.type, rows, columns) != 0;
                    fixup_seen |= fixup;
                    plain_seen |= !fixup;
                    compare_routes(c, a, host_weight, rows, k, columns,
                                   offered.at(column_tile(columns)));
                }
            }
            // Gemma 4's 704- and 2112-wide inputs: 32-value formats' rows start on 4 bytes only.
            if (gguf::block_shape(c.type).elements == 32) {
                for (const int k : {704, 2112}) {
                    const auto host_weight = make_weight(c, 256, k, 0x3B07u + unsigned(k));
                    for (const int columns : {9, 64, 129}) {
                        compare_routes(c, a, host_weight, 256, k, columns,
                                       offered.at(column_tile(columns)) != 0 ? 4 : 0, true);
                    }
                }
            }
            compare_moe_widths(c, a);
            if (!fixup_seen || !plain_seen) {
                throw std::runtime_error(std::string(c.name) +
                                         ": the async comparison lost its fixup or its plain "
                                         "stream-K shapes");
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GGUF MMQ BF16 store: " << error.what() << '\n';
        return 1;
    }
}
