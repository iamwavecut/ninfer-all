// GGUF block matrices through the public linear, linear_add and linear_swiglu Ops and the embedding
// gather, against an FP64 product of the exactly dequantized weights. The products quantize the
// activation to q8_1 as llama.cpp does, so they are held to that arithmetic's error, not to A16's.
// Q4_K and Q6_K run once with each vector load schedule a device profile can name.

#include "core/arena.h"
#include "core/device_buffer.h"
#include "core/host_pages.h"
#include "core/qtype.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/common/device_route.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninfer;

struct TypeCase {
    QType qtype;
    const char* name;
    // Byte offsets of the binary16 multipliers inside one block, set to finite values.
    std::vector<int> half_fields;
};

std::vector<TypeCase> cases() {
    return {
        {QType::GGUF_Q8_0, "q8_0", {0}},       {QType::GGUF_Q2_K, "q2_k", {80, 82}},
        {QType::GGUF_Q3_K, "q3_k", {108}},     {QType::GGUF_Q4_K, "q4_k", {0, 2}},
        {QType::GGUF_Q5_K, "q5_k", {0, 2}},    {QType::GGUF_Q6_K, "q6_k", {208}},
        {QType::GGUF_IQ2_XXS, "iq2_xxs", {0}}, {QType::GGUF_IQ2_XS, "iq2_xs", {0}},
        {QType::GGUF_IQ2_S, "iq2_s", {0}},     {QType::GGUF_IQ3_XXS, "iq3_xxs", {0}},
        {QType::GGUF_IQ3_S, "iq3_s", {0}},     {QType::GGUF_IQ1_S, "iq1_s", {0}},
        {QType::GGUF_IQ1_M, "iq1_m", {}},      {QType::GGUF_IQ4_NL, "iq4_nl", {0}},
        {QType::GGUF_IQ4_XS, "iq4_xs", {0}},   {QType::GGUF_Q4_0, "q4_0", {0}},
        {QType::GGUF_Q5_0, "q5_0", {0}},       {QType::GGUF_Q2_0, "q2_0", {0}},
        {QType::GGUF_MXFP4, "mxfp4", {}},      {QType::GGUF_NVFP4, "nvfp4", {}},
    };
}

float half_value(const std::uint8_t* p) {
    __half h;
    std::memcpy(&h, p, sizeof(h));
    return __half2float(h);
}

// An independent host decode of the formats simple enough to state here, each value d * (code -
// offset): Q4_0 (low nibbles first, then high), Q5_0 (Q4_0 plus bit j of qh as bit 4 of value j)
// and Q2_0 (two bits per value, low bits first); and MXFP4, the E2M1 value of each nibble (sign,
// two exponent bits, one mantissa bit; low nibbles first) times 2^(e - 127); and NVFP4, the E2M1
// value of each nibble (value j of sub-block s the low nibble of code byte 8 s + j, j < 8, else the
// high nibble of code byte 8 s + j - 8) times its sub-block's UE4M3 scale. Empty for every other
// type.
std::vector<float> host_decode(QType qtype, const std::vector<std::uint8_t>& bytes) {
    std::vector<float> out;
    const auto block = gguf_block_shape(qtype);
    if (qtype != QType::GGUF_Q4_0 && qtype != QType::GGUF_Q5_0 && qtype != QType::GGUF_Q2_0 &&
        qtype != QType::GGUF_MXFP4 && qtype != QType::GGUF_NVFP4) {
        return out;
    }
    const auto e2m1 = [](int code) {
        const int exponent = (code >> 1) & 3, mantissa = code & 1;
        const double magnitude =
            exponent == 0 ? 0.5 * mantissa : std::ldexp(1.0 + 0.5 * mantissa, exponent - 1);
        return code & 8 ? -magnitude : magnitude;
    };
    const std::size_t blocks = bytes.size() / block.bytes;
    out.reserve(blocks * block.elements);
    for (std::size_t i = 0; i < blocks; ++i) {
        const std::uint8_t* b = bytes.data() + i * block.bytes;
        if (qtype == QType::GGUF_MXFP4) {
            const double scale = std::ldexp(1.0, int(b[0]) - 127);
            for (int j = 0; j < block.elements; ++j) {
                const int code = j < 16 ? (b[1 + j] & 0xF) : (b[1 + j - 16] >> 4);
                out.push_back(float(e2m1(code) * scale));
            }
            continue;
        }
        if (qtype == QType::GGUF_NVFP4) {
            for (int j = 0; j < block.elements; ++j) {
                const int s = j / 16, v = j % 16;
                const int exponent = (b[s] >> 3) & 15, mantissa = b[s] & 7;
                const double scale = exponent == 0 ? std::ldexp(double(mantissa), -9)
                                                   : std::ldexp(1.0 + mantissa / 8.0, exponent - 7);
                const int byte = b[4 + 8 * s + v % 8];
                out.push_back(float(e2m1(v < 8 ? byte & 0xF : byte >> 4) * scale));
            }
            continue;
        }
        const float d         = half_value(b);
        for (int j = 0; j < block.elements; ++j) {
            int code = 0;
            if (qtype == QType::GGUF_Q2_0) {
                code = ((b[2 + j / 4] >> (2 * (j % 4))) & 3) - 1;
            } else {
                const std::uint8_t* qs = b + (qtype == QType::GGUF_Q5_0 ? 6 : 2);
                code                   = j < 16 ? (qs[j] & 0xF) : (qs[j - 16] >> 4);
                if (qtype == QType::GGUF_Q5_0) {
                    std::uint32_t qh;
                    std::memcpy(&qh, b + 2, sizeof(qh));
                    code = (code | int(((qh >> j) & 1U) << 4)) - 16;
                } else {
                    code -= 8;
                }
            }
            out.push_back(d * float(code));
        }
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

// Random blocks with every binary16 multiplier finite; IQ1_M spreads its multiplier over the top
// nibbles of its four scale words, MXFP4's exponent byte takes scales 2^-12 .. 2^-7 and NVFP4's
// UE4M3 scale bytes 2^-9 .. 2^-4 (subnormals included).
std::vector<std::uint8_t> random_blocks(const TypeCase& c, std::int32_t rows, std::int32_t k,
                                        std::mt19937& rng) {
    const auto block = gguf_block_shape(c.qtype);
    const std::int64_t count = std::int64_t(rows) * (k / block.elements);
    std::vector<std::uint8_t> bytes(count * block.bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    for (auto& b : bytes) { b = static_cast<std::uint8_t>(byte(rng)); }
    for (std::int64_t i = 0; i < count; ++i) {
        std::uint8_t* base = bytes.data() + i * block.bytes;
        for (const int offset : c.half_fields) {
            const std::uint16_t bits = half_bits(scale(rng));
            std::memcpy(base + offset, &bits, sizeof(bits));
        }
        if (c.qtype == QType::GGUF_IQ1_M) {
            const std::uint16_t d = half_bits(scale(rng));
            auto* sc              = reinterpret_cast<std::uint16_t*>(base + 48);
            for (int j = 0; j < 4; ++j) {
                sc[j] = static_cast<std::uint16_t>((sc[j] & 0x0fff) | (((d >> (4 * j)) & 0xf) << 12));
            }
        }
        if (c.qtype == QType::GGUF_MXFP4) { base[0] = static_cast<std::uint8_t>(115 + byte(rng) % 6); }
        if (c.qtype == QType::GGUF_NVFP4) {
            for (int s = 0; s < 4; ++s) { base[s] = static_cast<std::uint8_t>(1 + byte(rng) % 31); }
        }
    }
    return bytes;
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

std::vector<float> to_float(const std::vector<__nv_bfloat16>& values) {
    std::vector<float> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { out[i] = __bfloat162float(values[i]); }
    return out;
}

struct Error {
    double relative_rms = 0;
    double relative_max = 0;
};

Error compare(const std::vector<float>& got, const std::vector<double>& want) {
    double num = 0, den = 0, worst = 0, scale = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double d = double(got[i]) - want[i];
        num += d * d;
        den += want[i] * want[i];
        worst = std::max(worst, std::abs(d));
        scale = std::max(scale, std::abs(want[i]));
    }
    return {std::sqrt(num / std::max(den, 1e-30)), worst / std::max(scale, 1e-30)};
}

// K = 1536 and K = 2560, Qwen3.8-Flash-Next's hidden width, whose 80 slices leave the vector
// products' last pass over a row with half of a warp's lanes idle; and Gemma 4's widths 2816, 2112
// and 704, which are no whole number of the matrix kernel's 512-value activation steps (2112 and
// 704 not even of its 256-value weight steps).
int run_type(const TypeCase& c, std::uint32_t seed, std::int32_t k) {
    constexpr std::int32_t kRows = 256;
    const std::int32_t kK        = k;
    const std::int32_t widths[]  = {1, 2, 3, 4, 5, 8, 9, 17, 40, 70, 130, 257};
    const std::int32_t max_t     = 257;
    std::mt19937 rng(seed);
    int failures = 0;

    const auto host_blocks = random_blocks(c, 2 * kRows, kK, rng);
    // Zero bytes after the last row, as a bound device tail keeps them: a row whose K is no whole
    // number of the matrix kernel's 256-value steps reads its last step into the next row.
    DeviceBuffer blocks(host_blocks.size() + 4096);
    blocks.fill(0);
    blocks.copy_from_host(host_blocks.data(), host_blocks.size());
    const auto block        = gguf_block_shape(c.qtype);
    const std::int64_t row_bytes = std::int64_t(kK / block.elements) * block.bytes;
    const Weight gate = gguf_weight(c.qtype, blocks.p, kRows, kK);
    Weight up         = gate;
    up.qdata          = static_cast<const std::byte*>(blocks.p) + kRows * row_bytes;
    const Weight both = gguf_weight(c.qtype, blocks.p, 2 * kRows, kK);

    // Exact weights.
    DeviceBuffer exact_device(std::size_t(2) * kRows * kK * sizeof(float));
    ops::gguf::dequantize_rows(ops::detail::gguf_type(c.qtype), blocks.p, row_bytes, kK,
                               nullptr, 2 * kRows, static_cast<float*>(exact_device.p), kK,
                               nullptr);
    check(cudaDeviceSynchronize(), "dequantize");
    std::vector<float> exact(std::size_t(2) * kRows * kK);
    exact_device.copy_to_host(exact.data(), exact.size() * sizeof(float));
    for (const float v : exact) {
        if (!std::isfinite(v)) {
            std::cerr << c.name << ": non-finite dequantized weight\n";
            return 1;
        }
    }
    if (const auto host = host_decode(c.qtype, host_blocks); !host.empty() && host != exact) {
        std::cerr << "FAIL " << c.name << ": dequantization differs from the host decode\n";
        return 1;
    }

    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16> x_host(std::size_t(kK) * max_t);
    for (auto& v : x_host) { v = __float2bfloat16(normal(rng)); }
    const auto x_float = to_float(x_host);
    DeviceBuffer x_device(x_host.size() * sizeof(__nv_bfloat16));
    x_device.copy_from_host(x_host.data(), x_host.size() * sizeof(__nv_bfloat16));

    std::vector<std::int32_t> columns(kK);
    for (std::int32_t i = 0; i < kK; ++i) { columns[i] = i; }
    std::shuffle(columns.begin(), columns.end(), rng);
    DeviceBuffer columns_device(columns.size() * sizeof(std::int32_t));
    columns_device.copy_from_host(columns.data(), columns.size() * sizeof(std::int32_t));
    Weight gathered        = gate;
    gathered.input_columns = static_cast<const std::int32_t*>(columns_device.p);

    const auto product = [&](std::int32_t row0, std::int32_t t, bool permuted) {
        std::vector<double> out(std::size_t(kRows) * t);
        for (std::int32_t col = 0; col < t; ++col) {
            for (std::int32_t r = 0; r < kRows; ++r) {
                double acc = 0;
                const float* w = exact.data() + std::size_t(row0 + r) * kK;
                const float* x = x_float.data() + std::size_t(col) * kK;
                for (std::int32_t i = 0; i < kK; ++i) {
                    acc += double(w[i]) * x[permuted ? columns[i] : i];
                }
                out[std::size_t(col) * kRows + r] = acc;
            }
        }
        return out;
    };

    const std::size_t capacity = std::max(
        {ops::linear_workspace_capacity_bytes(c.qtype, kRows, kK, ops::LinearPolicy::A16Only, 1,
                                              max_t),
         ops::linear_swiglu_workspace_capacity_bytes(c.qtype, 2 * kRows, kK,
                                                     ops::LinearPolicy::A16Only, 1, max_t),
         ops::linear_swiglu_pair_workspace_capacity_bytes(c.qtype, c.qtype, kRows, kK,
                                                          ops::LinearPolicy::A16Only, 1, max_t)});
    WorkspaceArena workspace(capacity + (1U << 20));

    // q8_1 activations cost one product about 1 % RMS against the exact weights; a SwiGLU output
    // carries two such products, so it gets twice the budget.
    const auto run_check = [&](const std::string& label, const std::vector<float>& got,
                               const std::vector<double>& want) {
        const Error e     = compare(got, want);
        const double gain = label.starts_with("swiglu") ? 2.0 : 1.0;
        const bool ok     = e.relative_rms < 2e-2 * gain && e.relative_max < 8e-2 * gain;
        if (!ok) {
            std::cerr << "FAIL " << c.name << " " << label << " rms=" << e.relative_rms
                      << " max=" << e.relative_max << '\n';
        }
        return ok ? 0 : 1;
    };

    for (const std::int32_t t : widths) {
        Tensor x(x_device.p, DType::BF16, {kK, t});
        DeviceBuffer out_device(std::size_t(kRows) * t * sizeof(__nv_bfloat16));
        Tensor out(out_device.p, DType::BF16, {kRows, t});
        std::vector<__nv_bfloat16> host(std::size_t(kRows) * t);

        ops::linear(x, gate, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        const auto plain = product(0, t, false);
        failures += run_check("linear T=" + std::to_string(t), to_float(host), plain);

        // An FP32 out is the same accumulation unrounded: rounding it gives the BF16 out exactly.
        DeviceBuffer wide_device(std::size_t(kRows) * t * sizeof(float));
        Tensor wide(wide_device.p, DType::FP32, {kRows, t});
        ops::linear(x, gate, wide, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "fp32 linear");
        std::vector<float> wide_host(std::size_t(kRows) * t);
        wide_device.copy_to_host(wide_host.data(), wide_host.size() * sizeof(float));
        failures += run_check("fp32 linear T=" + std::to_string(t), wide_host, plain);
        for (std::size_t i = 0; i < wide_host.size(); ++i) {
            if (__bfloat162float(__float2bfloat16(wide_host[i])) != __bfloat162float(host[i])) {
                std::cerr << "FAIL " << c.name << " fp32 linear T=" << t << " element " << i
                          << " rounds to " << __bfloat162float(__float2bfloat16(wide_host[i]))
                          << ", the BF16 out holds " << __bfloat162float(host[i]) << '\n';
                ++failures;
                break;
            }
        }

        ops::linear(x, gathered, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "gathered linear");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("gathered T=" + std::to_string(t), to_float(host),
                              product(0, t, true));

        std::vector<__nv_bfloat16> residual(std::size_t(kRows) * t);
        for (auto& v : residual) { v = __float2bfloat16(normal(rng)); }
        out_device.copy_from_host(residual.data(), residual.size() * sizeof(__nv_bfloat16));
        ops::linear_add(x, gate, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear_add");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        std::vector<double> sum = plain;
        for (std::size_t i = 0; i < sum.size(); ++i) { sum[i] += __bfloat162float(residual[i]); }
        failures += run_check("linear_add T=" + std::to_string(t), to_float(host), sum);

        const auto upper = product(kRows, t, false);
        std::vector<double> swiglu(plain.size());
        for (std::size_t i = 0; i < swiglu.size(); ++i) {
            swiglu[i] = plain[i] / (1.0 + std::exp(-plain[i])) * upper[i];
        }
        ops::linear_swiglu(x, both, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear_swiglu");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("swiglu T=" + std::to_string(t), to_float(host), swiglu);

        ops::linear_swiglu(x, gate, up, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear_swiglu pair");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("swiglu pair T=" + std::to_string(t), to_float(host), swiglu);
    }
    // The planned capacity covers every call; the slack above it only keeps a shortfall from
    // aborting the numerical checks before this one reports it.
    if (workspace.peak_used() > capacity) {
        std::cerr << "FAIL " << c.name << " workspace peak " << workspace.peak_used()
                  << " bytes over the planned " << capacity << '\n';
        ++failures;
    }

    // The token-table gather is the exact weights rounded to BF16.
    const std::vector<std::int32_t> ids{3, 0, 255, 17, 3};
    DeviceBuffer ids_device(ids.size() * sizeof(std::int32_t));
    ids_device.copy_from_host(ids.data(), ids.size() * sizeof(std::int32_t));
    DeviceBuffer rows_device(ids.size() * kK * sizeof(__nv_bfloat16));
    Tensor ids_tensor(ids_device.p, DType::I32, {static_cast<std::int32_t>(ids.size())});
    Tensor rows(rows_device.p, DType::BF16, {kK, static_cast<std::int32_t>(ids.size())});
    ops::embedding(ids_tensor, gate, rows, nullptr);
    check(cudaDeviceSynchronize(), "embedding");
    std::vector<__nv_bfloat16> gathered_rows(ids.size() * kK);
    rows_device.copy_to_host(gathered_rows.data(), gathered_rows.size() * sizeof(__nv_bfloat16));
    for (std::size_t i = 0; i < ids.size(); ++i) {
        for (std::int32_t j = 0; j < kK; ++j) {
            const float want = __bfloat162float(__float2bfloat16(exact[std::size_t(ids[i]) * kK + j]));
            if (__bfloat162float(gathered_rows[i * kK + j]) != want) {
                std::cerr << "FAIL " << c.name << " embedding row " << ids[i] << " column " << j
                          << '\n';
                ++failures;
                i = ids.size();
                break;
            }
        }
    }
    return failures;
}

// The token-table gather reading its table in place from page-locked host memory mapped into the
// device (`--embedding-residency host`), at the Qwen3.8-27B width: the rows must be bit for bit the
// ones the same gather reads from a device copy, which are the exact weights rounded to BF16, in a
// direct launch and in a replayed CUDA graph. `vocabulary` rows tile a random 4096-row block.
int run_mapped_table(const TypeCase& c, std::uint32_t seed, std::int32_t vocabulary) {
    constexpr std::int32_t kK       = 5120;
    constexpr std::int32_t kDistinct = 4096;
    std::mt19937 rng(seed);
    int failures = 0;

    const auto distinct           = random_blocks(c, kDistinct, kK, rng);
    const auto block              = gguf_block_shape(c.qtype);
    const std::int64_t row_bytes  = std::int64_t(kK / block.elements) * block.bytes;
    const std::size_t table_bytes = std::size_t(vocabulary) * row_bytes;
    HostPages host(table_bytes);
    for (std::size_t offset = 0; offset < table_bytes; offset += distinct.size()) {
        std::memcpy(host.data() + offset, distinct.data(),
                    std::min(distinct.size(), table_bytes - offset));
    }
    host.pin(true);
    void* alias = nullptr;
    check(cudaHostGetDevicePointer(&alias, host.data(), 0), "map the host table");
    DeviceBuffer device_table(table_bytes);
    device_table.copy_from_host(host.data(), table_bytes);
    const Weight mapped   = gguf_weight(c.qtype, alias, vocabulary, kK);
    const Weight resident = gguf_weight(c.qtype, device_table.p, vocabulary, kK);

    constexpr std::int32_t kMaxT = 2048;
    DeviceBuffer ids_device(kMaxT * sizeof(std::int32_t));
    DeviceBuffer from_host(std::size_t(kK) * kMaxT * sizeof(__nv_bfloat16));
    DeviceBuffer from_device(from_host.bytes);
    DeviceBuffer exact_device(std::size_t(kK) * kMaxT * sizeof(float));
    std::vector<__nv_bfloat16> got(std::size_t(kK) * kMaxT);
    std::vector<__nv_bfloat16> want(got.size());
    std::vector<float> exact(got.size());
    std::uniform_int_distribution<std::int32_t> row(0, vocabulary - 1);
    const auto draw_ids = [&](std::int32_t t) {
        std::vector<std::int32_t> ids(t);
        for (auto& id : ids) { id = row(rng); }
        // The table's first and last rows, and a repeat.
        ids[0] = vocabulary - 1;
        if (t > 1) { ids[1] = 0; }
        if (t > 2) { ids[2] = ids[0]; }
        ids_device.copy_from_host(ids.data(), ids.size() * sizeof(std::int32_t));
        return ids;
    };
    const auto compare_rows = [&](const std::string& label, std::int32_t t) {
        const std::size_t count = std::size_t(kK) * t;
        from_host.copy_to_host(got.data(), count * sizeof(__nv_bfloat16));
        from_device.copy_to_host(want.data(), count * sizeof(__nv_bfloat16));
        exact_device.copy_to_host(exact.data(), count * sizeof(float));
        if (std::memcmp(got.data(), want.data(), count * sizeof(__nv_bfloat16)) != 0) {
            std::cerr << "FAIL " << c.name << ' ' << label << ": host-table rows differ from the "
                      << "device table's\n";
            ++failures;
            return;
        }
        for (std::size_t i = 0; i < count; ++i) {
            if (__bfloat162float(got[i]) != __bfloat162float(__float2bfloat16(exact[i]))) {
                std::cerr << "FAIL " << c.name << ' ' << label << ": element " << i
                          << " is not the exact weight rounded to BF16\n";
                ++failures;
                return;
            }
        }
    };
    const auto launch = [&](const Weight& table, DeviceBuffer& out, std::int32_t t,
                            cudaStream_t stream) {
        Tensor ids(ids_device.p, DType::I32, {t});
        Tensor rows(out.p, DType::BF16, {kK, t});
        ops::embedding(ids, table, rows, stream);
    };
    const auto oracle = [&](std::int32_t t) {
        ops::gguf::dequantize_rows(ops::detail::gguf_type(c.qtype), device_table.p, row_bytes, kK,
                                   static_cast<const std::int32_t*>(ids_device.p), t,
                                   static_cast<float*>(exact_device.p), kK, nullptr);
    };

    for (const std::int32_t t : {1, 7, 128, kMaxT}) {
        (void)draw_ids(t);
        launch(mapped, from_host, t, nullptr);
        launch(resident, from_device, t, nullptr);
        oracle(t);
        check(cudaDeviceSynchronize(), "mapped embedding");
        compare_rows("T=" + std::to_string(t), t);
    }

    // A decode graph captures the gather once and replays it with new ids each round.
    constexpr std::int32_t kGraphT = 8;
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec  = nullptr;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    launch(mapped, from_host, kGraphT, stream);
    check(cudaStreamEndCapture(stream, &graph), "end capture");
    check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    for (int round = 0; round < 3; ++round) {
        (void)draw_ids(kGraphT);
        check(cudaGraphLaunch(exec, stream), "replay");
        check(cudaStreamSynchronize(stream), "replay");
        launch(resident, from_device, kGraphT, nullptr);
        oracle(kGraphT);
        check(cudaDeviceSynchronize(), "graph reference");
        compare_rows("graph round " + std::to_string(round), kGraphT);
    }
    check(cudaGraphExecDestroy(exec), "destroy graph");
    check(cudaGraphDestroy(graph), "destroy graph");
    check(cudaStreamDestroy(stream), "destroy stream");
    return failures;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures      = 0;
        std::uint32_t seed = 9001;
        for (const auto& c : cases()) {
            const bool cooperative = c.qtype == QType::GGUF_Q4_K || c.qtype == QType::GGUF_Q6_K;
            for (const std::string_view schedule :
                 cooperative ? std::vector<std::string_view>{"scalar", "cooperative"}
                             : std::vector<std::string_view>{""}) {
                std::optional<ops::DeviceRouteForce> force;
                if (!schedule.empty()) {
                    force.emplace("gguf_vector/" + std::string(c.name), std::string(schedule));
                }
                for (const std::int32_t k : {1536, 2560, 2816, 2112, 704}) {
                    if (k % gguf_block_shape(c.qtype).elements != 0) { continue; }
                    const int f = run_type(c, seed++, k);
                    std::cout << (f == 0 ? "OK   " : "FAIL ") << "gguf " << c.name << " K=" << k
                              << (schedule.empty() ? "" : " ") << schedule << '\n';
                    failures += f;
                }
            }
        }
        // The latent widths of Edge0's routed experts: gate and up K = 1792 (145B) or 1280 (120B) in
        // NVFP4 or Q3_K, down K = 640 in NVFP4 or IQ4_NL.
        for (const auto& c : cases()) {
            if (c.qtype != QType::GGUF_NVFP4 && c.qtype != QType::GGUF_Q3_K &&
                c.qtype != QType::GGUF_IQ4_NL) {
                continue;
            }
            for (const std::int32_t k : {1792, 1280, 640}) {
                const bool down = k == 640;
                if ((c.qtype == QType::GGUF_Q3_K && down) || (c.qtype == QType::GGUF_IQ4_NL && !down)) {
                    continue;
                }
                const int f = run_type(c, seed++, k);
                std::cout << (f == 0 ? "OK   " : "FAIL ") << "gguf " << c.name << " K=" << k << '\n';
                failures += f;
            }
        }
        for (const std::string_view schedule : {"", "async"}) {
            ops::DeviceRouteForce force("gguf_matrix/q4_k", std::string(schedule));
            for (const auto& c : cases()) {
                if (c.qtype != QType::GGUF_Q4_K) { continue; }
                for (const std::int32_t k : {2560, 5120}) {
                    const int f = run_type(c, seed++, k);
                    std::cout << (f == 0 ? "OK   " : "FAIL ") << "gguf " << c.name << " K=" << k
                              << " matrix " << (schedule.empty() ? "compiled" : schedule) << '\n';
                    failures += f;
                }
            }
        }
        // The local Qwen3.8-27B artifact's IQ2_S table at its full vocabulary; the other formats
        // a 27B GGUF conversion may store its table in, over a smaller one.
        for (const auto& c : cases()) {
            const bool iq2_s = c.qtype == QType::GGUF_IQ2_S;
            if (!iq2_s && c.qtype != QType::GGUF_Q8_0 && c.qtype != QType::GGUF_Q4_K &&
                c.qtype != QType::GGUF_Q6_K && c.qtype != QType::GGUF_IQ3_S) {
                continue;
            }
            const int f = run_mapped_table(c, seed++, iq2_s ? 248320 : 8192);
            std::cout << (f == 0 ? "OK   " : "FAIL ") << "gguf " << c.name
                      << " embedding from mapped host memory\n";
            failures += f;
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " GGUF Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "GGUF Linear: " << error.what() << '\n';
        return 1;
    }
}
