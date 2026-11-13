// Edge0's latent MoE over GGUF block banks (moe_latent_gguf) against an FP64 evaluation of its
// formula from independently decoded stored blocks (this file's decoders, ggml-quants.c's
// dequantizers in FP64), at the four releases' shapes: 145B (latent width 1792, Q8_0 encoder and
// decoder) and 120B (1280, Q4_0), each with NVFP4 experts and their per-expert multipliers or with
// Q3_K gate/up, IQ4_NL down and the encoder's column scale. The products quantize their inputs to
// q8_1 at four points (the token, the encoded column, the BF16 middle, the group sum), so y is
// held to that arithmetic's error by relative L2 and gross error. Exact checks: a repeat, the
// staged form with the routed experts divided over two adds (and, up to eight tokens, one share
// computed by the CPU products from the stage's quantized inputs), the shared expert on a side
// stream, and group-parallel parts combine to the single call's y bit for bit. Up to eight tokens
// take the vector kernel; 9 and 64 tokens divide the experts between the vector and matrix kernels,
// 72 run every product on the matrix kernel. Each call runs in exactly
// moe_latent_gguf_workspace_bytes and leaves its scopes closed.

#include "core/arena.h"
#include "core/device_buffer.h"
#include "core/qtype.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/moe_experts.h"
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
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ninfer;

constexpr int kHidden = 2560, kMiddle = 640, kExperts = 512, kTopK = 10, kGroups = 8;
constexpr std::size_t kTail = 256; // zero bytes after each matrix, as device banks keep them

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

double exact_half(unsigned bits) {
    const unsigned exponent = (bits >> 10) & 31, fraction = bits & 1023;
    const double value = exponent ? std::ldexp(1024.0 + fraction, int(exponent) - 25)
                                  : std::ldexp(double(fraction), -24);
    return bits & 0x8000 ? -value : value;
}

std::uint16_t half_bits(float value) {
    const __half h = __float2half(value);
    std::uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

unsigned load16(const std::uint8_t* p) { return unsigned(p[0]) | (unsigned(p[1]) << 8); }

// ggml-quants.c's dequantizers (llama.cpp a25c9865, MIT) with FP64 output, one block each.
void decode_q8_0(const std::uint8_t* x, double* y) {
    const double d = exact_half(load16(x));
    for (int j = 0; j < 32; ++j) { y[j] = d * static_cast<std::int8_t>(x[2 + j]); }
}

void decode_q4_0(const std::uint8_t* x, double* y) {
    const double d = exact_half(load16(x));
    for (int j = 0; j < 16; ++j) {
        y[j]      = ((x[2 + j] & 0x0F) - 8) * d;
        y[j + 16] = ((x[2 + j] >> 4) - 8) * d;
    }
}

void decode_iq4_nl(const std::uint8_t* x, double* y) {
    static constexpr int kValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                        1,    13,   25,  38,  53,  69,  89,  113};
    const double d = exact_half(load16(x));
    for (int j = 0; j < 16; ++j) {
        y[j]      = d * kValues[x[2 + j] & 0xF];
        y[j + 16] = d * kValues[x[2 + j] >> 4];
    }
}

void decode_q3_k(const std::uint8_t* x, double* y) {
    const double d = exact_half(load16(x + 108));
    std::uint32_t aux[4];
    std::memcpy(aux, x + 96, 12);
    const std::uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f, tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const auto* scales     = reinterpret_cast<const std::int8_t*>(aux);
    const std::uint8_t* q  = x + 32;
    const std::uint8_t* hm = x;
    int is                 = 0;
    std::uint8_t m         = 1;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            double dl = d * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l) {
                *y++ = dl * (int((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            }
            dl = d * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l) {
                *y++ = dl * (int((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            }
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}

// The E2M1 value of each code times its 16 values' UE4M3 scale (bits 0-6; 0 and 0x7F zero).
void decode_nvfp4(const std::uint8_t* x, double* y) {
    for (int s = 0; s < 4; ++s) {
        const int exponent = (x[s] >> 3) & 15, mantissa = x[s] & 7;
        const double scale = (x[s] & 0x7F) == 0x7F ? 0.0
                             : exponent == 0       ? std::ldexp(double(mantissa), -9)
                                                   : std::ldexp(1.0 + mantissa / 8.0, exponent - 7);
        for (int j = 0; j < 16; ++j) {
            const int byte = x[4 + 8 * s + j % 8];
            const int code = j < 8 ? byte & 15 : byte >> 4;
            const int e = (code >> 1) & 3, f = code & 1;
            const double magnitude = e == 0 ? 0.5 * f : std::ldexp(1.0 + 0.5 * f, e - 1);
            y[16 * s + j] = (code & 8 ? -magnitude : magnitude) * scale;
        }
    }
}

struct Format {
    int elements, bytes;
    void (*decode)(const std::uint8_t*, double*);
};

Format format_of(QType q) {
    switch (q) {
    case QType::GGUF_Q8_0: return {32, 34, decode_q8_0};
    case QType::GGUF_Q4_0: return {32, 18, decode_q4_0};
    case QType::GGUF_IQ4_NL: return {32, 18, decode_iq4_nl};
    case QType::GGUF_Q3_K: return {256, 110, decode_q3_k};
    case QType::GGUF_NVFP4: return {64, 36, decode_nvfp4};
    default: throw std::runtime_error("no decoder for this format");
    }
}

// Random stored blocks with multipliers of a real release's sizes, and their decoded values.
struct Matrix {
    std::vector<std::uint8_t> bytes;
    std::vector<double> values; // [rows][k]
};

Matrix random_matrix(QType q, int rows, int k, std::mt19937& rng) {
    const Format f      = format_of(q);
    const int blocks    = rows * (k / f.elements);
    Matrix m;
    m.bytes.resize(std::size_t(blocks) * f.bytes);
    for (auto& b : m.bytes) { b = static_cast<std::uint8_t>(rng()); }
    std::uniform_real_distribution<float> plain(0.0005f, 0.002f), k_quant(0.0001f, 0.0004f),
        nonlinear(0.0002f, 0.001f);
    for (int i = 0; i < blocks; ++i) {
        std::uint8_t* p = m.bytes.data() + std::size_t(i) * f.bytes;
        std::uint16_t d = 0;
        switch (q) {
        case QType::GGUF_Q3_K:
            d = half_bits(k_quant(rng));
            std::memcpy(p + 108, &d, 2);
            break;
        case QType::GGUF_NVFP4:
            for (int s = 0; s < 4; ++s) { p[s] = static_cast<std::uint8_t>(1 + rng() % 31); }
            break;
        case QType::GGUF_IQ4_NL:
            d = half_bits(nonlinear(rng));
            std::memcpy(p, &d, 2);
            break;
        default:
            d = half_bits(plain(rng));
            std::memcpy(p, &d, 2);
            break;
        }
    }
    m.values.resize(std::size_t(rows) * k);
    for (int i = 0; i < blocks; ++i) {
        f.decode(m.bytes.data() + std::size_t(i) * f.bytes, m.values.data() + std::size_t(i) * f.elements);
    }
    return m;
}

// `distinct` random matrices, a device table of `entries` pointers (entry e at matrix e % distinct).
struct Bank {
    QType format;
    int rows, k;
    std::int64_t row_bytes;
    std::vector<Matrix> matrices;
    std::vector<DeviceBuffer> device;
    std::vector<const void*> pointers;
    DeviceBuffer table;

    Bank(QType q, int rows_, int k_, int distinct, int entries, std::mt19937& rng)
        : format(q), rows(rows_), k(k_), table(std::size_t(entries) * sizeof(void*)) {
        const Format f = format_of(q);
        row_bytes      = std::int64_t(k / f.elements) * f.bytes;
        for (int i = 0; i < distinct; ++i) {
            matrices.push_back(random_matrix(q, rows, k, rng));
            const auto& bytes = matrices.back().bytes;
            device.emplace_back(bytes.size() + kTail);
            device.back().fill(0);
            device.back().copy_from_host(bytes.data(), bytes.size());
        }
        for (int e = 0; e < entries; ++e) { pointers.push_back(device[e % distinct].p); }
        table.copy_from_host(pointers.data(), pointers.size() * sizeof(void*));
    }

    [[nodiscard]] ops::GgufExpertTable view() const {
        return {format, static_cast<const void* const*>(table.p), row_bytes};
    }
    [[nodiscard]] const Matrix& of(int entry) const { return matrices[entry % matrices.size()]; }
    [[nodiscard]] ops::GgufCpuMatrix cpu(int entry) const {
        return {format, of(entry).bytes.data(), row_bytes, rows, k};
    }
};

// out[r] = sum_i w[r][i] x[i] in FP64.
void product(const Matrix& w, int rows, int k, const double* x, double* out) {
    for (int r = 0; r < rows; ++r) {
        const double* row = w.values.data() + std::size_t(r) * k;
        double acc        = 0;
        for (int i = 0; i < k; ++i) { acc += row[i] * x[i]; }
        out[r] = acc;
    }
}

double silu(double x) { return x / (1.0 + std::exp(-x)); }

struct Case {
    const char* name;
    QType gate_up, down, projection;
    int width;
    bool expert_scales, column_scale;
};

// Runs fn(t) for t < count on the host's threads.
template <typename Fn>
void parallel_for(int count, Fn fn) {
    const int threads = std::max(1, std::min<int>(count, int(std::thread::hardware_concurrency())));
    std::vector<std::thread> pool;
    for (int w = 0; w < threads; ++w) {
        pool.emplace_back([&, w] {
            for (int t = w; t < count; t += threads) { fn(t); }
        });
    }
    for (auto& thread : pool) { thread.join(); }
}

int run_case(const Case& c, std::uint32_t seed) {
    std::mt19937 rng(seed);
    const int L = c.width;
    const Bank gate(c.gate_up, kMiddle, L, 3, kExperts, rng);
    const Bank up(c.gate_up, kMiddle, L, 3, kExperts, rng);
    const Bank down(c.down, L, kMiddle, 3, kExperts, rng);
    const Bank encode(c.projection, L, kHidden, kGroups, kGroups, rng);
    const Bank decode(c.projection, kHidden, L, kGroups, kGroups, rng);
    const Bank sgate(QType::GGUF_Q8_0, kMiddle, kHidden, 1, 1, rng);
    const Bank sup(QType::GGUF_Q8_0, kMiddle, kHidden, 1, 1, rng);
    const Bank sdown(QType::GGUF_Q8_0, kHidden, kMiddle, 1, 1, rng);
    std::vector<std::int32_t> labels(kExperts);
    for (auto& label : labels) { label = int(rng() % kGroups); }
    std::uniform_real_distribution<float> multiplier(0.5f, 2.0f);
    std::vector<float> scales(std::size_t(kExperts) * 3), column(L);
    for (auto& s : scales) { s = multiplier(rng); }
    for (auto& s : column) { s = multiplier(rng); }
    DeviceBuffer d_labels(labels.size() * 4), d_scales(scales.size() * 4), d_column(column.size() * 4);
    d_labels.copy_from_host(labels.data(), labels.size() * 4);
    d_scales.copy_from_host(scales.data(), scales.size() * 4);
    d_column.copy_from_host(column.data(), column.size() * 4);
    ops::GgufLatentWeights banks;
    banks.gate = gate.view(), banks.up = up.view(), banks.down = down.view();
    banks.encode = encode.view(), banks.decode = decode.view();
    banks.shared_gate = sgate.view(), banks.shared_up = sup.view(), banks.shared_down = sdown.view();
    banks.labels          = static_cast<const std::int32_t*>(d_labels.p);
    banks.expert_scales   = c.expert_scales ? static_cast<const float*>(d_scales.p) : nullptr;
    banks.column_scale    = c.column_scale ? static_cast<const float*>(d_column.p) : nullptr;
    banks.width           = L;
    banks.groups          = kGroups;
    banks.experts         = kExperts;
    banks.device_resident = true;

    cudaStream_t side_stream = nullptr;
    cudaEvent_t fork = nullptr, join = nullptr;
    check(cudaStreamCreateWithFlags(&side_stream, cudaStreamNonBlocking), "side stream");
    check(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming), "fork event");
    check(cudaEventCreateWithFlags(&join, cudaEventDisableTiming), "join event");
    const ops::GgufMoeSide side{side_stream, fork, join};

    int failures = 0;
    const auto report = [&](bool ok, int tokens, const std::string& what) {
        std::cout << (ok ? "OK   " : "FAIL ") << c.name << " T=" << tokens << ": " << what << '\n';
        failures += ok ? 0 : 1;
    };
    for (const int tokens : {1, 2, 8, 9, 64, 72}) {
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::uniform_real_distribution<float> unit(0.05f, 1.0f);
        std::vector<__nv_bfloat16> m(std::size_t(kHidden) * tokens);
        for (auto& v : m) { v = __float2bfloat16(normal(rng)); }
        std::vector<std::int32_t> ids(std::size_t(kTopK) * tokens);
        std::vector<float> weights(ids.size()), shared(tokens);
        // Routes over 60 experts: past eight tokens experts with few and with many pairs.
        std::vector<int> pool(60);
        std::iota(pool.begin(), pool.end(), 0);
        for (int t = 0; t < tokens; ++t) {
            std::shuffle(pool.begin(), pool.end(), rng);
            float total = 0;
            for (int j = 0; j < kTopK; ++j) {
                ids[t * kTopK + j]     = pool[j] * 7 % kExperts;
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
        Tensor t_m(d_m.p, DType::BF16, {kHidden, tokens}), t_ids(d_ids.p, DType::I32, {kTopK, tokens}),
            t_w(d_w.p, DType::FP32, {kTopK, tokens}), t_s(d_s.p, DType::FP32, {tokens}),
            t_y(d_y.p, DType::FP32, {kHidden, tokens});
        WorkspaceArena workspace(ops::moe_latent_gguf_workspace_bytes(tokens, L, kGroups));
        const auto read_y = [&] {
            check(cudaDeviceSynchronize(), "latent MoE");
            std::vector<float> out(std::size_t(kHidden) * tokens);
            d_y.copy_to_host(out.data(), out.size() * 4);
            return out;
        };
        ops::moe_latent_gguf(t_m, t_ids, t_w, t_s, banks, workspace, t_y, nullptr);
        const auto got = read_y();
        report(workspace.used() == 0, tokens, "scopes closed after the call");
        ops::moe_latent_gguf(t_m, t_ids, t_w, t_s, banks, workspace, t_y, nullptr);
        report(same_bits(read_y(), got), tokens, "a repeat gives the same bits");

        // The FP64 formula of every output.
        std::vector<double> oracle(std::size_t(kHidden) * tokens, 0.0);
        parallel_for(tokens, [&](int t) {
            std::vector<double> x(kHidden), xl(std::size_t(kGroups) * L), sum(std::size_t(kGroups) * L, 0.0);
            for (int i = 0; i < kHidden; ++i) { x[i] = __bfloat162float(m[std::size_t(t) * kHidden + i]); }
            std::vector<bool> used(kGroups, false);
            for (int j = 0; j < kTopK; ++j) { used[labels[ids[t * kTopK + j]]] = true; }
            for (int g = 0; g < kGroups; ++g) {
                if (!used[g]) { continue; }
                double* column_g = xl.data() + std::size_t(g) * L;
                product(encode.of(g), L, kHidden, x.data(), column_g);
                if (c.column_scale) {
                    for (int r = 0; r < L; ++r) { column_g[r] *= column[r]; }
                }
            }
            std::vector<double> g_out(kMiddle), u_out(kMiddle), h(kMiddle), yl(L);
            for (int j = 0; j < kTopK; ++j) {
                const int e = ids[t * kTopK + j], g = labels[e];
                const double* a = xl.data() + std::size_t(g) * L;
                product(gate.of(e), kMiddle, L, a, g_out.data());
                product(up.of(e), kMiddle, L, a, u_out.data());
                const double sg = c.expert_scales ? scales[3 * e] : 1.0,
                             su = c.expert_scales ? scales[3 * e + 1] : 1.0,
                             sd = c.expert_scales ? scales[3 * e + 2] : 1.0;
                for (int r = 0; r < kMiddle; ++r) { h[r] = silu(sg * g_out[r]) * (su * u_out[r]); }
                product(down.of(e), L, kMiddle, h.data(), yl.data());
                for (int r = 0; r < L; ++r) {
                    sum[std::size_t(g) * L + r] += weights[t * kTopK + j] * sd * yl[r];
                }
            }
            double* want = oracle.data() + std::size_t(t) * kHidden;
            std::vector<double> rows(kHidden);
            for (int g = 0; g < kGroups; ++g) {
                if (!used[g]) { continue; }
                product(decode.of(g), kHidden, L, sum.data() + std::size_t(g) * L, rows.data());
                for (int r = 0; r < kHidden; ++r) { want[r] += rows[r]; }
            }
            std::vector<double> sg(kMiddle), su(kMiddle), sh(kMiddle), so(kHidden);
            product(sgate.of(0), kMiddle, kHidden, x.data(), sg.data());
            product(sup.of(0), kMiddle, kHidden, x.data(), su.data());
            for (int r = 0; r < kMiddle; ++r) { sh[r] = silu(sg[r]) * su[r]; }
            product(sdown.of(0), kHidden, kMiddle, sh.data(), so.data());
            for (int r = 0; r < kHidden; ++r) { want[r] += shared[t] * so[r]; }
        });
        {
            double num = 0, den = 0, worst = 0, scale = 0;
            for (std::size_t i = 0; i < oracle.size(); ++i) {
                const double d = double(got[i]) - oracle[i];
                num += d * d;
                den += oracle[i] * oracle[i];
                worst = std::max(worst, std::abs(d));
                scale = std::max(scale, std::abs(oracle[i]));
            }
            const double rms = std::sqrt(num / std::max(den, 1e-30)), max = worst / std::max(scale, 1e-30);
            // Four q8_1 activations and a BF16 middle per chain.
            report(std::isfinite(rms) && rms < 4e-2 && max < 1.6e-1, tokens,
                   "against FP64 rms=" + std::to_string(rms) + " max=" + std::to_string(max));
        }

        // The staged form: experts below 256 in one add, the others in a second (or, up to eight
        // tokens, from the CPU products of the stage's quantized inputs), and the shared expert on
        // a side stream.
        std::vector<std::int32_t> low(ids), high(ids);
        std::vector<std::int32_t> high_pairs;
        for (std::size_t p = 0; p < ids.size(); ++p) {
            (ids[p] < 256 ? high : low)[p] = -1;
            if (ids[p] >= 256) { high_pairs.push_back(std::int32_t(p)); }
        }
        DeviceBuffer d_low(low.size() * 4), d_high(high.size() * 4);
        d_low.copy_from_host(low.data(), low.size() * 4);
        d_high.copy_from_host(high.data(), high.size() * 4);
        const Tensor t_low(d_low.p, DType::I32, {kTopK, tokens}), t_high(d_high.p, DType::I32, {kTopK, tokens});
        {
            auto scope       = workspace.scope();
            const auto stage = ops::moe_latent_gguf_begin(t_m, t_ids, t_s, banks, workspace, nullptr,
                                                          tokens <= 8 ? &side : nullptr);
            ops::moe_latent_gguf_add(stage, t_high, t_w, banks, workspace, nullptr);
            ops::moe_latent_gguf_add(stage, t_low, t_w, banks, workspace, nullptr);
            ops::moe_latent_gguf_finish(stage, banks, workspace, t_y, nullptr);
            report(same_bits(read_y(), got), tokens, "two adds equal the single call");
        }
        const bool cpu = tokens <= 8 && ops::gguf_cpu_supported(c.gate_up) &&
                         ops::gguf_cpu_supported(c.down);
        if (cpu) {
            auto scope       = workspace.scope();
            const auto stage = ops::moe_latent_gguf_begin(t_m, t_ids, t_s, banks, workspace, nullptr);
            ops::moe_latent_gguf_add(stage, t_low, t_w, banks, workspace, nullptr);
            check(cudaDeviceSynchronize(), "staged begin");
            const int pairs = stage.pairs;
            std::vector<std::int8_t> codes(std::size_t(pairs) * L);
            std::vector<std::uint16_t> ds(std::size_t(pairs) * (L / 32) * 2);
            check(cudaMemcpy(codes.data(), stage.inputs, codes.size(), cudaMemcpyDeviceToHost), "codes");
            const std::size_t offset = (std::size_t(L) * pairs + 15) / 16 * 16;
            check(cudaMemcpy(ds.data(), static_cast<const std::uint8_t*>(stage.inputs) + offset,
                             ds.size() * 2, cudaMemcpyDeviceToHost),
                  "scales");
            const int count = int(high_pairs.size());
            std::vector<float> values(std::size_t(std::max(count, 1)) * L);
            for (int i = 0; i < count; ++i) {
                const int p = high_pairs[i], e = ids[p];
                ops::GgufCpuActivation a(1, L), a_up(1, L), middle(1, kMiddle);
                const int columns[] = {p};
                a.load_q8_1(c.gate_up, codes.data(), ds.data(), columns);
                a_up.load_q8_1(c.gate_up, codes.data(), ds.data(), columns);
                std::vector<float> g(kMiddle), u(kMiddle);
                constexpr auto backend = ops::GgufCpuBackend::Automatic;
                ops::gguf_cpu_rows(gate.cpu(e), a, 0, kMiddle, g.data(), kMiddle, backend);
                ops::gguf_cpu_rows(up.cpu(e), a_up, 0, kMiddle, u.data(), kMiddle, backend);
                if (c.expert_scales) {
                    for (int r = 0; r < kMiddle; ++r) {
                        g[r] = scales[3 * e] * g[r];
                        u[r] = scales[3 * e + 1] * u[r];
                    }
                }
                std::vector<std::uint16_t> h(kMiddle);
                ops::gguf_cpu_swiglu_bf16(g.data(), u.data(), kMiddle, h.data(), backend);
                middle.prepare_bf16(c.down, h.data(), kMiddle, 1);
                ops::gguf_cpu_rows(down.cpu(e), middle, 0, L, values.data() + std::size_t(i) * L, L,
                                   backend);
            }
            DeviceBuffer d_values(values.size() * 4), d_pairs(std::max<std::size_t>(high_pairs.size(), 1) * 4),
                d_count(4);
            d_values.copy_from_host(values.data(), values.size() * 4);
            if (count > 0) { d_pairs.copy_from_host(high_pairs.data(), high_pairs.size() * 4); }
            d_count.copy_from_host(&count, 4);
            ops::moe_latent_gguf_add_products(stage, static_cast<const float*>(d_values.p),
                                              static_cast<const std::int32_t*>(d_pairs.p),
                                              static_cast<const std::int32_t*>(d_count.p), pairs,
                                              t_ids, t_w, banks, nullptr);
            ops::moe_latent_gguf_finish(stage, banks, workspace, t_y, nullptr);
            report(same_bits(read_y(), got), tokens,
                   std::to_string(count) + " CPU products (" +
                       ops::gguf_cpu_backend_name(ops::gguf_cpu_backend_selected(
                           ops::GgufCpuBackend::Automatic)) +
                       ") equal the GPU's");
        }

        // Group parallelism: parts over group ranges combine to the single call's bits.
        for (const auto& ends : std::vector<std::vector<int>>{{0, 3, 8}, {0, 1, 5, 8}}) {
            const std::size_t n = ends.size() - 1;
            std::vector<DeviceBuffer> fixed;
            std::vector<Tensor> parts;
            for (std::size_t p = 0; p < n; ++p) {
                fixed.emplace_back(std::size_t(kHidden) * tokens * 8);
                parts.push_back(Tensor(fixed.back().p, DType::I64, {kHidden, tokens}));
                const bool with_shared = ends.size() == 3 ? p == 0 : p == n - 1;
                ops::moe_latent_gguf_part(t_m, t_ids, t_w, with_shared ? &t_s : nullptr, banks,
                                          ends[p], ends[p + 1], workspace, parts.back(), nullptr);
            }
            ops::moe_experts_gguf_combine(parts, t_y, nullptr);
            report(same_bits(read_y(), got) && workspace.used() == 0, tokens,
                   std::to_string(n) + " group-parallel parts equal the single call");
        }
    }
    {
        // A call past kGgufLatentSliceTokens runs in two slices: the bits of the staged form, which
        // runs it whole, in the workspace of one slice.
        const int tokens = ops::kGgufLatentSliceTokens + 52;
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::uniform_real_distribution<float> unit(0.05f, 1.0f);
        std::uniform_int_distribution<int> expert(0, kExperts - 1);
        std::vector<__nv_bfloat16> m(std::size_t(kHidden) * tokens);
        for (auto& v : m) { v = __float2bfloat16(normal(rng)); }
        std::vector<std::int32_t> ids(std::size_t(kTopK) * tokens);
        std::vector<float> weights(ids.size()), shared(tokens);
        for (int t = 0; t < tokens; ++t) {
            for (int j = 0; j < kTopK; ++j) {
                int e = 0;
                do {
                    e = expert(rng);
                } while (std::find(ids.begin() + t * kTopK, ids.begin() + t * kTopK + j, e) !=
                         ids.begin() + t * kTopK + j);
                ids[t * kTopK + j]     = e;
                weights[t * kTopK + j] = unit(rng);
            }
            shared[t] = unit(rng);
        }
        DeviceBuffer d_m(m.size() * 2), d_ids(ids.size() * 4), d_w(weights.size() * 4),
            d_s(shared.size() * 4), d_y(std::size_t(kHidden) * tokens * 4);
        d_m.copy_from_host(m.data(), m.size() * 2);
        d_ids.copy_from_host(ids.data(), ids.size() * 4);
        d_w.copy_from_host(weights.data(), weights.size() * 4);
        d_s.copy_from_host(shared.data(), shared.size() * 4);
        Tensor t_m(d_m.p, DType::BF16, {kHidden, tokens}), t_ids(d_ids.p, DType::I32, {kTopK, tokens}),
            t_w(d_w.p, DType::FP32, {kTopK, tokens}), t_s(d_s.p, DType::FP32, {tokens}),
            t_y(d_y.p, DType::FP32, {kHidden, tokens});
        const auto read_y = [&] {
            check(cudaDeviceSynchronize(), "latent MoE");
            std::vector<float> out(std::size_t(kHidden) * tokens);
            d_y.copy_to_host(out.data(), out.size() * 4);
            return out;
        };
        {
            WorkspaceArena slice(ops::moe_latent_gguf_workspace_bytes(ops::kGgufLatentSliceTokens, L,
                                                                      kGroups));
            ops::moe_latent_gguf(t_m, t_ids, t_w, t_s, banks, slice, t_y, nullptr);
        }
        const auto sliced = read_y();
        WorkspaceArena whole(ops::moe_latent_gguf_workspace_bytes(tokens, L, kGroups));
        const auto stage = ops::moe_latent_gguf_begin(t_m, t_ids, t_s, banks, whole, nullptr);
        ops::moe_latent_gguf_add(stage, t_ids, t_w, banks, whole, nullptr);
        ops::moe_latent_gguf_finish(stage, banks, whole, t_y, nullptr);
        report(same_bits(read_y(), sliced), tokens, "two slices equal the whole call");
    }
    check(cudaStreamDestroy(side_stream), "side stream");
    check(cudaEventDestroy(fork), "fork event");
    check(cudaEventDestroy(join), "join event");
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
        using Q = QType;
        const std::vector<Case> cases = {
            {"145B nvfp4", Q::GGUF_NVFP4, Q::GGUF_NVFP4, Q::GGUF_Q8_0, 1792, true, false},
            {"145B q3_k", Q::GGUF_Q3_K, Q::GGUF_IQ4_NL, Q::GGUF_Q8_0, 1792, false, true},
            {"120B nvfp4", Q::GGUF_NVFP4, Q::GGUF_NVFP4, Q::GGUF_Q4_0, 1280, true, false},
            {"120B q3_k", Q::GGUF_Q3_K, Q::GGUF_IQ4_NL, Q::GGUF_Q4_0, 1280, false, true},
        };
        int failures = 0;
        std::uint32_t seed = 17;
        for (const auto& c : cases) { failures += run_case(c, seed++); }
        if (failures != 0) {
            std::cerr << failures << " check(s) failed\n";
            return 1;
        }
        std::cout << "all latent MoE checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
