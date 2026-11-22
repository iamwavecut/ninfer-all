// Inkling 2's own kernels against FP64 oracles at Inkling-2-Mini's shapes (hidden 2304, 24 query
// heads over 8 of width 128 with 16-wide relative features, a 1024-position window, 128 experts
// choosing 8 of width 1152):
//  - rms_norm's BF16, FP32 and q8_1 rows; mix (the depth attention over 1..8 states for 1..4
//    queries: raw and normalized rows); the loop controllers' inputs and their update.
//  - gemm_f32 by the GEMV kernel (1..8 rows, strided) and by cuBLAS; gemm_f32_batch's products
//    equal to gemm_f32's bit for bit.
//  - The split rows of a short call (mix and residual over one-warp blocks): the row kernel's bits.
//  - query_post: Q and the relative feature clipped per head and scaled by the full layers' log
//    factor below and above its floor; kv_post: the short convolutions over the call's rows and
//    the history ring (decode rows of several sequences, a verification's consecutive rows, a
//    prompt chunk after earlier positions), the dynamic taps, K's clip, V's norm, the cache slots
//    and the history a decode call keeps; MXFP8 planes (a full visit of an mxfp8 run), decoded on
//    the host from their codes and exponent bytes, within MXFP8's error of the oracle.
//  - attention: relative bias inside its extent only, per-query sinks far below, near and above the
//    scores, the sigmoid gate; sliding-window rings and full caches, one and many splits, prompt
//    chunks after cached positions and branch rows, with and without the far field on tensor
//    cores, verification rows in groups with the bits of single rows; BF16 and q8_1 outputs.
//  - residual over plain products, 2^-32 fixed-point sums and weighted pair rows, starting a block
//    or adding to it, keeping the history, with the next depth mix fused; keep_products and
//    keep_history.
//  - route: sigmoid scores, the selection bias, ties broken by the lower id, the weights' scales;
//    the sorted route of up to eight rows against route and gguf::moe_sort_routes.
//  - situ_rows: the BF16 middles equal to the host's situ() bit for bit, and within BF16 rounding
//    of FP64; softcap, f32_to_bf16, place_rows, ring_copy, add_fixed.
//  - bf16_expert_rows (UD-Q8_K_XL's BF16 expert banks): [gate; up] rows from each token's input
//    and down rows from each pair's own, decode and prompt pair counts, against FP64 sums of the
//    BF16 values; another device's pairs (-1) keep their rows, and a second call gives the same
//    bits. bf16_rows_f32: the table's rows exactly.
//  - the routed experts of a call on the CPU (CpuExperts) against the GPU's vector kernels over
//    the release's formats (NVFP4 with per-expert scales, Q5_K/Q5_0, Q4_K/Q5_0, IQ4_XS/IQ4_NL): the
//    same 2^-32 fixed-point sums bit for bit, all on the CPU and half on each (add_fixed).
// A BF16 output is compared with its oracle's value rounded once; FP32 arithmetic may land on the
// other side of a rounding boundary, so those checks bound an RMS error instead of exact bits.

#include "core/device_buffer.h"
#include "core/qtype.h"
#include "core/weight.h"
#include "models/inkling2/config.h"
#include "models/inkling2/cpu_experts.h"
#include "models/inkling2/kernels.h"
#include "models/inkling2/situ.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
namespace k    = ninfer::models::inkling2::kernels;
namespace mi   = ninfer::models::inkling2;
namespace gguf = ninfer::ops::gguf;
using bf16     = __nv_bfloat16;

constexpr int kHidden = 2304, kHeads = 24, kKv = 8, kD = 128, kRel = 16, kTaps = 4;
constexpr int kExperts = 128, kTopK = 8, kWidth = 1152, kWindow = 1024, kExtent = 1024;
constexpr int kSlots = 32; // history slots per site
constexpr float kEps = 1e-6f, kQkClip = 22.627417f, kRelClip = 8.0f;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void sync(const char* what) { check(cudaDeviceSynchronize(), what); }

int failures = 0;

void report(bool ok, const std::string& what) {
    std::cout << (ok ? "OK   " : "FAIL ") << what << '\n';
    failures += ok ? 0 : 1;
}

std::string num(double x) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.3g", x);
    return text;
}

template <typename T>
DeviceBuffer upload(const std::vector<T>& values) {
    DeviceBuffer buffer(std::max<std::size_t>(values.size() * sizeof(T), 16));
    if (!values.empty()) { buffer.copy_from_host(values.data(), values.size() * sizeof(T)); }
    return buffer;
}

template <typename T>
std::vector<T> download(const DeviceBuffer& buffer, std::size_t count) {
    std::vector<T> values(count);
    buffer.copy_to_host(values.data(), count * sizeof(T));
    return values;
}

template <typename T>
T* as(DeviceBuffer& buffer) {
    return static_cast<T*>(buffer.p);
}

template <typename T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

std::vector<float> normal(std::size_t n, float sigma, std::mt19937& rng) {
    std::normal_distribution<float> d(0.0f, sigma);
    std::vector<float> v(n);
    for (auto& x : v) { x = d(rng); }
    return v;
}

// The RMS of the errors over the RMS of the oracle.
struct Error {
    double num = 0, den = 0;
    void add(double got, double want) {
        num += (got - want) * (got - want);
        den += want * want;
    }
    [[nodiscard]] double rel() const { return den > 0 ? std::sqrt(num / den) : std::sqrt(num); }
};

double q8_value(const std::vector<std::int8_t>& qs, const std::vector<half2>& ds, int k,
                std::size_t row, int i) {
    return double(qs[row * k + i]) * double(__low2float(ds[row * (k / 32) + i / 32]));
}

double rms_scale(const double* x, int n, double eps) {
    double s = 0;
    for (int i = 0; i < n; ++i) { s += x[i] * x[i]; }
    return 1.0 / std::sqrt(s / n + eps);
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// ---- rms_norm, mix, controllers -----------------------------------------------------------------

void rms_norm_case(int tokens, bool weight, std::mt19937& rng) {
    const auto x = normal(std::size_t(tokens) * kHidden, 3.0f, rng);
    const auto w = normal(kHidden, 1.0f, rng);
    DeviceBuffer dx = upload(x), dw = upload(w), out(std::size_t(tokens) * kHidden * 2),
                 f32(std::size_t(tokens) * kHidden * 4), qs(std::size_t(tokens) * kHidden),
                 ds(std::size_t(tokens) * (kHidden / 32) * 4);
    k::rms_norm(as<float>(dx), weight ? as<float>(dw) : nullptr, kHidden, tokens, kEps,
                as<bf16>(out), nullptr, k::Q8Rows{as<std::int8_t>(qs), as<half2>(ds)},
                as<float>(f32));
    sync("rms_norm");
    const auto got   = download<bf16>(out, std::size_t(tokens) * kHidden);
    const auto got_f = download<float>(f32, std::size_t(tokens) * kHidden);
    const auto got_q = download<std::int8_t>(qs, std::size_t(tokens) * kHidden);
    const auto got_d = download<half2>(ds, std::size_t(tokens) * (kHidden / 32));
    Error eb, ef, eq;
    for (int t = 0; t < tokens; ++t) {
        std::vector<double> row(x.begin() + std::size_t(t) * kHidden,
                                x.begin() + std::size_t(t + 1) * kHidden);
        const double s = rms_scale(row.data(), kHidden, kEps);
        for (int i = 0; i < kHidden; ++i) {
            const double y = row[i] * s * (weight ? w[i] : 1.0);
            const std::size_t at = std::size_t(t) * kHidden + i;
            eb.add(__bfloat162float(got[at]), y);
            ef.add(got_f[at], y);
            eq.add(q8_value(got_q, got_d, kHidden, t, i), y);
        }
    }
    report(eb.rel() < 3e-3 && ef.rel() < 1e-6 && eq.rel() < 1e-2,
           "rms_norm " + std::to_string(tokens) + " row(s)" + (weight ? "" : " without a weight") +
               ": bf16 " + num(eb.rel()) + ", fp32 " + num(ef.rel()) + ", q8_1 " + num(eq.rel()));
}

// The depth mix of every row as the oracle computes it: [query][t][hidden].
std::vector<double> mix_oracle(const std::vector<std::vector<float>>& states,
                               const std::vector<float>& queries, int query_count, int tokens,
                               const std::vector<std::vector<float>>& norms, bool normalized) {
    const int count = int(states.size());
    std::vector<double> out(std::size_t(query_count) * tokens * kHidden);
    for (int t = 0; t < tokens; ++t) {
        std::vector<std::vector<double>> s(count, std::vector<double>(kHidden));
        for (int j = 0; j < count; ++j) {
            for (int i = 0; i < kHidden; ++i) { s[j][i] = states[j][std::size_t(t) * kHidden + i]; }
            const double r = rms_scale(s[j].data(), kHidden, mi::kMixEps);
            for (auto& v : s[j]) { v *= r; }
        }
        for (int q = 0; q < query_count; ++q) {
            std::vector<double> score(count);
            double top = -INFINITY;
            for (int j = 0; j < count; ++j) {
                double d = 0;
                for (int i = 0; i < kHidden; ++i) { d += queries[std::size_t(q) * kHidden + i] * s[j][i]; }
                score[j] = d / std::sqrt(double(kHidden));
                top      = std::max(top, score[j]);
            }
            double sum = 0;
            for (auto& v : score) { sum += (v = std::exp(v - top)); }
            std::vector<double> y(kHidden, 0.0);
            for (int j = 0; j < count; ++j) {
                for (int i = 0; i < kHidden; ++i) { y[i] += score[j] / sum * s[j][i]; }
            }
            if (normalized) {
                const double r = rms_scale(y.data(), kHidden, kEps);
                for (int i = 0; i < kHidden; ++i) {
                    y[i] *= r * (norms.empty() ? 1.0 : norms[q][i]);
                }
            }
            std::copy(y.begin(), y.end(), out.begin() + (std::size_t(q) * tokens + t) * kHidden);
        }
    }
    return out;
}

void mix_case(int count, int query_count, int tokens, std::mt19937& rng) {
    std::vector<std::vector<float>> states;
    std::vector<DeviceBuffer> d_states;
    for (int j = 0; j < count; ++j) {
        states.push_back(normal(std::size_t(tokens) * kHidden, 0.5f + float(j), rng));
        d_states.push_back(upload(states.back()));
    }
    const auto queries = normal(std::size_t(query_count) * kHidden, 0.3f, rng);
    std::vector<std::vector<float>> norms;
    std::vector<DeviceBuffer> d_norms;
    for (int q = 0; q < query_count; ++q) {
        norms.push_back(normal(kHidden, 1.0f, rng));
        d_norms.push_back(upload(norms.back()));
    }
    const std::size_t per = std::size_t(tokens) * kHidden;
    DeviceBuffer dq = upload(queries), raw(per * query_count * 4), out(per * query_count * 2),
                 f32(per * query_count * 4),
                 q8(std::size_t(query_count) * (per + std::size_t(tokens) * (kHidden / 32) * 4));
    k::MixArgs a;
    for (int j = 0; j < count; ++j) { a.states[j] = as<float>(d_states[j]); }
    a.count        = count;
    a.queries      = as<float>(dq);
    a.query_count  = query_count;
    a.tokens       = tokens;
    a.hidden       = kHidden;
    a.eps          = kEps;
    a.query_stride = std::int64_t(per);
    a.q8_stride    = std::int64_t(per + std::size_t(tokens) * (kHidden / 32) * 4);
    a.raw          = as<float>(raw);
    for (int q = 0; q < query_count; ++q) { a.norm[q] = as<float>(d_norms[q]); }
    a.out      = as<bf16>(out);
    a.out_q.qs = as<std::int8_t>(q8);
    a.out_q.ds = reinterpret_cast<half2*>(as<std::uint8_t>(q8) + per);
    a.f32      = as<float>(f32);
    k::mix(a, nullptr);
    sync("mix");
    const auto want_raw  = mix_oracle(states, queries, query_count, tokens, {}, false);
    const auto want_norm = mix_oracle(states, queries, query_count, tokens, norms, true);
    const auto got_raw   = download<float>(raw, per * query_count);
    const auto got_f     = download<float>(f32, per * query_count);
    const auto got_b     = download<bf16>(out, per * query_count);
    const auto bytes     = download<std::uint8_t>(q8, std::size_t(query_count) * a.q8_stride);
    Error er, ef, eb, eq;
    for (int q = 0; q < query_count; ++q) {
        const std::uint8_t* base = bytes.data() + std::size_t(q) * a.q8_stride;
        for (std::size_t i = 0; i < per; ++i) {
            const std::size_t at = std::size_t(q) * per + i;
            er.add(got_raw[at], want_raw[at]);
            ef.add(got_f[at], want_norm[at]);
            eb.add(__bfloat162float(got_b[at]), want_norm[at]);
            half2 d;
            std::memcpy(&d, base + per + (i / 32) * 4, 4);
            eq.add(double(static_cast<std::int8_t>(base[i])) * __low2float(d), want_norm[at]);
        }
    }
    // The split rows (one-warp blocks adding their warps' sums in order): the same bits.
    DeviceBuffer split(k::mix_split_floats(tokens, kHidden) * 4), raw2(per * query_count * 4),
        out2(per * query_count * 2), f32_2(per * query_count * 4), q8_2(q8.bytes);
    k::MixArgs b = a;
    b.split      = as<float>(split);
    b.raw        = as<float>(raw2);
    b.out        = as<bf16>(out2);
    b.out_q.qs   = as<std::int8_t>(q8_2);
    b.out_q.ds   = reinterpret_cast<half2*>(as<std::uint8_t>(q8_2) + per);
    b.f32        = as<float>(f32_2);
    k::mix(b, nullptr);
    sync("split mix");
    const bool same = same_bits(download<float>(raw2, per * query_count), got_raw) &&
                      same_bits(download<float>(f32_2, per * query_count), got_f) &&
                      same_bits(download<bf16>(out2, per * query_count), got_b) &&
                      same_bits(download<std::uint8_t>(q8_2, std::size_t(query_count) * a.q8_stride),
                                bytes);
    report(er.rel() < 2e-6 && ef.rel() < 2e-6 && eb.rel() < 3e-3 && eq.rel() < 1e-2 && same,
           "mix of " + std::to_string(count) + " state(s), " + std::to_string(query_count) +
               " quer" + (query_count == 1 ? "y" : "ies") + ": raw " + num(er.rel()) + ", fp32 " +
               num(ef.rel()) + ", bf16 " + num(eb.rel()) + ", q8_1 " + num(eq.rel()) +
               (same ? ", split rows the same bits" : ", split rows DIFFER"));
}

void gemm_case(int m, int n, int k_, std::int64_t x_stride, std::int64_t out_stride,
               std::mt19937& rng) {
    const std::int64_t xs = x_stride != 0 ? x_stride : k_, os = out_stride != 0 ? out_stride : m;
    const auto w = normal(std::size_t(m) * k_, 0.05f, rng);
    const auto x = normal(std::size_t(n) * xs, 1.0f, rng);
    DeviceBuffer dw = upload(w), dx = upload(x), out(std::size_t(n) * os * 4);
    k::gemm_f32(m, n, k_, as<float>(dw), as<float>(dx), as<float>(out), nullptr, x_stride, out_stride);
    sync("gemm_f32");
    const auto got = download<float>(out, std::size_t(n) * os);
    Error e;
    for (int t = 0; t < n; ++t) {
        for (int r = 0; r < m; ++r) {
            double s = 0;
            for (int i = 0; i < k_; ++i) { s += double(w[std::size_t(r) * k_ + i]) * x[t * xs + i]; }
            e.add(got[t * os + r], s);
        }
    }
    report(e.rel() < 2e-6, "gemm_f32 " + std::to_string(m) + "x" + std::to_string(k_) + " over " +
                               std::to_string(n) + " row(s)" + (x_stride ? " strided" : "") + ": " +
                               num(e.rel()));
}

// gemm_f32_batch: each product's bits those of gemm_f32 (the GEMV kernel, the short-k kernel by
// warps, cuBLAS past eight rows).
void gemm_batch_case(int m, int n, int k_, int count, std::mt19937& rng) {
    std::vector<DeviceBuffer> w, x, out, want;
    k::GemvBatch b;
    for (int e = 0; e < count; ++e) {
        w.push_back(upload(normal(std::size_t(m) * k_, 0.05f, rng)));
        x.push_back(upload(normal(std::size_t(n) * k_, 1.0f, rng)));
        out.emplace_back(std::size_t(n) * m * 4);
        want.emplace_back(std::size_t(n) * m * 4);
    }
    for (int e = 0; e < count; ++e) {
        b.w[e]   = as<float>(w[e]);
        b.x[e]   = as<float>(x[e]);
        b.out[e] = as<float>(out[e]);
        k::gemm_f32(m, n, k_, b.w[e], b.x[e], as<float>(want[e]), nullptr);
    }
    b.count = count;
    k::gemm_f32_batch(m, n, k_, b, nullptr);
    sync("gemm_f32_batch");
    int differ = 0;
    for (int e = 0; e < count; ++e) {
        differ += same_bits(download<float>(out[e], std::size_t(n) * m),
                            download<float>(want[e], std::size_t(n) * m))
                      ? 0
                      : 1;
    }
    report(differ == 0, "gemm_f32_batch of " + std::to_string(count) + " " + std::to_string(m) +
                            "x" + std::to_string(k_) + " over " + std::to_string(n) + " row(s): " +
                            std::to_string(differ) + " product(s) differ from gemm_f32");
}

void control_case(int count, int tokens, std::mt19937& rng) {
    const std::size_t per = std::size_t(tokens) * kHidden;
    std::vector<std::vector<float>> x, m, f, g, norm;
    std::vector<DeviceBuffer> dx, dm, df, dg, dn;
    for (int c = 0; c < count; ++c) {
        x.push_back(normal(per, 2.0f, rng));
        m.push_back(normal(per, 0.7f, rng));
        f.push_back(normal(per, 1.0f, rng));
        g.push_back(normal(per, 1.0f, rng));
        norm.push_back(normal(kHidden, 1.0f, rng));
        dx.push_back(upload(x.back()));
        dm.push_back(upload(m.back()));
        df.push_back(upload(f.back()));
        dg.push_back(upload(g.back()));
        dn.push_back(upload(norm.back()));
    }
    DeviceBuffer concat(std::size_t(count) * per * 2 * 4), out(std::size_t(count) * per * 2),
        f32(std::size_t(count) * per * 4);
    k::ControlInputs in;
    k::ControlApply ap;
    for (int c = 0; c < count; ++c) {
        in.x[c] = ap.x[c] = as<float>(dx[c]);
        in.message[c] = ap.message[c] = as<float>(dm[c]);
        ap.forget[c] = as<float>(df[c]);
        ap.gain[c]   = as<float>(dg[c]);
        ap.norm[c]   = as<float>(dn[c]);
    }
    in.count = ap.count = count;
    in.tokens = ap.tokens = tokens;
    in.hidden = ap.hidden = kHidden;
    in.concat       = as<float>(concat);
    ap.eps          = kEps;
    ap.query_stride = std::int64_t(per);
    ap.out          = as<bf16>(out);
    ap.f32          = as<float>(f32);
    k::control_inputs(in, nullptr);
    k::control_apply(ap, nullptr);
    sync("controllers");
    const auto got_c = download<float>(concat, std::size_t(count) * per * 2);
    const auto got_f = download<float>(f32, std::size_t(count) * per);
    const auto got_b = download<bf16>(out, std::size_t(count) * per);
    Error ec, ef, eb;
    for (int c = 0; c < count; ++c) {
        for (int t = 0; t < tokens; ++t) {
            const std::size_t row = std::size_t(t) * kHidden;
            std::vector<double> xr(x[c].begin() + row, x[c].begin() + row + kHidden),
                mr(m[c].begin() + row, m[c].begin() + row + kHidden), y(kHidden);
            const double sx = rms_scale(xr.data(), kHidden, mi::kMixEps);
            const double sm = rms_scale(mr.data(), kHidden, mi::kMixEps);
            const std::size_t cat = (std::size_t(c) * tokens + t) * 2 * kHidden;
            for (int i = 0; i < kHidden; ++i) {
                ec.add(got_c[cat + i], xr[i] * sx);
                ec.add(got_c[cat + kHidden + i], mr[i] * sm);
                y[i] = xr[i] - std::tanh(double(f[c][row + i])) * xr[i] +
                       std::tanh(double(g[c][row + i])) * mr[i];
            }
            const double sy = rms_scale(y.data(), kHidden, kEps);
            for (int i = 0; i < kHidden; ++i) {
                const double want    = y[i] * sy * norm[c][i];
                const std::size_t at = std::size_t(c) * per + row + i;
                ef.add(got_f[at], want);
                eb.add(__bfloat162float(got_b[at]), want);
            }
        }
    }
    report(ec.rel() < 1e-6 && ef.rel() < 2e-6 && eb.rel() < 3e-3,
           "controllers of " + std::to_string(count) + " input(s): inputs " + num(ec.rel()) +
               ", update fp32 " + num(ef.rel()) + ", bf16 " + num(eb.rel()));
}

// ---- Attention inputs ---------------------------------------------------------------------------

void query_case(bool log_scale, const std::vector<int>& positions, float sigma, std::mt19937& rng) {
    const int t = int(positions.size());
    const std::int64_t stride = 2 * kHeads * kD + kHeads * kRel + 8, rel_col = 2 * kHeads * kD;
    const auto qgr  = normal(std::size_t(t) * stride, sigma, rng);
    auto qclip      = normal(kHeads * kD, 0.3f, rng);
    auto rclip      = normal(kHeads * kRel, 0.3f, rng);
    for (auto& v : qclip) { v += 1.0f; }
    for (auto& v : rclip) { v += 1.0f; }
    std::vector<std::int32_t> pos(positions.begin(), positions.end());
    DeviceBuffer d_qgr = upload(qgr), d_qc = upload(qclip), d_rc = upload(rclip), d_pos = upload(pos),
                 q_out(std::size_t(t) * kHeads * kD * 4), r_out(std::size_t(t) * kHeads * kRel * 4);
    k::QueryArgs a;
    a.qgr             = as<float>(d_qgr);
    a.stride          = stride;
    a.relative_column = rel_col;
    a.tokens          = t;
    a.heads           = kHeads;
    a.query_clip      = as<float>(d_qc);
    a.relative_clip   = as<float>(d_rc);
    a.qk_clip         = kQkClip;
    a.rel_clip        = kRelClip;
    a.log_scale       = log_scale;
    a.alpha           = 0.1f;
    a.floor           = 8192.0f;
    a.positions       = as<std::int32_t>(d_pos);
    a.q_out           = as<float>(q_out);
    a.r_out           = as<float>(r_out);
    k::query_post(a, nullptr);
    sync("query_post");
    const auto gq = download<float>(q_out, std::size_t(t) * kHeads * kD);
    const auto gr = download<float>(r_out, std::size_t(t) * kHeads * kRel);
    Error eq, er;
    int clipped = 0;
    for (int r = 0; r < t; ++r) {
        const double tau =
            log_scale ? 1.0 + 0.1 * std::log(std::max(1.0, (positions[r] + 1.0) / 8192.0)) : 1.0;
        for (int h = 0; h < kHeads; ++h) {
            const auto clip = [&](int width, std::int64_t column, const std::vector<float>& scale,
                                  double limit, const std::vector<float>& got, Error& e) {
                std::vector<double> x(width);
                double n = 0;
                for (int d = 0; d < width; ++d) {
                    x[d] = double(qgr[r * stride + column + h * width + d]) * scale[h * width + d];
                    n += x[d] * x[d];
                }
                const double f = std::min(1.0, limit / (std::sqrt(n) + 1e-6));
                clipped += f < 1.0 ? 1 : 0;
                for (int d = 0; d < width; ++d) {
                    e.add(got[(std::size_t(r) * kHeads + h) * width + d], x[d] * f * tau);
                }
            };
            clip(kD, 0, qclip, kQkClip, gq, eq);
            clip(kRel, rel_col, rclip, kRelClip, gr, er);
        }
    }
    report(eq.rel() < 1e-6 && er.rel() < 1e-6,
           std::string("query_post") + (log_scale ? " with the log factor" : "") + ", sigma " +
               num(sigma) + ", " + std::to_string(clipped) + " clipped heads: q " + num(eq.rel()) +
               ", r " + num(er.rel()));
}

// The call's rows: positions, their sequences and, for a chunk, the first physical position.
struct RowsCase {
    std::vector<int> positions, sequences;
    bool chunk = false;
    int first  = 0, shared = 0;
    std::vector<int> starts; // branches of a chunk
};

struct RowsDevice {
    DeviceBuffer positions, sequences, starts;
    k::Rows rows;
    explicit RowsDevice(const RowsCase& c)
        : positions(upload(std::vector<std::int32_t>(c.positions.begin(), c.positions.end()))),
          sequences(upload(std::vector<std::int32_t>(c.sequences.begin(), c.sequences.end()))),
          starts(upload(std::vector<std::int32_t>(c.starts.begin(), c.starts.end()))) {
        rows.positions = as<std::int32_t>(positions);
        rows.sequences = c.sequences.empty() ? nullptr : as<std::int32_t>(sequences);
        rows.tokens    = int(c.positions.size());
        rows.chunk     = c.chunk;
        rows.first     = c.first;
        rows.shared    = c.shared;
        rows.starts    = c.starts.empty() ? nullptr : as<std::int32_t>(starts);
    }
};

int sequence_of(const RowsCase& c, int t) { return c.sequences.empty() ? 0 : c.sequences[t]; }

// The row of the call holding the input `lag` positions before row t, or -1 (the ring) or -2
// (before position 0), as the kernels define it (kernels.h).
int source_row(const RowsCase& c, int t, int lag) {
    const int p = c.positions[t] - lag;
    if (lag == 0) { return t; }
    if (p < 0) { return -2; }
    if (!c.chunk) {
        const int s = t - lag;
        return s >= 0 && c.positions[s] == p && sequence_of(c, s) == sequence_of(c, t) ? s : -1;
    }
    if (!c.starts.empty() && t >= c.shared) {
        const int b0 = c.starts[t];
        if (p >= c.positions[b0]) { return b0 + (p - c.positions[b0]); }
    }
    return p >= c.first ? p - c.first : -1;
}

int physical(const RowsCase& c, int t) { return c.chunk ? c.first + t : c.positions[t]; }

// MXFP8 planes of `slots` positions of each sequence's eight heads: codes, then exponent bytes.
mi::KvPlaneView mxfp8_plane(DeviceBuffer& codes, DeviceBuffer& scales, int slots) {
    mi::KvPlaneView view;
    view.codes      = static_cast<std::uint8_t*>(codes.p);
    view.scales     = static_cast<std::uint8_t*>(scales.p);
    view.code_row   = kD;
    view.scale_row  = kD / 32;
    view.code_head  = std::int64_t(slots) * view.code_row;
    view.scale_head = std::int64_t(slots) * view.scale_row;
    view.code_seq   = view.code_head * kKv;
    view.scale_seq  = view.scale_head * kKv;
    view.coding     = mi::KvCoding::Mxfp8;
    return view;
}

double e4m3_value(std::uint8_t code) {
    const int e = (code >> 3) & 15, m = code & 7;
    const double magnitude = e == 0 ? m * std::ldexp(1.0, -9) : (1.0 + m / 8.0) * std::ldexp(1.0, e - 7);
    return code & 0x80 ? -magnitude : magnitude;
}

void kv_case(const std::string& name, const RowsCase& c, bool dynamic, std::mt19937& rng,
             bool mxfp8 = false) {
    const int t = int(c.positions.size());
    const int sequences = c.sequences.empty() ? 1 : *std::max_element(c.sequences.begin(), c.sequences.end()) + 1;
    const int channels = kKv * kD;
    const std::int64_t kv_stride = 2 * channels + 16, taps_stride = kKv * kTaps + 4;
    int ring = 0;
    for (int r = 0; r < t; ++r) { ring = std::max(ring, physical(c, r) + 1); }
    const auto kv     = normal(std::size_t(t) * kv_stride, 1.5f, rng);
    const auto k_taps = normal(std::size_t(t) * taps_stride, 0.3f, rng);
    const auto v_taps = normal(std::size_t(t) * taps_stride, 0.3f, rng);
    const auto k_conv = normal(std::size_t(channels) * kTaps, 0.4f, rng);
    const auto v_conv = normal(std::size_t(channels) * kTaps, 0.4f, rng);
    const std::int64_t seq_hist = std::int64_t(kSlots) * channels;
    const auto k_hist = normal(std::size_t(sequences) * seq_hist, 1.5f, rng);
    const auto v_hist = normal(std::size_t(sequences) * seq_hist, 1.5f, rng);
    const std::int64_t head_stride = std::int64_t(ring) * kD, seq_stride = head_stride * kKv;
    DeviceBuffer d_kv = upload(kv), d_kt = upload(k_taps), d_vt = upload(v_taps),
                 d_kc = upload(k_conv), d_vc = upload(v_conv), d_kh = upload(k_hist),
                 d_vh = upload(v_hist), kc(std::size_t(sequences) * seq_stride * 2),
                 vc(std::size_t(sequences) * seq_stride * 2);
    check(cudaMemset(kc.p, 0, kc.bytes), "zero K");
    check(cudaMemset(vc.p, 0, vc.bytes), "zero V");
    RowsDevice rows(c);
    k::KvArgs a;
    a.rows        = rows.rows;
    a.kv          = as<float>(d_kv);
    a.kv_stride   = kv_stride;
    a.k_taps      = dynamic ? as<float>(d_kt) : nullptr;
    a.v_taps      = dynamic ? as<float>(d_vt) : nullptr;
    a.taps_stride = taps_stride;
    a.k_conv      = as<float>(d_kc);
    a.v_conv      = as<float>(d_vc);
    a.k_history   = k::History{as<float>(d_kh), seq_hist, channels, kSlots};
    a.v_history   = k::History{as<float>(d_vh), seq_hist, channels, kSlots};
    a.keep        = !c.chunk;
    a.kv_heads    = kKv;
    a.qk_clip     = kQkClip;
    a.eps         = kEps;
    a.k_cache     = as<__half>(kc);
    a.v_cache     = as<__half>(vc);
    a.seq_stride  = seq_stride;
    a.head_stride = head_stride;
    a.ring        = 0;
    // MXFP8 planes of every position (the cache's halves stay zero), read back as values below.
    DeviceBuffer mk(mxfp8 ? std::size_t(sequences) * kKv * ring * kD : 16),
        mks(mxfp8 ? std::size_t(sequences) * kKv * ring * 4 : 16), mv(mk.bytes), mvs(mks.bytes);
    if (mxfp8) {
        a.k_plane = mxfp8_plane(mk, mks, ring);
        a.v_plane = mxfp8_plane(mv, mvs, ring);
    }
    k::kv_post(a, nullptr);
    sync("kv_post");
    auto got_k        = download<__half>(kc, std::size_t(sequences) * seq_stride);
    auto got_v        = download<__half>(vc, std::size_t(sequences) * seq_stride);
    const auto kept_k = download<float>(d_kh, std::size_t(sequences) * seq_hist);
    const auto kept_v = download<float>(d_vh, std::size_t(sequences) * seq_hist);
    std::vector<double> plane_k, plane_v; // the planes' values in the cache's layout
    if (mxfp8) {
        const auto codes_k  = download<std::uint8_t>(mk, mk.bytes), codes_v = download<std::uint8_t>(mv, mv.bytes);
        const auto scales_k = download<std::uint8_t>(mks, mks.bytes), scales_v = download<std::uint8_t>(mvs, mvs.bytes);
        plane_k.resize(got_k.size());
        plane_v.resize(got_v.size());
        for (std::size_t i = 0; i < plane_k.size(); ++i) {
            plane_k[i] = e4m3_value(codes_k[i]) * std::ldexp(1.0, scales_k[i / 32] - 127);
            plane_v[i] = e4m3_value(codes_v[i]) * std::ldexp(1.0, scales_v[i / 32] - 127);
        }
    }
    const auto stored = [&](bool key, std::int64_t at) {
        if (mxfp8) { return key ? plane_k[std::size_t(at)] : plane_v[std::size_t(at)]; }
        return double(__half2float(key ? got_k[std::size_t(at)] : got_v[std::size_t(at)]));
    };
    Error ek, ev;
    int clipped = 0;
    std::size_t kept_wrong = 0;
    const double divisor = 1.0 / std::sqrt(double((dynamic ? 2 : 1) * kTaps + 1));
    for (int r = 0; r < t; ++r) {
        const int s = sequence_of(c, r);
        for (int h = 0; h < kKv; ++h) {
            std::vector<double> kk(kD), vv(kD);
            for (int d = 0; d < kD; ++d) {
                const int ch = h * kD + d;
                const auto conv = [&](std::int64_t column, const std::vector<float>& w,
                                      const std::vector<float>& taps, const std::vector<float>& hist) {
                    double in[kTaps];
                    for (int lag = 0; lag < kTaps; ++lag) {
                        const int src = source_row(c, r, lag);
                        const int p   = c.positions[r] - lag;
                        in[lag] = src >= 0   ? kv[src * kv_stride + column]
                                  : src == -1 ? hist[s * seq_hist + (p % kSlots) * channels + ch]
                                              : 0.0;
                    }
                    double out = in[0];
                    for (int i = 0; i < kTaps; ++i) { out += w[std::size_t(ch) * kTaps + i] * in[kTaps - 1 - i]; }
                    if (dynamic) {
                        for (int lag = 0; lag < kTaps; ++lag) { out += in[lag] * taps[r * taps_stride + h * kTaps + lag]; }
                    }
                    return out * divisor;
                };
                kk[d] = conv(ch, k_conv, k_taps, k_hist);
                vv[d] = conv(channels + ch, v_conv, v_taps, v_hist);
            }
            double n = 0;
            for (const double v : kk) { n += v * v; }
            const double f = std::min(1.0, kQkClip / (std::sqrt(n) + 1e-6));
            clipped += f < 1.0 ? 1 : 0;
            const double vs = rms_scale(vv.data(), kD, kEps);
            const std::int64_t at = s * seq_stride + h * head_stride + std::int64_t(physical(c, r)) * kD;
            for (int d = 0; d < kD; ++d) {
                ek.add(stored(true, at + d), kk[d] * f);
                ev.add(stored(false, at + d), vv[d] * vs);
            }
        }
        if (!c.chunk) {
            const std::int64_t slot = s * seq_hist + (c.positions[r] % kSlots) * channels;
            for (int ch = 0; ch < channels; ++ch) {
                kept_wrong += kept_k[slot + ch] != kv[r * kv_stride + ch] ? 1 : 0;
                kept_wrong += kept_v[slot + ch] != kv[r * kv_stride + channels + ch] ? 1 : 0;
            }
        }
    }
    // MXFP8 keeps three bits of each value's significand: a relative error near 2^-4 / sqrt(3).
    const double bound = mxfp8 ? 4e-2 : 1e-3;
    report(ek.rel() < bound && ev.rel() < bound && kept_wrong == 0,
           "kv_post " + name + (dynamic ? " with dynamic taps" : "") + (mxfp8 ? ", MXFP8 planes" : "") +
               ", " + std::to_string(clipped) + " clipped heads: K " + num(ek.rel()) + ", V " +
               num(ev.rel()) + (c.chunk ? "" : ", history values wrong " + std::to_string(kept_wrong)));
}

// ---- Attention ----------------------------------------------------------------------------------

struct AttentionCase {
    std::string name;
    RowsCase rows;
    int ring = 0, window = 0, splits = 1;
    float sink_offset = 0.0f;
    bool far = false; // a full visit's chunk: its far field on tensor cores (AttentionArgs::far)
    int row_group = 1; // a full visit's decode or verification rows grouped (AttentionArgs::row_group)
};

void attention_case(const AttentionCase& c, std::mt19937& rng) {
    const RowsCase& rc = c.rows;
    const int t = int(rc.positions.size());
    const int sequences = rc.sequences.empty() ? 1 : *std::max_element(rc.sequences.begin(), rc.sequences.end()) + 1;
    int top = 0;
    for (int r = 0; r < t; ++r) { top = std::max(top, physical(rc, r)); }
    const int slots = c.ring != 0 ? c.ring : top + 1;
    const std::int64_t head_stride = std::int64_t(slots) * kD, seq_stride = head_stride * kKv;
    std::vector<__half> kc(std::size_t(sequences) * seq_stride), vc(kc.size());
    {
        std::normal_distribution<float> n(0.0f, 1.0f);
        for (auto& v : kc) { v = __float2half(1.5f * n(rng)); }
        for (auto& v : vc) { v = __float2half(n(rng)); }
    }
    const auto q     = normal(std::size_t(t) * kHeads * kD, 0.6f, rng);
    const auto r     = normal(std::size_t(t) * kHeads * kRel, 1.0f, rng);
    const auto table = normal(std::size_t(kExtent) * kRel, 1.0f, rng);
    const std::int64_t gate_stride = kHeads * kD + 32, sink_stride = kHeads + 3;
    const auto gate = normal(std::size_t(t) * gate_stride, 1.0f, rng);
    auto sinks      = normal(std::size_t(t) * sink_stride, 2.0f, rng);
    for (auto& v : sinks) { v += c.sink_offset; }
    const float scale = 1.0f / std::sqrt(float(kD));
    DeviceBuffer dk = upload(kc), dv = upload(vc), dq = upload(q), dr = upload(r), dt = upload(table),
                 dg = upload(gate), ds = upload(sinks),
                 partial(k::attention_workspace_floats(t, kHeads, c.splits) * 4),
                 far(k::attention_workspace_floats(t, kHeads, 1) * 4),
                 out(std::size_t(t) * kHeads * kD * 2),
                 q8(std::size_t(t) * kHeads * kD + std::size_t(t) * (kHeads * kD / 32) * 4);
    RowsDevice rows(rc);
    k::AttentionArgs a;
    a.rows        = rows.rows;
    a.q           = as<float>(dq);
    a.r           = as<float>(dr);
    a.sinks       = as<float>(ds);
    a.sink_stride = sink_stride;
    a.table       = as<float>(dt);
    a.extent      = kExtent;
    a.k_cache     = as<__half>(dk);
    a.v_cache     = as<__half>(dv);
    a.seq_stride  = seq_stride;
    a.head_stride = head_stride;
    a.ring        = c.ring;
    a.window      = c.window;
    a.query_heads = kHeads;
    a.kv_heads    = kKv;
    a.scale       = scale;
    a.splits      = c.splits;
    a.partial     = as<float>(partial);
    a.far         = c.far ? as<float>(far) : nullptr;
    a.row_group   = c.row_group;
    a.gate        = as<float>(dg);
    a.gate_stride = gate_stride;
    a.out         = as<bf16>(out);
    k::attention(a, nullptr);
    sync("attention");
    const auto got = download<bf16>(out, std::size_t(t) * kHeads * kD);
    a.out      = nullptr;
    a.out_q.qs = as<std::int8_t>(q8);
    a.out_q.ds = reinterpret_cast<half2*>(as<std::uint8_t>(q8) + std::size_t(t) * kHeads * kD);
    k::attention(a, nullptr);
    sync("attention q8");
    const int width = kHeads * kD;
    std::vector<std::int8_t> got_q(std::size_t(t) * width);
    std::vector<half2> got_d(std::size_t(t) * (width / 32));
    check(cudaMemcpy(got_q.data(), a.out_q.qs, got_q.size(), cudaMemcpyDeviceToHost), "q8 qs");
    check(cudaMemcpy(got_d.data(), a.out_q.ds, got_d.size() * 4, cudaMemcpyDeviceToHost), "q8 ds");
    // Grouped rows give each row the bits of a block of its own.
    std::size_t differ = 0;
    if (c.row_group > 1) {
        a.row_group = 1;
        k::attention(a, nullptr);
        sync("attention q8, single rows");
        std::vector<std::int8_t> one_q(got_q.size());
        std::vector<half2> one_d(got_d.size());
        check(cudaMemcpy(one_q.data(), a.out_q.qs, one_q.size(), cudaMemcpyDeviceToHost), "q8 qs");
        check(cudaMemcpy(one_d.data(), a.out_q.ds, one_d.size() * 4, cudaMemcpyDeviceToHost), "q8 ds");
        a.out_q = {};
        a.out   = as<bf16>(out);
        k::attention(a, nullptr);
        sync("attention, single rows");
        const auto one = download<bf16>(out, std::size_t(t) * kHeads * kD);
        differ += std::memcmp(one.data(), got.data(), got.size() * sizeof(bf16)) != 0;
        differ += std::memcmp(one_q.data(), got_q.data(), got_q.size()) != 0;
        differ += std::memcmp(one_d.data(), got_d.data(), got_d.size() * sizeof(half2)) != 0;
    }

    const int group = kHeads / kKv;
    Error eb, eq;
    int biased = 0, unbiased = 0;
    for (int row = 0; row < t; ++row) {
        const int s = sequence_of(rc, row), p = rc.positions[row];
        // The keys the row sees: (physical position, logical position).
        std::vector<std::pair<int, int>> keys;
        if (!rc.chunk) {
            for (int j = 0; j <= p; ++j) { keys.emplace_back(j, j); }
        } else {
            for (int j = 0; j < rc.first; ++j) { keys.emplace_back(j, j); }
            for (int o = 0; o <= row; ++o) {
                if (!rc.starts.empty() && o >= rc.shared && rc.starts[o] != rc.starts[row]) { continue; }
                keys.emplace_back(rc.first + o, rc.positions[o]);
            }
        }
        for (int h = 0; h < kHeads; ++h) {
            const int kv = h / group;
            const float* qh = q.data() + (std::size_t(row) * kHeads + h) * kD;
            const float* rh = r.data() + (std::size_t(row) * kHeads + h) * kRel;
            std::vector<double> score;
            std::vector<std::int64_t> at;
            const double sink = sinks[std::size_t(row) * sink_stride + h];
            double m = sink;
            for (const auto& [phys, logical] : keys) {
                const int distance = p - logical;
                if (distance < 0 || (c.window != 0 && distance >= c.window)) { continue; }
                const int slot = c.ring != 0 ? phys % c.ring : phys;
                const std::int64_t base = s * seq_stride + kv * head_stride + std::int64_t(slot) * kD;
                double d = 0;
                for (int i = 0; i < kD; ++i) { d += double(qh[i]) * scale * __half2float(kc[base + i]); }
                if (distance < kExtent) {
                    for (int i = 0; i < kRel; ++i) { d += double(rh[i]) * table[std::size_t(distance) * kRel + i]; }
                    ++biased;
                } else {
                    ++unbiased;
                }
                score.push_back(d);
                at.push_back(base);
                m = std::max(m, d);
            }
            double l = std::exp(sink - m);
            for (auto& v : score) { l += (v = std::exp(v - m)); }
            for (int i = 0; i < kD; ++i) {
                double o = 0;
                for (std::size_t j = 0; j < score.size(); ++j) { o += score[j] * __half2float(vc[at[j] + i]); }
                o = o / l * sigmoid(gate[std::size_t(row) * gate_stride + h * kD + i]);
                const std::size_t idx = std::size_t(row) * width + h * kD + i;
                eb.add(__bfloat162float(got[idx]), o);
                eq.add(double(got_q[idx]) * __low2float(got_d[idx / 32]), o);
            }
        }
    }
    report(eb.rel() < 6e-3 && eq.rel() < 1.5e-2 && differ == 0,
           "attention " + c.name + ", splits " + std::to_string(c.splits) +
               (c.sink_offset != 0.0f ? ", sinks " + num(c.sink_offset) : "") +
               (c.far ? ", far field" : "") +
               (c.row_group > 1 ? ", rows in groups of " + std::to_string(c.row_group) +
                                      (differ == 0 ? " (single rows' bits)" : " (bits differ)")
                                : "") +
               " (" +
               std::to_string(biased) + " biased, " + std::to_string(unbiased) +
               " unbiased scores): bf16 " + num(eb.rel()) + ", q8_1 " + num(eq.rel()));
}

// ---- Residual updates ---------------------------------------------------------------------------

enum class Product { Plain, Fixed, Pairs };

void residual_case(const std::string& name, const RowsCase& c, Product kind, bool start,
                   int mix_states, std::mt19937& rng) {
    const int t = int(c.positions.size());
    const int sequences = c.sequences.empty() ? 1 : *std::max_element(c.sequences.begin(), c.sequences.end()) + 1;
    const std::size_t per = std::size_t(t) * kHidden;
    // The product of each row in FP32 as the kernel reads it.
    std::vector<float> y = normal(per, 1.0f, rng), moe_rows, moe_weights;
    std::vector<unsigned long long> fixed;
    if (kind == Product::Fixed) {
        fixed.resize(per);
        for (std::size_t i = 0; i < per; ++i) {
            const auto v = static_cast<long long>(std::llround(double(y[i]) * 4294967296.0));
            fixed[i]     = static_cast<unsigned long long>(v);
            y[i]         = float(double(v) * (1.0 / 4294967296.0));
        }
    } else if (kind == Product::Pairs) {
        moe_rows    = normal(per * kTopK, 0.5f, rng);
        moe_weights = normal(std::size_t(t) * kTopK, 0.4f, rng);
        for (int r = 0; r < t; ++r) {
            for (int i = 0; i < kHidden; ++i) {
                float s = 0.0f;
                for (int j = 0; j < kTopK; ++j) {
                    const std::size_t pair = std::size_t(r) * kTopK + j;
                    s = std::fma(moe_rows[pair * kHidden + i], moe_weights[pair], s);
                }
                y[std::size_t(r) * kHidden + i] = s;
            }
        }
    }
    const auto conv   = normal(std::size_t(kHidden) * kTaps, 0.4f, rng);
    const std::int64_t seq_hist = std::int64_t(kSlots) * kHidden;
    const auto hist   = normal(std::size_t(sequences) * seq_hist, 1.0f, rng);
    const auto before = normal(per, 2.0f, rng);
    std::vector<std::vector<float>> states;
    std::vector<DeviceBuffer> d_states;
    for (int j = 0; j + 1 < mix_states; ++j) {
        states.push_back(normal(per, 1.0f + float(j), rng));
        d_states.push_back(upload(states.back()));
    }
    const auto queries = normal(std::size_t(2) * kHidden, 0.3f, rng);
    DeviceBuffer d_y = upload(kind == Product::Plain ? y : std::vector<float>{}),
                 d_fixed = upload(fixed), d_rows = upload(moe_rows), d_w = upload(moe_weights),
                 d_conv = upload(conv), d_hist = upload(hist), d_cur = upload(before),
                 d_q = upload(queries), raw(per * 2 * 4);
    RowsDevice rows(c);
    k::ResidualArgs a;
    a.rows        = rows.rows;
    a.y           = kind == Product::Plain ? as<float>(d_y) : nullptr;
    a.fixed       = kind == Product::Fixed ? as<unsigned long long>(d_fixed) : nullptr;
    a.moe_rows    = kind == Product::Pairs ? as<float>(d_rows) : nullptr;
    a.moe_weights = kind == Product::Pairs ? as<float>(d_w) : nullptr;
    a.top_k       = kTopK;
    a.conv        = as<float>(d_conv);
    a.history     = k::History{as<float>(d_hist), seq_hist, kHidden, kSlots};
    a.keep        = !c.chunk;
    a.start       = start;
    a.cur         = as<float>(d_cur);
    a.hidden      = kHidden;
    if (mix_states > 0) {
        for (int j = 0; j + 1 < mix_states; ++j) { a.mix.states[j] = as<float>(d_states[j]); }
        a.mix.states[mix_states - 1] = a.cur;
        a.mix.count        = mix_states;
        a.mix.queries      = as<float>(d_q);
        a.mix.query_count  = 2;
        a.mix.tokens       = t;
        a.mix.hidden       = kHidden;
        a.mix.query_stride = std::int64_t(per);
        a.mix.raw          = as<float>(raw);
    }
    // The split rows over copies of the inputs, before the row kernel changes them.
    DeviceBuffer d_cur2 = upload(before), d_hist2 = upload(hist), raw2(per * 2 * 4),
                 split(k::mix_split_floats(t, kHidden) * 4);
    k::ResidualArgs b = a;
    b.cur             = as<float>(d_cur2);
    b.history.ring    = as<float>(d_hist2);
    b.mix.split       = as<float>(split);
    if (mix_states > 0) {
        b.mix.states[mix_states - 1] = b.cur;
        b.mix.raw                    = as<float>(raw2);
    }
    k::residual(b, nullptr);
    k::residual(a, nullptr);
    sync("residual");
    const auto got   = download<float>(d_cur, per);
    const auto kept  = download<float>(d_hist, std::size_t(sequences) * seq_hist);
    const bool same  = same_bits(download<float>(d_cur2, per), got) &&
                      same_bits(download<float>(d_hist2, std::size_t(sequences) * seq_hist), kept) &&
                      (mix_states == 0 ||
                       same_bits(download<float>(raw2, per * 2), download<float>(raw, per * 2)));
    Error e;
    std::vector<float> want(per);
    std::size_t kept_wrong = 0;
    for (int r = 0; r < t; ++r) {
        const int s = sequence_of(c, r);
        for (int i = 0; i < kHidden; ++i) {
            double in[kTaps];
            for (int lag = 0; lag < kTaps; ++lag) {
                const int src = source_row(c, r, lag), p = c.positions[r] - lag;
                in[lag] = src >= 0   ? y[std::size_t(src) * kHidden + i]
                          : src == -1 ? hist[s * seq_hist + (p % kSlots) * kHidden + i]
                                      : 0.0;
            }
            double v = in[0];
            for (int j = 0; j < kTaps; ++j) { v += conv[std::size_t(i) * kTaps + j] * in[kTaps - 1 - j]; }
            v /= std::sqrt(double(kTaps + 1));
            const double w = start ? v : before[std::size_t(r) * kHidden + i] + v;
            e.add(got[std::size_t(r) * kHidden + i], w);
            want[std::size_t(r) * kHidden + i] = float(w);
        }
        if (!c.chunk) {
            const std::int64_t slot = s * seq_hist + (c.positions[r] % kSlots) * kHidden;
            for (int i = 0; i < kHidden; ++i) {
                kept_wrong += kept[slot + i] != y[std::size_t(r) * kHidden + i] ? 1 : 0;
            }
        }
    }
    double mix_error = 0;
    if (mix_states > 0) {
        auto all = states;
        all.push_back(got); // the mix reads the kernel's own updated rows
        const auto want_mix = mix_oracle(all, queries, 2, t, {}, false);
        const auto got_mix  = download<float>(raw, per * 2);
        Error em;
        for (std::size_t i = 0; i < per * 2; ++i) { em.add(got_mix[i], want_mix[i]); }
        mix_error = em.rel();
    }
    const std::string kinds[] = {"products", "fixed-point sums", "weighted pairs"};
    report(e.rel() < 1e-6 && kept_wrong == 0 && mix_error < 2e-6 && same,
           "residual " + name + " over " + kinds[int(kind)] + (start ? ", a new block" : "") +
               (mix_states ? ", mix of " + std::to_string(mix_states) : "") + ": " + num(e.rel()) +
               (c.chunk ? "" : ", history values wrong " + std::to_string(kept_wrong)) +
               (mix_states ? ", mix " + num(mix_error) : "") +
               (same ? ", split rows the same bits" : ", split rows DIFFER"));

    // A chunk keeps its products and raw inputs afterwards, at the rows' logical positions.
    if (c.chunk) {
        k::keep_products(a, 0, t, nullptr);
        DeviceBuffer d_raw = upload(y), d_h2 = upload(hist);
        const k::History h2{as<float>(d_h2), seq_hist, kHidden, kSlots};
        k::keep_history(as<float>(d_raw), kHidden, kHidden, rows.rows.positions, 0, t, h2, 0, nullptr);
        sync("keep");
        const auto kp = download<float>(d_hist, std::size_t(sequences) * seq_hist);
        const auto kh = download<float>(d_h2, std::size_t(sequences) * seq_hist);
        std::size_t wrong = 0;
        for (int r = std::max(0, t - kSlots); r < t; ++r) {
            const std::int64_t slot = (c.positions[r] % kSlots) * kHidden;
            for (int i = 0; i < kHidden; ++i) {
                wrong += kp[slot + i] != y[std::size_t(r) * kHidden + i] ? 1 : 0;
                wrong += kh[slot + i] != y[std::size_t(r) * kHidden + i] ? 1 : 0;
            }
        }
        report(wrong == 0, "keep_products and keep_history " + name + ": " +
                               std::to_string(wrong) + " values wrong");
    }
}

// ---- Routing ------------------------------------------------------------------------------------

void route_case(int tokens, bool scales, std::mt19937& rng) {
    auto logits = normal(std::size_t(tokens) * kExperts, 2.0f, rng);
    auto bias   = normal(kExperts, 0.05f, rng);
    // Row 0: experts 9 and 3 tie on the selection; the lower id goes first.
    logits[9] = logits[3] = 6.0f;
    bias[9] = bias[3] = 0.5f;
    std::uniform_real_distribution<float> unit(0.5f, 2.0f);
    std::vector<float> gu_scale(kExperts), down_scale(kExperts);
    for (auto& v : gu_scale) { v = unit(rng); }
    for (auto& v : down_scale) { v = unit(rng); }
    DeviceBuffer d_logits = upload(logits), d_bias = upload(bias), d_gu = upload(gu_scale),
                 d_down = upload(down_scale), ids(std::size_t(tokens) * kTopK * 4),
                 weights(std::size_t(tokens) * kTopK * 4), pair(std::size_t(tokens) * kTopK * 4);
    k::RouteArgs a;
    a.hidden        = kHidden;
    a.tokens        = tokens;
    a.bias          = as<float>(d_bias);
    a.experts       = kExperts;
    a.top_k         = kTopK;
    a.scale         = 8.0f;
    a.gate_up_scale = scales ? as<float>(d_gu) : nullptr;
    a.down_scale    = scales ? as<float>(d_down) : nullptr;
    a.logits        = as<float>(d_logits);
    a.ids           = as<std::int32_t>(ids);
    a.weights       = as<float>(weights);
    a.pair_scale    = as<float>(pair);
    k::route(a, nullptr);
    sync("route");
    const auto gi = download<std::int32_t>(ids, std::size_t(tokens) * kTopK);
    const auto gw = download<float>(weights, std::size_t(tokens) * kTopK);
    const auto gp = download<float>(pair, std::size_t(tokens) * kTopK);
    int wrong = 0, near = 0;
    Error e;
    for (int t = 0; t < tokens; ++t) {
        std::vector<std::pair<double, int>> sel(kExperts);
        for (int x = 0; x < kExperts; ++x) {
            sel[x] = {sigmoid(logits[std::size_t(t) * kExperts + x]) + bias[x], x};
        }
        std::stable_sort(sel.begin(), sel.end(), [](const auto& l, const auto& r) {
            return l.first > r.first || (l.first == r.first && l.second < r.second);
        });
        // A margin below FP32's resolution may order either way.
        if (std::abs(sel[kTopK - 1].first - sel[kTopK].first) < 1e-6 && t != 0) {
            ++near;
            continue;
        }
        double sum = 0;
        for (int j = 0; j < kTopK; ++j) { sum += sigmoid(logits[std::size_t(t) * kExperts + sel[j].second]); }
        for (int j = 0; j < kTopK; ++j) {
            const int x = sel[j].second;
            const std::size_t at = std::size_t(t) * kTopK + j;
            wrong += gi[at] != x ? 1 : 0;
            const double w = sigmoid(logits[std::size_t(t) * kExperts + x]) / (sum + 1e-20) * 8.0 *
                             (scales ? down_scale[x] : 1.0);
            e.add(gw[at], w);
            wrong += gp[at] != (scales ? gu_scale[x] : 1.0f) ? 1 : 0;
        }
    }
    report(wrong == 0 && e.rel() < 1e-6 && gi[0] == 3 && gi[1] == 9,
           "route " + std::to_string(tokens) + " row(s)" + (scales ? " with expert scales" : "") +
               ": " + std::to_string(wrong) + " wrong, weights " + num(e.rel()) +
               (near ? ", " + std::to_string(near) + " near ties skipped" : ""));
    if (tokens > k::kSortedRouteRows) { return; }

    // The sorted route: the same choices and weights, and the pairs grouped as
    // gguf::moe_sort_routes groups route_kernel's ids (each expert's pairs as a set: its order
    // within an expert is its own).
    const int pairs = tokens * kTopK;
    DeviceBuffer ids2(std::size_t(pairs) * 4), weights2(std::size_t(pairs) * 4),
        pair2(std::size_t(pairs) * 4), space(gguf::moe_routing_bytes(kExperts, pairs)),
        space2(gguf::moe_routing_bytes(kExperts, pairs));
    const gguf::MoeRouting want = gguf::moe_sort_routes(as<std::int32_t>(ids), pairs, kExperts,
                                                        space.p, nullptr);
    const gguf::MoeRouting view = gguf::moe_sorted_routes(space2.p, pairs, kExperts);
    k::RouteArgs b = a;
    b.ids          = as<std::int32_t>(ids2);
    b.weights      = as<float>(weights2);
    b.pair_scale   = as<float>(pair2);
    b.sorted       = k::RouteSort{const_cast<std::int32_t*>(view.bounds),
                                  const_cast<std::int32_t*>(view.sorted),
                                  const_cast<std::int32_t*>(view.active),
                                  const_cast<std::int32_t*>(view.active_count),
                                  const_cast<std::int32_t*>(view.tile_count),
                                  const_cast<std::int32_t*>(view.tiles), view.tile_capacity};
    k::route(b, nullptr);
    sync("sorted route");
    const auto fetch = [](const std::int32_t* p, std::size_t n) {
        std::vector<std::int32_t> v(n);
        check(cudaMemcpy(v.data(), p, n * 4, cudaMemcpyDeviceToHost), "routing copy");
        return v;
    };
    const auto wb = fetch(want.bounds, kExperts + 1), gb = fetch(view.bounds, kExperts + 1);
    auto ws = fetch(want.sorted, pairs), gs = fetch(view.sorted, pairs);
    const auto wc = fetch(want.active_count, 1), gc = fetch(view.active_count, 1);
    const auto wa = fetch(want.active, wc[0]), ga = fetch(view.active, gc[0]);
    const auto wt = fetch(want.tile_count, 4), gt = fetch(view.tile_count, 4);
    bool tiles = wt == gt;
    for (int w = 0; w < 4 && tiles; ++w) {
        tiles = fetch(want.tiles + std::int64_t(w) * want.tile_capacity, wt[w]) ==
                fetch(view.tiles + std::int64_t(w) * view.tile_capacity, gt[w]);
    }
    for (int x = 0; x < kExperts; ++x) {
        std::sort(ws.begin() + wb[x], ws.begin() + wb[x + 1]);
        std::sort(gs.begin() + wb[x], gs.begin() + wb[x + 1]);
    }
    const bool same = same_bits(download<std::int32_t>(ids2, pairs), gi) &&
                      same_bits(download<float>(weights2, pairs), gw) &&
                      same_bits(download<float>(pair2, pairs), gp) && wb == gb && ws == gs &&
                      wc == gc && wa == ga && tiles;
    report(same, "sorted route " + std::to_string(tokens) + " row(s): choices, weights and the "
                     "pairs by expert " + (same ? "as route and moe_sort_routes give them"
                                                : "DIFFER from route and moe_sort_routes"));
}

// ---- SiTU and the elementwise kernels -----------------------------------------------------------

void situ_case(std::mt19937& rng) {
    constexpr int pairs = 37, stride = 2 * kWidth + 64;
    constexpr float gate_cap = 4.0f, up_cap = 25.0f;
    auto gu = normal(std::size_t(pairs) * stride, 3.0f, rng);
    // Some values past both caps, some near zero and the tanh polynomial's boundary.
    for (int i = 0; i < 64; ++i) {
        gu[std::size_t(i) * 101] = float(i % 2 ? 1 : -1) * 120.0f;
        gu[std::size_t(i) * 103 + 7] = 0.625f * gate_cap * (i % 3 ? 1.0f : -1.0f);
        gu[std::size_t(i) * 107 + 11] = 1e-6f * float(i - 32);
    }
    std::uniform_real_distribution<float> unit(0.25f, 3.0f);
    std::vector<float> scale(pairs);
    for (auto& v : scale) { v = unit(rng); }
    DeviceBuffer d_gu = upload(gu), d_s = upload(scale), out(std::size_t(pairs) * kWidth * 2);
    k::situ_rows(as<float>(d_gu), stride, pairs, kWidth, as<float>(d_s), gate_cap, up_cap,
                 as<bf16>(out), nullptr);
    sync("situ_rows");
    const auto got = download<bf16>(out, std::size_t(pairs) * kWidth);
    std::size_t differ = 0;
    double worst = 0;
    for (int p = 0; p < pairs; ++p) {
        for (int r = 0; r < kWidth; ++r) {
            const float g = ops::silu_detail::mul(gu[std::size_t(p) * stride + r], scale[p]);
            const float u = ops::silu_detail::mul(gu[std::size_t(p) * stride + kWidth + r], scale[p]);
            const bf16 host = __float2bfloat16(mi::situ(g, u, gate_cap, up_cap));
            const bf16 dev  = got[std::size_t(p) * kWidth + r];
            differ += std::memcmp(&host, &dev, 2) != 0 ? 1 : 0;
            const double exact = gate_cap * std::tanh(double(g) / gate_cap) * sigmoid(g) * up_cap *
                                 std::tanh(double(u) / up_cap);
            const double ulp = std::max(std::abs(exact), 1e-30) * std::ldexp(1.0, -8);
            worst = std::max(worst, std::abs(double(__bfloat162float(dev)) - exact) / ulp);
        }
    }
    report(differ == 0 && worst <= 1.01, "situ_rows: " + std::to_string(differ) +
                                             " middles differ from the host's, worst error " +
                                             num(worst) + " BF16 steps");
}

// BF16 experts: `experts` banks of `rows` x `k` BF16 rows, the pairs of `tokens` tokens (top 8 of
// the banks, some marked -1 as another device's), inputs per token (x_div 8) or per pair (1).
void bf16_experts_case(int tokens, int rows, int k_, int x_div, int experts, std::mt19937& rng) {
    const int pairs = tokens * kTopK, inputs = x_div == 1 ? pairs : tokens;
    std::vector<std::int32_t> ids(pairs);
    std::vector<int> order(experts);
    std::iota(order.begin(), order.end(), 0);
    for (int t = 0; t < tokens; ++t) {
        std::shuffle(order.begin(), order.end(), rng);
        for (int j = 0; j < kTopK; ++j) { ids[t * kTopK + j] = order[j]; }
    }
    for (int p = 5; p < pairs; p += 11) { ids[p] = -1; }
    std::vector<bf16> w(std::size_t(experts) * rows * k_), x(std::size_t(inputs) * k_);
    {
        const auto wf = normal(w.size(), 0.05f, rng), xf = normal(x.size(), 1.0f, rng);
        for (std::size_t i = 0; i < w.size(); ++i) { w[i] = __float2bfloat16(wf[i]); }
        for (std::size_t i = 0; i < x.size(); ++i) { x[i] = __float2bfloat16(xf[i]); }
    }
    DeviceBuffer d_w = upload(w), d_x = upload(x), d_ids = upload(ids);
    std::vector<const void*> table(experts);
    for (int e = 0; e < experts; ++e) { table[e] = as<bf16>(d_w) + std::int64_t(e) * rows * k_; }
    DeviceBuffer d_table = upload(table);
    DeviceBuffer space(gguf::moe_routing_bytes(experts, pairs));
    const gguf::MoeRouting routing =
        gguf::moe_sort_routes(as<std::int32_t>(d_ids), pairs, experts, space.p, nullptr);
    const float sentinel = 12345.0f;
    const std::vector<float> fill(std::size_t(pairs) * rows, sentinel);
    DeviceBuffer out = upload(fill);
    const auto run = [&] {
        k::bf16_expert_rows(static_cast<const void* const*>(d_table.p), rows, k_, routing.bounds,
                            routing.sorted, routing.active, routing.active_count,
                            std::min(pairs, experts), as<bf16>(d_x), x_div, as<float>(out), nullptr);
        sync("bf16_expert_rows");
        return download<float>(out, std::size_t(pairs) * rows);
    };
    const auto got = run(), again = run();
    // A prompt call checks every 13th row; decode calls check them all.
    const int step = pairs > 64 ? 13 : 1;
    Error e;
    std::size_t kept_wrong = 0;
    for (int p = 0; p < pairs; ++p) {
        if (ids[p] < 0) {
            for (int r = 0; r < rows; ++r) { kept_wrong += got[std::size_t(p) * rows + r] != sentinel; }
            continue;
        }
        const bf16* xp = x.data() + std::size_t(x_div == 1 ? p : p / kTopK) * k_;
        const bf16* we = w.data() + std::size_t(ids[p]) * rows * k_;
        for (int r = 0; r < rows; r += step) {
            double sum = 0;
            for (int j = 0; j < k_; ++j) {
                sum += double(__bfloat162float(we[std::size_t(r) * k_ + j])) *
                       double(__bfloat162float(xp[j]));
            }
            e.add(got[std::size_t(p) * rows + r], sum);
        }
    }
    report(e.rel() < 2e-6 && kept_wrong == 0 && same_bits(got, again),
           "bf16_expert_rows " + std::to_string(rows) + "x" + std::to_string(k_) + ", " +
               std::to_string(pairs) + " pairs over " + std::to_string(experts) + " experts (x per " +
               (x_div == 1 ? "pair" : "token") + "): rel " + num(e.rel()) + ", " +
               std::to_string(kept_wrong) + " foreign rows changed");

    // bf16_rows_f32 of some table rows (the expert bank as a [experts rows][k] table).
    const std::vector<std::int32_t> pick{0, 7, rows * experts - 1, 3};
    DeviceBuffer d_pick = upload(pick), d_rows(pick.size() * std::size_t(k_ + 16) * 4);
    k::bf16_rows_f32(as<bf16>(d_w), k_, as<std::int32_t>(d_pick), int(pick.size()),
                     as<float>(d_rows), k_ + 16, nullptr);
    sync("bf16_rows_f32");
    const auto table_rows = download<float>(d_rows, pick.size() * std::size_t(k_ + 16));
    std::size_t wrong = 0;
    for (std::size_t i = 0; i < pick.size(); ++i) {
        for (int j = 0; j < k_; ++j) {
            wrong += table_rows[i * (k_ + 16) + j] !=
                     __bfloat162float(w[std::size_t(pick[i]) * k_ + j]);
        }
    }
    report(wrong == 0, "bf16_rows_f32 of " + std::to_string(pick.size()) + " rows of width " +
                           std::to_string(k_) + ": " + std::to_string(wrong) + " values differ");
}

void elementwise_case(std::mt19937& rng) {
    constexpr std::int64_t n = 200058;
    auto x = normal(n, 20.0f, rng);
    DeviceBuffer d_x = upload(x), cap(n * 2), plain(n * 2);
    k::softcap_bf16(as<float>(d_x), as<bf16>(cap), n, 30.0f, nullptr);
    k::f32_to_bf16(as<float>(d_x), as<bf16>(plain), n, nullptr);
    sync("softcap");
    const auto gc = download<bf16>(cap, n), gp = download<bf16>(plain, n);
    std::size_t far = 0, wrong = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        const double want = 30.0 * std::tanh(double(x[i]) / 30.0);
        if (std::abs(__bfloat162float(gc[i]) - want) > std::abs(want) * std::ldexp(1.0, -8) + 1e-30) { ++far; }
        const bf16 h = __float2bfloat16(x[i]);
        wrong += std::memcmp(&h, &gp[i], 2) != 0 ? 1 : 0;
    }
    report(far == 0 && wrong == 0, "softcap_bf16: " + std::to_string(far) +
                                       " logits past one BF16 step; f32_to_bf16: " +
                                       std::to_string(wrong) + " wrong");

    // place_rows, ring_copy, add_fixed: exact.
    const int count = 5;
    std::vector<bf16> rows(std::size_t(count) * kHidden);
    for (std::size_t i = 0; i < rows.size(); ++i) { rows[i] = __float2bfloat16(x[i]); }
    std::vector<float> h = normal(std::size_t(count + 4) * kHidden, 1.0f, rng);
    DeviceBuffer d_rows = upload(rows), d_h = upload(h);
    k::place_rows(as<bf16>(d_rows), count, kHidden, as<float>(d_h) + 2 * kHidden, nullptr);
    sync("place_rows");
    const auto gh = download<float>(d_h, h.size());
    std::size_t placed = 0;
    for (std::size_t i = 0; i < h.size(); ++i) {
        const std::size_t r = i / kHidden;
        const float want = r >= 2 && r < 2 + count ? __bfloat162float(rows[i - 2 * kHidden]) : h[i];
        placed += gh[i] != want ? 1 : 0;
    }

    const int src_ring = 1040, dst_ring = 1072, begin = 2000, end = 2600;
    std::vector<__half> sk(std::size_t(kKv) * src_ring * kD), sv(sk.size());
    for (std::size_t i = 0; i < sk.size(); ++i) {
        sk[i] = __float2half(float(int(i % 977)));
        sv[i] = __float2half(-float(int(i % 1009)));
    }
    std::vector<__half> dk0(std::size_t(kKv) * dst_ring * kD, __float2half(7.0f));
    DeviceBuffer d_sk = upload(sk), d_sv = upload(sv), d_dk = upload(dk0), d_dv = upload(dk0);
    k::RingCopyArgs rc;
    rc.src_k = as<__half>(d_sk);
    rc.src_v = as<__half>(d_sv);
    rc.dst_k = as<__half>(d_dk);
    rc.dst_v = as<__half>(d_dv);
    rc.src_head_stride = std::int64_t(src_ring) * kD;
    rc.dst_head_stride = std::int64_t(dst_ring) * kD;
    rc.src_ring = src_ring;
    rc.dst_ring = dst_ring;
    rc.kv_heads = kKv;
    rc.begin    = begin;
    rc.end      = end;
    k::ring_copy(rc, nullptr);
    sync("ring_copy");
    const auto gk = download<__half>(d_dk, dk0.size()), gv = download<__half>(d_dv, dk0.size());
    std::size_t copied = 0;
    for (int hh = 0; hh < kKv; ++hh) {
        for (int p = begin; p < end; ++p) {
            for (int d = 0; d < kD; ++d) {
                const std::size_t s = (std::size_t(hh) * src_ring + p % src_ring) * kD + d;
                const std::size_t o = (std::size_t(hh) * dst_ring + p % dst_ring) * kD + d;
                copied += std::memcmp(&gk[o], &sk[s], 2) != 0 || std::memcmp(&gv[o], &sv[s], 2) != 0;
            }
        }
    }

    std::vector<unsigned long long> sums(4099), products(4099);
    for (std::size_t i = 0; i < sums.size(); ++i) {
        sums[i]     = 0xFFFFFFFF00000000ull + i * 977;
        products[i] = 0x0000000100000000ull * (i % 5) + i;
    }
    DeviceBuffer d_s = upload(sums), d_p = upload(products), on = upload(std::vector<std::int32_t>{1}),
                 off = upload(std::vector<std::int32_t>{0}), d_s2 = upload(sums);
    k::add_fixed(as<unsigned long long>(d_s), d_p.p, as<std::int32_t>(on), std::int64_t(sums.size()), nullptr);
    k::add_fixed(as<unsigned long long>(d_s2), d_p.p, as<std::int32_t>(off), std::int64_t(sums.size()), nullptr);
    sync("add_fixed");
    const auto gs = download<unsigned long long>(d_s, sums.size());
    const auto gs2 = download<unsigned long long>(d_s2, sums.size());
    std::size_t added = 0;
    for (std::size_t i = 0; i < sums.size(); ++i) {
        added += gs[i] != sums[i] + products[i] ? 1 : 0;
        added += gs2[i] != sums[i] ? 1 : 0;
    }
    report(placed == 0 && copied == 0 && added == 0,
           "place_rows " + std::to_string(placed) + ", ring_copy " + std::to_string(copied) +
               ", add_fixed " + std::to_string(added) + " values wrong");
}

// ---- Routed experts: CPU against GPU ------------------------------------------------------------

std::vector<std::uint8_t> expert_blocks(QType q, int rows, int depth, std::mt19937& rng) {
    const auto block = gguf_block_shape(q);
    std::vector<std::uint8_t> bytes(std::size_t(rows) * (depth / block.elements) * block.bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : bytes) { b = static_cast<std::uint8_t>(byte(rng)); }
    const bool k_quant = block.elements == 256 && q != QType::GGUF_IQ4_XS;
    std::uniform_real_distribution<float> scale(k_quant ? 0.0002f : 0.002f, k_quant ? 0.001f : 0.02f);
    for (std::size_t at = 0; at < bytes.size(); at += block.bytes) {
        if (q == QType::GGUF_NVFP4) {
            // UE4M3 scale bytes 1..31 (2^-9 .. 2^-4, subnormals included).
            for (int s = 0; s < 4; ++s) { bytes[at + s] = static_cast<std::uint8_t>(1 + byte(rng) % 31); }
            continue;
        }
        const __half h = __float2half(scale(rng));
        std::memcpy(&bytes[at], &h, 2);
        if (q == QType::GGUF_Q4_K || q == QType::GGUF_Q5_K) { std::memcpy(&bytes[at + 2], &h, 2); }
    }
    return bytes;
}

void cpu_experts_case(const char* name, QType gate_up_type, QType down_type, bool scales,
                      int tokens, std::mt19937& rng) {
    constexpr int distinct = 3, pool_size = 20;
    const int pairs     = tokens * kTopK;
    const auto gu_block = gguf_block_shape(gate_up_type), down_block = gguf_block_shape(down_type);
    const std::int64_t gu_row   = std::int64_t(kHidden / gu_block.elements) * gu_block.bytes;
    const std::int64_t down_row = std::int64_t(kWidth / down_block.elements) * down_block.bytes;
    std::vector<std::vector<std::uint8_t>> gate_up, down;
    std::vector<DeviceBuffer> banks;
    for (int i = 0; i < distinct; ++i) {
        gate_up.push_back(expert_blocks(gate_up_type, 2 * kWidth, kHidden, rng));
        down.push_back(expert_blocks(down_type, kHidden, kWidth, rng));
        banks.push_back(upload(gate_up.back()));
        banks.push_back(upload(down.back()));
    }
    std::vector<const void*> table(2 * std::size_t(kExperts));
    for (int e = 0; e < kExperts; ++e) {
        table[std::size_t(e)]            = banks[2 * std::size_t(e % distinct)].p;
        table[std::size_t(kExperts + e)] = banks[2 * std::size_t(e % distinct) + 1].p;
    }
    DeviceBuffer d_table = upload(table);
    const auto* tables   = static_cast<const void* const*>(d_table.p);
    const gguf::MoeTable gu_table{tables, gu_row, 2 * kWidth, kHidden};
    const gguf::MoeTable down_table{tables + kExperts, down_row, kHidden, kWidth};

    std::normal_distribution<float> n(0.0f, 1.0f);
    std::vector<bf16> x(std::size_t(tokens) * kHidden);
    for (auto& v : x) { v = __float2bfloat16(2.0f * n(rng)); }
    std::vector<std::int32_t> ids(static_cast<std::size_t>(pairs));
    std::vector<float> weights(ids.size()), pair_scale(ids.size());
    std::vector<float> gu_scales(kExperts, 1.0f);
    std::uniform_real_distribution<float> unit(0.05f, 2.5f);
    if (scales) {
        for (auto& v : gu_scales) { v = unit(rng); }
    }
    std::vector<int> pool(pool_size);
    std::iota(pool.begin(), pool.end(), 41);
    for (int t = 0; t < tokens; ++t) {
        std::shuffle(pool.begin(), pool.end(), rng);
        for (int j = 0; j < kTopK; ++j) {
            const std::size_t at = std::size_t(t) * kTopK + j;
            ids[at]        = pool[std::size_t(j)];
            weights[at]    = unit(rng);
            pair_scale[at] = gu_scales[std::size_t(ids[at])];
        }
    }
    DeviceBuffer d_x = upload(x), d_ids = upload(ids), d_w = upload(weights), d_ps = upload(pair_scale);
    DeviceBuffer d_xq(gguf::vector_activation_bytes(kHidden, tokens));
    gguf::quantize_vector_activation(as<bf16>(d_x), kHidden, tokens, nullptr, d_xq.p, nullptr);
    DeviceBuffer d_routing(gguf::moe_routing_bytes(kExperts, pairs));
    const gguf::MoeRouting routing =
        gguf::moe_sort_routes(as<std::int32_t>(d_ids), pairs, kExperts, d_routing.p, nullptr);

    const std::size_t sums = std::size_t(tokens) * kHidden;
    DeviceBuffer d_gu(std::size_t(pairs) * 2 * kWidth * 4), d_middle(std::size_t(pairs) * kWidth * 2),
        d_act(gguf::vector_activation_bytes(kWidth, pairs)), d_fixed(sums * 8);
    const int chunk = tokens <= 1 ? 1 : tokens <= 2 ? 2 : tokens <= 4 ? 4 : 8;
    const auto gpu  = [&](const gguf::MoeRouting& part) {
        check(cudaMemset(d_fixed.p, 0, sums * 8), "zero the sums");
        check(cudaMemset(d_gu.p, 0, std::size_t(pairs) * 2 * kWidth * 4), "zero the rows");
        gguf::MoeOutput rows;
        rows.f32 = as<float>(d_gu);
        gguf::MoeOutput out;
        out.weighted = as<unsigned long long>(d_fixed);
        out.weights  = as<float>(d_w);
        const int max_active = std::min(pairs, kExperts);
        gguf::moe_vector_product(ops::detail::gguf_type(gate_up_type), gu_table, part, max_active,
                                 kTopK, kTopK, d_xq.p, tokens, rows, chunk, nullptr);
        k::situ_rows(as<float>(d_gu), 2 * kWidth, pairs, kWidth, as<float>(d_ps), 4.0f, 25.0f,
                     as<bf16>(d_middle), nullptr);
        gguf::quantize_vector_activation(as<bf16>(d_middle), kWidth, pairs, nullptr, d_act.p,
                                         nullptr);
        gguf::moe_vector_product(ops::detail::gguf_type(down_type), down_table, part, max_active, 1,
                                 kTopK, d_act.p, pairs, out, chunk, nullptr);
    };
    gpu(routing);
    sync("routed experts");
    const auto whole = download<unsigned long long>(d_fixed, sums);

    mi::CpuExperts cpu({mi::CpuExpertLayer{gate_up_type, down_type, gu_row, down_row}},
                       {scales ? gu_scales : std::vector<float>{}}, kHidden, kWidth, kTopK, 4.0f,
                       25.0f);
    std::vector<std::byte> inputs(cpu.input_bytes()), products(cpu.product_bytes());
    d_xq.copy_to_host(inputs.data(), cpu.activation_bytes(tokens));
    d_w.copy_to_host(inputs.data() + cpu.activation_bytes(tokens), cpu.weight_bytes(tokens));
    const auto on_cpu = [&](const std::vector<std::int32_t>& experts) {
        std::vector<std::vector<const std::byte*>> blobs;
        for (const std::int32_t e : experts) {
            const auto b = std::size_t(e % distinct);
            blobs.push_back({reinterpret_cast<const std::byte*>(gate_up[b].data()),
                             reinterpret_cast<const std::byte*>(down[b].data())});
        }
        ExpertCpuProducts::Work work;
        work.experts  = experts;
        work.blobs    = blobs;
        work.ids      = ids;
        work.inputs   = inputs.data();
        work.products = products.data();
        (void)cpu.compute(work);
        std::vector<unsigned long long> out(sums);
        std::memcpy(out.data(), products.data(), sums * 8);
        return out;
    };
    std::vector<std::int32_t> active(ids.begin(), ids.end()), even, odd;
    std::sort(active.begin(), active.end());
    active.erase(std::unique(active.begin(), active.end()), active.end());
    for (const std::int32_t e : active) { (e % 2 == 0 ? even : odd).push_back(e); }
    const auto all_cpu = on_cpu(active);
    std::size_t wrong = 0, zero = 0;
    for (std::size_t i = 0; i < sums; ++i) {
        wrong += all_cpu[i] != whole[i] ? 1 : 0;
        zero += whole[i] == 0 ? 1 : 0;
    }
    std::vector<std::int32_t> set{std::int32_t(even.size())};
    set.insert(set.end(), even.begin(), even.end());
    DeviceBuffer d_set = upload(set), d_products = upload(on_cpu(odd)),
                 flag = upload(std::vector<std::int32_t>{1});
    gguf::MoeRouting part = routing;
    part.active_count     = as<std::int32_t>(d_set);
    part.active           = as<std::int32_t>(d_set) + 1;
    gpu(part);
    k::add_fixed(as<unsigned long long>(d_fixed), d_products.p, as<std::int32_t>(flag),
                 std::int64_t(sums), nullptr);
    sync("the even experts");
    const auto halves = download<unsigned long long>(d_fixed, sums);
    std::size_t split_wrong = 0;
    for (std::size_t i = 0; i < sums; ++i) { split_wrong += halves[i] != whole[i] ? 1 : 0; }
    report(wrong == 0 && split_wrong == 0 && zero < sums / 100,
           std::string("CPU experts ") + name + (scales ? " with expert scales" : "") + ", " +
               std::to_string(tokens) + " token(s): " + std::to_string(wrong) +
               " sums differ all on the CPU, " + std::to_string(split_wrong) + " split, " +
               std::to_string(zero) + " zero");
}

std::vector<int> iota(int first, int count) {
    std::vector<int> v(count);
    std::iota(v.begin(), v.end(), first);
    return v;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "no CUDA device; skipped\n";
        return 77;
    }
    std::mt19937 rng(20261121);
    try {
        rms_norm_case(1, true, rng);
        rms_norm_case(9, false, rng);
        for (const auto& [count, queries] : std::vector<std::pair<int, int>>{{1, 1}, {3, 2}, {5, 4}, {8, 3}}) {
            mix_case(count, queries, 6, rng);
        }
        for (const int n : {1, 3, 8}) { gemm_case(320, n, kHidden, 0, 0, rng); }
        gemm_case(192, 5, 2 * kHidden, 2 * kHidden + 8, 200, rng);
        gemm_case(256, 24, kHidden, 0, 0, rng);
        // The controllers' shapes: down 64 x 2 hidden, forget and gain hidden x 64.
        for (const int n : {1, 4, 8}) {
            gemm_batch_case(64, n, 2 * kHidden, 3, rng);
            gemm_batch_case(kHidden, n, 64, 6, rng);
        }
        gemm_batch_case(kHidden, 11, 64, 2, rng);
        control_case(1, 3, rng);
        control_case(3, 7, rng);

        query_case(false, {0, 5, 1000}, 1.0f, rng);
        query_case(true, {0, 8190, 8191, 8192, 20000, 1048575}, 4.0f, rng);

        // Decode rows of three sequences; a verification's four rows of one sequence; a prompt
        // chunk after 50 positions; one at position 0; a chunk's branches.
        RowsCase decode{{0, 2, 100}, {0, 1, 2}};
        RowsCase verify{{40, 41, 42, 43, 7}, {1, 1, 1, 1, 0}};
        RowsCase chunk{iota(50, 12), {}, true, 50};
        RowsCase origin{iota(0, 9), {}, true, 0};
        RowsCase branches{{}, {}, true, 30, 4, {}};
        branches.positions = {30, 31, 32, 33, 34, 35, 36, 34, 35};
        branches.starts    = {0, 0, 0, 0, 4, 4, 4, 7, 7};
        for (const bool dynamic : {false, true}) {
            kv_case("decode rows", decode, dynamic, rng);
            kv_case("verification rows", verify, dynamic, rng);
            kv_case("prompt chunk", chunk, dynamic, rng);
            kv_case("chunk at position 0", origin, dynamic, rng);
            kv_case("branches", branches, dynamic, rng);
        }
        kv_case("decode rows", decode, true, rng, true);
        kv_case("verification rows", verify, true, rng, true);
        kv_case("prompt chunk", chunk, true, rng, true);

        const int ring = kWindow + 64;
        attention_case({"window ring, decode", {{5, 1100, 3000}, {0, 1, 2}}, ring, kWindow, 1}, rng);
        attention_case({"window ring, decode", {{5, 1100, 3000}, {0, 1, 2}}, ring, kWindow, 9}, rng);
        attention_case({"full cache, decode", {{0, 900, 1500}, {0, 1, 2}}, 0, 0, 1}, rng);
        attention_case({"full cache, decode", {{0, 900, 1500}, {0, 1, 2}}, 0, 0, 48}, rng);
        attention_case({"full cache, verification", {{2000, 2001, 2002, 2003}, {0, 0, 0, 0}}, 0, 0, 16}, rng);
        // Grouped verification rows: one group, groups across two sequences, rows whose keys
        // split into different counts (126-129 at 16 splits) and lengths (8,191-8,194 at 128).
        attention_case({"full cache, verification", {{2000, 2001, 2002, 2003}, {0, 0, 0, 0}}, 0, 0, 16, 0.0f, false, 4}, rng);
        attention_case({"full cache, verification", {{2000, 2001, 2002, 1500, 1501, 1502, 1503, 1504}, {0, 0, 0, 1, 1, 1, 1, 1}}, 0, 0, 128, 0.0f, false, 4}, rng);
        attention_case({"full cache, verification", {{126, 127, 128, 129}, {0, 0, 0, 0}}, 0, 0, 16, 0.0f, false, 4}, rng);
        attention_case({"full cache, verification", {{8190, 8191, 8192, 8193}, {0, 0, 0, 0}}, 0, 0, 128, 0.0f, false, 4}, rng);
        attention_case({"full cache, verification", {{3000, 3001}, {0, 0}}, 0, 0, 128, -30.0f, false, 2}, rng);
        // More splits than the combine's 128 threads: 300 at 20,000 keys, 512 at 40,000.
        attention_case({"full cache, decode", {{20000}, {0}}, 0, 0, 300}, rng);
        attention_case({"full cache, verification", {{40000, 40001, 40002}, {0, 0, 0}}, 0, 0, 512, 0.0f, false, 3}, rng);
        attention_case({"window ring, verification", {{2000, 2001, 2002}, {0, 0, 0}}, ring, kWindow, 4}, rng);
        attention_case({"full cache, decode", {{700, 1300}, {0, 1}}, 0, 0, 7, -30.0f}, rng);
        attention_case({"full cache, decode", {{700, 1300}, {0, 1}}, 0, 0, 7, 12.0f}, rng);
        attention_case({"window ring, chunk after 1100", {iota(1100, 37), {}, true, 1100}, ring, kWindow, 1}, rng);
        attention_case({"full cache, chunk after 600", {iota(600, 21), {}, true, 600}, 0, 0, 1}, rng);
        attention_case({"full cache, chunk at 0", {iota(0, 64), {}, true, 0}, 0, 0, 1}, rng);
        attention_case({"full cache, branches", branches, 0, 0, 1}, rng);
        // Chunks far enough in for a far field: one group of 32 rows and part of one, three groups
        // with a short last one, and branches after 1,600 positions.
        attention_case({"full cache, chunk after 1100", {iota(1100, 37), {}, true, 1100}, 0, 0, 1, 0.0f, true}, rng);
        attention_case({"full cache, chunk after 3000", {iota(3000, 70), {}, true, 3000}, 0, 0, 1, 0.0f, true}, rng);
        attention_case({"full cache, chunk after 3000", {iota(3000, 70), {}, true, 3000}, 0, 0, 1, -30.0f, true}, rng);
        {
            RowsCase far_branches{{}, {}, true, 1600, 4, {}};
            far_branches.positions = {1600, 1601, 1602, 1603, 1604, 1605, 1606, 1604, 1605};
            far_branches.starts    = {0, 0, 0, 0, 4, 4, 4, 7, 7};
            attention_case({"full cache, branches after 1600", far_branches, 0, 0, 1, 0.0f, true}, rng);
        }
        attention_case({"window ring, branches", branches, ring, kWindow, 1}, rng);

        residual_case("decode rows", decode, Product::Plain, false, 0, rng);
        residual_case("verification rows", verify, Product::Fixed, false, 3, rng);
        residual_case("decode rows", decode, Product::Pairs, true, 2, rng);
        residual_case("prompt chunk", chunk, Product::Plain, true, 4, rng);
        residual_case("prompt chunk", chunk, Product::Fixed, false, 1, rng);
        residual_case("chunk at position 0", origin, Product::Pairs, false, 0, rng);

        route_case(1, false, rng);
        route_case(3, true, rng);
        route_case(8, false, rng);
        route_case(64, true, rng);
        situ_case(rng);
        elementwise_case(rng);
        for (const int tokens : {1, 8, 300}) {
            bf16_experts_case(tokens, 2 * kWidth, kHidden, kTopK, 16, rng);
            bf16_experts_case(tokens, kHidden, kWidth, 1, 16, rng);
        }

        cpu_experts_case("NVFP4/NVFP4", QType::GGUF_NVFP4, QType::GGUF_NVFP4, true, 1, rng);
        cpu_experts_case("NVFP4/NVFP4", QType::GGUF_NVFP4, QType::GGUF_NVFP4, true, 5, rng);
        cpu_experts_case("Q5_K/Q5_0", QType::GGUF_Q5_K, QType::GGUF_Q5_0, false, 3, rng);
        cpu_experts_case("Q4_K/Q5_0", QType::GGUF_Q4_K, QType::GGUF_Q5_0, false, 8, rng);
        cpu_experts_case("IQ4_XS/IQ4_NL", QType::GGUF_IQ4_XS, QType::GGUF_IQ4_NL, false, 2, rng);
    } catch (const std::exception& e) {
        std::cout << "FAIL " << e.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "inkling2 kernels: OK\n" : "inkling2 kernels: FAILED\n");
    return failures == 0 ? 0 : 1;
}
