// DeepSeek-V4.1-Flash's own kernels against FP64 oracles, at the release's shapes (hidden 5120,
// four streams, 64 query heads over one 512-value latent with a 64-value rotated tail, a
// 128-position window, 32 indexer heads of 128 values, 384 experts choosing 6 of width 2304, Engram
// rows of 256 values). The caches hold packed codes and scales (the layouts in kernels.h); the tests
// decode them on the host and fill read-only caches with random codes:
//  - gemm_f32 by the GEMV kernel (one to eight rows, whole and partial float4s) and by SGEMM.
//  - expand_streams; hc_split (pre, post and the Sinkhorn comb after 1, 3 and 20 iterations);
//    hc_pre_norm with a pre mix and with stream 0; hc_post; rms_rows of strided rows.
//  - kv_post into decode rings of several sequences, at a round's own slots and into a chunk's
//    staging ring; rotate_heads of FP32 rows with the indexer's fp4 rounding and of BF16 rows in
//    place.
//  - compress of ratio 1 and 2: pairs within the call, the earlier row from the sequence's tail at
//    a chunk's odd first position and in decode, rows that complete no group untouched;
//    compress_tail. latent_post: the indexer keys and the compressed latents at their groups' first
//    positions, nothing for the rows that complete no group.
//  - index_scores of decode rows and of a chunk's rows in fours, with and without candidate blocks;
//    select_topk and select_candidates with ties (both zeros among them), -inf and NaN scores,
//    fewer selectable positions or blocks than chosen, and rows over several 1024-value chunks;
//    select_candidates' block maxima in rows of the blocks a row sees, under masks of a
//    1,000,000-position context, and its refusal of a shorter scratch row.
//    candidate_lists against each group's union of masks; index_scores and select_topk over listed
//    blocks against the same rows over every visible position, bit for bit.
//  - sparse_attention over the window rings and the selected latents: window-only blocks, one to
//    40 splits (empty ones among them), a round's rows of one sequence, a chunk's staging ring,
//    a chunk's rows over a window floor (a prefill tail's first rows), 1024 latents, sinks far
//    below and above the scores; attention_output in both layouts.
//  - the grouped output projection of one to eight tokens as one vector product over the groups:
//    every value the bits of its group's own vector product (Q5_K).
//  - route (ties to the lower id, the softplus branch past 20, image rows by the image routing
//    bias; a NaN row takes experts 0..top_k-1, the other rows select as without it),
//    clamped_swiglu (the clamps, rows scaled per token), moe_combine (weighted rows and 2^-32
//    fixed-point sums).
//  - engram_rows (IQ4_NL), engram_gate (image rows passed through); the Engram hash ids of dead
//    (image) positions and of the positions after them, in one call and across calls.
//  - place_rows: an image's BF16 rows into stream 0 of their positions, nothing else written.
//  - guard_columns (in one call and in calls of six columns) and unscale_columns, ring_copy,
//    f32_to_bf16.
//  - the DSpark drafter's: sparse_attention of a draft block (the window and the block's own
//    rows), stream_mean, markov_step (a chain, ties, the domain) and dspark_confidence.
//  - the routed experts of a call on the CPU (CpuExperts) against the GPU's vector kernels over
//    IQ1_S, IQ1_M, IQ2_XXS and IQ2_S banks: the same 2^-32 fixed-point sums bit for bit, all on
//    the CPU and half on each (add_fixed).
// A rounded output (BF16, fp8, fp4) is compared with the oracle's rounding of its exact value. The
// two may differ where the FP32 arithmetic lands on the other side of a rounding boundary: rarely,
// and by at most one or two steps of the format there. Values that pass through without arithmetic
// before an exact rounding must match exactly.

#include "core/device_buffer.h"
#include "core/qtype.h"
#include "core/weight.h"
#include "models/deepseek_v41/cpu_experts.h"
#include "models/deepseek_v41/engram.h"
#include "models/deepseek_v41/kernels.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"
#include "ops/moe_experts/gguf_expert_cpu.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
namespace k    = ninfer::models::deepseek_v41::kernels;
namespace md   = ninfer::models::deepseek_v41;
namespace gguf = ninfer::ops::gguf;
using bf16  = __nv_bfloat16;

constexpr int kHidden = 5120, kStreams = 4, kLatent = 512, kRope = 64, kIndex = 128;
constexpr int kHeads = 64, kGroups = 8, kIndexHeads = 32, kWindow = 128, kRing = 256;
constexpr int kExperts = 384, kTopK = 6, kWidth = 2304;
constexpr float kEps = 1e-20f, kHcEps = 1e-6f;
constexpr std::uint16_t kSentinel = 0xFFFF; // a BF16 NaN that no kernel writes

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

// The larger of two errors, NaN the largest.
void worse(double& worst, double error) {
    if (!(error <= worst)) { worst = error; }
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

float bf(double x) { return __bfloat162float(__float2bfloat16(float(x))); }

float val(bf16 x) { return __bfloat162float(x); }

std::uint16_t bits(bf16 x) {
    std::uint16_t b = 0;
    std::memcpy(&b, &x, 2);
    return b;
}

bf16 sentinel() {
    bf16 x;
    std::memcpy(&x, &kSentinel, 2);
    return x;
}

std::vector<bf16> random_bf16(std::size_t n, float sigma, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, sigma);
    std::vector<bf16> v(n);
    for (auto& x : v) { x = __float2bfloat16(normal(rng)); }
    return v;
}

std::vector<std::int32_t> ints(const std::vector<int>& v) { return {v.begin(), v.end()}; }

// ---- Packed caches (kernels.h), decoded here from the formats' definitions ----------------------

constexpr std::uint8_t kSentinelByte = 0xA5; // the bytes of a cache row no kernel writes

float e4m3_of(unsigned c) {
    const int e = int((c >> 3) & 15U), m = int(c & 7U);
    const float a = e == 0 ? std::ldexp(float(m), -9) : std::ldexp(float(8 + m), e - 10);
    return (c & 0x80U) != 0 ? -a : a;
}
float e2m1_of(unsigned n) {
    static constexpr float kMagnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (n & 8U) != 0 ? -kMagnitude[n & 7U] : kMagnitude[n & 7U];
}
float e8m0_of(unsigned c) { return std::ldexp(1.0f, int(c) - 127); }

// The 512 values of a window row, of a compressed latent row, and the 128 of an indexer key.
std::vector<float> window_values(const std::uint8_t* row) {
    std::vector<float> v(kLatent);
    for (int i = 0; i < kLatent; ++i) {
        v[std::size_t(i)] = e4m3_of(row[i]) * e8m0_of(row[kLatent + i / 32]);
    }
    return v;
}
std::vector<float> latent_values(const std::uint8_t* row) {
    std::vector<float> v(kLatent);
    for (int i = 0; i < kLatent; ++i) {
        v[std::size_t(i)] = e2m1_of((row[i / 2] >> (4 * (i & 1))) & 15U) * e4m3_of(row[kLatent / 2 + i / 16]);
    }
    return v;
}
std::vector<float> key_values(const std::uint8_t* codes, const std::uint8_t* scales) {
    std::vector<float> v(kIndex);
    for (int i = 0; i < kIndex; ++i) {
        v[std::size_t(i)] = e2m1_of((codes[i / 2] >> (4 * (i & 1))) & 15U) * e8m0_of(scales[i / 32]);
    }
    return v;
}

// Random rows of each format: any finite code, scales over a few binades.
std::vector<std::uint8_t> random_window_rows(std::size_t rows, std::mt19937& rng) {
    std::vector<std::uint8_t> out(rows * k::kWindowRowBytes);
    for (std::size_t r = 0; r < rows; ++r) {
        std::uint8_t* row = out.data() + r * k::kWindowRowBytes;
        for (int i = 0; i < kLatent; ++i) {
            const auto c = std::uint8_t(rng());
            row[i]       = (c & 0x7FU) == 0x7FU ? std::uint8_t(c - 1) : c; // no NaN
        }
        for (int g = 0; g < kLatent / 32; ++g) { row[kLatent + g] = std::uint8_t(119 + rng() % 4); }
    }
    return out;
}
std::vector<std::uint8_t> random_latent_rows(std::size_t rows, std::mt19937& rng) {
    std::vector<std::uint8_t> out(rows * k::kLatentRowBytes);
    for (std::size_t r = 0; r < rows; ++r) {
        std::uint8_t* row = out.data() + r * k::kLatentRowBytes;
        for (int i = 0; i < kLatent / 2; ++i) { row[i] = std::uint8_t(rng()); }
        for (int g = 0; g < kLatent / 16; ++g) { row[kLatent / 2 + g] = std::uint8_t(0x18 + rng() % 24); }
    }
    return out;
}
std::vector<std::uint8_t> random_key_codes(std::size_t rows, std::mt19937& rng) {
    std::vector<std::uint8_t> out(rows * k::kKeyRowBytes);
    for (auto& c : out) { c = std::uint8_t(rng()); }
    return out;
}
std::vector<std::uint8_t> random_key_scales(std::size_t rows, std::mt19937& rng) {
    std::vector<std::uint8_t> out(rows * k::kKeyScaleBytes);
    for (auto& c : out) { c = std::uint8_t(123 + rng() % 5); }
    return out;
}

// Rounded outputs against the oracle's rounding of the exact values: how many differ, and the
// largest difference in steps of the output format at the oracle's value.
struct Tally {
    std::size_t values = 0, off = 0;
    double worst = 0.0;

    void add(double got, double want, double step) {
        ++values;
        if (got == want) { return; }
        ++off;
        worse(worst, std::isfinite(got) ? std::fabs(got - want) / step : INFINITY);
    }

    [[nodiscard]] bool within(double fraction, double steps) const {
        return double(off) <= fraction * double(values) && worst <= steps;
    }

    [[nodiscard]] std::string str() const {
        return std::to_string(off) + "/" + std::to_string(values) + " off by " + num(worst) +
               " steps";
    }
};

double bf16_step(double x) {
    return std::ldexp(1.0, std::max(std::ilogb(std::max(std::fabs(x), 1e-37)), -126) - 7);
}

// The release's roundings of a group in place, in FP32 arithmetic as its kernels compute them, and
// each output's step.
float pow2_ceil(float x) {
    int e         = 0;
    const float m = std::frexp(x, &e);
    return std::ldexp(1.0f, m == 0.5f ? e - 1 : e);
}

double e4m3_step(float q) {
    return std::ldexp(1.0, std::max(std::ilogb(std::max(std::fabs(q), 1e-30f)), -6) - 3);
}

double e2m1_step(float q) {
    return std::ldexp(1.0, std::max(std::ilogb(std::max(std::fabs(q), 1e-30f)), 0) - 1);
}

float group_amax(const float* v, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) { amax = std::max(amax, std::fabs(v[i])); }
    return amax;
}

void fp8_round(float* v, double* step) {
    const float s = pow2_ceil(std::max(group_amax(v, 32), 1e-4f) * (1.0f / 448.0f));
    for (int i = 0; i < 32; ++i) {
        const float q = k::round_e4m3(std::clamp(v[i] / s, -448.0f, 448.0f));
        v[i]          = q * s;
        step[i]       = s * e4m3_step(q);
    }
}

void fp4_e8m0_round(float* v, double* step) {
    const float s = pow2_ceil(std::max(group_amax(v, 32), 6.0f * 1.17549435e-38f) * (1.0f / 6.0f));
    for (int i = 0; i < 32; ++i) {
        const float q = k::round_e2m1(std::clamp(v[i] / s, -6.0f, 6.0f));
        v[i]          = q * s;
        step[i]       = s * e2m1_step(q);
    }
}

void fp4_e4m3_round(float* v, double* step) {
    const float s =
        k::round_e4m3(std::min(std::max(group_amax(v, 16), 6.0f * 0.001953125f) / 6.0f, 448.0f));
    for (int i = 0; i < 16; ++i) {
        const float q = k::round_e2m1(std::clamp(v[i] / s, -6.0f, 6.0f));
        v[i]          = q * s;
        step[i]       = s * e2m1_step(q);
    }
}

std::vector<float> frequencies(double theta) {
    std::vector<float> f(kRope / 2);
    for (int i = 0; i < kRope / 2; ++i) {
        f[i] = float(1.0 / std::pow(theta, double(2 * i) / kRope));
    }
    return f;
}

// The 64-value tail of a row turned at `position`: pairs (2i, 2i + 1) by position * freq[i], the
// angle formed in FP32 as the kernels and the release form it, each value then rounded to BF16.
void rotate_tail(float* tail, int position, const std::vector<float>& freq, bool inverse) {
    for (int i = 0; i < kRope / 2; ++i) {
        const double angle = double(float(position) * freq[std::size_t(i)]);
        const double c = std::cos(angle), s = (inverse ? -1.0 : 1.0) * std::sin(angle);
        const double x0 = tail[2 * i], x1 = tail[2 * i + 1];
        tail[2 * i]     = bf(x0 * c - x1 * s);
        tail[2 * i + 1] = bf(x0 * s + x1 * c);
    }
}

// bf16(rms(bf16(x)) * w) of n values.
template <typename T>
std::vector<float> norm_row(const T* x, int n, const float* w) {
    std::vector<float> v(static_cast<std::size_t>(n));
    double ss = 0.0;
    for (int i = 0; i < n; ++i) {
        v[std::size_t(i)] = bf(double(x[i]));
        ss += double(v[std::size_t(i)]) * v[std::size_t(i)];
    }
    const double inv = 1.0 / std::sqrt(ss / n + kEps);
    for (int i = 0; i < n; ++i) { v[std::size_t(i)] = bf(v[std::size_t(i)] * inv * w[i]); }
    return v;
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// ---- Products and hyper-connections ------------------------------------------------------------

void gemm(int m, int n, int depth, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> w(std::size_t(m) * depth), x(std::size_t(n) * depth);
    for (auto& v : w) { v = normal(rng); }
    for (auto& v : x) { v = normal(rng); }
    DeviceBuffer dw = upload(w), dx = upload(x), out(std::size_t(n) * m * 4);
    k::gemm_f32(m, n, depth, as<float>(dw), as<float>(dx), as<float>(out), nullptr);
    sync("gemm_f32");
    const auto got = download<float>(out, std::size_t(n) * m);
    double worst   = 0.0;
    for (int t = 0; t < n; ++t) {
        for (int r = 0; r < m; ++r) {
            double sum = 0.0, magnitude = 0.0;
            for (int i = 0; i < depth; ++i) {
                const double p =
                    double(w[std::size_t(r) * depth + i]) * x[std::size_t(t) * depth + i];
                sum += p;
                magnitude += std::fabs(p);
            }
            worse(worst, std::fabs(got[std::size_t(t) * m + r] - sum) / magnitude);
        }
    }
    report(worst <= 1e-5, "gemm_f32 " + std::to_string(m) + "x" + std::to_string(depth) + ", " +
                              std::to_string(n) + " rows: worst " + num(worst));
}

void hyper_connections(int tokens, int iterations, std::mt19937& rng) {
    constexpr int S = kStreams, H = kHidden, M = (2 + S) * S;
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    const std::size_t width = std::size_t(S) * H;
    const std::string rows  = std::to_string(tokens) + " rows";
    std::vector<float> h(tokens * width);
    for (auto& v : h) { v = normal(rng); }

    {
        DeviceBuffer d = upload(h);
        k::expand_streams(as<float>(d), tokens, S, H, nullptr);
        sync("expand_streams");
        const auto got    = download<float>(d, h.size());
        std::size_t wrong = 0;
        for (int t = 0; t < tokens; ++t) {
            for (std::size_t i = 0; i < width; ++i) {
                wrong += got[t * width + i] != h[t * width + i % H] ? 1 : 0;
            }
        }
        report(wrong == 0, "expand_streams " + rows + ": " + std::to_string(wrong) + " wrong");
    }

    // hc_split: the mixes take values that put m = mixes * rsqrt(mean(h^2)) near N(0, 2).
    std::vector<float> mixes(std::size_t(tokens) * M), base(M), scale(3);
    std::vector<double> rstd(static_cast<std::size_t>(tokens));
    for (int t = 0; t < tokens; ++t) {
        double ss = 0.0;
        for (std::size_t i = 0; i < width; ++i) {
            ss += double(h[t * width + i]) * h[t * width + i];
        }
        rstd[std::size_t(t)] = 1.0 / std::sqrt(ss / double(width) + kEps);
        for (int j = 0; j < M; ++j) {
            mixes[std::size_t(t) * M + j] = float(2.0 * normal(rng) / rstd[std::size_t(t)]);
        }
    }
    for (auto& v : base) { v = normal(rng); }
    for (auto& v : scale) { v = 0.5f + uniform(rng); }
    DeviceBuffer dh = upload(h), dm = upload(mixes), db = upload(base), ds = upload(scale),
                 pre(std::size_t(tokens) * S * 4), post(std::size_t(tokens) * S * 4),
                 comb(std::size_t(tokens) * S * S * 4);
    k::HcSplitArgs a;
    a.h          = as<float>(dh);
    a.mixes      = as<float>(dm);
    a.tokens     = tokens;
    a.streams    = S;
    a.hidden     = H;
    a.scale      = as<float>(ds);
    a.base       = as<float>(db);
    a.norm_eps   = kEps;
    a.hc_eps     = kHcEps;
    a.iterations = iterations;
    a.pre        = as<float>(pre);
    a.post       = as<float>(post);
    a.comb       = as<float>(comb);
    k::hc_split(a, nullptr);
    sync("hc_split");
    const auto got_pre  = download<float>(pre, std::size_t(tokens) * S);
    const auto got_post = download<float>(post, std::size_t(tokens) * S);
    const auto got_comb = download<float>(comb, std::size_t(tokens) * S * S);
    double worst_pre = 0.0, worst_post = 0.0, worst_comb = 0.0;
    for (int t = 0; t < tokens; ++t) {
        const auto m = [&](int j) {
            return double(mixes[std::size_t(t) * M + j]) * rstd[std::size_t(t)];
        };
        for (int s = 0; s < S; ++s) {
            const double want_pre  = sigmoid(m(s) * scale[0] + base[s]) + kHcEps;
            const double want_post = 2.0 * sigmoid(m(S + s) * scale[1] + base[S + s]);
            worse(worst_pre, std::fabs(got_pre[t * S + s] - want_pre) / want_pre);
            worse(worst_post, std::fabs(got_post[t * S + s] - want_post) / want_post);
        }
        double c[S][S];
        for (int j = 0; j < S; ++j) {
            double mx = -INFINITY, sum = 0.0;
            for (int q = 0; q < S; ++q) {
                c[j][q] = m(2 * S + j * S + q) * scale[2] + base[2 * S + j * S + q];
                mx      = std::max(mx, c[j][q]);
            }
            for (int q = 0; q < S; ++q) {
                c[j][q] = std::exp(c[j][q] - mx);
                sum += c[j][q];
            }
            for (int q = 0; q < S; ++q) { c[j][q] = c[j][q] / sum + kHcEps; }
        }
        const auto columns = [&] {
            for (int q = 0; q < S; ++q) {
                double sum = 0.0;
                for (int j = 0; j < S; ++j) { sum += c[j][q]; }
                for (int j = 0; j < S; ++j) { c[j][q] /= sum + kHcEps; }
            }
        };
        columns();
        for (int it = 1; it < iterations; ++it) {
            for (int j = 0; j < S; ++j) {
                double sum = 0.0;
                for (int q = 0; q < S; ++q) { sum += c[j][q]; }
                for (int q = 0; q < S; ++q) { c[j][q] /= sum + kHcEps; }
            }
            columns();
        }
        for (int j = 0; j < S; ++j) {
            for (int q = 0; q < S; ++q) {
                const double got = got_comb[(std::size_t(t) * S + j) * S + q];
                worse(worst_comb, std::fabs(got - c[j][q]) / c[j][q]);
            }
        }
    }
    report(worst_pre <= 1e-5 && worst_post <= 1e-5 && worst_comb <= 2e-5,
           "hc_split " + rows + ", " + std::to_string(iterations) + " iterations: pre " +
               num(worst_pre) + " post " + num(worst_post) + " comb " + num(worst_comb));

    // hc_pre_norm: the streams mixed (or stream 0), rounded to BF16, normalized.
    std::vector<float> mix(std::size_t(tokens) * S), w(H);
    for (auto& v : mix) { v = 0.1f + uniform(rng); }
    for (auto& v : w) { v = 1.0f + 0.2f * normal(rng); }
    DeviceBuffer dmix = upload(mix), dw = upload(w), out(std::size_t(tokens) * H * 2),
                 out32(std::size_t(tokens) * H * 4);
    for (const bool mixed : {true, false}) {
        k::hc_pre_norm(as<float>(dh), mixed ? as<float>(dmix) : nullptr, as<float>(dw), tokens, S,
                       H, kEps, as<bf16>(out), as<float>(out32), nullptr);
        sync("hc_pre_norm");
        const auto got   = download<bf16>(out, std::size_t(tokens) * H);
        const auto got32 = download<float>(out32, std::size_t(tokens) * H);
        Tally tally;
        std::size_t split = 0;
        for (int t = 0; t < tokens; ++t) {
            std::vector<double> y(H);
            for (int i = 0; i < H; ++i) {
                double v = 0.0;
                for (int s = 0; s < (mixed ? S : 1); ++s) {
                    v += (mixed ? double(mix[std::size_t(t) * S + s]) : 1.0) *
                         h[t * width + s * H + i];
                }
                y[std::size_t(i)] = bf(v);
            }
            const auto want = norm_row(y.data(), H, w.data());
            for (int i = 0; i < H; ++i) {
                const std::size_t at = std::size_t(t) * H + i;
                tally.add(val(got[at]), want[std::size_t(i)], bf16_step(want[std::size_t(i)]));
                split += got32[at] != val(got[at]) ? 1 : 0;
            }
        }
        report(tally.within(1e-3, 1.0) && split == 0,
               std::string("hc_pre_norm ") + (mixed ? "mixed" : "stream 0") + ", " + rows + ": " +
                   tally.str() + ", " + std::to_string(split) + " FP32 rows unequal");
    }

    // hc_post: h[k] = post[k] y + sum_j comb[j][k] h[j].
    std::vector<float> y(std::size_t(tokens) * H), post_v(std::size_t(tokens) * S),
        comb_v(std::size_t(tokens) * S * S);
    for (auto& v : y) { v = normal(rng); }
    for (auto& v : post_v) { v = 2.0f * uniform(rng); }
    for (auto& v : comb_v) { v = 0.5f * uniform(rng); }
    DeviceBuffer dh2 = upload(h), dy = upload(y), dp = upload(post_v), dc = upload(comb_v);
    k::hc_post(as<float>(dh2), as<float>(dy), as<float>(dp), as<float>(dc), tokens, S, H, nullptr);
    sync("hc_post");
    const auto got = download<float>(dh2, h.size());
    double worst   = 0.0;
    for (int t = 0; t < tokens; ++t) {
        for (int q = 0; q < S; ++q) {
            for (int i = 0; i < H; ++i) {
                const double py =
                    double(post_v[std::size_t(t) * S + q]) * y[std::size_t(t) * H + i];
                double want = py, magnitude = std::fabs(py);
                for (int j = 0; j < S; ++j) {
                    const double ch =
                        double(comb_v[(std::size_t(t) * S + j) * S + q]) * h[t * width + j * H + i];
                    want += ch;
                    magnitude += std::fabs(ch);
                }
                worse(worst, std::fabs(got[t * width + q * H + i] - want) / magnitude);
            }
        }
    }
    report(worst <= 1e-6, "hc_post " + rows + ": worst " + num(worst));
}

// rms_rows of the query-rank part of q.kv projection rows (1280 of each 1792 values).
void rms_rows(int tokens, std::mt19937& rng) {
    constexpr int n = 1280, stride = 1280 + kLatent;
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x(std::size_t(tokens) * stride), w(n);
    for (auto& v : x) { v = 3.0f * normal(rng); }
    for (auto& v : w) { v = 1.0f + 0.2f * normal(rng); }
    DeviceBuffer dx = upload(x), dw = upload(w), out(std::size_t(tokens) * n * 2),
                 out32(std::size_t(tokens) * n * 4);
    k::rms_rows(as<float>(dx), stride, n, tokens, as<float>(dw), kEps, as<bf16>(out),
                as<float>(out32), nullptr);
    sync("rms_rows");
    const auto got   = download<bf16>(out, std::size_t(tokens) * n);
    const auto got32 = download<float>(out32, std::size_t(tokens) * n);
    Tally tally;
    std::size_t split = 0;
    for (int t = 0; t < tokens; ++t) {
        const auto want = norm_row(x.data() + std::size_t(t) * stride, n, w.data());
        for (int i = 0; i < n; ++i) {
            const std::size_t at = std::size_t(t) * n + i;
            tally.add(val(got[at]), want[std::size_t(i)], bf16_step(want[std::size_t(i)]));
            split += got32[at] != val(got[at]) ? 1 : 0;
        }
    }
    report(tally.within(1e-3, 1.0) && split == 0,
           "rms_rows " + std::to_string(tokens) + " strided rows: " + tally.str());
}

// ---- Window latents and rotations
// ----------------------------------------------------------------

struct KvCase {
    std::string name;
    std::vector<int> positions = {};
    std::vector<int> sequences = {}; // empty: every row of `sequence`
    int sequence = 0, ring = kRing;
    std::vector<int> slots = {};    // empty: the rows' positions
    bool staging           = false; // a chunk's staging ring: one ring, no sequence stride
    double theta           = 10000.0;
};

void kv_post(const KvCase& c, std::mt19937& rng) {
    constexpr int query = 1280, stride = query + kLatent;
    const int t    = int(c.positions.size());
    const int seqs = c.staging ? 1
                     : c.sequences.empty()
                         ? c.sequence + 1
                         : *std::max_element(c.sequences.begin(), c.sequences.end()) + 1;
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> qkv(std::size_t(t) * stride), norm(kLatent);
    for (auto& v : qkv) { v = 2.0f * normal(rng); }
    for (auto& v : norm) { v = 1.0f + 0.2f * normal(rng); }
    const auto freq               = frequencies(c.theta);
    const std::int64_t seq_stride = c.staging ? 0 : std::int64_t(c.ring) * k::kWindowRowBytes;
    std::vector<std::uint8_t> cache(std::size_t(seqs) * c.ring * k::kWindowRowBytes, kSentinelByte);
    DeviceBuffer dqkv = upload(qkv), dn = upload(norm), df = upload(freq),
                 dp = upload(ints(c.positions)), dslots = upload(ints(c.slots)),
                 dseq = upload(ints(c.sequences)), dc = upload(cache);
    k::KvPostArgs a;
    a.kv         = as<float>(dqkv) + query;
    a.kv_stride  = stride;
    a.tokens     = t;
    a.norm       = as<float>(dn);
    a.eps        = kEps;
    a.positions  = as<std::int32_t>(dp);
    a.slots      = c.slots.empty() ? nullptr : as<std::int32_t>(dslots);
    a.freq       = as<float>(df);
    a.cache      = as<std::uint8_t>(dc);
    a.seq_stride = seq_stride;
    a.sequences  = c.sequences.empty() ? nullptr : as<std::int32_t>(dseq);
    a.sequence   = c.sequence;
    a.ring       = c.ring;
    k::kv_post(a, nullptr);
    sync("kv_post");
    const auto got = download<std::uint8_t>(dc, cache.size());
    std::vector<bool> written(std::size_t(seqs) * c.ring);
    Tally tally;
    for (int r = 0; r < t; ++r) {
        const int s = c.staging             ? 0
                      : c.sequences.empty() ? c.sequence
                                            : c.sequences[std::size_t(r)];
        const int slot =
            (c.slots.empty() ? c.positions[std::size_t(r)] : c.slots[std::size_t(r)]) % c.ring;
        written[std::size_t(s) * c.ring + slot] = true;
        auto v = norm_row(qkv.data() + std::size_t(r) * stride + query, kLatent, norm.data());
        rotate_tail(v.data() + kLatent - kRope, c.positions[std::size_t(r)], freq, false);
        std::vector<double> step(kLatent);
        for (int g = 0; g < kLatent / 32; ++g) {
            fp8_round(v.data() + 32 * g, step.data() + 32 * g);
        }
        const auto stored = window_values(got.data() + (std::size_t(s) * c.ring + slot) * k::kWindowRowBytes);
        for (int i = 0; i < kLatent; ++i) {
            tally.add(stored[std::size_t(i)], v[std::size_t(i)], step[std::size_t(i)]);
        }
    }
    std::size_t stray = 0;
    for (std::size_t slot = 0; slot < written.size(); ++slot) {
        if (written[slot]) { continue; }
        for (int i = 0; i < k::kWindowRowBytes; ++i) {
            stray += got[slot * k::kWindowRowBytes + i] != kSentinelByte ? 1 : 0;
        }
    }
    report(tally.within(1e-3, 2.0) && stray == 0,
           "kv_post " + c.name + ": " + tally.str() + ", " + std::to_string(stray) + " stray");
}

void rotations(std::mt19937& rng) {
    const std::vector<int> positions = {0, 3, 129, 4000, 100000};
    const int t                      = int(positions.size());
    const auto freq                  = frequencies(160000.0);
    DeviceBuffer dp = upload(ints(positions)), df = upload(freq);
    std::normal_distribution<float> normal(0.0f, 1.5f);

    // The indexer's queries: FP32 rows, fp4 groups of 32 after the rotation.
    {
        std::vector<float> x(std::size_t(t) * kIndexHeads * kIndex);
        for (auto& v : x) { v = normal(rng); }
        DeviceBuffer dx = upload(x), out(x.size() * 2);
        k::rotate_heads(as<float>(dx), t, kIndexHeads, kIndex, as<std::int32_t>(dp), as<float>(df),
                        true, as<bf16>(out), nullptr);
        sync("rotate_heads fp4");
        const auto got = download<bf16>(out, x.size());
        Tally plain, turned;
        for (int r = 0; r < t; ++r) {
            for (int h = 0; h < kIndexHeads; ++h) {
                const std::size_t row = (std::size_t(r) * kIndexHeads + h) * kIndex;
                std::vector<float> v(kIndex);
                for (int i = 0; i < kIndex; ++i) { v[std::size_t(i)] = bf(x[row + i]); }
                rotate_tail(v.data() + kIndex - kRope, positions[std::size_t(r)], freq, false);
                std::vector<double> step(kIndex);
                for (int g = 0; g < kIndex / 32; ++g) {
                    fp4_e8m0_round(v.data() + 32 * g, step.data() + 32 * g);
                }
                for (int i = 0; i < kIndex; ++i) {
                    (i < kIndex - kRope ? plain : turned)
                        .add(val(got[row + i]), v[std::size_t(i)], step[std::size_t(i)]);
                }
            }
        }
        report(plain.off == 0 && turned.within(1e-3, 2.0),
               "rotate_heads indexer queries (fp4): unturned " + plain.str() + ", turned " +
                   turned.str());
    }
    // The attention's queries: BF16 rows turned in place.
    {
        const auto x   = random_bf16(std::size_t(t) * kHeads * kLatent, 1.0f, rng);
        DeviceBuffer d = upload(x);
        k::rotate_heads(as<bf16>(d), t, kHeads, kLatent, as<std::int32_t>(dp), as<float>(df), false,
                        as<bf16>(d), nullptr);
        sync("rotate_heads in place");
        const auto got = download<bf16>(d, x.size());
        Tally turned;
        std::size_t moved = 0;
        for (int r = 0; r < t; ++r) {
            for (int h = 0; h < kHeads; ++h) {
                const std::size_t row = (std::size_t(r) * kHeads + h) * kLatent;
                std::vector<float> tail(kRope);
                for (int i = 0; i < kRope; ++i) {
                    tail[std::size_t(i)] = val(x[row + kLatent - kRope + i]);
                }
                rotate_tail(tail.data(), positions[std::size_t(r)], freq, false);
                for (int i = 0; i < kLatent - kRope; ++i) {
                    moved += bits(got[row + i]) != bits(x[row + i]) ? 1 : 0;
                }
                for (int i = 0; i < kRope; ++i) {
                    const float want = tail[std::size_t(i)];
                    turned.add(val(got[row + kLatent - kRope + i]), want, bf16_step(want));
                }
            }
        }
        report(moved == 0 && turned.within(1e-3, 1.0),
               "rotate_heads attention queries in place: " + std::to_string(moved) +
                   " unturned values changed, turned " + turned.str());
    }
}

// ---- Compressor ---------------------------------------------------------------------------------

struct CompressCase {
    std::string name;
    int ratio                  = 1;
    std::vector<int> positions = {};
    std::vector<int> sequences = {}; // empty: a chunk of `sequence`
    int sequence = 0, tail_from = 0;
};

void compress(const CompressCase& c, std::mt19937& rng) {
    constexpr int slots = 16;
    const int t         = int(c.positions.size());
    const int seqs      = c.sequences.empty()
                              ? c.sequence + 1
                              : *std::max_element(c.sequences.begin(), c.sequences.end()) + 1;
    const auto seq_of   = [&](int r) {
        return c.sequences.empty() ? c.sequence : c.sequences[std::size_t(r)];
    };
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> kv(std::size_t(t) * kLatent), score(kv.size()), norm(kLatent),
        tail_kv(std::size_t(seqs) * slots * kLatent), tail_score(tail_kv.size()),
        latent(kv.size(), std::numeric_limits<float>::quiet_NaN());
    for (auto& v : kv) { v = 2.0f * normal(rng); }
    for (auto& v : score) { v = 1.5f * normal(rng); }
    for (auto& v : norm) { v = 1.0f + 0.2f * normal(rng); }
    for (auto& v : tail_kv) { v = 2.0f * normal(rng); }
    for (auto& v : tail_score) { v = 1.5f * normal(rng); }
    DeviceBuffer dkv = upload(kv), dscore = upload(score), dn = upload(norm),
                 dp = upload(ints(c.positions)), dseq = upload(ints(c.sequences)),
                 dtk = upload(tail_kv), dts = upload(tail_score), dl = upload(latent);
    k::CompressArgs a;
    a.kv              = as<float>(dkv);
    a.score           = c.ratio > 1 ? as<float>(dscore) : nullptr;
    a.tokens          = t;
    a.ratio           = c.ratio;
    a.positions       = as<std::int32_t>(dp);
    a.sequences       = c.sequences.empty() ? nullptr : as<std::int32_t>(dseq);
    a.sequence        = c.sequence;
    a.tail_kv         = as<float>(dtk);
    a.tail_score      = as<float>(dts);
    a.tail_seq_stride = std::int64_t(slots) * kLatent;
    a.slots           = slots;
    a.tail_from       = c.tail_from;
    a.norm            = as<float>(dn);
    a.eps             = kEps;
    a.latent          = as<float>(dl);
    k::compress(a, nullptr);
    k::compress_tail(a, nullptr);
    sync("compress");
    const auto got       = download<float>(dl, latent.size());
    const auto got_kv    = download<float>(dtk, tail_kv.size());
    const auto got_score = download<float>(dts, tail_score.size());
    Tally tally;
    std::size_t stray = 0, groups = 0;
    for (int r = 0; r < t; ++r) {
        const int p = c.positions[std::size_t(r)], s = seq_of(r);
        const float* row = got.data() + std::size_t(r) * kLatent;
        std::vector<double> pooled(kLatent);
        if (c.ratio == 1) {
            for (int i = 0; i < kLatent; ++i) {
                pooled[std::size_t(i)] = kv[std::size_t(r) * kLatent + i];
            }
        } else if ((p + 1) % c.ratio == 0) {
            const bool in_call =
                r > 0 && c.positions[std::size_t(r) - 1] == p - 1 && seq_of(r - 1) == s;
            const std::size_t slot =
                (std::size_t(s) * slots + std::size_t((p - 1) % slots)) * kLatent;
            const float* k0 =
                in_call ? kv.data() + std::size_t(r - 1) * kLatent : tail_kv.data() + slot;
            const float* s0 =
                in_call ? score.data() + std::size_t(r - 1) * kLatent : tail_score.data() + slot;
            for (int i = 0; i < kLatent; ++i) {
                const double k1 = kv[std::size_t(r) * kLatent + i],
                             s1 = score[std::size_t(r) * kLatent + i];
                const double m  = std::max<double>(s0[i], s1);
                const double e0 = std::exp(s0[i] - m), e1 = std::exp(s1 - m);
                pooled[std::size_t(i)] = (k0[i] * e0 + k1 * e1) / (e0 + e1);
            }
        } else {
            for (int i = 0; i < kLatent; ++i) { stray += std::isnan(row[i]) ? 0 : 1; }
            continue;
        }
        ++groups;
        const auto want = norm_row(pooled.data(), kLatent, norm.data());
        for (int i = 0; i < kLatent; ++i) {
            tally.add(row[i], want[std::size_t(i)], bf16_step(want[std::size_t(i)]));
        }
    }
    // The tails: the rows from tail_from on (ratio 2) at their slots, the other slots unchanged.
    std::vector<float> want_kv = tail_kv, want_score = tail_score;
    for (int r = c.tail_from; c.ratio > 1 && r < t; ++r) {
        const std::size_t slot =
            (std::size_t(seq_of(r)) * slots + std::size_t(c.positions[std::size_t(r)] % slots)) *
            kLatent;
        std::copy_n(kv.data() + std::size_t(r) * kLatent, kLatent, want_kv.data() + slot);
        std::copy_n(score.data() + std::size_t(r) * kLatent, kLatent, want_score.data() + slot);
    }
    std::size_t tail_wrong = 0;
    for (std::size_t i = 0; i < want_kv.size(); ++i) {
        tail_wrong += got_kv[i] != want_kv[i] || got_score[i] != want_score[i] ? 1 : 0;
    }
    report(tally.within(1e-3, 1.0) && stray == 0 && tail_wrong == 0,
           "compress " + c.name + ": " + std::to_string(groups) + " latents " + tally.str() + ", " +
               std::to_string(stray) + " stray, tails " + std::to_string(tail_wrong) + " wrong");
}

struct LatentCase {
    std::string name;
    int ratio                  = 1;
    std::vector<int> positions = {};
    std::vector<int> sequences = {}; // empty: a chunk of `sequence`
    int sequence               = 0;
};

void latent_post(const LatentCase& c, std::mt19937& rng) {
    const int t    = int(c.positions.size());
    const int seqs = c.sequences.empty()
                         ? c.sequence + 1
                         : *std::max_element(c.sequences.begin(), c.sequences.end()) + 1;
    const int rows = *std::max_element(c.positions.begin(), c.positions.end()) / c.ratio + 2;
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> latent(std::size_t(t) * kLatent), key(std::size_t(t) * kIndex),
        key_norm(kIndex);
    for (auto& v : latent) { v = bf(normal(rng)); }
    for (auto& v : key) { v = 2.0f * normal(rng); }
    for (auto& v : key_norm) { v = 1.0f + 0.2f * normal(rng); }
    const auto freq = frequencies(160000.0);
    std::vector<std::uint8_t> index_cache(std::size_t(seqs) * rows * k::kKeyRowBytes, kSentinelByte),
        index_scales(std::size_t(seqs) * rows * k::kKeyScaleBytes, kSentinelByte),
        kv_cache(std::size_t(seqs) * rows * k::kLatentRowBytes, kSentinelByte);
    DeviceBuffer dl = upload(latent), dk = upload(key), dkn = upload(key_norm), df = upload(freq),
                 dp = upload(ints(c.positions)), dseq = upload(ints(c.sequences)),
                 di = upload(index_cache), dis = upload(index_scales), dc = upload(kv_cache);
    k::LatentArgs a;
    a.latent           = as<float>(dl);
    a.key              = as<float>(dk);
    a.tokens           = t;
    a.ratio            = c.ratio;
    a.positions        = as<std::int32_t>(dp);
    a.sequences        = c.sequences.empty() ? nullptr : as<std::int32_t>(dseq);
    a.sequence         = c.sequence;
    a.key_norm         = as<float>(dkn);
    a.eps              = kEps;
    a.freq             = as<float>(df);
    a.index_cache            = as<std::uint8_t>(di);
    a.index_seq_stride       = std::int64_t(rows) * k::kKeyRowBytes;
    a.index_scales           = as<std::uint8_t>(dis);
    a.index_scale_seq_stride = std::int64_t(rows) * k::kKeyScaleBytes;
    a.kv_cache               = as<std::uint8_t>(dc);
    a.kv_seq_stride          = std::int64_t(rows) * k::kLatentRowBytes;
    k::latent_post(a, nullptr);
    sync("latent_post");
    const auto got_index  = download<std::uint8_t>(di, index_cache.size());
    const auto got_scales = download<std::uint8_t>(dis, index_scales.size());
    const auto got_kv     = download<std::uint8_t>(dc, kv_cache.size());
    std::vector<bool> written(std::size_t(seqs) * rows);
    Tally keys, plain, turned;
    for (int r = 0; r < t; ++r) {
        const int p = c.positions[std::size_t(r)];
        if ((p + 1) % c.ratio != 0) { continue; }
        const int j = (p + 1) / c.ratio - 1, g = j * c.ratio;
        const int s = c.sequences.empty() ? c.sequence : c.sequences[std::size_t(r)];
        written[std::size_t(s) * rows + j] = true;
        auto kn = norm_row(key.data() + std::size_t(r) * kIndex, kIndex, key_norm.data());
        rotate_tail(kn.data() + kIndex - kRope, g, freq, false);
        std::vector<double> step(kLatent);
        for (int q = 0; q < kIndex / 32; ++q) {
            fp4_e8m0_round(kn.data() + 32 * q, step.data() + 32 * q);
        }
        const std::size_t row = std::size_t(s) * rows + j;
        const auto stored_key = key_values(got_index.data() + row * k::kKeyRowBytes,
                                           got_scales.data() + row * k::kKeyScaleBytes);
        for (int i = 0; i < kIndex; ++i) {
            keys.add(stored_key[std::size_t(i)], kn[std::size_t(i)], step[std::size_t(i)]);
        }
        std::vector<float> v(latent.begin() + std::ptrdiff_t(r) * kLatent,
                             latent.begin() + std::ptrdiff_t(r + 1) * kLatent);
        rotate_tail(v.data() + kLatent - kRope, g, freq, false);
        for (int q = 0; q < kLatent / 16; ++q) {
            fp4_e4m3_round(v.data() + 16 * q, step.data() + 16 * q);
        }
        const auto stored = latent_values(got_kv.data() + row * k::kLatentRowBytes);
        for (int i = 0; i < kLatent; ++i) {
            (i < kLatent - kRope ? plain : turned)
                .add(stored[std::size_t(i)], v[std::size_t(i)], step[std::size_t(i)]);
        }
    }
    std::size_t stray = 0;
    for (std::size_t row = 0; row < written.size(); ++row) {
        if (written[row]) { continue; }
        for (int i = 0; i < k::kKeyRowBytes; ++i) {
            stray += got_index[row * k::kKeyRowBytes + i] != kSentinelByte ? 1 : 0;
        }
        for (int i = 0; i < k::kKeyScaleBytes; ++i) {
            stray += got_scales[row * k::kKeyScaleBytes + i] != kSentinelByte ? 1 : 0;
        }
        for (int i = 0; i < k::kLatentRowBytes; ++i) {
            stray += got_kv[row * k::kLatentRowBytes + i] != kSentinelByte ? 1 : 0;
        }
    }
    report(keys.within(1e-3, 2.0) && plain.off == 0 && turned.within(1e-3, 2.0) && stray == 0,
           "latent_post " + c.name + ": keys " + keys.str() + ", latents unturned " + plain.str() +
               ", turned " + turned.str() + ", " + std::to_string(stray) + " stray");
}

// ---- Indexer and selection ----------------------------------------------------------------------

struct ScoreCase {
    std::string name;
    std::vector<int> positions = {};
    std::vector<int> sequences = {}; // empty: a chunk of `sequence`
    int sequence = 0, ratio = 1, group = 1;
    bool candidates = false;
};

void index_scores(const ScoreCase& c, std::mt19937& rng) {
    constexpr int block = 8;
    const int t         = int(c.positions.size());
    const bool chunk    = c.sequences.empty();
    const int seqs =
        chunk ? c.sequence + 1 : *std::max_element(c.sequences.begin(), c.sequences.end()) + 1;
    std::vector<int> visible(static_cast<std::size_t>(t));
    for (int r = 0; r < t; ++r) {
        visible[std::size_t(r)] = (c.positions[std::size_t(r)] + 1) / c.ratio;
    }
    const int most  = *std::max_element(visible.begin(), visible.end());
    const int rows  = most + 5; // the keys a sequence's cache holds
    const int width = chunk ? std::max(1, most) : rows;
    const int words = ((width + block - 1) / block + 31) / 32;
    const auto key_codes  = random_key_codes(std::size_t(seqs) * rows, rng);
    const auto key_scales = random_key_scales(std::size_t(seqs) * rows, rng);
    const auto q          = random_bf16(std::size_t(t) * kIndexHeads * kIndex, 1.0f, rng);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> weights(std::size_t(t) * kIndexHeads),
        out(std::size_t(t) * width, std::numeric_limits<float>::quiet_NaN());
    for (auto& v : weights) { v = normal(rng); }
    std::vector<std::uint32_t> candidates(std::size_t(t) * words);
    for (auto& v : candidates) { v = std::uint32_t(rng()); }
    const float weight_scale = 1.0f / std::sqrt(float(kIndex)) / std::sqrt(float(kIndexHeads));
    DeviceBuffer dq = upload(q), dw = upload(weights), dk = upload(key_codes),
                 dks = upload(key_scales), dp = upload(ints(c.positions)),
                 dseq = upload(ints(c.sequences)), dc = upload(candidates), dout = upload(out);
    k::ScoreArgs a;
    a.q              = as<bf16>(dq);
    a.weights        = as<float>(dw);
    a.weight_scale   = weight_scale;
    a.heads          = kIndexHeads;
    a.tokens         = t;
    a.ratio          = c.ratio;
    a.width          = width;
    a.group          = c.group;
    a.positions      = as<std::int32_t>(dp);
    a.sequences      = chunk ? nullptr : as<std::int32_t>(dseq);
    a.sequence       = c.sequence;
    a.keys                 = as<std::uint8_t>(dk);
    a.key_seq_stride       = std::int64_t(rows) * k::kKeyRowBytes;
    a.key_scales           = as<std::uint8_t>(dks);
    a.key_scale_seq_stride = std::int64_t(rows) * k::kKeyScaleBytes;
    if (c.candidates) {
        a.candidates      = as<std::uint32_t>(dc);
        a.candidate_words = words;
        a.block           = block;
    }
    a.out = as<float>(dout);
    k::index_scores(a, nullptr);
    sync("index_scores");
    const auto got        = download<float>(dout, out.size());
    double worst          = 0.0;
    std::size_t wrong_inf = 0, stray = 0, masked = 0;
    for (int r = 0; r < t; ++r) {
        const int s = chunk ? c.sequence : c.sequences[std::size_t(r)];
        for (int j = 0; j < width; ++j) {
            const float g = got[std::size_t(r) * width + j];
            if (j >= visible[std::size_t(r)]) {
                stray += std::isnan(g) ? 0 : 1;
                continue;
            }
            if (c.candidates &&
                ((candidates[std::size_t(r) * words + j / block / 32] >> (j / block % 32)) & 1U) ==
                    0) {
                wrong_inf += g == -INFINITY ? 0 : 1;
                ++masked;
                continue;
            }
            const std::size_t key_row = std::size_t(s) * rows + j;
            const auto kr = key_values(key_codes.data() + key_row * k::kKeyRowBytes,
                                       key_scales.data() + key_row * k::kKeyScaleBytes);
            double want = 0.0, bound = 0.0;
            for (int h = 0; h < kIndexHeads; ++h) {
                const bf16* qh = q.data() + (std::size_t(r) * kIndexHeads + h) * kIndex;
                double d = 0.0, magnitude = 0.0;
                for (int i = 0; i < kIndex; ++i) {
                    const double p = double(val(qh[i])) * kr[std::size_t(i)];
                    d += p;
                    magnitude += std::fabs(p);
                }
                const double wh = double(weights[std::size_t(r) * kIndexHeads + h] * weight_scale);
                want += std::max(d, 0.0) * wh;
                bound += magnitude * std::fabs(wh);
            }
            worse(worst, std::fabs(g - want) / bound);
        }
    }
    report(worst <= 1e-5 && wrong_inf == 0 && stray == 0,
           "index_scores " + c.name + ": worst " + num(worst) + ", " + std::to_string(masked) +
               " masked (" + std::to_string(wrong_inf) + " wrong), " + std::to_string(stray) +
               " stray");
}

// Scores of n visible positions: pattern 0 normal; 1 a few distinct values (ties, both zeros among
// them); 2 with -inf and NaN among them; 3 mostly -inf, fewer finite values than are chosen.
std::vector<float> score_row(int n, int pattern, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::vector<float> v(static_cast<std::size_t>(n));
    for (auto& x : v) {
        x = normal(rng);
        if (pattern == 1) { x = std::round(2.0f * x) / 2.0f; } // -0.0 among the zeros
        const float u = uniform(rng);
        if (pattern == 2 && u < 0.3f) { x = -INFINITY; }
        if (pattern == 2 && u > 0.95f) { x = std::numeric_limits<float>::quiet_NaN(); }
        if (pattern == 3 && u < 0.97f) { x = -INFINITY; }
    }
    return v;
}

void select_topk(const std::vector<int>& positions, int ratio, int count, int pattern,
                 std::mt19937& rng) {
    const int t = int(positions.size());
    std::vector<int> n(static_cast<std::size_t>(t));
    for (int r = 0; r < t; ++r) { n[std::size_t(r)] = (positions[std::size_t(r)] + 1) / ratio; }
    const int width = *std::max_element(n.begin(), n.end()) + 7;
    // Past a row's visible positions lie scores larger than any: a read there would choose them.
    std::vector<float> scores(std::size_t(t) * width, 1e30f);
    for (int r = 0; r < t; ++r) {
        const auto row = score_row(n[std::size_t(r)], pattern, rng);
        std::copy(row.begin(), row.end(), scores.begin() + std::ptrdiff_t(r) * width);
    }
    DeviceBuffer ds = upload(scores), dp = upload(ints(positions)), out(std::size_t(t) * count * 4);
    out.fill(0x7F);
    k::select_topk(as<float>(ds), width, t, as<std::int32_t>(dp), ratio, count,
                   as<std::int32_t>(out), nullptr);
    sync("select_topk");
    const auto got    = download<std::int32_t>(out, std::size_t(t) * count);
    std::size_t wrong = 0;
    for (int r = 0; r < t; ++r) {
        const float* row = scores.data() + std::size_t(r) * width;
        std::vector<int> order;
        for (int i = 0; i < n[std::size_t(r)]; ++i) {
            if (row[i] != -INFINITY && !std::isnan(row[i])) { order.push_back(i); }
        }
        std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return row[x] > row[y]; });
        order.resize(std::min<std::size_t>(order.size(), std::size_t(count)));
        std::sort(order.begin(), order.end());
        order.resize(std::size_t(count), -1);
        for (int i = 0; i < count; ++i) {
            wrong += got[std::size_t(r) * count + i] != order[std::size_t(i)] ? 1 : 0;
        }
    }
    std::string sizes;
    for (const int v : n) { sizes += (sizes.empty() ? "" : ",") + std::to_string(v); }
    report(wrong == 0, "select_topk " + std::to_string(count) + " of " + sizes + " (pattern " +
                           std::to_string(pattern) + "): " + std::to_string(wrong) + " wrong");
}

// `context` > 0 sizes the masks for a context of that many positions, far past the rows' own (a
// chunk's rows early in a long context); the block maxima then still take one row of the blocks
// the rows see, followed by a guard the kernel must not write.
void select_candidates(const std::vector<int>& positions, int ratio, int keep, int pattern,
                       std::mt19937& rng, int context = 0) {
    constexpr int block = 8;
    const int t         = int(positions.size());
    std::vector<int> n(static_cast<std::size_t>(t));
    for (int r = 0; r < t; ++r) { n[std::size_t(r)] = (positions[std::size_t(r)] + 1) / ratio; }
    const int most  = *std::max_element(n.begin(), n.end());
    const int width = most + 3;
    const int words = context > 0 ? ((context / ratio + block - 1) / block + 31) / 32
                                  : ((most + block - 1) / block + 31) / 32 + 1;
    const int stride = (width + block - 1) / block;
    constexpr int kGuard = 256;
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::vector<float> scores(std::size_t(t) * width, 1e30f);
    for (int r = 0; r < t; ++r) {
        auto row = score_row(n[std::size_t(r)], pattern == 2 ? 0 : pattern, rng);
        // Pattern 2: whole blocks without a finite score (the last one among them at times).
        for (int b = 0; pattern == 2 && b * block < n[std::size_t(r)]; ++b) {
            if (uniform(rng) < 0.3f) {
                for (int i = b * block; i < std::min(n[std::size_t(r)], (b + 1) * block); ++i) {
                    row[std::size_t(i)] = -INFINITY;
                }
            }
        }
        std::copy(row.begin(), row.end(), scores.begin() + std::ptrdiff_t(r) * width);
    }
    DeviceBuffer ds = upload(scores), dp = upload(ints(positions)),
                 mask(std::size_t(t) * words * 4),
                 scratch((std::size_t(t) * stride + kGuard) * 4);
    mask.fill(0xFF);
    scratch.fill(0x5A);
    k::select_candidates(as<float>(ds), width, t, as<std::int32_t>(dp), ratio, block, keep,
                         as<std::uint32_t>(mask), words, as<float>(scratch), stride, nullptr);
    sync("select_candidates");
    const auto got    = download<std::uint32_t>(mask, std::size_t(t) * words);
    const auto tail   = download<std::uint32_t>(scratch, std::size_t(t) * stride + kGuard);
    std::size_t wrong = 0;
    for (std::size_t i = std::size_t(t) * stride; i < tail.size(); ++i) {
        wrong += tail[i] != 0x5A5A5A5AU ? 1 : 0;
    }
    for (int r = 0; r < t; ++r) {
        const float* row = scores.data() + std::size_t(r) * width;
        const int blocks = (n[std::size_t(r)] + block - 1) / block;
        std::vector<float> maxima(std::size_t(blocks), -INFINITY);
        for (int i = 0; i < n[std::size_t(r)]; ++i) {
            maxima[std::size_t(i / block)] = std::max(maxima[std::size_t(i / block)], row[i]);
        }
        if (blocks > 0) { maxima.back() = INFINITY; }
        std::vector<int> order(static_cast<std::size_t>(blocks));
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int x, int y) {
            return maxima[std::size_t(x)] > maxima[std::size_t(y)];
        });
        std::vector<std::uint32_t> want(static_cast<std::size_t>(words));
        for (int i = 0; i < std::min(blocks, keep); ++i) {
            const int b = order[std::size_t(i)];
            if (maxima[std::size_t(b)] > -INFINITY) { want[std::size_t(b / 32)] |= 1U << (b % 32); }
        }
        for (int w = 0; w < words; ++w) {
            wrong += got[std::size_t(r) * words + w] != want[std::size_t(w)] ? 1 : 0;
        }
    }
    std::string sizes;
    for (const int v : n) { sizes += (sizes.empty() ? "" : ",") + std::to_string(v); }
    report(wrong == 0, "select_candidates " + std::to_string(keep) + " blocks of " + sizes +
                           " (pattern " + std::to_string(pattern) +
                           (context > 0 ? ", masks of " + std::to_string(context) + " positions"
                                        : std::string()) +
                           "): " + std::to_string(wrong) + " wrong words");
}

// A scratch stride below the blocks of a row's width is refused before any launch.
void select_candidates_stride() {
    bool refused = false;
    try {
        k::select_candidates(nullptr, 801, 1, nullptr, 1, 8, 2048, nullptr, 4, nullptr, 100,
                             nullptr);
    } catch (const std::invalid_argument&) { refused = true; }
    report(refused, "select_candidates refuses a scratch stride of 100 for 801 positions");
}

// Rows that see more blocks than they keep: candidate_lists against the union of each group's
// masks, index_scores over the listed blocks against the same rows over every visible position
// (bit for bit, -inf where a row does not keep a listed block) and select_topk over the listed rows
// against select_topk over the whole rows (the same positions).
void listed_index(const std::string& name, const std::vector<int>& positions,
                  const std::vector<int>& sequences, int group, int keep, std::mt19937& rng) {
    constexpr int block = 8;
    const int t         = int(positions.size());
    const bool chunk    = sequences.empty();
    const int seqs = chunk ? 1 : *std::max_element(sequences.begin(), sequences.end()) + 1;
    std::vector<int> visible(static_cast<std::size_t>(t));
    for (int r = 0; r < t; ++r) { visible[std::size_t(r)] = positions[std::size_t(r)] + 1; }
    const int most     = *std::max_element(visible.begin(), visible.end());
    const int rows     = most + 3;
    const int width    = chunk ? most : rows;
    const int blocks   = (width + block - 1) / block;
    const int words    = (blocks + 31) / 32;
    const int capacity = std::min(group * keep, blocks);
    const int groups   = (t + group - 1) / group;
    // Each row keeps `keep` of its visible blocks at random, its last one among them.
    std::vector<std::uint32_t> mask(std::size_t(t) * words);
    for (int r = 0; r < t; ++r) {
        const int own = (visible[std::size_t(r)] + block - 1) / block;
        std::vector<int> order(static_cast<std::size_t>(own - 1));
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        order.resize(std::size_t(std::min(own - 1, keep - 1)));
        order.push_back(own - 1);
        for (const int b : order) { mask[std::size_t(r) * words + b / 32] |= 1U << (b % 32); }
    }
    const auto key_codes  = random_key_codes(std::size_t(seqs) * rows, rng);
    const auto key_scales = random_key_scales(std::size_t(seqs) * rows, rng);
    const auto q          = random_bf16(std::size_t(t) * kIndexHeads * kIndex, 1.0f, rng);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> weights(std::size_t(t) * kIndexHeads);
    for (auto& v : weights) { v = normal(rng); }
    DeviceBuffer dq = upload(q), dw = upload(weights), dk = upload(key_codes),
                 dks = upload(key_scales), dp = upload(ints(positions)), dseq = upload(ints(sequences)), dm = upload(mask),
                 dlist(std::size_t(groups) * capacity * 4), dcounts(std::size_t(groups) * 4),
                 whole(std::size_t(t) * width * 4), listed(std::size_t(t) * capacity * block * 4),
                 top_whole(std::size_t(t) * 512 * 4), top_listed(std::size_t(t) * 512 * 4);
    whole.fill(0xFF);
    listed.fill(0xFF);
    k::candidate_lists(as<std::uint32_t>(dm), words, t, group, capacity, as<std::int32_t>(dlist),
                       as<std::int32_t>(dcounts), nullptr);
    sync("candidate_lists");
    const auto list   = download<std::int32_t>(dlist, std::size_t(groups) * capacity);
    const auto counts = download<std::int32_t>(dcounts, std::size_t(groups));
    std::size_t wrong_lists = 0;
    for (int g = 0; g < groups; ++g) {
        std::vector<int> want;
        for (int b = 0; b < blocks; ++b) {
            for (int r = g * group; r < std::min(t, (g + 1) * group); ++r) {
                if ((mask[std::size_t(r) * words + b / 32] >> (b % 32)) & 1U) {
                    want.push_back(b);
                    break;
                }
            }
        }
        wrong_lists += counts[std::size_t(g)] != int(want.size()) ? 1 : 0;
        for (std::size_t i = 0; i < want.size() && i < std::size_t(capacity); ++i) {
            wrong_lists += list[std::size_t(g) * capacity + i] != want[i] ? 1 : 0;
        }
    }
    k::ScoreArgs a;
    a.q               = as<bf16>(dq);
    a.weights         = as<float>(dw);
    a.weight_scale    = 1.0f / std::sqrt(float(kIndex)) / std::sqrt(float(kIndexHeads));
    a.heads           = kIndexHeads;
    a.tokens          = t;
    a.width           = width;
    a.group           = group;
    a.positions       = as<std::int32_t>(dp);
    a.sequences       = chunk ? nullptr : as<std::int32_t>(dseq);
    a.keys                 = as<std::uint8_t>(dk);
    a.key_seq_stride       = std::int64_t(rows) * k::kKeyRowBytes;
    a.key_scales           = as<std::uint8_t>(dks);
    a.key_scale_seq_stride = std::int64_t(rows) * k::kKeyScaleBytes;
    a.candidates      = as<std::uint32_t>(dm);
    a.candidate_words = words;
    a.block           = block;
    a.out             = as<float>(whole);
    k::index_scores(a, nullptr);
    a.list     = as<std::int32_t>(dlist);
    a.counts   = as<std::int32_t>(dcounts);
    a.capacity = capacity;
    a.width    = capacity * block;
    a.out      = as<float>(listed);
    k::index_scores(a, nullptr);
    sync("index_scores listed");
    const auto got_whole  = download<std::uint32_t>(whole, std::size_t(t) * width);
    const auto got_listed = download<std::uint32_t>(listed, std::size_t(t) * capacity * block);
    std::size_t differ = 0, compared = 0;
    for (int r = 0; r < t; ++r) {
        const int g = r / group;
        for (int s = 0; s < counts[std::size_t(g)] * block; ++s) {
            const int j = list[std::size_t(g) * capacity + s / block] * block + s % block;
            if (j >= visible[std::size_t(r)]) { continue; }
            ++compared;
            differ += got_listed[std::size_t(r) * capacity * block + s] !=
                              got_whole[std::size_t(r) * width + j]
                          ? 1
                          : 0;
        }
    }
    k::select_topk(as<float>(whole), width, t, as<std::int32_t>(dp), 1, 512,
                   as<std::int32_t>(top_whole), nullptr);
    k::select_topk(as<float>(listed), capacity * block, t, as<std::int32_t>(dp), 1, 512,
                   k::ListedRows{.list     = as<std::int32_t>(dlist),
                                 .counts   = as<std::int32_t>(dcounts),
                                 .capacity = capacity,
                                 .group    = group,
                                 .block    = block},
                   as<std::int32_t>(top_listed), nullptr);
    sync("select_topk listed");
    const auto want_top = download<std::int32_t>(top_whole, std::size_t(t) * 512);
    const auto got_top  = download<std::int32_t>(top_listed, std::size_t(t) * 512);
    std::size_t wrong_top = 0;
    for (std::size_t i = 0; i < want_top.size(); ++i) { wrong_top += got_top[i] != want_top[i] ? 1 : 0; }
    report(wrong_lists == 0 && differ == 0 && compared > 0 && wrong_top == 0,
           "listed index " + name + ": " + std::to_string(wrong_lists) + " wrong list entries, " +
               std::to_string(differ) + " of " + std::to_string(compared) + " scores differ, " +
               std::to_string(wrong_top) + " wrong selections");
}

// ---- Attention ----------------------------------------------------------------------------------

struct AttentionCase {
    std::string name;
    std::vector<int> positions = {};
    std::vector<int> sequences =
        {}; // empty: a chunk, one staging ring and the latents of `sequence`
    int sequence = 0, ring = kRing, topk = 0, ratio = 2, splits = 1;
    float sink_offset = 0.0f;
    int window = kWindow; // a draft block's rows see the window and the block
    int floor  = 0;       // a prefill tail's first position: no window position below it
};

void sparse_attention(const AttentionCase& c, std::mt19937& rng) {
    const int t      = int(c.positions.size());
    const bool chunk = c.sequences.empty();
    const int seqs =
        chunk ? c.sequence + 1 : *std::max_element(c.sequences.begin(), c.sequences.end()) + 1;
    const int rings       = chunk ? 1 : seqs;
    const int rows        = *std::max_element(c.positions.begin(), c.positions.end()) / c.ratio + 3;
    const auto q          = random_bf16(std::size_t(t) * kHeads * kLatent, 2.0f, rng);
    const auto window     = random_window_rows(std::size_t(rings) * c.ring, rng);
    const auto compressed = random_latent_rows(std::size_t(seqs) * rows, rng);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> sinks(kHeads);
    for (auto& v : sinks) { v = 2.0f * normal(rng) + c.sink_offset; }
    // Each row's selection: distinct visible latents in ascending order, -1 past them.
    const int kk = std::max(c.topk, 1);
    std::vector<std::int32_t> topk(std::size_t(t) * kk, -1);
    for (int r = 0; c.topk > 0 && r < t; ++r) {
        std::vector<int> pick((c.positions[std::size_t(r)] + 1) / c.ratio);
        std::iota(pick.begin(), pick.end(), 0);
        std::shuffle(pick.begin(), pick.end(), rng);
        pick.resize(std::min<std::size_t>(pick.size(), std::size_t(c.topk)));
        std::sort(pick.begin(), pick.end());
        std::copy(pick.begin(), pick.end(), topk.begin() + std::ptrdiff_t(r) * kk);
    }
    const float scale = 1.0f / std::sqrt(float(kLatent));
    DeviceBuffer dq = upload(q), dw = upload(window), dc = upload(compressed), ds = upload(sinks),
                 dt = upload(topk), dp = upload(ints(c.positions)),
                 dseq = upload(ints(c.sequences)),
                 partial(std::max<std::size_t>(
                     k::sparse_attention_workspace_floats(t, kHeads, c.splits) * 4, 16)),
                 out(std::size_t(t) * kHeads * kLatent * 2);
    k::AttentionArgs a;
    a.q                     = as<bf16>(dq);
    a.tokens                = t;
    a.heads                 = kHeads;
    a.positions             = as<std::int32_t>(dp);
    a.sequences             = chunk ? nullptr : as<std::int32_t>(dseq);
    a.sequence              = c.sequence;
    a.window                = as<std::uint8_t>(dw);
    a.window_seq_stride     = chunk ? 0 : std::int64_t(c.ring) * k::kWindowRowBytes;
    a.ring                  = c.ring;
    a.window_size           = c.window;
    a.window_floor          = c.floor;
    a.compressed            = c.topk > 0 ? as<std::uint8_t>(dc) : nullptr;
    a.compressed_seq_stride = std::int64_t(rows) * k::kLatentRowBytes;
    a.topk                  = as<std::int32_t>(dt);
    a.k                     = c.topk;
    a.sinks                 = as<float>(ds);
    a.scale                 = scale;
    a.splits                = c.splits;
    a.partial               = c.splits > 1 ? as<float>(partial) : nullptr;
    a.out                   = as<bf16>(out);
    k::sparse_attention(a, nullptr);
    sync("sparse_attention");
    const auto got = download<bf16>(out, q.size());
    double num2 = 0.0, den2 = 0.0;
    for (int r = 0; r < t; ++r) {
        const int p  = c.positions[std::size_t(r)];
        const int s  = chunk ? c.sequence : c.sequences[std::size_t(r)];
        const int lo = std::max(std::max(0, std::min(c.floor, p)), p - c.window + 1);
        std::vector<std::vector<float>> keys;
        for (int j = lo; j <= p; ++j) {
            keys.push_back(window_values(
                window.data() + (std::size_t(chunk ? 0 : s) * c.ring + std::size_t(j % c.ring)) *
                                    k::kWindowRowBytes));
        }
        for (int i = 0; i < c.topk; ++i) {
            const int j = topk[std::size_t(r) * kk + i];
            if (j >= 0) {
                keys.push_back(latent_values(compressed.data() +
                                             (std::size_t(s) * rows + j) * k::kLatentRowBytes));
            }
        }
        for (int h = 0; h < kHeads; ++h) {
            const bf16* qh = q.data() + (std::size_t(r) * kHeads + h) * kLatent;
            std::vector<double> e(keys.size());
            double m = sinks[std::size_t(h)];
            for (std::size_t j = 0; j < keys.size(); ++j) {
                double d = 0.0;
                for (int i = 0; i < kLatent; ++i) { d += double(val(qh[i])) * keys[j][std::size_t(i)]; }
                e[j] = d * scale;
                m    = std::max(m, e[j]);
            }
            double l = std::exp(double(sinks[std::size_t(h)]) - m);
            for (auto& v : e) {
                v = std::exp(v - m);
                l += v;
            }
            for (int i = 0; i < kLatent; ++i) {
                double o = 0.0;
                for (std::size_t j = 0; j < keys.size(); ++j) { o += e[j] * keys[j][std::size_t(i)]; }
                o /= l;
                const double err =
                    double(val(got[(std::size_t(r) * kHeads + h) * kLatent + i])) - o;
                num2 += err * err;
                den2 += o * o;
            }
        }
    }
    const double rms = std::sqrt(num2 / den2);
    report(rms <= 3e-3, "sparse_attention " + c.name + ", " + std::to_string(c.splits) +
                            " splits: rms " + num(rms));
}

// The DSpark drafter's kernels: the taps' stream means into one target's columns, the Markov
// steps' argmax (a chain of two, ties to the lower token, nothing past the domain), and the
// confidence head.
void drafter(std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    {
        const int t = 5, targets = 3, j = 1;
        std::vector<float> h(std::size_t(t) * kStreams * kHidden);
        for (auto& v : h) { v = 4.0f * normal(rng); }
        DeviceBuffer dh = upload(h), out(std::size_t(t) * targets * kHidden * 4);
        out.fill(0xFF);
        k::stream_mean(as<float>(dh), t, kStreams, kHidden, as<float>(out) + j * kHidden,
                       std::int64_t(targets) * kHidden, nullptr);
        sync("stream_mean");
        const auto got = download<float>(out, std::size_t(t) * targets * kHidden);
        double worst = 0.0;
        bool kept    = true;
        for (int r = 0; r < t; ++r) {
            for (int c = 0; c < targets; ++c) {
                for (int i = 0; i < kHidden; ++i) {
                    const float g = got[(std::size_t(r) * targets + c) * kHidden + i];
                    if (c != j) {
                        kept = kept && std::isnan(g);
                        continue;
                    }
                    // FP32 sums carry an error of a few steps of the summands' magnitude, not of
                    // their (possibly cancelling) mean.
                    double want = 0.0, scale = 0.0;
                    for (int s = 0; s < kStreams; ++s) {
                        const double v = h[(std::size_t(r) * kStreams + s) * kHidden + i];
                        want += v;
                        scale += std::fabs(v);
                    }
                    want /= kStreams;
                    scale /= kStreams;
                    worse(worst, std::fabs(g - want) / (scale + 1e-30));
                }
            }
        }
        report(worst <= 1e-6 && kept, "stream_mean: worst " + num(worst) +
                                          (kept ? "" : ", other columns written"));
    }
    constexpr int kRank = k::kMarkovRank;
    const int vocab = 20000, domain = 19000;
    const auto head  = random_bf16(std::size_t(vocab) * kRank, 0.05f, rng);
    const auto embed = random_bf16(std::size_t(vocab) * kRank, 1.0f, rng);
    {
        std::vector<float> logits(2 * std::size_t(vocab));
        for (auto& v : logits) { v = 3.0f * normal(rng); }
        // Row 1: two equal leaders (same logit and head row) and a higher one past the domain.
        auto tied = head;
        float* second = logits.data() + vocab;
        const int a = 777, b = 15000;
        std::copy_n(tied.begin() + std::ptrdiff_t(a) * kRank, kRank, tied.begin() + std::ptrdiff_t(b) * kRank);
        second[a] = second[b] = 60.0f;
        second[domain + 5]    = 90.0f;
        const auto value = [&](const float* row, int prev, int v) {
            double dot = 0.0;
            for (int r = 0; r < kRank; ++r) {
                dot += double(val(tied[std::size_t(v) * kRank + r])) * val(embed[std::size_t(prev) * kRank + r]);
            }
            return double(row[v]) + dot;
        };
        DeviceBuffer dl = upload(logits), dh = upload(tied), de = upload(embed),
                     ids = upload(std::vector<std::int32_t>{4242, -1, -1}),
                     scratch(k::markov_scratch_bytes());
        k::markov_step(as<float>(dl), as<bf16>(dh), as<bf16>(de), domain, as<std::int32_t>(ids), 0,
                       scratch.p, nullptr);
        k::markov_step(as<float>(dl) + vocab, as<bf16>(dh), as<bf16>(de), domain,
                       as<std::int32_t>(ids), 1, scratch.p, nullptr);
        sync("markov_step");
        const auto got = download<std::int32_t>(ids, 3);
        // Step 0 against the oracle's best: the same token, or one within FP32's reach of it.
        int best = 0;
        for (int v = 1; v < domain; ++v) {
            if (value(logits.data(), 4242, v) > value(logits.data(), 4242, best)) { best = v; }
        }
        const bool first = got[1] == best || (got[1] >= 0 && got[1] < domain &&
                                               value(logits.data(), 4242, best) -
                                                       value(logits.data(), 4242, got[1]) <= 1e-4);
        report(first && got[2] == a, "markov_step: chain " + std::to_string(got[1]) + " (oracle " +
                                         std::to_string(best) + "), then " + std::to_string(got[2]) +
                                         " (the lower of two equal leaders, " + std::to_string(a) + ")");
    }
    {
        const int t = 5;
        std::vector<float> h(std::size_t(t) * kStreams * kHidden), pre(std::size_t(t) * kStreams),
            w(std::size_t(kHidden) + kRank);
        for (auto& v : h) { v = 2.0f * normal(rng); }
        for (auto& v : pre) { v = std::fabs(normal(rng)); }
        for (auto& v : w) { v = 0.01f * normal(rng); }
        const std::vector<std::int32_t> prev = {4242, 3, 19999, 0, 777};
        DeviceBuffer dh = upload(h), dp = upload(pre), dw = upload(w), de = upload(embed),
                     di = upload(prev), out(std::size_t(t) * 4);
        k::dspark_confidence(as<float>(dh), as<float>(dp), t, kStreams, kHidden, as<float>(dw),
                             as<bf16>(de), as<std::int32_t>(di), as<float>(out), nullptr);
        sync("dspark_confidence");
        const auto got = download<float>(out, t);
        double worst   = 0.0;
        for (int r = 0; r < t; ++r) {
            double z = 0.0;
            for (int i = 0; i < kHidden; ++i) {
                double x = 0.0;
                for (int s = 0; s < kStreams; ++s) {
                    x += double(pre[std::size_t(r) * kStreams + s]) *
                         h[(std::size_t(r) * kStreams + s) * kHidden + i];
                }
                z += x * w[std::size_t(i)];
            }
            for (int q = 0; q < kRank; ++q) {
                z += double(val(embed[std::size_t(prev[std::size_t(r)]) * kRank + q])) * w[kHidden + q];
            }
            worse(worst, std::fabs(got[std::size_t(r)] - 1.0 / (1.0 + std::exp(-z))));
        }
        report(worst <= 1e-5, "dspark_confidence: worst " + num(worst));
    }
}

// Group-major (out[g][t][heads / groups * 512]) and token-major (out[t][heads * 512]) layouts.
void attention_output(bool token_major, std::mt19937& rng) {
    const std::vector<int> positions = {0, 7, 300, 77777};
    const int t                      = int(positions.size());
    const int per                    = kHeads / kGroups;
    const auto freq                  = frequencies(160000.0);
    const auto o                     = random_bf16(std::size_t(t) * kHeads * kLatent, 0.5f, rng);
    DeviceBuffer d = upload(o), dp = upload(ints(positions)), df = upload(freq),
                 out(std::size_t(t) * kHeads * kLatent * 2);
    out.fill(0xFF);
    k::attention_output(as<bf16>(d), t, kHeads, kGroups, as<std::int32_t>(dp), as<float>(df),
                        token_major, as<bf16>(out), nullptr);
    sync("attention_output");
    const auto got = download<bf16>(out, o.size());
    Tally plain, turned;
    for (int r = 0; r < t; ++r) {
        for (int h = 0; h < kHeads; ++h) {
            std::vector<float> v(kLatent);
            for (int i = 0; i < kLatent; ++i) {
                v[std::size_t(i)] = val(o[(std::size_t(r) * kHeads + h) * kLatent + i]);
            }
            rotate_tail(v.data() + kLatent - kRope, positions[std::size_t(r)], freq, true);
            std::vector<double> step(kLatent);
            for (int g = 0; g < kLatent / 32; ++g) {
                fp8_round(v.data() + 32 * g, step.data() + 32 * g);
            }
            const std::size_t at =
                token_major ? (std::size_t(r) * kHeads + h) * kLatent
                            : (std::size_t(h / per) * t + r) * per * kLatent + std::size_t(h % per) * kLatent;
            for (int i = 0; i < kLatent; ++i) {
                (i < kLatent - kRope ? plain : turned)
                    .add(val(got[at + i]), v[std::size_t(i)], step[std::size_t(i)]);
            }
        }
    }
    report(plain.off == 0 && turned.within(1e-3, 2.0),
           std::string("attention_output ") + (token_major ? "token-major" : "group-major") +
               ": unturned " + plain.str() + ", turned " + turned.str());
}

// ---- Routing and experts ------------------------------------------------------------------------

// The router of `tokens` rows; with `ties`, every logit equal and no bias: the lowest ids.
// With `images`, every third row is an image's, which selects by the image routing bias.
void route(int tokens, bool ties, std::mt19937& rng, bool images = false) {
    std::normal_distribution<float> normal(0.0f, 2.0f);
    std::uniform_real_distribution<float> uniform(-0.1f, 0.1f);
    std::vector<float> logits(std::size_t(tokens) * kExperts), bias(kExperts), image_bias(kExperts);
    std::vector<std::uint8_t> image(static_cast<std::size_t>(tokens));
    for (auto& v : logits) { v = ties ? 0.5f : normal(rng); }
    for (auto& v : bias) { v = ties ? 0.0f : uniform(rng); }
    for (auto& v : image_bias) { v = 4.0f * uniform(rng); }
    for (int t = 0; images && t < tokens; ++t) { image[std::size_t(t)] = t % 3 == 1; }
    // The softplus's linear branch (past 20).
    for (int t = 0; !ties && t < tokens; ++t) {
        logits[std::size_t(t) * kExperts + (t * 37) % kExperts] = 25.0f;
    }
    const float scale = 1.5f;
    DeviceBuffer dl = upload(logits), db = upload(bias), di = upload(image_bias),
                 dm = upload(image), ids(std::size_t(tokens) * kTopK * 4),
                 weights(std::size_t(tokens) * kTopK * 4);
    k::route(as<float>(dl), as<float>(db), tokens, kExperts, kTopK, scale, as<std::int32_t>(ids),
             as<float>(weights), nullptr, images ? as<float>(di) : nullptr,
             images ? as<std::uint8_t>(dm) : nullptr);
    sync("route");
    const auto got_ids = download<std::int32_t>(ids, std::size_t(tokens) * kTopK);
    const auto got_w   = download<float>(weights, std::size_t(tokens) * kTopK);
    std::size_t wrong = 0, close = 0;
    double worst = 0.0;
    for (int t = 0; t < tokens; ++t) {
        std::vector<double> s(kExperts), sel(kExperts);
        for (int e = 0; e < kExperts; ++e) {
            s[std::size_t(e)] =
                std::sqrt(std::log1p(std::exp(double(logits[std::size_t(t) * kExperts + e]))));
            sel[std::size_t(e)] =
                s[std::size_t(e)] + (image[std::size_t(t)] ? image_bias : bias)[std::size_t(e)];
        }
        std::vector<int> order(kExperts);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](int x, int y) { return sel[std::size_t(x)] > sel[std::size_t(y)]; });
        // Choices within FP32's reach of each other may fall either way: those rows' ids are not
        // judged, unless they are the exact ties of equal inputs.
        bool ambiguous = false;
        for (int i = 0; !ties && i < kTopK; ++i) {
            ambiguous = ambiguous || sel[std::size_t(order[std::size_t(i)])] -
                                             sel[std::size_t(order[std::size_t(i) + 1])] <
                                         1e-5;
        }
        close += ambiguous ? 1 : 0;
        double sum = 0.0;
        for (int i = 0; i < kTopK; ++i) { sum += s[std::size_t(order[std::size_t(i)])]; }
        for (int i = 0; i < kTopK; ++i) {
            const std::size_t at = std::size_t(t) * kTopK + i;
            if (!ambiguous) { wrong += got_ids[at] != order[std::size_t(i)] ? 1 : 0; }
            const double want = s[std::size_t(order[std::size_t(i)])] / (sum + 1e-20) * scale;
            if (!ambiguous) { worse(worst, std::fabs(got_w[at] - want) / want); }
        }
    }
    report(wrong == 0 && worst <= 1e-5 && (tokens < 4 || close * 4 <= std::size_t(tokens)),
           std::string("route ") + (ties ? "ties" : std::to_string(tokens) + " rows") +
               (images ? " with image rows" : "") + ": " +
               std::to_string(wrong) + " wrong ids, weights " + num(worst) + ", " +
               std::to_string(close) + " rows too close to judge");
}

// A NaN logits row (an overflowed hidden state): its token selects experts 0..top_k-1, and the
// other rows select as they do without it.
void route_nan(std::mt19937& rng) {
    constexpr int kTokens = 3;
    std::normal_distribution<float> normal(0.0f, 2.0f);
    std::uniform_real_distribution<float> uniform(-0.1f, 0.1f);
    std::vector<float> logits(std::size_t(kTokens) * kExperts), bias(kExperts);
    for (auto& v : logits) { v = normal(rng); }
    for (auto& v : bias) { v = uniform(rng); }
    DeviceBuffer db = upload(bias), ids(std::size_t(kTokens) * kTopK * 4),
                 weights(std::size_t(kTokens) * kTopK * 4);
    const auto route = [&](const std::vector<float>& input) {
        DeviceBuffer dl = upload(input);
        k::route(as<float>(dl), as<float>(db), kTokens, kExperts, kTopK, 1.0f,
                 as<std::int32_t>(ids), as<float>(weights), nullptr);
        sync("route NaN row");
        return download<std::int32_t>(ids, std::size_t(kTokens) * kTopK);
    };
    const auto clean = route(logits);
    for (int e = 0; e < kExperts; ++e) { logits[std::size_t(kExperts) + e] = std::nanf(""); }
    const auto got = route(logits);
    bool ok = true;
    for (int t = 0; t < kTokens; ++t) {
        std::set<int> chosen;
        for (int j = 0; j < kTopK; ++j) {
            const int e = got[std::size_t(t) * kTopK + j];
            ok = ok && e >= 0 && e < kExperts && chosen.insert(e).second;
            // Every expert of the NaN row ties at the lowest score, so the lower index wins: the
            // row takes experts 0..top_k-1 (stale shared memory would leave other ids).
            ok = ok && e == (t == 1 ? j : clean[std::size_t(t) * kTopK + j]);
        }
    }
    report(ok, "route NaN row");
}

void swiglu(int tokens, bool scaled, std::mt19937& rng) {
    const int per = scaled ? kTopK : 1, rows = tokens * per;
    const std::int64_t x_stride = 2 * kWidth, out_stride = kWidth + 64;
    const float limit = 10.0f;
    std::normal_distribution<float> normal(0.0f, 6.0f);
    std::vector<float> x(std::size_t(rows) * x_stride), scale(static_cast<std::size_t>(tokens));
    for (auto& v : x) { v = normal(rng); }
    for (int t = 0; t < tokens; ++t) { scale[std::size_t(t)] = std::ldexp(1.0f, t % 4); }
    DeviceBuffer dx = upload(x), ds = upload(scale), out(std::size_t(rows) * out_stride * 2);
    out.fill(0xFF);
    k::clamped_swiglu(as<float>(dx), x_stride, rows, kWidth, limit,
                      scaled ? as<float>(ds) : nullptr, per, as<bf16>(out), out_stride, nullptr);
    sync("clamped_swiglu");
    const auto got = download<bf16>(out, std::size_t(rows) * out_stride);
    Tally tally;
    std::size_t stray = 0, clamped = 0;
    for (int r = 0; r < rows; ++r) {
        const double sc = scaled ? scale[std::size_t(r / per)] : 1.0;
        for (int i = 0; i < kWidth; ++i) {
            const double g  = bf(x[std::size_t(r) * x_stride + i] * sc);
            const double u  = bf(x[std::size_t(r) * x_stride + kWidth + i] * sc);
            const double gc = std::min(g, double(limit)),
                         uc = std::clamp(u, -double(limit), double(limit));
            clamped += gc != g || uc != u ? 1 : 0;
            const float want = bf(gc / (1.0 + std::exp(-gc)) * uc);
            // Past -88.7 e^-g overflows FP32 and the SiLU is zero; both products are below 2^-100
            // there and count as equal.
            const double value = val(got[std::size_t(r) * out_stride + i]);
            const bool tiny    = std::fabs(want) < 0x1p-100 && std::fabs(value) < 0x1p-100;
            tally.add(tiny ? want : value, want, bf16_step(want));
        }
        for (std::int64_t i = kWidth; i < out_stride; ++i) {
            stray += bits(got[std::size_t(r) * out_stride + i]) != kSentinel ? 1 : 0;
        }
    }
    report(tally.within(1e-3, 1.0) && stray == 0 && clamped > 0,
           "clamped_swiglu " + std::to_string(rows) + " rows" +
               (scaled ? " scaled per token" : "") + ": " + tally.str() + ", " +
               std::to_string(clamped) + " clamped, " + std::to_string(stray) + " stray");
}

void moe_combine(int tokens, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(0.0f, 0.5f);
    std::vector<float> y(std::size_t(tokens) * kHidden),
        rows(std::size_t(tokens) * kTopK * kHidden), weights(std::size_t(tokens) * kTopK);
    for (auto& v : y) { v = normal(rng); }
    for (auto& v : rows) { v = normal(rng); }
    for (auto& v : weights) { v = uniform(rng); }
    std::vector<unsigned long long> fixed(y.size());
    std::vector<double> fixed_value(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        const long long q = std::llround(double(10.0f * normal(rng)) * 4294967296.0);
        fixed[i]          = static_cast<unsigned long long>(q);
        fixed_value[i]    = double(q) * (1.0 / 4294967296.0);
    }
    DeviceBuffer dy = upload(y), dr = upload(rows), dw = upload(weights), dfx = upload(fixed);
    k::MoeCombineArgs a;
    a.y       = as<float>(dy);
    a.tokens  = tokens;
    a.hidden  = kHidden;
    a.rows    = as<float>(dr);
    a.weights = as<float>(dw);
    a.top_k   = kTopK;
    k::moe_combine(a, nullptr);
    sync("moe_combine");
    const auto got = download<float>(dy, y.size());
    double worst   = 0.0;
    for (int t = 0; t < tokens; ++t) {
        for (int i = 0; i < kHidden; ++i) {
            const std::size_t at = std::size_t(t) * kHidden + i;
            double want = y[at], magnitude = std::fabs(y[at]);
            for (int q = 0; q < kTopK; ++q) {
                const std::size_t pair = std::size_t(t) * kTopK + q;
                want += double(weights[pair]) * rows[pair * kHidden + i];
                magnitude += std::fabs(double(weights[pair]) * rows[pair * kHidden + i]);
            }
            worse(worst, std::fabs(got[at] - want) / magnitude);
        }
    }
    DeviceBuffer dy2 = upload(y);
    a.y              = as<float>(dy2);
    a.rows           = nullptr;
    a.weights        = nullptr;
    a.fixed          = as<unsigned long long>(dfx);
    k::moe_combine(a, nullptr);
    sync("moe_combine fixed");
    const auto got_fixed = download<float>(dy2, y.size());
    std::size_t wrong    = 0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const float want = y[i] + float(fixed_value[i]);
        wrong += got_fixed[i] != want ? 1 : 0;
    }
    report(worst <= 1e-6 && wrong == 0, "moe_combine " + std::to_string(tokens) +
                                            " rows: weighted " + num(worst) + ", fixed point " +
                                            std::to_string(wrong) + " wrong");
}

// ---- Engram -------------------------------------------------------------------------------------

void engram_rows(std::mt19937& rng) {
    constexpr int count = 72, row_bytes = 144, block_bytes = 18;
    constexpr int kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                1,    13,   25,  38,  53,  69,  89,  113};
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(1e-3f, 0.2f);
    std::vector<std::uint8_t> rows(std::size_t(count) * row_bytes);
    for (auto& b : rows) { b = std::uint8_t(byte(rng)); }
    for (std::size_t b = 0; b < rows.size(); b += block_bytes) {
        const __half d = __float2half((byte(rng) & 1 ? -1.0f : 1.0f) * scale(rng));
        std::memcpy(rows.data() + b, &d, 2);
    }
    DeviceBuffer dr = upload(rows), out(std::size_t(count) * 256 * 2);
    k::engram_rows(as<std::uint8_t>(dr), count, as<bf16>(out), nullptr);
    sync("engram_rows");
    const auto got    = download<bf16>(out, std::size_t(count) * 256);
    std::size_t wrong = 0;
    for (std::size_t block = 0; block < rows.size() / block_bytes; ++block) {
        const std::uint8_t* b = rows.data() + block * block_bytes;
        __half d;
        std::memcpy(&d, b, 2);
        for (int j = 0; j < 32; ++j) {
            const std::uint8_t q = b[2 + j % 16];
            const int code       = j < 16 ? (q & 15) : (q >> 4);
            const float want     = bf(double(__half2float(d)) * kIq4nl[code]);
            wrong += val(got[block * 32 + std::size_t(j)]) != want ? 1 : 0;
        }
    }
    report(wrong == 0, "engram_rows IQ4_NL: " + std::to_string(wrong) + " wrong");
}

// With `images`, odd rows are an image's, which the Engram leaves as they are.
void engram_gate(int tokens, std::mt19937& rng, bool images = false) {
    constexpr int S = kStreams, H = kHidden;
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> h(std::size_t(tokens) * S * H), kv(std::size_t(tokens) * (S + 1) * H),
        w(std::size_t(S) * H);
    for (int t = 0; t < tokens; ++t) {
        for (int s = 0; s < S; ++s) {
            for (int i = 0; i < H; ++i) {
                h[(std::size_t(t) * S + s) * H + i] = float(1 + s) * normal(rng);
            }
        }
    }
    for (auto& v : kv) { v = normal(rng); }
    for (auto& v : w) { v = 0.5f * normal(rng); }
    // Keys that match the streams (large dots of both signs) beside the random ones.
    for (int i = 0; i < H; ++i) {
        kv[std::size_t(0) * H + i] = 3.0f * h[i] * w[std::size_t(i)];
        kv[std::size_t(1) * H + i] = -2.0f * h[std::size_t(H) + i] * w[std::size_t(H) + i];
    }
    std::vector<std::uint8_t> image(static_cast<std::size_t>(tokens));
    for (int t = 0; images && t < tokens; ++t) { image[std::size_t(t)] = t % 2; }
    DeviceBuffer dh = upload(h), dkv = upload(kv), dw = upload(w), dm = upload(image);
    k::engram_gate(as<float>(dh), as<float>(dkv), as<float>(dw), tokens, S, H, kEps, nullptr,
                   images ? as<std::uint8_t>(dm) : nullptr);
    sync("engram_gate");
    const auto got = download<float>(dh, h.size());
    double worst = 0.0, low = 1.0, high = 0.0;
    std::size_t moved = 0;
    for (int t = 0; t < tokens; ++t) {
        if (image[std::size_t(t)]) {
            for (std::size_t i = 0; i < std::size_t(S) * H; ++i) {
                moved += got[std::size_t(t) * S * H + i] != h[std::size_t(t) * S * H + i];
            }
            continue;
        }
        const float* kt = kv.data() + std::size_t(t) * (S + 1) * H;
        for (int s = 0; s < S; ++s) {
            const float* hs = h.data() + (std::size_t(t) * S + s) * H;
            double dot = 0.0, hh = 0.0, kk = 0.0;
            for (int i = 0; i < H; ++i) {
                const double key = bf(kt[std::size_t(s) * H + i]);
                dot += double(hs[i]) * w[std::size_t(s) * H + i] * key;
                hh += double(hs[i]) * hs[i];
                kk += key * key;
            }
            const double d =
                dot / std::sqrt(hh / H + kEps) / std::sqrt(kk / H + kEps) / std::sqrt(double(H));
            const double gate = sigmoid(std::copysign(std::sqrt(std::max(std::fabs(d), 1e-6)), d));
            low               = std::min(low, gate);
            high              = std::max(high, gate);
            for (int i = 0; i < H; ++i) {
                const double v    = bf(kt[std::size_t(S) * H + i]);
                const double want = hs[i] + gate * v;
                worse(worst, std::fabs(got[(std::size_t(t) * S + s) * H + i] - want) /
                                 (std::fabs(hs[i]) + std::fabs(v)));
            }
        }
    }
    report(worst <= 1e-5 && moved == 0,
           "engram_gate " + std::to_string(tokens) + " rows" + (images ? " with image rows" : "") +
               ": worst " + num(worst) + ", gates " + num(low) + " .. " + num(high) +
               (images ? ", image values changed " + std::to_string(moved) : ""));
}

// The hash ids of a sequence with an image span (dead positions) in it, against an oracle that
// builds each position's n-gram window by the release's rule: the compressed ids of the token and
// the n - 1 before it, the pad id from the first dead position back and before the sequence; the
// same ids again when the sequence arrives in two calls split inside the span.
void engram_dead() {
    md::EngramHash hash;
    hash.layers     = {1, 3};
    hash.layer_rows = {100000, 200000};
    hash.ngram_size = 3;
    hash.heads      = 2;
    hash.pad_id     = 7;
    hash.multipliers = {{3, 1000003, 2000029}, {5, 1500007, 999983}};
    hash.primes      = {{9973, 9967, 9949, 9941}, {19997, 19993, 19991, 19979}};
    hash.offsets     = {{0, 9973, 19940, 29889}, {0, 19997, 39990, 59981}};
    std::vector<std::int32_t> token_map(64);
    for (std::size_t i = 0; i < token_map.size(); ++i) { token_map[i] = std::int32_t(i * 5 % 61 + 11); }
    const md::EngramHasher hasher(hash, token_map);
    const std::vector<std::int32_t> tokens = {5, 9, 40, 40, 40, 40, 12, 13, 14, 40, 15};
    const std::vector<std::uint8_t> dead   = {0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0};
    const std::size_t n = tokens.size(), per = hasher.rows_per_token();
    std::vector<std::uint64_t> want(n * per);
    for (std::size_t t = 0; t < n; ++t) {
        std::uint64_t window[3];
        bool blocked = false;
        for (std::size_t s = 0; s < 3; ++s) {
            blocked   = blocked || s > t || dead[t - s];
            window[s] = blocked ? std::uint64_t(hash.pad_id) : std::uint64_t(token_map[tokens[t - s]]);
        }
        std::uint64_t base = 0;
        for (std::size_t l = 0; l < 2; ++l) {
            for (std::uint32_t c = 0; c < hash.columns(); ++c) {
                const std::uint32_t order = c / hash.heads + 1; // n-gram of order + 1 tokens
                std::uint64_t mixed = window[0] * hash.multipliers[l][0];
                for (std::uint32_t i = 1; i <= order; ++i) { mixed ^= window[i] * hash.multipliers[l][i]; }
                want[t * per + l * hash.columns() + c] =
                    base + hash.offsets[l][c] + mixed % hash.primes[l][c];
            }
            base += hash.layer_rows[l];
        }
    }
    std::vector<std::uint64_t> whole(n * per), parts(n * per);
    md::EngramHasher::History history, split;
    hasher.rows(history, tokens, whole.data(), true, dead);
    const std::size_t cut = 4; // inside the span
    hasher.rows(split, std::span(tokens).first(cut), parts.data(), true, std::span(dead).first(cut));
    hasher.rows(split, std::span(tokens).subspan(cut), parts.data() + cut * per, true,
                std::span(dead).subspan(cut));
    std::size_t wrong = 0, wrong_parts = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        wrong += whole[i] != want[i];
        wrong_parts += parts[i] != want[i];
    }
    report(wrong == 0 && wrong_parts == 0,
           "engram dead positions: " + std::to_string(wrong) + " ids wrong in one call, " +
               std::to_string(wrong_parts) + " in two");
}

// An image's rows land in stream 0 of their positions (stride: the four streams of a position);
// the other streams and positions keep their values.
void place_rows(std::mt19937& rng) {
    constexpr int positions = 6, first = 2, count = 3;
    const std::int64_t stride = std::int64_t(kStreams) * kHidden;
    std::vector<float> h(std::size_t(positions) * stride, 1234.5f);
    const auto rows = random_bf16(std::size_t(count) * kHidden, 1.0f, rng);
    DeviceBuffer dh = upload(h), dr = upload(rows);
    k::place_rows(static_cast<const bf16*>(dr.p), count, kHidden, as<float>(dh) + first * stride,
                  stride, nullptr);
    sync("place_rows");
    const auto got = download<float>(dh, h.size());
    std::size_t wrong = 0;
    for (int t = 0; t < positions; ++t) {
        for (std::int64_t i = 0; i < stride; ++i) {
            const bool placed = t >= first && t < first + count && i < kHidden;
            const float want =
                placed ? val(rows[std::size_t(t - first) * kHidden + std::size_t(i)]) : 1234.5f;
            wrong += got[std::size_t(t * stride + i)] != want;
        }
    }
    report(wrong == 0, "place_rows: " + std::to_string(wrong) + " values wrong");
}

// ---- Guards and copies --------------------------------------------------------------------------

double largest_sum(const float* column, int rows) {
    double largest = 0.0;
    for (int g = 0; g < rows; g += 32) {
        double sum = 0.0;
        for (int i = g; i < g + 32; ++i) { sum += column[i]; }
        largest = std::max(largest, std::fabs(sum));
    }
    return largest;
}

void column_guard(std::mt19937& rng) {
    constexpr int rows                  = kHidden;
    const std::vector<float> magnitudes = {1, 50, 400, 600, 2000, 5000, 1e4, 3e4, 1e5, 1e6, 3, 7e6};
    const int columns                   = int(magnitudes.size());
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<bf16> x(std::size_t(columns) * rows);
    std::vector<float> xf(x.size());
    for (int c = 0; c < columns; ++c) {
        for (int r = 0; r < rows; ++r) {
            const std::size_t at = std::size_t(c) * rows + r;
            x[at]                = __float2bfloat16(magnitudes[std::size_t(c)] * normal(rng));
            xf[at]               = val(x[at]);
        }
    }
    DeviceBuffer dx = upload(x), ds(std::size_t(columns) * 4);
    k::guard_columns(as<bf16>(dx), rows, columns, as<float>(ds), nullptr);
    // The same columns in calls of at most six, as decode and verification calls guard theirs
    // (one wide block per column): the same scales and values.
    constexpr int narrow = 6;
    DeviceBuffer nx = upload(x), ns(std::size_t(columns) * 4);
    for (int c = 0; c < columns; c += narrow) {
        k::guard_columns(as<bf16>(nx) + std::size_t(c) * rows, rows, std::min(narrow, columns - c),
                         as<float>(ns) + c, nullptr);
    }
    sync("guard_columns");
    const auto got          = download<bf16>(dx, x.size());
    const auto scale        = download<float>(ds, std::size_t(columns));
    const auto narrow_got   = download<bf16>(nx, x.size());
    const auto narrow_scale = download<float>(ns, std::size_t(columns));
    std::size_t narrow_differ = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        narrow_differ += std::memcmp(&got[i], &narrow_got[i], sizeof(bf16)) != 0 ? 1 : 0;
    }
    for (int c = 0; c < columns; ++c) {
        narrow_differ += scale[std::size_t(c)] != narrow_scale[std::size_t(c)] ? 1 : 0;
    }
    std::size_t wrong = 0, scaled = 0, judged = 0;
    double after = 0.0;
    for (int c = 0; c < columns; ++c) {
        const float* column  = xf.data() + std::size_t(c) * rows;
        const double largest = largest_sum(column, rows);
        // A largest sum within FP32's reach of a power-of-two multiple of the limit may round
        // either way: such a column's exponent is not judged.
        const int lo = k::f16_sum_exponent(float(largest * (1 - 1e-5)));
        const int hi = k::f16_sum_exponent(float(largest * (1 + 1e-5)));
        if (lo != hi) { continue; }
        ++judged;
        scaled += lo > 0 ? 1 : 0;
        wrong += scale[std::size_t(c)] != std::ldexp(1.0f, lo) ? 1 : 0;
        std::vector<float> column_after(static_cast<std::size_t>(rows));
        for (int r = 0; r < rows; ++r) {
            column_after[std::size_t(r)] = val(got[std::size_t(c) * rows + r]);
            wrong += column_after[std::size_t(r)] != std::ldexp(column[r], -lo) ? 1 : 0;
        }
        after = std::max(after, largest_sum(column_after.data(), rows));
    }
    // unscale_columns multiplies the product's columns back, FP32 and BF16.
    constexpr int out_rows = 300, stride = 320;
    std::vector<float> out(std::size_t(columns) * stride);
    for (auto& v : out) { v = normal(rng); }
    std::vector<bf16> out16(out.size());
    for (std::size_t i = 0; i < out.size(); ++i) { out16[i] = __float2bfloat16(out[i]); }
    DeviceBuffer dout = upload(out), dout16 = upload(out16);
    k::unscale_columns(as<float>(dout), stride, out_rows, columns, as<float>(ds), nullptr);
    k::unscale_columns(as<bf16>(dout16), stride, out_rows, columns, as<float>(ds), nullptr);
    sync("unscale_columns");
    const auto back            = download<float>(dout, out.size());
    const auto back16          = download<bf16>(dout16, out16.size());
    std::size_t unscaled_wrong = 0;
    for (int c = 0; c < columns; ++c) {
        for (int r = 0; r < stride; ++r) {
            const std::size_t at = std::size_t(c) * stride + r;
            const float s        = r < out_rows ? scale[std::size_t(c)] : 1.0f;
            unscaled_wrong += back[at] != out[at] * s ? 1 : 0;
            unscaled_wrong += val(back16[at]) != val(out16[at]) * s ? 1 : 0;
        }
    }
    report(wrong == 0 && unscaled_wrong == 0 && narrow_differ == 0 && scaled >= 4 &&
               judged + 2 >= std::size_t(columns) && after <= double(k::kF16SumLimit),
           "guard_columns " + std::to_string(scaled) + " of " + std::to_string(judged) +
               " judged columns scaled, " + std::to_string(wrong) + " wrong, largest sum after " +
               num(after) + ", calls of " + std::to_string(narrow) + " columns " +
               std::to_string(narrow_differ) + " differ; unscale_columns " +
               std::to_string(unscaled_wrong) + " wrong");
}

void copies(std::mt19937& rng) {
    constexpr int small = 256, large = 768, row = k::kWindowRowBytes;
    const auto src = random_window_rows(small, rng);
    std::vector<std::uint8_t> dst(std::size_t(large) * row, kSentinelByte);
    DeviceBuffer ds = upload(src), dd = upload(dst),
                 back = upload(std::vector<std::uint8_t>(src.size(), kSentinelByte));
    k::ring_copy(as<std::uint8_t>(ds), small, as<std::uint8_t>(dd), large, 700, 1000, nullptr);
    k::ring_copy(as<std::uint8_t>(dd), large, as<std::uint8_t>(back), small, 872, 1000, nullptr);
    sync("ring_copy");
    const auto got      = download<std::uint8_t>(dd, dst.size());
    const auto got_back = download<std::uint8_t>(back, src.size());
    std::size_t wrong   = 0;
    for (int slot = 0; slot < large; ++slot) {
        int p = -1;
        for (int q = 700; q < 1000; ++q) { p = q % large == slot ? q : p; }
        for (int i = 0; i < row; ++i) {
            const std::uint8_t want = p < 0 ? kSentinelByte : src[std::size_t(p % small) * row + i];
            wrong += got[std::size_t(slot) * row + i] != want ? 1 : 0;
        }
    }
    for (int slot = 0; slot < small; ++slot) {
        int p = -1;
        for (int q = 872; q < 1000; ++q) { p = q % small == slot ? q : p; }
        for (int i = 0; i < row; ++i) {
            const std::uint8_t want = p < 0 ? kSentinelByte : src[std::size_t(p % small) * row + i];
            wrong += got_back[std::size_t(slot) * row + i] != want ? 1 : 0;
        }
    }
    report(wrong == 0,
           "ring_copy between rings of 256 and 768: " + std::to_string(wrong) + " wrong");

    // f32_to_bf16 rounds to nearest, ties to even, as the host conversion does.
    constexpr std::size_t n = 100003;
    std::normal_distribution<float> normal(0.0f, 100.0f);
    std::vector<float> in(n);
    for (std::size_t i = 0; i < n; ++i) {
        in[i] = normal(rng);
        if (i % 3 == 0) { // exact ties
            std::uint32_t u = 0;
            std::memcpy(&u, &in[i], 4);
            u = (u & 0xFFFF0000u) | 0x8000u;
            std::memcpy(&in[i], &u, 4);
        }
    }
    DeviceBuffer din = upload(in), dout(n * 2);
    k::f32_to_bf16(as<float>(din), as<bf16>(dout), std::int64_t(n), nullptr);
    sync("f32_to_bf16");
    const auto out              = download<bf16>(dout, n);
    std::size_t converted_wrong = 0;
    for (std::size_t i = 0; i < n; ++i) {
        converted_wrong += bits(out[i]) != bits(__float2bfloat16(in[i])) ? 1 : 0;
    }
    report(converted_wrong == 0, "f32_to_bf16: " + std::to_string(converted_wrong) + " wrong");
}

// ---- Routed experts on the CPU ------------------------------------------------------------------

// Random blocks of an expert type, their binary16 multipliers finite.
std::vector<std::uint8_t> expert_blocks(QType q, int rows, int depth, std::mt19937& rng) {
    const auto block = gguf_block_shape(q);
    std::vector<std::uint8_t> bytes(std::size_t(rows) * (depth / block.elements) * block.bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    for (auto& b : bytes) { b = static_cast<std::uint8_t>(byte(rng)); }
    for (std::size_t at = 0; at < bytes.size(); at += block.bytes) {
        const __half h = __float2half(scale(rng));
        std::uint16_t d = 0;
        std::memcpy(&d, &h, 2);
        if (q != QType::GGUF_IQ1_M) {
            std::memcpy(&bytes[at], &d, 2);
            continue;
        }
        // IQ1_M keeps its multiplier's four nibbles in the top of its four scale words.
        for (int j = 0; j < 4; ++j) {
            std::uint16_t w = 0;
            std::memcpy(&w, &bytes[at + 48 + 2 * j], 2);
            w = static_cast<std::uint16_t>((w & 0x0FFF) | (((d >> (4 * j)) & 0xF) << 12));
            std::memcpy(&bytes[at + 48 + 2 * j], &w, 2);
        }
    }
    return bytes;
}

// The grouped output projection of a decode or verification call as the executor runs it: the
// groups the experts of one vector product over the token-major activation, group g of token j
// pair j * groups + g. Each pair's rows must be the bits of its group's own vector product.
void grouped_output(int tokens, std::mt19937& rng) {
    constexpr int groups = kGroups, rows = 96, depth = 1024;
    const auto block       = gguf_block_shape(QType::GGUF_Q5_K);
    const std::int64_t row = std::int64_t(depth / block.elements) * block.bytes;
    auto weight            = expert_blocks(QType::GGUF_Q5_K, groups * rows, depth, rng);
    for (std::size_t at = 0; at < weight.size(); at += block.bytes) { // a finite minimum's factor
        std::memcpy(&weight[at + 2], &weight[at], 2);
    }
    const int pairs        = tokens * groups;
    const auto x           = random_bf16(std::size_t(pairs) * depth, 1.0f, rng);
    DeviceBuffer d_w = upload(weight), d_x = upload(x);
    std::vector<const void*> parts(groups);
    for (int g = 0; g < groups; ++g) {
        parts[std::size_t(g)] = static_cast<const std::uint8_t*>(d_w.p) + std::int64_t(g) * rows * row;
    }
    std::vector<std::int32_t> words;
    for (int g = 0; g <= groups; ++g) { words.push_back(g * tokens); }
    for (int g = 0; g < groups; ++g) {
        for (int j = 0; j < tokens; ++j) { words.push_back(j * groups + g); }
    }
    for (int g = 0; g < groups; ++g) { words.push_back(g); }
    words.push_back(groups);
    DeviceBuffer d_parts = upload(parts), d_words = upload(words);
    gguf::MoeRouting routing;
    routing.bounds       = as<std::int32_t>(d_words);
    routing.sorted       = routing.bounds + groups + 1;
    routing.active       = routing.sorted + pairs;
    routing.active_count = routing.active + groups;
    DeviceBuffer d_act(gguf::vector_activation_bytes(depth, pairs));
    gguf::quantize_vector_activation(as<bf16>(d_x), depth, pairs, nullptr, d_act.p, nullptr);
    DeviceBuffer grouped(std::size_t(pairs) * rows * 4);
    gguf::MoeOutput out;
    out.f32         = as<float>(grouped);
    const int chunk = tokens <= 1 ? 1 : tokens <= 2 ? 2 : tokens <= 4 ? 4 : 8;
    gguf::moe_vector_product(gguf::GgmlType::Q5_K,
                             {static_cast<const void* const*>(d_parts.p), row, rows, depth},
                             routing, groups, 1, 1, d_act.p, pairs, out, chunk, nullptr);
    // Each group alone: its tokens' columns gathered, as the per-group products read them.
    std::vector<bf16> xg(std::size_t(tokens) * depth);
    DeviceBuffer d_xg(xg.size() * 2), d_gact(gguf::vector_activation_bytes(depth, tokens)),
        single(std::size_t(tokens) * rows * 4);
    std::size_t differ = 0;
    const auto got = download<float>(grouped, std::size_t(pairs) * rows);
    for (int g = 0; g < groups; ++g) {
        for (int j = 0; j < tokens; ++j) {
            std::copy_n(x.begin() + (std::int64_t(j) * groups + g) * depth, depth,
                        xg.begin() + std::int64_t(j) * depth);
        }
        d_xg.copy_from_host(xg.data(), xg.size() * 2);
        gguf::quantize_vector_activation(as<bf16>(d_xg), depth, tokens, nullptr, d_gact.p, nullptr);
        gguf::VectorOutput one;
        one.f32           = as<float>(single);
        one.column_stride = rows;
        gguf::vector_product(gguf::GgmlType::Q5_K, parts[std::size_t(g)], row, rows, depth,
                             d_gact.p, tokens, one, nullptr);
        const auto want = download<float>(single, std::size_t(tokens) * rows);
        for (int j = 0; j < tokens; ++j) {
            for (int r = 0; r < rows; ++r) {
                const float a = got[(std::size_t(j) * groups + g) * rows + r];
                const float b = want[std::size_t(j) * rows + r];
                differ += std::memcmp(&a, &b, 4) != 0 ? 1 : 0;
            }
        }
    }
    report(differ == 0, "grouped output, " + std::to_string(tokens) + " token(s): " +
                            std::to_string(differ) + " values differ from the groups' own products");
}

// A call's routed experts as the executor runs them on the GPU (the vector kernels over the
// guarded input's q8_1 activation, the clamped SwiGLU, the middle's activation, the weighted down
// rows) and on the CPU from the inputs ExpertDelivery copies (CpuExperts): their 2^-32 fixed-point
// sums must be the same bits, with every expert on the CPU and with the GPU's experts of a half
// added to the CPU's of the other.
void cpu_experts(const char* name, QType gate_up_type, QType down_type, int tokens,
                 std::mt19937& rng) {
    constexpr int distinct = 3, pool_size = 14;
    constexpr float limit  = 10.0f;
    const int pairs        = tokens * kTopK;
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
    // Expert e's matrices are bank e % distinct's: [gate; up] pointers, then down.
    std::vector<const void*> table(2 * std::size_t(kExperts));
    for (int e = 0; e < kExperts; ++e) {
        table[std::size_t(e)]            = banks[2 * std::size_t(e % distinct)].p;
        table[std::size_t(kExperts + e)] = banks[2 * std::size_t(e % distinct) + 1].p;
    }
    DeviceBuffer d_table = upload(table);
    const auto* tables   = static_cast<const void* const*>(d_table.p);
    const gguf::MoeTable gu_table{tables, gu_row, 2 * kWidth, kHidden};
    const gguf::MoeTable down_table{tables + kExperts, down_row, kHidden, kWidth};

    // The last token large enough that the guard scales it.
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<bf16> x(std::size_t(tokens) * kHidden);
    for (int t = 0; t < tokens; ++t) {
        const float magnitude = t == tokens - 1 && tokens > 1 ? 3e4f : 1.0f;
        for (int i = 0; i < kHidden; ++i) {
            x[std::size_t(t) * kHidden + i] = __float2bfloat16(magnitude * normal(rng));
        }
    }
    // Each token's six experts from a small pool, so that tokens share experts.
    std::vector<std::int32_t> ids(static_cast<std::size_t>(pairs));
    std::vector<float> weights(ids.size());
    std::uniform_real_distribution<float> unit(0.05f, 2.5f);
    std::vector<int> pool(pool_size);
    std::iota(pool.begin(), pool.end(), 37);
    for (int t = 0; t < tokens; ++t) {
        std::shuffle(pool.begin(), pool.end(), rng);
        for (int j = 0; j < kTopK; ++j) {
            ids[std::size_t(t) * kTopK + j]     = pool[std::size_t(j)];
            weights[std::size_t(t) * kTopK + j] = unit(rng);
        }
    }
    DeviceBuffer d_x = upload(x), d_scale(std::size_t(tokens) * 4), d_ids = upload(ids),
                 d_w = upload(weights);
    k::guard_columns(as<bf16>(d_x), kHidden, tokens, as<float>(d_scale), nullptr);
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
        k::clamped_swiglu(as<float>(d_gu), 2 * kWidth, pairs, kWidth, limit, as<float>(d_scale),
                          kTopK, as<bf16>(d_middle), kWidth, nullptr);
        gguf::quantize_vector_activation(as<bf16>(d_middle), kWidth, pairs, nullptr, d_act.p,
                                         nullptr);
        gguf::moe_vector_product(ops::detail::gguf_type(down_type), down_table, part, max_active, 1,
                                 kTopK, d_act.p, pairs, out, chunk, nullptr);
    };
    gpu(routing);
    sync("routed experts");
    const auto whole = download<unsigned long long>(d_fixed, sums);

    md::CpuExperts cpu({md::CpuExpertLayer{gate_up_type, down_type, gu_row, down_row}}, kHidden,
                       kWidth, kTopK, limit);
    std::vector<std::byte> inputs(cpu.input_bytes()), products(cpu.product_bytes());
    d_xq.copy_to_host(inputs.data(), cpu.activation_bytes(tokens));
    d_w.copy_to_host(inputs.data() + cpu.activation_bytes(tokens), cpu.weight_bytes(tokens));
    d_scale.copy_to_host(inputs.data() + cpu.activation_bytes(tokens) + cpu.weight_bytes(tokens),
                         md::CpuExperts::scale_bytes(tokens));
    std::size_t done_bytes = 0;
    const auto on_cpu      = [&](const std::vector<std::int32_t>& experts) {
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
        done_bytes    = cpu.compute(work).bytes;
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
    // The even experts on the GPU (an active set as a stage of the call passes it), the odd ones
    // on the CPU, their sums added unless the flag is zero.
    std::vector<std::int32_t> set{std::int32_t(even.size())};
    set.insert(set.end(), even.begin(), even.end());
    DeviceBuffer d_set = upload(set), d_products = upload(on_cpu(odd));
    gguf::MoeRouting part = routing;
    part.active_count     = as<std::int32_t>(d_set);
    part.active           = as<std::int32_t>(d_set) + 1;
    gpu(part);
    sync("the even experts");
    const auto gpu_half = download<unsigned long long>(d_fixed, sums);
    const std::int32_t off = 0, on = 1;
    DeviceBuffer d_off = upload(std::vector<std::int32_t>{off}),
                 d_on  = upload(std::vector<std::int32_t>{on});
    k::add_fixed(as<unsigned long long>(d_fixed), d_products.p, as<std::int32_t>(d_off),
                 std::int64_t(sums), nullptr);
    sync("add_fixed off");
    const auto unchanged = download<unsigned long long>(d_fixed, sums);
    k::add_fixed(as<unsigned long long>(d_fixed), d_products.p, as<std::int32_t>(d_on),
                 std::int64_t(sums), nullptr);
    sync("add_fixed");
    const auto split    = download<unsigned long long>(d_fixed, sums);
    std::size_t split_wrong = 0, moved = 0;
    for (std::size_t i = 0; i < sums; ++i) {
        split_wrong += split[i] != whole[i] || unchanged[i] != gpu_half[i] ? 1 : 0;
        moved += gpu_half[i] != whole[i] ? 1 : 0;
    }
    report(wrong == 0 && split_wrong == 0 && zero * 100 < sums && moved * 2 > sums &&
               done_bytes == sums * 8 && !even.empty() && !odd.empty(),
           std::string("CPU experts ") + name + " T=" + std::to_string(tokens) + " (" +
               ops::gguf_cpu_backend_name(ops::GgufCpuBackend::Automatic) + "): " +
               std::to_string(active.size()) + " experts, " + std::to_string(wrong) +
               " sums differ all on the CPU, " + std::to_string(split_wrong) + " split, " +
               std::to_string(zero) + " zero");
}

std::vector<int> range(int first, int count) {
    std::vector<int> v(static_cast<std::size_t>(count));
    std::iota(v.begin(), v.end(), first);
    return v;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        std::mt19937 rng(4117);
        // The hyper-connection projection (24 x 4 * 5120) by GEMV and by SGEMM, the compressor's,
        // and a GEMV over partial float4s.
        for (const int rows : {1, 3, 8, 9}) { gemm(24, rows, kStreams * kHidden, rng); }
        gemm(512, 40, kHidden, rng);
        gemm(7, 5, 130, rng);
        {
            std::mt19937 own(5200); // a tail past the loads ahead, beside the other inputs
            gemm(24, 2, 5200, own);
        }

        hyper_connections(3, 20, rng);
        hyper_connections(1, 1, rng);
        hyper_connections(9, 3, rng);
        rms_rows(1, rng);
        rms_rows(13, rng);

        kv_post(
            {"decode rings of three sequences", {0, 1, 127, 128, 255, 5000}, {0, 1, 2, 0, 1, 2}},
            rng);
        kv_post({"a round's own slots",
                 {300, 301, 302, 300, 301},
                 {},
                 1,
                 kRing,
                 {300, 301, 302, 303, 304}},
                rng);
        kv_post({"a chunk's staging ring", range(1000, 32), {}, 2, 768, {}, true, 160000.0}, rng);
        rotations(rng);

        compress({"ratio 1 chunk", 1, range(300, 10), {}, 1}, rng);
        compress({"ratio 2 chunk from an even position", 2, range(100, 32), {}, 1, 16}, rng);
        compress({"ratio 2 chunk from an odd position", 2, range(101, 21), {}, 2, 5}, rng);
        compress(
            {"ratio 2 decode and a round", 2, {7, 8, 14, 15, 9, 10, 11}, {0, 1, 2, 2, 3, 3, 3}},
            rng);
        latent_post({"ratio 1 chunk", 1, range(64, 16), {}, 1}, rng);
        latent_post({"ratio 2 decode", 2, {33, 34, 35, 1, 0}, {0, 1, 1, 2, 3}}, rng);
        latent_post({"ratio 2 chunk", 2, range(1000, 16)}, rng);

        index_scores({"decode ratio 2", {100, 4097, 9, 0}, {0, 1, 2, 1}, 0, 2, 1}, rng);
        index_scores({"decode ratio 1", {3000, 1}, {1, 0}, 0, 1, 1}, rng);
        index_scores({"chunk ratio 1 in fours", range(1000, 10), {}, 1, 1, 4}, rng);
        index_scores({"chunk ratio 2 in fours", range(2001, 13), {}, 0, 2, 4}, rng);
        index_scores({"chunk with candidate blocks", range(500, 8), {}, 0, 1, 4, true}, rng);
        index_scores({"decode with candidate blocks", {700, 31}, {0, 1}, 0, 1, 1, true}, rng);

        select_topk({0, 98, 510, 511, 512, 1023, 1024, 2999, 8999}, 1, 512, 0, rng);
        select_topk({0, 1, 1999, 6001}, 2, 512, 1, rng);
        select_topk({3000, 700, 9000}, 1, 512, 2, rng);
        select_topk({9000, 2000}, 1, 512, 3, rng);
        select_topk({4999}, 1, 1024, 1, rng);
        select_candidates({99, 39999, 16391}, 1, 2048, 0, rng);
        select_candidates({0, 1, 20001}, 2, 300, 1, rng);
        select_candidates({29999, 4999}, 1, 2048, 2, rng);
        // A 2048-token chunk's rows early in a 1,000,000-position context (#263): fewer blocks than
        // kept, masks of 3,907 words, maxima of 1,024 floats a row.
        select_candidates({0, 2047, 4095, 8191}, 1, 2048, 0, rng, 1000000);
        select_candidates({999, 8190}, 1, 300, 2, rng, 1000000);
        select_candidates_stride();
        listed_index("decode rows", {40000, 16500, 70001}, {0, 1, 2}, 1, 2048, rng);
        listed_index("a chunk's rows in fours", range(30000, 14), {}, 4, 2048, rng);
        listed_index("a small keep", range(9000, 9), {}, 4, 64, rng);

        sparse_attention({"window only", {0, 50, 3000}, {0, 1, 2}, 0, kRing, 0, 2, 1}, rng);
        sparse_attention({"window only", {0, 50, 3000}, {0, 1, 2}, 0, kRing, 0, 2, 8}, rng);
        for (const int splits : {1, 5, 40}) {
            sparse_attention(
                {"window and 512 latents", {5000, 77, 1023}, {0, 1, 2}, 0, kRing, 512, 2, splits},
                rng);
        }
        sparse_attention({"a round's rows of one sequence",
                          {300, 301, 302, 303},
                          {1, 1, 1, 1},
                          0,
                          kRing,
                          512,
                          1,
                          4},
                         rng);
        sparse_attention({"a chunk's staging ring", range(200, 6), {}, 1, 768, 512, 1, 1}, rng);
        sparse_attention({"a chunk's rows over a window floor", range(150, 140), {}, 1, 768, 512, 1, 1,
                          0.0f, kWindow, 150},
                         rng);
        sparse_attention({"1024 latents", {9000}, {0}, 0, kRing, 1024, 2, 3}, rng);
        sparse_attention({"1024 latents, empty splits", {9000}, {0}, 0, kRing, 1024, 2, 40}, rng);
        sparse_attention({"sinks far below", {5000, 77}, {0, 1}, 0, kRing, 512, 2, 4, -40.0f}, rng);
        sparse_attention({"sinks above", {5000, 77}, {0, 1}, 0, kRing, 512, 2, 1, 8.0f}, rng);
        sparse_attention(
            {"a draft block", {304, 304, 304, 304, 304}, {1, 1, 1, 1, 1}, 0, kRing, 0, 2, 3, 0.0f, 133},
            rng);
        sparse_attention({"a draft block at the start", {6, 6, 6, 6, 6}, {0, 0, 0, 0, 0}, 0, kRing, 0,
                          2, 1, 0.0f, 133},
                         rng);
        attention_output(false, rng);
        attention_output(true, rng);
        grouped_output(1, rng);
        grouped_output(3, rng);
        grouped_output(8, rng);
        drafter(rng);

        route(1, false, rng);
        route(40, false, rng);
        route(1, true, rng);
        route(9, false, rng, true);
        route_nan(rng);
        swiglu(5, true, rng);
        swiglu(3, false, rng);
        moe_combine(1, rng);
        moe_combine(9, rng);

        engram_rows(rng);
        engram_gate(3, rng);
        engram_gate(4, rng, true);
        engram_dead();
        place_rows(rng);

        column_guard(rng);
        copies(rng);

        cpu_experts("IQ1_S/IQ1_M", QType::GGUF_IQ1_S, QType::GGUF_IQ1_M, 1, rng);
        cpu_experts("IQ2_XXS/IQ1_S", QType::GGUF_IQ2_XXS, QType::GGUF_IQ1_S, 3, rng);
        cpu_experts("IQ1_M/IQ2_S", QType::GGUF_IQ1_M, QType::GGUF_IQ2_S, 8, rng);
        cpu_experts("IQ2_S/IQ2_XXS", QType::GGUF_IQ2_S, QType::GGUF_IQ2_XXS, 5, rng);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " DeepSeek-V4.1 kernels\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "DeepSeek-V4.1 kernels: " << error.what() << '\n';
        return 1;
    }
}
