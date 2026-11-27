// The route bias toward held experts (core/expert_route_bias.h) in every family's router, against
// an FP64 oracle from the router's own FP32 logits (from the represented inputs for Flash-Next's,
// which keeps them in its workspace): the top_k by score plus the bias for each held expert (ties
// to the lower id), weighted by the unbiased scores renormalized over the chosen experts. The
// score is the one each router weights by: Inkling 2's sigmoid (plus its selection bias),
// DeepSeek-V4.1's sqrt-softplus (plus its selection bias), Flint's sigmoid, and the softmax
// probability of Gemma 4, Flash-Next and Qwen3.5/3.6 MoE. Held experts lie in the fixed range or
// in either device range of an ExpertDelivery-style bias, or in one of Flash-Next's six ranges.
// Each case of four or more tokens also checks that the bias changed some token's experts (a
// single token's experts may stay), and a choice whose biased score lies within 1e-5 of the next
// expert's counts as a tie.

#include "core/arena.h"
#include "core/device_buffer.h"
#include "core/expert_route_bias.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/deepseek_v41/kernels.h"
#include "models/flint/kernels.h"
#include "models/gemma4/kernels.h"
#include "models/inkling2/kernels.h"
#include "ninfer/ops/moe_route.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/sparse_moe/streamed.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gguf = ninfer::ops::gguf;

int failures = 0;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void report(bool ok, const std::string& what) {
    std::cout << (ok ? "OK   " : "FAIL ") << what << '\n';
    failures += ok ? 0 : 1;
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

std::vector<float> normal(std::size_t n, float sigma, std::mt19937& rng) {
    std::normal_distribution<float> d(0.0f, sigma);
    std::vector<float> v(n);
    for (auto& x : v) { x = d(rng); }
    return v;
}

std::uint16_t bf16_bits(float v) {
    std::uint32_t u = 0;
    std::memcpy(&u, &v, 4);
    return std::uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
float bf16_value(std::uint16_t b) {
    const std::uint32_t u = std::uint32_t(b) << 16;
    float v               = 0;
    std::memcpy(&v, &u, 4);
    return v;
}

// A table of `experts` entries whose held experts (about half, at random) lie in the bias's
// ranges: with `ranges` count 2 the fixed range and both device ranges (ExpertDelivery's form),
// with count 6 only device ranges (Flash-Next's misses).
struct Held {
    std::vector<bool> held;
    DeviceBuffer table, ranges;
    ExpertRouteBias bias;
};
Held make_held(int experts, int count, float bias, std::mt19937& rng) {
    Held h;
    h.held.resize(experts);
    constexpr std::uintptr_t kBase = 0x10000000, kSpan = 0x01000000;
    std::vector<std::uintptr_t> ranges(2 * count);
    for (int i = 0; i < count; ++i) {
        ranges[2 * i]     = kBase + std::uintptr_t(i + 1) * kSpan;
        ranges[2 * i + 1] = ranges[2 * i] + kSpan / 2;
    }
    const std::uintptr_t low = count == 2 ? kBase : 0, high = count == 2 ? kBase + kSpan / 2 : 0;
    std::vector<std::uintptr_t> table(experts);
    std::bernoulli_distribution coin(0.5);
    for (int e = 0; e < experts; ++e) {
        h.held[e] = coin(rng);
        // Held: the fixed range or a device range in turn; else past each range's end.
        const int slot = e % (count == 2 ? 3 : count);
        const std::uintptr_t start = count == 2 ? (slot == 2 ? low : ranges[2 * slot])
                                                : ranges[2 * slot];
        table[e] = start + std::uintptr_t(e) * 64 + (h.held[e] ? 0 : kSpan / 2);
    }
    h.table  = upload(table);
    h.ranges = upload(ranges);
    h.bias   = {.tables = static_cast<const void* const*>(h.table.p),
                .low    = low,
                .high   = high,
                .ranges = static_cast<const std::uintptr_t*>(h.ranges.p),
                .count  = count,
                .bias   = bias};
    return h;
}

// One token's expectation: the top_k by selection (ties to the lower id), the weights
// score / sum * factor[id], and whether the k-th choice is within 1e-5 of the next.
struct Expect {
    std::vector<int> ids;
    std::vector<double> weights;
    bool tie = false;
};
Expect expect(const std::vector<double>& selection, const std::vector<double>& score, int top_k,
              const std::function<double(int)>& factor) {
    std::vector<int> order(selection.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return selection[a] > selection[b]; });
    Expect out;
    out.ids.assign(order.begin(), order.begin() + top_k);
    for (int k = 0; k + 1 < int(order.size()) && k < top_k; ++k) {
        out.tie = out.tie || selection[order[k]] - selection[order[k + 1]] < 1e-5;
    }
    double sum = 0;
    for (const int e : out.ids) { sum += score[e]; }
    for (const int e : out.ids) { out.weights.push_back(score[e] / sum * factor(e)); }
    return out;
}

// Compares a call's ids and weights ([T][top_k]) with the biased expectation per token, and
// counts the tokens whose biased experts differ from the unbiased ones.
struct Tally {
    int wrong = 0, ties = 0, changed = 0;
    double weight_error = 0;
};
void compare(Tally& tally, const std::vector<int>& ids, const std::vector<float>& weights, int t,
             int top_k, const Expect& biased, const Expect& plain) {
    std::vector<int> a(biased.ids), b(plain.ids);
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    tally.changed += a != b ? 1 : 0;
    bool same = true;
    for (int k = 0; k < top_k; ++k) { same = same && ids[t * top_k + k] == biased.ids[k]; }
    if (!same) {
        if (biased.tie) {
            ++tally.ties;
        } else {
            ++tally.wrong;
        }
        return;
    }
    for (int k = 0; k < top_k; ++k) {
        const double want = biased.weights[k];
        tally.weight_error =
            std::max(tally.weight_error, std::abs(double(weights[t * top_k + k]) - want) /
                                             std::max(std::abs(want), 1e-6));
    }
}
void report_tally(const Tally& tally, int tokens, const std::string& what,
                  double weight_tolerance = 2e-5) {
    report(tally.wrong == 0 && tally.ties * 4 <= tokens && (tally.changed > 0 || tokens < 4) &&
               tally.weight_error < weight_tolerance,
           what + ": " + std::to_string(tokens) + " tokens, " + std::to_string(tally.changed) +
               " with other experts than unbiased, " + std::to_string(tally.wrong) + " wrong, " +
               std::to_string(tally.ties) + " ties, weight error " +
               std::to_string(tally.weight_error));
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
std::vector<double> softmax(const float* logits, int n) {
    double top = -INFINITY, sum = 0;
    for (int e = 0; e < n; ++e) { top = std::max(top, double(logits[e])); }
    std::vector<double> p(n);
    for (int e = 0; e < n; ++e) { sum += p[e] = std::exp(double(logits[e]) - top); }
    for (auto& v : p) { v /= sum; }
    return p;
}

// Inkling 2: selection sigmoid(logit) + bias, both route kernels.
void inkling2_case(int tokens, bool sorted, std::mt19937& rng) {
    namespace k = ninfer::models::inkling2::kernels;
    constexpr int kExperts = 128, kTopK = 8;
    constexpr float kScale = 2.5f, kBias = 0.1f;
    const auto logits = normal(std::size_t(tokens) * kExperts, 2.0f, rng);
    const auto bias   = normal(kExperts, 0.05f, rng);
    Held held         = make_held(kExperts, 2, kBias, rng);
    const int pairs   = tokens * kTopK;
    DeviceBuffer d_logits = upload(logits), d_bias = upload(bias), ids(pairs * 4),
                 weights(pairs * 4), pair(pairs * 4),
                 space(gguf::moe_routing_bytes(kExperts, pairs));
    k::RouteArgs a;
    a.tokens     = tokens;
    a.bias       = static_cast<const float*>(d_bias.p);
    a.experts    = kExperts;
    a.top_k      = kTopK;
    a.scale      = kScale;
    a.logits     = static_cast<float*>(d_logits.p);
    a.ids        = static_cast<std::int32_t*>(ids.p);
    a.weights    = static_cast<float*>(weights.p);
    a.pair_scale = static_cast<float*>(pair.p);
    a.held       = held.bias;
    if (sorted) {
        const gguf::MoeRouting r = gguf::moe_sorted_routes(space.p, pairs, kExperts);
        a.sorted = k::RouteSort{const_cast<std::int32_t*>(r.bounds),
                                const_cast<std::int32_t*>(r.sorted),
                                const_cast<std::int32_t*>(r.active),
                                const_cast<std::int32_t*>(r.active_count),
                                const_cast<std::int32_t*>(r.tile_count),
                                const_cast<std::int32_t*>(r.tiles), r.tile_capacity};
    }
    k::route(a, nullptr);
    check(cudaDeviceSynchronize(), "inkling2 route");
    const auto got_ids = download<int>(ids, pairs);
    const auto got_w   = download<float>(weights, pairs);
    Tally tally;
    for (int t = 0; t < tokens; ++t) {
        std::vector<double> score(kExperts), plain(kExperts), biased(kExperts);
        for (int e = 0; e < kExperts; ++e) {
            score[e]  = sigmoid(logits[std::size_t(t) * kExperts + e]);
            plain[e]  = score[e] + bias[e];
            biased[e] = plain[e] + (held.held[e] ? kBias : 0.0);
        }
        const auto f = [](int) { return double(kScale); };
        compare(tally, got_ids, got_w, t, kTopK, expect(biased, score, kTopK, f),
                expect(plain, score, kTopK, f));
    }
    report_tally(tally, tokens, std::string("inkling2 ") + (sorted ? "sorted route" : "route"));
}

// DeepSeek-V4.1: selection sqrt(softplus(logit)) + bias.
void deepseek_case(int tokens, std::mt19937& rng) {
    namespace k = ninfer::models::deepseek_v41::kernels;
    constexpr int kExperts = 256, kTopK = 6;
    constexpr float kScale = 1.5f, kBias = 0.1f;
    const auto logits = normal(std::size_t(tokens) * kExperts, 2.0f, rng);
    const auto bias   = normal(kExperts, 0.05f, rng);
    Held held         = make_held(kExperts, 2, kBias, rng);
    const int pairs   = tokens * kTopK;
    DeviceBuffer d_logits = upload(logits), d_bias = upload(bias), ids(pairs * 4),
                 weights(pairs * 4);
    k::route(static_cast<const float*>(d_logits.p), static_cast<const float*>(d_bias.p), tokens,
             kExperts, kTopK, kScale, static_cast<std::int32_t*>(ids.p),
             static_cast<float*>(weights.p), nullptr, nullptr, nullptr, held.bias);
    check(cudaDeviceSynchronize(), "deepseek route");
    const auto got_ids = download<int>(ids, pairs);
    const auto got_w   = download<float>(weights, pairs);
    Tally tally;
    for (int t = 0; t < tokens; ++t) {
        std::vector<double> score(kExperts), plain(kExperts), biased(kExperts);
        for (int e = 0; e < kExperts; ++e) {
            const double x = logits[std::size_t(t) * kExperts + e];
            score[e]       = std::sqrt(x > 20.0 ? x : std::log1p(std::exp(x)));
            plain[e]       = score[e] + bias[e];
            biased[e]      = plain[e] + (held.held[e] ? kBias : 0.0);
        }
        const auto f = [](int) { return double(kScale); };
        compare(tally, got_ids, got_w, t, kTopK, expect(biased, score, kTopK, f),
                expect(plain, score, kTopK, f));
    }
    report_tally(tally, tokens, "deepseek_v41 route");
}

// Gemma 4 and Flint: the router's logits from x (their own FP32 logits are the oracle's input),
// route() and route_select().
template <bool Gemma>
void gemma_flint_case(int tokens, bool select, std::mt19937& rng) {
    constexpr int kHidden = 256, kExperts = Gemma ? 128 : 160, kTopK = 8;
    constexpr float kBias = Gemma ? 0.02f : 0.1f;
    const auto x      = normal(std::size_t(tokens) * kHidden, 1.0f, rng);
    const auto router = normal(std::size_t(kExperts) * kHidden, 0.15f, rng);
    std::vector<float> scale(kExperts);
    std::uniform_real_distribution<float> unit(0.5f, 2.0f);
    for (auto& v : scale) { v = unit(rng); }
    Held held       = make_held(kExperts, 2, kBias, rng);
    const int pairs = tokens * kTopK;
    DeviceBuffer d_x = upload(x), d_router = upload(router), d_scale = upload(scale),
                 logits(std::size_t(tokens) * kExperts * 4), ids(pairs * 4), weights(pairs * 4),
                 space(gguf::moe_routing_bytes(kExperts, pairs)), done(16);
    check(cudaMemset(done.p, 0, done.bytes), "zero the counter");
    auto* base    = static_cast<std::int32_t*>(space.p);
    const auto* xp = static_cast<const float*>(d_x.p);
    const auto* rp = static_cast<const float*>(d_router.p);
    auto* lp       = static_cast<float*>(logits.p);
    auto* ip       = static_cast<std::int32_t*>(ids.p);
    auto* wp       = static_cast<float*>(weights.p);
    if constexpr (Gemma) {
        namespace k     = ninfer::models::gemma4::kernels;
        const auto* sp = static_cast<const float*>(d_scale.p);
        if (select) {
            k::route_select(xp, kHidden, tokens, rp, sp, kExperts, kTopK, lp, ip, wp, base,
                            base + kExperts + 1, base + kExperts + 1 + pairs,
                            base + kExperts + 1 + pairs + kExperts, nullptr, held.bias);
        } else {
            k::route(xp, kHidden, tokens, rp, sp, kExperts, kTopK, lp, ip, wp, nullptr,
                     held.bias);
        }
    } else {
        namespace k = ninfer::models::flint::kernels;
        if (select) {
            k::route_select(xp, kHidden, tokens, rp, kExperts, kTopK, lp, ip, wp, base,
                            base + kExperts + 1, base + kExperts + 1 + pairs,
                            base + kExperts + 1 + pairs + kExperts,
                            static_cast<std::uint32_t*>(done.p), nullptr, held.bias);
        } else {
            k::route(xp, kHidden, tokens, rp, kExperts, kTopK, lp, ip, wp, nullptr, held.bias);
        }
    }
    check(cudaDeviceSynchronize(), Gemma ? "gemma4 route" : "flint route");
    const auto got_logits = download<float>(logits, std::size_t(tokens) * kExperts);
    const auto got_ids    = download<int>(ids, pairs);
    const auto got_w      = download<float>(weights, pairs);
    Tally tally;
    for (int t = 0; t < tokens; ++t) {
        const float* row = got_logits.data() + std::size_t(t) * kExperts;
        std::vector<double> score(kExperts), plain(kExperts), biased(kExperts);
        const auto p = softmax(row, kExperts);
        for (int e = 0; e < kExperts; ++e) {
            score[e]  = Gemma ? p[e] : sigmoid(row[e]);
            plain[e]  = Gemma ? double(row[e]) : score[e];
            biased[e] = score[e] + (held.held[e] ? kBias : 0.0);
        }
        const auto f = [&](int e) { return Gemma ? double(scale[e]) : 1.0; };
        compare(tally, got_ids, got_w, t, kTopK, expect(biased, score, kTopK, f),
                expect(plain, score, kTopK, f));
    }
    report_tally(tally, tokens,
                 std::string(Gemma ? "gemma4 " : "flint ") + (select ? "route_select" : "route"));
}

// Flash-Next (and Edge0): moe_route, its logits from the BF16 inputs in FP64.
void flash_next_case(int tokens, bool counter, std::mt19937& rng) {
    constexpr int kHidden = 2560, kExperts = 512, kTop = 10;
    constexpr float kBias = 0.02f;
    auto m      = normal(std::size_t(tokens) * kHidden, 1.0f, rng);
    auto router = normal(std::size_t(kExperts) * kHidden, 0.04f, rng);
    auto gate   = normal(kHidden, 0.04f, rng);
    std::vector<std::uint16_t> mb(m.size()), rb(router.size()), gb(gate.size());
    for (std::size_t i = 0; i < m.size(); ++i) { m[i] = bf16_value(mb[i] = bf16_bits(m[i])); }
    for (std::size_t i = 0; i < router.size(); ++i) {
        router[i] = bf16_value(rb[i] = bf16_bits(router[i]));
    }
    for (std::size_t i = 0; i < gate.size(); ++i) { gb[i] = bf16_bits(gate[i]); }
    Held held = make_held(kExperts, 6, kBias, rng);
    DeviceBuffer d_m = upload(mb), d_router = upload(rb), d_gate = upload(gb),
                 ids(std::size_t(kTop) * tokens * 4), weights(std::size_t(kTop) * tokens * 4),
                 shared(std::size_t(tokens) * 4), done(16);
    check(cudaMemset(done.p, 0, done.bytes), "zero the counter");
    Tensor t_m(d_m.p, DType::BF16, {kHidden, tokens});
    Tensor t_router(d_router.p, DType::BF16, {kHidden, kExperts});
    Tensor t_gate(d_gate.p, DType::BF16, {kHidden});
    Tensor t_ids(ids.p, DType::I32, {kTop, tokens});
    Tensor t_weights(weights.p, DType::FP32, {kTop, tokens});
    Tensor t_shared(shared.p, DType::FP32, {tokens});
    WorkspaceArena workspace(ops::moe_route_workspace_bytes(tokens, kExperts));
    ops::moe_route(t_m, t_router, t_gate, workspace, t_ids, t_weights, t_shared, nullptr,
                   counter ? static_cast<std::uint32_t*>(done.p) : nullptr, held.bias);
    check(cudaDeviceSynchronize(), "moe_route");
    const auto got_ids = download<int>(ids, std::size_t(kTop) * tokens);
    const auto got_w   = download<float>(weights, std::size_t(kTop) * tokens);
    Tally tally;
    for (int t = 0; t < tokens; ++t) {
        std::vector<float> row(kExperts);
        for (int e = 0; e < kExperts; ++e) {
            double dot = 0;
            for (int d = 0; d < kHidden; ++d) {
                dot += double(router[std::size_t(e) * kHidden + d]) * m[std::size_t(t) * kHidden + d];
            }
            row[e] = float(dot);
        }
        const auto p = softmax(row.data(), kExperts);
        std::vector<double> plain(kExperts), biased(kExperts);
        for (int e = 0; e < kExperts; ++e) {
            plain[e]  = p[e];
            biased[e] = p[e] + (held.held[e] ? kBias : 0.0);
        }
        const auto f = [](int) { return 1.0; };
        // FP32 logits differ from FP64 ones: a near tie at 1e-5 of the probabilities counts.
        compare(tally, got_ids, got_w, t, kTop, expect(biased, p, kTop, f),
                expect(plain, p, kTop, f));
    }
    // The device's FP32 logits move the weights by about their rounding.
    report_tally(tally, tokens, std::string("flash-next moe_route") + (counter ? " fused" : ""),
                 2e-3);
}

// Qwen3.5/3.6 MoE: the streamed route of 1 (decode) or several columns; the oracle reads the
// router's FP32 scores from the workspace (S1's four partial sums added as S2 adds them).
void qwen35_case(int tokens, std::mt19937& rng) {
    constexpr int kHidden = 2048, kRows = 257, kExperts = 256, kTop = 8;
    constexpr float kBias = 0.02f;
    auto x      = normal(std::size_t(tokens) * kHidden, 1.0f, rng);
    auto router = normal(std::size_t(kRows) * kHidden, 0.04f, rng);
    std::vector<std::uint16_t> xb(x.size()), rb(router.size());
    for (std::size_t i = 0; i < x.size(); ++i) { xb[i] = bf16_bits(x[i]); }
    for (std::size_t i = 0; i < router.size(); ++i) { rb[i] = bf16_bits(router[i]); }
    Held held        = make_held(kExperts, 2, kBias, rng);
    DeviceBuffer d_x = upload(xb), d_router = upload(rb);
    Tensor t_x(d_x.p, DType::BF16, {kHidden, tokens});
    Weight w;
    w.qdata = d_router.p;
    WorkspaceArena arena(std::size_t(64) << 20);
    const auto views = ops::detail::allocate_sparse_moe_small_t_workspace(arena, tokens);
    ops::detail::sparse_moe_stream_route(t_x, w, views, nullptr, held.bias);
    check(cudaDeviceSynchronize(), "sparse_moe_stream_route");
    const int pairs = tokens * kTop;
    std::vector<int> got_ids(pairs);
    std::vector<float> got_w(pairs);
    check(cudaMemcpy(got_ids.data(), views.token_ids.data, pairs * 4, cudaMemcpyDeviceToHost), "ids");
    check(cudaMemcpy(got_w.data(), views.token_alpha.data, pairs * 4, cudaMemcpyDeviceToHost), "alpha");
    const std::size_t scratch = tokens == 1 ? kRows : std::size_t(tokens) * kRows * 4;
    std::vector<float> partial(scratch);
    check(cudaMemcpy(partial.data(), views.scratch.data, scratch * 4, cudaMemcpyDeviceToHost),
          "scores");
    Tally tally;
    for (int t = 0; t < tokens; ++t) {
        std::vector<float> row(kExperts);
        for (int e = 0; e < kExperts; ++e) {
            if (tokens == 1) {
                row[e] = partial[e];
            } else {
                float sum = 0.0f;
                for (int p = 0; p < 4; ++p) { sum += partial[(std::size_t(t) * kRows + e) * 4 + p]; }
                row[e] = sum;
            }
        }
        const auto p = softmax(row.data(), kExperts);
        std::vector<double> biased(kExperts);
        for (int e = 0; e < kExperts; ++e) { biased[e] = p[e] + (held.held[e] ? kBias : 0.0); }
        const auto f = [](int) { return 1.0; };
        compare(tally, got_ids, got_w, t, kTop, expect(biased, p, kTop, f),
                expect(p, p, kTop, f));
    }
    report_tally(tally, tokens, "qwen3.5 streamed route");
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cout << "SKIP no CUDA device\n";
        return 77;
    }
    try {
        std::mt19937 rng(20261127);
        for (const int t : {1, 5, 37}) { inkling2_case(t, false, rng); }
        for (const int t : {1, 8}) { inkling2_case(t, true, rng); }
        for (const int t : {1, 6, 40}) { deepseek_case(t, rng); }
        for (const int t : {1, 8}) {
            gemma_flint_case<true>(t, true, rng);
            gemma_flint_case<false>(t, true, rng);
        }
        for (const int t : {3, 24}) {
            gemma_flint_case<true>(t, false, rng);
            gemma_flint_case<false>(t, false, rng);
        }
        for (const int t : {1, 8, 20}) { flash_next_case(t, false, rng); }
        flash_next_case(4, true, rng);
        for (const int t : {1, 4, 30}) { qwen35_case(t, rng); }
    } catch (const std::exception& error) {
        std::cout << "FAIL " << error.what() << '\n';
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
