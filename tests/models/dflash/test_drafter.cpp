// The DFlash drafter's configuration and its own kernels (models/dflash):
//  - parse_config over the released Gemma 4 DFlash and DSpark configs (five layers, 64 query
//    heads over 8 KV heads, a full-attention fifth layer, the Markov head's rank), and the fields
//    it refuses; check_target against another width, vocabulary or tap count; full_window.
//  - softcap against cap * tanh(x / cap) in FP64; best_candidate's drafts, one-hot laws and
//    confidences (the first candidate's softmax weight over the 16 scores, -inf slots weighing
//    nothing, 0 after a non-finite first score); fill_bf16.

#include "core/device_buffer.h"
#include "models/dflash/config.h"
#include "models/dflash/drafter.h"
#include "models/dflash/kernels.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using artifact::Json;
namespace dflash = ninfer::models::dflash;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

// The converter's config of manateelazycat's Gemma-4-31B-IT-Heretic-Draft-DFlash.
Json released(std::uint32_t markov_rank) {
    return Json{{"hidden_size", 5376},
                {"intermediate_size", 10752},
                {"num_hidden_layers", 5},
                {"num_attention_heads", 64},
                {"num_key_value_heads", 8},
                {"head_dim", 128},
                {"vocab_size", 262144},
                {"num_target_layers", 60},
                {"block_size", 16},
                {"sliding_window", 2048},
                {"layer_types", {"sliding_attention", "sliding_attention", "sliding_attention",
                                 "sliding_attention", "full_attention"}},
                {"rms_norm_eps", 1e-6},
                {"rope_theta", 1e6},
                {"target_layer_ids", {1, 12, 23, 35, 46, 57}},
                {"mask_token_id", 4},
                {"logit_softcap", 30.0},
                {"markov_rank", markov_rank}};
}

bool refused(const std::function<void()>& call) {
    try {
        call();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

void configs() {
    const dflash::Config c = dflash::parse_config(released(0));
    expect(c.taps() == 6 && c.query_width() == 8192 && c.key_width() == 1024 &&
               c.query_groups() == 2 && c.has_full_attention() && c.full_attention[4] &&
               !c.full_attention[3] && c.markov_rank == 0 && c.mask_token_id == 4 &&
               c.logit_softcap == 30.0F,
           "the released DFlash config");
    expect(dflash::parse_config(released(256)).markov_rank == 256, "the released DSpark config");
    // The full-attention ring: the smallest of 2048, 4096 and 8192 that holds the context and a
    // block, at most 8192.
    expect(dflash::full_window(c, 1000, 8) == 2048 && dflash::full_window(c, 2040, 8) == 2048 &&
               dflash::full_window(c, 2041, 8) == 4096 &&
               dflash::full_window(c, 4097, 8) == 8192 &&
               dflash::full_window(c, 100000, 8) == 8192,
           "full_window");
    expect(dflash::layer_capacity(c, 0, 8192) == 2048 && dflash::layer_capacity(c, 4, 8192) == 8192,
           "layer_capacity");
    const auto with = [](const char* field, Json value) {
        Json config   = released(0);
        config[field] = std::move(value);
        return config;
    };
    for (const auto& [what, config] : std::vector<std::pair<std::string, Json>>{
             {"48 query heads", with("num_attention_heads", 48)},
             {"4 KV heads", with("num_key_value_heads", 4)},
             {"head width 64", with("head_dim", 64)},
             {"a rank-128 Markov head", with("markov_rank", 128)},
             {"window 1024", with("sliding_window", 1024)},
             {"a 17-wide block", with("block_size", 17)},
             {"a tap past the target", with("target_layer_ids", Json{1, 60})},
             {"a repeated tap", with("target_layer_ids", Json{1, 1})},
             {"a missing layer type", with("layer_types", Json{"sliding_attention"})},
             {"an unknown layer type", with("layer_types", Json{"a", "b", "c", "d", "e"})},
             {"a mask outside the vocabulary", with("mask_token_id", 262144)},
         }) {
        expect(refused([&] { (void)dflash::parse_config(config); }), "refuses " + what);
    }
    Json missing = released(0);
    missing.erase("rope_theta");
    expect(refused([&] { (void)dflash::parse_config(missing); }), "refuses a missing field");
    expect(!refused([&] { dflash::check_target(c, 5376, 262144, 60, "Gemma 4"); }) &&
               refused([&] { dflash::check_target(c, 2816, 262144, 60, "Gemma 4"); }) &&
               refused([&] { dflash::check_target(c, 5376, 262143, 60, "Gemma 4"); }) &&
               refused([&] { dflash::check_target(c, 5376, 262144, 30, "Gemma 4"); }),
           "check_target");
}

void kernels(std::mt19937& rng) {
    constexpr int kC = 16, kU = 21;
    std::normal_distribution<float> normal(0.0F, 20.0F);
    // softcap
    std::vector<float> x(kC * kU);
    for (auto& v : x) { v = normal(rng); }
    DeviceBuffer scores(x.size() * 4);
    scores.copy_from_host(x.data(), x.size() * 4);
    Tensor s(scores.p, DType::FP32, {kC, kU});
    dflash::kernels::softcap(s, 30.0F, nullptr);
    check(cudaDeviceSynchronize(), "softcap");
    std::vector<float> capped(x.size());
    scores.copy_to_host(capped.data(), capped.size() * 4);
    double worst = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double ideal = 30.0 * std::tanh(double(x[i]) / 30.0);
        worst              = std::max(worst, std::abs(double(capped[i]) - ideal));
    }
    expect(worst <= 1e-5, "softcap within 1e-5 of FP64 (" + std::to_string(worst) + ")");
    // best_candidate: descending scores per column, a -inf tail in column 1, a NaN first score in
    // column 2.
    std::vector<float> sc(kC * kU);
    std::vector<std::int32_t> ids(kC * kU);
    for (int u = 0; u < kU; ++u) {
        float top = normal(rng);
        for (int c = 0; c < kC; ++c) {
            sc[u * kC + c]  = top;
            ids[u * kC + c] = u * 100 + c;
            top -= std::abs(normal(rng)) * 0.1F;
        }
    }
    for (int c = 5; c < kC; ++c) { sc[1 * kC + c] = -std::numeric_limits<float>::infinity(); }
    sc[2 * kC] = std::numeric_limits<float>::quiet_NaN();
    DeviceBuffer d_ids(ids.size() * 4), d_scores(sc.size() * 4), d_drafts(kU * 4),
        d_confidence(kU * 4), d_q(sc.size() * 4);
    d_ids.copy_from_host(ids.data(), ids.size() * 4);
    d_scores.copy_from_host(sc.data(), sc.size() * 4);
    d_q.fill(0x7f);
    const Tensor t_ids(d_ids.p, DType::I32, {kC, 3, 7});
    const Tensor t_scores(d_scores.p, DType::FP32, {kC, 3, 7});
    Tensor t_drafts(d_drafts.p, DType::I32, {3, 7});
    Tensor t_confidence(d_confidence.p, DType::FP32, {3, 7});
    Tensor t_q(d_q.p, DType::FP32, {kC, 3, 7});
    dflash::kernels::best_candidate(t_ids, t_scores, t_drafts, t_confidence, t_q, nullptr);
    check(cudaDeviceSynchronize(), "best_candidate");
    std::vector<std::int32_t> drafts(kU);
    std::vector<float> confidence(kU), q(sc.size());
    d_drafts.copy_to_host(drafts.data(), drafts.size() * 4);
    d_confidence.copy_to_host(confidence.data(), confidence.size() * 4);
    d_q.copy_to_host(q.data(), q.size() * 4);
    for (int u = 0; u < kU; ++u) {
        double ideal = 0;
        if (std::isfinite(sc[u * kC])) {
            double sum = 0;
            for (int c = 0; c < kC; ++c) { sum += std::exp(double(sc[u * kC + c]) - sc[u * kC]); }
            ideal = 1.0 / sum;
        }
        bool one_hot = q[u * kC] == 1.0F;
        for (int c = 1; c < kC; ++c) { one_hot = one_hot && q[u * kC + c] == 0.0F; }
        expect(drafts[u] == u * 100 && one_hot &&
                   std::abs(double(confidence[u]) - ideal) <= 1e-6 + 1e-5 * ideal,
               "best_candidate column " + std::to_string(u));
    }
    // Without confidences.
    Tensor none;
    dflash::kernels::best_candidate(t_ids, t_scores, t_drafts, none, t_q, nullptr);
    check(cudaDeviceSynchronize(), "best_candidate without confidence");
    // fill_bf16
    DeviceBuffer ones(256 * 7 * 2 * 2);
    ones.fill(0);
    Tensor t_ones(ones.p, DType::BF16, {256, 7, 2});
    dflash::kernels::fill_bf16(t_ones, 1.0F, nullptr);
    check(cudaDeviceSynchronize(), "fill_bf16");
    std::vector<std::uint16_t> words(256 * 7 * 2);
    ones.copy_to_host(words.data(), words.size() * 2);
    bool all = true;
    for (const auto w : words) { all = all && w == 0x3F80; }
    expect(all, "fill_bf16 writes BF16 one everywhere");
}

} // namespace

int main() {
    try {
        configs();
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::cerr << "no CUDA device: the kernel checks are skipped\n";
            return failures == 0 ? 77 : 1;
        }
        std::mt19937 rng(20261127);
        kernels(rng);
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) { return 1; }
    std::cout << "dflash drafter checks passed\n";
    return 0;
}
