// moe_route against an FP64 oracle: the 512- and 256-expert softmax top-10 with renormalised
// weights and the shared-expert sigmoid gate, from FP32 and BF16 block inputs, including tied
// logits. A token whose input is NaN (an overflowed hidden state) still selects ten distinct
// experts in range, and the other tokens keep their oracle selection. Every case runs twice more
// with a completion counter (the selection in the GEMV's last CTA for up to eight tokens): ids,
// weights and shared equal the first call's bit for bit, and the counter is zero after each call.
#include "core/device_launch.h"
#include "ninfer/ops/moe_route.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kHidden = 2560, kTop = 10;

// `kExperts` experts: 512, or the 256 an expert-pruned release keeps.
int run(int tokens, bool fp32_input, bool ties, std::uint32_t seed, int kExperts = 512,
        int nan_token = -1) {
    std::vector<float> m(static_cast<std::size_t>(kHidden) * tokens), router(static_cast<std::size_t>(kExperts) * kHidden),
        gate(kHidden);
    fill_uniform(m, seed, -3.0f, 3.0f);
    fill_uniform(router, seed + 1, -0.05f, 0.05f);
    fill_uniform(gate, seed + 2, -0.05f, 0.05f);
    if (!fp32_input) round_to_bf16(m);
    round_to_bf16(router);
    round_to_bf16(gate);
    if (ties) {
        // Experts 3, 7, 200 and the last share one router row aligned with token 0's input, so
        // their logits tie exactly among the ten largest.
        for (int d = 0; d < kHidden; ++d) {
            const float v = router[3 * kHidden + d] + (m[d] > 0 ? 0.05f : -0.05f);
            for (int e : {3, 7, 200, kExperts - 1})
                router[static_cast<std::size_t>(e) * kHidden + d] = v;
        }
        round_to_bf16(router);
    }
    if (nan_token >= 0) {
        for (int d = 0; d < kHidden; ++d) {
            m[static_cast<std::size_t>(nan_token) * kHidden + d] = std::nanf("");
        }
    }
    const auto encode = [](const std::vector<float>& values) {
        std::vector<std::uint16_t> bits(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
        return bits;
    };
    GuardedDeviceBuffer d_m(m.size() * (fp32_input ? 4 : 2)), d_router(router.size() * 2), d_gate(gate.size() * 2),
        d_ids(static_cast<std::size_t>(kTop) * tokens * 4), d_weights(static_cast<std::size_t>(kTop) * tokens * 4),
        d_shared(static_cast<std::size_t>(tokens) * 4);
    if (fp32_input) {
        d_m.copy_from_host(m.data(), d_m.bytes());
    } else {
        const auto bits = encode(m);
        d_m.copy_from_host(bits.data(), d_m.bytes());
    }
    const auto rb = encode(router), gb = encode(gate);
    d_router.copy_from_host(rb.data(), d_router.bytes());
    d_gate.copy_from_host(gb.data(), d_gate.bytes());
    Tensor t_m(d_m.data(), fp32_input ? DType::FP32 : DType::BF16, {kHidden, tokens});
    Tensor t_router(d_router.data(), DType::BF16, {kHidden, kExperts});
    Tensor t_gate(d_gate.data(), DType::BF16, {kHidden});
    Tensor t_ids(d_ids.data(), DType::I32, {kTop, tokens});
    Tensor t_weights(d_weights.data(), DType::FP32, {kTop, tokens});
    Tensor t_shared(d_shared.data(), DType::FP32, {tokens});
    WorkspaceArena workspace(ops::moe_route_workspace_bytes(tokens, kExperts));
    ops::moe_route(t_m, t_router, t_gate, workspace, t_ids, t_weights, t_shared, nullptr);
    cuda_synchronize();
    const auto ids     = from_device<int>(d_ids.data(), static_cast<std::size_t>(kTop) * tokens);
    const auto weights = from_device<float>(d_weights.data(), static_cast<std::size_t>(kTop) * tokens);
    const auto shared  = from_device<float>(d_shared.data(), tokens);
    const std::string label = "moe_route T=" + std::to_string(tokens) +
                              (fp32_input ? " fp32" : " bf16") + (ties ? " ties" : "") +
                              " E=" + std::to_string(kExperts) +
                              (nan_token >= 0 ? " NaN token " + std::to_string(nan_token) : "");
    int failures = 0;
    std::vector<double> expected_weights, got_weights, expected_shared, got_shared;
    for (int t = 0; t < tokens; ++t) {
        if (t == nan_token) {
            std::vector<int> chosen(ids.begin() + static_cast<std::ptrdiff_t>(t) * kTop,
                                    ids.begin() + static_cast<std::ptrdiff_t>(t + 1) * kTop);
            std::sort(chosen.begin(), chosen.end());
            if (chosen.front() < 0 || chosen.back() >= kExperts ||
                std::adjacent_find(chosen.begin(), chosen.end()) != chosen.end()) {
                std::cerr << label << ": the NaN token selected experts out of range or twice\n";
                ++failures;
            }
            continue;
        }
        std::vector<double> logits(kExperts);
        for (int e = 0; e < kExperts; ++e) {
            double dot = 0;
            for (int d = 0; d < kHidden; ++d) dot += double(router[static_cast<std::size_t>(e) * kHidden + d]) * m[static_cast<std::size_t>(t) * kHidden + d];
            logits[e] = dot;
        }
        double g = 0;
        for (int d = 0; d < kHidden; ++d) g += double(gate[d]) * m[static_cast<std::size_t>(t) * kHidden + d];
        expected_shared.push_back(1.0 / (1.0 + std::exp(-g)));
        got_shared.push_back(shared[t]);
        std::vector<int> order(kExperts);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
        const double tenth = logits[order[kTop - 1]];
        for (int k = 0; k < kTop; ++k) {
            const int got = ids[static_cast<std::size_t>(t) * kTop + k];
            if (got != order[k] && std::abs(logits[got] - logits[order[k]]) > 1e-5 * std::max(1.0, std::abs(tenth))) {
                std::cerr << label << ": token " << t << " rank " << k << " expert " << got << " expected " << order[k] << '\n';
                ++failures;
                break;
            }
        }
        double sum = 0;
        for (int k = 0; k < kTop; ++k) sum += std::exp(logits[order[k]] - logits[order[0]]);
        for (int k = 0; k < kTop; ++k) {
            expected_weights.push_back(std::exp(logits[order[k]] - logits[order[0]]) / sum);
            got_weights.push_back(weights[static_cast<std::size_t>(t) * kTop + k]);
        }
    }
    failures += verify_pointwise(label + " weights", got_weights, expected_weights, {1e-6, 2e-5});
    failures += verify_pointwise(label + " shared gate", got_shared, expected_shared, {1e-6, 2e-5});
    // The counter form: twice over one counter, which each call must leave at zero.
    GuardedDeviceBuffer d_done(4), c_ids(d_ids.bytes()), c_weights(d_weights.bytes()),
        c_shared(d_shared.bytes());
    d_done.fill(0);
    Tensor u_ids(c_ids.data(), DType::I32, {kTop, tokens});
    Tensor u_weights(c_weights.data(), DType::FP32, {kTop, tokens});
    Tensor u_shared(c_shared.data(), DType::FP32, {tokens});
    for (int call = 0; call < 2; ++call) {
        c_ids.fill(0xff);
        c_weights.fill(0xff);
        c_shared.fill(0xff);
        ops::moe_route(t_m, t_router, t_gate, workspace, u_ids, u_weights, u_shared, nullptr,
                       static_cast<std::uint32_t*>(d_done.data()));
        cuda_synchronize();
        const auto counter = from_device<std::uint32_t>(d_done.data(), 1);
        const auto same    = [&](const GuardedDeviceBuffer& a, const GuardedDeviceBuffer& b) {
            const auto x = from_device<std::uint8_t>(a.data(), a.bytes());
            const auto y = from_device<std::uint8_t>(b.data(), b.bytes());
            return x == y;
        };
        if (counter[0] != 0 || !same(c_ids, d_ids) || !same(c_weights, d_weights) ||
            !same(c_shared, d_shared)) {
            std::cerr << label << ": the counter form, call " << call
                      << ", differs from the two-launch form or leaves the counter at "
                      << counter[0] << '\n';
            ++failures;
        }
    }
    for (auto* buffer : {&d_m, &d_router, &d_gate, &d_ids, &d_weights, &d_shared, &d_done, &c_ids,
                         &c_weights, &c_shared}) {
        failures += buffer->verify_guards(label.c_str());
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run(1, true, false, 9200u);
    failures += run(7, false, false, 9201u);
    failures += run(8, true, false, 9206u);
    failures += run(33, true, false, 9202u);
    failures += run(64, false, false, 9207u);
    failures += run(1024, false, true, 9208u);
    failures += run(3, true, true, 9203u);
    failures += run(5, true, false, 9204u, 256);
    failures += run(2, false, true, 9205u, 256);
    failures += run(3, true, false, 9209u, 512, 1);
    failures += run(2, true, false, 9210u, 256, 0);
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " moe_route\n";
    return failures == 0 ? 0 : 1;
}
