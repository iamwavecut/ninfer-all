// moe_guess_experts against an exact host ranking: each row's per_row best experts in decreasing
// order, by logit or by sigmoid(logit) + bias, ties to the lower index, for 128 to 512 experts,
// one to eight rows and one to sixteen experts a row, including tied scores; the ids past the
// requested ones are untouched. moe_expert_prefetch over tables with null entries, repeated and
// out-of-range ids and spans that are not whole 16-byte words leaves every byte of the experts and
// the guarded buffers around them unchanged. moe_prefetch_budget is 0 for experts larger than the
// device's L2 cache and never above 16.
#include "core/device_launch.h"
#include "ninfer/ops/moe_expert_prefetch.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

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

int guess(int rows, int experts, int per_row, bool bias, bool ties, std::uint32_t seed) {
    std::vector<float> logits(std::size_t(rows) * experts), b(experts);
    fill_uniform(logits, seed, -4.0f, 4.0f);
    fill_uniform(b, seed + 1, -0.5f, 0.5f);
    if (ties) {
        // Every row's experts 5, 9, 63 and the last share the row's largest score.
        for (int r = 0; r < rows; ++r) {
            float* row      = logits.data() + std::size_t(r) * experts;
            const float top = *std::max_element(row, row + experts) + 1.0f;
            for (int e : {5, 9, 63, experts - 1}) { row[e] = top; }
        }
        if (bias) {
            for (int e : {5, 9, 63, experts - 1}) { b[e] = 0.25f; }
        }
    }
    constexpr int kStride = 16;
    GuardedDeviceBuffer d_logits(logits.size() * 4), d_bias(b.size() * 4),
        d_ids(std::size_t(rows) * kStride * 4);
    d_logits.copy_from_host(logits.data(), d_logits.bytes());
    d_bias.copy_from_host(b.data(), d_bias.bytes());
    d_ids.fill(0xff);
    // Rows lie per_row ids apart; the rest of the buffer stays -1.
    ops::moe_guess_experts(static_cast<const float*>(d_logits.data()),
                           bias ? static_cast<const float*>(d_bias.data()) : nullptr, rows, experts,
                           per_row, static_cast<std::int32_t*>(d_ids.data()), nullptr);
    cuda_synchronize();
    const auto ids = from_device<std::int32_t>(d_ids.data(), std::size_t(rows) * kStride);
    const std::string label =
        "moe_guess_experts rows " + std::to_string(rows) + " E=" + std::to_string(experts) +
        " per_row " + std::to_string(per_row) + (bias ? " bias" : "") + (ties ? " ties" : "");
    int failures = 0;
    for (int r = 0; r < rows; ++r) {
        std::vector<float> score(experts);
        for (int e = 0; e < experts; ++e) {
            const float l = logits[std::size_t(r) * experts + e];
            // The kernel's own FP32 expression, so the ranking compares the same values.
            score[e] = bias ? 1.0f / (1.0f + std::exp(-l)) + b[e] : l;
        }
        std::vector<int> order(experts);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](int x, int y) { return score[x] > score[y]; });
        for (int j = 0; j < per_row; ++j) {
            const int got  = ids[std::size_t(r) * per_row + j];
            const int want = order[j];
            // sigmoid on the device may differ from the host's in the last place; then only an
            // expert of an equal host score may stand in.
            if (got != want && !(bias && got >= 0 && got < experts &&
                                 std::abs(score[got] - score[want]) <= 4e-7f)) {
                std::cerr << label << ": row " << r << " rank " << j << " expert " << got
                          << " expected " << want << '\n';
                ++failures;
                break;
            }
        }
    }
    for (std::size_t i = std::size_t(rows) * per_row; i < ids.size(); ++i) {
        if (ids[i] != -1) {
            std::cerr << label << ": wrote past the requested ids at " << i << '\n';
            ++failures;
            break;
        }
    }
    for (auto* buffer : {&d_logits, &d_bias, &d_ids}) {
        failures += buffer->verify_guards(label.c_str());
    }
    return failures;
}

int prefetch() {
    constexpr int kExperts = 64;
    // Two projections; projection 1's span is not a whole number of 16-byte words and starts
    // off a word, so the reads must stop short of both ends.
    const std::int64_t bytes[2] = {4096, 1000};
    GuardedDeviceBuffer bank0(std::size_t(kExperts) * bytes[0]),
        bank1(std::size_t(kExperts) * bytes[1] + 8);
    std::vector<std::uint8_t> host0(bank0.bytes()), host1(bank1.bytes());
    for (std::size_t i = 0; i < host0.size(); ++i) { host0[i] = std::uint8_t(i * 7 + 1); }
    for (std::size_t i = 0; i < host1.size(); ++i) { host1[i] = std::uint8_t(i * 13 + 3); }
    bank0.copy_from_host(host0.data(), host0.size());
    bank1.copy_from_host(host1.data(), host1.size());
    std::vector<const void*> t0(kExperts), t1(kExperts);
    for (int e = 0; e < kExperts; ++e) {
        t0[e] = static_cast<const std::uint8_t*>(bank0.data()) + std::size_t(e) * bytes[0];
        // Every third expert of projection 1 has no table entry.
        t1[e] = e % 3 == 0 ? nullptr
                           : static_cast<const std::uint8_t*>(bank1.data()) + 3 +
                                 std::size_t(e) * bytes[1];
    }
    GuardedDeviceBuffer d_t0(t0.size() * sizeof(void*)), d_t1(t1.size() * sizeof(void*));
    d_t0.copy_from_host(t0.data(), d_t0.bytes());
    d_t1.copy_from_host(t1.data(), d_t1.bytes());
    const std::vector<std::int32_t> ids = {3, 7, 3, -1, 63, kExperts, 12, 0, 7, 40};
    GuardedDeviceBuffer d_ids(ids.size() * 4);
    d_ids.copy_from_host(ids.data(), d_ids.bytes());
    ops::ExpertPrefetch args;
    args.tables[0]   = static_cast<const void* const*>(d_t0.data());
    args.tables[1]   = static_cast<const void* const*>(d_t1.data());
    args.bytes[0]    = bytes[0];
    args.bytes[1]    = bytes[1];
    args.projections = 2;
    args.experts     = kExperts;
    args.ids         = static_cast<const std::int32_t*>(d_ids.data());
    args.count       = int(ids.size());
    ops::moe_expert_prefetch(args, nullptr);
    cuda_synchronize();
    int failures = 0;
    if (from_device<std::uint8_t>(bank0.data(), host0.size()) != host0 ||
        from_device<std::uint8_t>(bank1.data(), host1.size()) != host1) {
        std::cerr << "moe_expert_prefetch: an expert's bytes changed\n";
        ++failures;
    }
    for (auto* buffer : {&bank0, &bank1, &d_t0, &d_t1, &d_ids}) {
        failures += buffer->verify_guards("moe_expert_prefetch");
    }
    return failures;
}

int budget() {
    int device = 0, l2 = 0;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cuda_check(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device), "L2 size");
    int failures = 0;
    if (ops::moe_prefetch_budget(device, std::size_t(l2) + 1) != 0 ||
        ops::moe_prefetch_budget(device, 0) != 0) {
        std::cerr
            << "moe_prefetch_budget: an expert larger than L2 (or of no bytes) gets a budget\n";
        ++failures;
    }
    const int small = ops::moe_prefetch_budget(device, 4096);
    if (small < 0 || small > 16 || (l2 >= 4 * 2 * 4096 && small < 2)) {
        std::cerr << "moe_prefetch_budget: " << small << " for 4 KiB experts\n";
        ++failures;
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
    failures += guess(1, 128, 6, false, false, 7100u);
    failures += guess(8, 128, 1, false, true, 7101u);
    failures += guess(4, 384, 16, true, false, 7102u);
    failures += guess(3, 512, 10, false, true, 7103u);
    failures += guess(2, 256, 8, true, true, 7104u);
    failures += guess(5, 129, 3, false, false, 7105u);
    failures += prefetch();
    failures += budget();
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " moe_expert_prefetch\n";
    return failures == 0 ? 0 : 1;
}
