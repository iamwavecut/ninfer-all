#include "core/qtype.h"
#include "ops/linear/linear_test_common.h"

#include <exception>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

int run_nvfp4_a16() {
    std::vector<Invocation> invocations;
    for (int t = 1; t <= 33; ++t) invocations.push_back({t});
    for (int t : {63, 64, 65, 95, 96, 97, 127, 128, 129, 512, 1024}) invocations.push_back({t});
    for (int t : {3, 7, 11, 15, 19, 23, 27, 31, 33})
        invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    invocations.push_back({1, CallForm::A16Convenience});
    invocations.push_back({33, CallForm::Policy, ops::LinearPolicy::AllowA8});
    int failures = 0;
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {14336, 5120, 701U, Comparison::SampledRows, true, invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {16384, 5120, 703U, Comparison::SampledRows, true, invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {34816, 5120, 704U, Comparison::SampledRows, true, invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 6144, 705U, Comparison::SampledRows, true, invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 17408, 707U, Comparison::SampledRows, true, invocations});
    for (auto [n, k] : {std::pair{14336, 5120}, std::pair{16384, 5120}, std::pair{34816, 5120},
                        std::pair{5120, 6144}, std::pair{5120, 17408}}) {
        failures += verify_workspace_envelopes(QType::NVFP4, n, k);
    }

    // Every valid scale word, 0x00 through 0x7E, under a divisor that is not a power of two. A
    // production artifact's scales reach 0x7E; the default fixture's eight small words stay at the
    // bottom of the exponent range. The widths add the boundaries a speculative round or a prompt
    // chunk puts in front of each route table's kernels. The route tables run different kernels
    // with different K splits, so each answers to the oracle on its own: this test runs the device
    // profile's table, its legacy_routes and unified_routes variants run each table.
    const WeightGenerator valid_scale_words = [](std::int32_t n, std::int32_t k,
                                                 std::uint32_t seed) {
        return make_nvfp4_weight(n, k, seed, 0.1F, /*wide_scale_range=*/true);
    };
    std::vector<Invocation> wide_invocations;
    for (int t : {1,  2,  3,  4,   5,   6,   7,   8,   9,   12,  15,  16,  17,  24,  32,  48,  63,
                  64, 65, 96, 104, 105, 106, 107, 108, 112, 120, 127, 128, 129, 130, 512, 1024}) {
        wide_invocations.push_back({t});
    }
    for (const auto [n, k, seed] : {std::tuple{14336, 5120, 7301U}, std::tuple{16384, 5120, 7303U},
                                    std::tuple{34816, 5120, 7307U}, std::tuple{5120, 6144, 7311U},
                                    std::tuple{5120, 17408, 7309U}}) {
        failures +=
            run_shape("NVFP4_A16 valid scale words", ActivationCompute::A16, valid_scale_words,
                      {n, k, seed, Comparison::SampledRows, false, wide_invocations});
    }
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = run_nvfp4_a16();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
