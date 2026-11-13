#include "core/qtype.h"
#include "core/weight.h"
#include "ops/linear/linear_test_common.h"

#include <vector>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

int run_nvfp4_a4() {
    std::vector<Invocation> invocations;
    for (int t :
         {1,   2,   3,   4,   5,   7,   8,   16,  17,  63,  64,   65,   95,   96,   97,   127, 128,
          129, 191, 192, 193, 383, 384, 385, 511, 512, 513, 1023, 1024, 1025, 1280, 1500, 2048}) {
        invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4});
    }
    for (int t : {63, 97, 193, 385, 1023, 1024})
        invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4, true});
    int failures = 0;
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {14336, 5120, 719U, Comparison::SampledRows, true, invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {16384, 5120, 721U, Comparison::SampledRows, true, invocations});
    // These calls regenerate A4 codes/scales on graph replay and cover complete and partial tiles.
    const std::vector<Invocation> n34816_full{
        {1, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {4, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {8, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {32, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {33, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {64, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {65, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
        {128, CallForm::Policy, ops::LinearPolicy::AllowA4, true},
    };
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {34816, 5120, 722U, Comparison::Full, true, n34816_full});
    auto n34816_invocations = invocations;
    for (int t : {31, 32, 33, 34, 255, 256, 257, 511, 512, 513, 767, 768, 769})
        n34816_invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4, true});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {34816, 5120, 722U, Comparison::SampledRows, false, n34816_invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {5120, 6144, 723U, Comparison::SampledRows, true, invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {5120, 17408, 725U, Comparison::SampledRows, true, invocations});
    return failures;
}

// A tensor-parallel rank's share of Qwen3.8-27B's MLP on the runtime-shape route, A16 up to 8
// tokens and A4 from 9: gate and up rows into FP32 products and the down projection over the
// share's intermediate into an FP32 partial sum, at two and four ranks and a 3:1 split, whole
// 128-row and 256-column units. A share into BF16 and every row of a small matrix cover the BF16
// output. On an RTX 5090 the widths reach every tile the A4 route picks by waves
// (common/wave_tiles.h) into either output.
int run_nvfp4_a4_shares() {
    std::vector<Invocation> invocations;
    for (int t : {1,  2,   4,   7,   8,   9,   16,  31,  32,   33,   64,
                  65, 128, 129, 255, 256, 257, 384, 512, 1024, 1025, 2048})
        invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4});
    for (int t : {1, 4, 8, 9, 10, 33, 129, 257, 512})
        invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA4, true});

    struct Share {
        std::int32_t n;
        std::int32_t k;
        std::uint32_t seed;
        DType out;
    };

    int failures = 0;
    for (const Share share :
         {Share{8704, 5120, 731U, DType::FP32}, Share{5120, 8704, 733U, DType::FP32},
          Share{13056, 5120, 735U, DType::FP32}, Share{5120, 4352, 737U, DType::FP32},
          Share{4352, 5120, 739U, DType::BF16}}) {
        failures += run_shape("NVFP4_A4 share", ActivationCompute::A4, make_nvfp4_weight,
                              {share.n, share.k, share.seed, Comparison::SampledRows, true,
                               invocations, ActivationSigns::Centered, share.out});
        failures += verify_workspace_envelopes(QType::NVFP4, share.n, share.k);
    }
    for (const DType out : {DType::BF16, DType::FP32}) {
        failures += run_shape(
            "NVFP4_A4 share", ActivationCompute::A4, make_nvfp4_weight,
            {384, 768, 741U, Comparison::Full, true, invocations, ActivationSigns::Centered, out});
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
        const int failures = run_nvfp4_a4() + run_nvfp4_a4_shares();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4_A4 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4_A4 Linear: " << error.what() << '\n';
        return 1;
    }
}
