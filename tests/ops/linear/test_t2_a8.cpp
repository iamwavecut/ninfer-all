#include "ops/common/device_route.h"
#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>
#include <string>
#include <utility>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

constexpr Invocation a8(std::int32_t t) {
    return {t, CallForm::Policy, ops::LinearPolicy::AllowA8Int};
}

// The ternary integer-activation routes through one policy: the small-T kernel (1..192 tokens, in
// launches of one, two and four column tiles) and above it the prefill GEMM, which pads T to its
// cheapest tile width inside.
int t2_a8_conformance() {
    int failures = 0;

    constexpr std::array kWidths{a8(1),   a8(2),   a8(3),    a8(4),   a8(5),   a8(7),   a8(8),
                                 a8(9),   a8(12),  a8(15),   a8(16),  a8(17),  a8(24),  a8(32),
                                 a8(33),  a8(40),  a8(63),   a8(64),  a8(65),  a8(95),  a8(100),
                                 a8(127), a8(128), a8(129),  a8(192), a8(193), a8(256), a8(300),
                                 a8(385), a8(600), a8(1007), a8(1024)};

    failures += run_shape("T2_A8", ActivationCompute::A8, make_t2_g128_fp16_weight,
                          {1024, 5120, 301U, Comparison::Full, true, kWidths});
    failures += run_shape("T2_A8", ActivationCompute::A8, make_t2_g128_fp16_weight,
                          {4096, 5120, 303U, Comparison::SampledRows, false, kWidths});
    failures += run_shape("T2_A8", ActivationCompute::A8, make_t2_g128_fp16_weight,
                          {7168, 5120, 305U, Comparison::SampledRows, false, kWidths});
    failures += run_shape("T2_A8", ActivationCompute::A8, make_t2_g128_fp16_weight,
                          {34816, 5120, 307U, Comparison::SampledRows, false, kWidths});
    failures += run_shape("T2_A8", ActivationCompute::A8, make_t2_g128_fp16_weight,
                          {5120, 6144, 309U, Comparison::Full, false, kWidths});
    failures += run_shape("T2_A8", ActivationCompute::A8, make_t2_g128_fp16_weight,
                          {5120, 17408, 311U, Comparison::Full, false, kWidths});

    return failures;
}

// The 64-column small-T schedules a device profile may name for 33..64 columns, forced on the
// shapes Ternary Bonsai 2 runs, against the same oracle.
int t2_a8_wide_schedules() {
    int failures = 0;
    constexpr std::array kWidths{a8(24), a8(33), a8(40), a8(48), a8(57), a8(64), a8(65), a8(100)};
    for (const char* schedule : {"r32c64", "r32c64w1", "r16c64w1"}) {
        const std::string name = std::string("T2_A8 ") + schedule;
        {
            const ops::DeviceRouteForce force("t2_i8_small/34816x5120", schedule);
            failures += run_shape(name.c_str(), ActivationCompute::A8, make_t2_g128_fp16_weight,
                                  {34816, 5120, 313U, Comparison::SampledRows, false, kWidths});
        }
        {
            const ops::DeviceRouteForce force("t2_i8_small/5120x17408", schedule);
            failures += run_shape(name.c_str(), ActivationCompute::A8, make_t2_g128_fp16_weight,
                                  {5120, 17408, 315U, Comparison::Full, false, kWidths});
        }
    }
    return failures;
}

// A tensor-parallel rank's shares: rows of gate and up at the hidden width (not a whole 128-row
// prefill block, so past the small-T band the A16 route answers), and columns of down and of the
// mixer outputs, into the FP32 partial the ranks add and into BF16. Down's share is an odd number
// of 512-k slabs, which takes the one-word schedules at 8 and 16 columns.
int t2_a8_shares() {
    int failures = 0;
    constexpr std::array kWidths{a8(1),  a8(2),  a8(4),  a8(7),  a8(8),  a8(9),  a8(16), a8(17),
                                 a8(24), a8(32), a8(33), a8(48), a8(64), a8(65), a8(100)};
    for (const DType out : {DType::FP32, DType::BF16}) {
        ShapeCase rows{1056, 5120, 317U, Comparison::Full, false, kWidths};
        rows.out = out;
        failures += run_shape("T2_A8 share", ActivationCompute::A8, make_t2_g128_fp16_weight, rows);
        ShapeCase down{5120, 8704, 319U, Comparison::SampledRows, false, kWidths};
        down.out = out;
        failures += run_shape("T2_A8 share", ActivationCompute::A8, make_t2_g128_fp16_weight, down);
        ShapeCase mixer{5120, 3072, 321U, Comparison::SampledRows, false, kWidths};
        mixer.out = out;
        failures +=
            run_shape("T2_A8 share", ActivationCompute::A8, make_t2_g128_fp16_weight, mixer);
    }
    // The registered input width into FP32: the small-T band and the prefill tile.
    ShapeCase whole{1024, 5120, 323U, Comparison::Full, false, kWidths};
    whole.out = DType::FP32;
    failures += run_shape("T2_A8", ActivationCompute::A8, make_t2_g128_fp16_weight, whole);
    // A declared interval covers every width in it, the shares' small-T planes included.
    for (const auto [n, k] :
         {std::pair{1056, 5120}, std::pair{5120, 8704}, std::pair{5120, 3072}}) {
        for (const auto [first, last] : {std::pair{1, 8}, std::pair{1, 64}, std::pair{9, 129}}) {
            const std::size_t capacity = ops::linear_workspace_capacity_bytes(
                QType::T2_G128_FP16, n, k, ops::LinearPolicy::AllowA8Int, first, last);
            for (std::int32_t t = first; t <= last; ++t) {
                if (ops::linear_workspace_capacity_bytes(QType::T2_G128_FP16, n, k,
                                                         ops::LinearPolicy::AllowA8Int, t,
                                                         t) > capacity) {
                    std::cerr << "T2_A8 share workspace [" << first << ',' << last
                              << "] misses T=" << t << " at [" << n << ',' << k << "]\n";
                    ++failures;
                    break;
                }
            }
        }
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
        const int failures = t2_a8_conformance() + t2_a8_wide_schedules() + t2_a8_shares();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " T2_A8 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "T2_A8 Linear: " << error.what() << '\n';
        return 1;
    }
}
