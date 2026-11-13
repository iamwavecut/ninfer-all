#include "core/device_buffer.h"
#include "ninfer/ops/steer_residual.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

using Step = ops::SteerResidualStep;

// One BF16 rounding of the updated residual.
constexpr PointwiseCriterion steer_residual_x_criterion() {
    return {/*absolute*/ 2e-3, /*relative*/ 3.95e-3};
}

struct Case {
    const char* label   = "";
    Step step           = Step::Write;
    std::int32_t hidden = 0;
    std::int32_t columns = 0;
    std::int32_t count   = 0; // directions
    std::int32_t lanes   = 4;
    std::int32_t row_width = 1;
    std::vector<std::int32_t> rows;
    // lane_scales [count, lanes]; a lane listed here has every scale zero.
    std::vector<std::int32_t> idle_lanes;
    bool vector       = false;
    // x(t) = bf16(along * r_0 + noise) instead of uniform values.
    float along       = 0.0F;
    std::uint32_t seed = 1;
};

std::vector<float> unit_directions(std::int32_t hidden, std::int32_t count, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> out(std::size_t(hidden) * count);
    for (std::int32_t e = 0; e < count; ++e) {
        double norm2 = 0.0;
        for (std::int32_t i = 0; i < hidden; ++i) {
            const float value = normal(generator);
            out[std::size_t(e) * hidden + i] = value;
            norm2 += double(value) * value;
        }
        const double inverse = 1.0 / std::sqrt(norm2);
        for (std::int32_t i = 0; i < hidden; ++i) {
            out[std::size_t(e) * hidden + i] =
                static_cast<float>(out[std::size_t(e) * hidden + i] * inverse);
        }
    }
    return out;
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

int run_case(const Case& c) {
    const auto hidden  = static_cast<std::size_t>(c.hidden);
    const auto columns = static_cast<std::size_t>(c.columns);
    const auto count   = static_cast<std::size_t>(c.count);
    std::vector<float> x(hidden * columns);
    fill_uniform(x, c.seed, -8.0F, 8.0F);
    const std::vector<float> directions = unit_directions(c.hidden, c.count, c.seed + 1);
    if (c.along != 0.0F) {
        for (std::size_t t = 0; t < columns; ++t) {
            for (std::size_t i = 0; i < hidden; ++i) {
                x[t * hidden + i] = c.along * directions[i] + 0.01F * x[t * hidden + i];
            }
        }
    }
    round_to_bf16(x);
    std::vector<float> scales(count * c.lanes);
    fill_uniform(scales, c.seed + 2, -2.0F, 2.0F);
    for (const std::int32_t lane : c.idle_lanes) {
        for (std::size_t e = 0; e < count; ++e) { scales[std::size_t(lane) * count + e] = 0.0F; }
    }
    std::vector<float> factors(count);
    fill_uniform(factors, c.seed + 3, 0.5F, 1.5F);
    std::vector<float> measures(count * columns);
    fill_uniform(measures, c.seed + 4, -20.0F, 20.0F);
    std::vector<float> vector(c.vector ? hidden : 0);
    fill_uniform(vector, c.seed + 5, -0.5F, 0.5F);

    const auto lane_of = [&](std::size_t t) { return c.rows[t / std::size_t(c.row_width)]; };
    const auto scale_of = [&](std::int32_t lane, std::size_t e) {
        return lane >= 0 && lane < c.lanes ? double(scales[std::size_t(lane) * count + e]) : 0.0;
    };
    const auto active = [&](std::size_t t) {
        for (std::size_t e = 0; e < count; ++e) {
            if (scale_of(lane_of(t), e) != 0.0) { return true; }
        }
        return false;
    };
    const auto dot = [&](std::size_t e, const auto& values, std::size_t t) {
        double sum = 0.0;
        for (std::size_t i = 0; i < hidden; ++i) {
            sum += double(directions[e * hidden + i]) * double(values[t * hidden + i]);
        }
        return sum;
    };

    std::vector<double> expected_x(x.begin(), x.end());
    std::vector<double> expected_measures(measures.begin(), measures.end());
    for (std::size_t t = 0; t < columns; ++t) {
        if (active(t)) {
            if (c.step == Step::Measure) {
                for (std::size_t e = 0; e < count; ++e) {
                    expected_measures[t * count + e] = dot(e, x, t);
                }
                continue;
            }
            for (std::size_t e = 0; e < count; ++e) {
                const double written =
                    dot(e, x, t) - (c.step == Step::Write ? double(measures[t * count + e]) : 0.0);
                const double k = scale_of(lane_of(t), e) * double(factors[e]) * written;
                for (std::size_t i = 0; i < hidden; ++i) {
                    expected_x[t * hidden + i] -= k * double(directions[e * hidden + i]);
                }
            }
        }
        if (c.vector) {
            for (std::size_t i = 0; i < hidden; ++i) { expected_x[t * hidden + i] += vector[i]; }
        }
    }

    const auto x_bits = encode_bf16(x);
    GuardedDeviceBuffer device_x(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_measures(std::max<std::size_t>(measures.size(), 1) * sizeof(float));
    device_x.copy_from_host(x_bits.data(), device_x.bytes());
    device_measures.copy_from_host(measures.data(), measures.size() * sizeof(float));
    DeviceBuffer device_directions = to_device_f32(directions.empty() ? std::vector<float>{0.0F}
                                                                      : directions);
    DeviceBuffer device_scales  = to_device_f32(scales.empty() ? std::vector<float>{0.0F} : scales);
    DeviceBuffer device_factors = to_device_f32(factors.empty() ? std::vector<float>{0.0F} : factors);
    DeviceBuffer device_rows    = to_device_i32(c.rows);
    DeviceBuffer device_vector  = to_device_f32(c.vector ? vector : std::vector<float>{0.0F});

    Tensor x_tensor(device_x.data(), DType::BF16, {c.hidden, c.columns});
    Tensor measures_tensor = c.count > 0 ? Tensor(device_measures.data(), DType::FP32,
                                                  {c.count, c.columns})
                                         : Tensor();
    const Tensor directions_tensor =
        c.count > 0 ? Tensor(device_directions.p, DType::FP32, {c.hidden, c.count}) : Tensor();
    const Tensor factors_tensor =
        c.count > 0 ? Tensor(device_factors.p, DType::FP32, {c.count}) : Tensor();
    const ops::SteerResidualLanes lanes{
        .lane_scales = c.count > 0 ? Tensor(device_scales.p, DType::FP32, {c.count, c.lanes})
                                   : Tensor(),
        .rows        = Tensor(device_rows.p, DType::I32,
                              {static_cast<std::int32_t>(c.rows.size())}),
        .row_width   = c.row_width,
    };
    const Tensor vector_tensor =
        c.vector ? Tensor(device_vector.p, DType::FP32, {c.hidden}) : Tensor();
    ops::steer_residual(c.step, x_tensor, directions_tensor, factors_tensor, lanes,
                        measures_tensor, vector_tensor, nullptr);
    cuda_synchronize();

    const std::vector<double> got_x = from_device_bf16(device_x.data(), x.size());
    int failures = verify_pointwise(std::string(c.label) + " x", got_x, expected_x,
                                    steer_residual_x_criterion());
    // A Write's new measures are the dots of the stored residual; every other measure is either
    // the Measure step's dot of the input or untouched.
    std::vector<float> got_measures(measures.size());
    device_measures.copy_to_host(got_measures.data(), got_measures.size() * sizeof(float));
    double worst = 0.0;
    for (std::size_t t = 0; t < columns; ++t) {
        for (std::size_t e = 0; e < count; ++e) {
            double reference = expected_measures[t * count + e];
            double magnitude = 0.0;
            if (c.step == Step::Write && active(t)) { reference = dot(e, got_x, t); }
            for (std::size_t i = 0; i < hidden; ++i) {
                magnitude += std::abs(double(directions[e * hidden + i]) *
                                      (c.step == Step::Write ? got_x[t * hidden + i]
                                                             : double(x[t * hidden + i])));
            }
            const bool touched = active(t) && c.step != Step::Embedding;
            const double error = std::abs(double(got_measures[t * count + e]) - reference);
            const double limit = touched ? 1e-5 * magnitude + 1e-6 : 0.0;
            worst              = std::max(worst, limit == 0.0 ? error : error / limit);
            if (error > limit) {
                std::cerr << c.label << ": measure (" << e << ", " << t << ") is "
                          << got_measures[t * count + e] << ", expected " << reference << '\n';
                ++failures;
            }
        }
    }
    failures += device_x.verify_guards(std::string(c.label) + " x");
    failures += device_measures.verify_guards(std::string(c.label) + " measures");
    return failures;
}

// Measure, then a write, then Write at scale one: the write's part along each direction is gone,
// so the measures come back to what the first Measure read, up to the residual's BF16 storage.
int run_round_trip() {
    constexpr std::int32_t hidden = 5120;
    constexpr std::int32_t count  = 2;
    const std::vector<float> directions = unit_directions(hidden, count, 77);
    std::vector<float> x(hidden), write(hidden);
    fill_uniform(x, 78, -4.0F, 4.0F);
    fill_uniform(write, 79, -1.0F, 1.0F);
    for (std::int32_t i = 0; i < hidden; ++i) { write[i] += 3.0F * directions[i]; }
    round_to_bf16(x);
    DeviceBuffer device_x          = to_device_bf16(x);
    DeviceBuffer device_directions = to_device_f32(directions);
    DeviceBuffer device_scales     = to_device_f32({1.0F, 1.0F});
    DeviceBuffer device_factors    = to_device_f32({1.0F, 1.0F});
    DeviceBuffer device_rows       = to_device_i32({0});
    DeviceBuffer device_measures   = to_device_f32({0.0F, 0.0F});
    Tensor x_tensor(device_x.p, DType::BF16, {hidden, 1});
    const Tensor directions_tensor(device_directions.p, DType::FP32, {hidden, count});
    const Tensor factors_tensor(device_factors.p, DType::FP32, {count});
    Tensor measures_tensor(device_measures.p, DType::FP32, {count, 1});
    const ops::SteerResidualLanes lanes{
        .lane_scales = Tensor(device_scales.p, DType::FP32, {count, 1}),
        .rows        = Tensor(device_rows.p, DType::I32, {1}),
        .row_width   = 1,
    };
    ops::steer_residual(Step::Measure, x_tensor, directions_tensor, factors_tensor, lanes,
                        measures_tensor, Tensor(), nullptr);
    cuda_synchronize();
    const std::vector<double> before = from_device_f32(device_measures, count);
    std::vector<float> written(hidden);
    for (std::int32_t i = 0; i < hidden; ++i) { written[i] = x[i] + write[i]; }
    round_to_bf16(written);
    const std::vector<std::uint16_t> bits = encode_bf16(written);
    cuda_check(cudaMemcpy(device_x.p, bits.data(), bits.size() * 2, cudaMemcpyHostToDevice),
               "round trip write");
    ops::steer_residual(Step::Write, x_tensor, directions_tensor, factors_tensor, lanes,
                        measures_tensor, Tensor(), nullptr);
    cuda_synchronize();
    const std::vector<double> after = from_device_f32(device_measures, count);
    int failures = 0;
    for (std::int32_t e = 0; e < count; ++e) {
        // Two BF16 roundings of values up to about 5, projected on a unit direction.
        if (std::abs(after[e] - before[e]) > 0.05) {
            std::cerr << "steer_residual round trip: direction " << e << " moved from " << before[e]
                      << " to " << after[e] << '\n';
            ++failures;
        }
    }
    return failures;
}

int run_refusals() {
    int failures      = 0;
    const auto refuse = [&](const char* label, auto&& call) {
        try {
            call();
            std::cerr << "steer_residual: " << label << " was accepted\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    DeviceBuffer x_storage(64 * 2 * sizeof(std::uint16_t));
    DeviceBuffer f32(64 * 16 * sizeof(float));
    DeviceBuffer i32(16 * sizeof(std::int32_t));
    Tensor x(x_storage.p, DType::BF16, {64, 2});
    Tensor measures(f32.p, DType::FP32, {2, 2});
    const ops::SteerResidualLanes lanes{.lane_scales = Tensor(f32.p, DType::FP32, {2, 1}),
                                        .rows = Tensor(i32.p, DType::I32, {2}), .row_width = 1};
    refuse("nine directions", [&] {
        Tensor m(f32.p, DType::FP32, {9, 2});
        const ops::SteerResidualLanes l{.lane_scales = Tensor(f32.p, DType::FP32, {9, 1}),
                                        .rows = lanes.rows, .row_width = 1};
        ops::steer_residual(Step::Write, x, Tensor(f32.p, DType::FP32, {64, 9}),
                            Tensor(f32.p, DType::FP32, {9}), l, m, Tensor(), nullptr);
    });
    refuse("a vector on Measure", [&] {
        ops::steer_residual(Step::Measure, x, Tensor(f32.p, DType::FP32, {64, 2}), Tensor(), lanes,
                            measures, Tensor(f32.p, DType::FP32, {64}), nullptr);
    });
    refuse("short measures", [&] {
        Tensor short_measures(f32.p, DType::FP32, {2, 1});
        ops::steer_residual(Step::Write, x, Tensor(f32.p, DType::FP32, {64, 2}),
                            Tensor(f32.p, DType::FP32, {2}), lanes, short_measures, Tensor(),
                            nullptr);
    });
    refuse("BF16 directions", [&] {
        ops::steer_residual(Step::Write, x, Tensor(f32.p, DType::BF16, {64, 2}),
                            Tensor(f32.p, DType::FP32, {2}), lanes, measures, Tensor(), nullptr);
    });
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_case({.label = "steer_residual Measure [5120,7] E3", .step = Step::Measure,
                          .hidden = 5120, .columns = 7, .count = 3, .lanes = 4,
                          .rows = {0, 1, 2, 3, 2, 7, 1}, .idle_lanes = {2}, .seed = 11});
    failures += run_case({.label = "steer_residual Embedding [5120,5] E2", .step = Step::Embedding,
                          .hidden = 5120, .columns = 5, .count = 2, .lanes = 1, .row_width = 5,
                          .rows = {0}, .seed = 21});
    failures += run_case({.label = "steer_residual Write [5120,6] E2 vector",
                          .step = Step::Write, .hidden = 5120, .columns = 6, .count = 2,
                          .lanes = 4, .row_width = 2, .rows = {0, 3, 1}, .idle_lanes = {3},
                          .vector = true, .seed = 31});
    failures += run_case({.label = "steer_residual Write [2048,9] E8", .step = Step::Write,
                          .hidden = 2048, .columns = 9, .count = 8, .lanes = 8,
                          .rows = {0, 1, 2, 3, 4, 5, 6, 7, 0}, .seed = 41});
    failures += run_case({.label = "steer_residual Write [4096,3] vector only",
                          .step = Step::Write, .hidden = 4096, .columns = 3, .count = 0,
                          .lanes = 1, .rows = {0, 0, 0}, .vector = true, .seed = 51});
    failures += run_case({.label = "steer_residual Embedding cancellation [2048,2]",
                          .step = Step::Embedding, .hidden = 2048, .columns = 2, .count = 1,
                          .lanes = 1, .rows = {0, 0}, .along = 6.0F, .seed = 61});
    failures += run_case({.label = "steer_residual Write [1000,4] partial chunks",
                          .step = Step::Write, .hidden = 1000, .columns = 4, .count = 1,
                          .lanes = 2, .rows = {1, 0, 1, 0}, .vector = true, .seed = 71});
    // Widths the chunked kernel does not take: not a multiple of 8, and past its registers.
    failures += run_case({.label = "steer_residual Write [1003,4] element loop",
                          .step = Step::Write, .hidden = 1003, .columns = 4, .count = 2,
                          .lanes = 2, .rows = {1, 0, 1, 0}, .vector = true, .seed = 81});
    failures += run_case({.label = "steer_residual Measure [9216,3] element loop",
                          .step = Step::Measure, .hidden = 9216, .columns = 3, .count = 2,
                          .lanes = 2, .rows = {0, 1, 0}, .seed = 91});
    failures += run_case({.label = "steer_residual Embedding [9216,2] element loop",
                          .step = Step::Embedding, .hidden = 9216, .columns = 2, .count = 1,
                          .lanes = 1, .rows = {0, 0}, .seed = 101});
    failures += run_round_trip();
    failures += run_refusals();
    std::cout << (failures ? "FAIL" : "OK") << " steer_residual\n";
    return failures ? 1 : 0;
}
