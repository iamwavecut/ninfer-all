#include "core/device_buffer.h"
#include "ninfer/ops/logprob_topk.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int K = ops::kLogprobTopK;

// A logprob is a logit difference less a log-sum-exp over the whole domain; both are O(10) and
// the FP32 reduction's error is a few ulps of them.
const PointwiseCriterion kLogprobCriterion{/*absolute=*/3.0e-5, /*relative=*/1.0e-5};

// Qwen3.5's head: 248,320 physical rows, of which 248,077 are public tokens.
constexpr std::int32_t kPhysicalRows = 248320;
constexpr std::int32_t kTokenDomain  = 248077;

struct CaseSpec {
    std::int32_t physical_rows = kPhysicalRows;
    std::int32_t token_domain  = kTokenDomain;
    std::int32_t columns       = 1;
    std::int32_t batch         = 1;
    float temperature          = 1.0f;
    float presence             = 0.0f;
    float frequency            = 0.0f;
    bool counts                = false;
    bool mask                  = false;
    bool ties                  = false;
    // FP32 logits, unrounded; BF16 otherwise.
    bool fp32 = false;
};

struct Inputs {
    std::vector<float> logits;              // the represented values, [physical, columns, batch]
    std::vector<std::int32_t> counts;       // [token_domain, batch]
    std::vector<std::int32_t> drafts;       // [columns - 1, batch]
    std::vector<std::uint32_t> masks;       // [words, columns, batch]
    std::int32_t words = 0;
};

Inputs make_inputs(const CaseSpec& spec, std::uint32_t seed) {
    Inputs in;
    const std::size_t rows = static_cast<std::size_t>(spec.columns) * spec.batch;
    in.logits.resize(static_cast<std::size_t>(spec.physical_rows) * rows);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> logit(-6.0f, 6.0f);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::int32_t v = 0; v < spec.physical_rows; ++v) {
            const float value = spec.ties ? static_cast<float>((v + r) % 5) : logit(rng);
            in.logits[r * spec.physical_rows + v] =
                spec.fp32 ? value : bf16_to_f32(f32_to_bf16(value));
        }
    }
    if (spec.counts) {
        in.counts.resize(static_cast<std::size_t>(spec.token_domain) * spec.batch);
        for (auto& count : in.counts) { count = static_cast<std::int32_t>(rng() % 3u); }
    }
    if (spec.columns > 1) {
        in.drafts.resize(static_cast<std::size_t>(spec.columns - 1) * spec.batch);
        // Drafts repeat a few tokens so the overlay changes later columns' penalties.
        for (auto& token : in.drafts) {
            token = static_cast<std::int32_t>(rng() % 64u) * (spec.token_domain / 64);
        }
    }
    if (spec.mask) {
        in.words = (spec.token_domain + 31) / 32;
        in.masks.assign(static_cast<std::size_t>(in.words) * rows, 0u);
        for (std::int32_t b = 0; b < spec.batch; ++b) {
            for (std::int32_t c = 0; c < spec.columns; ++c) {
                std::uint32_t* words =
                    in.masks.data() + (static_cast<std::size_t>(b) * spec.columns + c) * in.words;
                // Every other column admits a random half of the domain; the others admit seven
                // tokens, fewer than the reported top set.
                const bool narrow = (b + c) % 2 == 1;
                for (std::int32_t v = 0; v < spec.token_domain; ++v) {
                    const std::int32_t step = spec.token_domain / 7;
                    const bool allowed =
                        narrow ? v % step == 3 && v / step < 7 : (rng() & 1u) != 0;
                    if (allowed) { words[v / 32] |= 1u << (v % 32); }
                }
            }
        }
    }
    return in;
}

struct Oracle {
    std::vector<std::int32_t> ids;    // [K, columns, batch]
    std::vector<double> values;       // finite slots only, in slot order
    std::vector<std::size_t> finite;  // the slots `values` covers
    std::vector<double> lse;          // [columns, batch]
};

// Independent FP64 evaluation of the contract over the represented inputs.
Oracle oracle(const CaseSpec& spec, const Inputs& in) {
    Oracle out;
    const std::size_t rows = static_cast<std::size_t>(spec.columns) * spec.batch;
    out.ids.assign(rows * K, -1);
    out.lse.assign(rows, 0.0);
    const double temperature = spec.temperature > 0.0f ? spec.temperature : 1.0;
    const bool penalties     = spec.presence != 0.0f || spec.frequency != 0.0f;
    std::vector<double> scaled(spec.token_domain);
    std::vector<std::int32_t> order;
    for (std::int32_t b = 0; b < spec.batch; ++b) {
        for (std::int32_t c = 0; c < spec.columns; ++c) {
            const std::size_t row = static_cast<std::size_t>(c) + static_cast<std::size_t>(b) * spec.columns;
            order.clear();
            for (std::int32_t v = 0; v < spec.token_domain; ++v) {
                if (spec.mask) {
                    const std::uint32_t* words =
                        in.masks.data() + (static_cast<std::size_t>(b) * spec.columns + c) * in.words;
                    if ((words[v / 32] & (1u << (v % 32))) == 0) { continue; }
                }
                double value = in.logits[row * spec.physical_rows + v];
                if (penalties) {
                    std::int32_t count =
                        spec.counts ? in.counts[static_cast<std::size_t>(b) * spec.token_domain + v] : 0;
                    for (std::int32_t j = 0; j < c; ++j) {
                        count += in.drafts[static_cast<std::size_t>(b) * (spec.columns - 1) + j] == v;
                    }
                    if (count > 0) { value -= spec.presence; }
                    value -= static_cast<double>(spec.frequency) * count;
                }
                scaled[v] = value / temperature;
                order.push_back(v);
            }
            double maximum = -std::numeric_limits<double>::infinity();
            for (const std::int32_t v : order) { maximum = std::max(maximum, scaled[v]); }
            double sum = 0.0;
            for (const std::int32_t v : order) { sum += std::exp(scaled[v] - maximum); }
            const double lse = maximum + std::log(sum);
            out.lse[row]     = lse;
            const std::size_t top = std::min<std::size_t>(K, order.size());
            std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(top),
                              order.end(), [&](std::int32_t a, std::int32_t z) {
                                  return scaled[a] != scaled[z] ? scaled[a] > scaled[z] : a < z;
                              });
            for (std::size_t k = 0; k < top; ++k) {
                out.ids[row * K + k] = order[k];
                out.finite.push_back(row * K + k);
                out.values.push_back(scaled[order[k]] - lse);
            }
        }
    }
    return out;
}

int run_case(const std::string& label, const CaseSpec& spec, std::uint32_t seed) {
    const Inputs in        = make_inputs(spec, seed);
    const Oracle expected  = oracle(spec, in);
    const std::size_t rows = static_cast<std::size_t>(spec.columns) * spec.batch;

    DeviceBuffer counts(in.counts.size() * sizeof(std::int32_t) + 4);
    DeviceBuffer masks(in.masks.size() * sizeof(std::uint32_t) + 4);
    DeviceBuffer drafts(in.drafts.size() * sizeof(std::int32_t) + 4);
    if (!in.counts.empty()) { counts.copy_from_host(in.counts.data(), in.counts.size() * 4); }
    if (!in.masks.empty()) { masks.copy_from_host(in.masks.data(), in.masks.size() * 4); }
    if (!in.drafts.empty()) { drafts.copy_from_host(in.drafts.data(), in.drafts.size() * 4); }
    std::vector<ops::SamplingConfig> configs(spec.batch);
    for (std::int32_t b = 0; b < spec.batch; ++b) {
        ops::SamplingConfig& config = configs[b];
        config.temperature          = spec.temperature;
        config.top_k                = 20;
        config.top_p                = 0.9f; // truncation does not change the reported values
        config.min_p                = 0.05f;
        config.presence_penalty     = spec.presence;
        config.frequency_penalty    = spec.frequency;
        config.token_counts         = spec.counts ? static_cast<std::int32_t*>(counts.p) +
                                                static_cast<std::ptrdiff_t>(b) * spec.token_domain
                                                  : nullptr;
        if (spec.mask) {
            config.token_mask = static_cast<const std::uint32_t*>(masks.p) +
                                static_cast<std::ptrdiff_t>(b) * spec.columns * in.words;
            config.token_mask_stride = in.words;
        }
    }
    DeviceBuffer config_buffer(configs.size() * sizeof(ops::SamplingConfig));
    config_buffer.copy_from_host(configs.data(), config_buffer.bytes);

    std::vector<std::uint16_t> packed(spec.fp32 ? 0 : in.logits.size());
    for (std::size_t i = 0; i < packed.size(); ++i) { packed[i] = f32_to_bf16(in.logits[i]); }
    GuardedDeviceBuffer logits(spec.fp32 ? in.logits.size() * sizeof(float)
                                         : packed.size() * sizeof(std::uint16_t));
    if (spec.fp32) {
        logits.copy_from_host(in.logits.data(), logits.bytes());
    } else {
        logits.copy_from_host(packed.data(), logits.bytes());
    }
    GuardedDeviceBuffer ids(rows * K * sizeof(std::int32_t));
    GuardedDeviceBuffer values(rows * K * sizeof(float));
    GuardedDeviceBuffer lse(rows * sizeof(float));
    DeviceBuffer active(sizeof(std::int32_t));
    const std::int32_t on = 1;
    active.copy_from_host(&on, sizeof(on));
    ids.fill(0xcd);
    values.fill(0xcd);
    lse.fill(0xcd);

    Tensor logits_tensor(logits.data(), spec.fp32 ? DType::FP32 : DType::BF16,
                         {spec.physical_rows, spec.columns, spec.batch});
    Tensor ids_tensor(ids.data(), DType::I32, {K, spec.columns, spec.batch});
    Tensor values_tensor(values.data(), DType::FP32, {K, spec.columns, spec.batch});
    Tensor lse_tensor(lse.data(), DType::FP32, {spec.columns, spec.batch});
    Tensor active_tensor(active.p, DType::I32, {1});
    Tensor drafts_tensor(drafts.p, DType::I32, {std::max(spec.columns - 1, 1), spec.batch});
    WorkspaceArena workspace(ops::logprob_topk_workspace_capacity_bytes(
        spec.token_domain, static_cast<std::int32_t>(rows)));
    ops::logprob_topk(logits_tensor, static_cast<const ops::SamplingConfig*>(config_buffer.p),
                      spec.columns > 1 ? &drafts_tensor : nullptr, spec.token_domain, ids_tensor,
                      values_tensor, lse_tensor, active_tensor, workspace, nullptr);
    cuda_synchronize();

    int failures = 0;
    const auto got_ids    = from_device<std::int32_t>(ids.data(), rows * K);
    const auto got_values = from_device<float>(values.data(), rows * K);
    failures += verify_exact((label + " top ids").c_str(), got_ids, expected.ids);
    std::vector<double> finite_values;
    for (const std::size_t slot : expected.finite) { finite_values.push_back(got_values[slot]); }
    failures += verify_pointwise(label + " top logprobs", finite_values, expected.values,
                                 kLogprobCriterion);
    std::size_t empty_slots = 0;
    for (std::size_t slot = 0; slot < got_ids.size(); ++slot) {
        if (expected.ids[slot] >= 0) { continue; }
        ++empty_slots;
        if (got_values[slot] != -std::numeric_limits<float>::infinity()) {
            std::cerr << label << ": empty slot " << slot << " holds " << got_values[slot] << '\n';
            ++failures;
            break;
        }
    }
    const auto got_lse = from_device<float>(lse.data(), rows);
    failures += verify_pointwise(label + " lse", std::vector<double>(got_lse.begin(), got_lse.end()),
                                 expected.lse, kLogprobCriterion);
    failures += logits.verify_guards(label + " logits");
    failures += ids.verify_guards(label + " ids");
    failures += values.verify_guards(label + " values");
    failures += lse.verify_guards(label + " lse");
    std::cout << label << ": " << rows << " rows, " << empty_slots << " empty slots\n";
    return failures;
}

// A cleared flag must leave every output untouched.
int run_inactive_case() {
    const CaseSpec spec{.physical_rows = 4096, .token_domain = 4096, .batch = 2};
    const Inputs in = make_inputs(spec, 7u);
    std::vector<std::uint16_t> packed(in.logits.size());
    for (std::size_t i = 0; i < packed.size(); ++i) { packed[i] = f32_to_bf16(in.logits[i]); }
    DeviceBuffer logits(packed.size() * sizeof(std::uint16_t));
    logits.copy_from_host(packed.data(), logits.bytes);
    std::vector<ops::SamplingConfig> configs(2);
    DeviceBuffer config_buffer(configs.size() * sizeof(ops::SamplingConfig));
    config_buffer.copy_from_host(configs.data(), config_buffer.bytes);
    GuardedDeviceBuffer ids(2 * K * sizeof(std::int32_t));
    GuardedDeviceBuffer values(2 * K * sizeof(float));
    GuardedDeviceBuffer lse(2 * sizeof(float));
    DeviceBuffer active(sizeof(std::int32_t));
    const std::int32_t off = 0;
    active.copy_from_host(&off, sizeof(off));
    ids.fill(0xcd);
    values.fill(0xcd);
    lse.fill(0xcd);
    Tensor ids_tensor(ids.data(), DType::I32, {K, 1, 2});
    Tensor values_tensor(values.data(), DType::FP32, {K, 1, 2});
    Tensor lse_tensor(lse.data(), DType::FP32, {1, 2});
    WorkspaceArena workspace(ops::logprob_topk_workspace_capacity_bytes(4096, 2));
    ops::logprob_topk(Tensor(logits.p, DType::BF16, {4096, 1, 2}),
                      static_cast<const ops::SamplingConfig*>(config_buffer.p), nullptr, 4096,
                      ids_tensor, values_tensor, lse_tensor, Tensor(active.p, DType::I32, {1}),
                      workspace, nullptr);
    cuda_synchronize();
    const auto got = from_device<std::uint8_t>(ids.data(), 2 * K * sizeof(std::int32_t));
    const auto lse_bytes = from_device<std::uint8_t>(lse.data(), 2 * sizeof(float));
    const auto untouched = [](const std::vector<std::uint8_t>& bytes) {
        return std::all_of(bytes.begin(), bytes.end(), [](std::uint8_t b) { return b == 0xcd; });
    };
    if (!untouched(got) || !untouched(lse_bytes)) {
        std::cerr << "inactive flag: outputs were written\n";
        return 1;
    }
    return 0;
}

struct Outputs {
    std::vector<std::int32_t> ids;
    std::vector<float> values, lse;
};

// The outputs of `batch` one-column rows of `physical` BF16-represented logits each, under
// default sampling configs.
Outputs compute_rows(const std::vector<float>& logits, std::int32_t physical, std::int32_t domain,
                     std::int32_t batch) {
    std::vector<std::uint16_t> packed(logits.size());
    for (std::size_t i = 0; i < packed.size(); ++i) { packed[i] = f32_to_bf16(logits[i]); }
    DeviceBuffer device_logits(packed.size() * sizeof(std::uint16_t));
    device_logits.copy_from_host(packed.data(), device_logits.bytes);
    const std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch));
    DeviceBuffer config_buffer(configs.size() * sizeof(ops::SamplingConfig));
    config_buffer.copy_from_host(configs.data(), config_buffer.bytes);
    const auto rows = static_cast<std::size_t>(batch);
    DeviceBuffer ids(rows * K * sizeof(std::int32_t)), values(rows * K * sizeof(float)),
        lse(rows * sizeof(float)), active(sizeof(std::int32_t));
    const std::int32_t on = 1;
    active.copy_from_host(&on, sizeof(on));
    Tensor ids_tensor(ids.p, DType::I32, {K, 1, batch});
    Tensor values_tensor(values.p, DType::FP32, {K, 1, batch});
    Tensor lse_tensor(lse.p, DType::FP32, {1, batch});
    WorkspaceArena workspace(ops::logprob_topk_workspace_capacity_bytes(domain, batch));
    ops::logprob_topk(Tensor(device_logits.p, DType::BF16, {physical, 1, batch}),
                      static_cast<const ops::SamplingConfig*>(config_buffer.p), nullptr, domain,
                      ids_tensor, values_tensor, lse_tensor, Tensor(active.p, DType::I32, {1}),
                      workspace, nullptr);
    cuda_synchronize();
    return {from_device<std::int32_t>(ids.p, rows * K), from_device<float>(values.p, rows * K),
            from_device<float>(lse.p, rows)};
}

// A row's top set, logprobs and log-sum-exp have the same bits alone (one decoding request) and
// beside other rows (a verification's columns, several requests' rows), so that a verified token's
// logprob is plain decoding's. One token of each row is 0 and the others lie in [-13, -11]: together
// they add about 2 to a sum of about 3, so a sum reduced in another order rounds differently and
// the log-sum-exp, near 1, shows it.
int run_row_independence_case() {
    const CaseSpec spec{.batch = 16};
    Inputs in = make_inputs(spec, 22u);
    for (std::size_t i = 0; i < in.logits.size(); ++i) {
        const std::size_t row = i / static_cast<std::size_t>(spec.physical_rows);
        const bool top        = i % static_cast<std::size_t>(spec.physical_rows) == 17 * row;
        in.logits[i] = top ? 0.0f : bf16_to_f32(f32_to_bf16(-12.0f + in.logits[i] / 6.0f));
    }
    const Outputs together =
        compute_rows(in.logits, spec.physical_rows, spec.token_domain, spec.batch);
    std::size_t differ = 0;
    for (std::int32_t b = 0; b < spec.batch; ++b) {
        const auto first = in.logits.begin() + static_cast<std::ptrdiff_t>(b) * spec.physical_rows;
        const Outputs alone = compute_rows(std::vector<float>(first, first + spec.physical_rows),
                                           spec.physical_rows, spec.token_domain, 1);
        const auto at = static_cast<std::size_t>(b);
        differ += std::memcmp(&alone.lse[0], &together.lse[at], sizeof(float)) != 0 ? 1 : 0;
        for (int k = 0; k < K; ++k) {
            const std::size_t slot = at * K + static_cast<std::size_t>(k);
            differ += alone.ids[static_cast<std::size_t>(k)] != together.ids[slot] ? 1 : 0;
            differ += std::memcmp(&alone.values[static_cast<std::size_t>(k)], &together.values[slot],
                                  sizeof(float)) != 0
                          ? 1
                          : 0;
        }
    }
    std::cout << "rows alone and beside others: " << differ << " outputs differ\n";
    return differ == 0 ? 0 : 1;
}

template <class Function>
int expect_invalid(const char* label, Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return 0;
    } catch (const std::exception& error) {
        std::cerr << label << ": expected invalid_argument, got " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

int run_validation_cases() {
    DeviceBuffer logits_data(64 * 2 * 3 * sizeof(std::uint16_t));
    DeviceBuffer config_data(3 * sizeof(ops::SamplingConfig));
    DeviceBuffer drafts_data(3 * sizeof(std::int32_t));
    DeviceBuffer ids_data(K * 6 * sizeof(std::int32_t));
    DeviceBuffer values_data(K * 6 * sizeof(float));
    DeviceBuffer lse_data(6 * sizeof(float));
    DeviceBuffer active_data(sizeof(std::int32_t));
    const Tensor logits(logits_data.p, DType::BF16, {64, 2, 3});
    const Tensor drafts(drafts_data.p, DType::I32, {1, 3});
    Tensor ids(ids_data.p, DType::I32, {K, 2, 3});
    Tensor values(values_data.p, DType::FP32, {K, 2, 3});
    Tensor lse(lse_data.p, DType::FP32, {2, 3});
    const Tensor active(active_data.p, DType::I32, {1});
    const auto* configs = static_cast<const ops::SamplingConfig*>(config_data.p);
    const auto call     = [&](const Tensor& l, const ops::SamplingConfig* c, const Tensor* d,
                          std::int32_t domain, Tensor& i, Tensor& v, Tensor& s) {
        WorkspaceArena workspace(1u << 20);
        ops::logprob_topk(l, c, d, domain, i, v, s, active, workspace, nullptr);
    };

    int failures = 0;
    failures += expect_invalid("null configs", [&] { call(logits, nullptr, &drafts, 64, ids, values, lse); });
    failures += expect_invalid("domain below the top set",
                               [&] { call(logits, configs, &drafts, K - 1, ids, values, lse); });
    failures += expect_invalid("domain above the physical rows",
                               [&] { call(logits, configs, &drafts, 65, ids, values, lse); });
    failures += expect_invalid("columns without drafts",
                               [&] { call(logits, configs, nullptr, 64, ids, values, lse); });
    failures += expect_invalid("drafts of the wrong width", [&] {
        const Tensor wide(drafts_data.p, DType::I32, {2, 1});
        call(logits, configs, &wide, 64, ids, values, lse);
    });
    failures += expect_invalid("ids of the wrong shape", [&] {
        Tensor wrong(ids_data.p, DType::I32, {K, 6});
        call(logits, configs, &drafts, 64, wrong, values, lse);
    });
    failures += expect_invalid("lse of the wrong dtype", [&] {
        Tensor wrong(lse_data.p, DType::BF16, {2, 3});
        call(logits, configs, &drafts, 64, ids, values, wrong);
    });
    failures += expect_invalid("strided logits", [&] {
        Tensor strided = logits;
        strided.nb[1] += 2;
        call(strided, configs, &drafts, 64, ids, values, lse);
    });
    failures += expect_invalid("ids over the logits", [&] {
        Tensor alias(logits_data.p, DType::I32, {K, 2, 3});
        call(logits, configs, &drafts, 64, alias, values, lse);
    });
    failures += expect_invalid("an empty workspace profile",
                               [&] { (void)ops::logprob_topk_workspace_capacity_bytes(64, 0); });
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    // Prefill and decode rows.
    failures += run_case("one row", {}, 11u);
    failures += run_case("eight rows at T=0.7", {.batch = 8, .temperature = 0.7f}, 12u);
    failures += run_case("greedy rows", {.batch = 4, .temperature = 0.0f}, 13u);
    failures += run_case("penalties", {.batch = 8, .presence = 0.5f, .frequency = 0.3f, .counts = true}, 14u);
    failures += run_case("token masks", {.batch = 8, .mask = true}, 15u);
    failures +=
        run_case("FP32 eight rows at T=0.7", {.batch = 8, .temperature = 0.7f, .fp32 = true}, 21u);
    failures += run_case("FP32 penalties and masks",
                         {.batch     = 8,
                          .presence  = 0.5f,
                          .frequency = 0.3f,
                          .counts    = true,
                          .mask      = true,
                          .fp32      = true},
                         22u);
    failures += run_case("ties", {.physical_rows = 4096, .token_domain = 4096, .ties = true}, 16u);
    failures += run_case("smallest domain", {.physical_rows = 20, .token_domain = 20, .batch = 3}, 17u);
    failures += run_case("split boundary 128", {.physical_rows = 128, .token_domain = 128, .batch = 2}, 18u);
    failures += run_case("split boundary 129", {.physical_rows = 129, .token_domain = 129, .batch = 2}, 19u);
    // Speculative verification: per-column masks and the drafts' penalty overlay.
    failures += run_case("MTP verification",
                         {.columns = 16, .batch = 8, .temperature = 0.6f, .presence = 0.4f,
                          .frequency = 0.2f, .counts = true, .mask = true},
                         20u);
    failures += run_case("widest ngram verification",
                         {.columns = 64, .batch = 8, .presence = 0.3f, .counts = true}, 21u);
    failures += run_row_independence_case();
    failures += run_inactive_case();
    failures += run_validation_cases();
    std::cout << (failures != 0 ? "FAIL" : "OK") << " logprob_topk\n";
    return failures != 0 ? 1 : 0;
}
