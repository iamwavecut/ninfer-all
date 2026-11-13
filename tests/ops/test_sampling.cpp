// Public-contract qualification for sample().
//
// The deterministic branch is checked exactly against an independent CPU
// argmax.  The stochastic branch is checked against one FP64 mathematical
// distribution oracle built from the values represented at the public input.
// Every contract runs on BF16 logits, then on FP32 logits holding the same
// values.  The test never reproduces the device RNG algorithm or uses another
// production path as a golden.
#include "core/device_buffer.h"
#include "ninfer/ops/sampling.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

struct Candidate {
    double adjusted = 0.0;
    int token       = 0;
};

struct Distribution {
    std::vector<int> tokens;
    std::vector<double> probabilities;
};

struct RunResult {
    std::vector<int> tokens;
    std::vector<std::vector<int>> counts;
    int integrity_failures = 0;
};

bool same_config(const ops::SamplingConfig& a, const ops::SamplingConfig& b) {
    return a.temperature == b.temperature && a.top_k == b.top_k && a.top_p == b.top_p &&
           a.min_p == b.min_p && a.presence_penalty == b.presence_penalty &&
           a.frequency_penalty == b.frequency_penalty && a.seed == b.seed &&
           a.token_counts == b.token_counts && a.token_mask == b.token_mask &&
           a.token_mask_stride == b.token_mask_stride && a.draw == b.draw &&
           a.non_finite_token == b.non_finite_token;
}

// The device's counter-based uniform (sampling_device.cuh), bit for bit: splitmix64 over the
// seed, the position and then the purpose and sub-key, its top 24 bits over 2^24.
std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

double keyed_uniform(std::uint64_t seed, int position, int purpose, unsigned sub) {
    std::uint64_t key = splitmix64(
        seed ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(position)) *
                0xD1B54A32D192ED03ull));
    key = splitmix64(key ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(purpose)) << 21) ^
                     (static_cast<std::uint64_t>(sub) * 0x2545F4914F6CDD1Dull));
    return static_cast<double>(key >> 40) / 16777216.0;
}

// The Gumbel draw's FP64 oracle: the support token whose log probability plus
// -log(-log(u + 2^-25)) keyed by its id is largest; -1 when the best two lie within `margin`,
// where the device's FP32 arithmetic may pick either.
int gumbel_oracle(const Distribution& distribution, std::uint64_t seed, int position,
                  int purpose, double margin) {
    int best = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    double second     = -std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < distribution.tokens.size(); ++j) {
        const double u = keyed_uniform(seed, position, purpose,
                                       static_cast<unsigned>(distribution.tokens[j]) + 1u) +
                         0.5 / 16777216.0;
        const double score = std::log(distribution.probabilities[j]) - std::log(-std::log(u));
        if (score > best_score) {
            second     = best_score;
            best_score = score;
            best       = distribution.tokens[j];
        } else if (score > second) {
            second = score;
        }
    }
    return best_score - second < margin ? -1 : best;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

// The logits dtype the contracts run on.
DType g_logits = DType::BF16;

// Logits in g_logits on the device, and their bytes for the read-only check.
struct DeviceLogits {
    DeviceBuffer buffer;
    std::vector<std::uint8_t> bytes;
};

DeviceLogits upload_logits(const std::vector<float>& values) {
    std::vector<std::uint8_t> bytes;
    if (g_logits == DType::FP32) {
        bytes.resize(values.size() * sizeof(float));
        std::memcpy(bytes.data(), values.data(), bytes.size());
    } else {
        const std::vector<std::uint16_t> bits = bf16_bits(values);
        bytes.resize(bits.size() * sizeof(std::uint16_t));
        std::memcpy(bytes.data(), bits.data(), bytes.size());
    }
    DeviceBuffer buffer = to_device(bytes);
    return {std::move(buffer), std::move(bytes)};
}

std::vector<float> repeat_column(const std::vector<float>& column, int columns) {
    std::vector<float> logits(column.size() * static_cast<std::size_t>(columns));
    for (int t = 0; t < columns; ++t) {
        std::copy(column.begin(), column.end(),
                  logits.begin() + static_cast<std::ptrdiff_t>(t) * column.size());
    }
    return logits;
}

std::vector<int> greedy_oracle(const std::vector<float>& logits, int physical_rows,
                               int token_domain, int columns, const ops::SamplingConfig& config,
                               const std::vector<int>& counts) {
    std::vector<int> expected(static_cast<std::size_t>(columns));
    for (int t = 0; t < columns; ++t) {
        const std::size_t base = static_cast<std::size_t>(t) * physical_rows;
        int best               = 0;
        for (int token = 1; token < token_domain; ++token) {
            const auto adjusted = [&](int candidate) {
                const int count = counts[static_cast<std::size_t>(candidate)];
                return logits[base + candidate] - config.presence_penalty * (count > 0) -
                       config.frequency_penalty * count;
            };
            if (adjusted(token) > adjusted(best)) { best = token; }
        }
        expected[static_cast<std::size_t>(t)] = best;
    }
    return expected;
}

Distribution distribution_oracle(const std::vector<float>& column, int token_domain,
                                 const ops::SamplingConfig& config,
                                 const std::vector<int>* counts = nullptr) {
    std::vector<Candidate> candidates(static_cast<std::size_t>(token_domain));
    for (int token = 0; token < token_domain; ++token) {
        const int count = counts == nullptr ? 0 : (*counts)[static_cast<std::size_t>(token)];
        double adjusted = static_cast<double>(column[static_cast<std::size_t>(token)]);
        if (count > 0) { adjusted -= static_cast<double>(config.presence_penalty); }
        adjusted -= static_cast<double>(config.frequency_penalty) * static_cast<double>(count);
        candidates[static_cast<std::size_t>(token)] = {adjusted, token};
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (a.adjusted != b.adjusted) { return a.adjusted > b.adjusted; }
        return a.token < b.token;
    });

    int cap = 20;
    if (config.top_k > 0 && config.top_k < 20) { cap = config.top_k; }
    cap = std::min(cap, token_domain);
    candidates.resize(static_cast<std::size_t>(cap));

    std::vector<double> weights(static_cast<std::size_t>(cap));
    const double max_scaled = candidates.front().adjusted / config.temperature;
    double total_weight     = 0.0;
    for (int rank = 0; rank < cap; ++rank) {
        const double weight = std::exp(
            candidates[static_cast<std::size_t>(rank)].adjusted / config.temperature - max_scaled);
        weights[static_cast<std::size_t>(rank)] = weight;
        total_weight += weight;
    }

    const bool use_min_p     = config.min_p > 0.0f;
    const bool use_top_p     = config.top_p < 1.0f;
    const double min_weight  = static_cast<double>(config.min_p) * weights.front();
    const double top_p_limit = static_cast<double>(config.top_p) * total_weight;
    double cumulative        = 0.0;
    int support              = 0;
    for (int rank = 0; rank < cap; ++rank) {
        if (use_min_p && weights[static_cast<std::size_t>(rank)] < min_weight) { break; }
        cumulative += weights[static_cast<std::size_t>(rank)];
        support = rank + 1;
        if (use_top_p && cumulative >= top_p_limit) { break; }
    }
    support = std::max(support, 1);

    Distribution out;
    out.tokens.reserve(static_cast<std::size_t>(support));
    out.probabilities.reserve(static_cast<std::size_t>(support));
    double kept_weight = 0.0;
    for (int rank = 0; rank < support; ++rank) {
        kept_weight += weights[static_cast<std::size_t>(rank)];
    }
    for (int rank = 0; rank < support; ++rank) {
        out.tokens.push_back(candidates[static_cast<std::size_t>(rank)].token);
        out.probabilities.push_back(weights[static_cast<std::size_t>(rank)] / kept_weight);
    }
    return out;
}

RunResult run_batch(const std::vector<float>& logits, int physical_rows, int token_domain,
                    std::vector<ops::SamplingConfig> configs,
                    const std::vector<int>& logical_positions, int purpose,
                    const std::vector<std::vector<int>>& initial_counts = {}) {
    const int batch = static_cast<int>(configs.size());
    if (batch <= 0 || logical_positions.size() != configs.size() ||
        logits.size() != static_cast<std::size_t>(physical_rows) * configs.size() ||
        (!initial_counts.empty() && initial_counts.size() != configs.size())) {
        throw std::invalid_argument("invalid sample batch fixture");
    }
    const DeviceLogits device_logits = upload_logits(logits);
    GuardedDeviceBuffer device_out(static_cast<std::size_t>(batch) * sizeof(std::int32_t));
    const std::vector<int> output_sentinel(static_cast<std::size_t>(batch), -777777);
    device_out.copy_from_host(output_sentinel.data(),
                              output_sentinel.size() * sizeof(std::int32_t));

    std::vector<std::unique_ptr<GuardedDeviceBuffer>> device_counts;
    if (!initial_counts.empty()) {
        device_counts.reserve(configs.size());
        for (std::size_t row = 0; row < configs.size(); ++row) {
            if (initial_counts[row].size() != static_cast<std::size_t>(token_domain)) {
                throw std::invalid_argument("invalid sample token-count fixture");
            }
            auto counts = std::make_unique<GuardedDeviceBuffer>(initial_counts[row].size() *
                                                                sizeof(std::int32_t));
            counts->copy_from_host(initial_counts[row].data(), counts->bytes());
            configs[row].token_counts = static_cast<std::int32_t*>(counts->data());
            device_counts.push_back(std::move(counts));
        }
    }
    const std::vector<ops::SamplingConfig> expected_configs = configs;
    DeviceBuffer device_configs                             = to_device(configs);
    DeviceBuffer device_positions                           = to_device(logical_positions);

    Tensor logits_tensor(device_logits.buffer.p, g_logits, {physical_rows, batch});
    Tensor out_tensor(device_out.data(), DType::I32, {batch});
    Tensor positions_tensor(device_positions.p, DType::I32, {batch});
    const std::size_t workspace_bytes =
        ops::sampling_workspace_capacity_bytes(token_domain, batch, batch);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    ops::sample(logits_tensor, out_tensor, token_domain,
                static_cast<const ops::SamplingConfig*>(device_configs.p), positions_tensor,
                purpose, workspace, nullptr);
    cuda_synchronize();

    RunResult result;
    result.tokens = from_device<int>(device_out.data(), static_cast<std::size_t>(batch));
    result.integrity_failures += device_out.verify_guards("sample output");
    result.integrity_failures +=
        verify_exact("sample read-only logits",
                     from_device<std::uint8_t>(device_logits.buffer, device_logits.bytes.size()),
                     device_logits.bytes);

    const std::vector<ops::SamplingConfig> actual_configs =
        from_device<ops::SamplingConfig>(device_configs, configs.size());
    for (std::size_t row = 0; row < configs.size(); ++row) {
        if (!same_config(actual_configs[row], expected_configs[row])) {
            std::cerr << "sample modified SamplingConfig row " << row << '\n';
            ++result.integrity_failures;
        }
    }
    result.integrity_failures += verify_exact(
        "sample read-only logical positions",
        from_device<std::int32_t>(device_positions, configs.size()), logical_positions);

    result.counts.reserve(device_counts.size());
    for (std::size_t row = 0; row < device_counts.size(); ++row) {
        result.counts.push_back(
            from_device<int>(device_counts[row]->data(), static_cast<std::size_t>(token_domain)));
        result.integrity_failures += device_counts[row]->verify_guards("sample token_counts");
    }
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << "sample workspace query/execution high-water mismatch\n";
        ++result.integrity_failures;
    }
    return result;
}

RunResult run_homogeneous_batch(const std::vector<float>& logits, int physical_rows,
                                int token_domain, int batch, ops::SamplingConfig config,
                                int first_position, int purpose,
                                const std::vector<int>* initial_counts = nullptr) {
    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch), config);
    std::vector<int> positions(static_cast<std::size_t>(batch));
    for (int row = 0; row < batch; ++row) {
        positions[static_cast<std::size_t>(row)] = first_position + row;
    }
    std::vector<std::vector<int>> row_counts;
    if (initial_counts != nullptr) {
        row_counts.assign(static_cast<std::size_t>(batch), *initial_counts);
    }
    return run_batch(logits, physical_rows, token_domain, std::move(configs), positions, purpose,
                     row_counts);
}

// `offset`: sample()'s position_offset; the keys are then position + offset + i.
RunResult run_repeated(const std::vector<float>& column, int token_domain, int total, int batch,
                       ops::SamplingConfig config, int position, int purpose, int offset = 0) {
    if (batch <= 0 || total <= 0 || total % batch != 0) {
        throw std::invalid_argument("repeated sample count must be a positive batch multiple");
    }
    const int physical_rows                     = static_cast<int>(column.size());
    const DeviceLogits device_logits            = upload_logits(repeat_column(column, batch));
    GuardedDeviceBuffer collected(static_cast<std::size_t>(total) * sizeof(std::int32_t));
    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch), config);
    const std::vector<ops::SamplingConfig> expected_configs = configs;
    DeviceBuffer device_configs                             = to_device(configs);
    std::vector<int> positions(static_cast<std::size_t>(total));
    for (int i = 0; i < total; ++i) { positions[static_cast<std::size_t>(i)] = position + i; }
    DeviceBuffer device_positions = to_device(positions);

    Tensor logits_tensor(device_logits.buffer.p, g_logits, {physical_rows, batch});
    const std::size_t workspace_bytes =
        ops::sampling_workspace_capacity_bytes(token_domain, batch, batch);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));

    for (int produced = 0; produced < total; produced += batch) {
        auto* out = static_cast<std::int32_t*>(collected.data()) + produced;
        auto* pos = static_cast<std::int32_t*>(device_positions.p) + produced;
        Tensor out_tensor(out, DType::I32, {batch});
        Tensor positions_tensor(pos, DType::I32, {batch});
        ops::sample(logits_tensor, out_tensor, token_domain,
                    static_cast<const ops::SamplingConfig*>(device_configs.p), positions_tensor,
                    purpose, workspace, nullptr, offset);
    }
    cuda_synchronize();

    RunResult result;
    result.tokens = from_device<int>(collected.data(), static_cast<std::size_t>(total));
    result.integrity_failures += collected.verify_guards("sample repeated output");
    result.integrity_failures +=
        verify_exact("sample repeated read-only logits",
                     from_device<std::uint8_t>(device_logits.buffer, device_logits.bytes.size()),
                     device_logits.bytes);
    const std::vector<ops::SamplingConfig> actual_configs =
        from_device<ops::SamplingConfig>(device_configs, configs.size());
    for (std::size_t row = 0; row < configs.size(); ++row) {
        if (!same_config(actual_configs[row], expected_configs[row])) {
            std::cerr << "sample repeated modified SamplingConfig row " << row << '\n';
            ++result.integrity_failures;
        }
    }
    result.integrity_failures +=
        verify_exact("sample repeated read-only logical positions",
                     from_device<std::int32_t>(device_positions, positions.size()), positions);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << "sample repeated workspace query/execution high-water mismatch\n";
        ++result.integrity_failures;
    }
    return result;
}

int verify_distribution(const char* label, const std::vector<int>& samples,
                        const Distribution& expected) {
    std::vector<int> observed(expected.tokens.size(), 0);
    for (int token : samples) {
        const auto it = std::find(expected.tokens.begin(), expected.tokens.end(), token);
        if (it == expected.tokens.end()) {
            std::cerr << label << ": sampled token " << token << " outside oracle support\n";
            return 1;
        }
        ++observed[static_cast<std::size_t>(it - expected.tokens.begin())];
    }

    const double n              = static_cast<double>(samples.size());
    double max_standardized_gap = 0.0;
    for (std::size_t i = 0; i < expected.tokens.size(); ++i) {
        const double probability = expected.probabilities[i];
        const double frequency   = static_cast<double>(observed[i]) / n;
        const double sigma       = std::sqrt(probability * (1.0 - probability) / n);
        const double limit       = 7.0 * sigma + 2.0 / n;
        const double gap         = std::abs(frequency - probability);
        if (gap > limit) {
            std::cerr << label << ": token=" << expected.tokens[i] << " frequency=" << frequency
                      << " oracle=" << probability << " gap=" << gap << " limit=" << limit << '\n';
            return 1;
        }
        if (sigma > 0.0) { max_standardized_gap = std::max(max_standardized_gap, gap / sigma); }
    }
    std::cout << "    " << label << " FP64 distribution match (max z=" << max_standardized_gap
              << ")\n";
    return 0;
}

int greedy_contract() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    constexpr int batch         = 8;
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * batch, -9.0f);
    for (int row = 0; row < batch; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * physical_rows;
        int first              = (17 + 7919 * row) % token_domain;
        int second             = token_domain - 1 - ((31 + 65537 * row) % token_domain);
        if (second == first) { second = (first + 1) % token_domain; }
        if (second < first) { std::swap(first, second); }
        logits[base + first]             = row == 0 ? -0.0f : 16.0f + row;
        logits[base + second]            = row == 0 ? 0.0f : 16.0f + row;
        logits[base + token_domain]      = 100.0f;
        logits[base + physical_rows - 1] = 200.0f;
    }
    round_to_bf16(logits);

    std::vector<int> counts(static_cast<std::size_t>(token_domain), 0);
    counts[17] = 9;
    ops::SamplingConfig config;
    config.temperature       = 0.0f;
    config.top_k             = 1;
    config.top_p             = 0.01f;
    config.min_p             = 0.99f;
    config.presence_penalty  = 100.0f;
    config.frequency_penalty = 100.0f;
    config.seed              = 12345;

    const RunResult result = run_homogeneous_batch(logits, physical_rows, token_domain, batch,
                                                   config, 77, ops::kSamplePurposeDecode, &counts);
    int failures           = result.integrity_failures;
    failures +=
        verify_exact("sample greedy mathematical result", result.tokens,
                     greedy_oracle(logits, physical_rows, token_domain, batch, config, counts));
    for (std::size_t row = 0; row < result.counts.size(); ++row) {
        std::vector<int> expected_counts = counts;
        ++expected_counts[static_cast<std::size_t>(result.tokens[row])];
        failures += verify_exact("sample greedy increments selected token", result.counts[row],
                                 expected_counts);
    }
    return failures;
}

int deterministic_stochastic_contract() {
    std::vector<float> column = {5.0f, 4.5f, 4.0f, 3.0f, -1.0f};
    round_to_bf16(column);
    int failures = 0;

    struct Case {
        const char* label;
        float presence;
        float frequency;
        std::vector<int> counts;
    };

    const Case cases[] = {
        {"sample positive-temperature presence penalty", 1.0f, 0.0f, {1, 0, 0, 0, 0}},
        {"sample positive-temperature frequency penalty", 0.0f, 0.5f, {2, 0, 0, 0, 0}},
        {"sample adjusted-logit tie break", 0.5f, 0.0f, {1, 0, 0, 0, 0}},
    };
    for (const Case& test_case : cases) {
        ops::SamplingConfig config;
        config.temperature       = 0.8f;
        config.top_k             = 1;
        config.presence_penalty  = test_case.presence;
        config.frequency_penalty = test_case.frequency;
        config.seed              = 9981;
        const Distribution oracle =
            distribution_oracle(column, static_cast<int>(column.size()), config, &test_case.counts);
        RunResult result = run_homogeneous_batch(column, static_cast<int>(column.size()),
                                                 static_cast<int>(column.size()), 1, config, 11,
                                                 ops::kSamplePurposeDecode, &test_case.counts);

        std::vector<int> expected_counts = test_case.counts;
        ++expected_counts[static_cast<std::size_t>(oracle.tokens.front())];
        failures += result.integrity_failures;
        failures += verify_exact(test_case.label, result.tokens, {oracle.tokens.front()});
        failures += verify_exact("sample increments only selected token", result.counts.front(),
                                 expected_counts);
    }
    return failures;
}

int heterogeneous_batch_contract() {
    constexpr int physical_rows = 260;
    constexpr int token_domain  = 257;
    constexpr int batch         = 4;
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * batch, -8.0f);
    const auto set = [&](int row, int token, float value) {
        logits[static_cast<std::size_t>(row) * physical_rows + token] = value;
    };
    set(0, 5, 4.0f);
    set(0, 7, 3.0f);
    set(1, 11, 4.0f);
    set(1, 12, 3.0f);
    set(2, 13, 5.0f);
    set(2, 17, 4.5f);
    set(3, 19, 2.0f);
    set(3, 23, 2.0f);
    for (int row = 0; row < batch; ++row) { set(row, physical_rows - 1, 100.0f); }
    round_to_bf16(logits);

    std::vector<ops::SamplingConfig> configs(batch);
    configs[0].temperature      = 0.0f;
    configs[1].temperature      = 0.7f;
    configs[1].top_k            = 1;
    configs[1].seed             = 101;
    configs[2].temperature      = 0.7f;
    configs[2].top_k            = 1;
    configs[2].presence_penalty = 1.0f;
    configs[2].seed             = 202;
    configs[3].temperature      = 0.0f;

    std::vector<std::vector<int>> counts(batch, std::vector<int>(token_domain, 0));
    counts[0][5]           = 9;
    counts[2][13]          = 1;
    const RunResult result = run_batch(logits, physical_rows, token_domain, std::move(configs),
                                       {7, 103, 999, 41}, ops::kSamplePurposeDecode, counts);

    int failures = result.integrity_failures;
    failures += verify_exact("sample heterogeneous batch tokens", result.tokens, {5, 11, 17, 19});
    std::vector<std::vector<int>> expected = counts;
    for (int row = 0; row < batch; ++row) {
        ++expected[static_cast<std::size_t>(row)]
                  [static_cast<std::size_t>(result.tokens[static_cast<std::size_t>(row)])];
    }
    for (int row = 0; row < batch; ++row) {
        failures += verify_exact("sample heterogeneous batch isolated token counts",
                                 result.counts[static_cast<std::size_t>(row)],
                                 expected[static_cast<std::size_t>(row)]);
    }
    return failures;
}

int filtered_distribution_contract() {
    std::vector<float> column = {3.0f, 2.7f,  2.7f,  2.1f,  1.5f,  0.7f,
                                 0.1f, -0.4f, -1.0f, -2.0f, -3.0f, -4.0f};
    round_to_bf16(column);
    ops::SamplingConfig config;
    config.temperature = 0.75f;
    config.top_k       = 6;
    config.top_p       = 0.86f;
    config.min_p       = 0.12f;
    config.seed        = 20260726;

    const Distribution oracle =
        distribution_oracle(column, static_cast<int>(column.size()), config);
    if (oracle.tokens.size() < 2 || oracle.tokens.size() >= 6) {
        std::cerr << "filtered distribution fixture did not exercise both filters\n";
        return 1;
    }

    constexpr int samples  = 16384;
    const RunResult result = run_repeated(column, static_cast<int>(column.size()), samples, 8,
                                          config, 400, ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_distribution("sample top-k/top-p/min-p", result.tokens, oracle);
}

// The Gumbel draw samples the truncated distribution the inverse-CDF draw samples, and each of
// its draws is the FP64 oracle's token for its key, on the small-column route, the multi-block
// workspace route and the single-block route past 16 lanes.
int gumbel_contract() {
    int failures = 0;
    std::vector<float> small = {3.0f, 2.7f,  2.7f,  2.1f,  1.5f,  0.7f,
                                0.1f, -0.4f, -1.0f, -2.0f, -3.0f, -4.0f};
    round_to_bf16(small);
    ops::SamplingConfig config;
    config.temperature = 0.75f;
    config.top_k       = 6;
    config.top_p       = 0.86f;
    config.min_p       = 0.12f;
    config.seed        = 20261111;
    config.draw        = ops::kSampleDrawGumbel;
    {
        const Distribution oracle =
            distribution_oracle(small, static_cast<int>(small.size()), config);
        const RunResult result = run_repeated(small, static_cast<int>(small.size()), 16384, 8,
                                              config, 400, ops::kSamplePurposeDecode);
        failures += result.integrity_failures +
                    verify_distribution("sample Gumbel top-k/top-p/min-p", result.tokens, oracle);
    }
    std::vector<float> wide(300, -30.0f);
    for (int token = 0; token < 24; ++token) {
        wide[static_cast<std::size_t>(11 * token + 7)] = 1.5f - 0.11f * static_cast<float>(token);
    }
    round_to_bf16(wide);
    config.top_k = 16;
    config.top_p = 0.97f;
    config.min_p = 0.0f;
    struct Route {
        const std::vector<float>* column;
        int batch;
        const char* name;
    };
    for (const Route& route : {Route{&small, 8, "small"}, Route{&wide, 8, "workspace"},
                               Route{&wide, 17, "single-block"}}) {
        const int domain          = static_cast<int>(route.column->size());
        const Distribution oracle = distribution_oracle(*route.column, domain, config);
        const int total           = route.batch * 16;
        const RunResult gumbel =
            run_repeated(*route.column, domain, total, route.batch, config, 900,
                         ops::kSamplePurposeDecode);
        ops::SamplingConfig inverse = config;
        inverse.draw                = ops::kSampleDrawInverseCdf;
        const RunResult cdf = run_repeated(*route.column, domain, total, route.batch, inverse, 900,
                                           ops::kSamplePurposeDecode);
        failures += gumbel.integrity_failures + cdf.integrity_failures;
        int compared = 0;
        int differ   = 0;
        for (int i = 0; i < total; ++i) {
            const int token = gumbel.tokens[static_cast<std::size_t>(i)];
            differ += token != cdf.tokens[static_cast<std::size_t>(i)];
            const int expected =
                gumbel_oracle(oracle, config.seed, 900 + i, ops::kSamplePurposeDecode, 1e-3);
            if (expected < 0) { continue; }
            ++compared;
            if (token != expected) {
                if (failures == 0) {
                    std::cerr << "sample Gumbel " << route.name << " position " << 900 + i
                              << ": token " << token << ", oracle " << expected << '\n';
                }
                ++failures;
            }
        }
        if (compared < total * 3 / 4 || differ == 0) {
            std::cerr << "sample Gumbel " << route.name << ": " << compared << " of " << total
                      << " draws compared, " << differ << " differ from the inverse-CDF draw\n";
            ++failures;
        }
    }
    return failures;
}

int capped_distribution_contract() {
    std::vector<float> column(24, 0.0f);
    for (int token = 0; token < 24; ++token) {
        column[static_cast<std::size_t>(token)] = 2.0f - 0.1f * token;
    }
    round_to_bf16(column);
    ops::SamplingConfig config;
    config.temperature = 1.1f;
    config.top_k       = 64;
    config.seed        = 884422;

    const Distribution oracle =
        distribution_oracle(column, static_cast<int>(column.size()), config);
    if (oracle.tokens.size() != 20) {
        std::cerr << "top-k cap oracle fixture has unexpected support\n";
        return 1;
    }

    constexpr int samples  = 16384;
    const RunResult result = run_repeated(column, static_cast<int>(column.size()), samples, 8,
                                          config, 900, ops::kSamplePurposePrefill);
    return result.integrity_failures +
           verify_distribution("sample top-k public cap", result.tokens, oracle);
}

int real_shape_distribution_contract() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    std::vector<float> column(physical_rows, -20.0f);
    const int ids[]      = {17, 7919, 65537, 200003};
    const float logits[] = {3.0f, 2.0f, 1.0f, 0.0f};
    for (int i = 0; i < 4; ++i) { column[ids[i]] = logits[i]; }
    column[token_domain]      = 100.0f;
    column[physical_rows - 1] = 200.0f;
    round_to_bf16(column);

    ops::SamplingConfig config;
    config.temperature        = 1.0f;
    config.top_k              = 4;
    config.seed               = 7654321;
    const Distribution oracle = distribution_oracle(column, token_domain, config);

    constexpr int samples = 4096;
    RunResult result =
        run_repeated(column, token_domain, samples, 8, config, 2000, ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_distribution("sample real token-domain B=8", result.tokens, oracle);
}

int rng_key_contract() {
    std::vector<float> column = {0.0f, 0.0f};
    round_to_bf16(column);
    ops::SamplingConfig config;
    config.temperature = 1.0f;
    config.top_k       = 2;
    config.seed        = 424242;

    constexpr int samples = 128;
    const RunResult baseline =
        run_repeated(column, 2, samples, 8, config, 100, ops::kSamplePurposeDecode);
    const RunResult repeat =
        run_repeated(column, 2, samples, 8, config, 100, ops::kSamplePurposeDecode);
    const RunResult rechunked =
        run_repeated(column, 2, samples, 1, config, 100, ops::kSamplePurposeDecode);
    const RunResult shifted =
        run_repeated(column, 2, samples, 8, config, 101, ops::kSamplePurposeDecode);
    const RunResult other_purpose =
        run_repeated(column, 2, samples, 8, config, 100, ops::kSamplePurposePrefill);
    const RunResult offset =
        run_repeated(column, 2, samples, 8, config, 97, ops::kSamplePurposeDecode, 3);
    ops::SamplingConfig other_seed_config = config;
    ++other_seed_config.seed;
    const RunResult other_seed =
        run_repeated(column, 2, samples, 8, other_seed_config, 100, ops::kSamplePurposeDecode);

    int failures = baseline.integrity_failures + repeat.integrity_failures +
                   rechunked.integrity_failures + shifted.integrity_failures +
                   other_purpose.integrity_failures + other_seed.integrity_failures +
                   offset.integrity_failures;
    failures += verify_exact("sample position offset adds to every row's position", offset.tokens,
                             baseline.tokens);
    failures += verify_exact("sample identical counter key is reproducible", repeat.tokens,
                             baseline.tokens);
    failures += verify_exact("sample RNG does not depend on compact row", rechunked.tokens,
                             baseline.tokens);
    failures += verify_exact("sample position selects the corresponding counter",
                             std::vector<int>(shifted.tokens.begin(), shifted.tokens.end() - 1),
                             std::vector<int>(baseline.tokens.begin() + 1, baseline.tokens.end()));
    if (other_purpose.tokens == baseline.tokens) {
        std::cerr << "sample purpose did not separate the counter stream\n";
        ++failures;
    }
    if (other_seed.tokens == baseline.tokens) {
        std::cerr << "sample seed did not separate the counter stream\n";
        ++failures;
    }
    return failures;
}

int workspace_route_boundary_contract() {
    constexpr int token_domain = 257;
    constexpr int batch        = 8;
    std::vector<float> logits(static_cast<std::size_t>(token_domain) * batch, 0.0f);
    const RunResult result =
        run_homogeneous_batch(logits, token_domain, token_domain, batch, ops::SamplingConfig{}, 0,
                              ops::kSamplePurposeDecode);
    int failures = result.integrity_failures;
    failures +=
        verify_exact("sample workspace route boundary", result.tokens, std::vector<int>(batch, 0));
    return failures;
}

int workspace_route_rng_contract() {
    std::vector<float> column(257, -100.0f);
    for (int i = 0; i < 4; ++i) { column[i] = static_cast<float>(i) * 0.25f; }
    ops::SamplingConfig config;
    config.temperature = 1.0f;
    config.top_k       = 4;
    config.seed        = 424242;
    int failures       = 0;
    for (int width : {15, 16, 17}) {
        const auto single =
            run_repeated(column, 257, width * 2, 1, config, 100, ops::kSamplePurposeDecode);
        const auto wide =
            run_repeated(column, 257, width * 2, width, config, 100, ops::kSamplePurposeDecode);
        const auto offset =
            run_repeated(column, 257, width * 2, width, config, 90, ops::kSamplePurposeDecode, 10);
        failures += single.integrity_failures + wide.integrity_failures + offset.integrity_failures;
        failures +=
            verify_exact("sampler routes preserve counter keys", wide.tokens, single.tokens);
        failures += verify_exact("sampler routes add the position offset", offset.tokens,
                                 single.tokens);
    }
    return failures;
}

int increment_counts_contract() {
    const std::vector<std::int32_t> ids{1, 3, 1, 7};
    const std::vector<std::int32_t> initial{0, 2, 0, 4, 0, 0, 0, 1};
    const std::vector<std::int32_t> expected{0, 4, 0, 5, 0, 0, 0, 2};
    DeviceBuffer device_ids = to_device(ids);
    GuardedDeviceBuffer device_counts(initial.size() * sizeof(std::int32_t));
    device_counts.copy_from_host(initial.data(), initial.size() * sizeof(std::int32_t));
    Tensor token_ids(device_ids.p, DType::I32, {static_cast<std::int32_t>(ids.size())});
    Tensor counts(device_counts.data(), DType::I32, {static_cast<std::int32_t>(initial.size())});
    ops::increment_token_counts(token_ids, counts, nullptr);
    cuda_synchronize();

    int failures =
        verify_exact("increment token counts",
                     from_device<std::int32_t>(device_counts.data(), initial.size()), expected);
    failures += verify_exact("increment token counts read-only ids",
                             from_device<std::int32_t>(device_ids, ids.size()), ids);
    failures += device_counts.verify_guards("increment token counts guards");
    return failures;
}

int masked_sampling_contract() {
    int failures = 0;
    for (int domain : {64, 257, 248077}) {
        const int words = (domain + 31) / 32;
        std::vector<std::uint32_t> mask(words, 0);
        std::vector<float> column(domain, 1000.0f), reference(domain, -INFINITY);
        const std::vector<int> allowed{3, 7, domain - 1};
        for (std::size_t i = 0; i < allowed.size(); ++i) {
            int id = allowed[i];
            mask[id / 32] |= 1U << (id % 32);
            column[id] = reference[id] = 1.0f - static_cast<float>(i);
        }
        auto device_mask = to_device(mask);
        ops::SamplingConfig cfg;
        cfg.token_mask        = static_cast<const std::uint32_t*>(device_mask.p);
        cfg.token_mask_stride = words;
        auto greedy = run_homogeneous_batch(repeat_column(column, 8), domain, domain, 8, cfg, 0,
                                            ops::kSamplePurposeDecode);
        failures += greedy.integrity_failures;
        failures +=
            verify_exact("masked argmax before raw winner", greedy.tokens, std::vector<int>(8, 3));
        cfg.temperature = 0.8f;
        cfg.top_k       = 20;
        cfg.top_p       = 0.95f;
        cfg.min_p       = 0.05f;
        cfg.seed        = 9517;
        auto samples = run_repeated(column, domain, 4096, 8, cfg, 123, ops::kSamplePurposeDecode);
        failures += samples.integrity_failures;
        failures += verify_distribution("masked distribution before filters", samples.tokens,
                                        distribution_oracle(reference, domain, cfg));
        std::fill(mask.begin(), mask.end(), 0);
        mask[(domain - 1) / 32] = 1U << ((domain - 1) % 32);
        auto singleton          = to_device(mask);
        cfg.token_mask          = static_cast<const std::uint32_t*>(singleton.p);
        auto one = run_homogeneous_batch(repeat_column(column, 8), domain, domain, 8, cfg, 42,
                                         ops::kSamplePurposePrefill);
        failures += verify_exact("singleton mask with top-k=20", one.tokens,
                                 std::vector<int>(8, domain - 1));
    }
    return failures;
}

// Two logits 1/512 apart round to one BF16 value, 10.0, whose tie the lower id wins; FP32 logits
// keep the gap, so the larger one wins the greedy draw.
int fp32_near_tie_contract() {
    constexpr int domain = 257;
    std::vector<float> column(domain, -100.0f);
    column[5]        = 10.0f;
    column[9]        = 10.0f + 1.0f / 512.0f;
    const int winner = g_logits == DType::FP32 ? 9 : 5;
    const RunResult result =
        run_homogeneous_batch(repeat_column(column, 8), domain, domain, 8, ops::SamplingConfig{}, 0,
                              ops::kSamplePurposeDecode);
    int failures = result.integrity_failures;
    failures += verify_exact("sample near tie", result.tokens, std::vector<int>(8, winner));
    return failures;
}

// A row whose logits are all NaN writes its config's non_finite_token (kSamplerNonFiniteToken
// unless a drafter names a valid id) and counts nothing, greedy or sampled, on the single-block
// and the workspace routes.
int non_finite_contract() {
    int failures = 0;
    for (const int domain : {256, 248077}) {
        const int physical_rows = domain == 256 ? 256 : 248320;
        constexpr int batch     = 4;
        const std::vector<float> logits(static_cast<std::size_t>(physical_rows) * batch,
                                        std::numeric_limits<float>::quiet_NaN());
        std::vector<ops::SamplingConfig> configs(batch);
        for (int row = 0; row < batch; ++row) {
            auto& config            = configs[static_cast<std::size_t>(row)];
            config.temperature      = row % 2 == 0 ? 0.0f : 0.9f;
            config.seed             = 7 + static_cast<unsigned>(row);
            config.draw             = row == 3 ? ops::kSampleDrawGumbel : ops::kSampleDrawInverseCdf;
            config.non_finite_token = row < 2 ? ops::kSamplerNonFiniteToken : 0;
        }
        const std::vector<std::vector<int>> counts(batch,
                                                   std::vector<int>(static_cast<std::size_t>(domain)));
        const RunResult result = run_batch(logits, physical_rows, domain, configs, {3, 4, 5, 6},
                                           ops::kSamplePurposeDecode, counts);
        failures += result.integrity_failures;
        failures += verify_exact("sample all-NaN rows write their non-finite token", result.tokens,
                                 std::vector<int>{ops::kSamplerNonFiniteToken,
                                                  ops::kSamplerNonFiniteToken, 0, 0});
        for (std::size_t row = 0; row < result.counts.size(); ++row) {
            failures += verify_exact("sample all-NaN rows count nothing", result.counts[row],
                                     counts[row]);
        }
    }
    return failures;
}

int run_contracts() {
    int failures = 0;
    failures += non_finite_contract();
    failures += masked_sampling_contract();
    failures += greedy_contract();
    failures += deterministic_stochastic_contract();
    failures += heterogeneous_batch_contract();
    failures += filtered_distribution_contract();
    failures += gumbel_contract();
    failures += capped_distribution_contract();
    failures += real_shape_distribution_contract();
    failures += rng_key_contract();
    failures += workspace_route_boundary_contract();
    failures += workspace_route_rng_contract();
    failures += fp32_near_tie_contract();
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures            = 0;
    const std::size_t at_16 = ops::sampling_workspace_capacity_bytes(257, 16, 16);
    if (ops::sampling_workspace_capacity_bytes(256, 1, 16) != 0 || at_16 == 0 ||
        ops::sampling_workspace_capacity_bytes(257, 17, 17) != 0 ||
        ops::sampling_workspace_capacity_bytes(257, 1, 17) != at_16) {
        std::cerr << "sampling workspace route boundary contract failed\n";
        ++failures;
    }
    try {
        (void)ops::sampling_workspace_capacity_bytes(257, 0, 16);
        std::cerr << "sampling workspace accepted an invalid lane interval\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    for (const DType dtype : {DType::BF16, DType::FP32}) {
        g_logits                 = dtype;
        const int dtype_failures = run_contracts();
        std::cout << (dtype_failures == 0 ? "OK" : "FAIL") << " sample contracts on "
                  << (dtype == DType::FP32 ? "FP32" : "BF16") << " logits\n";
        failures += dtype_failures;
    }
    failures += increment_counts_contract();

    std::cout << (failures == 0 ? "OK" : "FAIL") << " sample public contract\n";
    return failures == 0 ? 0 : 1;
}
