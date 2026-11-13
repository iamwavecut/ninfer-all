// The Ops a GGUF DFlash2 drafter adds beside the fused routes (docs/maintainer/dflash.md): the
// hyper-connection streams' mean, one side of a projected dynamic grouped convolution, the dense
// logits' top 16 and the candidate selector over Q4_0 codebooks with its confidence output. Each
// against an FP64 (or exact) oracle from the represented inputs.
#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ninfer/ops/hyper_connection.h"

#include "core/device_buffer.h"
#include "core/qtype.h"
#include "core/weight.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr PointwiseCriterion kBf16Store{1.0e-6, 4.0e-3};
constexpr PointwiseCriterion kProbability{2.0e-6, 2.0e-5};

// --- hyper_connection_stream_mean -------------------------------------------------------------

int run_stream_mean(int tokens, std::uint32_t seed) {
    constexpr int kHidden = 2560, kStreams = 4, kTaps = 5;
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 3.0F);
    std::vector<float> stack(std::size_t(kHidden) * kStreams * tokens);
    for (auto& v : stack) { v = normal(rng); }
    DeviceBuffer dstack = to_device(stack);
    // A feature matrix [5 x 2560, tokens]: the mean goes to rows [2 x 2560, 3 x 2560).
    const std::size_t features = std::size_t(kTaps) * kHidden * tokens;
    DeviceBuffer dout(features * 2);
    dout.fill(0xAB);
    Tensor in(dstack.p, DType::FP32, {kHidden, kStreams, tokens});
    Tensor out(static_cast<std::uint16_t*>(dout.p) + 2 * kHidden, DType::BF16, {kHidden, tokens});
    out.nb[1] = std::int64_t(kTaps) * kHidden * 2;
    out.nb[2] = out.nb[1] * tokens;
    out.nb[3] = out.nb[2];
    ops::hyper_connection_stream_mean(in, out, nullptr);
    cuda_synchronize();
    const auto got = from_device<std::uint16_t>(dout, features);
    std::vector<double> actual, expected;
    int failures = 0;
    for (int t = 0; t < tokens; ++t) {
        for (int row = 0; row < kTaps * kHidden; ++row) {
            const std::uint16_t value = got[std::size_t(t) * kTaps * kHidden + row];
            if (row < 2 * kHidden || row >= 3 * kHidden) {
                if (value != 0xABAB) { ++failures; }
                continue;
            }
            const int d = row - 2 * kHidden;
            double sum  = 0;
            for (int c = 0; c < kStreams; ++c) {
                sum += stack[(std::size_t(t) * kStreams + c) * kHidden + d];
            }
            expected.push_back(sum / kStreams);
            actual.push_back(bf16_to_f32(value));
        }
    }
    if (failures != 0) {
        std::cerr << "stream_mean T=" << tokens << ": wrote outside its rows\n";
        failures = 1;
    }
    return failures + verify_pointwise("stream_mean T=" + std::to_string(tokens), actual, expected,
                                       kBf16Store);
}

// --- dynamic_grouped_conv ----------------------------------------------------------------------

int run_conv(int width, int batch, bool fp32_x, bool fp32_coefficients, int side,
             std::uint32_t seed) {
    constexpr int kHidden = 2560, kGroups = kHidden / 16;
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    const std::size_t cells = std::size_t(kHidden) * width * batch;
    const std::size_t rows  = std::size_t(4) * kGroups * width * batch;
    std::vector<float> x(cells), coefficients(rows), base(std::size_t(kHidden) * 4), out0(cells);
    for (auto& v : x) { v = normal(rng); }
    for (auto& v : coefficients) { v = 0.25F * normal(rng); }
    for (auto& v : base) { v = 0.5F * normal(rng); }
    for (auto& v : out0) { v = normal(rng); }
    round_to_bf16(base);
    round_to_bf16(out0);
    if (!fp32_x) { round_to_bf16(x); }
    if (!fp32_coefficients) { round_to_bf16(coefficients); }
    DeviceBuffer dx    = fp32_x ? to_device(x) : to_device_bf16(x);
    DeviceBuffer dc    = fp32_coefficients ? to_device(coefficients) : to_device_bf16(coefficients);
    DeviceBuffer dbase = to_device_bf16(base);
    DeviceBuffer dout  = to_device_bf16(out0);
    Tensor tx(dx.p, fp32_x ? DType::FP32 : DType::BF16, {kHidden, width, batch});
    Tensor tc(dc.p, fp32_coefficients ? DType::FP32 : DType::BF16, {4 * kGroups, width, batch});
    Tensor tb(dbase.p, DType::BF16, {kHidden, 2, 2});
    Tensor to(dout.p, DType::BF16, {kHidden, width, batch});
    ops::dynamic_grouped_conv(tx, tc, tb, side, to, nullptr);
    cuda_synchronize();
    std::vector<double> expected(cells);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < width; ++i) {
            const std::size_t column = std::size_t(b) * width + i;
            for (int h = 0; h < kHidden; ++h) {
                const int g = h / 16;
                const auto coefficient = [&](int tap) {
                    return double(coefficients[column * 4 * kGroups + (2 * side + tap) * kGroups + g]);
                };
                double y = (double(base[(side * 2 + 0) * kHidden + h]) + coefficient(0)) *
                           double(x[column * kHidden + h]);
                if (i > 0) {
                    y += (double(base[(side * 2 + 1) * kHidden + h]) + coefficient(1)) *
                         double(x[(column - 1) * kHidden + h]);
                }
                expected[column * kHidden + h] = side == 1 ? double(out0[column * kHidden + h]) + y : y;
            }
        }
    }
    const std::string label = "dynamic_grouped_conv W=" + std::to_string(width) +
                              " B=" + std::to_string(batch) + " side " + std::to_string(side) +
                              (fp32_x ? " x32" : " x16") + (fp32_coefficients ? " c32" : " c16");
    return verify_pointwise(label, from_device_bf16(dout, cells), expected, kBf16Store);
}

// --- candidate_topk ----------------------------------------------------------------------------

int run_topk(int rows, int valid, int columns, bool fp32, std::uint32_t seed, bool nan_column) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 4.0F);
    std::vector<float> logits(std::size_t(rows) * columns);
    for (auto& v : logits) { v = normal(rng); }
    if (!fp32) { round_to_bf16(logits); }
    for (int u = 0; u < columns; ++u) {
        float* column = logits.data() + std::size_t(u) * rows;
        // Ties around the top, a NaN and -inf, and a large value past the valid rows.
        const float top = *std::max_element(column, column + valid);
        for (int k = 0; k < 6; ++k) { column[(u * 7919 + k * 104729) % valid] = top; }
        column[(u * 31 + 17) % valid] = std::numeric_limits<float>::quiet_NaN();
        column[(u * 61 + 5) % valid]  = -std::numeric_limits<float>::infinity();
        if (valid < rows) { column[valid] = top + 100.0F; }
        if (nan_column && u == columns - 1) {
            for (int v = 0; v < rows; ++v) { column[v] = std::numeric_limits<float>::quiet_NaN(); }
        }
    }
    DeviceBuffer dl = fp32 ? to_device(logits) : to_device_bf16(logits);
    GuardedDeviceBuffer ids(std::size_t(columns) * 16 * 4), scores(std::size_t(columns) * 16 * 4);
    const std::size_t capacity = ops::candidate_topk_workspace_capacity_bytes(columns);
    GuardedDeviceBuffer scratch(capacity);
    WorkspaceArena workspace(DeviceSpan{scratch.data(), scratch.bytes()});
    Tensor tl(dl.p, fp32 ? DType::FP32 : DType::BF16, {rows, columns});
    Tensor ti(ids.data(), DType::I32, {16, columns});
    Tensor ts(scores.data(), DType::FP32, {16, columns});
    ops::candidate_topk(tl, valid, ti, ts, workspace, nullptr);
    cuda_synchronize();
    std::vector<std::int32_t> expected_ids;
    std::vector<float> expected_scores;
    for (int u = 0; u < columns; ++u) {
        std::vector<int> order;
        for (int v = 0; v < valid; ++v) {
            if (!std::isnan(logits[std::size_t(u) * rows + v])) { order.push_back(v); }
        }
        const auto value = [&](int v) { return logits[std::size_t(u) * rows + v]; };
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return value(a) > value(b) || (value(a) == value(b) && a < b);
        });
        for (int k = 0; k < 16; ++k) {
            if (k < int(order.size())) {
                expected_ids.push_back(order[k]);
                expected_scores.push_back(value(order[k]));
            } else {
                expected_ids.push_back(0);
                expected_scores.push_back(-std::numeric_limits<float>::infinity());
            }
        }
    }
    const std::string label = "candidate_topk rows " + std::to_string(rows) + " U=" +
                              std::to_string(columns) + (fp32 ? " fp32" : " bf16");
    int failures = verify_exact((label + " ids").c_str(),
                                from_device<std::int32_t>(ids.data(), expected_ids.size()),
                                expected_ids);
    failures += verify_exact((label + " scores").c_str(),
                             from_device<float>(scores.data(), expected_scores.size()),
                             expected_scores);
    failures += ids.verify_guards(label + " ids") + scores.verify_guards(label + " scores") +
                scratch.verify_guards(label + " workspace");
    return failures;
}

// --- candidate_selector_path over Q4_0 codebooks -----------------------------------------------

constexpr int kCandidates = 16, kRank = 256, kCodebookRows = 248320, kRowBytes = 144;

// A codebook row's value r from its stored blocks (the contract's decoding).
float q4_value(const std::uint8_t* row, int r) {
    const std::uint8_t* block = row + (r / 32) * 18;
    std::uint16_t bits;
    std::memcpy(&bits, block, 2);
    __half_raw raw;
    raw.x         = bits;
    const float d = __half2float(__half(raw));
    const int j   = r % 32;
    const int q   = j < 16 ? (block[2 + j] & 0x0F) : (block[2 + j - 16] >> 4);
    return d * float(q - 8);
}

void q4_row(std::mt19937& rng, std::uint8_t* row) {
    std::uniform_int_distribution<int> code(0, 15), scale(8, 40);
    for (int block = 0; block < kRank / 32; ++block) {
        std::uint8_t* p             = row + block * 18;
        const __half d              = __float2half(float(scale(rng)) / 256.0F);
        const std::uint16_t d_bits = __half_raw(d).x;
        std::memcpy(p, &d_bits, 2);
        for (int j = 0; j < 16; ++j) {
            p[2 + j] = std::uint8_t(code(rng) | (code(rng) << 4));
        }
    }
}

int run_selector_q4(int steps, int batch, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<std::uint8_t> predecessor(std::size_t(kCodebookRows) * kRowBytes, 0);
    std::vector<std::uint8_t> successor(predecessor.size(), 0);
    std::vector<std::int32_t> ids(std::size_t(kCandidates) * steps * batch);
    std::vector<std::int32_t> anchors(batch), positions(batch);
    std::set<std::int32_t> tokens;
    for (int b = 0; b < batch; ++b) {
        anchors[b]   = 200000 + b * 37;
        positions[b] = 500 + b * 11;
        tokens.insert(anchors[b]);
        for (int i = 0; i < steps; ++i) {
            for (int c = 0; c < kCandidates; ++c) {
                const std::int32_t token = 1000 + (b * steps + i) * 101 + c * 3;
                ids[(std::size_t(b) * steps + i) * kCandidates + c] = token;
                tokens.insert(token);
            }
        }
    }
    for (const std::int32_t token : tokens) {
        q4_row(rng, predecessor.data() + std::size_t(token) * kRowBytes);
        q4_row(rng, successor.data() + std::size_t(token) * kRowBytes);
    }
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> unary(ids.size());
    for (auto& v : unary) { v = 2.0F * normal(rng); }
    std::vector<std::uint16_t> hidden(std::size_t(kRank) * steps * batch);
    for (auto& v : hidden) { v = f32_to_bf16(0.05F * normal(rng)); }
    std::vector<ops::SamplingConfig> configs(batch);
    for (int b = 0; b < batch; ++b) {
        configs[b].temperature = b % 2 == 0 ? 0.0F : 0.8F;
        configs[b].seed        = 77 + b;
    }
    DeviceBuffer dpre = to_device(predecessor), dsuc = to_device(successor);
    DeviceBuffer dids = to_device(ids), dunary = to_device(unary), dhidden = to_device(hidden);
    DeviceBuffer danchors = to_device(anchors), dpositions = to_device(positions);
    DeviceBuffer dconfigs = to_device(configs);
    GuardedDeviceBuffer drafts(std::size_t(steps) * batch * 4);
    GuardedDeviceBuffer q(std::size_t(kCandidates) * steps * batch * 4);
    GuardedDeviceBuffer confidence(std::size_t(steps) * batch * 4);
    const auto capacity =
        ops::candidate_selector_path_workspace_capacity_bytes(steps, steps, batch, batch);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 1));
    WorkspaceArena workspace(DeviceSpan{scratch.data(), scratch.bytes()});
    const auto codebook = [&](const DeviceBuffer& data) {
        Weight w;
        w.qtype         = QType::GGUF_Q4_0;
        w.n             = kCodebookRows;
        w.k             = kRank;
        w.qdata         = data.p;
        w.payload_bytes = data.bytes;
        return w;
    };
    const Weight wpre = codebook(dpre), wsuc = codebook(dsuc);
    Tensor tids(dids.p, DType::I32, {kCandidates, steps, batch});
    Tensor tunary(dunary.p, DType::FP32, {kCandidates, steps, batch});
    Tensor thidden(dhidden.p, DType::BF16, {kRank, steps, batch});
    Tensor tanchors(danchors.p, DType::I32, {batch});
    Tensor tpositions(dpositions.p, DType::I32, {batch});
    Tensor tdrafts(drafts.data(), DType::I32, {steps, batch});
    Tensor tq(q.data(), DType::FP32, {kCandidates, steps, batch});
    Tensor tconfidence(confidence.data(), DType::FP32, {steps, batch});
    ops::candidate_selector_path(tids, tunary, thidden, tanchors, wpre, wsuc, tpositions,
                                 static_cast<const ops::SamplingConfig*>(dconfigs.p), tdrafts, tq,
                                 workspace, nullptr, &tconfidence);
    cuda_synchronize();
    const auto got_drafts     = from_device<std::int32_t>(drafts.data(), std::size_t(steps) * batch);
    const auto got_q          = from_device<float>(q.data(), std::size_t(kCandidates) * steps * batch);
    const auto got_confidence = from_device<float>(confidence.data(), std::size_t(steps) * batch);
    // The oracle follows the Op's own draws: each step's edges from the predecessor it drew.
    std::vector<double> want_q, want_confidence, have_q, have_confidence;
    int failures = 0;
    for (int b = 0; b < batch; ++b) {
        std::int32_t previous = anchors[b];
        for (int i = 0; i < steps; ++i) {
            const std::size_t column = std::size_t(b) * steps + i;
            std::vector<double> edge(kCandidates);
            for (int c = 0; c < kCandidates; ++c) {
                const std::int32_t token = ids[column * kCandidates + c];
                double sum               = unary[column * kCandidates + c];
                for (int r = 0; r < kRank; ++r) {
                    sum += double(q4_value(predecessor.data() + std::size_t(previous) * kRowBytes, r)) *
                           double(bf16_to_f32(hidden[column * kRank + r])) *
                           double(q4_value(successor.data() + std::size_t(token) * kRowBytes, r));
                }
                edge[c] = sum;
            }
            const std::int32_t drawn = got_drafts[column];
            int selected             = -1;
            for (int c = 0; c < kCandidates; ++c) {
                if (ids[column * kCandidates + c] == drawn) { selected = c; }
            }
            if (selected < 0) {
                std::cerr << "selector q4: draft " << drawn << " is no candidate\n";
                return failures + 1;
            }
            const double maximum = *std::max_element(edge.begin(), edge.end());
            if (configs[b].temperature <= 0) {
                const int best = int(std::max_element(edge.begin(), edge.end()) - edge.begin());
                std::vector<double> sorted = edge;
                std::sort(sorted.rbegin(), sorted.rend());
                if (sorted[0] - sorted[1] > 1.0e-3 && best != selected) {
                    std::cerr << "selector q4: greedy row " << b << " step " << i
                              << " drew rank " << selected << ", expected " << best << '\n';
                    ++failures;
                }
                for (int c = 0; c < kCandidates; ++c) {
                    if (got_q[column * kCandidates + c] != (c == selected ? 1.0F : 0.0F)) {
                        ++failures;
                    }
                }
            } else {
                double total = 0;
                for (int c = 0; c < kCandidates; ++c) {
                    total += std::exp((edge[c] - maximum) / configs[b].temperature);
                }
                for (int c = 0; c < kCandidates; ++c) {
                    want_q.push_back(std::exp((edge[c] - maximum) / configs[b].temperature) / total);
                    have_q.push_back(got_q[column * kCandidates + c]);
                }
                if (got_q[column * kCandidates + selected] <= 0) { ++failures; }
            }
            double total = 0;
            for (int c = 0; c < kCandidates; ++c) { total += std::exp(edge[c] - maximum); }
            want_confidence.push_back(std::exp(edge[selected] - maximum) / total);
            have_confidence.push_back(got_confidence[column]);
            previous = drawn;
        }
    }
    const std::string label =
        "selector q4 K=" + std::to_string(steps) + " B=" + std::to_string(batch);
    if (!want_q.empty()) { failures += verify_pointwise(label + " q", have_q, want_q, kProbability); }
    failures += verify_pointwise(label + " confidence", have_confidence, want_confidence,
                                 kProbability);
    failures += drafts.verify_guards(label + " drafts") + q.verify_guards(label + " q") +
                confidence.verify_guards(label + " confidence") +
                scratch.verify_guards(label + " workspace");
    return failures;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        int failures = 0;
        for (const int tokens : {1, 7, 37}) { failures += run_stream_mean(tokens, 100U + tokens); }
        std::uint32_t seed = 300;
        for (const int width : {1, 8, 16}) {
            for (const int batch : {1, 3, 8}) {
                for (const int side : {0, 1}) {
                    failures += run_conv(width, batch, side == 0, side == 1, side, seed++);
                    failures += run_conv(width, batch, side == 1, side == 0, side, seed++);
                }
            }
        }
        failures += run_topk(248320, 248077, 1, false, 500U, false);
        failures += run_topk(248320, 248077, 7, true, 501U, false);
        failures += run_topk(248320, 248077, 56, false, 502U, true);
        failures += run_topk(16, 16, 3, true, 503U, true);
        for (const int steps : {3, 7}) {
            for (const int batch : {1, 2, 8}) {
                failures += run_selector_q4(steps, batch, 700U + steps * 10 + batch);
            }
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " dflash2 ops\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "dflash2 ops test: " << error.what() << '\n';
        return 1;
    }
}
