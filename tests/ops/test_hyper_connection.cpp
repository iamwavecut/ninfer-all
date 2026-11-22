// hyper_connection_read/write against an FP64 oracle of the Qwen3.8-Flash-Next and dense Qwen4
// gated residual, at both implemented shapes (4 streams with hidden 2560 and lowrank 320, or hidden
// 5120 and lowrank 640) and decode, verify and prefill widths,
// with and without the inject rows (the final mixer has none), with BF16 and with Q8_0 matrices
// (the oracle reads the values the blocks represent) and with Q8_0 matrices beside a BF16 inject,
// eagerly and under graph replay; and
// hyper_connection_expand, which must widen the embedding into every stream exactly, and its MTP
// form, which adds each stream's own term.
#include "core/arena.h"
#include "core/device_launch.h"
#include "ninfer/ops/hyper_connection.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kStreams = 4;
constexpr float kEps   = 1e-6f;
// The geometry under test (main runs each implemented one).
int kHidden  = 2560;
int kLowrank = 320;
int kWidth   = kStreams * kHidden;

void use_geometry(int hidden, int lowrank) {
    kHidden  = hidden;
    kLowrank = lowrank;
    kWidth   = kStreams * hidden;
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// ggml's Q8_0 of row-major `values` (rows of whole 32-value blocks): a binary16 d = amax / 127 and
// codes round(x / d) per block. `values` becomes what the blocks represent, d * q.
std::vector<std::uint8_t> encode_q8_0(std::vector<float>& values) {
    std::vector<std::uint8_t> out(values.size() / 32 * 34);
    for (std::size_t b = 0; b < values.size() / 32; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) { amax = std::max(amax, std::abs(values[32 * b + i])); }
        const float d        = amax / 127.0f;
        const float inverse  = d != 0.0f ? 1.0f / d : 0.0f;
        const __half stored  = __float2half(d);
        const float decoded  = __half2float(stored);
        std::memcpy(&out[34 * b], &stored, 2);
        for (int i = 0; i < 32; ++i) {
            const auto q    = static_cast<std::int8_t>(std::lround(values[32 * b + i] * inverse));
            out[34 * b + 2 + i] = static_cast<std::uint8_t>(q);
            values[32 * b + i]  = decoded * float(q);
        }
    }
    return out;
}

struct Oracle {
    std::vector<double> mixed;  // [tokens][hidden]
    std::vector<double> inject; // [tokens][streams]
};

Oracle oracle(const std::vector<float>& stack, const std::vector<float>& norm,
              const std::vector<float>& down, const std::vector<float>& up,
              const std::vector<float>* inject, int tokens, bool unit_offset = true) {
    Oracle out{std::vector<double>(static_cast<std::size_t>(tokens) * kHidden),
               std::vector<double>(static_cast<std::size_t>(tokens) * kStreams)};
    std::vector<double> xn(kWidth), low(kLowrank);
    for (int t = 0; t < tokens; ++t) {
        const float* xs = stack.data() + static_cast<std::size_t>(t) * kWidth;
        for (int c = 0; c < kStreams; ++c) {
            double sum = 0.0;
            for (int d = 0; d < kHidden; ++d) sum += double(xs[c * kHidden + d]) * xs[c * kHidden + d];
            const double scale = 1.0 / std::sqrt(sum / kHidden + double(kEps));
            for (int d = 0; d < kHidden; ++d) {
                xn[c * kHidden + d] = xs[c * kHidden + d] * scale *
                                      ((unit_offset ? 1.0 : 0.0) + norm[c * kHidden + d]);
            }
        }
        for (int r = 0; r < kLowrank; ++r) {
            double v = 0.0;
            for (int k = 0; k < kWidth; ++k) v += double(down[static_cast<std::size_t>(r) * kWidth + k]) * xn[k];
            v /= kStreams;
            low[r] = v * sigmoid(v);
        }
        for (int d = 0; d < kHidden; ++d) {
            double mixed = 0.0;
            for (int c = 0; c < kStreams; ++c) {
                double g = 0.0;
                const std::size_t row = static_cast<std::size_t>(c * kHidden + d) * kLowrank;
                for (int k = 0; k < kLowrank; ++k) g += double(up[row + k]) * low[k];
                mixed += sigmoid(g) * xn[c * kHidden + d];
            }
            out.mixed[static_cast<std::size_t>(t) * kHidden + d] = mixed / kStreams;
        }
        if (inject != nullptr) {
            for (int c = 0; c < kStreams; ++c) {
                double v = 0.0;
                for (int k = 0; k < kWidth; ++k) v += double((*inject)[static_cast<std::size_t>(c) * kWidth + k]) * xn[k];
                out.inject[static_cast<std::size_t>(t) * kStreams + c] = 2.0 * sigmoid(v / kStreams);
            }
        }
    }
    return out;
}

// bf16_inject: Q8_0 down and up beside a BF16 inject (a Basalt pack's prompt injection).
int run_case(int tokens, bool with_inject, bool graph, std::uint32_t seed, bool fused = false,
             bool fp32_y = false, bool q8 = false, bool fp32_norm = false,
             bool bf16_inject = false) {
    const std::string label = "hyper_connection H=" + std::to_string(kHidden) +
                              " T=" + std::to_string(tokens) +
                              (with_inject ? "" : " mixer") + (graph ? " graph" : "") +
                              (fused ? (fp32_y ? " write_read FP32" : " write_read BF16") : "") +
                              (q8 ? " Q8_0" : "") + (fp32_norm ? " FP32 gamma" : "") +
                              (bf16_inject ? " BF16 inject" : "");
    std::vector<float> stack(static_cast<std::size_t>(kWidth) * tokens), norm(kWidth),
        down(static_cast<std::size_t>(kLowrank) * kWidth), up(static_cast<std::size_t>(kWidth) * kLowrank),
        inject(static_cast<std::size_t>(kStreams) * kWidth), y(static_cast<std::size_t>(kHidden) * tokens);
    fill_uniform(stack, seed, -2.0f, 2.0f);
    // One stream of every token far larger than the others: the norm is per stream.
    for (int t = 0; t < tokens; ++t) {
        for (int d = 0; d < kHidden; ++d) stack[static_cast<std::size_t>(t) * kWidth + d] *= 64.0f;
    }
    fill_uniform(norm, seed + 1, -0.5f, 0.5f);
    fill_uniform(down, seed + 2, -0.03f, 0.03f);
    fill_uniform(up, seed + 3, -0.1f, 0.1f);
    fill_uniform(inject, seed + 4, -0.03f, 0.03f);
    fill_uniform(y, seed + 5, -4.0f, 4.0f);
    if (fp32_norm) {
        for (auto& value : norm) value += 1.0f;
    } else {
        round_to_bf16(norm);
    }
    std::vector<std::uint8_t> down_q8, up_q8, inject_q8;
    if (q8) {
        down_q8 = encode_q8_0(down);
        up_q8   = encode_q8_0(up);
        if (bf16_inject) {
            round_to_bf16(inject);
        } else {
            inject_q8 = encode_q8_0(inject);
        }
    } else {
        for (auto* v : {&down, &up, &inject}) round_to_bf16(*v);
    }
    if (!fp32_y) { round_to_bf16(y); }
    std::vector<float> input_stack = stack;
    std::vector<float> previous(static_cast<std::size_t>(kStreams) * tokens);
    fill_uniform(previous, seed + 7, -0.5f, 2.0f);
    previous[0] = 0.0f;

    const auto matrix_bytes = [&](const std::vector<float>& values, bool matrix_q8) {
        return matrix_q8 ? values.size() / 32 * 34 : values.size() * 2;
    };
    const bool inject_q8_format = q8 && !bf16_inject;
    GuardedDeviceBuffer d_stack(stack.size() * 4), d_norm(norm.size() * (fp32_norm ? 4 : 2)),
        d_down(matrix_bytes(down, q8)), d_up(matrix_bytes(up, q8)),
        d_inject(matrix_bytes(inject, inject_q8_format)),
        d_mixed(static_cast<std::size_t>(kHidden) * tokens * 2),
        d_weights(static_cast<std::size_t>(kStreams) * tokens * 4),
        d_y(y.size() * (fp32_y ? 4 : 2));
    d_stack.copy_from_host(stack.data(), d_stack.bytes());
    const auto copy_bf16 = [](GuardedDeviceBuffer& buffer, const std::vector<float>& values) {
        const auto bits = encode_bf16(values);
        buffer.copy_from_host(bits.data(), buffer.bytes());
    };
    if (fp32_norm)
        d_norm.copy_from_host(norm.data(), d_norm.bytes());
    else
        copy_bf16(d_norm, norm);
    if (q8) {
        d_down.copy_from_host(down_q8.data(), d_down.bytes());
        d_up.copy_from_host(up_q8.data(), d_up.bytes());
        if (inject_q8_format) {
            d_inject.copy_from_host(inject_q8.data(), d_inject.bytes());
        } else {
            copy_bf16(d_inject, inject);
        }
    } else {
        copy_bf16(d_down, down);
        copy_bf16(d_up, up);
        copy_bf16(d_inject, inject);
    }
    if (fp32_y) { d_y.copy_from_host(y.data(), d_y.bytes()); }
    else { copy_bf16(d_y, y); }
    if (fused) { d_weights.copy_from_host(previous.data(), d_weights.bytes()); }

    Tensor t_stack(d_stack.data(), DType::FP32, {kHidden, kStreams, tokens});
    Tensor t_norm(d_norm.data(), fp32_norm ? DType::FP32 : DType::BF16, {kWidth});
    Tensor t_mixed(d_mixed.data(), DType::BF16, {kHidden, tokens});
    Tensor t_weights(d_weights.data(), DType::FP32, {kStreams, tokens});
    Tensor t_y(d_y.data(), fp32_y ? DType::FP32 : DType::BF16, {kHidden, tokens});
    const auto format = q8 ? ops::HyperConnectionMatrix::Format::Q8_0
                           : ops::HyperConnectionMatrix::Format::BF16;
    const ops::HyperConnectionWeights weights{&t_norm,
                                              {d_down.data(), format},
                                              {d_up.data(), format},
                                              {with_inject ? d_inject.data() : nullptr,
                                               inject_q8_format
                                                   ? ops::HyperConnectionMatrix::Format::Q8_0
                                                   : ops::HyperConnectionMatrix::Format::BF16},
                                              !fp32_norm};
    WorkspaceArena workspace(
        ops::hyper_connection_read_workspace_bytes(kStreams, kHidden, kLowrank, tokens));

    const auto run = [&](cudaStream_t stream) {
        if (fused) {
            // Reuse the previous inject plane as the next read's output, as the Engine does.
            ops::hyper_connection_write_read(t_stack, t_y, t_weights, weights, kEps, workspace,
                                             t_mixed, with_inject ? &t_weights : nullptr, stream);
        } else {
            ops::hyper_connection_read(t_stack, weights, kEps, workspace, t_mixed,
                                       with_inject ? &t_weights : nullptr, stream);
        }
    };
    if (graph) {
        cudaStream_t stream;
        cudaGraph_t captured;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        run(stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
        CUDA_CHECK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        for (int replay = 0; replay < 2; ++replay) {
            if (fused) {
                // Change graph inputs between replays; a captured old value must not survive.
                if (replay == 1) {
                    for (auto& value : input_stack) { value *= -0.5f; }
                    for (auto& value : previous) { value += 0.125f; }
                }
                CUDA_CHECK(cudaMemcpyAsync(d_stack.data(), input_stack.data(), d_stack.bytes(),
                                            cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(d_weights.data(), previous.data(), d_weights.bytes(),
                                            cudaMemcpyHostToDevice, stream));
            }
            CUDA_CHECK(cudaMemsetAsync(d_mixed.data(), 0xff, d_mixed.bytes(), stream));
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(captured));
        CUDA_CHECK(cudaStreamDestroy(stream));
    } else {
        run(nullptr);
    }
    cuda_synchronize();

    int failures = 0;
    if (fused) {
        // The write's FP32 store is observable. Evaluate it independently in FP64, round once,
        // then evaluate the whole read in FP64 from those represented stack values.
        std::vector<double> written(stack.size());
        for (std::size_t i = 0; i < stack.size(); ++i) {
            const std::size_t column = i / kHidden, t = column / kStreams;
            written[i] = double(input_stack[i]) + double(y[t * kHidden + i % kHidden]) * previous[column];
            stack[i] = static_cast<float>(written[i]);
        }
        const auto got = from_device<float>(d_stack.data(), stack.size());
        failures += verify_pointwise(label + " written stack",
                                     std::vector<double>(got.begin(), got.end()), written,
                                     {1.0e-6, 1.2e-7});
        if (!with_inject) {
            failures += verify_exact((label + " preserved previous inject").c_str(),
                                      from_device<float>(d_weights.data(), previous.size()), previous);
        }
    }
    const Oracle expected =
        oracle(stack, norm, down, up, with_inject ? &inject : nullptr, tokens, !fp32_norm);
    failures += verify_reduction(label + " mixed",
                                    from_device_bf16(d_mixed.data(), expected.mixed.size()),
                                    expected.mixed, {4.0e-3, 1.0e-6, 2.0 * 3.90625e-3});
    if (with_inject) {
        const auto got = from_device<float>(d_weights.data(), expected.inject.size());
        failures += verify_pointwise(label + " inject",
                                     std::vector<double>(got.begin(), got.end()), expected.inject,
                                     {2.0e-6, 2.0e-5});
        // The write: stack += y (x) inject, against the oracle's inject weights.
        ops::hyper_connection_write(t_stack, t_y, t_weights, nullptr);
        cuda_synchronize();
        std::vector<double> written(stack.size());
        for (int t = 0; t < tokens; ++t) {
            for (int c = 0; c < kStreams; ++c) {
                for (int d = 0; d < kHidden; ++d) {
                    const std::size_t i = (static_cast<std::size_t>(t) * kStreams + c) * kHidden + d;
                    written[i] = double(stack[i]) + double(y[static_cast<std::size_t>(t) * kHidden + d]) *
                                                        expected.inject[static_cast<std::size_t>(t) * kStreams + c];
                }
            }
        }
        const auto got_stack = from_device<float>(d_stack.data(), stack.size());
        failures += verify_pointwise(label + " write",
                                     std::vector<double>(got_stack.begin(), got_stack.end()),
                                     written, {1.0e-4, 2.0e-5});
        // And an FP32 block output (the MoE's) on top of it.
        std::vector<float> y32(y.size());
        fill_uniform(y32, seed + 6, -4.0f, 4.0f);
        GuardedDeviceBuffer d_y32(y32.size() * 4);
        d_y32.copy_from_host(y32.data(), d_y32.bytes());
        Tensor t_y32(d_y32.data(), DType::FP32, {kHidden, tokens});
        ops::hyper_connection_write(t_stack, t_y32, t_weights, nullptr);
        cuda_synchronize();
        for (std::size_t i = 0; i < written.size(); ++i) {
            const std::size_t column = i / kHidden, t = column / kStreams;
            written[i] = double(got_stack[i]) +
                         double(y32[t * kHidden + i % kHidden]) * expected.inject[column];
        }
        const auto got_fp32 = from_device<float>(d_stack.data(), stack.size());
        failures += verify_pointwise(label + " FP32 write",
                                     std::vector<double>(got_fp32.begin(), got_fp32.end()),
                                     written, {1.0e-4, 2.0e-5});
        failures += d_y32.verify_guards(label.c_str());
    }
    for (auto* buffer : {&d_stack, &d_norm, &d_down, &d_up, &d_inject, &d_mixed, &d_weights, &d_y}) {
        failures += buffer->verify_guards(label.c_str());
    }
    if (fused && tokens == 1) {
        for (int invalid = 0; invalid < 2; ++invalid) {
            Tensor bad_y = t_y, bad_inject = t_weights;
            if (invalid == 0) { bad_y.ne[1] = tokens + 1; }
            else { bad_inject.ne[2] = 2; }
            bool refused = false;
            try {
                ops::hyper_connection_write_read(t_stack, bad_y, bad_inject, weights, kEps,
                                                 workspace, t_mixed, &t_weights, nullptr);
            } catch (const std::invalid_argument&) { refused = true; }
            failures += refused ? 0 : 1;
        }
    }
    return failures;
}

std::vector<float> bf16_values(std::size_t count, std::uint32_t seed, float scale) {
    std::vector<float> out(count);
    std::uint32_t state = seed;
    for (auto& v : out) {
        state = state * 1664525u + 1013904223u;
        v = bf16_to_f32(f32_to_bf16(static_cast<float>(static_cast<std::int32_t>(state)) * scale));
    }
    return out;
}

// Without `streams` every stream must be the embedding exactly; with them (the MTP layer's start)
// each stream is its own term plus the shared one, against the FP64 sum.
int run_expand(int tokens, bool with_streams, std::uint32_t seed) {
    const auto x       = bf16_values(static_cast<std::size_t>(kHidden) * tokens, seed, 1e-9f);
    const auto streams = bf16_values(static_cast<std::size_t>(kWidth) * tokens, seed + 1, 3e-9f);
    const auto bits    = encode_bf16(x);
    const auto stream_bits = encode_bf16(streams);
    GuardedDeviceBuffer d_x(bits.size() * 2), d_streams(stream_bits.size() * 2),
        d_stack(x.size() * kStreams * 4);
    d_x.copy_from_host(bits.data(), bits.size() * 2);
    d_streams.copy_from_host(stream_bits.data(), stream_bits.size() * 2);
    Tensor t_x(d_x.data(), DType::BF16, {kHidden, tokens});
    Tensor t_streams(d_streams.data(), DType::BF16, {kHidden, kStreams, tokens});
    Tensor t_stack(d_stack.data(), DType::FP32, {kHidden, kStreams, tokens});
    if (with_streams) {
        ops::hyper_connection_expand(t_x, t_streams, t_stack, nullptr);
    } else {
        ops::hyper_connection_expand(t_x, t_stack, nullptr);
    }
    cuda_synchronize();
    const auto got = from_device<float>(d_stack.data(), x.size() * kStreams);
    std::vector<double> want(got.size());
    for (int t = 0; t < tokens; ++t)
        for (int c = 0; c < kStreams; ++c)
            for (int d = 0; d < kHidden; ++d) {
                const std::size_t i = (static_cast<std::size_t>(t) * kStreams + c) * kHidden + d;
                want[i]             = double(x[static_cast<std::size_t>(t) * kHidden + d]) +
                                      (with_streams ? double(streams[i]) : 0.0);
            }
    const std::string label = std::string(with_streams ? "expand+streams" : "expand") +
                              " H=" + std::to_string(kHidden) + " T=" + std::to_string(tokens);
    int failures = 0;
    if (with_streams) {
        // One FP32 rounding of an exact sum of two widened BF16 values.
        for (std::size_t i = 0; i < got.size(); ++i) {
            const double error = std::abs(double(got[i]) - want[i]);
            if (error > std::abs(want[i]) * 6.0e-8 + 1e-30) {
                std::cerr << label << ": element " << i << " got " << got[i] << " want " << want[i]
                          << "\n";
                ++failures;
                break;
            }
        }
    } else {
        failures += verify_exact(label.c_str(), got, std::vector<float>(want.begin(), want.end()));
    }
    failures += d_x.verify_guards("expand input");
    failures += d_streams.verify_guards("expand streams");
    failures += d_stack.verify_guards("expand stack");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    use_geometry(2560, 320);
    // Every narrow width has its own kernels (1..8), then the wide path.
    for (const int tokens : {1, 2, 3, 6, 7, 8, 9, 16, 37}) {
        failures += run_case(tokens, true, false, 4100u + tokens);
    }
    failures += run_case(1, false, false, 4200u);
    failures += run_case(5, false, false, 4201u);
    failures += run_case(4, true, true, 4300u);
    // Fused write/read: both block-output dtypes, every narrow specialization, the GEMM seam,
    // a prefill interior, the final mixer, and graph replays with changed stack/inject inputs.
    for (const int tokens : {1, 2, 3, 4, 5, 6, 7, 8, 9, 37}) {
        failures += run_case(tokens, true, false, 4500u + tokens, true, tokens % 2 == 0);
    }
    failures += run_case(1, true, false, 4550u, true, true);
    failures += run_case(8, true, false, 4551u, true, false);
    failures += run_case(9, true, true, 4552u, true, true);
    failures += run_case(5, false, false, 4553u, true, false);
    failures += run_case(4, true, true, 4554u, true, false);
    // Q8_0 matrices: the narrow kernels, the wide path's BF16 operands and FP32 inject rows,
    // the final mixer, the fused write/read and a graph replay.
    for (const int tokens : {1, 3, 8, 9, 37, 300}) {
        failures += run_case(tokens, true, false, 4600u + tokens, false, false, true);
    }
    failures += run_case(1, false, false, 4650u, false, false, true);
    failures += run_case(40, false, false, 4655u, false, false, true);
    failures += run_case(2, true, false, 4651u, true, false, true);
    failures += run_case(9, true, false, 4652u, true, true, true);
    failures += run_case(4, true, true, 4653u, true, false, true);
    failures += run_case(1, true, false, 4660u, false, false, true, true);
    failures += run_case(9, true, true, 4661u, true, true, true, true);
    // Q8_0 down and up beside a BF16 inject: the narrow widths, the GEMM seam, a prefill
    // interior and the fused write/read.
    for (const int tokens : {1, 4, 8, 9, 300}) {
        failures += run_case(tokens, true, false, 4670u + tokens, false, false, true, false, true);
    }
    failures += run_case(3, true, false, 4690u, true, true, true, false, true);
    failures += run_case(5, true, true, 4691u, true, false, true, true, true);
    failures += run_expand(1, false, 4400u);
    failures += run_expand(7, false, 4401u);
    failures += run_expand(1, true, 4402u);
    failures += run_expand(9, true, 4403u);
    // Hidden 5120 with lowrank 640 (dense Qwen4): every narrow width, the GEMM seam, a prefill
    // interior, the fused write/read with both block-output dtypes, the final mixer, Q8_0 matrices
    // beside BF16 and Q8_0 injects, graph replays and the expansions.
    use_geometry(5120, 640);
    for (const int tokens : {1, 2, 3, 4, 5, 6, 7, 8, 9, 37}) {
        failures += run_case(tokens, true, false, 5100u + tokens);
        failures += run_case(tokens, true, false, 5200u + tokens, true, tokens % 2 == 1);
    }
    failures += run_case(1, false, false, 5300u);
    failures += run_case(4, true, true, 5301u);
    failures += run_case(9, true, true, 5302u, true, true);
    failures += run_case(3, false, false, 5303u, true, false);
    for (const int tokens : {1, 3, 8, 9, 37, 96}) {
        failures += run_case(tokens, true, false, 5400u + tokens, false, false, true);
        failures += run_case(tokens, true, false, 5500u + tokens, false, false, true, false, true);
    }
    failures += run_case(1, false, false, 5600u, false, false, true);
    failures += run_case(2, true, false, 5601u, true, false, true);
    failures += run_case(9, true, true, 5602u, true, true, true, true);
    failures += run_case(5, true, true, 5603u, true, false, true, true, true);
    failures += run_expand(1, false, 5700u);
    failures += run_expand(9, true, 5701u);
    // Refusals: an unsupported geometry and an inject output without inject rows.
    bool refused = false;
    try {
        (void)ops::hyper_connection_read_workspace_bytes(4, 5120, 320, 1);
    } catch (const std::invalid_argument&) { refused = true; }
    failures += refused ? 0 : 1;
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " hyper_connection\n";
    return failures == 0 ? 0 : 1;
}
