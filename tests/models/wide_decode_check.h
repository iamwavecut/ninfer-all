#pragma once

// A decode step of more rows than the vector kernels take (an ordinary round of 9..64 requests)
// on a family executor with `sequences` sequences: every row reads only its own sequence and
// token, so a row gives the same bits in a step of 9 rows as in a step of every sequence, in any
// place, and the same step twice gives the same bits. Against the prompt route of the same
// tokens it lies no farther than the vector route (8 rows) does.

#include "core/device_launch.h"
#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <vector>

namespace ninfer::test {

inline int wide_failures = 0;

inline void wide_expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++wide_failures;
    }
}

// Deterministic ordinary token ids.
inline std::vector<std::int32_t> wide_tokens(std::size_t n, std::uint32_t seed) {
    std::vector<std::int32_t> out(n);
    std::uint32_t state = seed;
    for (auto& token : out) {
        state = state * 1664525u + 1013904223u;
        token = std::int32_t(1000 + (state >> 8) % 60000);
    }
    return out;
}

template <typename Executor>
std::vector<std::vector<float>> wide_step(Executor& executor, std::span<const std::uint32_t> rows,
                                          std::uint32_t sequences) {
    for (std::uint32_t s = 0; s < sequences; ++s) {
        executor.reset(s);
        executor.forward(s, wide_tokens(24 + 5 * s, 1 + s), 1);
    }
    std::vector<std::int32_t> next(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) { next[i] = wide_tokens(1, 100 + rows[i])[0]; }
    executor.decode(rows, next);
    const Tensor logits = executor.logits(std::uint32_t(rows.size()));
    const std::size_t vocab = std::size_t(logits.ne[0]);
    std::vector<__nv_bfloat16> raw(vocab * rows.size());
    CUDA_CHECK(cudaStreamSynchronize(executor.head_stream()));
    CUDA_CHECK(cudaMemcpy(raw.data(), logits.data, raw.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<std::vector<float>> out(rows.size(), std::vector<float>(vocab));
    for (std::size_t r = 0; r < rows.size(); ++r) {
        for (std::size_t v = 0; v < vocab; ++v) { out[r][v] = __bfloat162float(raw[r * vocab + v]); }
    }
    return out;
}

inline std::int32_t wide_argmax(const std::vector<float>& row, std::int32_t domain) {
    return std::int32_t(std::max_element(row.begin(), row.begin() + domain) - row.begin());
}

// Returns the number of failed expectations; `sequences` is at least 16.
template <typename Executor>
int wide_decode_check(Executor& executor, std::uint32_t sequences, std::int32_t domain) {
    std::vector<std::uint32_t> all(sequences);
    std::iota(all.begin(), all.end(), 0U);
    const auto wide  = wide_step(executor, all, sequences);
    const auto again = wide_step(executor, all, sequences);
    for (std::uint32_t r = 0; r < sequences; ++r) {
        wide_expect(wide[r] == again[r], "row " + std::to_string(r) + " repeats bit for bit");
    }
    // Nine rows in another order: sequence 7 first, then 0..6 and 15.
    const std::vector<std::uint32_t> nine{7, 0, 1, 2, 3, 4, 5, 6, sequences - 1};
    const auto part = wide_step(executor, nine, sequences);
    for (std::size_t i = 0; i < nine.size(); ++i) {
        wide_expect(part[i] == wide[nine[i]], "sequence " + std::to_string(nine[i]) +
                                                  " reads the same in 9 rows as in " +
                                                  std::to_string(sequences));
    }
    // The vector route (eight rows) and the prompt route (the prompt and its next token as one
    // chunk, a matrix-kernel pass like the wide step's) round differently from the wide step and
    // from each other: their largest differences from the wide rows are printed.
    const std::vector<std::uint32_t> eight{0, 1, 2, 3, 4, 5, 6, 7};
    const auto vector = wide_step(executor, eight, sequences);
    double vector_worst = 0, prompt_worst = 0, between = 0;
    int vector_agree = 0, prompt_agree = 0;
    for (std::uint32_t s = 0; s < 8; ++s) {
        std::vector<std::int32_t> chunk = wide_tokens(24 + 5 * s, 1 + s);
        chunk.push_back(wide_tokens(1, 100 + s)[0]);
        executor.reset(s);
        executor.forward(s, chunk, 1);
        const Tensor logits = executor.logits(1);
        std::vector<__nv_bfloat16> raw(std::size_t(logits.ne[0]));
        CUDA_CHECK(cudaStreamSynchronize(executor.head_stream()));
        CUDA_CHECK(cudaMemcpy(raw.data(), logits.data, raw.size() * 2, cudaMemcpyDeviceToHost));
        std::vector<float> prompt(raw.size());
        for (std::size_t v = 0; v < raw.size(); ++v) { prompt[v] = __bfloat162float(raw[v]); }
        for (std::int32_t v = 0; v < domain; ++v) {
            vector_worst = std::max(vector_worst, double(std::fabs(vector[s][v] - wide[s][v])));
            prompt_worst = std::max(prompt_worst, double(std::fabs(prompt[v] - wide[s][v])));
            between      = std::max(between, double(std::fabs(prompt[v] - vector[s][v])));
        }
        vector_agree += wide_argmax(vector[s], domain) == wide_argmax(wide[s], domain) ? 1 : 0;
        prompt_agree += wide_argmax(prompt, domain) == wide_argmax(wide[s], domain) ? 1 : 0;
    }
    std::cout << "largest logit difference from the wide rows: vector route " << vector_worst
              << " (argmax agrees on " << vector_agree << " of 8), prompt route " << prompt_worst
              << " (" << prompt_agree << " of 8); vector against prompt route " << between << '\n';
    wide_expect(prompt_worst <= between + 0.5,
                "the wide rows lie no farther from the prompt route than the vector route does");
    return wide_failures;
}

} // namespace ninfer::test
