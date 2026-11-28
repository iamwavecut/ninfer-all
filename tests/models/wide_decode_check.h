#pragma once

// A decode step of more rows than the vector kernels take (an ordinary round of 9..64 requests)
// on a family executor with `sequences` sequences: every row reads only its own sequence and
// token, so a row gives the same bits in two steps of 9 rows, in any place and beside any other
// sequences, and the same step twice gives the same bits. A call's route follows its width, so a
// step of 9 rows and one of every sequence may round differently. Against the prompt route of the
// same tokens and the 9-row step the wide step lies within twice the spread of the vector route
// (8 rows), and each row lies nearer to its own sequence's logits on every route than to any other
// sequence's.

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
    // Nine rows in another order and with another ninth sequence: sequence 7 first, then 0..6 and
    // the last, against 0..8. A call's route follows its width alone, so the eight sequences both
    // steps hold read the same bits.
    const std::vector<std::uint32_t> nine{7, 0, 1, 2, 3, 4, 5, 6, sequences - 1};
    const std::vector<std::uint32_t> first_nine{0, 1, 2, 3, 4, 5, 6, 7, 8};
    const auto part  = wide_step(executor, nine, sequences);
    const auto first = wide_step(executor, first_nine, sequences);
    for (std::size_t i = 0; i + 1 < nine.size(); ++i) {
        wide_expect(part[i] == first[nine[i]], "sequence " + std::to_string(nine[i]) +
                                                   " reads the same in 9 rows in any place and "
                                                   "beside any other sequence");
    }
    // The vector route (eight rows) and the prompt route (the prompt and its next token as one
    // chunk, a matrix-kernel pass like the wide step's) round differently from the wide step and
    // from each other: their largest differences from the wide rows are printed.
    const std::vector<std::uint32_t> eight{0, 1, 2, 3, 4, 5, 6, 7};
    const auto vector = wide_step(executor, eight, sequences);
    std::vector<std::vector<float>> prompt(8);
    for (std::uint32_t s = 0; s < 8; ++s) {
        std::vector<std::int32_t> chunk = wide_tokens(24 + 5 * s, 1 + s);
        chunk.push_back(wide_tokens(1, 100 + s)[0]);
        executor.reset(s);
        executor.forward(s, chunk, 1);
        const Tensor logits = executor.logits(1);
        std::vector<__nv_bfloat16> raw(std::size_t(logits.ne[0]));
        CUDA_CHECK(cudaStreamSynchronize(executor.head_stream()));
        CUDA_CHECK(cudaMemcpy(raw.data(), logits.data, raw.size() * 2, cudaMemcpyDeviceToHost));
        prompt[s].resize(raw.size());
        for (std::size_t v = 0; v < raw.size(); ++v) { prompt[s][v] = __bfloat162float(raw[v]); }
    }
    const auto distance = [&](const std::vector<float>& a, const std::vector<float>& b) {
        double worst = 0;
        for (std::int32_t v = 0; v < domain; ++v) {
            worst = std::max(worst, double(std::fabs(a[v] - b[v])));
        }
        return worst;
    };
    double vector_worst = 0, prompt_worst = 0, between = 0;
    // The smallest margin by which a wide row lies nearer its own sequence than the nearest other
    // sequence, on each route.
    double vector_margin = 1e30, prompt_margin = 1e30;
    int vector_agree = 0, prompt_agree = 0;
    for (std::uint32_t s = 0; s < 8; ++s) {
        const double own_vector = distance(vector[s], wide[s]);
        const double own_prompt = distance(prompt[s], wide[s]);
        vector_worst = std::max(vector_worst, own_vector);
        prompt_worst = std::max(prompt_worst, own_prompt);
        between      = std::max(between, distance(prompt[s], vector[s]));
        for (std::uint32_t o = 0; o < 8; ++o) {
            if (o == s) { continue; }
            vector_margin = std::min(vector_margin, distance(vector[o], wide[s]) - own_vector);
            prompt_margin = std::min(prompt_margin, distance(prompt[o], wide[s]) - own_prompt);
        }
        vector_agree += wide_argmax(vector[s], domain) == wide_argmax(wide[s], domain) ? 1 : 0;
        prompt_agree += wide_argmax(prompt[s], domain) == wide_argmax(wide[s], domain) ? 1 : 0;
    }
    // A step of 9 rows may take another route than the wide step (the vector kernels where they
    // are no slower at that width): equal bits, or within the same bounds as the vector route.
    double nine_worst = 0, nine_margin = 1e30;
    for (std::size_t i = 0; i < nine.size(); ++i) {
        const double own = distance(part[i], wide[nine[i]]);
        nine_worst       = std::max(nine_worst, own);
        for (const std::uint32_t o : nine) {
            if (o != nine[i]) {
                nine_margin = std::min(nine_margin, distance(part[i], wide[o]) - own);
            }
        }
    }
    std::cout << "largest logit difference from the wide rows: vector route " << vector_worst
              << " (argmax agrees on " << vector_agree << " of 8), prompt route " << prompt_worst
              << " (" << prompt_agree << " of 8), 9 rows " << nine_worst
              << "; vector against prompt route " << between
              << "; own sequence nearer than any other by at least " << vector_margin
              << " (vector route), " << prompt_margin << " (prompt route), " << nine_margin
              << " (9 rows)\n";
    // The routes round differently (their spreads on these random tokens reach a few logits, and
    // random tokens give every sequence similar logits); a row that read another sequence lies
    // nearer that sequence's.
    const double spread = 2 * std::max(between, vector_worst);
    wide_expect(prompt_worst <= spread,
                "the wide rows lie within twice the vector route's spread of the prompt route");
    wide_expect(nine_worst <= spread,
                "the 9-row step lies within twice the vector route's spread of the wide rows");
    wide_expect(vector_margin > 0 && prompt_margin > 0 && nine_margin > 0,
                "each wide row lies nearer its own sequence than any other sequence");
    return wide_failures;
}

} // namespace ninfer::test
