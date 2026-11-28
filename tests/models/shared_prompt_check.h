#pragma once

// A shared prompt pass (Executor::forward_pieces) on a family executor: the prompt pieces of four
// sequences in one pass, one of them after a history longer than the attention window (its rings
// and caches carry over into the pass) and one after a short history. The same pass gives the
// same bits twice and leaves every sequence at its piece's end, and each piece gives the same bits
// beside other tokens and histories of the same lengths (the same route): no row reads another
// sequence. A call's route follows its width, so a row in a shared pass and the same row alone may
// round differently, and a row of random tokens can sit where any rounding moves it far (on Flint,
// one row in about six at 130 to 193 columns moves 6 to 17 logits between two routes of the same
// sequence run alone). Against each piece run alone, the next decode step after the pass (which
// reads every row's keys, values and states) lies nearer its own sequence than any other piece run
// alone; so do the pass's logit rows, all but one of each pass's four (its last rows, which no
// later row reads). A piece cut before its prompt's end continues in a later pass the same way.
// The differences from alone and the chunk route's spread (a piece alone in one chunk against two)
// are printed.

#include "core/device_launch.h"
#include "core/tensor.h"
#include "wide_decode_check.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <vector>

namespace ninfer::test {

inline int shared_failures = 0;

inline void shared_expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++shared_failures;
    }
}

template <typename Executor>
std::vector<std::vector<float>> shared_logits(Executor& executor, std::uint32_t rows) {
    const Tensor logits     = executor.logits(rows);
    const std::size_t vocab = std::size_t(logits.ne[0]);
    std::vector<__nv_bfloat16> raw(vocab * rows);
    CUDA_CHECK(cudaStreamSynchronize(executor.head_stream()));
    CUDA_CHECK(cudaMemcpy(raw.data(), logits.data, raw.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<std::vector<float>> out(rows, std::vector<float>(vocab));
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t v = 0; v < vocab; ++v) { out[r][v] = __bfloat162float(raw[r * vocab + v]); }
    }
    return out;
}

// Returns the number of failed expectations. `history` (longer than the attention window) and
// the pieces fit the executor's max_context; `chunk` is its prefill_chunk (at least 256).
template <typename Executor, typename Piece>
int shared_prompt_check(Executor& executor, std::int32_t domain, std::uint32_t history,
                        std::uint32_t chunk) {
    constexpr std::uint32_t kPieces = 4;
    const std::array<std::uint32_t, kPieces> lengths{29, 6, 140, 75};
    const std::array<std::uint32_t, kPieces> histories{0, 0, history, 50};
    std::vector<std::vector<std::int32_t>> prompts, befores;
    std::vector<std::int32_t> next(kPieces);
    for (std::uint32_t s = 0; s < kPieces; ++s) {
        prompts.push_back(wide_tokens(lengths[s], 500 + s));
        befores.push_back(wide_tokens(histories[s], 900 + s));
        next[s] = wide_tokens(1, 300 + s)[0];
    }
    const auto prepare = [&](std::uint32_t s) {
        executor.reset(s);
        const std::span<const std::int32_t> tokens(befores[s]);
        for (std::size_t at = 0; at < tokens.size(); at += chunk) {
            executor.forward(s, tokens.subspan(at, std::min<std::size_t>(chunk, tokens.size() - at)), 0);
        }
    };
    const auto decode_one = [&](std::uint32_t s) {
        const std::array<std::uint32_t, 1> rows{s};
        executor.decode(rows, std::span<const std::int32_t>(&next[s], 1));
        return shared_logits(executor, 1)[0];
    };
    // Each piece alone in one chunk and in two, and the decode step after each.
    std::vector<std::vector<float>> alone, alone_next, split, split_next;
    for (std::uint32_t s = 0; s < kPieces; ++s) {
        prepare(s);
        executor.forward(s, std::span<const std::int32_t>(prompts[s]), 1);
        alone.push_back(shared_logits(executor, 1)[0]);
        alone_next.push_back(decode_one(s));
        prepare(s);
        const std::span<const std::int32_t> prompt(prompts[s]);
        const std::size_t half = prompt.size() / 2;
        executor.forward(s, prompt.first(half), 0);
        executor.forward(s, prompt.subspan(half), 1);
        split.push_back(shared_logits(executor, 1)[0]);
        split_next.push_back(decode_one(s));
    }
    std::vector<std::uint32_t> all(kPieces);
    std::iota(all.begin(), all.end(), 0U);
    // Every piece in one pass, then one decode step of every sequence.
    const auto packed_run = [&] {
        for (std::uint32_t s = 0; s < kPieces; ++s) { prepare(s); }
        std::vector<Piece> pieces;
        for (std::uint32_t s = 0; s < kPieces; ++s) {
            Piece piece;
            piece.sequence = s;
            piece.tokens   = prompts[s];
            piece.logits   = true;
            pieces.push_back(piece);
        }
        executor.forward_pieces(pieces);
        auto logits = shared_logits(executor, kPieces);
        for (std::uint32_t s = 0; s < kPieces; ++s) {
            shared_expect(executor.position(s) == histories[s] + lengths[s],
                          "sequence " + std::to_string(s) + " ends at its piece's end");
        }
        executor.decode(all, next);
        return std::make_pair(std::move(logits), shared_logits(executor, kPieces));
    };
    const auto packed = packed_run();
    const auto again  = packed_run();
    for (std::uint32_t s = 0; s < kPieces; ++s) {
        shared_expect(packed.first[s] == again.first[s] && packed.second[s] == again.second[s],
                      "piece " + std::to_string(s) + " repeats bit for bit");
    }
    // The third piece cut after 60 tokens: its first part beside the first two pieces, its rest
    // with the fourth piece in a second pass.
    for (std::uint32_t s = 0; s < kPieces; ++s) { prepare(s); }
    const std::span<const std::int32_t> third(prompts[2]);
    std::vector<Piece> first_pass(3), second_pass(2);
    first_pass[0].sequence  = 0;
    first_pass[0].tokens    = prompts[0];
    first_pass[0].logits    = true;
    first_pass[1].sequence  = 2;
    first_pass[1].tokens    = third.first(60);
    first_pass[2].sequence  = 1;
    first_pass[2].tokens    = prompts[1];
    first_pass[2].logits    = true;
    second_pass[0].sequence = 2;
    second_pass[0].tokens   = third.subspan(60);
    second_pass[0].logits   = true;
    second_pass[1].sequence = 3;
    second_pass[1].tokens   = prompts[3];
    second_pass[1].logits   = true;
    executor.forward_pieces(first_pass);
    const auto first_logits = shared_logits(executor, 2);
    executor.forward_pieces(second_pass);
    const auto second_logits = shared_logits(executor, 2);
    executor.decode(all, next);
    const auto cut_next = shared_logits(executor, kPieces);
    const std::vector<std::vector<float>> cut{first_logits[0], first_logits[1], second_logits[0],
                                              second_logits[1]};
    const auto distance = [&](const std::vector<float>& a, const std::vector<float>& b) {
        double worst = 0;
        for (std::int32_t v = 0; v < domain; ++v) {
            worst = std::max(worst, double(std::fabs(a[v] - b[v])));
        }
        return worst;
    };
    double spread = 0;
    for (std::uint32_t s = 0; s < kPieces; ++s) {
        const double prompt = distance(alone[s], split[s]), decode = distance(alone_next[s], split_next[s]);
        std::cout << "piece " << s << " (" << lengths[s] << " tokens after " << histories[s]
                  << "): one chunk against two " << prompt << ", the decode step after them "
                  << decode << '\n';
        spread = std::max({spread, prompt, decode});
    }
    // Each route's rows against the rows alone: the difference, the margin by which a row lies
    // nearer its own sequence's alone than any other sequence's, and whether the argmax agrees.
    // `tolerated` rows may lie nearer another sequence (logit rows where rounding moves them far).
    const auto against_alone = [&](const std::vector<std::vector<float>>& rows,
                                   const std::vector<std::vector<float>>& reference,
                                   const std::string& what, std::uint32_t tolerated) {
        std::uint32_t far = 0;
        for (std::uint32_t s = 0; s < kPieces; ++s) {
            const double own = distance(rows[s], reference[s]);
            double margin    = 1e30;
            for (std::uint32_t o = 0; o < kPieces; ++o) {
                if (o != s) { margin = std::min(margin, distance(rows[s], reference[o]) - own); }
            }
            far += margin > 0 ? 0 : 1;
            std::cout << what << ", piece " << s << ": difference from alone " << own
                      << ", nearer its own than any other by " << margin << ", argmax "
                      << (wide_argmax(rows[s], domain) == wide_argmax(reference[s], domain) ? "agrees"
                                                                                          : "differs")
                      << '\n';
        }
        shared_expect(far <= tolerated, what + ": at most " + std::to_string(tolerated) +
                                            " rows lie nearer another sequence run alone");
    };
    std::cout << "chunk route spread (one chunk against two) " << spread << '\n';
    against_alone(packed.first, alone, "shared pass logits", 1);
    against_alone(packed.second, alone_next, "decode after the shared pass", 0);
    against_alone(cut, alone, "cut piece passes' logits", 1);
    against_alone(cut_next, alone_next, "decode after the cut piece passes", 0);
    // Each piece beside other tokens and histories of the same lengths: the same widths, so the
    // same routes, and a row that reads only its own sequence gives the same bits.
    const auto own_prompts = prompts, own_befores = befores;
    const auto own_next    = next;
    for (std::uint32_t s = 0; s < kPieces; ++s) {
        for (std::uint32_t o = 0; o < kPieces; ++o) {
            prompts[o] = o == s ? own_prompts[o] : wide_tokens(lengths[o], 700 + 10 * s + o);
            befores[o] = o == s ? own_befores[o] : wide_tokens(histories[o], 1100 + 10 * s + o);
            next[o]    = o == s ? own_next[o] : wide_tokens(1, 1300 + 10 * s + o)[0];
        }
        const auto beside = packed_run();
        shared_expect(beside.first[s] == packed.first[s] && beside.second[s] == packed.second[s],
                      "piece " + std::to_string(s) + " reads the same bits beside other pieces' tokens");
        std::cout << "piece " << s << " beside other tokens: largest logit difference "
                  << distance(beside.first[s], packed.first[s]) << ", after its decode step "
                  << distance(beside.second[s], packed.second[s]) << '\n';
    }
    return shared_failures;
}

} // namespace ninfer::test
