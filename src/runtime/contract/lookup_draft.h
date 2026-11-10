#pragma once

#include "ninfer/base.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::runtime {

// Context-lookup drafting: propose the continuation that followed the last time this n-gram
// appeared.
//
// A draft model guesses what comes next from weights; this guesses it from the text already in
// front of it, which costs no device work at all. It is worthless on free-form prose and close to
// perfect wherever the output repeats the input -- summarising, editing, answering from a retrieved
// passage, or emitting code that reuses names and structure already on screen. Those are the cases
// a draft head is weakest on, which is why the two are complements rather than alternatives.
//
// It is exact. The proposal is one-hot, the verify step accepts the longest correct prefix and
// commits the target model's own argmax at the first mismatch, so a wrong guess costs throughput
// and never changes a token.
//
// MEASURED, 2026-09-19, Qwen3.8-27B on one 3090, `-pg 4096,256`, MTP with the optimised draft head,
// lookup preferred over the head's proposal whenever it matches:
//
//   n-gram    2      3      4      6      8     16     off
//   decode  135.6  148.9  143.5  145.2  148.7  148.5  144.9 tok/s
//   accept   0.87   1.00   0.96   0.97   1.00   1.00   0.964
//
// **Short n-grams lose.** At 2 the proposal is worse than the draft head's and decode drops 7%: a
// two-token context matches all over the place and the continuations disagree. From 8 up every
// match that fires is correct -- acceptance is exactly 1.0 -- and decode gains ~3%.
//
// **The gain is small because there was little to win.** The draft head already accepts 95-96% on
// this model, so a perfect drafter is worth +5.5% at K=3 and +13% at K=5 and no more; this takes
// about half of that (K=5: 182.7 -> 191.0 tok/s). Anyone expecting the large numbers reported for
// lookup drafting elsewhere should check the baseline's acceptance first -- against a weak drafter
// the same technique is transformative, and against this one it is a trim.
//
// **Where the headroom was:** at 100% acceptance tokens per round is set by the draft window, which
// was capped at five (K=3/4/5: 148.9/169.3/191.0 tok/s). The cap is now kMtpDecodeMaximumDrafts =
// 15, and the lookup proposes up to the same window. MEASURED, 2026-09-28, same card and model,
// rk4v4, 512 tokens of a "rename an identifier, return the whole file" answer: K=15 decodes at
// 262.6 tok/s without lookup and 262.4 with --lookup-ngram 8, at the same 12.46 tokens per round.
// On output that copies its input the draft head copies too, so the wider window, not the lookup,
// is what pays there; the lookup's remaining value is the case the head misses.
//
// **A copy keeps its source.** When the text occurs several times (two versions of a file, a passage
// quoted twice), the newest occurrence of the last N tokens is often not the one the output is
// copying: the newest version wins every tie and the round fails at each place the versions differ.
// `lag` remembers the copy a sequence followed, as the distance from the ledger's end to the copy's
// next source token. While that source still matches the last N tokens it is proposed again, and the
// newest occurrence only when it does not. A wrong lag costs nothing: the source is compared before it
// is proposed, so a stale value from another request, a rewound ledger or a rejected round finds no
// match and falls back to the scan. Simulated with the Qwen3.5 tokenizer and greedy acceptance on six
// of this repository's files, each shown as two versions (three identifiers renamed, small numbers
// changed) and then copied again, 400 tokens each; rounds newest-first / keeping the copy / an oracle
// that takes the best of every occurrence (an n-gram tree's ceiling):
//
//   N, drafts           two versions       one file, one identifier renamed
//   4, 7                381 / 312 / 306    461 / 438 / 430
//   8, 7                363 / 331 / 330    581 / 581 / 578
//   8, 15               208 / 185 / 180    457 / 454 / 453
//
// Returns the number of drafts written, and zero when the n-gram has not been seen before -- which
// the round already treats as a plain decode step. `lag` is the sequence's own; zero is "none".
[[nodiscard]] inline std::uint32_t lookup_draft(std::span<const TokenId> ledger,
                                                std::uint32_t ngram, std::uint32_t max_drafts,
                                                TokenId* out, std::size_t& lag) {
    if (ngram == 0 || max_drafts == 0 || out == nullptr) { return 0; }
    const std::size_t size = ledger.size();
    if (size <= ngram) { return 0; }

    // The n-gram to match is the tail of the ledger, whose last token is the one just committed.
    const TokenId* const tail = ledger.data() + (size - ngram);
    const auto matches = [&](std::size_t start) {
        for (std::uint32_t j = 0; j < ngram; ++j) {
            if (ledger[start + j] != tail[j]) { return false; }
        }
        return true;
    };
    const auto propose = [&](std::size_t follow) {
        const std::size_t available = size - follow;
        const auto count =
            static_cast<std::uint32_t>(available < max_drafts ? available : max_drafts);
        for (std::uint32_t j = 0; j < count; ++j) { out[j] = ledger[follow + j]; }
        lag = size - follow;
        return count;
    };

    // The copy this sequence followed, where its source has advanced with the ledger.
    if (lag > 0 && lag < size) {
        const std::size_t follow = size - lag;
        if (follow >= ngram && matches(follow - ngram)) {
            return propose(follow);
        }
    }

    // Backwards, so the most recent occurrence wins: in repetitive text the nearest copy is the one
    // most likely to continue the same way.
    for (std::size_t start = size - ngram; start-- > 0;) {
        if (matches(start)) { return propose(start + ngram); }
    }
    lag = 0;
    return 0;
}

} // namespace ninfer::runtime
