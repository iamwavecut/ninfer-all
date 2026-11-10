// Context-lookup drafting is exact by construction -- the verify step rejects a wrong guess -- so
// what these checks defend is not correctness of output but the properties the round economics
// depend on: that a miss reports zero rather than a stale or partial draft, that the *most recent*
// occurrence is the one continued unless the sequence is already copying another, that a stale copy
// source falls back to the scan, and that a proposal never runs off the end of the ledger.

#include "runtime/contract/lookup_draft.h"

#include <cstddef>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::runtime;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void continues_the_earlier_occurrence() {
    std::size_t lag = 0;
    const std::vector<TokenId> ledger{1, 2, 3, 4, 1, 2};
    TokenId out[8]{};
    // The earlier "1 2" was followed by "3 4", then by the tail itself.
    require(lookup_draft(ledger, 2, 4, out, lag) == 4, "expected four drafts");
    require(out[0] == 3 && out[1] == 4 && out[2] == 1 && out[3] == 2, "wrong continuation");
}

void a_miss_reports_zero() {
    std::size_t lag = 0;
    const std::vector<TokenId> ledger{1, 2, 3, 4, 5, 6};
    TokenId out[8]{};
    // No earlier "5 6": the round must see zero and fall back to a plain step.
    require(lookup_draft(ledger, 2, 4, out, lag) == 0, "a miss must report zero");
}

void the_most_recent_occurrence_wins() {
    std::size_t lag = 0;
    // Two earlier "9 9" runs. The nearer one was followed by 7, the older by 5; in repetitive text
    // the nearer copy is the better guess, so the scan runs backwards.
    const std::vector<TokenId> ledger{9, 9, 5, 9, 9, 7, 0, 9, 9};
    TokenId out[8]{};
    require(lookup_draft(ledger, 2, 2, out, lag) == 2, "expected two drafts");
    require(out[0] == 7 && out[1] == 0, "took an older occurrence over a nearer one");
}

void a_proposal_never_runs_past_the_ledger() {
    std::size_t lag = 0;
    const std::vector<TokenId> ledger{4, 5, 6, 4, 5};
    TokenId out[8]{};
    // Only three tokens followed the earlier "4 5", however many were asked for.
    require(lookup_draft(ledger, 2, 8, out, lag) == 3, "clamp to what actually followed");
    require(out[0] == 6 && out[1] == 4 && out[2] == 5, "wrong clamped continuation");
}

void degenerate_arguments_are_refused() {
    std::size_t lag = 0;
    const std::vector<TokenId> ledger{1, 2, 3};
    TokenId out[4]{};
    require(lookup_draft(ledger, 0, 4, out, lag) == 0, "a zero n-gram matches everything; refuse it");
    require(lookup_draft(ledger, 2, 0, out, lag) == 0, "no room for drafts");
    require(lookup_draft(ledger, 2, 4, nullptr, lag) == 0, "no destination");
    const std::vector<TokenId> shorter{1, 2};
    require(lookup_draft(shorter, 3, 2, out, lag) == 0, "ledger shorter than the n-gram");
    const std::vector<TokenId> exact{1, 2};
    require(lookup_draft(exact, 2, 2, out, lag) == 0, "nothing precedes the tail to match against");
}

void a_copy_keeps_its_source() {
    // Two versions of one passage: "10 11 12 13" and the newer "10 11 12 14". The sequence copies the
    // older one, which only the token before "10" tells apart.
    std::vector<TokenId> ledger{5, 10, 11, 12, 13, 7, 10, 11, 12, 14, 8, 5, 10};
    TokenId out[8]{};
    std::size_t lag = 0;
    require(lookup_draft(ledger, 2, 3, out, lag) == 3 && out[0] == 11 && out[2] == 13,
            "the only occurrence of \"5 10\" starts the copy");
    // "10 11" and then "11 12" occur in both versions; the newer one would win a scan.
    ledger.push_back(11);
    require(lookup_draft(ledger, 2, 2, out, lag) == 2 && out[0] == 12 && out[1] == 13,
            "the copy left its source for the newer version");
    ledger.push_back(12);
    require(lookup_draft(ledger, 2, 1, out, lag) == 1 && out[0] == 13, "the copy lost its source");
    std::size_t fresh = 0;
    require(lookup_draft(ledger, 2, 1, out, fresh) == 1 && out[0] == 14,
            "without a copy the newest occurrence wins");
}

void a_stale_source_falls_back_to_the_scan() {
    const std::vector<TokenId> ledger{9, 9, 5, 9, 9, 7, 0, 9, 9};
    TokenId out[8]{};
    // Beyond the ledger (another request's value), and an earlier position whose tokens differ.
    for (const std::size_t stale : {std::size_t{1000}, std::size_t{9}, std::size_t{6}}) {
        std::size_t lag = stale;
        require(lookup_draft(ledger, 2, 2, out, lag) == 2 && out[0] == 7 && out[1] == 0,
                "a stale copy source was proposed");
        require(lag == 4, "the scan's occurrence is the new copy source");
    }
    // A miss forgets the source.
    const std::vector<TokenId> unique{1, 2, 3, 4, 5, 6};
    std::size_t lag = 3;
    require(lookup_draft(unique, 2, 4, out, lag) == 0 && lag == 0, "a miss must forget the source");
}

} // namespace

int main() {
    try {
        continues_the_earlier_occurrence();
        a_miss_reports_zero();
        the_most_recent_occurrence_wins();
        a_proposal_never_runs_past_the_ledger();
        degenerate_arguments_are_refused();
        a_copy_keeps_its_source();
        a_stale_source_falls_back_to_the_scan();
        std::printf("lookup draft checks passed\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
