// How tensor-parallel ranks divide a dimension (split_shares), the work by speed beside rank 0's
// own work (shares_beside_first) and by speed and room (fill_shares), and delivered experts by
// cache room and bus (delivery_shares). The expected splits are worked out by hand from the stated
// rule: every inner end is the allowed position nearest its weight's target, the lower one on a
// tie, leaving each later share a position.

#include "core/tensor_split.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (condition) { return; }
    std::cerr << "expectation failed: " << label << '\n';
    ++failures;
}

template <class Fn>
void check_throws(Fn&& fn, const char* label) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return;
    }
    std::cerr << "expectation failed: " << label << " (nothing was thrown)\n";
    ++failures;
}

using Ends = std::vector<std::uint64_t>;

Ends split(std::uint64_t total, std::uint64_t unit, std::vector<double> weights) {
    return ninfer::split_shares(total, ninfer::unit_positions(total, unit), weights);
}

bool near(const std::vector<double>& actual, const std::vector<double>& expected) {
    if (actual.size() != expected.size()) { return false; }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(actual[i] - expected[i]) > 1e-12) { return false; }
    }
    return true;
}

void split_cases() {
    // Equal weights over evenly spaced positions: equal shares (16 GDN key heads).
    check(split(16, 1, {1, 1}) == Ends{0, 8, 16}, "two equal ranks");
    check(split(16, 1, {1, 1, 1, 1}) == Ends{0, 4, 8, 12, 16}, "four equal ranks");
    // 3:1 puts the end at 12.
    check(split(16, 1, {3, 1}) == Ends{0, 12, 16}, "3:1");
    // Three equal ranks of 16: targets 5.33 and 10.67.
    check(split(16, 1, {1, 1, 1}) == Ends{0, 5, 11, 16}, "three equal ranks");
    // Only even ends (two heads per unit): 5.33 -> 6, 10.67 -> 10.
    check(split(16, 2, {1, 1, 1}) == Ends{0, 6, 10, 16}, "three ranks, two-head units");
    // A tie: units of three over 15 put the target 7.5 midway between 6 and 9; the lower wins.
    check(split(15, 3, {1, 1}) == Ends{0, 6, 15}, "a tie takes the lower end");
    // An overwhelming weight still leaves the last rank one position.
    check(split(16, 1, {1000, 1}) == Ends{0, 15, 16}, "the last share keeps one unit");
    check(split(16, 1, {1, 1000}) == Ends{0, 1, 16}, "the first share keeps one unit");
    check(split(16, 4, {1, 1000, 1000, 1000}) == Ends{0, 4, 8, 12, 16},
          "later ranks keep their positions");
    // An FFN of 17408 rows in 256-row blocks at an RTX 3090 : RTX 3060 bandwidth ratio: the target
    // 12572.4 lies nearest the block end 12544.
    check(split(17408, 256, {936, 360}) == Ends{0, 12544, 17408}, "FFN rows by bandwidth");
    // Explicit positions (a gathered projection's whole blocks).
    const Ends allowed{3, 4, 9, 10};
    check(ninfer::split_shares(12, allowed, std::vector<double>{1, 1}) == Ends{0, 4, 12},
          "explicit positions: 6 is nearer 4 than 9");
    check(ninfer::split_shares(12, Ends{}, std::vector<double>{1}) == Ends{0, 12}, "one rank");

    check_throws([] { (void)split(4, 2, {1, 1, 1}); }, "too few positions for the ranks");
    check_throws([] { (void)split(16, 1, {1, 0}); }, "a zero weight");
    check_throws([] { (void)split(16, 1, {1, -2}); }, "a negative weight");
    check_throws([] { (void)split(16, 1, {}); }, "no ranks");
    check_throws([] { (void)ninfer::split_shares(8, Ends{4, 4}, std::vector<double>{1, 1, 1}); },
                 "repeated positions");
    check_throws([] { (void)ninfer::split_shares(8, Ends{8}, std::vector<double>{1, 1}); },
                 "a position at the end");
}

void beside_first_cases() {
    // Four equal ranks, rank 0 also running an eighth of the divided work (a shared expert of
    // 512 rows beside 8 routed experts of 512): (s + 1/8) on rank 0 and s on the others meet at
    // 9/32, so rank 0 takes 9/32 - 1/8 = 5/32.
    check(near(ninfer::shares_beside_first(std::vector<double>{1, 1, 1, 1}, 0.125),
               {5.0 / 32.0, 9.0 / 32.0, 9.0 / 32.0, 9.0 / 32.0}),
          "rank 0 sheds its own work");
    // A faster rank 0 keeps more: speeds 2 and 1, first 0.5, level 1.5 / 3 = 0.5 per unit of speed.
    check(near(ninfer::shares_beside_first(std::vector<double>{2, 1}, 0.5), {0.5, 0.5}),
          "speeds weigh the level");
    check(near(ninfer::shares_beside_first(std::vector<double>{3, 1}, 0.0), {0.75, 0.25}),
          "no own work follows speed");
    // Rank 0's own work alone outlasts the others' shares: it keeps almost nothing.
    const std::vector<double> lone = ninfer::shares_beside_first(std::vector<double>{1, 1}, 2.0);
    check(lone[0] > 0.0 && lone[0] < 1e-5 && std::abs(lone[1] - 1.0) < 1e-12,
          "own work past the others' shares");
    check(near(ninfer::shares_beside_first(std::vector<double>{1}, 0.25), {1.0}), "one rank");
    check_throws([] { (void)ninfer::shares_beside_first(std::vector<double>{1, 1}, -0.1); },
                 "negative own work");
    check_throws([] { (void)ninfer::shares_beside_first(std::vector<double>{0, 1}, 0.1); },
                 "a zero speed beside own work");
}

void fill_cases() {
    // Room everywhere: in proportion to speed.
    check(near(ninfer::fill_shares(std::vector<double>{3, 1}, std::vector<double>{9, 9}),
               {0.75, 0.25}),
          "proportional to speed");
    // A full rank's excess goes to the other.
    check(near(ninfer::fill_shares(std::vector<double>{1, 1}, std::vector<double>{0.3, 2}),
               {0.3, 0.7}),
          "capped rank");
    // The excess goes to the open ranks in proportion to their speed.
    check(near(ninfer::fill_shares(std::vector<double>{2, 1, 1}, std::vector<double>{0.25, 1, 1}),
               {0.25, 0.375, 0.375}),
          "excess shared by speed");
    // Capping one rank can overflow another, which is capped in turn.
    check(near(ninfer::fill_shares(std::vector<double>{1, 1, 1}, std::vector<double>{0.1, 0.2, 1}),
               {0.1, 0.2, 0.7}),
          "cascading caps");
    check(near(ninfer::fill_shares(std::vector<double>{4, 1, 1}, std::vector<double>{0.3, 0.6, 0.3}),
               {0.3, 0.4, 0.3}),
          "a cap reached only after redistribution");
    // Not enough room anywhere: speed alone.
    check(near(ninfer::fill_shares(std::vector<double>{1, 3}, std::vector<double>{0.2, 0.2}),
               {0.25, 0.75}),
          "no fit follows speed");
    // Not enough room, but the work shrinks to fit (automatic KV): in proportion to the room, so
    // both ranks outgrow theirs by the same factor. Without room on one rank, speed again.
    check(near(ninfer::fill_shares(std::vector<double>{1, 1}, std::vector<double>{0.3, 0.6}, true),
               {1.0 / 3.0, 2.0 / 3.0}),
          "a shrinking fit follows room");
    check(near(ninfer::fill_shares(std::vector<double>{1, 1}, std::vector<double>{0.4, 0.8}, true),
               {0.4, 0.6}),
          "a shrinking fit that fits caps as usual");
    check(near(ninfer::fill_shares(std::vector<double>{1, 3}, std::vector<double>{0, 0.5}, true),
               {0.25, 0.75}),
          "a rank without room falls back to speed");
    check_throws([] { (void)ninfer::fill_shares(std::vector<double>{1}, std::vector<double>{}); },
                 "a capacity per rank");
    check_throws(
        [] { (void)ninfer::fill_shares(std::vector<double>{0, 1}, std::vector<double>{1, 1}); },
        "a zero speed");
}

void delivery_cases() {
    using V = std::vector<double>;
    // Rank 0 holds the model: its cache has 10 of the 100 expert bytes, the peer's 20. Equal buses
    // copy the same 35 each: 45 and 55 bytes of experts.
    check(near(ninfer::delivery_shares(V{10, 20}, V{1, 1}, 100, V{1, 1}), {0.45, 0.55}),
          "more room takes more experts");
    // No cache: the shares follow the buses.
    check(near(ninfer::delivery_shares(V{0, 0}, V{1, 3}, 40, V{1, 1}), {0.25, 0.75}),
          "no room follows the bus");
    // Room and a faster bus: time (100 - 30) / 3 per rank, shares (10 + t) and (20 + 2t) of 100.
    check(near(ninfer::delivery_shares(V{10, 20}, V{1, 2}, 100, V{1, 1}),
               {(10 + 70.0 / 3) / 100, (20 + 140.0 / 3) / 100}),
          "room and bus together");
    // The caches hold every expert: speed within the room (fill_shares).
    check(near(ninfer::delivery_shares(V{60, 60}, V{1, 1}, 100, V{1, 3}), {0.4, 0.6}),
          "everything cached follows speed within the room");
    check_throws([] { (void)ninfer::delivery_shares(V{1, 1}, V{1, 1}, 0, V{1, 1}); },
                 "no expert bytes");
    check_throws([] { (void)ninfer::delivery_shares(V{1, 1}, V{0, 1}, 10, V{1, 1}); },
                 "a zero bus");
    check_throws([] { (void)ninfer::delivery_shares(V{-1, 1}, V{1, 1}, 10, V{1, 1}); },
                 "a negative room");
    check_throws([] { (void)ninfer::delivery_shares(V{1}, V{1, 1}, 10, V{1, 1}); },
                 "a room per rank");
}

} // namespace

int run_tensor_split_test() {
    try {
        split_cases();
        beside_first_cases();
        fill_cases();
        delivery_cases();
    } catch (const std::exception& error) {
        std::cerr << "unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) { return 1; }
    std::cout << "tensor split tests passed\n";
    return 0;
}
