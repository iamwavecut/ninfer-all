#pragma once

// Which routed experts a device expert cache holds, decided by how often decode and verification
// calls route to them, for every MoE family with host or disk experts. Each such call adds one to
// the count of every (token, expert) pair it routes. Every `every` decode passes a round swaps the
// most routed experts the cache lacks into the slots of the least routed ones it holds, at most
// `swaps` of them over all layers, best gain first, each only when its candidate's count reaches
// `gain_ratio` times the victim's plus `min_gain` (and at least `min_count`); then every count is
// multiplied by `decay`, so the cache follows the conversation and forgets what it stopped routing.
// A miss the GPU copies anyway may also take a slot at once, under the same rule against the
// coldest expert its call does not route; a miss the CPU computes enters only through a round, so
// the CPU's share of the misses moves no expert bytes across the bus.
//
// A layer's counts settle near a steady sum: `every` passes of a token's routes between rounds,
// over 1 - decay (steady()). A routing profile seeds them at that sum (seed()), so the experts it
// placed keep their slots until the conversation's own routes beat them; a prompt's routes join
// them scaled down to that sum (prompt()), so a prompt places the cache its reply starts with and
// its thousands of tokens never outweigh the reply's routes.
//
// Slots of one pool are interchangeable: a candidate replaces a victim of its own pool (one pool
// per layer when every layer owns its slots, as Flash-Next's cache does; one for each slot size in
// ExpertDelivery's). The tier keeps no slots and copies nothing; the cache that owns the slots
// reports each expert's state and carries the swaps out: a victim leaves at once (the CPU or a copy
// serves it from then on), a candidate is pending until its copy has landed.
//
// The constants: a round every four decode passes, up to 64 swaps, decay 0.95, a candidate's count
// at least 1.5 times its victim's plus 2. Measured with host experts and 512-token replies after a
// 2K prompt against Strata's earlier ones (every four, 96 swaps, decay 0.7, plus 1.5, no ratio):
// Qwen3.6-35B-A3B +14.9% (RTX 5090, 6 GiB cache), Flint +4.5%, Inkling 2 +1.8%, Qwen3.8-Flash-Next
// +0.9%, Gemma 4 26B +0.4% (RTX 3060); the slow decay and the ratio stop the cache from trading
// experts whose counts differ by a token's routes.

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace ninfer {

struct ExpertTierOptions {
    std::uint32_t every = 4;  // decode passes between rounds; 0: no rounds
    std::uint32_t swaps = 64; // swaps a round, at most, over every pool
    float decay         = 0.95F;
    float min_gain      = 2.0F;
    float min_count     = 2.0F;
    float gain_ratio    = 1.5F;
};

class ExpertTier {
public:
    enum class State : std::uint8_t { Out, Pending, Held };

    // experts[l]: layer l's routed experts; pools[l]: the slot pool layer l's experts live in.
    ExpertTier(std::vector<std::uint32_t> experts, std::vector<std::uint32_t> pools,
               ExpertTierOptions options = {});

    [[nodiscard]] const ExpertTierOptions& options() const noexcept { return options_; }
    [[nodiscard]] std::size_t layers() const noexcept { return experts_.size(); }

    // One decode or verification call's routes through `layer`: every id in range counts once.
    void route(std::size_t layer, std::span<const std::int32_t> ids);
    void route(std::size_t layer, std::int32_t expert, float pairs);

    void set(std::size_t layer, std::int32_t expert, State state);
    [[nodiscard]] State state(std::size_t layer, std::int32_t expert) const;
    [[nodiscard]] float count(std::size_t layer, std::int32_t expert) const;
    // Free slots of `pool` a round may fill without a victim.
    void set_free(std::uint32_t pool, std::uint32_t slots);
    // Forgets every count and state: nothing held, no free slots, no passes.
    void reset();
    // The sum a layer's counts settle near when each pass routes `per_pass` pairs through it; 0
    // without rounds or decay, where no sum is steady.
    [[nodiscard]] float steady(float per_pass) const noexcept;
    // Sets `layer`'s counts to the shares of `counts` (a routing profile's, per expert) of
    // steady(per_pass); without a steady sum, to the counts themselves.
    void seed(std::size_t layer, std::span<const std::uint64_t> counts, float per_pass);
    // A wide call's routes through `layer` (a prompt chunk: `counts` per expert) join its counts,
    // which then scale down to steady(per_pass) when they exceed it.
    void prompt(std::size_t layer, std::span<const std::uint32_t> counts, float per_pass);

    struct Victim {
        std::uint32_t layer = 0;
        std::int32_t expert = -1; // -1: none qualifies
    };
    // For a miss of `layer` the GPU copies anyway, with `pairs` routes in its call: among the
    // coldest held experts of its pool as the last pass, seed or prompt ended (the first 32 still
    // held), the first that `busy` (a flag per expert of `layer`, the call's routes) does not mark
    // and whose count, times gain_ratio plus min_gain, the candidate's with these pairs reaches.
    [[nodiscard]] Victim admission_victim(std::size_t layer, std::int32_t expert, float pairs,
                                          std::span<const unsigned char> busy);

    struct Swap {
        std::uint32_t layer = 0;
        std::int32_t in     = -1;
        // The victim (of the same pool), or out == -1 for a free slot.
        std::uint32_t out_layer = 0;
        std::int32_t out        = -1;
        float gain              = 0;
    };
    // Ends a decode pass. Every `every` passes: the round's swaps, best gain first, and the counts'
    // decay. The caller carries them out and reports the states.
    [[nodiscard]] std::vector<Swap> end_pass();
    // Every swap the counts call for now, best gain first, by the round's rule without its cap,
    // and no pass or decay: after a prompt has joined its routes (prompt()), the cache its reply
    // starts with.
    [[nodiscard]] std::vector<Swap> settle();

private:
    struct Entry {
        float count = 0;
        State state = State::Out;
    };
    void refresh_cold();
    [[nodiscard]] std::vector<Swap> round_swaps(std::size_t cap) const;
    [[nodiscard]] std::size_t index(std::size_t layer, std::int32_t expert) const;

    ExpertTierOptions options_;
    std::vector<std::uint32_t> experts_, pools_;
    std::vector<std::size_t> offset_; // first entry of each layer
    std::vector<Entry> entries_;
    std::vector<std::uint32_t> free_; // by pool
    // By pool: held entries with their counts as the last pass ended, coldest first (lower entry
    // first among equal counts), and the first one still held. A pass orders only the first
    // kColdOrdered of them; a scan that reaches past the ordered ones orders the rest by the same
    // counts (cold_ordered_), so the order is a whole sort's at the cost of a selection.
    std::vector<std::vector<std::pair<float, std::size_t>>> cold_;
    std::vector<std::size_t> cold_next_, cold_ordered_;
    static constexpr std::size_t kScan        = 32;
    static constexpr std::size_t kColdOrdered = 512;
    void order_cold(std::uint32_t pool, std::size_t at);
    std::vector<std::uint32_t> layer_of_; // by entry
    std::uint32_t passes_ = 0;
};

} // namespace ninfer
