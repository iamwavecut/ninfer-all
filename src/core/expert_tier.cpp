#include "core/expert_tier.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ninfer {

ExpertTier::ExpertTier(std::vector<std::uint32_t> experts, std::vector<std::uint32_t> pools,
                       ExpertTierOptions options)
    : options_(options), experts_(std::move(experts)), pools_(std::move(pools)) {
    if (experts_.empty() || pools_.size() != experts_.size() ||
        !std::isfinite(options_.decay) || options_.decay < 0 || options_.decay > 1 ||
        !std::isfinite(options_.min_gain) || options_.min_gain < 0 ||
        !std::isfinite(options_.min_count) || options_.min_count < 0 ||
        !std::isfinite(options_.gain_ratio) || options_.gain_ratio < 1) {
        throw std::invalid_argument("expert tier: invalid layers, pools or options");
    }
    std::uint32_t pool_count = 0;
    for (std::size_t l = 0; l < experts_.size(); ++l) {
        if (experts_[l] == 0) { throw std::invalid_argument("expert tier: a layer has no experts"); }
        offset_.push_back(entries_.size());
        entries_.resize(entries_.size() + experts_[l]);
        layer_of_.resize(entries_.size(), std::uint32_t(l));
        pool_count = std::max(pool_count, pools_[l] + 1);
    }
    free_.assign(pool_count, 0);
    cold_.resize(pool_count);
    cold_next_.assign(pool_count, 0);
    cold_ordered_.assign(pool_count, 0);
}

std::size_t ExpertTier::index(std::size_t layer, std::int32_t expert) const {
    if (layer >= experts_.size() || expert < 0 || std::uint32_t(expert) >= experts_[layer]) {
        throw std::out_of_range("expert tier: no such expert");
    }
    return offset_[layer] + std::size_t(expert);
}

void ExpertTier::route(std::size_t layer, std::span<const std::int32_t> ids) {
    if (layer >= experts_.size()) { throw std::out_of_range("expert tier: no such layer"); }
    for (const std::int32_t id : ids) {
        if (id >= 0 && std::uint32_t(id) < experts_[layer]) {
            entries_[offset_[layer] + std::size_t(id)].count += 1.0F;
        }
    }
}

void ExpertTier::route(std::size_t layer, std::int32_t expert, float pairs) {
    entries_[index(layer, expert)].count += pairs;
}

void ExpertTier::set(std::size_t layer, std::int32_t expert, State state) {
    entries_[index(layer, expert)].state = state;
}

ExpertTier::State ExpertTier::state(std::size_t layer, std::int32_t expert) const {
    return entries_[index(layer, expert)].state;
}

float ExpertTier::count(std::size_t layer, std::int32_t expert) const {
    return entries_[index(layer, expert)].count;
}

void ExpertTier::set_free(std::uint32_t pool, std::uint32_t slots) { free_.at(pool) = slots; }

void ExpertTier::reset() {
    std::fill(entries_.begin(), entries_.end(), Entry{});
    std::fill(free_.begin(), free_.end(), 0);
    for (auto& list : cold_) { list.clear(); }
    std::fill(cold_next_.begin(), cold_next_.end(), 0);
    std::fill(cold_ordered_.begin(), cold_ordered_.end(), 0);
    passes_ = 0;
}

float ExpertTier::steady(float per_pass) const noexcept {
    if (options_.every == 0 || options_.decay >= 1 || !(per_pass > 0)) { return 0; }
    return per_pass * float(options_.every) / (1.0F - options_.decay);
}

void ExpertTier::seed(std::size_t layer, std::span<const std::uint64_t> counts, float per_pass) {
    if (layer >= experts_.size() || counts.size() != experts_[layer]) {
        throw std::invalid_argument("expert tier: the seed describes another layer");
    }
    double total = 0;
    for (const std::uint64_t c : counts) { total += double(c); }
    const float sum = steady(per_pass);
    const double scale = sum > 0 && total > 0 ? double(sum) / total : 1.0;
    for (std::size_t e = 0; e < counts.size(); ++e) {
        entries_[offset_[layer] + e].count = float(double(counts[e]) * scale);
    }
    refresh_cold();
}

void ExpertTier::prompt(std::size_t layer, std::span<const std::uint32_t> counts, float per_pass) {
    if (layer >= experts_.size() || counts.size() != experts_[layer]) {
        throw std::invalid_argument("expert tier: the prompt describes another layer");
    }
    double total = 0;
    for (std::size_t e = 0; e < counts.size(); ++e) {
        Entry& entry = entries_[offset_[layer] + e];
        entry.count += float(counts[e]);
        total += double(entry.count);
    }
    const float sum = steady(per_pass);
    if (sum > 0 && total > double(sum)) {
        const auto scale = float(double(sum) / total);
        for (std::size_t e = 0; e < counts.size(); ++e) { entries_[offset_[layer] + e].count *= scale; }
    }
    refresh_cold();
}

// Orders the leading `count` pairs of `list` ((key, entry), ascending; entries are unique, so
// this is a stable sort's order by key) and leaves the rest after them in some order.
template <class Pair>
void order_prefix(std::vector<Pair>& list, std::size_t count) {
    count = std::min(count, list.size());
    if (count < list.size()) {
        std::nth_element(list.begin(), list.begin() + std::ptrdiff_t(count), list.end());
    }
    std::sort(list.begin(), list.begin() + std::ptrdiff_t(count));
}

void ExpertTier::refresh_cold() {
    for (auto& list : cold_) { list.clear(); }
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].state == State::Held) {
            cold_[pools_[layer_of_[i]]].emplace_back(entries_[i].count, i);
        }
    }
    // Most passes admit a few misses: the coldest few hundred serve every scan of a pass.
    for (std::size_t pool = 0; pool < cold_.size(); ++pool) {
        order_prefix(cold_[pool], kColdOrdered);
        cold_ordered_[pool] = std::min(kColdOrdered, cold_[pool].size());
    }
    std::fill(cold_next_.begin(), cold_next_.end(), 0);
}

void ExpertTier::order_cold(std::uint32_t pool, std::size_t at) {
    auto& list = cold_[pool];
    if (at < cold_ordered_[pool] || at >= list.size()) { return; }
    std::sort(list.begin() + std::ptrdiff_t(cold_ordered_[pool]), list.end());
    cold_ordered_[pool] = list.size();
}

ExpertTier::Victim ExpertTier::admission_victim(std::size_t layer, std::int32_t expert,
                                                float pairs,
                                                std::span<const unsigned char> busy) {
    const float candidate = entries_[index(layer, expert)].count + pairs;
    const std::uint32_t pool = pools_[layer];
    const auto& list         = cold_[pool];
    std::size_t& first       = cold_next_[pool];
    while (first < list.size()) {
        order_cold(pool, first);
        if (entries_[list[first].second].state == State::Held) { break; }
        ++first;
    }
    // Counts grow within a pass, so the order is a guide: the first beatable expert wins.
    std::size_t held = 0;
    for (std::size_t at = first; at < list.size() && held < kScan; ++at) {
        order_cold(pool, at);
        const std::size_t i   = list[at].second;
        const Entry& victim   = entries_[i];
        const std::uint32_t l = layer_of_[i];
        const auto e          = std::int32_t(i - offset_[l]);
        if (victim.state != State::Held) { continue; }
        ++held;
        if (l == layer && std::size_t(e) < busy.size() && busy[std::size_t(e)] != 0) { continue; }
        if (candidate >= options_.gain_ratio * victim.count + options_.min_gain) { return {l, e}; }
    }
    return {};
}

std::vector<ExpertTier::Swap> ExpertTier::round_swaps(std::size_t cap) const {
    std::vector<Swap> swaps;
    // Candidates hottest first, victims coldest first, each (key, entry) with the lower entry
    // first among equal counts. A pool's swaps come out best gain first and the round keeps at
    // most `cap`, so only its first free + cap candidates and cap victims need an order.
    std::vector<std::vector<std::pair<float, std::size_t>>> candidates(cold_.size()),
        victims(cold_.size());
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        const std::uint32_t pool = pools_[layer_of_[i]];
        if (entries_[i].state == State::Held) {
            victims[pool].emplace_back(entries_[i].count, i);
        } else if (entries_[i].state == State::Out && entries_[i].count >= options_.min_count) {
            candidates[pool].emplace_back(-entries_[i].count, i);
        }
    }
    const auto split = [&](std::size_t i) {
        const std::uint32_t l = layer_of_[i];
        return std::pair{l, std::int32_t(i - offset_[l])};
    };
    for (std::size_t pool = 0; pool < cold_.size(); ++pool) {
        auto& in  = candidates[pool];
        auto& out = victims[pool];
        const std::size_t used = std::min(in.size(), std::size_t(free_[pool]) + cap);
        order_prefix(in, used);
        order_prefix(out, std::min(out.size(), cap));
        std::size_t next = 0;
        for (std::size_t c = 0; c < used; ++c) {
            const auto [layer, expert] = split(in[c].second);
            if (c < free_[pool]) {
                swaps.push_back({layer, expert, 0, -1, entries_[in[c].second].count});
                continue;
            }
            if (next == std::min(out.size(), cap)) { break; }
            const float gain = entries_[in[c].second].count -
                               options_.gain_ratio * entries_[out[next].second].count;
            if (gain < options_.min_gain) { break; }
            const auto [victim_layer, victim] = split(out[next++].second);
            swaps.push_back({layer, expert, victim_layer, victim, gain});
        }
    }
    std::stable_sort(swaps.begin(), swaps.end(),
                     [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
    if (swaps.size() > cap) { swaps.resize(cap); }
    return swaps;
}

std::vector<ExpertTier::Swap> ExpertTier::end_pass() {
    std::vector<Swap> result;
    if (options_.every != 0 && ++passes_ >= options_.every) {
        passes_ = 0;
        result = round_swaps(options_.swaps);
        for (auto& entry : entries_) { entry.count *= options_.decay; }
    }
    refresh_cold();
    return result;
}

std::vector<ExpertTier::Swap> ExpertTier::settle() {
    auto result = round_swaps(entries_.size());
    refresh_cold();
    return result;
}

} // namespace ninfer
