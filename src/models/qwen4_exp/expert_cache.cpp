#include "models/qwen4_exp/expert_cache.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>

namespace ninfer::models::qwen4_exp {
namespace {

constexpr std::uint64_t kAlign = 256;
// Per-token decay of an expert's route count: a half-life of about 700 tokens.
constexpr double kDecay = 0.999;

// Admission's frequency tiers of a decayed route count: about once lately, a few times, often.
// Measured on Q2_0 host experts, RTX 3090: against LRU alone, a revisited prompt's first token
// came 23-28% sooner, a new prompt's decode 0-4% slower; two-tier and higher thresholds did less.
constexpr double kTierOnce = 1.5;
constexpr double kTierFew  = 6.0;
int frequency_tier(double score) { return score < kTierOnce ? 0 : score < kTierFew ? 1 : 2; }
// A candidate displaces a cached expert only when it is clearly hotter, so near-ties do not thrash.
constexpr double kHysteresis = 1.25;
constexpr double kFloor      = 0.5;
// Zero bytes after every slot's down matrix, which the expert matrix kernel reads past a down
// row's end (expert_stream.cpp says why they must be zeros, not the next slot's gate).
constexpr std::uint64_t kTail = 256;

std::uint64_t aligned(std::uint64_t value) { return (value + kAlign - 1) / kAlign * kAlign; }

} // namespace

struct ExpertCache::Layer {
    ExpertCacheLayer banks;
    std::uint32_t slots      = 0;
    std::uint64_t slot_bytes = 0;
    DeviceBuffer storage;
    // Slots from base_slots on live in `extra`, allocated by grow() once startup is done.
    DeviceBuffer extra;
    std::uint32_t base_slots = 0;
    std::vector<std::int32_t> slot_of;
    std::vector<std::int32_t> expert_of;
    std::vector<double> score;
    std::vector<std::uint64_t> total;
    // Admitted by the misses since the last observe: its routes then were misses.
    std::vector<unsigned char> fresh;
    bool handed_over = false;
    std::vector<const void*> current[3];
    std::size_t staging = 0; // byte offset of its three tables in the pinned staging

    [[nodiscard]] const ExpertBank& bank(int k) const {
        return k == 0 ? banks.gate : k == 1 ? banks.up : banks.down;
    }

    [[nodiscard]] std::byte* slot_base(std::int64_t slot) const {
        return slot < std::int64_t(base_slots)
                   ? static_cast<std::byte*>(storage.p) + std::size_t(slot) * slot_bytes
                   : static_cast<std::byte*>(extra.p) +
                         std::size_t(slot - std::int64_t(base_slots)) * slot_bytes;
    }

    [[nodiscard]] std::uint64_t offset(int k) const {
        return k == 0   ? 0
               : k == 1 ? aligned(banks.gate.expert_bytes)
                        : aligned(banks.gate.expert_bytes) + aligned(banks.up.expert_bytes);
    }
};

ExpertCache::ExpertCache(DeviceContext& device, std::vector<ExpertCacheLayer> layers,
                         std::span<const std::uint64_t> bytes_by_rank)
    : device_(device) {
    const auto slot_bytes_of = [](const ExpertCacheLayer& banks) {
        return aligned(banks.gate.expert_bytes) + aligned(banks.up.expert_bytes) +
               aligned(banks.down.expert_bytes + kTail);
    };
    // Every layer of a rank gets the same number of slots: a layer of wider experts (the MTP
    // block's) takes a proportionally larger share of the bytes.
    std::map<std::size_t, std::uint64_t> slot_bytes_on_rank;
    for (const auto& layer : layers) { slot_bytes_on_rank[layer.rank] += slot_bytes_of(layer); }
    std::size_t staging_bytes = 0;
    for (auto& banks : layers) {
        const std::size_t experts = banks.gate.host.size();
        if (experts == 0 || banks.up.host.size() != experts || banks.down.host.size() != experts) {
            throw std::invalid_argument("expert cache: a layer's banks differ in expert count");
        }
        Layer layer;
        layer.slot_bytes = slot_bytes_of(banks);
        layer.banks      = std::move(banks);
        const std::uint64_t slots =
            layer.banks.rank < bytes_by_rank.size()
                ? bytes_by_rank[layer.banks.rank] / slot_bytes_on_rank[layer.banks.rank]
                : 0;
        layer.slots = static_cast<std::uint32_t>(std::min<std::uint64_t>(slots, experts));
        layer.slot_of.assign(experts, -1);
        layer.expert_of.assign(layer.slots, -1);
        layer.score.assign(experts, 0.0);
        layer.total.assign(experts, 0);
        layer.fresh.assign(experts, 0);
        for (int k = 0; k < 3; ++k) { layer.current[k] = layer.bank(k).host; }
        if (layer.slots != 0) {
            RankBinding bind(device_, layer.banks.rank);
            layer.storage = DeviceBuffer(std::size_t(layer.slots) * layer.slot_bytes);
            layer.storage.fill(0);
        }
        layer.base_slots = layer.slots;
        layer.staging = staging_bytes;
        staging_bytes += 3 * experts * sizeof(void*);
        stats_.slots += layer.slots;
        layers_.push_back(std::move(layer));
    }
    staging_ = std::make_unique<PinnedHostBuffer>(std::max<std::size_t>(staging_bytes, 64));
}

ExpertCache::~ExpertCache() = default;

void ExpertCache::observe(std::size_t index, std::span<const std::int32_t> ids,
                          std::uint32_t tokens) {
    std::lock_guard lock(mutex_);
    Layer& layer       = layers_.at(index);
    const double decay = std::pow(kDecay, double(tokens));
    for (double& score : layer.score) { score *= decay; }
    for (const std::int32_t id : ids) {
        if (id < 0 || std::size_t(id) >= layer.score.size()) { continue; }
        layer.score[id] += 1.0;
        ++layer.total[std::size_t(id)];
        ++stats_.routes;
        if (layer.slot_of[id] >= 0 && !layer.fresh[std::size_t(id)]) { ++stats_.hits; }
    }
    std::fill(layer.fresh.begin(), layer.fresh.end(), 0);
}

void ExpertCache::rebalance(std::uint64_t byte_budget, bool handed_over_too) {
    std::lock_guard lock(mutex_);
    // The budget is the layers' that rebalance now: handed-over ones admit on their own, except
    // after a prompt chunk, whose calls admit nothing.
    const auto rebalances = [&](const Layer& layer) { return handed_over_too || !layer.handed_over; };
    const auto rebalancing =
        static_cast<std::uint64_t>(std::count_if(layers_.begin(), layers_.end(), rebalances));
    if (rebalancing == 0 || stats_.slots == 0) { return; }
    const std::uint64_t per_layer = byte_budget / rebalancing;
    auto* staging                 = static_cast<std::byte*>(staging_->data());
    for (Layer& layer : layers_) {
        if (layer.slots == 0 || !rebalances(layer)) { continue; }
        const std::size_t experts = layer.score.size();
        std::vector<std::int32_t> candidates;
        for (std::size_t e = 0; e < experts; ++e) {
            if (layer.slot_of[e] < 0 && layer.score[e] > 0.0) {
                candidates.push_back(std::int32_t(e));
            }
        }
        if (candidates.empty()) { continue; }
        const std::size_t limit =
            std::max<std::size_t>(1, static_cast<std::size_t>(per_layer / layer.slot_bytes));
        const std::size_t considered = std::min(limit, candidates.size());
        std::partial_sort(candidates.begin(), candidates.begin() + std::ptrdiff_t(considered),
                          candidates.end(), [&](std::int32_t a, std::int32_t b) {
                              return layer.score[a] != layer.score[b]
                                         ? layer.score[a] > layer.score[b]
                                         : a < b;
                          });
        candidates.resize(considered);
        // Free slots first, then the coldest cached experts, coldest first.
        std::vector<std::int32_t> free_slots, victims;
        for (std::uint32_t s = 0; s < layer.slots; ++s) {
            if (layer.expert_of[s] < 0) {
                free_slots.push_back(std::int32_t(s));
            } else {
                victims.push_back(layer.expert_of[s]);
            }
        }
        std::sort(victims.begin(), victims.end(), [&](std::int32_t a, std::int32_t b) {
            return layer.score[a] != layer.score[b] ? layer.score[a] < layer.score[b] : a > b;
        });
        std::size_t next_free = 0, next_victim = 0;
        bool changed = false;
        RankBinding bind(device_, layer.banks.rank);
        for (const std::int32_t expert : candidates) {
            std::int32_t slot = -1;
            if (next_free < free_slots.size()) {
                slot = free_slots[next_free++];
            } else if (next_victim < victims.size()) {
                const std::int32_t victim = victims[next_victim];
                if (!(layer.score[expert] > kHysteresis * layer.score[victim] + kFloor)) { break; }
                ++next_victim;
                slot                  = layer.slot_of[victim];
                layer.slot_of[victim] = -1;
                for (int k = 0; k < 3; ++k) {
                    layer.current[k][victim] = layer.bank(k).host[victim];
                }
            } else {
                break;
            }
            auto* base =
                layer.slot_base(slot);
            for (int k = 0; k < 3; ++k) {
                const ExpertBank& bank = layer.bank(k);
                std::byte* target      = base + layer.offset(k);
                CUDA_CHECK(cudaMemcpyAsync(target, bank.host[expert],
                                           std::size_t(bank.expert_bytes), cudaMemcpyHostToDevice,
                                           layer.banks.stream));
                layer.current[k][expert] = target;
                stats_.copied_bytes += std::uint64_t(bank.expert_bytes);
            }
            layer.slot_of[expert] = slot;
            layer.expert_of[slot] = expert;
            ++stats_.admitted;
            changed = true;
        }
        if (!changed) { continue; }
        for (int k = 0; k < 3; ++k) {
            const std::size_t bytes = experts * sizeof(void*);
            std::byte* mirror       = staging + layer.staging + std::size_t(k) * bytes;
            std::memcpy(mirror, layer.current[k].data(), bytes);
            CUDA_CHECK(cudaMemcpyAsync(layer.bank(k).table, mirror, bytes, cudaMemcpyHostToDevice,
                                       layer.banks.stream));
        }
    }
}

void ExpertCache::seed(const std::vector<std::vector<std::uint64_t>>& counts) {
    std::lock_guard lock(mutex_);
    if (counts.size() != layers_.size()) {
        throw std::invalid_argument("expert cache: the profile describes other layers");
    }
    auto* staging = static_cast<std::byte*>(staging_->data());
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        Layer& layer              = layers_[index];
        const std::size_t experts = layer.score.size();
        if (counts[index].size() != experts) {
            throw std::invalid_argument("expert cache: the profile's bank width differs");
        }
        std::uint64_t sum = 0;
        for (const auto c : counts[index]) { sum += c; }
        if (layer.slots == 0 || sum == 0) { continue; }
        std::vector<std::int32_t> order(experts);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](std::int32_t a, std::int32_t b) {
            return counts[index][std::size_t(a)] > counts[index][std::size_t(b)];
        });
        // Scores start as if the profile's share of routes had been seen over one half-life.
        const double scale = 700.0 * 10.0 / double(sum);
        for (std::size_t e = 0; e < experts; ++e) { layer.score[e] = double(counts[index][e]) * scale; }
        RankBinding bind(device_, layer.banks.rank);
        std::uint32_t slot = 0;
        for (const std::int32_t expert : order) {
            if (slot == layer.slots || counts[index][std::size_t(expert)] == 0) { break; }
            if (layer.slot_of[std::size_t(expert)] >= 0) { continue; }
            auto* base = layer.slot_base(slot);
            for (int k = 0; k < 3; ++k) {
                const ExpertBank& bank = layer.bank(k);
                std::byte* target      = base + layer.offset(k);
                CUDA_CHECK(cudaMemcpyAsync(target, bank.host[std::size_t(expert)],
                                           std::size_t(bank.expert_bytes), cudaMemcpyHostToDevice,
                                           layer.banks.stream));
                layer.current[k][std::size_t(expert)] = target;
                stats_.copied_bytes += std::uint64_t(bank.expert_bytes);
            }
            layer.slot_of[std::size_t(expert)] = std::int32_t(slot);
            layer.expert_of[slot]             = expert;
            ++slot;
            ++stats_.admitted;
        }
        for (int k = 0; k < 3; ++k) {
            const std::size_t bytes = experts * sizeof(void*);
            std::byte* mirror       = staging + layer.staging + std::size_t(k) * bytes;
            std::memcpy(mirror, layer.current[k].data(), bytes);
            CUDA_CHECK(cudaMemcpyAsync(layer.bank(k).table, mirror, bytes, cudaMemcpyHostToDevice,
                                       layer.banks.stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(layer.banks.stream));
    }
}

std::vector<std::vector<std::uint64_t>> ExpertCache::totals() const {
    std::lock_guard lock(mutex_);
    std::vector<std::vector<std::uint64_t>> out;
    out.reserve(layers_.size());
    for (const Layer& layer : layers_) { out.push_back(layer.total); }
    return out;
}

const void* ExpertCache::cached(std::size_t index, int k, std::int32_t expert) const {
    std::lock_guard lock(mutex_);
    const Layer& layer = layers_.at(index);
    return layer.slot_of.at(static_cast<std::size_t>(expert)) >= 0 ? layer.current[k][expert]
                                                                   : nullptr;
}

ExpertCacheStats ExpertCache::stats() const noexcept {
    std::lock_guard lock(mutex_);
    return stats_;
}

void ExpertCache::hand_over(std::size_t index) {
    std::lock_guard lock(mutex_);
    layers_.at(index).handed_over = true;
}

bool ExpertCache::admit(std::size_t index, std::int32_t expert,
                        std::span<const std::uint32_t> last_use,
                        std::span<const unsigned char> in_use, Admitted& out) {
    std::lock_guard lock(mutex_);
    Layer& layer = layers_.at(index);
    if (!layer.handed_over || layer.slots == 0 || expert < 0 ||
        std::size_t(expert) >= layer.slot_of.size() || layer.slot_of[std::size_t(expert)] >= 0 ||
        last_use.size() < layer.slot_of.size() || in_use.size() < layer.slot_of.size()) {
        return false;
    }
    // The victim is the least recently used expert of the lowest frequency tier: an expert routed
    // about once lately goes before one other prompts keep routing, so a decode's passing experts
    // do not push out a working set it may come back to.
    std::int32_t slot = -1;
    int lowest        = 0;
    std::uint32_t oldest = 0;
    for (std::uint32_t s = 0; s < layer.slots; ++s) {
        const std::int32_t held = layer.expert_of[s];
        if (held < 0) {
            slot = std::int32_t(s);
            break;
        }
        if (in_use[std::size_t(held)] != 0) { continue; }
        const int tier           = frequency_tier(layer.score[std::size_t(held)]);
        const std::uint32_t used = last_use[std::size_t(held)];
        if (slot < 0 || tier < lowest || (tier == lowest && used < oldest)) {
            slot   = std::int32_t(s);
            lowest = tier;
            oldest = used;
        }
    }
    if (slot < 0) { return false; }
    out        = Admitted{};
    auto* base = layer.slot_base(slot);
    const std::int32_t victim = layer.expert_of[std::size_t(slot)];
    if (victim >= 0) {
        out.victim             = victim;
        layer.slot_of[victim]  = -1;
        for (int k = 0; k < 3; ++k) {
            out.victim_host[k]       = layer.bank(k).host[victim];
            layer.current[k][victim] = layer.bank(k).host[victim];
        }
    }
    for (int k = 0; k < 3; ++k) {
        out.slot[k]              = base + layer.offset(k);
        layer.current[k][expert] = out.slot[k];
        stats_.copied_bytes += std::uint64_t(layer.bank(k).expert_bytes);
    }
    layer.slot_of[std::size_t(expert)] = slot;
    layer.expert_of[std::size_t(slot)] = expert;
    layer.fresh[std::size_t(expert)]   = 1;
    ++stats_.admitted;
    return true;
}

std::array<void*, 3> ExpertCache::tables(std::size_t index) const {
    const Layer& layer = layers_.at(index);
    return {layer.banks.gate.table, layer.banks.up.table, layer.banks.down.table};
}

ExpertCache::Ranges ExpertCache::storage(std::size_t index) const {
    std::lock_guard lock(mutex_);
    const Layer& layer = layers_.at(index);
    const auto range   = [](const DeviceBuffer& buffer) {
        const auto begin = reinterpret_cast<std::uintptr_t>(buffer.p);
        return std::pair<std::uintptr_t, std::uintptr_t>{begin, begin + buffer.bytes};
    };
    return {range(layer.storage), range(layer.extra)};
}

std::vector<std::uint64_t> ExpertCache::grow(std::span<const std::uint64_t> bytes_by_rank) {
    std::lock_guard lock(mutex_);
    std::vector<std::uint64_t> added(bytes_by_rank.size(), 0);
    // As at construction, every growing layer of a rank gains the same number of slots.
    std::map<std::size_t, std::uint64_t> slot_bytes_on_rank;
    for (const Layer& layer : layers_) {
        if (layer.extra.p == nullptr && layer.slots < layer.slot_of.size()) {
            slot_bytes_on_rank[layer.banks.rank] += layer.slot_bytes;
        }
    }
    for (Layer& layer : layers_) {
        const std::size_t rank = layer.banks.rank;
        if (layer.extra.p != nullptr || layer.slots >= layer.slot_of.size() ||
            rank >= bytes_by_rank.size() || slot_bytes_on_rank[rank] == 0) {
            continue;
        }
        const std::uint64_t more = std::min<std::uint64_t>(
            bytes_by_rank[rank] / slot_bytes_on_rank[rank], layer.slot_of.size() - layer.slots);
        if (more == 0) { continue; }
        RankBinding bind(device_, rank);
        layer.extra = DeviceBuffer(std::size_t(more) * layer.slot_bytes);
        layer.extra.fill(0);
        layer.base_slots = layer.slots;
        layer.slots += static_cast<std::uint32_t>(more);
        layer.expert_of.resize(layer.slots, -1);
        stats_.slots += static_cast<std::uint32_t>(more);
        added[rank] += more * layer.slot_bytes;
    }
    return added;
}

} // namespace ninfer::models::qwen4_exp
