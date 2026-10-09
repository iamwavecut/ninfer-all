// The expert cache must never change what a kernel reads: after any sequence of routes, rebalances,
// admissions and growth, every table entry points either at the expert's bytes in the pinned host
// block or at a device slot holding the same bytes, a cached down matrix followed by zeros (the
// expert matrix kernel reads past a down row's end). It must also admit the experts a layer routes
// to most, keep within its slots, give every layer the same number of slots, take on an admission
// the least recently used slot not in use of the lowest frequency tier, and count a fresh
// admission's routes as misses.
#include "core/arena.h"
#include "core/device.h"
#include "models/qwen4_exp/expert_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::models::qwen4_exp;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

constexpr int kExperts = 64;

struct Bank {
    std::int64_t bytes;
    std::unique_ptr<PinnedHostBuffer> host;
    DeviceBuffer table;
    std::vector<const void*> pointers;

    Bank(std::int64_t expert_bytes, std::mt19937& random)
        : bytes(expert_bytes), host(std::make_unique<PinnedHostBuffer>(expert_bytes * kExperts)),
          table(kExperts * sizeof(void*)) {
        auto* data = static_cast<std::uint8_t*>(host->data());
        for (std::int64_t i = 0; i < expert_bytes * kExperts; ++i) {
            data[i] = std::uint8_t(random());
        }
        for (int e = 0; e < kExperts; ++e) { pointers.push_back(data + e * expert_bytes); }
        table.copy_from_host(pointers.data(), pointers.size() * sizeof(void*));
    }

    [[nodiscard]] ExpertBank view() const {
        return {.expert_bytes = bytes, .host = pointers, .table = table.p};
    }

    // Every entry's bytes equal the expert's, and `tail` zero bytes follow a cached one; returns
    // how many entries point at device memory.
    int verify(const std::string& label, std::size_t tail = 0) const {
        std::vector<const void*> entries(kExperts);
        table.copy_to_host(entries.data(), entries.size() * sizeof(void*));
        int cached = 0;
        std::vector<std::uint8_t> bytes_read(static_cast<std::size_t>(bytes) + tail);
        for (int e = 0; e < kExperts; ++e) {
            if (entries[e] == pointers[e]) { continue; }
            ++cached;
            require(cudaMemcpy(bytes_read.data(), entries[e], bytes_read.size(),
                               cudaMemcpyDeviceToHost) == cudaSuccess,
                    label + ": a cached entry is not readable");
            require(std::memcmp(bytes_read.data(), pointers[e], std::size_t(bytes)) == 0,
                    label + ": a slot differs from its expert's bytes");
            require(std::all_of(bytes_read.begin() + bytes, bytes_read.end(),
                                [](std::uint8_t b) { return b == 0; }),
                    label + ": a cached down matrix is not followed by zeros");
        }
        return cached;
    }
};

int run() {
    DeviceContext device;
    std::mt19937 random(9100u);
    // Two layers; their banks have different expert sizes, as mixed block types do.
    Bank g0(4096, random), u0(4096, random), d0(2048, random);
    Bank g1(1536, random), u1(3072, random), d1(1024, random);
    std::vector<ExpertCacheLayer> layers = {
        {.rank = 0, .stream = device.stream, .gate = g0.view(), .up = u0.view(), .down = d0.view()},
        {.rank = 0, .stream = device.stream, .gate = g1.view(), .up = u1.view(), .down = d1.view()},
    };
    // 2 * 102,400 bytes over slots of 4096 + 4096 + 2304 bytes for layer 0 and 1536 + 3072 + 1280
    // for layer 1 (each down followed by its 256 zero bytes, aligned): 16,384 bytes a slot pair,
    // so twelve slots each.
    const std::vector<std::uint64_t> budget = {2 * 102400};
    ExpertCache cache(device, layers, budget);
    const auto slots = cache.stats().slots;
    require(slots == 12 + 12,
            "every layer gets the same number of slots: " + std::to_string(slots));

    // Layer 0 routes heavily to experts 5 and 7; layer 1 spreads.
    std::vector<std::int32_t> hot;
    for (int t = 0; t < 40; ++t) {
        hot.push_back(5);
        hot.push_back(7);
        hot.push_back(t % kExperts);
    }
    for (int round = 0; round < 6; ++round) {
        cache.observe(0, hot, 40);
        std::vector<std::int32_t> spread;
        for (int i = 0; i < 120; ++i) { spread.push_back(int(random() % kExperts)); }
        cache.observe(1, spread, 40);
        cache.rebalance(round % 2 == 0 ? (1u << 20) : 30000);
        device.synchronize();
        const int cached0 = g0.verify("gate 0") + 0;
        require(u0.verify("up 0") == cached0 && d0.verify("down 0", 256) == cached0,
                "a layer's three banks cache the same experts");
        require(cached0 <= 12, "layer 0 keeps within its slots");
        const int cached1 = g1.verify("gate 1");
        require(u1.verify("up 1") == cached1 && d1.verify("down 1", 256) == cached1,
                "layer 1's banks agree");
        require(cached1 <= 12, "layer 1 keeps within its slots");
    }
    std::vector<const void*> entries(kExperts);
    g0.table.copy_to_host(entries.data(), entries.size() * sizeof(void*));
    require(entries[5] != g0.pointers[5] && entries[7] != g0.pointers[7],
            "the most routed experts are cached");
    const auto stats = cache.stats();
    require(stats.routes == 6 * 240 && stats.hits > 0 && stats.admitted > 0,
            "the cache counts routes, hits and admissions");

    // Layer 1 admits on its own from now on: an admission takes the least recently used slot
    // whose expert the call does not use, and the caller copies the expert in and patches the
    // tables, as the expert misses do.
    cache.hand_over(1);
    const auto admit = [&](std::int32_t expert, const std::vector<std::uint32_t>& last_use,
                           const std::vector<unsigned char>& in_use) {
        ExpertCache::Admitted out;
        if (!cache.admit(1, expert, last_use, in_use, out)) { return out; }
        const Bank* banks[3] = {&g1, &u1, &d1};
        for (int k = 0; k < 3; ++k) {
            require(cudaMemcpy(out.slot[k], banks[k]->pointers[expert], banks[k]->bytes,
                               cudaMemcpyHostToDevice) == cudaSuccess,
                    "copy an admitted expert");
            auto* table = static_cast<const void**>(banks[k]->table.p);
            require(cudaMemcpy(table + expert, &out.slot[k], sizeof(void*),
                               cudaMemcpyHostToDevice) == cudaSuccess,
                    "patch an admitted entry");
            if (out.victim >= 0) {
                require(cudaMemcpy(table + out.victim, &out.victim_host[k], sizeof(void*),
                                   cudaMemcpyHostToDevice) == cudaSuccess,
                        "patch a victim entry");
            }
        }
        return out;
    };
    // Fill every free slot first, then evict by recency.
    std::vector<std::uint32_t> last_use(kExperts, 0);
    std::vector<std::int32_t> held;
    for (int e = 0; e < kExperts; ++e) {
        if (cache.cached(1, 0, e) != nullptr) { held.push_back(e); }
    }
    for (std::size_t i = 0; i < held.size(); ++i) { last_use[std::size_t(held[i])] = 100 + std::uint32_t(i); }
    std::int32_t newcomer = 0;
    while (cache.cached(1, 0, newcomer) != nullptr) { ++newcomer; }
    std::vector<unsigned char> in_use(kExperts, 0);
    in_use[std::size_t(newcomer)] = 1;
    in_use[std::size_t(held.front())] = 1; // the oldest is in use: the next oldest goes
    const auto admitted = admit(newcomer, last_use, in_use);
    require(admitted.slot[0] != nullptr, "an admission into a full layer finds a slot");
    if (held.size() == 12) {
        require(admitted.victim == held[1],
                "the least recently used expert the call does not use leaves");
    }
    require(cache.cached(1, 0, newcomer) == admitted.slot[0], "the newcomer is cached");
    std::vector<std::int32_t> again(10, newcomer);
    const auto before = cache.stats();
    cache.observe(1, again, 1);
    require(cache.stats().hits == before.hits, "a fresh admission's routes are misses");
    cache.observe(1, again, 1);
    require(cache.stats().hits == before.hits + 10, "afterwards they are hits");
    device.synchronize();
    require(g1.verify("gate 1 admitted") <= 12 && d1.verify("down 1 admitted", 256) <= 12,
            "admitted slots hold their experts");

    // Frequency outranks recency: once every route count has decayed, an expert routed often
    // since stays although it was used longest ago, and the least recently used of the rest goes.
    cache.observe(1, {}, 100000);
    std::vector<std::int32_t> by_age;
    for (int e = 0; e < kExperts; ++e) {
        if (cache.cached(1, 0, e) != nullptr) { by_age.push_back(e); }
    }
    for (std::size_t i = 0; i < by_age.size(); ++i) {
        last_use[std::size_t(by_age[i])] = 1000 + std::uint32_t(i);
    }
    cache.observe(1, std::vector<std::int32_t>(8, by_age.front()), 1);
    std::int32_t third = 0;
    while (cache.cached(1, 0, third) != nullptr) { ++third; }
    std::fill(in_use.begin(), in_use.end(), 0);
    in_use[std::size_t(third)] = 1;
    const auto tiered = admit(third, last_use, in_use);
    require(by_age.size() == 12 && tiered.victim == by_age[1],
            "the least recently used expert of the lowest frequency tier leaves");
    require(cache.cached(1, 0, by_age.front()) != nullptr, "the often routed expert stays");

    // Growth after startup: both layers gain the same number of free slots in a second block,
    // which the next admissions and rebalances fill.
    const auto ranges_before = cache.storage(0);
    require(ranges_before[1].first == ranges_before[1].second, "no second block before growth");
    const auto added = cache.grow(std::vector<std::uint64_t>{4 * 16384});
    require(added.size() == 1 && added[0] == 4 * 16384 && cache.stats().slots == 24 + 8,
            "growth gives each growing layer the same slots: " + std::to_string(cache.stats().slots));
    const auto ranges_after = cache.storage(1);
    require(ranges_after[1].second - ranges_after[1].first == 4 * 5888, "layer 1's second block");
    std::int32_t next = 0;
    while (cache.cached(1, 0, next) != nullptr) { ++next; }
    std::fill(in_use.begin(), in_use.end(), 0);
    const auto grown = admit(next, last_use, in_use);
    const auto at    = reinterpret_cast<std::uintptr_t>(grown.slot[0]);
    require(grown.victim < 0 && at >= ranges_after[1].first && at < ranges_after[1].second,
            "a grown layer admits into a free slot of its second block");
    for (int round = 0; round < 4; ++round) {
        cache.observe(0, hot, 40);
        cache.rebalance(1u << 20);
    }
    device.synchronize();
    const int cached0 = g0.verify("gate 0 grown");
    require(cached0 <= 16 && u0.verify("up 0 grown") == cached0 &&
                d0.verify("down 0 grown", 256) == cached0,
            "a grown rebalancing layer stays consistent");
    require(g1.verify("gate 1 grown") <= 16 && d1.verify("down 1 grown", 256) <= 16,
            "a grown handed-over layer stays consistent");
    return 0;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        run();
        std::cout << "PASS qwen4_exp expert cache\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
