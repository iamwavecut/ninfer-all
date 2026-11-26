#include "models/inkling2/page_dedup.h"

#include <stdexcept>

namespace ninfer::models::inkling2::page_dedup {

std::uint64_t hash_of_group(std::span<const std::int32_t> tokens,
                            std::uint32_t first_position) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    const auto mix = [&hash](const std::uint8_t* bytes, std::size_t count) {
        for (std::size_t i = 0; i < count; ++i) {
            hash ^= bytes[i];
            hash *= 1099511628211ULL;
        }
    };
    const auto first = static_cast<std::uint32_t>(first_position);
    mix(reinterpret_cast<const std::uint8_t*>(&first), sizeof(first));
    mix(reinterpret_cast<const std::uint8_t*>(tokens.data()), tokens.size_bytes());
    return hash;
}

const PageSharing::Entry* PageSharing::find(std::uint64_t hash) const {
    const auto it = entries_.find(hash);
    return it != entries_.end() ? &it->second : nullptr;
}

void PageSharing::insert(std::uint64_t hash, DeviceKVPageLease page) {
    Entry entry;
    entry.page = std::move(page);
    entries_.emplace(hash, std::move(entry));
}

void PageSharing::acquire(std::uint64_t hash) {
    const auto it = entries_.find(hash);
    if (it == entries_.end()) { throw std::out_of_range("PageSharing: no entry for the hash"); }
    ++it->second.refs;
}

DeviceKVPageLease PageSharing::unshare(std::uint64_t hash) {
    const auto it = entries_.find(hash);
    if (it == entries_.end()) { throw std::out_of_range("PageSharing: no entry for the hash"); }
    Entry& entry = it->second;
    if (--entry.refs != 0) { return DeviceKVPageLease{}; }
    DeviceKVPageLease page = std::move(entry.page);
    entries_.erase(it);
    return page;
}

ShareOutcome share_completed_group(DeviceKVPagePool& pool, PageSharing& sharing,
                                   std::uint64_t hash, DeviceKVPageLease&& lease,
                                   DeviceKVPageReservation& reservation, LaneGroup& group,
                                   RankStreams streams) {
    ++sharing.stats_.offers;
    const DeviceKVPageHandle fresh = lease.handle();
    if (const PageSharing::Entry* hit = sharing.find(hash)) {
        if (pool.pages_equal(fresh, hit->page.handle(), streams)) {
            sharing.acquire(hash);
            pool.dematerialize_one(reservation, std::move(lease));
            group.mapped = hit->page.handle();
            group.shared = hash;
            ++sharing.stats_.shared;
            return ShareOutcome::shared_onto_existing;
        }
        // A collision, or the same-tokens-same-bytes premise broken: keep the lane's own page
        // and do not insert (keep-first).
        ++sharing.stats_.collisions;
        return ShareOutcome::kept_fresh;
    }
    sharing.insert(hash, std::move(lease));
    group.mapped = fresh;
    group.shared = hash;
    return ShareOutcome::entry_took_lease;
}

void copy_on_write_group(DeviceKVPagePool& pool, PageSharing& sharing, std::uint64_t hash,
                         DeviceKVPageReservation& reservation, DeviceKVPageLease& lease_slot,
                         LaneGroup& group, RankStreams streams) {
    if (!group.shared.has_value() || *group.shared != hash) {
        throw std::invalid_argument("inkling2: copy-on-write needs the lane's shared group");
    }
    DeviceKVPageLease fresh = pool.materialize_one(reservation);
    pool.copy_page(group.mapped, fresh.handle(), streams);
    DeviceKVPageLease retirable = sharing.unshare(hash);
    if (retirable.valid()) { pool.dematerialize_one(reservation, std::move(retirable)); }
    group.mapped = fresh.handle();
    group.shared.reset();
    lease_slot = std::move(fresh);
    ++sharing.stats_.copies;
}

} // namespace ninfer::models::inkling2::page_dedup
