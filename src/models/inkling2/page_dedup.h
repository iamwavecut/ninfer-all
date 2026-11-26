// Content-addressed sharing of byte-identical page groups across the lanes of one pool
// (--kv-page-dedup): sessions with the same leading tokens (a shared system prompt) store the
// prefix's page groups once and map their execution-table rows at the same physical page,
// reference-counted, and a write into a shared group copies first (copy-on-write).
//
// The hash nominates a candidate only: hash_of_group runs FNV-1a 64 over the group's token ids
// and its first position (host-side, a few dozen bytes, no device traffic, no extra pass over
// the encoded planes), and share_completed_group verifies the candidate byte for byte on the
// device before anything is shared. Neither a 64-bit collision nor a break of the
// same-tokens-same-positions-store-identical-bytes premise can corrupt a page — a mismatch keeps
// the fresh page and is not inserted (keep-first).
//
// Race safety: one lane's calls run at a time on the executor's stream and the table lives on
// the host, so the map needs no lock. Every device effect of a sharing decision — the byte
// compare, the row publish, the freed page's later reuse — orders on that same stream, and the
// pool's generations catch any stale handle.
//
// The table owns the first owner's lease: the shared page returns to the pool only when its last
// reference drops, however long ago the first owner retired. A reference drops with unshare,
// whose lease the caller frees against whatever reservation it holds (a reservation counts pages
// forward only, so which lane's grant retires the page is free). The caller publishes the lane's
// row itself, after the call's offers and copies, from its LaneGroups.

#pragma once

#include "core/paged_kv_cache.h"

#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace ninfer::models::inkling2::page_dedup {

// FNV-1a 64 over the group's token ids and its first position.
[[nodiscard]] std::uint64_t hash_of_group(std::span<const std::int32_t> tokens,
                                          std::uint32_t first_position) noexcept;

// One lane's view of one logical page group: what its execution-table row names, and the shared
// entry it maps (empty while the lane owns the page).
struct LaneGroup {
    DeviceKVPageHandle mapped{};
    std::optional<std::uint64_t> shared; // the table entry's hash, when the lane maps one
};

enum class ShareOutcome {
    entry_took_lease,     // a miss: the table now owns the fresh page (the caller's slot is null)
    shared_onto_existing, // a verified match: the fresh page went back to the reservation
    kept_fresh,           // a hash match with different bytes: nothing changed, the page stays
};

class PageSharing {
public:
    struct Entry {
        DeviceKVPageLease page; // the shared physical page; the table owns it
        std::uint32_t refs = 1; // lanes mapping it, the first owner included
    };

    struct Stats {
        std::uint64_t offers     = 0; // completed groups offered
        std::uint64_t shared     = 0; // offers mapped onto an existing, byte-equal page
        std::uint64_t collisions = 0; // hash matches whose bytes differed (kept fresh)
        std::uint64_t copies     = 0; // copy-on-write copies of shared pages
    };

    [[nodiscard]] const Entry* find(std::uint64_t hash) const;
    // Drops one reference (a lane's release); when it was the last, the lease moves out for the
    // caller to free against a reservation (an empty lease while references remain).
    [[nodiscard]] DeviceKVPageLease unshare(std::uint64_t hash);
    [[nodiscard]] std::uint64_t groups() const noexcept { return entries_.size(); }
    // Entries two or more lanes map: the pages the sharing holds in physical memory once.
    [[nodiscard]] std::uint64_t pages_shared() const noexcept {
        std::uint64_t pages = 0;
        for (const auto& [hash, entry] : entries_) {
            if (entry.refs >= 2) { ++pages; }
        }
        return pages;
    }
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

private:
    friend ShareOutcome share_completed_group(DeviceKVPagePool&, PageSharing&, std::uint64_t,
                                              DeviceKVPageLease&&, DeviceKVPageReservation&,
                                              LaneGroup&, RankStreams);
    friend void copy_on_write_group(DeviceKVPagePool&, PageSharing&, std::uint64_t,
                                    DeviceKVPageReservation&, DeviceKVPageLease&, LaneGroup&,
                                    RankStreams);

    // Takes ownership of the first owner's lease.
    void insert(std::uint64_t hash, DeviceKVPageLease page);
    void acquire(std::uint64_t hash); // another lane maps the page

    std::unordered_map<std::uint64_t, Entry> entries_;
    Stats stats_{};
};

// Offers a completed group the lane just wrote: `lease` is the lane's own page at this logical
// group (the caller's slot; it is emptied when the outcome takes or frees the page). On a
// verified hash match the lane maps the shared page; on a miss the table takes the lease; on a
// byte mismatch the lane keeps its page. Publishes nothing; the caller republishes the row.
ShareOutcome share_completed_group(DeviceKVPagePool& pool, PageSharing& sharing,
                                   std::uint64_t hash, DeviceKVPageLease&& lease,
                                   DeviceKVPageReservation& reservation, LaneGroup& group,
                                   RankStreams streams);

// Ensures the lane's group maps a page the lane owns: its shared page is copied to a fresh one
// from `reservation` (which must hold a page), the shared entry loses this lane's reference (the
// page returns to the pool when it was the last), and `lease_slot` (the caller's null slot at
// this group) receives the fresh lease.
void copy_on_write_group(DeviceKVPagePool& pool, PageSharing& sharing, std::uint64_t hash,
                         DeviceKVPageReservation& reservation, DeviceKVPageLease& lease_slot,
                         LaneGroup& group, RankStreams streams);

} // namespace ninfer::models::inkling2::page_dedup
