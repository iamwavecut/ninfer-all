#include "runtime/engine/kv_capacity.h"

#include "ninfer/engine_options.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::runtime {
namespace {

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

void validate_curve(const SequenceCapacityCurve& curve) {
    if (curve.main_page_tokens == 0 || curve.minimum_main_page_groups == 0 ||
        curve.minimum_main_page_groups > curve.maximum_main_page_groups ||
        curve.minimum_device_reservation_bytes == 0) {
        throw std::invalid_argument("sequence capacity curve is invalid");
    }
    for (const RankCapacityCurve& rank : curve.extra_ranks) {
        if (rank.minimum_device_reservation_bytes == 0) {
            throw std::invalid_argument("sequence capacity curve has an empty device reservation");
        }
    }
    if (curve.minimum_main_page_groups < curve.maximum_main_page_groups &&
        curve.bytes_per_additional_main_page_group == 0) {
        throw std::invalid_argument("expandable sequence capacity curve has zero byte stride");
    }
    // The slots that may be given up are part of each device's smallest reservation.
    if (curve.minimum_state_slots > curve.state_slots) {
        throw std::invalid_argument("sequence capacity curve has fewer state slots than its floor");
    }
    const std::size_t spare = curve.state_slots - curve.minimum_state_slots;
    const auto holds_spare_slots = [&](std::size_t minimum, std::size_t per_slot) {
        return per_slot == 0 || (spare <= (minimum - 1) / per_slot);
    };
    bool held = holds_spare_slots(curve.minimum_device_reservation_bytes, curve.bytes_per_state_slot);
    for (const RankCapacityCurve& rank : curve.extra_ranks) {
        held = held && holds_spare_slots(rank.minimum_device_reservation_bytes,
                                         rank.bytes_per_state_slot);
    }
    if (!held) {
        throw std::invalid_argument(
            "sequence capacity curve's state slots exceed its smallest reservation");
    }
}

std::uint32_t explicit_page_groups(const KvCapacityPolicy& policy,
                                   const SequenceCapacityCurve& curve) {
    if (policy.explicit_tokens == 0) {
        throw std::invalid_argument("explicit KV capacity must be positive");
    }
    const std::uint64_t pages =
        1ULL + (static_cast<std::uint64_t>(policy.explicit_tokens) - 1ULL) / curve.main_page_tokens;
    if (pages + curve.resident_main_pages > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("explicit KV page count exceeds uint32");
    }
    return static_cast<std::uint32_t>(pages + curve.resident_main_pages);
}

bool has_state_slot_cost(const SequenceCapacityCurve& curve) {
    if (curve.bytes_per_state_slot != 0) { return true; }
    return std::ranges::any_of(curve.extra_ranks, [](const RankCapacityCurve& rank) {
        return rank.bytes_per_state_slot != 0;
    });
}

// The curve with `slots` of its state slots given up: every device's reservation falls by what
// they held there.
SequenceCapacityCurve without_state_slots(const SequenceCapacityCurve& curve,
                                          std::uint32_t slots) {
    SequenceCapacityCurve out = curve;
    out.state_slots -= slots;
    out.minimum_device_reservation_bytes -= std::size_t{slots} * curve.bytes_per_state_slot;
    for (RankCapacityCurve& rank : out.extra_ranks) {
        rank.minimum_device_reservation_bytes -= std::size_t{slots} * rank.bytes_per_state_slot;
    }
    return out;
}

// Whether the capacity asked for fits every device: the explicit page count, or under automatic
// capacity the smallest reservation beside the headroom. An explicit count outside the curve is
// left to the resolver to reject.
bool reservation_fits(const KvCapacityPolicy& policy, const SequenceCapacityCurve& curve,
                      std::size_t available_runtime_bytes,
                      std::span<const std::size_t> extra_rank_available_bytes) {
    if (policy.mode == KvCapacityMode::Automatic) {
        const std::size_t headroom = policy.automatic_headroom_bytes;
        if (available_runtime_bytes < headroom ||
            available_runtime_bytes - headroom < curve.minimum_device_reservation_bytes) {
            return false;
        }
        for (std::size_t index = 0; index < curve.extra_ranks.size(); ++index) {
            const std::size_t available = extra_rank_available_bytes[index];
            if (available < headroom ||
                available - headroom < curve.extra_ranks[index].minimum_device_reservation_bytes) {
                return false;
            }
        }
        return true;
    }
    if (policy.mode != KvCapacityMode::Explicit || policy.explicit_tokens == 0) { return true; }
    const std::uint32_t pages = explicit_page_groups(policy, curve);
    if (pages < curve.minimum_main_page_groups || pages > curve.maximum_main_page_groups) {
        return true;
    }
    if (curve.reservation_bytes(pages) > available_runtime_bytes) { return false; }
    for (std::size_t index = 0; index < curve.extra_ranks.size(); ++index) {
        if (curve.extra_rank_reservation_bytes(index, pages) > extra_rank_available_bytes[index]) {
            return false;
        }
    }
    return true;
}

std::string mib_text(std::size_t bytes) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.1f MiB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    return text;
}

// What a capacity failure tells the user to change: the context and KV sizes, and the state slots
// when fewer of them would be accepted. `yielded` default slots were already given up.
std::string capacity_hint(const KvCapacityPolicy& policy, const SequenceCapacityCurve& curve,
                          std::uint32_t yielded) {
    std::string out = policy.mode == KvCapacityMode::Automatic
                          ? "; lower --max-context or --kv-headroom-mib"
                          : "; lower --kv-capacity or --max-context";
    if (curve.state_slot_option == nullptr || curve.bytes_per_state_slot == 0) { return out; }
    if (yielded != 0) {
        return out + " (the default " + curve.state_slot_option + " was already lowered from " +
               std::to_string(curve.state_slots + yielded) + " to " +
               std::to_string(curve.state_slots) + ")";
    }
    if (curve.state_slots > curve.minimum_state_slots) {
        out += ", or set " + std::string(curve.state_slot_option) + " below " +
               std::to_string(curve.state_slots) + " (minimum " +
               std::to_string(curve.minimum_state_slots) + "; each slot holds " +
               mib_text(curve.bytes_per_state_slot) + ")";
    }
    return out;
}

KvCapacityResolution resolve_on_curve(const KvCapacityPolicy& policy,
                                      const SequenceCapacityCurve& curve,
                                      std::size_t available_runtime_bytes,
                                      std::span<const std::size_t> extra_rank_available_bytes,
                                      const std::string& hint) {
    const auto device_name = [](std::size_t rank) { return "device " + std::to_string(rank); };

    std::uint32_t pages         = curve.minimum_main_page_groups;
    std::size_t capacity_budget = available_runtime_bytes;
    std::size_t binding_rank_of_pages = 0;
    switch (policy.mode) {
    case KvCapacityMode::Explicit:
        if (policy.automatic_headroom_bytes != 0) {
            throw std::invalid_argument("explicit KV capacity must not carry automatic headroom");
        }
        pages = explicit_page_groups(policy, curve);
        break;
    case KvCapacityMode::Automatic:
        if (available_runtime_bytes < policy.automatic_headroom_bytes) {
            throw std::invalid_argument(
                "automatic KV headroom requires " +
                std::to_string(policy.automatic_headroom_bytes) + " bytes, but only " +
                std::to_string(available_runtime_bytes) + " bytes are available after weights" +
                hint);
        }
        capacity_budget -= policy.automatic_headroom_bytes;
        if (capacity_budget < curve.minimum_device_reservation_bytes) {
            throw std::invalid_argument(
                "minimum Engine runtime reservation requires " +
                std::to_string(curve.minimum_device_reservation_bytes) + " bytes in addition to " +
                std::to_string(policy.automatic_headroom_bytes) +
                " bytes of automatic headroom, but only " +
                std::to_string(available_runtime_bytes) + " bytes are available after weights" +
                (curve.extra_ranks.empty() ? std::string() : " on " + device_name(0)) + hint);
        }
        if (curve.minimum_main_page_groups < curve.maximum_main_page_groups) {
            // The capacity is the smallest any device allows. Start from the primary device.
            const std::size_t additional =
                (capacity_budget - curve.minimum_device_reservation_bytes) /
                curve.bytes_per_additional_main_page_group;
            std::uint64_t candidate =
                static_cast<std::uint64_t>(curve.minimum_main_page_groups) + additional;
            for (std::size_t index = 0; index < curve.extra_ranks.size(); ++index) {
                const RankCapacityCurve& rank = curve.extra_ranks[index];
                const std::size_t available   = extra_rank_available_bytes[index];
                if (available < policy.automatic_headroom_bytes ||
                    available - policy.automatic_headroom_bytes <
                        rank.minimum_device_reservation_bytes) {
                    throw std::invalid_argument(
                        "minimum Engine runtime reservation requires " +
                        std::to_string(rank.minimum_device_reservation_bytes) +
                        " bytes in addition to " +
                        std::to_string(policy.automatic_headroom_bytes) +
                        " bytes of automatic headroom, but only " + std::to_string(available) +
                        " bytes are available after weights on " + device_name(index + 1) + hint);
                }
                // A device that holds no KV does not limit how many pages fit.
                if (rank.bytes_per_additional_main_page_group == 0) { continue; }
                const std::size_t rank_additional =
                    (available - policy.automatic_headroom_bytes -
                     rank.minimum_device_reservation_bytes) /
                    rank.bytes_per_additional_main_page_group;
                const std::uint64_t rank_candidate =
                    static_cast<std::uint64_t>(curve.minimum_main_page_groups) + rank_additional;
                if (rank_candidate < candidate) {
                    candidate            = rank_candidate;
                    binding_rank_of_pages = index + 1;
                }
            }
            pages = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(candidate, curve.maximum_main_page_groups));
        }
        break;
    default:
        throw std::invalid_argument("unknown KV capacity policy");
    }

    const std::size_t reservation = curve.reservation_bytes(pages);
    if (reservation > capacity_budget) {
        throw std::invalid_argument("requested Engine runtime reservation requires " +
                                    std::to_string(reservation) + " bytes, but only " +
                                    std::to_string(capacity_budget) +
                                    " bytes are available for runtime capacity" +
                                    (curve.extra_ranks.empty() ? std::string()
                                                               : " on " + device_name(0)) +
                                    hint);
    }
    std::vector<std::size_t> extra_reservations;
    for (std::size_t index = 0; index < curve.extra_ranks.size(); ++index) {
        const std::size_t need = curve.extra_rank_reservation_bytes(index, pages);
        // Explicit capacity is not reduced by headroom on any device; automatic already was.
        const std::size_t budget =
            policy.mode == KvCapacityMode::Automatic
                ? extra_rank_available_bytes[index] - policy.automatic_headroom_bytes
                : extra_rank_available_bytes[index];
        if (need > budget) {
            throw std::invalid_argument("requested Engine runtime reservation requires " +
                                        std::to_string(need) + " bytes, but only " +
                                        std::to_string(budget) +
                                        " bytes are available for runtime capacity on " +
                                        device_name(index + 1) + hint);
        }
        extra_reservations.push_back(need);
    }

    return KvCapacityResolution{
        .mode                                 = policy.mode,
        .main_page_groups                     = pages,
        .maximum_main_page_groups             = curve.maximum_main_page_groups,
        .resolved_tokens                      = curve.resolved_tokens(pages),
        .minimum_runtime_reservation_bytes    = curve.minimum_device_reservation_bytes,
        .bytes_per_additional_main_page_group = curve.bytes_per_additional_main_page_group,
        .runtime_reservation_bytes            = reservation,
        .extra_rank_reservation_bytes         = std::move(extra_reservations),
        .binding_rank                         = binding_rank_of_pages,
        .available_after_weights_bytes        = available_runtime_bytes,
        .automatic_headroom_bytes             = policy.automatic_headroom_bytes,
        .planned_slack_bytes                  = available_runtime_bytes - reservation,
    };
}

} // namespace

std::size_t SequenceCapacityCurve::reservation_bytes(std::uint32_t main_page_groups) const {
    validate_curve(*this);
    if (main_page_groups < minimum_main_page_groups ||
        main_page_groups > maximum_main_page_groups) {
        throw std::invalid_argument("Main KV page count is outside the target capacity curve");
    }
    const std::size_t additional =
        static_cast<std::size_t>(main_page_groups - minimum_main_page_groups);
    return checked_add(minimum_device_reservation_bytes,
                       checked_mul(additional, bytes_per_additional_main_page_group,
                                   "sequence capacity increment overflows size_t"),
                       "sequence capacity reservation overflows size_t");
}

std::size_t SequenceCapacityCurve::extra_rank_reservation_bytes(
    std::size_t extra_rank, std::uint32_t main_page_groups) const {
    validate_curve(*this);
    if (extra_rank >= extra_ranks.size()) {
        throw std::out_of_range("capacity curve has no such device");
    }
    if (main_page_groups < minimum_main_page_groups ||
        main_page_groups > maximum_main_page_groups) {
        throw std::invalid_argument("Main KV page count is outside the target capacity curve");
    }
    const RankCapacityCurve& rank = extra_ranks[extra_rank];
    const std::size_t additional =
        static_cast<std::size_t>(main_page_groups - minimum_main_page_groups);
    return checked_add(rank.minimum_device_reservation_bytes,
                       checked_mul(additional, rank.bytes_per_additional_main_page_group,
                                   "sequence capacity increment overflows size_t"),
                       "sequence capacity reservation overflows size_t");
}

std::uint32_t SequenceCapacityCurve::resolved_tokens(std::uint32_t main_page_groups) const {
    validate_curve(*this);
    if (main_page_groups < minimum_main_page_groups ||
        main_page_groups > maximum_main_page_groups) {
        throw std::invalid_argument("Main KV page count is outside the target capacity curve");
    }
    const std::uint64_t tokens = static_cast<std::uint64_t>(main_page_groups) * main_page_tokens;
    if (tokens > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("resolved KV token capacity exceeds uint32");
    }
    return static_cast<std::uint32_t>(tokens);
}

bool kv_capacity_fits(const KvCapacityPolicy& policy, const SequenceCapacityCurve& curve,
                      std::size_t available_runtime_bytes,
                      std::span<const std::size_t> extra_rank_available_bytes) {
    validate_curve(curve);
    if (extra_rank_available_bytes.size() != curve.extra_ranks.size()) {
        throw std::invalid_argument("free memory was not given for every device in the plan");
    }
    return reservation_fits(policy, curve, available_runtime_bytes, extra_rank_available_bytes);
}

KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                         const SequenceCapacityCurve& curve,
                                         std::size_t available_runtime_bytes,
                                         std::span<const std::size_t> extra_rank_available_bytes) {
    validate_curve(curve);
    if (extra_rank_available_bytes.size() != curve.extra_ranks.size()) {
        throw std::invalid_argument("free memory was not given for every device in the plan");
    }
    // Default state slots yield to the capacity asked for, the fewest that make it fit; when even
    // the floor does not, the failure below is priced at the floor.
    std::uint32_t yielded = 0;
    if (curve.state_slots_automatic && has_state_slot_cost(curve)) {
        const std::uint32_t spare = curve.state_slots - curve.minimum_state_slots;
        while (yielded < spare &&
               !reservation_fits(policy, without_state_slots(curve, yielded),
                                 available_runtime_bytes, extra_rank_available_bytes)) {
            ++yielded;
        }
    }
    const SequenceCapacityCurve fitted =
        yielded == 0 ? curve : without_state_slots(curve, yielded);
    KvCapacityResolution out =
        resolve_on_curve(policy, fitted, available_runtime_bytes, extra_rank_available_bytes,
                         capacity_hint(policy, fitted, yielded));
    out.yielded_state_slots = yielded;
    return out;
}

std::vector<std::int64_t>
kv_capacity_spare(const KvCapacityPolicy& policy, const SequenceCapacityCurve& curve,
                  std::uint32_t automatic_page_groups, std::size_t available_runtime_bytes,
                  std::span<const std::size_t> extra_rank_available_bytes) {
    validate_curve(curve);
    if (extra_rank_available_bytes.size() != curve.extra_ranks.size()) {
        throw std::invalid_argument("free memory was not given for every device in the plan");
    }
    const std::uint64_t asked =
        policy.mode == KvCapacityMode::Automatic
            ? static_cast<std::uint64_t>(automatic_page_groups) + curve.resident_main_pages
            : explicit_page_groups(policy, curve);
    const auto pages = static_cast<std::uint32_t>(
        std::clamp<std::uint64_t>(asked, curve.minimum_main_page_groups,
                                  curve.maximum_main_page_groups));
    const std::size_t headroom =
        policy.mode == KvCapacityMode::Automatic ? policy.automatic_headroom_bytes : 0;
    const auto left = [&](std::size_t available, std::size_t reservation) {
        return static_cast<std::int64_t>(available) - static_cast<std::int64_t>(reservation) -
               static_cast<std::int64_t>(headroom);
    };
    std::vector<std::int64_t> out{left(available_runtime_bytes, curve.reservation_bytes(pages))};
    for (std::size_t index = 0; index < curve.extra_ranks.size(); ++index) {
        out.push_back(left(extra_rank_available_bytes[index],
                           curve.extra_rank_reservation_bytes(index, pages)));
    }
    return out;
}

} // namespace ninfer::runtime
