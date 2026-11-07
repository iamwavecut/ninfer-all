#include "ninfer/engine_options.h"
#include "runtime/engine/kv_capacity.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    const ninfer::runtime::SequenceCapacityCurve curve{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 6,
        .minimum_device_reservation_bytes     = 1000,
        .bytes_per_additional_main_page_group = 128,
    };

    const auto automatic =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 1360);
    failures +=
        check(automatic.main_page_groups == 4 && automatic.resolved_tokens == 256 &&
                  automatic.runtime_reservation_bytes == 1256 &&
                  automatic.automatic_headroom_bytes == 50 && automatic.planned_slack_bytes == 104,
              "automatic KV capacity did not select the largest fitting page count");

    const auto capped =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 10000);
    failures += check(capped.main_page_groups == 6 && capped.resolved_tokens == 384,
                      "automatic KV capacity exceeded or missed the target maximum");

    const auto explicit_capacity = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129), curve, 1200);
    failures +=
        check(explicit_capacity.main_page_groups == 3 && explicit_capacity.resolved_tokens == 192 &&
                  explicit_capacity.runtime_reservation_bytes == 1128,
              "explicit KV capacity did not use page-aligned token semantics");

    // Pages an injected graft holds for good come on top of the request: an explicit capacity still
    // leaves the asked-for tokens free, and the range moves up by the same amount.
    ninfer::runtime::SequenceCapacityCurve resident = curve;
    resident.minimum_main_page_groups += 3;
    resident.maximum_main_page_groups += 3;
    resident.resident_main_pages = 3;
    const auto with_resident = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129), resident, 2000);
    failures += check(with_resident.main_page_groups == 3 + 3 &&
                          with_resident.resolved_tokens == (3 + 3) * 64,
                      "explicit KV capacity did not add the resident graft pages");
    const auto resident_auto = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(50), resident, 10000);
    failures += check(resident_auto.main_page_groups == 6 + 3,
                      "automatic KV capacity did not include the resident graft pages");

    bool insufficient_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve,
                                                   1049);
    } catch (const std::invalid_argument&) { insufficient_rejected = true; }
    failures += check(insufficient_rejected,
                      "automatic KV capacity accepted less than the minimum reservation");

    // A model split across devices: each holds its own layers' KV, so a page group costs each a
    // different amount, and the capacity is the smallest any one allows.
    ninfer::runtime::SequenceCapacityCurve split{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 20,
        .minimum_device_reservation_bytes     = 1000,
        .bytes_per_additional_main_page_group = 100,
        .extra_ranks                          = {{.minimum_device_reservation_bytes     = 500,
                                                  .bytes_per_additional_main_page_group = 300}},
    };
    const std::size_t split_available[] = {1500};
    // Device 0 alone would allow 11 pages; device 1 allows 2 + (1500 - 50 - 500) / 300 = 5.
    const auto both = ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50),
                                                           split, 2000, split_available);
    failures += check(both.main_page_groups == 5 && both.binding_rank == 1 &&
                          both.extra_rank_reservation_bytes.size() == 1 &&
                          both.extra_rank_reservation_bytes[0] == 500 + 3 * 300,
                      "the capacity was not limited by the tighter device");
    failures += check(split.extra_rank_reservation_bytes(0, 5) == 1400,
                      "a further device's reservation does not follow its own curve");

    // A device holding no KV does not limit the count, so the primary device decides.
    split.extra_ranks[0].bytes_per_additional_main_page_group = 0;
    const auto free_rank = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(50), split, 2000, split_available);
    failures += check(free_rank.main_page_groups == 11 && free_rank.binding_rank == 0,
                      "a device without KV constrained the capacity");
    split.extra_ranks[0].bytes_per_additional_main_page_group = 300;

    // A failure names the device that could not hold the plan.
    const auto names_device = [&](auto&& call, const char* device) {
        try {
            call();
        } catch (const std::invalid_argument& error) {
            return std::string(error.what()).find(device) != std::string::npos;
        }
        return false;
    };
    const std::size_t starved[] = {520};
    failures += check(names_device(
                          [&] {
                              (void)ninfer::runtime::resolve_kv_capacity(
                                  ninfer::KvCapacityPolicy::automatic(50), split, 2000, starved);
                          },
                          "device 1"),
                      "a device below its minimum reservation was not named");
    failures += check(names_device(
                          [&] {
                              (void)ninfer::runtime::resolve_kv_capacity(
                                  ninfer::KvCapacityPolicy::explicit_capacity(64 * 19), split, 100000,
                                  split_available);
                          },
                          "device 1"),
                      "an explicit capacity too large for one device was not named");
    failures += check(names_device(
                          [&] {
                              (void)ninfer::runtime::resolve_kv_capacity(
                                  ninfer::KvCapacityPolicy::automatic(50), split, 2000);
                          },
                          "every device"),
                      "missing free memory for a device was accepted");

    // What each device keeps beyond the reservation for the KV asked for. Automatic capacity at 4
    // pages: device 0 needs 1000 + 2 * 100 + 50 of 2000, device 1 needs 500 + 2 * 300 + 50 of 1000.
    const std::size_t tight[] = {1000};
    const auto spare = ninfer::runtime::kv_capacity_spare(ninfer::KvCapacityPolicy::automatic(50),
                                                          split, 4, 2000, tight);
    failures += check(spare.size() == 2 && spare[0] == 750 && spare[1] == -150,
                      "the spare of each device does not follow its curve and the headroom");
    // An explicit capacity asks for its own pages (3 for 129 tokens) and keeps no headroom.
    const auto explicit_spare = ninfer::runtime::kv_capacity_spare(
        ninfer::KvCapacityPolicy::explicit_capacity(129), split, 4, 2000, tight);
    failures += check(explicit_spare[0] == 2000 - 1100 && explicit_spare[1] == 1000 - 800,
                      "an explicit capacity's spare used the automatic page count or headroom");
    // The page count stays inside the curve, and resident graft pages come on top.
    const auto held = ninfer::runtime::kv_capacity_spare(ninfer::KvCapacityPolicy::automatic(0),
                                                         split, 100, 100000, tight);
    failures += check(held[1] == 1000 - (500 + 18 * 300),
                      "the spare was priced outside the curve's page range");
    // No automatic pages price the smallest reservation, the one every start needs.
    const auto least = ninfer::runtime::kv_capacity_spare(ninfer::KvCapacityPolicy::automatic(0),
                                                          split, 0, 2000, tight);
    failures += check(least[0] == 2000 - 1000 && least[1] == 1000 - 500,
                      "the spare at no automatic pages was not the smallest reservation's");
    const auto resident_spare = ninfer::runtime::kv_capacity_spare(
        ninfer::KvCapacityPolicy::automatic(0), resident, 4, 2000);
    failures += check(resident_spare.size() == 1 && resident_spare[0] == 2000 - (1000 + 2 * 128),
                      "automatic spare did not add the resident graft pages");

    // Default device state slots yield to the capacity asked for: the fewest slots that make it
    // fit, never below the cache's floor. Units are MiB so the hint's per-slot size reads exactly.
    constexpr std::size_t M = std::size_t{1} << 20;
    const ninfer::runtime::SequenceCapacityCurve slotted{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 20,
        .minimum_device_reservation_bytes     = 1000 * M,
        .bytes_per_additional_main_page_group = 50 * M,
        .state_slots                          = 3,
        .minimum_state_slots                  = 1,
        .state_slots_automatic                = true,
        .bytes_per_state_slot                 = 100 * M,
        .state_slot_option                    = "--device-snapshot-slots",
    };
    // 384 tokens are 6 pages: 1000 + 4 * 50 = 1200 MiB with all three slots.
    const auto explicit_six = ninfer::KvCapacityPolicy::explicit_capacity(384);
    const auto kept = ninfer::runtime::resolve_kv_capacity(explicit_six, slotted, 1200 * M);
    failures += check(kept.yielded_state_slots == 0 && kept.runtime_reservation_bytes == 1200 * M,
                      "default state slots yielded although the capacity fit beside them");
    const auto one_fewer = ninfer::runtime::resolve_kv_capacity(explicit_six, slotted, 1150 * M);
    failures += check(one_fewer.yielded_state_slots == 1 && one_fewer.main_page_groups == 6 &&
                          one_fewer.runtime_reservation_bytes == 1100 * M &&
                          one_fewer.minimum_runtime_reservation_bytes == 900 * M &&
                          one_fewer.planned_slack_bytes == 50 * M,
                      "default state slots did not yield the fewest slots that fit");
    const auto to_floor = ninfer::runtime::resolve_kv_capacity(explicit_six, slotted, 1000 * M);
    failures += check(to_floor.yielded_state_slots == 2 &&
                          to_floor.runtime_reservation_bytes == 1000 * M,
                      "default state slots did not yield down to the floor");

    const auto failure_text = [](auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument& error) { return std::string(error.what()); }
        return std::string();
    };
    const auto contains = [](const std::string& text, const char* part) {
        return text.find(part) != std::string::npos;
    };
    // Below the floor the slots stop yielding, and the failure prices the floor and says so.
    const std::string past_floor = failure_text([&] {
        (void)ninfer::runtime::resolve_kv_capacity(explicit_six, slotted, 999 * M);
    });
    failures += check(contains(past_floor, "requires 1048576000 bytes") &&
                          contains(past_floor, "--kv-capacity") &&
                          contains(past_floor, "--max-context") &&
                          contains(past_floor, "already lowered from 3 to 1"),
                      "a capacity past the slot floor did not fail with the floor and its knobs");

    // A given slot count does not yield: the failure names the option and what a slot holds.
    ninfer::runtime::SequenceCapacityCurve given = slotted;
    given.state_slots_automatic = false;
    const std::string kept_count = failure_text([&] {
        (void)ninfer::runtime::resolve_kv_capacity(explicit_six, given, 1150 * M);
    });
    failures += check(contains(kept_count, "requires 1258291200 bytes") &&
                          contains(kept_count, "--kv-capacity or --max-context") &&
                          contains(kept_count, "--device-snapshot-slots below 3") &&
                          contains(kept_count, "each slot holds 100.0 MiB"),
                      "a given slot count yielded, or its failure did not name the option");
    // At the floor there is nothing left to lower but the capacity.
    given.state_slots                      = 1;
    given.minimum_device_reservation_bytes = 800 * M;
    const std::string at_floor = failure_text([&] {
        (void)ninfer::runtime::resolve_kv_capacity(explicit_six, given, 900 * M);
    });
    failures += check(contains(at_floor, "--max-context") &&
                          !contains(at_floor, "--device-snapshot-slots"),
                      "a failure at the slot floor suggested fewer slots");

    // Automatic capacity keeps its default slots while one --max-context window fits beside the
    // headroom (the block cache gives up pages instead), and yields them when it does not.
    const auto automatic_kept = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(50 * M), slotted, 1150 * M);
    failures += check(automatic_kept.yielded_state_slots == 0 &&
                          automatic_kept.main_page_groups == 4,
                      "automatic capacity gave up slots although its minimum fit");
    const auto automatic_yield = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(50 * M), slotted, 1000 * M);
    failures += check(automatic_yield.yielded_state_slots == 1 &&
                          automatic_yield.main_page_groups == 3 &&
                          automatic_yield.runtime_reservation_bytes == 950 * M,
                      "automatic capacity did not yield slots to fit its minimum");
    const std::string automatic_short = failure_text([&] {
        (void)ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50 * M),
                                                   slotted, 800 * M);
    });
    failures += check(contains(automatic_short, "--kv-headroom-mib") &&
                          contains(automatic_short, "already lowered from 3 to 1"),
                      "automatic capacity past the slot floor did not name its knobs");

    // Every device must fit: a further device that holds state slots yields them too.
    ninfer::runtime::SequenceCapacityCurve slotted_split = slotted;
    slotted_split.extra_ranks = {{.minimum_device_reservation_bytes     = 500 * M,
                                  .bytes_per_additional_main_page_group = 10 * M,
                                  .bytes_per_state_slot                 = 120 * M}};
    const std::size_t short_rank[] = {300 * M};
    const auto split_yield = ninfer::runtime::resolve_kv_capacity(explicit_six, slotted_split,
                                                                  100000 * M, short_rank);
    // Device 1 needs 500 + 40 = 540 MiB with three slots, 300 MiB with one.
    failures += check(split_yield.yielded_state_slots == 2 &&
                          split_yield.extra_rank_reservation_bytes.size() == 1 &&
                          split_yield.extra_rank_reservation_bytes[0] == 300 * M &&
                          split_yield.runtime_reservation_bytes == 1000 * M,
                      "a further device short of memory did not make the slots yield");

    // Slots whose bytes exceed the smallest reservation are not a valid curve.
    ninfer::runtime::SequenceCapacityCurve oversized = slotted;
    oversized.bytes_per_state_slot = 600 * M;
    failures += check(!failure_text([&] {
                          (void)ninfer::runtime::resolve_kv_capacity(explicit_six, oversized,
                                                                     100000 * M);
                      }).empty(),
                      "a curve whose state slots exceed its reservation was accepted");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
