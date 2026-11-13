// Disk-resident experts: after prepare(), every table entry of an expert the call routed to points
// at a device slot holding exactly that expert's bytes from the files, including experts that
// straddle the boundary between two files, whatever was evicted to make room, and every down matrix
// is followed by zero bytes even where a larger down of another layer was before (the expert matrix
// kernel reads past a down row's end); a call that routes to more experts than the cache holds is
// refused; experts already resident are not read again; a call that releases its layer's other
// residents gives their slots to later calls before any other resident's but after the empty
// ones, except a released expert routed to again; a layer with slots of its own (the MTP block's
// wider experts) evicts only its own, and shares the others' when the budget cannot hold both; a
// pool that starts at half its budget keeps the budget's layout, and growing it adds empty slots
// to the shared pool, which misses take before any resident's; a read that fails raises
// FileReadError and leaves no slot taken for an expert it did not finish reading; and with two
// devices, layers of two ranks that alternate through the one staging ring each read their experts
// into slots on their own device. All of it holds with direct reads (through the host window), with
// a drive mirror (the first file's experts lie in its second stripe, which the mirror serves), and
// with both.
#include "core/device.h"
#include "core/device_buffer.h"
#include "core/file_read_error.h"
#include "core/qtype.h"
#include "models/qwen4_exp/expert_stream.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <memory>
#include <fstream>
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

constexpr int kExperts = 32;

int run(int devices, bool direct, bool mirror) {
    DeviceContext device;
    std::mt19937 random(9300u);
    const std::string stem = "ninfer_expert_stream_" + std::to_string(std::random_device{}());
    ExpertDiskReads reads;
    reads.direct      = direct ? ExpertDirect::On : ExpertDirect::Off;
    reads.host_window = direct;
    if (mirror) { reads.mirror = std::filesystem::temp_directory_path() / (stem + ".mirror"); }
    // Three layers; layer 1's experts are smaller, as mixed block types are, and layer 2's wider,
    // as the MTP block's can be. Each projection of each expert is a distinct random byte run at
    // its own offset of one logical byte stream.
    const std::int64_t sizes[3][3] = {{3000, 3000, 1500}, {1200, 2400, 900}, {6000, 6000, 3000}};
    // The experts start past the first MiB: a drive mirror serves the first file's reads.
    std::vector<std::uint8_t> file_bytes(std::size_t(1) << 20);
    std::vector<std::vector<std::uint64_t>> logical(9);
    for (int layer = 0; layer < 3; ++layer) {
        for (int k = 0; k < 3; ++k) {
            for (int e = 0; e < kExperts; ++e) {
                const std::uint64_t offset = file_bytes.size() + 17; // unaligned on purpose
                file_bytes.resize(offset + sizes[layer][k]);
                for (std::int64_t i = 0; i < sizes[layer][k]; ++i) {
                    file_bytes[offset + i] = std::uint8_t(random());
                }
                logical[layer * 3 + k].push_back(offset);
            }
        }
    }
    // The stream is stored as three files cut inside two experts (layer 0's up of expert 5 and
    // layer 1's down of expert 20), which then straddle a file boundary.
    const std::vector<std::uint64_t> cuts = {logical[1][5] + 1234, logical[5][20] + 450,
                                             file_bytes.size()};
    std::vector<std::filesystem::path> paths;
    for (std::size_t f = 0; f < cuts.size(); ++f) {
        const std::uint64_t begin = f == 0 ? 0 : cuts[f - 1];
        paths.push_back(std::filesystem::temp_directory_path() /
                        (stem + "." + std::to_string(f) + ".bin"));
        std::ofstream out(paths.back(), std::ios::binary);
        out.write(reinterpret_cast<const char*>(file_bytes.data() + begin),
                  static_cast<std::streamsize>(cuts[f] - begin));
    }
    if (mirror) {
        std::filesystem::create_directories(reads.mirror);
        for (const auto& path : paths) {
            std::filesystem::copy_file(path, reads.mirror / path.filename(),
                                       std::filesystem::copy_options::overwrite_existing);
        }
    }
    std::vector<std::vector<ExpertLocation>> locations(9);
    int straddling = 0;
    for (int i = 0; i < 9; ++i) {
        const std::int64_t bytes = sizes[i / 3][i % 3];
        for (const std::uint64_t offset : logical[i]) {
            ExpertLocation location{.runs      = {},
                                    .bytes     = std::uint64_t(bytes),
                                    .format    = QType::GGUF_Q8_0,
                                    .row_bytes = 34,
                                    .rows      = 1};
            std::size_t run = 0;
            for (std::size_t f = 0; f < cuts.size(); ++f) {
                const std::uint64_t begin = f == 0 ? 0 : cuts[f - 1];
                const std::uint64_t lo    = std::max(begin, offset);
                const std::uint64_t hi    = std::min(cuts[f], offset + bytes);
                if (lo < hi) { location.runs.at(run++) = {f, lo - begin, hi - lo}; }
            }
            straddling += run == 2 ? 1 : 0;
            locations[i].push_back(location);
        }
    }
    require(straddling == 2, "two experts straddle a file boundary");
    std::vector<DeviceBuffer> tables;
    for (int i = 0; i < 9; ++i) { tables.emplace_back(kExperts * sizeof(void*)); }
    // Layers 0 and 1 route to at most ten experts a call, layer 2 to four, in slots of its own.
    const auto stream_layers = [&] {
        std::vector<ExpertStreamLayer> layers;
        for (int layer = 0; layer < 3; ++layer) {
            layers.push_back(ExpertStreamLayer{
                .rank        = 0,
                .stream      = device.stream,
                .experts     = {locations[layer * 3], locations[layer * 3 + 1],
                                locations[layer * 3 + 2]},
                .tables      = {tables[layer * 3].p, tables[layer * 3 + 1].p,
                                tables[layer * 3 + 2].p},
                .widest_call = std::size_t(layer == 2 ? 4 : 10),
                .own_pool    = layer == 2});
        }
        return layers;
    };
    // Slots are sized for the widest projections and the zeros after the down: ten of 3072 + 3072
    // + 1792 bytes for layers 0 and 1, four of 6144 + 6144 + 3328 for layer 2.
    constexpr std::uint64_t kShared = 3072 + 3072 + 1792, kOwn = 6144 + 6144 + 3328;
    // Owned so it closes its files before they are removed: Windows cannot delete an open file.
    const std::vector<std::uint64_t> budget = {10 * kShared + 4 * kOwn};
    auto owner =
        std::make_unique<ExpertStream>(device, paths, stream_layers(), budget, false, reads);
    ExpertStream& stream = *owner;
    ExpertStream* checked = &stream; // the stream check() prepares and reads
    require(stream.stats().slots == 14, "ten shared slots and four of layer 2's own");
    require(stream.storage(0) == stream.storage(1) && stream.storage(0) != stream.storage(2),
            "layer 2 has slots of its own");

    // Every expert of `ids` that layer `layer`'s tables in `in` point at holds its bytes, each
    // down followed by zeros.
    const auto verify = [&](const std::vector<DeviceBuffer>& in, int layer,
                            const std::vector<std::int32_t>& ids, const std::string& label) {
        for (int k = 0; k < 3; ++k) {
            std::vector<const void*> entries(kExperts);
            in[layer * 3 + k].copy_to_host(entries.data(), entries.size() * sizeof(void*));
            for (const std::int32_t e : ids) {
                const auto& location = locations[layer * 3 + k][e];
                const std::uint64_t offset = logical[layer * 3 + k][e];
                const std::size_t tail     = k == 2 ? 256 : 0;
                std::vector<std::uint8_t> got(location.bytes + tail);
                require(entries[e] != nullptr, label + ": a routed expert has no slot");
                require(cudaMemcpy(got.data(), entries[e], got.size(), cudaMemcpyDeviceToHost) ==
                            cudaSuccess,
                        label + ": a slot is not readable");
                require(std::memcmp(got.data(), file_bytes.data() + offset, location.bytes) == 0,
                        label + ": a slot differs from the expert's bytes");
                require(std::all_of(got.begin() + location.bytes, got.end(),
                                    [](std::uint8_t b) { return b == 0; }),
                        label + ": a down matrix is not followed by zeros");
            }
        }
    };
    const auto check = [&](int layer, const std::vector<std::int32_t>& ids,
                           const std::string& label, bool release_others = false) {
        checked->prepare(layer, ids, release_others);
        device.synchronize();
        verify(tables, layer, ids, label);
    };
    check(0, {3, 7, 3, 12}, "first call");
    check(1, {20, 5}, "straddling experts");
    check(0, {5}, "a straddling expert");
    const auto after_first = stream.stats();
    require(after_first.routes == 6 && after_first.hits == 0, "six distinct experts read");
    check(0, {7, 12, 3}, "resident again");
    require(stream.stats().hits == 3 && stream.stats().read_bytes == after_first.read_bytes,
            "resident experts are not read again");
    // Layer 1 and more layer-0 calls churn through the ten slots.
    for (int round = 0; round < 12; ++round) {
        std::vector<std::int32_t> ids;
        for (int i = 0; i < 6; ++i) { ids.push_back(std::int32_t(random() % kExperts)); }
        check(round % 2, ids, "churn " + std::to_string(round));
    }
    // A span: layer 1 fills the ten slots, layer 0's first chunk takes four of them, and its last
    // chunk (6 and 7) releases 3, 4 and 5; 7 takes one of their slots. Expert 5 is routed to again.
    // Two later misses take the slot 4 left, then one by the hand: 5, 6 and 7 stay, and five of
    // layer 1's experts.
    check(1, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, "layer 1 fills the slots");
    check(0, {3, 4, 5, 6}, "a span's first chunk");
    check(0, {6, 7}, "a span's last chunk", true);
    check(0, {5}, "a released expert routed again");
    check(1, {20, 21}, "misses after a release");
    auto before = stream.stats();
    check(0, {5, 6, 7}, "after the misses");
    require(stream.stats().hits - before.hits == 3,
            "the last chunk's experts and a released one routed again stay resident");
    before = stream.stats();
    check(1, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, "layer 1 again");
    require(stream.stats().hits - before.hits == 5,
            "a release gives its slots to later misses before other residents'");
    // Layer 2 evicts only its own slots: layer 1's ten experts stay resident.
    check(2, {0, 1, 2, 3}, "layer 2's own slots");
    check(2, {4, 30}, "layer 2 evicts its own");
    before = stream.stats();
    check(1, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, "layer 1 beside layer 2");
    require(stream.stats().hits - before.hits == 10, "layer 2 takes none of the shared slots");
    bool refused = false;
    try {
        std::vector<std::int32_t> too_many;
        for (int e = 0; e < 11; ++e) { too_many.push_back(e); }
        stream.prepare(1, too_many, false);
    } catch (const std::runtime_error&) { refused = true; }
    require(refused, "a call that needs more experts than slots is refused");
    refused = false;
    try {
        stream.prepare(2, std::vector<std::int32_t>{0, 1, 2, 3, 4}, false);
    } catch (const std::runtime_error&) { refused = true; }
    require(refused, "a call that needs more experts than its own slots is refused");
    {
        // Empty slots go before released ones: in a fresh pool, layer 0's last chunk releases 0, 1
        // and 2, and its miss and layer 1's five take empty slots, so all five stay resident.
        ExpertStream fresh(device, paths, stream_layers(), budget, false, reads);
        fresh.prepare(0, std::vector<std::int32_t>{0, 1, 2, 3}, false);
        fresh.prepare(0, std::vector<std::int32_t>{3, 4}, true);
        fresh.prepare(1, std::vector<std::int32_t>{0, 1, 2, 3, 4}, false);
        const auto hits = fresh.stats().hits;
        fresh.prepare(0, std::vector<std::int32_t>{0, 1, 2, 3, 4}, false);
        require(fresh.stats().hits - hits == 5,
                "released experts keep their slots while empty ones remain");
        device.synchronize();
    }
    {
        // A budget short of four own slots beside ten shared ones: one pool of the wider slots.
        ExpertStream narrow(device, paths, stream_layers(),
                            std::vector<std::uint64_t>{10 * kShared + 3 * kOwn}, false, reads);
        require(narrow.stats().slots == (10 * kShared + 3 * kOwn) / kOwn,
                "a short budget shares slots fitting the widest expert");
        require(narrow.storage(0) == narrow.storage(2), "layer 2 shares the slots");
        narrow.prepare(2, std::vector<std::int32_t>{0, 1, 2, 3}, false);
        narrow.prepare(0, std::vector<std::int32_t>{3, 7}, false);
        narrow.prepare(2, std::vector<std::int32_t>{0, 1, 2, 3}, false);
        require(narrow.stats().hits == 4, "the shared slots hold both layers' experts");
        device.synchronize();
    }
    {
        // Started at half the budget: the same layout, layer 2's four own slots and the shared
        // pool at its widest call (ten slots, more than half the budget's). Growth adds whole
        // shared slots only; five misses take them, so layer 1's ten experts stay resident and
        // every expert reads as stored.
        ExpertStream half(device, paths, stream_layers(),
                          std::vector<std::uint64_t>{20 * kShared + 4 * kOwn}, true, reads);
        require(half.stats().slots == 14 && half.pool_bytes(0) == 10 * kShared + 4 * kOwn &&
                    half.storage(0) != half.storage(2),
                "a pool that grows starts at its widest call in the budget's layout");
        checked = &half;
        check(1, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, "layer 1 fills the started slots");
        const auto added = half.grow(std::vector<std::uint64_t>{5 * kShared + kShared / 2});
        require(added.size() == 1 && added[0] == 5 * kShared && half.stats().slots == 19 &&
                    half.storage(0)[1].second - half.storage(0)[1].first == 5 * kShared &&
                    half.storage(2)[1].first == half.storage(2)[1].second,
                "growth adds whole slots to the shared pool only");
        check(0, {10, 11, 12, 13, 14}, "misses into the grown slots");
        const auto hits = half.stats().hits;
        check(1, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, "layer 1 after growth");
        require(half.stats().hits - hits == 10, "grown slots are taken before residents'");
        checked = &stream;
        device.synchronize();
    }
#ifndef _WIN32
    // Layer 1's downs past expert 20 live in the last file. Cut short, it fails a call to experts
    // whose gates and ups read fine; once whole again, the same call reads every byte as stored.
    std::filesystem::resize_file(paths[2], 0);
    bool failed = false;
    try {
        const std::vector<std::int32_t> past_cut{25, 26, 27};
        stream.prepare(1, past_cut, false);
    } catch (const FileReadError&) { failed = true; }
    require(failed, "a failed read raises FileReadError");
    {
        std::fstream out(paths[2], std::ios::binary | std::ios::in | std::ios::out);
        out.write(reinterpret_cast<const char*>(file_bytes.data() + cuts[1]),
                  static_cast<std::streamsize>(cuts[2] - cuts[1]));
    }
    check(1, {25, 26, 27}, "after a failed read");
    check(0, {3, 7, 12}, "layer 0 after a failed read");
#endif
    if (devices >= 2) {
        // Layer 0 on the first device, layers 1 and 2 (slots of its own) on the second, each on
        // its rank's stream and tables. Calls that alternate between the ranks close a staging
        // half behind one device's copies and fill it again for the other's.
        const std::array<int, 2> ids{0, 1};
        DeviceContext pair(ids);
        std::vector<DeviceBuffer> rank_tables;
        for (int i = 0; i < 9; ++i) {
            RankBinding bind(pair, i < 3 ? 0 : 1);
            rank_tables.emplace_back(kExperts * sizeof(void*));
        }
        std::vector<ExpertStreamLayer> layers;
        for (int layer = 0; layer < 3; ++layer) {
            const std::size_t rank = layer == 0 ? 0 : 1;
            layers.push_back(ExpertStreamLayer{
                .rank        = rank,
                .stream      = pair.rank(rank).stream,
                .experts     = {locations[layer * 3], locations[layer * 3 + 1],
                                locations[layer * 3 + 2]},
                .tables      = {rank_tables[layer * 3].p, rank_tables[layer * 3 + 1].p,
                                rank_tables[layer * 3 + 2].p},
                .widest_call = std::size_t(layer == 2 ? 4 : 10),
                .own_pool    = layer == 2});
        }
        // Ten slots of layer 0's widest projections on the first device; on the second, ten of
        // layer 1's (1280 + 2560 + 1280 bytes) and layer 2's four.
        constexpr std::uint64_t kLayer1 = 1280 + 2560 + 1280;
        ExpertStream across(pair, paths, std::move(layers),
                            std::vector<std::uint64_t>{10 * kShared, 10 * kLayer1 + 4 * kOwn},
                            false, reads);
        require(across.stats().slots == 24 && across.pool_bytes(0) == 10 * kShared &&
                    across.pool_bytes(1) == 10 * kLayer1 + 4 * kOwn,
                "each rank's pools take its own budget");
        for (int layer = 0; layer < 3; ++layer) {
            const auto at = reinterpret_cast<const void*>(across.storage(layer)[0].first);
            require(memory_device(at) == pair.rank(layer == 0 ? 0 : 1).device,
                    "a layer's slots lie on its rank's device");
        }
        const auto step = [&](int layer, const std::vector<std::int32_t>& experts,
                              const std::string& label) {
            across.prepare(layer, experts, false);
            for (std::size_t r = 0; r < 2; ++r) {
                RankBinding bind(pair, r);
                require(cudaDeviceSynchronize() == cudaSuccess, label + ": a device failed");
            }
            RankBinding bind(pair, layer == 0 ? 0 : 1);
            verify(rank_tables, layer, experts, label);
        };
        step(0, {3, 7, 12}, "the first rank");
        step(1, {20, 5}, "the second rank");
        step(0, {5, 1}, "the first rank again");
        step(2, {0, 1, 2, 3}, "the second rank's own slots");
        for (int round = 0; round < 12; ++round) {
            const int layer = round % 3;
            std::vector<std::int32_t> experts;
            for (int i = 0; i < (layer == 2 ? 4 : 6); ++i) {
                experts.push_back(std::int32_t(random() % kExperts));
            }
            step(layer, experts, "alternating " + std::to_string(round));
        }
        std::cout << "two ranks checked\n";
    } else {
        std::cout << "two ranks skipped: one CUDA device\n";
    }
    owner.reset();
    for (const auto& path : paths) { std::filesystem::remove(path); }
    if (mirror) { std::filesystem::remove_all(reads.mirror); }
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
        for (const bool direct : {false, true}) {
            for (const bool mirror : {false, true}) {
                run(devices, direct, mirror);
                std::cout << "checked" << (direct ? " direct" : "") << (mirror ? " mirror" : "")
                          << '\n';
            }
        }
        std::cout << "PASS qwen4_exp expert stream\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
