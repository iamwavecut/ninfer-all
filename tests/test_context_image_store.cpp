// The own families' context cache (runtime/engine/context_image_store.h): the host budget and its
// least recently used order, the disk tier's spill, read-back and startup scan, session slot
// records, the prefix cache file and its build identity, and where a prompt's message checkpoint
// lies. Images here are byte patterns of a fake executor; the families' probes check real state.

#include "cuda_availability.h"
#include "runtime/engine/context_image_store.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::PrefixCacheSaveControl;
using ninfer::PrefixReusePath;
using ninfer::TokenId;
using ninfer::runtime::prefix_cache::TapHint;
using ninfer::runtime::prefix_cache::TapHintKind;
using ninfer::runtime::ContextImageStore;
using ninfer::runtime::ContextImageStoreOptions;
using ninfer::runtime::ContextMediaMark;
using ninfer::runtime::MessageCheckpoint;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

struct Header {
    std::uint32_t position = 0;
};
using Store = ContextImageStore<Header>;
constexpr std::uint64_t kPage = Store::kPageBytes;

// Three and a half pages plus a position's worth: an image spans four pages, the last one partly.
std::uint64_t image_bytes(std::uint32_t position) { return 3 * kPage + kPage / 2 + position; }

std::vector<TokenId> prompt(std::uint32_t count, TokenId first = 1) {
    std::vector<TokenId> out(count);
    for (std::uint32_t i = 0; i < count; ++i) { out[i] = first + TokenId(i); }
    return out;
}

std::byte pattern(std::uint32_t seed, std::uint64_t at) {
    return static_cast<std::byte>((at * 2654435761ULL + seed) >> 7);
}

// The fake executor's save of a sequence at `position`: its bytes follow `seed`.
auto save(std::uint32_t position, std::uint32_t seed) {
    return [=](std::span<std::byte* const> pages, std::size_t page_bytes) {
        const std::uint64_t bytes = image_bytes(position);
        for (std::uint64_t at = 0; at < bytes; ++at) {
            pages[at / page_bytes][at % page_bytes] = pattern(seed, at);
        }
        return Header{.position = position};
    };
}

// Whether the store loads the image of `tokens` back with `seed`'s bytes at `position`.
bool loads(Store& store, const std::vector<TokenId>& tokens, std::uint32_t position,
           std::uint32_t seed) {
    bool same        = true;
    const auto bytes = store.load(
        tokens, {}, [&](const Header& header, std::span<std::byte* const> pages, std::size_t pb) {
            same = header.position == position;
            for (std::uint64_t at = 0; same && at < image_bytes(position); ++at) {
                same = pages[at / pb][at % pb] == pattern(seed, at);
            }
        });
    return bytes && *bytes == image_bytes(position) && same;
}

std::filesystem::path scratch() {
    const auto root = std::filesystem::temp_directory_path() /
                      ("ninfer_context_image_store_" + std::to_string(std::random_device{}()));
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    return root;
}

void configure(Store& store, std::uint64_t pages, const std::filesystem::path& disk,
               bool restore, const std::string& profile = "test-image2-profile") {
    ContextImageStoreOptions options;
    options.host_budget  = pages * kPage;
    options.host_reserve = 4 * kPage;
    options.disk_dir     = disk;
    options.disk_budget  = disk.empty() ? 0 : 64 * kPage;
    options.disk_restore = restore;
    store.configure(options, profile, image_bytes, {});
    store.pin_reserve();
}

std::size_t on_disk(const std::filesystem::path& dir) {
    std::size_t n = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        n += entry.path().extension() == ".img";
    }
    return n;
}

void check_host_budget() {
    // Two images fit in eight pages; a third drops the least recently used without a disk tier.
    Store store;
    configure(store, 8, {}, false);
    const auto a = prompt(100), b = prompt(200), c = prompt(300);
    check(store.put(a, {}, PrefixReusePath::PrivateEndpoint, 100, save(100, 1)).has_value(),
          "first image stored");
    check(store.put(b, {}, PrefixReusePath::PrivateEndpoint, 200, save(200, 2)).has_value(),
          "second image stored");
    check(!store.put(b, {}, PrefixReusePath::PrivateEndpoint, 200, save(200, 9)).has_value(),
          "an image of a held prefix is not stored twice");
    check(loads(store, a, 100, 1), "first image reads back");
    check(store.put(c, {}, PrefixReusePath::PrivateEndpoint, 300, save(300, 3)).has_value(),
          "third image stored");
    check(store.host_used() == 8 * kPage, "the images fill the budget");
    check(!store.index(b, {}).has_value(), "the least recently used image is dropped");
    check(loads(store, a, 100, 1) && loads(store, c, 300, 3), "the kept images read back");
}

void check_disk_tier(const std::filesystem::path& root) {
    const auto dir = root / "disk";
    const auto a = prompt(100), b = prompt(200), c = prompt(300);
    {
        Store store;
        configure(store, 8, dir, false);
        (void)store.put(a, {}, PrefixReusePath::PrivateEndpoint, 100, save(100, 1));
        (void)store.put(b, {}, PrefixReusePath::PrivateEndpoint, 200, save(200, 2));
        (void)store.put(c, {}, PrefixReusePath::PrivateEndpoint, 300, save(300, 3));
        check(on_disk(dir) == 1 && store.disk_used() > image_bytes(100),
              "the least recently used image spills to the disk tier");
        check(store.images().size() == 3, "a spilled image stays in the store");
        const auto [served, length] = store.best([&](const Store::Image& image) {
            return image.tokens == a ? 100U : 0U;
        });
        check(length == 0, "without restore a prompt does not resume from the disk tier");
        check(loads(store, a, 100, 1), "a spilled image reads back from its file");
        check(store.host_used() <= 8 * kPage, "reading it back stays within the budget");
        // A clean stop writes what only host pages hold: the third image.
        const auto [written, bytes] = store.spill_to_disk();
        check(written == 1 && bytes > image_bytes(300) && on_disk(dir) == 3,
              "a clean stop writes the images only in host pages");
    }
    {
        // A restart with --disk-kv-restore finds what the previous run spilled.
        Store store;
        configure(store, 8, dir, true);
        check(store.images().size() == on_disk(dir), "the scan indexes every file");
        const auto [served, length] = store.best([&](const Store::Image& image) {
            return image.tokens == a ? 100U : 0U;
        });
        check(length == 100, "with restore a prompt resumes from the disk tier");
        check(loads(store, a, 100, 1), "the scanned image reads back");
    }
    {
        // Another profile's store reads none of those files.
        Store store;
        configure(store, 8, dir, true, "test-image2-other");
        check(store.images().empty(), "another profile's files are not indexed");
    }
}

void check_records(const std::filesystem::path& root) {
    const auto a = prompt(150, 7);
    const std::vector<ContextMediaMark> media{{.begin = 10, .count = 20, .key = 0x1234}};
    std::vector<std::uint8_t> record;
    {
        Store store;
        configure(store, 8, {}, false);
        (void)store.put(a, media, PrefixReusePath::PrivateEndpoint, 150, save(150, 5));
        record = store.record(a, media);
        bool threw = false;
        try {
            (void)store.record(prompt(10), {});
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "a record of an image the store lacks is refused");
    }
    {
        Store store;
        configure(store, 8, {}, false);
        const std::size_t at = store.adopt(record);
        check(store.images()[at].tokens == a && store.images()[at].media == media,
              "an adopted record keeps its tokens and media");
        bool same        = false;
        const auto bytes = store.load(a, media, [&](const Header& h, auto pages, std::size_t pb) {
            same = h.position == 150;
            for (std::uint64_t i = 0; same && i < image_bytes(150); ++i) {
                same = pages[i / pb][i % pb] == pattern(5, i);
            }
        });
        check(bytes && same, "an adopted record reads back");
    }
    {
        // A slot file holds the session's checkpoint, then its end: both are adopted, the last
        // one's index returned.
        const auto shorter = std::vector<TokenId>(a.begin(), a.begin() + 100);
        std::vector<std::uint8_t> both;
        {
            Store store;
            configure(store, 8, {}, false);
            (void)store.put(shorter, {}, PrefixReusePath::PrivateEndpoint, 100, save(100, 6));
            both = store.record(shorter, {});
        }
        both.insert(both.end(), record.begin(), record.end());
        Store store;
        configure(store, 8, {}, false);
        const std::size_t at = store.adopt(both);
        check(store.images().size() == 2 && store.images()[at].tokens == a,
              "a two-record slot file adopts both, the last returned");
        check(loads(store, shorter, 100, 6), "the checkpoint's record reads back");
    }
    for (const auto& [name, refused] :
         {std::pair<std::string, std::vector<std::uint8_t>>{
              "another profile", std::vector<std::uint8_t>(record)},
          {"a truncated record", std::vector<std::uint8_t>(record.begin(), record.end() - 1)},
          {"not a record", std::vector<std::uint8_t>(64, 0x5a)}}) {
        Store store;
        configure(store, 8, {}, false,
                  name == "another profile" ? "test-image2-other" : "test-image2-profile");
        bool threw = false;
        try {
            (void)store.adopt(refused);
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw && store.images().empty() && store.host_used() == 0,
              name + " is refused and leaves nothing");
    }
    (void)root;
}

void check_cache_file(const std::filesystem::path& root) {
    const auto file = root / "prefix.cache";
    const auto a = prompt(100), b = prompt(200);
    {
        Store store;
        configure(store, 8, {}, false);
        (void)store.put(a, {}, PrefixReusePath::PrivateEndpoint, 100, save(100, 1));
        (void)store.put(b, {}, PrefixReusePath::PrivateLongAnchor, 200, save(200, 2));
        (void)loads(store, a, 100, 1); // a is now the most recently used
        PrefixCacheSaveControl control;
        check(store.save_file(file, "build-1", control) == 2U, "both images saved");
    }
    {
        Store store;
        configure(store, 8, {}, false);
        check(store.load_file(file, "build-2") == 0, "another build's file is not read");
        check(store.load_file(file, "build-1") == 2, "the build's file is read");
        check(loads(store, a, 100, 1) && loads(store, b, 200, 2), "the file's images read back");
        const auto at = store.index(b, {});
        check(at && store.images()[*at].path == PrefixReusePath::PrivateLongAnchor,
              "an image keeps its reuse path");
    }
    {
        // A budget for one image keeps the most recently used one.
        Store store;
        configure(store, 4, {}, false);
        check(store.load_file(file, "build-1") == 1 && store.index(a, {}).has_value(),
              "a small budget keeps the most recently used image");
    }
    {
        PrefixCacheSaveControl control;
        (void)control.abandon(std::chrono::milliseconds(0));
        Store store;
        configure(store, 8, {}, false);
        (void)store.put(prompt(300), {}, PrefixReusePath::PrivateEndpoint, 300, save(300, 3));
        check(!store.save_file(file, "build-1", control).has_value(),
              "an abandoned save writes nothing");
        Store again;
        configure(again, 8, {}, false);
        check(again.load_file(file, "build-1") == 2, "an abandoned save keeps the previous file");
    }
    {
        // With a restoring disk tier the scan finds the images first, on disk only; the file
        // brings them into host pages, and they keep their files.
        const auto dir = root / "disk-and-file";
        {
            Store store;
            configure(store, 8, dir, false);
            (void)store.put(a, {}, PrefixReusePath::PrivateEndpoint, 100, save(100, 1));
            (void)store.put(b, {}, PrefixReusePath::PrivateEndpoint, 200, save(200, 2));
            PrefixCacheSaveControl control;
            (void)store.save_file(file, "build-1", control);
            (void)store.spill_to_disk();
        }
        Store store;
        configure(store, 8, dir, true);
        check(store.images().size() == 2 && store.host_used() == 0, "the scan finds both files");
        check(store.load_file(file, "build-1") == 2 && store.images().size() == 2 &&
                  store.host_used() == 8 * kPage,
              "the file brings the scanned images into host pages");
        check(!store.images()[0].file.empty() && !store.images()[1].file.empty(),
              "the scanned images keep their files");
        check(loads(store, a, 100, 1) && loads(store, b, 200, 2), "both images read back");
    }
}

std::vector<TapHint> boundaries(std::initializer_list<std::uint32_t> at) {
    std::vector<TapHint> out;
    for (const std::uint32_t p : at) {
        out.push_back(TapHint{.position = p, .kind = TapHintKind::MessageBoundary});
    }
    out.push_back(TapHint{.position = 50, .kind = TapHintKind::Structural}); // not a boundary
    return out;
}

void check_steps() {
    // Steps of 2,048 tokens from 0: a checkpoint at 4,608 moves down to 4,096.
    const std::vector<std::uint32_t> none;
    check(MessageCheckpoint::on_steps(4608, none, {}, 2048) == 4096,
          "a checkpoint between step ends moves to the one below");
    check(MessageCheckpoint::on_steps(4096, none, {}, 2048) == 4096, "a step end stays");
    // A shared prefix at 300 ends a step: the ends are 300, 2348, 4396.
    const std::vector<std::uint32_t> shared{300};
    check(MessageCheckpoint::on_steps(4608, shared, {}, 2048) == 4396,
          "steps restart at a shared frontier");
    // One span of 16,384 tokens: no end below a 4,608-token checkpoint.
    check(MessageCheckpoint::on_steps(4608, none, {}, 16384) == 0,
          "no step end below the checkpoint: none");
    // The step end is a kept state already, or near one.
    const std::vector<std::uint32_t> at_end{4096};
    check(MessageCheckpoint::on_steps(4608, at_end, {}, 2048) == 0,
          "a step end that is a frontier gets no second image");
    const std::vector<std::uint32_t> near_end{2048, 4130};
    check(MessageCheckpoint::on_steps(4608, near_end, {}, 2048) == 0,
          "a step end near a frontier is dropped");
    // Inside a media unit: none.
    const std::vector<ContextMediaMark> media{{.begin = 4000, .count = 200, .key = 1}};
    check(MessageCheckpoint::on_steps(4608, none, media, 2048) == 0,
          "a step end inside a media unit is dropped");
}

void check_message_checkpoint() {
    const std::vector<std::uint32_t> none;
    // A document question in one long last message: the checkpoint lies kSlack before its end,
    // on the grain, clear of the request's frontier one token before the end.
    const std::vector<std::uint32_t> endpoint{5128};
    check(MessageCheckpoint::at(boundaries({100, 5129}), 5129, endpoint, {}) == 4608,
          "a long last message gets a checkpoint inside it");
    // Without boundaries the whole prompt is the message.
    check(MessageCheckpoint::at({}, 2000, none, {}) == 1280, "a prompt without boundaries");
    // A short question after a long document message: the boundary between them.
    check(MessageCheckpoint::at(boundaries({100, 4100, 4300}), 4300, none, {}) == 4100,
          "a short last message after a long one checkpoints at their boundary");
    // A chat of short turns gets none.
    check(MessageCheckpoint::at(boundaries({100, 400, 700, 1000}), 1000, none, {}) == 0,
          "short messages get no checkpoint");
    // Boundaries past the anchor do not count.
    check(MessageCheckpoint::at(boundaries({100, 5129, 6000}), 5129, none, {}) == 4608,
          "boundaries past the anchor are ignored");
    // Never inside a media unit.
    const std::vector<ContextMediaMark> media{{.begin = 4500, .count = 300, .key = 1}};
    check(MessageCheckpoint::at(boundaries({100, 5129}), 5129, none, media) == 4500,
          "a checkpoint inside a media unit moves to its start");
    // Within kSeparation of another frontier, above or below: none.
    for (const std::uint32_t other : {4560U, 4650U}) {
        const std::vector<std::uint32_t> near{other};
        check(MessageCheckpoint::at(boundaries({100, 5129}), 5129, near, {}) == 0,
              "a checkpoint near frontier " + std::to_string(other) + " is dropped");
    }
    const std::vector<std::uint32_t> far{4500, 4700};
    check(MessageCheckpoint::at(boundaries({100, 5129}), 5129, far, {}) == 4608,
          "frontiers kSeparation away keep the checkpoint");
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (ninfer::test::cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device to pin host pages with\n";
        return 77;
    }
    const auto root = scratch();
    try {
        check_message_checkpoint();
        check_steps();
        check_host_budget();
        check_disk_tier(root);
        check_records(root);
        check_cache_file(root);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        ++failures;
    }
    std::filesystem::remove_all(root);
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "context image store: all checks passed\n";
    return 0;
}
