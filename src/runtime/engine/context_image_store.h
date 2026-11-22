#pragma once

// The context cache of the families with an Engine core of their own (Gemma 4, Flint,
// DeepSeek-V4.1, Inkling 2 and Qwen3.8-Flash-Next) past their sequences: images of the prefixes
// the sequences give up and of the prompt frontiers worth keeping, in pinned host pages within a
// budget. With a disk tier (--disk-kv-path) the least recently used images go to files there,
// where a restart finds them (--disk-kv-restore); without one they are dropped. The executor
// copies a sequence's state into an image's pages and back (image_bytes, save_image,
// load_image); the core decides what to keep and which image serves a prompt; the store owns the
// pages, the files and the order of use.
//
// One record layout serves the disk tier's files, the prefix cache file (--prefix-cache-file:
// every image, written at a clean stop and read at startup) and session slot files: a header, the
// profile the image was written for (the artifact, KV storage and whatever else gives the state
// its layout and meaning), the family's header fields past the position, the prompt's tokens and
// media units, then the image's bytes.

#include "core/host_pages.h"
#include "ninfer/base.h"
#include "ninfer/engine_options.h"
#include "ninfer/kv_cache_storage.h"
#include "ninfer/observability/execution.h"
#include "runtime/engine/diagnostics.h"
#include "runtime/prefix_cache/tap_planner.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// A prompt's media unit: its soft tokens [begin, begin + count) and a key of the unit's content.
struct ContextMediaMark {
    std::uint32_t begin = 0, count = 0;
    std::uint64_t key   = 0;

    friend bool operator==(const ContextMediaMark&, const ContextMediaMark&) = default;
};
static_assert(sizeof(ContextMediaMark) == 16);

// Session digests (SlotState): FNV-1a 64 over the little-endian token id bytes, as 16 lowercase
// hex characters, the Qwen3.5 Program's digest of its token ledger.
[[nodiscard]] inline std::string session_digest(std::span<const TokenId> tokens) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const TokenId token : tokens) {
        const auto value = static_cast<std::uint32_t>(token);
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<std::uint8_t>(value >> shift);
            hash *= 1099511628211ULL;
        }
    }
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(hash));
    return text;
}

// The checkpoint of a prompt's last messages: a later prompt that shares a long span with this
// one but goes on in other words (a document followed by another question, in the same message or
// the next) resumes there. Images copy a sequence's whole state, so a prompt gets at most one, and
// only past a long message: a chat of short turns resumes from its anchors and endpoints instead.
// With the last message (between the last two message boundaries at or below `end`, the request's
// anchor; the whole prompt without boundaries) at least kLong tokens, the checkpoint lies inside
// it, at the last multiple of kGrain kSlack tokens before its end, which a question of up to
// kSlack tokens after the shared span leaves shared. Otherwise, with the message before it that
// long, at the boundary between them. Moved to a media unit's start rather than inside one; none
// (0) within kSeparation of `frontiers` (the request's other image frontiers). The caller checks
// that an image there pays.
struct MessageCheckpoint {
    static constexpr std::uint32_t kGrain = 256, kSlack = 512, kLong = 768, kSeparation = 64;

    [[nodiscard]] static std::uint32_t at(std::span<const prefix_cache::TapHint> hints,
                                          std::uint32_t end,
                                          std::span<const std::uint32_t> frontiers,
                                          std::span<const ContextMediaMark> media) {
        std::vector<std::uint32_t> bounds{0};
        for (const prefix_cache::TapHint& hint : hints) {
            if (hint.kind == prefix_cache::TapHintKind::MessageBoundary && hint.position <= end) {
                bounds.push_back(hint.position);
            }
        }
        std::sort(bounds.begin(), bounds.end());
        bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
        const std::size_t k     = bounds.size();
        const std::uint32_t top = k > 1 ? bounds[k - 1] : end;
        const std::uint32_t low = k > 1 ? bounds[k - 2] : 0;
        std::uint32_t frontier  = 0;
        if (top - low >= kLong) {
            frontier = (top - kSlack) / kGrain * kGrain;
        } else if (k > 2 && low - bounds[k - 3] >= kLong) {
            frontier = low;
        }
        for (const ContextMediaMark& mark : media) {
            if (frontier > mark.begin && frontier < mark.begin + mark.count) {
                frontier = mark.begin;
            }
        }
        if (frontier < kSeparation || frontier >= end) { return 0; }
        for (const std::uint32_t other : frontiers) {
            if (other + kSeparation > frontier && frontier + kSeparation > other) { return 0; }
        }
        return frontier;
    }

    // The checkpoint `at` moved down to the end of a prompt step, for an Engine whose prompt steps
    // each copy or read the experts the step routes to (experts off the GPU): a checkpoint
    // between two step ends would add a step, and a pass of the experts, to every prompt that
    // keeps one. Steps run from `from` (where the prompt resumes), end at each of `frontiers`
    // (sorted, the request's other kept states) and every `step` tokens after the last end. 0
    // when no step end other than a frontier lies above `from` and at or below `at`, or when that
    // end lies inside a media unit.
    [[nodiscard]] static std::uint32_t on_steps(std::uint32_t at,
                                                std::span<const std::uint32_t> frontiers,
                                                std::span<const ContextMediaMark> media,
                                                std::uint32_t step, std::uint32_t from = 0) {
        if (at <= from || step == 0) { return 0; }
        std::uint32_t position = from, last = 0;
        auto next = frontiers.begin();
        for (;;) {
            while (next != frontiers.end() && *next <= position) { ++next; }
            const std::uint32_t end =
                next != frontiers.end() ? std::min(position + step, *next) : position + step;
            if (end > at) { break; }
            last     = end;
            position = end;
        }
        if (last < from + kSeparation) { return 0; }
        for (const std::uint32_t other : frontiers) {
            if (other + kSeparation > last && last + kSeparation > other) { return 0; }
        }
        for (const ContextMediaMark& mark : media) {
            if (last > mark.begin && last < mark.begin + mark.count) { return 0; }
        }
        return last;
    }
};

// An image's header fields past `position` in a record: none by default.
struct PositionOnlyImageHeader {
    template <class Header>
    static void write(const Header&, std::vector<std::byte>&) {}

    template <class Header>
    static bool read(std::span<const std::byte> extra, Header&) {
        return extra.empty();
    }
};

struct ContextImageStoreOptions {
    // Host pages the images may hold, rounded down to whole pages; zero disables the store.
    std::uint64_t host_budget = 0;
    // Of those, the pages pin_reserve() pins at startup; the store pins the rest as images need
    // them, and pin_ahead() while the worker idles, up to the budget.
    std::uint64_t host_reserve = 0;
    // The disk tier's directory for this profile; empty: none.
    std::filesystem::path disk_dir;
    std::uint64_t disk_budget = 0;
    // Prompts may resume from images only on disk (those a previous run left included); off
    // leaves the tier write-only.
    bool disk_restore = false;
};

// One record: this header, the profile, the family's fields, the tokens, the media units, the
// bytes.
struct ContextImageRecordHeader {
    char magic[8]               = {'N', 'I', 'N', 'F', 'I', 'M', 'G', '2'};
    std::uint32_t position      = 0;
    std::uint32_t path          = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t media_units   = 0;
    std::uint32_t extra_bytes   = 0;
    std::uint32_t profile_bytes = 0;
    std::uint64_t bytes         = 0;
};
static_assert(sizeof(ContextImageRecordHeader) == 40);

// The prefix cache file: this header, the build identity it was written by, then `images`
// records, the most recently used first.
struct ContextImageFileHeader {
    char magic[8]                = {'N', 'I', 'N', 'F', 'C', 'C', 'H', '2'};
    std::uint32_t images         = 0;
    std::uint32_t identity_bytes = 0;
};
static_assert(sizeof(ContextImageFileHeader) == 16);

// The store's options for a family's cache: the disk tier, when --disk-kv-path names one, in the
// directory of `profile` beneath it.
[[nodiscard]] inline ContextImageStoreOptions
context_image_store_options(const ContextCacheOptions& cache, std::uint64_t host_budget,
                            std::uint64_t host_reserve, const std::string& profile) {
    ContextImageStoreOptions out;
    out.host_budget  = host_budget;
    out.host_reserve = host_reserve;
    if (!cache.disk_kv_path.empty()) {
        out.disk_dir     = cache.disk_kv_path / profile;
        out.disk_budget  = cache.disk_kv_capacity_bytes != 0 ? cache.disk_kv_capacity_bytes
                                                             : (std::uint64_t(64) << 30);
        out.disk_restore = cache.disk_kv_restore;
    }
    return out;
}

// A family's image profile: the family, its artifact, the KV storage and the image's size at
// positions 0 and 1 (which differ wherever the cache's rows, the rings or a drafter's state do),
// then `extra` (a drafter artifact, whatever else gives the state its meaning).
template <class ArtifactId>
[[nodiscard]] std::string context_image_profile(std::string_view family, const ArtifactId& id,
                                                KvCacheStorage kv, std::uint64_t bytes_at_0,
                                                std::uint64_t bytes_at_1, std::string_view extra) {
    std::string out(family);
    out += "-image2-";
    for (const auto b : id) {
        char hex[3];
        std::snprintf(hex, sizeof(hex), "%02x", static_cast<unsigned>(b));
        out += hex;
    }
    out += "-kv" + std::to_string(static_cast<int>(kv)) + "-" + std::to_string(bytes_at_0) + "-" +
           std::to_string(bytes_at_1 - bytes_at_0);
    out += extra;
    return out;
}

template <class Header, class Codec = PositionOnlyImageHeader>
class ContextImageStore {
public:
    static constexpr std::size_t kPageBytes = std::size_t(8) << 20;

    struct Image {
        std::vector<TokenId> tokens;
        std::vector<ContextMediaMark> media;
        Header header{};
        PrefixReusePath path = PrefixReusePath::PrivateEndpoint;
        std::vector<std::byte*> pages; // empty while only on disk
        std::filesystem::path file;    // empty unless on disk
        std::uint64_t bytes     = 0;
        std::uint64_t last_used = 0;
    };

    // `profile` names what an image's bytes mean; `image_bytes` is the executor's image size at a
    // position; `sync` waits for the copies the executor queued into or out of the pages, before
    // the host reads or writes them.
    void configure(const ContextImageStoreOptions& options, std::string profile,
                   std::function<std::uint64_t(std::uint32_t)> image_bytes,
                   std::function<void()> sync) {
        host_budget_  = options.host_budget / kPageBytes * kPageBytes;
        host_reserve_ = std::min(options.host_reserve / kPageBytes * kPageBytes, host_budget_);
        profile_      = std::move(profile);
        image_bytes_  = std::move(image_bytes);
        sync_         = std::move(sync);
        if (host_budget_ == 0 || options.disk_dir.empty()) { return; }
        disk_dir_     = options.disk_dir;
        disk_budget_  = options.disk_budget;
        disk_restore_ = options.disk_restore;
        std::filesystem::create_directories(disk_dir_);
        if (disk_restore_) { scan_disk(); }
    }

    [[nodiscard]] bool enabled() const noexcept { return host_budget_ != 0; }
    [[nodiscard]] std::uint64_t host_budget() const noexcept { return host_budget_; }
    [[nodiscard]] std::uint64_t host_used() const noexcept { return host_used_; }
    // The pinned pages: those the images hold and the free ones.
    [[nodiscard]] std::uint64_t host_pinned() const noexcept {
        return std::uint64_t(free_pages_.size()) * kPageBytes + host_used_;
    }
    [[nodiscard]] std::uint64_t disk_used() const noexcept { return disk_used_; }
    [[nodiscard]] const std::string& profile() const noexcept { return profile_; }
    [[nodiscard]] const std::vector<Image>& images() const noexcept { return images_; }

    [[nodiscard]] static std::uint64_t page_rounded(std::uint64_t bytes) noexcept {
        return (bytes + kPageBytes - 1) / kPageBytes * kPageBytes;
    }

    // Pins the reserve now; throws when the host cannot pin it.
    void pin_reserve() {
        const std::uint64_t pinned = host_pinned();
        if (host_reserve_ > pinned) {
            pin_pages(static_cast<std::size_t>((host_reserve_ - pinned) / kPageBytes));
        }
    }

    // Whether the idle worker pins another chunk: the pinned pages hold less than the budget, so
    // an image stored later takes pinned pages instead of pinning its own while its request
    // waits. A failed pin ends idle pinning; take_pages still pins.
    [[nodiscard]] bool pins_ahead() const noexcept { return pages_ahead() != 0; }

    void pin_ahead() {
        try {
            pin_pages(pages_ahead());
        } catch (const std::exception&) { pin_ahead_failed_ = true; }
    }

    void stop_pinning_ahead() noexcept { pin_ahead_failed_ = true; }

    // The image of exactly these tokens and media.
    [[nodiscard]] std::optional<std::size_t>
    index(std::span<const TokenId> tokens, std::span<const ContextMediaMark> media) const {
        for (std::size_t i = 0; i < images_.size(); ++i) {
            if (std::ranges::equal(images_[i].tokens, tokens) &&
                std::ranges::equal(images_[i].media, media)) {
                return i;
            }
        }
        return std::nullopt;
    }

    // The image that serves the most, by `length(image)`, of those a prompt may resume from: in
    // host pages, or on disk when the disk tier restores. {index, 0} when none serves.
    template <class Length>
    [[nodiscard]] std::pair<std::size_t, std::uint32_t> best(Length&& length) const {
        std::pair<std::size_t, std::uint32_t> out{0, 0};
        for (std::size_t i = 0; i < images_.size(); ++i) {
            if (images_[i].pages.empty() && !disk_restore_) { continue; }
            const std::uint32_t served = length(images_[i]);
            if (served > out.second) { out = {i, served}; }
        }
        return out;
    }

    void touch(std::size_t i) { images_.at(i).last_used = ++clock_; }

    // Keeps an image of the state at `position`, which `save(pages, page_bytes)` copies into the
    // image's pages and returns the header of, unless the store holds that prefix already or
    // cannot make room. Returns the bytes saved, or nothing.
    template <class Save>
    std::optional<std::uint64_t> put(std::vector<TokenId> tokens,
                                     std::vector<ContextMediaMark> media, PrefixReusePath path,
                                     std::uint32_t position, Save&& save) {
        if (!enabled()) { return std::nullopt; }
        if (const auto known = index(tokens, media)) {
            touch(*known);
            return std::nullopt;
        }
        const std::uint64_t bytes = image_bytes_(position);
        if (!make_room(page_rounded(bytes))) { return std::nullopt; }
        Image image;
        image.tokens = std::move(tokens);
        image.media  = std::move(media);
        image.path   = path;
        image.bytes  = bytes;
        image.pages  = take_pages(bytes);
        if (image.pages.empty()) { return std::nullopt; }
        try {
            image.header = save(std::span<std::byte* const>(image.pages), kPageBytes);
        } catch (...) {
            give_back(image);
            throw;
        }
        image.last_used = ++clock_;
        images_.push_back(std::move(image));
        return bytes;
    }

    // Copies the image of `tokens` and `media` back through `load(header, pages, page_bytes)`,
    // reading it from its file first when it is only on disk. Returns its bytes, or nothing when
    // the store no longer holds it.
    template <class Load>
    std::optional<std::uint64_t> load(std::span<const TokenId> tokens,
                                      std::span<const ContextMediaMark> media, Load&& load) {
        auto at = index(tokens, media);
        if (!at) { return std::nullopt; }
        if (images_[*at].pages.empty()) {
            if (!make_room(page_rounded(images_[*at].bytes)) || !(at = index(tokens, media))) {
                return std::nullopt;
            }
            if (!read_file(images_[*at])) { return std::nullopt; }
        }
        Image& image = images_[*at];
        load(static_cast<const Header&>(image.header), std::span<std::byte* const>(image.pages),
             kPageBytes);
        image.last_used           = ++clock_;
        const std::uint64_t bytes = image.bytes;
        // Within the budget again, the image just read in being the most recently used.
        (void)make_room(0);
        return bytes;
    }

    // Drops the images `drop(image)` names, their files too.
    template <class Drop>
    void drop_if(Drop&& drop) {
        for (std::size_t i = images_.size(); i-- > 0;) {
            Image& image = images_[i];
            if (!drop(static_cast<const Image&>(image))) { continue; }
            remove_file(image);
            give_back(image);
            images_.erase(images_.begin() + static_cast<std::ptrdiff_t>(i));
        }
    }

    [[nodiscard]] bool has_disk_tier() const noexcept { return !disk_dir_.empty(); }

    // A clean stop: every image held only in host pages goes to the disk tier, the least recently
    // used first, so that the tier's budget keeps the most recent. Returns how many and their
    // bytes.
    std::pair<std::uint32_t, std::uint64_t> spill_to_disk() {
        if (disk_dir_.empty()) { return {0, 0}; }
        std::vector<std::size_t> order;
        for (std::size_t i = 0; i < images_.size(); ++i) {
            if (!images_[i].pages.empty() && images_[i].file.empty()) { order.push_back(i); }
        }
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return images_[a].last_used < images_[b].last_used;
        });
        std::uint32_t written = 0;
        std::uint64_t bytes   = 0;
        for (const std::size_t i : order) {
            write_file(images_[i]);
            if (!images_[i].file.empty()) {
                ++written;
                bytes += record_bytes(images_[i]);
            }
        }
        return {written, bytes};
    }

    // ---- Records: session slot files and the prefix cache file ---------------------------------

    // One record from `in` into host pages as the most recently used image, replacing an image
    // of the same prefix; returns its index. A record of another profile, or not a record, throws
    // std::invalid_argument.
    std::size_t read_record(std::istream& in) {
        Image image;
        if (!read_record_head(in, image)) {
            throw std::invalid_argument(
                "session snapshot file was not written for this model and configuration");
        }
        drop_if([&](const Image& held) {
            return held.tokens == image.tokens && held.media == image.media;
        });
        if (!make_room(page_rounded(image.bytes))) {
            throw std::invalid_argument("session snapshot does not fit the host cache budget");
        }
        image.pages = take_pages(image.bytes);
        if (image.pages.empty()) {
            throw std::runtime_error("context image: cannot pin host pages for the session");
        }
        if (sync_) { sync_(); }
        if (!read_bytes(in, image)) {
            give_back(image);
            throw std::invalid_argument("session snapshot file is truncated");
        }
        image.last_used = ++clock_;
        images_.push_back(std::move(image));
        return images_.size() - 1;
    }

    // The image of `tokens` and `media` as one record in memory (a session slot's file).
    std::vector<std::uint8_t> record(std::span<const TokenId> tokens,
                                     std::span<const ContextMediaMark> media) {
        auto at = index(tokens, media);
        if (!at) { throw std::invalid_argument("the host cache cannot hold the session"); }
        if (images_[*at].pages.empty() &&
            (!make_room(page_rounded(images_[*at].bytes)) || !(at = index(tokens, media)) ||
             !read_file(images_[*at]))) {
            throw std::invalid_argument("the session's image file is unreadable");
        }
        if (sync_) { sync_(); }
        const Image& image = images_[*at];
        std::vector<std::uint8_t> out(record_bytes(image));
        std::size_t used = 0;
        const auto put   = [&](const void* data, std::size_t bytes) {
            std::memcpy(out.data() + used, data, bytes);
            used += bytes;
        };
        std::vector<std::byte> extra;
        const ContextImageRecordHeader head = record_head(image, extra);
        put(&head, sizeof(head));
        put(profile_.data(), profile_.size());
        put(extra.data(), extra.size());
        put(image.tokens.data(), 4 * image.tokens.size());
        put(image.media.data(), sizeof(ContextMediaMark) * image.media.size());
        for (std::uint64_t at_byte = 0; at_byte < image.bytes; at_byte += kPageBytes) {
            const std::uint64_t part = std::min<std::uint64_t>(kPageBytes, image.bytes - at_byte);
            put(image.pages[at_byte / kPageBytes], static_cast<std::size_t>(part));
        }
        return out;
    }

    // Adopts the records of a session slot's file into host pages, each the most recently used
    // image as it comes, replacing an image of the same prefix; returns the last one's index. A
    // record of another profile, or not a record, throws std::invalid_argument.
    std::size_t adopt(std::span<const std::uint8_t> records) {
        struct View : std::streambuf {
            explicit View(std::span<const std::uint8_t> bytes) {
                auto* begin = reinterpret_cast<char*>(const_cast<std::uint8_t*>(bytes.data()));
                setg(begin, begin, begin + bytes.size());
            }
        } view(records);
        std::istream in(&view);
        std::size_t last = read_record(in);
        while (in.peek() != std::char_traits<char>::eof()) { last = read_record(in); }
        return last;
    }

    // Writes every image, the most recently used first, to the prefix cache file for builds of
    // `identity`; returns how many, or nothing when the save failed or `control` abandoned it,
    // which leaves the previous file.
    std::optional<std::uint32_t> save_file(const std::filesystem::path& file,
                                           std::string_view identity,
                            const PrefixCacheSaveControl& control) {
        if (!control.begin()) { return std::nullopt; }
        const auto temporary = std::filesystem::path(file).concat(".tmp");
        std::vector<std::size_t> order(images_.size());
        for (std::size_t i = 0; i < order.size(); ++i) { order[i] = i; }
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return images_[a].last_used > images_[b].last_used;
        });
        if (sync_) { sync_(); }
        ContextImageFileHeader head;
        head.identity_bytes = static_cast<std::uint32_t>(identity.size());
        bool saved          = false;
        {
            std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(&head), sizeof(head));
            out.write(identity.data(), std::streamsize(identity.size()));
            for (const std::size_t i : order) {
                if (control.abandoned()) { break; }
                Image& image = images_[i];
                if (image.pages.empty()) {
                    // Copied from its disk file as it lies.
                    std::ifstream in(image.file, std::ios::binary);
                    if (!in) { continue; }
                    out << in.rdbuf();
                } else {
                    (void)write_record(out, image);
                }
                ++head.images;
            }
            out.seekp(0);
            out.write(reinterpret_cast<const char*>(&head), sizeof(head));
            saved = static_cast<bool>(out) && !control.abandoned();
        }
        std::error_code failed;
        if (saved) { std::filesystem::rename(temporary, file, failed); }
        if (!saved || failed) {
            std::filesystem::remove(temporary, failed);
            control.end(false);
            return std::nullopt;
        }
        control.end(true);
        return head.images;
    }

    // Reads the prefix cache file's images into host pages while they fit, the most recently
    // used first, also those the disk tier's scan found only on disk; a file another build
    // wrote, a missing one, or a record of another profile ends the read. Returns how many it
    // kept.
    std::uint32_t load_file(const std::filesystem::path& file, std::string_view identity) {
        std::ifstream in(file, std::ios::binary);
        ContextImageFileHeader head;
        if (!in || !in.read(reinterpret_cast<char*>(&head), sizeof(head)) ||
            std::memcmp(head.magic, ContextImageFileHeader{}.magic, 8) != 0 ||
            head.identity_bytes != identity.size()) {
            return 0;
        }
        std::string written(head.identity_bytes, '\0');
        if (!in.read(written.data(), std::streamsize(written.size())) || written != identity) {
            return 0;
        }
        std::uint32_t kept = 0;
        for (std::uint32_t n = 0; n < head.images; ++n) {
            Image image;
            if (!read_record_head(in, image)) { break; }
            const std::uint64_t rounded = page_rounded(image.bytes);
            const auto known            = index(image.tokens, image.media);
            if ((known && !images_[*known].pages.empty()) ||
                host_used_ + rounded > host_budget_) {
                in.seekg(std::streamoff(image.bytes), std::ios::cur);
                continue;
            }
            // An image the scan found on disk keeps its file and gets its bytes from this one.
            Image& into = known ? images_[*known] : image;
            into.pages  = take_pages(into.bytes);
            if (into.pages.empty()) { break; }
            if (!read_bytes(in, into)) {
                give_back(into);
                break;
            }
            into.last_used = head.images - n; // the file's order of use
            if (!known) { images_.push_back(std::move(image)); }
            ++kept;
        }
        clock_ = std::max<std::uint64_t>(clock_, head.images);
        return kept;
    }

private:
    static constexpr std::size_t kPinAheadPages = 8;

    // The pages pin_ahead() pins: a chunk, or what is left of the budget.
    [[nodiscard]] std::size_t pages_ahead() const noexcept {
        const std::uint64_t pinned = host_pinned();
        if (pin_ahead_failed_ || pinned >= host_budget_) { return 0; }
        return static_cast<std::size_t>(
            std::min<std::uint64_t>(kPinAheadPages, (host_budget_ - pinned) / kPageBytes));
    }

    // Pins `count` more free pages in one allocation.
    void pin_pages(std::size_t count) {
        if (count == 0) { return; }
        auto chunk = std::make_unique<HostPages>(count * kPageBytes);
        chunk->pin();
        auto* base = static_cast<std::byte*>(chunk->data());
        for (std::size_t i = count; i-- > 0;) { free_pages_.push_back(base + i * kPageBytes); }
        chunks_.push_back(std::move(chunk));
    }

    // Pages for an image of `bytes`, pinning more when the free ones fall short; the caller has
    // made room for them (make_room), so the pages stay within the budget. Empty when more pages
    // cannot be pinned.
    std::vector<std::byte*> take_pages(std::uint64_t bytes) {
        const auto count = static_cast<std::size_t>(page_rounded(bytes) / kPageBytes);
        if (free_pages_.size() < count) {
            try {
                pin_pages(count - free_pages_.size());
            } catch (const std::exception&) { return {}; }
        }
        std::vector<std::byte*> pages(free_pages_.end() - std::ptrdiff_t(count), free_pages_.end());
        free_pages_.resize(free_pages_.size() - count);
        host_used_ += count * kPageBytes;
        return pages;
    }

    void give_back(Image& image) {
        host_used_ -= image.pages.size() * kPageBytes;
        free_pages_.insert(free_pages_.end(), image.pages.begin(), image.pages.end());
        image.pages.clear();
    }

    // Brings the images in host pages within the budget with `bytes` more (whole pages): the
    // least recently used go to the disk tier, or are dropped without one. False when the budget
    // cannot hold `bytes` at all. Images may be dropped, so indices change.
    bool make_room(std::uint64_t bytes) {
        if (bytes > host_budget_) { return false; }
        while (host_used_ + bytes > host_budget_) {
            std::optional<std::size_t> oldest;
            for (std::size_t i = 0; i < images_.size(); ++i) {
                if (!images_[i].pages.empty() &&
                    (!oldest || images_[i].last_used < images_[*oldest].last_used)) {
                    oldest = i;
                }
            }
            if (!oldest) { return false; }
            write_file(images_[*oldest]);
            give_back(images_[*oldest]);
            drop_dead();
        }
        drop_dead();
        return true;
    }

    void drop_dead() {
        std::erase_if(images_,
                      [](const Image& image) { return image.pages.empty() && image.file.empty(); });
    }

    [[nodiscard]] std::uint64_t record_bytes(const Image& image) const {
        std::vector<std::byte> extra;
        Codec::write(image.header, extra);
        return sizeof(ContextImageRecordHeader) + profile_.size() + extra.size() +
               4 * image.tokens.size() + sizeof(ContextMediaMark) * image.media.size() +
               image.bytes;
    }

    // A record's header, and the family's fields into `extra`.
    ContextImageRecordHeader record_head(const Image& image, std::vector<std::byte>& extra) const {
        Codec::write(image.header, extra);
        ContextImageRecordHeader head;
        head.position      = image.header.position;
        head.path          = static_cast<std::uint32_t>(image.path);
        head.prompt_tokens = static_cast<std::uint32_t>(image.tokens.size());
        head.media_units   = static_cast<std::uint32_t>(image.media.size());
        head.extra_bytes   = static_cast<std::uint32_t>(extra.size());
        head.profile_bytes = static_cast<std::uint32_t>(profile_.size());
        head.bytes         = image.bytes;
        return head;
    }

    // One record of an image in host pages, whose copies have completed.
    std::uint64_t write_record(std::ostream& out, const Image& image) const {
        std::vector<std::byte> extra;
        const ContextImageRecordHeader head = record_head(image, extra);
        out.write(reinterpret_cast<const char*>(&head), sizeof(head));
        out.write(profile_.data(), std::streamsize(profile_.size()));
        out.write(reinterpret_cast<const char*>(extra.data()), std::streamsize(extra.size()));
        out.write(reinterpret_cast<const char*>(image.tokens.data()),
                  std::streamsize(4 * image.tokens.size()));
        out.write(reinterpret_cast<const char*>(image.media.data()),
                  std::streamsize(sizeof(ContextMediaMark) * image.media.size()));
        for (std::uint64_t at = 0; at < image.bytes; at += kPageBytes) {
            out.write(reinterpret_cast<const char*>(image.pages[at / kPageBytes]),
                      std::streamsize(std::min<std::uint64_t>(kPageBytes, image.bytes - at)));
        }
        return record_bytes(image);
    }

    // A record's head up to its bytes, checked against this store's profile and the executor's
    // image size. False when it is not a record of this profile.
    bool read_record_head(std::istream& in, Image& image) const {
        ContextImageRecordHeader head;
        if (!in.read(reinterpret_cast<char*>(&head), sizeof(head)) ||
            std::memcmp(head.magic, ContextImageRecordHeader{}.magic, 8) != 0 ||
            head.profile_bytes != profile_.size() || head.extra_bytes > (1U << 20)) {
            return false;
        }
        std::string profile(head.profile_bytes, '\0');
        std::vector<std::byte> extra(head.extra_bytes);
        if (!in.read(profile.data(), std::streamsize(profile.size())) || profile != profile_ ||
            !in.read(reinterpret_cast<char*>(extra.data()), std::streamsize(extra.size())) ||
            head.bytes != image_bytes_(head.position)) {
            return false;
        }
        image.header.position = head.position;
        if (!Codec::read(std::span<const std::byte>(extra), image.header)) { return false; }
        image.tokens.resize(head.prompt_tokens);
        image.media.resize(head.media_units);
        if (!in.read(reinterpret_cast<char*>(image.tokens.data()),
                     std::streamsize(4 * image.tokens.size())) ||
            !in.read(reinterpret_cast<char*>(image.media.data()),
                     std::streamsize(sizeof(ContextMediaMark) * image.media.size()))) {
            return false;
        }
        image.path  = static_cast<PrefixReusePath>(head.path);
        image.bytes = head.bytes;
        return true;
    }

    // A record's bytes into the image's pages, which no copy uses.
    static bool read_bytes(std::istream& in, Image& image) {
        for (std::uint64_t at = 0; in && at < image.bytes; at += kPageBytes) {
            in.read(reinterpret_cast<char*>(image.pages[at / kPageBytes]),
                    std::streamsize(std::min<std::uint64_t>(kPageBytes, image.bytes - at)));
        }
        return static_cast<bool>(in);
    }

    [[nodiscard]] static std::string file_name(const Image& image) {
        std::uint64_t hash = 1469598103934665603ULL; // FNV-1a over the token ids and media units
        const auto mix     = [&](std::uint64_t value, int bytes) {
            for (int i = 0; i < bytes; ++i) {
                hash ^= (value >> (8 * i)) & 0xffU;
                hash *= 1099511628211ULL;
            }
        };
        for (const TokenId token : image.tokens) { mix(std::uint32_t(token), 4); }
        for (const ContextMediaMark& mark : image.media) {
            mix(mark.begin, 4);
            mix(mark.count, 4);
            mix(mark.key, 8);
        }
        char name[40];
        std::snprintf(name, sizeof(name), "%016llx-%u.img", static_cast<unsigned long long>(hash),
                      static_cast<unsigned>(image.tokens.size()));
        return name;
    }

    // Reads the heads of the images a previous run left in the disk tier.
    void scan_disk() {
        for (const auto& entry : std::filesystem::directory_iterator(disk_dir_)) {
            if (entry.path().extension() != ".img") { continue; }
            std::ifstream in(entry.path(), std::ios::binary);
            Image image;
            if (!read_record_head(in, image) || index(image.tokens, image.media)) { continue; }
            image.file      = entry.path();
            image.last_used = ++clock_;
            disk_used_ += entry.file_size();
            images_.push_back(std::move(image));
        }
    }

    // Writes an image held in host pages to the disk tier, making room by deleting the least
    // recently used files; an image left with neither copy is dropped by the caller.
    void write_file(Image& image) {
        if (disk_dir_.empty() || !image.file.empty() || image.pages.empty()) { return; }
        const std::uint64_t size = record_bytes(image);
        if (size > disk_budget_) { return; }
        while (disk_used_ + size > disk_budget_) {
            Image* oldest = nullptr;
            for (Image& other : images_) {
                if (!other.file.empty() &&
                    (oldest == nullptr || other.last_used < oldest->last_used)) {
                    oldest = &other;
                }
            }
            if (oldest == nullptr) { return; }
            remove_file(*oldest);
        }
        if (sync_) { sync_(); }
        const auto path      = disk_dir_ / file_name(image);
        const auto temporary = std::filesystem::path(path).concat(".tmp");
        {
            std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
            (void)write_record(out, image);
            if (!out) {
                std::error_code ignored;
                std::filesystem::remove(temporary, ignored);
                return;
            }
        }
        std::error_code failed;
        std::filesystem::rename(temporary, path, failed);
        if (failed) { return; }
        image.file = path;
        disk_used_ += size;
    }

    void remove_file(Image& image) {
        if (image.file.empty()) { return; }
        std::error_code ignored;
        const auto bytes = std::filesystem::file_size(image.file, ignored);
        disk_used_ -= std::min<std::uint64_t>(disk_used_, ignored ? 0 : bytes);
        std::filesystem::remove(image.file, ignored);
        image.file.clear();
    }

    // Reads an image only on disk into host pages, for which the caller has made room.
    bool read_file(Image& image) {
        image.pages = take_pages(image.bytes);
        if (image.pages.empty()) { return false; }
        if (sync_) { sync_(); }
        std::ifstream in(image.file, std::ios::binary);
        Image head;
        if (!read_record_head(in, head) || head.tokens != image.tokens ||
            head.media != image.media || !read_bytes(in, image)) {
            give_back(image);
            return false;
        }
        return true;
    }

    std::uint64_t host_budget_ = 0, host_reserve_ = 0, host_used_ = 0;
    std::uint64_t disk_budget_ = 0, disk_used_ = 0;
    bool disk_restore_      = false;
    bool pin_ahead_failed_  = false;
    std::uint64_t clock_    = 0;
    std::string profile_;
    std::filesystem::path disk_dir_;
    std::function<std::uint64_t(std::uint32_t)> image_bytes_;
    std::function<void()> sync_;
    std::vector<std::unique_ptr<HostPages>> chunks_;
    std::vector<std::byte*> free_pages_;
    std::vector<Image> images_;
};

// The prefix cache file (--prefix-cache-file) of a family's store: read at startup and written at
// a clean stop, each reported through the Engine's diagnostics.
template <class Store>
void restore_prefix_cache(Store& store, const std::filesystem::path& file,
                          std::string_view identity, const DiagnosticObserver& diagnostics) {
    if (file.empty() || !store.enabled()) { return; }
    std::error_code missing;
    if (!std::filesystem::exists(file, missing)) {
        publish_diagnostic(diagnostics, DiagnosticLevel::Info,
                           "prefix cache: %s does not exist yet; it is written at a clean stop",
                           file.string().c_str());
        return;
    }
    const auto started       = std::chrono::steady_clock::now();
    const std::uint32_t kept = store.load_file(file, identity);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (kept == 0) {
        publish_diagnostic(diagnostics, DiagnosticLevel::Warning,
                           "prefix cache not restored: %s holds no image of this build, model and "
                           "configuration",
                           file.string().c_str());
        return;
    }
    std::error_code unknown;
    const auto bytes = std::filesystem::file_size(file, unknown);
    publish_diagnostic(diagnostics, DiagnosticLevel::Info,
                       "prefix cache restored: %u image%s from %.1f MiB in %.1f s", kept,
                       kept == 1 ? "" : "s",
                       unknown ? 0.0 : static_cast<double>(bytes) / 1048576.0, seconds);
}

template <class Store>
void save_prefix_cache(Store& store, const std::filesystem::path& file, std::string_view identity,
                       const PrefixCacheSaveControl& control,
                       const DiagnosticObserver& diagnostics) {
    const auto started                       = std::chrono::steady_clock::now();
    const std::optional<std::uint32_t> saved = store.save_file(file, identity, control);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (!saved) {
        publish_diagnostic(diagnostics, DiagnosticLevel::Warning,
                           "prefix cache not saved: the save failed or was abandoned");
        return;
    }
    std::error_code unknown;
    const auto bytes = std::filesystem::file_size(file, unknown);
    publish_diagnostic(diagnostics, DiagnosticLevel::Info,
                       "prefix cache saved: %u image%s, %.1f MiB in %.1f s", *saved,
                       *saved == 1 ? "" : "s",
                       unknown ? 0.0 : static_cast<double>(bytes) / 1048576.0, seconds);
}

// A clean stop's write of the images only in host pages to the disk tier, reported.
template <class Store>
void spill_to_disk_tier(Store& store, const DiagnosticObserver& diagnostics) {
    if (!store.has_disk_tier()) { return; }
    const auto [written, bytes] = store.spill_to_disk();
    if (written != 0) {
        publish_diagnostic(diagnostics, DiagnosticLevel::Info,
                           "disk tier: %u image%s, %.1f MiB written at the stop", written,
                           written == 1 ? "" : "s", static_cast<double>(bytes) / 1048576.0);
    }
}

} // namespace ninfer::runtime
