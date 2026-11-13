// The Engram table a DeepSeek-V4.1 model reads and the hot-row profile a table artifact may store:
// engram_table_source() reads `engram.resources.hot_profile` only when asked, accepts a profile
// counted for the model's Engram hash and token map, and refuses one counted for another; the
// fingerprint follows every value that moves a row.

#include "artifact/framing.h"
#include "artifact/reader.h"
#include "models/deepseek_v41/engram.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer::models::deepseek_v41;
using ninfer::artifact::Json;

namespace {

constexpr std::uint32_t kVocab = 16;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

void refuses(const std::function<void()>& fn, const std::string& needle,
             const std::string& label) {
    try {
        fn();
    } catch (const std::exception& error) {
        require(std::string(error.what()).find(needle) != std::string::npos,
                label + ": unexpected message: " + error.what());
        return;
    }
    throw std::runtime_error("not refused: " + label);
}

// Two Engram blocks of 40 and 50 rows, 2- and 3-grams, two heads each.
TextConfig text_config() {
    TextConfig config;
    config.vocab_size = kVocab;
    auto& e           = config.engram;
    e.layers          = {1, 3};
    e.layer_rows      = {40, 50};
    e.ngram_size      = 3;
    e.heads           = 2;
    e.head_dim        = 256;
    e.pad_id          = 0;
    e.multipliers     = {{3, 5, 7}, {11, 13, 17}};
    e.primes          = {{7, 11, 13, 5}, {13, 11, 17, 7}};
    e.offsets         = {{0, 7, 18, 31}, {0, 13, 24, 41}};
    return config;
}

std::uint64_t table_rows(const TextConfig& config) {
    std::uint64_t rows = 0;
    for (const auto r : config.engram.layer_rows) { rows += r; }
    return rows;
}

Json descriptor(const TextConfig& config, const std::string& digest) {
    return {{"architectures", {"DeepseekV41EngramTable"}},
            {"model_type", "deepseek_v41_engram"},
            {"layers", config.engram.layers},
            {"layer_rows", config.engram.layer_rows},
            {"rows", table_rows(config)},
            {"row_width", config.engram.head_dim},
            {"format", "gguf_iq4_nl"},
            {"table_sha256", digest}};
}

std::vector<std::byte> token_map_bytes(const std::vector<std::int32_t>& map) {
    std::vector<std::byte> out(map.size() * 4);
    for (std::size_t i = 0; i < map.size(); ++i) {
        for (unsigned b = 0; b < 4; ++b) {
            out[i * 4 + b] = static_cast<std::byte>((std::uint32_t(map[i]) >> (8U * b)) & 255U);
        }
    }
    return out;
}

std::vector<std::byte> profile_bytes(std::uint64_t rows, std::uint64_t fingerprint,
                                     const std::vector<std::uint32_t>& ranked) {
    std::vector<std::byte> out;
    for (const char c : std::string("NFNGHOT1")) { out.push_back(static_cast<std::byte>(c)); }
    const auto append = [&](std::uint64_t value, unsigned bytes) {
        for (unsigned i = 0; i < bytes; ++i) {
            out.push_back(static_cast<std::byte>((value >> (8U * i)) & 255U));
        }
    };
    append(rows, 8);
    append(fingerprint, 8);
    append(4321, 8);
    append(ranked.size(), 8);
    for (const auto row : ranked) { append(row, 4); }
    return out;
}

// A stored object: its directory entry without an offset, and its bytes (zeros when empty).
struct Object {
    Json entry;
    std::uint64_t bytes = 0;
    std::vector<std::byte> content;
};

class Directory {
public:
    Directory() {
        std::random_device entropy;
        root_ = std::filesystem::temp_directory_path() /
                ("ninfer-engram-table-" + std::to_string(entropy()));
        std::filesystem::create_directories(root_);
    }
    ~Directory() {
        std::error_code error;
        std::filesystem::remove_all(root_, error);
    }
    Directory(const Directory&)            = delete;
    Directory& operator=(const Directory&) = delete;

    // A one-file artifact; objects follow each other at 256-byte boundaries.
    std::filesystem::path write(const std::string& name, const Json& components,
                                std::vector<Object> objects, const Json& bindings) const {
        Json listed           = Json::array();
        std::uint64_t payload = 0;
        std::vector<std::uint64_t> offsets;
        for (auto& o : objects) {
            offsets.push_back(payload);
            o.entry["offset"] = payload;
            o.entry["bytes"]  = o.bytes;
            listed.push_back(o.entry);
            payload += (o.bytes + 255) / 256 * 256;
        }
        const Json root = {{"components", components},
                           {"objects", listed},
                           {"bindings", bindings},
                           {"uses", Json::array()},
                           {"files", Json::array({{{"path", nullptr}, {"payload_bytes", payload}}})}};
        const std::string text = root.dump();
        std::vector<char> header(ninfer::artifact::kHeaderBytes, 0);
        for (std::size_t i = 0; i < 8; ++i) {
            header[i] = static_cast<char>(ninfer::artifact::kEntryMagic[i]);
        }
        for (unsigned i = 0; i < 8; ++i) {
            header[8 + i] = static_cast<char>((text.size() >> (8 * i)) & 255);
        }
        header[16]          = 0x5a;
        const auto path     = root_ / name;
        const auto position = header.size() + text.size();
        const auto aligned  = (position + ninfer::artifact::kPayloadAlignment - 1) /
                             ninfer::artifact::kPayloadAlignment *
                             ninfer::artifact::kPayloadAlignment;
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file.write(header.data(), static_cast<std::streamsize>(header.size()));
        file.write(text.data(), static_cast<std::streamsize>(text.size()));
        const std::vector<char> zeros(aligned - position + payload, 0);
        file.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
        for (std::size_t i = 0; i < objects.size(); ++i) {
            if (objects[i].content.empty()) { continue; }
            file.seekp(static_cast<std::streamoff>(aligned + offsets[i]));
            file.write(reinterpret_cast<const char*>(objects[i].content.data()),
                       static_cast<std::streamsize>(objects[i].content.size()));
        }
        return path;
    }

private:
    std::filesystem::path root_;
};

Object resource(const std::string& id, std::vector<std::byte> content) {
    const std::uint64_t bytes = content.size();
    return {{{"id", id}, {"kind", "resource"}, {"encoding", "raw_bytes_v1"}}, bytes,
            std::move(content)};
}

Object table_tensor(std::uint64_t rows) {
    return {{{"id", "table"},
             {"kind", "tensor"},
             {"shape", {rows, 256}},
             {"format", "gguf_iq4_nl"},
             {"layout", "gguf_blocks_v1"}},
            rows * kEngramRowBytes,
            {}};
}

int run() {
    const TextConfig config  = text_config();
    const std::uint64_t rows = table_rows(config);
    const std::string digest(64, 'c');
    std::vector<std::int32_t> map(kVocab);
    for (std::uint32_t t = 0; t < kVocab; ++t) { map[t] = std::int32_t(t % 9); }
    const Directory dir;

    // The model, its table stored apart.
    const auto model = dir.write(
        "model.ninfer",
        {{"text", {{"config", Json::object()}, {"resources", {{"engram_token_map", "map"}}}}},
         {"engram", {{"config", descriptor(config, digest)}}}},
        {Object{{{"id", "w"}, {"kind", "tensor"}, {"shape", {4}}, {"format", "fp32"},
                 {"layout", "contiguous_le_v1"}},
                16,
                {}},
         resource("map", token_map_bytes(map))},
        {{"text/weight", {{"object", "w"}}}});
    const ninfer::artifact::Reader model_reader(model);
    require(engram_token_map(model_reader, config) == map, "the token map the model stores");

    // The fingerprint follows the hash, the geometry and the token map. The converter computes the
    // same value (tests/convert/test_deepseek_v41.py, test_the_engram_fingerprint_matches_the_engine).
    const std::uint64_t fingerprint = engram_hash_fingerprint(config.engram, map);
    require(fingerprint == 0xae571dffa7a86675ULL, "the fingerprint's definition changed");
    const auto differs              = [&](const std::function<void(TextConfig&)>& change,
                             const std::string& label) {
        TextConfig other = config;
        change(other);
        require(engram_hash_fingerprint(other.engram, map) != fingerprint,
                "the fingerprint ignores " + label);
    };
    differs([](TextConfig& c) { c.engram.multipliers[1][2] += 2; }, "a multiplier");
    differs([](TextConfig& c) { c.engram.primes[0][3] = 3; }, "a prime");
    differs([](TextConfig& c) { c.engram.offsets[1][1] += 1; }, "an offset");
    differs([](TextConfig& c) { c.engram.pad_id = 1; }, "the pad id");
    differs([](TextConfig& c) { c.engram.layers[1] = 4; }, "a block");
    differs([](TextConfig& c) { c.engram.layer_rows[0] = 41; }, "a block's rows");
    auto remapped = map;
    remapped[5]   = 7;
    require(engram_hash_fingerprint(config.engram, remapped) != fingerprint,
            "the fingerprint ignores the token map");

    // The hasher's rows lie in the table, so a profile counted from them passes the check.
    const EngramHasher hasher(config.engram, map);
    std::vector<std::int32_t> tokens = {3, 9, 12, 3, 9, 15, 1};
    std::vector<std::uint64_t> ids(tokens.size() * hasher.rows_per_token());
    EngramHasher::History history;
    hasher.rows(history, tokens, ids.data());
    std::vector<std::uint32_t> ranked;
    for (const auto id : ids) {
        require(id < rows, "a hashed row past the table");
        if (std::find(ranked.begin(), ranked.end(), std::uint32_t(id)) == ranked.end()) {
            ranked.push_back(std::uint32_t(id));
        }
    }

    const auto table_with = [&](const std::string& name, const std::vector<std::byte>& profile) {
        Json engram = {{"config", descriptor(config, digest)}};
        std::vector<Object> objects{table_tensor(rows)};
        if (!profile.empty()) {
            engram["resources"] = {{"hot_profile", "hot"}};
            objects.push_back(resource("hot", profile));
        }
        return dir.write(name, {{"engram", engram}}, std::move(objects),
                         {{"engram/table", {{"object", "table"}}}});
    };
    const auto plain = table_with("plain.ninfer", {});
    const auto hot   = table_with("hot.ninfer", profile_bytes(rows, fingerprint, ranked));

    const auto bare = engram_table_source(model_reader, model, config, plain, true);
    require(bare.layout.rows == rows && bare.layout.row_bytes == kEngramRowBytes &&
                bare.layout.segments.size() == 1 && bare.layout.segments[0].path == plain &&
                !bare.hot_profile,
            "a table without a profile");
    require(!engram_table_source(model_reader, model, config, hot).hot_profile,
            "the profile stays unread unless asked for");
    const auto with = engram_table_source(model_reader, model, config, hot, true);
    require(with.hot_profile && with.hot_profile->rows == ranked &&
                with.hot_profile->tokens == 4321 && with.layout.segments[0].path == hot,
            "the table artifact's profile, in rank order");

    // A profile of another token map, of another table size, or with repeated rows is refused.
    const auto foreign = table_with(
        "foreign.ninfer",
        profile_bytes(rows, engram_hash_fingerprint(config.engram, remapped), ranked));
    (void)engram_table_source(model_reader, model, config, foreign);
    refuses([&] { (void)engram_table_source(model_reader, model, config, foreign, true); },
            "another n-gram hash", "a profile of another token map");
    const auto smaller =
        table_with("smaller.ninfer", profile_bytes(rows - 1, fingerprint, ranked));
    refuses([&] { (void)engram_table_source(model_reader, model, config, smaller, true); },
            "another n-gram hash", "a profile of another table size");
    auto repeated = ranked;
    repeated.push_back(ranked.front());
    const auto twice = table_with("twice.ninfer", profile_bytes(rows, fingerprint, repeated));
    refuses([&] { (void)engram_table_source(model_reader, model, config, twice, true); },
            "duplicate or out-of-range", "a profile that repeats a row");
    ninfer::models::qwen4_exp::NgramProfile past{
        .table_rows = rows, .fingerprint = fingerprint, .tokens = 1, .rows = {std::uint32_t(rows)}};
    refuses([&] { check_engram_profile(past, model_reader, config, "past"); },
            "duplicate or out-of-range", "a profile row past the table");

    // A model without its token map has no fingerprint to check against.
    const auto mapless = dir.write(
        "mapless.ninfer",
        {{"text", {{"config", Json::object()}}}, {"engram", {{"config", descriptor(config, digest)}}}},
        {Object{{{"id", "w"}, {"kind", "tensor"}, {"shape", {4}}, {"format", "fp32"},
                 {"layout", "contiguous_le_v1"}},
                16,
                {}}},
        {{"text/weight", {{"object", "w"}}}});
    refuses(
        [&] {
            (void)engram_table_source(ninfer::artifact::Reader(mapless), mapless, config, hot, true);
        },
        "token map", "a model without its token map");
    refuses([&] { (void)decode_engram_token_map(token_map_bytes({1, -2}), 2); }, "negative",
            "a negative compressed id");
    refuses([&] { (void)decode_engram_token_map(token_map_bytes({1, 2}), 3); },
            "a compressed id per token", "a token map of another vocabulary");
    std::cout << "PASS deepseek_v41 engram table and hot-row profile\n";
    return 0;
}

} // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
