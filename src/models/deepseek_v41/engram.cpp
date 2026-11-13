#include "models/deepseek_v41/engram.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ninfer::models::deepseek_v41 {
namespace {

using artifact::ArtifactError;
using artifact::Json;

constexpr const char* kTableBinding = "engram/table";

// What an `engram` component says about its table, checked against the model's text config.
struct Descriptor {
    std::string format;
    std::string digest; // SHA-256 of the table's bytes, lowercase hex
    std::uint64_t rows = 0;
};

Descriptor descriptor(const artifact::Reader& reader, const std::filesystem::path& artifact,
                      const TextConfig& config) {
    const auto& directory = reader.directory();
    if (!directory.components.contains("engram")) {
        throw ArtifactError(artifact.string() +
                            " describes no Engram table (its `engram` component); convert it "
                            "with its table source, --source engram=PATH");
    }
    const Json& c = directory.component("engram").config;
    artifact::require_members(c,
                              {"architectures", "model_type", "layers", "layer_rows", "rows",
                               "row_width", "format", "table_sha256"},
                              {}, "Engram table config");
    if (!c.at("architectures").is_array() || c.at("architectures").size() != 1 ||
        c.at("architectures")[0] != "DeepseekV41EngramTable") {
        throw ArtifactError(artifact.string() +
                            ": the `engram` component is not a DeepSeek-V4.1 Engram table");
    }
    const auto& hash = config.engram;
    std::vector<std::uint64_t> layers, layer_rows;
    for (const auto& v : c.at("layers")) { layers.push_back(artifact::require_u64(v, "layers")); }
    for (const auto& v : c.at("layer_rows")) {
        layer_rows.push_back(artifact::require_u64(v, "layer_rows"));
    }
    std::uint64_t rows = 0;
    for (const auto r : hash.layer_rows) { rows += r; }
    if (layers != std::vector<std::uint64_t>(hash.layers.begin(), hash.layers.end()) ||
        layer_rows != hash.layer_rows || artifact::require_u64(c.at("rows"), "rows") != rows ||
        artifact::require_u64(c.at("row_width"), "row_width") != hash.head_dim) {
        throw ArtifactError(artifact.string() +
                            ": the Engram table's geometry differs from the model's");
    }
    Descriptor out;
    out.rows   = rows;
    out.format = artifact::require_id(c.at("format"), "Engram table format");
    if (out.format != "gguf_iq4_nl") {
        throw ArtifactError(artifact.string() + ": the Engram table is stored as " + out.format +
                            "; the runtime decodes gguf_iq4_nl rows");
    }
    if (!c.at("table_sha256").is_string()) {
        throw ArtifactError(artifact.string() + ": table_sha256 must be a string");
    }
    out.digest = c.at("table_sha256").get<std::string>();
    if (out.digest.size() != 64 ||
        out.digest.find_first_not_of("0123456789abcdef") != std::string::npos) {
        throw ArtifactError(artifact.string() +
                            ": table_sha256 must be 64 lowercase hexadecimal digits");
    }
    return out;
}

qwen4_exp::NgramTableLayout stored_rows(const artifact::Reader& reader,
                                        const std::filesystem::path& artifact,
                                        const Descriptor& d) {
    const auto& directory = reader.directory();
    const auto found      = directory.bindings.find(kTableBinding);
    if (found == directory.bindings.end() || found->second.parts.size() != 1) {
        throw ArtifactError(artifact.string() + ": the Engram table must be one stored object");
    }
    const auto& part   = found->second.parts.front();
    const auto& tensor = directory.tensor(part.object);
    if (tensor.format != d.format) {
        throw ArtifactError(artifact.string() + ": the Engram table is stored as " +
                            tensor.format + " but described as " + d.format);
    }
    qwen4_exp::NgramTableLayout out;
    out.row_bytes                   = kEngramRowBytes;
    out.rows                        = d.rows;
    const std::uint64_t table_bytes = d.rows * kEngramRowBytes;
    if (tensor.shape != artifact::Shape{d.rows, 256} || part.begin != 0 ||
        tensor.bytes != table_bytes) {
        throw ArtifactError(artifact.string() + ": the Engram table has an unexpected geometry");
    }
    (void)reader.geometry(part.object);
    for (const auto& segment : reader.segments(tensor.offset, table_bytes)) {
        const auto& record = directory.files.at(segment.file_index);
        const std::filesystem::path file =
            segment.file_index == 0 ? artifact : artifact.parent_path() / record.path.value();
        out.segments.push_back({file, segment.file_offset, segment.bytes});
    }
    return out;
}

} // namespace

bool is_engram_table_artifact(const artifact::Reader& reader) {
    const auto& directory = reader.directory();
    return !directory.components.contains("text") && directory.components.contains("engram") &&
           directory.bindings.contains(kTableBinding);
}

EngramTableSource engram_table_source(const artifact::Reader& reader,
                                      const std::filesystem::path& artifact,
                                      const TextConfig& config,
                                      const std::filesystem::path& table, bool read_hot_profile) {
    const Descriptor model = descriptor(reader, artifact, config);
    // The profile the selected table artifact stores, checked against the model's hash.
    const auto with_profile = [&](const artifact::Reader& selected,
                                  const std::filesystem::path& path,
                                  qwen4_exp::NgramTableLayout layout) {
        EngramTableSource out{.layout = std::move(layout), .hot_profile = std::nullopt};
        if (!read_hot_profile) { return out; }
        const auto& resources = selected.directory().component("engram").resources;
        const auto found      = resources.find("hot_profile");
        if (found == resources.end()) { return out; }
        out.hot_profile = qwen4_exp::decode_ngram_profile(selected.read_object(found->second),
                                                          path.string() + ":engram.hot_profile");
        check_engram_profile(*out.hot_profile, reader, config, path);
        return out;
    };
    if (table.empty()) {
        if (!reader.directory().bindings.contains(kTableBinding)) {
            throw ArtifactError(
                artifact.string() + ": the model's Engram table (sha256 " +
                model.digest.substr(0, 16) +
                "...) is stored separately; pass its table artifact with --ngram-table PATH. "
                "--no-ngram-table runs the model without it, a non-standard experimental mode "
                "that badly degrades the output");
        }
        return with_profile(reader, artifact, stored_rows(reader, artifact, model));
    }
    const artifact::Reader other(table);
    if (!other.directory().bindings.contains(kTableBinding)) {
        throw ArtifactError(table.string() + " stores no Engram table");
    }
    const Descriptor theirs = descriptor(other, table, config);
    if (theirs.digest != model.digest || theirs.format != model.format) {
        throw ArtifactError(table.string() + " holds a different Engram table (" + theirs.format +
                            ", sha256 " + theirs.digest.substr(0, 16) + "...) than the model's (" +
                            model.format + ", sha256 " + model.digest.substr(0, 16) + "...)");
    }
    return with_profile(other, table, stored_rows(other, table, theirs));
}

std::vector<std::int32_t> decode_engram_token_map(std::span<const std::byte> bytes,
                                                  std::uint32_t vocab_size) {
    if (bytes.size() != std::size_t(vocab_size) * sizeof(std::int32_t)) {
        throw ArtifactError("the Engram token map must hold a compressed id per token");
    }
    std::vector<std::int32_t> out(vocab_size);
    std::memcpy(out.data(), bytes.data(), bytes.size());
    if (std::any_of(out.begin(), out.end(), [](std::int32_t id) { return id < 0; })) {
        throw ArtifactError("the Engram token map holds a negative id");
    }
    return out;
}

std::vector<std::int32_t> engram_token_map(const artifact::Reader& reader,
                                           const TextConfig& config) {
    const auto& resources = reader.directory().component("text").resources;
    const auto found      = resources.find("engram_token_map");
    if (found == resources.end()) {
        throw ArtifactError("the DeepSeek-V4.1 model stores no Engram token map");
    }
    return decode_engram_token_map(reader.read_object(found->second), config.vocab_size);
}

std::uint64_t engram_hash_fingerprint(const EngramHash& hash,
                                      std::span<const std::int32_t> token_map) {
    std::uint64_t value = 0xcbf29ce484222325ULL;
    const auto feed     = [&](std::uint64_t word) {
        for (unsigned byte = 0; byte < 8; ++byte) {
            value ^= (word >> (8U * byte)) & 0xffU;
            value *= 0x100000001b3ULL;
        }
    };
    feed(hash.ngram_size);
    feed(hash.heads);
    feed(std::uint64_t(std::int64_t(hash.pad_id)));
    feed(hash.layers.size());
    for (const auto layer : hash.layers) { feed(layer); }
    for (std::size_t l = 0; l < hash.layers.size(); ++l) {
        feed(hash.layer_rows.at(l));
        for (const auto* values : {&hash.multipliers.at(l), &hash.primes.at(l), &hash.offsets.at(l)}) {
            for (const auto v : *values) { feed(v); }
        }
    }
    feed(token_map.size());
    for (const auto id : token_map) { feed(std::uint64_t(std::int64_t(id))); }
    return value;
}

void check_engram_profile(const qwen4_exp::NgramProfile& profile, const artifact::Reader& reader,
                          const TextConfig& config, const std::filesystem::path& path) {
    std::uint64_t rows = 0;
    for (const auto r : config.engram.layer_rows) { rows += r; }
    qwen4_exp::check_ngram_profile(
        profile, rows, engram_hash_fingerprint(config.engram, engram_token_map(reader, config)),
        path);
}

void EngramHasher::History::push(std::int32_t compressed_id, std::uint32_t keep) {
    if (keep == 0) { return; }
    if (count < keep) {
        ids[count++] = compressed_id;
        return;
    }
    std::copy(ids + 1, ids + keep, ids);
    ids[keep - 1] = compressed_id;
}

EngramHasher::EngramHasher(const EngramHash& hash, std::vector<std::int32_t> token_map)
    : hash_(hash), token_map_(std::move(token_map)) {
    if (hash_.ngram_size < 2 || hash_.ngram_size > 9) {
        throw std::invalid_argument("Engram: n-grams of 2 to 9 tokens");
    }
    std::uint64_t base = 0;
    for (const auto rows : hash_.layer_rows) {
        layer_base_.push_back(base);
        base += rows;
    }
}

std::int32_t EngramHasher::compressed(std::int32_t token) const {
    if (token < 0 || std::size_t(token) >= token_map_.size()) {
        throw std::out_of_range("Engram: token " + std::to_string(token) + " outside the token map");
    }
    return token_map_[std::size_t(token)];
}

void EngramHasher::rows(History& history, std::span<const std::int32_t> tokens,
                        std::uint64_t* out, bool advance, std::span<const std::uint8_t> dead) const {
    const std::uint32_t n = hash_.ngram_size, heads = hash_.heads, columns = hash_.columns();
    const std::uint32_t keep = n - 1;
    if (!dead.empty() && dead.size() != tokens.size()) {
        throw std::invalid_argument("Engram: a dead mark per token");
    }
    History h = history;
    std::uint64_t window[9];
    for (std::size_t t = 0; t < tokens.size(); ++t) {
        // window[s]: the compressed id s tokens back (s = 0: the token), the pad id from the first
        // dead position back and before the sequence's first token.
        const std::int32_t own = !dead.empty() && dead[t] ? kDead : compressed(tokens[t]);
        bool blocked = own == kDead;
        window[0]    = blocked ? std::uint64_t(hash_.pad_id) : std::uint64_t(own);
        for (std::uint32_t s = 1; s < n; ++s) {
            blocked   = blocked || s > h.count || h.ids[h.count - s] == kDead;
            window[s] = blocked ? std::uint64_t(hash_.pad_id) : std::uint64_t(h.ids[h.count - s]);
        }
        std::uint64_t* row = out + t * rows_per_token();
        for (std::size_t l = 0; l < hash_.layers.size(); ++l) {
            const auto& multipliers = hash_.multipliers[l];
            // Products and their XOR stay below 2^63, as the release's hash assumes.
            std::uint64_t rolling = window[0] * multipliers[0];
            for (std::uint32_t i = 1; i < n; ++i) {
                rolling ^= window[i] * multipliers[i];
                for (std::uint32_t head = 0; head < heads; ++head) {
                    const std::uint32_t c = (i - 1) * heads + head;
                    row[l * columns + c] =
                        layer_base_[l] + hash_.offsets[l][c] + rolling % hash_.primes[l][c];
                }
            }
        }
        h.push(own, keep);
    }
    if (advance) { history = h; }
}

} // namespace ninfer::models::deepseek_v41
