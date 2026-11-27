#pragma once

// Model artifacts named by their Hugging Face repository, `OWNER/REPO` or `OWNER/REPO:FILE`, where a
// program takes its artifact path. The file is fetched once into the Hugging Face cache, in the
// layout `hf download` writes, so either tool finds what the other fetched, and the program runs the
// cached file.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::product::hub {

struct Reference {
    std::string repository; // OWNER/REPO
    std::string file;       // a path in the repository; empty: its one .ninfer file
};

// `argument` as a reference when nothing exists at that path: OWNER/REPO, then optionally ':' and
// FILE. OWNER and REPO hold letters, digits, '-', '_' and '.', start with a letter or digit and
// contain no "..". A path that ends in ".ninfer" and has no ":FILE" is not a reference, so a
// mistyped local artifact is reported as a missing file and not looked up on the Hub.
[[nodiscard]] std::optional<Reference> parse_reference(std::string_view argument);

// HF_HUB_CACHE, else HF_HOME/hub, else XDG_CACHE_HOME/huggingface/hub, else
// ~/.cache/huggingface/hub (%USERPROFILE%\.cache\huggingface\hub on Windows).
[[nodiscard]] std::filesystem::path cache_directory();

// A repository's folder in the cache: models--OWNER--REPO.
[[nodiscard]] std::filesystem::path repository_directory(const std::filesystem::path& cache,
                                                         std::string_view repository);

// The pieces a download is split into, and the ones a resumed download still needs: a piece is
// `kPieceBytes` long, the last one shorter.
inline constexpr std::uint64_t kPieceBytes = 64ULL << 20;
[[nodiscard]] inline std::uint64_t piece_count(std::uint64_t size) noexcept {
    return (size + kPieceBytes - 1) / kPieceBytes;
}

// Progress and status lines for the person who started the program.
using Report = std::function<void(const std::string&)>;

// The local path of the reference's file at the repository's main revision. A file the cache holds
// at that revision is used as it is; any other is downloaded into the cache over 16 connections,
// resumed after an interruption, and checked against the SHA-256 the Hub records for it. HF_TOKEN,
// or the token file `hf auth login` writes, reaches gated and private repositories; with
// HF_HUB_OFFLINE=1, or when the Hub cannot be reached, the cache's last revision serves alone.
// Throws std::runtime_error naming what failed.
[[nodiscard]] std::filesystem::path fetch(const Reference& reference, const Report& report);

// `argument` itself when it names no reference, else the local path fetch() returns.
[[nodiscard]] std::string resolve_artifact(const std::string& argument, const Report& report);

// The usage lines of a program that takes its artifact as the first argument.
inline constexpr const char* kArtifactReferenceHelp =
    "  <model.ninfer> may also be OWNER/REPO or OWNER/REPO:FILE, a Hugging Face\n"
    "                                repository's artifact, fetched once into the Hub's\n"
    "                                cache (HF_TOKEN reaches gated and private ones)\n";

} // namespace ninfer::product::hub
