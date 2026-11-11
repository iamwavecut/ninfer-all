#pragma once

// Disk experts read from two drives: a directory holding byte-identical copies of the artifact's
// files (by file name) on a second drive, the mirror. Each read goes to one of the two copies by
// the stripe of its file offset, so the misses of a call spread over both drives, and a range is
// always read from the same copy, so the OS page cache never holds it twice.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace ninfer {

// Bytes of one stripe: a read starting in an even stripe goes to the artifact's copy, in an odd
// one to the mirror's.
inline constexpr std::uint64_t kExpertMirrorStripe = 1ULL << 20;

// 0: the artifact's copy, 1: the mirror's.
[[nodiscard]] constexpr std::size_t expert_mirror_copy(std::uint64_t offset) noexcept {
    return static_cast<std::size_t>((offset / kExpertMirrorStripe) & 1U);
}

// The mirror copies of `files` in `directory`: each the same file name, a regular file of the same
// size whose first and last 64 KiB equal the original's. Throws std::invalid_argument naming the
// first copy that is missing or differs. An empty `directory` gives no copies.
[[nodiscard]] std::vector<std::filesystem::path>
expert_mirror_files(std::span<const std::filesystem::path> files,
                    const std::filesystem::path& directory);

} // namespace ninfer
