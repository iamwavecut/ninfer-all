#pragma once

// Integrity check over a whole session slot file, stored as a trailing u64. A bit flip that keeps
// every length and domain valid would otherwise restore silently corrupted KV or state. Word-wise,
// so a multi-GB snapshot hashes at memory speed; not cryptographic, since the files are trusted
// local state and the threat is corruption, not tampering.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace ninfer {

[[nodiscard]] inline std::uint64_t snapshot_checksum(std::span<const std::uint8_t> bytes) noexcept {
    std::uint64_t hash      = 0x9e3779b97f4a7c15ULL ^ static_cast<std::uint64_t>(bytes.size());
    const std::size_t words = bytes.size() / sizeof(std::uint64_t);
    for (std::size_t index = 0; index < words; ++index) {
        std::uint64_t word = 0;
        std::memcpy(&word, bytes.data() + index * sizeof(std::uint64_t), sizeof(word));
        hash = (hash ^ word) * 0xff51afd7ed558ccdULL;
        hash ^= hash >> 32U;
    }
    for (std::size_t index = words * sizeof(std::uint64_t); index < bytes.size(); ++index) {
        hash = (hash ^ bytes[index]) * 0xc4ceb9fe1a85ec53ULL;
        hash ^= hash >> 29U;
    }
    return hash;
}

} // namespace ninfer
