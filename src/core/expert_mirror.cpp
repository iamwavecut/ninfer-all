#include "core/expert_mirror.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace ninfer {
namespace {

constexpr std::uint64_t kCompared = 64ULL << 10;

std::vector<char> read_range(const std::filesystem::path& path, std::uint64_t offset,
                             std::uint64_t bytes) {
    std::ifstream file(path, std::ios::binary);
    std::vector<char> out(static_cast<std::size_t>(bytes));
    file.seekg(static_cast<std::streamoff>(offset));
    file.read(out.data(), static_cast<std::streamsize>(out.size()));
    if (!file) { throw std::invalid_argument("--expert-mirror: cannot read " + path.string()); }
    return out;
}

} // namespace

std::vector<std::filesystem::path> expert_mirror_files(std::span<const std::filesystem::path> files,
                                                       const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> out;
    if (directory.empty()) { return out; }
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) {
        throw std::invalid_argument("--expert-mirror: " + directory.string() +
                                    " is not a directory");
    }
    for (const auto& original : files) {
        const auto copy = directory / original.filename();
        const auto size = std::filesystem::file_size(original, error);
        if (error) {
            throw std::invalid_argument("--expert-mirror: cannot stat " + original.string());
        }
        if (!std::filesystem::is_regular_file(copy, error) ||
            std::filesystem::file_size(copy, error) != size || error) {
            throw std::invalid_argument("--expert-mirror: " + copy.string() + " is not a copy of " +
                                        original.string() + " (missing or another size)");
        }
        const std::uint64_t head = std::min(size, kCompared);
        const std::uint64_t tail = std::min(size - head, kCompared);
        if (read_range(original, 0, head) != read_range(copy, 0, head) ||
            read_range(original, size - tail, tail) != read_range(copy, size - tail, tail)) {
            throw std::invalid_argument("--expert-mirror: " + copy.string() + " differs from " +
                                        original.string());
        }
        out.push_back(copy);
    }
    return out;
}

} // namespace ninfer
