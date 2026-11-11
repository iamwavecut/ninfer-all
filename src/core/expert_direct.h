#pragma once

// Direct expert reads (ExpertDirect, --expert-direct): on Linux with the kernel modules of
// kernel/ninfer-direct loaded, the NVMe drive writes the bytes of disk experts straight into a
// window of device memory (peer-to-peer DMA into the GPU's BAR1), and CUDA copies them from the
// window to where they belong, so host memory carries none of them. A read covers whole 4 KiB
// blocks of its file: its window cover starts at a 4 KiB-aligned window offset, and the range's
// first byte follows direct_lead() bytes of the block it starts in.

#include "core/expert_disk.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer {

inline constexpr std::uint64_t kDirectBlock = 4096;

// The bytes before `offset` in its 4 KiB block.
[[nodiscard]] constexpr std::uint64_t direct_lead(std::uint64_t offset) noexcept {
    return offset % kDirectBlock;
}

// Window bytes a read of [offset, offset + bytes) takes: the whole blocks that hold it.
[[nodiscard]] constexpr std::uint64_t direct_cover(std::uint64_t offset,
                                                   std::uint64_t bytes) noexcept {
    return (direct_lead(offset) + bytes + kDirectBlock - 1) / kDirectBlock * kDirectBlock;
}

// [offset, offset + bytes) of window file `file` read into the window at `window_offset` (a
// multiple of kDirectBlock); its first byte lands at window_offset + direct_lead(offset).
struct DirectRead {
    std::size_t file            = 0;
    std::uint64_t offset        = 0;
    std::uint64_t bytes         = 0;
    std::uint64_t window_offset = 0;
};

// A run of a file on its block device: the file's [logical, logical + bytes) at `physical`.
struct FileExtent {
    std::uint64_t logical  = 0;
    std::uint64_t physical = 0;
    std::uint64_t bytes    = 0;
};

// The device runs that hold [offset, offset + bytes) of a file whose extents are `extents`
// (sorted by logical offset), in file order, physically adjacent runs joined. Throws
// std::runtime_error when the extents do not cover the range.
[[nodiscard]] std::vector<FileExtent> direct_device_runs(std::span<const FileExtent> extents,
                                                         std::uint64_t offset, std::uint64_t bytes);

class DirectExpertWindow {
public:
    enum class Refusal {
        // This build, OS or machine has no direct path, or its modules are not loaded.
        NotSetUp,
        // The modules are loaded, but a step failed: driver, kernel, drive, filesystem, access,
        // or a self-test read that returned other bytes than a normal read.
        Failed,
    };

    // A window of at least `bytes` (rounded up to whole 64 KiB pages) on CUDA device `device` for
    // reads of `files` (the artifact's files, then a drive mirror's), after a self-test read of
    // each file over the window. Null with the reason in `why` when the direct path is not
    // available. `host_window` (tests): a pinned host window filled by reads of the files
    // themselves, which runs every step but the kernel modules' and the drive's DMA.
    [[nodiscard]] static std::unique_ptr<DirectExpertWindow>
    open(int device, std::span<const std::filesystem::path> files, std::uint64_t bytes,
         bool host_window, Refusal& refusal, std::string& why);

    virtual ~DirectExpertWindow() = default;

    // The window's first byte: CUDA copies on its device read from here (cudaMemcpyDefault).
    [[nodiscard]] virtual const std::byte* data() const noexcept = 0;
    [[nodiscard]] virtual std::uint64_t bytes() const noexcept   = 0;

    // Reads every range, then makes its bytes visible to the device's later work. The covers must
    // lie inside the window and not overlap each other or a range a queued copy still reads.
    // Throws FileReadError when a read fails, after every read has ended.
    virtual void read(std::span<const DirectRead> reads) = 0;

    // What fills the window, for the startup line.
    [[nodiscard]] virtual std::string describe() const = 0;
};

// The window that disk.direct asks for (`bytes` on `device` for `files`): none with Off; with Auto
// none when the direct path is unavailable, after a line on stderr naming the reason unless its
// modules are not set up; with On a std::runtime_error naming the reason. With a window, a line on
// stderr describes it. `owner` names the reader in those lines.
[[nodiscard]] std::unique_ptr<DirectExpertWindow>
open_direct_window(const ExpertDiskReads& disk, int device,
                   std::span<const std::filesystem::path> files, std::uint64_t bytes,
                   const char* owner);

} // namespace ninfer
