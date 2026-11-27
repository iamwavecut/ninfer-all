#pragma once

// Where the routed experts of a Qwen3.8-Flash-Next layer sit in the artifact's files, for disk
// residency. The model binds these at load; the miss service reads by them.

#include "core/qtype.h"
#include "core/weight.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen4_exp {

// Where one expert's rows of one projection sit in the artifact's files (disk-resident experts):
// one run of one file, or two when the rows straddle the boundary between two files.
struct ExpertLocation {
    struct Run {
        std::size_t file     = 0; // index into Model::files()
        std::uint64_t offset = 0; // byte offset in that file
        std::uint64_t bytes  = 0; // 0: unused
    };

    std::array<Run, 2> runs{};
    // A host expert's bytes in the pinned block, which its reader copies instead of the runs; null
    // for an expert the block could not hold (or a disk expert): read from the files.
    const std::byte* pinned = nullptr;
    std::uint64_t bytes    = 0; // of all runs
    QType format           = QType::GGUF_Q8_0;
    std::int64_t row_bytes = 0;
    std::int32_t rows      = 0;
    const std::byte* pinned_device = nullptr; // alias used when the GPU reads pinned bytes
};

} // namespace ninfer::models::qwen4_exp
