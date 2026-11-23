#pragma once

#include "artifact/materializer.h"

#include <string>

namespace ninfer::runtime {

// What ended the pinned block of host experts, for the startup line that names the pinned and the
// file amounts: " (pinned up to --expert-host-mib)", up to the host's available memory, up to what
// the GPU's shared memory segment can still lock less the headroom kept for later pins, or
// " (pinned until the host refused a part)". Empty when the block holds every expert.
[[nodiscard]] inline std::string pinned_block_end_note(const artifact::MaterializationStats& stats) {
    const auto mib = [](std::uint64_t bytes) { return std::to_string(bytes >> 20) + " MiB"; };
    switch (stats.pinned_block_end) {
    case artifact::PinnedBlockEnd::Complete: return {};
    case artifact::PinnedBlockEnd::Cap: return " (pinned up to --expert-host-mib)";
    case artifact::PinnedBlockEnd::Available:
        return " (pinned up to the host's available memory, " +
               mib(stats.pinned_block_limit_bytes) + ")";
    case artifact::PinnedBlockEnd::Pinnable:
        return " (pinned up to what the GPU's shared memory segment can still lock, " +
               mib(stats.pinned_block_limit_bytes) + ", less " +
               mib(artifact::kPinnedHeadroomBytes) + " for later pins)";
    case artifact::PinnedBlockEnd::Refused: return " (pinned until the host refused a part)";
    }
    return {};
}

} // namespace ninfer::runtime
