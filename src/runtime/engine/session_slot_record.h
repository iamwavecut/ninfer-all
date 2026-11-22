#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::runtime {

// One session slot saved by a family with an Engine core of its own (session_slots.h): the
// record the Engine writes to the slot file, its tokens and session digest.
struct SessionSlotRecord {
    std::vector<std::uint8_t> bytes;
    std::uint32_t tokens = 0;
    std::string session_digest;
};

} // namespace ninfer::runtime
