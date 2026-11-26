#pragma once

// The paging queue's admission policy arithmetic, pure so the unit tests can lock the
// contracts: the generation grant size per reserve mode, the extension target of a decoding
// lane, and the Retry-After seconds of a queue timeout.

#include <algorithm>
#include <cstdint>

namespace ninfer::runtime::kv_queue {

// The tokens admission reserves for a request: its prompt plus, in chunked mode, only the
// first `chunk` generation tokens (the grant then grows per decode round). `chunk` 0 or a
// chunk at or past `output` grants the whole generation.
[[nodiscard]] constexpr std::uint32_t generation_grant_tokens(std::uint32_t prompt,
                                                              std::uint32_t output,
                                                              std::uint32_t chunk) {
    return prompt + std::min(output, chunk == 0 ? output : chunk);
}

// The tokens a decoding lane at `position` asks the pool to grant: its in-flight token plus
// the generation budget it still may use, capped at the chunk size. Once the remaining budget
// fits the chunk, the grant covers the request to completion, so its final stretch never
// straddles an extension.
[[nodiscard]] constexpr std::uint32_t extension_target_tokens(std::uint32_t position,
                                                              std::uint32_t remaining,
                                                              std::uint32_t chunk) {
    return position + 1 + std::min(remaining, chunk == 0 ? remaining : chunk);
}

// The Retry-After seconds of a queue timeout: the milliseconds the request still had before
// its deadline, as whole seconds, clamped to at least one.
[[nodiscard]] constexpr std::uint32_t retry_after_seconds(std::uint64_t remaining_ms) {
    return static_cast<std::uint32_t>(std::max<std::uint64_t>(1, (remaining_ms + 999) / 1000));
}

} // namespace ninfer::runtime::kv_queue
