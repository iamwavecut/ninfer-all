#include "models/qwen3_5/program/prefix/block_keys.h"

#include "runtime/prefix_cache/block_hash.h"

#include <algorithm>
#include <array>
#include <bit>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {

namespace pc = runtime::prefix_cache;

constexpr std::uint64_t mix64(std::uint64_t state, std::uint64_t value) noexcept {
    state ^= value + 0x9e3779b97f4a7c15ULL + (state << 6U) + (state >> 2U);
    state *= 0xbf58476d1ce4e5b9ULL;
    return state ^ (state >> 31U);
}

constexpr std::uint64_t kVisionExtraSeed = 0x6e696e666572ULL;
constexpr std::uint64_t kLoraExtraSeed   = 0x6e696e6665722d6cULL;

} // namespace

std::vector<VisionTokenRange> vision_ranges(const PreparedPromptData& prompt) {
    std::vector<VisionTokenRange> ranges;
    ranges.reserve(prompt.vision_items.size());
    for (const VisionItem& item : prompt.vision_items) {
        if (item.token_spans.empty()) {
            throw std::logic_error("hybrid prefix cache: Vision item has no token span");
        }
        std::uint64_t key = 0x6e696e6665722d76ULL;
        for (const std::uint8_t byte : item.content_digest) { key = mix64(key, byte); }
        key = mix64(key, static_cast<std::uint64_t>(item.modality));
        key = mix64(key, static_cast<std::uint32_t>(item.grid.temporal));
        key = mix64(key, static_cast<std::uint32_t>(item.grid.height));
        key = mix64(key, static_cast<std::uint32_t>(item.grid.width));
        key = mix64(key, item.patch_count);
        for (const double timestamp : item.timestamps) {
            key = mix64(key, std::bit_cast<std::uint64_t>(timestamp));
        }
        for (const TokenSpan& span : item.token_spans) {
            key = mix64(key, span.begin);
            key = mix64(key, span.count);
        }
        const TokenSpan& first = item.token_spans.front();
        const TokenSpan& last  = item.token_spans.back();
        ranges.push_back(VisionTokenRange{
            .begin = static_cast<std::uint32_t>(first.begin),
            .end   = static_cast<std::uint32_t>(last.begin + last.count),
            .key   = key,
        });
    }
    std::sort(ranges.begin(), ranges.end(),
              [](const VisionTokenRange& left, const VisionTokenRange& right) {
                  return left.begin < right.begin;
              });
    return ranges;
}

std::uint64_t accumulate_vision(std::uint64_t cumulative, std::uint64_t key) noexcept {
    return mix64(cumulative == 0 ? kVisionExtraSeed : cumulative, key);
}

std::uint64_t prompt_extra_seed(const PreparedPromptData& prompt) noexcept {
    if (prompt.lora_key == std::array<std::uint64_t, 2>{}) { return 0; }
    const std::uint64_t seed = mix64(mix64(kLoraExtraSeed, prompt.lora_key[0]), prompt.lora_key[1]);
    return seed == 0 ? 1 : seed;
}

void prompt_block_keys(const PreparedPromptData& prompt, std::vector<std::uint64_t>& hashes,
                       std::vector<std::uint64_t>& extras) {
    const std::size_t blocks = prompt.token_ids.size() / pc::kBlockTokens;
    const std::uint64_t seed = prompt_extra_seed(prompt);
    extras.clear();
    if (!prompt.vision_items.empty() || seed != 0) {
        const std::vector<VisionTokenRange> ranges =
            prompt.vision_items.empty() ? std::vector<VisionTokenRange>{} : vision_ranges(prompt);
        extras.resize(blocks);
        std::uint64_t cumulative = seed;
        std::size_t next         = 0;
        for (std::size_t block = 0; block < blocks; ++block) {
            const std::uint64_t end = static_cast<std::uint64_t>(block + 1U) * pc::kBlockTokens;
            while (next < ranges.size() && ranges[next].begin < end) {
                cumulative = accumulate_vision(cumulative, ranges[next].key);
                ++next;
            }
            extras[block] = cumulative;
        }
    }
    hashes = pc::block_lookup_hashes(prompt.token_ids, extras);
}

} // namespace ninfer::models::qwen3_5::detail
