#pragma once

// Banned words (vLLM's `bad_words`): strings that the generated text never contains.
//
// The text of a generation is the bytes of its tokens in order. A token that the tokenizer marks
// special has no text: it ends the text before it, so a word cannot span it. A token is banned
// after some text when that text followed by the token's bytes contains a banned word that the
// text alone does not: the word lies inside the token, or it begins in the text and the token
// completes it. Every spelling of a word is banned this way, whatever tokens would write it,
// including the token sequences that vLLM bans (the word's own tokenization and the one after a
// space).
//
// A BadWordsFilter holds an Aho-Corasick automaton of the words. Its state after some text is the
// longest suffix of that text that begins a word; the suffixes on its failure chain are all the
// others. The tokens banned after a state are those whose bytes contain a word, which do not
// depend on the text, and for each such suffix p of a word w = p + r, the tokens whose bytes
// start with r: a range of the vocabulary sorted by bytes.

#include "ninfer/base.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::text {

inline constexpr std::size_t kMaximumBadWords     = 128;
inline constexpr std::size_t kMaximumBadWordBytes = 256;

// The words a request bans as a filter reads them: each without its leading ASCII whitespace (as
// vLLM strips it), duplicates removed, in first-seen order. Throws std::invalid_argument for an
// entry that is empty or whitespace only, longer than kMaximumBadWordBytes bytes, or for more than
// kMaximumBadWords distinct entries.
[[nodiscard]] std::vector<std::string> normalize_bad_words(std::span<const std::string> words);

class BadWordsFilter {
public:
    using State                   = std::uint32_t;
    static constexpr State kStart = 0; // before any text, and after a token without text

    // The state after `token`, or none when its bytes complete a banned word. A token without
    // text (special, or outside the vocabulary) returns kStart.
    [[nodiscard]] std::optional<State> advance(State state, TokenId token) const noexcept;
    // Clears the bit of every token banned after `state` in a mask of the vocabulary (bit t % 32 of
    // word t / 32), at least vocab_size() bits.
    void clear_banned(State state, std::span<std::uint32_t> mask) const noexcept;
    // The bytes of the text's longest suffix that begins a word: the text a word may still need.
    [[nodiscard]] std::uint32_t depth(State state) const noexcept { return nodes_[state].depth; }
    // The bytes of `token`'s text: zero for a token without text.
    [[nodiscard]] std::size_t token_bytes(TokenId token) const noexcept;
    [[nodiscard]] std::size_t vocab_size() const noexcept;
    [[nodiscard]] const std::vector<std::string>& words() const noexcept { return words_; }

    struct Tokens; // the vocabulary's token text and its order by bytes (BadWordsVocabulary)
    BadWordsFilter(std::shared_ptr<const Tokens> tokens, std::vector<std::string> words);

private:
    struct Node {
        std::uint32_t fail        = 0;
        std::uint32_t first_edge  = 0, edges = 0;  // edges_, ascending by byte
        std::uint32_t first_range = 0, ranges = 0; // ranges_: the tokens this suffix completes
        std::uint32_t depth       = 0;             // bytes of this node's string
        bool ends                 = false;         // a word ends here or on the failure chain
    };
    struct Edge {
        std::uint8_t byte  = 0;
        std::uint32_t next = 0;
    };
    struct Range {
        std::uint32_t begin = 0, end = 0; // positions in Tokens::sorted
    };
    [[nodiscard]] State step(State state, std::uint8_t byte) const noexcept;

    std::shared_ptr<const Tokens> tokens_;
    std::vector<std::string> words_;
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
    std::vector<Range> ranges_;
    std::vector<TokenId> always_; // tokens whose bytes contain a word
};

// A vocabulary's tokens as text, and the filters of the word lists its requests ban: the last
// few lists are kept, since a client tends to send the same list with every request.
class BadWordsVocabulary {
public:
    // One entry per token id: its bytes, or empty for a token without text.
    explicit BadWordsVocabulary(std::vector<std::string> token_text);
    ~BadWordsVocabulary();
    BadWordsVocabulary(const BadWordsVocabulary&)            = delete;
    BadWordsVocabulary& operator=(const BadWordsVocabulary&) = delete;

    // The filter of words that normalize_bad_words() returned.
    [[nodiscard]] std::shared_ptr<const BadWordsFilter>
    filter(const std::vector<std::string>& words) const;

private:
    static constexpr std::size_t kCachedFilters = 8;
    std::shared_ptr<const BadWordsFilter::Tokens> tokens_;
    mutable std::mutex mutex_;
    mutable std::vector<std::shared_ptr<const BadWordsFilter>> cache_; // most recent first
};

} // namespace ninfer::text
