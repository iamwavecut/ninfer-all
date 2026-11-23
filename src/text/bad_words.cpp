#include "text/bad_words.h"

#include <algorithm>
#include <deque>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>

namespace ninfer::text {

struct BadWordsFilter::Tokens {
    std::vector<std::string> text;
    std::vector<TokenId> sorted; // the tokens with text, ascending by bytes, then by id
};

std::vector<std::string> normalize_bad_words(std::span<const std::string> words) {
    std::vector<std::string> out;
    for (const std::string& word : words) {
        const std::size_t start = word.find_first_not_of(" \t\n\r\v\f");
        if (start == std::string::npos) {
            throw std::invalid_argument("a bad_words entry is empty or whitespace only");
        }
        std::string text = word.substr(start);
        if (text.size() > kMaximumBadWordBytes) {
            throw std::invalid_argument("a bad_words entry is longer than " +
                                        std::to_string(kMaximumBadWordBytes) + " bytes");
        }
        if (std::find(out.begin(), out.end(), text) != out.end()) { continue; }
        if (out.size() == kMaximumBadWords) {
            throw std::invalid_argument("bad_words holds more than " +
                                        std::to_string(kMaximumBadWords) + " distinct entries");
        }
        out.push_back(std::move(text));
    }
    return out;
}

BadWordsFilter::BadWordsFilter(std::shared_ptr<const Tokens> tokens, std::vector<std::string> words)
    : tokens_(std::move(tokens)), words_(std::move(words)) {
    // The trie of the words, its children unordered while it grows.
    std::vector<std::vector<Edge>> children(1);
    std::vector<bool> terminal(1, false);
    std::vector<std::uint32_t> depths(1, 0);
    // Per word, the trie node after each of its first k bytes (index k - 1).
    std::vector<std::vector<State>> prefixes(words_.size());
    for (std::size_t w = 0; w < words_.size(); ++w) {
        State node = kStart;
        for (const char byte : words_[w]) {
            const auto b = static_cast<std::uint8_t>(byte);
            auto& edges  = children[node];
            const auto found =
                std::find_if(edges.begin(), edges.end(), [b](const Edge& e) { return e.byte == b; });
            if (found != edges.end()) {
                node = found->next;
            } else {
                const auto next = static_cast<State>(children.size());
                edges.push_back(Edge{.byte = b, .next = next});
                children.emplace_back();
                terminal.push_back(false);
                depths.push_back(depths[node] + 1);
                node = next;
            }
            prefixes[w].push_back(node);
        }
        terminal[node] = true;
    }
    nodes_.resize(children.size());
    for (std::size_t n = 0; n < children.size(); ++n) {
        auto& edges = children[n];
        std::sort(edges.begin(), edges.end(),
                  [](const Edge& a, const Edge& b) { return a.byte < b.byte; });
        nodes_[n].depth      = depths[n];
        nodes_[n].first_edge = static_cast<std::uint32_t>(edges_.size());
        nodes_[n].edges      = static_cast<std::uint32_t>(edges.size());
        edges_.insert(edges_.end(), edges.begin(), edges.end());
    }
    // Failure links in breadth-first order: a node's link is the longest proper suffix of its
    // string that the trie holds, found from its parent's link.
    std::deque<State> queue;
    for (std::uint32_t e = 0; e < nodes_[kStart].edges; ++e) {
        const State child   = edges_[nodes_[kStart].first_edge + e].next;
        nodes_[child].fail = kStart;
        nodes_[child].ends = terminal[child];
        queue.push_back(child);
    }
    while (!queue.empty()) {
        const State node = queue.front();
        queue.pop_front();
        for (std::uint32_t e = 0; e < nodes_[node].edges; ++e) {
            const Edge edge    = edges_[nodes_[node].first_edge + e];
            const State fail   = step(nodes_[node].fail, edge.byte);
            nodes_[edge.next].fail = fail;
            nodes_[edge.next].ends = terminal[edge.next] || nodes_[fail].ends;
            queue.push_back(edge.next);
        }
    }
    // What each proper prefix of a word leaves to complete it: the tokens that start with the
    // rest of the word.
    const auto& sorted = tokens_->sorted;
    const auto& text   = tokens_->text;
    std::vector<std::pair<State, Range>> completions;
    for (std::size_t w = 0; w < words_.size(); ++w) {
        const std::string_view word = words_[w];
        for (std::size_t k = 1; k < word.size(); ++k) {
            const std::string_view rest = word.substr(k);
            const auto begin = std::lower_bound(
                sorted.begin(), sorted.end(), rest,
                [&](TokenId token, std::string_view key) { return std::string_view(text[token]) < key; });
            const auto end = std::partition_point(begin, sorted.end(), [&](TokenId token) {
                return std::string_view(text[token]).starts_with(rest);
            });
            if (begin == end) { continue; }
            completions.emplace_back(
                prefixes[w][k - 1],
                Range{.begin = static_cast<std::uint32_t>(begin - sorted.begin()),
                      .end   = static_cast<std::uint32_t>(end - sorted.begin())});
        }
    }
    std::sort(completions.begin(), completions.end(), [](const auto& a, const auto& b) {
        return std::tie(a.first, a.second.begin, a.second.end) <
               std::tie(b.first, b.second.begin, b.second.end);
    });
    completions.erase(std::unique(completions.begin(), completions.end(),
                                  [](const auto& a, const auto& b) {
                                      return a.first == b.first && a.second.begin == b.second.begin &&
                                             a.second.end == b.second.end;
                                  }),
                      completions.end());
    for (const auto& [node, range] : completions) {
        if (nodes_[node].ranges == 0) {
            nodes_[node].first_range = static_cast<std::uint32_t>(ranges_.size());
        }
        ++nodes_[node].ranges;
        ranges_.push_back(range);
    }
    // Tokens that hold a word whole are banned after any text.
    for (TokenId token = 0; token < static_cast<TokenId>(text.size()); ++token) {
        if (!advance(kStart, token)) { always_.push_back(token); }
    }
}

BadWordsFilter::State BadWordsFilter::step(State state, std::uint8_t byte) const noexcept {
    while (true) {
        const Node& node = nodes_[state];
        const auto begin = edges_.begin() + node.first_edge;
        const auto end   = begin + node.edges;
        const auto found = std::lower_bound(
            begin, end, byte, [](const Edge& edge, std::uint8_t b) { return edge.byte < b; });
        if (found != end && found->byte == byte) { return found->next; }
        if (state == kStart) { return kStart; }
        state = node.fail;
    }
}

std::optional<BadWordsFilter::State> BadWordsFilter::advance(State state,
                                                              TokenId token) const noexcept {
    if (token < 0 || static_cast<std::size_t>(token) >= tokens_->text.size()) { return kStart; }
    const std::string& bytes = tokens_->text[static_cast<std::size_t>(token)];
    if (bytes.empty()) { return kStart; }
    for (const char byte : bytes) {
        state = step(state, static_cast<std::uint8_t>(byte));
        if (nodes_[state].ends) { return std::nullopt; }
    }
    return state;
}

void BadWordsFilter::clear_banned(State state, std::span<std::uint32_t> mask) const noexcept {
    const auto clear = [&](TokenId token) {
        mask[static_cast<std::size_t>(token) / 32] &= ~(1U << (static_cast<unsigned>(token) % 32));
    };
    for (const TokenId token : always_) { clear(token); }
    for (State node = state; node != kStart; node = nodes_[node].fail) {
        for (std::uint32_t r = 0; r < nodes_[node].ranges; ++r) {
            const Range range = ranges_[nodes_[node].first_range + r];
            for (std::uint32_t at = range.begin; at < range.end; ++at) {
                clear(tokens_->sorted[at]);
            }
        }
    }
}

std::size_t BadWordsFilter::token_bytes(TokenId token) const noexcept {
    if (token < 0 || static_cast<std::size_t>(token) >= tokens_->text.size()) { return 0; }
    return tokens_->text[static_cast<std::size_t>(token)].size();
}

std::size_t BadWordsFilter::vocab_size() const noexcept { return tokens_->text.size(); }

BadWordsVocabulary::BadWordsVocabulary(std::vector<std::string> token_text) {
    auto tokens  = std::make_shared<BadWordsFilter::Tokens>();
    tokens->text = std::move(token_text);
    for (TokenId token = 0; token < static_cast<TokenId>(tokens->text.size()); ++token) {
        if (!tokens->text[static_cast<std::size_t>(token)].empty()) {
            tokens->sorted.push_back(token);
        }
    }
    const auto& text = tokens->text;
    std::sort(tokens->sorted.begin(), tokens->sorted.end(), [&](TokenId a, TokenId b) {
        const int order = text[static_cast<std::size_t>(a)].compare(text[static_cast<std::size_t>(b)]);
        return order != 0 ? order < 0 : a < b;
    });
    tokens_ = std::move(tokens);
}

BadWordsVocabulary::~BadWordsVocabulary() = default;

std::shared_ptr<const BadWordsFilter>
BadWordsVocabulary::filter(const std::vector<std::string>& words) const {
    {
        std::lock_guard lock(mutex_);
        const auto found = std::find_if(cache_.begin(), cache_.end(),
                                        [&](const auto& entry) { return entry->words() == words; });
        if (found != cache_.end()) {
            std::rotate(cache_.begin(), found, found + 1);
            return cache_.front();
        }
    }
    auto built = std::make_shared<const BadWordsFilter>(tokens_, words);
    std::lock_guard lock(mutex_);
    cache_.insert(cache_.begin(), built);
    if (cache_.size() > kCachedFilters) { cache_.pop_back(); }
    return built;
}

} // namespace ninfer::text
