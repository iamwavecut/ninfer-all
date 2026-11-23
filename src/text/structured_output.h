#pragma once
#include "ninfer/base.h"
#include "ninfer/types_fwd.h"
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::text {
// A token the request's grammar does not admit. The masks the request sampled under should have
// excluded it, so it is an execution fault, but one confined to that request.
class GrammarViolation final : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

// Reject unsupported constraints instead of letting a compiler silently weaken a schema.
void validate_structured_output(const StructuredOutputOptions& options);

// Model-owned framing around the final response. The formats are XGrammar structural-tag
// descriptors (e.g. a model's native tool-call serialization), not a user API.
struct StructuredOutputEnvelope {
    std::string reasoning_close;
    // What comes before reasoning_close, in place of any text without it.
    std::string reasoning_format;
    // In place of the final response, after reasoning_close.
    std::string alternative_format;
    // In place of the whole response: reasoning_close and the final response.
    std::string response_alternative_format;
};

class BadWordsFilter;

// A request's token constraint: a grammar (structured output, strict tool arguments), banned words
// (bad_words), or both. Its masks admit the tokens that the grammar admits and, while completions
// are banned (the default), that complete no banned word; accept() throws GrammarViolation for a
// token the grammar does not admit or that completes a banned word. With completions left to the
// caller (OutputSession, which takes the output back to where such a word began), the masks keep
// out only the tokens banned at the position the caller returned to.
class GrammarState {
public:
    ~GrammarState();
    GrammarState(GrammarState&&) noexcept;
    GrammarState& operator=(GrammarState&&) noexcept;
    // `grammar` (a state nobody else holds yet, or null for none) banning `filter`'s words too.
    // A null filter returns `grammar` unchanged.
    [[nodiscard]] static std::shared_ptr<GrammarState>
    with_bad_words(std::shared_ptr<GrammarState> grammar,
                   std::shared_ptr<const BadWordsFilter> filter);
    [[nodiscard]] std::unique_ptr<GrammarState> fork() const;
    // False for a state that only bans words: the output is otherwise free.
    [[nodiscard]] bool has_grammar() const noexcept;
    void accept(std::span<const TokenId> tokens);
    // The tokens accepted so far: the generated position of the next column.
    [[nodiscard]] std::uint64_t accepted_tokens() const noexcept;
    // Whether the masks keep out every token that would complete a banned word.
    void ban_completions(bool banned) noexcept;
    // Keeps `tokens` out of the column at generated position `position`, and with `completions`
    // every token that would complete a banned word there too. Replaces the bans of any other
    // position; the tokens add to those banned at the same position before.
    void ban_at(std::uint64_t position, std::span<const TokenId> tokens, bool completions);
    // Column i is conditioned on drafts[0..i). Unreachable suffix columns are unrestricted.
    void fill_masks(std::span<std::uint32_t> masks, std::span<const TokenId> drafts) const;
private:
    struct Impl;
    explicit GrammarState(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class StructuredCompiler;
};

class StructuredCompiler {
public:
    StructuredCompiler(std::vector<std::string> decoded_vocab, std::vector<int> stop_tokens);
    ~StructuredCompiler();
    std::shared_ptr<GrammarState> compile(const StructuredOutputOptions& options,
                                          const StructuredOutputEnvelope& envelope = {});
    // A response with no final format of its own: `format`, an XGrammar structural-tag format
    // (free text with a model's tool calls), after the reasoning close when the envelope has one.
    std::shared_ptr<GrammarState> compile_format(const std::string& format,
                                                 const StructuredOutputEnvelope& envelope = {});
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ninfer::text
