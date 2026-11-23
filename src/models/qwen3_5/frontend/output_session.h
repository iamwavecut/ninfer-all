#pragma once
#include "ninfer/base.h"
#include "ninfer/observability/execution.h"
#include "ninfer/request.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/request.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::text {
class BadWordsFilter;
} // namespace ninfer::text

namespace ninfer::models::qwen3_5 {
namespace frontend {
class Tokenizer;
struct ToolCallOutputContract;
struct GemmaMarkers;
struct FlintMarkers;
struct DeepSeekMarkers;
struct InklingMarkers;
} // namespace frontend
class Frontend;

// The channel deltas a commit publishes, in order: one per channel transition.
class PublishedOutput {
public:
    using iterator       = std::vector<OutputDelta>::iterator;
    using const_iterator = std::vector<OutputDelta>::const_iterator;

    PublishedOutput()                                  = default;
    PublishedOutput(const PublishedOutput&)            = default;
    PublishedOutput& operator=(const PublishedOutput&) = default;
    PublishedOutput(PublishedOutput&& other) noexcept;
    PublishedOutput& operator=(PublishedOutput&& other) noexcept;

    [[nodiscard]] bool empty() const noexcept { return values_.empty(); }

    [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }

    [[nodiscard]] iterator begin() noexcept { return values_.begin(); }

    [[nodiscard]] const_iterator begin() const noexcept { return values_.begin(); }

    [[nodiscard]] iterator end() noexcept { return values_.end(); }

    [[nodiscard]] const_iterator end() const noexcept { return values_.end(); }

    [[nodiscard]] OutputDelta& back() noexcept { return values_.back(); }

    [[nodiscard]] const OutputDelta& back() const noexcept { return values_.back(); }

    void clear() noexcept;
    void push_back(OutputDelta value);
    // Appends `other`'s deltas, joining a delta to the last one when their channels match.
    void append(PublishedOutput other);

private:
    std::vector<OutputDelta> values_;
};

class OutputSession {
public:
    OutputSession() noexcept;
    ~OutputSession();
    OutputSession(OutputSession&&) noexcept;
    OutputSession& operator=(OutputSession&&) noexcept;

    OutputSession(const OutputSession&)            = delete;
    OutputSession& operator=(const OutputSession&) = delete;

    // `logprobs` is empty, or aligned with `tokens` when the request asked for logprobs.
    [[nodiscard]] runtime::OutputDecision
    preview_model(std::span<const TokenId> tokens, std::uint32_t total_budget_remaining,
                  FinishReason limit_reason, std::span<const runtime::RawTokenLogprob> logprobs = {});
    [[nodiscard]] std::uint32_t
    model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept;
    [[nodiscard]] std::span<const TokenId> pending_control_tokens() const noexcept;
    // Tokens of the thinking-control suffix which may still be forced through the output budget.
    // Zero once this session can no longer apply one.
    [[nodiscard]] std::uint32_t control_suffix_tokens() const noexcept;
    [[nodiscard]] runtime::OutputDecision preview_control(std::span<const TokenId> tokens,
                                                          std::uint32_t total_budget_remaining);
    [[nodiscard]] runtime::OutputDecision preview_terminal(FinishReason reason);
    [[nodiscard]] PublishedOutput commit_preview();
    // The logprob records of the content tokens whose first byte the last commit published, in
    // generation order. A content token's bytes go to the content channel: after the reasoning
    // block and its closing whitespace, including the markup of a tool call.
    [[nodiscard]] std::vector<TokenLogprob> take_content_logprobs() noexcept;
    [[nodiscard]] std::shared_ptr<text::GrammarState> grammar_state() const;
    [[nodiscard]] std::vector<GeneratedToolCall> take_tool_calls() noexcept;
    [[nodiscard]] ToolCallParseDiagnostics tool_call_parse_diagnostics() const noexcept;
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept;
    [[nodiscard]] ThinkingBudgetStats thinking_stats() const noexcept;
    // True once a session that began inside a reasoning block has committed its close.
    [[nodiscard]] bool reasoning_closed() const noexcept;
    // The reasoning opener a model that opens its own reasoning (Inkling, Flint, Gemma) cannot
    // write again after an early close.
    [[nodiscard]] std::optional<TokenId> banned_token() const noexcept;
    [[nodiscard]] std::optional<std::string> matched_stop_string() const;
    // Banned words (bad_words): whether the request's core can take the output back to where a
    // banned word began (OutputDecision::bad_words_rewind) in the rounds from here on. Until a
    // core allows it, the masks keep out every token that would complete a word, and nothing is
    // held back. Call it between rounds, before the next round's masks are filled.
    void allow_bad_words_rewind(bool allowed);

private:
    class Impl;
    struct BadWords;
    // The round without banned words: what preview_model and commit_preview do for a request
    // that bans none.
    [[nodiscard]] runtime::OutputDecision
    preview_tokens(std::span<const TokenId> tokens, std::uint32_t total_budget_remaining,
                   FinishReason limit_reason, std::span<const runtime::RawTokenLogprob> logprobs);
    [[nodiscard]] PublishedOutput commit_tokens();
    // A preview of no tokens, which commit_preview commits without a change.
    void preview_nothing();
    OutputSession(std::shared_ptr<const frontend::Tokenizer> tokenizer, StopPolicy policy,
                  OutputOptions output, bool starts_in_reasoning, ThinkingControlOptions thinking,
                  std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens,
                  std::shared_ptr<const frontend::ToolCallOutputContract> tool_call_output,
                  std::shared_ptr<text::GrammarState> grammar               = {},
                  std::shared_ptr<const frontend::GemmaMarkers> gemma       = {},
                  std::shared_ptr<const frontend::FlintMarkers> flint       = {},
                  std::shared_ptr<const frontend::DeepSeekMarkers> deepseek = {},
                  std::shared_ptr<const frontend::InklingMarkers> inkling   = {},
                  std::shared_ptr<const text::BadWordsFilter> bad_words     = {});
    std::unique_ptr<Impl> impl_;
    std::unique_ptr<BadWords> bad_words_;

    friend class Frontend;
};

} // namespace ninfer::models::qwen3_5
