#pragma once

// Inkling 2's output markup in the shared frontend. A turn is a list of blocks, each a header
// `<|message_model|>` (followed by the function's name before a call), a content token, the
// block's text and `<|end_message|>`: `<|content_thinking|>` opens reasoning, `<|content_text|>`
// (or `<|content_tool_error|>`, `<|content_invoke_tool_text|>`) content, and
// `<|content_invoke_tool_json|>` a call `{"name":NAME,"args":{...}}`. The model writes every
// header itself (the chat template has no generation prompt) and decides whether it reasons; the
// turn ends with `<|content_model_end_sampling|>`. All of these are special tokens.

#include "ninfer/request.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

class Tokenizer;
struct ToolCallOutputContract;

inline constexpr std::string_view kInklingHeader    = "<|message_model|>";
inline constexpr std::string_view kInklingThinking  = "<|content_thinking|>";
inline constexpr std::string_view kInklingText      = "<|content_text|>";
inline constexpr std::string_view kInklingToolError = "<|content_tool_error|>";
inline constexpr std::string_view kInklingToolText  = "<|content_invoke_tool_text|>";
inline constexpr std::string_view kInklingCallOpen  = "<|content_invoke_tool_json|>";
inline constexpr std::string_view kInklingEnd       = "<|end_message|>";

// The block markers, single tokens of Inkling's vocabulary; -1 where the vocabulary lacks one.
struct InklingMarkers {
    std::int32_t header     = -1;
    std::int32_t thinking   = -1;
    std::int32_t text       = -1;
    std::int32_t tool_error = -1;
    std::int32_t tool_text  = -1;
    std::int32_t call       = -1;
    std::int32_t end        = -1;

    [[nodiscard]] static InklingMarkers from(const Tokenizer& tokenizer);
    [[nodiscard]] bool complete() const noexcept;
    // A token that opens a block's body: thinking, text, an error, a text call or a JSON call.
    [[nodiscard]] bool content(std::int32_t token) const noexcept;
};

// What the block being read is: no block or a header still naming one, reasoning, content, or a
// JSON call.
enum class InklingBlock : std::uint8_t { None, Header, Thinking, Text, Call };

// Pulls Inkling's JSON calls out of the content stream, in which the output session writes a call
// block as `<|content_invoke_tool_json|>JSON<|end_message|>`: text outside calls is published as it
// arrives (a possible opener's prefix is held back), each complete call becomes a structured call
// with the JSON's name and arguments. An unterminated or unreadable call returns to content at the
// end.
//
// A forced call is one the prompt already opened: it ends with
// `<|message_model|>NAME<|content_invoke_tool_json|>`, so generation resumes in the JSON and the
// decoder starts in that call. The model never emits the opener, so a failed forced call returns
// only the model's bytes.
class InklingToolCallDecoder {
public:
    struct Terminal {
        std::string content;
        std::vector<GeneratedToolCall> tool_calls;
        ToolCallParseDiagnostics diagnostics;
    };

    explicit InklingToolCallDecoder(std::shared_ptr<const ToolCallOutputContract> contract);

    [[nodiscard]] std::string feed(std::string_view text);
    [[nodiscard]] Terminal finish();

private:
    std::shared_ptr<const ToolCallOutputContract> contract_;
    std::string pending_;
    std::string region_;
    bool in_call_ = false;
    // The call being read is the forced one, whose opener came from the prompt.
    bool forced_ = false;
    std::vector<GeneratedToolCall> calls_;
    ToolCallParseDiagnostics diagnostics_;
};

// A call's JSON `{"name":NAME,"args":{...}}` as a structured call; false when unreadable. `forced`
// names the function the prompt opened, which the JSON has to name too.
[[nodiscard]] bool parse_inkling_call(std::string_view region, std::string_view forced,
                                      GeneratedToolCall& out);

// Grammars of Inkling's calls, XGrammar structural-tag formats. Every JSON call block
// (`<|content_invoke_tool_json|>` to `<|end_message|>`) holds `{"name":NAME,"args":{...}}` for a
// declared tool, its arguments under the tool's parameters' JSON Schema when the tool is strict;
// the text around call blocks (headers, the function names after them, other blocks) is free. A
// forced call starts the format inside its JSON, after the opener the prompt ends with. A strict
// tool's schema that strict arguments cannot honor throws std::invalid_argument.
//
// A response without a structured final format: the turn's blocks, calls as above.
[[nodiscard]] std::string inkling_tool_call_text_format(const ToolCallOutputContract& contract);
// A structured final response beside tools, as the envelope's reasoning format: the blocks before
// the text block that holds the response, calls as above.
[[nodiscard]] std::string inkling_blocks_before_text_format(const ToolCallOutputContract& contract);
// A structured final response beside tools, as the envelope's response alternative: a turn of
// blocks without a text block and with at least one call.
[[nodiscard]] std::string inkling_calls_turn_format(const ToolCallOutputContract& contract);

} // namespace ninfer::models::qwen3_5::frontend
