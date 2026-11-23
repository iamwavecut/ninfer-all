// Gemma 4's tool-call markup in the shared frontend: GemmaToolCallDecoder pulls
// `<|tool_call>call:NAME{key:value,...}<tool_call|>` regions out of streamed content (markers split
// across feeds included) and returns them as structured calls with JSON arguments; text around
// them stays content, and an unreadable or unterminated call returns to content. A forced call is
// already open in the prompt (`<|tool_call>call:NAME{`): the decoder starts inside it, a turn that
// ends on the closed arguments completes it, and a failed one returns only the model's own bytes.
//
// Thinking control: a thinking budget counts the tokens of the thought channel the model opens,
// stops a round at the budget's last token, closes the channel with the guidance and <channel|>,
// and bans <|channel> for the rest of the response.

#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/gemma_output.h"
#include "models/qwen3_5/frontend/output_session.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/registry.h"
#include "text/structured_output.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

using ninfer::ToolCallParseFallbackReason;
using ninfer::models::qwen3_5::frontend::GemmaToolCallDecoder;
using ninfer::models::qwen3_5::frontend::parse_gemma_call;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

// `streamed` collects what feed() published during the stream; the terminal content is added to it.
GemmaToolCallDecoder::Terminal run(const std::vector<std::string>& pieces, std::string& streamed,
                                   std::string_view forced_tool_name = {}) {
    GemmaToolCallDecoder decoder(forced_tool_name);
    for (const auto& piece : pieces) { streamed += decoder.feed(piece); }
    auto terminal = decoder.finish();
    streamed += terminal.content;
    return terminal;
}

// Feeds the text one byte at a time: every marker is split at every possible point.
GemmaToolCallDecoder::Terminal run_bytewise(const std::string& text, std::string& streamed,
                                            std::string_view forced_tool_name = {}) {
    std::vector<std::string> pieces;
    for (const char byte : text) { pieces.emplace_back(1, byte); }
    return run(pieces, streamed, forced_tool_name);
}

// Gemma's channel, call and turn markers as special tokens after the 256 bytes, and a template
// that renders the turns between them.
constexpr std::array<std::string_view, 8> kMarkers = {
    "<bos>", "<eos>", "<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", "<|\"|>", "<turn|>"};
constexpr int kEosId = 257, kChannelOpenId = 258, kChannelCloseId = 259;
constexpr std::string_view kTemplate =
    "{{ bos_token }}{% for m in messages %}<|turn>{{ m.role }}\n{{ m.content }}<turn|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|turn>model\n{% endif %}";

ninfer::models::qwen3_5::Frontend gemma_frontend() {
    namespace fi        = ninfer::models::qwen3_5::frontend;
    using Json          = nlohmann::json;
    static const auto files = [] {
        // Byte-fallback pieces <0x00>..<0xFF> as ids 0..255, as Gemma's vocabulary carries them.
        Json vocab = Json::object();
        for (int byte = 0; byte < 256; ++byte) {
            char name[8];
            std::snprintf(name, sizeof(name), "<0x%02X>", byte);
            vocab[name] = byte;
        }
        Json added   = Json::array();
        Json decoder = Json::object();
        int id       = 256;
        for (const std::string_view marker : kMarkers) {
            const Json entry = {{"content", std::string(marker)}, {"single_word", false},
                                {"lstrip", false},                {"rstrip", false},
                                {"normalized", false},            {"special", true}};
            Json token  = entry;
            token["id"] = id;
            added.push_back(token);
            decoder[std::to_string(id)] = entry;
            ++id;
        }
        const Json tokenizer =
            Json{{"normalizer",
                  {{"type", "Replace"}, {"pattern", {{"String", " "}}}, {"content", "\xe2\x96\x81"}}},
                 {"pre_tokenizer", nullptr},
                 {"model",
                  {{"type", "BPE"},
                   {"byte_fallback", true},
                   {"vocab", vocab},
                   {"merges", Json::array()}}},
                 {"added_tokens", added}};
        const Json config = {{"added_tokens_decoder", decoder},
                             {"bos_token", "<bos>"},
                             {"eos_token", "<eos>"}};
        return std::array<std::string, 3>{tokenizer.dump(), config.dump(),
                                          Json{{"eos_token_id", kEosId}}.dump()};
    }();
    static const std::string chat(kTemplate);
    ninfer::models::qwen3_5::FrontendResources resources{.tokenizer_json         = files[0],
                                                         .tokenizer_config_json  = files[1],
                                                         .chat_template_jinja    = chat,
                                                         .generation_config_json = files[2]};
    resources.tokenizer = std::make_shared<const fi::Tokenizer>(
        fi::TokenizerResources{files[0], files[1], files[2]});
    resources.public_token_count = static_cast<std::uint32_t>(resources.tokenizer->vocab_size());
    ninfer::models::qwen3_5::FrontendOptions options;
    options.architecture   = ninfer::models::Architecture::Gemma4;
    options.vision_enabled = false;
    options.max_context    = std::numeric_limits<std::uint32_t>::max();
    return ninfer::models::qwen3_5::make_frontend(resources, options);
}

std::vector<ninfer::TokenId> tokens_of(std::initializer_list<ninfer::TokenId> markers,
                                       std::string_view text = {}) {
    std::vector<ninfer::TokenId> out(markers);
    for (const char byte : text) { out.push_back(static_cast<unsigned char>(byte)); }
    return out;
}

void test_thinking_control() {
    const auto frontend = gemma_frontend();
    ninfer::PromptInput input;
    input.messages.push_back(
        {.role  = ninfer::ChatRole::User,
         .parts = {{.kind = ninfer::MessagePartKind::Text, .text = "Hi"}}});
    input.options.continuation = ninfer::PromptContinuationMode::NewAssistantTurn;
    const auto prompt = frontend.prepare(std::move(input));
    expect(frontend.thinking_control_token_count() > 1, "Gemma has an early-close control span");
    auto session = frontend.make_output_session(
        prompt, {}, ninfer::OutputOptions{.tool_name_max_length = 64},
        ninfer::ThinkingControlOptions{.budget = 3});
    std::string reasoning, content;
    const auto take = [&](const auto& deltas) {
        for (const ninfer::OutputDelta& delta : deltas) {
            (delta.channel == ninfer::OutputChannel::Reasoning ? reasoning : content) += delta.text;
        }
    };
    // The channel's name and newline are its header, not reasoning.
    const auto open     = tokens_of({kChannelOpenId}, "thought\nabcdef");
    const auto decision = session.preview_model(open, 256, ninfer::FinishReason::OutputLimit);
    take(session.commit_preview());
    // <|channel>, the 8 header bytes and 3 thinking tokens.
    expect(decision.accepted_tokens == 12 && !decision.finished() &&
               decision.continuation == ninfer::runtime::ContinuationAction::ApplyTargetControl,
           "the budget stops the round at its third thinking token, accepted " +
               std::to_string(decision.accepted_tokens));
    const auto pending = session.pending_control_tokens();
    expect(!pending.empty() && pending.back() == kChannelCloseId,
           "the control span ends with <channel|>");
    if (pending.empty()) { return; }
    const std::vector<ninfer::TokenId> control(pending.begin(), pending.end());
    (void)session.preview_control(control, 245);
    take(session.commit_preview());
    const auto stats = session.thinking_stats();
    expect(stats.applied && stats.model_thinking_tokens == 3 &&
               session.banned_token() == std::optional<ninfer::TokenId>(kChannelOpenId),
           "the control closes the channel, counts only the model's tokens and bans <|channel>");
    const auto answer = tokens_of({}, "OK");
    (void)session.preview_model(answer, 200, ninfer::FinishReason::OutputLimit);
    take(session.commit_preview());
    const ninfer::TokenId eos = kEosId;
    (void)session.preview_model(std::span(&eos, 1), 198, ninfer::FinishReason::OutputLimit);
    take(session.commit_preview());
    expect(reasoning.starts_with("abc") &&
               reasoning.find("Considering the limited time") != std::string::npos &&
               content == "OK",
           "reasoning '" + reasoning + "', content '" + content + "'");
}

// Banned words beside thinking: the reasoning and content publish as without them, the session's
// constraint holds no grammar, its masks ban the token that would complete a word (inside the
// reasoning too, but not across the channel's markers), and a banned token fails the preview.
void test_bad_words() {
    const auto frontend = gemma_frontend();
    ninfer::PromptInput input;
    input.messages.push_back(
        {.role  = ninfer::ChatRole::User,
         .parts = {{.kind = ninfer::MessagePartKind::Text, .text = "Hi"}}});
    input.options.continuation = ninfer::PromptContinuationMode::NewAssistantTurn;
    const auto prompt = frontend.prepare(std::move(input));
    const std::vector<std::string> words = {"  <accent", "<name"};
    const auto run = [&](std::span<const std::string> banned, std::string& reasoning,
                         std::string& content) {
        auto session = frontend.make_output_session(prompt, {}, ninfer::OutputOptions{}, {}, {},
                                                    banned);
        const auto take = [&](const auto& deltas) {
            for (const ninfer::OutputDelta& delta : deltas) {
                (delta.channel == ninfer::OutputChannel::Reasoning ? reasoning : content) +=
                    delta.text;
            }
        };
        auto tokens = tokens_of({kChannelOpenId}, "thought\n<acc");
        tokens.push_back(kChannelCloseId);
        for (const char byte : std::string("ent <acce")) { tokens.push_back(static_cast<unsigned char>(byte)); }
        (void)session.preview_model(tokens, 256, ninfer::FinishReason::OutputLimit);
        take(session.commit_preview());
        return session;
    };
    std::string plain_reasoning, plain_content, reasoning, content;
    (void)run({}, plain_reasoning, plain_content);
    auto session = run(words, reasoning, content);
    expect(reasoning == plain_reasoning && content == plain_content,
           "banned words leave the channels as they are: '" + reasoning + "' / '" + content + "'");
    const auto grammar = session.grammar_state();
    expect(grammar != nullptr && !grammar->has_grammar(), "banned words alone hold no grammar");
    if (!grammar) { return; }
    // 256 byte tokens and the 8 markers.
    constexpr std::size_t words_per_mask = (264 + 31) / 32;
    std::vector<std::uint32_t> masks(words_per_mask * 2);
    const ninfer::TokenId n = 'n';
    grammar->fill_masks(masks, std::span(&n, 1));
    const auto allowed = [&](std::size_t column, ninfer::TokenId token) {
        return ((masks[column * words_per_mask + token / 32] >> (token % 32)) & 1U) != 0;
    };
    expect(allowed(0, 'n') && allowed(0, 't') && allowed(1, 'e') && !allowed(1, 't'),
           "the masks ban the byte that completes <accent after its draft");
    expect(allowed(1, kEosId), "the end of the turn stays allowed");
    const std::vector<ninfer::TokenId> completing = {'n', 't'};
    bool refused = false;
    try {
        (void)session.preview_model(completing, 128, ninfer::FinishReason::OutputLimit);
    } catch (const std::logic_error&) { refused = true; }
    expect(refused, "a token that completes a banned word fails the preview");
}

// A core that rewinds (allow_bad_words_rewind), simulated over byte tokens: a model that writes a
// script one byte at a time, in rounds of several tokens as a speculative round verifies them,
// and that writes another byte where the masks ban the script's. Each round's token i is drawn
// under mask column i, conditioned on the round's tokens before it. The published text never
// holds a banned word nor the start of one that a rewind took back; it ends as the generated
// text; every rewind returns to the hold the core was told to keep.
struct RewindRun {
    std::string published;
    std::vector<ninfer::TokenId> generated;
    int rewinds  = 0;
    bool clean   = true; // no published prefix ever held a banned word
    bool holds   = true; // every rewind went to a kept hold
    bool counted = true; // the grammar counted exactly the generated tokens
};

RewindRun run_rewinds(const std::vector<std::string>& words, const std::string& script,
                      const std::vector<std::size_t>& round_sizes, char fallback) {
    const auto frontend = gemma_frontend();
    ninfer::PromptInput input;
    input.messages.push_back(
        {.role  = ninfer::ChatRole::User,
         .parts = {{.kind = ninfer::MessagePartKind::Text, .text = "Hi"}}});
    input.options.continuation = ninfer::PromptContinuationMode::NewAssistantTurn;
    const auto prompt = frontend.prepare(std::move(input));
    auto session = frontend.make_output_session(prompt, {}, ninfer::OutputOptions{}, {}, {}, words);
    const auto grammar = session.grammar_state();
    RewindRun run;
    constexpr std::size_t words_per_mask = (264 + 31) / 32;
    // The script position the model reached before each generated token.
    std::vector<std::size_t> cursor_at{0};
    std::optional<std::size_t> kept;
    for (std::size_t round = 0; round < 400; ++round) {
        session.allow_bad_words_rewind(true);
        const std::size_t size = round_sizes[round % round_sizes.size()];
        std::vector<ninfer::TokenId> tokens;
        std::size_t cursor = cursor_at[run.generated.size()];
        for (std::size_t i = 0; i < size; ++i) {
            std::vector<std::uint32_t> masks(words_per_mask * (tokens.size() + 1));
            grammar->fill_masks(masks, tokens);
            const auto allowed = [&](ninfer::TokenId token) {
                const std::uint32_t word = masks[tokens.size() * words_per_mask + token / 32];
                return ((word >> (token % 32)) & 1U) != 0;
            };
            ninfer::TokenId token = kEosId;
            if (cursor < script.size()) {
                token = static_cast<unsigned char>(script[cursor]);
                if (!allowed(token)) { token = static_cast<unsigned char>(fallback); }
                ++cursor;
            }
            if (!allowed(token)) { token = kEosId; }
            tokens.push_back(token);
            if (token == kEosId) { break; }
        }
        const std::size_t first = run.generated.size();
        const auto decision =
            session.preview_model(tokens, 4096, ninfer::FinishReason::OutputLimit);
        for (std::size_t i = 0; i < decision.accepted_tokens; ++i) {
            run.generated.push_back(tokens[i]);
            std::size_t next = cursor_at[first + i];
            if (tokens[i] != kEosId) { ++next; }
            cursor_at.resize(first + i + 1);
            cursor_at.push_back(next);
        }
        if (decision.bad_words_hold_after) { kept = first + *decision.bad_words_hold_after; }
        if (decision.bad_words_rewind) {
            ++run.rewinds;
            run.holds = run.holds && kept == *decision.bad_words_rewind;
            run.generated.resize(*decision.bad_words_rewind);
            cursor_at.resize(run.generated.size() + 1);
        }
        for (const ninfer::OutputDelta& delta : session.commit_preview()) {
            if (delta.channel == ninfer::OutputChannel::Content) { run.published += delta.text; }
        }
        for (const std::string& word : words) {
            if (run.published.find(word) != std::string::npos) { run.clean = false; }
        }
        if (!decision.bad_words_hold_alive) { kept.reset(); }
        run.counted = run.counted && grammar->accepted_tokens() == run.generated.size();
        if (decision.finished()) { break; }
    }
    return run;
}

std::string text_of(const std::vector<ninfer::TokenId>& tokens) {
    std::string text;
    for (const ninfer::TokenId token : tokens) {
        if (token < 256) { text += static_cast<char>(token); }
    }
    return text;
}

void test_bad_words_rewind() {
    const std::vector<std::string> words = {"<daily_persona", "<accent", "na_a"};
    for (const std::vector<std::size_t>& sizes :
         {std::vector<std::size_t>{1}, std::vector<std::size_t>{3, 1, 2},
          std::vector<std::size_t>{5}}) {
        const std::string label = "rounds of " + std::to_string(sizes.front()) + ": ";
        // The model copies a prompt tag: the rewind takes back its '<' too, and the model writes
        // its fallback byte there instead; the rest of the tag then holds "na_a", whose 'n' a
        // second rewind takes back.
        const RewindRun copy =
            run_rewinds(words, "Ok <daily_persona_accent> and <acc! done", sizes, '.');
        const std::string text = text_of(copy.generated);
        expect(copy.clean && copy.holds && copy.counted, label + "rewinds keep the output clean");
        expect(copy.published == text, label + "the published text is the generated text: '" +
                                            copy.published + "' / '" + text + "'");
        expect(copy.rewinds == 2 && text == "Ok .daily_perso.a_accent> and <acc! done",
               label + "rewinds take back the words' first bytes: '" + text + "'");
        // A word that begins inside the text another hold covers ("na_a" inside "persona_a").
        const RewindRun inner = run_rewinds(words, "x persona_accent y", sizes, '-');
        expect(inner.clean && inner.holds && inner.counted &&
                   inner.published == text_of(inner.generated),
               label + "a word inside a word's text: '" + inner.published + "'");
        // A word that begins and ends in one round: the rewind goes to its first token there.
        const RewindRun one = run_rewinds(words, "ab na_a cd", {8}, '-');
        expect(one.clean && one.holds && one.rewinds == 1 &&
                   text_of(one.generated) == "ab -a_a cd" && one.published == "ab -a_a cd",
               label + "a word within one round: '" + one.published + "'");
        // A suffix that dies ("t" of "beat") before the word's own begins: the hold moves to
        // where the word began, and the rewind goes there.
        for (const auto& round_sizes : {sizes, std::vector<std::size_t>{6}}) {
            const RewindRun moved = run_rewinds({"the w"}, "a beat the wave", round_sizes, '-');
            expect(moved.clean && moved.holds && moved.counted && moved.rewinds == 1 &&
                       text_of(moved.generated) == "a beat -he wave" &&
                       moved.published == "a beat -he wave",
                   label + "a suffix that died before the word: '" + moved.published + "'");
        }
        // A model that writes the tag whatever comes before it: after the request's rewinds the
        // masks keep the completing byte out, and no banned word is ever published.
        std::string stubborn;
        for (int i = 0; i < 40; ++i) { stubborn += "<accent"; }
        const RewindRun insist = run_rewinds(words, stubborn, sizes, '<');
        expect(insist.clean && insist.holds && insist.counted &&
                   insist.published == text_of(insist.generated) && insist.rewinds <= 32,
               label + "a stubborn model: " + std::to_string(insist.rewinds) + " rewinds, '" +
                   insist.published.substr(0, 60) + "'");
    }
}

} // namespace

int main() {
    const std::string q = "<|\"|>";
    {
        ninfer::GeneratedToolCall call;
        expect(parse_gemma_call("call:get_weather{city:" + q + "Paris" + q + ",days:3}", call),
               "parse a call");
        expect(call.name == "get_weather", "call name");
        const auto args = nlohmann::json::parse(call.arguments_json);
        expect(args == nlohmann::json({{"city", "Paris"}, {"days", 3}}), "call arguments");
    }
    {
        // Nested object and array values, a boolean, a string holding braces and commas.
        ninfer::GeneratedToolCall call;
        expect(parse_gemma_call("call:f{a:{b:[1,2," + q + "x,}" + q + "]},flag:true}", call),
               "parse nested values");
        const auto args = nlohmann::json::parse(call.arguments_json);
        expect(args == nlohmann::json({{"a", {{"b", {1, 2, "x,}"}}}}, {"flag", true}}),
               "nested arguments");
    }
    {
        ninfer::GeneratedToolCall call;
        expect(!parse_gemma_call("get_weather{city:Paris}", call), "reject a region without call:");
    }
    {
        // Streamed with the markers split across feeds, text before the call kept as content.
        std::string streamed;
        const auto terminal = run({"Checking. <|tool", "_call>call:get_weather{city:" + q, "Paris",
                                   q + ",days:3}<tool_", "call|>"},
                                  streamed);
        expect(terminal.tool_calls.size() == 1, "one streamed call");
        expect(streamed == "Checking. ", "content around the call: '" + streamed + "'");
        if (!terminal.tool_calls.empty()) {
            expect(terminal.tool_calls[0].name == "get_weather", "streamed call name");
        }
    }
    {
        // Two calls in one turn.
        std::string streamed;
        const auto terminal =
            run({"<|tool_call>call:a{x:1}<tool_call|><|tool_call>call:b{y:" + q + "z" + q +
                 "}<tool_call|>"},
                streamed);
        expect(terminal.tool_calls.size() == 2, "two calls");
        expect(streamed.empty(), "no content beside two calls");
    }
    {
        // An unterminated call returns to content.
        std::string streamed;
        const auto terminal = run({"<|tool_call>call:a{x:1"}, streamed);
        expect(terminal.tool_calls.empty(), "unterminated call is not a call");
        expect(streamed.find("call:a{x:1") != std::string::npos, "unterminated call stays content");
    }
    {
        // An unterminated call that does not parse reports why it went back to content.
        std::string streamed;
        const auto terminal = run({"<|tool_call>call:a{x:1"}, streamed);
        expect(terminal.diagnostics.fallback_reason == ToolCallParseFallbackReason::MalformedStructure,
               "unterminated unreadable call reports malformed_structure");
        expect(!terminal.diagnostics.forced_call_closed, "a free call is never a closed forced call");
    }
    {
        // A free call that ends on its closing brace is read as before, and is not a forced one.
        std::string streamed;
        const auto terminal = run({"<|tool_call>call:a{x:1}"}, streamed);
        expect(terminal.tool_calls.size() == 1 && streamed.empty(), "free call without marker");
        expect(!terminal.diagnostics.forced_call_closed, "free call is not a closed forced call");
    }

    // Forced call: the prompt supplied `<|tool_call>call:NAME{`, the model continues with arguments.
    {
        const std::string text = "caption:" + q + "a cat, <b>}" + q + ",count:3,ratio:-0.5,ok:true,"
                                 "off:false,none:null,tags:[" + q + "x" + q + ",2],"
                                 "opts:{n:{k:" + q + "v" + q + "}}}<tool_call|>";
        const auto expected    = nlohmann::json::parse(
            R"({"caption":"a cat, <b>}","count":3,"ratio":-0.5,"ok":true,"off":false,)"
            R"("none":null,"tags":["x",2],"opts":{"n":{"k":"v"}}})");
        std::string whole;
        const auto one = run({text}, whole, "draw_image");
        expect(one.tool_calls.size() == 1 && one.tool_calls[0].name == "draw_image",
               "forced call is one call with the forced name");
        if (!one.tool_calls.empty()) {
            expect(nlohmann::json::parse(one.tool_calls[0].arguments_json) == expected,
                   "forced call arguments: " + one.tool_calls[0].arguments_json);
        }
        expect(whole.empty(), "a forced call leaves no content: '" + whole + "'");
        expect(one.diagnostics.marker_seen && one.diagnostics.structured_call_count == 1 &&
                   !one.diagnostics.forced_call_closed &&
                   one.diagnostics.fallback_reason == ToolCallParseFallbackReason::None,
               "forced call diagnostics");

        // The same text split at every byte, including inside <|"|> and <tool_call|>.
        std::string bytewise;
        const auto split = run_bytewise(text, bytewise, "draw_image");
        expect(split.tool_calls.size() == 1 && bytewise.empty(), "byte-split forced call");
        if (!split.tool_calls.empty()) {
            expect(nlohmann::json::parse(split.tool_calls[0].arguments_json) == expected,
                   "byte-split forced call arguments");
        }
        expect(!split.diagnostics.forced_call_closed, "a terminated forced call was not closed");

        // Fragments that end mid-marker at several points.
        for (const std::size_t cut : {text.size() - 1, text.size() - 6, text.size() - 11}) {
            std::string pieces_streamed;
            const auto pieces = run({text.substr(0, cut), text.substr(cut)}, pieces_streamed,
                                    "draw_image");
            expect(pieces.tool_calls.size() == 1 && pieces_streamed.empty() &&
                       !pieces.diagnostics.forced_call_closed,
                   "forced call split at " + std::to_string(cut));
        }
    }
    {
        // The turn ends on the closing brace: the call is complete without its marker.
        std::string streamed;
        const auto terminal = run({"caption:" + q + "x" + q + "}"}, streamed, "draw_image");
        expect(terminal.tool_calls.size() == 1 && terminal.tool_calls[0].name == "draw_image",
               "unterminated forced call completes");
        expect(terminal.diagnostics.forced_call_closed, "forced_call_closed on a missing marker");
        expect(terminal.diagnostics.fallback_reason == ToolCallParseFallbackReason::None &&
                   streamed.empty(),
               "a completed forced call leaves no fallback and no content");
        std::string bytewise;
        const auto split = run_bytewise("caption:" + q + "x" + q + "}", bytewise, "draw_image");
        expect(split.tool_calls.size() == 1 && split.diagnostics.forced_call_closed &&
                   bytewise.empty(),
               "byte-split unterminated forced call");
    }
    {
        // A function with no arguments.
        std::string streamed;
        const auto terminal = run({"}"}, streamed, "ping");
        expect(terminal.tool_calls.size() == 1 && terminal.tool_calls[0].arguments_json == "{}" &&
                   terminal.diagnostics.forced_call_closed,
               "forced call without arguments");
    }
    {
        // Less than a complete call is never completed: the model's own bytes return as content,
        // without the prompt's `<|tool_call>call:NAME{`.
        const std::vector<std::string> unfinished = {
            "caption:" + q + "x" + q + ",cou", "caption:" + q + "x", "caption:" + q + "x" + q + ",",
            "a:{b:1}", std::string{}, "caption:" + q + "x" + q + ",opts:{a:1}"};
        for (const std::string& text : unfinished) {
            std::string streamed;
            const auto terminal = run({text}, streamed, "draw_image");
            expect(terminal.tool_calls.empty(), "unfinished forced call is not a call: '" + text + "'");
            expect(streamed == text, "unfinished forced call returns the model's bytes: '" + text +
                                         "' -> '" + streamed + "'");
            expect(!terminal.diagnostics.forced_call_closed &&
                       terminal.diagnostics.fallback_reason ==
                           ToolCallParseFallbackReason::MalformedStructure,
                   "unfinished forced call reports malformed_structure: '" + text + "'");
        }
        // A terminated region that does not read keeps the model's closing marker.
        std::string streamed;
        const auto terminal = run({"caption " + q + "x" + q + "}<tool_call|>"}, streamed, "draw_image");
        expect(terminal.tool_calls.empty() &&
                   streamed == "caption " + q + "x" + q + "}<tool_call|>" &&
                   terminal.diagnostics.fallback_reason ==
                       ToolCallParseFallbackReason::MalformedStructure,
               "unreadable terminated forced call: '" + streamed + "'");
    }
    {
        // The forced call comes first; later text and calls are the model's own. A later call that
        // lacks its marker is a free call, not a closed forced one.
        std::string streamed;
        const auto terminal = run({"id:1}<tool_call|>Done. <|tool_call>call:b{y:2}"}, streamed, "a");
        expect(terminal.tool_calls.size() == 2 && terminal.tool_calls[0].name == "a" &&
                   terminal.tool_calls[1].name == "b",
               "forced call followed by a free call");
        expect(streamed == "Done. ", "text between calls stays content: '" + streamed + "'");
        expect(!terminal.diagnostics.forced_call_closed, "only the forced call can be closed");

        // A failed forced region does not leak its seeded head into the next region's content.
        std::string failed;
        const auto second = run({"junk<tool_call|><|tool_call>call:b{y:2}<tool_call|>"}, failed, "a");
        expect(second.tool_calls.size() == 1 && second.tool_calls[0].name == "b" &&
                   failed == "junk<tool_call|>",
               "failed forced region then a free call: '" + failed + "'");
    }
    try {
        test_thinking_control();
        test_bad_words();
        test_bad_words_rewind();
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        ++failures;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " Gemma output\n";
    return failures == 0 ? 0 : 1;
}
