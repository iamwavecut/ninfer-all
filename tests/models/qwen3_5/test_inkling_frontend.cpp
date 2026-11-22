// Inkling 2 in the shared frontend.
//
// Pre-tokenizer: the word ends of o200k_base's one split against Hugging Face tokenizers' own
// pre-tokenization of the same texts (tests/fixtures/inkling2/pretokenizer_cases.json, generated
// from the release's tokenizer.json). A byte vocabulary without merges makes a token of every
// byte, so the stable frontier of a boundary at every byte is the last word end at or before it.
// A long run that continues a kept one encodes as a fresh tokenizer encodes it.
//
// Whole words: under model.ignore_merges a word the vocabulary holds whole is that token, as
// Hugging Face tokenizers encodes it, and other words merge as usual.
//
// Chat template: the release's template (tests/fixtures/inkling2/chat_template.jinja, Apache-2.0,
// thinkingmachines/Inkling-2-Mini) renders every conversation of
// tests/fixtures/inkling2/chat_cases.json as transformers renders it (the fixture's `expected`),
// at each reasoning effort, with calls, tool results and an image; disabled thinking is effort
// level 0.
//
// Prompt and output: a Frontend over a byte vocabulary with Inkling's marker tokens tokenizes the
// prompt with those tokens, routes thinking blocks to reasoning and text blocks to content, drops
// the headers the model writes, and returns JSON calls as structured calls, one round or a token
// per round; a forced call opens its block after the prompt, and a read of the answer's first token
// its text block; structured output admits the headers
// and thinking before the JSON text block, and no call block without tools.
//
// Call grammars: a strict tool's call blocks hold a declared name and arguments under its schema
// with free blocks around them, a forced strict call starts in its JSON, and a structured response
// beside tools is call blocks before the JSON text block or a turn of calls without one.
//
// Tool calls: InklingToolCallDecoder pulls calls out of streamed content (markers split at every
// byte), returns an unreadable or unterminated call to content, and starts a forced call in its
// JSON.
//
// Images: the processor plans an image's size and grid and writes its patches as llama.cpp's
// inkling2v preprocessor does (tests/fixtures/inkling2/image_cases.json, written by
// tools/reference/inkling2_vision.py's plan and Pillow: plans within and past the budget, the
// patches of a synthetic picture padded with -1 and resized past its budget, and those of
// examples/cli/media/natural_scene.png at the release's budget and at --vision-max-merged 100),
// and a prompt's image becomes one <|unused_200054|> per 2 x 2 cell after <|content_image|>.

#include "models/qwen3_5/frontend/chat_template.h"
#include "models/qwen3_5/frontend/digest.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/inkling_output.h"
#include "models/qwen3_5/frontend/output_session.h"
#include "models/qwen3_5/frontend/processor.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/frontend/test_access.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/frontend/tool_call_parser.h"
#include "models/registry.h"
#include "text/structured_output.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifndef NINFER_SOURCE_DIR
#    error "NINFER_SOURCE_DIR is required"
#endif

namespace {

namespace fi      = ninfer::models::qwen3_5::frontend;
using Json        = nlohmann::json;
using OrderedJson = nlohmann::ordered_json;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

std::string read_fixture(const std::string& name) {
    std::ifstream file(std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/inkling2/" + name,
                       std::ios::binary);
    if (!file) { throw std::runtime_error("missing fixture " + name); }
    std::ostringstream out;
    out << file.rdbuf();
    return out.str();
}

// The release's split, as its tokenizer.json writes it.
constexpr std::string_view kSplit =
    R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

// The marker tokens after the 256 bytes, all special as in the release.
constexpr std::string_view kMarkers[] = {"<|message_user|>",
                                         "<|message_model|>",
                                         "<|message_system|>",
                                         "<|message_tool|>",
                                         "<|content_text|>",
                                         "<|content_image|>",
                                         "<|content_model_end_sampling|>",
                                         "<|content_thinking|>",
                                         "<|end_message|>",
                                         "<|content_tool_error|>",
                                         "<|content_xml|>",
                                         "<|content_invoke_tool_json|>",
                                         "<|content_invoke_tool_text|>",
                                         "<|unused_200054|>",
                                         "<|endoftext|>"};
constexpr int kUserId = 256, kModelId = 257, kSystemId = 258, kTextId = 260, kContentImageId = 261,
              kEosId = 262, kThinkingId = 263, kEndId = 264, kCallId = 267, kImageId = 269;
constexpr int kTokens = 256 + static_cast<int>(std::size(kMarkers));

// GPT-2's byte-to-symbol table, which byte-level BPE vocabularies are written in.
std::string byte_level_symbol(std::uint8_t byte) {
    const auto visible = [](int b) {
        return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    };
    int code = byte;
    if (!visible(byte)) {
        int next = 256;
        for (int b = 0; b < byte; ++b) {
            if (!visible(b)) { ++next; }
        }
        code = next;
    }
    std::string out;
    if (code < 0x80) {
        out.push_back(static_cast<char>(code));
    } else {
        out.push_back(static_cast<char>(0xC0 | (code >> 6)));
        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
    return out;
}

struct TokenizerFiles {
    std::string tokenizer_json, tokenizer_config_json, generation_config_json;
};

// A byte vocabulary behind the release's pipeline (no normalizer, its split, ByteLevel, whole
// words), with `extra` entries and `merges`, and with or without its marker tokens.
TokenizerFiles byte_tokenizer_files(bool markers, const Json& extra = Json::object(),
                                    const Json& merges = Json::array()) {
    Json vocab = Json::object();
    for (int byte = 0; byte < 256; ++byte) {
        vocab[byte_level_symbol(static_cast<std::uint8_t>(byte))] = byte;
    }
    vocab.update(extra);
    Json added   = Json::array();
    Json decoder = Json::object();
    if (markers) {
        int id = 256;
        for (const std::string_view marker : kMarkers) {
            const Json entry = {{"content", std::string(marker)},
                                {"single_word", false},
                                {"lstrip", false},
                                {"rstrip", false},
                                {"normalized", false},
                                {"special", true}};
            Json token       = entry;
            token["id"]      = id;
            added.push_back(token);
            decoder[std::to_string(id)] = entry;
            ++id;
        }
    }
    const Json split = {{"type", "Sequence"},
                        {"pretokenizers",
                         {{{"type", "Split"},
                           {"pattern", {{"Regex", std::string(kSplit)}}},
                           {"behavior", "Isolated"},
                           {"invert", false}},
                          {{"type", "ByteLevel"},
                           {"add_prefix_space", false},
                           {"trim_offsets", true},
                           {"use_regex", false}}}}};
    TokenizerFiles files;
    files.tokenizer_json = Json{{"normalizer", nullptr},
                                {"pre_tokenizer", split},
                                {"decoder",
                                 {{"type", "ByteLevel"},
                                  {"add_prefix_space", true},
                                  {"trim_offsets", true},
                                  {"use_regex", true}}},
                                {"post_processor",
                                 {{"type", "ByteLevel"},
                                  {"add_prefix_space", true},
                                  {"trim_offsets", false},
                                  {"use_regex", true}}},
                                {"model",
                                 {{"type", "BPE"},
                                  {"dropout", nullptr},
                                  {"ignore_merges", true},
                                  {"vocab", vocab},
                                  {"merges", merges}}},
                                {"added_tokens", added}}
                               .dump();
    files.tokenizer_config_json =
        Json{{"added_tokens_decoder", decoder},
             {"eos_token", "<|content_model_end_sampling|>"},
             {"pad_token", "<|endoftext|>"}}
            .dump();
    files.generation_config_json = markers ? Json{{"eos_token_id", kEosId}}.dump()
                                           : std::string(R"({"eos_token_id":0})");
    return files;
}

fi::Tokenizer load(const TokenizerFiles& files) {
    return fi::Tokenizer({.tokenizer_json         = files.tokenizer_json,
                          .tokenizer_config_json  = files.tokenizer_config_json,
                          .generation_config_json = files.generation_config_json});
}

// The word ends of `text`: the distinct stable frontiers of a boundary at every byte.
std::vector<std::size_t> word_ends(const fi::Tokenizer& tokenizer, const std::string& text) {
    std::vector<std::size_t> boundaries(text.size() + 1);
    for (std::size_t i = 0; i <= text.size(); ++i) { boundaries[i] = i; }
    const fi::BoundaryEncodedText encoded = tokenizer.encode_with_boundaries(text, boundaries);
    std::set<std::size_t> ends;
    for (const auto& boundary : encoded.boundaries) {
        if (boundary.stable_frontier != 0) { ends.insert(boundary.stable_frontier); }
    }
    return {ends.begin(), ends.end()};
}

void test_pretokenizer() {
    const fi::Tokenizer tokenizer = load(byte_tokenizer_files(false));
    const Json cases              = Json::parse(read_fixture("pretokenizer_cases.json"));
    for (const Json& item : cases) {
        const std::string text = item.at("text").get<std::string>();
        const auto expected    = item.at("word_ends").get<std::vector<std::size_t>>();
        const auto actual      = word_ends(tokenizer, text);
        if (actual != expected) {
            std::ostringstream got;
            for (const std::size_t end : actual) { got << end << ' '; }
            expect(false, "word ends of " + Json(text).dump() + ": " + got.str());
        }
    }
    // No normalizer: a decomposed accent stays decomposed.
    const std::string decomposed = "e\xcc\x81";
    expect(tokenizer.encode(decomposed).size() == decomposed.size(), "no NFC before Inkling's BPE");
}

void test_run_reuse() {
    // A kept run ends its words where a lower-case letter later on would join a capital run to the
    // word before it, and where a later line break would join a whitespace run into one word; a run
    // that shares it up to there must not reuse those word ends.
    std::string filler;
    while (filler.size() < 6000) { filler += "word, more text; "; }
    for (const auto& [first, second] : std::vector<std::pair<std::string, std::string>>{
             {"\xe0\xb9\x84\xe0\xb8\x97\xe0\xb8\xa2" "ABCDEFGHIJ tail, end.", // Thai, capitals
              "\xe0\xb9\x84\xe0\xb8\x97\xe0\xb8\xa2" "ABCDEFGHIJk tail, end."},
             {"\n" + std::string(10, ' ') + "x tail", "\n" + std::string(10, ' ') + "\n tail"}}) {
        const fi::Tokenizer warm = load(byte_tokenizer_files(false));
        (void)word_ends(warm, filler + first);
        const fi::Tokenizer cold = load(byte_tokenizer_files(false));
        expect(word_ends(warm, filler + second) == word_ends(cold, filler + second),
               "a kept run's undecided word end was reused: " + Json(second).dump());
    }
    // Long runs that share a kept run's text encode as a fresh tokenizer encodes them, whatever
    // follows the shared text: capitals, other letters (CJK and Thai), digits, punctuation, line
    // breaks and contractions.
    const std::vector<std::string> pieces = {"word", "Mixed", "ABC", "x", " ", "  ", "\n",
                                             "\t",   "'s",    "'LL", "1", "2345", ".", ", ",
                                             "//",   "\xe4\xb8\xad\xe6\x96\x87",
                                             "\xe0\xb9\x84\xe0\xb8\x97", "\xc3\xa9",
                                             "e\xcc\x81", "!?", "\xc5\xbf"};
    std::mt19937 rng(20261121);
    std::uniform_int_distribution<std::size_t> pick(0, pieces.size() - 1);
    const auto text_of = [&](std::size_t bytes) {
        std::string text;
        while (text.size() < bytes) { text += pieces[pick(rng)]; }
        return text;
    };
    const fi::Tokenizer warm = load(byte_tokenizer_files(false));
    std::string previous     = text_of(6000);
    (void)word_ends(warm, previous);
    for (int round = 0; round < 40; ++round) {
        std::size_t cut = std::uniform_int_distribution<std::size_t>(4096, previous.size())(rng);
        while (cut < previous.size() && (static_cast<unsigned char>(previous[cut]) & 0xC0) == 0x80) {
            ++cut;
        }
        const std::string next = previous.substr(0, cut) + text_of(64);
        const fi::Tokenizer cold = load(byte_tokenizer_files(false));
        expect(word_ends(warm, next) == word_ends(cold, next) &&
                   warm.encode(next) == cold.encode(next),
               "a run continuing a kept one, round " + std::to_string(round));
        previous = next;
    }
}

void test_whole_words() {
    // "bc" merges; "abc" is in the vocabulary without a merge that builds it.
    const TokenizerFiles files =
        byte_tokenizer_files(false, Json{{"bc", 256}, {"abc", 257}}, Json::array({{"b", "c"}}));
    const fi::Tokenizer tokenizer = load(files);
    expect(tokenizer.encode("abc") == std::vector<int>{257}, "a whole word is its token");
    expect(tokenizer.encode("abcd") == std::vector<int>{'a', 256, 'd'},
           "a word the vocabulary lacks merges");
    expect(tokenizer.encode("x abc") == std::vector<int>{'x', ' ', 'a', 256},
           "a word with its space prefix is another word");
    const auto encoded = tokenizer.encode_with_boundaries("abc!", std::vector<std::size_t>{2, 3});
    expect(encoded.input_ids == std::vector<int>{257, '!'} &&
               !encoded.boundaries[0].exact_frontier &&
               encoded.boundaries[1].exact_frontier == std::optional<std::size_t>(1),
           "a whole word's token ends where the word ends");
}

std::vector<fi::ChatMessage> messages_of(const OrderedJson& item) {
    std::vector<fi::ChatMessage> messages;
    for (const OrderedJson& source : item.at("messages")) {
        fi::ChatMessage message;
        const std::string role    = source.at("role");
        message.role              = role == "system"      ? ninfer::ChatRole::System
                                    : role == "user"      ? ninfer::ChatRole::User
                                    : role == "assistant" ? ninfer::ChatRole::Assistant
                                                          : ninfer::ChatRole::Tool;
        const OrderedJson content = source.value("content", OrderedJson(""));
        if (content.is_array()) {
            for (const OrderedJson& block : content) {
                message.parts.push_back(block.at("type") == "image"
                                            ? fi::ChatPart::image(fi::MediaData{})
                                            : fi::ChatPart::text_part(block.at("text")));
            }
        } else if (!content.get<std::string>().empty()) {
            message.parts.push_back(fi::ChatPart::text_part(content.get<std::string>()));
        }
        message.reasoning_content = source.value("reasoning_content", "");
        message.tool_call_id      = source.value("tool_call_id", "");
        for (const OrderedJson& call : source.value("tool_calls", OrderedJson::array())) {
            message.tool_calls.push_back({.id             = call.at("id"),
                                          .name           = call.at("name"),
                                          .arguments_json = call.at("arguments").dump()});
        }
        messages.push_back(std::move(message));
    }
    return messages;
}

std::optional<ninfer::ReasoningEffort> effort(const OrderedJson& item) {
    if (!item.contains("reasoning_effort")) { return std::nullopt; }
    const std::string name = item["reasoning_effort"];
    if (name == "minimal") { return ninfer::ReasoningEffort::Minimal; }
    if (name == "low") { return ninfer::ReasoningEffort::Low; }
    if (name == "medium") { return ninfer::ReasoningEffort::Medium; }
    if (name == "high") { return ninfer::ReasoningEffort::High; }
    if (name == "xhigh") { return ninfer::ReasoningEffort::XHigh; }
    if (name == "max") { return ninfer::ReasoningEffort::Max; }
    throw std::runtime_error("fixture effort " + name);
}

fi::CompiledChatTemplate compiled_template() {
    return fi::CompiledChatTemplate::resolve(read_fixture("chat_template.jinja"), "inkling2",
                                             Json{{"eos_token", "<|content_model_end_sampling|>"},
                                                  {"pad_token", "<|endoftext|>"}});
}

void test_chat_template() {
    const auto compiled     = compiled_template();
    const OrderedJson cases = OrderedJson::parse(read_fixture("chat_cases.json"));
    for (const OrderedJson& item : cases) {
        fi::ChatRenderOptions options;
        options.tool_call_syntax = fi::ToolCallSyntax::Inkling; // the frontend's choice
        options.reasoning_effort = effort(item);
        if (item.contains("thinking")) { options.enable_thinking = item["thinking"].get<bool>(); }
        for (const OrderedJson& tool : item.value("tools", OrderedJson::array())) {
            options.tool_jsons.push_back(tool.dump());
        }
        const auto messages             = messages_of(item);
        const fi::RenderedChat rendered = compiled.render(messages, options);
        expect(rendered.text == item.at("expected").get<std::string>(),
               "chat case '" + item.at("name").get<std::string>() + "':\n" + rendered.text);
        expect(!rendered.starts_in_reasoning, "an Inkling prompt does not open reasoning");
    }
    // A forced call opens its block after the prompt: the header, the function's name and the
    // JSON call's opener.
    const OrderedJson& calls = cases.at(3);
    fi::ChatRenderOptions options;
    options.tool_call_syntax = fi::ToolCallSyntax::Inkling;
    options.enable_thinking  = false;
    options.forced_tool_name = "search";
    for (const OrderedJson& tool : calls.at("tools")) { options.tool_jsons.push_back(tool.dump()); }
    const fi::RenderedChat forced = compiled.render(messages_of(calls), options);
    expect(forced.text.find("Thinking effort level: 0<|end_message|>") != std::string::npos &&
               forced.text.ends_with("<|message_user|><|content_text|>Thanks<|end_message|>"
                                     "<|message_model|>search<|content_invoke_tool_json|>"),
           "forced call opener: " + forced.text);
}

// The converter's processor_config.json of the release's image processing.
const std::string kProcessorConfig =
    R"({"image_processor":{"patch_size":20,"merge_size":2,"max_image_tokens":4096,)"
    R"("pad_value":-1.0,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],)"
    R"("image_token":"<|unused_200054|>","image_token_id":269}})";

ninfer::models::qwen3_5::Frontend inkling_frontend(bool vision = false,
                                                   std::uint32_t max_merged = 16384) {
    // The resources are views the Frontend may keep.
    static const TokenizerFiles files = byte_tokenizer_files(true);
    static const std::string chat     = read_fixture("chat_template.jinja");
    ninfer::models::qwen3_5::FrontendResources resources{
        .tokenizer_json         = files.tokenizer_json,
        .tokenizer_config_json  = files.tokenizer_config_json,
        .chat_template_jinja    = chat,
        .generation_config_json = files.generation_config_json};
    if (vision) { resources.processor_config_json = kProcessorConfig; }
    resources.tokenizer          = std::make_shared<const fi::Tokenizer>(fi::TokenizerResources{
        files.tokenizer_json, files.tokenizer_config_json, files.generation_config_json});
    resources.public_token_count = static_cast<std::uint32_t>(resources.tokenizer->vocab_size());
    ninfer::models::qwen3_5::FrontendOptions options;
    options.architecture             = ninfer::models::Architecture::Inkling2;
    options.vision_enabled           = vision;
    options.vision_max_merged_tokens = max_merged;
    options.max_context              = std::numeric_limits<std::uint32_t>::max();
    return ninfer::models::qwen3_5::make_frontend(resources, options);
}

std::vector<ninfer::TokenId> bytes(std::string_view text) {
    std::vector<ninfer::TokenId> out;
    for (const char byte : text) { out.push_back(static_cast<unsigned char>(byte)); }
    return out;
}

std::vector<ninfer::TokenId> joined(std::initializer_list<std::vector<ninfer::TokenId>> parts) {
    std::vector<ninfer::TokenId> out;
    for (const auto& part : parts) { out.insert(out.end(), part.begin(), part.end()); }
    return out;
}

struct Turn {
    std::string reasoning, content;
    std::vector<ninfer::GeneratedToolCall> calls;
};

// Generates `tokens` and EOS, one round or one token per round.
Turn generate(const ninfer::models::qwen3_5::Frontend& frontend,
              const ninfer::models::qwen3_5::PreparedPrompt& prompt,
              std::vector<ninfer::TokenId> tokens, bool streamed) {
    auto session =
        frontend.make_output_session(prompt, {}, ninfer::OutputOptions{.tool_name_max_length = 64});
    tokens.push_back(kEosId);
    Turn turn;
    const auto commit = [&] {
        for (const ninfer::OutputDelta& delta : session.commit_preview()) {
            (delta.channel == ninfer::OutputChannel::Reasoning ? turn.reasoning : turn.content) +=
                delta.text;
        }
    };
    const auto budget = static_cast<std::uint32_t>(tokens.size() + 8);
    if (streamed) {
        for (const ninfer::TokenId token : tokens) {
            (void)session.preview_model(std::span(&token, 1), budget,
                                        ninfer::FinishReason::OutputLimit);
            commit();
        }
    } else {
        (void)session.preview_model(tokens, budget, ninfer::FinishReason::OutputLimit);
        commit();
    }
    turn.calls = session.take_tool_calls();
    return turn;
}

ninfer::PromptInput user_prompt(std::string text) {
    ninfer::PromptInput input;
    input.messages.push_back(
        {.role  = ninfer::ChatRole::User,
         .parts = {{.kind = ninfer::MessagePartKind::Text, .text = std::move(text)}}});
    input.options.continuation = ninfer::PromptContinuationMode::NewAssistantTurn;
    return input;
}

const std::string kTool =
    R"({"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}})";

void test_call_grammars(const ninfer::models::qwen3_5::Frontend& frontend);

void test_prompt_and_output() {
    const auto frontend = inkling_frontend();
    const auto prompt   = frontend.prepare(user_prompt("Hi"));
    {
        // The markers are their tokens, the rest bytes; the prompt ends after the user's block.
        const auto ids = prompt.token_ids();
        expect(std::vector<ninfer::TokenId>(ids.begin(), ids.end()) ==
                   joined({{kSystemId, kTextId},
                           bytes("Thinking effort level: 0.9"),
                           {kEndId, kUserId, kTextId},
                           bytes("Hi"),
                           {kEndId}}),
               "prompt tokens");
        expect(!prompt.summary().starts_in_reasoning, "an Inkling prompt does not reason");
    }
    {
        // A read of the answer's first token: the prompt opens a text block for it, and the token
        // read is content.
        auto input                    = user_prompt("Hi");
        input.options.enable_thinking = false;
        input.options.open_answer     = true;
        const auto read               = frontend.prepare(std::move(input));
        const auto ids                = read.token_ids();
        expect(ids.size() > 2 && ids[ids.size() - 2] == kModelId && ids.back() == kTextId,
               "an opened answer ends with a text block's header and opener");
        const Turn turn = generate(frontend, read, bytes("A"), false);
        expect(turn.content == "A" && turn.reasoning.empty(), "an opened answer's token is content");
    }
    for (const bool streamed : {false, true}) {
        // A thinking block, then a text block: reasoning, then content; the headers are neither.
        Turn turn = generate(frontend, prompt,
                             joined({{kModelId, kThinkingId},
                                     bytes("Let me think.\nDone."),
                                     {kEndId, kModelId, kTextId},
                                     bytes("Hello!"),
                                     {kEndId}}),
                             streamed);
        expect(turn.reasoning == "Let me think.\nDone." && turn.content == "Hello!" &&
                   turn.calls.empty(),
               "reasoning then content: [" + turn.reasoning + "] [" + turn.content + "]");
        // Without thinking, and with a thinking block the model left open: the next header
        // closes it.
        turn = generate(frontend, prompt, joined({{kModelId, kTextId}, bytes("Hi!"), {kEndId}}),
                        streamed);
        expect(turn.reasoning.empty() && turn.content == "Hi!", "content only: " + turn.content);
        turn = generate(frontend, prompt,
                        joined({{kModelId, kThinkingId}, bytes("Hmm"), {kModelId, kTextId},
                                bytes("Yes"), {kEndId}}),
                        streamed);
        expect(turn.reasoning == "Hmm" && turn.content == "Yes",
               "an open thinking block closed by a header: [" + turn.reasoning + "] [" +
                   turn.content + "]");
    }
    const std::string call = R"({"name":"get_weather","args":{"city":"Paris","days":3}})";
    {
        // Calls after the reasoning and a text block: the text, the calls structured.
        auto input = user_prompt("Weather in Paris?");
        input.options.tool_jsons.push_back(kTool);
        const auto tools = frontend.prepare(std::move(input));
        for (const bool streamed : {false, true}) {
            const Turn turn = generate(frontend, tools,
                                       joined({{kModelId, kThinkingId},
                                               bytes("Use the tool."),
                                               {kEndId, kModelId, kTextId},
                                               bytes("Checking."),
                                               {kEndId, kModelId},
                                               bytes("get_weather"),
                                               {kCallId},
                                               bytes(call),
                                               {kEndId, kModelId},
                                               bytes("get_weather"),
                                               {kCallId},
                                               bytes(R"({"name":"get_weather","args":{}})"),
                                               {kEndId}}),
                                       streamed);
            expect(turn.reasoning == "Use the tool." && turn.content == "Checking." &&
                       turn.calls.size() == 2 && turn.calls[0].name == "get_weather" &&
                       Json::parse(turn.calls[0].arguments_json) ==
                           Json({{"city", "Paris"}, {"days", 3}}) &&
                       turn.calls[1].arguments_json == "{}",
                   "tool call output: [" + turn.reasoning + "] [" + turn.content + "]");
        }
    }
    {
        // A forced call: the prompt ends with the call's opener, the model writes its JSON.
        auto input                     = user_prompt("Weather in Paris?");
        input.options.enable_thinking  = false;
        input.options.forced_tool_name = "get_weather";
        input.options.tool_jsons.push_back(kTool);
        const auto forced = frontend.prepare(std::move(input));
        const auto ids    = forced.token_ids();
        const auto opener = joined({{kModelId}, bytes("get_weather"), {kCallId}});
        expect(ids.size() > opener.size() &&
                   std::vector<ninfer::TokenId>(
                       ids.end() - static_cast<std::ptrdiff_t>(opener.size()), ids.end()) ==
                       opener,
               "forced call prompt ends with the opener");
        Turn turn = generate(frontend, forced, joined({bytes(call), {kEndId}}), false);
        expect(turn.content.empty() && turn.calls.size() == 1 &&
                   Json::parse(turn.calls[0].arguments_json) ==
                       Json({{"city", "Paris"}, {"days", 3}}),
               "forced call output: [" + turn.content + "]");
        turn = generate(frontend, forced, bytes(R"({"name":"other","args":{}})"), false);
        expect(turn.calls.empty() && turn.content == R"({"name":"other","args":{}})",
               "a forced call naming another function returns to content: " + turn.content);
    }
    {
        // Structured output: the headers and any thinking block, the text block's opener, then
        // the JSON.
        const auto session = frontend.make_output_session(
            prompt, {}, {}, {},
            ninfer::StructuredOutputOptions{.kind = ninfer::StructuredOutputKind::JsonObject});
        const auto grammar = session.grammar_state();
        const auto admits  = [&](const std::vector<ninfer::TokenId>& prefix, ninfer::TokenId next) {
            auto state = grammar->fork();
            try {
                state->accept(prefix);
            } catch (const std::exception&) { return false; }
            std::vector<std::uint32_t> mask((kTokens + 31) / 32);
            state->fill_masks(mask, {});
            return (mask[next / 32] & (1U << (next % 32))) != 0;
        };
        expect(grammar != nullptr && admits({}, kModelId) &&
                   admits({kModelId, kThinkingId, 'a', kEndId, kModelId}, kTextId) &&
                   admits({kModelId, kTextId}, '{') && !admits({kModelId, kTextId}, 'x') &&
                   !admits({kModelId, kTextId, '{', '}'}, 'x'),
               "structured output admits headers and thinking, then the JSON");
        expect(!admits({kModelId}, kCallId), "structured output without tools admits no call");
    }
    test_call_grammars(frontend);
}

// Whether `grammar` admits `next` after `prefix`.
bool admits(const std::shared_ptr<ninfer::text::GrammarState>& grammar,
            const std::vector<ninfer::TokenId>& prefix, ninfer::TokenId next) {
    auto state = grammar->fork();
    try {
        state->accept(prefix);
    } catch (const std::exception&) { return false; }
    std::vector<std::uint32_t> mask((kTokens + 31) / 32);
    state->fill_masks(mask, {});
    return (mask[next / 32] & (1U << (next % 32))) != 0;
}

const std::string kStrictTool =
    R"({"type":"function","function":{"name":"get_weather","strict":true,"parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"],"additionalProperties":false}}})";

void test_call_grammars(const ninfer::models::qwen3_5::Frontend& frontend) {
    const auto prepared = [&](const std::string& tool, std::string_view forced = {}) {
        auto input = user_prompt("Weather in Oslo?");
        input.options.tool_jsons.push_back(tool);
        if (!forced.empty()) {
            input.options.enable_thinking  = false;
            input.options.forced_tool_name = std::string(forced);
        }
        return frontend.prepare(std::move(input));
    };
    const auto opened = joined({{kModelId}, bytes("get_weather"), {kCallId}});
    const auto after  = [&](std::string_view json) { return joined({opened, bytes(json)}); };
    const std::string call = R"({"name":"get_weather","args":{"city":"Oslo"}})";
    {
        // A strict tool: the text around call blocks is free, a call block holds the declared
        // function's name and its arguments under the parameters' schema.
        const auto grammar = frontend.make_output_session(prepared(kStrictTool), {}, {}, {}, {})
                                 .grammar_state();
        expect(grammar != nullptr && admits(grammar, {}, kModelId) &&
                   admits(grammar, {kModelId, kThinkingId, 'a', kEndId, kModelId, kTextId}, 'b') &&
                   admits(grammar, {kModelId, kTextId, 'b', kEndId}, kEosId),
               "strict tool: free blocks around calls");
        expect(admits(grammar, opened, '{') && !admits(grammar, opened, 'x') &&
                   admits(grammar, after(R"({"name":")"), 'g') &&
                   !admits(grammar, after(R"({"name":")"), 'x'),
               "strict tool: the call names a declared function");
        expect(admits(grammar, after(R"({"name":"get_weather","args":{")"), 'c') &&
                   !admits(grammar, after(R"({"name":"get_weather","args":{")"), 'z') &&
                   !admits(grammar, after(R"({"name":"get_weather","args":{"city":"Oslo","days":)"),
                           '"') &&
                   !admits(grammar, after(R"({"name":"get_weather","args":{)"), '}'),
               "strict tool: arguments under the schema (declared, typed, required)");
        expect(admits(grammar, after(call), kEndId) &&
                   admits(grammar, joined({after(call), {kEndId}}), kModelId) &&
                   admits(grammar, joined({after(call), {kEndId}}), kEosId),
               "strict tool: the call block closes and free text follows");
        // Beside a response format, a tool without strict takes any object as arguments.
        const auto loose =
            frontend.make_output_session(prepared(kTool), {}, {}, {},
                                         ninfer::StructuredOutputOptions{
                                             .kind = ninfer::StructuredOutputKind::JsonObject})
                .grammar_state();
        expect(admits(loose, after(R"({"name":"get_weather","args":{")"), 'z'),
               "a tool without strict takes any arguments");
    }
    {
        // A forced strict call: the prompt ends with its opener, the grammar starts in its JSON.
        const auto grammar =
            frontend.make_output_session(prepared(kStrictTool, "get_weather"), {}, {}, {}, {})
                .grammar_state();
        expect(admits(grammar, {}, '{') && !admits(grammar, {}, kModelId) &&
                   !admits(grammar, bytes(R"({"name":"get_weather","args":{")"), 'z') &&
                   admits(grammar, joined({bytes(call), {kEndId}}), kModelId),
               "forced strict call starts in its JSON");
    }
    {
        // A structured response beside tools: call blocks before the text block that holds the
        // JSON, or a turn of calls without one.
        const auto grammar =
            frontend.make_output_session(prepared(kStrictTool), {}, {}, {},
                                         ninfer::StructuredOutputOptions{
                                             .kind = ninfer::StructuredOutputKind::JsonObject})
                .grammar_state();
        expect(admits(grammar, {kModelId, kTextId}, '{') &&
                   !admits(grammar, {kModelId, kTextId}, 'x') &&
                   !admits(grammar, {kModelId, kTextId}, kCallId),
               "structured beside tools: the text block holds the JSON");
        expect(admits(grammar, opened, '{') &&
                   !admits(grammar, after(R"({"name":"get_weather","args":{")"), 'z'),
               "structured beside tools: strict calls");
        expect(admits(grammar, joined({after(call), {kEndId, kModelId}}), kTextId),
               "structured beside tools: the text block after a call");
        expect(admits(grammar, joined({after(call), {kEndId}}), kEosId),
               "structured beside tools: the turn's end after a call");
        expect(!admits(grammar, {kModelId, kThinkingId, 'a', kEndId}, kEosId),
               "structured beside tools: a turn ends with the JSON or a call");
    }
}

std::shared_ptr<const fi::ToolCallOutputContract> weather_contract(std::string_view forced = {}) {
    const std::vector<std::string> tools = {kTool};
    return fi::build_tool_call_output_contract(tools, true, forced);
}

fi::InklingToolCallDecoder::Terminal run(const std::string& text, std::string& streamed,
                                         std::string_view forced = {}) {
    fi::InklingToolCallDecoder decoder(weather_contract(forced));
    for (const char byte : text) { streamed += decoder.feed(std::string(1, byte)); }
    auto terminal = decoder.finish();
    streamed += terminal.content;
    return terminal;
}

void test_tool_calls() {
    const std::string open  = std::string(fi::kInklingCallOpen);
    const std::string close = std::string(fi::kInklingEnd);
    {
        std::string streamed;
        const auto terminal =
            run("Checking. " + open + R"( {"name":"get_weather","args":{"city":"Oslo"}} )" +
                    close + " Done.",
                streamed);
        expect(streamed == "Checking.  Done.", "text around a call: " + streamed);
        expect(terminal.tool_calls.size() == 1 && terminal.diagnostics.structured_call_count == 1,
               "one call");
        if (terminal.tool_calls.size() == 1) {
            expect(terminal.tool_calls[0].name == "get_weather" &&
                       terminal.tool_calls[0].arguments_json == R"({"city":"Oslo"})",
                   "call arguments: " + terminal.tool_calls[0].arguments_json);
        }
    }
    for (const std::string body : {R"({"name":"get_weather","args":[1]})",
                                   R"({"args":{}})", R"({"name":"x","args":{},"extra":1})",
                                   R"({"name":"x","args":{)"}) {
        // An unreadable call returns to content as it was written.
        std::string streamed;
        const auto terminal = run("a" + open + body + close + "b", streamed);
        expect(terminal.tool_calls.empty() && streamed == "a" + open + body + close + "b" &&
                   terminal.diagnostics.fallback_reason ==
                       ninfer::ToolCallParseFallbackReason::MalformedStructure,
               "unreadable call returns to content: " + streamed);
    }
    {
        // An unterminated call that reads completely is a call; one that does not is content.
        std::string streamed;
        auto terminal = run(open + R"({"name":"get_weather"})", streamed);
        expect(terminal.tool_calls.size() == 1 && streamed.empty() &&
                   terminal.tool_calls[0].arguments_json == "{}",
               "complete unterminated call");
        streamed.clear();
        terminal = run(open + R"({"name":"get_wea)", streamed);
        expect(terminal.tool_calls.empty() && streamed == open + R"({"name":"get_wea)",
               "incomplete call returns to content");
    }
    {
        // A forced call: the prompt holds the opener; the model writes the JSON.
        std::string streamed;
        auto terminal = run(R"({"name":"get_weather","args":{"city":"Lima"}})", streamed,
                            "get_weather");
        expect(terminal.tool_calls.size() == 1 && streamed.empty() &&
                   terminal.diagnostics.forced_call_closed,
               "forced call closed by the decoder");
        streamed.clear();
        terminal = run(R"({"name":"get_)", streamed, "get_weather");
        expect(terminal.tool_calls.empty() && streamed == R"({"name":"get_)",
               "a failed forced call returns only the model's bytes: " + streamed);
    }
}

std::vector<std::uint8_t> synthetic_image(int width, int height) {
    std::vector<std::uint8_t> rgb(std::size_t(width) * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                rgb[(std::size_t(y) * width + x) * 3 + c] =
                    std::uint8_t((x * 7 + y * 13 + c * 101 + (x * y) % 17 * 5) % 256);
            }
        }
    }
    return rgb;
}

std::string words_digest(std::span<const std::uint16_t> words) {
    return fi::sha256_hex(fi::sha256(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(words.data()), words.size() * 2)));
}

void test_image_processing() {
    const OrderedJson cases = OrderedJson::parse(read_fixture("image_cases.json"));
    const fi::InklingMediaOptions options;
    for (const OrderedJson& c : cases.at("plans")) {
        const auto plan = fi::inkling_image_plan(c[0], c[1], options, c[2]);
        expect(plan.width == c[3] && plan.height == c[4] && plan.grid_w == c[5] &&
                   plan.grid_h == c[6] && plan.cells(options.merge) <= c[2].get<std::uint64_t>(),
               "image plan of " + c.dump());
    }
    const OrderedJson& synthetic = cases.at("synthetic");
    const int width = synthetic["width"], height = synthetic["height"];
    const std::vector<std::uint8_t> rgb = synthetic_image(width, height);
    for (const OrderedJson& c : synthetic.at("patches")) {
        const auto plan = fi::inkling_image_plan(width, height, options, c["budget"]);
        expect(plan.width == c["size"][0] && plan.height == c["size"][1] &&
                   plan.grid_w == c["grid"][0] && plan.grid_h == c["grid"][1],
               "synthetic plan at budget " + c["budget"].dump());
        const auto words = fi::inkling_patches(rgb, width, height, plan, options);
        expect(words_digest(words) == c["sha256"].get<std::string>(),
               "synthetic patches at budget " + c["budget"].dump());
    }
}

// A picture between two texts: its patches as the reference derives them, one image token per
// cell after <|content_image|>, in the user's turn of the picture.
void test_image_prompt() {
    const OrderedJson cases = OrderedJson::parse(read_fixture("image_cases.json"));
    for (const OrderedJson& image : cases.at("image")) {
        std::ifstream file(std::string(NINFER_SOURCE_DIR) + "/" + image["path"].get<std::string>(),
                           std::ios::binary);
        std::vector<std::uint8_t> png((std::istreambuf_iterator<char>(file)),
                                      std::istreambuf_iterator<char>());
        expect(!png.empty(), "the example picture reads");
        const std::uint32_t budget = image["budget"];
        const auto frontend        = inkling_frontend(true, budget == 4096 ? 16384 : budget);
        ninfer::PromptInput input;
        ninfer::MessagePart picture;
        picture.kind             = ninfer::MessagePartKind::Media;
        picture.media.kind       = ninfer::MediaKind::Image;
        picture.media.bytes      = std::move(png);
        picture.media.media_type = "image/png";
        input.messages.push_back(
            {.role  = ninfer::ChatRole::User,
             .parts = {{.kind = ninfer::MessagePartKind::Text, .text = "A"},
                       std::move(picture),
                       {.kind = ninfer::MessagePartKind::Text, .text = "B"}}});
        input.options.continuation    = ninfer::PromptContinuationMode::NewAssistantTurn;
        input.options.enable_thinking = false;
        const auto prompt = frontend.prepare(std::move(input));
        const auto& data  = ninfer::models::qwen3_5::FrontendTestAccess::inspect(prompt);
        const int gw = image["grid"][0], gh = image["grid"][1];
        const std::size_t cells = std::size_t(gw / 2) * (gh / 2);
        const std::string label = "budget " + std::to_string(budget) + ": ";
        expect(data.vision_items.size() == 1 && data.media_payloads.size() == 1, label + "one image");
        if (data.vision_items.size() != 1 || data.media_payloads.size() != 1) { continue; }
        const auto& item = data.vision_items.front();
        expect(item.grid.height == gh && item.grid.width == gw && item.token_spans.size() == 1 &&
                   item.token_spans.front().count == cells,
               label + "the image's grid and span");
        expect(words_digest(data.media_payloads.front()->span()) ==
                   image["patches_sha256"].get<std::string>(),
               label + "the image's patches");
        std::vector<ninfer::TokenId> expected =
            joined({{kSystemId, kTextId}, bytes("Thinking effort level: 0"), {kEndId},
                    {kUserId, kTextId}, bytes("A"), {kEndId, kUserId, kContentImageId}});
        const std::size_t begin = expected.size();
        expected.insert(expected.end(), cells, kImageId);
        const auto tail = joined({{kEndId, kUserId, kTextId}, bytes("B"), {kEndId}});
        expected.insert(expected.end(), tail.begin(), tail.end());
        expect(data.token_ids == expected, label + "the image's cells among the prompt's tokens");
        expect(!item.token_spans.empty() && item.token_spans.front().begin == begin,
               label + "the cells start after <|content_image|>");
    }
    // A video is refused: Inkling 2 reads images only.
    const auto frontend = inkling_frontend(true);
    ninfer::PromptInput input;
    ninfer::MessagePart clip;
    clip.kind             = ninfer::MessagePartKind::Media;
    clip.media.kind       = ninfer::MediaKind::Video;
    clip.media.bytes      = {0, 1, 2, 3};
    clip.media.media_type = "video/mp4";
    input.messages.push_back({.role = ninfer::ChatRole::User, .parts = {std::move(clip)}});
    input.options.continuation = ninfer::PromptContinuationMode::NewAssistantTurn;
    bool refused = false;
    try {
        (void)frontend.prepare(std::move(input));
    } catch (const std::exception&) { refused = true; }
    expect(refused, "a video is refused");
}

void test_thinking_control_and_loop() {
    const auto frontend = inkling_frontend();
    const auto prompt   = frontend.prepare(user_prompt("Hi"));
    // Thinking budget triggers early-close control (<guidance><|end_message|>) and suppresses
    // re-opening <|content_thinking|> until a non-thinking block opens.
    {
        auto session = frontend.make_output_session(
            prompt, {}, ninfer::OutputOptions{.tool_name_max_length = 64},
            ninfer::ThinkingControlOptions{.budget = 4});
        const auto open = joined({{kModelId, kThinkingId}, bytes("abcd")});
        const auto decision =
            session.preview_model(open, 256, ninfer::FinishReason::OutputLimit);
        (void)session.commit_preview();
        expect(!decision.finished() &&
                   decision.continuation == ninfer::runtime::ContinuationAction::ApplyTargetControl,
               "Inkling thinking budget requests ApplyTargetControl");
        const auto pending = session.pending_control_tokens();
        expect(!pending.empty() && pending.back() == kEndId,
               "Inkling thinking control span ends with <|end_message|>");
        if (!pending.empty()) {
            const std::vector<ninfer::TokenId> control(pending.begin(), pending.end());
            (void)session.preview_control(control, 250);
            (void)session.commit_preview();
            const auto stats = session.thinking_stats();
            expect(session.reasoning_closed() && stats.applied &&
                       stats.model_thinking_tokens == 4 && stats.injected_tokens == control.size() &&
                       session.banned_token() == std::optional<ninfer::TokenId>(kThinkingId),
                   "Inkling control excludes injected tokens from model_thinking_tokens and bans "
                   "<|content_thinking|>");
            const auto header = std::vector<ninfer::TokenId>{kModelId};
            (void)session.preview_model(header, 200, ninfer::FinishReason::OutputLimit);
            (void)session.commit_preview();
            expect(session.banned_token() == std::optional<ninfer::TokenId>(kThinkingId),
                   "Inkling keeps <|content_thinking|> banned inside next header");
            const auto answer = joined({{kTextId}, bytes("OK"), {kEndId, kEosId}});
            (void)session.preview_model(answer, 199, ninfer::FinishReason::OutputLimit);
            const auto out = session.commit_preview();
            expect(session.banned_token() == std::optional<ninfer::TokenId>(kThinkingId) &&
                       !out.empty(),
                   "Inkling keeps <|content_thinking|> banned across post-conclusion blocks");
        }
    }
    // Speculative round starting in a header crossing <|content_thinking|>, excess reasoning, and
    // <|end_message|> must still stop at the exact budget token and request ApplyTargetControl.
    {
        auto session = frontend.make_output_session(
            prompt, {}, ninfer::OutputOptions{.tool_name_max_length = 64},
            ninfer::ThinkingControlOptions{.budget = 2});
        const auto header = std::vector<ninfer::TokenId>{kModelId};
        (void)session.preview_model(header, 256, ninfer::FinishReason::OutputLimit);
        (void)session.commit_preview();
        const auto spec_span = joined({{kThinkingId}, bytes("abcd"), {kEndId}});
        const auto decision =
            session.preview_model(spec_span, 255, ninfer::FinishReason::OutputLimit);
        (void)session.commit_preview();
        expect(decision.accepted_tokens == 3 && !decision.finished() &&
                   decision.continuation == ninfer::runtime::ContinuationAction::ApplyTargetControl &&
                   session.thinking_stats().model_thinking_tokens == 2,
               "Inkling enforces budget per token inside speculative spans crossing <|content_thinking|>");
    }
    // Reasoning loop guard retains history across natural <|end_message|> closes so alternating
    // short thinking/text blocks cannot evade detection.
    {
        auto session = frontend.make_output_session(
            prompt, {}, ninfer::OutputOptions{.tool_name_max_length = 64},
            ninfer::ThinkingControlOptions{.loop = ninfer::ReasoningLoopAction::Conclude});
        const std::string pass =
            "Wait, let me double-check the edge case where the input list is empty. If the list is "
            "empty, the function returns zero, which matches the expected output. ";
        bool triggered = false;
        std::uint32_t budget = 100000;
        for (int block = 0; block < 32 && !triggered; ++block) {
            std::string body;
            for (int r = 0; r < 3; ++r) { body += pass; }
            const auto chunk = joined({{kModelId, kThinkingId}, bytes(body), {kEndId, kModelId, kTextId},
                                       bytes("ok"), {kEndId}});
            for (std::size_t at = 0; at < chunk.size() && !triggered; at += 128) {
                const std::size_t n = std::min<std::size_t>(128, chunk.size() - at);
                const auto d = session.preview_model(
                    std::span<const ninfer::TokenId>(chunk.data() + at, n), budget,
                    ninfer::FinishReason::OutputLimit);
                (void)session.commit_preview();
                budget -= d.accepted_tokens;
                if (d.continuation == ninfer::runtime::ContinuationAction::ApplyTargetControl ||
                    d.finished()) {
                    triggered = true;
                }
            }
        }
        expect(triggered && session.thinking_stats().loop_detected,
               "Inkling loop guard catches repeated passages across reopened thinking blocks");
    }
}

} // namespace

int main() {
    try {
        test_pretokenizer();
        test_run_reuse();
        test_whole_words();
        test_chat_template();
        test_prompt_and_output();
        test_tool_calls();
        test_image_processing();
        test_image_prompt();
        test_thinking_control_and_loop();
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
