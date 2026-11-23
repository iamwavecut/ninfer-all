// Banned words over a real vocabulary (Gemma 4's, for one): the tag openers that a chat bot bans
// are spelled by many token splits, after a line break, a space or other text, and the masks ban
// the token that would complete any of them, whatever split leads there, and nothing else.
//   NINFER_BAD_WORDS_TOKENIZER  a directory with a release's tokenizer.json, tokenizer_config.json
//                               and generation_config.json (skipped without one)
// The vocabulary is read as the frontend reads it for banned words: a special token has no text.
// The oracle is a substring search over the text since the last special token.

#include "models/qwen3_5/frontend/tokenizer.h"
#include "text/bad_words.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::TokenId;
using ninfer::text::BadWordsFilter;
namespace fi = ninfer::models::qwen3_5::frontend;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path.string()); }
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

std::string printable(std::string_view bytes) {
    std::string out;
    for (const char byte : bytes) {
        if (byte == '\n') {
            out += "\\n";
        } else if (static_cast<unsigned char>(byte) < 0x20) {
            out += "\\x" + std::to_string(static_cast<unsigned char>(byte));
        } else {
            out += byte;
        }
    }
    return out;
}

// The openers of a chat bot's prompt markup, which it bans so that the model does not copy the
// prompt's blocks into its reply.
const std::vector<std::string> kWords = {
    "<message",       "<messages",         "<last_message",  "<message_type", "<text",
    "<user",          "<reply_to",         "<attach",        "<history",      "<chat_context",
    "<current_",      "<locale",           "<reference_context", "<memory",   "<shield",
    "<custom_persona", "<daily_persona",   "<accent",        "<system_contract", "<identity",
    "<base_voice",    "<persona",          "<task",          "<rule",         "<dialog_",
    "<answer_policy", "<output_and_memory", "<transport",    "<tool_contract", "<naming",
    "<final_check",   "<check",            "<step",          "<answer",       "<final_",
    "<response",      "<reply",            "<think",         "<thought",      "<reasoning",
    "<channel",       "<assistant",        "<context"};

struct Vocabulary {
    std::vector<std::string> text;
    bool contains_word(std::string_view value) const {
        return std::any_of(kWords.begin(), kWords.end(),
                           [&](const std::string& word) { return value.find(word) != value.npos; });
    }
    bool banned_after(const std::string& tail, TokenId token) const {
        const std::string& bytes = text[static_cast<std::size_t>(token)];
        return !bytes.empty() && contains_word(tail + bytes);
    }
};

// The mask after `tail` (reached through `state`) against the oracle, over the whole vocabulary.
void check_mask(const Vocabulary& vocabulary, const BadWordsFilter& filter,
                BadWordsFilter::State state, const std::string& tail, std::size_t& banned) {
    std::vector<std::uint32_t> mask((vocabulary.text.size() + 31) / 32, ~std::uint32_t{0});
    filter.clear_banned(state, mask);
    banned = 0;
    for (TokenId token = 0; token < static_cast<TokenId>(vocabulary.text.size()); ++token) {
        const bool allowed = (mask[static_cast<std::size_t>(token) / 32] >> (token % 32)) & 1U;
        if (allowed == vocabulary.banned_after(tail, token)) {
            expect(false, "token " + std::to_string(token) + " \"" +
                              printable(vocabulary.text[static_cast<std::size_t>(token)]) +
                              "\" after \"" + printable(tail) + "\"");
            return;
        }
        banned += allowed ? 0 : 1;
    }
}

int run() {
    const char* directory = std::getenv("NINFER_BAD_WORDS_TOKENIZER");
    if (directory == nullptr || *directory == '\0') {
        std::cerr << "SKIP: NINFER_BAD_WORDS_TOKENIZER is not set\n";
        return 77;
    }
    const std::filesystem::path root(directory);
    const std::string tokenizer_json  = read_file(root / "tokenizer.json");
    const std::string config_json     = read_file(root / "tokenizer_config.json");
    const std::string generation_json = read_file(root / "generation_config.json");
    const fi::Tokenizer tokenizer(fi::TokenizerResources{.tokenizer_json         = tokenizer_json,
                                                         .tokenizer_config_json  = config_json,
                                                         .generation_config_json = generation_json});
    Vocabulary vocabulary;
    vocabulary.text.resize(tokenizer.vocab_size());
    for (std::size_t id = 0; id < vocabulary.text.size(); ++id) {
        const int token = static_cast<int>(id);
        if (tokenizer.is_valid_token(token) && !tokenizer.is_special_token(token)) {
            vocabulary.text[id] = tokenizer.decode_token_bytes(token, false);
        }
    }
    ninfer::text::BadWordsVocabulary words_vocabulary(vocabulary.text);
    const auto filter = words_vocabulary.filter(ninfer::text::normalize_bad_words(kWords));

    // Replies that copy the prompt's blocks, as a model writes them: each is cut at the token
    // that completes its first banned word, and no token before it is banned.
    const std::vector<std::string> replies = {
        "<daily_persona_accent><name>Ира</name><accent>сухо</accent></daily_persona_accent>",
        "Конечно, вот ответ.\n\n<daily_persona_accent><name>Ира</name>",
        "Sure. <daily_persona_accent>",
        "ok<daily_persona_accent>",
        "<message type=\"text\">hi</message>",
        "\n<messages>",
        " <last_message>",
        "текст</text>\n<text>",
        "<think>hmm</think>",
        "a <answer_policy> b",
        "<|channel>thought\n<channel|>",
    };
    for (const std::string& reply : replies) {
        const std::vector<int> ids = tokenizer.encode(reply, fi::EncodeOptions{});
        std::string pieces;
        BadWordsFilter::State state = BadWordsFilter::kStart;
        std::string tail;
        bool cut = false;
        for (const int id : ids) {
            const std::string& bytes = vocabulary.text[static_cast<std::size_t>(id)];
            pieces += "[" + printable(bytes.empty() ? tokenizer.decode_token_bytes(id, false) : bytes) + "]";
            const bool oracle = vocabulary.banned_after(tail, id);
            std::vector<std::uint32_t> mask((vocabulary.text.size() + 31) / 32, ~std::uint32_t{0});
            filter->clear_banned(state, mask);
            const bool masked = ((mask[static_cast<std::size_t>(id) / 32] >> (id % 32)) & 1U) == 0;
            const auto next = filter->advance(state, id);
            expect(masked == oracle && !next == oracle,
                   "reply \"" + printable(reply) + "\" at token " + std::to_string(id));
            if (!next) {
                cut = true;
                break;
            }
            state = *next;
            tail  = bytes.empty() ? std::string() : tail + bytes;
        }
        std::cout << (cut ? "cut  " : "free ") << printable(reply) << "\n     " << pieces << '\n';
        expect(cut == vocabulary.contains_word(reply) ||
                   reply.find("<|channel>") != std::string::npos,
               "a reply holding a banned word is cut: " + printable(reply));
    }

    // The whole mask against the oracle after texts that end in most of a word, in the splits
    // the tokenizer gives them.
    const std::vector<std::string> contexts = {
        "",           "<",       "Hi <",        "\n\n<",        "<daily",  "<daily_pers",
        "<daily_persona_ac", "<mess", "<messa", "word<", "<final", "<final_che", "<chec",
        "<reasonin",  "<dialog", "x <curren", "<custom_persona"};
    for (const std::string& context : contexts) {
        BadWordsFilter::State state = BadWordsFilter::kStart;
        std::string tail;
        bool usable = true;
        for (const int id : tokenizer.encode(context, fi::EncodeOptions{})) {
            const auto next = filter->advance(state, id);
            if (!next) {
                usable = false;
                break;
            }
            state = *next;
            const std::string& bytes = vocabulary.text[static_cast<std::size_t>(id)];
            tail = bytes.empty() ? std::string() : tail + bytes;
        }
        if (!usable) {
            std::cout << "context \"" << printable(context) << "\" holds a banned word\n";
            continue;
        }
        std::size_t banned = 0;
        check_mask(vocabulary, *filter, state, tail, banned);
        std::cout << "after \"" << printable(context) << "\": " << banned << " tokens banned\n";
    }

    // Random walks over tokens that spell the words and their surroundings: every token the
    // filter accepts leaves text without a word, and every token it refuses would complete one.
    std::vector<int> pieces;
    for (const std::string& word : kWords) {
        for (const std::string& variant : {word, " " + word, "\n" + word, word + "_" , word + ">"}) {
            for (const int id : tokenizer.encode(variant, fi::EncodeOptions{})) { pieces.push_back(id); }
        }
    }
    for (const char* filler : {"a", "e", "s", "_", ">", " ", "\n", "x"}) {
        for (const int id : tokenizer.encode(filler, fi::EncodeOptions{})) { pieces.push_back(id); }
    }
    std::mt19937 random(11);
    for (int walk = 0; walk < 2000; ++walk) {
        BadWordsFilter::State state = BadWordsFilter::kStart;
        std::string tail;
        for (int step = 0; step < 24; ++step) {
            const int id     = pieces[random() % pieces.size()];
            const auto next  = filter->advance(state, id);
            const bool would = vocabulary.banned_after(tail, id);
            if (!next != would) {
                expect(false, "walk refuses token " + std::to_string(id) + " after \"" +
                                  printable(tail) + "\" wrongly");
                return 1;
            }
            if (!next) { continue; }
            state = *next;
            const std::string& bytes = vocabulary.text[static_cast<std::size_t>(id)];
            tail = bytes.empty() ? std::string() : tail + bytes;
        }
    }
    return failures == 0 ? 0 : 1;
}

} // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::cerr << "FAIL exception: " << e.what() << '\n';
        return 1;
    }
}
