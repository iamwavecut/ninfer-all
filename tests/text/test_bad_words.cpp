// Banned words (bad_words) as token masks: against an independent oracle, the masks after any text
// and after any drafts ban exactly the tokens whose bytes would make the text contain a word, a
// token without text separates the text around it, accept() refuses a banned token, and a grammar
// and banned words together admit the tokens both admit.
//
// The oracle reads the text as a string: a token is banned after text T when T + bytes(token)
// contains a word (T itself never does). Random walks under the masks reach the deep states, where
// the text ends in most of a word.

#include "ninfer/request.h"
#include "runtime/contract/grammar_masks.h"
#include "text/bad_words.h"
#include "text/structured_output.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::TokenId;
using namespace ninfer::text;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

bool allowed(const std::vector<std::uint32_t>& mask, std::size_t column, std::size_t words,
             TokenId token) {
    return (mask[column * words + static_cast<std::size_t>(token) / 32] >>
            (static_cast<unsigned>(token) % 32)) & 1U;
}

// Bytes 0..255 as ids 0..255, then pieces of the words in several splits, words whole inside
// longer tokens, a space and a line break before a word's opener, and two tokens without text
// (a special token and the end of the turn).
struct Fixture {
    std::vector<std::string> text;
    TokenId special = -1, eos = -1;
    // "na_" ends inside "persona_x", which it keeps from being written past "persona".
    std::vector<std::string> words = {"<daily_persona", "<message", "<accent", "aa", "persona_x",
                                      "na_"};

    Fixture() {
        for (int byte = 0; byte < 256; ++byte) { text.emplace_back(1, static_cast<char>(byte)); }
        for (const char* piece :
             {"<", "<d", "<da", "<daily", "daily", "aily", "_", "_p", "_persona", "persona", "pers",
              "ona", "on", "a", "ac", "<a", "<acc", "ent", "cent", "accent", "<message", "<mess",
              "age", "ages", "message", " <", "\n<", "<daily_persona_accent>", "x<accent>y", "aaa",
              "ba", "ab", "persona_", "na_x", "_x", "lo<", "hello", " persona", "\n\n"}) {
            text.emplace_back(piece);
        }
        special = static_cast<TokenId>(text.size());
        text.emplace_back();
        eos = static_cast<TokenId>(text.size());
        text.emplace_back();
    }

    [[nodiscard]] bool contains_word(std::string_view value) const {
        return std::any_of(words.begin(), words.end(),
                           [&](const std::string& word) { return value.find(word) != value.npos; });
    }
    // The text since the last token without text.
    [[nodiscard]] std::string tail(const std::vector<TokenId>& tokens) const {
        std::string out;
        for (const TokenId token : tokens) {
            if (text[static_cast<std::size_t>(token)].empty()) {
                out.clear();
            } else {
                out += text[static_cast<std::size_t>(token)];
            }
        }
        return out;
    }
    [[nodiscard]] bool banned_after(const std::string& tail, TokenId token) const {
        const std::string& bytes = text[static_cast<std::size_t>(token)];
        return !bytes.empty() && contains_word(tail + bytes);
    }
};

// Masks of `state` after the committed `tokens` and the `drafts`, checked column by column
// against the oracle up to the first banned draft; the columns after it are unrestricted.
void check_columns(const Fixture& f, const GrammarState& state,
                   const std::vector<TokenId>& tokens, const std::vector<TokenId>& drafts,
                   const std::string& what) {
    const std::size_t words = (f.text.size() + 31) / 32;
    std::vector<std::uint32_t> masks(words * (drafts.size() + 1));
    state.fill_masks(masks, drafts);
    std::vector<TokenId> seen = tokens;
    bool reachable            = true;
    for (std::size_t column = 0; column <= drafts.size(); ++column) {
        const std::string tail = f.tail(seen);
        for (TokenId token = 0; token < static_cast<TokenId>(f.text.size()); ++token) {
            const bool expected = !reachable || !f.banned_after(tail, token);
            if (allowed(masks, column, words, token) != expected) {
                expect(false, what + ": column " + std::to_string(column) + " token " +
                                  std::to_string(token) + " after \"" + tail + "\"");
                return;
            }
        }
        if (column == drafts.size()) { break; }
        if (reachable && f.banned_after(tail, drafts[column])) { reachable = false; }
        seen.push_back(drafts[column]);
    }
}

void test_normalize() {
    const auto words = normalize_bad_words(std::vector<std::string>{
        "<a", "  <a", "\t\n<b c", "<a", "x"});
    expect(words == std::vector<std::string>{"<a", "<b c", "x"},
           "normalize strips leading whitespace and duplicates");
    const auto refused = [](std::vector<std::string> words) {
        try {
            (void)normalize_bad_words(words);
            return false;
        } catch (const std::invalid_argument&) { return true; }
    };
    expect(refused({""}), "an empty word is refused");
    expect(refused({" \t\n"}), "a whitespace-only word is refused");
    expect(refused({std::string(kMaximumBadWordBytes + 1, 'x')}), "an over-long word is refused");
    expect(!refused({std::string(kMaximumBadWordBytes, 'x')}), "a word at the length limit passes");
    std::vector<std::string> many;
    for (std::size_t i = 0; i <= kMaximumBadWords; ++i) { many.push_back("w" + std::to_string(i)); }
    expect(refused(many), "more than the maximum distinct words are refused");
    many.pop_back();
    many.push_back("w0");
    expect(!refused(many), "duplicates do not count toward the limit");
}

void test_random_walks() {
    const Fixture f;
    BadWordsVocabulary vocabulary(f.text);
    const auto filter = vocabulary.filter(normalize_bad_words(f.words));
    expect(vocabulary.filter(normalize_bad_words(f.words)) == filter,
           "the same words reuse the cached filter");
    std::mt19937 random(7);
    // Pieces of the words are drawn far more often than other bytes, so the walks keep arriving
    // at the end of most of a word.
    std::vector<TokenId> pieces;
    for (TokenId token = 256; token < static_cast<TokenId>(f.text.size()); ++token) {
        pieces.push_back(token);
    }
    for (const char byte : std::string("<daily_persona_accent>messagex ")) {
        pieces.push_back(static_cast<unsigned char>(byte));
    }
    for (int walk = 0; walk < 300; ++walk) {
        auto state = GrammarState::with_bad_words(nullptr, filter);
        expect(!state->has_grammar(), "banned words alone hold no grammar");
        std::vector<TokenId> tokens;
        for (int step = 0; step < 40; ++step) {
            std::vector<TokenId> drafts;
            const std::size_t count = random() % 4;
            for (std::size_t d = 0; d < count; ++d) { drafts.push_back(pieces[random() % pieces.size()]); }
            check_columns(f, *state, tokens, drafts, "walk " + std::to_string(walk));
            if (failures > 0) { return; }
            const TokenId token = pieces[random() % pieces.size()];
            const bool banned   = f.banned_after(f.tail(tokens), token);
            auto preview        = state->fork();
            bool refused        = false;
            try {
                preview->accept(std::vector<TokenId>{token});
            } catch (const GrammarViolation&) { refused = true; }
            expect(refused == banned, "accept refuses exactly the banned tokens");
            if (!banned) {
                *state = std::move(*preview);
                tokens.push_back(token);
                expect(!f.contains_word(f.tail(tokens)), "an accepted token completed a word");
            }
        }
    }
}

void test_separators_and_wholes() {
    const Fixture f;
    BadWordsVocabulary vocabulary(f.text);
    auto state = GrammarState::with_bad_words(nullptr, vocabulary.filter(normalize_bad_words(f.words)));
    const auto id = [&](std::string_view piece) {
        for (TokenId token = 256; token < static_cast<TokenId>(f.text.size()); ++token) {
            if (f.text[static_cast<std::size_t>(token)] == piece) { return token; }
        }
        throw std::logic_error("no such piece");
    };
    const std::size_t words = (f.text.size() + 31) / 32;
    std::vector<std::uint32_t> mask(words);
    state->fill_masks(mask, {});
    expect(!allowed(mask, 0, words, id("<daily_persona_accent>")), "a token holding a word is banned");
    expect(!allowed(mask, 0, words, id("x<accent>y")), "a word inside a token is banned");
    expect(!allowed(mask, 0, words, id("aaa")), "a token holding a short word is banned");
    expect(allowed(mask, 0, words, f.eos) && allowed(mask, 0, words, f.special),
           "tokens without text stay allowed");
    state->accept(std::vector<TokenId>{id("<daily"), id("_"), id("pers"), id("on")});
    state->fill_masks(mask, {});
    expect(!allowed(mask, 0, words, 'a') && !allowed(mask, 0, words, id("ac")),
           "the last byte of a word is banned");
    expect(allowed(mask, 0, words, 'e') && allowed(mask, 0, words, f.eos),
           "other continuations stay allowed");
    // A special token ends the text: the word's rest after it is new text.
    state->accept(std::vector<TokenId>{f.special});
    state->fill_masks(mask, {});
    expect(allowed(mask, 0, words, 'a') && allowed(mask, 0, words, id("ona")),
           "a special token separates the text");
    expect(!allowed(mask, 0, words, id("<daily_persona_accent>")),
           "a whole word stays banned after a separator");
    // A word that begins in the text's last byte: "aa" after "xa".
    state->accept(std::vector<TokenId>{'x', 'a'});
    state->fill_masks(mask, {});
    expect(!allowed(mask, 0, words, 'a') && !allowed(mask, 0, words, id("ac")) &&
               !allowed(mask, 0, words, id("aaa")),
           "a word that starts in the text's last byte is banned");
}

void test_with_grammar() {
    // A grammar over the 256 byte tokens and a stop token: lower-case letters, '<' and '_', then
    // the stop. With "<ab" banned, the masks are the grammar's until the text ends in "<a", and
    // then the grammar's without 'b'.
    std::vector<std::string> vocab(257);
    for (int i = 0; i < 256; ++i) { vocab[i] = std::string(1, static_cast<char>(i)); }
    StructuredCompiler compiler(vocab, {256});
    auto grammar = compiler.compile({.kind = ninfer::StructuredOutputKind::Regex, .grammar = "[a-z<_]+"});
    std::vector<std::string> text = vocab;
    text[256].clear();
    BadWordsVocabulary vocabulary(text);
    auto state = GrammarState::with_bad_words(grammar, vocabulary.filter(normalize_bad_words(
                                                           std::vector<std::string>{"<ab"})));
    expect(state->has_grammar(), "a grammar with banned words keeps its grammar");
    constexpr std::size_t words = 9;
    std::vector<std::uint32_t> masks(words * 3);
    state->fill_masks(masks, std::vector<TokenId>{'<', 'a'});
    expect(allowed(masks, 0, words, 'a') && !allowed(masks, 0, words, 'A'),
           "the first column is the grammar's");
    expect(allowed(masks, 1, words, 'a') && allowed(masks, 1, words, 'b'),
           "after '<' nothing is banned yet");
    expect(!allowed(masks, 2, words, 'b') && allowed(masks, 2, words, 'c') &&
               !allowed(masks, 2, words, '0'),
           "after \"<a\" the grammar's mask loses 'b'");
    bool refused = false;
    try {
        state->accept(std::vector<TokenId>{'<', 'a', 'b'});
    } catch (const GrammarViolation&) { refused = true; }
    expect(refused, "accept refuses a banned token under a grammar");

    // A grammar that admits only the banned word leaves no token: the mask fails.
    auto only = GrammarState::with_bad_words(
        compiler.compile({.kind = ninfer::StructuredOutputKind::Regex, .grammar = "<ab"}),
        vocabulary.filter(normalize_bad_words(std::vector<std::string>{"<ab"})));
    only->accept(std::vector<TokenId>{'<', 'a'});
    std::vector<std::uint32_t> one(words);
    bool empty = false;
    try {
        only->fill_masks(one, {});
    } catch (const std::runtime_error&) { empty = true; }
    expect(empty, "a grammar and banned words that leave no token fail the mask");
}

void test_mask_job() {
    const Fixture f;
    BadWordsVocabulary vocabulary(f.text);
    auto state = GrammarState::with_bad_words(nullptr, vocabulary.filter(normalize_bad_words(f.words)));
    state->accept(std::vector<TokenId>{'<', 'd', 'a', 'i', 'l', 'y', '_', 'p', 'e', 'r', 's', 'o'});
    const std::size_t words = (f.text.size() + 31) / 32;
    const auto job = ninfer::runtime::GrammarMaskJob::start(*state, words, {'n'});
    expect(job->ready(), "a banned-words job is ready when it starts");
    expect(job->columns() == 2 && !job->error(), "the job fills its columns");
    std::vector<std::uint32_t> masks(job->masks(), job->masks() + 2 * words);
    expect(allowed(masks, 0, words, 'n') && !allowed(masks, 1, words, 'a'),
           "the job's columns follow its drafts");
}

} // namespace

int main() {
    try {
        test_normalize();
        test_random_walks();
        test_separators_and_wholes();
        test_with_grammar();
        test_mask_job();
    } catch (const std::exception& e) {
        std::cerr << "FAIL exception: " << e.what() << '\n';
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
