// The reasoning-loop measure: words split across token boundaries join; a reasoning that keeps
// rewriting the same verification passes reaches the threshold; varied work, and a short text, stay
// below it; a passage repeated twice is not a loop; a loop that counts is found by its tail, and
// numbers that do not count are not.
#include "models/qwen3_5/frontend/reasoning_loop.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer::models::qwen3_5;

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (!condition) {
        std::cerr << "FAIL " << label << '\n';
        ++failures;
    }
}

std::vector<std::uint64_t> words_of(const std::string& text, std::size_t piece = 3) {
    std::vector<std::uint64_t> words;
    std::string partial;
    for (std::size_t at = 0; at < text.size(); at += piece) {
        append_words(std::string_view(text).substr(at, piece), partial, words);
    }
    append_words(" ", partial, words);
    return words;
}

// Deterministic varied prose: a new combination of words in every sentence.
std::string varied(int sentences) {
    static const char* subjects[] = {"the parser", "this branch", "the cache", "our bound",
                                     "the loop",   "a tile",      "the queue", "that edge",
                                     "the schema", "each lane",   "the budget"};
    static const char* verbs[]    = {"handles", "skips",  "rounds", "checks", "splits",
                                     "merges",  "drops",  "keeps",  "orders", "limits",
                                     "scales",  "tracks", "maps"};
    static const char* objects[]  = {"empty input",    "the last token", "two columns",
                                     "a ragged width", "every retry",    "the null case",
                                     "large values",   "unicode names",  "negative offsets",
                                     "the first page", "stale entries",  "short reads"};
    std::string text;
    unsigned state = 12345;
    for (int s = 0; s < sentences; ++s) {
        state = state * 1103515245U + 12345U;
        text += std::string(subjects[(state >> 8) % 11]) + " " + verbs[(state >> 12) % 13] + " " +
                objects[(state >> 16) % 12] + " when step " + std::to_string(s) +
                " runs with value " + std::to_string((state >> 4) % 997) + ". ";
    }
    return text;
}

} // namespace

int main() {
    // Splitting: case folding, punctuation as words, a word cut by a token boundary.
    {
        std::vector<std::uint64_t> a, b;
        std::string pa, pb;
        append_words("Hello, World", pa, a);
        append_words("!", pa, a);
        append_words("hel", pb, b);
        append_words("lo , world!", pb, b);
        expect(a == b && a.size() == 4, "split and fold");
    }
    // A non-word character is a word of its own even without separators, whether it arrives whole
    // or one byte per call.
    {
        std::vector<std::uint64_t> a, b;
        std::string pa, pb;
        append_words("\xe2\x9a\xa1\xe2\x9a\xa1"
                     "ab",
                     pa, a);
        append_words(" ", pa, a);
        append_words("\xe2", pb, b);
        append_words("\x9a", pb, b);
        append_words("\xa1", pb, b);
        append_words("\xe2\x9a\xa1"
                     "ab",
                     pb, b);
        append_words(" ", pb, b);
        expect(a == b && a.size() == 3, "unseparated symbol splits per character");
    }
    // A run of CJK letters is one word, as `\w+` reads it.
    {
        std::vector<std::uint64_t> words;
        std::string partial;
        append_words("\xe4\xb8\xad\xe6\x96\x87", partial, words);
        append_words(" ", partial, words);
        expect(words.size() == 1, "cjk is one word");
    }
    // ASCII punctuation keeps one word per character.
    {
        std::vector<std::uint64_t> words;
        std::string partial;
        append_words("?!?!?!", partial, words);
        append_words(" ", partial, words);
        expect(words.size() == 6, "punctuation per character");
    }
    // Below a full window nothing is measured.
    expect(repeated_passage_coverage(words_of(varied(20))) == 0.0, "short text");
    // Varied work stays low.
    const double novel = repeated_passage_coverage(words_of(varied(400)));
    expect(novel < 0.05, "varied reasoning coverage " + std::to_string(novel));
    // The same verification pass over and over, after some real work.
    const std::string pass =
        "Wait, let me double-check the edge case where the input list is empty. If the list is "
        "empty, "
        "the function returns zero, which matches the expected output. So that case is fine. ";
    std::string looping = varied(200);
    for (int i = 0; i < 40; ++i) { looping += pass; }
    const double loop = repeated_passage_coverage(words_of(looping));
    expect(loop >= kLoopCoverage, "looping coverage " + std::to_string(loop));
    // The same loop written as one unseparated run of symbols: every symbol is a word of its own,
    // so the passage repeats and the measure sees it (a byte-level split makes it a single word).
    {
        std::string symbols;
        for (int i = 0; i < 3000; ++i) { symbols += "\xe2\x9a\xa1"; }
        const double symbol_loop = repeated_passage_coverage(words_of(varied(150) + symbols, 1));
        expect(symbol_loop >= kLoopCoverage,
               "unseparated symbol loop coverage " + std::to_string(symbol_loop));
    }
    // A passage seen twice is not repeated.
    std::string twice  = varied(300) + pass + varied(10) + pass + varied(150);
    const double recap = repeated_passage_coverage(words_of(twice));
    expect(recap < kLoopCoverage, "recap coverage " + std::to_string(recap));
    // A loop written as one long word: a periodic tail reads as full coverage.
    {
        std::string digits;
        for (int i = 0; i < 200; ++i) { digits += "58747693387"; } // 2200 characters, period 11
        expect(periodic_tail(digits), "periodic tail");
        expect(!periodic_tail("58747693387"), "tail below the window");
        expect(!periodic_tail(varied(200)), "varied tail is not periodic");
    }
    // The period counts characters, not bytes: 30 CJK characters (90 bytes) repeat, 65 do not.
    {
        const auto cjk = [](int n) {
            std::string unit;
            for (int i = 0; i < n; ++i) {
                const int c = 0x4e00 + i; // three UTF-8 bytes each
                unit += static_cast<char>(0xe0 | (c >> 12));
                unit += static_cast<char>(0x80 | ((c >> 6) & 0x3f));
                unit += static_cast<char>(0x80 | (c & 0x3f));
            }
            std::string text;
            while (text.size() < 3U * (kLoopTailChars + 2U * n)) { text += unit; }
            return text;
        };
        expect(periodic_tail(cjk(30)), "30-character period of 3-byte characters");
        expect(periodic_tail(cjk(64)), "64-character period of 3-byte characters");
        expect(!periodic_tail(cjk(65)), "65-character period is beyond the limit");
    }
    // A loop that counts writes a new word every time (#260): with its numbers masked the tail is
    // periodic, and one number of each period moves by the same step.
    {
        const auto counting = [](const std::string& name, int from, int step, int count) {
            std::string text = varied(150) + "BiquadFilter types: 'deviceDirection','gain','GPS',";
            for (int i = 0, v = from; i < count; ++i, v += step) {
                text += "'" + name + std::to_string(v) + "',";
            }
            return text;
        };
        const std::string reported = counting("time", 1, 1, 2394);
        expect(counting_tail(reported), "counting tail");
        expect(!periodic_tail(reported), "a counting tail has no plain period");
        expect(counting_tail(counting("x", 1000, 2, 400)), "counting by two");
        expect(counting_tail(counting("level", 900, -1, 400)), "counting down");
        // Arriving one token's bytes at a time through the bounded tail changes nothing.
        std::string tail;
        for (std::size_t at = 0; at < reported.size(); at += 3) {
            append_tail(std::string_view(reported).substr(at, 3), tail);
        }
        expect(counting_tail(tail), "counting tail through append_tail");
        expect(!counting_tail(counting("time", 1, 1, 60)), "counting tail below the window");
        // A counter beside constants still counts, and so do numbers that all keep their steps.
        std::string of_total, pairs;
        for (int i = 1; i <= 300; ++i) {
            of_total += "item " + std::to_string(i) + " of 300, ";
            pairs += "(" + std::to_string(i) + ", " + std::to_string(3 * i + 1) + ") ";
        }
        expect(counting_tail(of_total), "a counter beside a constant");
        expect(counting_tail(pairs), "two numbers with their own steps");
    }
    // Numbers that do not count stay quiet: varied work, a repeated constant (the plain period's
    // case), unrelated values in a periodic frame, squares, whose steps grow, and an enumeration:
    // a counter beside values that change otherwise.
    {
        expect(!counting_tail(varied(400)), "varied work with numbers");
        std::string constant, unrelated, squares, cases, enumeration;
        unsigned state = 7;
        for (int i = 1; i <= 400; ++i) {
            constant += "x = 5, ";
            state = state * 1103515245U + 12345U;
            unrelated += "'id" + std::to_string((state >> 8) % 100000) + "',";
            squares += std::to_string(i * i) + ", ";
            enumeration += "step " + std::to_string(i) + " checks value " +
                           std::to_string((state >> 8) % 9973) + ". ";
        }
        expect(!counting_tail(enumeration), "a counter beside changing values");
        for (int n = 1; n <= 40; ++n) {
            cases += "For n = " + std::to_string(n) + ", the sum is " + std::to_string(n * (n + 1) / 2) +
                     ((n % 3) == 0 ? ", which is divisible by 3. " : ", which is not. ");
        }
        expect(!counting_tail(constant), "a repeated constant");
        expect(!counting_tail(unrelated), "unrelated values in a periodic frame");
        expect(!counting_tail(squares), "squares");
        expect(!counting_tail(cases), "worked cases with varied text");
    }
    // The tail stays bounded and keeps its characters however the bytes arrive.
    {
        std::string tail;
        const std::string symbol = "\xe2\x9a\xa1";
        for (int i = 0; i < 8000; ++i) { append_tail(symbol, tail); }
        expect(periodic_tail(tail), "bounded tail is periodic");
        expect(tail.size() <= 8U * kLoopTailChars, "tail stays bounded");
    }
    if (failures != 0) { return 1; }
    std::cout << "OK reasoning loop (varied " << novel << ", looping " << loop << ", recap "
              << recap << ")\n";
    return 0;
}
