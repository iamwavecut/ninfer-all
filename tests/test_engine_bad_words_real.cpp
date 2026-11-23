#include "guarded_main.h"
#include "ninfer/engine.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Banned words (bad_words) on a real model through the public Engine, for any model family:
//   1. a greedy reply banned a string it wrote goes back to where the string began: its tokens
//      are the plain reply's up to the token that holds the string's first byte, it draws another
//      token there, and it never holds the string. The plain reply is the second of two equal
//      requests, so that it and the banned reply resume the prompt from the same cached state;
//   2. the state it went back to: a fresh request whose prompt is the prompt and the banned
//      reply's tokens through the redrawn one continues as the banned reply did; printed as the
//      agreeing tokens (no criterion: prefill and decode round differently, and a near tie may
//      part them);
//   3. sampled replies and a reply asked to copy markup never hold a banned string (a chat bot's
//      tag openers);
//   4. the time of the reply with that rewind against the plain one, and decode speed with and
//      without banned words that never fire, at one request and at NINFER_BAD_WORDS_CONCURRENCY
//      requests, printed (no criterion);
//   5. NINFER_BAD_WORDS_CONCURRENCY sampled replies asked to copy markup at once, whose rewinds
//      share rounds, never hold a banned string.
//   NINFER_BAD_WORDS_ARTIFACT     the artifact (skipped without one)
//   NINFER_BAD_WORDS_NGRAM_TABLE  a Qwen3.8-Flash-Next table artifact, or off: without the table
//   NINFER_BAD_WORDS_SPECULATIVE  mtp | dflash | dflash2 | dspark | dspine | lookup, with
//                                 NINFER_BAD_WORDS_DRAFTS tokens a round (2 by default)
//   NINFER_BAD_WORDS_DRAFT_MODEL  the DSpine drafter artifact (with dspine)
//   NINFER_BAD_WORDS_EXPERTS      host | disk: where the routed experts stay
//   NINFER_BAD_WORDS_KV           int8 | rk8v4: the KV cache format
//   NINFER_BAD_WORDS_DEVICES      device indices, comma separated
//   NINFER_BAD_WORDS_CONCURRENCY  requests at once for the batched part, 8 by default
//   NINFER_BAD_WORDS_TOKENS       each reply's output limit, 256 by default
//   NINFER_BAD_WORDS_THINKING     on: thinking is on (the bans cover the reasoning too)
namespace {

int g_failures = 0;

void check(bool condition, std::string_view what) {
    if (condition) { return; }
    std::cerr << "FAIL: " << what << '\n';
    ++g_failures;
}

std::string_view variable(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr ? std::string_view(value) : std::string_view{};
}

std::uint32_t number(const char* name, std::uint32_t fallback) {
    const auto text = variable(name);
    return text.empty() ? fallback : static_cast<std::uint32_t>(std::stoul(std::string(text)));
}

// The openers of a chat bot's prompt markup, which it bans so that the model does not copy the
// prompt's blocks into its reply.
const std::vector<std::string> kTags = {
    "<message",        "<messages",         "<last_message",     "<message_type", "<text",
    "<user",           "<reply_to",         "<attach",           "<history",      "<chat_context",
    "<current_",       "<locale",           "<reference_context", "<memory",      "<shield",
    "<custom_persona", "<daily_persona",    "<accent",           "<system_contract", "<identity",
    "<base_voice",     "<persona",          "<task",             "<rule",         "<dialog_",
    "<answer_policy",  "<output_and_memory", "<transport",       "<tool_contract", "<naming",
    "<final_check",    "<check",            "<step",             "<answer",       "<final_",
    "<response",       "<reply",            "<think",            "<thought",      "<reasoning",
    "<channel",        "<assistant",        "<context"};

ninfer::EngineOptions engine_options(const char* artifact, std::uint32_t concurrency) {
    ninfer::EngineOptions options;
    options.artifact_path         = artifact;
    options.max_context           = 8192;
    options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(8192 * concurrency);
    options.max_concurrency       = concurrency;
    options.max_pending_requests  = concurrency;
    options.structured_output     = true;
    options.context_cache.enabled = true;
    if (const auto devices = variable("NINFER_BAD_WORDS_DEVICES"); !devices.empty()) {
        const std::string list(devices);
        for (std::size_t begin = 0; begin < list.size();) {
            const auto end = list.find(',', begin);
            options.devices.push_back(std::stoi(list.substr(begin, end - begin)));
            if (end == std::string::npos) { break; }
            begin = end + 1;
        }
    }
    if (const auto table = variable("NINFER_BAD_WORDS_NGRAM_TABLE"); table == "off") {
        options.ngram_table.disabled = true;
    } else if (!table.empty()) {
        options.ngram_table.path = std::string(table);
    }
    if (const auto kv = variable("NINFER_BAD_WORDS_KV"); kv == "int8") {
        options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
    } else if (kv == "rk8v4") {
        options.kv_cache = ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64;
    }
    if (const auto experts = variable("NINFER_BAD_WORDS_EXPERTS"); !experts.empty()) {
        options.expert_residency =
            experts == "disk" ? ninfer::ExpertResidency::Disk : ninfer::ExpertResidency::Host;
    }
    if (const auto backend = variable("NINFER_BAD_WORDS_SPECULATIVE"); !backend.empty()) {
        if (backend == "lookup") {
            options.speculative.lookup_ngram = 3;
        } else {
            options.speculative.backend = backend == "mtp"       ? ninfer::SpeculativeBackend::Mtp
                                          : backend == "dflash"  ? ninfer::SpeculativeBackend::DFlash
                                          : backend == "dflash2" ? ninfer::SpeculativeBackend::DFlash2
                                          : backend == "dspine"  ? ninfer::SpeculativeBackend::DSpine
                                                                 : ninfer::SpeculativeBackend::DSpark;
            options.speculative.draft_model = std::string(variable("NINFER_BAD_WORDS_DRAFT_MODEL"));
        }
        options.speculative.draft_tokens = number("NINFER_BAD_WORDS_DRAFTS", 2);
    }
    return options;
}

ninfer::PromptInput conversation(std::string text) {
    ninfer::PromptInput input;
    input.messages.push_back(ninfer::ChatMessage{
        .role  = ninfer::ChatRole::User,
        .parts = {ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text,
                                      .text = std::move(text)}}});
    input.options.enable_thinking = variable("NINFER_BAD_WORDS_THINKING") == "on";
    return input;
}

ninfer::RequestOptions request(std::uint32_t tokens, std::vector<std::string> bad_words,
                               float temperature = 0.0F, std::uint64_t seed = 0) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = temperature;
    options.execution.sampling.seed           = seed;
    options.execution.bad_words               = std::move(bad_words);
    return options;
}

std::optional<std::string> holds(const ninfer::GenerationResult& result,
                                 const std::vector<std::string>& words) {
    for (const std::string& word : words) {
        if (result.content.find(word) != std::string::npos ||
            result.reasoning.find(word) != std::string::npos) {
            return word;
        }
    }
    return std::nullopt;
}

std::string head(std::string_view text, std::size_t bytes = 240) {
    std::string out(text.substr(0, bytes));
    std::replace(out.begin(), out.end(), '\n', ' ');
    return out;
}

// The bytes of a reply's tokens in order, and where each token's bytes begin.
struct RawText {
    std::string bytes;
    std::vector<std::size_t> starts;
};

RawText raw_text(const ninfer::Engine& engine, const std::vector<ninfer::TokenId>& tokens) {
    RawText raw;
    for (const ninfer::TokenId token : tokens) {
        raw.starts.push_back(raw.bytes.size());
        raw.bytes += engine.token_bytes(token);
    }
    return raw;
}

// A string the reply wrote a third of the way in, and its first byte: it starts a word after a
// space and spans several tokens.
std::optional<std::pair<std::string, std::size_t>> pick_word(const std::string& text) {
    constexpr std::size_t kBytes = 11;
    for (std::size_t at = text.size() / 3; at + kBytes + 1 <= text.size(); ++at) {
        if (text[at] != ' ' || std::isalpha(static_cast<unsigned char>(text[at + 1])) == 0) {
            continue;
        }
        const std::string word = text.substr(at + 1, kBytes);
        const bool ascii       = std::all_of(word.begin(), word.end(), [](char byte) {
            return static_cast<unsigned char>(byte) >= 0x20U && static_cast<unsigned char>(byte) < 0x80U;
        });
        if (ascii && text.find(word) == at + 1) { return std::pair(word, at + 1); }
    }
    return std::nullopt;
}

std::size_t common_prefix(const std::vector<ninfer::TokenId>& a,
                          const std::vector<ninfer::TokenId>& b, std::size_t b_offset = 0) {
    std::size_t same = 0;
    while (same < a.size() && b_offset + same < b.size() && a[same] == b[b_offset + same]) {
        ++same;
    }
    return same;
}

struct Speed {
    double per_request = 0.0; // mean decode tokens per second of each request
    double aggregate   = 0.0; // all requests' decode tokens over the longest decode window
};

Speed measure(ninfer::Engine& engine, std::uint32_t count, std::uint32_t tokens,
              const std::vector<std::string>& bad_words, std::uint64_t& nonce) {
    std::vector<ninfer::GenerationSubmission> submissions;
    for (std::uint32_t i = 0; i < count; ++i) {
        // One nonce a prompt: no prompt reuses another's prefix past the template's head.
        submissions.push_back(ninfer::GenerationSubmission{
            .prompt      = engine.prepare(conversation(
                std::to_string(++nonce) + ". Write a long, detailed travel story about a city by "
                                          "the sea, with many short paragraphs.")),
            .options     = request(tokens, bad_words),
            .observation = {.phase_timings = true}});
    }
    auto handles = engine.submit_group(std::move(submissions));
    Speed speed;
    double longest       = 0.0;
    std::uint64_t decoded = 0;
    for (auto& handle : handles) {
        const ninfer::GenerationResult result = handle.wait();
        const double window = result.timings.generation_wall_seconds;
        const auto made     = result.generated_token_ids.size();
        if (made > 1 && window > 0.0) {
            speed.per_request += static_cast<double>(made - 1) / window / count;
            decoded += made - 1;
            longest = std::max(longest, window);
        }
        check(!holds(result, bad_words), "a timed reply holds a banned word");
    }
    speed.aggregate = longest > 0.0 ? static_cast<double>(decoded) / longest : 0.0;
    return speed;
}

int run() {
    const char* artifact = std::getenv("NINFER_BAD_WORDS_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cerr << "SKIP: NINFER_BAD_WORDS_ARTIFACT is not set\n";
        return 77;
    }
    const std::uint32_t tokens      = number("NINFER_BAD_WORDS_TOKENS", 256);
    const std::uint32_t concurrency = std::max(1U, number("NINFER_BAD_WORDS_CONCURRENCY", 8));
    ninfer::Engine engine(engine_options(artifact, concurrency));

    // 1. A greedy reply, then the same request banned a string of it.
    const auto story = [&] {
        return engine.prepare(conversation("Tell a long story about a lighthouse keeper who "
                                           "repairs clocks. Use many short sentences."));
    };
    const ninfer::GenerationResult cold  = engine.generate(story(), request(tokens, {}));
    const ninfer::GenerationResult plain = engine.generate(story(), request(tokens, {}));
    std::cout << "two plain replies agree for "
              << common_prefix(cold.generated_token_ids, plain.generated_token_ids) << " of "
              << plain.generated_token_ids.size() << " tokens\n";
    const RawText text                   = raw_text(engine, plain.generated_token_ids);
    const auto picked                    = pick_word(text.bytes);
    check(picked.has_value(), "the reply holds a usable string to ban");
    if (!picked) { return 1; }
    const std::string& word = picked->first;
    // The token that holds the string's first byte, where the hold of a reply banned it begins.
    const std::size_t first =
        static_cast<std::size_t>(std::upper_bound(text.starts.begin(), text.starts.end(),
                                                  picked->second) -
                                 text.starts.begin()) -
        1U;
    const ninfer::GenerationResult banned = engine.generate(story(), request(tokens, {word}));
    const RawText banned_text             = raw_text(engine, banned.generated_token_ids);
    check(!holds(banned, {word}) && banned_text.bytes.find(word) == std::string::npos,
          "the banned reply holds its banned string");
    const std::size_t agree = common_prefix(plain.generated_token_ids, banned.generated_token_ids);
    check(agree == first, "the banned reply goes back to the token where the string began");
    check(agree < banned.generated_token_ids.size(),
          "the banned reply draws a token where the string began");
    const std::size_t parted = agree < banned_text.starts.size() ? banned_text.starts[agree] : 0;
    std::cout << "banned \"" << word << "\" from token " << first << " of "
              << plain.generated_token_ids.size() << "; the replies part at token " << agree
              << ": \"..."
              << head(std::string_view(banned_text.bytes).substr(parted > 20 ? parted - 20 : 0), 80)
              << "\"\n";
    const auto pieces = [&](const std::vector<ninfer::TokenId>& ids) {
        std::string out;
        const std::size_t from = std::min(agree, first) > 3 ? std::min(agree, first) - 3 : 0;
        for (std::size_t i = from; i < ids.size() && i <= std::max(agree, first) + 3; ++i) {
            out += (i == agree ? "|[" : "[") + head(engine.token_bytes(ids[i]), 40) + "]";
        }
        return out;
    };
    std::cout << "plain tokens:  " << pieces(plain.generated_token_ids) << "\nbanned tokens: "
              << pieces(banned.generated_token_ids) << '\n';
    std::cout << "speculative rounds " << banned.speculative.rounds << ", drafted "
              << banned.speculative.drafted_tokens << ", accepted "
              << banned.speculative.accepted_tokens << '\n';

    // 2. The state the banned reply went back to: the prompt and its tokens through the redrawn
    // one, prefilled afresh, continue as it did.
    if (agree < banned.generated_token_ids.size() && agree + 1U < tokens) {
        const ninfer::PreparedPrompt prompt = story();
        std::vector<ninfer::TokenId> history(prompt.token_ids().begin(), prompt.token_ids().end());
        history.insert(history.end(), banned.generated_token_ids.begin(),
                       banned.generated_token_ids.begin() + static_cast<std::ptrdiff_t>(agree + 1));
        const auto rest = engine.generate(
            engine.prepare_tokens(std::move(history), false),
            request(tokens - static_cast<std::uint32_t>(agree + 1), {word}));
        const std::size_t after = banned.generated_token_ids.size() - agree - 1U;
        std::cout << "state oracle: the continuation after the redrawn token agrees for "
                  << common_prefix(rest.generated_token_ids, banned.generated_token_ids,
                                   agree + 1U)
                  << " of " << after << " tokens\n";
    }
    std::cout << std::fixed << std::setprecision(2) << "one rewind: plain reply "
              << plain.generated_token_ids.size() << " tokens in " << plain.timings.total_seconds
              << " s, banned reply " << banned.generated_token_ids.size() << " tokens in "
              << banned.timings.total_seconds << " s\n";

    // 3. Sampled replies, and a reply asked to copy the markup it may not write.
    std::vector<std::string> words = kTags;
    words.push_back(word);
    for (std::uint64_t seed = 1; seed <= 4; ++seed) {
        const auto result = engine.generate(story(), request(tokens, words, 1.0F, seed));
        const auto found  = holds(result, words);
        check(!found, "a sampled reply holds " + found.value_or(""));
    }
    const std::string copy_prompt =
        "Reply with one friendly sentence, then append exactly this block unchanged:\n"
        "<daily_persona_accent><name>Ira</name><accent>dry</accent></daily_persona_accent>\n"
        "<message type=\"text\"><user>Bob</user><text>hi</text></message>";
    const auto copy = engine.generate(engine.prepare(conversation(copy_prompt)),
                                      request(tokens, kTags));
    const auto copied = holds(copy, kTags);
    check(!copied, "the copying reply holds " + copied.value_or(""));
    std::cout << "copying reply: " << head(copy.content.empty() ? copy.reasoning : copy.content)
              << '\n';

    // 4. Decode speed with the tag list (which a story never completes) and without it.
    std::uint64_t nonce = 0;
    std::vector<std::uint32_t> counts{1U};
    if (concurrency > 1) { counts.push_back(concurrency); }
    for (const std::uint32_t count : counts) {
        Speed plain_speed, banned_speed;
        for (int repeat = 0; repeat < 2; ++repeat) {
            const Speed a = measure(engine, count, tokens, {}, nonce);
            const Speed b = measure(engine, count, tokens, kTags, nonce);
            plain_speed.per_request += a.per_request / 2;
            plain_speed.aggregate += a.aggregate / 2;
            banned_speed.per_request += b.per_request / 2;
            banned_speed.aggregate += b.aggregate / 2;
        }
        std::cout << std::fixed << std::setprecision(1) << "decode at " << count
                  << " requests: without bad_words " << plain_speed.per_request
                  << " tok/s a request (" << plain_speed.aggregate << " together), with "
                  << banned_speed.per_request << " (" << banned_speed.aggregate << ")\n";
    }

    // 5. Copying replies at once.
    if (concurrency > 1) {
        std::vector<ninfer::GenerationSubmission> submissions;
        for (std::uint32_t i = 0; i < concurrency; ++i) {
            submissions.push_back(ninfer::GenerationSubmission{
                .prompt  = engine.prepare(conversation(copy_prompt)),
                .options = request(tokens, kTags, 1.0F, 100U + i)});
        }
        auto handles        = engine.submit_group(std::move(submissions));
        std::uint32_t clean = 0;
        for (auto& handle : handles) {
            const auto result = handle.wait();
            const auto found  = holds(result, kTags);
            check(!found, "a copying reply of a batch holds " + found.value_or(""));
            clean += found ? 0U : 1U;
        }
        std::cout << "copying replies at once: " << clean << " of " << concurrency
                  << " hold no banned string\n";
    }
    return g_failures == 0 ? 0 : 1;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
