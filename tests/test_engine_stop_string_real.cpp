#include "guarded_main.h"
#include "ninfer/engine.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// A turn that a stop string ends, on a real model through the public Engine, for any model family:
// the next turn of the conversation resumes from the state where the stop string's match began,
// not from the end of the prompt before it, and answers as it does after a turn that an output
// limit ended at that position.
//   NINFER_STOP_ARTIFACT       the artifact (skipped without one)
//   NINFER_STOP_NGRAM_TABLE    a Qwen3.8-Flash-Next table artifact
//   NINFER_STOP_SPECULATIVE    mtp | dflash | dflash2 | dspark | dspine | lookup: the reply decodes
//                              in speculative rounds (NINFER_STOP_DRAFTS tokens, 4 by default)
//   NINFER_STOP_DRAFT_MODEL    the DSpine drafter artifact (with dspine)
//   NINFER_STOP_EXPERTS        host | disk: where the routed experts stay
//   NINFER_STOP_CONTEXT_CACHE  hybrid: the hybrid prefix cache
//   NINFER_STOP_KV             int8: an int8 KV cache
//   NINFER_STOP_DEVICES        device indices, comma separated
//   NINFER_STOP_TENSOR_PARALLEL 1: the devices share every layer (tensor parallelism) in
//                              place of pipeline stages
//   NINFER_STOP_THINKING       1: thinking stays as the template has it; on: thinking is on
//                              (a template that renders a reply without thinking differently in
//                              the history)
//   NINFER_STOP_CONTINUE       1: the next request continues the stopped reply as the final
//                              assistant message, in place of a new user turn
//   NINFER_STOP_PROMPT         the first user message, in place of the built-in one
//   NINFER_STOP_TOKENS         the turn's output limit, 240 by default (a model that reasons
//                              before it answers needs more)
//   NINFER_STOP_REASONING      1: the next request's history carries the reply's reasoning
//   NINFER_STOP_RAW            1: the next request is raw tokens, the prompt's and the reply's
//                              before the match, then new text (what a raw completion client
//                              sends; it repeats the reply's tokens under any template)
//   NINFER_STOP_BYTES          the stop string's bytes, 14 by default (more than a round of
//                              drafts holds, so its match spans rounds)
// The next prompt repeats the turn's tokens up to some position, which the test measures; a
// family whose template renders the reply anew repeats none of it, and its reuse is only reported.
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

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path         = artifact;
    options.max_context           = 4096;
    options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.max_concurrency       = 1;
    options.max_pending_requests  = 1;
    options.context_cache.enabled = true;
    if (const auto devices = variable("NINFER_STOP_DEVICES"); !devices.empty()) {
        const std::string list(devices);
        for (std::size_t begin = 0; begin < list.size();) {
            const auto end = list.find(',', begin);
            options.devices.push_back(std::stoi(list.substr(begin, end - begin)));
            if (end == std::string::npos) { break; }
            begin = end + 1;
        }
    }
    if (variable("NINFER_STOP_TENSOR_PARALLEL") == "1") { options.tensor_parallel = true; }
    if (const auto table = variable("NINFER_STOP_NGRAM_TABLE"); !table.empty()) {
        options.ngram_table.path = std::string(table);
    }
    if (variable("NINFER_STOP_CONTEXT_CACHE") == "hybrid") {
        options.context_cache.mode                    = ninfer::ContextCacheMode::Hybrid;
        options.context_cache.host_cache_budget_bytes = 1ULL << 30;
    }
    if (variable("NINFER_STOP_KV") == "int8") {
        options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
    }
    if (const auto experts = variable("NINFER_STOP_EXPERTS"); !experts.empty()) {
        options.expert_residency =
            experts == "disk" ? ninfer::ExpertResidency::Disk : ninfer::ExpertResidency::Host;
    }
    if (const auto backend = variable("NINFER_STOP_SPECULATIVE"); !backend.empty()) {
        const auto drafts = variable("NINFER_STOP_DRAFTS");
        const std::uint32_t count =
            drafts.empty() ? 4U : static_cast<std::uint32_t>(std::stoul(std::string(drafts)));
        if (backend == "lookup") {
            options.speculative.lookup_ngram = 3;
        } else {
            options.speculative.backend = backend == "mtp"       ? ninfer::SpeculativeBackend::Mtp
                                          : backend == "dflash"  ? ninfer::SpeculativeBackend::DFlash
                                          : backend == "dflash2" ? ninfer::SpeculativeBackend::DFlash2
                                          : backend == "dspine"  ? ninfer::SpeculativeBackend::DSpine
                                                                 : ninfer::SpeculativeBackend::DSpark;
            options.speculative.draft_model = std::string(variable("NINFER_STOP_DRAFT_MODEL"));
        }
        options.speculative.draft_tokens = count;
    }
    return options;
}

ninfer::ChatMessage message(ninfer::ChatRole role, std::string text) {
    return ninfer::ChatMessage{
        .role  = role,
        .parts = {ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text,
                                      .text = std::move(text)}}};
}

ninfer::PromptInput conversation(std::vector<ninfer::ChatMessage> messages,
                                 bool continue_reply = false) {
    ninfer::PromptInput input;
    input.messages = std::move(messages);
    if (continue_reply) {
        input.options.continuation = ninfer::PromptContinuationMode::ContinueFinalAssistant;
    }
    if (const auto thinking = variable("NINFER_STOP_THINKING"); thinking == "on") {
        input.options.enable_thinking = true;
    } else if (thinking != "1") {
        input.options.enable_thinking = false;
    }
    return input;
}

ninfer::RequestOptions request(std::uint32_t count) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = count;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = true;
    return options;
}

int run() {
    const char* artifact = std::getenv("NINFER_STOP_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cerr << "SKIP: NINFER_STOP_ARTIFACT is not set\n";
        return 77;
    }
    const ninfer::EngineOptions options = engine_options(artifact);
    const bool continues                = variable("NINFER_STOP_CONTINUE") == "1";
    const auto asked_for                = variable("NINFER_STOP_PROMPT");
    const ninfer::ChatMessage user      = message(
        ninfer::ChatRole::User,
        asked_for.empty() ? std::string("Tell a long story about a lighthouse keeper who repairs "
                                        "clocks. Use many short sentences.")
                          : std::string(asked_for));
    const ninfer::ChatMessage next =
        message(ninfer::ChatRole::User, "Continue the story with what happened at dawn.");

    const auto asked_tokens   = variable("NINFER_STOP_TOKENS");
    const std::uint32_t limit = asked_tokens.empty()
                                    ? 240U
                                    : static_cast<std::uint32_t>(std::stoul(std::string(asked_tokens)));
    const bool with_reasoning = variable("NINFER_STOP_REASONING") == "1";
    const bool raw            = variable("NINFER_STOP_RAW") == "1";

    std::string whole;
    std::vector<ninfer::TokenId> whole_ids;
    {
        ninfer::Engine engine(options);
        auto reply = engine.generate(engine.prepare(conversation({user})), request(limit));
        whole      = std::move(reply.content);
        whole_ids  = std::move(reply.generated_token_ids);
    }
    // A stop string of several tokens, taken from the reply itself where it first occurs: late
    // enough that the reply before it is long, early enough that the reply reaches it. It begins at
    // a space after a word where the reply has one: a match that begins inside a word leaves a
    // fragment of it in the visible text, which the next prompt's tokenizer joins to the token
    // before it (Qwen3.8-27B's "E" and the "l" of "lias" become "El"), so no state the turn kept
    // continues that prompt.
    const auto asked_bytes       = variable("NINFER_STOP_BYTES");
    const std::size_t stop_bytes = asked_bytes.empty() ? 14U : std::stoul(std::string(asked_bytes));
    std::optional<std::size_t> cut;
    for (const bool after_word : {true, false}) {
        for (std::size_t at = whole.size() * 3 / 4; !cut && at + stop_bytes <= whole.size(); ++at) {
            const std::string_view candidate = std::string_view(whole).substr(at, stop_bytes);
            const bool ascii = std::all_of(candidate.begin(), candidate.end(), [](char byte) {
                return static_cast<unsigned char>(byte) < 0x80U;
            });
            const bool word_ends = at > 0 && whole[at] == ' ' &&
                                   std::isalnum(static_cast<unsigned char>(whole[at - 1])) != 0;
            if (ascii && (word_ends || !after_word) && whole.find(candidate) == at) { cut = at; }
        }
        if (cut) { break; }
    }
    check(cut.has_value(), "the reply holds a usable stop string");
    if (!cut) { return 1; }

    // One Engine at a time: a large model leaves no room for a second.
    ninfer::RequestOptions stopped = request(limit);
    stopped.stop.strings.push_back(ninfer::StopString{.text = whole.substr(*cut, stop_bytes)});
    ninfer::GenerationResult first, second, reference;
    std::vector<ninfer::ChatMessage> turns;
    std::vector<ninfer::TokenId> read, asked, raw_next;
    // The next request: the conversation's next turn, or with NINFER_STOP_RAW the raw tokens of
    // the prompt and of the reply before the match, which every family's sequence continues.
    const auto next_prompt = [&](const ninfer::Engine& engine) {
        return raw ? engine.prepare_tokens(raw_next) : engine.prepare(conversation(turns, continues));
    };
    {
        ninfer::Engine engine(options);
        auto prompt = engine.prepare(conversation({user}));
        read.assign(prompt.token_ids().begin(), prompt.token_ids().end());
        first = engine.generate(std::move(prompt), stopped);
        turns = {user, message(ninfer::ChatRole::Assistant, first.content)};
        if (with_reasoning) { turns.back().reasoning_content = first.reasoning; }
        if (!continues) { turns.push_back(next); }
        if (raw) {
            // The reply's tokens whose bytes end at or before the cut (the content drops the
            // reply's leading whitespace), then a few tokens of new text.
            raw_next = read;
            std::string text;
            std::size_t lead = std::string::npos;
            for (const ninfer::TokenId id : first.generated_token_ids) {
                text += engine.token_bytes(id);
                if (lead == std::string::npos) {
                    const std::size_t at = text.find_first_not_of(" \t\r\n");
                    if (at == std::string::npos) {
                        raw_next.push_back(id);
                        continue;
                    }
                    lead = at;
                }
                if (text.size() - lead > *cut) { break; }
                raw_next.push_back(id);
            }
            const auto more = engine.tokenize_text(" Then the wind rose over the water.", false);
            raw_next.insert(raw_next.end(), more.begin(), more.end());
        }
        auto continued = next_prompt(engine);
        asked.assign(continued.token_ids().begin(), continued.token_ids().end());
        second = engine.generate(std::move(continued), request(48));
    }
    // How far the next prompt repeats what the turn read and produced.
    read.insert(read.end(), first.generated_token_ids.begin(), first.generated_token_ids.end());
    const auto repeated = static_cast<std::uint32_t>(
        std::mismatch(read.begin(), read.end(), asked.begin(), asked.end()).first - read.begin());
    const bool resumes = repeated >= first.prompt.prompt_tokens + 100U;
    {
        ninfer::Engine fresh(options);
        reference = fresh.generate(next_prompt(fresh), request(48));
    }

    const auto generated = static_cast<std::uint32_t>(first.generated_token_ids.size());
    std::cout << "stop string: prompt=" << first.prompt.prompt_tokens << " reply=" << generated
              << " tokens, next turn reused=" << second.reused_prompt_tokens << " of "
              << second.prompt.prompt_tokens << ", which repeats " << repeated
              << " of the turn's tokens\n";
    check(first.finish_reason == ninfer::FinishReason::StopString,
          "the first turn ends at the stop string");
    // A stop string changes no token: the turn's tokens are those of the turn without it.
    const bool same_tokens =
        first.generated_token_ids.size() <= whole_ids.size() &&
        std::equal(first.generated_token_ids.begin(), first.generated_token_ids.end(),
                   whole_ids.begin());
    check(same_tokens && first.content == whole.substr(0, *cut),
          "the turn with the stop string produces the tokens of the turn without it");
    if (!same_tokens) {
        const auto at = std::mismatch(first.generated_token_ids.begin(),
                                      first.generated_token_ids.end(), whole_ids.begin(),
                                      whole_ids.end())
                            .first -
                        first.generated_token_ids.begin();
        std::cout << "stop string: the tokens differ from token " << at << " of "
                  << first.generated_token_ids.size() << '\n';
    }
    check(generated >= 100, "the reply before the stop string is long enough to tell a resume");
    if (first.finish_reason != ninfer::FinishReason::StopString) { return 1; }
    check(continues || raw || second.reused_prompt_tokens < second.prompt.prompt_tokens,
          "the next turn computes its new tokens");
    if (resumes) {
        // The repeated tokens end inside the match or at the cut: the state before the token that
        // begins the match is a few tokens short of them.
        check(second.reused_prompt_tokens + 24U >= repeated,
              "the next turn resumes where the stop string's match began");
    } else {
        std::cout << "stop string: the next prompt renders the reply anew, so no state of the "
                     "reply continues it\n";
    }
    // The kept state is the state of a turn that ended there. A turn limited to n tokens reads
    // n - 1 of them, so the turn limited to the reply's tokens before the match and the token
    // that begins it leaves its ordinary end state at the same position, and the next turn
    // answers from either with the same tokens. Without drafts the two turns take the same steps,
    // so the comparison is exact; rounds of drafts end differently near an output limit.
    std::uint32_t before = 0;
    ninfer::GenerationResult limited_next;
    bool compared = false;
    for (std::uint32_t count = generated; count >= 1 && !compared; --count) {
        ninfer::Engine limited(options);
        const auto prefix = limited.generate(limited.prepare(conversation({user})), request(count));
        if (prefix.content.size() <= *cut) {
            before   = count;
            compared = count < generated;
            break;
        }
        limited_next = limited.generate(next_prompt(limited), request(48));
    }
    check(compared, "a turn limited to the tokens before the match ends before the cut");
    if (compared) {
        std::cout << "stop string: " << before << " tokens before the match, the next turn of a "
                  << "turn limited to " << before + 1 << " reused="
                  << limited_next.reused_prompt_tokens << '\n';
        if (resumes && variable("NINFER_STOP_SPECULATIVE").empty()) {
            check(limited_next.reused_prompt_tokens == second.reused_prompt_tokens &&
                      limited_next.generated_token_ids == second.generated_token_ids,
                  "the kept state is the state of a turn that ended where the match began");
        }
    }
    const auto print = [](const char* name, const std::vector<ninfer::TokenId>& ids) {
        std::cout << name;
        for (const ninfer::TokenId id : ids) { std::cout << ' ' << id; }
        std::cout << '\n';
    };
    if (second.generated_token_ids != reference.generated_token_ids ||
        (compared && limited_next.generated_token_ids != second.generated_token_ids)) {
        print("resumed: ", second.generated_token_ids);
        print("limited: ", limited_next.generated_token_ids);
        print("computed:", reference.generated_token_ids);
    }
    // A prompt computed whole takes other chunk shapes than one resumed mid-way, so its low bits
    // differ and a near tie may fall the other way: reported, not required.
    if (second.generated_token_ids != reference.generated_token_ids) {
        std::cout << "stop string: the resumed turn's tokens differ from a prompt computed whole\n";
    }
    return g_failures == 0 ? 0 : 1;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
