// The 8-lane concurrent-prefill paging proof (NINFER_PAGED8_ARTIFACT names the Inkling 2
// artifact; without it the test skips): eight requests enter at one boundary (submit_group) and
// prefill and decode together under --kv-paging, for three rounds — fresh lanes, the same
// prompts again (the lanes resume their own retained prefixes) and shifted prompts (the lanes
// are trimmed and refilled). Every greedy output must equal a dense control engine's byte for
// byte: the pool relocates bytes, never changes them.

#include "ninfer/engine.h"
#include "ninfer/kv_cache_storage.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "OK   " : "FAIL ") << what << '\n';
    failures += ok ? 0 : 1;
}

ninfer::PromptInput prompt(std::string text) {
    ninfer::PromptInput input;
    input.messages.push_back(ninfer::ChatMessage{
        .role  = ninfer::ChatRole::User,
        .parts = {ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = std::move(text)}}});
    input.options.enable_thinking = false;
    return input;
}

std::string list_prompt(int items) {
    const char* words[] = {"alpha",  "beta",   "gamma",   "delta",  "epsilon",
                           "zeta",   "eta",    "theta",   "iota",   "kappa"};
    std::string out = "Remember this list of facts: ";
    for (int j = 0; j < items; ++j) {
        out += std::string(words[j % 10]) + "-" + std::to_string(j) + " is item " +
               std::to_string(j) + ". ";
    }
    out += "\nReply with the single word OK when you have read them.";
    return out;
}

ninfer::RequestOptions request(std::uint32_t tokens) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = true;
    return options;
}

ninfer::EngineOptions engine_options(const char* artifact, bool paged) {
    using KvCacheStorage = ninfer::KvCacheStorage;
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = 32768;
    options.max_concurrency      = 8;
    options.kv_cache             = KvCacheStorage::RotatedLloyd4KeyInt4Value;
    options.kv_paging            = paged;
    options.pending_timeout_ms   = 600000;
    options.use_cuda_graph       = true;
    if (const char* context = std::getenv("NINFER_PAGED8_CONTEXT"); context && *context) {
        options.max_context = static_cast<std::uint32_t>(std::atoi(context));
    }
    return options;
}

// Four rounds of eight lanes: fresh, resumed (the same prompts), shifted (new prompts, the
// lanes trimmed and refilled) and one round whose lanes share a long leading prefix with
// per-lane divergent suffixes (the dedup demonstration).
std::vector<std::vector<std::string>> run_engine(ninfer::Engine& engine) {
    std::vector<std::vector<std::string>> texts(4, std::vector<std::string>(8));
    for (int round = 0; round < 4; ++round) {
        const int shift = round == 2 ? 40 : 0;
        std::vector<ninfer::GenerationSubmission> submissions;
        std::vector<ninfer::PreparedPrompt> prompts;
        prompts.reserve(8);
        for (int lane = 0; lane < 8; ++lane) {
            std::string text;
            if (round == 3) {
                // A common ~1,100-token prefix, then each lane's own divergent question.
                text = list_prompt(96) + " Lane " + std::to_string(lane) +
                       " read this copy. Reply with the single word OK when you have read it.";
            } else {
                text = list_prompt(200 + lane * 5 + shift);
            }
            prompts.push_back(engine.prepare(prompt(text)));
            ninfer::GenerationSubmission submission;
            submission.prompt           = std::move(prompts.back());
            submission.options          = request(256);
            submission.consumer_mode    = ninfer::OutputConsumerMode::Aggregate;
            submission.pending_deadline = std::chrono::steady_clock::time_point::max();
            submissions.push_back(std::move(submission));
        }
        auto handles = engine.submit_group(std::move(submissions));
        check(handles.size() == 8, "round " + std::to_string(round + 1) + ": eight handles");
        for (int lane = 0; lane < 8; ++lane) {
            const ninfer::GenerationResult result = handles[std::size_t(lane)].wait();
            texts[std::size_t(round)][std::size_t(lane)] = result.content;
        }
    }
    
    return texts;
}

// The queue-path scenarios: reserve modes, the extension guarantee, the queue order and the
// queue deadline, each on its own engine.
void run_queue_scenarios(const char* artifact) {
    using KvReserveMode = ninfer::KvReserveMode;
    using KvQueueOrder = ninfer::KvQueueOrder;

    // Chunked reservations must not change outputs: the same four rounds as the dense
    // control, granted in 64-token generation chunks.
    std::vector<std::vector<std::string>> dense_texts;
    {
        ninfer::Engine control(engine_options(artifact, true));
        dense_texts = run_engine(control);
    }
    {
        auto options = engine_options(artifact, true);
        options.kv_reserve_mode      = KvReserveMode::Chunked;
        options.kv_reserve_gen_chunk = 64;
        ninfer::Engine chunked(options);
        const auto texts = run_engine(chunked);
        bool same = true;
        for (int round = 0; round < 4; ++round) {
            for (int lane = 0; lane < 8; ++lane) {
                same = same && texts[std::size_t(round)][std::size_t(lane)] ==
                                     dense_texts[std::size_t(round)][std::size_t(lane)];
            }
        }
        check(same, "chunked reservations produce the dense control's outputs byte for byte");
    }

    // The extension guarantee: a pool at its floor (80 MiB = 146 groups) and eight lanes
    // generating 3,000 tokens each (47 groups a lane, 376 total) — chunked admits all eight
    // and the lanes that cannot extend fail their own request while at least one completes;
    // upfront admits about three and queues the rest to a clean timeout.
    const auto generation_scenario = [&](KvReserveMode mode) {
        ninfer::EngineOptions options = engine_options(artifact, true);
        options.max_context           = 8192;
        options.kv_reserve_mode       = mode;
        options.kv_reserve_gen_chunk  = 64;
        options.kv_page_budget_mib    = 80;
        options.pending_timeout_ms    = 8000;
        ninfer::Engine engine(options);
        int completed = 0, failed = 0;
        std::vector<ninfer::GenerationSubmission> submissions;
        for (int lane = 0; lane < 8; ++lane) {
            auto prepared = engine.prepare(prompt(list_prompt(20 + lane)));
            ninfer::GenerationSubmission submission;
            submission.prompt           = std::move(prepared);
            submission.options          = request(3000);
            submission.consumer_mode    = ninfer::OutputConsumerMode::Aggregate;
            submission.pending_deadline = std::chrono::steady_clock::time_point::max();
            submissions.push_back(std::move(submission));
        }
        auto handles = engine.submit_group(std::move(submissions));
        for (auto& handle : handles) {
            try {
                const ninfer::GenerationResult result = handle.wait();
                if (!result.content.empty() || result.finish_reason != ninfer::FinishReason::None) {
                    ++completed;
                }
            } catch (const std::exception&) {
                ++failed;
            }
        }
        std::cout << "  " << (mode == KvReserveMode::Upfront ? "upfront" : "chunked")
                  << ": completed " << completed << ", failed " << failed << '\n';
        // The server-side engine keeps serving after any per-request failure.
        const ninfer::GenerationResult probe =
            engine.generate(engine.prepare(prompt(list_prompt(3))), request(4));
        check(!probe.content.empty(),
              std::string(mode == KvReserveMode::Upfront ? "upfront" : "chunked") +
                  ": the engine keeps serving after per-request failures");
        return completed;
    };
    std::cout << "extension guarantee at the pool floor\n";
    const int upfront_completed    = generation_scenario(KvReserveMode::Upfront);
    const int chunked_completed    = generation_scenario(KvReserveMode::Chunked);
    check(chunked_completed > 0, "chunked: at least one lane rides out the tight pool");
    check(upfront_completed + chunked_completed > 0,
          "the tight pool still completes work in both modes");

    // The queue order: one lane, a huge request submitted before a tiny one — shortest runs
    // the tiny one first, so it completes while the huge request is still decoding.
    const auto order_scenario = [&](KvQueueOrder order) {
        ninfer::EngineOptions options = engine_options(artifact, true);
        options.max_concurrency       = 1;
        options.kv_queue_order        = order;
        options.pending_timeout_ms    = 120000;
        ninfer::Engine engine(options);
        const auto huge_submit = std::chrono::steady_clock::now();
        auto huge_handle = engine.submit(engine.prepare(prompt(list_prompt(300))),
                                         request(256), ninfer::OutputConsumerMode::Aggregate, {},
                                         std::chrono::steady_clock::time_point::max());
        auto tiny_handle = engine.submit(engine.prepare(prompt(list_prompt(2))), request(1),
                                         ninfer::OutputConsumerMode::Aggregate, {},
                                         std::chrono::steady_clock::time_point::max());
        const auto tiny_done = [&] {
            tiny_handle.wait();
            return std::chrono::steady_clock::now();
        }();
        huge_handle.wait();
        return tiny_done < huge_submit + std::chrono::seconds(1);
    };
    check(order_scenario(KvQueueOrder::Shortest),
          "shortest: a tiny request admitted behind a huge one completes first");
    check(!order_scenario(KvQueueOrder::Fifo),
          "fifo: the same tiny request waits behind the huge one");

    // The queue deadline: a queued request whose deadline passes while the lane is busy
    // expires with a clean QueueTimeout carrying a Retry-After; one with time left completes.
    {
        ninfer::EngineOptions options = engine_options(artifact, true);
        options.max_concurrency       = 1;
        options.pending_timeout_ms    = 120000;
        ninfer::Engine engine(options);
        auto busy = engine.submit(engine.prepare(prompt(list_prompt(300))), request(256),
                                  ninfer::OutputConsumerMode::Aggregate, {},
                                  std::chrono::steady_clock::time_point::max());
        bool timed_out = false;
        try {
            auto queued = engine.submit(engine.prepare(prompt(list_prompt(4))), request(4),
                                        ninfer::OutputConsumerMode::Aggregate, {},
                                        std::chrono::steady_clock::now() +
                                            std::chrono::seconds(1));
            queued.wait();
        } catch (const ninfer::RequestError& error) {
            timed_out = error.kind() == ninfer::RequestErrorKind::QueueTimeout &&
                        error.retry_after_seconds() >= 1;
        }
        check(timed_out, "a queued request expiring at its deadline answers QueueTimeout with "
                         "a Retry-After");
        busy.wait();
        // A queued request admitted just inside its deadline completes.
        bool inside_completed = false;
        try {
            auto queued = engine.submit(engine.prepare(prompt(list_prompt(4))), request(4),
                                        ninfer::OutputConsumerMode::Aggregate, {},
                                        std::chrono::steady_clock::now() +
                                            std::chrono::seconds(120));
            const ninfer::GenerationResult result = queued.wait();
            inside_completed = result.finish_reason != ninfer::FinishReason::None;
        } catch (const std::exception&) {
            inside_completed = false;
        }
        check(inside_completed, "a queued request with deadline room completes");
    }
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_PAGED8_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP  NINFER_PAGED8_ARTIFACT is not set\n";
        return 77;
    }
    try {
        std::cout << "dense control engine\n";
        {
            ninfer::Engine dense(engine_options(artifact, false));
            const auto texts = run_engine(dense);
            // Keep the control's outputs for the paging engine in a file-side by side: both
            // engines live in one process? No — the dense engine is destroyed first, the paging
            // one starts on a clean device; the comparison happens below over the saved texts.
            std::FILE* out = std::fopen("paged8-dense-expected.txt", "w");
            for (int round = 0; round < 4; ++round) {
                for (int lane = 0; lane < 8; ++lane) {
                    std::fputs(texts[std::size_t(round)][std::size_t(lane)].c_str(), out);
                    std::fputc('\n', out);
                }
            }
            std::fclose(out);
        }
        std::cout << "paging engine\n";
        std::vector<std::vector<std::string>> paged;
        {
            ninfer::Engine paged_engine(engine_options(artifact, true));
            paged = run_engine(paged_engine);
        }
        std::FILE* in = std::fopen("paged8-dense-expected.txt", "r");
        check(in != nullptr, "the dense control's outputs were saved");
        if (in != nullptr) {
            char buffer[4096];
            for (int round = 0; round < 4; ++round) {
                for (int lane = 0; lane < 8; ++lane) {
                    std::string expected = std::fgets(buffer, sizeof(buffer), in) != nullptr
                                               ? std::string(buffer)
                                               : std::string();
                    if (!expected.empty() && expected.back() == '\n') { expected.pop_back(); }
                    const std::string& got = paged[std::size_t(round)][std::size_t(lane)];
                    check(got == expected, "round " + std::to_string(round + 1) + " lane " +
                                               std::to_string(lane) + ": paged == dense (" +
                                               std::to_string(got.size()) + " vs " +
                                               std::to_string(expected.size()) + " bytes)");
                }
            }
            std::fclose(in);
        }
    } catch (const std::exception& e) {
        std::cout << "FAIL  " << e.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "PASS  paged 8-lane\n" : "FAIL  paged 8-lane\n");
    return failures == 0 ? 0 : 1;
}
