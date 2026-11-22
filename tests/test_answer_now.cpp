#include "serve/answer_now.h"

#include <iostream>
#include <memory>

namespace {

using ninfer::AnswerNowSignal;
using ninfer::serve::AnswerNowRegistry;
using Outcome = AnswerNowRegistry::Outcome;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

// A request reaches the enrolled reply by its id, a reply past its reasoning refuses it, and an
// ended reply, an unknown id and an empty id find nothing.
int test_registry() {
    int failures = 0;
    AnswerNowRegistry registry;
    auto thinking  = std::make_shared<AnswerNowSignal>();
    auto answering = std::make_shared<AnswerNowSignal>();
    answering->mark_answering();
    auto first  = registry.enroll("chatcmpl-a", thinking);
    auto second = registry.enroll("req_b", answering);
    failures += check(!registry.enroll("", thinking) && !registry.enroll("x", nullptr),
                      "an empty id or signal was enrolled");
    failures += check(registry.request("chatcmpl-a") == Outcome::Accepted && thinking->requested(),
                      "a reply in reasoning refused answer now");
    failures += check(registry.request("chatcmpl-a") == Outcome::Accepted,
                      "a repeated request was not accepted");
    failures += check(registry.request("req_b") == Outcome::Answering && !answering->requested(),
                      "a reply past its reasoning accepted answer now");
    failures += check(registry.request("missing") == Outcome::Unknown &&
                          registry.request("") == Outcome::Unknown,
                      "an unknown id found a reply");
    first.reset();
    failures += check(registry.request("chatcmpl-a") == Outcome::Unknown && registry.size() == 1,
                      "an ended reply stayed enrolled");
    // A later reply under the same id keeps its entry when the earlier enrollment ends.
    auto older = registry.enroll("same", std::make_shared<AnswerNowSignal>());
    auto later = std::make_shared<AnswerNowSignal>();
    auto newer = registry.enroll("same", later);
    older.reset();
    failures += check(registry.request("same") == Outcome::Accepted && later->requested(),
                      "an earlier enrollment removed a later reply's entry");
    // An enrollment outliving its registry does nothing when it ends.
    std::shared_ptr<void> orphan;
    {
        AnswerNowRegistry scoped;
        orphan = scoped.enroll("orphan", std::make_shared<AnswerNowSignal>());
    }
    orphan.reset();
    return failures;
}

// The signal accepts a request until the reply answers, and keeps an accepted request.
int test_signal() {
    int failures = 0;
    AnswerNowSignal signal;
    failures += check(signal.state() == AnswerNowSignal::State::Waiting && !signal.requested(),
                      "a new signal was not waiting");
    failures += check(signal.request() && signal.requested(), "a waiting signal refused a request");
    signal.mark_answering();
    failures += check(signal.state() == AnswerNowSignal::State::Requested,
                      "answering discarded an accepted request");
    AnswerNowSignal late;
    late.mark_answering();
    failures +=
        check(!late.request() && !late.requested(), "an answering signal accepted a request");
    return failures;
}

} // namespace

int main() {
    const int failures = test_registry() + test_signal();
    if (failures != 0) { std::cerr << failures << " answer-now checks failed\n"; }
    return failures == 0 ? 0 : 1;
}
