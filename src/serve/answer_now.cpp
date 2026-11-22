#include "serve/answer_now.h"

#include <utility>

namespace ninfer::serve {

std::shared_ptr<void> AnswerNowRegistry::enroll(std::string id,
                                                std::shared_ptr<ninfer::AnswerNowSignal> signal) {
    if (id.empty() || !signal) { return {}; }
    {
        std::lock_guard lock(shared_->mutex);
        shared_->replies.insert_or_assign(id, signal);
    }
    // The enrollment removes its own entry only: a later reply enrolled under the same id keeps its
    // entry.
    struct Enrollment {
        Enrollment(std::weak_ptr<Shared> owner, std::string key,
                   std::weak_ptr<ninfer::AnswerNowSignal> reply)
            : shared(std::move(owner)), id(std::move(key)), signal(std::move(reply)) {}
        Enrollment(const Enrollment&)            = delete;
        Enrollment& operator=(const Enrollment&) = delete;

        std::weak_ptr<Shared> shared;
        std::string id;
        std::weak_ptr<ninfer::AnswerNowSignal> signal;

        ~Enrollment() {
            const std::shared_ptr<Shared> owner = shared.lock();
            if (!owner) { return; }
            std::lock_guard lock(owner->mutex);
            const auto found = owner->replies.find(id);
            if (found != owner->replies.end() && !found->second.owner_before(signal) &&
                !signal.owner_before(found->second)) {
                owner->replies.erase(found);
            }
        }
    };
    return std::make_shared<Enrollment>(shared_, std::move(id), signal);
}

AnswerNowRegistry::Outcome AnswerNowRegistry::request(std::string_view id) {
    std::shared_ptr<ninfer::AnswerNowSignal> signal;
    {
        std::lock_guard lock(shared_->mutex);
        const auto found = shared_->replies.find(std::string(id));
        if (found != shared_->replies.end()) { signal = found->second.lock(); }
    }
    if (!signal) { return Outcome::Unknown; }
    return signal->request() ? Outcome::Accepted : Outcome::Answering;
}

std::size_t AnswerNowRegistry::size() const {
    std::lock_guard lock(shared_->mutex);
    return shared_->replies.size();
}

} // namespace ninfer::serve
