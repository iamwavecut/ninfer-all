#pragma once

#include "ninfer/request.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace ninfer::serve {

// The running chat replies a client may ask to answer now (POST /v1/answer_now), by the response's
// request id: `x-request-id` on OpenAI paths, `request-id` on Anthropic's. A reply is enrolled for
// as long as its enrollment lives; the registry keeps no reply alive.
class AnswerNowRegistry {
public:
    enum class Outcome : std::uint8_t {
        Accepted,  // the reply closes its reasoning at the end of its next round in reasoning
        Answering, // the reply is past its reasoning or does not reason
        Unknown,   // no running reply has this id
    };

    // Enrolls `signal` under `id` until the returned enrollment is destroyed. An empty id or signal
    // enrolls nothing.
    [[nodiscard]] std::shared_ptr<void> enroll(std::string id,
                                               std::shared_ptr<ninfer::AnswerNowSignal> signal);

    [[nodiscard]] Outcome request(std::string_view id);

    [[nodiscard]] std::size_t size() const;

private:
    struct Shared {
        mutable std::mutex mutex;
        std::unordered_map<std::string, std::weak_ptr<ninfer::AnswerNowSignal>> replies;
    };
    std::shared_ptr<Shared> shared_ = std::make_shared<Shared>();
};

} // namespace ninfer::serve
