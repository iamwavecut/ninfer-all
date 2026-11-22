#pragma once

// Session slots (Engine::save_slot, restore_slot, erase_slot, slot_states) of the families with
// an Engine core of their own: slot i is the core's sequence i, its retained session what the
// sequence holds once no request runs on it. A save copies the sequence's state into the context
// cache (context_image_store.h) and returns the session's images as records, its checkpoint (the
// sequence's anchor, where a next turn that re-renders the last reply resumes) first and its end
// last, which the Engine writes to the slot file; a restore adopts the records into the cache and
// loads the last into the sequence, after what the sequence held goes to the cache; an erase resets
// the sequence. The worker runs each
// between its steps (WorkerTasks), where nothing else uses the sequence; file I/O stays with the
// caller.
//
// Each core's Impl supplies: slots (each with request, fed, anchor and last_used, and media where
// the family reads media), stats_mutex, use_clock, store, instance.executor, retire(s),
// resume_stored(s, tokens, media) (the sequence resumes from a stored image, as a request would)
// and clear_session(s) (the sequence holds nothing).

#include "ninfer/observability/snapshot.h"
#include "ninfer/request.h"
#include "runtime/engine/context_image_store.h"
#include "runtime/engine/session_slot_record.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// Work a caller hands the worker, run between its steps.
class WorkerTasks {
public:
    // Queues `task` under `mutex` (the worker's queue mutex), wakes the worker and waits for the
    // result, rethrowing what the task throws. A stopped worker refuses it.
    template <class F>
    std::invoke_result_t<F&> run(std::mutex& mutex, std::condition_variable& wake, F&& task) {
        using Result = std::invoke_result_t<F&>;
        auto job     = std::make_shared<std::packaged_task<Result()>>(std::forward<F>(task));
        auto result  = job->get_future();
        {
            std::lock_guard lock(mutex);
            if (closed_) {
                throw RequestError(RequestErrorKind::Unavailable, "inference engine is stopping");
            }
            queue_.emplace_back([job] { (*job)(); });
        }
        wake.notify_all();
        return result.get();
    }

    // Under the worker's queue mutex.
    [[nodiscard]] bool pending() const noexcept { return !queue_.empty(); }

    // Runs the queued tasks on the worker; `lock` holds the queue mutex, released while each runs.
    void run_queued(std::unique_lock<std::mutex>& lock) {
        while (!queue_.empty()) {
            auto task = std::move(queue_.front());
            queue_.pop_front();
            lock.unlock();
            task();
            lock.lock();
        }
    }

    // The stopping worker runs what is queued and refuses what comes after.
    void close(std::unique_lock<std::mutex>& lock) {
        closed_ = true;
        run_queued(lock);
    }

private:
    std::deque<std::function<void()>> queue_;
    bool closed_ = false;
};

// A core's session slots: the worker's tasks and the states it publishes for slot_states().
struct SessionSlots {
    WorkerTasks tasks;
    std::vector<SlotState> published; // under the core's stats mutex

    // A retained session's digests, kept while its sequence's last request and length stay.
    struct Digests {
        std::uint64_t used = ~std::uint64_t(0);
        std::size_t tokens = 0;
        std::string session;
        std::vector<SlotCheckpoint> checkpoints;
    };
    std::vector<Digests> digests;
};

namespace session_slots {

template <class Slot>
[[nodiscard]] std::vector<ContextMediaMark> media_of(const Slot& slot) {
    if constexpr (requires { slot.media; }) {
        return slot.media;
    } else {
        return {};
    }
}

template <class Slot>
[[nodiscard]] std::vector<ContextMediaMark> anchor_media_of(const Slot& slot) {
    if constexpr (requires { slot.anchor_media; }) {
        return slot.anchor_media;
    } else {
        return {};
    }
}

template <class Impl>
void require_settled(const Impl& impl, std::uint32_t slot) {
    if (slot >= impl.slots.size()) {
        throw std::invalid_argument("slot " + std::to_string(slot) + " is outside the Engine's " +
                                    std::to_string(impl.slots.size()) + " slots");
    }
    if (impl.slots[slot].request) {
        throw RequestError(RequestErrorKind::Overloaded, "slot is processing a request");
    }
}

template <class Impl>
void require_digest(const Impl& impl, std::uint32_t slot, std::string_view expected) {
    if (expected.empty()) { return; }
    const auto& fed = impl.slots[slot].fed;
    if ((fed.empty() ? std::string() : session_digest(fed)) != expected) {
        throw SlotSessionMismatch("slot session does not match if_digest");
    }
}

} // namespace session_slots

// Publishes the slots' states for slot_states(); on the worker, after a step.
template <class Impl>
void publish_slot_states(Impl& impl) {
    SessionSlots& session = impl.session;
    session.digests.resize(impl.slots.size());
    std::vector<SlotState> out(impl.slots.size());
    for (std::size_t s = 0; s < impl.slots.size(); ++s) {
        const auto& slot = impl.slots[s];
        SlotState& state = out[s];
        if (slot.request) {
            state.processing    = true;
            state.prompt_tokens = static_cast<std::uint32_t>(slot.request->prompt_tokens.size());
            state.cached_tokens = slot.request->reused;
            continue;
        }
        if (slot.fed.empty()) { continue; }
        SessionSlots::Digests& digests = session.digests[s];
        if (digests.used != slot.last_used || digests.tokens != slot.fed.size()) {
            digests.used    = slot.last_used;
            digests.tokens  = slot.fed.size();
            digests.session = session_digest(slot.fed);
            digests.checkpoints.clear();
            const auto& anchor = slot.anchor;
            if (!anchor.empty() && anchor.size() < slot.fed.size() &&
                std::equal(anchor.begin(), anchor.end(), slot.fed.begin())) {
                digests.checkpoints.push_back(
                    SlotCheckpoint{.frontier       = static_cast<std::uint32_t>(anchor.size()),
                                   .session_digest = session_digest(anchor)});
            }
        }
        state.retained       = true;
        state.prompt_tokens  = static_cast<std::uint32_t>(slot.fed.size());
        state.cached_tokens  = state.prompt_tokens;
        state.session_digest = digests.session;
        state.checkpoints    = digests.checkpoints;
    }
    std::lock_guard lock(impl.stats_mutex);
    session.published = std::move(out);
}

// The record of sequence `slot`'s retained session; on the worker.
template <class Impl>
SessionSlotRecord save_slot_record(Impl& impl, std::uint32_t slot, std::string_view expected) {
    session_slots::require_settled(impl, slot);
    const auto& held = impl.slots[slot];
    if (held.fed.empty()) { throw std::invalid_argument("slot holds no retained session"); }
    session_slots::require_digest(impl, slot, expected);
    if (!impl.store.enabled()) {
        throw std::invalid_argument("session slots require the context cache and a host budget");
    }
    auto& executor = *impl.instance.executor;
    if (executor.position(slot) != held.fed.size()) {
        throw std::invalid_argument("the slot's state holds other tokens than its session");
    }
    // The session's images in the store: its checkpoint (the anchor) and its end.
    impl.retire(slot);
    const std::vector<ContextMediaMark> media = session_slots::media_of(held);
    (void)impl.store.put(held.fed, media, PrefixReusePath::PrivateEndpoint, executor.position(slot),
                         [&](std::span<std::byte* const> pages, std::size_t page_bytes) {
                             return executor.save_image(slot, nullptr, pages, page_bytes);
                         });
    SessionSlotRecord out;
    const auto& anchor = held.anchor;
    if (!anchor.empty() && anchor.size() < held.fed.size() &&
        std::equal(anchor.begin(), anchor.end(), held.fed.begin())) {
        const std::vector<ContextMediaMark> anchor_media = session_slots::anchor_media_of(held);
        if (impl.store.index(anchor, anchor_media)) {
            out.bytes = impl.store.record(anchor, anchor_media);
        }
    }
    const std::vector<std::uint8_t> end = impl.store.record(held.fed, media);
    out.bytes.insert(out.bytes.end(), end.begin(), end.end());
    out.tokens         = static_cast<std::uint32_t>(held.fed.size());
    out.session_digest = session_digest(held.fed);
    return out;
}

// Loads a saved record into sequence `slot`, what it held going to the cache first; on the
// worker. Returns the session's tokens and digest.
template <class Impl>
std::pair<std::uint32_t, std::string> restore_slot_record(Impl& impl, std::uint32_t slot,
                                                          std::span<const std::uint8_t> record) {
    session_slots::require_settled(impl, slot);
    if (!impl.store.enabled()) {
        throw std::invalid_argument("session restore requires the context cache to be enabled");
    }
    impl.retire(slot);
    std::vector<TokenId> tokens;
    std::vector<ContextMediaMark> media;
    try {
        const std::size_t at = impl.store.adopt(record);
        tokens               = impl.store.images()[at].tokens;
        media                = impl.store.images()[at].media;
        if (!impl.resume_stored(slot, tokens, media)) {
            throw std::invalid_argument("the context cache could not load the session");
        }
    } catch (...) {
        // A failed restore leaves the slot empty.
        impl.clear_session(slot);
        publish_slot_states(impl);
        throw;
    }
    impl.slots[slot].last_used = ++impl.use_clock;
    publish_slot_states(impl);
    return {static_cast<std::uint32_t>(tokens.size()), session_digest(tokens)};
}

// Resets sequence `slot`; on the worker. Returns the tokens its session held.
template <class Impl>
std::uint32_t erase_slot_session(Impl& impl, std::uint32_t slot, std::string_view expected) {
    session_slots::require_settled(impl, slot);
    session_slots::require_digest(impl, slot, expected);
    const auto tokens = static_cast<std::uint32_t>(impl.slots[slot].fed.size());
    impl.clear_session(slot);
    publish_slot_states(impl);
    return tokens;
}

} // namespace ninfer::runtime
