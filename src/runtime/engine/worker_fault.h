#pragma once

// Verification seam for the worker's failure recovery (engine-architecture §7.4). A host-side
// worker failure is a defect, and once a defect is fixed no public input reaches it any more, so
// the recovery contract cannot be exercised through the public API alone. Arming N failures makes
// the next N prefill units throw after the Program has executed them, so each recovery releases a
// lane that holds live KV pages and state. The injected failure is an invariant failure
// (std::logic_error), which the worker recovers from under EngineOptions::recover_invariant_failures
// and latches on otherwise. Disarmed, the cost is one relaxed atomic load per prefill unit.

#include <atomic>
#include <cstdint>
#include <stdexcept>

namespace ninfer::runtime {

inline std::atomic<std::uint32_t> armed_worker_failures{0};

inline void arm_worker_failures(std::uint32_t count) noexcept {
    armed_worker_failures.store(count, std::memory_order_relaxed);
}

// Worker-only: throws once per armed failure.
inline void consume_armed_worker_failure() {
    std::uint32_t armed = armed_worker_failures.load(std::memory_order_relaxed);
    while (armed != 0 && !armed_worker_failures.compare_exchange_weak(
                             armed, armed - 1U, std::memory_order_relaxed)) {}
    if (armed != 0) { throw std::logic_error("injected worker failure"); }
}

// The same seam for a numerical failure (non-finite logits), which no public input produces on a
// healthy model. A decode fault marks row 0 of the next decode round as the Program would mark a
// row whose sampled token left the vocabulary: that request ends with
// RequestErrorKind::NumericalFailure and the round's other rows commit. A prefill fault makes the
// next completed prompt throw runtime::NumericalFault after the Program executed it, as the Program
// does when a prompt's first token is non-finite: the Engine recovers.
inline std::atomic<std::uint32_t> armed_decode_numerical_faults{0};
inline std::atomic<std::uint32_t> armed_prefill_numerical_faults{0};

inline void arm_decode_numerical_faults(std::uint32_t count) noexcept {
    armed_decode_numerical_faults.store(count, std::memory_order_relaxed);
}

inline void arm_prefill_numerical_faults(std::uint32_t count) noexcept {
    armed_prefill_numerical_faults.store(count, std::memory_order_relaxed);
}

// Worker-only: true once per armed fault.
[[nodiscard]] inline bool consume_armed_numerical_fault(std::atomic<std::uint32_t>& armed_faults) noexcept {
    std::uint32_t armed = armed_faults.load(std::memory_order_relaxed);
    while (armed != 0 &&
           !armed_faults.compare_exchange_weak(armed, armed - 1U, std::memory_order_relaxed)) {}
    return armed != 0;
}

} // namespace ninfer::runtime
