// The paging queue's admission policy arithmetic (kv_queue_policy.h): the generation grant per
// reserve mode, the extension target of a decoding lane, and the Retry-After seconds of a queue
// timeout. The queue changes must not alter outputs — the grant arithmetic is what locks the
// sizes the executor is told to reserve.

#include "runtime/engine/kv_queue_policy.h"

#include <cstdint>

#include <iostream>

namespace kvq = ninfer::runtime::kv_queue;

int failures = 0;

void expect(bool ok, const char* what) {
    std::cout << (ok ? "OK   " : "FAIL ") << what << '\n';
    failures += ok ? 0 : 1;
}

int main() {
    // Admission grants: upfront reserves the whole generation; chunked reserves prompt + the
    // first chunk, never more than the generation needs, and a zero chunk means full.
    expect(kvq::generation_grant_tokens(1000, 4096, 0) == 5096, "upfront: prompt + whole output");
    expect(kvq::generation_grant_tokens(1000, 4096, 256) == 1256,
           "chunked: prompt + first chunk");
    expect(kvq::generation_grant_tokens(1000, 128, 256) == 1128,
           "chunk capped at the generation's length");
    expect(kvq::generation_grant_tokens(1000, 4096, 0) ==
               kvq::generation_grant_tokens(1000, 4096, 4096),
           "a chunk the size of the output equals upfront");

    // The extension target: the in-flight token plus the budget still available, capped at the
    // chunk; the final stretch reserves through completion.
    expect(kvq::extension_target_tokens(2000, 3000, 256) == 2257,
           "extension: position + 1 + chunk while the budget outlasts it");
    expect(kvq::extension_target_tokens(2000, 100, 256) == 2101,
           "the last chunk reserves through completion");
    expect(kvq::extension_target_tokens(2000, 0, 256) == 2001, "a spent budget reserves one");

    // Retry-After: whole seconds of the remainder, clamped to at least one.
    expect(kvq::retry_after_seconds(0) == 1, "an expired deadline answers Retry-After: 1");
    expect(kvq::retry_after_seconds(1) == 1, "a millisecond of remainder clamps to 1");
    expect(kvq::retry_after_seconds(1000) == 1, "exactly one second");
    expect(kvq::retry_after_seconds(1500) == 2, "a remainder rounds up");
    expect(kvq::retry_after_seconds(65000) == 65, "whole seconds pass through");

    std::cout << (failures == 0 ? "PASS  queue policy\n" : "FAIL  queue policy\n");
    return failures == 0 ? 0 : 1;
}
