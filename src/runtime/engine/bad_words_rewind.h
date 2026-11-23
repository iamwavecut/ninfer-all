#pragma once

// Banned words (bad_words) with rewinds: what the cores share when OutputDecision::bad_words_rewind
// takes a request back to where a banned word began.

#include "core/device_wait.h"
#include "ninfer/base.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::runtime {

// Takes the dropped `tokens` back out of a request's penalty counts (`counts`, one I32 per id of
// the vocabulary on the device), one for each occurrence. Waits for `stream` first: a rewind is
// rare, and its few tokens each cost two small copies.
inline void uncount_tokens(std::int32_t* counts, std::span<const TokenId> tokens,
                           cudaStream_t stream) {
    if (tokens.empty()) { return; }
    std::vector<TokenId> ids(tokens.begin(), tokens.end());
    std::sort(ids.begin(), ids.end());
    const auto check = [](cudaError_t status) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("banned-word rewind: ") +
                                     cudaGetErrorString(status));
        }
    };
    check(ninfer::wait_for_stream(stream));
    for (std::size_t i = 0; i < ids.size();) {
        std::size_t j = i;
        while (j < ids.size() && ids[j] == ids[i]) { ++j; }
        std::int32_t value = 0;
        check(ninfer::copy_and_wait(&value, counts + ids[i], sizeof(value),
                                    cudaMemcpyDeviceToHost));
        value -= static_cast<std::int32_t>(j - i);
        check(ninfer::copy_and_wait(counts + ids[i], &value, sizeof(value),
                                    cudaMemcpyHostToDevice));
        i = j;
    }
}

} // namespace ninfer::runtime
