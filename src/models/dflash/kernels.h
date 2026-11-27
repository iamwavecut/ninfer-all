#pragma once

// The DFlash drafter's own small kernels around its candidates (models/dflash/drafter.h): the
// head's soft cap on the candidates' scores, a DFlash row's best candidate with its confidence, and
// the constant rows the Markov chain's selector reads.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::models::dflash::kernels {

// scores[i] = cap * tanh(scores[i] / cap) over the FP32 tensor's elements (cap > 0).
void softcap(Tensor& scores, float cap, cudaStream_t stream);

// Per column u of the descending candidates (ids I32 [16, U], scores FP32 [16, U], as
// ops::candidate_topk writes them): drafts[u] = ids[0, u]; confidence[u] the first candidate's
// probability under the softmax of the 16 scores at temperature 1 (confidence may be empty);
// proposal_q [16, U] the one-hot law of that greedy draft. A column whose first score is not
// finite drafts its id with confidence 0.
void best_candidate(const Tensor& ids, const Tensor& scores, Tensor& drafts, Tensor& confidence,
                    Tensor& proposal_q, cudaStream_t stream);

// Every element of the contiguous BF16 tensor equals `value`.
void fill_bf16(Tensor& out, float value, cudaStream_t stream);

} // namespace ninfer::models::dflash::kernels
