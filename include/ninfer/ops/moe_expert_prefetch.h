#pragma once

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * The experts a MoE layer's router is likely to choose, read into the device's L2 cache before the
 * router has run. A decode step's expert products stream their weights from DRAM once each; when
 * the next layer's router, applied to this layer's MoE input, predicts most of the next layer's
 * choices, reading the predicted experts while the next layer's attention runs (its kernels leave
 * most of the DRAM bandwidth unused) lets the expert products read those bytes from L2. No value
 * changes: the reads are discarded, and a wrong guess costs only DRAM bandwidth.
 *
 * moe_guess_experts writes, for each of `rows` rows of FP32 `logits` ([rows][experts]), its
 * `per_row` best experts into ids[row * per_row + j] in decreasing order: by logit, or by
 * sigmoid(logit) + bias[e] when `bias` is given (the selection of a sigmoid router with a
 * selection bias); ties go to the lower index. experts <= 512, per_row <= 16, rows <= 8.
 */
void moe_guess_experts(const float* logits, const float* bias, int rows, int experts, int per_row,
                       std::int32_t* ids, cudaStream_t stream);

/**
 * moe_expert_prefetch reads into L2 the weights of each distinct expert among ids[0, count)
 * (count <= 128; an id outside [0, experts) is skipped): for each projection j < projections,
 * expert x's bytes start at tables[j][x] (device arrays of `experts` pointers; a null pointer is
 * skipped) and span bytes[j]. Only whole 16-byte words inside each span are read, so no read
 * leaves the expert's own bytes. One launch of one block per multiprocessor.
 */
struct ExpertPrefetch {
    static constexpr int kMaxProjections       = 3;
    const void* const* tables[kMaxProjections] = {};
    std::int64_t bytes[kMaxProjections]        = {};
    int projections                            = 0;
    int experts                                = 0;
    const std::int32_t* ids                    = nullptr;
    int count                                  = 0;
};

void moe_expert_prefetch(const ExpertPrefetch& args, cudaStream_t stream);

/**
 * How many experts of `expert_bytes` each a decode step may read ahead on this device: a quarter
 * of its L2 cache, at most 16; 0 when that is fewer than 2 experts (a card whose L2 holds too few
 * experts gains nothing and loses DRAM bandwidth to the reads). More evicts the bytes the step
 * itself streams: Inkling 2 Mini's 4.5 MB experts, 4 on an RTX 4090 (72 MiB L2) and 5-6 on an RTX
 * 5090 (96 MiB) measured fastest, 8 slower than none on the 4090.
 */
[[nodiscard]] int moe_prefetch_budget(int device, std::size_t expert_bytes);

} // namespace ninfer::ops
