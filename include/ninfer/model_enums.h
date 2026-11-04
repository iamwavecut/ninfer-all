#pragma once

// Engine-wide choices of purpose, KV capacity, speculative drafting and weight residency that
// the Engine options, the statistics and the model loaders share.

#include <cstdint>

namespace ninfer {

enum class EnginePurpose : std::uint8_t {
    Generation,
    CausalScoring,
};

enum class KvCapacityMode : std::uint8_t {
    Explicit,
    Automatic,
};

enum class ProposalHead : std::uint8_t {
    // The artifact's optimized proposal head when it stores one, the full output head otherwise;
    // resolved when the model loads.
    Auto,
    Full,
    Optimized,
};

enum class SpeculativeBackend : std::uint8_t {
    None,
    Mtp,
    DFlash,
    DFlash2,
    DSpark, // DeepSeek-V4.1's DSpark drafter
};

// How many of its drafts an MTP round verifies: always all of them, or a width the Engine adapts
// to the drafts' measured survival and the measured round cost. Greedy output does not depend on
// it.
enum class MtpDraftPolicy : std::uint8_t {
    Fixed,
    Adaptive,
};

enum class VisionResidency : std::uint8_t {
    Resident, // fixed Vision GPU allocations for the process lifetime
    Overlay,  // tower host-pinned; each image borrows device memory inside a bounded window
    Cpu,      // tower decoded to FP32 host memory and run on CPU threads; no device Vision memory
};

// Qwen3.5 family: where the input token embedding lives. The table is only read as a gather of
// one row per token, so it can stay in host RAM that the device reads in place.
enum class EmbeddingResidency : std::uint8_t {
    // In the primary device's weight arena.
    Device,
    // In page-locked host RAM mapped into the primary device's address space: the gather reads
    // each step's rows over the bus, and the table's device memory goes to the KV cache.
    Host,
};

// Where supported MoE families keep routed expert banks (see each family guide).
enum class ExpertResidency : std::uint8_t {
    // In the weight arena of each layer's stage device.
    Device,
    // In host RAM: experts the device cache lacks are copied to the GPU or computed on the CPU
    // (HybridExpertOptions).
    Host,
    // Left in the artifact's files and copied into a device cache when a pass routes to them,
    // through the OS page cache: the least host memory, at the cost of a wait for every expert a
    // pass needs that the cache does not hold.
    Disk,
};

// Qwen3.5-core tensor parallelism: how the devices divide a full-attention layer.
enum class TensorAttention : std::uint8_t {
    // Each device holds a run of the KV heads with their cache and query heads; devices past the
    // KV-head count hold no attention.
    Heads,
    // Each device holds every KV head of every page p with p % devices == its index, projects every
    // head, attends over its own keys, and merges its query heads' partial outputs with the
    // others' before its share of the output projection.
    Positions,
};

// Tensor parallelism over a Qwen3.5-core model or Qwen3.8-Flash-Next: how the devices exchange
// their partial sums.
enum class TensorParallelTransport : std::uint8_t {
    // Each exchange size takes whichever of the others, with the reduce's form, was fastest when
    // the engine timed them on these devices at startup.
    Auto,
    // Through pinned host memory: the copy engines write each partial there and every device's
    // kernels read the others' copies. Works between any cards.
    Host,
    // Every device's kernels read the other devices' memory (peer access).
    PeerRead,
    // Every device writes what another device sums into that device's memory, and reads only its
    // own (peer access).
    PeerWrite,
};

// Tensor parallelism: how a prompt chunk's partial sums cross between the devices.
enum class TensorParallelPartials : std::uint8_t {
    // As the devices computed them, in FP32.
    Fp32,
    // Each device's partial rounded to BF16 first, which moves half their bytes; the sum differs
    // from the FP32 partials' by that rounding.
    Bf16,
    // Each device's partial rounded to 8-bit blocks first (32 values as one FP32 scale and 32
    // signed codes), which moves 9/32 of their bytes; the sum differs from the FP32 partials' by
    // that rounding, which is coarser than BF16's.
    Int8,
};

// Qwen3.8-Flash-Next: where the n-gram table's rows come from.
enum class NgramResidency : std::uint8_t {
    // The table's file, with a bounded cache of rows read as the hash addresses them.
    Disk,
    // RAM: the whole table is read at startup.
    Ram,
    // RAM for the rows a hot-row profile ranks most frequent, as many as the RAM budget holds,
    // read at startup; the other rows come from the file.
    RamHot,
};

// Qwen3.8-Flash-Next: how rows are read from the table's file.
enum class NgramIo : std::uint8_t {
    // Positioned reads through the OS page cache.
    Buffered,
    // Reads that bypass the page cache (O_DIRECT, FILE_FLAG_NO_BUFFERING): the table takes no
    // RAM beyond what the residency keeps, and every cache miss reaches the drive.
    Direct,
    // Copies out of a read-only mapping of the file, faulted in through the page cache.
    Mapped,
};

} // namespace ninfer
