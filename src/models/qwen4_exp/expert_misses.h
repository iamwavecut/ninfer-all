#pragma once

// The routed experts a layer's call needs but its device cache lacks, served while the cached ones
// run. After a layer's routing a kernel splits its pairs: the cached ones run at once from the
// device tables.
//
// A narrow call (up to kNarrowMissTokens tokens: decode steps, short verifications) posts its
// missing experts to pinned host memory (one "doorbell" per device) without any host
// synchronization, so the whole pass can replay as one CUDA graph. A host thread picks the request
// up at once while passes keep it spinning (keep_alive), within milliseconds once it sleeps, and,
// for each missing expert, either copies it into a device staging region (or, when the cache's
// expert tier admits it, straight into a slot) by the copy engine, reads it across the bus when
// the staging is full, or has the CPU compute its pairs from the bytes in RAM (host experts) or
// from the file (disk experts); an expert the CPU computes crosses no bus and takes no slot. The
// second stage of the layer waits for the request on the GPU, then runs the staged experts and adds
// the CPU's products. The staged experts are reached through a second table that only the second
// stage reads, so the resident stage needs no host at all. A narrow call takes the vector
// products, which the CPU's equal bit for bit.
//
// A wide call runs as over device banks (ops::moe_experts_gguf): its experts with at most
// ops::kGgufMoeVectorPairs pairs on the vector products, the others on the matrix kernel. The
// calling thread waits for the split, then admits or stages the missing experts in rounds of half
// the staging, each round's products running while the next round's copies cross the bus; of
// host experts with few pairs, the CPU computes a share meanwhile, up to kCpuMissTokens tokens
// (stage_wide). Every missing expert with more pairs reaches device memory.

#include "core/device.h"
#include "core/expert_disk.h"
#include "core/qtype.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/qwen4_exp/expert_cache.h"
#include "models/qwen4_exp/expert_location.h"
#include "ninfer/ops/moe_experts.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen4_exp {

struct MissLayer {
    std::size_t rank   = 0;
    std::size_t experts = 0;
    struct Projection {
        QType format              = QType::GGUF_Q2_0;
        std::int64_t row_bytes    = 0;
        std::int32_t rows         = 0;
        std::int32_t k            = 0;
        std::int64_t expert_bytes = 0;
        std::vector<const void*> host;           // host experts: each expert's bytes in RAM
        std::span<const ExpertLocation> located; // disk experts: where each expert's bytes are
    };
    std::array<Projection, 3> projections;   // gate, up, down
    const void* const* gate_table = nullptr; // the device table the cached stage reads
    // Device memory holding this layer's cached experts (ExpertCache::storage): a gate entry
    // inside any of its blocks is resident.
    ExpertCache::Ranges resident{};
    // Host experts whose layer the cache handed over: a missing expert the GPU copies anyway goes
    // straight into a slot when the cache's tier admits it (ExpertCache::admit), and stays.
    bool admit = false;
    // Edge0's latent experts (ops::moe_latent_gguf), `latent_width` > 0: the CPU reads each pair's
    // quantized latent input (the call's GgufLatentStage::inputs, given to begin() as `m`) instead
    // of its token's hidden column, multiplies gate and up by the expert's `latent_scales` (FP32
    // [experts][3] gate, up, down; empty: ones), and writes down products `latent_width` long,
    // before s_down (ops::moe_latent_gguf_add_products).
    std::int32_t latent_width = 0;
    std::vector<float> latent_scales;
};

// The widest narrow call: ops::moe_experts_gguf takes the vector products up to this width; past
// it, the matrix kernel (up to kCpuMissTokens, for experts with more than
// ops::kGgufMoeVectorPairs pairs), which the CPU's products do not reproduce.
inline constexpr std::uint32_t kNarrowMissTokens = 8;
// The widest call whose missing experts with few pairs the CPU may compute: the widest whose
// few-pair experts ops::moe_experts_gguf runs on the vector products.
inline constexpr std::uint32_t kCpuMissTokens = std::uint32_t(ops::kGgufMoeSplitTokens);

// How long keep_alive() and each served request keep the service and its CPU workers spinning.
inline constexpr std::chrono::milliseconds kMissKeepAlive{500};

struct MissOptions {
    std::uint32_t token_capacity = 8;    // tokens of one call, narrow or wide
    // Share of a call's missing experts the CPU computes: of a wide call's, of those with at most
    // ops::kGgufMoeVectorPairs pairs.
    double cpu_share             = 0.0;
    // Instead of a fixed share, each request gives the CPU the experts that make its copies and
    // its CPU products end soonest, by times fitted to the earlier requests.
    bool cpu_balance             = false;
    std::uint32_t cpu_threads    = 0;    // 0: one per physical core (pool_cpu_count())
    // Device staging per rank, two halves. Disk experts also take pinned file staging of two
    // halves, each the larger of a device half and a narrow call's every missing expert.
    std::uint64_t staging_bytes = 128ULL << 20;
    // 1..9: a call leaves out each missing expert that all its tokens rank past this many places
    // of their top ten (lossy); 0 serves every miss.
    std::uint32_t tail_skip_rank = 0;
    // Disk experts: a drive mirror and the direct path (core/expert_disk.h).
    ExpertDiskReads disk{};
};

struct MissStats {
    std::uint64_t requests    = 0; // layers that had missing experts
    std::uint64_t staged      = 0; // experts copied to device staging or straight into slots
    std::uint64_t mapped      = 0; // experts the GPU read across the bus (narrow, staging full)
    std::uint64_t computed    = 0; // experts the CPU computed
    std::uint64_t admitted    = 0; // of the staged, copied straight into cache slots
    std::uint64_t cpu_pairs   = 0;
    std::uint64_t staged_bytes = 0;
    double service_seconds     = 0; // host time from a request to its readiness, summed
};

class ExpertMisses {
public:
    // `cache` (may be null) holds the slots of the layers that admit; its layer indices are the
    // misses' own.
    ExpertMisses(DeviceContext& device, std::vector<MissLayer> layers,
                 std::vector<std::filesystem::path> files, MissOptions options, ExpertCache* cache);
    ~ExpertMisses();
    ExpertMisses(const ExpertMisses&)            = delete;
    ExpertMisses& operator=(const ExpertMisses&) = delete;

    // Device views a layer's call uses: the cached pairs' ids, then the missing pairs' ids, tables
    // and CPU products the host fills (a wide call's counted by stage_wide).
    struct Call {
        Tensor cached_ids;
        Tensor missing_ids;
        ops::GgufMoeWeights missing_banks;
        const float* products        = nullptr;
        const std::int32_t* pairs    = nullptr;
        const std::int32_t* count    = nullptr;
        std::int32_t max_products    = 0;
    };
    // Enqueued on `stream` (capturable when narrow): splits `ids` of `tokens` tokens and, for a
    // narrow call, posts the missing experts. `m` is the layer's BF16 input [2560, tokens], or a
    // latent layer's quantized pair inputs (U8, ops::moe_latent_gguf_input_bytes), copied to the
    // host when the CPU takes a share. A wide call's missing_banks are device-resident.
    [[nodiscard]] Call begin(std::size_t layer, const Tensor& ids, const Tensor& m,
                             std::int32_t tokens, const ops::GgufMoeWeights& banks,
                             cudaStream_t stream);
    // Narrow calls, enqueued: waits until the request's experts are staged and its CPU products
    // written. Eager (the stream not capturing), it waits on the host and wakes the service first:
    // the device may post the request long after begin() was enqueued.
    void await(std::size_t layer, cudaStream_t stream);
    // Wide calls, eager, after begin(): waits on the host for the split, then brings the missing
    // experts the GPU runs into device memory, straight into a cache slot when the tier admits it,
    // else into a half of the staging, a round of experts at a time, while CPU threads compute the
    // others (up to kCpuMissTokens tokens: a share of those with at most
    // ops::kGgufMoeVectorPairs pairs). `stage` enqueues on `stream` a round's products over
    // Call::missing_banks; `missing_ids` lists the round's pairs (-1 for the others). A round's
    // copies wait for the products of the round before it in the same half. Returns the pairs the
    // CPU computed, whose products Call::products holds once this returns (add them with
    // ops::moe_experts_gguf_add_products); 0 when none.
    [[nodiscard]] std::int32_t stage_wide(
        std::size_t layer, cudaStream_t stream,
        const std::function<void(const Tensor& missing_ids)>& stage);
    // Keeps the service (and the CPU workers) spinning for kMissKeepAlive; a pass calls it before
    // its first layer. Rethrows a service failure. Asleep, the service still looks for requests
    // every few milliseconds, as no host call wakes it for a request a replayed layer posts.
    void keep_alive();
    // Rethrows a failure the service recorded while serving a call. A failed call released the
    // GPU with its missing experts left out, so a pass checks this once it has completed, before
    // anything it computed is used.
    void raise_failure();
    // The layer's cached blocks changed (the cache grew); the device copy the kernels read follows.
    void set_resident(std::size_t layer, const ExpertCache::Ranges& resident);
    [[nodiscard]] MissStats stats() const;
    [[nodiscard]] bool cpu_enabled() const noexcept;
    // With a CPU share: its threads and vector backend, e.g. "15 pinned threads, avx2"; else
    // empty.
    [[nodiscard]] std::string cpu_description() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen4_exp
