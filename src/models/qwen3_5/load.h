#pragma once

#include "models/qwen3_5/model.h"
#include "ninfer/kv_cache_storage.h"
#include "ninfer/startup.h"
#include "ninfer/types_fwd.h"
#include "ops/common/device_tier.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::artifact {
class Reader;
struct ParameterReference;
} // namespace ninfer::artifact

namespace ninfer::models::qwen3_5 {

// Cold load plan borrows its Reader until materialization. Its selected Host bytes and parsed
// resources already have owners; no Program or device allocation is needed for plan_load.
class LoadPlan {
public:
    ~LoadPlan();
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] const ModelWeights& weights() const;
    [[nodiscard]] const FrontendResources& resources() const;
    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;
    [[nodiscard]] const artifact::ParameterReference& parameter(WeightId id) const;
    [[nodiscard]] std::span<const WeightUse> uses(WeightId id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit LoadPlan(std::unique_ptr<Impl> impl);
    friend LoadPlan plan_load(const artifact::Reader&, LoadOptions, std::filesystem::path,
                              std::vector<ops::DeviceTier>);
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*,
                                                    const DiagnosticObserver*,
                                                    const artifact::MaterializeOptions&);
};

// `rank_tiers` holds the code tier of each pipeline rank's device (ops/common/device_tier.h), which
// decides the stored layout of the projection weights that rank reads; empty takes the current
// device's tier for every rank.
[[nodiscard]] LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options = {},
                                 std::filesystem::path artifact_path     = {},
                                 std::vector<ops::DeviceTier> rank_tiers = {});
// The code tier of each rank's device, in rank order.
[[nodiscard]] std::vector<ops::DeviceTier> rank_tiers(const DeviceContext& device);
// How many layers each of `free_bytes.size()` pipeline stages should own so that the most KV cache
// fits on every device at once, given the bytes free on each. Sized from the artifact's stored
// layers; the head stage (rank 0) also carries the embedding, head and round buffers, and as
// selected a resident Vision tower, MTP's layer and its KV, a DFlash drafter and the proposal head.
// The default when `--stage-layers` is not given.
//
// The per-device KV and recurrent-state costs use the same layouts startup planning reserves, so
// they follow the runtime choices that size them.
struct StageSizing {
    // Stored K/V format of the Paged KV cache.
    KvCacheStorage kv_storage = KvCacheStorage::BFloat16;
    // Recurrent-state slots each Linear Attention layer holds: concurrent requests plus the device
    // state slots the context cache keeps.
    std::uint32_t state_slots = 1;
    // Tensor parallelism: the tokens of KV cache rank 0 holds for the attention layers (the longest
    // context times the concurrent requests), and the fewest automatic KV capacity starts with
    // (the longest context).
    std::uint64_t kv_tokens         = 0;
    std::uint64_t minimum_kv_tokens = 0;
    // Automatic KV capacity: the KV shrinks to what fits beside `kv_headroom_bytes` left free on
    // every device.
    bool automatic_kv               = false;
    std::uint64_t kv_headroom_bytes = 0;
};
[[nodiscard]] std::vector<std::uint32_t> default_stage_layers(const artifact::Reader& reader,
                                                              LoadOptions options,
                                                              const StageSizing& sizing,
                                                              std::span<const std::uint64_t> free_bytes);
// The relative share of each of `speed.size()` tensor-parallel ranks that the first load takes
// when `--tensor-split` is not given: in proportion to `speed` (rank_copy_bandwidth), but no more
// of the divided weights, recurrent state and KV cache than fits beside what a rank holds whole
// (rank 0: the head stage's parts as for default_stage_layers, and MTP's KV) and the automatic KV
// headroom, sized from the artifact. The Engine's own reservation is not known before a load;
// refit_tensor_split moves what a device then lacks. When the longest context fits nowhere,
// automatic KV capacity divides by room (fill_shares), so the KV it shrinks to is the most that
// fits; otherwise the shares follow speed alone.
[[nodiscard]] std::vector<double> default_tensor_split(const artifact::Reader& reader,
                                                       LoadOptions options,
                                                       const StageSizing& sizing,
                                                       std::span<const std::uint64_t> free_bytes,
                                                       std::span<const double> speed);
// The split to load again after a load with `options.tensor_split`, or nullopt when that one
// stands. `spare` is what each rank kept free beyond the Engine's reservation for the KV cache
// asked for, negative where it fell short (runtime::kv_capacity_spare). Each rank may then take
// its share plus its spare, in bytes of the divided work per share; the shares follow `speed`
// within that room as in default_tensor_split, so a rank that fell short gives up only what it
// lacked. A candidate's divided weights, KV cache and GDN state are sized as plan_load places them
// on `rank_tiers`, and a rank's room shrinks until the candidate fits it. When no candidate fits,
// automatic KV capacity shrinks the KV until the rank short of the largest part of its own KV
// fits, so the candidate whose ranks lack the smallest such part is taken (the KV is then the most
// that fits), or nullopt when that gains less than a page or holds less than
// `sizing.minimum_kv_tokens`; an explicit capacity takes nullopt.
// nullopt also when the nearest split the weights allow is the loaded one.
[[nodiscard]] std::optional<std::vector<double>>
refit_tensor_split(const artifact::Reader& reader, const LoadOptions& options,
                   const StageSizing& sizing, std::span<const std::int64_t> spare,
                   std::span<const double> speed, const std::filesystem::path& artifact_path,
                   const std::vector<ops::DeviceTier>& rank_tiers);
// `materialize` makes the weight arenas suspendable when asked (see Model::release_device_weights).
[[nodiscard]] std::unique_ptr<Model>
materialize_model(LoadPlan&& plan, DeviceContext& device, const StartupObserver* observer = nullptr,
                  const DiagnosticObserver* diagnostics = nullptr,
                  const artifact::MaterializeOptions& materialize = {});
[[nodiscard]] std::unique_ptr<Model> load_model(const std::filesystem::path& path,
                                                LoadOptions options, DeviceContext& device,
                                                const StartupObserver* observer = nullptr,
                                                const DiagnosticObserver* diagnostics = nullptr);

} // namespace ninfer::models::qwen3_5
