#pragma once

// DeepSeek-V4.1-Flash (DeepseekV41ForCausalLM): the immutable model a v3 artifact describes.
// Matrices stay in the GGUF block formats the converter imported; vectors and the BF16 matrices
// of the router, compressor, indexer and hyper-connections are FP32. The routed experts sit on the
// device, in pinned host memory or in the artifact's files (ExpertResidency); the Engram table
// stays where the artifact (or a table artifact) stores it. The frontend resources are DeepSeek's
// tokenizer and chat template, which the shared frontend serves in its DeepSeek dialect. With
// LoadOptions::dspark the DSpark drafter's blocks bind too, all on the device; with
// LoadOptions::vision the vision tower, and the image routing bias the MoE routers select image
// positions with. With LoadOptions::dspine a standalone DSpine drafter artifact loads beside it.

#include "artifact/materializer.h"
#include "core/device.h"
#include "core/expert_delivery.h"
#include "core/startup.h"
#include "models/deepseek_v41/config.h"
#include "models/dspine/model.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/model.h"
#include "models/qwen3_5/weights.h"
#include "ninfer/ops/weight_input.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::artifact {
class Reader;
}

namespace ninfer::models::deepseek_v41 {

using qwen3_5::BoundWeight;
using qwen3_5::FrontendResources;
using qwen3_5::InstanceInfo;
using qwen3_5::WeightId;

// A hyper-connection site: the [mixes][hc * hidden] projection, its bias and its three scales.
struct HyperWeights {
    WeightId fn, base, scale;
};

struct AttentionWeights {
    WeightId query_a, kv, query_a_norm, kv_norm, query_b, output_a, output_b, sinks;
    // kv sources: the compressor and the indexer's key projection.
    std::optional<WeightId> compressor_kv, compressor_gate, compressor_norm, index_key,
        index_key_norm;
    // Index sources: the indexer's query projection and head weights.
    std::optional<WeightId> index_query, index_weights;
};

// The router, its selection bias, the shared expert, and the routed experts' banks when they are
// resident on the device (gate and up rows interleaved per expert).
struct MoeWeights {
    WeightId router, router_bias, shared_gate, shared_up, shared_down;
    std::optional<WeightId> gate_up, down;
};

struct EngramWeights {
    WeightId projection, gate;
};

struct LayerWeights {
    WeightId attention_norm, ffn_norm;
    HyperWeights hc_attention, hc_ffn;
    AttentionWeights attention;
    MoeWeights moe;
    std::optional<EngramWeights> engram;
};

struct TextWeights {
    WeightId token_embedding, output_head, final_norm;
    std::vector<LayerWeights> layers;
};

// An STQ matrix (ops/linear/stq): its trellis words (UINT16 [in/16][out/16][words]), input signs
// (FP16 [in]) and output signs (FP16 [out]); a set of matrices of one shape (the grouped output
// projection's groups, the routed experts) leads each with [count].
struct StqWeights {
    WeightId trellis, suh, svh;
};

struct StqLayerWeights {
    StqWeights query_a, kv, query_b, output_a, output_b, shared_gate, shared_up, shared_down;
    StqWeights gate, up, down; // the routed experts
};

// The drafter: the taps' projection and its norm, the final norm, the Markov tables ([vocab][rank]
// BF16: the embedding of the token before a row, and the bias head), the confidence head
// ([hidden + rank] FP32), and blocks whose routed experts are resident. An STQ drafter
// (DSparkConfig::stq) binds its matrices as stq_fc and stq_layers instead of fc and the blocks'
// matrix parameters.
struct DSparkWeights {
    WeightId fc, tap_norm, final_norm, markov_embed, markov_head, confidence;
    std::vector<LayerWeights> layers;
    std::optional<StqWeights> stq_fc;
    std::vector<StqLayerWeights> stq_layers;
};

// The vision tower: BF16 matrices (a block's gate and up one parent), FP32 vectors; the learned
// rows of an image span's start, row ends and end, and per backbone block the routing bias of
// image positions ([layers][experts]), FP32.
struct VisionLayerWeights {
    WeightId attention_norm, qkv, qkv_bias, output, output_bias, mlp_norm, gate, up, down;
};

struct VisionWeights {
    WeightId patch_embedding, patch_bias, final_norm;
    WeightId aligner_up, aligner_up_bias, aligner_out, aligner_out_bias;
    WeightId image_start, image_newline, image_end, image_router_bias;
    std::vector<VisionLayerWeights> layers;
};

// Where the tower's matrices live: on the device, or in pinned host memory, streamed block by
// block through a two-block device window as the tower runs. Its vectors, span rows and routing
// bias are on the device either way.
enum class VisionPlacement : std::uint8_t {
    None,
    Device,
    Pinned,
};

struct ExpertBanks {
    QType gate_up_type{}, down_type{};
    std::uint64_t gate_up_row_bytes = 0, down_row_bytes = 0;
    ExpertDeliveryLayer delivery;
};

struct LoadOptions {
    std::filesystem::path artifact;
    ExpertResidency experts = ExpertResidency::Device;
    bool dspark             = false; // bind the artifact's DSpark drafter
    // The drafter's routed experts each block keeps on the device (kDSparkExpertRanking's first),
    // the rest pinned on the host; 0 or the block's whole count keeps every one on the device.
    // Unset: kDefaultDraftExperts with the STQ drafter and host or disk experts, else every expert.
    std::optional<std::uint32_t> draft_experts;
    // Binds the artifact's `vision` component, which must then exist, in this placement.
    VisionPlacement vision = VisionPlacement::None;
    // Loads the standalone DSpine drafter artifact at this path (--spec dspine --draft-model) on
    // rank 0, the head's device; empty for none. It excludes the DSpark drafter.
    std::filesystem::path dspine;
};

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;

    [[nodiscard]] const TextConfig& config() const noexcept { return config_; }
    [[nodiscard]] const TextWeights& weights() const noexcept { return weights_; }
    // The drafter, when loaded.
    [[nodiscard]] const std::optional<DSparkConfig>& dspark_config() const noexcept {
        return dspark_config_;
    }
    [[nodiscard]] const std::optional<DSparkWeights>& dspark() const noexcept { return dspark_; }
    // LoadOptions::draft_experts below the block's count: each drafter block's kept experts in
    // ascending order, whose routed weights (pinned on the host) the executor copies to the
    // device; empty when every expert is on the device.
    [[nodiscard]] const std::vector<std::vector<std::int32_t>>& dspark_kept() const noexcept {
        return dspark_kept_;
    }
    // The vision tower, when loaded.
    [[nodiscard]] const std::optional<VisionConfig>& vision_config() const noexcept {
        return vision_config_;
    }
    [[nodiscard]] const std::optional<VisionWeights>& vision() const noexcept { return vision_; }
    [[nodiscard]] VisionPlacement vision_placement() const noexcept { return vision_placement_; }
    // The standalone DSpine drafter, when loaded (LoadOptions::dspine).
    [[nodiscard]] const dspine::Model* dspine() const noexcept { return dspine_.get(); }
    [[nodiscard]] ExpertResidency expert_residency() const noexcept { return expert_residency_; }
    // Host or disk experts: one bank per block, in block order.
    [[nodiscard]] std::span<const ExpertBanks> expert_banks() const noexcept { return expert_banks_; }
    [[nodiscard]] std::uint64_t expert_pinned_bytes() const noexcept {
        std::uint64_t bytes = 0;
        for (const auto& bank : expert_banks_) {
            for (const auto& expert : bank.delivery.experts) {
                for (const auto& blob : expert) { if (blob.pinned) { bytes += blob.bytes; } }
            }
        }
        return bytes;
    }
    [[nodiscard]] const std::vector<std::filesystem::path>& expert_files() const noexcept {
        return expert_files_;
    }
    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }
    [[nodiscard]] std::span<const BoundWeight> weight_data() const noexcept { return bound_; }
    [[nodiscard]] ops::WeightInput input(WeightId id) const;
    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }
    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }
    // The compressed id of every token, the Engram hash's alphabet.
    [[nodiscard]] const std::vector<std::int32_t>& engram_token_map() const noexcept {
        return engram_token_map_;
    }
    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

private:
    friend std::unique_ptr<Model> load_model(const artifact::Reader&, const LoadOptions&,
                                             DeviceContext&, const StartupObserver*);
    Model(TextConfig config, TextWeights weights, std::vector<BoundWeight> bound,
          FrontendResources resources, InstanceInfo info, artifact::MaterializedArtifact backing);

    artifact::MaterializedArtifact backing_;
    TextConfig config_;
    TextWeights weights_;
    std::optional<DSparkConfig> dspark_config_;
    std::optional<DSparkWeights> dspark_;
    std::vector<std::vector<std::int32_t>> dspark_kept_;
    std::optional<VisionConfig> vision_config_;
    std::optional<VisionWeights> vision_;
    VisionPlacement vision_placement_ = VisionPlacement::None;
    std::vector<BoundWeight> bound_;
    FrontendResources resources_;
    InstanceInfo info_;
    std::vector<std::int32_t> engram_token_map_;
    ExpertResidency expert_residency_ = ExpertResidency::Device;
    std::vector<ExpertBanks> expert_banks_;
    std::vector<std::filesystem::path> expert_files_;
    std::unique_ptr<dspine::Model> dspine_;
};

// Whether the artifact's text component is DeepSeek-V4.1, and whether it carries the DSpark drafter
// and the vision tower.
[[nodiscard]] bool is_deepseek_v41(const artifact::Reader& reader);
[[nodiscard]] bool has_dspark(const artifact::Reader& reader);
[[nodiscard]] bool has_vision(const artifact::Reader& reader);

[[nodiscard]] std::unique_ptr<Model> load_model(const artifact::Reader& reader,
                                                const LoadOptions& options, DeviceContext& device,
                                                const StartupObserver* observer = nullptr);

} // namespace ninfer::models::deepseek_v41
