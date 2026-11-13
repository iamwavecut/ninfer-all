#pragma once

// Qwen3.8-Flash-Next (Qwen4ExpForCausalLM): the immutable model a v3 artifact describes. The text
// tower's parameters are bound as the artifact stores them (GGUF blocks, BF16 or NInfer's own
// formats); the frontend resources are the Qwen3.5 tokenizer and chat template, which this family
// shares. Expert banks live on the device of their layer's pipeline stage, or in the pinned Host
// block where the GPU reads them across the bus. Under tensor parallelism every device holds a run
// of every layer: Gated DeltaNet heads, sparse-attention KV heads and routed experts, beside
// replicas of what each device computes for itself (LoadOptions::tensor_split).

#include "artifact/materializer.h"
#include "core/device.h"
#include "core/stage_plan.h"
#include "core/startup.h"
#include "core/tensor.h"
#include "models/qwen3_5/auxiliary_replicas.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/model.h"
#include "models/qwen3_5/weights.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/expert_location.h"
#include "ninfer/ops/weight_input.h"
#include "ninfer/model_enums.h"
#include "ninfer/startup.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::artifact {
class Reader;
}

namespace ninfer::models::qwen4_exp {

using qwen3_5::BoundWeight;
using qwen3_5::FrontendResources;
using qwen3_5::InstanceInfo;
using qwen3_5::WeightId;
using qwen3_5::WeightUseId;

struct HyperConnectionWeights {
    WeightId norm, down, up;
    std::optional<WeightId> inject; // absent for the final mixer
};

struct SparseAttentionWeights {
    WeightId query, gate, key, value, output, query_norm, key_norm;
    WeightId index_query, index_key, index_query_norm, index_key_norm;
};

struct GdnWeights {
    WeightId query, key, value, z, a_projection, b_projection, a_log, dt_bias;
    WeightId convolution, norm, output;
};

// A latent MoE's own operands (TextConfig::latent_moe): the groups' encoders [groups * width,
// hidden] and decoders [groups * hidden, width] as one parameter each, the experts' group labels,
// and the optional multipliers. Under tensor parallelism a rank's encoder and decoder hold the
// layer's groups [group_begin, group_end), else every group.
struct LatentWeights {
    WeightId encode, decode, labels;
    std::optional<WeightId> expert_scales, column_scale;
    std::uint32_t group_begin = 0, group_end = 0;
};

struct MoeWeights {
    WeightId router, shared_score;
    // Edge0's latent experts: then each expert's gate and up are [640, width], its down [width, 640].
    std::optional<LatentWeights> latent;
    // One per expert; empty when the experts stay in the files, which `located` then describes.
    std::vector<WeightId> gate, up, down;
    std::vector<ExpertLocation> located_gate, located_up, located_down;
    // The shared expert, or under tensor parallelism the rank's run of it (TensorShare).
    WeightId shared_gate, shared_up, shared_down;
};

struct PleWeights {
    WeightId key, value, norm_key, norm_query, norm_conv, convolution;
};

struct LayerWeights {
    HyperConnectionWeights attn_hc, mlp_hc;
    // std::monostate: a sparse-attention layer of a tensor-parallel rank that holds no KV head.
    std::variant<GdnWeights, SparseAttentionWeights, std::monostate> mixer;
    MoeWeights moe;
    std::optional<PleWeights> ple;
};

struct TextWeights {
    WeightId token_embedding, output_head;
    HyperConnectionWeights final_mixer;
    std::vector<LayerWeights> layers;
    // Tensor parallelism: what every rank after the first holds, [rank - 1]: its replica of the
    // token embedding and its part of each layer (its MoE weights are the router's and the shared
    // score's replicas and its run of the shared expert; `layers` lists every rank's routed
    // experts).
    std::vector<WeightId> rank_embeddings;
    std::vector<std::vector<LayerWeights>> rank_layers;
    // Tensor parallelism: rank r projects rows [head_rows[r], head_rows[r + 1]) of the head with
    // head_shares[r], for the text from its own stack, read through its final mixer
    // (rank_final_mixers[r - 1] after the first), and for the MTP block's drafts from rank 0's
    // read of the block's stack; rank 0 also holds `output_head` whole, for prompt passes.
    std::vector<std::uint64_t> head_rows;
    std::vector<WeightId> head_shares;
    std::vector<HyperConnectionWeights> rank_final_mixers;
};

// Tensor parallelism: the runs of every layer one rank holds. A rank holds the value heads of its
// Gated DeltaNet key heads and the 12 query heads of each of its sparse-attention KV heads; the
// ranks holding KV heads are a prefix of the ranks. Each rank also holds a run of the shared
// expert's 640-wide middle: those rows of its gate and up and those values of its down rows.
struct TensorShare {
    std::uint32_t key_head_begin = 0, key_head_end = 0;
    std::uint32_t kv_head_begin = 0, kv_head_end = 0;
    // A latent MoE's ranks hold whole projection groups, chosen in each layer by its labels
    // (LatentWeights::group_begin); expert_begin and expert_end then count the rank's share of
    // the experts, which only the shared expert's runs follow.
    std::uint32_t expert_begin = 0, expert_end = 0;
    std::uint32_t shared_begin = 0, shared_end = 0;
};

// The MTP block (vLLM Qwen4ExpMultiTokenPredictor): the two input norms and projections, one
// sparse-attention layer with its own MoE, and its final mixer. The token embedding and the head
// are the text model's.
struct MtpWeights {
    WeightId embedding_norm, hidden_norm, fc_embedding, fc_hidden;
    LayerWeights layer;
    HyperConnectionWeights final_mixer;
};

// The DFlash2 drafter (docs/maintainer/dflash.md): the feature projection with its context norm,
// the sliding-window layers (each with the query block's projections, the context's K/V uses of
// the same matrices, the q/k norms, the dense SwiGLU and a dynamic convolution around each branch)
// and the conditional selector. The token embedding and the head are the text model's.
struct DFlash2LayerWeights {
    WeightId input_norm, post_attention_norm;
    WeightId query, key, value, context_key, context_value, query_norm, key_norm, output;
    WeightId gate, up, down;
    WeightId attention_base, attention_projection, mlp_base, mlp_projection;
};

struct DFlash2Weights {
    WeightId feature_projection, context_norm, final_norm;
    WeightId selector_projection, predecessor_codebook, successor_codebook;
    std::vector<DFlash2LayerWeights> layers;
};

struct LoadOptions {
    // The artifact's entry file; its part files sit next to it.
    std::filesystem::path artifact;
    // Pipeline stages, one per device rank; `stage_layers` lists each one's layer count (empty:
    // balanced by the bytes each stage holds).
    std::size_t ranks = 1;
    std::vector<std::uint32_t> stage_layers;
    ExpertResidency experts = ExpertResidency::Device;
    // Loads the Vision tower the artifact carries (its `vision` component, the Qwen3.5 tower) onto
    // the first stage's device, beside the token embedding its output joins.
    bool vision = false;
    // Loads the MTP block the artifact carries (its `mtp` component) onto the last stage's device,
    // beside the head it shares; its experts take the text layers' residency.
    bool mtp = false;
    // Loads the DFlash2 drafter the artifact carries (its `dflash2` component) onto the device of
    // the head, which it reads, as it reads the token embedding.
    bool dflash2 = false;
    // Loads the standalone DSpine drafter artifact at this path (--draft-model) onto the device of
    // the head; empty for none. One device or tensor parallelism, as DFlash2.
    std::filesystem::path dspine;
    // Tensor parallelism (`--tensor-parallel`): each rank's relative share of every layer, one
    // positive weight per rank; empty for none. Each rank holds a run of every Gated DeltaNet
    // layer's heads, of the sparse-attention layers' KV heads (as far as the two allow) and of the
    // routed experts, and replicas of the token embedding, the hyper-connections, the routers and
    // the PLE layer, a run of each shared expert and of the head's rows; rank 0 also the MTP block
    // and the Vision tower. One stage; the experts must be device resident.
    std::vector<double> tensor_split;
};

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;

    [[nodiscard]] const TextConfig& config() const noexcept { return config_; }

    [[nodiscard]] const LoadOptions& options() const noexcept { return options_; }

    [[nodiscard]] const TextWeights& weights() const noexcept { return weights_; }

    [[nodiscard]] const StagePlan& stages() const noexcept { return stages_; }

    // The stage that holds the embedding (the first) and the one that holds the final mixer and
    // the output head (the last).
    [[nodiscard]] std::size_t head_rank() const noexcept { return stages_.stages() - 1; }

    // Tensor parallelism: whether every layer is divided over the ranks, a rank's runs of it, and
    // a rank's weights of layer `layer` (rank 0's are weights().layers) and its token embedding.
    [[nodiscard]] bool tensor_parallel() const noexcept { return !options_.tensor_split.empty(); }

    [[nodiscard]] const TensorShare& share(std::size_t rank) const { return shares_.at(rank); }

    // The ranks holding sparse-attention KV heads, a prefix.
    [[nodiscard]] std::size_t attention_ranks() const noexcept;

    [[nodiscard]] const LayerWeights& layer_weights(std::size_t rank, std::size_t layer) const;

    [[nodiscard]] WeightId embedding(std::size_t rank) const;

    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }

    [[nodiscard]] std::span<const BoundWeight> weight_data() const noexcept { return bound_; }

    // The weight's one mathematical use, as a projection operand, its auxiliaries on the weight's
    // device.
    [[nodiscard]] ops::WeightInput input(WeightId id) const;
    // The use of a weight read by several inputs (the head, which the MTP block reads too).
    [[nodiscard]] ops::WeightInput input(WeightId id, std::string_view use) const;

    // The MTP block when the model was loaded with it.
    [[nodiscard]] const std::optional<MtpConfig>& mtp_config() const noexcept {
        return mtp_config_;
    }

    [[nodiscard]] const std::optional<MtpWeights>& mtp_weights() const noexcept {
        return mtp_weights_;
    }

    // The DFlash2 drafter when the model was loaded with it.
    [[nodiscard]] const std::optional<DFlash2Config>& dflash2_config() const noexcept {
        return dflash2_config_;
    }

    [[nodiscard]] const std::optional<DFlash2Weights>& dflash2_weights() const noexcept {
        return dflash2_weights_;
    }

    // The standalone DSpine drafter when the model was loaded with one (LoadOptions::dspine).
    [[nodiscard]] const dspine::Model* dspine() const noexcept { return dspine_.get(); }

    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }

    // The artifact's files, in its order (disk-resident experts are read from them).
    [[nodiscard]] const std::vector<std::filesystem::path>& files() const noexcept {
        return files_;
    }

    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }

    // The Vision tower when the model was loaded with it.
    [[nodiscard]] const std::optional<qwen3_5::VisionConfig>& vision_config() const noexcept {
        return vision_config_;
    }
    [[nodiscard]] const std::optional<qwen3_5::VisionWeights>& vision_weights() const noexcept {
        return vision_weights_;
    }

    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

private:
    friend std::unique_ptr<Model> load_model(const artifact::Reader&, const LoadOptions&,
                                             DeviceContext&, const StartupObserver*);
    Model(TextConfig config, LoadOptions options, TextWeights weights, StagePlan stages,
          std::vector<BoundWeight> bound, FrontendResources resources, InstanceInfo info,
          artifact::MaterializedArtifact backing);

    // Destroy every borrower before the backing.
    artifact::MaterializedArtifact backing_;
    TextConfig config_;
    LoadOptions options_;
    TextWeights weights_;
    StagePlan stages_;
    std::vector<BoundWeight> bound_;
    FrontendResources resources_;
    InstanceInfo info_;
    std::vector<std::filesystem::path> files_;
    qwen3_5::AuxiliaryReplicas replicas_;
    std::optional<qwen3_5::VisionConfig> vision_config_;
    std::optional<qwen3_5::VisionWeights> vision_weights_;
    std::optional<MtpConfig> mtp_config_;
    std::optional<MtpWeights> mtp_weights_;
    std::optional<DFlash2Config> dflash2_config_;
    std::optional<DFlash2Weights> dflash2_weights_;
    std::unique_ptr<dspine::Model> dspine_;
    std::vector<TensorShare> shares_; // by rank under tensor parallelism
};

// Whether the artifact's text component is this family.
[[nodiscard]] bool is_qwen4_exp(const artifact::Reader& reader);

[[nodiscard]] std::unique_ptr<Model> load_model(const artifact::Reader& reader,
                                                const LoadOptions& options, DeviceContext& device,
                                                const StartupObserver* observer = nullptr);

// The routed experts [out[r], out[r + 1]) of `experts` each rank holds under `split`.
[[nodiscard]] std::vector<std::uint64_t> expert_ends(std::uint32_t experts,
                                                     std::span<const double> split);

// A tensor split for ranks with `free_bytes` of device memory and `speed`: in proportion to speed,
// except that a rank takes no more of the text layers' routed experts than it has room for once
// rank 0 holds everything else `options` loads (the stored bytes of the components it names) and
// every rank keeps a margin for state and workspace.
[[nodiscard]] std::vector<double> default_tensor_split(const artifact::Reader& reader,
                                                       const LoadOptions& options,
                                                       std::span<const std::uint64_t> free_bytes,
                                                       std::span<const double> speed);

} // namespace ninfer::models::qwen4_exp
