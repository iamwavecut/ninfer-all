#pragma once
#include "artifact/materializer.h"
#include "core/device.h"
#include "core/startup.h"
#include "models/llama/config.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/model.h"
#include "models/qwen3_5/weights.h"
#include "ninfer/ops/weight_input.h"

namespace ninfer::artifact {
class Reader;
}

namespace ninfer::models::llama {
using qwen3_5::BoundWeight;
using qwen3_5::FrontendResources;
using qwen3_5::InstanceInfo;
using qwen3_5::WeightId;

struct AttentionWeights {
    WeightId query, key, value, output;
};

struct LayerWeights {
    WeightId input_norm, ffn_norm, mlp_gate, mlp_up, mlp_down;
    AttentionWeights attention;
};

struct TextWeights {
    WeightId token_embedding, output_head, final_norm;
    std::vector<LayerWeights> layers;
};

struct LoadOptions {
    std::filesystem::path artifact;
};

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;

    const TextConfig& config() const noexcept { return config_; }

    const TextWeights& weights() const noexcept { return weights_; }

    const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }

    std::span<const BoundWeight> weight_data() const noexcept { return bound_; }

    ops::WeightInput input(WeightId id) const;

    const FrontendResources& resources() const noexcept { return resources_; }

    const InstanceInfo& info() const noexcept { return info_; }

    const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }
private:
    friend std::unique_ptr<Model> load_model(const artifact::Reader&, const LoadOptions&,
                                             DeviceContext&, const StartupObserver*);
    Model(TextConfig, TextWeights, std::vector<BoundWeight>, FrontendResources, InstanceInfo,
          artifact::MaterializedArtifact);
    artifact::MaterializedArtifact backing_;
    TextConfig config_;
    TextWeights weights_;
    std::vector<BoundWeight> bound_;
    FrontendResources resources_;
    InstanceInfo info_;
};

bool is_llama(const artifact::Reader&);
std::unique_ptr<Model> load_model(const artifact::Reader&, const LoadOptions&, DeviceContext&,
                                  const StartupObserver* = nullptr);
} // namespace ninfer::models::llama
