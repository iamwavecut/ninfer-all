#include "models/llama/model.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/load/bindings.h"

#include <array>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::llama {
namespace {

using qwen3_5::loading::Bindings;

// Zero bytes after a matrix whose rows are not whole 256-value steps: the integer matrix kernel
// reads past the last row's end (gguf_project, moe_matrix_product).
constexpr std::uint64_t kRowTail = 512;

FrontendResources bind_resources(artifact::Binder& binder, const TextConfig& config) {
    const auto resource = [&](std::string_view role) {
        const auto bytes = binder.host_object(binder.resource("text", role));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    FrontendResources out;
    out.tokenizer_json         = resource("tokenizer.json");
    out.tokenizer_config_json  = resource("tokenizer_config.json");
    out.chat_template_jinja    = resource("chat_template.jinja");
    out.generation_config_json = resource("generation_config.json");
    out.tokenizer =
        std::make_shared<const qwen3_5::frontend::Tokenizer>(qwen3_5::frontend::TokenizerResources{
            out.tokenizer_json, out.tokenizer_config_json, out.generation_config_json});
    const auto count = out.tokenizer->vocab_size();
    if (!count || count > config.vocab_size || !out.tokenizer->has_exact_token_domain(count)) {
        throw artifact::ArtifactError(
            "tokenizer must expose a contiguous public domain within embedding rows");
    }
    out.public_token_count = static_cast<std::uint32_t>(count);
    return out;
}

void tail(Bindings& b, WeightId id) {
    for (const auto& part : b.at(id).reference.binding.parts) {
        if (is_gguf(b.binder.geometry(part.object).format)) {
            b.binder.device_tail(part.object, kRowTail);
        }
    }
}

} // namespace

Model::Model(TextConfig config, TextWeights weights, std::vector<BoundWeight> bound,
             FrontendResources resources, InstanceInfo info, artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), weights_(std::move(weights)),
      bound_(std::move(bound)), resources_(std::move(resources)), info_(std::move(info)) {}

Model::~Model() = default;

ops::WeightInput Model::input(WeightId id) const {
    const auto& parameter = weight(id);
    if (parameter.uses.size() != 1) {
        throw std::invalid_argument(parameter.name + ": a weight input needs exactly one use");
    }
    const auto& use = parameter.uses.front();
    return ops::WeightInput{parameter.view, use.policy, use.activation_input_divisor};
}

bool is_llama(const artifact::Reader& reader) {
    const auto& components = reader.directory().components;
    const auto text        = components.find("text");
    if (text == components.end()) { return false; }
    const auto& config = text->second.config;
    return config.is_object() && config.contains("architectures") &&
           config.at("architectures").is_array() && config.at("architectures").size() == 1 &&
           config.at("architectures")[0] == "LlamaForCausalLM";
}

std::unique_ptr<Model> load_model(const artifact::Reader& reader, const LoadOptions& options,
                                  DeviceContext& device, const StartupObserver* observer) {
    if (!is_llama(reader)) {
        throw artifact::ArtifactError("the artifact's text component is not LlamaForCausalLM");
    }
    TextConfig config = parse_text_config(reader.directory().component("text").config);
    artifact::Binder binder(reader);
    FrontendResources resources = bind_resources(binder, config);
    Bindings b(binder);
    if (device.size() != 1) throw std::invalid_argument("Llama requires one device");
    const std::uint64_t h = config.hidden_size;
    TextWeights weights;
    weights.token_embedding = b.parameter("text/token_embedding", {config.vocab_size, h});
    weights.output_head =
        b.parameter("text/output_head", {config.vocab_size, h}, {"text/final_hidden"});
    weights.final_norm     = b.direct("text/final_norm", {h}, QType::FP32);
    const std::uint64_t d  = config.head_dim;
    const std::uint64_t kv = config.num_key_value_heads;
    const std::uint64_t q  = std::uint64_t(config.num_attention_heads) * d;
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const std::string p = "text/layers/" + std::to_string(i) + "/";
        const std::string a = p + "attention/";
        LayerWeights layer;
        layer.input_norm       = b.direct(p + "input_norm", {h}, QType::FP32);
        layer.ffn_norm         = b.direct(p + "ffn_norm", {h}, QType::FP32);
        layer.attention.query  = b.parameter(a + "query", {q, h}, {p + "mixer_input"});
        layer.attention.key    = b.parameter(a + "key", {kv * d, h}, {p + "mixer_input"});
        layer.attention.value  = b.parameter(a + "value", {kv * d, h}, {p + "mixer_input"});
        layer.attention.output = b.parameter(a + "output", {h, q}, {p + "attention_output"});
        tail(b, layer.attention.output);
        const std::uint64_t width = config.intermediate_size;
        layer.mlp_gate            = b.parameter(p + "mlp/gate", {width, h}, {p + "mlp_input"});
        layer.mlp_up              = b.parameter(p + "mlp/up", {width, h}, {p + "mlp_input"});
        layer.mlp_down            = b.parameter(p + "mlp/down", {h, width}, {p + "mlp_product"});
        for (auto id : {layer.attention.query, layer.attention.key, layer.attention.value,
                        layer.mlp_gate, layer.mlp_up, layer.mlp_down})
            tail(b, id);
        weights.layers.push_back(std::move(layer));
    }
    InstanceInfo info;
    info.name            = reader.directory().metadata.value("name", std::string("Llama"));
    info.metadata_json   = reader.directory().metadata.dump();
    info.provenance_json = reader.directory().provenance.dump();
    info.artifact_id     = reader.artifact_id();
    auto pending         = std::move(b.weights);
    auto materialization = std::move(binder).finish();
    auto backing = artifact::materialize(reader, std::move(materialization), device, observer);
    auto bound   = qwen3_5::loading::resolve_weights(std::move(pending), backing);
    auto model   = std::unique_ptr<Model>(new Model(std::move(config), std::move(weights),
                                                    std::move(bound), std::move(resources),
                                                    std::move(info), std::move(backing)));
    return model;
}

} // namespace ninfer::models::llama
