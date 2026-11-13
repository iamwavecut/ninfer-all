#include "models/deepseek_v41/model.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "core/qtype.h"
#include "models/deepseek_v41/dspark_ranking.h"
#include "models/deepseek_v41/engram.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/load/bindings.h"

#include <algorithm>
#include <array>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::deepseek_v41 {
namespace {

using qwen3_5::loading::Bindings;

// Zero bytes after a matrix: the integer matrix kernel reads whole 256-value steps of a row
// (gguf_project, moe_matrix_product), so a stored row whose k is not a whole number of them reads
// past its end.
constexpr std::uint64_t kRowTail = 512;

FrontendResources bind_resources(artifact::Binder& binder, const TextConfig& config,
                                 std::vector<std::int32_t>& token_map, bool vision) {
    const auto resource = [&](std::string_view role, std::string_view component = "text") {
        const auto bytes = binder.host_object(binder.resource(component, role));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    FrontendResources out;
    out.tokenizer_json         = resource("tokenizer.json");
    out.tokenizer_config_json  = resource("tokenizer_config.json");
    out.chat_template_jinja    = resource("chat_template.jinja");
    out.generation_config_json = resource("generation_config.json");
    if (vision) { out.processor_config_json = resource("processor_config.json", "vision"); }
    out.tokenizer =
        std::make_shared<const qwen3_5::frontend::Tokenizer>(qwen3_5::frontend::TokenizerResources{
            out.tokenizer_json, out.tokenizer_config_json, out.generation_config_json});
    const auto count = out.tokenizer->vocab_size();
    if (!count || count > config.vocab_size || !out.tokenizer->has_exact_token_domain(count)) {
        throw artifact::ArtifactError(
            "tokenizer must expose a contiguous public domain within embedding rows");
    }
    out.public_token_count = static_cast<std::uint32_t>(count);
    const auto map = resource("engram_token_map");
    token_map      = decode_engram_token_map(
        std::as_bytes(std::span(map.data(), map.size())), config.vocab_size);
    return out;
}

void tail(Bindings& b, WeightId id) {
    for (const auto& part : b.at(id).reference.binding.parts) {
        if (is_gguf(b.binder.reader().geometry(part.object).format)) {
            b.binder.device_tail(part.object, kRowTail);
        }
    }
}

// Resolve stored GGUF rows directly. Disk mode validates without materializing the bank.
// Host mode borrows the materializer's pinned parent. Neither path repacks model weights.
std::vector<ExpertBlob> expert_blobs(const artifact::Reader& reader,
                                    const artifact::ParameterReference& reference,
                                    const artifact::MaterializedArtifact& backing,
                                    std::uint64_t experts, std::uint64_t rows, std::uint64_t k,
                                    QType& format, std::uint64_t& row_bytes) {
    std::vector<ExpertBlob> result;
    const auto& parts = reference.binding.parts;
    if (parts.size() != 1) {
        throw artifact::ArtifactError(reference.name + ": expert bank must be one stored region");
    }
    const auto& part = parts.front();
    const auto& geometry = reader.geometry(part.object);
    if (!is_gguf(geometry.format) || geometry.shape.size() != 2 || geometry.shape[1] != k ||
        part.begin % k || part.end - part.begin != experts * rows * k) {
        throw artifact::ArtifactError(reference.name + ": experts must be stored GGUF block rows");
    }
    format = geometry.format;
    const auto block = gguf_block_shape(format);
    if (k % block.elements) {
        throw artifact::ArtifactError(reference.name + ": partial stored quantization block");
    }
    row_bytes = k / block.elements * block.bytes;
    reader.validate_object(part.object);
    const auto start = part.begin / k * row_bytes;
    for (std::uint64_t e = 0; e < experts; ++e) {
        ExpertBlob blob;
        blob.bytes = rows * row_bytes;
        blob.zero_tail = kRowTail;
        const auto offset = start + e * blob.bytes;
        if (reference.residency == artifact::Residency::Pinned) {
            const auto* parent = backing.pinned_parent(part.object).data;
            blob.pinned = parent + offset;
            blob.pinned_region = reinterpret_cast<std::uintptr_t>(parent);
        } else {
            const auto logical = reader.directory().tensor(part.object).offset + offset;
            for (const auto& segment : reader.segments(logical, blob.bytes)) {
                blob.extents.push_back({segment.file_index, segment.file_offset, segment.bytes});
            }
        }
        result.push_back(std::move(blob));
    }
    return result;
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

bool has_dspark(const artifact::Reader& reader) {
    return reader.directory().components.contains("dspark");
}

bool has_vision(const artifact::Reader& reader) {
    return is_deepseek_v41(reader) && reader.directory().components.contains("vision");
}

bool is_deepseek_v41(const artifact::Reader& reader) {
    const auto& components = reader.directory().components;
    const auto text        = components.find("text");
    if (text == components.end()) { return false; }
    const auto& config = text->second.config;
    return config.is_object() && config.contains("architectures") &&
           config.at("architectures").is_array() && config.at("architectures").size() == 1 &&
           config.at("architectures")[0] == "DeepseekV41ForCausalLM";
}

std::unique_ptr<Model> load_model(const artifact::Reader& reader, const LoadOptions& options,
                                  DeviceContext& device, const StartupObserver* observer) {
    if (!is_deepseek_v41(reader)) {
        throw artifact::ArtifactError("the artifact's text component is not DeepseekV41ForCausalLM");
    }
    TextConfig config = parse_text_config(reader.directory().component("text").config);
    if (!options.dspine.empty()) {
        if (options.dspark) {
            throw std::invalid_argument("DeepSeek-V4.1 drafts with one drafter: its DSpark drafter "
                                        "or a DSpine drafter");
        }
        dspine::check_target(dspine::read_config(options.dspine), config.hidden_size,
                             config.vocab_size, config.num_hidden_layers, "DeepSeek-V4.1");
    }
    const bool vision = options.vision != VisionPlacement::None;
    if (vision && !has_vision(reader)) {
        throw artifact::ArtifactError("the artifact carries no vision tower (convert with "
                                      "--components text,dspark,vision)");
    }
    artifact::Binder binder(reader);
    std::vector<std::int32_t> token_map;
    FrontendResources resources = bind_resources(binder, config, token_map, vision);
    Bindings b(binder);
    const bool offload = options.experts != ExpertResidency::Device;
    std::vector<std::array<artifact::ParameterReference, 2>> expert_references;
    const std::uint64_t h  = config.hidden_size;
    const std::uint64_t d  = config.head_dim;
    const std::uint64_t hc = config.hc_mult;
    TextWeights weights;
    weights.token_embedding = b.parameter("text/token_embedding", {config.vocab_size, h});
    weights.output_head     = b.parameter("text/output_head", {config.vocab_size, h},
                                          {"text/final_hidden"});
    weights.final_norm      = b.direct("text/final_norm", {h}, QType::FP32);
    tail(b, weights.output_head);
    const std::uint64_t q_rank = config.q_lora_rank;
    const std::uint64_t heads  = config.num_attention_heads;
    const std::uint64_t groups = config.o_groups, o_rank = config.o_lora_rank;
    const std::uint64_t ih = config.index_heads, idim = config.index_head_dim;
    const std::uint64_t e = config.num_experts, w = config.moe_intermediate_size;
    // What every block binds, the backbone's and the drafter's: hyper-connections, norms, the
    // window attention, the router and the shared expert.
    // `matrices` false leaves the attention's and the shared expert's matrices to an STQ binding.
    const auto block = [&](const std::string& p, std::uint64_t experts, bool matrices = true) {
        const std::string a = p + "attention/";
        LayerWeights layer;
        for (auto [site, prefix] : {std::pair{&layer.hc_attention, p + "hc_attention/"},
                                    std::pair{&layer.hc_ffn, p + "hc_ffn/"}}) {
            site->fn    = b.direct(prefix + "fn", {config.hc_mixes(), hc * h}, QType::FP32);
            site->base  = b.direct(prefix + "base", {config.hc_mixes()}, QType::FP32);
            site->scale = b.direct(prefix + "scale", {3}, QType::FP32);
        }
        layer.attention_norm = b.direct(p + "attention_norm", {h}, QType::FP32);
        layer.ffn_norm       = b.direct(p + "ffn_norm", {h}, QType::FP32);
        auto& at             = layer.attention;
        at.query_a_norm = b.direct(a + "query_a_norm", {q_rank}, QType::FP32);
        at.kv_norm      = b.direct(a + "kv_norm", {d}, QType::FP32);
        at.sinks        = b.direct(a + "sinks", {heads}, QType::FP32);
        if (matrices) {
            at.query_a  = b.parameter(a + "query_a", {q_rank, h}, {p + "attention_input"});
            at.kv       = b.parameter(a + "kv", {d, h}, {p + "attention_input"});
            at.query_b  = b.parameter(a + "query_b", {heads * d, q_rank}, {a + "query_latent"});
            at.output_a = b.parameter(a + "output_a", {groups * o_rank, heads * d / groups},
                                      {a + "grouped_output"});
            at.output_b = b.parameter(a + "output_b", {h, groups * o_rank}, {a + "output_latent"});
            for (const WeightId id : {at.query_a, at.kv, at.query_b, at.output_a, at.output_b}) {
                tail(b, id);
            }
        }
        const std::string m = p + "moe/";
        MoeWeights& moe     = layer.moe;
        moe.router          = b.direct(m + "router", {experts, h}, QType::FP32);
        moe.router_bias     = b.direct(m + "router_bias", {experts}, QType::FP32);
        if (matrices) {
            moe.shared_gate = b.parameter(m + "shared_gate", {w, h}, {m + "input"});
            moe.shared_up   = b.parameter(m + "shared_up", {w, h}, {m + "input"});
            moe.shared_down = b.parameter(m + "shared_down", {h, w}, {m + "shared_product"});
            tail(b, moe.shared_down);
            tail(b, moe.shared_gate);
        }
        return layer;
    };
    // An STQ matrix of k inputs and n outputs, tiles of `words` words; `count` > 0 leads a set.
    const auto stq = [&](const std::string& name, std::uint64_t k, std::uint64_t n,
                         std::uint64_t words, std::uint64_t count = 0,
                         artifact::Residency residency = artifact::Residency::Device) {
        artifact::Shape tiles{k / 16, n / 16, words}, in{k}, out{n};
        if (count != 0) {
            for (artifact::Shape* shape : {&tiles, &in, &out}) { shape->insert(shape->begin(), count); }
        }
        return StqWeights{b.direct(name + "/trellis", std::move(tiles), QType::UINT16, residency),
                          b.direct(name + "/suh", std::move(in), QType::FP16, residency),
                          b.direct(name + "/svh", std::move(out), QType::FP16, residency)};
    };
    // `residency` Pinned: the drafter's experts that --draft-experts leaves on the host, of which
    // the executor copies the kept ones to the device (with the zeros the kernels read past them).
    const auto resident_experts = [&](LayerWeights& layer, const std::string& p,
                                      std::uint64_t experts,
                                      artifact::Residency residency = artifact::Residency::Device) {
        const std::string m = p + "moe/";
        layer.moe.gate_up = b.parameter(m + "gate_up", {experts * 2 * w, h}, {m + "input"}, {},
                                        residency);
        layer.moe.down    = b.parameter(m + "down", {experts * h, w}, {m + "product"}, {},
                                        residency);
        if (residency == artifact::Residency::Device) {
            tail(b, *layer.moe.down);
            tail(b, *layer.moe.gate_up);
        }
    };
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const std::string p = "text/layers/" + std::to_string(i) + "/";
        const std::string a = p + "attention/";
        LayerWeights layer = block(p, e);
        auto& at           = layer.attention;
        if (config.kv_source[i]) {
            at.compressor_kv = b.direct(a + "compressor/kv", {d, h}, QType::FP32);
            if (config.compress_ratios[i] > 1) {
                at.compressor_gate = b.direct(a + "compressor/gate", {d, h}, QType::FP32);
            }
            at.compressor_norm = b.direct(a + "compressor/norm", {d}, QType::FP32);
            at.index_key       = b.direct(a + "indexer/key", {idim, d}, QType::FP32);
            at.index_key_norm  = b.direct(a + "indexer/key_norm", {idim}, QType::FP32);
        }
        if (config.index_source[i]) {
            at.index_query   = b.direct(a + "indexer/query", {ih * idim, q_rank}, QType::FP32);
            at.index_weights = b.direct(a + "indexer/weights", {ih, h}, QType::FP32);
        }
        const std::string m = p + "moe/";
        if (offload) {
            const auto residency = options.experts == ExpertResidency::Host
                                       ? artifact::Residency::Pinned : artifact::Residency::Values;
            expert_references.push_back({
                binder.parameter(m + "gate_up", {e * 2 * w, h}, residency),
                binder.parameter(m + "down", {e * h, w}, residency)});
            // Uses still belong to the family contract, even without a resident operand.
            for (const auto& use : {std::pair{m + "gate_up", m + "input"},
                                    std::pair{m + "down", m + "product"}}) {
                const auto& contract = binder.use(use.first, use.second);
                if (!contract.activation_policy || !contract.auxiliaries.empty()) {
                    throw artifact::ArtifactError(use.first +
                        ": streamed GGUF experts require an activation policy and no auxiliaries");
                }
            }
        } else {
            resident_experts(layer, p, e);
        }
        if (config.engram.index(i) >= 0) {
            const std::string n = p + "engram/";
            const std::uint64_t width = std::uint64_t(config.engram.columns()) *
                                        config.engram.head_dim;
            EngramWeights engram;
            engram.projection = b.parameter(n + "projection", {(hc + 1) * h, width}, {n + "rows"});
            engram.gate       = b.direct(n + "gate", {hc, h}, QType::FP32);
            tail(b, engram.projection);
            layer.engram = engram;
        }
        weights.layers.push_back(std::move(layer));
    }
    std::optional<DSparkConfig> dspark_config;
    std::optional<DSparkWeights> dspark;
    std::vector<std::vector<std::int32_t>> dspark_kept;
    if (options.dspark) {
        if (!has_dspark(reader)) {
            throw artifact::ArtifactError("the artifact carries no DSpark drafter (convert with "
                                          "--components text,dspark)");
        }
        dspark_config = parse_dspark_config(reader.directory().component("dspark").config, config);
        const std::uint64_t taps = dspark_config->target_layers.size();
        const std::uint64_t rank = dspark_config->markov_rank;
        DSparkWeights& drafter   = dspark.emplace();
        const bool stq_drafter   = dspark_config->stq;
        constexpr std::uint64_t kTrunkWords = 64, kExpertWords = 32;
        if (stq_drafter) {
            drafter.stq_fc = stq("dspark/fc", taps * h, h, kTrunkWords);
        } else {
            drafter.fc = b.parameter("dspark/fc", {h, taps * h}, {"dspark/taps"});
            tail(b, drafter.fc);
        }
        drafter.tap_norm         = b.direct("dspark/tap_norm", {h}, QType::FP32);
        drafter.final_norm       = b.direct("dspark/final_norm", {h}, QType::FP32);
        drafter.markov_embed     = b.direct("dspark/markov_embed", {config.vocab_size, rank});
        drafter.markov_head      = b.direct("dspark/markov_head", {config.vocab_size, rank});
        drafter.confidence       = b.direct("dspark/confidence", {h + rank}, QType::FP32);
        const std::uint64_t de   = dspark_config->experts;
        // Measured with the STQ drafter only (docs/deepseek-v41.md#the-drafters-experts).
        const std::uint32_t keep =
            options.draft_experts.value_or(
                stq_drafter && options.experts != ExpertResidency::Device ? kDefaultDraftExperts : 0U);
        if (keep != 0 && keep < de) {
            if (dspark_config->layers != kDSparkExpertRanking.size() ||
                de != kDSparkExpertRanking[0].size()) {
                throw artifact::ArtifactError(
                    "--draft-experts: the expert ranking describes a drafter of 3 blocks of 128 "
                    "experts");
            }
            if (keep < dspark_config->top_k) {
                throw std::invalid_argument("--draft-experts must keep at least the " +
                                            std::to_string(dspark_config->top_k) +
                                            " experts a row routes to");
            }
            for (const auto& ranking : kDSparkExpertRanking) {
                std::vector<std::int32_t> kept(ranking.begin(),
                                               ranking.begin() + keep);
                std::sort(kept.begin(), kept.end());
                dspark_kept.push_back(std::move(kept));
            }
        }
        const auto experts_residency =
            dspark_kept.empty() ? artifact::Residency::Device : artifact::Residency::Pinned;
        for (std::uint32_t i = 0; i < dspark_config->layers; ++i) {
            const std::string p = "dspark/layers/" + std::to_string(i) + "/";
            LayerWeights layer  = block(p, de, !stq_drafter);
            if (stq_drafter) {
                const std::string a = p + "attention/", m = p + "moe/";
                StqLayerWeights s;
                s.query_a     = stq(a + "query_a", h, q_rank, kTrunkWords);
                s.kv          = stq(a + "kv", h, d, kTrunkWords);
                s.query_b     = stq(a + "query_b", q_rank, heads * d, kTrunkWords);
                s.output_a    = stq(a + "output_a", heads * d / groups, o_rank, kTrunkWords, groups);
                s.output_b    = stq(a + "output_b", groups * o_rank, h, kTrunkWords);
                s.shared_gate = stq(m + "shared_gate", h, w, kTrunkWords);
                s.shared_up   = stq(m + "shared_up", h, w, kTrunkWords);
                s.shared_down = stq(m + "shared_down", w, h, kTrunkWords);
                s.gate        = stq(m + "gate", h, w, kExpertWords, de, experts_residency);
                s.up          = stq(m + "up", h, w, kExpertWords, de, experts_residency);
                s.down        = stq(m + "down", w, h, kExpertWords, de, experts_residency);
                drafter.stq_layers.push_back(s);
            } else {
                resident_experts(layer, p, de, experts_residency);
            }
            drafter.layers.push_back(std::move(layer));
        }
    }
    std::optional<VisionConfig> vision_config;
    std::optional<VisionWeights> vision_weights;
    if (vision) {
        const VisionConfig& vc =
            vision_config.emplace(parse_vision_config(reader.directory().component("vision").config,
                                                      config));
        const auto residency = options.vision == VisionPlacement::Pinned
                                   ? artifact::Residency::Pinned : artifact::Residency::Device;
        const std::uint64_t vh = vc.hidden_size, vw = vc.intermediate_size;
        const auto matrix = [&](const std::string& name, artifact::Shape shape,
                                const std::string& input) {
            return b.parameter(name, std::move(shape), {input}, QType::BF16, residency);
        };
        const auto vector = [&](const std::string& name, std::uint64_t n) {
            return b.direct(name, {n}, QType::FP32);
        };
        VisionWeights& v   = vision_weights.emplace();
        v.patch_embedding  = matrix("vision/patch_embedding", {vh, vc.patch_features()},
                                    "vision/patches");
        v.patch_bias       = vector("vision/patch_bias", vh);
        v.final_norm       = vector("vision/final_norm", vh);
        v.aligner_up       = matrix("vision/aligner/up", {h, vc.merged_features()}, "vision/merged");
        v.aligner_up_bias  = vector("vision/aligner/up_bias", h);
        v.aligner_out      = matrix("vision/aligner/out", {h, h}, "vision/aligner_hidden");
        v.aligner_out_bias = vector("vision/aligner/out_bias", h);
        v.image_start      = vector("vision/image_start", h);
        v.image_newline    = vector("vision/image_newline", h);
        v.image_end        = vector("vision/image_end", h);
        v.image_router_bias =
            b.direct("vision/image_router_bias", {config.num_hidden_layers, e}, QType::FP32);
        for (std::uint32_t i = 0; i < vc.num_hidden_layers; ++i) {
            const std::string p = "vision/layers/" + std::to_string(i) + "/";
            VisionLayerWeights layer;
            layer.attention_norm = vector(p + "attention_norm", vh);
            layer.qkv            = matrix(p + "attention/qkv", {3 * vh, vh}, p + "attention_input");
            layer.qkv_bias       = vector(p + "attention/qkv_bias", 3 * vh);
            layer.output         = matrix(p + "attention/output", {vh, vh}, p + "attention_output");
            layer.output_bias    = vector(p + "attention/output_bias", vh);
            layer.mlp_norm       = vector(p + "mlp_norm", vh);
            layer.gate           = matrix(p + "mlp/gate", {vw, vh}, p + "mlp_input");
            layer.up             = matrix(p + "mlp/up", {vw, vh}, p + "mlp_input");
            layer.down           = matrix(p + "mlp/down", {vh, vw}, p + "mlp_product");
            v.layers.push_back(layer);
        }
    }
    InstanceInfo info;
    info.name            = reader.directory().metadata.value("name", std::string("DeepSeek-V4.1-Flash"));
    info.metadata_json   = reader.directory().metadata.dump();
    info.provenance_json = reader.directory().provenance.dump();
    info.artifact_id     = reader.artifact_id();
    auto pending         = std::move(b.weights);
    auto materialization = std::move(binder).finish();
    auto backing = artifact::materialize(reader, std::move(materialization), device, observer);
    auto bound   = qwen3_5::loading::resolve_weights(std::move(pending), backing);
    std::vector<ExpertBanks> expert_banks;
    for (const auto& reference : expert_references) {
        ExpertBanks bank;
        auto gu = expert_blobs(reader, reference[0], backing, e, 2 * w, h, bank.gate_up_type,
                               bank.gate_up_row_bytes);
        auto down = expert_blobs(reader, reference[1], backing, e, h, w, bank.down_type,
                                 bank.down_row_bytes);
        for (std::size_t x = 0; x < gu.size(); ++x) {
            bank.delivery.experts.push_back({std::move(gu[x]), std::move(down[x])});
        }
        // Gate rows, up rows, the [gate; up] parent, down rows: the executor's expert tables.
        bank.delivery.aliases = {{0, 0}, {0, w * bank.gate_up_row_bytes}, {0, 0}, {1, 0}};
        expert_banks.push_back(std::move(bank));
    }
    auto model = std::unique_ptr<Model>(new Model(std::move(config), std::move(weights),
                                                  std::move(bound), std::move(resources),
                                                  std::move(info), std::move(backing)));
    model->engram_token_map_ = std::move(token_map);
    model->dspark_config_    = std::move(dspark_config);
    model->dspark_           = std::move(dspark);
    model->dspark_kept_      = std::move(dspark_kept);
    model->vision_config_    = std::move(vision_config);
    model->vision_           = std::move(vision_weights);
    model->vision_placement_ = options.vision;
    model->expert_residency_ = options.experts;
    model->expert_banks_     = std::move(expert_banks);
    if (options.experts == ExpertResidency::Disk) {
        for (const auto& file : reader.directory().files) {
            model->expert_files_.push_back(file.path ? options.artifact.parent_path() / *file.path
                                                     : options.artifact);
        }
    }
    if (!options.dspine.empty()) {
        model->dspine_ = dspine::load_model(options.dspine, device, observer);
    }
    return model;
}

} // namespace ninfer::models::deepseek_v41
