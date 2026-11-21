#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/reader.h"
#include "models/qwen3_5/load.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace qwen = ninfer::models::qwen3_5;
using ninfer::test::artifact_fixture::require;

struct Sample {
    std::uint64_t offset;
    std::vector<std::byte> bytes;
};

void check_selection(const qwen::LoadPlan& plan, const LoadOptions& options) {
    const auto& weights = plan.weights();
    require(weights.vision.has_value() == options.vision,
            "Vision demand differs from startup selection");
    require(weights.mtp.has_value() == (options.speculative == SpeculativeBackend::Mtp),
            "MTP selection differs");
    require(weights.draft.has_value() == (options.speculative == SpeculativeBackend::DFlash ||
                                          options.speculative == SpeculativeBackend::DFlash2),
            "draft selection differs");
    require(weights.proposal.has_value() == options.proposal_enabled(),
            "proposal selection differs");
    require(weights.text.layers.size() == plan.config().text.num_hidden_layers,
            "layer data differs from instance config");
    const auto check_head = [&](qwen::WeightId head, qwen::WeightUseId use,
                                const std::string& input) {
        require(use.parameter == head && plan.uses(head)[use.use_index].input == input,
                "shared output head selected the wrong mathematical Use");
    };
    check_head(weights.text.output_head, weights.text.output_head_use, "text/final_hidden");
    if (weights.mtp) {
        check_head(weights.mtp->output_head, weights.mtp->output_head_use, "mtp/final_hidden");
        require(weights.mtp->token_embedding == weights.text.token_embedding,
                "MTP embedding did not share target");
    }
    if (weights.draft) {
        check_head(weights.draft->output_head, weights.draft->output_head_use,
                   std::string(options.speculative_component()) + "/final_hidden");
        require(weights.draft->token_embedding == weights.text.token_embedding,
                "draft embedding did not share target");
    }
    if (!options.vision) {
        require(plan.resources().preprocessor_config_json.empty() &&
                    plan.resources().video_preprocessor_config_json.empty(),
                "Text-only loading retained Vision resources");
    }
    std::set<std::size_t> parents;
    std::uint64_t end = 0;
    for (const auto& item : plan.materialization().device_objects) {
        require(parents.insert(item.object.index).second,
                "device parent was planned more than once");
        require(item.offset >= end && item.offset % item.alignment == 0,
                "device placement overlaps or is unaligned");
        end = item.offset + item.bytes;
    }
    require(end == plan.materialization().device_capacity(0),
            "weight capacity differs from actual placements");
}

// Exercise native grouping after Reader destruction, independently of Program selection. Weights
// are prepared as execution prepares them, through rotated_input, so a Hadamard-rotated artifact
// is covered too.
void check_native_inputs(const qwen::Model& model) {
    const auto linear = [&](qwen::WeightId id) {
        return ops::prepare_linear_weight(model.rotated_input(id));
    };
    const auto dense = [&](const qwen::DenseWeights& weights) {
        (void)ops::prepare_linear_swiglu_weight(model.rotated_input(weights.gate),
                                                model.rotated_input(weights.up));
        (void)linear(weights.down);
    };
    const auto block = [&](const qwen::BlockWeights& weights, bool mtp = false) {
        if (const auto* attention = std::get_if<qwen::AttentionWeights>(&weights.mixer)) {
            if (mtp) {
                const std::array bank{
                    model.rotated_input(attention->query), model.rotated_input(attention->key),
                    model.rotated_input(attention->gate), model.rotated_input(attention->value)};
                (void)ops::prepare_linear_weight(bank);
                for (const auto& input : bank) { (void)ops::prepare_linear_weight(input); }
            } else {
                (void)ops::prepare_attn_input_proj_weights(
                    model.rotated_input(attention->query), model.rotated_input(attention->key),
                    model.rotated_input(attention->gate), model.rotated_input(attention->value));
            }
            (void)linear(attention->output);
        } else {
            const auto& gdn = std::get<qwen::GdnWeights>(weights.mixer);
            (void)ops::prepare_gdn_input_proj_weights(
                model.rotated_input(gdn.query), model.rotated_input(gdn.key),
                model.rotated_input(gdn.value), model.rotated_input(gdn.z));
            (void)ops::prepare_gdn_gating_proj_weights(model.rotated_input(gdn.a_projection),
                                                       model.rotated_input(gdn.b_projection));
            (void)linear(gdn.output);
        }
        if (const auto* mlp = std::get_if<qwen::DenseWeights>(&weights.ffn)) {
            dense(*mlp);
        } else {
            const auto& moe = std::get<qwen::MoeWeights>(weights.ffn);
            std::vector<ops::WeightInput> gate_up, down;
            for (const auto& expert : moe.experts) {
                gate_up.push_back(model.rotated_input(expert.gate));
                gate_up.push_back(model.rotated_input(expert.up));
                down.push_back(model.rotated_input(expert.down));
            }
            (void)ops::prepare_sparse_moe_weights(
                model.rotated_input(moe.router), model.rotated_input(moe.shared_score), gate_up,
                down, model.rotated_input(moe.shared.gate), model.rotated_input(moe.shared.up),
                model.rotated_input(moe.shared.down));
        }
    };
    const auto& weights = model.weights();
    (void)native_weight(model.weight(weights.text.token_embedding).view);
    (void)ops::prepare_linear_weight(model.rotated_input(weights.text.output_head_use));
    for (const auto& layer : weights.text.layers) { block(layer); }
    if (weights.vision) {
        const auto& vision = *weights.vision;
        (void)linear(vision.patch_embedding);
        for (const auto& layer : vision.layers) {
            const std::array qkv{model.rotated_input(layer.query), model.rotated_input(layer.key),
                                 model.rotated_input(layer.value)};
            (void)ops::prepare_linear_weight(qkv);
            (void)linear(layer.output);
            (void)linear(layer.fc1);
            (void)linear(layer.fc2);
        }
        (void)linear(vision.merger_fc1);
        (void)linear(vision.merger_fc2);
    }
    if (weights.mtp) {
        (void)linear(weights.mtp->input_projection);
        block(weights.mtp->layer, true);
        (void)ops::prepare_linear_weight(model.rotated_input(weights.mtp->output_head_use));
    }
    if (weights.draft) {
        const auto& draft = *weights.draft;
        (void)linear(draft.feature_projection);
        for (const auto& layer : draft.layers) {
            const auto& a = layer.attention;
            (void)ops::prepare_attn_input_proj_weights(model.rotated_input(a.query),
                                                       model.rotated_input(a.key),
                                                       model.rotated_input(a.value));
            const std::array kv{model.rotated_input(a.context_key),
                                model.rotated_input(a.context_value)};
            (void)ops::prepare_linear_weight(kv);
            (void)linear(a.output);
            dense(layer.mlp);
            if (layer.attention_conv) { (void)linear(layer.attention_conv->kernel_projection); }
            if (layer.mlp_conv) { (void)linear(layer.mlp_conv->kernel_projection); }
        }
        if (draft.selector) { (void)linear(draft.selector->hidden_projection); }
        (void)ops::prepare_linear_weight(model.rotated_input(draft.output_head_use));
    }
    if (model.weight(weights.text.output_head).uses.size() > 1) {
        ninfer::test::artifact_fixture::rejects<std::invalid_argument>(
            [&] { (void)model.rotated_input(weights.text.output_head); },
            "ambiguous shared output-head Use was silently selected");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path path;
        LoadOptions options;
        bool host_only = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            const auto value = [&]() -> std::string {
                if (++i >= argc) { throw std::invalid_argument("missing value for " + arg); }
                return argv[i];
            };
            if (arg == "--artifact") {
                path = value();
            } else if (arg == "--vision") {
                options.vision = true;
            } else if (arg == "--host-only") {
                host_only = true;
            } else if (arg == "--scoring") {
                options.purpose = EnginePurpose::CausalScoring;
            } else if (arg == "--speculative") {
                const auto backend = value();
                if (backend == "none") {
                    options.speculative = SpeculativeBackend::None;
                } else if (backend == "mtp") {
                    options.speculative = SpeculativeBackend::Mtp;
                } else if (backend == "dflash") {
                    options.speculative = SpeculativeBackend::DFlash;
                } else if (backend == "dflash2") {
                    options.speculative = SpeculativeBackend::DFlash2;
                } else {
                    throw std::invalid_argument("unknown speculative backend");
                }
            } else if (arg == "--proposal") {
                const auto head = value();
                if (head == "full") {
                    options.proposal_head = ProposalHead::Full;
                } else if (head == "optimized") {
                    options.proposal_head = ProposalHead::Optimized;
                } else {
                    throw std::invalid_argument("unknown proposal head");
                }
            } else if (arg == "--help") {
                std::cout << "--artifact PATH [--vision] [--speculative none|mtp|dflash|dflash2]\n"
                             "[--proposal full|optimized] [--host-only] [--scoring]\n";
                return 0;
            } else {
                throw std::invalid_argument("unknown argument " + arg);
            }
        }
        if (path.empty()) {
            std::cout << "SKIP: supply an explicit --artifact\n";
            return 77;
        }
        std::unique_ptr<DeviceContext> device;
        std::unique_ptr<qwen::Model> model;
        std::map<std::string, std::vector<Sample>> samples;
        std::uint64_t expected_h2d = 0;
        {
            artifact::Reader reader(path);
            auto plan = qwen::plan_load(reader, options);
            check_selection(plan, options);
            {
                // The default pipeline split follows each device's free memory, and the first stage
                // also holds the embedding and head.
                const std::uint32_t layers = plan.config().text.num_hidden_layers;
                const auto sum = [](const std::vector<std::uint32_t>& counts) {
                    std::uint32_t total = 0;
                    for (const std::uint32_t count : counts) { total += count; }
                    return total;
                };
                const std::array<std::uint64_t, 2> equal{40ULL << 30, 40ULL << 30};
                const qwen::StageSizing sizing{.kv_storage = KvCacheStorage::BFloat16,
                                               .state_slots = 2};
                const auto even = qwen::default_stage_layers(reader, options, sizing, equal);
                require(even.size() == 2 && sum(even) == layers && even[0] >= 1 && even[1] >= 1,
                        "default split does not cover the model");
                require(even[0] <= even[1], "the stage holding the head took more layers");
                const std::array<std::uint64_t, 2> lopsided{40ULL << 30, 80ULL << 30};
                const auto skewed =
                    qwen::default_stage_layers(reader, options, sizing, lopsided);
                require(sum(skewed) == layers && skewed[1] > skewed[0],
                        "the device with twice the memory did not take more layers");
                std::cout << "default stage layers: equal devices " << even[0] << '/' << even[1]
                          << ", 40/80 GiB " << skewed[0] << '/' << skewed[1] << '\n';
            }
            if (plan.config().text.gdn) {
                // After a load, the automatic tensor split moves only what a rank lacked, or gives
                // a rank the room the estimate did not.
                LoadOptions tp     = options;
                tp.ranks           = 2;
                tp.tensor_parallel = true;
                tp.tensor_split    = {1.0, 1.0};
                const qwen::StageSizing sizing{.kv_storage = KvCacheStorage::BFloat16,
                                               .state_slots = 1};
                const std::vector<double> equal_speed{1.0, 1.0};
                const auto rank_weights = [&](const std::vector<double>& split, std::size_t rank) {
                    LoadOptions loaded  = tp;
                    loaded.tensor_split = split;
                    const auto at       = qwen::plan_load(reader, loaded, path);
                    return at.materialization().device_capacity(rank);
                };
                const auto rank0_bytes = [&](const std::vector<double>& split) {
                    return rank_weights(split, 0);
                };
                const std::array<std::int64_t, 2> room{1LL << 30, 1LL << 30};
                require(!qwen::refit_tensor_split(reader, tp, sizing, room, equal_speed, path, {}),
                        "a split that fits equal devices was refit");
                const std::array<std::int64_t, 2> lacking{-(512LL << 20), 4LL << 30};
                const auto smaller =
                    qwen::refit_tensor_split(reader, tp, sizing, lacking, equal_speed, path, {});
                require(smaller && (*smaller)[0] < 0.5, "rank 0 kept the share it lacked room for");
                const std::uint64_t even0 = rank0_bytes(tp.tensor_split);
                const std::uint64_t off   = even0 - rank0_bytes(*smaller);
                // The GDN state of the heads that move is part of the 512 MiB.
                require(off >= (480ULL << 20), "the refit left rank 0 short");
                // Halfway back to the even split, rank 0 lacks room again: the refit moved no
                // whole unit more than it had to.
                const double back = ((*smaller)[0] + 0.5) / 2.0;
                const std::uint64_t off_back = even0 - rank0_bytes({back, 1.0 - back});
                require(off_back == off || off_back < (512ULL << 20),
                        "the refit moved more than rank 0 lacked");
                const std::vector<double> faster0{3.0, 1.0};
                const auto larger =
                    qwen::refit_tensor_split(reader, tp, sizing, room, faster0, path, {});
                require(larger && (*larger)[0] > 0.5, "the faster rank was not given its room");
                const std::uint64_t on = rank0_bytes(*larger) - rank0_bytes(tp.tensor_split);
                require(on <= (1ULL << 30), "the faster rank was given more than its room");
                // When the ranks together lack room for the KV asked for, automatic capacity
                // shrinks it until the rank short of the largest part of its own KV fits, so the
                // refit shares the shortfall out; an even shortfall stands.
                qwen::StageSizing automatic = sizing;
                automatic.kv_tokens         = 262144;
                automatic.automatic_kv      = true;
                const std::array<std::int64_t, 2> uneven{-(1536LL << 20), -(128LL << 20)};
                const auto shared =
                    qwen::refit_tensor_split(reader, tp, automatic, uneven, equal_speed, path, {});
                require(shared && (*shared)[0] < 0.5, "rank 0 kept the share it was shortest of");
                LoadOptions refit     = tp;
                refit.tensor_split    = *shared;
                const auto refit_plan = qwen::plan_load(reader, refit, path);
                const auto& heads     = refit_plan.weights().text.tensor_shares;
                // An uneven split (#384's 0.53/0.47) ends the head shares on whole 128-row tiles,
                // the only ones a GGUF head's matrix product takes.
                LoadOptions lopsided_head     = tp;
                lopsided_head.tensor_split    = {0.53, 0.47};
                const auto lopsided_head_plan = qwen::plan_load(reader, lopsided_head, path);
                const auto& refit_head    = refit_plan.weights().text.output_head_shares;
                const auto& lopsided_rows = lopsided_head_plan.weights().text.output_head_shares;
                for (const auto& head_rows : {refit_head.rows, lopsided_rows.rows}) {
                    require(head_rows.size() == 3 && head_rows[1] % 128 == 0 && head_rows[1] > 0 &&
                                head_rows[1] < head_rows[2],
                            "a head share does not end on a whole 128-row tile");
                }
                require(heads[0].kv_head_end - heads[0].kv_head_begin ==
                            heads[1].kv_head_end - heads[1].kv_head_begin,
                        "the refit moved a KV head to share a shortfall smaller than its KV");
                const double moved  = static_cast<double>(even0 - rank0_bytes(*shared));
                const double short0 = static_cast<double>(1536LL << 20) - moved;
                const double short1 = static_cast<double>(128LL << 20) +
                                      static_cast<double>(rank_weights(*shared, 1)) -
                                      static_cast<double>(rank_weights(tp.tensor_split, 1));
                require(std::max(short0, short1) < static_cast<double>(1152LL << 20),
                        "the refit did not share the shortfall out");
                // No split that leaves a rank short of the longest context starts.
                qwen::StageSizing whole_context = automatic;
                whole_context.minimum_kv_tokens = automatic.kv_tokens;
                require(!qwen::refit_tensor_split(reader, tp, whole_context, uneven, equal_speed,
                                                  path, {}),
                        "a split short of the smallest KV was refit");
                const std::array<std::int64_t, 2> even{-(512LL << 20), -(512LL << 20)};
                require(
                    !qwen::refit_tensor_split(reader, tp, automatic, even, equal_speed, path, {}),
                    "an even shortfall was refit");
                std::cout << "tensor split refit: " << (off >> 20)
                          << " MiB off rank 0 short of 512, " << (on >> 20)
                          << " MiB onto rank 0 with 1024 to spare, ranks short of "
                          << static_cast<long long>(short0) / (1LL << 20) << " and "
                          << static_cast<long long>(short1) / (1LL << 20)
                          << " MiB after 1536 and 128\n";
            }
            if (host_only) {
                std::cout << path.filename().string() << ": Host binding passed, architecture="
                          << architecture_name(plan.config().text.architecture)
                          << " V=" << plan.resources().public_token_count
                          << " device_bytes=" << plan.materialization().device_capacity(0)
                          << '\n';
                return 0;
            }
            for (const auto& item : plan.materialization().device_objects) {
                const auto& object   = reader.directory().tensor(item.object);
                const auto& geometry = reader.geometry(item.object);
                expected_h2d += object.bytes;
                const auto size = std::min<std::uint64_t>(64, object.bytes);
                std::set<std::uint64_t> offsets{0, (object.bytes - size) / 2, object.bytes - size};
                if (geometry.high_bytes) { offsets.insert(geometry.high_offset); }
                if (geometry.scale_bytes) { offsets.insert(geometry.scale_offset); }
                for (const auto offset : offsets) {
                    const auto count = std::min<std::uint64_t>(size, object.bytes - offset);
                    samples[object.id].push_back(
                        {offset, reader.read_range(object.offset + offset, count)});
                }
            }
            device = std::make_unique<DeviceContext>();
            model  = qwen::materialize_model(std::move(plan), *device);
        }
        require(model->storage_stats().h2d_bytes == expected_h2d,
                "H2D did not cover all selected parent bytes");
        std::set<const WeightParent*> visited;
        for (const auto& weight : model->weight_data()) {
            require(weight.view.parts.size() == weight.source_objects.size(),
                    "diagnostic parent association lost");
            for (std::size_t i = 0; i < weight.view.parts.size(); ++i) {
                const auto* parent = weight.view.parts[i].parent;
                if (!visited.insert(parent).second) { continue; }
                for (const auto& sample : samples.at(weight.source_objects[i])) {
                    std::vector<std::byte> received(sample.bytes.size());
                    CUDA_CHECK(cudaMemcpy(received.data(), parent->data + sample.offset,
                                          received.size(), cudaMemcpyDeviceToHost));
                    require(received == sample.bytes,
                            "resident parent differs from original v3 bytes");
                }
            }
        }
        require(visited.size() == samples.size(),
                "some resident parents are not reachable from model weights");
        check_native_inputs(*model);
        const auto ids = model->resources().tokenizer->encode("Hello");
        require(model->resources().tokenizer->decode(ids) == "Hello",
                "tokenizer did not survive Reader destruction");
        require(!model->resources().chat_template_jinja.empty(),
                "template bytes were lost after loading");
        std::cout << path.filename().string()
                  << ": GPU loading and native inputs passed, parents=" << visited.size()
                  << " device_bytes=" << model->storage_stats().device_capacity_bytes
                  << " V=" << model->resources().public_token_count << '\n';
        model.reset();
        device->synchronize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
