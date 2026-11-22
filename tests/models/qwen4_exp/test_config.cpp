// parse_text_config on the text config the converter writes for the Qwen3.8-Flash-Next
// checkpoint (tests/fixtures/qwen4_exp/text_config.json, which tests/convert/test_qwen4_exp.py
// holds to tools/convert/qwen4_exp.py) and for dense Qwen4 (tests/fixtures/qwen4, at the 27B's
// inferred geometry), parse_mtp_config on the MTP blocks', and their refusals.
#include "models/qwen4_exp/config.h"

#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace ninfer::models::qwen4_exp;
using ninfer::artifact::Json;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

Json fixture(const char* family = "qwen4_exp") {
    std::ifstream file(std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/" + family +
                       "/text_config.json");
    require(file.good(), "fixture missing");
    return Json::parse(file);
}

void expect_refusal(const std::function<void(Json&)>& edit, const std::string& label,
                    const char* family = "qwen4_exp") {
    Json config = fixture(family);
    edit(config);
    bool refused = false;
    try {
        (void)parse_text_config(config);
    } catch (const ninfer::artifact::ArtifactError&) { refused = true; }
    require(refused, "not refused: " + label);
}

// Dense Qwen4: a SwiGLU MLP in every block, hidden 2560 or 5120 with its hyper-connection rank
// hidden / 8, two or four KV heads, an optional PLE, and the MTP block's MLP.
int dense() {
    const TextConfig config = parse_text_config(fixture("qwen4"));
    require(config.dense() && config.ffn == FfnKind::Dense, "dense Qwen4");
    require(config.hidden_size == 5120 && config.hc_lowrank == 640 &&
                config.num_key_value_heads == 4 && config.intermediate_size == 14336,
            "the dense geometry");
    require(config.num_hidden_layers == 64 && config.num_experts == 0, "64 blocks, no experts");
    require(config.ple_layers == std::vector<std::uint32_t>{1} && config.ngram_heads() == 16,
            "PLE before block 1");
    Json narrow = fixture("qwen4");
    narrow["hidden_size"] = 2560;
    narrow["hc_lowrank"] = 320;
    narrow["num_key_value_heads"] = 2;
    require(parse_text_config(narrow).hidden_size == 2560, "the dense model at hidden 2560");

    // Without PLE: no PLE or n-gram field, and no table.
    Json plain = fixture("qwen4");
    plain["ple_layers"] = Json::array();
    for (const char* field : {"ple_embed_dim", "ple_conv_kernel_size", "ngram_size",
                              "heads_per_ngram", "ngram_vocab_size_base",
                              "make_ngram_vocab_size_divisible_by", "ngram_seed"}) {
        plain.erase(field);
    }
    require(parse_text_config(plain).ple_layers.empty(), "dense Qwen4 without PLE");
    expect_refusal([](Json& c) { c["ple_layers"] = Json::array(); },
                   "PLE fields without PLE blocks", "qwen4");
    expect_refusal([](Json& c) { c.erase("ngram_seed"); }, "PLE blocks without their fields",
                   "qwen4");

    expect_refusal([](Json& c) { c["intermediate_size"] = 14337; },
                   "an MLP off 256-value blocks", "qwen4");
    expect_refusal([](Json& c) { c.erase("intermediate_size"); }, "a missing MLP width", "qwen4");
    expect_refusal([](Json& c) { c["hidden_size"] = 4096; c["hc_lowrank"] = 512; },
                   "another hidden width", "qwen4");
    expect_refusal([](Json& c) { c["hc_lowrank"] = 320; }, "another hyper-connection rank", "qwen4");
    expect_refusal([](Json& c) { c["num_key_value_heads"] = 3; }, "three KV heads", "qwen4");
    expect_refusal([](Json& c) { c["num_experts"] = 512; }, "an MoE field", "qwen4");
    expect_refusal([](Json& c) { c["expert_output_scales"] = false; }, "an expert flag", "qwen4");
    expect_refusal([](Json& c) { c["model_type"] = "qwen4_exp_text"; },
                   "the MoE model type", "qwen4");

    const Json mtp = {{"architectures", Json::array({"Qwen4MTP"})},
                      {"rope_theta", 10000000.0},
                      {"intermediate_size", 14336}};
    require(parse_mtp_config(mtp, config).intermediate_size == 14336, "the dense MTP block");
    for (const auto& [edit, label] :
         std::vector<std::pair<std::function<void(Json&)>, std::string>>{
             {[](Json& c) { c["num_experts"] = 512; }, "an MTP MoE"},
             {[](Json& c) { c["intermediate_size"] = 100; }, "an MTP MLP off 256-value blocks"},
             {[](Json& c) { c["architectures"] = Json::array({"Qwen4ExpMTP"}); },
              "the MoE MTP architecture"},
         }) {
        Json c = mtp;
        edit(c);
        bool refused = false;
        try {
            (void)parse_mtp_config(c, config);
        } catch (const ninfer::artifact::ArtifactError&) { refused = true; }
        require(refused, "not refused: " + label);
    }
    return 0;
}

int run() {
    const TextConfig config = parse_text_config(fixture());
    require(config.num_hidden_layers == 48 && config.layer_types.size() == 48, "48 blocks");
    std::size_t attention = 0;
    for (std::size_t i = 0; i < config.layer_types.size(); ++i) {
        const bool qsa = config.layer_types[i] == MixerKind::SparseAttention;
        require(qsa == (i % 4 == 3), "QSA every fourth block");
        attention += qsa;
    }
    require(attention == 12, "12 QSA blocks");
    require(config.ple_layers == std::vector<std::uint32_t>{1}, "PLE before block 1");
    require(config.eos_token_id == 248044 && !config.tie_word_embeddings, "EOS, untied head");
    require(config.indexer_block_budget() == 512, "512 selected blocks");
    require(config.ngram_heads() == 16, "16 n-gram heads");
    const NgramHashConstants constants = derive_ngram_hash_constants(config.ngram);
    require(constants.rows == 320001536ULL, "n-gram table rows");
    require(!config.fp32_direct && config.norm_unit_offset && !config.hc_prefill_inject &&
                config.gdn_a_is_log && !config.expert_output_scales,
            "existing artifacts retain their weight conventions");
    Json basalt                 = fixture();
    basalt["fp32_direct"]       = true;
    basalt["norm_unit_offset"]  = false;
    basalt["hc_prefill_inject"] = true;
    basalt["gdn_a_is_log"]      = false;
    basalt["expert_output_scales"] = true;
    const auto preserved        = parse_text_config(basalt);
    require(preserved.fp32_direct && !preserved.norm_unit_offset && preserved.hc_prefill_inject &&
                !preserved.gdn_a_is_log && preserved.expert_output_scales,
            "Basalt's stored weight conventions parse");
    for (const char* field : {"fp32_direct", "norm_unit_offset", "hc_prefill_inject",
                              "gdn_a_is_log", "expert_output_scales"}) {
        expect_refusal([&](Json& c) { c[field] = 1; }, std::string(field) + " must be boolean");
    }

    // A 3..5 source slice starts at a QSA block and contains no PLE. Local layer numbers
    // describe this stack; they do not imply the full checkpoint's every-fourth phase.
    Json slice = fixture();
    slice["num_hidden_layers"] = 2;
    slice["layer_types"] = Json::array({"full_attention", "linear_attention"});
    slice["ple_layers"] = Json::array();
    const auto sliced = parse_text_config(slice);
    require(sliced.num_hidden_layers == 2 && sliced.ple_layers.empty() &&
                sliced.layer_types.front() == MixerKind::SparseAttention,
            "non-prefix layer slice without PLE");
    slice["layer_types"] = Json::array({"linear_attention", "linear_attention"});
    slice["ple_layers"] = Json::array({0});
    require(parse_text_config(slice).ple_layers == std::vector<std::uint32_t>{0},
            "slice retains PLE at local block zero");

    expect_refusal([](Json& c) { c["hidden_size"] = 2048; }, "another hidden width");
    // An expert-pruned release keeps 256 of the 512 experts; fewer than the ten a token selects,
    // or more than 512, are refused.
    Json pruned           = fixture();
    pruned["num_experts"] = 256;
    require(parse_text_config(pruned).num_experts == 256, "an expert-pruned release parses");
    expect_refusal([](Json& c) { c["num_experts"] = 9; }, "fewer experts than a token selects");
    expect_refusal([](Json& c) { c["num_experts"] = 1024; }, "more experts than implemented");
    expect_refusal([](Json& c) { c["ple_layers"] = Json::array({3}); }, "PLE on a QSA block");
    expect_refusal([](Json& c) { c.erase("hc_lowrank"); }, "a missing field");
    expect_refusal([](Json& c) { c["architectures"] = Json::array({"Qwen3_5ForCausalLM"}); },
                   "another architecture");
    expect_refusal([](Json& c) { c["eos_token_id"] = 248320; }, "EOS past the vocabulary");
    expect_refusal([](Json& c) { c["rope_parameters"]["mrope_section"] = Json::array({11, 11, 11}); },
                   "MRoPE sections past the rotary width");

    // Edge0's latent MoE (tools/convert/qwen4_exp.py latent_moe_config): absent for Flash-Next's
    // own experts; widths in whole 256-value steps, one to sixteen groups, both flags boolean.
    require(!config.latent_moe, "Flash-Next's experts are not latent");
    Json latent          = fixture();
    latent["latent_moe"] = {{"width", 1792}, {"groups", 8}, {"expert_scales", true},
                            {"column_scale", false}};
    const auto edge0 = parse_text_config(latent);
    expect_refusal(
        [&](Json& c) {
            c["latent_moe"]           = latent["latent_moe"];
            c["expert_output_scales"] = true;
        },
        "ordinary expert scales on a latent MoE");
    require(edge0.latent_moe && edge0.latent_moe->width == 1792 && edge0.latent_moe->groups == 8 &&
                edge0.latent_moe->expert_scales && !edge0.latent_moe->column_scale,
            "Edge0's latent MoE parses");
    const auto refuses_latent = [&](const std::function<void(Json&)>& edit, const std::string& label) {
        expect_refusal(
            [&](Json& c) {
                c["latent_moe"] = latent["latent_moe"];
                edit(c["latent_moe"]);
            },
            label);
    };
    refuses_latent([](Json& l) { l["width"] = 1800; }, "a latent width off 256-value steps");
    refuses_latent([](Json& l) { l["width"] = 0; }, "a zero latent width");
    refuses_latent([](Json& l) { l["groups"] = 0; }, "no groups");
    refuses_latent([](Json& l) { l["groups"] = 17; }, "more groups than implemented");
    refuses_latent([](Json& l) { l["expert_scales"] = 1; }, "a non-boolean flag");
    refuses_latent([](Json& l) { l.erase("column_scale"); }, "a missing latent field");
    refuses_latent([](Json& l) { l["rank"] = 4; }, "an unknown latent field");

    // The MTP block: the full model's 512 experts beside an expert-pruned text model, the text
    // model's theta, and nothing else.
    const Json mtp         = {{"architectures", Json::array({"Qwen4ExpMTP"})},
                              {"rope_theta", 10000000.0},
                              {"num_experts", 512}};
    const TextConfig coder = parse_text_config(pruned);
    require(parse_mtp_config(mtp, coder).num_experts == 512, "the MTP block keeps its experts");
    const auto refuses_mtp = [&](const std::function<void(Json&)>& edit, const std::string& label) {
        Json c = mtp;
        edit(c);
        bool refused = false;
        try {
            (void)parse_mtp_config(c, config);
        } catch (const ninfer::artifact::ArtifactError&) { refused = true; }
        require(refused, "not refused: " + label);
    };
    refuses_mtp([](Json& c) { c["rope_theta"] = 1000000.0; }, "another MTP theta");
    refuses_mtp([](Json& c) { c["num_experts"] = 9; }, "an MTP block with too few experts");
    refuses_mtp([](Json& c) { c["architectures"] = Json::array({"Qwen3_5MTP"}); },
                "another MTP architecture");
    refuses_mtp([](Json& c) { c["hidden"] = 1; }, "an unknown MTP field");
    expect_refusal([](Json& c) { c["hidden_size"] = 5120; c["hc_lowrank"] = 640; },
                   "Flash-Next at the dense model's width");
    return dense();
}

} // namespace

int main() {
    try {
        const int result = run();
        std::cout << "PASS qwen4_exp config\n";
        return result;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
