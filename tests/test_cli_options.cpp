#include "options.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

ninfer::cli::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}


// --config files: layers of options below the command line (product/option_layers.h).
int launch_configuration_tests() {
    int failures         = 0;
    const auto directory = std::filesystem::temp_directory_path() / "ninfer_cli_config_test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto write = [&](const char* name, const char* text) {
        const auto path = directory / name;
        std::ofstream(path) << text;
        return path.string();
    };
    const std::string base = write("base.ini", "; a launch profile\n"
                                               "model = from-file.ninfer\n"
                                               "prompt = \"from file\"\n"
                                               "max-context = 4096\n"
                                               "spec = mtp\n"
                                               "draft-tokens = 4\n"
                                               "devices = 0,1\n"
                                               "stop = alpha\n"
                                               "thinking-budget = 512\n"
                                               "cuda-graph = false\n"
                                               "vision = true\n"
                                               "vision-residency = overlay\n");
    const auto from_file = parse({"ninfer", "--config", base});
    failures += check(from_file.artifact_path == "from-file.ninfer" &&
                          from_file.prompt == "from file" && from_file.max_context == 4096 &&
                          from_file.speculative.backend == ninfer::SpeculativeBackend::Mtp &&
                          from_file.speculative.draft_tokens == 4 &&
                          from_file.devices == std::vector<int>({0, 1}) &&
                          from_file.stop_strings.size() == 1 && from_file.thinking_budget == 512U &&
                          !from_file.use_cuda_graph && from_file.enable_vision &&
                          from_file.vision_residency == ninfer::VisionResidency::Overlay,
                      "a configuration file's options, artifact and false switch were not applied");

    const auto overridden =
        parse({"ninfer", "cli.ninfer", "--config", base, "--max-context", "8192", "--device", "1",
               "--stop", "beta", "--stop", "gamma", "--cuda-graph", "--spec", "none",
               "--no-thinking", "--no-vision", "--prompt", "from cli"});
    failures += check(overridden.artifact_path == "cli.ninfer" && overridden.prompt == "from cli" &&
                          overridden.max_context == 8192,
                      "the command line's artifact and values must replace the file's");
    failures += check(overridden.device == 1 && overridden.devices.empty(),
                      "--device must replace a file's --devices");
    failures += check(overridden.stop_strings.size() == 2 &&
                          overridden.stop_strings[0].text == "beta",
                      "a repeatable option on the command line must replace the file's list");
    failures += check(overridden.use_cuda_graph, "--cuda-graph must undo a file's false switch");
    failures += check(overridden.speculative.backend == ninfer::SpeculativeBackend::None &&
                          overridden.speculative.draft_tokens == 0,
                      "--spec none must also take the file's draft settings away");
    failures += check(overridden.enable_thinking == false && !overridden.thinking_budget,
                      "--no-thinking must take the file's thinking budget away");
    failures += check(!overridden.enable_vision &&
                          overridden.vision_residency == ninfer::VisionResidency::Resident,
                      "--no-vision must take the file's vision residency away");

    const std::string later = write("later.ini", "max-context = 16384\nspec = dflash2\n");
    const auto two_files    = parse({"ninfer", "--config", base, "--config", later});
    failures += check(two_files.max_context == 16384 &&
                          two_files.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
                          two_files.speculative.draft_tokens == 4,
                      "a later file must override an earlier one and keep its other options");

    // A DSpine file's drafter: another backend on the command line takes it away.
    const std::string spine = write("spine.ini", "spec = dspine\ndraft-model = drafter.ninfer\n"
                                                 "draft-tokens = 7\n");
    const auto from_spine   = parse({"ninfer", "m.ninfer", "--prompt", "x", "--config", spine});
    failures += check(from_spine.speculative.backend == ninfer::SpeculativeBackend::DSpine &&
                          from_spine.speculative.draft_model == "drafter.ninfer" &&
                          from_spine.speculative.draft_tokens == 7,
                      "a DSpine file's drafter was not applied");
    const auto spine_to_mtp =
        parse({"ninfer", "m.ninfer", "--prompt", "x", "--config", spine, "--spec", "mtp"});
    failures += check(spine_to_mtp.speculative.backend == ninfer::SpeculativeBackend::Mtp &&
                          spine_to_mtp.speculative.draft_model.empty() &&
                          spine_to_mtp.speculative.draft_tokens == 7,
                      "--spec mtp must take a file's DSpine drafter away and keep its drafts");
    const auto spine_off =
        parse({"ninfer", "m.ninfer", "--prompt", "x", "--config", spine, "--spec", "none"});
    failures += check(spine_off.speculative.backend == ninfer::SpeculativeBackend::None &&
                          spine_off.speculative.draft_model.empty() &&
                          spine_off.speculative.draft_tokens == 0,
                      "--spec none must take a file's DSpine drafter and drafts away");

    // A combination one layer gives is the user's error, not an override.
    const std::string contradictory = write("contradictory.ini", "spec = none\ndraft-tokens = 4\n");
    failures += check(rejects([&] {
                          (void)parse({"ninfer", "m.ninfer", "--prompt", "x", "--config",
                                       contradictory});
                      }),
                      "one file's contradictory options were accepted");
    failures += check(rejects([] {
                          (void)parse({"ninfer", "m.ninfer", "--prompt", "x", "--spec", "none",
                                       "--draft-tokens", "4"});
                      }),
                      "the command line's contradictory options were accepted");
    const std::string sections = write("sections.ini", "[model]\nmax-context = 1\n");
    const std::string dashed   = write("dashed.ini", "--max-context = 1\n");
    const std::string nested   = write("nested.ini", "config = other.ini\n");
    for (const std::string& file : {sections, dashed, nested, (directory / "missing.ini").string()}) {
        failures += check(rejects([&] {
                              (void)parse({"ninfer", "m.ninfer", "--prompt", "x", "--config", file});
                          }),
                          "a malformed, missing or nesting configuration file was accepted");
    }
    failures += check(rejects([] { (void)parse({"ninfer", "m.ninfer", "--config"}); }),
                      "--config without a file was accepted");
    failures += check(parse({"ninfer", "--config", (directory / "missing.ini").string(), "--help"})
                          .help_requested,
                      "--help must not read the configuration files");
    std::filesystem::remove_all(directory);
    return failures;
}

} // namespace

int run_tests() {
    int failures = 0;
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--gpu-clock-floor",
                             "1500"}).gpu_clock_floor_mhz == 1500 &&
                          parse({"ninfer-cli", "model.ninfer", "--prompt", "x"})
                                  .gpu_clock_floor_mhz == 0,
                      "--gpu-clock-floor was not parsed, or is not off by default");
    for (const char* value : {"0", "5001", "fast"}) {
        failures += check(rejects([&] {
                              (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
                                           "--gpu-clock-floor", value});
                          }),
                          "--gpu-clock-floor accepted 0, 5001 or a word");
    }
    failures += check(!parse({"ninfer-cli", "model.ninfer", "--prompt", "x"})
                           .speculative.draft_min_p, "draft confidence must be opt-in");
    for (const auto* floor : {"0", "0.3", "1"}) {
        failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", "mtp",
                                "--draft-tokens", "4", "--draft-min-p", floor})
                              .speculative.draft_min_p == std::stof(floor),
                          "CLI did not preserve draft confidence");
    }
    for (const auto* floor : {"-0.1", "1.1", "nan", "inf", "0.3x"}) {
        failures += check(rejects([&] {
            (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", "mtp",
                         "--draft-tokens", "4", "--draft-min-p", floor});
        }), "CLI accepted invalid draft confidence");
    }
    failures += check(rejects([&] {
        (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", "dflash",
                     "--draft-tokens", "4", "--draft-min-p", "0.3"});
    }), "CLI accepted draft confidence outside MTP and DFlash2");
    // Qwen3.8-Flash-Next's DFlash2 drafter verifies through its first draft at or below the floor.
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", "dflash2",
                             "--draft-tokens", "4", "--draft-min-p", "0.3"})
                              .speculative.draft_min_p == 0.3f,
                      "CLI did not preserve the DFlash2 confidence floor");
    // Without --spec the floor is kept for the MTP the Engine chooses, zero included.
    for (const auto* floor : {"0", "0.3"}) {
        const auto automatic =
            parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--draft-min-p", floor});
        failures += check(automatic.speculative.automatic &&
                              automatic.speculative.draft_min_p == std::stof(floor),
                          "CLI did not keep draft confidence for automatic speculation");
    }
    failures += check(rejects([] {
        (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", "none",
                     "--draft-min-p", "0.3"});
    }), "CLI accepted draft confidence without MTP");
    {
        // DeepSeek's DSpark drafter: its whole block without --draft-tokens, no copy drafts beside
        // it unless asked, and neither a proposal head nor MTP's confidence floor.
        const auto block = parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", "dspark"});
        const auto three = parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", "dspark",
                                  "--draft-tokens", "3"});
        failures += check(block.speculative.backend == ninfer::SpeculativeBackend::DSpark &&
                              block.speculative.draft_tokens == 0 &&
                              block.speculative.ngram_draft_tokens == 0 &&
                              three.speculative.draft_tokens == 3,
                          "CLI did not preserve --spec dspark");
        for (const std::vector<std::string> tail :
             {std::vector<std::string>{"--lm-head-draft"}, {"--draft-min-p", "0.3"},
              {"--draft-tokens", "16"}}) {
            std::vector<std::string> args{"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", "dspark"};
            args.insert(args.end(), tail.begin(), tail.end());
            failures += check(rejects([&] { (void)parse(args); }),
                              ("CLI accepted " + tail.front() + " beside --spec dspark").c_str());
        }
        // The drafter's kept experts: the Engine's default unless set, DSpark only.
        failures += check(!block.speculative.draft_experts, "--draft-experts must default");
        for (const auto* kept : {"0", "64"}) {
            const auto set = parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec",
                                    "dspark", "--draft-experts", kept});
            failures += check(set.speculative.draft_experts == std::stoul(kept),
                              "CLI did not preserve --draft-experts");
        }
        for (const std::vector<std::string> tail :
             {std::vector<std::string>{"--spec", "mtp", "--draft-tokens", "4", "--draft-experts", "64"},
              {"--draft-experts", "64"}, {"--spec", "dspark", "--draft-experts", "-1"}}) {
            std::vector<std::string> args{"ninfer-cli", "model.ninfer", "--prompt", "x"};
            args.insert(args.end(), tail.begin(), tail.end());
            failures += check(rejects([&] { (void)parse(args); }),
                              "CLI accepted an invalid --draft-experts");
        }
        // The confidence floor: the drafter's default unless set, 0..1, DSpark only.
        failures += check(!block.speculative.draft_confidence, "DSpark's floor must default");
        for (const auto* floor : {"0", "0.65", "1"}) {
            const auto set = parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec",
                                    "dspark", "--draft-confidence", floor});
            failures += check(set.speculative.draft_confidence == std::stof(floor),
                              "CLI did not preserve --draft-confidence");
        }
        for (const std::vector<std::string> tail :
             {std::vector<std::string>{"--spec", "dspark", "--draft-confidence", "1.5"},
              {"--spec", "dspark", "--draft-confidence", "nan"},
              {"--spec", "mtp", "--draft-tokens", "4", "--draft-confidence", "0.5"},
              {"--draft-confidence", "0.5"}}) {
            std::vector<std::string> args{"ninfer-cli", "model.ninfer", "--prompt", "x"};
            args.insert(args.end(), tail.begin(), tail.end());
            failures += check(rejects([&] { (void)parse(args); }),
                              "CLI accepted an invalid --draft-confidence");
        }
    }
    {
        // A DSpine drafter: the artifact --draft-model names, 1..15 drafts, either head and the
        // confidence floor; no copy drafts beside it unless asked.
        const auto spine = parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec",
                                  "dspine", "--draft-model", "drafter.ninfer", "--draft-tokens",
                                  "15", "--draft-min-p", "0.2", "--full-draft-head"});
        failures += check(spine.speculative.backend == ninfer::SpeculativeBackend::DSpine &&
                              spine.speculative.draft_model == "drafter.ninfer" &&
                              spine.speculative.draft_tokens == 15 &&
                              spine.speculative.draft_min_p == 0.2f &&
                              spine.speculative.proposal_head == ninfer::ProposalHead::Full &&
                              spine.speculative.ngram_draft_tokens == 0,
                          "CLI did not preserve --spec dspine");
        const auto listed = parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec",
                                   "dspine", "--draft-model", "drafter.ninfer", "--draft-tokens",
                                   "7", "--lm-head-draft"});
        failures += check(listed.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                          "CLI did not let --spec dspine draft with the proposal head");
        for (const std::vector<std::string> tail :
             {std::vector<std::string>{"--spec", "dspine", "--draft-tokens", "7"},
              {"--spec", "dspine", "--draft-model", "d.ninfer"},
              {"--spec", "dspine", "--draft-model", "d.ninfer", "--draft-tokens", "16"},
              {"--spec", "mtp", "--draft-tokens", "4", "--draft-model", "d.ninfer"},
              {"--draft-model", "d.ninfer"}}) {
            std::vector<std::string> args{"ninfer-cli", "model.ninfer", "--prompt", "x"};
            args.insert(args.end(), tail.begin(), tail.end());
            failures += check(rejects([&] { (void)parse(args); }),
                              "CLI accepted --draft-model or --spec dspine out of their contract");
        }
    }
    {
        // Without --spec the Engine chooses; --spec none and every backend are explicit.
        const auto unset = parse({"ninfer-cli", "model.ninfer", "--prompt", "x"}).speculative;
        const auto none  = parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", "none"}).speculative;
        const auto mtp   = parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", "mtp",
                                  "--draft-tokens", "4"}).speculative;
        failures += check(unset.automatic && unset.backend == ninfer::SpeculativeBackend::None &&
                              !none.automatic && none.backend == ninfer::SpeculativeBackend::None &&
                              !mtp.automatic && mtp.backend == ninfer::SpeculativeBackend::Mtp,
                          "CLI did not leave the backend to the Engine only without --spec");
    }
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
                             "--expert-residency", "host"}).hybrid_experts.dma_share.has_value() == false,
                      "CLI must default the expert DMA share to automatic");
    {
        // --kv-resident: unset (auto) leaves the choice to the Engine, all keeps every position.
        const auto resident = [&](std::vector<std::string> tail) {
            std::vector<std::string> args{"ninfer-cli", "model.ninfer", "--prompt", "x"};
            args.insert(args.end(), tail.begin(), tail.end());
            return parse(args).kv_resident_positions;
        };
        failures += check(!resident({}) && resident({"--kv-resident", "4096"}) == 4096U &&
                              resident({"--kv-resident", "all"}) == 0U &&
                              !resident({"--kv-resident", "4096", "--kv-resident", "auto"}),
                          "CLI did not preserve --kv-resident");
        for (const auto* value : {"0", "-1", "12x", "", "All"}) {
            failures += check(rejects([&] { (void)resident({"--kv-resident", value}); }),
                              "CLI accepted an invalid --kv-resident");
        }
    }
    for (const auto* share : {"0", "0.25", "1"}) {
        const auto hybrid = parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
            "--expert-residency", "host", "--expert-dma-share", share,
            "--expert-cpu-threads", "8", "--expert-cache-adaptive",
            "--expert-profile", "routes.json", "--expert-profile-out", "recorded.json"});
        failures += check(hybrid.hybrid_experts.dma_share == std::stof(share) &&
            hybrid.hybrid_experts.cpu_threads == 8 && hybrid.hybrid_experts.adaptive_cache &&
            hybrid.hybrid_experts.routing_profile == "routes.json" &&
            hybrid.hybrid_experts.record_profile == "recorded.json",
            "CLI did not preserve the native hybrid configuration");
    }
    {
        const auto automatic = parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
            "--expert-residency", "host", "--expert-dma-share", "auto"});
        const auto fixed_after = parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
            "--expert-dma-share", "auto", "--expert-dma-share", "0.5"});
        failures += check(!automatic.hybrid_experts.dma_share &&
                              fixed_after.hybrid_experts.dma_share == 0.5F,
                          "CLI did not preserve an automatic hybrid DMA share");
    }
    for (const auto* share : {"-0.1", "1.1", "nan", "inf", "0.5x", "Auto"}) {
        failures += check(rejects([&] {
            (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--expert-dma-share", share});
        }), "CLI accepted an invalid hybrid DMA share");
    }
    for (const auto* threads : {"0", "257", "-1", "1.5"}) {
        failures += check(rejects([&] {
            (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--expert-cpu-threads", threads});
        }), "CLI accepted an invalid hybrid CPU worker count");
    }
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "x"}).embedding_residency ==
                              ninfer::EmbeddingResidency::Device &&
                          parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
                                 "--embedding-residency", "host"})
                                  .embedding_residency == ninfer::EmbeddingResidency::Host,
                      "CLI did not default the embedding to the device or keep a host one");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
                                       "--embedding-residency", "disk"});
                      }),
                      "CLI accepted an invalid embedding residency");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--expert-residency",
                             "host"}).hybrid_experts.tail_skip_rank == 0 &&
                          parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
                                 "--expert-residency", "host", "--expert-tail-skip", "7"})
                                  .hybrid_experts.tail_skip_rank == 7,
                      "CLI did not keep every miss by default or preserve the tail skip");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "x"})
                              .hybrid_experts.prefill_buffers == 0 &&
                          parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
                                 "--expert-prefill-buffers", "1"})
                                  .hybrid_experts.prefill_buffers == 1 &&
                          parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
                                 "--expert-prefill-buffers", "2"})
                                  .hybrid_experts.prefill_buffers == 2,
                      "CLI did not leave the prefill expert buffers automatic by default or "
                      "preserve one or two");
    for (const auto* count : {"0", "3", "two"}) {
        failures += check(rejects([&] {
            (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--expert-prefill-buffers",
                         count});
        }), "CLI accepted an invalid prefill expert buffer count");
    }
    for (const auto* rank : {"0", "10", "-1", "7x"}) {
        failures += check(rejects([&] {
            (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--expert-tail-skip", rank});
        }), "CLI accepted an invalid expert tail skip");
    }
    for (const auto* budget : {"0", "32", "4096"}) {
        const auto disk = parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
                                 "--ngram-residency", "disk", "--ngram-ram-mib", budget});
        failures += check(disk.ngram_table.ram_budget_bytes == (std::stoull(budget) << 20U),
                          "CLI did not preserve the disk row-cache budget");
    }
    for (const auto* budget : {"-1", "1.5", "1073741825"}) {
        failures += check(rejects([&] {
            (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--ngram-ram-mib", budget});
        }), "CLI accepted an invalid row-cache budget");
    }
    {
        const auto defaults = parse({"ninfer-cli", "model.ninfer", "--prompt", "x"});
        failures += check(defaults.ngram_table.dedupe && defaults.ngram_table.read_ahead &&
                              defaults.prompt_routes.prefill_tail == ninfer::PrefillTail::Exact &&
                              defaults.prompt_routes.iq1m_expert_matrix,
                          "CLI did not default to deduplicated, read-ahead Engram rows, the exact "
                          "prefill tail and IQ1_M experts on the matrix kernel");
        const auto off = parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--ngram-dedupe",
                                "off", "--ngram-read-ahead", "off", "--prefill-tail", "full",
                                "--iq1m-experts", "vector"});
        failures += check(!off.ngram_table.dedupe && !off.ngram_table.read_ahead &&
                              off.prompt_routes.prefill_tail == ninfer::PrefillTail::Full &&
                              !off.prompt_routes.iq1m_expert_matrix,
                          "CLI did not preserve the n-gram read and prompt route switches");
        const auto rows = parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--prefill-tail",
                                 "512"});
        failures += check(rows.prompt_routes.prefill_tail == ninfer::PrefillTail::Rows &&
                              rows.prompt_routes.prefill_tail_rows == 512,
                          "CLI did not preserve a prefill tail row count");
    }
    for (const auto* tail : {"127", "0", "exactly", "-5", "2414x"}) {
        failures += check(rejects([&] {
            (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--prefill-tail", tail});
        }), "CLI accepted an invalid prefill tail");
    }
    for (const auto& option : {std::pair{"--iq1m-experts", "mmq"},
                               std::pair{"--ngram-dedupe", "yes"},
                               std::pair{"--ngram-read-ahead", "1"}}) {
        failures += check(rejects([&] {
            (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x", option.first,
                         option.second});
        }), "CLI accepted an invalid on/off or kernel value");
    }
    for (const auto* backend : {"mtp", "dflash", "dflash2"}) {
        for (unsigned width = 0; width <= 63; ++width) {
            const auto mixed =
                parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", backend,
                       "--draft-tokens", "5", "--ngram-draft-tokens", std::to_string(width)});
            failures += check(mixed.speculative.draft_tokens == 5 &&
                                  mixed.speculative.ngram_draft_tokens == width,
                              "CLI failed to preserve independent neural/ngram widths");
        }
        failures +=
            check(rejects([&] {
                      (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", backend,
                                   "--draft-tokens", "5", "--ngram-draft-tokens", "64"});
                  }),
                  "CLI admitted unsupported ngram width");
        failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", backend,
                                 "--draft-tokens", "3"})
                                  .speculative.ngram_draft_tokens == 15,
                          "CLI ngram drafting is not on by default beside a drafter");
    }
    failures += check(
        parse({"ninfer-cli", "model.ninfer", "--prompt", "x"}).speculative.ngram_draft_tokens == 0,
        "CLI ngram default needs a neural backend");
    failures += check(ninfer::cli::usage_text("ninfer-cli").find("--ngram-draft-tokens 0..63") !=
                          std::string::npos,
                      "CLI help has a stale ngram width limit");
    failures +=
        check(parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--ngram-draft-tokens", "0"})
                      .speculative.ngram_draft_tokens == 0,
              "explicit ngram-off must not require a neural backend");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", "mtp",
                             "--draft-tokens", "3", "--ngram-draft-tokens", "0"})
                              .speculative.ngram_draft_tokens == 0,
                      "explicit ngram-off beside a drafter was overridden");
    const auto ngram = parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", "dflash2",
                              "--draft-tokens", "5", "--ngram-draft-tokens", "15"});
    failures +=
        check(ngram.speculative.ngram_draft_tokens == 15 && ngram.speculative.draft_tokens == 5,
              "CLI ngram widths changed");
    const auto mtp_ngram = parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", "mtp",
                                  "--draft-tokens", "5", "--ngram-draft-tokens", "1"});
    failures += check(mtp_ngram.speculative.draft_tokens == 5 &&
                          mtp_ngram.speculative.ngram_draft_tokens == 1,
                      "CLI narrow ngram/wide MTP pair changed");
    failures +=
        check(rejects([] {
                  (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", "mtp",
                               "--draft-tokens", "16", "--ngram-draft-tokens", "15"});
              }),
              "CLI admitted unsupported MTP neural width");
    const ninfer::cli::Options configured =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "37"});
    failures += check(configured.thinking_budget == 37,
                      "--thinking-budget did not preserve its positive value");
    failures +=
        check(ninfer::cli::usage_text("ninfer-cli").find("--thinking-budget") != std::string::npos,
              "CLI help omits --thinking-budget");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "0"});
                      }),
                      "zero --thinking-budget was accepted");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "8", "--no-thinking"});
                      }),
                      "--thinking-budget was accepted with --no-thinking");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--vision",
                             "--vision-offload", "on"})
                              .vision_residency == ninfer::VisionResidency::Overlay,
                      "--vision-offload on is not an alias of overlay residency");
    {
        const ninfer::cli::Options cpu =
            parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--vision-cpu"});
        failures +=
            check(cpu.enable_vision && cpu.vision_residency == ninfer::VisionResidency::Cpu &&
                      cpu.vision_max_merged_tokens == 256U,
                  "--vision-cpu did not select CPU residency at 256 merged tokens");
    }
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--vision",
                                       "--vision-offload", "maybe"});
                      }),
                      "--vision-offload maybe was accepted");
    const ninfer::cli::Options headroom = parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
                                                 "--kv-capacity", "auto", "--vram-headroom-mib",
                                                 "256"});
    failures += check(headroom.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                          headroom.kv_capacity.automatic_headroom_bytes == (256ULL << 20),
                      "--vram-headroom-mib is not an alias of --kv-headroom-mib");
    const ninfer::cli::Options with_effort =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "8",
               "--reasoning-effort", "medium"});
    failures += check(with_effort.thinking_budget == 8 && with_effort.reasoning_effort,
                      "thinking budget did not coexist with reasoning effort");
    failures += check(
        configured.reasoning_loop == ninfer::ReasoningLoopAction::Off &&
            parse(
                {"ninfer-cli", "model.ninfer", "--prompt", "hello", "--reasoning-loop", "conclude"})
                    .reasoning_loop == ninfer::ReasoningLoopAction::Conclude &&
            parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--reasoning-loop", "stop"})
                    .reasoning_loop == ninfer::ReasoningLoopAction::Stop,
        "--reasoning-loop did not select its action (or was not off by default)");
    failures +=
        check(ninfer::cli::usage_text("ninfer-cli").find("--reasoning-loop") != std::string::npos,
              "CLI help omits --reasoning-loop");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--reasoning-loop", "on"});
                      }),
                      "--reasoning-loop on was accepted");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--reasoning-loop", "stop", "--no-thinking"});
                      }),
                      "--reasoning-loop was accepted with --no-thinking");
    const ninfer::cli::Options dflash_vision =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--vision", "--spec", "dflash",
               "--draft-tokens", "7"});
    failures += check(dflash_vision.enable_vision &&
                          dflash_vision.speculative.backend == ninfer::SpeculativeBackend::DFlash &&
                          dflash_vision.speculative.draft_tokens == 7,
                      "CLI did not preserve the combined DFlash and Vision startup features");
    for (const auto k : {1U, 2U, 7U, 15U}) {
        const auto dflash2 = parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                    "dflash2", "--draft-tokens", std::to_string(k)});
        failures += check(dflash2.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
                              dflash2.speculative.draft_tokens == k,
                          "CLI did not preserve the DFlash2 draft count");
    }
    for (const auto k : {0U, 16U}) {
        failures +=
            check(rejects([&] {
                      (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                   "dflash2", "--draft-tokens", std::to_string(k)});
                  }),
                  "CLI accepted an unsupported DFlash2 draft count");
    }
    const ninfer::cli::Options nvfp4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.kv_cache == ninfer::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const ninfer::cli::Options k8v4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "k8v4"});
    failures += check(k8v4.kv_cache == ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    const std::string help = ninfer::cli::usage_text("ninfer-cli");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "x"}).rope_yarn_factor ==
                          1.0F,
                      "CLI YaRN factor must default to 1");
    for (const auto* factor : {"1", "2.5", "4"}) {
        const auto yarn =
            parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--rope-yarn-factor", factor});
        failures += check(yarn.rope_yarn_factor == std::stof(factor) && yarn.max_context == 2048 &&
                              yarn.kv_capacity.explicit_tokens == 2048,
                          "a YaRN factor must not grow the default context or KV");
    }
    for (const auto* factor : {"0", "0.99", "4.01", "-1", "nan", "inf", "-inf", "1e999", "2x", ""}) {
        failures += check(rejects([&] {
                              (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
                                           "--rope-yarn-factor", factor});
                          }),
                          "invalid YaRN factor accepted");
    }
    failures += check(rejects([] {
                          (void)parse(
                              {"ninfer-cli", "model.ninfer", "--prompt", "x", "--rope-yarn-factor"});
                      }),
                      "missing YaRN factor accepted");
    // Every option the parser accepts is described in the grouped help.
    for (const std::string_view flag : {"--chat-template",
                                        "--device",
                                        "--devices",
                                        "--draft-tokens",
                                        "--embedding-q4",
                                        "--embedding-q6",
                                        "--embedding-residency",
                                        "--frequency-penalty",
                                        "--full-draft-head",
                                        "--gdn-state-fp16",
                                        "--gpu-clock-floor",
                                        "--greedy",
                                        "--gumbel-sampling",
                                        "--json",
                                        "--json-schema",
                                        "--kv-capacity",
                                        "--kv-dtype",
                                        "--kv-headroom-mib",
                                        "--lm-head-draft",
                                        "--lm-head-q4",
                                        "--lm-head-q6",
                                        "--log-colours",
                                        "--log-level",
                                        "--lookup-ngram",
                                        "--max-context",
                                        "--max-new",
                                        "--messages",
                                        "--min-p",
                                        "--mlp-a8-decode",
                                        "--mtp-attention-window",
                                        "--mtp-experts-q4",
                                        "--ngram-draft-tokens",
                                        "--ngram-min-match",
                                        "--no-cuda-graph",
                                        "--no-prefill-a8",
                                        "--no-prefill-cublas-projections",
                                        "--no-thinking",
                                        "--post-thinking",
                                        "--post-thinking-sampler",
                                        "--post-thinking-temperature",
                                        "--post-thinking-top-k",
                                        "--post-thinking-top-p",
                                        "--prefill-chunk",
                                        "--prefill-cublas",
                                        "--presence-penalty",
                                        "--print-token-ids",
                                        "--prompt",
                                        "--raw-output",
                                        "--reasoning-effort",
                                        "--reasoning-stop",
                                        "--rope-yarn",
                                        "--rope-yarn-factor",
                                        "--seed",
                                        "--spec",
                                        "--stage-layers",
                                        "--stop",
                                        "--stop-token-id",
                                        "--temperature",
                                        "--thinking-budget",
                                        "--top-k",
                                        "--top-p",
                                        "--vision",
                                        "--vision-max-merged",
                                        "--vision-offload",
                                        "--vision-residency",
                                        "--vram-headroom-mib",
                                        "--wddm-evictable-budget"}) {
        const std::string message = "CLI help omits " + std::string(flag);
        failures += check(help.find(flag) != std::string::npos, message.c_str());
    }
    for (const char* section :
         {"INPUT", "CONTEXT", "KV CACHE", "SPECULATIVE DECODING", "PRECISION & KERNELS", "SAMPLING",
          "THINKING", "OUTPUT", "VISION", "LOGGING"}) {
        failures += check(help.find(section) != std::string::npos, "CLI help omits a category");
    }
    failures +=
        check(help.find("nvfp4") != std::string::npos && help.find("k8v4") != std::string::npos,
              "CLI help omits a production KV storage mode");
    const ninfer::cli::Options route_defaults =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello"});
    failures += check(!route_defaults.prefill_cublas && route_defaults.prefill_cublas_projections &&
                          route_defaults.speculative.lookup_ngram == 0,
                      "the cuBLAS prefill route or context lookup is on by default");
    const ninfer::cli::Options route =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec", "mtp", "--draft-tokens",
               "3", "--lookup-ngram", "5", "--prefill-cublas", "--no-prefill-cublas-projections"});
    failures += check(route.prefill_cublas && !route.prefill_cublas_projections &&
                          route.speculative.lookup_ngram == 5 &&
                          route.speculative.backend == ninfer::SpeculativeBackend::Mtp &&
                          route.speculative.draft_tokens == 3,
                      "CLI did not parse the cuBLAS prefill and context-lookup controls");
    // Without --spec, --draft-tokens sizes a context lookup (Flint, DeepSeek-V4.1); alone it is refused.
    const ninfer::cli::Options lookup_only = parse(
        {"ninfer-cli", "model.ninfer", "--prompt", "hello", "--lookup-ngram", "4", "--draft-tokens", "5"});
    failures += check(lookup_only.speculative.backend == ninfer::SpeculativeBackend::None &&
                          lookup_only.speculative.lookup_ngram == 4 &&
                          lookup_only.speculative.draft_tokens == 5,
                      "CLI refused or lost a context lookup's --draft-tokens");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--draft-tokens", "5"});
                      }),
                      "CLI accepted --draft-tokens without --spec or --lookup-ngram");
    for (const char* flag : {"--prefill-cublas", "--no-prefill-cublas-projections", "--lookup-ngram"}) {
        failures += check(help.find(flag) != std::string::npos,
                          "CLI help omits an accepted prefill or drafting control");
    }
    const ninfer::cli::Options logging =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--log-level", "debug"});
    failures += check(logging.log_level == ninfer::product::LogLevel::Debug,
                      "CLI log level was not parsed");
    failures += check(help.find("--log-level") != std::string::npos,
                      "CLI help omits the log-level control");
    failures += check(help.find("--log-colours") != std::string::npos,
                      "CLI help omits the log-colours control");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--log-level", "verbose"});
                      }),
                      "CLI accepted an unknown log level");
    failures +=
        check(rejects([] {
                  (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--top-k", "21"});
              }),
              "CLI accepted top_k beyond the executable candidate domain");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--device",
                                       "1", "--devices", "0,1"});
                      }),
                      "--device and --devices were accepted together");
    failures +=
        check(rejects([] {
                  (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--devices",
                               "4294967296"});
              }),
              "--devices silently narrowed an out-of-range id instead of rejecting it");
    const ninfer::cli::Options same_card = parse(
        {"ninfer-cli", "model.ninfer", "--prompt", "hello", "--devices", "0,0"});
    failures += check(same_card.devices.size() == 2 && same_card.devices[0] == 0 &&
                           same_card.devices[1] == 0,
                      "--devices 0,0 was not accepted for single-card split coverage");
    const ninfer::cli::Options three = parse(
        {"ninfer-cli", "model.ninfer", "--prompt", "hello", "--devices", "0,1,2"});
    failures += check(three.devices.size() == 3, "--devices did not accept more than two stages");
    const ninfer::cli::Options staged = parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                               "--devices", "0,0", "--stage-layers", "20,44"});
    failures += check(staged.stage_layers == std::vector<std::uint32_t>({20, 44}),
                      "--stage-layers did not reach the options");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--stage-layers", "20,44"});
                      }),
                      "--stage-layers without a multi-device split was accepted");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--devices", "0,0", "--stage-layers", "0,64"});
                      }),
                      "a stage with no layers was accepted");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--devices", "0,1",
                             "--tensor-parallel"})
                          .tensor_parallel,
                      "--tensor-parallel did not reach the options");
    failures += check(!parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--devices", "0,1"})
                           .tensor_parallel,
                      "tensor parallelism is not off by default");
    for (const std::vector<std::string>& devices :
         {std::vector<std::string>{}, std::vector<std::string>{"--devices", "1"}}) {
        failures += check(rejects([&] {
                              std::vector<std::string> args{"ninfer-cli", "model.ninfer", "--prompt",
                                                            "hello", "--tensor-parallel"};
                              args.insert(args.end(), devices.begin(), devices.end());
                              (void)parse(args);
                          }),
                          "--tensor-parallel without several devices was accepted");
    }
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--devices", "0,1",
                             "--tensor-parallel", "--tensor-split", "3,1.5"})
                              .tensor_split == std::vector<double>{3.0, 1.5},
                      "--tensor-split did not reach the options");
    for (const std::vector<std::string>& split :
         {std::vector<std::string>{"--tensor-parallel", "--tensor-split", "1,2,3"},
          std::vector<std::string>{"--tensor-split", "1,1"},
          std::vector<std::string>{"--tensor-parallel", "--tensor-split", "1,0"},
          std::vector<std::string>{"--tensor-parallel", "--tensor-split", "1,"},
          std::vector<std::string>{"--tensor-parallel", "--tensor-split", "1,x"}}) {
        failures += check(rejects([&] {
                              std::vector<std::string> args{"ninfer-cli", "model.ninfer", "--prompt",
                                                            "hello", "--devices", "0,1"};
                              args.insert(args.end(), split.begin(), split.end());
                              (void)parse(args);
                          }),
                          "a malformed or misplaced --tensor-split was accepted");
    }
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--devices", "0,1",
                             "--tensor-parallel", "--tensor-attention", "positions"})
                              .tensor_attention == ninfer::TensorAttention::Positions,
                      "--tensor-attention did not reach the options");
    failures += check(rejects([&] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--tensor-attention", "positions"});
                      }),
                      "--tensor-attention positions without --tensor-parallel was accepted");
    failures += check(rejects([&] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--devices",
                                       "0,1", "--tensor-parallel", "--tensor-attention", "rows"});
                      }),
                      "an unknown --tensor-attention mode was accepted");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--devices", "0,1",
                             "--tensor-parallel", "--tensor-parallel-transport", "peer-write"})
                              .tensor_parallel_transport ==
                          ninfer::TensorParallelTransport::PeerWrite,
                      "--tensor-parallel-transport did not reach the options");
    failures += check(rejects([&] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--tensor-parallel-transport", "host"});
                      }),
                      "--tensor-parallel-transport without --tensor-parallel was accepted");
    failures += check(rejects([&] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--devices",
                                       "0,1", "--tensor-parallel", "--tensor-parallel-transport",
                                       "peer"});
                      }),
                      "an unknown --tensor-parallel-transport was accepted");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--devices", "0,1",
                             "--tensor-parallel", "--tensor-parallel-prompt-partials", "bf16"})
                              .tensor_parallel_prompt_partials ==
                          ninfer::TensorParallelPartials::Bf16,
                      "--tensor-parallel-prompt-partials did not reach the options");
    failures +=
        check(parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--devices", "0,1",
                     "--tensor-parallel", "--tensor-parallel-prompt-partials", "int8"})
                      .tensor_parallel_prompt_partials == ninfer::TensorParallelPartials::Int8,
              "--tensor-parallel-prompt-partials int8 did not reach the options");
    failures += check(rejects([&] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--tensor-parallel-prompt-partials", "bf16"});
                      }),
                      "--tensor-parallel-prompt-partials without --tensor-parallel was accepted");
    failures += check(rejects([&] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--devices",
                                       "0,1", "--tensor-parallel",
                                       "--tensor-parallel-prompt-partials", "fp16"});
                      }),
                      "an unknown --tensor-parallel-prompt-partials was accepted");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                             "--invariant-attention"})
                          .invariant_attention,
                      "--invariant-attention did not reach the options");
    failures += check(parse({"ninfer", "model.ninfer", "--prompt", "x", "--spec", "mtp",
                             "--draft-tokens", "3", "--mtp-attention-window", "2048"})
                              .speculative.mtp_attention_window == 2048,
                      "CLI --mtp-attention-window not parsed");
    failures += check(rejects([] {
                          (void)parse({"ninfer", "model.ninfer", "--prompt", "x",
                                       "--mtp-attention-window", "2048"});
                      }),
                      "CLI --mtp-attention-window accepted without --spec mtp");
    failures += check(!parse({"ninfer", "model.ninfer", "--prompt", "x"}).post_thinking_sampling,
                      "CLI post-thinking sampling is not off by default");
    const auto post_thinking =
        parse({"ninfer", "model.ninfer", "--prompt", "x", "--post-thinking-top-k", "5",
               "--post-thinking-sampler", "temp=0.4,frequency=-1", "--greedy"});
    failures += check(post_thinking.post_thinking_sampling &&
                          post_thinking.post_thinking_sampling->top_k == 5 &&
                          post_thinking.post_thinking_sampling->frequency_penalty == -1.0F &&
                          post_thinking.post_thinking_sampling->temperature == 0.0F,
                      "CLI post-thinking fields or --greedy over them");
    failures += check(rejects([] {
                          (void)parse({"ninfer", "model.ninfer", "--prompt", "x",
                                       "--post-thinking-sampler", "top_k=3,"});
                      }),
                      "a trailing empty post-thinking field was accepted");
    const auto structured = parse({"ninfer", "model.ninfer", "--prompt", "hello", "--json"});
    failures +=
        check(structured.structured_output.kind == ninfer::StructuredOutputKind::JsonObject &&
                  structured.enable_thinking == false,
              "CLI JSON mode");
    const auto structured_reasoning = parse(
        {"ninfer", "model.ninfer", "--prompt", "hello", "--json", "--reasoning-effort", "medium"});
    failures += check(structured_reasoning.enable_thinking != false &&
                          structured_reasoning.reasoning_effort == ninfer::ReasoningEffort::Medium,
                      "CLI structured output preserves explicit reasoning");
    failures += check(
        rejects([] {
            (void)parse({"ninfer", "model.ninfer", "--prompt", "x", "--json", "--raw-output"});
        }),
        "raw JSON output should be rejected");
    failures += launch_configuration_tests();
    return failures == 0 ? 0 : 1;
}

int main() {
    try {
        return run_tests();
    } catch (const std::exception& error) {
        std::cerr << "Unexpected CLI test exception: " << error.what() << '\n';
        return 1;
    }
}
