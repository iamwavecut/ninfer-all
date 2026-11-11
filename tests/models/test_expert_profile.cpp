#include "models/expert_profile.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    using namespace ninfer::models;
    const auto directory = std::filesystem::temp_directory_path() /
        ("ninfer-expert-profile-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::filesystem::remove_all(path); } } cleanup{directory};
    try {
        const auto path = directory / "profile.json";
        ninfer::artifact::ArtifactId artifact{};
        const std::array<std::size_t, 2> widths{3, 2};
        const ExpertRouteCounts expected{{0, 17, std::numeric_limits<std::uint64_t>::max()}, {3, 4}};
        write_expert_profile(path, artifact, expected);
        if (read_expert_profile(path, artifact, widths) != expected) {
            throw std::runtime_error("expert profile did not preserve exact counts");
        }
        const auto rejects = [&](auto id, auto geometry) {
            try { (void)read_expert_profile(path, id, geometry); }
            catch (const std::exception&) { return; }
            throw std::runtime_error("expert profile accepted incompatible input");
        };
        auto wrong = artifact; wrong[0] = std::byte{1};
        rejects(wrong, widths);
        // A run without trailing layers (Flash-Next's MTP block) reads the leading ones; a model
        // with more layers than the profile, or other widths, is refused.
        if (read_expert_profile(path, artifact, std::array<std::size_t, 1>{3}) !=
            ExpertRouteCounts{expected[0]}) {
            throw std::runtime_error("expert profile did not serve its leading layers");
        }
        rejects(artifact, std::array<std::size_t, 3>{3, 2, 2});
        rejects(artifact, std::array<std::size_t, 1>{2});
        rejects(artifact, std::array<std::size_t, 2>{2, 3});
        for (const auto* counts : {"[[1,-2,3],[4,5]]", "[[1,2.5,3],[4,5]]", "[[1,2,3],[4]]"}) {
            std::ofstream(path) << "{\"version\":1,\"artifact_id\":\"00000000000000000000000000000000\",\"counts\":" << counts << '}';
            rejects(artifact, widths);
        }
        std::ofstream(path) << "{\"version\":1,";
        rejects(artifact, widths);
        // A family's built-in profiles are told apart by their bank widths: DeepSeek-V4.1 has one
        // for the release's 384 routed experts a block and one for the Coder build's 192.
        for (const std::size_t experts : {std::size_t{384}, std::size_t{192}}) {
            const std::vector<std::size_t> geometry(40, experts);
            const auto profile = builtin_expert_profile("deepseek-v4.1", geometry);
            if (!profile || profile->size() != geometry.size()) {
                throw std::runtime_error("no built-in DeepSeek-V4.1 profile for " +
                                         std::to_string(experts) + " experts a block");
            }
            for (const auto& layer : *profile) {
                const auto unrouted = [](auto count) { return count == 0; };
                if (layer.size() != experts || std::all_of(layer.begin(), layer.end(), unrouted)) {
                    throw std::runtime_error("built-in profile layer without routes");
                }
            }
        }
        if (builtin_expert_profile("deepseek-v4.1", std::vector<std::size_t>(40, 256)) ||
            builtin_expert_profile("deepseek-v4.1", std::vector<std::size_t>(41, 384))) {
            throw std::runtime_error("built-in profile served a geometry it was not recorded on");
        }
        // Flash-Next's was recorded with the MTP block's layer after the 48 text layers; a run
        // without the block (no drafts, DFlash2, perplexity) takes the text layers' counts.
        const auto with_mtp = builtin_expert_profile("qwen3.8-flash-next", std::vector<std::size_t>(49, 512));
        const auto text     = builtin_expert_profile("qwen3.8-flash-next", std::vector<std::size_t>(48, 512));
        if (!with_mtp || !text || text->size() != 48 ||
            !std::equal(text->begin(), text->end(), with_mtp->begin())) {
            throw std::runtime_error("no built-in Flash-Next profile for the text layers alone");
        }
        std::cout << "EXPERT_PROFILE_CONTRACT_PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
