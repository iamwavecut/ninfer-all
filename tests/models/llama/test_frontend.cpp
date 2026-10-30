#include "models/qwen3_5/frontend/tokenizer.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
using namespace ninfer::models::qwen3_5::frontend;

std::string read(const std::filesystem::path& p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("missing " + p.string());
    return {std::istreambuf_iterator<char>(f), {}};
}

int main(int argc, char** argv) {
    try {
        const auto fixture   = nlohmann::json::parse(read(
            argc > 1 ? argv[1]
                     : std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/llama/metaspace.json"));
        const std::string tj = fixture.at("tokenizer").dump(),
                          tc = fixture.value("tokenizer_config", nlohmann::json::object()).dump();
        Tokenizer tokenizer({.tokenizer_json         = tj,
                             .tokenizer_config_json  = tc,
                             .generation_config_json = R"({"eos_token_id":0})"});
        for (const auto& c : fixture.at("cases")) {
            auto text = c.at("text").get<std::string>();
            auto ids  = tokenizer.encode(text);
            if (ids != c.at("ids").get<std::vector<int>>())
                throw std::runtime_error("encode differs: " + text);
            if (tokenizer.decode(ids) != c.at("decoded").get<std::string>())
                throw std::runtime_error("decode differs: " + text);
            // Every stable frontier is an independently re-encoded prefix of this input.
            std::vector<std::size_t> boundaries(text.size() + 1);
            for (std::size_t i = 0; i < boundaries.size(); ++i) boundaries[i] = i;
            auto encoded = tokenizer.encode_with_boundaries(text, boundaries);
            if (encoded.input_ids != ids) throw std::runtime_error("boundary encoding differs");
            for (std::size_t i = 0; i < boundaries.size(); ++i)
                if (encoded.boundaries[i].exact_frontier) {
                    auto n = *encoded.boundaries[i].exact_frontier;
                    // Byte fallback frontiers inside a Unicode scalar cannot be encoded
                    // independently.
                    if (i < text.size() && (static_cast<unsigned char>(text[i]) & 0xc0) == 0x80)
                        continue;
                    auto prefix = tokenizer.encode(std::string_view(text).substr(0, i));
                    if (prefix.size() != n ||
                        !std::equal(prefix.begin(), prefix.end(), ids.begin()))
                        throw std::runtime_error("incorrect exact frontier");
                }
        }
        std::cout << fixture.at("cases").size() << " HF tokenizer cases passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
