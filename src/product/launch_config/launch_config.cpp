#include "product/launch_config/launch_config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ninfer::product {
namespace {

std::string trim(std::string_view text) {
    const auto space = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!text.empty() && space(static_cast<unsigned char>(text.front()))) {
        text.remove_prefix(1);
    }
    while (!text.empty() && space(static_cast<unsigned char>(text.back()))) {
        text.remove_suffix(1);
    }
    return std::string(text);
}

std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

std::string unquote(std::string value) {
    if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                              (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

} // namespace

ConfigFile parse_config_text(std::string_view text) {
    ConfigFile out;
    std::size_t line_number      = 0;
    ConfigFile::Section* current = nullptr;
    ConfigFile::Section top{.name = "*", .entries = {}};
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t newline = text.find('\n', start);
        const std::string line =
            trim(text.substr(start, newline == std::string_view::npos ? std::string_view::npos
                                                                       : newline - start));
        start = newline == std::string_view::npos ? text.size() + 1 : newline + 1;
        ++line_number;
        if (line.empty() || line.front() == ';' || line.front() == '#') { continue; }
        if (line.front() == '[') {
            if (line.back() != ']' || line.size() < 3) {
                throw std::invalid_argument("line " + std::to_string(line_number) +
                                            ": malformed section header");
            }
            const std::string name = trim(std::string_view(line).substr(1, line.size() - 2));
            const bool repeated =
                std::any_of(out.sections.begin(), out.sections.end(),
                            [&](const ConfigFile::Section& section) { return section.name == name; });
            if (repeated) {
                throw std::invalid_argument("line " + std::to_string(line_number) +
                                            ": duplicate section [" + name + "]");
            }
            out.sections.push_back({.name = name, .entries = {}});
            current = &out.sections.back();
            continue;
        }
        const auto equals = line.find('=');
        if (equals == std::string::npos) {
            throw std::invalid_argument("line " + std::to_string(line_number) +
                                        ": expected key = value");
        }
        std::string key   = lower(trim(std::string_view(line).substr(0, equals)));
        std::string value = unquote(trim(std::string_view(line).substr(equals + 1)));
        (current != nullptr ? *current : top).entries.emplace_back(std::move(key), std::move(value));
    }
    // Entries before any section (llama.cpp's `version = 1`) belong to the global section.
    if (!top.entries.empty()) {
        auto global = std::find_if(out.sections.begin(), out.sections.end(),
                                   [](const ConfigFile::Section& section) { return section.name == "*"; });
        if (global == out.sections.end()) {
            out.sections.insert(out.sections.begin(), std::move(top));
        } else {
            global->entries.insert(global->entries.begin(), top.entries.begin(), top.entries.end());
        }
    }
    return out;
}

ConfigFile read_config_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::invalid_argument("cannot read " + path.string()); }
    std::stringstream text;
    text << file.rdbuf();
    try {
        return parse_config_text(text.str());
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(path.string() + ": " + error.what());
    }
}

void append_config_option(std::vector<std::string>& arguments, std::string_view where,
                          std::string_view key, std::string_view value) {
    if (key.empty() || key.front() == '-') {
        throw std::invalid_argument(std::string(where) + ": option '" + std::string(key) +
                                    "' must be written without leading dashes");
    }
    const std::string word = lower(std::string(value));
    if (word == "true") {
        arguments.push_back("--" + std::string(key));
    } else if (word == "false") {
        arguments.push_back(key.starts_with("no-") ? "--" + std::string(key.substr(3))
                                                   : "--no-" + std::string(key));
    } else {
        arguments.push_back("--" + std::string(key));
        arguments.emplace_back(value);
    }
}

std::vector<std::vector<std::string>> CommandLine::layer_arguments() const {
    std::vector<std::vector<std::string>> out;
    out.reserve(layers.size());
    for (const ArgumentLayer& layer : layers) { out.push_back(layer.arguments); }
    return out;
}

CommandLine read_command_line(int argc, const char* const* argv, PositionalArtifact positional) {
    CommandLine out;
    out.program = argc > 0 && argv[0] != nullptr ? argv[0] : "";
    int first   = 1;
    if (positional == PositionalArtifact::Taken && argc > 1 && argv[1] != nullptr &&
        argv[1][0] != '\0' && argv[1][0] != '-') {
        out.artifact = argv[1];
        first        = 2;
    }
    ArgumentLayer command{.source = "command line", .arguments = {}};
    std::vector<std::string> files;
    for (int i = first; i < argc; ++i) {
        const std::string_view argument = argv[i] == nullptr ? "" : argv[i];
        if (argument == "--config") {
            if (i + 1 >= argc) { throw std::invalid_argument("--config needs a file"); }
            files.emplace_back(argv[++i]);
            continue;
        }
        command.arguments.emplace_back(argument);
    }
    if (!command.arguments.empty()) {
        const std::string& head = command.arguments.front();
        out.help                = head == "--help" || head == "-h";
        out.version             = head == "--version";
    }
    if (out.help || out.version) {
        out.layers.push_back(std::move(command));
        return out;
    }
    std::optional<std::string> file_artifact;
    for (const std::string& file : files) {
        const ConfigFile config = read_config_file(file);
        ArgumentLayer layer{.source = file, .arguments = {}};
        for (const ConfigFile::Section& section : config.sections) {
            if (section.name != "*") {
                throw std::invalid_argument(file + ": [" + section.name +
                                            "]: a launch configuration has no sections; models of "
                                            "a router belong in --models-preset");
            }
            for (const auto& [key, value] : section.entries) {
                const std::string where = file + ": " + key;
                if (key == "config") {
                    throw std::invalid_argument(where + ": a configuration file cannot name another");
                }
                if (key == "model" || key == "artifact") {
                    if (positional == PositionalArtifact::None) {
                        throw std::invalid_argument(
                            where + ": this program takes no artifact argument; name the model "
                                    "with its own option");
                    }
                    if (value.empty()) { throw std::invalid_argument(where + " is empty"); }
                    file_artifact = value;
                    continue;
                }
                if (key == "cuda-backend") {
                    out.cuda_backend = value;
                    continue;
                }
                if (key == "version") { continue; }
                append_config_option(layer.arguments, where, key, value);
            }
        }
        out.layers.push_back(std::move(layer));
    }
    if (!out.artifact) { out.artifact = std::move(file_artifact); }
    out.layers.push_back(std::move(command));
    return out;
}

std::vector<std::string> redact_arguments(std::vector<std::string> arguments,
                                          std::initializer_list<std::string_view> secret_options) {
    for (std::size_t i = 0; i + 1 < arguments.size(); ++i) {
        if (std::find(secret_options.begin(), secret_options.end(), arguments[i]) !=
            secret_options.end()) {
            arguments[++i] = "<redacted>";
        }
    }
    return arguments;
}

} // namespace ninfer::product
