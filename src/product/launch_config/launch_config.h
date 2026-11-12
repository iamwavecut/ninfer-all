#pragma once

// Launch configuration files: a run's options written once and passed as `--config FILE`, in
// llama.cpp's preset format, which ninfer-serve's router presets (--models-preset) share. Each file
// is a layer of options below the command line; product/option_layers.h says how layers override
// each other.

#include <filesystem>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::product {

// `[section]` headers and `key = value` entries. Entries before the first header belong to the
// section `*`.
struct ConfigFile {
    struct Section {
        std::string name;
        std::vector<std::pair<std::string, std::string>> entries;
    };

    std::vector<Section> sections;
};

// Blank lines and lines starting with ';' or '#' are skipped, keys are lowercased, and a value in
// matching quotes loses them. Throws std::invalid_argument naming the line of a malformed entry or
// a repeated section.
[[nodiscard]] ConfigFile parse_config_text(std::string_view text);
// The same for a file; errors name the file.
[[nodiscard]] ConfigFile read_config_file(const std::filesystem::path& path);

// One entry as the options it stands for: `true` is the switch `--key`, `false` turns it off
// (`--no-key`, or `--key` for a key spelled `no-key`), and any other value is `--key value`. A key
// is written without leading dashes; `where` names the entry in errors.
void append_config_option(std::vector<std::string>& arguments, std::string_view where,
                          std::string_view key, std::string_view value);

// One layer of a run's options: a configuration file, or the command line.
struct ArgumentLayer {
    std::string source;
    std::vector<std::string> arguments;
};

// Whether a program takes its artifact as the first argument (ninfer, ninfer-serve,
// ninfer-perplexity, ninfer-ngram-profile) or names it with an option (ninfer-bench's --weights).
enum class PositionalArtifact { Taken, None };

// A run's arguments: the program, its artifact, and its option layers, lowest precedence first.
struct CommandLine {
    std::string program;
    std::optional<std::string> artifact;
    // The command line's first argument is --help or -h, or --version.
    bool help    = false;
    bool version = false;
    // A file's `cuda-backend`, which only the portable launcher reads.
    std::optional<std::string> cuda_backend;
    std::vector<ArgumentLayer> layers;

    [[nodiscard]] std::vector<std::vector<std::string>> layer_arguments() const;
};

// Reads argv. With PositionalArtifact::Taken, argv[1] is the artifact unless it starts with '-'.
// Every `--config FILE` adds the file's entries as a layer, in the order given, before the command
// line's own layer, which holds the remaining arguments. A file's `model` (or `artifact`) names the
// artifact when the command line does not, a later file's winning; `cuda-backend` goes to the
// launcher and `version` (llama.cpp's preset header) is ignored. A file has no sections other than
// `[*]` and no `config` entry. With --help, -h or --version first, the files are not read.
[[nodiscard]] CommandLine read_command_line(int argc, const char* const* argv,
                                            PositionalArtifact positional);

// `arguments` with the value after each of `secret_options` replaced by "<redacted>".
[[nodiscard]] std::vector<std::string>
redact_arguments(std::vector<std::string> arguments,
                 std::initializer_list<std::string_view> secret_options);

} // namespace ninfer::product
