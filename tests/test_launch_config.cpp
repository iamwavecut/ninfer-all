// Launch configuration files (product/launch_config): the preset format, the options each entry
// stands for, and how a command line's --config files become option layers.

#include "product/launch_config/launch_config.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer::product;

int failures = 0;

void expect(bool condition, const char* label) {
    if (condition) { return; }
    std::cerr << "expectation failed: " << label << '\n';
    ++failures;
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

CommandLine read(std::vector<std::string> arguments,
                 PositionalArtifact positional = PositionalArtifact::Taken) {
    std::vector<const char*> argv;
    for (const std::string& argument : arguments) { argv.push_back(argument.c_str()); }
    return read_command_line(static_cast<int>(argv.size()), argv.data(), positional);
}

std::vector<std::string> options(std::string_view key, std::string_view value) {
    std::vector<std::string> out;
    append_config_option(out, "test", key, value);
    return out;
}

void format() {
    const ConfigFile file = parse_config_text("version = 1\n"
                                              "  ; comment\n"
                                              "# comment\n"
                                              "\n"
                                              "Max-Context = 8192\n"
                                              "prompt = \"two words\"\n"
                                              "stop = 'a = b'\n"
                                              "[*]\n"
                                              "spec = mtp\n");
    expect(file.sections.size() == 1 && file.sections[0].name == "*",
           "entries before any section join the global section");
    const auto& entries = file.sections[0].entries;
    expect(entries.size() == 5 && entries[0].first == "version" &&
               entries[1] == std::pair<std::string, std::string>("max-context", "8192") &&
               entries[2].second == "two words" && entries[3].second == "a = b" &&
               entries[4].first == "spec",
           "keys are lowercased, quotes removed, the first '=' splits, comments skipped");
    expect(rejects([] { (void)parse_config_text("max-context 8192\n"); }),
           "a line without '=' was accepted");
    expect(rejects([] { (void)parse_config_text("[a]\nx = 1\n[a]\n"); }),
           "a repeated section was accepted");
    expect(rejects([] { (void)parse_config_text("[a\n"); }), "a malformed header was accepted");
    try {
        (void)parse_config_text("x = 1\nbroken\n");
    } catch (const std::invalid_argument& error) {
        expect(std::string(error.what()).find("line 2") != std::string::npos,
               "an error names its line");
    }
}

void translation() {
    expect(options("vision", "true") == std::vector<std::string>({"--vision"}),
           "true is the switch");
    expect(options("vision", "FALSE") == std::vector<std::string>({"--no-vision"}),
           "false is the switch's --no- form");
    expect(options("no-prefix-reuse", "false") == std::vector<std::string>({"--prefix-reuse"}),
           "false of a no- key is the positive switch");
    expect(options("device-profile", "off") ==
               std::vector<std::string>({"--device-profile", "off"}),
           "off is a value, not a switch");
    expect(options("max-context", "8192") == std::vector<std::string>({"--max-context", "8192"}),
           "any other value follows its option");
    expect(rejects([] { (void)options("--max-context", "1"); }), "a dashed key was accepted");
}

void command_line() {
    const auto directory = std::filesystem::temp_directory_path() / "ninfer_launch_config_test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto write = [&](const char* name, const char* text) {
        const auto path = directory / name;
        std::ofstream(path) << text;
        return path.string();
    };
    const std::string first  = write("first.ini", "model = a.ninfer\nmax-context = 1\n"
                                                  "cuda-backend = 12\nversion = 1\n");
    const std::string second = write("second.ini", "artifact = b.ninfer\nthinking = false\n");

    const CommandLine both = read({"ninfer", "--config", first, "--prompt", "x", "--config", second});
    expect(both.program == "ninfer" && both.artifact == "b.ninfer",
           "a later file's model wins over an earlier file's");
    expect(both.cuda_backend == "12", "cuda-backend goes to the launcher");
    expect(both.layers.size() == 3 && both.layers[0].source == first &&
               both.layers[1].source == second && both.layers[2].source == "command line",
           "the files in order, then the command line");
    expect(both.layers[0].arguments == std::vector<std::string>({"--max-context", "1"}) &&
               both.layers[1].arguments == std::vector<std::string>({"--no-thinking"}) &&
               both.layers[2].arguments == std::vector<std::string>({"--prompt", "x"}),
           "each layer holds its own options only");

    const CommandLine positional = read({"ninfer", "c.ninfer", "--config", first});
    expect(positional.artifact == "c.ninfer", "the command line's artifact wins over a file's");
    expect(rejects([&] {
               (void)read({"ninfer-bench", "--config", first}, PositionalArtifact::None);
           }),
           "a model key was accepted by a program that takes no artifact argument");

    const std::string sections = write("sections.ini", "[small]\nmax-context = 1\n");
    const std::string nested   = write("nested.ini", "config = first.ini\n");
    const std::string empty    = write("empty.ini", "model =\n");
    for (const std::string& file :
         {sections, nested, empty, (directory / "missing.ini").string()}) {
        expect(rejects([&] { (void)read({"ninfer", "--config", file}); }),
               "a sectioned, nesting, empty-model or missing file was accepted");
    }
    expect(rejects([] { (void)read({"ninfer", "m.ninfer", "--config"}); }),
           "--config without a file was accepted");

    const CommandLine help = read({"ninfer", "--config", (directory / "missing.ini").string(), "-h"});
    expect(help.help && help.layers.size() == 1, "help reads no files");
    expect(read({"ninfer", "--version"}).version, "--version first is the version request");
    std::filesystem::remove_all(directory);
}

void redaction() {
    const auto redacted = redact_arguments({"--api-key", "secret", "--port", "1"}, {"--api-key"});
    expect(redacted == std::vector<std::string>({"--api-key", "<redacted>", "--port", "1"}),
           "the value after a secret option is redacted");
}

} // namespace

int main() {
    try {
        format();
        translation();
        command_line();
        redaction();
    } catch (const std::exception& error) {
        std::cerr << "unexpected exception: " << error.what() << '\n';
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
