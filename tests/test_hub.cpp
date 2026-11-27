// Artifacts named by their Hugging Face repository (product/hub): which arguments are references,
// where the Hub's cache lives, and the cache-only resolution that serves offline.

#include "product/hub/hub.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

namespace fs = std::filesystem;
using namespace ninfer::product::hub;

int failures = 0;

void expect(bool condition, const char* label) {
    if (condition) { return; }
    std::cerr << "expectation failed: " << label << '\n';
    ++failures;
}

void set_variable(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    if (value.empty()) {
        unsetenv(name);
    } else {
        setenv(name, value.c_str(), 1);
    }
#endif
}

bool refuses(const std::function<void()>& operation, const std::string& fragment) {
    try {
        operation();
    } catch (const std::runtime_error& error) {
        return std::string(error.what()).find(fragment) != std::string::npos;
    }
    return false;
}

void references() {
    const auto plain = parse_reference("WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3");
    expect(plain && plain->repository == "WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3" &&
               plain->file.empty(),
           "OWNER/REPO is a reference to the repository's artifact");
    const auto named = parse_reference("neroued/Qwen3.8-27B-nvfp4-NInfer:sub/model.ninfer");
    expect(named && named->repository == "neroued/Qwen3.8-27B-nvfp4-NInfer" &&
               named->file == "sub/model.ninfer",
           "OWNER/REPO:FILE names a file in the repository");
    expect(!parse_reference("models/qwen3_8_27b.ninfer"),
           "a missing local .ninfer path is not a reference");
    expect(parse_reference("owner/repo.ninfer:repo.ninfer").has_value(),
           "a .ninfer repository name with :FILE is a reference");
    expect(!parse_reference("qwen.ninfer"), "a bare file name is not a reference");
    expect(!parse_reference("a/b/c"), "three segments are a path");
    expect(!parse_reference("/abs/path"), "an absolute path is not a reference");
    expect(!parse_reference("./owner/repo"), "a relative path from . is not a reference");
    expect(!parse_reference("-owner/repo"), "an option is not a reference");
    expect(!parse_reference("owner/..repo"), "'..' in a name is refused");
    expect(!parse_reference("owner/re po"), "a space in a name is refused");
    expect(!parse_reference("owner/repo:"), "an empty file is refused");
    expect(!parse_reference("owner/repo:../x.ninfer"), "a file outside the repository is refused");
    expect(!parse_reference("owner/repo:/x.ninfer"), "an absolute file is refused");
    expect(!parse_reference("C:/models/x.ninfer"), "a Windows drive path is not a reference");

    // Whatever exists on disk is a path, whatever its shape.
    const fs::path existing = fs::path("hub-test-owner") / "repo";
    fs::create_directories(existing.parent_path());
    std::ofstream(existing).put('x');
    expect(!parse_reference("hub-test-owner/repo"), "an existing path is not a reference");
    fs::remove_all(existing.parent_path());
}

void cache_location() {
    set_variable("HF_HUB_CACHE", "/cache/hub-root");
    set_variable("HF_HOME", "/home/hf");
    expect(cache_directory() == fs::path("/cache/hub-root"), "HF_HUB_CACHE names the cache");
    set_variable("HF_HUB_CACHE", "");
    expect(cache_directory() == fs::path("/home/hf") / "hub", "HF_HOME/hub otherwise");
    set_variable("HF_HOME", "");
    set_variable("XDG_CACHE_HOME", "/xdg");
    expect(cache_directory() == fs::path("/xdg") / "huggingface" / "hub",
           "XDG_CACHE_HOME/huggingface/hub otherwise");
    set_variable("XDG_CACHE_HOME", "");
    expect(repository_directory("/c", "WaveCut/Qwen3.8-27B") ==
               fs::path("/c") / "models--WaveCut--Qwen3.8-27B",
           "a repository's folder is models--OWNER--REPO");
}

void pieces() {
    expect(piece_count(0) == 0, "an empty file has no pieces");
    expect(piece_count(1) == 1, "one byte is one piece");
    expect(piece_count(kPieceBytes) == 1, "a whole piece is one");
    expect(piece_count(kPieceBytes + 1) == 2, "one byte more starts a second piece");
}

// The cache-only path: HF_HUB_OFFLINE=1 resolves from refs/main and the snapshot it names.
void offline() {
    const fs::path root = fs::temp_directory_path() / "ninfer-hub-test";
    fs::remove_all(root);
    set_variable("HF_HUB_CACHE", root.string());
    set_variable("HF_HUB_OFFLINE", "1");
    const Report quiet = [](const std::string&) {};
    const std::string commit(40, 'a');
    const fs::path repository = repository_directory(root, "owner/repo");
    expect(refuses([&] { (void)fetch({"owner/repo", ""}, quiet); }, "holds no owner/repo"),
           "offline without a cached revision fails");

    fs::create_directories(repository / "refs");
    std::ofstream(repository / "refs" / "main") << commit << '\n';
    const fs::path snapshot = repository / "snapshots" / commit;
    fs::create_directories(snapshot / "extra");
    // As the Hub's client lays it out: the snapshot's entries link to blobs named by checksum.
    fs::create_directories(repository / "blobs");
    std::ofstream(repository / "blobs" / std::string(64, 'b')).put('x');
    std::error_code link_error;
    fs::create_symlink(fs::path("..") / ".." / "blobs" / std::string(64, 'b'),
                       snapshot / "model.ninfer", link_error);
    if (link_error) { std::ofstream(snapshot / "model.ninfer").put('x'); }
    std::ofstream(snapshot / "README.md").put('x');
    expect(fetch({"owner/repo", ""}, quiet) == snapshot / "model.ninfer",
           "offline resolves the cached revision's one artifact");
    expect(fetch({"owner/repo", "README.md"}, quiet) == snapshot / "README.md",
           "offline resolves a named file");
    expect(refuses([&] { (void)fetch({"owner/repo", "other.ninfer"}, quiet); }, "has no file"),
           "a file the snapshot lacks fails");
    std::ofstream(snapshot / "extra" / "second.ninfer").put('x');
    expect(refuses([&] { (void)fetch({"owner/repo", ""}, quiet); }, "owner/repo:extra/second.ninfer"),
           "several artifacts are listed for the person to choose");
    expect(resolve_artifact("owner/repo:extra/second.ninfer", quiet) ==
               (snapshot / "extra" / "second.ninfer").string(),
           "resolve_artifact returns the cached path of a reference");
    expect(resolve_artifact("local.ninfer", quiet) == "local.ninfer",
           "resolve_artifact leaves a path alone");

    set_variable("HF_HUB_OFFLINE", "");
    set_variable("HF_HUB_CACHE", "");
    fs::remove_all(root);
}

} // namespace

int main() {
    references();
    cache_location();
    pieces();
    offline();
    if (failures != 0) {
        std::cerr << failures << " hub expectation(s) failed\n";
        return 1;
    }
    std::cout << "hub tests passed\n";
    return 0;
}
