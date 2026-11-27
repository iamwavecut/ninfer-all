#include "product/hub/hub.h"

#include "product/https_client.h"

#include <nlohmann/json.hpp>
#include <psa/crypto.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

namespace ninfer::product::hub {

namespace fs = std::filesystem;

namespace {

constexpr std::string_view kHubHost    = "huggingface.co";
constexpr std::string_view kRevision   = "main";
constexpr unsigned kConnections         = 16;
constexpr int kPieceAttempts            = 5;
constexpr int kRedirects                = 5;
constexpr auto kReportInterval          = std::chrono::seconds(5);

bool id_character(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' ||
           c == '_' || c == '.';
}

bool id_segment(std::string_view segment) {
    if (segment.empty() || segment.size() > 96 || segment.find("..") != std::string_view::npos) {
        return false;
    }
    if (segment.front() == '-' || segment.front() == '.' || segment.front() == '_') { return false; }
    return std::all_of(segment.begin(), segment.end(), id_character);
}

std::string environment(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

fs::path home_directory() {
#ifdef _WIN32
    const std::string home = environment("USERPROFILE");
#else
    const std::string home = environment("HOME");
#endif
    if (home.empty()) {
        throw std::runtime_error("cannot locate the Hugging Face cache: no home directory");
    }
    return home;
}

fs::path hf_home() {
    if (const std::string home = environment("HF_HOME"); !home.empty()) { return home; }
    if (const std::string xdg = environment("XDG_CACHE_HOME"); !xdg.empty()) {
        return fs::path(xdg) / "huggingface";
    }
    return home_directory() / ".cache" / "huggingface";
}

std::string trim(std::string text) {
    const auto space = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
    while (!text.empty() && space(static_cast<unsigned char>(text.back()))) { text.pop_back(); }
    std::size_t start = 0;
    while (start < text.size() && space(static_cast<unsigned char>(text[start]))) { ++start; }
    return text.substr(start);
}

// HF_TOKEN, else the token file of `hf auth login` (HF_TOKEN_PATH, else HF_HOME/token).
std::string token() {
    if (std::string value = trim(environment("HF_TOKEN")); !value.empty()) { return value; }
    std::string path = environment("HF_TOKEN_PATH");
    std::ifstream file(path.empty() ? hf_home() / "token" : fs::path(path));
    std::string line;
    if (file && std::getline(file, line)) { return trim(line); }
    return {};
}

struct Url {
    std::string host;
    int port = 443;
    std::string target; // path and query
};

Url parse_https(std::string_view url) {
    constexpr std::string_view scheme = "https://";
    if (!url.starts_with(scheme)) {
        throw std::runtime_error("the Hub redirected a download away from HTTPS");
    }
    url.remove_prefix(scheme.size());
    const std::size_t slash = url.find('/');
    std::string_view authority = url.substr(0, slash);
    Url out;
    out.target = slash == std::string_view::npos ? "/" : std::string(url.substr(slash));
    if (const std::size_t colon = authority.rfind(':'); colon != std::string_view::npos) {
        out.port = std::stoi(std::string(authority.substr(colon + 1)));
        authority = authority.substr(0, colon);
    }
    out.host = std::string(authority);
    if (out.host.empty()) { throw std::runtime_error("the Hub returned a URL without a host"); }
    return out;
}

// A Location header against the URL that answered with it: absolute, or a path on its host.
std::string follow(const Url& from, const std::string& location) {
    if (location.starts_with("https://") || location.starts_with("http://")) { return location; }
    if (!location.starts_with("/")) {
        throw std::runtime_error("the Hub returned an unsupported redirect");
    }
    return "https://" + from.host + (from.port == 443 ? "" : ":" + std::to_string(from.port)) +
           location;
}

std::unique_ptr<httplib::SSLClient> client_for(const Url& url) {
    auto client = verifying_https_client(url.host, url.port);
    client->set_follow_location(false);
    client->set_path_encode(false);
    client->set_decompress(false);
    client->set_connection_timeout(std::chrono::seconds(30));
    client->set_read_timeout(std::chrono::seconds(60));
    return client;
}

// The Hub's own requests carry the token; a redirect to another host does not.
httplib::Headers headers_for(const Url& url, const std::string& bearer) {
    httplib::Headers headers{{"User-Agent", "ninfer"}};
    if (!bearer.empty() && url.host == kHubHost) {
        headers.emplace("Authorization", "Bearer " + bearer);
    }
    return headers;
}

std::string describe(const httplib::Result& result) {
    return result ? "HTTP " + std::to_string(result->status) : httplib::to_string(result.error());
}

[[noreturn]] void refuse(const Reference& reference, int status) {
    const std::string name = reference.repository;
    if (status == 401) {
        throw std::runtime_error("Hugging Face repository " + name +
                                 " does not exist, or it is private or gated: set HF_TOKEN, and "
                                 "accept a gated repository's terms on huggingface.co");
    }
    if (status == 403) {
        throw std::runtime_error("Hugging Face repository " + name +
                                 " is gated: accept its terms on huggingface.co and set HF_TOKEN");
    }
    if (status == 404) {
        throw std::runtime_error("no Hugging Face repository " + name);
    }
    throw std::runtime_error("Hugging Face repository " + name + " answered HTTP " +
                             std::to_string(status));
}

struct Revision {
    std::string commit;
    std::vector<std::string> files;
};

// The repository's main revision: its commit and files. Empty when the Hub cannot be reached; a
// refusal (missing, private or gated) throws.
std::optional<Revision> remote_revision(const Reference& reference, const std::string& bearer) {
    const Url url{std::string(kHubHost), 443,
                  "/api/models/" + reference.repository + "/revision/" + std::string(kRevision)};
    const auto client = client_for(url);
    const httplib::Result result = client->Get(url.target, headers_for(url, bearer));
    if (!result) { return std::nullopt; }
    if (result->status != 200) { refuse(reference, result->status); }
    const nlohmann::json body = nlohmann::json::parse(result->body, nullptr, false);
    if (body.is_discarded() || !body.contains("sha") || !body["sha"].is_string()) {
        throw std::runtime_error("the Hub's answer for " + reference.repository +
                                 " names no revision");
    }
    Revision out;
    out.commit = body["sha"].get<std::string>();
    for (const auto& sibling : body.value("siblings", nlohmann::json::array())) {
        if (sibling.contains("rfilename") && sibling["rfilename"].is_string()) {
            out.files.push_back(sibling["rfilename"].get<std::string>());
        }
    }
    return out;
}

// The commit the cache last recorded for the main revision, from refs/main.
std::string cached_commit(const fs::path& repository) {
    std::ifstream file(repository / "refs" / std::string(kRevision));
    std::string commit;
    if (file) { std::getline(file, commit); }
    commit = trim(std::move(commit));
    return commit.size() == 40 &&
                   std::all_of(commit.begin(), commit.end(),
                               [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); })
               ? commit
               : std::string();
}

std::string choose_file(const Reference& reference, const std::vector<std::string>& files) {
    if (!reference.file.empty()) {
        if (std::find(files.begin(), files.end(), reference.file) == files.end()) {
            throw std::runtime_error("Hugging Face repository " + reference.repository +
                                     " has no file " + reference.file);
        }
        return reference.file;
    }
    std::vector<std::string> artifacts;
    for (const std::string& file : files) {
        if (file.ends_with(".ninfer")) { artifacts.push_back(file); }
    }
    if (artifacts.size() == 1) { return artifacts.front(); }
    if (artifacts.empty()) {
        throw std::runtime_error("Hugging Face repository " + reference.repository +
                                 " holds no .ninfer artifact");
    }
    std::string list;
    for (const std::string& file : artifacts) { list += "\n  " + reference.repository + ":" + file; }
    throw std::runtime_error("Hugging Face repository " + reference.repository +
                             " holds several .ninfer artifacts; name one:" + list);
}

struct Remote {
    std::string url; // the download's address after the Hub's redirects
    std::uint64_t size = 0;
    std::string etag;  // the blob's name: the SHA-256 of a large file, else the Hub's ETag
    bool sha256 = false;
};

std::string unquote(std::string value) {
    if (value.starts_with("W/")) { value.erase(0, 2); }
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
        value = value.substr(1, value.size() - 2);
    }
    return value;
}

std::uint64_t parse_size(const std::string& text) {
    if (text.empty() || !std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; })) {
        return 0;
    }
    return std::stoull(text);
}

// The file's size and SHA-256 from the Hub's answer for it, and where its bytes are served.
Remote locate(const Reference& reference, const std::string& commit, const std::string& file,
              const std::string& bearer) {
    std::string address = "https://" + std::string(kHubHost) + "/" + reference.repository +
                          "/resolve/" + commit + "/" + file;
    Remote out;
    for (int hop = 0; hop <= kRedirects; ++hop) {
        const Url url  = parse_https(address);
        const auto client = client_for(url);
        const httplib::Result result = client->Head(url.target, headers_for(url, bearer));
        if (!result) {
            throw std::runtime_error("cannot reach " + url.host + " for " + file + ": " +
                                     describe(result));
        }
        const httplib::Response& response = *result;
        if (url.host == kHubHost) {
            // The Hub names a large file's size and SHA-256 in X-Linked-*, a small one's in the
            // standard headers of the answer that serves it.
            if (const auto linked = response.get_header_value("X-Linked-Etag"); !linked.empty()) {
                out.etag = unquote(linked);
            }
            if (const auto linked = parse_size(response.get_header_value("X-Linked-Size")); linked) {
                out.size = linked;
            }
        }
        if (response.status >= 300 && response.status < 400) {
            const std::string location = response.get_header_value("Location");
            if (location.empty()) { throw std::runtime_error("the Hub redirected nowhere"); }
            address = follow(url, location);
            continue;
        }
        if (response.status != 200) {
            if (url.host == kHubHost) { refuse(reference, response.status); }
            throw std::runtime_error(url.host + " answered HTTP " +
                                     std::to_string(response.status) + " for " + file);
        }
        if (out.etag.empty()) { out.etag = unquote(response.get_header_value("ETag")); }
        if (out.size == 0) { out.size = parse_size(response.get_header_value("Content-Length")); }
        out.url = address;
        break;
    }
    if (out.url.empty()) { throw std::runtime_error("too many redirects for " + file); }
    if (out.size == 0 || out.etag.empty() || out.etag.find('/') != std::string::npos ||
        out.etag.find("..") != std::string::npos) {
        throw std::runtime_error("the Hub gave no size or checksum for " + file);
    }
    out.sha256 = out.etag.size() == 64 &&
                 std::all_of(out.etag.begin(), out.etag.end(), [](char c) {
                     return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
                 });
    return out;
}

std::string gigabytes(std::uint64_t bytes) {
    char text[32];
    if (bytes < 1'000'000'000ULL) {
        std::snprintf(text, sizeof(text), "%.1f MB", static_cast<double>(bytes) / 1e6);
    } else {
        std::snprintf(text, sizeof(text), "%.1f GB", static_cast<double>(bytes) / 1e9);
    }
    return text;
}

// Downloads `remote` into `partial` in pieces over several connections. `pieces` lists the pieces
// done by an earlier attempt; each finished piece is appended to it. The pieces are hashed in order
// as they arrive. Throws when a piece fails every attempt or the checksum differs.
void download(const Reference& reference, const std::string& commit, const std::string& file,
              const Remote& remote, const fs::path& partial, const fs::path& pieces,
              const std::string& bearer, const Report& report) {
    const std::uint64_t count = piece_count(remote.size);
    std::vector<char> done(count, 0);
    {
        std::ifstream recorded(pieces);
        std::uint64_t index = 0;
        while (recorded >> index) {
            if (index < count) { done[index] = 1; }
        }
    }
    std::error_code error;
    if (!fs::exists(partial, error)) {
        std::ofstream(partial, std::ios::binary).close();
        std::fill(done.begin(), done.end(), 0);
        fs::remove(pieces, error);
    }
    fs::resize_file(partial, remote.size, error);
    if (error) {
        throw std::runtime_error("cannot size " + partial.string() + ": " + error.message());
    }
    std::uint64_t resumed = 0;
    for (std::uint64_t index = 0; index < count; ++index) {
        if (done[index]) { resumed += std::min(kPieceBytes, remote.size - index * kPieceBytes); }
    }

    std::mutex mutex;
    std::condition_variable finished;
    std::ofstream log(pieces, std::ios::app);
    std::atomic<std::uint64_t> next{0};
    std::atomic<std::uint64_t> received{resumed};
    std::atomic<bool> stopped{false};
    std::string failure; // under `mutex`
    std::string address = remote.url;

    const auto fetch_piece = [&](std::uint64_t index, const std::string& from) -> std::string {
        const std::uint64_t first = index * kPieceBytes;
        const std::uint64_t bytes = std::min(kPieceBytes, remote.size - first);
        const Url url             = parse_https(from);
        const auto client         = client_for(url);
        std::fstream out(partial, std::ios::binary | std::ios::in | std::ios::out);
        if (!out) { return "cannot open " + partial.string(); }
        out.seekp(static_cast<std::streamoff>(first));
        httplib::Headers headers = headers_for(url, bearer);
        headers.emplace("Range", "bytes=" + std::to_string(first) + "-" +
                                     std::to_string(first + bytes - 1));
        std::uint64_t written = 0;
        int status            = 0;
        const httplib::Result result = client->Get(
            url.target, headers,
            [&](const httplib::Response& response) {
                status = response.status;
                return status == 206 || (status == 200 && first == 0 && bytes == remote.size);
            },
            [&](const char* data, std::size_t length) {
                if (written + length > bytes) { return false; }
                out.write(data, static_cast<std::streamsize>(length));
                written += length;
                received += length;
                return static_cast<bool>(out) && !stopped.load();
            });
        out.flush();
        if (status != 0 && status != 200 && status != 206) {
            received -= written;
            return "HTTP " + std::to_string(status);
        }
        if (!result || written != bytes || !out) {
            received -= written;
            return result ? std::string("short piece") : describe(result);
        }
        return {};
    };

    const auto worker = [&] {
        for (;;) {
            std::uint64_t index = next++;
            while (index < count && done[index]) { index = next++; }
            if (index >= count) { return; }
            std::string problem;
            for (int attempt = 0; attempt < kPieceAttempts; ++attempt) {
                if (stopped.load()) { return; }
                std::string from;
                {
                    const std::lock_guard lock(mutex);
                    from = address;
                }
                problem = fetch_piece(index, from);
                if (problem.empty()) { break; }
                if (problem == "HTTP 403" || problem == "HTTP 410") {
                    // A signed address expired: ask the Hub for a fresh one.
                    try {
                        const Remote again = locate(reference, commit, file, bearer);
                        const std::lock_guard lock(mutex);
                        address = again.url;
                    } catch (const std::exception&) {}
                }
                std::this_thread::sleep_for(std::chrono::seconds(1 << attempt));
            }
            const std::lock_guard lock(mutex);
            if (!problem.empty()) {
                if (failure.empty()) {
                    failure = "piece " + std::to_string(index) + " of " + file + ": " + problem;
                }
                stopped = true;
                finished.notify_all();
                return;
            }
            done[index] = 1;
            log << index << '\n';
            log.flush();
            finished.notify_all();
        }
    };

    if (psa_crypto_init() != PSA_SUCCESS) { throw std::runtime_error("cannot start PSA Crypto"); }
    psa_hash_operation_t hash = PSA_HASH_OPERATION_INIT;
    if (remote.sha256 && psa_hash_setup(&hash, PSA_ALG_SHA_256) != PSA_SUCCESS) {
        throw std::runtime_error("cannot start a SHA-256 hash");
    }

    report("downloading " + reference.repository + ":" + file + " (" + gigabytes(remote.size) +
           (resumed != 0 ? ", " + gigabytes(resumed) + " from an earlier attempt" : "") + ")");
    std::vector<std::thread> workers;
    const auto connections = static_cast<unsigned>(std::min<std::uint64_t>(kConnections, count));
    for (unsigned c = 0; c < connections; ++c) { workers.emplace_back(worker); }

    // Hash the pieces in order as they land, and report progress meanwhile.
    std::ifstream in(partial, std::ios::binary);
    std::vector<char> buffer(4U << 20);
    const auto start       = std::chrono::steady_clock::now();
    auto last_report       = start;
    std::string hash_error;
    for (std::uint64_t index = 0; index < count; ++index) {
        {
            std::unique_lock lock(mutex);
            while (!done[index] && failure.empty()) {
                finished.wait_for(lock, std::chrono::seconds(1));
                const auto now = std::chrono::steady_clock::now();
                if (now - last_report >= kReportInterval) {
                    last_report       = now;
                    const double secs = std::chrono::duration<double>(now - start).count();
                    const std::uint64_t got = received.load();
                    char rate[48];
                    std::snprintf(rate, sizeof(rate), ", %.0f MB/s",
                                  static_cast<double>(got - resumed) / 1e6 / std::max(secs, 1e-3));
                    report("  " + gigabytes(got) + " of " + gigabytes(remote.size) + rate);
                }
            }
            if (!failure.empty()) { break; }
        }
        if (!remote.sha256) { continue; }
        const std::uint64_t first = index * kPieceBytes;
        std::uint64_t left       = std::min(kPieceBytes, remote.size - first);
        in.clear();
        in.seekg(static_cast<std::streamoff>(first));
        while (left != 0 && hash_error.empty()) {
            const auto take = static_cast<std::size_t>(std::min<std::uint64_t>(left, buffer.size()));
            in.read(buffer.data(), static_cast<std::streamsize>(take));
            if (static_cast<std::size_t>(in.gcount()) != take ||
                psa_hash_update(&hash, reinterpret_cast<const std::uint8_t*>(buffer.data()), take) !=
                    PSA_SUCCESS) {
                hash_error = "cannot read back " + partial.string();
            }
            left -= take;
        }
        if (!hash_error.empty()) {
            const std::lock_guard lock(mutex);
            if (failure.empty()) { failure = hash_error; }
            stopped = true;
            break;
        }
    }
    for (std::thread& thread : workers) { thread.join(); }
    log.close();
    if (!failure.empty()) {
        psa_hash_abort(&hash);
        throw std::runtime_error("download stopped (run again to resume): " + failure);
    }
    if (remote.sha256) {
        std::array<std::uint8_t, PSA_HASH_LENGTH(PSA_ALG_SHA_256)> digest{};
        std::size_t length = 0;
        if (psa_hash_finish(&hash, digest.data(), digest.size(), &length) != PSA_SUCCESS) {
            throw std::runtime_error("cannot finish the SHA-256 hash");
        }
        std::string hex;
        constexpr char kDigits[] = "0123456789abcdef";
        for (std::size_t i = 0; i < length; ++i) {
            hex += kDigits[digest[i] >> 4];
            hex += kDigits[digest[i] & 15];
        }
        if (hex != remote.etag) {
            fs::remove(partial, error);
            fs::remove(pieces, error);
            throw std::runtime_error(file + " arrived with SHA-256 " + hex + ", not the Hub's " +
                                     remote.etag + "; the partial file was removed");
        }
    }
}

// The snapshot entry for `file` at `commit`: a relative link to its blob as the Hub's own client
// writes it, or the blob itself moved there where the file system refuses links (Windows without
// developer mode), as that client does too.
void link_snapshot(const fs::path& repository, const std::string& commit, const std::string& file,
                   const fs::path& blob) {
    const fs::path entry = repository / "snapshots" / commit / fs::path(file);
    fs::create_directories(entry.parent_path());
    std::error_code error;
    fs::remove(entry, error);
    const fs::path target = fs::relative(blob, entry.parent_path(), error);
    if (!error) { fs::create_symlink(target, entry, error); }
    if (error) {
        fs::rename(blob, entry, error);
        if (error) {
            throw std::runtime_error("cannot place " + entry.string() + ": " + error.message());
        }
    }
}

void record_commit(const fs::path& repository, const std::string& commit) {
    fs::create_directories(repository / "refs");
    std::ofstream(repository / "refs" / std::string(kRevision), std::ios::trunc) << commit;
}

} // namespace

std::optional<Reference> parse_reference(std::string_view argument) {
    if (argument.empty() || argument.front() == '-') { return std::nullopt; }
    std::error_code error;
    if (fs::exists(fs::path(std::string(argument)), error)) { return std::nullopt; }
    const std::size_t colon       = argument.find(':');
    const std::string_view name   = argument.substr(0, colon);
    const std::string_view file   = colon == std::string_view::npos ? std::string_view{}
                                                                    : argument.substr(colon + 1);
    const std::size_t slash       = name.find('/');
    if (slash == std::string_view::npos || !id_segment(name.substr(0, slash)) ||
        !id_segment(name.substr(slash + 1))) {
        return std::nullopt;
    }
    if (colon == std::string_view::npos && name.ends_with(".ninfer")) { return std::nullopt; }
    if (colon != std::string_view::npos) {
        if (file.empty() || file.front() == '/' || file.find('\\') != std::string_view::npos ||
            file.find("..") != std::string_view::npos) {
            return std::nullopt;
        }
    }
    return Reference{std::string(name), std::string(file)};
}

fs::path cache_directory() {
    if (const std::string cache = environment("HF_HUB_CACHE"); !cache.empty()) { return cache; }
    return hf_home() / "hub";
}

fs::path repository_directory(const fs::path& cache, std::string_view repository) {
    std::string folder = "models--";
    for (const char c : repository) {
        if (c == '/') {
            folder += "--";
        } else {
            folder += c;
        }
    }
    return cache / folder;
}

fs::path fetch(const Reference& reference, const Report& report) {
    const fs::path cache      = cache_directory();
    const fs::path repository = repository_directory(cache, reference.repository);
    const std::string bearer  = token();
    const bool offline        = environment("HF_HUB_OFFLINE") == "1";

    std::optional<Revision> revision;
    if (!offline) { revision = remote_revision(reference, bearer); }
    if (!revision) {
        // The Hub is out of reach: the revision the cache last recorded serves alone.
        const std::string commit = cached_commit(repository);
        if (!commit.empty()) {
            std::vector<std::string> files;
            const fs::path snapshot = repository / "snapshots" / commit;
            std::error_code error;
            for (const auto& entry : fs::recursive_directory_iterator(snapshot, error)) {
                if (!entry.is_directory(error)) {
                    // Lexically: the entries are links into blobs/, which fs::relative would follow.
                    files.push_back(entry.path().lexically_relative(snapshot).generic_string());
                }
            }
            const std::string file = choose_file(reference, files);
            report("using the cached " + reference.repository + ":" + file + " at revision " +
                   commit.substr(0, 10) + (offline ? " (HF_HUB_OFFLINE)" : " (the Hub is out of reach)"));
            return snapshot / fs::path(file);
        }
        throw std::runtime_error(offline ? "HF_HUB_OFFLINE is set and the cache holds no " +
                                               reference.repository
                                         : "cannot reach huggingface.co for " +
                                               reference.repository + ", and the cache holds none");
    }
    const std::string file  = choose_file(reference, revision->files);
    const fs::path snapshot = repository / "snapshots" / revision->commit / fs::path(file);
    std::error_code error;
    if (fs::exists(snapshot, error)) {
        record_commit(repository, revision->commit);
        return snapshot;
    }

    const Remote remote = locate(reference, revision->commit, file, bearer);
    const fs::path blobs = repository / "blobs";
    fs::create_directories(blobs);
    const fs::path blob = blobs / remote.etag;
    if (!fs::exists(blob, error) || fs::file_size(blob, error) != remote.size) {
        const fs::path partial = blobs / (remote.etag + ".incomplete");
        const fs::path pieces  = blobs / (remote.etag + ".pieces");
        download(reference, revision->commit, file, remote, partial, pieces, bearer, report);
        fs::rename(partial, blob, error);
        if (error) {
            throw std::runtime_error("cannot move the download to " + blob.string() + ": " +
                                     error.message());
        }
        fs::remove(pieces, error);
        report("downloaded " + reference.repository + ":" + file);
    }
    link_snapshot(repository, revision->commit, file, blob);
    record_commit(repository, revision->commit);
    return snapshot;
}

std::string resolve_artifact(const std::string& argument, const Report& report) {
    const std::optional<Reference> reference = parse_reference(argument);
    if (!reference) { return argument; }
    return fetch(*reference, report).string();
}

} // namespace ninfer::product::hub
