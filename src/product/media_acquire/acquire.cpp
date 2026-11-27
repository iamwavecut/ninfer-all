#include "product/media_acquire/acquire.h"

#include "product/https_client.h"

#include <httplib.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace ninfer::product::media_acquire {
namespace {

using Clock = std::chrono::steady_clock;

void check_control(const Policy& policy) {
    if (policy.is_cancelled && policy.is_cancelled()) {
        throw Error(ErrorKind::Cancelled, "media acquisition was cancelled");
    }
    if (policy.deadline != Clock::time_point{} && Clock::now() >= policy.deadline) {
        throw Error(ErrorKind::DeadlineExceeded, "media acquisition exceeded request deadline");
    }
}

long bounded_timeout_ms(const Policy& policy, int configured) {
    if (policy.deadline == Clock::time_point{}) { return configured; }
    check_control(policy);
    // Rounded up, with one millisecond more: a limit that the deadline bounds expires after the
    // deadline, so check_control reports the deadline.
    const auto remaining =
        std::chrono::ceil<std::chrono::milliseconds>(policy.deadline - Clock::now()).count() + 1;
    return std::min<long>(configured, std::max<std::int64_t>(1, remaining));
}

std::vector<std::uint8_t> decode_base64(std::string_view text) {
    static constexpr std::array<std::int8_t, 256> table = [] {
        std::array<std::int8_t, 256> out{};
        out.fill(-1);
        for (int i = 0; i < 26; ++i) {
            out[static_cast<std::size_t>('A' + i)] = static_cast<std::int8_t>(i);
            out[static_cast<std::size_t>('a' + i)] = static_cast<std::int8_t>(26 + i);
        }
        for (int i = 0; i < 10; ++i) {
            out[static_cast<std::size_t>('0' + i)] = static_cast<std::int8_t>(52 + i);
        }
        out[static_cast<std::size_t>('+')] = 62;
        out[static_cast<std::size_t>('/')] = 63;
        return out;
    }();

    std::vector<std::uint8_t> out;
    out.reserve(text.size() * 3 / 4);
    std::uint32_t bits = 0;
    int count          = 0;
    bool padded        = false;
    for (const unsigned char c : text) {
        if (c == '=') {
            padded = true;
            continue;
        }
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') { continue; }
        if (padded || table[c] < 0) { throw std::invalid_argument("malformed base64 media data"); }
        bits = (bits << 6U) | static_cast<std::uint32_t>(table[c]);
        count += 6;
        if (count >= 8) {
            count -= 8;
            out.push_back(static_cast<std::uint8_t>((bits >> count) & 0xffU));
        }
    }
    if (count >= 6) { throw std::invalid_argument("malformed base64 media padding"); }
    return out;
}

// Addresses a fetch must not reach unless private networks are allowed: this host, private and
// shared networks, link-local, documentation, benchmarking and reserved blocks, multicast, and the
// IPv6 forms that carry an IPv4 address (mapped, translated, 6to4) when that address is one.
bool disallowed_ipv4(std::uint32_t network_order) {
    const std::uint32_t a = ntohl(network_order);
    return (a >> 24U) == 0 ||                                  // 0.0.0.0/8
           (a >> 24U) == 10 ||                                 // 10.0.0.0/8
           (a >> 22U) == ((100U << 2U) | 1U) ||                // 100.64.0.0/10
           (a >> 24U) == 127 ||                                // 127.0.0.0/8
           (a >> 16U) == 0xa9feU ||                            // 169.254.0.0/16
           (a >> 20U) == 0xac1U ||                             // 172.16.0.0/12
           (a >> 8U) == 0xc00000U ||                           // 192.0.0.0/24
           (a >> 8U) == 0xc00002U ||                           // 192.0.2.0/24
           (a >> 8U) == 0xc05863U ||                           // 192.88.99.0/24
           (a >> 16U) == 0xc0a8U ||                            // 192.168.0.0/16
           (a >> 17U) == (0xc612U >> 1U) ||                    // 198.18.0.0/15
           (a >> 8U) == 0xc63364U ||                           // 198.51.100.0/24
           (a >> 8U) == 0xcb0071U ||                           // 203.0.113.0/24
           (a >> 28U) >= 0xeU;                                 // 224.0.0.0/4, 240.0.0.0/4
}

bool embedded_ipv4_disallowed(const std::uint8_t* bytes) {
    std::uint32_t v4 = 0;
    std::memcpy(&v4, bytes, sizeof(v4));
    return disallowed_ipv4(v4);
}

bool disallowed_ipv6(const in6_addr& address) {
    const std::uint8_t* b = address.s6_addr;
    const auto zero       = [b](int first, int last) {
        for (int i = first; i < last; ++i) {
            if (b[i] != 0) { return false; }
        }
        return true;
    };
    if (zero(0, 12)) { return true; } // ::/96: unspecified, loopback, IPv4-compatible
    if (zero(0, 10) && b[10] == 0xff && b[11] == 0xff) {
        return embedded_ipv4_disallowed(b + 12); // ::ffff:0:0/96, IPv4-mapped
    }
    if (b[0] == 0x00 && b[1] == 0x64 && b[2] == 0xff && b[3] == 0x9b) {
        // 64:ff9b::/96 translates an IPv4 address; 64:ff9b:1::/48 is local-use.
        return !zero(4, 12) || embedded_ipv4_disallowed(b + 12);
    }
    if (b[0] == 0x01 && b[1] == 0x00 && zero(2, 8)) { return true; }      // 100::/64, discard
    if (b[0] == 0x20 && b[1] == 0x01 && b[2] == 0x00 && b[3] == 0x00) {   // 2001::/32, Teredo
        return true;
    }
    if (b[0] == 0x20 && b[1] == 0x01 && b[2] == 0x0d && b[3] == 0xb8) {   // 2001:db8::/32
        return true;
    }
    if (b[0] == 0x20 && b[1] == 0x02) { return embedded_ipv4_disallowed(b + 2); } // 2002::/16
    return (b[0] & 0xfeU) == 0xfcU ||                      // fc00::/7, unique local
           (b[0] == 0xfe && (b[1] & 0xc0U) == 0x80U) ||    // fe80::/10, link-local
           (b[0] == 0xfe && (b[1] & 0xc0U) == 0xc0U) ||    // fec0::/10, site-local
           b[0] == 0xff;                                   // ff00::/8, multicast
}

bool disallowed_address(const sockaddr* address) {
    if (address->sa_family == AF_INET) {
        return disallowed_ipv4(reinterpret_cast<const sockaddr_in*>(address)->sin_addr.s_addr);
    }
    if (address->sa_family == AF_INET6) {
        return disallowed_ipv6(reinterpret_cast<const sockaddr_in6*>(address)->sin6_addr);
    }
    return true;
}

// A media URL as a fetch uses it: the scheme and host in lower case, the host without the brackets
// of an IPv6 literal, the port (the scheme's default when the URL has none) and the request target
// (path and query; "/" when the path is empty).
struct UrlParts {
    std::string scheme;
    std::string host;
    int port = 0;
    std::string target;
    bool ipv6 = false;
};

char lower(char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; }

bool alpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

bool digit(char c) { return c >= '0' && c <= '9'; }

// The length of a reference's scheme with its colon, or 0 when the reference has none.
std::size_t scheme_length(std::string_view value) {
    if (value.empty() || !alpha(value.front())) { return 0; }
    for (std::size_t i = 1; i < value.size(); ++i) {
        const char c = value[i];
        if (c == ':') { return i + 1; }
        if (!alpha(c) && !digit(c) && c != '+' && c != '-' && c != '.') { return 0; }
    }
    return 0;
}

// The request target as sent: bytes outside ASCII percent-encoded, everything else as given.
std::string request_target(std::string_view path_and_query) {
    static constexpr char hex[] = "0123456789ABCDEF";
    std::string out = path_and_query.empty() || path_and_query.front() == '?' ? "/" : "";
    for (const char c : path_and_query) {
        const auto byte = static_cast<unsigned char>(c);
        if (byte >= 0x80) {
            out += '%';
            out += hex[byte >> 4U];
            out += hex[byte & 15U];
        } else {
            out += c;
        }
    }
    return out;
}

UrlParts parse_url(std::string_view value) {
    for (const char c : value) {
        const auto byte = static_cast<unsigned char>(c);
        if (byte <= 0x20 || byte == 0x7f) { throw std::invalid_argument("invalid media URL"); }
    }
    const std::size_t colon = scheme_length(value);
    if (colon == 0) { throw std::invalid_argument("invalid media URL"); }
    UrlParts out;
    for (const char c : value.substr(0, colon - 1)) { out.scheme += lower(c); }
    if (out.scheme != "http" && out.scheme != "https") {
        throw std::invalid_argument("media URL must be credential-free HTTP(S)");
    }
    std::string_view rest = value.substr(colon);
    if (!rest.starts_with("//")) { throw std::invalid_argument("invalid media URL"); }
    rest.remove_prefix(2);
    const std::size_t authority_end = std::min(rest.find_first_of("/?#"), rest.size());
    const std::string_view authority = rest.substr(0, authority_end);
    if (authority.find('@') != std::string_view::npos) {
        throw std::invalid_argument("media URL must be credential-free HTTP(S)");
    }
    std::string_view port;
    if (authority.starts_with('[')) {
        const std::size_t close = authority.find(']');
        if (close == std::string_view::npos) { throw std::invalid_argument("invalid media URL"); }
        out.host = std::string(authority.substr(1, close - 1));
        in6_addr address{};
        if (inet_pton(AF_INET6, out.host.c_str(), &address) != 1) {
            throw std::invalid_argument("invalid media URL host");
        }
        out.ipv6 = true;
        const std::string_view after = authority.substr(close + 1);
        if (!after.empty()) {
            if (after.front() != ':') { throw std::invalid_argument("invalid media URL"); }
            port = after.substr(1);
        }
    } else {
        const std::size_t separator = authority.find(':');
        const std::string_view host = authority.substr(0, separator);
        if (separator != std::string_view::npos) { port = authority.substr(separator + 1); }
        for (const char c : host) {
            if (!alpha(c) && !digit(c) && c != '-' && c != '.' && c != '_') {
                throw std::invalid_argument("invalid media URL host");
            }
            out.host += lower(c);
        }
    }
    if (out.host.empty()) { throw std::invalid_argument("media URL must be credential-free HTTP(S)"); }
    out.port = out.scheme == "https" ? 443 : 80;
    if (!port.empty()) {
        if (port.size() > 5 || !std::all_of(port.begin(), port.end(), digit)) {
            throw std::invalid_argument("invalid media URL port");
        }
        out.port = std::stoi(std::string(port));
        if (out.port < 1 || out.port > 65535) { throw std::invalid_argument("invalid media URL port"); }
    }
    std::string_view target = rest.substr(authority_end);
    target = target.substr(0, target.find('#'));
    out.target = request_target(target);
    return out;
}

// RFC 3986 section 5.2.4.
std::string remove_dot_segments(std::string_view input) {
    std::string output;
    while (!input.empty()) {
        if (input.starts_with("../")) {
            input.remove_prefix(3);
        } else if (input.starts_with("./")) {
            input.remove_prefix(2);
        } else if (input.starts_with("/./")) {
            input.remove_prefix(2);
        } else if (input == "/.") {
            input = "/";
        } else if (input.starts_with("/../") || input == "/..") {
            input = input.size() == 3 ? std::string_view("/") : input.substr(3);
            const std::size_t last = output.rfind('/');
            output.erase(last == std::string::npos ? 0 : last);
        } else if (input == "." || input == "..") {
            input = {};
        } else {
            const std::size_t next = input.find('/', input.front() == '/' ? 1 : 0);
            const std::size_t end  = next == std::string_view::npos ? input.size() : next;
            output.append(input.substr(0, end));
            input.remove_prefix(end);
        }
    }
    return output;
}

// The absolute URL a Location header names, resolved against the URL that answered with it
// (RFC 3986 section 5.2.2).
std::string resolve_location(const UrlParts& base, std::string_view location) {
    if (scheme_length(location) != 0) { return std::string(location); }
    if (location.starts_with("//")) { return base.scheme + ":" + std::string(location); }
    std::string origin = base.scheme + "://" + (base.ipv6 ? "[" + base.host + "]" : base.host) +
                         ":" + std::to_string(base.port);
    location = location.substr(0, location.find('#'));
    const std::size_t query_at = location.find('?');
    const std::string_view path = location.substr(0, query_at);
    const std::string_view query =
        query_at == std::string_view::npos ? std::string_view{} : location.substr(query_at);
    const std::string_view base_target = base.target;
    const std::size_t base_query_at    = base_target.find('?');
    const std::string_view base_path   = base_target.substr(0, base_query_at);
    if (path.empty()) {
        const std::string_view base_query = base_query_at == std::string_view::npos
                                                ? std::string_view{}
                                                : base_target.substr(base_query_at);
        return origin + std::string(base_path) +
               std::string(query_at == std::string_view::npos ? base_query : query);
    }
    if (path.front() == '/') { return origin + remove_dot_segments(path) + std::string(query); }
    const std::size_t slash = base_path.rfind('/');
    const std::string merged =
        std::string(base_path.substr(0, slash == std::string_view::npos ? 0 : slash + 1)) +
        std::string(path);
    return origin + remove_dot_segments(merged) + std::string(query);
}

std::string resolve_public(const UrlParts& url, bool allow_private) {
#ifdef _WIN32
    static std::once_flag winsock_init;
    std::call_once(winsock_init, [] {
        WSADATA data{};
        const int result = ::WSAStartup(MAKEWORD(2, 2), &data);
        if (result != 0) {
            throw Error(ErrorKind::RemoteUnavailable,
                        "failed to initialize Winsock: " + std::to_string(result));
        }
    });
#endif
    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* raw     = nullptr;
    const std::string port = std::to_string(url.port);
    const int rc           = getaddrinfo(url.host.c_str(), port.c_str(), &hints, &raw);
    if (rc != 0) {
        throw Error(ErrorKind::RemoteUnavailable,
#ifdef _WIN32
                    "failed to resolve media URL host: " + std::to_string(rc));
#else
                    "failed to resolve media URL host: " + std::string(gai_strerror(rc)));
#endif
    }
    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> addresses(raw, freeaddrinfo);
    std::string selected;
    for (const addrinfo* it = raw; it != nullptr; it = it->ai_next) {
        if (!allow_private && disallowed_address(it->ai_addr)) { continue; }
        std::array<char, INET6_ADDRSTRLEN> text{};
        const void* bytes =
            it->ai_family == AF_INET
                ? static_cast<const void*>(
                      &reinterpret_cast<const sockaddr_in*>(it->ai_addr)->sin_addr)
                : static_cast<const void*>(
                      &reinterpret_cast<const sockaddr_in6*>(it->ai_addr)->sin6_addr);
        if (inet_ntop(it->ai_family, bytes, text.data(), text.size()) != nullptr) {
            selected = text.data();
            if (it->ai_family == AF_INET) { break; }
        }
    }
    if (selected.empty()) {
        throw std::invalid_argument("media URL resolves only to disallowed network addresses");
    }
    return selected;
}

// Stops a request from its own thread when the request's time runs out, its deadline passes or it
// is cancelled: a connect or read that waits for the server wakes only so. While a request runs
// only this thread calls the policy's cancellation callback; the fetch's thread waits in the
// request meanwhile and calls it again once this thread has ended.
class Watchdog {
public:
    Watchdog(httplib::ClientImpl& client, const Policy& policy, Clock::time_point end)
        : client_(client), policy_(policy), end_(end), thread_([this] { run(); }) {}
    ~Watchdog() {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            done_ = true;
        }
        wake_.notify_all();
        thread_.join();
    }
    Watchdog(const Watchdog&)            = delete;
    Watchdog& operator=(const Watchdog&) = delete;

    [[nodiscard]] bool stopped() const noexcept { return stopped_.load(); }
    [[nodiscard]] bool timed_out() const noexcept { return timed_out_.load(); }

private:
    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!done_) {
            const auto now = Clock::now();
            bool stop      = stopped_.load();
            if (!stop && now >= end_) { timed_out_ = stop = true; }
            if (!stop && policy_.deadline != Clock::time_point{} && now >= policy_.deadline) {
                stop = true;
            }
            if (!stop && policy_.is_cancelled) {
                try {
                    stop = policy_.is_cancelled();
                } catch (...) { stop = true; }
            }
            if (stop) {
                stopped_ = true;
                // A request that starts clears the client's stop, so this repeats until it ends.
                client_.stop();
            }
            wake_.wait_for(lock, std::chrono::milliseconds(20));
        }
    }

    httplib::ClientImpl& client_;
    const Policy& policy_;
    const Clock::time_point end_;
    std::mutex mutex_;
    std::condition_variable wake_;
    bool done_ = false;
    std::atomic<bool> stopped_{false};
    std::atomic<bool> timed_out_{false};
    std::thread thread_;
};

struct Reply {
    int status = 0;
    std::string location;
    std::vector<std::uint8_t> body;
};

// A response's Content-Length, or 0 without one; a length past 64 bits reads as the largest.
std::uint64_t declared_length(const httplib::Response& response) {
    if (!response.has_header("Content-Length")) { return 0; }
    std::uint64_t length = 0;
    for (const char c : response.get_header_value("Content-Length")) {
        if (!digit(c)) { return 0; }
        if (length > (std::numeric_limits<std::uint64_t>::max() - 9) / 10) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        length = length * 10 + static_cast<std::uint64_t>(c - '0');
    }
    return length;
}

Reply request(const UrlParts& url, const std::string& address, const Policy& policy) {
    const auto hop_ms = bounded_timeout_ms(policy, policy.timeout_ms);
    std::unique_ptr<httplib::ClientImpl> client;
    if (url.scheme == "https") {
        client = product::verifying_https_client(url.host, url.port);
    } else {
        client = std::make_unique<httplib::ClientImpl>(url.host, url.port);
    }
    client->set_direct_hostname_addr_map({{url.host, address}});
    client->set_follow_location(false);
    client->set_decompress(false);
    client->set_path_encode(false);
    client->set_keep_alive(false);
    client->set_connection_timeout(
        std::chrono::milliseconds(bounded_timeout_ms(policy, policy.connect_timeout_ms)));
    // The watchdog ends the request at hop_ms; the client's own limits come later, so a request
    // that runs out of time is always reported as a timeout.
    client->set_read_timeout(std::chrono::milliseconds(hop_ms + 1'000));
    client->set_write_timeout(std::chrono::milliseconds(hop_ms + 1'000));

    Reply reply;
    bool over_limit = false;
    httplib::Result result;
    bool timed_out = false;
    {
        const Watchdog watchdog(*client, policy, Clock::now() + std::chrono::milliseconds(hop_ms));
        const httplib::Headers headers = {{"User-Agent", "ninfer/vision"}};
        result = client->Get(
            url.target, headers,
            [&](const httplib::Response& response) {
                reply.status = response.status;
                if (response.status >= 300 && response.status < 400) {
                    reply.location = response.get_header_value("Location");
                }
                if (response.status >= 200 && response.status < 300 &&
                    declared_length(response) > policy.max_bytes) {
                    over_limit = true;
                    return false;
                }
                return !watchdog.stopped();
            },
            [&](const char* data, std::size_t length) {
                if (reply.status < 200 || reply.status >= 300) { return !watchdog.stopped(); }
                if (length > policy.max_bytes - reply.body.size()) {
                    over_limit = true;
                    return false;
                }
                reply.body.insert(reply.body.end(), reinterpret_cast<const std::uint8_t*>(data),
                                  reinterpret_cast<const std::uint8_t*>(data) + length);
                return !watchdog.stopped();
            });
        timed_out = watchdog.timed_out();
    }
    check_control(policy);
    if (over_limit) { throw Error(ErrorKind::BudgetExceeded, "media URL exceeds byte limit"); }
    if (!result) {
        const httplib::Error error = result.error();
        if (timed_out || error == httplib::Error::ConnectionTimeout ||
            error == httplib::Error::Timeout) {
            throw Error(ErrorKind::RemoteTimeout,
                        "media URL fetch timed out: " + httplib::to_string(error));
        }
        throw Error(ErrorKind::RemoteUnavailable,
                    "failed to fetch media URL: " + httplib::to_string(error));
    }
    return reply;
}

std::vector<std::uint8_t> fetch_url(std::string url, const Policy& policy) {
    if (!policy.allow_remote) { throw std::invalid_argument("remote media URLs are disabled"); }
    std::string previous_scheme;
    for (int redirect = 0; redirect <= policy.max_redirects; ++redirect) {
        check_control(policy);
        const UrlParts parts = parse_url(url);
        if (previous_scheme == "https" && parts.scheme != "https") {
            throw std::invalid_argument("media URL redirects from HTTPS to HTTP");
        }
        previous_scheme      = parts.scheme;
        const std::string ip = resolve_public(parts, policy.allow_private_network);
        check_control(policy);
        Reply reply = request(parts, ip, policy);
        if (reply.status >= 200 && reply.status < 300) {
            check_control(policy);
            return std::move(reply.body);
        }
        if (reply.status < 300 || reply.status >= 400 || redirect == policy.max_redirects) {
            throw Error(ErrorKind::RemoteUnavailable,
                        "media URL returned HTTP " + std::to_string(reply.status));
        }
        if (reply.location.empty()) {
            throw Error(ErrorKind::RemoteUnavailable, "media URL redirect has no location");
        }
        url = resolve_location(parts, reply.location);
    }
    throw Error(ErrorKind::RemoteUnavailable, "too many media URL redirects");
}

std::vector<std::uint8_t> read_path(const Source& source, const Policy& policy) {
    check_control(policy);
    std::error_code ec;
    std::filesystem::path path = std::filesystem::weakly_canonical(source.value, ec);
    if (ec || !std::filesystem::is_regular_file(path, ec)) {
        throw std::invalid_argument("media path is not a regular file: " + source.value);
    }
    if (!policy.media_root.empty()) {
        const std::filesystem::path root = std::filesystem::weakly_canonical(policy.media_root, ec);
        const auto relative              = std::filesystem::relative(path, root, ec);
        if (ec || relative.empty() || relative.generic_string().starts_with("..")) {
            throw std::invalid_argument("media path is outside configured media root");
        }
    }
    const std::uintmax_t size = std::filesystem::file_size(path, ec);
    check_control(policy);
    if (ec) { throw std::invalid_argument("failed to inspect media file: " + source.value); }
    if (size > policy.max_bytes) {
        throw Error(ErrorKind::BudgetExceeded, "media file exceeds byte limit");
    }

    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::invalid_argument("failed to open media path: " + path.string()); }
    stream.seekg(0, std::ios::end);
    const std::streamoff end = stream.tellg();
    if (end < 0 || static_cast<std::uintmax_t>(end) > policy.max_bytes) {
        throw Error(ErrorKind::BudgetExceeded, "media file exceeds byte limit");
    }
    stream.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    if (!bytes.empty()) {
        stream.read(reinterpret_cast<char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()));
        if (!stream) { throw std::invalid_argument("failed to read media path: " + path.string()); }
    }
    check_control(policy);
    return bytes;
}

} // namespace

bool detail::disallowed_numeric_address(std::string_view address) {
    const std::string text(address);
    in_addr v4{};
    if (inet_pton(AF_INET, text.c_str(), &v4) == 1) { return disallowed_ipv4(v4.s_addr); }
    in6_addr v6{};
    if (inet_pton(AF_INET6, text.c_str(), &v6) == 1) { return disallowed_ipv6(v6); }
    throw std::invalid_argument("not a numeric IP address: " + text);
}

std::vector<std::uint8_t> acquire_bytes(const Source& source, const Policy& policy) {
    if (policy.max_bytes == 0) { throw std::invalid_argument("media byte limit must be positive"); }
    check_control(policy);
    if (source.kind == SourceKind::Bytes) {
        if (source.bytes.empty()) { throw std::invalid_argument("media source is empty"); }
        if (source.bytes.size() > policy.max_bytes) {
            throw Error(ErrorKind::BudgetExceeded, "media bytes exceed byte limit");
        }
        std::vector<std::uint8_t> bytes = source.bytes;
        check_control(policy);
        return bytes;
    }
    if (source.value.empty()) { throw std::invalid_argument("media source is empty"); }

    if (source.kind == SourceKind::Url) {
        std::vector<std::uint8_t> bytes = fetch_url(source.value, policy);
        if (bytes.empty()) { throw std::invalid_argument("media source contains no data"); }
        return bytes;
    }
    if (source.kind == SourceKind::Data) {
        const std::size_t comma = source.value.find(',');
        if (!source.value.starts_with("data:") || comma == std::string::npos ||
            source.value.substr(0, comma).find(";base64") == std::string::npos) {
            throw std::invalid_argument("media data source must be a base64 data URI");
        }
        const std::size_t encoded = source.value.size() - comma - 1;
        if (encoded > (policy.max_bytes / 3 + 1) * 4) {
            throw Error(ErrorKind::BudgetExceeded, "media data exceeds byte limit");
        }
        std::vector<std::uint8_t> bytes =
            decode_base64(std::string_view(source.value).substr(comma + 1));
        check_control(policy);
        if (bytes.size() > policy.max_bytes) {
            throw Error(ErrorKind::BudgetExceeded, "media data exceeds byte limit");
        }
        if (bytes.empty()) { throw std::invalid_argument("media source contains no data"); }
        return bytes;
    }
    return read_path(source, policy);
}

} // namespace ninfer::product::media_acquire
