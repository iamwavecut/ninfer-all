#include "product/media_acquire/acquire.h"

#include <curl/curl.h>

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
    // Rounded up, with one millisecond for curl's whole-millisecond clock: a curl timer that the
    // deadline bounds expires after the deadline, so check_control reports the deadline.
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

struct UrlParts {
    std::string scheme;
    std::string host;
    std::string port;
};

std::string curlu_part(CURLU* url, CURLUPart part, unsigned flags = 0) {
    char* raw            = nullptr;
    const CURLUcode code = curl_url_get(url, part, &raw, flags);
    if (code != CURLUE_OK || raw == nullptr) { return {}; }
    std::string out(raw);
    curl_free(raw);
    return out;
}

UrlParts parse_url(std::string_view value) {
    std::unique_ptr<CURLU, decltype(&curl_url_cleanup)> url(curl_url(), curl_url_cleanup);
    if (!url ||
        curl_url_set(url.get(), CURLUPART_URL, std::string(value).c_str(), 0) != CURLUE_OK) {
        throw std::invalid_argument("invalid media URL");
    }
    UrlParts out;
    out.scheme             = curlu_part(url.get(), CURLUPART_SCHEME);
    out.host               = curlu_part(url.get(), CURLUPART_HOST);
    out.port               = curlu_part(url.get(), CURLUPART_PORT, CURLU_DEFAULT_PORT);
    const std::string user = curlu_part(url.get(), CURLUPART_USER);
    if ((out.scheme != "http" && out.scheme != "https") || out.host.empty() || out.port.empty() ||
        !user.empty()) {
        throw std::invalid_argument("media URL must be credential-free HTTP(S)");
    }
    return out;
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
    // An IPv6 literal comes bracketed from the URL; the resolver takes it bare.
    const std::string host = url.host.size() > 2 && url.host.front() == '[' && url.host.back() == ']'
                                 ? url.host.substr(1, url.host.size() - 2)
                                 : url.host;
    const int rc      = getaddrinfo(host.c_str(), url.port.c_str(), &hints, &raw);
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

struct CurlBuffer {
    std::vector<std::uint8_t> bytes;
    std::size_t limit = 0;
};

int curl_progress(void* opaque, curl_off_t, curl_off_t, curl_off_t, curl_off_t) noexcept {
    const auto& policy = *static_cast<const Policy*>(opaque);
    try {
        return (policy.is_cancelled && policy.is_cancelled()) ||
                       (policy.deadline != Clock::time_point{} && Clock::now() >= policy.deadline)
                   ? 1
                   : 0;
    } catch (...) { return 1; }
}

std::size_t curl_write(char* data, std::size_t size, std::size_t count, void* opaque) {
    auto& out = *static_cast<CurlBuffer*>(opaque);
    if (size != 0 && count > std::numeric_limits<std::size_t>::max() / size) { return 0; }
    const std::size_t amount = size * count;
    if (amount > out.limit - std::min(out.limit, out.bytes.size())) { return 0; }
    out.bytes.insert(out.bytes.end(), reinterpret_cast<std::uint8_t*>(data),
                     reinterpret_cast<std::uint8_t*>(data) + amount);
    return amount;
}

std::vector<std::uint8_t> fetch_url(std::string url, const Policy& policy) {
    if (!policy.allow_remote) { throw std::invalid_argument("remote media URLs are disabled"); }
    static std::once_flag init;
    std::call_once(init, [] {
        if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
            throw std::runtime_error("failed to initialize libcurl");
        }
    });

    std::string previous_scheme;
    for (int redirect = 0; redirect <= policy.max_redirects; ++redirect) {
        check_control(policy);
        const UrlParts parts = parse_url(url);
        if (previous_scheme == "https" && parts.scheme != "https") {
            throw std::invalid_argument("media URL redirects from HTTPS to HTTP");
        }
        previous_scheme = parts.scheme;
        const std::string ip = resolve_public(parts, policy.allow_private_network);
        check_control(policy);
        std::string resolve = parts.host + ":" + parts.port + ":";
        resolve += ip.find(':') == std::string::npos ? ip : "[" + ip + "]";
        curl_slist* resolve_list = curl_slist_append(nullptr, resolve.c_str());
        if (resolve_list == nullptr) { throw std::bad_alloc(); }
        std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> resolve_guard(
            resolve_list, curl_slist_free_all);
        std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(),
                                                                 curl_easy_cleanup);
        if (!curl) { throw std::runtime_error("failed to create libcurl handle"); }
        CurlBuffer buffer{.bytes = {}, .limit = policy.max_bytes};
        curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl.get(), CURLOPT_PROTOCOLS_STR, "http,https");
        curl_easy_setopt(curl.get(), CURLOPT_REDIR_PROTOCOLS_STR, "http,https");
        curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 0L);
        curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT_MS,
                         bounded_timeout_ms(policy, policy.connect_timeout_ms));
        curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT_MS,
                         bounded_timeout_ms(policy, policy.timeout_ms));
        curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl.get(), CURLOPT_SSL_VERIFYPEER, 1L);
        curl_easy_setopt(curl.get(), CURLOPT_SSL_VERIFYHOST, 2L);
        curl_easy_setopt(curl.get(), CURLOPT_PROXY, "");
        curl_easy_setopt(curl.get(), CURLOPT_RESOLVE, resolve_list);
        curl_easy_setopt(curl.get(), CURLOPT_MAXFILESIZE_LARGE,
                         static_cast<curl_off_t>(policy.max_bytes));
        curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, curl_write);
        curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &buffer);
        curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl.get(), CURLOPT_XFERINFOFUNCTION, curl_progress);
        curl_easy_setopt(curl.get(), CURLOPT_XFERINFODATA, &policy);
        curl_easy_setopt(curl.get(), CURLOPT_USERAGENT, "ninfer/vision");
        const CURLcode code = curl_easy_perform(curl.get());
        if (code != CURLE_OK) {
            check_control(policy);
            if (code == CURLE_OPERATION_TIMEDOUT) {
                throw Error(ErrorKind::RemoteTimeout,
                            "media URL fetch timed out: " + std::string(curl_easy_strerror(code)));
            }
            if (code == CURLE_FILESIZE_EXCEEDED || code == CURLE_WRITE_ERROR) {
                throw Error(ErrorKind::BudgetExceeded, "media URL exceeds byte limit");
            }
            throw Error(ErrorKind::RemoteUnavailable,
                        "failed to fetch media URL: " + std::string(curl_easy_strerror(code)));
        }
        long status = 0;
        curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &status);
        if (status >= 200 && status < 300) {
            check_control(policy);
            return buffer.bytes;
        }
        if (status < 300 || status >= 400 || redirect == policy.max_redirects) {
            throw Error(ErrorKind::RemoteUnavailable,
                        "media URL returned HTTP " + std::to_string(status));
        }
        char* next = nullptr;
        curl_easy_getinfo(curl.get(), CURLINFO_REDIRECT_URL, &next);
        if (next == nullptr || *next == '\0') {
            throw Error(ErrorKind::RemoteUnavailable, "media URL redirect has no location");
        }
        url = next;
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
