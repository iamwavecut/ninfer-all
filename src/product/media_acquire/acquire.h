#pragma once

#include "product/media_acquire/source.h"

#include <cstddef>
#include <chrono>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::product::media_acquire {

enum class ErrorKind {
    BudgetExceeded,
    RemoteUnavailable,
    RemoteTimeout,
    DeadlineExceeded,
    Cancelled,
};

class Error final : public std::runtime_error {
public:
    Error(ErrorKind kind, std::string message)
        : std::runtime_error(std::move(message)), kind_(kind) {}

    [[nodiscard]] ErrorKind kind() const noexcept { return kind_; }

private:
    ErrorKind kind_;
};

struct Policy {
    std::size_t max_bytes      = 256ULL << 20;
    int connect_timeout_ms     = 5'000;
    int timeout_ms             = 60'000;
    int max_redirects          = 3;
    bool allow_remote          = true;
    bool allow_private_network = false;
    std::filesystem::path media_root;
    std::chrono::steady_clock::time_point deadline;
    std::function<bool()> is_cancelled;
};

// A URL is fetched over HTTP(S) only, without credentials, a proxy or content decoding. HTTPS
// verifies the server's certificate and host name (product/https_client.h). Its host must resolve
// to an address outside private, loopback, link-local, reserved and multicast networks (unless
// allow_private_network), and the fetch connects to that address alone. Redirects are followed by hand, at most max_redirects, each
// checked the same way and none from HTTPS to HTTP. The body stops at max_bytes, each request at
// timeout_ms and the deadline. While a request waits for its server, another thread of the fetch
// calls is_cancelled; the caller's thread is inside acquire_bytes meanwhile, so no two calls
// overlap.
std::vector<std::uint8_t> acquire_bytes(const Source& source, const Policy& policy = {});

namespace detail {
// Whether a numeric IPv4 or IPv6 address is one a fetch may not reach without
// allow_private_network. Throws std::invalid_argument for text that is not an address.
[[nodiscard]] bool disallowed_numeric_address(std::string_view address);
} // namespace detail

} // namespace ninfer::product::media_acquire
