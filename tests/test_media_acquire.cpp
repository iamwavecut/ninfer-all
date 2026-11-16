// The media fetcher (product/media_acquire) against a local HTTP server: what it fetches, the
// limits it keeps and the addresses it refuses; and the serving policy that decides whether a
// request's URL is fetched at all (--media-urls).

#include "product/media_acquire/acquire.h"
#include "serve/generation_service.h"
#include "serve/request.h"

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace acquire = ninfer::product::media_acquire;
using Clock       = std::chrono::steady_clock;

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

acquire::Source url(const std::string& value) {
    acquire::Source source;
    source.kind  = acquire::SourceKind::Url;
    source.value = value;
    return source;
}

// The local server is loopback, so these fetches allow private networks; the refusal of loopback
// and private addresses is checked separately below.
acquire::Policy local_policy() {
    acquire::Policy policy;
    policy.allow_private_network = true;
    policy.max_bytes             = 1 << 20;
    policy.connect_timeout_ms    = 2'000;
    policy.timeout_ms            = 5'000;
    return policy;
}

enum class Outcome { Bytes, Invalid, Budget, Unavailable, Timeout, Deadline, Cancelled };

const char* name(Outcome outcome) {
    switch (outcome) {
    case Outcome::Bytes: return "bytes";
    case Outcome::Invalid: return "invalid_argument";
    case Outcome::Budget: return "BudgetExceeded";
    case Outcome::Unavailable: return "RemoteUnavailable";
    case Outcome::Timeout: return "RemoteTimeout";
    case Outcome::Deadline: return "DeadlineExceeded";
    case Outcome::Cancelled: return "Cancelled";
    }
    return "?";
}

Outcome fetch(const acquire::Source& source, const acquire::Policy& policy,
              std::vector<std::uint8_t>* bytes = nullptr) {
    try {
        std::vector<std::uint8_t> out = acquire::acquire_bytes(source, policy);
        if (bytes != nullptr) { *bytes = std::move(out); }
        return Outcome::Bytes;
    } catch (const acquire::Error& error) {
        switch (error.kind()) {
        case acquire::ErrorKind::BudgetExceeded: return Outcome::Budget;
        case acquire::ErrorKind::RemoteUnavailable: return Outcome::Unavailable;
        case acquire::ErrorKind::RemoteTimeout: return Outcome::Timeout;
        case acquire::ErrorKind::DeadlineExceeded: return Outcome::Deadline;
        case acquire::ErrorKind::Cancelled: return Outcome::Cancelled;
        }
    } catch (const std::invalid_argument&) { return Outcome::Invalid; }
    return Outcome::Invalid;
}

void expect_outcome(const std::string& what, Outcome actual, Outcome wanted) {
    expect(actual == wanted, what + ": got " + name(actual) + ", expected " + name(wanted));
}

std::string body_of(std::size_t size) {
    std::string body(size, '\0');
    for (std::size_t i = 0; i < size; ++i) { body[i] = static_cast<char>(i * 131U + 7U); }
    return body;
}

class LocalServer {
public:
    LocalServer() {
        server_.Get("/image", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(body_of(4096), "image/png");
        });
        server_.Get("/large", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(body_of(64 * 1024), "image/png");
        });
        // No Content-Length: the limit must hold while the body streams.
        server_.Get("/chunked", [](const httplib::Request&, httplib::Response& res) {
            res.set_chunked_content_provider("image/png",
                                             [](std::size_t offset, httplib::DataSink& sink) {
                                                 if (offset >= 64 * 1024) {
                                                     sink.done();
                                                     return true;
                                                 }
                                                 const std::string piece = body_of(4096);
                                                 return sink.write(piece.data(), piece.size());
                                             });
        });
        server_.Get("/empty", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("", "image/png");
        });
        server_.Get("/missing", [](const httplib::Request&, httplib::Response& res) {
            res.status = 404;
            res.set_content("not found", "text/plain");
        });
        server_.Get("/error", [](const httplib::Request&, httplib::Response& res) {
            res.status = 500;
            res.set_content("broken", "text/plain");
        });
        // The bytes as sent: a fetch does not decode a Content-Encoding.
        server_.Get("/encoded", [](const httplib::Request&, httplib::Response& res) {
            res.set_header("Content-Encoding", "gzip");
            res.set_content(body_of(128), "image/png");
        });
        server_.Get("/slow", [this](const httplib::Request&, httplib::Response& res) {
            for (int i = 0; i < 40 && !stopping_; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            res.set_content(body_of(16), "image/png");
        });
        server_.Get("/relative", [](const httplib::Request&, httplib::Response& res) {
            res.set_redirect("/image");
        });
        server_.Get("/chain1", [](const httplib::Request&, httplib::Response& res) {
            res.set_redirect("/chain2");
        });
        server_.Get("/chain2", [](const httplib::Request&, httplib::Response& res) {
            res.set_redirect("/chain3");
        });
        server_.Get("/chain3", [](const httplib::Request&, httplib::Response& res) {
            res.set_redirect("/image");
        });
        server_.Get("/loop", [](const httplib::Request&, httplib::Response& res) {
            res.set_redirect("/loop");
        });
        server_.Get("/to-ftp", [](const httplib::Request&, httplib::Response& res) {
            res.set_redirect("ftp://127.0.0.1/image");
        });
        server_.Get("/to-file", [](const httplib::Request&, httplib::Response& res) {
            res.set_redirect("file:///etc/hosts");
        });
        server_.Get("/no-location", [](const httplib::Request&, httplib::Response& res) {
            res.status = 302;
        });
        server_.Get("/headers", [this](const httplib::Request& req, httplib::Response& res) {
            seen_authorization_ = req.has_header("Authorization");
            seen_cookie_        = req.has_header("Cookie");
            seen_encoding_      = req.has_header("Accept-Encoding");
            res.set_content(body_of(8), "image/png");
        });
        port_ = server_.bind_to_any_port("127.0.0.1");
        if (port_ <= 0) { throw std::runtime_error("cannot bind the local test server"); }
        thread_ = std::thread([this] { server_.listen_after_bind(); });
        server_.wait_until_ready();
    }

    ~LocalServer() {
        stopping_ = true;
        server_.stop();
        if (thread_.joinable()) { thread_.join(); }
    }

    [[nodiscard]] std::string url(const std::string& path) const {
        return "http://127.0.0.1:" + std::to_string(port_) + path;
    }
    [[nodiscard]] int port() const { return port_; }

    std::atomic<bool> seen_authorization_{false};
    std::atomic<bool> seen_cookie_{false};
    std::atomic<bool> seen_encoding_{false};

private:
    httplib::Server server_;
    std::thread thread_;
    std::atomic<bool> stopping_{false};
    int port_ = 0;
};

void fetches(const LocalServer& server) {
    std::vector<std::uint8_t> bytes;
    expect_outcome("a plain image", fetch(url(server.url("/image")), local_policy(), &bytes),
                   Outcome::Bytes);
    const std::string expected = body_of(4096);
    expect(bytes == std::vector<std::uint8_t>(expected.begin(), expected.end()),
           "the fetched bytes differ from the served ones");

    expect_outcome("a relative redirect", fetch(url(server.url("/relative")), local_policy()),
                   Outcome::Bytes);
    expect_outcome("three redirects", fetch(url(server.url("/chain1")), local_policy()),
                   Outcome::Bytes);
    auto two = local_policy();
    two.max_redirects = 2;
    expect_outcome("a fourth hop over a limit of two redirects",
                   fetch(url(server.url("/chain1")), two), Outcome::Unavailable);
    expect_outcome("a redirect loop", fetch(url(server.url("/loop")), local_policy()),
                   Outcome::Unavailable);
    expect_outcome("a redirect to FTP", fetch(url(server.url("/to-ftp")), local_policy()),
                   Outcome::Invalid);
    expect_outcome("a redirect to a file", fetch(url(server.url("/to-file")), local_policy()),
                   Outcome::Invalid);
    expect_outcome("a redirect without a location",
                   fetch(url(server.url("/no-location")), local_policy()), Outcome::Unavailable);

    expect_outcome("HTTP 404", fetch(url(server.url("/missing")), local_policy()),
                   Outcome::Unavailable);
    expect_outcome("HTTP 500", fetch(url(server.url("/error")), local_policy()),
                   Outcome::Unavailable);
    expect_outcome("an empty body", fetch(url(server.url("/empty")), local_policy()),
                   Outcome::Invalid);

    bytes.clear();
    expect_outcome("an encoded body", fetch(url(server.url("/encoded")), local_policy(), &bytes),
                   Outcome::Bytes);
    const std::string encoded = body_of(128);
    expect(bytes == std::vector<std::uint8_t>(encoded.begin(), encoded.end()),
           "a fetch decoded a Content-Encoding");

    expect_outcome("credentials in the URL",
                   fetch(url("http://user:secret@127.0.0.1:" + std::to_string(server.port()) +
                             "/image"),
                         local_policy()),
                   Outcome::Invalid);
    expect(!server.seen_authorization_, "credentials reached the server");
    expect_outcome("a plain request", fetch(url(server.url("/headers")), local_policy()),
                   Outcome::Bytes);
    expect(!server.seen_authorization_ && !server.seen_cookie_ && !server.seen_encoding_,
           "a fetch sent credentials, cookies or an Accept-Encoding");
}

void limits(const LocalServer& server) {
    auto small      = local_policy();
    small.max_bytes = 16 * 1024;
    expect_outcome("a declared length over the limit", fetch(url(server.url("/large")), small),
                   Outcome::Budget);
    expect_outcome("a streamed body over the limit", fetch(url(server.url("/chunked")), small),
                   Outcome::Budget);
    auto exact      = local_policy();
    exact.max_bytes = 4096;
    expect_outcome("a body of exactly the limit", fetch(url(server.url("/image")), exact),
                   Outcome::Bytes);
    exact.max_bytes = 4095;
    expect_outcome("a body one byte over the limit", fetch(url(server.url("/image")), exact),
                   Outcome::Budget);

    auto quick       = local_policy();
    quick.timeout_ms = 300;
    auto started     = Clock::now();
    expect_outcome("a server slower than the timeout", fetch(url(server.url("/slow")), quick),
                   Outcome::Timeout);
    expect(Clock::now() - started < std::chrono::seconds(2), "the timeout did not stop the fetch");

    auto deadline     = local_policy();
    deadline.deadline = Clock::now() + std::chrono::milliseconds(300);
    started           = Clock::now();
    expect_outcome("a server slower than the deadline", fetch(url(server.url("/slow")), deadline),
                   Outcome::Deadline);
    expect(Clock::now() - started < std::chrono::seconds(2), "the deadline did not stop the fetch");

    auto cancelled         = local_policy();
    std::atomic<bool> flag = false;
    cancelled.is_cancelled = [&flag] { return flag.load(); };
    std::thread cancel([&flag] {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        flag = true;
    });
    started = Clock::now();
    expect_outcome("a fetch cancelled mid-transfer", fetch(url(server.url("/slow")), cancelled),
                   Outcome::Cancelled);
    cancel.join();
    expect(Clock::now() - started < std::chrono::seconds(2), "cancellation did not stop the fetch");

    auto expired     = local_policy();
    expired.deadline = Clock::now() - std::chrono::milliseconds(1);
    expect_outcome("an expired deadline", fetch(url(server.url("/image")), expired),
                   Outcome::Deadline);

    auto off         = local_policy();
    off.allow_remote = false;
    expect_outcome("remote fetches turned off", fetch(url(server.url("/image")), off),
                   Outcome::Invalid);
}

void addresses(const LocalServer& server) {
    acquire::Policy policy;
    policy.connect_timeout_ms = 2'000;
    policy.timeout_ms         = 5'000;
    const std::string port    = std::to_string(server.port());
    for (const std::string& host : {std::string("127.0.0.1"), std::string("localhost"),
                                    std::string("[::1]"), std::string("0.0.0.0"),
                                    std::string("[::ffff:127.0.0.1]")}) {
        expect_outcome("a loopback host " + host,
                       fetch(url("http://" + host + ":" + port + "/image"), policy),
                       Outcome::Invalid);
    }
    for (const char* bad :
         {"ftp://example.com/a.png", "file:///etc/hosts", "gopher://example.com/",
          "http://:80/a.png", "http://", "not a url", "https://user@example.com/a.png"}) {
        expect_outcome(std::string("the URL ") + bad, fetch(url(bad), policy), Outcome::Invalid);
    }

    for (const char* address :
         {"0.1.2.3",        "10.0.0.1",       "100.64.0.1",       "100.127.255.255", "127.0.0.1",
          "169.254.169.254", "172.16.0.1",    "172.31.255.255",   "192.0.0.8",       "192.0.2.1",
          "192.88.99.1",    "192.168.1.1",    "198.18.0.1",       "198.19.255.255",  "198.51.100.7",
          "203.0.113.9",    "224.0.0.1",      "240.0.0.1",        "255.255.255.255", "::",
          "::1",            "::127.0.0.1",    "::ffff:10.0.0.1",  "::ffff:127.0.0.1",
          "64:ff9b::a00:1", "64:ff9b:1::1",   "100::1",           "2001::1",         "2001:db8::1",
          "2002:a00:1::1",  "2002:7f00:1::1", "fc00::1",          "fd12:3456::1",    "fe80::1",
          "febf::1",        "fec0::1",        "ff02::1"}) {
        expect(acquire::detail::disallowed_numeric_address(address),
               std::string("the fetcher may reach ") + address);
    }
    for (const char* address : {"1.1.1.1", "8.8.8.8", "100.63.255.255", "100.128.0.1",
                                "172.15.255.255", "172.32.0.1", "192.0.1.1", "192.169.0.1",
                                "198.17.255.255", "198.20.0.1", "223.255.255.255",
                                "2606:4700:4700::1111", "2a00:1450:4001::200e",
                                "::ffff:8.8.8.8", "64:ff9b::808:808", "2002:808:808::1"}) {
        expect(!acquire::detail::disallowed_numeric_address(address),
               std::string("the fetcher refuses the public address ") + address);
    }
    bool refused = false;
    try {
        (void)acquire::detail::disallowed_numeric_address("example.com");
    } catch (const std::invalid_argument&) { refused = true; }
    expect(refused, "a host name was classified as an address");
}

void serving_policy() {
    using ninfer::serve::ApiException;
    using ninfer::serve::media_acquire_policy;
    const acquire::Source link = url("https://example.com/a.png");
    bool refused               = false;
    try {
        (void)media_acquire_policy(link, 64 << 20, 0);
    } catch (const ApiException& error) {
        refused = error.error().status == 400 && error.error().code == "media_urls_disabled";
    }
    expect(refused, "a server without --media-urls accepted a media URL");

    expect(media_acquire_policy(link, 64 << 20, 20 << 20).max_bytes == (20U << 20),
           "--media-urls did not bound a URL fetch");
    expect(media_acquire_policy(link, 4 << 20, 20 << 20).max_bytes == (4U << 20),
           "a URL fetch passed the request's remaining media bytes");
    expect(!media_acquire_policy(link, 64 << 20, 20 << 20).allow_private_network,
           "serving lets a URL reach private networks");

    acquire::Source data;
    data.kind  = acquire::SourceKind::Data;
    data.value = "data:image/png;base64,AA==";
    expect(media_acquire_policy(data, 64 << 20, 0).max_bytes == (64U << 20),
           "a data URL depends on --media-urls");

    acquire::Source path;
    path.kind  = acquire::SourceKind::Path;
    path.value = "/etc/hosts";
    bool path_refused = false;
    try {
        (void)media_acquire_policy(path, 64 << 20, 20 << 20);
    } catch (const std::logic_error&) { path_refused = true; }
    expect(path_refused, "serving would read a local file");
}

} // namespace

int main() {
    try {
        LocalServer server;
        fetches(server);
        limits(server);
        addresses(server);
        serving_policy();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " media acquisition check(s) failed\n";
        return 1;
    }
    std::cout << "media acquisition tests passed\n";
    return 0;
}
