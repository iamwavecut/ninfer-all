#pragma once

// A test CA, an HTTPS server on a loopback port that presents a certificate the CA signed for
// "localhost" alone, and the environment that makes product/https_client.h trust the CA.

#include <httplib.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace ninfer::test_tls {

// ECDSA P-256, valid to 2126.
inline constexpr const char* kCa = R"(-----BEGIN CERTIFICATE-----
MIIBeDCCAR6gAwIBAgIUJOKH8gCJOQi8iNneJ9nYMue4b34wCgYIKoZIzj0EAwIw
GTEXMBUGA1UEAwwOTkluZmVyIHRlc3QgQ0EwIBcNMjYxMTI2MjE1MTUyWhgPMjEy
NjExMDIyMTUxNTJaMBkxFzAVBgNVBAMMDk5JbmZlciB0ZXN0IENBMFkwEwYHKoZI
zj0CAQYIKoZIzj0DAQcDQgAE2R5EIWfJ/cy9vUNBtp/QyIG7qTGz+YzZ+9/O1lfA
KhrkboeyQTEg68sCnens10iN+zONruavDJxp7VRACwWnvqNCMEAwHQYDVR0OBBYE
FJcQvJ8NsZRulfbWeQkbI9F4lxN9MA8GA1UdEwEB/wQFMAMBAf8wDgYDVR0PAQH/
BAQDAgEGMAoGCCqGSM49BAMCA0gAMEUCIQCA4i5eYlxE1oiejslwpgXYEiPwHWM7
kqvrDO4g60cm2AIgcYstmQ2sa1TqNSTYtHtYJzMiY1i8tjFGu/6NXCTBHWI=
-----END CERTIFICATE-----
)";
inline constexpr const char* kServerCertificate = R"(-----BEGIN CERTIFICATE-----
MIIBeTCCAR+gAwIBAgIUOAfqyMa/cGwFG62YYbz9BqSBzK4wCgYIKoZIzj0EAwIw
GTEXMBUGA1UEAwwOTkluZmVyIHRlc3QgQ0EwIBcNMjYxMTI2MjE1MTUyWhgPMjEy
NjExMDIyMTUxNTJaMBQxEjAQBgNVBAMMCWxvY2FsaG9zdDBZMBMGByqGSM49AgEG
CCqGSM49AwEHA0IABKPqszvIzhpH7V1LH1a1krauo4C6iT3Iv9xL5/IFPZkGhVEj
ZHr77/exsmi9Uuu0vwse7Ru81DvozV7ONzgoWjKjSDBGMAkGA1UdEwQCMAAwDgYD
VR0PAQH/BAQDAgeAMBMGA1UdJQQMMAoGCCsGAQUFBwMBMBQGA1UdEQQNMAuCCWxv
Y2FsaG9zdDAKBggqhkjOPQQDAgNIADBFAiALLHmZym78cBBKL90cMQ9etF+RjxqK
ZSDXKbRscvU+rQIhAM7PbaMzZj8A/fdbQ4Uh9yRefPeZND+grtpQ+TPb3jSB
-----END CERTIFICATE-----
)";
inline constexpr const char* kServerKey = R"(-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgqSQt0w/LwCS4bL0o
wKqscExc3LiT4vGG4V3rrwPaQ9WhRANCAASj6rM7yM4aR+1dSx9WtZK2rqOAuok9
yL/cS+fyBT2ZBoVRI2R6++/3sbJovVLrtL8LHu0bvNQ76M1ezjc4KFoy
-----END PRIVATE KEY-----
)";

// Sets an environment variable, or removes it for an empty value.
inline void set_environment(const char* name, const std::string& value) {
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

// kCa in a PEM file of the temporary directory, named after `tag`, that SSL_CERT_FILE names while
// this lives; SSL_CERT_DIR is unset meanwhile. Afterwards both are unset and the file is removed.
class TrustedCa {
public:
    explicit TrustedCa(const std::string& tag)
        : path_(std::filesystem::temp_directory_path() / ("ninfer-test-ca-" + tag + ".pem")) {
        std::ofstream(path_, std::ios::binary) << kCa;
        set_environment("SSL_CERT_DIR", "");
        set_environment("SSL_CERT_FILE", path_.string());
    }

    ~TrustedCa() {
        set_environment("SSL_CERT_FILE", "");
        set_environment("SSL_CERT_DIR", "");
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    TrustedCa(const TrustedCa&)            = delete;
    TrustedCa& operator=(const TrustedCa&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

// An HTTPS server on 127.0.0.1. Routes go on server() before start(); it listens on its own thread
// until destroyed.
class Server {
public:
    Server() : server_(httplib::SSLServer::PemMemory{kServerCertificate, 0, kServerKey, 0, nullptr, 0,
                                                     nullptr}) {
        if (!server_.is_valid()) { throw std::runtime_error("cannot start the local TLS server"); }
    }

    ~Server() {
        server_.stop();
        if (thread_.joinable()) { thread_.join(); }
    }

    Server(const Server&)            = delete;
    Server& operator=(const Server&) = delete;

    httplib::SSLServer& server() { return server_; }

    void start() {
        port_ = server_.bind_to_any_port("127.0.0.1");
        if (port_ <= 0) { throw std::runtime_error("cannot bind the local TLS server"); }
        thread_ = std::thread([this] { server_.listen_after_bind(); });
        server_.wait_until_ready();
    }

    [[nodiscard]] int port() const { return port_; }

    // An https URL of this server under `host`, which names 127.0.0.1.
    [[nodiscard]] std::string url(const std::string& host, const std::string& path) const {
        return "https://" + host + ":" + std::to_string(port_) + path;
    }

private:
    httplib::SSLServer server_;
    std::thread thread_;
    int port_ = 0;
};

} // namespace ninfer::test_tls
