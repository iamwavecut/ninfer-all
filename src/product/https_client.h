#pragma once

#include <httplib.h>

#include <cstdlib>
#include <memory>
#include <string>

namespace ninfer::product {

// An HTTPS client of the vendored cpp-httplib (its Mbed TLS backend, cmake/NinferMbedTLS.cmake)
// that verifies the server's certificate chain and host name. It trusts the PEM bundle that
// SSL_CERT_FILE names, else the certificate directory that SSL_CERT_DIR names, else the system's
// certificates: the Windows certificate store, the macOS keychain, or the first CA bundle of the
// common Linux paths that exists.
inline std::unique_ptr<httplib::SSLClient> verifying_https_client(const std::string& host, int port) {
    auto client = std::make_unique<httplib::SSLClient>(host, port);
    client->enable_server_certificate_verification(true);
    client->enable_server_hostname_verification(true);
    const char* file = std::getenv("SSL_CERT_FILE");
    const char* dir  = std::getenv("SSL_CERT_DIR");
    if (file != nullptr && *file != '\0') {
        client->set_ca_cert_path(file);
    } else if (dir != nullptr && *dir != '\0') {
        client->set_ca_cert_path("", dir);
    }
    return client;
}

} // namespace ninfer::product
