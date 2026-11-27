#pragma once

#include <httplib.h>
#include <mbedtls/threading.h>

#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <string>

namespace ninfer::product {

namespace tls_threading {

// Mbed TLS's threading layer (cmake/mbedtls/ninfer_crypto_config.h) on the standard library's
// mutexes and condition variables. Mbed TLS never locks a mutex its thread holds and unlocks from
// the locking thread.
inline std::mutex& mutex_of(mbedtls_platform_mutex_t* mutex) {
    return *static_cast<std::mutex*>(mutex->handle);
}
inline std::condition_variable& condition_of(mbedtls_platform_condition_variable_t* condition) {
    return *static_cast<std::condition_variable*>(condition->handle);
}
inline int mutex_init(mbedtls_platform_mutex_t* mutex) {
    mutex->handle = new (std::nothrow) std::mutex;
    return mutex->handle != nullptr ? 0 : MBEDTLS_ERR_THREADING_USAGE_ERROR;
}
inline void mutex_destroy(mbedtls_platform_mutex_t* mutex) {
    delete static_cast<std::mutex*>(mutex->handle);
    mutex->handle = nullptr;
}
inline int mutex_lock(mbedtls_platform_mutex_t* mutex) {
    try {
        mutex_of(mutex).lock();
        return 0;
    } catch (...) { return MBEDTLS_ERR_THREADING_USAGE_ERROR; }
}
inline int mutex_unlock(mbedtls_platform_mutex_t* mutex) {
    mutex_of(mutex).unlock();
    return 0;
}
inline int condition_init(mbedtls_platform_condition_variable_t* condition) {
    condition->handle = new (std::nothrow) std::condition_variable;
    return condition->handle != nullptr ? 0 : MBEDTLS_ERR_THREADING_USAGE_ERROR;
}
inline void condition_destroy(mbedtls_platform_condition_variable_t* condition) {
    delete static_cast<std::condition_variable*>(condition->handle);
    condition->handle = nullptr;
}
inline int condition_signal(mbedtls_platform_condition_variable_t* condition) {
    condition_of(condition).notify_one();
    return 0;
}
inline int condition_broadcast(mbedtls_platform_condition_variable_t* condition) {
    condition_of(condition).notify_all();
    return 0;
}
// The caller holds `mutex` and holds it again on return, as after pthread_cond_wait.
inline int condition_wait(mbedtls_platform_condition_variable_t* condition,
                          mbedtls_platform_mutex_t* mutex) {
    std::unique_lock lock(mutex_of(mutex), std::adopt_lock);
    int result = 0;
    try {
        condition_of(condition).wait(lock);
    } catch (...) { result = MBEDTLS_ERR_THREADING_USAGE_ERROR; }
    lock.release();
    return result;
}

} // namespace tls_threading

// Installs the threading layer once per process, before its first Mbed TLS call; every TLS client
// and server of the program starts here.
inline void start_tls_threading() {
    static const bool started = [] {
        mbedtls_threading_set_alt(tls_threading::mutex_init, tls_threading::mutex_destroy,
                                  tls_threading::mutex_lock, tls_threading::mutex_unlock,
                                  tls_threading::condition_init, tls_threading::condition_destroy,
                                  tls_threading::condition_signal,
                                  tls_threading::condition_broadcast,
                                  tls_threading::condition_wait);
        return true;
    }();
    (void)started;
}

// An HTTPS client of the vendored cpp-httplib (its Mbed TLS backend, cmake/NinferMbedTLS.cmake)
// that verifies the server's certificate chain and host name. It trusts the PEM bundle that
// SSL_CERT_FILE names, else the certificate directory that SSL_CERT_DIR names, else the system's
// certificates: the Windows certificate store, the macOS keychain, or the first CA bundle of the
// common Linux paths that exists. Clients may run on several threads at once.
inline std::unique_ptr<httplib::SSLClient> verifying_https_client(const std::string& host, int port) {
    start_tls_threading();
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
