/* The objects behind Mbed TLS's alternative threading layer (cmake/mbedtls/ninfer_crypto_config.h):
 * a std::mutex and a std::condition_variable that src/product/https_client.h allocates. */
#ifndef NINFER_MBEDTLS_THREADING_ALT_H
#define NINFER_MBEDTLS_THREADING_ALT_H

typedef struct {
    void *handle;
} mbedtls_platform_mutex_t;

typedef struct {
    void *handle;
} mbedtls_platform_condition_variable_t;

#endif
