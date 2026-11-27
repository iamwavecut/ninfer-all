/* NInfer's additions to the TF-PSA-Crypto configuration (cmake/NinferMbedTLS.cmake): the threading
 * layer, without which the PSA core is safe only when one thread at a time makes TLS calls. Its
 * mutexes and condition variables are the C++ standard library's (src/product/https_client.h). */
#define MBEDTLS_THREADING_C
#define MBEDTLS_THREADING_ALT
