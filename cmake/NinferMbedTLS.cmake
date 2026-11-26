include_guard(GLOBAL)
include(${CMAKE_CURRENT_LIST_DIR}/MediaSources.cmake)

# Every platform compiles Mbed TLS from its pinned release (cmake/MediaSources.cmake) and links it
# statically, as the TLS backend of cpp-httplib: media URL fetches (src/product/media_acquire) verify
# HTTPS servers with it, and no binary depends on a system TLS or HTTP library. Its configuration is
# the release's default (TLS 1.2 and 1.3, X.509) without programs or tests.
set(NINFER_MEDIA_SOURCES_DIR "${PROJECT_SOURCE_DIR}/third_party/media-sources" CACHE PATH
    "Directory with the release archives cmake/MediaSources.cmake pins (scripts/fetch-media-sources.sh)")
set(_ninfer_mbedtls_archive "${NINFER_MEDIA_SOURCES_DIR}/${NINFER_MBEDTLS_ARCHIVE}")
if(NOT EXISTS "${_ninfer_mbedtls_archive}")
  message(FATAL_ERROR
    "${_ninfer_mbedtls_archive} is missing. Run scripts/fetch-media-sources.sh, or set "
    "NINFER_MEDIA_SOURCES_DIR to a directory that holds the archives cmake/MediaSources.cmake pins.")
endif()
file(SHA256 "${_ninfer_mbedtls_archive}" _ninfer_mbedtls_digest)
if(NOT _ninfer_mbedtls_digest STREQUAL NINFER_MBEDTLS_SHA256)
  message(FATAL_ERROR "${_ninfer_mbedtls_archive} is not the pinned Mbed TLS release")
endif()

# The release archive carries its generated sources, so the build needs neither Python nor the
# release's development scripts.
set(_ninfer_mbedtls_root "${CMAKE_BINARY_DIR}/third_party/mbedtls")
set(_ninfer_mbedtls_source "${_ninfer_mbedtls_root}/mbedtls-${NINFER_MBEDTLS_VERSION}")
if(NOT EXISTS "${_ninfer_mbedtls_source}/CMakeLists.txt")
  file(REMOVE_RECURSE "${_ninfer_mbedtls_root}")
  file(ARCHIVE_EXTRACT INPUT "${_ninfer_mbedtls_archive}" DESTINATION "${_ninfer_mbedtls_root}")
endif()
# scripts/package-release.ps1 ships this licence beside a Windows build; Linux packages take it from
# the archive (scripts/package-linux-release.py).
file(COPY_FILE "${_ninfer_mbedtls_source}/LICENSE" "${_ninfer_mbedtls_root}/LICENSE" ONLY_IF_DIFFERENT)

foreach(_ninfer_option IN LISTS NINFER_MBEDTLS_OPTIONS)
  string(REPLACE "=" ";" _ninfer_pair "${_ninfer_option}")
  list(GET _ninfer_pair 0 _ninfer_name)
  list(GET _ninfer_pair 1 _ninfer_value)
  set(${_ninfer_name} ${_ninfer_value} CACHE INTERNAL "Mbed TLS build option (cmake/NinferMbedTLS.cmake)")
endforeach()
# The shared runtime may link the fetcher, so the archives are position-independent.
set(_ninfer_pic ${CMAKE_POSITION_INDEPENDENT_CODE})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory("${_ninfer_mbedtls_source}" "${_ninfer_mbedtls_root}/build" EXCLUDE_FROM_ALL SYSTEM)
set(CMAKE_POSITION_INDEPENDENT_CODE ${_ninfer_pic})

add_library(ninfer_mbedtls INTERFACE)
target_link_libraries(ninfer_mbedtls INTERFACE MbedTLS::mbedtls MbedTLS::mbedx509 TF-PSA-Crypto::tfpsacrypto)
if(WIN32)
  # cpp-httplib reads the Windows certificate store.
  target_link_libraries(ninfer_mbedtls INTERFACE crypt32)
endif()
set(NINFER_MBEDTLS_TARGET ninfer_mbedtls)
