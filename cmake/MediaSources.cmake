# Source releases that builds compile and link statically: the media decoders of Linux builds
# (cmake/NinferFFmpeg.cmake) and, on every platform, the TLS library of media URL fetches
# (cmake/NinferMbedTLS.cmake). scripts/fetch-media-sources.sh downloads them and checks these
# digests; the build itself never downloads anything. FFmpeg's archive is verified against the
# FFmpeg release signing key FCF986EA15E6E293A5644F10B4322F04D67658D8 and dav1d's against the
# VideoLAN release signing key 65F7C6B4206BD057A7EB73787180713BE58D1ADC before a digest is pinned
# here; Mbed TLS's against the SHA-256 file published with its GitHub release.
#
# FFmpeg's and dav1d's URL lists name the release server first, then Gentoo's distfiles mirror of
# the same file, which the pinned digest checks the same way; Gentoo has no copy of the Mbed TLS
# release archive. A mirror directory is the first two hex digits of the archive name's BLAKE2b
# digest:
#   python3 -c 'import hashlib, sys; print(hashlib.blake2b(sys.argv[1].encode()).hexdigest()[:2])' NAME
set(NINFER_FFMPEG_VERSION 9.0.2)
set(NINFER_FFMPEG_ARCHIVE ffmpeg-${NINFER_FFMPEG_VERSION}.tar.xz)
set(NINFER_FFMPEG_URLS
    "https://ffmpeg.org/releases/${NINFER_FFMPEG_ARCHIVE}"
    "https://distfiles.gentoo.org/distfiles/62/${NINFER_FFMPEG_ARCHIVE}")
set(NINFER_FFMPEG_SHA256 8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e)
set(NINFER_FFMPEG_LICENSE ffmpeg-${NINFER_FFMPEG_VERSION}/COPYING.LGPLv2.1)

set(NINFER_DAV1D_VERSION 1.5.4)
set(NINFER_DAV1D_ARCHIVE dav1d-${NINFER_DAV1D_VERSION}.tar.xz)
set(NINFER_DAV1D_URLS
    "https://downloads.videolan.org/pub/videolan/dav1d/${NINFER_DAV1D_VERSION}/${NINFER_DAV1D_ARCHIVE}"
    "https://distfiles.gentoo.org/distfiles/8b/${NINFER_DAV1D_ARCHIVE}")
set(NINFER_DAV1D_SHA256 686616b7c69eb88d44459391ab25cac13b6647a3b288835c5784e71c1514a5c5)
set(NINFER_DAV1D_LICENSE dav1d-${NINFER_DAV1D_VERSION}/COPYING)

set(NINFER_MBEDTLS_VERSION 4.3.0)
set(NINFER_MBEDTLS_ARCHIVE mbedtls-${NINFER_MBEDTLS_VERSION}.tar.bz2)
set(NINFER_MBEDTLS_URLS
    "https://github.com/Mbed-TLS/mbedtls/releases/download/mbedtls-${NINFER_MBEDTLS_VERSION}/${NINFER_MBEDTLS_ARCHIVE}")
set(NINFER_MBEDTLS_SHA256 1c87cba094cb40f097ebe0eba440a74939b71dab4e1d0d566bc90749ff381d18)
set(NINFER_MBEDTLS_LICENSE mbedtls-${NINFER_MBEDTLS_VERSION}/LICENSE)
# The options of its CMake build (cmake/NinferMbedTLS.cmake), which packages record.
set(NINFER_MBEDTLS_OPTIONS
  ENABLE_PROGRAMS=OFF ENABLE_TESTING=OFF GEN_FILES=OFF MBEDTLS_FATAL_WARNINGS=OFF
  TF_PSA_CRYPTO_FATAL_WARNINGS=OFF USE_STATIC_MBEDTLS_LIBRARY=ON USE_SHARED_MBEDTLS_LIBRARY=OFF
  USE_STATIC_TF_PSA_CRYPTO_LIBRARY=ON USE_SHARED_TF_PSA_CRYPTO_LIBRARY=OFF
  DISABLE_PACKAGE_CONFIG_AND_INSTALL=ON)

set(NINFER_MEDIA_SOURCES FFMPEG DAV1D MBEDTLS)
