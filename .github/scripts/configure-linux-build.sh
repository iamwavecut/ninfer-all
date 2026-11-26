#!/usr/bin/env bash
# Keep release and CI compilation inputs identical; CI additionally enables test targets.
set -euo pipefail
: "${ARCHS:?}"
case "${1:-OFF}" in ON|OFF) testing="${1:-OFF}" ;; *) exit 2 ;; esac
# Linux packages and the Docker images serve llama.cpp's WebUI at the release fetch-webui.sh pins.
webui="$(scripts/fetch-webui.sh | sed -n 's/^configure with -DNINFER_WEBUI_DIR=//p')"
test -n "$webui"
# FFmpeg, dav1d and Mbed TLS are compiled from their pinned release archives (cmake/MediaSources.cmake).
scripts/fetch-media-sources.sh
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" -DNINFER_SHARED_RUNTIME=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DBUILD_TESTING="$testing" -DNINFER_BUILD_BENCHMARKS=ON -DNINFER_WEBUI_DIR="$webui"
