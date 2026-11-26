find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Windows links the static CUDA runtime so shipped binaries do not depend on a cudart DLL.
# NINFER_STATIC_CUDA_LIBS links the CUDA runtime and cuBLAS statically on every platform.
if(NINFER_STATIC_CUDA_LIBS)
  set(NINFER_CUDART_TARGET CUDA::cudart_static)
  set(NINFER_CUBLAS_TARGET CUDA::cublas_static)
else()
  if(WIN32)
    set(NINFER_CUDART_TARGET CUDA::cudart_static)
  else()
    set(NINFER_CUDART_TARGET CUDA::cudart)
  endif()

  set(NINFER_CUBLAS_TARGET CUDA::cublas)
endif()

# The Windows build takes FFmpeg from vcpkg's CMake packages. Linux builds FFmpeg from its pinned
# release and links it statically (cmake/NinferFFmpeg.cmake). Both paths produce the same consumer
# target. With vcpkg's CMake toolchain its own FFmpeg find wrapper serves find_package.
# Without it a prebuilt triplet tree serves them:
# NINFER_VCPKG_TREE, which scripts\build-native.bat passes, or else the classic tree that the
# VCPKG_ROOT and VCPKG_TARGET_TRIPLET environment variables name. cmake/FindFFMPEG.cmake reads it
# and names it FFMPEG_VCPKG_TREE, for staging its DLLs beside the executables.
set(NINFER_VCPKG_TREE "" CACHE PATH
    "Prebuilt vcpkg triplet tree, <installed directory>/<triplet>, for a Windows build without the vcpkg toolchain")
if(WIN32 AND NOT VCPKG_TOOLCHAIN)
  set(_ninfer_vcpkg_tree "${NINFER_VCPKG_TREE}")
  if(NOT _ninfer_vcpkg_tree AND DEFINED ENV{VCPKG_ROOT} AND DEFINED ENV{VCPKG_TARGET_TRIPLET})
    set(_ninfer_vcpkg_tree "$ENV{VCPKG_ROOT}/installed/$ENV{VCPKG_TARGET_TRIPLET}")
  endif()
  if(NOT _ninfer_vcpkg_tree)
    message(FATAL_ERROR
      "A Windows build takes FFmpeg from vcpkg: configure with the vcpkg toolchain "
      "(CMAKE_TOOLCHAIN_FILE), or name a prebuilt triplet tree in NINFER_VCPKG_TREE "
      "(scripts\\build-native.bat passes one)")
  endif()
  file(TO_CMAKE_PATH "${_ninfer_vcpkg_tree}" _ninfer_vcpkg_tree)
  list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
  list(APPEND CMAKE_PREFIX_PATH "${_ninfer_vcpkg_tree}")
endif()
if(WIN32)
  find_package(FFMPEG REQUIRED)
  add_library(ninfer_ffmpeg_dependencies INTERFACE)
  target_include_directories(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_INCLUDE_DIRS})
  target_link_directories(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARY_DIRS})
  target_link_libraries(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARIES})
  set(NINFER_FFMPEG_TARGET ninfer_ffmpeg_dependencies)
else()
  include(${CMAKE_CURRENT_LIST_DIR}/NinferFFmpeg.cmake)
endif()

# Repository-pinned header dependencies. No configure-time downloads.
add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

# Source base for the custom-template frontend; consumers will link it explicitly.
add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)

# Tokenizer-aware grammar compiler and matcher behind structured (JSON) output.
add_subdirectory(third_party/xgrammar EXCLUDE_FROM_ALL)

if(NINFER_BUILD_PRODUCT_SUPPORT)
  # cpp-httplib serves the HTTP API and fetches media URLs; Mbed TLS (cmake/NinferMbedTLS.cmake) is
  # its TLS backend. Every target that includes the header sees the same backend, so its classes
  # have one layout in the whole program.
  include(${CMAKE_CURRENT_LIST_DIR}/NinferMbedTLS.cmake)
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  target_compile_definitions(ninfer::httplib INTERFACE CPPHTTPLIB_MBEDTLS_SUPPORT)
  target_link_libraries(ninfer::httplib INTERFACE ${NINFER_MBEDTLS_TARGET})
  add_subdirectory(third_party/spdlog)
endif()
