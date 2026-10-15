cmake_minimum_required(VERSION 3.28)

# The current implementation supports Ampere sm_86 and early Ada sm_89, plus sm_80 (GA100, e.g.
# CMP 170HX) as an unmeasured compatibility target: it shares the sm_86 instruction set the
# kernels use, but none of its route tables have been swept. sm_70 (Tesla V100) builds the same
# compatibility path with the Ampere warp primitives it lacks (cp.async, ldmatrix, the m16n8
# mma.sync shapes) emulated in src/ops/common, and its own prefill and attention routes. Turing
# sm_75 (TITAN RTX, RTX 20 series) runs the same mma.sync path through the Turing tier below. Set
# and validate the architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to build")
endif()
# 120a is admitted locally as an unqualified target. Consumer Blackwell is Ada-like at the warp
# level - mma.sync, ldmatrix and cp.async recompile natively, and there is no tcgen05 or tensor
# memory at all - so the compatibility path below is what actually runs there.
# "75;86" is the one mixed build: one binary for a Turing card and Ampere cards in the same
# process (a layer pipeline across a TITAN RTX and RTX 30 series cards). Its kernels take the
# Turing shapes and Turing kernel variants on every card, so the host launches each kernel the same
# way on both, and each architecture's device code takes its own instructions (NINFER_SM75_DEVICE).
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(70|75|80|86|89|120a|75;86)$")
  message(FATAL_ERROR
    "NInfer supports CMAKE_CUDA_ARCHITECTURES=70, 75, 80, 86, 89, 120a or 75;86; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# CUDA 13 removed Volta from nvcc's offline targets, so a 70 build stays on the CUDA 12 branch.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70" AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  message(FATAL_ERROR
    "A 70 build needs CUDA 12.x; CUDA ${CMAKE_CUDA_COMPILER_VERSION} no longer compiles sm_70")
endif()

# CUDA 12.8 and 12.9 miscompile sm_120a code. 12.8's ptxas reads a register under one predicate
# that it only wrote under another: on an RTX 5090 the PLE convolution then loads from an address it
# never computed, and the PLE injection, convolution and attention tests fail, upstream's convolution
# test included. With 12.9 the NVFP4 small-T attention kernels still go wrong, with no change in
# the sources: a wide verification's split reduce reads from a wild address, and two requests
# decoded together over an NVFP4 KV cache draw from corrupt logits. CUDA 13.1 builds them correctly.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "120a" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.1)
  message(FATAL_ERROR
    "A 120a build needs CUDA 13.1 or newer; CUDA ${CMAKE_CUDA_COMPILER_VERSION} miscompiles sm_120a "
    "kernels")
endif()

# NINFER_SM8X_COMPAT gates every Ampere/early-Ada compatibility path. ninfer_ops also sets it
# PUBLIC for its own consumers, but that link is PRIVATE through ninfer_engine, so apps and tests
# would not otherwise see it. Defining it globally keeps a single consistent answer for every
# target, including apps/ninfer-perplexity, whose upstream default KV profile is FP8.
#
# A 120a build gets the same path by default. Upstream's native routes would be available there,
# but they are a separate, unqualified code path, while the compatibility one is the tested route
# and is the only one that promotes T2G128 weights to the integer-activation prefill GEMM. Set
# NINFER_SM120_NATIVE=ON to build the native routes instead.
option(NINFER_SM120_NATIVE
       "On a 120a build, compile upstream's native routes instead of the mma.sync path" OFF)
if(CMAKE_CUDA_ARCHITECTURES MATCHES "^(70|75|80|86|89|75;86)$" OR NOT NINFER_SM120_NATIVE)
  set(NINFER_COMPAT_PATH ON)
  add_compile_definitions(NINFER_SM8X_COMPAT=1)
else()
  set(NINFER_COMPAT_PATH OFF)
endif()
# NINFER_SM70 marks a Volta build on the host side too (route selection, startup checks), where
# __CUDA_ARCH__ is not defined.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  add_compile_definitions(NINFER_SM70=1)
endif()
# NINFER_SM75 is the Turing tier of that path, for host and device code alike: the device helpers
# replace the sm_80+ instructions (cp.async, BF16/TF32 and m16n8k16/k32 MMA, packed conversions,
# warp reductions) with exact Turing forms (src/ops/common/memory.cuh, mma.cuh), kernels fit the
# 64 KiB per-block shared memory, and the host knows which routes this binary carries.
# A "75;86" build has it too; its Ampere code keeps the Ampere instructions (NINFER_SM75_DEVICE).
if(CMAKE_CUDA_ARCHITECTURES MATCHES "^75(;86)?$")
  add_compile_definitions(NINFER_SM75=1)
endif()
# Programmatic dependent launch lets a captured decode-graph kernel stage its weights while the
# kernel before it finishes (src/core/pdl.cuh). It needs sm_90 or newer, so the compatibility path
# launches in stream order; NINFER_PDL=ON turns it on for a 120a build of that path. The native
# routes always use it.
# MSVC cannot pass the over-aligned CUtensorMap descriptors of the NVFP4 TMA kernels by value, so
# Windows builds stage them through device memory with a capturable kernel
# (src/core/tma_descriptor_staging.cuh). NINFER_TMA_STAGING=ON builds that path elsewhere, to test
# it on the GPUs that run the TMA routes.
option(NINFER_TMA_STAGING "Pass NVFP4 TMA descriptors through device memory, as Windows does" OFF)
if(WIN32 OR NINFER_TMA_STAGING)
  add_compile_definitions(NINFER_TMA_STAGED_DESCRIPTORS=1)
endif()
option(NINFER_PDL "On a 120a build, launch decode-graph kernels as programmatic dependents" OFF)
if(NINFER_PDL)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
    message(FATAL_ERROR "NINFER_PDL requires CMAKE_CUDA_ARCHITECTURES=120a")
  endif()
  add_compile_definitions(NINFER_PDL=1)
endif()
# Every 120a build keeps the FP8 A8 and NVFP4 W4A4 units, the compatibility path included: without
# them an NVFP4 weight has no route wider than 16 columns, so an NVFP4 artifact could not start, and
# FP8 weights prefill through their dequantizing A16 route. They are selected only by FP8 and NVFP4
# weights, so every other format keeps the path chosen above.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  add_compile_definitions(NINFER_SM120_NVFP4=1 NINFER_SM120_FP8=1)
endif()

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI, server and perplexity evaluator" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)
option(NINFER_SFU_SIGMOID_SILU
       "Evaluate sigmoid and SiLU on the SFU instead of with expf and a divide" OFF)
if(NINFER_SFU_SIGMOID_SILU)
  add_compile_definitions(NINFER_SFU_SIGMOID_SILU=1)
endif()
option(NINFER_SFU_SOFTPLUS
       "Evaluate the GDN gate's softplus on the SFU instead of expf and log1pf" OFF)
if(NINFER_SFU_SOFTPLUS)
  add_compile_definitions(NINFER_SFU_SOFTPLUS=1)
endif()
option(NINFER_BF16_RESIDUAL_ADD
       "Add BF16 linear projections onto the residual in bf16 instead of fp32" OFF)
if(NINFER_BF16_RESIDUAL_ADD)
  add_compile_definitions(NINFER_BF16_RESIDUAL_ADD=1)
endif()
# Windows only: disk-tier restores through Microsoft DirectStorage, for --disk-kv-directstorage.
# dstorage.dll is delay-loaded, so a binary that never asks for DirectStorage runs without it.
option(NINFER_DIRECTSTORAGE
       "On Windows, offer --disk-kv-directstorage: disk-tier restores through DirectStorage" OFF)
if(WIN32 AND NINFER_DIRECTSTORAGE)
  include(FetchContent)
  FetchContent_Declare(directstorage
    URL "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.DirectStorage/1.3.0"
    DOWNLOAD_NAME "directstorage.1.3.0.zip")
  FetchContent_MakeAvailable(directstorage)
  add_library(ninfer_directstorage INTERFACE)
  target_include_directories(ninfer_directstorage INTERFACE
    "${directstorage_SOURCE_DIR}/native/include")
  target_link_libraries(ninfer_directstorage INTERFACE
    "${directstorage_SOURCE_DIR}/native/lib/x64/dstorage.lib" delayimp)
  target_link_options(ninfer_directstorage INTERFACE "/DELAYLOAD:dstorage.dll")
  set(NINFER_DIRECTSTORAGE_DLLS
    "${directstorage_SOURCE_DIR}/native/bin/x64/dstorage.dll"
    "${directstorage_SOURCE_DIR}/native/bin/x64/dstoragecore.dll")
  add_compile_definitions(NINFER_DIRECTSTORAGE=1)
endif()
# The largest attention units end a build on one single-threaded ptxas. --split-compile lets nvcc
# optimize one unit on several threads; 0 means every core, which a parallel ninja multiplies by its
# job count, so a bound is usually what is wanted.
set(NINFER_NVCC_SPLIT_COMPILE "" CACHE STRING
    "Threads nvcc may use per translation unit (--split-compile N); empty is off, 0 is every core")
if(NOT NINFER_NVCC_SPLIT_COMPILE STREQUAL "")
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--split-compile=${NINFER_NVCC_SPLIT_COMPILE}>)
endif()
option(NINFER_PTXAS_VERBOSE "Print each kernel's registers and local-memory spills (-Xptxas=-v)" OFF)
if(NINFER_PTXAS_VERBOSE)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# NInfer uses no C++20 modules. Scanning for them, CMake's default with C++20 and Ninja, gives every
# C++ source a dependency-scan step and compiler options under which ccache does not cache it: CI
# recompiled every C++ object on every run.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(MSVC)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
  # /Zc:__cplusplus makes MSVC report the standard it is actually compiling. Without it the macro
  # stays at 199711L even under /std:c++20, and CUDA's <vector_types.h> then hides the constexpr
  # dim3 constructors behind its own __cplusplus >= 201103L test, which fails to compile every
  # translation unit that builds a dim3 in a constant expression.
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>
    # spdlog's bundled fmt asserts the source/execution charset is UTF-8; MSVC defaults to the
    # active code page instead. Needed by every translation unit that includes fmt/spdlog
    # headers, not just spdlog's own .cpp files, so this is global rather than per-target.
    $<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
  # CUDA's static runtime requests LIBCMT even when the application and vcpkg dependencies use
  # the project-wide /MD runtime. Keep one CRT in the final process. $<HOST_LINK:...> keeps this
  # off the CUDA relocatable device-link step, which forwards it to cl.exe without the /link
  # passthrough separator and only produces a harmless but noisy D9002 "unknown option" warning.
  add_link_options($<HOST_LINK:/NODEFAULTLIB:LIBCMT>)
else()
  # Schedule dispatch in this project is a switch per Op mapping a schedule id to a kernel launch,
  # and a missing `return` there is close to invisible: the Op runs the intended kernel and then
  # the next case's kernel over the top of it. Both compute the same result, so the output stays
  # correct, every test passes, and the only symptom is that the Op costs twice what it should.
  # Two of these shipped in the last catch-up merge (q4_q5 attention input, q5 linear add) and
  # both were dormant because the live route table never selected the affected schedule.
  #
  # MSVC has no cheap equivalent -- C26819 needs /analyze -- so on Windows this is caught by
  # review and by comparing an Op's measured cost against its chosen kernel's. On every other
  # toolchain, and therefore in the direction upstream merges arrive from, it is a build error.
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:-Wimplicit-fallthrough>
    $<$<COMPILE_LANGUAGE:C,CXX>:-Werror=implicit-fallthrough>)

  # Keep the builder's absolute paths out of the shipped binaries. `__FILE__` and nvcc's own source
  # references otherwise embed wherever the tree happened to sit: the v0.9.0 Linux binaries carry
  # `/home/ash/ninfer-rel/src/...` about 200 times, and v0.8.1 carried `/mnt/c/ninfer-fork/...` 405
  # times. Not a secret -- the file names are public and no path contains a home directory -- but
  # it makes every assertion message shorter and more readable, and it makes two builds of the same
  # commit from different directories produce closer output.
  #
  # Verified on this box under real Linux g++: `/tmp/ftest/sub/a.cpp` becomes `./sub/a.cpp`, and
  # the absolute prefix disappears from `strings` entirely.
  #
  # **There is no working MSVC equivalent, and this was measured rather than assumed.** MSVC takes
  # `__FILE__` from the path as written on the command line, and CMake writes absolute paths;
  # `/d1trimfile:`, the usual suggestion, is undocumented and had *no effect* on `__FILE__` when
  # tested against 14.44.35207 with an absolute source path. So the Windows binaries keep their
  # ~466 occurrences until something better turns up.
  #
  # -Xcompiler reaches the host half of a .cu translation unit. Device-side `__FILE__` comes from
  # nvcc's own frontend and has no documented remapping flag, so it is not covered here.
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:-ffile-prefix-map=${PROJECT_SOURCE_DIR}=.>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffile-prefix-map=${PROJECT_SOURCE_DIR}=.>)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ninfer_link=1)
  set(CMAKE_JOB_POOL_LINK ninfer_link)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
  message(FATAL_ERROR
    "NInfer requires CUDA 12.8 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()

# Tests also exercise product input, logging and protocol adapters.
set(NINFER_BUILD_PRODUCT_SUPPORT OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_PRODUCT_SUPPORT ON)
endif()

include(cmake/Dependencies.cmake)
include(cmake/NinferTargets.cmake)

# The build id is stamped at BUILD time (see cmake/GenerateBuildId.cmake) into a generated header,
# so it reflects the tree actually being compiled rather than the last configure. Every product
# binary records it in its startup log ("build <id>"), and calibrated device profiles record it as
# their provenance (src/runtime/engine/device_profile_build_id.cpp). The header is only rewritten
# when its content changes, so only the files that include it recompile when the id does.
set(NINFER_BUILD_ID_HEADER "${CMAKE_BINARY_DIR}/generated/ninfer_build_id.h")
add_custom_target(ninfer_build_id_stamp ALL
  COMMAND ${CMAKE_COMMAND}
          -DSRC_DIR=${CMAKE_SOURCE_DIR}
          -DOUT=${NINFER_BUILD_ID_HEADER}
          -P ${CMAKE_SOURCE_DIR}/cmake/GenerateBuildId.cmake
  BYPRODUCTS ${NINFER_BUILD_ID_HEADER}
  COMMENT "Stamping NINFER build id from git (build-time)"
  VERBATIM)

# A product binary depends on the stamp, and lists the header as a source so a changed id
# recompiles its main.cpp under every generator.
function(ninfer_stamp_build_id target)
  add_dependencies(${target} ninfer_build_id_stamp)
  target_sources(${target} PRIVATE ${NINFER_BUILD_ID_HEADER})
  target_include_directories(${target} PRIVATE ${CMAKE_BINARY_DIR}/generated)
endfunction()

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
  add_subdirectory(tools)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
