# Standalone CUDA enumeration probe, plus a real DeviceContext construction check.
#
# It deliberately does not link ninfer_core. The enumeration half needs to answer "which CUDA
# indices are the two cards, and can they reach each other" before the model loader is involved,
# on a machine where the answer is not obvious -- a display GPU commonly takes cuda=0 even when
# nvidia-smi lists it elsewhere, so the indices for --devices must be read from a CUDA-level
# enumeration rather than assumed.
#
# device.cu, its launch helpers (device_launch.cpp), the device stall watch it starts
# (device_wait.cu) and the peer preflight it calls (peer_preflight.cpp) are compiled in directly
# rather than linked from the core. They only include core/device.h, core/device_wait.h,
# core/peer_preflight.h and standard headers, so this costs four translation units and keeps the
# probe around 1 MB instead of the ~500 MB a core link would produce -- which matters because this
# binary gets uploaded to rented hardware to be run there.
add_executable(ninfer-multi-gpu-probe
  multi_gpu_probe.cu
  ${CMAKE_SOURCE_DIR}/src/core/device.cu
  ${CMAKE_SOURCE_DIR}/src/core/device_launch.cpp
  ${CMAKE_SOURCE_DIR}/src/core/device_wait.cu
  ${CMAKE_SOURCE_DIR}/src/core/peer_preflight.cpp)
target_include_directories(ninfer-multi-gpu-probe PRIVATE ${CMAKE_SOURCE_DIR}/src)
# The host compiler needs the toolkit headers for device_launch.cpp; nvcc links the static runtime.
# DeviceBinding reads the thread's current context through the driver API (cuCtxGetCurrent), as the
# core does, so the probe links the driver library too.
target_link_libraries(ninfer-multi-gpu-probe PRIVATE CUDA::toolkit CUDA::cuda_driver)
set_target_properties(ninfer-multi-gpu-probe PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

# Go/no-go probe for tensor-parallel all-reduce transports (peer push, mapped host, staged host,
# optionally NCCL): per-collective latency in graphs, a correctness soak, and the pinned-memory
# portability check. Standalone like the probe above, so it can be uploaded to rented hardware.
add_executable(ninfer-tp-probe tp_probe.cu)
set_target_properties(ninfer-tp-probe PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

# Times the engine's own tensor-parallel exchange (core/tensor_parallel.cu) over the ranks of a
# device list: every route (host memory, peer reads, peer writes; words, whole reduces, the
# reduce-scatter and all-gather, scatter flags) at every size, and the bands the exchange measures
# for itself. The exchange only includes core/device.h and a public enum header, so it is compiled
# in with the device sources, as for ninfer-multi-gpu-probe, instead of linking the core.
add_executable(ninfer-tp-exchange-bench
  tp_exchange_bench.cu
  ${CMAKE_SOURCE_DIR}/src/core/tensor_parallel.cu
  ${CMAKE_SOURCE_DIR}/src/core/device.cu
  ${CMAKE_SOURCE_DIR}/src/core/device_launch.cpp
  ${CMAKE_SOURCE_DIR}/src/core/device_wait.cu
  ${CMAKE_SOURCE_DIR}/src/core/peer_preflight.cpp)
target_include_directories(ninfer-tp-exchange-bench PRIVATE ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(ninfer-tp-exchange-bench PRIVATE CUDA::toolkit CUDA::cuda_driver)
set_target_properties(ninfer-tp-exchange-bench PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

# Measures what the dequantise-to-int8 prefill route would cost on the weight side: the group-scale
# spread within a row, and the relative L2 of reconstructing a weight as int8 with one scale per
# row. Reads the real artifact, because the answer depends on the actual scale distribution.
add_executable(ninfer-w4-row-scale-error w4_row_scale_error.cpp)
target_link_libraries(ninfer-w4-row-scale-error PRIVATE ninfer_artifact)
ninfer_internal_includes(ninfer-w4-row-scale-error)
set_target_properties(ninfer-w4-row-scale-error PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
