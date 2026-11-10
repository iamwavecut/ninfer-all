# Standalone CUDA enumeration probe, plus a real DeviceContext construction check.
#
# The enumeration half needs to answer "which CUDA indices are the two cards, and can they reach
# each other" before the model loader is involved, on a machine where the answer is not obvious --
# a display GPU commonly takes cuda=0 even when nvidia-smi lists it elsewhere, so the indices for
# --devices must be read from a CUDA-level enumeration rather than assumed. It links only the core's
# device layer (ninfer_core_device), which brings its own link requirements: the CUDA runtime and
# driver libraries every NInfer program uses. The binary stays under 1 MB, which matters because it
# gets uploaded to rented hardware to be run there.
add_executable(ninfer-multi-gpu-probe multi_gpu_probe.cu)
ninfer_internal_includes(ninfer-multi-gpu-probe)
target_link_libraries(ninfer-multi-gpu-probe PRIVATE ninfer_core_device)
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
# for itself. The exchange belongs to the core's device layer, which this links, as
# ninfer-multi-gpu-probe does, instead of the whole core.
add_executable(ninfer-tp-exchange-bench tp_exchange_bench.cu)
ninfer_internal_includes(ninfer-tp-exchange-bench)
target_link_libraries(ninfer-tp-exchange-bench PRIVATE ninfer_core_device)
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
