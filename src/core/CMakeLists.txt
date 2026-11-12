# The device layer: device selection and binding, launch helpers, the device-wait watch, the peer
# preflight and the tensor-parallel exchange. It needs nothing else from the core, so it is an
# archive of its own: the standalone tools in tools/ link it, with the requirements declared here,
# instead of compiling copies of these units.
add_library(ninfer_core_device STATIC
  device.cu
  device_launch.cpp
  device_wait.cu
  peer_preflight.cpp
  tensor_parallel.cu
)
ninfer_internal_includes(ninfer_core_device)
# The architectures this build carries, for the message that refuses a device it cannot run. A
# list's semicolons would split the property into several definitions.
string(REPLACE ";" " " _ninfer_cuda_architectures "${NINFER_ARCHITECTURES}")
set_source_files_properties(device.cu PROPERTIES
  COMPILE_DEFINITIONS "NINFER_CUDA_ARCHITECTURES=\"${_ninfer_cuda_architectures}\"")
ninfer_cuda_archive(ninfer_core_device)
# DeviceBinding reads the thread's current context through the driver API (cuCtxGetCurrent).
target_link_libraries(ninfer_core_device PUBLIC
  ${NINFER_CUDART_TARGET} Threads::Threads CUDA::cuda_driver)
if(MSVC)
  # --help must run on build hosts without an NVIDIA driver. Resolve driver entry points only
  # when execution uses them; the CUDA runtime still checks device availability at startup. Every
  # target that links the core links this archive, so the delayed load covers the core's own
  # driver calls too.
  target_link_libraries(ninfer_core_device PUBLIC delayimp)
  target_link_options(ninfer_core_device INTERFACE "$<HOST_LINK:/DELAYLOAD:nvcuda.dll>")
endif()

add_library(ninfer_core STATIC
  arena.cu
  expert_parallel.cpp
  stage_link.cu
  tensor_split.cpp
  dtype.cpp
  gpu_clock_floor.cpp
  host_memory.cpp
  io_priority.cpp
  host_pages.cpp
  zero_pages.cpp
  host_worker_pool.cpp
  cpu_spin_pool.cpp
  expert_split.cpp
  expert_tier.cpp
  gdn_replay_records.cpp
  cyclic_kv_cache.cpp
  host_kv_arena.cpp
  disk_kv_store.cpp
  disk_kv_bridge.cpp
  direct_storage_reader.cpp
  file_read_queue.cpp
  expert_mirror.cpp
  expert_direct.cpp
  paged_kv_cache.cpp
  kv_expert_loan.cpp
  linear_attention_state.cpp
  layout.cpp
  tensor.cpp
  weight_view.cpp
  decode_graph.cpp
  cuda_host_handshake.cu
  mapped_host_copy.cu
  expert_delivery.cu
  evictable_kv_pool.cu
  evictable_weight_pool.cu
  vmm.cu
  growable_region.cu
  device_snapshot.cu
)
ninfer_internal_includes(ninfer_core)
ninfer_cuda_archive(ninfer_core)
# The virtual-memory and evictable pools call the driver API as well.
target_link_libraries(ninfer_core PUBLIC
  ninfer_core_device ${NINFER_CUDART_TARGET} CUDA::nvtx3 Threads::Threads CUDA::cuda_driver)
# gpu_clock_floor.cpp loads NVML at run time.
target_link_libraries(ninfer_core PRIVATE ${CMAKE_DL_LIBS})
if(WIN32 AND NINFER_DIRECTSTORAGE)
  target_link_libraries(ninfer_core PUBLIC ninfer_directstorage)
endif()
