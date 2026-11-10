target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/causal_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/merge.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_fp8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_nvfp4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_k8v4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_fp8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_sm70.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_nvfp4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_k8v4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/packed_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/context_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/sliding_window_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/launch.cu"
)

# The INT8-family small-T partial launch, one unit per query width, geometry and input. These units
# carry more than half of the release's device code; Windows kernel-DLL builds link them into the
# kernel DLL (src/ops/CMakeLists.txt) so that no PE image passes what the Windows loader maps. There
# they form an archive of their own, pruned like the others, with its own copy of the route key; the
# DLL exports only the instantiations.
set(_ninfer_small_t_i8_sources
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w1_h24_append.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w1_h24_cached.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w1_h16_append.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w1_h16_cached.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w2_h24_append.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w2_h24_cached.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w2_h16_append.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w2_h16_cached.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w3_h24_append.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w3_h24_cached.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w3_h16_append.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w3_h16_cached.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w4_h24_append.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w4_h24_cached.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w4_h16_append.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w4_h16_cached.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w5_h24_append.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w5_h24_cached.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w5_h16_append.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w5_h16_cached.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w6_h24_append.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w6_h24_cached.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w6_h16_append.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w6_h16_cached.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w7_h24_append.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w7_h24_cached.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w8_h24_append.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w8_h24_cached.cu"
)
if(NINFER_WINDOWS_KERNEL_DLL)
  # The launch units in the kernel DLL and the dispatcher in the Ops both read the route key, and
  # the DLL exports only the launch instantiations, so each image links its own copy of the key from
  # one archive of its own.
  add_library(ninfer_small_t_i8_route STATIC
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_route.cpp")
  add_library(ninfer_small_t_i8 STATIC ${_ninfer_small_t_i8_sources})
  # The route key is host code only: its archive takes no CUDA archive policy or PTX pruning.
  ninfer_cuda_archive(ninfer_small_t_i8)
  foreach(_ninfer_small_t_target ninfer_small_t_i8_route ninfer_small_t_i8)
    ninfer_internal_includes(${_ninfer_small_t_target})
    target_link_libraries(${_ninfer_small_t_target} PRIVATE ninfer_core)
    target_compile_definitions(${_ninfer_small_t_target} PRIVATE
      $<TARGET_PROPERTY:ninfer_ops,INTERFACE_COMPILE_DEFINITIONS>)
    # The PDB name of ninfer_ops keeps /Fd out of the difference, so the units' compile commands can
    # match those they had there and reuse the cached objects.
    set_target_properties(${_ninfer_small_t_target} PROPERTIES COMPILE_PDB_NAME ninfer_ops
      COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ninfer_ops.dir")
  endforeach()
  target_link_libraries(ninfer_small_t_i8 PRIVATE ninfer_small_t_i8_route)
  target_link_libraries(ninfer_ops PRIVATE ninfer_small_t_i8_route)
  list(LENGTH _ninfer_small_t_i8_sources NINFER_SMALL_T_I8_INSTANTIATIONS)
else()
  target_sources(ninfer_ops PRIVATE ${_ninfer_small_t_i8_sources}
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_route.cpp")
endif()

# The fast NVFP4 prompt kernel runs QK on block-scaled FP4 Tensor Cores (kind::mxf4nvf4), which
# only Blackwell has: a build with 120a carries it, and only a card running the 120a code takes it.
if(NINFER_ARCH_BLACKWELL)
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_nvfp4_fast.cu")
endif()
