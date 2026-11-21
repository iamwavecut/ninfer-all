target_sources(ninfer_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/model.cpp"
)
target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/cpu_experts.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/executor.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/kv_format.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/kv_planes.cu"
  "${CMAKE_CURRENT_LIST_DIR}/vision.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/vision_kernels.cu"
)
# SiTU's host arithmetic matches the device's bit for bit only without contraction (situ.h).
if(NOT MSVC)
  set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/cpu_experts.cpp"
    PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()
