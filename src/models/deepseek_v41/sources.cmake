target_sources(ninfer_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/engram.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/model.cpp"
)
target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/cpu_experts.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/executor.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/vision.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/vision_kernels.cu"
)
# The FP32 products of the compressor, indexer, router and hyper-connections (prompt chunks),
# and the vision tower's BF16 products.
target_link_libraries(ninfer_model_runtime PRIVATE ${NINFER_CUBLAS_TARGET})
