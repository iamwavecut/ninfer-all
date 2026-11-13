ninfer_add_test(ninfer_deepseek_v41_kernels_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kernels.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_deepseek_v41_kernels_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_op_oracle_options(ninfer_deepseek_v41_kernels_test)

ninfer_add_test(ninfer_deepseek_v41_vision_kernels_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_kernels.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_deepseek_v41_vision_kernels_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_deepseek_v41_vision_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_real.cpp"
  LIBRARIES ninfer_model_runtime ninfer::json)
set_tests_properties(ninfer_deepseek_v41_vision_real_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_deepseek_v41_engram_table_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engram_table.cpp"
  LIBRARIES ninfer_model_loading)
