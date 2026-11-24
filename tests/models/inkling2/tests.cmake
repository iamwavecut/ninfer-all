ninfer_add_test(ninfer_inkling2_kernels_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kernels.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_inkling2_kernels_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_op_oracle_options(ninfer_inkling2_kernels_test)

ninfer_add_test(ninfer_inkling2_kv_planes_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kv_planes.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_inkling2_kv_planes_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_inkling2_vision_kernels_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_kernels.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_inkling2_vision_kernels_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_inkling2_vision_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_real.cpp"
  LIBRARIES ninfer_model_runtime ninfer::json)
set_tests_properties(ninfer_inkling2_vision_real_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_inkling2_dspark_kernels_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dspark_kernels.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_inkling2_dspark_kernels_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_inkling2_loop_kv_real_test STANDALONE
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loop_kv_real.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_inkling2_loop_kv_real_test PROPERTIES SKIP_RETURN_CODE 77)
