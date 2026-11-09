add_library(ninfer_linear_swiglu_test_support STATIC
  "${CMAKE_CURRENT_LIST_DIR}/linear_swiglu_test_common.cpp")
ninfer_test_includes(ninfer_linear_swiglu_test_support)
ninfer_op_oracle_options(ninfer_linear_swiglu_test_support)
target_link_libraries(ninfer_linear_swiglu_test_support PUBLIC ninfer_ops)

ninfer_add_op_test(ninfer_linear_swiglu_q4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_a16.cpp"
  LIBRARIES ninfer_linear_swiglu_test_support)

ninfer_add_op_test(ninfer_linear_swiglu_q8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_a16.cpp"
  LIBRARIES ninfer_linear_swiglu_test_support)

# The two-parent form over BF16 and row-split parents (a tensor-parallel rank's gate and up rows).
ninfer_add_op_test(ninfer_linear_swiglu_pair_rowsplit_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_pair_rowsplit.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_swiglu_nvfp4_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4.cpp"
  LIBRARIES ninfer_linear_swiglu_test_support)

ninfer_add_op_test(ninfer_linear_swiglu_fp8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8.cpp"
  LIBRARIES ninfer_linear_swiglu_test_support)

# The A8 route with the producer RMSNorm folded into the quantizer, byte-exact against the
# standalone norm + A8 route and the separate quantizer's planes.
ninfer_add_op_test(ninfer_linear_swiglu_fp8_fused_rmsnorm_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_fused_rmsnorm.cpp"
  LIBRARIES ninfer_ops)
set_tests_properties(ninfer_linear_swiglu_fp8_fused_rmsnorm_test PROPERTIES SKIP_RETURN_CODE 77)

# Integer-activation (s8) routes selected by LinearPolicy::AllowA8Int / AllowA8IntDecode.
ninfer_add_op_test(ninfer_linear_swiglu_q4_a8int_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_a8int.cpp"
  LIBRARIES ninfer_linear_swiglu_test_support)

ninfer_add_op_test(ninfer_linear_swiglu_t2_a8int_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_t2_a8int.cpp"
  LIBRARIES ninfer_linear_swiglu_test_support)

ninfer_add_op_test(ninfer_linear_swiglu_q4a8_int_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4a8_int.cpp"
  LIBRARIES ninfer_ops)

# The cuBLAS prefill route (LinearPolicy::AllowPrefillCublas) across linear_swiglu, linear_add and
# the attention/GDN split projections, against an FP64 oracle.
ninfer_add_op_test(ninfer_w4_cublas_prefill_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_w4_cublas_prefill.cpp"
  LIBRARIES ninfer_ops)
