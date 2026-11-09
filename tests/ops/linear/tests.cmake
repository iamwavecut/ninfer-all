add_library(ninfer_linear_test_support STATIC
  "${CMAKE_CURRENT_LIST_DIR}/linear_test_common.cpp")
ninfer_test_includes(ninfer_linear_test_support)
ninfer_op_oracle_options(ninfer_linear_test_support)
target_link_libraries(ninfer_linear_test_support PUBLIC ninfer_ops ninfer_artifact)

ninfer_add_op_test(ninfer_linear_q4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q5_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q5_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q6_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q6_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

# A tensor-parallel share's projections: shapes outside the tuned tables, BF16 or FP32 out.
ninfer_add_op_test(ninfer_linear_rowsplit_runtime_shape_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_rowsplit_runtime_shape.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_t2_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_t2_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_t2_a8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_t2_a8.cpp"
  LIBRARIES ninfer_linear_test_support)

# A wide projection first called inside a CUDA graph capture (sm_70: the cuBLAS route's handle).
ninfer_add_op_test(ninfer_linear_capture_first_use_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_capture_first_use.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_nvfp4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_nvfp4_a4_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_a4.cpp"
  LIBRARIES ninfer_linear_test_support)

# The int8 route's own gates: the E4M3 scale table against a software reference, the workspace
# envelopes of both mainloops, the generic-geometry admission, and the grouped-vs-MMQ byte grid
# that puts the two wide mainloops back to back.
ninfer_add_op_test(ninfer_linear_nvfp4_int8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_int8.cpp"
  LIBRARIES ninfer_linear_test_support)

# The [RMSNorm -> int8 quantise] edge kernel against the two launches it replaces: plane bytes,
# narrow-band route bytes, and two negative controls that prove the comparisons have teeth.
ninfer_add_op_test(ninfer_linear_nvfp4_int8_fused_norm_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_int8_fused_norm.cpp"
  LIBRARIES ninfer_linear_test_support)

# Back-to-back TMA staging race probe for both the plain Linear and fused LinearSwiGLU routes:
# distinct weights fired without inter-launch sync, each checked against an isolated reference.
ninfer_add_op_test(ninfer_nvfp4_tma_staging_race_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_tma_staging_race.cpp"
  LIBRARIES ninfer_ops)

# Two ranks' FP8 and NVFP4 shares on two streams of one card, each against the call alone.
ninfer_add_op_test(ninfer_tma_shared_card_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tma_shared_card.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_fp8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_fp8_a8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_a8.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_bf16_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_bf16_a16.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_gguf_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gguf.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_gguf_moe_iq1m_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gguf_moe_iq1m.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_gguf_mmq_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gguf_mmq_store.cu"
  LIBRARIES ninfer_ops)

target_include_directories(ninfer_linear_gguf_mmq_store_test PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/ggml-quants/include"
  "${PROJECT_SOURCE_DIR}/third_party/ggml-quants/src")

ninfer_add_op_test(ninfer_linear_gguf_mmq_limits_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gguf_mmq_limits.cpp")

ninfer_add_op_test(ninfer_linear_gguf_cooperative_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gguf_cooperative.cu"
  LIBRARIES ninfer_ops)

target_include_directories(ninfer_linear_gguf_cooperative_test PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/ggml-quants/include"
  "${PROJECT_SOURCE_DIR}/third_party/ggml-quants/src")

# STQ products (the sushi DSpark drafter's matrices) against an FP64 oracle over an independent
# decode of the trellis.
ninfer_add_op_test(ninfer_linear_stq_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_stq.cpp"
  LIBRARIES ninfer_ops)
