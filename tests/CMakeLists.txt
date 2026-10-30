find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(cmake/NinferTests.cmake)
include(cmake/CoreTests.cmake)
include(artifact/tests.cmake)
include(cmake/RuntimeTests.cmake)
include(cmake/ProductTests.cmake)
include(cmake/BenchSupportTests.cmake)
include(models/qwen3_5/tests.cmake)
include(models/qwen4_exp/tests.cmake)
include(models/gemma4/tests.cmake)
include(models/flint/tests.cmake)
include(models/deepseek_v41/tests.cmake)
include(ops/tests.cmake)

ninfer_finalize_bundle(ninfer_tests)

# Exit code 77 is a skip for every test (tests/cuda_availability.h). Without a usable CUDA device --
# no GPU, no driver, or only the toolkit's stub driver, as on a CI runner -- a test that has no check
# of its own fails on its first CUDA call with one of these runtime messages; report that as skipped
# too, so such a box still runs and judges every host test. None of the messages can occur where a
# device is usable.
get_property(_ninfer_all_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_ninfer_all_tests} PROPERTIES SKIP_RETURN_CODE 77 SKIP_REGULAR_EXPRESSION
  "CUDA driver is a stub library;cudaErrorStubLibrary;no CUDA-capable device is detected;CUDA driver version is insufficient for CUDA runtime version")

# Every test executable shares one per-config runtime directory; stage the FFmpeg/curl DLL closure
# there once so they resolve their imports (the Windows loader only searches the executable's own
# directory, the system directories and PATH).
ninfer_stage_test_runtime_dlls(ninfer_tests)

ninfer_add_test(ninfer_llama_frontend_test STANDALONE NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/models/llama/test_frontend.cpp"
  LIBRARIES ninfer_model_runtime)

add_executable(ninfer_llama_real_test models/llama/test_real.cpp)
ninfer_test_includes(ninfer_llama_real_test)
target_link_libraries(ninfer_llama_real_test PRIVATE ninfer_model_runtime)
