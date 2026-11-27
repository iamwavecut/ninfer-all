# The DFlash drafter's configuration and its own kernels against FP64 references.
ninfer_add_test(ninfer_dflash_drafter_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_drafter.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_dflash_drafter_test PROPERTIES SKIP_RETURN_CODE 77)

# The DFlash and DSpark drafters in the masked-drafter runtime against their FP64 oracle:
# runtime_check.py writes random weights and a schedule and runs this driver from the ninfer_tests
# bundle.
ninfer_bundle_program(ninfer_tests ninfer_dflash_runtime_driver
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/runtime_driver.cpp"
  LIBRARIES ninfer_model_runtime)
ninfer_test_includes(ninfer_dflash_runtime_driver)

add_test(NAME ninfer_dflash_runtime_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/runtime_check.py"
    $<TARGET_FILE:ninfer_tests>)
set_tests_properties(ninfer_dflash_runtime_test PROPERTIES SKIP_RETURN_CODE 77)
