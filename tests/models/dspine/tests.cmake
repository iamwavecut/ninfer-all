# The DSpine runtime against its FP64 oracle: runtime_check.py converts a small random checkpoint
# and runs this driver from the ninfer_tests bundle over a schedule of runtime calls.
ninfer_bundle_program(ninfer_tests ninfer_dspine_runtime_driver
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/runtime_driver.cpp"
  LIBRARIES ninfer_model_runtime)
ninfer_test_includes(ninfer_dspine_runtime_driver)

add_test(NAME ninfer_dspine_runtime_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/runtime_check.py"
    $<TARGET_FILE:ninfer_tests>)
set_tests_properties(ninfer_dspine_runtime_test PROPERTIES SKIP_RETURN_CODE 77)
