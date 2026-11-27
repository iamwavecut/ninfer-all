# GPU-test support for ctest: classification, device pinning and scheduling labels.
#
# A test is GPU-bound when any of its sources is a .cu translation unit, when it links a
# library that reaches the device (the model runtime, the engine, the ops/core layers), or
# when it is marked GPU explicitly. GPU tests get:
#   - the label `gpu`, which scripts/run-tests.sh schedules one device per shard so two GPU
#     tests never share a card,
#   - CUDA_VISIBLE_DEVICES pinned at configure time to NINFER_TEST_CUDA_DEVICE (default 1,
#     the device production does not occupy). A parallel run over two free devices configures
#     two build directories, each pinned to its own card, and runs their GPU shards
#     concurrently; scripts/run-tests.sh does the whole dance.
# CPU tests get no pinning and run freely under ctest -j.
#
# Tests that hold large device allocations (real-model engines, decode-graph captures, deep
# context sweeps) also carry `gpu_serial` and run one at a time: two of them on one card is
# the GPU-contention flake prior lanes reported. The cheap kernel-oracle tests (a few MiB,
# seconds each) overlap freely. The list is explicit and maintained by hand.
#
# An empty ENVIRONMENT_MODIFICATION value breaks ctest's launcher ("missing = after the
# variable name"), so the property is only set when there is a modification to apply.

# The device test runs pin to (the 3090 in the production layout; reconfigure for another).
set(NINFER_TEST_CUDA_DEVICE "1" CACHE STRING "CUDA device the GPU tests pin to")

set(NINFER_GPU_SERIAL_TESTS
    ninfer_paged_8lane_real_test
    ninfer_engine_logprobs_real_test
    ninfer_engine_stop_string_real_test
    ninfer_context_cost_test
    ninfer_decode_graph_test
    ninfer_expert_delivery_test
    ninfer_expert_disk_test
    CACHE INTERNAL "")

function(ninfer_test_is_gpu name)
    cmake_parse_arguments(PARSE_ARGV 1 arg "GPU;SERIAL" "" "SOURCES;LIBRARIES")
    set(is_gpu ${arg_GPU})
    if(NOT is_gpu)
        foreach(source IN LISTS arg_SOURCES)
            if(source MATCHES "\\.cu$")
                set(is_gpu TRUE)
                break()
            endif()
        endforeach()
    endif()
    if(NOT is_gpu)
        foreach(library IN LISTS arg_LIBRARIES)
            if(library MATCHES "ninfer_(model_runtime|engine|runtime_support|ops|core)")
                set(is_gpu TRUE)
                break()
            endif()
        endforeach()
    endif()
    set(is_serial FALSE)
    if(is_gpu AND (arg_SERIAL OR name IN_LIST NINFER_GPU_SERIAL_TESTS))
        set(is_serial TRUE)
    endif()
    set(NINFER_LAST_TEST_IS_GPU ${is_gpu} PARENT_SCOPE)
    set(NINFER_LAST_TEST_IS_SERIAL ${is_serial} PARENT_SCOPE)
endfunction()

function(ninfer_apply_test_polarity name)
    if(NOT NINFER_LAST_TEST_IS_GPU)
        return()
    endif()
    set(labels gpu)
    if(NINFER_LAST_TEST_IS_SERIAL)
        list(APPEND labels gpu_serial)
    endif()
    if(NINFER_LAST_TEST_IS_SERIAL)
        set_tests_properties(${name} PROPERTIES
            LABELS "${labels}"
            ENVIRONMENT "CUDA_VISIBLE_DEVICES=${NINFER_TEST_CUDA_DEVICE}")
    else()
        set_tests_properties(${name} PROPERTIES
            LABELS "${labels}"
            ENVIRONMENT "CUDA_VISIBLE_DEVICES=${NINFER_TEST_CUDA_DEVICE}")
    endif()
endfunction()
