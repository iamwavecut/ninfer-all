target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/causal_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/fp8/small_t.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/small_t.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/k8v4/small_t.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/int8/small_t_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_reduce.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/small_t_reduce.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/k8v4/small_t_reduce.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/merge.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/int8/prompt_fast_merge.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/fp8/prompt.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/sm70/prompt.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/prompt.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/k8v4/prompt.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/packed_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/context_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/sliding_window_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/launch.cu"
)

# The small-T partial kernels compile in units generated from dense/causal_cache/kernel_unit.cu.in,
# one launch instantiation each; host code (small_t.cpp, fp8/small_t.cpp, int8/small_t_dispatch.cpp,
# ...) routes every launch to one, so an edit of the routing compiles no kernel, an edit of one
# kernel compiles only its units, and the units compile in parallel. BF16, bf16p, FP8, NVFP4 and
# K8V4 have one unit per geometry and input, over every query width. The INT8 family has one per
# tier, query width, geometry and input: the host router picks a tier per launch, and the tiers
# built for a launch's 16-row Q tiles are those of small_t_i8_tier_built() (int8/small_t_tier.h); a
# unit asserts that its tier is one, and the router names each one, so a list below that differs
# fails to compile or to link. Volta's kernels have units of their own, compiled for sm_70 alone in
# a build of several architectures: only a Volta card launches them, so the other images would hold
# only code no launch reaches. CUDA 12 merges the host stubs of identical template instantiations
# across units: those units instantiate no kernel template that any other unit does. The INT8-family
# units carry more than half of the release's device code; Windows kernel-DLL builds link them into
# the kernel DLL (src/ops/CMakeLists.txt) so that no PE image passes what the Windows loader maps.
# There they form an archive of their own, pruned like the others, and the DLL exports only their
# launch instantiations.
set(_ninfer_small_t_i8_rows1
  W2C4K32 W4C2K32 W8C2K32 W16C1K32 W2C2K32Q W4C2K32Q W4C2K32E W8C2K32E W16C1K32E W4C2K32QE)
set(_ninfer_small_t_i8_rows2
  W4C2K32 W8C2K32 W16C1K32 W32C1K32 W4C2K32Q W8C2K32Q W4C2K32E W8C2K32E W16C1K32E W4C2K32QE
  W8C2K32QE)
set(_ninfer_small_t_i8_rows3
  W6C2K32 W12C1K32 W12C1K64D W24C1K32 W6C2K32Q W12C1K32Q W6C2K32E W12C1K32E W6C2K32QE W12C1K32QE)
set(_ninfer_small_t_i8_sources "")
set(_ninfer_small_t_i8_sm70_sources "")
set(_ninfer_causal_cache_sources "")
set(_ninfer_causal_cache_sm70_sources "")
# _ninfer_causal_cache_launches(list name description header function arguments parameters ...)
# generates a unit of one or more launch instantiations, each given by four arguments: the launch
# header under dense/causal_cache, the launcher template, its template arguments and its parameter
# types. The unit instantiates them in the order given. The description completes the unit's first
# comment line and may continue on the next ("\n// ").
function(_ninfer_causal_cache_launches list name description)
  set(NINFER_KERNEL_UNIT_DESCRIPTION "${description}")
  set(NINFER_KERNEL_UNIT_INCLUDES "")
  set(NINFER_KERNEL_UNIT_INSTANTIATIONS "")
  set(launches ${ARGN})
  list(LENGTH launches count)
  math(EXPR last "${count} - 1")
  foreach(index RANGE 0 ${last} 4)
    math(EXPR function_index "${index} + 1")
    math(EXPR arguments_index "${index} + 2")
    math(EXPR parameters_index "${index} + 3")
    list(GET launches ${index} header)
    list(GET launches ${function_index} function)
    list(GET launches ${arguments_index} arguments)
    list(GET launches ${parameters_index} parameters)
    string(APPEND NINFER_KERNEL_UNIT_INCLUDES
      "#include \"ops/softmax_attention/dense/causal_cache/${header}\"\n")
    string(APPEND NINFER_KERNEL_UNIT_INSTANTIATIONS
      "\ntemplate void ${function}<${arguments}>(\n    ${parameters});\n")
  endforeach()
  set(unit "${CMAKE_CURRENT_BINARY_DIR}/causal_cache/${name}.cu")
  configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dense/causal_cache/kernel_unit.cu.in"
    "${unit}" @ONLY)
  set(${list} ${${list}} "${unit}" PARENT_SCOPE)
endfunction()
# A unit of one launch.
function(_ninfer_causal_cache_unit list name header function arguments parameters description)
  _ninfer_causal_cache_launches(${list} "${name}"
    "one causal-cache\n// attention kernel launch, ${description}"
    "${header}" "${function}" "${arguments}" "${parameters}")
  set(${list} ${${list}} PARENT_SCOPE)
endfunction()
if(NINFER_ARCH_TURING OR NINFER_ARCH_AMPERE OR NINFER_ARCH_BLACKWELL)
  set(_ninfer_small_t_post_volta ON)
else()
  set(_ninfer_small_t_post_volta OFF)
endif()
foreach(_ninfer_small_t_geometry IN ITEMS h24 h16)
  if(_ninfer_small_t_geometry STREQUAL h24)
    set(_ninfer_small_t_type CausalD256H24Kv4)
    set(_ninfer_small_t_group 6)
    set(_ninfer_small_t_widest 8)
  else()
    set(_ninfer_small_t_type CausalD256H16Kv2)
    set(_ninfer_small_t_group 8)
    set(_ninfer_small_t_widest 6)
  endif()
  # The prompt kernels of one geometry and block-table form share a unit: the fast INT8-family
  # launch (in a build with post-Volta code), the standard INT8-family launch over the codings with
  # packed int4 values and the BF16 launch, in this order. NVVM compiles a kernel differently after
  # other kernels of its module. Compiled alone, the standard kernels of the packed codings
  # recompute their shared-memory scale addresses in the key loop and run rk4v4 and rk4v4-e8
  # prompts 2-6 % slower on an RTX 4090; compiled after the fast ones, they keep them in registers.
  # In this order their PTX and the other prompt kernels' PTX for sm_70 through sm_120a is the PTX
  # of one unit that holds every geometry and form. The standard kernel over INT8 values is the
  # exception: compiled alone, it runs int8-g64 prompts 1-2 % faster on an RTX 3090 and an RTX
  # 4090, so it has a unit of its own.
  foreach(_ninfer_prompt_form IN ITEMS direct batch masked)
    if(_ninfer_prompt_form STREQUAL direct)
      set(_ninfer_prompt_form_id Direct)
    elseif(_ninfer_prompt_form STREQUAL batch)
      set(_ninfer_prompt_form_id Batch)
    else()
      set(_ninfer_prompt_form_id BatchMasked)
    endif()
    set(_ninfer_prompt_suffix "${_ninfer_small_t_geometry}_${_ninfer_prompt_form}")
    set(_ninfer_prompt_arguments
      "${_ninfer_small_t_type}, PromptMetadataForm::${_ninfer_prompt_form_id}")
    set(_ninfer_prompt_launches "")
    set(_ninfer_prompt_kernels "packed INT8, BF16")
    if(_ninfer_small_t_post_volta)
      list(APPEND _ninfer_prompt_launches int8/prompt_fast_launch.cuh launch_prompt_i8_fast
        "${_ninfer_prompt_arguments}" "const PromptLaunch&")
      set(_ninfer_prompt_kernels "fast INT8, ${_ninfer_prompt_kernels}")
    endif()
    list(APPEND _ninfer_prompt_launches
      int8/prompt_launch.cuh launch_prompt_i8 "${_ninfer_prompt_arguments}, true"
      "const PromptLaunch&"
      bf16/prompt_launch.cuh launch_prompt_bf16 "${_ninfer_prompt_arguments}" "const PromptLaunch&")
    _ninfer_causal_cache_launches(_ninfer_causal_cache_sources "prompt_${_ninfer_prompt_suffix}"
      "the prompt kernels\n// of ${_ninfer_prompt_arguments} (${_ninfer_prompt_kernels})"
      ${_ninfer_prompt_launches})
    _ninfer_causal_cache_unit(_ninfer_causal_cache_sources "prompt_i8_int8_${_ninfer_prompt_suffix}"
      int8/prompt_launch.cuh launch_prompt_i8 "${_ninfer_prompt_arguments}, false"
      "const PromptLaunch&" "the standard INT8-family prompt kernel over INT8 values")
  endforeach()
  foreach(_ninfer_small_t_input IN ITEMS append cached packed)
    if(_ninfer_small_t_input STREQUAL append)
      set(_ninfer_small_t_input_type CausalAppendInput)
    elseif(_ninfer_small_t_input STREQUAL cached)
      set(_ninfer_small_t_input_type CausalCachedInput)
    else()
      set(_ninfer_small_t_input_type CausalPackedInput)
    endif()
    set(_ninfer_small_t_suffix "${_ninfer_small_t_geometry}_${_ninfer_small_t_input}")
    set(_ninfer_small_t_parameters "const SmallTPartialLaunch&, ${_ninfer_small_t_input_type}")
    set(_ninfer_small_t_shape "${_ninfer_small_t_type}, ${_ninfer_small_t_input_type}")
    set(_ninfer_small_t_about "${_ninfer_small_t_type}, ${_ninfer_small_t_input_type}")
    if(_ninfer_small_t_post_volta)
      _ninfer_causal_cache_unit(_ninfer_causal_cache_sources
        "small_t_bf16_${_ninfer_small_t_suffix}" bf16/small_t_launch.cuh launch_small_t_bf16
        "${_ninfer_small_t_shape}" "${_ninfer_small_t_parameters}"
        "the BF16 kernel, ${_ninfer_small_t_about}")
    endif()
    if(_ninfer_small_t_input STREQUAL packed)
      continue()
    endif()
    if(NINFER_ARCH_VOLTA)
      _ninfer_causal_cache_unit(_ninfer_causal_cache_sm70_sources
        "small_t_bf16_sm70_${_ninfer_small_t_suffix}" bf16/small_t_sm70_launch.cuh
        launch_small_t_bf16_sm70 "${_ninfer_small_t_shape}" "${_ninfer_small_t_parameters}"
        "Volta's BF16 kernel, ${_ninfer_small_t_about}")
    endif()
    foreach(_ninfer_small_t_format IN ITEMS fp8 nvfp4 k8v4)
      if(_ninfer_small_t_format STREQUAL fp8)
        set(_ninfer_small_t_codings "true, true")
      elseif(_ninfer_small_t_format STREQUAL nvfp4)
        set(_ninfer_small_t_codings "false, false")
      else()
        set(_ninfer_small_t_codings "true, false")
      endif()
      if(_ninfer_small_t_post_volta)
        _ninfer_causal_cache_unit(_ninfer_causal_cache_sources
          "small_t_${_ninfer_small_t_format}_${_ninfer_small_t_suffix}"
          "${_ninfer_small_t_format}/small_t_launch.cuh" "launch_small_t_${_ninfer_small_t_format}"
          "${_ninfer_small_t_shape}" "${_ninfer_small_t_parameters}"
          "the ${_ninfer_small_t_format} kernel, ${_ninfer_small_t_about}")
      endif()
      if(NINFER_ARCH_VOLTA)
        _ninfer_causal_cache_unit(_ninfer_causal_cache_sm70_sources
          "small_t_${_ninfer_small_t_format}_sm70_${_ninfer_small_t_suffix}"
          sm70/small_t_quant_launch.cuh launch_small_t_quantized_sm70
          "${_ninfer_small_t_shape}, ${_ninfer_small_t_codings}" "${_ninfer_small_t_parameters}"
          "Volta's ${_ninfer_small_t_format} kernel, ${_ninfer_small_t_about}")
      endif()
    endforeach()
  endforeach()
  foreach(_ninfer_small_t_width RANGE 1 ${_ninfer_small_t_widest})
    math(EXPR _ninfer_small_t_rows
      "(${_ninfer_small_t_width} * ${_ninfer_small_t_group} + 15) / 16")
    foreach(_ninfer_small_t_input IN ITEMS append cached)
      if(_ninfer_small_t_input STREQUAL append)
        set(_ninfer_small_t_input_type CausalAppendInput)
      else()
        set(_ninfer_small_t_input_type CausalCachedInput)
      endif()
      set(_ninfer_small_t_prefix
        "small_t_i8_${_ninfer_small_t_geometry}_w${_ninfer_small_t_width}_${_ninfer_small_t_input}")
      set(_ninfer_small_t_i8_parameters "const SmallTI8Launch&, ${_ninfer_small_t_input_type}")
      set(_ninfer_small_t_shape
        "${_ninfer_small_t_type}, ${_ninfer_small_t_width}, ${_ninfer_small_t_input_type}")
      set(_ninfer_small_t_about
        "width ${_ninfer_small_t_width}, ${_ninfer_small_t_type}, ${_ninfer_small_t_input_type}")
      if(_ninfer_small_t_post_volta)
        foreach(_ninfer_small_t_tier IN LISTS _ninfer_small_t_i8_rows${_ninfer_small_t_rows})
          string(TOLOWER "${_ninfer_small_t_tier}" _ninfer_small_t_tier_name)
          _ninfer_causal_cache_unit(_ninfer_small_t_i8_sources
            "${_ninfer_small_t_prefix}_${_ninfer_small_t_tier_name}" int8/small_t_tiled.cuh
            launch_small_t_i8_tiled
            "${_ninfer_small_t_shape}, SmallTI8Tier::${_ninfer_small_t_tier}"
            "${_ninfer_small_t_i8_parameters}"
            "tier ${_ninfer_small_t_tier}, ${_ninfer_small_t_about}")
        endforeach()
        _ninfer_causal_cache_unit(_ninfer_small_t_i8_sources "${_ninfer_small_t_prefix}_pipelined"
          int8/small_t_pipelined_launch.cuh launch_small_t_i8_pipelined "${_ninfer_small_t_shape}"
          "${_ninfer_small_t_i8_parameters}"
          "the pipelined kernel, ${_ninfer_small_t_about}")
      endif()
      if(NINFER_ARCH_VOLTA)
        _ninfer_causal_cache_unit(_ninfer_small_t_i8_sm70_sources "${_ninfer_small_t_prefix}_sm70"
          int8/small_t_sm70_launch.cuh launch_small_t_i8_sm70 "${_ninfer_small_t_shape}"
          "${_ninfer_small_t_i8_parameters}" "Volta's kernel, ${_ninfer_small_t_about}")
      endif()
    endforeach()
  endforeach()
endforeach()
target_sources(ninfer_ops PRIVATE ${_ninfer_causal_cache_sources})
if(NINFER_WINDOWS_KERNEL_DLL)
  add_library(ninfer_small_t_i8 STATIC ${_ninfer_small_t_i8_sources})
  ninfer_cuda_archive(ninfer_small_t_i8)
  set(_ninfer_small_t_i8_target ninfer_small_t_i8)
else()
  set(_ninfer_small_t_i8_target ninfer_ops)
  target_sources(ninfer_ops PRIVATE ${_ninfer_small_t_i8_sources})
endif()
list(LENGTH NINFER_ARCHITECTURES _ninfer_small_t_architectures)
set(_ninfer_small_t_libraries "")
if(_ninfer_small_t_i8_sm70_sources AND _ninfer_small_t_architectures GREATER 1)
  # Volta's units compile for sm_70 alone (see above); they carry no PTX either.
  add_library(ninfer_small_t_i8_sm70 OBJECT ${_ninfer_small_t_i8_sm70_sources})
  set_target_properties(ninfer_small_t_i8_sm70 PROPERTIES CUDA_ARCHITECTURES 70-real)
  ninfer_cuda_archive(ninfer_small_t_i8_sm70)
  target_sources(${_ninfer_small_t_i8_target} PRIVATE $<TARGET_OBJECTS:ninfer_small_t_i8_sm70>)
  list(APPEND _ninfer_small_t_libraries ninfer_small_t_i8_sm70)
else()
  target_sources(${_ninfer_small_t_i8_target} PRIVATE ${_ninfer_small_t_i8_sm70_sources})
endif()
if(_ninfer_causal_cache_sm70_sources AND _ninfer_small_t_architectures GREATER 1)
  add_library(ninfer_causal_cache_sm70 OBJECT ${_ninfer_causal_cache_sm70_sources})
  set_target_properties(ninfer_causal_cache_sm70 PROPERTIES CUDA_ARCHITECTURES 70-real)
  ninfer_cuda_archive(ninfer_causal_cache_sm70)
  target_sources(ninfer_ops PRIVATE $<TARGET_OBJECTS:ninfer_causal_cache_sm70>)
  list(APPEND _ninfer_small_t_libraries ninfer_causal_cache_sm70)
else()
  target_sources(ninfer_ops PRIVATE ${_ninfer_causal_cache_sm70_sources})
endif()
if(NINFER_WINDOWS_KERNEL_DLL)
  list(APPEND _ninfer_small_t_libraries ninfer_small_t_i8)
endif()
foreach(_ninfer_small_t_target IN LISTS _ninfer_small_t_libraries)
  ninfer_internal_includes(${_ninfer_small_t_target})
  target_link_libraries(${_ninfer_small_t_target} PRIVATE ninfer_core)
  target_compile_definitions(${_ninfer_small_t_target} PRIVATE
    $<TARGET_PROPERTY:ninfer_ops,INTERFACE_COMPILE_DEFINITIONS>)
  if(NINFER_WINDOWS_KERNEL_DLL)
    # The PDB name of ninfer_ops keeps /Fd out of the difference, so the units' compile commands
    # match those they have there and reuse the cached objects.
    set_target_properties(${_ninfer_small_t_target} PROPERTIES COMPILE_PDB_NAME ninfer_ops
      COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ninfer_ops.dir")
  endif()
endforeach()
list(LENGTH _ninfer_small_t_i8_sources _ninfer_small_t_i8_tiered)
list(LENGTH _ninfer_small_t_i8_sm70_sources _ninfer_small_t_i8_volta)
math(EXPR NINFER_SMALL_T_I8_INSTANTIATIONS
  "${_ninfer_small_t_i8_tiered} + ${_ninfer_small_t_i8_volta}")

# The fast NVFP4 prompt kernel runs QK on block-scaled FP4 Tensor Cores (kind::mxf4nvf4), which
# only Blackwell has: a build with 120a carries it, and only a card running the 120a code takes it.
if(NINFER_ARCH_BLACKWELL)
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/prompt_fast.cu")
endif()
