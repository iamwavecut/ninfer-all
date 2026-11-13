# Compile policy is owned here; each semantic family explicitly contributes sources.
add_library(ninfer_ops STATIC)
ninfer_internal_includes(ninfer_ops)
ninfer_cuda_archive(ninfer_ops)
target_link_libraries(ninfer_ops PUBLIC ninfer_core)
# The opt-in cuBLAS prefill route hands its GEMM to cuBLAS; see
# ops/linear_swiglu/q4cublas/w4_cublas_prefill.h for why that route exists.
target_link_libraries(ninfer_ops PRIVATE CUDA::cublas)

# Warp-specialized SM120 NVFP4 kernels use setmaxnreg to transfer registers from
# producer to consumer warpgroups. RDC makes ptxas discard that contract. These
# self-contained kernels therefore use a non-RDC archive and host launchers.
# The SM120 tensor-core qualifiers do not exist on sm_86/sm_89, so this fork creates the archive
# only in a build with 120a, and compiles it for 120a alone; only a card that runs the 120a code
# launches its kernels. Family manifests add to it only when the target exists.
if(NINFER_ARCH_BLACKWELL)
  add_library(ninfer_nvfp4_non_rdc STATIC)
  set_target_properties(ninfer_nvfp4_non_rdc PROPERTIES CUDA_ARCHITECTURES 120a-real)
  ninfer_internal_includes(ninfer_nvfp4_non_rdc)
  ninfer_cuda_non_rdc_archive(ninfer_nvfp4_non_rdc)
  target_link_libraries(ninfer_nvfp4_non_rdc PRIVATE
    ninfer_core ${NINFER_CUDART_TARGET} CUDA::cuda_driver)
  target_link_libraries(ninfer_ops PRIVATE ninfer_nvfp4_non_rdc)
endif()

include(basic_sources.cmake)
include("${CMAKE_CURRENT_LIST_DIR}/attn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/candidate_selector/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/context_kv_materialize/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dspine/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_gating_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kv_cache/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_add/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_pair/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_swiglu/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_topk/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rmsnorm_rope/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/softmax_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sparse_moe/sources.cmake")

if(NINFER_COMPAT_PATH)
  # NVFP4 W4A4 uses mma.sync...kind::mxf4nvf4.block_scale; FP8 A8 uses
  # mma.sync...kind::f8f6f4. Both qualifiers are Blackwell-only, and sm_80/86/89 have no FP8 or FP4
  # tensor-core path to fall back to, so a build without 120a leaves the owning translation units
  # out. The A16 dequantizing routes for both formats are kept and remain fully functional. A build
  # with 120a on this path keeps both (NINFER_SM120_FP8, NINFER_SM120_NVFP4).
  if(NOT NINFER_ARCH_BLACKWELL)
    get_target_property(ninfer_ops_sources ninfer_ops SOURCES)
    list(FILTER ninfer_ops_sources EXCLUDE REGEX "nvfp4/.+_(w4a4|a4)\\.cu$")
    list(FILTER ninfer_ops_sources EXCLUDE REGEX "fp8/.+_a8(_unified)?\\.cu$")
    set_property(TARGET ninfer_ops PROPERTY SOURCES ${ninfer_ops_sources})
    target_sources(ninfer_ops PRIVATE
      "${CMAKE_CURRENT_LIST_DIR}/nvfp4_sm86_stubs.cpp"
      "${CMAKE_CURRENT_LIST_DIR}/fp8_sm86_stubs.cpp")
  endif()
  target_compile_definitions(ninfer_ops PUBLIC NINFER_SM8X_COMPAT=1)
endif()
if(NINFER_ARCH_BLACKWELL)
  target_compile_definitions(ninfer_ops PUBLIC NINFER_SM120_NVFP4=1 NINFER_SM120_FP8=1)
endif()

# In a build of several architectures, a route family whose kernels only one tier's cards launch
# compiles for that tier's architecture alone. The FP8 A8 and NVFP4 W4A4/A4 entry points refuse,
# and the fast NVFP4 prompt route declines, every card that does not run the 120a code
# (ops/common/blackwell_routes.h; prompt_nvfp4_fast() in causal_softmax_attention.cpp); the Volta
# projections decline every card but sm_70 (sm70::enabled() in ops/linear/sm70/sm70_gemm.h). Each
# does so before a kernel is launched, so the other images would hold only kernels that no launch
# reaches. A build without 70 compiles the Volta units for its first architecture: they must link
# and never run. None of these units carries PTX: no other card may run their kernels. The units stay in ninfer_ops's archive through an object library with its own
# architectures, and the tier definitions remain those of the whole build (device_tier.h), so every
# gate decides as before.
# CUDA 13 gives every translation unit its own host stub of a template kernel
# (-static-global-template-stub), so each unit launches its own module's copy. CUDA 12 merges the
# stubs of identical instantiations, and the runtime may then look a kernel up in the module of the
# unit with fewer architectures: under CUDA 12 (the Volta family), a kernel template such a unit
# instantiates must not be launched on another tier's card from any other unit.
function(ninfer_ops_route_family name architectures pattern)
  get_target_property(sources ninfer_ops SOURCES)
  set(family ${sources})
  list(FILTER family INCLUDE REGEX "${pattern}")
  if(NOT family)
    message(FATAL_ERROR "No ninfer_ops source belongs to the ${name} route family (${pattern})")
  endif()
  list(FILTER sources EXCLUDE REGEX "${pattern}")
  set_property(TARGET ninfer_ops PROPERTY SOURCES ${sources})
  add_library(ninfer_ops_${name} OBJECT ${family})
  set_target_properties(ninfer_ops_${name} PROPERTIES CUDA_ARCHITECTURES "${architectures}")
  ninfer_internal_includes(ninfer_ops_${name})
  ninfer_cuda_archive(ninfer_ops_${name})
  target_link_libraries(ninfer_ops_${name} PRIVATE ninfer_core)
  target_sources(ninfer_ops PRIVATE $<TARGET_OBJECTS:ninfer_ops_${name}>)
endfunction()

list(LENGTH NINFER_ARCHITECTURES _ninfer_ops_architecture_count)
if(_ninfer_ops_architecture_count GREATER 1)
  if(NINFER_ARCH_BLACKWELL)
    ninfer_ops_route_family(fp8_a8 120a-real "/fp8/[^/]+_a8(_unified)?\\.cu$")
    ninfer_ops_route_family(nvfp4_a4 120a-real
      "/nvfp4/[^/]+_(w4a4|a4)\\.cu$|/causal_cache/nvfp4/prompt_fast\\.cu$")
  endif()
  if(NINFER_ARCH_VOLTA)
    ninfer_ops_route_family(sm70 70-real "/linear/sm70/[^/]+\\.cu$")
  else()
    list(GET NINFER_ARCHITECTURES 0 _ninfer_ops_first_architecture)
    ninfer_ops_route_family(sm70 "${_ninfer_ops_first_architecture}-real"
      "/linear/sm70/[^/]+\\.cu$")
  endif()
endif()

# Windows release builds divide the device code between the shared runtime and this kernel DLL:
# the Windows loader refuses a PE image whose SizeOfImage passes 0x77000000 (1904 MiB). The DLL
# holds the GGML kernels and the INT8-family small-T attention units, with the device-tier probe
# and the installed route profiles that both they and the static Ops use. The GGML, tier and route
# objects compile independently of DLL-only definitions, preserving their cache identity, and are
# exported whole; their bridges import no global data or virtual tables. The small-T archive is
# linked whole and exports only its launch instantiations (launch_small_t_i8_tiled, _pipelined and
# _sm70, one per unit; the router in ninfer_ops calls them): each unit instantiates dozens of
# kernels, and a DLL can name at most 65535 exports. The DLL links its own copy of the device
# layer's launch helpers (core/device_launch.cpp), bound to its own CUDA runtime.
if(NINFER_WINDOWS_KERNEL_DLL)
  add_library(ninfer_kernels SHARED
    $<TARGET_OBJECTS:ninfer_ggml_quants> $<TARGET_OBJECTS:ninfer_device_tier>
    $<TARGET_OBJECTS:ninfer_device_route>)
  set_target_properties(ninfer_kernels PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    LINKER_LANGUAGE CUDA CUDA_RUNTIME_LIBRARY Static
    CUDA_SEPARABLE_COMPILATION OFF CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  ninfer_export_archive_instantiations(ninfer_kernels ninfer_small_t_i8 "launch_small_t_i8_[a-z0-9]+"
    ${NINFER_SMALL_T_I8_INSTANTIATIONS})
  target_link_libraries(ninfer_kernels PRIVATE ninfer_core ${NINFER_CUDART_TARGET} CUDA::cublas)
  target_link_libraries(ninfer_ops PRIVATE ninfer_kernels)
endif()
