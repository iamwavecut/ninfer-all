# FindFFMPEG.cmake — bridge for NInfer's `find_package(FFMPEG)` on native Windows.
#
# The ffmpeg vcpkg port installs no CMake package config. Without the
# vcpkg toolchain NInfer consumes a prebuilt vcpkg triplet tree that provides av*
# import libs + headers + DLLs: the tree cmake/Dependencies.cmake resolved into
# _ninfer_vcpkg_tree, from NINFER_VCPKG_TREE (scripts\build-native.bat passes it) or
# the VCPKG_ROOT and VCPKG_TARGET_TRIPLET environment variables. This module
# assembles the standard FFMPEG_* variables from that tree so the top-level
# CMakeLists.txt (which consumes FFMPEG_INCLUDE_DIRS / FFMPEG_LIBRARY_DIRS /
# FFMPEG_LIBRARIES) works unmodified, and creates imported FFMPEG_<lib> targets
# whose runtime DLL dependencies are staged next to the linking executable.
#
# Also sets FFMPEG_VCPKG_TREE (the resolved triplet root) for callers.

include(FindPackageHandleStandardArgs)

if(NOT _ninfer_vcpkg_tree)
  set(_ffmpeg_reason "no vcpkg triplet tree: set NINFER_VCPKG_TREE (scripts\\build-native.bat passes it)")
  find_package_handle_standard_args(FFMPEG
    REQUIRED_VARS FFMPEG_INCLUDE_DIRS FFMPEG_LIBRARY_DIRS FFMPEG_LIBRARIES
    FAIL_MESSAGE "${_ffmpeg_reason}")
  return()
endif()

set(FFMPEG_VCPKG_TREE "${_ninfer_vcpkg_tree}")

set(_ffmpeg_needed avformat avcodec avutil swscale)
set(_ffmpeg_libs_found TRUE)
foreach(_lib IN LISTS _ffmpeg_needed)
  if(NOT EXISTS "${FFMPEG_VCPKG_TREE}/lib/${_lib}.lib")
    set(_ffmpeg_libs_found FALSE)
    set(_ffmpeg_reason "missing ${_lib}.lib in ${FFMPEG_VCPKG_TREE}/lib (the vcpkg tree lacks ffmpeg: `vcpkg install` in the checkout installs vcpkg.json's packages into vcpkg_installed)")
  endif()
endforeach()

if(EXISTS "${FFMPEG_VCPKG_TREE}/include/libavformat/avformat.h")
  set(_ffmpeg_include "${FFMPEG_VCPKG_TREE}/include")
else()
  set(_ffmpeg_include "")
endif()

if(NOT _ffmpeg_include OR NOT _ffmpeg_libs_found)
  if(NOT _ffmpeg_reason)
    set(_ffmpeg_reason "ffmpeg headers not found in the vcpkg tree")
  endif()
  find_package_handle_standard_args(FFMPEG
    REQUIRED_VARS FFMPEG_INCLUDE_DIRS FFMPEG_LIBRARY_DIRS FFMPEG_LIBRARIES
    FAIL_MESSAGE "${_ffmpeg_reason}")
  return()
endif()

set(FFMPEG_INCLUDE_DIRS "${_ffmpeg_include}")
set(FFMPEG_LIBRARY_DIRS "${FFMPEG_VCPKG_TREE}/lib")
set(FFMPEG_LIBRARIES "")
foreach(_lib IN LISTS _ffmpeg_needed)
  add_library(FFMPEG_${_lib} STATIC IMPORTED)
  set_target_properties(FFMPEG_${_lib} PROPERTIES
    IMPORTED_LOCATION "${FFMPEG_VCPKG_TREE}/lib/${_lib}.lib")
  list(APPEND FFMPEG_LIBRARIES FFMPEG_${_lib})
endforeach()

# Stage the shared-library artifacts (import libs reference these DLLs) next to
# any executable that links the FFMPEG targets. CMake de-duplicates the copies.
file(GLOB _ffmpeg_dlls "${FFMPEG_VCPKG_TREE}/bin/*.dll")
foreach(_lib IN LISTS _ffmpeg_needed)
  if(_ffmpeg_dlls)
    set_target_properties(FFMPEG_${_lib} PROPERTIES
      IMPORTED_LINK_DEPENDENT_LIBRARIES "${_ffmpeg_dlls}")
  endif()
endforeach()

add_library(FFMPEG::FFMPEG INTERFACE IMPORTED)
target_include_directories(FFMPEG::FFMPEG INTERFACE "${_ffmpeg_include}")
target_link_libraries(FFMPEG::FFMPEG INTERFACE ${FFMPEG_LIBRARIES})

find_package_handle_standard_args(FFMPEG
  REQUIRED_VARS FFMPEG_INCLUDE_DIRS FFMPEG_LIBRARY_DIRS FFMPEG_LIBRARIES)
