include_guard(GLOBAL)
include(ExternalProject)
include(${CMAKE_CURRENT_LIST_DIR}/MediaSources.cmake)

# Linux builds compile FFmpeg from its pinned release (cmake/MediaSources.cmake) and link it
# statically, so no binary depends on the system's FFmpeg or its SONAMEs. FFmpeg is configured with
# the demuxers, decoders and parsers of the formats NInfer reads and nothing else (the list below,
# documented in docs/serving.md#media-formats), with dav1d for AV1 and zlib for PNG, and without any
# GPL or nonfree component: its licence stays LGPL 2.1 or later. The shared runtime links with
# --exclude-libs, so these symbols stay inside it. Windows builds take vcpkg's FFmpeg DLLs
# (cmake/FindFFMPEG.cmake).
set(NINFER_MEDIA_SOURCES_DIR "${PROJECT_SOURCE_DIR}/third_party/media-sources" CACHE PATH
    "Directory with the release archives cmake/MediaSources.cmake pins (scripts/fetch-media-sources.sh)")
foreach(_ninfer_source IN LISTS NINFER_MEDIA_SOURCES)
  if(NOT EXISTS "${NINFER_MEDIA_SOURCES_DIR}/${NINFER_${_ninfer_source}_ARCHIVE}")
    message(FATAL_ERROR
      "${NINFER_MEDIA_SOURCES_DIR}/${NINFER_${_ninfer_source}_ARCHIVE} is missing. Run "
      "scripts/fetch-media-sources.sh, or set NINFER_MEDIA_SOURCES_DIR to a directory that holds "
      "the archives cmake/MediaSources.cmake pins.")
  endif()
endforeach()

# Machine-independent configuration, which packages record next to the source archives.
set(NINFER_DAV1D_OPTIONS
  --buildtype=release --default-library=static -Db_ndebug=true --wrap-mode=nodownload
  -Denable_tools=false -Denable_tests=false -Denable_examples=false -Denable_docs=false)
set(NINFER_FFMPEG_OPTIONS
  --enable-static --disable-shared --enable-pic --disable-autodetect --disable-programs
  --disable-doc --disable-debug --disable-network --disable-avdevice --disable-avfilter
  --disable-swresample --disable-everything --enable-zlib --enable-libdav1d
  --enable-demuxer=mov,matroska,gif,image_jpeg_pipe,image_png_pipe,image_webp_pipe,image_bmp_pipe
  --enable-demuxer=image_ppm_pipe,image_pgm_pipe,image_pbm_pipe
  --enable-decoder=h264,hevc,libdav1d,vp8,vp9,mpeg4,mjpeg,png,webp,gif,bmp,ppm,pgm,pbm
  --enable-parser=h264,hevc,av1,vp8,vp9,mpeg4video,mjpeg,png,webp,gif,bmp,pnm)

find_package(PkgConfig REQUIRED)
find_package(ZLIB REQUIRED)
find_program(NINFER_MESON meson REQUIRED)
find_program(NINFER_NINJA NAMES ninja ninja-build REQUIRED)
find_program(NINFER_MAKE NAMES gmake make REQUIRED)
set(_ninfer_meson_nasm "")
set(_ninfer_ffmpeg_nasm "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  # The decoders' x86 SIMD routines are NASM sources.
  find_program(NINFER_NASM nasm REQUIRED)
  set(_ninfer_meson_nasm "nasm = '${NINFER_NASM}'\n")
  set(_ninfer_ffmpeg_nasm "--x86asmexe=${NINFER_NASM}")
endif()

# Both builds compile through the project's C compiler and its launcher, so a compiler cache
# serves them as it serves the rest of the build.
set(_ninfer_media_cc ${CMAKE_C_COMPILER_LAUNCHER} ${CMAKE_C_COMPILER})
list(JOIN _ninfer_media_cc " " _ninfer_ffmpeg_cc)
list(TRANSFORM _ninfer_media_cc PREPEND "'")
list(TRANSFORM _ninfer_media_cc APPEND "'")
list(JOIN _ninfer_media_cc ", " _ninfer_meson_cc)
# The inner builds run beside the outer one; a bound keeps a quota-limited host from running a
# compiler per reported core.
cmake_host_system_information(RESULT _ninfer_media_jobs QUERY NUMBER_OF_LOGICAL_CORES)
if(_ninfer_media_jobs GREATER 16)
  set(_ninfer_media_jobs 16)
endif()

set(_ninfer_media "${CMAKE_BINARY_DIR}/third_party/media")
set(_ninfer_media_prefix "${_ninfer_media}/install")
file(WRITE "${_ninfer_media}/dav1d-native.ini"
  "[binaries]\nc = [${_ninfer_meson_cc}]\n${_ninfer_meson_nasm}")

ExternalProject_Add(ninfer_dav1d_build
  URL "${NINFER_MEDIA_SOURCES_DIR}/${NINFER_DAV1D_ARCHIVE}"
  URL_HASH SHA256=${NINFER_DAV1D_SHA256}
  PREFIX "${_ninfer_media}/dav1d"
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E rm -rf <BINARY_DIR>/meson
  COMMAND ${NINFER_MESON} setup <BINARY_DIR>/meson <SOURCE_DIR>
    --native-file ${_ninfer_media}/dav1d-native.ini --prefix=${_ninfer_media_prefix} --libdir=lib
    ${NINFER_DAV1D_OPTIONS}
  BUILD_COMMAND ${NINFER_NINJA} -C <BINARY_DIR>/meson -j ${_ninfer_media_jobs}
  INSTALL_COMMAND ${NINFER_NINJA} -C <BINARY_DIR>/meson install
  INSTALL_BYPRODUCTS ${_ninfer_media_prefix}/lib/libdav1d.a
  LOG_CONFIGURE ON LOG_BUILD ON LOG_INSTALL ON LOG_OUTPUT_ON_FAILURE ON)

set(_ninfer_ffmpeg_libraries avformat avcodec swscale avutil)
set(_ninfer_ffmpeg_archives ${_ninfer_ffmpeg_libraries})
list(TRANSFORM _ninfer_ffmpeg_archives REPLACE "(.+)" "${_ninfer_media_prefix}/lib/lib\\1.a")
ExternalProject_Add(ninfer_ffmpeg_build
  URL "${NINFER_MEDIA_SOURCES_DIR}/${NINFER_FFMPEG_ARCHIVE}"
  URL_HASH SHA256=${NINFER_FFMPEG_SHA256}
  PREFIX "${_ninfer_media}/ffmpeg"
  DEPENDS ninfer_dav1d_build
  # Only the dav1d built above may satisfy --enable-libdav1d.
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env --unset=PKG_CONFIG_PATH
    PKG_CONFIG_LIBDIR=${_ninfer_media_prefix}/lib/pkgconfig
    <SOURCE_DIR>/configure --prefix=${_ninfer_media_prefix} --cc=${_ninfer_ffmpeg_cc}
    --pkg-config=${PKG_CONFIG_EXECUTABLE} --pkg-config-flags=--static ${_ninfer_ffmpeg_nasm}
    ${NINFER_FFMPEG_OPTIONS}
  BUILD_COMMAND ${NINFER_MAKE} -j${_ninfer_media_jobs}
  INSTALL_COMMAND ${NINFER_MAKE} install
  INSTALL_BYPRODUCTS ${_ninfer_ffmpeg_archives}
  LOG_CONFIGURE ON LOG_BUILD ON LOG_INSTALL ON LOG_OUTPUT_ON_FAILURE ON)

# Imported targets must name an existing include directory when the build is generated.
file(MAKE_DIRECTORY "${_ninfer_media_prefix}/include")
add_library(ninfer_dav1d STATIC IMPORTED GLOBAL)
set_target_properties(ninfer_dav1d PROPERTIES
  IMPORTED_LOCATION "${_ninfer_media_prefix}/lib/libdav1d.a"
  INTERFACE_LINK_LIBRARIES "Threads::Threads;${CMAKE_DL_LIBS}")
foreach(_ninfer_library IN LISTS _ninfer_ffmpeg_libraries)
  add_library(ninfer_ffmpeg_${_ninfer_library} STATIC IMPORTED GLOBAL)
  set_target_properties(ninfer_ffmpeg_${_ninfer_library} PROPERTIES
    IMPORTED_LOCATION "${_ninfer_media_prefix}/lib/lib${_ninfer_library}.a")
endforeach()
set_target_properties(ninfer_ffmpeg_avutil PROPERTIES INTERFACE_LINK_LIBRARIES "Threads::Threads;m")
set_target_properties(ninfer_ffmpeg_swscale PROPERTIES INTERFACE_LINK_LIBRARIES ninfer_ffmpeg_avutil)
set_target_properties(ninfer_ffmpeg_avcodec PROPERTIES
  INTERFACE_LINK_LIBRARIES "ninfer_ffmpeg_avutil;ninfer_dav1d;ZLIB::ZLIB")
set_target_properties(ninfer_ffmpeg_avformat PROPERTIES
  INTERFACE_LINK_LIBRARIES "ninfer_ffmpeg_avcodec;ninfer_ffmpeg_avutil;ZLIB::ZLIB")
add_library(ninfer_ffmpeg INTERFACE)
target_include_directories(ninfer_ffmpeg SYSTEM INTERFACE "${_ninfer_media_prefix}/include")
target_link_libraries(ninfer_ffmpeg INTERFACE ninfer_ffmpeg_avformat ninfer_ffmpeg_swscale)
add_dependencies(ninfer_ffmpeg ninfer_ffmpeg_build)
set(NINFER_FFMPEG_TARGET ninfer_ffmpeg)

# scripts/package-linux-release.py ships each archive with its licence and records this.
function(_ninfer_json_string out value)
  string(REPLACE "\\" "\\\\" value "${value}")
  string(REPLACE "\"" "\\\"" value "${value}")
  set(${out} "\"${value}\"" PARENT_SCOPE)
endfunction()
set(_ninfer_manifest "{}")
foreach(_ninfer_source IN LISTS NINFER_MEDIA_SOURCES)
  string(TOLOWER "${_ninfer_source}" _ninfer_name)
  set(_ninfer_entry "{}")
  foreach(_ninfer_field VERSION SHA256 LICENSE)
    string(TOLOWER "${_ninfer_field}" _ninfer_key)
    _ninfer_json_string(_ninfer_value "${NINFER_${_ninfer_source}_${_ninfer_field}}")
    string(JSON _ninfer_entry SET "${_ninfer_entry}" "${_ninfer_key}" "${_ninfer_value}")
  endforeach()
  _ninfer_json_string(_ninfer_value
    "${NINFER_MEDIA_SOURCES_DIR}/${NINFER_${_ninfer_source}_ARCHIVE}")
  string(JSON _ninfer_entry SET "${_ninfer_entry}" archive "${_ninfer_value}")
  set(_ninfer_options "[]")
  foreach(_ninfer_option IN LISTS NINFER_${_ninfer_source}_OPTIONS)
    _ninfer_json_string(_ninfer_value "${_ninfer_option}")
    string(JSON _ninfer_length LENGTH "${_ninfer_options}")
    string(JSON _ninfer_options SET "${_ninfer_options}" ${_ninfer_length} "${_ninfer_value}")
  endforeach()
  string(JSON _ninfer_entry SET "${_ninfer_entry}" options "${_ninfer_options}")
  string(JSON _ninfer_manifest SET "${_ninfer_manifest}" "${_ninfer_name}" "${_ninfer_entry}")
endforeach()
file(WRITE "${_ninfer_media}/sources.json" "${_ninfer_manifest}\n")
