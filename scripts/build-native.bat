@echo off
setlocal
REM Native MSVC + CUDA build against a prebuilt vcpkg triplet tree, without the vcpkg toolchain
REM file: FFmpeg is found through cmake\FindFFMPEG.cmake and CMAKE_PREFIX_PATH in the tree this
REM script passes as NINFER_VCPKG_TREE. That tree is NINFER_VCPKG_TREE when it is set; else the
REM checkout's vcpkg_installed\<triplet> when it holds FFmpeg, which `vcpkg install` writes from
REM vcpkg.json in manifest mode (the only mode of the vcpkg that Visual Studio bundles); else the
REM classic tree NINFER_VCPKG_ROOT\installed\<triplet> of a standalone vcpkg. `configure` fetches
REM the pinned Mbed TLS release archive (cmake\MediaSources.cmake) into NINFER_MEDIA_SOURCES,
REM default third_party\media-sources, and the build compiles it. Every path below can be
REM overridden from the environment.
REM
REM   scripts\build-native.bat configure
REM   scripts\build-native.bat build
REM   scripts\build-native.bat target ninfer-serve [more targets]
if not defined NINFER_VCVARS set "NINFER_VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not defined NINFER_GENERATOR set "NINFER_GENERATOR=Visual Studio 18 2026"
if not defined NINFER_CUDA_PATH set "NINFER_CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
if not defined NINFER_VCPKG_ROOT set "NINFER_VCPKG_ROOT=C:\vcpkg"
if not defined NINFER_VCPKG_TRIPLET set "NINFER_VCPKG_TRIPLET=x64-windows"
if not defined NINFER_CUDA_ARCH set "NINFER_CUDA_ARCH=120a"
if not defined NINFER_BUILD_DIR set "NINFER_BUILD_DIR=build-windows"

call "%NINFER_VCVARS%" >nul
if errorlevel 1 exit /b 1
set "CUDA_PATH=%NINFER_CUDA_PATH%"
cd /d "%~dp0.."
if not defined NINFER_VCPKG_TREE (
  if exist "%CD%\vcpkg_installed\%NINFER_VCPKG_TRIPLET%\include\libavformat\avformat.h" (
    set "NINFER_VCPKG_TREE=%CD%\vcpkg_installed\%NINFER_VCPKG_TRIPLET%"
  ) else (
    set "NINFER_VCPKG_TREE=%NINFER_VCPKG_ROOT%\installed\%NINFER_VCPKG_TRIPLET%"
  )
)

if not defined NINFER_MEDIA_SOURCES set "NINFER_MEDIA_SOURCES=%CD%\third_party\media-sources"

if "%1"=="configure" (
  echo vcpkg tree: "%NINFER_VCPKG_TREE%"
  cmake -DNINFER_MEDIA_SOURCES_DIR="%NINFER_MEDIA_SOURCES%" -DNINFER_MEDIA_SOURCES_ONLY=MBEDTLS ^
    -P cmake\FetchMediaSources.cmake
  if errorlevel 1 exit /b 1
  cmake -S . -B "%NINFER_BUILD_DIR%" -G "%NINFER_GENERATOR%" -A x64 ^
    -T "cuda=%CUDA_PATH%" ^
    -DNINFER_VCPKG_TREE="%NINFER_VCPKG_TREE%" ^
    -DNINFER_MEDIA_SOURCES_DIR="%NINFER_MEDIA_SOURCES%" ^
    -DCMAKE_CUDA_ARCHITECTURES=%NINFER_CUDA_ARCH% ^
    -DBUILD_TESTING=ON
) else if "%1"=="build" (
  cmake --build "%NINFER_BUILD_DIR%" --config Release --parallel
) else if "%1"=="target" (
  if "%2"=="" (
    echo usage: build-native.bat target ^<target^> [^<target^> ...]
    exit /b 2
  )
  for /f "tokens=1,*" %%a in ("%*") do cmake --build "%NINFER_BUILD_DIR%" --config Release --parallel --target %%b
) else (
  echo usage: build-native.bat configure^|build^|target ^<target^>...
  exit /b 2
)
exit /b %errorlevel%
