@echo off
REM A16 test-build for the hybrid/vision real-test A/B (isolate A8-specific rounding from
REM 4090-vs-3090 measurement difference): the same tree with NINFER_SM89_FP8=OFF, in its own
REM build directory (build-win11-a16-tests).
REM
REM  - The qwen3_5 real tests are object libraries bundled into ninfer_tests.exe (dispatched
REM    by name), so the useful artifact is the A16 ninfer_tests bundle, not standalone exes.
REM  - build-win11-a16 stays the perplexity-only A16 build (M4 gate baseline).
REM  - Compilation only: no GPU is needed, so this may run while the production serve is up.
REM  - Run "F:\gsoft\gw\a16tests\ninfer_tests.exe <test-name>" with NINFER_TEST_ARTIFACT set
REM    when the GPU is free.
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "VCPKG_ROOT=%~dp0vcpkg"
set "VCPKG_TARGET_TRIPLET=x64-windows"
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "http_proxy="
set "https_proxy="
set "ALL_PROXY="
set "all_proxy="
cmake -S . -B build-win11-a16-tests -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=OFF -DNINFER_DIRECTSTORAGE=ON -DNINFER_SM89_FP8=OFF -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=ProgramDatabase -DCMAKE_EXE_LINKER_FLAGS="/DEBUG" -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -Xcompiler=/Zc:__cplusplus -Xcompiler=/Zi" -DPython3_EXECUTABLE="D:\msys64\home\z\src\dsh\t03\.venv\Scripts\python.exe" > configure-a16test.log 2>&1
echo CONFIGURE_EXIT=%ERRORLEVEL%
if errorlevel 1 (
  echo CONFIGURE_FAILED - see configure-a16test.log
  exit /b 1
)
cmake --build build-win11-a16-tests --target ninfer_tests --parallel -- -k 0 > build-a16test.log 2>&1
echo BUILD_EXIT=%ERRORLEVEL%
if errorlevel 1 (
  echo BUILD_FAILED - see build-a16test.log
  exit /b 1
)
echo A16TEST_BUILT
