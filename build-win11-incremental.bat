@echo off
REM Incremental A8 dev build of build-win11 (configure + build only, NO ctest).
REM
REM  - Same configure line as w11buildtest.bat; reconfigure picks up CMake-side changes
REM    (e.g. the fp8 A8 schedule sweep bench target) that a plain ninja pass would miss.
REM  - Skips the ctest section on purpose: run w11buildtest.bat (or ctest by hand) when the
REM    GPU is free; ctest under the production serve's VRAM contention stalls.
REM  - Compilation only: safe while the production serve runs.
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
REM Python3_EXECUTABLE: the t03-workspace uv venv (Python 3.11, jinja2 + pytest) that the
REM Python-driven tests launch. See w11buildtest.bat for the recreate command.
cmake -S . -B build-win11 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON -DNINFER_DIRECTSTORAGE=ON -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=ProgramDatabase -DCMAKE_EXE_LINKER_FLAGS="/DEBUG" -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -Xcompiler=/Zc:__cplusplus -Xcompiler=/Zi" -DPython3_EXECUTABLE="D:\msys64\home\z\src\dsh\t03\.venv\Scripts\python.exe" > configure.log 2>&1
echo CONFIGURE_EXIT=%ERRORLEVEL%
if errorlevel 1 (
  echo CONFIGURE_FAILED - see configure.log
  exit /b 1
)
cmake --build build-win11 --parallel -- -k 0 > build.log 2>&1
echo BUILD_EXIT=%ERRORLEVEL%
if errorlevel 1 (
  echo BUILD_FAILED - see build.log
  exit /b 1
)
echo A8_BUILD_DONE
