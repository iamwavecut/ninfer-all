@echo off
REM M4 rollback / A16 baseline build: the same tree with NINFER_SM89_FP8=OFF, in its own build
REM directory (build-win11-a16), building only the ninfer-perplexity target.
REM
REM  - The A16 side of the perplexity A/B (the A8 side uses build-win11 as-is).
REM  - Doubles as the proof that the NINFER_SM89_FP8 rollback path configures and builds clean.
REM  - Compilation only: no GPU is needed, so this may run while the production serve is up.
REM  - Does not touch build-win11 (the A8 dev build).
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
cmake -S . -B build-win11-a16 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF -DNINFER_DIRECTSTORAGE=ON -DNINFER_SM89_FP8=OFF -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=ProgramDatabase -DCMAKE_EXE_LINKER_FLAGS="/DEBUG" -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -Xcompiler=/Zc:__cplusplus -Xcompiler=/Zi" > configure-a16.log 2>&1
echo CONFIGURE_EXIT=%ERRORLEVEL%
if errorlevel 1 (
  echo CONFIGURE_FAILED - see configure-a16.log
  exit /b 1
)
cmake --build build-win11-a16 --target ninfer-perplexity --parallel -- -k 0 > build-a16.log 2>&1
echo BUILD_EXIT=%ERRORLEVEL%
if errorlevel 1 (
  echo BUILD_FAILED - see build-a16.log
  exit /b 1
)
echo A16_PERPLEXITY_BUILT
