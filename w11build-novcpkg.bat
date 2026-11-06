@echo off
REM w11build.bat minus the vcpkg install step: the vcpkg tree at %VCPKG_ROOT%\installed
REM is already complete (vcpkg.json unchanged in this merge) and re-running vcpkg needs
REM network fetches. Same toolchain/flags as w11build.bat.
REM Run w11build.bat once after a vcpkg.json change so the checkout-local tree exists.
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

set "VCPKG_ROOT=%~dp0vcpkg"
set "VCPKG_TARGET_TRIPLET=x64-windows"

REM Same stale-object note as w11build.bat: CMake 4.4 + Ninja here emit "unscanned" rules
REM (no .d depfiles), so header-only changes never recompile incrementally. If a pull's first
REM build fails on a stale symbol, run: w11build.bat clean
set "NINFER_HEAD_SHA="
for /f "delims=" %%a in ('git rev-parse HEAD 2^>nul') do set "NINFER_HEAD_SHA=%%a"
set "NINFER_BUILT_SHA="
if exist "build-win11\.git-sha" set /p NINFER_BUILT_SHA < "build-win11\.git-sha"
if not "%NINFER_HEAD_SHA%"=="" (
  if not "%NINFER_HEAD_SHA%"=="%NINFER_BUILT_SHA%" (
    echo HEAD_CHANGED - building incrementally; if it fails on a stale symbol, run: w11build.bat clean
  )
)

cmake -S . -B build-win11 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_TESTING=OFF -DNINFER_DIRECTSTORAGE=ON -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=ProgramDatabase -DCMAKE_EXE_LINKER_FLAGS="/DEBUG" -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -Xcompiler=/Zc:__cplusplus -Xcompiler=/Zi" > configure.log 2>&1
echo CONFIGURE_EXIT=%ERRORLEVEL%
if errorlevel 1 (
  echo CONFIGURE_FAILED - see configure.log
  exit /b 1
)
REM Record the commit this tree was configured against (consumed by the HEAD_CHANGED guard).
echo %NINFER_HEAD_SHA%> build-win11\.git-sha
REM -k 0: keep going after a failure so one pass reports every broken target instead of stopping
REM at the first.
cmake --build build-win11 --target ninfer-serve --parallel -- -k 0 > build.log 2>&1
echo BUILD_EXIT=%ERRORLEVEL%
