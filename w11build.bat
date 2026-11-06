@echo off
REM Win11 build: install vcpkg deps (vcpkg.json manifest) into the prebuilt triplet tree,
REM then configure and build build-win11 (sm_89).
REM   w11build.bat          incremental (default)
REM   w11build.bat clean    wipe build-win11 first (full rebuild; a pull's header changes are
REM                          not recompiled incrementally on this host -- see HEAD_CHANGED note)
REM
REM Toolchain on this host (see configure.log):
REM  - MSVC 14.51 from VS 2026 Community (18\Community). The only VS installed; nvcc accepts it
REM    because -allow-unsupported-compiler is passed in CMAKE_CUDA_FLAGS below.
REM  - CUDA 13.4, the newest of v11.8/v12.6/v12.9/v13.0/v13.3/v13.4; CMake picks it up unpinned.
REM  - Ninja ships with the VS install and is on PATH after vcvars.
REM
REM vcpkg is checked out into THIS repository at %~dp0vcpkg; nothing is ever installed into a
REM shared/public vcpkg root. The manifest baseline pins older ffmpeg/zlib/curl, and installing
REM them into a public root is what made UniGetUI nag about upgrading those versions after every
REM build.
REM vcpkg installs the manifest into %VCPKG_ROOT%\installed\x64-windows (the prebuilt triplet tree
REM that cmake/FindFFMPEG.cmake and cmake/Dependencies.cmake read via VCPKG_ROOT +
REM VCPKG_TARGET_TRIPLET, which also stages the ffmpeg/curl DLLs beside the executables).
REM --x-install-root keeps manifest mode (baseline-pinned versions) but puts the tree in the
REM canonical location instead of <source>/vcpkg_installed. It names the parent of the triplet
REM directory, hence \installed.
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 

set "VCPKG_ROOT=%~dp0vcpkg"
set "VCPKG_TARGET_TRIPLET=x64-windows"

REM The local proxy http://192.168.0.1:7890 truncates vcpkg's downloads: it kept
REM leaving the powershell-core extract dir empty ("expected this path to exist
REM after extracting powershell-core") and once broke a git extract (...partial).
REM Clear the proxy vars so vcpkg fetches directly; a direct fetch is verified to
REM work on this host (a full 106 MB powershell zip came through clean).
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "http_proxy="
set "https_proxy="
set "ALL_PROXY="
set "all_proxy="

if not exist "%VCPKG_ROOT%\vcpkg.exe" (
  REM First run: clone vcpkg into the checkout and bootstrap its tool binary.
  REM --depth 1 keeps it small; the manifest baseline commit is fetched below on demand.
  git clone --depth 1 https://github.com/microsoft/vcpkg "%VCPKG_ROOT%"
  if errorlevel 1 (
    echo VCPKG_CLONE_FAILED
    exit /b 1
  )
  "%VCPKG_ROOT%\bootstrap-vcpkg.bat" > vcpkg-bootstrap.log 2>&1
  if errorlevel 1 (
    echo VCPKG_BOOTSTRAP_FAILED - see vcpkg-bootstrap.log
    exit /b 1
  )
)

REM vcpkg resolves the manifest baseline with `git show <baseline>:versions/baseline.json`,
REM so the shallow clone must also contain vcpkg.json's pinned baseline commit. Fetch
REM exactly that commit when the checkout lacks it (one commit, no history).
set "NINFER_VCPKG_BASELINE="
for /f "delims=" %%a in ('findstr /c:"builtin-baseline" vcpkg.json') do set "NINFER_VCPKG_BASELINE=%%a"
REM Unquoted set: a literal " inside a quoted set statement would terminate it early.
set NINFER_VCPKG_BASELINE=%NINFER_VCPKG_BASELINE:*": "=%
set NINFER_VCPKG_BASELINE=%NINFER_VCPKG_BASELINE:"=%
set NINFER_VCPKG_BASELINE=%NINFER_VCPKG_BASELINE:,=%
git -C "%VCPKG_ROOT%" rev-parse --verify --quiet "%NINFER_VCPKG_BASELINE%^{commit}" >nul 2>&1
if errorlevel 1 (
  git -C "%VCPKG_ROOT%" fetch --depth 1 origin "%NINFER_VCPKG_BASELINE%"
  if errorlevel 1 (
    echo VCPKG_BASELINE_FETCH_FAILED
    exit /b 1
  )
)

"%VCPKG_ROOT%\vcpkg.exe" install --triplet %VCPKG_TARGET_TRIPLET% --x-install-root "%VCPKG_ROOT%\installed" > vcpkg-install.log 2>&1
if errorlevel 1 (
  echo VCPKG_INSTALL_FAILED - see vcpkg-install.log
  exit /b 1
)

REM CMake caches the vcpkg tree a build dir was configured against (CMAKE_PREFIX_PATH,
REM FFMPEG_DIR). If the vcpkg root moved, wipe build-win11 so configure re-resolves the
REM tree; otherwise an incremental build would silently keep the old one.
set "VCPKG_ROOT_FWD=%VCPKG_ROOT:\=/%"
if exist "build-win11\CMakeCache.txt" (
  findstr /C:"%VCPKG_ROOT_FWD%/installed" build-win11\CMakeCache.txt >nul
  if errorlevel 1 (
    echo VCPKG_ROOT_CHANGED - wiping build-win11 to reconfigure against %VCPKG_ROOT%
    rmdir /s /q build-win11
  )
)

REM CMake 4.4 + Ninja on this host emit "unscanned" compile rules (no .d depfiles), so ninja
REM only recompiles when a source file itself changes -- never when only a header does.
REM Default is an incremental build; when a pull's first build fails on an old signature or a
REM stale symbol (observed: flint/model.cpp.obj linked against an old qwen3_5
REM Bindings::piece), rebuild clean:  w11build.bat clean
REM Local header-only edits under the same commit have the same hazard.
if /i "%~1"=="clean" (
  if exist build-win11 (
    echo CLEAN - wiping build-win11
    rmdir /s /q build-win11
  )
)
set "NINFER_HEAD_SHA="
for /f "delims=" %%a in ('git rev-parse HEAD 2^>nul') do set "NINFER_HEAD_SHA=%%a"
set "NINFER_BUILT_SHA="
if exist "build-win11\.git-sha" set /p NINFER_BUILT_SHA < "build-win11\.git-sha"
if not "%NINFER_HEAD_SHA%"=="" (
  if not "%NINFER_HEAD_SHA%"=="%NINFER_BUILT_SHA%" (
    echo HEAD_CHANGED - building incrementally; if it fails on a stale symbol, rerun: w11build.bat clean
  )
)

REM PDB: ProgramDatabase puts /Zi on the C/CXX compile lines; -Xcompiler=/Zi covers the host
REM code nvcc compiles; CMAKE_EXE_LINKER_FLAGS=/DEBUG makes the linker emit ninfer-*.pdb next
REM to the exes (CMake 4.4 does NOT add /DEBUG by itself). Codegen is unchanged.
cmake -S . -B build-win11 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_TESTING=OFF -DNINFER_DIRECTSTORAGE=ON -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=ProgramDatabase -DCMAKE_EXE_LINKER_FLAGS="/DEBUG" -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -Xcompiler=/Zc:__cplusplus -Xcompiler=/Zi" > configure.log 2>&1
echo CONFIGURE_EXIT=%ERRORLEVEL%
if errorlevel 1 (
  echo CONFIGURE_FAILED - see configure.log
  exit /b 1
)
REM Record the commit this tree was configured against (consumed by the HEAD_CHANGED guard).
echo %NINFER_HEAD_SHA%> build-win11\.git-sha
REM -k 0: keep going after a failure so one pass reports every broken target instead of stopping
REM at the first. Ninja otherwise exits after the first error, which costs a full cycle per fix.
cmake --build build-win11 --target ninfer-serve --parallel -- -k 0 > build.log 2>&1
echo BUILD_EXIT=%ERRORLEVEL%
